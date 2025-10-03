#!/usr/bin/env python3
import os
import json
import random
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm.model_executor import set_random_seed as vllm_set_seed
from vllm import LLM, SamplingParams
from unittest.mock import patch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from sft_helpers import get_response_log_probs, sft_microbatch_train_step, tokenize_prompt_and_output

NUM_EI_STEPS = 5     
ROLLOUT_COUNT = 16 
EI_BATCH_SIZE = 1024
SFT_EPOCHS_PER_EI = 5

POLICY_DEVICE = "cuda:0"
VLLM_DEVICE = "cuda:1"
LR = 2e-5
SFT_BATCH_SIZE = 4
SFT_GRAD_ACCUM_STEPS = 8
SFT_MAX_GRAD_NORM = 1.0
SFT_VALIDATE_EVERY_STEPS = 200   
SFT_SEED = 42
VALIDATION_SUBSET_SIZE = 256

TRAIN_JSONL = "/data/a5-alignment/MATH/train.jsonl"
VALIDATION_JSONL = "/data/a5-alignment/MATH/validation.jsonl"
BASE_MODEL_PATH = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"

OUTPUT_DIR = "./ei_wandb_outputs"

with open("./prompts/r1_zero.prompt", "r", encoding="utf-8") as f:
    PROMPT_TEMPLATE = f.read()

def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

class MathSFTDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokenized = tokenize_prompt_and_output(
            [ex["prompt"]],
            [ex["response"]],
            self.tokenizer
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        labels = tokenized["labels"].squeeze(0)
        response_mask = tokenized["response_mask"].squeeze(0)
        return input_ids, labels, response_mask

def collate_fn(batch):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    maxlen = max(item[0].size(0) for item in batch)
    B = len(batch)

    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    labels = torch.full((B, maxlen), pad_id, dtype=torch.long)
    resp_mask = torch.zeros((B, maxlen), dtype=torch.bool)

    for i, (inp, lab, rm) in enumerate(batch):
        L = inp.size(0)
        input_ids[i, :L] = inp
        labels[i, :L] = lab
        resp_mask[i, :L] = rm

    return input_ids, labels, resp_mask

def load_policy_into_vllm(policy_model, llm):
    state_dict = policy_model.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float=0.85):
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch  = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        llm = LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    return llm

def extract_entropy_from_logprobs(gen_obj):
    try:
        if hasattr(gen_obj, 'logprobs') and gen_obj.logprobs is not None:
            all_logprobs = []
            for token_logprobs in gen_obj.logprobs:
                if token_logprobs is not None and len(token_logprobs) > 0:
                    for token_id, logprob_obj in token_logprobs.items():
                        all_logprobs.append(logprob_obj.logprob)
                        break  
            
            if all_logprobs:
                lp_tensor = torch.tensor(all_logprobs, dtype=torch.float32)
                entropy = -lp_tensor.mean().item()
                return entropy
                
    except Exception as e:
        print(f"Warning: Could not extract entropy: {e}")
    
    return None

def run_validation(llm, val_examples, sampling_params, output_path, subset_size=None):
    examples_to_eval = val_examples
    if subset_size and subset_size < len(val_examples):
        examples_to_eval = random.sample(val_examples, subset_size)
        print(f"      Using {len(examples_to_eval)}/{len(val_examples)} validation examples")
    
    from math_baseline import evaluate_vllm as eval_fn
    eval_metrics = eval_fn(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        examples=examples_to_eval,
        sampling_params=sampling_params,
        output_jsonl_path=output_path,
        batch_size=1024 
    )
    return eval_metrics

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wandb.init(
        project="math_ei_experiment",
        name=f"EI_G{ROLLOUT_COUNT}_B{EI_BATCH_SIZE}_E{SFT_EPOCHS_PER_EI}_lr{LR}",
        config={
            "NUM_EI_STEPS": NUM_EI_STEPS,
            "ROLLOUT_COUNT": ROLLOUT_COUNT,
            "EI_BATCH_SIZE": EI_BATCH_SIZE,
            "SFT_EPOCHS_PER_EI": SFT_EPOCHS_PER_EI,
            "POLICY_DEVICE": POLICY_DEVICE,
            "VLLM_DEVICE": VLLM_DEVICE,
            "LR": LR,
            "SFT_BATCH_SIZE": SFT_BATCH_SIZE,
            "SFT_GRAD_ACCUM_STEPS": SFT_GRAD_ACCUM_STEPS,
            "SFT_MAX_GRAD_NORM": SFT_MAX_GRAD_NORM,
            "SFT_VALIDATE_EVERY_STEPS": SFT_VALIDATE_EVERY_STEPS,
        }
    )

    all_train = load_jsonl(TRAIN_JSONL)
    val_examples = load_jsonl(VALIDATION_JSONL)

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    policy_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(POLICY_DEVICE)
    policy_model.train()

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=LR)

    llm = init_vllm(
        model_id=BASE_MODEL_PATH,
        device=VLLM_DEVICE,
        seed=SFT_SEED,
        gpu_memory_utilization=0.75,  
    )

    sampling_params = SamplingParams(
        temperature=0.8, 
        top_p=0.95, 
        max_tokens=1024,
        min_tokens=1,
        n=ROLLOUT_COUNT,      
        stop=["</answer>"],
        logprobs=1
    )
    sampling_params.include_stop_str_in_output = True

    global_step = 0


    for ei_step in range(1, NUM_EI_STEPS + 1):
        print(f"\n EI Step {ei_step}/{NUM_EI_STEPS}")

        ei_batch = random.sample(all_train, EI_BATCH_SIZE)

        prompts = []
        for ex in ei_batch:
            question_text = ex["problem"]
            prompt_str = PROMPT_TEMPLATE.format(question=question_text)
            prompts.append(prompt_str)

        policy_model.eval()
        load_policy_into_vllm(policy_model, llm)

        results = llm.generate(prompts, sampling_params=sampling_params)

        D_sft_ei = []
        total_kept = 0
        total_generated = 0
        all_rollout_entropies = []

        for idx, ex in enumerate(ei_batch):
            true_answer = ex["answer"]
            req_out = results[idx]

            if hasattr(req_out, "outputs"):
                gens_for_this = req_out.outputs  
            else:
                gens_for_this = [req_out] 

            for gen_obj in gens_for_this:
                total_generated += 1
                candidate = gen_obj.text

                r = r1_zero_reward_fn(candidate, true_answer)
                if r["reward"] > 0:
                    D_sft_ei.append({"prompt": prompts[idx], "response": candidate})
                    total_kept += 1

                entropy = extract_entropy_from_logprobs(gen_obj)
                if entropy is not None:
                    all_rollout_entropies.append(entropy)

        success_rate = total_kept / total_generated if total_generated > 0 else 0.0
        print(f"    Kept {total_kept}/{total_generated} correct rollouts (success rate: {success_rate:.2%})")

        if len(all_rollout_entropies) > 0:
            avg_ent = float(sum(all_rollout_entropies) / len(all_rollout_entropies))
        else:
            avg_ent = None

        wandb.log({
            "ei_step": ei_step,
            "rolled_out_examples": total_generated,
            "kept_correct_rollouts": total_kept,
            "rollout_success_rate": success_rate,
            "avg_rollout_entropy": avg_ent,
            "step": global_step,
        })

        if len(D_sft_ei) == 0:
            policy_model.train()
            continue

        train_ds_ei = MathSFTDataset(D_sft_ei, tokenizer)
        train_dl_ei = DataLoader(
            train_ds_ei,
            batch_size=SFT_BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=False,
        )

        policy_model.train()
        for epoch in range(1, SFT_EPOCHS_PER_EI + 1):
            print(f"[EI‐SFT] Epoch {epoch}/{SFT_EPOCHS_PER_EI} on {len(D_sft_ei)} examples...")
            
            for batch_idx, (input_ids, labels, resp_mask) in enumerate(train_dl_ei, start=1):
                input_ids = input_ids.to(POLICY_DEVICE)
                labels    = labels.to(POLICY_DEVICE)
                resp_mask = resp_mask.to(POLICY_DEVICE)

                out_dict = get_response_log_probs(policy_model, input_ids, labels)
                log_probs = out_dict["log_probs"]

                micro_loss, _ = sft_microbatch_train_step(
                    policy_log_probs=log_probs,
                    response_mask=resp_mask,
                    gradient_accumulation_steps=SFT_GRAD_ACCUM_STEPS,
                )

                if (batch_idx % SFT_GRAD_ACCUM_STEPS) == 0:
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), SFT_MAX_GRAD_NORM)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    wandb.log({
                        "train_loss": micro_loss,
                        "ei_step": ei_step,
                        "step": global_step,
                    })

                    if global_step % SFT_VALIDATE_EVERY_STEPS == 0:
                        print(f"      → Running validation at global_step {global_step}")
                        policy_model.eval()
                        load_policy_into_vllm(policy_model, llm)

                        eval_path = os.path.join(
                            OUTPUT_DIR, f"ei{ei_step}_eval_step_{global_step}.jsonl"
                        )
                        
                        eval_metrics = run_validation(
                            llm, val_examples, sampling_params, eval_path, 
                            subset_size=VALIDATION_SUBSET_SIZE
                        )
                        val_acc = eval_metrics.get("accuracy", None)

                        wandb.log({
                            "validation_accuracy": val_acc,
                            "ei_step": ei_step,
                            "step": global_step,
                        })

                        policy_model.train()

        policy_model.eval()
        load_policy_into_vllm(policy_model, llm)

        step_eval_path = os.path.join(OUTPUT_DIR, f"ei_step_{ei_step}_full_eval.jsonl")
        step_metrics = run_validation(llm, val_examples, sampling_params, step_eval_path, subset_size=VALIDATION_SUBSET_SIZE)
        step_acc = step_metrics.get("accuracy", None)
        
        print(f"EI Step {ei_step} Validation Accuracy: {step_acc:.4f}")
        
        wandb.log({
            "validation_accuracy": step_acc,
            "ei_step": ei_step,
            "step": global_step,
        })

        ckpt_path = os.path.join(OUTPUT_DIR, f"policy_after_ei_step_{ei_step}.pt")
        torch.save(policy_model.state_dict(), ckpt_path)

        policy_model.train()

    policy_model.eval()
    load_policy_into_vllm(policy_model, llm)

    final_eval_path = os.path.join(OUTPUT_DIR, "final_zero_shot_eval.jsonl")
    final_metrics = run_validation(llm, val_examples, sampling_params, final_eval_path)
    final_acc = final_metrics.get("accuracy", None)
    print(f"Final Validation Accuracy: {final_acc:.4f}")

    wandb.log({
        "final_validation_accuracy": final_acc,
        "step": global_step
    })

    wandb.finish()

if __name__ == "__main__":
    main()