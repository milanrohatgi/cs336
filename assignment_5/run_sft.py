#!/usr/bin/env python3
import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm.model_executor import set_random_seed as vllm_set_seed
from vllm import LLM, SamplingParams
from unittest.mock import patch
from math_baseline import evaluate_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from sft_helpers import *

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_policy_into_vllm(policy_model, llm):
    state_dict = policy_model.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

SFT_JSONL = "/data/a5-alignment/MATH/sft.jsonl"
VALIDATION_JSONL= "/data/a5-alignment/MATH/validation.jsonl"
BASE_MODEL_PATH = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"

POLICY_DEVICE = "cuda:0"    
VLLM_DEVICE = "cuda:1"    

LR = 5e-5
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS = 3
VALIDATE_EVERY_STEPS = 200  
MAX_GRAD_NORM = 1.0
SEED = 42

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
        input_ids     = tokenized["input_ids"].squeeze(0)
        labels        = tokenized["labels"].squeeze(0)
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

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float=0.85) -> LLM:
    vllm_set_seed(seed)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="./sft_outputs")
    args = parser.parse_args()

    all_sft = load_jsonl(SFT_JSONL)
    random.shuffle(all_sft)
    N = args.num_examples
    if N <= 0 or N > len(all_sft):
        N = len(all_sft)
    subset = all_sft[:N]
    print(f">>> Training SFT on {N} / {len(all_sft)} examples.")

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

    train_ds = MathSFTDataset(subset, tokenizer)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    llm = init_vllm(
        model_id=BASE_MODEL_PATH,
        device=VLLM_DEVICE,
        seed=SEED,
        gpu_memory_utilization=0.85,
    )

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=LR)

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
    )
    sampling_params.include_stop_str_in_output = True

    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        for batch_idx, (input_ids, labels, resp_mask) in enumerate(train_dl, start=1):
            input_ids = input_ids.to(POLICY_DEVICE)
            labels = labels.to(POLICY_DEVICE)
            resp_mask = resp_mask.to(POLICY_DEVICE)


            out_dict = get_response_log_probs(policy_model, input_ids, labels)
            log_probs = out_dict["log_probs"]   # <-- extract the tensor

            micro_loss, _ = sft_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=resp_mask,
                gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            )

            if (batch_idx % GRAD_ACCUM_STEPS) == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % VALIDATE_EVERY_STEPS == 0:
                    load_policy_into_vllm(policy_model, llm)

                    eval_path = os.path.join(
                        args.output_dir, f"eval_zero_shot_step_{global_step}.jsonl"
                    )
                    print(f"↪ Running evaluate_vllm at step {global_step} → {eval_path}")
                    evaluate_vllm(
                        vllm_model=llm,
                        reward_fn=r1_zero_reward_fn,
                        examples=val_examples,
                        sampling_params=sampling_params,
                        output_jsonl_path=eval_path,
                        batch_size=BATCH_SIZE,
                    )

        print(f"=== Finished epoch {epoch}/{NUM_EPOCHS}. Doing end‐of‐epoch eval…")
        load_policy_into_vllm(policy_model, llm)
        final_eval_path = os.path.join(
            args.output_dir, f"eval_zero_shot_epoch_{epoch}.jsonl"
        )
        evaluate_vllm(
            vllm_model=llm,
            reward_fn=r1_zero_reward_fn,
            examples=val_examples,
            sampling_params=sampling_params,
            output_jsonl_path=final_eval_path,
            batch_size=256,
        )

    print(">>> SFT training complete.")

if __name__ == "__main__":
    import argparse
    main()
