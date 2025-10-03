import os
import json
import random
import argparse
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm.model_executor import set_random_seed as vllm_set_seed
from vllm import LLM, SamplingParams
from unittest.mock import patch
from typing import Literal, List, Dict, Any
import numpy as np

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn 
from sft_helpers import *
from grpo_parts import *

N_GRPO_STEPS = 200
LEARNING_RATE = 2e-5
ADVANTAGE_EPS = 1e-6
ROLLOUT_BATCH_SIZE = 256
GROUP_SIZE = 8
SAMPLING_TEMPERATURE = 1.0
SAMPLING_MIN_TOKENS = 4
SAMPLING_MAX_TOKENS = 1024
EPOCHS_PER_ROLLOUT_BATCH = 1
TRAIN_BATCH_SIZE = 256
GRADIENT_ACCUMULATION_STEPS = 128
GPU_MEMORY_UTILIZATION = 0.2
LOSS_TYPE = "reinforce_with_baseline"
USE_STD_NORMALIZATION = False
CLIPRANGE = 0.2
MAX_GRAD_NORM = 1.0

TRAIN_JSONL = "/data/a5-alignment/MATH/train.jsonl"
VALIDATION_JSONL = "/data/a5-alignment/MATH/validation.jsonl"
BASE_MODEL_PATH = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"

POLICY_DEVICE = "cuda:0"
VLLM_DEVICE = "cuda:0"
SEED = 42

VALIDATE_EVERY_STEPS = 25
VALIDATION_SIZE = 512

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_policy_into_vllm(policy_model, llm):
    """Load policy model weights into vLLM."""
    # Handle torch.compile models by accessing the original model
    if hasattr(policy_model, '_orig_mod'):
        # Model is compiled, get original state_dict
        state_dict = policy_model._orig_mod.state_dict()
    else:
        # Model is not compiled
        state_dict = policy_model.state_dict()
    
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
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

class RolloutDataset(Dataset):
    def __init__(self, rollouts, tokenizer):
        self.rollouts = rollouts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        rollout = self.rollouts[idx]
        tokenized = tokenize_prompt_and_output(
            [rollout["prompt"]],
            [rollout["response"]],
            self.tokenizer
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        labels = tokenized["labels"].squeeze(0)
        response_mask = tokenized["response_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "response_mask": response_mask,
            "reward": rollout["reward"],
            "advantage": rollout["advantage"],
        }

def collate_rollouts(batch):
    pad_id = 0
    maxlen = max(item["input_ids"].size(0) for item in batch)
    B = len(batch)

    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    labels = torch.full((B, maxlen), pad_id, dtype=torch.long)
    response_mask = torch.zeros((B, maxlen), dtype=torch.bool)
    rewards = torch.zeros((B, 1), dtype=torch.float32)
    advantages = torch.zeros((B, 1), dtype=torch.float32)

    for i, item in enumerate(batch):
        L = item["input_ids"].size(0)
        input_ids[i, :L] = item["input_ids"]
        labels[i, :L] = item["labels"]
        response_mask[i, :L] = item["response_mask"]
        rewards[i, 0] = item["reward"]
        advantages[i, 0] = item["advantage"]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
        "rewards": rewards,
        "advantages": advantages,
    }

def run_validation(policy_model, llm, val_examples, tokenizer, sampling_params, use_full_set=False):
    policy_model.eval()
    load_policy_into_vllm(policy_model, llm)

    if use_full_set:
        val_sample = val_examples
        print(f"    Using full validation set: {len(val_sample)} examples")
    else:
        val_sample = random.sample(val_examples, min(VALIDATION_SIZE, len(val_examples)))
        print(f"    Using validation sample: {len(val_sample)}/{len(val_examples)} examples")
 
    with open("./prompts/r1_zero.prompt", "r", encoding="utf-8") as f:
        prompt_template = f.read()
    
    prompts = []
    true_answers = []
    for ex in val_sample:
        prompt = prompt_template.format(question=ex["problem"])
        prompts.append(prompt)
        true_answers.append(ex["answer"])

    results = llm.generate(prompts, sampling_params=sampling_params)
  
    correct = 0
    total_reward = 0.0
    format_reward = 0.0
    answer_reward = 0.0
    
    for i, result in enumerate(results):
        response = result.outputs[0].text
        reward_dict = r1_zero_reward_fn(response, true_answers[i])
        
        if reward_dict["reward"] > 0:
            correct += 1
        
        total_reward += reward_dict["reward"]
        format_reward += reward_dict["format_reward"]
        answer_reward += reward_dict["answer_reward"]
    
    accuracy = correct / len(val_sample)
    avg_total_reward = total_reward / len(val_sample)
    avg_format_reward = format_reward / len(val_sample)
    avg_answer_reward = answer_reward / len(val_sample)
    
    policy_model.train()
    
    return {
        "accuracy": accuracy,
        "avg_total_reward": avg_total_reward,
        "avg_format_reward": avg_format_reward,
        "avg_answer_reward": avg_answer_reward,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./grpo_outputs")
    parser.add_argument("--wandb_project", type=str, default="grpo_math_training")
    parser.add_argument("--wandb_name", type=str, default=None)
    args = parser.parse_args()

    assert TRAIN_BATCH_SIZE % GRADIENT_ACCUMULATION_STEPS == 0
    assert ROLLOUT_BATCH_SIZE % GROUP_SIZE == 0
    assert TRAIN_BATCH_SIZE >= GROUP_SIZE

    micro_train_batch_size = TRAIN_BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS
    n_prompts_per_rollout_batch = ROLLOUT_BATCH_SIZE // GROUP_SIZE
    n_microbatches_per_rollout_batch = ROLLOUT_BATCH_SIZE // micro_train_batch_size

    os.makedirs(args.output_dir, exist_ok=True)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            "n_grpo_steps": N_GRPO_STEPS,
            "learning_rate": LEARNING_RATE,
            "rollout_batch_size": ROLLOUT_BATCH_SIZE,
            "group_size": GROUP_SIZE,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "loss_type": LOSS_TYPE,
            "use_std_normalization": USE_STD_NORMALIZATION,
        }
    )

    train_examples = load_jsonl(TRAIN_JSONL)
    val_examples = load_jsonl(VALIDATION_JSONL)

    with open("./prompts/r1_zero.prompt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    global pad_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    policy_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(POLICY_DEVICE)
    policy_model.train()
    policy_model = torch.compile(policy_model)

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    llm = init_vllm(
        model_id=BASE_MODEL_PATH,
        device=VLLM_DEVICE,
        seed=SEED,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    sampling_params = SamplingParams(
        temperature=SAMPLING_TEMPERATURE,
        top_p=1.0,
        max_tokens=SAMPLING_MAX_TOKENS,
        min_tokens=SAMPLING_MIN_TOKENS,
        n=GROUP_SIZE,
        stop=["</answer>"],
    )
    sampling_params.include_stop_str_in_output = True

    val_sampling_params = SamplingParams(
        temperature=SAMPLING_TEMPERATURE,
        top_p=1.0,
        max_tokens=SAMPLING_MAX_TOKENS,
        min_tokens=SAMPLING_MIN_TOKENS,
        n=1,
        stop=["</answer>"],
    )
    val_sampling_params.include_stop_str_in_output = True

    def collate_rollouts_with_pad(batch):
        maxlen = max(item["input_ids"].size(0) for item in batch)
        B = len(batch)

        input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
        labels = torch.full((B, maxlen), pad_id, dtype=torch.long)
        response_mask = torch.zeros((B, maxlen), dtype=torch.bool)
        rewards = torch.zeros((B, 1), dtype=torch.float32)
        advantages = torch.zeros((B, 1), dtype=torch.float32)

        for i, item in enumerate(batch):
            L = item["input_ids"].size(0)
            input_ids[i, :L] = item["input_ids"]
            labels[i, :L] = item["labels"]
            response_mask[i, :L] = item["response_mask"]
            rewards[i, 0] = item["reward"]
            advantages[i, 0] = item["advantage"]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "response_mask": response_mask,
            "rewards": rewards,
            "advantages": advantages,
        }


    for grpo_step in range(1, N_GRPO_STEPS + 1):
        print(f"\n=== GRPO Step {grpo_step}/{N_GRPO_STEPS} ===")

        prompt_examples = random.sample(train_examples, n_prompts_per_rollout_batch)
        prompts = []
        true_answers = []
        
        for ex in prompt_examples:
            prompt = prompt_template.format(question=ex["problem"])
            prompts.append(prompt)
            true_answers.append(ex["answer"])

        print(f"Generating {ROLLOUT_BATCH_SIZE} rollouts ({GROUP_SIZE} per prompt)...")
        policy_model.eval()
        load_policy_into_vllm(policy_model, llm)
        
        results = llm.generate(prompts, sampling_params=sampling_params)

        rollouts = []
        all_responses = []
        repeated_ground_truths = []
        
        for i, result in enumerate(results):
            true_answer = true_answers[i]
            prompt = prompts[i]
            
            for output in result.outputs:
                response = output.text
                rollouts.append({
                    "prompt": prompt,
                    "response": response,
                    "true_answer": true_answer,
                })
                all_responses.append(response)
                repeated_ground_truths.append(true_answer)

        raw_rewards = []
        for rollout in rollouts:
            reward_dict = r1_zero_reward_fn(rollout["response"], rollout["true_answer"])
            raw_rewards.append(reward_dict["reward"])
            rollout["reward"] = reward_dict["reward"]

        advantages, raw_rewards_tensor, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=all_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=GROUP_SIZE,
            advantage_eps=ADVANTAGE_EPS,
            normalize_by_std=USE_STD_NORMALIZATION,
        )

        for i, rollout in enumerate(rollouts):
            rollout["advantage"] = advantages[i].item()

        print(f"Mean reward: {reward_metadata['raw_reward_mean']:.4f}")
        print(f"Mean advantage: {reward_metadata['advantage_mean']:.4f}")

        wandb.log({
            "grpo_step": grpo_step,
            "rollout_reward_mean": reward_metadata['raw_reward_mean'],
            "rollout_reward_std": reward_metadata['raw_reward_std'],
            "rollout_advantage_mean": reward_metadata['advantage_mean'],
            "rollout_advantage_std": reward_metadata['advantage_std'],
        })

        rollout_dataset = RolloutDataset(rollouts, tokenizer)
        rollout_dataloader = DataLoader(
            rollout_dataset,
            batch_size=micro_train_batch_size,
            shuffle=True,
            collate_fn=collate_rollouts_with_pad,
            drop_last=False,
        )

        policy_model.train()
        
        for epoch in range(EPOCHS_PER_ROLLOUT_BATCH):
            print(f"  Training epoch {epoch + 1}/{EPOCHS_PER_ROLLOUT_BATCH}...")
            
            epoch_losses = []
            
            for batch_idx, batch in enumerate(rollout_dataloader):
                input_ids = batch["input_ids"].to(POLICY_DEVICE)
                labels = batch["labels"].to(POLICY_DEVICE)
                response_mask = batch["response_mask"].to(POLICY_DEVICE)
                advantages = batch["advantages"].to(POLICY_DEVICE)
                rewards = batch["rewards"].to(POLICY_DEVICE)
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    out_dict = get_response_log_probs(policy_model, input_ids, labels)
                    policy_log_probs = out_dict["log_probs"]

                    loss, loss_metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                        loss_type=LOSS_TYPE,
                        advantages=advantages,
                    )

                    epoch_losses.append(loss.item())

                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy_model.parameters(), MAX_GRAD_NORM
                    )
                    
                    optimizer.step()                  
                    optimizer.zero_grad()

                    wandb.log({
                        "train_loss": loss.item(),
                        "grad_norm": grad_norm.item(),
                        "grpo_step": grpo_step,
                    })

            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Average loss: {avg_epoch_loss:.4f}")

        if grpo_step % VALIDATE_EVERY_STEPS == 0:
            print(f"  Running validation...")
            val_metrics = run_validation(
                policy_model, llm, val_examples, tokenizer, val_sampling_params, use_full_set=False
            )
            
            print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
            
            wandb.log({
                "val_accuracy": val_metrics['accuracy'],
                "val_avg_total_reward": val_metrics['avg_total_reward'],
                "val_avg_format_reward": val_metrics['avg_format_reward'],
                "val_avg_answer_reward": val_metrics['avg_answer_reward'],
                "grpo_step": grpo_step,
            })

        if grpo_step % 50 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"grpo_step_{grpo_step}.pt")
            torch.save(policy_model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    final_metrics = run_validation(
        policy_model, llm, val_examples, tokenizer, val_sampling_params, use_full_set=True
    )
    
    print(f"Final validation accuracy: {final_metrics['accuracy']:.4f}")
    
    wandb.log({
        "final_val_accuracy": final_metrics['accuracy'],
        "final_val_avg_total_reward": final_metrics['avg_total_reward'],
    })

    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(policy_model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")

    wandb.finish()
    print("GRPO training completed!")

def compute_group_normalized_rewards(reward_fn, rollout_responses, repeated_ground_truths, group_size, advantage_eps, normalize_by_std):
    rollout_batch_size = len(rollout_responses)
    assert len(repeated_ground_truths) == rollout_batch_size
    assert rollout_batch_size % group_size == 0
    
    n_groups = rollout_batch_size // group_size
    
    raw_rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        raw_rewards.append(reward_dict["reward"])
    
    raw_rewards = torch.tensor(raw_rewards, dtype=torch.float32)
    grouped_rewards = raw_rewards.view(n_groups, group_size)
    group_means = torch.mean(grouped_rewards, dim=1, keepdim=True)
    
    if normalize_by_std:
        group_stds = torch.std(grouped_rewards, dim=1, keepdim=True, unbiased=True)
        advantages_grouped = (grouped_rewards - group_means) / (group_stds + advantage_eps)
    else:
        advantages_grouped = grouped_rewards - group_means
    
    advantages = advantages_grouped.view(rollout_batch_size)
    
    metadata = {
        "raw_reward_mean": float(torch.mean(raw_rewards)),
        "raw_reward_std": float(torch.std(raw_rewards, unbiased=True)),
        "raw_reward_min": float(torch.min(raw_rewards)),
        "raw_reward_max": float(torch.max(raw_rewards)),
        "advantage_mean": float(torch.mean(advantages)),
        "advantage_std": float(torch.std(advantages, unbiased=True)),
        "group_mean_avg": float(torch.mean(group_means)),
    }
    
    if normalize_by_std:
        group_stds = torch.std(grouped_rewards, dim=1, unbiased=True)
        metadata.update({
            "group_std_avg": float(torch.mean(group_stds)),
            "group_std_min": float(torch.min(group_stds)),
            "group_std_max": float(torch.max(group_stds)),
        })
    
    return advantages, raw_rewards, metadata

def grpo_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, loss_type, **kwargs):
    per_token_losses, loss_metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        **kwargs
    )
    
    per_example_losses = masked_mean(per_token_losses, response_mask, dim=1)
    '''
    per_example_losses = masked_normalize(
        per_token_losses, 
        response_mask, 
        normalize_constant=SAMPLING_MAX_TOKENS,  # 1024 in your case
        dim=1
    )
    '''
    batch_loss = torch.mean(per_example_losses)
    scaled_loss = batch_loss / gradient_accumulation_steps
    scaled_loss.backward()
    
    metadata = loss_metadata.copy()
    metadata.update({
        "batch_loss": batch_loss.detach(),
        "scaled_loss": scaled_loss.detach(),
        "per_example_loss_mean": torch.mean(per_example_losses).detach(),
        "per_example_loss_std": torch.std(per_example_losses).detach(),
        "response_tokens_per_example": torch.sum(response_mask, dim=1).float().mean().detach(),
        "total_response_tokens": torch.sum(response_mask).detach(),
    })
    
    return batch_loss, metadata

def compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards=None, advantages=None, old_log_probs=None, cliprange=None):
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards is required for loss_type 'no_baseline'")
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {"loss_mean": torch.mean(loss), "loss_std": torch.std(loss)}
    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages is required for loss_type 'reinforce_with_baseline'")
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {"loss_mean": torch.mean(loss), "loss_std": torch.std(loss)}
    elif loss_type == "grpo_clip":
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError("advantages, old_log_probs, and cliprange are required for grpo_clip")
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return loss, metadata

def compute_naive_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs):
    batch_size, seq_length = policy_log_probs.shape
    advantages_broadcasted = raw_rewards_or_advantages.expand(batch_size, seq_length)
    return -advantages_broadcasted * policy_log_probs

def compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange):
    batch_size, seq_length = policy_log_probs.shape
    advantages_broadcasted = advantages.expand(batch_size, seq_length)
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    term1 = ratio * advantages_broadcasted
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    term2 = clipped_ratio * advantages_broadcasted
    
    loss = -torch.min(term1, term2)
    clipped_mask = (term2 < term1).float()
    
    metadata = {
        "clipped_fraction": torch.mean(clipped_mask),
        "ratio_mean": torch.mean(ratio),
        "ratio_std": torch.std(ratio),
        "ratio_min": torch.min(ratio),
        "ratio_max": torch.max(ratio),
        "log_ratio_mean": torch.mean(log_ratio),
        "log_ratio_std": torch.std(log_ratio),
    }
    
    return loss, metadata

def masked_mean(tensor, mask, dim=None):
    mask = mask.float()
    masked_tensor = tensor * mask
    
    if dim is None:
        total_sum = torch.sum(masked_tensor)
        count = torch.sum(mask)
        if count == 0:
            return torch.tensor(float('nan'), dtype=tensor.dtype, device=tensor.device)
        return total_sum / count
    else:
        total_sum = torch.sum(masked_tensor, dim=dim)
        count = torch.sum(mask, dim=dim)
        return torch.where(count > 0, total_sum / count, torch.full_like(total_sum, float('nan')))


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None) -> torch.Tensor:
    masked = tensor * mask
    summed = masked.sum(dim=dim)
    return summed / normalize_constant

if __name__ == "__main__":
    main()