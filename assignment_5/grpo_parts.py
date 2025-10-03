import torch
from typing import Callable, List, Dict, Tuple, Literal, Optional

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
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
        metadata["group_std_avg"] = float(torch.mean(group_stds))
        metadata["group_std_min"] = float(torch.min(group_stds))
        metadata["group_std_max"] = float(torch.max(group_stds))
    
    return advantages, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_length = policy_log_probs.shape
    assert raw_rewards_or_advantages.shape == (batch_size, 1), \
        f"Expected raw_rewards_or_advantages shape ({batch_size}, 1), got {raw_rewards_or_advantages.shape}"

    advantages_broadcasted = raw_rewards_or_advantages.expand(batch_size, seq_length)

    policy_gradient_loss = -advantages_broadcasted * policy_log_probs
    
    return policy_gradient_loss

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, seq_length = policy_log_probs.shape
    assert advantages.shape == (batch_size, 1), \
        f"Expected advantages shape ({batch_size}, 1), got {advantages.shape}"
    assert old_log_probs.shape == (batch_size, seq_length), \
        f"Expected old_log_probs shape ({batch_size}, {seq_length}), got {old_log_probs.shape}"
    
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


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards is required for loss_type 'no_baseline'")

        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)

        metadata = {
            "loss_mean": torch.mean(loss),
            "loss_std": torch.std(loss),
        }
        
    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages is required for loss_type 'reinforce_with_baseline'")

        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)

        metadata = {
            "loss_mean": torch.mean(loss),
            "loss_std": torch.std(loss),
        }
        
    elif loss_type == "grpo_clip":
        if advantages is None:
            raise ValueError("advantages is required for loss_type 'grpo_clip'")
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for loss_type 'grpo_clip'")
        if cliprange is None:
            raise ValueError("cliprange is required for loss_type 'grpo_clip'")
        

        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange
        )
        
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Must be one of 'no_baseline', 'reinforce_with_baseline', 'grpo_clip'")
    
    return loss, metadata

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
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
    

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    per_token_losses, loss_metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    per_example_losses = masked_mean(
        tensor=per_token_losses,
        mask=response_mask,
        dim=1  
    )
    
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
