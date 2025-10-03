import torch

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    batch_size = len(prompt_strs)
    pids = []
    oids = []
    seqs = []
    lengths = []
    for p, o in zip(prompt_strs, output_strs):
        pid = tokenizer.encode(p, add_special_tokens=False)
        oid = tokenizer.encode(o, add_special_tokens=False)
        pids.append(pid)
        oids.append(oid)

        seq = pid + oid
        seqs.append(seq)
        lengths.append(len(seq))

    max_len = max(lengths)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    full = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        full[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    input_ids = full[:, :-1]
    labels = full[:, 1:]

    response_mask = torch.zeros_like(labels, dtype=torch.bool)
    for i, (pid, oid) in enumerate(zip(pids, oids)):
        plen = len(pid)
        olen = len(oid)
        if olen == 0:
            continue
        start = plen - 1
        end = plen + olen - 2
        response_mask[i, start : end + 1] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    lse = logits.logsumexp(dim=-1, keepdim=True) 
    log_probs = logits - lse 
    probs = log_probs.exp()
    entropy = - (probs * log_probs).sum(dim=-1)
    return entropy

def get_response_log_probs(model, input_ids: torch.Tensor, labels: torch.Tensor, return_token_entropy: bool = False) -> dict[str, torch.Tensor]:
    outputs = model(input_ids)
    logits = outputs.logits          

    log_probs_all = torch.log_softmax(logits, dim=-1) 

    log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    result = {"log_probs": log_probs}

    if return_token_entropy:
        with torch.no_grad():
            vocab_log_probs = log_probs_all
            vocab_probs = vocab_log_probs.exp()  
            token_entropy = - (vocab_probs * vocab_log_probs).sum(dim=-1) 
        result["token_entropy"] = token_entropy

    return result

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None) -> torch.Tensor:
    masked = tensor * mask
    summed = masked.sum(dim=dim)
    return summed / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    batch_size = policy_log_probs.size(0)
    total_logprob = (policy_log_probs * response_mask).sum()
    nll = - total_logprob / normalize_constant
    avg_nll = nll / batch_size
    microbatch_loss = avg_nll / gradient_accumulation_steps

    microbatch_loss.backward()

    return microbatch_loss, {}

import torch
from typing import List, Dict, Any

@torch.no_grad()
def log_generations(
    model,
    tokenizer,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn,
    device: torch.device = None,
    max_generation_length: int = 128) -> Dict[str, Any]:

    if device is None:
        device = next(model.parameters()).device
    model.eval()

    per_example = []
    total_response_tokens = 0
    total_entropy_sum = 0.0
    lengths_correct = []
    lengths_incorrect = []
    total_resp_len_all = 0

    for prompt_str, gt_str in zip(prompts, ground_truths):
        prompt_ids = tokenizer.encode(
            prompt_str,
            return_tensors="pt",
            add_special_tokens=False
        ).to(device)

        gen_out = model.generate(
            prompt_ids,
            max_new_tokens=max_generation_length,
            pad_token_id=tokenizer.eos_token_id,
        )

        gen_ids = gen_out[0]         
        prompt_len = prompt_ids.size(1)
        resp_ids = gen_ids[prompt_len:] 

        response_str = tokenizer.decode(resp_ids, skip_special_tokens=True).strip()

        reward_info = reward_fn(prompt_str, response_str, gt_str)

        full_ids = torch.cat([prompt_ids, resp_ids.unsqueeze(0)], dim=1) 

        out = _response_log_probs(
            model=model,
            input_ids=full_ids,
            labels=full_ids,
            return_token_entropy=True
        )

        token_entropies = out["token_entropy"].squeeze(0) 

        if resp_ids.numel() > 0:
            resp_entropies = token_entropies[prompt_len : prompt_len + resp_ids.size(0)]
            avg_entropy = resp_entropies.mean().item()
        else:
            resp_entropies = torch.tensor([], device=device)
            avg_entropy = 0.0

        resp_len = resp_ids.size(0)

        is_correct = (response_str.strip() == gt_str.strip())

        total_resp_len_all += resp_len
        total_response_tokens += resp_len
        total_entropy_sum += resp_entropies.sum().item()

        if is_correct:
            lengths_correct.append(resp_len)
        else:
            lengths_incorrect.append(resp_len)

        per_example.append({
            "prompt": prompt_str,
            "generation": response_str,
            "ground_truth": gt_str,
            "reward": reward_info,
            "avg_token_entropy": avg_entropy,
            "response_length": resp_len,
            "is_correct": is_correct,
        })

    num_examples = len(prompts)
    avg_resp_len_all = (total_resp_len_all / num_examples) if num_examples > 0 else 0.0
    avg_entropy_all = (total_entropy_sum / total_response_tokens) if total_response_tokens > 0 else 0.0
    avg_resp_len_corr = (
        sum(lengths_correct) / len(lengths_correct) if lengths_correct else 0.0
    )
    avg_resp_len_incorr = (
        sum(lengths_incorrect) / len(lengths_incorrect) if lengths_incorrect else 0.0
    )

    summary = {
        "num_examples": num_examples,
        "avg_response_length": avg_resp_len_all,
        "avg_response_length_correct": avg_resp_len_corr,
        "avg_response_length_incorrect": avg_resp_len_incorr,
        "avg_token_entropy_all": avg_entropy_all,
    }

    return {"per_example": per_example, "summary": summary}
