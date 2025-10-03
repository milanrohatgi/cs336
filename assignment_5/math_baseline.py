#!/usr/bin/env python3
import os
import json
from typing import List, Callable, Dict
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

MODEL_PATH = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
VALIDATION_JSONL = "/data/a5-alignment/MATH/validation.jsonl"
OUTPUT_DIR = "./math_zero_shot_results"
BATCH_SIZE = 256

with open("./prompts/r1_zero.prompt", "r", encoding="utf-8") as f:
    R1_ZERO_TEMPLATE = f.read()

def load_examples(path: str) -> List[Dict]:
    exs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            exs.append(json.loads(line))
    return exs

def format_prompt(q: str) -> str:
    return R1_ZERO_TEMPLATE.format(question=q)

def extract_answer(text: str) -> str:
    start = text.find("<answer>")
    if start == -1:
        return ""
    start += len("<answer>")
    end = text.find("</answer>", start)
    return text[start:end].strip() if end != -1 else text[start:].strip()

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    examples: List[Dict],
    sampling_params: SamplingParams,
    output_jsonl_path: str,
    batch_size: int = 8
) -> None:
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    out_f = open(output_jsonl_path, "w", encoding="utf-8")
    num_correct = 0
    total = len(examples)

    for i in range(0, total, batch_size):
        batch = examples[i : i + batch_size]
        prompts = [format_prompt(e["problem"]) for e in batch]
        outputs = vllm_model.generate(prompts, sampling_params)
        for ex, ro in zip(batch, outputs):
            full = ro.outputs[0].text
            ans = extract_answer(full)
            metrics = reward_fn(full, ex["answer"])
            if metrics.get("answer_reward", 0) >= 1.0:
                num_correct += 1
            record = {
                "ground_truth": ex["answer"],
                "answer": ans,
                "full answer": full,
                "metrics": metrics
            }
            out_f.write(json.dumps(record) + "\n")

    out_f.close()
    accuracy = num_correct / total
    print(f"Total examples: {total}")
    print(f"Number correct: {num_correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    summary = {"total": total, "correct": num_correct, "accuracy": accuracy}
    summary_path = os.path.splitext(output_jsonl_path)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)
    return summary

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    examples = load_examples(VALIDATION_JSONL)

    llm = LLM(model=MODEL_PATH)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"]
    )
    sampling_params.include_stop_str_in_output = True

    output_jsonl = os.path.join(OUTPUT_DIR, "math_zero_shot_outputs.jsonl")
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        examples=examples,
        sampling_params=sampling_params,
        output_jsonl_path=output_jsonl,
        batch_size=BATCH_SIZE
    )
