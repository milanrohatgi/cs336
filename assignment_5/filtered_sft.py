import json

INPUT_JSONL = "/data/a5-alignment/MATH/sft.jsonl"
OUTPUT_JSONL = "./filtered_sft.jsonl"

examples = []
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))
print(len(examples))

filtered = []
for ex in examples:
    resp = ex["response"]
    start = resp.find("<answer>")
    end   = resp.find("</answer>")
    if start != -1 and end != -1:
        ans = resp[start + len("<answer>") : end].strip()
        if ans == ex["ground_truth"].strip():
            filtered.append(ex)

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    print(len(filtered))
    for e in filtered:
        f.write(json.dumps(e) + "\n")
