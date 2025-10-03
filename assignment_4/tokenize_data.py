import os
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_line_and_add_eos(line):
    line = line.strip().replace("\n", " ") 
    return tokenizer.encode(line) + [tokenizer.eos_token_id]

def run_job(file_index: int, input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    all_files = sorted(p for p in input_dir.iterdir() if p.is_file())
    file_path = all_files[file_index]
    file_name = file_path.stem

    output_path = output_dir / f"{file_index:05d}_{file_name}.npy"
    token_ids = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            token_ids.extend(tokenize_line_and_add_eos(line))

    token_array = np.array(token_ids, dtype=np.uint16)
    token_array.tofile(output_path)
    return f"Processed {file_path} -> {output_path}"
