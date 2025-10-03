
import os
import hashlib
from typing import List

def exact_line_deduplication(input_paths: List[str], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    counts: dict[bytes, int] = {}
    for path in input_paths:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                h = hashlib.sha256(line.encode('utf-8')).digest()
                counts[h] = counts.get(h, 0) + 1

    for path in input_paths:
        fname = os.path.basename(path)
        out_path = os.path.join(output_dir, fname)
        with open(path, 'r', encoding='utf-8', errors='ignore') as src, \
             open(out_path, 'w', encoding='utf-8') as dst:
            for line in src:
                h = hashlib.sha256(line.encode('utf-8')).digest()
                if counts.get(h, 0) == 1:
                    dst.write(line)
