import numpy as np
import os

def compute_compression_ratio(txt_path: str, npy_path: str) -> float:
    original_size_bytes = os.path.getsize(txt_path)
    token_ids = np.load(npy_path, mmap_mode="r")
    num_tokens = len(token_ids)

    return original_size_bytes / num_tokens

if __name__ == "__main__":
    print(compute_compression_ratio("../data/owt_train.txt", "owt-train-tokenized.npy"))