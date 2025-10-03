import numpy as np
import os
import multiprocessing
from functools import partial
from tokenizer import Tokenizer
from typing import BinaryIO, Tuple
import re
import cProfile
import time
from tqdm import tqdm


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def encode_chunk_from_file(args: Tuple[str, int, int, Tokenizer, int, str, np.dtype]) -> None:
    """
    Encode a chunk and write directly to a unique file with chunk index in name.
    Returns nothing to avoid memory overhead.
    """
    file_path, start, end, tokenizer, chunk_idx, temp_dir, dtype = args
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="replace")
        print(f"chunk idx is {chunk_idx}, length of chunk is {len(chunk)}")
        tokens = tokenizer.encode(chunk)
        
        # Write to a temporary chunk file
        temp_chunk_file = os.path.join(temp_dir, f"chunk_{chunk_idx:06d}.npy")
        np.array(tokens, dtype=dtype).tofile(temp_chunk_file)
        
        return chunk_idx

def encode(
    tokenizer: Tokenizer, 
    filepath: str, 
    output_path: str,
    split_token: str, 
    num_processes: int = min(8, multiprocessing.cpu_count()),
    dtype=np.uint16,
) -> None:
    split_bytes = split_token.encode("utf-8")
    
    temp_dir = f"{output_path}_chunks"
    os.makedirs(temp_dir, exist_ok=True)
    
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes * 8, split_bytes)
    
    chunk_ranges = [
        (filepath, boundaries[i], boundaries[i+1], tokenizer, i, temp_dir, dtype) 
        for i in range(len(boundaries) - 1)
    ]
    
    total_chunks = len(chunk_ranges)
    print(f"Processing {total_chunks} chunks with {num_processes} processes")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        for _ in tqdm(
            pool.imap_unordered(encode_chunk_from_file, chunk_ranges),
            total=total_chunks,
            desc="Encoding chunks"
        ):
            pass  

    print("Combining chunks in order...")
    with open(f"{output_path}.tmp", 'wb') as outfile:
        for i in tqdm(range(total_chunks), desc="Merging"):
            chunk_file = os.path.join(temp_dir, f"chunk_{i:06d}.npy")
            with open(chunk_file, 'rb') as infile:
                outfile.write(infile.read())
            # Delete chunk file after use
            os.remove(chunk_file)
    
    print(f"Finalizing output...")
    tokens = np.fromfile(f"{output_path}.tmp", dtype=dtype)
    np.save(output_path, tokens)
        
    
    os.remove(f"{output_path}.tmp")
    os.rmdir(temp_dir)
    print(f"Saved {len(tokens)} tokens to {output_path}")


def save_encoded_tokens(encoded_array: np.ndarray, output_path: str):
    np.save(output_path, encoded_array)

def main():

    tokenizer = Tokenizer.from_files("tiny_bpe_vocab.json", "tiny_bpe_merges.json", special_tokens=['<|endoftext|>'])
    encode(tokenizer, "openwebtext_sample.txt", "owt-sample-tokenized.npy", split_token='<|endoftext|>')

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Took {end - start:.2f} seconds")