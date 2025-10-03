import numpy as np
from pathlib import Path
from tqdm import tqdm
input_dir = Path("/data/c-mrohatgi/tokenized_chunks_2")
out_path = "/data/c-mrohatgi/tokenized_data_2.npy"

all_files = sorted(input_dir.glob("*.npy"))
with open(out_path, "wb") as fout:
    for f in tqdm(all_files, desc="Concatenating shards"):
        chunk = np.fromfile(f, dtype=np.uint16)
        fout.write(chunk.tobytes())
