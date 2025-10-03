import os
import shutil
import hashlib
import unicodedata
import re
import random
from typing import List, Set, Dict
def normalize(text):
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_ngrams(text, n):
    tokens = text.split()
    return {' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

def hash_fn(gram: str, seed: int) -> int:
    data = f"{seed}-{gram}".encode('utf-8')
    return int(hashlib.md5(data).hexdigest(), 16)

def minhash_deduplication(
    input_paths: List[str],
    num_hashes: int,
    num_bands: int,
    ngram_len: int,
    output_dir: str,
    threshold: float
) -> None:

    docs_ngrams: Dict[str, Set[str]] = {}
    for path in input_paths:
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()
        norm = normalize(raw)
        docs_ngrams[path] = get_ngrams(norm, ngram_len)

    signatures: Dict[str, List[int]] = {}
    for path, ngrams in docs_ngrams.items():
        sig = []
        for i in range(num_hashes):
            min_h = min(hash_fn(g, i) for g in ngrams) if ngrams else 0
            sig.append(min_h)
        signatures[path] = sig

    r = num_hashes // num_bands
    buckets: Dict[tuple, List[str]] = {}
    for path, sig in signatures.items():
        for b in range(num_bands):
            start, end = b*r, (b+1)*r
            band_sig = tuple(sig[start:end])
            key = (b, band_sig)
            buckets.setdefault(key, []).append(path)

    cand_pairs = set()
    for band_docs in buckets.values():
        if len(band_docs) > 1:
            for i in range(len(band_docs)):
                for j in range(i+1, len(band_docs)):
                    a, b = band_docs[i], band_docs[j]
                    cand_pairs.add((a, b))

    graph = {path: set() for path in input_paths}
    for a, b in cand_pairs:
        A, B = docs_ngrams[a], docs_ngrams[b]
        if not A and not B:
            sim = 0.0
        else:
            inter = len(A & B)
            union = len(A | B)
            sim = inter/union if union else 0.0
        if sim >= threshold:
            graph[a].add(b)
            graph[b].add(a)

    visited = set()
    clusters: List[Set[str]] = []
    for node in input_paths:
        if node not in visited:
            stack = [node]
            comp = set()
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.add(u)
                for v in graph[u]:
                    if v not in visited:
                        stack.append(v)
            clusters.append(comp)

    keep = set()
    for comp in clusters:
        keep.add(random.choice(list(comp)))

    os.makedirs(output_dir, exist_ok=True)
    for path in input_paths:
        if path in keep:
            dst = os.path.join(output_dir, os.path.basename(path))
            shutil.copy2(path, dst)
