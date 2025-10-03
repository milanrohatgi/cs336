import torch
from collections.abc import Iterator, Iterable
import json
import multiprocessing
import regex as re

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        if special_tokens:
            escaped_specials = [re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)]
            self.special_pattern = re.compile("|".join(escaped_specials))
        else:
            self.special_pattern = ""
        
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r') as vf:
            raw_vocab = json.load(vf)

        vocab = {
            int(k): bytes(v['bytes'])
            for k, v in raw_vocab.items()
        }

        with open(merges_filepath, 'r') as mf:
            raw_merges = json.load(mf)

        merges = [
            (bytes(entry['first']['bytes']), bytes(entry['second']['bytes']))
            for entry in raw_merges
        ]

        return cls(vocab, merges, special_tokens)
    
    def _bpe_encode(self, text: str) -> list[int]:
        matches = list(self.PAT.finditer(text))
        tokens = []
        
        for match in matches:
            token = match.group(0)
            token_bytes = token.encode("utf-8")
            
            parts = [bytes([b]) for b in token_bytes]
            
            pairs = [(i, (parts[i], parts[i+1])) for i in range(len(parts)-1)]
            
            while pairs:
                best_idx = -1
                best_rank = float('inf')
                
                for i, (idx, pair) in enumerate(pairs):
                    if pair in self.merge_ranks:
                        rank = self.merge_ranks[pair]
                        if rank < best_rank:
                            best_rank = float('inf') if rank is None else rank
                            best_idx = i
                
                if best_idx == -1:
                    break 
                    
                idx, pair = pairs.pop(best_idx)
                parts[idx] = pair[0] + pair[1]
                parts.pop(idx + 1)
                
                pairs = []
                for i in range(len(parts)-1):
                    pairs.append((i, (parts[i], parts[i+1])))
            
            for part in parts:
                token_id = self.inv_vocab.get(part)
                tokens.append(token_id)
                    
        return tokens


    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            segments = self.special_pattern.split(text)
            matches = list(self.special_pattern.finditer(text))

            tokens = []
            for i, segment in enumerate(segments):
                tokens.extend(self._bpe_encode(segment))
                if i < len(matches):
                    special_token = matches[i].group(0)
                    token_bytes = special_token.encode("utf-8")
                    token_id = next(
                        k for k, v in self.vocab.items() if v == token_bytes
                    )
                    tokens.append(token_id)

            return tokens
        else:
            return self._bpe_encode(text)


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        byte_sequence = b"".join(self.vocab[i] for i in ids)
        return byte_sequence.decode("utf-8", errors="replace")