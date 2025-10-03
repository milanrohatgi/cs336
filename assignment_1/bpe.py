import regex as re
import os
from collections import Counter
from collections import defaultdict
from typing import BinaryIO
import multiprocessing
import math
from tqdm import tqdm  
import heapq
import gc

# Helper class for inverting string comparison in heap
class InvertedBytes:
    def __init__(self, b):
        self.b = b  # b is a bytes object

    def __lt__(self, other):
        # We want the lexicographically greater bytes to win ties.
        # So, if self.b > other.b, then self should be considered "less" for a min-heap.
        return self.b > other.b

    def __eq__(self, other):
        return self.b == other.b

    def __repr__(self):
        return f"InvertedBytes({self.b})"


class BPETokenizer:
    def __init__(self, input_path, vocab_size, special_tokens):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        self.frequency_counts = Counter()
        
        self.vocab = {}
        self.merges = []

    def initialize_vocab(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i in range(len(self.special_tokens)):
            self.vocab[256+i] = self.special_tokens[i].encode("utf-8")

    def find_chunk_boundaries(self, file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]  # Chunks start on previous index, don't include last index
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk
                if mini_chunk == b"":  # If EOF, this boundary should be at the end of the file
                    chunk_boundaries[bi] = file_size
                    break
                found_at = mini_chunk.find(split_special_token)  # Find the special token in the mini chunk
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))


    def pretokenize(self, num_chunks: int = 8):
        split_token = self.special_tokens[0].encode("utf-8")

        with open(self.input_path, 'rb') as f:
            boundaries = self.find_chunk_boundaries(f, num_chunks * 6, split_token)
        
        chunk_boundaries = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]
        
        worker_args = [(self.input_path, start, end, self.special_tokens) 
                    for start, end in chunk_boundaries]

        with multiprocessing.Pool(min(num_chunks, multiprocessing.cpu_count())) as pool:
            results = list(pool.imap(self.process_chunk_from_file, worker_args, chunksize=1))

        for counter in results:
            self.frequency_counts.update(counter)

    @staticmethod
    def process_chunk_from_file(args):
        file_path, start, end, special_tokens = args
        
        PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        escaped_specials = [re.escape(token) for token in special_tokens]
        special_split_pattern = re.compile('|'.join(escaped_specials))
        
        with open(file_path, 'rb') as f:
            f.seek(start)
            chunk_data = f.read(end - start).decode("utf-8", errors="ignore")
        
        frequency_counts = Counter()
        segments = re.split(special_split_pattern, chunk_data)
        
        for segment in segments:
            for match in PAT.finditer(segment):
                token = match.group(0)
                token_bytes = tuple(token.encode("utf-8"))
                frequency_counts[token_bytes] += 1
        
        return frequency_counts

    @staticmethod
    def compute_local_pair_counts(chunk):
        local_counter = Counter()
        for token, count in chunk:
            for i in range(len(token) - 1):
                local_counter[(token[i], token[i+1])] += count
        return local_counter

    
    def merge(self):
        num_merges = self.vocab_size - len(self.vocab)
        initial_vocab_length = len(self.vocab)

        token_items = list(self.frequency_counts.items())
        num_processes = min(8, multiprocessing.cpu_count())
        chunk_size = math.ceil(len(token_items) / num_processes)
        chunks = [token_items[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]

        with multiprocessing.Pool(num_processes) as pool:
            results = list(pool.imap(self.compute_local_pair_counts, chunks))

        pair_counts = Counter()
        for local_counter in results:
            pair_counts.update(local_counter)

        pair_to_tokens = defaultdict(set)
        for token in self.frequency_counts:
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_to_tokens[pair].add(token)


        priority_queue = []
        for pair, count in pair_counts.items():
            token1_bytes = self.vocab[pair[0]]
            token2_bytes = self.vocab[pair[1]]

            entry = (-count, InvertedBytes(token1_bytes), InvertedBytes(token2_bytes), pair)
            priority_queue.append(entry)
        
        heapq.heapify(priority_queue)
        
        for j in tqdm(range(num_merges)):
            
            while priority_queue:
                neg_count, token1_str, token2_str, most_frequent_pair = heapq.heappop(priority_queue)
                actual_count = pair_counts.get(most_frequent_pair, 0)
                if actual_count > 0 and -neg_count == actual_count:
                    break
            
            self.merges.append((self.vocab[most_frequent_pair[0]], self.vocab[most_frequent_pair[1]]))
            new_index = initial_vocab_length + j
            new_token = self.vocab[most_frequent_pair[0]] + self.vocab[most_frequent_pair[1]]
            self.vocab[new_index] = new_token

            affected_tokens = pair_to_tokens[most_frequent_pair].copy()
            del pair_counts[most_frequent_pair]
            del pair_to_tokens[most_frequent_pair]
            
            pairs_to_update = set()

            for token in affected_tokens:

                for i in range(len(token) - 1):
                    current_pair = (token[i], token[i+1])
                    pair_counts[current_pair] -= self.frequency_counts[token]
                    pair_to_tokens[current_pair].discard(token)
                    pairs_to_update.add(current_pair)

                new_tuple = []
                i = 0
                while i < len(token):
                    if i < len(token) - 1 and (token[i], token[i+1]) == most_frequent_pair:
                        new_tuple.append(new_index)
                        i += 2
                    else:
                        new_tuple.append(token[i])
                        i += 1

                new_tuple = tuple(new_tuple)

                for i in range(len(new_tuple) - 1):
                    current_pair = (new_tuple[i], new_tuple[i+1])
                    pair_counts[current_pair] += self.frequency_counts[token]
                    pair_to_tokens[current_pair].add(new_tuple)
                    pairs_to_update.add(current_pair)

                if new_tuple != token:
                    self.frequency_counts[new_tuple] += self.frequency_counts[token]
                    del self.frequency_counts[token]
            
            for pair in pairs_to_update:
                count = pair_counts.get(pair, 0)
                if count <= 0:
                    if pair in pair_counts:
                        del pair_counts[pair]
                    if pair in pair_to_tokens and not pair_to_tokens[pair]:
                        del pair_to_tokens[pair]
                else:
                    token1_bytes = self.vocab[pair[0]]
                    token2_bytes = self.vocab[pair[1]]
                    new_entry = (-count, InvertedBytes(token1_bytes), InvertedBytes(token2_bytes), pair)
                    heapq.heappush(priority_queue, new_entry)


    def train(self):
        print("Initializing vocabulary")
        self.initialize_vocab()
        
        print("Pretokenizing data")
        self.pretokenize()
        
        print("Performing BPE merges")
        self.merge()
        
        return (self.vocab, self.merges)
