import json
import os
import time
import sys
from bpe import BPETokenizer


def main():
    start_time = time.time()
    
    tokenizer = BPETokenizer("../data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    
    (vocab, merges) = tokenizer.train()

    end_time = time.time()
    training_time = end_time - start_time

    serializable_vocab = {}
    for k, v in vocab.items():
        bytes_list = list(v)
        try:
            utf8_str = v.decode('utf-8')
            display_str = repr(utf8_str)[1:-1]
        except UnicodeDecodeError:
            display_str = f"<bytes: {bytes_list}>"
        
        serializable_vocab[k] = {
            "bytes": bytes_list,
            "string": display_str
        }
    
    with open("tiny_bpe_vocab.json", "w") as f:
        json.dump(serializable_vocab, f, indent=2)
    
    serializable_merges = []
    for first, second in merges:
        first_bytes = list(first)
        second_bytes = list(second)
        
        try:
            first_str = repr(first.decode('utf-8'))[1:-1]
        except UnicodeDecodeError:
            first_str = f"<bytes: {first_bytes}>"
            
        try:
            second_str = repr(second.decode('utf-8'))[1:-1]
        except UnicodeDecodeError:
            second_str = f"<bytes: {second_bytes}>"
        
        serializable_merges.append({
            "first": {
                "bytes": first_bytes,
                "string": first_str
            },
            "second": {
                "bytes": second_bytes,
                "string": second_str
            }
        })
    
    with open("tiny_bpe_merges.json", "w") as f:
        json.dump(serializable_merges, f, indent=2)
    
    
    # Find the longest token in the vocabulary
    longest_token = max(vocab.values(), key=len)
    longest_token_bytes = list(longest_token)
    longest_token_str = longest_token.decode('utf-8', errors='replace')
    
    print("\nTraining Statistics:")
    print(f"Training time: {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    print("\nLongest Token Analysis:")
    print(f"Longest token length: {len(longest_token)} bytes")
    print(f"Longest token as bytes: {longest_token_bytes}")
    print(f"Longest token as string: '{longest_token_str}'")

if __name__ == "__main__":
    main()