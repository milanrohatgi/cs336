import cProfile
from bpe import BPETokenizer

def main():
    tokenizer = BPETokenizer("../data/TinyStoriesV2-GPT4-valid.txt", 10000, ["<|endoftext|>"])
    (vocab, merges) = tokenizer.train()

if __name__ == "__main__":
    cProfile.run("main()", "profile.out")
