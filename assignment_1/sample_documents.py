import random

def sample_documents(tinystories_path: str, openwebtext_path: str, seed: int = 42):
    random.seed(seed)
    delimiter = "<|endoftext|>"
    max_samples = 10

    def stream_and_sample(path, name):
        reservoir = []
        current_doc = []
        doc_count = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if delimiter in line:
                    parts = line.split(delimiter)
                    for i, part in enumerate(parts):
                        current_doc.append(part)
                        doc = "".join(current_doc).strip()
                        if doc:
                            doc_count += 1
                            if len(reservoir) < max_samples:
                                reservoir.append(doc)
                            else:
                                r = random.randint(0, doc_count - 1)
                                if r < max_samples:
                                    reservoir[r] = doc
                        current_doc = [] if i < len(parts) - 1 else [parts[-1]]
                else:
                    current_doc.append(line)

        output_text = f"{delimiter}\n".join(reservoir)
        out_path = f"{name}_sample.txt"
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(output_text)
        byte_size = len(output_text.encode("utf-8"))
        print(f"{name}_sample.txt has {byte_size} bytes")
        return out_path

    tinystories_sample = stream_and_sample(tinystories_path, "tinystories")
    openwebtext_sample = stream_and_sample(openwebtext_path, "openwebtext")

    return tinystories_sample, openwebtext_sample

if __name__ == "__main__":
    sample_documents("../data/TinyStoriesV2-GPT4-train.txt", "../data/owt_train.txt")
