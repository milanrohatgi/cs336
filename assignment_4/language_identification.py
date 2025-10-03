import gzip
import random
from fastwarc import ArchiveIterator
import fasttext

def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    try:
        html_str = html_bytes.decode('utf-8')
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding)
    return extract_plain_text(html_str)

def identify_language(text: str) -> tuple[str, float]:
    model = fasttext.load_model("/data/classifiers/lid.176.bin")
    sanitized_text = text.replace('\n', ' ')
    predictions = model.predict(sanitized_text, k=1)
    label = predictions[0][0]
    score = predictions[1][0]
    lang_code = label.replace('__label__', '')
    return lang_code, score

def run_langid_on_random_warc(warc_path: str, target_samples: int = 20, sample_prob: float = 0.002):
    model = fasttext.load_model("/data/classifiers/lid.176.bin")
    samples_collected = 0

    with gzip.open(warc_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.headers.get("WARC-Type") != "response":
                continue
            if random.random() > sample_prob:
                continue

            html_bytes = record.reader.read()
            text = extract_text_from_html_bytes(html_bytes)
            if not text.strip():
                continue

            lang, conf = identify_language(text)
            excerpt = text[:300].replace("\n", " ").strip()

            print(f"Example {samples_collected + 1}")
            print(f"Excerpt: {excerpt}")
            print(f"Predicted Language: {lang}, Confidence: {conf:.3f}")
            print("=" * 80)

            samples_collected += 1
            if samples_collected >= target_samples:
                break

if __name__ == "__main__":
    run_langid_on_random_warc("/data/CC/example.warc.gz")
