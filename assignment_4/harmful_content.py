import gzip
import random
from fastwarc import ArchiveIterator
import fasttext
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

nsfw_model = fasttext.load_model("/data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
toxic_model = fasttext.load_model("/data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")

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

def classify_nsfw(text: str) -> tuple[str, float]:
    text = ' '.join(text.splitlines())
    labels, probs = nsfw_model.predict(text, k=1)
    label = labels[0].replace('__label__', '')
    confidence = probs[0]
    return label, confidence


def classify_toxic_speech(text: str) -> tuple[str, float]:
    text = ' '.join(text.splitlines())
    labels, probs = toxic_model.predict(text, k=1)
    label = labels[0].replace('__label__', '')
    confidence = probs[0]
    return label, confidence

def run_harmful_content_filter_on_english(warc_path: str, target_samples: int = 20, sample_prob: float = 0.02):
    samples_collected = 0

    with gzip.open(warc_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.headers.get("WARC-Type") != "response":
                continue
            if random.random() > sample_prob:
                continue

            html_bytes = record.reader.read()
            text = extract_text_from_html_bytes(html_bytes)
            
            lang, conf = identify_language(text)
            if lang != "en" or conf < 0.7:
                continue
            
            nsfw_label, nsfw_score = classify_nsfw(text)
            toxic_label, toxic_score = classify_toxic_speech(text)

            excerpt = text[:300].replace("\n", " ").strip()

            print(f"Example {samples_collected + 1}")
            print(f"Excerpt: {excerpt}")
            print(f"NSFW: {nsfw_label.replace('__label__', '')} ({nsfw_score:.3f})")
            print(f"Toxic: {toxic_label.replace('__label__', '')} ({toxic_score:.3f})")
            print("=" * 80)

            samples_collected += 1
            if samples_collected >= target_samples:
                break

if __name__ == "__main__":
    run_harmful_content_filter_on_english("/data/CC/example.warc.gz")
