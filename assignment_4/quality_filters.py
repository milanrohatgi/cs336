import re
import nltk
from nltk.tokenize import word_tokenize
import re
import gzip
import random
from fastwarc import ArchiveIterator
import fasttext
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

nltk.download('punkt_tab')

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

def gopher_quality_filter(text: str) -> bool:
    words = word_tokenize(text)
    num_words = len(words)

    if num_words < 50 or num_words > 100_000:
        return False

    word_lengths = [len(w) for w in words]
    mean_word_len = sum(word_lengths) / num_words
    if mean_word_len < 3 or mean_word_len > 10:
        return False
    lines = text.splitlines()
    if lines:
        num_ellipsis_lines = sum(1 for line in lines if line.strip().endswith("..."))
        if num_ellipsis_lines / len(lines) > 0.3:
            return False

    has_alpha = [bool(re.search(r"[A-Za-z]", w)) for w in words]
    alpha_frac = sum(has_alpha) / num_words
    if alpha_frac < 0.8:
        return False

    return True

def gopher_filter_on_random_warc(warc_path: str, target_samples: int = 20, sample_prob: float = 0.002):
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

            lang, _ = identify_language(text)
            if lang != "en":
                continue

            excerpt = text[:5000].replace("\n", " ").strip()
            passed_filter = gopher_quality_filter(text)

            samples_collected += 1
            if samples_collected >= target_samples:
                break

if __name__ == "__main__":
    gopher_filter_on_random_warc("/data/CC/example.warc.gz")
