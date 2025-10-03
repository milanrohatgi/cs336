import gzip
import re
import os
from fastwarc.warc import ArchiveIterator
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
from nltk.tokenize import word_tokenize

lang_model = fasttext.load_model("/data/classifiers/lid.176.bin")
nsfw_model = fasttext.load_model("/data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
toxic_model = fasttext.load_model("/data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")

def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    try:
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding)
    except:
        html_str = ""
    return extract_plain_text(html_str)

def identify_language(text: str) -> tuple[str, float]:
    sanitized_text = text.replace('\n', ' ')
    predictions = lang_model.predict(sanitized_text, k=1)
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

def process_all_warcs(input_dir: str, output_path: str):
    good_count = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for fname in os.listdir(input_dir):
            if not fname.endswith(".warc.gz"):
                continue
            warc_path = os.path.join(input_dir, fname)
            print(f"Processing {warc_path}...")

            with gzip.open(warc_path, "rb") as stream:
                for record in ArchiveIterator(stream):
                    if record.headers.get("WARC-Type") != "response":
                        continue
                    html_bytes = record.reader.read()
                    text = extract_text_from_html_bytes(html_bytes)
                    if not text.strip():
                        continue

                    lang, lang_score = identify_language(text)
                    if lang != "en" or lang_score < 0.75:
                        continue

                    nsfw_label, nsfw_score = classify_nsfw(text)
                    if nsfw_label != "non-nsfw" or nsfw_score < 0.98:
                        continue

                    toxic_label, toxic_score = classify_toxic_speech(text)
                    if toxic_label != "non-toxic" or toxic_score < 0.98:
                        continue

                    if not gopher_quality_filter(text):
                        continue

                    line = "__label__positive " + ' '.join(text.split())
                    out_f.write(line + "\n")
                    good_count += 1

process_all_warcs("/data/c-mrohatgi/web_quality/warc", "positive_examples.txt")
