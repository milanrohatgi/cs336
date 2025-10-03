import gzip
import re
import os
from fastwarc.warc import ArchiveIterator
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
from nltk.tokenize import word_tokenize
import random

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


warc_path = "/data/CC/example.warc.gz"
output_path = "negative_examples.txt"
target_count = 20000
sample_prob = 1 

output_file = output_path
count = 0

with gzip.open(warc_path, "rb") as stream, open(output_file, "w", encoding="utf-8") as out_f:
    for record in ArchiveIterator(stream):
        if record.headers.get("WARC-Type") != "response":
            continue
        if random.random() > sample_prob:
            continue

        html_bytes = record.reader.read()
        text = extract_text_from_html_bytes(html_bytes).strip()
        if not text:
            continue

        try:
            lang, lang_score = identify_language(text)
            nsfw_label, nsfw_score = classify_nsfw(text)
            toxic_label, toxic_score = classify_toxic_speech(text)
            quality = gopher_quality_filter(text)
        except Exception:
            continue 

        if (
            lang != "en" or lang_score < 0.8 or
            nsfw_label == "nsfw" and nsfw_score > 0.98 or
            toxic_label == "toxic" and toxic_score > 0.98 or
            not quality
        ):
            out_f.write("__label__negative " + text.replace("\n", " ").strip() + "\n")
            count += 1

        if count >= target_count:
            print(f'count: {count}, target_count: {target_count}')
            break

print(f"Saved {count} negative examples to {output_path}")
