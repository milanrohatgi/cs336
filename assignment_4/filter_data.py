#!/usr/bin/env python3

import argparse
import gzip
from pathlib import Path
import submitit
import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
import gzip
import re
import os
from fastwarc.warc import ArchiveIterator
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
import re


lang_model     = fasttext.load_model("/data/classifiers/lid.176.bin")
#quality_model  = fasttext.load_model("/home/c-mrohatgi/.../quality_classifier.bin")
quality_model = fasttext.load_model("/home/c-mrohatgi/assignment4-data/cs336_data/quality_classifier.bin")
def classify_quality(text: str) -> tuple[str, float]:
    sanitized = text.replace("\n", " ")
    labels, probs = quality_model.predict(sanitized, k=1)
    top_label, top_prob = labels[0], probs[0]

    if top_label == "__label__positive":
        return "wiki", top_prob
    else:
        return "cc", top_prob

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

import re

def gopher_quality_filter(text: str) -> bool:
    words = text.split()
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

    num_hashes = text.count("#")
    num_ellipses = text.count("…")
    if (num_hashes / num_words > 0.1) or (num_ellipses / num_words > 0.1):
        return False

    bullet_starts = ("- ", "* ", "• ")
    num_bullet_lines = sum(1 for line in lines if line.lstrip().startswith(bullet_starts))
    if len(lines) > 0 and (num_bullet_lines / len(lines)) > 0.9:
        return False

    required_words = {"the", "be", "to", "of", "and", "that", "have", "with"}
    found = 0
    lowered_text = text.lower()
    for word in required_words:
        if f" {word} " in lowered_text or lowered_text.startswith(word + " ") or lowered_text.endswith(" " + word):
            found += 1
        if found >= 2:
            break
    if found < 2:
        return False

    return True

def type_token_ratio(text: str) -> float:
    tokens = word_tokenize(text)
    num_tokens = len(tokens)
    if num_tokens == 0:
        return 0.0
    return len(set(tokens)) / num_tokens

def process_single_wet_file(
    wet_path,
    output_dir,
    lang_thresh: float = 0.9,
    nsfw_thresh: float = 0.9,
    tox_thresh: float = 0.9,
    quality_thresh: float = 0.95
):
    
    from pathlib import Path
    wet_path   = Path(wet_path)
    output_dir = Path(output_dir)
    """Process one .warc.wet.gz file; return a dict of counts."""
    cnt = {
        'total_records': 0,
        'http_responses': 0,
        'extracted':     0,
        'lang_ok':       0,
        'nsfw_ok':       0,
        'tox_ok':        0,
        'gopher_ok':     0,
        'quality_ok':    0,
    }

    out_file = output_dir / f"{wet_path.stem}.filtered.txt"
    print(out_file)
    with gzip.open(wet_path, 'rb') as stream, \
         open(out_file, 'w', encoding='utf-8') as fout:

        for record in ArchiveIterator(stream):
            cnt['total_records'] += 1
            cnt['http_responses'] += 1

            html = record.reader.read()
            text = extract_text_from_html_bytes(html)
            if not text.strip():
                continue
            cnt['extracted'] += 1

            lang, score = identify_language(text)
            if lang != 'en' or score < lang_thresh:
                continue
            cnt['lang_ok'] += 1

            if not gopher_quality_filter(text):
                continue
            cnt['gopher_ok'] += 1
            
            label, qual_score = classify_quality(text)

            MIN_TTR = 0.2
            ttr_score = type_token_ratio(text)
            if ttr_score < MIN_TTR:
                continue
            
            if label != 'wiki' or qual_score < quality_thresh:
                continue
            cnt['quality_ok'] += 1
            
            fout.write(text.replace('\n', ' ') + "\n")
    
    print(cnt)
    return cnt

