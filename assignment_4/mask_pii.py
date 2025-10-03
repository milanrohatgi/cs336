import gzip
import random
from fastwarc import ArchiveIterator
import fasttext
from cs336_data.extract_text import extract_plain_text, detect_encoding
import re

def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    try:
        html_str = html_bytes.decode('utf-8')
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding)
    return extract_plain_text(html_str)

def mask_emails(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    masked_text, count = re.subn(email_pattern, '|||EMAIL_ADDRESS|||', text)
    return masked_text, count

def mask_phone_numbers(text):
    phone_pattern = re.compile(
        r"""
        (?<!\d)                 
        (?:\+?1[\s.-]?)?          
        (?:\(?\d{3}\)?[\s.-]?)    
        \d{3}[\s.-]?\d{4}      
        (?!\d)
        """, re.VERBOSE
    )
    masked_text, count = re.subn(phone_pattern, '|||PHONE_NUMBER|||', text)
    return masked_text, count

def mask_ips(text):
    ip_pattern = r'''
        \b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}
        (?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b
    '''
    masked_text, count = re.subn(re.compile(ip_pattern, re.VERBOSE), '|||IP_ADDRESS|||', text)
    return masked_text, count

def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    try:
        html_str = html_bytes.decode('utf-8')
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding)
    return extract_plain_text(html_str)

def run_pii_masking_on_random_warc(warc_path: str, target_samples: int = 20, sample_prob: float = 0.01):
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

            original_text = text

            email_masked, email_count = mask_emails(text)
            phone_masked, phone_count = mask_phone_numbers(email_masked)
            final_masked, ip_count = mask_ips(phone_masked)

            total_replacements = email_count + phone_count + ip_count
            if total_replacements == 0:
                continue

            for pii_type, pattern in [
                ("EMAIL_ADDRESS", r'\|\|\|EMAIL_ADDRESS\|\|\|'),
                ("PHONE_NUMBER", r'\|\|\|PHONE_NUMBER\|\|\|'),
                ("IP_ADDRESS", r'\|\|\|IP_ADDRESS\|\|\|')
            ]:
                for match in re.finditer(pattern, final_masked):
                    start = max(0, match.start() - 100)
                    end = match.end() + 100
                    snippet = final_masked[start:end].replace('\n', ' ')
                    print(f"Example {samples_collected + 1}: {pii_type}")
                    print(f"Masked Snippet: {snippet}")

                    # Show the original text that was replaced, if possible
                    original_snippet = original_text[start:end].replace('\n', ' ')
                    print(f"Original Text: {original_snippet}")
                    print("=" * 100)

                    samples_collected += 1
                    if samples_collected >= target_samples:
                        return

if __name__ == "__main__":
    run_pii_masking_on_random_warc("/data/CC/example.warc.gz")