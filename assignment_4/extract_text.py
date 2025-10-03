import gzip
import random
from fastwarc.warc import ArchiveIterator
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    try:
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding)
    except:
        html_str = ""
    return extract_plain_text(html_str)

if __name__ == "__main__":
    sample_uri = None
    sample_wet_text = None 
    with gzip.open("/data/CC/example.warc.wet.gz", "rt", encoding="utf-8") as f:
        record_lines = []
        headers = {}
        in_content = False

        for line in f:
            if line.startswith("WARC/1.0"):
                if sample_uri is None and record_lines and random.random() < 0.002:
                    sample_wet_text = "".join(record_lines).strip()
                    sample_uri = headers.get("WARC-Target-URI")
                    if sample_uri:
                        break
                record_lines = []
                headers = {}
                in_content = False
            elif line.strip() == "":
                in_content = True
            elif in_content:
                record_lines.append(line)
            elif ":" in line:
                key, val = line.split(":", 1)
                headers[key.strip()] = val.strip()

    sample_html_text = None
    if sample_uri:
        with gzip.open("/data/CC/example.warc.gz", "rb") as stream:
            for record in ArchiveIterator(stream):
                if record.headers.get("WARC-Type") == "response" and record.headers.get("WARC-Target-URI") == sample_uri:
                    html_bytes = record.reader.read()
                    sample_html_text = extract_text_from_html_bytes(html_bytes)
                    break

    print("======================================================")
    print("Text from WET file")
    print("======================================================")
    print(sample_wet_text)
    print("======================================================")
    print("Text from extracted WARC file")
    print("======================================================")
    print(sample_html_text)
