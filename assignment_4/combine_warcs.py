import os
import gzip
from warcio.archiveiterator import ArchiveIterator
from warcio.warcwriter import WARCWriter

input_dir = "/data/c-mrohatgi/web_quality/warc"
output_path = "/data/c-mrohatgi/web_quality/warc/combined.warc.gz"

with gzip.open(output_path, 'wb') as out_stream:
    writer = WARCWriter(out_stream, gzip=True)
    max = 0
    for filename in sorted(os.listdir(input_dir)):
        if max == 64:
            break
        file_path = os.path.join(input_dir, filename)
        with gzip.open(file_path, 'rb') as stream:
            for record in ArchiveIterator(stream):
                try:
                    writer.write_record(record)
                except Exception as e:
                    print(e)
        max += 1
                    
                    
