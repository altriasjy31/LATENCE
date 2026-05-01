import os
import sys
import concurrent.futures as cf
from tqdm import tqdm

root = "/home/dataset-assist-0/datafile/shaojiangyi/latence-dataset/sprot_2204_MSA"
out_file = "sprot_2204_a3m_counts.tsv"
err_file = "sprot_2204_a3m_counts.err"

def count_headers(path):
    try:
        n = 0
        with open(path, "rb") as f:
            first = f.read(1)
            if first == b">":
                n += 1
            prev = first
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                data = prev + chunk
                n += data.count(b"\n>")
                prev = chunk[-1:]
        return path, n, max(n - 1, 0), None
    except Exception as e:
        return path, None, None, repr(e)

paths = [
    os.path.join(r, f)
    for r, _, fs in os.walk(root)
    for f in fs
    if f.endswith(".a3m")
]

with open(out_file, "w", encoding="utf-8") as fout, \
     open(err_file, "w", encoding="utf-8") as ferr:

    fout.write("path\tcounts (queryl)\tcounts\n")
    ferr.write("path\terror\n")

    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        for i, result in tqdm(enumerate(ex.map(count_headers, paths, chunksize=200), 1),total=len(paths)):
            path, total, retrieved, err = result

            if err is None:
                fout.write(f"{path}\t{total}\t{retrieved}\n")
            else:
                ferr.write(f"{path}\t{err}\n")

            # if i % 10000 == 0:
            #     print(f"Processed {i}/{len(paths)}", file=sys.stderr)