import os
import concurrent.futures as cf
from tqdm import tqdm

root = "/home/dataset-assist-0/datafile/shaojiangyi/latence-dataset/sprot_2204_anno_MSA"

def count_headers_fast(path):
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
    retrieved = max(n - 1, 0)
    return path, n, retrieved

paths = [
    os.path.join(r, f)
    for r, _, fs in os.walk(root)
    for f in fs
    if f.endswith(".a3m")
]

with cf.ThreadPoolExecutor(max_workers=2) as ex:
    for path, total, retrieved in tqdm(ex.map(count_headers_fast, paths), total=len(paths)):
        print(f"{path}\t{total}\t{retrieved}")