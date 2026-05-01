from pathlib import Path

raw = Path("sprot_2204_anno_a3m_counts.raw")
out = Path("sprot_2204_anno_a3m_counts.tsv")

data = raw.read_bytes()

with out.open("w", encoding="utf-8", errors="replace") as fout:
    i = 0
    while i < len(data):
        j = data.find(b"\0", i)
        if j == -1:
            break

        path = data[i:j]

        k = data.find(b"\n", j + 1)
        if k == -1:
            k = len(data)

        count_bytes = data[j + 1:k].strip()

        try:
            total = int(count_bytes)
        except ValueError:
            print("Bad record:", repr(data[i:k + 1]))
            i = k + 1
            continue

        retrieved = max(total - 1, 0)
        path_str = path.decode("utf-8", errors="replace")
        fout.write(f"{path_str}\t{total}\t{retrieved}\n")

        i = k + 1
        i = k + 1