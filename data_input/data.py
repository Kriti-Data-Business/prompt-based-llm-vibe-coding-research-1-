"""
build_dataset.py
────────────────
Pulls 150 problems from LiveCodeBench and saves problems_dataset_150.csv.

Key findings from diagnosis:
  - public_test_cases  : JSON string → list of {input, output, testtype}
  - private_test_cases : base64 → zlib (wbits=15) → JSON, but may have
                         encoding issues → decode with latin-1 fallback
  - inputs/outputs     : stdin/stdout strings (multi-line)

Run:
    pip install datasets
    python build_dataset.py
"""

import json, zlib, base64
import pandas as pd
from datasets import load_dataset

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("Loading LiveCodeBench...")
ds = load_dataset("livecodebench/code_generation_lite", split="test")
print(f"Total problems: {len(ds)}")

# ─────────────────────────────────────────────────────────────────────────────
# DECODERS
# ─────────────────────────────────────────────────────────────────────────────
def decode_private(raw):
    """base64 -> zlib decompress -> JSON. Try multiple wbits and encodings."""
    if not raw or not isinstance(raw, str) or not raw.strip():
        return []
    try:
        compressed = base64.b64decode(raw)
        for wbits in (15, -15, 47):
            try:
                decompressed = zlib.decompress(compressed, wbits=wbits)
                for enc in ("utf-8", "latin-1", "utf-8-sig"):
                    try:
                        text = decompressed.decode(enc)
                        return json.loads(text)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
            except zlib.error:
                continue
    except Exception:
        pass
    return []

def parse_public(raw):
    if not raw or not isinstance(raw, str) or not raw.strip():
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []

def format_tests(cases, max_n):
    results = []
    for tc in cases[:max_n]:
        inp = tc.get("input", "").strip()
        out = tc.get("output", "").strip()
        if inp and out:
            results.append({"input": inp, "expected": out})
    return results

def map_difficulty(d):
    d = str(d).lower().strip()
    if d in ("easy",   "1"): return "Easy"
    if d in ("medium", "2"): return "Medium"
    return "Hard"

# ─────────────────────────────────────────────────────────────────────────────
# COLLECT
# ─────────────────────────────────────────────────────────────────────────────
easy, medium, hard = [], [], []
skipped_no_pub = 0
used_pub_as_hidden = 0
used_real_hidden = 0

for item in ds:
    diff = map_difficulty(item.get("difficulty", "hard"))

    if diff == "Easy"   and len(easy)   >= 150: continue
    if diff == "Medium" and len(medium) >= 150: continue
    if diff == "Hard"   and len(hard)   >= 150: continue

    pub_cases  = parse_public(item.get("public_test_cases",  ""))
    priv_cases = decode_private(item.get("private_test_cases", ""))

    pub  = format_tests(pub_cases,  max_n=2)
    priv = format_tests(priv_cases, max_n=5)

    if not pub:
        skipped_no_pub += 1
        continue

    if priv:
        used_real_hidden += 1
        hidden = priv
    else:
        used_pub_as_hidden += 1
        hidden = pub  # fallback: reuse public as hidden

    record = {
        "ProblemName": item.get("question_title", "Unknown"),
        "Difficulty":  diff,
        "Description": item.get("question_content", ""),
        "PublicTests": json.dumps(pub),
        "HiddenTests": json.dumps(hidden),
        "GroundTruth": "",
    }

    if   diff == "Easy":   easy.append(record)
    elif diff == "Medium": medium.append(record)
    else:                  hard.append(record)

    if len(easy) == 150 and len(medium) == 150 and len(hard) == 150:
        break

print(f"\nCollected  : Easy={len(easy)}, Medium={len(medium)}, Hard={len(hard)}")
print(f"Real hidden: {used_real_hidden}  |  Pub-as-hidden fallback: {used_pub_as_hidden}")
print(f"Skipped    : {skipped_no_pub} (no public tests)")

for name, lst in [("Easy", easy), ("Medium", medium), ("Hard", hard)]:
    if len(lst) < 150:
        print(f"  Warning: Only {len(lst)}/50 {name} problems found")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for i, p in enumerate(easy,   1): rows.append({"ProblemID": f"E{i:02d}", **p})
for i, p in enumerate(medium, 1): rows.append({"ProblemID": f"M{i:02d}", **p})
for i, p in enumerate(hard,   1): rows.append({"ProblemID": f"H{i:02d}", **p})

df = pd.DataFrame(rows, columns=["ProblemID","ProblemName","Difficulty",
                                  "Description","PublicTests","HiddenTests","GroundTruth"])
df.to_csv("problems_dataset_150x3.csv", index=False)
print(f"\nSaved problems_dataset_150x3.csv ({len(df)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SANITY CHECK")
print("="*60)
for pid in ["E01", "M01", "H01"]:
    r = df[df["ProblemID"]==pid]
    if r.empty: continue
    r = r.iloc[0]
    pub = json.loads(r["PublicTests"])
    hid = json.loads(r["HiddenTests"])
    print(f"\n{pid} - {r['ProblemName']} ({r['Difficulty']})")
    print(f"  Public  [{len(pub)}] input : {repr(pub[0]['input'][:80])}")
    print(f"  Public  [{len(pub)}] expect: {repr(pub[0]['expected'][:60])}")
    print(f"  Hidden  [{len(hid)}] input : {repr(hid[0]['input'][:80])}")
    print(f"  Hidden  [{len(hid)}] expect: {repr(hid[0]['expected'][:60])}")
