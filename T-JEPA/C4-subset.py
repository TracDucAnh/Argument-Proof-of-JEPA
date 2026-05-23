"""
C4-subset.py
------------
Tải 100K record đầu tiên từ tập C4 tiếng Anh (allenai/c4),
giữ nguyên độ dài văn bản gốc, chia 90/10 train/val ngẫu nhiên,
lưu thành data/train.json và data/val.json (cùng cấp với file này).

Mỗi record: {"id": <int>, "text": <str>, "url": <str>, "timestamp": <str>}
"""

import json
import os
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
TOTAL_SAMPLES   = 100_000
TRAIN_RATIO     = 0.9
SEED            = 42
OUTPUT_DIR      = Path(__file__).parent / "data"   # T-JEPA/data/
# ─────────────────────────────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  C4 Subset Downloader — {TOTAL_SAMPLES:,} records")
    print(f"  Output : {OUTPUT_DIR.resolve()}")
    print("=" * 60)

    # 1. Stream C4 en train split
    print("\n[1/3] Connecting to C4 dataset (streaming)...")
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    # 2. Collect 100K records với progress bar
    print(f"\n[2/3] Downloading {TOTAL_SAMPLES:,} records...")
    records = []
    with tqdm(
        total=TOTAL_SAMPLES,
        desc="  Fetching",
        unit=" records",
        dynamic_ncols=True,
        colour="cyan",
    ) as pbar:
        for sample in dataset:
            records.append({
                "text":      sample["text"],        # giữ nguyên độ dài gốc
                "url":       sample.get("url", ""),
                "timestamp": sample.get("timestamp", ""),
            })
            pbar.update(1)
            if len(records) >= TOTAL_SAMPLES:
                break

    # 3. Shuffle và split 90/10
    print(f"\n[3/3] Shuffling & splitting (seed={SEED})...")
    random.seed(SEED)
    random.shuffle(records)

    split_idx   = int(len(records) * TRAIN_RATIO)
    train_data  = records[:split_idx]
    val_data    = records[split_idx:]

    # Gán id từ 0
    for idx, rec in enumerate(train_data):
        rec["id"] = idx
    for idx, rec in enumerate(val_data):
        rec["id"] = idx

    # Reorder keys: id first
    def reorder(rec):
        return {"id": rec["id"], "text": rec["text"],
                "url": rec["url"], "timestamp": rec["timestamp"]}

    train_data = [reorder(r) for r in train_data]
    val_data   = [reorder(r) for r in val_data]

    # 4. Ghi file
    train_path = OUTPUT_DIR / "train.json"
    val_path   = OUTPUT_DIR / "val.json"

    print(f"\n  Writing train.json  ({len(train_data):,} records)...")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    print(f"  Writing val.json    ({len(val_data):,} records)...")
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    # 5. Summary
    train_mb = train_path.stat().st_size / 1024 / 1024
    val_mb   = val_path.stat().st_size   / 1024 / 1024

    print("\n" + "=" * 60)
    print("  Done!")
    print(f"  train.json : {len(train_data):>7,} records  ({train_mb:.1f} MB)")
    print(f"  val.json   : {len(val_data):>7,} records  ({val_mb:.1f} MB)")
    print(f"  Location   : {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()