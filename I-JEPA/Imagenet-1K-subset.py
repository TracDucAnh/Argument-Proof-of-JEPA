"""
Imagenet-1K-subset.py  (thực tế dùng STL-10 unlabeled — 100K ảnh 96×96)
------------------------------------------------------------------------
1. Tải STL-10 binary từ Stanford nếu chưa có
2. Đọc toàn bộ 100K ảnh unlabeled
3. Shuffle với seed 42, chia 90/10 train/val
4. Lưu ảnh PNG vào:
       data/train/   (90 000 ảnh)
       data/val/     (10 000 ảnh)
   Tên file: 000000.png … 099999.png  (id toàn cục)

Cấu trúc thư mục kết quả:
   I-JEPA/
   ├── Imagenet-1K-subset.py
   └── data/
       ├── stl10_binary/          ← binary gốc (giữ lại để không tải lại)
       ├── train/
       │   └── xxxxxx.png
       └── val/
           └── xxxxxx.png
"""

import os
import random
import struct
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 42
TRAIN_RATIO = 0.9
DATA_URL    = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
DATA_DIR    = Path(__file__).parent / "data"
BIN_DIR     = DATA_DIR / "stl10_binary"
UNLABELED_BIN = BIN_DIR / "unlabeled_X.bin"   # 100K unlabeled images
HEIGHT, WIDTH, DEPTH = 96, 96, 3
# ─────────────────────────────────────────────────────────────────────────────


# ── 1. Download & extract ─────────────────────────────────────────────────────
def download_and_extract():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = DATA_DIR / "stl10_binary.tar.gz"

    if UNLABELED_BIN.exists():
        print(f"[1/4] Binary already exists, skipping download.")
        return

    if not tar_path.exists():
        print(f"[1/4] Downloading STL-10 (~2.6 GB)...")
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            pct = min(downloaded / total_size * 100, 100)
            bar = int(pct // 2)
            print(
                f"\r  [{'█' * bar:<50}] {pct:5.1f}%  "
                f"({downloaded/1e6:.0f}/{total_size/1e6:.0f} MB)",
                end="", flush=True,
            )
        urllib.request.urlretrieve(DATA_URL, tar_path, reporthook=_progress)
        print()  # newline sau progress bar
        print(f"  Saved → {tar_path}")
    else:
        print(f"[1/4] Tar already downloaded, skipping.")

    print(f"  Extracting...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(DATA_DIR)
    print(f"  Extracted → {BIN_DIR}")


# ── 2. Read unlabeled binary ──────────────────────────────────────────────────
def read_unlabeled_images() -> np.ndarray:
    """
    STL-10 binary format:
      - mỗi ảnh = DEPTH * HEIGHT * WIDTH bytes (uint8, channel-first, column-major)
      - unlabeled_X.bin chứa 100 000 ảnh liên tiếp
    """
    print(f"[2/4] Reading unlabeled_X.bin ...")
    with open(UNLABELED_BIN, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    # reshape → (N, 3, 96, 96)  rồi transpose → (N, 96, 96, 3) HWC
    n = data.shape[0] // (DEPTH * HEIGHT * WIDTH)
    images = data.reshape(n, DEPTH, HEIGHT, WIDTH)
    images = np.transpose(images, (0, 3, 2, 1))   # (N, W, H, C) → đúng HWC
    # STL-10 lưu column-major (width trước height), transpose lại cho đúng
    images = np.transpose(images, (0, 2, 1, 3))   # swap H và W
    print(f"  Loaded {n:,} images  shape={images.shape}")
    return images


# ── 3. Shuffle & split ────────────────────────────────────────────────────────
def split_indices(n: int):
    indices = list(range(n))
    random.seed(SEED)
    random.shuffle(indices)
    cut = int(n * TRAIN_RATIO)
    return indices[:cut], indices[cut:]


# ── 4. Save images ────────────────────────────────────────────────────────────
def save_split(images: np.ndarray, indices: list, split: str):
    out_dir = DATA_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    desc = f"[{'3' if split == 'train' else '4'}/4] Saving {split:5s}"
    for idx in tqdm(indices, desc=f"  {desc}", unit=" img", dynamic_ncols=True, colour="green"):
        img_arr = images[idx]                      # (96, 96, 3) uint8
        img = Image.fromarray(img_arr, mode="RGB")
        img.save(out_dir / f"{idx:06d}.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  STL-10 Unlabeled → I-JEPA image dataset")
    print(f"  Output : {DATA_DIR.resolve()}")
    print("=" * 60 + "\n")

    download_and_extract()
    images = read_unlabeled_images()

    n = len(images)
    train_idx, val_idx = split_indices(n)
    print(f"\n  Seed={SEED} | Train: {len(train_idx):,}  Val: {len(val_idx):,}\n")

    save_split(images, train_idx, "train")
    save_split(images, val_idx,   "val")

    print("\n" + "=" * 60)
    print("  Done!")
    print(f"  data/train/ : {len(train_idx):,} images")
    print(f"  data/val/   : {len(val_idx):,} images")
    print("=" * 60)


if __name__ == "__main__":
    main()