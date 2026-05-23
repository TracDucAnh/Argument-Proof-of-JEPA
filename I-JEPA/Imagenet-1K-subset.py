"""
Imagenet-1K-subset.py  —  ImageNet-1K resized via evanarlian/imagenet_1k_resized_256
--------------------------------------------------------------------------------------
- Stream 100K ảnh từ split "test" của evanarlian/imagenet_1k_resized_256
  (ảnh gốc ImageNet-1K, cạnh ngắn resize về 256, đúng pipeline chuẩn I-JEPA)
- RandomCrop 224×224 khi lưu
- Shuffle seed 42, chia 90/10 train/val
- Lưu JPEG vào data/train/ và data/val/ (cùng cấp với file này)
  Tên file: 000000.jpg … 099999.jpg  (id toàn cục trước khi split)

Cấu trúc kết quả:
  I-JEPA/
  ├── Imagenet-1K-subset.py
  └── data/
      ├── train/   (90 000 ảnh, 224×224)
      └── val/     (10 000 ảnh, 224×224)
"""

import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
SEED        = 42
TRAIN_RATIO = 0.9
TOTAL       = 100_000
DATA_DIR    = Path(__file__).parent / "data"
HF_REPO     = "evanarlian/imagenet_1k_resized_256"
HF_SPLIT    = "test"          # test split = đúng 100K ảnh
CROP_SIZE   = 224
JPEG_Q      = 95              # quality JPEG khi lưu
# ────────────────────────────────────────────────────────────────────────────

crop = transforms.RandomCrop(CROP_SIZE)


def to_224(img: Image.Image) -> Image.Image:
    """Đảm bảo RGB + RandomCrop 224×224."""
    img = img.convert("RGB")
    # Nếu ảnh nhỏ hơn 224 ở chiều nào đó thì pad trước
    w, h = img.size
    if w < CROP_SIZE or h < CROP_SIZE:
        img = transforms.Resize(CROP_SIZE)(img)
    return crop(img)


def main():
    print("=" * 60)
    print("  ImageNet-1K resized-256 → I-JEPA image dataset (224×224)")
    print(f"  Source  : {HF_REPO}  [{HF_SPLIT}]")
    print(f"  Output  : {DATA_DIR.resolve()}")
    print("=" * 60)

    # 1. Stream từ HuggingFace
    print(f"\n[1/4] Connecting to {HF_REPO} (streaming)...")
    ds = load_dataset(HF_REPO, split=HF_SPLIT, streaming=True)

    # 2. Collect 100K ảnh, crop 224 on-the-fly
    print(f"[2/4] Downloading & cropping {TOTAL:,} images (224×224)...")
    records: list[Image.Image] = []
    with tqdm(total=TOTAL, desc="  Fetching", unit=" img",
              dynamic_ncols=True, colour="cyan") as pbar:
        for sample in ds:
            img = sample["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            records.append(to_224(img))
            pbar.update(1)
            if len(records) >= TOTAL:
                break

    # 3. Shuffle & split
    print(f"\n[3/4] Shuffling & splitting (seed={SEED})...")
    indices = list(range(len(records)))
    random.seed(SEED)
    random.shuffle(indices)

    cut       = int(len(indices) * TRAIN_RATIO)
    train_idx = indices[:cut]
    val_idx   = indices[cut:]
    print(f"  Train: {len(train_idx):,}  |  Val: {len(val_idx):,}")

    # 4. Lưu ảnh
    (DATA_DIR / "train").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "val").mkdir(parents=True, exist_ok=True)

    print(f"\n[4/4] Saving images...")
    for idx in tqdm(train_idx, desc="  Saving train", unit=" img",
                    dynamic_ncols=True, colour="green"):
        records[idx].save(DATA_DIR / "train" / f"{idx:06d}.jpg",
                          quality=JPEG_Q)

    for idx in tqdm(val_idx, desc="  Saving val  ", unit=" img",
                    dynamic_ncols=True, colour="yellow"):
        records[idx].save(DATA_DIR / "val" / f"{idx:06d}.jpg",
                          quality=JPEG_Q)

    print("\n" + "=" * 60)
    print("  Done!")
    print(f"  data/train/ : {len(train_idx):,} images  (224×224 JPEG)")
    print(f"  data/val/   : {len(val_idx):,} images  (224×224 JPEG)")
    print("=" * 60)


if __name__ == "__main__":
    main()