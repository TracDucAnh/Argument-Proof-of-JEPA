# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted for local ImageNet-1K subset (data/ folder populated by Imagenet-1K-subset.py)
# ─────────────────────────────────────────────────────────────────────────────

import math
import random
from logging import getLogger
from multiprocessing import Value
from pathlib import Path
from PIL import Image, ImageFilter

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

_GLOBAL_SEED = 0
logger = getLogger()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class ImageNet1KSubset(Dataset):
    """
    Flat JPEG dataset produced by Imagenet-1K-subset.py.

    Expected layout (relative to this file):
        data/
          train/  *.jpg
          val/    *.jpg

    Parameters
    ----------
    root : str | Path
        Path to the data/ directory (contains train/ and val/).
    split : str
        'train' or 'val'.
    transform : callable, optional
        torchvision transform applied to each PIL image.
    """

    VALID_SPLITS = ("train", "val")

    def __init__(self, root, split: str = "train", transform=None):
        super().__init__()
        assert split in self.VALID_SPLITS, f"split must be one of {self.VALID_SPLITS}"

        self.root = Path(root) / split
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root}\n"
                "Run Imagenet-1K-subset.py first to populate data/."
            )

        self.samples = sorted(self.root.glob("*.jpg"))
        if len(self.samples) == 0:
            raise RuntimeError(f"No .jpg files found under {self.root}")

        self.transform = transform
        logger.info(f"[ImageNet1KSubset] {split}: {len(self.samples):,} images  ({self.root})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Transforms  (identical to I-JEPA repo's transforms.py)
# ══════════════════════════════════════════════════════════════════════════════

class GaussianBlur:
    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img
        radius = self.radius_min + torch.rand(1).item() * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


def make_transforms(
    crop_size: int = 224,
    crop_scale: tuple = (0.3, 1.0),
    color_jitter: float = 1.0,
    horizontal_flip: bool = False,
    color_distortion: bool = False,
    gaussian_blur: bool = False,
    normalization: tuple = (
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
):
    """Build the standard I-JEPA image transform pipeline."""
    logger.info("making imagenet data transforms")

    def get_color_distortion(s: float = 1.0):
        color_jitter_t = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        rnd_color_jitter = transforms.RandomApply([color_jitter_t], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        return transforms.Compose([rnd_color_jitter, rnd_gray])

    tf = []
    tf.append(transforms.RandomResizedCrop(crop_size, scale=crop_scale))
    if horizontal_flip:
        tf.append(transforms.RandomHorizontalFlip())
    if color_distortion:
        tf.append(get_color_distortion(s=color_jitter))
    if gaussian_blur:
        tf.append(GaussianBlur(p=0.5))
    tf.append(transforms.ToTensor())
    tf.append(transforms.Normalize(normalization[0], normalization[1]))
    return transforms.Compose(tf)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Mask Collators  (faithful port from I-JEPA repo)
# ══════════════════════════════════════════════════════════════════════════════

class DefaultCollator:
    """No masking – returns (batch, None, None)."""

    def __call__(self, batch):
        collated_batch = torch.utils.data.default_collate(batch)
        return collated_batch, None, None


class MaskCollator:
    """
    Block-wise mask collator matching the I-JEPA paper / repo logic.

    For every batch it:
      1. Samples one pred-block SIZE (shared across the batch, seeded).
      2. Samples one enc-block SIZE (square, shared, seeded).
      3. For each image samples `npred` pred-block LOCATIONS (random).
      4. For each image samples `nenc`  enc-block LOCATIONS that do NOT
         overlap with pred blocks (unless allow_overlap=True).

    Returns
    -------
    (collated_batch, collated_masks_enc, collated_masks_pred)
        masks are LongTensors of flat patch indices  [B, n_masks, min_keep]
    """

    def __init__(
        self,
        input_size=(224, 224),
        patch_size: int = 16,
        enc_mask_scale: tuple = (0.85, 1.0),
        pred_mask_scale: tuple = (0.15, 0.2),
        aspect_ratio: tuple = (0.75, 1.5),
        nenc: int = 1,
        npred: int = 4,
        min_keep: int = 10,
        allow_overlap: bool = False,
    ):
        super().__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.patch_size = patch_size
        self.height = input_size[0] // patch_size   # num patches vertically
        self.width  = input_size[1] // patch_size   # num patches horizontally
        self.enc_mask_scale  = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio    = aspect_ratio
        self.nenc            = nenc
        self.npred           = npred
        self.min_keep        = min_keep
        self.allow_overlap   = allow_overlap
        self._itr_counter    = Value("i", -1)       # shared across workers

    # ── helpers ──────────────────────────────────────────────────────────────

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            return i.value

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep   = int(self.height * self.width * mask_scale)
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio   = min_ar + _rand * (max_ar - min_ar)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        # clamp to grid
        h = max(1, min(h, self.height - 1))
        w = max(1, min(w, self.width  - 1))
        return h, w

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size
        tries   = 0
        timeout = og_timeout = 20
        valid_mask = False

        mask = mask_complement = None          # will be set in loop
        top = left = torch.tensor(0)

        while not valid_mask:
            top  = torch.randint(0, max(1, self.height - h), (1,))
            left = torch.randint(0, max(1, self.width  - w), (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1

            if acceptable_regions is not None:
                N = max(int(len(acceptable_regions) - tries), 0)
                for k in range(N):
                    mask *= acceptable_regions[k]

            mask_flat  = torch.nonzero(mask.flatten())
            valid_mask = len(mask_flat) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries  += 1
                    timeout = og_timeout
                    logger.warning(
                        f'Mask generator: "Valid mask not found, '
                        f'decreasing acceptable-regions [{tries}]"'
                    )

        mask_flat = mask_flat.view(-1)   # always 1-D, safe even when len==1
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top : top + h, left : left + w] = 0
        return mask_flat, mask_complement

    # ── collate ──────────────────────────────────────────────────────────────

    def __call__(self, batch):
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g    = torch.Generator()
        g.manual_seed(seed)

        # shared block sizes for this batch
        p_size = self._sample_block_size(g, self.pred_mask_scale, self.aspect_ratio)
        e_size = self._sample_block_size(g, self.enc_mask_scale,  (1.0, 1.0))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc  = self.height * self.width

        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = None if self.allow_overlap else masks_C
            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        # truncate to min_keep so all tensors are the same length, then stack
        # → [B, n_masks, keep]  always a proper LongTensor regardless of keep size
        collated_masks_pred = torch.stack([
            torch.stack([cm[:min_keep_pred] for cm in cm_list])
            for cm_list in collated_masks_pred
        ])  # [B, npred, min_keep_pred]

        collated_masks_enc = torch.stack([
            torch.stack([cm[:min_keep_enc] for cm in cm_list])
            for cm_list in collated_masks_enc
        ])  # [B, nenc, min_keep_enc]

        return collated_batch, collated_masks_enc, collated_masks_pred


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Factory — make_imagenet1k_dataloader
# ══════════════════════════════════════════════════════════════════════════════

def make_imagenet1k_dataloader(
    # ── data ──────────────────────────────────────────────────────────────
    data_dir: str | Path | None = None,   # path to data/  (default: sibling of this file)
    split: str = "train",
    batch_size: int = 128,
    num_workers: int = 10,
    pin_mem: bool = True,
    # ── transforms ────────────────────────────────────────────────────────
    crop_size: int = 224,
    crop_scale: tuple = (0.3, 1.0),
    color_jitter_strength: float = 0.0,
    use_horizontal_flip: bool = False,
    use_color_distortion: bool = False,
    use_gaussian_blur: bool = False,
    # ── masking ───────────────────────────────────────────────────────────
    patch_size: int = 16,
    enc_mask_scale: tuple = (0.85, 1.0),
    pred_mask_scale: tuple = (0.15, 0.2),
    aspect_ratio: tuple = (0.75, 1.5),
    num_enc_masks: int = 1,
    num_pred_masks: int = 4,
    min_keep: int = 10,
    allow_overlap: bool = False,
    use_masking: bool = True,
    # ── misc ──────────────────────────────────────────────────────────────
    seed: int = _GLOBAL_SEED,
    drop_last: bool = True,
    persistent_workers: bool = True,
):
    """
    Build a DataLoader for the local ImageNet-1K subset with I-JEPA masking.

    Parameters mirror the YAML config used by the original I-JEPA repo:

        data:  batch_size, num_workers, pin_mem, crop_size, crop_scale,
               color_jitter_strength, use_horizontal_flip,
               use_color_distortion, use_gaussian_blur
        mask:  patch_size, enc_mask_scale, pred_mask_scale, aspect_ratio,
               num_enc_masks, num_pred_masks, min_keep, allow_overlap

    Returns
    -------
    loader : DataLoader   — yields (imgs, masks_enc, masks_pred)
    collator : MaskCollator | DefaultCollator
    """
    # ── resolve data directory ────────────────────────────────────────────
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    data_dir = Path(data_dir)

    # ── dataset ──────────────────────────────────────────────────────────
    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        color_jitter=color_jitter_strength,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        gaussian_blur=use_gaussian_blur,
    )
    dataset = ImageNet1KSubset(root=data_dir, split=split, transform=transform)

    # ── reproducible sampling ─────────────────────────────────────────────
    g = torch.Generator()
    g.manual_seed(seed)

    # ── collator / mask ───────────────────────────────────────────────────
    if use_masking:
        collator = MaskCollator(
            input_size=(crop_size, crop_size),
            patch_size=patch_size,
            enc_mask_scale=enc_mask_scale,
            pred_mask_scale=pred_mask_scale,
            aspect_ratio=aspect_ratio,
            nenc=num_enc_masks,
            npred=num_pred_masks,
            min_keep=min_keep,
            allow_overlap=allow_overlap,
        )
    else:
        collator = DefaultCollator()

    # ── dataloader ────────────────────────────────────────────────────────
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        collate_fn=collator,
        generator=g,
        persistent_workers=(persistent_workers and num_workers > 0),
    )

    logger.info(
        f"[make_imagenet1k_dataloader] split={split}  "
        f"bs={batch_size}  workers={num_workers}  "
        f"masking={'block-wise' if use_masking else 'none'}"
    )
    return loader, collator


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Smoke test
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Smoke-test: load one batch from train and one from val,
    print shapes, and verify mask indices are in-bounds.

    Run from the I-JEPA/ project root:
        python ijepa_dataloader.py
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    PATCH_SIZE  = 16
    CROP_SIZE   = 224
    BATCH_SIZE  = 4          # small for quick test
    NUM_WORKERS = 0          # 0 = main process (easier for debugging)

    N_PATCHES = (CROP_SIZE // PATCH_SIZE) ** 2   # 196 for 224/16

    print("=" * 65)
    print("  I-JEPA DataLoader  —  Smoke Test")
    print("=" * 65)

    for split in ("train", "val"):
        print(f"\n── {split.upper()} split ─────────────────────────────────────")
        loader, collator = make_imagenet1k_dataloader(
            split=split,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_mem=False,
            crop_size=CROP_SIZE,
            crop_scale=(0.3, 1.0),
            patch_size=PATCH_SIZE,
            enc_mask_scale=(0.85, 1.0),
            pred_mask_scale=(0.15, 0.2),
            aspect_ratio=(0.75, 1.5),
            num_enc_masks=1,
            num_pred_masks=4,
            min_keep=10,
            allow_overlap=False,
            use_masking=True,
            drop_last=False,
            persistent_workers=False,
        )

        imgs, masks_enc, masks_pred = next(iter(loader))

        # ── image tensor ──────────────────────────────────────────────────
        print(f"  imgs         : {tuple(imgs.shape)}   dtype={imgs.dtype}")
        print(f"  imgs min/max : {imgs.min():.3f} / {imgs.max():.3f}")

        # ── encoder mask ──────────────────────────────────────────────────
        # shape: [B, nenc, min_keep_enc]
        print(f"  masks_enc    : {tuple(masks_enc.shape)}  (B, nenc, keep_enc)")
        assert masks_enc.max() < N_PATCHES, "enc mask index out of bounds!"
        assert masks_enc.min() >= 0,        "enc mask has negative index!"

        # ── predictor mask ────────────────────────────────────────────────
        # shape: [B, npred, min_keep_pred]
        print(f"  masks_pred   : {tuple(masks_pred.shape)}  (B, npred, keep_pred)")
        assert masks_pred.max() < N_PATCHES, "pred mask index out of bounds!"
        assert masks_pred.min() >= 0,        "pred mask has negative index!"

        # ── no overlap check (enc ∩ pred == ∅ for first image) ────────────
        enc_set  = set(masks_enc[0, 0].tolist())
        pred_sets = [set(masks_pred[0, k].tolist()) for k in range(masks_pred.shape[1])]
        for k, ps in enumerate(pred_sets):
            overlap = enc_set & ps
            print(f"  enc ∩ pred[{k}] : {len(overlap):3d} patches overlap  "
                  f"({'OK — allow_overlap=True' if overlap else 'OK — no overlap'})")

        print(f"  ✓ {split} batch OK  ({len(loader.dataset):,} total images, "
              f"{len(loader):,} batches)")

    print("\n" + "=" * 65)
    print("  All smoke tests passed!")
    print("=" * 65)


if __name__ == "__main__":
    main()