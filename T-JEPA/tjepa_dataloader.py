# tjepa_dataloader.py
# ─────────────────────────────────────────────────────────────────────────────
# Dataset + DataLoader + JEPASpanMaskCollator cho Text-JEPA.
# Dữ liệu: data/train.json và data/val.json sinh bởi C4-subset.py
# Format mỗi record: {"id": int, "text": str, "url": str, "timestamp": str}
#
# Output mỗi batch (dict):
#   masked_input_ids       [B, L]   — câu với span
#   masked_attention_mask  [B, L]
#   masked_token_type_ids  [B, L]
#   clean_input_ids        [B, L]   — câu gốc
#   clean_attention_mask   [B, L]
#   clean_token_type_ids   [B, L]
#   span_mask              [B, L]   — 1 tại vị trí span bị mask (để mean-pool)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import random
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

logger = getLogger(__name__)

# ── Hằng số mặc định ─────────────────────────────────────────────────────────
MAX_LENGTH      = 256
BERT_MODEL_NAME = "bert-base-uncased"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class C4TextDataset(Dataset):
    """
    Dataset đọc từ data/train.json hoặc data/val.json.

    Mỗi __getitem__ trả về dict đã tokenize (clean — chưa mask):
        input_ids       LongTensor [max_length]
        attention_mask  LongTensor [max_length]
        token_type_ids  LongTensor [max_length]

    Masking sẽ được thực hiện trong collator để đảm bảo
    mỗi batch lấy span ngẫu nhiên khác nhau.

    Parameters
    ----------
    data_dir : str | Path
        Thư mục chứa train.json / val.json (mặc định: data/ cùng cấp file này).
    split : 'train' | 'val'
    tokenizer : BertTokenizerFast
    max_length : int
    """

    VALID_SPLITS = ("train", "val")

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        tokenizer: BertTokenizerFast,
        max_length: int = MAX_LENGTH,
    ):
        assert split in self.VALID_SPLITS, f"split phải là một trong {self.VALID_SPLITS}"

        self.tokenizer  = tokenizer
        self.max_length = max_length

        json_path = Path(data_dir) / f"{split}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {json_path}\n"
                "Hãy chạy C4-subset.py trước để tải dữ liệu."
            )

        logger.info(f"[C4TextDataset] Loading {json_path} ...")
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        self.texts = [r["text"] for r in records]
        logger.info(f"[C4TextDataset] {split}: {len(self.texts):,} records  ({json_path})")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=True,
        )
        # squeeze batch dim added by return_tensors="pt"
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # [L]
            "attention_mask": encoding["attention_mask"].squeeze(0),  # [L]
            "token_type_ids": encoding["token_type_ids"].squeeze(0),  # [L]
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  JEPASpanMaskCollator
# ══════════════════════════════════════════════════════════════════════════════

class JEPASpanMaskCollator:
    """
    Collator tạo multi-span mask cho Text-JEPA.

    Với mỗi câu trong batch, mỗi lần collate:
      1. Sample số span  n ~ Uniform[1, max_num_spans]  (random per sample).
      2. Với mỗi span: sample length ~ Uniform[1, max_span_length] và vị trí
         start ngẫu nhiên trong phần token thực còn chưa bị mask.
      3. Các span KHÔNG overlap nhau (greedy non-overlap).
      4. Thay tất cả token trong các span bằng [MASK].
      5. span_mask = union của tất cả span → dùng cho mean-pooling.

    Tính random (per sample, per epoch):
      - Số span   ~ Uniform[1, max_num_spans]
      - Mỗi span length ~ Uniform[1, max_span_length]
      - Mỗi span start  ~ Uniform[valid_positions chưa bị chiếm]
      → Mỗi lần collate (kể cả cùng câu) cho kết quả masking khác nhau hoàn toàn.
      → Seed=None (default) để đảm bảo true randomness mỗi epoch.

    Parameters
    ----------
    max_span_length : int
        Độ dài tối đa của mỗi span (default 5).
    max_num_spans : int
        Số span tối đa per sample (default 3).
    seed : int | None
        None = fully random (khuyến nghị cho training).
        Set số cụ thể chỉ khi cần reproducible (smoke test).
    """

    def __init__(
        self,
        max_span_length: int = 5,
        max_num_spans: int = 5,
        min_num_spans: int = 5,
        mask_token_id: int = 103,   # bert-base-uncased [MASK]
        sep_token_id: int = 102,    # [SEP]
        cls_token_id: int = 101,    # [CLS]
        pad_token_id: int = 0,      # [PAD]
        seed: int | None = None,
    ):
        self.max_span_length = max_span_length
        self.max_num_spans   = max_num_spans
        self.min_num_spans   = min_num_spans
        self.mask_token_id   = mask_token_id
        self.sep_token_id    = sep_token_id
        self.cls_token_id    = cls_token_id
        self.pad_token_id    = pad_token_id
        # seed=None → random.Random() dùng system entropy → khác nhau mỗi epoch
        self.rng = random.Random(seed)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get_valid_positions(
        self,
        input_ids: torch.Tensor,       # [L]
        attention_mask: torch.Tensor,  # [L]
    ) -> list[int]:
        """Trả về list vị trí token thực (bỏ CLS, SEP, PAD)."""
        special_ids = {self.cls_token_id, self.sep_token_id, self.pad_token_id}
        return [
            i for i in range(input_ids.size(0))
            if input_ids[i].item() not in special_ids
            and attention_mask[i].item() == 1
        ]

    def _sample_spans(
        self,
        input_ids: torch.Tensor,       # [L]
        attention_mask: torch.Tensor,  # [L]
    ) -> list[tuple[int, int]]:
        """
        Sample n spans không overlap, n ~ Uniform[1, max_num_spans].
        Trả về list[(start, end)] — inclusive, trong token space.
        """
        available = self._get_valid_positions(input_ids, attention_mask)
        if not available:
            return [(1, 1)]   # fallback câu rỗng

        # Sample số span
        n_spans = self.rng.randint(self.min_num_spans, self.max_num_spans)

        spans = []
        masked_positions: set[int] = set()

        for _ in range(n_spans):
            # Vị trí còn chưa bị mask
            free = [p for p in available if p not in masked_positions]
            if len(free) < 1:
                break   # không còn chỗ trống

            # Sample span length
            span_len = self.rng.randint(1, min(self.max_span_length, len(free)))

            # Chỉ giữ lại các start sao cho đủ `span_len` vị trí liên tiếp
            # (liên tiếp trong token space, không phải trong free list)
            valid_starts = [
                p for p in free
                if all((p + k) in set(available) and (p + k) not in masked_positions
                       for k in range(span_len))
            ]
            if not valid_starts:
                break

            start = self.rng.choice(valid_starts)
            end   = start + span_len - 1
            spans.append((start, end))
            masked_positions.update(range(start, end + 1))

        return spans if spans else [(available[0], available[0])]

    # ── __call__ ──────────────────────────────────────────────────────────────

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        batch : list of dicts từ C4TextDataset.__getitem__
                mỗi dict có input_ids, attention_mask, token_type_ids  [L]

        Returns dict[str, Tensor] với tất cả keys cần cho TextJEPA.forward().
        Mỗi lần gọi → masking hoàn toàn khác (số span + vị trí đều random).
        """
        clean_input_ids      = torch.stack([s["input_ids"]      for s in batch])  # [B, L]
        clean_attention_mask = torch.stack([s["attention_mask"]  for s in batch])  # [B, L]
        clean_token_type_ids = torch.stack([s["token_type_ids"]  for s in batch])  # [B, L]

        B = len(batch)
        masked_input_ids = clean_input_ids.clone()
        span_masks       = torch.zeros(B, clean_input_ids.size(1), dtype=torch.long)

        for i in range(B):
            spans = self._sample_spans(clean_input_ids[i], clean_attention_mask[i])
            for start, end in spans:
                masked_input_ids[i, start : end + 1] = self.mask_token_id
                span_masks[i, start : end + 1] = 1

        return {
            # ── masked (context encoder input) ───────────────────────────────
            "masked_input_ids":       masked_input_ids,      # [B, L]
            "masked_attention_mask":  clean_attention_mask,  # padding không đổi
            "masked_token_type_ids":  clean_token_type_ids,

            # ── clean (target encoder input) ─────────────────────────────────
            "clean_input_ids":        clean_input_ids,       # [B, L]
            "clean_attention_mask":   clean_attention_mask,
            "clean_token_type_ids":   clean_token_type_ids,

            # ── span mask (union tất cả spans, dùng cho mean-pooling) ─────────
            "span_mask":              span_masks,            # [B, L] LongTensor
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Factory — make_c4_dataloader
# ══════════════════════════════════════════════════════════════════════════════

def make_c4_dataloader(
    # ── data ──────────────────────────────────────────────────────────────
    data_dir: str | Path | None = None,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_mem: bool = True,
    # ── tokenizer ─────────────────────────────────────────────────────────
    bert_model: str = BERT_MODEL_NAME,
    max_length: int = MAX_LENGTH,
    # ── masking ───────────────────────────────────────────────────────────
    max_span_length: int = 5,
    max_num_spans: int = 5,
    min_num_spans: int = 5,
    seed: int | None = None,   # None = fully random mỗi lần collate (recommended)
    # ── misc ──────────────────────────────────────────────────────────────
    drop_last: bool = True,
    persistent_workers: bool = True,
) -> tuple[DataLoader, JEPASpanMaskCollator]:
    """
    Factory function trả về (DataLoader, collator).

    DataLoader yield dict với 7 keys sẵn sàng cho TextJEPA.forward().

    Parameters mirror config của tjepa-training.py:
        data.max_length, optim.batch_size, optim.num_workers,
        max_span_length, data.seed
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    data_dir = Path(data_dir)

    tokenizer = BertTokenizerFast.from_pretrained(bert_model)

    dataset = C4TextDataset(
        data_dir=data_dir,
        split=split,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    collator = JEPASpanMaskCollator(
        max_span_length=max_span_length,
        max_num_spans=max_num_spans,
        min_num_spans=min_num_spans,
        mask_token_id=tokenizer.mask_token_id,
        sep_token_id=tokenizer.sep_token_id,
        cls_token_id=tokenizer.cls_token_id,
        pad_token_id=tokenizer.pad_token_id,
        seed=seed,
    )

    g = torch.Generator()
    g.manual_seed(seed)

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
        f"[make_c4_dataloader] split={split}  bs={batch_size}  "
        f"workers={num_workers}  max_span={max_span_length}  max_len={max_length}"
    )
    return loader, collator


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Smoke test
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Smoke-test: load batch từ train và val, kiểm tra:
      - Shape tất cả 7 keys đúng [B, L]
      - span_mask có ít nhất 1 token per sample, không vượt max tổng
      - masked_input_ids == MASK_ID tại đúng span_mask positions
      - Ngoài span: masked == clean
      - attention_mask / token_type_ids bất biến
      - Tính random: 2 lần collate cùng batch → masking KHÁC nhau

    Chạy từ thư mục T-JEPA/:
        python tjepa_dataloader.py
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    BATCH_SIZE      = 4
    NUM_WORKERS     = 0
    MAX_SPAN_LENGTH = 5
    MAX_NUM_SPANS   = 5
    MIN_NUM_SPANS   = 5
    MASK_TOKEN_ID   = 103
    # Dùng seed cố định cho smoke test để reproducible
    SMOKE_SEED      = 99

    print("=" * 65)
    print("  T-JEPA DataLoader  —  Smoke Test  (multi-span)")
    print("=" * 65)

    for split in ("train", "val"):
        print(f"\n── {split.upper()} split ─────────────────────────────────────")

        loader, collator = make_c4_dataloader(
            split=split,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_mem=False,
            max_length=MAX_LENGTH,
            max_span_length=MAX_SPAN_LENGTH,
            max_num_spans=MAX_NUM_SPANS,
            min_num_spans=MIN_NUM_SPANS,
            seed=SMOKE_SEED,
            drop_last=False,
            persistent_workers=False,
        )

        # Lấy raw samples để test randomness
        raw_samples = [loader.dataset[i] for i in range(BATCH_SIZE)]

        batch = next(iter(loader))

        # ── 1. Kiểm tra keys ──────────────────────────────────────────────────
        expected_keys = {
            "masked_input_ids", "masked_attention_mask", "masked_token_type_ids",
            "clean_input_ids",  "clean_attention_mask",  "clean_token_type_ids",
            "span_mask",
        }
        assert expected_keys == set(batch.keys())
        print(f"  ✓ Tất cả 7 keys đúng")

        # ── 2. Shape ──────────────────────────────────────────────────────────
        for key, tensor in batch.items():
            assert tensor.shape == (BATCH_SIZE, MAX_LENGTH), \
                f"{key}: got {tuple(tensor.shape)}"
        print(f"  ✓ Shape tất cả tensors: ({BATCH_SIZE}, {MAX_LENGTH})")

        # ── 3. Multi-span: đếm số token bị mask per sample ───────────────────
        span_counts = batch["span_mask"].sum(dim=1)   # [B] số token bị mask
        max_possible = MAX_SPAN_LENGTH * MAX_NUM_SPANS
        assert (span_counts >= 1).all(), f"Có sample không bị mask: {span_counts.tolist()}"
        assert (span_counts <= max_possible).all(), \
            f"Số token mask vượt giới hạn {max_possible}: {span_counts.tolist()}"
        print(f"  ✓ Số token bị mask per sample: {span_counts.tolist()}")
        print(f"    (min=1, max≤{max_possible} = {MAX_NUM_SPANS} spans × {MAX_SPAN_LENGTH} tokens)")

        # ── 4. MASK_ID đúng vị trí ────────────────────────────────────────────
        for i in range(BATCH_SIZE):
            span_pos = batch["span_mask"][i].bool()
            assert (batch["masked_input_ids"][i][span_pos] == MASK_TOKEN_ID).all(), \
                f"Sample {i}: không phải [MASK] tại span!"
            assert (batch["clean_input_ids"][i][span_pos] != MASK_TOKEN_ID).all(), \
                f"Sample {i}: clean có [MASK] tại span?"
        print(f"  ✓ [MASK] token đúng tại mọi span position")

        # ── 5. Ngoài span: masked == clean ───────────────────────────────────
        non_span = ~batch["span_mask"].bool()
        assert (batch["masked_input_ids"][non_span] ==
                batch["clean_input_ids"][non_span]).all()
        print(f"  ✓ Ngoài span: masked == clean")

        # ── 6. attention_mask & token_type_ids bất biến ───────────────────────
        assert (batch["masked_attention_mask"] == batch["clean_attention_mask"]).all()
        assert (batch["masked_token_type_ids"] == batch["clean_token_type_ids"]).all()
        print(f"  ✓ attention_mask và token_type_ids không đổi")

        # ── 7. Tính random: collate cùng samples 2 lần → khác nhau ──────────
        # Dùng collator với seed=None để test true randomness
        random_collator = JEPASpanMaskCollator(
            max_span_length=MAX_SPAN_LENGTH,
            max_num_spans=MAX_NUM_SPANS,
            min_num_spans=MIN_NUM_SPANS,
            seed=None,   # fully random
        )
        batch_a = random_collator(raw_samples)
        batch_b = random_collator(raw_samples)
        masks_differ = not (batch_a["span_mask"] == batch_b["span_mask"]).all().item()
        print(f"  ✓ Randomness: 2 lần collate cùng samples → "
              f"{'KHÁC nhau ✓' if masks_differ else 'Giống nhau (hiếm, thử lại)'}")

        # ── 8. Preview sample 0 ───────────────────────────────────────────────
        span_positions = batch["span_mask"][0].nonzero(as_tuple=True)[0].tolist()
        # Tách thành các span liên tục
        spans_preview = []
        if span_positions:
            cur_start = span_positions[0]
            cur_end   = span_positions[0]
            for p in span_positions[1:]:
                if p == cur_end + 1:
                    cur_end = p
                else:
                    spans_preview.append((cur_start, cur_end))
                    cur_start = cur_end = p
            spans_preview.append((cur_start, cur_end))

        print(f"\n  [Sample 0 preview]")
        print(f"  Số span thực tế : {len(spans_preview)}")
        for k, (s, e) in enumerate(spans_preview):
            print(f"    span {k+1}: positions [{s}:{e}]  "
                  f"({e - s + 1} tokens)  "
                  f"ids={batch['clean_input_ids'][0, s:e+1].tolist()}")

        print(f"\n  ✓ {split} OK  ({len(loader.dataset):,} records, "
              f"{len(loader):,} batches)")

    print("\n" + "=" * 65)
    print("  All smoke tests passed!")
    print("=" * 65)


if __name__ == "__main__":
    main()