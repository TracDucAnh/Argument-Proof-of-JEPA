# find_instability_point.py
# ─────────────────────────────────────────────────────────────────────────────
# Tìm các đỉnh loss (spikes) của T-JEPA để xác định instability points.
# In ra step, giá trị, và ranking của từng đỉnh.
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path

HERE = Path(__file__).parent.resolve()
TJEPA_JSON = HERE / "T-JEPA.json"
STEP_CAP   = 15_000

# ── Tham số phát hiện đỉnh ────────────────────────────────────────────────────
# Một điểm được coi là đỉnh khi loss của nó lớn hơn
# SPIKE_RATIO lần so với median của toàn bộ data.
SPIKE_RATIO   = 5.0    # lớn hơn 5× median → là spike
MIN_GAP_STEPS = 500    # hai đỉnh phải cách nhau ít nhất N steps
                       # (tránh đếm cùng 1 spike nhiều lần)


def load(path: Path, step_cap: int = STEP_CAP):
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    filtered = [r for r in records if r["global_step"] <= step_cap]
    steps  = [r["global_step"] for r in filtered]
    losses = [r["loss"]        for r in filtered]
    return steps, losses


def find_peaks(steps, losses,
               spike_ratio: float = SPIKE_RATIO,
               min_gap: int = MIN_GAP_STEPS) -> list[dict]:
    """
    Tìm tất cả đỉnh local vượt ngưỡng spike_ratio × median.
    Áp dụng non-maximum suppression với cửa sổ min_gap steps
    để tránh đếm cùng một spike nhiều lần.
    """
    import statistics
    median_loss = statistics.median(losses)
    threshold   = spike_ratio * median_loss

    print(f"  Median loss : {median_loss:.6f}")
    print(f"  Threshold   : {threshold:.6f}  ({spike_ratio}× median)\n")

    # Bước 1: tìm tất cả local maxima vượt ngưỡng
    candidates = []
    n = len(losses)
    for i in range(1, n - 1):
        if losses[i] > threshold:
            if losses[i] >= losses[i - 1] and losses[i] >= losses[i + 1]:
                candidates.append({"step": steps[i], "loss": losses[i], "idx": i})

    if not candidates:
        return []

    # Bước 2: non-maximum suppression — trong mỗi cửa sổ min_gap, giữ đỉnh cao nhất
    candidates.sort(key=lambda x: x["loss"], reverse=True)
    kept = []
    for c in candidates:
        too_close = any(abs(c["step"] - k["step"]) < min_gap for k in kept)
        if not too_close:
            kept.append(c)

    # Sắp xếp theo step để in theo thứ tự thời gian
    kept.sort(key=lambda x: x["step"])
    return kept


def main():
    print(f"Loading: {TJEPA_JSON}\n")
    steps, losses = load(TJEPA_JSON)
    print(f"  Records: {len(steps)} (steps 0–{STEP_CAP:,})\n")

    peaks = find_peaks(steps, losses)

    if not peaks:
        print("Không tìm thấy spike nào vượt ngưỡng.")
        return

    print(f"{'Rank':<6} {'Step':>8} {'Loss':>14}  {'Ghi chú'}")
    print("─" * 50)
    for rank, p in enumerate(peaks, 1):
        note = ""
        if rank == 1:
            note = "← đỉnh cao nhất (spike 1)"
        elif rank == 2:
            note = "← đỉnh cao thứ 2 (instability onset)"
        print(f"{rank:<6} {p['step']:>8,} {p['loss']:>14.4f}  {note}")

    print()
    if len(peaks) >= 2:
        p2 = peaks[1]
        print(f"► Instability point (đỉnh thứ 2): step {p2['step']:,}  |  loss = {p2['loss']:.4f}")
    else:
        p1 = peaks[0]
        print(f"► Chỉ tìm được 1 đỉnh: step {p1['step']:,}  |  loss = {p1['loss']:.4f}")
        print("  (thử giảm SPIKE_RATIO hoặc MIN_GAP_STEPS nếu muốn tìm thêm)")


if __name__ == "__main__":
    main()