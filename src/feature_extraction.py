"""압력 텐서 → 수치 피처 추출"""
import numpy as np


def extract_footstep_features(pressure: np.ndarray) -> dict | None:
    """
    단일 발걸음 (101, 75, 40) → 피처 딕셔너리
    유효하지 않은 발걸음이면 None 반환
    """
    frame_sums = pressure.sum(axis=(1, 2))
    contact = frame_sums > 100
    if contact.sum() < 5:
        return None

    # 최대 압력 맵
    peak_map = pressure.max(axis=0)
    h = peak_map.shape[0]

    heel     = peak_map[:int(h * 0.25)]
    midfoot  = peak_map[int(h * 0.25):int(h * 0.60)]
    forefoot = peak_map[int(h * 0.60):int(h * 0.90)]

    # Cavanagh Arch Index
    active = peak_map > 5
    total = active.sum()
    mid_area = active[int(h * 0.25):int(h * 0.60)].sum()
    arch_index = float(mid_area / total) if total > 0 else 0.5

    # CoP 궤적
    cop_xs = []
    for frame in pressure:
        s = frame.sum()
        if s > 100:
            _, xs = np.indices(frame.shape)
            cop_xs.append((xs * frame).sum() / s)

    return {
        "peak_heel":     float(heel.max()),
        "peak_midfoot":  float(midfoot.max()),
        "peak_forefoot": float(forefoot.max()),
        "contact_area":  float(total),
        "arch_index":    arch_index,
        "cop_std_x":     float(np.std(cop_xs)) if len(cop_xs) > 1 else 0.0,
    }


def aggregate_features(feats_list: list[dict], suffix: str) -> dict:
    """여러 발걸음 피처 평균 + suffix 추가"""
    if not feats_list:
        return {f"{k}_{suffix}": 0.0
                for k in ["peak_heel", "peak_midfoot", "peak_forefoot",
                          "contact_area", "arch_index", "cop_std_x"]}
    keys = feats_list[0].keys()
    return {f"{k}_{suffix}": float(np.mean([f[k] for f in feats_list]))
            for k in keys}