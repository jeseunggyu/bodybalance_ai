"""공개 데이터셋 전체 → train/test CSV 생성"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.config import (
    RAW_DIR, PROC_DIR, TRAIN_CSV, TEST_CSV,
    N_TRAIN_SUBJECTS, N_TEST_SUBJECTS, SPEEDS,
)
from src.feature_extraction import (
    extract_footstep_features, aggregate_features,
)


def process_trial(npz_file: Path, meta_file: Path) -> dict | None:
    """한 trial (예: P001/BF_W1) → 좌우 평균 피처"""
    if not (npz_file.exists() and meta_file.exists()):
        return None

    try:
        with np.load(npz_file) as d:
            footsteps = d["arr_0"]
        meta = pd.read_csv(meta_file)
    except Exception:
        return None

    # 유효 발걸음만
    valid = (
        (meta["Exclude"]    != 1) &
        (meta["Outlier"]    != 1) &
        (meta["Incomplete"] != 1)
    )
    meta_valid = meta[valid].reset_index()

    left, right = [], []
    L_meta, R_meta = [], []

    for _, row in meta_valid.iterrows():
        idx = int(row["index"])
        if idx >= len(footsteps):
            continue

        feats = extract_footstep_features(footsteps[idx])
        if feats is None:
            continue

        if row["Side"] == "Left":
            left.append(feats)
            L_meta.append(row)
        elif row["Side"] == "Right":
            right.append(feats)
            R_meta.append(row)

    if not left or not right:
        return None

    L_avg = aggregate_features(left,  "L")
    R_avg = aggregate_features(right, "R")

    foot_L = pd.DataFrame(L_meta)["FootLength"].mean()
    foot_R = pd.DataFrame(R_meta)["FootLength"].mean()
    width_L = pd.DataFrame(L_meta)["FootWidth"].mean()
    width_R = pd.DataFrame(R_meta)["FootWidth"].mean()

    # 비대칭 지수
    peak_L = L_avg["peak_heel_L"] + L_avg["peak_midfoot_L"] + L_avg["peak_forefoot_L"]
    peak_R = R_avg["peak_heel_R"] + R_avg["peak_midfoot_R"] + R_avg["peak_forefoot_R"]
    grf_asym = abs(peak_L - peak_R) / (peak_L + peak_R + 1e-6) * 100

    cop_asym = abs(L_avg["cop_std_x_L"] - R_avg["cop_std_x_R"]) / \
               (L_avg["cop_std_x_L"] + R_avg["cop_std_x_R"] + 1e-6)

    return {
        "foot_length_L_mm": float(foot_L),
        "foot_length_R_mm": float(foot_R),
        "foot_width_L_mm":  float(width_L),
        "foot_width_R_mm":  float(width_R),
        **L_avg, **R_avg,
        "grf_asymmetry_pct": float(grf_asym),
        "cop_asymmetry":     float(cop_asym),
    }


def process_subject_all_speeds(subject_dir: Path) -> list[dict]:
    """한 사람의 모든 속도 trial 처리 → 여러 row 반환"""
    rows = []
    for speed in SPEEDS:
        trial_dir = subject_dir / speed
        result = process_trial(
            trial_dir / "pipeline_1.npz",
            trial_dir / "metadata.csv",
        )
        if result:
            result["subject_id"] = subject_dir.name
            result["speed"]      = speed
            rows.append(result)
    return rows


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    train_rows, test_rows = [], []
    subject_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])

    for sd in tqdm(subject_dirs, desc="참가자 처리"):
        # P001 → 1
        try:
            num = int(sd.name.replace("P", ""))
        except ValueError:
            continue

        rows = process_subject_all_speeds(sd)
        if num <= N_TRAIN_SUBJECTS:
            train_rows.extend(rows)
        elif num <= N_TRAIN_SUBJECTS + N_TEST_SUBJECTS:
            test_rows.extend(rows)

    train_df = pd.DataFrame(train_rows)
    test_df  = pd.DataFrame(test_rows)

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f"\n✅ Train: {len(train_df)} 샘플 ({train_df['subject_id'].nunique()}명)")
    print(f"✅ Test:  {len(test_df)} 샘플 ({test_df['subject_id'].nunique()}명)")
    print(f"   저장: {TRAIN_CSV}, {TEST_CSV}")


if __name__ == "__main__":
    main()