"""
비대칭 유형 라벨링 (문헌 근거 기반)

근거:
- Cavanagh & Rodgers (1987): Arch Index 기반 발 분류
- Robinson et al. (1987): Symmetry Index 10% 임계값
- Menz (1998): 좌우 발 형태 비대칭의 임상적 정의
"""
import pandas as pd


# ── 문헌 기반 임계값 ────────────────────────────────────
# Cavanagh & Rodgers (1987), J. Biomechanics 20(5)
ARCH_INDEX_LOW  = 0.21   # 높은 아치 (high arch)
ARCH_INDEX_HIGH = 0.26   # 평발 (flat foot)

# Robinson et al. (1987), Journal of Manipulative Physiological Therapeutics
# 좌우 대칭성 지수 임계값 (clinically meaningful asymmetry)
SI_THRESHOLD_NORMAL = 10.0   # % — 10% 이상이면 비대칭

# Menz (1998), Foot 8(4)
# 발 형태 좌우 차이의 임상적 임계값
ARCH_INDEX_DIFF_THRESHOLD = 0.03


def label_asymmetry(row) -> int:
    """
    네 가지 유형으로 분류:
    0 = Symmetric Normal (대칭 정상)
    1 = Asymmetric Loading (좌우 압력 비대칭) — Robinson SI > 10%
    2 = Bilateral Flat Foot (양측 평발) — Cavanagh AI > 0.26
    3 = Asymmetric Arch (좌우 아치 비대칭) — Menz Δ > 0.03
    """
    grf_si     = row["grf_asymmetry_pct"]              # Robinson SI
    avg_arch   = (row["arch_index_L"] + row["arch_index_R"]) / 2
    arch_diff  = abs(row["arch_index_L"] - row["arch_index_R"])

    # 정상 (모든 기준 통과)
    if (grf_si < SI_THRESHOLD_NORMAL
        and arch_diff < ARCH_INDEX_DIFF_THRESHOLD
        and ARCH_INDEX_LOW <= avg_arch <= ARCH_INDEX_HIGH):
        return 0

    # 좌우 아치 비대칭 (Menz)
    if arch_diff >= ARCH_INDEX_DIFF_THRESHOLD:
        return 3

    # 양측 평발 (Cavanagh)
    if avg_arch > ARCH_INDEX_HIGH:
        return 2

    # 압력 비대칭 (Robinson)
    if grf_si >= SI_THRESHOLD_NORMAL:
        return 1

    return 0


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["asym_type"] = df.apply(label_asymmetry, axis=1)
    return df


# 외부 노출용
ASYM_TYPE_NAMES = {
    0: "대칭 정상 (Symmetric Normal)",
    1: "좌우 압력 비대칭 (Asymmetric Loading)",
    2: "양측 평발 (Bilateral Flat Foot)",
    3: "좌우 아치 비대칭 (Asymmetric Arch)",
}

REFERENCES = {
    "cavanagh_1987": (
        "Cavanagh PR, Rodgers MM. The arch index: a useful measure from "
        "footprints. Journal of Biomechanics. 1987;20(5):547-551."
    ),
    "robinson_1987": (
        "Robinson RO, Herzog W, Nigg BM. Use of force platform variables to "
        "quantify the effects of chiropractic manipulation on gait symmetry. "
        "J Manipulative Physiol Ther. 1987;10(4):172-176."
    ),
    "menz_1998": (
        "Menz HB. Alternative techniques for the clinical assessment of foot "
        "pronation. Foot. 1998;8(4):207-211."
    ),
}