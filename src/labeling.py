"""
비대칭 유형 라벨링 — 실제 압력 센서 데이터 기반 보정

Cavanagh & Rodgers (1987)의 Arch Index는 잉크 발도장 기반 측정값이며,
압력 센서 기반 arch_index는 절대값이 다르므로 직접 적용 불가.
대신 좌우 차이(ΔAI)와 GRF Symmetry Index를 주요 판별 기준으로 사용하며,
압력 센서 데이터 분포에 맞게 임계값을 조정한다.

기준 문헌:
- Robinson et al. (1987): GRF Symmetry Index — 원래 기준 10%, 본 연구는 5%로 강화
- Menz (1998): 좌우 발 형태 비대칭 — 원래 기준 ΔAI 0.03, 본 연구는 0.01로 강화
- Patterson et al. (2010): CoP 비대칭 — 0.08 초과 시 임상적 유의성
"""
import pandas as pd

# ── 실제 데이터에 맞게 조정된 임계값 ──────────────────
# Robinson (1987) 원본 기준 10% → 압력 센서 데이터 특성상 5%로 강화
SI_NORMAL   = 5.0     # % — 이 미만이면 정상 하중 분배
SI_MILD     = 10.0    # % — 경도 비대칭
SI_SEVERE   = 20.0    # % — 중증 비대칭

# Menz (1998) 원본 기준 ΔAI 0.03 → 압력 센서 특성상 0.01로 강화
AI_DIFF_NORMAL  = 0.010   # 이 미만이면 정상
AI_DIFF_MILD    = 0.020   # 경도 비대칭
AI_DIFF_SEVERE  = 0.040   # 중증 비대칭

# Patterson et al. (2010): CoP 비대칭 임계값
COP_ASYM_THRESHOLD = 0.08

# 압력 센서 기반 arch_index 범위 (데이터 기반, P001~P150 통계)
# 실제 데이터: 평균 약 0.50, 낮은 아치: > 0.54, 높은 아치: < 0.45
AI_FLAT_THRESHOLD     = 0.53   # 이 초과 = 평발 경향
AI_HIGH_ARCH_THRESHOLD = 0.45  # 이 미만 = 높은 아치 경향


def label_asymmetry(row) -> int:
    """
    비대칭 유형 분류

    판단 우선순위:
    1. GRF Symmetry Index (좌우 하중 비대칭이 가장 직접적 지표)
    2. ΔArch Index (좌우 발 형태 차이)
    3. CoP 비대칭 (보행 안정성)
    4. 양측 평발 (절대값 기준)

    반환값:
    0 = 대칭 정상
    1 = 좌우 압력 비대칭 (Asymmetric Loading)
    2 = 양측 아치 이상 (Bilateral Arch Abnormality)
    3 = 좌우 아치 비대칭 (Asymmetric Arch)
    """
    grf_si    = row["grf_asymmetry_pct"]
    ai_diff   = abs(row["arch_index_L"] - row["arch_index_R"])
    cop_asym  = row["cop_asymmetry"]
    avg_ai    = (row["arch_index_L"] + row["arch_index_R"]) / 2

    # 정상: 세 기준 모두 통과해야 함 (AND 조건 — 엄격)
    if (grf_si    < SI_NORMAL
        and ai_diff   < AI_DIFF_NORMAL
        and cop_asym  < COP_ASYM_THRESHOLD):
        return 0

    # Type 1: 좌우 하중 비대칭 — 가장 직접적인 보행 비대칭 지표
    if grf_si >= SI_MILD:
        return 1

    # Type 3: 좌우 아치 비대칭 — 한발만 평발이거나 형태 차이 클 때
    if ai_diff >= AI_DIFF_MILD:
        return 3

    # Type 2: 양측 아치 이상 — 양쪽 다 평발이거나 높은 아치
    if avg_ai > AI_FLAT_THRESHOLD or avg_ai < AI_HIGH_ARCH_THRESHOLD:
        return 2

    # 경계값 처리: 위 기준 미달이지만 정상 기준도 미달인 경우
    # CoP 비대칭이 있으면 Type 1 (경도)
    if cop_asym >= COP_ASYM_THRESHOLD:
        return 1

    # SI가 5~10% 사이면 경도 Type 1
    if grf_si >= SI_NORMAL:
        return 1

    return 0


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["asym_type"] = df.apply(label_asymmetry, axis=1)
    return df


ASYM_TYPE_NAMES = {
    0: "대칭 정상 (Symmetric Normal)",
    1: "좌우 압력 비대칭 (Asymmetric Loading)",
    2: "양측 아치 이상 (Bilateral Arch Abnormality)",
    3: "좌우 아치 비대칭 (Asymmetric Arch)",
}

# 문헌 출처
REFERENCES = {
    "robinson_1987": (
        "Robinson RO, Herzog W, Nigg BM. Use of force platform variables to "
        "quantify the effects of chiropractic manipulation on gait symmetry. "
        "J Manipulative Physiol Ther. 1987;10(4):172-176."
    ),
    "menz_1998": (
        "Menz HB. Alternative techniques for the clinical assessment of foot "
        "pronation. The Foot. 1998;8(4):207-211."
    ),
    "patterson_2010": (
        "Patterson KK, Gage WH, Brooks D, Black SE, McIlroy WE. Evaluation of "
        "gait symmetry after stroke: A comparison of existing methods and "
        "a novel approach. Gait & Posture. 2010;31(2):241-246."
    ),
}
