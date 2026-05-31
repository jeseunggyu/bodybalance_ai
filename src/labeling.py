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


# ══════════════════════════════════════════════════════════
# 직접 측정 기반 임상 비대칭 평가
# ══════════════════════════════════════════════════════════
# [중요] 본 데이터의 동적 예측기는 정적 발형태 → 동적 보행지표를
# 예측하지만 검증 결과 예측력이 매우 낮아(평균회귀) 입력이 달라져도
# 출력이 평균에 수렴, 사실상 항상 '정상'으로 표시된다. 또한 하지 길이
# 차이는 학습 피처에 포함되지 않아 분석에 전혀 반영되지 않는다.
# 따라서 좌우가 '실제로 다른' 직접 측정값을 문헌 기준으로 직접 평가한다.

# 하지 길이 차이(LLD) — Gurney (2002), Knutson (2005)
LLD_NORMAL_MM   = 6.0    # 미만이면 기능적으로 무의미 (Gurney 2002)
LLD_MILD_MM     = 10.0   # 6~10mm 경도
LLD_MODERATE_MM = 20.0   # 10~20mm 중등도, 초과 시 현저 (Knutson 2005)

# 좌우 발 형태 비대칭 — 정상인의 좌우 차이는 대부분 1~4mm
# (Atamturk 2009; Krishan 2013 족부 계측). 아래는 '주의' 임계값.
FOOT_LENGTH_DIFF_MM = 8.0   # ≈ 신발 0.5문수(8.46mm)
FOOT_WIDTH_DIFF_MM  = 6.0
ARCH_HEIGHT_DIFF_MM = 6.0   # 좌우 아치 높이 차 → 편측 회내 시사
HEEL_WIDTH_DIFF_MM  = 5.0

SURVEY_DIRECTIONAL_MIN = 2   # 같은 방향 2개 이상 → 비대칭 시사

_SEVERITY_NAME = {0: "정상", 1: "경도", 2: "중등도", 3: "현저"}


def _grade(value, mild, moderate, marked):
    """절대값을 4등급으로: 0 정상 / 1 경도 / 2 중등도 / 3 현저"""
    v = abs(value)
    if v >= marked:   return 3
    if v >= moderate: return 2
    if v >= mild:     return 1
    return 0


def clinical_assessment(user_input: dict) -> dict:
    """
    직접 측정값만으로 구조적 비대칭을 평가한다 (ML 비의존).

    반환: {severity, severity_name, findings[], leg_length_evaluated}
    """
    findings = []

    def _add(item, raw_diff, mild, moderate, marked, unit, note):
        grade = _grade(raw_diff, mild, moderate, marked)
        if grade == 0:
            return
        side = "좌측" if raw_diff < 0 else "우측"
        findings.append({
            "item": item, "side": side,
            "value": f"{abs(raw_diff):.1f}{unit}",
            "grade": grade, "grade_name": _SEVERITY_NAME[grade],
            "note": note,
        })

    # 1) 하지 길이 차이 — 가장 직접적인 구조적 비대칭 (입력된 경우만)
    leg_known = bool(user_input.get("leg_length_known", False))
    if leg_known:
        diff = user_input.get("leg_length_diff_mm", 0) or 0
        _add("하지 길이 차이", diff,
             LLD_NORMAL_MM, LLD_MILD_MM, LLD_MODERATE_MM, "mm",
             "구조적 하지 길이 차이 (Gurney 2002 / Knutson 2005)")

    # 2) 좌우 발 형태 비대칭 (직접 측정)
    def _pair(lk, rk):
        l, r = user_input.get(lk), user_input.get(rk)
        if l is None or r is None:
            return None
        return l - r

    for name, lk, rk, thr in [
        ("발 길이 비대칭",   "foot_length_L_mm", "foot_length_R_mm", FOOT_LENGTH_DIFF_MM),
        ("발 너비 비대칭",   "foot_width_L_mm",  "foot_width_R_mm",  FOOT_WIDTH_DIFF_MM),
        ("아치 높이 비대칭", "arch_height_L_mm", "arch_height_R_mm", ARCH_HEIGHT_DIFF_MM),
        ("뒤꿈치 폭 비대칭", "heel_width_L_mm",  "heel_width_R_mm",  HEEL_WIDTH_DIFF_MM),
    ]:
        d = _pair(lk, rk)
        if d is None:
            continue
        _add(name, d, thr, thr * 1.5, thr * 2.0, "mm",
             f"좌우 차이 임계값 {thr:.0f}mm (족부 계측 정상 범위 초과)")

    # 3) 설문 방향성 — 좌/우 자각 증상이 한쪽으로 일관되면 비대칭 시사
    side_keys = ["survey_shoe_wear", "survey_fatigue_side",
                 "survey_posture_tilt", "survey_shoulder_drop"]
    sides = [user_input.get(k) for k in side_keys]
    n_left  = sum(1 for s in sides if s == -1)
    n_right = sum(1 for s in sides if s == 1)
    dominant = max(n_left, n_right)
    if dominant >= SURVEY_DIRECTIONAL_MIN:
        side = "좌측" if n_left >= n_right else "우측"
        grade = 2 if dominant >= 3 else 1
        findings.append({
            "item": "자가진단 방향성", "side": side,
            "value": f"{dominant}개 항목",
            "grade": grade, "grade_name": _SEVERITY_NAME[grade],
            "note": "여러 자각 증상이 한쪽으로 일관됨",
        })

    severity = max((f["grade"] for f in findings), default=0)
    # 동일 최고등급 소견이 2개 이상이면 누적 부담으로 한 단계 상향
    if severity > 0:
        n_top = sum(1 for f in findings if f["grade"] == severity)
        if n_top >= 2 and severity < 3:
            severity += 1

    return {
        "severity": severity,
        "severity_name": _SEVERITY_NAME[severity],
        "findings": sorted(findings, key=lambda f: -f["grade"]),
        "leg_length_evaluated": leg_known,
    }


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
