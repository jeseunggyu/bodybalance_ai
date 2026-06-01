"""
파생 피처 생성 — 학습/추론 공용

발 치수 절대값(L/R)만으로는 RandomForest가 '좌우 차이'라는 개념을 명시적으로
학습하지 못해 입력이 달라도 평균값만 출력하는 문제가 있었다. 좌우 차이/비율/
비대칭 지수를 명시적 피처로 추가해 모델이 좌우 불균형에 반응하도록 한다.

학습(DataFrame)과 앱 추론(dict)에서 동일한 정의를 쓰도록 두 진입점을 제공한다.
"""

# 파생 피처 이름 (학습·추론 양쪽에서 이 순서/이름을 공유)
DERIVED_FEATURES = [
    "foot_length_diff",   # L - R (부호 있음: +면 좌발이 큼)
    "foot_width_diff",
    "foot_length_asym",   # |L-R| / (L+R) — 정규화 비대칭 지수
    "foot_width_asym",
    "foot_length_ratio",  # L / R
    "foot_width_ratio",
]


def _derive(fl_L, fl_R, fw_L, fw_R) -> dict:
    """원시 발 치수 4개 → 파생 피처 dict (스칼라 계산)"""
    eps = 1e-6
    return {
        "foot_length_diff":  fl_L - fl_R,
        "foot_width_diff":   fw_L - fw_R,
        "foot_length_asym":  abs(fl_L - fl_R) / (fl_L + fl_R + eps),
        "foot_width_asym":   abs(fw_L - fw_R) / (fw_L + fw_R + eps),
        "foot_length_ratio": fl_L / (fl_R + eps),
        "foot_width_ratio":  fw_L / (fw_R + eps),
    }


def add_derived_features(df):
    """DataFrame에 파생 피처 컬럼 추가 (학습용)"""
    df = df.copy()
    eps = 1e-6
    df["foot_length_diff"]  = df["foot_length_L_mm"] - df["foot_length_R_mm"]
    df["foot_width_diff"]   = df["foot_width_L_mm"]  - df["foot_width_R_mm"]
    df["foot_length_asym"]  = (df["foot_length_L_mm"] - df["foot_length_R_mm"]).abs() \
                              / (df["foot_length_L_mm"] + df["foot_length_R_mm"] + eps)
    df["foot_width_asym"]   = (df["foot_width_L_mm"] - df["foot_width_R_mm"]).abs() \
                              / (df["foot_width_L_mm"] + df["foot_width_R_mm"] + eps)
    df["foot_length_ratio"] = df["foot_length_L_mm"] / (df["foot_length_R_mm"] + eps)
    df["foot_width_ratio"]  = df["foot_width_L_mm"]  / (df["foot_width_R_mm"] + eps)
    return df


def derive_for_input(user_input: dict) -> dict:
    """앱 user_input dict → 파생 피처 dict (추론용). 결측 시 빈 dict."""
    keys = ("foot_length_L_mm", "foot_length_R_mm",
            "foot_width_L_mm", "foot_width_R_mm")
    vals = [user_input.get(k) for k in keys]
    if any(v is None for v in vals):
        return {f: 0.0 for f in DERIVED_FEATURES}
    return _derive(*vals)


# ══════════════════════════════════════════════════════════
# 사용자 직접 입력 피처 (하지 길이 + 설문)
# ══════════════════════════════════════════════════════════
# 분류기의 기존 핵심 입력(grf/cop)은 라벨을 만든 동적지표라, 앱에서는
# predictor 추정값(거의 평균)으로 채워져 항상 정상으로 분류되는 문제가 있었다.
# 하지 길이 차이와 설문을 '직접 입력 피처'로 추가하면 추정값에 의존하지 않고
# 사용자 입력이 비대칭 판단에 직접 반영된다. 미입력 시 0(중립).
USER_DIRECT_FEATURES = [
    "leg_length_diff_mm",     # 하지 길이 차이 (좌-우, mm). 미입력 0
    "survey_shoe_wear",       # -1(좌)/0/+1(우)/2(모름)
    "survey_fatigue_side",    # -1/0/+1/2
    "survey_posture_tilt",    # -1/0/+1/2
    "survey_shoulder_drop",   # -1/0/+1/2
    "survey_pain_flag",       # 통증 부위 호소 여부 0/1 (pain_location>0)
    "survey_direction_score", # 좌/우 쏠림 일관성: 같은 방향 항목 수의 부호합
]

_SURVEY_SIDE_KEYS = ["survey_shoe_wear", "survey_fatigue_side",
                     "survey_posture_tilt", "survey_shoulder_drop"]


def _survey_direction_score(d) -> int:
    """좌/우(-1/+1) 응답의 합 — 한쪽으로 쏠릴수록 절댓값 큼. '모름'(2)/0 무시."""
    score = 0
    for k in _SURVEY_SIDE_KEYS:
        v = d.get(k, 0)
        if v in (-1, 1):
            score += v
    return score


def user_direct_for_input(user_input: dict) -> dict:
    """앱 user_input → 사용자 직접 입력 피처 dict (미입력은 0/중립)."""
    leg = user_input.get("leg_length_diff_mm", 0) if \
        user_input.get("leg_length_known", False) else 0
    pain = user_input.get("survey_pain_location", 0)
    return {
        "leg_length_diff_mm":     leg or 0,
        "survey_shoe_wear":       _norm_side(user_input.get("survey_shoe_wear")),
        "survey_fatigue_side":    _norm_side(user_input.get("survey_fatigue_side")),
        "survey_posture_tilt":    _norm_side(user_input.get("survey_posture_tilt")),
        "survey_shoulder_drop":   _norm_side(user_input.get("survey_shoulder_drop")),
        "survey_pain_flag":       1 if (pain not in (0, 4, None)) else 0,
        "survey_direction_score": _survey_direction_score({
            k: _norm_side(user_input.get(k)) for k in _SURVEY_SIDE_KEYS
        }),
    }


def _norm_side(v):
    """설문 좌/우 값 정규화: -1/0/1은 그대로, '모름'(2)·None은 0(중립)."""
    if v in (-1, 0, 1):
        return v
    return 0


def modulate_dynamic_by_geometry(dynamic: dict, user_input: dict) -> dict:
    """
    ML이 예측한 좌/우 동적지표를 사용자가 실측한 좌우 발 기하 비율로 보정.

    RandomForest는 좌우를 거의 같은 값으로 예측하는 경향이 있어(평균회귀),
    예측값만으로는 좌우 차이가 드러나지 않는다. 사용자가 실제로 측정한
    좌우 발 길이·너비 비율을 ML 추정 부하량에 곱해, 측정된 기하 비대칭이
    부하 추정에 반영되도록 한다. (ML 추정 × 실측 기하 보정)

    좌우 차이가 없으면 비율이 1이라 원본 예측을 그대로 유지한다.
    """
    fl_L = user_input.get("foot_length_L_mm")
    fl_R = user_input.get("foot_length_R_mm")
    fw_L = user_input.get("foot_width_L_mm")
    fw_R = user_input.get("foot_width_R_mm")
    if None in (fl_L, fl_R, fw_L, fw_R):
        return dynamic

    eps = 1e-6
    # 면적성 지표는 길이×너비 비, 압력성 지표는 너비 비로 보정
    area_L = (fl_L * fw_L)
    area_R = (fl_R * fw_R)
    area_ratio_L = area_L / ((area_L + area_R) / 2 + eps)
    area_ratio_R = area_R / ((area_L + area_R) / 2 + eps)
    w_ratio_L = fw_L / ((fw_L + fw_R) / 2 + eps)
    w_ratio_R = fw_R / ((fw_L + fw_R) / 2 + eps)
    l_ratio_L = fl_L / ((fl_L + fl_R) / 2 + eps)
    l_ratio_R = fl_R / ((fl_L + fl_R) / 2 + eps)

    out = dict(dynamic)

    def _scale(key_L, key_R, rL, rR):
        if key_L in out:
            out[key_L] = out[key_L] * rL
        if key_R in out:
            out[key_R] = out[key_R] * rR

    _scale("contact_area_L", "contact_area_R", area_ratio_L, area_ratio_R)
    _scale("peak_heel_L",     "peak_heel_R",     w_ratio_L, w_ratio_R)
    _scale("peak_midfoot_L",  "peak_midfoot_R",  w_ratio_L, w_ratio_R)
    _scale("peak_forefoot_L", "peak_forefoot_R", w_ratio_L, w_ratio_R)
    _scale("arch_index_L",    "arch_index_R",    l_ratio_L, l_ratio_R)

    # 보정된 좌우 부하로 GRF 비대칭 재계산 (입력에 따라 실제로 변함)
    sumL = out.get("peak_heel_L", 0) + out.get("peak_midfoot_L", 0) + out.get("peak_forefoot_L", 0)
    sumR = out.get("peak_heel_R", 0) + out.get("peak_midfoot_R", 0) + out.get("peak_forefoot_R", 0)
    if sumL + sumR > 0:
        out["grf_asymmetry_pct"] = abs(sumL - sumR) / (sumL + sumR) * 100
    # CoP 비대칭도 아치 좌우차 기반으로 살짝 반영
    out["cop_asymmetry"] = abs(out.get("arch_index_L", 0) - out.get("arch_index_R", 0))
    return out


def add_synthetic_user_features(df, seed: int = 42):
    """
    학습 DataFrame에 사용자 직접 입력 피처를 '실측 비대칭에 비례해' 합성.

    공개 데이터셋엔 하지 길이·설문이 없으므로, 실측 grf/cop 비대칭이 큰
    샘플일수록 (1) 하지 길이 차이가 크고 (2) 설문이 한쪽으로 쏠릴 확률이
    높도록 확률적으로 생성한다. 좌우 방향은 좌우 압력 우세(peak 합)로 결정.
    → 비대칭이 실제로 있는 샘플에 일관된 '사용자 신호'가 부여되어,
      분류기가 이 신호에 반응하도록 학습된다.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)

    grf = df["grf_asymmetry_pct"].to_numpy()          # 0~17.6
    cop = df["cop_asymmetry"].to_numpy()              # 0~0.24
    # 비대칭 강도 0~1 정규화 (둘의 평균)
    grf_n = np.clip(grf / 15.0, 0, 1)
    cop_n = np.clip(cop / 0.20, 0, 1)
    intensity = (grf_n + cop_n) / 2

    # 좌우 방향: 좌측 peak 합 > 우측이면 좌측(-1) 우세
    peakL = df[["peak_heel_L", "peak_midfoot_L", "peak_forefoot_L"]].sum(axis=1).to_numpy()
    peakR = df[["peak_heel_R", "peak_midfoot_R", "peak_forefoot_R"]].sum(axis=1).to_numpy()
    side = np.where(peakL >= peakR, -1, 1)            # -1 좌 우세, +1 우 우세

    # (1) 하지 길이 차이: 강도에 비례(최대 ~18mm) + 노이즈, 방향은 side
    leg = side * (intensity * 18.0 + rng.normal(0, 1.5, n))
    leg = np.where(intensity < 0.15, rng.normal(0, 1.0, n), leg)  # 약한 비대칭은 거의 0
    df["leg_length_diff_mm"] = np.round(leg, 1)

    # (2) 설문 4항목: 강도가 높을수록 side 방향 응답 확률↑, 아니면 0/모름
    for k in _SURVEY_SIDE_KEYS:
        p_directional = np.clip(intensity * 0.9, 0, 0.9)  # 쏠림 응답 확률
        draw = rng.random(n)
        val = np.where(draw < p_directional, side, 0)
        # 일부는 '모름'(여기선 0으로 — 정규화 정책과 일치)
        df[k] = val.astype(int)

    # (3) 통증 호소: 강도 비례 확률
    pain_p = np.clip(intensity * 0.7, 0, 0.7)
    df["survey_pain_flag"] = (rng.random(n) < pain_p).astype(int)

    # (4) 방향 일관성 점수
    df["survey_direction_score"] = (
        df["survey_shoe_wear"] + df["survey_fatigue_side"]
        + df["survey_posture_tilt"] + df["survey_shoulder_drop"]
    )
    return df
