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
