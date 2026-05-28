"""프로젝트 전역 설정"""
from pathlib import Path

# ── 경로 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data"
RAW_DIR     = DATA_DIR / "raw" / "public"
PROC_DIR    = DATA_DIR / "processed"
MODELS_DIR  = ROOT / "models"

# 출력 파일
TRAIN_CSV = PROC_DIR / "train_features.csv"
TEST_CSV  = PROC_DIR / "test_features.csv"
PREDICTOR_PKL  = MODELS_DIR / "dynamic_predictor.pkl"
CLASSIFIER_PKL = MODELS_DIR / "asymmetry_classifier.pkl"

# ── 데이터 분할 ───────────────────────────────────────
N_TRAIN_SUBJECTS = 100      # P001 ~ P100
N_TEST_SUBJECTS  = 50       # P101 ~ P150
SPEEDS = ["W1_BF", "W3_BF", "W4_BF"]

# ── 입력 피처 (LiDAR + 설문에서 얻을 수 있는 것) ───────
LIDAR_FEATURES = [
    "foot_length_L_mm",
    "foot_length_R_mm",
    "foot_width_L_mm",
    "foot_width_R_mm",
    "arch_height_L_mm",
    "arch_height_R_mm",
    "heel_width_L_mm",
    "heel_width_R_mm",
    "instep_height_L_mm",
    "instep_height_R_mm",
]

# 설문 점수 (각 문항 -1/0/+1/2 등으로 점수화)
SURVEY_FEATURES = [
    "survey_shoe_wear",        # 신발 마모: -1(좌), 0(같음), +1(우), 2(모름)
    "survey_pain_location",    # 통증 부위: 0(없음), 1(무릎), 2(허리), 3(발목), 4(모름)
    "survey_fatigue_side",     # 피로감: -1(좌), 0(없음), +1(우), 2(모름)
    "survey_posture_tilt",     # 자세 기울기: -1(좌), 0(없음), +1(우), 2(모름)
    "survey_shoulder_drop",    # 어깨 처짐: -1(좌), 0(없음), +1(우), 2(모름)
]

# 기본 정보
BASIC_FEATURES = ["height_cm", "weight_kg", "age"]

# 동적 타겟 (모델이 예측할 것)
DYNAMIC_TARGETS = [
    "peak_heel_L",      "peak_heel_R",
    "peak_midfoot_L",   "peak_midfoot_R",
    "peak_forefoot_L",  "peak_forefoot_R",
    "contact_area_L",   "contact_area_R",
    "arch_index_L",     "arch_index_R",
    "cop_std_x_L",      "cop_std_x_R",
    "cop_asymmetry",
    "grf_asymmetry_pct",
]

# 모델 입력 (정적 피처 모두)
STATIC_INPUT = BASIC_FEATURES + LIDAR_FEATURES + SURVEY_FEATURES

# ── 비대칭 분류 ───────────────────────────────────────
ASYM_LABELS = {
    0: "정상",
    1: "Type A - 전족부 하중형",
    2: "Type B - 아치 붕괴형",
    3: "Type C - 회내·회외형",
}

# ── 임상 가이드라인 (문헌 기반) ───────────────────────
CLINICAL = {
    "leg_length_threshold_mm":  6,        # Gurney 2002
    "leg_lift_ratio":           0.5,
    "max_wedge_angle_deg":      6,        # McPoil & Cornwall 2009
    "normal_asym_threshold":    5,         # %
    "severe_asym_threshold":    20,
}

REFERENCES = {
    "cavanagh_1987": "Cavanagh PR & Rodgers MM (1987). J Biomech, 20(5), 547-551.",
    "gurney_2002":   "Gurney B (2002). Gait & Posture, 15(2), 195-206.",
    "mcpoil_2009":   "McPoil TG & Cornwall MW (2009). JOSPT, 39(7), 507-514.",
    "hsi_2005":      "Hsi WL et al. (2005). Foot Ankle Int, 26(8), 616-620.",
}