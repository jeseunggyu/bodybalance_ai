"""
N=10 입력 샘플 데이터 생성기

앱(web/pages/1_input.py)의 입력 스키마·검증 범위에 맞는, 인체측정학적으로
타당한 피험자 10명(남 7 / 여 3, 대학생 나이대)의 입력 데이터를 생성한다.
산출물은 순수 입력값 CSV(sample_data/sample_inputs.csv) 한 개다.

[설계 근거 — 입력값 타당성]
- 발 길이 ≈ 신장 × 0.15 (인체측정 표준, Anderson 1956)
- 발 너비 ≈ 발 길이 × 0.37~0.40
- 뒤꿈치 폭 ≈ 발 너비 × 0.62~0.66
- 정상인의 좌우 차이는 발 길이 1~4mm, 하지 길이 <6mm (Gurney 2002)
- 검증 범위(앱): 나이 15~80, 신장 140~210, 체중 30~150,
  발길이 180~320, 발너비 60~120, 뒤꿈치 40~80, 아치 0~60,
  발등 40~100, 하지 700~1100mm
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 설문 코드(앱과 동일): 좌측 -1 / 같음 0 / 우측 +1 / 모름 2
#                        통증: 없음 0 / 무릎 1 / 허리 2 / 발목 3 / 모름 4

# ── N=10 피험자 (인체측정학적으로 타당하게 설계) ──────────────
# 대학생 피험자 10명 (남 7 / 여 3, 나이 19~26). 모든 입력은 앱 검증 범위 내이며
# 발 길이≈신장×0.15 등 인체측정 표준에 따라 설계.
SUBJECTS = [
    # 1) 남, 거의 대칭
    dict(id="S01", sex="남성",
         age=22, height_cm=175.0, weight_kg=70.0,
         foot_length_L_mm=262.0, foot_length_R_mm=261.0,
         foot_width_L_mm=99.0,  foot_width_R_mm=98.5,
         heel_width_L_mm=64.0,  heel_width_R_mm=63.5,
         arch_height_L_mm=24.0, arch_height_R_mm=24.0,
         instep_height_L_mm=62.0, instep_height_R_mm=61.5,
         leg_length_known=True, leg_length_L_mm=910.0, leg_length_R_mm=910.0,
         survey_shoe_wear=0, survey_pain_location=0, survey_fatigue_side=0,
         survey_posture_tilt=0, survey_shoulder_drop=0),

    # 2) 여, 거의 대칭
    dict(id="S02", sex="여성",
         age=21, height_cm=162.0, weight_kg=54.0,
         foot_length_L_mm=237.0, foot_length_R_mm=237.5,
         foot_width_L_mm=88.0,  foot_width_R_mm=88.0,
         heel_width_L_mm=56.0,  heel_width_R_mm=56.0,
         arch_height_L_mm=22.0, arch_height_R_mm=22.5,
         instep_height_L_mm=55.0, instep_height_R_mm=55.0,
         leg_length_known=True, leg_length_L_mm=845.0, leg_length_R_mm=843.0,
         survey_shoe_wear=0, survey_pain_location=0, survey_fatigue_side=0,
         survey_posture_tilt=0, survey_shoulder_drop=0),

    # 3) 남, 하지 길이 차이(우 18mm 짧음)
    dict(id="S03", sex="남성",
         age=24, height_cm=178.0, weight_kg=76.0,
         foot_length_L_mm=266.0, foot_length_R_mm=265.0,
         foot_width_L_mm=101.0, foot_width_R_mm=100.0,
         heel_width_L_mm=65.0,  heel_width_R_mm=64.0,
         arch_height_L_mm=25.0, arch_height_R_mm=24.0,
         instep_height_L_mm=63.0, instep_height_R_mm=62.0,
         leg_length_known=True, leg_length_L_mm=925.0, leg_length_R_mm=907.0,
         survey_shoe_wear=1, survey_pain_location=2, survey_fatigue_side=1,
         survey_posture_tilt=1, survey_shoulder_drop=1),

    # 4) 남, 양측 평발(낮은 아치)
    dict(id="S04", sex="남성",
         age=23, height_cm=170.0, weight_kg=78.0,
         foot_length_L_mm=258.0, foot_length_R_mm=258.0,
         foot_width_L_mm=104.0, foot_width_R_mm=103.0,
         heel_width_L_mm=67.0,  heel_width_R_mm=66.0,
         arch_height_L_mm=10.0, arch_height_R_mm=11.0,
         instep_height_L_mm=58.0, instep_height_R_mm=58.0,
         leg_length_known=True, leg_length_L_mm=880.0, leg_length_R_mm=879.0,
         survey_shoe_wear=0, survey_pain_location=3, survey_fatigue_side=0,
         survey_posture_tilt=0, survey_shoulder_drop=0),

    # 5) 여, 좌우 발 형태 비대칭(좌발 큼)
    dict(id="S05", sex="여성",
         age=20, height_cm=168.0, weight_kg=58.0,
         foot_length_L_mm=248.0, foot_length_R_mm=240.0,
         foot_width_L_mm=94.0,  foot_width_R_mm=88.0,
         heel_width_L_mm=60.0,  heel_width_R_mm=56.0,
         arch_height_L_mm=15.0, arch_height_R_mm=26.0,
         instep_height_L_mm=58.0, instep_height_R_mm=55.0,
         leg_length_known=True, leg_length_L_mm=862.0, leg_length_R_mm=858.0,
         survey_shoe_wear=-1, survey_pain_location=1, survey_fatigue_side=-1,
         survey_posture_tilt=-1, survey_shoulder_drop=0),

    # 6) 남, 경도 비대칭(하지길이 미입력)
    dict(id="S06", sex="남성",
         age=26, height_cm=172.0, weight_kg=68.0,
         foot_length_L_mm=257.0, foot_length_R_mm=255.0,
         foot_width_L_mm=96.0,  foot_width_R_mm=94.0,
         heel_width_L_mm=62.0,  heel_width_R_mm=60.0,
         arch_height_L_mm=18.0, arch_height_R_mm=20.0,
         instep_height_L_mm=60.0, instep_height_R_mm=59.0,
         leg_length_known=False, leg_length_L_mm=None, leg_length_R_mm=None,
         survey_shoe_wear=2, survey_pain_location=1, survey_fatigue_side=2,
         survey_posture_tilt=0, survey_shoulder_drop=1),

    # 7) 남, 장신 대칭
    dict(id="S07", sex="남성",
         age=21, height_cm=188.0, weight_kg=84.0,
         foot_length_L_mm=283.0, foot_length_R_mm=283.0,
         foot_width_L_mm=108.0, foot_width_R_mm=108.0,
         heel_width_L_mm=70.0,  heel_width_R_mm=69.5,
         arch_height_L_mm=27.0, arch_height_R_mm=27.0,
         instep_height_L_mm=66.0, instep_height_R_mm=66.0,
         leg_length_known=True, leg_length_L_mm=985.0, leg_length_R_mm=984.0,
         survey_shoe_wear=0, survey_pain_location=0, survey_fatigue_side=0,
         survey_posture_tilt=0, survey_shoulder_drop=0),

    # 8) 남, 자각 우측 쏠림(설문), 하지길이 미입력
    dict(id="S08", sex="남성",
         age=25, height_cm=173.0, weight_kg=72.0,
         foot_length_L_mm=259.0, foot_length_R_mm=258.0,
         foot_width_L_mm=97.0,  foot_width_R_mm=99.0,
         heel_width_L_mm=62.0,  heel_width_R_mm=63.0,
         arch_height_L_mm=23.0, arch_height_R_mm=21.0,
         instep_height_L_mm=60.0, instep_height_R_mm=61.0,
         leg_length_known=False, leg_length_L_mm=None, leg_length_R_mm=None,
         survey_shoe_wear=1, survey_pain_location=1, survey_fatigue_side=1,
         survey_posture_tilt=1, survey_shoulder_drop=1),

    # 9) 여, 약한 비대칭 무증상
    dict(id="S09", sex="여성",
         age=19, height_cm=165.0, weight_kg=55.0,
         foot_length_L_mm=242.0, foot_length_R_mm=241.0,
         foot_width_L_mm=90.0,  foot_width_R_mm=90.0,
         heel_width_L_mm=57.0,  heel_width_R_mm=57.0,
         arch_height_L_mm=25.0, arch_height_R_mm=24.0,
         instep_height_L_mm=56.0, instep_height_R_mm=56.0,
         leg_length_known=True, leg_length_L_mm=858.0, leg_length_R_mm=856.0,
         survey_shoe_wear=0, survey_pain_location=0, survey_fatigue_side=0,
         survey_posture_tilt=0, survey_shoulder_drop=0),

    # 10) 남, 복합 비대칭(하지차+발형태차+설문 좌측 쏠림)
    dict(id="S10", sex="남성",
         age=24, height_cm=171.0, weight_kg=74.0,
         foot_length_L_mm=255.0, foot_length_R_mm=263.0,
         foot_width_L_mm=96.0,  foot_width_R_mm=102.0,
         heel_width_L_mm=61.0,  heel_width_R_mm=66.0,
         arch_height_L_mm=28.0, arch_height_R_mm=14.0,
         instep_height_L_mm=59.0, instep_height_R_mm=63.0,
         leg_length_known=True, leg_length_L_mm=895.0, leg_length_R_mm=916.0,
         survey_shoe_wear=-1, survey_pain_location=2, survey_fatigue_side=-1,
         survey_posture_tilt=-1, survey_shoulder_drop=-1),
]

# CSV 컬럼 정의 (순수 입력값)
FIELDS = [
    ("id", "ID"),
    ("age", "나이"), ("sex", "성별"),
    ("height_cm", "신장(cm)"), ("weight_kg", "체중(kg)"),
    ("foot_length_L_mm", "발길이L(mm)"), ("foot_length_R_mm", "발길이R(mm)"),
    ("foot_width_L_mm", "발너비L(mm)"), ("foot_width_R_mm", "발너비R(mm)"),
    ("heel_width_L_mm", "뒤꿈치L(mm)"), ("heel_width_R_mm", "뒤꿈치R(mm)"),
    ("arch_height_L_mm", "아치L(mm)"), ("arch_height_R_mm", "아치R(mm)"),
    ("instep_height_L_mm", "발등L(mm)"), ("instep_height_R_mm", "발등R(mm)"),
    ("leg_length_L_mm", "하지L(mm)"), ("leg_length_R_mm", "하지R(mm)"),
    ("survey_shoe_wear", "Q1신발마모"), ("survey_pain_location", "Q2통증"),
    ("survey_fatigue_side", "Q3피로"), ("survey_posture_tilt", "Q4자세"),
    ("survey_shoulder_drop", "Q5어깨"),
]


def main():
    out_dir = ROOT / "sample_data"
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "sample_inputs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([h for _, h in FIELDS])
        for s in SUBJECTS:
            w.writerow([s.get(k) if s.get(k) is not None else ""
                        for k, _ in FIELDS])
    print(f"[OK] {csv_path}  (N={len(SUBJECTS)})")


if __name__ == "__main__":
    main()
