"""
교수님 제출용 N=10 입력 샘플 생성기

앱(web/pages/1_input.py)의 입력 스키마·검증 범위에 맞는, 인체측정학적으로
타당한 가상 피험자 10명의 입력 데이터를 생성한다. 다양한 임상 케이스를
포함하도록 설계하고, 실제 분석/추천 엔진을 돌려 결과까지 함께 기록한다.

산출물:
  sample_data/sample_inputs.csv   — 입력 + 분석결과 요약 (스프레드시트용)
  sample_data/sample_inputs.md    — 보기 좋은 표 (문서 첨부용)

[설계 근거 — 입력값 타당성]
- 발 길이 ≈ 신장 × 0.15 (인체측정 표준, Anderson 1956)
- 발 너비 ≈ 발 길이 × 0.37~0.40
- 뒤꿈치 폭 ≈ 발 너비 × 0.62~0.66
- 정상인의 좌우 차이는 발 길이 1~4mm, 하지 길이 <6mm (Gurney 2002)
- 남성이 여성보다 발이 큼. 검증 범위(앱): 나이 15~80, 신장 140~210,
  체중 30~150, 발길이 180~320, 발너비 60~120, 뒤꿈치 40~80,
  아치 0~60, 발등 40~100, 하지 700~1100mm
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import (
    derive_for_input, user_direct_for_input, modulate_dynamic_by_geometry,
)
from src.labeling import clinical_assessment
from src.recommend import recommend
import joblib

# 설문 코드(앱과 동일): 좌측 -1 / 같음 0 / 우측 +1 / 모름 2
#                        통증: 없음 0 / 무릎 1 / 허리 2 / 발목 3 / 모름 4

# ── N=10 가상 피험자 (인체측정학적으로 타당하게 설계) ──────────────
# 각 dict는 앱 user_input 스키마와 동일.
SUBJECTS = [
    # 1) 표준 남성, 거의 대칭 — 정상 기대
    dict(id="S01", note="표준 성인 남성, 대칭",
         age=25, height_cm=175.0, weight_kg=70.0, sex="남성",
         foot_length_L_mm=262.0, foot_length_R_mm=261.0,
         foot_width_L_mm=99.0,  foot_width_R_mm=98.5,
         heel_width_L_mm=64.0,  heel_width_R_mm=63.5,
         arch_height_L_mm=24.0, arch_height_R_mm=24.0,
         instep_height_L_mm=62.0, instep_height_R_mm=61.5,
         leg_length_known=True, leg_length_L_mm=910.0, leg_length_R_mm=910.0,
         survey_shoe_wear=0, survey_pain_location=0, survey_fatigue_side=0,
         survey_posture_tilt=0, survey_shoulder_drop=0),

    # 2) 표준 여성, 거의 대칭 — 정상 기대
    dict(id="S02", note="표준 성인 여성, 대칭",
         age=23, height_cm=162.0, weight_kg=54.0, sex="여성",
         foot_length_L_mm=237.0, foot_length_R_mm=237.5,
         foot_width_L_mm=88.0,  foot_width_R_mm=88.0,
         heel_width_L_mm=56.0,  heel_width_R_mm=56.0,
         arch_height_L_mm=22.0, arch_height_R_mm=22.5,
         instep_height_L_mm=55.0, instep_height_R_mm=55.0,
         leg_length_known=True, leg_length_L_mm=845.0, leg_length_R_mm=843.0,
         survey_shoe_wear=0, survey_pain_location=0, survey_fatigue_side=0,
         survey_posture_tilt=0, survey_shoulder_drop=0),

    # 3) 하지 길이 차이 큼(우측 짧음 18mm) — 비대칭/힐리프트 기대
    dict(id="S03", note="하지 길이 차이(우 18mm 짧음)",
         age=31, height_cm=178.0, weight_kg=76.0, sex="남성",
         foot_length_L_mm=266.0, foot_length_R_mm=265.0,
         foot_width_L_mm=101.0, foot_width_R_mm=100.0,
         heel_width_L_mm=65.0,  heel_width_R_mm=64.0,
         arch_height_L_mm=25.0, arch_height_R_mm=24.0,
         instep_height_L_mm=63.0, instep_height_R_mm=62.0,
         leg_length_known=True, leg_length_L_mm=925.0, leg_length_R_mm=907.0,
         survey_shoe_wear=1, survey_pain_location=2, survey_fatigue_side=1,
         survey_posture_tilt=1, survey_shoulder_drop=1),

    # 4) 양측 평발(아치 높이 낮음) — 양측 아치 서포트 기대
    dict(id="S04", note="양측 평발 경향(낮은 아치)",
         age=44, height_cm=170.0, weight_kg=82.0, sex="남성",
         foot_length_L_mm=258.0, foot_length_R_mm=258.0,
         foot_width_L_mm=104.0, foot_width_R_mm=103.0,
         heel_width_L_mm=67.0,  heel_width_R_mm=66.0,
         arch_height_L_mm=10.0, arch_height_R_mm=11.0,
         instep_height_L_mm=58.0, instep_height_R_mm=58.0,
         leg_length_known=True, leg_length_L_mm=880.0, leg_length_R_mm=879.0,
         survey_shoe_wear=0, survey_pain_location=3, survey_fatigue_side=0,
         survey_posture_tilt=0, survey_shoulder_drop=0),

    # 5) 좌우 발 형태 비대칭(좌발 큼) — 비대칭 아치 서포트 기대
    dict(id="S05", note="좌우 발 형태 비대칭(좌발 큼)",
         age=28, height_cm=168.0, weight_kg=63.0, sex="여성",
         foot_length_L_mm=248.0, foot_length_R_mm=240.0,
         foot_width_L_mm=94.0,  foot_width_R_mm=88.0,
         heel_width_L_mm=60.0,  heel_width_R_mm=56.0,
         arch_height_L_mm=15.0, arch_height_R_mm=26.0,
         instep_height_L_mm=58.0, instep_height_R_mm=55.0,
         leg_length_known=True, leg_length_L_mm=862.0, leg_length_R_mm=858.0,
         survey_shoe_wear=-1, survey_pain_location=1, survey_fatigue_side=-1,
         survey_posture_tilt=-1, survey_shoulder_drop=0),

    # 6) 고령 여성, 경도 비대칭 — 경과 관찰 수준
    dict(id="S06", note="고령 여성, 경도 비대칭",
         age=67, height_cm=156.0, weight_kg=58.0, sex="여성",
         foot_length_L_mm=231.0, foot_length_R_mm=229.0,
         foot_width_L_mm=89.0,  foot_width_R_mm=87.0,
         heel_width_L_mm=57.0,  heel_width_R_mm=55.0,
         arch_height_L_mm=18.0, arch_height_R_mm=20.0,
         instep_height_L_mm=54.0, instep_height_R_mm=53.0,
         leg_length_known=False, leg_length_L_mm=None, leg_length_R_mm=None,
         survey_shoe_wear=2, survey_pain_location=1, survey_fatigue_side=2,
         survey_posture_tilt=0, survey_shoulder_drop=1),

    # 7) 키 큰 남성, 대칭 — 정상(큰 발 절대값 확인용)
    dict(id="S07", note="장신 남성, 대칭",
         age=21, height_cm=188.0, weight_kg=84.0, sex="남성",
         foot_length_L_mm=283.0, foot_length_R_mm=283.0,
         foot_width_L_mm=108.0, foot_width_R_mm=108.0,
         heel_width_L_mm=70.0,  heel_width_R_mm=69.5,
         arch_height_L_mm=27.0, arch_height_R_mm=27.0,
         instep_height_L_mm=66.0, instep_height_R_mm=66.0,
         leg_length_known=True, leg_length_L_mm=985.0, leg_length_R_mm=984.0,
         survey_shoe_wear=0, survey_pain_location=0, survey_fatigue_side=0,
         survey_posture_tilt=0, survey_shoulder_drop=0),

    # 8) 압력 비대칭 호소(설문 우측 쏠림) 하지길이 미입력
    dict(id="S08", note="자각 우측 쏠림(설문), 하지길이 미입력",
         age=35, height_cm=173.0, weight_kg=72.0, sex="남성",
         foot_length_L_mm=259.0, foot_length_R_mm=258.0,
         foot_width_L_mm=97.0,  foot_width_R_mm=99.0,
         heel_width_L_mm=62.0,  heel_width_R_mm=63.0,
         arch_height_L_mm=23.0, arch_height_R_mm=21.0,
         instep_height_L_mm=60.0, instep_height_R_mm=61.0,
         leg_length_known=False, leg_length_L_mm=None, leg_length_R_mm=None,
         survey_shoe_wear=1, survey_pain_location=1, survey_fatigue_side=1,
         survey_posture_tilt=1, survey_shoulder_drop=1),

    # 9) 젊은 여성, 약한 비대칭 + 무증상
    dict(id="S09", note="젊은 여성, 약한 비대칭 무증상",
         age=19, height_cm=165.0, weight_kg=55.0, sex="여성",
         foot_length_L_mm=242.0, foot_length_R_mm=241.0,
         foot_width_L_mm=90.0,  foot_width_R_mm=90.0,
         heel_width_L_mm=57.0,  heel_width_R_mm=57.0,
         arch_height_L_mm=25.0, arch_height_R_mm=24.0,
         instep_height_L_mm=56.0, instep_height_R_mm=56.0,
         leg_length_known=True, leg_length_L_mm=858.0, leg_length_R_mm=856.0,
         survey_shoe_wear=0, survey_pain_location=0, survey_fatigue_side=0,
         survey_posture_tilt=0, survey_shoulder_drop=0),

    # 10) 복합 비대칭(하지차+발형태차+설문 좌측 쏠림) — 현저 기대
    dict(id="S10", note="복합 비대칭(하지+발형태+설문)",
         age=52, height_cm=171.0, weight_kg=79.0, sex="남성",
         foot_length_L_mm=255.0, foot_length_R_mm=263.0,
         foot_width_L_mm=96.0,  foot_width_R_mm=102.0,
         heel_width_L_mm=61.0,  heel_width_R_mm=66.0,
         arch_height_L_mm=28.0, arch_height_R_mm=14.0,
         instep_height_L_mm=59.0, instep_height_R_mm=63.0,
         leg_length_known=True, leg_length_L_mm=895.0, leg_length_R_mm=916.0,
         survey_shoe_wear=-1, survey_pain_location=2, survey_fatigue_side=-1,
         survey_posture_tilt=-1, survey_shoulder_drop=-1),
]

PRED_PKL = ROOT / "models" / "dynamic_predictor.pkl"
CLF_PKL  = ROOT / "models" / "asymmetry_classifier.pkl"

SEV_NAME = {0: "정상", 1: "경도", 2: "중등도", 3: "현저"}


def analyze(ui):
    """앱과 동일 경로로 분석 → (임상등급, ML정상확률, 추천 수, 추천요약)"""
    pred = joblib.load(PRED_PKL)
    clf  = joblib.load(CLF_PKL)

    # 하지 길이 차이 파생(앱 1_input.py 로직과 동일)
    if ui.get("leg_length_known") and ui.get("leg_length_L_mm") is not None:
        ui = {**ui, "leg_length_diff_mm":
              ui["leg_length_L_mm"] - ui["leg_length_R_mm"]}
    else:
        ui = {**ui, "leg_length_diff_mm": 0}

    feat = {**ui, **derive_for_input(ui), **user_direct_for_input(ui)}
    pX = [feat.get(f, 0) for f in pred["features"]]
    dyn = dict(zip(pred["targets"], pred["model"].predict([pX])[0]))
    dyn = modulate_dynamic_by_geometry(dyn, ui)

    cX = [feat.get(f, dyn.get(f, 0)) if f in feat else dyn.get(f, 0)
          for f in clf["features"]]
    probas = clf["pipeline"].predict_proba([cX])[0]
    asym_type = int(probas.argmax())
    conf = float(probas[asym_type])
    p_normal = float(probas[0])

    clinical = clinical_assessment(ui)
    out = recommend(ui, dyn, asym_type, conf)
    devices = "; ".join(r["device"] for r in out["recommendations"]) or "없음"
    return (clinical["severity_name"], p_normal, dyn["grf_asymmetry_pct"],
            len(out["recommendations"]), devices)


# CSV 컬럼 정의 (입력 + 결과)
FIELDS = [
    ("id", "피험자ID"), ("note", "설명"),
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
RESULT_FIELDS = [
    ("res_severity", "비대칭등급"), ("res_pnormal", "ML정상확률"),
    ("res_grf", "GRF비대칭(%)"), ("res_nrec", "추천수"),
    ("res_devices", "추천장비"),
]


def main():
    out_dir = ROOT / "sample_data"
    out_dir.mkdir(exist_ok=True)

    rows = []
    for s in SUBJECTS:
        sev, pn, grf, nrec, devices = analyze(s)
        row = dict(s)
        row["res_severity"] = sev
        row["res_pnormal"]  = round(pn, 3)
        row["res_grf"]      = round(grf, 1)
        row["res_nrec"]     = nrec
        row["res_devices"]  = devices
        rows.append(row)

    # ── CSV ──────────────────────────────────────────
    csv_path = out_dir / "sample_inputs.csv"
    all_fields = FIELDS + RESULT_FIELDS
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([h for _, h in all_fields])
        for row in rows:
            w.writerow([row.get(k, "") if row.get(k) is not None else ""
                        for k, _ in all_fields])
    print(f"[OK] {csv_path}")

    # ── Markdown (보기 좋은 표) ───────────────────────
    md = ["# BodyBalance AI — 입력 샘플 데이터 (N=10)", "",
          "앱 입력 스키마에 맞춘 가상 피험자 10명의 입력값과 분석 결과.",
          "발 길이≈신장×0.15 등 인체측정 표준에 따라 설계.", ""]

    # 입력 요약 표
    md += ["## 입력 데이터", "",
           "| ID | 설명 | 나이 | 성별 | 신장 | 체중 | 발길이 L/R | 발너비 L/R | 아치 L/R | 하지 L/R |",
           "|----|------|-----|------|------|------|-----------|-----------|---------|---------|"]
    for r in rows:
        leg = (f"{r['leg_length_L_mm']:.0f}/{r['leg_length_R_mm']:.0f}"
               if r.get("leg_length_L_mm") is not None else "미입력")
        md.append(
            f"| {r['id']} | {r['note']} | {r['age']} | {r['sex']} | "
            f"{r['height_cm']:.0f} | {r['weight_kg']:.0f} | "
            f"{r['foot_length_L_mm']:.0f}/{r['foot_length_R_mm']:.0f} | "
            f"{r['foot_width_L_mm']:.0f}/{r['foot_width_R_mm']:.0f} | "
            f"{r['arch_height_L_mm']:.0f}/{r['arch_height_R_mm']:.0f} | {leg} |")

    # 설문 표
    md += ["", "## 자가진단 설문",
           "(좌 −1 / 같음 0 / 우 +1 / 모름 2, 통증: 없음0·무릎1·허리2·발목3·모름4)", "",
           "| ID | Q1 신발마모 | Q2 통증 | Q3 피로 | Q4 자세 | Q5 어깨 |",
           "|----|----|----|----|----|----|"]
    for r in rows:
        md.append(
            f"| {r['id']} | {r['survey_shoe_wear']} | {r['survey_pain_location']} | "
            f"{r['survey_fatigue_side']} | {r['survey_posture_tilt']} | "
            f"{r['survey_shoulder_drop']} |")

    # 분석 결과 표
    md += ["", "## 분석 결과 (시스템 출력)", "",
           "| ID | 비대칭 등급 | ML 정상확률 | GRF 비대칭 | 추천 장비 |",
           "|----|-----------|-----------|-----------|----------|"]
    for r in rows:
        md.append(
            f"| {r['id']} | {r['res_severity']} | {r['res_pnormal']} | "
            f"{r['res_grf']}% | {r['res_devices']} |")

    md += ["", "### 결과 해석 안내", "",
           "- **비대칭 등급**: 직접 측정값(하지 길이·좌우 발형태·설문)을 임상 "
           "문헌 기준으로 평가한 종합 판정 (정상/경도/중등도/현저).",
           "- **ML 정상확률**: 머신러닝 분류기가 산출한 '정상' 클래스 확률. "
           "모집단 중앙값(≈0.22) 대비 낮을수록 이상 신호. (보조 지표)",
           "- **GRF 비대칭**: 발 치수 기반 ML이 추정한 좌우 하중 차이. 임상 "
           "임계값 10% 이상이면 압력 보정 깔창을 추천.",
           "- **추천 장비**: 하지 길이차>6mm→힐 리프트, 평균 아치<15mm→양측 "
           "아치 서포트, 좌우 아치차≥6mm→비대칭 서포트, GRF≥10%→압력 보정.",
           "",
           "> 비대칭 등급(직접 측정 기반)과 GRF 비대칭(ML 추정 기반)은 서로 다른 "
           "근거의 지표이므로 값이 항상 비례하지는 않는다. 등급은 구조적 비대칭, "
           "GRF는 추정 동적 하중을 나타낸다."]

    md_path = out_dir / "sample_inputs.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] {md_path}")


if __name__ == "__main__":
    main()
