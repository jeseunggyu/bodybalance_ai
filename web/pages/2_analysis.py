"""Step 2 — 비대칭 분석 (ML + 임상 규칙 하이브리드 종합 평가)"""
import sys
import json
import hashlib
from pathlib import Path
import streamlit as st
import joblib
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import PREDICTOR_PKL, CLASSIFIER_PKL
from src.labeling import clinical_assessment
from src.features import derive_for_input, user_direct_for_input

st.title("📊 Step 2 — 비대칭 분석")

if st.session_state.get("user_input") is None:
    st.warning("⚠️ `📥 Input` 페이지에서 데이터를 먼저 입력해주세요.")
    st.stop()

user_input = st.session_state.user_input


@st.cache_resource
def load_models():
    return joblib.load(PREDICTOR_PKL), joblib.load(CLASSIFIER_PKL)


try:
    pred_pkg, clf_pkg = load_models()
except FileNotFoundError:
    st.error("❌ 모델 파일 없음. 학습 먼저 실행:\n"
             "`python -m src.preprocess && python -m src.train`")
    st.stop()


def _input_hash(data: dict) -> str:
    return hashlib.md5(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


def _ml_grade(prob_normal: float) -> int:
    """
    ML 정상 확률을 모집단 분포 기반으로 재보정해 이상 등급 산출.

    모집단(UNB P150, 파생피처+사용자 직접 입력 피처 포함 재학습 후 앱 추론) 분포:
      중앙값 ≈ 0.22, Q25 ≈ 0.14, Q10 ≈ 0.07
    (정상 확률의 절대값보다 '모집단 대비 상대 위치'가 의미 있으므로
     백분위 기반으로 등급화한다.)

    기준 해석: 중앙값 이상 → 정상, 아래로 내려갈수록 이상 신호.
    """
    if prob_normal >= 0.22:   return 0   # 모집단 중앙값 이상 → 정상
    if prob_normal >= 0.14:   return 1   # Q25 이상 → 경도 이상 신호
    if prob_normal >= 0.07:   return 2   # Q10 이상 → 중등도 이상 신호
    return 3                             # Q10 미만 → 현저한 이상 신호


def _combined_grade(ml_grade: int, clinical_grade: int) -> int:
    """
    ML 등급과 임상 등급을 결합해 최종 등급 산출.
    - 둘 다 이상 신호면 한 단계 상향 (상호 보강)
    - 하나라도 이상 신호면 해당 등급 반영
    """
    base = max(ml_grade, clinical_grade)
    if ml_grade >= 1 and clinical_grade >= 1 and base < 3:
        base += 1
    return base


current_hash = _input_hash(user_input)
cached_hash  = st.session_state.get("analysis_input_id")

if st.session_state.get("analysis_result") is None or cached_hash != current_hash:
    # 파생 피처(좌우 차이/비율) + 사용자 직접 입력 피처(하지 길이·설문)를
    # 추론 입력에도 학습과 동일하게 주입
    feat_input = {
        **user_input,
        **derive_for_input(user_input),
        **user_direct_for_input(user_input),
    }

    pred_X = [feat_input.get(f, 0) for f in pred_pkg["features"]]
    pred_y = pred_pkg["model"].predict([pred_X])[0]
    dynamic = dict(zip(pred_pkg["targets"], pred_y))

    clf_X = [
        (feat_input.get(f, 0) if f in feat_input else dynamic.get(f, 0))
        for f in clf_pkg["features"]
    ]
    probas     = clf_pkg["pipeline"].predict_proba([clf_X])[0]
    asym_type  = int(clf_pkg["pipeline"].predict([clf_X])[0])
    confidence = float(probas[asym_type])
    prob_normal = float(probas[0])

    ml_grade = _ml_grade(prob_normal)
    clinical = clinical_assessment(user_input)
    combined = _combined_grade(ml_grade, clinical["severity"])

    st.session_state.analysis_result = {
        "dynamic":      dynamic,
        "asym_type":    asym_type,
        "confidence":   confidence,
        "probas":       probas.tolist(),
        "prob_normal":  prob_normal,
        "ml_grade":     ml_grade,
        "clinical":     clinical,
        "combined":     combined,
    }
    st.session_state.analysis_input_id = current_hash
else:
    res         = st.session_state.analysis_result
    dynamic     = res["dynamic"]
    asym_type   = res["asym_type"]
    confidence  = res["confidence"]
    probas      = res["probas"]
    prob_normal = res["prob_normal"]
    ml_grade    = res["ml_grade"]
    clinical    = res["clinical"]
    combined    = res["combined"]

# ══════════════════════════════════════════════════════════
# 종합 판정 (ML + 임상 규칙 통합)
# ══════════════════════════════════════════════════════════
SEV_COLOR = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}
SEV_NAME  = {0: "정상", 1: "경도", 2: "중등도", 3: "현저"}
SEV_MSG   = {
    0: "ML 및 직접 측정 모두 임상적으로 유의한 비대칭이 없습니다.",
    1: "경도의 비대칭 신호가 감지되었습니다. 경과 관찰을 권장합니다.",
    2: "중등도의 비대칭이 감지되었습니다. 교정 및 전문가 평가를 권장합니다.",
    3: "현저한 비대칭이 감지되었습니다. 전문의 평가를 강력히 권장합니다.",
}
ml_label = clf_pkg["labels"][asym_type]

st.subheader("🧭 종합 비대칭 판정")
c1, c2, c3 = st.columns([2, 2, 1])
c1.markdown(f"**종합 등급**  \n## {SEV_COLOR[combined]} {SEV_NAME[combined]}")
c2.markdown(f"**ML 압력 패턴 추정**  \n### {ml_label}")
c3.metric("ML 정상 확률", f"{prob_normal * 100:.0f}%",
          help="모집단 중앙값(93%) 대비 위치로 이상 신호 판단")

if combined == 0:
    st.success(SEV_MSG[0])
elif combined == 1:
    st.warning(SEV_MSG[1])
else:
    st.error(SEV_MSG[combined])

# ML과 임상의 기여도 표시
ml_contrib = SEV_COLOR[ml_grade] + f" ML({SEV_NAME[ml_grade]})"
cl_contrib = SEV_COLOR[clinical["severity"]] + f" 임상({SEV_NAME[clinical['severity']]})"
st.caption(f"판정 근거 — {ml_contrib} × {cl_contrib} → 종합 {SEV_COLOR[combined]} {SEV_NAME[combined]}")

if not clinical["leg_length_evaluated"]:
    st.caption("ℹ️ 하지 길이 미입력 — 입력 시 더 정확한 판정이 가능합니다.")

st.divider()

# ══════════════════════════════════════════════════════════
# 세부 소견 (ML + 임상 통합 표시)
# ══════════════════════════════════════════════════════════
st.subheader("📋 세부 소견")

# ML 소견
with st.container(border=True):
    mc1, mc2, mc3 = st.columns([2, 1, 1])
    mc1.markdown(f"**ML 압력 패턴 분류**  \n{ml_label}")
    mc2.metric("이상 등급", f"{SEV_COLOR[ml_grade]} {SEV_NAME[ml_grade]}")
    # 모집단 중앙값(0.22) → 0%, 최저(0.02) → 100%
    normalized_pct = min(100, max(0, (0.22 - prob_normal) / (0.22 - 0.02) * 100))
    mc3.metric("이상 신호 강도", f"{normalized_pct:.0f}%",
               help="정상 확률이 모집단 중앙값(22%)에서 얼마나 벗어났는지 (0%=중앙값, 100%=최저)")

# 임상 소견들
for f in clinical["findings"]:
    with st.container(border=True):
        cc1, cc2, cc3 = st.columns([2, 1, 1])
        cc1.markdown(f"**{f['item']}**  \n{f['note']}")
        cc2.metric("정도", f"{SEV_COLOR[f['grade']]} {f['grade_name']}")
        cc3.metric("측정", f"{f['side']} {f['value']}")

if not clinical["findings"] and ml_grade == 0:
    st.info("이상 소견 없음")

st.divider()

# ══════════════════════════════════════════════════════════
# 수치 지표
# ══════════════════════════════════════════════════════════
st.subheader("📐 측정 지표")

# ML 예측 족저압
st.markdown("**예측 족저압 분포** (ML 추정 — 발 치수 기반)")
regions = ["Heel", "Midfoot", "Forefoot"]
L_vals = [dynamic[f"peak_{r.lower()}_L"] for r in regions]
R_vals = [dynamic[f"peak_{r.lower()}_R"] for r in regions]
fig_p = go.Figure()
fig_p.add_trace(go.Bar(name="좌발", x=regions, y=L_vals, marker_color="#4F87C5"))
fig_p.add_trace(go.Bar(name="우발", x=regions, y=R_vals, marker_color="#E8825A"))
fig_p.update_layout(barmode="group", height=280,
                    yaxis_title="Peak Pressure", margin=dict(t=10, b=40))
st.plotly_chart(fig_p, use_container_width=True)

# 직접 측정 좌우 차이
st.markdown("**좌우 직접 측정 차이**")


def _diff(lk, rk):
    l, r = user_input.get(lk), user_input.get(rk)
    return (l - r) if l is not None and r is not None else None


m1, m2, m3 = st.columns(3)
fl = _diff("foot_length_L_mm", "foot_length_R_mm")
fw = _diff("foot_width_L_mm", "foot_width_R_mm")
ah = _diff("arch_height_L_mm", "arch_height_R_mm")

if fl is not None:
    m1.metric("발 길이 차이", f"{abs(fl):.1f}mm",
              delta="정상" if abs(fl) < 8 else "주의",
              delta_color="normal" if abs(fl) < 8 else "inverse")
if fw is not None:
    m2.metric("발 너비 차이", f"{abs(fw):.1f}mm",
              delta="정상" if abs(fw) < 6 else "주의",
              delta_color="normal" if abs(fw) < 6 else "inverse")
if ah is not None:
    m3.metric("아치 높이 차이", f"{abs(ah):.1f}mm",
              delta="정상" if abs(ah) < 6 else "주의",
              delta_color="normal" if abs(ah) < 6 else "inverse")

if user_input.get("leg_length_known"):
    ld = user_input.get("leg_length_diff_mm", 0) or 0
    st.metric("하지 길이 차이 (LLD)", f"{abs(ld):.1f}mm",
              delta="정상" if abs(ld) < 6 else "주의",
              delta_color="normal" if abs(ld) < 6 else "inverse")

st.divider()
st.info("➡️ 사이드바에서 `🦿 Recommend` 페이지로 이동하세요.")
