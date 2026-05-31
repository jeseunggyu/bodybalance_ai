"""Step 2 — 비대칭 분석"""
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


current_hash = _input_hash(user_input)
cached_hash  = st.session_state.get("analysis_input_id")

# 입력이 바뀐 경우(또는 분석 결과가 없는 경우) 항상 재계산
if st.session_state.get("analysis_result") is None or cached_hash != current_hash:
    pred_X = [user_input.get(f, 0) for f in pred_pkg["features"]]
    pred_y = pred_pkg["model"].predict([pred_X])[0]
    dynamic = dict(zip(pred_pkg["targets"], pred_y))

    clf_X = [
        (user_input.get(f, 0) if f in user_input else dynamic.get(f, 0))
        for f in clf_pkg["features"]
    ]
    asym_type  = int(clf_pkg["pipeline"].predict([clf_X])[0])
    probas     = clf_pkg["pipeline"].predict_proba([clf_X])[0]
    confidence = float(probas[asym_type])

    clinical = clinical_assessment(user_input)

    st.session_state.analysis_result = {
        "dynamic":    dynamic,
        "asym_type":  asym_type,
        "confidence": confidence,
        "probas":     probas.tolist(),
        "clinical":   clinical,
    }
    st.session_state.analysis_input_id = current_hash
else:
    res        = st.session_state.analysis_result
    dynamic    = res["dynamic"]
    asym_type  = res["asym_type"]
    confidence = res["confidence"]
    probas     = res["probas"]
    clinical   = res["clinical"]

# ══════════════════════════════════════════════════════════
# 주 판정 — 직접 측정 기반 구조적 비대칭 임상 평가
# ══════════════════════════════════════════════════════════
SEV_COLOR = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}
SEV_MSG = {
    0: "직접 측정값에서 임상적으로 유의한 좌우 비대칭이 발견되지 않았습니다.",
    1: "경도의 구조적 비대칭이 있습니다. 경과 관찰을 권장합니다.",
    2: "중등도의 구조적 비대칭이 있습니다. 교정 및 전문가 평가를 권장합니다.",
    3: "현저한 구조적 비대칭이 있습니다. 전문의 평가를 강력히 권장합니다.",
}
sev = clinical["severity"]

st.subheader("🧭 구조적 비대칭 판정 (직접 측정 기반)")
c1, c2 = st.columns([3, 1])
c1.markdown(f"## {SEV_COLOR[sev]} {clinical['severity_name']}")
c2.metric("소견 수", f"{len(clinical['findings'])}건")

if sev == 0:
    st.success(SEV_MSG[sev])
elif sev == 1:
    st.warning(SEV_MSG[sev])
else:
    st.error(SEV_MSG[sev])

if not clinical["leg_length_evaluated"]:
    st.caption("ℹ️ 하지 길이를 입력하지 않아 구조적 비대칭의 핵심 항목이 평가에서 제외되었습니다. "
               "정확한 판정을 위해 Input 페이지에서 하지 길이 입력을 권장합니다.")

if clinical["findings"]:
    st.markdown("##### 📋 발견된 소견")
    for f in clinical["findings"]:
        with st.container(border=True):
            cc1, cc2, cc3 = st.columns([2, 1, 1])
            cc1.markdown(f"**{f['item']}**  \n{f['note']}")
            cc2.metric("정도", f"{SEV_COLOR[f['grade']]} {f['grade_name']}")
            cc3.metric("측정", f"{f['side']} {f['value']}")

st.divider()

# ── 직접 측정 좌우 차이 (입력에 실제로 반응하는 지표) ──
st.subheader("📐 좌우 직접 측정 차이")
st.caption("아래 값은 사용자가 입력/측정한 값에서 직접 계산됩니다.")


def _diff(lk, rk):
    l, r = user_input.get(lk), user_input.get(rk)
    if l is None or r is None:
        return None
    return l - r


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

# ══════════════════════════════════════════════════════════
# 참고 — 데이터 기반 ML 분류 (정확도 제한적)
# ══════════════════════════════════════════════════════════
with st.expander("🤖 참고: 데이터 기반 ML 분류 (정확도 제한적, 보조 지표)"):
    st.caption(
        "⚠️ 본 ML 모델은 정적 발 형태로 동적 보행 비대칭을 예측하도록 학습되었으나 "
        "예측 신뢰도가 낮아(동적 예측기 평균회귀, 분류기 정확도 제한적) 입력이 달라져도 "
        "결과가 평균에 수렴할 수 있습니다. 위의 '직접 측정 기반 판정'을 우선하세요."
    )

    c1, c2 = st.columns([2, 1])
    c1.markdown(f"### {clf_pkg['labels'][asym_type]}")
    c2.metric("모델 신뢰도", f"{confidence * 100:.1f}%")

    fig = go.Figure(go.Bar(
        x=[clf_pkg["labels"][i] for i in range(len(probas))],
        y=probas,
        marker_color=[
            "#534AB7" if i == asym_type else "#CCCCCC"
            for i in range(len(probas))
        ],
    ))
    fig.update_layout(title="유형별 분류 확률", yaxis_range=[0, 1],
                      height=280, margin=dict(t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### 예측 족저압 분포 (추정치)")
    regions = ["Heel", "Midfoot", "Forefoot"]
    L_vals = [dynamic[f"peak_{r.lower()}_L"] for r in regions]
    R_vals = [dynamic[f"peak_{r.lower()}_R"] for r in regions]
    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(name="좌발", x=regions, y=L_vals, marker_color="#4F87C5"))
    fig_p.add_trace(go.Bar(name="우발", x=regions, y=R_vals, marker_color="#E8825A"))
    fig_p.update_layout(barmode="group", height=300,
                        yaxis_title="Peak Pressure", margin=dict(t=30, b=40))
    st.plotly_chart(fig_p, use_container_width=True)

st.info("➡️ 사이드바에서 `🦿 Recommend` 페이지로 이동하세요.")
