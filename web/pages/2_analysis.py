"""Step 2 — AI 분석"""
import sys
from pathlib import Path
import streamlit as st
import joblib
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import PREDICTOR_PKL, CLASSIFIER_PKL

st.title("📊 Step 2 — AI 비대칭 분석")

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

# ── 동적 데이터 예측 ──────────────────────────────────
pred_X = [user_input.get(f, 0) for f in pred_pkg["features"]]
pred_y = pred_pkg["model"].predict([pred_X])[0]
dynamic = dict(zip(pred_pkg["targets"], pred_y))

# ── 비대칭 분류 ───────────────────────────────────────
clf_X = [
    (user_input.get(f, 0) if f in user_input else dynamic.get(f, 0))
    for f in clf_pkg["features"]
]
asym_type  = int(clf_pkg["pipeline"].predict([clf_X])[0])
probas     = clf_pkg["pipeline"].predict_proba([clf_X])[0]
confidence = float(probas[asym_type])

# 세션 저장
st.session_state.analysis_result = {
    "dynamic":   dynamic,
    "asym_type": asym_type,
    "confidence": confidence,
}

# ── 유형 판정 표시 ────────────────────────────────────
st.subheader("🧭 비대칭 유형 판정")
c1, c2 = st.columns([2, 1])
c1.markdown(f"### {clf_pkg['labels'][asym_type]}")
c2.metric("AI 신뢰도", f"{confidence * 100:.1f}%")

# 확률 분포
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

st.divider()

# ── 핵심 지표 ─────────────────────────────────────────
st.subheader("📈 핵심 비대칭 지표")
c1, c2, c3 = st.columns(3)

grf_asym  = dynamic.get("grf_asymmetry_pct", 0)
cop_asym  = dynamic.get("cop_asymmetry", 0)
arch_diff = abs(dynamic.get("arch_index_L", 0) - dynamic.get("arch_index_R", 0))

c1.metric("GRF 비대칭", f"{grf_asym:.1f}%",
          delta="정상" if grf_asym < 10 else "주의",
          delta_color="normal" if grf_asym < 10 else "inverse")
c2.metric("Arch Index 차이", f"{arch_diff:.4f}",
          delta="정상" if arch_diff < 0.02 else "주의",
          delta_color="normal" if arch_diff < 0.02 else "inverse")
c3.metric("CoP 비대칭", f"{cop_asym:.4f}",
          delta="정상" if cop_asym < 0.10 else "주의",
          delta_color="normal" if cop_asym < 0.10 else "inverse")

st.divider()

# ── 압력 분포 시각화 ──────────────────────────────────
st.subheader("🔥 예측 족저압 분포")
regions = ["Heel", "Midfoot", "Forefoot"]
L_vals = [dynamic[f"peak_{r.lower()}_L"] for r in regions]
R_vals = [dynamic[f"peak_{r.lower()}_R"] for r in regions]

fig_p = go.Figure()
fig_p.add_trace(go.Bar(name="좌발", x=regions, y=L_vals, marker_color="#4F87C5"))
fig_p.add_trace(go.Bar(name="우발", x=regions, y=R_vals, marker_color="#E8825A"))
fig_p.update_layout(barmode="group", height=350,
                    yaxis_title="Peak Pressure", margin=dict(t=30, b=40))
st.plotly_chart(fig_p, use_container_width=True)

st.info("➡️ 사이드바에서 `🦿 Recommend` 페이지로 이동하세요.")