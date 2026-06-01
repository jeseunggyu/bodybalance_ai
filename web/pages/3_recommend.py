"""Step 3 — 교정 추천"""
import sys
import json
import hashlib
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.recommend import recommend

st.title("🦿 Step 3 — 교정 장비 추천")

if st.session_state.get("analysis_result") is None:
    st.warning("⚠️ `📊 Analysis` 페이지에서 분석을 먼저 완료해주세요.")
    st.stop()

user_input = st.session_state.user_input
result     = st.session_state.analysis_result

# 입력이 변경됐는데 분석을 다시 하지 않은 경우 경고
current_hash = hashlib.md5(
    json.dumps(user_input, sort_keys=True, default=str).encode()
).hexdigest()
if st.session_state.get("analysis_input_id") != current_hash:
    st.warning("⚠️ 입력값이 변경되었습니다. `📊 Analysis` 페이지에서 재분석 후 다시 확인해주세요.")
    st.stop()

output = recommend(
    user_input= user_input,
    predicted=  result["dynamic"],
    asym_type=  result["asym_type"],
    confidence= result["confidence"],
)

SEV_COLOR = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}
SEV_NAME  = {0: "정상", 1: "경도", 2: "중등도", 3: "현저"}
combined  = result.get("combined", result["clinical"]["severity"])

c1, c2, c3 = st.columns(3)
c1.metric("종합 비대칭 등급",
          f"{SEV_COLOR[combined]} {SEV_NAME[combined]}")
c2.metric("AI 분류 신뢰도", output["ai_confidence"])
c3.metric("입력 품질", output["input_quality"])

st.divider()

# ── 교정 장비 추천 ────────────────────────────────────
recs = output["recommendations"]
st.subheader(f"🎯 추천 교정 장비 ({len(recs)}건)")

if not recs:
    st.success("✅ 현재 비대칭 수준이 임상 기준 이내입니다.")
else:
    for rec in recs:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])

            icon = "🥇" if rec["priority"] == 1 else "🥈"
            c1.markdown(f"### {icon} {rec['device']}")
            c1.markdown(f"**사양**: `{rec['spec']}`")
            c1.markdown(f"**근거**: {rec['rationale']}")
            c1.caption(f"📚 {rec['evidence']}")

            conf_icon = {"높음": "🟢", "중간": "🟡", "낮음": "🔴"}.get(
                rec["confidence"], "⚪")
            c2.markdown(f"### {conf_icon}")
            c2.markdown(f"**{rec['confidence']}**")

st.divider()

# ── 상위 신체 평가 알림 ───────────────────────────────
alerts = output["alerts"]
if alerts:
    st.subheader("⚠️ 상위 신체 평가 권고")
    for alert in alerts:
        sev_color = {"높음": "🔴", "중간": "🟡"}.get(alert["severity"], "⚪")
        with st.container(border=True):
            st.markdown(f"{sev_color} **{alert['message']}**")
            st.markdown(f"→ {alert['action']}")

st.divider()
st.warning(output["disclaimer"])