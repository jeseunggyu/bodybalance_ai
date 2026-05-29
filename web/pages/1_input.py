"""Step 1 — 데이터 입력"""
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.lidar import process_scan

st.title("📥 Step 1 — 데이터 입력")
st.caption("모든 필수 항목을 정확히 입력해주세요. 숫자만 입력 가능합니다.")

# ──────────────────────────────────────────────
# 기본 정보 (필수)
# ──────────────────────────────────────────────
st.subheader("👤 기본 정보 (필수)")
c1, c2, c3, c4 = st.columns(4)

age    = c1.number_input("나이", min_value=15, max_value=80, value=None,
                          placeholder="예: 22", step=1)
height = c2.number_input("신장 (cm)", min_value=140.0, max_value=210.0,
                          value=None, placeholder="예: 170.0", step=0.1)
weight = c3.number_input("체중 (kg)", min_value=30.0, max_value=150.0,
                          value=None, placeholder="예: 65.0", step=0.1)
sex    = c4.selectbox("성별", ["선택", "남성", "여성"], index=0)

st.divider()

# ──────────────────────────────────────────────
# 발 측정 (필수)
# ──────────────────────────────────────────────
st.subheader("🦶 발 측정 (필수)")
input_mode = st.radio(
    "측정 방법",
    ["수동 입력 (줄자로 직접 측정)", "LiDAR 스캔 파일 업로드"],
    horizontal=True,
)

foot_data = {}

if input_mode.startswith("수동"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**좌발**")
        foot_data["foot_length_L_mm"] = st.number_input(
            "발 길이 좌 (mm)", min_value=180.0, max_value=320.0,
            value=None, placeholder="예: 260.0", step=0.1, key="fl_L")
        foot_data["foot_width_L_mm"] = st.number_input(
            "발 너비 좌 (mm)", min_value=60.0, max_value=120.0,
            value=None, placeholder="예: 95.0", step=0.1, key="fw_L")
        foot_data["heel_width_L_mm"] = st.number_input(
            "뒤꿈치 폭 좌 (mm)", min_value=40.0, max_value=80.0,
            value=None, placeholder="예: 60.0", step=0.1, key="hw_L")
        foot_data["arch_height_L_mm"] = st.number_input(
            "아치 높이 좌 (mm)", min_value=0.0, max_value=60.0,
            value=None, placeholder="예: 25.0", step=0.1, key="ah_L")
        foot_data["instep_height_L_mm"] = st.number_input(
            "발등 높이 좌 (mm)", min_value=40.0, max_value=100.0,
            value=None, placeholder="예: 60.0", step=0.1, key="ih_L")

    with c2:
        st.markdown("**우발**")
        foot_data["foot_length_R_mm"] = st.number_input(
            "발 길이 우 (mm)", min_value=180.0, max_value=320.0,
            value=None, placeholder="예: 258.0", step=0.1, key="fl_R")
        foot_data["foot_width_R_mm"] = st.number_input(
            "발 너비 우 (mm)", min_value=60.0, max_value=120.0,
            value=None, placeholder="예: 94.0", step=0.1, key="fw_R")
        foot_data["heel_width_R_mm"] = st.number_input(
            "뒤꿈치 폭 우 (mm)", min_value=40.0, max_value=80.0,
            value=None, placeholder="예: 59.0", step=0.1, key="hw_R")
        foot_data["arch_height_R_mm"] = st.number_input(
            "아치 높이 우 (mm)", min_value=0.0, max_value=60.0,
            value=None, placeholder="예: 22.0", step=0.1, key="ah_R")
        foot_data["instep_height_R_mm"] = st.number_input(
            "발등 높이 우 (mm)", min_value=40.0, max_value=100.0,
            value=None, placeholder="예: 58.0", step=0.1, key="ih_R")

else:
    c1, c2 = st.columns(2)
    L_file = c1.file_uploader("좌발 스캔 (.ply/.obj)", type=["ply", "obj"])
    R_file = c2.file_uploader("우발 스캔 (.ply/.obj)", type=["ply", "obj"])

    if L_file and R_file:
        if st.button("🔬 자동 측정 실행", type="secondary"):
            tmp_L = Path("/tmp/L.ply"); tmp_L.write_bytes(L_file.read())
            tmp_R = Path("/tmp/R.ply"); tmp_R.write_bytes(R_file.read())

            with st.spinner("LiDAR 처리 중..."):
                feat_L = process_scan(str(tmp_L))
                feat_R = process_scan(str(tmp_R))

            foot_data.update({
                "foot_length_L_mm":   feat_L["foot_length_mm"],
                "foot_length_R_mm":   feat_R["foot_length_mm"],
                "foot_width_L_mm":    feat_L["foot_width_mm"],
                "foot_width_R_mm":    feat_R["foot_width_mm"],
                "heel_width_L_mm":    feat_L["heel_width_mm"],
                "heel_width_R_mm":    feat_R["heel_width_mm"],
                "arch_height_L_mm":   feat_L["arch_height_mm"],
                "arch_height_R_mm":   feat_R["arch_height_mm"],
                "instep_height_L_mm": feat_L["instep_height_mm"],
                "instep_height_R_mm": feat_R["instep_height_mm"],
            })
            st.session_state.lidar_extracted = foot_data
            c1, c2 = st.columns(2)
            c1.json(feat_L)
            c2.json(feat_R)

    if "lidar_extracted" in st.session_state and not any(foot_data.values()):
        foot_data = st.session_state.lidar_extracted

st.divider()

# ──────────────────────────────────────────────
# 하지 길이 (선택)
# ──────────────────────────────────────────────
st.subheader("🦵 하지 길이 (선택)")
st.caption(
    "하지 길이는 정확한 측정이 어려운 항목입니다. "
    "정확히 알지 못하면 입력하지 않아도 됩니다. "
    "다만 알고 있다면 더 정확한 추천을 받을 수 있습니다."
)

provide_leg_length = st.checkbox(
    "하지 길이를 알고 있어 입력하겠습니다",
    value=False,
    help="ASIS(앞쪽 골반 돌출부)에서 내측 복사뼈까지의 거리"
)

leg_L, leg_R = None, None
if provide_leg_length:
    c1, c2 = st.columns(2)
    leg_L = c1.number_input(
        "좌 하지 길이 (mm)", min_value=700.0, max_value=1100.0,
        value=None, placeholder="예: 900.0", step=0.1, key="leg_L")
    leg_R = c2.number_input(
        "우 하지 길이 (mm)", min_value=700.0, max_value=1100.0,
        value=None, placeholder="예: 898.0", step=0.1, key="leg_R")

st.divider()

# ──────────────────────────────────────────────
# 설문 (필수)
# ──────────────────────────────────────────────
st.subheader("📋 자가진단 설문 (필수)")
st.caption("정확한 답이 어려운 경우 '잘 모르겠음'을 선택하세요.")

SIDE_OPT = {"좌측": -1, "차이 없음/같음": 0, "우측": 1, "잘 모르겠음": 2}
PAIN_OPT = {"없음": 0, "무릎": 1, "허리": 2, "발목": 3, "잘 모르겠음": 4}

q1 = st.radio("**Q1.** 신발 굽이 더 빨리 닳는 쪽은?",
              list(SIDE_OPT.keys()), horizontal=True, index=None)
q2 = st.radio("**Q2.** 평소 통증이 있는 부위는?",
              list(PAIN_OPT.keys()), horizontal=True, index=None)
q3 = st.radio("**Q3.** 오래 걸을 때 더 빨리 피로해지는 쪽은?",
              list(SIDE_OPT.keys()), horizontal=True, index=None)
q4 = st.radio("**Q4.** 거울로 봤을 때 자세가 한쪽으로 기울어 보이는 쪽은?",
              list(SIDE_OPT.keys()), horizontal=True, index=None)
q5 = st.radio("**Q5.** 어깨가 한쪽이 더 처져 보이는 쪽은?",
              list(SIDE_OPT.keys()), horizontal=True, index=None)

st.divider()

# ──────────────────────────────────────────────
# 제출 (검증 포함)
# ──────────────────────────────────────────────
if st.button("✅ 입력 완료 — 다음 단계로", type="primary",
             use_container_width=True):
    errors = []

    # 기본 정보 검증
    if age is None:    errors.append("• 나이를 입력해주세요")
    if height is None: errors.append("• 신장을 입력해주세요")
    if weight is None: errors.append("• 체중을 입력해주세요")
    if sex == "선택":  errors.append("• 성별을 선택해주세요")

    # 발 측정 검증
    if not foot_data or any(v is None for v in foot_data.values()):
        missing = [k for k, v in foot_data.items() if v is None]
        if not foot_data:
            errors.append("• 발 측정 데이터를 입력해주세요")
        else:
            errors.append(
                f"• 발 측정 항목 중 {len(missing)}개가 비어있습니다"
            )

    # 하지 길이 검증 (선택이지만 체크했으면 둘 다 필요)
    if provide_leg_length:
        if leg_L is None or leg_R is None:
            errors.append(
                "• 하지 길이를 입력하기로 선택했지만 값이 비어있습니다. "
                "양쪽 모두 입력하거나 체크박스를 해제하세요"
            )

    # 설문 검증
    survey_answers = [q1, q2, q3, q4, q5]
    if any(a is None for a in survey_answers):
        n_missing = sum(1 for a in survey_answers if a is None)
        errors.append(f"• 자가진단 설문 {n_missing}개 문항이 미응답입니다")

    # 에러 출력
    if errors:
        st.error("**입력을 완료해주세요:**\n\n" + "\n".join(errors))
    else:
        # 하지 길이 정보 처리
        if provide_leg_length and leg_L is not None and leg_R is not None:
            leg_diff = leg_L - leg_R
            leg_known = True
        else:
            leg_diff = 0
            leg_known = False

        st.session_state.user_input = {
            "age": age, "height_cm": height, "weight_kg": weight,
            "sex": sex,
            **foot_data,
            "leg_length_known":   leg_known,
            "leg_length_L_mm":    leg_L,
            "leg_length_R_mm":    leg_R,
            "leg_length_diff_mm": leg_diff,
            "survey_shoe_wear":     SIDE_OPT[q1],
            "survey_pain_location": PAIN_OPT[q2],
            "survey_fatigue_side":  SIDE_OPT[q3],
            "survey_posture_tilt":  SIDE_OPT[q4],
            "survey_shoulder_drop": SIDE_OPT[q5],
        }
        # 새 입력이 제출될 때마다 이전 분석 결과를 무효화
        st.session_state.analysis_result = None
        st.session_state.analysis_input_id = None
        st.session_state.input_id = id(st.session_state.user_input)
        st.success("✅ 입력 완료! 사이드바에서 `📊 Analysis`로 이동하세요.")
        st.balloons()