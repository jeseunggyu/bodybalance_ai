"""BodyBalance AI 메인 페이지"""
import streamlit as st

st.set_page_config(
    page_title="BodyBalance AI",
    page_icon="🦶",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🦶 BodyBalance AI")
st.caption("신체 비대칭 분석 기반 교정 장비 추천 시스템")

c1, c2, c3 = st.columns(3)
c1.info("### 📥 1\n입력\n발 측정 + 설문")
c2.info("### 📊 2\n분석\nAI 동적 예측")
c3.info("### 🦿 3\n추천\n교정 장비")

st.divider()

st.markdown("""
### 🎯 프로젝트 개요
공개 족압 데이터셋(UNB StepUP-P150, 150명)으로 학습된 AI가
사용자의 정적 발 형상 데이터로부터 **동적 보행 비대칭을 예측**하고,
임상 가이드라인 기반의 교정 장비를 추천합니다.

### 🛠 입력 방식
- **수동 입력**: 줄자/체중계로 직접 측정한 값 입력
- **LiDAR 스캔**: iPhone Pro 등으로 스캔한 `.ply` 파일 업로드
- **설문 5문항**: 자가진단으로 모델 정확도 보완

### 📖 사용 순서
왼쪽 사이드바에서 `📥 Input` → `📊 Analysis` → `🦿 Recommend`
""")

# 세션 상태 초기화
for key in ["user_input", "analysis_result"]:
    if key not in st.session_state:
        st.session_state[key] = None