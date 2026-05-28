# BodyBalance AI

신체 비대칭 분석 기반 교정 장비 추천 시스템 (IMEN 343 HF/E Term Project)

## 빠른 실행
```bash
# 1. 가상환경
python -m venv venv && source venv/bin/activate

# 2. 패키지
pip install -r requirements.txt

# 3. 전처리
python -m src.ml.preprocess

# 4. 모델 학습
python -m src.ml.train_models

# 5. 웹 앱 실행
streamlit run web/app.py
```