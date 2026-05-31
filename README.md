# BodyBalance AI

신체 비대칭 분석 기반 교정 장비 추천 시스템 (IMEN 343 HF/E Term Project)

## 빠른 실행
```bash
# 1. 가상환경
python -m venv venv && source venv/bin/activate

# 2. 패키지
pip install -r requirements.txt

# 3. 실행
python run.py
```

## 분석 방식 — ML + 임상 규칙 하이브리드

공개 족압 데이터셋(UNB StepUP-P150, 150명)으로 학습한 RandomForest를 정량 평가한 뒤, **모델이 잘하는 영역과 못하는 영역을 분리**해 설계했습니다.

| 예측 대상 | 검증 R² | 활용 |
|---|---|---|
| Arch Index (아치 유형) | ≈ 0.66 | ✅ ML 사용 |
| 접촉면적 (족부 부하) | ≈ 0.62–0.70 | ✅ ML 사용 |
| 좌우 비대칭 (GRF/CoP) | ≤ 0 | ❌ 예측 불가 → 규칙 엔진 |

- **머신러닝** (`src/train.py`): 발 치수 → 아치 유형·족부 부하·압력 패턴 추정
- **임상 규칙** (`src/labeling.py` `clinical_assessment`): 하지 길이·좌우 발형태·설문을 문헌 기준(Gurney 2002, Knutson 2005 등)으로 평가해 비대칭 등급(정상/경도/중등도/현저) 판정

> 정적 발형태만으로 동적 보행 비대칭을 예측하는 것은 데이터 구조상 한계가 있음을 검증으로 확인하고(R²<0), ML을 유효 영역에만 적용한 엔지니어링 의사결정입니다.

## 데이터셋
UNB StepUP-P150 (150명 보행 족압). 원본 데이터는 용량 문제로 repo에서 제외됩니다.
