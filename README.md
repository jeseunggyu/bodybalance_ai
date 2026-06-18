# BodyBalance AI

**신체 비대칭 분석 기반 교정 장비 추천 시스템**
IMEN 343 Human Factors / Ergonomics Term Project

3D 발 스캔(LiDAR)과 줄자 측정·자가진단 설문을 입력받아, 머신러닝과 임상
문헌 기반 규칙을 결합해 좌우 신체 비대칭을 분석하고 근골격계 질환(MSD)
예방을 위한 교정 장비(아치 서포트·힐 리프트 등)를 추천한다.

---

## 1. 프로젝트 개요

장시간 보행·기립 시 발생하는 좌우 비대칭은 족저근막염, 슬관절통, 요통 등
근골격계 질환의 위험인자다. 본 시스템은 4차 산업혁명 기술(3D 센싱·머신러닝)을
활용해 개인의 발 형태와 비대칭을 정량 분석하고, 문헌 근거에 기반한 교정 장비를
제안하는 의사결정 지원 도구다.

## 2. 분석 방식 — ML + 임상 규칙 하이브리드

공개 족압 데이터셋(UNB StepUP-P150, 150명)으로 학습한 RandomForest를 정량
평가한 뒤, **모델이 잘하는 영역과 못하는 영역을 분리**해 설계했다.

| 예측 대상 | 검증 R² | 활용 |
|---|---|---|
| Arch Index (아치 유형) | ≈ 0.66 | ✅ ML 사용 |
| 접촉면적 (족부 부하) | ≈ 0.62–0.70 | ✅ ML 사용 |
| 좌우 비대칭 (GRF/CoP) | ≤ 0 | ❌ 예측 불가 → 규칙 엔진 |

- **머신러닝** (`src/train.py`): 발 치수 → 아치 유형·족부 부하·압력 패턴 추정
- **임상 규칙** (`src/labeling.py`의 `clinical_assessment`): 하지 길이·좌우
  발형태·설문을 문헌 기준(Gurney 2002, Knutson 2005 등)으로 평가해 비대칭
  등급(정상/경도/중등도/현저) 판정

> 정적 발형태만으로 동적 보행 비대칭을 예측하는 것은 데이터 구조상 한계가
> 있음을 검증으로 확인하고(R²<0), ML을 유효 영역에만 적용한 엔지니어링
> 의사결정이다. 모델을 맹신하지 않고 평가→한계 발견→적절한 방법(규칙)과
> 결합하는 과정 자체가 본 프로젝트의 핵심이다.

## 3. 처리 파이프라인

```
[입력]  3D 발 스캔(.ply/.obj)  +  줄자 측정  +  자가진단 설문
           │
           ▼
[측정]  src/lidar.py — 3D 메시 → 발 길이·너비·아치 높이 등 추출
           │
           ▼
[분석]  src/train.py 모델(ML)  +  src/labeling.py(임상 규칙)
        - ML: 아치 유형·족부 부하 추정
        - 규칙: 하지 길이·좌우 차이로 비대칭 등급 판정
           │
           ▼
[추천]  src/recommend.py — 문헌 기준 교정 장비(아치 서포트/힐 리프트)
```

## 4. 빠른 실행

```bash
# 1. 가상환경
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 실행 (전처리·학습 자동 → Streamlit 웹앱 실행)
python run.py
```

브라우저에서 `http://localhost:8501` 접속 → 사이드바에서
`📥 Input` → `📊 Analysis` → `🦿 Recommend` 순서로 진행한다.

> **참고**: 모델 파일(`models/*.pkl`)이 포함되어 있어, 원본 데이터 없이도
> 웹앱이 바로 실행된다. (원본 족압 데이터셋은 약 1.8GB로 repo에서 제외)

### 시연용 샘플 데이터
LiDAR 업로드 기능은 `sample_data/foot_L.obj`, `sample_data/foot_R.obj`로
바로 시연할 수 있다. (`tools/make_foot_obj.py`로 생성된 합성 발 메시)

## 5. 디렉터리 구조

```
src/
  config.py            전역 설정·피처 정의·임계값
  preprocess.py        원본 족압 데이터 → 학습용 CSV
  feature_extraction.py 압력 텐서 → 수치 피처
  features.py          파생 피처(좌우 차이·비율) + 합성 사용자 피처
  labeling.py          비대칭 유형 라벨링 + 임상 규칙 평가
  train.py             ML 모델 학습(예측기 + 분류기)
  lidar.py             3D 발 스캔 → 인체측정값 추출
  recommend.py         문헌 기반 교정 장비 추천 엔진
web/
  app.py               메인 페이지
  pages/               입력·분석·추천 3단계 UI
models/                학습된 모델(.pkl)
sample_data/           시연용 3D 발 메시(.obj)
tools/                 시연용 발 메시 생성기
```

## 6. 데이터셋 & 근거 문헌

- **데이터셋**: UNB StepUP-P150 (150명 보행 족압)
- **임상 근거**:
  - Cavanagh & Rodgers (1987) — Arch Index 발 분류
  - Robinson et al. (1987) — GRF 대칭 지수
  - Menz (1998) — 좌우 발 형태 비대칭
  - Gurney (2002) — 하지 길이 차이 보정
  - Knutson (2005) — 하지 길이 차이 임상 등급
  - Razeghi & Batt (2002), Kogler et al. (1996), Postema et al. (1998) — 교정 장비

---

> 본 시스템은 학부 인간공학 프로젝트의 의사결정 지원 도구이며, 의학적 진단을
> 대체하지 않는다. 통증이 있는 경우 전문의 상담을 권장한다.
