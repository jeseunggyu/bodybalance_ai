"""두 AI 모델 학습"""
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, classification_report

from src.config import (
    TRAIN_CSV, TEST_CSV, MODELS_DIR,
    PREDICTOR_PKL, CLASSIFIER_PKL,
    DYNAMIC_TARGETS, ASYM_LABELS,
)
from src.labeling import add_labels
from src.features import add_derived_features, DERIVED_FEATURES

# 학습에 사용할 피처 (공개 데이터에 있는 것만)
# 발 치수 절대값 + 좌우 차이/비율/비대칭 지수(파생) — 모델이 좌우 불균형에 반응하도록
TRAIN_FEATURES = [
    "foot_length_L_mm",
    "foot_length_R_mm",
    "foot_width_L_mm",
    "foot_width_R_mm",
] + DERIVED_FEATURES


def train_predictor(train_df, test_df):
    print("\n" + "=" * 50)
    print("[1/2] Dynamic Predictor — 발 형상 → 동적 족압")
    print("=" * 50)

    X_tr = train_df[TRAIN_FEATURES].values
    y_tr = train_df[DYNAMIC_TARGETS].fillna(0).values
    X_te = test_df[TRAIN_FEATURES].values
    y_te = test_df[DYNAMIC_TARGETS].fillna(0).values

    model = RandomForestRegressor(
        n_estimators=300, max_depth=15,
        min_samples_leaf=3, n_jobs=-1, random_state=42,
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    r2s = r2_score(y_te, y_pred, multioutput="raw_values")

    print(f"\n  Test MAE: {mae:.3f}")
    print(f"  타겟별 R²:")
    for name, s in zip(DYNAMIC_TARGETS, r2s):
        bar = "█" * int(max(s, 0) * 20)
        print(f"    {name:25s} R²={s:5.3f} {bar}")

    joblib.dump({
        "model": model,
        "features": TRAIN_FEATURES,
        "targets":  DYNAMIC_TARGETS,
    }, PREDICTOR_PKL)
    print(f"\n  ✅ 저장: {PREDICTOR_PKL}")


def train_classifier(train_df, test_df):
    print("\n" + "=" * 50)
    print("[2/2] Asymmetry Classifier — 비대칭 유형 분류")
    print("=" * 50)

    train_df = add_labels(train_df)
    test_df  = add_labels(test_df)

    print("\n  Train 클래스 분포:")
    for cls, count in train_df["asym_type"].value_counts().sort_index().items():
        print(f"    {cls} ({ASYM_LABELS[cls]}): {count}샘플")

    # 분류는 정적 피처 + 동적 타겟까지 활용
    clf_features = TRAIN_FEATURES + DYNAMIC_TARGETS
    X_tr = train_df[clf_features].values
    y_tr = train_df["asym_type"].values
    X_te = test_df[clf_features].values
    y_te = test_df["asym_type"].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=12,
            min_samples_leaf=3, n_jobs=-1, random_state=42,
        )),
    ])
    pipe.fit(X_tr, y_tr)

    target_names = [ASYM_LABELS[i] for i in sorted(set(y_te))]
    print("\n  Test 성능:")
    print(classification_report(
        y_te, pipe.predict(X_te),
        target_names=target_names, zero_division=0,
    ))

    joblib.dump({
        "pipeline": pipe,
        "features": clf_features,
        "labels":   ASYM_LABELS,
    }, CLASSIFIER_PKL)
    print(f"  ✅ 저장: {CLASSIFIER_PKL}")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_CSV.exists():
        print("❌ 전처리 먼저 실행: python -m src.preprocess")
        return

    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    print(f"📊 Train {len(train_df)} / Test {len(test_df)}")

    # 좌우 차이/비율/비대칭 지수 파생 피처 추가
    train_df = add_derived_features(train_df)
    test_df  = add_derived_features(test_df)

    train_predictor(train_df, test_df)
    train_classifier(train_df, test_df)
    print("\n🎉 학습 완료")


if __name__ == "__main__":
    main()