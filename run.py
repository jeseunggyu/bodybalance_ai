"""모든 단계 한 번에 실행"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: str):
    print(f"\n{'='*60}\n▶ {cmd}\n{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"❌ 실패: {cmd}")


def is_valid(path: Path) -> bool:
    """파일이 존재하고 내용이 있는지 확인 (0바이트 빈 파일 제외)"""
    return path.exists() and path.stat().st_size > 0


def main():
    # 1. 전처리
    if not is_valid(ROOT / "data/processed/train_features.csv"):
        run("python -m src.preprocess")
    else:
        print("✅ 전처리 이미 완료 (스킵)")

    # 2. 모델 학습 — dynamic_predictor + asymmetry_classifier 모두 확인
    if not is_valid(ROOT / "models/dynamic_predictor.pkl") or \
       not is_valid(ROOT / "models/asymmetry_classifier.pkl"):
        run("python -m src.train")
    else:
        print("✅ 모델 학습 이미 완료 (스킵)")

    # 3. 웹 앱 실행
    print("\n🚀 웹 앱 시작...")
    run("streamlit run web/app.py")


if __name__ == "__main__":
    main()