"""
test.py — Assignment 1 전체 실행 엔트리포인트

과제 PDF 요구사항에 맞춰 Problem 1, 2, 3의 검증 스크립트를 순차 실행합니다.

사용법:
    python test.py
    python test.py --fast
    python test.py --retrain

동작 방식:
  - Problem 1을 먼저 실행해 모델을 학습하거나 체크포인트를 준비합니다.
  - Problem 2, 3은 그 체크포인트를 재사용해 추가 공격을 평가합니다.
  - --retrain이 주어지면 Problem 1에서 모델을 새로 학습한 뒤, 이후 단계는
    새 체크포인트를 재사용합니다.
  - 각 문제 폴더의 results/ 아래 생성된 PNG를 루트 results/ 아래로 복사합니다.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
ROOT_RESULTS_DIR = ROOT_DIR / "results"


def copy_problem_results(problem_name: str) -> None:
    """
    각 문제 폴더의 PNG 결과물을 루트 results/ 아래로 모읍니다.
    """
    src_dir = ROOT_DIR / problem_name / "results"
    if not src_dir.exists():
        return

    dst_dir = ROOT_RESULTS_DIR / problem_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    for png_path in src_dir.glob("*.png"):
        shutil.copy2(png_path, dst_dir / png_path.name)


def run_problem(problem_name: str, extra_args: list[str]) -> None:
    """
    문제별 test.py를 현재 Python 인터프리터로 실행합니다.
    """
    script_path = ROOT_DIR / problem_name / "test.py"
    cmd = [sys.executable, str(script_path), *extra_args]

    print("\n" + "=" * 72)
    print(f"[ROOT] {problem_name} 실행")
    print(f"[ROOT] 명령: {' '.join(cmd)}")
    print("=" * 72)

    subprocess.run(cmd, cwd=ROOT_DIR, check=True)
    copy_problem_results(problem_name)


def main() -> None:
    fast_mode = "--fast" in sys.argv
    retrain = "--retrain" in sys.argv

    shared_args: list[str] = []
    if fast_mode:
        shared_args.append("--fast")

    ROOT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[ROOT] Assignment 1 전체 검증 시작")
    print(f"[ROOT] Python: {sys.executable}")
    print(f"[ROOT] fast_mode={fast_mode}, retrain={retrain}")

    # Problem 1에서만 --retrain을 넘겨 새 체크포인트를 만들고,
    # 이후 문제는 그 체크포인트를 재사용해 중복 학습을 줄입니다.
    problem1_args = list(shared_args)
    if retrain:
        problem1_args.append("--retrain")

    run_problem("problem1", problem1_args)
    run_problem("problem2", list(shared_args))
    run_problem("problem3", list(shared_args))

    print("\n" + "=" * 72)
    print("[ROOT] 전체 검증 완료")
    print(f"[ROOT] 통합 결과 폴더: {ROOT_RESULTS_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
