"""
test.py — Problem 1: Targeted FGSM 검증 및 시각화

이 스크립트는 다음을 순서대로 수행합니다:
  1. MNIST CNN 학습 (또는 저장된 체크포인트 로드)
  2. CIFAR-10 ResNet-18 학습 (또는 저장된 체크포인트 로드)
  3. Problem 1 — Targeted FGSM:
       - ε ∈ {0.05, 0.1, 0.2, 0.3} 각각에 대해 공격 성공률 측정
       - 공격 성공률: 모델이 target 클래스로 예측한 비율
       - 5개 이상 샘플 시각화 → results/ 폴더에 PNG 저장

사용법:
    python test.py              # 전체 실행 (500 샘플 평가)
    python test.py --fast       # 빠른 데모 (100 샘플)
    python test.py --retrain    # 체크포인트 무시하고 처음부터 재학습
"""

import os
import sys
import random

# 부모 디렉토리(assignment1/)를 경로에 추가
# → models.py, train.py를 루트에서 공통으로 사용
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # GUI 없는 서버 환경에서도 동작
import matplotlib.pyplot as plt

from models import MNISTNet, CIFAR10Net
from attacks import fgsm_targeted
from train import (
    get_mnist_loaders,
    get_cifar10_loaders,
    train_model,
    evaluate_model,
)

# 클래스 이름 정의
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
MNIST_CLASSES = [str(i) for i in range(10)]


# ============================================================
# 재현성 설정
# ============================================================

def set_seed(seed: int = 42):
    """모든 난수 생성기 시드를 고정해 재현성을 보장합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    """
    현재 환경에서 안전하게 사용할 장치를 선택합니다.

    cuda:1을 하드코딩하지 않고, 현재 프로세스가 사용할 수 있는 기본 CUDA 장치를
    그대로 사용합니다. 단일 GPU 환경에서도 안전하게 동작합니다.
    """
    if torch.cuda.is_available():
        gpu_index = torch.cuda.current_device()
        device = torch.device(f'cuda:{gpu_index}')
        print(f"[INFO] 사용 장치: {device}")
        print(f"[INFO] GPU: {torch.cuda.get_device_name(gpu_index)}")
        return device

    device = torch.device('cpu')
    print(f"[INFO] 사용 장치: {device}")
    return device


# ============================================================
# Problem 1 — 공격 성공률 계산
# ============================================================

def compute_targeted_success_rate(
    model,
    test_loader,
    eps: float,
    n_samples: int,
    device: torch.device,
) -> float:
    """
    Targeted FGSM 공격 성공률을 계산합니다.

    성공 기준 (Targeted): 모델이 target 클래스로 예측
    Target 선택 전략: target = (true_label + 1) % 10
      → 항상 정답과 다른 클래스를 target으로 선택

    Args:
        model       : 공격 대상 모델
        test_loader : 테스트 데이터 로더
        eps         : FGSM perturbation 크기
        n_samples   : 평가 샘플 수 (과제 요구 ≥ 100)
        device      : 연산 장치

    Returns:
        success_rate (float): 공격 성공률 (0~100%)
    """
    model.eval()
    success = 0
    total = 0

    for images, labels in test_loader:
        if total >= n_samples:
            break

        images = images.to(device)
        labels = labels.to(device)

        # clean에서 이미 틀린 샘플은 제외해 공격 성공률 과대평가를 방지합니다.
        with torch.no_grad():
            clean_preds = model(images).argmax(dim=1)

        correct_mask = clean_preds == labels
        if not correct_mask.any():
            continue

        images = images[correct_mask]
        labels = labels[correct_mask]

        remaining = n_samples - total
        images = images[:remaining]
        labels = labels[:remaining]

        # target 클래스: (true_label + 1) % 10
        # 예: 정답이 3이면 target은 4, 정답이 9이면 target은 0
        targets = (labels + 1) % 10

        # Targeted FGSM 공격 수행
        adv_images = fgsm_targeted(model, images, targets, eps)

        # 적대적 이미지에 대한 예측
        with torch.no_grad():
            logits = model(adv_images)
            preds = logits.argmax(dim=1)

        # 성공 판정: 예측 == target 클래스
        success += (preds == targets).sum().item()
        total   += labels.size(0)

    return 100.0 * success / total if total > 0 else 0.0


# ============================================================
# Problem 1 — 시각화
# ============================================================

def visualize_targeted_attack(
    model,
    test_loader,
    eps: float,
    device: torch.device,
    dataset_name: str,
    class_names: list,
    n_samples: int = 5,
):
    """
    Targeted FGSM 결과를 시각화하고 PNG로 저장합니다.

    각 샘플에 대해 3개 패널 (좌→우):
      1. 원본 이미지 + 정답 및 예측 레이블 (초록: 올바름)
      2. 적대적 이미지 + 모델 예측 (빨강: 공격 성공)
      3. Perturbation (차이) × 10 확대 — 육안으로 보기 위해

    과제 요구사항 (Section 5.1):
      - 최소 5개 샘플
      - 원본 이미지 + 예측 레이블
      - 적대적 이미지 + 모델의 (틀린) 예측
      - perturbation을 확대해서 시각화
      - results/ 디렉토리에 PNG로 저장

    Args:
        model       : 공격 대상 모델
        test_loader : 테스트 데이터 로더
        eps         : FGSM epsilon
        device      : 연산 장치
        dataset_name: 'MNIST' 또는 'CIFAR-10'
        class_names : 클래스 이름 목록
        n_samples   : 시각화할 샘플 수 (≥ 5)
    """
    model.eval()

    # 올바르게 분류된 샘플 n_samples개 수집
    # (공격 효과가 명확히 드러나도록 올바르게 예측된 것만 사용)
    collected_imgs   = []
    collected_labels = []

    for images, labels in test_loader:
        for i in range(images.size(0)):
            if len(collected_imgs) >= n_samples:
                break
            img   = images[i:i+1].to(device)
            label = labels[i:i+1].to(device)
            with torch.no_grad():
                pred = model(img).argmax(dim=1)
            if pred.item() == label.item():  # 올바르게 분류된 경우만
                collected_imgs.append(img)
                collected_labels.append(label)
        if len(collected_imgs) >= n_samples:
            break

    if len(collected_imgs) == 0:
        print(f"  [경고] 올바르게 분류된 샘플을 찾지 못했습니다.")
        return

    actual_n = len(collected_imgs)
    is_grayscale = (dataset_name == 'MNIST')

    # 그림 생성: 행 = 샘플 수, 열 = 3 (원본, 적대, perturbation)
    fig, axes = plt.subplots(actual_n, 3, figsize=(10, 3.2 * actual_n))
    if actual_n == 1:
        axes = axes[np.newaxis, :]  # 단일 샘플일 때 2D 배열로

    fig.suptitle(
        f'Problem 1: Targeted FGSM — {dataset_name}  (ε = {eps})\n'
        f'Target = (true label + 1) % 10',
        fontsize=13, fontweight='bold'
    )

    # 열 헤더
    axes[0, 0].set_title('Original Image\n(Correct Prediction)', fontsize=11, pad=8)
    axes[0, 1].set_title('Adversarial Image\n(Model Prediction)', fontsize=11, pad=8)
    axes[0, 2].set_title('Perturbation × 10\n(Magnified)', fontsize=11, pad=8)

    def to_display(tensor):
        """
        (1, C, H, W) 텐서 → 표시용 numpy array
          - MNIST (grayscale): (H, W)
          - CIFAR-10 (RGB):    (H, W, 3)
        """
        arr = tensor.squeeze(0).cpu().numpy()
        if is_grayscale:
            return arr.squeeze(0)          # (1, H, W) → (H, W)
        else:
            return arr.transpose(1, 2, 0)  # (C, H, W) → (H, W, C)

    for idx, (img, label) in enumerate(zip(collected_imgs, collected_labels)):
        true_cls   = label.item()
        target_cls = (true_cls + 1) % 10          # target 클래스
        target_t   = torch.tensor([target_cls], device=device)

        # 적대적 이미지 생성
        adv_img = fgsm_targeted(model, img, target_t, eps)

        # 예측
        with torch.no_grad():
            orig_pred = model(img).argmax(dim=1).item()
            adv_pred  = model(adv_img).argmax(dim=1).item()

        # perturbation = |x_adv - x|, 확대 × 10
        perturbation = (adv_img - img).abs()

        orig_np  = to_display(img)
        adv_np   = to_display(adv_img)
        pert_np  = np.clip(to_display(perturbation) * 10, 0, 1)

        # CIFAR-10 perturbation: 채널 평균으로 표시 (어디가 변했는지 직관적)
        if not is_grayscale:
            pert_np = pert_np.mean(axis=2)  # (H, W, 3) → (H, W)

        img_cmap  = 'gray' if is_grayscale else None
        pert_cmap = 'hot'   # 뜨거울수록 perturbation 큼

        # 패널 1: 원본 이미지
        axes[idx, 0].imshow(orig_np, cmap=img_cmap, vmin=0, vmax=1)
        axes[idx, 0].set_xlabel(
            f'True: {class_names[true_cls]}\nPred: {class_names[orig_pred]}',
            fontsize=9,
            color='green' if orig_pred == true_cls else 'red'
        )

        # 패널 2: 적대적 이미지
        axes[idx, 1].imshow(adv_np, cmap=img_cmap, vmin=0, vmax=1)
        attack_success = (adv_pred == target_cls)
        axes[idx, 1].set_xlabel(
            f'Target: {class_names[target_cls]}\nPred: {class_names[adv_pred]}  '
            f'{"✓ Success" if attack_success else "✗ Failed"}',
            fontsize=9,
            color='red' if attack_success else 'gray'
        )

        # 패널 3: perturbation 확대
        axes[idx, 2].imshow(pert_np, cmap=pert_cmap, vmin=0, vmax=1)
        max_pert = perturbation.max().item()
        axes[idx, 2].set_xlabel(f'max |Δ| = {max_pert:.4f}', fontsize=9)

        # 축 눈금 제거
        for ax in axes[idx]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()

    # results/ 디렉토리에 저장 (problem1/ 폴더 기준)
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    fname = f"{dataset_name.lower().replace('-','')}_problem1_eps{eps:.2f}.png"
    save_path = os.path.join(results_dir, fname)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  시각화 저장 완료: {save_path}")


# ============================================================
# 메인
# ============================================================

def main():
    # ── 커맨드라인 인수 ──
    fast_mode = '--fast'    in sys.argv   # 빠른 데모 (100 샘플)
    retrain   = '--retrain' in sys.argv   # 강제 재학습

    # ── 경로 설정 (가장 먼저 정의) ──
    root_dir = os.path.dirname(__file__)
    assignment_root = os.path.abspath(os.path.join(root_dir, '..'))
    data_dir = os.path.join(assignment_root, 'data')

    # ── 재현성 및 장치 ──
    set_seed(42)
    device = select_device()

    # 공격 평가 샘플 수 (과제: 최소 100개)
    n_samples  = 100 if fast_mode else 500
    # 과제 5.2에서 요구하는 ε 값 범위
    eps_values = [0.05, 0.1, 0.2, 0.3]

    # ==================================================================
    # PHASE 1: MNIST 학습
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: MNIST CNN 학습")
    print("=" * 60)

    mnist_train, mnist_test = get_mnist_loaders(data_dir=data_dir, batch_size=64)
    mnist_model = MNISTNet().to(device)

    mnist_ckpt = os.path.join(root_dir, 'mnist_model.pth')
    if os.path.exists(mnist_ckpt) and not retrain:
        print(f"[INFO] 저장된 MNIST 모델을 로드합니다: {mnist_ckpt}")
        mnist_model.load_state_dict(torch.load(mnist_ckpt, map_location=device))
    else:
        train_model(
            model=mnist_model,
            train_loader=mnist_train,
            num_epochs=10,
            device=device,
            lr=1e-3,
            weight_decay=1e-4,
            desc="MNIST",
        )
        torch.save(mnist_model.state_dict(), mnist_ckpt)
        print(f"[INFO] MNIST 모델 저장: {mnist_ckpt}")

    mnist_clean_acc = evaluate_model(mnist_model, mnist_test, device, desc="MNIST 클린 평가")
    print(f"\n✓ MNIST 클린 정확도: {mnist_clean_acc:.2f}%  (목표 ≥ 95%)")
    assert mnist_clean_acc >= 95.0, (
        f"MNIST 클린 정확도 {mnist_clean_acc:.2f}%가 목표(95%) 미달!\n"
        f"  → --retrain 플래그를 사용해 재학습하세요."
    )

    # ==================================================================
    # PHASE 2: CIFAR-10 학습
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: CIFAR-10 ResNet-18 학습")
    print("=" * 60)

    cifar_train, cifar_test = get_cifar10_loaders(data_dir=data_dir, batch_size=128)
    cifar_model = CIFAR10Net(pretrained=False).to(device)

    cifar_ckpt = os.path.join(root_dir, 'cifar10_model.pth')
    if os.path.exists(cifar_ckpt) and not retrain:
        print(f"[INFO] 저장된 CIFAR-10 모델을 로드합니다: {cifar_ckpt}")
        cifar_model.load_state_dict(torch.load(cifar_ckpt, map_location=device))
    else:
        # StepLR: epoch 20에서 lr 1e-3 → 1e-4로 감소
        train_model(
            model=cifar_model,
            train_loader=cifar_train,
            num_epochs=30,
            device=device,
            lr=1e-3,
            weight_decay=1e-4,
            scheduler_step=20,
            scheduler_gamma=0.1,
            desc="CIFAR-10",
        )
        torch.save(cifar_model.state_dict(), cifar_ckpt)
        print(f"[INFO] CIFAR-10 모델 저장: {cifar_ckpt}")

    cifar_clean_acc = evaluate_model(cifar_model, cifar_test, device, desc="CIFAR-10 클린 평가")
    print(f"\n✓ CIFAR-10 클린 정확도: {cifar_clean_acc:.2f}%  (목표 ≥ 80%)")
    assert cifar_clean_acc >= 80.0, (
        f"CIFAR-10 클린 정확도 {cifar_clean_acc:.2f}%가 목표(80%) 미달!\n"
        f"  → --retrain 플래그를 사용해 재학습하세요."
    )

    # ==================================================================
    # PHASE 3: Problem 1 — Targeted FGSM
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Problem 1 — Targeted FGSM")
    print(f"  Target 선택: (true_label + 1) % 10")
    print(f"  성공률 집계: clean에서 먼저 맞춘 샘플만 사용")
    print(f"  평가 샘플 수: {n_samples}개")
    print(f"  ε 값: {eps_values}")
    print("=" * 60)

    # results[(dataset_name, eps)] = success_rate
    results = {}

    configs = [
        ('MNIST',    mnist_model, mnist_test, MNIST_CLASSES),
        ('CIFAR-10', cifar_model, cifar_test, CIFAR10_CLASSES),
    ]

    for dataset_name, model, test_loader, class_names in configs:
        print(f"\n--- {dataset_name}: Targeted FGSM ---")

        # 각 ε 값에 대해 공격 성공률 측정
        for eps in eps_values:
            success_rate = compute_targeted_success_rate(
                model, test_loader, eps, n_samples, device
            )
            results[(dataset_name, eps)] = success_rate
            print(f"  ε = {eps:.2f}  →  공격 성공률: {success_rate:.2f}%")

        # 시각화: MNIST는 ε=0.3, CIFAR-10은 ε=0.1
        vis_eps = 0.3 if dataset_name == 'MNIST' else 0.1
        print(f"\n  [시각화] ε = {vis_eps}, 5개 샘플 생성 중...")
        visualize_targeted_attack(
            model=model,
            test_loader=test_loader,
            eps=vis_eps,
            device=device,
            dataset_name=dataset_name,
            class_names=class_names,
            n_samples=5,
        )

    # ==================================================================
    # 결과 요약 테이블 출력
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Problem 1: Targeted FGSM — 공격 성공률 (%)")
    print("  (Target = 정답 클래스가 아닌 특정 클래스로 분류)")
    print("=" * 60)
    header = f"{'Dataset':<12}" + "".join(f"   ε={e:.2f}" for e in eps_values)
    print(header)
    print("-" * 60)
    for dataset_name, _, _, _ in configs:
        row = f"{dataset_name:<12}"
        for eps in eps_values:
            rate = results[(dataset_name, eps)]
            row += f"  {rate:6.2f}%"
        print(row)
    print("=" * 60)

    print("\n[완료] Problem 1 (Targeted FGSM) 검증 완료")
    print(f"  결과 이미지: problem1/results/")
    print(f"  MNIST 체크포인트: {mnist_ckpt}")
    print(f"  CIFAR-10 체크포인트: {cifar_ckpt}")


if __name__ == '__main__':
    main()
