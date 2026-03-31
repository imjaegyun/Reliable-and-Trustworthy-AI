"""
test.py — Problem 2: Untargeted FGSM 검증 및 시각화

이 스크립트는 다음을 순서대로 수행합니다:
  1. MNIST / CIFAR-10 모델 로드 (problem1 체크포인트 우선 재사용)
  2. Problem 2 — Untargeted FGSM:
       - ε ∈ {0.05, 0.1, 0.2, 0.3} 각각에 대해 공격 성공률 측정
       - 공격 성공률: 모델이 정답이 아닌 클래스로 예측한 비율
       - 5개 이상 샘플 시각화 → results/ 폴더에 PNG 저장

Problem 1과의 차이:
  - 공격 성공 기준이 다름:
    Problem 1 (Targeted)  : pred == target_class  (특정 클래스로 유도)
    Problem 2 (Untargeted): pred != true_class     (아무 틀린 클래스면 성공)

사용법:
    python test.py              # 전체 실행 (500 샘플 평가)
    python test.py --fast       # 빠른 데모 (100 샘플)
    python test.py --retrain    # 체크포인트 무시하고 처음부터 재학습
"""

import os
import sys
import random

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 부모 디렉토리(assignment1/)를 경로에 추가 → models.py, train.py 공통 사용
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import MNISTNet, CIFAR10Net
from attacks import fgsm_untargeted
from train import (
    get_mnist_loaders,
    get_cifar10_loaders,
    train_model,
    evaluate_model,
)

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

    cuda:1을 고정하지 않고, 현재 프로세스에 노출된 기본 CUDA 장치를 사용합니다.
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
# Problem 2 — 공격 성공률 계산
# ============================================================

def compute_untargeted_success_rate(
    model,
    test_loader,
    eps: float,
    n_samples: int,
    device: torch.device,
) -> float:
    """
    Untargeted FGSM 공격 성공률을 계산합니다.

    성공 기준 (Untargeted): 모델이 정답이 아닌 어떤 클래스로든 예측
      → pred != true_label 이면 성공
      (Problem 1은 pred == target_class 이어야 성공)

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
    total   = 0

    for images, labels in test_loader:
        if total >= n_samples:
            break

        images = images.to(device)
        labels = labels.to(device)

        # clean에서 이미 틀린 샘플은 공격 성공률에서 제외합니다.
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

        # Untargeted FGSM 공격 수행
        adv_images = fgsm_untargeted(model, images, labels, eps)

        # 적대적 이미지에 대한 예측
        with torch.no_grad():
            preds = model(adv_images).argmax(dim=1)

        # 성공 판정: 예측 != 정답 (어떤 클래스든 틀리면 성공)
        success += (preds != labels).sum().item()
        total   += labels.size(0)

    return 100.0 * success / total if total > 0 else 0.0


# ============================================================
# Problem 2 — 시각화
# ============================================================

def visualize_untargeted_attack(
    model,
    test_loader,
    eps: float,
    device: torch.device,
    dataset_name: str,
    class_names: list,
    n_samples: int = 5,
):
    """
    Untargeted FGSM 결과를 시각화하고 PNG로 저장합니다.

    각 샘플에 대해 3개 패널:
      1. 원본 이미지 + 정답 및 예측 레이블
      2. 적대적 이미지 + 모델의 (틀린) 예측
      3. Perturbation × 10 확대

    과제 요구사항 (Section 5.1):
      - 원본 이미지 + 예측 레이블
      - 적대적 이미지 + 모델의 (틀린) 예측
      - perturbation 확대
      - results/ 에 PNG 저장
    """
    model.eval()

    # 올바르게 분류된 샘플 n_samples개 수집
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
            if pred.item() == label.item():
                collected_imgs.append(img)
                collected_labels.append(label)
        if len(collected_imgs) >= n_samples:
            break

    if len(collected_imgs) == 0:
        print("  [경고] 올바르게 분류된 샘플을 찾지 못했습니다.")
        return

    actual_n     = len(collected_imgs)
    is_grayscale = (dataset_name == 'MNIST')

    fig, axes = plt.subplots(actual_n, 3, figsize=(10, 3.2 * actual_n))
    if actual_n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f'Problem 2: Untargeted FGSM — {dataset_name}  (ε = {eps})\n'
        f'Goal: misclassify to any wrong class (pred != true label)',
        fontsize=13, fontweight='bold'
    )

    axes[0, 0].set_title('Original Image\n(Correct Prediction)', fontsize=11, pad=8)
    axes[0, 1].set_title('Adversarial Image\n(Wrong Prediction)', fontsize=11, pad=8)
    axes[0, 2].set_title('Perturbation × 10\n(Magnified)', fontsize=11, pad=8)

    def to_display(tensor):
        arr = tensor.squeeze(0).cpu().numpy()
        if is_grayscale:
            return arr.squeeze(0)
        else:
            return arr.transpose(1, 2, 0)

    for idx, (img, label) in enumerate(zip(collected_imgs, collected_labels)):
        true_cls = label.item()

        # Untargeted 공격: 정답 레이블만 넘기면 됨
        adv_img = fgsm_untargeted(model, img, label, eps)

        with torch.no_grad():
            orig_pred = model(img).argmax(dim=1).item()
            adv_pred  = model(adv_img).argmax(dim=1).item()

        perturbation = (adv_img - img).abs()

        orig_np = to_display(img)
        adv_np  = to_display(adv_img)
        # perturbation: CIFAR-10은 채널 평균으로 표시
        pert_np = to_display(perturbation)
        if not is_grayscale:
            pert_np = pert_np.mean(axis=2)
        pert_np = np.clip(pert_np * 10, 0, 1)

        img_cmap  = 'gray' if is_grayscale else None

        # 패널 1: 원본
        axes[idx, 0].imshow(orig_np, cmap=img_cmap, vmin=0, vmax=1)
        axes[idx, 0].set_xlabel(
            f'True: {class_names[true_cls]}\nPred: {class_names[orig_pred]}',
            fontsize=9, color='green'
        )

        # 패널 2: 적대적 이미지
        axes[idx, 1].imshow(adv_np, cmap=img_cmap, vmin=0, vmax=1)
        attack_success = (adv_pred != true_cls)
        axes[idx, 1].set_xlabel(
            f'Pred: {class_names[adv_pred]}  '
            f'{"✓ Success (misclassified)" if attack_success else "✗ Failed (correct)"}',
            fontsize=9,
            color='red' if attack_success else 'gray'
        )

        # 패널 3: perturbation 확대
        axes[idx, 2].imshow(pert_np, cmap='hot', vmin=0, vmax=1)
        axes[idx, 2].set_xlabel(
            f'max |Δ| = {perturbation.max().item():.4f}', fontsize=9
        )

        for ax in axes[idx]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    fname     = f"{dataset_name.lower().replace('-','')}_problem2_eps{eps:.2f}.png"
    save_path = os.path.join(results_dir, fname)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  시각화 저장 완료: {save_path}")


# ============================================================
# 메인
# ============================================================

def main():
    fast_mode = '--fast'    in sys.argv
    retrain   = '--retrain' in sys.argv

    set_seed(42)
    device = select_device()

    n_samples  = 100 if fast_mode else 500
    eps_values = [0.05, 0.1, 0.2, 0.3]

    root_dir = os.path.dirname(__file__)
    assignment_root = os.path.abspath(os.path.join(root_dir, '..'))

    # 체크포인트 경로: problem1에서 이미 학습한 것 재사용
    # problem1 체크포인트가 없으면 problem2 폴더에 새로 저장
    p1_dir         = os.path.join(root_dir, '..', 'problem1')
    mnist_ckpt     = os.path.join(p1_dir, 'mnist_model.pth')
    cifar_ckpt     = os.path.join(p1_dir, 'cifar10_model.pth')
    mnist_ckpt_p2  = os.path.join(root_dir, 'mnist_model.pth')
    cifar_ckpt_p2  = os.path.join(root_dir, 'cifar10_model.pth')

    # problem1 체크포인트가 없으면 problem2 자체 체크포인트 확인
    if not os.path.exists(mnist_ckpt):
        mnist_ckpt = mnist_ckpt_p2
    if not os.path.exists(cifar_ckpt):
        cifar_ckpt = cifar_ckpt_p2

    data_dir = os.path.join(assignment_root, 'data')

    # ==================================================================
    # PHASE 1: MNIST 모델 준비
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: MNIST CNN 준비")
    print("=" * 60)

    mnist_train, mnist_test = get_mnist_loaders(data_dir=data_dir, batch_size=64)
    mnist_model = MNISTNet().to(device)

    if os.path.exists(mnist_ckpt) and not retrain:
        print(f"[INFO] 체크포인트 로드: {mnist_ckpt}")
        mnist_model.load_state_dict(torch.load(mnist_ckpt, map_location=device))
    else:
        print("[INFO] 체크포인트 없음 → 처음부터 학습합니다.")
        train_model(mnist_model, mnist_train, num_epochs=10, device=device,
                    lr=1e-3, weight_decay=1e-4, desc="MNIST")
        torch.save(mnist_model.state_dict(), mnist_ckpt_p2)
        print(f"[INFO] MNIST 모델 저장: {mnist_ckpt_p2}")

    mnist_clean_acc = evaluate_model(mnist_model, mnist_test, device, desc="MNIST 클린 평가")
    print(f"\n✓ MNIST 클린 정확도: {mnist_clean_acc:.2f}%  (목표 ≥ 95%)")
    assert mnist_clean_acc >= 95.0, f"MNIST 정확도 {mnist_clean_acc:.2f}% < 95% 미달"

    # ==================================================================
    # PHASE 2: CIFAR-10 모델 준비
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: CIFAR-10 ResNet-18 준비")
    print("=" * 60)

    cifar_train, cifar_test = get_cifar10_loaders(data_dir=data_dir, batch_size=128)
    cifar_model = CIFAR10Net(pretrained=False).to(device)

    if os.path.exists(cifar_ckpt) and not retrain:
        print(f"[INFO] 체크포인트 로드: {cifar_ckpt}")
        cifar_model.load_state_dict(torch.load(cifar_ckpt, map_location=device))
    else:
        print("[INFO] 체크포인트 없음 → 처음부터 학습합니다.")
        train_model(cifar_model, cifar_train, num_epochs=30, device=device,
                    lr=1e-3, weight_decay=1e-4,
                    scheduler_step=20, scheduler_gamma=0.1, desc="CIFAR-10")
        torch.save(cifar_model.state_dict(), cifar_ckpt_p2)
        print(f"[INFO] CIFAR-10 모델 저장: {cifar_ckpt_p2}")

    cifar_clean_acc = evaluate_model(cifar_model, cifar_test, device, desc="CIFAR-10 클린 평가")
    print(f"\n✓ CIFAR-10 클린 정확도: {cifar_clean_acc:.2f}%  (목표 ≥ 80%)")
    assert cifar_clean_acc >= 80.0, f"CIFAR-10 정확도 {cifar_clean_acc:.2f}% < 80% 미달"

    # ==================================================================
    # PHASE 3: Problem 2 — Untargeted FGSM
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Problem 2 — Untargeted FGSM")
    print(f"  성공 기준: pred ≠ true_label (어떤 오분류든 성공)")
    print(f"  성공률 집계: clean에서 먼저 맞춘 샘플만 사용")
    print(f"  평가 샘플 수: {n_samples}개")
    print(f"  ε 값: {eps_values}")
    print("=" * 60)

    results = {}

    configs = [
        ('MNIST',    mnist_model, mnist_test, MNIST_CLASSES),
        ('CIFAR-10', cifar_model, cifar_test, CIFAR10_CLASSES),
    ]

    for dataset_name, model, test_loader, class_names in configs:
        print(f"\n--- {dataset_name}: Untargeted FGSM ---")

        for eps in eps_values:
            success_rate = compute_untargeted_success_rate(
                model, test_loader, eps, n_samples, device
            )
            results[(dataset_name, eps)] = success_rate
            print(f"  ε = {eps:.2f}  →  공격 성공률: {success_rate:.2f}%")

        # 시각화: MNIST는 ε=0.3, CIFAR-10은 ε=0.1 사용
        # (CIFAR-10에서 ε=0.3은 이미지를 완전히 파괴하므로 0.1이 더 적절)
        vis_eps = 0.3 if dataset_name == 'MNIST' else 0.1
        print(f"\n  [시각화] ε = {vis_eps}, 5개 샘플 생성 중...")
        visualize_untargeted_attack(
            model=model,
            test_loader=test_loader,
            eps=vis_eps,
            device=device,
            dataset_name=dataset_name,
            class_names=class_names,
            n_samples=5,
        )

    # ==================================================================
    # 결과 요약 테이블
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Problem 2: Untargeted FGSM — 공격 성공률 (%)")
    print("  (성공 기준: 정답이 아닌 클래스로 예측)")
    print("=" * 60)
    header = f"{'Dataset':<12}" + "".join(f"   ε={e:.2f}" for e in eps_values)
    print(header)
    print("-" * 60)
    for dataset_name, _, _, _ in configs:
        row = f"{dataset_name:<12}"
        for eps in eps_values:
            row += f"  {results[(dataset_name, eps)]:6.2f}%"
        print(row)
    print("=" * 60)

    print("\n[완료] Problem 2 (Untargeted FGSM) 검증 완료")
    print(f"  결과 이미지: problem2/results/")


if __name__ == '__main__':
    main()
