"""
test.py — Problem 3: PGD (Targeted + Untargeted) 검증 및 시각화

이 스크립트는 다음을 순서대로 수행합니다:
  1. MNIST / CIFAR-10 모델 준비 (problem1 체크포인트 우선 재사용)
  2. Problem 3-A — Targeted PGD:
       - ε ∈ {0.05, 0.1, 0.2, 0.3}, k=40, eps_step=ε/10
       - 공격 성공률: pred == target_class
  3. Problem 3-B — Untargeted PGD:
       - 동일 하이퍼파라미터
       - 공격 성공률: pred != true_label
  4. 각각 5개 샘플 시각화 → results/ 폴더에 PNG 저장

PGD vs FGSM 비교 (참고):
  FGSM (k=1)  : 빠르지만 약함
  PGD  (k=40) : 느리지만 훨씬 강력 → 동일 ε에서 성공률이 훨씬 높음

사용법:
    python test.py              # 전체 실행 (500 샘플 평가)
    python test.py --fast       # 빠른 데모 (100 샘플, k=10)
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
from attacks import pgd_targeted, pgd_untargeted
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    """
    현재 환경에서 안전하게 사용할 장치를 선택합니다.

    cuda:1을 하드코딩하지 않고 현재 기본 CUDA 장치를 사용해,
    단일 GPU 환경에서도 그대로 실행되게 합니다.
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
# Problem 3 — 공격 성공률 계산
# ============================================================

def compute_pgd_success_rate(
    model,
    test_loader,
    attack_fn,
    n_samples: int,
    device: torch.device,
    is_targeted: bool,
) -> float:
    """
    PGD 공격 성공률을 계산합니다.

    Args:
        model       : 공격 대상 모델
        test_loader : 테스트 데이터 로더
        attack_fn   : 공격 함수 (functools.partial로 하이퍼파라미터 바인딩)
                      시그니처: (model, images, labels_or_targets) → adv_images
        n_samples   : 평가 샘플 수
        device      : 연산 장치
        is_targeted : True이면 Targeted (pred==target), False이면 Untargeted (pred!=true)

    Returns:
        success_rate (float): 0~100%
    """
    model.eval()
    success = 0
    total   = 0

    for images, labels in test_loader:
        if total >= n_samples:
            break

        images = images.to(device)
        labels = labels.to(device)

        # clean에서 이미 틀린 샘플을 제거해 공격 성공률을 정확히 집계합니다.
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

        if is_targeted:
            # Targeted: target = (true_label + 1) % 10
            targets    = (labels + 1) % 10
            adv_images = attack_fn(model, images, targets)
            with torch.no_grad():
                preds = model(adv_images).argmax(dim=1)
            success += (preds == targets).sum().item()
        else:
            # Untargeted: 정답 레이블 전달
            adv_images = attack_fn(model, images, labels)
            with torch.no_grad():
                preds = model(adv_images).argmax(dim=1)
            success += (preds != labels).sum().item()

        total += labels.size(0)

    return 100.0 * success / total if total > 0 else 0.0


# ============================================================
# Problem 3 — 시각화
# ============================================================

def visualize_pgd_attack(
    model,
    test_loader,
    attack_fn,
    eps: float,
    device: torch.device,
    dataset_name: str,
    class_names: list,
    is_targeted: bool,
    n_samples: int = 5,
):
    """
    PGD 공격 결과 시각화 및 PNG 저장.

    각 샘플 3개 패널: 원본 / 적대적 이미지 / perturbation × 10
    """
    model.eval()

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

    if not collected_imgs:
        print("  [경고] 올바르게 분류된 샘플을 찾지 못했습니다.")
        return

    actual_n     = len(collected_imgs)
    is_grayscale = (dataset_name == 'MNIST')
    attack_type  = 'Targeted' if is_targeted else 'Untargeted'

    fig, axes = plt.subplots(actual_n, 3, figsize=(10, 3.2 * actual_n))
    if actual_n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f'Problem 3: {attack_type} PGD — {dataset_name}  (ε = {eps})\n'
        + (f'Target = (true label + 1) % 10' if is_targeted
           else f'Goal: misclassify to any wrong class'),
        fontsize=13, fontweight='bold'
    )
    axes[0, 0].set_title('Original Image\n(Correct Prediction)', fontsize=11, pad=8)
    axes[0, 1].set_title('Adversarial Image\n(Model Prediction)', fontsize=11, pad=8)
    axes[0, 2].set_title('Perturbation × 10\n(Magnified)', fontsize=11, pad=8)

    def to_display(tensor):
        arr = tensor.squeeze(0).cpu().numpy()
        return arr.squeeze(0) if is_grayscale else arr.transpose(1, 2, 0)

    for idx, (img, label) in enumerate(zip(collected_imgs, collected_labels)):
        true_cls   = label.item()
        target_cls = (true_cls + 1) % 10 if is_targeted else None
        query      = torch.tensor([target_cls], device=device) if is_targeted else label

        adv_img = attack_fn(model, img, query)

        with torch.no_grad():
            orig_pred = model(img).argmax(dim=1).item()
            adv_pred  = model(adv_img).argmax(dim=1).item()

        perturbation = (adv_img - img).abs()

        orig_np = to_display(img)
        adv_np  = to_display(adv_img)
        pert_np = to_display(perturbation)
        if not is_grayscale:
            pert_np = pert_np.mean(axis=2)
        pert_np = np.clip(pert_np * 10, 0, 1)

        img_cmap = 'gray' if is_grayscale else None

        # 원본
        axes[idx, 0].imshow(orig_np, cmap=img_cmap, vmin=0, vmax=1)
        axes[idx, 0].set_xlabel(
            f'True: {class_names[true_cls]}\nPred: {class_names[orig_pred]}',
            fontsize=9, color='green'
        )

        # 적대적
        axes[idx, 1].imshow(adv_np, cmap=img_cmap, vmin=0, vmax=1)
        if is_targeted:
            success = (adv_pred == target_cls)
            label_txt = (f'Target: {class_names[target_cls]}\n'
                         f'Pred: {class_names[adv_pred]}  '
                         f'{"✓ Success" if success else "✗ Failed"}')
        else:
            success = (adv_pred != true_cls)
            label_txt = (f'Pred: {class_names[adv_pred]}  '
                         f'{"✓ Success (misclassified)" if success else "✗ Failed (correct)"}')
        axes[idx, 1].set_xlabel(label_txt, fontsize=9,
                                color='red' if success else 'gray')

        # perturbation
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
    attack_tag = 'targeted' if is_targeted else 'untargeted'
    fname      = f"{dataset_name.lower().replace('-','')}_problem3_{attack_tag}_eps{eps:.2f}.png"
    save_path  = os.path.join(results_dir, fname)
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

    # PGD 하이퍼파라미터 (PDF 권장값: ε=0.3, eps_step=0.01, k=40 for MNIST)
    K          = 10 if fast_mode else 40   # fast_mode에서 k 줄여 속도 향상
    EPS_STEP_RATIO = 0.1                   # eps_step = eps × 0.1

    root_dir = os.path.dirname(__file__)
    assignment_root = os.path.abspath(os.path.join(root_dir, '..'))
    data_dir = os.path.join(assignment_root, 'data')

    # 체크포인트: problem1 → problem2 → problem3 순으로 탐색
    def find_ckpt(filename):
        for folder in ['problem1', 'problem2', 'problem3']:
            path = os.path.join(root_dir, '..', folder, filename)
            if os.path.exists(path):
                return path
        return os.path.join(root_dir, filename)  # 없으면 현재 폴더

    mnist_ckpt = find_ckpt('mnist_model.pth')
    cifar_ckpt = find_ckpt('cifar10_model.pth')

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
        save_path = os.path.join(root_dir, 'mnist_model.pth')
        torch.save(mnist_model.state_dict(), save_path)
        print(f"[INFO] MNIST 모델 저장: {save_path}")

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
        save_path = os.path.join(root_dir, 'cifar10_model.pth')
        torch.save(cifar_model.state_dict(), save_path)
        print(f"[INFO] CIFAR-10 모델 저장: {save_path}")

    cifar_clean_acc = evaluate_model(cifar_model, cifar_test, device, desc="CIFAR-10 클린 평가")
    print(f"\n✓ CIFAR-10 클린 정확도: {cifar_clean_acc:.2f}%  (목표 ≥ 80%)")
    assert cifar_clean_acc >= 80.0, f"CIFAR-10 정확도 {cifar_clean_acc:.2f}% < 80% 미달"

    # ==================================================================
    # PHASE 3: Problem 3 — Targeted PGD + Untargeted PGD
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"PHASE 3: Problem 3 — PGD (k={K}, eps_step = ε × {EPS_STEP_RATIO})")
    print(f"  성공률 집계: clean에서 먼저 맞춘 샘플만 사용")
    print(f"  평가 샘플 수: {n_samples}개  |  ε 값: {eps_values}")
    print("=" * 60)

    # {(dataset, variant, eps): success_rate}
    results = {}

    configs = [
        ('MNIST',    mnist_model, mnist_test, MNIST_CLASSES),
        ('CIFAR-10', cifar_model, cifar_test, CIFAR10_CLASSES),
    ]

    for dataset_name, model, test_loader, class_names in configs:
        print(f"\n--- {dataset_name} ---")

        for variant, is_targeted in [('Targeted', True), ('Untargeted', False)]:
            print(f"\n  [{variant} PGD]")

            for eps in eps_values:
                eps_step = eps * EPS_STEP_RATIO  # eps_step = ε / 10

                # functools.partial 없이 lambda로 하이퍼파라미터 바인딩
                # (lambda 캡처 문제를 피하기 위해 기본 인수로 바인딩)
                if is_targeted:
                    attack_fn = lambda m, x, t, e=eps, s=eps_step, kk=K: pgd_targeted(m, x, t, kk, e, s)
                else:
                    attack_fn = lambda m, x, l, e=eps, s=eps_step, kk=K: pgd_untargeted(m, x, l, kk, e, s)

                success_rate = compute_pgd_success_rate(
                    model, test_loader, attack_fn, n_samples, device, is_targeted
                )
                results[(dataset_name, variant, eps)] = success_rate
                print(f"    ε = {eps:.2f}, eps_step = {eps_step:.4f}  "
                      f"→  공격 성공률: {success_rate:.2f}%")

            # 시각화: ε = 0.3 기준 5개 샘플
            eps_viz  = 0.3 if dataset_name == 'MNIST' else 0.1
            step_viz = eps_viz * EPS_STEP_RATIO
            if is_targeted:
                viz_fn = lambda m, x, t, e=eps_viz, s=step_viz, kk=K: pgd_targeted(m, x, t, kk, e, s)
            else:
                viz_fn = lambda m, x, l, e=eps_viz, s=step_viz, kk=K: pgd_untargeted(m, x, l, kk, e, s)

            print(f"\n  [시각화] {variant} PGD, ε = {eps_viz}, k = {K}")
            visualize_pgd_attack(
                model=model,
                test_loader=test_loader,
                attack_fn=viz_fn,
                eps=eps_viz,
                device=device,
                dataset_name=dataset_name,
                class_names=class_names,
                is_targeted=is_targeted,
                n_samples=5,
            )

    # ==================================================================
    # 결과 요약 테이블
    # ==================================================================
    print("\n" + "=" * 65)
    print(f"  Problem 3: PGD 공격 성공률 (%)  [k={K}]")
    print("=" * 65)

    for variant in ['Targeted', 'Untargeted']:
        print(f"\n  [{variant} PGD]  성공 기준: "
              + ("pred == target" if variant == 'Targeted' else "pred != true_label"))
        header = f"  {'Dataset':<12}" + "".join(f"   ε={e:.2f}" for e in eps_values)
        print(header)
        print("  " + "-" * 55)
        for dataset_name, _, _, _ in configs:
            row = f"  {dataset_name:<12}"
            for eps in eps_values:
                row += f"  {results[(dataset_name, variant, eps)]:6.2f}%"
            print(row)

    print("\n" + "=" * 65)
    print("\n[완료] Problem 3 (PGD) 검증 완료")
    print(f"  결과 이미지: problem3/results/")


if __name__ == '__main__':
    main()
