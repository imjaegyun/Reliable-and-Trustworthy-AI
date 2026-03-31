"""
attacks.py — Problem 3: PGD (Projected Gradient Descent)

PGD는 FGSM을 k번 반복하는 iterative 공격입니다.
한 번에 큰 eps로 이동하는 FGSM과 달리,
작은 eps_step으로 여러 번 이동하면서 매 스텝마다
원본 이미지 주변의 ε-ball 안으로 투영(projection)합니다.

Targeted / Untargeted 두 가지 모두 구현합니다:
  - pgd_targeted  : x - eps_step·sign(∇L)  (손실 최소화 → target으로 유도)
  - pgd_untargeted: x + eps_step·sign(∇L)  (손실 최대화 → 정답에서 멀어짐)

FGSM(Problem 1, 2)과의 관계:
  - FGSM     = k=1 인 PGD (단, eps_step = eps)
  - PGD k=40 = FGSM을 40번 반복 (훨씬 강력)

참고 논문:
  Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks",
  ICLR 2018. https://arxiv.org/abs/1706.06083
"""

import torch
import torch.nn as nn


# ============================================================
# 공통 유틸: 한 스텝 gradient 계산
# ============================================================

def _get_grad_sign(model, x_adv, label_tensor):
    """
    현재 x_adv에서 손실의 gradient 부호를 계산합니다.

    매 PGD 스텝마다 호출되어 gradient를 새로 계산합니다.
    .detach() → .requires_grad_(True) 패턴으로 이전 계산 그래프와 분리해
    메모리 누수(OOM)를 방지합니다.

    Args:
        model        : 공격 대상 신경망 (eval 모드여야 함)
        x_adv        : 현재 적대적 이미지 (B, C, H, W)
        label_tensor : 클래스 레이블 텐서 (B,)

    Returns:
        grad_sign : gradient의 부호, shape (B, C, H, W), 값 ∈ {-1, +1}
    """
    # 이전 스텝의 계산 그래프와 완전히 분리 후 gradient 활성화
    x_adv = x_adv.detach().requires_grad_(True)

    logits = model(x_adv)
    loss   = nn.CrossEntropyLoss()(logits, label_tensor)
    loss.backward()

    return x_adv.grad.sign()


# ============================================================
# Problem 3-A: Targeted PGD
# ============================================================

def pgd_targeted(model, x, target, k, eps, eps_step):
    """
    Problem 3: Targeted PGD 공격.

    목표: 모델이 특정 target 클래스로 예측하도록 k번 반복 공격.

    알고리즘 (PDF 기준):
        x_adv^(0) = x  (클린 입력에서 시작)
        for i = 1, ..., k:
            (a) x_adv^(i) = x_adv^(i-1) - eps_step · sign(∇_{x_adv} L)
                            ↑ MINUS: target 클래스 손실 최소화 (Problem 1과 동일 방향)
            (b) x_adv^(i) = clamp(x_adv^(i), x - eps, x + eps)   ← ε-ball 투영
            (c) x_adv^(i) = clamp(x_adv^(i), 0, 1)               ← 픽셀 범위 제한
        return x_adv^(k)

    FGSM(Problem 1)과의 차이:
        FGSM   : 한 번에 eps 크기로 이동 (k=1, eps_step=eps)
        PGD k=40: eps_step 씩 40번 이동 → ε-ball 경계 탐색, 훨씬 강력

    Args:
        model   : 공격 대상 신경망
        x       : 입력 이미지 (B, C, H, W), 값 범위 [0, 1]
        target  : 원하는 (잘못된) 클래스 레이블
                  - int       : 모든 배치에 동일 target
                  - LongTensor (B,): 이미지별 개별 target
        k       : 반복 횟수 (예: 40)
        eps     : 전체 perturbation 예산 (L∞ ball 반지름, 예: 0.3)
        eps_step: 스텝당 이동 크기 (예: 0.01)

    Returns:
        x_adv^(k): 최종 적대적 이미지 (B, C, H, W), 값 범위 [0, 1]
    """
    model.eval()
    device = next(model.parameters()).device

    x      = x.clone().detach().to(device)
    x_orig = x.clone()  # ε-ball 투영의 기준점 (원본 이미지, 변하지 않음)

    # target 처리
    if isinstance(target, int):
        target_tensor = torch.full(
            (x.size(0),), fill_value=target, dtype=torch.long, device=device
        )
    else:
        target_tensor = target.clone().detach().to(device)

    # 초기값: 클린 입력에서 시작 (PDF 기준)
    x_adv = x.clone()

    for i in range(k):
        # (a) gradient 부호 계산 후 MINUS 방향으로 eps_step 이동
        grad_sign = _get_grad_sign(model, x_adv, target_tensor)
        x_adv = x_adv.detach() - eps_step * grad_sign

        # (b) ε-ball 투영: 원본에서 eps 이상 벗어나지 않도록 제한
        delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = x_orig + delta

        # (c) 유효 픽셀 범위 [0, 1] 클리핑
        x_adv = torch.clamp(x_adv, min=0.0, max=1.0)

    return x_adv.detach()


# ============================================================
# Problem 3-B: Untargeted PGD
# ============================================================

def pgd_untargeted(model, x, label, k, eps, eps_step):
    """
    Problem 3: Untargeted PGD 공격.

    목표: 모델이 정답이 아닌 어떤 클래스로든 예측하도록 k번 반복 공격.

    알고리즘 (PDF 기준):
        x_adv^(0) = x
        for i = 1, ..., k:
            (a) x_adv^(i) = x_adv^(i-1) + eps_step · sign(∇_{x_adv} L)
                            ↑ PLUS: 정답 클래스 손실 최대화 (Problem 2와 동일 방향)
            (b) x_adv^(i) = clamp(x_adv^(i), x - eps, x + eps)
            (c) x_adv^(i) = clamp(x_adv^(i), 0, 1)
        return x_adv^(k)

    Args:
        model   : 공격 대상 신경망
        x       : 입력 이미지 (B, C, H, W), 값 범위 [0, 1]
        label   : 정답(올바른) 클래스 레이블
                  - int       : 모든 배치에 동일 레이블
                  - LongTensor (B,): 이미지별 개별 레이블
        k       : 반복 횟수 (예: 40)
        eps     : 전체 perturbation 예산 (예: 0.3)
        eps_step: 스텝당 이동 크기 (예: 0.01)

    Returns:
        x_adv^(k): 최종 적대적 이미지 (B, C, H, W), 값 범위 [0, 1]
    """
    model.eval()
    device = next(model.parameters()).device

    x      = x.clone().detach().to(device)
    x_orig = x.clone()

    # label 처리
    if isinstance(label, int):
        label_tensor = torch.full(
            (x.size(0),), fill_value=label, dtype=torch.long, device=device
        )
    else:
        label_tensor = label.clone().detach().to(device)

    x_adv = x.clone()

    for i in range(k):
        # (a) gradient 부호 계산 후 PLUS 방향으로 eps_step 이동
        grad_sign = _get_grad_sign(model, x_adv, label_tensor)
        x_adv = x_adv.detach() + eps_step * grad_sign

        # (b) ε-ball 투영
        delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = x_orig + delta

        # (c) 픽셀 범위 클리핑
        x_adv = torch.clamp(x_adv, min=0.0, max=1.0)

    return x_adv.detach()
