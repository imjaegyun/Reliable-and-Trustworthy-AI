"""
attacks.py — 적대적 공격 구현

Problem 1: Targeted FGSM   → fgsm_targeted()
Problem 2: Untargeted FGSM → fgsm_untargeted()  
Problem 3: PGD              → pgd_targeted(), pgd_untargeted()  
참고 논문:
  - FGSM: Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015.
          https://arxiv.org/abs/1412.6572
  - PGD:  Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018.
          https://arxiv.org/abs/1706.06083
"""

import torch
import torch.nn as nn


# ============================================================
# Problem 1: Targeted FGSM
# ============================================================

def fgsm_targeted(model, x, target, eps):
    """
    Problem 1: Targeted FGSM (Fast Gradient Sign Method) 공격.

    목표: 모델이 입력 이미지를 특정 target 클래스로 분류하도록 perturbation 추가.

    Untargeted FGSM과의 핵심 차이:
      - Untargeted (+ε): 정답 클래스 손실을 최대화 → 아무 틀린 클래스로 오분류
      - Targeted   (-ε): target 클래스 손실을 최소화 → 특정 클래스로 유도

    공식:
        x_adv = clamp(x - ε · sign(∇_x L(f(x), y_target)), 0, 1)

    step-by-step (과제 PDF 기준):
        1. x에 대한 모델 출력(logits) 계산
        2. target 레이블 기준 손실 계산 (CrossEntropy)
        3. 역전파 → ∇_x L(f(x), y_target) 계산
        4. x_adv = x - ε · sign(∇_x L)   ← MINUS 부호: 손실 최소화
        5. [0, 1] 범위로 클리핑

    Args:
        model  : 공격 대상 신경망
        x      : 입력 이미지 텐서, shape (B, C, H, W), 값 범위 [0, 1]
                 (requires_grad는 내부에서 자동 설정)
        target : 원하는 (잘못된) 클래스 레이블
                 - int       : 배치 내 모든 이미지에 동일 target
                 - LongTensor (B,): 이미지별 개별 target
        eps    : perturbation 크기 (예: 0.05, 0.1, 0.2, 0.3)

    Returns:
        x_adv  : 적대적 이미지, shape (B, C, H, W), 값 범위 [0, 1]
    """
    # 평가 모드: Dropout 비활성화, BatchNorm은 running statistics 사용
    model.eval()
    device = next(model.parameters()).device

    # 원본 이미지를 device로 이동, 기존 계산 그래프와 분리
    x = x.clone().detach().to(device)

    # target 처리: 정수이면 배치 전체에 동일한 target tensor 생성
    if isinstance(target, int):
        target_tensor = torch.full(
            (x.size(0),), fill_value=target, dtype=torch.long, device=device
        )
    else:
        target_tensor = target.clone().detach().to(device)

    # Step 1: gradient 계산을 위해 requires_grad 활성화
    x.requires_grad_(True)

    # Step 2: 순전파 + target 레이블 기준 손실 계산
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, target_tensor)

    # Step 3: 역전파 → x.grad 에 ∇_x L 저장
    loss.backward()

    with torch.no_grad():
        # Step 4: MINUS 부호 — target 클래스 손실을 최소화하는 방향으로 이동
        #         gradient의 반대 방향 → target 클래스 확률 증가
        x_adv = x - eps * x.grad.sign()

        # Step 5: 유효 픽셀 범위 [0, 1] 클리핑
        x_adv = torch.clamp(x_adv, min=0.0, max=1.0)

    return x_adv.detach()


