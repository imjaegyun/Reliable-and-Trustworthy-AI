"""
attacks.py — Problem 2: Untargeted FGSM

Problem 1 (Targeted)과의 핵심 차이:
  - Targeted   : 특정 target 클래스 손실을 최소화 → x_adv = x - ε·sign(∇L)
  - Untargeted : 정답 클래스 손실을 최대화        → x_adv = x + ε·sign(∇L)

즉, 부호(sign)만 반대입니다.

참고 논문:
  Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015.
  https://arxiv.org/abs/1412.6572
"""

import torch
import torch.nn as nn


def fgsm_untargeted(model, x, label, eps):
    """
    Problem 2: Untargeted FGSM (Fast Gradient Sign Method) 공격.

    목표: 모델이 어떤 클래스든 정답이 아닌 클래스로 예측하도록 유도.
          어느 클래스로 오분류되든 상관없음.

    공식:
        x_adv = clamp(x + ε · sign(∇_x L(f(x), y_true)), 0, 1)

    step-by-step (과제 PDF 기준):
        1. x에 대한 모델 출력(logits) 계산
        2. 정답 레이블(label) 기준 손실 계산 (CrossEntropy)
        3. 역전파 → ∇_x L(f(x), y_true) 계산
        4. x_adv = x + ε · sign(∇_x L)   ← PLUS 부호: 손실 최대화
        5. [0, 1] 범위로 클리핑

    Problem 1과의 비교:
        Problem 1 (Targeted):   x - ε·sign(∇L)  → 손실 최소화 → target 쪽으로 당김
        Problem 2 (Untargeted): x + ε·sign(∇L)  → 손실 최대화 → 정답에서 멀어짐

    Args:
        model : 공격 대상 신경망
        x     : 입력 이미지 텐서, shape (B, C, H, W), 값 범위 [0, 1]
        label : 정답(올바른) 클래스 레이블
                - int       : 모든 배치에 동일 레이블
                - LongTensor (B,): 이미지별 개별 레이블
        eps   : perturbation 크기 (예: 0.05, 0.1, 0.2, 0.3)

    Returns:
        x_adv : 적대적 이미지, shape (B, C, H, W), 값 범위 [0, 1]
    """
    # 평가 모드: Dropout 비활성화, BatchNorm running stats 사용
    model.eval()
    device = next(model.parameters()).device

    # 원본 이미지를 device로 이동, 기존 계산 그래프와 분리
    x = x.clone().detach().to(device)

    # label 처리: 정수이면 배치 전체에 동일한 레이블 tensor 생성
    if isinstance(label, int):
        label_tensor = torch.full(
            (x.size(0),), fill_value=label, dtype=torch.long, device=device
        )
    else:
        label_tensor = label.clone().detach().to(device)

    # Step 1: gradient 계산을 위해 requires_grad 활성화
    x.requires_grad_(True)

    # Step 2: 순전파 + 정답 레이블 기준 손실 계산
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, label_tensor)

    # Step 3: 역전파 → x.grad에 ∇_x L 저장
    loss.backward()

    with torch.no_grad():
        # Step 4: PLUS 부호 — 정답 클래스 손실을 최대화하는 방향으로 이동
        #         gradient 방향으로 이동 → 정답 클래스 확률 감소
        x_adv = x + eps * x.grad.sign()

        # Step 5: 유효 픽셀 범위 [0, 1] 클리핑
        x_adv = torch.clamp(x_adv, min=0.0, max=1.0)

    return x_adv.detach()
