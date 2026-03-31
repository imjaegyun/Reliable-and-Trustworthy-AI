"""
models.py — 신경망 모델 정의

이 파일에는 세 가지 구성 요소가 있습니다:
  1. NormalizeLayer : 모델 내부에서 입력 정규화 수행
  2. MNISTNet       : MNIST용 2층 CNN (처음부터 직접 구현)
  3. CIFAR10Net     : CIFAR-10용 ResNet-18 변형 (torchvision 기반)

핵심 설계 원칙:
  - 정규화(Normalize)는 DataLoader가 아닌 모델 내부에서 수행합니다.
  - DataLoader는 ToTensor()만 적용하여 픽셀 값을 [0, 1] 범위로 유지합니다.
  - 이렇게 하면 적대적 공격이 [0, 1] 공간에서 깔끔하게 동작합니다.
  - 모델 내부에서 정규화하면 denormalize 계산 없이도 perturbation을 올바르게 추가할 수 있습니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ============================================================
# NormalizeLayer
# ============================================================

class NormalizeLayer(nn.Module):
    """
    채널별 평균(mean)과 표준편차(std)로 입력 텐서를 정규화합니다.

    register_buffer를 사용하므로:
      - 모델과 함께 .to(device)로 자동 이동합니다.
      - model.state_dict()에 포함되어 저장/로드됩니다.
      - 학습 파라미터가 아니라 고정값입니다.

    Args:
        mean (list[float]): 채널별 평균값. 예: [0.1307] (MNIST) 또는 [0.4914, 0.4822, 0.4465] (CIFAR-10)
        std  (list[float]): 채널별 표준편차. 예: [0.3081] (MNIST) 또는 [0.2023, 0.1994, 0.2010] (CIFAR-10)
    """

    def __init__(self, mean: list, std: list):
        super().__init__()
        # view(-1, 1, 1)로 브로드캐스팅 가능한 형태로 변환: (C, 1, 1)
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1))
        self.register_buffer('std',  torch.tensor(std,  dtype=torch.float32).view(-1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 입력 x는 [0, 1] 범위의 텐서 (B, C, H, W)
        return (x - self.mean) / self.std


# ============================================================
# MNISTNet — MNIST 분류기 (처음부터 직접 구현)
# ============================================================

class MNISTNet(nn.Module):
    """
    MNIST 손글씨 숫자 분류를 위한 2층 CNN.

    아키텍처:
        [입력] (B, 1, 28, 28) in [0, 1]
            ↓ NormalizeLayer (mean=0.1307, std=0.3081)
            ↓ Conv2d(1, 32, kernel=3)   → (B, 32, 26, 26)
            ↓ ReLU
            ↓ Conv2d(32, 64, kernel=3)  → (B, 64, 24, 24)
            ↓ ReLU
            ↓ MaxPool2d(2)              → (B, 64, 12, 12)
            ↓ Dropout2d(p=0.25)
            ↓ Flatten                  → (B, 9216)   [= 64 × 12 × 12]
            ↓ Linear(9216, 128)
            ↓ ReLU
            ↓ Dropout(p=0.5)
            ↓ Linear(128, 10)
        [출력] logits (B, 10)

    목표 정확도: ≥ 95% (clean test set)
    """

    def __init__(self):
        super().__init__()

        # 정규화 레이어 (입력은 [0,1], 출력은 MNIST 통계 기준 정규화)
        self.normalize = NormalizeLayer(mean=[0.1307], std=[0.3081])

        # 합성곱 레이어
        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=32, kernel_size=3)  # 28→26
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)  # 26→24

        # 풀링 및 드롭아웃
        self.pool   = nn.MaxPool2d(kernel_size=2)   # 24→12
        self.drop1  = nn.Dropout2d(p=0.25)          # 채널 단위 드롭아웃 (feature map 전체를 0으로)
        self.drop2  = nn.Dropout(p=0.5)             # 뉴런 단위 드롭아웃

        # 완전 연결 레이어
        # Flatten 후 크기: 64 채널 × 12 × 12 = 9216
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)               # 10개 클래스 (0~9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1단계: 정규화
        x = self.normalize(x)

        # 2단계: 첫 번째 합성곱 블록
        x = F.relu(self.conv1(x))

        # 3단계: 두 번째 합성곱 블록 + 풀링 + 드롭아웃
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        # 4단계: Flatten (배치 차원 유지)
        x = torch.flatten(x, start_dim=1)   # (B, 9216)

        # 5단계: 완전 연결 레이어
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)                     # logits (B, 10)

        return x


# ============================================================
# CIFAR10Net — CIFAR-10 분류기 (ResNet-18 기반)
# ============================================================

class CIFAR10Net(nn.Module):
    """
    CIFAR-10 분류를 위한 ResNet-18 수정 버전.

    원본 ResNet-18은 224×224 입력에 맞게 설계되었습니다.
    CIFAR-10의 32×32 입력에 맞게 두 가지를 수정했습니다:
      1. conv1: 7×7 stride-2 → 3×3 stride-1 (공간 해상도 보존)
      2. maxpool: 제거 (nn.Identity() 대체) — 조기 spatial collapse 방지
         * 원본 maxpool을 유지하면 layer4 이후 feature map이 1×1로 축소되어
           BatchNorm이 제대로 동작하지 않음

    출처:
      - torchvision.models.resnet18
      - He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
        https://arxiv.org/abs/1512.03385
      - PyTorch 공식 문서:
        https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

    목표 정확도: ≥ 80% (clean test set), 일반적으로 ~93% 달성 가능

    Args:
        pretrained (bool): True이면 ImageNet 사전 학습 가중치 로드.
                           단, conv1이 수정되므로 해당 레이어 가중치는 폐기됨.
                           기본값 False (from scratch 학습 권장).
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        # CIFAR-10 채널별 통계 (학습 셋 기준)
        self.normalize = NormalizeLayer(
            mean=[0.4914, 0.4822, 0.4465],
            std= [0.2023, 0.1994, 0.2010]
        )

        # ResNet-18 기본 구조 로드
        if pretrained:
            from torchvision.models import ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            backbone = resnet18(weights=None)

        # 수정 1: conv1을 32×32에 맞게 변경
        # 원본: Conv2d(3, 64, kernel_size=7, stride=2, padding=3) → 112×112
        # 수정: Conv2d(3, 64, kernel_size=3, stride=1, padding=1) → 32×32 유지
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 수정 2: maxpool 제거 (32×32에서 공간 정보 손실 방지)
        backbone.maxpool = nn.Identity()

        # 수정 3: 출력 클래스 수를 10으로 변경 (원본은 ImageNet 1000)
        backbone.fc = nn.Linear(512, 10)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 정규화 후 ResNet 백본 통과
        return self.backbone(self.normalize(x))
