"""
train.py — 데이터 로더, 학습, 평가 유틸리티

이 파일은 순수 함수들로 구성됩니다 (전역 상태 없음):
  - get_mnist_loaders    : MNIST 데이터 로더 생성
  - get_cifar10_loaders  : CIFAR-10 데이터 로더 생성
  - train_model          : 모델 학습 (Adam + optional StepLR)
  - evaluate_model       : 클린 이미지에 대한 정확도 측정
  - evaluate_adversarial : 적대적 이미지에 대한 정확도 측정

중요: DataLoader 변환에는 ToTensor()만 사용합니다 (Normalize 없음).
      정규화는 모델 내부의 NormalizeLayer에서 처리합니다.
      이렇게 해야 공격이 [0, 1] 공간에서 올바르게 동작합니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


# ============================================================
# 데이터 로더
# ============================================================

def get_mnist_loaders(
    data_dir: str = './data',
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple:
    """
    MNIST 학습/테스트 데이터 로더를 반환합니다.

    변환:
      - ToTensor(): PIL Image를 [0, 1] 범위의 FloatTensor로 변환
      - Normalize는 적용하지 않음 (모델 내부에서 처리)

    데이터셋이 없으면 자동으로 다운로드합니다.

    Args:
        data_dir    : 데이터 저장 경로
        batch_size  : 배치 크기
        num_workers : DataLoader 워커 수

    Returns:
        (train_loader, test_loader): 학습용, 테스트용 DataLoader 튜플
    """
    # ToTensor만 사용: PIL [0,255] → Tensor [0,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        # 주의: Normalize를 여기서 적용하면 안 됨!
        # NormalizeLayer가 모델 내부에서 처리함
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # GPU 학습 시 데이터 전송 속도 향상
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_cifar10_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple:
    """
    CIFAR-10 학습/테스트 데이터 로더를 반환합니다.

    학습 변환 (데이터 증강 포함):
      - RandomCrop(32, padding=4): 랜덤 크롭으로 위치 불변성 학습
      - RandomHorizontalFlip(): 좌우 반전으로 데이터 다양성 증가
      - ToTensor(): [0, 1] 범위로 변환

    테스트 변환:
      - ToTensor(): 증강 없이 그대로 사용

    Args:
        data_dir    : 데이터 저장 경로
        batch_size  : 배치 크기 (CIFAR-10은 MNIST보다 이미지가 복잡해 128 사용)
        num_workers : DataLoader 워커 수

    Returns:
        (train_loader, test_loader): 학습용, 테스트용 DataLoader 튜플
    """
    # 학습용: 데이터 증강 적용
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # 패딩 4픽셀 후 32×32 크롭
        transforms.RandomHorizontalFlip(),           # 50% 확률로 좌우 반전
        transforms.ToTensor(),
        # 주의: Normalize를 여기서 적용하면 안 됨!
    ])

    # 테스트용: 증강 없이 ToTensor만
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


# ============================================================
# 학습 함수
# ============================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler_step: int = None,
    scheduler_gamma: float = 0.1,
    desc: str = "Training",
) -> list:
    """
    Adam 옵티마이저로 모델을 학습합니다.

    Args:
        model          : 학습할 신경망
        train_loader   : 학습 데이터 로더
        num_epochs     : 학습 epoch 수
        device         : 연산 장치
        lr             : 초기 학습률 (default: 1e-3)
        weight_decay   : L2 정규화 계수 (default: 1e-4)
        scheduler_step : StepLR의 step_size. None이면 스케줄러 미사용.
                         예: 20이면 20 epoch마다 lr을 gamma배로 감소
        scheduler_gamma: StepLR의 감소 비율 (default: 0.1)
        desc           : tqdm 진행바 prefix

    Returns:
        train_losses (list[float]): epoch별 평균 손실값 목록
    """
    model.to(device)
    model.train()

    # Adam: 적응적 학습률, 대부분의 경우 잘 동작
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # StepLR: scheduler_step epoch마다 lr *= gamma
    # CIFAR-10에서는 epoch 20에 lr 1e-3 → 1e-4로 감소 (성능 향상에 효과적)
    scheduler = None
    if scheduler_step is not None:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step, gamma=scheduler_gamma
        )

    criterion = nn.CrossEntropyLoss()
    train_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # tqdm으로 배치 단위 진행률 표시
        pbar = tqdm(train_loader, desc=f"{desc} Epoch {epoch:2d}/{num_epochs}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # 기울기 초기화
            optimizer.zero_grad()

            # 순전파
            logits = model(images)
            loss = criterion(logits, labels)

            # 역전파 및 파라미터 업데이트
            loss.backward()
            optimizer.step()

            # 통계 집계
            running_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 진행바에 현재 배치 손실 표시
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # epoch 평균 손실 및 정확도
        epoch_loss = running_loss / total
        epoch_acc  = 100.0 * correct / total
        train_losses.append(epoch_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  → Epoch {epoch:2d}/{num_epochs} | "
              f"Loss: {epoch_loss:.4f} | "
              f"Train Acc: {epoch_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

        # 스케줄러 업데이트 (epoch 끝에)
        if scheduler is not None:
            scheduler.step()

    return train_losses


# ============================================================
# 평가 함수
# ============================================================

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating",
) -> float:
    """
    클린(원본) 이미지에 대한 모델 정확도를 측정합니다.

    Args:
        model       : 평가할 신경망
        data_loader : 평가 데이터 로더
        device      : 연산 장치
        desc        : tqdm 진행바 prefix

    Returns:
        accuracy (float): 정확도 (0~100%)
    """
    model.eval()  # Dropout 비활성화, BatchNorm running stats 사용

    correct = 0
    total = 0

    with torch.no_grad():  # 평가 시 gradient 계산 불필요 → 메모리/속도 절약
        pbar = tqdm(data_loader, desc=desc)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            _, predicted = logits.max(1)

            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def evaluate_adversarial(
    model: nn.Module,
    data_loader: DataLoader,
    attack_fn,
    device: torch.device,
    max_batches: int = None,
    desc: str = "Adv Eval",
) -> float:
    """
    적대적 이미지에 대한 모델 정확도를 측정합니다.
    배치마다 attack_fn을 호출해 적대적 예제를 생성한 뒤 평가합니다.

    Args:
        model       : 공격 대상 및 평가 대상 신경망
        data_loader : 원본 테스트 데이터 로더
        attack_fn   : 공격 함수. 시그니처: (model, images, labels) → adv_images
                      functools.partial로 epsilon 등을 미리 바인딩하세요.
        device      : 연산 장치
        max_batches : 평가할 최대 배치 수. None이면 전체 데이터셋 평가.
                      빠른 데모를 위해 일부 배치만 사용할 때 지정.
        desc        : tqdm 진행바 prefix

    Returns:
        adv_accuracy (float): 적대적 정확도 (0~100%)
    """
    model.eval()  # 공격 및 평가 모두 eval 모드에서 수행

    correct = 0
    total = 0

    pbar = tqdm(data_loader, desc=desc, total=max_batches)

    for batch_idx, (images, labels) in enumerate(pbar):
        # max_batches 도달 시 조기 종료
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, labels = images.to(device), labels.to(device)

        # 적대적 예제 생성
        adv_images = attack_fn(model, images, labels)

        # 생성된 적대적 예제로 예측
        with torch.no_grad():
            logits = model(adv_images)
            _, predicted = logits.max(1)

        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    adv_accuracy = 100.0 * correct / total if total > 0 else 0.0
    return adv_accuracy
