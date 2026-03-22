# Anomaly Detection using One-Class Neural Networks

- **저자**: Raghavendra Chalapathy (University of Sydney, CMCRC), Aditya Krishna Menon (Data61/CSIRO, ANU), Sanjay Chawla (QCRI, HBKU)
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1802.06360

## 1. 논문 개요

### 연구 목표 및 문제 정의

본 논문의 핵심 목표는 복잡한 데이터셋에서 이상치(anomaly)를 탐지할 수 있는 **One-Class Neural Network (OC-NN)** 모델을 제안하는 것이다. 이 연구는 산업 현장에서 발생하는 다양한 이상 징후—예컨대 금융 사기, 네트워크 침입, 장비 고장 등—를 자동으로 감지하는 시스템을 개발하는 데 초점을 맞춘다.

### 연구 문제의 중요성

이상치 탐지는 라벨링되지 않은 테스트 데이터에서 비정상 패턴을 발견하는 unsupervised learning 작업으로, 실제 응용 분야에서 광범위하게 활용된다. 특히 자동주행 자동차의 adversarial 공격 탐지는 안전과 직결되는 중요한 문제이다. 기존 One-Class SVM (OC-SVM)은 고차원 복잡한 데이터에서 성능 저하를 보이는 반면, 딥러닝 기반 방법은 특징 추출과 이상치 탐지가 분리되어 있어 최적의 탐지 성능을 달성하기 어렵다는 한계가 존재한다.

## 2. 핵심 아이디어

### 중심 직관

OC-NN의 핵심 아이디어는 **OC-SVM의 목적 함수를 신경망 아키텍처에 직접 통합**하는 것이다. 기존 hybrid 접근법이 autoencoder로 특징을 추출한 후 별도의 이상치 탐지 알고리즘(OC-SVM 등)에 feeding하는 2단계 방식을 사용하는 것과 달리, OC-NN은 단일 end-to-end 모델에서 특징 학습과 이상치 경계 학습을 동시에 수행한다.

### 기존 접근법과의 차별점

| 구분 | Hybrid OC-SVM | OC-NN (제안 방법) |
|------|--------------|-------------------|
| 특징 학습 | 사전 정의된 CNN/VGGNet | OC-NN 목적 함수에 의해 구동 |
| 최적화 | 이원화 (분리된 파이프라인) | 통합된 목적 함수 |
| 특징 표현 | 태스크에 종속적이지 않음 | 이상치 탐지에 맞춤화 |

## 3. 상세 방법 설명

### 전체 파이프라인 구조

OC-NN의 학습 파이프라인은 다음 두 단계로 구성된다:

1. **Autoencoder 사전 학습**: 입력 데이터의 대표 특징을 추출하기 위해 deep autoencoder를 학습
2. **OC-NN 학습**: pre-trained encoder를 입력으로 사용하여 one-class 분류 목적 함수로 fine-tuning

### OC-NN 목적 함수

OC-NN은 하나의 은닉층을 가진 feedforward 네트워크로, 선형 또는 sigmoid 활성화 함수 $g(\cdot)$을 사용한다. 목적 함수는 다음과 같이 정의된다:

$$\min_{w, V, r} \frac{1}{2}\|w\|_2^2 + \frac{1}{2}\|V\|_F^2 + \frac{1}{\nu} \cdot \frac{1}{N} \sum_{n=1}^{N} \max\left(0, r - \langle w, g(Vx_n) \rangle\right) - r$$

여기서:
- $w$: 은닉층에서 출력층으로의 가중치 벡터
- $V$: 입력층에서 은닉층으로의 가중치 행렬
- $r$: 편향 (초평면의 위치)
- $\nu \in (0,1)$: 초평면으로부터의 거리 최대화와 경계 통과 허용 데이터 수 사이의 trade-off를 제어하는 파라미터

### 최적화 알고리즘

OC-NN은 **교대 최소화 (alternating minimization)** 방식으로 학습된다:

1. **단계 1**: $r$을 고정하고 $w$, $V$를 학습 (Standard Backpropagation)
2. **단계 2**: $w$, $V$를 고정하고 $r$을 최적화

**Theorem 1**: $w$와 $V$가 주어졌을 때, $r$의 최적값은 스코어 $\{\hat{y}_n\}_{n=1}^N$의 $\nu$-quantile이다. 여기서 $\hat{y}_n = \langle w, g(Vx_n) \rangle$이다.

### 의사결정 함수

학습이 완료되면, 각 데이터 포인트에 대해 다음 의사결정 스코어를 계산한다:

$$S_n = \hat{y}_n - r$$

- $S_n \geq 0$: 정상 데이터
- $S_n < 0$: 이상치

## 4. 실험 및 결과

### 실험 설정

**사용 데이터셋**:

| 데이터셋 | 설명 | 특징 수 |
|---------|------|--------|
| Synthetic | 190개 정상점 + 10개 이상치 (d=512) | 512 |
| MNIST | 단일 클래스를 정상으로 설정, 1% 이상치 포함 | 784 |
| CIFAR-10 | 단일 클래스를 정상으로 설정, 10% 이상치 포함 | 3072 |
| GTSRB | 정지 표지판 + Boundary Attack adversarial 샘플 | 3072 |

**비교 대상 방법**:
- Shallow: OC-SVM/SVDD, Kernel Density Estimation (KDE), Isolation Forest
- Deep: DCAE, AnoGAN, Soft/One-Class Deep SVDD, RCAE

### 주요 정량적 결과

**MNIST 데이터셋** (AUC, %):

| 클래스 | OC-SVM | Deep SVDD | OC-NN | RCAE |
|-------|--------|-----------|-------|------|
| 0 | 96.75 | 98.66 | 97.78 | **99.92** |
| 2 | 79.34 | 88.09 | 87.32 | **98.01** |
| 4 | 94.18 | 93.88 | 93.25 | **99.23** |
| 8 | 88.65 | 90.69 | 88.54 | **98.50** |

**CIFAR-10 데이터셋** (AUC, %):

| 클래스 | OC-SVM | Deep SVDD | OC-NN | RCAE |
|-------|--------|-----------|-------|------|
| Bird | 63.47 | 61.99 | **63.66** | 71.67 |
| Deer | 69.15 | 63.36 | **67.40** | 72.75 |
| Frog | **71.57** | 63.93 | 63.31 | 64.88 |

### 실험 결과 분석

1. **RCAE가 전체적으로 최고 성능**: Reconstruction error 기반 RCAE가 대부분의 데이터셋에서 최상의 AUC를 달성
2. **OC-NN의 강점**: 전역 대비가 약한 클래스(Bird, Deer, Automobile)에서 shallow 방법보다 우수한 성능
3. **Shallow 방법의 강점**: 전역 구조가 명확한 클래스(Frog, Truck)에서는 OC-SVM이 효과적
4. **GTSRB adversarial 탐지**: RCAE가 adversarial boundary attack에 대해 가장 robust한 성능

## 5. 강점, 한계

### 강점

- **통합된 목적 함수**: 특징 학습과 이상치 경계 학습이 단일 목적 함수로 통합되어 end-to-end 최적화 가능
- **전환 학습 친화성**: 사전 학습된 encoder를 활용하여 다양한 도메인에 적용 가능
- **이론적 기반**: OC-SVM의 이론적 특성을 신경망에 확장하여 $\nu$-quantile 해석 가능
- **복잡한 데이터 처리**: CIFAR-10, GTSRB와 같은 고차원 이미지 데이터에서 효과적인 성능

### 한계 및 미해결 질문

- **비볼록 최적화**: 목적 함수가 비볼록(non-convex)이므로 전역 최적값 수렴이 보장되지 않음
- **아키텍처 의존성**: 성능이 네트워크 구조 선택에 민감하게 의존
- **재구성 능력이 없음**: OC-NN은 이상치 탐지를 위한 scoring만 제공하며, RCAE와 달리 재구성 기반 해석이 불가능
- **정지 표지판 특화**: GTSRB 실험은 단일 클래스에 한정되어 범용성 검증 필요
- **실시간 응용 미검증**: 실시간 탐지 시나리오에서의 계산 효율성 평가 미흡

## 6. 결론

### 주요 기여 사항

본 논문은 OC-SVM의 one-class 목적 함수를 신경망에 통합한 OC-NN 모델을 제안하였다. 이 접근법은 특징 표현 학습이 이상치 탐지 목적 함수에 의해 직접 구동된다는 점에서 기존 hybrid 방법론과 근본적으로 다르다. 실험 결과, OC-NN은 전역 대비가 약한 복잡한 이미지 데이터셋에서 기존 deep learning 방법보다 우수한 성능을 보였으며, 특히 자동주행 분야의 adversarial 공격 탐지 가능성을 시연하였다.

### 향후 연구 방향

- 더 깊은 아키텍처에서의 OC-NN 확장
- 재구성 기반 방법과 OC-NN의 결합
- 비지도 학습에서 부분 지도 학습으로의 확장
- 실시간 응용을 위한 모델 경량화
