# Explainable Deep One-Class Classification

- **저자**: Philipp Liznerski, Lukas Ruff, Robert A. Vandermeulen, Billy Joe Franks, Marius Kloft, Klaus-Robert Müller
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2007.01760

## 1. 논문 개요

본 논문의 목표는 이상 감지(anomaly detection)를 위한 새로운 설명 가능한(deep explainable) 딥러닝 방법을 제안하는 것이다. 특히 Deep Support Vector Data Description(DSVDD)과 같은 deep one-class classification 방법들이 높은 비선형 변환을 학습하기 때문에 결과에 대한 해석이 어렵다는 문제를 해결하고자 한다.

연구의 핵심 문제는 deep one-class classification 모델이 비정상 표본을 탐지할 수는 있지만, 왜 특정 영역이 비정상이라고 판단했는지를 설명할 수 없다는 점이다. 산업 현장에서는 안전과 보안 요구사항을 충족하기 위해 탐지 결과에 대한 해석이 필수적이다.

연구의 중요성은 제조 결함 탐지, 암 검출, 품질 관리 등 실제 응용 분야에서 모델의 결정을 인간 전문가가 검토하고 신뢰할 수 있어야 하기 때문이다. 특히 이상 탐지 결과를 pixel 단위로 해석할 수 있다면, 결함의 위치와 범위를 직접 식별할 수 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 Fully Convolutional Data Description(FCDD)라는 새로운 방법을 제안하는 것이다. FCDD는 변환된 샘플 자체가 이상 발열도(anomaly heatmap)가 되도록 설계되어, 탐지 성능과 설명 가능성을 동시에 달성한다.

기존 접근 방식과의 주요 차별점은 다음과 같다. DSVDD와 같은 기존 방법들은 특징 공간에서 nominally 데이터를 중심으로 집중시키지만, 출력의 공간적 정보를 보존하지 않아 해석이 어렵다. FCDD는 convolutional과 pooling 레이어만 사용하여 각 출력 픽셀의 수용 영역(receptive field)을 제한함으로써, 출력 특징이 다운샘플링된 이상 발열도가 되도록 한다. 이는 Reconstruction 기반 방법(Autoencoder)과 달리 OE(Outlier Exposure) 샘플을 자연스럽게 통합할 수 있다.

## 3. 상세 방법 설명

### 3.1 Deep One-Class Classification (Hypersphere Classifier)

Deep one-class classification은 신경망을 학습시켜 nominally 데이터를 사전 정의된 중심점 $\mathbf{c}$ 근처에 매핑하고, 이상 데이터를远离된 위치에 매핑하는 방식으로 이상 감지를 수행한다. Hypersphere Classifier(HSC)의 목적 함수는 다음과 같다:

$$\min_{\mathcal{W}, \mathbf{c}} \frac{1}{n} \sum_{i=1}^{n} (1-y_i) h(\phi(X_i; \mathcal{W}) - \mathbf{c}) - y_i \log\left(1 - \exp\left(-h(\phi(X_i; \mathcal{W}) - \mathbf{c})\right)\right)$$

여기서 $h$는 pseudo-Huber 손실 함수 $h(\mathbf{a}) = \sqrt{\|\mathbf{a}\|_2^2 + 1} - 1$이며, outliers에 대해 robust한 손실이다.

### 3.2 Fully Convolutional Architecture

FCDD는 Fully Convolutional Network(FCN)을 사용한다. FCN은 convolutional 레이어와 pooling 레이어만으로 구성되어 공간 정보를 보존하며, $\phi: \mathbb{R}^{c \times h \times w} \to \mathbb{R}^{1 \times u \times v}$로 매핑한다. 각 출력 픽셀의 수용 영역은 입력 이미지의 특정 영역에 대응되어, 출력과 입력의 공간적 대응 관계가 유지된다.

### 3.3 FCDD 목적 함수

FCN $\phi$와 pseudo-Huber 손실을 사용하여 $A(X) = (\sqrt{\phi(X; \mathcal{W})^2 + 1} - 1)$를 출력 행렬로 정의한다. FCDD 목적 함수는 다음과 같다:

$$\min_{\mathcal{W}} \frac{1}{n} \sum_{i=1}^{n} (1-y_i) \frac{1}{u \cdot v} \|A(X_i)\|_1 - y_i \log\left(1 - \exp\left(-\frac{1}{u \cdot v} \|A(X_i)\|_1\right)\right)$$

여기서 $\|A(X)\|_1$은 행렬 모든 원소의 합이며, 이를 이상 점수로 사용한다. $\|A(X)\|_1$ 값이 큰 원소는 입력 이미지의 이상 점수에 기여하는 영역에 해당한다.

### 3.4 Receptive Field Upsampling

저해상도 발열도 $A(X)$를 원본 이미지 해상도로 업샘플링하기 위해, 각 출력 픽셀이 입력 이미지의 수용 영역 중심에 Gaussian 분포로 영향을 미친다는 사실을 이용한다. 이를 strided transposed convolution with Gaussian kernel으로 구현하며, 커널 크기는 FCDD의 수용 영역 범위로, stride는 누적 stride로 설정한다.

### 3.5 Semi-supervised FCDD

소수의 알려진 이상 샘플과 해당 이상 맵을 학습에 활용할 수 있다. Pixel-wise 목적 함수는 다음과 같다:

$$\min_{\mathcal{W}} \frac{1}{n} \sum_{i=1}^{n} \left(\frac{1}{m} \sum_{j=1}^{m} (1-(Y_i)_j) A'(X_i)_j\right) - \log\left(1 - \exp\left(-\frac{1}{m} \sum_{j=1}^{m} (Y_i)_j A'(X_i)_j\right)\right)$$

여기서 $Y_i$는 ground-truth 이상 맵, $A'(X_i)$는 업샘플링된 예측 발열도이다.

## 4. 실험 및 결과

### 4.1 표준 이상 감지 벤치마크

Fashion-MNIST, CIFAR-10, ImageNet 데이터셋에서 one-vs-rest 설정으로 평가하였다. Outlier Exposure(OE)로 CIFAR-100, EMNIST, ImageNet22k를 각각 사용하였다.

주요 정량적 결과는 다음과 같다:

| 데이터셋      | AE   | DSVDD | GEO  | GEO+ | Deep SAD | HSC  | FCDD |
| ------------- | ---- | ----- | ---- | ---- | -------- | ---- | ---- |
| Fashion-MNIST | 0.82 | 0.93  | 0.94 | -    | -        | -    | 0.89 |
| CIFAR-10      | 0.59 | 0.65  | 0.86 | 0.90 | 0.95     | 0.96 | 0.95 |
| ImageNet      | 0.56 | -     | -    | -    | 0.97     | 0.97 | 0.94 |

FCDD는 제한된 FCN 아키텍처를 사용함에도 불구하고 최첨단 방법과 유사한 성능을 달성하였다.

### 4.2 MVTec-AD 제조 결함 탐지

MVTec-AD 데이터셋에서 ground-truth 이상 세그멘테이션 맵을 제공한다. 비지도 설정에서 FCDD는 0.92의 pixel-wise mean AUC를 달성하여 state-of-the-art를 설정하였다. 반지도(semi-supervised) 설정에서 클래스당 결함 유형당 단 1개의 이상 샘플만 사용해도 성능이 0.96으로 향상되었다.

### 4.3 Clever Hans 효과 분석

PASCAL VOC 데이터셋에서 "horse" 클래스를 이상으로 설정하고 실험한 결과, 모델이 이미지의 워터마크에 집중하는 Clever Hans 효과가 관찰되었다. 이는 deep one-class classification 모델이 spurious features에 취약할 수 있음을 보여주며, FCDD의 투명성이 이러한 문제를 식별하고 해결하는 데 도움이 된다.

## 5. 강점, 한계

### 5.1 강점

본 논문의 강점은 다음과 같다. 탐지 성능과 설명 가능성을 동시에 달성한다. FCDD는 CIFAR-10, ImageNet에서 최첨단 방법과 유사한 AUC를 달성하면서 해석 가능한 발열도를 제공한다. Semi-supervised 학습을 자연스럽게 지원한다. 소수의 레이블된 이상 샘플(5개 정도)만으로도 성능이 크게 향상된다. MVTec-AD에서 새로운 state-of-the-art를 달성하였다(0.92 → 0.96 with semi-supervised). Reconstruction 기반 방법과 달리 OE 샘플을 통합할 수 있다. 투명한 탐지 결과를 제공하여 모델이 잘못된 특징에 집중하는 것을 식별할 수 있다.

### 5.2 한계 및 미해결 질문

본 연구의 한계는 다음과 같다. 수용 영역 크기에 따라 발열도의 해상도가 결정되므로, 매우 미세한 이상을 탐지하기 어려울 수 있다. Semi-supervised 설정에서는 ground-truth 이상 맵을 필요로 한다. 모델이 adversarial attack에 취약할 수 있으며, 이에 대한 분석은 향후 작업으로 남아 있다. Synthetic anomalies( confetti noise)의 효과는 데이터셋 특성에 따라 달라질 수 있다.

## 6. 결론

본 논문의 주요 기여는 FCDD라는 설명 가능한 deep one-class classification 방법을 제안한 것이다. FCDD는 fully convolutional 아키텍처를 활용하여 변환된 샘플 자체가 이상 발열도가 되도록 설계되었다. 이 방법은 탐지 성능과 설명 가능성을 동시에 달성하며, semi-supervised 설정을 통해 소수의 레이블된 이상만으로도 성능을 크게 향상시킬 수 있다.

향후 연구 방향으로, adversarial attack에 대한 취약성 분석, 더 미세한 이상 탐지를 위한 방법 개선, 그리고 실제 산업 응용에서의 검증이 있다. FCDD의 투명한 탐지 결과는 산업 현장에서의 안전 요구사항 충족과 모델 신뢰성 향상에 기여할 수 있다.
