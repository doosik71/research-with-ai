# Relational Autoencoder for Feature Extraction

Qinxue Meng, Daniel Catchpoole, David Skillicorn, and Paul J. Kennedy

## 🧩 Problem to Solve

고차원 데이터가 증가함에 따라 차원 축소 및 특징 추출의 중요성이 커지고 있습니다. 오토인코더(Autoencoder, AE)는 신경망 기반 특징 추출 방법으로 추상적인 특징을 생성하는 데 성공적이지만, 데이터 샘플 간의 관계를 명시적으로 고려하지 못한다는 한계가 있습니다. 기존의 관계 기반 오토인코더(Generalized Autoencoder, GAE)는 쌍별 거리 가중치 설정이 어렵고, 특정 관계를 과도하게 강조하여 모델이 지도 학습으로 변질될 위험이 있으며, 데이터 자체의 손실은 무시하고 관계 손실에만 초점을 맞춰 정보 손실을 초래할 수 있습니다. 본 논문은 이러한 문제점을 해결하여 데이터 특징과 관계를 모두 고려하는 강건한 특징 추출 방법을 제안합니다.

## ✨ Key Contributions

- 데이터 특징과 그 관계 모두의 손실을 최소화하여 이를 동시에 고려하는 새로운 Relational Autoencoder(RAE) 모델을 제안합니다.
- 데이터 관계를 고려할 때 발생하는 계산 복잡성을 완화하기 위해 활성화 함수를 도입하여 약하고 사소한 관계를 필터링합니다.
- 제안된 관계 모델을 Sparse Autoencoder(SAE), Denoising Autoencoder(DAE), Variational Autoencoder(VAE)를 포함한 다른 주요 오토인코더 모델에 확장하여 적용 가능함을 보입니다.

## 📎 Related Works

- **차원 축소 방법:**
  - **특징 선택(Feature Selection):** 전체 특징의 부분 집합을 사용 (예: Subset Selection, Random Forest).
  - **특징 추출(Feature Extraction):** 기존 특징들의 조합으로 새로운 특징 집합을 생성 (예: PCA, LDA).
- **초기 특징 추출 방법:**
  - **선형 투영:** Principal Component Analysis(PCA), Linear Discriminant Analysis(LDA).
  - **비선형 투영:** 커널 함수를 활용한 방법들.
- **데이터 관계를 활용하는 차원 축소:**
  - Multidimensional Scaling(MDS), ISOMAP, Locally Linear Embedding(LLE), Laplacian Eigenmaps(LE).
  - 한계: 고차원 공간에서 지역적 데이터 관계를 포착하는 방식이 고정되어 있거나 미리 정의되어 있어 저차원 공간에서 정확하지 않을 수 있으며, 특징을 한 번에 추출하는 방식이 깊은 표현 학습에 부적합.
- **오토인코더(Autoencoders):**
  - **기본 오토인코더(AE):** 재구성 손실 최소화를 통해 차원 축소. 계층을 쌓아 깊은 구조 학습 가능.
  - **확장 모델:** Sparse Autoencoder(SAE), Denoising Autoencoder(DAE), Contractive Autoencoder(CAE), Variational Autoencoder(VAE).
  - **관계 기반 오토인코더:** Generalized Autoencoder(GAE) [26]는 데이터 관계 재구성에 초점. 하지만 거리 가중치 설정의 어려움과 데이터 자체의 손실을 무시할 수 있다는 문제점이 있음.

## 🛠️ Methodology

본 논문에서는 데이터 특징과 관계를 모두 고려하는 Relational Autoencoder(RAE)를 제안합니다.

- **RAE의 목적 함수:**
  데이터 재구성 손실과 관계 재구성 손실을 가중 평균하여 최소화합니다.
  $$ \Theta = (1-\alpha) \min*{\theta} L(X,X') + \alpha \min*{\theta} L(R(X),R(X')) $$
    여기서 $X$는 원본 데이터, $X'$는 재구성된 데이터, $R(X)$는 $X$의 데이터 샘플 간 관계, $R(X')$는 $X'$의 데이터 샘플 간 관계를 나타냅니다. $\alpha$는 데이터 재구성 손실과 관계 재구성 손실의 가중치를 조절하는 스케일 파라미터입니다.

- **데이터 관계 모델링:**
  데이터 관계 $R(X)$는 데이터 $X$와 그 전치 행렬 $X^{T}$의 곱인 $X X^{T}$로 모델링하여 샘플 간 유사도를 표현합니다.
  $$ \Theta = (1-\alpha) \min*{\theta} L(X,X') + \alpha \min*{\theta} L(X X^{T}, X' X'^{T}) $$

- **약한 관계 필터링:**
  계산 효율성을 높이고 불필요한 관계를 제거하기 위해 정류 함수(rectifier function) $\tau_t$를 사용하여 특정 임계값 $t$보다 작은 유사도 값 ($r_{ij}$)을 0으로 만듭니다.
  $$ \tau*t(r*{ij}) = \begin{cases} r*{ij}, & \text{if } r*{ij} \geq t \\ 0, & \text{otherwise} \end{cases} $$
    따라서 RAE의 최종 목적 함수는 다음과 같습니다:
    $$ \Theta = (1-\alpha) \min*{\theta} L(X,X') + \alpha \min*{\theta} L(\tau_t(X X^{T}), \tau_t(X' X'^{T})) $$
    손실 함수 $L$로는 제곱 오차(squared error)를 사용합니다.

- **학습 절차:**
  RAE는 확률적 경사 하강법(SGD)을 사용하여 모델이 수렴할 때까지 반복적으로 학습됩니다.

- **기존 오토인코더 모델로의 확장:**
  - **Relational Sparse Autoencoder (RSAE):** RAE 목적 함수에 가중치 감소 정규화 항 $\lambda ||W||^{2}$을 추가합니다.
    $$ \Theta = (1-\alpha) \min*{\theta} L(X,X') + \alpha \min*{\theta} L(\tau_t(X X^{T}), \tau_t(X' X'^{T})) + \lambda ||W||^{2} $$
  - **Relational Denoising Autoencoder (RDAE):** 원본 데이터 $X$와 노이즈가 추가된 $\tilde{X}$ 간의 데이터 및 관계 재구성 손실을 최소화합니다. 노이즈는 가산 등방성 가우시안 노이즈($\tilde{X} = X + \Delta$, $\Delta \sim N(0, \delta^2)$)로 설정됩니다.
    $$ \Theta = (1-\alpha) \min*{\theta} L(X,g(f(\tilde{X}))) + \alpha \min*{\theta} L(\tau_t(X X^{T}), \tau_t(\tilde{X} \tilde{X}^{T})) \text{ s. t. } \tilde{X} \sim q(\tilde{X}|X) $$
  - **Relational Variational Autoencoder (RVAE):** VAE의 목적 함수에 데이터 관계에 대한 KL-divergence 항을 추가하여 잠재 변수 분포를 고려합니다.
    $$ \Theta = (1-\alpha) \min D*{KL}(q*{\phi}(Y|X)||p*{\theta}(X|Y)) + \alpha \min D*{KL}(q*{\phi}(Y|X X^{T})||p*{\theta}(X X^{T}|Y)) $$

## 📊 Results

- **데이터셋:** MNIST, CIFAR-10.
- **RAE, BAE, GAE 비교:**
  - RAE는 BAE(Basic Autoencoder)와 GAE(Generative Autoencoder)에 비해 더 낮은 재구성 손실을 달성합니다.
  - GAE는 BAE보다 손실이 적어 데이터 관계 고려의 이점을 확인했습니다.
  - RAE의 성능은 스케일 파라미터 $\alpha$에 따라 달라지지만, $\alpha$가 1(관계만 고려)인 경우에도 GAE보다 우수했는데, 이는 RAE가 약한 관계를 필터링하는 활성화 함수를 사용했기 때문입니다.
  - 분류 정확도에서 RAE는 MNIST 3.8%, CIFAR-10 12.7%로 가장 낮은 오차율을 보였습니다 (GAE: 5.7%, 14.9%; BAE: 8.9%, 15.6%).
  - RAE는 MNIST보다 CIFAR-10(더 복잡한 이미지)에서 더 나은 결과를 보여, 복잡한 데이터셋에서 데이터 관계 유지의 중요성을 시사합니다.
- **확장된 오토인코더 변형 모델과 원본 모델 비교:**
  - Relational Sparse Autoencoder(RSAE)는 Sparse Autoencoder(SAE)보다 재구성 손실 및 분류 오차율이 낮았습니다. (MNIST 기준: RSAE 0.296 손실, 1.8% 오차; SAE 0.312 손실, 2.2% 오차)
  - Relational Denoising Autoencoder(RDAE)는 Denoising Autoencoder(DAE)보다 더 나은 성능을 보였습니다. (MNIST 기준: RDAE 0.217 손실, 1.1% 오차; DAE 0.269 손실, 1.6% 오차)
  - Relational Variational Autoencoder(RVAE)는 Variational Autoencoder(VAE)보다 성능이 향상되었습니다. (MNIST 기준: RVAE 0.183 손실, 0.9% 오차; VAE 0.201 손실, 1.2% 오차)
  - 전반적으로, 데이터 관계를 고려한 오토인코더 변형 모델들은 재구성 손실을 줄이고 분류 성능을 향상시키는 데 기여했습니다.

## 🧠 Insights & Discussion

- 데이터 특징과 관계를 동시에 고려하는 것이 특징 추출 과정에서 더 강건하고 의미 있는 특징을 생성하는 데 매우 효과적입니다.
- 정류 함수를 사용하여 약하고 불필요한 관계를 필터링함으로써 계산 효율성을 높이고 모델 성능을 향상시킬 수 있습니다. 이는 GAE의 한계(가중치 설정의 어려움)를 해결하는 데 기여합니다.
- 데이터와 관계 재구성의 최적 균형(파라미터 $\alpha$)은 데이터셋에 따라 달라집니다.
- RAE 모델은 MNIST보다 CIFAR-10과 같은 더 복잡한 데이터셋에서 더 큰 성능 향상을 보여, 데이터 복잡성이 높을수록 관계 모델링의 이점이 커짐을 시사합니다.
- 제안된 관계 모델링 원리는 기존의 다양한 오토인코더 아키텍처(Sparse, Denoising, Variational)에 일반화하여 적용될 수 있습니다.
- 본 연구의 한계점으로는 주로 $X X^{T}$를 통한 유클리드 거리 기반 관계 모델링에 초점을 맞추었으며, 고정된 임계값 $t$를 사용하는 점이 있습니다. 향후 다른 관계 메트릭이나 적응형 필터링 방법이 탐구될 수 있습니다.

## 📌 TL;DR

기존 오토인코더가 데이터 관계를 무시하고, 이전 관계 기반 모델(GAE)은 가중치 설정 및 정보 손실 문제가 있었습니다. 본 논문은 데이터 특징과 관계 ($X X^{T}$로 모델링) 모두의 재구성 손실을 최소화하는 Relational Autoencoder(RAE)를 제안합니다. 약한 관계 필터링을 위한 활성화 함수를 도입하고, 이 원리를 Sparse, Denoising, Variational Autoencoder에 확장 적용합니다. 실험 결과, RAE와 그 관계형 변형 모델들은 MNIST 및 CIFAR-10 데이터셋에서 기존 모델들보다 일관되게 낮은 재구성 손실과 높은 분류 정확도를 달성하여, 데이터 관계 고려의 효과를 입증했습니다.
