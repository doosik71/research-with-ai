# Autoencoders

Dor Bank, Noam Koenigstein, Raja Giryes

## 🧩 Problem to Solve

오토인코더(Autoencoder)의 핵심 연구 문제는 입력 데이터를 압축되고 의미 있는 표현으로 인코딩한 다음, 원본과 최대한 유사하게 디코딩하여 재구성하는 것입니다. 이는 주로 비지도 학습(unsupervised learning) 방식으로 데이터의 유용한 잠재 표현(latent representation)을 학습하여 데이터 압축, 특징 추출, 클러스터링 등 다양한 후속 작업에 활용하는 것을 목표로 합니다.

수학적으로, 이 문제는 인코더 함수 $A: R^n \rightarrow R^p$와 디코더 함수 $B: R^p \rightarrow R^n$를 학습하여 재구성 손실 $E[\Delta(x - B(A(x)))]$를 최소화하는 것입니다. 여기서 $x$는 입력 데이터, $p$는 잠재 공간의 차원으로, 일반적으로 $p < n$인 경우가 많습니다.

## ✨ Key Contributions

- **Autoencoder의 기본 개념과 PCA와의 관계 정립**: 오토인코더가 입력 재구성을 위해 학습되는 신경망임을 명확히 하고, 비선형 매니폴드를 학습할 수 있는 주성분 분석(PCA)의 일반화된 형태임을 설명합니다.
- **다양한 정규화(Regularization) 기법 소개**: 스파스 오토인코더, 디노이징 오토인코더, 컨트랙티브 오토인코더 등 의미 있는 압축 표현 학습을 위한 여러 정규화 방법을 제시합니다.
- **변분 오토인코더(VAE)의 상세 분석**: 확률적 생성 모델로서 VAE의 이론적 기반(변분 베이즈 추론), 재매개변수화 트릭(reparameterization trick)을 통한 학습 과정, 그리고 disentangled VAE와 같은 확장 모델을 설명합니다.
- **Autoencoder의 광범위한 응용 분야 제시**: 생성 모델, 분류(classification), 클러스터링(clustering), 이상 탐지(anomaly detection), 추천 시스템(recommendation systems), 차원 축소(dimensionality reduction) 등 다양한 실제 활용 사례를 논의합니다.
- **최신 고급 Autoencoder 기술 소개**: 생성적 적대 신경망(GAN)과의 결합(Adversarially Learned Inference, Wasserstein Autoencoders), Deep Feature Consistent VAE, PixelCNN 디코더 등 재구성 품질 및 생성 능력 개선을 위한 발전된 기법들을 다룹니다.

## 📎 Related Works

- **Principal Component Analysis (PCA)**: Autoencoder가 비선형 PCA의 일반화로 설명됩니다.
- **Generative Adversarial Networks (GANs)**: VAE와 함께 주요 생성 모델로 비교되며, Autoencoder와 GAN을 결합한 Adversarial Autoencoders, Adversarially Learned Inference, Wasserstein Autoencoders 등이 논의됩니다.
- **Variational Bayes (VB) Inference**: VAE의 확률적 기반이 되는 추론 방법입니다.
- **L1 정규화($L_1$-regularization) 및 KL-Divergence**: 스파스 오토인코더의 정규화 기법으로 사용됩니다.
- **Dropout**: 디노이징 오토인코더에서 입력에 노이즈를 주입하는 한 가지 방법으로 언급됩니다.
- **K-means, Support Vector Machine (SVM)**: Autoencoder의 잠재 표현을 활용한 클러스터링 및 분류 응용에서 사용되는 알고리즘입니다.
- **Optimal Transport (Wasserstein distance)**: Wasserstein autoencoders (WAE)의 핵심 개념입니다.
- **PixelCNN**: 조건부 이미지 생성 및 Autoencoder 디코더로 활용되는 구조입니다.
- **Pretrained classification networks**: Deep Feature Consistent VAE에서 손실 함수 정의를 위해 사용됩니다.

## 🛠️ Methodology

1. **기본 Autoencoder 구조**:
   - **인코더 ($A$)**: 입력 $x \in R^n$를 압축된 잠재 표현 $z \in R^p$로 매핑합니다.
   - **디코더 ($B$)**: 잠재 표현 $z$를 원본 입력 $x$와 유사한 재구성된 출력 $\hat{x} \in R^n$로 매핑합니다.
   - **학습**: $x$와 $\hat{x}$ 사이의 재구성 손실(주로 $L_2$-norm)을 최소화하도록 엔드투엔드(end-to-end) 또는 계층별(layer-by-layer)로 학습됩니다.
2. **정규화된 Autoencoder**:
   - **스파스 Autoencoder**: 은닉 계층의 활성화에 $L_1$ 정규화 또는 KL-divergence 패널티를 추가하여 적은 수의 뉴런만 활성화되도록 강제하여 과적합을 방지하고 의미 있는 특징을 학습합니다.
   - **디노이징 Autoencoder**: 노이즈가 추가된 입력 $\tilde{x}$를 받아 원본 깨끗한 입력 $x$를 재구성하도록 학습하여 노이즈에 강인한 특징을 추출합니다.
   - **컨트랙티브 Autoencoder**: 인코더의 잠재 표현이 입력의 작은 변화에 덜 민감하도록 인코더의 야코비안(Jacobian) norm에 패널티를 부여합니다.
3. **변분 Autoencoder (VAE)**:
   - **확률적 생성 모델**: 데이터를 확률 분포를 통해 생성하는 모델로, 관측된 데이터 $x_i$에 대해 보이지 않는 잠재 변수 $z_i$에 조건화된 생성 모델 $p_{\theta}(x_i|z_i)$(디코더)와 잠재 변수 $z_i$의 근사 사후 분포 $q_{\phi}(z_i|x_i)$(인코더)를 학습합니다.
   - **목표 함수**: marginal log-likelihood의 변분 하한(ELBO, Variational Lower Bound) $L(\theta, \phi; x_i) = E_{q_{\phi}(z|x_i)}[\log p_{\theta}(x_i|z)] - D_{KL}(q_{\phi}(z|x_i) || p_{\theta}(z))$을 최대화합니다.
   - **재매개변수화 트릭**: 잠재 변수 $z$를 $z = g_{\phi}(\epsilon, x)$와 같이 보조 노이즈 변수 $\epsilon \sim p(\epsilon)$로부터 미분 가능한 변환으로 재구성하여, 샘플링 과정에서 그래디언트가 흐를 수 있도록 하여 역전파를 통한 학습을 가능하게 합니다.
   - **Disentangled VAE**: ELBO의 KL-divergence 항에 가중치 $\beta$를 곱하여 잠재 특징들의 독립성(분리성)을 조절합니다.
4. **고급 Autoencoder 기술**:
   - **Adversarially Learned Inference (ALI)**: GAN의 판별자를 오토인코더 구조에 통합하여 인코더-디코더와 판별자가 동시에 학습되며, 잠재 표현과 데이터 간의 결합 분포를 일치시키도록 학습하여 보다 사실적인 생성을 가능하게 합니다.
   - **Wasserstein Autoencoders (WAE)**: 재구성 손실과 잠재 공간 분포 간의 거리를 Wasserstein 거리로 측정하여 VAE의 고질적인 흐릿한 재구성 문제와 학습 안정성을 개선합니다.
   - **Deep Feature Consistent VAE**: 픽셀 수준의 $L_2$ 손실 대신 사전 학습된 신경망의 중간 계층 특징 간의 $L_2$ 거리를 재구성 손실로 사용하여 더 현실적인 이미지 재구성을 유도합니다.
   - **PixelCNN Decoders**: 디코더를 PixelCNN과 같은 순차적 픽셀 생성 모델로 구성하여 이미지의 지역 공간 통계를 고려한 선명하고 고품질의 이미지 생성을 가능하게 합니다.

## 📊 Results

- **의미 있는 잠재 표현 학습**: 정규화 기법을 통해 과적합 방지, 노이즈에 강인한 특징 추출, 그리고 입력 섭동에 덜 민감한 잠재 표현 학습이 가능합니다.
- **생성 능력**: VAE는 잠재 공간으로부터의 샘플링을 통해 원본 데이터와 유사한 새로운 데이터를 생성하는 능력(예: MNIST 데이터셋에서 고유한 숫자 이미지 생성)을 보여줍니다.
- **다양한 응용 분야에서의 성능 향상**:
  - **분류 및 클러스터링**: Autoencoder가 학습한 저차원, 고품질 특징은 후속 분류 또는 클러스터링 알고리즘의 성능을 향상시킵니다.
  - **이상 탐지**: 정상 데이터에 대해 낮은 재구성 오류를 보이고 이상 데이터에 대해 높은 재구성 오류를 보여 이상 샘플을 효과적으로 식별할 수 있습니다.
  - **추천 시스템**: AutoRec, CDAE 등은 사용자-아이템 선호도 행렬의 저차원 표현을 학습하여 추천의 정확도를 높입니다.
  - **차원 축소**: 선형 PCA보다 비선형 매니폴드를 학습하여 더욱 효과적인 차원 축소 성능을 제공합니다.
- **GAN과의 결합 및 고급 기술의 효과**: 기존 오토인코더의 한계였던 '흐릿한 이미지 생성' 문제를 해결하고, GAN과의 결합(ALI, WAE) 및 특징 기반 손실 함수, PixelCNN 디코더 등을 통해 훨씬 더 사실적이고 시각적으로 매력적인 데이터 생성(예: CelebA 데이터셋에서 얼굴 특징 제어) 및 재구성 품질 향상을 달성했습니다.

## 🧠 Insights & Discussion

오토인코더는 입력 데이터의 압축되고 의미 있는 표현을 비지도 학습 방식으로 학습하는 데 매우 강력한 모델입니다. 이는 PCA의 비선형적 일반화로서 데이터의 복잡한 매니폴드를 효과적으로 학습하여 차원 축소, 특징 추출, 데이터 압축 등의 기본 기능 외에도 생성 모델링, 분류, 클러스터링, 이상 탐지, 추천 시스템 등 광범위한 분야에서 활용됩니다.

그러나 초기 오토인코더는 픽셀 단위의 재구성 손실(예: MSE)로 인해 이미지 재구성 시 결과물이 종종 '흐릿'하다는 주요 단점을 가졌습니다. 또한, VAE의 경우 잠재 상태의 크기 및 분포를 선택하는 과정이 여전히 경험적이고 학습 후 실험에 의존하는 경향이 있습니다.

이러한 한계를 극복하기 위해 최근에는 GAN과의 결합(ALI, WAE), Deep Feature Consistent VAE와 같은 특징 기반 손실 함수, 그리고 PixelCNN 디코더와 같은 고급 구조가 개발되었습니다. 이들은 재구성 품질을 향상시키고 생성된 결과의 사실성 및 제어 가능성을 높이는 데 크게 기여했습니다. 향후 연구는 이러한 파라미터(잠재 상태의 크기 및 분포)를 더 효율적으로 설정하는 방법을 모색해야 할 것입니다.

궁극적으로 오토인코더의 목표는 '의미 있는 압축 표현'과 '우수한 재구성'이라는 두 가지 목표 사이의 균형을 찾는 것이며, 이를 위해 다양한 아키텍처와 학습 목표가 계속해서 발전하고 있습니다.

## 📌 TL;DR

**문제**: 오토인코더는 입력 데이터를 압축되고 의미 있는 잠재 표현으로 인코딩한 후 재구성하여, 비지도 학습 방식으로 데이터의 핵심 특징을 파악하고 다양한 다운스트림 작업에 활용하는 것을 목표로 합니다.
**방법**: 기본 오토인코더는 인코더와 디코더로 구성되며 재구성 손실을 최소화합니다. 스파스, 디노이징, 컨트랙티브 오토인코더는 정규화를 통해 강인한 특징을 학습합니다. 변분 오토인코더(VAE)는 확률적 생성 모델로서 재매개변수화 트릭을 이용해 학습하며 ELBO를 최적화합니다. 고급 기법으로는 GAN과의 결합(ALI, WAE), 특징 일관성 손실 함수, PixelCNN 디코더 등이 포함됩니다.
**주요 발견**: 오토인코더는 차원 축소, 특징 추출, 데이터 압축 및 생성 모델링(특히 VAE)에 효과적이며, 분류, 클러스터링, 이상 탐지, 추천 시스템 등 다양한 응용 분야에서 뛰어난 성능을 보입니다. 최신 발전된 기술들은 전통적인 오토인코더의 흐릿한 재구성 품질 문제를 해결하고, 생성 모델로서의 사실성과 제어 가능성을 크게 향상시켰습니다.
