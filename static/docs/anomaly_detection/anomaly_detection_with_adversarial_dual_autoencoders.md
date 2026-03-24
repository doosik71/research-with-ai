# Anomaly Detection with Adversarial Dual Autoencoders

- **저자**: Ha Son Vu, Daisuke Ueta, Kiyoshi Hashimoto, Kazuki Maeno, Sugiri Pranata, Sheng Mei Shen
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1902.06924

## 1. 논문 개요

이 논문은 anomaly detection, 특히 정상 데이터만 주로 확보할 수 있고 이상 데이터는 희귀하거나 미리 충분히 수집하기 어려운 상황을 다룬다. 저자들은 semi-supervised 또는 unsupervised setting에서 널리 쓰이는 GAN 기반 이상 탐지 방법들이 높은 표현력을 가지지만, 정작 GAN 학습 자체가 불안정하고 어렵다는 문제의식에서 출발한다. 대표적으로 generator와 discriminator의 역량 불균형 때문에 mode collapse나 non-convergence가 발생할 수 있고, 이로 인해 실제 anomaly detection 성능도 흔들릴 수 있다.

이 문제를 해결하기 위해 논문은 **Adversarial Dual Autoencoders (ADAE)** 라는 구조를 제안한다. 핵심은 generator와 discriminator를 서로 다른 종류의 네트워크로 두는 대신, **둘 다 autoencoder로 구성**하여 학습 안정성을 높이는 것이다. 즉, GAN의 대립적 학습 프레임은 유지하되, 양쪽 네트워크의 구조적 역량을 비슷하게 맞춰 불균형을 줄이겠다는 발상이다.

이 연구 문제가 중요한 이유는 실제 anomaly detection 응용에서 이상 샘플은 드물고 종류도 다양하기 때문이다. 따라서 supervised classification처럼 이상 클래스별로 충분한 라벨을 모으기 어렵다. 정상 데이터 분포를 잘 학습한 뒤, 거기서 벗어나는 입력을 이상으로 판정하는 접근이 매우 실용적이다. 논문은 이 아이디어를 MNIST, CIFAR-10 같은 benchmark뿐 아니라 **brain tumor detection**이라는 실제 의료 영상 문제에도 적용하여, 방법의 범용성과 실용성을 함께 보여주려 한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 크게 두 가지다.

첫째, **GAN의 generator와 discriminator를 모두 autoencoder로 구성한다**는 점이다. 전통적인 GAN에서는 discriminator가 보통 binary classifier 역할을 하며 real/fake를 직접 판별한다. 이런 구조는 generator와 discriminator의 학습 성격이 너무 달라 한쪽이 지나치게 강해지기 쉽다. 반면 ADAE에서는 두 네트워크 모두 입력을 재구성하는 autoencoder이므로, 표현 방식과 학습 난이도가 상대적으로 유사해진다. 저자들은 이 구조가 adversarial learning의 안정성을 높인다고 본다.

둘째, **anomaly score를 generator의 reconstruction error가 아니라 discriminator 쪽의 reconstruction error로 계산한다**는 점이다. 많은 기존 방법은 “정상 데이터만 학습했으니 이상 입력은 generator가 잘 복원하지 못할 것”이라는 직관에 따라 generator reconstruction error를 이상 점수로 사용한다. 하지만 이 논문은 한 단계 더 나아가, generator를 통과한 결과를 discriminator가 다시 복원하는 과정에서 이상 입력의 왜곡이 더 크게 드러난다고 본다. 즉, generator만 보는 것보다 **$D(G(\hat{x}))$까지 거친 최종 복원 오차**가 정상과 이상의 분리를 더 잘 만든다는 주장이다.

기존 접근과의 차별점은 다음처럼 정리할 수 있다. GANomaly나 EGBAD 같은 방법은 latent space consistency 또는 encoder–generator inverse mapping에 큰 비중을 둔다. 반면 ADAE는 latent representation 비교보다 **dual autoencoder의 adversarial reconstruction dynamics 자체**를 더 핵심 메커니즘으로 둔다. 또한 anomaly score 설계도 latent discrepancy가 아니라 discriminator reconstruction error에 두고 있다. 다시 말해, 이 논문은 “좋은 latent space를 어떻게 만들 것인가”보다 “안정적인 adversarial reconstruction 구조를 어떻게 만들고, 그 구조에서 이상 신호를 어디서 읽어낼 것인가”에 초점을 둔다.

## 3. 상세 방법 설명

### 3.1 전체 구조

ADAE는 두 개의 sub-network로 구성된다.

하나는 **generator $G$** 이고, 다른 하나는 **discriminator $D$** 이다. 그런데 둘 다 일반적인 GAN처럼 생성기와 판별기로 완전히 다른 형태를 쓰지 않고, **둘 다 autoencoder** 로 설계된다. 입력 이미지 $X$가 들어오면 generator는 먼저 이를 재구성하여 $G(X)$를 만든다. 그 다음 discriminator는 실제 입력 $X$ 또는 generator 출력 $G(X)$를 받아 다시 재구성한다.

이 구조에서 discriminator는 단순히 “진짜/가짜” 확률 하나를 내는 분류기가 아니라, 입력을 얼마나 잘 복원하는지로 real/generated를 구분하는 네트워크처럼 동작한다. 즉, 재구성 품질 자체가 판별 신호가 된다.

### 3.2 GAN 관점에서의 목적

논문은 일반적인 GAN의 목적을 다음처럼 상기한다.

$$
\min_G \max_D ; V(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] \mathbb{E}_{z \sim p_z(z)}[\log D(G(z))]
$$

하지만 ADAE는 discriminator가 classifier가 아니라 autoencoder이므로, adversarial objective를 **pixel-wise reconstruction error**로 다시 정의한다. 핵심 quantity는 다음과 같다.

$$
|G(X)-D(G(X))|_1
$$

이 항은 generator가 만든 결과 $G(X)$를 discriminator가 다시 복원했을 때의 오차이다. 여기서 discriminator는 이 오차를 **크게** 만들고 싶어 한다. 즉, 생성된 샘플은 잘 복원하지 못하게 만들려는 것이다. 반대로 generator는 이 오차를 **작게** 만들고 싶어 한다. 결국 generator는 discriminator가 “real distribution에서 온 것처럼” 받아들이는 출력을 만들도록 압박받는다.

### 3.3 손실 함수

논문은 discriminator와 generator의 학습 목표를 각각 다음과 같이 둔다.

#### Discriminator loss

$$
\mathcal{L}_D = |X-D(X)|_1 |G(X)-D(G(X))|_1
$$

이 식의 의미는 분명하다. 첫 번째 항 $|X-D(X)|_1$ 는 실제 정상 입력 $X$를 discriminator가 잘 복원하도록 유도한다. 이 값은 작아져야 한다. 두 번째 항은 생성된 입력 $G(X)$를 discriminator가 복원할 때의 오차인데, 앞에 마이너스가 붙어 있으므로 전체적으로는 이 값을 크게 만드는 방향으로 작동한다. 즉 discriminator는 **real input은 잘 복원하고, generated input은 못 복원하도록** 학습된다.

#### Generator loss

$$
\mathcal{L}_G = |X-G(X)|_1 + |G(X)-D(G(X))|_1
$$

첫 번째 항은 generator 자신이 정상 입력을 잘 복원하도록 한다. 두 번째 항은 generator 출력이 discriminator에서도 잘 복원될 수 있도록 만든다. 따라서 generator는 단순히 입력을 복사하는 것이 아니라, discriminator가 real distribution으로 받아들일 만한 출력을 생성하도록 유도된다.

논문은 reconstruction loss로 $\ell_1$ norm을 선택하는데, 이는 $\ell_2$보다 더 sharp한 결과를 주는 경향이 있다는 기존 image-to-image translation 연구의 통찰을 따른 것이다. 쉽게 말해, 픽셀 평균화로 흐릿해지는 현상을 줄이려는 선택이다.

### 3.4 네트워크 설계와 학습 안정화

저자들은 두 sub-network의 능력이 비슷하기 때문에 학습 안정성이 높아진다고 본다. 여기에 더해, generator와 discriminator의 상대적 성능 차이를 완화하기 위해 learning rate balancing 기법도 사용한다고 설명한다. 설명에 따르면 learning rate는 0을 중심으로 한 sigmoid 형태로 조절되며, 상대적으로 약한 네트워크가 강한 네트워크를 따라잡을 수 있도록 한다. 다만 정확한 balancing 공식은 본문에 자세히 제시되지 않았으므로, 구체 구현은 추출 텍스트만으로는 확인할 수 없다.

구조적으로는 fully connected layer 없이 **fully convolutional layers**만 사용한다. 이는 파라미터 수를 줄이고 이미지 구조를 유지하는 데 유리하다. 또한 pooling/upsampling 대신 stride를 가진 convolution/deconvolution을 사용하여 reconstruction 품질을 높인다. 이 역시 DCGAN 계열 설계 선택을 따른 것이다.

### 3.5 이상 점수 계산

이 논문의 가장 중요한 설계 중 하나는 anomaly score 정의다. 테스트 입력 $\hat{x}$에 대해 논문은 다음을 anomaly score로 쓴다.

$$
\mathcal{A}(\hat{x}) = |\hat{x} - D(G(\hat{x}))|_2
$$

즉, 입력을 generator에 통과시킨 뒤 discriminator까지 거친 최종 결과 $D(G(\hat{x}))$와 원래 입력 $\hat{x}$ 사이의 $\ell_2$ 오차를 측정한다.

이 선택의 직관은 다음과 같다. 정상 입력은 generator가 비교적 잘 복원할 수 있고, 그렇게 복원된 결과는 discriminator 입장에서도 정상 분포에 가깝기 때문에 다시 잘 복원된다. 반면 이상 입력은 generator 단계에서 이미 완전한 faithful reconstruction이 어렵고, 그 결과는 discriminator가 정상 분포에서 벗어난 것으로 인식하여 복원 실패가 더 커진다. 그 결과 **오차가 generator 단계에서 한 번, discriminator 단계에서 다시 증폭**된다.

논문은 이 점이 기존 방법들과 다르다고 명시한다. 많은 anomaly detection 방법은 generator reconstruction error만 사용하지만, ADAE는 discriminator reconstruction error를 쓰는 편이 정상과 이상의 score 분리가 더 잘 된다고 주장한다. 최종적으로 threshold $\phi$를 두고, 어떤 테스트 샘플 $\hat{x}_i$가

$$
\mathcal{A}(\hat{x}_i) > \phi
$$

이면 anomaly로 판단한다.

### 3.6 방법의 해석

이 방법을 직관적으로 해석하면, ADAE는 정상 데이터에 대해 “두 단계 재구성 체인”을 학습한다. 정상 데이터는 $X \to G(X) \to D(G(X))$ 경로를 지나도 원래 구조가 유지된다. 그러나 이상 데이터는 처음부터 정상 manifold에 잘 맞지 않으므로 generator가 정상적인 형태로 끌어당기거나 일부 abnormal pattern을 잃게 되고, 이어 discriminator가 이를 다시 정상/생성 분포의 기준에서 복원하는 과정에서 차이가 더욱 커진다. 결과적으로 최종 reconstruction residual이 anomaly signal이 된다.

## 4. 실험 및 결과

### 4.1 데이터셋과 설정

논문은 세 가지 유형의 데이터셋에서 실험한다.

첫째는 **MNIST**다. 각 실험에서 숫자 하나를 anomaly class로 두고, 나머지 아홉 클래스를 정상 데이터로 사용한다. 따라서 총 10개의 실험 설정이 존재한다. 훈련에는 정상 데이터의 80%를 사용하고, 테스트에는 남은 정상 데이터 20%와 anomaly class 전체를 사용한다.

둘째는 **CIFAR-10**이다. 설정은 MNIST와 유사하게 한 클래스를 이상으로 두고 나머지를 정상으로 둔다. 자연 이미지 데이터이므로 MNIST보다 훨씬 복잡한 분포를 가진다. 따라서 이 데이터셋에서의 성능은 모델의 일반성을 보는 더 까다로운 시험으로 볼 수 있다.

셋째는 의료 영상 응용으로 **HCP + BRATS 2017** 조합을 사용한다. HCP의 healthy brain MRI 65명 분량을 정상 학습 데이터로 쓰고, BRATS 2017 전체를 테스트에 사용한다. BRATS 2017은 HGG 210명, LGG 75명으로 구성되어 있다. 이때 목적은 pixel-wise segmentation이 아니라, 각 slice에 tumor가 존재하는지 여부를 판단하는 **slice-wise detection**이다.

전처리와 학습 설정도 제시된다. MNIST와 CIFAR-10은 이미지를 $[-1,1]$ 범위로 정규화하고 $32 \times 32 \times \text{colorChannels}$ 크기로 맞춘다. HCP/BRATS는 intensity variation을 줄이기 위해 z-score normalization을 사용하고, 각 slice를 square가 되도록 zero-padding한 뒤 $32 \times 32$로 resize한다. 학습은 batch size 64, Adam optimizer, $\alpha = 10^{-5}$, $\beta = 0.5$, 총 100 epoch로 수행한다.

latent dimension은 데이터셋에 따라 다르게 설정된다. MNIST는 32, CIFAR-10과 HCP/BRATS는 128을 사용한다. 저자들은 autoencoder의 표현 병목이 너무 작으면 중요한 정보가 손실되고, 너무 크면 단순 복사에 가까워져 불필요한 정보를 걸러내지 못한다고 설명한다.

### 4.2 평가 지표

평가는 주로 **AUROC**를 사용한다. 테스트 샘플마다 anomaly score를 계산하고, threshold를 바꾸어가며 TPR과 FPR을 구해 ROC curve를 만든 뒤, 그 면적인 AUROC를 측정한다. 논문은 anomaly detection에서 이상 샘플 비율이 낮아 class imbalance가 심한 경우가 많기 때문에 ROC 기반 지표가 적절하다고 본다.

의료 영상 실험에서는 각 slice에 tumor mask가 존재하면 anomaly, 없으면 normal로 이진 라벨링한다. 따라서 segmentation이 아니라 slice 단위 classification 성격의 평가다.

### 4.3 MNIST 결과

MNIST에서 ADAE는 평균 AUROC **0.858**을 기록한다. 표에 제시된 평균 성능은 다음과 같다.

- AnoGAN: 0.430
- EGBAD: 0.500
- GANomaly: 0.780
- IGMM-GAN: 0.852
- ADAE: 0.858

즉, ADAE는 GANomaly, EGBAD, AnoGAN보다 뚜렷하게 좋고, IGMM-GAN보다도 소폭 높은 평균 성능을 달성한다. 특히 class별로 보아도 대부분의 숫자 클래스에서 강한 결과를 보인다. 예를 들어 anomaly class가 2일 때 0.948, 5일 때 0.906, 8일 때 0.925로 높다. 반면 1이나 9처럼 상대적으로 어려운 경우는 각각 0.821, 0.631로 다른 클래스보다 낮다. 이는 anomaly class의 모양이 정상 클래스들과 얼마나 쉽게 구별되는지에 따라 난도가 다름을 시사한다.

저자들은 IGMM-GAN이 multimodal dataset에 특화된 방법임에도 불구하고 ADAE가 평균적으로 더 높다는 점을 강조한다. 이는 ADAE가 latent clustering을 명시적으로 하지 않더라도 정상 분포의 다중 모드 구조를 어느 정도 견딜 수 있음을 보여준다.

### 4.4 CIFAR-10 결과

CIFAR-10은 훨씬 더 어려운 데이터셋이다. 자연 이미지 클래스 사이에는 low-level texture나 shape가 서로 겹치는 부분이 많고, 한 클래스 내부 다양성도 크다. 이 환경에서 ADAE는 평균 AUROC **0.610**을 기록한다.

표의 평균값을 보면:

- AnoGAN: 0.434
- EGBAD: 0.462
- GANomaly: 0.605
- ADAE: 0.610

개선 폭은 MNIST보다 작지만, 기존 대비 최고 성능을 달성한다. 특히 Car 클래스 이상 탐지에서 0.729, Truck에서 0.697, Plane에서 0.633으로 비교적 높다. 반면 Deer에서 0.496으로 0.5에 근접해 어려움을 보인다. 이는 클래스 간 시각적 유사성과 intra-class variability가 높을수록 단순 reconstruction 기반 분리가 어렵다는 일반적 한계를 반영한다.

그럼에도 ADAE가 GANomaly를 소폭이나마 넘어선 것은 의미가 있다. CIFAR-10처럼 복잡한 분포에서도 dual autoencoder adversarial setup이 완전히 무너지지 않고 작동한다는 근거이기 때문이다.

### 4.5 BRATS 2017 뇌종양 탐지 결과

실제 응용 문제인 BRATS 2017 slice-wise tumor detection에서 ADAE는 **AUC 0.892**를 기록한다. 비교 대상은 다음과 같다.

- AE: 0.764
- VAE: 0.816
- ceVAE: 0.867
- ADAE: 0.892

의료 영상 anomaly detection에서는 정상 데이터만 학습했을 때, 종양이 있는 입력을 넣으면 모델이 종양 부위를 정상 조직처럼 복원하려고 하면서 residual이 발생하는 원리가 자주 사용된다. ADAE도 같은 큰 흐름에 속하지만, adversarial dual autoencoder 구조와 discriminator-based anomaly score 덕분에 더 나은 분리를 만든 것으로 해석된다.

논문은 qualitative 결과도 설명한다. 정상 뇌 이미지는 generator가 비교적 충실히 복원한다. 반면 종양이 있는 경우에는 모델이 “정상 뇌”를 복원하려고 하므로 종양 부위가 사라지거나 약화된 재구성이 나온다. 이로 인해 입력과 최종 재구성의 차이가 종양 영역에서 커진다. 또한 anomaly score histogram에서 normal과 anomaly가 잘 분리되어, 종양 존재 여부를 신뢰성 있게 탐지할 수 있음을 보였다고 설명한다.

### 4.6 결과 해석

실험 결과 전체를 종합하면, ADAE의 장점은 단순히 특정 데이터셋에서 숫자가 조금 높은 것이 아니라, **단순 benchmark부터 복잡한 자연 이미지, 실제 의료 영상까지 일관되게 강한 성능**을 보인다는 데 있다. 특히 MNIST와 CIFAR-10 모두에서 기존 GAN 기반 anomaly detection을 넘고, BRATS에서도 AE/VAE/ceVAE를 앞선다. 이는 제안한 구조가 특정 도메인 편향에만 기대지 않고 비교적 일반적인 anomaly detection framework로 작동할 가능성을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정과 방법 설계가 잘 맞물린다는 점이다. 저자들은 GAN의 불안정성이 anomaly detection 실전 적용을 가로막는 핵심 문제 중 하나라고 보고, 이를 해결하기 위해 generator와 discriminator를 모두 autoencoder로 통일했다. 이 아이디어는 단순하지만 설득력이 있다. 실제로 손실 함수도 reconstruction 중심으로 일관되게 정리되어 있어서, 왜 이 구조가 학습 안정성을 높일 수 있는지 이해하기 쉽다.

또 다른 강점은 anomaly score 설계가 명확하다는 점이다. 많은 reconstruction-based 방법은 “어디의 reconstruction error를 쓸 것인가”가 성능에 결정적이다. 이 논문은 discriminator reconstruction error가 더 잘 분리된다고 제안하고, benchmark와 medical dataset에서 모두 좋은 결과를 제시한다. 즉, architectural novelty와 scoring rule이 함께 작동한다는 점이 강하다.

실험 측면에서도 강점이 있다. 단일 toy dataset에 머무르지 않고 MNIST, CIFAR-10, BRATS 2017까지 평가하여 방법의 범용성을 보여주려 했다. 특히 brain tumor detection은 실제 활용 가능성을 보여주는 사례로서 의미가 있다.

반면 한계도 분명하다. 첫째, 본문에 따르면 learning rate balancing을 사용하지만, 그 **정확한 상대 성능 지표나 업데이트 공식**은 추출 텍스트에 충분히 제시되지 않았다. 따라서 재현성 관점에서는 구현 세부가 더 필요하다. 둘째, 성능 향상의 원인이 두 가지 요소, 즉 “dual autoencoder 구조”와 “discriminator-based anomaly score” 중 어디에서 얼마나 오는지에 대한 **ablation study**가 본문 추출 범위에서는 보이지 않는다. 예를 들어 같은 구조에서 generator reconstruction score를 썼을 때와의 직접 비교가 있었다면 기여를 더 명확히 분리할 수 있었을 것이다.

셋째, anomaly detection의 threshold $\phi$는 실제 배포에서 중요하지만, 논문 추출 텍스트에는 threshold 설정 방식이 구체적으로 설명되지 않는다. AUROC는 threshold-free metric이므로 연구 비교에는 적합하지만, 실제 운영에서는 어느 threshold에서 어떤 TPR/FPR trade-off를 선택할지가 중요하다. 이 부분은 후속 논의가 필요하다.

넷째, 의료 영상 실험은 **slice-wise detection**에 초점을 맞춘다. 따라서 실제 임상 활용에서 중요한 tumor localization 또는 pixel-wise segmentation 성능까지 입증한 것은 아니다. 저자도 이에 집중하지 않는다고 명시한다. 즉, “종양이 있는 슬라이스를 찾는 것”과 “종양 경계를 정확히 분할하는 것”은 다른 문제이며, 이 논문은 전자에 더 가깝다.

마지막으로, CIFAR-10 평균 AUROC 0.610은 SOTA라고 주장되지만 절대적 수치 자체는 여전히 높다고 보기는 어렵다. 복잡한 자연 이미지 anomaly detection이 본질적으로 어려운 문제임을 보여주는 동시에, ADAE 역시 복잡한 시각 분포에서는 한계가 남아 있음을 시사한다.

## 6. 결론

이 논문은 anomaly detection을 위한 GAN 기반 프레임워크로 **Adversarial Dual Autoencoders (ADAE)** 를 제안한다. 핵심 기여는 두 가지로 요약할 수 있다. 하나는 generator와 discriminator를 모두 autoencoder로 구성하여 adversarial training의 안정성을 높인 것이고, 다른 하나는 discriminator reconstruction error를 anomaly score로 사용하여 정상과 이상을 더 잘 분리한 것이다.

실험 결과를 보면 ADAE는 MNIST와 CIFAR-10 같은 대표 benchmark에서 기존 GAN 기반 방법들보다 우수하거나 최소한 경쟁력 있는 성능을 보였고, BRATS 2017 brain tumor slice-wise detection에서도 AE, VAE, ceVAE를 넘어서는 성능을 달성했다. 따라서 이 연구는 “GAN 기반 anomaly detection은 불안정하다”는 문제를 단순 회피하는 대신, 구조를 다시 설계해 실용적인 대안을 제시했다는 점에서 의미가 있다.

향후 연구 측면에서는 더 고해상도 이미지, 더 복잡한 자연 분포, localization 수준의 이상 탐지, 그리고 threshold calibration이나 uncertainty estimation과의 결합이 중요한 방향이 될 수 있다. 실제 적용 관점에서는 제조 검사, 의료 영상 판독 보조, 감시 영상 이상 탐지처럼 **정상 데이터는 많지만 이상 사례는 희귀한 분야**에서 특히 유용할 가능성이 있다. 이 논문은 그 출발점으로서, reconstruction 기반 anomaly detection과 adversarial learning을 비교적 안정적으로 결합한 설계라는 점에서 가치가 있다.
