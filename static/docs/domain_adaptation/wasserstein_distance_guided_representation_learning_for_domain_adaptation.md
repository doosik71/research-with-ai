# Wasserstein Distance Guided Representation Learning for Domain Adaptation

* **저자**: Jian Shen, Yanru Qu, Weinan Zhang, Yong Yu
* **발표연도**: 2017
* **arXiv**: [https://arxiv.org/abs/1707.01217](https://arxiv.org/abs/1707.01217)

## 1. 논문 개요

이 논문은 **unsupervised domain adaptation** 문제를 다룬다. 구체적으로는 라벨이 충분한 source domain과 라벨이 거의 없거나 없는 target domain 사이에 데이터 분포 차이, 즉 **covariate shift**가 존재할 때, source에서 학습한 분류기가 target에서도 잘 동작하도록 만드는 것이 목표다. 논문의 핵심 문제의식은 단순히 source 데이터에 대해 높은 정확도를 보이는 표현을 학습하는 것만으로는 target에서 일반화가 어렵다는 점이다. 따라서 학습된 feature representation이 두 조건을 동시에 만족해야 한다고 본다. 첫째, source와 target 사이의 분포 차이를 줄이는 **domain invariant** 특성을 가져야 하고, 둘째, 실제 분류에 필요한 **discriminative** 정보도 유지해야 한다.

기존의 대표적 접근은 MMD, CORAL, DANN처럼 어떤 형태로든 source와 target의 representation 분포를 가깝게 만드는 것이다. 그러나 논문은 특히 adversarial adaptation 계열에서 사용되는 domain classifier 기반 divergence가, 두 분포가 멀리 떨어져 있거나 domain classifier가 너무 쉽게 두 도메인을 구분할 수 있을 때 **gradient vanishing** 문제를 일으킬 수 있다고 지적한다. 이 문제를 해결하기 위해 저자들은 Wasserstein distance를 representation learning의 안내 신호로 사용한다.

제안 방법인 **WDGRL (Wasserstein Distance Guided Representation Learning)** 은 source와 target representation 분포 사이의 empirical Wasserstein distance를 추정하고, feature extractor가 이 값을 줄이도록 adversarial하게 학습된다. 논문은 이 접근이 이론적으로 더 나은 gradient 성질을 제공하며, 일반화 관점에서도 target error bound와 연결될 수 있다고 주장한다. 실험적으로도 감성 분류와 이미지 분류 domain adaptation 벤치마크에서 기존 방법보다 더 나은 평균 성능을 보였다고 보고한다.

이 문제는 실제 응용에서 매우 중요하다. 데이터 라벨링 비용이 크고 도메인이 자주 바뀌는 환경에서는 target domain의 대량 라벨 확보가 어렵기 때문이다. 예를 들어 상품 리뷰 분류, 이메일 스팸 필터링, 객체 인식 등에서 source와 target의 데이터 생성 과정이 달라지면 성능이 급격히 저하될 수 있다. 따라서 domain discrepancy를 안정적으로 줄이는 학습 원리는 transfer learning과 domain adaptation 전반에서 중요한 의미를 가진다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 명확하다. **representation space에서 source와 target 분포 간 거리를 Wasserstein distance로 측정하고, 그 거리를 직접 줄이는 방향으로 feature extractor를 학습하자**는 것이다. 기존 adversarial domain adaptation은 대체로 domain classifier가 source/target을 구분하지 못하게 만드는 방식이었다. 하지만 이런 방식은 classifier가 너무 잘 맞아버리면 feature extractor에 전달되는 gradient가 약해질 수 있다. 저자들은 Wasserstein distance가 이런 상황에서도 더 안정적이고 유의미한 gradient를 줄 수 있다고 본다.

이를 위해 논문은 GAN에서 generator-discriminator 관계를 domain adaptation으로 옮겨온다. 다만 여기서 discriminator 대신 **domain critic**을 사용한다. 이 critic은 source와 target feature distribution의 차이를 binary classification 확률이 아니라 **Wasserstein-1 distance의 dual form**으로 추정한다. feature extractor는 critic이 크게 만들려는 그 거리를 줄이도록 업데이트된다. 즉, 두 네트워크는 다음과 같은 역할을 가진다.

하나는 입력을 latent representation으로 바꾸는 feature extractor이고, 다른 하나는 해당 representation이 source 쪽인지 target 쪽인지에 대한 “판별 확률”이 아니라, 두 분포의 차이를 나타내는 실수값을 출력하는 critic이다. critic은 source representation에는 큰 값을, target representation에는 작은 값을 주는 방향으로 학습되며, feature extractor는 이 차이가 줄어들도록 학습된다.

기존 방법과의 차별점은 크게 두 가지다. 첫째, divergence measure가 cross-entropy 기반 domain classification loss가 아니라 **Wasserstein distance**라는 점이다. 둘째, 학습 방식도 단순한 GRL 기반의 한 번에 섞인 업데이트가 아니라, critic을 여러 step 충분히 업데이트한 다음 feature extractor와 classifier를 업데이트하는 **iterative adversarial training**을 사용한다. 이 구조는 WGAN의 학습 철학과 유사하다.

또한 논문은 WDGRL이 완전히 독립적인 프레임워크이면서도, 다른 symmetric feature-based adaptation 구조 안에 쉽게 삽입될 수 있다고 강조한다. 다시 말해, MMD나 DANN이 수행하던 “representation alignment” 부분을 WDGRL로 바꾸어 넣는 것이 가능하다는 것이다.

## 3. 상세 방법 설명

### 전체 구조

논문의 방법은 크게 세 모듈로 구성된다.

첫째는 **feature extractor** $f_g$ 이다. 입력 $x \in \mathbb{R}^m$ 를 받아 latent representation $h \in \mathbb{R}^d$ 로 변환한다. 즉,
$h = f_g(x)$
이다.

둘째는 **domain critic** $f_w$ 이다. 이 네트워크는 representation $h$ 를 입력받아 실수값을 출력한다. 목적은 source representation과 target representation 사이의 Wasserstein distance를 추정하는 것이다.

셋째는 **discriminator** 또는 실제로는 분류기 역할을 하는 classifier $f_c$ 이다. 이 모듈은 source의 라벨 데이터를 사용해 supervised classification loss를 학습하고, 최종적으로 target 예측에도 사용된다.

이 구조의 핵심은 feature extractor가 두 종류의 gradient를 동시에 받는다는 점이다. 하나는 source label을 맞추기 위한 classification gradient이고, 다른 하나는 source-target discrepancy를 줄이기 위한 Wasserstein gradient이다. 전자는 representation을 **구분 가능하게** 만들고, 후자는 representation을 **도메인 불변하게** 만든다.

### Wasserstein distance 도입

논문은 먼저 일반적인 $p$-Wasserstein distance를 소개하지만, 실제로는 1차 Wasserstein distance, 즉 Earth-Mover distance만 사용한다. 두 분포 $\mathbb{P}$ 와 $\mathbb{Q}$ 사이의 1-Wasserstein distance는 dual form으로 다음과 같이 쓸 수 있다.

$$
W_1(\mathbb{P}, \mathbb{Q}) = \sup_{|f|_L \le 1} \mathbb{E}_{x \sim \mathbb{P}}[f(x)] - \mathbb{E}_{x \sim \mathbb{Q}}[f(x)]
$$

여기서 중요한 제약은 $f$ 가 **1-Lipschitz** 함수여야 한다는 점이다. 논문에서는 이 $f$ 를 neural network critic $f_w$ 로 근사한다.

source와 target의 representation 분포를 각각 $\mathbb{P}_{h^s}$, $\mathbb{P}_{h^t}$ 라고 하면, representation 수준의 Wasserstein distance는 다음과 같이 표현된다.

$$
W_1(\mathbb{P}_{h^s}, \mathbb{P}_{h^t}) = \sup_{|f_w|_L \le 1} \mathbb{E}_{\mathbb{P}_{h^s}}[f_w(h)] - \mathbb{E}_{\mathbb{P}_{h^t}}[f_w(h)]
$$

그리고 $h = f_g(x)$ 이므로 실제로는 원 입력 분포를 feature extractor를 거쳐 비교하는 꼴이 된다.

### domain critic loss

미니배치 기반으로 empirical Wasserstein distance를 추정하기 위해 critic loss를 다음처럼 둔다.

$$
\mathcal{L}_{wd}(x^s, x^t) = - \frac{1}{n^s}\sum_{x^s \in X^s} f_w(f_g(x^s)) \frac{1}{n^t}\sum_{x^t \in X^t} f_w(f_g(x^t))
$$

직관적으로 critic은 source representation에 대해서는 큰 점수를, target representation에 대해서는 작은 점수를 주어 이 차이를 키우려 한다. 이 값을 최대화하면 Wasserstein distance의 empirical estimate가 된다.

### Lipschitz 제약과 gradient penalty

Wasserstein dual formulation이 성립하려면 critic이 1-Lipschitz여야 한다. 초기 WGAN에서는 weight clipping을 사용했지만, 논문은 이 방법이 capacity underuse, gradient exploding/vanishing을 일으킬 수 있다고 보고, 대신 **gradient penalty**를 사용한다.

Gradient penalty는 다음과 같다.

$$
\mathcal{L}_{grad}(\hat{h}) = (|\nabla_{\hat{h}} f_w(\hat{h})|_2 - 1)^2
$$

여기서 $\hat{h}$ 는 source representation과 target representation, 그리고 두 representation 사이를 잇는 선분 위의 랜덤한 점들이다. 즉, Lipschitz 제약을 source/target 샘플 그 자체뿐 아니라 그 사이 경로에도 적용한다.

최종적으로 critic은 다음 목표를 최대화한다.

$$
\max_{\theta_w} \left\{ \mathcal{L}_{wd} - \gamma \mathcal{L}_{grad} \right\}
$$

여기서 $\gamma$ 는 gradient penalty의 강도를 조절하는 계수다.

### feature extractor의 adversarial 학습

critic이 충분히 잘 Wasserstein distance를 추정할 수 있게 학습되면, feature extractor는 그 추정값을 줄이는 방향으로 학습된다. 따라서 representation learning 측면의 minimax 문제는 다음과 같다.

$$
\min_{\theta_g} \max_{\theta_w} \left\{ \mathcal{L}_{wd} - \gamma \mathcal{L}_{grad} \right\}
$$

논문에서 중요한 설명은, **최소화 단계에서는 gradient penalty를 feature extractor 학습에 사용하지 않는다**는 점이다. 즉, $\gamma$ 는 critic 최적화에는 들어가지만 feature extractor 최적화에는 사실상 $0$ 으로 둔다. 이는 gradient penalty가 critic의 Lipschitz 조건을 위한 것이지, representation 자체를 직접 규제하기 위한 항은 아니기 때문이다.

### classifier와 결합

단지 domain discrepancy만 줄이면 representation이 분류에 필요한 class information까지 잃을 수 있다. 이를 막기 위해 논문은 source 라벨을 사용하는 classification loss를 함께 둔다. classifier $f_c$ 는 $d$차원 representation을 받아 $l$개의 클래스에 대한 softmax 출력을 만든다.

분류 loss는 source labeled data에 대한 cross-entropy이다.

$$
\mathcal{L}_c(x^s, y^s) = -\frac{1}{n^s} \sum_{i=1}^{n^s} \sum_{k=1}^{l} 1(y_i^s = k)\cdot \log \left( f_c(f_g(x_i^s))_k \right)
$$

최종 목적함수는 다음과 같다.

$$
\min_{\theta_g,\theta_c} \left\{ \mathcal{L}_c + \lambda \max_{\theta_w} \left[ \mathcal{L}_{wd} - \gamma \mathcal{L}_{grad} \right] \right\}
$$

여기서 $\lambda$ 는 **분류 성능과 도메인 정렬의 trade-off** 를 제어한다.

직관적으로 보면 다음과 같다. classifier $f_c$ 는 source classification을 잘 하도록 학습되고, critic $f_w$ 는 source/target representation의 차이를 크게 하도록 학습되며, feature extractor $f_g$ 는 classifier가 잘 맞히도록 하면서 동시에 critic이 보는 source-target 차이를 줄이는 방향으로 학습된다. 결국 $f_g$ 는 **class는 유지하면서 domain 차이만 줄이는 표현** 을 배우도록 압박받는다.

### 알고리즘 흐름

논문의 Algorithm 1은 학습 절차를 두 단계 반복 구조로 정리한다.

먼저 source 미니배치와 target 미니배치를 뽑는다. 그다음 critic을 $n$ step 동안 반복 업데이트한다. 이때 source와 target representation을 계산하고, 둘 사이 직선 위 random interpolation point도 샘플링해서 gradient penalty를 계산한다. critic은 gradient ascent로
$\mathcal{L}_{wd} - \gamma \mathcal{L}_{grad}$
를 최대화한다.

그 다음 classifier를 source cross-entropy로 한 번 업데이트한다. 마지막으로 feature extractor를
$\mathcal{L}_c + \mathcal{L}_{wd}$
를 최소화하도록 업데이트한다. 논문 알고리즘 표기에서는 이 단계에서 $\lambda$ 계수를 별도로 명시하지 않고 식 (9)에서 설명하지만, 실제 구현에서는 논문 서술대로 trade-off coefficient를 튜닝했다고 보는 것이 자연스럽다. 다만 추출 텍스트만으로는 식 (9)의 $\lambda$ 가 알고리즘 구현식에 정확히 어떤 방식으로 반영되었는지는 완전히 명시적이지 않다.

### 왜 Wasserstein이 더 좋은가

논문은 두 가지 층위에서 설명한다.

첫째는 **gradient superiority** 다. domain classifier 기반 adversarial loss는 source와 target이 너무 잘 구분되면, source에 대해 sigmoid 출력이 거의 1, target에 대해 거의 0이 되어 gradient가 매우 작아질 수 있다. 특히 어떤 영역에서 한 도메인의 확률이 사실상 0에 가까우면 그 영역의 샘플들은 거의 gradient를 제공하지 못한다. 반면 Wasserstein loss에서는 source 샘플은 critic 방향 미분에서 $+1$, target 샘플은 $-1$의 일관된 신호를 주므로 더 안정적인 gradient를 기대할 수 있다는 것이다.

둘째는 **generalization bound** 다. 논문은 가설 클래스 $H$ 가 $K$-Lipschitz라는 가정 하에 target error가 source error와 Wasserstein distance, 그리고 ideal hypothesis의 combined error 항으로 upper bound됨을 보인다.

핵심 결과는 다음과 같다.

$$
\epsilon_t(h) \le \epsilon_s(h) + 2K W_1(\mu_s, \mu_t) + \lambda
$$

여기서 $\lambda$ 는 source와 target에서 동시에 가장 잘 작동하는 ideal hypothesis의 combined error이다. 이 식은 source error가 작고, domain 간 Wasserstein distance가 작으며, 두 도메인 모두를 설명하는 이상적 가설이 존재한다면 target error도 작아질 수 있음을 말한다. 즉, Wasserstein distance를 줄이는 것이 단순한 경험적 정렬이 아니라 generalization 측면에서도 의미가 있다는 주장을 뒷받침한다.

## 4. 실험 및 결과

### 실험 설정 개요

논문은 주로 두 벤치마크에서 WDGRL을 비교한다. 하나는 **Amazon review sentiment adaptation** 이고, 다른 하나는 **Office-Caltech object recognition** 이다. 추가로 appendix에는 synthetic data, SURF feature 기반 Office-Caltech, email spam filtering, 20 newsgroup 분류 결과도 제시한다.

주요 비교 대상은 다음 네 가지다.

**S-only** 는 source labeled data만으로 학습한 뒤 target에 직접 테스트하는 하한선 역할의 baseline이다.
**MMD** 는 RKHS 상의 mean embedding 차이를 줄이는 방법이다.
**DANN** 은 GRL을 사용한 adversarial domain classifier 기반 접근이다.
**CORAL** 은 source와 target feature의 covariance를 맞추는 접근이다.
**WDGRL** 은 논문 제안 방법이다.

모든 방법은 공정 비교를 위해 동일한 MLP 기반 구조를 사용하며, 마지막 hidden layer에 adaptation loss를 적용한다. optimizer는 Adam, learning rate는 $10^{-4}$ 이고, 총 batch size는 64이며 source 32개, target 32개를 사용한다. WDGRL의 critic은 hidden node 100개짜리 한 층 구조이고, critic update step 수는 $n=5$, gradient penalty 계수 $\gamma=10$ 으로 설정했다.

### Amazon review 데이터셋

이 데이터셋은 books (B), DVDs (D), electronics (E), kitchen appliances (K)의 네 도메인으로 구성되며, 각 도메인에 2,000개의 labeled review와 약 4,000개의 unlabeled review가 있다. 입력은 unigram과 bigram에서 가장 빈도가 높은 5,000개 term을 사용한다. 총 $4 \times 3 = 12$개의 adaptation task가 구성된다.

논문 결과에 따르면 WDGRL의 평균 정확도는 **82.43%** 로, MMD의 **81.22%**, DANN의 **80.74%**, CORAL의 **82.16%**, S-only의 **77.84%** 보다 높다.

특히 WDGRL은 12개 중 **10개 task에서 최고 성능**, 나머지 **2개 task에서는 두 번째 성능**을 기록했다. 예를 들어,
$B \rightarrow K$ 에서는 WDGRL이 **85.45%** 로 CORAL의 **84.81%**, DANN의 **82.76%** 보다 높았고,
$D \rightarrow K$ 에서는 **86.24%** 로 가장 좋았다.
반면 $K \rightarrow E$ 에서는 WDGRL이 **86.29%** 로 CORAL의 **86.83%** 에 미치지 못했다. 따라서 “모든 task에서 압도적 우위”라고 말할 수는 없지만, 평균적으로 가장 강하고 매우 일관적인 성능을 보였다고 해석할 수 있다.

논문은 이 결과를 두 가지로 해석한다. 첫째, WDGRL이 DANN보다 좋다는 점은 Wasserstein 기반 adversarial signal이 더 안정적인 gradient를 제공한다는 이론적 주장과 맞아떨어진다. 둘째, MMD와 CORAL은 비모수적 접근이라 계산 비용은 낮을 수 있지만, 표현 정렬의 질에서는 WDGRL이 더 강할 수 있다는 것이다.

### Office-Caltech 데이터셋

이 데이터셋은 Amazon (A), Webcam (W), DSLR (D), Caltech (C)의 네 도메인과 10개 공통 카테고리로 구성된다. 샘플 수는 A 958개, W 295개, D 157개, C 1123개이다. 입력은 AlexNet FC7에서 추출한 **Decaf 4096차원 feature**다.

이 실험에서 WDGRL의 평균 정확도는 **92.74%** 이며, MMD의 **92.03%**, CORAL의 **90.76%**, DANN의 **87.67%** 보다 높다. 평균 차이는 Amazon 리뷰에 비해 크지 않지만, 여러 task에서 WDGRL이 분명한 개선을 보인다.

예를 들어,
$A \rightarrow D$ 는 WDGRL이 **93.68%** 로 최고이며,
$W \rightarrow A$ 는 **93.67%**,
$D \rightarrow A$ 는 **91.69%**,
$D \rightarrow C$ 는 **90.24%**,
$C \rightarrow D$ 는 **94.74%** 로 모두 강한 성능을 보였다.

다만 모든 task에서 최고는 아니다. 예를 들어 $A \rightarrow W$ 에서는 MMD가 **91.58%** 로 WDGRL의 **89.47%** 보다 높고, $C \rightarrow W$ 에서도 CORAL이 **92.63%** 로 WDGRL의 **91.58%** 보다 높다. 또한 $W \rightarrow D$ 는 여러 방법이 100%에 도달해 WDGRL의 우위를 보여주기 어려운 쉬운 task로 보인다.

논문은 이 데이터셋이 도메인당 수백 개 수준의 비교적 작은 데이터셋이며 10-class 문제라는 점을 강조한다. 그럼에도 WDGRL이 효과적이었다는 것은 empirical Wasserstein distance 기반 adaptation이 소규모 데이터셋에서도 작동 가능함을 시사한다.

### Feature visualization

논문은 Amazon 리뷰의 $D \rightarrow E$ task에 대해 t-SNE 시각화를 제공한다. source의 positive/negative 샘플과 target의 positive/negative 샘플을 서로 다른 색으로 표시해, 같은 class끼리는 source-target을 넘어 가까이 모이고 다른 class끼리는 분리되는지 본다.

논문 해석에 따르면 대부분의 방법이 어느 정도 domain invariant하고 discriminative한 표현을 학습했지만, WDGRL은 source와 target의 같은 class가 더 잘 정렬되고, target 내 positive와 negative가 섞이는 영역도 더 작다. 이는 정량 성능뿐 아니라 representation geometry 측면에서도 WDGRL이 더 좋은 정렬을 만들었다는 정성적 근거로 제시된다.

다만 제공된 추출 텍스트에는 실제 Figure 이미지 자체는 없고 저자 설명만 있으므로, 시각적 패턴의 세부 모양을 독립적으로 판독할 수는 없다. 해석은 논문 본문 서술에 기반한다.

### Appendix의 추가 실험

부록에는 몇 가지 추가 결과가 있다.

첫째, **synthetic data** 실험은 DANN이 gradient vanishing 때문에 실패할 수 있고 WDGRL은 성공할 수 있다는 장난감 예제를 제공한다. 저자들은 이 예제가 다소 제한적이라고 스스로 인정하지만, 주장의 직관을 보여주는 보조 사례로 제시한다.

둘째, **Office-Caltech with SURF features** 에서 WDGRL 평균은 **47.25%** 로, MMD **45.95%**, DANN **46.57%**, D-CORAL **46.10%** 보다 높다. 개선 폭은 크지 않지만 평균상 가장 좋다.

셋째, **email spam filtering** 에서는 public inbox를 source로, 세 private inbox를 target으로 두고 실험했다. WDGRL 평균은 **89.90%** 로 MMD **87.00%**, DANN **86.98%**, D-CORAL **84.45%** 보다 높다. 논문은 특히 WDGRL이 MMD, DANN보다 약 **2.90%p** 가량 더 높다고 강조한다.

넷째, **20 newsgroup** 에서는 WDGRL 평균이 **95.77%** 로 DANN **95.47%**, MMD **94.11%**, D-CORAL **93.00%** 보다 약간 높다. 다만 이 경우는 방법들 간 차이가 크지 않고 거의 비슷한 수준이라고 해석한다.

전체적으로 보면 WDGRL은 여러 데이터셋에서 평균적으로 가장 높거나 상위권이며, 특히 gradient 문제가 상대적으로 중요할 수 있는 setting에서 DANN보다 안정적으로 보인다는 것이 논문의 실험 메시지다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 제기, 방법, 이론, 실험이 비교적 일관되게 연결되어 있다**는 점이다. 단순히 새로운 loss를 제안하는 데 그치지 않고, 왜 기존 adversarial loss가 불안정할 수 있는지 설명하고, Wasserstein distance가 왜 더 적절할 수 있는지 이론적 직관을 제공한다. 특히 gradient vanishing 문제를 domain adaptation 문맥에서 다시 해석한 점은 설득력이 있다.

둘째 강점은 **구조가 단순하고 기존 프레임워크에 쉽게 결합 가능하다**는 점이다. WDGRL은 feature extractor 뒤에 critic을 붙이고 adversarially Wasserstein distance를 줄이도록 만드는 방식이기 때문에, 기존 symmetric feature-based adaptation 프레임워크의 alignment 모듈을 대체하는 형태로 이해할 수 있다. 이는 방법론의 재사용성과 확장 가능성을 높인다.

셋째, 논문은 **분류 성능만이 아니라 representation의 질** 도 함께 논의한다. domain invariant representation은 단지 source-target을 섞는 것만으로는 부족하고, class discrimination도 유지해야 한다는 점을 분명히 하고 classifier loss를 함께 사용하는 구조를 제안한다. 실험의 t-SNE 분석도 그 메시지와 잘 맞는다.

넷째, **generalization bound** 를 Wasserstein distance와 연결한 점도 의미 있다. 물론 이 bound 자체가 실제 성능을 직접 보장하는 것은 아니지만, 최소한 왜 Wasserstein alignment가 adaptation에 이론적 의미를 갖는지 설명하는 틀을 제공한다.

반면 한계도 분명하다.

첫째, 논문이 주장하는 gradient superiority는 직관적으로 타당하지만, 실제 deep network 학습 전체에서 그 효과가 얼마나 결정적인지는 더 신중히 봐야 한다. synthetic example은 제한적인 장난감 예제이며, 실제 복잡한 데이터셋에서는 architecture, optimization, hyperparameter tuning의 영향도 크다. 논문도 일부 task에서는 WDGRL이 최고가 아니며, MMD나 CORAL이 더 좋은 경우가 있음을 스스로 보여준다. 즉 “항상 Wasserstein이 우월하다”는 강한 결론까지는 어렵다.

둘째, **계산 비용** 문제다. 논문도 WDGRL이 DANN보다 gradient 장점은 있지만 Wasserstein distance를 추정하기 위해 critic을 여러 step 학습해야 하므로 시간이 더 든다고 인정한다. 실제 large-scale vision adaptation에서는 이 비용이 더 커질 수 있다.

셋째, 이론 분석에는 가정이 필요하다. hypothesis가 $K$-Lipschitz라는 가정은 neural network에 대해 아주 비현실적이라고까지는 할 수 없지만, 실제로 학습된 깊은 네트워크가 이 가정을 얼마나 적절히 만족하는지는 별개의 문제다. 또한 bound의 $\lambda$ 항, 즉 source와 target 모두에서 잘 작동하는 ideal hypothesis의 존재 여부는 실제 문제에서 알기 어렵다. 따라서 이 bound는 엄밀한 예측도구라기보다 방향성 있는 정당화에 가깝다.

넷째, 실험 범위도 당시 기준으로는 괜찮지만 현대 기준으로 보면 제한적이다. Amazon review, Office-Caltech, SURF/Decaf feature 기반 실험은 의미 있지만, end-to-end CNN feature learning이나 더 큰 규모의 시각 데이터셋에서의 검증은 부족하다. 논문도 향후 image data를 위한 더 정교한 architecture를 다루겠다고 적고 있다.

다섯째, 추출 텍스트 기준으로는 하이퍼파라미터 $\lambda$ 의 실제 설정 범위나 각 task별 최적값, critic과 classifier 업데이트의 세부 scheduling 등에 대한 매우 구체적인 수치는 모두 제공되지 않는다. 논문은 grid search로 best result를 보고했다고만 말한다. 따라서 재현 관점에서는 코드가 있더라도 본문만으로는 일부 세부 사항이 부족할 수 있다.

종합하면, 이 논문은 domain adaptation에서 Wasserstein distance를 representation alignment의 핵심 원리로 밀어붙인 초기의 의미 있는 작업이며, 특히 DANN 계열의 약점을 겨냥한 개선안으로서 설득력이 있다. 다만 성능 우위는 전반적으로는 분명하지만 task별로 절대적이지 않고, 계산 비용과 이론 가정의 현실성이라는 한계는 남아 있다.

## 6. 결론

이 논문은 domain adaptation에서 source와 target의 representation 분포 차이를 줄이기 위해 **Wasserstein distance를 직접 최적화하는 adversarial representation learning 방법 WDGRL** 을 제안했다. 핵심 기여는 세 가지로 요약할 수 있다.

첫째, domain classifier 기반 adversarial loss 대신 Wasserstein distance를 사용해 더 안정적인 gradient를 제공하는 representation alignment 방식을 제안했다.
둘째, gradient penalty를 사용한 domain critic과 classifier를 결합해, **domain invariant** 하면서도 **target-discriminative** 한 표현을 학습하는 실용적 알고리즘을 제시했다.
셋째, Wasserstein distance와 target error bound 사이의 연결을 보이며, 이 접근이 단지 경험적으로만이 아니라 이론적으로도 domain adaptation에 의미 있음을 설명했다.

실험 결과는 Amazon review, Office-Caltech, 그리고 부록의 추가 데이터셋들에서 WDGRL이 평균적으로 강한 성능을 보인다는 점을 보여준다. 특히 DANN과 비교했을 때 더 안정적인 adversarial alignment가 가능하다는 주장을 어느 정도 뒷받침한다.

이 연구의 실제적 의미는 크다. 이후 domain adaptation과 distribution alignment 연구에서 optimal transport, Wasserstein objective, critic-based matching이 적극적으로 탐구되는 흐름과 맞닿아 있기 때문이다. 또한 feature alignment를 단순 분류기 혼란(confusion) 문제가 아니라 **분포 사이의 거리 최소화 문제**로 보는 관점을 강화했다는 점에서도 중요하다. 향후 더 큰 규모의 비전 모델, end-to-end representation learning, 혹은 multi-layer adaptation 구조와 결합될 때 WDGRL의 아이디어는 더 넓은 응용 가능성을 가질 수 있다.

결론적으로 이 논문은 domain adaptation에서 Wasserstein distance를 실질적인 representation learning 원리로 끌어들인 깔끔한 연구이며, 이론과 실험 모두에서 당시의 state-of-the-art 방법을 넘어서는 유의미한 기여를 했다.
