# Learning to Transfer Examples for Partial Domain Adaptation

* **저자**: Zhangjie Cao, Kaichao You, Mingsheng Long, Jianmin Wang, Qiang Yang
* **발표연도**: 2019
* **arXiv**: [https://arxiv.org/abs/1903.12230](https://arxiv.org/abs/1903.12230)

## 1. 논문 개요

이 논문은 **Partial Domain Adaptation (PDA)** 문제를 다룬다. PDA는 source domain의 label space가 target domain의 label space를 포함하는 상황, 즉 $ \mathcal{C}_s \supset \mathcal{C}_t $ 인 경우를 의미한다. 기존의 unsupervised domain adaptation은 보통 source와 target이 같은 클래스 집합을 공유한다고 가정하지만, 실제 응용에서는 큰 범주의 source dataset과 더 작은 범주의 unlabeled target dataset이 주어지는 경우가 많다. 이때 source에만 존재하고 target에는 없는 클래스들이 포함되며, 이런 클래스들은 transfer 과정에서 오히려 해로운 영향을 줄 수 있다.

논문의 핵심 문제는 다음과 같다. target label이 학습 시점에 알려져 있지 않은 상태에서, source 예제 중 어떤 것이 target과 관련 있는 shared classes에 속하는지 판별해야 하며, 동시에 target과 무관한 outlier source classes의 영향을 줄여야 한다. 기존 adversarial domain adaptation은 source와 target 전체 분포를 정렬하려 하기 때문에, 실제로는 맞춰지면 안 되는 outlier source classes까지 target 쪽으로 끌어당기게 된다. 이로 인해 **negative transfer**가 발생한다.

이 문제는 실용적으로 중요하다. 대규모 labeled source 데이터셋은 풍부하지만, 새 환경의 target 데이터는 적고 unlabeled인 경우가 많다. 따라서 “큰 데이터셋에서 작은 관심 영역으로 안전하게 전이”하는 능력은 실제 시스템의 확장성과 비용 효율성에 직접 연결된다. 이 논문은 그러한 상황에서 어떤 source 예제를 얼마나 신뢰하고 얼마나 강하게 전이할지 학습하는 방법을 제안한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **모든 source sample을 동일하게 transfer하지 말고, sample 단위로 transferability를 추정하여 가중치를 부여하자**는 것이다. 논문은 이를 위해 **Example Transfer Network (ETN)** 을 제안한다. ETN은 source example마다 “이 예제가 target에 얼마나 관련 있는가”를 나타내는 weight를 계산하고, 이 weight를 단순히 domain discriminator에만 적용하는 것이 아니라 **source classifier와 domain discriminator 양쪽 모두에 적용**한다.

이 점이 기존 PDA 방법들과의 중요한 차별점이다. 예를 들어 이전 방법들은 주로 domain alignment 단계에서만 outlier source example의 영향을 줄이려 했다. 하지만 source classifier는 여전히 모든 source class를 학습하므로, outlier classes가 classifier의 결정 경계를 흐리게 만들 수 있다. ETN은 이 문제를 명시적으로 해결하기 위해 **분류기 학습 자체도 shared label space에 집중되도록 설계**했다.

또 하나의 핵심 아이디어는 transferability를 정할 때 단순한 domain similarity만 보는 것이 아니라, **discriminative structure**, 즉 클래스 구분 정보를 함께 활용한다는 점이다. 이를 위해 auxiliary label predictor와 auxiliary domain discriminator를 도입한다. 이 auxiliary branch는 adversarial training에 직접 참여하지 않으면서, 어떤 source sample이 target과 가깝고 동시에 shared class에 속할 가능성이 높은지 더 선명하게 구분하는 역할을 한다. 결과적으로 ETN은 shared classes에는 큰 weight를, outlier classes에는 거의 0에 가까운 weight를 부여하도록 학습된다.

## 3. 상세 방법 설명

### 3.1 문제 설정

source domain은 labeled dataset $ \mathcal{D}_s = {(\mathbf{x}_i^s, \mathbf{y}_i^s)}_{i=1}^{n_s} $ 이고, target domain은 unlabeled dataset $ \mathcal{D}_t = {\mathbf{x}_j^t}_{j=1}^{n_t} $ 이다. source label space와 target label space는 각각 $ \mathcal{C}_s $ 와 $ \mathcal{C}_t $ 이며, PDA에서는 $ \mathcal{C}_s \supset \mathcal{C}_t $ 이다.

학습 목표는 feature extractor $G_f$ 와 classifier $G_y$ 를 학습하여, source에서 얻은 지식을 target에 잘 일반화하도록 만드는 것이다. 하지만 여기서는 source에 존재하는 일부 클래스가 target에 없기 때문에, 단순한 전체 분포 정렬은 잘못된 정렬을 유도한다.

### 3.2 기본 출발점: domain adversarial learning

기존 DANN류 방법은 feature extractor $G_f$, source classifier $G_y$, domain discriminator $G_d$ 로 구성된다. source 분류 손실은 줄이고, 동시에 domain discriminator가 source/target을 구분하지 못하도록 adversarial learning을 수행한다.

논문에서 제시한 기본 objective는 다음과 같다.

$$
E(\theta_f,\theta_y,\theta_d) = \frac{1}{n_s}\sum_{\mathbf{x}_i \in \mathcal{D}_s} L_y(G_y(G_f(\mathbf{x}_i)), \mathbf{y}_i) - \frac{1}{n_a}\sum_{\mathbf{x}_i \in \mathcal{D}_a} L_d(G_d(G_f(\mathbf{x}_i)), \mathbf{d}_i)
$$

여기서 $L_y$ 와 $L_d$ 는 각각 classification loss와 domain discrimination loss이다. 이 구조는 standard DA에서는 잘 작동하지만, PDA에서는 outlier source classes까지 target 쪽으로 맞추려는 문제가 있다.

### 3.3 ETN의 전체 구조

ETN은 Figure 2 기준으로 다음 모듈들로 구성된다.

첫째, **feature extractor $G_f$** 는 입력 이미지에서 feature를 뽑는다.

둘째, **source classifier $G_y$** 는 source label을 예측한다. 최종 target 분류에도 이 classifier가 사용된다.

셋째, **domain discriminator $G_d$** 는 adversarial training에 사용된다. source와 target feature를 구별하려 하고, feature extractor는 이를 속이도록 학습된다.

넷째, **auxiliary label predictor $ \tilde{G}_y $** 가 추가된다. 이 모듈은 source label supervision을 이용하여 class-discriminative information을 인코딩한다.

다섯째, **auxiliary domain discriminator $ \tilde{G}_d $** 가 있다. 이 모듈은 adversarial branch와는 달리 feature extractor를 속이는 용도로 쓰이지 않고, source example의 transferability를 추정하는 데 사용된다.

논문이 강조하는 새 설계는 파란색 모듈, 즉 $ \tilde{G}_y $ 와 $ \tilde{G}_d $ 이다.

### 3.4 transferability weighting framework

ETN은 source example마다 weight $ w(\mathbf{x}_i^s) $ 를 부여한다. 직관적으로 이 값이 크면 target과 관련 있는 sample이므로 더 많이 사용하고, 작으면 outlier 가능성이 높으므로 영향력을 줄인다.

이 weight는 두 곳에 동시에 사용된다.

첫째, **source classifier loss** 에 사용된다.

$$
E_{G_y} = \frac{1}{n_s}\sum_{i=1}^{n_s} w(\mathbf{x}_i^s) L(G_y(G_f(\mathbf{x}_i^s)), \mathbf{y}_i^s) + \frac{\gamma}{n_t}\sum_{j=1}^{n_t} H(G_y(G_f(\mathbf{x}_j^t)))
$$

여기서 앞 항은 weighted source classification loss이고, 뒤 항은 target entropy minimization이다. $H(\cdot)$ 는 prediction entropy이며, $ \gamma $ 는 source supervised term과 target entropy term의 균형을 조절하는 하이퍼파라미터다.

이 entropy minimization은 unlabeled target sample에 대해 예측이 너무 불확실하지 않도록 만든다. 즉, target sample을 더 confident하게 분류하게 유도하는 semi-supervised learning 성격의 보조 신호다.

둘째, **domain discriminator loss** 에도 같은 weight를 사용한다.

$$
E_{G_d} = - \frac{1}{n_s}\sum_{i=1}^{n_s} w(\mathbf{x}_i^s)\log(G_d(G_f(\mathbf{x}_i^s))) - \frac{1}{n_t}\sum_{j=1}^{n_t} \log(1-G_d(G_f(\mathbf{x}_j^t)))
$$

이 식은 source 쪽에서만 weight를 적용한다. 즉, target과 관련이 큰 source sample은 domain alignment에 더 많이 기여하고, irrelevant한 source sample은 alignment 과정에서 거의 무시된다.

이 설계의 핵심은 “어느 sample을 align할지” 뿐 아니라 “어느 sample로 classifier를 학습할지”까지 동시에 제어한다는 점이다. 이것이 ETN이 이전 방법보다 더 강한 이유다.

### 3.5 transferability를 어떻게 계산하는가

이 논문의 가장 중요한 부분은 weight를 어떻게 얻는지이다.

기존 IWAN은 auxiliary domain classifier를 이용해 source sample이 target domain에 속할 확률 비슷한 값을 추정했다. 하지만 논문은 이것만으로는 shared classes와 outlier classes를 충분히 잘 구분하지 못한다고 본다. 이유는 단순히 domain 정보만 보면, 같은 source 내부에서도 shared class와 outlier class 사이의 차이가 선명하지 않을 수 있기 때문이다.

그래서 ETN은 **auxiliary label predictor $ \tilde{G}_y $** 를 넣어서 class-discriminative information을 auxiliary domain discriminator에 주입한다.

#### leaky-softmax

$ \tilde{G}_y $ 는 feature를 $|\mathcal{C}_s|$ 차원 logit $ \mathbf{z} $ 로 바꾼 뒤, 일반 softmax가 아니라 **leaky-softmax** 를 적용한다.

$$
\tilde{\sigma}(\mathbf{z}) = \frac{\exp(\mathbf{z})} {|\mathcal{C}_s| + \sum_{c=1}^{|\mathcal{C}_s|}\exp(z_c)}
$$

일반 softmax와 달리 분모에 $|\mathcal{C}_s|$ 가 추가되어 있기 때문에, 출력 원소들의 합이 항상 1보다 작다. 이 성질이 중요하다.

source sample은 label supervision을 받으므로 특정 클래스에 대한 logit이 커질 수 있고, 따라서 leaky-softmax 출력의 원소 합이 1에 가까워진다. 반면 target sample은 label이 없고 source class에 명확히 대응하지 않을 수 있어서 출력이 더 불확실하고, 원소 합이 작아지는 경향이 있다.

논문은 auxiliary domain discriminator를 다음처럼 정의한다.

$$
\tilde{G}_d(G_f(\mathbf{x}_i)) = \sum_{c=1}^{|\mathcal{C}_s|} \tilde{G}_y^c(G_f(\mathbf{x}_i))
$$

즉, auxiliary classifier의 클래스별 출력을 모두 더한 값이 $ \tilde{G}_d $ 가 된다. 이 값은 “해당 sample이 source domain에 속할 가능성”처럼 해석할 수 있다. source example 중에서도 target과 더 가까운, 즉 shared class 쪽 sample일수록 이 값이 더 작고, 따라서 target과 더 관련 있다고 볼 수 있다.

#### auxiliary label predictor의 학습

$ \tilde{G}_y $ 는 source label로 학습된다. 논문은 이를 one-vs-rest binary classification 형태의 multitask loss로 표현한다.

$$
E_{\tilde{G}_y} = -\frac{\lambda}{n_s} \sum_{i=1}^{n_s}\sum_{c=1}^{|\mathcal{C}_s|} \left[ y_{i,c}^s \log \tilde{G}_y^c(G_f(\mathbf{x}_i^s)) + (1-y_{i,c}^s)\log(1-\tilde{G}_y^c(G_f(\mathbf{x}_i^s))) \right]
$$

여기서 $ \lambda $ 는 auxiliary classification term의 가중치다.

#### auxiliary domain discriminator의 학습

$ \tilde{G}_d $ 는 source와 target을 구분하는 보조 domain loss로 학습된다.

$$
E_{\tilde{G}_d} = - \frac{1}{n_s}\sum_{i=1}^{n_s} \log(\tilde{G}_d(G_f(\mathbf{x}_i^s))) - \frac{1}{n_t}\sum_{j=1}^{n_t} \log(1-\tilde{G}_d(G_f(\mathbf{x}_j^t)))
$$

중요한 점은 $ \tilde{G}_d $ 가 $ \tilde{G}_y $ 의 출력 합으로 정의되어 있으므로, 이 auxiliary branch는 **domain 정보와 label 정보가 결합된 상태로** 학습된다는 것이다. 이것이 ETN이 “공유 클래스와 outlier 클래스를 더 잘 분리하는 weight”를 얻는 핵심 메커니즘이다.

### 3.6 최종 weight와 정규화

최종적으로 source sample의 transferability는 다음처럼 정의된다.

$$
w(\mathbf{x}_i^s) = 1 - \tilde{G}_d(G_f(\mathbf{x}_i^s))
$$

즉, auxiliary domain discriminator가 “이건 source 같다”고 강하게 판단할수록 weight는 작아지고, 반대로 target과 가깝다고 판단될수록 weight는 커진다.

논문은 source sample에 대한 $ \tilde{G}_d $ 출력이 대체로 1에 가까워져 weight가 매우 작아질 수 있으므로, mini-batch 단위로 평균이 1이 되도록 정규화한다.

$$
w(\mathbf{x})
\leftarrow
\frac{w(\mathbf{x})}
{\frac{1}{B}\sum_{i=1}^{B}w(\mathbf{x}_i)}
$$

이 정규화는 학습 안정성을 높이고, batch마다 weight scale이 지나치게 달라지는 문제를 완화한다.

### 3.7 최종 minimax optimization

최종 ETN은 아래와 같은 saddle-point 문제로 정리된다.

$$
(\hat{\theta}_f,\hat{\theta}_y) = \arg\min_{\theta_f,\theta_y} E_{G_y} - E_{G_d}
$$

$$
\hat{\theta}_d = \arg\min_{\theta_d} E_{G_d}
$$

$$
\hat{\theta}_{\tilde{y}} = \arg\min_{\theta_{\tilde{y}}} E_{\tilde{G}_y} + E_{\tilde{G}_d}
$$

즉, 주 모델은 adversarial adaptation을 하고, auxiliary branch는 weight estimation을 담당한다. 이 구조를 통해 ETN은 “무엇을 얼마나 transfer할지”를 sample 수준에서 점진적으로 학습한다.

## 4. 실험 및 결과

### 4.1 데이터셋과 설정

논문은 세 가지 대표 벤치마크를 사용했다.

첫째, **Office-31** 이다. 총 31개 클래스와 세 도메인 Amazon(A), DSLR(D), Webcam(W)으로 구성된다. PDA 설정을 위해 Office-31과 Caltech-256의 공통 10개 클래스를 target로 사용해 여섯 개 task를 만든다. 즉 source는 31 classes, target은 10 classes이다.

둘째, **Office-Home** 이다. Artistic, Clip Art, Product, Real-World 네 도메인과 65개 카테고리로 구성된다. PDA 설정에서는 알파벳 순 첫 25개 클래스를 target domain으로 사용하고, source는 전체 65개 클래스를 사용한다. 총 12개 adaptation task가 있다.

셋째, **ImageNet-Caltech** 이다. ImageNet-1K와 Caltech-256이 공유하는 84개 클래스를 이용한다. 두 개의 large-scale PDA task, 즉 ImageNet(1000)$\rightarrow$Caltech(84), Caltech(256)$\rightarrow$ImageNet(84)를 구성한다. 특히 ImageNet$\rightarrow$Caltech에서는 1000개 source class 중 84개만 shared class이고 나머지 916개가 사실상 outlier라는 점이 매우 어렵다.

비교 방법으로는 ResNet-50, DAN, DANN, ADDA, RTN, SAN, IWAN, PADA가 포함되었다. 추가로 Office-31에서는 VGG backbone 기반 비교도 수행했다. 구현은 PyTorch이며, ResNet-50과 VGG는 ImageNet pretrained 모델을 fine-tuning했고, 새로 추가된 layer는 더 큰 learning rate로 학습했다.

### 4.2 주요 정량 결과

#### Office-Home (ResNet-50)

Table 1에서 ETN의 평균 정확도는 **70.45%** 로, 비교 방법 중 가장 높다. 주요 baseline들의 평균은 다음과 같다.

* ResNet: 61.35
* DANN: 61.72
* ADDA: 62.82
* RTN: 63.07
* IWAN: 63.56
* SAN: 65.30
* PADA: 62.06
* **ETN: 70.45**

즉, ETN은 SAN 대비 약 5.15%p, IWAN 대비 약 6.89%p 높은 평균 성능을 보인다. 특히 여러 task에서 큰 개선이 관찰된다. 예를 들어 Ar$\rightarrow$Pr에서는 SAN 68.68, IWAN 54.45에 비해 ETN이 **77.03** 이고, Pr$\rightarrow$Rw에서는 SAN 80.07, IWAN 81.28보다 높은 **84.37** 을 기록했다.

이 결과는 단순히 특정 task에서만 잘 되는 것이 아니라, 다양한 도메인 조합 전반에서 일관되게 강하다는 점을 보여준다.

#### Office-31 및 ImageNet-Caltech (ResNet-50)

Table 2에서 Office-31 평균 정확도는 ETN이 **96.73%** 로 최고다. SAN은 94.96, IWAN은 94.69, PADA는 92.69 수준이다. 특히 A$\rightarrow$W, A$\rightarrow$D, D$\rightarrow$A 같은 어려운 task에서 개선이 있다.

ImageNet-Caltech의 평균도 ETN이 **79.08%** 로 가장 높다. 비교 대상 평균은 다음 정도다.

* ResNet: 70.49
* DAN: 65.72
* DANN: 69.23
* ADDA: 70.57
* RTN: 70.85
* IWAN: 75.70
* SAN: 76.51
* PADA: 72.76
* **ETN: 79.08**

이 large-scale 결과는 ETN의 강점을 특히 잘 보여준다. outlier classes 수가 매우 많은 상황에서 ETN이 기존 방식보다 더 robust하다는 논문의 주장과 잘 맞는다.

#### Office-31 (VGG)

Table 3에서도 ETN은 평균 **96.74%** 로 최고 성능을 기록한다. 즉, ETN의 효과가 특정 backbone에만 의존하지 않고 VGG에서도 재현됨을 보였다.

### 4.3 ablation study

Table 4는 ETN의 각 설계 요소가 실제로 필요한지 검증한다.

* **ETN w/o classifier**: source classifier loss에는 weight를 적용하지 않은 variant
* **ETN w/o auxiliary**: auxiliary label predictor 없이 auxiliary domain discriminator만 사용하는 variant
* **ETN**: full model

Office-Home 평균 성능은 다음과 같다.

* ETN w/o classifier: 68.93
* ETN w/o auxiliary: 61.79
* ETN: 70.45

여기서 두 가지 중요한 결론이 나온다.

첫째, source classifier에도 weight를 적용하는 것이 실제로 도움이 된다. 이 부분이 빠지면 70.45에서 68.93으로 떨어진다. 즉, outlier classes를 domain alignment에서만 줄이는 것은 충분하지 않고, classifier 자체도 shared classes 중심으로 학습되어야 함을 보여준다.

둘째, auxiliary branch의 discriminative weighting이 훨씬 더 중요하다. auxiliary label predictor를 제거하면 평균 성능이 61.79까지 크게 하락한다. 이는 단순 domain similarity만으로는 좋은 weight를 얻기 어렵고, class-discriminative information이 반드시 필요하다는 논문의 핵심 주장에 대한 강한 근거다.

### 4.4 분석 실험

#### Feature visualization

Figure 3의 t-SNE 시각화에서 ETN이 DANN, SAN, IWAN보다 더 뚜렷한 cluster를 형성한다고 논문은 보고한다. 이는 ETN이 target feature를 더 잘 분리하고, shared classes 중심으로 representation을 정렬하고 있음을 시사한다.

#### Class overlap 변화

Figure 4에서는 target classes의 개수를 바꾸면서 성능을 비교한다. target classes 수가 줄어들수록, 즉 shared label space overlap이 작아질수록 DANN의 성능은 빠르게 악화된다. 이는 negative transfer가 overlap 감소에 매우 민감하다는 의미다. SAN은 비교적 안정적이지만, ETN은 전 범위에서 가장 안정적이고 지속적으로 우수한 성능을 보인다. 흥미롭게도 label space가 완전히 겹치는 standard DA 상황에서도 ETN은 DANN보다 나쁘지 않으며, 오히려 더 좋을 수 있다고 보고한다. 이는 weighting mechanism이 outlier가 없는 상황에서 크게 해를 끼치지 않는다는 점을 의미한다.

#### Convergence

Figure 5에서 ETN은 다른 방법들보다 더 낮은 target test error로 수렴한다. 논문은 이를 두고 ETN이 더 효율적이고 안정적으로 학습된다고 해석한다.

#### Weight visualization

Figure 6은 IWAN과 ETN이 산출한 weight의 density를 비교한다. ETN은 shared classes 쪽에는 더 큰 weight를, outlier classes 쪽에는 거의 0에 가까운 weight를 부여한다. 이 그림은 ETN의 성능 향상이 단순한 결과 숫자만이 아니라, 실제로 sample weighting quality가 좋아졌기 때문임을 뒷받침한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 PDA 문제의 핵심인 negative transfer를 **sample-wise weighting** 관점에서 매우 직접적으로 다뤘다는 점이다. 기존 연구들이 domain discriminator 쪽만 부분적으로 조정하던 것을 넘어서, source classifier와 domain discriminator 모두를 shared label space 중심으로 유도했다. 이는 설계상 명확하고, ablation 결과로도 효과가 뒷받침된다.

또 다른 강점은 auxiliary branch 설계가 꽤 정교하다는 점이다. 단순히 “target과 비슷한 source sample”을 찾는 것이 아니라, label discrimination까지 함께 사용해 shared classes와 outlier classes를 구분하려 했다. leaky-softmax와 auxiliary label predictor를 결합한 방식은 이 논문에서 중요한 기술적 기여로 볼 수 있다.

실험적으로도 설득력이 있다. Office-Home, Office-31, ImageNet-Caltech처럼 규모와 난도가 다른 여러 벤치마크에서 일관된 향상을 보였고, 특히 outlier class가 매우 많은 large-scale PDA에서 강점을 보였다. 이는 ETN이 진짜로 PDA 상황에 맞게 설계되었음을 보여준다.

반면 한계도 있다. 우선 이 논문은 **target label space가 source의 부분집합** 이라는 PDA 가정에 의존한다. target에 source에 없는 완전한 unknown class가 섞여 있는 open-set domain adaptation과는 다른 문제이며, 논문도 이를 별도 확장 과제로 언급한다. 따라서 실제 환경에서 target이 source subset이라는 가정이 맞지 않으면 그대로 적용하기 어렵다.

또한 weight가 auxiliary domain discriminator 출력에 크게 의존하므로, 이 branch의 품질이 낮으면 잘못된 weight가 전파될 가능성이 있다. 논문은 이를 discriminative auxiliary classifier로 개선했지만, 여전히 pseudo target relevance estimation 문제 자체는 완전히 쉬운 문제가 아니다.

더 나아가, 논문은 왜 특정 task에서 얼마나 성능이 좋아졌는지를 정성적으로는 설명하지만, 이 weight가 훈련 초기에 얼마나 noisy하고 후반에 얼마나 정제되는지 같은 동적 분석은 제한적이다. 즉, “progressive weighting scheme” 이라는 표현에 비해 시간에 따른 weight evolution 분석은 본문에서 아주 상세히 제공되지는 않는다.

비판적으로 보면, ETN은 기존 adversarial DA 구조 위에 auxiliary modules와 weighting mechanism을 얹은 형태이므로, 구조 복잡도가 증가한다. 실용적 적용에서는 tuning해야 할 요소가 더 많아질 수 있다. 다만 논문이 보고한 결과만 놓고 보면, 이 추가 복잡도는 충분한 성능 이득으로 보상되는 편이다.

## 6. 결론

이 논문은 Partial Domain Adaptation에서 가장 중요한 문제인 negative transfer를 해결하기 위해 **Example Transfer Network (ETN)** 을 제안했다. ETN의 핵심은 source example마다 transferability를 추정하고, 이를 이용해 **source classifier와 domain discriminator 양쪽 모두를 weighted learning** 하도록 만든 점이다. 또한 auxiliary label predictor와 auxiliary domain discriminator를 결합하여, 단순 domain similarity를 넘어 discriminative information까지 반영한 weight를 학습한다.

실험 결과는 ETN이 다양한 PDA benchmark에서 기존 방법들을 안정적으로 능가함을 보여준다. 특히 large-scale setting에서 성능 차이가 더 커지는 점은, ETN이 outlier source classes가 많은 현실적 상황에 강하다는 것을 시사한다.

실제 적용 측면에서 이 연구는 “큰 범용 labeled source에서 작은 unlabeled target으로 안전하게 전이”해야 하는 문제에 유용하다. 예를 들어 대규모 범용 데이터셋을 기반으로 특정 산업, 특정 장비, 특정 생물학적 기능 영역으로 적응해야 하는 상황에서 의미가 크다. 향후 연구로는 open-set setting, 더 복잡한 label mismatch, 혹은 self-training과의 결합 등이 자연스러운 확장 방향이 될 수 있다.
