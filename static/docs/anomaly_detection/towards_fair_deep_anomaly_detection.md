# Towards Fair Deep Anomaly Detection

- **저자**: Hongjing Zhang, Ian Davidson (University of California, Davis)
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2012.14961

## 1. 논문 개요

이 논문은 deep anomaly detection, 그중에서도 **deep one-class classification** 계열의 대표 방법인 **Deep SVDD**에 공정성(fairness)을 도입하는 문제를 다룬다. 기본적인 anomaly detection의 목적은 정상적인 패턴과 다른, 즉 비정상적이거나 이상한 샘플을 찾아내는 것이다. 이런 기술은 사기 탐지, 침입 탐지, 시스템 이상 감지, 의료 영상 분석 등에서 매우 중요하다. 그런데 실제 응용에서는 사람이 탐지 대상이 되는 경우가 많고, 이때 모델이 성별, 인종, 출처 같은 **sensitive attribute**와 연관된 편향을 학습하면 사회적으로 문제가 될 수 있다.

논문이 문제 삼는 핵심은 다음과 같다. Deep SVDD는 정상 샘플만으로 학습하여 “정상 데이터가 모이는 중심(center)” 주변에 표현을 압축하는 방식인데, 이때 deep neural network가 데이터 안에 들어 있는 복잡한 패턴을 잘 학습하는 장점이 오히려 **민감 속성과 관련된 편향된 특징까지 포착할 수 있다**는 점이다. 즉, anomaly detection 성능은 좋아 보일 수 있지만, 예측 결과를 보면 특정 집단이 비정상으로 더 많이 분류되는 불공정성이 생길 수 있다.

논문은 이 문제를 해결하기 위해 **Deep Fair SVDD**라는 새로운 구조를 제안한다. 핵심 아이디어는 encoder가 만든 latent representation에서 sensitive attribute 정보를 제거하도록 **adversarial learning**을 사용한다는 것이다. 저자들은 fairness를 regularizer나 constraint로 추가하는 기존 접근과 달리, **discriminator가 민감 속성을 맞히지 못하도록 encoder를 학습**시키는 방식을 택했다. 또한 anomaly detection 맥락에 맞춘 두 가지 fairness measure를 제안하고, 기존 deep anomaly detection 방법들이 실제로 unfair하다는 점을 실험적으로 보인다.

이 연구가 중요한 이유는, anomaly detection이 단순한 성능 문제를 넘어 **누가 의심받고 누가 비정상으로 간주되는가**와 직접 연결되기 때문이다. 특히 얼굴 이미지, 범죄 재범 예측, 숫자 이미지처럼 민감 속성이나 데이터 출처가 결과에 영향을 줄 수 있는 경우, 높은 AUC만으로는 충분하지 않다. 따라서 이 논문은 “정확한 anomaly detection”과 “공정한 prediction”을 동시에 만족시키려는 초기 시도로서 의미가 크다.

## 2. 핵심 아이디어

논문의 중심 직관은 매우 명확하다. **좋은 anomaly detector는 정상 데이터를 compact하게 모아야 하지만, 그 representation 안에 sensitive attribute 정보가 남아 있으면 fairness 문제가 생긴다**는 것이다. 따라서 저자들은 anomaly detection을 위한 compact representation과 fairness를 위한 invariant representation을 동시에 추구한다.

기존 Deep SVDD는 입력 $x$를 encoder $f(x;\theta)$로 임베딩한 뒤, 이 임베딩이 미리 정한 중심 $c$에 최대한 가깝게 모이도록 학습한다. 하지만 이 representation이 예를 들어 gender나 race를 쉽게 구분할 수 있을 정도로 민감 속성 정보를 담고 있다면, 테스트 시 anomaly score 분포 자체가 집단별로 달라질 수 있다. 그 결과 특정 그룹이 더 자주 anomaly로 예측된다.

이 논문은 이를 막기 위해 encoder 뒤에 **sensitive attribute discriminator**를 붙인다. discriminator는 encoder가 만든 임베딩만 보고 protected status variable $z$를 예측하려고 한다. 반면 encoder는 두 가지 목표를 동시에 가진다. 하나는 원래의 SVDD 목적처럼 정상 샘플을 중심 근처에 모으는 것이고, 다른 하나는 discriminator가 민감 속성을 맞히지 못하도록 representation을 바꾸는 것이다. 즉, encoder와 discriminator가 서로 경쟁하는 **min-max game**을 형성한다.

이 접근의 차별점은 fairness를 직접 수식 제약으로 넣는 대신, **representation level에서 sensitive information leakage를 줄인다**는 점이다. 논문은 이 방법이 tabular 데이터와 image 데이터 모두에 적용 가능하다고 주장하며, 실제로 COMPAS, celebA, MNIST-USPS, MNIST-Invert에서 실험한다.

또 하나의 중요한 기여는 fairness를 평가하는 방식이다. 저자들은 anomaly detection에 맞게 두 가지 그룹 수준 fairness 지표를 제안한다. 첫 번째는 abnormal group에 대한 **$p%$-rule**이고, 두 번째는 anomaly score의 전체 분포 차이를 반영하는 **Wasserstein-1 distance 기반 분포 거리(distribution distance)**이다. 첫 번째는 threshold 기반의 group fairness, 두 번째는 threshold 없이 전체 score distribution의 독립성을 보는 지표라고 이해할 수 있다. 이 둘을 함께 사용함으로써, 단순히 “최종 anomaly 집합”만이 아니라 “점수 분포 전체가 민감 속성과 얼마나 결합되어 있는가”까지 평가한다.

## 3. 상세 방법 설명

### 3.1 Deep SVDD 배경

논문의 기반 모델은 Deep SVDD이다. 학습 데이터는 오직 정상 샘플만 포함한다고 가정한다. 입력 데이터 $\mathcal{X} \in \mathbb{R}^{n \times d}$에 대해, neural network $f(\cdot;\theta)$가 모든 정상 샘플을 고정된 중심 $c$ 근처로 매핑하도록 학습한다.

Deep SVDD의 목적함수는 다음과 같다.

$$
\arg\min_{\theta} \frac{1}{n}\sum_{i=1}^{n}|f(x_i;\theta)-c|^2 + \frac{\alpha}{2}\sum_{\ell=1}^{L}|\theta_\ell|^2
$$

여기서 첫 번째 항은 모든 정상 샘플의 임베딩을 중심 $c$ 가까이 모으는 항이고, 두 번째 항은 weight decay regularization이다. $\alpha > 0$는 정규화 강도를 조절하는 하이퍼파라미터다.

테스트 시 각 샘플의 anomaly score는 중심으로부터의 거리로 정의된다.

$$
s(x)=|f(x;\theta)-c|^2
$$

즉, 중심에서 멀수록 anomaly일 가능성이 크다. 이 구조는 generative model이나 reconstruction model처럼 “복원 실패”를 이용하는 것이 아니라, representation space에서 정상성을 직접 모델링한다는 점이 특징이다.

### 3.2 논문이 정의한 fairness

이 논문은 **group-level fairness**를 다룬다. protected status variable $z \in {0,1}$가 있을 때, anomaly prediction이 특정 집단에 disproportionate하게 몰리지 않도록 하는 것이 목표다.

#### (1) Fairness by $p%$-rule

먼저 anomaly score threshold $t$를 기준으로, $s(x)\le t$이면 normal, $s(x)>t$이면 abnormal로 본다. 그 다음 protected group과 non-protected group에서 abnormal로 분류되는 비율을 비교한다.

논문이 제시한 fairness measure는 다음과 같다.

$$
\min\left(
\frac{P(s(x)>t \mid z=1)}{P(s(x)>t \mid z=0)},
\frac{P(s(x)>t \mid z=0)}{P(s(x)>t \mid z=1)}
\right)\ge \frac{p}{100}
$$

이 값은 0에서 1 사이이며, 1에 가까울수록 두 집단의 anomaly 판정 비율이 비슷하므로 더 공정하다. 논문은 이를 미국 고용평등위원회의 **80% rule**과 연결한다. 예를 들어 이 값이 0.8 이상이면 80% rule을 만족한다고 해석한다.

다만 저자들은 이 지표의 한계를 명시한다. 첫째, threshold $t$를 정하려면 test set에서 anomaly 개수를 알아야 할 수 있다. 둘째, abnormal group만 본다는 점에서 전체 score distribution의 편향을 다 반영하지 못한다.

#### (2) Fairness by distribution distance

이 한계를 보완하기 위해 논문은 anomaly score 분포 자체를 비교하는 지표를 제안한다. protected status가 $z=0$인 샘플들의 anomaly score 분포를 $\mathbb{P}$, $z=1$인 샘플들의 분포를 $\mathbb{Q}$라고 할 때, 두 분포 사이의 Wasserstein-1 distance를 계산한다.

$$
W(\mathbb{P},\mathbb{Q})=
\inf_{\gamma \in \Pi(\mathbb{P},\mathbb{Q})}
\mathbb{E}_{(x,y)\sim\gamma}[|x-y|]
$$

여기서 $\Pi(\mathbb{P},\mathbb{Q})$는 주변분포가 각각 $\mathbb{P}$와 $\mathbb{Q}$인 모든 joint distribution의 집합이다. 직관적으로는 한 분포를 다른 분포로 옮기기 위해 필요한 “mass transport cost”를 뜻한다. 값이 작을수록 두 집단의 anomaly score 분포가 유사하고, 따라서 더 공정하다고 본다.

이 지표의 장점은 threshold가 필요 없고, abnormal group뿐 아니라 normal/abnormal 전 구간의 score 분포 차이를 반영한다는 점이다.

### 3.3 Deep Fair SVDD의 핵심 구조

논문은 학습 데이터 $\mathcal{X}$가 정상 샘플만 포함하고, 각 샘플마다 binary protected status variable $Z$를 알고 있다고 가정한다. 모델은 두 부분으로 구성된다.

첫째는 encoder $f(\theta)$이다. 입력 데이터를 낮은 차원의 compact representation으로 바꾼다. 둘째는 discriminator $g(\theta_d)$이다. 이 discriminator는 encoder가 만든 임베딩 $f(\mathcal{X};\theta)$만 보고 protected status $z$를 예측하려고 한다.

논문이 바라는 이상적인 목표는 representation이 민감 속성과 독립이 되는 것이다. 이를 식으로 쓰면 다음과 같다.

$$
p(f(\mathcal{X};\theta)\mid z=0)=p(f(\mathcal{X};\theta)\mid z=1)
$$

즉, sensitive attribute가 달라도 representation distribution이 같아지기를 원한다.

### 3.4 손실 함수

#### (1) Encoder의 SVDD loss

정상 샘플을 중심 근처로 모으기 위한 encoder loss는 다음과 같다.

$$
L_{SVDD} = \frac{1}{M}\sum_{i=1}^{M}|f(x_i;\theta)-c|^2 + \frac{\alpha}{2}\sum_{\ell=1}^{L}|\theta_\ell|^2
$$

여기서 $M$은 training normal instance 수이고, $c$는 predetermined center다.

#### (2) Discriminator의 sensitive attribute prediction

discriminator는 encoder의 임베딩을 받아 protected status를 binary classification으로 예측한다. 예측 확률은 sigmoid를 사용하여

$$
\hat z_i=\frac{1}{1+\exp\left(-g(f(x_i;\theta)\mid \theta_d)\right)}
$$

로 정의된다.

이때 discriminator loss는 binary cross entropy다.

$$
L_D = -\frac{1}{M}\sum_{i=1}^{M} \left( z_i\log(\hat z_i) + (1-z_i)\log(1-\hat z_i) \right)
$$

즉, discriminator는 representation에서 sensitive attribute를 최대한 잘 복원하려고 한다.

#### (3) Adversarial loss

encoder는 두 목표를 동시에 갖는다. 정상 데이터는 중심으로 모아야 하고, 동시에 discriminator가 민감 속성을 맞히지 못하게 해야 한다. 이를 위해 다음 adversarial loss를 설계한다.

$$
L_{Adv}=L_{SVDD}-\lambda L_D
$$

여기서 $\lambda>0$는 fairness와 anomaly detection 성능 사이의 trade-off를 조절하는 하이퍼파라미터다.

이 식의 의미는 중요하다. encoder 입장에서는 $L_{SVDD}$를 줄이고 싶지만, 동시에 $- \lambda L_D$ 항 때문에 discriminator loss $L_D$를 **크게 만들고 싶다**. 즉, encoder는 discriminator를 속이는 방향으로 representation을 바꾼다. 반면 discriminator는 $L_D$를 최소화하려 하므로, 두 네트워크는 경쟁 관계에 놓인다.

### 3.5 학습 절차

논문은 이를 adversarial training으로 학습한다. 절차는 세 단계다.

먼저 encoder를 단독으로 $L_{SVDD}$만 사용해 초기 학습한다. 이렇게 하면 representation이 기본적으로 정상 데이터를 중심 주변에 모으는 성질을 갖게 된다.

그 다음 encoder를 고정하고 discriminator를 학습한다. 이때 discriminator는 현재 representation에서 sensitive attribute를 최대한 잘 예측하도록 훈련된다.

그 후 alternating optimization을 수행한다. 한 단계에서는 encoder를 고정하고 discriminator를 업데이트하여 $L_D$를 최소화한다. 다음 단계에서는 discriminator를 고정하고 encoder를 업데이트하여 $L_{Adv}=L_{SVDD}-\lambda L_D$를 최소화한다. 이 과정을 반복하여 min-max equilibrium에 접근한다.

논문이 제시한 최적화 문제는 다음과 같다.

$$
\arg\min_{\theta_d} L_D
$$

$$
\arg\min_{\theta} \left(L_{SVDD}-\lambda L_D\right)
$$

최종적으로 학습이 끝난 뒤 테스트 샘플의 anomaly score는 기존 SVDD와 동일하게 중심 거리로 계산한다.

$$
\mathcal{S}=|f(\mathcal{X};\theta)-c|^2
$$

즉, 추론 단계의 anomaly scoring은 단순하지만, 학습 과정에서 fairness-aware representation을 만들도록 encoder를 바꾸는 것이 핵심이다.

### 3.6 잠재적 확장

논문은 세 가지 확장 방향도 제시한다.

첫째, binary sensitive attribute 대신 **multi-state protected variable**로 확장할 수 있다고 말한다. 예를 들어 nationality나 education level처럼 범주가 여러 개인 경우 discriminator를 multi-class classifier로 바꾸면 된다.

둘째, **multiple protected attributes**도 지원 가능하다고 주장한다. 예를 들어 gender와 race를 동시에 고려할 때, 각 조합을 Cartesian product로 묶어 multi-state variable로 바꿀 수 있다는 아이디어다.

셋째, 현재는 정상 샘플만으로 학습하는 unsupervised one-class setting이지만, 일부 labeled anomaly가 있는 **semi-supervised anomaly detection**으로도 확장 가능하다고 제안한다. 다만 이 부분은 실제 구현 결과가 아니라 향후 연구 방향으로만 언급된다.

## 4. 실험 및 결과

### 4.1 데이터셋

논문은 네 개의 공개 데이터셋에서 실험한다.

첫 번째는 **COMPAS Recidivism**이다. tabular 데이터이며, protected status variable은 race이다. African-American 여부를 binary attribute로 만들고, 2년 내 재체포 여부를 기준으로 normal은 not reoffending, abnormal은 reoffending으로 둔다. 총 3878개 인스턴스, 11차원이다.

두 번째는 **celebA**다. 얼굴 이미지 데이터이며, protected status는 gender다. normal group은 attractive faces, abnormal group은 plain faces로 구성한다. 입력 크기는 $64\times64\times3$이다. 이 설정 자체가 사회적으로 매우 민감한 정의를 포함하지만, 논문은 fairness 문제를 보이기 위한 motivating example로 활용한다.

세 번째는 **MNIST-USPS**다. digit 3을 normal, digit 5를 abnormal로 두고, protected attribute는 digit source(MNIST 또는 USPS)다. 즉, 같은 숫자라도 스타일 출처가 anomaly score에 영향을 미치는지를 본다.

네 번째는 **MNIST-Invert**다. MNIST 이미지와 색을 뒤집은 inverted version을 섞어 만들고, protected attribute는 original/inverted 여부다. 역시 digit 3이 normal, digit 5가 abnormal이다.

논문은 원래 training set과 balanced training set도 구성한다. 예를 들어 celebA는 원래 $z=0$ 그룹이 16000개, $z=1$ 그룹이 4000개로 크게 불균형하며, balanced training set에서는 모두 4000개로 맞춘다. 이를 통해 “데이터 균형만으로 fairness 문제가 해결되는가”도 함께 확인한다.

### 4.2 구현 세부사항

데이터 성격에 따라 encoder 구조를 다르게 사용했다.

MNIST-USPS와 MNIST-Invert에는 두 개의 convolution module과 마지막 32-unit fully connected layer를 사용했다. celebA에는 더 깊은 CNN을 사용했고 마지막 fully connected layer는 128 units다. COMPAS에는 hidden unit이 32와 16인 fully connected network를 사용했다. batch normalization과 ReLU를 사용한다.

discriminator는 모든 데이터셋에서 공통으로 세 개 hidden layer $(500-2000-500)$를 가진 fully connected network를 사용한다. 기본 하이퍼파라미터는 $\lambda=1$, learning rate는 $10^{-3}$, optimizer는 Adam, batch size는 128, weight decay 계수 $\alpha=5\times10^{-6}$이다. 중심 $c$는 모든 instance embedding의 mean으로 둔다.

### 4.3 평가 지표와 비교 대상

평가의 두 축은 anomaly detection 성능과 fairness다.

anomaly detection 성능은 **AUC**로 측정한다. AUC는 anomaly score가 normal보다 abnormal에 더 높게 부여될 확률로 이해할 수 있으며, threshold 독립적인 장점이 있다.

fairness는 앞서 정의한 두 지표, 즉 **$p%$-rule**과 **distribution distance**를 사용한다.

비교 대상은 두 개의 대표 deep anomaly detection baseline이다. 하나는 **Deep SVDD**, 다른 하나는 **DCAE(Deep Convolutional Auto-Encoder)**다. 저자들은 encoder architecture를 최대한 맞춰 공정 비교를 시도했다고 설명한다.

### 4.4 기존 deep anomaly detection의 unfairness

논문은 먼저 “기존 deep anomaly detection이 실제로 unfair한가?”를 검증한다. 이를 위해 Deep SVDD와 DCAE를 original training set과 balanced training set 모두에서 학습시키고 fairness를 비교한다.

결과적으로 균형 잡힌 training set을 쓰면 일부 데이터셋에서는 fairness가 약간 좋아진다. 예를 들어 COMPAS와 celebA에서는 balanced training이 $p%$-rule을 다소 개선한다. 하지만 그 개선은 충분하지 않으며, 대부분의 경우 여전히 **80% rule을 만족하지 못한다**고 보고한다. 더 흥미로운 점은 MNIST-USPS에서는 balanced training을 써도 오히려 더 unfair해질 수 있다는 것이다.

distribution distance 결과도 비슷하다. balanced training이 약간의 개선을 주더라도, representation 자체가 sensitive attribute와 독립이 되지는 않는다. 즉, **단순히 데이터를 균형 맞추는 것만으로 fairness 문제가 해결되지 않는다**는 것이 논문의 결론이다. 이것이 Deep Fair SVDD의 필요성을 뒷받침하는 실험이다.

### 4.5 Deep Fair SVDD의 성능

핵심 결과는 Deep Fair SVDD가 네 개 데이터셋 모두에서 baseline보다 더 나은 fairness를 보인다는 점이다.

먼저 $p%$-rule 기준으로 Deep Fair SVDD는 Deep SVDD와 DCAE를 모두 능가한다. 논문은 네 개 데이터셋 모두에서 제안법의 $p%$-rule이 **80% 이상**이어서 80% rule을 만족한다고 주장한다.

distribution distance 기준으로도 제안법이 더 좋다. 특히 celebA에서 개선 폭이 크다고 언급한다. 이는 얼굴 데이터에서 민감 속성과 시각적 특징이 강하게 얽혀 있기 때문에 representation debiasing의 효과가 크게 나타났다고 해석할 수 있다.

AUC 측면에서는 완전히 공짜 개선은 아니다. COMPAS, MNIST-Invert, MNIST-USPS에서는 Deep SVDD가 약간 더 좋고, celebA에서는 오히려 Deep Fair SVDD가 약간 더 좋다. 전체적으로 저자들은 **fairness 개선에 비해 AUC 손실은 작다**고 결론내린다.

이 결과는 논문의 가장 중요한 메시지다. 즉, fairness를 높이면 무조건 anomaly detection 성능이 크게 희생되는 것이 아니라, 적절한 adversarial training을 통해 **minimal loss on AUC**로 fairness를 크게 개선할 수 있다는 것이다.

### 4.6 $\lambda$에 따른 trade-off

논문은 adversarial loss의 계수 $\lambda$를 $10^{-2}$부터 $10^2$까지 바꾸며 trade-off를 분석한다. 결과는 대체로 예상 가능하다.

모든 데이터셋에서 $\lambda$가 커질수록 $p%$-rule은 증가한다. 즉, fairness pressure를 강하게 줄수록 model은 더 공정해진다. 반면 COMPAS, MNIST-Invert, MNIST-USPS에서는 AUC가 대체로 감소한다. 따라서 fairness와 detection performance 사이에 trade-off가 있음을 보여준다.

하지만 celebA에서는 예외적으로 $\lambda$가 증가할수록 fairness와 AUC가 함께 증가한다. 저자들은 celebA test set에서 남녀 비율이 균형 잡혀 있기 때문에, fairness constraint가 representation을 더 잘 정리하여 anomaly detection에도 도움을 준다고 해석한다. 이는 fairness 정보가 어떤 경우에는 regularization처럼 작동할 수 있음을 시사한다.

### 4.7 예측 결과 분석

논문은 단순히 지표만 보여주지 않고, Deep SVDD와 Deep Fair SVDD의 anomaly prediction overlap도 분석한다. Table 3에 따르면 anomaly prediction overlap ratio는 COMPAS 0.78, celebA 0.70, MNIST-Invert 0.81, MNIST-USPS 0.82로 높다.

이 결과는 의미가 있다. 제안법이 fairness를 높인다고 해서 예측을 완전히 뒤집는 것이 아니라, **대부분의 anomaly prediction은 유지하면서 일부 경계 샘플을 조정**하는 방식으로 fairness를 개선하고 있음을 뜻한다. 저자들은 이것이 AUC가 약간만 감소하는 이유라고 해석한다.

Figure 8의 비중첩 샘플 분석도 같은 메시지를 준다. Deep Fair SVDD가 normal에서 abnormal로 옮긴 샘플, 혹은 그 반대 방향으로 옮긴 샘플이 완전히 랜덤한 것이 아니라, 실제로 “anomaly에 더 가까워 보이는 샘플” 또는 “normal에 더 가까워 보이는 샘플”이라는 것이다. 예를 들어 MNIST-Invert에서는 digit 3과 덜 비슷하거나 digit 5처럼 보이는 샘플이 fair model에서 anomaly 쪽으로 이동한다. 즉, fairness를 위해 무작위로 예측을 바꾸는 것이 아니라, **의미 있는 decision boundary 조정**을 한다고 주장한다.

### 4.8 임베딩 시각화

t-SNE로 본 latent embedding에서도 차이가 나타난다. Deep SVDD의 경우에는 특정 영역이 특정 sensitive group 색으로 지배되는 패턴이 보인다. 이는 representation과 sensitive attribute가 상관되어 있음을 시사한다.

반면 Deep Fair SVDD에서는 서로 다른 민감 속성 값을 가진 점들이 더 고르게 섞여 있다. 특히 celebA에서 red/blue points가 잘 blend된다고 설명한다. 이는 논문의 핵심 목표였던

$$
p(f(\mathcal{X};\theta)\mid z=0)=p(f(\mathcal{X};\theta)\mid z=1)
$$

에 더 가까운 representation을 학습했음을 정성적으로 보여준다.

### 4.9 실행 시간

학습 시간은 분명히 증가한다. Table 4에 따르면 Deep Fair SVDD는 adversarial min-max optimization을 하므로 Deep SVDD보다 훨씬 느리다. 예를 들어 celebA에서는 Deep SVDD가 285.10초인데 제안법은 1703.49초다. MNIST-USPS도 13.12초 대 231.78초로 차이가 크다.

즉, fairness를 얻는 대가로 계산 비용이 상당히 증가한다. 저자들도 이를 한계로 인정하며, training efficiency와 scalability 개선을 future work로 남긴다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **fairness in deep anomaly detection**이라는 거의 다뤄지지 않던 문제를 정면으로 제기했다는 점이다. 기존 fairness 연구가 주로 supervised classification에 집중되어 있었던 상황에서, 정상 데이터만으로 학습하는 one-class anomaly detection에도 사회적 편향이 존재할 수 있음을 실험적으로 보여준 것은 중요한 기여다.

또한 방법론이 비교적 깔끔하다. Deep SVDD라는 잘 알려진 base learner 위에 discriminator를 붙이고 adversarial training을 적용하여 representation에서 sensitive information을 줄이는 구조는 직관적이다. fairness를 복잡한 constraint나 regularizer 대신 adversarial objective로 다룬 점도 설계적으로 명료하다. 특히 tabular와 image 데이터 모두에 적용했다는 점은 방법의 범용성을 어느 정도 뒷받침한다.

fairness evaluation도 강점이다. 논문은 단순히 하나의 fairness metric만 사용하지 않고, threshold 기반의 $p%$-rule과 threshold-free 분포 거리라는 두 관점을 함께 제시한다. anomaly detection에서는 score distribution 자체가 중요하므로, distribution distance를 함께 사용한 것은 적절한 판단이다.

실험 분석도 비교적 충실하다. baseline의 unfairness, balanced training의 한계, trade-off 분석, overlap analysis, embedding visualization, runtime analysis까지 포함하고 있어 단순 성능 비교를 넘어 방법의 작동 방식을 이해하도록 돕는다.

반면 한계도 분명하다.

첫째, 보호 속성을 **binary variable**로 제한한다. 논문은 multi-state나 multiple protected attributes로 확장 가능하다고 말하지만, 실제 실험은 모두 단일 이진 속성에서만 수행되었다. 따라서 복합적 편향(intersectional fairness)에 대한 실증은 없다.

둘째, fairness 개념이 **group-level fairness**에 한정되어 있다. individual fairness나 calibration 같은 다른 fairness notion은 다루지 않는다. anomaly detection에서 어떤 fairness가 가장 적절한지는 응용 도메인에 따라 달라질 수 있으므로, 이 논문의 정의가 유일한 정답은 아니다.

셋째, $p%$-rule은 threshold나 anomaly 개수 정보에 의존한다는 한계를 저자들도 인정한다. distribution distance가 이를 보완하긴 하지만, 이 지표가 실제 의사결정 fairness와 어떻게 연결되는지에 대한 더 깊은 논의는 부족하다.

넷째, anomaly의 정의 자체가 사회적으로 민감한 데이터셋이 있다. 예를 들어 celebA에서 “attractive faces”를 normal, “plain faces”를 abnormal로 두는 설정은 fairness 실험용으로는 직관적일 수 있지만, 현실 적용 관점에서는 조심스럽게 해석해야 한다. 논문도 이 설정의 윤리적 정당성까지는 깊이 논의하지 않는다.

다섯째, 학습 효율성이 낮다. adversarial training은 원래도 불안정할 수 있는데, 논문은 convergence difficulty나 training stability 문제를 깊게 다루지 않는다. 또한 runtime이 크게 증가하는데, 대규모 실제 시스템에 바로 쓰기엔 부담이 있을 수 있다.

여섯째, 이 논문은 fairness 향상과 representation disentanglement를 보여주지만, 왜 특정 데이터셋에서는 fairness가 AUC 향상으로도 이어지고, 어떤 데이터셋에서는 그렇지 않은지에 대한 이론적 설명은 제한적이다. 특히 fairness regularization이 언제 representation learning에 도움이 되는지에 대한 일반적 원리는 제시하지 않는다.

종합하면, 이 논문은 매우 강한 완성형 해법이라기보다는 **문제 제기와 첫 번째 실용적 접근**으로서의 가치가 크다. 공정성 있는 anomaly detection의 필요성을 설득력 있게 보여주고, adversarial representation learning이 그 출발점이 될 수 있음을 실험적으로 제시했다는 점이 중요하다.

## 6. 결론

이 논문은 deep anomaly detection, 특히 Deep SVDD 기반 one-class anomaly detection에서 발생할 수 있는 fairness 문제를 체계적으로 다룬다. 저자들은 기존 deep anomaly detection 모델이 실제로 unfair할 수 있음을 보였고, 이를 완화하기 위해 **Deep Fair SVDD**라는 adversarially trained fair representation learning framework를 제안했다.

핵심 기여는 세 가지로 요약할 수 있다. 첫째, deep anomaly detection에도 fairness 문제가 존재한다는 실증적 증거를 제시했다. 둘째, sensitive attribute를 예측하는 discriminator와 경쟁하도록 encoder를 학습시켜 fairness-aware representation을 만드는 방법을 제안했다. 셋째, anomaly detection에 적합한 두 fairness measure인 $p%$-rule과 distribution distance를 정의하고 이를 통해 실험을 평가했다.

실험 결과에 따르면 제안법은 COMPAS, celebA, MNIST-USPS, MNIST-Invert에서 baseline보다 더 공정한 결과를 보였으며, 대체로 AUC 손실은 작았다. 즉, 이 논문은 “공정성과 anomaly detection 성능은 완전히 상충하는 것만은 아니다”라는 점을 보여준다. 또한 representation visualization과 overlap analysis는 이 모델이 단순히 예측을 무작위로 조정하는 것이 아니라, latent space 자체를 더 fair하게 재구성하고 있음을 시사한다.

실제 적용 측면에서 이 연구는 매우 중요하다. 사람과 관련된 anomaly detection 시스템에서는 단순한 정확도보다 **누가 anomaly로 지목되는가**가 더 큰 사회적 영향을 가질 수 있다. 따라서 이 논문은 향후 공정한 fraud detection, security screening, medical anomaly detection, human-centered monitoring system 같은 영역에서 중요한 출발점이 될 가능성이 있다.

동시에, multiple sensitive attributes, semi-supervised setting, 더 나은 학습 안정성, 더 효율적인 optimization 같은 과제가 남아 있다. 따라서 이 논문은 문제를 완전히 해결한 최종 답이라기보다, **fair deep anomaly detection이라는 연구 방향을 본격적으로 연 논문**으로 보는 것이 적절하다.
