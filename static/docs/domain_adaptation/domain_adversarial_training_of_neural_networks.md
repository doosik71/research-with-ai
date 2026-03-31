# Domain-Adversarial Training of Neural Networks

* **저자**: Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, Victor Lempitsky
* **발표연도**: 2015
  extracted text에는 최종 저널/학회 표기가 명확하지 않지만, arXiv 식별자 `1505.07818`에 따라 2015년으로 표기한다.
* **arXiv**: [http://arxiv.org/abs/1505.07818](http://arxiv.org/abs/1505.07818)

## 1. 논문 개요

이 논문은 **unsupervised domain adaptation** 문제를 딥러닝 학습 과정 안으로 직접 통합하는 방법을 제안한다. 문제 설정은 다음과 같다. source domain에는 라벨이 있는 데이터가 충분히 있고, target domain에는 라벨이 없지만 실제 테스트 시에는 target domain에서 좋은 성능이 필요하다. 예를 들어, synthetic image로 학습했는데 실제 real image에서 잘 동작해야 하거나, 영화 리뷰로 학습했는데 책 리뷰 감성 분석에도 잘 적용되어야 하는 경우다.

핵심 문제는 source와 target의 데이터 분포가 다르다는 점이다. 일반적인 신경망은 source 분포에서 classification loss를 잘 줄일 수 있어도, 그 내부 feature가 domain-specific한 신호를 강하게 담고 있으면 target으로 넘어갔을 때 성능이 무너질 수 있다. 논문은 이 점을 이론적으로도 설명한다. Ben-David 계열의 domain adaptation 이론에 따르면, target risk를 낮추려면 source risk만 낮추는 것으로는 부족하고, source와 target feature 분포 사이의 divergence도 함께 줄여야 한다.

이 논문의 목표는 **분류에 유용하면서도 domain을 구분하기 어려운 feature representation**을 학습하는 것이다. 이를 위해 저자들은 label predictor와 domain classifier를 동시에 두고, feature extractor는 한쪽으로는 label classification에는 도움이 되도록, 다른 한쪽으로는 domain discrimination에는 방해가 되도록 학습시킨다. 즉, source label은 잘 맞추되, 그 feature를 보고 이것이 source에서 왔는지 target에서 왔는지는 맞추기 어렵게 만든다.

이 문제의 중요성은 매우 크다. 실제로 많은 응용에서는 target domain에 라벨을 붙이는 비용이 높거나 거의 불가능하다. 따라서 **라벨 없는 target 데이터만으로도 domain shift를 흡수하는 표현을 학습하는 방법**은 컴퓨터 비전, 자연어 처리, person re-identification 같은 다양한 분야에 직접적인 가치가 있다. 이 논문은 바로 그 방향을 매우 단순하고 일반적인 형태로 제시했고, 이후의 adversarial adaptation 계열 연구에 큰 영향을 준 대표작이다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 간단하지만 매우 강력하다. 좋은 domain adaptation 표현은 다음 두 성질을 동시에 만족해야 한다.

첫째, **source domain의 정답 라벨을 예측하는 데 충분히 discriminative**해야 한다. 즉, class boundary를 학습할 수 있어야 한다.
둘째, **source와 target을 구분하는 단서는 최대한 제거된 domain-invariant representation**이어야 한다.

저자들은 이를 하나의 네트워크 안에서 구현한다. 네트워크는 크게 세 부분으로 나뉜다.

하나는 입력을 feature로 바꾸는 **feature extractor**이고,
둘째는 그 feature로 클래스 라벨을 예측하는 **label predictor**이며,
셋째는 그 feature가 source에서 왔는지 target에서 왔는지를 예측하는 **domain classifier**이다.

여기서 흥미로운 점은 feature extractor가 두 목적 사이에서 상반된 압력을 받는다는 것이다. label predictor는 feature extractor가 class discrimination에 좋은 feature를 만들도록 유도한다. 반면 domain classifier는 source와 target을 잘 구분하려고 하므로, feature extractor 입장에서는 오히려 domain classifier를 실패하게 만들어야 한다. 이 adversarial 관계가 feature를 domain-invariant하게 만든다.

기존 접근과의 차별점은 크게 두 가지다.

첫째, 많은 기존 방법은 고정된 feature 위에서 domain mapping을 학습하거나, autoencoder류의 사전학습을 따로 수행한 뒤 그 위에 classifier를 따로 얹었다. 반면 이 논문은 **feature learning, domain adaptation, classifier learning을 하나의 end-to-end 학습으로 통합**했다.

둘째, 분포 차이를 줄이는 방식이 RKHS mean matching이나 subspace alignment 같은 기하학적 정렬이 아니라, **“domain classifier가 source와 target을 구별하지 못하게 만든다”는 판별적(discriminative) 기준**을 사용한다. 즉, 분포 간 차이를 “도메인 분류가 얼마나 잘 되느냐”로 측정하고, 그것을 feature 수준에서 직접 무너뜨린다.

이 아이디어를 실용적으로 가능하게 만든 장치가 바로 **Gradient Reversal Layer, GRL**이다. forward에서는 아무 일도 하지 않고 입력을 그대로 통과시키지만, backward에서는 gradient 부호를 뒤집어서 feature extractor가 domain classification loss를 줄이는 대신 늘리는 방향으로 업데이트되게 만든다. 이 때문에 구현은 거의 일반적인 backpropagation과 동일하게 유지된다.

## 3. 상세 방법 설명

### 3.1 문제 설정과 이론적 출발점

논문은 source domain 분포를 $\mathcal{D}_S$, target domain 분포를 $\mathcal{D}_T$라고 둔다. source에서는 라벨이 있는 표본 집합 $S={(\mathbf{x}_i, y_i)}_{i=1}^{n}$를 갖고, target에서는 라벨 없는 표본 집합 $T={\mathbf{x}_i}_{i=n+1}^{N}$만 관측한다. 목표는 target risk

$$
R_{\mathcal{D}_T}(\eta)=\Pr_{(\mathbf{x},y)\sim \mathcal{D}_T}(\eta(\mathbf{x}) \neq y)
$$

를 낮추는 classifier $\eta$를 만드는 것이다.

이때 Ben-David 이론은 target error가 source error와 domain divergence의 합으로 상계될 수 있음을 보여준다. 논문이 집중하는 divergence는 **$\mathcal{H}$-divergence**이다. 이것은 어떤 가설 클래스 $\mathcal{H}$에 속한 분류기가 source와 target을 얼마나 잘 구분할 수 있는지를 나타낸다.

$$
d_{\mathcal{H}}(\mathcal{D}_S^X,\mathcal{D}_T^X) = 2 \sup_{\eta\in \mathcal{H}} \left| \Pr_{\mathbf{x}\sim \mathcal{D}_S^X}[\eta(\mathbf{x})=1] - \Pr_{\mathbf{x}\sim \mathcal{D}_T^X}[\eta(\mathbf{x})=1] \right|
$$

직관적으로 보면, 어떤 representation에서 source와 target을 쉽게 구분할 수 있다면 divergence가 크고, 구분하기 어려우면 divergence가 작다. 따라서 좋은 adaptation representation은 이 divergence를 줄이는 방향으로 학습되어야 한다.

논문은 또한 target risk에 대한 generalization bound를 제시한다. 핵심 메시지만 정리하면 다음과 같다. target risk는 대략적으로 source empirical risk, source-target divergence, 그리고 양 도메인에서 동시에 잘 되는 classifier가 존재하는 정도를 나타내는 $\beta$ 항의 합으로 제어된다. 따라서 **좋은 feature는 source risk를 줄이면서 동시에 domain divergence도 줄여야 한다**. DANN은 바로 이 이론을 신경망 구조에 구현한 것이다.

### 3.2 Shallow DANN의 구조

가장 단순한 경우로, 저자들은 single hidden layer neural network를 먼저 설명한다.

feature extractor $G_f$는 입력 $\mathbf{x}\in \mathbb{R}^m$를 $D$차원 representation으로 매핑한다.

$$
G_f(\mathbf{x};\mathbf{W},\mathbf{b}) = \mathrm{sigm}(\mathbf{W}\mathbf{x}+\mathbf{b})
$$

여기서 $\mathrm{sigm}$은 element-wise sigmoid이다.

그 위에 label predictor $G_y$가 붙는다.

$$
G_y(G_f(\mathbf{x});\mathbf{V},\mathbf{c}) = \mathrm{softmax}(\mathbf{V}G_f(\mathbf{x})+\mathbf{c})
$$

source sample에 대해서는 정답 라벨 $y_i$가 있으므로 일반적인 classification loss를 사용한다. 논문은 정답 클래스의 negative log-probability를 사용한다.

$$
\mathcal{L}_y(G_y(G_f(\mathbf{x}_i)), y_i) = \log \frac{1}{G_y(G_f(\mathbf{x}_i))_{y_i}}
$$

만약 domain adaptation이 없다면, 이 신경망은 source classification loss만 최소화하면 된다.

하지만 DANN에서는 여기에 domain classifier $G_d$를 추가한다. 이 분기는 feature가 source에서 왔는지 target에서 왔는지를 예측한다. shallow case에서는 logistic regressor로 두었다.

$$
G_d(G_f(\mathbf{x});\mathbf{u},z) = \mathrm{sigm}(\mathbf{u}^{\top}G_f(\mathbf{x}) + z)
$$

여기서 domain label $d_i$는 source면 0, target이면 1이다. domain classification loss는 binary cross-entropy 형태이다.

$$
\mathcal{L}_d(G_d(G_f(\mathbf{x}_i)), d_i) = d_i \log \frac{1}{G_d(G_f(\mathbf{x}_i))} + (1-d_i)\log \frac{1}{1-G_d(G_f(\mathbf{x}_i))}
$$

### 3.3 목적 함수의 의미

이 논문에서 가장 중요한 부분은 전체 목적 함수가 **min-max saddle point** 구조라는 점이다.

label predictor와 feature extractor는 source label loss를 줄이고 싶다.
domain classifier는 domain loss를 줄이고 싶다.
하지만 feature extractor는 동시에 domain classifier가 잘하지 못하도록 만들어야 한다.

논문은 이를 다음과 같은 형태로 쓴다.

$$
E(\mathbf{W},\mathbf{V},\mathbf{b},\mathbf{c},\mathbf{u},z) = - \frac{1}{n}\sum_{i=1}^{n}\mathcal{L}_y^i \lambda \left( \frac{1}{n}\sum_{i=1}^{n}\mathcal{L}_d^i + \frac{1}{n'}\sum_{i=n+1}^{N}\mathcal{L}_d^i \right)
$$

여기서 $\lambda$는 adaptation strength를 조절하는 hyper-parameter다.

이 식의 의미를 쉬운 말로 풀면 다음과 같다.

* 첫 번째 항은 source label classification을 잘하라는 압력이다.
* 두 번째 항은 domain classification loss인데, 앞에 마이너스가 붙어 있으므로 feature extractor 쪽에서는 이 손실을 **최대화**하려는 방향으로 작용한다.
* 따라서 feature는 라벨 예측에는 좋고, domain 분류에는 나쁜 방향으로 이동한다.

최종적으로 찾고 싶은 것은 saddle point다. 즉,

* feature extractor와 label predictor 파라미터는 $E$를 **최소화**
* domain classifier 파라미터는 $E$를 **최대화**

하는 점을 찾는다.

이 구조는 이후 adversarial learning의 매우 전형적인 형태가 된다.

### 3.4 Gradient Reversal Layer의 역할

이 min-max 구조를 일반적인 SGD로 구현하는 데 핵심이 되는 것이 GRL이다. 논문이 제안한 GRL은 forward에서는 입력을 그대로 통과시킨다.

$$
\mathcal{R}(\mathbf{x}) = \mathbf{x}
$$

하지만 backward에서는 미분값의 부호를 뒤집는다.

$$
\frac{d\mathcal{R}}{d\mathbf{x}} = -\mathbf{I}
$$

즉, domain classifier에서 올라오는 gradient를 feature extractor에 전달할 때, 부호를 반대로 바꾼다. 그러면 domain classifier 자체는 domain loss를 줄이도록 학습되지만, feature extractor는 그 반대 방향으로 업데이트되어 domain classifier를 헷갈리게 만드는 feature를 학습한다.

이 장치 덕분에 전체 네트워크는 구현상 거의 표준 feed-forward network와 같아진다. 단지 feature extractor와 domain classifier 사이에 GRL 하나를 넣으면 된다. 논문이 큰 영향력을 갖게 된 이유도 여기에 있다. 이론은 adversarial하지만, 구현은 매우 단순하다.

### 3.5 일반적인 Deep Architecture로의 확장

논문은 single hidden layer 설명 후, 이를 임의의 deep network로 확장한다. 표기만 바꾸면 구조는 동일하다.

* $G_f(\cdot;\theta_f)$: feature extractor
* $G_y(\cdot;\theta_y)$: label predictor
* $G_d(\cdot;\theta_d)$: domain classifier

전체 목적 함수는 다음과 같다.

$$
E(\theta_f,\theta_y,\theta_d) = - \frac{1}{n}\sum_{i=1}^{n}\mathcal{L}_y^i(\theta_f,\theta_y) \lambda \left( \frac{1}{n}\sum_{i=1}^{n}\mathcal{L}_d^i(\theta_f,\theta_d) + \frac{1}{n'}\sum_{i=n+1}^{N}\mathcal{L}_d^i(\theta_f,\theta_d) \right)
$$

이때 업데이트는 다음과 같은 의미를 가진다.

feature extractor는

$$
\theta_f \leftarrow \theta_f - \mu \left( \frac{\partial \mathcal{L}_y^i}{\partial \theta_f} - \lambda \frac{\partial \mathcal{L}_d^i}{\partial \theta_f} \right)
$$

label predictor는 일반적인 classification gradient로 업데이트된다.

$$
\theta_y \leftarrow \theta_y - \mu \frac{\partial \mathcal{L}_y^i}{\partial \theta_y}
$$

domain classifier는 domain loss를 잘 줄이도록 업데이트된다.

$$
\theta_d \leftarrow \theta_d - \mu \lambda \frac{\partial \mathcal{L}_d^i}{\partial \theta_d}
$$

핵심은 feature extractor 식에서 domain gradient가 빼기 부호로 들어간다는 점이다. 이것이 바로 domain confusion을 유도하는 부분이다. 이 식을 GRL을 넣은 표준 backprop으로 자연스럽게 구현할 수 있다.

### 3.6 학습 절차와 실무적 포인트

논문은 shallow DANN에 대해 구체적인 stochastic update pseudo-code도 제시한다. 학습의 개념적 흐름은 다음과 같다.

먼저 source sample 하나와 target sample 하나를 가져온다.
source sample은 label predictor와 domain classifier 모두에 들어간다.
target sample은 label이 없으므로 domain classifier에만 들어간다.
label loss는 source에 대해서만 계산한다.
domain loss는 source와 target 모두에 대해 계산한다.
이후 domain classifier 파라미터는 domain discrimination을 더 잘하도록 업데이트하고, feature extractor는 그 반대 방향으로 업데이트한다.

deep CNN 실험에서는 batch의 절반을 source, 나머지 절반을 target으로 구성한다. 또한 adaptation parameter $\lambda$는 처음에는 0에 가깝게 두고 점점 1에 가깝게 늘리는 스케줄을 사용한다.

$$
\lambda_p = \frac{2}{1+\exp(-\gamma p)} - 1
$$

여기서 $p$는 학습 진행률이고, $\gamma=10$을 사용했다. 이 스케줄의 목적은 초반에 feature가 아직 안정되지 않았을 때 domain adversarial signal이 너무 세게 들어가 학습을 망치는 것을 막는 것이다.

learning rate도 다음과 같은 schedule을 사용한다.

$$
\mu_p = \frac{\mu_0}{(1+\alpha p)^\beta}
$$

논문에 따르면 $\mu_0=0.01$, $\alpha=10$, $\beta=0.75$를 사용했다. 이 부분은 오늘날 관점에서 보면 상당히 실용적인 heuristic으로 볼 수 있다.

## 4. 실험 및 결과

이 논문은 shallow network와 deep network 모두에서 실험을 수행한다. 또한 classification뿐 아니라 descriptor learning에도 적용한다.

### 4.1 Toy problem: inter-twinning moons

가장 먼저 2D toy example을 사용해 DANN의 작동 원리를 직관적으로 보여준다. source는 두 개의 반달 모양 데이터이고, target은 그것을 $35^\circ$ 회전시킨 뒤 라벨을 제거한 데이터다.

비교 대상은 일반 NN과 DANN이다. 두 모델은 같은 hidden layer 크기 15를 사용하지만, NN은 adversarial backpropagation을 끈 버전이다.

결과는 매우 직관적이다.

일반 NN은 source decision boundary는 잘 맞추지만 rotated target에는 충분히 적응하지 못한다. 반면 DANN은 source와 target을 모두 잘 분리하는 decision boundary를 학습한다. PCA projection으로 hidden representation을 보면, DANN은 target point가 source cluster와 더 잘 섞이도록 표현을 바꾼다. domain classifier 결과도 DANN에서는 source/target 분리가 잘 되지 않는데, 이것이 곧 domain-invariant feature가 학습되었음을 보여준다.

이 toy example은 정량적 벤치마크라기보다, **DANN이 वास्तव में 어떤 representation을 만드는가**를 시각적으로 설명하는 데 중요한 역할을 한다.

### 4.2 Hyper-parameter selection: reverse validation

unsupervised domain adaptation에서는 target 라벨을 쓸 수 없기 때문에 hyper-parameter selection이 어렵다. 논문은 이를 위해 **reverse validation**을 사용한다.

절차는 다음과 같다.
source와 target을 train/validation으로 나눈다.
source labeled train과 target unlabeled train으로 적응 모델 $\eta$를 학습한다.
그 다음, $\eta$가 target train에 예측한 pseudo-label을 붙여 reverse task를 만든다.
즉, self-labeled target을 새로운 source처럼 쓰고, 원래 source unlabeled 데이터를 target처럼 써서 reverse classifier $\eta_r$를 학습한다.
마지막으로 reverse classifier를 원래 source validation에서 평가한다.

이 reverse validation risk가 낮은 hyper-parameter를 고른다. 논문은 sentiment analysis 실험에서 이 방법으로 DANN, NN, SVM의 hyper-parameter를 모두 선택했다.

이 부분은 이 논문의 핵심 기여는 아니지만, 실제 unsupervised adaptation을 평가할 때 target label 없이 모델 선택을 어떻게 할지에 대한 실용적 해법을 제시한다는 점에서 중요하다.

### 4.3 Sentiment analysis: Amazon reviews

논문은 Amazon reviews 데이터셋에서 4개 도메인인 books, dvd, electronics, kitchen 사이의 12개 adaptation task를 수행한다. 각 예시는 5000차원의 unigram/bigram feature로 표현되며, binary sentiment classification 문제다.

각 task에서 2000개의 labeled source example과 2000개의 unlabeled target example을 사용하고, 별도의 target test set에서 정확도를 평가한다.

비교 대상은 다음 세 가지다.

* DANN
* 같은 구조이지만 adaptation regularizer가 없는 shallow NN
* linear kernel SVM

원문 표에 따르면 raw feature에서 DANN은 다수의 task에서 NN과 SVM보다 우수하다. 논문은 Poisson binomial test를 통해 DANN이 NN보다 우수할 확률이 **0.87**, SVM보다 우수할 확률이 **0.83**이라고 보고한다. 즉, 전체적으로 DANN이 통계적으로 더 좋은 경향을 보인다는 뜻이다.

이 결과의 의미는 명확하다. shallow network 수준에서도, 단순히 source risk를 줄이는 것보다 **representation 자체를 domain confusion 방향으로 학습하는 것이 target generalization에 실질적인 이득**을 준다.

### 4.4 mSDA와의 결합

당시 sentiment adaptation의 강력한 baseline 중 하나는 mSDA였다. 논문은 mSDA가 만든 representation 위에 다시 DANN을 적용했을 때의 효과도 검증한다.

mSDA는 source와 target의 unlabeled data를 활용해 robust representation을 만드는 비지도 표현학습 기법이다. 저자들은 corruption probability 50%, 5개 layer를 사용해 mSDA representation을 만든 뒤, 그 위에서 DANN, NN, SVM을 다시 학습했다.

결과적으로 DANN은 이 설정에서도 가장 좋은 성능을 보였고, DANN이 NN보다 우수할 확률은 **0.92**, SVM보다 우수할 확률은 **0.88**로 보고되었다.

이 결과는 중요한 해석을 가능하게 한다. mSDA와 DANN은 모두 representation learning이지만 최적화 목표가 다르다. mSDA는 reconstruction 기반의 robust feature를 만들고, DANN은 domain confusion 기반의 transferable feature를 만든다. 따라서 둘은 일정 부분 상보적일 수 있다. 다만 논문도 일부 task에서는 NN이나 SVM이 최고 성능을 낸 경우가 있어, 완전한 상보성이라고까지는 말하지 않는다.

### 4.5 Proxy A-distance 분석

논문은 이론적 주장과 실제 representation의 연결을 확인하기 위해 **Proxy A-distance (PAD)**를 계산한다. PAD는 source와 target representation이 얼마나 구분 가능한지를 근사하는 지표다. domain classification error를 $\epsilon$이라 할 때 PAD는

$$
\hat{d}_{\mathcal{A}} = 2(1-2\epsilon)
$$

로 근사된다.

해석은 간단하다. domain classifier error가 높아질수록, 즉 source와 target을 구별하기 어려울수록 PAD는 낮아진다. 논문은 Amazon 리뷰 실험에서 DANN, NN, mSDA, mSDA+DANN representation들에 대해 PAD를 비교했다.

결과는 다음과 같다.

* raw data 대비 DANN representation은 PAD를 낮춘다.
* 같은 hidden size에서 NN보다 DANN의 PAD가 더 낮다.
* mSDA alone은 PAD를 오히려 키우는 경향이 있었는데, 그 위에 DANN을 적용하면 PAD가 다시 낮아진다.

이 분석은 DANN이 실제로 **이론이 요구하는 “domain indistinguishable representation”**을 만들고 있음을 간접적으로 보여준다.

### 4.6 Deep image classification

논문은 deep CNN 기반 DANN도 평가한다. 여기서는 MNIST, MNIST-M, SVHN, synthetic digits, synthetic signs, GTSRB, Office 데이터셋 등을 사용한다.

비교 baseline은 다음과 같다.

* source-only model
* train-on-target upper bound
* Subspace Alignment (SA)
* Office에서는 기존 deep adaptation 방법들과의 비교

#### MNIST $\rightarrow$ MNIST-M

MNIST-M은 MNIST 숫자를 자연 이미지 patch와 합성해 만든 target domain이다. 사람에게는 여전히 비교적 쉬운 문제지만, 배경과 stroke 통계가 달라 CNN에는 domain shift가 크다.

논문에 따르면 source-only 모델은 이 설정에서 성능이 좋지 않고, DANN은 feature distribution alignment에 성공해 의미 있는 adaptation 성능 향상을 달성한다. SA의 개선은 상대적으로 작았다고 서술한다.

#### Synthetic Numbers $\rightarrow$ SVHN

source는 약 50만 장의 synthetic digit 이미지이고, target은 실제 street-view house number 이미지인 SVHN이다. synthetic-to-real 전형적인 설정이다.

논문은 DANN이 source-only와 train-on-target 사이 성능 차이의 **약 80%를 메웠다**고 설명한다. 반면 SA는 오히려 약간의 성능 하락까지 보였다고 적는다. 수치 표 자체는 현재 추출 텍스트에 완전하게 포함되지 않았지만, 저자 서술상 DANN이 매우 강한 개선을 보였음은 분명하다.

#### MNIST $\leftrightarrow$ SVHN

이 실험은 domain gap가 훨씬 더 크다. 논문은 이 경우 adaptation이 비대칭적이라고 분석한다. SVHN로 학습한 모델은 비교적 다양한 숫자 스타일을 포함하므로 MNIST에 더 잘 일반화하지만, MNIST로 학습한 모델은 SVHN에 적응하기 어렵다.

실제로 DANN은 **SVHN $\rightarrow$ MNIST** 방향에서는 성능 향상을 보였지만, **MNIST $\rightarrow$ SVHN**에서는 개선에 실패했다고 논문이 명시한다. source-only 성능이 약 0.25 accuracy 수준이고, DANN도 이를 넘지 못했다고 설명한다. 저자들은 이 경우를 자신들의 방법의 **failure example**로 솔직하게 제시한다. 이 점은 논문의 신뢰성을 높이는 부분이기도 하다.

#### Synthetic Signs $\rightarrow$ GTSRB

43개 traffic sign 클래스를 가진 더 복잡한 synthetic-to-real 적응 문제에서도 DANN은 성능 향상을 보여준다. 클래스 수가 많아 feature 분포가 더 복잡하지만, synthetic-to-real adaptation에 DANN이 유효함을 입증한다.

#### Semi-supervised adaptation

또한 GTSRB 설정에서 target domain의 소량 labeled data, 정확히 클래스당 10장씩 총 430장을 추가로 공개했을 때도 DANN이 도움이 될 수 있음을 보인다. 다만 이 부분은 논문이 예비적 결과로 제시하며, thorough verification은 future work라고 명시한다.

### 4.7 Office benchmark

Office는 Amazon, DSLR, Webcam 세 도메인으로 구성된 전통적인 domain adaptation benchmark다. 데이터 수는 적지만 domain shift가 크고, 비전 DA 연구에서 표준처럼 쓰인다.

논문은 AlexNet fine-tuning 기반 구조를 사용하고, Tzeng et al.의 architecture와 유사하게 구성하되 domain mean-based regularization 대신 domain classifier를 붙인다. 이 설정에서 저자들은 fully-transductive protocol 아래에서 기존 unsupervised adaptation state-of-the-art를 **상당히 크게 능가했다**고 주장한다. 특히 가장 어려운 Amazon $\rightarrow$ Webcam에서 큰 개선이 있었다고 강조한다.

다만 추출 텍스트에는 Office 결과의 정확한 수치 표가 포함되어 있지 않아, 구체적인 숫자를 여기서 재현할 수는 없다. 따라서 이 보고서는 논문이 Office benchmark에서 매우 강한 성능을 보고했다고만 서술한다.

### 4.8 Person re-identification로의 확장

논문은 DANN을 classification을 넘어 **descriptor learning**에도 적용한다. person re-identification에서는 각 이미지를 descriptor vector로 바꾸고, probe와 gallery 사이 descriptor distance로 matching한다. 따라서 여기서는 label predictor 대신 **descriptor predictor**가 들어가고, verification loss를 사용한다.

사용 데이터셋은 PRID, VIPeR, CUHK이고, 각 데이터셋을 서로 다른 domain으로 간주한다. 네트워크는 Yi et al. (2014)의 siamese-like architecture를 기반으로 하며, descriptor predictor에는 Binomial Deviance loss를 사용한다. domain classifier는 별도의 fully connected branch로 구성된다.

논문은 총 8개의 cross-dataset 실험에서 domain-adversarial training이 **일관되게 re-identification 성능을 향상시켰다**고 보고한다. 특히 PRID처럼 다른 데이터셋과 더 이질적인 domain이 끼는 경우 improvement가 상당했다고 서술한다.

이 결과는 중요하다. DANN이 단순한 multiclass classification 전용 기법이 아니라, **representation learning 문제 전반에 붙일 수 있는 일반적인 adaptation 메커니즘**임을 보여주기 때문이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 아이디어의 단순성과 일반성이다. domain adaptation 이론에서 요구하는 조건, 즉 source risk를 낮추면서 source-target divergence를 줄여야 한다는 조건을 신경망 구조로 매우 자연스럽게 옮겼다. 특히 GRL은 구현 난이도를 거의 늘리지 않으면서 adversarial objective를 backpropagation으로 학습하게 해 주는 매우 우아한 장치다. 이 때문에 방법론이 특정 모델군에 갇히지 않고, 거의 모든 feed-forward architecture에 쉽게 붙을 수 있다.

또 다른 강점은 이론과 실험의 연결이 좋다는 점이다. 논문은 단순히 성능만 보여주지 않고, $\mathcal{H}$-divergence와 PAD를 통해 왜 이런 방법이 맞는지 설명한다. toy example, PAD 분석, t-SNE visualization이 모두 같은 방향의 이야기를 한다. 즉, DANN은 באמת로 source와 target feature를 섞이게 만들고, 그 결과 target 성능이 좋아진다.

응용 범위가 넓다는 점도 강점이다. sentiment analysis, image classification, synthetic-to-real transfer, Office benchmark, person re-identification까지 포괄한다. 이는 논문의 핵심 아이디어가 task-specific trick이 아니라는 점을 설득력 있게 보여준다.

반면 한계도 분명하다.

첫째, 이론적 bound의 $\beta$ 항은 여전히 남아 있다. 즉, source와 target 양쪽에서 동시에 잘 되는 classifier가 애초에 존재하지 않으면, feature를 아무리 domain-invariant하게 만들어도 adaptation은 실패할 수 있다. 논문이 MNIST $\rightarrow$ SVHN에서 실패한 것이 이를 잘 보여준다. domain confusion이 항상 성능 향상을 보장하지는 않는다.

둘째, domain invariance를 강하게 밀면 class-discriminative 정보까지 손상될 위험이 있다. 논문은 $\lambda$를 통해 trade-off를 조절하지만, 실제로는 이 하이퍼파라미터가 매우 중요하며 잘못 설정하면 source classification도 망가질 수 있다. 논문도 shallow 실험에서는 grid search와 reverse validation을 사용했고, deep 실험에서는 heuristic schedule을 사용했다. 즉, 방법은 단순해 보여도 안정적인 학습을 위해서는 적절한 tuning이 필요하다.

셋째, domain classifier의 구조 선택은 다소 임의적이다. 논문도 domain classifier architecture가 임의적이며 더 튜닝하면 성능이 좋아질 수 있다고 인정한다. 다시 말해, “얼마나 강한 discriminator를 둘 것인가”는 실제 성능에 큰 영향을 줄 수 있는데, 이에 대한 체계적 분석은 부족하다.

넷째, 실험 결과의 보고 방식은 매우 긍정적이지만 일부 설정에서는 정량 표가 추출 텍스트에 완전하게 남아 있지 않다. 예를 들어 deep image classification과 Office benchmark의 세부 수치는 현재 제공된 텍스트에 전부 보존되지 않았다. 따라서 해당 부분은 저자 서술 중심으로만 해석할 수 있다.

비판적으로 보면, 이 논문은 domain confusion을 매우 강한 원리로 제시하지만, 실제 adaptation이 어려운 경우에는 **class-conditional alignment** 문제까지 자동으로 해결되지는 않는다. 즉, 전체 marginal distribution을 섞는다고 해서 클래스별 정렬까지 항상 잘 되지는 않는다. 이 논문 자체는 그 지점까지 다루지 않지만, 이후 연구들이 conditional adaptation, multi-discriminator, entropy minimization 등을 추가하게 되는 배경이 된다.

## 6. 결론

이 논문은 domain adaptation을 딥러닝 내부 표현학습 과정에 직접 통합한 대표적인 방법인 **DANN (Domain-Adversarial Neural Networks)**를 제안한다. 핵심 기여는 다음과 같이 요약할 수 있다.

첫째, source label prediction에는 유용하지만 domain discrimination에는 불리한 feature를 학습해야 한다는 아이디어를 명확히 제시했다.
둘째, 이를 label predictor, domain classifier, feature extractor의 adversarial 상호작용으로 정식화했다.
셋째, 이 min-max 구조를 **Gradient Reversal Layer**라는 매우 간단한 구성 요소로 backpropagation 안에 녹여냈다.
넷째, sentiment analysis, image classification, synthetic-to-real adaptation, Office benchmark, person re-identification 등 다양한 작업에서 강한 실험 결과를 제시했다.

실제 적용 측면에서 이 연구의 가치는 매우 크다. target 라벨이 없는 상황에서도 source의 supervision과 target의 unlabeled data만으로 transferable representation을 만들 수 있기 때문이다. 특히 synthetic-to-real, cross-device, cross-camera, cross-domain NLP 같은 실무 문제에 직접 연결된다.

향후 연구 관점에서도 이 논문은 매우 중요하다. 이후의 adversarial domain adaptation, domain confusion, invariant representation learning, fair representation learning, disentanglement 일부 흐름까지도 이 논문의 핵심 철학과 맞닿아 있다. 다시 말해, 이 논문은 단순한 한 편의 DA 논문이 아니라, **“representation은 task에는 유용하지만 특정 nuisance factor에는 불변이어야 한다”**는 현대 딥러닝의 중요한 원리를 매우 명쾌하게 구현한 고전적 연구라고 평가할 수 있다.
