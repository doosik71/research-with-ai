# Learning From Few Examples: A Summary of Approaches to Few-Shot Learning

- **저자**: Archit Parnami, Minwoo Lee
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2203.04291

## 1. 논문 개요

이 논문은 Few-Shot Learning(FSL), 즉 매우 적은 수의 학습 예제만으로도 새로운 작업에 일반화할 수 있는 학습 방법들을 정리한 survey paper이다. 저자들은 딥러닝이 대규모 라벨 데이터와 긴 학습 시간에 크게 의존한다는 점을 출발점으로 삼는다. 실제 응용에서는 데이터 수집, 전처리, 라벨링 비용이 크고, 의료·보안·프라이버시 같은 이유로 데이터를 충분히 확보하기 어려운 경우가 많기 때문에, 적은 데이터로도 학습 가능한 방법론이 중요하다고 본다.

논문이 다루는 핵심 연구 문제는 “학습 샘플이 극히 적을 때 어떻게 모델이 안정적으로 일반화할 수 있는가”이다. 특히 표준 supervised learning에서는 충분한 학습 데이터가 있어야 함수 $f(x;\theta)$를 잘 근사할 수 있지만, few-shot setting에서는 학습 샘플 수 $t$가 매우 작아 일반적인 empirical risk minimization이 불안정해진다. 저자들은 이 문제를 해결하려는 기존 연구들을 크게 meta-learning 기반 방법과 non-meta-learning 기반 방법으로 나누고, meta-learning 내부에서도 metric-based, optimization-based, model-based로 다시 체계화한다.

이 논문의 중요성은 새로운 알고리즘 제안 자체보다도, 당시까지의 FSL 연구 흐름을 구조적으로 정리하고, 어떤 가정 위에서 각 방법이 작동하는지 비교 가능하게 만든 데 있다. 특히 few-shot image classification을 중심으로 설명하면서도, object detection, segmentation, recommendation, reinforcement learning 등 다른 영역으로의 확장 가능성도 함께 언급한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 Few-Shot Learning을 단일한 방법이 아니라 “적은 데이터에서 일반화하기 위한 지식 이전 방식”의 관점으로 재구성하는 데 있다. 저자들은 대부분의 성과가 meta-learning에서 나왔지만, transfer learning 계열의 단순한 baseline도 매우 강력하다는 점을 함께 보여준다. 즉, FSL을 이해하려면 단순히 “meta-learning이 좋다”가 아니라, 어떤 방식으로 prior를 만들고, 그것을 새 작업에 어떻게 적용하는지 봐야 한다는 것이다.

논문은 meta-learning을 먼저 배경으로 설명한다. 일반 supervised learning에서는 하나의 task $T$에 대해 학습셋 $D_{train}$으로 파라미터 $\theta$를 최적화하고, 테스트셋 $D_{test}$에서 성능을 본다. 반면 meta-learning에서는 task distribution $p(T)$가 존재하고, 여러 개의 training task들로부터 “빠르게 적응하는 방법” 자체를 학습한다. 즉, 모델은 단순히 하나의 분류기를 배우는 것이 아니라, “새로운 분류 문제를 적은 샘플로 푸는 방법”을 배운다.

기존 접근과의 차별점은 분류 기준에 따라 명확히 드러난다. metric-based 방법은 임베딩 공간에서 support와 query의 거리를 비교해 분류하고, optimization-based 방법은 몇 번의 gradient update만으로 새로운 task에 빠르게 적응 가능한 초기화나 optimizer를 학습하며, model-based 방법은 memory나 sequence model처럼 빠른 적응을 구조적으로 내장한 아키텍처를 사용한다. 또한 cross-domain, generalized FSL, semi-supervised FSL 같은 변형 문제들도 별도의 hybrid setting으로 정리해, 실제 응용에서 FSL이 어떤 식으로 확장되는지 보여준다.

중요한 점은 이 논문이 survey라는 특성상 하나의 새 이론을 세우기보다, 다양한 방법의 공통 골격을 “task distribution 위의 학습”으로 통합해서 설명한다는 것이다. 따라서 핵심 기여는 새로운 loss나 architecture가 아니라, 연구 지형도를 정리하고 방법들의 장단점을 비교 가능한 틀 안에 배치한 데 있다.

## 3. 상세 방법 설명

논문은 먼저 일반 supervised learning과 meta-learning을 수식으로 구분한다. 일반 supervised learning에서는 테스트 샘플 $(x,y)\in D_{test}$에 대해

$$
y \approx f(x;\theta)
$$

가 되도록 하고, 파라미터는

$$
\theta = \arg\min_{\theta}\sum_{(x,y)\in D_{train}} L(f(x,\theta), y)
$$

로 학습한다. 여기서 $L$은 예측과 정답 사이의 오차를 측정하는 loss function이다.

반면 meta-learning에서는 각 task $T_i$가 자체적인 학습셋과 테스트셋 $D_i = \{D_i^{train}, D_i^{test}\}$를 가진다. 목표는 새 task의 소량 학습 예제 $D_i^{train}$를 보고 해당 task의 테스트 데이터에 잘 일반화하는 함수

$$
y \approx f(D_i^{train}, x; \theta)
$$

를 학습하는 것이다. 최적화 목표는

$$
\theta = \arg\min_{\theta}\sum_{D_i \in D_{meta-train}}\sum_{(x,y)\in D_i^{test}} L(f(D_i^{train}, x; \theta), y)
$$

로 주어진다. 즉, task 내부의 작은 training set을 바탕으로 task test set에서 잘 작동하도록 메타 수준의 파라미터를 최적화한다.

Few-shot classification은 보통 $M$-way $K$-shot 문제로 정의된다. 여기서 $M$은 클래스 수, $K$는 클래스당 support example 수이다. 따라서 support set 크기는 $|D_{train}| = M \times K$가 된다. 이 논문은 Omniglot, miniImageNet, FC100, tieredImageNet 같은 대표 데이터셋을 소개한다.

메타러닝 기반 FSL의 첫 번째 큰 축은 metric-based 방법이다. 이 계열은 입력 $x$를 임베딩 함수 $g$로 저차원 특징 공간으로 보낸 뒤, query와 support 간 거리 또는 유사도로 분류한다. 이때 거리 함수는 고정된 Euclidean, cosine일 수도 있고, 학습 가능한 네트워크일 수도 있다. 학습은 episode 단위로 진행된다. 각 episode에서 $M$개 클래스를 샘플링하고, 각 클래스에서 support와 query를 뽑는다. 이후 support-query 간 거리를 계산하고, 같은 클래스는 가깝고 다른 클래스는 멀어지도록 loss를 최소화한다.

Siamese Networks는 두 입력 이미지를 같은 CNN에 통과시켜 얻은 임베딩 간 L1 거리를 계산하고, 이를 sigmoid를 통해 같은 클래스일 확률로 바꾼다. 학습에는 binary cross-entropy를 사용한다. Matching Networks는 support와 query 사이 cosine similarity를 attention kernel로 사용한다. query $\hat{x}$와 support sample $x_k$의 attention은

$$
a(\hat{x}, x_k)=\frac{e^{\cos(f(\hat{x}), g(x_k))}}{\sum_{k=1}^{t} e^{\cos(f(\hat{x}), g(x_k))}}
$$

처럼 정의되며, 최종 예측은 support label들의 attention-weighted sum이다.

Prototypical Networks는 클래스별 prototype을 support embedding의 평균으로 만든다.

$$
v_c = \frac{1}{|S_c|}\sum_{(x_k,y_k)\in S_c} g_{\theta_1}(x_k)
$$

그 뒤 query 임베딩과 각 prototype 사이의 거리를 비교해 softmax 확률을 계산한다.

$$
P(y=c|\hat{x}) = \text{softmax}(-d(g_{\theta_1}(\hat{x}), v_c))
$$

loss는 정답 클래스의 negative log-likelihood이다.

$$
L(\theta_1) = - \log P_{\theta_1}(y=c|\hat{x})
$$

Relation Networks는 거리 함수를 직접 설계하지 않고, support prototype과 query embedding을 이어붙인 뒤 또 다른 CNN이 relation score를 출력하게 만든다. 정답 클래스의 relation score는 1, 나머지는 0이 되도록 mean squared error를 최소화한다. TADAM은 task-dependent embedding을 도입해 task별로 feature extractor를 조절하고, distance metric 앞에 learnable temperature $\lambda$를 곱해 softmax scaling 효과를 조정한다.

$$
P_{\lambda}(y=c|x)=\text{softmax}(-\lambda d(g_{\theta_1}(x), v_c))
$$

TapNet은 class prototype 대신 학습 가능한 reference vector $\Phi_c$와 task별 projection space $M$을 도입한다.

$$
P(y=c|x)=\text{softmax}(-d(M(g_{\theta_1}(x)), M(\Phi_c)))
$$

CTM은 Category Traversal Module을 통해 현재 task에 중요한 feature dimension만 강조하는 mask를 생성한 뒤 metric learner에 넘긴다. CAN, SAML 같은 attention-based 변형도 이 범주에서 소개된다.

두 번째 축은 optimization-based meta-learning이다. 여기서는 “적은 데이터에서도 잘 적응하는 optimization procedure”를 배우는 것이 핵심이다. 논문은 learner $f_\theta$와 meta-learner $g_\phi$를 구분한다. meta-learner는 support set을 보고 learner의 파라미터를 더 좋은 값으로 업데이트한다.

$$
\theta^* = g_\phi(\theta, D^{train})
$$

LSTM Meta-Learner는 gradient descent 자체를 LSTM이 대체하도록 설계한다. 일반 gradient update

$$
\theta_{i+1} = \theta_i - \alpha \nabla f(\theta_i)
$$

대신,

$$
\theta_{i+1} = g_i(\nabla f(\theta_i), \theta_i; \phi)
$$

같이 optimizer를 학습한다. 즉, 어떤 gradient가 들어왔을 때 어떻게 업데이트할지를 LSTM이 배운다.

MAML은 가장 대표적인 optimization-based 방법으로, 새로운 task가 들어왔을 때 소수의 gradient step만으로 빠르게 적응할 수 있는 초기 파라미터 $\theta$를 찾는다. 각 task $i$에 대해 먼저 support set loss로 task-specific adaptation을 수행한다.

$$
\theta_i^* = \theta - \alpha \nabla_\theta L_i^{train}
$$

그다음 adapted parameter $\theta_i^*$로 query loss를 계산하고, 그 query loss들의 합으로 원래 초기 파라미터 $\theta$를 업데이트한다. 즉, “몇 번만 업데이트해도 잘 되는 초기점”을 메타 수준에서 학습하는 구조다. 논문은 Proto-MAML, TAML, MAML++, HSML, CAVIA 같은 MAML 파생 방법들도 설명한다. 예를 들어 CAVIA는 파라미터를 shared part와 context part로 나누고, test time에는 low-dimensional context parameter만 업데이트한다.

Meta-Transfer Learning(MTL)은 깊은 네트워크 전체를 task마다 적응시키면 overfitting될 수 있다는 점을 지적한다. 그래서 pretrained backbone $\Theta$는 유지하고, 마지막 classifier와 scale-shift 파라미터만 meta-learn한다. LEO는 고차원 파라미터 공간에서 직접 최적화하는 대신, latent space $z$를 만들어 그 공간에서 adaptation을 수행한 뒤 decoder로 classifier parameter를 생성한다. 저자들은 이를 통해 저데이터 환경에서 더 안정적인 적응이 가능하다고 설명한다.

세 번째 축은 model-based meta-learning이다. 이 계열은 $P_\theta(y|x)$의 특정 형태를 가정하기보다, 빠른 적응이 가능하도록 memory나 sequence processing 구조를 모델 안에 넣는다. MANN은 Neural Turing Machine 기반 외부 메모리를 사용해 sample-label binding을 빠르게 저장하고 불러온다. 메모리 read는 key와 memory row의 cosine similarity 기반 soft attention으로 수행된다. MM-Net은 Matching Networks에 key-value memory를 결합한다. Meta Networks는 fast weights를 외부 메모리에 저장해 빠르게 task-specific adaptation을 수행하고, CSN은 memory에서 가져온 task-specific shift로 neuron activation을 조절한다. SNAIL은 temporal convolution과 causal attention을 교차 배치해, episode 내에서 이전 support-label sequence를 보고 마지막 query를 예측하는 sequence-to-sequence 형태의 meta-learning을 수행한다.

논문은 여기에 더해 hybrid approaches도 정리한다. Cross-modal FSL은 이미지 외의 텍스트 같은 다른 modality를 활용하고, semi-supervised FSL은 unlabeled data를 함께 사용한다. Generalized FSL은 query가 base class인지 novel class인지 모두 포함해 분류해야 하는 설정이다. Generative FSL은 hallucinator로 synthetic sample을 생성한다. Cross-domain FSL은 train/test domain이 다를 때를 다루고, transductive FSL은 query set 전체 구조를 함께 사용해 예측한다. Unsupervised FSL은 support label조차 없고, Zero-shot learning은 support example이 전혀 없는 경우다.

마지막으로 non-meta-learning 접근도 상세히 다룬다. 대표적으로 pretrained network의 embedding 위에서 nearest neighbor 분류를 수행하는 SimpleShot이 있다. 이는 feature centering과 L2 normalization 후 Euclidean distance로 분류한다. 또 다른 접근은 pretrained embedding 위에 새로운 classifier만 학습하는 것이다. transductive fine-tuning은 query의 unlabeled structure까지 활용한다. 예를 들어 Dhillon et al.이 제시한 목적함수는

$$
\theta^* = \arg\min_{\theta} \frac{1}{|S|}\sum_{(x,y)\in S} -\log p_\theta(y|x) + \frac{1}{|Q|}\sum_{x\in Q} H(p_\theta(.|x))
$$

인데, 첫 항은 support label fitting이고 둘째 항은 query prediction entropy를 낮추는 regularizer다. 즉, unlabeled query에 대해 더 확신 있는 예측을 하도록 유도한다. 또한 Laplacian regularization 기반 transductive inference도 소개한다.

## 4. 실험 및 결과

이 논문은 새로운 실험을 설계해 제시하는 논문이라기보다, 기존 few-shot classification 결과를 종합해 비교하는 survey이다. 따라서 “실험” 섹션은 개별 모델들의 정량 결과를 표와 서술로 정리한 성격이 강하다. 저자들은 대표 벤치마크로 miniImageNet의 5-way 1-shot, 5-shot 분류 정확도를 중심 지표로 사용한다. 데이터셋 맥락으로는 Omniglot, miniImageNet, FC100, tieredImageNet을 대표적으로 소개하지만, 최종 비교표는 miniImageNet 기준 결과를 모은 것이다.

표 9에 따르면, 초기 Matching Networks는 5-way 1-shot에서 43.56%, 5-shot에서 55.31%를 기록한다. 이후 MAML은 48.7% / 63.15%, Prototypical Networks는 49.42% / 68.2%, Relation Networks는 50.44% / 65.32%로 개선된다. TADAM은 58.5% / 76.7%, MTL은 61.2% / 75.5%, LEO는 61.76% / 77.59%, CTM은 62.05% / 78.63%, CAN은 63.85% / 79.44%를 기록한다. 특히 non-meta-learning 계열의 SimpleShot은 64.29% / 81.5%로 매우 강력한 baseline임을 보여준다. hybrid 계열에서는 AM3가 65.3% / 78.1%, LST가 70.1% / 78.7%를 기록했다고 정리되어 있다.

저자들은 2016년부터 2020년 1월까지 few-shot image classification 성능이 miniImageNet 5-way 1-shot 기준 약 43%에서 80% 수준까지 향상되었다고 요약한다. 이는 few-shot learning이 빠르게 발전해 왔음을 보여주지만, 동시에 특정 계열이 절대적으로 우세하다고 결론 내리지는 않는다. metric-based, optimization-based, hybrid, non-meta-learning이 모두 상위권을 차지하고 있기 때문이다.

실험 결과의 중요성은 두 가지다. 첫째, 복잡한 meta-learning 기법만이 정답은 아니라는 점이다. pretrained embedding 기반의 단순한 transfer learning baseline도 매우 경쟁력이 있다. 둘째, 대부분의 성능 평가는 같은 분포에서 sampled episode에 기반하고 있어, 실제 환경의 distribution shift나 generalized setting까지 충분히 반영하지 못한다는 점이다. 저자들은 이 한계를 뒤의 challenges 섹션에서 명시적으로 지적한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 few-shot learning 연구를 방법론적 기준에 따라 명확하게 조직했다는 점이다. 단순히 논문들을 나열하는 수준이 아니라, meta-learning의 수학적 정의에서 출발해 metric-based, optimization-based, model-based, hybrid, non-meta-learning으로 이어지는 구조를 제공한다. 그래서 독자는 개별 방법이 무엇을 학습하는지, 즉 metric인지, optimizer인지, architecture인지 구분하면서 읽을 수 있다.

또 다른 강점은 대표 방법들의 핵심 메커니즘을 비교 가능한 형태로 정리한 점이다. 특히 표 4, 5, 6, 7, 8, 9를 통해 방법별 핵심 아이디어, 예측 방식, 손실 함수, 강점과 약점을 한 자리에서 비교할 수 있다. survey 논문으로서 교육적 가치가 높고, 입문자뿐 아니라 연구 방향을 정리하려는 독자에게도 유용하다.

또한 저자들은 meta-learning만 강조하지 않고, transfer learning 기반 단순 기법들이 실제로 강력하다는 점을 포함시킨다. 이는 FSL 연구에서 흔히 발생하는 “복잡한 방법이 더 낫다”는 편향을 줄여 준다. 실제로 SimpleShot 같은 baseline이 매우 높은 정확도를 보인다는 정리는 중요한 메시지다.

한편 한계도 분명하다. 우선 이 논문은 survey이므로 새로운 방법, 새로운 이론 분석, 새로운 empirical finding을 제시하지 않는다. 따라서 독창적 기여는 “정리와 비교”에 있다. 둘째, 정리 범위가 주로 few-shot image classification 중심이어서 다른 도메인 언급은 상대적으로 짧다. 논문은 오디오, 그래프, 추천, 강화학습 등으로 확장을 언급하지만, 각 영역의 방법론 차이를 깊게 분석하지는 않는다.

셋째, 논문이 인용하는 성능 비교는 대부분 동일한 benchmark, 특히 miniImageNet 중심이다. 저자들도 인정하듯이 실제 응용에서는 $M$과 $K$가 고정되지 않고, train/test task distribution이 달라질 수 있으며, base와 novel class를 동시에 분류해야 하는 경우도 많다. 따라서 벤치마크 성능만으로 실용성을 판단하기 어렵다.

넷째, 일부 설명은 survey 특성상 압축적이다. 예를 들어 각 모델의 세부 학습 스케줄, backbone 차이, augmentation, confidence interval 등은 원문 요약 수준으로만 제시된다. 따라서 어떤 방법의 정확한 구현 세부가 필요하면 해당 원 논문을 직접 봐야 한다. 이 점은 논문 자체도 암묵적으로 전제하고 있다.

비판적으로 보면, 이 논문은 FSL의 “왜 작동하는가”에 대한 이론적 통찰보다는 “어떻게 분류할 것인가”에 더 초점을 맞춘다. 예를 들어 empirical risk minimizer의 불안정성, task distribution 가정의 타당성, benchmark saturation 문제 등에 대한 이론적 논의는 제한적이다. 그러나 survey로서 목적이 넓은 지형도 제시에 있다는 점을 고려하면 이는 어느 정도 의도된 범위 제한으로 볼 수 있다.

## 6. 결론

이 논문은 Few-Shot Learning을 적은 샘플로 새로운 task에 일반화하는 문제로 정의하고, 그 해결책을 크게 meta-learning과 non-meta-learning으로 나누어 정리한다. 메타러닝 기반에서는 metric-based, optimization-based, model-based가 핵심 축이며, 여기에 cross-domain, generalized, semi-supervised, transductive 같은 hybrid setting이 확장 문제로 연결된다. 반면 pretrained embedding을 활용한 transfer learning baseline 역시 매우 강력하며, 실제로 일부 벤치마크에서는 복잡한 메타러닝 방법보다 더 좋은 성능을 보인다고 정리한다.

실제 적용 측면에서 이 연구는 “few-shot learning에서는 반드시 복잡한 메타러닝이 필요하다”는 단순한 결론 대신, 문제 설정과 deployment 조건에 따라 적절한 접근이 달라진다는 점을 보여준다. 향후 연구에서는 같은 분포의 고정된 $M$-way $K$-shot episode를 넘어서, cross-domain generalization, generalized few-shot classification, 이미지 외 데이터 도메인, 그리고 실제 환경에서의 유연한 task 구성에 대응하는 방향이 중요하다는 메시지를 남긴다.

즉, 이 논문의 핵심 기여는 few-shot learning 연구를 체계적으로 지도화하고, 어떤 방법이 어떤 가정 위에서 강점을 가지는지 분명히 보여준 데 있다. 원문 기준으로 보았을 때, 이는 FSL 분야의 입문과 연구 방향 설정 모두에 유용한 정리 논문이다.
