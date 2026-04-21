# An Overview of Deep Learning Architectures in Few-Shot Learning Domain

- **저자**: Shruti Jadon, Aryan Jadon
- **발표연도**: 2023 *(제공된 본문에 포함된 arXiv v4 날짜: 2023-04-16 기준)*
- **arXiv**: https://arxiv.org/abs/2008.06365

## 1. 논문 개요

이 논문은 few-shot learning 분야에서 사용되어 온 대표적인 딥러닝 아키텍처들을 정리하는 **survey/review 논문**이다. 즉, 새로운 알고리즘 하나를 제안하는 논문이라기보다, 적은 수의 예시만으로도 새로운 개념을 빠르게 일반화하려는 문제를 기존 연구들이 어떻게 다뤄 왔는지를 구조적으로 설명하는 데 목적이 있다.

논문이 다루는 핵심 연구 문제는 분명하다. 현대 딥러닝은 뛰어난 성능을 내지만, 일반적으로 대규모 라벨 데이터와 많은 파라미터 업데이트를 필요로 한다. 반면 인간은 한두 개의 예시만 보고도 새로운 개념을 빠르게 이해하고 변형 사례를 인식할 수 있다. Few-shot learning은 바로 이 간극을 줄이려는 시도이며, 특히 컴퓨터 비전의 object categorization 문제에서 중요한 의미를 가진다.

저자들은 이 문제의 중요성을 크게 세 가지 맥락에서 설명한다. 첫째, 의료나 제조처럼 데이터가 충분하지 않은 산업에서는 대규모 데이터 기반 딥러닝이 곧바로 적용되기 어렵다. 둘째, 적은 데이터에서는 과적합과 느린 최적화가 더 심각해진다. 셋째, 실제 응용에서는 새로운 클래스나 새로운 태스크에 빠르게 적응하는 능력이 필요하다. 이런 배경에서 논문은 few-shot learning용 딥러닝 접근을 네 부류로 나누어 소개한다. 즉, **Data Augmentation Methods**, **Metrics Based Methods**, **Models Based Methods**, **Optimization Based Methods**이다.

또한 논문은 대표 데이터셋으로 **Omniglot**과 **Mini-ImageNet**을 소개한다. Omniglot은 50개 언어, 1623개 문자, 각 문자당 20개 샘플로 구성되며 인간과 기계의 학습 능력을 비교하기 위해 자주 쓰인다. Mini-ImageNet은 few-shot image classification을 위해 설계된 소형 ImageNet 계열 데이터셋으로 설명된다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 few-shot learning을 하나의 단일 기법으로 보지 않고, “적은 데이터에서 어떻게 일반화 능력을 확보할 것인가”라는 질문에 대한 여러 설계 철학으로 나누어 보는 데 있다.

첫 번째 축은 **데이터를 늘리거나 풍부하게 만드는 방식**이다. Data augmentation은 적은 샘플 수 자체를 직접 보완하려는 접근이다. 저자들은 단순 기하 변환, 색 공간 변환, GAN 기반 생성, neural style transfer 등을 예로 든다. 다만 데이터 분포가 원래 치우쳐 있으면 증강된 데이터도 왜곡될 수 있고, 과적합 위험도 남는다고 지적한다.

두 번째 축은 **좋은 embedding과 similarity metric을 학습하는 방식**이다. Metrics-based methods는 “적은 데이터에서도 분류기 자체를 새로 학습하기보다, 예시 간 유사도를 잘 재는 표현 공간을 배우자”는 생각에 기반한다. Siamese Networks와 Matching Networks가 대표적이다.

세 번째 축은 **외부 메모리나 빠른 가중치 적응 메커니즘을 모델 구조 안에 포함하는 방식**이다. Models-based methods는 인간이 과거 경험을 기억하고 새 태스크에 빠르게 적응하는 방식을 본떠, 메모리 모듈이나 meta-learner를 통해 few-shot generalization을 달성하려 한다. Neural Turing Machine, Memory-Augmented Neural Network, Meta Networks가 이 범주에 있다.

네 번째 축은 **최적화 자체를 few-shot에 맞게 바꾸는 방식**이다. Optimization-based methods는 적은 샘플에서도 몇 번의 gradient step만으로 새 태스크에 잘 적응할 수 있도록 초기 파라미터나 업데이트 규칙을 학습한다. MAML과 LSTM Meta-Learner가 여기에 속한다.

기존 접근과의 차별점이라는 관점에서 보면, 이 논문은 개별 모델의 성능 비교를 깊게 파고들기보다, 서로 다른 few-shot 설계 원리를 “임베딩 학습”, “메모리 기반 적응”, “초기화/최적화 학습”이라는 큰 흐름 속에 배치해 이해하게 해 준다. 따라서 이 논문의 기여는 새로운 방법 제안이라기보다, 분야 입문자에게 개념 지도를 제공하는 데 있다.

## 3. 상세 방법 설명

### 전체 분류 체계

논문은 few-shot 딥러닝 아키텍처를 네 범주로 나누지만, 실제로 가장 자세히 설명하는 것은 **metrics-based**, **models-based**, **optimization-based** 세 계열이다. 각 방법의 공통 목표는 적은 수의 labeled example만 보고도 새로운 카테고리나 태스크에 빠르게 적응하는 것이다. 다만 그 수단은 다르다. 어떤 방법은 similarity를 직접 학습하고, 어떤 방법은 메모리에 저장하며, 어떤 방법은 좋은 초기화나 업데이트 규칙을 학습한다.

### Siamese Networks

Siamese Network는 동일한 파라미터를 공유하는 두 개의 병렬 네트워크로 구성된다. 각 네트워크는 입력 하나씩을 받아 feature representation을 만든다. 중요한 점은 이 구조가 “클래스를 직접 분류”하는 대신, 두 입력이 같은 클래스인지 다른 클래스인지를 **구별(discriminate)** 하도록 훈련된다는 점이다.

이 방법을 쓰려면 데이터 전처리가 특별하다. 일반 분류처럼 개별 샘플을 넣는 것이 아니라, 반드시 두 종류의 pair를 만든다.

- similar pair: 같은 클래스의 두 이미지
- dissimilar pair: 다른 클래스의 두 이미지

그리고 라벨은 논문 설명에 따라 similar에 $y=1$, dissimilar에 $y=0$을 부여한다. 이 pair가 Siamese 구조에 입력되고, 마지막에서 contrastive loss로 두 임베딩 간 거리를 학습한다.

논문이 제시한 contrastive loss는 다음과 같다.

$$
Loss = (1 - Y)\frac{1}{2}D_w^2 + Y\frac{1}{2}\max(0, m - D_w)^2
$$

여기서 거리 항은 다음과 같이 정의된다.

$$
D_w = \sqrt{G_w(X_1) - G_w(X_2)}
$$

본문 표기는 다소 깨져 있지만, 설명 의도는 두 입력 $X_1, X_2$를 shared network $G_w$에 통과시켜 얻은 representation 사이의 거리를 사용한다는 것이다. 저자 설명에 따르면 이 loss는 두 부분으로 나뉜다. 하나는 비슷한 pair의 “에너지”를 낮추는 항이고, 다른 하나는 서로 다른 pair의 에너지를 margin $m$ 이상 벌리도록 하는 항이다. 특히 $m$보다 충분히 멀어진 dissimilar pair는 더 이상 loss에 크게 기여하지 않게 하여, 모델이 이미 잘 구분한 pair에 불필요하게 집중하지 않도록 한다.

이 구조는 face detection, signature verification, handwriting detection, spam detection 등에서 유용하다고 설명된다. 동시에 논문은 contrastive loss만으로 decision boundary를 충분히 잘 학습하지 못하는 경우가 있어 이후 triplet loss 같은 대안이 제안되었다고 지적한다. 또 transfer learning을 Siamese 구조에 적용할 때는 source domain과 target domain이 유사해야 하며, 그렇지 않으면 domain adaptation 문제가 중요하다고 명시한다.

### Matching Networks

Matching Networks는 작은 labeled support set과 unlabeled query example이 주어졌을 때, support set을 참고해 query의 라벨을 예측하도록 설계된 구조다. 저자들은 이를 **parametric model의 표현 학습 능력**과 **non-parametric model의 빠른 적응 능력**을 결합한 것으로 설명한다.

핵심 개념은 다음과 같다.

- **Label Set**: 현재 episode에서 고려하는 클래스 집합
- **Support Set**: 각 클래스에서 뽑은 few-shot 예시 집합
- **Batch**: 같은 label set에서 뽑은 또 다른 샘플 집합
- **N-way K-shot**: 클래스 수가 $N$, 클래스당 예시 수가 $K$

이 모델은 support set과 query를 같은 embedding space로 보낸 뒤, cosine similarity를 이용해 query와 support example들의 유사도를 계산한다. Support set의 각 샘플 $(x_i, y_i)$는 feature extractor $g$와 bidirectional LSTM을 거쳐 **Fully Contextual Embeddings**가 된다. Query $\hat{x}$ 역시 유사한 방식으로 contextual embedding을 얻는다. 이후 kernel과 cosine similarity를 통해 query가 어느 support sample들과 가장 유사한지 계산한다.

Matching Networks의 중요한 특징은 **훈련 절차 자체를 few-shot 테스트 상황과 비슷하게 만든다**는 점이다. 즉, training에서도 매번 작은 subset을 뽑아 support set과 query를 구성하고, 새로운 태스크를 연속적으로 바꾸어가며 episode 형태로 학습한다. 논문은 이것이 전통적인 mini-batch 학습과 다르며, few-shot test condition을 train time에 모사하려는 설계라고 설명한다.

### Neural Turing Machine

Neural Turing Machine(NTM)은 few-shot 전용 모델은 아니지만, 이후 MANN을 이해하기 위한 기반 구조로 소개된다. NTM은 **controller neural network**와 **external memory matrix**로 구성된다. 메모리 행렬 $M_t$는 $R$개의 row와 $C$개의 column을 가지며, controller는 매 시점마다 입력을 받아 출력을 생성하고 동시에 메모리에 read/write를 수행한다.

중요한 점은 메모리 접근이 완전히 이산적 주소 지정이 아니라, 모든 memory row에 대해 연속적인 가중치를 두는 **differentiable addressing**이라는 점이다. 즉, row index를 직접 선택하면 미분이 불가능하므로, weight vector $w_t$를 사용해 각 row를 어느 정도 읽고 쓸지 부드럽게 정한다.

가중치 벡터는 정규화되어 다음 조건을 만족한다.

$$
0 \le w_t(i) \le 1, \quad \sum_{i=1}^{R} w_t(i)=1
$$

read 연산은 메모리 row들의 가중합이다.

$$
r_t \leftarrow \sum_{i=1}^{R} w_t(i)M_t(i)
$$

write는 erase와 add 두 단계로 진행된다. 먼저 erase vector $e_t$로 기존 정보를 일부 지우고,

$$
M_t^{erased}(i) \leftarrow M_{t-1}(i)[1 - w_t(i)e_t]
$$

그 다음 add vector $a_t$를 더해 새로운 정보를 기록한다.

$$
M_t(i) \leftarrow M_t^{erased} + w_t(i)a_t
$$

addressing은 네 단계로 설명된다.

1. **Content-based addressing**: key vector $k_t$와 각 memory row의 cosine similarity를 계산한다.
2. **Interpolation**: 현재 content-based weight와 이전 시점의 weight를 gate $g_t$로 섞는다.
3. **Shift**: location-based shift를 적용한다.
4. **Sharpening**: 이동 후 흐려진 분포를 다시 날카롭게 만든다.

cosine similarity는 다음과 같다.

$$
K(u, v)=\frac{u \cdot v}{||u|| \, ||v||}
$$

content-based weight는 다음과 같이 softmax로 정규화된다.

$$
w_t^c(i)=\frac{\exp(\beta_t K(k_t, M_t(i)))}{\sum_j \exp(\beta_t K(k_t, M_t(j)))}
$$

이후 interpolation은

$$
w_t^g \leftarrow g_t w_t^c + (1-g_t)w_{t-1}
$$

shift는

$$
\tilde{w}_t(i) \leftarrow \sum_{j=0}^{R-1} w_t^g(j)s_t(i-j)
$$

sharpening은

$$
w_t(i) \leftarrow \frac{\tilde{w}_t(i)^{\gamma_t}}{\sum_j \tilde{w}_t(j)^{\gamma_t}}
$$

로 정의된다. 요점은 모든 연산이 미분 가능하므로, 모델 전체를 end-to-end로 학습할 수 있다는 것이다.

### Memory-Augmented Neural Networks

MANN은 few-shot learning에 맞게 NTM을 수정한 구조다. 논문에 따르면 MANN은 NTM과 달리 **location-based addressing을 제거하고 content-based addressing만 사용**한다. 이유는 few-shot classification에서는 “위치 자체”보다 “현재 입력이 기억 속 어떤 내용과 비슷한가”가 더 중요하기 때문이다.

읽기(read)는 NTM과 거의 같지만, content-based read weight $w_t^r$만 사용한다.

$$
r_t \leftarrow \sum_i w_t^r(i) M_t(i)
$$

read weight는 cosine similarity 기반 softmax로 얻는다.

쓰기(write)는 최근에 읽은 위치와 least recently used location 사이를 보간하여 결정한다. 논문은 이를 **Least Recently Used Access (LRUA)** 모듈이라 부른다. 직관적으로 말하면, 현재 입력이 기존 메모리와 매우 유사하면 관련 메모리를 업데이트하고, 전혀 새로우면 최근에 쓰지 않은 위치에 기록하는 식이다. 따라서 외부 메모리를 이용해 새로운 클래스 정보를 빠르게 저장하고 재사용할 수 있다.

### Meta Networks

Meta Networks는 base learner와 meta learner가 함께 작동하는 model-based meta-learning 구조다. 논문이 강조하는 핵심은 **loss gradient 같은 higher-order meta information을 이용해 fast weights를 생성한다**는 점이다.

구성 요소는 네 부분으로 설명된다.

- 느린 embedding function $f_\theta$
- 느린 base learner $g_\phi$
- embedding function의 fast weight를 만드는 LSTM $F_w$
- base learner의 fast weight를 만드는 network $G_v$

즉, 일반적인 느린 가중치(slow weights)로 이루어진 네트워크가 있고, 별도의 meta learner가 현재 태스크 상태를 나타내는 gradient 정보를 받아 빠르게 적응하는 fast weights를 생성한다. 최종 예측은 slow weights와 fast weights의 조합으로 만들어진다.

훈련 시에는 Matching Networks처럼 support set $S=(x'_i, y'_i)$와 training set $U=(x_i, y_i)$를 사용한다. 전체적으로는 task space에서 base learner가 특정 태스크를 배우고, meta space에서 meta learner가 여러 태스크를 통틀어 공통 적응 규칙을 배운다. 이 구조의 장점은 SGD만으로 느리게 파라미터를 갱신하는 대신, meta learner가 현재 태스크에 맞춘 빠른 적응을 직접 제공한다는 데 있다.

### Model-Agnostic Meta-Learning (MAML)

MAML의 목표는 새 태스크가 왔을 때 몇 번의 gradient step만으로 빠르게 적응할 수 있는 **좋은 초기 파라미터 $\theta$**를 찾는 것이다. 논문은 이것을 “각 태스크의 최적점에 가까운 공통 초기화”를 학습하는 문제로 설명한다.

메타 학습 과정에서는 task distribution $P(T)$에서 태스크 $T_i$를 샘플링한다. 각 태스크마다 $K$개의 샘플로 inner-loop adaptation을 수행하고, 적응 후 새로운 샘플에서의 test error를 측정한다. 그리고 이 test error가 작아지도록 원래 초기 파라미터 $\theta$를 업데이트한다. 즉, 태스크별 학습 후의 성능이 메타 수준의 학습 신호가 된다.

논문은 loss를 다음과 같이 제시한다.

$$
L_{T_i}(f_\phi)=\sum_{x^{(j)}, y^{(j)} \sim T_i} ||f_\phi(x^{(j)}) - y^{(j)}||_2^2
$$

본문은 세부 알고리즘을 원 논문 [13]을 참조하라고 하고 있어, inner-loop/outer-loop의 정확한 업데이트 식을 이 survey 본문만으로 완전하게 재구성할 수는 없다. 다만 핵심 메시지는 명확하다. MAML은 태스크-불변적인 표현을 직접 강제하기보다, **빠른 gradient-based adaptation이 가능한 initialization**을 메타 수준에서 학습한다.

### LSTM Meta-Learner

LSTM Meta-Learner는 최적화 과정을 LSTM 셀 업데이트와 연결해서 본다. 논문은 gradient descent 업데이트

$$
\theta_t = \theta_{t-1} - \alpha_t \nabla L_t
$$

와 LSTM cell state 업데이트

$$
c_t = f_t c_{t-1} + i_t \bar{c}_t
$$

가 구조적으로 유사하다고 설명한다. 여기서

- $c_{t-1}$를 이전 파라미터 $\theta_{t-1}$로,
- $i_t$를 learning rate 역할로,
- $\bar{c}_t$를 gradient 정보로

해석하면, LSTM이 곧 “파라미터 업데이트 규칙”을 학습하는 메타 최적화기로 볼 수 있다는 것이다.

논문은 input gate와 forget gate를 다음과 같이 쓴다.

$$
i_t = \sigma(W_I \cdot [\nabla L_t, L_t, \theta_{t-1}, i_{t-1}] + b_I)
$$

$$
f_t = \sigma(W_F \cdot [\nabla L_t, L_t, \theta_{t-1}, f_{t-1}] + b_F)
$$

즉, 현재 gradient, 현재 loss, 이전 파라미터, 이전 gate 상태를 입력으로 받아, 이번 step에서 어느 정도 업데이트할지 학습한다. 이 방식은 단순히 learning rate를 사람이 정하는 것이 아니라, 메타 수준에서 최적의 업데이트 규칙을 학습하려는 시도다.

데이터 구성도 일반 supervised learning과 다르다. 여러 태스크의 집합 $\mathcal{D}$가 있고, 각 태스크는 $D^{train}$과 $D^{test}$로 나뉜다. 또한 메타 수준에서는 $D_{meta-train}$, $D_{meta-validation}$, $D_{meta-test}$로 다시 나뉜다. 모델은 태스크별 소량 데이터로 learner를 훈련시키고, 그 결과를 바탕으로 meta-learner가 업데이트 규칙을 점점 더 잘 학습한다.

## 4. 실험 및 결과

이 논문은 개별 모델의 새로운 실험을 중심으로 한 empirical paper가 아니라, 기존 few-shot learning 아키텍처를 정리하는 survey이다. 따라서 통일된 실험 설정 아래 여러 방법을 재현하고 직접 비교한 대규모 정량 실험 결과표를 제시하는 논문은 아니다. 이 점은 분명히 해둘 필요가 있다.

다만 실험 맥락에서 논문이 제공하는 정보는 다음과 같다.

첫째, few-shot learning에서 널리 사용되는 데이터셋으로 **Omniglot**과 **Mini-ImageNet**을 소개한다. Omniglot은 105×105 grayscale 문자 이미지로 구성되며, background set과 evaluation set을 엄격히 분리해 few-shot generalization을 평가하도록 설계되었다고 설명한다. Mini-ImageNet은 ImageNet의 축소 버전으로 few-shot image classification용 벤치마크로 언급된다.

둘째, Matching Networks와 Meta Networks, MAML 등은 공통적으로 **episodic setting** 또는 그것과 유사한 train/test condition matching을 중요하게 여긴다고 설명한다. 즉, 실험은 보통 $N$-way $K$-shot 형태로 설계되며, 훈련 시에도 support set과 query set을 인위적으로 구성해 테스트 상황을 모사한다.

셋째, 결과에 관한 서술은 전반적으로 정성적이다. 예를 들어 Siamese Networks는 적은 데이터로도 face detection, signature verification, handwriting-related task에서 높은 효용을 보였다고 소개한다. Matching Networks와 Meta Networks는 classification에서 state-of-the-art accuracy를 달성하는 방향으로 발전해 왔다고 요약한다. 반면 object detection이나 image segmentation 같은 더 복잡한 vision task에서는 여전히 어려움이 있다고 결론부에서 언급한다.

즉, 이 논문에서 실험 섹션의 핵심 가치는 “어떤 방법이 어떤 벤치마크에서 몇 퍼센트 향상되었다”는 수치 자체보다, **few-shot 분야에서 어떤 평가 프로토콜과 어떤 문제 유형이 중요하게 다뤄지는가**를 보여주는 데 있다. 구체적인 수치 비교나 동일 조건 재현 결과는 제공된 본문 범위에서는 충분히 제시되지 않는다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 few-shot learning을 처음 접하는 독자에게 분야의 큰 흐름을 잡아 준다는 점이다. Data augmentation, metric learning, memory-based model, meta-optimization이라는 네 갈래의 관점을 제시하고, 각 관점에서 대표 모델이 왜 등장했는지 직관과 함께 설명한다. 특히 Siamese Networks, Matching Networks, NTM/MANN, Meta Networks, MAML, LSTM Meta-Learner까지 이어지는 계보를 한 문서 안에서 볼 수 있다는 점은 교육적 가치가 높다.

또 다른 강점은 모델의 작동 원리를 가능한 한 직관적으로 풀어 설명하려고 한다는 점이다. 예를 들어 Siamese Networks에서는 contrastive loss가 “비슷한 쌍은 가깝게, 다른 쌍은 margin 밖으로 밀어내는 역할”을 한다고 설명하고, NTM에서는 differentiable memory addressing을 단계별로 소개한다. MAML과 LSTM Meta-Learner도 “좋은 initialization 학습”과 “업데이트 규칙 자체 학습”이라는 시각에서 대비된다. 입문자 입장에서는 이런 설명이 단순 참고문헌 나열보다 훨씬 유용하다.

하지만 한계도 뚜렷하다. 첫째, 이 논문은 survey이므로 엄밀한 공정 비교 실험이나 새로운 방법론 검증을 제공하지 않는다. 따라서 어떤 기법이 실제로 어느 조건에서 더 우월한지 판단하려면 원 논문들을 직접 봐야 한다. 둘째, 일부 수식 표기와 설명은 다소 매끄럽지 않다. 제공된 본문에서도 contrastive loss, MANN 수식, 일부 기호 표기에 깨짐이나 생략이 있어, 수학적으로 엄밀한 독해를 위해서는 cited original paper를 다시 확인하는 것이 필요하다. 셋째, 실험 결과는 전반적으로 개괄적이며, 데이터셋별 정량 성능 비교나 재현 세부 설정은 충분하지 않다.

비판적으로 보면, 이 논문은 “few-shot learning이 왜 필요한가”와 “대표 모델들이 어떤 아이디어를 쓰는가”를 정리하는 데는 성공적이지만, 최근 few-shot literature에서 중요한 더 세밀한 쟁점들, 예를 들어 evaluation protocol의 민감성, backbone 차이, transductive setting 여부, pretraining 규모의 영향 등은 깊게 다루지 않는다. 물론 이는 논문의 목적이 광범위한 개관에 있기 때문에 어느 정도 자연스러운 한계다.

또한 결론부에서는 OpenAI, Google, Microsoft, Amazon 등 대형 기업의 투자와 인류적 파급효과를 언급하지만, 이는 비전 제시에는 도움이 되어도 survey 본문에서 앞서 정리한 기술적 논의를 직접적으로 강화하는 증거는 아니다. 따라서 이 부분은 다소 전망 중심의 서술로 읽는 것이 적절하다.

## 6. 결론

이 논문은 few-shot learning용 딥러닝 아키텍처를 체계적으로 정리한 개관 논문으로서, 적은 데이터 환경에서 어떻게 일반화 성능을 얻을 수 있는지에 대한 대표적 접근들을 소개한다. 핵심 기여는 새로운 알고리즘 제안이 아니라, **metrics-based 방법(Siamese, Matching Networks)**, **model-based 방법(NTM, MANN, Meta Networks)**, **optimization-based 방법(MAML, LSTM Meta-Learner)** 을 하나의 큰 틀 안에서 연결해 설명한 데 있다.

논문이 전달하는 메시지는 분명하다. 성공적인 few-shot learning을 위해서는 단순히 데이터가 적다는 문제를 우회하는 것이 아니라, 좋은 representation, 빠른 메모리 기반 적응, 그리고 gradient descent를 넘어서는 효율적인 optimization 전략이 함께 필요하다. 특히 classification에서는 상당한 진전이 있었지만, object detection이나 segmentation처럼 더 복잡한 문제는 여전히 어려운 과제로 남아 있다.

실제 적용 측면에서 이 연구 흐름은 데이터 수집 비용이 높거나 희귀 사례가 중요한 분야에서 의미가 크다. 의료 영상, 서명 검증, 제조 불량 탐지처럼 라벨이 적은 상황에서 특히 가치가 있다. 향후 연구에서는 더 복잡한 시각 태스크로의 확장, 도메인 차이 대응, 안정적인 메타학습 프로토콜, 그리고 보다 강건한 메모리/최적화 설계가 중요해질 가능성이 높다.

제공된 본문만 기준으로 보면, 이 논문은 few-shot learning의 세부 최신 성능 경쟁을 정밀 분석하는 문서라기보다는, 분야의 큰 설계 원리와 대표 모델들을 이해하기 위한 출발점으로 가장 적합하다.
