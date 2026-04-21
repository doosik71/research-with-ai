# Meta-Learning Approaches for Few-Shot Learning: A Survey of Recent Advances

- **저자**: Hassan Gharoun, Fereshteh Momenifar, Fang Chen, Amir H. Gandomi
- **발표연도**: 2023
- **arXiv**: https://arxiv.org/abs/2303.07502

## 1. 논문 개요

이 논문은 few-shot learning을 위한 **meta-learning** 연구들을 정리한 survey 논문이다. 저자들은 기존 deep learning이 대규모 학습 데이터가 충분할 때는 강력하지만, 새로운 task나 distribution shift가 있는 상황에서는 일반화 성능이 크게 떨어진다는 점을 문제로 제시한다. 특히 학습 샘플이 매우 적은 상황에서는 모델이 처음부터 다시 학습되거나, pre-training 후 fine-tuning을 하더라도 성능이 쉽게 무너진다는 점이 핵심 배경이다.

이 논문이 다루는 연구 문제는 명확하다. 어떻게 하면 모델이 과거 여러 task에서 얻은 경험을 이용해, **새로운 unseen task를 적은 샘플만으로 빠르게 적응**할 수 있는가 하는 것이다. 저자들은 이를 “learning to learn”의 관점에서 설명한다. 즉, 일반적인 supervised learning이 개별 데이터 샘플 중심의 학습이라면, meta-learning은 **task들의 집합 자체로부터 학습하는 프레임워크**라는 것이다.

문제의 중요성도 분명하다. 실제 응용에서는 대규모 라벨 데이터를 확보하기 어려운 경우가 많고, 학습과 테스트 분포가 동일하다는 가정도 자주 깨진다. 따라서 few-shot learning, domain adaptation, cross-domain generalization 같은 현실 문제에서 meta-learning은 중요한 대안이 된다. 이 논문은 이러한 맥락에서 최근 대표 방법들을 체계적으로 분류하고, 벤치마크 성능과 함께 장단점 및 향후 과제를 정리한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 meta-learning 방법들을 단순히 나열하는 것이 아니라, **알고리즘의 작동 메커니즘(mechanics)** 관점에서 재구성하는 데 있다. 저자들은 방법들을 크게 세 부류로 나눈다. 첫째는 support와 query 사이의 유사도를 학습하는 **metric-based methods**, 둘째는 과거 경험을 memory에 저장하고 불러오는 **memory-based methods**, 셋째는 initialization, parameter, optimizer 자체를 학습하는 **learning-based methods**이다.

특히 이 survey의 차별점은 기존 survey들이 model-based 혹은 optimization-based라는 다소 넓은 분류를 사용했던 것과 달리, learning-based methods 내부를 다시 **learning the initialization**, **learning the parameters**, **learning the optimizer**로 세분화한 점이다. 저자들은 이 세 부류가 실제로는 학습 대상이 서로 다르므로 기계적으로도 구분되어야 한다고 본다.

또 하나 중요한 점은, 이 논문은 개별 방법 하나를 새로 제안하는 논문이 아니라는 것이다. 따라서 “핵심 아이디어”는 새로운 알고리즘 제안이 아니라, 최근 meta-learning 연구의 흐름을 조직적으로 설명하고 비교하는 데 있다. 저자들은 특히 metric-based 방법에서 intra-class / inter-class 관계, weighted prototype, learnable similarity의 중요성이 커지고 있다고 해석한다.

## 3. 상세 방법 설명

이 논문은 먼저 meta-learning의 기본 학습 구조를 설명한다. 일반 supervised learning에서는 데이터셋 $D$를 $D_{train}$과 $D_{test}$로 나누고 epoch 단위로 학습한다. 반면 meta-learning에서는 데이터셋을 먼저 $D_{meta\_train}$과 $D_{meta\_test}$로 나눈 뒤, 각 meta set 안에서 다시 **support set**과 **query set**으로 나눈다. 각 episode는 보통 $N$-way $k$-shot 문제 형태로 구성되며, support set으로 task를 학습하고 query set에서 예측 오차를 계산해 meta-learner를 업데이트한다. 이렇게 여러 episode를 거치며 “적은 데이터로 학습하는 법” 자체를 익히게 된다.

### Metric-based methods

metric-based 방법의 기본 구조는 support sample과 query sample을 embedding 공간으로 보낸 뒤, 유사도 함수 $s(\cdot,\cdot)$로 거리를 측정해 query label을 정하는 것이다. 즉, embedding function $f$와 $g$가 각각 support와 query를 latent vector로 바꾸고, 가장 가까운 support class 혹은 prototype에 따라 분류한다.

**Siamese Network**는 가장 초기의 형태로, 두 입력 $x_1, x_2$를 동일한 파라미터를 공유하는 두 네트워크에 통과시킨 뒤 energy function으로 유사도를 측정한다. 원래 목적은 다중 분류가 아니라 pair matching이다. 이때 contrastive loss를 사용한다. 논문은 다음 형태를 소개한다.

$$
L(E,Y)=\sum_{i=1}^{P}(1-Y)\,L_G(E(Z_1,Z_2)_i)+Y\,L_I(E(Z_1,Z_2)_i)
$$

여기서 $Y=0$이면 similar pair, $Y=1$이면 dissimilar pair이다. 후속 연구에서는 margin을 둔 형태도 사용한다.

$$
L(E,Y)=(1-Y)E^2 + Y \max(0, margin-E)^2
$$

즉, 비슷한 샘플은 embedding distance를 줄이고, 다른 샘플은 최소 margin 이상 떨어지도록 학습한다.

**Prototypical Networks**는 각 class의 support embedding 평균을 prototype으로 정의한다. query embedding은 각 prototype과의 Euclidean distance를 계산해 가장 가까운 class로 분류된다. 학습은 정답 클래스의 negative log-probability를 최소화하는 방식이다.

$$
-\frac{1}{T}\sum_i \log p(y_i^* \mid x_i^*, p_c)
$$

이 방법의 핵심은 “class를 대표하는 중심점 하나만 잘 만들면 few-shot 분류가 가능하다”는 점이다. survey는 이 계열의 여러 확장도 상세히 소개한다. semi-supervised PN은 unlabeled data를 soft k-means 방식으로 prototype refinement에 활용한다. Gaussian PN은 embedding과 함께 covariance matrix를 만들어 각 샘플의 불확실성과 가중치를 반영한다. query와 prototype 사이의 거리는 다음처럼 계산된다.

$$
d_c^2(i) = (\tilde{x}_i - \tilde{p}_c)^T S_c (\tilde{x}_i - \tilde{p}_c)
$$

여기서 $S_c$는 inverse covariance matrix이다. prototype은 variance-weighted average로 계산된다.

$$
\tilde{p}_c = \frac{\sum_i \tilde{s}_{ic}\cdot \tilde{x}_{ic}}{\sum_i \tilde{s}_{ic}}
$$

또한 class covariance는

$$
\tilde{S}_c = \sum_i \tilde{s}_{ic}
$$

로 정의된다. 이 외에도 domain alignment, weighted prototype, multi-prototype, multi-modal prototype 같은 확장들이 소개된다. 전반적으로 prototype 하나만 쓰는 단순 구조에서, intra-class variation과 domain shift, multi-label, local descriptor 문제를 해결하려는 흐름이 강조된다.

**Matching Networks**는 support set 전체를 메모리처럼 사용한다. query $\hat{x}$에 대한 예측은 support sample 각각의 label을 attention weight로 가중합해서 만든다.

$$
p(\hat{y}\mid \hat{x}, S)=\sum_{i=1}^{k} a(\hat{x},x_i) y_i
$$

attention은 embedding 사이 cosine similarity에 softmax를 취해 구한다.

$$
a(\hat{x},x_i)=\frac{e^{cosine(f(\hat{x}),g(x_i))}}{\sum_{j=1}^{k} e^{cosine(f(\hat{x}),g(x_j))}}
$$

즉, query와 가장 비슷한 support sample들이 더 큰 영향력을 갖는다. 이 계열에서는 feature-level attention, hard label-set sampling, cascaded matching 같은 개선 방향이 논의된다.

**Relation Networks**는 similarity를 고정된 Euclidean distance나 cosine으로 계산하지 않고, neural network 자체가 relation score를 배우게 만든다. support와 query embedding을 relation module에 넣어 0에서 1 사이의 relation score를 출력하게 하며, 저자들은 이를 위해 MSE loss를 사용했다고 정리한다. 후속 연구들은 prototype-relation 결합, multi-scale feature, local feature attention, spatial correlation을 도입해 relation score의 표현력을 높인다.

이 외에도 survey는 **Graph Neural Network**, **Global Class Representation**, **Attentive Recurrent Comparators**, **Region Comparison Network**, **Metric-Agnostic Conditional Embeddings** 등을 소개한다. 공통적으로는 “어떤 similarity를 어떻게 학습할 것인가”가 핵심이며, 고정 metric 대신 learnable relation을 사용하려는 흐름이 강하다.

### Memory-based methods

memory-based 방법은 내부 혹은 외부 memory를 두고 과거 task 경험을 저장해 새로운 task 적응에 활용한다.

대표적으로 **MANN (Memory-Augmented Neural Networks)**는 Neural Turing Machine 기반이며, controller와 memory bank로 구성된다. 모델은 differentiable read/write 연산으로 memory에 접근한다. read는 memory에서 정보를 꺼내고, write는 erase와 add 연산으로 메모리를 갱신한다. 저자들 설명에 따르면 MANN은 representation-class label의 결합 정보를 외부 memory에 축적해 나중 classification에 사용한다.

**SNAIL**은 external memory 대신 temporal convolution과 soft attention을 결합해 과거 정보를 빠르게 참조한다. temporal convolution은 넓은 범위의 과거 정보를, soft attention은 특정 relevant experience를 선택하는 역할을 한다.

**Conditional Neural Processes (CNPs)**는 support set의 정보를 embedding $h$로 바꾸고, aggregation operator $a$로 하나의 요약 representation으로 합친 뒤, task learner $g$가 이를 바탕으로 query 예측을 수행한다. 즉, support set 전체를 하나의 압축된 task representation으로 바꾸는 방식이다.

### Learning-based methods

이 논문에서 learning-based methods는 세 하위 부류로 나뉜다.

첫째, **learning the initialization**은 좋은 초기 파라미터를 학습하는 방법이다. 대표적인 **MAML**은 task $T_i$마다 inner loop에서 task-specific parameter를 업데이트하고,

$$
\theta_i' = \theta - \alpha \nabla_\theta L_{T_i}(f_\theta)
$$

outer loop에서 여러 task의 성능이 좋아지도록 초기 파라미터 $\theta$를 다시 업데이트한다.

$$
\theta = \theta - \beta \nabla \sum_{T_i \sim p(T)} L_{T_i}(f_\theta)
$$

핵심은 “새 task에서도 gradient step 몇 번만으로 빨리 적응할 수 있는 initialization”을 찾는 것이다. survey는 여기에 uncertainty를 추가한 PLATIPUS, BMAML, latent low-dimensional space에서 최적화하는 LEO, adversarial robustness를 다루는 ADML, task bias를 줄이려는 TAML, exact meta-gradient를 다루는 iMAML 등을 소개한다.

둘째, **learning the parameters**는 meta-learner가 base learner의 파라미터 자체를 생성하는 접근이다. 예를 들어 **MetaNet**은 meta information을 입력받아 빠르게 가중치를 생성하고, **LGM-Net**은 task context encoder와 weight generator를 통해 matching network용 파라미터를 만든다. **Weight imprinting**은 novel class sample의 normalized embedding을 classifier 마지막 층의 weight로 직접 사용한다. 이 계열은 fast adaptation뿐 아니라 base class를 잊지 않는 문제도 함께 다룬다.

셋째, **learning the optimizer**는 optimizer 자체를 학습하는 접근이다. 논문은 Ravi와 Larochelle의 LSTM-based meta-learner를 대표 예로 든다. 표준 gradient descent는

$$
\theta_t = \theta_{t-1} - \alpha_t \nabla_{\theta_{t-1}} L_{T_j}(\theta_{t-1})
$$

인데, 이를 LSTM 기반 업데이트 함수 $g_\phi$로 대체한다.

$$
\theta_{t+1} = \theta_t + g_\phi(\nabla_{\theta_t} L_{T_j}(\theta_t))
$$

또한 **Meta-SGD**는 initialization뿐 아니라 element-wise learning rate vector $\alpha$도 함께 학습한다. **Reptile**은 MAML과 유사하지만 더 단순하게, 각 task의 최적점 $\theta_i'$에 가까워지도록 초기 파라미터를 이동시킨다.

$$
\min \mathbb{E}\left[\frac{1}{2}D(\theta,\theta_i')\right]
$$

업데이트는

$$
\theta = \theta - \epsilon(\theta_i' - \theta)
$$

형태로 이루어진다. 이 밖에 R2-D2, LR-D2, MetaOptNet처럼 differentiable closed-form solver나 convex optimization을 base learner로 쓰는 방법도 포함된다. 예를 들어 R2-D2는 ridge regression 해를 미분 가능하게 사용한다.

$$
\Lambda(z)=\arg\min \|XW-Y\|^2 + \lambda \|W\|^2 = (X^T X+\lambda I)^{-1}X^T Y
$$

고차원 계산 비용을 줄이기 위해 Woodbury identity를 사용한 변형식도 제시된다.

## 4. 실험 및 결과

이 논문은 survey이므로 새로운 실험을 직접 수행하지는 않고, 각 원 논문에 보고된 benchmark 결과를 모아 비교한다. 주요 데이터셋은 **Omniglot**, **miniImageNet**, **CUB-200-2011**이다. task 설정은 주로 5-way 1-shot, 5-way 5-shot, 때로는 20-way도 포함된다. 평가지표는 기본적으로 classification accuracy이며, 일부 결과는 95% confidence interval도 함께 보고된다.

저자들이 정리한 결과에 따르면, Omniglot처럼 비교적 단순하고 task 간 관련성이 높은 데이터셋에서는 여러 방법이 매우 높은 정확도를 보인다. 예를 들어 5-way 5-shot Omniglot에서 **Meta-SGD 99.91%**, **Prototype-relation networks 99.91%**, **MAML 99.90%**, **GCR 99.90% 수준**으로 거의 포화에 가까운 성능을 낸다. 저자들은 이를 근거로, task가 서로 유사한 경우에는 방법 간 기계적 차이에도 불구하고 전반적으로 높은 성능을 쉽게 달성한다고 해석한다.

반면 miniImageNet은 훨씬 어렵다. 저자들의 정리에 따르면 5-way 5-shot miniImageNet에서 **MLFRNet 83.16%**, **LMPNet 80.23%**, **MetaOptNet 80.00%**, **LEO 77.59%**, **MAML+L2F 78.13%** 등이 강한 성능을 보인다. 특히 metric-based 계열 중에서도 local feature, multiple prototype, relation refinement를 도입한 모델들이 매우 경쟁력 있다는 점이 강조된다. 이는 단순한 global prototype보다 **local descriptor**, **intra/inter-class structure**, **attention**이 어려운 few-shot recognition에서 중요하다는 해석으로 이어진다.

CUB-200-2011에서는 fine-grained classification 특성상 local region 정보를 다루는 방법들이 상대적으로 좋은 성능을 보인다. 예를 들어 **RCN**은 1-shot에서 74.65%, 5-shot에서 88.81%를 보고했고, **multi-modal prototypical network**도 75.01%, 85.30%를 기록했다. 텍스트 설명까지 활용하는 multimodal 접근이 세밀한 클래스 구분에 도움을 준다는 점도 드러난다.

다만 이 결과 비교에는 survey 논문 특유의 한계가 있다. 모든 방법이 동일한 backbone, 동일한 split, 동일한 implementation protocol 위에서 재현 비교된 것은 아니다. 따라서 표의 숫자는 “원 논문 보고값들의 집합”으로 해석해야 하며, 완전한 공정 비교라고 보기는 어렵다. 이 점은 논문 본문에서도 직접 크게 문제화하지는 않지만, 독자가 주의해야 할 부분이다.

## 5. 강점, 한계

이 논문의 강점은 우선 구조화가 명확하다는 점이다. meta-learning을 task-level learning이라는 관점에서 다시 설명하고, episodic training, support/query, meta-train/meta-test 구조를 초심자도 따라갈 수 있도록 정리했다. 또한 단순히 대표 모델만 언급하는 것이 아니라, Prototypical Networks, Matching Networks, Relation Networks, MAML 계열 등에서 나온 다양한 변형들을 폭넓게 다룬다. 실제로 benchmark 표를 통해 Omniglot, miniImageNet, CUB-200-2011의 성능까지 함께 비교한 점은 실용적이다.

또 다른 강점은 연구 흐름을 방법론적 관점에서 읽게 해 준다는 점이다. 예를 들어 metric-based methods에서는 weighted prototype, local descriptor, domain alignment, learnable metric 같은 방향으로 발전하고 있고, learning-based methods에서는 initialization, parameter generation, optimizer learning이라는 별도 축이 있다는 점이 잘 드러난다. 따라서 이 논문은 입문 survey이면서도 연구 지형도를 파악하기에 유용하다.

한계도 분명하다. 첫째, 이 논문은 survey이므로 어떤 방법의 수학적 엄밀성이나 실험적 공정성을 깊게 검증하는 논문은 아니다. 각 원 논문의 주장을 비교적 우호적으로 정리하며, 재현성 문제나 benchmark 설정 차이에 대한 비판적 검토는 제한적이다. 둘째, 일부 설명은 개념 정리 수준에 머물고, 방법 간 이론적 연결이나 실패 사례 분석은 깊지 않다. 셋째, 표의 숫자 비교는 유용하지만, 서로 다른 backbone과 학습 설정 차이를 충분히 통제하지 못한다.

논문 자체가 제시하는 미해결 질문도 중요하다. 저자들은 meta-learner의 **uncertainty**, episodic training에서의 **catastrophic forgetting**, adversarial robustness, multi-domain / cross-domain / multi-modal generalization, backbone architecture의 영향, computational cost 문제를 핵심 과제로 제시한다. 특히 “현재 few-shot 성능이 진짜 few-shot adaptation인지, 아니면 기존 feature 재사용 효과인지”는 매우 중요한 비판적 질문이다. 이는 이 논문이 단순 정리에서 끝나지 않고 향후 연구의 쟁점을 어느 정도 정확히 짚고 있음을 보여준다.

## 6. 결론

이 논문은 few-shot learning을 위한 meta-learning 연구를 **metric-based**, **memory-based**, **learning-based**의 세 큰 축으로 정리하고, 대표 방법들과 확장 모델들을 폭넓게 소개한 survey다. 핵심 기여는 새로운 알고리즘 제안이 아니라, 빠르게 확장되는 meta-learning 문헌을 메커니즘 중심으로 재조직하고, 주요 benchmark 결과와 함께 현재의 성과와 병목을 정리한 데 있다.

실제 적용 측면에서 이 연구는 적은 데이터로 빠른 적응이 필요한 컴퓨터 비전, 자연어처리, 의료, 원격탐사 같은 분야에 중요한 참고 틀을 제공한다. 향후 연구에서는 단순 accuracy 향상뿐 아니라 uncertainty calibration, cross-domain robustness, multimodal adaptation, 계산 효율성, 공정한 benchmark 설계가 더욱 중요해질 가능성이 크다. 이 논문은 그런 후속 연구를 위한 출발점으로는 충분히 유용하지만, 개별 방법의 엄밀한 성능 비교나 이론 분석까지 기대하기보다는 **최근 meta-learning 지형을 정리한 폭넓은 survey**로 읽는 것이 가장 적절하다.
