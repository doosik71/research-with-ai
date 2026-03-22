# Meta-Learning Representations for Continual Learning

* **저자**: Khurram Javed, Martha White
* **발표연도**: 2019
* **arXiv**: [https://arxiv.org/abs/1905.12588](https://arxiv.org/abs/1905.12588)

## 1. 논문 개요

이 논문은 continual learning에서 가장 중요한 두 목표, 즉 **새로운 데이터를 빠르게 배우는 능력**과 **기존 지식을 잊지 않는 능력**을 동시에 높이기 위한 representation learning 방법을 제안한다. 저자들은 기존의 신경망이 catastrophic forgetting에 매우 취약하고, 새로운 데이터를 빠르게 흡수하도록 훈련되지도 않는 핵심 이유 중 하나가, **표현(representation) 자체가 그런 목적을 직접 반영해서 학습되지 않기 때문**이라고 본다.

논문이 다루는 연구 문제는 다음과 같다. 입력과 정답이 시간 순서대로 들어오는 데이터 스트림에서, 모델은 오직 한 번 지나가는 온라인 업데이트만으로 계속 학습해야 한다. 이때 일반적인 end-to-end 신경망 학습은 두 가지 한계를 보인다. 첫째, sample efficiency가 낮아서 적은 데이터와 단일 패스 학습으로는 잘 적응하지 못한다. 둘째, 상관된 순서로 들어오는 데이터에 대해 이전에 학습한 내용을 쉽게 덮어써 버린다. 저자들은 이 문제를 단순히 optimizer나 replay buffer의 문제로만 보지 않고, **어떤 표현 공간 위에서 prediction head를 학습하느냐**의 문제로 재정의한다.

이 문제가 중요한 이유는 continual learning이 단순한 벤치마크 문제가 아니라, 실제 지능형 시스템이 환경과 계속 상호작용하면서 학습해야 하는 거의 모든 상황의 기본 조건이기 때문이다. 로봇 위치 예측, 순차적 분류, 강화학습의 value prediction 등 다양한 문제를 동일한 틀로 볼 수 있다. 따라서 interference를 줄이면서 future learning을 가속하는 representation을 학습할 수 있다면, 많은 online learning 시스템의 기반을 바꿀 수 있다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 **“계속 학습에 유리한 representation을 meta-learning으로 직접 학습하자”** 는 것이다. 저자들은 단순히 sparse representation을 강제하거나, weight regularization으로 중요한 파라미터를 보호하거나, 과거 샘플을 replay하는 방식만으로는 충분하지 않다고 본다. 대신, 실제 온라인 업데이트가 일어났을 때 그 업데이트가 미래 성능에 어떤 영향을 주는지를 메타 수준에서 최적화해야 한다고 주장한다.

이를 위해 제안한 것이 **OML (Online-aware Meta-Learning)**이다. OML은 representation learning network(RLN)와 prediction learning network(PLN)를 분리하고, inner loop에서는 **PLN만 온라인 SGD로 업데이트**한다. 그런 다음, 이렇게 실제 온라인 학습을 흉내 낸 뒤의 성능이 좋아지도록 RLN의 파라미터를 meta-update한다. 즉, representation은 “한 번의 SGD 업데이트가 최대한 덜 망가뜨리고, 동시에 새로운 정보는 더 빨리 받아들이도록” 학습된다.

기존 gradient-based meta-learning, 특히 MAML과의 차별점은 분명하다. 일반적인 MAML류는 빠른 적응을 위해 **초기화(initialization)**를 학습하는 데 집중한다. 하지만 이 논문은 단순 초기화만으로는 catastrophic interference를 다루기 어렵다고 실험적으로 주장한다. OML은 초기화보다 한 단계 더 들어가, **입력을 어떤 feature space로 바꾸어 놓으면 이후의 온라인 업데이트가 서로 덜 충돌하는지**를 학습한다. 논문은 이를 직관적으로 설명하기 위해, 서로 다른 데이터 분포에 대한 해 공간(solution manifold)이 representation에 따라 평행하거나 직교에 가까워질 수 있으며, 이런 구조가 online update의 positive generalization 혹은 non-interference를 만든다고 해석한다.

흥미로운 점은 OML이 sparsity를 직접 규제하지 않는데도, 결과적으로 **자연스럽고 instance-sparse한 표현**을 학습한다는 것이다. 즉, 저자들의 메시지는 “sparsity를 목표로 삼은 것이 아니라, interference를 줄이는 representation을 직접 최적화했더니 sparsity가 emergent property로 나타났다”에 가깝다.

## 3. 상세 방법 설명

논문은 continual learning 문제를 Continual Learning Prediction (CLP) 문제로 공식화한다. 데이터는 끝없이 이어지는 샘플 스트림

$$
\mathcal{T} = (X_1, Y_1), (X_2, Y_2), \ldots, (X_t, Y_t), \ldots
$$

로 주어지며, 입력 $X_t$는 서로 상관된 시계열적 구조를 가질 수 있지만, 타깃 $Y_t$는 현재 입력 $X_t$에만 의존한다고 가정한다. 전체 목표는 predictor $f_{W,\theta}$가 장기적으로 작은 예측 오차를 갖도록 하는 것이다. 논문은 이를 다음 기대 손실로 정의한다.

$$
\mathcal{L}_{CLP}(W,\theta)
\mathrel{\overset{\tiny def}{=}}
\mathbb{E}[\ell(f*{W,\theta}(X),Y)] =
\int\left[\int \ell(f_{W,\theta}(x),y)p(y|x)dy\right]\mu(x)dx
$$

여기서 $\mu(x)$는 각 입력이 얼마나 자주 관측되는지를 나타내는 입력 분포이고, $\ell(\hat y, y)$는 예측 오차이다. 중요한 점은 이 기대 손실 자체보다, 실제 학습이 iid 샘플링이 아니라 **길이 $k$의 단일 trajectory**에서 이루어진다는 데 있다. 즉, 모델은 상관된 순서의 데이터 한 번 보기만으로 잘 배워야 한다.

### 모델 구조

저자들은 모델을 두 부분으로 나눈다.

첫째는 representation learning network인 $\phi_\theta(X)$ 이다. 이것은 입력 $X$를 $d$차원 feature space로 보내는 encoder 역할을 한다.

둘째는 prediction learning network인 $g_W$ 이다. 이것은 feature를 받아 최종 예측을 수행한다.

전체 모델은 다음처럼 합성된다.

$$
f_{W,\theta}(X) = g_W(\phi_\theta(X))
$$

여기서 핵심 설계는 **$\theta$와 $W$의 역할을 분리**한 것이다. $\theta$는 meta-parameter로서 meta-training 동안 학습된 뒤 meta-test에서는 고정된다. 반면 $W$는 실제 continual learning 동안 온라인 SGD로 계속 업데이트된다. 다시 말해, representation은 “학습하기 좋은 기반”으로 고정해 두고, prediction head만 빠르게 적응시키는 구조이다.

### OML의 목적 함수

OML은 여러 CLP 문제 $\mathcal{T}_i \sim p(\mathcal{T})$ 위에서 학습된다. 각 문제마다 trajectory $\mathcal{S}_k^j$를 샘플링하고, 그 trajectory를 따라 online SGD를 수행한 뒤, 그 결과 파라미터가 CLP 손실을 얼마나 잘 줄였는지를 평가한다. 논문에 제시된 메타 목적은 다음과 같다.

$$
\min_{W,\theta}\sum_{\mathcal{T}_i \sim p(\mathcal{T})}\text{OML}(W,\theta)
\mathrel{\overset{\tiny def}{=}}
\sum*{\mathcal{T}_i \sim p(\mathcal{T})}
\sum*{\mathcal{S}_k^j \sim p(\mathcal{S}_k|\mathcal{T}_i)}
\left[
\mathcal{L}_{CLP_i}\big(U(W,\theta,\mathcal{S}_k^j)\big)
\right]
$$

여기서 $U(W,\theta,\mathcal{S}_k^j)$ 는 trajectory를 따라 $k$번 SGD를 적용한 뒤의 파라미터를 의미한다. 중요한 것은 이 inner update 동안 **$\theta$는 고정되고 $W$만 갱신된다는 점**이다. 따라서 representation은 직접 자주 흔들리지 않고, “이 representation 위에서의 online update가 미래 성능을 잘 유지하는가”를 기준으로 간접적으로 최적화된다.

### OML의 학습 절차

논문 알고리즘 2의 흐름을 쉬운 말로 정리하면 다음과 같다.

먼저 하나의 CLP 문제를 샘플링하고, 그 문제에서 순차적 학습에 해당하는 training trajectory $\mathcal{S}_{train}$ 을 뽑는다. 초기 prediction head 파라미터 $W_0$ 에서 시작해, trajectory의 각 샘플 $(X_j, Y_j)$ 를 하나씩 순서대로 보면서

$$
W_j = W_{j-1} - \alpha \nabla_{W_{j-1}} \ell_i(f_{\theta, W_{j-1}}(X_j), Y_j)
$$

로 업데이트한다. 이 과정은 실제 online learning과 거의 동일하다. 이후 별도의 test trajectory $\mathcal{S}_{test}$ 를 뽑아, 최종 $W_k$ 를 사용한 예측 손실을 계산하고, 그 손실을 줄이는 방향으로 representation 파라미터 $\theta$ 를 meta-update한다.

즉, OML은 단순히 “몇 샷 후 정확도가 높은 표현”이 아니라, “순서 의존적 온라인 업데이트를 여러 번 겪은 후에도 성능이 유지되는 표현”을 학습한다.

### MAML-Rep와의 차이

논문은 비교 대상으로 MAML-Rep도 제안한다. MAML-Rep 역시 RLN과 PLN을 분리하지만, inner loop에서 전체 batch를 사용해 몇 번의 적응을 수행하는 few-shot 스타일 목적을 따른다. 반면 OML은 **trajectory의 샘플을 하나씩 순서대로 사용**하며, 온라인 학습에서 생기는 interference를 inner loop 안에 직접 포함시킨다. 저자들이 강조하는 차이는 바로 이것이다. MAML-Rep는 빠른 적응은 학습하지만, OML처럼 catastrophic forgetting까지 직접 훈련 신호로 쓰지는 않는다.

### OML이 유도하는 표현의 직관

논문은 representation이 solution manifold의 구조를 바꾼다고 설명한다. 서로 다른 세 데이터 분포 $p_1(Y|x), p_2(Y|x), p_3(Y|x)$ 가 있다고 하자. 같은 선형 predictor라도 representation에 따라 각 분포에 대해 좋은 성능을 내는 parameter 집합의 위치와 방향이 달라진다. 어떤 representation에서는 이 manifold들이 서로 멀리 꼬여 있어 한 분포에 맞춘 업데이트가 다른 분포의 성능을 크게 해칠 수 있다. 반대로 manifold가 평행하면 한쪽 학습이 다른 쪽에도 도움이 되고, 직교하면 적어도 서로 덜 간섭한다. OML은 바로 이런 “업데이트 간 간섭이 적은 표현 공간”을 찾도록 유도하는 것으로 해석할 수 있다.

### 실제 구현상의 근사

Split-Omniglot에서는 trajectory 길이가 $k=1000$ 이라 OML 목적을 정확히 unroll하면 계산량이 매우 크다. 논문은 이를 해결하기 위해 truncated backpropagation과 유사한 근사를 사용한다. 구체적으로, inner loop를 5 step씩 끊어서 처리하고, 각 구간 뒤에 누적된 meta-loss를 이용해 gradient를 계산한다. 논문은 이 근사 구현이 interference 효과를 여전히 반영하면서도 그래프를 1000 step 전체로 유지할 필요가 없게 해 준다고 설명한다.

## 4. 실험 및 결과

논문은 두 가지 मुख्य 벤치마크에서 representation 품질을 검증한다. 하나는 synthetic regression 문제인 **Incremental Sine Waves**, 다른 하나는 실제 이미지 기반 순차 분류 문제인 **Split-Omniglot** 이다. 추가로 mini-ImageNet에도 확장 실험을 수행했다고 보고한다.

### Incremental Sine Waves

이 문제는 10개의 서로 다른 sine function을 순차적으로 학습하는 회귀 문제이다. 입력은 $x=(z,n)$ 으로, $z \in [-5,5]$ 는 연속값 입력이고, $n$ 은 현재 어떤 sine function인지 나타내는 one-hot task identifier이다. 타깃은 결정적이며 $y=\sin_n(z)$ 이다. 각 trajectory는 첫 번째 함수에서 40 mini-batch, 두 번째 함수에서 40 mini-batch, 이런 식으로 총 10개 함수를 순차적으로 본다. mini-batch 크기는 8이다.

이 설정은 task id가 주어지더라도 representation과 update 방식이 나쁘면 함수 간 interference가 크게 발생할 수 있게 만들어져 있다. 온라인 업데이트는 MSE loss에 대한 SGD로 수행하며, OML 메타 업데이트는 Adam으로 학습한다. 이 실험에서 RLN은 6층, PLN은 2층이며 각 hidden layer 폭은 300이다. RLN은 meta-update로만 학습되고, 평가 시에는 고정된 feature extractor로 사용된다.

결과적으로 OML은 10개 함수를 순차적으로 배워도 평균 MSE 증가가 매우 작았다. 반면 Pre-training은 뒤로 갈수록 평균 오차가 커져 이전 함수들을 잊는 현상이 분명히 드러났다. SR-NN도 sparse representation을 학습하지만 OML보다 더 큰 interference를 보였다. 논문은 함수별 오차 분포도 제시하는데, Pre-training과 SR-NN은 초기에 배운 함수들에서 큰 오차를 보이는 반면, OML은 앞쪽 함수들에서 약간의 성능 저하만 있을 뿐 전반적으로 훨씬 안정적이었다. Oracle은 iid sampling과 multiple epoch를 허용한 상한선 역할로 제시되며, OML이 이 이상적인 설정에는 못 미치더라도 단일 패스 온라인 학습 기준으로 매우 강력함을 보여준다.

### Split-Omniglot

Omniglot은 1623개 문자 클래스로 이루어진 데이터셋이다. 논문은 앞 963개 클래스를 meta-training, 나머지 660개 클래스를 meta-testing에 사용한다. 하나의 CLP 문제는 200개 클래스를 순서 있게 뽑아 구성되며, 각 클래스에서 이미지 5장씩을 순서대로 봐서 총 길이 1000의 trajectory를 만든다. 즉, 모델은 클래스 1의 5장, 클래스 2의 5장, …, 클래스 200의 5장을 차례대로 보게 된다.

여기서 RLN은 6개의 convolution layer와 2개의 fully connected layer로 이루어진 CNN encoder이다. convolution 파라미터를 $\theta$, FC layer 파라미터를 $W$ 로 둔다. MAML-Rep와 OML 모두 fair comparison을 위해 inner gradient step 수와 유사한 아키텍처를 맞춘다. OML은 1000 step 전체 unroll이 비싸므로, 앞서 설명한 truncated approximation을 사용한다.

결과는 분명하다. OML은 training trajectory 상의 정확도를 거의 완벽하게 유지하며, forgetting이 매우 작다. 논문은 continual learning에서 training accuracy도 중요한데, 이는 과거에 본 샘플을 얼마나 잊지 않았는지를 보여주기 때문이라고 설명한다. test accuracy 역시 OML이 더 좋았고, 이는 단지 암기만이 아니라 일반화에서도 이득이 있음을 뜻한다. 반면 baselines는 클래스 수가 늘어날수록 training accuracy가 떨어졌고, 이는 순차 학습 과정에서 forgetting이 누적된다는 의미이다.

논문은 추가 sanity check로 같은 representation 위에서 iid sampling으로 5 epoch 학습하는 실험도 수행한다. 여기서는 OML과 MAML-Rep의 성능이 비슷했다. 이 결과는 매우 중요하다. 두 방법 모두 representation의 “일반적 품질” 자체는 비슷할 수 있지만, correlated stream에서 online learning을 할 때는 OML representation이 훨씬 유리하다는 뜻이다. 즉, OML의 이점은 단순 feature quality가 아니라 **incremental learning suitability** 에 있다.

### Mini-ImageNet 확장

논문은 OML이 더 복잡한 데이터셋인 mini-ImageNet에도 적용 가능하다고 보고한다. 메타 테스트 시 20-way classifier를 각 클래스당 30개 샘플로 학습하는 설정을 사용했다고 기술하지만, 본문에서 수치 결과를 자세히 제공하지는 않고 Figure 5를 통해 확장성을 보여주는 수준으로 제시한다. 따라서 구체적 정량 비교는 주어진 텍스트만으로는 상세히 복원할 수 없다.

### 비교 기준선

논문은 세 가지 기본 baseline과 두 가지 메타러닝 기반 표현 학습 방법을 비교한다.

Scratch는 랜덤 초기화에서 온라인 학습을 시작한다.

Pre-training은 meta-training set에서 일반적인 gradient descent로 예측 오차를 줄이도록 학습한 뒤, 앞부분 레이어를 고정된 representation으로 사용한다.

SR-NN은 Set-KL 방법으로 sparse representation을 학습한다.

여기에 MAML-Rep와 OML을 더해 representation learning 방법들을 비교한다. 결과적으로 OML은 “그냥 좋은 표현”이 아니라 “순차적 온라인 업데이트에 강한 표현”을 학습한다는 점에서 가장 일관된 우위를 보인다.

### OML이 학습한 표현의 성질

논문은 OML representation이 실제로 어떤 성질을 갖는지도 분석한다. 저자들은 좋은 continual learning 표현은 **instance sparse** 하면서도 동시에 **dead neuron이 없어야 한다**고 본다. 즉, 하나의 입력을 표현할 때는 적은 수의 뉴런만 활성화되지만, 전체 데이터셋을 보면 모든 뉴런이 언젠가는 쓰여야 한다는 것이다. 이렇게 되면 한 번의 업데이트가 바꾸는 weight 수가 적어져 interference가 줄 수 있다.

Omniglot training set에서 측정한 결과는 다음과 같다.

OML의 instance sparsity는 **3.8%**, dead neuron 비율은 **0%** 이다.
SR-NN(Best)은 **15%**, dead neuron **0.7%** 이다.
SR-NN(Sparse)은 **4.9%**, dead neuron **14%** 이다.
Pre-Training은 **38%**, dead neuron **3%** 이다.

이 결과는 꽤 인상적이다. OML은 가장 희소한 수준의 활성화를 보이면서도 dead neuron이 전혀 없다. 반면 SR-NN은 비슷한 sparsity를 만들 수는 있지만 representation space의 큰 부분을 아예 쓰지 않게 되는 문제가 생긴다. Pre-training은 sparsity를 거의 만들지 못하고, 일부 dead neuron도 발생한다. 논문은 시각화를 통해 OML이 representation capacity를 넓게 활용하면서도 입력별로는 매우 sparse한 활성 패턴을 형성한다고 주장한다.

### 기존 continual learning 방법과의 결합

논문 5장에서는 OML representation이 다른 지식 보존 기법들과도 결합 가능한지 পরীক্ষা한다. 여기서는 EWC, MER, ER-Reservoir를 사용한다. 각 방법에 대해 세 가지 표현 조건을 비교한다.

첫째, Standard는 표현까지 포함한 전체 네트워크를 온라인으로 학습한다.
둘째, OML은 OML로 미리 학습한 representation을 고정하고 그 위에서 해당 방법을 적용한다.
셋째, Pre-training은 일반 pre-trained representation을 고정하고 그 위에서 학습한다.

Split-Omniglot 기준으로 결과는 매우 명확하다. 예를 들어 one class per task, 50 tasks 설정에서 Online 기본 SGD는 Standard가 **4.64 ± 2.61** 인 반면 OML representation 위에서는 **64.72 ± 2.57** 까지 올라간다. 같은 Online 방법이라도 Pre-training representation 위에서는 **21.16 ± 2.71** 에 그친다. MER 역시 Standard가 **54.88 ± 4.12** 인데 OML representation 위에서는 **76.00 ± 2.07** 로 향상된다. EWC와 ER-Reservoir도 같은 패턴을 보인다.

five classes per task, 20 tasks 설정에서도 마찬가지다. Online은 Standard **1.40 ± 0.43**, OML **55.32 ± 2.25**, Pre-training **11.80 ± 1.92** 이다. EWC는 Standard **2.04 ± 0.35**, OML **56.03 ± 3.20**, Pre-training **10.03 ± 1.53** 이다. MER과 ER-Reservoir도 OML representation을 쓸 때 크게 좋아진다.

이 결과에서 논문이 내리는 결론은 세 가지다. 첫째, OML은 거의 모든 기존 continual learning 알고리즘을 개선한다. 둘째, 단순히 fixed representation을 제공하는 것만으로는 충분하지 않고, **continual learning에 맞게 메타학습된 representation** 이어야 한다. 셋째, OML representation 위에서의 단순 online update만으로도 replay 기반 강한 방법들과 경쟁 가능하며, 어떤 경우에는 그보다 낫다. 특히 OML이 approximate IID보다도 좋은 결과를 보인다는 점은, interference 감소뿐 아니라 **새로운 데이터에 대한 빠른 적응 자체도 향상**되었음을 시사한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 continual learning의 핵심 병목을 **업데이트 규칙이 아니라 representation 문제로 다시 본 점**이다. 기존 연구는 주로 가중치 보호, replay, distillation 같은 관점에서 interference를 다뤘지만, 이 논문은 “표현 공간이 좋으면 같은 SGD도 훨씬 덜 파괴적일 수 있다”는 방향을 명확히 제시했다. 그리고 이를 OML이라는 구체적 meta-objective로 구현해, online update의 결과를 직접 최적화 대상으로 삼았다는 점이 설득력 있다.

두 번째 강점은 RLN과 PLN의 분리다. 논문은 단순 initialization learning보다 fixed encoder learning이 continual learning에 더 잘 맞는다고 보여준다. 이는 단지 성능 차이만이 아니라, 왜 초기 레이어를 계속 업데이트하면 interference가 커지는지에 대한 구조적 해석과도 연결된다. 특히 Appendix에서 “encoder를 학습하는 쪽이 optimization도 더 안정적”이라고 보고한 점도 실용적으로 의미가 있다.

세 번째 강점은 sparse representation이 **명시적 sparsity regularization 없이 emergent하게 나타난다**는 관찰이다. 이는 논문의 핵심 가설, 즉 interference를 직접 줄이도록 최적화하면 좋은 표현 구조가 자연스럽게 생긴다는 주장을 뒷받침한다. 게다가 OML은 sparse하면서도 dead neuron이 없는 representation을 만든다는 점에서, sparsity만 강제하는 기존 접근보다 질적으로 더 좋은 표현을 배운다고 볼 수 있다.

네 번째 강점은 기존 continual learning 기법들과의 상보성이다. OML은 EWC, MER, ER-Reservoir 같은 방법을 대체하는 것이라기보다, 그들이 더 잘 작동하도록 기반 표현을 제공한다. 이런 성질은 실제 시스템에서 매우 유리하다. 새로운 알고리즘으로 전체 파이프라인을 갈아엎는 대신, representation 층의 설계를 바꾸는 식으로 다양한 방법에 접목할 수 있기 때문이다.

반면 한계도 분명하다. 첫째, OML은 meta-training 단계가 필요하다. 논문 스스로도 결론에서 “별도의 메타학습 단계 없이 representation을 온라인으로 어떻게 지속적으로 적응시킬 것인가”가 다음 중요한 과제라고 밝힌다. 즉, 이 논문이 제시한 방법은 강력하지만 완전한 lifelong autonomous learner라기보다는, **오프라인 메타학습으로 온라인 학습에 좋은 초기 표현을 준비해 두는 접근**에 가깝다.

둘째, 계산 비용이 크다. Sine 실험에서는 400-step unroll, Omniglot에서는 1000-step에 가까운 순차 업데이트를 고려해야 하므로, 실제 구현에서는 truncated approximation이 필요했다. 이는 충분히 합리적인 공학적 선택이지만, 엄밀한 원래 목적 함수와 근사 구현 사이의 차이가 성능이나 안정성에 어떤 영향을 미치는지 텍스트에서는 깊게 분석하지 않는다.

셋째, 실험 범위가 representation의 일반성 전체를 증명한다고 보기는 어렵다. regression, Omniglot, mini-ImageNet까지는 보여 주지만, reinforcement learning이나 더 긴 기간의 비정상 환경, task descriptor가 없는 완전 task-free continual learning 등에서 어떻게 작동하는지는 본문만으로는 알 수 없다. 논문은 CLP formulation이 이런 문제들까지 포함할 수 있다고 설명하지만, 실제 실험 증거는 제한적이다.

넷째, OML의 우수성이 representation suitability 때문이라는 해석은 상당히 설득력 있지만, 그 내부 메커니즘이 완전히 해부되었다고 보기는 어렵다. 예를 들어 왜 특정 표현이 manifold를 더 평행하거나 직교하게 만드는지, 혹은 어떤 데이터 구조에서 OML이 특히 유리한지에 대한 이론적 분석은 제한적이다. 논문은 주로 직관과 실험적 증거에 기대고 있다.

종합하면, 이 논문은 매우 강한 실험적 메시지를 전달하지만, 온라인 representation adaptation의 완전한 해법이나 이론적 완결성을 제시한 것은 아니다. 그럼에도 불구하고 continual learning 연구 방향을 바꾸기에 충분히 의미 있는 기여를 한다.

## 6. 결론

이 논문은 continual learning에서 **representation을 직접 메타학습해야 한다**는 강한 주장을 실험적으로 뒷받침한다. 저자들은 OML이라는 meta-objective를 제안하여, representation learning network가 online update에 강하고 future learning을 촉진하도록 학습했다. 그 결과, correlated data stream에서 prediction head만 단순 SGD로 업데이트해도 forgetting이 크게 줄고, 새로운 클래스나 함수에 대한 적응도 빨라졌다. 또한 이런 표현은 별도의 sparsity penalty 없이도 highly sparse하고 dead neuron이 거의 없는 형태로 나타났다.

논문의 주요 기여는 세 가지로 요약할 수 있다. 첫째, continual learning에 적합한 representation을 학습하는 문제를 명확히 정식화하고, 이를 직접 최적화하는 OML을 제안했다. 둘째, OML representation이 synthetic regression과 sequential classification에서 기존 baselines보다 더 robust한 온라인 학습을 가능하게 함을 보였다. 셋째, OML이 EWC, MER, ER-Reservoir 같은 기존 방법과도 상보적으로 결합되어 성능을 크게 끌어올릴 수 있음을 보였다.

실제 적용 관점에서도 의미가 크다. 많은 시스템에서 온라인 학습 자체를 완전히 새로 설계하기보다, 좋은 encoder를 먼저 만들고 상단 predictor만 빠르게 적응시키는 구조는 구현과 유지 측면에서 현실적이다. 향후 연구에서는 논문이 제안하듯, separate meta-training phase 없이 representation을 스트림 중간중간 재최적화하는 방식, 혹은 OML 아이디어를 local learning rule, attention mechanism 등 다른 학습 구성 요소에 확장하는 방향이 유망해 보인다.

전체적으로 이 논문은 continual learning에서 “어떻게 기억할 것인가”뿐 아니라 “어떤 표현 위에서 배울 것인가”를 본격적으로 전면에 올려놓은 중요한 작업으로 볼 수 있다.
