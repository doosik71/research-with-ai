# On the Soft-Subnetwork for Few-Shot Class Incremental Learning

- **저자**: Haeyong Kang, Jaehong Yoon, Sultan Rizky Madjid, Sung Ju Hwang, Chang D. Yoo
- **발표연도**: 2023
- **arXiv**: https://arxiv.org/abs/2209.07529

## 1. 논문 개요

이 논문은 Few-Shot Class-Incremental Learning(FSCIL) 문제를 다룬다. FSCIL은 첫 번째 base session에서는 충분한 데이터로 여러 클래스를 학습하지만, 그 이후의 incremental session들에서는 각 클래스당 소수의 샘플만 주어진 상태에서 새로운 클래스를 계속 배워야 하는 설정이다. 이때 모델은 이전 클래스 지식을 잃지 않아야 하고, 동시에 아주 적은 샘플에 과적합하지도 않아야 한다.

저자들은 이 문제가 두 가지 난제를 동시에 갖는다고 본다. 첫째는 catastrophic forgetting이다. 새 클래스에 적응하기 위해 가중치를 업데이트하면 과거 클래스 성능이 크게 무너질 수 있다. 둘째는 overfitting이다. 새 세션마다 데이터가 매우 적기 때문에, 전체 네트워크를 그대로 미세조정하면 모델이 새 클래스의 몇 개 샘플에 지나치게 맞춰져 일반화가 악화된다.

이 논문의 핵심 목표는, dense network 내부에서 적절한 subnetwork를 구성하고 이를 soft mask 형태로 학습함으로써, 이전 지식은 보존하고 새 지식은 제한된 부분만 업데이트하여 배우는 방법을 제안하는 것이다. 제안 방법은 **SoftNet**이며, base session에서 모델 가중치와 soft mask를 함께 학습한 뒤, 이후 세션에서는 일부 minor subnetwork만 업데이트한다. 저자들은 이것이 forgetting과 overfitting을 동시에 완화한다고 주장한다.

이 문제의 중요성은 실제 응용에서 분명하다. 현실 환경에서는 새로운 클래스를 학습할 때 충분한 데이터가 항상 사용가능한 것이 아니며, 이미 학습한 클래스를 다시 모두 저장해 재학습하는 것도 어렵다. 따라서 적은 샘플만으로 새로운 개념을 안정적으로 추가하는 FSCIL은 실용적인 continual learning의 중요한 하위 문제라고 볼 수 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 dense neural network 안에 “이전 지식 보존용”과 “새 지식 적응용” 역할을 분리한 soft subnetwork가 존재한다는 관점이다. 저자들은 Lottery Ticket Hypothesis에서 출발하지만, 단순한 binary winning ticket이 아니라 **smooth, non-binary subnetwork**를 사용해야 FSCIL에 더 적합하다고 주장한다.

이를 위해 저자들은 **Regularized Lottery Ticket Hypothesis (RLTH)** 를 제안한다. 논문이 말하는 RLTH의 요지는, 무작위 초기화된 dense network 안에 이전 클래스 지식을 유지하면서도 새로운 클래스 지식을 배울 수 있는 “regularized subnetwork”가 존재한다는 것이다. 여기서 regularized라는 표현은 단순한 sparse selection이 아니라, 일부 가중치를 완전히 켜고 일부는 $0$과 $1$ 사이의 연속값으로 두어 학습을 부드럽게 제어한다는 의미에 가깝다.

SoftNet의 핵심 설계는 mask를 두 부분으로 나누는 것이다.

첫째, **major subnetwork**는 중요한 파라미터 집합으로, base session의 핵심 지식을 담는 역할을 한다. 이 부분은 top-$c\%$ weight score를 갖는 가중치에 대응하며 binary하게 $m=1$로 둔다. 이후 incremental session에서는 이 major 부분을 사실상 고정하여 forgetting을 줄인다.

둘째, **minor subnetwork**는 덜 중요한 파라미터 집합으로, $m \sim U(0,1)$의 soft value를 갖는다. 이 부분은 새 클래스 학습 시 제한적으로 업데이트되며, 이 soft한 scaling 자체가 regularizer처럼 작동해 few-shot 과적합을 줄인다는 것이 저자들의 주장이다.

기존 접근과의 차별점은 다음과 같다. FSLL 같은 기존 architecture-based FSCIL 방법은 세션별 subnetworks를 반복적인 pruning/retraining 과정으로 찾기 때문에 계산량이 크다. 반면 SoftNet은 **base session에서 한 번 soft subnetwork를 공동 학습한 후, 이후 세션들에서는 같은 soft subnetwork 구조를 유지하면서 minor weights만 업데이트**한다. 즉, 구조 탐색과 증분 학습이 비교적 단순하고 효율적으로 연결되어 있다는 점이 차별점이다.

## 3. 상세 방법 설명

논문은 먼저 FSCIL을 순차적 supervised learning 문제로 정식화한다. 세션 $t$의 데이터셋을 $D_t=\{(x_i^t,y_i^t)\}_{i=1}^{n_t}$라고 하고, 모델을 $f(\cdot;\theta)$라 하면, 일반적인 세션 학습은 다음처럼 표현된다.

$$
\theta^*=\arg\min_\theta \frac{1}{n_t}\sum_{i=1}^{n_t} L_t(f(x_i^t;\theta), y_i^t)
$$

여기서 $L_t$는 보통 cross-entropy 같은 분류 손실이다. 하지만 FSCIL에서는 이 단순한 전체 파라미터 업데이트가 바람직하지 않다. 새 세션 데이터가 너무 적어서 overfitting이 심해지고, 동시에 이전 클래스 지식도 망가질 수 있기 때문이다.

### Subnetwork 기반 관점

저자들은 먼저 binary mask를 쓰는 HardNet 관점을 소개한다. 세션 $t$에 대해 mask $m_t \in \{0,1\}^{|\theta|}$를 두고, 선택된 가중치만 사용하는 subnetwork를 학습하는 방식이다. 그러나 이진 마스크만으로는 미래 세션을 위한 유연한 조정 여지를 충분히 남기기 어렵다고 본다. 특히 FSCIL에서는 base 지식을 유지하면서도 새 클래스에 소량 적응할 수 있어야 하므로, 단순한 binary subnetwork보다 soft subnetwork가 더 적합하다고 주장한다.

그래서 저자들은 $m_t \in [0,1]^{|\theta|}$인 soft mask를 도입하고, 다음 목적함수를 제시한다.

$$
m_t^*=\arg\min_{m_t \in [0,1]^{|\theta|}} \frac{1}{n_t}\sum_{i=1}^{n_t} L_t(f(x_i^t;\theta \odot m_t), y_i^t) - J
\quad \text{subject to } |m_t| \le c
$$

본문의 식 표기는 다소 거칠게 추출되어 있어 $J$의 정확한 의미는 원문 OCR 상 완전히 명확하지 않다. 논문 텍스트에는 “session loss $J=L(f(x_i^t;\theta), y_i^t)$”라고 나오지만, 이 식이 왜 목적함수에서 빼기 형태로 들어가는지는 제공된 추출본만으로는 완전히 명료하지 않다. 따라서 이 항의 정확한 설계 의도는 추측하지 않는 것이 안전하다. 다만 중요한 점은, 모델 전체가 아니라 $\theta \odot m_t$ 형태의 masked parameter를 사용하고, sparsity budget $c$ 하에서 subnetwork를 찾는다는 점이다.

### Weight score와 mask 생성

각 weight에는 importance를 나타내는 **weight score** $s$가 대응된다. 점수가 높을수록 중요한 weight로 간주된다. base session에서는 모델 파라미터 $\theta$와 score $s$를 함께 최적화한다.

$$
\theta^*, m_t^* = \arg\min_{\theta, s} L_t(\theta \odot m_t; D_t)
$$

여기서 $m_t$는 score $s$에 indicator function을 적용해 top-$c\%$ 가중치를 선택함으로써 만들어진다. 즉, score가 높은 일부 파라미터가 major subnetwork가 된다.

하지만 indicator function은 미분이 불가능하므로, 저자들은 backward pass에서 **Straight-Through Estimator (STE)** 를 사용한다. 즉, forward에서는 discrete selection처럼 동작하게 하고, backward에서는 indicator의 gradient를 무시해 score도 업데이트할 수 있게 한다.

### Soft subnetwork의 구성

SoftNet의 핵심 식은 다음과 같다.

$$
m_{\text{soft}} = m_{\text{major}} \oplus m_{\text{minor}}
$$

여기서 $m_{\text{major}}$는 binary mask이고, $m_{\text{minor}} \sim U(0,1)$이다. 즉, major는 완전히 보존되는 핵심 경로이고, minor는 연속값으로 스케일된 보조 경로다. 결과적으로 전체 mask는 $[0,1]^{|\theta|}$ 범위의 soft mask가 된다.

base session에서는 learning rate $\alpha$로 다음과 같이 업데이트한다.

$$
\theta \leftarrow \theta - \alpha \frac{\partial L}{\partial \theta} m_{\text{soft}}
$$

또한 score도 비슷하게 업데이트한다.

$$
s \leftarrow s - \alpha \frac{\partial L}{\partial s} m_{\text{soft}}
$$

논문 설명에 따르면, base session에서 optimal soft-subnetwork를 한 번 얻은 뒤에는 이후 세션들에서 이 mask 자체는 더 이상 바꾸지 않는다. 즉, **subnetwork 구조는 base session에서 결정되고, incremental session에서는 그 구조를 활용만 한다.**

### 학습 절차

알고리즘은 크게 두 단계다.

첫째, **Base Training ($t=1$)**:
랜덤 초기화된 $\theta$와 $s$로 시작해, 각 layer에서 softmask $m_{\text{soft}}$를 구성하고, cross-entropy 기반 손실로 $\theta$와 $s$를 jointly optimize한다. 이 단계에서 major와 minor subnetwork가 정해진다.

둘째, **Incremental Training ($t \ge 2$)**:
현재 세션 데이터 $D_t$와 이전 세션에서 저장한 exemplar를 함께 사용한다. 이때 전체 $\theta$를 업데이트하지 않고, **minor subnetwork에 해당하는 일부 파라미터만 업데이트**한다. major는 base 지식 보존을 위해 고정된다.

### Incremental session의 분류 손실

incremental session에서는 prototype 기반 metric learning 형태의 손실을 사용한다. 저자들은 Euclidean distance 대신 **cosine distance**를 사용한다. 논문은 cosine distance가 표현 벡터의 방향 정보를 반영해 더 normalized informative measurement를 제공한다고 설명한다.

손실 함수는 다음과 같다.

$$
L_m(z;\theta \odot m_{\text{soft}})
=
-\sum_{z \in D}\sum_{o \in O}\mathbf{1}(y=o)\log
\left(
\frac{e^{-d(p_o, f(x;\theta \odot m_{\text{soft}}))}}
{\sum_{o_k \in O} e^{-d(p_{o_k}, f(x;\theta \odot m_{\text{soft}}))}}
\right)
$$

여기서 $d(\cdot,\cdot)$는 cosine distance이고, $p_o$는 class $o$의 prototype이다. $O$는 지금까지 관측한 전체 클래스 집합이고, $D = D_t \cup P$는 현재 세션 데이터와 이전 세션 exemplar/prototype 정보를 합친 것이다.

새 클래스 prototype은 다음처럼 계산한다.

$$
p_o = \frac{1}{N_o}\sum_i \mathbf{1}(y_i=o) f(x_i; \theta \odot m_{\text{soft}})
$$

즉, 각 클래스의 feature 평균을 prototype으로 저장한다. base class prototype은 base session에서 저장하고, 이후 세션에서도 모든 클래스 prototype을 유지한다.

### 추론 과정

추론에서는 nearest class mean (NCM) classifier를 사용한다. 모든 샘플을 feature extractor $f$의 embedding space로 보낸 뒤, 각 클래스 prototype과의 Euclidean distance를 계산하여 가장 가까운 prototype의 클래스를 예측한다.

$$
o_k^* = \arg\min_{o \in O} d_u(f(x; \theta \odot m_{\text{soft}}), p_o)
$$

흥미로운 점은 학습 손실에서는 cosine distance를 쓰지만, 추론에서는 공정 비교를 위해 기존 FSCIL 문헌과 동일하게 Euclidean distance 기반 NCM을 사용했다는 것이다.

## 4. 실험 및 결과

### 실험 설정

주요 벤치마크는 CIFAR-100과 miniImageNet이다. 두 데이터셋 모두 100개 클래스를 사용하며, 60개 base class와 40개 novel class로 나눈다. base session 이후에는 8개의 incremental session이 이어지고, 각 세션은 **5-way 5-shot** 문제로 구성된다. 즉, 세션마다 5개 새 클래스를 고르고 클래스당 5개의 학습 샘플만 사용한다.

추가 실험으로 CUB-200-2011도 다룬다. 이 경우 첫 100개 클래스를 base class로 쓰고, 나머지 100개 클래스를 10개 session에 걸친 **10-way 5-shot**으로 학습한다.

백본은 주로 ResNet18이며, 일부 architecture-wise ablation에서는 ResNet20, ResNet32, ResNet50도 사용한다. base session에서는 top-$c\%$ 가중치를 layer별로 선택해 best validation accuracy를 주는 soft-subnetwork를 얻는다. incremental session에서는 CIFAR-100과 miniImageNet의 경우 6 epochs, learning rate $0.02$로 학습한다. 논문에 따르면 ResNet18에서는 conv4x, ResNet20에서는 conv3x의 minor weights를 주로 업데이트한다.

비교 대상은 iCaRL, Rebalance, TOPIC, IDLVQ-C, F2M, FSLL, 그리고 상한선 성격의 cRT 등이다. 또한 binary subnetwork 기반 **HardNet**을 직접 비교군으로 둔다.

### CIFAR-100 결과

ResNet18 기반 5-way 5-shot 설정에서, 표 1의 최종 session(9) 정확도를 보면 다음과 같다.

- cRT: 45.28
- F2M: 44.67
- HardNet, $c=99\%$: 46.31
- SoftNet, $c=99\%$: 46.63

즉, SoftNet은 이 설정에서 cRT보다도 약간 높은 성능을 보이며, 논문 기준 strongest baseline보다 우수하다. 특히 FSLL이 38.16 혹은 보고값 기준 42.22 수준인 것과 비교하면, SoftNet은 pruning/subnetwork 기반 접근에서 훨씬 강한 결과를 보였다.

또한 표 9를 보면 SoftNet은 $c$가 증가할수록 대체로 성능이 개선된다. 예를 들어 최종 session 정확도는 $c=10\%$일 때 39.58, $c=50\%$일 때 44.61, $c=90\%$일 때 46.15, $c=99\%$일 때 46.63이다. 즉, 지나치게 작은 capacity보다는 상대적으로 큰 subnetwork가 CIFAR-100 FSCIL에 더 유리했다.

### miniImageNet 결과

miniImageNet에서는 SoftNet의 이점이 더 크다. 표 2에서 최종 session(9) 정확도를 보면,

- cRT: 44.85
- F2M: 44.65
- HardNet, $c=80\%$: 45.66
- SoftNet, $c=80\%$: 50.48
- SoftNet, $c=90\%$: 50.39
- SoftNet, $c=97\%$: 50.24

즉, SoftNet은 cRT 대비 약 5.6 point 높은 성능을 기록한다. 이는 단순한 소폭 개선이 아니라, benchmark FSCIL setting에서 상당히 큰 차이다. 논문은 이를 통해 SoftNet이 few-shot incremental setting에서 forgetting과 overfitting을 동시에 완화한다고 해석한다.

표 11 전체를 보면 miniImageNet에서는 CIFAR-100보다 sparsity에 더 민감하다. SoftNet은 $c=30\%$에서도 최종 47.03으로 이미 cRT를 넘기고, $c=80\% \sim 97\%$ 구간에서 가장 높은 결과를 낸다. 반면 HardNet은 같은 데이터셋에서 capacity 설정에 따라 성능 변동 폭이 크고, SoftNet보다 전반적으로 뒤처진다. 이는 저자들이 주장하는 “softness가 주는 regularization 효과”와 연결된다.

### Layer-wise 결과

표 3에서는 miniImageNet에서 minor weights를 어느 layer에 둘 것인지 비교한다. $c=97\%$ 고정 시 Conv2x, Conv3x, Conv4x, Conv5x 모두 비슷하지만, Conv5x가 최종 50.24로 가장 높다. 반면 모든 layer의 minor weights를 다 학습하는 경우 최종 25.64로 급락한다.

이 결과는 중요하다. 논문은 이를 통해 lower layer feature는 상대적으로 일반적이고 재사용 가능하며, higher layer 쪽 일부만 조정하는 것이 few-shot incremental adaptation에 더 적합하다고 해석한다. 동시에 “모든 layer를 다 미세조정하면” catastrophic forgetting과 overfitting이 다시 심해진다는 것을 보여준다.

### Architecture-wise 결과

표 4에 따르면 CIFAR-100에서 architecture에 따라 최적 sparsity가 달라진다.

- ResNet18: SoftNet $c=99\%$에서 최종 46.63
- ResNet20: SoftNet $c=90\%$에서 최종 49.20
- ResNet32: SoftNet $c=93\%$에서 최종 50.76
- ResNet50: SoftNet $c=80\%$에서 최종 52.18

즉, 더 큰 모델에서는 SoftNet 성능이 더 높아지고, 최적 $c$도 달라진다. 저자들은 이것이 subnetwork의 유효 sparsity가 architecture-dependent하다는 신호라고 본다.

### 추가 비교 실험

부록의 TOPIC split 기준 비교에서도 SoftNet은 매우 강하다.

CIFAR-100(Table 5)에서는 SoftNet $c=50\%$가 최종 55.33, $c=80\%$가 54.94를 기록하며, ALICE 54.10, LIMIT 51.23, MetaFSCIL 49.97보다 높다.

miniImageNet(Table 6)에서는 SoftNet $c=87\%$가 최종 54.68로 매우 강하지만, ALICE 55.70보다는 낮다. 따라서 “모든 경우 절대 최고”라고 말하기는 어렵고, 적어도 제공된 표 기준으로는 miniImageNet TOPIC split에서는 ALICE가 더 높다.

CUB-200-2011(Table 7)에서는 SoftNet $c=90\%$가 최종 56.75로, ALICE 60.10이나 cRT 59.30보다 낮다. 따라서 본문 결론의 “state-of-the-art” 주장은 실험 설정과 비교 기준에 따라 부분적으로 해석해야 한다. 적어도 제공된 표만 보면, 모든 벤치마크/모든 split에서 항상 최고라고 말할 수는 없다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 FSCIL의 두 핵심 문제인 forgetting과 overfitting을 하나의 구조적 아이디어로 함께 다뤘다는 점이다. major/minor subnetworks로 역할을 분리하고, 새 세션에서는 minor weights만 업데이트하는 방식은 직관적으로도 설득력이 있다. 특히 “soft mask가 regularizer처럼 작동한다”는 설명은 few-shot setting과 잘 맞는다.

두 번째 강점은 실험이 비교적 폭넓다는 점이다. CIFAR-100, miniImageNet, CUB-200-2011을 포함하고, HardNet과 SoftNet 비교, layer-wise 분석, architecture-wise 분석, loss landscape, t-SNE 등 다양한 관점의 ablation을 제공한다. 특히 “모든 layer를 업데이트하면 성능이 크게 망가진다”는 표 3 결과는 제안 방식의 핵심 가정이 실제로 중요함을 잘 보여준다.

세 번째 강점은 효율성 측면의 위치 설정이 명확하다는 점이다. 저자들은 FSLL이 iterative reidentification과 retraining/pruning으로 계산량이 크다고 비판하고, SoftNet은 base session에서 soft-subnetwork를 한 번 학습한 뒤 계속 재사용한다고 설명한다. 제공된 텍스트에는 구체적인 학습 시간 비교 표는 없지만, 설계 자체는 분명히 더 단순하다.

반면 한계도 분명하다. 첫째, 이론적 정식화 일부가 다소 불명확하다. 특히 식 (2)의 $-J$ 항은 제공된 추출 텍스트만으로는 정확한 의미가 충분히 명료하지 않다. 본문 OCR 문제일 수 있지만, 적어도 현재 제공된 텍스트 기준으로는 목적함수 설명이 매끄럽지 않다.

둘째, SoftNet이 왜 soft mask를 쓰면 과적합이 줄어드는지에 대한 이론적 설명은 완전히 엄밀하지 않다. 논문은 flat minima, regularization, loss landscape 등의 직관을 제시하지만, 이것이 FSCIL 일반 setting에서 언제나 성립하는지까지 강하게 증명한 것은 아니다. 부록의 Lipschitz 기반 논의도 직관 보강에 가깝고, 완전한 엄밀 증명으로 읽히지는 않는다.

셋째, 성능 주장은 일부 과장될 여지가 있다. 본문에서는 state-of-the-art를 강조하지만, 제공된 추가 표를 보면 ALICE나 LIMIT가 더 높은 결과를 보이는 경우도 있다. 따라서 “모든 설정에서 명백한 최고”라기보다는, **subnetwork 기반 방법 중 매우 강력하고, 특정 benchmark setting에서는 기존 강한 baseline을 넘는 방법** 정도로 해석하는 편이 더 정확하다.

넷째, 논문 스스로도 제한점을 인정한다. major subnetwork가 base knowledge를 보존하는 역할을 하기 때문에, 이 부분이 잘못 튜닝되면 이전 지식 손실이 생길 수 있다. 또한 magnitude criterion으로 중요한 파라미터가 비교적 명시적으로 드러나므로, 파라미터 노출 시 민감한 지식이 공격에 취약할 수 있다는 보안 관점의 우려도 언급한다.

다섯째, long sequence의 많은 세션이 들어오는 일반적인 CIL로 확장할 때는 한계가 있을 수 있다. 현재 방법은 base session에서 정한 major/minor 구조를 유지하는데, 매우 긴 시퀀스에서 새로운 지식이 계속 들어오면 minor capacity만으로 충분한지 확실하지 않다. 저자들도 미래 연구로 parameter expansion을 고려한다고 밝혔다.

## 6. 결론

이 논문은 FSCIL에서 dense network 전체를 업데이트하는 대신, base session에서 미리 학습한 soft-subnetwork를 이용해 incremental session에서는 일부 minor weights만 조정하는 **SoftNet**을 제안했다. 핵심은 major subnetwork가 이전 지식을 보존하고, minor subnetwork가 새 클래스 적응을 담당하게 하는 것이다. 이때 minor 부분을 $0$과 $1$ 사이의 soft value로 두어 few-shot 과적합을 줄이는 것이 중요한 설계 포인트다.

실험적으로 SoftNet은 CIFAR-100과 miniImageNet의 표준 FSCIL 설정에서 강한 성능을 보였고, 특히 miniImageNet에서는 cRT 및 여러 기존 방법보다 상당히 높은 정확도를 기록했다. 또한 layer-wise, architecture-wise 분석을 통해 어떤 층을 얼마나 sparse하게 조정해야 하는지가 성능에 중요하다는 점도 보여주었다.

종합하면, 이 연구의 주요 기여는 “FSCIL에서 subnetwork는 단순히 sparse해야 하는 것이 아니라, **보존용 major와 적응용 minor로 구성된 soft-subnetwork**여야 한다”는 관점을 제시한 데 있다. 실제 적용 측면에서는 적은 데이터로 새로운 클래스를 추가해야 하는 비전 시스템에 유용할 가능성이 있으며, 향후에는 더 긴 continual setting, adaptive capacity expansion, 그리고 더 엄밀한 이론 분석으로 확장될 여지가 크다.
