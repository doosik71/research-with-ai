# Decoupling Features in Hierarchical Propagation for Video Object Segmentation

- **저자**: Zongxin Yang, Yi Yang
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2210.09782

## 1. 논문 개요

이 논문은 semi-supervised Video Object Segmentation(VOS)을 위한 새로운 hierarchical propagation 방법인 **DeAOT**를 제안한다. 문제 설정은 비디오의 첫 프레임 등 일부 프레임에서 객체 마스크가 주어졌을 때, 나머지 모든 프레임에서 동일 객체를 계속 추적하고 분할하는 것이다. 이 작업은 비디오 이해, 추적, 편집, 자율주행 등 여러 응용의 기반이 되기 때문에 중요하다.

논문의 출발점은 기존의 AOT 계열 방법이 hierarchical propagation을 통해 과거 프레임의 정보를 현재 프레임으로 점진적으로 전달하면서 좋은 성능을 얻었지만, 그 과정에서 **object-agnostic visual information**과 **object-specific ID information**을 같은 embedding space 안에서 함께 처리한다는 점이다. 저자들은 이 설계가 깊은 propagation layer로 갈수록 원래의 시각적 단서를 잃게 만들 수 있다고 본다. 특히 현재 프레임에서 객체를 찾고 구분하는 데 중요한 것은 결국 visual feature matching인데, ID 정보가 feature 공간을 잠식하면 성능이 제한될 수 있다는 것이 핵심 문제의식이다.

따라서 이 논문은 계층적 전파 과정에서 시각 특징과 ID 특징을 분리해 다루면 더 나은 visual embedding 학습과 더 안정적인 객체 전파가 가능하다고 주장한다. 동시에, dual-branch 구조로 인해 늘어날 수 있는 계산량을 줄이기 위해 **Gated Propagation Module (GPM)**이라는 더 효율적인 propagation 모듈도 함께 제안한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 간단하지만 설계상 매우 중요하다. 기존 AOT는 현재 프레임의 feature가 계층적 전파를 거치며 점차 object-agnostic 상태에서 object-specific 상태로 바뀌도록 만든다. 즉, 같은 feature tensor 안에 visual 정보와 ID 정보가 함께 섞인다. 저자들은 이 과정이 깊어질수록 visual 정보가 희석되거나 잊히는 문제가 생긴다고 본다.

이를 해결하기 위해 DeAOT는 propagation을 두 갈래로 분리한다.

첫째, **Visual Branch**는 객체 자체의 시각적 단서를 유지하고 정제하는 역할을 한다. 여기서는 object-specific mask/ID를 직접 섞지 않고, 과거 프레임의 visual embedding을 이용해 현재 프레임의 visual embedding을 점차 더 discriminative하게 만든다.

둘째, **ID Branch**는 객체 정체성(identity)과 마스크 정보를 전달하는 역할을 맡는다. 이 branch는 과거 프레임의 ID embedding과 mask encoding을 현재 프레임으로 전달하여 최종 segmentation 예측이 가능하게 만든다.

중요한 점은 두 branch가 완전히 독립적으로 따로 객체를 찾는 것이 아니라, **attention map은 visual branch에서 계산하고 이를 ID branch가 공유해서 사용한다는 것**이다. 저자들의 관점에서는 객체의 대응 관계를 정하는 근거는 ID 번호가 아니라 visual feature이기 때문이다. 이 설계 덕분에 visual branch는 object-specific bias에 덜 오염된 채 객체 매칭을 수행하고, ID branch는 그 결과를 이용해 객체 정체성을 전달할 수 있다.

기존 접근과의 차별점은 명확하다. AOT는 하나의 embedding 공간 안에서 visual 정보와 ID 정보를 함께 처리하지만, DeAOT는 둘을 **명시적으로 decouple**한다. 또한 propagation 모듈도 AOT의 LSTT(Long Short-Term Transformer) 대신 single-head attention 기반의 GPM으로 바꾸어, 정확도뿐 아니라 속도까지 개선하려고 했다.

## 3. 상세 방법 설명

전체 파이프라인은 reference frame의 annotation을 시작점으로 하여, 비디오를 frame-by-frame으로 처리한다. 각 시점에서 현재 프레임은 encoder를 통해 feature로 바뀌고, 과거 프레임의 메모리와 함께 여러 propagation layer를 거친 뒤 decoder를 통해 현재 프레임의 segmentation mask가 예측된다. DeAOT는 이 과정에서 visual embedding과 ID embedding을 각각 별도 branch에서 전파한다.

### 기본 attention 기반 propagation

논문은 일반적인 attention 기반 propagation을 다음과 같이 정리한다.

$$
Att(Q, K, V) = Corr(Q, K)V = softmax\left(\frac{QK^{tr}}{\sqrt{C}}\right)V
$$

여기서 $Q$는 현재 프레임의 query embedding, $K$와 $V$는 메모리 프레임들의 key/value embedding이다. 즉, 현재 프레임 feature가 과거 프레임 feature와 얼마나 유사한지 attention으로 계산한 뒤, 그 정보를 현재 프레임으로 읽어오는 구조이다.

AOT의 계층적 propagation에서는 각 layer에서 현재 feature에 과거 feature뿐 아니라 mask의 ID embedding도 함께 넣는다. 논문은 이를 단순화하여 다음과 같이 표현한다.

$$
\tilde{X}_l^t = Att(X_l^t W_l^K, X_l^m W_l^K, X_l^m W_l^V + ID(Y^m))
$$

여기서 $ID(Y^m)$는 과거 프레임 마스크를 ID embedding으로 바꾼 것이고, 결과적으로 layer가 깊어질수록 현재 프레임 feature는 점점 object-specific해진다.

저자들은 바로 이 지점이 문제라고 본다. 채널 수는 제한되어 있으므로 ID 정보가 많이 들어올수록 원래 visual 정보가 줄어들 가능성이 높다. 실제로 논문은 ID embedding 안의 최대 ID 수를 늘리면 AOT 성능이 감소한다는 실험 결과를 제시한다. 이것은 시각 정보와 ID 정보가 같은 공간을 공유하는 방식이 근본적으로 충돌을 일으킬 수 있음을 보여주려는 근거다.

### Dual-branch propagation

DeAOT에서는 feature를 두 종류로 나눈다.

- $I$: object-agnostic visual embedding
- $M$: object-specific identification embedding

#### Visual Branch

Visual branch는 과거 프레임의 visual embedding만 사용하여 현재 프레임의 visual embedding을 갱신한다.

$$
\tilde{I}_l^t = Att(I_l^t W_l^K, I_l^m W_l^K, I_l^m W_l^V)
= Corr(I_l^t W_l^K, I_l^m W_l^K) I_l^m W_l^V
$$

여기에는 $ID(Y^m)$가 들어가지 않는다. 따라서 이 branch는 특정 객체의 ID 정보에 끌려가지 않고, 객체를 구분하는 데 필요한 시각적 feature를 유지하고 더 잘 정제하는 방향으로 학습될 수 있다.

#### ID Branch

반면 ID branch는 과거 프레임의 ID embedding과 mask 정보를 현재 프레임으로 옮긴다.

$$
\tilde{M}_l^t = Att(I_l^t W_l^K, I_l^m W_l^K, M_l^m W_l^V + ID(Y^m))
$$

즉,

$$
\tilde{M}_l^t = Corr(I_l^t W_l^K, I_l^m W_l^K)(M_l^m W_l^V + ID(Y^m))
$$

여기서 중요한 점은 attention map인 $Corr(I_l^t W_l^K, I_l^m W_l^K)$를 visual branch의 계산에 기반해 만든다는 것이다. ID branch는 그 attention을 재사용할 뿐이다. 이는 "객체를 다시 찾는 근거는 ID index가 아니라 visual similarity"라는 저자들의 설계 철학을 반영한다.

### Gated Propagation Function

기존 AOT의 LSTT는 multi-head attention을 사용한다. 논문은 이것이 효율 병목이라고 지적한다. 특히 long-term attention의 계산 복잡도는 head 수 $N$에 비례해서 증가하므로, head 수를 줄이면 빨라지지만 성능 손실이 생긴다.

이를 해결하기 위해 제안된 핵심 함수가 **Gated Propagation (GP)** 이다.

$$
GP(U, Q, K, V) = F_{dw}(\sigma(U) \odot Corr(Q, K)V) W^O
$$

여기서:

- $U$는 gating embedding
- $\sigma(\cdot)$는 비선형 gating 함수이며 논문에서는 SiLU/Swish를 사용
- $\odot$는 element-wise multiplication
- $F_{dw}$는 depth-wise 2D convolution
- $W^O$는 출력 projection

이 식의 의미는 다음과 같다.

먼저 일반 attention 결과 $Corr(Q,K)V$를 얻는다. 그 다음 gating term인 $\sigma(U)$로 이 결과를 조건부로 조절한다. 즉, 모든 attention 결과를 동일하게 통과시키는 것이 아니라 현재 위치의 feature 상태에 따라 어느 정도를 통과시킬지 정한다. 마지막으로 depth-wise convolution을 통해 local spatial context를 가볍게 보강한다. 논문은 이것이 segmentation 같은 dense prediction 문제에서 중요하다고 본다.

### Gated Propagation Module (GPM)

GPM은 세 종류의 propagation으로 구성된다.

- **Long-term propagation**: 메모리 프레임들에서 현재 프레임으로 정보 전달
- **Short-term propagation**: 직전 프레임의 local neighborhood에서 현재 프레임으로 정보 전달
- **Self-propagation**: 현재 프레임 내부에서 객체 관계를 정리

기존 LSTT와 비교하면 feed-forward module을 제거했고, 각 propagation을 모두 GP 함수로 구현한다.

Visual branch의 long-term propagation은 다음과 같이 쓰인다.

$$
GP_{vis}^{lt}(I_l^t, I_l^t, I_l^m, I_l^m)
=
GP(I_l^t W_l^U, I_l^t W_l^K, I_l^m W_l^K, I_l^m W_l^V)
$$

ID branch의 long-term propagation은 다음과 같다.

$$
GP_{id}^{lt}(M_l^t, I_l^t, I_l^m, M_l^m, Y^m)
=
GP(M_l^t W_l^U, I_l^t W_l^K, I_l^m W_l^K, M_l^m W_l^V + ID(Y^m))
$$

Short-term propagation은 직전 프레임의 전체 공간이 아니라 위치 $p$ 주변의 $\lambda \times \lambda$ neighborhood 안에서만 attention을 수행한다. 이는 연속 프레임 사이의 motion이 대체로 smooth하므로, short-term matching에 전역 non-local attention이 꼭 필요하지 않다는 가정에 기반한다. 논문에서 $\lambda=15$를 사용한다.

Self-propagation에서는 visual embedding과 ID embedding을 channel 차원에서 concatenate한 $(I_l^t \oplus M_l^t)$를 key/query 계산에 사용한다. 저자들은 여기서 ID embedding이 일종의 positional embedding처럼 작용하여 현재 프레임 내부에서 객체 association을 더 잘하도록 돕는다고 설명한다. 다만 이 부분에서 정확히 왜 positional embedding과 유사한 효과가 생기는지는 개념적으로 설명할 뿐, 더 깊은 이론적 분석은 제공하지 않는다.

### 구현 세부사항

논문은 세 가지 backbone encoder를 사용한다.

- MobileNet-V2
- ResNet-50
- Swin-B

Decoder는 FPN을 사용한다. Embedding 차원은 visual과 ID 모두 $C=256$, matching feature 차원은 $C_k=128$, propagation feature 차원은 $C_v=512$이다. GPM의 depth-wise convolution 커널 크기는 5이고, gating 함수는 SiLU/Swish다. ID embedding의 최대 객체 수는 10으로 둔다.

모델 변형은 propagation layer 수 $L$와 long-term memory 구성에 따라 나뉜다.

- DeAOT-T: $L=1$, 메모리는 reference frame만 사용
- DeAOT-S: $L=2$
- DeAOT-B: $L=3$
- DeAOT-L: $L=3$이며 일정 간격 $\delta$로 long-term memory 추가

학습은 먼저 static image dataset들로 만든 synthetic video sequence에서 pre-training을 하고, 이후 YouTube-VOS와 DAVIS에서 main training을 수행한다. optimization 전략과 관련 hyper-parameter는 AOT와 동일하게 맞췄다고 설명한다. 구체적인 optimizer 설정 수치 등은 본문에 자세히 없고 supplementary에 있다고만 되어 있다.

## 4. 실험 및 결과

실험은 세 개의 VOS benchmark와 하나의 tracking benchmark에서 수행된다.

- **YouTube-VOS**
- **DAVIS 2017**
- **DAVIS 2016**
- **VOT 2020**

VOS 평가는 주로 $J$ score(IoU 기반 region similarity), $F$ score(boundary similarity), 그리고 평균인 $J\&F$를 사용한다. YouTube-VOS에서는 seen/unseen class에 대해 $J_S, F_S, J_U, F_U$도 함께 본다. VOT 2020은 official **EAO** 기준을 사용한다.

### YouTube-VOS

YouTube-VOS는 대규모 multi-object benchmark로, 일반화 성능과 속도를 평가하기에 중요하다. 여기서 DeAOT는 모든 AOT 대응 모델을 정확도와 속도 모두에서 앞선다.

예를 들어:

- **AOT-T**: 80.2% at 41.0fps
- **DeAOT-T**: 82.0% at 53.4fps

즉 가장 작은 모델에서도 정확도는 약 1.8%p 높고, 속도도 더 빠르다.

중간 규모 모델에서도 같은 경향이 이어진다.

- **AOT-L**: 83.8% at 16.0fps
- **DeAOT-L**: 84.8% at 24.7fps

backbone을 ResNet-50로 바꾸면:

- **R50-AOT-L**: 84.1% at 14.9fps
- **R50-DeAOT-L**: 86.0% at 22.4fps

가장 큰 Swin-B 기반 모델은:

- **SwinB-AOT-L**: 84.5%
- **SwinB-DeAOT-L**: 86.2%

로 보고되며, 논문은 이를 당시 SOTA라고 주장한다. 특히 DeAOT-T가 SST보다 약간 높은 정확도를 내면서도 CFBI보다 약 15배 빠르다고 강조한다.

이 결과가 시사하는 바는 두 가지다. 하나는 feature decoupling이 실제 성능 향상으로 이어진다는 점이고, 다른 하나는 GPM이 dual-branch 구조의 추가 계산을 상쇄하고도 오히려 더 빠르게 만들 수 있다는 점이다.

### DAVIS 2017

DAVIS 2017은 multi-object 분할의 대표 벤치마크다. 여기서도 DeAOT는 전반적으로 AOT보다 우수하다.

예를 들어:

- **R50-AOT-L**: validation/test에서 84.9% / 79.6%
- **R50-DeAOT-L**: 85.2% / 80.7%, 27fps

SwinB-DeAOT-L은 validation/test에서 **86.2% / 82.8%**를 기록한다. 다만 일부 작은 변형에서는 DeAOT가 AOT보다 항상 큰 폭으로 높지는 않다. 예를 들어 DeAOT-S와 DeAOT-B는 DAVIS 2017 validation 평균에서 AOT 대응 모델과 비슷하거나 약간 낮은 수치도 보인다. 따라서 “모든 변형에서 절대적으로 항상 우세”라기보다는, 전체적으로 특히 큰 모델에서 개선이 뚜렷하다고 해석하는 것이 더 정확하다.

### DAVIS 2016

DAVIS 2016은 single-object benchmark다. DeAOT는 멀티 객체 설정에 초점을 둔 계열임에도 단일 객체에서도 좋은 성능을 보인다.

- **STCN**: 91.6%
- **DeAOT-L**: 92.0%
- **R50-DeAOT-L**: 92.3%
- **SwinB-DeAOT-L**: 92.9%

특히 SwinB-DeAOT-L이 가장 높은 수치를 기록했다. 이는 제안 방법이 단지 multi-object identity propagation에만 특화된 것이 아니라, video object matching 자체를 더 잘 수행하게 했음을 뒷받침한다.

### VOT 2020

VOT 2020은 더 긴 시퀀스와 어려운 상황을 포함한다. 여기서도 DeAOT는 강한 결과를 보인다.

- **MixFormer-L**: EAO 0.555
- **AOT-L**: EAO 0.574
- **DeAOT-L**: EAO 0.591
- **R50-DeAOT-L**: EAO 0.613
- **SwinB-DeAOT-L**: EAO 0.622

실시간 제약이 있는 EAO RT에서도 R50-DeAOT-L이 0.571을 기록해 AlphaRef보다 높다고 보고한다. 이는 VOS 모델이 tracking benchmark에서도 강한 일반화 성능을 보여준다는 점에서 의미가 있다.

### 정성적 결과

정성적 비교에서는 DeAOT가 AOT보다 작은 객체나 크기 변화가 큰 객체에서 더 잘 동작한다고 제시한다. 예로 ski poles, ski board 같은 사례가 언급된다. 반면 심한 occlusion이 있고 서로 매우 비슷한 multiple objects가 등장할 때는 여전히 실패할 수 있다고 한다. 예시로 dancer와 cow가 제시된다.

즉, 시각 특징 보존을 강화했어도 심한 가림과 고유한 시각 단서 부족 문제는 완전히 해결하지 못했다는 점을 스스로 인정한다.

### Ablation Study

이 논문의 설계를 뒷받침하는 핵심은 ablation이다.

#### Propagation module 비교

YouTube-VOS 2018에서 DeAOT-S 기준:

- **GPM + decoupling**: 82.5
- **w/o decoupling**: 81.5
- **w/o decoupling, 채널 512**: 82.0
- **LSTT 사용**: 80.3

이 결과는 두 가지를 보여준다. 첫째, 단순히 채널 수를 늘려도 decoupling의 이득을 완전히 대체하지 못한다. 둘째, GPM 자체도 중요한 기여를 한다.

#### Head number 실험

- DeAOT는 head 수를 1에서 8로 바꿔도 정확도 변화가 거의 없다: 82.5 대 82.5
- 반면 속도는 38.7fps에서 24.7fps로 느려진다
- AOT는 single-head로 바꾸면 빨라지지만 정확도가 0.7%p 하락한다

즉, GPM은 single-head attention 환경에서도 성능을 유지하도록 설계되었고, 이것이 효율 향상의 핵심이라는 주장에 설득력을 준다.

#### Attention map 공유 방식

Long-term / short-term propagation에서 attention map 계산에 visual embedding만 쓸 때가 가장 좋았다. 여기에 ID embedding까지 넣으면 82.5에서 82.1로 떨어졌다. 반대로 self-propagation에서는 ID embedding을 함께 넣는 것이 유리했다. 이는 저자들의 설계 선택, 즉 “객체 매칭은 visual 중심, 현재 프레임 내부 association은 visual+ID 결합”이 경험적으로 타당하다는 뜻이다.

#### Depth-wise convolution 커널 크기

$F_{dw}$를 제거하면 82.5에서 81.1로 크게 하락한다. 커널 크기 3, 5, 9 중에서는 5가 가장 좋았다. 이는 local spatial context 보강이 실제로 segmentation 성능에 중요함을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정과 해법이 매우 논리적으로 연결되어 있다는 점이다. 저자들은 단순히 성능을 높인 것이 아니라, 기존 hierarchical propagation의 구조적 약점을 먼저 짚고 그에 맞는 설계를 제안했다. “visual 정보와 ID 정보를 같은 공간에 계속 누적하면 visual clue가 약해질 수 있다”는 주장은 직관적이고, ablation도 그 주장을 꽤 잘 뒷받침한다.

또 다른 강점은 정확도와 효율을 동시에 개선했다는 점이다. 보통 dual-branch 구조는 계산량을 늘릴 위험이 있는데, 이 논문은 GPM을 함께 제안해 오히려 AOT보다 빠른 경우를 보여준다. 특히 single-head attention으로도 성능을 유지한 점은 실용성이 높다. 실시간 또는 near real-time 처리에 관심 있는 응용에서는 상당히 중요한 장점이다.

실험 범위도 비교적 탄탄하다. large-scale multi-object benchmark, single-object benchmark, tracking benchmark까지 포함해 모델의 일반화 가능성을 보여주려 했다. 또한 qualitative result에서 실패 사례를 숨기지 않고 제시한 점도 좋다.

반면 한계도 분명하다. 첫째, 논문은 visual 정보 손실 문제를 설득력 있게 제기하지만, 이것을 직접 측정하거나 representation 관점에서 정량적으로 분석하지는 않는다. 예를 들어 깊은 layer에서 visual embedding이 어떻게 변하는지, decoupling이 feature separability를 어떻게 개선하는지에 대한 representation-level 분석은 없다. 따라서 핵심 가설은 경험적으로 지지되지만, 이론적으로 깊게 검증된 것은 아니다.

둘째, 심한 occlusion과 highly similar objects 상황에서는 여전히 실패한다. 이는 VOS에서 가장 어려운 경우 중 하나인데, dual-branch 설계만으로는 충분하지 않음을 보여준다. attention map을 visual branch에 의존시키는 구조가 오히려 시각적으로 구분이 어려운 경우에는 취약할 가능성도 있다.

셋째, 학습 세부 설정은 AOT를 따른다고 하나, 본문에는 optimizer, learning schedule, loss 구성에 대한 구체 설명이 거의 없다. 본문만 읽고 재현하려는 입장에서는 부족하다. 특히 사용된 loss function이 명시적으로 설명되지 않았고, decoder 학습 목표가 어떤 형태의 segmentation loss인지도 본문에는 없다. 이 부분은 추측할 수는 있으나, 논문 본문에서 명확히 적지 않았으므로 확정적으로 말할 수 없다.

넷째, attention map 공유가 항상 최선인지에 대한 논의는 제한적이다. 논문은 visual 기반 attention이 더 낫다고 보이지만, 특정 상황에서는 ID-aware matching이 도움이 될 가능성도 있다. 본문에는 그 경계 조건에 대한 심층 분석이 없다.

종합하면 이 논문은 설계 문제를 정확히 짚고 실용적인 개선을 제시한 강한 engineering-oriented 연구이지만, representation 분석이나 실패 조건에 대한 근본적 해석은 상대적으로 약하다.

## 6. 결론

이 논문은 semi-supervised VOS를 위한 hierarchical propagation을 다시 생각하고, visual embedding과 ID embedding을 분리하는 **DeAOT**를 제안했다. 핵심 기여는 두 가지다. 하나는 visual branch와 ID branch를 분리하여 깊은 propagation에서도 object-agnostic visual information을 유지하도록 만든 점이고, 다른 하나는 single-head attention 기반의 **GPM**을 도입해 정확도와 속도를 동시에 개선한 점이다.

실험적으로 DeAOT는 YouTube-VOS, DAVIS 2017, DAVIS 2016, VOT 2020에서 강한 성능을 보였고, 여러 설정에서 기존 AOT를 일관되게 앞섰다. 특히 큰 backbone뿐 아니라 작은 모델에서도 속도-정확도 균형이 좋아 실제 시스템에 적용하기 유리해 보인다.

향후 연구 측면에서는, 이 논문이 제안한 “feature decoupling” 관점이 다른 video segmentation 또는 tracking 구조로도 확장될 가능성이 크다. 동시에 occlusion, highly similar objects, 장기적 identity ambiguity 같은 더 어려운 문제를 해결하려면 visual cue 외의 구조적 단서나 더 강한 memory reasoning이 추가로 필요할 수 있다. 이 점에서 DeAOT는 단순한 성능 개선을 넘어, hierarchical propagation 내부에서 어떤 정보를 어떻게 분리해서 다뤄야 하는지에 대한 중요한 설계 방향을 제시한 연구라고 볼 수 있다.
