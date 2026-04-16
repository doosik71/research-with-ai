# Associating Objects with Transformers for Video Object Segmentation

- **저자**: Zongxin Yang, Yunchao Wei, Yi Yang
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2106.02638

## 1. 논문 개요

이 논문은 semi-supervised video object segmentation(VOS), 특히 여러 개의 객체를 동시에 추적하고 분할해야 하는 multi-object scenario에서 더 정확하고 더 효율적인 방법을 제안하는 것이 목표이다. 문제 설정은 첫 프레임에서 주어진 객체 마스크를 바탕으로, 이후 전체 비디오 프레임에서 같은 객체들을 계속 찾아 분할하는 것이다.

기존의 강력한 VOS 방법들은 대부분 본질적으로 “한 번에 하나의 positive object만” 처리하도록 설계되어 있다. 그래서 비디오 안에 객체가 여러 개 있으면 각 객체를 따로 매칭하고 따로 segmentation한 뒤, 마지막에 이를 합치는 post-ensemble 방식을 사용한다. 논문은 이 구조가 두 가지 문제를 가진다고 본다. 첫째, 여러 객체 사이의 관계를 모델링하지 못해 representation learning이 비효율적이다. 둘째, 객체 수가 늘수록 메모리와 연산량이 거의 선형적으로 증가해 실제 multi-object 환경에서 비싸다.

이 문제는 중요하다. 실제 비디오에서는 한 객체만 나오는 경우보다 여러 객체가 동시에 존재하는 경우가 흔하며, 자율주행, 증강현실, 비디오 편집, 실시간 추적 시스템 같은 응용에서는 multi-object 처리 능력과 속도가 모두 중요하다. 논문은 이 점에서 “여러 객체를 한 번에 같은 feature space 안에서 다루는 방식”이 필요하다고 주장한다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 여러 객체를 서로 독립적으로 처리하지 않고, 각 객체에 고유한 identity를 부여한 뒤 동일한 embedding space 안에서 함께 처리하자는 것이다. 이를 위해 저자들은 AOT(Associating Objects with Transformers)라는 프레임워크를 제안한다.

가장 중요한 직관은 다음과 같다. 객체마다 서로 다른 identification vector를 부여하면, 네트워크는 여러 객체를 하나의 통합된 표현 공간 안에서 동시에 매칭하고 전파할 수 있다. 그러면 기존처럼 객체 수만큼 네트워크를 반복 실행할 필요가 없어진다. 즉, multi-object VOS를 single-object VOS에 가까운 비용으로 수행할 수 있다.

두 번째 핵심은 이러한 identification-based representation 위에 hierarchical transformer 구조를 쌓는 것이다. 저자들은 Long Short-Term Transformer(LSTT)를 설계하여, 현재 프레임 내부의 객체 관계, 첫 프레임 및 오래된 메모리 프레임과의 long-term association, 인접 프레임과의 short-term temporal smoothness를 함께 모델링한다.

기존 방법과의 차별점은 명확하다. 기존 memory-based 또는 matching-based VOS는 attention을 쓰더라도 보통 단일 객체 중심으로 동작했다. 반면 AOT는 다수 객체를 동일한 구조 안에서 직접 association하고, 이를 transformer 기반의 계층적 matching/propagation으로 처리한다. 논문은 이것이 정확도뿐 아니라 효율성 면에서도 큰 이점을 준다고 주장한다.

## 3. 상세 방법 설명

AOT의 전체 구조는 크게 세 부분으로 이해할 수 있다. 첫째는 encoder가 각 프레임의 시각 특징을 추출하는 단계이다. 둘째는 identification mechanism을 사용해 메모리 프레임의 multi-object mask를 identity-aware embedding으로 바꾸는 단계이다. 셋째는 LSTT가 현재 프레임과 메모리 프레임 사이에서 객체 정보를 계층적으로 매칭하고 전파한 뒤, decoder가 최종 multi-object segmentation을 출력하는 단계이다.

기존 방법의 multi-object 처리 방식은 수식으로 다음과 같이 표현된다.

$$
Y' = A(F_N(I_t, I_m, Y_m^1), \dots, F_N(I_t, I_m, Y_m^N))
$$

여기서 $F_N$은 single-object segmentation network이고, $A$는 여러 객체의 예측을 합치는 ensemble 함수이다. 즉 객체가 $N$개면 사실상 네트워크를 $N$번 돌리는 셈이다. 논문은 이 구조가 비효율적이라고 본다.

### Identification Embedding

저자들은 먼저 identity bank $D \in \mathbb{R}^{M \times C}$를 정의한다. 여기서 $M$은 사용할 수 있는 identity vector의 개수이고, 각 벡터는 $C$차원이다. 비디오에 등장하는 $N$개의 객체에 대해, 각 객체는 identity bank에서 서로 다른 identity vector 하나를 할당받는다. 단, $N < M$이어야 한다.

객체 마스크를 one-hot 형태의 $Y \in \{0,1\}^{THW \times N}$라고 할 때, 이를 identity embedding으로 바꾸는 식은 다음과 같다.

$$
E = ID(Y, D) = YPD
$$

여기서 $P \in \{0,1\}^{N \times M}$는 random permutation matrix이다. 쉽게 말해, 객체마다 identity bank에서 다른 벡터를 뽑아 붙여주는 역할을 한다. 이 과정을 거치면 각 픽셀은 “어느 객체에 속하는가”를 단순한 class index가 아니라 고차원 벡터 표현으로 갖게 된다.

이 설계의 장점은 여러 객체를 같은 feature space에 넣되, 서로 구별 가능한 표현으로 유지할 수 있다는 점이다. 논문은 학습 중 각 video sample과 optimization iteration마다 $P$를 새로 랜덤화하여, identity bank의 모든 벡터가 공평하게 학습되도록 한다고 설명한다.

### Attention 기반 Propagation

기본적인 attention matching은 다음과 같다.

$$
Att(Q, K, V) = Corr(Q, K)V = softmax\left(\frac{QK^{tr}}{\sqrt{C}}\right)V
$$

여기서 $Q$는 현재 프레임의 query embedding, $K$는 메모리 프레임의 key embedding, $V$는 메모리 프레임의 value embedding이다. 일반적인 attention과 동일하게, 현재 프레임 위치가 메모리 프레임 어디를 참고할지 계산한 뒤 그 정보를 가져온다.

AOT는 여기에 identification embedding을 더한다. 즉 value에 단순 visual feature만 넣지 않고, multi-object identity 정보까지 함께 넣는다.

$$
V' = AttID(Q, K, V, Y \mid D) = Att(Q, K, V + ID(Y, D)) = Att(Q, K, V + E)
$$

즉, 메모리 프레임의 feature $V$에 객체 identity embedding $E$를 더한 뒤 attention propagation을 수행한다. 결과적으로 현재 프레임의 각 위치는 메모리로부터 “어떤 객체 정보가 얼마나 전달되었는지”를 포함한 feature $V'$를 얻게 된다.

### Identification Decoding

전파된 feature $V'$에서 최종적으로 각 객체의 segmentation probability를 복원해야 한다. 이를 위해 decoder $F_D$는 identity bank 전체 $M$개에 대한 logit을 먼저 출력한다.

$$
Y' = softmax(PF_D(V')) = softmax(PL_D)
$$

여기서 $L_D \in \mathbb{R}^{HW \times M}$는 모든 identity에 대한 logit이고, 같은 permutation matrix $P$를 이용해 이번 sample에 실제 할당된 객체 identity만 선택한다. 그 뒤 softmax를 적용해 최종 multi-object segmentation probability $Y'$를 얻는다.

이 방식의 의미는 분명하다. 네트워크는 고정된 크기의 identity vocabulary($M$개)에 대해 예측하고, 현재 비디오에서 실제 쓰인 identity만 골라 객체 예측으로 바꾼다. 따라서 객체 수가 달라져도 구조를 바꿀 필요가 없다.

학습에는 일반적인 multi-class segmentation loss를 쓸 수 있다고 논문은 설명한다. 부록에 따르면 실제로는 bootstrapped cross-entropy loss와 soft Jaccard loss를 0.5:0.5로 결합해 사용했다.

### Long Short-Term Transformer(LSTT)

저자들은 identification만으로는 충분하지 않고, 여러 객체의 관계를 더 잘 모델링하려면 hierarchical attention 구조가 필요하다고 본다. 그래서 LSTT block을 설계한다. 각 block은 다음 순서로 구성된다.

첫째, self-attention이 현재 프레임 내부의 객체 간 관계를 학습한다.  
둘째, long-term attention이 첫 프레임과 저장된 과거 프레임들로부터 객체 정보를 가져온다.  
셋째, short-term attention이 직전 프레임 등 가까운 시점의 local temporal consistency를 반영한다.  
넷째, 2-layer feed-forward MLP가 비선형 변환을 수행한다.

논문은 long-term attention을 다음과 같이 정의한다.

$$
AttLT(X_t^l, X_m^l, Y_m) = AttID(X_t^l W_K^l, X_m^l W_K^l, X_m^l W_V^l, Y_m \mid D)
$$

여기서 $X_t^l$은 현재 시점 $t$, LSTT의 $l$번째 block 입력 feature이다. $X_m^l$은 메모리 프레임 feature들의 concat이며, $Y_m$은 그 프레임들의 mask 정보이다. 중요한 점은 현재 프레임과 메모리 프레임이 동일한 projection space에서 매칭되도록 siamese-like projection을 썼다는 것이다. 저자들은 이것이 학습 안정성에 더 좋다고 한다.

short-term attention은 인접 프레임에서 전체 프레임을 보지 않고, 각 현재 위치 $p$ 주변의 작은 spatial-temporal neighborhood만 본다.

$$
AttST(X_t^l, X_n^l, Y_n \mid p) = AttLT(X_{t,p}^l, X_{n,\mathcal{N}(p)}^l, Y_{n,\mathcal{N}(p)}^l)
$$

여기서 $\mathcal{N}(p)$는 위치 $p$를 중심으로 한 $\lambda \times \lambda$ local window이다. 즉 short-term attention은 “직전 프레임 근처에서만 찾는 지역 매칭”에 가깝다. 논문은 contiguous frame들 사이에서는 motion이 대체로 smooth하기 때문에, 이런 local attention이 더 효율적이고 효과적이라고 본다.

이 설계는 역할 분담이 분명하다. long-term attention은 멀리 떨어진 프레임에서도 객체를 다시 찾는 역할, short-term attention은 프레임 간 부드러운 변화를 안정적으로 따라가는 역할을 담당한다. 부록의 시각화에서도 short-term attention은 초기 layer부터 비교적 잘 작동하지만, long-term attention은 깊은 layer로 갈수록 점점 정확해진다고 설명한다.

### 구현 세부사항

주요 기본 설정은 MobileNet-V2 encoder와 FPN decoder이다. LSTT 입력 채널은 256, head 수는 8이다. short-term local window 크기 $\lambda$는 15, identity vector 개수 $M$은 10으로 설정했다. 이는 사용한 benchmark의 최대 객체 수와 일치한다.

모델 변형은 다음과 같다.

- `AOT-Tiny`: $L=1$, long-term memory는 첫 프레임만 사용
- `AOT-Small`: $L=2$
- `AOT-Base`: $L=3$
- `AOT-Large`: $L=3$이며 일정 간격 $\delta$로 예측 프레임을 long-term memory에 추가 저장

또한 더 강한 backbone으로 ResNet-50과 Swin-B도 실험했다.

훈련은 2단계이다. 먼저 정적 이미지 데이터셋으로부터 synthetic video를 만들어 pre-training하고, 이후 실제 VOS benchmark에서 main training한다. optimizer는 AdamW이며, sequence length는 5이다. batch size는 16, GPU는 4개의 Tesla V100을 사용했다.

부록의 중요한 보완점은 patch-wise identity bank이다. LSTT feature 해상도가 입력의 $1/16$이기 때문에, 고해상도 mask를 그대로 low-resolution identity embedding으로 내리기 어렵다. 이를 위해 16x16 patch 단위로 identity bank를 세분화하여 각 patch 내부 위치마다 sub-identity vector를 두고, patch 안에서 이들을 합산해 low-resolution embedding을 만든다. 이는 patch 내부의 형태 정보까지 일부 유지하려는 장치로 볼 수 있다.

## 4. 실험 및 결과

실험은 multi-object benchmark인 YouTube-VOS, DAVIS 2017과 single-object benchmark인 DAVIS 2016에서 수행되었다. 평가지표는 region similarity인 $J$와 boundary similarity인 $F$, 그리고 이 둘의 평균인 $J$ & $F$이다.

### YouTube-VOS 결과

YouTube-VOS 2018 validation split에서 AOT 계열은 전반적으로 매우 강한 결과를 보인다. 예를 들어:

- AOT-S: 82.6 $J$ & $F$, 27.1 FPS
- AOT-B: 83.5, 20.5 FPS
- AOT-L: 83.8, 16.0 FPS
- R50-AOT-L: 84.1, 14.9 FPS
- SwinB-AOT-L: 84.5, 9.3 FPS

비교 기준으로 CFBI+는 82.8에 4.0 FPS이다. 즉 AOT-S는 정확도가 거의 비슷하면서도 훨씬 빠르고, R50-AOT-L은 정확도까지 앞선다. 논문이 강조하는 포인트는 여기서 분명하다. multi-object accuracy를 올리면서도 speed를 크게 개선했다는 점이다.

2019 split에서도 비슷한 경향이 유지된다. R50-AOT-L이 84.1, SwinB-AOT-L이 84.5를 기록한다. 특히 AOT-T는 79.7 수준이지만 41.0 FPS로 매우 빠르다. 즉 정확도와 속도 사이에서 다양한 선택지를 제공한다.

### DAVIS 2017 결과

DAVIS 2017 validation split에서는:

- AOT-T (Y): 79.9, 51.4 FPS
- AOT-B (Y): 82.5, 29.6 FPS
- AOT-L (Y): 83.8, 18.7 FPS
- R50-AOT-L (Y): 84.9, 18.0 FPS
- SwinB-AOT-L (Y): 85.4, 12.1 FPS

기존 강한 baseline인 CFBI+ (Y)는 82.9, 5.6 FPS이므로, R50-AOT-L과 SwinB-AOT-L은 성능에서 더 좋고 속도도 더 빠르다.

DAVIS 2017 test split에서도 SwinB-AOT-L은 81.2를 기록해 표 안에서 가장 높다. R50-AOT-L도 79.6으로 강하다. 논문은 특히 “AOT에서 multi-object speed가 single-object speed와 동일하다”고 강조한다. 이는 구조상 객체 수에 따라 네트워크를 반복 실행하지 않기 때문에 가능한 결과이다.

### DAVIS 2016 결과

DAVIS 2016은 single-object benchmark이다. AOT는 multi-object를 목표로 설계되었지만 여기서도 성능이 좋다.

- AOT-B (Y): 89.9, 29.6 FPS
- AOT-L (Y): 90.4, 18.7 FPS
- R50-AOT-L (Y): 91.1, 18.0 FPS
- SwinB-AOT-L (Y): 92.0, 12.1 FPS

기존 KMN (Y)은 90.5, 8.3 FPS이고, CFBI+ (Y)는 89.9, 5.9 FPS이다. 즉 AOT는 single-object에서도 경쟁력이 있으며, 속도 면에서는 여전히 우수하다. 다만 이 경우 multi-object association의 직접적 장점은 제한적이라고 논문도 인정한다.

### 정성적 결과

정성적 비교에서 AOT는 여러 개의 매우 비슷한 객체가 동시에 있을 때 강점을 보인다. 예시로 carousel이나 zebra 같은 유사 객체 장면에서, CFBI는 객체를 혼동하기 쉽지만 AOT는 객체들을 통합적으로 association하기 때문에 더 정확히 구분한다.

반면 failure case도 제시된다. ski poles나 watch처럼 매우 작은 객체는 잘 분할하지 못하는 경우가 있다. 논문은 tiny object 처리를 위한 특별한 설계가 없기 때문이라고 명시한다.

### Ablation Study

ablation은 AOT-S를 기준으로 YouTube-VOS validation 2018 split에서 수행되었다.

identity number $M$에 대한 실험에서, 기본값인 $M=10$이 가장 좋고 $M$이 커질수록 성능이 떨어진다. 논문은 그 이유로 학습 데이터에 그렇게 많은 객체가 없다는 점, 그리고 256차원 공간에서 너무 많은 identity vector를 서로 잘 분리하기 어렵다는 점을 제시한다. 부록의 cosine similarity visualization도 $M$이 커질수록 identity bank가 덜 안정적임을 보여준다.

local window size $\lambda$는 클수록 좋다. 기본값 15에서 가장 좋고, local attention을 제거한 $\lambda=0$에서는 80.3에서 74.3으로 크게 하락한다. 이는 short-term local attention이 필수적임을 보여준다.

local frame number $n$은 직전 한 프레임만 쓰는 것이 가장 좋다. 2프레임, 3프레임으로 늘리면 오히려 성능이 약간 떨어진다. 논문은 너무 먼 이전 프레임까지 local matching에 넣으면 motion 차이가 커져 error가 늘어날 수 있다고 해석한다.

LSTT block 수 $L$를 늘리면 성능이 좋아진다. 1개 block은 77.9지만 41 FPS로 매우 빠르고, 2개 block은 80.3, 3개 block은 80.9다. 즉 AOT는 layer 수 조절만으로 accuracy-speed tradeoff를 유연하게 맞출 수 있다.

positional embedding 실험에서는 self-attention의 sine positional embedding 제거는 영향이 작지만, local attention의 relative positional embedding 제거는 성능 하락이 더 크다. 즉 인접 프레임 사이의 상대적 위치 관계가 중요하다는 뜻이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 해결책이 매우 잘 맞아떨어진다는 점이다. 기존 multi-object VOS가 사실상 single-object pipeline의 반복이라는 구조적 한계를 갖고 있었는데, AOT는 identification mechanism을 통해 이를 직접적으로 해결한다. 단순히 성능을 조금 높인 것이 아니라, multi-object VOS의 계산 구조 자체를 바꾸었다는 점에서 의미가 크다.

두 번째 강점은 효율성과 정확도를 동시에 보여준다는 점이다. 많은 논문이 속도를 높이면 성능이 떨어지고, 성능을 높이면 속도가 느려진다. 그러나 AOT는 작은 모델인 AOT-T/AOT-S에서 real-time에 가까운 속도를 유지하고, 큰 모델에서는 state-of-the-art 성능을 달성한다. 같은 프레임워크 안에서 여러 operating point를 제공하는 점도 실용적이다.

세 번째 강점은 LSTT의 설계가 직관적이면서도 실험적으로 설득력이 있다는 점이다. self-attention, long-term attention, short-term attention을 분리해 각 역할을 명확히 했고, ablation과 attention visualization으로 각 모듈의 필요성을 뒷받침했다.

반면 한계도 분명하다. 첫째, identity 수 $M$은 benchmark의 최대 객체 수에 맞춰 10으로 설정되어 있으며, $N < M$이라는 가정을 둔다. 따라서 매우 많은 객체가 동시에 등장하는 상황으로의 확장성은 논문에서 충분히 검증되지 않았다. 둘째, 작은 객체 처리 성능은 약하다. 저자들도 tiny objects에 대한 특수 설계가 없다고 인정한다. 셋째, identity bank가 잘 작동하는 이유와 한계는 일부 시각화로 설명되지만, 더 일반적인 이론적 분석은 제공되지 않는다.

또 하나의 실질적 한계는 학습 설정이 다소 복잡하다는 점이다. synthetic video pre-training, benchmark별 main training, patch-wise identity bank, memory scheduling 등 여러 요소가 들어간다. 각 요소의 상대적 기여도를 완전히 분리해서 보여주지는 않는다. 예를 들어 patch-wise identity bank가 전체 성능에 미치는 독립적 영향은 본문에서 정량 ablation으로 자세히 제시되지 않았다.

비판적으로 보면, AOT의 핵심 주장은 매우 강하지만, “객체 수가 single-object와 같은 효율”이라는 표현은 구조적 관점에서는 맞아도 실제로는 backbone 크기, memory size, resolution, long-term memory 저장 전략 등에 따라 절대 연산량이 달라질 수 있다. 즉 객체 수 증가에 따른 반복 실행 문제를 없앤 것은 분명하지만, 모든 환경에서 완전히 같은 비용이라고 단순화해서 받아들이면 안 된다. 그래도 논문이 제시한 표의 FPS 결과는 적어도 기존 multi-object post-ensemble 방식보다 훨씬 효율적이라는 점을 충분히 보여준다.

## 6. 결론

이 논문은 multi-object VOS를 위해 AOT라는 새로운 프레임워크를 제안했다. 핵심 기여는 두 가지다. 첫째, identification mechanism을 통해 여러 객체를 하나의 embedding space 안에서 동시에 association, matching, decoding할 수 있게 했다. 둘째, LSTT를 통해 self, long-term, short-term attention을 계층적으로 결합하여 multi-object propagation을 효과적으로 수행했다.

실험적으로 AOT는 YouTube-VOS, DAVIS 2017, DAVIS 2016에서 매우 강한 성능을 보였고, 특히 multi-object 상황에서 기존 방법보다 훨씬 빠르면서도 더 높은 정확도를 달성했다. 작은 모델은 real-time에 가까운 속도를 유지하고, 큰 모델은 state-of-the-art 성능을 보여 practical range가 넓다.

향후 연구 측면에서도 의미가 크다. 논문이 직접 언급하듯, 이 identification mechanism은 video instance segmentation, interactive VOS, multi-object tracking 같은 관련 문제에도 확장 가능성이 있다. 또한 LSTT는 video representation을 다루는 transformer 기반 계층 구조의 하나의 강한 baseline으로 기능할 수 있다. 요약하면, 이 연구는 단순한 성능 개선을 넘어 multi-object VOS의 계산 패러다임을 보다 직접적이고 효율적인 방향으로 재구성한 작업이라고 볼 수 있다.
