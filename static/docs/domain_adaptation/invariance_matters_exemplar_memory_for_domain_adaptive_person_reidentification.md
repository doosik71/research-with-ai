# Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification

* **저자**: Zhun Zhong, Liang Zheng, Zhiming Luo, Shaozi Li, Yi Yang
* **발표연도**: 2019
* **arXiv**: [https://arxiv.org/abs/1904.01990](https://arxiv.org/abs/1904.01990)

## 1. 논문 개요

이 논문은 **domain adaptive person re-identification (re-ID)** 문제를 다룬다. 구체적으로는, **라벨이 있는 source domain**과 **라벨이 없는 target domain**이 주어졌을 때, target domain에서도 잘 동작하는 person re-ID 모델을 학습하는 것이 목표다. person re-ID는 서로 다른 카메라에서 촬영된 사람 이미지들 사이에서 같은 사람을 찾아내는 문제이기 때문에, 카메라 스타일, 자세, 배경, 조명 변화 등에 매우 민감하다. 이 때문에 한 데이터셋에서 잘 학습된 모델이 다른 데이터셋으로 옮겨가면 성능이 크게 떨어지는 경우가 많다.

기존의 unsupervised domain adaptation 방법들은 주로 **source와 target 사이의 분포 차이(inter-domain gap)**를 줄이는 데 집중했다. 하지만 이 논문은 그런 접근만으로는 충분하지 않다고 본다. 이유는 target domain 내부에도 성능을 크게 흔드는 다양한 변화, 즉 **intra-domain variations**가 존재하기 때문이다. 예를 들어 같은 사람이라도 카메라가 달라지면 전혀 다른 외관으로 보일 수 있고, 반대로 다른 사람인데도 배경이나 자세 때문에 비슷하게 보일 수 있다. 저자들은 이러한 target 내부 변화가 실제 테스트 성능에 직접적인 영향을 준다고 보고, 단순한 분포 정렬이 아니라 **target 내부 구조를 반영하는 representation learning**이 필요하다고 주장한다.

이 논문의 핵심 문제의식은 다음과 같다. person re-ID의 UDA는 일반적인 closed-set UDA와 다르게, source와 target이 같은 클래스를 공유하지 않는다. 즉 source의 사람 ID와 target의 사람 ID는 완전히 다르다. 따라서 일반적인 도메인 정렬처럼 두 도메인의 feature distribution을 직접 맞추는 방식은 적절하지 않을 수 있다. 오히려 target에서 **보지 못한 새로운 identity들을 서로 잘 분리하면서도**, 같은 identity에 해당할 가능성이 있는 샘플들끼리는 가깝게 학습되도록 해야 한다. 이 점에서 이 논문은 person re-ID의 UDA를 **open-set에 가까운 더 어려운 문제**로 해석하고 접근한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 target domain의 unlabeled data에 대해, 단순히 source와 target을 맞추는 것이 아니라 **세 가지 invariance**를 학습시키는 것이다. 저자들은 이를 통해 target domain 내부 변화에 강건한 표현을 만들고자 한다. 세 가지는 다음과 같다.

첫째는 **exemplar-invariance**이다. 각 target 이미지 인스턴스를 하나의 독립적인 exemplar로 보고, 해당 샘플은 자기 자신과는 가깝고 다른 샘플들과는 멀어지도록 학습한다. 이는 semantic identity를 직접 알 수 없는 상황에서라도, 모델이 각 인스턴스의 외관적 특징을 더 뚜렷하게 구분하도록 유도한다.

둘째는 **camera-invariance**이다. target domain에서 카메라 스타일 변화는 매우 중요한 변동 요인이다. 저자들은 CamStyle transfer를 이용해 한 이미지의 카메라 스타일만 바꾼 synthetic image들을 만들고, 원본과 스타일 변환본이 같은 identity를 유지한다고 가정한다. 따라서 이들을 가깝게 학습시키면 카메라 변화에 덜 민감한 표현을 얻을 수 있다.

셋째는 **neighborhood-invariance**이다. source와 target으로 어느 정도 학습된 모델이 있다면, target에서 어떤 샘플의 nearest neighbors는 같은 identity일 가능성이 있다. 물론 완전히 보장되지는 않지만, 일정 수준 신뢰 가능한 이웃들을 positive-like signal로 활용하면 pose, viewpoint, background 변화에도 강한 표현을 학습할 수 있다.

이 세 가지 invariance를 동시에 구현하기 위해 저자들이 제안한 장치가 **exemplar memory**이다. 이것은 target 데이터 전체의 최신 feature를 저장하는 메모리 구조다. 핵심 장점은 mini-batch 내부 샘플만 보는 것이 아니라, **전체 target set과의 관계를 기준으로** similarity를 계산하고 학습할 수 있다는 점이다. 즉, global context를 사용해 invariance learning을 수행하면서도 계산량 증가는 제한적으로 억제한다.

기존 접근과의 차별점은 명확하다. 이전 연구들은 image-level adaptation, attribute-level alignment, 또는 camera-invariance만 부분적으로 고려했다. 반면 이 논문은 **target 내부의 variation을 세 종류로 구조화**하고, 이를 하나의 메모리 기반 프레임워크 안에서 통합적으로 최적화한다. 특히 exemplar-invariance와 neighborhood-invariance는 서로 상충할 수도 있는데, 저자들은 이 둘을 camera-invariance와 함께 균형 있게 묶어 더 강한 target representation을 만들고자 한다.

## 3. 상세 방법 설명

전체 프레임워크는 크게 두 갈래로 구성된다. 하나는 **source domain supervised learning**, 다른 하나는 **target domain invariance learning**이다. backbone은 ImageNet으로 사전학습된 **ResNet-50**이며, Pooling-5 뒤에 4096차원의 fully convolutional layer를 붙인다. 그 뒤에 batch normalization, ReLU, dropout을 적용하고, 최종적으로 두 개의 학습 컴포넌트로 분기한다.

첫 번째 컴포넌트는 source용 분류 모듈이다. source에는 identity label이 있으므로 일반적인 classification으로 학습한다. FC-#id 층과 softmax를 통해 source identity를 예측하고, cross-entropy loss를 사용한다. 식은 다음과 같다.

$$
\mathcal{L}_{src}=-\frac{1}{n_s}\sum_{i=1}^{n_s}\log p(y_{s,i}\mid x_{s,i})
$$

여기서 $n_s$는 source mini-batch 크기이고, $p(y_{s,i}\mid x_{s,i})$는 source 이미지 $x_{s,i}$가 정답 identity $y_{s,i}$에 속할 확률이다. 이 항은 모델이 person re-ID에 필요한 기본적인 discrimination 능력을 유지하도록 돕는다.

두 번째 컴포넌트가 이 논문의 핵심인 **exemplar memory module**이다. target에는 라벨이 없기 때문에, 저자들은 각 target image instance를 하나의 고유 클래스처럼 다룬다. exemplar memory는 key-value 구조이며, key memory $\mathcal{K}$에는 각 target 이미지의 최신 feature를 저장하고, value memory $\mathcal{V}$에는 해당 샘플의 인덱스를 저장한다. 즉, target 이미지가 $N_t$개라면 메모리 슬롯도 $N_t$개 존재한다.

학습 도중 target 이미지 $x_{t,i}$가 네트워크를 통과하면, FC-4096의 L2-normalized output $f(x_{t,i})$를 얻게 된다. 이후 메모리의 해당 슬롯은 다음과 같이 update된다.

$$
\mathcal{K}[i]\leftarrow \alpha \mathcal{K}[i] + (1-\alpha)f(x_{t,i})
$$

업데이트 후 다시 L2 normalization을 수행한다. 여기서 $\alpha \in [0,1]$는 memory update rate다. 이 설계는 메모리 feature를 완전히 새 feature로 치환하지 않고, 과거 정보와 현재 정보를 섞어 안정적으로 추적하게 해 준다.

이제 이 메모리를 사용해 target에 대한 세 가지 invariance를 정의한다.

### 3.1 Exemplar-invariance

각 target 이미지를 자기 자신만의 클래스로 분류한다. 이미지 $x_{t,i}$에 대해, 현재 feature $f(x_{t,i})$와 메모리에 저장된 모든 target feature 사이의 cosine similarity를 구한 뒤 softmax를 적용한다.

$$
p(i\mid x_{t,i})=
\frac{\exp(\mathcal{K}[i]^T f(x_{t,i})/\beta)}
{\sum_{j=1}^{N_t}\exp(\mathcal{K}[j]^T f(x_{t,i})/\beta)}
$$

여기서 $\beta$는 temperature factor다. $\beta$가 작을수록 분포가 sharper해지고, 더 강한 구분을 유도한다. 이때 exemplar-invariance loss는 다음과 같다.

$$
\mathcal{L}_{ei}=-\log p(i\mid x_{t,i})
$$

이 손실은 각 인스턴스가 자기 메모리 슬롯과 강하게 매칭되고 다른 샘플들과는 구분되도록 학습시킨다. 쉽게 말해, “각 이미지가 자기 자신을 알아보게” 만든다.

### 3.2 Camera-invariance

target domain에서는 camera ID를 알 수 있다고 가정한다. 저자들은 각 카메라를 하나의 style domain으로 보고, **StarGAN**을 이용해 camera-style transfer model을 학습한다. 이후 한 실제 target 이미지로부터 다른 카메라 스타일을 가진 synthetic image들을 생성한다. 원본 이미지 $x_{t,i}$와 스타일 변환 이미지 $\hat{x}_{t,i}$는 identity는 같고 style만 다르다고 본다.

camera-invariance는 이 synthetic image가 원본과 같은 exemplar class에 속하도록 분류한다.

$$
\mathcal{L}_{ci}=-\log p(i\mid \hat{x}_{t,i})
$$

이 항은 카메라 변화에도 representation이 유지되도록 만든다. 즉, 같은 사람을 서로 다른 카메라에서 촬영했을 때 발생하는 appearance shift를 줄이는 역할이다.

### 3.3 Neighborhood-invariance

가장 흥미로운 부분은 neighborhood-invariance다. 저자들은 메모리에 저장된 feature를 기준으로, 각 target 이미지의 $k$-nearest neighbors를 찾는다. 이 neighbor 집합을 $\mathcal{M}(x_{t,i},k)$라고 하자. 이웃은 같은 identity일 가능성이 높지만, 완전히 확실하지는 않다. 그래서 hard label이 아니라 **soft-label loss**를 쓴다.

neighbor class에 대한 weight는 다음처럼 정의한다.

$$
w_{i,j} =
\begin{cases}
\frac{1}{k}, & j \neq i \\
1, & j = i
\end{cases} \;, \quad
\forall j \in \mathcal{M}(x_{t,i},k)
$$

그리고 neighborhood-invariance loss는 다음과 같다.

$$
\mathcal{L}_{ni}=-\sum_{j\neq i} w_{i,j}\log p(j\mid x_{t,i}),
\quad \forall j \in \mathcal{M}(x_{t,i},k)
$$

여기서 중요한 점은, 저자들이 식 설명에서 **exemplar-invariance와 neighborhood-invariance를 구분하기 위해** neighborhood loss에서는 자기 자신의 class를 positive target으로 직접 쓰지 않는다고 명시했다는 것이다. 즉, 자기 자신과의 일치는 exemplar/camera 쪽이 담당하고, neighborhood는 주변 샘플들과의 근접성만 반영한다.

직관적으로 보면 exemplar-invariance는 샘플을 잘 흩어 놓는 힘이고, neighborhood-invariance는 샘플을 다시 모아 주는 힘이다. 전자는 다른 identity를 분리하는 데 좋지만 같은 identity도 갈라놓을 위험이 있고, 후자는 같은 identity를 모으는 데 좋지만 잘못된 neighbor를 끌어들일 위험이 있다. 이 논문은 이 두 힘의 균형이 중요하다고 본다.

### 3.4 최종 target loss와 전체 loss

세 가지 invariance를 통합한 target loss는 다음과 같다.

$$
\mathcal{L}_{tgt} = -\frac{1}{n_t}\sum_{i=1}^{n_t}\sum_j w_{i,j}\log p(j\mid x^*_{t,i})
$$

여기서 $x^*_{t,i}$는 원본 target 이미지와 그 camera-style transferred 이미지들의 합집합에서 무작위로 뽑은 샘플이다. 따라서 $i=j$일 때는 exemplar-invariance와 camera-invariance가 반영되고, $i\neq j$일 때는 neighborhood-invariance가 반영된다.

최종 전체 loss는 source loss와 target loss를 가중합한 형태다.

$$
\mathcal{L}=(1-\lambda)\mathcal{L}_{src}+\lambda \mathcal{L}_{tgt}
$$

$\lambda \in [0,1]$는 source supervision과 target invariance learning의 균형을 조절한다. $\lambda=0$이면 source-only baseline이고, $\lambda=1$이면 source 없이 target invariance만으로 학습하는 셈이다.

### 3.5 학습 절차와 구현 세부

논문에 따르면 backbone은 ResNet-50이고, 처음 두 residual layers는 GPU 메모리를 아끼기 위해 고정한다. 입력 해상도는 $256\times128$이며, random flipping, random cropping, random erasing을 쓴다. dropout은 0.5, optimizer는 SGD다. 초기 40 epoch 동안 base layer learning rate는 0.01, 나머지는 0.1을 사용하고, 이후 20 epoch 동안 learning rate를 10분의 1로 줄인다. source와 target 모두 batch size는 128이다.

중요한 하이퍼파라미터는 $\beta=0.05$, $k=6$, $\lambda=0.3$이다. 또한 메모리 업데이트율 $\alpha$는 초기 0.01에서 epoch에 따라 선형 증가시키며, neighborhood-invariance는 처음 5 epoch 동안은 쓰지 않고, **먼저 exemplar-invariance와 camera-invariance로 모델을 어느 정도 안정화한 뒤** 이후 epoch부터 추가한다. 이것은 noisy neighbor를 초기에 섣불리 쓰지 않기 위한 실용적인 설계로 읽힌다.

## 4. 실험 및 결과

실험은 **Market-1501**, **DukeMTMC-reID**, **MSMT17** 세 데이터셋에서 수행되며, 평가 지표는 **CMC**와 **mAP**다. person re-ID에서는 일반적으로 rank-1 accuracy와 mAP가 가장 핵심적인 숫자이므로, 이 논문도 그 둘을 중심으로 비교한다.

### 4.1 하이퍼파라미터 분석

먼저 temperature $\beta$를 분석한 결과, 너무 작으면 모델이 수렴하지 않았고, 적당히 작은 값일 때 성능이 좋았다. Table 1에서 $\beta=0.05$가 가장 좋은 결과를 보인다. 예를 들어 Duke $\rightarrow$ Market-1501에서는 rank-1 75.1%, mAP 43.0%, Market-1501 $\rightarrow$ Duke에서는 rank-1 63.3%, mAP 40.4%를 기록한다. 이는 similarity-based softmax에서 temperature가 분포 sharpness를 결정하며, 너무 큰 값은 discrimination을 약하게 만들고, 너무 작은 값은 optimization을 불안정하게 만든다는 전형적인 현상과 맞닿아 있다.

$\lambda$에 대한 분석에서는, target invariance learning을 조금이라도 넣는 순간 baseline보다 성능이 크게 좋아진다. 논문은 특히 $\lambda=1$이어도 baseline보다 좋다고 언급하는데, 이것은 source supervision 없이도 target 내부 structure만으로 상당한 이득을 얻는다는 뜻이다. 다만 가장 좋은 범위는 대략 0.3~0.8로 보고한다. 이는 source에서 얻은 기본 discrimination 능력과 target 내부 적응이 서로 보완적임을 보여준다.

$k$에 대한 분석에서는, neighborhood-invariance를 도입하는 순간 성능이 좋아지고, $k$가 6~8 사이에서 가장 좋다. 너무 큰 $k$는 false positive neighbor를 많이 포함하므로 성능이 떨어진다. 이것은 neighborhood-based pseudo-positive 전략이 가지는 전형적인 trade-off를 잘 보여준다.

### 4.2 Baseline과 invariance 구성요소별 ablation

Table 2는 논문의 핵심 결과 중 하나다. 먼저 supervised target upper-bound를 보면, target label이 있을 때는 Market-1501에서 rank-1 87.6%, mAP 69.4, DukeMTMC-reID에서 rank-1 75.6%, mAP 57.8을 얻는다. 하지만 source-only baseline은 Market에서 rank-1 43.1%, mAP 17.7, Duke에서 rank-1 28.9%, mAP 14.8로 급락한다. 이 차이는 도메인 갭이 얼마나 큰지를 잘 보여준다.

그 다음 각 invariance를 붙였을 때 성능 향상을 보면 다음과 같다.

**Ours w/ E**는 exemplar-invariance만 추가한 경우다. Market에서 rank-1이 43.1%에서 48.7%로, Duke에서는 28.9%에서 34.2%로 오른다. 즉 exemplar-invariance만으로도 target representation의 discrimination이 개선된다.

**Ours w/ E+C**는 camera-invariance까지 넣은 경우다. Market에서 rank-1 63.1%, mAP 28.4, Duke에서 rank-1 53.9%, mAP 29.7을 기록한다. 특히 Market 방향에서 rank-1이 E 대비 14.4%p 상승한다. 이는 카메라 스타일 변화가 re-ID 성능에 매우 큰 영향을 미치며, 이를 명시적으로 다루는 것이 매우 효과적임을 시사한다.

**Ours w/ E+N**은 neighborhood-invariance를 exemplar와 결합한 경우다. Market에서는 rank-1 58.0%, mAP 27.7, Duke에서는 rank-1 39.7%, mAP 23.6이다. camera-invariance만큼의 폭발적 상승은 아니지만, E 대비 꾸준한 개선을 보여준다.

최종적으로 **Ours w/ E+C+N**, 즉 ECN은 Market에서 rank-1 75.1%, mAP 43.0, Duke에서 rank-1 63.3%, mAP 40.4를 기록한다. 이 수치는 구성 요소들이 단순히 각각 조금씩 도움이 되는 수준이 아니라, **함께 사용할 때 상호보완적으로 더 큰 효과를 낸다**는 것을 보여준다. 특히 neighborhood-invariance는 더 강한 기반 표현 위에서 작동할수록 더 reliable한 positive sample을 찾게 되므로, E+C 위에 붙었을 때 더 큰 이득을 준다고 논문은 해석한다.

### 4.3 Exemplar memory의 효과

Table 3은 memory 기반 구현과 mini-batch 기반 구현을 비교한다. mini-batch 방식은 target sample, CamStyle sample, 그리고 $k$-NN sample을 배치 안에 넣어 학습한다. 하지만 이렇게 하면 global target structure를 충분히 반영하기 어렵다. 실제 결과도 이를 보여준다. mini-batch 방식은 rank-1 67.2를 기록한 반면, memory 방식은 75.1을 기록한다. 그런데 추가 비용은 training time 약 1.3분 증가, GPU memory 약 260MB 증가에 불과하다.

이 결과는 exemplar memory가 이 논문의 실질적 핵심임을 보여준다. 단순히 invariance 아이디어만이 아니라, 그것을 **전체 target set 수준에서 효율적으로 계산하는 메커니즘**이 성능 향상의 중요한 원인이라는 뜻이다.

### 4.4 SOTA 비교

Table 4에서 저자들은 기존 unsupervised 또는 UDA re-ID 방법들과 비교한다. 비교 대상에는 LOMO, BoW 같은 hand-crafted feature 방법, CAMEL, UMDL, PUL 같은 unsupervised transfer 방법, 그리고 PTGAN, SPGAN, MMFA, TJ-AIDL, CamStyle, HHL 같은 UDA 방법이 포함된다.

가장 인상적인 비교는 당시 강력한 baseline인 **HHL**과의 비교다. ECN은 Duke $\rightarrow$ Market-1501에서 rank-1 75.1%, mAP 43.0을 얻는데, HHL은 rank-1 62.2%, mAP 31.4다. 즉 rank-1 기준 **12.9%p 상승**이다. 반대로 Market-1501 $\rightarrow$ Duke에서도 ECN은 rank-1 63.3%, mAP 40.4, HHL은 rank-1 46.9%, mAP 27.2이므로 rank-1 기준 **16.4%p 상승**이다. 이는 단순한 미세 개선이 아니라 상당히 큰 폭의 성능 향상이다.

MSMT17에서도 ECN은 PTGAN보다 크게 앞선다. Market source에서는 rank-1 25.3%, mAP 8.5, Duke source에서는 rank-1 30.2%, mAP 10.2를 기록한다. PTGAN 대비 Duke source 기준으로 rank-1이 18.4%p 높고 mAP도 6.9%p 높다. MSMT17은 더 크고 어려운 데이터셋이므로, 이 결과는 방법의 확장 가능성을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정에 대한 해석이 매우 적절하다는 점이다. 저자들은 person re-ID의 UDA가 일반적인 closed-set domain adaptation과 다르며, target 내부의 구조와 variation을 명시적으로 다뤄야 한다는 점을 설득력 있게 제시한다. 단순한 domain alignment 대신 target domain의 학습 신호를 **invariance**라는 형태로 구조화한 점은 개념적으로 깔끔하다.

둘째 강점은 세 가지 invariance의 역할이 서로 분명하다는 점이다. exemplar-invariance는 instance discrimination을 강화하고, camera-invariance는 camera-induced variation을 줄이며, neighborhood-invariance는 잠재적 positive pair를 끌어들인다. 각각이 어떤 문제를 해결하는지 논문 내 설명과 ablation 결과가 잘 맞아떨어진다.

셋째는 exemplar memory라는 구현 장치의 실용성이다. 전체 target set을 기준으로 similarity를 계산할 수 있으면서도 시간과 메모리 오버헤드가 크지 않다. 단순히 성능이 좋은 것뿐 아니라, 왜 mini-batch보다 better한지에 대한 실험적 근거도 제시했다.

넷째는 실험 결과의 설득력이다. 세 데이터셋 모두에서 이전 방법들을 큰 폭으로 앞서고, ablation도 꽤 체계적으로 수행했다. 특히 “왜 이 요소가 필요한가”를 각 구성 요소별 성능 차이로 보여주는 방식이 명확하다.

하지만 한계도 분명하다. 가장 큰 한계는 **neighborhood-invariance의 가정이 완전히 안전하지 않다**는 것이다. nearest neighbor가 같은 identity일 가능성은 높지만 보장되지는 않는다. 논문도 이를 인정하고 있으며, 그래서 $k$가 너무 크면 false positive가 늘어나 성능이 떨어진다고 보고한다. 즉 이 방법은 target feature quality가 어느 정도 확보되어야 잘 작동한다.

또 다른 한계는 **camera-invariance가 camera ID와 CamStyle generation 품질에 의존**한다는 점이다. 논문은 camera ID는 쉽게 얻을 수 있다고 가정하지만, 실제 환경에 따라 이 가정은 약해질 수 있다. 또한 style transfer 모델이 생성한 이미지가 항상 identity-preserving인지도 완전히 보장되지는 않는다. 이 부분은 논문 텍스트 기준으로 qualitative failure case나 생성 품질 분석이 자세히 제시되지는 않았다.

또한 exemplar-invariance는 본질적으로 같은 identity 샘플들까지 서로 밀어낼 수 있다. 논문은 이를 neighborhood-invariance로 보완하려 하지만, 결국 두 힘 사이의 균형은 하이퍼파라미터와 학습 단계 설계에 의존한다. 이 설계가 다른 환경에서도 항상 안정적으로 유지되는지는 본문만으로는 완전히 판단하기 어렵다.

비판적으로 보면, 이 논문은 “distribution alignment” 계열보다 “target structure mining” 계열에 더 가깝다. 그 자체로 매우 강력하지만, source-target 간 전역적 정렬을 거의 명시적으로 다루지 않기 때문에, source initialization의 질이 너무 낮거나 target이 지나치게 복잡할 경우 어느 정도까지 견딜 수 있는지는 추가 검증이 필요해 보인다. 다만 제공된 텍스트에는 이런 실패 조건에 대한 정교한 분석은 명시되어 있지 않다.

## 6. 결론

이 논문은 person re-ID를 위한 unsupervised domain adaptation에서, 기존처럼 source-target 분포 차이만 줄이는 접근이 아니라 **target domain 내부의 variation을 직접 다루는 것이 중요하다**는 점을 강하게 보여준다. 이를 위해 exemplar-invariance, camera-invariance, neighborhood-invariance라는 세 가지 학습 원리를 제안했고, 이를 효율적으로 구현하기 위한 **exemplar memory**를 도입했다.

방법론적으로는 매우 명확하다. source의 supervised classification으로 기본적인 person discrimination 능력을 유지하면서, target에서는 memory를 통해 전체 데이터 수준의 similarity를 활용해 invariance learning을 수행한다. 수식적으로도 target loss가 각 invariance를 하나의 soft classification 틀 안에서 통합하고 있어 일관성이 있다.

실험적으로는 당시 기준 SOTA를 큰 폭으로 넘는 결과를 보였고, 세 가지 invariance가 각각 유효하며 함께 사용할 때 가장 강하다는 점을 설득력 있게 입증했다. 특히 camera variation과 neighborhood structure를 함께 고려한 것이 핵심 성공 요인으로 보인다.

실제 적용 측면에서 보면, 이 연구는 라벨 없는 target surveillance 환경으로 re-ID 모델을 옮겨야 하는 상황에 매우 실용적인 방향을 제시한다. 향후 연구에서는 더 신뢰도 높은 neighbor selection, 더 강건한 synthetic view/camera augmentation, 혹은 memory를 활용한 self-paced pseudo labeling과 결합하는 방향으로 확장될 가능성이 크다. 요약하면, 이 논문은 UDA re-ID에서 **target domain 내부 구조를 활용하는 memory-based learning**의 중요성을 잘 보여준 대표적 연구라고 평가할 수 있다.
