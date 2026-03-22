# Pixelwise Instance Segmentation with a Dynamically Instantiated Network

* **저자**: Anurag Arnab and Philip H.S Torr
* **발표연도**: 2017
* **arXiv**: [https://arxiv.org/abs/1704.02386](https://arxiv.org/abs/1704.02386)

## 1. 논문 개요

이 논문은 instance segmentation을 “픽셀 단위의 semantic segmentation”과 “instance 구분”을 동시에 수행하는 문제로 다룬다. 즉, 각 픽셀에 대해 단순히 클래스만 붙이는 것이 아니라, 같은 클래스 안에서도 어느 객체 인스턴스에 속하는지까지 함께 예측하는 시스템을 제안한다.

당시 많은 instance segmentation 방법은 object detector를 먼저 돌려 bounding box를 얻고, 각 box 내부를 따로 segmentation하는 구조였다. 이런 방식은 탐지 결과에 크게 의존하고, proposal들을 독립적으로 처리하기 때문에 이미지 전체의 일관성을 충분히 반영하지 못한다는 문제가 있다. 예를 들어 false positive detection이 있으면 이를 회복하기 어렵고, bounding box가 객체 전체를 덮지 못하면 segmentation 품질이 제한된다. 또한 proposal들이 서로 겹치는 상황에서 어느 픽셀이 최종적으로 어느 인스턴스에 속하는지를 일관되게 정하기도 쉽지 않다.

이 논문은 이러한 한계를 해결하기 위해, 먼저 semantic segmentation을 수행한 뒤 그 결과와 object detector의 출력을 함께 사용하여 instance를 구분하는 구조를 제안한다. 핵심은 semantic segmentation을 출발점으로 삼고, 그 위에 동적으로 생성되는 instance-level CRF를 붙여 이미지마다 달라지는 개수의 인스턴스를 자연스럽게 다룬다는 점이다. 이 방식은 proposal을 하나씩 독립적으로 다루는 대신 이미지 전체를 한 번에 고려하므로, occlusion 처리와 전역적 일관성 측면에서 유리하다.

문제의 중요성도 분명하다. instance segmentation은 semantic segmentation보다 더 세밀한 장면 이해를 제공하며, object detection보다 훨씬 정밀한 공간적 경계를 제공한다. 자율주행, 로보틱스, 이미지 편집처럼 객체를 “정확히 어디까지” 인식해야 하는 응용에서는 특히 중요하다. 논문은 이 문제를 단순히 detection의 후처리 문제로 보지 않고, semantic segmentation과 구조적 추론을 결합한 픽셀 단위 예측 문제로 재정의했다는 점에서 의미가 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 instance segmentation을 detector 기반 proposal refinement 문제로 다루지 않고, semantic segmentation의 확장 문제로 본다는 데 있다. 저자들의 관점에서는 instance segmentation이란 “각 픽셀의 클래스”뿐 아니라 “각 픽셀이 어떤 객체 인스턴스에 속하는지”를 추가로 예측하는 문제이다. 따라서 먼저 강한 semantic segmentation을 얻고, 이를 바탕으로 instance identity를 추론하는 것이 더 자연스럽다는 주장이다.

이를 위해 네트워크는 두 단계로 구성된다. 첫 번째는 semantic segmentation subnetwork이고, 두 번째는 instance segmentation subnetwork이다. 전자는 고정된 클래스 집합에 대해 픽셀별 확률 맵을 생성한다. 후자는 이 semantic segmentation 결과와 object detector의 bounding box, detection score, 그리고 shape prior를 결합하여 각 픽셀이 어떤 detection 인스턴스에 속할지를 예측한다. 이때 최종 결정은 CRF에서 이뤄진다.

기존 접근과의 가장 큰 차별점은 세 가지로 요약할 수 있다.

첫째, 이미지 전체를 한 번에 본다. 많은 prior work는 proposal이나 detection별로 독립적으로 mask를 예측한 뒤 후처리로 합친다. 반면 이 논문은 처음부터 픽셀 전체를 대상으로 instance label을 할당한다. 그래서 한 픽셀이 여러 인스턴스에 동시에 속하는 모순이 발생하지 않는다.

둘째, detection을 “절대적인 시작점”이 아니라 “하나의 단서”로만 사용한다. bounding box는 instance를 구분하는 데 도움을 주지만, semantic segmentation 결과와 결합되어 사용되므로 false positive detection이나 localization error에 더 강하다.

셋째, 동적으로 instantiated되는 CRF를 사용한다. 이미지마다 detection 수가 다르므로 instance label의 개수도 달라져야 한다. 저자들은 이 가변 길이 출력 공간을 직접 다루는 구조를 제안했고, mean field inference를 RNN처럼 unroll하여 end-to-end 학습 가능하게 만들었다.

## 3. 상세 방법 설명

전체 파이프라인은 다음과 같다. 입력 이미지를 먼저 semantic segmentation subnetwork에 넣어, 각 픽셀 $i$에 대해 클래스 $l \in L$의 확률 $Q_i(l)$를 예측한다. 여기서 $L$은 클래스 집합이며, background를 포함한다. 이후 object detector가 제공하는 detection 집합 $(l_i, s_i, B_i)$를 사용한다. 각 detection은 클래스 라벨 $l_i$, confidence score $s_i$, 그리고 bounding box 내부 픽셀 집합 $B_i$로 구성된다. 이후 instance segmentation subnetwork는 semantic 결과 $Q$와 detection 정보를 바탕으로 각 픽셀에 대해 “어느 detection instance에 속하는지”를 예측한다.

중요한 점은 출력 차원이 이미지마다 달라진다는 것이다. semantic segmentation의 출력은 $W \times H \times (K+1)$처럼 고정된 차원을 가지지만, instance segmentation의 출력은 detection 개수를 $D$라고 할 때 $W \times H \times (D+1)$이다. 여기서 0은 background이고, 1부터 $D$까지는 각 detection 인스턴스를 뜻한다. 즉, 같은 네트워크가 이미지별로 다른 수의 instance label을 처리해야 한다.

논문은 이 문제를 CRF로 정식화한다. 각 픽셀 $i$에 대해 multinomial random variable $V_i$를 두고, $V_i \in {0,1,\dots,D}$가 되도록 한다. 전체 라벨링 $V$에 대한 에너지는 다음과 같다.

$$
E(V=v) = \sum_i U(v_i) + \sum_{i<j} P(v_i, v_j)
$$

여기서 $U(v_i)$는 unary energy이고, $P(v_i,v_j)$는 pairwise energy이다. unary는 각 픽셀이 특정 인스턴스에 속할 가능성을 나타내고, pairwise는 비슷한 픽셀들이 같은 인스턴스에 속하도록 유도한다.

이 논문에서 unary energy는 세 개의 항을 결합해 정의된다.

$$
U(v_i) = - \ln \big[w_1 \psi_{\text{Box}}(v_i) + w_2 \psi_{\text{Global}}(v_i) + w_3 \psi_{\text{Shape}}(v_i)\big]
$$

여기서 $w_1, w_2, w_3$는 학습되는 가중치이며, 세 항은 각각 다른 종류의 instance 단서를 반영한다.

### Box Term

첫 번째 unary term은 Box term이다. 직관은 단순하다. 어떤 픽셀이 detection box 안에 있고, semantic segmentation이 그 픽셀을 detection class로 높게 예측한다면, 그 픽셀은 해당 detection 인스턴스일 가능성이 높다. 수식은 다음과 같다.

$$
\psi_{\text{Box}}(V_i = k) =
\begin{cases}
Q_i(l_k) s_k & \text{if } i \in B_k \
0 & \text{otherwise}
\end{cases}
$$

즉, 픽셀이 $k$번째 detection box 안에 있을 때만 값이 있고, 그 값은 semantic confidence $Q_i(l_k)$와 detection score $s_k$의 곱이다. 이 항은 detection localization이 적절하고 semantic segmentation이 좋을 때 매우 효과적이다. 또한 false positive detection이라 하더라도 semantic segmentation이 해당 영역을 그 클래스라고 보지 않으면 값이 낮아지므로, detector의 실수를 어느 정도 무시할 수 있다.

### Global Term

두 번째 unary term은 Global term이다. 이 항은 bounding box에 의존하지 않는다. 어떤 클래스의 detection이 이미지에 존재한다면, semantic segmentation이 그 클래스라고 보는 모든 픽셀은 그 detection 인스턴스일 가능성이 있다고 본다. 수식은 매우 단순하다.

$$
\psi_{\text{Global}}(V_i = k) = Q_i(l_k)
$$

이 항의 역할은 detector의 box가 객체 전체를 덮지 못하는 경우를 보완하는 것이다. Box term은 box 바깥 픽셀에 대해 0을 주기 때문에, 잘린 bounding box 바깥에 있는 객체 부분은 해당 인스턴스로 복원하기 어렵다. 반면 Global term은 이미지 전체에 대해 semantic confidence를 전달하므로, box 밖 픽셀도 같은 인스턴스로 이어질 가능성을 남긴다. 저자들은 이 항이 특히 높은 IoU threshold에서 성능을 크게 높였다고 분석한다. 그 이유는 더 정확한 경계 복원이 가능해지기 때문이다.

또 하나의 중요한 설명은 학습 안정성이다. Global term이 있으면 출력이 이미지 전체의 semantic prediction에 의존하게 되므로, 더 많은 픽셀에서 gradient가 전파된다. 저자들은 이것이 batch 간 gradient를 더 안정적으로 만들어 backpropagation에 유리하다고 본다.

### Shape Term

세 번째 unary term은 Shape term이다. 이것은 occlusion 상황, 특히 appearance가 비슷한 같은 클래스 객체들이 겹칠 때 도움이 되도록 설계되었다. 예를 들어 비슷한 색과 질감을 가진 양 여러 마리가 겹친 경우, 단순 appearance 기반 pairwise term만으로는 잘 분리되지 않을 수 있다. 이때 “이 클래스의 객체는 보통 이런 모양을 가진다”는 shape prior가 도움이 된다.

논문은 여러 shape template 집합 $T$를 미리 준비하고, 각 detection box 크기에 맞게 bilinear interpolation으로 warp한다. 그런 다음 해당 detection class의 semantic segmentation 맵 $Q_{B_k}(l_k)$와 가장 잘 맞는 template를 normalized cross-correlation로 고른다.

$$
t^* = \arg\max_{t \in \tilde{T}}
\frac{\sum Q_{B_k}(l_k) \odot t}{|Q_{B_k}(l_k)| , |t|}
$$

여기서 $\odot$는 elementwise product이다. 이후 선택된 template $t^*$와 semantic segmentation unaries를 곱해 shape-based potential을 만든다.

$$
\psi(V_{B_k}=k) = Q_{B_k}(l_k) \odot t^*
$$

직관적으로 보면, semantic segmentation이 “이 위치가 해당 클래스일 가능성”을 말해주고, shape prior가 “그 클래스의 객체라면 이런 형태일 것”을 추가로 말해준다. 둘의 곱은 detection box 안에서 어느 부분이 foreground instance인지 더 정교하게 가려내는 역할을 한다.

저자들은 shape prior를 학습 가능한 파라미터처럼 다룰 수도 있다고 설명한다. 실제 구현에서는 기존 연구의 shape priors를 초기값으로 사용했으며, 대략 다섯 개 aspect ratio에 대해 각 클래스당 약 250개의 template를 사용했다. 다만 본 논문에서는 한 detection에 대해 하나의 template만 매칭한다. 저자들은 미래에는 여러 template를 매칭해 object part 수준까지 확장할 수 있다고 언급한다.

### Pairwise Term

pairwise term은 fully connected CRF의 Gaussian pairwise potentials를 사용한다. 자세한 수식은 본문에서 완전히 전개하지는 않았지만, 핵심은 spatially 가까우면서 appearance가 비슷한 픽셀들이 같은 instance label을 갖도록 유도하는 것이다. 이는 semantic segmentation에서 흔히 쓰이는 dense CRF의 철학과 같다.

이 항은 특히 같은 클래스의 여러 객체가 가려져 있으나 외관 차이가 있는 경우에 유용하다. 예를 들어 사람 여러 명이 겹쳐 있어도 옷 색이 다르면, pairwise term이 경계 분리에 도움을 준다. 반대로 appearance가 매우 유사한 경우에는 Shape term의 도움이 더 중요해진다.

### 추론과 Dynamic CRF

이 CRF의 MAP 추론은 mean field inference로 근사한다. 확률 분포는 다음과 같은 Gibbs distribution으로 주어진다.

$$
P(V=v) = \frac{1}{Z}\exp(-E(v))
$$

여기서 $Z$는 정규화 상수이다. mean field inference는 반복적 업데이트 알고리즘이며, 저자들은 이를 RNN처럼 unroll하여 네트워크 내부 레이어로 포함시킨다. 이 방식은 CRF inference 자체를 미분 가능하게 만들어 end-to-end 학습을 가능하게 한다.

중요한 특징은 이 CRF가 동적으로 instantiated된다는 점이다. detection 개수 $D$가 이미지마다 다르므로 label 수 자체가 다르다. 따라서 클래스별 고정 가중치를 쓰는 방식은 사용할 수 없다. 저자들은 class-specific weight 대신 공유된 가중치를 사용한다. 이는 variable-length input을 처리하게 해줄 뿐 아니라, instance label 자체가 특정 고정 의미를 갖지 않는다는 문제 설정과도 잘 맞는다.

### Loss Function

instance segmentation에서 어려운 점 하나는 label permutation이다. 예를 들어 사람 두 명이 있을 때, 예측에서 첫 번째 사람을 label 1, 두 번째 사람을 label 2로 주든 반대로 주든 결과는 사실상 동일하다. 그런데 일반적인 cross-entropy는 라벨 번호가 다르면 다른 예측으로 간주하므로 그대로는 사용할 수 없다.

이 논문은 먼저 예측 결과와 ground truth instance를 IoU 기준으로 matching한 뒤, ground truth의 label ordering을 예측에 맞게 재정렬한다. 그 재정렬된 ground truth를 $G^*$라고 두고 다음과 같이 정의한다.

$$
G^* = \arg\max_{m \in M} \text{IoU}(m, P)
$$

여기서 $M$은 ground truth의 모든 permutation 집합이고, $P$는 네트워크 예측이다. 실제 계산에서는 모든 permutation을 직접 시도하지 않고, bipartite matching 문제로 바꾸어 효율적으로 푼다. 두 인스턴스가 같은 semantic class를 가질 때만 IoU를 edge weight로 두고, 나머지는 0으로 둔다.

이렇게 matching된 ground truth를 얻은 뒤에는 표준 cross-entropy loss를 사용한다. 저자들은 approximate IoU loss보다 cross-entropy가 더 잘 작동했다고 보고한다.

### 학습 절차

학습은 두 단계이다. 먼저 semantic segmentation network를 cross-entropy loss로 pretraining한다. 이때 FCN8s 기반 semantic subnetwork와 CRF-RNN 구조를 end-to-end로 학습한다. 이후 instance segmentation subnetwork를 붙이고, instance annotation으로 fine-tuning한다. 이 fine-tuning 단계에서는 instance loss만 사용하지만, gradient는 semantic segmentation subnetwork까지 거슬러 올라가 전체 파라미터를 함께 업데이트한다.

논문에 따르면 semantic segmentation pretraining에서는 learning rate를 $10^{-8}$, momentum을 0.9, batch size를 20으로 사용한다. 이후 instance segmentation fine-tuning에서는 learning rate를 $10^{-12}$로 낮추고 batch size를 1로 둔다. 또한 gradient clipping을 적용해 $\ell_2$ norm이 $10^9$를 넘지 않도록 했다. 저자들은 이 구조 안에 두 개의 CRF-RNN이 들어가므로 학습 안정성 확보를 위해 clipping이 중요했다고 설명한다.

## 4. 실험 및 결과

실험은 Pascal VOC 2012 Validation Set, Semantic Boundaries Dataset(SBD), 그리고 Cityscapes에서 수행되었다. VOC는 경계가 정교하게 라벨링되어 픽셀 수준 평가에 적합한 데이터셋이며, SBD는 VOC에 비해 더 많은 이미지가 있으나 annotation이 상대적으로 거칠다. Cityscapes는 도심 주행 장면에서 많은 객체 인스턴스가 등장하는 데이터셋이다.

평가는 주로 $AP^r$를 사용한다. 이는 object detection의 AP와 유사하지만 bounding box 대신 region mask 간 IoU를 사용한다. 저자들은 $AP^r$를 다양한 IoU threshold에서 측정하며, 0.1부터 0.9까지 평균한 $AP^r_{vol}$도 함께 보고한다. 또한 proposal-based 평가가 아닌 “전체 segmentation map의 일관성”을 보기 위해 Matching IoU도 제안해 사용한다. 이 지표는 예측 segmentation map과 ground truth map을 instance matching한 뒤 전체 IoU를 측정하는 방식이다.

### Ablation Study

VOC validation에서 unary potentials와 end-to-end training의 효과를 분석한 표가 제시된다. 결과를 보면 Box term만 사용했을 때보다 Global term과 Shape term을 추가했을 때 전반적인 성능이 향상된다. 특히 Global term은 높은 IoU threshold에서 큰 효과를 보였는데, 이는 박스 밖까지 객체를 복원하는 데 도움을 주기 때문이다. Shape term은 occlusion 상황에서 가려진 인스턴스를 되찾는 데 기여하여 주로 낮은 threshold에서 개선을 준다고 해석된다.

또한 piecewise training보다 end-to-end training이 일관되게 더 좋다. 예를 들어 Box+Global+Shape 조합에서는 piecewise 대비 end-to-end가 $AP^r_{vol}$과 Matching IoU 모두에서 향상된다. 저자들은 특히 Global term이 있을 때 semantic segmentation 전 영역에서 gradient가 흐르기 때문에 학습 효율이 좋아진다고 설명한다.

### VOC 2012 Validation Set 결과

VOC validation set에서 제안 방법은 당시 강력한 방법들과 비교해 매우 경쟁력 있는 결과를 보인다. 특히 높은 IoU threshold에서 우수하다. 논문에 따르면 제안 방법의 성능은 다음과 같다.

* $AP^r@0.5 = 61.7$
* $AP^r@0.6 = 55.5$
* $AP^r@0.7 = 48.6$
* $AP^r@0.8 = 39.5$
* $AP^r@0.9 = 25.1$
* $AP^r_{vol} = 57.5$

비교 대상 중 MPA 3-scale은 $AP^r@0.5$에서는 62.1로 약간 높지만, 높은 threshold로 갈수록 제안 방법이 더 우수하다. 특히 $AP^r@0.9$에서 기존 최고 18.5 대비 25.1을 기록해 크게 앞선다. 논문은 이것을 “더 정밀한 segmentation boundary”를 생성하기 때문으로 해석한다. 상대 개선폭도 크다. 저자들은 0.9 threshold에서 이전 SOTA 대비 6.6%p, 상대적으로 36% 향상이라고 설명한다.

흥미로운 점은 proposal-based method들이 낮은 threshold에서는 다소 유리할 수 있다는 해석이다. proposal을 많이 내는 방식은 대략적인 겹침만 맞아도 되는 낮은 threshold에서 점수를 얻기 쉽다. 반면 이 논문의 방식은 이미지 전체에 대해 coherent map을 생성하는 방향에 더 가깝기 때문에 높은 threshold에서 강점을 보인다.

또한 런타임도 언급된다. MPA가 같은 Titan X GPU에서 이미지당 8.7초가 걸리는 반면, 제안 방법은 약 1.5초라고 한다. 즉, 정밀도뿐 아니라 실용적 속도 측면에서도 장점이 있다.

### SBD 결과

SBD에서도 제안 방법은 높은 threshold에서 강한 결과를 보인다. 보고된 주요 수치는 다음과 같다.

* piecewise: $AP^r@0.5 = 59.1$, $AP^r@0.7 = 42.1$, $AP^r_{vol} = 52.3$, Matching IoU = 41.8
* end-to-end: $AP^r@0.5 = 62.0$, $AP^r@0.7 = 44.8$, $AP^r_{vol} = 55.4$, Matching IoU = 47.3

특히 $AP^r@0.7$에서 이전 강한 방법인 IIS 계열을 앞선다. 또한 Matching IoU에서도 MNC보다 8.3%p 개선했다고 보고한다. 이는 제안 방법이 전체 segmentation map의 품질 면에서 특히 강하다는 주장을 뒷받침한다.

논문은 SBD의 라벨이 VOC보다 거칠어 매우 높은 threshold 평가에는 덜 적합할 수 있다고 해석한다. 실제로 SBD에서의 $AP^r@0.9$는 VOC보다 많이 낮다. 이는 모델의 한계라기보다 annotation granularity와 학습 데이터 질의 차이도 영향을 준다는 설명이다.

### Semantic Segmentation 성능 향상

이 논문의 흥미로운 결과 중 하나는 instance segmentation으로 fine-tuning했더니 semantic segmentation 성능도 좋아졌다는 점이다. VOC에서는 mean IoU가 74.2에서 75.1로, SBD에서는 71.5에서 72.5로 향상된다. 저자들은 instance segmentation이 semantic segmentation의 더 세밀한 형태이므로, 관련 task로 fine-tuning하면 semantic understanding도 좋아진다고 해석한다. 이는 두 task의 상호보완성을 보여준다.

### Cityscapes 결과

Cityscapes test set에서는 semantic segmentation backbone으로 ResNet-101 기반 구조를 사용하고, 모든 unary term을 활용했다. 제안 방법은 다음과 같은 결과를 보고한다.

* AP = 20.0
* AP at 0.5 = 38.8
* AP 100m = 32.6
* AP 50m = 37.6

이는 당시 비교된 SAIS, DWT, InstanceCut, Graph Decomp., Pixel Encoding 등의 방법보다 높은 수치이다. 저자들은 이를 새로운 state-of-the-art라고 주장한다. Cityscapes처럼 인스턴스 수가 많고, 서로 가려지고, 비연속적으로 보이는 객체가 많은 환경에서 이 방식의 전역 추론 능력이 특히 유리하다는 점을 강조한다.

### 정성적 결과와 실패 사례

성공 사례로는 false positive detection을 무시하거나, detector box가 객체를 완전히 감싸지 못해도 semantic signal과 Global term 덕분에 box 밖 픽셀까지 올바르게 인스턴스에 할당하는 장면이 제시된다. 또한 일부 강한 occlusion 상황에서도 Shape term이 도움을 준다.

실패 사례는 주로 detector가 객체를 놓친 경우에 집중된다. 이 방법은 detection을 절대적인 결과가 아니라 단서로 사용하지만, 그래도 instance label 후보는 detection 개수에 의해 정해진다. 따라서 detector가 어떤 객체를 전혀 검출하지 못하면, semantic segmentation이 그 객체를 알아봐도 최종 instance segmentation에 포함되지 못할 수 있다. 또한 appearance가 매우 유사한 가축 무리처럼 복잡한 occlusion에서는 Shape term만으로 충분하지 않은 경우도 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 instance segmentation을 proposal refinement가 아니라 “semantic segmentation + structured instance reasoning”으로 재설계했다는 점이다. 이 관점 전환 덕분에 기존 detector-centered pipeline의 구조적 약점을 피해 간다. 결과적으로 높은 IoU threshold에서 매우 강한 성능을 보이며, 정밀한 mask 품질을 증명했다. 실제 수치에서도 VOC와 SBD에서 높은 threshold 성능이 확실히 좋다.

또 다른 강점은 전역적 일관성이다. proposal-based 방법은 여러 proposal이 서로 충돌하거나 동일 픽셀이 여러 인스턴스에 속하는 상황이 발생할 수 있는데, 이 논문의 방법은 처음부터 pixelwise label assignment 문제로 정의되므로 이런 모순을 피할 수 있다. “이미지 전체를 한 번에 reasoning한다”는 설계가 잘 살아 있다.

세 번째 강점은 end-to-end differentiability이다. semantic segmentation 모듈, instance CRF, mean field inference가 모두 연결되어 있어 최종 instance loss가 앞단 semantic representation까지 개선한다. 실제로 semantic segmentation 성능도 함께 좋아졌다는 실험은 이 설계의 타당성을 뒷받침한다.

네 번째 강점은 detector error에 대한 상대적 강건성이다. false positive detection을 semantic segmentation이 걸러줄 수 있고, localization이 부정확한 detection도 Global term이 보완한다. detector에 완전히 종속되는 구조보다 훨씬 유연하다.

반면 한계도 분명하다. 가장 본질적인 한계는 detection에 여전히 의존한다는 점이다. 이 방법이 detector를 단서로만 사용한다 해도, 가능한 instance label의 집합 자체는 detection 수에 의해 정해진다. 따라서 detector가 놓친 객체는 final instance map에서 복원하기 어렵다. 논문 속 실패 사례도 이 문제가 반복적으로 나타난다.

둘째, Shape term은 흥미롭지만 hand-crafted prior의 성격이 남아 있다. shape template를 클러스터링해 미리 준비하고, detection box에 맞춰 warping하는 방식은 완전히 data-driven representation learning과는 거리가 있다. 저자들도 향후 multiple template matching이나 part-based extension 가능성을 언급하지만, 본 논문에서는 제한적인 형태다.

셋째, object detector 자체는 end-to-end 공동학습되지 않는다. 저자들은 이것이 학습 안정성과 failure mode 분리 측면에서 장점이라고 보지만, 반대로 보면 최종 task에 맞게 detector를 최적화하지 못한다는 뜻이기도 하다. 논문 말미에서 detector까지 포함한 joint training을 future work로 제안한 이유도 여기에 있다.

넷째, 수식 수준에서 pairwise term의 세부 정의나 mean field update 식이 본문에 자세히 전개되지는 않는다. 기반 아이디어는 기존 dense CRF와 CRF-RNN을 따른다고 이해되지만, 이 논문만으로 모든 구현 세부를 완전히 복원하기는 약간 어렵다. 특히 “어떤 Gaussian kernels를 정확히 썼는지”, “iteration 수는 어떻게 설정했는지” 같은 정보는 제공된 텍스트에서 충분히 확인되지 않는다.

비판적으로 보면, 이 논문은 semantic segmentation과 detector를 잘 결합한 구조적 모델이라는 점에서 매우 설득력 있지만, 이후의 완전한 single-stage 혹은 stronger end-to-end instance segmentation 계열과 비교하면 detector 의존성과 shape prior 수작업 요소가 남아 있는 과도기적 성격도 있다. 그럼에도 불구하고 proposal 중심이던 흐름과 다른 방향을 명확히 제시했고, 높은 정밀 경계 예측이라는 목표를 실험적으로 강하게 증명했다는 점에서 학술적 가치가 높다.

## 6. 결론

이 논문은 instance segmentation을 픽셀 단위 문제로 다루면서, semantic segmentation subnetwork와 dynamically instantiated instance CRF를 결합한 end-to-end 시스템을 제안했다. 핵심 기여는 semantic segmentation을 출발점으로 삼아 detection 정보, shape prior, dense pairwise consistency를 결합하고, 이미지마다 달라지는 인스턴스 수를 처리할 수 있는 동적 구조를 설계했다는 데 있다.

실험적으로는 VOC, SBD, Cityscapes에서 강력한 성능을 보였고, 특히 높은 IoU threshold에서 뚜렷한 우위를 보여 더 정교한 mask를 생성한다는 점을 입증했다. 또한 instance segmentation으로 fine-tuning했을 때 semantic segmentation도 향상된다는 결과는 두 과제가 긴밀히 연결되어 있음을 잘 보여준다.

실제 적용 측면에서는 자율주행, 로보틱스, 장면 이해처럼 픽셀 수준의 정확한 인스턴스 분리가 필요한 문제에 의미가 크다. 향후 detector까지 함께 joint training하는 방향으로 확장된다면, semantic segmentation, object detection, instance segmentation을 통합하는 multi-task scene understanding 시스템으로 발전할 가능성이 있다. 제공된 텍스트 기준으로 보았을 때, 이 논문은 detection 기반 instance segmentation의 한계를 정면으로 짚고, semantic segmentation 중심의 대안을 제시한 중요한 연구로 평가할 수 있다.
