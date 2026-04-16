# Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation

- **저자**: Youngmin Oh, Beomjun Kim, Bumsub Ham
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2104.00905

## 1. 논문 개요

이 논문은 bounding box annotation만 사용하여 semantic segmentation 모델을 학습하는 weakly-supervised semantic segmentation (WSSS) 문제를 다룬다. 핵심 문제는 bounding box가 객체의 대략적인 위치는 알려주지만, 정확한 object boundary는 주지 않는다는 점이다. 따라서 box 내부에는 foreground와 background가 섞여 있고, 이를 그대로 사용하면 segmentation network가 잘못된 pseudo label에 크게 영향을 받는다.

논문은 이 문제를 두 단계로 나누어 접근한다. 첫째, bounding box로부터 가능한 한 좋은 pseudo segmentation label을 만드는 문제다. 둘째, 그렇게 만든 pseudo label이 여전히 noisy하다는 전제에서, segmentation network를 noise에 덜 민감하게 학습시키는 문제다. 저자들은 이 두 문제를 각각 `Background-Aware Pooling (BAP)`과 `Noise-Aware Loss (NAL)`로 해결한다.

이 문제는 실제적으로 중요하다. pixel-level annotation은 매우 비싸고 시간이 많이 들지만, bounding box는 훨씬 저렴하다. 논문에서도 box annotation이 pixel-wise segmentation annotation보다 약 15배 저렴하다고 언급한다. 따라서 bounding box만으로 segmentation 품질을 충분히 끌어올릴 수 있다면, 대규모 데이터셋 구축과 실제 응용에 큰 의미가 있다.

## 2. 핵심 아이디어

논문의 중심 직관은 "한 이미지 안에서 background는 적어도 일부 영역에서 perceptually consistent하다"는 것이다. 다시 말해, bounding box 바깥의 확실한 background 영역을 이용하면, bounding box 내부에서도 어떤 픽셀이 background와 유사한지 추정할 수 있다는 발상이다. 이 아이디어를 retrieval 관점으로 구현한 것이 BAP이다.

기존의 CAM 기반 WSSS는 보통 global average pooling (GAP)을 사용한다. 그런데 GAP는 박스 내부 foreground와 background를 구분하지 않고 평균내기 때문에, CAM이 객체 전체보다는 가장 discriminative한 부분만 강조하는 경향이 있다. 예를 들어 person class에서 얼굴만 강하게 뜨고 몸통이나 다리는 약하게 뜰 수 있다. 논문은 BAP를 통해 foreground 가능성이 높은 feature를 더 많이 모으고, background 가능성이 높은 feature는 덜 반영하게 만든다. 그 결과 CAM이 객체 전체 영역을 더 잘 덮게 된다.

또 하나의 핵심 아이디어는 pseudo label이 완벽하지 않다는 사실을 손실 함수 수준에서 직접 다루는 것이다. 저자들은 CRF 기반 pseudo label과 retrieval 기반 pseudo label을 함께 만들고, 두 label이 일치하는 영역은 비교적 신뢰하고, 불일치하는 영역은 feature와 classifier weight 사이의 유사도를 이용해 confidence를 추정한 뒤 가중치를 둔 cross-entropy로 학습한다. 즉, noisy label을 완전히 버리지도 않고, 무조건 믿지도 않는다.

기존 bounding-box 기반 WSSS 방법들이 GrabCut, MCG 같은 off-the-shelf segmentation 도구에 크게 의존했던 것과 달리, 이 논문은 classification network 자체에서 pseudo label을 생성하는 방향을 택한다. 이 점이 방법론적으로도 일관되고, 실험적으로도 좋은 결과를 보인다.

## 3. 상세 방법 설명

전체 파이프라인은 세 단계다. 먼저 BAP를 사용해 image classification network를 학습한다. 다음으로, 이 classification network로부터 pseudo segmentation label을 만든다. 마지막으로, 그 pseudo label을 이용해 DeepLab segmentation network를 NAL과 함께 학습한다.

### 3.1 BAP를 이용한 image classification

분류 네트워크는 feature extractor와 $(L+1)$-way softmax classifier로 구성된다. 여기서 $L$은 object class 수이고, 나머지 1개는 background class다. 입력 이미지를 feature map $f$로 변환한 뒤, bounding box 바깥의 확실한 background 영역을 먼저 정의한다. 이 영역을 나타내는 binary mask를 $M$이라 하며, 어떤 위치 $p$가 모든 bounding box 밖이면 $M(p)=1$, 아니면 $M(p)=0$이다.

저자들은 feature map을 $N \times N$ grid로 나누고, 각 grid cell에서 definite background에 해당하는 feature만 평균내어 background query를 만든다. 수식은 다음과 같다.

$$
q_j = \frac{\sum_{p \in G(j)} M(p) f(p)}{\sum_{p \in G(j)} M(p)}
$$

여기서 $q_j$는 $j$번째 grid cell에서 얻은 background query다. 중요한 점은 이 query가 이미지마다 adaptive하게 정해진다는 것이다.

그 다음, 각 query와 bounding box 내부 pixel feature 사이의 cosine similarity를 계산해 background attention map을 만든다. 전체 attention은 여러 query의 평균이다.

$$
A(p) = \frac{1}{J}\sum_j A_j(p)
$$

그리고 각 query에 대한 attention은 다음과 같다.

$$
A_j(p) =
\begin{cases}
\text{ReLU}\left( \frac{f(p)}{\|f(p)\|} \cdot \frac{q_j}{\|q_j\|} \right), & p \in B \\
1, & p \notin B
\end{cases}
$$

즉, bounding box 내부에서 어떤 pixel feature가 background query와 매우 비슷하면 그 픽셀은 background일 가능성이 높다. 반대로 background query와 덜 비슷하면 foreground일 가능성이 높다.

이 attention map을 이용해 foreground feature를 weighted average pooling으로 집계한다.

$$
r_i = \frac{\sum_{p \in B_i} (1-A(p)) f(p)}{\sum_{p \in B_i} (1-A(p))}
$$

여기서 $1-A(p)$는 foreground일 확률처럼 작동한다. 만약 $A(p)=0$이면 일반적인 GAP와 동일해진다. 즉, BAP는 GAP의 일반화된 형태라고 볼 수 있다.

분류 학습은 foreground feature $r_i$와 background query $q_j$를 모두 softmax classifier에 넣고 standard cross-entropy loss로 수행한다. 이렇게 하면 네트워크가 foreground와 background를 더 잘 구분하는 feature를 배우게 되고, 결과적으로 더 좋은 CAM을 얻는다.

### 3.2 Pseudo label 생성

논문은 pseudo label을 두 방식으로 만든다. 첫 번째는 CAM과 DenseCRF를 이용한 $Y_{crf}$이고, 두 번째는 feature retrieval을 이용한 $Y_{ret}$이다. 둘은 서로 보완적이라고 주장한다.

#### 1) CAM + DenseCRF 기반 pseudo label

각 class $c$에 대한 CAM은 classifier weight $w_c$를 사용해 다음과 같이 계산한다.

$$
CAM_c(p) = \text{ReLU}(f(p) \cdot w_c)
$$

이를 box 내부에서 정규화한 unary term으로 사용한다.

$$
u_c(p) =
\begin{cases}
\frac{CAM_c(p)}{\max_p CAM_c(p)}, & p \in B_c \\
0, & p \notin B_c
\end{cases}
$$

background에 대해서는 background CAM을 쓰지 않고, 앞서 구한 background attention map 자체를 unary term으로 쓴다.

$$
u_0(p) = A(p)
$$

저자들은 background CAM은 dataset에서 자주 나타나는 배경 패턴만 강조하는 경향이 있어, foreground/background 구분에 덜 적합하다고 설명한다. 반면 $A(p)$는 이미지별로 adaptive하다.

이 unary term들과 이미지의 color, position을 활용한 pairwise term을 DenseCRF에 넣어 pseudo segmentation label $Y_{crf}$를 얻는다. DenseCRF는 low-level cue에 강하므로 boundary를 비교적 잘 정리해 준다.

#### 2) Retrieval 기반 pseudo label

하지만 CRF는 color나 texture 같은 low-level feature에 의존하므로 잘못된 label을 만들 수도 있다. 이를 보완하기 위해 classification network의 high-level feature를 사용한다.

먼저 $Y_{crf}$에서 class $c$로 분류된 모든 위치의 feature 평균을 class prototype으로 만든다.

$$
q_c = \frac{1}{|Q_c|}\sum_{p \in Q_c} f(p)
$$

여기서 $Q_c$는 $Y_{crf}$에서 class $c$로 표시된 pixel 집합이다. 이후 각 픽셀 feature와 class prototype의 cosine similarity를 계산한다.

$$
C_c(p) = \frac{f(p)}{\|f(p)\|} \cdot \frac{q_c}{\|q_c\|}
$$

마지막으로 모든 class에 대해 argmax를 취해 retrieval 기반 pseudo label $Y_{ret}$을 만든다. 이 방식은 high-level semantic consistency를 반영하므로, CRF가 놓치거나 잘못 판단한 부분을 보완할 수 있다.

### 3.3 NAL을 이용한 semantic segmentation 학습

최종 segmentation network로는 DeepLab을 사용한다. pseudo label은 $Y_{crf}$와 $Y_{ret}$ 두 개가 있다. 두 label이 같은 곳은 상대적으로 신뢰할 수 있는 영역 $S$, 다르면 불확실한 영역 $\tilde{S}$로 나눈다.

먼저 일치 영역 $S$에서는 일반적인 cross-entropy를 사용한다.

$$
L_{ce} = - \frac{1}{\sum_c |S_c|} \sum_c \sum_{p \in S_c} \log H_c(p)
$$

여기서 $H_c(p)$는 segmentation network가 픽셀 $p$에 대해 class $c$일 확률이다.

불일치 영역 $\tilde{S}$는 완전히 버리지 않는다. 저자들은 classifier weight $W_c$가 feature space에서 class center 역할을 한다고 가정하고, penultimate feature $\phi(p)$와 classifier weight 사이의 cosine similarity로 confidence를 구한다.

$$
D_c(p) = 1 + \left( \frac{\phi(p)}{\|\phi(p)\|} \cdot \frac{W_c}{\|W_c\|} \right)
$$

양수로 만들기 위해 1을 더했다. 그 다음, $Y_{crf}$가 준 label $c^*$에 대해 confidence를 다음처럼 정의한다.

$$
\sigma(p) = \left( \frac{D_{c^*}(p)}{\max_c D_c(p)} \right)^\gamma
$$

이 식의 의미는 단순하다. $Y_{crf}$가 준 class와 네트워크 feature가 가장 가깝다고 판단한 class가 비슷하면 confidence가 높다. 반대로 둘이 많이 다르면 confidence가 낮다. $\gamma$는 damping parameter이며, 클수록 confidence를 더 hard하게 만든다.

이 confidence를 weight로 써서 불일치 영역의 weighted cross-entropy를 계산한다.

$$
L_{wce} = - \frac{1}{\sum_c \sum_{p \in \tilde{S}_c} \sigma(p)}
\sum_c \sum_{p \in \tilde{S}_c} \sigma(p)\log H_c(p)
$$

최종 loss는 다음과 같다.

$$
L = L_{ce} + \lambda L_{wce}
$$

즉, 신뢰 높은 라벨은 더 강하게 학습하고, 의심스러운 라벨은 약하게 반영한다. 이것이 NAL의 핵심이다.

## 4. 실험 및 결과

### 4.1 데이터셋과 설정

semantic segmentation 실험은 PASCAL VOC 2012를 사용했다. train/val/test는 각각 1,464 / 1,449 / 1,456장이고, 추가로 10,582장의 augmented training set을 사용했다. 평가지표는 mIoU다.

instance segmentation 실험은 MS-COCO를 사용했다. train/val/test는 115K / 5K / 20K 샘플이고, 평가지표는 AP다.

classification network는 ImageNet pre-trained VGG-16 기반의 AffinityNet 구조를 사용했고, segmentation network는 DeepLab-V1(VGG-16), DeepLab-V2(ResNet-101)를 사용했다. 논문 본문에서 주요 hyperparameter는 $N=4$로 classification 학습, pseudo label 생성 시에는 $N=1$이 더 좋았다고 보고한다. NAL에서는 $\gamma=7$, $\lambda=0.1$을 사용했다.

### 4.2 Pseudo label 품질

PASCAL VOC에서 pseudo label 자체의 품질을 비교한 Table 1이 매우 중요하다. 단순히 bounding box 전체를 foreground로 보는 baseline도 train 65.4, val 62.2 mIoU를 기록해 box supervision 자체가 강한 신호임을 보여준다. GrabCut과 MCG는 대략 66 수준, WSSL은 val 71.1까지 올라간다.

하지만 저자들의 접근은 더 강하다. GAP 기반 CAM + CRF만 써도 train 75.5, val 76.1로 WSSL보다 훨씬 높다. 여기에 BAP를 넣은 `BAP: Y_crf`는 train 78.7, val 79.2로 추가 개선을 보인다. 이는 BAP가 실제로 CAM 품질을 끌어올린다는 직접적인 증거다.

retrieval 기반 pseudo label인 `BAP: Y_ret`는 단독으로는 더 낮은 성능(train 70.8, val 69.9)을 보이는데, 저자들은 feature map 해상도가 낮기 때문이라고 설명한다. 하지만 이 label은 보조적 역할을 하며 NAL에서 중요한 의미를 가진다.

Supplement에서는 GAP와 BAP의 CAM precision도 직접 비교한다. mean precision이 GAP 73.0에서 BAP 77.9로 올라가고, 특히 bike, boat, plant 등에서 큰 차이가 난다. classification accuracy 역시 80.3%에서 82.3%로 향상되었다고 보고한다. 이 결과는 BAP가 단순히 segmentation용 trick이 아니라 classification representation 자체를 개선한다는 점을 시사한다.

### 4.3 NAL의 효과

Table 3은 noisy region $\tilde{S}$를 어떻게 다룰지 비교한다. baseline은 불일치 영역을 완전히 무시하는 방식으로 61.8 / 67.5 (CRF 전/후) mIoU를 기록한다. entropy regularization은 오히려 약간 나쁘고, bootstrapping은 거의 차이가 없다. 반면 제안한 $L_{wce}$를 쓰면 62.4 / 68.1로 가장 좋다.

즉, 불확실한 pseudo label을 무조건 버리기보다 confidence-weighted하게 활용하는 것이 실제로 효과적이라는 뜻이다. Figure 5에서 confidence map이 학습이 진행되며 noisy stripe 영역에는 낮은 confidence를, dog의 다리처럼 상대적으로 맞는 영역에는 높은 confidence를 주는 예시를 보여준다. 이 시각화는 NAL의 동작 논리를 비교적 잘 뒷받침한다.

### 4.4 최종 segmentation 성능

PASCAL VOC에서 DeepLab-V1 기준 weak supervision만 사용한 결과를 보면, 기존 방법 BoxSup 62.0/64.6, SDI 65.7/67.5, BCM 66.8(val only) 수준인데, 저자들은 `Ours w/ NAL`로 val 68.1, test 69.4를 달성한다. box supervision만으로 당시 state of the art를 넘는다.

semi-supervised 설정에서도 성능이 강하다. boxes 9K + masks 1K 설정에서 `Ours w/ NAL`은 val 70.5, test 71.5로 다른 방법들보다 높다.

DeepLab-V2를 쓰면 더 올라간다. weak supervision에서 `Ours w/ NAL`은 val 74.6, test 76.1을 기록한다. Box2Seg가 val 76.4로 더 높지만, 논문은 Box2Seg가 UPerNet, FPN, PPM, 추가 decoder를 쓰는 더 강한 구조를 사용하므로 직접 비교가 완전히 공정하지 않다고 지적한다. 같은 DeepLab 계열 기준에서는 상당히 경쟁력 있는 결과다.

Supplement의 Table 6에서는 fully-supervised 성능 대비 비율도 제시한다. DeepLab-V2에서 ImageNet initialization 기준으로 저자 방법은 fully-supervised의 96.6%, MS-COCO pretrain 기준 96.3%까지 도달한다. 이는 weak supervision임에도 fully-supervised와의 격차가 꽤 좁다는 의미다.

### 4.5 MS-COCO와 일반화

이 논문은 pseudo label generator가 unseen class에도 어느 정도 generalize된다고 주장한다. VOC로 학습한 generator를 COCO에 적용한 `VOC-to-COCO` 실험에서 pseudo label AP가 `Y_crf` 기준 11.7, `Y_ret` 기준 9.0으로 완벽하진 않지만 의미 있는 결과를 낸다. 같은 COCO로 학습한 `COCO-to-COCO`는 더 좋은 성능을 보인다.

instance segmentation에서는 Mask-RCNN을 pseudo label로 학습했다. 본문 Table 6에 따르면 `Ours (VOC-to-COCO)`는 AP 16.9, `Ours (COCO-to-COCO)`는 AP 22.2다. 이는 image-level label 기반 AISI의 AP 13.7보다 좋다. 다만 이 부분에서는 NAL을 그대로 적용하지 못했고, binary cross-entropy 구조 때문에 일치 영역 $S$만 사용했다고 명시한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제를 매우 직접적으로 푼다는 점이다. bounding box supervision의 핵심 어려움은 box 내부 foreground/background 혼합과 pseudo label noise인데, BAP는 전자를, NAL은 후자를 각각 명확하게 겨냥한다. 설계 논리가 분명하고, 수식도 간결하며, 각 구성요소가 왜 필요한지 실험으로 비교적 잘 보여준다.

두 번째 강점은 off-the-shelf segmentation 기법 의존도를 줄였다는 점이다. 기존 box-supervised WSSS는 GrabCut이나 MCG 같은 도구를 자주 사용했지만, 이 논문은 classification network와 CRF, retrieval만으로 pseudo label을 구성한다. 특히 BAP는 추가 파라미터 없이 GAP를 대체하는 방식이라 구조적으로도 깔끔하다.

세 번째 강점은 pseudo label 품질과 최종 segmentation 성능 사이의 연결을 설득력 있게 제시한 점이다. pseudo label 성능 향상, CAM quality 향상, filling rate 분포 개선, 최종 mIoU 향상이 서로 일관된 방향으로 나타난다. Supplement에서 runtime도 비교하는데, WSSL 대비 0.1초 GPU overhead 정도만 더 들고 성능 향상은 크다고 보고한다.

하지만 한계도 분명하다. 첫째, background가 이미지 내에서 어느 정도 perceptually consistent하다는 가정이 항상 성립하는 것은 아니다. 복잡한 장면, 다양한 texture background, 또는 foreground와 background가 시각적으로 매우 비슷한 경우에는 retrieval 기반 background estimation이 흔들릴 수 있다. 논문은 이 가정의 실패 사례를 체계적으로 분석하지는 않는다.

둘째, pseudo label 생성 과정은 여전히 DenseCRF에 의존한다. 저자들은 off-the-shelf segmentation 도구 의존은 줄였지만, boundary refinement는 여전히 CRF가 담당한다. 따라서 성능의 일부는 CRF의 hyperparameter tuning과 저수준 image cue에 기대고 있다. 실제로 CRF parameter는 fully annotated validation subset으로 cross-validation했다고 명시되어 있다.

셋째, $Y_{ret}$는 단독 품질이 $Y_{crf}$보다 낮다. 즉 retrieval branch는 독립적으로 강력한 pseudo label generator라기보다, noisy region 탐지와 보조 신호 역할에 더 가깝다. 논문도 이를 보완적 신호로 다루며, 단독 사용의 강점은 제한적이다.

넷째, state-of-the-art 주장에는 약간의 맥락이 필요하다. DeepLab-V1/V2 기준으로는 매우 강하지만, 더 강한 backbone과 decoder를 사용하는 방법과의 비교에서는 구조 차이가 있다. 저자들도 Box2Seg와의 비교에서 이를 인정한다. 따라서 "절대적 최고"라기보다 "동일하거나 유사한 segmentation backbone 기준 매우 강한 방법"으로 이해하는 편이 정확하다.

마지막으로, 일부 설계 선택은 경험적이다. 예를 들어 classification 학습 때는 $N=4$, pseudo label 생성 때는 $N=1$이 좋았다고 하지만, 왜 이런 설정이 일반적으로 안정적인지에 대한 이론적 설명은 제한적이다. 다만 Supplement에서 grid size와 $\gamma, \lambda$에 대한 분석을 제공해 최소한 경험적 뒷받침은 한다.

## 6. 결론

이 논문은 bounding box supervision만으로 semantic segmentation을 학습하는 문제에서, pseudo label 생성과 noisy label 학습이라는 두 핵심 병목을 각각 BAP와 NAL로 해결한 연구다. BAP는 definite background를 query로 삼아 bounding box 내부의 foreground를 더 잘 집계하게 만들고, 그 결과 기존 GAP보다 더 정확한 CAM과 pseudo label을 생성한다. NAL은 두 종류의 pseudo label이 일치하지 않는 영역을 버리지 않고 confidence-weighted하게 활용하여 segmentation network를 더 견고하게 학습시킨다.

실험적으로도 논문은 PASCAL VOC 2012에서 strong한 결과를 보였고, MS-COCO instance segmentation으로 확장 가능성도 제시했다. 특히 bounding box annotation의 실용성을 고려하면, 이 연구는 pixel-level annotation 비용을 줄이면서도 높은 segmentation 성능을 얻기 위한 중요한 방향을 보여준다. 향후 연구에서는 BAP의 background modeling을 더 정교하게 만들거나, CRF 의존도를 줄인 end-to-end pseudo label generation, 더 강한 backbone과의 결합, open-vocabulary 또는 unseen class generalization 쪽으로 확장할 여지가 크다.
