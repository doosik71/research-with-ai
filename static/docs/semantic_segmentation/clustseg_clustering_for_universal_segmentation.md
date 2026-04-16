# CLUSTSEG: Clustering for Universal Segmentation

- **저자**: James Liang, Tianfei Zhou, Dongfang Liu, Wenguan Wang
- **발표연도**: 2023
- **arXiv**: https://arxiv.org/abs/2305.02187

## 1. 논문 개요

이 논문은 image segmentation의 여러 하위 문제를 하나의 공통된 관점으로 묶으려는 시도이다. 저자들은 superpixel segmentation, semantic segmentation, instance segmentation, panoptic segmentation을 각각 별개의 문제로 다루는 기존 흐름에서 벗어나, 모두를 “pixel clustering” 문제로 다시 해석한다. 즉, 각 pixel을 어떤 그룹에 배정할지 결정하면 segmentation mask가 나온다는 점에 주목하고, 이 과정을 transformer 기반의 neural clustering으로 구현한 통합 프레임워크 `CLUSTSEG`를 제안한다.

논문의 핵심 연구 문제는 다음과 같다. 서로 다른 segmentation task들은 목표가 다르다. semantic segmentation은 category-level grouping이 중요하고, instance segmentation은 object instance별 분리가 중요하며, superpixel은 low-level visual coherence와 spatial compactness가 중요하다. 그런데 기존 방법들은 이 차이를 반영하기 위해 task별로 완전히 다른 architecture나 pipeline을 사용해 왔다. 저자들은 “이질적인 요구를 유지하면서도 architecture는 하나로 통일할 수 있는가?”라는 질문을 던진다.

이 문제가 중요한 이유는 분명하다. segmentation 분야는 task별로 연구가 분절되어 있고, semantic은 per-pixel classification, instance는 detect-then-segment, panoptic은 proxy-task composition 등 서로 다른 기술적 전통 위에서 발전해 왔다. 이런 상황에서는 공통 원리를 재사용하기 어렵고, 연구 effort도 중복된다. 따라서 논문은 universal segmentation framework를 제안함으로써, segmentation 문제들을 더 근본적인 원리인 clustering 관점에서 통합하려 한다.

저자들은 특히 기존 universal segmenter들조차도 사실상 “mask classification” 관점에 머물러 있다고 비판한다. 반면 CLUSTSEG는 segmentation mask 자체가 clustering assignment의 결과라는 점을 직접 모델링한다. 이 때문에 저자들은 자신들의 접근이 더 transparent하고, clustering algorithm의 원리를 더 충실히 반영한다고 주장한다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 매우 명확하다. transformer의 query를 단순한 learnable token이 아니라 “cluster center”로 해석하면, segmentation은 결국 pixel feature를 여러 cluster center에 할당하는 문제로 볼 수 있다는 것이다. 그리고 cross-attention을 적절히 바꾸면 이것을 EM-style clustering처럼 사용할 수 있다.

이 아이디어는 두 개의 설계 축으로 구체화된다.

첫째는 `Dreamy-Start`이다. 저자들은 clustering에서 초기 center 선택이 매우 중요하다는 고전적 사실에 주목한다. 그런데 기존 transformer-based segmenter는 query를 대부분 fully parametric하게 학습한다. 즉, 왜 그 query가 그 task에 적합한 초기 중심점인지 명확하지 않다. CLUSTSEG는 task별 성격에 따라 query 초기화를 다르게 설계한다. semantic/stuff segmentation에서는 dataset 전반의 class-level structure를 반영한 class center를 쓰고, instance/things segmentation에서는 현재 입력 이미지에서 adaptive하게 seed를 만들며, superpixel segmentation에서는 image grid 기반으로 seed를 만든다. architecture는 유지하면서 initialization만 task-aware하게 바꾸는 것이 핵심이다.

둘째는 `Recurrent Cross-Attention`이다. 기존 segmentation transformer는 보통 여러 decoder layer를 쌓아서 query를 갱신하지만, 저자들은 이것이 EM clustering의 반복 구조를 충분히 닮지 않았다고 본다. EM은 assignment와 center update를 충분히 반복해야 local optimum에 수렴할 가능성이 높아진다. 따라서 CLUSTSEG는 같은 projection weights를 공유하는 non-parametric recurrence를 도입해, 한 layer 안에서 여러 번 `E-step`과 `M-step`을 반복한다. 추가 파라미터 없이 iterative clustering을 더 직접적으로 구현하려는 것이다.

기존 접근과의 차별점은 크게 세 가지다. 첫째, mask prediction을 단순 분류 문제가 아니라 clustering assignment로 해석한다. 둘째, query initialization을 task에 맞게 설계해 heterogeneous task를 한 architecture에 담는다. 셋째, cross-attention을 반복적 clustering solver로 재구성해 EM의 구조를 더 가깝게 따른다. 논문은 이 조합이 universal architecture이면서도 specialized model보다 강한 성능을 보인다고 주장한다.

## 3. 상세 방법 설명

### 3.1 문제 정식화

논문은 image segmentation을 다음처럼 정의한다. 입력 이미지 $I \in \mathbb{R}^{HW \times 3}$를 $K$개의 segment mask 집합으로 나누는 것이다.

$$
\mathrm{segment}(I) = \{M_k \in \{0,1\}^{HW}\}_{k=1}^{K}
$$

여기서 $M_k(i)$는 pixel $i$가 $k$번째 segment에 속하는지 나타낸다. 중요한 점은 $K$의 의미가 task마다 다르다는 것이다. superpixel에서는 사용자가 정한 개수이고, semantic segmentation에서는 semantic class 수로 고정되며, instance/panoptic에서는 이미지마다 달라진다.

저자들은 이것을 clustering으로 다시 쓴다. 각 mask는 하나의 cluster assignment 결과이고, segmentation은 결국 pixel을 적절한 cluster에 배정하는 과정이라는 것이다.

### 3.2 EM clustering과의 연결

논문은 EM clustering의 형식을 먼저 제시한다. 데이터 포인트 집합 $X \in \mathbb{R}^{N \times D}$를 $K$개 cluster로 나누는 문제는 cluster center $C \in \mathbb{R}^{K \times D}$와 assignment matrix $M \in \mathbb{R}^{K \times N}$를 찾는 문제로 볼 수 있다.

EM의 핵심은 두 단계 반복이다.

`E-step`에서는 현재 center를 기준으로 각 데이터가 어떤 cluster에 속할 확률이 높은지 계산한다.

$$
\hat{M}^{(t)} = \mathrm{softmax}_K(C^{(t)}X^\top)
$$

`M-step`에서는 그 확률을 가중치로 사용해 각 cluster center를 다시 계산한다.

$$
C^{(t+1)} = \hat{M}^{(t)}X
$$

저자들은 transformer query를 이 $C$에 대응시키고, pixel feature를 $X$에 대응시킨다.

### 3.3 Cross-attention의 clustering 해석

일반적인 cross-attention은 query가 image feature로부터 정보를 모아오는 방식이다.

$$
C \leftarrow C + \mathrm{softmax}_{HW}(Q^C(K^I)^\top)V^I
$$

여기서 softmax가 image 위치 축에 걸린다. 반면 CLUSTSEG는 softmax를 query 축, 즉 cluster 축에 걸어 pixel-to-cluster assignment처럼 만든다.

$$
C \leftarrow C + \mathrm{softmax}_K(Q^C(K^I)^\top)V^I
$$

이 변화가 중요하다. 이제 각 pixel이 어떤 query, 즉 어떤 cluster center에 더 가깝게 배정되는지 계산하는 형태가 되며, cross-attention이 EM의 assignment 계산과 구조적으로 비슷해진다.

### 3.4 Dreamy-Start: task-aware query initialization

논문의 첫 번째 핵심 기법은 초기 query를 task에 맞게 만드는 것이다.

#### Semantic segmentation / stuff segmentation

semantic segmentation에서는 같은 class의 pixel은 서로 다른 이미지, 서로 다른 instance에 있어도 같은 semantic cluster로 모여야 한다. 따라서 scene-specific한 seed보다 class-level 통계가 중요하다. 이를 위해 논문은 class별 memory bank $B=\{B_1,\dots,B_K\}$를 유지하고, 각 class queue에 저장된 pixel embedding의 평균을 class center로 사용한다.

$$
[c_1^{(0)};\dots;c_K^{(0)}] = \mathrm{FFN}([\bar{x}_1;\dots;\bar{x}_K]), \quad \bar{x}_k = \mathrm{AvgPool}(B_k)
$$

즉, semantic class마다 dataset-level prototype 비슷한 초기 center를 만든다. 학습이 끝나면 이 seed는 테스트에도 그대로 유지된다고 설명한다.

#### Instance segmentation / thing segmentation

instance segmentation에서는 서로 같은 class라도 다른 instance는 다른 cluster가 되어야 한다. 따라서 전역 class center보다 현재 이미지에 적응된 seed가 필요하다. 저자들은 position embedding을 더한 image feature에서 직접 초기 query를 생성한다.

$$
[c_1^{(0)};\dots;c_K^{(0)}] = \mathrm{FFN}(\mathrm{PE}(I))
$$

여기서 $K$는 보통 100으로 고정하며, 실제 객체 수보다 크게 둔다. 이 설계는 “changeless query” 대신 image-adaptive query를 사용한다는 점에서 기존 방법과 대비된다.

#### Panoptic segmentation

panoptic은 stuff와 thing을 동시에 다루므로, semantic/stuff용 초기화와 instance/thing용 초기화를 각각 별도로 적용한다. 즉, stuff는 scene-agnostic class center, thing은 scene-adaptive seed를 쓴다.

#### Superpixel segmentation

superpixel은 미리 정한 개수 $K$개의 작은 coherent region을 만드는 문제라서 grid 기반 seed가 자연스럽다. 논문은 position-embedded image feature에서 grid sampling을 통해 $K$개의 seed를 뽑는다.

$$
[c_1^{(0)};\dots;c_K^{(0)}] = \mathrm{FFN}(\mathrm{GridSample}_K(\mathrm{PE}(I)))
$$

이 방식은 전통적 superpixel 방법의 grid prior를 neural framework 안에 넣은 것으로 해석할 수 있다.

요약하면 Dreamy-Start는 architecture를 task별로 바꾸지 않으면서도, initialization만 바꿔 각 task의 성격을 반영한다.

### 3.5 Recurrent Cross-Attention: iterative clustering

초기 center를 얻은 뒤에는 `Recurrent Cross-Attention`으로 반복적으로 pixel-cluster assignment와 center update를 수행한다. 각 iteration은 EM의 E-step과 M-step에 대응한다.

E-step:
$$
\hat{M}^{(t)} = \mathrm{softmax}_K(Q^{C^{(t)}}(K^I)^\top)
$$

M-step:
$$
C^{(t+1)} = \hat{M}^{(t)}V^I
$$

여기서 $\hat{M}^{(t)} \in [0,1]^{K \times HW}$는 soft assignment matrix이며, 각 pixel이 각 cluster에 속할 확률을 의미한다. $Q$, $K$, $V$는 각각 query, key, value projection이다.

이 layer의 중요한 특징은 다음과 같다.

첫째, 반복 동안 projection weights를 공유하므로 recurrence가 생기지만 추가 학습 파라미터는 늘어나지 않는다.

둘째, $K$와 $V$는 한 번만 계산하고 반복 중에는 주로 $Q$만 다시 계산하므로 효율적이다.

셋째, 반복적 center refinement를 통해 더 나은 clustering configuration에 도달할 수 있다고 본다.

논문은 여러 해상도에서 이 layer를 계층적으로 사용한다.

$$
C^l = C^{l+1} + \mathrm{RCrossAttention}^{l+1}(I^{l+1}, C^{l+1})
$$

즉, multi-scale pixel feature 위에서 cluster center를 점진적으로 refinement한다.

### 3.6 전체 구조

CLUSTSEG는 네 부분으로 구성된다.

첫째, `Pixel Encoder`가 multi-scale dense feature $\{I^l\}$를 추출한다. 논문은 ResNet, ConvNeXt, Swin 등 다양한 backbone을 지원한다고 한다.

둘째, `Pixel Decoder`가 상위 feature에서 더 세밀한 representation을 복원한다. 논문은 axial block을 사용하며, 한 level과 그 아래 level에 총 여섯 개 block을 사용했다고 설명한다.

셋째, `Recurrent Cross-Attention based Decoder`가 실제 clustering을 수행한다. 각 recurrent layer는 기본적으로 $T=3$ iteration을 사용하고, 총 여섯 개 decoder를 사용한다.

넷째, `Dreamy-Start`가 첫 번째 decoder에 들어갈 초기 center를 task별 방식으로 만든다.

### 3.7 학습 목표와 loss

논문은 task마다 “standard loss design”을 따른다고 하며, 본문에서는 세부 사항을 모두 쓰지 않고 supplement에 더 자세히 넘긴다. 따라서 모든 loss를 본문만으로 완전히 복원할 수는 없다. 다만 제공된 supplementary 텍스트에는 일부 구체값이 있다.

공통적으로 중요한 점은, 각 recurrent cross-attention의 매 E-step에서 나오는 $\hat{M}^{(t)}$를 segment logit map처럼 볼 수 있기 때문에, 각 단계마다 ground truth segment mask로 supervision을 줄 수 있다는 것이다. 저자들은 이를 deep supervision이라고 설명한다.

panoptic segmentation에서는 다음 objective를 사용한다고 supplementary에 명시한다.

$$
L_{\mathrm{Panoptic}} = \lambda_{\mathrm{th}}L_{\mathrm{th}} + \lambda_{\mathrm{st}}L_{\mathrm{st}} + \lambda_{\mathrm{aux}}L_{\mathrm{aux}}
$$

계수는 $\lambda_{\mathrm{th}}=5$, $\lambda_{\mathrm{st}}=3$, $\lambda_{\mathrm{aux}}=1$이다. 또한 최종 “thing” center는 작은 FFN에 넣어 semantic classification을 수행하고 binary cross-entropy loss로 학습한다. $L_{\mathrm{aux}}$는 PQ-style loss, mask-ID cross-entropy, instance discrimination loss, semantic segmentation loss의 가중합이라고 적혀 있다.

instance segmentation에서는 binary cross-entropy loss와 dice loss를 사용하며 계수는 각각 5와 2라고 supplementary에 명시한다. semantic segmentation에서는 cross-entropy loss와 auxiliary dice loss를 결합하고 계수는 각각 5와 1이다. superpixel segmentation에서는 smooth L1 loss와 SLIC loss를 사용하며 계수는 각각 10과 1이다.

이처럼 task별 최종 objective는 다르지만, backbone과 clustering decoder의 기본 구조는 공통으로 유지된다.

## 4. 실험 및 결과

## 4.1 Panoptic Segmentation

실험 데이터셋은 COCO Panoptic이다. train2017으로 학습하고 val2017으로 평가했다. metric은 PQ, PQ Th, PQ St이며, 추가로 AP Th pan과 mIoU pan도 보고한다. 학습은 learning rate $1e{-5}$, 50 epoch, batch size 16으로 수행했고, random scale jittering 범위는 $[0.1, 2.0]$, crop size는 $1024 \times 1024$이다. 테스트는 짧은 변 800의 single-scale 입력을 사용한다.

결과는 매우 강하다. Swin-B backbone에서 CLUSTSEG는 `59.0 PQ`, `64.9 PQ Th`, `48.7 PQ St`, `47.1 AP Th pan`, `66.2 mIoU pan`을 기록했다. 같은 계열의 universal method인 Mask2Former의 Swin-B 결과 `56.3 PQ`보다 2.7 포인트 높다. ResNet-50과 ResNet-101에서도 각각 `54.3 PQ`, `55.3 PQ`를 기록해 Mask2Former보다 2.3, 2.9 포인트 높다.

specialized method와 비교해도 우수하다. 예를 들어 ConvNeXt-B backbone에서 kMaX-Deeplab의 `56.2 PQ` 대비 CLUSTSEG는 `58.8 PQ`를 기록한다. 논문은 이것이 PQ뿐 아니라 AP Th pan과 mIoU pan에서도 우세하다고 강조한다. 즉, panoptic task에서 thing/stuff 둘 다 개선되었다는 주장이다.

## 4.2 Instance Segmentation

데이터셋은 COCO instance segmentation이며 train2017으로 학습하고 test-dev로 평가했다. metric은 AP, AP50, AP75, APS, APM, APL이다. 학습 설정은 panoptic과 유사하게 learning rate $1e{-5}$, 50 epoch, batch size 16, scale jittering $[0.1,2.0]$, crop size $1024 \times 1024$이다.

대표 결과를 보면, Swin-B backbone에서 CLUSTSEG는 `49.1 AP`, `70.3 AP50`, `52.9 AP75`, `30.1 APS`, `53.2 APM`, `68.4 APL`을 기록했다. Mask2Former의 Swin-B 결과 `47.9 AP`보다 높다. ResNet-101에서는 `45.5 AP`로 Mask2Former의 `43.9 AP`보다 1.6 포인트 높고, K-Net의 `40.1 AP`보다 훨씬 높다.

특히 specialized method와 비교해도 강하다. 예를 들어 ResNet-50 기준 kMaX-Deeplab이 `40.2 AP`인데 CLUSTSEG는 `44.2 AP`다. 논문은 별다른 “bells and whistles” 없이 새로운 state-of-the-art를 기록했다고 주장한다.

## 4.3 Semantic Segmentation

ADE20K를 사용하며 train/val/test split은 20K/2K/3K이다. 논문 본문 실험은 val에서 보고한다. metric은 mIoU이다. 학습은 learning rate $1e{-5}$, 100 epoch, batch size 16, scale jittering $[0.5,2.0]$, crop size $640 \times 640$이다. 테스트 시에는 짧은 변을 640으로 리사이즈하고 test-time augmentation은 사용하지 않는다.

결과는 다음과 같다. ResNet-50 backbone에서 `50.5 mIoU`, ConvNeXt-B에서 `57.3 mIoU`, Swin-B에서 `57.4 mIoU`를 기록했다. Mask2Former의 ResNet-50 결과 `48.2 mIoU`, Swin-B 결과 `54.5 mIoU`와 비교하면 각각 2.3, 2.9 포인트 개선이다. Segformer `51.4`, Segmenter `53.5`, SETR `49.3`보다도 높다.

ADE20K가 이미 경쟁이 치열한 benchmark라는 점을 고려하면, universal architecture가 specialized semantic segmentation model까지 앞선다는 점이 논문의 주요 설득 포인트다.

## 4.4 Superpixel Segmentation

데이터셋은 BSDS500을 사용한다. split은 200/100/200이다. 학습은 learning rate $1e{-4}$, 300K iterations, batch size 128으로 수행했다. augmentation은 horizontal/vertical flip, scale jittering $[0.5,2.0]$, crop size $480 \times 480$이다. 학습 중 superpixel 개수는 50에서 2500 사이에서 랜덤하게 고른다. 추론 시에는 원본 이미지 크기를 사용한다. 평가 metric은 ASA와 CO이다. ASA는 boundary adherence를 반영하고, CO는 compactness를 평가한다.

논문 본문에는 정확한 수치 표 대신 그래프가 제시되어 있으며, CLUSTSEG가 SLIC, SSFCN, LNS보다 전반적으로 더 높은 ASA와 CO를 달성했다고 설명한다. 특히 edge preservation과 compactness 사이에는 일반적으로 trade-off가 있는데, CLUSTSEG는 둘 다 잘 유지한다고 주장한다.

Supplementary에서는 NYUv2에도 zero-shot에 가까운 방식으로 generalization을 확인했다. BSDS500에서 학습한 모델을 NYUv2에 fine-tuning 없이 적용했을 때도 ASA와 CO 모두 경쟁 방법보다 우수했다고 한다. 이는 superpixel setting에서의 generalizability를 보여주려는 실험이다.

## 4.5 Ablation Study

### Key component analysis

COCO Panoptic val, ResNet-50 기준 baseline은 fully learned initial query와 vanilla cross-attention decoder를 사용하며 `49.7 PQ`를 기록한다. 여기에 Dreamy-Start만 추가하면 `51.0 PQ`, Recurrent Cross-Attention만 추가하면 `53.2 PQ`, 둘 다 쓰면 `54.3 PQ`가 된다.

이 결과는 두 기법이 모두 유효하며, 특히 recurrent clustering의 기여가 더 크다는 점을 보여준다. 동시에 둘을 함께 쓸 때 가장 높은 성능이 나와 상호 보완적이라는 해석이 가능하다.

### Dreamy-Start 세부 분석

query를 free parameter로 둘 때는 `53.2 PQ`, `59.1 PQ Th`, `44.9 PQ St`였다. thing center를 scene-adaptive하게 초기화하면 PQ Th가 크게 오른다. stuff center를 scene-agnostic class center로 초기화하면 PQ St가 오른다. 최종적으로 thing은 scene-adaptive, stuff는 scene-agnostic으로 두는 조합이 최고 성능 `54.3 PQ`, `60.4 PQ Th`, `45.8 PQ St`를 낸다.

즉, 이 논문은 “모든 query를 같은 방식으로 초기화하면 안 된다”는 메시지를 강하게 전달한다. task뿐 아니라 thing/stuff 성질 차이도 반영해야 한다는 것이다.

### Recurrent Cross-Attention 세부 분석

vanilla cross-attention은 `51.0 PQ`, K-Means cross-attention은 `53.4 PQ`, 제안한 recurrent 방식은 `54.3 PQ`를 기록했다. 속도 측면에서는 vanilla보다 훨씬 효율적이고, K-Means 기반 방법과도 비슷한 수준이라고 한다. 예를 들어 training speed는 recurrent가 `1.62 hour/epoch`, K-Means가 `1.58`, vanilla가 `1.89`이다. inference speed는 recurrent `7.59 fps`, K-Means `7.81`, vanilla `5.88`이다.

추가 ablation에서는 같은 총 iteration 수를 맞춰 여러 non-recurrent cross-attention layer를 쌓아도 recurrent sharing 구조보다 성능이 낮았고, 추가 파라미터만 증가했다. 저자들은 parameter sharing이 EM-style iterative refinement의 본질을 유지하는 데 중요하다고 해석한다.

### Iteration 수 분석

반복 횟수 $T$를 늘리면 `T=1`에서 `53.8 PQ`, `T=2`에서 `54.1`, `T=3`에서 `54.3`으로 좋아지다가 이후에는 거의 포화된다. 반면 속도는 점점 느려진다. 그래서 기본값을 $T=3$으로 설정했다. supplementary에서도 final recurrent layer의 iteration별 예측 성능이 점진적으로 개선되는 것을 보여 준다.

### Deep supervision

각 recurrent cross-attention의 모든 E-step에 supervision을 주면 `54.3 PQ`이고, 마지막 E-step에만 supervision을 주면 `53.0 PQ`다. 따라서 iterative clustering의 각 단계에 직접적인 학습 신호를 주는 것이 성능 향상에 도움이 된다고 볼 수 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 segmentation을 clustering 관점으로 일관되게 다시 정리했다는 점이다. 단순히 여러 task를 한 model에 넣은 것이 아니라, 왜 그것이 하나의 공통 문제인지에 대한 개념적 설명이 비교적 분명하다. query를 cluster center로 보고, cross-attention을 assignment 연산으로 보고, 반복 갱신을 EM과 연결하는 서술은 설계 의도를 이해하기 쉽게 만든다. 논문이 스스로 강조하듯 “transparent”하다는 표현은 과장이 아니다.

두 번째 강점은 heterogeneity를 무시하지 않는다는 점이다. universal architecture를 주장하는 논문은 종종 모든 task를 같은 input-output 형식으로 강제로 맞추는 경향이 있는데, 이 논문은 semantic/stuff와 instance/thing, superpixel이 서로 다르다는 점을 인정하고 initialization 단계에서 이를 흡수한다. architecture를 바꾸지 않으면서도 task-aware behavior를 구현한 점이 설계상 깔끔하다.

세 번째 강점은 empirical result가 넓고 강하다는 점이다. panoptic, instance, semantic, superpixel 네 가지 task 전부에서 competitive하거나 state-of-the-art 수준의 결과를 보인다. universal method일 뿐 아니라 specialized method와도 비교해 우세하다는 것은 논문의 주장을 강하게 뒷받침한다.

하지만 한계도 있다. 첫째, task마다 loss design은 결국 다르다. architecture는 통합되었지만 학습 목표는 상당 부분 task-specific하며, 특히 panoptic loss에는 auxiliary term이 여럿 들어간다. 따라서 완전히 동일한 learning formulation까지 통합했다고 보기는 어렵다.

둘째, semantic/stuff용 Dreamy-Start는 memory bank에 저장된 class-wise embedding 평균을 사용한다. 이 방식은 closed-set semantic vocabulary와 training distribution에 의존한다. 새로운 class나 open-vocabulary setting으로 일반화될지는 본 논문만으로는 알 수 없다. 논문은 이 부분을 직접 다루지 않는다.

셋째, instance/panoptic에서 $K$개의 query 수를 크게 잡는 방식은 여전히 query budget에 의존한다. 실제 이미지 내 instance 수를 자동으로 정확히 조절하는지, 또는 과잉 query가 어떤 방식으로 억제되는지는 본문에서 충분히 상세히 설명되지는 않는다. 최종 center에 작은 FFN을 붙여 semantic classification을 한다는 정도만 명시되어 있다.

넷째, 계산 비용 측면에서 반복 clustering loop가 추가된다. 논문은 실제 오버헤드가 작고, supplementary에서 training speed 감소가 약 5.19%라고 설명하지만, 이것이 더 큰 backbone이나 더 높은 resolution에서도 항상 유지되는지는 본문만으로는 판단하기 어렵다.

다섯째, failure case 분석에서 저자들 스스로 매우 복잡한 장면, highly similar and occluded instances, small objects, complex topology, distorted backgrounds에서 약점을 인정한다. 결국 clustering 기반 접근은 feature space에서 분리가 어려운 경우에 성능이 흔들릴 수 있다.

비판적으로 보면, 논문이 EM clustering과의 유사성을 강하게 내세우지만, 실제 모델은 선형 projection, transformer decoder hierarchy, task-specific loss 등 신경망 구성요소를 포함한다. 따라서 이는 엄밀한 의미의 EM이라기보다 “EM-inspired neural clustering”으로 보는 편이 정확하다. 다만 논문도 이 점을 숨기지는 않고, EM의 원리를 가까이 따르도록 설계했다는 수준에서 주장한다.

## 6. 결론

이 논문은 image segmentation의 네 가지 핵심 하위 문제를 하나의 clustering 관점으로 통합하려는 야심 있는 시도이며, 그 결과물로 `CLUSTSEG`를 제안한다. 핵심 기여는 두 가지로 요약된다. 하나는 task-aware query initialization인 `Dreamy-Start`, 다른 하나는 parameter-efficient iterative clustering module인 `Recurrent Cross-Attention`이다. 이 둘을 통해 transformer query를 의미 있는 cluster center로 만들고, pixel-cluster assignment를 반복적으로 개선한다.

실험 결과는 이 설계가 단순한 아이디어 수준이 아니라 실제로 매우 강한 성능을 낸다는 것을 보여준다. panoptic, instance, semantic, superpixel segmentation 모두에서 기존 universal method를 넘어서고, 일부 specialized strong baseline도 앞선다. 따라서 이 연구는 “universal segmentation은 성능을 희생해야 한다”는 인식을 상당 부분 깨는 결과라고 볼 수 있다.

향후 연구 측면에서는 두 방향이 중요해 보인다. 하나는 더 복잡한 장면과 ambiguous boundary를 다룰 수 있는 강한 clustering solver 개발이고, 다른 하나는 closed-set segmentation을 넘어 open-vocabulary 또는 broader dense prediction으로 확장하는 것이다. 논문도 마지막에 이 연구가 broader dense visual prediction에 기여할 가능성을 언급한다. 적어도 이 논문이 segmentation을 clustering으로 다시 생각하게 만들었다는 점은 분명한 학술적 가치가 있다.
