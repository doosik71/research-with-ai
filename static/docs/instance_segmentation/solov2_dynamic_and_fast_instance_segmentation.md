# SOLOv2: Dynamic and Fast Instance Segmentation

* **저자**: Xinlong Wang, Rufeng Zhang, Tao Kong, Lei Li, Chunhua Shen
* **발표연도**: 2020
* **arXiv**: [https://arxiv.org/abs/2003.10152](https://arxiv.org/abs/2003.10152)

## 1. 논문 개요

이 논문은 instance segmentation을 bounding box에 의존하지 않고 직접 수행하는 빠르고 정확한 방법인 **SOLOv2**를 제안한다. 저자들의 문제의식은 분명하다. 기존의 많은 instance segmentation 방법은 실제로는 “object detection 후 segmentation”이라는 구조를 따르며, 중심 표현으로 bounding box를 사용한다. 하지만 bounding box는 물체의 실제 경계를 거칠게 감싸는 표현일 뿐이며, 픽셀 단위의 정밀한 위치 표현에는 본질적인 한계가 있다. 저자들은 “물체를 box가 아니라 mask 자체로 직접 localize할 수 없을까?”라는 질문에서 출발한다.

이 논문은 이전 작업인 SOLO의 기본 철학, 즉 **segmenting objects by locations**를 계승한다. 입력 이미지를 $S \times S$ grid로 나누고, 각 grid cell이 특정 위치의 instance를 담당하게 만드는 방식이다. 이 접근은 instance segmentation을 복잡한 grouping 문제나 proposal 기반 문제로 보지 않고, fully convolutional network가 직접 풀 수 있는 구조로 단순화한다는 장점이 있다. 그러나 원래 SOLO에는 세 가지 병목이 있었다. 첫째, mask representation과 학습이 비효율적이었다. 둘째, mask 해상도가 충분히 높지 않아 경계가 정교하지 않았다. 셋째, mask-based NMS가 느렸다.

SOLOv2의 목표는 이 세 병목을 한 번에 해결하는 것이다. 이를 위해 저자들은 두 가지 핵심 축을 제안한다. 하나는 **dynamic instance segmentation**으로, mask 생성 과정을 동적으로 예측된 kernel과 공유된 high-resolution feature의 결합으로 재구성하는 것이다. 다른 하나는 **Matrix NMS**로, 기존의 순차적 mask suppression을 병렬 행렬 연산으로 바꾸어 추론 속도를 높이는 것이다. 결과적으로 SOLOv2는 COCO에서 정확도와 속도 모두에서 강력한 성능을 보이며, 저자들은 이 방법이 instance segmentation뿐 아니라 object detection과 panoptic segmentation의 강한 baseline이 될 수 있다고 주장한다.

이 문제가 중요한 이유는 명확하다. instance segmentation은 autonomous driving, medical imaging, AR, image editing 등에서 픽셀 단위의 정교한 object understanding을 요구하는 핵심 과제다. 만약 bounding box 없이도 빠르고 정확하게 instance segmentation을 수행할 수 있다면, detection 중심의 기존 인식 파이프라인을 다시 생각하게 만들 수 있다. 이 논문은 바로 그 가능성을 보여주려는 시도다.

## 2. 핵심 아이디어

SOLOv2의 중심 아이디어는 **instance mask를 직접 큰 출력 tensor로 예측하는 대신, mask kernel과 mask feature를 분리해서 학습한 뒤 동적으로 convolution하여 최종 mask를 만든다**는 것이다. 원래 SOLO에서는 각 위치별 instance mask를 직접 출력하려고 했기 때문에, $S^2$개의 위치마다 고해상도 mask channel이 필요했고, 이는 메모리와 연산 측면에서 비효율적이었다. SOLOv2는 이 점을 바꾼다. 각 위치는 이제 완성된 mask를 직접 내놓는 대신, 해당 위치를 담당하는 **작은 convolution kernel**을 예측한다. 그리고 전체 이미지에 대해 공유되는 **unified mask feature map**을 따로 만든다. 최종 instance mask는 이 kernel을 해당 feature map에 적용해서 생성된다.

이 설계의 직관은 매우 강하다. 이미지 안의 대부분 grid cell은 실제 object center를 가지지 않는다. 즉, $S^2$개의 모든 위치에 대해 거대한 mask를 미리 계산하는 것은 낭비다. 반면 동적 kernel 방식에서는 필요한 위치만 골라 해당 kernel을 feature map에 적용하면 되므로 훨씬 효율적이다. 동시에 kernel이 입력과 위치에 따라 예측되므로, mask 생성기가 정적으로 고정된 것이 아니라 **location-conditioned dynamic segmenter**가 된다.

두 번째 핵심 아이디어는 **unified high-resolution mask feature representation**이다. SOLOv2는 FPN의 여러 레벨 특징을 fuse하여 하나의 공통 mask feature map을 만든다. 이 unified feature는 $1/4$ scale의 비교적 높은 해상도를 가지며, medium/large object의 경계 표현에 유리하다. 별도의 level별 mask branch를 두는 대신 하나의 통합 표현을 사용함으로써 정확도와 효율을 동시에 얻는다.

세 번째 핵심 아이디어는 **Matrix NMS**이다. 기존 NMS나 Soft-NMS는 score가 높은 예측을 하나씩 처리하면서 나머지를 억제하는 순차적 구조다. mask NMS는 box NMS보다 pairwise IoU 계산 비용도 더 커서 병목이 심하다. Matrix NMS는 이 과정을 행렬 연산으로 한 번에 처리한다. 각 mask가 다른 더 높은 score의 mask들에 의해 얼마나 억제되어야 하는지를 병렬적으로 계산하고, 그중 가장 강한 suppression 효과를 선택한다. 이 방식은 단순한 속도 개선뿐 아니라 AP도 향상시킨다.

기존 접근과의 차별점은 다음과 같이 정리할 수 있다. Mask R-CNN류는 detection-first paradigm이고, YOLACT도 anchor box와 box crop에 의존한다. bottom-up 계열은 pixel embedding 후 clustering이 필요하다. 반면 SOLOv2는 box도, anchor도, clustering도 없이 **FCN-style direct instance segmentation**을 유지하면서 성능과 속도를 끌어올렸다. 논문에서 특히 강조하는 차별점은, CondInst 같은 동시대 dynamic convolution 계열과 달리 SOLOv2는 **absolute position**에 기반한 “location-wise segmentation”을 수행한다는 점이다. 따라서 instance 수가 늘어날수록 위치 정보를 반복 인코딩할 필요가 없고, 전체 이미지를 한 번에 처리할 수 있다.

## 3. 상세 방법 설명

### 전체 파이프라인

SOLOv2는 backbone과 FPN을 통해 multi-scale feature를 추출한 뒤, 크게 두 가지 분기를 만든다. 하나는 **mask kernel branch**, 다른 하나는 **mask feature branch**다. 전자는 각 위치(grid cell)에 대해 동적 convolution kernel을 예측하고, 후자는 전체 이미지에 대해 공유되는 high-resolution mask feature map을 만든다. 추론 시 category score가 높은 위치만 남긴 후, 해당 위치의 predicted kernel을 unified mask feature에 적용해 instance mask를 생성한다. 마지막으로 Matrix NMS를 적용해 중복 mask를 제거한다.

즉, 최종 mask 생성은 전통적인 “head가 직접 mask를 출력”하는 구조가 아니라 다음의 분해된 구조를 따른다.

$$
M_{i,j} = G_{i,j} * F
$$

여기서 $G_{i,j}$는 위치 $(i,j)$에 대해 예측된 convolution kernel이고, $F$는 전체 이미지에 대한 공통 mask feature이다. $M_{i,j}$는 해당 위치가 담당하는 최종 instance mask다. 이 식은 SOLOv2의 핵심을 가장 잘 보여준다. 원래는 $M$ 전체를 직접 예측했지만, 이제는 $G$와 $F$를 따로 예측하고 필요한 경우에만 둘을 결합한다.

### 위치 기반 instance 정의

SOLO 계열의 기본 가정은, 입력 이미지를 $S \times S$ grid로 나누고 object center가 속한 grid cell이 그 object instance를 담당한다는 것이다. 따라서 각 level은 총 $S^2$개의 위치를 담당하며, 각 위치는 하나의 instance mask를 생성할 잠재력을 가진다. 이 구조는 anchor나 proposal 없이 instance를 구분하게 해준다.

다만 원래 SOLO에서는 이 $S^2$개의 mask를 모두 직접 예측하는 방식이 비효율적이었다. SOLOv2는 이를 개선하기 위해 위치별로 kernel만 예측한다. 실제 object가 존재하는 위치만 이후 단계에서 활용되므로 계산 낭비가 줄어든다.

### Mask Kernel Branch

Mask kernel branch는 FPN의 각 pyramid level에서 동작한다. 입력 feature $F_I \in \mathbb{R}^{H_I \times W_I \times C}$를 먼저 $S \times S \times C$ 크기로 resize한 후, 여러 convolution을 거쳐 각 grid cell에 대해 $D$차원의 출력을 만든다. 이 $D$차원 벡터가 곧 해당 위치의 convolution kernel weight다.

논문은 첫 convolution 입력에 **CoordConv 스타일의 normalized coordinates**를 추가한다. 즉, $[-1,1]$ 범위로 정규화된 $x, y$ 좌표 채널 두 개를 feature에 concatenate한다. 이는 매우 중요하다. SOLOv2가 “objects by locations”를 수행하려면, 같은 appearance를 가진 두 물체라도 위치가 다르면 다른 kernel을 예측해야 한다. 좌표 정보가 없으면 네트워크는 동일한 외형의 물체를 동일한 segmenter로 처리해버릴 수 있다.

논문은 kernel shape도 ablation으로 비교한다. $3 \times 3$ kernel과 $1 \times 1$ kernel은 비슷한 성능을 보였고, input channel 수를 256까지 늘릴 때 성능이 좋아졌다. 최종적으로는 $1 \times 1 \times 256$ 구성이 사용되었다. 즉, 복잡한 큰 kernel보다 상대적으로 간단한 동적 kernel로도 충분히 좋은 성능을 얻는다.

### Unified Mask Feature Branch

Mask feature branch는 하나의 통합된 고해상도 feature map $F$를 만드는 역할을 한다. FPN의 $P2$부터 $P5$까지를 이용해 repeated $3 \times 3$ convolution, group normalization, ReLU, bilinear upsampling을 거친 후 모두 $1/4$ scale로 맞춰 더한다. 마지막에 $1 \times 1$ convolution, group norm, ReLU를 적용해 최종 unified mask feature를 얻는다.

이 unified representation의 의미는 크다. level별로 따로 mask feature를 만들면, 큰 물체는 높은 level의 낮은 spatial resolution feature에서 처리되므로 경계가 거칠어질 수 있다. 반면 SOLOv2의 unified feature는 모든 level의 정보를 모아 상대적으로 높은 해상도에서 mask를 생성할 수 있어, medium/large object의 boundary quality가 좋아진다.

또한 이 branch에서도 좌표 정보가 중요하다. 논문은 가장 깊은 FPN level인 $1/32$ scale feature에 normalized coordinate를 주입한다. 이는 feature가 절대 위치에 민감하도록 만들고, mask kernel이 기대하는 채널-공간 대응을 feature가 맞춰줄 수 있게 한다.

### Instance Mask 형성

추론 시 각 grid 위치 $(i,j)$에 대해 category score $\mathbf{p}*{i,j}$를 얻는다. 먼저 confidence threshold 0.1로 낮은 score를 제거한다. 남은 위치들의 predicted kernel $G*{i,j,:}$를 unified mask feature $F$에 적용해 soft mask를 생성한다. 이후 sigmoid를 통과시키고 threshold 0.5로 binary mask를 얻는다. 마지막으로 Matrix NMS를 적용해 최종 결과를 정리한다.

이 과정의 핵심은 “모든 위치에 대해 거대한 mask tensor를 한 번에 예측하지 않는다”는 점이다. 먼저 category branch가 유효한 위치를 좁히고, 그 위치에 대해서만 dynamic convolution을 적용한다. 이 덕분에 메모리와 계산량이 절감된다.

### 학습 목표와 손실 함수

학습 손실은 다음과 같다.

$$
L = L_{cate} + \lambda L_{mask}
$$

여기서 $L_{cate}$는 semantic category classification을 위한 **Focal Loss**이고, $L_{mask}$는 mask prediction을 위한 **Dice Loss**다. Focal Loss는 많은 negative location과 적은 positive location 사이의 class imbalance를 줄이는 데 적합하고, Dice Loss는 픽셀 단위 mask overlap을 직접 반영하기 때문에 instance mask 학습에 자연스럽다.

논문은 SOLO의 loss 정의를 계승하므로, 보다 세부적인 assignment rule이나 target 구성은 원 논문 [1]을 참고하라고 한다. 따라서 여기서 assignment 세부 규칙까지는 본문에 충분히 풀어 쓰이지 않았고, 제공된 텍스트만으로는 정확한 implementation detail을 모두 복원할 수는 없다.

### Matrix NMS

SOLOv2의 또 다른 핵심은 post-processing이다. Soft-NMS는 높은 score prediction이 이웃 prediction의 score를 overlap에 따라 감소시키는 방식이지만, 처리 순서가 재귀적이고 순차적이다. Matrix NMS는 “각 prediction이 최종적으로 얼마나 억제되어야 하는가”를 병렬적으로 계산한다.

특정 mask $m_j$에 대해 suppression은 두 요소에 의해 결정된다. 하나는 더 높은 score를 가진 mask $m_i$가 $m_j$와 얼마나 겹치는지, 즉 $f(\text{iou}_{i,j})$이다. 다른 하나는 그 $m_i$ 자신이 또 다른 더 강한 prediction에 의해 얼마나 suppression될 가능성이 있는지다. 논문은 후자를 정확히 계산하지 않고, $m_i$가 더 높은 score의 다른 prediction들과 갖는 최대 overlap을 이용해 근사한다.

이를 식으로 쓰면, 먼저 $m_i$의 suppression 가능성을 다음처럼 근사한다.

$$
f(\text{iou}*{\cdot,i}) = \min*{\forall s_k > s_i} f(\text{iou}_{k,i})
$$

그리고 $m_j$의 최종 decay factor는

$$
decay_j = \min_{\forall s_i > s_j} \frac{f(\text{iou}*{i,j})}{f(\text{iou}*{\cdot,i})}
$$

로 계산된다. 그 후 score는

$$
s_j = s_j \cdot decay_j
$$

로 갱신된다.

여기서 $f$는 overlap이 클수록 더 큰 억제를 주는 감소 함수다. 논문은 두 가지를 사용한다. 선형 버전은 $f(\text{iou}*{i,j}) = 1 - \text{iou}*{i,j}$이고, Gaussian 버전은

$$
f(\text{iou}*{i,j}) = \exp\left(-\frac{\text{iou}*{i,j}^2}{\sigma}\right)
$$

이다.

구현 측면에서 Matrix NMS는 top-$N$ predictions 간의 pairwise IoU matrix를 구한 뒤, 열 방향 max와 min 연산을 조합하여 모든 decay factor를 한 번에 계산한다. 즉, 재귀 없이 전부 matrix operation으로 처리한다. 논문 부록의 pseudo-code도 이 구조를 그대로 보여준다. 핵심 장점은 mask NMS의 병목을 거의 없애면서 정확도도 잃지 않는다는 점이다.

## 4. 실험 및 결과

### 실험 설정

주요 instance segmentation 실험은 **MS COCO**에서 수행되었다. ablation은 COCO val2017 5K split에서, 주요 비교 결과는 test-dev에서 보고한다. 학습은 SGD를 사용하며, 8 GPU에서 total batch size 16으로 학습한다. 기본 설정은 36 epochs, 즉 $3\times$ schedule이다. 초기 learning rate는 0.01이며 27 epoch와 33 epoch에서 10배씩 감소한다. 입력은 shorter side를 640에서 800 사이에서 랜덤 샘플링하는 scale jitter를 사용한다.

추가로 **LVIS v0.5**에서 long-tail instance segmentation 성능도 평가한다. 그리고 instance segmentation 모델에서 생성된 mask를 bounding box로 변환해 object detection 성능도 보고하며, semantic branch를 추가한 panoptic segmentation 확장도 실험한다.

### COCO instance segmentation 메인 결과

COCO test-dev에서 SOLOv2는 매우 강한 결과를 보인다. box-free 계열인 SOLO, PolarMask보다 확실히 좋고, 일부 box-based 강력 baseline과도 경쟁하거나 앞선다.

대표 수치는 다음과 같다.
Res-50-FPN SOLOv2는 **38.8 AP**, Res-101-FPN은 **39.7 AP**, Res-DCN-101-FPN은 **41.7 AP**를 달성한다. 이는 같은 표에서 SOLO Res-101-FPN의 **37.8 AP**보다 높고, BlendMask Res-101-FPN의 **38.4 AP**보다도 높다. YOLACT Res-101-FPN의 **31.2 AP**와 비교하면 큰 차이가 난다. 저자들이 본문에서 강조하듯, comparable speed에서 약 6 AP 이상 차이가 난다.

특히 large object에서 강점이 뚜렷하다. 예를 들어 Mask R-CNN Res-101-FPN이 $APL=52.4$ 또는 개선판에서 $49.3$인데 비해, SOLOv2 Res-101-FPN은 **57.4**, Res-DCN-101-FPN은 **61.6**을 기록한다. 이는 unified high-resolution mask feature와 direct mask generation 구조가 큰 물체의 경계 품질에 유리함을 뒷받침한다.

또한 논문은 Figure 1에서 속도-정확도 trade-off를 강조한다. 정확한 그래프 수치를 텍스트만으로 모두 복원할 수는 없지만, 본문에 따르면 light-weight SOLOv2는 **31.3 FPS에서 37.1 AP**를 달성한다. 이는 real-time에 가까운 처리 속도에서 상당히 높은 segmentation 성능이다.

### LVIS 결과

LVIS는 1,000개가 넘는 category를 가진 long-tail segmentation dataset으로 더 어렵다. SOLOv2는 이 데이터셋에서도 Mask R-CNN baseline을 능가한다. 예를 들어 Res-50-FPN에서 Mask R-CNN 재구현 $AP=24.6$에 비해 SOLOv2는 **25.5 AP**, Res-101-FPN에서는 **26.8 AP**를 기록한다.

흥미로운 점은 LVIS에서도 large object에서 강점이 뚜렷하다는 것이다. Res-50-FPN 기준으로 Mask R-CNN 재구현의 $APL=38.2$에 비해 SOLOv2는 **44.9**를 기록한다. 본문에서도 이 large-scale object 성능 향상이 COCO 결과와 일관적이라고 설명한다.

### Ablation 분석

#### Kernel shape

kernel shape를 바꿔도 성능은 비교적 안정적이었다. $3 \times 3 \times 64$와 $1 \times 1 \times 64$는 비슷한 성능을 보였고, channel 수를 256으로 늘리면 **37.8 AP**까지 상승한다. 512로 늘려도 더 좋아지지는 않았다. 이는 dynamic kernel이 지나치게 복잡할 필요는 없고, 적절한 channel capacity가 더 중요하다는 점을 시사한다.

#### Explicit coordinates

좌표 입력의 효과는 매우 크다. kernel과 feature branch 모두 좌표를 넣지 않으면 **36.3 AP**다. feature branch에만 넣으면 **37.1 AP**, 둘 다 넣으면 **37.8 AP**가 된다. 즉, explicit coordinate가 약 **1.5 AP** 향상을 만든다. 이는 SOLOv2의 본질이 “위치별 segmentation”이라는 점을 실험적으로 확인해준다. CNN의 zero-padding만으로는 절대 위치 정보가 충분히 정밀하지 않다는 해석도 타당하다.

#### Unified mask feature

separate feature representation보다 unified representation이 더 좋다. separate는 **37.3 AP**, unified는 **37.8 AP**다. 특히 medium/large object에서 이득이 크다고 논문은 설명한다. 이는 큰 물체를 낮은 해상도의 상위 FPN 레벨에서만 처리하면 경계가 거칠어진다는 가설과 잘 맞는다.

#### Matrix NMS

NMS 비교는 이 논문의 중요한 실험이다. Hard-NMS는 **9 ms, 36.3 AP**, Soft-NMS는 **22 ms, 36.5 AP**, Fast NMS는 **< 1 ms, 36.2 AP**, Matrix NMS는 **< 1 ms, 36.6 AP**다. 즉, Matrix NMS는 Fast NMS만큼 빠르면서도 AP는 오히려 더 높다. 재귀적 suppression을 병렬 행렬 연산으로 바꾸면서 정확도 희생을 막았다는 점이 핵심이다.

#### Training schedule 및 real-time 설정

$1\times$ schedule은 **34.8 AP**이고, $3\times$ schedule은 **37.8 AP**다. multi-scale과 longer training이 상당한 성능 향상을 준다.

real-time 모델의 경우 SOLOv2-448은 **46.5 FPS / 34.0 AP**, SOLOv2-512는 **31.3 FPS / 37.1 AP**다. 즉, 실시간 응용을 위한 accuracy-speed 조절이 가능하다.

### Object detection 확장

논문은 별도의 box supervision 없이 predicted mask에서 bounding box를 직접 추출해 object detection 성능도 평가한다. 이것은 개념적으로 매우 흥미롭다. instance segmentation이 detection보다 더 풍부한 표현이라면, 좋은 mask만 있으면 box는 파생물이어야 한다는 철학이 반영되어 있다.

실제로 COCO test-dev에서 SOLOv2 Res-DCN-101-FPN은 **44.9 box AP**를 달성한다. 이는 RetinaNet 39.1, FCOS 41.5, CenterNet 42.1보다 높고, 다수의 modern detector를 능가한다. Res-101-FPN 버전도 **42.6 AP**다. 저자들은 이 결과를 바탕으로, box annotation과 mask annotation의 비용 차이를 크게 문제 삼지 않는다면 downstream application에서 굳이 box detector를 쓸 이유가 줄어들 수 있다고까지 주장한다.

### Panoptic segmentation 확장

SOLOv2는 semantic segmentation branch를 추가하여 panoptic segmentation에도 확장된다. COCO val2017에서 **PQ 42.1**, **PQTh 49.6**, **PQSt 30.7**을 달성한다. 이는 AUNet 39.6, Panoptic-FPN 재구현 40.8, Pano-DeepLab 39.7보다 높은 수치다. box-free 계열 중에서는 AdaptIS 35.9, SSAP 36.5보다 확실히 앞선다. 즉, direct instance segmentation backbone이 panoptic setting에서도 강한 기반 모델이 될 수 있음을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 정의와 구현을 동시에 단순화하면서 성능까지 높였다**는 점이다. 많은 segmentation 논문이 복잡한 proposal 설계, matching, refinement, multi-stage heuristic에 의존하는 반면, SOLOv2는 FCN-like direct prediction이라는 깔끔한 구조를 유지한다. 그럼에도 COCO와 LVIS에서 강력한 성능을 보여, “단순한 구조는 성능이 낮다”는 통념을 반박한다.

두 번째 강점은 **dynamic mask kernel + unified feature**라는 분해 방식이 매우 설득력 있다는 점이다. 이는 단지 engineering trick이 아니라, sparse한 instance activation 구조에 맞는 더 자연스러운 표현이다. 필요한 위치에서만 dynamic convolution을 수행하므로 연산 효율이 높고, unified feature를 통해 boundary quality도 개선한다.

세 번째 강점은 **Matrix NMS**의 실용성이다. 많은 논문이 backbone이나 head만 개선하고 후처리는 그대로 두는 반면, 이 논문은 direct instance segmentation에서 mask NMS가 실질적 병목임을 짚고 이를 해결한다. 게다가 Matrix NMS는 SOLOv2에만 국한되지 않고 다른 segmentation/detection system에도 붙일 수 있는 일반적 아이디어로 보인다.

네 번째 강점은 **확장성**이다. instance segmentation에만 머물지 않고, mask로부터 box를 생성해 object detection 성능까지 보여주고, semantic branch를 더해 panoptic segmentation까지 확장한다. 이는 proposed representation이 특정 벤치마크용 요령이 아니라 instance-level recognition 전반에 적용 가능한 공통 기반일 수 있음을 시사한다.

반면 한계도 있다. 첫째, 방법의 핵심 가정은 여전히 **object center가 특정 grid cell에 대응된다**는 위치 기반 assignment에 있다. 이 방식은 구조적으로 우아하지만, 매우 밀집된 장면이나 center가 가깝게 몰린 복잡한 상황에서 어떤 한계를 가질 수 있는지는 본문에서 깊게 분석되지 않는다. 둘째, 좌표 정보가 성능에 크게 기여한다는 점은, 모델이 appearance만으로는 충분하지 않고 상당 부분 위치 encoding에 의존한다는 뜻이기도 하다. 이것이 복잡한 geometric variation이나 domain shift에서 어떤 영향을 주는지는 논문에서 다루지 않는다.

셋째, Matrix NMS의 suppression probability는 엄밀한 확률 모델이 아니라 **최대 overlap 기반 근사**다. 실험적으로는 잘 작동하지만, 이 근사의 이론적 정당성이 깊게 분석되지는 않는다. 넷째, instance assignment, positive sample 정의, level selection 등 원래 SOLO에서 물려받는 일부 세부 설계는 제공된 본문만으로 완전히 복원되기 어렵다. 즉, 전체 구현을 완전히 재현하려면 원 논문과 supplementary, 혹은 공개 코드를 함께 보는 것이 필요하다.

비판적으로 보면, 이 논문은 강력한 empirical paper다. 문제의 핵심 병목을 잘 짚고, 구조를 단순화하고, 실험으로 설득한다. 다만 이론적 분석보다는 실용적 설계와 benchmark 우위에 더 무게가 실려 있다. 그럼에도 컴퓨터 비전의 detection/segmentation 연구에서는 이런 종류의 기여가 매우 가치 있다. 특히 “box를 제거하고도 충분히 강한 recognition이 가능하다”는 메시지는 이후 direct segmentation 계열 연구에 중요한 방향을 제시한다.

## 6. 결론

SOLOv2는 instance segmentation을 bounding box 기반 보조 문제로 다루지 않고, **직접적이고 동적인 mask prediction 문제**로 다시 설계한 논문이다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, 위치별 dynamic convolution kernel을 예측해 compact하면서도 강력한 mask head를 만들었다. 둘째, unified high-resolution mask feature representation을 통해 더 정교한 object boundary를 얻었다. 셋째, Matrix NMS를 도입해 mask-based post-processing의 병목을 크게 줄이면서 정확도도 개선했다.

실험 결과는 이 설계가 단순한 아이디어 차원에 머물지 않음을 보여준다. COCO와 LVIS에서 높은 mask AP를 달성했고, light-weight 설정에서는 real-time에 가까운 속도도 확보했다. 더 나아가 predicted mask에서 직접 bounding box를 생성해 object detection에서도 강한 성능을 보였고, panoptic segmentation으로도 자연스럽게 확장되었다.

실제 적용 측면에서 이 연구는 매우 중요하다. autonomous driving이나 medical imaging처럼 픽셀 단위 localization이 중요한 분야에서는 box보다 mask가 본질적으로 더 풍부한 표현이다. SOLOv2는 그 표현을 정확하고 빠르게 만들 수 있음을 보여준다. 향후 연구에서는 더 정교한 assignment 전략, transformer 기반 backbone과의 결합, video 혹은 3D setting으로의 확장, long-tail 및 open-vocabulary 환경에서의 direct instance segmentation 등이 자연스러운 후속 방향이 될 것이다.

결론적으로 SOLOv2는 “instance segmentation을 어떻게 표현하고 계산할 것인가”에 대해 매우 강한 답을 제시한 논문이다. 구조는 단순하지만 메시지는 크다. **box-free, proposal-free, FCN-like direct instance segmentation도 충분히 빠르고 강력할 수 있다**는 점을 명확히 보여준다는 점에서, 이 논문은 instance-level recognition 연구의 중요한 기준점으로 볼 수 있다.
