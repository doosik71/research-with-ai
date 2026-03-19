# Sparse Instance Activation for Real-Time Instance Segmentation

## 1. Paper Overview

이 논문은 **실시간(real-time) instance segmentation**을 위해, 기존의 detection-first 또는 dense center/anchor 기반 접근과 다른 새로운 표현 방식을 제안합니다. 저자들은 대부분의 기존 방법이 bounding box, center point, dense anchor에 의존해 물체를 먼저 찾고 그 다음 mask를 예측하기 때문에, 중복 예측이 많고 연산량이 크며, NMS 같은 후처리도 병목이 된다고 지적합니다. 특히 robotics나 autonomous driving처럼 latency가 중요한 환경에서는 이러한 구조가 치명적일 수 있습니다.

이 문제를 해결하기 위해 논문은 **instance activation map(IAM)** 이라는 새로운 object representation을 제안합니다. 핵심 발상은 “물체를 box나 center로 대표하지 말고, 그 물체를 잘 설명하는 픽셀 영역을 sparse한 activation map으로 직접 강조하자”는 것입니다. 이 activation map을 이용해 instance-level feature를 뽑고, 이를 다시 분류와 mask prediction에 사용합니다. 결과적으로 SparseInst는 detector-independent한 fully convolutional framework이면서도, COCO에서 **37.9 mask AP와 40 FPS**를 달성해 속도와 정확도 모두에서 강한 실시간 성능을 보입니다. 더 작은 입력에서는 **58.5 FPS**도 달성합니다.

이 문제가 중요한 이유는 분명합니다. instance segmentation은 자율주행, 로보틱스, 비전 기반 edge deployment에서 자주 필요하지만, 많은 기존 방법은 너무 무겁거나 후처리에 의존합니다. 따라서 논문은 “정확도는 유지하면서 더 단순하고 빠른 구조를 만들 수 있는가?”라는 실용적이면서도 중요한 질문에 답하려고 합니다.

## 2. Core Idea

이 논문의 핵심 아이디어는 다음과 같습니다.

> **각 객체를 bounding box나 center point가 아니라, sparse한 instance activation map으로 표현하고, 이 activation map이 강조하는 영역에서 instance feature를 직접 집계해 분류와 mask 생성을 수행한다.**

기존 center-based 방식은 center가 객체를 충분히 대표하지 못할 수 있고, region-based 방식은 RoI 내부에 다른 객체나 배경 정보가 섞일 수 있습니다. 반면 IAM은 객체에 해당하는 **discriminative region**을 강조하고 방해가 되는 픽셀을 억제하므로, instance-aware feature를 더 직접적으로 만들 수 있습니다. 또한 feature를 whole image 문맥에서 aggregate할 수 있어 local receptive field 한계를 줄일 수 있습니다.

또 하나의 중요한 아이디어는 **bipartite matching**입니다. IAM은 anchor나 center처럼 사람이 hand-crafted rule로 target assignment를 하기가 어렵습니다. 그래서 DETR 계열처럼 Hungarian matching을 사용해 prediction과 GT를 one-to-one로 매칭합니다. 이 구조 덕분에 중복 예측이 억제되고, inference 시 **NMS가 필요 없어집니다.**

정리하면 SparseInst의 novelty는 세 부분입니다.

1. **IAM 기반 object representation**
2. **single-level, detector-free, fully convolutional 구조**
3. **Hungarian matching 기반 one-to-one prediction으로 NMS 제거**

## 3. Detailed Method Explanation

### 3.1 전체 구조

SparseInst는 세 개의 큰 모듈로 구성됩니다.

* **Backbone**
* **Instance Context Encoder**
* **IAM-based Decoder**

Backbone은 입력 이미지에서 multi-scale feature $\{C_3, C_4, C_5\}$를 뽑습니다. Encoder는 이 feature들을 더 넓은 receptive field와 멀티스케일 정보를 반영하도록 정제합니다. 마지막 Decoder는 instance branch와 mask branch로 나뉘어, IAM 생성, instance feature 집계, classification, mask kernel prediction, 최종 segmentation을 수행합니다.

논문이 강조하는 설계 철학은 “복잡한 multi-level detection head를 늘리지 말고, **single-level feature에서 sparse prediction**만 하자”는 것입니다. 이를 통해 연산량과 중복 예측을 줄입니다.

### 3.2 Instance Activation Maps

논문은 입력 feature를 $\mathbf{X} \in \mathbb{R}^{D \times (H \times W)}$라고 두고, IAM을 다음처럼 정의합니다.

$$
\mathbf{A} = \mathcal{F}_{iam}(\mathbf{X}) \in \mathbb{R}^{N \times (H \times W)}
$$

여기서 $N$은 잠재적 instance 수이고, 각 activation map은 한 객체의 informative region을 강조하는 역할을 합니다. $\mathcal{F}_{iam}$은 sigmoid를 포함한 간단한 네트워크입니다.

그 다음 정규화된 activation map $\bar{\mathbf{A}}$를 이용해 instance feature를 만듭니다.

$$
z = \bar{\mathbf{A}} \cdot \mathbf{X}^{T} \in \mathbb{R}^{N \times D}
$$

즉, 각 instance map이 강조한 위치의 feature를 가중합해 instance-level representation을 얻습니다. 이 $z_i$들은 각 객체의 분류와 mask kernel 생성을 위한 표현으로 사용됩니다. 이 수식의 의미는 단순하지만 중요합니다. 기존 방식처럼 박스를 자르거나 중심점 주변만 보지 않고, **map이 선택한 픽셀 전체에서 feature를 모은다**는 것입니다.

### 3.3 IAM이 왜 유효한가

논문은 IAM이 다음 장점을 가진다고 설명합니다.

* discriminative pixel을 강조하고 obstructive pixel을 억제한다.
* whole-image context에서 instance feature를 모을 수 있다.
* RoI-Align 같은 별도 연산 없이 단순한 방식으로 instance feature를 계산할 수 있다.

또한 IAM 자체에 직접적인 supervision이 있는 것은 아니고, 뒤쪽 recognition/segmentation module의 간접 supervision과 bipartite matching이 IAM으로 하여금 “각 map이 하나의 객체만 잘 보이게” 학습되도록 유도합니다. 즉, IAM은 수작업 mask target을 직접 맞추는 방식이 아니라, end task loss가 거꾸로 informative region discovery를 학습시키는 구조입니다.

### 3.4 Instance Context Encoder

실시간성을 위해 SparseInst는 **single-level prediction**을 택하지만, 그러면 객체 scale variation을 다루기 어렵습니다. 이를 보완하기 위해 저자들은 **Instance Context Encoder**를 둡니다. Encoder는 C5 뒤에 **PPM (Pyramid Pooling Module)** 을 붙여 receptive field를 키우고, P3~P5 feature를 fuse해 단일 출력 feature에 multi-scale 정보를 담습니다. 최종 출력은 입력 대비 $\frac{1}{8}$ 해상도의 single-level feature입니다.  

즉, 이 모듈은 “multi-level head는 쓰지 않되, backbone의 multi-scale 정보는 버리지 않겠다”는 절충입니다. 실시간성을 위해 head는 단순화하고, scale robustness는 encoder에서 확보하려는 설계라고 볼 수 있습니다.

### 3.5 IAM-based Decoder

Decoder는 두 갈래입니다.

* **Instance branch**: IAM 생성, instance feature 생성, classification, mask kernel 예측
* **Mask branch**: mask feature $\mathbf{M}$ 생성

논문은 instance branch와 mask branch가 모두 여러 개의 $3\times3$ conv(256 채널)로 구성된다고 설명합니다. 또한 위치 정보가 객체 분리에 도움이 된다고 보고, normalized absolute $(x,y)$ 좌표 2채널을 feature에 concat하여 **location-sensitive feature**를 만듭니다. 이는 CoordConv와 비슷한 발상입니다.

최종 mask는 예측된 instance-aware kernel과 mask feature를 곱해 생성됩니다. thresholding으로 binary mask를 얻고, category와 confidence score를 함께 계산합니다. 이 과정에서 **sorting과 NMS가 필요 없기 때문에 inference가 매우 빠릅니다.**

### 3.6 Bipartite Matching

IAM은 anchor처럼 고정된 spatial prior가 없기 때문에, GT assignment를 사람이 규칙으로 정하기 어렵습니다. 따라서 SparseInst는 DETR처럼 Hungarian matching을 사용하여 각 GT object를 하나의 prediction과 one-to-one로 대응시킵니다. 이 과정은 두 가지 역할을 합니다.

* 각 IAM이 하나의 객체를 담당하도록 유도
* redundant prediction을 억제해 NMS 제거 가능하게 함

이 구조는 SparseInst가 query-based transformer는 아니지만, “end-to-end one-to-one prediction”이라는 DETR류 장점을 convolutional framework에 가져온 것으로 볼 수 있습니다.

### 3.7 IoU-aware Objectness

논문은 one-to-one assignment 때문에 많은 prediction이 background가 되면서 classification confidence와 실제 mask quality가 어긋날 수 있다고 지적합니다. 이를 완화하기 위해 **IoU-aware objectness**를 도입합니다. foreground object에 대해서는 예측 mask와 GT mask 사이의 estimated IoU를 objectness target으로 사용하고, inference 때 classification probability를 이 objectness로 rescore합니다.

흥미로운 점은, 저자들이 별도의 IoU prediction head를 크게 늘리는 방향이 아니라 **IoU를 supervision target으로만 활용하는 간결한 방식**을 택했다는 것입니다. 논문 ablation에 따르면 이 기법은 baseline 대비 **1.3 AP** 향상을 제공합니다. rescoring 없이 objectness prediction만 추가해도 **0.7 AP**가 오르는데, 이는 objectness loss 자체가 instance branch가 더 instance-aware feature를 배우도록 돕기 때문이라고 해석합니다.

### 3.8 Group-IAM

논문은 더 강한 변형으로 **Group-IAM**도 제시합니다. 이는 여러 activation map을 그룹으로 묶어 instance feature를 concat하는 방식으로, 더 풍부한 표현을 만들려는 의도입니다. 구체적으로, 기본 IAM보다 확장된 표현력을 제공해 accuracy-speed tradeoff를 조절하는 옵션처럼 쓰입니다.

## 4. Experiments and Findings

### 4.1 실험 설정

논문은 MS-COCO에서 accuracy와 inference speed를 함께 평가합니다. SparseInst는 특히 real-time instance segmentation을 목표로 하므로, YOLACT 같은 실시간 지향 방법들과 speed-accuracy tradeoff를 중심으로 비교합니다. 또한 ResNet-50, ResNet-d 등 backbone 변형과 Group-IAM을 조합해 다양한 operating point를 제시합니다.

### 4.2 Main Results

가장 핵심적인 결과는 다음입니다.

* COCO benchmark에서 **37.9 AP, 40 FPS**
* 더 작은 448 입력에서는 **58.5 FPS**로 더 빠른 추론
* 실시간 segmentation 방법들 대비 전반적으로 더 좋은 speed-accuracy tradeoff

논문은 SparseInst가 YOLACT보다 더 빠르면서 더 높은 성능을 낸다고 강조합니다. 또한 detector-free이고 NMS-free라는 구조적 단순성을 고려하면, 이 결과는 단순 수치 이상의 의미가 있습니다. “더 복잡한 detection pipeline 없이도 실시간 instance segmentation이 가능하다”는 것을 보여주기 때문입니다.

### 4.3 Ablation: IoU-aware Objectness

IoU-aware objectness는 baseline 대비 **1.3 AP**를 개선합니다. 단순히 classification score calibration 역할만 하는 것이 아니라, foreground 인스턴스마다 다른 objectness target을 주기 때문에 instance branch가 더 구분력 있는 feature를 학습하도록 도와준다고 설명합니다.

이 실험은 SparseInst가 단순히 “IAM만 있으면 된다”가 아니라, **one-to-one assignment에서 생기는 confidence mismatch 문제를 별도로 다뤄야 한다**는 점을 보여줍니다.

### 4.4 Ablation: IAM vs Cross-Attention

논문은 IAM이 정말 필요한지 보기 위해, 4-head cross attention + 100 queries 방식으로 바꿔 실험합니다. 결과는 cross attention이 IAM 또는 Group-IAM보다 **0.2 AP 혹은 0.9 AP 낮습니다.** 저자들은 IAM이 더 큰 context와 지역 패턴을 효과적으로 본다고 해석합니다.

이 부분은 중요합니다. SparseInst는 얼핏 DETR류 query 방식으로도 바꿀 수 있을 것 같지만, 논문은 **단순한 activation map 기반 집계가 오히려 더 효과적**이라고 주장합니다. 즉, novelty가 단순한 구조적 단순화에만 있는 것이 아니라, representation 자체에도 의미가 있다는 근거입니다.

### 4.5 Error Analysis

TIDE 분석에 따르면 SparseInst는 YOLACT++나 SOLOv2보다 **miss error가 낮아 더 많은 객체를 발견**합니다. 반면 classification error와 duplicate error 비중은 다소 높습니다. 저자들은 duplicate 제거가 classification score에 의존하기 때문에, 더 좋은 classification 능력이 SparseInst의 추가 개선 포인트라고 해석합니다.

이 분석은 SparseInst의 장단점을 잘 보여줍니다. 구조적으로 sparse prediction을 잘 수행해 object discovery는 강하지만, query별 confidence calibration이나 class discrimination은 아직 병목일 수 있다는 뜻입니다.

### 4.6 Qualitative Results

시각화 결과에서 IAM은 객체의 scale, occlusion, pose와 관계없이 **discriminative region**을 잘 강조하는 경향을 보입니다. 이 점은 논문이 주장한 “highlight to segment” 패러다임이 실제로 동작하고 있음을 보여줍니다. 즉, IAM은 단순 attention map 시각화가 아니라, 실제 segmentation에 유의미한 instance-aware region selection 역할을 합니다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 **문제 정의와 구조의 단순성**입니다. detector 없이, single-level prediction만으로, NMS 없이 instance segmentation을 빠르게 수행한다는 점이 매우 인상적입니다. IAM은 region-based와 center-based 방식의 약점을 동시에 비판하면서 새로운 representation을 제시했고, 실제로 그 representation이 speed-accuracy tradeoff에서 효과적임을 보여줍니다.

또한 SparseInst는 transformer-heavy한 DETR류 접근과 달리 **fully convolutional**이어서 edge deployment 친화적입니다. 실무적으로도 이 점은 의미가 큽니다.

### Limitations

다만 한계도 분명합니다. 첫째, 논문 자체의 TIDE 분석에서 드러나듯이 classification/duplicate error가 상대적으로 큽니다. 즉, “객체를 찾는 것”은 강하지만, “정확히 어떤 class이며 얼마나 신뢰할 수 있는가”는 추가 개선 여지가 있습니다.

둘째, single-level prediction은 효율적이지만, 근본적으로 모든 scale variation을 다루는 데 한계가 있어 encoder 설계에 많이 의존합니다. 논문은 이를 PPM과 feature fusion으로 보완하지만, extreme scale variation 상황에서는 여전히 multi-level 방식보다 불리할 가능성이 있습니다.

셋째, IAM은 매우 흥미로운 표현이지만, 직접적인 mask supervision으로 map을 가르치는 것이 아니므로, 학습이 간접 supervision에 의존합니다. 이 점은 설계상 elegant하지만, optimization 안정성이나 해석 가능성 면에서는 양날의 검일 수 있습니다.

### Brief Critical Interpretation

비판적으로 보면 SparseInst의 진짜 기여는 단순히 “새로운 map 하나 제안”이 아니라, **instance segmentation 전체 파이프라인을 sparse prediction 중심으로 다시 설계한 것**에 있습니다. IAM, Hungarian matching, NMS 제거, single-level encoder-decoder 구조가 함께 맞물릴 때 비로소 SparseInst의 강점이 드러납니다.

또한 이 논문은 실시간 segmentation에서 자주 보이는 tradeoff를 잘 다룹니다. 정확도만 올리는 방향이 아니라, 어떤 연산이 latency를 키우는지를 분석하고, representation 수준에서 병목을 제거하려고 합니다. 이런 점에서 매우 engineering-aware한 연구입니다.

## 6. Conclusion

SparseInst는 실시간 instance segmentation을 위해 **instance activation map(IAM)** 이라는 새로운 object representation을 제안한 논문입니다. 이 방법은 box나 center 대신 sparse activation map으로 객체의 중요한 영역을 강조하고, 그 영역에서 집계한 instance-level feature로 분류와 segmentation을 수행합니다. Hungarian matching을 통해 one-to-one prediction을 학습하므로, inference에서 **NMS 없이 빠른 처리**가 가능합니다. 실제로 COCO에서 **37.9 AP와 40 FPS**, 더 작은 입력에서는 **58.5 FPS**를 달성해 매우 강한 real-time 성능을 보입니다.

이 논문의 실질적 기여는 “더 빠른 모델 하나”를 넘습니다. object representation을 box/center에서 activation map으로 옮기고, sparse prediction과 end-to-end assignment를 결합해 instance segmentation 파이프라인을 단순화했다는 점에서 의미가 큽니다. 이후 real-time instance segmentation이나 lightweight end-to-end segmentation 연구를 이해할 때 중요한 기준점이 되는 논문입니다.
