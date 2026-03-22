# SOLO: A Simple Framework for Instance Segmentation

이 논문은 instance segmentation을 기존의 **top-down(detect-then-segment)** 또는 **bottom-up(grouping/embedding)** 관점이 아니라, **위치(location) 기반의 직접 분할 문제**로 다시 정의합니다. 저자들은 이미지 안의 대부분의 인스턴스가 서로 **중심 위치가 다르거나 크기가 다르다**는 관찰에서 출발해, 각 객체를 “어느 grid cell에 속한 인스턴스인가”라는 **instance category**로 분류할 수 있다고 봅니다. 이를 바탕으로 제안한 **SOLO (Segmenting Objects by Locations)**는 bounding box, anchor, pixel embedding grouping 없이도 입력 이미지를 바로 **semantic category + instance mask**로 매핑하는 단순한 fully convolutional framework입니다. 논문은 Vanilla SOLO, Decoupled SOLO, Dynamic SOLO(SOLOv2)라는 변형들을 제시하고, Matrix NMS까지 포함해 COCO에서 속도와 정확도 모두에서 강한 결과를 보고합니다.

## 1. Paper Overview

이 논문이 해결하려는 핵심 문제는 instance segmentation이 왜 object detection보다 훨씬 복잡한가입니다. detection은 bounding box 하나로 객체 위치를 표현할 수 있지만, instance segmentation은 **픽셀 수준에서 각 객체의 불규칙한 경계**를 정확히 분리해야 합니다. 기존 주류 방법은 두 계열로 나뉩니다. 하나는 Mask R-CNN처럼 먼저 box를 찾고 그 안에서 마스크를 자르는 top-down 방식이고, 다른 하나는 픽셀마다 embedding이나 affinity를 학습한 뒤 clustering/grouping으로 인스턴스를 복원하는 bottom-up 방식입니다. 저자들은 두 방식 모두 **간접적이고 단계적**이라서, 정확한 box 또는 후처리에 과하게 의존한다고 지적합니다.

논문은 여기서 더 근본적인 질문을 던집니다. “이미지 안의 인스턴스들을 구분하는 가장 본질적인 차이는 무엇인가?” COCO val 분석에 따르면 객체 쌍의 **98.3%**는 중심 거리 차이가 30픽셀보다 크고, 나머지 가까운 쌍도 상당수는 크기 비율이 다릅니다. 즉 대부분의 인스턴스는 **위치와 크기**만으로도 충분히 분리될 수 있다는 것입니다. SOLO는 바로 이 통찰을 기반으로, instance segmentation을 **위치로 객체를 구별하는 per-pixel classification 문제**로 바꿉니다.

이 문제가 중요한 이유는, 이 재정식화가 성공하면 instance segmentation을 detection의 부속 문제가 아니라 **독립적이고 직접적인 dense prediction 문제**로 풀 수 있기 때문입니다. 논문은 실제로 이렇게 하면 box detection도, grouping post-processing도, anchor design도 제거할 수 있으며, 그 결과 더 단순하면서도 빠른 시스템이 가능하다고 주장합니다.

## 2. Core Idea

논문의 핵심 아이디어는 다음과 같습니다.

> **인스턴스를 “semantic class”가 아니라 “위치가 양자화된 instance category”로 보자.**

구체적으로 입력 이미지를 $S \times S$ grid로 나누고, 어떤 객체의 중심이 특정 grid cell 안에 떨어지면 **그 cell이 그 객체의 semantic class와 mask를 책임지는 방식**입니다. 즉 semantic segmentation이 “각 픽셀이 어떤 클래스인가”를 채널별로 예측하듯, SOLO는 “각 픽셀이 어느 위치의 인스턴스인가”를 채널 구조로 인코딩합니다. 이때 객체 크기는 FPN level로 분리해 처리합니다. 결국 인스턴스는 **(위치 category, scale level)** 조합으로 구별됩니다.

이 관점이 새로운 이유는, 기존의 direct instance segmentation조차 대개 point proposal을 먼저 뽑거나, polar contour 같은 타협된 파라미터 표현을 쓰거나, 후처리를 필요로 했기 때문입니다. SOLO는 full mask annotation만으로 학습하고, 예측 시에도 **그룹핑 없이 바로 instance mask와 class probability를 출력**하려고 합니다. 저자들은 이를 fully convolutional, box-free, grouping-free paradigm으로 제시합니다.

또 하나의 중요한 아이디어는 SOLO가 하나의 모델이 아니라 **설계 원리**라는 점입니다. 논문은 같은 원리를 따라:

* **Vanilla SOLO**
* **Decoupled SOLO**
* **Dynamic SOLO (SOLOv2)**

를 제안합니다. 즉 핵심은 “위치 기반 instance category”라는 formulation이고, 각 변형은 mask head 설계를 더 효율적이거나 유연하게 바꾸는 방향입니다.  

## 3. Detailed Method Explanation

### 3.1 Problem formulation: instance category

SOLO는 입력 이미지를 개념적으로 $S \times S$ grid로 나눕니다. 객체 중심이 어떤 grid cell에 속하면, 그 grid cell이 해당 객체의 예측을 담당합니다. semantic branch는 각 grid에 대해 $C$차원 class probability를 내고, mask branch는 그 grid에 대응하는 instance mask를 냅니다. 이 방식의 장점은 **임의 개수의 인스턴스**를, 고정된 수의 채널과 grid 구조로 다룰 수 있다는 것입니다. 즉 location prediction을 regression이 아니라 **classification**으로 바꿔, varying number of instances 문제를 fixed channel problem으로 환원합니다.

또한 객체 크기는 FPN을 이용해 분리합니다. 위치만으로는 중심이 가까운 큰/작은 객체를 완전히 분리하기 어려울 수 있으므로, 서로 다른 scale의 객체를 서로 다른 pyramid level로 할당해 regularize합니다. 이로써 논문이 말하는 “instance categories”는 사실상 **quantized center location + object size**의 조합입니다.

### 3.2 Semantic category branch

각 grid cell은 하나의 semantic class를 예측합니다. 출력 공간은 $S \times S \times C$입니다. 저자들의 핵심 가정은 각 grid cell이 하나의 인스턴스만 책임질 수 있다는 것입니다. 이건 엄밀히 말하면 근사이지만, 앞서 말한 중심 위치와 크기 분포 관찰 덕분에 실용적으로 충분히 작동한다고 봅니다.

### 3.3 Instance mask branch

mask branch는 같은 grid 분할 아래에서 최대 $S^2$개의 mask를 예측할 수 있도록 설계됩니다. 각 채널이 하나의 grid 위치에 대응하며, 해당 채널 map이 그 위치 인스턴스의 class-agnostic mask를 의미합니다. 즉 semantic branch와 mask branch 사이에는 **one-to-one correspondence**가 성립합니다. 이 점이 SOLO 구조를 매우 단순하게 만듭니다. 별도의 ROI crop이나 mask coefficient 조합이 필요 없습니다.

하지만 일반 FCN은 spatially invariant 성질이 강하므로, “이 마스크가 어느 grid cell의 인스턴스를 위한 것인가”를 구별하기 어렵습니다. 이를 해결하기 위해 SOLO는 mask branch 입력에 **정규화된 x, y 좌표 채널을 추가하는 CoordConv 스타일 설계**를 사용합니다. 논문 ablation에서는 이 coordinate channel이 **3.6 AP 절대 향상**을 준다고 보고합니다. 즉 위치 기반 formulation을 실제로 학습 가능하게 만드는 핵심 장치 중 하나가 explicit coordinate encoding입니다.  

### 3.4 Vanilla SOLO

Vanilla SOLO는 가장 직접적인 구현입니다. 각 FPN level에 두 개의 head를 둡니다.

* instance category prediction head
* instance mask prediction head

mask head는 feature와 좌표 채널을 concat하여 spatially variant prediction을 수행합니다. 이 구조는 개념적으로 가장 단순하지만, $S^2$ 채널 구조 자체가 다소 무겁고, mask prediction과 location encoding이 강하게 얽혀 있습니다.

### 3.5 Decoupled SOLO

Decoupled SOLO는 mask prediction을 좀 더 효율적으로 만들기 위해 x축과 y축을 분리하는 방식으로 확장된 변형입니다. 제공된 snippet에서는 전체 세부식이 전부 보이지 않지만, 논문 도식은 Decoupled SOLO가 Vanilla보다 더 parameter-efficient한 방향으로 channel coupling을 완화하는 변형임을 보여줍니다. 핵심은 같은 “location-based instance category” 원리를 유지하면서 mask head 구조를 더 잘 분해해 학습과 계산을 개선하는 것입니다.

### 3.6 Dynamic SOLO / SOLOv2

논문이 “enhanced version”으로 제시하는 Dynamic SOLO, 즉 **SOLOv2**는 원래 mask prediction을 **kernel learning + feature learning**으로 decouple합니다. 이는 YOLACT나 CondInst와 일부 유사해 보이지만, 저자들은 SOLOv2의 차별점을 분명히 말합니다. CondInst가 인스턴스마다 **relative position**을 반복적으로 encode해야 하는 반면, SOLOv2는 SOLO의 절대 위치 개념을 유지하므로 **global coordinates를 한 번에 사용**합니다. 즉 인스턴스 수 $N$에 비례해 좌표 인코딩을 반복할 필요가 없습니다.  

이 설계는 inference를 더 단순하게 만듭니다. Dynamic SOLO에서는 predicted mask kernel을 mask feature에 convolution하여 soft mask를 만들고, sigmoid 뒤 threshold 0.5로 이진화합니다. 그 후 Matrix NMS를 적용합니다.

### 3.7 Learning objective

mask branch loss에 대한 ablation에서 논문은 BCE, Focal Loss, Dice Loss를 비교합니다. 결과적으로 **Dice Loss가 가장 좋았고**, 별도의 복잡한 weight tuning 없이 foreground/background imbalance를 자동으로 다루는 장점이 있다고 설명합니다. 이는 instance mask가 본질적으로 sparse binary object이기 때문에, 픽셀 독립 손실보다 object-level overlap 관점이 더 잘 맞는다는 해석이 가능합니다.

### 3.8 Matrix NMS

SOLO의 또 다른 실질적 기여는 **Matrix NMS**입니다. mask NMS는 box NMS보다 훨씬 비싼데, mask pair IoU 계산과 sequential suppression이 큰 병목이 되기 때문입니다. Matrix NMS는 이를 **병렬 행렬 연산으로 한 번에 처리**합니다. 논문은 simple Python implementation에서도 **500 masks를 1ms 미만**에 처리하고, Fast NMS보다 **0.4 AP** 더 좋다고 보고합니다. SOLO가 “빠른 direct instance segmentation”이라는 주장을 가능하게 해 주는 핵심 보조 기술입니다.  

## 4. Experiments and Findings

### 4.1 COCO 성능

논문은 COCO에서 ResNet-50 backbone 기준 **38.8 mask AP at 18 FPS**를 보고합니다. 또한 경량 버전은 **31.3 FPS에서 37.1 mask AP**를 낸다고 요약합니다. 이는 당시 기준으로 speed-accuracy tradeoff가 매우 강력하다는 메시지입니다. 특히 이 결과가 box-free, grouping-free 구조에서 나온다는 점이 중요합니다.  

추가로 table snippet에서는 SOLOv2가 Res-50-FPN에서 **37.4**, Res-101-FPN에서 **38.0** 같은 수치를 보이며 strong baseline임을 확인할 수 있습니다. 다만 이 부분 snippet만으로는 해당 표의 정확한 평가 setting 전체를 완전히 복원할 수는 없으므로, 숫자는 보수적으로 해석해야 합니다.

### 4.2 Detection byproduct

흥미로운 결과는 SOLO가 bounding box를 직접 예측하지 않으면서도, **예측 mask를 box로 변환한 byproduct**만으로 **44.9 box AP**를 얻었다는 점입니다. 이건 “mask를 잘 예측하면 detection은 부산물로 따라온다”는 논문의 철학을 뒷받침합니다. 즉 SOLO는 detection을 거쳐 segmentation하는 것이 아니라, segmentation으로부터 detection을 파생시킵니다.

### 4.3 Panoptic segmentation과 image matting 확장

논문은 semantic branch를 추가하면 panoptic segmentation으로 자연스럽게 확장 가능하다고 말합니다. 또한 appendix에서는 SOLOv2 기반으로 instance-level image matting까지 확장한 **SOSO / Soft SOLO**를 제시합니다. 이는 SOLO가 단지 binary instance mask만 잘 만드는 모델이 아니라, **instance-aware dense prediction framework**로 일반화될 수 있음을 보여줍니다.  

### 4.4 Ablation의 핵심 메시지

ablation에서 특히 중요한 건 두 가지입니다.

첫째, **CoordConv-style 좌표 주입**이 크리티컬합니다. spatial invariance를 깨고 grid-conditioned prediction을 가능하게 해서 **3.6 AP 절대 향상**을 줍니다.

둘째, **Dice Loss**가 BCE나 Focal보다 더 잘 작동합니다. foreground/background imbalance를 자동으로 균형 있게 다룬다는 점이 이유로 제시됩니다.

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **문제 재정의가 매우 강력하고 단순하다**는 점입니다. 인스턴스를 box나 embedding clustering으로 다루지 않고, **location-conditioned category prediction**으로 바꿨습니다. 이 덕분에 box-free, grouping-free, fully convolutional pipeline이 가능해졌습니다.

둘째, 실제 엔지니어링도 좋습니다. Matrix NMS는 단순한 보조 trick처럼 보이지만, mask-based method에서 inference latency를 줄이는 데 매우 중요합니다. SOLO가 “fast and strong”로 기억되는 이유 중 하나입니다.

셋째, 확장성이 큽니다. detection byproduct, panoptic segmentation, instance-level image matting까지 같은 틀에서 연결된다는 점은 SOLO가 특정 benchmark용 설계가 아니라 더 넓은 dense perception 관점을 제시했음을 뜻합니다.  

### 한계

첫째, 이 방법은 인스턴스를 **grid cell 중심**으로 양자화하는 설계에 의존합니다. 따라서 중심이 매우 가깝거나 복잡하게 겹치는 객체들에서는 representation conflict가 생길 수 있습니다. 논문은 COCO 통계로 대부분의 경우 괜찮다고 보였지만, 구조적으로는 근사입니다.

둘째, absolute position을 강하게 쓰기 때문에 위치 민감성이 장점인 동시에 inductive bias가 됩니다. CondInst와 비교 설명에서도 저자들은 relative position이 아니라 absolute position을 쓴다고 명시합니다. 이는 계산 효율은 좋지만, 다른 conditioning 방식보다 더 rigid할 수 있습니다.

셋째, Dynamic SOLO가 매우 세련된 방향으로 발전했지만, 이후 query-based segmentation 계열과 비교하면 여전히 **grid assignment prior**를 유지합니다. 따라서 오늘 시점에서는 완성형보다는, dense direct instance segmentation의 중요한 전환점으로 보는 편이 적절합니다.

### 비판적 해석

제 해석으로 SOLO의 진짜 의의는 “instance segmentation도 semantic segmentation처럼 **고정 채널 dense prediction**으로 바꿔볼 수 있다”는 것을 설득력 있게 보여준 데 있습니다. 이건 꽤 큰 관점 전환입니다. detection 중심의 사고에서 벗어나, 위치 자체를 카테고리화해 인스턴스를 구분하겠다는 발상은 이후의 box-free segmentation 연구들에 강한 영향을 줬다고 볼 수 있습니다.

또 SOLOv2의 dynamic formulation은 CondInst와 닮은 지점이 있지만, 논문이 강조하듯 **absolute location prior를 전역적으로 사용**한다는 점이 철학적으로 다릅니다. 즉 이 논문은 “인스턴스를 상대 좌표로 매번 조건화할 필요 없이, 전역 위치 구조 안에서 한 번에 분리할 수 있다”는 쪽에 더 가깝습니다.  

## 6. Conclusion

SOLO는 instance segmentation을 **instance category classification + mask generation** 문제로 다시 정식화한, 매우 단순하면서도 영향력 큰 논문입니다. 핵심은 이미지를 grid로 나누고, 객체를 **중심 위치와 크기**로 구분해 box 없이 직접 mask와 class를 예측하는 것입니다. 이 formulation 위에서 Vanilla SOLO, Decoupled SOLO, Dynamic SOLO(SOLOv2)가 제안되었고, CoordConv와 Dice Loss, Matrix NMS 같은 설계가 실질적 성능 향상에 기여했습니다. COCO에서의 강한 AP/FPS, detection byproduct, panoptic 및 image matting 확장은 이 프레임워크의 범용성을 잘 보여줍니다.

연구적으로는 SOLO가 “instance segmentation은 꼭 detect-then-segment여야 하는가?”라는 질문에 강한 반례를 제시했다는 점이 중요합니다. 실무적으로는 간결한 one-stage, box-free mask model의 대표 사례이고, 역사적으로는 direct dense instance segmentation 흐름의 중심 논문으로 볼 수 있습니다.
