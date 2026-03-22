# TensorMask: A Foundation for Dense Object Segmentation

* **저자**: Xinlei Chen, Ross Girshick, Kaiming He, Piotr Dollár
* **발표연도**: 2019
* **arXiv**: [https://arxiv.org/abs/1903.12174](https://arxiv.org/abs/1903.12174)

## 1. 논문 개요

이 논문은 instance segmentation을 위한 **dense sliding-window** 패러다임을 본격적으로 탐구한 연구다. 당시 instance segmentation의 주류는 Mask R-CNN처럼 먼저 bounding box를 검출한 뒤, 그 영역을 crop해서 mask를 예측하는 **detect-then-segment** 방식이었다. 반면 object detection에서는 SSD나 RetinaNet처럼 refinement 단계 없이 조밀한 위치에서 직접 예측하는 dense 방식이 크게 발전했지만, instance segmentation에서는 이에 대응하는 강력한 dense 방법이 거의 없었다. 이 논문은 바로 그 공백을 메우려는 시도다.

저자들의 핵심 문제의식은 다음과 같다. box detection에서는 각 위치에서 예측해야 하는 출력이 4개의 수치로 비교적 단순하다. 하지만 instance mask는 각 위치에서 출력해야 하는 대상이 **그 자체로 2차원 공간 구조를 가진 mask**다. 즉, dense prediction이라고 해도 semantic segmentation처럼 “픽셀마다 클래스 하나를 붙이는 문제”와는 다르고, box detection처럼 “위치마다 저차원 벡터를 붙이는 문제”와도 다르다. 따라서 instance segmentation을 dense 방식으로 제대로 다루려면, mask를 단순한 채널 벡터로 취급하는 것이 아니라 **공간 구조를 가진 객체**로 다루는 새로운 표현이 필요하다는 것이 논문의 출발점이다.

이 문제는 중요하다. 왜냐하면 dense 예측은 convolution과 자연스럽게 결합되고, region crop 기반 방법과는 다른 계산 및 표현 특성을 가지기 때문이다. 만약 dense sliding-window 방식이 instance segmentation에서도 성립한다면, object detection과 segmentation 사이의 개념적 간극을 좁히고, 장기적으로는 region proposal이나 RoI crop에 의존하지 않는 새로운 segmentation 계열 모델의 기반이 될 수 있다. 저자들은 이를 위해 **TensorMask**라는 프레임워크를 제안하며, dense instance segmentation을 **4차원 tensor 위의 예측 문제**로 정식화한다.

## 2. 핵심 아이디어

TensorMask의 중심 아이디어는 간단하지만 매우 본질적이다. 각 spatial location에서 예측되는 출력 mask를 더 이상 단순한 channel 묶음으로 보지 않고, **$(V,U,H,W)$ 형태의 구조화된 4D tensor**로 본다. 여기서 $(H,W)$는 이미지 상에서 sliding window가 놓이는 위치 축이고, $(V,U)$는 각 윈도우 내부에서 mask를 표현하는 상대 좌표 축이다. 다시 말해, $(H,W)$는 “어디에서 예측하느냐”를 나타내고, $(V,U)$는 “그 위치에서 예측되는 mask의 내부 구조”를 나타낸다.

이 관점이 중요한 이유는 $(V,U)$와 $(H,W)$가 모두 기하학적 의미를 가지기 때문이다. 기존 DeepMask나 InstanceFCN 계열은 mask를 channel 차원에 펼쳐 넣는 식의 3D 표현을 사용했는데, channel 축은 그 자체로 공간적인 의미가 없다. 그래서 upsampling, coordinate transform, multi-scale 조작 같은 연산을 구조적으로 설계하기 어렵다. 반면 TensorMask는 mask 자체를 2D spatial object로 보므로, $(V,U)$ 축에 대해 bilinear interpolation을 적용하거나, representation을 변환하거나, pyramid 구조를 설계하는 것이 자연스러워진다.

기존 접근과의 가장 큰 차별점은 두 가지다. 첫째, dense mask prediction을 위한 **표현 자체를 재정의**했다는 점이다. 둘째, large object는 더 큰 mask 해상도를 필요로 하고 small object는 작은 mask 해상도로 충분하다는 직관을 반영해, feature pyramid와는 반대 방향으로 확장되는 **tensor bipyramid**를 설계했다는 점이다. 일반적인 feature pyramid는 큰 물체를 더 낮은 spatial resolution의 feature map에서 처리한다. TensorMask는 여기에 더해, 큰 물체일수록 $(V,U)$ mask grid를 더 크게 만들어 높은 mask 해상도를 제공한다. 즉, spatial localization 해상도는 내려가지만 mask 자체의 표현 해상도는 올라간다. 이것이 dense sliding-window segmentation에서 매우 자연스러운 multi-scale 설계라는 것이 논문의 주장이다.

## 3. 상세 방법 설명

### 3.1 마스크를 4차원 텐서로 표현하는 방식

논문은 먼저 모든 sliding-window mask를 표현하는 기본 구조로 $(V,U,H,W)$ tensor를 정의한다. 여기서 하나의 $(y,x)$ 위치에 놓인 sub-tensor $(V,U)$는 해당 위치의 window 안에서 예측되는 mask를 뜻한다. 이때 중요한 개념이 각 축의 **unit of length**다.

* $(H,W)$ 축의 단위는 $\sigma_{HW}$이며, 이는 feature map stride에 해당한다.
* $(V,U)$ 축의 단위는 $\sigma_{VU}$이며, mask 내부에서 한 칸 움직이는 것이 원본 이미지에서 몇 픽셀에 해당하는지를 나타낸다.

이 두 단위의 비율을 다음과 같이 둔다.

$$
\alpha = \frac{\sigma_{VU}}{\sigma_{HW}}
$$

이 정의가 필요한 이유는 같은 $(V,U,H,W)$ 모양의 tensor라도, $\sigma_{VU}$가 1인지 2인지에 따라 실제 이미지에서 mask가 덮는 영역의 크기가 달라지기 때문이다. 즉, tensor shape만으로는 충분하지 않고 각 축의 물리적 의미가 함께 정의되어야 한다.

### 3.2 Natural representation

가장 직관적인 표현은 **natural representation**이다. 이 표현에서는 tensor 값 $\mathcal{F}(v,u,y,x)$가, 중심이 $(y,x)$인 sliding window 안에서 상대 위치 $(v,u)$에 해당하는 mask 값을 뜻한다. 논문은 이를 다음처럼 설명한다.

$$
\mathcal{F}(v,u,y,x) \text{ 는 중심 } (y,x) \text{ 인 window 안의 } (y+\alpha v, x+\alpha u) \text{ 위치의 mask 값}
$$

즉 $(y,x)$에서 하나의 $(V,U)$ mask를 직접 읽는 방식이다. 이 표현은 해석하기 쉽지만, convolution으로 예측할 때 항상 픽셀 정렬이 자연스럽게 보장되는 것은 아니다.

### 3.3 Aligned representation

이를 보완하기 위해 저자들은 **aligned representation**을 제안한다. 이 표현에서는 각 $(\hat{y}, \hat{x})$ 위치에서 sub-tensor $(\hat{V},\hat{U})$가 “이 픽셀을 포함하는 여러 윈도우들에서의 값”을 모은 형태가 된다. 다시 말해, natural representation은 “한 위치에서 하나의 mask 전체를 읽는 방식”이고, aligned representation은 “한 픽셀 위치에 대해 그것을 포함하는 여러 윈도우의 값을 정렬해 저장하는 방식”이다.

이 표현의 장점은 convolution 기반 feature 계산에서 **pixel-to-pixel alignment**를 유지하기 쉽다는 점이다. 저자들은 이것을 RoIAlign의 동기와 유사하다고 설명한다. 실제 실험에서도 aligned representation은 특히 upscaling이 들어갈 때 큰 성능 향상을 보인다.

### 3.4 natural과 aligned 사이의 좌표 변환

두 표현을 같은 네트워크 안에서 함께 쓰기 위해 논문은 **coordinate transformation**을 정의한다. 단위가 같다고 가정하면, aligned representation $\hat{\mathcal{F}}$를 natural representation $\mathcal{F}$로 바꾸는 변환은 다음과 같다.

$$
\mathcal{F}(v,u,y,x)=\hat{\mathcal{F}}(v,u,y+\alpha v, x+\alpha u)
$$

이 연산을 논문은 **align2nat**라고 부른다. 반대로 natural에서 aligned로 바꾸는 변환도 정의한다. 핵심은 이 변환을 통해 중간 layer에서는 aligned representation처럼 학습 친화적인 구조를 쓰고, 최종 출력은 natural representation으로 통일할 수 있다는 점이다. 저자들은 실제 모델들에서 출력 포맷을 항상 natural representation으로 고정해, loss 정의와 네트워크 내부 설계를 분리한다.

### 3.5 up_align2nat: coarse한 표현에서 fine mask 생성

TensorMask의 중요한 장점 중 하나는 mask 내부 축 $(V,U)$ 자체를 upsample할 수 있다는 것이다. 논문은 이를 위해 **up_align2nat** 연산을 도입한다. 이 연산은 coarse한 aligned tensor를 먼저 $(V,U)$ 방향으로 bilinear upsampling한 후, align2nat를 적용해 natural representation으로 바꾼다.

이것이 중요한 이유는, 큰 해상도의 mask를 직접 예측하려면 preceding feature map에서 엄청난 수의 output channel이 필요하기 때문이다. 예를 들어 $15 \times 15$ mask를 직접 예측하면 225개 채널이 필요하지만, 이를 먼저 저해상도에서 예측하고 나중에 upsample하면 훨씬 적은 채널로도 고해상도 mask를 만들 수 있다. 논문 실험에서 이 방식은 특히 large upscaling factor일 때 natural upscaling보다 훨씬 안정적이고 정확하다.

### 3.6 Tensor bipyramid

이 논문의 가장 대표적인 구조적 기여는 **tensor bipyramid**다. 일반적인 feature pyramid는 level이 올라갈수록 feature map spatial resolution이 감소한다. TensorMask는 여기에 더해 level이 올라갈수록 mask sub-tensor 크기 $(V,U)$는 오히려 증가하게 만든다.

논문은 tensor bipyramid를 다음 shape의 tensor 리스트로 정의한다.

$$
(2^kV, 2^kU, \frac{1}{2^k}H, \frac{1}{2^k}W), \quad k=0,1,2,\dots
$$

이 구조의 의미는 명확하다.

* 큰 $k$일수록 sliding window가 놓이는 위치 해상도 $(H,W)$는 거칠어진다.
* 대신 각 위치에서 예측되는 mask 해상도 $(V,U)$는 더 커진다.

그래서 큰 물체는 coarse한 위치 격자 위에서 예측되지만, 그 대신 **더 높은 해상도의 mask**를 갖는다. 이는 large object segmentation에 자연스럽다.

이를 효율적으로 구현하기 위해 저자들은 **swap_align2nat** 연산을 정의한다. 개념적으로는 up_align2nat를 통해 $(V,U)$를 키운 뒤, $(H,W)$를 subsample해서 단위를 뒤바꾸는 형태다. 논문은 이 연산을 중간 giant tensor를 실제로 만들지 않고, 필요한 값만 계산하는 방식으로 구현해 복잡도를 $O(V \cdot U \cdot H \cdot W)$로 유지한다고 설명한다.

### 3.7 전체 아키텍처

TensorMask는 backbone으로 FPN을 사용한다. 각 FPN level에서 classification head, box head, mask head를 둔다. 이 구조는 RetinaNet과 매우 유사하지만, box 대신 dense mask prediction이 추가되었다는 점이 다르다. 분류와 box 회귀는 dense detector의 그것과 유사하고, mask 출력은 class-agnostic이다. 즉 mask head는 클래스 수만큼의 mask를 만들지 않고, 각 윈도우에 대해 하나의 mask만 예측한다. 클래스는 별도의 classification head가 담당한다.

논문은 네 가지 baseline head를 비교한다.

1. simple natural head
2. simple aligned head
3. upscaling natural head
4. upscaling aligned head

이 중 가장 강력한 최종 구조는 baseline feature pyramid 위의 head가 아니라, FPN을 수정해 모든 level에서 같은 $(H,W)$ 해상도를 갖도록 맞춘 뒤 tensor bipyramid head를 사용하는 방식이다. 이 구조가 TensorMask의 대표 모델이다.

### 3.8 학습 방법

#### 라벨 할당

논문은 box detector의 IoU 기반 anchor assignment 대신 **mask-driven assignment rule**을 사용한다. 어떤 sliding window가 ground-truth mask $m$에 대해 positive가 되려면 세 조건을 만족해야 한다.

첫째, **containment**: window가 mask를 완전히 포함해야 하고, mask의 긴 변이 window 긴 변의 절반 이상이어야 한다.
둘째, **centrality**: mask bounding box의 중심이 window 중심에서 $\sigma_{VU}$ 이내의 $\ell_2$ 거리 안에 있어야 한다.
셋째, **uniqueness**: 같은 window에 대해 다른 mask가 위 두 조건을 동시에 만족하면 안 된다.

이 규칙은 box anchor IoU와 달리 mask의 공간적 적합성에 직접 기반한다. 저자들은 이 규칙이 한두 개의 window size만으로도 잘 작동한다고 보고한다.

#### 손실 함수

mask head에는 **per-pixel binary classification loss**, 즉 binary cross-entropy를 사용한다. foreground/background 불균형을 줄이기 위해 foreground pixel 가중치를 1.5로 둔다. 각 positive window의 mask loss는 윈도우 내 전체 픽셀 평균이고, 전체 mask loss는 positive window들에 대해 평균한다.

classification head에는 focal loss를 사용하며, 파라미터는 $\gamma=3$, $\alpha=0.3$이다.
box regression에는 parameter-free $\ell_1$ loss를 사용한다.
전체 loss는 이 세 항의 가중합이다.

#### 구현 세부사항

FPN의 각 level은 4개의 $3 \times 3$ convolution + ReLU로 출력하며, top-down과 lateral connection은 합(sum) 대신 평균(avg)을 사용해 안정성을 높였다고 한다. 학습은 ImageNet pretrained initialization을 사용하고, shorter side를 $[640,800]$ 범위에서 랜덤 샘플링하는 scale jitter를 적용한다. 스케줄은 논문 기준 6× schedule, 약 72 epoch다. 배치 크기는 8 GPU에서 총 16장이다.

### 3.9 추론

추론 시에는 짧은 변 800 픽셀 단일 스케일을 사용한다. 각 sliding window에 대해 mask, class score, box를 출력하고, regressed box에 대한 IoU 기반 NMS를 적용한다. soft mask를 최종 binary mask로 바꾸는 방법은 Mask R-CNN과 동일한 후처리를 따른다고 명시한다.

## 4. 실험 및 결과

### 4.1 실험 설정

평가는 COCO instance segmentation에서 수행했다. 학습은 train2017 약 118k 이미지, 검증은 val2017 5k 이미지, 최종 비교는 test-dev에서 보고했다. 주요 지표는 COCO mask AP이며, box 결과는 $AP^{bb}$로 따로 표기했다.

### 4.2 simple head에서 natural vs aligned

Table 1에 따르면, upscaling이 없는 simple head에서는 natural과 aligned의 차이가 크지 않다.

* natural: AP 28.5
* aligned: AP 28.9

차이는 0.4 AP 수준이다. 즉 representation 차이가 항상 큰 효과를 내는 것은 아니다. 하지만 이는 “고해상도 mask를 만들기 위한 upscaling이 없을 때”에 한정된다.

### 4.3 upscaling head에서 aligned의 중요성

가장 인상적인 결과는 Table 2(a)다. 출력 mask는 항상 $15 \times 15$로 맞추되, 중간 표현을 얼마나 거칠게 둘지에 따라 upscaling factor $\lambda$를 바꿨다.

* $\lambda=1.5$: natural 28.0, aligned 28.9
* $\lambda=3$: natural 24.7, aligned 28.8
* $\lambda=5$: natural 19.2, aligned 28.4

특히 $\lambda=5$에서 aligned는 natural보다 **+9.2 AP** 높다. 이는 단순한 미세 개선이 아니라, natural upscaling이 큰 스케일 확대에서 거의 무너지는 반면 aligned upscaling은 안정적으로 작동함을 보여준다. 논문은 시각화에서도 natural upscaling이 coarse mask를 내고, aligned upscaling이 sharp한 mask를 유지한다고 보여준다.

이 결과는 TensorMask의 핵심 주장을 강하게 뒷받침한다. 즉 dense mask prediction에서는 단순한 채널 기반 upsampling이 아니라, **기하학적으로 정렬된 mask 표현**이 필수적이라는 것이다.

### 4.4 bilinear interpolation의 효과

Table 2(b)는 aligned upscaling head에서 bilinear와 nearest-neighbor interpolation을 비교한다.

* $\lambda=1.5$: bilinear가 +0.3 AP
* $\lambda=3$: bilinear가 +1.0 AP
* $\lambda=5$: bilinear가 +3.1 AP

확대 배율이 커질수록 bilinear의 이점이 커진다. 논문은 nearest-neighbor 기반 방식이 사실상 InstanceFCN과 유사한 동작을 한다고 appendix에서 연결한다. 특히 겹치는 객체가 있을 때 nearest-neighbor는 심한 artifact를 만들고, bilinear는 훨씬 안정적이다. 이는 TensorMask가 단순히 4D tensor를 쓰는 것에 그치지 않고, 그 위에서 **어떤 연산을 하느냐**가 성능에 직결된다는 점을 보여준다.

### 4.5 tensor bipyramid의 효과

Table 2(c)는 best baseline feature pyramid와 tensor bipyramid를 비교한다.

* feature pyramid, best: AP 28.9
* tensor bipyramid: AP 34.0

무려 **+5.1 AP** 상승이다. 세부적으로 보면:

* $AP_{50}$: 52.5 → 55.2
* $AP_{75}$: 29.3 → 35.8
* $AP_S$: 14.6 → 15.3
* $AP_M$: 30.8 → 36.3
* $AP_L$: 40.7 → 48.4

특히 large object 성능 $AP_L$이 **+7.7** 상승한 것이 매우 중요하다. 이는 tensor bipyramid가 큰 물체에서 더 높은 해상도의 mask를 예측하도록 설계된 목적과 정확히 일치한다. 즉 구조 설계의 효과가 지표에 직접 반영되었다.

### 4.6 multiple window sizes

Table 2(d)는 각 level에서 하나의 window size만 쓰는 대신 두 개의 window size를 사용하는 경우를 본다.

* $15 \times 15$만 사용: AP 34.0
* $15 \times 15$와 $11 \times 11$ 사용: AP 35.2

즉 **+1.2 AP** 개선이다. small, medium, large 전 영역에서 고르게 향상된다. 이는 dense detector에서 여러 anchor scale/aspect ratio를 쓰는 것과 유사한 이득이 segmentation에도 있다는 점을 보여준다.

### 4.7 Mask R-CNN과의 비교

Table 3은 TensorMask의 최종 성능을 Mask R-CNN과 비교한다.

ResNet-50-FPN, augmentation 포함, 72 epoch 기준:

* Mask R-CNN: AP 36.8
* TensorMask: AP 35.4

ResNet-101-FPN 기준:

* Mask R-CNN: AP 38.3
* TensorMask: AP 37.1

즉 gap은 각각 1.4 AP, 1.2 AP다. 당시 주류이자 매우 강력한 Mask R-CNN에 비해 dense sliding-window 방식이 이 정도까지 접근했다는 것이 논문의 핵심 성과다. 저자들의 주장처럼, 이것은 dense instance segmentation이 “원리적으로 가능하다”는 것을 실험적으로 입증한 결과라고 볼 수 있다.

### 4.8 속도와 box detection 결과

논문은 속도 한계도 솔직히 보고한다. 최종 ResNet-101-FPN TensorMask는 V100에서 0.38초/이미지인데, Mask R-CNN은 0.09초/이미지다. dense sliding window에서 10만 개가 넘는 위치에 mask를 예측해야 하므로 계산량이 큰 것은 자연스러운 결과다. 저자들도 가속화는 가능하지만 본 논문의 범위 밖이라고 명시한다.

Appendix의 box detection 결과(Table 4)를 보면 TensorMask는 box AP에서도 꽤 경쟁력 있다.

* RetinaNet, ours, 72 epochs: $AP^{bb}=39.3$
* Faster R-CNN, ours: $40.6$
* Mask R-CNN, ours: $41.7$
* TensorMask, box-only: $40.8$
* TensorMask: $41.6$

즉 detection box 성능도 Mask R-CNN에 매우 근접한다. 이는 TensorMask가 단지 mask만 잘 만드는 것이 아니라, dense detection 관점에서도 상당히 강력하다는 점을 시사한다.

### 4.9 mask-only 분석

Appendix Table 5는 box head가 mask 성능에 얼마나 기여하는지도 분석한다. 결과를 보면 box head를 제거해도 mask AP는 거의 유지된다.

* 기본(box head + box NMS): AP 35.2
* box head + mask-bb NMS: AP 34.9
* no box head + mask-bb NMS: AP 34.8

즉 box는 TensorMask에서 보조적인 역할일 뿐이며, mask 예측 자체는 box에 강하게 의존하지 않는다. 이는 detect-then-segment와 철학적으로 다른 지점이다. Mask R-CNN은 box가 사실상 핵심 입력이지만, TensorMask는 mask를 더 독립적인 1차 예측 대상으로 다룬다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제를 새롭게 정의했다는 점**이다. 많은 연구가 더 강한 backbone, 더 복잡한 head, 더 많은 trick에 집중하는 반면, TensorMask는 “dense mask prediction의 본질적 표현은 무엇인가”를 정면으로 다룬다. 특히 instance mask를 구조화된 4D tensor로 보고, $(V,U)$와 $(H,W)$를 모두 geometric axis로 취급한 것은 개념적으로 매우 깔끔하다. 이 덕분에 aligned representation, coordinate transform, tensor bipyramid 같은 연산과 구조가 자연스럽게 도출된다.

두 번째 강점은 **실험 설계가 논지와 잘 연결되어 있다는 점**이다. 단순히 최종 성능만 보고하는 것이 아니라, natural vs aligned, bilinear vs nearest, feature pyramid vs tensor bipyramid처럼 각 개념 요소를 단계적으로 검증한다. 특히 upscaling에서 aligned representation이 왜 중요한지, tensor bipyramid가 왜 large object에 유리한지를 수치적으로 설득력 있게 보여준다.

세 번째 강점은 **주류 방법과 다른 패러다임을 실제 경쟁 가능한 수준까지 끌어올렸다는 점**이다. 최종적으로 Mask R-CNN보다 다소 낮지만, 성능 차이가 1~1.5 AP 수준으로 줄어들었다는 것은 dense sliding-window instance segmentation이 단순한 아이디어 수준이 아니라 실용적 연구 방향이 될 수 있음을 보여준다.

반면 한계도 분명하다. 첫째, **속도와 계산량 문제**가 크다. 논문 스스로도 인정하듯이 dense mask prediction은 sliding window 수가 매우 많아 후처리 포함 계산 비용이 크다. Mask R-CNN이 sparse한 최종 box에 대해서만 mask를 예측하는 것과 비교하면 구조적으로 불리하다.

둘째, 최종 성능이 Mask R-CNN을 넘지는 못했다. 즉 이 논문은 dense 방식의 가능성을 입증했지만, 당시 최강의 detect-then-segment 패러다임을 대체했다고 보기는 어렵다. 논문의 표현대로 “close to”이지 “better than”은 아니다.

셋째, 라벨 할당 규칙과 multi-scale 설계가 여전히 손으로 설계된 부분이 많다. containment, centrality, uniqueness 규칙은 합리적이지만, 얼마나 일반적인지 또는 다른 데이터셋과 객체 분포에서도 최적인지는 본문만으로는 알 수 없다. 또한 multiple window sizes와 aspect ratio 확장은 여지가 있다고 했지만, 그 탐색은 충분히 진행되지 않았다.

넷째, 논문은 mask representation의 구조를 강조하지만, 이 표현이 이후 더 복잡한 transformer 기반 구조나 end-to-end set prediction과 어떻게 결합될 수 있는지는 다루지 않는다. 물론 이는 발표 시점을 고려하면 자연스럽지만, 후속 연구 관점에서는 열린 질문이다.

비판적으로 보면, TensorMask의 가장 큰 가치는 당장 최고 성능 모델을 만든 데 있다기보다, **instance segmentation을 dense prediction 관점에서 다시 생각할 수 있게 만든 이론적·구조적 프레임**을 제시한 데 있다. 따라서 이 논문은 “완성형 솔루션”이라기보다 “새로운 기반(foundation)”이라는 제목이 더 잘 맞는다.

## 6. 결론

TensorMask는 dense sliding-window instance segmentation을 위한 최초의 체계적인 프레임워크로서, instance mask를 $(V,U,H,W)$의 **구조화된 4D tensor**로 표현한다. 이 표현을 바탕으로 natural/aligned representation, align2nat 변환, up_align2nat, tensor bipyramid 같은 핵심 개념과 연산을 제안했고, 이들이 실제 COCO 실험에서 큰 성능 향상을 만든다는 것을 보여주었다.

특히 tensor bipyramid는 큰 객체일수록 더 고해상도의 mask를 예측하도록 설계되어, dense segmentation에서 multi-scale mask representation의 핵심 원리를 잘 구현한다. 그 결과 TensorMask는 Mask R-CNN과 정량·정성적으로 유사한 수준의 성능에 도달하며, dense sliding-window 방식도 instance segmentation에서 충분히 유효한 연구 방향임을 입증했다.

실제 적용 관점에서는 당시 기준으로 계산량 부담이 크기 때문에 바로 주류가 되기에는 어려움이 있었다. 그러나 연구적 의미는 매우 크다. 이 논문은 instance segmentation에서 “mask를 어떻게 표현할 것인가”라는 더 근본적인 질문을 던졌고, region-based 방법과는 다른 대안을 제공했다. 향후 더 효율적인 dense mask predictor, 더 나은 assignment 전략, 혹은 다른 backbone 및 sequence/set prediction 구조와 결합한다면, TensorMask의 아이디어는 후속 연구의 중요한 출발점이 될 가능성이 높다.
