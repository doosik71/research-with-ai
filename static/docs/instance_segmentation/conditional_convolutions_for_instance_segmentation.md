# Conditional Convolutions for Instance Segmentation

## 1. Paper Overview

이 논문은 instance segmentation을 **ROI-based two-stage pipeline 없이**, 완전히 **fully convolutional**하게 해결할 수 있는가를 다룹니다. 당시 대표 방법인 Mask R-CNN은 bounding box를 먼저 찾고, 각 ROI를 crop한 뒤, 고정된 mask head로 foreground/background mask를 예측했습니다. 저자들은 이 방식이 irregular object에 대해 불필요한 배경을 많이 포함하고, per-instance ROI 처리 때문에 계산량이 instance 수에 따라 증가하며, ROI resizing 때문에 출력 해상도와 경계 품질이 제한된다는 점을 문제로 제기합니다. 이에 대해 논문은 **CondInst**를 제안합니다. 핵심은 각 instance마다 별도의 ROI를 잘라 넣는 대신, **instance에 조건부로 동적으로 생성되는 mask head**를 사용해 전체 feature map 위에서 그 instance의 마스크만 “발화”하도록 만드는 것입니다. 논문은 COCO에서 well-tuned Mask R-CNN baseline보다 **정확도와 속도 모두에서 우수**하다고 주장합니다.

이 문제가 중요한 이유는 instance segmentation이 detection보다 더 정밀한 pixel-level 구조를 필요로 하지만, 기존 강한 모델들은 대부분 ROI crop, feature alignment, per-instance head 계산 같은 복잡한 절차에 의존했기 때문입니다. CondInst는 semantic segmentation에서 성공한 FCN 패러다임을 instance segmentation으로 확장하되, 단순 FCN이 같은 class의 여러 객체를 구분하기 어려운 근본 문제를 **instance-aware filter**와 **relative coordinates**로 해결하려 합니다. 따라서 이 논문은 “instance segmentation을 정말 FCN 방식으로 할 수 있는가”에 대한 강한 대답을 제시한 작업입니다.  

## 2. Core Idea

논문의 핵심 아이디어는 다음과 같습니다.

**“모든 객체에 같은 고정 mask head를 쓰지 말고, 각 instance마다 그 객체에 맞는 작은 mask network를 동적으로 생성하자.”**

Mask R-CNN은 ROI가 곧 instance 표현입니다. 즉, 어떤 객체를 분할할지는 bounding box crop 자체가 규정합니다. 반면 CondInst는 instance를 bounding box crop이 아니라 **동적으로 생성된 mask head의 파라미터**로 표현합니다. 논문은 이 파라미터가 target instance의 상대 위치, 형태, appearance 특성을 담고 있어, 전체 feature map에 적용해도 그 객체 픽셀에만 반응하게 만든다고 설명합니다. 이 점이 기존 ROI-based 방법과의 가장 근본적인 차이입니다.

이 설계의 핵심 novelty는 두 가지입니다.

첫째, **dynamic conditional convolution**을 instance segmentation에 본격 적용했다는 점입니다. 기존 dynamic filtering이나 CondConv 계열은 주로 분류 네트워크의 capacity 확대를 위해 쓰였는데, 이 논문은 이를 훨씬 더 어려운 per-pixel, per-instance task에 적용합니다.

둘째, **ROI cropping 없이도 instance-specific mask prediction이 가능하도록 location information을 필터 자체와 relative coordinates에 암묵적으로 집어넣었다**는 점입니다. 논문은 instance segmentation에 appearance와 location 두 종류의 정보가 모두 필요하다고 보고, location을 ROI crop 대신 **instance-sensitive filters + relative coordinates**로 처리합니다.

요약하면, CondInst는 ROI를 중심으로 설계된 기존 instance segmentation 패러다임에서 벗어나, **instance-conditioned fully convolutional prediction**이라는 새로운 관점을 제시한 논문입니다.

## 3. Detailed Method Explanation

### 3.1 전체 구조

CondInst는 크게 두 부분으로 구성됩니다.

1. **detector branch**
2. **mask branch**

detector는 **FCOS**를 기반으로 하며, FPN feature map인 ${P_3, P_4, P_5, P_6, P_7}$ 위에서 클래스, centerness, box, 그리고 controller 출력을 예측합니다. 저자들이 FCOS를 택한 이유는 구조가 단순하고 anchor-free라서 파라미터와 계산량을 줄이기 쉽기 때문입니다.

한편 mask branch는 별도의 shared feature map $\mathbf{F}_{mask}$를 생성합니다. 이 feature map은 모든 instance가 공유하지만, 각 instance마다 detector가 생성한 서로 다른 동적 파라미터를 사용해 서로 다른 mask head가 적용됩니다. 즉, 입력 feature는 공유되지만, **필터가 instance마다 다릅니다**.

### 3.2 Instance-Aware Dynamic Mask Head

CondInst의 중심은 **controller sub-network**입니다. detector가 어떤 위치 $(x, y)$에서 instance를 예측하면, controller는 그 instance에 대한 mask head 파라미터 $\boldsymbol{\theta}_{x,y}$를 생성합니다. 이 파라미터는 고정된 네트워크를 쓰는 대신, 그 객체에 맞는 **조건부 네트워크**를 즉석에서 구성합니다. 논문은 Fig. 3 설명에서 classification head가 클래스 확률 $\boldsymbol{p}_{x,y}$를 예측하고, controller가 같은 위치에서 mask head 필터 파라미터 $\boldsymbol{\theta}_{x,y}$를 생성한다고 설명합니다.  

이때 중요한 점은 mask head가 매우 작다는 것입니다. abstract에 따르면 예시 구성은 **3개의 convolution layer, 각 8 channels** 수준입니다. 일반적인 Mask R-CNN mask head가 여러 개의 $3\times3$, 256-channel convolution을 쓰는 것과 비교하면 매우 가볍습니다. 저자들은 이것이 가능한 이유를 “이 작은 네트워크가 모든 객체를 다 분할할 필요가 없고, 단 하나의 target instance만 예측하면 되기 때문”이라고 설명합니다. 즉, 학습 난도가 크게 낮아집니다.  

### 3.3 Relative Coordinates의 역할

논문은 단순히 dynamic filters만으로 끝내지 않고, mask feature $\mathbf{F}_{mask}$에 **relative coordinates**를 concat한 $\tilde{\mathbf{F}}_{mask}$를 입력으로 사용합니다. 이 상대 좌표는 feature map의 각 위치가 현재 예측 중인 instance 중심점으로부터 얼마나 떨어져 있는지를 나타냅니다. 논문은 이것이 strong cue라고 직접 말하며, 실험에서도 중요함을 보였다고 설명합니다.  

이 설계는 매우 중요합니다. FCN이 비슷하게 생긴 두 사람 A와 B를 구분하지 못하는 이유는 appearance만으로는 “누가 foreground인가”를 결정하기 어렵기 때문입니다. Relative coordinates는 지금 예측 중인 객체가 어느 위치를 기준으로 하는지를 제공해, 동적 필터가 shape과 position을 더 안정적으로 활용할 수 있게 만듭니다. 논문은 relative coordinates만 써도 **31.3% mask AP**를 얻는다고 하며, 이것이 generated filters가 appearance뿐 아니라 shape과 relative position도 강하게 encode하고 있음을 보여준다고 해석합니다.

### 3.4 ROI-Free Full-Image Mask Prediction

CondInst는 ROI crop을 하지 않기 때문에, mask head는 feature map의 일부 patch가 아니라 **전체 mask feature map** 위에 적용됩니다. 이 점에서 Mask R-CNN과 본질적으로 다릅니다. Mask R-CNN은 박스 내부의 제한된 ROI feature를 다루지만, CondInst는 full-image feature 위에서 특정 instance에만 반응하는 network를 만듭니다. 이 덕분에 ROIAlign, resize, feature alignment 단계가 사라집니다. 또한 large instance일수록 더 큰 mask resolution이 필요하다는 기존 문제도 완화됩니다. 논문은 CondInst가 feature resize를 피하기 때문에 **higher-resolution masks with more accurate edges**를 얻는다고 주장합니다.

이 구조는 BoxInst 같은 후속 연구에서도 중요한 기반이 되었습니다. BoxInst는 CondInst가 **full-image instance mask**를 생성할 수 있다는 점이 weak supervision setting에서 핵심적이었다고 다시 강조합니다. 이는 CondInst의 구조가 단지 faster alternative이 아니라, supervision design까지 가능하게 만드는 표현 방식이었다는 뜻입니다.

### 3.5 Detector와의 결합

논문은 CondInst가 FCOS 기반이므로 전체적으로 **instance-first** 구조라고 설명합니다. 즉, semantic segmentation처럼 먼저 전역 픽셀 label을 만든 뒤 instance를 조합하는 것이 아니라, detector가 먼저 어떤 위치에서 어떤 instance가 있는지 찾고, 그 위치에서 instance-aware filter를 만들어 mask를 예측합니다. 따라서 ROI-based의 “ROI first”와 비슷한 역할 분담은 유지하면서도, 실제 ROI crop 연산은 제거합니다.

이 점을 비판적으로 해석하면, CondInst는 완전한 dense grouping 방식이라기보다 **ROI-free proposal-conditioned segmentation**에 가깝습니다. 즉, detection 신호는 여전히 중요하지만, mask prediction만큼은 완전히 convolutional하게 바꾼 셈입니다.

## 4. Experiments and Findings

### 4.1 핵심 결과

논문이 가장 강조하는 실험 메시지는 CondInst가 **Mask R-CNN을 accuracy와 speed 양쪽에서 모두 능가한다**는 것입니다. abstract와 conclusion, contribution 부분에서 저자들은 이것이 최근 state-of-the-art를 둘 다 넘은 첫 framework라고까지 주장합니다. 또한 longer training schedule 없이도 recent methods보다 경쟁력이 있다고 강조합니다.

구체적으로 qualitative comparison에서는 YOLACT-700, Mask R-CNN보다 **more details**를 보존하는 높은 품질의 mask를 보여준다고 설명합니다. 이 부분은 CondInst가 단지 빠른 모델이 아니라, 경계 보존과 fine detail 측면에서도 강하다는 질적 증거입니다.

### 4.2 계산 효율

논문은 CondInst가 FCOS 대비 **최대 100 instances를 처리해도 약 10% 정도의 계산 시간만 추가**된다고 주장합니다. 이는 매우 중요한 결과입니다. ROI-based 방식은 per-instance head 비용이 큰 반면, CondInst는 shared mask feature와 extremely compact mask head를 사용하므로 추가 비용이 작습니다.

즉, CondInst의 속도 이점은 단순 구현 최적화 때문이 아니라 다음 구조적 이유에서 나옵니다.

* ROI crop이 없음
* feature alignment가 없음
* mask head가 매우 작음
* instance 수에 따른 per-instance overhead가 작음

이 구조는 실제 고밀도 장면이나 실시간 응용에서 특히 유리합니다.

### 4.3 Relative Coordinates와 Filter Encoding

논문은 generated filters가 단지 appearance만 encode하는 것이 아니라, **shape와 relative position까지 encode**한다고 해석합니다. 그 근거 중 하나가 “relative coordinates만으로도 31.3% mask AP”를 얻는 실험입니다. 이는 CondInst가 box처럼 명시적 geometric crop 없이도, 동적 필터와 좌표 정보를 결합해 instance 형상을 상당히 잘 표현할 수 있음을 시사합니다. 또한 absolute coordinates는 성능 향상이 크지 않았다고 하므로, 중요한 것은 이미지 전체의 절대 위치가 아니라 **instance 기준의 상대 위치**라는 점도 드러납니다.

### 4.4 기존 FCN 기반 방법과의 차이

논문은 InstanceFCN, semi-convolution, embedding-based 방법 등 이전 fully convolutional 계열이 있었지만, **COCO에서 Mask R-CNN을 accuracy와 speed 둘 다 넘지 못했다**고 정리합니다. 반면 CondInst는 그 장벽을 넘는 첫 사례라고 주장합니다. 이것은 단순히 one more FCN method가 아니라, FCN instance segmentation의 practical viability를 크게 높인 결과입니다.  

## 5. Strengths, Limitations, and Interpretation

### Strengths

가장 큰 강점은 문제 설정을 아주 날카롭게 바꿨다는 점입니다.
CondInst는 ROI를 feature crop이 아니라 **network parameter generation**으로 대체합니다. 이 전환은 단순한 engineering tweak가 아니라, instance segmentation representation 자체를 바꾸는 발상입니다.

두 번째 강점은 구조의 단순성과 효율입니다.
논문은 mask head를 3-layer, 8-channel 수준까지 줄였고, FCOS 대비 추가 시간도 약 10% 수준이라고 설명합니다. 이런 효율은 one-stage 또는 real-time 지향 시스템에서 매우 매력적입니다.  

세 번째 강점은 출력 품질입니다.
ROI resize를 피하기 때문에 large object에서 edge detail을 더 잘 보존할 수 있고, qualitative 결과에서도 YOLACT 및 Mask R-CNN보다 더 세밀한 마스크를 보인다고 주장합니다.  

### Limitations

첫째, CondInst는 ROI operation을 제거했지만, 여전히 **detector-conditioned instance-first** 구조입니다. 즉, detection 품질과 center-based instance representation에 계속 의존합니다. 완전히 detection-free dense grouping 접근과는 다릅니다.

둘째, 동적 필터는 이론적으로 유연하지만, 실제로는 instance 수만큼 mask head를 반복 적용해야 합니다. 논문은 overhead가 작다고 주장하지만, 극단적으로 많은 instance가 존재하는 경우 그 비용이 완전히 0은 아닙니다. 다만 저자들은 최대 100 instances에서도 FCOS 대비 약 10% 증가에 그친다고 해 이 점을 완화합니다.

셋째, 제공된 첨부 HTML 조각과 검색 결과만으로는 모든 테이블의 수치 전체가 다 노출되지는 않았습니다. 따라서 실험 비교의 모든 세부 수치를 완전히 재현하는 수준으로 정리하기보다는, 논문이 강조한 정성적·정량적 핵심 메시지 위주로 해석하는 것이 정확합니다. 이 점은 명확히 밝혀둡니다.  

### Interpretation

비판적으로 보면, CondInst의 진짜 공헌은 “conditional convolution을 썼다” 자체보다, **instance segmentation에서 instance를 어떻게 표현할 것인가**에 대한 답을 바꿨다는 데 있습니다.

기존:

* instance = ROI / box crop

CondInst:

* instance = dynamic mask head parameters + relative coordinates

이 재정의 덕분에 ROI-free instance segmentation이 실용적 수준으로 올라왔고, 이후 BlendMask, BoxInst 같은 후속 연구에도 직접적인 기반이 되었습니다. 특히 BoxInst가 CondInst의 full-image mask prediction 성질을 weak supervision에 활용한 점은 CondInst representation의 확장성을 잘 보여줍니다.

## 6. Conclusion

CondInst는 instance segmentation을 위한 **Conditional Convolutions** 기반의 새로운 프레임워크로, 각 instance마다 **controller가 동적으로 생성한 mask head**를 전체 feature map에 적용해 마스크를 예측합니다. 이 방식은 ROI cropping과 feature alignment를 제거하면서도, relative coordinates와 instance-aware filters를 통해 같은 클래스의 여러 객체를 구분할 수 있게 합니다. 그 결과 매우 작은 mask head로도 Mask R-CNN보다 더 빠르고 더 정확한 성능을 달성했다고 논문은 주장합니다. 또한 feature resize를 피하기 때문에 더 높은 해상도와 더 정확한 경계의 마스크를 만들 수 있다고 설명합니다.

연구적으로는 CondInst가 **ROI-based instance segmentation의 대안을 처음으로 강하게 입증한 논문**이라는 점에서 중요하고, 실무적으로는 빠르고 단순하면서도 고품질의 instance mask가 필요한 환경에서 매우 의미가 큽니다. 후속 weakly supervised 및 one-stage 계열 연구에도 영향을 준 중요한 전환점으로 볼 수 있습니다.
