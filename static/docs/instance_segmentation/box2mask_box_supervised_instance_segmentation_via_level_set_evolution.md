# Box2Mask: Box-supervised Instance Segmentation via Level-set Evolution

이 논문은 **bounding box annotation만으로도 fully supervised instance segmentation에 가까운 mask 품질을 낼 수 있는가**라는 문제를 다룬다. 기존 box-supervised instance segmentation은 대체로 pseudo mask 생성이나 pixel-pair affinity modeling에 의존했는데, 저자들은 이런 방식이 복잡한 배경이나 유사한 외관의 인접 객체 때문에 쉽게 noise에 오염된다고 본다. 이를 해결하기 위해 이 논문은 고전적인 **level-set evolution**, 특히 **Chan-Vese energy**를 딥러닝 프레임워크 안에 통합한 **Box2Mask**를 제안한다. 핵심은 네트워크가 직접 정답 mask를 강하게 맞추는 대신, 각 인스턴스의 mask map을 level-set function으로 보고 **bounding box 내부에서 energy minimization을 반복하며 경계를 점진적으로 진화시키도록 학습**하는 것이다. 또한 이전 conference 버전에서 더 확장되어, CNN 기반 프레임워크뿐 아니라 **transformer-based framework**, **box-level bipartite matching**, 그리고 **local consistency module (LCM)**까지 포함한다. 논문은 COCO, Pascal VOC, iSAID, LiTS, scene text 데이터셋에서 SOTA를 보고하며, 특히 Swin-L backbone에서는 COCO에서 **42.4% AP**를 달성해 fully supervised 방법과도 경쟁 가능하다고 주장한다.  

## 1. Paper Overview

이 논문의 출발점은 매우 실용적이다. instance segmentation은 자율주행, 로봇 조작, 이미지 편집, 세포 분할 등 다양한 응용에서 중요하지만, 기존 최고 성능 방법들은 대부분 **pixel-wise instance mask annotation**에 크게 의존한다. 문제는 이런 주석이 매우 비싸다는 점이다. 논문은 COCO 기준으로 polygon 기반 mask annotation에 평균 **79.2초**, bounding box annotation에는 **7초** 정도가 필요하다고 설명하면서, box annotation만으로 학습하는 box-supervised instance segmentation의 필요성을 강조한다.

기존 box-supervised 방법은 대체로 두 방향이었다. 하나는 MCG, CRF, saliency 같은 추가 절차로 pseudo mask를 만들어 supervision으로 쓰는 multi-stage 방식이고, 다른 하나는 BoxInst나 DiscoBox처럼 인접 픽셀 또는 color pair의 affinity를 활용하는 end-to-end 방식이다. 하지만 저자들은 이 두 부류 모두 약점이 있다고 본다. 전자는 파이프라인이 복잡하고 하이퍼파라미터가 많고, 후자는 “비슷한 픽셀끼리는 같은 라벨”이라는 단순한 가정 때문에 복잡한 배경이나 비슷한 객체 주변에서 label noise가 크다. 이 논문은 그 대안으로 **pairwise affinity 대신 variational level-set energy를 경계 학습의 중심에 두자**고 제안한다.

## 2. Core Idea

이 논문의 핵심 아이디어는 다음과 같이 요약할 수 있다.

> **box 안에서 pseudo mask를 직접 만들거나 픽셀 쌍 관계를 과하게 믿지 말고, level-set energy를 최소화하면서 객체 경계를 반복적으로 진화시키자.**

저자들은 고전적인 **Chan-Vese level-set model**을 딥네트워크 안에 녹여, 네트워크가 각 인스턴스에 대해 **연속적인 level-set function의 시퀀스**를 학습하도록 만든다. 여기서 중요한 점은 level-set evolution이 단순 후처리가 아니라, **fully differentiable energy function**을 통해 end-to-end 학습 안에 포함된다는 것이다. 즉 Box2Mask는 weak supervision에서 흔한 “좋은 pseudo mask를 만든 뒤 그걸 정답처럼 쓰는” 접근과 다르다. 대신 **경계가 어떻게 점진적으로 수렴해야 하는지 자체를 학습**한다.

또 하나의 핵심은 **입력 이미지의 low-level 정보와 네트워크의 high-level deep feature를 함께 사용**한다는 점이다. 논문은 기존 level-set 기반 딥러닝 방법들이 대부분 fully supervised였고, 심지어 일부는 원본 이미지의 low-level feature를 충분히 활용하지 못했다고 본다. Box2Mask는 원본 이미지와 deep feature를 함께 energy term에 반영해 더 robust한 curve evolution을 유도한다. 여기에 neighborhood 안에서 local affinity consistency를 강화하는 **LCM(Local Consistency Module)**까지 넣어, region intensity inhomogeneity 문제도 완화한다.

## 3. Detailed Method Explanation

### 3.1 전체 파이프라인

Box2Mask의 전체 구조는 네 가지 핵심 구성요소로 정리된다.

1. **Backbone**
2. **Instance-aware Decoder (IAD)**
3. **Box-level Matching Assignment**
4. **Level-set Evolution Module**

입력 이미지를 backbone으로 인코딩한 뒤, IAD가 각 인스턴스의 특징을 반영한 **full-image instance-aware mask map**을 생성한다. 이후 box-level matching이 그 중 어떤 mask map을 positive로 볼지 결정하고, positive mask map만 해당 GT box 내부에서 level-set evolution을 수행한다. 이 과정에서 energy minimization을 반복하며 각 instance의 boundary가 더 정교해진다. 논문은 이 전체 구조를 **single-stage** 프레임워크로 구현한다는 점을 강조한다.

### 3.2 Instance-aware Decoder

IAD의 목적은 “각 인스턴스마다 다른 특성”을 학습해 **instance-aware mask map**을 만드는 것이다. 논문은 intensity, appearance, shape, location 같은 instance-wise characteristics를 반영해야 한다고 설명한다. 구조적으로는 크게

* pixel-wise decoder
* kernel learning network

로 나뉜다. 즉, unified mask feature를 먼저 만들고, 각 인스턴스별 unique kernel을 학습해 둘을 결합함으로써 인스턴스별 mask를 생성한다.

#### CNN-based IAD

CNN 버전에서는 **SOLOv2의 dynamic convolution** 아이디어를 따른다. backbone feature에서 instance-unique kernel $K_{i,j}$를 예측하고, pixel-wise decoder가 만든 unified mask feature $F_{mask}$와 결합해

$$
M_{i,j} = K_{i,j} * F_{mask}
$$

형태로 full-image mask map을 만든다. 여기서 $M_{i,j}$는 위치 $(i,j)$에 중심을 둔 하나의 인스턴스에 대응하는 mask map이다. 이 방식은 ROI 없이도 인스턴스별 mask를 동적으로 생성할 수 있다는 장점이 있다.

#### Transformer-based IAD

확장 버전에서 저자들은 transformer decoder를 사용한 **transformer-based IAD**도 제안한다. 이 구조는 MaskFormer와 유사하게 object query를 사용한다. 기본적으로 $N=100$개의 instance-wise query를 두고, transformer decoder가 각 query에 대해 $K \in \mathbb{R}^{N \times C}$ 형태의 instance-aware kernel vector를 계산한다. 동시에 pixel decoder가 high-resolution mask feature $F_{mask} \in \mathbb{R}^{C \times H \times W}$를 만들고, 최종 mask map은 dot product로 계산된다.

$$
M^{N \times H \times W} = K^{N \times C} \cdot F_{mask}^{C \times H \times W}
$$

즉 CNN 버전이 dynamic conv 기반이라면, transformer 버전은 **query-based dynamic kernel learning**으로 볼 수 있다. 논문은 이 확장판에서 transformer decoder가 instance-wise characteristic embedding에 더 강하고, set prediction 기반 assignment와 잘 맞는다고 본다.

### 3.3 Box-level Matching Assignment

이 모듈은 어떤 predicted mask map을 positive sample로 둘지 결정하는 역할을 한다. CNN 기반 프레임워크에서는 box guidance를 바탕으로 high-quality mask map을 positive로 선택하고, transformer 기반 프레임워크에서는 이 개념이 더 일반화되어 **box-level bipartite matching**으로 확장된다. 논문은 extended version의 주요 추가점 중 하나로 이 **box-level bipartite matching scheme**을 명시적으로 언급한다. 이는 DETR/MaskFormer류의 set matching 장점을 weak supervision setting에 도입한 것으로 볼 수 있다.

중요한 해석은, fully supervised transformer segmentation에서 mask cost를 쓰던 자리에, Box2Mask는 **box 기반 segmentation quality proxy**를 넣어 weak supervision에서도 matching을 가능하게 만든다는 점이다.

### 3.4 Level-set Evolution Module

이 모듈이 논문의 핵심이다. Box2Mask는 classical **Chan-Vese energy functional**을 사용한다. level-set의 핵심 생각은 객체 경계를 명시적인 contour가 아니라 higher-dimensional function의 zero level로 표현하고, energy를 최소화하면서 contour를 진화시키는 것이다. 저자들은 이 energy가 pixel intensity, color, appearance, shape 같은 풍부한 context를 사용할 수 있다고 설명한다.

Box2Mask에서 중요한 점은 다음 세 가지다.

첫째, **bounding box 내부에서만 evolution이 이뤄진다.**
즉 weak supervision의 제약을 이용해 탐색 공간을 강하게 제한한다.

둘째, **각 evolution step마다 level-set을 box projection function으로 자동 초기화한다.**
이는 rough boundary estimate를 제공해 curve evolution을 안정화한다.

셋째, **low-level image cue와 high-level deep feature를 함께 사용한다.**
기존 일부 level-set 딥러닝 방법이 deep feature만 쓰거나 fully supervised였던 것과 달리, Box2Mask는 weak supervision에서도 robust한 evolution을 위해 두 종류의 정보를 동시에 쓴다.

이 구조 덕분에 네트워크는 직접 mask label을 받지 않아도, 반복적인 energy minimization을 통해 경계가 자연스럽게 box 내부 객체로 수렴하는 방향을 학습한다.

### 3.5 Local Consistency Module

extended journal version의 중요한 추가점이 바로 **LCM(Local Consistency Module)**이다. 저자들은 순수 Chan-Vese 계열의 region-based evolution이 **intensity inhomogeneity**에 약할 수 있음을 인정하고, 이를 보완하기 위해 pixel affinity kernel 기반 local consistency를 도입한다. 이 모듈은 neighborhood 안에서 pixel similarity와 spatial relation을 더 잘 활용해, evolution이 지역적으로 더 일관된 방향으로 진행되도록 한다.

이 모듈은 특히 pairwise affinity 기반 기존 방법과 미묘하게 다르다. BoxInst나 DiscoBox가 affinity를 supervision의 핵심으로 삼았다면, Box2Mask의 LCM은 **level-set evolution을 보조하는 local regularizer**에 가깝다. 즉 main driving force는 여전히 variational energy이고, affinity는 지역적 consistency를 보강하는 데 쓰인다.

### 3.6 Training과 Inference

중요한 실용적 포인트는 **level-set evolution이 training 시에만 사용된다**는 점이다. 논문은 inference에서는 직접적이고 효율적으로 instance mask를 출력하며, 추가적인 level-set evolution이 필요 없다고 명시한다. CNN 기반 모델은 matrix NMS가 필요하지만, transformer 기반 모델은 **NMS-free**로 동작한다.

이건 매우 중요한 설계다. 학습 때는 variational prior의 장점을 활용하고, 추론 때는 iterative optimization 비용을 제거한다. 따라서 Box2Mask는 “학습은 정교하게, 추론은 단순하게”라는 전략을 취한다.

## 4. Experiments and Findings

### 4.1 데이터셋

논문은 다섯 가지 이질적 testbed에서 실험한다.

* **Pascal VOC**
* **COCO**
* **iSAID** (remote sensing)
* **LiTS** (medical image)
* **ICDAR2019 ReCTS** (scene text)

즉 일반 장면, 원격탐사, 의료, 장면 텍스트까지 포함해 방법의 generality를 보이려는 설계다.

### 4.2 메인 결과

논문 초반 요약에서 가장 강조하는 수치는 다음과 같다.

* Pascal VOC에서 기존 SOTA **38.3% AP → 43.2% AP**
* COCO에서 ResNet-101 backbone 기준 **33.4% AP → 38.3% AP**
* Swin-L backbone에서는 COCO에서 **42.4% AP**

그리고 이 42.4% AP는 당시의 well-established fully mask-supervised methods와 동급이라고 주장한다. 저자들은 또한 Mask R-CNN, SOLO, PolarMask 같은 fully supervised 방법보다도 같은 backbone에서는 더 낫다고 말한다. weak supervision만으로 이 정도 수준에 도달했다는 점이 논문의 가장 큰 empirical claim이다.

### 4.3 왜 결과가 설득력 있는가

이 실험의 설득력은 단지 COCO 숫자 때문만이 아니다. 논문은 Box2Mask가 general scenes뿐 아니라 **remote sensing, medical, scene text**에서도 일관되게 강하다고 강조한다. 이는 단순히 natural image benchmark에 과적합된 방법이 아니라, **복잡한 배경과 미세한 경계가 중요한 문제에서 오히려 강점이 있다**는 해석을 가능하게 한다. Figure 1 설명에서도 BoxInst나 DiscoBox보다 finer details와 accurate boundaries를 더 잘 보존한다고 주장한다.

### 4.4 Ablation: LCM의 효과

검색 결과에 따르면 LCM의 dilation rate를 조절하는 ablation에서, 적절한 local region이 중요하다는 결론이 나온다. dilation rate 1, 2, 3은 비슷하게 좋고, 너무 크게 4나 5로 키우면 성능이 떨어진다. 특히 적절한 설정에서는 **36.3% AP**, 그리고 **+2.5% AP improvement**가 보고된다. 저자들은 이를 통해 LCM이 global dependency보다는 **local affinity consistency**에 기여하는 모듈이라는 점을 강조한다.

즉 이 모듈은 “멀리 있는 픽셀과 다 연결하자”가 아니라, 경계 주변의 지역적 smoothness와 consistency를 보강하는 역할을 한다.

### 4.5 Transformer 확장의 의미

이 논문은 이전 conference version 대비 **transformer-based framework**를 새로 포함한 확장판이다. 저자들은 transformer decoder를 이용해 instance-wise characteristic을 더 강하게 임베딩하고, box-level bipartite matching을 도입했다고 밝힌다. 이건 단순히 backbone만 바꾼 것이 아니라, weakly supervised box segmentation을 **query-based set prediction framework**로 연결한 것이다.

의미 있는 해석은, Box2Mask가 단순히 classical variational trick을 CNN에 얹은 논문이 아니라, CNN-based와 transformer-based라는 두 계열 모두에 적용 가능한 **general training principle**을 제시했다는 점이다.

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **문제 재정의가 좋다**는 점이다. 기존 weakly supervised box segmentation이 pseudo mask 생성이나 pairwise affinity를 중심에 두었다면, 이 논문은 경계 학습을 **level-set energy minimization**으로 재구성한다. 이 덕분에 객체-배경 경계를 단순한 local pair relation보다 더 구조적으로 다룰 수 있다.

두 번째 강점은 **classical variational method와 modern deep segmentation의 결합이 자연스럽다**는 점이다. Chan-Vese라는 오래된 segmentation 철학을, instance-aware decoder, transformer query, end-to-end differentiable training과 결합해 현대적인 weak supervision framework로 재구성했다. 이건 단순한 “고전 기법 재활용”이 아니라, weak supervision에서 부족한 inductive bias를 variational prior로 채운 사례다.

세 번째 강점은 **training/inference 분리**다. 학습 때만 level-set evolution을 쓰고, 추론은 직접 mask 예측으로 끝낸다. 그래서 이론적으로는 강한 structural prior를 학습에 쓰면서도, 실사용 추론 비용은 크게 늘지 않는다.

### 한계

첫째, 논문의 수식 부분은 업로드된 ar5iv HTML에서 완전하게 읽기 쉽지 않다. 개념 구조는 명확하지만, exact energy term과 gradient-based update를 구현 수준으로 재구성하려면 원문 PDF나 코드 저장소를 함께 보는 편이 안전하다.

둘째, level-set evolution이 training-time only라 하더라도, 학습 자체는 일반적인 direct mask supervision보다 더 구조적이고 복잡하다. 즉 annotation은 싸지만, 학습 설계는 결코 단순하지 않다. 실무에서는 이런 trade-off를 고려해야 한다.

셋째, Box2Mask가 높은 성능을 보이긴 하지만, 그 성과의 일부는 strong backbone과 instance-aware decoder 설계에서도 나온다. 따라서 이 논문의 핵심은 “level set만 있으면 된다”가 아니라, **instance-aware prediction + matching + level-set supervision**의 결합이라는 점을 함께 봐야 한다.

### 해석

비판적으로 보면, 이 논문의 진짜 기여는 “box-supervised SOTA 수치” 하나보다 더 크다. 더 중요한 것은 **weak supervision에서 direct pseudo-labeling 대신 implicit curve evolution supervision이 가능하다**는 점을 보여준 것이다. 이 관점은 앞으로 box supervision뿐 아니라 points, scribbles, clicks 같은 다른 약한 주석 형태에도 확장될 가능성이 있다. 또한 transformer-based extension은 이런 철학이 CNN에만 갇히지 않고 query-based segmentation에도 이어질 수 있음을 보여준다. 다만 이 확장 가능성은 논문의 실험 범위를 넘어선 해석이므로 가능성 수준으로 보는 게 타당하다.

## 6. Conclusion

이 논문은 box-supervised instance segmentation의 대표적 한계인 noisy pseudo masks와 oversimplified pairwise affinity를 비판하고, 그 대안으로 **Box2Mask**를 제안한다. Box2Mask는 **instance-aware decoder**, **box-level matching assignment**, **level-set evolution**을 결합한 single-stage 프레임워크이며, Chan-Vese energy를 사용해 bounding box 내부에서 객체 경계를 반복적으로 진화시킨다. journal extension에서는 여기에 **transformer-based framework**, **box-level bipartite matching**, **LCM**까지 추가해 방법을 더 일반화했다. 실험적으로는 Pascal VOC, COCO, iSAID, LiTS, scene text 데이터셋에서 강한 성능을 보였고, 특히 Swin-L 기준 COCO **42.4% AP**는 box supervision만으로 fully supervised 수준에 근접할 수 있음을 보여준다.  

실제로 이 논문은 weakly supervised instance segmentation에서 하나의 중요한 방향 전환으로 읽힌다. 즉, supervision이 약할수록 오히려 더 강한 구조적 prior가 필요할 수 있고, 그 prior를 classical variational model에서 가져오는 것이 충분히 효과적일 수 있음을 보여준다.
