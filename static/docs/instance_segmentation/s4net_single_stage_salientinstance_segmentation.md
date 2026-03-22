# S4Net: Single Stage Salient-Instance Segmentation

* **저자**: Ruochen Fan, Ming-Ming Cheng, Qibin Hou, Tai-Jiang Mu, Jingdong Wang, Shi-Min Hu
* **발표연도**: 2017
* **arXiv**: [https://arxiv.org/abs/1711.07618](https://arxiv.org/abs/1711.07618)

## 1. 논문 개요

이 논문은 **salient instance segmentation** 문제를 다룬다. 이는 단순히 장면에서 눈에 띄는 물체를 찾는 수준을 넘어서, **각 salient object를 개별 인스턴스로 분리하고 픽셀 단위 마스크까지 출력하는 문제**이다. 다시 말해, “무엇이 눈에 띄는가”와 “그 눈에 띄는 대상들이 각각 어디까지인가”를 동시에 해결하려는 작업이다.

기존의 salient object detection은 대체로 foreground와 background를 구분하는 **이진 분할**에 가깝고, object detection은 bounding box까지만 제공한다. 또한 semantic instance segmentation은 category-aware 설정, 즉 사전에 정의된 클래스 집합 안에서만 인스턴스를 구분하는 경우가 많다. 반면 이 논문이 다루는 문제는 **class-agnostic**하다. 즉, 특정 semantic category에 속하지 않더라도 눈에 띄는 개별 객체라면 분리해야 한다.

저자들은 이 문제를 위해 **S4Net**이라는 단일 단계(single-stage) 구조를 제안한다. 핵심 목표는 두 가지다. 첫째, **고품질 instance-level segmentation**을 수행하는 것, 둘째, 이를 **실시간에 가깝게 빠르게** 수행하는 것이다. 논문은 특히 기존 CNN 기반 인스턴스 분할 방식이 proposal 내부 특징에만 지나치게 집중하고, proposal 주변 배경 정보와 foreground/background separation을 충분히 활용하지 못한다는 점을 문제로 본다.

이 문제가 중요한 이유는 분명하다. salient instance segmentation은 이미지 편집, 장면 이해, 로봇 지각, weakly supervised semantic segmentation 같은 응용에서 직접적인 가치를 가진다. 단순 saliency map보다 인스턴스 단위 결과가 훨씬 유용하며, box만 있는 결과보다 정밀 마스크가 훨씬 실용적이다. 따라서 이 논문은 saliency와 instance segmentation 사이의 간극을 메우는 실용적 문제를 다룬다고 볼 수 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **proposal 내부 정보만 보는 것으로는 salient instance를 충분히 잘 분리할 수 없고, proposal 주변의 문맥과 배경 정보를 명시적으로 활용해야 한다**는 점이다. 저자들은 이 직관을 CNN 구조 안에 직접 넣기 위해 **RoIMasking**이라는 새로운 region feature extraction layer를 제안한다.

기존의 RoIPool이나 RoIAlign은 주어진 RoI 내부를 고정 크기로 샘플링하여 feature를 추출한다. 이 방식은 detection이나 semantic instance segmentation에서는 유용하지만, salient instance segmentation에서는 한계가 있다. 예를 들어, 서로 가까이 있거나 일부 가려진 두 객체를 구분하려면 물체 내부 모양뿐 아니라 **주변 배경과의 대비**, 그리고 **인접한 다른 객체와의 상대적 경계**도 중요하다. 그런데 기존 방식은 이 바깥 정보를 명시적으로 다루지 않는다.

S4Net은 여기서 한 걸음 더 나아간다. 단순히 RoI 내부만 잘라 쓰지 않고, **RoI 바깥 주변 영역까지 함께 본 뒤**, 그 바깥 부분의 feature 부호를 뒤집어(background임을 명시적으로 표현) segmentation branch가 foreground와 background를 더 쉽게 분리하도록 만든다. 이 설계는 전통적인 figure-ground segmentation, 특히 GrabCut류 접근이 foreground/background separation을 중요하게 여겼다는 점에서 영감을 받은 것으로 보인다.

또 하나의 중요한 아이디어는 구조 전체를 **single-stage**로 유지하는 것이다. 저자들은 FPN 기반 detector 위에 segmentation branch를 붙이되, proposal별로 무거운 후처리나 다단계 refinement를 하지 않고 효율적으로 동작하도록 설계했다. 그래서 정확도뿐 아니라 속도도 확보했다.

기존 접근과의 차별점은 다음과 같이 정리할 수 있다. 첫째, class-aware가 아니라 **class-agnostic salient instance segmentation**에 초점을 둔다. 둘째, Mask R-CNN류 segmentation branch와 달리 proposal 내부 feature만이 아니라 **주변 문맥과 foreground/background contrast**를 적극 사용한다. 셋째, RoIPool/RoIAlign 대신 **resolution-preserving, quantization-free** 성격의 RoIMasking을 사용하여 더 세밀한 segmentation을 노린다.

## 3. 상세 방법 설명

### 전체 구조

S4Net은 크게 두 부분으로 이루어진다. 하나는 **bounding box detector**, 다른 하나는 **segmentation branch**이다. 두 모듈은 backbone feature를 공유한다. backbone으로는 기본적으로 ResNet-50을 사용하며, 효율성을 위해 **single-shot detector + FPN** 구조를 채택한다.

Detection branch는 여러 scale의 feature를 다루기 위해 FPN을 사용한다. 논문에 따르면 conv2에 연결되는 lateral connection은 제거하고 conv3부터 conv6까지를 사용한다. 여러 detection head가 각 lateral layer에 연결되어 multi-scale detection을 수행한다. 구조적 개념은 Faster R-CNN의 head와 유사하지만, single-shot 설정으로 설계되어 속도를 높인다.

Segmentation branch는 detection branch가 예측한 bounding box들과 backbone의 stride 8 feature map을 입력으로 받는다. 이 branch 안에 논문의 핵심인 **RoIMasking layer**와 **salient instance discriminator**가 포함된다.

### RoIMasking의 동기

기존 RoIPool과 RoIAlign은 모두 RoI 내부 feature를 추출하지만, 바깥 문맥을 사실상 버린다. 또한 RoIPool은 quantization으로 인한 misalignment 문제가 있고, 둘 다 고정 크기로 샘플링하므로 **원래 해상도와 종횡비(aspect ratio)**를 그대로 유지하지 못한다. 저자들은 이것이 세밀한 mask 품질을 해칠 수 있다고 본다.

S4Net의 RoIMasking은 다음 두 목적을 가진다.

첫째, **RoI 주변 배경 정보까지 segmentation branch에 전달**한다.
둘째, foreground와 background를 단순히 함께 보여주는 것이 아니라 **명시적으로 구분되게 표현**한다.

### Binary RoIMasking

가장 단순한 형태는 binary RoIMasking이다. proposal rectangle 내부는 mask 값 1, 바깥은 0으로 놓는다. 입력 feature map에 이 mask를 곱하면 proposal 내부 feature만 남고 바깥은 사라진다.

이 방식은 기존 RoIPool/RoIAlign과 달리 feature를 고정 해상도로 재샘플링하지 않는다. 즉, **원본 feature map의 해상도와 종횡비를 유지**한다. 다만 배경 문맥 활용은 여전히 제한적이다.

### Expanded Binary RoIMasking

다음으로 proposal보다 더 큰 영역을 포함하도록 확장한 binary masking을 고려한다. 이는 segmentation branch의 receptive field를 넓혀 주변 context를 더 많이 보게 한다. 하지만 논문은 단순히 context를 많이 보는 것만으로는 충분하지 않다고 말한다. 단지 주변 영역을 포함시키는 것만으로는 salient instance와 배경을 명확히 구분하는 신호가 약하기 때문이다.

### Ternary RoIMasking

논문의 핵심은 ternary RoIMasking이다. 여기서는 mask가 세 값을 가진다. 개념적으로 proposal 내부는 target 영역으로 유지하고, proposal 바깥 확장 영역은 **-1**, 그 외는 0 혹은 무시 영역으로 취급하는 구조다. 중요한 점은 **확장된 주변 영역 feature의 부호를 뒤집는다**는 것이다.

ReLU 이후 feature map은 원래 음수가 없으므로, 주변 배경 영역을 -1 mask로 곱하면 그 부분은 음수 feature가 된다. 그러면 segmentation branch는 **RoI 내부의 양수 성격 특징**과 **주변 배경의 음수 성격 특징**을 더 쉽게 구분할 수 있다. 이는 단순한 context 추가를 넘어서, 네트워크가 foreground/background separation을 더 명시적으로 학습하도록 유도한다.

저자들이 강조하는 장점은 다음과 같다.

* 추가적인 복잡한 연산 없이 구현 가능하다.
* quantization-free이다.
* aspect ratio와 resolution을 보존한다.
* 배경과의 대비를 명시적으로 강화한다.
* 특히 가려짐(occlusion)이나 가까운 인접 인스턴스 구분에 유리하다.

### RoIMasking 효과 분석

논문은 ternary masking이 실제로 주변 정보를 더 적극 활용하는지 gradient map 분석도 제시한다. feature map의 특정 위치 $H_{i,j,c}$가 segmentation loss에 얼마나 중요한지를 보기 위해 gradient

$$
G_{i,j,c} = \frac{\partial L_{sal}}{\partial H_{i,j,c}}
$$

를 계산하고, 채널 방향 절댓값 합으로 2D importance map을 만든다. 즉,

$$
G_{i,j} = \sum_c |G_{i,j,c}|
$$

와 유사한 형태의 시각화를 사용하는 셈이다.

이 분석에서 ternary masking은 binary masking보다 proposal 주변 perimeter 영역에 더 강한 응답을 보인다. 저자들은 이를 근거로 **주변 배경 문맥이 salient instance segmentation에 실제로 중요하다**고 주장한다.

### Segmentation Branch 구조

RoIMasking 뒤에는 salient instance discriminator라고 부를 수 있는 segmentation branch가 붙는다. 입력은 conv3에 대응하는 stride 8 feature map에서 온 feature이다. 먼저 채널 수를 줄이기 위해 $1 \times 1$ convolution으로 256채널 압축을 수행한다.

이후 segmentation branch는 단순한 연속 convolution stack이 아니라, **skip connection**, **dilated convolution**, 그리고 **stride 1의 max pooling**을 활용해 receptive field를 넓힌다. 논문 설명상 두 개의 residual block과 두 개의 dilated convolution block이 포함되며, 모든 convolution은 $3 \times 3$ kernel, stride 1을 사용한다. 초반 세 층 채널 수는 128, 이후는 64로 설정했다.

이 branch의 목적은 단순한 binary foreground 분할이 아니라, **같은 RoI 안에 들어온 여러 객체 중 실제 salient instance를 더 잘 가려내는 것**이다. 따라서 RoIMasking이 배경 대비를 강화한다면, segmentation branch는 그 정보를 실제 mask prediction으로 변환하는 discriminator 역할을 한다고 이해하면 된다.

### 학습 목표와 손실 함수

전체 학습은 detection과 segmentation을 함께 수행하는 multi-task loss로 진행된다.

$$
L = L_{obj} + L_{coord} + L_{seg}
$$

여기서 $L_{obj}$는 objectness classification loss, $L_{coord}$는 bounding box regression loss, $L_{seg}$는 segmentation loss이다.

Objectness loss는 positive proposal 수가 negative보다 훨씬 적다는 점을 고려하여, positive와 negative 손실을 따로 정규화해 합친다.

$$
L_{obj} = -\left( \frac{1}{N_P}\sum_{i \in P}\log p_i + \frac{1}{N_N}\sum_{j \in N}\log(1-p_j) \right)
$$

여기서 $P$는 positive proposal 집합, $N$은 negative proposal 집합, $N_P$와 $N_N$는 각각의 개수, $p_i$는 proposal $i$가 positive일 확률이다. 이 설계는 negative sample 수가 너무 많아 gradient를 지배하는 현상을 줄이기 위한 것이다.

Bounding box regression에는 Fast R-CNN과 같은 Smooth L1 loss를 사용하고, segmentation branch에는 Mask R-CNN과 유사한 cross-entropy loss를 사용한다고 논문은 설명한다. 다만 segmentation loss의 정확한 픽셀 단위 식을 본문에서 자세히 전개하지는 않는다.

### 학습 및 추론 절차

훈련 시 detection branch에서 proposal의 positive/negative 여부는 IoU 기준으로 결정한다. 본문 추출 상태가 다소 깨져 있지만, 일반적인 문맥과 설명상 **IoU가 0.5보다 크면 positive**, 낮으면 negative로 취급하는 설정임을 알 수 있다. 다만 negative 임계값이 본문 추출에서 불명확하게 보이므로, 정확한 세부 기준은 원문 확인이 필요하다.

중요한 점은 **훈련 중 segmentation branch에는 detection 결과가 아니라 ground-truth bounding box를 직접 입력**한다는 것이다. 저자들은 이것이 더 안정적인 학습 데이터를 제공하고 학습도 빨라진다고 설명한다. 반면 테스트 시에는 detection branch가 예측한 bounding box를 RoIMasking에 넣는다.

추론 시 proposal 수는 성능과 속도 사이의 trade-off가 있다. 너무 많은 proposal을 쓰면 조금 더 좋아질 수 있지만 속도가 크게 느려진다. 논문은 기본적으로 **20개 proposal**을 선택해 균형을 맞췄다고 말한다.

## 4. 실험 및 결과

### 데이터셋과 평가 설정

모든 주요 실험은 Li et al.의 salient instance segmentation 데이터셋을 사용한다. 총 1,000장의 이미지가 있으며, 500장을 학습, 200장을 검증, 300장을 테스트에 사용한다. 이는 기존 MSRNet과의 공정 비교를 위해 동일한 분할을 따랐다고 한다.

평가 지표는 COCO 스타일의 mAP를 따른다. 구체적으로 $mAP^{0.5}$, $mAP^{0.7}$를 사용하며, occlusion이 있는 인스턴스만 따로 모은 subset에 대해서는 $mAP_O^{0.5}$, $mAP_O^{0.7}$도 보고한다. 이 occlusion subset 평가는 **겹치거나 가려지는 인스턴스를 얼마나 잘 구분하는지**를 보기 위해 중요하다.

### RoIMasking ablation

가장 핵심적인 ablation은 RoIAlign, RoIPool, Binary RoIMasking, Ternary RoIMasking 비교이다.

표 1에 따르면:

* RoIAlign: $mAP^{0.5}=85.2%$, $mAP^{0.7}=61.5%$, $mAP_O^{0.5}=79.2%$, $mAP_O^{0.7}=47.7%$
* RoIPool: $mAP^{0.5}=85.2%$, $mAP^{0.7}=61.1%$, $mAP_O^{0.5}=80.3%$, $mAP_O^{0.7}=50.9%$
* Binary RoIMasking: $mAP^{0.5}=85.5%$, $mAP^{0.7}=62.4%$, $mAP_O^{0.5}=80.1%$, $mAP_O^{0.7}=49.4%$
* Ternary RoIMasking: $mAP^{0.5}=86.7%$, $mAP^{0.7}=63.6%$, $mAP_O^{0.5}=81.2%$, $mAP_O^{0.7}=51.5%$

이 결과는 몇 가지를 말해준다. 먼저 RoIMasking 계열이 전반적으로 기존 RoIAlign/RoIPool보다 좋다. 특히 ternary 버전은 $mAP^{0.7}$ 기준으로 RoIAlign보다 약 2.1포인트 높다. 이는 strict IoU 기준에서 mask 품질이 더 좋다는 뜻이다.

또한 단순 binary masking보다 ternary masking이 더 낫다. 이는 논문이 주장하는 바와 정확히 맞아떨어진다. 즉, **주변을 그냥 더 보는 것만으로는 충분하지 않고, 그 주변이 배경임을 명시적으로 표현해야 효과가 크다**는 것이다. occlusion subset에서도 성능이 올라간다는 점은, ternary masking이 인접 인스턴스 분리에 특히 유리하다는 해석을 뒷받침한다.

### Context region 크기 분석

RoIMasking의 확장 폭은 $\alpha$라는 계수로 조절한다. bounding box 크기가 $(w,h)$이면 유효 영역은

$$
(w + 2\alpha w,; h + 2\alpha h)
$$

가 된다.

실험 결과 $\alpha = 1/3$일 때 가장 좋았다. 표 2에 따르면:

* $\alpha=0$: $mAP^{0.5}=85.9%$, $mAP^{0.7}=62.5%$
* $\alpha=1/6$: $86.4%$, $63.4%$
* $\alpha=1/3$: $86.7%$, $63.6%$
* $\alpha=1/2$: $86.5%$, $63.3%$
* $\alpha=2/3$: $86.2%$, $62.4%$
* $\alpha=1$: $85.9%$, $62.0%$

너무 작으면 주변 문맥을 충분히 못 보고, 너무 크면 다른 salient instance까지 섞여 들어와 오히려 “진짜 대상” 구분이 어려워진다는 설명이 타당하다. 즉, context는 중요하지만 **적절한 범위의 local context**가 중요하다는 결론이다.

### Proposal 수 분석

Segmentation branch에 보내는 proposal 수가 많을수록 성능은 다소 좋아지지만 계산량도 크게 늘어난다. 논문은 20개를 넘으면 개선 폭이 크지 않다고 말한다. 100개 proposal을 쓰면 성능이 약 1.5% 정도만 더 좋아지지만 runtime cost는 크게 증가한다고 한다. 따라서 inference에서는 20 proposals를 사용한다.

이 분석은 논문의 실용 지향성을 보여준다. 단순히 최고 성능만 추구한 것이 아니라, 실제 동작 속도와 정확도 균형을 고려했다.

### Backbone 비교

표 3에서 backbone별 성능과 속도를 비교한다.

* ResNet-101: $mAP@0.5=88.1%$, $mAP@0.7=66.8%$, 33.3 FPS
* ResNet-50: $86.7%$, $63.6%$, 40.0 FPS
* VGG16: $82.2%$, $53.0%$, 43.5 FPS
* MobileNet: $62.9%$, $33.5%$, 90.9 FPS

정확도는 분류 성능이 좋은 backbone일수록 대체로 높다. ResNet-101이 가장 좋지만 느리고, ResNet-50은 성능과 속도의 균형이 좋다. MobileNet은 매우 빠르지만 정확도가 많이 떨어진다. 논문이 주장하는 “실시간”은 주로 **ResNet-50 기반 320×320 입력에서 40 FPS**를 근거로 한다.

### 기존 방법과의 비교

직접 비교 가능한 기존 방법은 사실상 MSRNet뿐이라고 논문은 말한다. 표 4에 따르면 test set에서:

* MSRNet: $mAP^{0.5}=65.3%$, $mAP^{0.7}=52.3%$
* S4Net: $mAP^{0.5}=86.7%$, $mAP^{0.7}=63.6%$

특히 $mAP^{0.5}$에서 약 21포인트 차이가 난다. 이는 상당히 큰 격차다. 논문은 MSRNet이 precomputed edge map 품질에 많이 의존한다는 점을 한계로 지적했는데, S4Net은 end-to-end single-stage 구조로 더 안정적이고 강한 성능을 보인다고 주장한다.

다만 표에서 MSRNet의 occlusion subset 결과는 제공되지 않는다. 이유는 해당 segmentation map과 코드가 उपलब्ध하지 않기 때문이라고 적혀 있다. 따라서 이 부분의 직접 비교는 불가능하다.

### 응용: Weakly Supervised Semantic Segmentation

논문은 S4Net의 결과를 heuristic cue로 사용해 weakly supervised semantic segmentation을 학습시키는 응용도 보여준다. PASCAL VOC validation set에서:

* DeepLab-VGG16 + Sal maps [31]: 49.8%
* DeepLab-VGG16 + Sal maps [26]: 52.6%
* DeepLab-VGG16 + Att [58] + Sal [26]: 53.8%
* DeepLab-VGG16 + Salient Instances: 57.4%
* DeepLab-ResNet101 + Salient Instances: 61.8%

즉, 일반 saliency cue보다 **instance-level saliency cue**가 더 효과적이다. 이는 특히 한 이미지에 여러 object keyword가 존재할 때, 각각을 분리해주는 인스턴스 수준 정보가 semantic supervision의 질을 더 높여준다는 점을 시사한다.

이 응용 결과는 단순 벤치마크 성능을 넘어, S4Net이 다른 vision task의 전처리 또는 pseudo-label 생성에도 가치가 있음을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 방법 설계가 잘 맞아떨어진다는 점이다. salient instance segmentation은 본질적으로 **foreground/background contrast**가 중요한데, 저자들은 이를 CNN 구조 안에 억지로 복잡한 모듈로 넣지 않고, 매우 단순하면서도 해석 가능한 **RoIMasking**으로 구현했다. 단순한 아이디어지만 성능 향상이 분명하고, ablation도 설득력 있게 제시되었다.

둘째, **실용성**이 높다. ResNet-50 기반으로 40 FPS라는 결과는 당시 기준으로 상당히 인상적이다. 고품질 instance segmentation과 실시간 처리 사이 균형을 잘 잡았다는 점이 장점이다.

셋째, 단순 benchmark 성능뿐 아니라 **occlusion subset**, **context region size**, **proposal 개수**, **backbone 변화** 등 설계 선택을 세밀하게 검증했다. 특히 gradient visualization을 통해 ternary masking이 실제로 주변 문맥을 더 활용한다는 정성적 근거를 제공한 점도 좋다.

넷째, 이 방법은 **class-agnostic**한 salient instance를 다루므로, semantic class label에 덜 의존한다. 이는 미리 정의되지 않은 대상이라도 시각적으로 salient하면 분리 대상으로 볼 수 있다는 점에서 응용 범위가 넓다.

반면 한계도 분명하다. 첫째, 데이터셋 규모가 작다. 주 실험 데이터가 1,000장 규모이고, 학습은 500장에 불과하다. 따라서 이 방법의 일반화 능력을 더 큰 규모나 더 다양한 장면에서 충분히 검증했다고 보기는 어렵다.

둘째, 직접 비교 대상이 거의 MSRNet 하나뿐이다. 논문 시점에서 과제가 새롭다는 점은 이해되지만, 그만큼 **경쟁적 비교의 폭이 좁다**. 또한 COCO 같은 대규모 데이터셋에서 체계적 정량평가를 했다는 내용은 본문에 충분히 제시되지 않는다. Figure 7에서 COCO 예시를 보여주지만, 이는 정성 사례에 가깝다.

셋째, 훈련 시 segmentation branch에 ground-truth box를 넣고, 테스트 시에는 predicted box를 넣는 설정은 학습 안정성에는 도움이 되지만, train-test mismatch를 만들 수 있다. 논문은 이것이 유익하다고 말하지만, 예측 box 품질이 나쁠 때 segmentation 성능이 얼마나 민감하게 떨어지는지는 깊게 분석하지 않는다.

넷째, salient instance라는 개념 자체가 데이터셋 정의에 의존적이다. 무엇이 salient한지는 어느 정도 주관적일 수 있으며, 이 논문은 그 정의를 주어진 benchmark annotation에 의존한다. 따라서 다른 데이터나 사용자 목적에서는 saliency 기준이 달라질 수 있다.

다섯째, 방법의 핵심은 foreground/background 분리에 있는데, 이것이 매우 복잡한 배경이나 다수의 salient object가 동시에 있는 장면에서 항상 충분할지는 추가 검증이 필요하다. 논문은 주변 영역이 너무 크면 다른 salient instance가 섞여 성능이 약화된다고 이미 인정한다. 즉, 문맥 활용은 중요하지만 동시에 민감한 설계 요소이기도 하다.

비판적으로 보면, 이 논문은 **“saliency를 instance segmentation 구조에 어떻게 넣을 것인가”**에 대한 좋은 공학적 해법을 제시했지만, saliency 자체의 정의나 annotation 불확실성, 더 큰 데이터에서의 일반화 문제까지 깊게 다루지는 않는다. 그래도 문제의 성격상 충분히 설득력 있는 1차 해법이라고 볼 수 있다.

## 6. 결론

이 논문은 salient instance segmentation이라는 비교적 새로운 문제에 대해, **단일 단계 실시간 구조와 foreground/background separation을 결합한 S4Net**을 제안한다. 핵심 기여는 두 가지다. 하나는 **RoIMasking**으로, proposal 내부뿐 아니라 주변 배경 문맥까지 활용하면서 해상도와 종횡비를 유지하고, ternary masking을 통해 foreground/background contrast를 명시적으로 강화한다. 다른 하나는 receptive field를 키운 segmentation branch 설계로, 같은 RoI 안에서 실제 salient instance를 더 잘 구분하도록 만든 점이다.

실험 결과는 이 설계가 단순한 아이디어에 비해 꽤 강력함을 보여준다. RoIAlign, RoIPool 대비 개선이 있고, MSRNet보다 큰 폭의 성능 향상을 달성했으며, occlusion이 있는 상황에서도 유리하다. 또한 weakly supervised semantic segmentation의 heuristic cue로 활용했을 때도 일반 saliency cue보다 더 좋은 성능을 보였다.

종합하면, 이 논문은 “proposal 주변 문맥과 배경 대비를 명시적으로 모델링하면 salient instance segmentation이 좋아진다”는 메시지를 분명하게 전달한다. 이후 더 강력한 backbone, 더 큰 데이터, 더 정교한 instance-level attention이나 context modeling과 결합하면 확장 가능성도 높다. 실제 응용 측면에서도 이미지 편집, 로봇 비전, 약지도 학습 같은 분야에 의미 있는 기반을 제공할 수 있는 연구다.
