# YOLACT: Real-time Instance Segmentation

* **저자**: Daniel Bolya, Chong Zhou, Fanyi Xiao, Yong Jae Lee
* **발표연도**: 2019
* **arXiv**: [https://arxiv.org/abs/1904.02689](https://arxiv.org/abs/1904.02689)

## 1. 논문 개요

이 논문은 **real-time instance segmentation**을 목표로 제안된 YOLACT를 소개한다. 핵심 문제의식은 당시의 대표적 instance segmentation 방법들이 정확도는 높지만, 속도가 매우 느려 실시간 응용에 적합하지 않다는 점이다. 예를 들어 Mask R-CNN 계열 방법은 bounding box 후보를 먼저 만들고, 각 후보 영역에 대해 feature를 다시 모으는 re-pooling 단계(RoI Pooling / RoI Align 등)를 거친 뒤 mask를 예측한다. 이 과정은 구조적으로 순차적이어서 빠르게 만들기 어렵다.

논문은 이 병목을 피하기 위해 instance segmentation을 두 개의 병렬 문제로 나눈다. 첫째는 이미지 전체에 대해 공통으로 쓰일 **prototype masks**를 만드는 것이고, 둘째는 각 instance마다 그 prototype들을 얼마나 섞을지를 나타내는 **mask coefficients**를 예측하는 것이다. 최종 instance mask는 이 둘을 선형 결합한 뒤 bounding box로 crop하여 얻는다. 즉, 기존 방식처럼 각 객체마다 별도로 feature를 잘라서 mask를 만들지 않고, 전체 이미지 단위에서 공통 mask basis를 만든 뒤 객체별 coefficient만 예측한다.

이 문제가 중요한 이유는 분명하다. object detection에서는 YOLO나 SSD처럼 빠른 one-stage detector가 성공했지만, instance segmentation에서는 이에 대응하는 수준의 실시간 방법이 부족했다. 논문은 이 공백을 메우려 하며, 실제로 COCO에서 약 30 mask mAP 수준을 유지하면서 30 FPS를 넘는 성능을 달성했다고 주장한다. 이는 자율주행, 로보틱스, 실시간 비디오 분석처럼 속도가 매우 중요한 환경에서 큰 의미가 있다.

## 2. 핵심 아이디어

YOLACT의 중심 아이디어는 **instance mask를 직접 예측하지 않고, 미리 생성된 공통 prototype들의 선형 결합으로 표현한다**는 점이다. 이 아이디어는 mask prediction에서 spatial coherence와 semantic representation을 분리해 다루는 데 있다. 논문은 convolution이 공간적으로 일관된 구조를 잘 표현하는 반면, one-stage detector의 anchor별 prediction head는 본질적으로 coefficient 같은 벡터를 예측하는 데 더 적합하다고 본다. 따라서 공간 구조는 prototype branch가 담당하고, 각 객체별 의미적 조합은 coefficient branch가 담당하게 설계했다.

기존 접근과의 가장 큰 차별점은 **explicit localization 없이도 instance mask를 형성한다**는 점이다. Mask R-CNN이나 FCIS 같은 기존 방법들은 translation variance를 확보하기 위해 region-based localization 또는 position-sensitive mechanism을 적극적으로 사용했다. 반면 YOLACT는 이미지 전역 prototype과 instance별 coefficient만으로 mask를 조립한다. localization은 주로 마지막 crop 단계에서 보조되며, 논문은 이보다 더 중요한 것은 prototype들이 스스로 위치 정보를 학습한다는 점이라고 분석한다.

또 하나의 중요한 차별점은 **prototype 수가 category 수에 종속되지 않는다**는 점이다. 즉, 클래스마다 별도의 mask predictor를 두는 것이 아니라, 여러 클래스가 공통 prototype 집합을 공유한다. 결과적으로 YOLACT는 category-specific한 표현보다 더 분산된(distributed) 표현을 학습하게 되며, 일부 prototype은 이미지 공간을 분할하고, 일부는 object contour를 강조하고, 일부는 배경 또는 방향성 정보를 나타내는 등 흥미로운 emergent behavior를 보인다.

또한 논문은 단순히 mask 생성 방식만 제안한 것이 아니라, 실시간화를 위해 **Fast NMS**도 함께 제안한다. 이는 전통적인 sequential NMS를 병렬화한 근사 방식으로, 속도를 크게 높이면서 정확도 하락은 거의 없도록 설계되었다.

## 3. 상세 방법 설명

### 전체 파이프라인

YOLACT는 크게 세 부분으로 이해할 수 있다.

첫째는 backbone detector이다. 논문은 RetinaNet 계열 구조를 바탕으로 ResNet-101 + FPN을 기본 backbone으로 사용한다. 이 detector는 일반적인 one-stage detector처럼 각 anchor에 대해 class confidence와 bounding box regression을 예측한다.

둘째는 **prototype generation branch**, 즉 protonet이다. 이 branch는 이미지 전체를 대상으로 $k$개의 prototype mask를 생성한다. 이 prototype은 특정 instance에 직접 대응하지 않으며, 전체 이미지에서 공유되는 mask basis라고 볼 수 있다.

셋째는 **mask coefficient branch**이다. 이는 각 anchor에 대해 길이 $k$의 coefficient vector를 예측한다. NMS 이후 살아남은 각 detection마다 이 coefficient를 사용하여 prototype들을 선형 결합하면 해당 instance의 mask가 만들어진다.

즉, 최종적으로 detector는 box와 class를 예측하고, 동시에 각 instance가 prototype space에서 어떤 조합으로 표현될지를 나타내는 coefficient를 출력한다.

### Prototype generation branch

Prototype branch는 FCN 형태로 구성된다. 마지막 레이어가 $k$개 채널을 가지며, 각 채널이 하나의 prototype mask에 해당한다. 논문은 이 branch를 FPN의 큰 해상도를 가지는 깊은 feature map인 $P_3$에 붙인다. 그리고 small object 성능을 높이기 위해 prototype의 해상도를 입력 이미지의 $1/4$ 크기까지 upsample한다.

중요한 점은 prototype에 대해 별도의 직접 supervision을 주지 않는다는 것이다. 즉, "이 prototype은 무엇을 나타내야 한다" 같은 loss는 없다. 오직 최종 mask loss를 통해 간접적으로만 학습된다. 그럼에도 불구하고 다양한 공간 분할, 윤곽선, 배경, 방향성 정보를 나타내는 prototype이 자발적으로 형성된다.

논문은 prototype output이 **unbounded**인 것이 중요하다고 말한다. 특정 prototype이 매우 강한 activation을 가질 수 있어야 background 같은 명확한 패턴을 강하게 표현할 수 있기 때문이다. 실험에서는 prototype을 더 해석 가능하게 만들기 위해 ReLU를 선택했다고 설명한다.

### Mask coefficient branch

기존 anchor-based detector는 보통 class branch와 box branch를 가진다. YOLACT는 여기에 세 번째 branch로 mask coefficient prediction을 추가한다. 따라서 anchor 하나당 출력은 기존의 class score와 box regression뿐 아니라 $k$차원 coefficient vector까지 포함한다.

계수에 대해서는 **tanh** 비선형을 사용한다. 이유는 최종 mask를 만들 때 어떤 prototype을 더하는 것뿐 아니라 **빼는 것**도 가능해야 하기 때문이다. 실제로 논문은 subtraction이 가능해야 특정 instance를 다른 인접 instance와 분리하는 데 유리하다고 설명한다. coefficient가 모두 양수만 되면 표현력이 줄어들 수 있다.

### Mask assembly

YOLACT의 가장 핵심적인 조립 단계는 매우 단순하다. prototype tensor $P$와 coefficient matrix $C$가 있을 때, 살아남은 $n$개의 instance mask는 다음과 같이 계산된다.

$$
M = \sigma(PC^T)
$$

여기서 $P$는 크기 $h \times w \times k$의 prototype mask 집합이고, $C$는 크기 $n \times k$의 coefficient 행렬이다. 각 instance의 mask는 $k$개의 prototype에 대한 가중합으로 얻어진 뒤 sigmoid를 거쳐 확률 mask가 된다.

이 식의 의미는 직관적이다. 각 instance는 "prototype 1은 조금 더하고, prototype 2는 빼고, prototype 3은 강하게 더한다" 같은 방식으로 표현된다. 그리고 이 조합을 모든 픽셀 위치에 대해 동일하게 적용하면 full-image mask가 얻어진다. 이 연산은 GPU에서 matrix multiplication으로 매우 빠르게 수행할 수 있다.

### Crop 단계

조립된 mask는 이미지 전역 크기에서 만들어지므로, 최종적으로는 예측된 bounding box를 이용해 crop한다. 평가 시에는 예측 box로 crop하고, 학습 시에는 ground truth bounding box로 crop한다. 또한 학습 시 mask loss를 ground truth box area로 나누어 작은 객체가 loss에서 묻히지 않도록 한다.

이 crop은 YOLACT에서 localization을 보조하는 유일한 명시적 장치에 가깝다. 그러나 논문은 medium/large object에서는 crop이 없어도 어느 정도 동작한다고 하며, 이는 prototype 자체가 위치 정보를 암묵적으로 학습했기 때문이라고 해석한다.

### Loss function과 학습 목표

YOLACT는 세 가지 기본 loss를 사용한다.

분류 loss는 $L_{cls}$, box regression loss는 $L_{box}$, mask loss는 $L_{mask}$이다. 가중치는 각각 1, 1.5, 6.125로 설정되었다.

Mask loss는 조립된 mask와 ground truth mask 사이의 pixel-wise binary cross entropy이다.

$$
L_{mask} = \text{BCE}(M, M_{gt})
$$

즉, 각 instance마다 최종적으로 만들어진 mask가 정답 mask와 얼마나 일치하는지를 직접 학습한다. 중요한 점은 prototype 자체나 coefficient 자체에 대한 별도 정규화 항은 논문 본문에서 명시되지 않았다는 것이다. supervision은 최종 mask error를 통해서만 들어간다.

추가로 box regression에는 smooth-$L_1$ loss를 사용하고, class prediction에는 softmax cross entropy를 사용한다. 학습 샘플 선택에는 OHEM을 사용하며, neg:pos 비율은 3:1이다. RetinaNet과 달리 focal loss는 이 설정에서 적절하지 않았다고 논문은 보고한다.

### Emergent behavior에 대한 해석

논문에서 흥미로운 부분은 왜 fully-convolutional prototype으로 instance를 분리할 수 있는지에 대한 설명이다. 일반적으로 FCN은 translation invariant 성질이 강하므로 instance localization에는 불리하다고 여겨졌다. 그런데 YOLACT는 padding으로 인해 네트워크가 이미지의 가장자리로부터의 거리를 간접적으로 인식할 수 있고, 그 결과 translation-variant한 반응을 어느 정도 학습할 수 있다고 본다.

실제로 논문은 일부 prototype이 이미지의 한쪽 영역에만 활성화되는 partition-like behavior를 보인다고 설명한다. 예를 들어 같은 class의 두 umbrella가 겹쳐 있을 때, 어떤 prototype은 왼쪽 객체에, 다른 prototype은 오른쪽 객체에 더 강하게 반응한다. 그러면 coefficient 조합을 통해 서로 다른 instance를 분리할 수 있다.

또한 prototype 수 $k$를 너무 크게 늘려도 성능 향상이 크지 않다고 분석한다. 이유는 coefficient prediction 자체가 어려운 문제이기 때문이다. 선형 결합에서는 coefficient 하나만 크게 틀려도 mask가 사라지거나 leakage가 생길 수 있으므로, 너무 많은 prototype은 오히려 예측을 어렵게 만들 수 있다.

### Backbone detector와 실용적 설계

기본 detector는 ResNet-101 + FPN, 입력 크기 $550 \times 550$이다. anchor는 각 feature layer에 3개 aspect ratio $[1, 1/2, 2]$를 둔다. $P_3$ anchor area는 $24^2$이고, 이후 layer마다 scale을 2배씩 키운다.

Prediction head는 RetinaNet보다 더 얕고 가볍게 설계했다. 하나의 shared $3 \times 3$ conv 뒤에 각 branch별로 별도 $3 \times 3$ conv를 둔다. 이는 speed를 중시한 선택이다.

또한 논문은 학습 시에만 사용되는 **semantic segmentation loss**를 추가한다. $P_3$ feature map에 $1 \times 1$ conv를 붙여 class별 semantic prediction을 하고, test time에는 이 branch를 제거한다. 이 보조 loss는 속도 저하 없이 약 $+0.4$ mAP 향상을 준다고 보고한다.

### Fast NMS

Fast NMS는 전통적인 NMS의 sequential dependency를 완화한 방식이다. 각 class마다 상위 $n$개 detection을 score 순으로 정렬한 뒤, pairwise IoU matrix $X$를 만든다. lower triangle과 diagonal을 0으로 만든 후, 각 detection에 대해 자신보다 높은 score를 가진 detection들과의 최대 IoU를 계산한다.

논문에서 제시한 식은 다음과 같다.

$$
K_{kj} = \max_i (X_{kij}) \qquad \forall k, j
$$

여기서 $K_{kj}$가 threshold $t$보다 작으면 그 detection은 유지된다. 전통적 NMS와 달리 이미 제거된 box도 다른 box를 suppress할 수 있도록 허용하는 relaxation이 들어간다. 이 때문에 약간 더 많은 box를 제거할 수는 있지만, 전체적으로는 정확도 저하가 거의 없고 속도 향상이 매우 크다.

## 4. 실험 및 결과

### 데이터셋과 평가 설정

주요 실험은 **MS COCO**와 **Pascal 2012 SBD**에서 수행되었다. COCO에서는 train2017으로 학습하고 val2017 및 test-dev에서 평가한다. Pascal에서는 별도 설정으로 학습한다.

COCO의 main metric은 mask AP 계열 지표이며, $AP$, $AP_{50}$, $AP_{75}$, 그리고 object size별 $AP_S$, $AP_M$, $AP_L$가 보고된다. Pascal에서는 논문에 제시된 $mAP^r_{50}$, $mAP^r_{70}$를 사용한다.

### 구현 세부 사항

모델은 ImageNet pretrained weight로 초기화하고, batch size 8로 단일 GPU에서 학습한다. COCO에서는 SGD로 800k iterations 학습하며, 초기 learning rate는 $10^{-3}$이다. 일정 시점마다 learning rate를 10으로 나누고, weight decay는 $5 \times 10^{-4}$, momentum은 0.9를 사용한다. 데이터 증강은 SSD와 동일한 방식을 쓴다.

이 논문이 강조하는 실용적 포인트 중 하나는 **단일 GPU 학습**이다. 복잡한 대규모 분산학습 없이도 제안 성능을 낼 수 있다는 점을 내세운다.

### COCO main results

논문의 핵심 결과는 COCO test-dev에서 **YOLACT-550, ResNet-101-FPN이 29.8 mask AP를 33.5 FPS로 달성**했다는 것이다. 같은 표에서 Mask R-CNN은 35.7 AP지만 8.6 FPS, FCIS는 29.5 AP에 6.6 FPS, RetinaMask는 34.7 AP에 6.0 FPS이다. 즉, YOLACT는 최고 정확도는 아니지만, 이전 경쟁 방법보다 월등히 빠르다.

논문은 이를 "이전 가장 빠른 경쟁 방법보다 약 3.9배 빠르다"고 요약한다. 실제 수치상 YOLACT-550이 33.5 FPS이고 Mask R-CNN 계열이 대체로 8.6 FPS 수준이므로 실시간 응용 관점에서 차이가 매우 크다.

다른 설정도 함께 보고된다. YOLACT-400은 45.3 FPS로 더 빠르지만 AP가 24.9로 크게 감소한다. 반대로 YOLACT-700은 31.2 AP로 향상되지만 23.4 FPS로 느려진다. 따라서 instance segmentation에서는 입력 해상도가 성능에 매우 큰 영향을 준다는 점을 보여준다.

Backbone을 바꾼 실험에서는 YOLACT-550 + ResNet-50이 28.2 AP에 45.0 FPS, DarkNet-53-FPN이 28.7 AP에 40.7 FPS를 기록한다. 논문은 더 빠른 설정이 필요할 경우 입력 크기를 줄이기보다 backbone을 바꾸는 편이 더 낫다고 해석한다.

### Ablation study

#### Fast NMS 효과

표에 따르면 standard NMS를 쓴 YOLACT는 30.0 AP, 24.0 FPS이고, Fast NMS를 쓰면 29.9 AP, 33.5 FPS가 된다. 즉 AP는 0.1 정도만 감소하지만 속도는 크게 증가한다. Mask R-CNN에서도 유사한 경향이 관찰되며, standard 36.1 AP / 8.6 FPS에서 Fast NMS 적용 시 35.8 AP / 9.9 FPS가 된다.

이는 Fast NMS가 YOLACT 전용 기법이 아니라, 다른 detector에도 적용 가능한 실용적 최적화임을 보여준다.

#### Prototype 수 $k$

$k=8,16,32,64,128,256$에 대한 실험에서 성능은 26.8, 27.1, 27.7, 27.8, 27.6, 27.7 AP 정도로 큰 차이가 없다. 속도는 $k$가 커질수록 조금씩 감소한다. 논문은 이 결과를 바탕으로 **$k=32$가 속도와 성능의 균형점**이라고 선택한다.

이 결과는 YOLACT의 핵심 가정, 즉 적은 수의 shared prototype으로도 충분히 instance mask를 표현할 수 있다는 점을 뒷받침한다.

#### Baseline 비교

논문은 fc layer에서 바로 작은 mask를 뽑는 단순한 one-stage baseline인 fc-mask를 제시한다. 이 모델은 25.7 FPS로 어느 정도 빠르지만 AP가 20.7에 그친다. 반면 YOLACT-550은 33.0 FPS에 29.9 AP를 달성한다. 이는 단순히 "one-stage detector에 mask output을 붙이는 것"만으로는 부족하고, prototype-coefficient 분해가 핵심이라는 점을 보여준다.

### Pascal SBD 결과

Pascal 2012 SBD에서는 YOLACT-550 + ResNet-50-FPN이 47.6 FPS, $mAP^r_{50}=72.3$, $mAP^r_{70}=56.2$를 기록한다. FCIS는 9.6 FPS에서 65.7 / 52.1, MNC는 2.8 FPS에서 63.5 / 41.5이다. Pascal은 COCO보다 클래스 수와 장면 복잡도가 낮기 때문에 YOLACT가 더 강한 상대적 이점을 보인다고 논문은 해석한다.

### Mask quality와 AP 해석

논문은 흥미롭게도 전체 AP는 Mask R-CNN보다 낮지만, high-IoU 영역에서는 mask quality가 더 좋을 수 있다고 주장한다. 예를 들어 base model에서 $AP_{95}$는 YOLACT가 1.6, Mask R-CNN이 1.3이다. 또한 $AP_{50}$ 차이는 크지만 $AP_{75}$ 차이는 조금 줄어든다. 논문은 이를 통해 re-pooling이 large object mask quality를 떨어뜨릴 수 있다고 해석한다.

즉, YOLACT의 약점은 mask generation 자체보다 detector 품질에 더 가깝다고 본다. 실제로 base model에서 box AP는 32.3, mask AP는 29.8로 차이가 2.5에 불과하다. Mask R-CNN도 box AP와 mask AP 차이가 비슷하므로, 두 방법의 총 AP 차이 상당 부분은 detector accuracy 격차에서 온다는 것이 논문의 주장이다.

### Temporal stability

논문은 정량적 metric 대신 정성적으로 video에서 temporal stability를 언급한다. static image만으로 학습했음에도 YOLACT의 mask는 frame-to-frame jitter가 적다고 주장한다. 이유는 prototype이 이미지 전역에서 안정적으로 형성되고, two-stage 방식처럼 proposal 변화에 mask가 민감하게 흔들리지 않기 때문이라는 설명이다.

다만 이 부분은 본문에 정량 실험 테이블이 제시된 것은 아니며, 주로 qualitative observation 수준으로 제시된다. 따라서 장점으로 볼 수는 있지만, 엄밀한 수치 비교까지 논문이 제공했다고 보기는 어렵다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 분명하다. **instance segmentation을 실시간으로 만든 매우 단순하고 우아한 구조**를 제시했다는 점이다. 기존 방법들이 localization, repooling, mask head, post-processing에 많은 비용을 쓰는 반면, YOLACT는 prototype과 coefficient의 선형 결합이라는 간단한 아이디어로 문제를 다시 정의했다. 이 단순성이 곧 높은 속도로 이어진다.

또한 mask assembly가 거의 matrix multiplication 하나로 구현된다는 점은 구현 및 가속 측면에서 매우 강력하다. 실제로 논문은 mask branch 전체가 약 5~6 ms 정도의 추가 비용만 가진다고 설명한다. 이는 "mask를 붙여도 거의 detection 수준 속도를 유지할 수 있다"는 의미다.

두 번째 강점은 **emergent prototype behavior에 대한 분석**이다. 단순히 성능 숫자만 보여주는 것이 아니라, prototype이 이미지 partition, contour, background, direction-like map을 자발적으로 학습한다는 점을 시각적으로 분석한다. 이는 제안 방식이 단순한 heuristic이 아니라, 의미 있는 내부 표현을 학습한다는 근거가 된다.

세 번째 강점은 **Fast NMS와 semantic segmentation loss 같은 실용적 개선 요소**를 함께 제공했다는 점이다. 논문은 단순한 핵심 아이디어 외에도, 실제 배포 가능한 고속 모델을 만들기 위해 어디서 시간을 줄이고 어디서 성능을 보완하는지를 구체적으로 보여준다.

하지만 한계도 분명하다. 첫째, 전체 AP는 여전히 당시 최고 성능 방법보다 낮다. 논문이 강조하듯 이 격차의 상당 부분은 detector 때문일 수 있으나, 사용자의 관점에서는 결국 최종 성능 격차로 나타난다.

둘째, 논문이 직접 언급한 대표적 실패 사례는 **localization failure**이다. 밀집된 장면에서 여러 객체가 한 공간에 몰려 있으면, prototype이 개별 instance를 충분히 분리하지 못하고 foreground-like mask로 뭉개질 수 있다. 이는 shared prototype 기반 표현의 구조적 약점으로 볼 수 있다.

셋째, **leakage** 문제가 있다. YOLACT는 crop 이후의 box를 믿고 mask 바깥쪽 noise를 적극적으로 억제하지 않는다. 따라서 box가 부정확하면 멀리 있는 다른 객체의 일부 mask가 섞여 들어올 수 있다. 다시 말해, mask가 box quality에 상당히 의존한다.

넷째, temporal stability 주장은 흥미롭지만 정량적 검증이 충분하지 않다. qualitative하게 설득력은 있으나, 실제 비디오 benchmark에서 얼마나 안정적인지에 대한 엄밀한 수치는 본문에서 제공되지 않는다.

다섯째, prototype의 해석 가능성과 localization 원리에 대한 설명은 흥미롭지만, 어디까지나 관찰적 분석에 가깝다. 왜 이런 prototype이 형성되는지에 대한 이론적 보장은 부족하다. 또한 padding-induced translation variance 설명은 plausible하지만 완전히 엄밀한 수학적 증명은 아니다.

비판적으로 보면, 이 논문의 핵심 기여는 "최고 성능의 segmentation 방법"이라기보다 "속도-성능 trade-off를 근본적으로 재설계한 방법"에 더 가깝다. 따라서 accuracy 절대값만 보면 아쉬움이 있지만, 당시 문제 설정을 바꾸었다는 점에서 학술적 가치가 크다.

## 6. 결론

YOLACT는 instance segmentation을 **prototype mask 생성**과 **instance-specific coefficient 예측**이라는 두 병렬 문제로 분해함으로써, explicit repooling 없이도 빠른 one-stage instance segmentation을 가능하게 했다. 최종 mask는

$$
M = \sigma(PC^T)
$$

라는 매우 단순한 형태로 조립되며, 이 단순성이 실시간 성능의 핵심이다.

논문의 주요 기여는 세 가지로 요약할 수 있다. 첫째, COCO에서 약 30 AP 수준을 유지하면서 30 FPS를 넘는 최초의 경쟁력 있는 real-time instance segmentation 시스템을 제시했다. 둘째, prototype 기반 mask representation이라는 새로운 관점을 제안하고, 그 내부 표현의 emergent behavior를 분석했다. 셋째, Fast NMS와 training-only semantic segmentation loss 등 실제 시스템 성능을 끌어올리는 실용적 기법을 함께 제시했다.

이 연구의 중요성은 단지 YOLACT라는 단일 모델에 그치지 않는다. 논문 스스로도 언급하듯, prototype + coefficient라는 발상은 다른 detector와도 결합 가능하다. 실제로 이후 계열 연구들에서 이 아이디어는 매우 큰 영향을 미쳤다. 실시간 비전 시스템, 로봇 인지, 비디오 처리 등에서 instance segmentation을 실제로 사용할 수 있게 만드는 방향을 제시했다는 점에서, 이 논문은 정확도 경쟁과는 다른 축에서 매우 중요한 의미를 가진다.
