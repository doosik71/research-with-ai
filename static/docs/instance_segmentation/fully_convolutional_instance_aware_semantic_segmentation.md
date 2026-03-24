# Fully Convolutional Instance-aware Semantic Segmentation

* **저자**: Yi Li, Haozhi Qi, Jifeng Dai, Xiangyang Ji, Yichen Wei
* **발표연도**: 2016
* **arXiv**: <https://arxiv.org/abs/1611.07709>

## 1. 논문 개요

이 논문은 **instance-aware semantic segmentation**, 즉 이미지 안의 각 물체를 **클래스별로 구분**하는 동시에 **개별 인스턴스 단위의 mask**까지 예측하는 문제를 다룬다. 일반적인 semantic segmentation은 픽셀마다 클래스 라벨만 부여하므로 같은 클래스의 여러 객체를 서로 구분하지 못한다. 반면 이 문제는 “무엇인지”와 “어디까지가 한 개체인지”를 동시에 맞혀야 한다.

저자들은 당시 널리 쓰이던 방법들이 보통 세 단계 구조를 가진다고 지적한다. 먼저 이미지 전체에서 convolutional feature를 뽑고, 그 위에 ROI pooling으로 각 proposal의 고정 크기 feature를 만들고, 마지막으로 per-ROI fully connected layer나 별도 sub-network를 통해 mask와 classification을 수행한다. 이 방식은 구조적으로 자연스럽지만, ROI warping 과정에서 공간 정보가 손실되고, 큰 fc layer가 mask 예측을 담당하면서 파라미터가 비대해지며, 각 ROI마다 별도 계산이 필요해 속도가 느려지는 문제가 있다.

이 논문의 핵심 목표는 이런 한계를 넘어서, **FCN의 장점인 fully convolutional, end-to-end, shared computation**을 유지하면서도 instance-level segmentation이 가능하도록 만드는 것이다. 이를 위해 저자들은 **FCIS (Fully Convolutional Instance-aware Semantic Segmentation)** 를 제안한다. 이 모델은 detection과 segmentation을 분리하지 않고, 하나의 fully convolutional 구조 안에서 **jointly and simultaneously** 수행한다.

문제의 중요성은 분명하다. instance segmentation은 autonomous driving, robotics, image editing, scene understanding처럼 객체 단위 이해가 필요한 거의 모든 비전 응용에서 중요하다. 이 논문은 단순히 정확도를 조금 높인 수준이 아니라, 당시의 파이프라인 중심 접근을 fully convolutional 방향으로 재구성했다는 점에서 의미가 크다.

## 2. 핵심 아이디어

논문의 중심 직관은 다음과 같다. 일반적인 FCN은 **translation invariant** 하다. 즉, 같은 모양의 패턴이 이미지 어디에 있어도 비슷한 응답을 낸다. semantic segmentation에는 이것이 장점이지만, instance segmentation에서는 문제가 된다. 같은 픽셀이라도 어떤 ROI 안에서 보느냐에 따라 “이 인스턴스의 foreground인지 아닌지”가 달라질 수 있기 때문이다. 다시 말해, 인스턴스 단위 문제는 어느 정도 **translation-variant** 성질을 필요로 한다.

이를 해결하기 위해 저자들은 기존 InstanceFCN의 **position-sensitive score maps** 개념을 발전시킨다. 객체를 $k \times k$ 격자로 나누고, 각 상대적 위치에 대해 별도 score map을 두면, 픽셀이 객체 안에서 어느 상대 위치에 있는지에 따라 다른 점수를 받을 수 있다. 예를 들어 같은 픽셀도 어떤 ROI에서는 “top-left part of an object”로 해석되고, 다른 ROI에서는 “background”로 해석될 수 있다.

하지만 이 논문의 진짜 차별점은 단순히 position-sensitive map을 쓰는 데서 끝나지 않는다. 저자들은 **mask prediction과 classification을 같은 score map 집합으로 함께 푼다**. 즉, segmentation branch와 detection branch를 따로 두지 않고, **inside/outside**라는 두 종류의 position-sensitive map만으로 두 작업을 동시에 해결한다.

이 joint formulation은 매우 인상적이다. 보통 detection은 “이 ROI가 어떤 클래스의 객체인가”를 묻고, segmentation은 “ROI 안 각 픽셀이 객체 내부인가”를 묻는다. 저자들은 이 둘이 본질적으로 강하게 연결되어 있다고 보고, 하나의 표현으로 함께 풀어낸다. 그 결과 추가 파라미터 없이 두 과업이 같은 convolutional representation을 공유하고, ROI별 계산은 단순한 assembling, softmax, max, average pooling 정도로 끝난다.

기존 접근과의 차별점은 세 가지로 요약할 수 있다. 첫째, **ROI pooling 후 fc layer** 중심이 아니라 **fully convolutional shared computation** 중심이다. 둘째, segmentation과 detection을 **순차적 pipeline** 으로 풀지 않고 **joint formulation** 으로 풀었다. 셋째, mask prediction을 고정된 $28 \times 28$ 같은 저해상도 fc 출력으로 만들지 않고, ROI의 실제 위치 정보와 aspect ratio를 더 자연스럽게 보존하는 방식으로 수행했다.

## 3. 상세 방법 설명

### 3.1 Position-sensitive score maps의 의미

기존 FCN에서는 클래스마다 하나의 score map만 예측한다. 예를 들어 “person” 클래스 score map은 각 픽셀이 person인지 아닌지를 알려준다. 그러나 이것만으로는 같은 클래스의 인접한 두 사람을 구분하기 어렵다.

FCIS는 각 클래스를 하나의 score map으로 보지 않고, 객체를 $k \times k$ 셀로 나눈 상대 위치별 map으로 본다. 기본 실험에서는 $k=7$ 이다. 즉, 한 클래스당 $7 \times 7 = 49$ 개의 위치 민감 score map이 필요하다. 이 map의 각 채널은 “객체 내부의 특정 상대 위치”를 담당한다.

ROI가 주어지면, ROI를 동일하게 $k \times k$ 셀로 나누고, 각 셀은 대응되는 score map에서 값을 가져온다. 논문은 이를 assembling 또는 copy-paste로 설명한다. 이 과정 덕분에 같은 픽셀도 ROI가 달라지면 다른 score를 받게 되고, translation-variant 성질이 생긴다.

### 3.2 Joint mask prediction and classification

논문의 핵심은 이 부분이다. ROI 안 각 픽셀에 대해 저자들은 두 가지 질문을 동시에 고려한다.

첫 번째는 detection 관점의 질문이다.
이 픽셀이 “어떤 객체 bounding box의 상대 위치에 속하는가” 아니면 “아무 객체에도 해당하지 않는가”를 본다.

두 번째는 segmentation 관점의 질문이다.
이 픽셀이 “그 객체 인스턴스의 boundary 안쪽인가” 아니면 “바깥인가”를 본다.

저자들은 이 두 질문을 따로 예측하는 대신, **inside** 와 **outside** 두 점수로 묶는다. ROI 내 한 픽셀에 대해 가능한 해석은 세 가지다.

1. inside 높고 outside 낮음: detection+, segmentation+
2. inside 낮고 outside 높음: detection+, segmentation-
3. 둘 다 낮음: detection-, segmentation-

즉, 객체 ROI의 일부이지만 실제 mask 바깥일 수 있고, 아예 객체 ROI 자체가 아닐 수도 있다.
이때 segmentation은 inside와 outside를 픽셀 단위로 비교하면 되고, detection은 “inside 또는 outside 중 하나라도 강하면 객체 관련 픽셀”로 볼 수 있다.

논문 설명을 따라 정리하면 다음과 같다.

* **Segmentation score**: 각 픽셀에서 inside와 outside 사이에 softmax를 적용하여 foreground probability를 얻는다.
* **Detection score**: 각 픽셀에서 inside/outside 중 큰 값을 택해 detection likelihood를 만들고, ROI 전체에 대해 average pooling하여 category score를 계산한다. 이후 카테고리 간 softmax를 적용한다.

이 설계의 장점은 detection과 segmentation이 완전히 분리된 branch가 아니라는 점이다. 같은 inside/outside score map이 두 손실에서 동시에 학습되므로, 두 과업의 상호 보완성이 자연스럽게 반영된다.

### 3.3 네트워크 아키텍처

전체 구조는 크게 세 부분으로 이해할 수 있다.

첫째, backbone convolutional network이다.
저자들은 ResNet을 사용한다. 원래 ResNet의 마지막 feature stride는 32인데, 이는 instance segmentation에는 너무 거칠다. 그래서 conv5 첫 block의 stride를 2에서 1로 줄이고, receptive field를 유지하기 위해 dilation 2를 적용하는 **à trous / hole algorithm** 을 사용한다. 그 결과 최종 feature stride를 16으로 낮춘다.

둘째, region proposal network (RPN)이다.
RPN은 conv4 feature 위에 붙어 ROI를 생성한다. 이 RPN 역시 fully convolutional이며, FCIS와 feature를 공유한다.

셋째, score map head이다.
conv5 feature에서 $1 \times 1$ convolution을 사용해

$$
2k^2(C+1)
$$

개의 score map을 만든다.

여기서 $C$ 는 object category 수이고, 배경 1개를 포함하므로 총 $C+1$ categories를 고려한다. 각 category마다 inside/outside 두 세트가 있고, 세트마다 $k^2$ 개의 position-sensitive map이 필요하므로 위 식이 나온다.

또한 bbox regression을 위해 별도의 sibling $1 \times 1$ convolution layer를 추가하여

$$
4k^2
$$

채널의 bounding box shift를 예측한다. 이 부분도 fully convolutional feature에서 바로 계산된다.

### 3.4 Inference 절차

추론은 다음 순서로 이루어진다.

먼저 RPN이 상위 300개의 ROI를 생성한다.
이 ROI들은 bbox regression branch를 통과해 refinement된 300개 ROI를 추가로 만들고, 결과적으로 총 600개 수준의 후보가 활용된다.

각 ROI에 대해 모델은 다음을 출력한다.

* 모든 category에 대한 classification score
* 모든 category에 대한 foreground mask probability

그 후 IoU threshold 0.3으로 NMS를 적용해 중복 ROI를 제거한다. 살아남은 ROI는 가장 높은 classification score의 category로 분류된다.

최종 mask는 논문이 사용하는 **mask voting** 으로 얻는다. 어떤 ROI에 대해 IoU가 0.5보다 큰 다른 ROI들의 mask를 모아, classification score를 가중치로 사용해 픽셀 단위 평균을 낸다. 마지막에 이를 binarize하여 최종 mask로 출력한다.

즉, 단일 ROI 결과를 그대로 쓰지 않고 주변의 유사 후보를 모아 안정화한다는 뜻이다.

### 3.5 학습 목표와 학습 절차

각 ROI는 가장 가까운 ground-truth object와의 box IoU가 0.5보다 크면 positive, 아니면 negative이다.

각 ROI에 대해 세 가지 loss가 사용된다.

1. **Detection loss**: $C+1$ categories에 대한 softmax loss
2. **Segmentation loss**: ground-truth category에 대해서만 ROI 내부 픽셀에 대한 softmax loss
3. **BBox regression loss**: bounding box refinement loss

논문은 세 loss를 **equal weights** 로 둔다고 설명한다. 또한 segmentation loss와 bbox regression loss는 positive ROI에서만 적용된다.

수식 형태를 논문이 명시적으로 자세히 쓰지는 않았지만, 구조적으로는 다음처럼 이해할 수 있다.

$$
L = L_{\text{det}} + L_{\text{seg}} + L_{\text{bbox}}
$$

여기서 $L_{\text{seg}}$ 는 ROI 내 픽셀별 손실의 합을 ROI 크기로 정규화한 값이다.

학습은 ImageNet pre-trained model로 초기화하고, 새로 추가된 층은 random initialization을 사용한다.
입력 이미지는 짧은 변이 600 pixel이 되도록 resize한다.
최적화는 SGD를 사용하고, 8개의 GPU에서 각 GPU당 이미지 1장 mini-batch로 학습한다.

또한 ROI별 계산 비용이 매우 작기 때문에, 저자들은 **OHEM (Online Hard Example Mining)** 을 효과적으로 적용한다. 한 이미지에서 제안된 300개의 ROI 전체에 forward를 수행하고, 손실이 큰 128개 ROI만 골라 backward를 수행한다. 이 부분은 per-ROI sub-network가 비싼 기존 방법 대비 FCIS의 중요한 실용적 장점이다.

## 4. 실험 및 결과

### 4.1 PASCAL VOC ablation study

PASCAL VOC 2012 train으로 학습하고 validation set에서 평가한다. 추가 instance mask annotation을 사용하며, 평가지표는 mask-level IoU threshold 0.5와 0.7에서의 **mAPr** 이다.

비교 대상은 다음과 같다.

* **naïve MNC**: 거의 fully convolutional하게 바꾼 MNC 유사 baseline
* **InstFCN + R-FCN**: class-agnostic mask proposal과 분리된 classification 결합
* **FCIS (translation invariant)**: $k=1$ 로 두어 위치 민감성을 제거한 버전
* **FCIS (separate score maps)**: segmentation과 classification을 분리된 score map으로 학습한 버전
* **FCIS**: 제안 방식

결과는 다음과 같다.

* naïve MNC: mAPr@0.5 = 59.1, mAPr@0.7 = 36.0
* InstFCN + R-FCN: 62.7, 41.5
* FCIS (translation invariant): 52.5, 38.5
* FCIS (separate score maps): 63.9, 49.7
* FCIS: **65.7, 52.1**

이 결과는 몇 가지 중요한 결론을 준다.

첫째, **translation-variant property가 필수적**이다.
$k=1$ 인 translation invariant 버전은 성능이 크게 떨어진다. 이는 instance segmentation이 단순 semantic segmentation과 다르며, 상대 위치 정보를 명시적으로 모델링해야 함을 보여준다.

둘째, **joint formulation이 실제로 효과적**이다.
separate score maps 버전보다 FCIS가 더 좋다. 단순히 feature를 공유하는 것만으로는 부족하고, detection과 segmentation을 inside/outside score map으로 함께 학습시키는 구조 자체가 성능 향상에 기여한다.

셋째, **fully convolutional per-ROI design이 단순하면서도 강하다**.
InstFCN + R-FCN처럼 각 기능을 따로 결합한 방식보다, end-to-end joint 구조인 FCIS가 더 높은 정확도와 더 빠른 속도를 보인다.

### 4.2 COCO 실험

COCO에서는 80k+40k trainval로 학습하고 test-dev에서 평가한다.
주요 지표는 **mAPr@[0.5:0.95]** 와 **mAPr@0.5** 이다.

#### MNC와 비교

ResNet-101 기준 결과는 다음과 같다.

* **MNC, random sampling**:
  mAPr@[0.5:0.95] = 24.6, mAPr@0.5 = 44.3, test time = 1.37s/img
* **FCIS, random sampling**:
  28.8, 48.7, 0.24s/img
* **FCIS, OHEM**:
  29.2, 49.5, 0.24s/img

이 결과는 매우 강력하다. FCIS는 MNC보다 절대값으로 4.2%p 높은 mAPr@[0.5:0.95]를 기록하고, 상대적으로 약 17% 향상되었다고 저자들은 설명한다. 특히 큰 객체에서 개선 폭이 더 크다는 점이 중요하다. 이는 ROI warping이나 고정 크기 mask representation 없이, 공간 구조를 더 자연스럽게 유지한 설계가 큰 물체의 세부 mask를 더 잘 포착한다는 해석과 맞아떨어진다.

속도도 인상적이다. 추론 시간은 0.24초/이미지로, MNC의 1.37초 대비 약 6배 빠르다. 훈련도 약 4배 빠르다고 보고한다. 이는 per-ROI heavy sub-network를 제거한 효과가 직접적으로 드러난 결과다.

또한 FCIS는 OHEM을 거의 추가 비용 없이 활용할 수 있어 성능을 더 높일 수 있다. 반면 MNC는 per-ROI 계산이 비싸서 OHEM이 사실상 부담이 크다. 즉, FCIS는 단순히 “빠르다”가 아니라, **더 좋은 학습 전략을 적용하기 쉬운 구조**이기도 하다.

#### Backbone depth 비교

* ResNet-50: 27.1 / 46.7 / 0.16s
* ResNet-101: 29.2 / 49.5 / 0.24s
* ResNet-152: 29.5 / 49.8 / 0.27s

깊이가 50에서 101로 증가하면 성능 향상이 분명하다. 그러나 152에서는 개선이 거의 포화된다. 즉, backbone을 더 깊게 만드는 것보다 구조적 설계 자체가 더 큰 기여를 한다고 볼 수 있다.

### 4.3 COCO 2016 Segmentation Challenge 결과

논문은 FCIS 기반 시스템으로 COCO 2016 segmentation challenge에서 1위를 차지했다고 보고한다. baseline FCIS는 29.2%이고, 여기에 다음과 같은 기법을 순차적으로 추가한다.

* multi-scale testing: 32.0
* horizontal flip: 32.7
* multi-scale training: 33.6
* ensemble: 37.6

최종 ensemble 결과 37.6은 G-RMI의 33.8보다 3.8%p 높고, 상대적으로 약 11% 향상이라고 설명한다. 이 부분은 기본 모델의 구조적 강점 위에 테스트/학습 트릭을 더해 competition-level 성능을 달성했음을 보여준다.

### 4.4 Detection 성능

흥미롭게도 FCIS는 instance mask의 enclosing box를 detection box로 사용했을 때, COCO test-dev에서 **mAPb@[0.5:0.95] = 39.7** 의 detection 성능도 달성했다고 한다. 이는 이 모델이 segmentation 전용이 아니라 detection 관점에서도 경쟁력이 있다는 뜻이다. 즉, joint formulation이 detection 성능을 희생하고 segmentation만 챙긴 것이 아니라, 오히려 두 과업을 함께 잘 풀었다는 보조 증거로 볼 수 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 정의에 맞는 구조적 설계**를 제안했다는 점이다. 단순히 backbone을 바꾸거나 training trick을 추가한 것이 아니라, instance segmentation이 왜 기존 FCN으로 풀기 어려운지 정확히 짚고, 그것을 해결하기 위한 translation-variant 표현을 제시했다. 특히 **position-sensitive score maps** 를 detection과 segmentation의 공통 표현으로 재해석한 점은 매우 창의적이다.

둘째 강점은 **효율성과 정확도를 동시에 개선**했다는 점이다. 많은 논문이 정확도 향상을 위해 계산량을 늘리는데, 이 논문은 오히려 per-ROI 계산을 거의 없애면서도 정확도를 높였다. 이는 실제 시스템 적용 측면에서 매우 중요하다.

셋째 강점은 **joint formulation이 추가 파라미터 없이 작동**한다는 점이다. 복잡한 multi-branch head를 추가하지 않고도 detection과 segmentation supervision을 함께 받게 설계했다. 이 점은 모델의 간결성과 일반화 측면에서 장점으로 볼 수 있다.

넷째 강점은 실험 설계가 설득력 있다는 점이다. 단순한 SOTA 비교뿐 아니라, translation invariant 버전, separate score maps 버전, InstFCN + R-FCN 조합 등 의미 있는 ablation을 제시해 어떤 요소가 실제로 중요한지 보여준다.

반면 한계도 분명하다.

먼저, 이 모델은 **ROI proposal 기반**이다. 즉, 완전히 proposal-free 방식은 아니며 RPN 품질에 영향을 받는다. 이후 등장한 one-stage instance segmentation이나 transformer 기반 접근과 비교하면, 구조적으로 여전히 proposal pipeline의 틀 안에 있다.

둘째, 최종 성능 향상을 위해서는 **mask voting, multi-scale testing, horizontal flip, ensemble** 같은 후처리 및 테스트 기법에 의존하는 부분이 있다. baseline 자체도 강하지만 competition result는 순수 모델 구조의 힘만으로 해석해서는 안 된다.

셋째, 논문은 inside/outside joint formulation의 직관은 명확히 설명하지만, 왜 이 설계가 항상 최적의 결합인지에 대한 이론적 분석은 깊지 않다. 다시 말해, detection과 segmentation의 관계를 매우 영리하게 이용했지만, 그 결합 방식의 일반성이나 최적성은 논문에서 엄밀히 다루지 않는다.

넷째, score map 수가 $2k^2(C+1)$ 로 늘어나므로 클래스 수가 커질수록 head의 채널 수가 커진다. 논문에서는 이것이 큰 문제라고 하지는 않지만, 구조적으로 category-specific position-sensitive map 방식은 클래스 수 확장성 측면에서 비용이 증가할 수 있다.

다섯째, 논문은 성능과 속도 측면에서 매우 강하지만, 작은 객체의 성능 개선은 큰 객체만큼 인상적이지 않다. COCO 결과에서도 large object에서 개선 폭이 더 크다고 해석할 수 있다. 이는 해상도와 stride 16 표현 한계, proposal 품질 등의 영향을 시사한다.

비판적으로 보면, 이 논문은 당시 ROI 기반 instance segmentation의 병목을 매우 잘 해결했지만, 문제를 “ROI마다 position-sensitive map을 assemble하는 방식”으로 푸는 틀 자체는 여전히 남아 있다. 따라서 후속 세대의 dense prediction, dynamic convolution, transformer-based set prediction과 비교하면 더 단순하고 빠르지만 표현력의 유연성은 제한될 수 있다. 다만 이는 후대 관점의 해석이며, 논문 자체의 시대적 기여를 깎는 것은 아니다.

## 6. 결론

이 논문은 **instance-aware semantic segmentation을 위한 첫 fully convolutional end-to-end 방법**을 제시했다는 점에서 중요한 이정표다. 핵심 기여는 세 가지로 정리할 수 있다.

첫째, instance segmentation에 필요한 **translation-variant 표현**을 position-sensitive score maps로 구현했다.
둘째, detection과 segmentation을 분리하지 않고 **inside/outside joint formulation** 으로 함께 학습하게 만들었다.
셋째, ROI warping과 heavy fc sub-network 없이도 높은 정확도와 매우 빠른 추론 속도를 달성했다.

실험적으로도 PASCAL VOC와 COCO에서 강력한 성능을 보였고, COCO 2016 segmentation challenge 우승으로 실전 경쟁력까지 입증했다. 특히 이 논문은 “정확한 모델”일 뿐 아니라 “잘 설계된 시스템”이라는 점이 중요하다. 계산 공유, 구조 단순성, OHEM 활용 가능성, detection과 segmentation의 자연스러운 통합이라는 요소가 함께 작동한다.

향후 연구 관점에서도 이 논문은 의미가 크다. instance segmentation을 semantic segmentation과 object detection의 단순한 결합으로 보지 않고, **공통 표현 위에서 함께 푸는 문제**로 다시 정의했기 때문이다. 이후 등장한 많은 instance segmentation 연구들이 다른 방식으로 발전했더라도, “mask와 class를 joint하게 예측하는 fully convolutional 구조”라는 방향성은 이 논문이 강하게 밀어준 흐름이라고 볼 수 있다.
