# Boundary-aware Instance Segmentation

- **저자**: Zeeshan Hayder, Xuming He, Mathieu Salzmann
- **발표연도**: 2017
- **arXiv**: https://arxiv.org/abs/1612.03129

## 1. 논문 개요

이 논문은 instance-level semantic segmentation 문제를 다룬다. 이 문제는 이미지 안의 각 개별 객체에 대해 위치를 찾고, 객체의 픽셀 영역을 분할하고, 동시에 클래스까지 예측하는 작업이다. 저자들은 기존 방법들이 대체로 bounding box proposal을 먼저 만든 뒤, 각 box 내부에서만 binary mask를 예측한다는 점에 주목한다. 이 방식은 box가 너무 작거나, 위치가 약간 어긋나거나, 객체를 완전히 포함하지 못할 때 성능이 크게 떨어진다.

논문의 핵심 목표는 이러한 bounding box 오차에 더 강인한 instance segmentation 방법을 만드는 것이다. 이를 위해 저자들은 객체 마스크를 직접 binary mask로 예측하지 않고, 객체 경계까지의 거리 정보를 담은 distance transform 기반 표현으로 바꿔서 예측한다. 이렇게 하면 box 안에서 얻은 정보만으로도 box 바깥까지 확장되는 객체 마스크를 복원할 수 있다.

이 문제는 실제 응용에서 매우 중요하다. 예를 들어 autonomous driving이나 robotics에서는 객체의 정확한 외곽선과 개별 인스턴스 구분이 필요하다. 그런데 detection 단계에서 생성된 box가 완벽하지 않은 경우가 많기 때문에, box 내부에만 갇혀 있는 segmentation 방식은 구조적으로 한계를 가진다. 이 논문은 바로 그 구조적 한계를 완화하려는 시도라고 볼 수 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 객체를 “이 픽셀이 객체 내부인가 아닌가”라는 단순 binary mask로 표현하지 않고, “이 픽셀이 객체 경계까지 얼마나 떨어져 있는가”로 표현하는 것이다. 구체적으로는 객체 내부 픽셀마다 nearest boundary까지의 truncated distance를 저장하고, 객체 바깥 픽셀은 background로 처리한다. 이 표현은 객체의 내부 픽셀 하나하나가 경계 정보를 간접적으로 담고 있기 때문에, 일부만 보이는 상황에서도 전체 형태를 더 잘 복원할 수 있다.

기존 방법과의 가장 중요한 차이는 예측 결과가 bounding box 내부에만 제한되지 않는다는 점이다. 기존 방식은 box 안에서 binary mask를 자르는 구조라서 proposal 오류를 복구할 수 없다. 반면 이 논문은 distance map을 inverse distance transform 방식으로 binary mask로 디코딩하므로, 최종 마스크가 box 바깥으로 자연스럽게 확장될 수 있다. 저자들은 이것을 boundary-aware representation이라고 부른다.

또 하나의 중요한 설계는 distance 값을 그대로 회귀하는 대신, 이를 여러 개의 bin으로 양자화하여 one-hot binary vector로 바꾸는 것이다. 이렇게 하면 문제를 복잡한 연속값 회귀가 아니라 여러 개의 pixel-wise binary classification 문제로 바꿀 수 있어서, CNN이 학습하기 쉬워진다.

## 3. 상세 방법 설명

전체 방법은 두 단계로 이해할 수 있다. 첫째, 각 proposal box에 대해 boundary-aware mask representation을 예측한다. 둘째, 이 표현을 deconvolution 기반 네트워크로 binary mask로 복원한다. 이후 이 모듈을 MNC(Multitask Network Cascade)에 넣어 최종 instance segmentation 시스템을 구성한다.

### Boundary-aware mask representation

저자들은 정규화된 window 안에서 객체의 경계와 객체 바깥 픽셀 집합을 $Q$라고 두고, 각 픽셀 $p$에 대해 truncated distance를 다음과 같이 정의한다.

$$
D(p) = \min \left( \left\lceil \min_{q \in Q} d(p,q) \right\rceil, R \right)
$$

여기서 $d(p,q)$는 Euclidean distance이고, $R$은 최대 거리값을 잘라내는 truncation threshold이다. 즉, 객체 내부 픽셀은 가장 가까운 경계까지의 거리값을 갖고, 너무 큰 값은 $R$로 잘린다. 이 과정은 표현의 범위를 제한해 학습을 안정화하려는 목적을 가진다.

이 표현의 장점은 분명하다. binary mask는 내부 픽셀에 대해 “객체다”라는 정보만 담지만, distance transform은 내부 픽셀도 boundary location에 대한 정보를 담는다. 따라서 box가 객체를 일부만 포함해도, 내부에서 얻은 거리 정보만으로 더 큰 형태를 추론할 수 있다. 또한 픽셀마다 중복된 구조 정보가 들어 있어 일부 예측 노이즈에도 비교적 강하다.

### Distance quantization과 binary maps

거리값 $D(p)$를 그대로 예측하지 않고, 저자들은 이를 $K$개의 uniform bin으로 나누어 one-hot vector $b(p)$로 표현한다.

$$
D(p) = \sum_{n=1}^{K} r_n \cdot b_n(p), \qquad \sum_{n=1}^{K} b_n(p) = 1
$$

여기서 $r_n$은 $n$번째 bin에 대응하는 distance 값이고, $b_n(p)$는 해당 bin이 활성화되면 1, 아니면 0이다. 즉, 하나의 multi-valued distance map을 $K$개의 binary map으로 바꾸는 셈이다. 논문에서는 실제로 $K=5$를 사용한다.

이 설계는 매우 실용적이다. 네트워크 입장에서는 각 픽셀에 대해 “어느 distance bin인가”를 판단하면 되므로, segmentation에서 익숙한 pixel-wise classification 구조를 활용할 수 있다.

### Inverse distance transform에 의한 mask 복원

예측된 dense map에서 최종 객체 마스크는 각 픽셀마다 반지름 $D(p)$의 원판(disk)을 놓고, 이들의 합집합을 취하는 방식으로 근사적으로 복원한다. 이를 식으로 쓰면 다음과 같다.

$$
M = \bigcup_{p} T(p, D(p))
$$

여기서 $T(p,r)$는 중심이 $p$이고 반지름이 $r$인 disk이다. one-hot 표현을 이용하면 식은 다음처럼 바뀐다.

$$
M
= \bigcup_{p} T\left(p, \sum_{n=1}^{K} r_n b_n(p)\right)
= \bigcup_{n=1}^{K} \bigcup_{p} T(p, r_n b_n(p))
= \bigcup_{n=1}^{K} T(\cdot, r_n) * B_n
$$

여기서 $B_n$은 $n$번째 binary map이고, $*$는 convolution이다. 중요한 점은 inverse distance transform이 convolution들의 조합으로 표현될 수 있다는 것이다. 이 덕분에 디코딩 과정을 neural network 안에 differentiable하게 넣을 수 있다.

### Object Mask Network (OMN)

OMN은 proposal box의 RoI-warped feature를 입력으로 받아 작동한다.

첫 번째 모듈은 $K$개의 binary map을 예측한다. 각 map은 fully connected layer와 sigmoid를 통해 생성되는 pixel-wise probability map이다. 이는 각 픽셀이 해당 distance bin에 속할 확률로 해석할 수 있다.

두 번째 모듈은 이 $K$개 map을 받아 최종 binary mask로 변환하는 residual deconvolution network이다. 저자들은 Eq. 3의 morphological decoding이 고정 가중치 deconvolution 여러 개로 구현 가능하다는 점을 이용한다. 각 distance bin마다 서로 다른 kernel size와 padding을 갖는 deconvolution을 적용하고, 이후 weighted summation과 sigmoid로 union 연산을 근사한다. summation 가중치는 학습된다. 작은 반지름에 해당하는 출력을 더 큰 반지름 출력과 정렬하기 위해 upsampling도 사용한다.

이 네트워크의 핵심은 “distance map 예측 + deconvolution decoding” 전체가 end-to-end differentiable하다는 점이다. 따라서 ground-truth mask와 직접 비교하면서 학습할 수 있다. 논문에서는 decoding된 high-resolution output에 대해 binary cross-entropy loss를 사용한다.

### BAIS 네트워크

최종 시스템은 BAIS(Boundary-Aware Instance Segmentation) 네트워크로, MNC 프레임워크에 OMN을 넣은 형태이다. 전체 구조는 세 개의 하위 네트워크로 이루어진다.

첫 번째는 backbone CNN과 RPN이다. 논문에서는 VGG16을 backbone으로 사용하며, RPN이 bounding box proposal을 생성한다.

두 번째는 proposal별 OMN이다. 각 proposal에 대해 boundary-aware mask를 예측하고, 이 mask는 box 바깥으로 확장될 수 있다.

세 번째는 classification과 bounding box regression 모듈이다. 원래 MNC와 비슷하게 predicted mask를 feature masking layer에 사용해 mask feature를 만들고, 이를 box feature와 결합한 뒤 fully connected layer로 분류와 box regression을 수행한다.

저자들은 여기서 끝나지 않고 5-stage cascade도 제안한다. 처음 3-stage에서 나온 box regression 결과로 proposal box를 refinement한 뒤, 다시 두 번째 OMN과 classification stage를 적용한다. 두 OMN과 두 classification module은 weight sharing을 한다. 이 설계는 첫 단계의 segmentation 결과를 이용해 box 자체를 더 정확히 만들고, 다시 segmentation을 개선하려는 목적이다.

### 학습과 추론

학습은 multi-task, multi-stage loss로 진행된다. RPN과 classification에는 softmax loss를 사용하고, OMN에는 binary cross-entropy loss를 사용한다. bounding box regression에는 smooth $L_1$ loss를 쓴다. 5-stage 모델에서는 box loss와 mask loss를 3단계와 5단계 뒤에서 계산한다.

최적화는 SGD를 사용한다. minibatch는 이미지 8장이고, 입력 이미지는 짧은 변이 600픽셀이 되도록 resize한다. VGG16은 ImageNet pretrained weights로 초기화하고, 나머지는 표준편차 0.01의 Gaussian으로 초기화한다. 학습 스케줄은 learning rate 0.001로 20k iteration, 이후 0.0001로 5k iteration이다.

RPN은 처음 약 12k개의 bounding box를 만들고, NMS threshold 0.7로 pruning하여 상위 300개 proposal만 유지한다. OMN에서는 $K=5$를 사용하고, 디코딩 후 threshold 0.4로 binary mask를 얻는다. 테스트 시에도 300개 proposal을 사용하며, class-specific NMS는 IoU threshold 0.5로 적용한다. 마지막으로 MNC의 in-mask voting scheme으로 instance segmentation을 추가 정제한다.

## 4. 실험 및 결과

논문은 두 가지 측면을 평가한다. 첫째는 instance-level semantic segmentation 자체의 성능이고, 둘째는 OMN을 class-agnostic segment proposal generator로 썼을 때의 성능이다.

### 데이터셋과 평가 설정

Pascal VOC 2012는 20개 object class를 포함하며, 저자들은 5623개 training image와 5732개 validation image의 instance annotation을 사용했다. 학습에는 training set 전체를 사용했고, 평가는 validation set에서 수행했다. instance segmentation 평가는 IoU 0.5와 0.7에서의 mAP를 사용하고, segment proposal 평가는 AR@10, AR@100, AR@1000 등을 사용한다.

Cityscapes는 9개 object category에 대한 instance-level labeling을 제공한다. 이 데이터셋은 이미지마다 객체 수가 많고 작은 객체가 많아서 더 어렵다. 저자들은 2975개 training image로 학습하고, test server에서 최종 성능을 평가했다. 평가 지표는 AP, AP(50%), AP(100m), AP(50m) 등이다. 뒤의 두 지표는 각각 100m, 50m 이내 객체만 평가한 결과다.

### Pascal VOC 2012 instance segmentation 결과

논문은 SDS, Hypercolumn, InstanceFCN, MNC 등과 비교한다. 결과는 다음과 같다.

- SDS: mAP@0.5 49.7, mAP@0.7 25.3
- Hypercolumn: mAP@0.5 60.0, mAP@0.7 40.4
- InstanceFCN: mAP@0.5 61.5, mAP@0.7 43.0
- MNC: mAP@0.5 63.5, mAP@0.7 41.5
- MNC-new: mAP@0.5 65.01, mAP@0.7 46.23
- BAIS-insideBBox: mAP@0.5 64.97, mAP@0.7 44.58
- BAIS-full: mAP@0.5 65.69, mAP@0.7 48.30

가장 중요한 포인트는 IoU 0.7에서의 개선 폭이다. BAIS-full은 MNC-new 대비 mAP@0.7에서 48.30으로 더 높다. 이는 단지 객체를 “대충 맞히는” 수준이 아니라, 경계를 더 정확하게 맞춘다는 뜻이다. 특히 BAIS-insideBBox와 BAIS-full의 차이는 논문의 핵심 주장인 “mask가 box 바깥으로 갈 수 있어야 한다”는 점을 직접 뒷받침한다. 만약 마스크를 box 내부에만 제한하면 성능이 떨어진다.

또한 stage 수의 영향도 분석했다. 3-stage로 학습한 BAIS도 MNC baselines보다 좋았고, 5-stage로 학습한 BAIS가 가장 좋은 결과를 냈다. 즉, boundary-aware representation 자체가 유효하고, cascade refinement가 추가 이득을 준다고 해석할 수 있다.

### Cityscapes instance segmentation 결과

Cityscapes test set에서 BAIS-full은 다음 성능을 보인다.

- AP: 17.4
- AP(50%): 36.7
- AP(100m): 29.3
- AP(50m): 34.0

이는 표에 제시된 기존 방법들보다 전반적으로 높다. 예를 들어 DWT는 AP 15.6, AP(50%) 30.0, AP(100m) 26.2, AP(50m) 31.8인데, BAIS는 모든 지표에서 이를 넘어선다. 특히 AP(50%)에서 36.7로 차이가 크다.

클래스별 비교에서도 대부분 클래스에서 DWT보다 우수하다. AP(50m) 기준으로 person, rider, car, train, motorcycle, bicycle 등에서 높고, truck만 예외적으로 낮다. AP(100m) 기준도 유사하다. 저자들은 validation set에서도 MNC-new와 비교했는데, IoU 0.5와 0.7 모두 BAIS가 약간 더 좋았다. 이 결과 역시 box proposal 바깥으로 마스크를 확장할 수 있는 구조의 이점을 보여준다고 해석한다.

### 정성적 결과와 실패 사례

Cityscapes의 qualitative result에서는 복잡한 도시 장면에서 많은 인스턴스가 동시에 존재해도 비교적 정확한 segmentation을 보인다고 설명한다. 반면 failure case는 주로 하나의 객체 인스턴스가 여러 조각으로 분리되어 예측되는 경우라고 밝힌다. 즉, 이 방법은 경계 복원에는 강하지만, instance coherence를 완전히 보장하지는 못하는 것으로 보인다.

### Segment proposal generation 결과

OMN 자체를 class-agnostic segment proposal generator로 평가한 결과도 제시된다. Pascal VOC 2012 validation set에서 다음과 같은 AR을 얻는다.

- AR@10: 47.8
- AR@100: 51.8
- AR@1000: 54.7

이는 비교 대상들 중 AR@10과 AR@100에서 최고 수준이다. SharpMask는 AR@1000에서 56.5로 약간 더 높지만, 저자들은 실제 후속 처리에서는 1000개 proposal을 쓰기 어려운 경우가 많고, 보통 10개에서 300개 사이를 다룬다고 지적한다. 그런 실용적인 구간에서 OMN이 매우 강하다는 것이 논문의 주장이다. 또한 높은 IoU threshold 구간에서 recall이 강하다는 점도 보고한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정이 매우 명확하다는 점이다. 기존 instance segmentation 방법의 약점을 “bounding box에 mask가 갇혀 있다”는 구조적 한계로 짚고, 이를 해결하기 위한 표현과 네트워크 설계를 일관되게 제시한다. 단순히 backbone을 바꾸거나 후처리를 추가한 것이 아니라, mask 자체의 표현 방식을 바꿨다는 점에서 아이디어의 독창성이 있다.

또 다른 강점은 수식과 네트워크가 잘 연결되어 있다는 점이다. distance transform 기반 표현, one-hot quantization, inverse distance transform의 convolution 해석, deconvolution 기반 decoder가 서로 자연스럽게 이어진다. 즉, 수학적 직관을 실제 딥러닝 구조로 잘 변환한 논문이다.

실험적으로도 주장이 비교적 잘 뒷받침된다. 특히 BAIS-insideBBox와 BAIS-full 비교는 이 논문의 핵심 공헌이 단순한 네트워크 복잡도 증가가 아니라 “box 바깥 확장 가능성”에 있다는 점을 보여주는 중요한 ablation이다. Pascal VOC와 Cityscapes 둘 다에서 개선을 보인 점도 설득력을 높인다.

한계도 분명하다. 첫째, backbone이 VGG16에 기반해 있어 오늘 기준으로는 비교적 약한 특징 추출기를 사용한다. 저자들도 conclusion에서 향후 residual network 같은 deeper architecture로 바꾸겠다고 언급한다. 이는 당시에는 합리적이지만, 성능 향상 중 일부가 더 강한 backbone과 결합되면 어떻게 달라질지는 본문에서 직접 검증되지 않았다.

둘째, distance transform을 $K=5$개의 bin으로 양자화하는 것이 어느 정도 최적인지에 대한 자세한 분석은 제공되지 않는다. 즉, bin 수, truncation threshold $R$, decoder 구조의 민감도에 대한 깊은 ablation은 본문에 없다. 이런 부분은 실제 방법의 robustness를 더 이해하는 데 중요하지만, 논문 본문만으로는 충분히 설명되지 않는다.

셋째, 실패 사례에서 보이듯 하나의 인스턴스가 여러 조각으로 쪼개지는 문제가 남아 있다. 이는 경계 복원 능력과 별개로 instance grouping 또는 global consistency 문제가 완전히 해결되지 않았음을 뜻한다. 논문은 이 문제를 보여주지만, 이에 대한 추가 메커니즘은 제시하지 않는다.

넷째, proposal-based pipeline 자체의 한계는 완전히 사라진 것이 아니다. 논문은 부정확한 box에 더 강인해졌다고 주장하지만, 여전히 전체 시스템은 RPN proposal과 NMS, cascade refinement에 의존한다. 즉, proposal-free 접근으로 문제를 재정의한 것은 아니며, proposal quality의 영향을 완전히 제거하지는 못한다.

비판적으로 보면, 이 논문의 가장 강한 기여는 “경계를 고려한 mask representation”이지, 완전히 새로운 end-to-end instance segmentation 패러다임이라고 보기는 어렵다. 시스템 전체는 MNC의 확장형에 가깝다. 그럼에도 representation-level innovation이 실제 성능 향상으로 이어졌다는 점에서 충분히 가치 있는 기여라고 평가할 수 있다.

## 6. 결론

이 논문은 instance segmentation에서 bounding box 오차에 취약하다는 기존 방식의 구조적 문제를 해결하기 위해, distance transform 기반의 boundary-aware mask representation을 제안했다. 또한 이를 예측하고 binary mask로 복원하는 Object Mask Network(OMN)를 설계하고, 이를 MNC에 통합한 BAIS 네트워크를 end-to-end로 학습했다.

핵심 기여는 세 가지로 요약할 수 있다. 첫째, 객체 마스크를 경계까지의 거리 정보로 표현하는 새로운 방식. 둘째, 이를 neural network 안에서 differentiable하게 decoding하는 residual-deconvolution 구조. 셋째, 이 표현을 실제 instance segmentation 시스템에 통합해 Pascal VOC 2012와 Cityscapes에서 state-of-the-art를 달성한 점이다.

실제 적용 측면에서는, detection proposal이 완벽하지 않은 환경에서도 더 정확한 instance mask를 얻을 수 있다는 점에서 의미가 크다. 향후 연구 측면에서도, 이 논문은 “mask를 어떻게 표현할 것인가”가 segmentation 성능에 직접적인 영향을 준다는 점을 보여준다. 따라서 이후의 instance segmentation 연구에서도 단순 binary mask prediction을 넘어, shape-aware 또는 boundary-aware representation을 탐구하는 방향에 중요한 연결고리를 제공한다고 볼 수 있다.
