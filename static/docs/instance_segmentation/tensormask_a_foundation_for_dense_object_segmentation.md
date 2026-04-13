# TensorMask: A Foundation for Dense Object Segmentation

- **저자**: Xinlei Chen, Ross Girshick, Kaiming He, Piotr Dollar
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1903.12174

## 1. 논문 개요

이 논문은 instance segmentation을 위해 오랫동안 주류였던 `detect-then-segment` 방식, 즉 먼저 bounding box를 검출한 뒤 그 영역을 잘라 mask를 예측하는 방식과는 다른 방향을 제안한다. 저자들은 “bounding box 검출에서는 SSD, RetinaNet처럼 dense sliding-window 방식이 크게 성공했는데, 왜 instance segmentation에서는 이에 대응하는 직접적인 dense 방식이 거의 없는가?”라는 문제를 정면으로 다룬다. 이 질문에 대한 답으로 제시된 것이 `TensorMask`이다.

논문의 핵심 문제의식은 instance segmentation의 출력이 단순한 스칼라나 짧은 벡터가 아니라, 각 위치마다 다시 하나의 2D mask라는 점이다. semantic segmentation은 각 픽셀에 class label을 붙이면 되지만, dense instance segmentation은 각 sliding-window 위치마다 “하나의 mask 구조 전체”를 예측해야 한다. 저자들은 기존 방식이 이 구조적 성질을 충분히 반영하지 못했다고 보고, mask를 단순한 채널 묶음으로 다루지 않고 기하학적 의미를 가진 4차원 tensor로 명시적으로 표현해야 한다고 주장한다.

이 문제는 중요하다. 왜냐하면 dense prediction 기반 object detection은 이미 강력한 패러다임으로 자리 잡았고, 만약 instance segmentation에서도 dense sliding-window 방식이 실용적인 수준까지 도달할 수 있다면, Mask R-CNN 계열과는 다른 연구 축을 형성할 수 있기 때문이다. 논문은 실제로 TensorMask가 Mask R-CNN에 근접한 성능을 낼 수 있음을 보이며, dense mask prediction을 위한 기반(framework)을 제시하는 것이 목표라고 밝힌다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 dense instance segmentation을 `4D tensor prediction` 문제로 재정의하는 것이다. 구체적으로 저자들은 mask prediction을 $(V, U, H, W)$ 형태의 structured tensor로 본다. 여기서 $(H, W)$는 이미지 상에서 sliding-window의 위치를 나타내고, $(V, U)$는 각 위치에서 예측되는 mask 내부의 상대 좌표를 나타낸다. 즉, 출력의 두 축은 “어디에서 예측하는가”이고, 다른 두 축은 “그 위치에서 예측되는 mask의 내부 구조가 무엇인가”이다.

이 관점은 기존의 DeepMask류 접근과 다르다. 기존 방법은 보통 $V \times U$ mask를 하나의 channel 축에 펴서 $(C, H, W)$ 형태로 취급했다. 그러나 이렇게 하면 mask 내부의 2D 구조가 단순한 channel 집합으로 사라진다. 저자들은 이것이 이미지 자체를 2D grid가 아니라 1D 벡터로 취급하는 것과 비슷한 손실이라고 본다. TensorMask는 $(V, U)$ 자체를 공간적 의미를 갖는 sub-tensor로 남겨 둠으로써, upsampling, coordinate transform, pyramid 설계 같은 연산을 기하학적으로 일관된 방식으로 정의할 수 있게 한다.

또 하나의 중요한 차별점은 `tensor bipyramid`이다. 일반적인 feature pyramid는 더 낮은 해상도 feature map이 더 큰 객체를 담당한다. 그런데 box는 객체 크기와 상관없이 4개 숫자로 표현되지만, mask는 큰 객체일수록 더 많은 픽셀이 필요하다. TensorMask는 이를 반영해 pyramid level이 깊어질수록 $(H, W)$ 해상도는 낮아지지만, 반대로 $(V, U)$ mask 해상도는 증가하도록 설계한다. 즉 큰 객체는 더 적은 위치에서 예측하되, 각 위치에서 더 고해상도 mask를 내도록 만든다. 이것이 dense mask prediction에서 구조를 살리는 핵심 설계이다.

## 3. 상세 방법 설명

TensorMask의 출발점은 mask를 표현하는 tensor의 의미를 엄밀히 정의하는 것이다. 먼저 길이 단위(unit of length)를 도입한다. $(H, W)$ 축의 단위 $\sigma_{HW}$는 입력 이미지 기준 stride와 연결되고, $(V, U)$ 축의 단위 $\sigma_{VU}$는 mask 내부 축에서 한 칸 이동이 이미지에서 몇 픽셀에 해당하는지를 뜻한다. 이 둘의 비율을 $\alpha = \sigma_{VU} / \sigma_{HW}$로 두면, 하나의 tensor 값이 이미지 상에서 어느 위치의 mask 값을 의미하는지 정확히 쓸 수 있다.

가장 기본이 되는 것은 `natural representation`이다. 이 표현에서 4D tensor $F$의 원소 $F(v, u, y, x)$는 sliding-window 중심이 $(y, x)$일 때, 그 window 내부 상대 위치 $(v, u)$에 해당하는 mask 값을 나타낸다. 논문은 이를 다음과 같이 정의한다.

$$
F(v,u,y,x) \text{ 는 중심 } (y,x) \text{ 를 갖는 window 안의 } (y+\alpha v,\; x+\alpha u) \text{ 위치의 mask 값이다.}
$$

이 표현은 직관적이지만, convolution으로 계산되는 feature와 출력 픽셀 사이의 alignment를 직접 보장하지는 않는다. 그래서 저자들은 `aligned representation`을 추가로 정의한다. aligned representation에서는 $\hat{F}(\hat{v}, \hat{u}, \hat{y}, \hat{x})$가 “현재 픽셀 $(\hat{y}, \hat{x})$를 포함하는 여러 overlapping window들에서의 mask 값”을 나열하는 구조가 된다. 이 표현의 장점은 어떤 픽셀에 대한 정보가 그 픽셀 좌표에 정렬되어 있어, RoIAlign과 유사한 의미의 pixel-to-pixel alignment를 유지하기 쉽다는 점이다.

두 표현은 좌표 변환으로 연결된다. 논문은 aligned representation에서 natural representation으로 가는 `align2nat` 변환을 다음과 같이 정의한다.

$$
F(v,u,y,x) = \hat{F}(v,u,y+\alpha v,x+\alpha u)
$$

즉, aligned 표현에서 “현재 픽셀 기준”으로 저장된 값을 다시 “window 중심 기준” natural 표현으로 옮겨온다. 반대로 natural에서 aligned로 가는 `nat2align`도 정의되지만, 실제 모델에서는 주로 `align2nat`를 사용한다. 중요한 점은 이 변환이 단순한 reshape가 아니라, 좌표 의미를 보존하는 기하학적 연산이라는 것이다.

그 다음 핵심 연산이 `up_align2nat`이다. 이는 coarse한 aligned mask를 bilinear upsampling으로 세밀하게 만든 뒤, `align2nat`로 natural representation으로 바꾸는 연산이다. 이 방식의 장점은 고해상도 mask를 얻기 위해 앞단 feature map의 channel 수를 폭발적으로 늘릴 필요가 없다는 것이다. 논문 실험에서도 단순 natural upscaling보다 aligned + bilinear upscaling이 훨씬 안정적이고 정확하다.

이 아이디어를 multi-scale로 확장한 것이 `tensor bipyramid`이다. 각 level $k$에서 tensor shape를

$$
(2^k V,\; 2^k U,\; \frac{1}{2^k}H,\; \frac{1}{2^k}W)
$$

로 둔다. 즉 level이 커질수록 sliding 위치 수는 줄어들지만, 각 위치에서 예측되는 mask의 해상도는 증가한다. 이것은 큰 물체는 더 큰 mask grid로 표현해야 한다는 직관을 구현한 것이다. 저자들은 이를 효율적으로 계산하기 위해 `swap_align2nat`라는 연산을 도입한다. 개념적으로는 `up_align2nat` 후 $(H, W)$ 축을 subsample하는 방식이며, 실제 구현에서는 중간 거대 tensor를 만들지 않고 필요한 값만 계산해 복잡도를 $O(V \cdot U \cdot H \cdot W)$로 유지한다고 설명한다.

아키텍처 측면에서는 FPN backbone 위에 mask head와 classification head를 둔다. classification head는 object class를 예측하고, mask head는 class-agnostic mask를 출력한다. 즉 mask는 클래스별로 따로 내지 않고, 해당 window의 category는 별도 classification branch가 담당한다. 이는 출력 크기를 줄이고 학습을 단순화하는 선택이다. baseline으로는 natural/aligned representation, simple/upscaling head의 조합 네 가지를 비교하고, 최종적으로는 tensor bipyramid 기반 head가 가장 좋은 성능을 낸다.

학습에서 label assignment는 box IoU가 아니라 `mask-driven` 규칙을 따른다. 특정 window가 양성(positive)이 되려면 세 조건을 만족해야 한다. 첫째, window가 ground-truth mask를 완전히 포함해야 한다. 둘째, mask의 bounding box 중심이 window 중심에서 $\sigma_{VU}$ 이내에 있어야 한다. 셋째, 같은 조건을 만족하는 다른 mask가 없어야 한다. 이는 dense box detector의 anchor 할당과 달리, mask의 기하 구조를 기준으로 설계된 규칙이다.

손실 함수는 mask branch에 대해 per-pixel binary cross-entropy를 사용한다. foreground/background 불균형을 줄이기 위해 foreground 픽셀 가중치를 1.5로 둔다. mask loss는 positive window에 대해서만 계산되고, 각 window 내부 픽셀 평균 후 다시 positive window들에 대해 평균한다. classification은 focal loss를 사용하며, 논문에는 $\gamma = 3$, $\alpha = 0.3$이라고 적혀 있다. box regression은 parameter-free $L_1$ loss를 사용한다. 전체 loss는 이들의 가중합이다. 다만 논문은 각 loss의 구체적 가중치를 본문에 명시적으로 적지 않았다.

추론 시에는 각 sliding-window에서 mask, class score, box를 얻고, regressed box에 대해 NMS를 적용한다. soft mask를 원본 해상도 binary mask로 바꾸는 과정은 Mask R-CNN과 같은 방법과 하이퍼파라미터를 사용했다고 쓴다. 부록에서는 box branch 없이도 mask 성능이 거의 유지된다고 보고하여, TensorMask에서 box는 본질적 요소라기보다 보조적 역할에 가깝다고 해석할 수 있다.

## 4. 실험 및 결과

실험은 COCO instance segmentation에서 수행되었다. 학습은 약 118k장의 `train2017`, 평가는 5k장의 `val2017`, 최종 비교는 `test-dev`에서 진행한다. 주요 평가지표는 COCO mask AP이며, box 검출은 `AP_bb`로 따로 보고한다. backbone은 주로 ResNet-50-FPN과 ResNet-101-FPN을 사용한다. 기본 학습 스케줄은 약 72 epoch에 해당하는 `6x` schedule이며, 짧은 변 길이 640~800 사이 scale jitter를 적용한다.

먼저 representation 자체에 대한 ablation이 이루어진다. simple head에서는 natural과 aligned의 차이가 크지 않았다. $15 \times 15$ mask에서 natural은 28.5 AP, aligned는 28.9 AP였다. 즉 upscaling이 없으면 aligned의 이점이 제한적이다. 하지만 upscaling head에서는 상황이 크게 달라진다. 출력 mask는 계속 $15 \times 15$인데, 내부적으로 더 작은 representation에서 upsample하는 구조를 썼을 때, natural 방식은 upscaling factor $\lambda$가 커질수록 급격히 무너진다. 예를 들어 $\lambda=5$에서 natural은 19.2 AP까지 떨어지지만 aligned는 28.4 AP를 유지한다. 이는 dense mask prediction에서 단순한 채널 업샘플링이 아니라, 정렬된 기하 구조를 고려한 upsampling이 중요하다는 매우 직접적인 증거다.

보간 방식 비교도 논문의 중요한 포인트다. aligned head에서 bilinear interpolation과 nearest-neighbor interpolation을 비교했는데, $\lambda=5$일 때 nearest는 25.3 AP, bilinear는 28.4 AP로 3.1 AP 차이가 난다. 논문은 nearest-neighbor 기반 방식이 InstanceFCN과 연결된다고 설명하며, 특히 객체가 서로 겹칠 때 severe artifact가 발생한다고 정성적 예시로 보인다. 즉 TensorMask의 성능 향상은 단순히 4D tensor를 썼다는 형식적 차원이 아니라, 그 구조를 이용해 올바른 interpolation과 alignment를 수행한 데서 나온다.

가장 큰 성능 향상은 tensor bipyramid에서 나타난다. feature pyramid 기반 최선 모델은 28.9 AP였지만, tensor bipyramid로 바꾸면 34.0 AP로 5.1 AP 상승한다. 세부적으로는 $AP_L$가 40.7에서 48.4로 7.7 포인트 상승한다. 이는 큰 객체에 대해 더 높은 해상도 mask를 예측하는 설계가 실제로 매우 효과적임을 보여준다. 저자들의 주장이었던 “큰 객체는 더 고해상도 mask가 필요하다”는 점이 정량적으로 검증된 셈이다.

또한 window size를 하나만 쓰는 대신 두 개 $(15 \times 15, 11 \times 11)$를 사용하면 34.0 AP에서 35.2 AP로 1.2 포인트 상승한다. 이는 object detection에서 anchor 다양성이 도움이 되듯, dense mask prediction에서도 window scale 다양성이 성능에 기여함을 보여준다. 다만 논문은 더 많은 window size와 aspect ratio를 시도하지 않았고, 이는 향후 개선 여지가 있다고만 언급한다.

Mask R-CNN과의 본격 비교에서는 `test-dev`에서 TensorMask가 상당히 근접한 결과를 보인다. ResNet-50-FPN 기준으로 Mask R-CNN은 36.8 AP, TensorMask는 35.4 AP이다. ResNet-101-FPN에서는 Mask R-CNN이 38.3 AP, TensorMask가 37.1 AP로 차이는 1.2 AP다. 논문은 이를 통해 dense sliding-window 방식이 `detect-then-segment`와의 격차를 상당히 줄였다고 주장한다. 절대 최고 성능을 넘지는 못했지만, 그동안 부재했던 dense instance segmentation 패러다임이 실제 경쟁력을 가진다는 것을 보여준 것이 핵심이다.

속도 측면에서는 약점도 분명하다. ResNet-101-FPN 기준 TensorMask는 V100에서 0.38초/이미지인데, Mask R-CNN은 0.09초/이미지다. 논문은 TensorMask가 10만 개 이상의 dense window에 대해 mask를 예측해야 하므로 계산량 오버헤드가 크다고 설명한다. 가속화는 가능할 수 있지만 이 논문 범위 밖이라고 명시한다.

부록의 detection 결과도 의미가 있다. ResNet-50-FPN 기준 TensorMask는 `AP_bb = 41.6`으로 Mask R-CNN의 41.7과 거의 동일하고 RetinaNet의 39.3보다 높다. 즉 TensorMask는 mask prediction뿐 아니라 box detection 측면에서도 경쟁력 있는 특징 표현을 학습한다. 또 box head를 제거한 실험에서도 mask AP가 거의 유지되어, TensorMask의 mask quality가 box branch에 강하게 의존하지 않는다는 점을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 dense instance segmentation이라는 문제를 단순히 “Mask R-CNN의 대안” 수준이 아니라, 새로운 수학적 표현 틀로 재구성했다는 점이다. $(V,U,H,W)$로 dense mask를 구조화하고, unit, natural/aligned representation, coordinate transform, tensor bipyramid 같은 개념을 체계적으로 정의했다. 이는 특정 모델 하나를 제안한 것 이상으로, 이후 연구가 사용할 수 있는 공통 언어를 만든 작업에 가깝다.

두 번째 강점은 ablation이 매우 설득력 있다는 점이다. aligned vs natural, bilinear vs nearest, feature pyramid vs tensor bipyramid를 단계적으로 비교하면서, 제안된 구조가 왜 필요한지 실험적으로 분해해서 보여준다. 특히 단순히 “4D tensor가 더 좋다”가 아니라, alignment와 interpolation이 성능 차이를 만든다는 점을 명확히 입증한다.

세 번째 강점은 결과의 의미다. TensorMask는 Mask R-CNN을 이기지는 못했지만, 과거에는 사실상 공백에 가까웠던 dense sliding-window instance segmentation을 처음으로 강력한 baseline 수준까지 끌어올렸다. 논문 제목의 “foundation”이라는 표현이 과장이 아닌 이유가 여기에 있다. 저자들도 이를 첫 dense sliding-window instance segmentation system이라고 위치시킨다.

하지만 한계도 분명하다. 가장 직접적인 한계는 연산 비용이다. dense하게 10만 개 이상의 window에 대해 mask를 예측하기 때문에 속도가 느리다. 논문 스스로도 TensorMask가 Mask R-CNN보다 약 4배 이상 느리다고 보고한다. 따라서 실용 시스템 관점에서는 accuracy-gap보다 efficiency-gap이 더 심각한 문제일 수 있다.

또한 최종 성능이 Mask R-CNN에 “가깝다”는 것은 사실이지만, 넘어선 것은 아니다. COCO `test-dev`에서 여전히 1점 이상 차이가 난다. 즉 논문은 dense paradigm의 가능성을 입증했지만, 당시의 최고 실용 해법을 대체했다고 보기는 어렵다. 이 점에서 논문의 공헌은 “state of the art 갱신”보다 “문제 재정의와 방향 제시”에 있다.

학습 설정 면에서도 일부 세부는 완전히 명확하지 않다. 예를 들어 total loss의 구체적인 task weight는 본문에 명시되지 않았다. 또한 다양한 aspect ratio, 더 많은 window size, 더 효율적인 inference pruning 같은 설계는 가능하다고 언급되지만 본 논문에서는 충분히 탐색되지 않았다. 따라서 framework의 잠재력은 제시되었지만, 설계 공간 전체가 탐구된 것은 아니다.

비판적으로 보면, TensorMask는 “mask는 box보다 구조적이므로 더 구조적인 표현이 필요하다”는 주장을 매우 잘 뒷받침하지만, 그 구조를 neural architecture 전체에서 얼마나 깊게 활용했는지는 아직 초기 단계다. 많은 부분이 여전히 RetinaNet/FPN 스타일 설계를 계승하고 있으며, 보다 Tensor-specific한 backbone이나 sparse evaluation 전략은 후속 연구 과제로 남아 있다. 즉 foundation으로는 훌륭하지만, 완성형 시스템으로 보기에는 아직 거친 면이 있다.

## 6. 결론

이 논문은 dense sliding-window instance segmentation을 본격적으로 다루기 위한 첫 체계적 프레임워크로서 `TensorMask`를 제안한다. 핵심 기여는 instance mask를 단순 채널 벡터가 아닌 구조화된 4D tensor로 표현하고, 이를 바탕으로 natural/aligned representation, `align2nat`, `up_align2nat`, `tensor bipyramid` 같은 개념과 연산을 도입한 점이다. 실험적으로도 이러한 구조적 설계가 실제 성능 향상으로 이어지며, TensorMask는 COCO에서 Mask R-CNN에 근접한 성능을 달성한다.

실제 적용 측면에서는 아직 속도 문제가 크기 때문에 바로 주류 시스템이 되기는 어렵다. 그러나 연구적 관점에서는 매우 중요한 논문이다. 이 논문은 instance segmentation이 반드시 box-first 패러다임에 묶일 필요가 없다는 점을 보여주고, dense mask prediction을 위한 표현론적 기반을 제공한다. 따라서 향후 연구에서는 더 효율적인 dense mask inference, 더 정교한 multi-scale mask modeling, box-independent segmentation 설계 같은 방향으로 확장될 가능성이 크다. TensorMask의 진짜 의미는 단일 모델의 우수성보다, dense instance segmentation을 독립적인 연구 분야로 열어젖혔다는 데 있다.
