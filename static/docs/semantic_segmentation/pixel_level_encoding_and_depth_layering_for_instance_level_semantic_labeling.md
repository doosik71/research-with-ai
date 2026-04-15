# Pixel-level Encoding and Depth Layering for Instance-level Semantic Labeling

- **저자**: Jonas Uhrig, Marius Cordts, Uwe Franke, Thomas Brox
- **발표연도**: 2016
- **arXiv**: https://arxiv.org/abs/1604.05096

## 1. 논문 개요

이 논문은 하나의 monocular image만으로 세 가지 출력을 동시에 얻는 방법을 제안한다. 첫째는 pixel-level semantic labeling이고, 둘째는 object instance별 depth 추정이며, 셋째는 각 픽셀이 자신이 속한 객체 중심을 어느 방향으로 바라보는지를 나타내는 instance-based encoding이다. 저자들은 이 세 신호를 fully convolutional network(FCN)가 예측하도록 만든 뒤, 그 결과를 바탕으로 template matching과 proposal fusion 같은 비교적 단순한 저수준 computer vision 기법을 적용하여 instance-level semantic labeling을 수행한다.

연구 문제는 semantic segmentation과 object detection의 중간에 있는 instance-level semantic labeling이다. 기존 pixel-wise semantic labeling은 건물, 도로, 하늘처럼 넓은 배경 영역을 잘 나누지만 같은 클래스의 개별 객체들을 구분하지 못한다. 반대로 object detection은 개별 객체를 찾을 수 있지만 bounding box 수준의 거친 위치 정보만 제공한다. 자율주행이나 로보틱스에서는 개별 차량, 보행자, 자전거 등의 정확한 경계와 개체별 거리 정보가 중요하므로, 이 둘을 결합한 표현이 필요하다.

이 문제가 중요한 이유는 단순히 segmentation 정확도를 높이는 데 그치지 않는다. 논문에서 강조하듯, instance-level 표현은 occlusion reasoning, object tracking, motion estimation, behavior modeling 같은 후속 작업의 기반이 된다. 특히 street scene에서는 동일한 클래스의 객체가 많이 등장하고, 원근에 따라 크기가 크게 달라지며, 가림도 빈번하기 때문에, instance를 정확히 분리하면서 depth까지 함께 추정하는 것은 실제 응용 가치가 크다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 instance를 직접 ID로 예측하지 않고, 각 픽셀이 “자신이 속한 객체의 보이는 중심(visible center)”을 향하는 방향을 분류 문제로 예측하게 만드는 데 있다. 즉, 픽셀마다 instance ID를 붙이려 하지 않고, local한 시각 단서로부터 훨씬 학습하기 쉬운 “center direction”을 예측한 뒤, 이 방향 패턴으로부터 객체 중심을 복원한다. 저자들은 object boundary를 직접 예측하는 방식은 신호가 너무 얇고 예민하다고 본다. 반면 center direction은 물체 내부 전역에 걸쳐 풍부하게 존재하고, 인접 객체 경계에서는 방향이 거의 반대로 바뀌므로 instance separation에 유리하다.

여기에 depth를 함께 예측하는 것이 두 번째 핵심이다. 거리 정보는 단지 부가 정보가 아니라, 서로 가까이 붙어 있는 객체를 분리하고 template 크기를 조절하는 데 직접 사용된다. street scene에서는 멀리 있는 차량과 가까이 있는 차량의 영상상 크기가 크게 다르므로, depth-aware template matching은 매우 자연스러운 설계다. 저자들은 continuous depth를 그대로 회귀하지 않고 class로 discretize했는데, 가까운 물체에는 더 세밀한 해상도를 주고 먼 물체에는 더 넓은 구간을 배정했다.

기존 접근과의 차별점은 다음과 같다. proposal-based 계열처럼 region proposal 품질에 의존하지 않으며, proposal-free이지만 PFN처럼 복잡한 multi-branch 구조나 “이미지 안의 instance 개수”를 직접 예측하는 부담도 없다. 또한 [34], [35]처럼 instance ID나 relative depth ordering을 직접 예측하고 MRF/CRF 같은 복잡한 graphical model로 정제하는 대신, FCN이 semantic, depth, direction의 세 출력을 만들고 후처리는 standard computer vision 기법만 사용한다. 즉, 복잡성의 상당 부분을 네트워크 구조가 아니라 잘 설계된 pixel encoding으로 해결한 점이 이 논문의 가장 중요한 차별점이다.

## 3. 상세 방법 설명

전체 시스템은 크게 두 단계다. 먼저 FCN이 픽셀 단위의 세 출력을 예측한다. 그 다음 이 출력을 이용해 template matching으로 instance center 후보를 찾고, 픽셀 할당과 proposal fusion을 거쳐 최종 instance segmentation을 만든다.

### FCN 출력 표현

저자들은 FCN-8s를 확장하여 세 개의 output channel을 학습한다.

첫 번째 채널은 semantic channel이다. 각 픽셀의 semantic label을 예측하며, 이 정보는 객체 클래스 결정뿐 아니라 background와 object를 구분하고 서로 다른 클래스의 객체를 분리하는 기본 신호가 된다.

두 번째 채널은 depth channel이다. 각 object pixel에 대해 그 instance의 일정한 depth 값을 할당한다. KITTI에서는 3D bounding box center depth를 사용하고, Cityscapes에서는 disparity map 기반의 instance median disparity를 이용한다. 중요한 점은 객체 내부의 모든 픽셀이 같은 depth label을 공유한다는 것이다. 즉, per-pixel geometric depth map을 촘촘히 회귀하는 것이 아니라, instance-level depth를 픽셀에 퍼뜨려 분류 문제로 바꾼 셈이다. depth는 19개 class와 background class로 이산화한다. Supplementary에 따르면 이 구간은 등간격이 아니라, 동일 depth class 안에 들어가는 물체의 영상상 크기가 비슷해지도록 설계했다. 이는 이후 template matching의 안정성을 높이기 위한 선택이다.

세 번째 채널은 direction channel이다. 각 object pixel에 대해 그 픽셀에서 해당 객체의 visible center로 향하는 방향 각도를 계산하고, 이를 여러 개의 angle bin으로 나눈 classification target으로 사용한다. 실험에서는 8개 방향 class를 사용하며 각 class는 $45^\circ$를 담당한다. 경계 예측보다 이 방식이 유리한 이유는, 물체 전체에서 안정적인 supervision을 제공하고 인접 인스턴스 사이에서는 방향 패턴이 강하게 달라지기 때문이다. 논문은 occlusion이 있어도 physical center가 아니라 visible center를 기준으로 삼기 때문에 대부분의 가림 상황을 잘 처리한다고 설명한다.

이 세 채널은 모두 pixel-wise discrete labeling task로 학습되며, 각각 standard cross-entropy loss를 사용한다. 즉, 전체 학습 목표는 세 개 loss의 결합이다. 논문은 별도의 가중치 식을 명시하지는 않았고, “separate cross-entropy loss for each of our three output channels”라고 설명한다. 따라서 loss 결합 방식의 세부 가중치는 본문에 명확히 제시되지 않았다.

네트워크 구조 측면에서, 저자들은 가장 큰 downsampling 이후 deconvolution layer와 skip connection을 사용해 입력 해상도의 $\frac{1}{8}$ 크기 representation을 만든다. 중간 표현은 channel depth 100을 유지하고, 마지막에 $1 \times 1$ convolution으로 semantic, depth, direction 채널로 축소한다. 최종적으로 bilinear upsampling을 통해 full resolution 출력을 얻는다. 이 설계는 채널 수를 바꿔도 upsampling layer 전체를 다시 초기화하지 않기 위해 고안되었다고 설명한다.

### Direction의 연속값 복원

direction은 학습 시 이산 class이지만, 후처리를 위해 더 정확한 방향이 필요하다. 저자들은 softmax 정규화 후 각 방향 class vector를 score로 가중 평균하여 continuous direction estimate를 복원한다. 즉, hard argmax 하나만 쓰지 않고, 여러 방향 class의 확률분포를 이용해 더 부드러운 방향 벡터를 만든다.

### Template Matching

FCN이 만든 direction map에는 instance center 주변에서 독특한 방향 패턴이 나타난다. 저자들은 이를 이용해 rectangular template로 normalized cross-correlation(NCC)을 수행한다. template의 aspect ratio는 semantic category에 따라 달리 설정한다. 예를 들어 pedestrian과 vehicle은 형태 비가 다르므로 같은 템플릿을 쓰지 않는다. 또 predicted depth class에 따라 template 크기를 조절해, 가까운 물체에는 큰 template, 먼 물체에는 작은 template를 사용한다.

이 단계에서는 semantic class를 그대로 모두 따로 다루기보다 human, car, large vehicle, two wheeler의 4개 category로 묶는다. 이는 유사 class 간 혼동이 score map에 미치는 영향을 줄이기 위한 선택이다. 결과적으로 각 category마다 “여기가 instance center일 가능성”을 나타내는 score map이 만들어진다.

### Instance Generation

instance center 검출은 template matching score map에서 non-maximum suppression(NMS)으로 local maximum을 반복적으로 찾는 방식이다. suppression 영역 크기는 template size와 동일하게 둔다. 이렇게 얻은 점들이 temporary instance centers다.

그 다음 각 픽셀을 가장 가까운 temporary center 중에서, 픽셀의 상대 위치와 FCN이 예측한 direction이 일치하는 center에 할당한다. 이 과정을 통해 center별 pixel 집합이 생기며, 이것이 instance proposal이다.

### Proposal Fusion

초기 proposal은 over-segmentation이 발생하기 쉽다. 특히 길쭉한 객체이거나 depth prediction이 부정확할 때 하나의 물체가 여러 조각으로 나뉠 수 있다. 이를 해결하기 위해, 저자들은 proposal 내부 direction vector의 누적 편향을 본다. 완전한 객체라면 내부 방향 벡터들이 좌우로 상쇄되어 합이 작아지는 경향이 있다. 반면 객체 일부만 포함한 불완전한 proposal은 특정 방향으로 편향된다. 이때 그 방향 쪽에 semantic class와 depth가 잘 맞는 인접 proposal이 있으면 둘을 fuse한다.

최종적으로 남은 각 instance에는 영역 내부 평균 depth와 가장 빈도가 높은 semantic label을 부여한다. 그리고 instance에 속하지 않는 픽셀은 semantic channel의 argmax 결과를 그대로 사용한다. 이렇게 하면 foreground instance와 background semantic labeling이 함께 포함된 일관된 scene representation이 완성된다.

## 4. 실험 및 결과

### 데이터셋과 평가 설정

논문은 KITTI object detection dataset의 instance segmentation 확장판과 Cityscapes를 사용한다. 두 데이터셋 모두 semantic, instance, depth 관련 annotation을 제공한다. KITTI에서는 3D bounding box 중심 깊이를, Cityscapes에서는 disparity 기반 instance median depth를 ground truth depth로 사용했다. 데이터 분할은 공식 train/val/test split을 따른다.

instance segmentation 평가는 KITTI에서는 [35]의 metric들을 사용하고, Cityscapes에서는 [6]의 AP 계열 지표를 사용한다. depth 평가는 instance 단위로 MAE, RMSE, ARD, 그리고 $\delta_1, \delta_2, \delta_3$를 사용한다. 여기서 $\delta_i = 1.25^i$ 기준의 inlier ratio다. 이 depth metric은 ground truth와 50% 이상 겹치는 instance에 대해서만 계산한다.

### 구현 세부사항

Cityscapes에서는 19개 semantic class를 사용하고, 8개 object class를 4개 category로 합쳤다. KITTI는 instance annotation이 car에만 존재한다. depth는 두 데이터셋 모두 19개 class와 background class를 사용했다. direction은 8개 등분된 angle class다. 네트워크는 ImageNet으로 초기화된 FCN-8s를 사용하고, upsampling layer는 랜덤 초기화 후 두 데이터셋에서 fine-tuning했다.

Supplementary에 따르면 depth bin은 데이터셋마다 다르게 설정했다. 예를 들어 KITTI는 class 1이 0–2m, class 19가 76m 이상이고, Cityscapes는 class 1이 0–6m, class 19가 100m 이상이다. Cityscapes는 200m 이상 인스턴스도 있어 더 넓은 범위를 사용한다.

### Ablation Study

저자들은 각 구성요소의 효과를 보기 위해 세 가지 변형을 실험했다.

`Ours-D`는 depth channel을 제거한 버전이다. 이 경우 template 크기를 scale-agnostic하게 고정해야 한다. 작은 template를 써서 많은 proposal을 만들고, 이후 fusion에 많이 의존하는 것이 가장 잘 동작했지만, 전체 시스템보다 성능이 뚜렷하게 낮았다. 이는 depth가 instance separation에 핵심이라는 뜻이다.

`Ours-D-F`는 depth와 fusion을 모두 제거한 버전이다. over-segmentation을 줄이기 위해 더 큰 template가 필요했고, 성능은 더 크게 떨어졌다.

`Ours-F`는 depth는 유지하지만 proposal fusion 없이 초기 proposal을 그대로 최종 결과로 쓰는 버전이다. 이 역시 full model보다 성능이 낮았고, 심지어 일부 경우 `Ours-D`보다도 약간 나빴다.

즉, semantic, depth, direction 세 채널뿐 아니라, 그 위에 얹힌 fusion 과정도 최종 성능에 실질적으로 필요하다는 것이 두 데이터셋 모두에서 일관되게 관찰되었다.

### KITTI 결과

KITTI test에서 논문 방법은 기존 최고 성능인 [34], [35]를 큰 폭으로 앞선다. 표 1에 따르면 `InsF1`은 기존 최고 56.6에서 79.7로 증가했다. `IoU`도 77대에서 84.1로 상승했고, `InsPr`, `InsRe` 역시 각각 86.3, 74.1로 크게 높다. 논문은 [35]의 best variant 대비 모든 metric 평균 기준으로 37% relative improvement를 얻었다고 주장한다.

이 결과는 단순히 특정 metric 하나만 좋아진 것이 아니라, precision, recall, coverage 계열 지표 전반에서 개선이 나타났다는 점에서 의미가 있다. proposal-free 방식이면서도 graphical model 없이 이 정도 향상을 만든 것은 당시 기준으로 상당히 강한 결과다.

### Cityscapes 결과

Cityscapes test에서는 baseline인 `MCG+R-CNN` 대비 전체 AP가 4.6에서 8.9로 거의 두 배 수준으로 상승했다. `AP 50%`는 12.9에서 21.1, `AP 100m`는 7.7에서 15.3, `AP 50m`는 10.3에서 16.7로 모두 개선되었다. car class만 보면 AP가 10.5에서 22.5로 크게 상승했다.

Supplementary의 class-level 결과를 보면, `person`, `rider`, `car`에서 특히 강하고, `truck`, `bus`, `train`은 상대적으로 약하다. 저자들은 이 이유를 semantic labeling 단계에서 해당 클래스의 분류가 덜 안정적이기 때문이라고 설명한다. 실제 confusion matrix에서도 truck, bus, train이 car 등 다른 vehicle 계열로 혼동되는 경향이 보인다.

논문은 Cityscapes의 절대 수치가 KITTI보다 훨씬 낮다고 해석하는데, 이는 Cityscapes 장면이 더 복잡하고 인스턴스 수가 많으며 먼 거리 객체도 많기 때문이라고 본다. 이 해석은 표의 수치 차이와 정성적 예시 모두와 일치한다.

### Depth 평가

이 논문의 특징 중 하나는 instance segmentation과 함께 instance-level absolute depth를 예측한다는 점이다. KITTI test에서 MAE 1.7m, RMSE 2.8m, ARD 7.7%, $\delta_1$ 95.1%, $\delta_2$ 99.3%, $\delta_3$ 99.8%를 보고했다. Cityscapes val에서는 MAE 7.7m, RMSE 24.8m, ARD 11.3%, $\delta_1$ 86.2%다.

이 수치는 monocular single image만을 입력으로 사용했다는 점을 고려하면 상당히 인상적이다. 물론 Cityscapes에서 RMSE가 크게 증가하는데, 이는 원거리 객체와 복잡한 도시 장면의 영향으로 보이며, 논문도 Cityscapes가 훨씬 더 어려운 데이터셋이라고 해석한다.

### Semantic Segmentation 평가

Cityscapes test에서 semantic pixel-level 성능도 함께 보고한다. `IoU class`는 64.3으로 `FCN 8s` 65.3, `Dilation10` 67.1보다 약간 낮다. `iIoU class`도 41.6으로 거의 비슷한 수준이다. 반면 `iIoU category`는 73.9로, `FCN 8s` 70.1과 `Dilation10` 71.1보다 높다.

즉, 이 방법은 주된 목적이 instance segmentation임에도, semantic segmentation 단독 SOTA와 거의 같은 수준의 semantic quality를 유지한다. 저자들은 이를 “harder instance segmentation task에 초점을 맞췄음에도 state-of-the-art에 근접하거나 일부 지표에서는 능가한다”고 해석한다. 다만 `IoU class` 자체는 최고 성능보다 낮으므로, semantic segmentation 절대 최고 성능을 달성했다고 과장해서 읽어서는 안 된다. 논문이 더 조심스럽게 말하는 지점은 instance-aware category score에서 새로운 최고 성능이라는 점이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 instance segmentation 문제를 복잡한 structured inference 없이도 잘 풀 수 있음을 보여준 점이다. semantic, depth, direction이라는 세 가지 pixel encoding이 후처리를 단순하게 만들고, 그 결과 proposal-based 방법이나 MRF 기반 방법을 크게 앞서는 성능을 달성했다. 특히 instance 개수를 미리 예측하지 않아도 되므로, street scene처럼 이미지당 수십에서 수백 개 instance가 나올 수 있는 환경에 잘 맞는다.

또 다른 강점은 representation 설계의 적절성이다. direction-to-center encoding은 boundary보다 학습 신호가 풍부하고, depth discretization은 scale variation과 instance separation에 직접 기여한다. 이 둘을 결합해 template matching으로 center를 찾는 방식은 단순하지만 설득력이 높고, ablation study도 각 구성요소의 필요성을 뒷받침한다.

실용성 측면에서도 장점이 있다. 하나의 monocular image에서 instance mask, semantic label, absolute depth를 동시에 얻을 수 있기 때문이다. 자율주행처럼 거리와 개체 단위 구조가 모두 필요한 영역에 적합한 holistic scene representation을 제시했다는 점은 분명한 기여다.

한계도 있다. 먼저, 후처리 단계에서 semantic category별 template의 aspect ratio와 depth별 template size를 데이터셋 특성에 맞게 조정해야 한다. 논문도 KITTI와 Cityscapes에서 서로 다른 depth range와 template 설정을 사용했다고 밝힌다. 이는 방법이 완전히 parameter-free하거나 dataset-agnostic하지 않다는 뜻이다.

또한 direction 기반 template matching은 객체 중심 패턴이 비교적 명확한 street object에 잘 맞지만, 더 비정형적이거나 복잡한 형상의 객체에도 동일하게 잘 일반화되는지는 본문만으로는 확인할 수 없다. 논문은 urban street scenes에 초점을 맞추고 있으며, 다른 도메인에 대한 검증은 제공하지 않는다.

semantic 예측 품질에 대한 의존성도 분명하다. Supplementary에서 bus, truck, train의 성능 저하 원인을 semantic labeling의 낮은 신뢰도로 설명하는데, 이는 instance generation이 초기 semantic channel 품질에 크게 좌우된다는 뜻이다. 다시 말해, 이 방법은 instance segmentation만의 독립된 분리 능력만으로 작동한다기보다 semantic recognition과 강하게 결합된 시스템이다.

마지막으로, loss 조합의 세부 weighting, proposal fusion의 정확한 수치적 기준, 템플릿 파라미터의 구체값 등 일부 구현 세부는 본문에서 완전히 상세히 풀어 쓰이지 않았다. supplementary가 일부 보완하지만, 완전 재현에 필요한 모든 수치가 이 발췌본에 다 포함되어 있다고 보기는 어렵다. 따라서 재현성에 필요한 세부는 원문 전체 구현 정보나 공개 코드가 추가로 필요할 수 있다.

## 6. 결론

이 논문은 instance-level semantic labeling을 위해 각 픽셀에 semantic class, instance depth, center direction을 부여하는 새로운 encoding을 제안했고, 이를 FCN으로 예측한 뒤 template matching과 proposal fusion으로 instance를 복원하는 간결한 파이프라인을 제시했다. 핵심은 instance ID를 직접 예측하지 않고도, 잘 설계된 중간 표현만으로 강력한 instance segmentation이 가능하다는 점을 실험적으로 입증한 것이다.

실험적으로는 KITTI와 Cityscapes에서 당시 기존 방법들을 큰 폭으로 능가했고, 특히 KITTI에서는 평균 37% relative improvement, Cityscapes에서는 AP를 거의 두 배로 끌어올렸다. 동시에 monocular image 기반 instance depth 추정까지 제공해, 단순 segmentation을 넘어선 장면 이해 표현을 만들었다.

향후 연구 관점에서 보면, 이 논문은 proposal-free instance segmentation의 초기 흐름에서 매우 중요한 위치를 차지한다. 복잡한 structured model이나 region proposal 없이도 pixel encoding과 간단한 grouping만으로 높은 성능을 낼 수 있음을 보여주었기 때문이다. 실제 응용 측면에서도 자율주행, 로보틱스, scene understanding처럼 개별 객체의 경계와 거리 추정이 동시에 필요한 문제에 유의미한 기반이 될 가능성이 크다.
