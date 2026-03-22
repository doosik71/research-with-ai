# Pose2Seg: Detection Free Human Instance Segmentation

* **저자**: Song-Hai Zhang, Ruilong Li, Xin Dong, Paul Rosin, Zixi Cai, Xi Han, Dingcheng Yang, Haozhi Huang, Shi-Min Hu
* **발표연도**: 2018
* **arXiv**: [https://arxiv.org/abs/1803.10683](https://arxiv.org/abs/1803.10683)

## 1. 논문 개요

이 논문은 사람(human) instance segmentation을 일반적인 object detection 기반 방식이 아니라, 사람의 pose를 중심으로 해결하자는 문제의식을 제시한다. 기존의 대표적 instance segmentation 방법들은 대체로 먼저 bounding box를 검출한 뒤, 그 box 내부에서 mask를 예측하는 구조를 사용한다. Mask R-CNN도 detection과 segmentation을 긴밀하게 결합했지만, 여전히 사람 인스턴스를 box 단위로 다루는 큰 틀은 유지한다. 논문은 이 방식이 사람처럼 서로 많이 겹치고 얽히는 장면, 특히 심한 occlusion 상황에서 구조적으로 불리하다고 본다.

핵심 연구 문제는 다음과 같다. 사람은 다른 일반 객체와 달리 pose skeleton으로 잘 정의될 수 있는데, 왜 사람 instance segmentation에서도 여전히 bounding box 중심 사고에 머물러 있는가? 그리고 두 사람이 심하게 겹쳐 있을 때, box는 두 사람을 함께 감싸는 경우가 많아 segmentation 네트워크가 어느 사람이 목표인지 구분하기 어렵다. 반면 pose skeleton은 각 관절 위치와 body part의 배치를 통해 인스턴스를 더 명확히 구분할 수 있다. 논문은 이 점을 출발점으로 하여, detection-free human instance segmentation 프레임워크인 Pose2Seg를 제안한다.

이 문제가 중요한 이유는 실제 환경에서 사람 간 가림은 매우 흔하기 때문이다. 군중, 스포츠, 공연, 거리 장면 등에서는 사람끼리 서로 가리거나 엉켜 있는 경우가 많다. 그러나 기존의 public dataset은 이런 상황을 충분히 포함하지 않거나, pose와 mask를 함께 제공하지 않는 경우가 많았다. 따라서 논문은 방법론 제안에 그치지 않고, 심한 가림에 초점을 둔 OCHuman 데이터셋도 함께 제안한다. 즉, 이 논문은 방법과 벤치마크를 동시에 제시해 “occluded human instance segmentation”이라는 문제를 독립적인 연구 주제로 부각시킨다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 사람 인스턴스를 분리할 때 bounding box 대신 pose를 기준으로 삼는 것이다. 일반적인 detection 기반 framework에서는 proposal region을 많이 만든 뒤 NMS로 중복 box를 제거하는데, 같은 클래스 객체가 크게 겹치면 한 인스턴스가 제거되거나, 남더라도 box 내부에 여러 사람이 함께 들어와 segmentation이 어려워진다. 논문은 사람 범주에서는 pose가 훨씬 더 본질적인 인스턴스 표현이라고 주장한다.

이 직관은 두 가지 설계로 구체화된다. 첫째, pose를 이용해 사람을 정렬하는 **Affine-Align** 모듈을 설계한다. 이는 box를 잘라 정규화하는 RoI-Align과 대응되지만, 단순히 사각형 crop을 쓰지 않고 pose template에 맞춰 scale, translation, rotation, 좌우 반전까지 허용하는 affine transform을 적용한다. 이렇게 하면 누워 있거나 뒤집혀 있거나 비틀린 사람도 표준화된 자세 방향으로 정렬할 수 있다. 논문은 이것이 segmentation을 더 쉽게 만든다고 본다.

둘째, segmentation 네트워크에 이미지 특징만 넣지 않고 **Skeleton features**를 명시적으로 함께 넣는다. 사람 pose와 사람 mask는 독립이 아니라, pose가 mask의 구조적 골격이라고 볼 수 있다. 따라서 관절 confidence map과 Part Affinity Fields(PAFs)를 인공적으로 구성해 aligned RoI feature에 concat하면, 네트워크가 어떤 픽셀이 목표 사람에 속하는지 더 쉽게 판단할 수 있다. 특히 한 RoI 안에 여러 사람이 함께 들어오는 경우에도, skeleton 정보가 목표 인스턴스를 명시적으로 가리키는 역할을 한다.

기존 접근과의 차별점은 분명하다. Mask R-CNN이나 Pose2Instance 같은 방법은 여전히 detection에 의존한다. PersonLab은 detection-free 성격이 있지만 논문에 따르면 segmentation 정확도 면에서 Mask R-CNN보다 약하다. Pose2Seg는 bottom-up pose estimation의 장점을 가져오면서, segmentation 단계 전체를 pose 중심으로 다시 설계했다는 점에서 차별적이다.

## 3. 상세 방법 설명

Pose2Seg의 전체 파이프라인은 비교적 명확하다. 입력은 이미지와 각 사람의 human pose이다. 먼저 backbone network가 이미지 feature를 추출한다. 그런 다음 각 사람 pose에 대해 **Affine-Align**을 수행해 사람별 aligned RoI를 만든다. 동시에 해당 pose로부터 **Skeleton features**를 생성해 aligned RoI feature에 결합한다. 이후 이 결합된 feature를 **SegModule**에 입력해 instance mask를 예측한다. 마지막으로 Affine-Align에서 계산했던 변환 행렬을 역으로 적용해 원래 이미지 좌표계로 mask를 되돌린다.

### 3.1 Human pose 표현

논문은 사람 pose를 다음과 같이 표현한다.

$$
P = (C_1, C_2, \dots, C_m) \in \mathbb{R}^{m \times 3}
$$

여기서 $m$은 관절 개수이며 COCO에서는 17이다. 각 관절은

$$
C_i = (x, y, v) \in \mathbb{R}^3
$$

로 표현된다. $x, y$는 관절 좌표이고, $v$는 visibility 관련 상태값이다.

논문은 pose template clustering을 위해 각 관절 상태를 다음처럼 정리한다.

$$
C_j =
\begin{cases}
(x, y, 2) & \text{if } C_j \text{ is visible} \
(x, y, 1) & \text{if } C_j \text{ is not visible} \
(0.5, 0.5, 0) & \text{if } C_j \text{ is not in image}
\end{cases}
$$

즉, 단순히 좌표만 쓰는 것이 아니라 보이는지, 가려졌는지, 이미지 밖인지까지 포함한다. 이것은 occlusion 상황에서 특히 중요하다.

### 3.2 Pose template 생성

Affine-Align의 핵심은 각 사람 pose를 어떤 “표준 자세”에 맞춰 정렬하는 것이다. 이를 위해 논문은 training set의 pose들을 K-means로 clustering하여 pose template를 만든다. 최적화 목적은 다음과 같다.

$$
\arg\min_{S} \sum_{i=1}^{K} \sum_{P \in S_i} Dist(P, P_{\mu i})
$$

여기서 $P_{\mu i}$는 cluster $S_i$의 평균 pose이다. pose 간 거리는

$$
Dist(P, P_{\mu i}) = \sum_{j=1}^{m} |C_j - C_{\mu ij}|^2
$$

로 정의된다.

논문은 clustering 전에 몇 가지 전처리를 한다. 먼저 bounding box를 이용해 square RoI를 만들고 사람을 중앙에 놓는다. 그다음 이를 $1 \times 1$ 공간으로 정규화하여 pose 좌표를 $(0,1)$ 범위로 맞춘다. 또한 valid point가 8개 이하인 pose는 template 생성에 쓰지 않는다. 이런 pose는 정보가 부족하고 outlier처럼 작용하기 때문이다.

실험적으로 $K=3$이 최종 선택된다. 논문 해석에 따르면 COCO에서 가장 흔한 pose는 half-body pose와 full-body pose이며, $K=3$일 때 half-body, full-body backview, full-body frontview 정도의 template가 형성된다. $K \ge 4$에서는 left-right 차이가 생기지만, 어차피 Affine-Align이 좌우 반전을 다루므로 굳이 더 세분화할 필요가 없다고 본다.

### 3.3 Affine-Align

기존 RoI-Align은 bounding box 기반 crop-and-resize이다. 반면 Affine-Align은 pose를 template pose에 맞추는 affine transform을 계산한다. 이때 변환 행렬 $H$는 rotation, scale, x/y translation, left-right flip을 포함하는 $2 \times 3$ 행렬이다.

논문은 다음 최적화를 통해 각 template에 대한 최적 변환을 구한다.

$$
H^* = \arg\min_H |H \cdot P - P_{\mu}|
$$

그리고 각 template에 대해 계산된 오차를 score로 바꿔 가장 잘 맞는 template를 선택한다.

$$
score = \exp(-|H^* \cdot P - P_{\mu}|)
$$

직관적으로 보면, 입력 pose를 여러 표준 자세에 각각 맞춰 본 뒤 가장 잘 맞는 하나를 고르는 것이다. 이렇게 선택된 $H^*$를 이미지나 feature map에 bilinear interpolation과 함께 적용해 $64 \times 64$ 크기의 aligned RoI를 얻는다.

이 설계의 장점은 단순한 scale normalization을 넘어선다는 점이다. 사람이 비스듬히 서 있거나 뒤집혀 있거나 한쪽으로 크게 기울어져 있어도, rotation과 flip을 사용해 보다 canonical한 방향으로 정렬할 수 있다. 논문은 이것이 segmentation 네트워크가 배워야 할 변형을 줄여 준다고 본다.

논문은 공통 valid point가 최소 3개는 있어야 $H$를 유일하게 추정할 수 있다고 설명한다. 만약 어떤 template와도 3개 이상의 공통 valid point를 만들지 못한다면, 예를 들어 관절이 1개만 보이는 경우라면, 전체 이미지를 목표 해상도로 align하는 방식으로 처리한다고 한다. 이는 큰 단일 인물처럼 보이는 경우가 많아 실무적으로 괜찮다고 설명하지만, 이런 예외 처리가 정확히 얼마나 자주 발생하는지는 본문에 수치로 제시되지 않는다.

### 3.4 Skeleton features

논문은 pose를 단지 align에만 쓰지 않고, segmentation 네트워크의 입력 feature로도 직접 사용한다. 구체적으로는 두 종류의 feature를 만든다.

첫째는 **PAFs (Part Affinity Fields)** 이다. 이는 skeleton의 각 limb를 2-channel vector field로 나타낸다. COCO에서는 19개의 skeleton 연결이 있으므로 총 38채널이다.

둘째는 각 body part 주변의 중요도를 나타내는 **part confidence maps** 이다. COCO의 17개 keypoint에 대해 17채널을 사용한다.

따라서 Skeleton features의 총 채널 수는

$$
17 + 38 = 55
$$

이다.

이 55채널 feature map을 aligned image feature와 concatenate하면, 네트워크는 픽셀 수준 정보를 볼 때 “어디에 관절이 있고 어떤 limb가 연결되는지”를 동시에 참고할 수 있다. 논문은 이것이 특히 동일 RoI 안에 여러 사람이 함께 나타나는 상황에서 유리하다고 주장한다. 이미지 feature만으로는 경계가 모호할 수 있지만, skeleton feature는 목표 사람의 구조를 직접 알려 주기 때문이다.

### 3.5 SegModule

SegModule은 aligned feature와 Skeleton features를 입력으로 받아 mask를 예측하는 모듈이다. 저자들은 Skeleton features가 인공적으로 추가된 특징이므로, 네트워크가 이들과 이미지 feature의 관계를 충분히 이해하려면 넓은 receptive field가 필요하다고 본다.

구조는 다음과 같다. 먼저 $7 \times 7$ stride-2 convolution을 적용하고, 그 뒤에 여러 개의 standard residual unit을 쌓는다. 이후 bilinear upsampling으로 해상도를 복원하고, residual unit 하나와 $1 \times 1$ convolution으로 최종 mask를 예측한다.

논문은 10개의 residual unit을 사용한 구조가 약 50픽셀 receptive field를 가져, $64 \times 64$ alignment size에 충분하다고 설명한다. 너무 얕으면 전역 구조를 이해하기 어렵고, 너무 깊어도 성능 향상은 작다. 이 주장은 ablation에서 뒷받침된다.

### 3.6 학습과 추론

모든 모델은 COCOPersons training set으로 end-to-end 학습한다. 입력 해상도는 $512 \times 512$이다. 학습 스케줄은 learning rate $2 \times 10^{-4}$로 시작해 33 epoch 이후 0.1배 감소시키고, 총 40 epoch 동안 학습한다. batch size는 4이며, TITAN X (Pascal) 한 장으로 약 80시간 학습했다고 한다.

추론 시에는 먼저 pose detector가 keypoint를 예측하고, 그 결과로 Affine-Align과 Skeleton feature 생성이 진행된다. 논문은 이 전체 시스템이 약 20 FPS로 동작한다고 보고한다. 다만 이 속도에 pose estimation까지 모두 포함되는지, 또는 segmentation 부분만 기준인지 본문만으로는 완전히 명확하지 않다. 텍스트상으로는 “images and keypoints as inputs”라고 되어 있으므로, segmentation framework 기준 속도로 이해하는 것이 자연스럽다.

## 4. 실험 및 결과

논문은 두 가지 큰 축에서 실험한다. 하나는 심한 occlusion에 특화된 OCHuman에서의 성능이고, 다른 하나는 일반 장면에 가까운 COCOPersons에서의 성능이다. 또한 alignment 방식, Skeleton features 유무, SegModule 깊이에 대한 ablation도 수행한다.

### 4.1 OCHuman 데이터셋

OCHuman은 논문이 새로 제안하는 benchmark이다. 4731장 이미지에 8110명의 사람 instance가 있으며, 모두 심하게 가려진 경우만 포함한다. occlusion severity는 동일 이미지 내 다른 같은 클래스 객체와의 최대 IoU인 MaxIoU로 정의한다. MaxIoU $> 0.5$이면 heavy occlusion으로 간주한다.

데이터셋의 평균 MaxIoU는 0.67로 제시된다. 이는 COCO person subset의 평균 MaxIoU 0.08보다 훨씬 높다. 더 구체적으로 COCO(train+val) person 273,469명 중 MaxIoU $> 0.5$는 2,619명, MaxIoU $> 0.75$는 214명에 불과하지만, OCHuman은 8,110명 전원이 MaxIoU $> 0.5$이며 그중 2,614명이 MaxIoU $> 0.75$이다. 이 비교는 OCHuman이 정말로 occlusion 중심 benchmark임을 보여 준다.

데이터는 validation 2500장, test 2231장으로 나뉘며 각각 4313개, 3797개 instance를 포함한다. 또한 MaxIoU 0.5~0.75 구간을 OCHuman-Moderate, 0.75 초과를 OCHuman-Hard로 나눠 난이도별 평가도 가능하게 설계했다.

### 4.2 Occlusion 환경 성능

OCHuman에서 Mask R-CNN과 Pose2Seg를 비교한 결과는 논문의 가장 중요한 메시지다.

OCHuman validation set에서:

* Mask R-CNN: AP 0.163, $AP_M$ 0.194, $AP_H$ 0.113
* Ours: AP 0.222, $AP_M$ 0.261, $AP_H$ 0.150
* Ours (GT Kpt): AP 0.544, $AP_M$ 0.576, $AP_H$ 0.491

OCHuman test set에서:

* Mask R-CNN: AP 0.169, $AP_M$ 0.189, $AP_H$ 0.128
* Ours: AP 0.238, $AP_M$ 0.266, $AP_H$ 0.175
* Ours (GT Kpt): AP 0.552, $AP_M$ 0.579, $AP_H$ 0.495

여기서 $AP_M$은 Moderate subset, $AP_H$는 Hard subset으로 해석된다. predicted keypoint를 쓴 일반 Pose2Seg만 봐도 Mask R-CNN보다 validation에서 약 36%, test에서 약 41% 정도 높은 절대 향상을 보인다. 본문은 “nearly 50% higher”라고 표현하는데, 이는 상대 향상률 기준으로 보면 타당하다. 가장 중요한 점은 Hard subset에서도 일관된 개선이 있다는 것이다. 즉, 이 방법은 단순히 일반 장면에서 조금 좋아지는 수준이 아니라, 논문이 겨냥한 “심한 가림” 상황에서 특히 강점을 보인다.

또한 ground-truth keypoint를 넣으면 성능이 0.54~0.55 AP 수준으로 크게 상승한다. 이는 Pose2Seg 자체의 상한선은 꽤 높고, 실제 병목이 pose detector 품질에 있음을 시사한다. 논문이 제안하는 구조가 좋더라도, 입력 keypoint가 불안정하면 전체 성능이 제한된다는 의미다.

### 4.3 General case 성능

COCOPersons validation set에서도 Pose2Seg는 유리한 결과를 보인다.

* Mask R-CNN (ResNet50-FPN): AP 0.532, $AP_M$ 0.433, $AP_L$ 0.648
* PersonLab (ResNet101/152, 일부 multi-scale): 최고 $AP_M$ 0.497, $AP_L$ 0.621
* Ours (ResNet50-FPN): AP 0.555, $AP_M$ 0.498, $AP_L$ 0.670
* Ours (GT Kpt): AP 0.582, $AP_M$ 0.539, $AP_L$ 0.679

즉, Pose2Seg는 occlusion 전용 장면뿐 아니라 일반적인 COCO person segmentation에서도 Mask R-CNN보다 좋다. 이 결과는 중요한데, pose-based 설계가 “특수한 경우에만 쓰는 꼼수”가 아니라 일반 환경에서도 경쟁력이 있음을 보여 주기 때문이다. 논문은 PersonLab과도 비교하지만, PersonLab 수치는 그 논문에서 가져온 것이며 학습 설정이 완전히 같지는 않다. 따라서 가장 공정한 비교는 재학습한 Mask R-CNN과의 비교라고 보는 것이 적절하다.

### 4.4 Affine-Align vs RoI-Align

Ablation에서 논문은 alignment 전략의 차이를 집중적으로 본다. OCHuman validation set에서 ground-truth bounding box 기반 RoI-Align은 0.476 AP를 얻는다. 반면 ground-truth keypoint 기반 Affine-Align은 0.544 AP를 얻는다. 즉, NMS 문제를 GT box 사용으로 제거한 상황에서도, pose-based alignment 자체가 더 유리하다는 뜻이다.

논문은 그 이유를 rotation 허용에서 찾는다. Affine-Align은 사람을 판별하기 쉬운 방향으로 정렬해 더 discriminative한 RoI를 만든다. box crop은 같은 사람이라도 자세에 따라 나타나는 공간적 변형이 크고, 특히 서로 얽힌 두 사람을 분리하는 데 불리하다.

또한 “직관적 pose-based alignment”도 실험한다. 이는 keypoint들로 bounding box를 만든 뒤, 그 box를 RoI-Align으로 처리하는 방식이다. keypoint의 min/max로 box를 만들고, 이를 확장 비율 $\alpha$로 조절한다. 하지만 $\alpha$를 30%에서 100%까지 바꿔도 OCHuman과 COCOPerson 모두에서 Affine-Align을 넘지 못한다. 이는 pose를 단지 box 생성에만 활용하는 것은 충분하지 않고, rotation/flip까지 포함한 진짜 pose alignment가 중요하다는 점을 보여 준다.

### 4.5 Skeleton features의 기여

Table 5는 Skeleton features가 다양한 alignment 전략 전반에서 성능 향상에 기여함을 보여 준다. 표의 각 항목은 “with / without Skeleton” 순으로 제시되어 있다.

예를 들어 GT KPT + Affine-Align의 경우:

* OCHuman: 0.544 / 0.141
* COCOPerson: 0.582 / 0.386

즉 Skeleton features를 제거하면 성능이 크게 떨어진다. 특히 OCHuman에서 0.544에서 0.141로 떨어지는 것은 매우 큰 차이인데, 이는 occlusion 상황에서 skeleton 정보가 사실상 핵심 신호임을 강하게 시사한다.

다만 이 수치 차이는 꽤 극적이기 때문에, 표의 형식을 조심해서 읽어야 한다. 본문 설명상 표는 각 전략에서 “(+/-) Skeleton”을 비교하고 있으며, Affine-Align과 Skeleton의 조합이 가장 강력하다고 해석하는 것이 맞다. 논문이 말하고 싶은 바는 단순하다. aligned image feature만으로는 target person을 분리하기 어렵고, Skeleton features가 그 목표를 명시적으로 알려 준다.

### 4.6 SegModule 깊이

SegModule의 residual unit 수를 5, 10, 15, 20으로 바꿔 실험한 결과는 다음과 같다.

* 5 units, receptive field 약 30: AP 0.545
* 10 units, receptive field 약 50: AP 0.555
* 15 units, receptive field 약 70: AP 0.555
* 20 units, receptive field 약 90: AP 0.556

즉, 10개 정도면 충분하고 더 깊어져도 이득이 거의 없다. 이는 aligned RoI 크기인 $64 \times 64$에 대해 약 50픽셀 receptive field면 전체 구조를 이해하기에 충분하다는 저자 주장과 일치한다. 네트워크를 무조건 깊게 쌓기보다, 필요한 receptive field를 달성하는 적절한 깊이를 고르는 것이 중요하다는 점을 보여 준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 방법 설계가 매우 잘 맞물린다는 점이다. 저자들은 “사람”이라는 클래스의 고유성을 진지하게 활용한다. 일반 object instance segmentation 프레임워크를 그대로 사람에게 적용하는 대신, 사람을 pose로 표현할 수 있다는 사실을 중심에 두고 전체 pipeline을 재설계했다. 이는 단순한 feature 추가가 아니라, alignment 기준 자체를 바꾸는 구조적 발상이다.

두 번째 강점은 occlusion 문제를 정면으로 다뤘다는 점이다. 많은 기존 방법이 crowded scene을 어려운 예외 상황 정도로 취급했다면, 이 논문은 오히려 그 상황을 핵심 평가 대상으로 삼는다. 그리고 OCHuman이라는 데이터셋을 직접 제안해, 이후 연구가 실제로 이 문제를 평가할 수 있게 만들었다. 방법론과 benchmark를 함께 제시한 것은 학술적으로 의미가 크다.

세 번째 강점은 ablation이 비교적 설득력 있다는 점이다. 논문은 단순히 “성능이 좋아졌다”에서 멈추지 않고, Affine-Align, Skeleton features, SegModule depth를 분리해서 검증한다. 특히 GT bounding box를 사용한 RoI-Align과 비교해도 Affine-Align이 낫다는 결과는, 개선이 단지 detector failure를 피했기 때문만은 아님을 보여 준다.

하지만 한계도 분명하다. 가장 먼저, 이 방법은 pose estimation 품질에 크게 의존한다. OCHuman 실험에서 predicted keypoint와 GT keypoint 사이의 차이가 매우 크다. 이는 Pose2Seg의 실제 성능이 pose detector의 성능 상한에 의해 좌우된다는 뜻이다. 즉 detection dependency를 줄였지만, 대신 pose dependency를 크게 갖게 되었다고 볼 수 있다.

둘째, 사람 클래스에 특화된 방법이므로 일반 객체로의 확장성은 제한적이다. 논문은 애초에 이를 일반 object segmentation 방법으로 주장하지 않는다. 따라서 장점이 곧 한계이기도 하다. 사람처럼 skeleton으로 표현 가능한 범주에는 적합하지만, 구조화된 pose 표현이 없는 객체에는 바로 적용하기 어렵다.

셋째, end-to-end라고 표현되지만 실제 전체 시스템은 여전히 keypoint detector라는 별도 기능에 기대고 있다. 본문상 입력이 “image and human pose”로 정의되기 때문에, segmentation module 관점에서는 pose가 외부 입력처럼 보인다. pose estimation과 segmentation을 완전히 하나의 unified objective로 공동 학습하는지, 혹은 training pipeline에서 어느 정도 결합되는지는 제공된 텍스트만으로는 충분히 명확하지 않다. 따라서 “detection-free”라는 표현은 맞지만, “auxiliary-free”는 아니다.

넷째, OCHuman은 validation/testing용으로 설계되어 training에는 사용하지 않는다. 이는 occlusion robustness 평가에는 합리적이지만, 실제로 occlusion 특화 학습을 수행했을 때 얼마나 더 좋아질지, 혹은 데이터 분포 차이 문제가 있는지는 이 논문만으로는 알기 어렵다.

다섯째, method complexity 측면에서도 pose template clustering, affine estimation, Skeleton feature 생성 등 추가 단계가 들어간다. 논문은 20 FPS를 제시하지만, 이 수치가 전체 시스템 배치와 어떤 조건에서 측정되었는지는 제한적으로만 설명된다. 따라서 실제 배포 환경에서의 효율성은 별도 검토가 필요하다.

비판적으로 보면, 이 논문은 “사람은 pose로 정의된다”는 가정을 매우 강하게 전제한다. 대부분의 경우 타당하지만, 극단적으로 관절이 적게 보이거나 pose estimator가 오작동하는 상황에서는 alignment 자체가 흔들릴 수 있다. 저자도 valid point가 부족할 때 예외 처리를 넣었지만, 이 경우 성능이 얼마나 유지되는지는 자세히 분석하지 않았다. 또한 occlusion이 심할수록 pose estimation도 어려워지는 경향이 있으므로, 가장 어려운 경우일수록 입력 신호 자체가 불완전해지는 딜레마가 있다.

## 6. 결론

이 논문은 human instance segmentation을 detection box가 아니라 pose를 중심으로 재구성한 Pose2Seg를 제안한다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, 사람 pose를 기준으로 RoI를 정렬하는 Affine-Align을 설계했다. 둘째, skeleton 정보를 PAF와 confidence map 형태로 명시적 feature로 넣어 segmentation을 유도했다. 셋째, 심한 가림 상황을 체계적으로 평가할 수 있는 OCHuman benchmark를 제안했다.

실험 결과는 이 설계가 단순한 아이디어 차원이 아니라 실제 성능 향상으로 이어짐을 보여 준다. 특히 OCHuman에서 Mask R-CNN 대비 뚜렷한 개선을 보였고, GT keypoint 사용 시 훨씬 높은 상한도 확인되었다. 이는 사람 segmentation에서 occlusion 문제가 단순히 detector를 더 좋게 만든다고 해결되는 것이 아니라, 인스턴스 표현 자체를 바꿔야 할 수 있음을 시사한다.

실제 적용 관점에서 이 연구는 군중 장면 분석, 감시 영상, 스포츠 분석, 멀티-인물 편집, 인간 중심 로보틱스 등에서 중요할 가능성이 크다. 향후 연구에서는 더 강한 pose estimator와 결합하거나, pose estimation과 segmentation을 더 tightly coupled된 방식으로 공동 최적화하는 방향이 자연스럽다. 또한 video setting이나 temporal consistency까지 확장하면 실제 응용 가치가 더욱 커질 수 있다. 전체적으로 이 논문은 “human instance segmentation은 사람의 구조를 이용해 풀어야 한다”는 명확한 메시지를 남긴다.
