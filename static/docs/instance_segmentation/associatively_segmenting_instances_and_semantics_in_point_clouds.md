# Associatively Segmenting Instances and Semantics in Point Clouds

이 논문은 3D point cloud에서 **instance segmentation**과 **semantic segmentation**을 따로 푸는 대신, 두 작업이 서로 도와주도록 결합한 **ASIS** 프레임워크를 제안합니다. 논문은 “서로 다른 class의 점들은 서로 다른 instance여야 하고, 같은 instance의 점들은 같은 semantic category여야 한다”는 상호 제약에 주목합니다. 이를 이용해 instance 쪽에는 semantic 정보를, semantic 쪽에는 instance 정보를 흘려보내는 방식으로 두 과제를 end-to-end로 함께 학습합니다. 저자들은 이 방식이 S3DIS에서 3D instance segmentation 성능을 크게 높이고, semantic segmentation도 함께 개선한다고 주장합니다.

## 1. Paper Overview

이 논문의 핵심 문제는 3D 실내 장면 point cloud를 해석할 때, 각 점이 **무엇인지(semantic label)**와 **어느 개체에 속하는지(instance id)**를 동시에 정확히 추론하는 것입니다. 기존 연구는 대체로 두 과제를 별도로 다뤘고, 특히 3D에서는 두 작업을 **연관적으로(associatively)** 푸는 시도가 거의 없었다고 논문은 설명합니다. 자율주행, AR, 실내 장면 이해처럼 실제 응용에서는 “벽/의자/테이블” 같은 semantic 정보와 “의자 1, 의자 2” 같은 instance 정보가 함께 필요하므로, 두 작업을 따로 푸는 것은 비효율적일 수 있습니다.

저자들은 단순한 step-wise 방식도 검토합니다. 예를 들어 semantic segmentation을 먼저 한 뒤 class별로 instance를 나누거나, 반대로 instance를 먼저 찾은 뒤 각 instance를 분류하는 접근입니다. 하지만 이런 구조는 앞단의 오류가 뒷단으로 그대로 전파되는 문제가 있습니다. 논문은 이를 피하기 위해, 두 작업을 병렬 분기 구조에서 같이 학습시키되 서로의 정보를 soft하게 주고받는 방향으로 설계합니다.

## 2. Core Idea

이 논문의 핵심 아이디어는 다음 두 가지입니다.

첫째, **semantic-aware instance embedding**입니다. instance segmentation branch는 점마다 embedding을 출력하는데, 여기에 semantic branch의 특징을 섞어 넣어 서로 다른 semantic class의 점들이 embedding 공간에서도 더 잘 분리되도록 만듭니다. 직관적으로 chair와 table은 애초에 다른 class이므로, instance embedding에서도 더 멀어지는 것이 자연스럽다는 생각입니다.

둘째, **instance-fused semantic segmentation**입니다. semantic branch는 각 점의 category를 예측하지만, 같은 instance에 속한 점들은 본질적으로 같은 semantic class를 가져야 합니다. 그래서 instance embedding으로 찾은 같은 instance 내의 점들로부터 semantic feature를 모아 각 점의 semantic feature를 보강합니다. 즉, semantic은 instance로부터 “같은 물체 내부의 일관성”을 얻고, instance는 semantic으로부터 “서로 다른 class 경계”를 더 선명하게 얻습니다. 이것이 논문 제목의 “Associatively”가 의미하는 바입니다.

이 아이디어의 신선함은, 단순한 multi-task learning이 아니라 두 태스크가 서로의 구조적 제약을 명시적으로 이용한다는 점에 있습니다. 논문은 이를 통해 “win-win situation”이 만들어진다고 주장합니다.

## 3. Detailed Method Explanation

### 3.1 전체 파이프라인

기본 구조는 **shared encoder + 두 개의 parallel decoder**입니다. 하나의 decoder는 semantic segmentation을, 다른 decoder는 instance segmentation을 담당합니다. shared encoder는 PointNet 또는 PointNet++ 같은 backbone으로 point cloud를 특징 행렬로 변환합니다. 이후 semantic branch는 semantic feature $F_{\rm SEM}$ 을 만들고 point별 semantic prediction $P_{\rm SEM}$ 을 출력합니다. instance branch는 instance feature $F_{\rm INS}$ 를 만들고, point별 instance embedding $E_{\rm INS}$ 를 출력합니다. 이 embedding은 같은 instance 점끼리는 가깝고, 다른 instance 점끼리는 멀어지도록 학습됩니다.

### 3.2 Baseline: point-level embedding 기반 instance segmentation

논문의 baseline은 2D 이미지에서 associative embedding이나 discriminative loss를 쓰던 접근을 3D point cloud에 맞게 단순화한 형태입니다. 중요한 점은 저자들이 **class-agnostic instance embedding**을 채택했다는 것입니다. 기존 일부 방식은 semantic class별로 embedding을 따로 학습하는데, 그러면 semantic prediction이 틀렸을 때 instance segmentation도 함께 무너집니다. 이 논문은 이를 피하기 위해, embedding이 class 정보에 직접 의존하지 않고 오로지 instance 구분만 담당하게 합니다.

학습 시 semantic branch는 일반적인 cross-entropy loss를 사용합니다. instance branch는 discriminative loss를 사용하며, 총 loss는 다음과 같습니다.

$$
L = L_{var} + L_{dist} + \alpha L_{reg}
$$

여기서:

* $L_{var}$: 같은 instance 내부 점 embedding을 해당 instance 중심 $\mu_i$ 근처로 모읍니다.
* $L_{dist}$: 서로 다른 instance 중심끼리는 충분히 멀어지게 만듭니다.
* $L_{reg}$: embedding 값이 너무 커지지 않도록 regularization을 겁니다.
* 논문에서는 $\alpha = 0.001$ 로 둡니다.

즉, 이 baseline 자체만으로도 point-level embedding을 통해 instance grouping을 수행할 수 있고, 논문은 이 baseline만으로도 SGPN보다 더 좋고 더 빠르다고 주장합니다.

### 3.3 ASIS 모듈: 두 작업을 연결하는 방식

ASIS는 baseline 위에 두 개의 결합 장치를 추가합니다.

#### (1) Semantics Awareness for Instance Segmentation

semantic branch에서 나온 feature를 instance branch에 주입하여, instance embedding이 semantic class 경계를 더 잘 반영하도록 만듭니다. 예를 들어 chair와 table이 가까이 있어도 서로 다른 class라는 정보가 embedding 공간의 분리에 도움을 줍니다. 논문은 Figure 2에서 baseline 대비 ASIS가 embedding 경계를 더 선명하게 만든다고 설명합니다.

#### (2) Instance Fusion for Semantic Segmentation

instance embedding을 기반으로 같은 instance에 속한다고 판단되는 이웃 점들을 모아 semantic feature를 보강합니다. 논문 직관은 매우 명확합니다. 한 점이 어떤 class인지 판단할 때, 사실 그 점이 속한 **물체 전체**의 정체성이 중요합니다. 따라서 같은 instance 내의 다른 점들의 semantic 정보까지 합치면 point-wise semantic prediction이 더 안정적이 됩니다.

이 설계는 결국 양방향입니다.
semantic → instance: 다른 class 경계를 더 잘 나누게 함
instance → semantic: 같은 물체 내부 semantic 일관성을 높임

### 3.4 추론 과정

추론 시 network가 각 점의 semantic label과 instance embedding을 출력한 뒤, instance branch의 embedding에 대해 **mean-shift clustering**을 적용해 같은 instance를 묶습니다. S3DIS처럼 블록 단위로 처리되는 데이터에서는 서로 다른 블록에서 나온 instance를 이어 붙이기 위해 SGPN의 BlockMerging 알고리즘도 사용합니다. 테스트 시 mean-shift bandwidth는 0.6으로 설정합니다.

## 4. Experiments and Findings

### 4.1 실험 설정과 평가 포인트

논문은 주로 두 데이터셋을 사용합니다.

* **S3DIS**: 실내 장면 point cloud에서 instance/semantic segmentation 평가
* **ShapeNet**: part segmentation 및 part instance 관찰

backbone으로는 PointNet과 PointNet++를 사용해 방법의 일반성을 보이려 합니다. 이는 특정 backbone에만 맞춘 기법이 아니라는 메시지를 줍니다.

### 4.2 S3DIS: baseline 자체도 강하고, ASIS는 더 강함

논문은 baseline만으로도 기존 SOTA였던 **SGPN**보다 좋은 결과를 낸다고 보고합니다. 예를 들어 PointNet backbone 기준 6-fold cross validation에서 baseline은 **46.3 mWCov**를 얻어, SGPN 대비 **5.5 포인트 향상**을 보였다고 설명합니다. 그리고 이러한 우위가 여러 instance segmentation metric 전반에서 일관된다고 말합니다.

그 위에 ASIS를 얹으면 semantic segmentation과 instance segmentation이 함께 더 좋아집니다. 저자들은 “semantic-aware instance segmentation”과 “instance-fused semantic segmentation”이 동시에 기여한다고 주장합니다. 특히 논문 본문과 Figure 2 설명에 따르면, ASIS는 embedding 공간에서 서로 다른 class 경계를 더 뚜렷하게 만들어 instance 분리에 도움을 줍니다.

### 4.3 ShapeNet: part segmentation에도 효과

ShapeNet에서는 “진짜” instance GT가 아니라 SGPN이 생성한 annotation을 사용했기 때문에, part instance segmentation은 정량보다는 정성 평가 중심입니다. 논문은 자동차 바퀴, 의자 다리 같은 파트들이 ASIS에 의해 잘 분리된다고 보여줍니다. semantic segmentation 쪽은 정량 평가가 있으며, PointNet backbone에서는 **0.6 포인트**, PointNet++에서는 **0.7 mIoU** 향상을 보였다고 보고합니다. Table 6에서는 ShapeNet semantic segmentation에서 PointNet이 83.7, ASIS(PN)가 84.0, 재현 PointNet++가 84.3, ASIS(PN++)가 85.0으로 제시됩니다.

이는 ASIS의 아이디어가 단지 실내 scene instance segmentation에만 국한되지 않고, part-level segmentation에서도 유효하다는 근거로 해석할 수 있습니다.

### 4.4 계산 효율성

논문은 효율성도 강조합니다. S3DIS 실험에서 baseline은 single GPU에서 scratch부터 약 **4~5시간** 정도 학습된다고 서술합니다. 또한 inference를 network inference와 grouping으로 나누어 비교했을 때, ASIS의 mean-shift clustering은 SGPN의 GroupMerge보다 grouping 단계가 훨씬 빠르다고 주장합니다. 즉, 성능뿐 아니라 실용성 측면에서도 경쟁력이 있다는 메시지입니다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **문제 구조를 아주 자연스럽게 이용했다는 점**입니다. semantic과 instance는 서로 충돌하는 면도 있지만, 동시에 강한 제약도 공유합니다. 논문은 이 관계를 단순한 직관으로 끝내지 않고 feature fusion과 embedding 기반 grouping으로 구체화했습니다. 덕분에 설명이 깔끔하고, 왜 성능이 좋아질 수 있는지 설득력이 있습니다.

또 하나의 강점은 **단순성**입니다. 복잡한 proposal 생성이나 pairwise similarity matrix 전체를 예측하는 대신, point embedding + clustering이라는 비교적 단순한 구조를 사용합니다. 그래서 SGPN 대비 빠르고 구현도 단순하다는 장점이 있습니다.

마지막으로, PointNet과 PointNet++ 모두에서 이득을 보였다는 점은 방법의 **backbone-agnostic 성격**을 어느 정도 뒷받침합니다.

### 한계

논문이 직접 보여주는 실패 사례도 있습니다. 가까이 붙어 있는 같은 class의 두 chair가 하나의 instance로 잘못 합쳐지는 경우가 제시됩니다. 즉, semantic 정보는 class 간 경계를 선명하게 해 주지만, **같은 class 내부의 인접 instance 분리**는 여전히 어려운 문제로 남아 있습니다. 저자들도 이런 문제는 future work로 남긴다고 적습니다.

또한 ShapeNet의 part instance 평가는 진짜 GT가 아니라 생성된 annotation에 기반하므로, 이 부분의 결론은 상대적으로 약합니다. 즉, “part instance segmentation에도 유효하다”는 주장은 정성적 근거가 더 많고, 엄밀한 정량 검증은 제한적입니다.

### 해석

이 논문은 이후의 panoptic segmentation 관점에서도 의미가 큽니다. 실제로 저자들도 이 방법이 panoptic segmentation 같은 통합 과제로 확장될 수 있다고 언급합니다. 지금 시점에서 보면, semantic과 instance를 따로 최적화하지 않고 서로의 구조를 이용하는 발상은 이후 2D/3D panoptic 계열 연구들과도 잘 이어지는 방향입니다. 다만 이 논문 자체는 proposal-free embedding 기반 접근에 가깝기 때문에, 대규모 outdoor LiDAR나 매우 복잡한 장면에서의 scalability는 본문만으로는 충분히 검증되었다고 보기 어렵습니다. 이는 논문이 직접 다루지 않은 부분입니다.

## 6. Conclusion

이 논문은 3D point cloud에서 instance segmentation과 semantic segmentation을 **서로 독립된 과제**가 아니라 **상호 보완적인 과제**로 보고, 이를 실제 네트워크 설계에 녹인 ASIS를 제안합니다. baseline은 shared encoder와 두 개의 병렬 branch로 구성되고, ASIS는 여기에 semantic-aware instance embedding과 instance-fused semantic feature를 추가합니다. 결과적으로 S3DIS에서 instance segmentation SOTA를 넘기고 semantic segmentation도 개선했으며, ShapeNet part segmentation에도 긍정적 효과를 보였습니다.

실무적으로는 “장면을 class와 object 단위로 동시에 이해해야 하는 3D perception”에 잘 맞는 아이디어입니다. 연구적으로는 multi-task learning을 넘어, **task 간 구조적 일관성(structural consistency)**을 이용하는 설계의 좋은 예시라고 볼 수 있습니다. 특히 후속 3D panoptic segmentation, joint embedding, cross-task feature interaction 연구를 이해할 때 중요한 출발점이 되는 논문입니다.
