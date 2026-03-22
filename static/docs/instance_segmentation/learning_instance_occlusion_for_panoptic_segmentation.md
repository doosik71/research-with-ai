# Learning Instance Occlusion for Panoptic Segmentation

* **저자**: Justin Lazarow, Kwonjoon Lee, Kunyu Shi, Zhuowen Tu
* **발표연도**: 2019
* **arXiv**: <https://arxiv.org/abs/1906.05896>

## 1. 논문 개요

이 논문은 **panoptic segmentation**에서 발생하는 인스턴스 간 가림(occlusion) 문제를 명시적으로 다룬다. Panoptic segmentation은 이미지의 모든 픽셀에 대해, 셀 수 있는 객체인 **things**는 개별 인스턴스로, 배경성 영역인 **stuff**는 의미 클래스 단위로 동시에 예측해야 한다. 일반적으로는 instance segmentation 결과와 semantic segmentation 결과를 합쳐 하나의 non-overlapping 출력으로 만드는 **fusion** 단계가 필요하다.

문제는 이 fusion 과정에서 서로 겹치는 instance mask들의 우선순위를 정해야 한다는 점이다. 기존 방법은 대체로 detection confidence가 높은 인스턴스를 먼저 배치하는 greedy 전략을 사용한다. 그러나 논문은 이 점을 핵심적으로 비판한다. **검출 confidence는 실제 가림 순서와 잘 맞지 않는다.** 예를 들어 넥타이처럼 작지만 앞에 있어야 하는 객체가, 더 큰 사람 객체보다 confidence가 낮다면 기존 방식은 넥타이를 사라지게 만들 수 있다. 즉, 인스턴스 분할 자체는 성공했더라도 panoptic fusion에서 많은 진짜 인스턴스가 손실된다.

이 논문의 목표는 이 약점을 해결하기 위해, 두 인스턴스 mask가 겹칠 때 **어느 쪽이 위에 와야 하는지**를 예측하는 별도의 lightweight 모듈을 추가하는 것이다. 저자들은 이를 **occlusion head**라 부르고, 전체 방법을 **OCFusion**이라 명명한다. 핵심은 panoptic segmentation의 backbone 전체를 새로 설계하는 것이 아니라, 기존 Panoptic FPN + Mask R-CNN 파이프라인에 작은 분기만 추가하여 fusion을 더 똑똑하게 만드는 데 있다.

이 문제가 중요한 이유는 panoptic segmentation의 성능이 단순히 분할 branch의 정확도뿐 아니라, **최종적으로 서로 충돌하는 예측들을 어떻게 병합하느냐**에 크게 좌우되기 때문이다. 따라서 이 논문은 “더 좋은 분할기”를 만드는 대신, “더 좋은 병합 규칙”을 학습시키는 접근을 취한다는 점에서 의미가 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 명확하다. **겹치는 두 instance mask 사이의 occlusion ordering을 이진 분류 문제로 학습하자**는 것이다. 즉, mask $M_i$와 $M_j$가 충분히 많이 겹칠 때, 교차 영역을 어느 인스턴스에 할당해야 하는지를 confidence 기반의 휴리스틱이 아니라 학습된 관계 함수로 결정한다.

이를 위해 저자들은 다음과 같은 binary relation을 정의한다.

$$
\text{Occlude}(M_i, M_j) =
\begin{cases}
1 & \text{if } M_i \text{ should be placed on top of } M_j \
0 & \text{if } M_j \text{ should be placed on top of } M_i
\end{cases}
$$

이 정의의 장점은 다음과 같다.

첫째, 기존 방법처럼 전체 인스턴스를 일렬로 정렬하는 것이 아니라, **겹치는 두 mask에 대해 국소적으로 관계를 질의(query)** 할 수 있다. 따라서 한 객체가 모든 경우에 항상 앞이거나 뒤가 아니라, 실제 겹침 상황에 따라 더 유연하게 동작한다.

둘째, 이 방법은 **class-level ordering**이 아니라 **instance-level ordering**을 다룬다. 즉, “사람 클래스가 자동차 클래스보다 앞” 같은 조잡한 규칙이 아니라, “이 사람 A와 저 사람 B 중 누가 앞인가”처럼 같은 클래스 내부의 가림 관계까지 다룰 수 있다. 논문은 이것이 특히 Cityscapes 같은 장면에서 중요하다고 강조한다.

셋째, ground truth occlusion supervision을 새로 수집하지 않고도, 기존 panoptic annotation과 instance annotation으로부터 자동 생성할 수 있다. 즉, panoptic ground truth에서 교집합 영역의 다수 픽셀이 어느 인스턴스 소유인지를 기준으로 occlusion label을 만들기 때문에 추가 어노테이션 비용이 거의 없다.

결국 이 논문의 차별점은 backbone이나 segmentation head를 크게 바꾸지 않고도, **fusion 단계에 명시적 occlusion reasoning을 삽입**했다는 데 있다. 이 점이 lightweight하면서도 실제 PQ 향상으로 이어진다.

## 3. 상세 방법 설명

전체 구조는 **공유 FPN backbone** 위에 두 개의 큰 분기를 두는 전형적 Panoptic FPN 구조를 따른다. 하나는 semantic segmentation을 위한 **stuff branch**, 다른 하나는 Mask R-CNN 기반의 **thing branch**이다. OCFusion은 이 중 thing branch에 **occlusion head**를 추가한다.

### 3.1 기존 fusion by confidence

기존 panoptic fusion은 inference 시 instance proposal들을 detection confidence 내림차순으로 정렬한 뒤, 순서대로 panoptic mask에 붙이는 방식이다. 이미 배정된 픽셀과 많이 겹치면 버리고, 그렇지 않으면 아직 비어 있는 픽셀을 현재 instance에 할당한다. 이후 남은 빈 픽셀에 대해 semantic segmentation의 stuff를 채운다.

이 과정에서 현재 proposal $M_i$에 대해 실제로 남아 있는 픽셀 집합을 다음처럼 둘 수 있다.

$$
C_i = M_i - P
$$

여기서 $P$는 이미 panoptic output에 배정된 픽셀들의 집합이다. 기존 방법은 이 $C_i$가 너무 작아지면 해당 instance를 버린다. 문제는 이 과정이 오직 detection confidence에만 의존한다는 것이다. 그래서 큰 객체가 높은 confidence를 가질 경우, 실제로는 뒤에 있어야 해도 먼저 전체 영역을 차지해 버릴 수 있다.

### 3.2 appreciable overlap의 정의

저자들은 모든 mask pair를 다 비교하지 않고, **충분히 의미 있게 겹치는 pair만** occlusion head의 판단 대상으로 삼는다. 두 mask $M_i$, $M_j$의 교집합을

$$
I_{ij} = M_i \cap M_j
$$

라 두고, 각 마스크 기준의 교집합 비율을

$$
R_i = \frac{\text{Area}(I_{ij})}{\text{Area}(M_i)}, \qquad
R_j = \frac{\text{Area}(I_{ij})}{\text{Area}(M_j)}
$$

로 정의한다. 그리고

$$
R_i \ge \rho \quad \text{or} \quad R_j \ge \rho
$$

이면 두 mask가 **appreciable overlap**을 가진다고 본다. 즉, 단순히 한두 픽셀이 닿는 정도가 아니라, 적어도 한쪽 mask 기준으로는 꽤 의미 있는 비율만큼 겹쳐야 occlusion 문제로 간주한다.

이 $\rho$는 매우 중요한 hyperparameter다. 너무 작으면 노이즈성 겹침까지 다 들어오고, 너무 크면 학습할 occlusion 사례 수가 크게 줄어든다.

### 3.3 fusion with occlusion head

논문의 핵심은 기존 greedy fusion을 완전히 버리는 것이 아니라, **기존 fusion을 완화(soften)** 하는 방식으로 확장했다는 점이다.

현재 처리 중인 mask $M_i$와 이미 합쳐진 이전 mask $M_j$가 appreciable overlap을 가진다면, occlusion head에 $\text{Occlude}(M_i, M_j)$를 질의한다. 만약 결과가 1, 즉 $M_i$가 $M_j$ 위에 있어야 한다면, 현재 mask는 이전에 이미 차지된 교차 영역 일부를 다시 가져올 수 있다.

논문 알고리즘을 따라 쓰면, 현재 유효 영역은 처음에

$$
C_i = M_i - P
$$

로 시작한다. 이후 각 이전 인스턴스 $M_j$에 대해, 만약 appreciable overlap이 있고

$$
\text{Occlude}(M_i, M_j)=1
$$

이면 다음과 같이 업데이트한다.

$$
C_i = C_i \cup (C_j \cap I_{ij})
$$

$$
C_j = C_j - I_{ij}
$$

의미를 쉽게 설명하면 이렇다. 기존 방식에서는 먼저 들어간 $M_j$가 교차 영역을 모두 선점한다. 하지만 occlusion head가 “사실은 $M_i$가 앞이다”라고 판단하면, $M_j$가 차지했던 교차 영역 중 해당 부분을 다시 $M_i$에게 되돌린다. 즉, 나중에 처리되는 인스턴스라도 앞에 있는 객체라면 일부 픽셀을 reclaim할 수 있다.

그 뒤 최종적으로 남은 비율이 너무 적으면 버린다.

$$
\frac{\text{Area}(C_i)}{\text{Area}(M_i)} \le \tau
$$

이면 skip하고, 아니면 $C_i$를 panoptic output에 반영한다. 여기서 $\tau$는 fusion 단계에서 너무 많이 잘려 거의 의미 없는 segment를 제거하기 위한 threshold다.

중요한 점은 이 방식이 여전히 greedy order를 기본 틀로 유지하지만, 그 한계를 **pairwise occlusion prediction**으로 보정한다는 것이다. 그래서 기존 시스템에 넣기 쉽고 계산량도 통제 가능하다.

### 3.4 Occlusion head architecture

Occlusion head는 Mask R-CNN 내부의 추가 head로 구현된다. Mask R-CNN에는 원래 **box head**와 **mask head**가 있는데, 여기에 세 번째 분기인 occlusion head를 더한다.

입력은 두 instance mask $M_i$, $M_j$와 각각의 bounding box에 대응하는 FPN의 RoI feature이다. 논문에 따르면 각 soft mask는 max pooling을 거쳐 $14 \times 14$ 표현으로 바뀌고, 이것을 대응하는 RoI feature와 concatenate하여 head의 입력으로 사용한다.

이후 구조는 다음과 같다.

* $3 \times 3$ convolution 3개, 각 512 채널, stride 1
* 마지막 $3 \times 3$ convolution 1개, stride 2
* flatten
* 1024차원 fully connected layer
* single logit 출력

즉, 최종적으로는 한 쌍의 mask에 대해 “첫 번째가 위인가, 두 번째가 위인가”를 판정하는 **binary classifier**다. 복잡한 graph reasoning이나 transformer 같은 구조는 전혀 쓰지 않는다. 논문이 lightweight라고 주장하는 근거가 여기에 있다.

### 3.5 Ground truth occlusion 생성

이 논문에서 supervision을 만드는 방식은 실용적이다. 저자들은 panoptic ground truth와 instance ground truth를 함께 사용해, 각 이미지에 대해 인스턴스 쌍별 occlusion 관계를 담은 **occlusion matrix**를 미리 계산한다.

절차는 다음과 같다.

먼저 appreciable overlap을 가지는 모든 ground truth 인스턴스 쌍을 찾는다. 그다음 두 마스크의 교집합 픽셀을 panoptic ground truth에서 조회한다. 이 교집합 영역에서 **다수 픽셀을 소유한 인스턴스**를 “위에 있는 인스턴스”로 간주한다. 이렇게 하면 각 pair에 대해 0 또는 1의 label을 만들 수 있다. occlusion이 없거나 의미 없는 경우는 $-1$ 등 별도 값으로 저장한다.

이 방식은 annotation consistency에 의존하지만, 적어도 논문이 제시한 설정 안에서는 새 레이블링 비용 없이 학습 데이터를 자동 구성할 수 있다는 장점이 있다.

### 3.6 학습 절차와 손실 함수

저자들은 occlusion head를 처음부터 end-to-end로 강하게 결합해 학습하지 않고, **기존 panoptic model을 학습한 뒤 나머지 파라미터를 freeze하고 occlusion head만 fine-tuning** 하는 방식을 사용했다. 논문에 따르면 이것이 가장 안정적이고 효율적이었다.

원래 panoptic training의 총 손실은 다음과 같다.

$$
L = \lambda_i (L_c + L_b + L_m) + \lambda_s L_s
$$

여기서

* $L_c$는 box classification loss
* $L_b$는 bounding box regression loss
* $L_m$는 mask loss
* $L_s$는 semantic segmentation cross-entropy loss

이다.

fine-tuning 단계에서는 occlusion head의 loss $L_o$만 최소화한다. 이 $L_o$는 binary classification이므로 **binary cross-entropy loss**이다.

학습 샘플 구성도 중요하다. 예측된 mask 중 서로 다른 ground truth instance에 매칭되는 pair를 찾고, appreciable overlap 조건을 만족하는 쌍만 사용한다. 또한 consistency를 위해 pair를 뒤집은 순서도 함께 넣는다. 즉,

$$
\text{Occlude}(M_i, M_j)=0 \iff \text{Occlude}(M_j, M_i)=1
$$

관계를 명시적으로 반영한다. 이는 데이터 수를 늘리고 class imbalance를 완화하는 역할도 한다. 논문에서는 이미지당 128개의 mask occlusion pair를 subsample했다고 적고 있다.

## 4. 실험 및 결과

### 4.1 실험 설정

논문은 COCO와 Cityscapes panoptic benchmark에서 실험한다.

COCO는 80개의 things와 53개의 stuff 클래스를 사용하고, 2017 split 기준으로 학습/검증/테스트가 각각 118k / 5k / 20k이다. Cityscapes는 8개의 things와 11개의 stuff 클래스로 구성되며, fine annotation만 사용하고 split은 2975 / 500 / 1525이다.

평가지표는 panoptic segmentation 표준 지표인 **PQ (Panoptic Quality)** 이다. 그리고 이를 things 성능인 **PQTh**, stuff 성능인 **PQSt**로 분해하여 함께 제시한다. 이 논문은 방법의 효과가 주로 things 쪽, 즉 인스턴스 fusion 관련 부분에서 나타날 것이라고 예상할 수 있는데, 실제 결과도 그렇게 나온다.

### 4.2 COCO에서의 결과

자체 구현한 Panoptic FPN baseline과 비교한 Table 1을 보면, OCFusion은 COCO validation에서 일관되게 성능을 향상시킨다.

ResNet-50 backbone에서는:

* Baseline: PQ 39.5, PQTh 46.5, PQSt 29.0
* OCFusion: PQ 41.3, PQTh 49.4, PQSt 29.0

즉, PQ는 **+1.8**, PQTh는 **+3.0** 향상했고, PQSt는 변화가 없다.

ResNet-101 backbone에서는:

* Baseline: PQ 41.0, PQTh 47.9, PQSt 30.7
* OCFusion: PQ 43.0, PQTh 51.1, PQSt 30.7

즉, PQ는 **+2.0**, PQTh는 **+3.2**, PQSt는 변화가 없다.

이 결과는 매우 해석이 쉽다. 이 방법은 stuff branch를 개선하는 방법이 아니라 **instance fusion과 occlusion ordering**을 개선하는 방법이므로, 이득이 거의 전부 **PQTh**에서 나온다. 실제로 PQSt가 거의 그대로라는 점은 이 방법이 목표한 문제를 정확히 건드리고 있음을 보여준다.

더 넓게 prior work와 비교한 Table 2에서는 COCO validation에서 OCFusion이 강한 성능을 보인다. 특히 ResNeXt-101 + deformable convolution + multi-scale testing 설정에서 **PQ 46.3, PQTh 53.5, PQSt 35.4**를 달성했다. Table 3의 COCO test-dev에서도 **PQ 46.7**로 당시 단일 모델 SOTA를 주장한다.

다만 비교 시 주의할 점도 있다. 일부 방법은 backbone, deformable convolution 사용 여부, multi-scale testing 적용 여부가 다르므로 절대적 수치 비교는 조심해야 한다. 그래도 같은 계열 backbone 기준에서 OCFusion이 일관되게 strong baseline을 개선한다는 메시지는 충분히 설득력 있다.

### 4.3 Cityscapes에서의 결과

Cityscapes에서도 유사한 패턴이 나온다. Table 4에서 baseline 대비:

* Baseline: PQ 58.6, PQTh 51.7, PQSt 63.6
* OCFusion: PQ 59.3, PQTh 53.5, PQSt 63.6

즉, PQ는 **+0.7**, PQTh는 **+1.7**, PQSt는 변화가 없다.

COCO보다 절대 향상 폭은 작지만, 여전히 things 관련 지표가 개선되고 stuff는 그대로다. 이는 방법의 작동 원리와 잘 맞는다. Table 5에서는 Cityscapes validation에서 multi-scale testing 시 **PQ 60.2, PQTh 54.0, PQSt 64.7**을 기록해 경쟁력 있는 결과를 보인다. 특히 ResNet-50 기반 모델 중에서는 좋은 성능이라고 저자들이 주장한다.

### 4.4 Occlusion head 자체의 정확도

논문은 occlusion head가 정말 올바른 ordering을 학습했는지 별도로 측정한다. ResNet-50 backbone, $\rho=0.2$에서 ground truth boxes와 masks를 사용하여 true ordering 예측 정확도를 본 결과:

* COCO: **91.58%**
* Cityscapes: **93.60%**

의 분류 정확도를 보고한다.

이 수치는 “occlusion head가 엉뚱한 noise classifier가 아니라 실제로 의미 있는 ordering을 배운다”는 근거가 된다. 다만 이 평가는 ground truth boxes/masks 기반이며, 실제 end-to-end panoptic output의 모든 오차를 반영하는 수치는 아니라는 점은 염두에 둘 필요가 있다.

### 4.5 추론 시간 오버헤드

이 방법은 pairwise mask 비교를 하기 때문에 최악의 경우 $O(n^2)$ 비용이 든다. 여기서 $n$은 고려 대상 인스턴스 수다. 하지만 저자들은 실제로는 confidence threshold로 먼저 필터링하고, 그 뒤 appreciable overlap을 가진 pair만 occlusion head에 넣기 때문에 실용적 비용은 크지 않다고 주장한다.

Table 6에 따르면 이미지당 평균 runtime은:

* COCO: 153 ms → 156 ms, **+3 ms**
* Cityscapes: 378 ms → 396 ms, **+18 ms**

즉, 오버헤드는 각각 **2.0%**, **4.7%** 수준이다. 이 정도면 성능 향상 대비 충분히 감수 가능한 비용으로 보인다.

### 4.6 Ablation 분석

하이퍼파라미터 $\tau$와 $\rho$에 대한 민감도 분석도 제공된다. COCO의 Table 7을 보면 전체적으로 PQ 값이 큰 폭으로 흔들리지는 않는다. 이는 방법이 특정 threshold에 극단적으로 민감하지 않음을 시사한다. Cityscapes의 Table 8에서도 유사하지만, 일부 조합에서는 성능이 떨어진다. 예를 들어 $\tau=0.6$, $\rho=0.2$일 때 PQ가 58.70으로 감소한다. 이는 overlap 기준을 너무 빡빡하게 잡으면 학습할 occlusion 예제가 지나치게 줄어들 수 있음을 보여준다.

논문은 또한 각 $\rho$에서 확보되는 occlusion training sample 수 $N$도 보여준다. 예를 들어 COCO에서는 $\rho=0.05$일 때 192,519개, $\rho=0.20$일 때 132,165개다. Cityscapes는 더 적어서 $\rho=0.20$일 때 6,617개까지 감소한다. 즉, $\rho$는 “더 깨끗한 학습 데이터”와 “충분한 데이터 양” 사이의 trade-off를 만든다.

### 4.7 Intra-class occlusion의 중요성

논문이 강조하는 중요한 차별점 중 하나는 **intra-class occlusion**까지 다룰 수 있다는 점이다. Cityscapes Table 9에서:

* 기본 baseline: PQ 58.6, PQTh 51.7
* inter-class만 활성화: PQ 59.2, PQTh 53.0
* inter-class + intra-class 모두 활성화: PQ 59.3, PQTh 53.5

즉, intra-class handling을 추가하면 PQTh가 추가로 개선된다. 이는 같은 클래스끼리의 가림, 예를 들어 자동차와 자동차 사이의 occlusion이 실제로 중요하다는 것을 보여준다. class 간 우선순위만 학습하는 방식보다 instance-level relation이 더 일반적이고 강력하다는 저자들의 주장을 뒷받침한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의가 정확하다는 점이다. panoptic segmentation의 병목 중 하나인 **instance fusion의 ordering 오류**를 분명히 짚었고, 이를 segmentation backbone을 바꾸는 대신 **pairwise occlusion reasoning**으로 해결했다. 많은 논문이 더 큰 네트워크를 제안하지만, 이 논문은 작은 head 하나로 실제 PQ를 끌어올린다. 특히 PQTh만 선택적으로 좋아지고 PQSt는 거의 그대로라는 점은, 방법이 의도한 문제를 정확히 해결했음을 보여주는 깔끔한 증거다.

또 다른 강점은 **annotation 효율성**이다. ground truth occlusion을 별도 수작업 없이 기존 panoptic/instance annotation에서 자동 생성한다. 실전 연구에서 매우 중요한 장점이다.

또한 이 방법은 기존 Panoptic FPN/Mask R-CNN 기반 시스템 위에 쉽게 추가할 수 있다. 성능 향상 대비 runtime 오버헤드가 작고, training도 몇 천 iteration의 fine-tuning만으로 가능하다고 보고한다. 즉, 연구적 참신성뿐 아니라 engineering 관점에서도 설득력이 있다.

반면 한계도 분명하다.

첫째, 이 방법은 여전히 **기존 greedy fusion 프레임워크 안에서 동작하는 보정 장치**다. 즉, 전체 panoptic inference를 end-to-end differentiable하게 재구성한 것은 아니다. ordering을 pairwise로 보정할 뿐, 전역적으로 최적인 배치를 보장하지는 않는다.

둘째, occlusion label 생성 방식은 교집합 영역의 다수 픽셀 소유권에 의존한다. 이는 annotation 자체가 완전히 일관적이라는 가정에 기대며, 경계가 애매한 경우 label noise가 들어갈 수 있다. 논문은 이를 큰 문제로 다루지 않지만, 실제로는 얇은 물체나 annotation ambiguity가 있는 경우 binary ordering으로 단순화하는 데 한계가 있을 수 있다.

셋째, 계산량은 실용적으로는 작지만 이론적으로는 여전히 $O(n^2)$ pairwise 비교 구조다. crowded scene에서 인스턴스 수가 늘면 부담이 커질 가능성이 있다. 논문은 confidence threshold와 overlap threshold로 이를 통제한다고 설명하지만, 더 밀집된 장면이나 더 많은 proposal을 유지하는 시스템에서는 비용이 커질 수 있다.

넷째, 이 방법은 **stuff와 thing 사이의 더 일반적인 occlusion 관계**, 또는 amodal reasoning까지는 다루지 않는다. 결론에서도 저자들은 앞으로 stuff의 관계까지 확장해 보겠다고 말한다. 다시 말해, 현재 방법은 “instance-instance occlusion”에 초점을 맞춘 부분 해결책이다.

비판적으로 보면, 성능 향상은 분명하지만 대부분 **PQTh 개선**으로 제한되고, absolute gain이 아주 극적이진 않다. 따라서 이 방법은 panoptic segmentation 전체 패러다임을 바꾸는 수준이라기보다, 강한 baseline 위에 얹는 실용적 개선 모듈로 보는 것이 적절하다. 그럼에도 문제 설정, 구현 용이성, 일관된 실험 결과를 고려하면 학술적으로 충분히 가치 있는 기여다.

## 6. 결론

이 논문은 panoptic segmentation에서 기존 confidence-based fusion이 가지는 구조적 한계를 지적하고, 이를 해결하기 위해 **instance pair 사이의 occlusion ordering을 학습하는 occlusion head**를 제안했다. 방법은 Mask R-CNN 위에 binary classifier 형태의 head를 추가하고, 겹치는 두 mask 중 어느 것이 앞에 와야 하는지를 예측하여 fusion 과정에서 픽셀 소유권을 수정한다.

주요 기여는 세 가지로 요약할 수 있다. 첫째, panoptic fusion에서 occlusion ordering을 명시적으로 모델링했다. 둘째, 기존 annotation으로부터 ground truth occlusion supervision을 자동 생성하는 학습 절차를 제안했다. 셋째, 이 단순한 추가만으로 COCO와 Cityscapes에서 일관된 PQ, 특히 PQTh 향상을 달성했다.

실제 적용 측면에서도 의미가 있다. 이 연구는 “좋은 segmentation output을 만드는 것” 못지않게 “겹치는 예측을 어떻게 합리적으로 합칠 것인가”가 중요하다는 점을 보여준다. 이후 panoptic segmentation, scene parsing, instance relation modeling, even layered scene understanding 같은 방향으로 확장될 수 있는 아이디어다. 특히 같은 클래스 내부까지 포함한 instance-level occlusion reasoning은 자율주행이나 복잡한 crowded scene 이해에서 여전히 중요한 주제로 남아 있다.

전체적으로 이 논문은 거대한 구조 혁신보다는, panoptic segmentation의 실제 약점을 정확히 겨냥한 **작고 효과적인 개선**을 제시한 연구라고 평가할 수 있다.
