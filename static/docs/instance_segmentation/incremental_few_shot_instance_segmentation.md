# Incremental Few-Shot Instance Segmentation

이 논문은 few-shot instance segmentation(FSIS)에서 **새로운 클래스를 유연하게 추가할 수 없고**, 테스트 시 각 클래스의 support example 이미지를 계속 들고 있어야 하는 기존 방법들의 한계를 정면으로 다룹니다. 저자들은 이를 해결하기 위해 **증분적(incremental) 클래스 추가가 가능한 최초의 FSIS 방법인 iMTFA**를 제안합니다. 핵심은 support 이미지를 그대로 저장하는 대신, 객체 인스턴스를 **embedding vector**로 바꾸고 그 평균을 클래스 대표(class representative)로 저장하는 것입니다. 이렇게 하면 메모리 부담이 줄고, 새 클래스를 추가할 때 기존 학습 데이터나 재학습 없이도 확장이 가능해집니다. 논문은 또한 비교를 위해 비증분(non-incremental) 강한 baseline인 **MTFA**도 함께 제안하며, COCO와 VOC에서 기존 SOTA를 일관되게 능가했다고 보고합니다.  

## 1. Paper Overview

### 무엇을 해결하려는가

Few-shot learning에서는 base class에는 데이터가 많고 novel class에는 데이터가 매우 적습니다. 그런데 instance segmentation은 단순 분류보다 어렵습니다. 객체의 **class label**, **bounding box**, **mask**를 모두 예측해야 하기 때문입니다. 기존 FSIS 방법들은 대체로 support set을 네트워크에 직접 주입하거나, novel class를 추가할 때 base/novel 데이터를 다시 사용해 추가 학습을 해야 했습니다. 이는 실제 응용에서 불편합니다. 특히 “이미 잘 학습된 모델에 새로운 소수 클래스만 빠르게 추가”하려는 목적과 맞지 않습니다.

### 왜 중요한가

논문 서두에서 든 예처럼 자율주행용 특수 도로 시설물, 희귀 무기류, 특정 도메인의 드문 객체처럼 **픽셀 단위 annotation을 많이 모으기 어려운 클래스**는 현실에 많습니다. 이런 상황에서 새 클래스를 매번 재학습 없이 붙일 수 있다면, 배포된 segmentation 시스템의 활용성이 크게 올라갑니다. 저자들은 기존 방법이 정확도뿐 아니라 **실용성(practicality)**에서도 부족하다고 보고, incremental FSIS를 주요 문제로 설정합니다.

## 2. Core Idea

### 핵심 직관

이 논문의 핵심 아이디어는 다음 한 줄로 요약할 수 있습니다.

> **이미지를 저장하지 말고, 인스턴스 embedding을 저장하자.**

각 novel class에 대해 $K$개의 support instance가 주어지면, 이를 **Instance Feature Extractor (IFE)**가 embedding으로 변환합니다. 그리고 그 평균을 해당 클래스의 대표 벡터로 사용합니다. 테스트 시에는 proposal마다 얻은 embedding과 클래스 대표 벡터들 사이의 **cosine similarity**를 비교해 분류합니다.

### 무엇이 새로운가

기존 FSIS 방법들(FGN, Siamese Mask R-CNN, Meta R-CNN)은 대체로 support example을 테스트 시에도 직접 사용하거나, 클래스 수가 바뀌면 구조나 학습을 다시 맞춰야 했습니다. 반면 iMTFA는:

* support 이미지를 계속 들고 있지 않아도 되고
* 새로운 클래스를 추가할 때 base class 데이터를 다시 보지 않아도 되며
* 추가 학습 없이 class representative만 classifier에 삽입하면 됩니다.

이 점에서 iMTFA는 **metric-learning의 prototype/weight imprinting 관점**을 instance segmentation의 RoI-level 분류와 결합한 방법이라고 볼 수 있습니다. 논문은 이를 통해 **incremental few-shot instance segmentation의 첫 방법**이라고 주장합니다.

## 3. Detailed Method Explanation

### 3.1 문제 설정

논문은 few-shot learning의 표준 설정을 따릅니다. base class 집합을 $C_{base}$, novel class 집합을 $C_{novel}$이라 두고, 테스트는 두 가지 경우를 고려합니다.

* novel만 평가: $C_{test}=C_{novel}$
* base와 novel을 함께 평가: $C_{test}=C_{base}\cup C_{novel}$

FSIS에서는 query image의 모든 객체에 대해 class label $y_i$, box $b_i$, mask $\mathbf{M}_i$를 예측해야 하므로, few-shot classification보다 훨씬 어렵습니다.

### 3.2 MTFA: 비증분 baseline

저자들은 먼저 object detection용 TFA를 instance segmentation으로 확장한 **MTFA**를 제안합니다. 구조적으로는 **Mask R-CNN 기반 2-stage training**입니다.

1. **1단계**: 전체 Mask R-CNN을 base class로 학습
2. **2단계**: feature extractor는 고정하고, classifier + box head + mask head를 base/novel few-shot 데이터로 fine-tuning

즉, MTFA는 TFA의 “백본 고정 + 헤드만 미세조정” 전략을 segmentation으로 옮긴 strong baseline입니다. 논문에서 Figure 2는 MTFA가 TFA에 mask branch를 추가한 형태임을 보여줍니다.  

이 baseline은 강력하지만, novel class가 바뀌거나 늘어날 때마다 **2단계 fine-tuning을 다시 해야 하는 한계**가 있습니다. 이 점이 곧 iMTFA의 출발점입니다.

### 3.3 iMTFA: Incremental MTFA

#### 전체 구조

iMTFA 역시 Mask R-CNN에서 출발하지만, 핵심은 **RoI level feature extractor를 embedding generator로 재목적화(re-purpose)**한다는 점입니다. 두 단계는 다음과 같습니다.

1. **1단계**: 전체 네트워크를 base classes로 학습
2. **2단계**: backbone/RPN/일부 구성은 고정하고, RoI Feature Extractor $\mathcal{G}$와 cosine-similarity classifier $\mathcal{C}$가 **discriminative embedding**을 학습하도록 조정

이 두 단계 모두 **base classes만으로** 학습됩니다. 이후 novel class는 support 예시 $K$개만 있으면 별도 학습 없이 추가됩니다. Figure 3 설명이 바로 이 절차를 요약합니다.

#### Instance Feature Extractor (IFE)

IFE는 RoI feature를 받아 각 proposal에 대한 embedding을 출력합니다. 이 embedding은 같은 클래스끼리는 가깝고 다른 클래스와는 구별되도록 학습됩니다. 저자들이 중요하게 보는 점은, 이 embedding이 **classification weight space와 정렬(aligned)**되도록 만든다는 것입니다. 그래서 support example에서 얻은 embedding 평균을 classifier의 weight처럼 사용할 수 있습니다.

#### Cosine-similarity classifier

기존 linear classifier 대신 **cosine similarity classifier**를 사용합니다. 직관적으로 proposal embedding $\mathbf{z}$와 클래스 weight $\mathbf{w}_c$ 사이 유사도를 비교합니다. 일반적으로 형태는 다음과 같이 이해할 수 있습니다.

$$
s_c = \alpha \cdot \frac{\mathbf{z}^\top \mathbf{w}_c}{|\mathbf{z}||\mathbf{w}_c|}
$$

여기서 $\alpha$는 scaling factor입니다. 논문은 구현에서 이 $\alpha$를 실험 설정별로 다르게 두었고, ablation으로 그 영향도 분석합니다. COCO-Novel의 iMTFA에는 1.0, COCO-All에는 10.0, MTFA에는 20.0을 사용했다고 명시합니다.

이 cosine head의 장점은 클래스 weight를 **지원 샘플의 평균 embedding으로 자연스럽게 대체/추가**할 수 있다는 것입니다. 즉, novel class에 대해

$$
\mathbf{w}_{novel} = \frac{1}{K}\sum*{k=1}^{K}\mathbf{e}_k
$$

처럼 대표 벡터를 만들고, 이를 기존 base class weight 옆에 붙이면 됩니다. 저자들이 말하는 “incremental addition without training”이 바로 이 부분입니다.

#### 왜 box/mask는 class-agnostic인가

iMTFA의 중요한 설계는 **localization과 segmentation을 class-agnostic**으로 만드는 것입니다. 이 선택 덕분에 새 클래스를 넣을 때 box regressor나 mask predictor를 새로 학습할 필요가 없습니다. 분류만 embedding-based로 해결하면 되므로, novel class 추가 비용이 매우 낮아집니다. 저자들은 이 설계를 iMTFA의 실용성 핵심으로 제시합니다.

반대로 말하면, 이 설계는 나중에 논문이 스스로 인정하는 한계이기도 합니다. class-specific head보다 최적이 아닐 수 있기 때문입니다. 이 점은 뒤의 limitation에서 다시 다룹니다.

### 3.4 왜 2단계 학습이 필요한가

논문은 단순히 “Mask R-CNN + cosine head”만 붙인 one-stage 방식도 비교합니다. 하지만 ablation에서 **iMTFA의 2단계 fine-tuning이 더 낫다**고 보고합니다. 저자들의 해석은 이렇습니다.

* One-Stage-Cosine은 backbone과 RPN 쪽에 더 치우쳐 학습되는 경향이 있고
* One-Stage-Linear는 cosine similarity에 적합한 embedding geometry를 잘 만들지 못한다

즉, RoI feature extractor를 별도로 embedding-friendly하게 정렬시키는 **두 번째 단계**가 성능 향상에 중요합니다.

## 4. Experiments and Findings

### 4.1 데이터셋과 설정

논문은 COCO와 PASCAL VOC를 사용해 실험합니다. 특히 단순히 novel class만 보는 설정뿐 아니라, **COCO 전체 클래스를 한 번에 joint evaluation**하는 실험도 수행합니다. 저자들은 reduced memory requirement 덕분에 이것이 가능해졌다고 강조합니다. 이는 기존 support-image 기반 FSIS 방식과 비교되는 실질적 장점입니다.

논문은 episodic한 평가와 함께, 실제로 모든 클래스가 한 이미지에 등장할 수 있는 상황을 반영하는 joint evaluation도 다룹니다. 이 맥락에서 iMTFA가 “class representative만 저장하면 된다”는 구조적 이점을 실험적으로 보여주려 합니다.

### 4.2 구현 세부사항

구현은 Detectron2 기반 Mask R-CNN이며, backbone은 **ResNet-50 + FPN**입니다. 학습은 SGD, batch size 8, GPU는 NVIDIA V100 2장을 사용합니다. 2단계 fine-tuning의 learning rate는 iMTFA가 0.0007, MTFA가 0.0005입니다. 이런 세부정보는 재현성 관점에서 중요합니다.

### 4.3 주요 결과

논문은 여러 실험 시나리오에서 **MTFA와 iMTFA가 기존 SOTA를 능가**한다고 주장합니다. 특히 COCO에서 Table 1 설명에 따르면:

* MTFA와 iMTFA는 incremental FSOD SOTA인 **ONCE**보다 object detection에서 더 낫고
* 동시에 ONCE는 하지 못하는 **instance segmentation까지 수행**합니다.

즉, 이 논문의 기여는 단순히 incremental segmentation이 “된다”는 데 그치지 않고, detection 기준으로도 competitive하거나 superior하다는 데 있습니다.

또 결론부에서는 두 방법 모두 COCO와 VOC의 다양한 설정에서 current state-of-the-art를 능가했다고 다시 요약합니다. 논문의 전체 메시지는 “정확도와 실용성을 둘 다 챙겼다”는 것입니다.

### 4.4 MTFA와 iMTFA의 관계

흥미로운 점은, 저자들이 iMTFA만 제안한 것이 아니라 **MTFA라는 non-incremental 강한 baseline**도 함께 제시했다는 것입니다. 이는 incremental 방법이 보통 정확도에서 손해를 보는지, 또는 어디서 trade-off가 생기는지를 더 명확하게 보이기 위함입니다. 결론에서도 저자들은 MTFA가 class-specific이기 때문에 어떤 면에서는 더 강한 baseline이며, iMTFA는 그에 비해 유연성에서 큰 이점을 가진다고 정리합니다.

### 4.5 정성적 결과와 실패 사례

Figure 4는 성공 사례와 실패 사례를 함께 보여줍니다. 실패는 크게 다음 유형으로 요약됩니다.

* 잘못된 분류
* 잘못된 detection
* 부정확한 instance segmentation

이는 embedding-based classifier가 novel class 추가에는 유리하지만, class-agnostic localization/mask 때문에 세밀한 분리에서는 한계가 있음을 시사합니다.

## 5. Strengths, Limitations, and Interpretation

### 강점

첫째, **진짜 incremental**하다는 점이 가장 큽니다. 새로운 클래스를 추가할 때 이전 데이터나 재학습이 필요 없다는 것은 few-shot 설정에서 매우 큰 실용적 장점입니다.

둘째, **메모리 효율성**이 좋습니다. support 이미지를 계속 저장하거나 매번 전달하지 않고, embedding vector의 평균만 유지하면 되기 때문입니다. 저자들이 COCO 전체 클래스를 joint evaluation할 수 있었다고 말하는 이유도 여기에 있습니다.

셋째, 방법론적으로 깔끔합니다. Mask R-CNN의 기존 구조를 크게 부수지 않고, RoI-level feature space를 metric-learning적으로 재해석해 incremental FSIS로 확장했습니다. 즉, 새로운 task를 위해 완전히 새로운 detector를 설계하지 않고도 실용적인 해법을 만들었습니다.

### 한계

논문 결론은 한계를 꽤 솔직히 인정합니다.

1. **새 embedding 생성 시 기존 embedding에 적응하지 못함**
   현재 방식은 새 클래스 대표를 평균으로 추가할 뿐, 기존 클래스와의 관계를 동적으로 재조정하지 않습니다. 저자들은 attention mechanism이 향후 개선 방향이라고 말합니다.

2. **class-agnostic box/mask head의 한계**
   iMTFA의 localization과 segmentation은 class-agnostic이므로, class-specific predictor를 쓰는 MTFA보다 원리적으로 불리할 수 있습니다. 저자들 스스로 class-specific regressor/mask predictor로 옮기는 transfer function 학습이 가능할 것이라 제안합니다.

3. **base class bias**
   frozen box regressor와 mask predictor는 base class 기반으로 형성되어 있으므로, novel class에 대해 bias가 생길 수 있습니다. 이는 incremental 설계의 대가로 볼 수 있습니다.

### 비판적 해석

제 판단으로 이 논문의 진짜 가치는 “absolute 최고 성능” 그 자체보다, **few-shot instance segmentation을 배포 가능한 형태로 바꾸는 방향성**에 있습니다. 기존 FSIS 연구는 support-query episodic 설정에서 높은 수치를 내더라도 실제 시스템에 넣기엔 무거운 경우가 많았습니다. iMTFA는 그 지점을 정확히 찌릅니다.

반면, embedding 평균만으로 클래스를 대표시키는 방식은 클래스 내부 다양성이 큰 경우 한계가 분명할 수 있습니다. 예를 들어 appearance variation이 매우 큰 객체군에서는 단일 prototype이 충분하지 않을 수 있습니다. 이 부분은 논문이 직접 실험으로 깊게 파고들지는 않았고, 후속 연구 여지가 큽니다. 이 평가는 논문이 제시한 구조와 한계 진술에 기반한 해석입니다.  

## 6. Conclusion

이 논문은 **incremental few-shot instance segmentation**이라는 거의 비어 있던 문제를 처음으로 본격 제시하고, Mask R-CNN 기반의 실용적 해법인 **iMTFA**를 제안합니다. 방법의 핵심은 RoI-level embedding을 학습하고, support instance들의 평균 embedding을 novel class의 대표 weight로 사용하는 것입니다. 이 덕분에 새로운 클래스를 **재학습 없이** 추가할 수 있고, support image를 테스트 시 계속 들고 다닐 필요도 없습니다. 동시에 비교용 baseline인 MTFA도 제시해, incremental성과 성능 사이의 trade-off를 더 명확히 보여줍니다.

실무적으로는 “희귀 클래스가 계속 추가되는 segmentation 시스템”에서 유의미합니다. 연구적으로는 prototype-based metric learning, weight imprinting, Mask R-CNN 계열 detector를 하나로 잇는 흥미로운 접점이며, 이후 **class-specific transfer**, **attention-based adaptation**, **better prototype modeling** 같은 후속 연구의 발판이 되는 논문입니다.
