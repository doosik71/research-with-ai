# Robust Instance Segmentation through Reasoning about Multi-Object Occlusion

이 논문은 여러 객체가 서로 가리는 장면에서의 **instance segmentation** 문제를 다룬다. 저자들은 기존 딥러닝 기반 instance segmentation이 객체를 대체로 **독립적으로** 처리하기 때문에, 서로 인접한 객체들의 **상대적 occlusion 관계**를 제대로 활용하지 못한다고 본다. 이를 해결하기 위해 저자들은 기존 **Compositional Networks**를 확장해, 여러 객체를 동시에 다루는 생성적 모델과 **Occlusion Reasoning Module(ORM)** 을 결합한 네트워크를 제안한다. 이 모델은 각 객체의 class, instance segmentation, occluder segmentation을 먼저 feed-forward로 예측한 뒤, 객체 간 충돌(conflict)과 occlusion order를 추론하여 잘못된 segmentation을 수정하고, 수정된 mask를 다시 top-down으로 분류 개선에 활용한다. 또한 이 방법은 **bounding box supervision만으로 학습 가능**하다는 점도 중요한 특징이다.

## 1. Paper Overview

이 논문의 핵심 문제는 **부분 가림(partial occlusion)이 심한 복잡한 장면에서 각 객체를 얼마나 robust하게 분할할 수 있는가**이다. 실제 이미지 장면은 여러 객체가 겹쳐 있고, 일부 객체는 다른 객체에 의해 부분적으로 가려진다. 이런 상황에서 일반적인 segmentation 모델은 보이는 부분만을 근거로 객체를 이해하려 하므로, occluded object의 mask를 틀리게 예측하거나, 겹치는 영역에서 여러 객체가 같은 픽셀을 foreground라고 주장하는 충돌이 자주 발생한다. 저자들은 이런 문제가 단순히 더 많은 data augmentation만으로는 충분히 해결되지 않는다고 본다.  

논문이 중요한 이유는, occlusion을 단순 nuisance가 아니라 **추론의 대상**으로 본다는 점이다. 즉, “어떤 객체가 앞에 있고 어떤 객체가 뒤에 있는가”를 reasoning해야만 segmentation을 바로잡을 수 있다고 주장한다. 이를 위해 저자들은 기존 CompositionalNets의 장점인 생성적 표현, 즉 객체와 occluder를 분해하고 비가려진 부분에 기반해 인식하는 능력을 유지하면서, 그것을 **multi-object setting**으로 확장한다.  

## 2. Core Idea

이 논문의 핵심 아이디어는 다음과 같다.

> **객체를 독립적으로 segment한 뒤 끝내지 말고, 서로 겹치는 객체들 사이의 segmentation consistency와 occlusion order를 reasoning하여 잘못된 예측을 수정하자.**

기존 CompositionalNets도 occluder를 분리하고 비가려진 object part에 기반한 recognition을 할 수 있었지만, 여전히 **한 번에 하나의 객체를 독립적으로** 처리했다. 저자들은 바로 이 independence assumption이 문제의 원인이라고 본다. 실제 장면에서는 겹치는 두 bounding box의 overlap 영역에 대해 “모든 픽셀이 같은 객체에 일관되게 할당되어야” 자연스럽다. 픽셀별로, 객체별로 독립적으로 처리하면 이 일관성이 깨진다.  

그래서 제안된 ORM은 각 객체의 segmentation likelihood map을 받아, overlap 영역의 **segmentation conflict**를 먼저 찾고, 그다음 **pixel-level competition**과 **pair-wise occlusion order recovery**를 통해 이를 수정한다. 논문의 Figure 3 설명에 따르면, ORM의 입력은 인접한 두 객체의 foreground, background, occlusion likelihood map이며, 두 객체가 동시에 한 픽셀을 foreground로 주장하면 conflict로 간주한다. 이후 경쟁 과정을 통해 픽셀을 재할당하고, 그 결과를 바탕으로 어느 객체가 occludee인지도 추론한다.  

즉, 이 논문의 novelty는 단순히 occlusion-aware segmentation이 아니라, **feed-forward prediction → inconsistency detection → occlusion-order-based correction → top-down feedback**의 폐루프(closed loop)를 만든 데 있다.

## 3. Detailed Method Explanation

### 3.1 전체 파이프라인

![Figure 2:Proposed deep network architecture for multi-object instance segmentation under occlusion](https://ar5iv.labs.arxiv.org/html/2012.02107/assets/figures/pipe_new.png)

논문의 전체 구조는 Figure 2 설명이 잘 요약한다. 입력 이미지에서 bounding box를 기준으로 객체 crop을 만든 뒤, 각 crop은 동일한 **Compositional Network**로 순차 처리된다. 이 단계에서 각 객체에 대해 다음이 독립적으로 예측된다.

* object class
* instance segmentation
* occlusion segmentation  

그다음 이 독립 예측들을 **multi-object reasoning module**, 즉 ORM에 넣어 서로 모순되는 segmentation을 찾고 수정한다. 마지막으로 수정된 instance segmentation mask를 다시 위쪽으로 보내, 해당 객체의 occluded feature를 masking하고 classification score를 개선한다. 이 점이 중요하다. ORM은 segmentation post-processing 역할만 하는 것이 아니라, **classification robustness 향상에도 기여하는 top-down correction signal**이다.  

### 3.2 Base Model: CompositionalNets for Single Objects

논문은 기존 CompositionalNets를 출발점으로 삼는다. CompositionalNets는 일반 CNN의 fully connected classification head를 **differentiable compositional model**로 대체한 구조다. 이 compositional head는 특정 class에 대한 feature activation의 생성 모델 $p(F \mid y)$를 정의하며, 객체를 parts와 context로 분해해 설명할 수 있다. 이런 생성적 구조 덕분에 occluder를 분리하고, 보이는 부분만으로 object recognition을 수행하는 데 강점이 있다.

논문 본문에는 single-object CompositionalNet의 class-conditional feature model이 mixture 형태로 제시된다. 예를 들어,

$$
p(F \mid \Theta_y) = \sum_m \nu_m , p(F \mid \theta_y^m), \qquad \nu_m \in {0,1}, \quad \sum_{m=1}^M \nu_m = 1
$$

와 같이 class $y$에 대해 여러 mixture component를 둘 수 있다. 여기서 $\Theta_y$는 category별 compositional parameters를 나타낸다. 이 식 자체보다 중요한 것은, 분류 헤드가 discriminative classifier가 아니라 **feature에 대한 generative model**이라는 점이다. 이 속성이 이후 occlusion reasoning과 top-down correction의 기반이 된다.

### 3.3 Multi-Object Extension의 필요성

single-object CompNet은 object crop 하나를 처리하는 데는 강하지만, 이미지 내 객체 간 상호작용을 명시적으로 모델링하지 않는다. 논문은 이로 인해 겹치는 객체의 bounding box overlap 영역에서 segmentation 오류가 자주 발생한다고 설명한다. 특히 **같은 category의 객체가 서로 가릴 때** 오류가 더 자주 발생한다고 관찰한다. 이유는 segmentation이 객체별로 독립적이고, 그 내부에서도 픽셀별 독립 가정이 강하기 때문이다. 저자들은 이런 independence assumption이 계산 효율은 높이지만, 이미지에서 중요한 관계를 놓친다고 본다.

### 3.4 Occlusion Reasoning Module (ORM)

![Figure 3:Detailed method in Occlusion Reasoning Module](https://ar5iv.labs.arxiv.org/html/2012.02107/assets/figures/scm_pipeline_new.png)

ORM은 이 논문의 실질적 핵심이다. Figure 3 설명에 따르면 ORM의 입력은 **이웃한 두 객체의 segmentation likelihood maps**다. 각 likelihood map은 픽셀별로 세 상태를 가진다.

* foreground
* background/context
* occlusion

논문에서는 이를 시각적으로 foreground는 파란색, background는 초록색, occlusion은 빨간색으로 표현한다. 더 밝은 픽셀일수록 해당 상태의 likelihood가 높다.

ORM의 첫 단계는 **segmentation conflict detection**이다. 같은 픽셀을 두 객체가 모두 foreground라고 예측하면 conflict로 본다. 이는 feed-forward prediction이 독립적으로 수행되기 때문에 자연히 생기는 오류다. 저자들은 이 conflict를 단순 local heuristic으로 지우는 대신, 객체 간 관계를 반영한 reasoning으로 해결한다.

### 3.5 Pixel-level Competition과 Occlusion Order Recovery

논문 설명에 따르면, ORM은 conflict가 있는 픽셀에 대해 **pixel-level competition**을 수행하여 어느 객체에 속해야 하는지 다시 정한다. 이 과정의 결과는 단순한 pixel reassignment를 넘어, 두 객체 중 누가 앞에 있고 누가 뒤에 있는지에 대한 **pair-wise occlusion order recovery**로 연결된다. 그리고 occludee의 likelihood map은 이 recovered order를 반영해 업데이트된다.

Introduction의 설명은 이를 더 직관적으로 풀어준다. 각 객체는 자신의 bounding box 안 각 픽셀에 대해 “이 픽셀이 내 object가 차지하는가” 또는 “내가 가려져 있는가”에 대해 투표한다. 여러 객체에서 모호한 표가 모이면 segmentation error의 신호가 된다. 이후 분류 점수(classification scores)에 기반한 occlusion order를 사용해 잘못된 픽셀 할당을 수정한다.  

이 과정은 상당히 중요하다. 기존 방법이 각 object crop을 독립적으로 modal/amodal segmentation하는 데 그쳤다면, 이 논문은 **overlap region에서 어떤 object가 visible foreground를 가져야 하고, 어떤 object는 occluded state여야 하는지**를 일관적으로 강제한다.

### 3.6 Top-down Feedback for Classification

ORM으로 수정된 segmentation mask는 끝이 아니라 다시 CompositionalNet으로 피드백된다. 논문은 corrected instance and occlusion masks를 이용해 segmentation errors를 유발한 feature를 masking out하고, 이를 통해 object classification을 개선한다고 설명한다. 즉, segmentation이 classification을 보조하는 top-down pathway가 존재한다.

이 설계는 논문의 해석상 매우 설득력 있다. occlusion이 심한 상황에서는 분류와 분할이 분리된 문제가 아니라, “무엇이 보이고 무엇이 가려졌는가”를 함께 풀어야 하기 때문이다. 저자들은 segmentation correction을 recognition robustness 향상과 직접 연결한다.

### 3.7 Supervision과 학습 설정

논문은 이 네트워크가 **bounding box supervision only**로 학습 가능하다고 강조한다. 이는 pixel-level dense mask annotation 없이도, compositional/generative structure와 occlusion reasoning을 통해 segmentation을 학습한다는 의미다. Related Work section에서도 이 작업을 weakly-supervised instance segmentation 문맥에 위치시키며, 기존 weakly-supervised CompositionalNets를 multi-object occlusion reasoning이 가능하도록 일반화했다고 설명한다.  

## 4. Experiments and Findings

### 4.1 Datasets

논문은 두 종류의 데이터에서 실험한다.

첫째는 **KITTI INStance dataset (KINS)** 이다.
둘째는 저자들이 새롭게 만든 **synthetic occlusion challenge dataset** 이다. 이 synthetic dataset은 KITTI에서 non-occluded object를 segmentation mask로 crop한 뒤, random background 위에 합성해 부분 가림 상황을 인위적으로 만든 것이다. 이 방식은 invisible part에 대한 더 정확한 annotation을 제공하며, 다양한 occlusion scenario를 통제적으로 구성할 수 있다.  

저자들은 synthetic challenge에서 세 가지 유형의 occlusion scenario를 정의한다.

1. 두 객체가 겹치는 기본 pair-wise occlusion
2. 네 객체가 서로 얽혀 가리는 더 복잡한 multi-object occlusion
3. 학습 시 본 적 있는/없는 class가 섞여 occluder가 되는 mixed occlusion scenario

이 설계는 실험적으로 의미가 크다. 단순 benchmark 점수 비교를 넘어서, **어떤 종류의 occlusion reasoning이 실제로 어려운가**를 체계적으로 평가하려는 의도가 보인다.

### 4.2 평가 설정과 Occlusion Level

논문은 occlusion 수준을 네 단계로 나눈다.

* L0: 0%–1%
* L1: 1%–30%
* L2: 30%–60%
* L3: 60%–90%

또한 bounding box 품질의 영향을 제거하기 위해, 모든 모델에 대해 training/testing 모두에서 **ground-truth amodal bounding boxes**를 제공했다고 명시한다. 즉, 실험은 detection error보다 **occlusion-aware segmentation 자체의 난이도**를 보려는 설정이다.  

### 4.3 Main Results on KINS

논문에 따르면 KINS에서는 fully supervised method가 weakly supervised baseline보다 여전히 우수하다. 그러나 저자들의 multi-object extension + ORM은 weakly supervised method와 fully supervised baseline 사이의 격차를 상당히 줄인다. 특히 baseline CompNet 대비 모든 occlusion level에서 개선되며, 고 occlusion level에서 향상이 더 크다. 논문은 modal segmentation에서 **L2에서 mIoU 9.6%**, **L3에서 11.3%** 향상을 명시한다.

이 결과는 논문의 주장을 강하게 지지한다. 가림이 약한 L0/L1보다, 가림이 심한 L2/L3에서 이득이 더 크다는 점은 개선 원인이 단순 regularization이 아니라 **occlusion reasoning 자체**임을 시사한다.

### 4.4 Results on Synthetic Occlusion Challenge

synthetic challenge에서도 유사한 패턴이 나타난다. baseline CompNet은 basic pair-wise occlusion과 mixed occlusion에서는 어느 정도 버티지만, **네 객체가 상호 가리는 multi-object scenario**에서 성능이 크게 떨어진다. 반면 제안된 multi-object reasoning은 이 격차를 줄이고, 모든 occlusion level과 challenge scenario에서 segmentation 성능을 개선한다. 특히 high occlusion level인 L2, L3에서 효과가 크다고 보고된다.  

### 4.5 What the Experiments Actually Demonstrate

이 실험들이 실제로 보여주는 것은 세 가지다.

첫째, 객체를 독립적으로 처리하는 feed-forward segmentation만으로는 overlap region conflict를 잘 해결하지 못한다.
둘째, 객체 간 consistency와 occlusion order reasoning을 추가하면 특히 **심한 가림 상황**에서 성능이 크게 향상된다.
셋째, 이 효과는 알려진 class끼리의 가림뿐 아니라, 일부 **unknown occluder**가 포함된 시나리오에서도 유지된다. 저자들은 자신들의 아키텍처가 unknown occlusion과 multi-object occlusion을 동시에 다룰 수 있다고 명시한다.  

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 **문제를 올바르게 다시 정의했다는 점**이다. 많은 segmentation 모델은 occlusion을 데이터 부족이나 augmentation 문제로 취급하지만, 이 논문은 occlusion을 **관계 추론 문제**로 본다. 그리고 이를 단순 아이디어가 아니라, segmentation conflict detection과 occlusion order recovery라는 구체적 모듈로 구현했다.  

또 다른 강점은 **weak supervision**이다. bounding-box supervision만으로 multi-object occlusion-aware instance segmentation을 학습할 수 있다는 점은 annotation 비용 측면에서 매우 실용적이다.

마지막으로, segmentation correction이 classification 개선으로 이어지는 top-down 구조도 좋다. 이는 단순 segmentation pipeline이 아니라, **recognition과 segmentation이 상호작용하는 구조적 모델**이라는 뜻이다.

### Limitations

한계도 있다. 우선 이 접근은 기본적으로 **bounding box crop 기반 객체 중심 처리**에 의존한다. 논문 실험에서도 ground-truth amodal bounding boxes를 넣어 detection error의 영향을 제거했다. 따라서 실제 end-to-end 시스템에서는 bounding box 품질이 떨어질 경우 현재 보고된 성능만큼의 이득이 그대로 재현될지는 별도 검증이 필요하다.

또한 ORM은 pair-wise conflict reasoning을 중심으로 설명되며, 실제 장면에 객체 수가 많아질수록 관계 추론의 조합 폭이 커진다. 논문은 이를 효율적으로 다룬다고 주장하지만, 다수 객체가 복잡하게 얽히는 초대형 장면에서의 확장성은 추가 검토가 필요해 보인다. 이 부분은 논문의 아이디어상 자연스러운 해석이다.

### Interpretation

비판적으로 보면, 이 논문은 현대적인 end-to-end transformer segmentation과는 다른 계열이다. 대신 훨씬 더 **구조적이고 생성적인 방식**으로 occlusion을 다룬다. 하지만 정확히 그 점이 강점일 수 있다. 가림이 심한 상황에서는 단순 local evidence보다, **무엇이 앞에 있고 무엇이 뒤에 있는지**를 명시적으로 reasoning하는 것이 더 인간적인 접근이기 때문이다.

이 논문의 핵심 메시지는 분명하다.

**multi-object scene understanding에서는 객체를 독립적으로 segment하는 것만으로는 부족하고, overlap 영역의 일관성과 occlusion order를 추론해야 robust한 instance segmentation이 가능하다.**

## 6. Conclusion

이 논문은 multi-object occlusion 상황에서 robust한 instance segmentation을 위해, 기존 CompositionalNets를 확장한 생성적 네트워크와 **Occlusion Reasoning Module(ORM)** 을 제안했다. 모델은 각 객체의 class, instance segmentation, occluder segmentation을 독립적으로 예측한 뒤, 객체 간 conflict를 탐지하고 occlusion order를 추론해 segmentation을 수정하며, 수정된 mask를 top-down으로 분류 개선에도 활용한다. 또한 bounding box supervision만으로 학습 가능하다는 점이 실용적이다.  

실험 결과는 KINS와 synthetic occlusion challenge에서, 특히 **고 occlusion level**과 **복잡한 multi-object occlusion** 상황에서 제안 방법이 baseline CompNet을 의미 있게 개선함을 보여준다. modal segmentation에서는 L2에서 **9.6% mIoU**, L3에서 **11.3% mIoU** 향상이 보고되었다.

실무적 관점에서 이 연구는 자율주행, 복잡한 street scene parsing, amodal reasoning, weakly supervised segmentation 같은 분야에 의미가 있다. 후속 연구로는 stronger detector와의 결합, 더 복잡한 scene graph 수준의 occlusion reasoning, transformer 기반 feature extractor와의 통합이 자연스러운 확장 방향으로 보인다.
