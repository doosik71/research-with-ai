# FreePoint: Unsupervised Point Cloud Instance Segmentation

## 1. Paper Overview

이 논문은 **3D point cloud에서 아무런 수작업 annotation 없이 instance segmentation을 수행하는 문제**를 다룹니다. 기존의 point cloud instance segmentation은 보통 per-point mask, bounding box, semantic label 같은 고비용 annotation에 크게 의존하는데, 저자들은 이러한 의존성을 제거한 **unsupervised, class-agnostic instance segmentation** 설정을 정면으로 다룹니다. 핵심 목표는 point들을 적절한 feature space에서 표현한 뒤, 그 유사도를 기반으로 coarse한 pseudo instance mask를 만들고, 이를 다시 segmentation network 학습에 활용해 더 나은 instance mask를 얻는 것입니다. 논문은 이 접근이 단순한 비지도 clustering 수준을 넘어서, 실제로 **Mask3D 기반 fully supervised 방법과의 성능 격차를 상당히 줄이고**, 일부 기존 fully supervised 방법도 넘을 수 있음을 보입니다. 또한 이 작업을 **3D instance segmentation pretext task**로 활용하면, 적은 annotation만으로도 supervised fine-tuning 성능을 높일 수 있음을 보여줍니다.

연구 문제가 중요한 이유는 분명합니다. point cloud annotation은 매우 비싸고 오래 걸리며, 실내 3D scene처럼 객체 수가 많고 구조가 복잡한 환경에서는 더욱 그렇습니다. 따라서 annotation-free 혹은 label-efficient한 학습법은 실제 3D scene understanding 시스템의 확장성과 비용 측면에서 매우 중요합니다. 논문은 이 문제를 “semantic category를 맞히는 것”보다 더 근본적인 “instance 경계를 찾는 것”으로 재정의하고, 이를 class-agnostic하게 푸는 것이 유의미하다고 주장합니다.

## 2. Core Idea

이 논문의 핵심 아이디어는 다음 한 문장으로 요약할 수 있습니다.

> **좋은 point feature를 만들고, 그 feature affinity로 graph partitioning(multicut)을 수행해 pseudo instance mask를 만든 뒤, 그 noisy pseudo label을 직접 쓰지 않고 약지도 학습 방식으로 refinement하는 것**입니다.

좀 더 구체적으로 보면, 저자들은 point마다 다음 정보를 함께 사용합니다.

* 좌표 $(x,y,z)$
* 색상 $(r,g,b)$
* normal vector
* self-supervised backbone이 추출한 deep feature

딥 feature만으로는 서로 다른 instance가 충분히 떨어지지 않는 경우가 있어, 전통적 geometric/color cue를 함께 넣는 것이 중요하다고 봅니다. 이후 point들을 node로 하는 graph를 만들고, edge cost를 feature affinity로 정의한 다음, **minimum-cost multicut** 문제를 풀어 point set을 여러 instance cluster로 나눕니다. 이 결과는 아직 거칠지만 pseudo mask 역할을 합니다. 마지막으로, 이 pseudo mask를 사용해 Mask3D를 학습하되, **weakly-supervised loss와 two-step training**을 통해 pseudo label noise를 완화합니다.

이 논문의 novelty는 크게 세 가지로 볼 수 있습니다.

첫째, **3D point cloud class-agnostic instance segmentation을 완전 비지도 설정으로 밀어붙인 점**입니다. 둘째, **deep feature와 전통적 feature를 결합한 hybrid representation**을 중심으로 multicut 기반 pseudo mask generation을 설계한 점입니다. 셋째, **pseudo label의 오류를 그대로 두지 않고 weakly-supervised refinement로 보정하는 학습 전략**을 명시적으로 설계했다는 점입니다. 논문 후속 버전과 arXiv 원문 요약에서는 여기에 더해 multicut의 randomness를 줄이기 위한 **id-as-feature strategy**도 강조합니다.

## 3. Detailed Method Explanation

### 3.1 전체 파이프라인

FreePoint의 파이프라인은 크게 3단계입니다.

1. **Preprocessing + point feature extraction**
2. **Point graph 구성 및 multicut 기반 pseudo mask 생성**
3. **Pseudo mask를 이용한 two-step training**

논문 그림 기준으로 보면, 입력 point cloud에서 먼저 배경을 제거하고, foreground point에 대해 hybrid feature를 계산한 뒤, graph partitioning으로 coarse instance mask를 얻습니다. 이후 이를 supervision 신호처럼 사용해 Mask3D를 학습합니다. 다만 여기서 supervision은 정확한 GT가 아니므로, 일반적인 fully supervised loss를 그대로 쓰지 않고 약지도 형태로 바꿉니다.

### 3.2 Preprocessing

비지도 setting에서 특히 어려운 점은 **object와 background의 분리**입니다. 작은 물체가 벽, 바닥, 천장 같은 주변 구조물에 붙어 있으면 instance boundary를 찾기 어렵습니다. 저자들은 indoor point cloud라는 데이터 특성을 활용해, 큰 평면 구조물은 background일 가능성이 높다고 보고 **plane segmentation**으로 주요 평면을 제거합니다. 이 단계는 완벽한 semantic understanding이 아니라, “instance segmentation에 방해되는 큰 배경 면을 먼저 걷어내는” 전처리라고 볼 수 있습니다.

또한 계산 비용과 graph segmentation의 안정성을 위해 **farthest point sampling**으로 foreground point를 다운샘플링합니다. 저자들의 해석은 흥미로운데, point가 더 sparse해지면 feature space에서 샘플 간 분리가 커져 segmentation에 유리하다고 봅니다. 즉, 단순한 연산량 절감 이상의 역할을 기대합니다.

### 3.3 Feature Representation

이 논문에서 중요한 통찰 중 하나는 **deep feature만으로는 충분하지 않다**는 점입니다. self-supervised pretrained backbone이 point embedding을 제공하더라도, 서로 다른 instance가 여전히 가까워질 수 있고, 그러면 graph partitioning이 instance를 잘못 합칠 수 있습니다. 이를 보완하기 위해 논문은 다음 특성을 모두 함께 사용합니다.

* deep embedding
* coordinate feature
* RGB color
* normal vector

이 조합은 point cloud 분야의 전통적 segmentation 직관과 self-supervised representation learning을 결합한 것입니다. 특히 좌표와 색상, normal은 표면 구조나 물체 경계에 관한 강한 prior를 제공합니다. 따라서 이 방법은 “representation learning이 다 해결해준다”기보다는, **geometry-aware hybrid affinity modeling**에 가깝습니다.

### 3.4 Graph Construction과 Multicut

feature가 준비되면 point들을 vertex로 하는 graph $G=(V,E,A)$를 구성합니다. 여기서:

* $V$: point들
* $E$: point 간 연결
* $A$: affinity cost vector

입니다. $A$는 point feature 사이의 유사도 혹은 분할 비용을 나타냅니다. 이후 **minimum-cost multicut** 문제를 풀어 graph를 여러 connected subset으로 나누고, 각 subset을 하나의 coarse instance로 간주합니다.

직관적으로 보면, 유사한 point끼리는 같은 partition에 남기고, 다른 객체에 속할 가능성이 높은 point끼리는 cut을 유도하는 방식입니다. 이 접근의 장점은 object center regression이나 semantic category prediction 없이도, **point affinity만으로 instance hypothesis를 만들 수 있다**는 점입니다. 반면 약점은 affinity quality에 매우 민감하다는 것입니다. affinity가 불안정하면 과분할(over-segmentation) 또는 과병합(under-segmentation)이 쉽게 발생합니다.

후속 버전 설명에 따르면 affinity는 embedding, normal, xyz, rgb 항을 가중합한 형태로 구성됩니다. 즉 개략적으로는 다음과 같은 형태입니다.

$$
A = \alpha_1 A_{emb} + \alpha_2 A_{norm} + \alpha_3 A_{xyz} + \alpha_4 A_{rgb}
$$

여기서 각 항은 서로 다른 feature 공간에서 계산된 affinity이고, $\alpha_i$는 그 중요도를 조절하는 가중치입니다. 이 수식의 의미는 단순합니다. “딥 표현만 믿지 말고, 기하와 외관 단서도 같이 보자”입니다.

### 3.5 Pseudo Mask를 이용한 학습

multicut 결과는 coarse mask이므로 noise가 큽니다. 따라서 이를 GT처럼 직접 사용하면 학습이 흔들립니다. 논문은 이를 해결하기 위해 **weakly-supervised training strategy**를 제안합니다. 2D 비지도 instance segmentation의 FreeSOLO와 비슷한 방향이지만, point cloud에 맞게 단순화했습니다. 예를 들어 2D처럼 voxel/grid 축으로 projection하는 대신, 각 mask에 대해 **하나의 center와 bounding box**를 계산해 supervisory signal로 활용합니다. 이는 point cloud에 더 자연스럽고 계산도 단순합니다.

중요한 점은 논문이 “mask 자체가 정확하지 않다”는 사실을 인정하고, 그 불완전성을 center/box 수준의 더 약한 구조적 제약으로 보정하려 한다는 것입니다. 즉, mask supervision을 약화시키되, instance localization에 필요한 핵심 구조는 유지하는 방향입니다.

### 3.6 Two-Step Training

two-step training은 이 논문의 실질적 성능 향상 핵심입니다.

* **1단계**: 일반적으로 더 잘게 쪼개진 **over-segmented base masks**로 먼저 학습
* **2단계**: 그다음 비교적 덜 쪼개진 **under-segmented masks**로 짧게 추가 학습

왜 이렇게 하느냐가 중요합니다.

* over-segmentation은 하나의 객체가 여러 조각으로 나뉘는 문제가 있지만, 다른 객체끼리 잘못 합쳐질 위험은 상대적으로 작습니다.
* under-segmentation은 하나의 mask 안에 여러 객체가 섞일 수 있어 초반 학습에 치명적입니다.

논문은 실제로 **under-segmented mask로 바로 학습하면 training이 실패**한다고 보고합니다. 반대로 먼저 over-segmented mask로 모델이 어느 정도 instance structure를 배우게 한 뒤, under-segmented mask를 소량 활용하면 성능이 더 올라갑니다. 이는 noisy pseudo label 학습에서 “처음엔 보수적으로, 나중에 더 공격적으로” 접근하는 curriculum-like 전략으로 해석할 수 있습니다.

### 3.7 이 방법의 본질적 의미

결국 FreePoint는 “비지도 segmentation 방법”이라기보다, 더 정확히는

> **비지도 pseudo-label 생성 + 약지도 refinement를 결합한 hybrid framework**

입니다.

즉, multicut 자체가 최종 해답이 아니라, 그것을 시작점으로 삼아 neural instance segmenter가 더 나은 boundary와 grouping을 학습하도록 만드는 방식입니다. 이 때문에 논문은 순수 전통적 clustering보다 더 높은 ceiling을 가지며, 동시에 완전 supervised보다 낮은 label cost를 달성합니다.

## 4. Experiments and Findings

### 4.1 실험 설정

논문은 크게 두 가지 축에서 성능을 검증합니다.

첫째, **unsupervised class-agnostic instance segmentation** 자체의 성능입니다. 주로 indoor scene 데이터인 **ScanNet v2**에서 평가합니다.
둘째, FreePoint를 **pre-training task**로 사용했을 때, downstream semantic instance segmentation 성능이 좋아지는지 평가합니다. 이 경우 **ScanNet v2에서 pre-train하고 S3DIS에서 fine-tune**합니다.

평가 지표로는 AP, $AP_{50}$ 등이 사용되며, pretraining 실험에서는 기존 self-supervised 3D pretraining 방법들과 비교합니다.

### 4.2 Main Result: Unsupervised Instance Segmentation

가장 인상적인 메시지는, FreePoint가 **완전 비지도임에도 fully supervised counterpart와의 격차를 의미 있게 줄였다**는 점입니다. 논문은 FreePoint가 class-agnostic ScanNet 설정에서 **fully-supervised Mask3D 정확도의 50% 이상**에 도달하며, 일부 기존 fully supervised 방법을 넘는다고 보고합니다. 또한 arXiv 원문 메타데이터 기준으로는 **기존 전통적 방법보다 18.2% 이상, 동시대 경쟁 방법 UnScene3D보다 5.5% AP 우위**라고 요약됩니다.

이 결과가 의미하는 바는 명확합니다. point cloud instance segmentation에서 supervision의 대부분이 semantic category prediction과 dense annotation에 의해 좌우된다고 생각하기 쉬운데, 실제로는 **좋은 grouping prior와 refinement 전략만으로도 상당한 수준의 instance decomposition이 가능하다**는 것입니다.

### 4.3 Main Result: Pretext Task로서의 가치

논문은 FreePoint를 단지 비지도 segmentation 도구로만 제시하지 않습니다. 더 흥미로운 부분은 이것을 **unsupervised pre-training pretext**로 활용하는 것입니다. ScanNet v2에서 FreePoint 방식으로 backbone을 학습하고, 이를 S3DIS semantic instance segmentation에 fine-tuning했을 때, 제한된 annotation 상황에서 training from scratch보다 뚜렷한 향상이 나타납니다.

대표적으로:

* **10% mask annotation**만 사용할 때, training from scratch 대비 **약 +5.8% AP**
* 기존 CSC 대비도 **약 +3.4% AP**
* arXiv v2 요약 기준으로는 CSC 대비 **+6.0% AP**라고도 정리됩니다. 버전/표 기준에 따라 수치가 약간 다르게 기술된 것으로 보이며, 핵심 메시지는 “기존 self-supervised pretraining보다 강하다”입니다.

표 일부에 따르면 기존 방법들의 성능은 대략 다음 범위입니다.

* PointContrast: AP 44.4
* DepthContrast: AP 45.2
* CSC: AP 46.3

FreePoint는 이들보다 더 높은 AP를 보입니다. 즉, FreePoint는 단순한 point-level representation 학습이 아니라, **instance-aware pretraining**으로서 downstream task에 더 직접적으로 도움이 된다고 해석할 수 있습니다.

### 4.4 Ablation에서 드러나는 핵심

논문의 ablation은 이 방법이 왜 동작하는지를 잘 보여줍니다.

첫째, **two-step training이 실제로 효과적**입니다. 1단계만으로도 coarse mask보다 상당한 개선이 생기고, 2단계에서 under-segmented mask를 추가로 활용하면 성능이 더 오른다고 보고합니다. 반면 under-segmented mask를 처음부터 쓰면 학습이 실패합니다.

둘째, **딥 feature 단독보다 hybrid feature가 중요**합니다. 이는 논문의 기본 가정과 일치합니다. 즉, 3D indoor scene에서는 geometry와 color의 역할이 여전히 크며, self-supervised embedding만으로 instance discrimination을 충분히 보장하기 어렵습니다.

셋째, 일반적인 self-training보다 저자들의 전략이 더 낫습니다. Table 7은 regular self-training과 비교해 FreePoint식 pseudo mask 활용이 더 효과적임을 보여줍니다. 이는 noisy pseudo label을 무작정 반복 재학습하는 것보다, **어떤 형태의 noise를 어떤 순서로 학습할지 설계하는 것이 더 중요하다**는 점을 시사합니다.

### 4.5 Qualitative Result

추가 시각화 결과에서 FreePoint는 chair, sofa, table처럼 크기와 형태가 비교적 뚜렷하고 빈도가 높은 객체에서는 좋은 segmentation을 보입니다. 반면 **작고 혼잡한 객체**, 또는 **배경과 밀착된 객체**는 실패가 많습니다. 이는 비지도 affinity 기반 방법의 전형적 한계와 맞닿아 있습니다. feature space에서 구분이 어려운 경우 graph partition만으로는 경계를 안정적으로 회복하기 어렵기 때문입니다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 문제 설정 자체입니다. 3D point cloud instance segmentation에서 완전 비지도, class-agnostic 설정은 당시에도 매우 덜 탐구된 주제였고, 논문은 여기에 대해 구체적이고 성능 있는 baseline을 제시합니다. 또한 방법론이 지나치게 복잡하지 않습니다. plane segmentation, hybrid feature, multicut, Mask3D refinement라는 구성은 각각 이해 가능하고 역할도 분명합니다.

또 다른 강점은 **pretext task로서의 확장성**입니다. 단순히 “비지도 segmentation도 된다”에서 끝나지 않고, supervised downstream에 실제로 도움이 되는 representation을 학습한다는 점이 실용적입니다. 이 때문에 FreePoint는 annotation-free segmentation 알고리즘이면서 동시에 **label-efficient 3D learning framework**로도 읽을 수 있습니다.

### Limitations

논문이 스스로 밝히듯, FreePoint는 여전히 **fully supervised counterpart에 한참 못 미칩니다**. 또한 semantic label이 없기 때문에 instance category 자체는 예측하지 못합니다. 즉, 이 방법은 어디까지나 **class-agnostic grouping**이며, semantic instance segmentation 전체를 대체하지는 못합니다.

또한 성능은 indoor scene의 강한 구조적 prior에 의존합니다. plane segmentation으로 큰 배경을 제거하는 단계는 ScanNet/S3DIS 같은 실내 데이터에서는 잘 맞지만, 더 복잡하거나 비정형적인 야외 장면에서는 그대로 통하지 않을 가능성이 큽니다. 이 점은 논문이 직접 길게 논하진 않지만, 방법 구조상 자연스러운 한계입니다.

마지막으로, multicut 기반 pseudo label quality가 전체 파이프라인의 상한을 결정합니다. 즉, refinement network가 후처리를 잘해도, 초기 graph partition이 너무 나쁘면 회복이 어렵습니다. 이는 graph construction과 feature affinity 설계가 병목이 된다는 뜻입니다.

### Brief Critical Interpretation

비판적으로 보면, FreePoint의 진짜 기여는 “완전 비지도 end-to-end instance segmentation”이라기보다, **좋은 pseudo-label engineering과 training curriculum 설계**입니다. 하지만 이것이 오히려 강점이기도 합니다. 실제 응용에서는 완전한 이론적 순수성보다, label 없이도 돌아가고 downstream 성능에 도움이 되는 체계가 더 중요하기 때문입니다.

또 하나 흥미로운 점은, 논문이 2D 비지도 segmentation의 아이디어를 그대로 가져오지 않고 3D point cloud에 맞게 바꿨다는 것입니다. 특히 center/box supervision으로 단순화한 약지도 손실과 over/under-segmentation을 순차적으로 활용하는 전략은 3D 데이터 특성에 잘 맞는 설계입니다. 이 부분이 단순 이식이 아니라 실제 연구 기여라고 볼 수 있습니다.

## 6. Conclusion

FreePoint는 point cloud instance segmentation에서 **annotation-free, class-agnostic, unsupervised**라는 어려운 설정을 실질적인 성능으로 보여준 논문입니다. 핵심은 hybrid point feature를 기반으로 multicut pseudo mask를 만들고, 이를 weakly-supervised two-step training으로 refinement하는 것입니다. 이 조합을 통해 완전 비지도 환경에서도 surprisingly strong한 instance segmentation 성능을 얻었고, 더 나아가 3D semantic instance segmentation의 pretraining에도 유용함을 입증했습니다.

실무적으로는, dense annotation이 부족한 3D scene understanding 환경에서 이 논문이 특히 중요합니다. 연구적으로는 3D self-supervised learning이 단순 point-level contrastive feature를 넘어서, **instance structure 자체를 학습 신호로 사용할 수 있다**는 방향을 제시했다는 점에서 의미가 큽니다. 이후의 3D unsupervised instance segmentation, object discovery, panoptic pretraining 연구의 출발점 중 하나로 볼 수 있습니다.
