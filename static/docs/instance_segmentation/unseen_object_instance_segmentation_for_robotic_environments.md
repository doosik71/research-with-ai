# Unseen Object Instance Segmentation for Robotic Environments

이 논문은 로봇이 실제 비정형 환경에서 **한 번도 본 적 없는 물체(unseen objects)**를 분리해 인식해야 한다는 문제를 다룬다. 특히 저자들은 로봇 조작이 자주 일어나는 **tabletop environment** 에 초점을 맞춰, semantic class를 모르는 임의의 물체들을 각각 instance 단위로 분할하는 **UOIS (Unseen Object Instance Segmentation)** 문제를 제안한다. 이를 위해 저자들은 synthetic RGB-D만으로 학습하면서도 실제 환경으로 일반화되는 **UOIS-Net** 을 제안한다. 핵심은 RGB와 depth를 한꺼번에 섞어 쓰지 않고, **1단계에서는 depth만으로 거친 초기 mask를 만들고, 2단계에서는 RGB로 mask 경계를 정교화**하는 two-stage 구조다. 또한 학습용으로 **Tabletop Object Dataset (TOD라는 대규모 synthetic dataset도 함께 제시한다.  

## 1. Paper Overview

논문의 문제의식은 매우 실용적이다. 실제 로봇은 모든 물체의 CAD 모델이나 semantic category가 미리 주어진 환경에서만 동작할 수 없다. 물건을 집거나 정리하거나 도구를 사용하는 과정에서는, 로봇이 이전에 학습하지 않은 새로운 물체도 **“하나의 개체”로 구분해낼 수 있어야** 한다. 논문은 이를 category recognition이 아니라 **class-agnostic instance segmentation** 문제로 재정의한다. 즉, “이게 컵인지 책인지”보다 먼저 “여기 몇 개의 개별 물체가 있고 각각의 경계가 어디인지”를 아는 것이 중요하다는 입장이다.

하지만 이 문제에는 두 가지 난점이 있다. 첫째, unseen objects에 잘 일반화하려면 아주 다양한 물체가 들어간 대규모 데이터가 필요하지만, 로봇 환경용 실제 annotated dataset은 거의 없다. 둘째, synthetic data를 쓰더라도 RGB는 현실과 도메인 차이가 커서 sim-to-real generalization이 어렵다. 저자들은 이 한계를 정면으로 다루며, **depth는 상대적으로 현실로 일반화가 잘 되고**, **RGB는 경계 정교화에는 매우 유용하다**는 점을 활용해 모달리티를 분리 설계한다.

## 2. Core Idea

이 논문의 핵심 아이디어는 다음 한 문장으로 요약할 수 있다.

> **Unseen object segmentation에서는 RGB와 depth를 초반부터 단순 결합하는 것보다, depth로 instance seed를 만들고 RGB로 refinement하는 방식이 sim-to-real generalization에 더 유리하다.**

저자들은 synthetic RGB를 직접 segmentation 입력으로 쓰면 real-world로 일반화가 잘 안 된다고 본다. 반면 depth 기반 구조는 simulator와 현실 간 차이가 상대적으로 작으므로, 우선 depth만으로 object center voting을 통해 초기 mask를 만들게 한다. 그다음 RGB는 전체 장면 이해가 아니라 **이미 정해진 local object patch 안에서 경계를 다듬는 역할**만 맡긴다. 저자들은 바로 이 점 때문에, 비록 RGB가 non-photorealistic synthetic image여도 refinement network는 surprisingly well generalize한다고 주장한다.

이 설계의 novelty는 단순한 two-stream fusion이 아니라, **모달리티별 역할 분담**이 명확하다는 데 있다.

* **Depth**: objectness, center voting, 초기 instance mask 형성
* **RGB**: local boundary refinement, sharper mask 생성

즉, 이 논문은 “RGB-D를 같이 쓰자”가 아니라 **“각 모달리티가 잘하는 것만 쓰자”** 는 쪽에 가깝다.

## 3. Detailed Method Explanation

### 3.1 전체 파이프라인

UOIS-Net의 전체 구조는 세 부분으로 정리된다.

1. **Depth Seeding Network (DSN)**
2. **Initial Mask Processor (IMP)**
3. **Region Refinement Network (RRN)**

입력은 하나의 RGB-D 이미지다. 먼저 DSN이 depth만 이용해 물체별 초기 mask를 생성한다. 이 초기 mask는 노이즈나 depth sensor 오류 때문에 경계가 부정확할 수 있으므로, IMP가 표준 image processing 기법으로 이를 다듬는다. 마지막으로 RRN이 RGB patch와 초기 mask를 입력받아 object boundary에 맞게 최종 mask를 정제한다. 논문은 DSN 내부에 비미분(non-differentiable) 과정이 있기 때문에, DSN과 RRN을 **end-to-end가 아니라 separately training** 한다고 명시한다.

### 3.2 DSN: Depth Seeding Network

DSN의 목적은 **class-agnostic initial instance masks** 를 만드는 것이다. 입력은 depth map을 camera intrinsics로 backprojection한 **organized point cloud** $D \in \mathbb{R}^{H \times W \times 3}$ 이다. 즉 각 픽셀은 XYZ 좌표를 가진다. 저자들은 depth가 sim-to-real 문제에서 비교적 일반화가 잘 된다는 기존 관찰에 기반해, 첫 단계는 아예 depth만 사용한다.

DSN은 두 가지 구조를 검토한다.

* **2D center voting**
* **3D center voting**

논문의 기본 2D DSN에서는 U-Net 기반 encoder-decoder가 다음 두 출력을 만든다.

* semantic foreground mask $F \in \mathbb{R}^{H \times W \times C}$
* 각 foreground pixel이 object center를 향해 가리키는 2D direction map $V \in \mathbb{R}^{H \times W \times 2}$

여기서 $C=3$으로 두어 background, tabletop, tabletop objects를 분리한다. 각 object pixel은 자신의 observable mask 중심을 향하는 단위 벡터를 예측한다. 그 후 **Hough voting layer** 로 각 foreground pixel의 방향 예측을 모아 potential object centers를 찾고, NMS와 thresholding으로 center들을 선택한다. 마지막으로 각 픽셀은 자신이 가리키는 가장 가까운 center에 할당되어 초기 instance mask가 만들어진다. 즉, DSN은 proposal-based detector가 아니라 **center voting 기반 bottom-up grouping** 구조다.

### 3.3 왜 2D가 아니라 3D reasoning이 필요한가

논문은 기존 UOIS-Net-2D를 확장하면서, **3D 공간에서 center voting하는 DSN** 을 새로 제안한다. 저자들에 따르면 2D center voting은 state-of-the-art 수준 결과를 내지만 명확한 한계가 있다. 예를 들어 실험 section의 qualitative analysis에서는 다음 failure mode를 언급한다.

* 물체 중심이 occlusion으로 가려지면 center convergence가 사라져 object를 놓침
* false positive region이 생기면 RRN이 이를 되돌리기 어려움
* 하나의 물체 mask가 다른 물체에 의해 시각적으로 둘로 갈라지면 DSN이 올바르게 묶지 못함

이 한계 때문에 3D reasoning이 도입된다. 3D center voting은 깊이 구조를 더 직접적으로 활용하므로 cluttered tabletop scene에서 더 견고하게 작동한다. 저자들은 또 **center vote cluster separation을 유도하는 새로운 loss** 를 제안하며, 이것이 cluttered environment에서 강한 성능을 내는 데 중요하다고 주장한다.

### 3.4 IMP: Initial Mask Processor

논문 본문 snippet에서 IMP의 세부 알고리즘은 길게 보이지 않지만, 역할은 명확하다. DSN이 만든 초기 mask는 센서 노이즈, reflective table surface, voting 실수 등으로 거칠 수 있다. IMP는 이를 **standard image processing techniques** 로 robustify하는 중간 단계다. 즉, IMP는 learned module이라기보다 **초기 segmentation을 refinement-ready 상태로 만드는 preprocessing bridge** 에 가깝다.

### 3.5 RRN: Region Refinement Network

RRN은 이 논문의 두 번째 핵심이다. DSN이 object instance를 대략적으로 찾으면, RRN은 **RGB 이미지와 initial mask** 를 받아 최종 refined mask를 예측한다. 저자들의 핵심 주장은, “RGB로 처음부터 segmentation을 하는 것”은 sim-to-real에 취약하지만, **이미 어느 정도 object location이 주어진 상태에서 local patch mask refinement만 하는 것**은 훨씬 쉬운 문제라는 것이다. 그래서 non-photorealistic synthetic RGB만으로도 real-world generalization이 surprisingly strong하다고 말한다.

이 해석은 설득력이 있다. RRN은 전체 scene semantic understanding을 할 필요가 없고, 하나의 local object patch에만 집중하면 된다. 따라서 texture, color contrast, visible boundary를 활용해 noisy depth mask를 edge-aligned mask로 바꿀 수 있다. 논문은 실제로 reflective table surface로 인해 depth가 매우 noisy한 경우에도 sharp mask를 생성할 수 있다고 설명한다.

### 3.6 학습 전략

중요한 점은 DSN과 RRN이 **모두 synthetic data만으로 학습되고, real-world fine-tuning이 없다는 것**이다. 학습 데이터로는 논문이 새로 만든 **Tabletop Object Dataset (TOD)**를 사용한다. TOD는 ShapeNet object와 ShapeNet table을 random indoor home scene에 놓고, PyBullet physics simulator로 물리적으로 쌓이게 만든 뒤, depth와 non-photorealistic RGB를 렌더링해 생성한다. 이 데이터셋 자체가 논문의 중요한 기여 중 하나다.

## 4. Experiments and Findings

### 4.1 실험 목적

논문은 두 방향을 검증하려 한다.

1. synthetic RGB-D만으로 real-world unseen object segmentation이 가능한가
2. 2D DSN보다 3D DSN이 cluttered robotic environment에서 더 강한가

즉, 단순히 segmentation accuracy를 보는 것뿐 아니라, **sim-to-real transfer** 와 **clutter robustness** 를 동시에 평가한다.

### 4.2 Main Results

논문은 UOIS-Net-3D가 OCID와 OSD 같은 real-world UOIS benchmark에서 기존 방법보다 더 좋은 성능을 보였다고 보고한다. snippet에 따르면 Table II 비교에서 UOIS-Net-3D는 **Mask R-CNN**, **PointGroup**, 그리고 이전 버전인 **UOIS-Net-2D** 보다 높은 overlap / boundary F-measure를 기록한다. 구체적으로 OCID에서는 UOIS-Net-2D 대비 overlap F-measure가 **상대적으로 5.8%**, boundary F-measure가 **6.7%** 증가했다고 한다. 또 동일한 TOD로 학습한 best Mask R-CNN 대비 overlap F-measure **8.1%**, boundary F-measure **6.0%** 상대 향상을 보고하며, PointGroup 대비로도 overlap **7.9%**, boundary **6.3%** 향상을 주장한다. OSD overlap F-measure에서도 UOIS-Net-2D 대비 **4.3%**, Mask R-CNN 대비 **12.4%**, PointGroup 대비 **5.7%** 상대 향상을 제시한다.

이 결과는 중요한 함의를 가진다. 단순한 category-aware instance segmentation 모델을 class-agnostic하게 재학습하는 것보다, **robotic tabletop setting에 특화된 center-voting + refinement 구조** 가 더 효과적이라는 뜻이다.

### 4.3 3D DSN의 이점

논문은 3D reasoning이 2D보다 더 robust하다는 점도 강조한다. snippet에 따르면 UOIS-Net-3D는 convexity assumption이 없음에도 불구하고, convexity-based segmentation 방법인 LCCP의 overlap recall에 가깝게 접근한다. 저자들은 “2D에서 3D로 바뀔 때 DSN structure만 변경되므로, 이 recall 증가는 주로 3D center voting structure와 더 좋은 initial mask 덕분”이라고 해석한다.

즉, 성능 향상의 주요 원인은 backbone이 아니라 **center voting representation의 차이** 라는 점이 논문이 보여주고 싶은 메시지다.

### 4.4 Failure Modes

저자들은 failure mode도 비교적 명확히 분석한다. DSN의 실패 사례로는 다음이 제시된다.

* 중심이 occluded된 물체를 놓침
* false positive region 생성
* 한 물체가 다른 물체에 의해 시각적으로 둘로 split될 때 잘못 분할

RRN의 failure mode도 언급된다.

* object texture가 너무 복잡하면 refinement 실패
* initial mask에 충분한 padding이 없으면 전체 object를 복원하지 못함
* DSN이 여러 물체를 하나로 under-segment하면 RRN이 이를 고치기 어려움

이 분석은 구조적으로도 자연스럽다. RRN은 boundary refinement에는 강하지만, **instance topology를 새로 발명하는 모듈은 아니기 때문**이다. 즉, DSN이 심하게 틀리면 RRN의 수정 능력에도 한계가 있다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

가장 큰 강점은 **문제 분해가 정확하다**는 점이다. RGB-D를 early fusion하지 않고, depth는 grouping에, RGB는 boundary refinement에 쓰는 설계가 sim-to-real 관점에서 매우 합리적이다. 이는 단순 엔지니어링 선택이 아니라, 각 modality의 generalization 특성을 반영한 구조 설계다.

또 다른 강점은 **class-agnostic unseen object segmentation** 을 실제 robotic tabletop setting에 맞게 구체화했다는 점이다. 기존 Mask R-CNN류는 bounding box에 여러 물체가 들어가면 애매해지는 반면, UOIS-Net은 center voting 기반 bottom-up grouping을 사용해 clutter에 더 직접적으로 대응한다.

마지막으로 TOD dataset 제안도 중요한 기여다. 실제 annotation 없이도 다양한 ShapeNet object 조합으로 대규모 synthetic training set을 만들 수 있게 했고, 이것이 real-world benchmark 성능 향상으로 이어졌다는 점에서 실용성이 높다.

### Limitations

한계도 분명하다. 첫째, 2D DSN은 object center가 occluded되거나 mask가 분리되어 보이는 경우 취약하다. 그래서 3D reasoning이 도입되었지만, 이는 곧 **center-voting formulation 자체가 특정 failure mode를 가진다**는 뜻이기도 하다.

둘째, RRN은 local refinement에는 강하지만 global regrouping에는 약하다. 논문이 직접 말하듯 DSN이 여러 물체를 하나로 under-segment하면 RRN이 이를 완전히 수정하지 못할 수 있다. 따라서 전체 시스템의 upper bound는 여전히 DSN quality에 크게 의존한다.

셋째, 논문은 tabletop setting에 집중하므로, 복잡한 full-scene indoor/warehouse manipulation으로 그대로 일반화되는지는 추가 검증이 필요하다. 이는 방법의 약점이라기보다 적용 범위의 경계다.

### Interpretation

비판적으로 보면, 이 논문은 “unseen object recognition” 전반을 해결한 것이 아니라, **tabletop manipulation에서 필요한 instance segmentation 문제를 매우 실용적으로 푼 것**에 가깝다. 하지만 바로 그 점 때문에 가치가 있다. 로봇 조작에서 중요한 것은 category label보다 **graspable object mask** 인 경우가 많기 때문이다. 논문 abstract도 실제로 unseen objects for robot grasping에 적용 가능함을 강조한다.

또한 이 논문은 sim-to-real에서 자주 보이는 “더 realistic rendering” 또는 “domain adaptation” 대신, **문제 자체를 더 쉬운 형태로 재구성** 하는 전략이 얼마나 강력한지를 보여준다. 즉, RGB를 잘 일반화시키려 애쓰기보다, RGB가 맡을 일을 local refinement로 제한해버린 것이다.

## 6. Conclusion

이 논문은 robotic tabletop environment에서의 **Unseen Object Instance Segmentation** 문제를 다루며, synthetic RGB-D만으로 real-world unseen objects를 분할하는 **UOIS-Net** 을 제안했다. 핵심은 **Depth Seeding Network(DSN)**로 depth 기반 초기 instance mask를 만들고, **Region Refinement Network(RRN)**로 RGB를 이용해 경계를 정교화하는 two-stage 구조다. 여기에 학습용 synthetic benchmark인 **Tabletop Object Dataset (TOD)**도 함께 제안했다.

실험 결과는 이 설계가 단순 early RGB-D fusion보다 더 낫고, 특히 **UOIS-Net-3D** 가 cluttered real-world setting에서 **Mask R-CNN, PointGroup, UOIS-Net-2D** 보다 우수한 overlap / boundary F-measure를 보인다는 것을 보여준다. 또한 failure analysis는 DSN의 center-voting quality가 시스템 전체 성능에 매우 중요함을 드러낸다.  

실무적으로 이 논문은 로봇 grasping, bin/tabletop picking, manipulation pre-segmentation 같은 문제에 의미가 크다. 앞으로의 확장 방향으로는 더 복잡한 장면에서의 generalized 3D grouping, stronger refinement module, 그리고 foundation-model 기반 objectness prior와의 결합이 자연스럽게 떠오른다.
