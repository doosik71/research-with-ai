# FGN: Fully Guided Network for Few-Shot Instance Segmentation

이 논문은 few-shot learning과 instance segmentation을 결합한 **Few-Shot Instance Segmentation(FSIS)** 문제를 다룬다. 저자들은 FSIS를 “support set이 base segmentation network를 guide하는 조건부 instance segmentation”으로 해석하고, Mask R-CNN의 서로 다른 구성요소에 **서로 다른 guidance mechanism**을 넣는 **FGN(Fully Guided Network)**를 제안한다. 핵심은 “하나의 공통 guidance로는 RPN, detector, mask head가 겪는 서로 다른 일반화 문제를 모두 해결하기 어렵다”는 점이다. 이를 위해 논문은 **AG-RPN**, **RG-DET**, **AG-FCN**의 세 모듈을 도입하고, 공개 데이터셋 실험에서 기존 FSIS 방법보다 더 나은 성능을 보였다고 주장한다.  

## 1. Paper Overview

이 논문의 목표는 **소수의 annotated novel-class instance만 주어진 상황에서 query image 속 novel-class object를 detection + segmentation까지 수행하는 것**이다. 기존 instance segmentation은 대규모 fully annotated data에 강하게 의존하는데, 실제 환경에서는 instance-level annotation이 매우 비싸기 때문에 이런 가정이 잘 성립하지 않는다. 저자들은 이를 해결하기 위해 support examples를 활용하는 few-shot paradigm을 instance segmentation으로 확장한다.

논문이 특히 중요하게 보는 지점은, classification이나 semantic segmentation보다 instance segmentation이 구조적으로 더 복잡하다는 점이다. Mask R-CNN은 2-stage 구조를 가지며, 1단계의 RPN은 proposal을 만들고 2단계는 classification, bbox regression, mask prediction을 각각 수행한다. 따라서 support 정보를 한 군데에서만 주입하거나 하나의 방식으로만 modulation하면 각 단계의 역할 차이를 충분히 반영하기 어렵다. 논문은 바로 이 약점을 정면으로 겨냥한다.  

## 2. Core Idea

이 논문의 중심 아이디어는 한 문장으로 요약하면 다음과 같다.

> **“FSIS에서는 Mask R-CNN의 각 핵심 구성요소가 support set으로부터 서로 다르게 guided되어야 한다.”**

기존 FSIS 계열 방법은 대체로 support 정보를 backbone 초반이나 second stage 앞단의 한 지점에만 넣었다. 그러면 결과적으로 RPN과 detector, 혹은 detector 내부의 classification/mask branches가 같은 guidance를 공유하거나, 어떤 단계는 아예 guidance를 받지 못한다. FGN은 이를 “full guidance의 부재”로 보고, 아예 세 부분을 분리해서 설계한다.  

* **AG-RPN**: support 기반 attention으로 proposal 생성 단계 자체를 class-aware하게 만든다.
* **RG-DET**: support와 query RoI를 직접 비교하는 relation-based detector를 사용한다.
* **AG-FCN**: mask prediction 단계에 support-derived attention을 넣어 segmentation을 guide한다.  

즉, 이 논문의 novelty는 “guided Mask R-CNN” 자체라기보다, **Mask R-CNN의 여러 기능적 블록을 task-specific guidance로 분해해 설계한 점**에 있다.

## 3. Detailed Method Explanation

### 3.1 Problem Formulation

논문은 base classes $\mathcal{C}^{\text{base}}$와 novel classes $\mathcal{C}^{\text{novel}}$를 분리하고, 두 집합은 겹치지 않는다고 가정한다:

$$
\mathcal{C}^{\text{base}} \cap \mathcal{C}^{\text{novel}} = \phi
$$

base classes에 대해서는 충분한 annotation이 있는 데이터 $\mathcal{D}^{\text{base}}$가 있고, novel classes에 대해서는 매우 적은 수의 annotated instances만 존재하며 이를 support set $\mathcal{D}^{\text{novel}}$로 본다. query image $\mathbf{I}^q$에 대해 novel-class instances를 분할하는 것이 FSIS의 목표다. novel class 수가 $N$, 클래스당 support instance 수가 $K$이면 이를 **$N$-way $K$-shot instance segmentation**이라고 정의한다.

논문은 일반적인 supervised detector/segmenter $f_\theta(\mathbf{x})$를 직접 학습하는 대신, support-conditioned model

$$
f_\theta(\mathbf{x} \mid \mathcal{S})
$$

을 학습한다. 이 formulation은 few-shot classification의 episodic training 철학을 instance segmentation에 옮겨온 것이다. 즉, base data에서 support/query episode를 반복 샘플링하여, novel class 환경을 흉내 내며 조건부 분할기를 학습한다.

### 3.2 Overall Architecture

FGN은 Mask R-CNN 위에 세 종류의 guidance를 주입한다. 전체 흐름은 다음과 같이 이해할 수 있다.

1. support set에서 class-relevant representation을 추출한다.
2. query image feature에 이를 반영하여 proposal 생성 단계부터 class-awareness를 부여한다.
3. proposal별로 support-query 관계를 계산해 detector가 novel classes에 적응하도록 한다.
4. 마지막으로 mask head에도 support attention을 반영해 segmentation 품질을 높인다.

이 구조에서 중요한 것은 support set이 단순한 auxiliary input이 아니라, **RPN–Detector–Mask branch 전반을 관통하는 guidance source**라는 점이다.

### 3.3 AG-RPN: Attention-Guided RPN

기존 Mask R-CNN의 RPN은 class-agnostic proposal generator다. 그러나 FSIS에서는 “어떤 novel class가 지금 관심 대상인지”가 support set으로 주어진다. 따라서 저자들은 RPN이 완전히 class-agnostic하게 동작하면 비효율적이라고 본다. AG-RPN의 목적은 **support-derived class-aware attention을 사용해 RPN이 novel-class와 관련된 영역에 더 집중하게 만드는 것**이다. 논문 설명에 따르면 AG-RPN은 support를 class-aware attention 형태로 인코딩하고, 이를 RPN에 적용해 query에서 class-aware proposals를 생성한다.

이 설계의 의미는 크다. proposal 단계부터 novel-class prior를 반영하면, downstream detector와 mask head는 더 정제된 후보만 처리하면 된다. 즉, AG-RPN은 단순 성능 향상뿐 아니라 이후 단계의 검색 공간 자체를 줄여주는 역할도 한다.

### 3.4 RG-DET: Relation-Guided Detector

논문에서 가장 흥미로운 부분은 detector branch다. 저자들은 기존 방식처럼 support vector로 query feature를 단순 reweighting하는 대신, **support feature와 query RoI feature를 명시적으로 비교하는 relation-based detector**를 제안한다. 이 아이디어는 few-shot classification의 Relation Network에서 영감을 받았다. 저자들이 Relation Network를 선호하는 이유는 **feature embedding과 similarity measure가 모두 learnable**하기 때문이다.

하지만 FSIS에서는 일반 few-shot classification과 달리 **background rejection**이 필수다. query의 모든 RoI가 support에 해당하는 class 중 하나일 필요가 없기 때문이다. 논문은 바로 이 차이를 지적하며, RG-DET가 AG-RPN이 만든 개별 RoI에 대해 동작하고, background RoI를 걸러내는 문제를 함께 다룬다고 설명한다. 즉, RG-DET는 “support-query similarity를 계산하는 few-shot classifier”이면서 동시에 “background를 배제해야 하는 detector head”다.

개념적으로 보면 RG-DET는 다음과 같이 이해할 수 있다.

* query RoI feature $\mathbf{z}_j$를 뽑는다.
* support에서 각 class에 대한 대표 표현을 만든다.
* support-query feature를 relation module에 넣어 class-wise relatedness를 구한다.
* 이 결과로 class prediction을 수행하되, background rejection까지 고려한다.

즉, bbox regression과 분류 중 특히 **classification branch의 inter-class generalization 문제**를 relation comparison으로 풀려는 설계다. 논문 결론에서도 classification branch가 여전히 가장 어려운 부분이라고 언급하는데, 이는 RG-DET가 논문의 핵심이면서도 동시에 향후 개선 여지가 큰 부분임을 시사한다.

### 3.5 AG-FCN: Attention-Guided FCN

Mask branch에서도 저자들은 단순히 detector 결과를 받아서 분할하는 것이 아니라, support attention을 다시 활용한다. AG-FCN은 support에서 얻은 attentional information을 mask segmentation 절차에 주입한다. 기본 취지는 AG-RPN과 유사하지만, 목적은 proposal 생성보다 더 미세한 **pixel/region-level foreground delineation**이다. support-derived signal로 “이 novel class의 모양/영역 특성”을 더 잘 반영하게 하는 것이다.

Ablation 결과에서 AG-FCN이 basic FCN과 여러 변형들보다 가장 좋은 성능을 보였다고 논문은 말한다. 이는 segmentation head도 별도 guidance가 필요하다는 저자 주장을 지지한다.

### 3.6 Training Strategy

논문은 base dataset에서 support/query episode를 샘플링하는 few-shot style 학습을 사용한다. 사용 backbone은 **ResNet101**이며, first-stage AG-RPN과 second-stage RG-DET/AG-FCN을 학습하기 위한 SGD 설정을 제시한다. snippet상 세부 하이퍼파라미터 전체는 모두 보이지 않지만, 적어도 저자들이 standard Mask R-CNN fine-tuning이 아니라 **FGN 구조에 맞춘 단계적/모듈적 학습 전략**을 사용했음은 확인된다.

여기서 중요한 해석 포인트는, FGN의 성능이 단순히 backbone을 stronger하게 썼기 때문이 아니라, support-conditioned episode training과 module-wise guidance 설계가 함께 작동한 결과라는 점이다.

## 4. Experiments and Findings

### 4.1 Experimental Settings

논문은 공개 데이터셋에서 실험을 수행하며, 대표적으로 **COCO2VOC**와 **VOC2VOC** 설정을 사용한다. 이는 base/novel class split과 training source가 다른 경우를 비교하기 위한 것으로 보인다. 결과 section에서는 segmentation뿐 아니라 few-shot object detection 결과도 함께 보고한다. 평가 지표로는 주로 $\text{mAP}_{50}$를 사용한다.

특히 VOC2VOC는 COCO2VOC보다 training data 규모가 훨씬 작아 더 어려운 설정으로 제시된다. 논문은 VOC2VOC의 성능이 전체적으로 더 낮다고 설명하면서도, 그 경우에도 FGN이 가장 좋은 overall performance를 보인다고 주장한다.

### 4.2 Main Results

논문에 따르면 FGN은 **FSIS segmentation**과 **few-shot object detection** 모두에서 기존 방법보다 일관되게 우수한 성능을 보인다. 또한 detection에서 segmentation으로 갈수록 모든 방법의 성능이 크게 떨어지는 것도 관찰되는데, 저자들은 이를 통해 “FSIS는 few-shot object detection을 단순 확장해서는 잘 해결되지 않는다”고 해석한다. 즉, mask prediction은 별도의 어려움을 가진다는 뜻이고, 이 때문에 AG-FCN 같은 별도 guidance가 필요하다는 논리와도 연결된다.

정리하면 실험이 보여주는 메시지는 두 가지다.

1. support guidance를 여러 단계에 나눠 넣는 것이 실제 성능 향상으로 이어진다.
2. FSIS는 detection + segmentation의 단순 결합이 아니라, segmentation stage 특유의 어려움을 가진다.

### 4.3 Ablation Study

논문은 **COCO2VOC 3way-3shot** 설정에서 ablation study를 수행한다. 핵심 검증 대상은 세 guidance module의 기여도다. 저자들은 full model에서 하나 또는 여러 모듈을 제거한 variant를 만들고, 결과가 나빠지는지를 본다. 모든 모듈이 segmentation과 detection 양쪽 성능에 기여하며, 즉 **P(AG-RPN), D(RG-DET), S(AG-FCN)** 세 요소가 함께 있을 때 가장 좋다는 결론을 내린다.  

또한 개별 모듈 실험도 흥미롭다.

* **AG-RPN**은 basic RPN이나 기존 설계 기반 변형(AG-RPN-v1)보다 proposal quality 면에서 더 좋다고 보고된다. 이는 support guidance를 proposal 단계에서 주는 것이 실질적으로 유효함을 의미한다.
* **AG-FCN**은 basic FCN 및 다른 variant보다 가장 좋은 결과를 보이며, mask head도 support-conditioned attention으로 분리 설계할 가치가 있음을 보여준다.

이 ablation은 논문의 핵심 주장인 **full guidance**를 가장 직접적으로 뒷받침한다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 문제를 잘 쪼갰다는 데 있다. 많은 few-shot vision 논문이 support conditioning을 하나의 generic modulation으로 처리하는 반면, 이 논문은 Mask R-CNN의 내부 구조를 기능 단위로 분리해 각기 다른 guidance를 설계했다. 이는 architecture-aware한 few-shot adaptation이라는 점에서 설득력이 있다.

또한 RG-DET에서 background rejection 문제를 명시적으로 다룬 점도 좋다. few-shot classification의 relation idea를 그대로 가져오면 detector에는 맞지 않는데, 논문은 이 차이를 분명히 인식하고 있다.

실험적으로도 main result + ablation의 연결이 비교적 깔끔하다. 단지 “좋아졌다”가 아니라, 왜 좋아졌는지에 대해 module-level 증거를 제시한다는 점이 강점이다.

### Limitations

논문 스스로도 인정하듯이, FSIS는 여전히 매우 어려운 문제이고 특히 **classification branch** 쪽에 큰 개선 여지가 남아 있다. 저자들은 background rejection과 복잡한 feature handling 때문에 classification branch가 특히 어렵다고 본다. 이는 RG-DET가 중요한 동시에 완전한 해결책은 아니라는 의미다.

또 하나의 한계는, 현재 확보된 본문 snippet만으로는 모든 수식 세부 구조나 loss의 정확한 closed form을 완전하게 복원하기 어렵다는 점이다. 논문이 architecture intuition은 충분히 전달하지만, relation module의 정확한 계산식과 일부 training detail은 여기서 보인 텍스트만으로는 전부 명확하지 않다. 이 부분은 논문 원문 figure/table을 직접 함께 보면 더 정확히 이해될 것이다.

### Interpretation

비판적으로 보면, FGN은 “few-shot segmentation을 위한 범용 원리”라기보다 **Mask R-CNN에 특화된 guided decomposition**에 가깝다. 하지만 그것이 오히려 강점일 수도 있다. 실제 시스템은 추상적 few-shot formulation보다 detector/segmenter의 구체적 구조와 맞물려 작동하므로, 이런 구조적 세분화가 성능에 더 직접적일 수 있다.

현대 시점에서 보면 transformer 기반 unified segmentation 모델이 더 많이 쓰이지만, 이 논문의 통찰은 여전히 유효하다. 즉, **support information을 어디에, 어떤 방식으로 주입하느냐는 backbone보다도 중요할 수 있다**는 점이다.

## 6. Conclusion

이 논문은 FSIS를 support-conditioned Mask R-CNN 문제로 보고, RPN, detector, mask branch에 각각 다른 guidance를 넣는 **FGN**을 제안했다. 핵심 기여는 다음 두 가지로 정리된다.

1. few-shot instance segmentation을 위한 **Fully Guided Network** 프레임워크 제안
2. **AG-RPN, RG-DET, AG-FCN**의 세 guidance mechanism을 통해 성능 향상 달성

실험 결과와 ablation을 종합하면, 이 논문이 말하고 싶은 메시지는 분명하다.
**FSIS에서는 support set을 한 번만 쓰는 것이 아니라, proposal 생성부터 classification, mask prediction까지 각 단계의 역할에 맞게 다르게 써야 한다.** 바로 그 점이 FGN의 본질이다. 논문 결론에서도 저자들은 FGN이 SOTA를 넘었지만 여전히 classification branch 중심의 개선 여지가 크며, 앞으로 더 나은 guidance mechanism을 탐구하겠다고 밝힌다.
