# From Forks to Forceps: A New Framework for Instance Segmentation of Surgical Instruments

이 논문은 **수술 도구(surgical instruments)의 instance segmentation에서 실제 병목은 mask 생성보다 class 분류 오류일 수 있다**는 문제의식에서 출발한다. 저자들은 자연영상으로 사전학습된 Mask R-CNN류 instance segmentation 모델을 수술 영상에 fine-tuning하면, bounding box와 mask는 꽤 그럴듯하게 나오지만 **도구 종류(class label)를 자주 틀린다**고 분석한다. 특히 수술 도구는 길고 가늘며 비스듬히 놓이고, class 간 외형 차이도 작아서, 일반적인 bounding-box 기반 분류 head가 불리하다. 이를 해결하기 위해 논문은 기존 2-stage instance segmentation 모델 뒤에 **분류만 담당하는 3번째 단계**를 붙인 **S3Net**을 제안하고, 그 핵심 모듈로 **MSMA(Multi-Scale Mask Attended Classifier)**를 설계한다. 이 모듈은 box가 아니라 **예측된 mask를 이용해 도구 영역에 attention**을 주고, 작은 데이터셋과 낮은 inter-class variance에 대응하기 위해 **arc loss 기반 metric learning**을 사용한다. 논문은 EndoVis2017과 EndoVis2018에서 18개 이상의 기존 방법을 능가하며, EndoVis2017 benchmark에서 기존 SOTA 대비 최소 12포인트, 약 20% 향상을 보고한다.  

## 1. Paper Overview

이 논문이 다루는 문제는 단순한 surgical tool segmentation이 아니라, **multi-class instance segmentation of surgical instruments**다. 기존 많은 수술 영상 연구는 semantic segmentation으로 문제를 다뤘지만, 실제 응용에서는 겹치거나 끊겨 보이는 도구를 개별 인스턴스로 분리하고, 동시에 어떤 종류의 도구인지까지 알아야 한다. 이는 instrument tracking, downstream surgical workflow analysis, robotic assistance 같은 작업에 필수적이다. 저자들은 semantic segmentation만으로는 disconnected region이나 overlapping instrument를 안정적으로 다루기 어렵다고 보고, 문제를 명시적으로 instance segmentation으로 재정식화한다.

이 문제가 중요한 이유는 minimally invasive surgery(MIS) 영상의 도구들이 자연영상 객체와 매우 다르기 때문이다. 수술 도구는 대부분 **길고 가는 tubular 구조**를 가지며, 프레임 안에서 수직보다 비스듬히 놓이는 경우가 많고, 반사광, 연기, 블러, 혈흔, 가림(occlusion) 같은 어려운 조건도 흔하다. 더구나 EndoVis2017 같은 핵심 벤치마크는 클래스 수는 제한적이지만 클래스 간 외형 차이가 작고, 실제 구분은 도구의 끝부분(tip)에 달려 있는 경우가 많다. 이런 특성 때문에 자연영상용 detection/segmentation 모델의 분류 head가 잘 일반화되지 않는다는 것이 논문의 핵심 주장이다.  

## 2. Core Idea

논문의 핵심 아이디어는 아주 선명하다.

> **수술 도구 instance segmentation의 주된 실패 원인은 box나 mask localization보다 class misclassification이며, 이를 해결하려면 segmentation 네트워크 뒤에 도구 영역만 보는 전용 classifier를 따로 붙여야 한다.**

저자들은 여러 SOTA 결과를 분석한 뒤, 기존 방법들이 bounding box와 segmentation mask는 비교적 정확하게 얻더라도, 그 인스턴스를 잘못된 도구 클래스로 예측하는 경우가 많다고 주장한다. 이를 강하게 보여주는 예로, Mask R-CNN의 예측 label을 ground truth label로 단순 치환했을 때 AP50이 **0.65에서 0.90으로 크게 상승**했다고 보고한다. 즉 localization보다 classification이 더 큰 병목이라는 것이다.

이 관찰을 바탕으로 제안된 것이 **S3Net**이다. 기존 instance segmentation 모델의 첫 두 단계는 유지하고, 뒤에 **MSMA**라는 3번째 분류 단계를 추가한다. 여기서 novelty는 단순 classifier를 더한 것이 아니라, 수술 도구의 특성에 맞게 다음 세 가지 설계를 결합했다는 점이다.

첫째, **mask-based attention**이다. 수술 도구는 box 대비 실제 차지 면적이 작고 대각선 방향으로 놓여 background leakage가 심하므로, box 전체를 보고 분류하면 배경과 다른 도구가 혼입된다. 그래서 논문은 box가 아니라 **mask가 가리키는 도구 영역만 강조**한다.  

둘째, **multi-scale feature masking**이다. 하나의 해상도 feature만 쓰지 않고 여러 스케일에서 도구 영역을 강조해 분류한다. 이를 통해 tip, shaft, shape cue를 함께 활용하려는 의도다.

셋째, **arc loss 기반 metric learning**이다. 수술 도구는 inter-class variance가 낮고 데이터셋도 작기 때문에, 일반 cross-entropy만으로는 fine-grained discrimination이 어렵다. 논문은 먼저 arc loss로 feature embedding을 분리한 뒤, 그 다음 cross-entropy로 classifier를 fine-tuning하는 전략을 택한다.  

## 3. Detailed Method Explanation

### 3.1 전체 구조: S3Net

S3Net은 이름 그대로 **Three Stage Deep Neural Network**다. 구조는 다음과 같다.

1. **Stage 1: Box Proposal**

   * RPN이 후보 bounding box를 생성한다.

2. **Stage 2: Mask Prediction**

   * 기존 Mask R-CNN 스타일 head가 각 proposal에 대해 mask와 임시 class label $\hat{c}$를 예측한다.

3. **Stage 3: Specialized Classification**

   * 논문이 새로 추가한 **MSMA classifier**가 원본 이미지와 stage-2 mask를 입력으로 받아, class label을 $\hat{c}$에서 더 정확한 $c$로 교정한다.

즉 중요한 점은, 이 논문이 segmentation 자체를 처음부터 새로 설계한 것이 아니라, **기존 instance segmentation 파이프라인의 classification을 분리(decouple)**해서 별도 최적화했다는 것이다. 논문은 이 모듈이 Mask R-CNN에 검증되었지만, 원칙적으로는 다른 instance segmentation 모델에도 삽입 가능하다고 설명한다.

### 3.2 왜 classification이 병목인가

저자들은 두 단계 네트워크에서 **proposal은 1단계에서 부정확하게 나오더라도 2단계 box regression과 mask head로 localization은 상당히 개선**되지만, classification은 여전히 1단계에서 잘린 region crop에 크게 의존해 취약하다고 본다. 즉 bounding box와 mask는 edge 같은 더 robust한 단서로 일반화되지만, class discrimination은 그렇지 않다는 분석이다.

또한 일반적인 NMS는 서로 다른 class의 중복 box를 제거하지 않으므로, 같은 인스턴스가 다른 class들로 중복 검출되는 문제가 생긴다. 논문은 이를 완화하기 위해 **across-class overlapping segments도 제거하는 NMS 변형**을 추가한다. 이는 classification 오류가 단순 softmax 문제를 넘어서, post-processing 단계까지 영향을 준다는 점을 보여준다.

### 3.3 MSMA: Multi-Scale Mask Attended Classifier

MSMA는 이 논문의 기술적 핵심이다. 입력은

* 원본 RGB 이미지 $I_i$
* stage-2가 예측한 각 인스턴스 mask $P_{i,j,\hat{c}}$

이다. 목표는 이 mask가 가리키는 인스턴스의 class label을 다시 판별하는 것이다. 최종적으로 출력 mask는 동일하더라도 label만 $c$로 교정된 $P_{i,j,c}$가 된다.

동작은 다음과 같다.

* ResNet backbone으로 원본 이미지의 **multi-scale feature**를 추출한다.
* 예측 mask를 각 스케일 feature에 곱해 **mask-attended feature**를 만든다.
* 이들을 다시 $1 \times 1$ convolution으로 합쳐 인스턴스별 단일 feature map으로 만든다.
* 그 위에 embedding layer를 올려 인스턴스별 embedding $E_{i,j}$를 생성한다.
* 이 embedding을 이용해 최종 class를 분류한다.

이 설계의 의미는 분명하다. 기존 bounding-box classifier는 box 안의 배경, 다른 도구, specular highlight까지 함께 보게 되지만, MSMA는 **mask가 지정한 도구 공간 정보만 강조**하므로 class-discriminative cue에 더 집중할 수 있다. 특히 수술 도구는 box 안에서 차지하는 면적이 작고 대각선 배치가 많아 background contamination이 심하므로, 이 mask attention이 자연영상보다 더 큰 효과를 낸다.  

### 3.4 Hard-mask attention: training vs inference

학습 시에는 **ground-truth mask**를 사용하고, 테스트 시에는 **stage-2 predicted mask**를 사용한다. 즉, 분류기는 훈련 때는 이상적인 도구 영역에 집중하는 법을 배우고, 테스트 때는 segmentation stage가 뽑은 실제 mask에 의존한다. 이 설정은 “분류기 자체의 능력”을 최대한 잘 학습시키려는 의도다.

이 부분은 논문이 segmentation과 classification을 완전히 분리한 것이 아니라, **mask를 classifier의 spatial prior로 재사용**한다는 점에서 중요하다. S3Net은 결국 segmentation 결과를 더 잘 활용하는 분류 보정 모델이다.

### 3.5 Arc loss와 metric learning

논문은 MSMA를 바로 cross-entropy로만 학습하지 않는다. 먼저 **arc loss**로 embedding space를 정리하고, 그 다음 cross-entropy로 classifier를 fine-tune한다. 저자들의 논리는 다음과 같다.

* 수술 도구는 외형이 비슷하고 차이는 tip 같은 미세한 부분에 있다.
* 데이터셋 규모도 작다.
* 따라서 단순 cross-entropy만 쓰면 class 간 feature separation이 약하다.

arc loss는 face recognition에서 많이 쓰이듯, 클래스 간 angle margin을 강제해 **embedding 간 거리를 더 벌리는 metric learning**이다. 논문도 arc loss가 face recognition domain에서 surgical domain으로 옮겨져, low inter-class variance를 다루는 데 적합하다고 설명한다. 또한 cross-entropy가 weight vector와 embedding의 dot product에 의존하는 반면, arc loss는 angle 중심으로 margin을 강제한다는 점을 강조한다.  

수식적으로 arc loss는 정답 클래스 각도 $\theta_c$에 margin $m$을 더해
$$
\mathcal{L}=-\frac{1}{C}\sum_{c=1}^{C}\log
\frac{e^{\cos(\theta_c+m)}}
{e^{\cos(\theta_c+m)}+\sum_{j\ne c} e^{\cos\theta_j}}
$$
형태로 주어진다. 이 식의 직관은 “정답 클래스와의 angular similarity는 더 높이고, 다른 클래스와는 더 멀어지게 하자”는 것이다. 논문 맥락에서는 graspers, forceps, needle driver처럼 서로 닮은 도구의 fine-grained separation을 돕는 역할을 한다.

## 4. Experiments and Findings

### 4.1 데이터셋과 비교 대상

논문은 주로 **EndoVis2017(EV17)**과 **EndoVis2018(EV18)** benchmark를 사용한다. EV17은 7종류의 로봇 수술 도구를 포함하고, EV18은 validation split을 old/new로 구분해 더 다양한 generalization을 검증한다. 비교 대상은 Mask R-CNN fine-tuning 계열, ISINet, TraSeTR, AP-MTL, Mask-then-classify 등 기존 surgical instrument segmentation 방법들이다.  

### 4.2 메인 결과

논문이 가장 강하게 주장하는 결과는 EV17에서의 향상이다. 저자들은 S3Net이 기존 SOTA보다 **최소 12 points, 약 20%** 개선했다고 말하며, 특히 EV17에서는 ISINet 대비 **30% Challenge IoU**, **60% mcIoU** 향상을 보고한다. 이는 단순 backbone 변경이 아니라, 3단계 분류 보정이 큰 역할을 한다는 해석을 뒷받침한다.  

다만 EV18에서는 해석이 조금 더 미묘하다. 논문은 전반적으로 잘 일반화한다고 주장하지만, 일부 결과에서는 **TraSeTR가 약간 더 나은 경우도 있다**고 인정한다. 저자들은 이를 temporal information의 이점으로 해석한다. 즉 S3Net은 classification 개선에 집중했기 때문에, tracking cue를 활용하는 방법이 유리한 상황에서는 약간 밀릴 수 있다.

이 점은 오히려 논문을 더 신뢰하게 만든다. 모든 경우에 무조건 최고라고 하지 않고, **분류 개선과 temporal modeling은 다른 축의 장점**임을 인정하기 때문이다.

### 4.3 왜 성능이 좋아졌는가: 저자들의 검증

논문은 자신들의 세 가지 주장을 실험으로 검증한다.

첫째, **classification이 병목**이라는 주장이다.
Mask R-CNN의 predicted label을 ground-truth label로 바꾸면 AP50이 **0.65 → 0.90**으로 증가한다. 이는 localization보다 class prediction error가 훨씬 큰 손실 원인임을 직접 보여준다.  

둘째, **길고 비스듬한 도구 모양이 box-based classification을 어렵게 한다**는 주장이다.
EV17 Video 1의 ultrasound probe 예시에서, 높은 IoU의 elongated box가 많았고, 논문은 ground-truth box가 instrument를 더 타이트하게 감싸면 Mask R-CNN 정확도가 크게 오르는 점을 보여준다. 이 실험은 문제의 핵심이 box proposal 자체보다 **box 안에 섞여 들어오는 background**임을 시사한다.  

셋째, **arc loss가 cross-entropy보다 유리하다**는 주장이다.
논문은 third stage를 CE loss와 arc loss로 각각 학습해 비교했고, 최종적으로 metric learning 기반 학습이 sparse class와 fine-grained classes에서 더 낫다고 해석한다. qualitative section에서도 Grasping retractor 같은 sparse class의 향상이 metric learning 덕분이라고 말한다.  

### 4.4 Ablation study

Ablation Table 2는 논문의 설계를 꽤 잘 설명해 준다. 주요 수치만 보면:

* 1,2단계 baseline 변형들은 대체로 **57점대**에 머문다.
* Stage 3를 cross-entropy만으로 둔 **Stage 3_cel**은 **63.63**이다.
* 최종 **S3Net**은 **72.54**다.

이 결과는 몇 가지를 뜻한다.

* 3번째 분류 단계 자체가 효과가 있다.
* 하지만 단순 classifier 추가만으로는 충분하지 않다.
* **mask attention + multi-scale design + arc-loss 기반 학습**까지 포함한 최종 설계가 큰 차이를 만든다.

즉 논문의 성과는 “분류기 하나 붙이기”가 아니라, **수술 도구라는 도메인에 맞춘 분류 전용 설계 전체**에서 나온다.

### 4.5 Qualitative findings

질적 비교에서 논문은 S3Net이 다음 경우에 특히 강하다고 말한다.

* sparse class
* overlapping instruments
* 놓치기 쉬운 instance

반면 failure case도 분명하다.

* 도구의 **shaft만 보이는 경우**
* **orientation이 크게 바뀐 경우**

이는 논문의 방법이 tip 기반 fine-grained cue에 의존하는 만큼, tip이 가려지거나 안 보이면 여전히 어렵다는 뜻이다. 결국 이 논문은 분류 문제를 크게 줄였지만, 수술 도구의 시각적 ambiguity를 완전히 해소한 것은 아니다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **문제 진단이 정확하다**는 점이다. 많은 논문이 더 큰 backbone이나 더 복잡한 segmentation head를 제안하지만, 이 논문은 surgical instance segmentation에서 진짜 병목이 **misclassification**이라는 점을 명시적으로 파고든다. 그리고 그 진단을 “label 치환 시 AP50이 0.65에서 0.90으로 오른다”는 강한 실험으로 뒷받침한다.

두 번째 강점은 **도메인 특화 inductive bias**가 분명하다는 점이다. 수술 도구의 길쭉한 형태, 대각선 배치, box 내부 배경 혼입, tip 위주의 class distinction 같은 특성을 정확히 짚고, 이를 mask attention과 metric learning으로 연결한다. 이는 단순한 성능 최적화가 아니라 **problem-structure-aligned design**이다.  

세 번째 강점은 **plug-in nature**다. 저자들은 MSMA가 원칙적으로 어떤 instance segmentation 모델 뒤에도 붙을 수 있다고 주장한다. 즉 이 논문은 새로운 segmentation backbone보다 **후단 분류 교정 프레임워크**로 보는 편이 맞다.

### 한계

첫째, 이 방법은 segmentation 자체를 크게 개선하기보다 **classification correction**에 초점을 둔다. 따라서 mask 품질이 심하게 나쁜 경우에는 3단계가 근본적 해결책이 되기 어렵다. 실제로 Mask-then-classify 계열의 한계와 유사하게, 앞단 mask가 무너지면 뒤 classifier도 영향을 받을 수 있다. 논문은 이를 줄이려 했지만 완전히 자유롭지는 않다.

둘째, EV18에서 TraSeTR가 약간 더 나은 경우가 있다는 점은, **temporal context를 쓰지 않는 한계**를 보여준다. 논문도 미래 방향으로 temporal information을 활용한 mask/label 개선을 언급한다.  

셋째, failure case가 shaft-only 또는 큰 orientation shift에서 발생한다는 점은, 여전히 tip 중심 구분 전략이 불충분할 수 있음을 시사한다. 즉 mask attention이 background를 줄여도, **도구의 결정적 식별 부위가 보이지 않으면** 분류는 여전히 어렵다.

### 해석

비판적으로 보면, 이 논문의 진짜 기여는 “새로운 instance segmentation 모델”이라기보다, **segmentation과 classification을 decouple한 surgical-domain rethinking**에 있다. 자연영상에서 잘 되던 2-stage detector/segmenter를 의료 영상에 그대로 fine-tune하는 것으로는 부족하며, 특히 long-thin tools 같은 특수 객체는 **전용 classifier with spatial prior**가 필요하다는 메시지를 준다. 이 관점은 surgical instruments뿐 아니라, class 간 차이가 작고 box contamination이 심한 다른 의료/산업 비전 문제에도 확장 가능성이 있다. 다만 그 일반화는 이 논문이 직접 실험한 범위를 넘어서는 해석이므로 가능성 수준으로 보는 편이 적절하다.

## 6. Conclusion

이 논문은 surgical instrument instance segmentation에서 낮은 성능의 핵심 원인을 **mask localization 부족이 아니라 class misclassification**으로 진단하고, 이를 해결하기 위해 기존 instance segmentation 모델 뒤에 **MSMA 기반 3번째 분류 단계**를 추가한 **S3Net**을 제안한다. MSMA는 mask-based multi-scale attention으로 도구 영역에 집중하고, arc loss 기반 metric learning으로 낮은 inter-class variance 문제를 완화한다. 실험적으로는 EndoVis2017과 EndoVis2018에서 강한 성능을 보였고, 특히 EV17에서 기존 SOTA 대비 최소 12포인트, 약 20% 향상을 보고했다.  

실무적으로 이 논문은 매우 좋은 교훈을 준다. **의료 영상에서 자연영상용 segmentation 모델을 그대로 fine-tune했을 때, 어디가 실제 병목인지 먼저 진단해야 한다**는 것이다. 이 논문은 그 병목이 classification이라는 점을 보여줬고, 이를 위한 구조적 해법을 제안했다는 점에서 의미가 크다.
