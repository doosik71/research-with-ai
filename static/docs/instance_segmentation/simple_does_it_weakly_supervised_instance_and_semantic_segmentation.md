# Simple Does It: Weakly Supervised Instance and Semantic Segmentation

* **저자**: Anna Khoreva, Rodrigo Benenson, Jan Hosang, Matthias Hein, Bernt Schiele
* **발표연도**: 2016
* **arXiv**: [https://arxiv.org/abs/1603.07485](https://arxiv.org/abs/1603.07485)

## 1. 논문 개요

이 논문은 **bounding box annotation만으로 semantic segmentation과 instance segmentation을 얼마나 잘 학습할 수 있는가**라는 질문에 답한다. 일반적으로 semantic segmentation은 각 픽셀에 클래스 라벨을 붙여야 하고, instance segmentation은 같은 클래스 내부에서도 각 객체 인스턴스를 구분하는 마스크가 필요하므로, annotation 비용이 매우 크다. 논문은 이러한 비싼 pixel-wise annotation 대신, 상대적으로 훨씬 저렴한 **object bounding box supervision**만을 이용해 segmentation network를 학습하는 방법을 제안한다.

핵심 문제 설정은 분명하다. 기존 weakly supervised segmentation 연구들은 보통 네트워크 학습 자체를 바꾸거나, expectation-maximization 같은 별도 최적화 절차를 설계했다. 반면 이 논문은 문제를 **“훈련 라벨의 노이즈 문제”**로 해석한다. 즉, segmentation network 자체를 복잡하게 바꾸기보다, bounding box로부터 얼마나 좋은 pseudo label을 만들어 주느냐가 본질이라고 본다. 이 관점은 매우 실용적이다. 이미 잘 동작하는 fully supervised segmentation training pipeline을 그대로 유지하면서, 입력 라벨만 바꾸면 되기 때문이다.

이 문제가 중요한 이유는 분명하다. 논문에서도 인용하듯이 pixel-wise mask annotation은 bounding box annotation보다 약 $15\times$ 더 많은 시간이 든다. 따라서 box supervision만으로 fully supervised 성능의 상당 부분을 회복할 수 있다면, 실제 데이터 구축 비용을 크게 줄일 수 있다. 특히 Pascal VOC12와 COCO 같은 대규모 detection annotation 자원을 segmentation 학습에 재활용할 수 있다는 점에서 의미가 크다.

이 논문의 가장 강한 메시지는 단순하다. **좋은 pseudo mask만 만들 수 있다면, segmentation network의 training procedure를 바꾸지 않고도 fully supervised 성능의 약 95%까지 도달할 수 있다**는 것이다. 그리고 이 주장을 semantic labelling뿐 아니라 instance segmentation까지 확장해서 보여준다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 직관적이다. bounding box에는 이미 세 가지 중요한 정보가 들어 있다고 본다.

첫째, **background cue**이다. annotation box가 exhaustive하다고 가정하면, 어떤 box에도 속하지 않는 픽셀은 background로 둘 수 있다.
둘째, **object extent cue**이다. bounding box는 객체가 대략 어느 공간 범위 안에 있는지 알려 준다.
셋째, **objectness cue**이다. 객체는 보통 연속적인 영역을 이루고, 경계가 배경과 대비되며, plausible한 shape를 가진다. 이 정보는 GrabCut이나 object proposal 같은 classic vision 기법으로 반영할 수 있다.

이 논문이 기존 접근과 구별되는 지점은, weak supervision을 위해 네트워크 loss나 학습 알고리즘을 복잡하게 새로 설계하지 않았다는 점이다. 예를 들어 BoxSup은 proposal을 training loop 내부에서 반복적으로 선택하며 학습 절차를 바꾸고, WSSL은 EM 기반 bias를 도입한다. 반면 이 논문은 **학습 루프는 fully supervised 때와 동일하게 두고**, bounding box에서 생성한 pseudo segmentation mask를 training target으로 넣는다.

또 하나의 핵심 아이디어는 **ignore region의 적극적 사용**이다. 논문은 weak label에서 recall을 약간 포기하더라도 precision이 높은 라벨을 만드는 것이 중요하다고 본다. 예를 들어 box 전체를 foreground로 쓰기보다, 더 신뢰할 수 있는 중심부만 foreground로 두고 애매한 부분은 ignore하는 방식이 더 좋다는 것을 실험으로 보인다. 즉, weak supervision에서는 “많이 맞는 noisy label”보다 “적게 쓰더라도 더 정확한 label”이 더 효과적일 수 있다는 것이다.

논문의 최종 설계는 classic segmentation 도구와 deep network를 자연스럽게 결합한다. GrabCut+가 box 내부 foreground/background를 대략 분리하고, MCG proposal이 object-like shape prior를 제공한다. 그리고 두 방법이 **동시에 foreground라고 동의하는 픽셀만 foreground로 채택**하고, 나머지는 ignore로 둔다. 이 단순한 intersection 전략이 surprisingly strong한 pseudo mask를 만든다.

## 3. 상세 방법 설명

### 3.1 전체 접근의 큰 그림

논문의 전체 구조는 크게 두 단계로 볼 수 있다.

첫 번째는 **bounding box에서 training label 생성** 단계이다.
두 번째는 **생성된 label로 standard segmentation network를 학습**하는 단계이다.

여기서 중요한 점은 두 번째 단계가 완전히 표준적이라는 것이다. semantic segmentation에서는 DeepLab-LargeFOV를, instance segmentation에서는 DeepMask와 변형된 DeepLab을 사용한다. 즉, 논문의 핵심 공헌은 새로운 convnet architecture나 새로운 loss가 아니라, **box를 segmentation supervision으로 바꾸는 label generation strategy**에 있다.

---

### 3.2 Semantic segmentation용 box baseline

가장 단순한 baseline은 매우 직관적이다. 각 bounding box 내부의 모든 픽셀을 해당 객체 클래스의 foreground로 채운다. 두 box가 겹치면 작은 box가 앞에 있다고 가정한다. box 바깥 픽셀은 background로 둔다.

이렇게 만든 직사각형 마스크는 당연히 매우 거칠다. 하지만 저자들은 여기서 출발해도 의외로 강한 결과를 만들 수 있다고 본다.

#### Naive recursive training

첫 번째 실험은 이 직사각형 마스크로 네트워크를 한 번 학습한 뒤, 그 네트워크가 training image에 예측한 segmentation을 다시 pseudo label로 사용해 다음 round를 학습하는 방식이다. 즉,

$$
\text{boxes} \rightarrow \text{train network} \rightarrow \text{predict masks} \rightarrow \text{retrain}
$$

형태의 recursive training이다.

하지만 단순 recursion만으로는 성능이 오히려 나빠질 수 있다. 이유는 첫 round의 noisy prediction이 이후 round에 누적되며 drift하기 때문이다. 논문은 이를 직접 보여주며, **recursive training 자체보다 de-noising 규칙이 중요하다**고 주장한다.

#### Box 방식의 세 가지 후처리

논문은 recursive training 사이에 세 가지 post-processing을 넣는다.

첫째, **box enforcing**이다. box 바깥의 모든 픽셀은 다시 background로 강제한다. 이는 cue C1, 즉 background prior를 이용하는 것이다.

둘째, **outliers reset**이다. 어떤 predicted segment가 해당 bounding box에 비해 너무 작으면, 예를 들어 box와의 IoU가 50%보다 작으면, 그 객체 영역을 초기 직사각형 라벨로 되돌린다. 이는 최소 object area를 보장하려는 장치로 cue C2에 해당한다.

셋째, **DenseCRF**이다. 네트워크 출력이 실제 image boundary를 더 잘 따르도록 CRF filtering을 적용한다. 이는 cue C3, 즉 objectness와 boundary prior를 반영한다.

이 세 단계를 포함한 recursive training을 논문은 `Box`라고 부른다. 단순 Naive와 달리, `Box`는 반복 학습을 거치며 성능이 꾸준히 오른다.

---

### 3.3 Ignore region을 넣은 Box^i

논문은 여기서 한 단계 더 나아가 `Box^i`를 제안한다. 아이디어는 box 전체를 foreground로 두지 않고, **중심부 20% 영역만 foreground로 채우고 나머지 내부 영역은 ignore**하는 것이다.

이 방식의 직관은 분명하다. bounding box 중심부는 객체일 가능성이 높다. 반면 가장자리 부근은 background가 섞여 있을 수 있다. 따라서 처음부터 box 전체를 강하게 foreground로 감독하면 noise가 크다. 차라리 중심부만 학습에 사용하고 애매한 영역은 손실 계산에서 제외하는 편이 낫다.

이 설계는 weak supervision에서 precision-recall trade-off를 조절하는 매우 중요한 장치다. `Box^i`는 recall은 줄이지만 precision을 높여, 실제로 `Box`보다 더 좋은 결과를 낸다.

---

### 3.4 Box-driven segment: GrabCut+, GrabCut+^i, MCG, MCG∩GrabCut+

저자들은 rectangular label 자체가 근본적으로 suboptimal하다고 본다. 그래서 box를 입력으로 받아 보다 object-shaped한 pseudo mask를 만드는 classic vision 기법을 사용한다.

#### GrabCut+

기본 GrabCut은 bounding box로부터 foreground/background graph cut segmentation을 수행하는 전통적 방법이다. 논문은 여기에 변형을 가해 `GrabCut+`를 제안한다. 핵심 차이는 pairwise term에 일반적인 RGB color contrast 대신 **HED boundary detector의 경계 확률**을 사용한다는 점이다. 즉, 경계 prior를 더 잘 반영해 object boundary를 더 정교하게 따르도록 만든다.

보충 실험에서도 `GrabCut+`가 기본 GrabCut보다 성능이 좋다고 보고한다.

#### GrabCut+^i

`GrabCut+^i`는 ignore region을 더 적극적으로 활용한다. 각 box에 대해 약 150개의 perturbed GrabCut+ 결과를 만든다. perturbation은 다음과 같이 준다.

* box 좌표를 $\pm 5%$ jitter
* GrabCut에서 바깥 background region의 크기를 10%에서 60%까지 변경

이렇게 얻은 다수의 segmentation 결과에 대해 픽셀 단위 voting을 한다.

* 70% 이상이 foreground로 판단하면 foreground
* 20% 미만이 foreground로 판단하면 background
* 그 사이면 ignore

이 방식은 “여러 noisy segmentation이 공통으로 동의하는 부분만 신뢰한다”는 접근이다. 매우 practical하며 weak supervision에 잘 맞는다.

#### MCG proposal

MCG는 object proposal을 생성하는 방법이다. 논문은 box annotation이 주어졌을 때, MCG가 생성한 proposal들 중 **box와 overlap이 가장 큰 proposal**을 선택하여 pseudo segment로 사용한다. 이는 shape prior를 더 강하게 반영하는 방법이다.

#### 최종 방법: $M \cap G+$

논문의 최종 semantic segmentation용 pseudo label은 `MCG ∩ GrabCut+`이다. box 내부 픽셀 중에서 **MCG와 GrabCut+가 모두 foreground라고 판단한 픽셀만 foreground**로 두고, 나머지는 ignore로 둔다.

이를 식처럼 쓰면 개념적으로는 다음과 같다.

$$
Y_{\text{fg}} = Y_{\text{MCG}} \cap Y_{\text{GrabCut+}}
$$

그리고 box 내부에서 교집합에 속하지 않는 영역은 주로 ignore 처리된다.

이 설계의 핵심은 두 방법의 보완성이다. GrabCut+는 boundary-aware한 figure-ground 분리에 강하고, MCG는 object-like shape prior에 강하다. 둘의 교집합은 recall을 다소 희생하지만, 훨씬 더 신뢰할 수 있는 foreground seed를 준다. 결과적으로 네트워크가 더 정확한 supervision을 받게 된다.

논문이 강조하는 바는 분명하다. **복잡한 weakly supervised learning objective보다, 더 좋은 pseudo label generator가 더 중요할 수 있다**는 것이다.

---

### 3.5 Semantic segmentation 학습 절차

semantic segmentation 실험에서는 DeepLab-LargeFOV를 사용한다. VGG16 ImageNet pretraining에서 시작하고, Pascal VOC12 또는 Pascal+COCO 데이터로 SGD 학습한다. mini-batch는 30이고 initial learning rate는 0.001이다. Pascal에서는 2k iteration 후, COCO pretraining에서는 20k iteration 후 learning rate를 10배 감소시킨다. test time에는 DenseCRF를 적용한다.

중요한 점은 모든 실험에서 **network 구조와 training procedure는 고정**되어 있다는 것이다. 즉, 성능 차이는 대부분 input label quality 차이에서 나온다.

---

### 3.6 Instance segmentation으로의 확장

논문은 같은 box-driven segment idea를 instance segmentation에도 적용한다. instance segmentation은 각 bounding box에 대해 해당 인스턴스의 foreground/background mask를 예측하는 문제로 본다.

이를 위해 각 ground-truth box에서 `GrabCut+`로 instance-level pseudo mask를 생성하고, 이를 supervision으로 convnet을 학습한다. 이때 semantic class label보다 중요한 것은 **이 박스 안의 특정 객체 인스턴스의 mask**이다.

#### DeepLab_BOX

논문은 DeepLab을 instance segmentation에 맞게 변형한 `DeepLab_BOX`를 제안한다. 입력은 4채널이다.

* 3채널 RGB image
* 1채널 binary box map

즉, 네트워크는 이미지뿐 아니라 “어느 bounding box 안 객체를 분할해야 하는지”까지 함께 입력받는다. 출력은 해당 박스 객체의 segmentation mask이다. 이 binary box map은 일종의 spatial guidance 역할을 하며, 장면의 다른 객체들이 아니라 특정 인스턴스 하나만 분할하게 도와준다.

이 구조는 식으로 복잡하게 제시되지는 않았지만, 함수 형태로 쓰면 개념적으로 다음과 같다.

$$
M = f(I, B)
$$

여기서 $I$는 이미지, $B$는 bounding box binary map, $M$은 해당 box에 대응하는 instance mask이다.

논문에는 새로운 loss function이 명시적으로 서술되지 않는다. 문맥상 표준적인 pixel-wise foreground/background segmentation loss를 사용한 것으로 이해되지만, 본문 추출 텍스트에는 loss의 구체식이 제시되어 있지 않으므로 단정할 수는 없다.

## 4. 실험 및 결과

## 4.1 Semantic segmentation 실험 설정

주요 데이터셋은 Pascal VOC12 segmentation benchmark이다. 20개 foreground class와 1개 background class가 있다. 기본 segmentation train/val/test split 외에, Hariharan et al.의 augmentation을 사용해 train set을 10,582장으로 늘린다. 추가 실험에서는 COCO에서 Pascal 20 classes를 포함하고 box area가 200 pixel보다 큰 객체가 있는 이미지 99,310장을 더 사용한다.

평가 지표는 21개 클래스 평균 pixel IoU, 즉 mIoU이다. 이는 semantic segmentation의 표준 지표다.

---

### 4.2 Semantic segmentation: baseline과 최종 결과

Pascal VOC12 validation set에서 fully supervised DeepLab은 69.1 mIoU를 기록한다. weak supervision만 쓴 방법들은 다음과 같은 흐름을 보인다.

* `Box`: 61.2
* `Box^i`: 62.7
* `MCG`: 62.6
* `GrabCut+`: 63.4
* `GrabCut+^i`: 64.3
* `M \cap G+`: 65.7

가장 중요한 관찰은 두 가지다.

첫째, **ignore region이 거의 항상 도움이 된다**. `Box`보다 `Box^i`가 좋고, `GrabCut+`보다 `GrabCut+^i`가 좋다.
둘째, **최종 교집합 방식 $M \cap G+$가 단일 방법보다 낫다**. 이는 MCG와 GrabCut+가 truly complementary하다는 증거다.

논문은 test set에서도 strong한 결과를 보인다. VOC12만 사용한 weak supervision에서 `M \cap G+`는 validation 65.7, test 67.5 mIoU를 기록한다. 이는 fully supervised DeepLab의 test 70.5 mIoU 대비 약 95.7% 수준이다.

즉,

$$
\frac{67.5}{70.5} \approx 95.7%
$$

로, box supervision만으로 full supervision의 대부분을 회복한 셈이다.

---

### 4.3 기존 방법과의 비교

논문은 BoxSup과 WSSL을 중요한 비교 대상으로 둔다. VOC12 weak setting에서 reported numbers는 다음과 같다.

* WSSL with rectangles: test 54.2
* WSSL with segments: test 62.2
* BoxSup with MCG: test 64.6
* 제안 방법 $M \cap G+$: test 67.5

즉, 기존 strong baseline보다도 더 높은 성능을 달성한다. 특히 BoxSup처럼 training procedure를 바꾸지 않았는데도 더 좋다는 점이 중요하다.

저자들은 이를 통해 weak supervision의 핵심 병목이 학습 알고리즘이 아니라 **label quality**임을 뒷받침한다.

---

### 4.4 COCO 추가 데이터의 효과

VOC12 + COCO를 사용하면 weak supervision 결과는 더 좋아진다.

* weak `Box^i`: test 66.7
* weak `M \cap G+`: test 69.9
* full DeepLab: test 73.2

여기서 VOC12 + COCO weak `M \cap G+`의 validation 68.9는 VOC12 full DeepLab validation 69.1과 사실상 비슷하다. 논문은 이를 통해 **더 많은 bounding boxes만 있으면, segmentation mask 없이도 VOC12 full supervision 수준에 거의 맞출 수 있다**고 주장한다.

이 메시지는 매우 실용적이다. annotator가 expensive mask를 그리지 않아도 detection dataset을 활용해 충분히 강한 segmentation 모델을 만들 수 있기 때문이다.

---

### 4.5 Semi-supervised 결과

논문은 일부 이미지만 full mask가 있고 나머지는 box만 있는 semi-supervised setting도 실험한다. 흥미롭게도 VOC12에서 10% 정도의 full mask를 추가해도 `M \cap G+`의 성능 향상은 크지 않다. 65.7에서 65.8 수준이다. 이는 weak pseudo label 자체가 이미 상당히 강하다는 뜻이다.

반면 Pascal full supervision + COCO box supervision 조합에서는 69.1에서 71.6으로 2.5 point 상승한다. 즉, 많은 양의 추가 box annotation은 여전히 유의미한 이득을 제공한다.

---

### 4.6 Boundary supervision의 영향

`GrabCut+`와 `M \cap G+`는 HED boundary detector를 사용하고, HED는 BSDS500의 boundary annotation으로 학습된다. 따라서 “이 추가 supervision이 결과에 얼마나 기여했는가?”라는 질문이 생긴다.

논문은 weakly supervised boundary detector로 HED를 대체해 실험했고, validation 성능이 65.7에서 64.8로 약 1 point 떨어진다고 보고한다. 즉, BSDS500 boundary supervision은 도움이 되지만 결정적이지는 않다. 이 점은 제안 방법의 효과가 외부 boundary dataset에만 의존하는 것이 아님을 보여준다.

---

### 4.7 다른 backbone으로의 일반화

논문은 DeepLabv2-ResNet101으로도 결과를 보고한다. VOC12에서 weak `M \cap G+`는 69.4, full은 74.5이다. VOC12+COCO에서는 weak 74.2, full 77.7이다. 비율로는 약 93%~95.5% 수준이다.

즉, 제안 방식은 특정 VGG16-DeepLabv1 조합에만 맞는 trick이 아니라, 더 강한 backbone에도 적용 가능하다는 점을 보여준다.

---

### 4.8 Instance segmentation 결과

instance segmentation에서는 mAP$^r$ at IoU 0.5, mAP$^r$ at IoU 0.75, ABO를 보고한다. 이 지표들은 detection box가 아니라 predicted segment mask를 기준으로 평가한다.

training-free baselines를 먼저 보면:

* Rectangle: mAP$_{0.5}^r$ 21.6, ABO 38.5
* Ellipse: 29.5, ABO 41.7
* MCG: 28.3, ABO 44.7
* GrabCut: 38.5, ABO 45.8
* GrabCut+: 41.1, ABO 46.4

즉, instance segmentation에서도 `GrabCut+`가 가장 강한 hand-crafted baseline이다.

convnet 기반 weak supervision 결과는 다음과 같다.

VOC12 기준:

* weak DeepMask: 39.4 / 8.1 / 45.8
* weak DeepLab_BOX: 44.8 / 16.3 / 49.1
* full DeepMask: 41.7 / 9.7 / 47.1
* full DeepLab_BOX: 47.5 / 20.2 / 51.1

VOC12 + COCO 기준:

* weak DeepMask: 42.9 / 11.5 / 48.8
* weak DeepLab_BOX: 46.4 / 18.5 / 51.4
* full DeepMask: 44.7 / 13.1 / 49.7
* full DeepLab_BOX: 49.4 / 23.7 / 53.1

여기서도 weak supervision 결과가 full supervision의 약 95% 수준까지 도달한다. 특히 DeepLab_BOX가 DeepMask보다 consistently 좋다. 이는 box map을 별도 입력 채널로 넣는 설계가 instance-specific guidance로 잘 작동함을 시사한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제를 매우 단순하고 정확하게 재정의했다는 점**이다. weakly supervised segmentation을 복잡한 learning algorithm 문제로 보지 않고, **좋은 noisy label을 만드는 문제**로 재해석했다. 그리고 그 관점이 실제로 strong empirical result로 이어졌다. 실험 결과는 이 해석이 단순한 주장에 그치지 않음을 보여준다.

두 번째 강점은 **방법의 실용성**이다. 제안 방법은 segmentation network의 학습 절차를 거의 그대로 유지한다. 기존 DeepLab training pipeline을 건드릴 필요가 없고, box에서 pseudo label만 만들어 넣으면 된다. 실제 연구나 산업 현장에서 이런 종류의 단순성은 매우 큰 장점이다.

세 번째 강점은 **ignore region의 역할을 분명하게 보여준 점**이다. weak supervision에서는 모든 픽셀에 억지로 라벨을 붙이는 것보다, 애매한 픽셀을 버리고 신뢰도 높은 supervision만 쓰는 편이 더 낫다는 교훈을 실험적으로 설득력 있게 제시한다. 이는 이후 semi-supervised, self-training, pseudo-labelling 연구 전반에 통하는 중요한 통찰이다.

네 번째 강점은 **semantic segmentation과 instance segmentation을 함께 다룬 점**이다. 특히 weakly supervised instance segmentation 결과를 처음 보고했다는 점은 공헌으로 볼 만하다. 단순히 semantic segmentation 한 문제에만 갇히지 않고, box-to-mask paradigm의 일반성을 보여주었다.

다섯 번째 강점은 **추가 데이터 활용 가능성**을 명확히 제시한 점이다. COCO box annotations를 활용해 full supervision에 매우 근접하는 결과를 얻는 것은, “cheap annotation + large scale” 전략이 실제로 먹힌다는 강한 근거다.

반면 한계도 분명하다.

첫째, 이 방법은 여전히 **외부 전통적 segmentation 도구의 품질에 의존**한다. GrabCut+, MCG, HED boundary 등이 pseudo label quality를 좌우한다. 즉, weakly supervised segmentation이라기보다, 어느 정도는 strong handcrafted prior에 기반한 label engineering에 가깝다.

둘째, 최종 성능이 높기는 하지만 **라벨 생성 파이프라인 자체는 단순하지 않다**. GrabCut perturbation, voting, MCG selection, intersection, ignore region 등 여러 단계가 필요하다. training objective는 단순하지만, data preprocessing은 오히려 복잡해졌다고 볼 수 있다.

셋째, 논문은 pseudo label 생성의 성공을 주로 Pascal VOC12 같은 비교적 작은 class set과 전형적 object benchmark에서 보인다. 더 복잡한 장면, 더 많은 클래스, 더 심한 occlusion, 더 다양한 domain에서 같은 수준으로 작동할지는 본문만으로는 확실하지 않다.

넷째, instance segmentation 부분은 의미 있는 첫 결과이지만, 파이프라인이 매우 단순하게 구성되어 있어서 contemporary end-to-end instance segmentation 시스템과 직접 비교하기엔 제한이 있다. 특히 detection 품질에 의존하는 구조이며, mask branch 자체의 설계도 간단하다.

다섯째, 본문 추출 텍스트 기준으로는 학습 loss의 구체식, instance segmentation의 정확한 optimization objective, 혹은 일부 hyperparameter의 세부 근거는 충분히 자세히 설명되지 않는다. 따라서 방법의 모든 세부를 완전히 재현하려면 supplementary와 구현 코드가 더 필요할 수 있다.

비판적으로 보면, 이 논문은 “simple does it”라는 제목처럼 정말 단순한 관찰을 강하게 밀어붙인 좋은 empirical paper다. 다만 이 접근의 힘은 representation learning 자체의 혁신이라기보다, **careful pseudo label engineering**에서 나온다. 따라서 이후 더 강한 foundation model이나 self-supervised representation이 등장한 시대에는 이 아이디어가 더 일반화된 pseudo-labelling 프레임워크의 일부로 이해될 가능성이 크다. 그럼에도 당시 맥락에서는 매우 설득력 있고 영향력 있는 결과였다.

## 6. 결론

이 논문은 bounding box annotation만으로도 semantic segmentation과 instance segmentation을 높은 수준으로 학습할 수 있음을 체계적으로 보여준다. 핵심 기여는 새로운 deep architecture나 새로운 loss 설계가 아니라, **bounding box를 high-quality pseudo segmentation label로 바꾸는 간단하지만 효과적인 방법론**을 제시한 데 있다.

semantic segmentation에서는 `Box^i`, `GrabCut+^i`, `MCG ∩ GrabCut+` 같은 단계적 설계를 통해 weak supervision의 성능을 steadily 끌어올리고, 최종적으로 fully supervised 성능의 약 95% 수준에 도달한다. instance segmentation에서도 `GrabCut+` 기반 pseudo mask와 `DeepLab_BOX` 같은 간단한 구조만으로 full supervision에 가까운 성능을 얻는다.

실제 적용 측면에서 이 연구는 매우 중요한 메시지를 남긴다. segmentation mask annotation이 비싸다면, 많은 양의 box annotation과 좋은 pseudo label generator를 조합하는 전략이 충분히 경쟁력 있다는 것이다. 향후 연구에서는 논문이 제안하듯 co-segmentation, 더 약한 supervision, 더 큰 데이터셋, 혹은 더 강한 backbone과 결합하는 방향으로 확장될 수 있다. 오늘날의 관점에서도, 이 논문은 **weak supervision에서 label quality가 얼마나 결정적인가**를 보여준 대표적인 사례로 읽을 가치가 있다.
