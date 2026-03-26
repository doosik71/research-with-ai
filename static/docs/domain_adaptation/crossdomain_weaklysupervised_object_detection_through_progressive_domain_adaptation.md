# Cross-Domain Weakly-Supervised Object Detection through Progressive Domain Adaptation

* **저자**: Naoto Inoue, Ryosuke Furuta, Toshihiko Yamasaki, Kiyoharu Aizawa
* **발표연도**: 2018
* **arXiv**: [https://arxiv.org/abs/1803.11365](https://arxiv.org/abs/1803.11365)

## 1. 논문 개요

이 논문은 **source domain에서는 bounding box가 포함된 instance-level annotation을 충분히 가지고 있지만, target domain에서는 image-level annotation만 있는 상황**에서 object detection을 어떻게 수행할 것인가를 다룬다. 저자들은 이 문제를 **cross-domain weakly supervised object detection**이라는 새로운 과제로 정의한다. 예를 들어 natural image에서 잘 학습된 detector가 있지만, watercolor나 comic 같은 다른 시각적 도메인에서는 박스 주석이 없고 이미지에 어떤 클래스가 있는지만 아는 상황이 대표적이다.

연구 문제는 단순한 weakly supervised detection과도 다르고, 일반적인 domain adaptation과도 다르다. 일반적인 weakly supervised detection은 처음부터 박스 없이 detector를 학습해야 하므로 localization이 약하다. 반면 이 논문은 source domain에서 이미 강력한 fully supervised detector를 가지고 시작한다. 또한 일반적인 unsupervised domain adaptation은 target domain에 아예 label이 없다고 가정하는 경우가 많지만, 이 논문은 target domain에 최소한의 image-level annotation은 있다고 본다. 따라서 이 문제는 **강한 source supervision과 약한 target supervision을 결합하여 target-domain detector를 만드는 문제**로 볼 수 있다.

이 문제가 중요한 이유는 실제 응용에서 새로운 도메인마다 bounding box를 대규모로 수집하는 것이 매우 비싸기 때문이다. 자연 이미지에서는 VOC, COCO 같은 데이터셋이 잘 갖춰져 있지만, clipart, watercolor, comic 같은 비자연 이미지 도메인에서는 이미지 수집 자체도 어렵고, 저작권 이슈나 annotation 비용도 크다. 반면 image-level label은 기존 검색 엔진이나 약한 태깅 데이터셋에서 상대적으로 쉽게 얻을 수 있다. 따라서 이 논문은 “새 도메인마다 detector를 다시 full supervision으로 만들기 어려운 현실”을 직접 겨냥한다.

논문의 핵심 목표는 다음과 같이 요약할 수 있다. 먼저 source domain에서 fully supervised detector를 준비한다. 그 다음 target domain에 가까운 인공 샘플과 자동 생성 pseudo annotation을 이용해 detector를 점진적으로 적응시킨다. 저자들은 이를 통해 별도의 target-domain bounding box 없이도 target-domain detection 성능을 유의미하게 향상시키고자 한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 **target domain에서 instance-level annotation이 없더라도, detector를 fine-tuning할 수 있는 “대체 학습 데이터”를 두 단계에 걸쳐 만들어내자**는 것이다. 이 대체 데이터는 성질이 서로 다른 두 종류로 구성된다.

첫 번째는 **Domain Transfer (DT)** 이다. source domain의 이미지와 bounding box annotation은 그대로 가지고 있으므로, 이미지의 “스타일”만 target domain처럼 바꾸면 target-domain-like 학습 샘플을 만들 수 있다. 예를 들어 VOC 자연 이미지의 개, 자동차, 사람이 watercolor 스타일로 바뀌더라도, 원래 bounding box는 여전히 유효하다. 이렇게 하면 완벽한 target 이미지와는 다를 수 있지만, 적어도 박스 annotation은 정확하다.

두 번째는 **Pseudo-Labeling (PL)** 이다. 이번에는 실제 target-domain 이미지를 사용한다. 다만 target에는 bounding box가 없으므로, 현재 detector의 예측을 이용해 pseudo box를 생성한다. 이때 단순히 detector가 낸 모든 박스를 쓰는 것이 아니라, **이미지 수준에서 주어진 클래스 label을 이용하여 그 클래스에 해당하는 top-1 detection만 채택**한다. 즉, 이미지에 “dog”와 “person”이 있다고 알려져 있으면, detector가 예측한 dog 박스 중 가장 confidence가 높은 것 하나, person 박스 중 가장 confidence가 높은 것 하나를 pseudo annotation으로 삼는다. 이렇게 하면 class confusion을 줄이면서 실제 target-domain image에 대해 학습할 수 있다.

이 논문이 제안하는 차별점은 단순히 DT와 PL을 병렬로 쓰는 것이 아니라, **progressive하게 순차 적용**한다는 점이다. 먼저 source에서 pre-train된 detector를 DT 이미지로 한 번 적응시키고, 그 다음 그 detector를 이용해 더 나은 pseudo label을 생성한 뒤 PL로 다시 fine-tune한다. 저자들은 이 순서가 중요하다고 명시한다. 이유는 PL의 품질이 현재 detector의 성능에 직접 의존하기 때문이다. 즉, 초기 detector가 너무 source-biased 상태이면 target-domain pseudo label의 질이 낮아지고, 그 위에서 다시 학습하면 오히려 성능이 나빠질 수 있다. DT가 먼저 들어가면 low-level appearance gap을 완화해 주고, 그 결과 PL 단계에서 더 나은 pseudo annotation이 가능해진다.

이 직관은 실험 결과와도 잘 맞는다. DT는 주로 도메인 차이에서 오는 저수준 feature 차이를 줄이고, PL은 target 이미지의 실제 클래스 정보로 잘못된 고신뢰 오탐을 줄이는 역할을 한다. 저자들의 error analysis에 따르면 DT 이후에는 전반적인 detection이 좋아지고, PL 이후에는 특히 비슷한 다른 클래스나 배경으로의 혼동이 줄어든다. 따라서 두 단계는 경쟁 관계가 아니라 **상호보완적**이다.

## 3. 상세 방법 설명

전체 파이프라인은 세 단계로 이루어진다.

첫째, source domain의 instance-level annotation을 사용해 fully supervised detector를 준비한다. 실험에서는 주로 SSD300을 baseline detector로 사용했으며, VOC2007-trainval과 VOC2012-trainval로 사전학습된 모델을 사용했다. 이 단계는 일반적인 object detector 학습과 같다.

둘째, **Domain Transfer (DT)** 단계에서 source 이미지를 target 스타일로 변환한다. 여기서는 unpaired image-to-image translation인 **CycleGAN**을 사용한다. source domain을 $\mathcal{X}_s$, target domain을 $\mathcal{X}_t$라고 할 때, CycleGAN은 source에서 target으로 가는 매핑 $G:\mathcal{X}_s \rightarrow \mathcal{X}_t$와 target에서 source로 가는 역매핑 $F:\mathcal{X}_t \rightarrow \mathcal{X}_s$를 학습한다. 중요한 점은 source와 target 이미지가 pair로 정렬되어 있을 필요가 없다는 것이다. 학습이 끝나면 source 이미지를 $G$를 통해 target-like 이미지로 변환하고, 원래 source 이미지가 가지고 있던 bounding box annotation을 그대로 붙여서 detector fine-tuning에 사용한다.

논문은 CycleGAN의 손실식을 상세히 전개하지는 않았고, 핵심 개념만 사용한다. 따라서 여기서 저자가 직접 설명한 수준에 맞춰 이해하면, DT의 목적은 객체의 의미나 대략적인 구조는 유지하면서 색감, 질감, 스타일을 target domain처럼 바꾸는 것이다. 논문에서도 완벽한 mapping은 아니라고 인정한다. 특히 natural image와 watercolor/comic/clipart 사이의 표현 차이가 매우 크기 때문에, synthetic-to-real 적응보다 더 어려운 문제라고 본다. 그럼에도 불구하고 색과 texture를 옮기고 edge와 semantics를 어느 정도 보존하는 것만으로도 detector 적응에 도움이 된다고 주장한다.

셋째, **Pseudo-Labeling (PL)** 단계에서 target-domain image-level annotation을 활용해 pseudo instance annotation을 만든다. 이 부분은 논문에서 비교적 명시적으로 수식화되어 있다.

입력 이미지를 $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$라 하고, 클래스 집합을 $\mathcal{C}$라 하자. 이미지 수준 annotation $\mathbf{z}$는 이미지에 존재하는 클래스들의 집합이다. 생성하려는 pseudo instance annotation 집합을 $\mathbf{G}$라 하면, 그 원소는 $g=(\mathbf{b}, c)$ 형태이다. 여기서 $\mathbf{b} \in \mathbb{R}^{4}$는 bounding box이고, $c \in \mathcal{C}$는 클래스이다.

먼저 detector의 출력 집합을 $\mathbf{D}$라고 두면, 각 detection은 $d=(p, \mathbf{b}, c)$로 표현된다. 여기서 $p \in \mathbb{R}$는 박스 $\mathbf{b}$가 클래스 $c$일 확률 또는 confidence score를 의미한다. 그런 다음, 이미지-level label 집합 $\mathbf{z}$에 포함된 각 클래스 $c$에 대해 detector 출력 중 같은 클래스 $c$를 가진 detection들 가운데 **가장 confidence가 높은 top-1 detection 하나**를 선택한다. 즉, 각 $c \in \mathbf{z}$에 대해 top-1 detection $d=(p,\mathbf{b},c) \in \mathbf{D}$를 뽑고, pseudo label 집합 $\mathbf{G}$에 $(\mathbf{b}, c)$를 추가한다. 이후 detector는 $(\mathbf{x}, \mathbf{G})$ 쌍으로 fine-tuning된다.

이를 식으로 쓰면 개념적으로 다음과 같이 볼 수 있다.

$$
\mathbf{G} = \left \{(\mathbf{b}_c, c) \mid c \in \mathbf{z},; (\hat p_c, \mathbf{b}_c, c) = \operatorname*{\arg\max}_{(p,\mathbf{b},c)\in \mathbf{D}} \; p \right \}
$$

이 수식은 논문의 설명을 정리한 표현이며, 원문도 사실상 같은 절차를 서술한다. 중요한 점은 클래스당 하나의 박스만 사용한다는 것이다. 이 설계는 매우 단순하지만, target image-level label을 활용해 명백한 class confusion을 줄이는 데 효과적이다. 동시에 한계도 있다. 이미지 안에 같은 클래스의 객체가 여러 개 있으면 나머지는 label이 없는 배경처럼 취급될 수 있다. 저자들도 이 점을 discussion에서 명시적으로 한계로 언급한다.

이 방법의 또 다른 특징은 **detector 내부 구조를 바꾸지 않는다는 점**이다. 논문은 pseudo-labeling을 위해 detector의 intermediate layer를 수정하거나 별도의 branch를 붙이지 않는다. 즉, 어떤 fully supervised detector라도 출력 detection만 읽을 수 있으면 적용 가능하다. 실제로 SSD300뿐 아니라 YOLOv2, Faster R-CNN에도 실험했다. 이는 방법의 일반성과 구현 단순성을 높이는 장점이다.

훈련 절차를 정리하면 다음과 같다. 먼저 source domain에서 detector를 준비한다. 다음으로 DT로 생성한 target-like 이미지와 원래 박스를 사용해 1차 fine-tuning을 한다. 마지막으로 그 detector를 이용해 target-domain image-level labeled image에서 pseudo box를 생성하고 2차 fine-tuning을 한다. 논문은 이 순차성이 중요하다고 강조한다. DT 없이 바로 PL을 하면 초기 detector의 target-domain 인식이 약해서 pseudo label 품질이 떨어질 수 있기 때문이다.

손실 함수 자체는 detector마다 원래 사용하던 supervised detection loss를 그대로 사용한다고 보는 것이 맞다. 논문은 별도의 새로운 detection loss를 제안하지 않는다. SSD300을 예로 들면 localization loss와 classification loss를 포함한 기존 SSD 학습 목표를 pseudo-labeled sample이나 DT sample에 대해 동일하게 적용하는 구조다. 다만 이 논문은 그 수식을 직접 적어 설명하지는 않으므로, 구체적 loss decomposition은 원문에서 본 논문의 독자적 기여로 제시되지는 않는다.

## 4. 실험 및 결과

실험은 세 개의 새 데이터셋과 하나의 source 데이터셋을 중심으로 이루어진다. source domain은 **PASCAL VOC**이며, 20개 클래스를 포함하는 natural image domain이다. target domain은 논문이 새롭게 구축한 **Clipart1k**, **Watercolor2k**, **Comic2k**이다. 각각 clipart 1,000장, watercolor 2,000장, comic 2,000장으로 구성되어 있다. Clipart1k는 20개 클래스, Watercolor2k와 Comic2k는 6개 클래스만 대상으로 한다. 전체적으로 5,000장 이미지와 12,869개의 instance-level annotation을 수집했다고 한다.

이 데이터셋 구성은 논문의 중요한 공헌 중 하나다. 기존 cross-domain detection용 데이터셋은 single-class이거나 이미지당 객체가 하나뿐인 등 현실성이 떨어졌는데, 이 논문은 여러 객체와 복잡한 배경을 포함하는 보다 실제적인 benchmark를 제공하려고 했다. Appendix에 따르면 Clipart1k는 이미지당 평균 1.7개 클래스, 3.2개 인스턴스를 포함하여 VOC와 유사한 난이도를 갖는다. Watercolor2k는 평균 1.1개 클래스, 1.7개 인스턴스, Comic2k는 평균 1.1개 클래스, 3.2개 인스턴스를 가진다.

평가는 AP와 mAP로 수행되었다. source domain 학습에는 VOC2007-trainval과 VOC2012-trainval을 사용했다. target-domain dataset은 절반을 train, 절반을 test로 나누었고, train split에서는 bounding box를 무시하여 실제 제안 상황을 흉내 냈다. test split에서는 bounding box를 사용해 성능을 측정했다.

비교 대상은 다음과 같다. 먼저 baseline으로 SSD300을 사용했다. Weakly supervised detector로는 WSDDN과 CLNet을 썼다. Unsupervised domain adaptation으로는 ADDA를 썼다. 또 SSD300, CLNet, WSDDN의 ensemble도 비교했다. 마지막으로 target-domain training split의 instance-level annotation을 실제로 사용한 **Ideal case**를 통해 약한 upper bound를 제시했다.

### Clipart1k 결과

Clipart1k에서 baseline SSD300의 mAP는 **26.8**이었다. WSDDN은 **4.4**, CLNet은 **7.8**로 매우 낮았다. 이는 이미지-level supervision만으로 처음부터 detection을 학습하는 방식이 localization에 취약하고, 데이터도 충분하지 않기 때문으로 해석할 수 있다. Ensemble도 **26.7**로 baseline보다 거의 나아지지 않았고, ADDA는 **27.4**로 미미한 개선만 보였다.

제안 방법은 훨씬 큰 향상을 보였다. PL만 사용하면 mAP가 **36.4**, DT만 사용하면 **38.0**, DT와 PL을 모두 사용한 **DT+PL**은 **46.0**까지 올랐다. baseline 대비 **19.2 percentage points** 향상이며, Ideal case의 **55.4**와 비교해도 차이가 **9.4 points**에 불과하다. 이는 target-domain bounding box가 전혀 없다는 점을 생각하면 상당히 강한 결과다.

클래스별 결과를 보면 모든 클래스가 고르게 오르는 것은 아니다. 예를 들어 Clipart1k에서 DT+PL은 motorbike에서 83.3, bus에서 74.0, cow에서 72.7처럼 큰 향상을 보이지만, cat은 여전히 2.8로 매우 낮다. 즉, 도메인 적응이 전체적으로 유효하더라도 클래스별 난이도 차이는 여전히 남는다. 특히 시각적 변형이 심하거나 training signal이 약한 클래스는 어려운 것으로 보인다.

논문은 ablation도 제공한다. target domain에 image-level annotation이 아예 없다고 가정하면, DT는 그대로 적용 가능하지만 PL은 불가능하다. 이를 대신해 전체 detection 중 가장 confidence가 높은 단 하나의 detection만 pseudo-label로 쓰는 변형을 시도했는데, **PL w/o label**은 **25.3**, **DT+PL w/o label**은 **32.7**에 그쳤다. DT+PL의 46.0에 비하면 큰 차이가 난다. 즉, target domain에서 image-level annotation은 단순한 보조 정보가 아니라 PL의 성능을 좌우하는 핵심 signal이라고 해석할 수 있다.

### 다른 detector에 대한 일반성

Clipart1k에서 SSD300 외에 YOLOv2, Faster R-CNN도 실험했다. baseline mAP는 각각 YOLOv2가 **25.5**, Faster R-CNN이 **26.2**였다. DT+PL 적용 시 YOLOv2는 **39.9**, Faster R-CNN은 **34.9**가 되었다. SSD300은 **46.0**이었다. 즉, 세 detector 모두 향상되므로 방법은 detector-agnostic한 성격을 가진다. 다만 향상 폭은 SSD300이 가장 컸다.

저자들은 이를 pseudo-labeled annotation의 노이즈와 불완전성에 대해 SSD300의 data augmentation, 특히 zoom-in/zoom-out 등이 더 강인하게 작용했기 때문일 수 있다고 해석한다. 이는 흥미로운 관찰이다. 단지 adaptation 방법만 중요한 것이 아니라, noisy supervision을 견디는 detector의 학습 특성도 최종 성능에 큰 영향을 준다는 뜻이다.

### Error analysis

논문은 Hoiem 등의 분석 툴을 사용해 detection error를 다섯 종류로 나눈다. 정답 클래스이면서 IoU가 0.5보다 크면 Correct, 정답 클래스이지만 박스가 어긋나 IoU가 0.1에서 0.5 사이라면 Localization, 다른 클래스이지만 같은 category면 Similar, 다른 category면 Other, 모든 객체와 IoU가 0.1 미만이면 Background로 분류한다.

Clipart1k 시각화 결과를 보면 baseline에서 DT로 갈 때는 특히 **덜 자신 있는 detection들에서 recall이 늘어나는 경향**이 나타난다. 이는 domain appearance gap이 줄어들면서 detector가 target-domain object를 더 자주 찾기 시작했다는 뜻으로 읽을 수 있다. 반면 DT에서 DT+PL로 갈 때는 **Similar와 Other 같은 class confusion error가 크게 감소**한다. 이 결과는 PL이 image-level label을 이용해 “이 이미지에 없는 클래스”로의 혼동을 줄여준다는 논문의 주장과 잘 부합한다.

### Watercolor2k와 Comic2k 결과

Watercolor2k에서는 baseline SSD300이 이미 **49.6**으로 비교적 높다. 여기에 PL은 **54.0**, DT는 **50.4**, DT+PL은 **54.3**을 기록했다. Clipart1k만큼 dramatic하진 않지만 확실한 개선이다. 흥미롭게도 Watercolor2k에서는 **PL(+extra)** 와 **DT+PL(+extra)** 가 모두 **59.1**에 도달해 Ideal case의 **58.4**를 약간 넘는다. 여기서 +extra는 BAM!에서 가져온 다수의 noisy image-level labeled image를 추가로 pseudo-labeling해 사용한 경우다. 즉, 소량의 깨끗한 box annotation보다 대량의 약한 label이 더 유용할 수도 있음을 보여준다.

Comic2k에서는 baseline이 **24.9**로 낮고, 제안 방법의 효과가 더 두드러진다. PL은 **32.9**, DT는 **29.8**, DT+PL은 **37.2**를 기록했다. +extra를 쓰면 DT+PL(+extra)는 **42.2**까지 올라가며, Ideal case **46.4**에 꽤 근접한다. 이 역시 대량의 noisy image-level label이 실제적으로 매우 유용할 수 있다는 점을 보여준다.

### 정성적 결과와 생성 이미지 해석

논문은 DT로 생성된 이미지 예시도 보여준다. 저자들은 CycleGAN이 **mode collapse는 일으키지 않았지만 완벽한 mapping도 아니다**라고 솔직하게 말한다. 실제로 style transfer 결과는 natural image를 완전히 watercolor나 comic으로 재창조하는 수준은 아니며, 색상과 texture를 주로 바꾸는 편이다. 그럼에도 불구하고 detector fine-tuning에는 충분히 쓸모가 있었고, 표의 성능 향상이 이를 뒷받침한다고 주장한다.

또한 qualitative detection examples를 통해 다양한 depiction style에 대해 방법이 유효함을 보인다. Appendix에서는 전형적인 실패 사례도 제시하는데, 작은 객체를 놓치거나, 많이 겹친 같은 클래스 객체 둘을 하나로 합치거나, 객체의 가장 discriminative한 부분만 박스로 잡거나, 심하게 변형된 객체를 인식하지 못하는 경우가 대표적이다. 이는 weak supervision과 domain gap이 여전히 남아 있는 상황에서 자연스러운 한계로 볼 수 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정 자체가 현실적이고 명확하다는 점이다. source domain에는 박스가 많고, target domain에는 image-level label만 있는 상황은 실제 서비스나 산업 환경에서 충분히 자주 발생한다. 완전히 unsupervised도 아니고 완전히 supervised도 아닌 이 중간 지대를 잘 정의했다는 점이 의미 있다.

두 번째 강점은 방법이 매우 단순하면서도 효과적이라는 점이다. Domain Transfer와 Pseudo-Labeling 모두 개별적으로는 이해하기 쉬운 아이디어이며, detector 내부 구조를 크게 바꾸지 않는다. 복잡한 adversarial alignment나 구조 변경 없이도 성능 향상을 얻는다. 특히 PL은 intermediate feature를 건드리지 않고 detector 출력만 사용하므로 다른 detector로 확장하기 쉽다.

세 번째 강점은 두 단계의 역할이 분리되어 있다는 점이다. DT는 appearance gap을 줄이고, PL은 class confusion을 줄인다. 실험과 error analysis가 이 역할 분담을 꽤 설득력 있게 보여준다. 단순히 결과가 좋아졌다고만 말하는 것이 아니라, 왜 좋아졌는지 오류 유형 분석까지 제공한다는 점이 좋다.

네 번째 강점은 데이터셋 구축이다. Clipart1k, Watercolor2k, Comic2k는 이 논문의 방법 평가를 위한 도구일 뿐 아니라, 이후의 domain adaptation이나 weakly supervised detection 연구에도 활용 가능한 benchmark다. 특히 기존의 단일 객체 중심 데이터셋과 달리 여러 객체와 복잡한 배경을 포함한다는 점이 가치 있다.

반면 한계도 분명하다. 가장 중요한 한계는 PL에서 **클래스당 top-1 bounding box 하나만 사용한다는 점**이다. 이미지에 같은 클래스 객체가 여러 개 있어도 하나만 긍정 예제로 취급되고 나머지는 사실상 supervision에서 빠진다. 저자들도 이것이 미래 과제라고 직접 언급한다. 다중 인스턴스가 많은 이미지에서는 이 제약이 큰 손실일 수 있다.

또한 pseudo label의 localization 품질이 충분히 높지 않다. 논문도 target-domain detector의 실패 원인을 “주로 class confusion이지 localization은 아니다”라고 서술하지만, Appendix와 error analysis를 보면 localization 문제 역시 남아 있다. discriminative part만 박스로 잡거나 overlapping object를 합쳐버리는 문제는 weak supervision 계열의 고질적 한계와 닿아 있다.

Domain Transfer에도 한계가 있다. CycleGAN이 low-level appearance를 어느 정도 옮기더라도, natural image와 comic/watercolor 사이의 **semantic rendering gap**까지 완전히 해결하지는 못한다. 저자들 자신도 완벽한 mapping은 아니라고 인정한다. 즉, DT는 좋은 보조 수단이지 target-domain realism을 정확히 복제하는 도구는 아니다.

또 하나의 해석 포인트는 비교 baseline의 강도다. 논문은 ADDA 같은 distribution-matching UDA가 잘 안 된다고 보이지만, object detection에 대한 UDA 기법들은 이후 더욱 발전했다. 따라서 오늘의 기준으로 보면 “adversarial feature alignment는 다 약하다”는 식의 일반화는 조심해야 한다. 다만 이 논문이 출간된 시점에서는 detector adaptation에 structured output 문제가 크고, 단순 feature alignment가 충분치 않다는 문제의식은 타당하다.

비판적으로 보면, 이 논문은 방법의 단순성과 일반성은 뛰어나지만, pseudo-label selection의 정교함이나 detector-specific adaptation의 깊이는 제한적이다. 즉, 강력한 baseline paper로서의 성격이 강하고, 이후 연구에서는 multi-instance pseudo-labeling, iterative refinement, consistency learning, stronger domain translation 등으로 충분히 확장될 여지가 크다. 실제로 저자들도 MIL 기반 개선 가능성을 discussion에서 언급한다.

## 6. 결론

이 논문은 **cross-domain weakly supervised object detection**이라는 새로운 문제를 제안하고, 이를 해결하기 위한 간단하면서도 강력한 프레임워크를 제시했다. 핵심은 source-domain fully supervised detector를 출발점으로 삼아, 먼저 **Domain Transfer**로 target-like labeled image를 만들고, 이어서 **Pseudo-Labeling**으로 실제 target-domain image를 활용하는 **두 단계 progressive adaptation**이다.

실험 결과는 이 접근이 단순한 baseline detector, weakly supervised detector, 기존 UDA 방법보다 훨씬 효과적임을 보여준다. 특히 Clipart1k에서는 baseline 26.8 mAP를 46.0까지 끌어올렸고, Watercolor2k와 Comic2k에서도 일관된 개선을 보였다. 더 나아가 noisy image-level label이 대량으로 있을 때는 일부 설정에서 Ideal case를 능가하는 결과도 나왔다. 이는 고비용 bounding box annotation 없이도 강한 detector 적응이 가능하다는 실질적 가능성을 시사한다.

실제 적용 측면에서 이 연구는 새로운 시각 도메인, 새로운 콘텐츠 스타일, 혹은 annotation 비용이 큰 환경에서 매우 유의미하다. 완전한 target-domain annotation 없이도 일정 수준 이상의 detector를 빠르게 구축할 수 있기 때문이다. 향후 연구에서는 다중 인스턴스를 더 잘 다루는 pseudo-labeling, 더 정교한 localization refinement, 더 강한 image translation, 반복적 self-training과 결합한 방식 등이 자연스러운 확장 방향이 될 것이다.

종합하면, 이 논문은 복잡한 이론적 장치보다는 **현실적인 문제 정의, 구현 가능한 방법, 분명한 성능 개선, 새 benchmark 구축**이라는 네 가지 측면에서 가치가 크다. 특히 “약한 supervision이 있지만 전혀 없는 것은 아닌” 실제 도메인 적응 문제를 다루는 출발점으로서 의미 있는 논문이다.
