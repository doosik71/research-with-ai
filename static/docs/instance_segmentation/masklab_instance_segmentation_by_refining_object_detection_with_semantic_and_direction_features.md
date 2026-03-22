# MaskLab: Instance Segmentation by Refining Object Detection with Semantic and Direction Features

* **저자**: Liang-Chieh Chen, Alexander Hermans, George Papandreou, Florian Schroff, Peng Wang, Hartwig Adam
* **발표연도**: 2017
* **arXiv**: [https://arxiv.org/abs/1712.04837](https://arxiv.org/abs/1712.04837)

## 1. 논문 개요

이 논문은 instance segmentation 문제를 다룬다. 이 문제는 이미지 안의 각 객체를 bounding box 수준으로 검출하는 것만이 아니라, 각 객체의 정확한 픽셀 마스크까지 예측해야 하므로 object detection과 semantic segmentation을 동시에 해결해야 하는 어려운 과제이다. 저자들은 이 문제를 위해 **MaskLab**이라는 모델을 제안한다. 이 모델은 하나의 이미지로부터 세 가지 출력을 만든다. 첫째는 box detection, 둘째는 semantic segmentation logits, 셋째는 각 픽셀이 자신이 속한 객체 중심을 향해 어느 방향을 가리키는지를 나타내는 direction prediction logits이다.

논문의 핵심 문제의식은 기존 instance segmentation 방법들이 대체로 두 갈래로 나뉜다는 점이다. 하나는 detection 결과를 먼저 얻고 나중에 mask를 정교화하는 detection-based 방식이고, 다른 하나는 픽셀 단위 예측을 먼저 얻은 뒤 clustering이나 grouping으로 instance를 분리하는 segmentation-based 방식이다. 저자들은 이 둘의 장점을 결합하고자 한다. 즉, Faster R-CNN이 제공하는 강한 box localization 능력을 활용하면서도, semantic segmentation과 direction prediction이 제공하는 픽셀 수준의 구조 정보를 이용해 각 box 내부에서 더 정확한 foreground/background segmentation을 수행한다.

이 문제가 중요한 이유는 instance segmentation이 자율주행, 로보틱스, 이미지 편집, 장면 이해 같은 다양한 응용의 핵심 기반 기술이기 때문이다. 단순히 “무엇이 있는가”를 아는 것보다 “어디에 있으며 어떤 픽셀들이 그 객체에 속하는가”를 아는 것이 훨씬 풍부한 시각적 이해를 가능하게 한다. 논문은 특히 COCO benchmark에서 경쟁력 있는 성능을 보이며, detection과 segmentation을 연결하는 실용적인 설계 선택들을 체계적으로 검증했다는 점에서 의미가 있다.

## 2. 핵심 아이디어

MaskLab의 중심 아이디어는 **정확한 instance localization은 detector가 담당하고, instance mask의 세부 분리는 semantic feature와 direction feature가 담당한다**는 역할 분담이다. Faster R-CNN이 예측한 bounding box는 객체를 대략 정확히 둘러싸는 역할을 한다. 그런 다음 각 box 내부에서, semantic segmentation logits는 해당 픽셀이 어떤 semantic class에 속하는지 알려 주고, direction prediction logits는 그 픽셀이 자신이 속한 instance 중심을 향해 어느 방향을 가리키는지 알려 준다. 이 두 정보를 함께 쓰면 서로 다른 class의 객체는 semantic 정보로 구분하고, 같은 class에 속한 여러 객체는 direction 정보로 구분할 수 있다.

이 설계는 기존 FCIS와 비교했을 때 중요한 차별점을 가진다. FCIS는 inside/outside position-sensitive score maps를 사용하여 mask를 예측했는데, background를 반복적으로 인코딩해야 해서 출력 채널 수가 커지는 문제가 있었다. MaskLab은 semantic segmentation logits를 사용해 background와 class 정보를 직접 활용하므로 이런 중복을 줄인다. 또한 direction prediction은 FCIS의 position-sensitive idea와 segmentation-based 접근의 인스턴스 중심 방향 정보를 연결하는 역할을 한다. 논문은 [70]의 instance center direction 아이디어를 가져오되, 복잡한 template matching 없이 Faster R-CNN 박스를 기준으로 direction pooling을 수행해 훨씬 간단하게 instance 분리를 구현한다.

또 하나의 핵심은 단순히 주 아이디어만 제시하는 데 그치지 않고, 당시 detection/segmentation 분야에서 효과적이던 여러 설계 요소를 적극 통합했다는 점이다. 저자들은 atrous convolution으로 더 조밀한 feature map을 얻고, hypercolumn feature로 mask refinement를 수행하며, multi-grid atrous rate로 context scale을 조절하고, deformable crop and resize를 도입해 box classification 성능을 더 끌어올린다. 즉, MaskLab은 새로운 구조 하나만이 아니라, 여러 강력한 컴포넌트를 목적에 맞게 결합한 시스템 설계 논문이라고 볼 수 있다.

## 3. 상세 방법 설명

MaskLab의 backbone은 ResNet-101이다. 논문 설명에 따르면 feature는 conv4(res4x)까지 공유되고, Faster R-CNN box classifier를 위해 별도의 duplicate conv5(res5x) block이 추가된다. 반면 원래 conv5 block은 semantic segmentation과 direction prediction에 사용된다. 따라서 전체 구조는 크게 세 가지 브랜치로 이해할 수 있다. 하나는 detector branch, 하나는 semantic segmentation branch, 하나는 direction prediction branch이다.

먼저 detector branch는 Faster R-CNN에 기반하여 region proposal과 refined box prediction을 수행한다. 이 branch의 역할은 instance의 위치와 class label을 안정적으로 제공하는 것이다. 논문은 instance segmentation에서 detection이 여전히 매우 중요하다고 본다. 실제로 후반 실험에서도 segmentation branch보다 detection branch 개선이 전체 지표 향상에 더 직접적으로 기여한다고 해석한다.

그 다음 semantic segmentation branch는 이미지 전체에 대해 픽셀 단위 semantic logits를 생성한다. 이 logits는 각 픽셀이 background를 포함한 어떤 semantic class에 속하는지 표현한다. 예를 들어 predicted box가 person class라고 detector가 예측했다면, semantic logits 전체 채널 중 person 채널만 선택하고, 그 채널을 predicted box 위치에 맞게 crop한다. 이렇게 하면 box 내부 각 위치가 person일 가능성을 반영하는 class-specific regional semantic feature를 얻을 수 있다.

direction prediction branch는 각 픽셀이 자신이 속한 instance 중심을 향해 어느 방향에 있는지를 예측한다. 논문은 기본적으로 360도를 여러 개의 direction bin으로 양자화한다. 기본 설정에서는 8 방향을 사용하며, 여기에 거리(distance)도 여러 bin으로 양자화하여 최종적으로 더 세밀한 direction pooling을 수행한다. 최종 모델은 8개의 방향과 4개의 거리 bin을 사용하므로 direction 관련 채널 수는 $8 \times 4 = 32$가 된다. 이 direction feature는 같은 semantic class 안에서 서로 다른 instance를 분리하는 데 핵심 역할을 한다. 예를 들어 box 안에 person이 두 명 들어 있어도 각 픽셀이 향하는 중심 방향이 다르므로, foreground mask를 분리하는 데 도움이 된다.

논문에서 매우 중요한 연산은 **direction pooling**이다. semantic logits는 detector가 예측한 class 채널을 선택해 crop하면 되지만, direction logits는 각 direction channel에 퍼져 있는 정보를 box 내부 local representation으로 조합해야 한다. 저자들은 FCIS의 assembling operation과 유사한 아이디어를 사용해, region의 각 공간 위치가 대응되는 direction channel로부터 값을 가져와 regional logits를 구성한다. 이를 통해 box 내부에서 “이 픽셀이 어느 instance 중심을 향하는가”라는 정보를 구조적으로 모을 수 있다.

그 후, crop된 semantic logits와 direction pooling 결과를 channel-wise concatenation한 뒤, 여기에 **class-agnostic $1 \times 1$ convolution**을 적용해 foreground/background segmentation을 수행한다. 여기서 class-agnostic이라는 것은 각 semantic class마다 별도 마스크 예측기를 두지 않고, 모든 class에 공통된 가중치를 사용한다는 뜻이다. detector가 이미 class label을 예측했기 때문에, mask branch는 해당 box 안에서 foreground와 background만 구분하면 된다. 이 점은 파라미터 효율성과 일반화 측면에서 합리적인 설계다.

논문은 출력 채널 복잡도 측면의 장점도 강조한다. $K$개 semantic class가 있을 때, MaskLab은 semantic segmentation을 위해 $K$개 채널, direction pooling을 위해 32개 채널만 필요하므로 총 $K + 32$ 채널이 필요하다. 반면 FCIS는 $2 \times (K+1) \times 49$ 채널을 사용한다. 여기서 2는 inside/outside, 49는 position grid를 의미한다. 즉, MaskLab은 더 compact한 표현으로 유사한 목적을 달성하려 한다.

논문에는 별도의 **mask refinement** 단계도 있다. coarse mask는 semantic logits와 direction logits만으로도 얻을 수 있지만, 경계가 거칠 수 있다. 이를 보완하기 위해 저자들은 hypercolumn features를 사용한다. coarse mask logits를 ResNet의 낮은 수준 feature들과 concat하고, 그 뒤에 작은 ConvNet을 붙여 refined mask를 예측한다. 실험에서는 세 개의 $5 \times 5$ convolution layer, 각 64 filters를 가진 소형 네트워크를 사용했다고 명시한다. 이 refinement는 특히 세밀한 경계와 높은 IoU threshold에서 성능 향상에 기여한다.

또 하나의 중요한 기여는 **deformable crop and resize**이다. 일반적인 crop and resize는 bounding box 영역을 잘라 고정 크기로 resize한다. 논문은 이것을 더 유연하게 만들기 위해, crop된 영역을 여러 sub-box로 나눈 뒤, 작은 네트워크가 각 sub-box의 offset을 학습하게 한다. 이후 변형된 sub-box들에 대해 다시 crop and resize를 수행한다. 이는 deformable pooling과 유사한 효과를 낸다. 저자들에 따르면 이 구조는 box classification에서 object part 자체보다는 원형에 가까운 주변 context를 포착하는 방향으로 학습되는 경향이 있었다.

학습 절차 측면에서는 몇 가지 구현 세부가 중요하다. semantic과 direction branch를 위한 training에서는 groundtruth box만 사용한다. 논문은 jittered box를 쓰면 direction logits가 실제 instance center와 잘 맞지 않을 수 있기 때문이라고 설명한다. coarse mask와 refined mask 모두 sigmoid를 사용하여 예측한다. 또한 모델은 piecewise pretraining 없이 end-to-end로 학습된다. semantic과 direction feature를 결합하는 $1 \times 1$ convolution의 초기 가중치를 $(0.5, 1)$로 두어 direction feature 쪽에 더 큰 비중을 주면 학습 수렴이 빨라졌다고 보고한다. 이는 뒤의 실험 결과에서 direction feature가 semantic feature보다 더 중요하다는 사실과도 잘 맞는다.

수식 형태로 논문이 명시적 최적화 식을 자세히 적어주지는 않았지만, 구조를 개념적으로 쓰면 다음처럼 이해할 수 있다.

우선 detector가 box $b$와 class $c$를 예측한다. semantic logits를 $S \in \mathbb{R}^{H \times W \times K}$, direction logits를 $D \in \mathbb{R}^{H \times W \times C_d}$라고 하자. 여기서 $C_d$는 direction/거리 양자화 채널 수이다. 그러면 box $b$에 대한 semantic crop은
$$
S_b = \mathrm{Crop}(S_c, b)
$$
처럼 쓸 수 있다. 여기서 $S_c$는 class $c$에 대응하는 semantic channel이다.

direction feature는 단순 crop이 아니라 assembling 또는 pooling을 거쳐
$$
D_b = \mathrm{DirPool}(D, b)
$$
로 얻는다.

이 둘을 합쳐 foreground/background mask logit을
$$
M_b^{\text{coarse}} = f_{1 \times 1}([S_b ; D_b])
$$
처럼 예측한다. 여기서 $[\cdot ; \cdot]$는 채널 방향 concat이고, $f_{1 \times 1}$은 class-agnostic $1 \times 1$ convolution이다.

이후 refinement는 lower-level feature $F_{\text{low}}$를 사용하여
$$
M_b^{\text{refined}} = g([M_b^{\text{coarse}} ; F_{\text{low}}])
$$
처럼 이해할 수 있다. 여기서 $g$는 작은 refinement ConvNet이다.

이 식들은 논문 구조를 설명하기 위한 해석적 정리이며, 논문 본문에는 전체 손실식을 하나의 수식으로 명시하지는 않았다.

## 4. 실험 및 결과

실험은 COCO dataset에서 수행되었고, 구현은 TensorFlow 기반 object detection library 위에서 이루어졌다. 평가 지표는 mask IoU 기반의 mean average precision이며, 특히 $mAP@0.5$와 $mAP@0.75$를 자주 보고한다. 저자들은 먼저 minival set에서 ablation study를 통해 각 설계 요소의 효과를 검증하고, 이후 test-dev set에서 최종 성능을 보고한다.

먼저 mask crop size 실험에서는 semantic/direction feature를 crop하는 해상도가 성능에 미치는 영향을 본다. crop size를 21, 41, 81, 161, 321로 바꾸어 봤는데, $mAP@0.5$는 각각 50.92%, 51.29%, 51.17%, 51.36%, 51.24%였다. 즉, 41 이상에서는 큰 차이가 없었고, 따라서 이후 실험에서는 crop size 41을 사용한다. 이는 지나치게 큰 mask resolution이 필수는 아니라는 의미다.

semantic과 direction feature의 효과를 분리해 본 실험은 이 논문에서 매우 중요하다. semantic feature만 사용하면 $mAP@0.5 = 48.41%$, $mAP@0.75 = 24.44%$였다. direction feature만 사용하면 $mAP@0.5 = 50.21%$, $mAP@0.75 = 27.40%$로 더 높았다. 둘을 함께 쓰면 $mAP@0.5 = 51.83%$, $mAP@0.75 = 29.72%$가 된다. 여기에 direction pooling에서 distance quantization을 4 bins로 사용하면 $mAP@0.5 = 52.26%$, $mAP@0.75 = 30.57%$로 더 좋아진다. 이 결과는 두 가지를 보여 준다. 첫째, semantic과 direction은 상보적이다. 둘째, 같은 class instance를 나누는 데 direction 정보가 특히 중요하다.

direction 수 자체를 바꿔 본 실험에서는, distance bin을 4로 고정한 상태에서 directions를 2, 4, 6, 8로 바꾸었을 때 $mAP@0.75$가 각각 33.80%, 34.39%, 34.86%, 34.82%였다. 즉, 8방향이 충분하며, 그 이상 세밀한 방향 수가 꼭 필요하다고 볼 근거는 이 텍스트만으로는 없다. 저자들은 최종적으로 8 directions와 4 distance bins를 사용한다.

mask refinement 실험도 설득력 있다. refinement 없이 baseline은 $mAP@0.75 = 30.57%$였고, conv1 feature를 추가하면 32.92%, conv1과 conv2를 함께 쓰면 33.89%까지 오른다. 반면 conv3까지 더하면 오히려 32.88%로 떨어진다. 즉, 세밀한 경계 복원에는 낮은 수준의 local detail feature가 유리하고, 너무 높은 수준 feature를 추가하는 것은 도움이 되지 않을 수 있음을 보여 준다.

multi-grid 실험에서는 마지막 residual block의 atrous rate를 조절했다. box classifier 쪽 atrous rate를 다양하게 두는 것이 더 효과적이었고, semantic/direction 쪽 rate를 함께 조절하면 추가적인 소폭 개선이 있었다. 논문은 이 결과를 현재 평가 지표가 segmentation branch보다 detection branch 개선에 더 유리할 수 있다는 해석과 연결한다. 다시 말해, instance segmentation benchmark라 해도 실제 점수 향상은 detector 품질에 크게 좌우될 수 있다는 것이다.

pretraining의 효과도 크다. box classifier branch를 ImageNet pretrained weights로 초기화하면 $mAP@0.75$가 33.89%에서 34.82%로 오른다. 여기에 COCO semantic segmentation annotations로 backbone을 추가 pretraining하면 35.91%로 더 좋아진다. 이후 best multi-grid, larger anchors and 800-pixel shortest side, deformable crop and resize, random scale augmentation, JFT pretraining을 순차적으로 더하면 최종적으로 minival 기준 $mAP@0.75$는 41.59%까지 상승한다. 특히 output stride를 8에서 16으로 바꾸면 40.41%에서 38.61%로 내려가므로, atrous convolution을 통한 denser feature map이 매우 중요함을 알 수 있다.

COCO test-dev의 **mask results**를 보면, MaskLab(ResNet-101)은 mask mAP 35.4%, $mAP@0.5$ 57.4%, $mAP@0.75$ 37.4%를 기록한다. MaskLab+는 scale augmentation을 더한 버전으로 37.3%, 59.8%, 39.6%를 기록한다. JFT pretrained MaskLab+는 38.1%, 61.1%, 40.4%를 기록한다. 비교 대상으로는 FCIS, FCIS+++, Mask R-CNN이 제시된다. 논문에 따르면 MaskLab은 ResNet-101 기반 Mask R-CNN보다 좋고, ResNet-101-FPN 기반 Mask R-CNN과 유사한 수준에 도달한다. MaskLab+는 더 강한 ResNeXt-101-FPN 기반 Mask R-CNN과 유사한 mask mAP 수준을 보인다.

COCO test-dev의 **box detection results**도 인상적이다. MaskLab(ResNet-101)은 39.6% box mAP, MaskLab+는 41.9%, JFT pretrained MaskLab+는 43.0%를 기록한다. 이는 같은 표에 있는 G-RMI, TDM, 일부 Mask R-CNN 변형과 견줄 만하거나 그보다 좋은 결과다. 즉, MaskLab은 mask branch를 추가했음에도 detector 성능을 해치지 않았고, 오히려 box detection에서도 매우 경쟁력 있었다.

정성적 분석도 흥미롭다. semantic logits의 person 채널 시각화에서는 사람 주변의 elephant legs나 kite 같은 non-person 영역에도 높은 activation이 나타날 수 있다고 보고한다. 이는 semantic branch가 groundtruth boxes만으로 학습되어 hard negative를 충분히 보지 못했기 때문이라고 해석한다. 하지만 box detection branch가 잘못된 박스를 걸러 주므로 시스템 전체에서는 큰 문제를 일으키지 않는다고 설명한다. deformable crop and resize 시각화에서는 [20]처럼 object parts를 따라가려 하기보다 원형의 context를 포착하려는 형태가 나타난다. failure mode는 크게 두 가지인데, detection failure와 segmentation failure이다. detection failure에는 missed detection과 wrong class prediction이 포함되고, segmentation failure에는 coarse boundary가 포함된다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 detection-based와 segmentation-based 관점을 깔끔하게 결합했다는 점이다. Faster R-CNN의 강한 localization 능력을 유지하면서 semantic segmentation과 direction prediction을 box 내부 mask estimation에 활용하는 구조는 직관적이면서도 효과적이다. 특히 semantic feature는 다른 class와 background를 구분하고, direction feature는 같은 class instance를 분리한다는 역할 분담이 명확하다. 실험에서 semantic 단독보다 direction 단독이 더 강하고, 둘을 함께 쓸 때 가장 좋다는 결과는 이 설계의 타당성을 잘 뒷받침한다.

또 다른 강점은 모델의 표현 효율성이다. FCIS의 inside/outside position-sensitive map에 비해 MaskLab은 semantic + class-agnostic direction 채널만으로 compact한 구조를 만든다. 출력 채널 수를 줄이면서도 경쟁력 있는 성능을 얻었다는 점은 당시 기준으로 실용성이 높다. 게다가 refinement, atrous convolution, multi-grid, deformable crop and resize 같은 요소들을 체계적으로 ablation하여 어떤 요소가 실제로 성능에 기여하는지를 비교적 명확하게 보여 준다.

실험적으로도 설득력이 있다. 단순히 최종 수치만 제시한 것이 아니라, crop size, semantic/direction 분리, direction 수와 distance bin 수, refinement에 쓰는 feature level, pretraining, output stride 등 중요한 설계 선택을 폭넓게 검증했다. 덕분에 논문은 “왜 이 구조가 작동하는가”에 대한 경험적 근거를 충분히 제공한다.

하지만 한계도 분명하다. 첫째, semantic branch가 groundtruth boxes만으로 학습되기 때문에 negative region 학습이 부족하고, 따라서 non-target 영역에 activation이 뜨는 문제가 발생한다. 논문은 이를 detector가 상쇄한다고 설명하지만, 이는 시스템이 detector 품질에 강하게 의존함을 의미한다. 둘째, failure mode의 상당 부분이 detection failure에서 오므로, instance segmentation이지만 실제 병목은 detector일 가능성이 높다. 이는 곧 segmentation branch가 아무리 좋아도 detector가 놓치면 전체 성능이 제한된다는 뜻이다.

셋째, direction prediction 자체는 같은 class instance 분리에 유용하지만, 복잡한 형상, 강한 가림, 여러 중심 후보가 있는 구조에서 얼마나 안정적인지에 대한 이론적 논의는 텍스트상 충분하지 않다. 또한 본문 추출 텍스트만 보면 전체 손실 함수의 정식 수식이나 각 branch의 loss weighting이 명확히 드러나지 않는다. 따라서 학습 목적의 정확한 구성과 상호작용은 제한적으로만 파악된다.

넷째, 논문은 strong engineering integration의 성격이 강하다. 이는 장점이기도 하지만, 반대로 말하면 성능 향상이 새로운 핵심 원리 하나 때문이라기보다 여러 개선 요소의 누적 결과일 수 있다. 실제로 표 6을 보면 pretraining, scale augmentation, deformable crop and resize, JFT pretraining이 큰 폭의 성능 상승에 기여한다. 따라서 “MaskLab의 본질적 구조적 혁신”과 “강한 학습/구현 레시피의 효과”를 완전히 분리해서 보기 어렵다.

비판적으로 보면, 이 논문은 instance segmentation을 위한 매우 합리적이고 잘 구성된 시스템을 제시하지만, 이후 Mask R-CNN 계열이 보여 준 단순성과 범용성에 비해 구조가 다소 복합적이다. direction pooling, semantic channel selection, refinement, deformable crop and resize 등 여러 구성요소를 조합해야 한다. 따라서 구현 복잡도와 유지보수 측면에서는 보다 단순한 구조보다 부담이 있을 수 있다. 다만 이 평가는 후속 연구 흐름까지 고려한 해석이며, 논문 자체가 제시한 증거 범위 안에서는 충분히 경쟁력 있는 설계다.

## 6. 결론

이 논문은 instance segmentation을 위해 box detection, semantic segmentation, direction prediction을 함께 사용하는 **MaskLab**을 제안했다. 핵심은 detector가 제공하는 박스 안에서 semantic logits와 direction logits를 결합하여 foreground/background mask를 예측하는 것이다. semantic 정보는 다른 class와 background를 구분하고, direction 정보는 같은 class의 여러 instance를 분리한다. 여기에 hypercolumn 기반 refinement, atrous convolution, multi-grid, deformable crop and resize 같은 기법을 통합해 COCO benchmark에서 강한 성능을 달성했다.

논문의 주요 기여는 세 가지로 요약할 수 있다. 첫째, detection-based와 segmentation-based instance segmentation의 장점을 결합한 구조를 제시했다. 둘째, semantic과 direction feature의 역할을 분명히 나누고 이를 효율적인 pooling 방식으로 결합했다. 셋째, 다양한 구현 선택을 실험적으로 검증하여 실제 성능 향상에 효과적인 recipe를 정리했다.

실제 적용 측면에서 이 연구는 detector 중심의 instance segmentation 시스템에 픽셀 단위 구조 정보를 어떻게 넣을 수 있는지를 잘 보여 준다. 또한 같은 class instance를 분리하기 위한 representation으로 direction prediction이 유효하다는 점을 보여 주었고, 이는 이후 instance grouping이나 center-based representation 연구에도 연결될 수 있는 아이디어다. 비록 후속 분야 흐름에서는 더 단순한 구조나 다른 표현 방식들이 널리 쓰이게 되었지만, 이 논문은 detection과 dense prediction을 결합하는 하나의 설계 원형으로서 충분한 의미가 있다.
