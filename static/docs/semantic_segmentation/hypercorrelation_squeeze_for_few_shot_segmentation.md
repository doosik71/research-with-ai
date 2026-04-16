# Hypercorrelation Squeeze for Few-Shot Segmentation

- **저자**: Juhong Min, Dahyun Kang, Minsu Cho
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2104.01538

## 1. 논문 개요

이 논문은 few-shot semantic segmentation 문제를 다룬다. 즉, 새로운 클래스에 대해 아주 적은 수의 annotated support image와 mask만 주어졌을 때, query image 안에서 그 객체를 정확히 분할하는 것이 목표다. 일반적인 semantic segmentation은 대량의 pixel-level annotation에 크게 의존하지만, few-shot setting에서는 그런 데이터가 충분하지 않기 때문에 기존의 fully supervised 방식이 그대로 통하지 않는다.

저자들은 이 문제의 핵심을 “support와 query 사이에서 신뢰할 수 있는 correspondence를 얼마나 잘 찾고, 또 그것을 얼마나 잘 해석하느냐”로 본다. 특히 few-shot segmentation에서는 단순히 클래스 prototype 하나를 만드는 것만으로는 공간적 구조와 세밀한 경계 정보를 충분히 보존하기 어렵다. 따라서 support와 query의 feature map 사이의 dense correspondence를 더 직접적으로 모델링할 필요가 있다.

이 논문의 중요성은 두 가지다. 첫째, 제한된 supervision 하에서 segmentation 성능을 높이는 실용적 가치가 있다. 둘째, visual correspondence 연구에서 효과적이었던 multi-level feature 활용과 4D convolution을 few-shot segmentation에 본격적으로 접목했다는 점에서 방법론적 의미가 있다. 논문은 이를 위해 HSNet(Hypercorrelation Squeeze Networks)를 제안하며, PASCAL-5$^i$, COCO-20$^i$, FSS-1000에서 state-of-the-art 성능을 보고한다.

## 2. 핵심 아이디어

핵심 아이디어는 support-query 쌍 사이의 관계를 하나의 단순한 similarity map으로 축약하지 않고, 여러 intermediate layer에서 얻은 feature들로부터 다층적 상관관계 집합을 만든 뒤 이를 직접 학습적으로 해석하자는 것이다. 저자들은 이를 hypercorrelation이라고 부른다. 이것은 서로 다른 깊이의 CNN layer들이 제공하는 semantic cue와 geometric cue를 동시에 담는 4D correlation tensor들의 모음이다.

이 논문의 중심 직관은 다음과 같다. 깊은 layer의 feature는 객체의 semantic identity를 잘 담지만 해상도가 낮고, 얕은 layer의 feature는 경계와 위치 같은 geometric detail을 더 잘 담는다. 따라서 few-shot segmentation에서는 이 둘을 함께 써야 한다. HSNet은 FPN과 유사한 pyramidal 구조를 사용해 상위 레벨의 semantic correlation과 하위 레벨의 geometric correlation을 coarse-to-fine 방식으로 결합한다.

기존 접근과의 차별점도 분명하다. prototype-based 방법들은 masked average pooling 등을 통해 support 정보를 하나의 representative vector로 요약하는 경향이 있는데, 이 과정에서 spatial structure가 손실될 수 있다. 일부 graph-based 또는 dense correlation 기반 방법도 있었지만, 논문에 따르면 많은 방법이 intermediate layer 전반을 폭넓게 활용하지 않거나, dense pairwise correlation을 충분히 정교하게 처리하지 못했다. HSNet은 multi-level dense correlation 자체를 학습 대상의 중심에 놓고, 이를 4D convolution으로 분석한다는 점에서 다르다.

또 하나의 중요한 기여는 center-pivot 4D convolution이다. 일반 4D convolution은 계산량과 메모리 요구량이 매우 크다. 저자들은 4D 커널의 많은 weight가 실제로는 크게 중요하지 않을 수 있다고 보고, 중심 위치를 축으로 한 일부 이웃만 남기는 sparsification을 도입했다. 이를 통해 4D correlation 처리의 핵심 구조는 유지하면서도 계산을 크게 줄였다.

## 3. 상세 방법 설명

전체 구조는 세 부분으로 이루어진다. 첫째는 hypercorrelation construction, 둘째는 4D-convolutional pyramid encoder, 셋째는 2D-convolutional context decoder이다. 전체 입력은 query image $I^q$, support image $I^s$, 그리고 support mask $M^s$이며, 출력은 query mask prediction이다.

먼저 backbone CNN이 query와 support에서 여러 intermediate feature map 쌍 $\{(F_l^q, F_l^s)\}_{l=1}^L$를 뽑는다. backbone으로는 VGG16, ResNet50, ResNet101을 사용하며, ImageNet pretrained model을 그대로 feature extractor로 쓰고 학습 중에는 freeze한다. 이 결정은 뒤의 ablation에서 과적합 방지에 유리한 것으로 분석된다.

support feature는 support mask를 이용해 foreground 중심으로 걸러진다. 논문은 이를 다음과 같이 쓴다.

$$
\hat{F}_l^s = F_l^s \odot \zeta_l(M^s)
$$

여기서 $\odot$는 Hadamard product이고, $\zeta_l(\cdot)$는 support mask를 해당 layer feature의 spatial size에 맞게 bilinear interpolation한 뒤 channel 방향으로 확장하는 함수다. 이 과정의 목적은 support image 안의 irrelevant activation을 제거해 query-support correspondence를 더 신뢰성 있게 만드는 것이다.

그 다음 query feature와 masked support feature 사이의 모든 위치 쌍에 대해 cosine similarity를 계산하여 4D correlation tensor를 만든다.

$$
\hat{C}_l(x^q, x^s) =
\mathrm{ReLU}
\left(
\frac{F_l^q(x^q)\cdot \hat{F}_l^s(x^s)}
{\|F_l^q(x^q)\| \|\hat{F}_l^s(x^s)\|}
\right)
$$

여기서 $x^q$와 $x^s$는 각각 query와 support feature map 상의 2D spatial position이다. ReLU는 noisy correlation score를 억제하는 역할을 한다. 이렇게 하면 각 layer마다 $H_l \times W_l \times H_l \times W_l$ 형태의 4D tensor가 생긴다.

이후 spatial size가 같은 correlation tensor들을 channel 방향으로 concatenate해서 pyramidal layer별 hypercorrelation $C_p$를 만든다. 따라서 $C_p$는 단일 layer correlation이 아니라, 같은 resolution을 공유하는 여러 intermediate layer correlation들의 집합이다. 이 점이 논문의 multi-level relation modeling의 핵심이다.

다음 단계는 4D-convolutional pyramid encoder이다. 입력은 hypercorrelation pyramid $\mathcal{C} = \{C_p\}_{p=1}^P$이고, 출력은 query spatial domain에 정렬된 2D context feature $Z \in \mathbb{R}^{128 \times H_1 \times W_1}$이다. 이 encoder는 두 종류의 block으로 구성된다.

첫 번째는 squeezing block $f_p^{sqz}$이다. 이 블록은 4D convolution, group normalization, ReLU를 세 번 쌓은 구조다. 중요한 점은 stride를 이용해 마지막 두 개의 spatial dimension, 즉 support 쪽 좌표축만 점차 줄인다는 것이다. 반면 처음 두 개의 dimension, 즉 query 쪽 좌표축은 유지된다. 즉, correlation tensor 안에 담긴 support-query 관계를 query 중심 표현으로 점차 압축하는 셈이다.

두 번째는 mixing block $f_p^{mix}$이다. 이것도 4D convolution 기반 블록이며, FPN처럼 상위 pyramid layer 출력을 query dimension 쪽으로 upsampling한 뒤 인접한 하위 layer 출력과 element-wise addition으로 결합한다. 이후 mixing block이 이를 정제한다. 결과적으로 상위층의 semantic cue가 하위층의 finer geometric cue와 top-down 방식으로 합쳐진다.

최종적으로 가장 낮은 level의 mixing block 출력에서 support spatial dimension 두 개를 average pooling하여 제거하면 2D context map $Z$가 얻어진다. 이 단계는 관계 표현을 query 위치별 condensed representation으로 바꾸는 과정이라고 볼 수 있다.

그 다음 2D-convolutional context decoder가 $Z$를 받아 최종 segmentation map을 만든다. decoder는 일련의 2D convolution, ReLU, upsampling, 마지막 softmax로 구성된다. 출력은 두 채널의 확률 맵 $\hat{M}^q \in [0,1]^{2 \times H \times W}$이며, foreground와 background 확률을 의미한다. 학습 시에는 query ground truth $M^q$와 prediction 사이의 pixel-wise cross-entropy loss 평균을 사용한다. 추론 시에는 각 pixel에서 더 큰 확률을 갖는 채널을 선택해 최종 binary mask $\bar{M}^q$를 만든다.

이 논문의 방법론적 핵심 중 하나는 center-pivot 4D convolution이다. 일반적인 4D convolution은 다음처럼 정의된다.

$$
(c * k)(x, x') =
\sum_{(p,p') \in \mathcal{P}(x,x')}
c(p,p')\, k(p-x, p'-x')
$$

여기서 $\mathcal{P}(x,x') = \mathcal{P}(x)\times\mathcal{P}(x')$는 local 4D neighborhood다. 문제는 이 연산이 매우 비싸고, high-dimensional kernel이 과도하게 많은 parameter를 가져 numerical instability나 비효율을 낳을 수 있다는 점이다.

저자들은 중심 위치 $(x, x')$를 기준으로, query 쪽 중심 $x$를 고정한 채 support 쪽 이웃만 보거나, support 쪽 중심 $x'$를 고정한 채 query 쪽 이웃만 보는 두 종류의 subset만 남긴다. 이를 각각 $\mathcal{P}_c(x,x')$와 $\mathcal{P}_{c'}(x,x')$로 정의하고, union을 center-pivot neighborhood로 둔다. 그러면 center-pivot 4D convolution은

$$
(c * k^{CP})(x, x') =
(c * k_c)(x, x') + (c * k_{c'})(x, x')
$$

로 쓸 수 있고, 최종적으로는 두 개의 2D convolution 형태로 바뀐다.

$$
(c * k^{CP})(x, x') =
\sum_{p' \in \mathcal{P}(x')}
c(x,p')\, k_c^{2D}(p'-x')
+
\sum_{p \in \mathcal{P}(x)}
c(p,x')\, k_{c'}^{2D}(p-x)
$$

즉, 전체 4D neighborhood를 모두 훑지 않고 중심을 pivot으로 하는 중요한 단면들만 사용하여 두 개의 2D convolution으로 분해하는 것이다. 논문은 이를 통해 quadratic complexity 문제를 완화하고, 더 적은 parameter로 더 빠른 추론과 더 나은 혹은 비슷한 정확도를 얻었다고 주장한다.

$K$-shot 확장은 비교적 단순하다. support pair가 $K$개일 때 각 support에 대해 독립적으로 forward pass를 수행해 $\{\bar{M}_k^q\}_{k=1}^K$를 만든다. 이후 pixel 단위 voting을 하고, 최대 voting score로 normalize한 뒤 threshold $\tau=0.5$를 넘는 pixel을 foreground로 분류한다. 즉, 복잡한 joint fusion 대신 prediction-level aggregation을 사용한다.

## 4. 실험 및 결과

실험은 PASCAL-5$^i$, COCO-20$^i$, FSS-1000에서 수행되었다. PASCAL-5$^i$는 PASCAL VOC 2012 기반 20개 클래스를 4개 fold로 나눈 benchmark이고, COCO-20$^i$는 COCO 80개 클래스를 4개 fold로 나눈 benchmark다. 두 데이터셋 모두 각 fold마다 나머지 fold로 학습하고 해당 fold에서 평가하는 cross-validation 프로토콜을 따른다. FSS-1000은 1000개 클래스의 few-shot segmentation 데이터셋이며 train/val/test split이 따로 있다.

입력 이미지는 support와 query 모두 $400 \times 400$으로 맞춘다. ResNet backbone 기준 spatial resolution은 세 단계로 줄어들어 $(50,50)$, $(25,25)$, $(13,13)$이 된다. 최적화는 Adam, learning rate는 $10^{-3}$이다. backbone은 ImageNet pretrained 상태로 freeze한다.

평가 지표는 mIoU와 FB-IoU다. mIoU는 fold 내 클래스별 IoU 평균이며, 저자들은 이것이 일반화 능력과 segmentation quality를 더 잘 반영한다고 보고 주요 지표로 사용한다. FB-IoU는 class identity를 무시하고 foreground와 background의 IoU 평균을 계산한다.

PASCAL-5$^i$에서 HSNet은 세 backbone 모두에서 매우 강한 결과를 보인다. ResNet101 기준으로 1-shot mIoU는 66.2, 5-shot mIoU는 70.4이다. 논문은 이를 PFENet이나 RePRI 같은 당시 강한 baseline보다 높은 수치로 제시한다. 특히 ResNet101 backbone의 HSNet은 1-shot에서 PFENet 대비 6.1%p, 5-shot에서 RePRI 대비 4.8%p 향상을 보고한다. 흥미로운 점은 learnable parameter 수가 2.6M으로 매우 작다는 것이다. backbone 자체는 크지만 frozen이며, 실제 학습되는 부분만 보면 다른 방법보다 훨씬 가볍다.

COCO-20$^i$에서도 성능 향상이 뚜렷하다. ResNet101 기준 1-shot mIoU는 41.2, 5-shot은 49.5이다. 논문은 PFENet 대비 각각 2.7%p와 6.8%p 개선이라고 정리한다. COCO는 클래스와 장면 다양성이 더 크기 때문에 보통 더 어렵게 여겨지는데, HSNet이 여기서도 robust하게 작동한다는 점이 강조된다.

FSS-1000에서는 ResNet101 기반 HSNet이 1-shot 86.5, 5-shot 88.5 mIoU를 기록한다. 이는 비교된 기존 방법들보다 가장 높은 수치다. 따라서 단일 benchmark에 특화된 개선이 아니라 서로 다른 분포의 세 데이터셋에서 일관되게 강함을 보인다고 해석할 수 있다.

도메인 시프트 실험도 포함되어 있다. COCO에서 학습한 모델을 PASCAL-5$^i$에 적용하는 setting에서, data augmentation 없이도 ResNet101 기반 HSNet은 1-shot 64.1, 5-shot 70.3을 기록했다. 논문은 RePRI 대비 5-shot에서 1.0%p 더 높고, trainable parameter 수는 18배 더 적다고 강조한다. 이는 관계 학습 기반 접근이 dataset bias를 비교적 덜 타는 가능성을 보여주는 결과로 읽을 수 있다.

ablation study도 매우 충실하다. 먼저 hypercorrelation 자체의 효과를 보기 위해 multi-layer correlation 대신 단일 intermediate layer만 사용한 경우를 비교했다. ResNet101, PASCAL-5$^i$ 기준 full hypercorrelation은 1-shot 66.2, 5-shot 70.4인데, deep single-layer correlation은 61.7과 67.5, shallow single-layer correlation은 59.1과 65.7에 그쳤다. 즉, 깊은 layer 하나만 쓰는 것보다도 multi-level correlation을 함께 쓰는 것이 훨씬 낫다.

pyramid layer의 중요성도 분석했다. 가장 깊은 semantic layer만 사용하는 $ \mathcal{C}^{(3)} $는 1-shot 55.5, 5-shot 61.6으로 크게 떨어졌고, 중간과 깊은 layer만 사용하는 $ \mathcal{C}^{(2:3)} $도 full pyramid보다 낮았다. 이는 semantic 정보만으로는 거친 localization은 가능해도 정교한 경계 복원이 어렵고, 낮은 level geometric cue가 실제 segmentation 품질에 중요하다는 해석을 뒷받침한다.

4D kernel 비교는 이 논문의 또 다른 핵심 실험이다. Original 4D kernel, separable 4D kernel, center-pivot 4D kernel을 비교했을 때, center-pivot은 ResNet101, PASCAL-5$^i$ 기준 1-shot 66.2, 5-shot 70.4를 기록했다. 정확도는 가장 높거나 거의 동등한 수준이면서, per-episode inference time은 25.51ms로 가장 빠르고, memory footprint는 1.39GB, FLOPs는 20.56G로 가장 작다. original 4D kernel이 512.17ms, 4.12GB, 702.35G FLOPs였다는 점을 감안하면 계산 효율 개선 폭이 매우 크다.

building block 깊이도 시험했다. 4D conv layer 수를 늘릴수록 성능은 3개 층까지 의미 있게 증가했지만 이후 saturation되는 경향을 보여, 최종 모델은 각 block에 3개의 4D conv를 사용했다.

backbone finetuning 실험도 인상적이다. backbone을 함께 학습하면 training 성능은 빠르게 올라가지만 validation mIoU는 오히려 나빠지며 overfitting이 심해진다. 반면 backbone을 freeze한 HSNet은 training과 validation 사이 차이는 있지만 novel class generalization이 더 낫다. 저자들은 이를 few-shot regime에서는 새로운 feature representation 자체를 다시 배우기보다, 이미 큰 데이터에서 배운 representation 사이의 relation을 학습하는 것이 더 중요하다는 증거로 해석한다.

부록에는 추가 결과도 있다. 10-shot 결과에서도 HSNet은 PASCAL-5$^i$에서 70.6, COCO-20$^i$에서 48.7 mIoU를 기록해 기존 방법보다 높다. 또한 support feature masking 없이도 어느 정도 강한 결과를 보였는데, 이는 모델이 “support와 query에서 공통으로 나타나는 대상”을 찾는 co-segmentation 유사 문제에도 일정 부분 적용 가능함을 시사한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 few-shot segmentation에서 correspondence learning을 본질적 문제로 다시 정식화하고, 이를 실제로 효과적인 architecture로 연결했다는 점이다. 단순히 support 정보를 prototype으로 압축하는 대신, multi-level dense relation을 4D tensor로 유지한 채 분석하는 설계는 문제 구조와 잘 맞는다. 또한 semantic cue와 geometric cue를 coarse-to-fine으로 결합하는 pyramid encoder는 segmentation task의 요구와 자연스럽게 맞물린다.

두 번째 강점은 center-pivot 4D convolution의 설계다. 이것은 단순한 경량화 트릭이 아니라, 어떤 4D 이웃이 실제로 중요한가에 대한 구조적 가정을 반영한다. 실험 결과도 정확도, 속도, 메모리 면에서 설득력이 있다. 특히 2.6M learnable parameter로 여러 benchmark에서 strong baseline을 넘은 점은 매우 인상적이다.

세 번째 강점은 실험의 폭과 ablation의 충실성이다. 저자들은 성능 비교뿐 아니라 hypercorrelation, pyramid layer, kernel type, block depth, backbone freezing 등 핵심 설계 선택을 각각 분리해서 분석했다. 따라서 성능 향상의 원인을 비교적 명확하게 추적할 수 있다.

반면 한계도 있다. 첫째, 방법의 계산 효율이 많이 개선되었다고는 하지만, 여전히 4D correlation tensor를 다루는 구조 자체는 메모리와 연산 면에서 가볍지 않다. 논문은 center-pivot으로 크게 줄였지만, 매우 고해상도 환경이나 더 큰 backbone, 더 많은 support set에 대해 얼마나 잘 확장되는지는 본문에서 충분히 다루지 않는다.

둘째, $K$-shot 확장이 단순 voting 기반이라는 점은 다소 보수적이다. 여러 support example 사이의 상호보완 정보를 feature level에서 jointly reason하는 구조는 아니다. 이 방식이 충분히 잘 작동했다는 결과는 있지만, 왜 그것이 최적인지 혹은 더 정교한 fusion보다 나은지는 논문에서 깊게 논의하지 않는다.

셋째, failure case도 분명하다. 부록 설명에 따르면 severe occlusion, large intra-class variation, 매우 작은 object가 있을 때 성능이 떨어진다. 이는 dense correlation 기반 방법이 support-query 사이의 대응 관계가 지나치게 약하거나 불안정할 때 취약할 수 있음을 보여준다.

넷째, center-pivot kernel의 이론적 최적성은 논문이 주장하지 않는다. 즉, 왜 중심을 지나는 이웃만 남기는 것이 가장 적절한 sparsification인지에 대한 강한 이론 증명은 없다. 논문은 실험적으로 그 유효성을 보여주지만, 이것이 다른 구조적 sparsification보다 본질적으로 우월하다고 일반화하기는 어렵다.

비판적으로 보면, 이 논문은 feature relation 분석의 중요성을 강하게 설득하는 데 성공했지만, support-query 관계를 전적으로 cosine correlation과 local 4D pattern analysis에 의존한다는 점에서 장거리 구조 제약이나 object-level reasoning은 상대적으로 약할 수 있다. 다만 이는 논문이 직접 주장하지 않은 확장 가능성의 영역이며, 본문만으로는 더 강한 결론을 내리기 어렵다.

## 6. 결론

이 논문은 few-shot semantic segmentation을 위해 HSNet이라는 새로운 구조를 제안했다. 핵심은 multi-level intermediate feature들로부터 4D hypercorrelation을 만들고, 이를 4D-convolutional pyramid encoder로 coarse-to-fine하게 압축하여 query segmentation mask를 예측하는 것이다. 여기에 center-pivot 4D convolution을 도입해 고차원 correlation processing의 계산 비용을 크게 줄였다.

논문의 주요 기여는 세 가지로 요약된다. 첫째, few-shot segmentation에서 multi-level dense correspondence를 직접 학습적으로 분석하는 framework를 제시했다. 둘째, center-pivot 4D convolution으로 정확도와 효율을 동시에 확보했다. 셋째, PASCAL-5$^i$, COCO-20$^i$, FSS-1000에서 강한 실험적 우위를 보였다.

실제 적용 측면에서 이 연구는 적은 annotation으로 새로운 객체를 빠르게 분할해야 하는 의료영상, 로보틱스, 산업 비전 같은 분야에 잠재적 가치가 있다. 향후 연구로는 support set을 더 정교하게 통합하는 방식, transformer류의 relation modeling과의 결합, 더 큰 해상도나 더 복잡한 장면에 대한 확장 등이 자연스러운 방향으로 보인다. 전체적으로 이 논문은 few-shot segmentation에서 “무엇을 보느냐”보다 “support와 query의 관계를 어떻게 해석하느냐”가 중요하다는 점을 강하게 보여준 작업이다.
