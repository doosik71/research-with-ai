# Fully Convolutional Adaptation Networks for Semantic Segmentation

- **저자**: Yiheng Zhang, Zhaofan Qiu, Ting Yao, Dong Liu, Tao Mei
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1804.08286

## 1. 논문 개요

이 논문은 synthetic dataset에서 학습한 semantic segmentation 모델을 real-world image에 적용할 때 발생하는 domain shift 문제를 해결하는 방법을 다룬다. 구체적으로는, 라벨이 있는 source domain 데이터와 라벨이 없는 target domain 데이터를 이용하는 unsupervised domain adaptation 설정에서 semantic segmentation 성능을 높이는 것이 목표다.

문제의 핵심은 pixel-level annotation의 비용이 매우 크다는 점이다. semantic segmentation은 이미지의 모든 픽셀에 대해 클래스 라벨이 필요하므로, 실제 도시 주행 장면 같은 데이터셋을 구축할 때 인적 비용이 매우 높다. 반면 GTA5 같은 video game 기반 synthetic data는 자동으로 정답 라벨을 얻을 수 있다. 그러나 synthetic image에서 학습한 모델을 실제 도시 장면에 바로 적용하면, 조명, 질감, 채도, 경계 표현 등의 차이 때문에 일반화 성능이 크게 떨어진다.

이 논문은 이 문제를 두 수준에서 다룬다. 첫째는 image appearance 자체를 target domain처럼 보이게 바꾸는 appearance-level adaptation이고, 둘째는 feature representation이 domain을 구분하기 어렵게 만드는 representation-level adaptation이다. 저자들은 이 두 가지를 결합한 **FCAN (Fully Convolutional Adaptation Networks)** 을 제안하며, semantic segmentation에서 기존 비지도 domain adaptation 방법보다 더 나은 성능을 보고한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 domain adaptation을 단순히 feature space 정렬만으로 보지 않고, **“보이는 방식(appearance)”과 “표현되는 방식(representation)”을 함께 맞춘다**는 데 있다.

기존 adversarial domain adaptation 계열 방법은 주로 feature representation이 source와 target 사이에서 구분되지 않도록 만드는 데 초점을 둔다. 반면 이 논문은 그것만으로는 부족하다고 보고, 먼저 source image의 semantic content는 유지하면서 target domain의 style을 입힌 adaptive image를 만들고, 그 다음 feature level에서도 domain discriminator를 속이도록 학습한다. 즉, 입력 단계와 표현 단계 모두에서 domain gap을 줄이는 구조다.

이 차별점은 semantic segmentation이라는 dense prediction task에서 특히 중요하다. image-level classification과 달리 segmentation은 local texture, 경계, region-level context에 매우 민감하기 때문이다. 그래서 저자들은 discriminator도 image-level이 아니라 spatial unit 단위, 즉 region-level로 작동하도록 설계했고, 여기에 ASPP를 넣어 서로 다른 receptive field를 활용하도록 했다.

## 3. 상세 방법 설명

FCAN은 크게 **AAN (Appearance Adaptation Networks)** 과 **RAN (Representation Adaptation Networks)** 으로 구성된다.

### AAN: source content와 target style의 결합

AAN의 목적은 source image를 target domain에서 온 것처럼 보이게 바꾸는 것이다. 다만 semantic segmentation을 해야 하므로, source image 안의 객체 배치와 의미 정보는 유지되어야 한다. 저자들은 이 문제를 neural style transfer와 유사한 방식으로 푼다.

입력은 source image $x_s$ 하나와 target domain 이미지 집합 $X_t = \{x_t^i\}$ 이다. 출력 image $x_o$는 white noise에서 시작해서 반복적으로 최적화된다. 여기서 pretrained CNN이 각 이미지의 feature map을 추출한다.

저자들은 source image의 **semantic content preservation**을 위해, source image와 output image의 feature map이 비슷해지도록 만든다. 이를 다음과 같이 쓴다.

$$
\min_{x_o} \sum_{l \in L} w_s^l \, Dist(M_o^l, M_s^l)
$$

여기서 $M_o^l$와 $M_s^l$는 각각 layer $l$에서의 output image와 source image의 feature map이고, $w_s^l$는 해당 layer의 가중치다. 깊은 layer일수록 더 높은 수준의 semantic information을 담고 있으므로, 여러 layer를 함께 사용해 source의 의미 구조를 유지한다.

반대로 target domain의 **style preservation**은 feature correlation, 즉 Gram matrix를 이용한다. 각 layer에서 style은 다음과 같이 정의된다.

$$
G^{l,ij} = M^{l,i} M^{l,j}
$$

즉, 같은 layer 안에서 서로 다른 채널 간의 상관관계를 본다. 이는 spatial position보다는 texture, color, shading 같은 통계적 pattern을 반영한다. target domain 전체의 style은 각 target image의 Gram matrix를 평균한 $\bar{G}_t^l$로 정의한다. output image가 이 style을 따르도록 하는 목적함수는 다음과 같다.

$$
\min_{x_o} \sum_{l \in L} w_t^l \, Dist(G_o^l, \bar{G}_t^l)
$$

최종적으로 AAN의 손실은 content 보존과 style 적응을 합친 형태다.

$$
L_{AAN}(x_o) =
\sum_{l \in L} w_s^l Dist(M_o^l, M_s^l)
+
\alpha \sum_{l \in L} w_t^l Dist(G_o^l, \bar{G}_t^l)
$$

여기서 $\alpha$는 content와 style의 균형을 조절한다. 논문은 semantic content를 정확히 보존하는 것이 더 중요하다고 보고, $\alpha = 10^{-14}$라는 매우 작은 값을 사용했다. 즉 style은 강하게 덮어씌우는 것이 아니라, appearance를 살짝 조정하는 역할로 제한했다.

구현에서는 ResNet-50을 pretrained CNN으로 사용하고, $L = \{\text{conv1}, \text{res2c}, \text{res3d}, \text{res4f}, \text{res5c}\}$ 층을 사용한다. output image는 최대 1000 iteration 동안 gradient descent로 갱신된다.

### RAN: domain-invariant representation 학습

AAN으로 appearance gap을 줄인 뒤, RAN은 feature representation 수준에서 source와 target을 더 구분하기 어렵게 만든다. 기본 backbone은 ResNet-101 기반 dilated FCN이다.

RAN에는 두 개의 목표가 동시에 있다.

첫째, source domain에서는 semantic segmentation을 잘해야 한다. 즉 source image와 라벨을 이용해 일반적인 supervised segmentation loss $L_{seg}$를 최적화한다.

둘째, source와 target의 representation을 보고 domain discriminator $D$가 어느 domain에서 왔는지 맞히지 못하게 해야 한다. 이를 위해 adversarial loss $L_{adv}$를 사용한다. 논문은 spatial unit별로 discriminator가 domain을 예측하도록 설계했으며, 손실은 다음과 같다.

$$
L_{adv}(X_s, X_t)
=
- E_{x_t \sim X_t}
\left[
\frac{1}{Z}\sum_{i=1}^{Z}\log D_i(F(x_t))
\right]
-
E_{x_s \sim X_s}
\left[
\frac{1}{Z}\sum_{i=1}^{Z}\log (1 - D_i(F(x_s)))
\right]
$$

여기서 $F$는 shared FCN feature extractor이고, $D_i(F(x))$는 $i$번째 spatial unit이 target domain에 속할 확률이다. $Z$는 discriminator 출력의 spatial unit 수다.

논문은 adversarial training을 다음 minimax 문제로 둔다.

$$
\max_F \min_D L_{adv}(X_s, X_t)
$$

의미는 명확하다. discriminator $D$는 source와 target을 잘 구분하려 하고, feature extractor $F$는 그 구분을 어렵게 만들려 한다. 이렇게 하면 domain-invariant representation이 학습된다.

최종적으로 segmentation loss까지 포함한 RAN의 전체 목적함수는 다음과 같다.

$$
\max_F \min_D \{ L_{adv}(X_s, X_t) - \lambda L_{seg}(X_s) \}
$$

즉 $F$는 segmentation도 잘하고 domain confusion도 유도해야 한다. 논문에서는 $\lambda = 5$를 사용했다.

### ASPP를 이용한 discriminator 강화

저자들은 단일 scale representation만으로는 다양한 크기의 객체를 잘 다루기 어렵다고 보고, discriminator 쪽에 **ASPP (Atrous Spatial Pyramid Pooling)** 를 넣는다. 구체적으로, FCN의 마지막 feature map 위에 sampling rate가 서로 다른 $k=4$개의 dilated convolution을 병렬로 적용한다. 각 branch는 $c=128$ 채널을 출력하고, dilation rate는 1, 2, 3, 4다. 그런 다음 이들을 channel 방향으로 쌓고, $1 \times 1$ convolution과 sigmoid를 거쳐 최종 domain score map을 만든다.

이 구조는 서로 다른 receptive field를 활용해 domain 차이를 더 잘 감지하게 만들고, 결과적으로 feature extractor가 더 강한 adversarial pressure를 받게 한다.

### 학습 절차

학습은 두 단계다.

먼저 source domain만으로 segmentation loss를 사용해 RAN을 pre-train한다. learning rate는 0.0025, batch size는 6, 최대 iteration은 30k다. 이후 segmentation loss와 adversarial loss를 함께 사용해 fine-tuning한다. 이때 learning rate는 0.0001, batch size는 8, 최대 iteration은 10k다. optimizer는 SGD이며 momentum은 0.9, weight decay는 0.0005다. learning rate schedule은 “poly” policy를 사용한다.

논문은 Caffe 프레임워크를 사용했다고 명시한다.

## 4. 실험 및 결과

### 데이터셋과 평가 설정

주요 실험은 **GTA5 $\rightarrow$ Cityscapes** domain adaptation이다.

GTA5는 24,966장의 synthetic driving scene image를 포함하며, 해상도는 $1914 \times 1052$이다. Cityscapes는 실제 도시 거리 장면 5,000장으로 이루어져 있고, 해상도는 $2048 \times 1024$다. 두 데이터셋 모두 평가에는 19개 semantic class를 사용한다. 논문은 기존 설정을 따라 Cityscapes validation set 500장을 비지도 adaptation 평가에 사용했다.

추가로 **BDDS**도 target domain으로 사용한다. BDDS는 dashcam video frame 기반 데이터셋이며, 해상도는 $1280 \times 720$이다. 평가에는 1,500 frame을 사용했다.

평가 지표는 각 클래스별 IoU와 전체 평균인 mIoU다.

### AAN 자체 평가

AAN의 효과를 보기 위해, source만 adaptation하거나, target만 adaptation하거나, 둘 다 adaptation하는 여러 설정을 비교했다. 결과적으로 source image를 target style로 바꾸는 방식이 가장 유리했고, 여기에 RAN을 함께 쓸 때 성능이 더 올라갔다.

논문에서 가장 중요한 수치는 다음과 같다.

- adaptation 없이 FCN만 사용하면 Cityscapes validation에서 mIoU가 29.15%
- source adaptation과 RAN을 함께 사용하면 46.21%
- 여러 AAN 설정의 late fusion까지 하면 46.60%

즉 appearance-level adaptation만으로도 baseline보다 분명한 향상이 있고, representation-level adaptation과 결합했을 때 가장 좋다.

논문은 target image를 adaptation하는 경우도 실험했지만, 이 경우 object boundary 부근에 noise가 생겨 segmentation 안정성이 떨어질 수 있다고 해석한다. 이 부분은 저자들의 해석이며, 정량적으로 boundary artifact를 별도 측정한 결과가 제시되지는 않았다.

### Ablation study

논문은 FCAN의 각 설계 요소를 순차적으로 추가한 ablation study를 제시한다.

- FCN baseline: 29.15
- `+ABN`: 35.51
- `+ADA`: 41.29
- `+Conv`: 43.17
- `+ASPP`: 44.81
- `+AAN` 최종 FCAN: 46.60

여기서 `ABN`은 Adaptive Batch Normalization으로, source에서 학습한 BN의 mean/variance를 target 데이터 통계로 바꾸는 단순한 기법이다. 이것만으로도 6%p 이상 상승했다. 그 위에 image-level adversarial adaptation인 `ADA`, region-level adversarial classification인 `Conv`, multi-scale receptive field를 주는 `ASPP`, 그리고 appearance transfer인 `AAN`이 차례로 성능을 올린다.

이 결과는 논문의 주장을 잘 뒷받침한다. 즉 domain adaptation에서 단일 기법보다는, 통계 정렬, adversarial representation alignment, multi-scale region discrimination, style-level appearance transfer가 서로 보완적으로 작동한다는 것이다.

### 기존 방법과 비교

Cityscapes에서 기존 방법과 비교한 결과는 다음과 같다.

- DC: 37.64
- ADDA: 38.30
- FCNWild: 42.04
- FCAN: 46.60
- FCAN(MS): 47.75

여기서 FCAN은 FCNWild보다 4.56%p 높다. FCNWild도 region-level adversarial adaptation을 사용하지만, FCAN은 여기에 ASPP 기반 discriminator 강화와 AAN 기반 appearance adaptation을 추가했다. 따라서 improvement의 원인이 구조적으로 비교적 명확하다.

논문은 클래스별 IoU도 제시하며, FCAN이 19개 클래스 중 17개에서 가장 좋다고 보고한다. 특히 road, sky 같은 major class뿐 아니라 bicycle, truck 같은 상대적으로 어려운 클래스도 개선된다고 설명한다. 다만 각 클래스의 정확한 수치를 표 형식으로 모두 제공하지는 않고 figure로 제시한다.

### Domain discriminator 해석

논문은 domain discriminator의 prediction map도 시각화한다. 밝을수록 target domain일 확률이 높은 영역이다. 저자들은 어떤 region에서 discriminator가 target image임에도 target으로 잘 예측하지 못하면, 그 region의 representation이 더 domain-invariant하다고 해석한다. 예를 들어 sky 영역에서 discriminator가 혼동하는 경우 segmentation이 잘 되는 사례를 제시한다. 반대로 bicycle처럼 discriminator가 여전히 domain-dependent하게 보는 영역은 segmentation도 어렵다고 설명한다.

이 분석은 직관적으로 흥미롭지만, 어디까지나 qualitative analysis에 가깝다. representation indistinguishability와 segmentation 정확도 사이의 인과 관계를 엄밀히 검증한 것은 아니다.

### Semi-supervised adaptation

논문은 소량의 target 라벨이 있을 때의 semi-supervised 확장도 제시한다. 이 경우 target labeled set $X_t^l$에 대한 segmentation loss를 추가해 목적함수를 다음처럼 바꾼다.

$$
\max_F \min_D
\{
L_{adv}(X_s, X_t)
-
\lambda_s L_{seg}(X_s)
-
\lambda_t L_{seg}(X_t^l)
\}
$$

Cityscapes에서 target labeled image 수를 늘려가며 실험한 결과, 소량의 라벨만 있어도 FCAN의 이점이 크다. 예를 들어:

- 50장 라벨: FCN 47.57, FCAN 56.50
- 100장 라벨: FCN 54.41, FCAN 59.95
- 200장 라벨: FCN 59.53, FCAN 63.82
- 1000장 라벨: FCN 68.05, FCAN 69.17

즉 target 라벨이 매우 적을수록 domain adaptation의 가치가 더 크고, 라벨 수가 많아질수록 격차는 줄어든다. 이 경향은 직관과도 맞다.

### BDDS 결과

BDDS에 대한 결과는 다음과 같다.

- FCNWild: 39.37
- FCAN: 43.35
- FCAN(MS): 45.47
- FCAN(MS+EN): 47.53

논문은 이를 새로운 state of the art라고 주장한다. 제공된 본문 기준으로는 그 주장을 반박할 근거는 없지만, 이 평가는 2018년 당시 기준이며 이후 연구까지 포함한 최신 기준은 아니다. 본문만으로는 당시 모든 경쟁 방법을 완전하게 망라했는지는 판단할 수 없다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 domain adaptation을 단일 feature alignment 문제로 축소하지 않고, appearance-level과 representation-level을 분리해 설계했다는 점이다. AAN은 입력 image 분포를, RAN은 feature 분포를 다룬다. 이 이중 구조는 논리적으로 설득력이 있고, ablation 결과도 각 구성요소의 기여를 비교적 명확하게 보여 준다.

또 다른 강점은 semantic segmentation의 특성을 잘 반영했다는 점이다. discriminator를 image-level이 아니라 region-level로 설계했고, ASPP로 receptive field를 넓혀 multi-scale context를 활용했다. 이는 dense prediction task에 맞는 adaptation 설계라고 볼 수 있다. 실제로 image-level discriminator 기반 DC, ADDA보다 region-level 방식인 FCNWild와 FCAN이 더 좋은 결과를 낸다.

실험도 비교적 충실하다. GTA5 $\rightarrow$ Cityscapes뿐 아니라 BDDS까지 검증했고, unsupervised와 semi-supervised 설정을 모두 다뤘으며, AAN 사용 방식과 각 모듈의 기여도도 분석했다.

반면 한계도 분명하다. 첫째, AAN은 각 source image마다 white noise에서 시작해 iterative optimization으로 adaptive image를 만드는 방식이라 계산 비용이 크고 실용성이 떨어질 가능성이 있다. 논문은 최대 1k iteration을 사용한다고 적지만, 전체 파이프라인의 처리 시간이나 실제 학습 효율은 자세히 보고하지 않는다.

둘째, AAN의 style transfer 품질이 segmentation 성능에 어떤 방식으로 연결되는지 정량적으로 깊게 분석하지는 않는다. 저자들은 target adaptation 시 boundary noise 문제가 있을 수 있다고 해석하지만, 이 역시 별도의 정량 지표나 controlled study는 없다.

셋째, adversarial learning은 학습 안정성이 중요한데, discriminator와 feature extractor의 최적화 균형 문제, 수렴 특성, seed 민감도 등에 대한 논의는 없다. 이는 당시 많은 adversarial adaptation 논문에서 공통적으로 보이는 한계다.

넷째, 이 논문은 source와 target 사이의 label space가 동일하고, 클래스 정의가 호환된다는 가정을 둔다. GTA5, Cityscapes, BDDS는 모두 19 class 프로토콜을 공유하므로 설정상 타당하지만, 클래스 불일치나 open-set adaptation 같은 더 어려운 설정은 다루지 않는다.

다섯째, qualitative figure는 설득력 있지만, failure case를 체계적으로 분석하지는 않는다. 어떤 클래스가 끝까지 어려운지, 왜 2개 클래스에서는 최고 성능을 내지 못했는지, 또는 작은 객체와 경계에서 어떤 유형의 오류가 남는지에 대한 세부 분석은 제한적이다.

## 6. 결론

이 논문은 semantic segmentation을 위한 unsupervised domain adaptation에서 **appearance-level adaptation과 representation-level adaptation을 동시에 결합한 FCAN**을 제안했다. AAN은 source image의 semantic content를 유지하면서 target style을 입히고, RAN은 adversarial learning을 통해 domain-invariant representation을 학습한다. 또한 ASPP 기반 region-level discriminator를 통해 segmentation task에 맞는 multi-scale adaptation을 구현했다.

실험적으로 GTA5에서 Cityscapes, 그리고 BDDS로의 전이에서 기존 방법보다 높은 mIoU를 달성했고, ablation study를 통해 각 구성요소의 효과도 보여 주었다. 특히 단순 BN 통계 교체, adversarial adaptation, region-level discrimination, multi-scale context, appearance transfer가 모두 누적적으로 기여한다는 점이 잘 드러난다.

실제 적용 측면에서는, synthetic data를 충분히 만들 수 있지만 real annotation이 비싼 상황에서 이 연구의 의미가 크다. 자율주행, 로보틱스, 시뮬레이터 기반 학습 같은 분야에서 유용한 방향성을 제시한다. 향후 연구로는 더 효율적인 image translation 방식, 더 안정적인 adversarial optimization, class-level 또는 structure-level alignment, 그리고 indoor scenes나 portrait segmentation 같은 다른 segmentation 문제로의 확장이 자연스럽게 이어질 수 있다.
