# A Comprehensive Analysis of Weakly-Supervised Semantic Segmentation in Different Image Domains

- **저자**: Lyndon Chan, Mahdi S. Hosseini, Konstantinos N. Plataniotis
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/1912.11186

## 1. 논문 개요

이 논문은 image-level label만으로 semantic segmentation을 학습하는 weakly-supervised semantic segmentation(WSSS) 방법들이 서로 다른 이미지 도메인에서도 잘 작동하는지를 체계적으로 분석한다. 기존 WSSS 연구는 거의 대부분 PASCAL VOC2012 같은 natural scene image를 중심으로 발전해 왔는데, 저자들은 이러한 방법을 histopathology와 satellite image에 그대로 적용해도 되는지에 의문을 제기한다.

연구 문제는 명확하다. 자연 영상용으로 설계된 WSSS 방법들이 다른 도메인에서도 유효한가, 그리고 어떤 데이터셋에서는 어떤 방식이 더 적합한가를 밝히는 것이다. 이는 실용적으로 매우 중요하다. pixel-level annotation은 비용과 시간이 매우 많이 들지만, image-level annotation은 훨씬 저렴하다. 따라서 WSSS가 다양한 도메인에서 안정적으로 작동한다면 실제 적용 가치가 매우 크다. 반대로 도메인에 따라 성능이 크게 달라진다면, 현재의 WSSS 방법론은 일반화되지 않는다는 뜻이 된다.

저자들은 세 가지 대표 데이터셋을 선택했다. ADP는 histopathology, PASCAL VOC2012는 natural scene, DeepGlobe는 satellite domain을 대표한다. 그리고 자연 영상용 방법인 SEC, DSRG, IRNet, 그리고 병리 영상용 방법인 HistoSegNet을 같은 틀 안에서 비교해, 어떤 방법이 어떤 조건에서 유리한지 분석했다.

## 2. 핵심 아이디어

논문의 핵심 아이디어는 “WSSS는 하나의 보편적인 문제처럼 보이지만, 실제로는 이미지 도메인에 따라 어려움의 성격이 다르다”는 점을 실험적으로 보여주는 데 있다. natural scene image에서는 배경과 foreground를 구분하는 문제, 그리고 object의 일부가 아니라 전체를 분할해야 하는 문제가 크다. 반면 histopathology와 satellite image에서는 경계가 모호하거나 class co-occurrence가 심한 문제가 더 중요하게 나타난다.

이 논문의 중요한 차별점은 새로운 WSSS 알고리즘 하나를 제안하는 것이 아니라, 서로 다른 도메인에서 기존 state-of-the-art 방법들을 공정하게 비교하고, 어떤 방법이 왜 특정 도메인에서 잘 되거나 실패하는지를 분석한다는 점이다. 특히 저자들은 세 가지 일반 원칙을 끌어낸다. 첫째, classification network가 생성하는 activation cue의 sparsity가 중요하다. 둘째, self-supervised learning이 항상 도움이 되는 것은 아니다. 셋째, training set에서 class co-occurrence가 높으면 image-level supervision만으로는 class를 공간적으로 분리하기 매우 어렵다.

즉, 이 논문은 “현재 잘 알려진 WSSS 파이프라인이 자연 영상에서는 강하지만, 다른 도메인에서는 근본적으로 맞지 않을 수 있다”는 점을 데이터와 실험을 통해 설득력 있게 보여준다.

## 3. 상세 방법 설명

논문은 네 가지 방법을 비교한다. SEC, DSRG, IRNet은 자연 영상용 self-supervised WSSS 방법이고, HistoSegNet은 histopathology용 방법이다. 네 방법 모두 먼저 image-level label로 classification network를 학습하고, 그 네트워크에서 coarse localization map을 만든 뒤, 이를 pixel-level segmentation으로 바꾼다. 하지만 그 후처리 방식이 다르다.

SEC는 네 단계로 이루어진다. 먼저 classification CNN 두 개를 학습한다. 하나는 foreground network이고, 다른 하나는 background network다. 그 다음 CAM(Class Activation Map)을 생성한다. 이후 foreground CAM은 최대 activation의 20% 이상을 weak cue로 사용하고, background cue는 background network의 CAM들을 합친 뒤 median filtering을 적용하고 가장 낮은 10% activation 영역을 background로 본다. 겹치는 경우에는 더 작은 cue가 우선한다. 마지막으로 이 cue들을 pseudo ground truth로 사용해 FCN을 학습한다. 이때 손실 함수는 세 부분으로 구성된다. 하나는 seed와 맞추는 seeding loss, 하나는 image-level annotation과 일치하도록 하는 expansion loss, 마지막은 dense CRF를 적용한 결과와 일관되게 만드는 constrain loss다.

DSRG는 SEC와 비슷하지만 몇 가지 중요한 차이가 있다. background network를 따로 두지 않고 DRFI라는 saliency 방식으로 background activation을 만든다. foreground CAM은 최대값의 20% 이상만 seed로 삼은 뒤, convolutional feature를 이용한 region growing으로 cue를 확장한다. 이후 DeepLabv2를 학습하는데, 손실은 seed를 따르는 seeding loss와 dense CRF 결과와 맞추는 boundary loss 두 가지를 사용한다. 핵심은 coarse CAM seed를 주변의 비슷한 영역으로 퍼뜨려 pseudo label을 더 풍부하게 만드는 것이다.

IRNet은 직접 pixel class를 예측하기보다는, 보조적인 구조 정보를 먼저 학습하는 방식이다. 먼저 ResNet-50 기반 classification CNN을 학습하고 CAM을 만든다. 각 class의 CAM이 0.3 이상인 영역은 foreground seed, 0.05 미만이고 foreground로 할당되지 않은 영역은 background seed로 본다. 그 다음 backbone에서 두 개의 branch를 학습한다. 하나는 displacement field(DF)로, 각 pixel이 seed instance의 centroid로부터 얼마나 떨어져 있는지를 예측한다. 다른 하나는 class boundary map(CBM)으로, 이웃 pixel 사이에 class boundary가 있을 가능성을 예측한다. 마지막에는 inverse boundary map을 transition probability로 사용하는 random walk를 CAM에 적용해, confident region이 같은 object 내부의 덜 confident한 영역까지 확장되도록 한다.

HistoSegNet은 위 세 방법과 달리 downstream segmentation network를 self-supervised하게 학습하지 않는다. 먼저 ADP의 patch-level histological tissue type label로 classification CNN을 학습한다. 이 CNN은 VGG16 변형인데, softmax 대신 sigmoid를 쓰고, 각 convolution 뒤에 batch normalization을 넣고, flatten 대신 global max pooling을 사용한다. 그 다음 Grad-CAM으로 coarse class activation map을 만든다. 이후 histopathology 도메인에 맞춘 수작업 조정을 한다. 예를 들어 ADP에는 non-tissue label이 없으므로 `background` activation을 별도로 만들어야 하고, functional mode에서는 `other` activation도 필요하다. `background`는 평균 RGB가 밝은 영역을 기반으로 sigmoid 변환과 class subtraction, Gaussian blur를 통해 만든다. `other`는 다른 functional class들과 adipose, background activation을 조합한 뒤 보수적으로 생성한다. 또한 여러 class activation이 겹치는 경우에는 각 map에서 다른 map들의 최대값을 빼서 overlap을 줄인다. 마지막으로 dense CRF를 적용해 경계를 정교화한다.

논문은 추가로 classification backbone의 구조가 WSSS에 미치는 영향도 실험했다. VGG16 계열의 깊이, vectorization 방식(GAP, Flatten, GMP), hierarchical binary relevance(HBR) 사용 여부를 바꾼 여러 변형을 비교했다. classification 성능은 일반적으로 더 깊은 네트워크가 좋았지만, segmentation 성능은 반드시 그렇지 않았다. 특히 feature map 해상도가 더 큰 얕은 네트워크가 작은 segment가 많은 데이터셋에서는 더 유리했다. 이는 WSSS에서 좋은 classifier가 곧 좋은 segmenter는 아니라는 뜻이다.

평가 지표는 mean Intersection-over-Union(mIoU)이며, 논문은 이를 다음과 같이 사용한다.

$$
\mathrm{mIoU} = \frac{1}{C}\sum_{c=1}^{C}\frac{|P_c \cap T_c|}{|P_c \cup T_c|}
$$

여기서 $P_c$는 예측 mask, $T_c$는 ground-truth mask, $C$는 class 수다.

또한 seed 품질을 분석하기 위해 mean recall도 사용한다.

$$
\mathrm{Recall} = \frac{1}{C}\sum_{c=1}^{C}\frac{|P_c \cap T_c|}{|T_c|}
$$

이 값은 thresholded seed가 실제 ground-truth를 얼마나 덮는지를 보여준다.

## 4. 실험 및 결과

실험은 세 데이터셋에서 수행되었다. ADP는 histopathology patch 데이터셋으로, 훈련에는 14,134개의 image-labeled patch를 사용했고 validation과 evaluation에는 pixel annotation이 있는 50장씩을 사용했다. PASCAL VOC2012는 `trainaug` 12,031장을 training에, `val` 1,449장을 evaluation에 사용했다. DeepGlobe는 원래 fully-supervised segmentation dataset인데, 저자들은 803장의 annotation된 image를 603장 train, 200장 test로 나누어 weak supervision 실험에 사용했다. DeepGlobe의 `unknown` class는 제외했다.

비교된 방법은 SEC, DSRG, IRNet, HistoSegNet이며, classification backbone의 차이로 인한 편향을 줄이기 위해 각 방법을 VGG16과 M7/X1.7 계열 네트워크와 함께 시험했다. 모든 모델은 ImageNet pretrained weight로 초기화되었다.

ADP에서는 HistoSegNet이 가장 일관되게 강했다. 형태학적(morphological) 분할과 기능적(functional) 분할 모두에서, HistoSegNet만이 baseline Grad-CAM을 안정적으로 넘어섰고, self-supervised 방법인 SEC와 DSRG는 오히려 부정확한 확장을 많이 일으켰다. IRNet은 그 중에서는 상대적으로 강했지만, 전반적으로 HistoSegNet이 우세했다. 정성적으로도 SEC와 DSRG는 contour를 어느 정도 맞추지만 object 크기를 과장하는 경향이 있었고, HistoSegNet은 작은 vessel이나 세밀한 tissue type를 더 잘 보존했다.

PASCAL VOC2012에서는 결과가 반대로 나왔다. SEC와 DSRG가 Grad-CAM baseline을 일관되게 개선했고, 특히 SEC가 가장 강했다. HistoSegNet은 이 도메인에서 잘 맞지 않았다. 정성적 결과를 보면, 자연 영상에서는 self-supervised loss가 discriminative region만 잡은 CAM을 더 넓은 object 영역으로 확장하는 데 실제로 도움이 되었다. 반면 HistoSegNet은 CAM을 CRF로 다듬는 방식이라, 서로 자주 함께 등장하는 class를 잘못 연결하는 경우가 있었다. 예를 들어 `diningtable`과 `person`이 섞인 장면에서 잘못된 확장이 발생했다.

DeepGlobe에서는 DSRG와 IRNet이 상대적으로 좋았고, DSRG가 전체적으로 가장 우세했다. 흥미로운 점은 DeepGlobe에서는 Grad-CAM 자체가 대략적인 segment 위치를 이미 꽤 잘 잡고 있었고, 따라서 self-supervised 확장이 VOC만큼 극적 이득을 주지는 않았다. 그래도 DSRG와 IRNet은 baseline보다 좋아졌고, HistoSegNet도 일부 세부 구조를 유지하는 데 도움을 주었다. 그러나 fully-supervised 최고 성능인 DFCNet의 52.24% mIoU와 비교하면 weakly-supervised 성능은 여전히 많이 낮았다.

논문은 추가 분석으로 세 가지 패턴을 제시한다.

첫째, classification cue의 sparsity가 중요했다. ground-truth instance 수가 적은 데이터셋에서는 VGG16처럼 더 coarse한 cue가 유리했고, instance가 많고 segment가 작은 데이터셋에서는 M7/X1.7처럼 finer feature map을 가진 네트워크가 유리했다. 논문은 이 효과가 평균 5.22% mIoU 차이를 만들었다고 보고한다.

둘째, self-supervised learning이 유리한 경우와 불리한 경우가 갈렸다. 저자들은 seed recall이 낮은 데이터셋, 즉 thresholded CAM seed가 실제 object를 많이 놓치는 경우에는 SEC, DSRG, IRNet 같은 self-supervised 방법이 좋았다고 본다. 반대로 seed recall이 높은 데이터셋에서는 HistoSegNet 같은 비-self-supervised 방식이 더 낫다고 분석한다. 논문은 경험적으로 mean recall이 약 40%보다 낮으면 self-supervised, 40% 이상이면 non-self-supervised 방법이 더 적절할 수 있다고 제안한다.

셋째, class co-occurrence는 중요한 문제였다. 특히 DeepGlobe에서는 여러 land-cover class가 자주 함께 등장했다. 저자들은 많은 class를 포함한 이미지를 training set에서 절반 제거해 co-occurrence를 낮추는 간단한 balancing을 수행했다. 그 결과 `agriculture`, `forest`, `water` 같은 class의 IoU는 개선되었지만, 전체 mean mIoU는 27.5%에서 26.5%로 오히려 약간 감소했다. 즉, co-occurrence 완화는 분명 중요하지만, 단순한 데이터 제거 방식은 전체적으로 최선은 아니었다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 특정 알고리즘의 소폭 성능 향상보다 더 중요한 질문을 던졌다는 점이다. 즉, “자연 영상에서 잘 되던 WSSS가 다른 도메인에서도 잘 되느냐”를 직접 검증했다. 그리고 단순한 성능 비교에 그치지 않고, 왜 그런 결과가 나오는지를 cue sparsity, seed recall, class co-occurrence라는 관점에서 분석했다. 이는 실제로 새로운 데이터셋에 WSSS를 적용하려는 연구자에게 매우 실용적인 지침이 된다.

또 다른 강점은 비교 설정을 최대한 공정하게 맞추려 했다는 점이다. 서로 다른 논문에서 원래 사용한 backbone이 달라서 성능 차이가 backbone 때문인지 방법론 때문인지 혼동될 수 있는데, 저자들은 VGG16과 M7/X1.7을 공통으로 사용해 이를 줄이려 했다. 또한 quantitative evaluation뿐 아니라 qualitative visualization을 함께 제시해, 각 방법이 어떤 식으로 실패하는지 이해하기 쉽게 만들었다.

한편 한계도 분명하다. 첫째, 비교 대상이 완전히 동일 구현은 아니다. SEC, DSRG, IRNet의 공개 구현을 수정해 사용했고, 특히 DSRG의 경우 원래 방법에서 쓰인 DRFI background cue를 그대로 재현하지 못했다고 논문에서 인정한다. 저자들도 자신들의 VOC 결과가 원 논문보다 다소 낮은 이유로 이런 구현 차이를 언급한다. 따라서 절대적 성능 수치보다는 상대적 경향에 더 무게를 두어야 한다.

둘째, 데이터셋 수가 제한적이다. natural scene, histopathology, satellite의 각 대표 데이터셋 하나씩을 중심으로 분석했기 때문에, 모든 비자연 영상 도메인에 일반화된다고 단정할 수는 없다. urban scene dataset은 관련 배경 설명에는 포함되지만 실제 핵심 실험에는 들어가지 않았다.

셋째, 논문이 제안하는 실천적 원칙들, 예를 들어 “seed recall 40%를 기준으로 self-supervised 여부를 고르자” 같은 부분은 실험적 관찰에 기반한 heuristic에 가깝다. 매우 유용한 경험 법칙이지만, 이 기준이 이론적으로 증명된 것은 아니다.

넷째, class co-occurrence 문제를 다루는 방식은 아직 초기적이다. balancing이 일부 class를 개선했지만 전체 성능은 떨어졌고, 저자들 역시 더 나은 접근이 필요하다고 말한다. 즉, 문제 제기는 강하지만 완전한 해결책은 아직 제시되지 않았다.

## 6. 결론

이 논문은 weakly-supervised semantic segmentation이 자연 영상에서는 매우 유망하지만, 다른 이미지 도메인에서는 같은 방식이 그대로 통하지 않을 수 있음을 설득력 있게 보여준다. SEC와 DSRG는 PASCAL VOC2012 같은 natural scene에서 가장 강했고, HistoSegNet은 ADP 같은 histopathology에서 가장 강했다. DeepGlobe 같은 satellite domain에서는 일부 자연 영상용 방법이 어느 정도 통했지만, 성능은 여전히 제한적이었다.

저자들의 핵심 기여는 세 가지로 요약할 수 있다. 첫째, 서로 다른 도메인에서 WSSS 방법들을 직접 비교해 일반화 한계를 보여주었다. 둘째, classification cue sparsity와 seed recall이 어떤 방법이 적합한지 결정하는 중요한 신호임을 보였다. 셋째, class co-occurrence가 image-level supervision 기반 WSSS의 본질적 어려움 중 하나임을 실험으로 확인했다.

실제 적용 측면에서 이 연구는 매우 중요하다. 의료 영상, 위성 영상처럼 pixel annotation이 특히 비싼 분야에서는 WSSS의 잠재력이 크지만, 현재의 주류 방법론은 그 도메인 특성을 충분히 반영하지 못한다. 따라서 향후 연구는 단순히 seed recall을 높이는 방향뿐 아니라, 모호한 경계를 정교하게 다루는 loss, 높은 class co-occurrence를 완화하는 학습 전략, 그리고 도메인 특화 또는 도메인 일반화가 가능한 WSSS 설계로 나아갈 필요가 있다.
