# Prior Guided Feature Enrichment Network for Few-Shot Segmentation

- **저자**: Zhuotao Tian, Hengshuang Zhao, Michelle Shu, Zhicheng Yang, Ruiyu Li, Jiaya Jia
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2008.01449

## 1. 논문 개요

이 논문은 **few-shot semantic segmentation** 문제를 다룬다. 즉, 새로운 클래스를 분할해야 하지만 해당 클래스에 대해 주어진 라벨 데이터가 매우 적은 상황에서, support image 몇 장만 보고 query image의 해당 객체를 픽셀 단위로 분할하는 것이 목표다. 기존 semantic segmentation 모델은 충분한 fully-labeled data가 있어야 잘 동작하고, 학습 중 보지 못한 unseen class에는 fine-tuning 없이는 약하다는 한계가 있다.

저자들은 기존 few-shot segmentation 방법이 특히 두 가지 문제를 충분히 해결하지 못한다고 본다. 첫째는 **high-level semantic feature의 부적절한 사용으로 인한 generalization 저하**이고, 둘째는 **support와 query 객체 사이의 spatial inconsistency**이다. support의 객체 크기, 자세, 위치가 query와 크게 다를 수 있기 때문에, support에서 얻은 정보를 query의 모든 위치에 단순하게 대응시키는 방식은 비효율적이라는 것이다.

이 문제는 중요하다. few-shot segmentation이 잘 되면 새로운 클래스에 대해 대규모 재학습 없이 빠르게 적응할 수 있으므로, 자동 주행, 로봇 비전, 의료 영상처럼 새로운 대상이 계속 등장하는 환경에서 실용성이 크다. 논문은 이를 위해 **PFENet (Prior Guided Feature Enrichment Network)** 를 제안하며, 적은 파라미터와 높은 속도를 유지하면서도 PASCAL-5^i와 COCO에서 당시 SOTA를 달성했다고 보고한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 두 부분으로 구성된다.

첫째, **학습되지 않는(training-free) prior mask 생성 방식**이다. 기존 연구는 high-level feature를 직접 학습 과정에 강하게 끌어들이면, 학습 클래스인 $C_{train}$ 에 편향되어 unseen class인 $C_{test}$ 에 대한 일반화가 약해진다고 본다. 저자들은 이를 피하기 위해, **ImageNet pre-trained backbone의 고정된 high-level feature**를 사용하여 query와 support 사이의 pixel-wise correspondence를 계산하고, 이를 prior mask로 바꾼다. 이렇게 하면 semantic cue는 활용하되, prior 생성 자체는 학습되지 않으므로 base class 편향을 줄일 수 있다는 주장이다.

둘째, **Feature Enrichment Module (FEM)** 이다. 기존 방법은 support feature를 masked global average pooling으로 하나의 전역 벡터로 만들고 query에 매칭하는 경우가 많다. 하지만 이 방식은 support 객체와 query 객체의 크기 차이, 위치 차이, 자세 차이를 충분히 반영하지 못한다. FEM은 query feature를 여러 spatial scale로 나눈 뒤, 각 scale에서 support feature와 prior mask를 함께 결합하고, scale 간에도 top-down으로 정보를 전달해 query feature를 더 잘 보강한다.

기존 접근과의 차별점은 명확하다. 논문은 단순히 multi-scale context를 넣는 PPM, ASPP 같은 구조보다, **support-query-prior의 scale별 결합**과 **선택적 inter-scale interaction**이 더 중요하다고 주장한다. 즉, 이 논문은 일반 semantic segmentation의 문맥 집계 구조를 그대로 가져오지 않고, few-shot segmentation에 맞게 정보 흐름을 다시 설계했다.

## 3. 상세 방법 설명

few-shot segmentation episode는 query image $I_Q$ 와, 같은 클래스의 support set $S=\{(I_{S_i}, M_{S_i})\}_{i=1}^K$ 로 구성된다. 목표는 query image에서 unseen class 영역을 예측하는 것이다. 학습 클래스와 테스트 클래스는 겹치지 않으며, $C_{train} \cap C_{test} = \emptyset$ 이다.

### Prior generation

논문은 backbone의 high-level feature를 사용해 prior를 만든다. query와 support의 high-level feature를 각각 $X_Q$, $X_S$ 라고 하면,

$$
X_Q = F(I_Q), \quad X_S = F(I_S) \odot M_S
$$

이다. 여기서 $\odot$ 는 Hadamard product이고, support mask를 곱해 support feature의 배경을 0으로 만든다. 이후 query의 각 픽셀 feature $x_q$ 와 support의 모든 픽셀 feature $x_s$ 사이의 cosine similarity를 계산한다.

$$
\cos(x_q, x_s) = \frac{x_q^T x_s}{\|x_q\|\|x_s\|}
$$

query의 한 픽셀에 대해 support 전체 픽셀 중 최대 similarity를 correspondence 값으로 둔다.

$$
c_q = \max_s \cos(x_q, x_s)
$$

이 값들을 모아 query spatial map으로 reshape한 뒤, min-max normalization으로 $[0,1]$ 범위의 prior mask $Y_Q$ 를 만든다.

$$
Y_Q = \frac{Y_Q - \min(Y_Q)}{\max(Y_Q) - \min(Y_Q) + \epsilon}
$$

이 prior는 “query의 어느 위치가 support의 target과 닮았는가”를 나타내는 pixel-wise 힌트다. 중요한 점은 이 전체 과정이 **고정된 backbone feature 위에서 수행되므로 학습되지 않는다**는 것이다.

### Feature Enrichment Module (FEM)

FEM의 입력은 query feature, pooled support feature, prior mask다. 핵심은 세 단계다.

첫 번째는 **inter-source enrichment** 이다. query feature를 여러 spatial size $B=[B_1, B_2, ..., B_n]$ 로 adaptive average pooling한다. support feature는 global pooled vector를 각 scale에 맞게 확장하고, prior도 각 scale에 맞게 resize한다. 그런 다음 각 scale에서 query, support, prior를 concat하고 $1\times1$ convolution으로 결합한다.

$$
X^i_{Q,m} = F_{1\times1}(X^i_Q \oplus X^i_S \oplus Y^i_Q)
$$

여기서 $\oplus$ 는 concatenation이다.

두 번째는 **inter-scale interaction** 이다. 저자들은 특히 **top-down path** 가 효과적이라고 본다. finer scale의 feature가 coarser scale의 feature를 보강한다. 이를 위해 inter-scale merging module $M$ 을 사용한다. auxiliary feature를 main feature 크기에 맞게 resize한 뒤, $1\times1$ convolution으로 main feature에 조건부로 필요한 정보를 뽑고, 이어지는 두 개의 $3\times3$ convolution으로 정제한다. residual connection도 포함해 main feature의 원래 정보가 유지되도록 한다.

이 부분의 핵심은 단순한 scale fusion이 아니라, **보조 feature에서 필요한 정보만 선택적으로 전달**한다는 점이다. 저자들은 이것이 HRNet식 dense fusion보다 few-shot segmentation에 더 적합하다고 주장한다.

세 번째는 **information concentration** 이다. 모든 scale에서 정제된 feature를 원래 해상도로 upsample한 뒤 concat하고, 다시 $1\times1$ convolution으로 최종 query feature를 만든다.

$$
X_{Q,new} = F_{1\times1}(X^1_{Q,new} \oplus X^2_{Q,new} \oplus ... \oplus X^n_{Q,new})
$$

이후 convolution block과 classification head를 통해 최종 segmentation mask를 예측한다.

### PFENet 전체 구조와 loss

PFENet은 backbone에서 middle-level feature인 `conv3_x`, `conv4_x` 를 query-support feature로 사용하고, high-level feature인 `conv5_x` 를 prior 생성에 사용한다. 5-shot에서는 5개의 pooled support feature와 5개의 prior mask를 각각 평균내어 사용한다.

loss는 cross entropy이며, FEM 각 scale의 intermediate supervision loss와 최종 prediction loss를 함께 쓴다.

$$
L = \sigma \sum_{i=1}^{n} L^i_1 + L_2
$$

논문에서는 $\sigma=1.0$ 을 사용했다.

## 4. 실험 및 결과

평가는 PASCAL-5^i와 COCO에서 수행했다. PASCAL-5^i는 20개 클래스를 4개 fold로 나누고, COCO는 80개 클래스를 4개 fold로 나눈다. backbone은 VGG-16, ResNet-50, ResNet-101을 사용했다. backbone은 기본적으로 **고정(fixed)** 하며, 입력 이미지는 학습 시 $473 \times 473$ patch로 처리한다. 주요 평가지표는 **class mIoU** 이고, 비교를 위해 **FB-IoU** 도 함께 보고한다.

PASCAL-5^i에서 PFENet은 ResNet-50 기준으로 1-shot 평균 mIoU **60.8**, 5-shot 평균 mIoU **61.9**를 기록했다. 이는 CANet의 55.4 / 57.1, PGNet의 56.0 / 56.8보다 높다. VGG-16과 ResNet-101 backbone에서도 모두 기존 방법보다 좋았다. FB-IoU 역시 ResNet-50 기준 73.3 / 73.9로 더 높았다.

COCO에서는 성능 차이가 더 크다. 예를 들어 ResNet-101 기반 PFENet은 class mIoU에서 1-shot **32.4**, 5-shot **37.4**를 기록했고, 리사이즈된 라벨 기준 평가에서는 38.5 / 42.7까지 보고된다. 논문은 기존 방법 대비 10포인트 이상 향상되는 경우도 있다고 강조한다.

ablation study도 비교적 충실하다. FEM만 추가하면 baseline 56.3 / 58.0에서 **59.2 / 60.4**로 오른다. prior만 추가하면 **58.2 / 59.6**이 되고, 둘 다 넣으면 **60.8 / 61.9**가 된다. 즉 prior와 FEM이 모두 독립적으로 기여하며, 함께 쓸 때 가장 좋다.

FEM 내부에서도 top-down inter-scale interaction이 가장 낫다. spatial size는 `{60, 30, 15, 8}` 조합이 가장 좋았고, PPM, ASPP, GAU, HRNet 변형과 비교해도 성능 대비 파라미터 효율이 좋았다. 특히 HRB-TD-Cond는 FEM과 비슷한 성능이지만 파라미터가 더 많다.

prior 생성 방식에 대한 분석도 있다. 고정된 high-level feature로 prior를 만든 경우가 가장 좋았고, 학습 가능한 high-level feature를 사용하면 오히려 baseline보다 크게 나빠졌다. 이는 unseen class generalization이 무너졌기 때문이라고 해석한다. 또한 similarity를 평균내는 방식보다 max를 쓰는 방식이 더 좋았고, pooled support feature 기반 prior보다도 pixel-wise max correspondence가 더 효과적이었다.

흥미롭게도 zero-shot 실험도 수행했다. 이때는 support image 대신 Word2Vec과 FastText class embedding을 사용하고 prior는 제거했다. VGG-16 기반 baseline이 PASCAL-5^i에서 **53.2 mIoU**, FEM 추가 시 **54.2 mIoU**를 보였다. 논문은 이것을 PFENet 구조의 강건성 증거로 제시한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정이 분명하고, 제안 요소가 각각 어떤 문제를 해결하려는지 논리가 비교적 선명하다는 점이다. prior는 **generalization 저하 없이 semantic cue를 쓰기 위한 장치**이고, FEM은 **support-query의 spatial inconsistency를 다루기 위한 장치**다. 두 설계가 실험적으로도 분리 검증된다.

또 다른 강점은 효율성이다. ResNet-50 기반 PFENet은 learnable parameter가 **10.8M** 수준으로, 비교 대상보다 작거나 비슷하면서 성능은 더 높다. 속도도 1-shot에서 **15.9 FPS**로 보고되어, 무거운 구조를 붙여 얻은 성능 향상만은 아니라는 점이 설득력 있다.

ablation이 충분한 편이라는 점도 장점이다. prior의 feature source, max/avg 선택, inter-scale direction, FEM과 기존 multi-scale module 비교, backbone freeze 여부까지 비교한다. 따라서 “왜 잘 되는가”에 대해 저자 나름의 근거가 있다.

한계도 있다. 먼저, prior가 실제로 얼마나 class-agnostic한지에 대해서는 간접 증거가 많고, 이론적 보장은 없다. backbone이 ImageNet pretrained이므로, 완전히 새로운 개념적 클래스에 대해서도 일관되게 동작하는지까지는 제한적으로만 검증된다. FSS-1000 실험이 있긴 하지만, 그 역시 완전한 의미의 out-of-distribution 일반화를 보장하지는 않는다.

또한 COCO 평가 방식에서 저자들은 기존 연구의 1,000 pair 평가가 불안정하다고 보고 20,000 pair를 사용했다. 이것은 합리적이지만, 기존 논문과 완전히 동일한 프로토콜은 아닐 수 있어 직접 수치 비교에는 주의가 필요하다. 논문도 일부 결과에서 `†` 표시로 resized label 평가를 별도로 제시한다.

마지막으로, zero-shot 확장은 흥미롭지만 prior를 제거하고 class embedding을 대체 입력으로 넣는 수준이라, PFENet 전체가 zero-shot segmentation 문제를 완전히 해결했다고 보기는 어렵다. 이 부분은 구조의 확장 가능성을 보여주는 정도로 이해하는 것이 적절하다.

## 6. 결론

이 논문은 few-shot segmentation에서 자주 발생하는 두 문제, 즉 **unseen class generalization 저하**와 **support-query spatial inconsistency**를 해결하기 위해 PFENet을 제안했다. 핵심 기여는 고정된 high-level feature로부터 만드는 **training-free prior mask** 와, multi-scale에서 query를 support와 prior로 정제하는 **Feature Enrichment Module (FEM)** 이다.

실험 결과는 이 두 요소가 모두 실제 성능 향상에 기여하며, PFENet이 PASCAL-5^i와 COCO에서 강한 성능과 좋은 효율을 동시에 달성함을 보여준다. 특히 backbone을 고정한 상태에서 semantic prior를 안전하게 활용하는 아이디어는 이후 few-shot dense prediction 문제에도 확장될 여지가 크다. 논문이 직접 언급하듯, few-shot object detection이나 few-shot instance segmentation으로의 확장은 자연스러운 후속 연구 방향이다.
