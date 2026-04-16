# Constructing Self-motivated Pyramid Curriculums for Cross-Domain Semantic Segmentation: A Non-Adversarial Approach

- **저자**: Qing Lian, Fengmao Lv, Lixin Duan, Boqing Gong
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1908.09547

## 1. 논문 개요

이 논문은 synthetic source domain에서 학습한 semantic segmentation 모델을 real target domain에 잘 일반화시키기 위한 unsupervised domain adaptation 방법인 **PyCDA (self-motivated pyramid curriculum domain adaptation)** 를 제안한다. 대표적인 설정은 GTAV 또는 SYNTHIA에서 학습한 뒤 Cityscapes로 적응하는 문제이다. 핵심 목표는 target domain에 정답 라벨이 전혀 없는 상황에서도, target 이미지의 구조적 성질을 활용해 segmentation 성능을 끌어올리는 것이다.

연구 문제가 중요한 이유는 semantic segmentation이 픽셀 단위 정답이 필요해 annotation 비용이 매우 크기 때문이다. 반면 GTAV, SYNTHIA 같은 synthetic 데이터는 자동 라벨링이 가능해 값싸게 대규모 데이터를 만들 수 있다. 그러나 synthetic와 real 사이에는 강한 domain gap이 존재하므로, source에서만 학습한 모델은 real 이미지에서 성능이 크게 떨어진다. 이 논문은 이러한 gap을 줄이되, 당시 널리 쓰이던 adversarial adaptation처럼 discriminator를 따로 두거나 불안정한 $min\max$ 최적화를 수행하지 않는 더 단순한 대안을 제시한다.

저자들은 기존의 **curriculum domain adaptation (CDA)** 와 **self-training (ST)** 이 사실상 매우 유사한 구조를 가진다고 보고, 둘을 하나의 통합된 관점으로 재해석한다. 그 결과, global image level, local region level, pixel level의 supervision을 하나의 pyramid curriculum으로 결합하는 방법을 만든다.

## 2. 핵심 아이디어

논문의 중심 직관은 다음과 같다. target domain에 대해 직접 정답 라벨은 없지만, 모델 자신의 예측으로부터 상대적으로 신뢰할 수 있는 여러 수준의 “속성(properties)”을 뽑아낼 수 있다. 이 속성은 가장 거친 수준에서는 이미지 전체의 class distribution이고, 중간 수준에서는 작은 지역(square region)의 class label이며, 가장 미세한 수준에서는 개별 pixel의 pseudo label이다. 저자들은 이 여러 수준의 정보를 함께 쓰면, 단일 수준의 supervision보다 더 안정적이고 더 강한 adaptation 신호를 줄 수 있다고 본다.

기존 CDA는 target 이미지 전체나 superpixel 단위에서 class frequency distribution을 맞추는 방식이었다. 반면 기존 ST는 confidence가 높은 pixel에 pseudo label을 붙여 학습했다. 논문은 이 둘이 서로 다른 방법처럼 보이지만, 실제로는 모두 target에 대한 어떤 “추정된 라벨 정보”를 사용해 cross-entropy 형태의 제약을 거는 방식이라고 설명한다. 즉, pixel pseudo label도 target domain property의 한 종류로 볼 수 있다는 것이다.

이 통합 관점에서 PyCDA는 CDA와 ST를 단순 병렬 결합하는 것이 아니라, coarse-to-fine pyramid로 조직한다. 이미지 전체 수준의 분포는 어떤 클래스가 전체적으로 얼마나 등장해야 하는지 알려 주는 전역 prior 역할을 하고, square region 수준은 어느 위치 근방에서 어떤 클래스가 나올지에 대한 중간 수준의 구조 정보를 주며, pixel 수준 pseudo label은 가장 정밀한 supervision을 준다. 저자들은 이것이 “what to predict”와 “where to update”를 동시에 제공한다고 해석한다.

또 하나의 차별점은 **non-adversarial** 이라는 점이다. 당시 많은 domain adaptation 방법이 feature space나 output space에서 source/target를 맞추기 위해 discriminator를 두고 adversarial training을 수행했는데, 저자들은 PyCDA가 그런 추가 네트워크 없이도 competitive하거나 더 좋은 결과를 보인다고 주장한다.

## 3. 상세 방법 설명

### 3.1 문제 설정과 기본 표기

target 이미지 하나를 $I^t \in \mathbb{R}^{H \times W}$ 라고 하고, segmentation network의 출력 확률맵을 $\hat{Y}^t \in \mathbb{R}^{H \times W \times C}$ 라고 둔다. 여기서 $H, W, C$는 각각 높이, 너비, 클래스 수이다. 각 pixel $(i,j)$ 에 대해 $\hat{Y}^t(i,j,c)$ 는 클래스 $c$ 의 예측 확률이며, softmax 출력이므로 각 픽셀에서 클래스 확률의 합은 1이다.

source domain에는 라벨 $Y^s$ 가 주어지지만, target domain에는 라벨이 없다. 따라서 adaptation은 source supervision과 target에 대한 간접 제약을 함께 사용해 학습한다.

### 3.2 CDA와 ST의 연결

저자들은 먼저 기존 CDA의 목적식을 제시한다.

$$
\min \sum_{s \in S} L(Y^s, \hat{Y}^s) + \lambda \sum_{t \in T} \sum_{k \in P_t^1} C(p_t^k, \hat{p}_t^k)
$$

첫 번째 항은 source 라벨에 대한 일반적인 pixel-wise cross-entropy loss이다. 두 번째 항은 target 이미지에 대해, 미리 추정한 원하는 label distribution $p_t^k$ 와 네트워크가 예측한 distribution $\hat{p}_t^k$ 가 일치하도록 강제하는 항이다. 여기서 $k$ 는 전체 이미지 또는 이미지 내 특정 지역을 뜻한다.

예를 들어 이미지 전체에 대한 class distribution은 다음처럼 정의된다.

$$
p_t^0(c) = \frac{1}{WH}\sum_{i=1}^{W}\sum_{j=1}^{H}Y^t(i,j,c)
$$

이 식은 실제 target 정답이 있다면 이미지 안에서 클래스 $c$ 가 차지하는 비율을 뜻한다. 실제 adaptation에서는 target 정답이 없으므로 이 분포를 추정해야 한다.

반면 ST는 target의 pseudo label을 latent variable로 보고, pseudo label을 추정한 뒤 그것을 이용해 다시 네트워크를 학습한다. 논문이 제시한 ST의 형태는 다음과 같다.

$$
\min \sum_{s \in S} L(Y^s, \hat{Y}^s) + \lambda \sum_{t \in T}\sum_{(i,j)\in P_t^2} C(Y^t(i,j), \hat{Y}^t(i,j))
$$

여기서 $P_t^2$ 는 pseudo label이 부여된 target pixel 집합이다. confidence가 낮은 픽셀은 제외된다.

저자들이 강조하는 핵심은, CDA와 ST가 모두 “target에 대해 추정된 어떤 supervision signal”과 예측 사이의 cross-entropy를 최소화한다는 점이다. 차이는 CDA는 image/region distribution을 쓰고, ST는 pixel pseudo label을 쓴다는 것뿐이다. 이 관찰이 바로 두 방법을 한 pyramid로 통합하는 출발점이다.

### 3.3 PyCDA의 pyramid curriculum

PyCDA는 target 이미지마다 최소 3계층의 pyramid를 만든다.

첫째, **top layer** 는 이미지 전체이다. 여기서는 전체 이미지의 class distribution을 사용한다.

둘째, **middle layers** 는 작은 square region들이다. 논문은 $4\times4$, $8\times8$ 크기의 pixel square를 사용했다. 기존 CDA는 superpixel을 썼지만, superpixel 생성은 계산 비용이 크기 때문에 저자들은 단순한 square region으로 대체한다. square는 object boundary를 잘 따르지는 못하지만, GPU 연산이 쉽고 충분히 작으면 대체로 한 클래스 위주로 구성된다는 것이 논문의 판단이다.

셋째, **bottom layer** 는 pixel level이다. 여기서는 confidence가 높은 픽셀에 pseudo label을 부여한다.

이 구조는 coarse-to-fine curriculum으로 볼 수 있다. 전체 이미지 분포는 전역적인 클래스 prior를 제공하고, region은 위치와 지역적 일관성에 대한 신호를 주며, pixel은 세밀한 supervised signal을 제공한다.

### 3.4 target property의 self-motivated 추정

PyCDA에서 중요한 점은 target domain의 이런 property들을 별도 모델 없이 **현재 segmentation network 자체** 로부터 추정한다는 것이다. 이것이 self-motivated라는 이름의 이유다.

#### pixel pseudo label

한 픽셀 $(i,j)$ 에 대해 예측 확률이 가장 높은 클래스를

$$
c^* \leftarrow \arg\max_c \hat{Y}^t(i,j,c)
$$

로 둔다. 그리고 다음 규칙으로 pseudo label을 정한다.

$$
Y^t(i,j)=
\begin{cases}
c^* & \text{if } \hat{Y}^t(i,j,c^*) > 0.5\\
\text{null} & \text{otherwise}
\end{cases}
$$

즉, 최고 확률이 0.5를 넘는 픽셀만 살아남아 bottom layer의 supervision에 사용된다. 나머지는 무시된다.

#### square region label

각 square region에 대해서는 내부 픽셀 예측을 average pooling한다.

$$
\hat{Y}^{square}(i_0,j_0,c) \leftarrow \text{mean}_{(i,j)\in square}(\hat{Y}^t(i,j,c))
$$

이 pooled 값에 대해서도 pixel과 같은 방식의 thresholding을 적용해 square 전체의 클래스를 정한다. 이렇게 얻은 square label은 one-hot vector로 바꾸어 그 지역의 label distribution처럼 다룬다. 구현에서는 output 위에 average pooling layer를 추가해 효율적으로 계산한다.

이 아이디어는 region 내부 픽셀들의 “합의(consensus)”를 이용해 pseudo supervision의 신뢰도를 높이는 것으로 해석할 수 있다. 픽셀 하나의 예측은 불안정할 수 있지만, 작은 지역 평균은 더 안정적일 수 있다.

#### full-image class distribution

전체 이미지 class distribution은 더 단순하게 추정한다. 논문은 각 target 이미지마다 별도 추정 모델을 두지 않고, **source domain 전체 이미지들의 평균 class distribution을 그대로 target 이미지에 전달** 한다. 이는 urban scene이 비슷한 구성과 spatial layout을 공유한다는 가정에 기대는 방식이다. 저자들은 [38]의 실험 결과를 근거로, 이 단순한 평균 분포가 logistic regression 같은 별도 학습 기반 추정보다도 성능이 크게 뒤지지 않는다고 설명한다.

다만 이 부분은 도시 장면이 아닌 일반적인 데이터셋으로 확장될 때는 더 정교한 추정이 필요할 수 있다고 저자들이 직접 인정한다.

### 3.5 최종 목적함수

PyCDA의 전체 목적식은 다음과 같다.

$$
\min \frac{1}{|S|}\sum_{s\in S}L(Y^s,\hat{Y}^s)
+ \lambda_1\frac{1}{|T|}\sum_{t\in T} C(p_t^0,\hat{p}_t^0)
+ \lambda_2\frac{1}{|P|}\sum_{(t,k)\in P} C(Y_t^k,\hat{Y}_t^k)
$$

여기서 두 번째 항은 target 이미지 전체 분포에 대한 loss이고, 세 번째 항은 target의 square region과 pixel pseudo label에 대한 loss이다. 즉, top layer는 distribution matching, middle/bottom layers는 더 구체적인 pseudo supervision을 담당한다. 논문에서는 $\lambda_1=1$, $\lambda_2=0.5$ 를 사용했다.

### 3.6 학습 절차

논문이 설명한 실제 학습 흐름은 다음과 같이 이해할 수 있다.

1. 먼저 source 데이터만으로 segmentation network를 pre-train한다.
2. 이후 fine-tuning 단계에서, 현재 네트워크가 target 이미지에 대해 pixel pseudo label, square label, image-level distribution 관련 신호를 생성한다.
3. 이 self-generated supervision과 source 정답 라벨을 함께 사용해 네트워크를 다시 업데이트한다.
4. 네트워크가 바뀌면 target property 추정도 갱신되므로, 이런 self-motivated adaptation이 반복된다.

논문은 source pre-training 30000 iteration 후, PyCDA로 30000 iteration 추가 fine-tuning을 수행했다고 밝힌다. optimizer는 momentum 0.9의 SGD이고, 초기 learning rate는 0.016이며 fine-tuning 단계에서 10배 감소시켰다.

또 test 단계에서 adaptive batch normalization인 adabn을 적용해, batch normalization의 평균과 분산을 source+target 혼합 통계가 아니라 target domain 통계로 바꿨다고 적고 있다.

## 4. 실험 및 결과

### 4.1 실험 설정

실험은 대표적인 synthetic-to-real semantic segmentation adaptation benchmark 두 가지에서 수행된다.

첫째는 **GTAV $\rightarrow$ Cityscapes** 이다. GTAV는 24,966장의 synthetic urban scene 이미지와 19개 클래스 라벨을 제공한다. Cityscapes는 real-world urban scene 데이터셋이며, 공식 train/val/test 분할을 갖는다. 논문은 prior work [38]을 따라 공식 validation set 500장을 최종 test set으로 사용하고, 공식 training set 2,975장 중 500장을 validation, 나머지 2,475장을 unlabeled target train으로 사용했다.

둘째는 **SYNTHIA $\rightarrow$ Cityscapes** 이다. SYNTHIA-RAND-CITYSCAPES subset을 사용하며, Cityscapes와 공통인 16개 클래스를 맞춘다.

평가 지표는 Cityscapes 공식 evaluation code가 사용하는 **IoU** 와 **mIoU** 이다. 각 클래스에 대해

$$
IoU = \frac{TP}{TP + FP + FN}
$$

로 계산하고, 클래스 평균을 mIoU로 보고한다.

### 4.2 백본과 구현 세부

저자들은 다양한 기존 방법과 비교하기 위해 여러 backbone을 사용한다.

- FCN8s with VGG-16
- ResNet-38
- PSPNet with ResNet-101

모든 backbone은 ImageNet pretraining을 사용했다. 입력 이미지는 폭 1024로 resize하고, 학습 중에는 random crop을 사용했다. 테스트 시에는 폭 1024의 전체 이미지를 입력한 뒤, 출력 segmentation mask를 원본 크기 $2048\times1024$ 로 다시 resize해 평가했다.

### 4.3 GTAV to Cityscapes 결과

GTAV $\rightarrow$ Cityscapes에서 PyCDA는 여러 backbone에서 매우 강한 결과를 보인다.

VGG-16 기반에서는 source only가 24.3 mIoU인데, PyCDA는 **37.2 mIoU** 로 상승한다. 이는 CDA 31.4, ST 28.1, CBST 30.9, ROAD 35.9, CyCADA 35.4, CLAN 36.6, ADVENT 35.6보다 높다.

ResNet-38 기반에서는 source only 35.4에서 PyCDA가 **48.0 mIoU** 를 기록한다. 이는 ST 41.5, CBST 45.2보다 높다.

ResNet-101 기반에서는 source only 34.2에서 PyCDA가 **47.4 mIoU** 를 달성한다. 이는 OutputAdapt 42.4, FCAN 46.6, CLAN 43.2, ADVENT 44.8보다 높다.

논문은 특히 dominant class인 road, building, vegetation, car 같은 클래스에서 강점을 보인다고 해석한다. 동시에 rider, wall, fence 같은 작은 객체에서도 CBST보다 더 나은 경우가 있다고 주장한다. 표를 보면 PyCDA는 일부 클래스에서는 경쟁 방법보다 낮기도 하지만, 전체적으로 mIoU에서 가장 높은 수치를 낸다.

중요한 맥락은, 논문이 prior method 대부분은 Cityscapes training set 전체를 target training에 사용한 반면, 자신들은 그중 500장을 validation으로 따로 떼어 모델 선택에 썼다고 밝힌다는 점이다. 즉, target unlabeled training image 수가 더 적은 조건에서도 좋은 성능을 냈다는 것이 저자들의 주장이다.

### 4.4 SYNTHIA to Cityscapes 결과

SYNTHIA $\rightarrow$ Cityscapes에서도 PyCDA는 좋은 결과를 보인다.

VGG-16 기반에서는 source only 22.4에서 PyCDA가 **35.9 mIoU** 로 오른다. 이는 CDA 29.7, ST 23.9, CBST 35.4, ROAD 36.2, ADVENT 31.4와 비교 가능한 수준이며, 전체 mIoU 기준으로 표에서는 최고 수치다.

ResNet-101 기반에서는 source only 33.0에서 PyCDA가 **46.7 mIoU**, 그리고 13개 클래스 기준 mIoU*는 **53.3** 이다. 이는 OutputAdapt 46.7, CLAN 47.8, ADVENT 48.0과 비교할 때, 표의 마지막 열 기준으로 PyCDA가 가장 높다. 특히 건물, 신호등, 표지판, vegetation, sky, person, car, bus 등에서 강한 수치를 보인다.

즉, 논문이 제안한 pyramid curriculum이 GTAV뿐 아니라 SYNTHIA 같은 다른 synthetic domain에서도 일관되게 동작함을 보여 주려는 실험이다.

### 4.5 Ablation study

이 논문의 설득력은 ablation에서 꽤 잘 드러난다. Table 3은 top layer, bottom layer, pixel squares를 각각 넣거나 빼면서 성능을 비교한다.

예를 들어 GTAV 기준 VGG-16에서:

- source only: 24.3
- top만 사용: 28.0
- bottom만 사용: 32.6
- top + bottom: 34.9
- top + pixel squares: 35.4
- PyCDA: 37.2

ResNet-101에서도 비슷하게:

- source only: 34.2
- top만: 42.0
- bottom만: 40.6
- top + bottom: 46.2
- top + pixel squares: 46.3
- PyCDA: 47.4

즉, top과 bottom을 결합하면 각각 단독보다 좋아지고, 여기에 middle layer인 pixel squares까지 넣으면 추가 향상이 생긴다. 이는 논문의 주장대로 coarse supervision과 fine supervision이 상호보완적임을 지지한다.

또 superpixel 대신 pixel squares를 쓰는 대체가 성능 저하 없이 계산량을 줄인다는 것도 보여 준다. 표에서 top + pixel squares와 top + superpixels 성능은 거의 비슷하다. 논문은 superpixel 생성이 이미지당 약 3.6초 걸린다고 말하며, square 기반이 훨씬 실용적이라고 주장한다.

### 4.6 middle layer 개수에 대한 추가 분석

Appendix에서는 middle layer 수를 바꾸는 실험도 제시한다. ResNet-101, GTAV $\rightarrow$ Cityscapes 기준으로, middle layer를 추가하면 대체로 성능이 오른다.

- middle layer 없음: 46.3
- $4\times4$ 추가: 46.9
- $8\times8$ 추가: 47.4
- $16\times16$: 47.5
- $32\times32$: 47.5
- $64\times64$: 47.3
- $128\times128$: 47.0

즉, 중간 크기의 square를 추가하는 것이 유효하고, 너무 큰 square는 오히려 성능과 속도에 불리할 수 있다는 해석이 가능하다.

### 4.7 정성적 결과

논문은 정성적 예시도 제시한다. GTAV에서 적응한 모델이 SYNTHIA에서 적응한 모델보다 전반적으로 더 나은 segmentation을 보이며, 특히 road 클래스에서 강하다고 해석한다. 저자들은 이를 GTAV가 시각적 appearance와 spatial layout 면에서 real self-driving scene에 더 가깝기 때문이라고 본다. 이 주장은 Fig. 3, Fig. 4와 GTAV 기반 road IoU 90.5를 근거로 한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 개념적으로 단순하지만 설득력 있는 통합 프레임을 제시했다는 점이다. CDA와 ST를 별개의 계열로 취급하지 않고, 모두 target property에 대한 posterior regularization이라는 시각으로 묶은 뒤, 그것을 실제로 성능 향상으로 연결했다. 특히 image-level, region-level, pixel-level을 한 pyramid로 연결한 설계는 semantic segmentation처럼 공간 구조가 중요한 문제에 잘 맞는다.

또한 실용적 측면의 강점도 분명하다. adversarial adaptation과 달리 discriminator가 필요 없고, $min\max$ 최적화를 하지 않아 학습이 단순하다. superpixel 대신 square pooling을 사용한 것도 구현과 연산 효율 측면에서 합리적이다. ablation 결과도 각 구성 요소의 기여를 꽤 분명하게 보여 준다.

다만 한계도 있다. 첫째, target property 추정이 모두 현재 모델의 예측에 의존하므로, 초기 예측이 심하게 틀리면 잘못된 pseudo supervision이 누적될 위험이 있다. 논문은 thresholding과 region 평균으로 이를 완화하지만, noise propagation 문제를 근본적으로 해결했다고 보기는 어렵다.

둘째, full-image class distribution을 source 평균 분포로 대체하는 가정은 urban scene처럼 비교적 구조가 비슷한 환경에서는 통할 수 있으나, 더 일반적인 장면에서는 성립하지 않을 수 있다. 저자들도 도시 장면을 벗어나는 경우에는 더 sophisticated한 추정 알고리즘이 필요하다고 명시한다.

셋째, pseudo label threshold를 0.5로 두는 등 몇몇 설계는 경험적이다. 왜 이 값이 가장 적절한지에 대한 이론적 분석은 없다. 또한 square가 object boundary를 무시한다는 한계도 있다. 논문은 작기만 하면 대부분 한 클래스라고 보지만, 경계가 복잡한 장면에서는 mixed region이 생길 수 있다.

넷째, 논문은 strong empirical result를 보이지만, 일부 비교는 backbone, resolution, pretraining, target split 차이 등으로 완전히 동일 조건이라고 하기는 어렵다. 저자들은 이런 차이를 표 설명에 적어 두었지만, 따라서 결과를 절대적으로 해석하기보다는 “실용적으로 매우 경쟁력 있다” 정도로 보는 것이 적절하다.

비판적으로 보면, 이 방법은 adversarial 방법의 완전한 대체라기보다, self-training과 structured regularization을 잘 결합한 강한 baseline 또는 독립적인 대안으로 보는 편이 더 정확하다. 실제로 논문도 style transfer나 adversarial training과 orthogonal하다고 말한다. 즉, 이 방법 자체가 하나의 완결된 방향이면서 동시에 다른 adaptation 기법과 결합 가능한 모듈처럼 이해할 수 있다.

## 6. 결론

이 논문은 semantic segmentation의 unsupervised domain adaptation에서 **self-training과 curriculum domain adaptation을 하나의 관점으로 연결** 하고, 이를 바탕으로 **PyCDA** 라는 self-motivated pyramid curriculum을 제안했다. 핵심은 target 이미지에 대해 전체 이미지 분포, 작은 square region의 라벨, 개별 픽셀 pseudo label을 함께 사용하여 coarse-to-fine supervision을 구성하는 것이다.

실험적으로 PyCDA는 GTAV $\rightarrow$ Cityscapes와 SYNTHIA $\rightarrow$ Cityscapes에서 당시 강력한 adversarial 및 non-adversarial 방법들과 비교해 매우 경쟁력 있거나 더 좋은 성능을 보였다. 특히 추가 discriminator 없이도 높은 mIoU를 달성했다는 점이 실용적으로 의미 있다.

향후 연구 측면에서 이 논문은 두 가지 의미를 가진다. 하나는 segmentation adaptation에서 pseudo label을 단순히 pixel level에서만 다루지 않고, multi-scale structured supervision으로 확장하는 관점을 제시했다는 점이다. 다른 하나는 domain adaptation에서 반드시 adversarial training만이 답은 아니라는 점을 보여 주었다는 것이다. 따라서 실제 적용에서는 synthetic-to-real segmentation 파이프라인의 가볍고 안정적인 adaptation 모듈로 활용될 가능성이 있고, 후속 연구에서는 더 정교한 target property 추정이나 다른 구조적 prior와의 결합으로 확장될 여지가 크다.
