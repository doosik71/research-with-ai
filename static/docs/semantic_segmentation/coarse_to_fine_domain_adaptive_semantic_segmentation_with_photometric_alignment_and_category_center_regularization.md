# Coarse-to-Fine Domain Adaptive Semantic Segmentation with Photometric Alignment and Category-Center Regularization

- **저자**: Haoyu Ma, Xiangru Lin, Zifeng Wu, Yizhou Yu
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2103.13041

## 1. 논문 개요

이 논문은 unsupervised domain adaptation (UDA) semantic segmentation 문제를 다룬다. 구체적으로는, label이 있는 synthetic source domain 데이터로 학습한 segmentation 모델을, label이 없는 real-world target domain에서도 잘 동작하게 만드는 것이 목표다. 대표적인 예로 GTA5 또는 SYNTHIA 같은 합성 데이터로 학습하고 Cityscapes 같은 실제 도로 장면 데이터에 적용하는 상황이 해당된다.

저자들은 domain gap의 주요 원인을 크게 두 가지로 나눈다. 첫째는 **image-level domain shift**로, 조명, 노출, 대비, 카메라 imaging pipeline 차이 같은 저수준 photometric 차이이다. 둘째는 **category-level domain shift**로, 각 semantic category의 feature distribution과 category configuration 차이이다. 저자들의 핵심 문제의식은, 기존 방법들이 이 둘 중 하나만 주로 다루는 경향이 있어 전체적인 adaptation 성능이 충분히 올라가지 않는다는 점이다.

이 문제는 semantic segmentation에서 특히 중요하다. segmentation은 픽셀 단위 예측이기 때문에 단순한 global style 차이뿐 아니라, class 간 feature separation이 충분히 확보되지 않으면 실제 환경에서 쉽게 오분류가 발생한다. 논문은 이 점을 반영해, coarse 단계에서는 image-level alignment를 수행하고, fine 단계에서는 category-level feature regularization을 수행하는 **coarse-to-fine pipeline**을 제안한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 domain shift를 하나의 현상으로 뭉뚱그려 다루지 않고, **거친 수준의 photometric mismatch**와 **세밀한 수준의 category feature mismatch**로 나누어 순차적으로 해결하는 것이다. 즉, 먼저 source 이미지를 target처럼 보이게 대략 맞춘 뒤, 그 위에서 class별 feature center와 pseudo-label consistency를 이용해 더 정교하게 분포를 조정한다.

이 논문의 차별점은 크게 세 가지다.

첫째, **Global Photometric Alignment (GPA)** 라는 매우 가벼운 image alignment 방식을 제안한다. GAN 기반 style transfer처럼 별도 생성 모델을 학습하지 않고, Fourier 변환 기반 방법처럼 주파수 치환에 따른 artifact 문제에 크게 의존하지도 않는다. 대신 Lab color space에서 색상 채널과 밝기 채널을 다르게 처리한다.

둘째, source domain에서는 **Category-oriented Triplet Loss (CTL)** 를 사용해 class center 간 거리를 넓히고, 각 픽셀 feature가 자기 class center에는 더 가깝고 다른 class center와는 더 멀어지도록 만든다. 기존의 category anchor 기반 접근이 사실상 “center를 기준으로 맞추기만” 했다면, 이 논문은 class 간 margin을 적극적으로 키우는 soft regularization을 강조한다.

셋째, target domain에서는 **Target-domain Consistency Regularization (TCR)** 을 사용한다. 같은 target 이미지의 원본과 augmentation된 버전에 대해 예측이 일관되도록 강제함으로써, label이 없는 target domain에서도 category-level feature structure를 더 안정적으로 학습하게 한다.

## 3. 상세 방법 설명

전체 파이프라인은 크게 세 단계로 구성된다.

### 3.1 전체 파이프라인

먼저 source training set을 $M=\{m_k\}_{k=1}^{N_s}$, target training set을 $N=\{n_k\}_{k=1}^{N_t}$로 둔다.

**Step 0: Coarse Alignment**

source 이미지 $m$와 무작위로 선택한 target reference 이미지 $n$를 Lab color space로 변환해 $(L_m, a_m, b_m)$와 $(L_n, a_n, b_n)$를 얻는다. 이후:

- $a_m$, $b_m$에는 histogram matching 함수 $f_{match}$를 적용한다.
- $L_m$에는 gamma correction 함수 $f_{gamma}$를 적용한다.

그래서 정렬된 source 이미지는 다음과 같이 만들어진다.

$$
(f_{gamma}(L_m), f_{match}(a_m), f_{match}(b_m))
$$

이를 다시 RGB로 변환해 aligned image $m'$를 만들고, 정렬된 source set $M'$를 구성한다. 이후 stochastic augmentation 함수 $\tau$를 적용한 $\tau(M')$로 segmentation model $T_0$를 segmentation loss $L_{seg}$만으로 학습한다.

**Step 1: Category-level Feature Distribution Regularization**

이제 초기 모델 $T_0$를 이용해 target 이미지에 pseudo label을 만든다. 또한 source domain의 feature로부터 class별 category center $f_c$를 계산한다. 그 다음 $\tau(M')$와 target set $N$를 함께 사용해 모델 $T_1$을 fine-tuning한다. 이때 loss는 다음 세 가지를 합친다.

$$
L_{seg} + L_{triplet} + L_{consist}
$$

여기서:

- $L_{seg}$: source GT 및 신뢰도 높은 target pseudo-label에 대한 cross-entropy
- $L_{triplet}$: source domain의 CTL
- $L_{consist}$: target domain의 consistency regularization

**Step 2 to K: Iterative Self-Supervised Training**

이 과정을 self-training처럼 반복한다. 즉, $T_{i-1}$로 pseudo-label과 category center를 다시 계산하고, 이를 이용해 $T_i$를 학습한다. 논문에서는 $K=6$으로 설정했다.

### 3.2 Global Photometric Alignment (GPA)

저자들은 low-level domain shift는 주로 픽셀 수준의 photometric 차이에서 온다고 본다. 이를 위해 source와 target 이미지를 Lab space에서 정렬한다.

핵심은 **L 채널과 a/b 채널을 다르게 다루는 것**이다.

- $a$, $b$ 채널: 비교적 bell-shaped histogram을 가지므로 histogram matching을 적용
- $L$ 채널: 공간적 조명 변화가 복잡해 단순 histogram matching을 하면 overexposure나 가짜 구조가 생기기 쉬우므로 gamma correction 사용

#### 밝기 채널 정렬

저자들은 $L$ 채널 전체 histogram을 강제로 맞추지 않고, 평균 밝기만 target과 맞추도록 한다. gamma 변환은 다음과 같다.

$$
f_{gamma}(L) = L^\gamma
$$

여기서 $L$은 정규화된 lightness 값이다. 평균값 제약은 다음과 같이 쓴다.

$$
\sum_L L p_m^s(f_{gamma}(L)) = \sum_L L p_m^s(L^\gamma) = \sum_L L p_n^t(L)
$$

여기서 $p_m^s$는 source image $m$의 lightness histogram, $p_n^t$는 target reference image $n$의 lightness histogram이다.

실제로는 $\gamma$가 1에서 너무 멀어지지 않게 regularization을 둔다.

$$
\gamma^* = \arg\min_\gamma \left( \sum_L L p_m^s(L^\gamma) - \sum_L L p_n^t(L) \right)^2 + \beta(\gamma - 1)^2
$$

이 식은 변수 하나짜리 convex optimization 문제라 간단한 gradient descent로 풀 수 있다고 설명한다. 이 설계는 GAN처럼 별도 학습이 필요 없고, Fourier-based alignment보다 artifact가 적다는 것이 논문의 주장이다.

### 3.3 Category-oriented Triplet Loss (CTL)

GPA와 cross-entropy만으로는 class-wise feature distribution을 충분히 제어하지 못한다고 저자들은 본다. 어떤 class center들은 여전히 서로 가깝고, 특히 시각적으로 비슷한 class들은 target domain에서도 구분이 어렵다.

source domain에서 ground-truth label이 있으므로, 각 category $c$의 center는 다음처럼 계산한다.

$$
f_c = G\left(\frac{1}{N_c}\sum_s \sum_i \sum_j \mathbf{1}(y_{i,j}=c)x_{i,j}\right)
$$

여기서:

- $x_{i,j}$: penultimate layer의 픽셀 feature
- $y_{i,j}$: source domain의 픽셀 GT label
- $N_c$: class $c$에 속하는 전체 픽셀 수
- $G$: L2 normalization 함수

L2 normalization은 center들을 unit sphere 위에 놓아 scale 문제를 막는 역할을 한다.

그 다음 triplet loss는 다음과 같다.

$$
L_{triplet} =
\frac{1}{N_s}
\sum_s \sum_C \sum_i \sum_j
\max\left(
\|G(x_{i,j}) - f_{c=C}\| -
\|G(x_{i,j}) - f_{c\neq C}\| + \alpha,\ 0
\right)
$$

논문의 설명에 따르면 이 loss는 각 feature가 자기 category center에 대해 다른 category center보다 최소 $\alpha$만큼 더 가깝도록 강제한다. 즉, **intra-class compactness**와 **inter-class separability**를 동시에 강화하는 효과를 가진다. 특히 hard sample을 더 잘 다루게 해 generalization을 높인다고 해석할 수 있다.

중요한 점은 이 loss를 **source domain에만 적용**한다는 것이다. target의 pseudo-label은 hard sample일수록 신뢰하기 어렵기 때문에 CTL에 넣지 않는다.

### 3.4 Target Domain Consistency Regularization (TCR)

target domain에는 GT가 없으므로 source처럼 직접 center supervision을 줄 수 없다. 여기서 저자들은 self-supervised consistency regularization을 사용한다.

target 이미지 $n_k$에 대해, 이전 단계의 안정적인 모델 $T_{i-1}$를 사용해 pseudo-label $\hat{y}_j^k$를 만든다.

$$
\hat{y}_j^k = \arg\max(T_{i-1}(n_k)|_j)
$$

그리고 augmentation된 이미지 $n'_k$에 현재 학습 중인 모델 $T_i$를 적용해 예측 $p_j^{\prime k}$를 얻는다.

$$
p_j^{\prime k} = T_i(n'_k)|_j
$$

이제 confidence가 충분히 높은 픽셀에서만 pseudo-label과 augmented prediction이 일치하도록 cross-entropy를 건다.

$$
L_{cst} =
\sum_j
\mathbf{1}(\max(T_{i-1}(n_k)|_j)\ge t_c)\,
CELoss(\mathbf{1}[c=\hat{y}_j^k], p_j^{\prime k})
$$

여기서 $t_c$는 class별 confidence threshold다. 논문은 전체 threshold $P_h$와 category-specific percentile threshold를 조합해

$$
t_c = \min(P_h, P_{s,c})
$$

로 정의한다. 즉, class마다 pseudo-label 신뢰도가 다를 수 있다는 점을 반영한다.

또 하나 중요한 설계는 pseudo-label 생성에 현재 모델 $T_i$가 아니라 **이전 단계 모델 $T_{i-1}$** 를 쓴다는 점이다. 저자들은 학습 중인 모델의 pseudo-label은 흔들리기 쉽고, 이는 학습을 망칠 수 있다고 명시한다.

## 4. 실험 및 결과

### 4.1 데이터셋과 설정

논문은 다음 UDA semantic segmentation benchmark를 사용한다.

- **GTA5 → Cityscapes**
- **SYNTHIA → Cityscapes**

GTA5는 Cityscapes와 19개 공통 class를 공유하고, SYNTHIA는 16개 공통 class를 공유한다. SYNTHIA는 일부 기존 연구가 13-class subset만 평가했지만, 이 논문은 전체 set으로 학습하고 16-class와 13-class 설정 모두 보고한다.

backbone은 **DeepLabV3+ with ResNet-101** 이다. 구현은 PyTorch로 했고, 4개의 NVIDIA GeForce 2080Ti GPU를 사용했다고 적혀 있다. 학습 반복은 총 140k iteration이며, iterative self-supervision은 $K=6$, 각 단계 fine-tuning은 $U=20k$이다.

주요 하이퍼파라미터는 다음과 같다.

- pseudo-label threshold 상한 $P_h = 0.9$
- percentile 기준 $p = 10$
- triplet margin $\alpha = 0.2$
- gamma regularization 계수 $\beta = 0.01$
- augmentation $\tau(\cdot)$: standard color jittering

평가 지표는 표에서 보아 **mIoU**이다.

### 4.2 GTA5 → Cityscapes 결과

이 논문은 GTA5 → Cityscapes에서 **56.1% mIoU**를 달성했다고 보고한다. 표에 따르면 이전 최고 성능이던 CAG의 50.2%보다 **5.9%p** 높다.

특히 road, sidewalk, building, light, sky, car, person, train, motor, bike 같은 중요한 class에서 강한 성능을 보였다고 저자들은 강조한다. 이 중 road, sidewalk, motor, bike처럼 외형이나 local appearance가 비슷해 헷갈리기 쉬운 class에서 향상이 컸다는 해석을 제시한다. 이는 CTL이 confusing sample을 더 잘 분리했기 때문이라는 설명이다.

또 building과 sky처럼 intra-class variance가 큰 class의 향상은 TCR의 효과로 설명한다. 같은 class 내부 변화가 큰 경우에도 augmentation 전후 예측 일관성을 학습하면서 target domain 구조를 더 잘 잡게 된다는 주장이다.

### 4.3 SYNTHIA → Cityscapes 결과

SYNTHIA → Cityscapes에서는:

- **16-class mIoU: 48.2**
- **13-class mIoU*: 55.5**

를 보고한다. 논문은 SYNTHIA가 GTA5보다 photometric 차이 외에도 perspective와 layout 차이가 더 커서 더 어려운 설정이라고 설명한다. 그럼에도 기존 방법들보다 전반적으로 좋은 성능을 보였다고 주장한다.

같은 segmentation backbone을 쓰는 CAG와 비교하면, 이 논문의 방법은 SYNTHIA → Cityscapes에서 **3.7%p** 개선을 보였다고 적는다.

### 4.4 Ablation Study

가장 중요한 ablation은 Table 3이다.

- Source only: 37.6
- w/o GPA: 47.5
- w/o CTL and TCR: 47.3
- w/o CTL: 53.2
- w/o TCR: 53.1
- All: 56.1

이 결과는 세 모듈 모두 유효하며, 특히 GPA가 coarse alignment 단계로서 중요하다는 논문의 주장을 뒷받침한다. GPA가 빠지면 초기 pseudo-label 품질이 나빠지고, 그 위에 쌓이는 category-level regularization도 약해진다는 논리다.

CTL과 TCR 각각도 약 3% 안팎의 성능 향상을 제공한다. TCR은 구조가 단순함에도 효과가 컸는데, 저자들은 target domain의 유효 학습 샘플 수를 사실상 늘려주는 역할 때문이라고 설명한다.

### 4.5 GPA 자체 비교

Table 4에서는 coarse alignment 방법 자체를 비교한다.

- Frequency Align [33]: 52.0
- BDL-GAN [14]: 54.5
- Photometric Align. (제안 방법): 56.1

또 Lab-space 내부 세부 scheme 비교에서는:

- Lab Gamma Correction: 44.5
- Lab Histogram Match: 43.3
- Hybrid: 47.3

로 나타난다. 여기서 Hybrid는 coarse alignment stage만 따로 본 결과로 읽힌다. 논문은 모든 채널에 gamma correction만 적용하면 alignment가 충분하지 않고, 모든 채널에 histogram matching을 적용하면 artifact가 생긴다고 해석한다. 따라서 **L에는 gamma, a/b에는 histogram matching** 이 가장 적절하다고 주장한다.

### 4.6 Pseudo-label을 CTL에 넣지 않은 이유

논문은 CTL에 target pseudo-label을 쓰지 않는 선택도 검증한다. 결과는 다음과 같다.

- Triplet loss with pseudo-labels: 53.3
- Triplet loss w/o pseudo-labels: 56.1

즉, pseudo-label을 넣으면 오히려 성능이 떨어졌다. 저자들의 해석은 설득력 있다. CTL은 hard sample을 다루는 loss인데, target domain의 hard sample은 pseudo-label 신뢰도가 낮기 때문에 잘못된 supervision이 들어가기 쉽다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 domain shift를 다층적으로 해석했다는 점이다. 단순히 style transfer만 하거나, 반대로 feature alignment만 하는 대신, image-level과 category-level 문제를 분리해서 각각 적절한 도구로 해결한다. 이 설계는 논리적으로도 자연스럽고, ablation 결과도 이를 잘 뒷받침한다.

또 하나의 강점은 GPA의 실용성이다. GAN 기반 방법은 추가 학습 비용과 불안정성이 있고, Fourier 기반 방법은 artifact 문제가 있을 수 있는데, 이 논문은 매우 단순한 photometric transformation만으로도 강한 baseline을 만든다. 실제 응용 관점에서 계산 비용과 구현 복잡도를 낮춘다는 장점이 있다.

CTL도 의미 있는 기여다. 기존 category-anchor 방식이 center를 기준으로 맞추는 데 집중했다면, 이 논문은 class center 간 margin 자체를 키우는 방향으로 설계했다. segmentation에서 class 간 혼동이 잦은 환경을 고려하면 타당한 아이디어다.

한편 한계도 있다. 첫째, GPA는 photometric 차이에는 강하지만, geometry나 scene layout 차이 자체를 직접 해결하지는 못한다. 저자들도 SYNTHIA에서는 perspective와 layout shift가 더 크다고 언급한다. 즉, 이 방법은 photometric gap이 큰 상황에서는 강력하지만, 구조적 domain gap까지 충분히 다룬다고 보기는 어렵다.

둘째, iterative self-training에 여전히 pseudo-label 품질이 중요하다. 논문은 thresholding과 이전 단계 모델 사용으로 이를 완화하지만, 초기 모델이 매우 불안정한 경우 얼마나 견고한지는 본문만으로는 충분히 알 수 없다.

셋째, CTL의 계산은 class center 품질에 의존한다. center는 source domain feature 평균으로 계산되므로, class 내부 분포가 다봉적(multimodal)일 경우 단일 center가 충분한 표현인지 의문이 남는다. 논문은 이 점을 직접 분석하지 않는다.

넷째, qualitative 비교와 성능 표는 충분히 제시하지만, 추가적인 계산 비용, wall-clock time, alignment preprocessing overhead를 체계적으로 비교한 내용은 본문에 명확히 없다. “GAN보다 가볍다”는 주장은 타당해 보이지만, 정확한 효율성 수치 비교는 제공되지 않는다.

## 6. 결론

이 논문은 UDA semantic segmentation에서 domain shift를 **coarse image-level shift**와 **fine category-level shift**로 나누고, 이를 통합적으로 해결하는 coarse-to-fine framework를 제안했다. 구체적으로는 training-free에 가까운 **Global Photometric Alignment**, source domain용 **Category-oriented Triplet Loss**, target domain용 **Target-domain Consistency Regularization** 을 결합해 강한 성능 향상을 얻었다.

실험적으로도 GTA5 → Cityscapes와 SYNTHIA → Cityscapes에서 당시 state-of-the-art를 넘는 결과를 보고했으며, 각 구성 요소의 필요성을 ablation으로 보여준다. 특히 image alignment와 category-structure regularization을 동시에 다루어야 한다는 메시지가 논문의 핵심 기여라고 볼 수 있다.

실제 적용 측면에서는, annotation이 어려운 real-world segmentation 문제에서 synthetic-to-real adaptation을 더 실용적으로 만들 수 있는 방향을 제시한다는 점에서 의미가 크다. 향후 연구로는 단일 category center를 넘어선 더 복잡한 class distribution modeling, geometry/layout shift까지 포함하는 adaptation, 그리고 더 강건한 pseudo-label selection과 결합하는 방향으로 확장될 가능성이 크다.
