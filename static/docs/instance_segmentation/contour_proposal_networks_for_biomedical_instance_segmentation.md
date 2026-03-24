# Contour Proposal Networks for Biomedical Instance Segmentation

* **저자**: Eric Upschulte, Stefan Harmeling, Katrin Amunts, Timo Dickscheid
* **발표연도**: 2021
* **arXiv**: <https://arxiv.org/abs/2104.03393>

## 1. 논문 개요

이 논문은 biomedical instance segmentation, 특히 세포나 세포핵처럼 서로 맞닿거나 부분적으로 겹치는 객체를 분리하는 문제를 다룬다. 저자들은 이를 위해 **Contour Proposal Network (CPN)** 라는 단일 단계(single-stage) 프레임워크를 제안한다. 핵심 목표는 단순히 픽셀 단위로 foreground를 칠하는 것이 아니라, 각 객체를 **닫힌 closed contour**로 직접 표현하고 검출하는 것이다. 이를 통해 객체의 존재 여부와 윤곽선을 동시에 예측하고, 겹침이 있는 상황에서도 더 그럴듯한 개별 객체 형상을 복원하려고 한다.

논문이 문제로 보는 지점은 기존 biomedical segmentation 방법, 특히 U-Net류의 dense pixel classifier가 접촉하거나 겹친 객체를 분리할 때 쉽게 오류를 낸다는 점이다. 픽셀 몇 개만 잘못 분류되어도 두 객체가 하나로 합쳐질 수 있고, 이는 detection accuracy를 크게 떨어뜨린다. 또한 일부 방법은 객체를 한 픽셀당 하나의 instance label로만 표현하므로, 부분적으로 겹쳐진 객체를 완전한 형태로 복원하기 어렵다. 이는 세포 morphology 분석처럼 shape 자체가 중요한 downstream task에서 치명적일 수 있다.

저자들은 이런 한계를 넘기 위해 instance segmentation을 dense pixel labeling 문제가 아니라 **sparse detection + explicit shape regression** 문제로 재정의한다. 즉, 어떤 위치에 객체가 존재하는지 분류하고, 그 위치에서 객체 전체의 contour를 하나의 고정 길이 벡터로 회귀한다. 이 벡터는 Fourier Descriptor 기반이므로 해석 가능하고, 닫힌 윤곽선이라는 구조적 제약을 자연스럽게 포함한다. 이후 추가적인 local refinement를 통해 픽셀 수준 정밀도를 높이고, 마지막에 NMS로 중복 검출을 제거한다.

이 문제의 중요성은 두 가지 측면에서 크다. 첫째, biomedical imaging에서는 touching/overlapping objects가 매우 흔하며, 단순한 semantic segmentation으로는 개체 단위 분석이 충분하지 않다. 둘째, 실제 실험실 환경에서는 염색, 샘플 품질, 스캐닝 조건 등의 변동이 항상 존재하므로, 단순히 한 데이터셋에서만 잘 작동하는 모델이 아니라 **generalization**이 좋은 모델이 필요하다. 논문은 CPN이 이 두 요구, 즉 shape-aware instance segmentation과 domain generalization을 동시에 만족시킬 가능성을 보여주려 한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 객체 마스크를 픽셀 공간에서 직접 예측하는 대신, 각 객체를 **주파수 영역(frequency domain)의 contour descriptor**로 표현하고 이를 회귀하는 것이다. 저자들은 Elliptical Fourier Descriptor에서 영감을 받아, contour를 Fourier sine/cosine 계수로 표현한다. 이 표현은 고정 길이 벡터이면서 닫힌 contour를 항상 생성한다. 따라서 네트워크는 각 객체를 “하나의 좌표에 묶인 shape code”로 응축해서 예측하게 된다.

이 설계의 중요한 직관은 다음과 같다. 기존 pixel classifier는 객체를 수많은 픽셀에 분산된 방식으로 표현한다. 반면 CPN은 객체 하나의 경계를 하나의 위치와 하나의 descriptor로 표현하므로, 네트워크가 객체 전체와 그 부분들 사이의 공간적 관계를 더 압축적이고 구조적으로 이해하도록 강제한다. 논문은 이것이 더 robust한 representation과 better generalization으로 이어진다고 주장한다.

기존 접근과의 차별점도 명확하다.
우선 U-Net 같은 dense segmentation은 픽셀 단위 분류 후 connected component labeling이나 boundary class 같은 후처리를 통해 instance를 분리한다. 그러나 crowded scene에서는 매우 취약하다.
Mask R-CNN은 bounding box를 먼저 회귀하고, box 내부에서 mask를 생성한다. 이는 강력하지만 shape를 직접 모델링하지 않고 bounding box에 크게 의존한다.
StarDist나 PolarMask 같은 radial representation은 중심점에서 여러 ray 방향으로 contour를 표현하지만, 기본적으로 star-convex 제약이 있으며 비볼록(non-convex) 형태에 불리하다.
이 논문의 CPN은 이러한 방식들과 달리, **closed contour를 직접 주파수 계수로 모델링**하므로 비볼록 형태도 비교적 자연스럽게 다룰 수 있고, 샘플링 해상도 문제를 contour order $N$으로 조절할 수 있다.

또 하나의 중요한 아이디어는 **local refinement**다. Fourier descriptor는 low-frequency 중심의 매끄러운 contour를 잘 표현하지만, 매우 세밀한 픽셀 수준 경계까지는 부족할 수 있다. 이를 보완하기 위해 논문은 high-resolution feature에서 2채널 residual field를 예측하고, contour 좌표를 반복적으로 미세 조정한다. 즉, 전역 형태는 Fourier contour로 잡고, 세부 정렬은 residual field로 맞춘다. 이 조합이 CPN의 핵심 설계다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

CPN은 다섯 개의 기본 구성요소로 설명된다.

첫 단계에서 backbone CNN이 입력 이미지로부터 두 종류의 feature map을 만든다. 하나는 고해상도 $P_1 \in \mathbb{R}^{h_1 \times w_1 \times c_1}$, 다른 하나는 저해상도 $P_2 \in \mathbb{R}^{h_2 \times w_2 \times c_2}$ 이다.
저해상도 $P_2$는 detection과 contour regression에 사용되고, 고해상도 $P_1$는 refinement를 위한 residual field 생성에 사용된다.

$P_2$ 위에서 classification head는 각 위치가 객체를 대표하는지 점수화한다. 동시에 regression heads는 그 위치에 대응되는 contour representation을 출력한다. 이 contour representation은 결국 픽셀 공간의 contour coordinates로 변환된다. 객체가 있다고 분류된 위치들의 contour만 sparse proposal list로 추출한 뒤, local refinement를 거쳐 contour를 미세 보정하고, 마지막에 NMS를 적용해 중복 proposal을 제거한다.

이 구조는 single-stage instance segmentation처럼 동작한다. 별도의 ROI cropping이나 proposal-to-mask 이단계 구조가 아니라, 한 번의 전방 계산 안에서 detection과 contour generation이 함께 이루어진다.

### 3.2 Detection head

Detection은 각 위치별 binary classification이다. 논문에서는 multiclass가 아니라 객체 존재 여부만 판단하는 binary case에 집중한다. 즉, 어떤 출력 그리드의 한 픽셀이 “하나의 객체를 대표하는 위치(anchor-like location)”인지 아닌지를 판별한다.

이 설계는 dense detector와 비슷하지만, 여기서 예측되는 것은 bounding box가 아니라 contour descriptor라는 점이 다르다. 출력 해상도 $h_2 \times w_2$는 이론적으로 검출 가능한 최대 객체 수를 제한하는 grid 역할도 한다.

### 3.3 Contour representation: Fourier Descriptor 기반 윤곽선 표현

논문의 가장 중요한 부분이다. 객체 contour는 $N$차 Fourier series로 표현된다. contour 상의 점 $(x_N(t), y_N(t))$는 다음과 같이 정의된다.

$$
x_N(t)=a_0+\sum_{n=1}^{N}\left(a_n \sin(2n\pi t/T)+b_n \cos(2n\pi t/T)\right)
$$

$$
y_N(t)=c_0+\sum_{n=1}^{N}\left(c_n \sin(2n\pi t/T)+d_n \cos(2n\pi t/T)\right)
$$

논문에서는 $T=1$로 둔다. 따라서 $t \in [0,1]$를 따라 contour를 샘플링하면, 닫힌 2차원 곡선을 얻는다.

여기서 계수들의 의미는 다음과 같다.

$a_0, c_0$는 contour의 전체적인 위치, 즉 spatial offset을 담당한다.
$a_n, b_n, c_n, d_n$는 contour shape를 규정하는 주파수 계수들이다.
전체 parameter vector는 $[\mathbf{a}, \mathbf{b}, \mathbf{c}, \mathbf{d}] \in \mathbb{R}^{4N+2}$ 크기를 갖는다.

즉, order가 $N$이면 descriptor 차원은 $4N+2$다.
$N$이 작으면 매끄럽고 단순한 contour만 표현할 수 있고, $N$이 커질수록 더 높은 주파수 성분이 들어가 세부 형태를 더 잘 복원할 수 있다. 논문은 작은 $N$만으로도 꽤 복잡하고 비볼록한 세포 형태를 근사할 수 있다고 보여준다.

이 표현의 장점은 몇 가지다.
첫째, 항상 닫힌 contour를 생성한다.
둘째, 표현 차원이 고정되어 CNN이 직접 회귀하기 쉽다.
셋째, differentiable이므로 end-to-end 학습이 가능하다.
넷째, radial representation과 달리 star-convex에 제한되지 않는다.

논문은 contour의 **shape**와 **location**을 분리해서 회귀한다는 점도 강조한다. 즉, $(a_0, c_0)$와 나머지 shape coefficients를 서로 다른 regression head로 예측한다. 저자들의 설명에 따르면 이는 contour shape 표현의 translational invariance와 offset regression의 equivariance를 유지하려는 의도다. 쉽게 말해, 물체 모양 자체와 물체가 이미지 어디에 놓였는지를 분리해 학습시키는 것이다.

### 3.4 Pixel-space contour 생성

네트워크가 예측한 Fourier coefficients는 바로 contour coordinates로 바뀐다. $t_1,\dots,t_S$를 샘플링해서 $S$개의 contour points를 만든다. 이때 $S$는 contour supervision과 rasterization 정밀도에 영향을 주는 하이퍼파라미터다. 실험에서는 $S=64$를 사용했다.

이 변환은 fully differentiable이기 때문에, 최종적으로 contour 좌표 자체에 대한 손실을 걸 수 있다. 즉, descriptor space에서만 학습하는 것이 아니라, 실제 contour point들이 ground truth contour와 가깝도록 직접 학습한다.

### 3.5 Local refinement

Fourier contour는 본질적으로 매끄러운 shape prior를 제공하지만, 실제 이미지의 국소 경계와 정확히 맞지 않을 수 있다. 이를 보완하기 위해 CPN은 local refinement를 도입한다.

고해상도 feature $P_1$에서 추가 regression head가 2채널 refinement tensor $\mathbf{v}$를 생성한다. 이 $\mathbf{v}$는 픽셀 공간의 residual vector field로 볼 수 있다. contour 상의 각 점 $(x(t), y(t))$에 대해, 해당 위치를 반올림한 픽셀 좌표에서 residual을 읽어와 좌표를 조금 이동시킨다.

알고리즘은 다음과 같이 이해할 수 있다.

1. contour 점의 현재 좌표 $(x, y)$를 반올림해서 nearest pixel을 찾는다.
2. 그 위치의 residual vector $\mathbf{v}_{\lfloor x \rceil,\lfloor y \rceil}$를 읽는다.
3. $\sigma \tanh(\cdot)$를 적용해 최대 이동량을 제한한다.
4. 좌표를 갱신한다.
5. 이를 $r$번 반복한다.

논문 속 알고리즘을 수식 형태로 쓰면 대략 다음과 같다.

$$
\begin{bmatrix}
x \
y
\end{bmatrix}
\leftarrow
\begin{bmatrix}
\lfloor x \rceil \
\lfloor y \rceil
\end{bmatrix}
+
\sigma \tanh\left(\mathbf{v}_{\lfloor x \rceil,\lfloor y \rceil}\right)
$$

여기서 $\sigma$는 maximum correction margin이고, $r$은 refinement iteration 수다. 논문 실험에서는 refinement를 4회 적용한 R4 variant와 0회 적용한 R0 variant를 비교한다.

중요한 설계 포인트는, 반올림된 좌표를 사용한다는 점이다. 논문은 이것이 refinement head가 proposal head를 직접 교란하지 않도록 해 주며, refinement의 출발점을 일관되게 만든다고 설명한다. 또 refinement loss는 refined contour 좌표와 ground truth contour 좌표 사이의 거리로 간접적으로 학습된다. 즉, refinement tensor에 대해 별도의 직접 supervision을 주는 것이 아니라, 최종 contour 오차를 줄이는 방향으로 학습된다.

이 모듈은 두 가지 역할을 한다.
하나는 Fourier order $N$이 낮아서 표현하지 못한 고주파 세부 경계를 보완하는 것,
다른 하나는 contour localization error를 줄여 높은 IoU 기준에서 성능을 개선하는 것이다.

### 3.6 Non-Maximum Suppression

CPN은 dense proposal 방식이기 때문에 여러 인접 픽셀이 같은 객체를 나타낼 수 있다. 따라서 최종 inference에서는 bounding-box NMS를 쓴다. 각 contour proposal의 bounding box는 contour 샘플 점들의 최소/최대 좌표로 계산된다.

$$
b=[\min x(\mathbf{t}), \min y(\mathbf{t}), \max x(\mathbf{t}), \max y(\mathbf{t})]
$$

그 다음 detection score가 높은 proposal은 유지하고, bounding box IoU가 threshold를 넘는 proposal은 제거한다.
흥미로운 점은 contour 기반 모델이지만 NMS는 contour IoU가 아니라 bounding-box IoU에 기반한다는 것이다. 논문은 효율성을 위해 이 선택을 한 것으로 보인다.

### 3.7 손실 함수

CPN의 손실은 detection, contour coordinate, refinement, representation regularization을 결합한다.

#### Detection loss

각 위치의 객체 존재 여부에 대해 standard BCE를 사용한다. 이를 $\mathcal{L}_{inst}$라고 둔다.

#### Coordinate loss

개별 contour point의 좌표 오차는 $L_1$ 거리 평균으로 정의된다.

$$
\mathcal{L}_{coord}(x,y,\hat{x},\hat{y}) = \frac{1}{2}\left(|x-\hat{x}|_1 + |y-\hat{y}|_1\right)
$$

여기서 논문의 표기상 $| \cdot |_1$는 절대값 기반의 $L_1$ 차이를 의미한다.

#### Contour loss

샘플된 $S$개의 contour points 전체에 대해 평균을 취한다.

$$
\mathcal{L}_{contour} = \frac{1}{S}\sum_{s=1}^{S} \mathcal{L}_{coord}\left(x(t_s), y(t_s), \hat{x}(t_s), \hat{y}(t_s)\right)
$$

즉, descriptor 자체가 아니라 그것으로부터 생성된 contour points가 ground truth contour points에 가깝도록 학습한다.

#### Refinement loss

refined contour points에 대해서도 같은 방식의 coordinate loss를 적용한다.

$$
\mathcal{L}_{refine} = \frac{1}{S}\sum_{s=1}^{S} \mathcal{L}_{coord} \left( x(t_s), y(t_s), Refine(\hat{x}(t_s), \hat{y}(t_s)) \right)
$$

이 손실 덕분에 refinement tensor는 contour를 더 정확한 경계로 밀어주는 방향으로 학습된다.

#### Representation loss

논문은 추가로 descriptor coefficient 자체를 직접 감독한다. 이는 frequency domain에서의 regularization 역할을 한다.

본문의 표기가 다소 깨져 있지만, 요지는 예측 계수와 ground truth 계수 간의 $L_1$ 차이를 coefficient별 가중치 $\beta_n$로 조절한다는 것이다. 저자들은 낮은 주파수 계수에 더 큰 가중치를 두는 식으로 coarse outline를 더 강조할 수 있다고 설명한다.

즉, representation loss는 “실제 contour가 맞는가”뿐 아니라 “그 contour를 생성하는 Fourier parameter도 적절한가”를 동시에 학습시키는 역할을 한다.

#### 전체 손실

최종 per-pixel loss는 다음과 같다.

$$
\mathcal{L}_{CPN} = \mathcal{L}*{inst}(o) + o\left(\mathcal{L}*{contour} + \mathcal{L}*{refine} + \lambda \mathcal{L}*{repr}\right)
$$

여기서 $o=1$이면 해당 픽셀이 객체를 대표하는 positive 위치이고, $o=0$이면 background 위치다.
즉, contour 관련 손실은 object-positive pixel에서만 적용된다. 이는 매우 자연스럽다. background 위치에서는 contour를 회귀할 이유가 없기 때문이다.

## 4. 실험 및 결과

### 4.1 데이터셋과 과제

논문은 세 개의 주 데이터셋에서 instance segmentation 성능을 비교하고, 네 번째 데이터셋에서 cross-dataset generalization을 평가한다.

첫 번째는 **NCB (Neuronal Cell Bodies)** 로, 82개의 grayscale patch와 약 29,000개 cell body annotation으로 구성된다. 세포 형상, 밝기, 겹침, 노이즈, contrast variation, histological artifact가 크고 challenging한 데이터다. 중요한 점은 가려진 경우에도 cell body가 연속적이고 완전하게 annotation되었다는 것이다. 즉, 실제 보이지 않는 부분까지 plausible morphology를 학습하도록 설계된 데이터다.

두 번째는 **BBBC039** 로, U2OS cell nuclei chemical screen 데이터셋이다. 200개의 grayscale image와 약 23,000개의 nuclei annotation을 포함한다.

세 번째는 **SYNTH** 로, 4,129개의 synthetic grayscale image와 약 1,305,000개의 객체를 포함한다. 원, 타원, 삼각형뿐 아니라 복잡한 비볼록 shape도 포함하며, 겹침과 occlusion도 존재한다.

네 번째는 **BBBC041** 이며, malaria infected blood smear cell 이미지 1,364장과 약 80,000개의 bounding box annotation으로 구성된다. 이는 cross-dataset generalization 평가용이다.

### 4.2 비교 대상

비교 baseline은 두 가지다.

첫째, **U-Net**. 논문은 batch normalization을 추가한 22-layer U-Net 변형을 사용한다. 픽셀을 cell, background, boundary의 3개 클래스로 분류한다. 이후 instance segmentation은 픽셀 분류 결과를 이용해 얻는다.

둘째, **Mask R-CNN**. torchvision 기반 구현을 사용한다. bounding box proposal 후 mask를 생성하는 대표적인 instance segmentation baseline이다.

CPN은 backbone에 따라 네 가지 variant를 사용한다.

* CPNR4-R50-FPN: ResNet-50-FPN backbone + refinement 4회
* CPNR0-R50-FPN: 같은 backbone, refinement 없음
* CPNR4-U22: U-Net 22-layer backbone + refinement 4회
* CPNR0-U22: 같은 backbone, refinement 없음

즉, backbone이 같을 때도 CPN이라는 formulation 자체가 주는 이득이 있는지 비교할 수 있게 설계되어 있다.

### 4.3 평가 지표

평가는 IoU threshold $\tau$에 따른 $F1_{\tau}$를 사용한다.
$\tau=0.5$는 대략적인 coarse detection 성능을,
$\tau=0.9$는 매우 정밀한 contour quality를 본다.

또한 평균 점수로

$$
F1^{avg} = \frac{1}{9}\sum_{\tau \in {0.5, 0.55, 0.6, \dots, 0.9}} F1_{\tau}
$$

를 사용한다.

이 설계는 논문 주장과 잘 맞는다. CPN은 단순 검출만이 아니라 contour quality까지 중요하므로, 낮은 IoU와 높은 IoU를 모두 보는 것이 적절하다.

### 4.4 정량 결과: 주 데이터셋 성능

논문의 핵심 결과는 **local refinement를 포함한 CPN이 세 데이터셋 모두에서 최고 성능**을 보였다는 점이다.

#### NCB

가장 어려운 Neuronal Cell Bodies 데이터셋에서는 CPNR4-U22가 가장 좋다.
$F1^{avg}=0.55$로, U-Net의 $0.47$, Mask R-CNN의 $0.34$보다 높다.
특히 $F1_{0.7}=0.62$, $F1_{0.8}=0.40$으로 고정밀 구간에서도 앞선다.

이 결과는 crowded, noisy, overlapping object 상황에서 CPN의 explicit contour modeling이 특히 유리함을 보여준다.

#### BBBC039

BBBC039에서는 전체적으로 모든 모델 성능이 높지만, 그래도 CPNR4-U22가 가장 높다.
$F1^{avg}=0.91$이며, U-Net은 $0.89$, Mask R-CNN은 $0.86$이다.
특히 $F1_{0.9}=0.76$으로, 고정밀 contour 품질에서 우수하다.

#### SYNTH

Synthetic Shapes에서도 CPNR4-U22가 $F1^{avg}=0.90$으로 가장 높고, U-Net은 $0.87$, Mask R-CNN은 $0.85$이다.
여기서도 $F1_{0.9}=0.64$로 높은 threshold에서 강점을 보인다.

### 4.5 Local refinement의 효과

R4와 R0를 비교하면 refinement가 일관되게 성능을 올린다. 특히 높은 IoU threshold에서 차이가 커진다.

예를 들어 NCB에서 CPNR4-U22와 CPNR0-U22를 비교하면
$F1^{avg}$는 $0.55$ 대 $0.51$이고,
$F1_{0.8}$은 $0.40$ 대 $0.33$이다.

SYNTH에서도 CPNR4-U22가 CPNR0-U22보다
$F1_{0.9}$에서 $0.64$ 대 $0.51$로 크게 높다.

이는 refinement가 coarse contour를 실제 경계에 더 정밀하게 정렬시키며, 특히 세부 shape fidelity에 기여한다는 논문의 주장과 일치한다.

### 4.6 Cross-dataset generalization

논문은 BBBC039에서 학습한 모델을 아무 재학습 없이 BBBC041에 적용한다.
이는 단순 성능이 아니라 domain shift에 대한 robustness를 보기 위한 실험이다.

결과적으로 CPNR4-U22가 가장 높은 평균 성능을 보인다.
$F1^{avg}=0.54$로,
기본 U-Net의 $0.45$, Mask R-CNN의 $0.49$보다 높다.

특히 $F1_{0.5}=0.83$으로 coarse detection에서는 상당한 강점을 보인다. 반면 $F1_{0.9}=0.02$로 매우 정밀한 shape matching은 쉽지 않다. 이는 training domain과 test domain의 shape/annotation 차이, 그리고 BBBC041가 bounding box annotation 기반이라는 평가 설정의 영향도 있을 수 있다. 다만 이 부분의 해석은 논문에 명시된 수준을 넘어서면 추측이 될 수 있으므로 조심해야 한다.

논문은 qualitative하게 CPN이 false positive를 줄이기 위해 더 conservative하게 동작한다고 설명한다. U-Net은 noise를 객체로 잘못 감지하는 경우가 더 많고, Mask R-CNN은 contour precision이 떨어진다고 분석한다.

### 4.7 CPN backbone 재사용 실험

흥미로운 추가 실험으로, 저자들은 CPNR4-U22의 backbone으로 학습된 encoder/decoder를 다시 U-Net의 backbone으로 재사용해 본다. encoder는 고정하고 새로운 final prediction layer를 붙여 U-Net으로 재학습한다.

이 경우 cross-dataset generalization 성능이 기본 U-Net보다 크게 향상된다. 표에서 pretrained U-Net은 $F1^{avg}=0.52$로 기본 U-Net의 $0.45$보다 낫다. 일부 높은 IoU 구간에서는 오히려 최고 성능을 보이기도 한다. 다만 원래 BBBC039 test set 성능은 $0.89$에서 $0.87$로 조금 떨어졌다고 보고한다.

이 실험은 단순히 head 설계만이 아니라, **CPN objective 자체가 backbone feature space를 더 generalizable하게 만든다**는 저자들의 주장을 뒷받침한다.

### 4.8 추론 속도

추론 속도는 BBBC039 test set의 $520 \times 696$ 이미지에서 FPS로 측정했다. NVIDIA A100 위에서 PyTorch 모델로 실행했고, float32와 automatic mixed precision(amp)를 비교했다. 초기 warm-up과 post-processing은 제외했다.

주요 결과는 다음과 같다.

* CPNR0-R50-FPN: 30.19 FPS
* CPNR4-R50-FPN: 29.86 FPS
* U-Net: 23.42 FPS
* Mask R-CNN-R50-FPN: 13.74 FPS
* CPNR4-U22 (P2 stride 2): amp에서 42.20 FPS
* U-Net: amp에서 77.71 FPS

float32 기준으로는 CPNR4-R50-FPN이 29.86 FPS로 U-Net과 Mask R-CNN보다 빠르다. 즉, contour proposal과 refinement를 포함해도 충분히 실시간에 가까운 성능을 낸다.
amp 기준에서는 U-Net이 가장 빠르지만, CPNR4-U22 stride-2 variant도 42.2 FPS로 상당히 빠르다.

또한 refinement 4회가 R50-FPN 기반 CPN에서 0.33 FPS 정도만 감소시켰다고 보고한다. 즉, 정밀도 향상 대비 속도 손실이 작다는 것이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 instance segmentation을 **explicit contour representation** 문제로 재정의했다는 점이다. 객체의 경계를 주파수 계수로 직접 회귀하므로, 네트워크가 shape 자체를 명시적으로 다루게 된다. 이는 단순 픽셀 분류보다 해석 가능성이 높고, 결과로 생성되는 contour도 downstream morphology analysis에 바로 활용하기 좋다.

둘째, Fourier descriptor 선택이 매우 설득력 있다. radial representation처럼 star-convex 가정에 묶이지 않고, 고정 길이 벡터이면서 closed contour를 보장한다. 또한 contour complexity를 sampling point 개수 대신 order $N$으로 제어하는 방식은 개념적으로도 깔끔하다.

셋째, backbone에 크게 종속되지 않는다는 점도 장점이다. 논문은 U-Net backbone과 ResNet-FPN backbone 모두에서 CPN을 구성했고, 일관된 개선을 보였다. 이는 특정 아키텍처 트릭이 아니라 formulation 자체의 힘이 있음을 시사한다.

넷째, refinement 모듈이 단순하면서도 효과적이다. coarse shape prior와 local residual correction을 조합해, 높은 IoU 구간에서 실제로 성능 이득을 낸다. 그리고 속도 오버헤드가 크지 않다.

다섯째, cross-dataset generalization 실험이 포함되어 있다는 점도 강점이다. biomedical imaging에서는 domain shift가 매우 중요한데, 논문은 단순 test-set 성능을 넘어서 이 부분을 직접 확인한다.

반면 한계도 분명하다.

가장 먼저, 이 접근의 핵심 가정은 **객체가 닫힌 contour로 잘 표현될 수 있어야 한다**는 점이다. 논문도 마지막에 이 점을 사실상 전제한다. 따라서 열린 곡선 구조, 매우 복잡한 위상(topology), 끊어진 구조, 구멍이 있는 구조 등에는 바로 적용하기 어려울 수 있다. 적어도 제공된 텍스트에서는 이런 경우에 대한 확장 논의가 없다.

둘째, NMS가 contour IoU가 아니라 bounding box IoU를 사용한다는 점은 효율적이지만, shape-sensitive detector라는 관점에서는 다소 거친 선택일 수 있다. 서로 다른 contour인데 box가 유사한 경우의 처리에 한계가 있을 가능성이 있다. 다만 논문은 이 부분에 대한 별도 ablation을 제공하지 않는다.

셋째, multiclass setting에 대한 검증이 없다. 본 논문은 binary detection에 초점을 두고 있으며, 서로 다른 object category를 동시에 다루는 일반 instance segmentation으로 확장할 때 어떤 변화가 필요한지는 제공된 텍스트만으로는 알 수 없다.

넷째, contour order $N$, sample size $S$, refinement iteration 수, correction margin $\sigma$ 등의 하이퍼파라미터가 성능과 속도에 영향을 줄 것으로 보이지만, 이들에 대한 체계적인 ablation은 제공된 본문에서 충분히 보이지 않는다. 예를 들어 $N$을 얼마나 키우면 어떤 형태까지 잘 표현되는지, $S=64$가 왜 적절한지에 대한 자세한 비교는 제공되지 않았다.

다섯째, representation loss의 효과도 개념적으로는 타당하지만, 이 항이 실제로 얼마나 중요한지, coefficient weighting $\beta_n$ 설계가 결과에 어떤 영향을 주는지에 대한 독립적인 분석은 본문에 충분히 나오지 않는다.

여섯째, cross-dataset generalization에서 높은 IoU 구간 성능은 여전히 낮다. 이는 domain shift 문제의 어려움을 보여주며, CPN이 baseline보다 낫기는 해도 완전히 해결한 것은 아니라는 뜻이다.

비판적으로 보면, 이 논문의 진짜 기여는 “Fourier contour를 쓰는 것” 자체보다도, **instance를 sparse location에 응축된 explicit shape code로 표현하게 만드는 학습 프레임워크**에 있다. 이 관점은 매우 강력하지만, 동시에 positive anchor assignment나 representation capacity에 민감할 가능성이 있다. 그러나 제공된 텍스트에서는 positive 위치를 어떻게 구체적으로 정의하고 라벨링하는지 상세히 설명되지 않는다. 따라서 그 부분은 추정할 수 없다.

## 6. 결론

이 논문은 **Contour Proposal Network (CPN)** 라는 새로운 instance segmentation 프레임워크를 제안한다. 핵심은 객체를 픽셀 마스크로 직접 예측하는 대신, Fourier Descriptor 기반의 고정 길이 contour representation으로 직접 회귀한다는 점이다. 여기에 detection head, differentiable contour decoding, local refinement, NMS를 결합해 end-to-end로 학습 가능한 single-stage instance segmentation 모델을 구성했다.

실험적으로는 U-Net과 Mask R-CNN 대비 세 개의 데이터셋에서 더 높은 instance segmentation 성능을 보였고, 특히 높은 IoU 기준에서 우수한 contour 품질을 보였다. 또한 cross-dataset generalization에서도 더 좋은 결과를 보여, 이 접근이 단지 training set에 과적합된 shape prior가 아니라 비교적 robust한 instance representation을 학습한다는 점을 시사한다. 속도 측면에서도 일부 설정은 실시간 응용에 가까운 수준에 도달했다.

이 연구의 중요한 의미는 biomedical segmentation에서 **shape-aware explicit instance modeling**의 유효성을 설득력 있게 보여주었다는 점이다. 특히 morphology가 중요한 세포 분석, 겹침이 많은 현미경 영상, 객체 경계의 연속성과 완결성이 중요한 문제에서 실용적 가치가 크다. 더 넓게 보면, 닫힌 contour로 표현 가능한 객체 검출 문제라면 biomedical domain 밖에도 적용 가능성이 있다.

다만 제공된 텍스트 기준으로 볼 때, multiclass 확장, 복잡한 topology, contour/NMS 설계 선택의 세부 영향, 하이퍼파라미터 민감도 등은 후속 연구가 더 필요하다. 그럼에도 이 논문은 “instance segmentation을 어떤 표현 공간에서 풀 것인가”에 대해 매우 좋은 대안을 제시한 작업으로 볼 수 있다. 특히 shape를 hidden mask가 아니라 **직접 해석 가능한 객체 기술자(descriptor)** 로 다루었다는 점에서 학술적 가치가 높다.
