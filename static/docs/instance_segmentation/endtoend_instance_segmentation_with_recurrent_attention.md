# End-to-End Instance Segmentation with Recurrent Attention

* **저자**: Mengye Ren, Richard S. Zemel
* **발표연도**: 2016
* **arXiv**: [https://arxiv.org/abs/1605.09410](https://arxiv.org/abs/1605.09410)

## 1. 논문 개요

이 논문은 **instance segmentation**을 하나의 이미지에서 모든 객체 인스턴스를 한 번에 예측하는 문제가 아니라, **RNN이 시간축을 따라 한 개씩 순차적으로 분리해내는 문제**로 재정의한다. 저자들은 semantic segmentation처럼 픽셀 단위 레이블링은 가능하지만, 같은 클래스에 속한 여러 개체를 서로 구분하는 것은 훨씬 어렵다고 지적한다. 특히 서로 가깝거나 가려진 객체를 분리하는 문제는 기존 fully convolutional 방식만으로 처리하기 어렵고, 이를 해결하기 위해 많은 기존 방법이 복잡한 graphical model이나 후처리 파이프라인에 의존했다.

논문의 핵심 목표는 세 가지로 요약할 수 있다. 첫째, **instance segmentation을 end-to-end로 학습 가능한 구조**로 만들고, 둘째, **가려짐(occlusion)** 상황에서 이미 찾은 객체를 바탕으로 다음 객체를 찾도록 하며, 셋째, **객체 수 추정(counting)** 과 segmentation을 함께 다루어 모델이 스스로 언제 멈춰야 하는지 학습하게 만드는 것이다. 이 문제는 자율주행, 로봇 조작, image captioning, visual question answering 같은 응용에서 매우 중요하다. 단순히 “무엇이 있느냐”를 넘어서 “몇 개가 있고 각각 어디까지가 한 객체냐”를 알아야 하기 때문이다.

저자들은 인간이 복잡한 장면에서 물체를 셀 때 한 번에 전체를 처리하기보다 주의를 옮겨가며 하나씩 세는 방식에서 영감을 얻는다. 이에 따라 이 논문은 **attention이 있는 recurrent architecture**를 사용하여 매 시점마다 하나의 객체를 선택하고, 그 객체의 bounding box와 segmentation mask를 생성하며, 동시에 그 시점의 발견이 유효한지 점수화한다. 이 설계는 결국 detection, segmentation, counting, stopping criterion을 하나의 순차적 모델 안에 통합하는 시도라고 볼 수 있다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 **instance segmentation을 “순차적 주의 기반 탐색 과정”으로 보는 것**이다. 한 번의 dense prediction으로 전체 인스턴스를 모두 분리하려고 하면 출력 구조가 너무 크고, 가려진 객체를 처리하기 위해 복잡한 suppression이나 graphical reasoning이 필요해진다. 반면 이 논문은 현재까지 분리한 객체들의 정보를 외부 메모리에 저장한 뒤, 그 정보를 참고하여 **다음에 어디를 볼지(top-down attention)** 를 결정한다. 이렇게 하면 이미 찾은 전경 객체를 고려하면서 뒤에 가려진 물체를 점진적으로 드러낼 수 있다.

기존 접근과의 차별점은 크게 네 가지다. 첫째, proposal 기반 파이프라인이나 graphical model 기반 후처리를 최소화하고 **하나의 recurrent model 안에서 localization과 segmentation을 함께 학습**한다. 둘째, 단순 NMS가 아니라, 이전 단계의 segmentation 결과를 메모리로 누적하여 **동적인 suppression과 occlusion reasoning**을 수행한다. 셋째, global image에서 직접 mask를 뽑는 대신 먼저 **attention으로 box를 찾고**, 그 박스 내부에서 더 정교하게 segmentation하는 2단계 구조를 순환적으로 반복한다. 넷째, score branch를 두어 각 step의 출력 신뢰도를 예측하고, 이를 통해 **객체 수 추정과 종료 판단**을 동시에 수행한다.

이 아이디어는 당시의 recurrent instance segmentation 계열과도 구분된다. 논문에서 비교하는 Romera-Paredes and Torr의 방식은 ConvLSTM이 전역 이미지 상에서 detection, inhibition, segmentation을 모두 처리해야 해서 멀리 떨어진 인스턴스를 억제하거나 정밀한 경계 정보를 다루기 어렵다고 본다. 이에 비해 이 논문은 **박스 네트워크로 관심 영역을 좁히고, 이전 segmentation을 외부 메모리로 직접 피드백**하기 때문에 더 정밀한 억제와 attention 이동이 가능하다고 주장한다.

## 3. 상세 방법 설명

전체 시스템은 네 개의 주요 구성요소로 이루어진다. **A) external memory**, **B) box proposal network**, **C) segmentation network**, **D) scoring network** 이다. 모델은 시점 $t=1,2,\dots,T$ 에 대해 반복적으로 동작하며, 매 step마다 하나의 객체 인스턴스에 대한 segmentation $\mathbf{y}_t$ 와 score $s_t$ 를 생성한다.

### 3.1 입력 전처리와 전체 흐름

입력 이미지는 원본 RGB 이미지 $\mathbf{x}_0 \in \mathbb{R}^{H\times W\times C}$ 이다. 그런데 본 모델은 이 이미지를 바로 쓰는 대신, 먼저 사전학습된 FCN으로 전처리한다. 이 FCN은 두 가지 출력을 낸다.

첫 번째는 **foreground mask** 이고, 두 번째는 각 foreground pixel이 객체 중심을 향하는 방향을 8개 클래스로 양자화한 **angle map** 이다. 따라서 전처리된 입력 $\mathbf{x}$ 는 총 9채널이다. 구체적으로 foreground 1채널과 angle 8채널을 포함한다. 저자들은 이 angle map이 단순 foreground보다 더 풍부한 경계 정보를 담아 인스턴스 분리를 도와준다고 설명한다.

### 3.2 Part A: External memory

external memory는 이미 찾은 객체들을 누적 저장하여 다음 객체 탐색에 활용하는 장치다. 시점 $t$ 에서의 canvas $\mathbf{c}_t$ 는 이전까지 예측한 segmentation의 누적 결과를 담는다.

$$
\mathbf{c}*{t}=
\begin{cases}
\mathbf{0}, & \text{if } t=0\
\max(\mathbf{c}*{t-1}, \mathbf{y}_{t-1}), & \text{otherwise}
\end{cases}
$$

즉 첫 step에서는 비어 있고, 이후에는 이전 canvas와 이전 step의 segmentation mask를 픽셀 단위 최대값으로 합친다. 이렇게 하면 “지금까지 이미 설명된 영역”이 누적된다.

이 canvas와 전처리된 입력을 결합해 box network의 입력을 만든다.

$$
\mathbf{d}_t = [\mathbf{c}_t, \mathbf{x}]
$$

논문 설명에 따르면 canvas는 총 10채널 역할을 한다. 첫 채널은 누적 segmentation이고, 나머지는 입력 영상 정보를 저장한다. 핵심은 이 메모리가 다음 attention step에서 “이미 찾은 객체는 피하고, 아직 설명되지 않은 객체를 찾는” top-down reasoning을 가능하게 한다는 점이다.

### 3.3 Part B: Box network

box network는 다음 객체가 있을 만한 **region of interest** 를 찾는다. 먼저 $\mathbf{d}_t$ 를 CNN에 넣어 feature map $\mathbf{u}_t$ 를 계산한다.

$$
\mathbf{u}_t = \text{CNN}(\mathbf{d}_t)
$$

이 feature map은 공간 차원을 갖는 $H' \times W' \times L$ 텐서다. 문제는 이 전체 feature map을 한 번에 처리하면 비효율적이고, 단순 pooling은 위치 정보를 잃는다는 점이다. 그래서 저자들은 **soft attention 기반 dynamic pooling**을 사용한다.

box network 내부에는 glimpse를 여러 번 보는 LSTM이 있다. 시점 $t$ 안에서 내부 glimpse index를 $\tau$ 로 둔다. 처음에는 attention weight $\alpha_{t,0}^{h,w}$ 를 모든 위치에 균일하게 놓고, 각 glimpse에서 현재 attention으로 가중합한 feature를 LSTM에 넣는다.

$$
\mathbf{z}*{t,\tau}=
\begin{cases}
\mathbf{0}, & \text{if } \tau=0\
\text{LSTM}\left(\mathbf{z}*{t,\tau-1}, \sum_{h,w}\alpha_{t,\tau-1}^{h,w}u_t^{h,w,l}\right), & \text{otherwise}
\end{cases}
$$

그리고 LSTM hidden state로부터 다음 attention map을 만든다.

$$
\alpha_{t,\tau}^{h,w}=
\begin{cases}
1/(H'W'), & \text{if } \tau=0\
\text{MLP}(\mathbf{z}_{t,\tau}), & \text{otherwise}
\end{cases}
$$

이 과정은 한 번의 glimpse로 box를 정하지 않고, 몇 차례 시선을 이동하면서 점점 관심 영역을 정교하게 좁히는 역할을 한다.

최종 hidden state $\mathbf{z}_{t,\text{end}}$ 에서 선형층을 통해 box parameter를 예측한다.

$$
[\tilde{g}*{X,Y}, \log \tilde{\delta}*{X,Y}, \log \sigma_{X,Y}, \gamma]
=======================================================================

\mathbf{w}*b^\top \mathbf{z}*{t,\text{end}} + w_{b0}
$$

여기서 $\tilde{g}_X, \tilde{g}_Y$ 는 정규화된 중심 좌표, $\tilde{\delta}_X, \tilde{\delta}_Y$ 는 box 크기, $\sigma_X, \sigma_Y$ 는 추출용 Gaussian kernel의 폭, $\gamma$ 는 나중에 segmentation patch를 원본 크기로 되돌릴 때 강도를 조절하는 스케일 값이다. 실제 이미지 좌표계로의 변환은 다음과 같다.

$$
g_X = (\tilde{g}_X + 1)W/2,\qquad
g_Y = (\tilde{g}_Y + 1)H/2
$$

$$
\delta_X = \tilde{\delta}_X W,\qquad
\delta_Y = \tilde{\delta}_Y H
$$

### 3.4 Gaussian kernel을 이용한 sub-region 추출

이 논문은 DRAW에서 사용된 것과 유사한 **Gaussian interpolation kernel** 을 사용해 관심 영역 patch를 differentiable하게 추출한다. 즉 hard crop이 아니라, box 파라미터로 정의된 연속적인 attention window를 이용해 patch $\mathbf{p}_t$ 를 뽑는다.

먼저 patch 내 위치 $i,j$ 에 대응하는 원본 이미지상의 Gaussian 중심을 정의한다.

$$
\mu_X^i = g_X + (\delta_X+1)\cdot(i-\tilde{W}/2+0.5)/\tilde{W}
$$

$$
\mu_Y^j = g_Y + (\delta_Y+1)\cdot(j-\tilde{H}/2+0.5)/\tilde{H}
$$

그리고 Gaussian filter matrix를 만든다.

$$
F_X^{a,i}
=========

\frac{1}{\sqrt{2\pi}\sigma_X}
\exp\left(-\frac{(a-\mu_X^i)^2}{2\sigma_X^2}\right)
$$

$$
F_Y^{b,j}
=========

\frac{1}{\sqrt{2\pi}\sigma_Y}
\exp\left(-\frac{(b-\mu_Y^j)^2}{2\sigma_Y^2}\right)
$$

원본 RGB 이미지와 memory-augmented 입력을 결합한 것을 $\tilde{\mathbf{x}}_t=[\mathbf{x}_0,\mathbf{d}_t]$ 로 두고, patch는

$$
\mathbf{p}_t = \text{Extract}(\tilde{\mathbf{x}}_t, F_Y, F_X) \equiv F_Y^\top \tilde{\mathbf{x}}_t F_X
$$

로 구한다. 중요한 점은 이 연산이 미분 가능하므로 box localization과 segmentation 학습이 end-to-end로 연결된다는 것이다.

### 3.5 Part C: Segmentation network

segmentation network는 box network가 추출한 patch $\mathbf{p}_t$ 내부에서 **dominant object**, 즉 그 윈도우 안에서 가장 중심이 되는 객체 하나를 분리한다. 먼저 patch를 CNN에 넣어 feature $\mathbf{v}_t$ 를 만들고,

$$
\mathbf{v}_t = \text{CNN}(\mathbf{p}_t)
$$

그 다음 deconvolution network로 patch-level segmentation heatmap $\tilde{\mathbf{y}}_t$ 를 얻는다.

$$
\tilde{\mathbf{y}}_t = \text{D-CNN}(\mathbf{v}_t)
$$

이 patch-level mask를 다시 원본 이미지 좌표계로 re-project한다. 여기서 앞서 구한 Gaussian filter의 transpose를 사용하고, $\gamma$ 로 박스 안 신호를 증폭하고, 상수 $\beta$ 로 박스 바깥을 억제한다.

$$
\mathbf{y}_{t}
==============

\text{sigmoid}\left(\gamma\cdot \text{Extract}(\tilde{\mathbf{y}}_t, F_Y^\top, F_X^\top)-\beta\right)
$$

이 식의 의미는 직관적으로 분명하다. segmentation network는 작은 patch에서 섬세하게 물체를 자르고, 그것을 원본 이미지 공간에 다시 올려놓는다. 이때 박스 외부는 $\beta$ 때문에 강하게 억제된다. 따라서 global full-image segmentation을 직접 수행하는 것보다 해상도와 계산량 측면에서 유리하다.

### 3.6 Part D: Scoring network

이 논문은 단순히 mask만 내는 것이 아니라, 현재 step에서 “유효한 객체를 하나 찾았는가”를 나타내는 score $s_t$ 를 예측한다. score는 box network의 hidden state와 segmentation feature를 함께 받아 계산한다.

$$
s_t
===

\text{sigmoid}\left(\mathbf{w}*{zs}^{\top}\mathbf{z}*{t,\text{end}}+\mathbf{w}*{vs}^{\top}\mathbf{v}*{t}+w_{s0}\right)
$$

이 score는 두 가지 역할을 한다. 하나는 counting이다. score가 1에 가까운 step은 실제 객체가 존재한다고 보고, 0에 가까운 step은 더 이상 객체가 없다고 본다. 다른 하나는 종료 조건이다. 추론 시에는 $s_t<0.5$ 가 되면 반복을 멈춘다. 논문은 최대 객체 수보다 1 큰 길이로 학습해 마지막 빈 step까지 모델이 보게 만든다.

### 3.7 손실 함수

전체 손실은 segmentation loss, box loss, score loss의 합이다.

$$
\mathcal{L}(\mathbf{y},\mathbf{b},\mathbf{s})
=============================================

\mathcal{L}_y(\mathbf{y},\mathbf{y}^*)
+
\lambda_b \mathcal{L}_b(\mathbf{b},\mathbf{b}^*)
+
\lambda_s \mathcal{L}_s(\mathbf{s},\mathbf{s}^*)
$$

논문에서는 $\lambda_b=\lambda_s=1$ 로 둔다.

#### 3.7.1 Matching IoU loss

instance segmentation에서 예측 인스턴스와 ground truth 인스턴스의 **순서가 다를 수 있다는 점**이 핵심 문제다. 이를 해결하기 위해 Hungarian algorithm을 이용한 최대 가중 bipartite matching을 사용한다. 예측 mask $\mathbf{y}_i$ 와 GT mask $\mathbf{y}_j^*$ 사이의 matching weight는 soft IoU이다.

$$
\mathcal{M}_{i,j}
=================

# \text{softIOU}(\mathbf{y}_i,\mathbf{y}_j^*)

\frac{\sum \mathbf{y}_i \cdot \mathbf{y}_j^*}
{\sum \mathbf{y}_i + \mathbf{y}_j^* - \mathbf{y}_i \cdot \mathbf{y}_j^*}
$$

이 matching을 구한 뒤 평균 IoU를 최대화하도록 loss를 음수로 둔다.

$$
\mathcal{L}_y(\mathbf{y},\mathbf{y}^*)
======================================

-\text{mIOU}(\mathbf{y},\mathbf{y}^*)
$$

즉 순서에 민감하지 않게, 각 예측과 GT의 가장 적절한 짝을 찾아 segmentation 품질을 평가한다. Hungarian algorithm 자체에는 gradient를 흘리지 않는다.

#### 3.7.2 Soft box IoU loss

box 좌표 간 정확한 IoU는 box가 겹치지 않을 때 gradient가 0이 되어 학습이 어렵다. 이를 피하기 위해 논문은 박스도 Gaussian re-projection을 통해 **soft mask처럼 표현**하고, pad된 GT box와 mIoU를 계산한다.

예측 box의 soft representation은

$$
\mathbf{b}_t
============

\text{sigmoid}\left(\gamma\cdot \text{Extract}(\mathbf{1}, F_Y^\top, F_X^\top)-\beta\right)
$$

이고, box loss는

$$
\mathcal{L}_b(\mathbf{b},\mathbf{b}^*)
======================================

-\text{mIOU}(\mathbf{b}, \text{Pad}(\mathbf{b}^*))
$$

이다. 이 방식은 hard box overlap 대신 differentiable한 overlap surrogate를 사용하는 셈이다.

#### 3.7.3 Monotonic score loss

모델이 앞쪽 step에서 더 확실한 객체를 먼저 내고, 뒤로 갈수록 score가 감소하도록 유도하는 것이 중요하다. 이를 위해 단순 binary cross-entropy가 아니라 **단조 감소를 장려하는 score loss**를 사용한다.

$$
\begin{aligned}
\mathcal{L}*{s}(\mathbf{s},\mathbf{s}^{*})=\frac{1}{T}\sum*{t}
&-s^{*}*{t}\log\left(\min*{t^{\prime}\leq t}{s_{t^{\prime}}}\right)\
&-(1-s^{*}*{t})\log\left(1-\max*{t^{\prime}\geq t}{s_{t^{\prime}}}\right)
\end{aligned}
$$

의미를 풀어 말하면, GT가 1인 위치까지는 이전 step들 중 최소값도 충분히 높아야 하고, GT가 0인 이후 구간에서는 이후 step들 중 최대값도 낮아야 한다. 그래서 score sequence 전체가 자연스럽게 앞에서 높고 뒤에서 낮아지도록 만든다.

### 3.8 학습 절차: bootstrap training과 scheduled sampling

이 모델은 구조상 coupling이 강하다. box network는 segmentation 결과를 참고하고, segmentation network는 box가 있어야 잘 동작한다. 그래서 처음부터 전부 joint training하면 학습이 불안정할 수 있다.

이를 해결하기 위해 저자들은 **bootstrap training**을 사용한다. 초기에 box network와 segmentation network를 각각 ground-truth segmentation과 ground-truth box를 사용해 pre-train하고, 이후 학습 단계에서 점차 모델 자신의 예측을 입력으로 쓰도록 바꾼다.

또한 **scheduled sampling**을 적용한다. external memory에 들어가는 이전 step segmentation을 항상 GT로 넣으면 train-test mismatch가 생긴다. 그래서 훈련 중에는 확률 $\theta_t$ 로 GT를 넣고, 나머지 경우에는 모델 예측을 넣는다. 이 $\theta_t$ 는 학습이 진행될수록 감소한다.

$$
\theta_t
========

\min\left(\Gamma_t \exp\left(-\frac{epoch-S}{S_2}\right), 1\right)
$$

$$
\Gamma_t = 1+\log(1+Kt)
$$

즉 학습 초반에는 GT 의존도가 높고, 후반에는 거의 완전히 자기 자신의 예측을 사용한다. 저자들은 이것이 실제 테스트 상황을 더 잘 모사하여 성능 향상에 기여한다고 본다.

## 4. 실험 및 결과

### 4.1 데이터셋과 평가 설정

논문은 서로 성격이 다른 네 가지 데이터셋에서 실험한다.

첫째, **CVPPP leaf segmentation** 은 식물 잎 인스턴스 분할 벤치마크다. A1 subset을 사용했고, 학습 128장, 테스트 33장으로 매우 작은 데이터셋이다.

둘째, **KITTI car segmentation** 은 자율주행 장면에서 자동차 인스턴스를 분리하는 문제다. 3,712개 학습 이미지, 120개 validation 이미지, 144개 test 이미지를 사용한다.

셋째, **Cityscapes** 는 8개 semantic class를 포함하는 대규모 urban scene instance benchmark다. 2,975장 학습, 500장 validation, 1,525장 test 이미지가 있다. 이 논문은 이를 **class-agnostic instance segmentation** 으로 학습하고, 이후 semantic segmentation mask를 덧씌워 class를 부여한다. car 클래스가 가장 흔하기 때문에 전체 평균과 car 단독 성능을 모두 보고한다.

넷째, **MS-COCO zebra subset** 은 방법의 이식성을 보기 위한 초기 실험이다. zebra 이미지 1000장을 사용한다. 직접 비교 가능한 기존 instance segmentation 방법이 없어서 정량 분석은 appendix 중심으로 제시한다.

### 4.2 평가 지표

CVPPP에서는 segmentation 품질에 **SBD (Symmetric Best Dice)** 를 사용하고, counting에는 $|\text{DiC}|$ 즉 평균 절대 count error를 사용한다.

KITTI에서는 **MUCov** 와 **MWCov** 를 사용한다. 둘 다 ground-truth 인스턴스별 IoU coverage를 보는 지표인데, MWCov는 큰 객체에 더 큰 가중치를 준다. counting 관련 보조 지표로 AvgFP, AvgFN도 사용한다.

Cityscapes에서는 **AP**, **AP50%**, **AP50m**, **AP100m** 를 사용한다. 이는 다양한 IoU threshold에서의 average precision을 반영하며, 특히 거리 구간별 성능도 살핀다.

### 4.3 CVPPP 결과

CVPPP에서 이 방법은 매우 강한 성능을 보인다. 표 1에 따르면 제안 방법은 **SBD 84.9**, **$|\text{DiC}|=0.8$** 을 기록한다. 비교 대상 가운데 이전 최고 수준이던 IPK는 SBD 74.4였고, RNN 기반 RIS+CRF는 SBD 66.6, $|\text{DiC}|=1.1$ 이었다. 따라서 segmentation과 counting 모두에서 큰 폭의 개선을 보였다.

흥미로운 점은 이 데이터셋이 너무 작아서, 저자들이 오히려 **FCN 전처리를 제거한 더 단순한 모델을 사용**했다고 밝힌 부분이다. 전처리 FCN이 성능을 높일 것 같지만, 작은 데이터셋에서는 입력 차원과 파라미터 수가 커져 overfitting이 심해졌다는 것이다. 이 결과는 제안 구조의 강점이 반드시 복잡한 전처리에만 의존하는 것은 아님을 보여준다.

### 4.4 KITTI 결과

KITTI test set 결과에서 제안 방법은 **MWCov 80.0**, **MUCov 66.9**, **AvgFP 0.764**, **AvgFN 0.201** 을 기록한다. 이는 DepthOrder [46] 와 DenseCRF [45] 보다 MWCov 면에서 높다. 그러나 **AngleFCN+Depth [41]** 의 MUCov 75.8에는 미치지 못한다.

저자들은 그 이유를 두 가지로 해석한다. 첫째, [41]은 **depth 정보**를 학습에 사용했는데, 이는 멀리 있는 차량 경계를 분리하는 데 도움이 되었을 수 있다. 둘째, 그 방법의 bottom-up **instance fusion** 후처리가 작은 물체 처리에 크게 기여했을 가능성이 있다. 반면 본 논문의 box network는 멀리 있는 작은 차량을 안정적으로 찾는 데 한계가 있었다고 설명한다.

그럼에도 qualitative result에서는 다양한 차량 자세와 심한 가려짐 상황을 잘 처리하는 모습을 보여준다. 저자들은 특히 top-down attentional inference가 강하게 작동한다고 해석한다.

### 4.5 Cityscapes 결과

Cityscapes 전체 클래스 기준으로 제안 방법은 **AP 9.5**, **AP50% 18.9**, **AP50m 16.8**, **AP100m 20.9** 를 기록한다. 전체 AP는 AngleFCN+Depth의 8.9보다 높다. car 클래스만 보면 성능이 더 두드러진다. 제안 방법은 **AP 27.5**, **AP50% 41.9**, **AP50m 46.8**, **AP100m 54.2** 로, AngleFCN+Depth의 car AP 22.5를 넘는다.

특히 car에서 큰 향상을 보인 것은 이 모델이 road scene에서 가장 빈번한 객체에 대해 attention과 sequential inhibition을 효과적으로 학습했음을 시사한다. 또한 박스 기반 local segmentation이 자동차 같은 비교적 구조가 명확한 객체에서는 잘 맞아떨어졌다고 볼 수 있다.

### 4.6 MS-COCO zebra 실험

Appendix의 Table 5에 따르면 zebra subset에서 제안 방법은 **MWCov 69.2**, **MUCov 64.2**, **$|\text{DiC}|=0.79$**, **Accuracy 0.57** 을 기록한다. counting 기준으로는 detector+NMS 기반 baseline의 $|\text{DiC}|=2.56$, associative subitizing의 1.03보다 좋다.

이 실험은 완전한 COCO instance segmentation benchmark 비교는 아니지만, 본 논문의 순차적 attention 방식이 자율주행이나 잎 데이터셋뿐 아니라 다른 object category에도 어느 정도 일반화될 수 있음을 보여주는 초기 근거로 해석할 수 있다.

### 4.7 Ablation study

KITTI validation set에서 ablation 결과는 각 구성 요소의 중요성을 비교적 명확히 보여준다.

전처리를 제거한 **No Pre Proc.** 는 MWCov 55.6, MUCov 45.0으로 크게 떨어진다. 이는 foreground+angle 전처리가 다음 객체 localization과 segmentation에 매우 중요함을 뜻한다.

**No Box Net** 도 MWCov 57.0, MUCov 49.1로 낮다. 즉 전체 이미지를 곧바로 segmentation하는 것보다, 먼저 박스를 찾고 그 안에서 분할하는 구조가 분명히 유리하다.

**No Angle** 은 MWCov 71.2, MUCov 63.3으로 full model보다 낮다. 이는 foreground만으로는 부족하고 angle map이 instance boundary representation에 실제로 도움을 준다는 근거다.

**No Scheduled Sampling** 은 성능이 약간 떨어진다. 이로부터 scheduled sampling이 train-test mismatch 완화에 실질적 도움이 있음을 볼 수 있다.

glimpse 수를 줄인 **Iter-1**, **Iter-3** 도 full model인 **Iter-5** 보다 낮다. 이는 LSTM이 여러 번 attention을 이동하며 정보를 모은 뒤 box를 결정하는 것이 중요하다는 뜻이다.

전반적으로 ablation은 이 논문이 단순히 “RNN을 썼다”는 수준이 아니라, **전처리, memory, local box, multi-glimpse attention, scheduled sampling** 의 조합으로 성능을 만든다는 점을 잘 보여준다.

### 4.8 저자들의 해석과 실패 사례

논문은 qualitative analysis도 함께 제시한다. 저자들은 이 모델이 자전거처럼 객체의 부분이 서로 떨어져 있는 경우도 한 인스턴스로 묶어낼 수 있고, 심하게 가려진 자동차나 zebra처럼 일부만 보이는 경우도 순차적 attention으로 찾아낼 수 있다고 주장한다. 또한 다른 많은 방법과 달리 **후처리 없이 직접 최종 segmentation을 출력**한다는 점도 장점으로 든다.

반면 실패 사례도 분명히 언급한다. 먼 거리의 작은 객체를 놓치거나, under-segmentation이 발생하는 경우가 있다. 저자들은 주요 원인 중 하나로 **약 4배 downsampling** 을 지적한다. KITTI와 Cityscapes를 계산량 때문에 축소해서 학습했는데, 이는 작은 객체 경계 복원에 불리했을 것이다. 또 사람 그림에서 실제보다 “세 번째 다리” 같은 비현실적 부분을 포함하는 경우가 있었는데, 이는 higher-order reasoning이 부족하다는 신호로 해석한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **instance segmentation을 순차적 attention 과정으로 정식화하여, detection, segmentation, counting, stopping을 하나의 모델로 통합했다는 점**이다. 당시 많은 방법이 proposal 생성, CRF, instance fusion, NMS tuning 같은 복잡한 파이프라인에 의존했다는 점을 생각하면, end-to-end recurrent formulation은 분명히 의미가 크다. 또한 occlusion을 정면으로 다루기 위해 external memory를 도입한 설계는 이 논문의 독창적 기여다. 단순 suppression이 아니라 “이미 찾은 객체 정보를 바탕으로 다음 객체를 찾는다”는 top-down reasoning은 사람이 장면을 해석하는 방식과도 잘 맞는다.

또 다른 강점은 **permutation-invariant matching loss** 와 **monotonic score loss** 의 조합이다. 순차적 출력 모델에서 GT 순서 문제와 종료 문제는 매우 중요하지만 종종 간단히 처리되곤 한다. 이 논문은 Hungarian matching으로 순서 불변성을 보장하고, score가 점점 줄어들도록 손실을 직접 설계했다. 이 부분은 단순한 architecture novelty를 넘어 학습 문제 자체를 세심하게 다뤘다는 점에서 가치가 있다.

실험적으로도 CVPPP와 Cityscapes car에서 상당히 강한 결과를 내며, qualitative result는 occlusion 장면에서 attention 기반 순차 모델의 장점을 잘 보여준다. 특히 full-image global inhibition보다 local box segmentation이 더 정밀한 경계 예측을 도와준다는 점이 실험과 ablation 모두에서 뒷받침된다.

반면 한계도 뚜렷하다. 첫째, 모델이 **사전 전처리 FCN** 에 의존한다. 논문은 시스템 전체를 end-to-end라고 표현하지만, 실제로는 foreground와 angle map을 생성하는 pre-trained FCN이 별도 단계로 존재한다. 따라서 엄밀한 의미에서 “원본 RGB에서 완전히 단일 네트워크로 end-to-end” 라고 보기는 어렵다. 다만 본문 내에서는 이 전처리기가 학습 파이프라인의 일부로 기능하며 이후 recurrent model이 joint learning된다는 취지로 이해할 수 있다.

둘째, box network가 존재한다는 것은 곧 **작은 객체나 멀리 있는 객체를 찾지 못하면 이후 segmentation도 실패한다**는 의미다. 저자들 스스로 distant cars를 잘 못 찾는다고 인정한다. 즉 이 구조는 “어디를 볼지” 결정하는 attention/localization 단계에 상당히 민감하다.

셋째, multi-class general instance segmentation에 대해서는 본문 기준으로 충분히 검증되지 않았다. Cityscapes는 class-agnostic segmentation 후 semantic mask를 씌우는 방식이고, MS-COCO는 zebra subset만 다룬다. 따라서 이 방법이 복잡한 다중 클래스 장면 전체에 대해 어느 정도 확장 가능한지는 논문만으로는 완전히 판단하기 어렵다.

넷째, higher-order reasoning이 부족하다. 사람의 팔다리처럼 구조적 제약이 필요한 객체에서는 비현실적 결합이 일어날 수 있다. 저자들은 future work로 bottom-up merging이나 higher-order graphical model과의 결합을 제안하지만, 이 논문 자체는 그러한 구조 제약을 직접 해결하지 않는다.

비판적으로 보면, 이 논문은 당시 시점에서 매우 설득력 있는 recurrent attention formulation을 제시했지만, 이후 분야가 Mask R-CNN류의 더 단순하고 강력한 detection-then-mask 구조로 빠르게 이동했다는 점을 떠올리게 한다. 그럼에도 이 논문의 의미는 “instance segmentation을 순차적 structured prediction으로 다룰 수 있다”는 가능성을 보여주고, attention과 memory를 이용한 top-down reasoning이 실제로 occlusion에 도움이 될 수 있음을 실험적으로 증명했다는 데 있다.

## 6. 결론

이 논문은 instance segmentation을 인간의 counting 과정과 유사한 **recurrent attentive process** 로 해석하고, 이를 구현하는 end-to-end 아키텍처를 제안했다. 모델은 external memory로 이전 객체를 기억하고, box network로 다음 관심 영역을 찾고, segmentation network로 해당 객체의 픽셀 마스크를 생성하며, scoring network로 객체 존재 여부와 종료 시점을 판단한다. 여기에 Hungarian matching 기반 segmentation loss, soft box IoU loss, monotonic score loss를 결합하여 순차 예측 문제를 안정적으로 학습한다.

실험에서는 CVPPP에서 큰 폭의 성능 향상을 보였고, KITTI와 Cityscapes에서도 경쟁력 있는 결과를 얻었다. 특히 car 클래스와 occlusion 장면에서 top-down attention의 장점이 잘 드러났다. 다만 작은 객체 탐지, 고차 구조 이해, 완전한 multi-class 확장성 면에서는 한계가 남아 있다.

종합하면 이 연구의 주요 기여는 단순히 새로운 네트워크 블록 하나를 제안한 것이 아니라, **instance segmentation 문제 자체를 순차적 attention과 memory 기반 구조로 다시 설계했다는 점**이다. 이 아이디어는 이후의 visual reasoning, sequential scene parsing, set prediction 문제를 이해하는 데도 의미가 있으며, 특히 “한 번에 전체를 맞히기 어려운 structured output을 시간축으로 분해해 푼다”는 관점에서 여전히 중요한 참고점이 된다.
