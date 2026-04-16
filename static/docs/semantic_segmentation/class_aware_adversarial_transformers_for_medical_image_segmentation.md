# Class-Aware Adversarial Transformers for Medical Image Segmentation

- **저자**: Chenyu You, Ruihan Zhao, Fenglin Liu, Siyuan Dong, Sandeep Chinchali, Ufuk Topcu, Lawrence Staib, James S. Duncan
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2201.10737

## 1. 논문 개요

이 논문은 2D 의료영상 분할을 위해 제안된 transformer 기반 모델인 `CASTformer`를 소개한다. 저자들은 기존 transformer 기반 의료영상 분할 모델이 유망하긴 하지만, 실제 분할 품질을 높이기 위해서는 아직 몇 가지 핵심 문제가 해결되지 않았다고 본다. 논문이 지적하는 주요 문제는 세 가지다. 첫째, 일반적인 patch tokenization은 영상을 기계적으로 잘라 토큰으로 만들기 때문에, 해부학적 구조나 세밀한 경계 정보를 충분히 반영하지 못한다. 둘째, 단일 해상도 혹은 단일 스케일 표현만 사용하면 장기나 병변의 크기 변화와 다양한 형태를 충분히 다루기 어렵다. 셋째, 단순한 segmentation network만으로는 풍부한 semantic context와 anatomical texture를 충분히 반영하기 어려워 최종 label map의 정밀도가 떨어질 수 있다.

이 문제는 의료영상에서 특히 중요하다. 의료영상 분할은 단순한 객체 분할이 아니라, 장기 경계, 병변 위치, 미세한 형태 차이를 정확히 잡아야 하며, 이 결과는 진단, 치료 계획, 치료 후 평가에 직접 연결된다. 따라서 long-range dependency를 잘 모델링하는 transformer의 장점과, 의료영상 특유의 구조적 정밀성을 함께 만족시키는 설계가 필요하다.

이 논문은 이런 배경에서 `CASTformer`를 제안한다. 구조적으로는 transformer 기반 generator와 transformer 기반 discriminator를 결합한 adversarial segmentation framework이며, 구체적으로는 다음 세 가지 축을 결합한다. 첫째, pyramid 구조를 이용한 multi-scale feature representation, 둘째, 중요한 해부학적 영역을 점진적으로 샘플링하는 `Class-Aware Transformer (CAT)` 모듈, 셋째, segmentation mask와 원본 영상을 결합한 class-aware image를 활용하는 adversarial training이다. 저자들은 이를 통해 세 개의 의료영상 벤치마크에서 기존 transformer 기반 방법들보다 Dice 기준으로 2.54%에서 5.88%의 절대 성능 향상을 보고한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 “transformer가 전역 문맥은 잘 보지만, 의료영상 분할에 정말 중요한 해부학적 관심 영역을 더 잘 보게 만들고, 동시에 다중 해상도 정보와 adversarial supervision을 함께 주면 분할 품질이 크게 좋아진다”는 것이다.

가장 핵심적인 설계는 `class-aware`라는 개념이다. 기존 ViT 스타일 토큰화는 영상을 고정 크기 patch로 나누고 이를 순서열로 처리한다. 이런 방식은 classification에는 어느 정도 유효할 수 있지만, 픽셀 단위로 정확한 경계를 요구하는 segmentation에서는 구조적으로 불리할 수 있다. 저자들은 이를 보완하기 위해 feature map 위에서 고정된 patch만 보는 대신, 반복적으로 sampling location을 이동시키면서 점차 더 discriminative한 위치를 보도록 만든다. 즉, 모델이 “어디를 봐야 하는지”를 학습하도록 한 것이다.

두 번째 아이디어는 multi-scale representation이다. 기존 transformer 기반 segmentation 모델 중 일부는 단일 해상도 특징에 크게 의존하는데, 의료영상에서는 큰 장기와 작은 장기가 함께 등장하고, 병변은 크기와 형태 변화가 심하다. 저자들은 pyramid 구조를 generator에 넣어 여러 해상도의 feature를 동시에 사용함으로써 local detail과 global context를 함께 활용한다.

세 번째 아이디어는 adversarial training의 재해석이다. 논문은 GAN을 단순히 “더 realistic한 출력”을 강제하는 용도로 쓰는 것이 아니라, segmentation mask와 원본 영상을 pixel-wise multiplication하여 해부학적 정보를 강조한 `class-aware image`를 만들고, 이를 discriminator에 입력한다. 이렇게 하면 discriminator가 low-level anatomical feature와 high-level semantic correlation을 모두 보게 되고, generator는 더 anatomically plausible한 segmentation을 하도록 압박받는다.

기존 접근과의 차별점은 분명하다. `TransUNet`처럼 CNN과 transformer를 결합하되, 단일 해상도 인코딩에 머무르지 않고 multi-scale pyramid를 도입했다. 또한 일반적인 patch tokenization 대신 반복적 sampling을 통해 class-aware attention을 유도한다. 마지막으로, transformer 기반 segmentation에 transformer 기반 discriminator를 붙여 adversarial하게 학습시키는 시도를 저자들은 2D 의료영상 분할 맥락에서 처음이라고 주장한다.

## 3. 상세 방법 설명

전체 모델 `CASTformer`는 generator와 discriminator로 구성된다. generator는 논문에서 `CATformer`라고 부르며, encoder module, class-aware transformer module, transformer encoder module, decoder module의 네 부분으로 이루어진다. discriminator는 사전학습된 ViT 기반 hybrid model 위에 얹힌 비교적 단순한 분류기다.

### 전체 파이프라인

입력 영상 $x \in \mathbb{R}^{H \times W \times 3}$가 들어오면, 먼저 CNN 기반 encoder가 여러 해상도의 feature map을 만든다. 이 feature들은 각 stage에서 class-aware transformer module과 transformer encoder module을 통과하여 정제된다. 이후 decoder가 네 단계 feature를 합쳐 최종 segmentation mask $y'$를 예측한다. adversarial 학습에서는 이 예측 mask와 원본 영상을 결합해 discriminator 입력을 만든다.

### Encoder와 Multi-scale 표현

저자들은 pure transformer 대신 CNN-Transformer hybrid를 선택한다. encoder는 40개의 convolutional layer를 사용하며, 이를 통해 여러 해상도의 feature map을 만든다. 논문은 이를 convolutional stem의 장점으로 설명한다. 하나는 downstream vision task에서 transformer 성능을 높일 수 있다는 점이고, 다른 하나는 고해상도와 저해상도 특징을 함께 제공해 더 좋은 표현을 만들 수 있다는 점이다.

논문 본문에서는 네 단계 feature를 다음처럼 정의한다.

- $F_1 \in \mathbb{R}^{H/2 \times W/2 \times C_1}$
- $F_2 \in \mathbb{R}^{H/2 \times W/2 \times (4C_1)}$
- $F_3 \in \mathbb{R}^{H/4 \times W/4 \times (8C_1)}$
- $F_4 \in \mathbb{R}^{H/8 \times W/8 \times (12C_1)}$

부록의 구체적 아키텍처 표에서는 ResNetV2 backbone을 사용하며, 입력 $224 \times 224 \times 3$에서 다음과 같은 네 레벨 특징을 얻는다고 적고 있다.

- RN-L1: $112 \times 112 \times 64$
- RN-L2: $56 \times 56 \times 256$
- RN-L3: $28 \times 28 \times 512$
- RN-L4: $14 \times 14 \times 1024$

이후 각 레벨은 독립적인 `CATformer-k`를 통과한다. 예를 들어 L1, L2, L3는 $(28 \times 28)$ 토큰 시퀀스로 맞춰지고, L4는 $(14 \times 14)$ 수준으로 처리된다. 논문은 이 설계를 통해 high-resolution detail과 coarse semantic context를 동시에 유지하려고 한다.

### Class-Aware Transformer Module

이 논문의 가장 중요한 부분이다. 목표는 해부학적으로 중요한 영역을 adaptive하게 찾는 것이다. 일반 ViT처럼 고정 patch만 쓰지 않고, feature map 상의 sampling 위치를 반복적으로 이동시키며 더 discriminative한 정보를 추출한다.

sampling 위치 업데이트는 다음 식으로 주어진다.

$$
s_{t+1} = s_t + o_t, \quad t \in \{1, \ldots, M-1\}
$$

여기서 $s_t \in \mathbb{R}^{2 \times (n \times n)}$는 현재 sampling location이고, $o_t \in \mathbb{R}^{2 \times (n \times n)}$는 그 위치를 얼마나 움직일지 나타내는 offset이다. 즉, 현재 위치에서 learned offset만큼 다음 위치를 이동시킨다.

초기 위치 $s_1$은 규칙적인 grid로 시작한다. 각 샘플링 위치는

$$
s_1^i = [\beta_y^i \tau_h + \tau_h/2,\; \beta_x^i \tau_w + \tau_w/2]
$$

로 정의되며, 여기서

$$
\beta_y^i = \lfloor i/n \rfloor,\quad \beta_x^i = i - \beta_y^i \cdot n
$$

이고, $\tau_h = H/n$, $\tau_w = W/n$이다. 쉽게 말하면 feature map을 $n \times n$ 구획으로 나눈 뒤, 각 칸의 중심점에서 시작하는 구조다.

각 단계에서 현재 위치 $s_t$를 사용해 feature map $F_i$에서 bilinear interpolation으로 토큰을 샘플링한다. 이 초기 샘플 토큰을 $I'_t$라고 하면,

$$
I'_t = F_i(s_t)
$$

이다. bilinear interpolation을 쓴 이유는 sampling 위치와 feature map에 대해 모두 미분 가능해서 end-to-end 학습이 가능하기 때문이다.

그 다음 위치 임베딩과 이전 단계 토큰을 현재 샘플 토큰과 합쳐 transformer에 넣는다.

$$
S_t = W_t s_t
$$

$$
V_t = I'_t \oplus S_t \oplus I_{t-1}
$$

$$
I_t = Transformer(V_t), \quad t \in \{1, \ldots, M\}
$$

여기서 $W_t$는 위치 $s_t$를 positional embedding으로 바꾸는 학습 가능한 행렬이고, $\oplus$는 element-wise addition이다. 즉, 현재 단계는 “현재 샘플된 특징 + 현재 위치 정보 + 이전 단계에서 누적된 표현”을 함께 사용한다.

이후 현재 토큰 $I_t$로부터 다음 단계 offset을 예측한다.

$$
o_t = \theta_t(I_t), \quad t \in \{1, \ldots, M-1\}
$$

이 식의 의미는 단순하다. 모델이 지금까지 본 특징을 바탕으로 “다음에는 어디를 더 봐야 하는가”를 스스로 계산하는 것이다. 그래서 이 모듈은 고정 attention이 아니라 점진적이고 적응적인 region refinement에 가깝다.

논문 실험에서는 sampling number $n = 16$, iteration 수 $M = 4$를 기본으로 사용한다. 부록 분석에 따르면 iteration 수는 늘릴수록 어느 정도 성능이 좋아지지만 $N=4$ 이후에는 증가폭이 크지 않았고, sampling 수 역시 $n=16$일 때 가장 좋았다.

### Transformer Encoder Module

CAT 모듈이 “어디를 볼지”를 정제한다면, Transformer Encoder Module(TEM)은 long-range dependency를 통합하는 표준 ViT 스타일 self-attention block 역할을 한다. 식은 다음과 같다.

$$
E_0 = [x_p^1 H; x_p^2 H; \cdots; x_p^N H] + H_{pos}
$$

여기서 각 patch $x_p^i$를 projection matrix $H$로 임베딩하고, positional embedding $H_{pos}$를 더한다.

이후 각 layer는 pre-norm residual transformer 형태를 따른다.

$$
E'_i = MSA(LN(E_{i-1})) + E_{i-1}
$$

$$
E_i = MLP(LN(E'_i)) + E'_i
$$

즉, layer normalization 뒤 multi-head self-attention, residual 연결, 다시 normalization 뒤 MLP, residual 연결의 구조다. 논문은 구현에서 ViT와 동일한 TEM을 사용했다고 명시한다. 부록 아키텍처 표에서는 각 레벨마다 `CAT × 4` 뒤에 `TEM × 12`가 붙는다.

### Decoder

decoder는 손으로 복잡하게 설계한 CNN decoder가 아니라, `All-MLP decoder`를 사용한다. 이 부분은 SegFormer의 영향을 받은 설계다. 핵심 절차는 다음과 같다.

첫째, 서로 다른 스케일의 feature channel 차원을 MLP로 통일한다.  
둘째, feature들을 $1/4$ 해상도로 upsample한 뒤 concatenate한다.  
셋째, 다시 MLP로 이 결합된 표현을 fusion하고 multi-class segmentation mask $y'$를 예측한다.

저자들은 이 decoder가 계산량이 작으면서도 transformer 특징의 local attention과 global attention을 잘 보존할 수 있다고 주장한다. 부록 Table 9에서도 FPN decoder보다 MLP decoder가 더 높은 DSC와 Jaccard를 보였다.

### Discriminator와 Adversarial Learning

discriminator는 `R50+ViT-B/16 hybrid model`을 ImageNet 사전학습 가중치로 초기화한 뒤, 그 위에 2-layer MLP를 붙여 real/fake 판별을 한다.

여기서 discriminator 입력이 중요하다. 논문은 원본 영상 $x$와 예측 segmentation mask $y'$를 pixel-wise multiplication해서 class-aware image $\tilde{x}$를 만든다. 이렇게 하면 segmentation이 강조한 해부학적 영역이 입력에 반영되므로, discriminator는 단순히 전체 이미지 realism이 아니라 “해부학적으로 의미 있는 분할이 되었는가”를 더 잘 보게 된다.

이 구조의 직관은 명확하다. generator는 mask를 잘 예측해야 하고, discriminator는 그 mask가 실제 해부학적 구조와 잘 맞는지 평가한다. 이 과정에서 discriminator는 low-level anatomical detail과 high-level semantics를 동시에 학습할 수 있다고 저자들은 설명한다.

### 학습 목표

generator의 총 loss는 segmentation loss와 adversarial loss의 합이다.

$$
L_G = \lambda_1 L_{CE} + \lambda_2 L_{DICE} + \lambda_3 L_{WGAN-GP}
$$

여기서 $L_{CE}$는 cross-entropy loss, $L_{DICE}$는 Dice loss, $L_{WGAN-GP}$는 Wasserstein GAN with Gradient Penalty loss다. 실험에서는

- $\lambda_1 = 0.5$
- $\lambda_2 = 0.5$
- $\lambda_3 = 0.1$

을 사용했다.

이 식의 의미는 다음과 같다. CE loss는 픽셀별 분류 정확도를 올리고, Dice loss는 클래스 불균형 상황에서 전체 분할 중첩도를 개선하며, WGAN-GP loss는 출력 마스크가 더 해부학적으로 자연스럽고 실제적인 구조를 갖도록 regularization 역할을 한다.

부록 Table 8에 따르면 MM-GAN, NS-GAN, LS-GAN보다 WGAN-GP가 전반적으로 가장 좋은 Dice/Jaccard를 보였다. 다만 95HD는 꼭 항상 최적은 아니며, 실제로 Table 8에서는 WGAN-GP의 95HD가 LS-GAN보다 약간 높다. 따라서 저자들의 주장은 전체 지표 종합 관점에서 이해하는 것이 맞다.

## 4. 실험 및 결과

### 데이터셋과 설정

논문은 세 개의 데이터셋에서 평가한다.

`Synapse`는 다기관 abdominal CT 기반 multi-organ segmentation 데이터셋이며, 총 30개 CT scan, 3779개 axial slice로 구성된다. 학습에는 18개 volume, 검증에는 12개 volume을 사용한다. 분할 대상은 aorta, gallbladder, spleen, left kidney, right kidney, liver, pancreas, stomach 등 8개 구조다.

`LiTS`는 MICCAI 2017 Liver Tumor Segmentation Challenge 데이터셋으로, 131개의 contrast-enhanced 3D abdominal CT volume으로 구성된다. 논문은 100개 volume을 학습, 31개를 테스트에 사용했다고 적고 있다. 분할 대상은 liver와 tumor다.

`MP-MRI`는 저자들의 in-house multi-phasic MRI 데이터셋으로, HCC 환자 20명의 multi-phasic MRI를 포함한다. 세 시점의 T1-weighted DCE-MRI가 registration되어 있으며, 48개 volume을 학습, 12개를 테스트에 사용한다.

구현 세부사항은 다음과 같다.

- Optimizer: AdamW
- Learning rate: $5 \times 10^{-4}$
- Batch size: 6
- Epoch: 300
- Input resolution: $224 \times 224$
- Patch size: 14
- Sampling number $n$: 16
- Iteration number $M$: 4
- Framework: PyTorch 1.7.0
- GPU: RTX 3090 24GB

평가지표는 Dice, Jaccard, 95% Hausdorff Distance(95HD), ASD를 사용한다. Dice와 Jaccard는 높을수록 좋고, 95HD와 ASD는 낮을수록 좋다.

### Synapse 결과

Synapse multi-organ CT에서 `CASTformer`는 평균 Dice 82.55%, Jaccard 74.69%를 기록했다. 비교 대상 중 가장 강한 baseline으로 제시된 `TransUNet`은 Dice 77.48%, Jaccard 64.78%였으므로, `CASTformer`는 Dice 기준 +5.07%, Jaccard 기준 +9.91% 절대 향상을 보였다.

모델별 주요 평균 성능은 대략 다음과 같다.

- UNet: Dice 70.11
- AttnUNet: Dice 71.70
- CoTr: Dice 72.60
- TransUNet: Dice 77.48
- SwinUNet: Dice 76.33
- CATformer: Dice 82.17
- CASTformer: Dice 82.55

장기별로도 개선이 나타난다. 논문은 left kidney, right kidney, liver, stomach 같은 큰 장기에서 각각 +2.77%, +2.51%, +1.35%, +4.95% Dice 향상이 있었다고 설명한다. small organ 측면에서도 aorta 89.05%, gallbladder 67.48%, pancreas 67.49%의 Dice를 얻었고, 특히 pancreas는 기존 대비 +10.91% 향상이라고 강조한다.

정성 결과 Figure 3에서도 `CASTformer`가 다른 방법보다 더 자세한 anatomical detail과 더 선명한 boundary를 보인다고 설명한다.

### LiTS 결과

LiTS에서는 `CASTformer`가 평균 Dice 73.82%, Jaccard 64.91%를 기록했다. `TransUNet`의 Dice 67.94%, Jaccard 60.25%와 비교하면 각각 +5.88%, +4.66% 향상이다. `CATformer`도 Dice 72.39%로 강한 성능을 보인다.

장기/종양별로 보면 `CASTformer`는 liver Dice 95.88%, tumor Dice 51.76%를 기록했다. 특히 tumor의 경우 `TransUNet`의 42.49%에서 크게 향상되었다. 이는 병변처럼 작고 복잡한 구조에서도 class-aware sampling과 adversarial 학습이 도움이 된다는 주장을 뒷받침한다.

### MP-MRI 결과

MP-MRI에서는 `CASTformer`가 Dice 94.93%, Jaccard 87.81%를 기록했고, `CATformer`는 Dice 94.17%, Jaccard 86.50%를 기록했다. `SETR`이 Dice 92.39, `TransUNet`이 Dice 92.08이므로, 평균 Dice 기준으로 각각 +2.54%, +2.85% 정도 개선된 셈이다. 논문은 discriminator를 쓴 `CASTformer`가 `CATformer`보다 더 낫다는 점을 통해 adversarial training의 유효성을 다시 강조한다.

### Transfer Learning 분석

논문은 pretrained model의 중요성을 강하게 주장한다. Synapse에서 `CATformer`는 pretrained 없이 Dice 74.84였고, pretrained 사용 시 Dice 82.17로 크게 증가했다. `CASTformer`는 generator와 discriminator 둘 다 pretrained를 쓸 때 Dice 82.55를 얻었고, 둘 다 pretrained를 쓰지 않으면 73.64에 그쳤다.

흥미로운 점은 discriminator만 pretrained하는 것보다 generator만 pretrained하는 것이 더 효과적이었다는 점이다.

- only pretrained D: Dice 78.87
- only pretrained G: Dice 81.46

즉, 이 논문에서는 segmentation 성능의 핵심 표현 학습이 generator 쪽에 더 크게 의존한다고 해석할 수 있다. 다만 이 해석은 논문이 제공한 결과에 기반한 것이고, 정확한 원인 메커니즘까지 실험적으로 분해했다고 보긴 어렵다.

### 구성요소 Ablation

모델 구성요소의 기여를 보면, baseline은 Dice 77.48이다. 여기서 CAT를 제거한 `CATformer w/o CAT`는 80.09, TEM을 제거한 `CATformer w/o TEM`는 81.35, 둘 다 포함한 `CATformer`는 82.17, adversarial training까지 포함한 `CASTformer`는 82.55다.

이 결과는 두 가지를 보여준다. 첫째, CAT와 TEM은 각각 독립적으로 유의미한 기여를 한다. 둘째, 둘을 같이 쓸 때가 하나만 쓸 때보다 좋다. 따라서 이 논문은 “class-aware sampling”과 “global contextual transformer encoding”이 상호보완적이라는 그림을 제시한다.

### 추가 분석

부록에서는 몇 가지 추가 관찰을 제공한다.

iteration 수는 $N=4$까지 늘리면 성능이 좋아지지만 그 이후는 포화된다.  
sampling 수는 $n=16$일 때 가장 좋다.  
segmentation loss는 Dice + CE가 Focal loss 계열과 비슷하거나 더 좋았다.  
sampling 모듈 비교에서는 DCN, Deformable DETR, DAT보다 CATformer/CASTformer가 더 좋은 결과를 보였다.  
decoder 비교에서는 FPN보다 MLP decoder가 더 좋았다.

또한 시각화 결과에서는 sampling point가 liver, kidney, spleen 등 의미 있는 장기 영역으로 점차 이동하는 경향을 보였고, transformer attention은 얕은 층에서는 texture 유사성에, 중간 층에서는 같은 클래스와 경계에, 깊은 층에서는 다른 클래스 관계까지 반응한다고 설명한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 설계의 각 요소가 분명한 문제의식과 연결되어 있다는 점이다. multi-scale pyramid는 장기 크기 다양성과 해상도 문제를 겨냥하고, CAT 모듈은 naive tokenization의 한계를 겨냥하며, adversarial discriminator는 segmentation map의 해부학적 realism 부족을 보완하려 한다. 단순히 여러 기법을 붙인 것이 아니라, 각 요소가 기존 transformer segmentation의 약점을 직접 겨냥하는 형태다.

두 번째 강점은 실험 결과가 비교적 강하게 나온다는 점이다. 특히 Synapse와 LiTS에서 기존 transformer 기반 SOTA 대비 Dice 절대 향상이 5% 내외로 크다. 의료영상 분할에서는 몇 퍼센트의 절대 향상도 의미가 큰 경우가 많은데, 이 논문은 여러 데이터셋에서 일관된 개선을 보고한다.

세 번째 강점은 ablation과 추가 분석이 비교적 풍부하다는 점이다. transfer learning, iteration 수, sampling 수, adversarial loss 종류, decoder 구조, sampling module 비교까지 포함되어 있어서, 제안 방법의 성능 향상이 우연이 아니라는 주장을 어느 정도 뒷받침한다.

네 번째 강점은 pretrained vision model을 활용한 데이터 효율성 메시지다. 의료영상은 annotation cost가 높고 데이터 수가 적은 경우가 많기 때문에, natural image pretraining이 실제로 큰 차이를 만든다는 결과는 실용적으로 중요하다.

반면 한계도 분명하다. 첫째, 이 논문은 주로 2D segmentation을 다룬다. 의료영상은 본질적으로 3D 정보가 중요한 경우가 많기 때문에, 제안 방법이 3D volumetric segmentation에서도 같은 수준으로 효과적인지는 이 논문만으로는 알 수 없다.

둘째, adversarial training의 일반적인 문제인 학습 안정성 문제가 여전히 잠재적으로 존재한다. 논문은 WGAN-GP를 사용해 이를 완화했다고 설명하지만, 실제로 표를 보면 95HD는 항상 가장 좋은 것은 아니고, discriminator 도입이 모든 지표를 일관되게 개선한다고 보기는 어렵다. 예를 들어 Synapse에서 `CATformer`가 95HD 16.20인데 `CASTformer`는 22.73으로 더 높다. Dice와 Jaccard는 올라갔지만 boundary-distance 계열 지표는 악화되었다. 따라서 “전반적 성능 향상”과 “모든 형태의 spatial accuracy 향상”은 구분해서 봐야 한다.

셋째, 논문은 class-aware sampling이 왜 정확히 잘 작동하는지에 대해 좋은 시각화를 제공하지만, 이 메커니즘이 특정 데이터셋 특성에 얼마나 의존하는지는 충분히 분해되지 않았다. 예를 들어 작은 장기에서 variance가 커서 어렵다는 가설은 제시하지만, 그 원인을 엄밀히 검증한 실험은 제공되지 않는다.

넷째, 계산량과 추론 속도에 대한 분석이 본문에 충분히 제시되지 않는다. multi-scale structure, 반복 sampling, transformer stack, discriminator까지 포함하면 분명 구조가 가벼운 편은 아니다. decoder는 lightweight하다고 하지만, 전체 시스템 차원의 메모리/속도 trade-off는 본문에서 명확히 정리되지 않는다.

다섯째, MP-MRI 데이터셋은 in-house 데이터셋이므로 일반화 가능성을 판단할 때 주의가 필요하다. 데이터 수가 상대적으로 작고 공개 벤치마크가 아니기 때문에, 이 결과만으로 광범위한 임상 일반화를 주장하기는 어렵다.

비판적으로 보면, 이 논문은 매우 강한 empirical engineering paper다. 문제 정의, 모듈 설계, 실험 결과가 모두 설득력 있지만, 이론적 측면에서 CAT sampling이 어떤 종류의 attention이나 deformable mechanism과 본질적으로 어떻게 다른지를 엄밀하게 정식화한 논문은 아니다. 또한 adversarial discriminator가 실제로 anatomical prior를 얼마나 학습했는지에 대해서도 정성적 직관은 충분하지만 정량적 증거는 제한적이다.

## 6. 결론

이 논문은 의료영상 2D segmentation을 위해 `CASTformer`라는 transformer 기반 adversarial segmentation framework를 제안했다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, CNN-Transformer hybrid encoder와 pyramid 구조를 결합해 multi-scale anatomical representation을 학습했다. 둘째, 반복적으로 sampling 위치를 갱신하는 class-aware transformer module을 통해 관심 해부학 영역을 더 잘 보도록 만들었다. 셋째, segmentation 결과와 영상을 결합한 class-aware image를 사용하는 transformer 기반 discriminator를 도입해 adversarial하게 학습했다.

실험적으로는 Synapse, LiTS, MP-MRI에서 모두 기존 방법보다 높은 Dice와 Jaccard를 보고했고, 특히 Synapse와 LiTS에서는 기존 강력한 transformer baseline보다 상당한 절대 향상을 보였다. 또한 pretraining의 효과, 각 모듈의 기여, loss 선택, sampling 설정 등을 비교적 폭넓게 검증했다.

실제 적용 측면에서 이 연구는 의료영상처럼 데이터가 적고 구조적 정밀성이 중요한 문제에서, 단순 ViT 적용만으로는 부족하며 “어디를 볼 것인가”와 “여러 스케일을 어떻게 통합할 것인가”, 그리고 “해부학적으로 plausible한 출력을 어떻게 강제할 것인가”가 중요하다는 점을 보여준다. 향후 연구에서는 이 아이디어를 3D volumetric segmentation, 더 큰 다기관 데이터셋, 계산 효율 개선, 그리고 더 명확한 해석 가능성 분석으로 확장할 여지가 크다. 전체적으로 보면, 이 논문은 의료영상 분할에서 transformer를 실용적으로 강화하는 강한 baseline이자, class-aware sampling과 adversarial transformer 결합의 가능성을 보여준 작업이라고 평가할 수 있다.
