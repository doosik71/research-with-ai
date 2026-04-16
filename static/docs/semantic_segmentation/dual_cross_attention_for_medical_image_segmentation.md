# Dual Cross-Attention for Medical Image Segmentation

- **저자**: Gorkem Can Ates, Prasoon Mohan, Emrah Celik
- **발표연도**: 2023
- **arXiv**: https://arxiv.org/abs/2303.17696

## 1. 논문 개요

이 논문은 의료 영상 분할에서 널리 쓰이는 U-Net 계열 구조의 skip connection을 더 잘 동작하게 만들기 위한 모듈인 **Dual Cross-Attention (DCA)** 를 제안한다. 저자들의 문제의식은 분명하다. U-Net은 encoder에서 추출한 저수준 특징을 decoder로 직접 넘겨 주어 세부 경계를 복원하는 데 강점이 있지만, 단순 concatenation 기반 skip connection은 encoder 특징과 decoder 특징 사이의 **semantic gap** 을 충분히 줄이지 못한다. 즉, encoder의 다중 해상도 특징들이 decoder가 필요로 하는 형태로 정제되지 않은 채 전달된다는 것이다.

논문은 이 문제를 두 가지 한계와 연결해 설명한다. 첫째, convolution은 기본적으로 local receptive field를 점진적으로 넓혀 가므로, 멀리 떨어진 위치나 서로 다른 scale 사이의 전역적 상호작용을 한 번에 잡는 데 한계가 있다. 둘째, skip connection이 encoder와 decoder 사이의 의미적 차이를 그대로 남긴 채 결합을 수행하면, 단순히 정보를 많이 전달하는 것이 아니라 오히려 덜 정렬된 정보를 넘기게 될 수 있다. 의료 영상 분할에서는 경계, 작은 구조물, 장기/병변의 형태 보존이 중요하므로 이런 문제는 실제 성능 저하로 이어진다.

이 논문의 목표는 end-to-end transformer로 전체 네트워크를 무겁게 바꾸는 것이 아니라, **기존 U-Net 계열 모델의 encoder와 decoder 사이에 가볍고 일반적으로 붙일 수 있는 attention bridge** 를 설계하는 것이다. 저자들은 channel 축과 spatial 축의 전역 의존성을 순차적으로 포착하는 DCA를 통해 skip feature를 더 정제하고, 적은 파라미터 증가만으로 segmentation 성능을 안정적으로 높일 수 있음을 보이려 한다.

## 2. 핵심 아이디어

논문의 핵심 아이디어는 skip connection에 들어가는 multi-scale encoder feature들을 그대로 decoder에 보내지 말고, 먼저 서로를 참조하게 만들어 **“어떤 채널이 중요한가”와 “어떤 위치가 중요한가”를 전역적으로 정렬한 뒤** 넘기자는 것이다. 이를 위해 저자들은 두 단계 attention을 순차적으로 적용한다.

첫 번째는 **Channel Cross-Attention (CCA)** 이다. 여기서는 서로 다른 encoder stage에서 나온 multi-scale feature들을 channel 관점에서 함께 보면서, 각 scale의 feature가 다른 scale들의 channel 정보와 어떤 관계를 가지는지를 학습한다. 저자들의 설명대로 channel cross-attention은 spatial 위치를 통합적으로 고려하면서 “채널 대 채널” 관계를 본다. 따라서 어떤 semantic channel이 다른 scale에서 어떻게 보강되어야 하는지를 파악하는 역할을 한다.

두 번째는 **Spatial Cross-Attention (SCA)** 이다. CCA를 거친 뒤에는 spatial 차원에서 다시 cross-attention을 적용해, 여러 scale의 특징이 어떤 위치 관계를 가지는지 정교하게 반영한다. 즉 먼저 “무엇을 볼 것인가”를 channel 수준에서 정리한 뒤, 그 다음 “어디를 볼 것인가”를 spatial 수준에서 정제하는 구조다.

이 논문의 차별점은 단순히 channel attention과 spatial attention을 둘 다 쓴다는 데 있지 않다. 중요한 점은 다음과 같다.

첫째, **self-attention이 아니라 cross-attention** 을 사용한다. 각 stage를 독립적으로 attention 처리하는 것이 아니라, 여러 encoder stage의 특징을 함께 묶어 서로 참조하게 만든다. 이로 인해 multi-scale context를 직접 fusion할 수 있다.

둘째, **CCA 다음 SCA의 순차 구조** 를 채택했다. 저자들은 ablation을 통해 parallel fusion이나 역순(SCA-CCA)보다 이 순서가 더 좋다고 보고한다.

셋째, **경량화 설계** 가 강하게 들어가 있다. patch embedding에 학습 가능한 큰 convolution 대신 2D average pooling을 쓰고, attention의 projection도 linear layer 대신 $1 \times 1$ depth-wise convolution으로 대체하며, transformer류 구조에서 흔한 MLP 블록은 아예 제거했다. 따라서 이 방법은 “transformer를 붙였다”기보다, skip connection을 강화하기 위한 **lightweight attention adapter** 에 가깝다.

## 3. 상세 방법 설명

전체 구조는 일반적인 encoder-decoder segmentation 모델에 DCA block을 삽입하는 형태다. 논문 그림 설명에 따르면, encoder가 $n+1$개의 multi-scale stage를 가진다면 DCA는 마지막 bottleneck 직전까지의 처음 $n$개 encoder stage 출력을 입력으로 받는다. 그리고 이들을 정제한 뒤 대응되는 $n$개의 decoder stage로 연결한다. 즉, DCA는 encoder와 decoder 사이의 중간 다리 역할을 한다.

### 3.1 Multi-scale patch embedding

각 encoder stage의 feature를 $E_i \in \mathbb{R}^{C_i \times \frac{H}{2^{i-1}} \times \frac{W}{2^{i-1}}}$ 라고 두고, 각 stage마다 patch size를 $P_s^i = P_s^{2i-1}$ 로 정한다. 논문은 이 feature map들에서 patch를 뽑기 위해 2D average pooling을 사용한다. pooling size와 stride를 동일하게 $P_s^i$ 로 주어 non-overlapping patch들을 만들고, 이를 reshape한 다음 $1 \times 1$ depth-wise convolution으로 projection한다.

논문의 식은 다음과 같다.

$$
T_i = \mathrm{DConv1D}_{E_i}\big(\mathrm{Reshape}(\mathrm{AvgPool2D}_{E_i}(E_i))\big)
$$

여기서 $T_i \in \mathbb{R}^{P \times C_i}$ 이고, $P$는 patch 개수다. 중요한 점은 모든 scale에서 같은 개수의 patch token $P$를 갖도록 맞춘다는 것이다. 그래야 서로 다른 stage의 token들 사이에 cross-attention을 적용할 수 있다.

이 단계의 목적은 단순한 feature downsampling이 아니다. 서로 다른 해상도의 encoder feature들을 **공통된 token 형태** 로 바꾸어, 이후 channel/spatial cross-attention이 작동할 수 있게 만드는 것이다. 저자들은 여기서 average pooling을 쓰면 파라미터가 추가되지 않고, convolutional patch embedding보다 성능도 더 낫다고 ablation에서 보인다.

### 3.2 Channel Cross-Attention (CCA)

CCA에서는 각 $T_i$ 를 먼저 layer normalization한다. 그 다음 모든 stage의 token을 channel 차원으로 concatenation하여 하나의 통합 token $T_c$ 를 만든다. 각 stage별 token $T_i$ 는 query가 되고, 합쳐진 $T_c$ 는 key와 value를 만드는 데 사용된다. projection은 모두 $1 \times 1$ depth-wise convolution으로 수행한다.

$$
Q_i = \mathrm{DConv1D}_{Q_i}(T_i)
$$

$$
K = \mathrm{DConv1D}_K(T_c)
$$

$$
V = \mathrm{DConv1D}_V(T_c)
$$

여기서 $Q_i \in \mathbb{R}^{P \times C_i}$, $K \in \mathbb{R}^{P \times C_c}$, $V \in \mathbb{R}^{P \times C_c}$ 이다. $C_c$ 는 concatenated channel 수를 의미한다.

CCA의 핵심은 attention을 channel 축에서 수행하기 위해 query, key, value를 transpose해서 사용한다는 점이다. 논문 식은 다음과 같다.

$$
\mathrm{CCA}(Q_i, K, V) =
\mathrm{Softmax}\left(\frac{Q_i^T K}{\sqrt{C_c}}\right)V^T
$$

직관적으로 보면, 특정 stage의 각 channel이 다른 모든 stage의 channel들과 얼마나 관련 있는지를 attention으로 계산하고, 그 가중합으로 새로운 channel 표현을 만든다. 이 과정은 skip feature 안의 semantic channel들을 더 잘 정렬해 주는 역할을 한다. 예를 들어 어떤 scale에서는 경계 정보가 강하고, 다른 scale에서는 큰 구조의 문맥이 강할 수 있는데, CCA는 이런 정보를 channel 수준에서 섞어 준다.

저자들은 conventional transformer의 linear projection 대신 depth-wise convolution을 사용한다. 이는 지역 정보를 일부 유지하면서도 계산량과 파라미터를 줄이기 위한 선택이다. CCA의 출력은 다시 depth-wise convolution projection을 거친 후 SCA의 입력이 된다.

### 3.3 Spatial Cross-Attention (SCA)

CCA를 통과한 출력들을 $\bar{T}_i \in \mathbb{R}^{P \times C_i}$ 라고 하자. SCA에서는 이 출력들을 다시 layer normalization하고, channel 차원으로 concatenation하여 $\bar{T}_c$ 를 만든다. 이번에는 CCA와 반대로, concatenated token $\bar{T}_c$ 로부터 query와 key를 만들고, 각 stage의 $\bar{T}_i$ 를 value로 사용한다.

$$
Q = \mathrm{DConv1D}_Q(\bar{T}_c)
$$

$$
K = \mathrm{DConv1D}_K(\bar{T}_c)
$$

$$
V_i = \mathrm{DConv1D}_{V_i}(\bar{T}_i)
$$

여기서 $Q \in \mathbb{R}^{P \times C_c}$, $K \in \mathbb{R}^{P \times C_c}$, $V_i \in \mathbb{R}^{P \times C_i}$ 이다.

SCA는 spatial token 간 attention을 계산한다.

$$
\mathrm{SCA}(Q, K, V_i) =
\mathrm{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V_i
$$

여기서 $d_k$ 는 scaling factor이며, multi-head일 때는 $d_k = \frac{C_c}{h_c}$ 이다. 논문 구현에서는 CCA head 수는 1, SCA head 수는 4로 두었다.

이 식의 의미는, 여러 scale을 합친 문맥을 기준으로 spatial position들 사이의 관계를 계산한 뒤, 각 stage의 value feature를 재가중하는 것이다. 즉 CCA가 “어떤 feature channel이 중요한가”를 먼저 조정했다면, SCA는 “어떤 위치들이 서로 관련되는가”를 정리한다. 이 순서 때문에 논문은 DCA를 dual attention의 sequential cross-attention 구조로 본다.

### 3.4 최종 출력과 decoder 연결

SCA의 출력은 다시 depth-wise convolution projection을 통과해 DCA 출력을 형성한다. 이후 layer normalization과 GeLU를 적용한다. 마지막으로 각 출력은 up-sampling을 거친 뒤, $1 \times 1$ convolution, batch normalization, ReLU를 순서대로 적용해 decoder의 대응 stage와 연결된다.

여기서 중요한 개념적 차이는 다음과 같다. self-attention은 보통 하나의 feature map 내부 관계를 본다. 반면 이 논문의 cross-attention은 **서로 다른 encoder stage의 feature를 함께 attention의 재료로 넣어 attention map을 형성한다**. 따라서 단순히 각 stage를 강화하는 것이 아니라, multi-scale encoder 특징을 공동으로 정렬하여 semantic gap을 줄이는 효과를 노린다.

## 4. 실험 및 결과

### 4.1 데이터셋과 설정

논문은 다섯 개의 benchmark 의료 영상 분할 데이터셋에서 실험했다.

GlaS는 gland segmentation 데이터셋으로 학습 85장, 테스트 80장이다.  
MoNuSeg는 핵 분할(nuclear segmentation) 데이터셋으로 학습 30장, 테스트 14장이다.  
CVC-ClinicDB는 colonoscopy 이미지 612장을 포함한다.  
Kvasir-SEG는 polyp segmentation 데이터셋으로 1000장의 annotated image를 가진다.  
Synapse는 8개 복부 장기 분할을 위한 abdominal CT 데이터셋으로 30개 scan 중 18개를 학습, 12개를 테스트에 사용했다.

CVC-ClinicDB와 Kvasir-SEG는 80/20으로 랜덤 분할했다. 모든 이미지는 $224 \times 224$ 로 resize했다. optimizer는 Adam, 초기 learning rate는 $10^{-4}$, loss는 Dice loss, metric은 DSC(Dice Similarity Coefficient)와 IoU를 사용했다. 학습 epoch는 200이며, 가장 좋은 결과를 보고했다. 데이터 증강은 random rotation, vertical flip, horizontal flip을 사용했다.

실험 대상 모델은 총 6개다: U-Net, V-Net, R2Unet, ResUnet++, DoubleUnet, MultiResUnet. 즉 논문은 DCA를 하나의 전용 backbone으로 제안하는 것이 아니라, **서로 다른 U-Net 계열 모델에 꽂아 넣을 수 있는 plug-in 모듈** 로 검증한다.

### 4.2 전체 결과

가장 중요한 결과는 Table 1이다. 전반적으로 DCA는 거의 모든 조합에서 DSC와 IoU를 개선한다. 파라미터 증가는 매우 작다.

예를 들어 U-Net은 8.64M 파라미터에서 GlaS DSC 88.87, MoNuSeg DSC 77.14, CVC-ClinicDB DSC 89.63, Kvasir-Seg DSC 82.99, Synapse DSC 78.55를 기록했다. 여기에 DCA를 붙인 U-Net(DCA)은 8.75M으로 소폭 증가하면서 GlaS 89.66, MoNuSeg 78.13, Kvasir-Seg 84.03, Synapse 78.98로 좋아졌다. CVC-ClinicDB에서는 89.63에서 89.53으로 아주 미세하게 감소했지만, 전반 추세는 개선이다.

ResUnet++도 GlaS에서 85.43에서 87.35, MoNuSeg에서 75.68에서 77.40, CVC-ClinicDB에서 89.46에서 90.19, Synapse에서 75.91에서 77.35로 상승했다. Kvasir-Seg에서는 82.26에서 82.07로 아주 소폭 하락했다.

MultiResUnet은 GlaS에서 약간 하락했지만(88.99 → 88.86), MoNuSeg, CVC-ClinicDB, Kvasir-Seg, Synapse에서는 모두 개선되었다.  
R2Unet도 거의 모든 데이터셋에서 개선되었다.  
V-Net도 모든 데이터셋에서 개선되었다.  
DoubleUnet 역시 모든 데이터셋에서 개선되었고, 특히 MoNuSeg 77.16 → 79.50, CVC-ClinicDB 90.20 → 90.86, Kvasir-Seg 84.40 → 85.16로 좋은 향상을 보였다.

저자들이 강조한 최고 개선폭은 다음과 같다.

- GlaS에서 최대 DSC 향상: 2.05%
- MoNuSeg에서 최대 DSC 향상: 2.74%
- CVC-ClinicDB에서 최대 DSC 향상: 1.37%
- Kvasir-Seg에서 최대 DSC 향상: 1.12%
- Synapse에서 최대 DSC 향상: 1.44%

이 수치는 “항상 압도적 개선”이라기보다, **여러 모델과 여러 데이터셋에서 일관되게 손해가 적고 이득이 자주 발생하는 보편적 skip 강화 모듈** 이라는 점을 보여 준다. 의료 영상 분할에서는 1% 안팎의 Dice 향상도 의미가 큰 경우가 많기 때문에, 작은 파라미터 증가로 이러한 향상을 얻는 것은 실용적 가치가 있다.

### 4.3 파라미터 증가와 효율성

논문은 DCA가 성능뿐 아니라 효율성 측면에서도 실용적이라고 주장한다. 예를 들어 ResUnet++는 skip connection layer가 3개여서 DCA 추가 시 파라미터 증가가 0.7% 미만이다. U-Net, MultiResUnet, R2Unet, V-Net처럼 skip layer가 4개인 모델은 0.3%에서 1.5% 사이의 증가를 보인다. DoubleUnet은 세 개의 skip scheme에 각각 DCA를 넣어 총 3.4% 증가하지만, 저자들은 이것도 여전히 작은 증가라고 해석한다.

이 점은 논문의 주요 메시지와 맞닿아 있다. 즉, 전체 backbone을 transformer화하지 않고도, skip connection만 정교하게 손봐도 실제 성능을 끌어올릴 수 있다는 것이다.

### 4.4 정성적 결과

논문은 Figure 3에서 plain model과 DCA 적용 모델의 segmentation 결과를 시각적으로 비교한다. 텍스트 설명에 따르면, DCA를 사용한 모델은 경계(boundary)를 더 일관되게 복원하고, shape 정보를 더 잘 유지하며, false positive를 줄여 discrete part를 더 잘 구분한다. 이는 DCA가 단순히 평균적인 metric만 올린 것이 아니라, 실제 분할 마스크의 구조적 품질을 개선했음을 보여 주려는 결과다.

다만 현재 제공된 텍스트만으로는 그림 내부의 개별 사례를 세부적으로 해석할 수는 없으므로, 여기서는 논문 본문에 적힌 수준까지만 말할 수 있다.

### 4.5 Ablation study

#### DCA layout

U-Net 기반 ablation에서 CCA만 썼을 때도 GlaS에서 88.87 → 89.07, MoNuSeg에서 77.14 → 77.78로 개선된다. SCA만 썼을 때도 GlaS 89.48, MoNuSeg 77.36으로 개선된다. 그러나 **CCA-SCA 순차 구조** 가 GlaS 89.66, MoNuSeg 78.13으로 가장 좋다. 반대로 SCA-CCA 순서는 GlaS 89.03, MoNuSeg 77.90으로 덜 좋다.

이는 저자들의 설계 가설, 즉 먼저 channel 수준의 semantic alignment를 수행하고 그 다음 spatial refinement를 하는 것이 더 낫다는 주장을 뒷받침한다.

#### CCA와 SCA의 fusion 방식

저자들은 세 가지 fusion을 비교한다.

- summation: CCA와 SCA를 병렬로 계산해 합산
- concatenation: 병렬 계산 후 concat
- sequential: 순차 적용

결과는 sequential fusion이 가장 좋다. 이는 dual attention의 정보 결합이 단순 병렬 결합보다, 한 attention의 출력을 다음 attention이 받아 정제하는 방식이 더 효과적이라는 뜻이다.

#### Patch embedding 방식

average pooling 기반 patch embedding과 convolutional patch embedding을 비교한 결과, average pooling이 더 좋다. GlaS에서 DCA-Conv는 89.52, DCA-AP는 89.66이고, MoNuSeg에서도 77.62 대 78.13으로 average pooling이 우세하다. 게다가 convolutional patch embedding은 약 260K의 추가 파라미터가 든다.

이 결과는 논문의 경량화 주장에 힘을 실어 준다. 복잡한 embedding이 항상 유리한 것이 아니라, 이 문제에서는 간단한 pooling이 오히려 더 적절하다는 것이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 설정이 명확하고 해결 방식이 구조적으로 깔끔하다** 는 점이다. 저자들은 U-Net의 skip connection이 가진 semantic gap 문제를 정확히 짚고, 이를 해결하기 위해 encoder feature 자체를 multi-scale cross-attention으로 정제하는 접근을 제시한다. 방법론도 지나치게 복잡하지 않다. CCA와 SCA라는 두 모듈을 순차적으로 배치하고, patch embedding과 projection을 경량화해 plug-in처럼 사용할 수 있게 만들었다.

둘째, **범용성** 이 강점이다. 한 모델에서만 잘 되는 기법이 아니라 U-Net, V-Net, R2Unet, ResUnet++, DoubleUnet, MultiResUnet 등 여러 구조에서 실험했다. 이는 DCA가 특정 backbone에 과도하게 맞춰진 기법이 아니라, skip connection이 있는 encoder-decoder 구조 전반에 적용 가능한 아이디어임을 보여 준다.

셋째, **효율 대비 성능 향상** 이 설득력 있다. 논문은 거대한 transformer backbone을 새로 제안하지 않으면서도, 작은 파라미터 증가만으로 여러 데이터셋에서 Dice와 IoU를 개선한다. 실제 의료 영상 환경에서는 모델 경량성, 학습 비용, 재현성도 중요하므로, 이런 설계는 실용적이다.

넷째, **ablation study가 비교적 충실하다**. CCA만 쓴 경우, SCA만 쓴 경우, 순서 반대의 경우, fusion 전략 차이, patch embedding 차이까지 검증해, 왜 최종 설계를 택했는지 논리적으로 설명한다.

반면 한계도 있다.

첫째, 논문은 여러 데이터셋과 여러 모델에서 개선을 보였지만, **개선폭이 항상 큰 것은 아니고 일부 조합에서는 매우 미세하거나 소폭 하락도 있다**. 예를 들어 U-Net의 CVC-ClinicDB, ResUnet++의 Kvasir-Seg, MultiResUnet의 GlaS에서 절대적인 개선이 없거나 소폭 감소한다. 따라서 “항상 이긴다”기보다는 “대체로 도움이 된다”는 수준으로 읽는 것이 정확하다.

둘째, 본문 텍스트 기준으로는 **연산량(FLOPs), 추론 속도, 메모리 사용량** 에 대한 정량 비교가 제시되지 않는다. 저자들은 parameter overhead가 작다고 강조하지만, attention 연산은 파라미터 수와 별개로 runtime cost가 있을 수 있다. 특히 high-resolution feature에 spatial attention을 적용할 때 실제 latency가 어느 정도인지가 중요하지만, 제공된 텍스트에는 이 정보가 없다.

셋째, DCA가 왜 특정 데이터셋이나 특정 backbone에서 더 잘 듣는지에 대한 **심층 분석은 제한적** 이다. 예를 들어 histopathology, endoscopy, CT처럼 modality가 다른 환경에서 어떤 attention 경향이 나타나는지, channel/spatial attention map이 어떤 구조를 학습하는지에 대한 해석은 거의 없다.

넷째, 논문은 2D segmentation 중심으로 기술되어 있고, V-Net처럼 원래 3D에서 자주 쓰이는 구조를 포함하긴 하지만, **본 방법이 3D volumetric attention으로 자연스럽게 일반화되는지** 는 이 텍스트만으로 명확하지 않다. Synapse 결과는 포함되어 있지만, 어떤 입력 표현과 slice 처리 방식을 사용했는지 상세 설명은 제공되지 않았다. 따라서 3D 의료 영상 전체에 대한 일반성을 강하게 주장하기는 어렵다.

다섯째, 이 논문은 DCA를 skip connection 강화 모듈로 설계했기 때문에 실용적이지만, 반대로 말하면 **병목 지점은 skip connection만이 아닐 수 있다**. encoder 표현 자체가 약하거나 decoder 설계가 병목일 경우에는 DCA만으로 해결되지 않을 수 있다. 논문도 이 점을 직접 다루지는 않는다.

비판적으로 보면, 이 연구의 가치는 “새로운 대형 아키텍처”라기보다 **의료 영상 분할에서 skip connection을 어떻게 더 똑똑하게 만들 것인가에 대한 잘 정리된 engineering contribution** 에 있다. 이 목표에 대해서는 충분히 성공적이지만, 이론적 분석이나 attention 해석성 측면의 깊이는 상대적으로 제한적이다.

## 6. 결론

이 논문은 U-Net 계열 의료 영상 분할 모델의 skip connection에서 발생하는 semantic gap 문제를 해결하기 위해 **Dual Cross-Attention (DCA)** 를 제안했다. DCA는 multi-scale encoder feature를 공통 token 형태로 바꾼 뒤, **CCA로 channel 의존성을 먼저 정제하고, SCA로 spatial 의존성을 이어서 정제하는 순차적 cross-attention 구조** 를 사용한다. 또한 average pooling 기반 patch embedding, depth-wise convolution projection, MLP 제거를 통해 경량화까지 달성했다.

실험적으로 DCA는 여섯 개의 U-Net 기반 모델과 다섯 개의 의료 영상 분할 데이터셋에서 전반적으로 성능을 향상시켰다. 특히 파라미터 증가가 작고, 여러 backbone에 적용 가능하다는 점에서 실용성이 높다. 따라서 이 연구는 “더 큰 모델”을 만드는 방향보다, **기존 segmentation 구조의 정보 전달 경로를 정교화하는 방향** 이 실제로 의미 있다는 점을 보여 준다.

향후 연구 측면에서는, DCA를 3D medical segmentation이나 transformer encoder-decoder와 결합해 보는 확장, attention map 해석을 통한 작동 원리 분석, FLOPs 및 latency까지 포함한 더 엄밀한 효율성 검증이 중요한 후속 과제가 될 수 있다. 그럼에도 현재 논문이 제시한 결과만으로도, DCA는 의료 영상 분할에서 skip connection 설계의 유효한 개선 방향으로 평가할 만하다.
