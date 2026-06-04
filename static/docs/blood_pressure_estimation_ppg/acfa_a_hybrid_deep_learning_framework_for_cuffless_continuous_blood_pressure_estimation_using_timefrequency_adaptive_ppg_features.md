# ACFA: A Hybrid Deep Learning Framework for Cuffless Continuous Blood Pressure Estimation Using Time–Frequency Adaptive PPG Features

* **저자**: Feng Li, Lingjie Li, Li Wang, Hongzeng Xu
* **발표연도**: 2026

## 1. 논문 개요

이 논문은 PPG(Photoplethysmography) 신호만을 이용하여 커프 없이 연속적으로 혈압을 추정하는 딥러닝 기반 프레임워크인 **ACFA(Adaptive Cross-domain Fusion Architecture)** 를 제안한다. 논문은 IEEE Access에 2026년에 게재되었으며, DOI는 10.1109/ACCESS.2026.3657471이다. 논문 본문에는 arXiv 주소가 제시되어 있지 않으므로 메타데이터의 arXiv URL은 비워 두었다.

논문의 핵심 목표는 비침습적이고 연속적인 혈압 모니터링을 위해, 단일 PPG 신호로부터 SBP(Systolic Blood Pressure), DBP(Diastolic Blood Pressure), MAP(Mean Arterial Pressure)을 정확하게 예측하는 것이다. 기존 커프 기반 혈압계는 측정 시 압박으로 인한 불편함이 있고, 연속 측정이 어렵다는 한계가 있다. 반면 PPG는 광학 센서를 피부에 부착하여 혈류량 변화에 따른 파형을 측정하는 방식이므로, 웨어러블 기기와 결합하기 쉽고 장시간 모니터링에 적합하다.

연구 문제는 단순히 PPG에서 혈압을 예측하는 회귀 문제로 보일 수 있지만, 실제로는 매우 어렵다. PPG 파형은 심박, 혈관 탄성, 말초 저항, 반사파, 측정 위치, 움직임 잡음, 개인별 생리적 차이에 의해 크게 변한다. 또한 동일한 혈압이라도 사람마다 PPG 형태가 다를 수 있고, 반대로 유사한 PPG 형태가 서로 다른 혈압 상태를 나타낼 수도 있다. 따라서 모델은 단기적인 파형 형태뿐 아니라 여러 심박 주기에 걸친 장기적 변화, 주파수 영역의 생리적 패턴, 개인 간 변동성을 동시에 처리해야 한다.

논문은 기존 방법의 한계를 세 가지로 정리한다. 첫째, 기존 CNN 또는 LSTM 기반 방법은 PPG 시퀀스 내부의 장기 의존성(long-term dependency)을 충분히 포착하기 어렵다. 둘째, 주파수 영역 특징 추출이 얕거나 제한적이어서 다중 스케일 패턴을 충분히 활용하지 못한다. 셋째, 고차원 특징의 중복성이 모델 복잡도를 증가시키고 일반화 성능을 저하시킨다.

이를 해결하기 위해 저자들은 ACFA라는 하이브리드 딥러닝 구조를 제안한다. ACFA는 DyCASNet, xLSTM, Transformer, FKAN이라는 네 가지 모듈을 순차적으로 결합한다. DyCASNet은 시간 영역과 주파수 영역의 적응형 특징을 추출하고, xLSTM은 장기 시계열 의존성을 강화하며, Transformer는 전역 문맥 정보를 학습하고, FKAN은 복잡한 비선형 회귀 관계를 모델링한다. 이 구조는 PPG 신호를 시간, 주파수, 국소, 전역, 비선형 관점에서 통합적으로 해석하려는 시도로 볼 수 있다.

## 2. 핵심 아이디어

논문의 중심 직관은 PPG 기반 혈압 추정에서 단일한 특징 추출 방식만으로는 충분하지 않다는 것이다. PPG 신호에는 혈압과 관련된 정보가 여러 형태로 섞여 있다. 예를 들어 systolic peak, dicrotic notch, diastolic peak와 같은 파형상의 형태학적 특징은 시간 영역에서 관찰된다. 반면 혈관 탄성, 심박 변동, 반사파에 의해 생기는 주기성 및 에너지 분포는 주파수 영역에서도 의미를 가진다. 또한 한 심박 주기 내부의 국소적 변화와 여러 심박 주기에 걸친 전역적 변화가 모두 혈압 추정에 영향을 준다.

ACFA는 이러한 정보를 하나의 모델 안에서 단계적으로 처리하도록 설계되었다. 먼저 DyCASNet은 PPG 신호를 시간 영역과 주파수 영역 양쪽에서 분석하여 노이즈를 억제하고 혈압 추정에 유용한 특징을 강화한다. 그 다음 xLSTM은 PPG 시계열의 순차적 변화를 따라가며 장기 의존성을 학습한다. Transformer는 self-attention을 이용하여 시퀀스 내 임의의 두 위치 사이의 관계를 직접 학습한다. 마지막 FKAN은 kernel activation 기반 비선형 변환을 통해 최종 혈압 회귀 값을 산출한다.

기존 접근 방식과 비교할 때, 이 논문이 명확히 강조하는 차별점은 세 가지다. 첫째, CNN이나 LSTM 단독 모델이 아니라 DyCASNet, xLSTM, Transformer, FKAN을 결합한 다중 모듈 구조를 사용한다. 둘째, DyCASNet을 통해 시간 영역과 주파수 영역을 동시에 다루는 adaptive dual-domain feature extraction을 수행한다. 셋째, 평가 과정에서 subject-wise split을 사용하여 동일 피험자의 윈도우가 학습 데이터와 테스트 데이터에 동시에 들어가는 data leakage 문제를 줄이려 한다.

특히 DyCASNet은 논문에서 가장 중요한 모듈로 제시된다. Ablation study에 따르면 DyCASNet을 제거하면 MIMIC-III에서 SBP MAE가 2.21 mmHg에서 6.53 mmHg로 크게 증가한다. 이는 단순한 시계열 모델링보다 PPG의 시간-주파수 적응형 특징 추출이 성능 향상에 결정적인 역할을 했음을 시사한다.

## 3. 상세 방법 설명

### 3.1 전체 시스템 구조

ACFA의 전체 파이프라인은 다음과 같이 정리할 수 있다.

Raw PPG 신호가 먼저 전처리 과정을 거친다. 전처리된 PPG는 DyCASNet으로 입력되어 시간-주파수 특징으로 변환된다. 이후 xLSTM이 순차적 의존성을 학습하고, Transformer가 전체 시퀀스의 전역 관계를 학습한다. 마지막으로 FKAN이 비선형 회귀 변환을 수행하여 혈압 값을 출력한다.

전체 흐름은 다음과 같다.

$$
PPG \rightarrow DyCASNet \rightarrow xLSTM \rightarrow Transformer \rightarrow FKAN \rightarrow \hat{BP}
$$

논문에서는 최종 예측 대상으로 SBP, DBP, MAP을 모두 평가한다. 다만 방법 설명 중 FKAN 출력 수식에서는 $[\hat{SBP}, \hat{DBP}]$만 명시되어 있고 MAP 출력 방식은 명확하게 설명되어 있지 않다. MAP을 별도의 회귀 출력으로 예측했는지, 또는 SBP와 DBP로부터 계산했는지는 제공된 텍스트만으로는 확정할 수 없다. 이 점은 방법론 서술상의 불명확한 부분이다.

### 3.2 문제 정의

논문은 혈압 추정을 supervised learning 기반 회귀 문제로 정의한다. 입력 데이터는 PPG 신호이고, 정답 라벨은 혈압 값이다. 데이터셋은 다음과 같이 표현된다.

$$
S = {X, Y}
$$

여기서 $X = {x_1, x_2, \ldots, x_n} \in \mathbb{R}^{n \times d}$는 수집된 PPG 신호 시퀀스이고, $Y = {y_1, y_2, \ldots, y_n}$는 각 샘플에 대응되는 실제 혈압 값이다. $d$는 하나의 PPG 샘플이 가지는 특징 차원이다.

모델의 목표는 다음 함수를 학습하는 것이다.

$$
f:\mathbb{R}^{d} \rightarrow \mathbb{R}
$$

즉, 하나의 PPG 샘플로부터 혈압 값을 예측하는 함수 $f$를 학습한다. 실제 모델에서는 SBP, DBP, MAP처럼 여러 혈압 지표를 예측하므로, 개념적으로는 단일 출력 회귀 또는 다중 출력 회귀로 확장될 수 있다.

### 3.3 전처리 과정

논문은 PPG 신호 품질을 높이기 위해 비교적 표준적인 생체신호 전처리 과정을 적용한다. 먼저 비정상 혈압 라벨과 심하게 왜곡된 PPG 구간을 제거한다. 제거 기준은 SBP가 30 mmHg 미만 또는 220 mmHg 초과인 경우, DBP가 30 mmHg 미만 또는 150 mmHg 초과인 경우, 그리고 pulse pressure인 $PP = SBP - DBP$가 10 mmHg 미만인 경우이다. 이러한 기준은 생리적으로 비정상적이거나 측정 오류일 가능성이 큰 샘플을 제거하기 위한 것이다.

그 다음 4차 Butterworth band-pass filter를 사용한다. 주파수 범위는 0.5–10 Hz이다. 이 필터는 저주파 baseline drift와 고주파 잡음을 제거하기 위한 목적이다. 이후 db8 wavelet을 사용한 7-level discrete wavelet transform과 soft thresholding을 적용하여 파형의 주요 형태는 유지하면서 잡음을 줄인다. 신호 양끝 5%는 edge artifact를 줄이기 위해 잘라낸다. 마지막으로 길이를 정규화하고 평균 0, 분산 1이 되도록 standardization을 수행한다.

논문은 MIMIC-III에서 최종적으로 942명으로부터 4,710개의 고품질 PPG 샘플을 얻었다고 설명한다. 또한 PPG-BP 데이터셋에서는 219명으로부터 438개의 유효 샘플을 얻었다고 제시한다.

다만 텍스트 내에는 세그먼트 길이에 대한 서술이 완전히 일관되지는 않는다. 한 부분에서는 각 피험자에서 약 6초 길이의 신호 세그먼트 5개를 추출했다고 설명하지만, data split strategy 부분에서는 각 피험자의 raw PPG waveform을 10초 윈도우로 분할했다고 설명한다. 또한 base module에서는 최종 텐서가 $(N,L,F) = (4710,10,789)$라고 되어 있고, module integration strategy에서는 raw PPG signal을 789 sampling point로 구성된 1차원 시퀀스로 처리한다고 설명한다. 따라서 제공된 텍스트만으로는 실제 입력 길이와 윈도우 구성 방식이 완전히 명확하다고 보기 어렵다.

### 3.4 DyCASNet

DyCASNet은 ACFA에서 가장 중요한 특징 추출 모듈이다. 이름에서 알 수 있듯이 dynamic convolution과 channel-aware spectral processing을 결합하여 시간 영역과 주파수 영역 정보를 동시에 활용한다. DyCASNet은 크게 CASB(Channel-Aware Spectrum Block)와 DCB(Dynamic Convolution Block)로 구성된다.

CASB는 주파수 영역에서 중요한 성분을 추출하고 채널별 중요도를 조정한다. DCB는 시간 영역에서 multi-scale convolution을 수행하여 국소 패턴과 장기 패턴을 함께 반영한다. 이 두 구성 요소를 결합함으로써 PPG처럼 non-stationary하고 개인차가 큰 생체신호에 적응적으로 대응하려는 것이 DyCASNet의 설계 목적이다.

### 3.5 CASB: Channel-Aware Spectrum Block

CASB는 입력 시계열에 FFT(Fast Fourier Transform)를 적용하여 시간 영역 신호를 주파수 영역으로 변환한다. 입력 시계열을 $x[n]$이라고 하면, 주파수 표현은 다음과 같이 정의된다.

$$
F = \mathcal{F}[x] \in \mathbb{C}^{C \times L'}
$$

여기서 $C$는 채널 수, $L'$는 변환 후 주파수 차원을 의미한다. FFT를 사용하면 PPG 신호에 포함된 주기적 성분과 진동 패턴을 더 직접적으로 분석할 수 있다.

그 다음 power spectrum을 계산한다.

$$
P = |F|^2
$$

이 power spectrum은 각 주파수 성분이 얼마나 강한지를 나타낸다. 논문은 학습 가능한 threshold $\theta$를 사용하여 중요한 주파수 성분만 보존하는 adaptive frequency masking을 수행한다.

$$
F_{filtered} = F \odot (P > \theta)
$$

여기서 $\odot$는 element-wise multiplication이다. 이 수식의 의미는 단순하다. 파워가 threshold보다 큰 주파수 성분은 유지하고, 그렇지 않은 성분은 약화하거나 제거한다. 따라서 모델은 고정된 필터가 아니라 데이터에 따라 중요한 주파수 대역을 선택할 수 있다.

CASB는 여기에 learnable complex weights를 추가하여 전체 spectrum과 dominant frequency band를 모두 조정한다. 이 부분은 주파수 성분의 세기뿐 아니라 복소수 표현의 위상 및 진폭 관계를 조정할 수 있다는 점에서 일반적인 실수 기반 feature weighting보다 표현력이 높을 수 있다. 다만 제공된 텍스트에는 complex weights의 정확한 파라미터화 방식이나 구현 세부 사항은 충분히 제시되어 있지 않다.

이후 channel attention을 적용한다.

$$
F_{att} =
\sigma
\left(
W_2 \delta(W_1 \cdot AvgPool(F))
\right)
\odot F
$$

여기서 $\delta(\cdot)$는 ReLU, $\sigma(\cdot)$는 Sigmoid이다. 이 구조는 Squeeze-and-Excitation 계열 attention과 유사하다. 평균 pooling을 통해 각 채널의 요약 정보를 얻고, 두 개의 선형 변환과 비선형 함수를 거쳐 채널별 중요도를 계산한다. 중요한 채널은 강화하고 덜 중요한 채널은 억제한다.

마지막으로 IFFT(Inverse Fourier Transform)를 수행하여 다시 시간 영역 특징으로 변환한다.

$$
S' = \mathcal{F}^{-1}[F_{att}]
$$

결과적으로 CASB는 원래 PPG 신호에서 주파수적으로 의미 있는 성분을 강화한 시간 영역 특징 $S'$를 생성한다.

### 3.6 DCB: Dynamic Convolution Block

DCB는 CASB에서 얻은 향상된 시간 영역 특징 $S'$를 입력으로 받아 dynamic convolution을 수행한다. 일반적인 convolution layer는 학습 후 고정된 kernel을 모든 입력에 동일하게 적용한다. 반면 DCB는 여러 개의 convolution kernel을 준비하고, 입력에 따라 이 kernel들을 가중 결합한다.

논문에서는 $K$개의 convolution kernel을 다음과 같이 표현한다.

$$
W = {W_1, W_2, \ldots, W_K}
$$

DCB는 두 개의 병렬 convolution branch를 사용한다. 하나는 작은 receptive field를 사용하여 fine-grained local pattern을 포착하고, 다른 하나는 더 큰 receptive field를 사용하여 long-range dependency를 반영한다.

$$
A_1 = \phi(Conv_1(S'))
$$

$$
A_2 = \phi(Conv_2(S'))
$$

여기서 $\phi(\cdot)$는 GELU activation이다. 두 branch의 출력은 입력에 따라 학습된 동적 가중치 $\alpha_k$를 통해 결합된다.

$$
O_{fusion} =
\sum_{k=1}^{K}
\alpha_k
(A_1 \odot W_k + A_2 \odot W_k)
$$

이 수식은 입력 신호의 상태에 따라 convolution 조합이 달라진다는 의미를 가진다. 예를 들어 깨끗하고 주기성이 뚜렷한 PPG와 motion artifact가 포함된 PPG는 서로 다른 convolution 조합을 사용하는 것이 더 적합할 수 있다. DCB는 이러한 적응성을 제공한다.

이후 DCB에서도 channel attention을 적용한다.

$$
O_{att} =
\sigma
\left(
W_2 \delta(W_1 \cdot AvgPool(O_{fusion}))
\right)
\odot O_{fusion}
$$

최종 DCB 출력은 다음과 같다.

$$
O_{DCB} = O_{att}
$$

논문 텍스트에는 수식 (9)가 “$O_{fusion}O_{att}$”처럼 다소 어색하게 표기되어 있으나, 문맥상 fused feature에 attention weight를 곱하여 $O_{att}$를 생성하는 구조로 해석하는 것이 자연스럽다.

### 3.7 xLSTM

xLSTM은 기존 LSTM을 확장한 시계열 모델이다. 논문은 xLSTM이 sLSTM(scalar LSTM)과 mLSTM(matrix LSTM)을 결합한다고 설명한다. sLSTM은 causal one-dimensional convolution, gating mechanism, block-diagonal projection 등을 사용하여 순차적 특징 표현을 강화한다. mLSTM은 matrix memory update와 gated attention을 사용하여 고차 비선형 시계열 패턴을 모델링한다.

ACFA에서 xLSTM은 DyCASNet이 추출한 특징을 입력으로 받아 PPG의 시간적 의존성을 학습한다. 여러 cardiac cycle에 걸친 PPG 변화는 혈압 변화와 관련될 수 있으므로, 단기 파형만 보는 모델보다 장기 의존성을 처리하는 모델이 유리할 수 있다.

xLSTM layer 사이에는 residual connection과 normalization이 사용된다.

$$
H^{(l+1)} =
Norm
\left(
H^{(l)} + xLSTM(H^{(l)})
\right)
$$

여기서 $Norm(\cdot)$은 Layer Normalization과 Group Normalization을 포함한다고 설명된다. Residual connection은 깊은 네트워크에서 gradient vanishing 문제를 완화하고, normalization은 학습 안정성을 높인다.

xLSTM의 예측 관계는 다음과 같이 표현된다.

$$
\hat{y} = f_{xLSTM}(X), \quad X = (x_1, x_2, \ldots, x_T)
$$

이 수식은 xLSTM이 입력 시퀀스 $X$를 받아 예측값 $\hat{y}$를 생성한다는 의미다. ACFA 전체에서는 xLSTM이 최종 출력기를 담당한다기보다 Transformer와 FKAN으로 이어지는 중간 temporal representation을 제공하는 역할로 이해하는 것이 적절하다.

### 3.8 Transformer

Transformer branch는 self-attention을 이용하여 PPG 시퀀스의 전역 문맥 정보를 학습한다. RNN 계열 모델은 순차적으로 정보를 전달하므로 장기 의존성을 학습할 수 있지만, 먼 위치 사이의 관계가 여러 단계의 상태 전달을 거쳐야 한다. Transformer는 임의의 두 위치 사이 관계를 attention으로 직접 계산할 수 있으므로 장거리 관계를 더 직접적으로 모델링할 수 있다.

Self-attention의 핵심 수식은 다음과 같다.

$$
Attention(Q,K,V) =
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
V
$$

여기서 $Q$, $K$, $V$는 각각 query, key, value 행렬이고, $d_k$는 key vector의 차원이다. $QK^T$는 시퀀스 내 위치들 사이의 유사도를 계산한다. 이를 $\sqrt{d_k}$로 나누어 값의 스케일을 안정화한 뒤 softmax를 적용하면 각 위치가 다른 위치를 얼마나 참고해야 하는지에 대한 attention weight가 된다.

Transformer block 이후에는 feed-forward network가 적용된다.

$$
FFN(z) =
\phi(zW_1 + b_1)W_2 + b_2
$$

여기서 $\phi(\cdot)$는 비선형 activation function이다. 논문은 Transformer가 PPG 신호 내 서로 떨어진 구간들 사이의 전역 의존성을 포착하여 xLSTM의 국소적 temporal modeling을 보완한다고 설명한다.

### 3.9 FKAN

FKAN은 Fast Kernel Activation Network로 설명되며, 최종 비선형 회귀 변환을 담당한다. 논문은 FKAN이 RBF(Radial Basis Function) 기반 spline basis와 trainable linear weights를 결합하여 효율적인 nonlinear mapping을 수행한다고 설명한다.

FKAN layer의 주 경로는 다음 RBF 기반 함수를 사용한다.

$$
\phi(x) =
\sum_{i=1}^{M}
w_i
\exp
\left(
-\frac{(x-\mu_i)^2}{2\sigma_i^2}
\right)
$$

이 수식은 입력 $x$를 여러 개의 RBF basis로 표현한 뒤, 각 basis의 가중합으로 비선형 변환을 구성한다는 의미다. $\mu_i$는 각 basis의 중심, $\sigma_i$는 폭, $w_i$는 학습 가능한 가중치다. RBF는 특정 구간 주변에서 강하게 반응하므로, PPG 특징 공간의 국소적 비선형 구조를 표현하는 데 유용할 수 있다.

논문은 FKAN을 attention 구조 안에서도 사용한다고 설명한다. Query와 key에 FastKAN transformation을 적용한 뒤 kernel-based dot product로 attention coefficient를 계산한다.

$$
\alpha_{ij} =
\frac{
\phi(Q_i)^T \phi(K_j)
}
{\sqrt{d_k}}
$$

최종적으로 FKAN은 혈압 회귀 값을 출력한다.

$$
[\hat{SBP}, \hat{DBP}] =
FastKAN(X_{PPG})
$$

다만 앞서 언급했듯이, 논문 결과에서는 MAP도 평가하지만 이 수식에는 MAP 출력이 포함되어 있지 않다. 따라서 MAP을 별도 출력으로 예측했는지, 사후 계산했는지, 또는 다른 출력 head를 사용했는지는 제공된 텍스트만으로는 명확하지 않다.

### 3.10 학습 목표와 최적화

논문은 학습 손실 함수로 MAE(Mean Absolute Error)를 사용한다. MAE는 예측값과 실제값 사이 절대 오차의 평균이다.

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i - \hat{y}_i|
$$

MAE를 loss로 사용하면 큰 오차에 지나치게 민감한 MSE보다 outlier에 상대적으로 덜 민감한 장점이 있다. 혈압 데이터에는 측정 잡음이나 생리적 급변이 존재할 수 있으므로 MAE 기반 학습은 적절한 선택으로 볼 수 있다.

최적화에는 Adam optimizer를 사용하며, 초기 learning rate는 $3 \times 10^{-4}$, batch size는 128, 최대 epoch 또는 iteration은 100으로 제시되어 있다. 과적합 방지를 위해 early stopping을 적용한다. 하이퍼파라미터는 cross-validation과 grid search로 최적화했다고 설명되지만, 구체적인 탐색 범위는 제공된 텍스트에 명시되어 있지 않다.

## 4. 실험 및 결과

### 4.1 데이터셋

논문은 두 개의 공개 데이터셋을 사용한다.

첫 번째는 MIMIC-III Waveform Database Matched Subset이다. 이 데이터셋은 ICU 환자의 고해상도 생체신호를 포함하며, ECG, ABP, PPG, RESP 등의 다중 채널 waveform과 임상 정보를 포함한다. 논문은 이 중 PPG 신호와 ABP 기반 혈압 라벨을 사용한다. 최종적으로 942명의 피험자에서 각 5개 세그먼트를 추출하여 총 4,710개 샘플을 구성했다고 설명한다.

두 번째는 PPG-BP 데이터셋이다. 이 데이터셋은 20세부터 89세까지 다양한 피험자의 PPG와 임상 혈압 정보를 포함하며, 고혈압과 당뇨 같은 심혈관 위험 요인을 포함한다. 논문은 품질 검사를 거쳐 219명으로부터 438개의 유효 샘플을 사용했다고 제시한다. MIMIC-III가 ICU 환경에 가깝다면, PPG-BP는 상대적으로 일반적인 생리 모니터링 환경을 반영한다고 설명된다.

### 4.2 데이터 분할 전략

논문은 subject-wise split을 엄격히 사용했다고 강조한다. 이는 같은 피험자의 PPG 윈도우가 학습, 검증, 테스트 세트에 동시에 들어가지 않도록 하는 방식이다. 피험자 단위로 70%/10%/20% 비율의 training, validation, test set을 구성한다.

이 선택은 혈압 추정 연구에서 매우 중요하다. PPG 기반 혈압 예측에서는 개인별 파형 특성이 강하게 나타난다. 만약 window-wise random split을 사용하면 같은 사람의 서로 다른 윈도우가 학습과 테스트에 동시에 포함될 수 있다. 이 경우 모델은 혈압과 관련된 일반화 가능한 생리적 규칙을 학습했다기보다 특정 피험자의 패턴을 기억할 가능성이 있다. 따라서 테스트 성능이 실제보다 과대평가될 수 있다.

논문은 window-wise random split을 control experiment로 함께 보고하여, 이 방식이 더 낮은 오차를 보이지만 data leakage 위험으로 인해 실제 임상 적용 성능을 과대평가할 수 있다고 설명한다. 이 문제를 명시적으로 다룬 점은 논문의 중요한 강점이다.

### 4.3 평가 지표

논문은 MAE, ME, STD를 주요 지표로 사용한다. 각 샘플의 오차는 다음과 같이 정의된다.

$$
\Delta y_i = y_i - \hat{y}_i
$$

MAE는 평균 절대 오차다.

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|\Delta y_i|
$$

ME(Mean Error)는 평균 오차로, 예측이 전체적으로 높게 치우치는지 낮게 치우치는지를 나타낸다.

$$
ME =
\frac{1}{n}
\sum_{i=1}^{n}
\Delta y_i
$$

STD는 오차의 표준편차로, 모델 예측의 안정성을 나타낸다.

$$
STD =
\sqrt{
\frac{1}{n-1}
\sum_{i=1}^{n}
(\Delta y_i - ME)^2
}
$$

즉 MAE가 낮으면 평균적으로 정확하다는 뜻이고, ME가 0에 가까우면 systematic bias가 작다는 뜻이며, STD가 낮으면 오차 변동성이 작다는 뜻이다.

### 4.4 BHS 및 AAMI 기준

논문은 임상적 타당성을 평가하기 위해 BHS(British Hypertension Society)와 AAMI(Association for the Advancement of Medical Instrumentation) 기준을 사용한다.

BHS 기준은 절대 오차가 5 mmHg, 10 mmHg, 15 mmHg 이내에 들어오는 비율을 기반으로 등급을 부여한다. 논문은 SBP, DBP, MAP 모두 BHS Grade A를 달성했다고 보고한다.

AAMI 기준은 평균 오차와 표준편차를 함께 본다. 논문에서 설명한 기준은 평균 편차가 5 mmHg 이하이고 표준편차가 8 mmHg 이하인 경우 임상적으로 수용 가능하다는 것이다. ACFA는 SBP, DBP, MAP 모두에서 이 조건을 만족한다고 보고한다.

### 4.5 주요 정량 결과

MIMIC-III에서 ACFA의 주요 성능은 다음과 같다.

| 지표 |       MAE |
| ---- | --------: |
| SBP  | 2.21 mmHg |
| DBP  | 3.48 mmHg |
| MAP  | 3.64 mmHg |

PPG-BP에서 ACFA의 주요 성능은 다음과 같다.

| 지표 |       MAE |
| ---- | --------: |
| SBP  | 2.49 mmHg |
| DBP  | 3.16 mmHg |
| MAP  | 3.75 mmHg |

이 결과는 논문이 주장하는 핵심 성과다. 특히 subject-wise split을 사용했음에도 SBP MAE가 2–3 mmHg 수준이라는 점은 매우 높은 정확도에 해당한다. 다만 이 수치가 실제 임상 환경에서 그대로 재현될지는 별도 외부 검증이 필요하다.

### 4.6 회귀 분석과 Bland–Altman 분석

논문은 예측값과 참값 사이의 선형 관계를 확인하기 위해 regression plot을 사용한다. 보고된 결정계수는 다음과 같다.

| 지표 | $R^2$ |
| ---- | ----: |
| SBP  |  0.93 |
| DBP  |  0.93 |
| MAP  |  0.94 |

이는 모델 예측값이 참조 혈압값과 강한 양의 선형 관계를 가진다는 것을 의미한다.

또한 Bland–Altman 분석을 통해 예측값과 참조값 사이의 agreement를 평가한다. 논문은 SBP, DBP, MAP 모두에서 평균 차이가 0에 가깝고, 대부분의 샘플이 95% limits of agreement 안에 들어간다고 설명한다. 또한 평균 혈압이 커질수록 오차 분산이 커지는 뚜렷한 heteroscedasticity는 관찰되지 않았다고 보고한다. 일부 outlier는 motion artifact, baseline drift, 개인별 혈관 기능 차이, 긴장 상태, 약물 영향 등으로 설명될 수 있다고 서술한다.

### 4.7 기존 방법과의 비교

논문은 PAT+Adaboost, RNN, CNN, ANN, CNN+LSTM, CNN+Transformer, LSTM+KAN, BiLSTM+KAN, XLSTM-SENet, CNN+BiLSTM+Attention, Informer, TimesNet 등 다양한 방법과 비교한다.

초기 방법인 PAT+Adaboost와 RNN은 MIMIC-III에서 SBP MAE가 각각 11.17 mmHg와 12.08 mmHg로 높게 나타난다. CNN과 ANN은 오차를 줄이지만 여전히 충분하지 않다. CNN+LSTM과 CNN+Transformer는 시간적 특징을 더 잘 반영하여 SBP MAE를 약 3.64–4.44 mmHg 수준까지 낮춘다. 그러나 논문은 이들 방법의 STD가 여전히 만족스럽지 않다고 설명한다.

CNN+BiLSTM+Attention은 SBP MAE 1.97 mmHg, DBP MAE 2.25 mmHg로 매우 강한 비교 모델로 제시된다. 흥미롭게도 이 수치만 보면 SBP와 DBP MAE는 ACFA보다 더 낮거나 유사하게 보일 수 있다. 그러나 논문은 전체 지표, MAP, STD, split strategy, 데이터셋 조건 등을 고려할 때 ACFA가 더 우수하다고 주장한다. 제공된 텍스트만으로는 모든 비교 모델이 동일한 subject-wise split, 동일한 전처리, 동일한 데이터 샘플에서 공정하게 재구현되었는지 완전히 확인하기 어렵다. 따라서 이 비교 결과는 논문 내부 기준에서는 유효하지만, 외부 재현성 관점에서는 추가 검증이 필요하다.

Informer와 TimesNet도 비교에 포함된다. Informer는 sparse self-attention을 기반으로 긴 시계열 예측에 강점을 가지지만, PPG의 미세한 생리적 변동과 개인차를 포착하는 데 한계가 있다고 해석된다. TimesNet은 다중 주기 temporal variation modeling을 활용하지만, 주파수 영역 특징과 cross-subject variability를 충분히 반영하지 못한다고 논문은 설명한다. 두 모델 모두 subject-wise split에서 성능이 저하되는 것으로 보고된다.

### 4.8 Ablation Study

Ablation study는 ACFA의 각 모듈이 성능에 얼마나 기여하는지 평가한다. 완전한 모델은 DyCASNet, xLSTM, Transformer, FKAN을 모두 포함한다. 하나씩 제거하며 성능 변화를 확인한다.

가장 중요한 결과는 DyCASNet 제거 시 성능이 크게 악화된다는 것이다. SBP MAE는 2.21 mmHg에서 6.53 mmHg로 증가한다. 이는 DyCASNet이 단순 보조 모듈이 아니라 핵심 feature extractor임을 강하게 보여준다.

Transformer 또는 xLSTM을 제거해도 SBP, DBP, MAP의 MAE가 유의미하게 증가한다고 보고된다. 이는 xLSTM과 Transformer가 서로 보완적인 역할을 한다는 논문의 주장을 뒷받침한다. xLSTM은 국소적 temporal dynamics와 장기 상태 변화를 처리하고, Transformer는 전체 시퀀스의 global context를 학습한다.

FKAN 제거의 효과는 논문 내 서술에서 다소 미묘하다. 한 부분에서는 FKAN 제거 시 평균 오차 증가는 작지만 MAP error fluctuation이 커진다고 설명한다. 다른 부분에서는 FKAN 제거가 SBP MAE를 1.21 mmHg, 약 55% 증가시켜 성능 저하가 상당하다고 설명한다. 따라서 FKAN의 기여를 “주로 안정성 향상”으로 볼지, “정확도와 안정성 모두에 중요한 기여”로 볼지는 표 7의 정확한 수치가 필요하다. 제공된 텍스트만으로는 두 설명 사이의 강도를 완전히 조정하기 어렵지만, FKAN이 regression output의 비선형성과 안정성에 기여한다는 결론은 논문 내에서 일관되게 제시된다.

## 5. 강점, 한계

### 5.1 강점

이 논문의 가장 큰 강점은 PPG 기반 혈압 추정 문제를 단일 시계열 회귀 문제로 단순화하지 않고, 시간 영역, 주파수 영역, 국소 패턴, 전역 문맥, 비선형 회귀라는 여러 관점에서 통합적으로 접근했다는 점이다. 특히 DyCASNet은 FFT 기반 frequency-domain analysis와 dynamic convolution을 결합하여 PPG의 non-stationary 특성을 처리하려는 명확한 설계 의도를 가진다.

두 번째 강점은 subject-wise split을 강조했다는 점이다. PPG 혈압 추정 연구에서는 window-wise random split으로 인한 data leakage가 매우 큰 문제다. 동일 피험자의 신호가 학습과 테스트에 동시에 존재하면 모델이 개인별 특징을 암기할 수 있기 때문이다. 이 논문은 이러한 문제를 인식하고 subject-wise partitioning을 사용했으며, window-wise split과의 차이도 비교했다. 이는 실제 임상 적용 가능성을 평가하는 데 중요한 설계다.

세 번째 강점은 BHS와 AAMI 기준을 모두 사용하여 단순한 MAE 비교를 넘어 임상적 수용 가능성까지 평가했다는 점이다. 의료 신호 예측에서는 평균 오차뿐 아니라 오차 분포, systematic bias, 표준편차가 중요하다. 논문은 MAE, ME, STD, BHS, AAMI, Bland–Altman, regression plot을 함께 사용하여 모델 성능을 다각도로 평가한다.

네 번째 강점은 두 개의 서로 다른 성격의 데이터셋을 사용했다는 점이다. MIMIC-III는 ICU 환경의 고해상도 waveform 데이터이고, PPG-BP는 상대적으로 일반적인 측정 환경을 반영한다. 두 데이터셋에서 모두 성능을 보고했다는 점은 단일 데이터셋 실험보다 강한 근거를 제공한다.

### 5.2 한계

첫 번째 한계는 입력 구성과 세그먼트 길이 설명이 다소 불명확하다는 점이다. 본문에는 약 6초 세그먼트, 10초 윈도우, 789 sampling point, $(4710,10,789)$ 텐서라는 설명이 함께 등장한다. 이 값들이 정확히 어떤 관계인지, 예를 들어 $L=10$이 time step인지 segment count인지, $F=789$가 sampling point인지 feature dimension인지 명확히 정리되어 있지 않다. 재현성을 위해서는 입력 shape와 segmentation rule을 더 엄밀하게 설명할 필요가 있다.

두 번째 한계는 MAP 예측 방식이 명확하지 않다는 점이다. 논문은 결과에서 MAP을 중요한 평가 지표로 제시하지만, FKAN의 최종 출력 수식은 $[\hat{SBP}, \hat{DBP}]$만 포함한다. MAP을 직접 예측했는지, SBP와 DBP에서 계산했는지, 또는 별도 head를 사용했는지가 명확하지 않다. 혈압 추정 모델의 출력 정의는 실험 해석에 중요하므로 이 부분은 보완이 필요하다.

세 번째 한계는 모델 구조가 복잡하다는 점이다. ACFA는 DyCASNet, xLSTM, Transformer, FKAN을 모두 결합한다. 이 조합은 성능 측면에서는 강력할 수 있지만, 각 구성 요소가 실제로 얼마나 독립적으로 필요한지 완전히 분리하기 어렵다. Ablation study가 존재하지만, DyCASNet 내부의 CASB와 DCB 각각의 독립 기여, xLSTM과 Transformer의 병렬 또는 순차 연결 방식의 영향, FKAN과 일반 MLP head의 차이 등은 더 세밀하게 분석될 필요가 있다.

네 번째 한계는 실제 웨어러블 디바이스 배포 검증이 없다는 점이다. 논문은 ACFA가 약 4.6M parameters, 3.4 GFLOPs, NVIDIA V100 GPU에서 sample당 약 6.8 ms inference time을 가진다고 보고한다. 그러나 이 결과는 서버급 GPU 환경에서 측정된 것이다. 스마트워치, 패치형 센서, 저전력 MCU, 모바일 NPU 등 실제 wearable deployment 환경에서의 latency, memory footprint, battery consumption은 제시되지 않았다.

다섯 번째 한계는 데이터 규모와 다양성이다. MIMIC-III에서는 942명, PPG-BP에서는 219명을 사용했다고 보고되지만, 실제 임상 적용을 위해서는 훨씬 더 다양한 인구 집단, 피부색, 연령, 질환 상태, 측정 위치, 센서 종류, 움직임 조건에서 검증이 필요하다. 논문 역시 향후 연구로 더 큰 규모와 다양한 population에서 검증이 필요하다고 언급한다.

여섯 번째 한계는 일부 비교 실험의 공정성 확인이 어렵다는 점이다. 논문은 많은 baseline을 제시하지만, 제공된 텍스트만으로는 모든 baseline이 동일한 데이터 전처리, 동일한 subject-wise split, 동일한 hyperparameter tuning, 동일한 test set에서 재학습되었는지 확인하기 어렵다. 특히 기존 문헌의 결과를 그대로 가져온 경우와 자체 재구현 결과가 섞이면 공정한 비교가 어려울 수 있다. 이 부분은 원문 표와 실험 설정을 추가로 확인해야 정확히 판단할 수 있다.

### 5.3 비판적 해석

이 논문은 새로운 이론적 학습 원리를 제안했다기보다는, PPG 혈압 추정 문제에 적합한 여러 최신 딥러닝 구성 요소를 조합한 강한 engineering-oriented architecture로 볼 수 있다. DyCASNet은 이 논문에서 가장 독창적인 부분이며, 주파수 영역 masking, complex weighting, channel attention, dynamic convolution을 결합하여 PPG 신호의 생리적 패턴을 더 잘 반영하려는 시도는 설득력이 있다.

다만 ACFA의 매우 높은 성능 수치가 실제 임상 환경에서 그대로 유지될지는 신중하게 봐야 한다. PPG 기반 cuffless BP estimation은 센서 위치, 움직임, 피부 특성, 말초 혈관 상태, 체온, 약물, 질환 상태에 크게 영향을 받는다. 논문은 subject-wise split을 사용했다는 점에서 이전 연구보다 엄격하지만, 실제 제품 또는 임상 환경에서는 dataset shift가 더 크게 발생할 수 있다. 따라서 ACFA는 유망한 연구 결과이지만, 의료기기 수준의 신뢰성을 주장하려면 다기관 전향적 검증, 장기간 모니터링, calibration drift 분석, 실제 웨어러블 기기 실험이 추가로 필요하다.

또한 모델의 설명가능성 측면에서도 보완이 필요하다. CASB가 어떤 주파수 대역을 강조하는지, attention이 어떤 PPG 구간을 중요하게 보는지, FKAN이 어떤 비선형 관계를 학습하는지에 대한 생리학적 해석이 더 제공된다면 임상적 설득력이 높아질 것이다.

## 6. 결론

이 논문은 PPG 기반 커프리스 연속 혈압 추정을 위해 ACFA라는 하이브리드 딥러닝 프레임워크를 제안하였다. ACFA는 DyCASNet, xLSTM, Transformer, FKAN을 결합하여 PPG 신호의 시간 영역 특징, 주파수 영역 특징, 장기 시계열 의존성, 전역 문맥 정보, 복잡한 비선형 회귀 관계를 통합적으로 학습한다.

MIMIC-III 데이터셋에서 ACFA는 SBP, DBP, MAP에 대해 각각 2.21 mmHg, 3.48 mmHg, 3.64 mmHg의 MAE를 달성했다고 보고된다. PPG-BP 데이터셋에서는 각각 2.49 mmHg, 3.16 mmHg, 3.75 mmHg의 MAE를 달성했다. 또한 BHS Grade A와 AAMI 기준을 만족한다고 제시된다. Ablation study에서는 DyCASNet이 가장 중요한 성능 기여 요소로 나타나며, xLSTM과 Transformer는 각각 국소적 temporal dynamics와 global context modeling 측면에서 상호 보완적인 역할을 한다. FKAN은 최종 회귀의 비선형 표현력과 예측 안정성을 높이는 역할을 한다.

이 연구의 실제 적용 가능성은 높다. PPG는 이미 스마트워치, 스마트밴드, 패치형 센서 등에 널리 탑재되어 있으므로, 고성능 cuffless BP estimation 모델은 웨어러블 헬스케어와 원격 모니터링에 중요한 기반 기술이 될 수 있다. 특히 고혈압 조기 발견, 야간 혈압 변화 추적, 약물 반응 모니터링, 심혈관 위험 예측에 활용될 가능성이 있다.

그러나 실제 임상 및 제품 적용을 위해서는 입력 구성의 재현성 개선, MAP 출력 방식 명확화, 다양한 외부 데이터셋 검증, 실제 웨어러블 기기에서의 경량화 및 배포 실험, 장기적 calibration stability 평가가 필요하다. 종합적으로 볼 때, ACFA는 PPG 기반 연속 혈압 추정 분야에서 시간-주파수 적응형 특징 추출과 장기 시계열 모델링을 결합한 강력한 하이브리드 접근법이며, 후속 연구와 임상 검증을 통해 실용적 가치가 더 명확해질 수 있는 연구로 평가된다.
