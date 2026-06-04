# A paralleled CNN and Transformer network for PPG-based cuff-less blood pressure estimation

* **저자**: Zhonghe Tian, Aiping Liu, Guokang Zhu, Xun Chen
* **발표연도**: 2025

## 1. 논문 개요

이 논문은 PPG(Photoplethysmography) 신호만을 이용하여 커프 없이 혈압을 추정하는 딥러닝 모델인 **PCTN(Parallel Convolutional Transformer Network)** 을 제안한다. 논문은 *Biomedical Signal Processing and Control* 99권에 2025년 논문 번호 106741로 게재되었으며, DOI는 10.1016/j.bspc.2024.106741이다.

논문의 목표는 단일 PPG 신호로부터 DBP(Diastolic Blood Pressure), MAP(Mean Arterial Pressure), SBP(Systolic Blood Pressure)를 정확하게 추정하는 것이다. 혈압은 심혈관 건강 상태를 평가하는 핵심 지표이며, 고혈압은 장기간 관리되지 않을 경우 심장, 뇌, 신장 질환의 위험을 높인다. 그러나 기존 커프 기반 혈압 측정 방식은 불편하고 연속 측정에 적합하지 않다. 따라서 웨어러블 기기에서 쉽게 얻을 수 있는 PPG 신호를 활용한 cuff-less BP estimation은 임상적·실용적 가치가 크다.

이 연구가 다루는 핵심 문제는 기존 PPG 기반 딥러닝 방법들이 주로 국소 특징(local feature) 또는 시간적 특징(temporal feature)에만 집중하고, PPG 시계열 전체에 걸친 전역 특징(global feature)을 충분히 활용하지 못한다는 점이다. CNN은 국소 파형 패턴을 잘 추출하지만 receptive field가 제한적이다. RNN, LSTM, GRU 계열 모델은 시계열 처리에 적합하지만 장기 의존성 학습과 병렬 연산 효율성에 한계가 있다. 저자들은 이러한 문제를 해결하기 위해 CNN과 Transformer를 병렬로 배치하여 PPG의 국소 특징과 전역 특징을 동시에 학습하는 구조를 제안한다.

논문에서 제안한 PCTN은 Stem module, Feature extraction module, Feature fusion module, Regressor module로 구성된다. Feature extraction module은 CNN branch와 Transformer branch로 나뉘며, 각각 local feature와 global feature를 추출한다. Feature fusion module은 spatial attention과 channel attention을 이용해 두 종류의 특징을 융합한다. 마지막으로 regression module은 두 개의 fully connected layer를 통해 SBP와 DBP를 예측한다. 실험에서는 MAP도 평가하지만, 방법 설명에서는 최종 regressor가 SBP와 DBP를 출력한다고 설명되어 있어 MAP을 직접 예측했는지, ABP segment에서 계산한 MAP label을 별도 출력으로 예측했는지, 또는 SBP와 DBP로부터 계산했는지는 제공된 텍스트만으로는 명확하지 않다.

## 2. 핵심 아이디어

논문의 핵심 아이디어는 PPG 기반 혈압 추정에서 **local feature와 global feature를 모두 활용해야 한다**는 것이다. PPG 신호에는 한 심박 주기 안에서 나타나는 peak, valley, slope, waveform morphology와 같은 국소적 형태 정보가 포함되어 있다. 동시에 여러 시점에 걸친 파형 변화, 장기적 의존성, 시퀀스 내 멀리 떨어진 구간 간의 관계도 혈압 추정에 중요하다.

CNN은 작은 convolution kernel을 쌓아 파형의 국소 구조를 추출하는 데 강하다. 예를 들어 PPG의 상승 구간, 하강 구간, peak 주변의 모양, 특정 위치의 미세한 진폭 변화 등을 잘 포착할 수 있다. 그러나 CNN은 기본적으로 제한된 receptive field를 가지므로 긴 구간 전체의 상호작용을 직접 모델링하기 어렵다.

Transformer는 self-attention을 통해 시퀀스 내 모든 위치 사이의 관계를 계산할 수 있다. 따라서 PPG 신호의 특정 구간이 다른 먼 구간과 어떤 관계를 갖는지, 전체 시계열 패턴이 혈압과 어떻게 연결되는지를 학습하는 데 유리하다. 또한 Transformer는 RNN과 달리 병렬 계산이 가능하므로 긴 시계열 학습에서 효율적이다.

PCTN의 설계 직관은 다음과 같다.

$$
PPG \rightarrow \text{CNN branch for local features} + \text{Transformer branch for global features} \rightarrow \text{attention-based fusion} \rightarrow BP
$$

기존 연구와 비교했을 때, 이 논문의 차별점은 세 가지다. 첫째, PPG 기반 cuff-less BP estimation에 Transformer를 통합하여 global feature learning을 명시적으로 도입한다. 둘째, CNN과 Transformer를 순차적으로 연결하는 것이 아니라 병렬 구조로 배치하여 서로 다른 성격의 특징을 독립적으로 추출한다. 셋째, 단순 concatenation이 아니라 spatial attention과 channel attention을 이용하여 두 branch의 특징을 선택적으로 융합한다.

## 3. 상세 방법 설명

### 3.1 문제 정의

논문은 혈압 추정을 supervised regression 문제로 정의한다. 라벨이 있는 데이터셋은 다음과 같이 표현된다.

$$
S = {X, Y}
$$

여기서 $X = {x_1, \ldots, x_n} \in \mathbb{R}^{n \times d}$는 raw PPG data이고, $Y = {y_1, \ldots, y_n}$는 대응되는 BP values이다. $d$는 raw PPG segment의 차원이다. 모델의 목표는 학습 데이터 $S$를 이용해 새로운 PPG 데이터의 혈압 값을 정확히 예측하는 것이다.

이 논문에서 입력 PPG segment의 길이는 1024 samples로 설정된다. MIMIC 데이터에서 PPG와 ABP의 sampling rate는 125 Hz이므로, 하나의 segment 길이는 다음과 같다.

$$
\frac{1024}{125} = 8.192 \text{ seconds}
$$

즉 모델은 약 8.192초 길이의 PPG 신호 구간을 입력으로 받아 혈압을 추정한다.

### 3.2 전체 아키텍처

PCTN은 크게 네 부분으로 구성된다.

첫째, **Stem module**은 입력 PPG를 필터링하고 shallow feature를 미리 추출한다. 둘째, **Feature extraction module**은 CNN branch와 Transformer branch로 나뉜다. CNN branch는 local feature를 추출하고, Transformer branch는 global feature를 추출한다. 셋째, **Feature fusion module**은 spatial attention과 channel attention을 사용하여 두 branch의 특징을 융합한다. 넷째, **Regressor module**은 융합된 특징을 이용하여 BP 값을 회귀한다.

전체 흐름은 다음과 같이 정리할 수 있다.

$$
X_{PPG}
\rightarrow
\text{Stem}
\rightarrow
(\text{CNN branch}, \text{Transformer branch})
\rightarrow
\text{Fusion block}
\rightarrow
\text{Regressor}
\rightarrow
\hat{BP}
$$

논문 설명에 따르면 regressor는 SBP와 DBP를 출력한다. 그러나 결과 표에서는 DBP, MAP, SBP를 모두 평가한다. MAP의 생성 방식은 제공된 텍스트에서 명확히 설명되지 않는다. 이는 재현성 관점에서 보완이 필요한 부분이다.

### 3.3 Stem module

Stem module은 PPG 신호 입력, band-pass filtering, feature pre-extraction으로 구성된다. 입력은 길이 $L$의 PPG segment이며, 논문에서는 예시로 $L = 1024$를 사용한다.

먼저 PPG 신호에 0.5–10 Hz band-pass filter를 적용한다. 이 대역은 PPG에서 심박 관련 파형 정보를 유지하면서 baseline drift와 고주파 잡음을 줄이기 위한 선택으로 볼 수 있다. 이후 feature pre-extraction 단계에서는 1D convolution, 1D Batch Normalization, 1D Max Pooling을 사용한다.

Stem module에서 large convolution kernel을 사용하면 receptive field를 넓히면서 원본 신호 정보를 더 많이 보존할 수 있다고 논문은 설명한다. 즉 이 단계는 깊은 CNN branch와 Transformer branch에 들어가기 전, PPG의 기본적인 waveform representation을 만드는 역할을 한다.

### 3.4 CNN branch

CNN branch는 PPG의 local feature를 추출하기 위한 부분이다. 이 branch는 pyramid structure를 사용한다. 네트워크가 깊어질수록 channel 수는 증가하고 feature map의 크기는 감소한다. 이는 일반적인 CNN backbone에서 자주 사용되는 설계로, 낮은 층에서는 세밀한 위치 정보를 추출하고 높은 층에서는 더 추상적인 특징을 학습한다.

각 stage는 여러 개의 Conv Block으로 구성되며, 각 Conv Block은 $n$개의 bottleneck을 포함한다. 논문은 ResNet에서 정의된 bottleneck 구조를 따른다고 설명한다. 하나의 bottleneck은 다음 세 부분으로 구성된다.

첫째, kernel size 1의 1D under-projection convolution을 사용하여 channel dimension을 줄인다. 둘째, kernel size 3의 1D spatial convolution을 사용하여 국소 시계열 패턴을 추출한다. 셋째, kernel size 1의 1D up-projection convolution을 사용하여 channel dimension을 다시 확장한다. 입력과 출력 사이에는 residual connection이 존재한다.

이 구조의 의미는 단순하다. PPG의 특정 시간 구간 주변에서 나타나는 파형 모양, peak 주변의 변화, 상승·하강 기울기, 짧은 시간 내 amplitude pattern 등을 CNN이 집중적으로 학습한다. 혈압은 PPG의 국소 morphology와 밀접하게 관련되므로 CNN branch는 중요한 역할을 한다.

### 3.5 Transformer branch

Transformer branch는 PPG의 global feature를 학습하기 위한 부분이다. 이 branch는 여러 개의 Transformer block으로 구성된다. 각 Transformer block은 embedding module, Layer Normalization, Multi-head Self-Attention, MLP, residual connection을 포함한다.

Transformer의 핵심은 self-attention이다. 일반적으로 self-attention은 다음과 같이 표현된다.

$$
Attention(Q,K,V) = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

여기서 $Q$, $K$, $V$는 입력 특징으로부터 만들어진 query, key, value 행렬이고, $d_k$는 key vector의 차원이다. $QK^T$는 시퀀스 내 각 위치가 다른 위치와 얼마나 관련되는지를 나타낸다. 이를 $\sqrt{d_k}$로 나누어 스케일을 안정화하고 softmax를 적용하면 attention weight가 된다. 이 weight를 value에 곱하여 각 위치가 전체 시퀀스의 정보를 반영하도록 만든다.

PPG 신호에서 Transformer branch는 특정 시간 지점의 파형이 다른 시간 지점의 파형과 어떤 관계를 가지는지 학습할 수 있다. 예를 들어 한 segment 안의 여러 pulse cycle 간 형태 변화, 긴 구간에 걸친 amplitude 변화, 반복적 또는 비반복적 패턴을 포착할 수 있다.

논문에서 Transformer encoder의 key parameter로 head 수는 4, depth는 6으로 설정된다. 따라서 Transformer branch는 multi-head self-attention을 통해 여러 attention subspace에서 PPG의 global dependency를 분석한다.

### 3.6 Fusion block

Fusion block은 PCTN의 중요한 설계 요소다. CNN branch와 Transformer branch는 서로 다른 종류의 특징을 생성한다. CNN feature는 local waveform morphology에 강하고, Transformer feature는 long-range dependency와 global context에 강하다. 이 둘을 단순히 합치면 서로 다른 특성이 섞이면서 중요한 정보가 희석될 수 있다. 따라서 저자들은 attention 기반 feature fusion을 사용한다.

논문은 CBAM(Convolutional Block Attention Module)에서 영감을 받았다고 설명한다. CBAM은 channel attention과 spatial attention을 결합하여 중요한 feature를 강조하고 덜 중요한 feature를 억제한다.

PCTN의 fusion block은 먼저 local feature와 global feature 각각에 spatial attention을 개별적으로 적용한다. 저자들은 두 특징이 서로 다른 종류이므로 spatial attention을 각각 적용하여 섞이기 전에 위치 정보를 보존해야 한다고 설명한다. 이후 두 feature를 concatenation하고 channel attention을 적용한다.

중요한 차이점은 channel attention에서 일반적인 fully connected layer 대신 kernel size 1의 1D convolution을 사용한다는 것이다. 논문은 이 방식이 더 나은 local abstraction, 더 작은 parameter space, 더 낮은 overfitting 가능성을 제공한다고 설명한다.

Fusion block의 직관은 다음과 같다. 먼저 각 branch 안에서 중요한 시간 위치 또는 feature 위치를 고른다. 그 다음 두 branch의 feature를 합친 뒤, 혈압 예측에 중요한 channel을 선택적으로 강조한다. 따라서 fusion block은 local feature와 global feature의 상호 보완성을 살리는 역할을 한다.

### 3.7 Regressor module

Regressor module은 fusion block에서 얻은 feature를 혈압 값으로 변환한다. 이 모듈은 stride 2의 1D Global Pooling과 두 개의 fully connected layer로 구성된다. Global Pooling은 feature map 전체를 요약하여 고정 길이 표현을 만든다. 이후 fully connected layers가 이 표현을 더 분리 가능한 공간으로 매핑하고 최종 혈압 값을 예측한다.

논문 설명에서는 regressor가 SBP와 DBP를 계산한다고 되어 있다. MAP 결과는 실험에서 함께 보고되지만, MAP이 regressor의 세 번째 출력인지, ABP로부터 계산된 label을 별도로 예측한 것인지, 또는 SBP와 DBP를 이용해 사후 계산한 것인지는 명확하지 않다.

### 3.8 손실 함수와 학습 절차

모델 학습에는 MAE(Mean Absolute Error)를 loss function으로 사용한다. 예측값을 $\hat{Y} = [\hat{y}_1,\ldots,\hat{y}_n]$, 실제값을 $Y = [y_1,\ldots,y_n]$라고 할 때, 각 샘플의 오차는 다음과 같다.

$$
\Delta y_i = y_i - \hat{y}_i
$$

ME는 다음과 같이 정의된다.

$$
ME =
\frac{1}{n}
\sum_{i=1}^{n}
\Delta y_i
$$

MAE는 다음과 같이 정의된다.

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|\Delta y_i|
$$

STD는 오차의 표준편차이며, 논문 수식의 의도는 다음과 같이 해석할 수 있다.

$$
STD =
\sqrt{
\frac{1}{n-1}
\sum_{i=1}^{n}
(\Delta y_i - \overline{\Delta y})^2
}
$$

여기서 $\overline{\Delta y}$는 평균 오차다. 논문 수식에는 $\bar{\Delta y_i}$처럼 표기가 다소 혼동될 수 있는 형태로 제시되어 있지만, 문맥상 일반적인 error standard deviation으로 해석하는 것이 타당하다.

학습 환경은 Python과 PyTorch 기반이다. 하드웨어는 12-core CPU, 32 GB RAM, NVIDIA RTX3090Ti GPU이다. Optimizer는 Adam을 사용하고, initial learning rate는 0.01, 학습 epoch는 100이다. 각 convolution layer의 filter dimension은 1로 설정되며, CNN backbone은 ResNet-50을 사용한다.

## 4. 실험 및 결과

### 4.1 데이터셋

실험은 MIMIC-III waveform database를 사용한다. 이 데이터셋은 MIT에서 제공하는 ICU 환자 생체신호 데이터베이스이며, ECG, PPG, ABP 등 다양한 waveform을 포함한다. 저자들은 1000명의 subject pool에서 데이터를 수집한 뒤, PPG와 ABP가 짝지어지지 않은 경우, 신호 품질이 낮은 경우, signal duration이 10분 미만인 경우를 제외하였다. 최종적으로 808명의 subject가 사용되었다.

PPG와 ABP의 sampling rate는 모두 125 Hz이다. Segment length는 1024 samples이며, 이는 8.192초에 해당한다. SBP와 DBP label은 각 segment의 ABP signal에서 peak와 valley를 추출하고, 그 평균값을 segment의 SBP와 DBP로 사용하는 방식으로 생성된다. 즉 PPG를 입력으로 사용하지만, 혈압 참값은 ABP waveform에서 추출된다.

MIMIC 데이터의 혈압 통계는 다음과 같이 제시된다.

| 지표 | 최소 | 최대 | 평균 | 표준편차 |
| ---- | ---: | ---: | ---: | -------: |
| DBP  |   50 |  120 |   63 |    10.14 |
| MAP  |   60 |  135 |   84 |    11.56 |
| SBP  |   80 |  180 |  128 |    21.16 |

이 통계에서 SBP의 표준편차가 가장 크다. 이는 SBP가 DBP나 MAP보다 변동성이 크고 예측이 어려울 수 있음을 시사한다. 실제 결과에서도 SBP의 MAE와 STD가 DBP, MAP보다 높다.

### 4.2 전처리

전처리는 세 단계로 구성된다. 첫째, PPG와 ABP 신호에 대해 signal quality judgment를 수행한다. 과도한 amplitude, saturation 등의 문제가 있는 데이터는 제거된다. 둘째, 각 subject의 PPG와 ABP signal pair를 segment 단위로 분할한다. 셋째, PPG signal에 0.5–10 Hz band-pass filter를 적용한다.

ABP signal에서는 각 segment의 peak와 valley를 추출한다. Peak의 평균값은 SBP label로, valley의 평균값은 DBP label로 사용된다. MAP label의 구체적 추출 방식은 제공된 텍스트에서 명확히 설명되어 있지 않다.

### 4.3 데이터 분할 전략

논문은 data leakage를 방지하기 위해 subject-wise split을 사용한다. 전체 808명의 subject를 기본 단위로 보고, 같은 subject가 training set, validation set, test set에 동시에 나타나지 않도록 한다. 분할 비율은 training 65%, validation 10%, test 25%이다.

이 전략은 PPG 기반 혈압 추정에서 매우 중요하다. 만약 segment 단위로 random split을 하면 동일 subject의 다른 segment가 학습과 테스트에 동시에 포함될 수 있다. 이 경우 모델은 subject-specific pattern을 암기할 수 있으며, 실제 새로운 사람에게 적용했을 때보다 과도하게 좋은 성능이 나올 수 있다. PCTN 논문은 subject-wise split을 적용했다는 점에서 평가 설계가 비교적 엄격하다.

### 4.4 평가 지표와 국제 기준

논문은 ME, MAE, STD를 기본 정확도 지표로 사용한다. 또한 임상적 수용 가능성을 보기 위해 BHS와 AAMI 기준을 사용한다.

BHS 기준은 절대 오차가 5 mmHg, 10 mmHg, 15 mmHg 이하에 들어오는 비율에 따라 Grade A, B, C를 부여한다. Grade A를 만족하려면 각각 60%, 85%, 95% 이상이어야 한다. AAMI 기준은 subject 수가 85명 이상이고, ME가 5 mmHg 이하, STD가 8 mmHg 이하일 것을 요구한다.

또한 Bland–Altman analysis를 사용하여 PCTN 예측값과 ABP 기반 참값 사이의 agreement를 평가한다. Bland–Altman 분석에서는 각 sample에 대해 두 측정값의 차이와 평균을 계산한다.

$$
BP_{diff} = BP_{true} - BP_{predicted}
$$

$$
BP_{mean} =
\frac{1}{2}
(BP_{true} + BP_{predicted})
$$

이 분석은 단순 correlation뿐 아니라 모델이 systematic bias를 가지는지, 오차가 특정 혈압 범위에서 커지는지, 95% limits of agreement 안에 대부분의 샘플이 포함되는지를 확인하는 데 유용하다.

### 4.5 BHS 결과

PCTN은 DBP, MAP, SBP 모두에서 BHS Grade A를 달성한다. 구체적 결과는 다음과 같다.

| 지표 | ≤ 5 mmHg | ≤ 10 mmHg | ≤ 15 mmHg | Grade |
| ---- | -------: | --------: | --------: | ----- |
| DBP  |   89.09% |    98.83% |    99.92% | A     |
| MAP  |   90.17% |    99.35% |    99.95% | A     |
| SBP  |   66.78% |    91.05% |    97.58% | A     |

DBP와 MAP은 5 mmHg 이내 비율이 약 90%로 매우 높다. SBP는 5 mmHg 이내 비율이 66.78%로 DBP와 MAP보다 낮지만, Grade A 기준인 60%를 초과한다. 이는 SBP가 상대적으로 예측하기 어렵지만, BHS 기준에서는 여전히 높은 등급을 만족한다는 것을 의미한다.

### 4.6 AAMI 결과

AAMI 기준에서 PCTN의 결과는 다음과 같다.

| 지표 |        ME |       STD | Subjects |
| ---- | --------: | --------: | -------: |
| DBP  | 0.18 mmHg | 3.22 mmHg |      808 |
| MAP  | 0.18 mmHg | 3.07 mmHg |      808 |
| SBP  | 0.17 mmHg | 5.98 mmHg |      808 |

모든 지표에서 ME는 0에 매우 가깝다. 이는 PCTN이 전체적으로 혈압을 과대평가하거나 과소평가하는 systematic bias가 작다는 뜻이다. STD도 DBP 3.22, MAP 3.07, SBP 5.98로 모두 AAMI 기준인 8 mmHg 이하를 만족한다. 특히 SBP의 STD가 가장 크며, 이는 SBP의 생리적 변동성이 더 크다는 논문의 논의와도 일치한다.

### 4.7 주요 MAE 결과

논문 초록과 비교 표에서 제시한 PCTN의 MAE는 다음과 같다.

| 지표 |       MAE |
| ---- | --------: |
| DBP  | 2.36 mmHg |
| MAP  | 2.03 mmHg |
| SBP  | 4.44 mmHg |

MAP의 MAE가 가장 낮고, SBP의 MAE가 가장 높다. 이는 SBP가 혈관 탄성, 개인차, 넓은 변동 범위의 영향을 더 크게 받기 때문이라고 논문은 해석한다.

### 4.8 Bland–Altman 및 회귀 분석

Bland–Altman 분석에서 DBP, MAP, SBP의 limits of agreement는 각각 다음과 같이 제시된다.

| 지표 | Limits of Agreement  |
| ---- | -------------------- |
| DBP  | [-6.14, 6.49] mmHg   |
| MAP  | [-5.84, 6.19] mmHg   |
| SBP  | [-11.56, 11.90] mmHg |

DBP와 MAP은 비교적 좁은 범위 안에 오차가 분포하지만, SBP는 범위가 더 넓다. 논문은 특히 SBP graph에서 outlier가 존재한다고 언급한다. 이는 SBP 예측이 DBP나 MAP보다 더 어렵다는 결과와 일관된다.

회귀 분석에서는 PCC(Pearson Correlation Coefficient)를 사용한다. 결과는 다음과 같다.

| 지표 |  PCC |
| ---- | ---: |
| DBP  | 0.78 |
| MAP  | 0.91 |
| SBP  | 0.93 |

DBP는 강한 양의 상관관계를 보이지만 MAP과 SBP보다 낮다. MAP과 SBP는 매우 높은 positive correlation을 보인다. p-value는 모두 0.000001보다 작다고 보고되어 통계적으로 유의미한 상관관계를 가진다.

### 4.9 기존 방법과의 비교

논문은 전통적 머신러닝 방법과 딥러닝 방법을 포함한 여러 기존 연구와 PCTN을 비교한다. 비교 대상에는 PTT + SVM, PAT + Adaboost, Random Forest, CNN 기반 deep learning, GRU + Attention, modified ResNet 등이 포함된다.

표 6에 따르면 PCTN은 DBP, MAP, SBP의 MAE와 STD에서 기존 방법보다 우수한 결과를 보인다. 특히 기존 방법들의 SBP MAE는 대체로 5 mmHg 이상이지만, PCTN은 4.44 mmHg를 달성한다. DBP MAE도 2.36 mmHg로 비교 방법들보다 낮다. STD 역시 DBP 3.22, MAP 3.07, SBP 5.98로 가장 낮은 수준이다.

다만 비교 표의 일부 기존 연구는 MIMIC-II를 사용하고, PCTN은 MIMIC-III 808명을 사용한다. 또한 각 연구의 preprocessing, split strategy, subject 수, label extraction 방식이 완전히 동일하지 않을 가능성이 있다. 따라서 표의 비교는 PCTN의 상대적 강점을 보여주는 참고 자료로는 유용하지만, 완전히 통제된 공정 비교라고 단정하기는 어렵다.

### 4.10 Ablation study: CNN과 Transformer block 수의 영향

저자들은 local feature와 global feature의 중요성을 확인하기 위해 ablation study를 수행한다. CNN block을 $C_n$, Transformer block을 $Tr$로 표기하고, 총 block 수가 4가 되도록 여러 조합을 비교한다. 이 실험에서는 fusion module을 사용하지 않고 feature extraction 이후 단순 concatenation만 수행한다.

결과는 다음과 같다.

| Model   | CNN block | Transformer block | DBP MAE | SBP MAE | DBP STD | SBP STD |
| ------- | --------: | ----------------: | ------: | ------: | ------: | ------: |
| CCC / C |         4 |                 0 |    5.23 |    8.78 |    6.59 |   11.12 |
| CCC / T |         3 |                 1 |    4.78 |    8.29 |    5.90 |    9.90 |
| CC / TT |         2 |                 2 |    4.30 |    7.62 |    5.42 |    9.73 |
| C / TTT |         1 |                 3 |    4.16 |    7.33 |    5.22 |    9.29 |
| TTT / T |         0 |                 4 |    4.61 |    7.74 |    6.97 |   10.15 |

이 결과는 두 가지 점을 보여준다. 첫째, CNN만 사용하거나 Transformer만 사용하는 것보다 CNN과 Transformer를 함께 사용하는 것이 더 좋다. 둘째, local feature와 global feature를 함께 사용할 때 Transformer block의 비율이 높아질수록 대체로 성능이 좋아진다. 최적의 단순 조합은 C|TTT로 나타난다. 이는 PPG에서 global temporal dependency가 중요하며, self-attention이 혈압 추정에 유리한 정보를 제공할 수 있음을 시사한다.

그러나 Transformer만 사용한 TTT|T는 C|TTT보다 성능이 낮다. 이는 global feature만으로는 충분하지 않고, CNN이 제공하는 local waveform feature가 여전히 필요하다는 점을 보여준다.

### 4.11 Ablation study: Fusion strategy의 효과

두 번째 ablation study는 fusion strategy의 효과를 평가한다. 기본 구조는 C|TTT이며, 여기에 SE, CBAM, PCTN fusion 방식을 차례로 적용한다.

| Model          | DBP MAE | SBP MAE | DBP STD | SBP STD |
| -------------- | ------: | ------: | ------: | ------: |
| C / TTT        |    4.16 |    7.33 |    5.22 |    9.29 |
| C / TTT + SE   |    4.07 |    6.23 |    5.10 |    7.89 |
| C / TTT + CBAM |    3.73 |    5.97 |    4.52 |    7.56 |
| PCTN           |    2.36 |    4.44 |    3.22 |    5.98 |

SE를 추가하면 성능이 개선되고, CBAM을 추가하면 더 개선된다. 최종 PCTN fusion strategy는 가장 낮은 MAE와 STD를 달성한다. 이는 local feature와 global feature를 단순히 합치는 것보다 attention 기반으로 위치와 채널 중요도를 조정하는 것이 혈압 추정에 매우 중요하다는 것을 보여준다.

## 5. 강점, 한계

### 5.1 강점

이 논문의 가장 큰 강점은 PPG 기반 혈압 추정에서 local feature와 global feature의 상호 보완성을 명확한 아키텍처로 구현했다는 점이다. CNN branch는 PPG waveform의 국소 morphology를 추출하고, Transformer branch는 시퀀스 전체의 장거리 의존성을 학습한다. 이 병렬 구조는 단일 CNN, 단일 Transformer, 단일 RNN 계열 모델보다 더 균형 잡힌 표현을 제공한다.

두 번째 강점은 fusion strategy를 별도로 설계했다는 점이다. 단순 concatenation이 아니라 branch별 spatial attention과 이후 channel attention을 사용하여, 두 종류의 feature가 무분별하게 섞이지 않도록 한다. Ablation study는 이 fusion block이 성능 향상에 중요한 역할을 한다는 것을 보여준다.

세 번째 강점은 subject-wise split을 사용했다는 점이다. PPG 기반 혈압 추정에서는 동일 subject의 segment가 학습과 테스트에 동시에 포함되면 data leakage가 발생할 수 있다. 이 논문은 subject를 최소 단위로 하여 train, validation, test set을 분할하므로, 평가가 비교적 현실적인 cross-subject generalization을 반영한다.

네 번째 강점은 BHS와 AAMI 기준을 모두 만족했다는 점이다. 단순 MAE뿐 아니라 임상적 혈압계 평가 기준을 사용했다는 점은 biomedical signal processing 연구로서 타당성을 높인다. 또한 Bland–Altman analysis와 regression analysis를 함께 제시하여 agreement와 correlation을 모두 확인했다.

### 5.2 한계

첫 번째 한계는 MAP 예측 방식이 명확하지 않다는 점이다. 논문은 DBP, MAP, SBP 결과를 모두 보고하지만, regressor module 설명에서는 SBP와 DBP를 출력한다고 되어 있다. MAP을 별도로 예측했는지, ABP에서 계산한 MAP label을 대상으로 학습했는지, 또는 SBP와 DBP로부터 계산했는지는 제공된 텍스트만으로 확인할 수 없다.

두 번째 한계는 외부 데이터셋 검증이 부족하다는 점이다. 실험은 MIMIC-III 데이터셋에 집중되어 있다. MIMIC는 ICU 환경의 데이터이므로, 일반 웨어러블 환경에서의 motion artifact, sensor placement variation, 피부 특성, 일상 활동 중 신호 왜곡을 충분히 반영하지 못할 수 있다. 실제 cuff-less BP monitoring을 위해서는 스마트워치나 패치형 센서에서 수집한 외부 데이터셋 검증이 필요하다.

세 번째 한계는 subject-wise split을 사용했더라도, 실제 장기 모니터링에서 발생하는 calibration drift, sensor drift, physiological change over time을 충분히 평가하지 못했다는 점이다. 논문은 discussion에서 continuous monitoring에서는 생리적 변화, sensor drift, individual variability가 문제가 된다고 언급하지만, 이를 직접 장기간 실험으로 검증하지는 않았다.

네 번째 한계는 모델이 비교적 크다는 점이다. Ablation table에 따르면 PCTN의 parameter 수는 약 27.81M이다. 이는 웨어러블 기기에 직접 탑재하기에는 부담이 될 수 있다. 논문은 향후 lightweight model 개발을 계획한다고 언급하지만, 현재 모델의 실제 wearable deployment 결과는 제시되지 않았다.

다섯 번째 한계는 일부 baseline 비교의 공정성을 완전히 확인하기 어렵다는 점이다. 비교 표에는 MIMIC-II와 MIMIC-III 결과가 함께 포함되어 있고, subject 수와 preprocessing 방식도 연구마다 다르다. 따라서 PCTN이 표에 나열된 모든 방법보다 절대적으로 우수하다고 단정하기보다는, 해당 설정에서 강한 성능을 보였다고 해석하는 것이 더 적절하다.

여섯 번째 한계는 demographic features를 사용하지 못했다는 점이다. 논문은 MIMIC 데이터의 confidentiality 문제로 age, weight, height와 같은 static feature에 접근할 수 없었다고 설명한다. 혈압은 나이, 체중, 혈관 상태, 질환 이력 등과 밀접하게 관련되므로 이러한 정보를 포함하면 성능과 개인화 가능성이 개선될 수 있다.

### 5.3 비판적 해석

이 논문은 PPG 기반 cuff-less BP estimation에서 Transformer의 역할을 명확하게 보여준다는 점에서 의미가 있다. CNN만으로는 국소 특징에 치우치고, Transformer만으로는 세밀한 파형 morphology를 놓칠 수 있다. PCTN은 두 branch를 병렬로 구성하고 attention-based fusion을 추가함으로써 두 모델의 장점을 결합한다.

다만 방법론적으로 완전히 새로운 self-attention 구조나 혈압 생리 모델을 제안한 논문이라기보다는, 기존 CNN, Transformer, CBAM 계열 attention을 PPG 기반 혈압 추정 문제에 적절히 조합한 architecture-oriented 연구로 보는 것이 타당하다. 성능은 우수하지만, 실제 임상 또는 웨어러블 적용 가능성을 주장하기 위해서는 외부 데이터셋, 장기 측정, 개인화 calibration, real-world artifact 환경에서 추가 검증이 필요하다.

특히 SBP에서 outlier가 더 많고 limits of agreement가 넓다는 점은 중요한 관찰이다. SBP는 동맥 탄성, 혈관 반응성, 개인차, 측정 상황에 영향을 많이 받기 때문에 PPG만으로 안정적으로 예측하기 어렵다. PCTN이 기존 방법보다 개선된 것은 분명하지만, SBP 예측의 신뢰도를 실제 의료 의사결정에 사용할 수 있는 수준으로 보장하려면 더 엄격한 검증이 필요하다.

## 6. 결론

이 논문은 PPG 기반 cuff-less blood pressure estimation을 위해 CNN과 Transformer를 병렬로 결합한 PCTN을 제안하였다. PCTN은 CNN branch를 통해 local waveform feature를 추출하고, Transformer branch를 통해 global temporal dependency를 학습한다. 이후 spatial attention과 channel attention을 결합한 fusion block을 사용하여 두 feature를 효과적으로 통합하고, fully connected regressor를 통해 혈압 값을 예측한다.

MIMIC-III 데이터셋의 808명 subject를 대상으로 한 실험에서 PCTN은 DBP, MAP, SBP에 대해 각각 2.36 mmHg, 2.03 mmHg, 4.44 mmHg의 MAE를 달성하였다. 또한 BHS Grade A와 AAMI 기준을 모두 만족하였다. Ablation study는 CNN과 Transformer를 함께 사용하는 것이 단일 branch보다 효과적이며, attention-based fusion이 성능 향상에 핵심적임을 보여준다.

이 연구는 PPG 기반 혈압 추정에서 local feature와 global feature를 함께 고려하는 설계가 중요하다는 근거를 제공한다. 특히 Transformer가 PPG 시계열의 전역 의존성을 학습하는 데 유용하다는 점을 실험적으로 보여준다. 향후 연구에서는 모델 경량화, 외부 데이터셋 검증, 실제 웨어러블 환경에서의 장기 측정, 개인화 calibration 전략, demographic feature 결합이 중요한 발전 방향이 될 것이다.

종합적으로 PCTN은 PPG 기반 비침습 혈압 추정 분야에서 CNN의 local representation과 Transformer의 global representation을 결합한 실용적이고 성능 지향적인 딥러닝 접근법으로 평가할 수 있다. 실제 임상 적용을 위해서는 추가 검증이 필요하지만, 웨어러블 기반 연속 혈압 모니터링 연구에 중요한 기반 모델 중 하나로 볼 수 있다.
