# Deep learning approaches for continuous blood pressure estimation from photoplethysmography signal

* **저자**: R. Vanithamani, S. Sri Jayabharathi, S. Pavithra, E. Smily Jeya Jothi
* **발표연도**: 2025

## 1. 논문 개요

이 논문은 PPG(Photoplethysmography) 신호를 이용하여 연속적인 혈압, 특히 SBP(Systolic Blood Pressure)와 DBP(Diastolic Blood Pressure)를 추정하기 위한 여러 deep learning 접근법을 비교·평가한다. 논문은 *Measurement: Sensors* 39권에 2025년 논문 번호 101866으로 게재되었으며, DOI는 10.1016/j.measen.2025.101866이다.

논문의 목표는 PPG waveform의 full cycle을 입력으로 사용하여 SBP와 DBP를 정확히 예측하는 것이다. 저자들은 TCN(Temporal Convolutional Network), LSTM(Long Short-Term Memory), TCN-LSTM, Autoencoder-LSTM을 후보 모델로 제시하고, 이들 중 Autoencoder-LSTM이 가장 낮은 오차를 달성한다고 보고한다. 초록에 따르면 Autoencoder-LSTM은 SBP에 대해 MAE 1.05, SD 1.89를 달성했고, DBP에 대해 MAE 0.92, SD 1.05를 달성했다.

이 연구의 문제의식은 혈압이 빠르게 변할 수 있기 때문에 단발성 측정보다 연속적 모니터링이 중요하다는 데 있다. 기존 커프 기반 혈압 측정은 비교적 정확하지만 불편하고 연속 측정에 한계가 있다. 반면 PPG는 피부 표면에서 혈액량 변화를 광학적으로 측정하는 비침습적이고 저비용의 방법이며, 웨어러블 기기에 쉽게 탑재될 수 있다. 따라서 PPG 기반 연속 혈압 추정은 고혈압, 심혈관 질환, 동맥 경직도 등과 관련된 건강 상태를 장시간 모니터링하는 데 중요한 응용 가능성을 가진다.

다만 제공된 본문에는 중요한 불명확성이 존재한다. 논문 초록과 방법 설명은 SBP와 DBP 추정을 목표로 한다고 말하지만, Introduction 말미에서는 “TCN, LSTM, Autoencoder-based unsupervised deep learning framework that can identify artifacts in a PPG signal”이라고 표현한다. 이는 혈압 추정 모델을 설명하는 흐름과 약간 맞지 않는다. 실제로 모델이 artifact detection을 수행한 뒤 혈압 추정을 했는지, 또는 autoencoder를 feature learning 용도로 사용했는지는 제공된 추출 텍스트만으로 완전히 확인하기 어렵다. 또한 원문 추출이 LSTM 설명 중간에서 끊겨 있어, 이후에 제시되었을 가능성이 있는 전체 수식, 실험 표, 모델 세부 구조를 직접 확인할 수 없다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 PPG 신호의 전체 주기 waveform이 혈압 추정에 충분한 정보를 포함하고 있으며, deep learning 모델이 이 waveform에서 SBP와 DBP에 관련된 특징을 자동으로 학습할 수 있다는 것이다. 기존의 전통적 방법은 PPG에서 maximum slope point, systolic peak, dicrotic notch, inflection point, diastolic peak 같은 morphology feature를 사람이 설계하고, 이를 회귀 모델에 입력하여 혈압을 예측했다. 그러나 이러한 handcrafted feature 기반 방법은 feature 선택에 의존하고, motion artifact나 개인별 혈관 특성 변화에 취약할 수 있다.

저자들은 이러한 한계를 해결하기 위해 TCN, LSTM, TCN-LSTM, Autoencoder-LSTM을 비교한다. 각 모델은 PPG 시계열의 서로 다른 측면을 학습하도록 설계된다. TCN은 convolution 기반 구조를 통해 시간축의 국소적 패턴과 장기 의존성을 병렬적으로 학습할 수 있다. LSTM은 gating mechanism을 이용해 시계열의 장기 의존성을 학습한다. TCN-LSTM은 convolution 기반 feature extraction과 recurrent temporal modeling을 결합한다. Autoencoder-LSTM은 autoencoder를 통해 PPG 신호의 압축된 latent representation을 학습한 뒤, LSTM을 사용해 시간적 관계를 반영하는 구조로 해석된다.

논문에서 가장 강하게 제시되는 결론은 Autoencoder-LSTM이 가장 우수하다는 것이다. Autoencoder는 입력 신호를 저차원 latent representation으로 압축하고 다시 복원하는 과정에서 데이터의 핵심 구조를 학습한다. 이 latent representation은 원 신호보다 잡음이나 중복성이 줄어든 표현일 수 있다. 여기에 LSTM을 결합하면 압축된 특징의 시간적 의존성을 학습할 수 있으므로, PPG 기반 SBP와 DBP 추정에 유리하다는 것이 논문의 중심 설계 아이디어로 볼 수 있다.

기존 접근 방식과의 차별점은 단일 모델을 제안하는 데 그치지 않고 여러 deep learning 구조를 같은 문제에 적용해 비교했다는 점이다. 또한 ECG와 PPG를 모두 사용하는 복합 센서 기반 연구들과 달리, 이 논문은 PPG 신호 기반 연속 혈압 추정에 초점을 맞춘다. 다만 제공된 텍스트에 따르면 사용한 PhysioNet 데이터셋은 wrist PPG activity data로 설명되며, 혈압 label이 어떻게 연결되어 있는지에 대한 설명은 충분하지 않다. 이 부분은 논문 방법론의 핵심이므로 원문 전체에서 추가 확인이 필요하다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

제공된 텍스트에 근거하면, 논문의 전체 파이프라인은 다음과 같이 정리할 수 있다.

PPG 신호를 입력으로 사용하고, 데이터셋을 training set과 testing/validation set으로 나눈다. Training set은 전체 데이터의 70%, testing 및 validation set은 30%로 구성된다. Deep learning 모델은 training data에서 PPG 신호의 특징을 학습하고, 이후 testing data에서 SBP와 DBP를 예측한다. 마지막으로 예측된 SBP와 DBP를 실제 값과 비교하여 MAE와 SD를 계산한다.

전체 흐름은 다음과 같이 표현할 수 있다.

$$
PPG \ signal \rightarrow Deep \ learning \ model \rightarrow \hat{SBP}, \hat{DBP}
$$

보다 구체적으로 Autoencoder-LSTM의 관점에서 보면 다음과 같은 구조로 이해할 수 있다.

$$
PPG \ waveform \rightarrow Encoder \rightarrow Latent \ representation \rightarrow LSTM \rightarrow Regression \rightarrow \hat{SBP}, \hat{DBP}
$$

다만 제공된 텍스트에는 Autoencoder-LSTM의 정확한 layer 구성, latent dimension, decoder 사용 방식, 최종 regression head 구조가 포함되어 있지 않다. 따라서 위 구조는 논문이 언급한 autoencoder와 LSTM의 일반적 역할에 기반한 해석이며, 원문에서 명시된 세부 구현이라고 단정할 수는 없다.

### 3.2 PPG 신호와 혈압 추정의 생리적 배경

논문은 PPG waveform이 심혈관계 상태를 반영한다고 설명한다. PPG는 피부 표면의 미세혈관 혈액량 변화를 측정하는 광학 신호이다. 심장이 박동하면서 혈액이 말초 조직으로 이동하면 PPG amplitude가 변화하고, 이 변화가 pulse waveform으로 나타난다.

논문은 PPG waveform의 주요 지점을 설명한다. Systolic peak는 좌심실에서 혈액이 박출되어 혈압파가 손가락 방향으로 이동할 때 나타나는 주요 peak이다. Diastolic peak는 말초 혈관에서 반사된 혈압파가 대동맥 및 손가락 방향으로 되돌아오면서 발생하는 추가 peak이다. Dicrotic notch는 대동맥판이 닫히면서 일시적으로 혈류가 변할 때 생기는 작은 움푹 들어간 지점이다. Inflection point 또는 incisura는 systolic phase에서 diastolic phase로 넘어가는 전환점을 의미한다.

이러한 morphology point들은 혈관 탄성, 심장 박출, 반사파, 말초 혈관 상태와 관련될 수 있다. 따라서 PPG waveform은 SBP와 DBP 추정에 유용한 정보를 포함할 수 있다. 하지만 실제 환경의 PPG는 motion artifact, ambient light interference, sensor contact variation, high-frequency noise 등의 영향을 받는다. 특히 웨어러블 기기에서는 움직임에 의한 잡음이 강하게 발생하므로, 신호 품질과 모델의 잡음 견고성이 중요하다.

### 3.3 데이터셋

논문은 PhysioNet에서 제공되는 공개 PPG 데이터셋을 사용한다고 설명한다. 이 데이터셋은 손목 PPG 신호를 포함하며, 피험자들이 다양한 신체 활동을 수행하는 동안 기록되었다. 활동 조건은 treadmill walking, treadmill running, exercise bike low-resistance workout, exercise bike high-resistance workout의 네 가지로 제시된다.

PPG 신호의 sampling rate는 256 Hz이다. 논문은 실제 활동 중 기록된 신호를 사용하므로 motion artifact가 포함되어 있다고 설명한다. 이는 정적인 실험실 환경보다 웨어러블 환경에 가까운 데이터라는 장점을 가진다. Cycling data에는 high-frequency noise가 많았기 때문에 second-order IIR Butterworth digital filter와 15 Hz cut-off, zero group delay를 이용해 low-pass filtering을 수행했다고 설명한다. 이후 필터링된 신호는 WFDB(WaveForm DataBase) format으로 변환되었다.

그러나 제공된 텍스트만으로는 이 데이터셋에 SBP와 DBP label이 어떻게 포함되어 있는지 명확하지 않다. PhysioNet의 wrist PPG activity dataset 설명은 주로 PPG와 motion signal 및 활동 조건을 중심으로 되어 있으며, 원문 발췌에는 ABP 또는 cuff blood pressure reference를 어떻게 확보했는지 설명이 없다. 혈압 추정 연구에서 label 생성 방식은 매우 중요하다. SBP와 DBP가 어떤 장비로 측정되었는지, segment마다 label이 있는지, 활동 전후의 단일 혈압값을 사용했는지, 연속 ABP를 사용했는지에 따라 문제의 난이도와 결과 해석이 크게 달라진다. 따라서 이 논문은 제공된 텍스트만 기준으로 보면 target label의 출처 설명이 부족하다.

### 3.4 데이터 분할

논문은 데이터셋을 70% training set과 30% testing 및 validation set으로 나누었다고 설명한다. Training phase에서는 deep learning algorithm이 PPG 신호에서 관련 feature를 학습하고, testing phase에서는 SBP와 DBP를 예측한다.

다만 제공된 텍스트에는 subject-wise split 여부가 명확히 제시되어 있지 않다. PPG 기반 혈압 추정에서는 동일 피험자의 여러 segment가 training set과 test set에 동시에 들어가면 data leakage가 발생할 수 있다. 이 경우 모델은 일반적인 혈압-PPG 관계를 학습했다기보다 특정 피험자의 waveform pattern을 기억할 가능성이 있다. 따라서 이 논문 결과를 엄격하게 평가하려면 split이 subject-wise인지, segment-wise random split인지 반드시 확인해야 한다. 제공된 발췌문만으로는 이를 알 수 없다.

### 3.5 LSTM 모델

논문은 LSTM을 시계열 데이터 처리 모델로 설명한다. 일반 RNN은 시퀀스 길이가 길어질수록 gradient vanishing 또는 gradient exploding 문제가 발생하여 장기 의존성을 학습하기 어렵다. LSTM은 이러한 문제를 줄이기 위해 memory cell과 세 가지 gate를 사용한다.

LSTM cell은 forget gate, input gate, output gate, cell state로 구성된다. Forget gate는 이전 cell state에서 어떤 정보를 버릴지 결정한다. Input gate는 현재 입력에서 어떤 정보를 새로 저장할지 결정한다. Output gate는 cell state에서 어떤 정보를 hidden output으로 내보낼지 결정한다.

일반적인 LSTM 수식은 다음과 같이 표현할 수 있다.

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$

$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
$$

$$
\tilde{c}*t = \tanh(W_c [h*{t-1}, x_t] + b_c)
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

여기서 $x_t$는 현재 입력, $h_{t-1}$은 이전 hidden state, $c_{t-1}$은 이전 cell state이다. $\sigma$는 sigmoid 함수, $\odot$는 element-wise multiplication이다. 논문 발췌는 LSTM 설명 중간에서 끊겨 있으므로 원문에 이 수식들이 그대로 제시되었는지는 확인할 수 없다. 위 수식은 LSTM의 표준 동작을 설명하기 위한 것이다.

PPG 혈압 추정에서 LSTM의 역할은 waveform의 시간적 순서를 학습하는 것이다. PPG waveform은 심박 주기별로 반복되지만, 각 cycle의 형태와 amplitude가 미세하게 달라진다. LSTM은 이러한 시간적 변화를 순차적으로 반영하여 SBP와 DBP 예측에 필요한 정보를 축적할 수 있다.

### 3.6 TCN 모델

논문은 TCN을 후보 deep learning model 중 하나로 제시한다. 제공된 텍스트에는 TCN의 세부 수식이나 구조 설명이 포함되어 있지 않지만, TCN은 일반적으로 causal convolution, dilated convolution, residual connection을 사용하여 시계열의 장기 의존성을 병렬적으로 학습하는 모델이다.

TCN의 장점은 RNN처럼 순차적으로 계산하지 않아도 되므로 병렬화가 쉽고, dilation을 통해 긴 receptive field를 확보할 수 있다는 점이다. PPG 신호에서는 한 심박 주기 내의 국소 패턴뿐 아니라 여러 주기에 걸친 변화가 혈압과 관련될 수 있으므로, TCN은 합리적인 후보 모델이다.

일반적인 dilated convolution은 다음과 같이 표현할 수 있다.

$$
y_t =
\sum_{k=0}^{K-1}
w_k x_{t-dk}
$$

여기서 $K$는 kernel size, $d$는 dilation factor이다. $d$가 커질수록 convolution이 더 넓은 시간 간격을 건너뛰며 입력을 참조하므로, 적은 layer 수로도 긴 temporal context를 볼 수 있다. 다만 이 논문에서 사용한 TCN의 dilation, kernel size, channel 수 등은 제공된 텍스트에 명확히 포함되어 있지 않다.

### 3.7 TCN-LSTM 모델

TCN-LSTM은 TCN과 LSTM을 결합한 구조로 이해된다. TCN은 PPG 신호에서 temporal convolution을 통해 국소 및 장기 패턴을 먼저 추출하고, LSTM은 그 결과를 순차적으로 처리하여 시간 의존성을 추가로 학습한다.

개념적으로는 다음 흐름으로 표현할 수 있다.

$$
PPG \rightarrow TCN \ feature \ extraction \rightarrow LSTM \ temporal \ modeling \rightarrow \hat{SBP}, \hat{DBP}
$$

이 구조는 CNN-LSTM 계열 모델과 유사한 철학을 가진다. Convolution 계열 모듈은 feature extraction에 강하고, LSTM은 sequential dependency modeling에 강하므로, 두 모듈을 결합하면 PPG waveform의 morphology와 temporal evolution을 함께 학습할 수 있다.

### 3.8 Autoencoder-LSTM 모델

논문 초록에 따르면 Autoencoder-LSTM이 가장 좋은 성능을 보인다. Autoencoder는 encoder와 decoder로 구성된다. Encoder는 입력 신호를 latent space의 저차원 표현으로 압축하고, decoder는 이 latent representation으로부터 원래 입력을 복원한다.

기본적인 autoencoder의 학습 목표는 입력 $x$와 복원값 $\hat{x}$의 차이를 줄이는 것이다.

$$
z = Encoder(x)
$$

$$
\hat{x} = Decoder(z)
$$

$$
L_{rec} =
|x-\hat{x}|^2
$$

여기서 $z$는 latent representation이다. Autoencoder가 잘 학습되면 $z$는 원 신호의 핵심 구조를 담는 압축 표현이 된다. PPG 신호에서는 잡음, 중복 정보, 활동성 motion artifact가 존재할 수 있으므로, autoencoder가 유용한 latent feature를 학습할 가능성이 있다.

Autoencoder-LSTM은 이 latent representation을 LSTM에 입력하여 SBP와 DBP를 추정하는 구조로 해석할 수 있다.

$$
z_t = Encoder(x_t)
$$

$$
h_t = LSTM(z_t, h_{t-1})
$$

$$
(\hat{SBP}, \hat{DBP}) = Regression(h_t)
$$

이 구조의 장점은 autoencoder가 먼저 PPG 신호의 중요한 구조를 압축하고, LSTM이 그 압축 표현의 시간적 관계를 학습할 수 있다는 점이다. 초록의 결과에 따르면 Autoencoder-LSTM은 SBP MAE 1.05, DBP MAE 0.92로 가장 좋은 성능을 달성했다. 다만 제공된 텍스트에는 Autoencoder의 정확한 layer 수, latent dimension, reconstruction loss 사용 여부, LSTM과의 결합 방식이 나오지 않는다. 따라서 방법론의 완전한 재현성은 제공된 발췌문만으로는 확보하기 어렵다.

### 3.9 학습 목표와 평가 지표

논문은 결과를 MAE와 SD로 보고한다. MAE는 예측값과 실제값 사이 절대 오차의 평균이다.

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
$$

여기서 $y_i$는 실제 혈압값, $\hat{y}_i$는 모델이 예측한 혈압값이다. MAE가 낮을수록 평균적인 예측 오차가 작다.

SD 또는 standard deviation은 예측 오차의 분산 정도를 나타낸다. 오차를 $e_i = y_i-\hat{y}_i$라고 하면, 오차의 표준편차는 다음과 같이 쓸 수 있다.

$$
SD =
\sqrt{
\frac{1}{n-1}
\sum_{i=1}^{n}
(e_i-\bar{e})^2
}
$$

여기서 $\bar{e}$는 평균 오차이다. SD가 낮다는 것은 모델의 오차가 일정하고 안정적이라는 의미이다. 혈압 추정에서는 평균 오차뿐 아니라 오차 변동성도 중요하다. 일부 상황에서 큰 오차가 자주 발생하면 실제 모니터링 시스템으로 사용하기 어렵기 때문이다.

논문 초록은 Autoencoder-LSTM이 SBP에 대해 MAE 1.05, SD 1.89, DBP에 대해 MAE 0.92, SD 1.05를 달성했다고 보고한다. 이는 매우 낮은 오차 수준이다. 그러나 이 수치의 신뢰성을 판단하려면 label 생성 방식, 데이터 분할 방식, 피험자 수, segment 수, cross-subject 평가 여부를 추가로 확인해야 한다.

## 4. 실험 및 결과

### 4.1 실험 대상 모델

논문은 네 가지 deep learning 모델을 비교한다.

첫 번째는 TCN이다. TCN은 convolution 기반 시계열 모델로, PPG waveform의 temporal pattern을 학습한다. 두 번째는 LSTM이다. LSTM은 장기 의존성을 학습하는 recurrent model이다. 세 번째는 TCN-LSTM이다. 이 모델은 TCN의 feature extraction 능력과 LSTM의 temporal modeling 능력을 결합한다. 네 번째는 Autoencoder-LSTM이다. 이 모델은 autoencoder를 통해 PPG 신호의 latent representation을 학습하고, LSTM으로 temporal dependency를 모델링하는 구조로 해석된다.

### 4.2 데이터와 작업

실험 데이터는 PhysioNet에서 가져온 wrist PPG 신호이다. 피험자는 walking, running, low-resistance cycling, high-resistance cycling 등 다양한 활동을 수행한다. 이 설정은 운동 중 발생하는 motion artifact가 포함된 PPG를 다룬다는 점에서 실용적 의미가 있다.

작업은 PPG 신호를 입력으로 받아 SBP와 DBP를 예측하는 regression task이다. 논문은 continuous BP estimation이라는 표현을 사용하지만, 제공된 텍스트에서는 실제 연속 ABP waveform 전체를 예측하는 것이 아니라 SBP와 DBP 수치 추정에 초점이 있다. 따라서 정확히는 PPG 기반 SBP/DBP regression으로 이해하는 것이 적절하다.

### 4.3 주요 정량 결과

초록에 따르면 Autoencoder-LSTM의 성능은 다음과 같다.

| 모델             | SBP MAE | DBP MAE | SBP SD | DBP SD |
| ---------------- | ------: | ------: | -----: | -----: |
| Autoencoder-LSTM |    1.05 |    0.92 |   1.89 |   1.05 |

이 결과는 SBP와 DBP 모두에서 매우 낮은 오차를 의미한다. 특히 DBP MAE 0.92는 평균적으로 1 mmHg 미만의 오차라는 뜻이므로, 기존 PPG 기반 혈압 추정 연구들과 비교해도 매우 높은 성능이다.

다만 제공된 텍스트에는 TCN, LSTM, TCN-LSTM의 실제 수치 결과가 포함되어 있지 않다. 따라서 Autoencoder-LSTM이 다른 모델들보다 얼마나 개선되었는지, 개선폭이 통계적으로 유의한지, 특정 혈압 범위에서 성능이 어떻게 달라지는지는 확인할 수 없다. 또한 train/test split이 subject-wise인지 불명확하기 때문에 결과가 실제 새로운 피험자에게 일반화되는 성능인지, 아니면 segment-level 평가에서 얻은 성능인지는 확정할 수 없다.

### 4.4 기존 연구와의 관계

Literature survey에서는 여러 기존 연구가 제시된다. MLR 기반 morphology feature 모델은 MIMIC I에서 SBP MAE 6.10, DBP MAE 4.65를 보고했다. ECG와 PPG에서 PTT, PIR, HR 등을 추출하고 clustering 및 regression을 적용한 연구는 MIMIC-II에서 SBP MAE 2.56, DBP MAE 2.23을 달성했다. CNN-SVR hybrid model은 MIMIC III의 120명 데이터에서 SBP MAE 1.23 ± 2.45, DBP MAE 3.08 ± 5.67을 보고했다고 소개된다. PPG2BP-cGAN은 BP waveform estimation에서 Pearson coefficient 0.99, MAE 2.86, RMSE 3.54를 달성했다고 언급된다.

이러한 문헌과 비교하면, 본 논문의 Autoencoder-LSTM 결과는 매우 우수하다. 특히 SBP MAE 1.05, DBP MAE 0.92는 기존 연구들보다 낮은 수준이다. 그러나 비교 대상 연구들은 데이터셋, 입력 신호, label 생성 방식, 평가 프로토콜이 다르므로 단순 수치 비교만으로 우열을 단정하기는 어렵다. 예를 들어 ECG와 PPG를 함께 사용한 연구, ABP waveform을 예측한 연구, subject-wise 평가를 수행한 연구, segment-wise 평가를 수행한 연구는 서로 난이도가 다르다.

### 4.5 결과의 중요성

논문이 보고한 수치가 엄격한 평가 설정에서도 재현된다면, Autoencoder-LSTM은 PPG 기반 연속 혈압 추정에 매우 유망한 접근법이 될 수 있다. 낮은 MAE와 SD는 평균 정확도뿐 아니라 예측 안정성도 높다는 의미이다. 특히 PPG 신호만으로 SBP와 DBP를 예측할 수 있다면, 웨어러블 기기에서 비침습적 혈압 모니터링을 구현하는 데 큰 장점이 있다.

그러나 제공된 원문 발췌만으로는 실험 결과를 완전히 신뢰하기에 필요한 정보가 부족하다. 혈압 label의 출처, subject 수, 총 segment 수, subject-wise split 여부, train/validation/test 구성, 각 모델의 architecture detail, hyperparameter, statistical significance, AAMI 또는 BHS 기준 충족 여부가 충분히 확인되지 않는다. 따라서 결과는 흥미롭지만, 임상적 또는 제품 수준의 성능으로 해석하기에는 추가 검증이 필요하다.

## 5. 강점, 한계

### 5.1 강점

이 논문의 첫 번째 강점은 PPG 신호만을 이용한 연속 혈압 추정이라는 실용적 문제를 다룬다는 점이다. PPG는 웨어러블 기기에 널리 사용되는 센서이며, 신호 획득이 간단하고 비침습적이다. 따라서 PPG 기반 SBP/DBP 추정 모델은 장기 건강 모니터링과 고혈압 관리에 실질적 가치가 있다.

두 번째 강점은 여러 deep learning 구조를 비교했다는 점이다. 단일 모델만 제안한 것이 아니라 TCN, LSTM, TCN-LSTM, Autoencoder-LSTM을 함께 평가하여, PPG 혈압 추정에 어떤 구조가 더 적합한지 탐색한다. 특히 Autoencoder-LSTM이 가장 낮은 MAE와 SD를 보였다는 결론은 PPG 신호의 압축 표현과 temporal modeling을 결합하는 접근이 효과적일 수 있음을 시사한다.

세 번째 강점은 운동 중 wrist PPG 데이터처럼 artifact가 포함될 수 있는 데이터를 고려한다는 점이다. 웨어러블 환경에서는 사용자가 걷거나 뛰거나 운동할 때 신호 잡음이 크게 증가한다. 따라서 다양한 physical activity 조건에서 기록된 PPG를 사용했다는 점은 실제 적용 가능성과 관련된 장점이다.

네 번째 강점은 PPG waveform의 생리적 특징을 설명하고, systolic peak, diastolic peak, dicrotic notch, inflection point가 혈압 추정과 관련될 수 있음을 논의했다는 점이다. 이는 deep learning 모델을 단순한 black-box로만 제시하지 않고, PPG morphology와 혈압 사이의 생리적 연관성을 배경으로 삼았다는 의미가 있다.

### 5.2 한계

가장 큰 한계는 혈압 label의 출처가 제공된 텍스트에서 명확하지 않다는 점이다. 논문은 PhysioNet wrist PPG activity dataset을 사용한다고 설명하지만, 이 발췌문에는 SBP와 DBP reference가 어떻게 측정되었는지 나오지 않는다. 혈압 추정 논문에서 reference BP label은 핵심이다. ABP를 사용했는지, cuff measurement를 사용했는지, 활동 중 연속 label이 있었는지, 각 활동 세션에 단일 label을 부여했는지에 따라 결과 해석이 크게 달라진다.

두 번째 한계는 data split 방식이 불명확하다는 점이다. 70% training, 30% testing/validation split을 사용했다고 되어 있지만, subject-wise split인지 segment-wise random split인지 알 수 없다. PPG 기반 생체신호 모델은 같은 피험자의 segment가 train과 test에 동시에 포함될 경우 과도하게 높은 성능을 보일 수 있다. 특히 보고된 MAE가 매우 낮기 때문에 data leakage 가능성을 반드시 검토해야 한다.

세 번째 한계는 논문 텍스트 내 목표의 일관성이 다소 부족하다는 점이다. 초록과 결과는 SBP/DBP estimation을 말하지만, Introduction 말미에서는 autoencoder-based unsupervised framework가 PPG artifact를 identify한다고 설명한다. Artifact detection이 실제 pipeline에 포함되는지, 아니면 autoencoder의 일반적 설명인지 명확하지 않다.

네 번째 한계는 모델 구조의 세부 정보가 제공된 발췌문에 부족하다는 점이다. Autoencoder-LSTM의 encoder/decoder layer 수, latent dimension, activation function, LSTM unit 수, dropout, optimizer, learning rate, epoch 수, batch size 등이 확인되지 않는다. 이러한 정보가 없으면 결과 재현이 어렵다.

다섯 번째 한계는 국제 혈압 측정 기준인 AAMI 또는 BHS 평가가 제공된 텍스트에 나타나지 않는다는 점이다. MAE와 SD는 중요한 지표지만, 혈압 측정 모델의 임상적 수용 가능성을 평가하려면 평균 오차, 표준편차, subject 수, error distribution 기준을 함께 확인해야 한다. 특히 혈압 추정 연구에서는 Bland–Altman analysis, AAMI, BHS 기준이 자주 사용된다.

여섯 번째 한계는 비교 실험의 전체 결과가 제공되지 않았다는 점이다. 초록은 Autoencoder-LSTM 결과만 명확하게 제시한다. TCN, LSTM, TCN-LSTM의 MAE와 SD가 본문 후반에 있을 가능성이 있지만, 제공된 텍스트에는 포함되어 있지 않다. 따라서 실제 성능 차이를 정량적으로 분석하기 어렵다.

### 5.3 비판적 해석

이 논문은 PPG 기반 혈압 추정에서 Autoencoder-LSTM의 가능성을 보여주는 연구로 볼 수 있다. Autoencoder가 PPG 신호의 중요한 latent feature를 추출하고, LSTM이 시간적 의존성을 학습한다는 구성은 생체 시계열 처리 관점에서 합리적이다. 특히 motion artifact가 포함될 가능성이 있는 wrist PPG 데이터를 다룬다는 점은 웨어러블 응용과 잘 연결된다.

그러나 보고된 성능 수치가 매우 낮기 때문에, 평가 프로토콜의 엄격성이 더 중요하다. SBP MAE 1.05, DBP MAE 0.92는 실제 cuff-less BP estimation 분야에서 상당히 우수한 수치이다. 만약 subject-wise split 없이 segment-wise split을 사용했다면, 모델이 피험자별 또는 세션별 패턴을 기억했을 가능성이 있다. 또한 overlapping 또는 유사 segment가 train/test에 동시에 포함되었을 경우 성능이 과대평가될 수 있다.

따라서 이 논문은 모델 아이디어 측면에서는 유망하지만, 제공된 텍스트만으로는 결과를 임상적 수준으로 신뢰하기 어렵다. 후속 검증에서는 subject-wise 또는 leave-one-subject-out 평가, 외부 데이터셋 검증, 명확한 reference BP label 설명, AAMI/BHS 기준 평가, 실제 웨어러블 환경에서의 실시간 검증이 필요하다.

## 6. 결론

이 논문은 PPG 신호를 이용하여 SBP와 DBP를 연속적으로 추정하기 위한 deep learning 접근법을 비교하고, 그중 Autoencoder-LSTM이 가장 우수한 성능을 달성한다고 보고한다. 연구는 PPG waveform의 full cycle을 사용하여 혈압과 관련된 형태학적 및 시간적 정보를 학습하려고 하며, TCN, LSTM, TCN-LSTM, Autoencoder-LSTM을 후보 모델로 제시한다.

제공된 텍스트 기준으로 가장 중요한 결과는 Autoencoder-LSTM이 SBP MAE 1.05, DBP MAE 0.92, SBP SD 1.89, DBP SD 1.05를 달성했다는 점이다. 이는 PPG 기반 cuff-less BP estimation에서 매우 높은 정확도에 해당한다. Autoencoder가 입력 PPG의 압축된 latent representation을 학습하고, LSTM이 시간적 의존성을 반영한다는 조합은 생체신호 처리에 적합한 설계로 볼 수 있다.

이 연구는 웨어러블 센서를 활용한 연속 혈압 모니터링의 가능성을 뒷받침한다. 특히 운동 중 wrist PPG처럼 잡음이 포함될 수 있는 환경에서 PPG 기반 혈압 추정을 시도한다는 점은 실제 응용 측면에서 의미가 있다. 그러나 논문의 핵심 결과를 강하게 주장하기 위해서는 혈압 label의 출처, subject-wise data split 여부, 모델 세부 구조, 전체 비교 결과, 임상 기준 평가가 더 명확해야 한다.

종합적으로 이 논문은 PPG 기반 연속 혈압 추정에서 Autoencoder-LSTM 구조의 가능성을 제시하는 응용 연구로 평가할 수 있다. 다만 제공된 텍스트만으로는 재현성과 평가 엄격성에 대한 정보가 부족하므로, 결과를 해석할 때는 data leakage 가능성과 reference label 정의의 불명확성을 반드시 고려해야 한다.
