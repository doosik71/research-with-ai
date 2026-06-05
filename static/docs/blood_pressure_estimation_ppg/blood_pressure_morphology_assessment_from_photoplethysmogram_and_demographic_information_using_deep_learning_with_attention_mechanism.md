blood_pressure_morphology_assessment_from_photoplethysmogram_and_demographic_information_using_deep_learning_with_attention_mechanism.md

# Blood Pressure Morphology Assessment from Photoplethysmogram and Demographic Information Using Deep Learning with Attention Mechanism

* **저자**: Nicolas Aguirre, Edith Grall-Maës, Leandro J. Cymberknop, Ricardo L. Armentano
* **발표연도**: 2021

## 1. 논문 개요

이 논문은 손가락에서 측정한 photoplethysmogram, 즉 PPG 신호와 피험자의 demographic information, 즉 DI를 이용하여 arterial blood pressure morphology, 즉 ABP 파형의 평균 pulse morphology를 추정하는 deep learning 기반 방법을 제안한다. 기존 cuff 기반 혈압 측정은 systolic blood pressure와 diastolic blood pressure, 즉 SBP와 DBP 두 값만 제공하고 연속적인 arterial blood pressure waveform을 제공하지 못한다. 반면 invasive arterial catheter 방식은 연속 ABP 파형을 직접 측정할 수 있어 gold standard로 여겨지지만, 출혈과 감염 위험이 있어 일반적인 건강 모니터링이나 wearable device에는 적합하지 않다.

논문의 핵심 목표는 단순히 SBP와 DBP 값을 회귀하는 것이 아니라, PPG 신호를 입력으로 받아 average arterial blood pressure pulse morphology, 즉 ABPM 전체 파형을 생성하는 것이다. 생성된 ABPM으로부터 DBP, dicrotic notch, SBP, pulse duration, dicrotic notch time occurrence 같은 생리학적으로 의미 있는 marker를 추출할 수 있다. 이 점에서 본 연구는 혈압을 두 개의 scalar 값으로만 추정하는 다수의 cuffless BP estimation 연구보다 더 풍부한 정보를 제공하려는 시도이다.

연구 문제는 “PPG 신호만으로, 또는 PPG 신호와 age 및 gender 정보를 함께 사용하여, invasive ABP로부터 얻을 수 있는 평균 arterial pressure pulse morphology를 얼마나 정확하게 재구성할 수 있는가”이다. 저자들은 MIMIC-III Matched Waveform Database와 MIMIC-III Clinical Database를 결합하여 1131명의 subject로부터 6478개의 segment를 구성하고, 같은 subject의 데이터가 train과 test에 동시에 들어가지 않는 설정을 기본적으로 평가한다. 이는 subject leakage로 인해 성능이 과대평가되는 것을 줄이기 위한 설계이다.

이 문제는 임상적으로 중요하다. ABP morphology는 단순한 SBP와 DBP보다 더 많은 cardiovascular information을 포함한다. 예를 들어 dicrotic notch는 aortic valve closure와 관련되어 ejection period와 diastolic phase의 시작을 판단하는 데 쓰일 수 있고, 파형의 모양은 arterial stiffness, wave reflection, vascular aging, diastolic dysfunction 등과 관련될 수 있다. 따라서 PPG 기반으로 ABP morphology를 비침습적으로 추정할 수 있다면, wearable device나 mobile health 환경에서 더 풍부한 cardiovascular monitoring이 가능해질 수 있다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 PPG와 ABP가 시간 및 주파수 영역에서 형태적 유사성을 가지므로, PPG sequence를 ABP pulse morphology sequence로 변환하는 sequence-to-sequence 문제로 모델링할 수 있다는 것이다. 기존의 많은 cuffless BP 연구는 PPG에서 handcrafted feature를 추출한 뒤 SBP와 DBP를 회귀하거나, PPG와 ECG를 함께 이용해 pulse transit time 또는 pulse arrival time을 계산한다. 하지만 이 접근은 feature extraction 품질에 민감하고, ECG와 PPG를 동시에 측정해야 하거나, subject-specific calibration이 필요할 수 있다.

본 논문은 이러한 문제를 줄이기 위해 raw PPG signal과 그 1차 derivative인 PPG’를 deep learning model에 직접 입력한다. 모델은 encoder-decoder 구조를 사용하며, natural language processing에서 사용되는 seq2seq architecture와 attention mechanism을 생체신호 변환 문제에 적용한다. 입력 sequence의 어느 부분이 출력 ABP morphology의 특정 point를 생성하는 데 중요한지를 attention weight로 학습하게 하여, PPG waveform과 ABP waveform 사이의 복잡한 시간적 대응 관계를 모델링한다.

또 하나의 중요한 아이디어는 demographic information을 모델에 통합하는 것이다. 저자들은 age와 gender가 혈압 파형에 영향을 줄 수 있다고 보고, decoder의 각 time-step에서 출력 정보와 함께 DI vector를 결합한다. 실제 결과에서도 DI를 사용한 Mixno + DI scenario가 DI를 사용하지 않은 Mixno scenario보다 DBP, SBP, ABPM value error에서 약간 더 좋은 성능을 보였다. 이는 PPG morphology와 혈압 사이의 mapping이 완전히 보편적인 함수가 아니라, subject의 생리적 특성에 따라 달라질 수 있음을 시사한다.

또한 이 논문은 calibration-free와 calibration-based 설정을 구분한다. Mixno와 Mixno + DI는 같은 subject의 segment가 train과 test에 동시에 들어가지 않도록 한 설정이므로 calibration-free에 가깝다. 반면 Mixyes + DI는 segment 단위로 train/test를 나누어 같은 subject의 데이터가 양쪽에 포함될 수 있으므로 calibration-based에 해당한다. 예상대로 Mixyes + DI에서 성능이 가장 좋지만, 저자들은 Mixno + DI 결과를 더 현실적인 subject-independent 성능으로 해석한다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

논문의 전체 파이프라인은 preprocessing, processing, deep learning, evaluation의 네 단계로 구성된다. 먼저 MIMIC-III Matched Waveform Database에서 ABP와 finger PPG가 모두 존재하는 record를 선택한다. 이후 MIMIC-III Clinical Database에서 해당 subject의 age와 gender를 가져온다. Age는 18세에서 89세 사이로 제한된다.

Processing 단계에서는 신호 품질을 엄격하게 확인한다. 각 record에서 15초 segment 두 개를 선택하되, 두 segment 사이에는 5분의 간격을 둔다. 이는 subject가 안정 상태에 있다고 가정하고, 한 record에서 너무 가까운 segment만 선택되는 것을 피하기 위한 설계이다. Segment가 품질 기준을 통과하지 못하면 1분 뒤의 새로운 구간을 다시 분석한다. 한 record가 끝까지 기준을 만족하지 못하면 제외된다.

품질 검사를 통과한 뒤에는 raw PPG segment와 평균 ABP pulse morphology, 즉 ABPM이 paired data로 저장된다. 이후 deep learning model은 5초 길이의 random PPG window를 입력으로 받아 fixed-size target ABPM sequence를 예측한다. 최종적으로 예측된 $\widehat{ABPM}$으로부터 DBP, dicrotic notch, SBP, dicrotic notch time occurrence, pulse duration, waveform similarity 등을 평가한다.

### 3.2 데이터 처리와 ABPM 생성

저자들은 MIMIC-III MWDB에서 최소 15분 이상의 길이를 가지고 ABP와 PPG가 모두 존재하는 record를 선택했다. 이후 WFDB Toolbox for Matlab으로 record를 로드하고, 두 개의 15초 segment를 분석했다. 각 segment는 Flat, Peak, PPG-SQ, ABP-SQ 단계를 거친다.

Flat 단계는 null data를 탐지하고, Peak 단계는 신호의 valley와 peak에서 saturation point를 탐지한다. PPG 신호에는 cutoff frequency가 0.5 Hz와 8 Hz인 Butterworth filter가 적용되고, MinMax normalization이 수행된다. 이후 Li et al.의 arterial blood pressure waveform delineator와 관련된 marker를 이용해 pulse-by-pulse analysis를 수행한다. 원문에서 PPG-SQ는 Slapničar et al.의 feature extraction 일부에서 영감을 받았지만, 본 논문에서는 예측 feature로 쓰는 것이 아니라 signal quality filtering 목적으로만 사용된다.

ABP pulse의 평균 morphology를 만들기 위해, 각 ABP pulse를 onset 기준으로 동기화한다. 이후 각 time-step $t=i\Delta t$에서 ABP pulse들의 평균 $\mu_{ABPM_i}$와 표준편차 $\sigma_{ABPM_i}$를 계산한다. 최종 ABPM은 각 time-step에서 $\mu_{ABPM_i} \pm 1.25\sigma_{ABPM_i}$ 범위 안에 있는 point들만 사용하여 계산된다. 이 방식은 outlier pulse가 평균 파형을 왜곡하지 않도록 하기 위한 것이다.

각 ABPM point는 cardiac cycle stage에 따라 class label도 부여된다. 세 기본 class는 onset에서 systolic peak까지의 구간 $C[O,SP]$, systolic peak에서 dicrotic notch까지의 구간 $C[SP,DN]$, dicrotic notch에서 end까지의 구간 $C[DN,E]$이다. 이후 target sequence를 fixed length로 만들 때 반복되어 추가된 point에는 ended class인 $C[ED]$가 부여된다.

품질 기준도 비교적 엄격하다. Pulse duration은 0.5초에서 1.5초 사이로 제한되고, segment당 pulse 수는 10개에서 30개 사이여야 한다. $SBP-DBP$ 차이는 10 mmHg보다 커야 하며, moment coefficient of skewness는 0보다 커야 한다. 추가적인 visual inspection 이후 ABP 값은 180 mmHg 이하, pulse duration은 1.2초 이하, skewness는 0.2보다 큰 경우만 허용했다.

초기에는 1131명의 subject에서 10,696 segment가 남았지만, 169명의 subject가 전체 segment의 50% 이상을 차지하는 bias가 있었다. 이를 줄이기 위해 subject당 segment 수를 최대 10개로 제한했다. 최종 dataset은 1131명의 subject와 6478개의 segment로 구성되며, 333명의 subject가 segment의 50%를 구성한다. 평균 age는 $58.6 \pm 14.1$세, 평균 DBP는 $64.48 \pm 9.51$ mmHg, 평균 SBP는 $130.84 \pm 20.27$ mmHg이다. Gender는 female 464명, male 667명이다.

Deep learning 입력으로 사용되기 전에 raw PPG segment에는 cutoff frequency가 0.5 Hz와 45 Hz인 band-pass Butterworth filter가 적용된다. 또한 Savitzky–Golay filter를 사용하여 PPG’가 계산된다. Window size는 7이고 polynomial degree는 3이다. Filter artifact를 피하기 위해 segment의 시작 1초와 끝 1초를 제거한다. 따라서 deep learning stage에서 사용되는 데이터는 6478개의 13초 segment, 총 23.4시간에 해당한다.

### 3.3 Seq2seq encoder-decoder 구조

모델은 recurrent neural network 기반 seq2seq encoder-decoder 구조를 사용한다. Encoder는 입력 sequence를 읽어 hidden state representation으로 변환하고, decoder는 이 representation을 바탕으로 output sequence인 $\widehat{ABPM}$을 한 time-step씩 생성한다. 본 논문에서 사용된 recurrent unit은 gated recurrent unit, 즉 GRU이다.

GRU는 일반 RNN의 vanishing gradient 및 exploding gradient 문제를 완화하기 위해 reset gate와 update gate를 사용하는 구조이다. 각 time-step에서 입력은 현재 입력 $x_t$와 이전 hidden state $h_{t-1}$이며, 출력은 현재 hidden state $h_t$이다. 원문에서 GRU의 계산은 다음과 같이 정의된다.

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t]) = \sigma(W_{hz}h_{t-1} + W_{tz}x_t)
$$

$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t]) = \sigma(W_{hr}h_{t-1} + W_{tr}x_t)
$$

$$
\tilde{h}*t = \tanh(W_h \cdot [r_t \otimes h*{t-1}, x_t]) = \tanh(W_{hh}(r_t \otimes h_{t-1}) + W_{th}x_t)
$$

$$
h_t = (1-z_t)\otimes h_{t-1} + z_t \otimes \tilde{h}_t
$$

여기서 $z_t$는 update gate, $r_t$는 reset gate이다. $z_t$는 이전 hidden state를 얼마나 유지하고 새로운 candidate hidden state를 얼마나 반영할지를 결정한다. $r_t$는 과거 정보를 candidate hidden state 계산에 얼마나 사용할지 조절한다. $\sigma$는 logistic sigmoid function이고, $\tanh$는 hyperbolic tangent function이며, $\otimes$는 element-wise multiplication이다.

이 구조를 혈압 파형 예측 관점에서 해석하면, encoder는 PPG와 PPG’의 시간적 형태를 순차적으로 읽으면서 pulse morphology 정보를 hidden state에 저장한다. Decoder는 이 정보를 사용하여 ABP 평균 pulse의 각 point를 생성한다. 일반적인 scalar BP regression과 달리, 출력은 time-series 형태이므로 모델은 압력 값뿐 아니라 waveform의 시간적 구조까지 학습해야 한다.

### 3.4 입력과 target sequence 구성

모델 입력은 5초 길이의 random window signal이다. 입력 channel은 PPG와 PPG’이다. 두 신호는 on-the-fly로 독립적으로 $[0,1]$ 범위로 scaling된다. Target인 $ABPM_v$도 $[0,1]$로 scaling되지만, 이 경우에는 전체 ABPM dataset의 global minimum과 maximum을 기준으로 한다.

ABPM은 subject와 pulse마다 duration이 다를 수 있으므로, 고정 길이 target sequence를 만들기 위한 homogenization이 필요하다. 가장 긴 ABPM duration에 0.12초, 즉 15 time-step을 더하여 fixed length를 1.312초, 즉 164 time-step으로 설정한다. 각 ABPM은 fixed length에 도달할 때까지 반복된다. 반복되어 추가된 point에는 새로운 class $C[ED]$가 부여된다. 따라서 class는 총 네 개가 된다: $C[O,SP]$, $C[SP,DN]$, $C[DN,E]$, $C[ED]$.

다만 모델이 반복된 target 부분의 pressure value까지 불필요하게 맞추도록 강제하면 학습이 왜곡될 수 있다. 그래서 저자들은 mask vector를 만들어 $\widehat{ABPM}_v$의 error는 실제 ABPM 구간과 0.12초 추가 구간까지만 penalize한다. 반면 $\widehat{ABPM}_c$의 class prediction은 전체 fixed-size target signal에 대해 penalize한다. 이는 모델이 실제 pulse가 끝난 뒤 $C[ED]$ class를 제대로 예측하도록 하기 위해서이다.

### 3.5 Attention mechanism과 demographic information 통합

제안 모델은 encoder, decoder, attention module로 구성된다. Encoder는 세 개의 bidirectional GRU layer로 이루어져 있고, decoder는 세 개의 GRU layer와 두 개의 multiperceptron layer, 즉 $MPL_v$와 $MPL_c$로 이루어진다. Encoder와 decoder 모두 dense connection을 사용하여 layer 간 information flow를 개선한다.

입력 $X_l$은 5초 길이의 PPG 및 PPG’ sequence이다. Encoder의 전체 output $h_s$는 attention module로 전달된다. 또한 각 encoder GRU layer의 마지막 hidden state $h_{sL}$는 해당 decoder GRU layer의 initial hidden state로 사용된다. Decoder의 마지막 GRU layer output $h_{t_i}$는 attention module과 $MPL_c$ layer에 전달된다. Attention module에서 얻은 context vector $c_i$와 $h_{t_i}$는 concatenate되어 $MPL_v$로 들어간다. $MPL_v$는 pressure value $\widehat{ABPM}_v$를 예측하고, $MPL_c$는 class $\widehat{ABPM}_c$를 예측한다.

본 논문에서 사용된 attention은 Luong Attention이다. Context vector $c_i$는 attention weight vector $a_i$와 decoder hidden state 사이의 가중합으로 표현된다.

$$
c_i = \sum_{i=0}^{T} a_i h_{t_i}
$$

Attention weight $a_i$는 softmax function으로 정규화된다.

$$
a_i = \frac{\exp(score(h_{t_i}, h_s))}{\sum_{s' \in s}\exp(score(h_{t_i}, h_{s'}))}
$$

Score function은 general context-based function으로 정의된다.

$$
score(h_{t_i}, h_s)=h_{t_i}^{>}Wh_s
$$

여기서 $W$는 학습 가능한 weight matrix이다. 이 attention mechanism은 decoder가 ABPM의 특정 time-step을 생성할 때 encoder output 중 어떤 PPG time-step에 집중해야 하는지 학습하게 한다. 논문 Figure 7은 attention weight heatmap을 통해 PPG/PPG’ 입력과 출력 ABPM point 간의 대응 관계를 시각화한다.

Demographic information은 decoder의 각 time-step에서 입력에 결합된다. 구체적으로 age와 gender로 구성된 $X_{DI}$ vector가 decoder의 이전 출력 $y_i$와 concatenate되어 다음 decoder input $y_{i\&DI}$를 구성한다. 첫 decoder input인 $y_0$는 prediction의 시작을 나타내기 위해 1로 채워진 vector이다. 이 설계는 subject의 age와 gender가 전체 출력 파형 생성 과정에 지속적으로 영향을 줄 수 있도록 한다.

### 3.6 Multi-task loss function

모델은 두 가지 출력을 동시에 학습한다. 첫 번째 출력은 ABPM의 pressure value인 $\widehat{ABPM}_v$이고, 두 번째 출력은 각 time-step의 cardiac cycle class인 $\widehat{ABPM}_c$이다. 이는 regression과 classification을 동시에 수행하는 multitask learning 문제로 볼 수 있다.

Pressure value error에는 mean squared error, 즉 MSE가 사용된다.

$$
MSE = \frac{1}{N}\sum_{j=1}^{N}\frac{1}{M}\sum_{i=1}^{M}(ABPM^v_{ji}-\widehat{ABPM}^v_{ji})^2
$$

여기서 $N$은 sample 수, $M$은 mask가 적용되는 length이다. 즉 pressure value loss는 반복되어 추가된 모든 fixed-length 구간 전체에 대해 계산되지 않고, 실제 pulse와 약간의 여유 구간에 대해서만 계산된다.

Class prediction error에는 categorical cross-entropy, 즉 CE가 사용된다.

$$
CE = \frac{1}{N}\frac{1}{T}\sum_{j=1}^{N}\sum_{i=1}^{T}ABPM^c_{ji}\log(\widehat{ABPM}^c_{ji})
$$

여기서 $T$는 fixed input length이다. 원문 식은 일반적인 cross-entropy의 음수 부호를 생략한 형태로 표기되어 있으나, 실제 학습에서는 class mismatch를 penalize하는 categorical cross-entropy로 이해하는 것이 적절하다.

최종 training loss는 다음과 같다.

$$
Loss_{train}=MSE+\lambda CE
$$

여기서 $\lambda$는 classification loss의 가중치이며, 실험적으로 0.01로 설정되었다. 이 설계는 pressure waveform value를 정확히 맞추는 것을 주목표로 하면서도, 각 time-step이 cardiac cycle의 어느 단계에 속하는지를 함께 학습하도록 만든다. Class prediction은 dicrotic notch time occurrence와 pulse duration을 추출하는 데 직접 사용된다.

### 3.7 Hyperparameter와 실험 설정

Encoder의 Bi-GRU unit 수는 layer별로 4, 20, 100이다. Decoder의 GRU unit 수는 8, 40, 200이다. $MPL_v$는 1 unit, $MPL_c$는 4 unit을 가지며, activation function으로 ELU가 사용된다. $MPL_v$의 출력은 $\widehat{ABPM}_v$, $MPL_c$의 출력은 $\widehat{ABPM}_c$에 대응한다.

Optimizer는 Adam이고 learning rate는 $10^{-3}$이다. Loss가 25 epoch 동안 개선되지 않으면 learning rate를 50% 줄이고, patience가 50 epoch에 도달하면 training을 중단한다. Batch size는 48이다.

Weight initialization도 명시되어 있다. $MPL_v$, $MPL_c$, attention layer의 weight는 $U(-\sqrt{w},\sqrt{w})$에서 초기화되고, GRU layer의 weight는 $U(-\sqrt{k},\sqrt{k})$에서 초기화된다.

$$
w=\frac{1}{\text{\#Layer\ input\ size}}, \quad
k=\frac{1}{\text{\#GRU\ units}}
$$

GRU의 transition matrix weight인 $W_{hz}$, $W_{hr}$, $W_{hh}$에는 random orthogonal initialization이 사용된다.

### 3.8 세 가지 평가 scenario

저자들은 세 가지 scenario를 통해 DI의 효과와 subject split의 효과를 평가했다.

첫 번째는 Mixno이다. 같은 subject의 segment가 train과 test에 동시에 들어가지 않도록 제한하고, DI를 사용하지 않는다. 이는 PPG와 PPG’만으로 calibration-free 성능을 평가하는 scenario이다.

두 번째는 Mixno + DI이다. 같은 subject가 train/test에 동시에 들어가지 않도록 유지하면서, age와 gender를 입력에 추가한다. 이 scenario는 subject-independent setting에서 demographic information이 도움이 되는지 평가한다.

세 번째는 Mixyes + DI이다. Train/test split을 segment 단위로 수행하여 같은 subject의 segment가 train과 test에 동시에 포함될 수 있고, DI도 사용한다. 이는 calibration-based setting에 가까우며, subject-specific information이 train에 포함될 때 성능이 얼마나 좋아지는지 보여준다.

Mixno와 Mixno + DI에서는 subject의 20%가 test set을 구성한다. Mixyes + DI에서는 전체 segment의 20%가 test set을 구성한다. 각 scenario는 5-fold cross-validation으로 평가된다.

### 3.9 평가 지표

예측된 $\widehat{ABPM}*v$는 global minimum과 maximum을 이용하여 원래 ABP scale로 복원된다. 이후 $\widehat{ABPM}$에서 DBP, DN, SBP를 계산한다. $\widehat{ABPM}*{DBP}$는 $\widehat{ABPM}*v$의 첫 번째 값과 마지막 값의 평균으로 계산한다. $\widehat{ABPM}*{DN}$은 $\widehat{ABPM}*c$에서 $C[SP,DN]$ class의 마지막 occurrence로 간주한다. $\widehat{ABPM}*{SBP}$는 $\widehat{ABPM}_v$의 maximum value로 계산한다.

DBP, DN, SBP 평가는 RMSE, MAE, STD, $R^2$를 사용한다. RMSE와 MAE는 다음과 같다.

$$
RMSE=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(z_i-\widehat{z}_i)^2}
$$

$$
MAE=\frac{1}{N}\sum_{i=1}^{N}|z_i-\widehat{z}_i|
$$

$R^2$는 원문에 다음과 같은 형태로 제시되어 있으나, 일반적인 coefficient of determination의 분모와 분자는 squared residual sum을 사용한다. 원문 표기에는 제곱 항이 빠져 있는 것으로 보이므로, 보고서에서는 원문 수치 해석에만 사용하고 식 자체의 표기 오류 가능성을 언급하는 것이 타당하다.

Dicrotic notch time occurrence와 pulse duration은 RMSE, MAE, $R^2$로 평가된다. DNTO는 $\widehat{ABPM}_c$에서 $C[DN,E]$ class의 마지막 occurrence로 계산하고, pulse duration은 $C[ED]$ class의 첫 occurrence로 계산한다.

Waveform 자체는 RMSE, MAE, Pearson correlation coefficient $R$로 평가된다. $R$은 다음과 같이 정의된다.

$$
R=\frac{\sum_{i=1}^{T}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^{T}(x_i-\bar{x})^2(y_i-\bar{y})^2}}
$$

여기서 $x$와 $y$는 각각 실제 ABPM과 예측 ABPM이다. 원문 식은 분모에서 두 신호의 제곱합 곱을 한 번에 표기했으나, 일반적인 Pearson correlation은 각 신호의 centered squared sum의 곱의 제곱근을 사용한다. 실험 결과 해석에는 원문 표의 $R$ 값을 따른다.

## 4. 실험 및 결과

### 4.1 정성적 결과: Attention과 ABPM waveform 생성

논문 Figure 7은 Mixno + DI scenario에서 하나의 test example을 보여준다. 입력 PPG 및 PPG’ signal, attention weight heatmap, 실제 ABPM과 예측 $\widehat{ABPM}$이 함께 제시된다. Heatmap은 모델이 출력 ABPM의 각 point를 생성할 때 입력 PPG sequence의 어느 부분에 attention을 두는지를 보여준다.

Figure 8은 다양한 morphology를 가진 여러 test sample의 실제 ABPM과 예측 ABPM을 비교한다. 저자들은 이 그림을 통해 모델이 단순히 global average morphology만 학습한 것이 아니라, 서로 다른 형태의 ABP pulse를 어느 정도 구분해 생성할 수 있음을 확인한다. 이는 본 연구의 중요한 정성적 근거이다. 만약 모델이 모든 subject에게 비슷한 평균 파형만 출력한다면 waveform correlation은 일부 높게 나올 수 있어도 실제 morphology assessment에는 의미가 약할 수 있다. Figure 8은 적어도 일부 예시에서 모델이 morphology 차이를 반영한다는 점을 보여준다.

### 4.2 DBP, DN, SBP 추정 결과

Table 1은 DBP, dicrotic notch, SBP marker에 대한 결과를 보여준다. 세 scenario 중 전반적으로 Mixyes + DI가 가장 좋은 성능을 보이고, 그 다음이 Mixno + DI, 마지막이 Mixno이다.

DBP의 경우 Mixno는 MAE $7.01 \pm 0.23$ mmHg, RMSE $8.88 \pm 0.27$ mmHg, STD $8.84 \pm 0.25$ mmHg, $R^2$ $0.10 \pm 0.03$이다. Mixno + DI에서는 MAE가 $6.57 \pm 0.20$ mmHg로 감소하고, RMSE도 $8.47 \pm 0.29$ mmHg로 개선된다. $R^2$도 $0.19 \pm 0.04$로 증가한다. 이는 age와 gender 정보가 calibration-free DBP estimation에 도움이 되었음을 의미한다. Mixyes + DI에서는 MAE가 $5.56 \pm 0.18$ mmHg, RMSE가 $7.40 \pm 0.20$ mmHg, $R^2$가 $0.41 \pm 0.04$로 더 좋아진다.

Dicrotic notch의 pressure value 추정에서도 비슷한 경향이 있다. Mixno MAE는 $8.72 \pm 0.31$ mmHg, Mixno + DI MAE는 $8.54 \pm 0.37$ mmHg, Mixyes + DI MAE는 $7.08 \pm 0.19$ mmHg이다. $R^2$도 각각 $0.29 \pm 0.02$, $0.32 \pm 0.04$, $0.50 \pm 0.02$로 증가한다. Dicrotic notch는 waveform morphology에서 중요한 위치이므로, 이 결과는 모델이 단순 peak만이 아니라 중간 morphology marker도 어느 정도 추정할 수 있음을 보여준다.

SBP는 가장 어려운 target으로 나타난다. Mixno에서 SBP MAE는 $14.55 \pm 0.56$ mmHg이고 RMSE는 $18.20 \pm 0.52$ mmHg이다. Mixno + DI에서는 MAE가 $14.39 \pm 0.42$ mmHg로 약간 개선되지만 큰 차이는 아니다. Mixyes + DI에서는 MAE가 $12.08 \pm 0.36$ mmHg, RMSE가 $15.96 \pm 0.60$ mmHg로 상당히 개선된다. 하지만 calibration-free setting에서 SBP error가 여전히 크다는 점은 본 방법의 주요 한계이다. DBP보다 SBP의 variation range가 크고 reflected wave, arterial stiffness, peripheral amplification 등의 영향을 더 많이 받기 때문에 PPG만으로 정확히 추정하기 어려운 것으로 해석할 수 있다.

### 4.3 Dicrotic notch time occurrence와 pulse duration 결과

Table 2는 dicrotic notch time occurrence, 즉 DNTO와 pulse duration의 시간적 추정 결과를 보여준다. DNTO의 경우 Mixno는 MAE $24 \pm 2$ ms, RMSE $33 \pm 3$ ms, $R^2$ $0.55 \pm 0.10$이다. Mixno + DI는 MAE $25 \pm 1$ ms, RMSE $35 \pm 3$ ms, $R^2$ $0.54 \pm 0.05$로, DI 추가가 시간적 marker에는 뚜렷한 개선을 주지 않았다. Mixyes + DI는 MAE $23 \pm 1$ ms, RMSE $33 \pm 1$ ms, $R^2$ $0.61 \pm 0.02$로 약간 더 좋다.

Pulse duration의 경우 성능은 매우 높다. Mixno와 Mixno + DI 모두 $R^2$가 $0.97$이고, Mixyes + DI는 $0.98$이다. Mixno의 MAE는 $15 \pm 9$ ms, Mixno + DI는 $16 \pm 4$ ms, Mixyes + DI는 $11 \pm 1$ ms이다. Pulse duration은 PPG waveform에서 비교적 직접적으로 반영되는 시간 구조이기 때문에 pressure amplitude보다 더 안정적으로 예측된 것으로 보인다.

이 결과는 모델이 ABPM의 시간적 길이나 cardiac cycle phase 구분을 상당히 잘 학습하고 있음을 보여준다. 다만 dicrotic notch의 정확한 시간 위치는 pressure value나 class boundary 모두에 의존하기 때문에 pulse duration보다 더 어렵다.

### 4.4 전체 waveform 평가 결과

Table 3은 예측 ABPM waveform 전체에 대한 평가 결과를 제시한다. 세 scenario 모두 Pearson correlation coefficient $R$은 약 $0.98$로 매우 높다. Mixno는 RMSE $10.39 \pm 0.11$ mmHg, MAE $9.06 \pm 0.09$ mmHg이다. Mixno + DI는 RMSE $10.26 \pm 0.11$ mmHg, MAE $8.89 \pm 0.10$ mmHg로 약간 개선된다. Mixyes + DI는 RMSE $8.65 \pm 0.20$ mmHg, MAE $7.37 \pm 0.21$ mmHg로 가장 좋다.

높은 $R$ 값은 예측 파형의 형태가 실제 ABPM과 매우 유사함을 의미한다. 그러나 MAE가 7–9 mmHg 수준이라는 점은 waveform shape은 잘 맞지만 absolute pressure calibration에는 여전히 error가 존재함을 의미한다. 이는 본 연구의 제목에서 “morphology assessment”를 강조한 것과도 연결된다. 모델은 ABPM의 형태를 잘 따라가지만, 임상적 혈압 수치로서 SBP와 DBP를 정확히 맞추는 데에는 특히 SBP에서 한계가 있다.

DI 사용의 효과는 waveform MAE에서도 확인된다. Mixno + DI는 Mixno보다 RMSE와 MAE가 낮다. 차이는 크지 않지만 일관되게 긍정적이다. 이는 age와 gender가 ABP morphology 예측에 보조 정보를 제공한다는 저자들의 주장을 뒷받침한다.

### 4.5 BHS 기준 평가

Table 4는 British Hypertension Society, 즉 BHS standard에 따른 cumulative error percentage를 보여준다. BHS 기준은 5, 10, 15 mmHg 이하의 error 비율에 따라 Grade A, B, C를 부여한다. Grade A는 각각 60%, 85%, 95% 이상, Grade B는 50%, 75%, 90% 이상, Grade C는 40%, 65%, 85% 이상을 요구한다.

DBP에서는 Mixno가 5 mmHg 미만 41.8%, 10 mmHg 미만 76.6%, 15 mmHg 미만 92.9%로 Grade C 수준이다. Mixno + DI는 45.5%, 80.2%, 93.5%로 역시 Grade C이지만 Grade B에 더 가까워진다. Mixyes + DI는 56.6%, 86.0%, 95.5%로 Grade B를 만족한다. 저자들은 Mixyes + DI의 DBP가 Grade A에 도달하려면 5 mmHg 미만 구간에서 3.4%가 더 필요하다고 설명한다.

SBP에서는 모든 scenario가 BHS 기준에 크게 미치지 못한다. Mixno + DI의 SBP cumulative error는 5 mmHg 미만 21.3%, 10 mmHg 미만 41.9%, 15 mmHg 미만 58.6%이다. Mixyes + DI에서도 29.6%, 53.2%, 70.3%에 그친다. 이는 제안 방법이 DBP estimation에서는 어느 정도 가능성을 보이지만, SBP estimation에서는 임상적 혈압 측정 기준을 만족하지 못한다는 것을 명확히 보여준다.

이 결과는 논문의 해석에서 중요하다. ABPM waveform correlation이 높다고 해서 SBP와 DBP 같은 clinical marker가 모두 충분히 정확하다는 뜻은 아니다. 특히 SBP는 BHS 기준에서 매우 낮은 cumulative accuracy를 보였으므로, 이 방법을 혈압계 대체 기술로 보기는 어렵고, waveform morphology estimation 및 연구용 가능성에 더 가깝게 해석해야 한다.

### 4.6 Bland–Altman 분석

Table 5는 Bland–Altman plot의 mean bias와 limits of agreement를 제시한다. DBP에서 Mixno + DI의 limits of agreement는 [-16.87, 15.90] mmHg이고 mean은 -0.49 mmHg이다. 이는 평균 bias는 거의 0에 가깝지만, 개별 sample의 오차 범위는 여전히 넓다는 의미이다. Mixyes + DI에서는 limits가 [-14.23, 14.48] mmHg이고 mean은 0.13 mmHg로 더 좁아진다.

SBP의 경우 오차 범위가 훨씬 넓다. Mixno + DI의 limits of agreement는 [-34.36, 35.60] mmHg이고 mean은 0.62 mmHg이다. 평균 bias는 작지만, 실제 개별 예측에서는 ±35 mmHg 수준의 큰 error가 발생할 수 있다. Mixyes + DI에서도 limits는 [-28.78, 32.69] mmHg로 여전히 넓다.

Figure 9는 Mixno + DI scenario의 regression plots, Bland–Altman plots, error histograms를 보여준다. DBP는 SBP보다 예측값과 target의 관계가 더 안정적이며 error distribution도 상대적으로 좁다. SBP는 regression scatter가 더 퍼져 있고 error histogram도 넓다. 이는 Table 1과 Table 5의 정량 결과와 일치한다.

### 4.7 기존 연구와의 비교

저자들은 cuffless calibration 결과를 기존 연구와 비교한다. 다만 dataset, subject 수, input signal, calibration setting, metric이 서로 달라 직접 비교는 어렵다고 명시한다.

Feature-based approach 중 Kurylyak et al.은 MIMIC의 15,000 beats를 사용하여 PPG feature와 neural network로 cal-based DBP MAE 2.21 mmHg, SBP MAE 3.80 mmHg를 보고했다. 그러나 subject 수가 명시되지 않았고, train/test subject separation이 적용되었는지 불분명하다. Chowdhury et al.은 126명의 dataset에서 GPR과 PPG 및 demographic features를 사용하여 cal-based DBP MAE 1.74 mmHg, SBP MAE 3.02 mmHg를 보고했다. 하지만 dataset 규모와 calibration 조건이 다르다.

대규모 subject를 사용하고 subject split을 명시한 연구와 비교하면 본 연구의 의의가 더 분명하다. Kachuee et al.은 MIMIC-II의 1000명을 사용하고 ECG와 PPG feature를 함께 사용하여 cal-free DBP MAE 5.35 mmHg, SBP MAE 11.17 mmHg를 보고했다. 이 성능은 본 연구의 Mixno + DI DBP MAE 6.57 mmHg, SBP MAE 14.39 mmHg보다 더 좋지만, Kachuee et al.은 ECG까지 사용하고 feature extraction이 필요하다. Slapničar et al.은 MIMIC-III의 510명을 사용하고 raw PPG 기반 deep learning을 적용해 cal-free DBP MAE 12.38 mmHg, SBP MAE 15.41 mmHg를 보고했다. 이와 비교하면 본 연구는 DBP에서는 훨씬 좋고, SBP에서도 약간 더 좋은 cal-free 성능을 보인다.

Continuous ABP waveform estimation 연구와 비교하면, Sideris et al.은 MIMIC 42명으로 LSTM 기반 personalized model을 사용해 waveform RMSE 6.04 mmHg, $R=0.95$를 보고했다. Sadrawi et al.은 18명의 closed data로 DCAE를 사용해 RMSE 3.46 mmHg, MAE 2.33 mmHg, $R=0.98$을 보고했다. 본 연구는 cal-free setting에서 RMSE 10.26 mmHg, MAE 8.89 mmHg, $R=0.98$이며, cal-based setting에서는 RMSE 8.67 mmHg, MAE 7.39 mmHg, $R=0.98$이다. Error 값은 기존 두 연구보다 크지만, subject 수가 1131명으로 훨씬 많고, subject data restriction을 고려했다는 점에서 더 엄격한 평가로 볼 수 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 혈압을 단순히 SBP와 DBP 두 값으로 예측하지 않고, average arterial blood pressure pulse morphology 전체를 생성하려 했다는 점이다. ABPM은 혈압의 형태적 정보를 담고 있으며, dicrotic notch, pulse duration, cardiac cycle phase 같은 추가 marker를 제공한다. 이 접근은 wearable 또는 cuffless monitoring에서 단순 scalar 혈압값보다 더 풍부한 cardiovascular assessment를 가능하게 할 수 있다.

두 번째 강점은 raw PPG와 demographic information을 하나의 end-to-end model 안에서 결합했다는 점이다. 기존 feature-based 연구는 PPG waveform에서 handcrafted features를 추출해야 하고, feature extraction의 품질이 성능을 좌우한다. 반면 본 연구는 PPG와 PPG’를 sequence로 입력하여 seq2seq model이 representation을 학습하게 한다. Age와 gender를 decoder time-step마다 결합함으로써 subject-specific physiological variation도 일부 반영하려 했다.

세 번째 강점은 attention mechanism의 도입이다. Attention은 출력 ABPM point가 입력 PPG sequence의 어느 부분을 참조하는지 학습한다. 이는 PPG와 ABP 사이에 단순한 point-to-point 대응이 아니라 시간적 shift와 morphology mapping이 존재하는 문제에 적합하다. Figure 7의 attention heatmap은 모델의 내부 동작을 어느 정도 해석할 수 있게 해준다.

네 번째 강점은 dataset bias를 줄이려는 노력이 명확하다는 것이다. 저자들은 subject당 segment 수를 최대 10개로 제한하여 일부 subject가 dataset을 지배하지 않도록 했다. 또한 Mixno와 Mixno + DI scenario에서는 같은 subject의 segment가 train과 test에 동시에 들어가지 않도록 하여 subject leakage를 방지했다. 이는 PPG 기반 BP estimation 연구에서 매우 중요한 평가 설계이다.

다섯 번째 강점은 reproducibility이다. 저자들은 processed dataset, subject ID, temporal information, model architecture, training source code를 공개한다고 명시했다. MIMIC-III Clinical Database 접근 제한 때문에 DI 자체는 공유하지 않지만, 접근 권한을 가진 사용자가 DI를 추출할 수 있는 code는 제공한다고 설명한다.

그러나 한계도 분명하다. 첫째, SBP estimation 성능이 낮다. Calibration-free Mixno + DI에서 SBP MAE는 $14.39 \pm 0.42$ mmHg이고, BHS cumulative error 기준에서도 15 mmHg 미만 비율이 58.6%에 그친다. 이는 임상적 혈압 측정 장치로 사용하기에는 부족하다. DBP에서는 어느 정도 가능성을 보이지만, SBP는 여전히 큰 문제이다.

둘째, waveform correlation $R=0.98$은 매우 높지만, 이는 pressure value calibration이 충분히 정확하다는 뜻은 아니다. 실제로 ABPM waveform MAE는 Mixno + DI에서 $8.89 \pm 0.10$ mmHg이고, SBP error는 더 크다. 따라서 morphology shape reconstruction과 absolute BP calibration을 구분해 해석해야 한다.

셋째, ABP가 arterial tree의 어느 부위에서 측정되었는지 명확하지 않다. 모든 PPG는 finger에서 측정되었지만, ABP signal의 구체적 측정 위치는 알려져 있지 않다. ABP morphology는 측정 위치에 따라 달라진다. Central aortic pressure와 peripheral arterial pressure는 wave reflection과 amplification 때문에 형태가 다를 수 있다. 따라서 source PPG와 target ABP의 anatomical site가 불명확한 것은 모델 학습과 해석 모두에 한계가 된다.

넷째, MIMIC 데이터의 장비, filter, 약물, 기존 질환 정보가 충분히 통제되지 않았다. ICU database는 실제 임상 데이터라는 장점이 있지만, measurement device, sensor placement, medication, pathology, treatment state 등이 다양하다. 저자들도 이 scenario가 rigorous medical protocol을 만족하지 않는다고 명시한다. 이는 모델의 generalization과 causal interpretation을 어렵게 만든다.

다섯째, DI로 age와 gender만 사용했다. 저자들은 ethnicity, weight, height, diabetes, chronic kidney disease, smoking, dyslipidemia 같은 정보도 모델에 통합할 수 있다고 언급하지만, 좋은 quality record와 extra information을 모두 가진 subject 수가 전체의 30% 미만이어서 사용하지 않았다. 혈압 morphology는 이러한 변수들과 강하게 관련될 수 있으므로, 더 풍부한 DI가 있었다면 성능 개선 가능성이 있다.

여섯째, 논문에서 일부 수식 표기에 오류 가능성이 있다. 예를 들어 $R^2$ 식은 일반적인 coefficient of determination 형태와 다르게 제시되어 있으며, categorical cross-entropy도 일반적인 음수 부호가 생략된 형태로 보인다. 실험 구현에서는 올바르게 사용되었을 가능성이 있지만, 원문 수식 자체는 주의해서 해석해야 한다.

일곱째, Mixyes + DI 성능은 가장 좋지만 calibration-based 또는 subject leakage 가능성이 있는 setting이다. 같은 subject의 segment가 train과 test에 동시에 존재할 수 있으므로, 이 결과는 신규 subject에 대한 일반화 성능으로 해석하기 어렵다. 실제 wearable device 적용에서 사용자가 초기 calibration을 수행하는 시나리오라면 의미가 있지만, 완전한 calibration-free model로 주장할 수는 없다.

## 6. 결론

이 논문은 finger PPG와 demographic information을 이용하여 average arterial blood pressure pulse morphology를 추정하는 seq2seq attention 기반 deep learning framework를 제안했다. 모델은 PPG와 PPG’를 입력으로 받아 $\widehat{ABPM}$을 생성하고, 그로부터 DBP, dicrotic notch, SBP, dicrotic notch time occurrence, pulse duration 같은 marker를 추출한다. 단순 scalar BP regression이 아니라 ABP waveform morphology 자체를 생성한다는 점이 본 연구의 가장 중요한 기여이다.

실험 결과, calibration-free에 해당하는 Mixno + DI scenario에서 DBP MAE는 $6.57 \pm 0.20$ mmHg, SBP MAE는 $14.39 \pm 0.42$ mmHg였고, ABPM waveform MAE는 $8.89 \pm 0.10$ mmHg, Pearson correlation은 $0.98 \pm 0.001$이었다. DI를 사용하면 DI를 사용하지 않은 경우보다 전반적으로 성능이 개선되었고, 같은 subject의 segment가 train과 test에 동시에 포함될 수 있는 Mixyes + DI에서는 가장 좋은 성능을 보였다. 이는 demographic information과 subject-specific calibration이 PPG-to-ABP mapping에 도움이 된다는 점을 시사한다.

다만 본 방법은 DBP와 waveform morphology에서는 가능성을 보였지만, SBP estimation에서는 임상 기준을 만족하지 못했다. BHS standard 기준으로 DBP는 Mixyes + DI에서 Grade B에 도달했지만, SBP는 모든 scenario에서 기준에 크게 미달했다. 따라서 이 방법은 현재 단계에서 cuff-based sphygmomanometer를 대체할 수 있는 완성된 혈압 측정 기술이라기보다, PPG 기반 ABP morphology reconstruction을 위한 유망한 연구 방향으로 보는 것이 적절하다.

향후 연구에서는 ABP 측정 위치가 명확한 dataset, 더 풍부한 demographic and clinical information, sensor 및 filter 조건의 통제, disease state와 medication 정보의 반영, 더 큰 subject-independent validation이 필요하다. 또한 SBP calibration을 개선하기 위해 subject adaptation, transfer learning, domain-invariant representation, uncertainty estimation 등을 결합할 필요가 있다.

종합하면, 이 논문은 PPG 기반 cuffless BP 연구에서 “혈압값 예측”을 넘어 “혈압 파형 morphology 추정”으로 문제를 확장한 의미 있는 연구이다. 특히 seq2seq attention architecture와 demographic information 통합은 향후 wearable device 기반 cardiovascular monitoring 연구에 참고할 가치가 크다. 그러나 clinical deployment를 위해서는 SBP 정확도와 subject-independent generalization을 크게 개선해야 한다.
