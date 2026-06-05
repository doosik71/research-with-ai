# A Benchmark Study of Machine Learning for Analysis of Signal Feature Extraction Techniques for Blood Pressure Estimation Using Photoplethysmography (PPG)

* **저자**: Sumbal Maqsood, Shuxiang Xu, Matthew Springer, Rami Mohawesh
* **발표연도**: 2021

## 1. 논문 개요

이 논문은 photoplethysmography, 즉 PPG 신호를 이용한 cuffless blood pressure estimation에서 어떤 feature extraction technique과 machine learning model이 더 적합한지를 비교 분석한 benchmark study이다. 논문의 핵심 목적은 새로운 단일 모델을 제안하는 것이라기보다, PPG 기반 혈압 추정에서 널리 사용되는 feature group과 전통적 machine learning 및 deep learning 모델을 체계적으로 비교하여 연구자들이 적절한 feature와 모델을 선택할 수 있도록 기준을 제공하는 데 있다.

연구 문제는 크게 두 가지로 정리된다. 첫째, PPG 신호에서 추출되는 전통적 feature extraction technique 중 어떤 종류의 feature가 systolic blood pressure, 즉 SBP와 diastolic blood pressure, 즉 DBP 추정에 가장 효과적인가이다. 둘째, conventional machine learning 모델과 deep learning 모델 중 어떤 계열이 혈압 추정에서 더 높은 정확도를 보이는가이다. 논문은 이를 위해 feature를 세 그룹으로 나누어 평가한다. Group A는 time-domain feature, Group B는 statistical feature, Group C는 frequency-domain feature이다.

혈압은 cardiovascular disease, 즉 CVD를 조기에 발견하고 예방하는 데 중요한 생체 지표이다. 그러나 전통적인 cuff 기반 혈압 측정은 간헐적 측정에 적합할 뿐 continuous monitoring에는 불편하고 실용성이 낮다. Invasive arterial blood pressure 측정은 정확하지만 catheter 삽입이 필요하므로 ICU나 고위험 수술 상황에 제한된다. 따라서 저비용이고 비침습적이며 wearable device에 통합하기 쉬운 PPG 기반 혈압 추정이 중요한 연구 주제가 된다.

이 논문은 PPG-BP dataset과 MIMIC-II dataset이라는 두 공개 데이터셋을 사용한다. 두 데이터셋에서 PPG 신호를 전처리하고, 서로 다른 feature extraction 기법을 적용한 뒤, linear regression, random forest, AdaBoost, support vector regression 같은 전통적 machine learning 모델과 LSTM, bidirectional LSTM, GRU 같은 deep learning 모델을 적용한다. 평가 지표로는 mean absolute error, 즉 MAE, root mean square error, 즉 RMSE, standard deviation of error, 즉 SD를 사용한다.

논문의 주요 결론은 time-domain feature가 statistical feature와 frequency-domain feature보다 전반적으로 더 신뢰성 있는 성능을 보였고, deep learning 모델이 전통적 machine learning 모델보다 더 좋은 성능을 달성했다는 것이다. 특히 time-domain feature를 사용할 때 GRU와 Bi-LSTM이 SBP와 DBP 추정에서 가장 우수한 성능을 보였다고 보고한다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 PPG 기반 혈압 추정 성능이 모델 자체뿐 아니라 입력 feature의 종류에 크게 좌우된다는 점을 benchmark 형태로 검증하는 것이다. PPG waveform은 cardiac cycle, vascular compliance, arterial stiffness, peripheral circulation 등의 정보를 간접적으로 포함하지만, PPG와 혈압 사이의 관계는 단순한 선형 관계가 아니다. 따라서 어떤 feature가 혈압 변동을 가장 잘 반영하는지, 그리고 어떤 모델이 그 feature와 혈압 사이의 복잡한 관계를 잘 학습하는지가 중요하다.

기존 cuffless BP estimation 연구에서는 ECG와 PPG를 동시에 사용하여 pulse transit time, 즉 PTT 또는 pulse arrival time, 즉 PAT를 계산하는 방식이 자주 사용되었다. 그러나 PTT와 PAT는 두 개 이상의 sensor가 필요하고, 신호 동기화가 필요하며, 개인별 physiology에 따른 calibration이 요구된다. 또한 두 신호를 동시에 측정하고 처리해야 하므로 computational complexity와 착용 부담이 증가한다. 이 논문은 이러한 배경에서 PPG-only 혈압 추정의 필요성을 강조한다.

논문은 PPG-only 접근에서 feature extraction을 세 종류로 나눈다. Time-domain feature는 PPG 한 주기 안의 systolic time, diastolic time, pulse width, pulse height, ratio 등 waveform morphology를 직접적으로 나타낸다. Statistical feature는 skewness, kurtosis, perfusion, maximum, minimum, mean absolute deviation 등 PPG segment의 통계적 분포 특성을 표현한다. Frequency-domain feature는 multitaper method를 이용해 spectral representation을 추출한다.

이 연구의 또 다른 핵심은 전통적 machine learning과 deep learning을 같은 feature group에 적용하여 비교한다는 점이다. Linear regression은 단순하고 해석이 쉽지만 nonlinear relationship을 표현하는 데 한계가 있다. Random forest와 AdaBoost는 ensemble 기반 nonlinear model이며, SVR도 nonlinear regression에 자주 사용된다. 반면 LSTM, Bi-LSTM, GRU는 순차 데이터의 temporal dependency를 학습할 수 있어 PPG waveform 기반 혈압 추정에 더 적합할 가능성이 있다.

논문의 중요한 직관은 SBP와 DBP를 독립적으로 추정하는 것보다 둘 사이의 관련성을 고려할 수 있는 sequence model 또는 shared learning 구조가 더 유리할 수 있다는 점이다. 본 논문이 multi-task learning 모델을 직접 제안하지는 않지만, DBP와 SBP가 서로 상관되어 있으므로 두 값을 동시에 또는 연관된 구조로 모델링하는 것이 중요하다는 관점을 제시한다.

## 3. 상세 방법 설명

### 3.1 전체 연구 파이프라인

연구 방법은 네 단계로 구성된다. 첫째, 공개 데이터셋을 준비한다. 둘째, PPG 신호를 전처리한다. 셋째, PPG 신호로부터 여러 종류의 feature를 추출한다. 넷째, 추출된 feature를 다양한 machine learning 및 deep learning 모델에 입력하여 SBP와 DBP를 추정하고 성능을 비교한다.

논문에서 제시한 전체 파이프라인은 다음과 같이 요약할 수 있다. 먼저 PPG signal을 scale하고 outlier를 제거한 뒤 Butterworth low-pass filtering으로 denoising한다. 이후 signal의 minimum과 maximum을 찾고, 일정한 길이로 signal size를 맞춘다. 그다음 time-domain, statistical, frequency-domain feature를 추출하고, 각 feature set을 machine learning model에 입력하여 SBP와 DBP를 예측한다.

전통적 machine learning의 경우 feature vector를 LR, RF, AdaBoost, SVR에 입력한다. Deep learning의 경우 같은 feature set을 LSTM, Bi-LSTM, GRU에 입력한다. 논문은 각 모델에 대해 parameter tuning을 수행하고, MAE, RMSE, SD를 기준으로 성능을 평가한다.

### 3.2 데이터셋

논문은 두 개의 공개 데이터셋을 사용한다. 첫 번째는 MIMIC-II dataset이다. MIMIC-II는 Beth Israel Deaconess Medical Center의 critical care unit 환자 데이터를 포함하는 공개 데이터베이스이며, BP, PPG, ECG, respiration 등 다양한 physiological signal을 제공한다. 신호들은 125 Hz sampling rate로 동시 측정된다. 논문에서는 BP와 PPG signal을 30초 window로 나누어 안정적인 BP 값을 보이는 구간을 사용했다고 설명한다.

두 번째는 PPG-BP dataset이다. 이 데이터셋은 중국 Guilin People’s Hospital에서 수집된 데이터로, 21세에서 86세 사이의 219명 환자 기록을 포함한다. 여기에는 age, height, weight, heart rate, BMI, blood pressure readings, PPG signal 등이 포함되며, hypertension과 diabetes 같은 질환 정보도 포함되어 있다. 논문은 이 두 데이터셋을 사용해 feature extraction technique과 model 성능이 특정 데이터셋에만 의존하지 않는지 비교하려 한다.

### 3.3 전처리

PPG 기반 혈압 추정에서는 원 신호에 motion artifact, baseline shift, ambient light 변화, sensor contact 문제 등이 포함될 수 있으므로 전처리가 매우 중요하다. 논문은 irregular signal segment를 제거하고 PPG와 BP 신호를 정렬하였다. 환자의 움직임은 PPG acquisition baseline에 shift를 만들 수 있으므로, 이러한 noise와 artifact를 줄이기 위해 Butterworth filter를 적용하였다.

Butterworth filter는 passband에서 가능한 한 평탄한 frequency response를 갖도록 설계된 filter이다. 논문은 이를 통해 PPG signal의 noise를 제거하고, 이후 min-max normalization을 사용해 PPG signal을 $[0,1]$ 범위로 정규화하였다. 일반적인 min-max normalization은 다음과 같이 쓸 수 있다.

$$
x_{norm}=\frac{x-\min(x)}{\max(x)-\min(x)}
$$

그 후 signal을 동일한 time frame으로 segment하고, 각 segment에서 feature를 추출한다. 이 과정은 모든 모델이 동일한 길이와 형식의 입력을 받을 수 있게 하기 위한 단계이다.

### 3.4 Group A: Time-domain feature

Group A는 PPG waveform의 시간 영역 morphology를 나타내는 feature들이다. PPG waveform은 각 cardiac cycle에서 amplitude와 duration으로 특징지을 수 있다. 논문은 기존 연구에서 널리 사용된 21개의 time-domain feature를 추출한다. 이 feature들은 systolic part와 diastolic part의 시간, pulse width, pulse height, width ratio 등 PPG 한 주기의 형태적 정보를 포함한다.

예를 들어 systolic upstroke time, diastolic time, pulse amplitude의 1/2, 2/3, 10%, 25%, 33%, 75% 수준에서의 pulse width 등이 사용된다. 이러한 feature는 PPG waveform의 상승부와 하강부가 혈관 탄성, 심박 주기, pressure wave reflection과 관련될 수 있다는 생리학적 직관에 기반한다.

MIMIC-II의 경우 ABP signal에서 SBP와 DBP ground truth를 얻는다. SBP는 ABP pulse waveform의 systolic peak이고, DBP는 end-diastolic point이다. 이렇게 얻은 reference BP와 같은 시간대의 PPG feature를 연결하여 regression dataset을 구성한다.

논문은 time-domain feature에 대해 두 가지 feature set을 구성한다. 하나는 21개 feature 전체를 사용한 set이고, 다른 하나는 feature importance 분석을 통해 선택된 11개 feature set이다. Feature importance에는 CART regression, random forest feature importance, XGBoost feature importance 등이 사용된다. 목적은 redundant feature를 제거하여 모델 복잡도를 줄이고 성능을 개선할 수 있는지 확인하는 것이다.

### 3.5 Group B: Statistical feature

Group B는 PPG signal segment의 통계적 특성을 나타내는 feature이다. 여기에는 skewness, kurtosis, perfusion, maximum, minimum, mean absolute deviation 등이 포함된다.

Skewness는 signal distribution의 비대칭성을 나타낸다. 논문에서 제시한 식은 다음과 같다.

$$
S=\frac{1}{N}\sum_{i=1}^{N}\left[\frac{x_i-\hat{\mu}_x}{\sigma}\right]^3
$$

여기서 $x_i$는 PPG sample, $\hat{\mu}_x$는 평균의 empirical estimate, $\sigma$는 standard deviation, $N$은 PPG signal sample 수이다. Skewness는 PPG waveform quality 평가에도 사용될 수 있으며, waveform이 정상적인 맥파 형태를 유지하는지 판단하는 데 도움이 된다.

Kurtosis는 signal distribution의 뾰족함과 tail의 두꺼움을 나타낸다. 논문에서 제시한 식은 다음과 같다.

$$
K=\frac{1}{N}\sum_{i=1}^{N}\left[\frac{x_i-\hat{\mu}_x}{\sigma}\right]^4
$$

Perfusion은 pulsatile component와 non-pulsatile component의 비율로 정의된다. 이는 조직에서 pulse에 의해 변화하는 light absorption과 baseline absorption의 차이를 반영한다. 논문은 perfusion을 다음과 같이 정의한다.

$$
P=\left[\frac{y_{max}-y_{min}}{|\bar{x}|}\right]\times100
$$

여기서 $x$는 raw PPG signal, $\bar{x}$는 raw PPG signal의 평균, $y$는 filtered PPG signal이다.

Mean absolute deviation은 signal이 평균으로부터 얼마나 떨어져 있는지를 나타낸다.

$$
MeanAbs(x)=\frac{1}{N}\sum_{i=1}^{N}|x_i-\bar{x}|
$$

Maximum과 minimum은 각각 signal segment 내 최대값과 최소값이다.

$$
maximum(x)=\max(x_i)
$$

$$
minimum(x)=\min(x_i)
$$

이러한 statistical feature들은 waveform의 세부 fiducial point를 찾지 않아도 계산할 수 있다는 장점이 있다. 그러나 PPG와 혈압 사이의 구체적인 waveform morphology 정보를 충분히 반영하지 못할 수 있다는 한계도 있다.

### 3.6 Group C: Frequency-domain feature

Group C는 frequency-domain feature이다. 논문은 multitaper method, 즉 MTM을 사용하여 PPG signal의 spectral feature를 추출한다. MTM은 classical periodogram이나 Welch periodogram보다 high-frequency resolution과 low variance를 동시에 얻기 위해 여러 개의 orthogonal window, 즉 discrete prolate spheroidal sequences 또는 Slepian sequences를 사용하는 방법이다.

논문은 stationary process의 spectral representation을 다음과 같이 제시한다.

$$
x_t=\int_{-1/2}^{1/2}e^{-i\omega t}dZ(t)
$$

또한 반복적인 periodic component를 가진 process는 다음과 같이 표현된다.

$$
x_t=\sum_j C_j\cos(\omega_j+\phi_j)+\xi_t
$$

MTM에서는 Slepian sequence를 이용하여 eigen coefficient를 계산한다.

$$
y_k(f)=\sum_{t=0}^{N-1}x_t\nu_i^{(k)}(N,W)e^{-i2\pi ft}
$$

그리고 spectrum estimate는 다음과 같이 계산된다.

$$
\hat{S}(f)=\frac{1}{K}\sum_{k=0}^{K-1}|y_k(f)|^2
$$

Frequency-domain feature는 PPG signal의 주파수 구성, spectral power, band power, relative band power 등을 반영할 수 있다. 심박 주기나 호흡, 저주파 변동 등은 spectral domain에서 분석될 수 있으므로 혈압 추정에 도움이 될 가능성이 있다. 그러나 논문 결과에서는 frequency-domain feature가 statistical feature보다는 더 나은 경우가 있었지만 time-domain feature보다는 성능이 떨어졌다고 보고한다.

### 3.7 Feature importance

논문은 feature dimension을 줄이고 redundant feature를 제거하기 위해 feature importance 기법을 적용한다. 사용된 방법은 CART regression feature importance, random forest feature importance, XGBoost feature importance이다.

CART는 binary recursive partitioning을 사용하여 decision tree를 구성하고, 각 feature가 split에 얼마나 기여하는지를 기반으로 중요도를 평가한다. Random forest는 여러 decision tree를 만들고, feature subset을 무작위로 사용하기 때문에 전체 feature의 상대적 기여도를 평가하는 데 적합하다. XGBoost는 gradient boosting 기반 decision tree ensemble이며, gain, frequency, cover 등의 기준으로 feature importance를 계산할 수 있다.

논문은 Group A의 21개 time-domain feature 중 feature importance를 통해 11개 feature를 선택한 결과도 평가한다. 그러나 결과적으로 21개 feature 전체를 사용한 경우가 11개 feature만 사용한 경우보다 더 좋은 성능을 보였다고 보고한다. 이는 제거된 feature 중 일부가 단독 중요도는 낮더라도 sequence model 또는 nonlinear model 내부에서 보완적 정보를 제공했을 가능성을 시사한다.

### 3.8 Machine learning 모델

논문은 전통적 machine learning 모델로 linear regression, random forest, AdaBoost, support vector regression을 사용한다.

Linear regression은 feature와 target 사이의 선형 관계를 가정한다. 단순하고 빠르며 overfitting이 비교적 적지만, PPG와 BP 사이의 비선형성을 표현하기 어렵다. Random forest는 여러 regression tree의 평균을 사용해 최종 예측을 수행하는 ensemble model이다. Nonlinear relationship을 처리할 수 있고 feature importance도 계산할 수 있다. AdaBoost는 이전 weak learner가 잘 예측하지 못한 sample에 더 큰 가중치를 부여하면서 weak learner를 순차적으로 결합하는 boosting model이다. Support vector regression은 margin 기반 regression 방법이며, 논문에서는 linear kernel을 사용했다고 설명한다.

Deep learning 모델로는 LSTM, Bi-LSTM, GRU를 사용한다. LSTM은 recurrent neural network의 한 종류로, input gate, forget gate, output gate와 cell state를 통해 long-term dependency를 학습한다. 논문에서 제시한 LSTM 식은 다음과 같다.

$$
i_t=\sigma(W_{xi}x_t+W_{hi}h_{t-1}+W_{ci}c_{t-1}+b_i)
$$

$$
f_t=\sigma(W_{xf}x_t+W_{hf}h_{t-1}+W_{cf}c_{t-1}+b_f)
$$

$$
c_t=f_tc_{t-1}+i_t\tanh(W_{xc}x_t+W_{hc}h_{t-1}+b_c)
$$

$$
o_t=\sigma(W_{xo}x_t+W_{ho}h_{t-1}+W_{co}c_t+b_o)
$$

$$
h_t=o_t\tanh(c_t)
$$

여기서 $i_t$는 input gate, $f_t$는 forget gate, $o_t$는 output gate, $c_t$는 cell state, $h_t$는 hidden state이다. LSTM은 PPG feature sequence의 시간적 관계를 학습하는 데 유리하다.

Bi-LSTM은 forward LSTM과 backward LSTM을 함께 사용하여 과거 방향과 미래 방향의 정보를 모두 반영한다. 따라서 단방향 LSTM보다 전체 sequence context를 더 잘 활용할 수 있다. PPG segment 전체의 morphology를 분석할 때 앞뒤 방향 정보를 모두 사용하는 것이 도움이 될 수 있다.

GRU는 LSTM보다 단순한 recurrent architecture이다. Update gate와 reset gate를 사용하며 별도의 memory cell이 없다. 논문에서 GRU의 gate는 다음과 같이 정의된다.

$$
r_t=\delta(W_rh_{t-1}+U_rx_t+b_r)
$$

$$
z_t=\delta(W_zh_{t-1}+U_zx_t+b_z)
$$

Hidden state update는 다음과 같다.

$$
h_t=(1-z_t)h_{t-1}+z_t\tilde{h}_t
$$

$$
\tilde{h}*t=\tanh(W*{\tilde{h}t}(h_{t-1}r_t)+U_{\tilde{h}t}x_t)
$$

GRU는 LSTM보다 parameter 수가 적고 계산이 빠르며, 긴 의존성도 어느 정도 처리할 수 있다. 논문 결과에서 GRU는 특히 time-domain feature 기반 SBP 추정에서 좋은 성능을 보였다.

### 3.9 평가 지표

논문은 MAE, RMSE, SD를 사용한다. MAE는 실제값과 예측값의 절대 오차 평균이다.

$$
MAE=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|
$$

RMSE는 큰 오차에 더 민감한 지표이다. 제공된 텍스트의 식에는 표기상 오류가 일부 있지만, 일반적으로 다음과 같이 해석된다.

$$
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

SD는 예측 오차의 분산 정도를 나타낸다.

$$
SD=\sqrt{\frac{\sum(x-\bar{x})^2}{N-1}}
$$

혈압 추정에서는 단순 평균 오차뿐 아니라 오차의 dispersion도 중요하기 때문에 SD가 함께 보고된다.

## 4. 실험 및 결과

### 4.1 Time-domain feature 결과

Group A에서는 21개 time-domain feature를 추출하고, 추가적으로 feature importance를 적용하여 11개 feature만 선택한 set도 구성하였다. 결과적으로 time-domain feature는 전체 feature group 중 가장 우수한 성능을 보였다.

논문은 LSTM, Bi-LSTM, GRU가 traditional machine learning algorithm보다 좋은 정확도를 달성했다고 보고한다. 21개 time-domain feature를 사용한 경우, GRU는 SBP에 대해 평균 오차 형태로 $3.68+4.28$ mmHg, DBP에 대해 $5.34+5.25$ mmHg 수준의 성능을 보였다고 서술한다. 여기서 표기는 MAE와 SD를 함께 나타낸 것으로 해석된다. 다만 제공된 텍스트에는 table 전체 수치가 충분히 보이지 않아 모든 모델의 정확한 수치를 완전하게 재구성할 수는 없다.

Feature importance로 선택한 11개 feature를 사용할 경우에도 deep learning 모델이 좋은 성능을 보였지만, 21개 feature 전체를 사용한 경우보다 평균 오차가 더 커졌다. 논문은 이를 근거로 time-domain feature에서는 21개 feature 전체를 사용하는 것이 더 바람직하다고 결론짓는다. 이는 혈압 추정에서 여러 waveform morphology 정보가 함께 작용하며, 단일 feature importance 기준으로 제거한 feature가 실제 예측에서 보완적 역할을 할 수 있음을 의미한다.

PPG-BP dataset에서는 11개 feature set에 대해 Bi-LSTM이 가장 좋은 MAE와 SD를 달성했다고 언급된다. 그러나 전체 결론에서는 21개 time-domain feature가 11개 feature보다 더 안정적이라고 해석한다.

### 4.2 Statistical feature 결과

Group B에서는 skewness, kurtosis, perfusion, maximum, minimum, mean absolute deviation 등 statistical feature를 사용하였다. Statistical feature는 PPG waveform의 전체 분포 특성을 반영하지만, systolic upstroke나 diastolic time 같은 구체적인 morphology를 직접적으로 표현하지는 않는다.

논문 결과에 따르면 statistical feature에서도 deep learning 모델은 traditional machine learning 모델보다 대체로 좋은 성능을 보였다. PPG-BP dataset에서는 vanilla LSTM이 다른 알고리즘보다 좋은 성능을 보였으며, SBP에 대해 $4.70+5.21$ mmHg, DBP에 대해 $4.68+5.34$ mmHg의 error를 보였다고 보고한다. MIMIC dataset에서는 DBP 추정에서 random forest가 좋은 성능을 보였고, SBP 추정에서는 Bi-LSTM이 좋은 성능을 보였다고 설명한다.

이 결과는 statistical feature만으로도 어느 정도 혈압 추정이 가능하지만, time-domain feature보다 정보량이 제한적일 수 있음을 보여준다. 특히 PPG 기반 혈압 추정은 waveform shape의 세부 시간 구조가 중요하므로, 단순 통계량은 morphology 기반 feature보다 정확도가 떨어질 가능성이 있다.

### 4.3 Frequency-domain feature 결과

Group C에서는 multitaper method를 기반으로 frequency-domain feature를 추출하였다. 또한 band power와 relative band power를 계산하였다. Frequency feature는 PPG 신호의 spectral composition을 반영하고, 심박 주기 및 주파수 대역별 에너지 정보를 표현한다.

논문은 frequency-domain feature가 statistical feature보다 더 좋은 성능을 보였지만, time-domain feature보다는 낮은 성능을 보였다고 결론짓는다. PPG-BP dataset에서는 vanilla LSTM이 가장 좋은 성능을 보였으며, SBP 예측에서 $4.60+5.17$ mmHg, DBP 예측에서 $4.98+5.52$ mmHg의 error를 보였다고 보고한다. MIMIC dataset에서는 Bi-LSTM이 SBP에 대해 $5.42+5.21$ mmHg, DBP에 대해 $6.17+5.89$ mmHg의 error를 보였다고 설명한다.

이 결과는 frequency-domain feature가 혈압과 관련된 일부 정보를 담고 있지만, PPG waveform의 beat-level morphology를 직접적으로 표현하는 time-domain feature에는 미치지 못한다는 점을 시사한다.

### 4.4 모델별 비교 결과

전체적으로 deep learning 모델이 traditional machine learning 모델보다 우수했다. 논문은 특히 LSTM, Bi-LSTM, GRU가 PPG feature sequence의 temporal dependency를 학습할 수 있기 때문에 혈압 추정에서 더 좋은 성능을 냈다고 해석할 수 있다.

전통적 모델 중에서는 random forest가 일부 feature group과 dataset에서 좋은 성능을 보였다. 이는 PPG feature와 BP 사이의 관계가 nonlinear하고 feature interaction이 존재하기 때문일 수 있다. 그러나 linear regression은 복잡한 nonlinear relation을 표현하기 어렵고, SVR과 AdaBoost도 deep recurrent model보다 전반적으로 낮은 성능을 보인 것으로 설명된다.

논문의 최종 결론은 SBP 추정에서는 GRU와 time-domain feature 조합이 특히 우수하고, DBP 추정에서는 Bi-LSTM과 time-domain feature 조합이 강력하다는 것이다. 다만 제공된 텍스트에서는 일부 table의 정확한 수치가 생략되어 있으므로, 세부 ranking을 완전하게 재현하는 데에는 한계가 있다.

### 4.5 기존 연구와의 비교

논문은 MIMIC dataset을 사용한 기존 연구들과 결과를 비교한다. Kachuee et al.은 PAT 기반 calibration-free blood pressure estimation을 여러 machine learning algorithm으로 수행했으며, DBP와 MAP에서는 AAMI 기준을 만족했지만 SBP에서는 만족하지 못했다고 논문은 설명한다. 그러나 이 방법은 ECG와 PPG 두 신호를 요구한다.

Liu et al.은 PPG의 second derivative feature와 time-domain feature를 결합하여 SVR 기반 BP estimation을 수행했다. 이 연구는 35개 feature를 사용했고, SBP에 대해 $8.54+10.9$ mmHg, DBP에 대해 $4.34+5.8$ mmHg의 error를 보였다고 비교된다. 논문은 SDPPG feature가 다섯 개 peak point의 가시성에 의존하므로 정확한 feature 추출이 어렵다고 지적한다.

Slapničar et al.은 PPG와 derivative를 입력으로 사용하는 ResNet 기반 spectro-temporal deep neural network를 제안했지만, 구조가 복잡하고 MAE가 SBP 15.41 mmHg, DBP 12.38 mmHg로 낮은 성능을 보였다고 논문은 설명한다. 표준편차 값은 제공되지 않았다고 언급한다.

이 비교를 통해 저자들은 PPG-only, time-domain feature, LSTM/GRU 계열 모델이 비교적 단순하면서도 합리적인 정확도를 달성할 수 있다고 주장한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 benchmark study로서의 폭넓은 비교 설계이다. 많은 연구가 특정 feature set과 특정 model의 성능만 보고하는 반면, 이 논문은 time-domain, statistical, frequency-domain feature를 나누고, traditional machine learning과 deep learning을 모두 적용하여 비교한다. 따라서 PPG 기반 혈압 추정 연구자에게 어떤 feature group이 상대적으로 유용한지에 대한 실용적인 기준을 제공한다.

두 번째 강점은 두 개의 공개 데이터셋을 사용했다는 점이다. MIMIC-II와 PPG-BP는 서로 다른 환경과 대상에서 수집된 데이터이므로, 단일 데이터셋 기반 결과보다 비교의 폭이 넓다. 특히 PPG-BP dataset은 비교적 최근 공개된 dataset으로, patient demographic information과 disease information을 포함한다는 점에서 의미가 있다.

세 번째 강점은 feature importance 분석을 포함했다는 점이다. PPG feature는 서로 중복되거나 target과 관련성이 낮을 수 있으며, feature dimension이 크면 overfitting이나 계산량 증가 문제가 생긴다. CART, random forest, XGBoost feature importance를 사용하여 feature subset을 분석한 것은 model interpretability와 feature selection 관점에서 의미가 있다.

네 번째 강점은 deep learning 모델의 장점을 실험적으로 확인했다는 점이다. LSTM, Bi-LSTM, GRU가 time-domain feature 기반 혈압 추정에서 전통적 machine learning 모델보다 우수한 결과를 보였다는 결론은, PPG waveform의 temporal structure가 중요하다는 기존 직관과도 잘 맞는다.

다섯 번째 강점은 SBP와 DBP를 모두 평가했다는 점이다. 일부 연구는 SBP만 추정하거나 DBP만 간략히 다루지만, 이 논문은 두 목표 변수를 모두 비교한다. 또한 MAE, RMSE, SD를 함께 사용하여 평균 오차와 오차 분산을 동시에 평가한다.

반면 한계도 명확하다. 첫째, 논문은 benchmark study임에도 데이터 분할 방식이 subject-independent인지 명확하게 충분히 설명되지 않는다. PPG 기반 혈압 추정에서는 같은 subject의 segment가 train과 test에 동시에 포함되면 성능이 과대평가될 수 있다. 제공된 텍스트에서는 train, validation, test 분할을 수행했다고만 설명되어 있으며, subject-wise split 여부는 명확하지 않다.

둘째, 전처리 및 segment filtering의 세부 조건이 제한적으로 제시된다. Irregular segment 제거, alignment, Butterworth filtering, normalization을 수행했다고 설명하지만, cutoff frequency, segment length, artifact rejection threshold 등 재현에 필요한 일부 세부 정보는 충분히 명확하지 않다. Benchmark 연구라면 재현 가능성이 특히 중요하므로 이 점은 한계이다.

셋째, deep learning 모델에 사용된 입력 형식이 완전히 명확하지 않다. 논문은 feature vector를 추출한 뒤 LSTM, Bi-LSTM, GRU에 입력한다고 설명하지만, feature sequence를 어떻게 구성했는지, time step은 무엇인지, 각 time step의 dimension은 무엇인지가 제공된 텍스트만으로는 분명하지 않다. LSTM 계열 모델은 sequence structure에 민감하므로 이 정보가 중요하다.

넷째, AAMI 또는 BHS 같은 clinical validation standard와의 직접적 비교는 제한적이다. 논문은 MAE, RMSE, SD를 사용하고 기존 연구와 비교하지만, 각 모델이 AAMI 기준을 만족하는지 또는 BHS grade를 달성하는지는 체계적으로 제시하지 않는다. 따라서 실제 혈압 측정 장치로서의 임상적 적합성을 판단하기에는 부족하다.

다섯째, frequency-domain feature 설명은 수학적으로 상세하지만, 실제 사용한 spectral feature의 구체적 목록과 feature dimension이 명확하게 정리되어 있지 않다. Multitaper method의 이론은 제시되지만, 최종적으로 어떤 band power와 relative band power를 모델 입력으로 사용했는지에 대한 설명은 제한적이다.

여섯째, feature importance 결과의 생리학적 해석이 충분하지 않다. Time-domain feature가 가장 좋다는 결론은 중요하지만, 어떤 time-domain feature가 SBP 또는 DBP에 가장 중요한지, 왜 그런 feature가 혈압과 관련되는지에 대한 생리학적 논의는 상대적으로 부족하다. Benchmark study의 의의를 더 높이려면 feature ranking과 physiological interpretation이 함께 제시될 필요가 있다.

일곱째, 논문은 PPG-only 접근의 장점을 강조하지만, 실제 wearable 환경에서 motion artifact에 얼마나 견고한지는 검증하지 않는다. Public dataset의 PPG와 실제 일상생활 wearable PPG는 noise 특성이 다르다. 따라서 결과를 mobile health device나 wearable device로 바로 일반화하기는 어렵다.

마지막으로, 논문은 deep learning 모델이 더 좋은 성능을 보였다고 결론짓지만, model complexity, inference time, memory usage 같은 practical deployment cost를 함께 평가하지 않는다. 실제 cuffless BP monitoring device에서는 정확도뿐 아니라 low-power computation, latency, on-device inference 가능성도 중요하다.

## 6. 결론

이 논문은 PPG 기반 cuffless blood pressure estimation에서 feature extraction technique과 machine learning model의 영향을 체계적으로 비교한 benchmark study이다. 연구는 time-domain feature, statistical feature, frequency-domain feature를 각각 추출하고, LR, RF, AdaBoost, SVR 같은 traditional machine learning 모델과 LSTM, Bi-LSTM, GRU 같은 deep learning 모델을 비교하였다. 데이터셋으로는 MIMIC-II와 PPG-BP라는 두 개의 공개 데이터셋을 사용하였다.

핵심 결론은 time-domain feature가 statistical feature와 frequency-domain feature보다 혈압 추정에 더 유용하며, deep learning 모델이 traditional machine learning 모델보다 전반적으로 더 높은 정확도를 보인다는 것이다. 특히 GRU는 time-domain feature 기반 SBP estimation에서 좋은 성능을 보였고, Bi-LSTM은 DBP estimation에서 강점을 보인 것으로 보고된다. 이는 PPG waveform의 시간적 형태와 cardiac cycle morphology가 혈압 추정에 중요한 정보를 제공한다는 점을 뒷받침한다.

이 논문은 새로운 복잡한 network architecture를 제안하기보다는, PPG 기반 BP estimation 연구에서 feature 선택과 model 선택의 기준을 제공한다는 점에서 가치가 있다. 특히 PPG-only 방식이 ECG와 PPG를 함께 사용하는 PTT/PAT 방식보다 sensor 구성과 계산 복잡도를 줄일 수 있다는 점을 강조하며, wearable healthcare device로의 적용 가능성을 보여준다.

다만 실제 임상 적용을 위해서는 추가 검증이 필요하다. Subject-independent split 여부, 전처리 조건, feature construction 방식, deep learning 입력 구조, 외부 데이터셋 검증, motion artifact 환경에서의 robust performance, AAMI/BHS 기준 기반 평가 등이 더 명확히 제시되어야 한다. 또한 모델 복잡도와 on-device deployment 가능성도 함께 평가되어야 한다.

종합하면, 이 논문은 PPG-only blood pressure estimation에서 time-domain feature와 recurrent deep learning model의 조합이 유망하다는 실험적 근거를 제공한다. 향후 연구는 attention mechanism, multi-task learning, subject-independent validation, wearable artifact robustness를 통합하여 더 신뢰성 높은 continuous cuffless BP estimation system을 구축하는 방향으로 확장될 수 있다.
