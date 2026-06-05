# Continuous Blood Pressure Estimation Using Exclusively Photopletysmography by LSTM-Based Signal-to-Signal Translation

* **저자**: Latifa Nabila Harfiya, Ching-Chun Chang, Yung-Hui Li
* **발표연도**: 2021

## 1. 논문 개요

이 논문은 photoplethysmography, 즉 PPG 신호만을 사용하여 arterial blood pressure, 즉 ABP 신호의 전체 waveform을 추정하는 방법을 제안한다. 기존의 많은 cuffless blood pressure estimation 연구는 PPG 또는 ECG-PPG 조합에서 hand-crafted feature를 추출한 뒤 systolic blood pressure, 즉 SBP와 diastolic blood pressure, 즉 DBP라는 두 개의 discrete value를 예측하는 데 초점을 맞추었다. 반면 이 논문은 SBP와 DBP를 직접 회귀하는 대신, raw PPG 신호를 ABP 신호로 변환하는 signal-to-signal translation 문제로 혈압 추정을 재정의한다.

연구의 주요 목표는 PPG 신호 하나만을 입력으로 사용하여 continuous ABP waveform을 생성하고, 생성된 ABP waveform에서 SBP와 DBP를 추출하는 것이다. 이를 위해 저자들은 LSTM-based autoencoder 구조를 사용한다. 먼저 autoencoder가 PPG를 다시 PPG로 복원하도록 학습하여 PPG waveform의 latent representation을 익히게 하고, 이후 transfer learning을 통해 encoder를 고정한 상태에서 decoder가 PPG representation을 ABP waveform으로 변환하도록 학습한다.

이 연구가 다루는 문제는 임상적으로 중요하다. 혈압은 환경, 신체 활동, 정서 상태, 약물, 질환 상태에 따라 분 단위 또는 초 단위로 변할 수 있다. 예를 들어 white-coat hypertension처럼 병원 환경에서만 혈압이 높게 측정되는 경우도 있으며, 이러한 상황에서는 intermittent cuff measurement만으로 실제 혈압 상태를 정확히 파악하기 어렵다. 따라서 continuous BP monitoring은 혈압 변동성, 즉 blood pressure variability를 평가하고 심혈관 위험을 조기에 감지하는 데 중요하다.

기존 cuff-based method는 널리 쓰이지만 연속 측정이 어렵고 반복적인 cuff inflation으로 불편함을 유발한다. Invasive ABP monitoring은 continuous waveform을 제공하지만 catheter 삽입이 필요하므로 일반적인 웨어러블 또는 일상 모니터링에는 적합하지 않다. PPG는 저렴하고 비침습적이며 wearable device에 쉽게 적용될 수 있으므로, PPG를 이용해 ABP waveform을 추정할 수 있다면 cuffless continuous BP monitoring의 실용 가능성이 커진다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 혈압 추정을 단순한 scalar regression 문제가 아니라 PPG waveform에서 ABP waveform으로의 translation 문제로 보는 것이다. 즉 입력과 출력이 모두 시간에 따라 변하는 1차원 생체신호이며, 모델은 한 신호의 형태적 정보를 다른 신호의 형태적 정보로 변환해야 한다.

기존 PPG 기반 혈압 추정 연구는 PPG waveform에서 foot, systolic peak, dicrotic notch, diastolic peak, pulse width, derivative peak 등의 feature를 추출한 뒤 이를 machine learning model에 입력하는 경우가 많다. 그러나 PPG signal은 motion artifact, sensor 위치 변화, 피험자의 나이, 혈관 상태, 질환, 약물 등에 따라 waveform이 왜곡될 수 있다. 특히 dicrotic notch는 고령자나 혈관 탄성이 떨어진 사람에게서 명확하게 나타나지 않을 수 있다. 따라서 hand-crafted feature extraction은 실제 환경에서 실패할 가능성이 있다.

이 논문은 이러한 한계를 줄이기 위해 feature engineering을 배제하고, raw PPG와 그 derivative를 모델 입력으로 사용한다. 모델이 PPG의 시간적 구조와 ABP와의 대응 관계를 직접 학습하도록 설계한 것이다. PPG의 first derivative는 혈액량 변화의 속도를 반영하고, second derivative는 혈액량 변화의 가속도를 반영하므로 혈관 상태와 관련된 정보를 보존할 수 있다고 본다.

또 하나의 중요한 설계 아이디어는 LSTM-based autoencoder와 transfer learning의 결합이다. Autoencoder는 먼저 PPG를 PPG로 reconstruct하는 self-supervised task를 수행한다. 이 과정을 통해 encoder는 PPG waveform의 중요한 표현을 학습한다. 그 다음 encoder를 고정하고 decoder를 ABP reconstruction task에 맞게 학습하여 PPG representation이 ABP waveform으로 변환되도록 한다. 이 접근은 PPG의 시간적 특성을 명시적으로 학습한 뒤 target signal인 ABP로 변환한다는 점에서 단순 end-to-end regression보다 구조적인 의미가 있다.

기존의 PPG-to-ABP translation 연구로 논문은 PPG2ABP 계열의 fully convolutional network 접근을 언급한다. 그러나 해당 방식은 approximation network와 refinement network라는 두 개의 별도 network를 사용해 계산 비용이 크다고 설명한다. 본 논문은 LSTM-based autoencoder라는 단일 구조를 통해 signal-to-signal translation을 수행하려고 한다는 점에서 차별성을 둔다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

논문의 방법은 크게 데이터 수집, PPG preprocessing, signal alignment, derivative extraction, signal quality filtering, LSTM-based autoencoder training, transfer learning 기반 ABP translation, SBP 및 DBP extraction, 성능 평가로 구성된다.

데이터는 PhysioNet의 MIMIC II online database에서 가져왔다. 이 데이터베이스에는 ICU 환자에게서 수집된 PPG와 ABP 신호가 포함되어 있으며, 두 신호의 sampling frequency는 125 Hz이다. 원문은 약 12,000명의 subject에 해당하는 신호가 존재한다고 설명하고, preprocessing 이후 5289명의 subject 데이터를 실험에 사용했다고 보고한다. ICU 데이터는 약물과 중증 질환의 영향을 받을 수 있으므로, 혈압 변동성이 비정상적일 가능성이 있다는 점도 논문에서 언급된다.

모델의 입력은 raw PPG signal, first derivative of PPG, second derivative of PPG를 stacked form으로 구성한 sequence이다. 모델의 출력은 입력과 같은 시간 길이를 갖는 estimated ABP signal이다. 이후 estimated ABP signal에서 maximum 값을 SBP, minimum 값을 DBP로 추출한다.

### 3.2 Denoising

첫 번째 preprocessing 단계는 PPG denoising이다. 저자들은 third-order Butterworth bandpass filter를 사용하며, passband는 0.5 Hz에서 8 Hz로 설정한다. 0.5 Hz 이하의 성분은 baseline wandering이나 매우 느린 drift로 간주될 수 있고, 8 Hz 이상의 성분은 고주파 noise로 처리된다. PPG 기반 생체신호 분석에서 이러한 bandpass filtering은 일반적으로 pulse-related component를 남기고 noise를 줄이기 위한 절차이다.

### 3.3 Z-score normalization

다음으로 PPG signal은 z-score normalization을 거친다. 어떤 신호의 평균을 $\mu$, 표준편차를 $\sigma$라고 할 때 normalized signal $x'$는 다음과 같이 계산된다.

$$
x' = \frac{x-\mu}{\sigma}
$$

여기서 $x$는 원래 PPG signal이고, $x'$는 normalized PPG signal이다. 논문은 training stage에서 각 signal의 normalization parameter를 계산하고, 모든 training data에서 평균 normalization parameter를 구한 뒤 testing stage의 모든 signal에 사용한다고 설명한다. 이 방식은 test data 자체의 통계량을 직접 사용하지 않기 위한 의도로 보이지만, normalization 과정의 정확한 구현은 원문만으로 완전히 상세하게 설명되어 있지는 않다.

### 3.4 Signal alignment

MIMIC II database의 알려진 문제 중 하나는 PPG와 ABP waveform 사이의 misalignment이다. 실제 생리적으로도 ABP와 PPG는 측정 위치가 다르기 때문에 일정한 phase lag가 존재할 수 있다. 저자들은 cross-correlation function을 사용하여 ABP와 PPG 사이의 시간 offset $\Delta t$를 추정한다.

논문에서 cross-correlation function은 다음과 같이 표현된다.

$$
g(\Delta t)=\sum ABP[t]PPG[t+\Delta t]
$$

여기서 $g(\Delta t)$가 최대가 되는 $\Delta t$를 두 신호가 가장 잘 정렬되는 time offset으로 본다. 원문 예시에서는 phase lag가 0.28초로 나타났고, 이 경우 PPG signal을 해당 offset만큼 이동시켜 ABP와 alignment를 맞춘다.

이 alignment는 본 연구에서 매우 중요하다. 모델은 PPG sequence를 ABP sequence로 변환하도록 학습하므로, 입력 PPG의 특정 time point와 출력 ABP의 특정 time point가 가능한 한 같은 cardiac event를 반영해야 한다. Alignment가 맞지 않으면 모델은 실제 생리적 변환이 아니라 시간 지연까지 함께 학습해야 하므로 학습 난이도가 증가하고 SBP, DBP 추출 오류가 커질 수 있다.

### 3.5 PPG derivative extraction

저자들은 raw PPG뿐 아니라 first derivative와 second derivative도 모델 입력으로 사용한다. PPG는 측정 위치에서 blood volume change를 반영한다. 따라서 first derivative는 혈액량 변화의 속도, second derivative는 혈액량 변화의 가속도를 나타내며, 혈관 탄성, stiffness, 혈류역학적 변화와 관련된 정보를 포함할 수 있다.

논문은 derivative 계산식을 다음과 같이 제시한다.

$$
f'(x_i) \cong \frac{f(x_{i+1})-f(x_{i-1})}{2h}+O(h^2)
$$

여기서 $x_i$는 시간 $i$에서의 data point이고, $h$는 일정한 step size이다. 이 식은 central difference approximation으로 볼 수 있다. First derivative를 한 번 더 적용하면 second derivative를 얻을 수 있다. 최종적으로 모델 입력은 PPG, dPPG, sdPPG가 결합된 multichannel time series가 된다.

### 3.6 Inappropriate signal elimination

논문은 학습 데이터의 품질을 높이기 위해 몇 가지 조건에 맞지 않는 signal segment를 제거한다. 첫째, reference ABP에서 계산한 SBP가 180 mmHg보다 크거나 80 mmHg보다 작은 경우 제거한다. SBP는 다음과 같이 계산된다.

$$
SBP = \max(ABP)
$$

둘째, DBP가 130 mmHg보다 크거나 60 mmHg보다 작은 경우 제거한다. DBP는 다음과 같이 계산된다.

$$
DBP = \min(ABP)
$$

셋째, PPG와 reference ABP 사이의 average Pearson correlation coefficient가 0.8보다 낮은 segment를 제거한다. 논문은 morphology similarity를 평가하기 위해 Pearson correlation coefficient $r$를 계산한다. 원문 수식은 일부 조판상 불완전하게 추출되어 있지만, 일반적인 Pearson correlation은 두 신호 $A$와 $P$에 대해 다음과 같은 형태로 이해할 수 있다.

$$
r =
\frac{n\sum AP-\sum A\sum P}
{\sqrt{n\sum A^2-(\sum A)^2}\sqrt{n\sum P^2-(\sum P)^2}}
$$

여기서 $A$는 ABP signal, $P$는 PPG signal, $n$은 sample 수이다. 이 조건은 PPG와 ABP가 형태적으로 유사한 segment만 학습에 사용하기 위한 것이다.

넷째, PPG systolic peak가 검출되지 않는 signal을 제거한다. 이를 위해 heartpy toolkit을 사용한다. Systolic peak가 검출되지 않는 경우는 주로 waveform이 불규칙하거나 sensor position change, subject movement 등의 영향을 받은 경우라고 설명된다.

이러한 filtering 후 최종 데이터는 약 250,000개의 signal segment로 구성된다. 다만 논문은 이러한 elimination 과정으로 abnormal BP, 즉 hypertensive 또는 hypotensive 범위의 데이터가 상당히 줄어든다고 후반부에서 언급한다. 따라서 모델이 일반적인 혈압 범위에서는 좋은 성능을 보일 수 있지만, 극단적인 혈압 상황에서의 성능은 제한될 수 있다.

### 3.7 LSTM 기반 sequence modeling

LSTM은 recurrent neural network의 한 종류로, long-term dependency를 더 안정적으로 학습하기 위해 설계되었다. 일반 RNN은 sequence가 길어질수록 gradient vanishing 또는 exploding 문제가 발생할 수 있다. LSTM은 cell state와 gate mechanism을 사용하여 중요한 정보를 장시간 보존하거나 불필요한 정보를 제거한다.

LSTM cell은 forget gate, input gate, output gate, cell state로 구성된다. Forget gate는 이전 cell state에서 어떤 정보를 버릴지 결정한다. Input gate는 현재 입력으로부터 어떤 새로운 정보를 저장할지 결정한다. Output gate는 업데이트된 cell state에서 어떤 정보를 현재 output으로 내보낼지 결정한다. 이러한 구조는 PPG와 ABP처럼 시간적 패턴과 morphology가 중요한 생체신호에 적합하다.

논문에서는 두 개의 LSTM layer를 사용하며, 각 phase에 128 hidden node가 있다고 설명한다. Decoder phase 끝에는 dropout layer를 추가하고 dropout rate는 0.2로 설정한다. Optimizer는 Adam이고 learning rate는 0.0025이다. Source field와 target field 모두 최대 50 epoch 동안 학습된다. 구현은 Python, Keras, TensorFlow 2.2를 사용했으며, NVIDIA Titan X GPU 4대를 사용해 PPG-to-PPG translation과 PPG-to-ABP translation을 모두 학습하는 데 최대 9시간이 소요되었다고 보고한다.

### 3.8 Autoencoder와 transfer learning

Autoencoder는 입력을 압축된 representation으로 encoding한 뒤 다시 원래 입력으로 decoding하는 neural network이다. 일반적으로 input과 output이 동일한 reconstruction task를 학습하면서 hidden layer가 중요한 latent representation을 학습하게 된다.

본 논문에서는 LSTM-based autoencoder를 사용한다. 첫 번째 단계에서는 PPG를 입력으로 받아 PPG를 다시 출력하는 PPG-to-PPG reconstruction task를 학습한다. 이 단계의 목적은 encoder가 PPG morphology를 잘 이해하고 압축 표현을 생성하도록 만드는 것이다.

두 번째 단계에서는 transfer learning을 적용한다. 먼저 학습된 encoder를 freeze한다. 이후 decoder 쪽을 ABP waveform을 생성하도록 학습한다. 즉 encoder는 PPG waveform의 representation을 만들고, decoder는 이 representation을 ABP waveform으로 변환한다. 원문은 이 과정을 통해 intermediate waveform representation을 명시적으로 학습할 수 있다고 설명한다.

이 구조는 다음과 같이 이해할 수 있다. 먼저 모델은 “PPG란 어떤 형태의 시간 신호인가”를 자기복원 방식으로 배운다. 그 다음 “PPG의 표현이 주어졌을 때 대응되는 ABP waveform은 어떤 형태인가”를 supervised signal-to-signal translation으로 배운다. 최종 출력인 estimated ABP signal에서 SBP와 DBP를 추출한다.

### 3.9 평가 지표

논문은 SBP와 DBP 추정 성능을 mean absolute error와 root mean square error로 평가한다. MAE는 다음과 같다.

$$
MAE = \frac{1}{N}\sum_{i=1}^{N}|e_i|
$$

RMSE는 다음과 같다.

$$
RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}e_i^2}
$$

여기서 $e_i$는 observed BP value와 predicted BP value의 차이이며, 단위는 mmHg이다. 또한 BHS standard와 AAMI standard를 사용하여 혈압 추정 장치로서의 기준 충족 여부를 평가한다.

## 4. 실험 및 결과

### 4.1 데이터 구성과 실험 설정

실험에는 preprocessing 이후 5289명의 subject가 사용되었다. 최종 segment 수는 약 250,000개이다. 데이터 partition은 training 70%, validation 10%, testing 20% 비율로 나뉜다. 원문은 “completely disjoining our data partition”이라고 설명하지만, 이 disjoint split이 subject-level split인지 segment-level split인지는 명확하게 설명하지 않는다. 이 점은 모델의 일반화 성능 해석에서 중요한 한계로 볼 수 있다.

입력은 raw PPG와 그 first derivative, second derivative로 구성되고, 출력은 ABP waveform이다. 이후 predicted ABP waveform에서 SBP와 DBP를 추출하여 reference ABP에서 계산한 SBP, DBP와 비교한다.

### 4.2 기존 연구와의 MAE 및 RMSE 비교

논문은 동일한 데이터 출처를 사용한 이전 연구들과 성능을 비교한다. 제안 모델의 성능은 다음과 같다.

| Method         |       Dataset |   SBP MAE |  SBP RMSE |   DBP MAE |  DBP RMSE |
| -------------- | ------------: | --------: | --------: | --------: | --------: |
| Proposed model | 5289 subjects | 4.05 mmHg | 5.25 mmHg | 2.41 mmHg | 3.17 mmHg |

비교 대상 중 일부 feature engineering 기반 연구는 더 낮은 MAE를 보고한다. 예를 들어 저자들의 이전 연구는 9000명 데이터에서 SBP MAE 3.21 mmHg, DBP MAE 2.23 mmHg를 보고하고, 다른 연구는 500명 데이터에서 SBP MAE 3.25 mmHg, DBP MAE 1.43 mmHg를 보고한다. 그러나 이 논문은 제안 모델이 ABP waveform 전체를 추정한다는 점에서 단순 SBP/DBP scalar prediction과 차이가 있다고 강조한다.

가장 유사한 목적을 가진 연구로는 PPG-to-ABP translation을 수행한 Ibtehaz et al.의 fully convolutional neural network 기반 접근이 있다. 해당 연구는 942 subjects에서 SBP MAE 5.73 mmHg, DBP MAE 3.45 mmHg를 보고한다. 이에 비해 본 논문은 5289 subjects에서 SBP MAE 4.05 mmHg, DBP MAE 2.41 mmHg를 달성하여, 비슷한 signal-to-signal translation 계열 연구보다 낮은 오차를 보였다고 주장할 수 있다.

### 4.3 BHS standard 평가

BHS standard는 BP measurement system의 오차가 일정 threshold 이하에 들어가는 누적 비율을 기준으로 Grade A, B, C를 부여한다. 제안 모델의 결과는 다음과 같다.

| Cumulative Error | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg |
| ---------------- | ------: | -------: | -------: |
| SBP              |   70.6% |    94.1% |    98.6% |
| DBP              |   91.1% |    99.1% |    99.8% |
| BHS Grade A 기준   |     60% |      85% |      95% |

SBP와 DBP 모두 세 기준에서 Grade A 조건을 충족한다. 특히 DBP는 5 mmHg 이하 오차 비율이 91.1%로 매우 높고, 15 mmHg 이하 오차 비율은 99.8%에 달한다. SBP도 5 mmHg 이하 70.6%, 10 mmHg 이하 94.1%, 15 mmHg 이하 98.6%로 Grade A 기준을 만족한다.

### 4.4 AAMI standard 평가

AAMI standard는 일반적으로 85명 이상의 subject를 대상으로 mean error 또는 MAE와 standard deviation이 일정 기준 이하인지 평가한다. 원문 Table 3은 다음 결과를 제시한다.

| Measurement |       MAE |       STD | # Subjects |
| ----------- | --------: | --------: | ---------: |
| SBP         | 4.05 mmHg | 4.60 mmHg |       5289 |
| DBP         | 2.41 mmHg | 3.11 mmHg |       5289 |
| AAMI 기준     |   <5 mmHg |   <8 mmHg |        >85 |

제안 모델은 SBP와 DBP 모두에서 MAE가 5 mmHg보다 작고 STD가 8 mmHg보다 작으며 subject 수 역시 85명을 크게 초과한다. 따라서 원문 기준으로는 AAMI standard를 충족한다고 해석된다.

### 4.5 PPG와 ABP의 morphology coherence 분석

논문은 PPG와 ABP waveform 사이의 형태적 유사성을 분석한다. 전체 12,000 subjects에서 먼저 4,028,466개 segment를 얻고, Pearson correlation coefficient $r$이 0.8보다 큰 segment를 선택했다. 그 결과 3,841,600개 segment가 남았으며, 이는 전체 input signal의 약 95%가 reference signal과 높은 correlation을 가진다는 의미라고 설명한다. 평균 morphology correlation은 0.84로 보고된다.

이 분석은 PPG-to-ABP signal translation의 타당성을 뒷받침한다. 즉 PPG와 ABP가 시간 영역에서 어느 정도 유사한 waveform structure를 공유하기 때문에, 모델이 PPG waveform representation으로부터 ABP waveform을 복원할 수 있다는 것이다. 다만 correlation이 높은 segment만 선택했다는 점은 성능 평가를 유리하게 만들 수 있으므로, 실제 잡음 환경과 낮은 correlation segment에서의 성능은 별도로 검증되어야 한다.

### 4.6 Predicted SBP 및 DBP correlation

논문은 predicted SBP와 observed SBP 사이의 Pearson correlation coefficient가 0.92이고, predicted DBP와 observed DBP 사이의 Pearson correlation coefficient가 0.93이라고 보고한다. 이는 추정값과 reference 값 사이에 높은 양의 상관관계가 있음을 의미한다.

또한 predicted ABP sequence 예시를 통해 모델이 reference ABP waveform과 유사한 형태의 waveform을 생성한다고 설명한다. 논문은 estimated waveform이 단순히 SBP와 DBP만 제공하는 것이 아니라 ABP의 전체 waveshape를 제공하므로, 의료진이 환자의 cardiovascular condition을 평가하는 데 더 풍부한 정보를 제공할 수 있다고 주장한다.

### 4.7 결과 해석의 주의점

논문 결과는 수치적으로 매우 우수하지만, 데이터 필터링 과정이 강하게 적용되었다는 점을 함께 고려해야 한다. 특히 SBP가 80 mmHg 미만 또는 180 mmHg 초과인 segment, DBP가 60 mmHg 미만 또는 130 mmHg 초과인 segment, PPG와 ABP correlation이 0.8 미만인 segment, PPG systolic peak가 검출되지 않는 segment가 제거되었다. 논문 후반부에서도 이러한 elimination process로 인해 abnormal BP values가 최대 90%까지 줄어들었다고 언급한다.

따라서 제안 모델은 clean PPG와 ABP 간 morphology coherence가 높은 조건에서 강한 성능을 보이지만, hypertensive crisis, hypotension, irregular waveform, motion artifact가 많은 실제 wearable 환경에서도 동일한 성능을 낸다고 단정할 수는 없다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 PPG 기반 혈압 추정을 scalar value regression이 아니라 waveform translation으로 다룬다는 점이다. SBP와 DBP만 예측하는 방식은 혈압의 최고점과 최저점만 제공하지만, ABP waveform 전체를 추정하면 pulse morphology, waveform shape, beat-to-beat dynamic, 잠재적 cardiovascular information을 더 많이 보존할 수 있다.

두 번째 강점은 hand-crafted feature engineering을 제거했다는 점이다. PPG feature extraction은 waveform landmark가 명확할 때는 유용하지만, motion artifact나 sensor displacement, age-related waveform change가 존재하면 불안정하다. 제안 모델은 raw PPG와 derivative를 직접 입력으로 사용하므로, 사전에 복잡한 feature를 설계하거나 dicrotic notch를 안정적으로 검출할 필요가 없다.

세 번째 강점은 LSTM-based autoencoder와 transfer learning을 결합한 구조이다. 단순히 PPG에서 ABP로 바로 mapping하는 것이 아니라, 먼저 PPG reconstruction을 학습하여 PPG representation을 확보하고 이를 ABP translation에 활용한다. 이는 time-series morphology를 다루는 데 자연스러운 접근이며, PPG와 ABP가 모두 sequential physiological signal이라는 특성을 반영한다.

네 번째 강점은 비교적 많은 subject를 사용했다는 점이다. 일부 기존 연구는 수십 명 또는 수백 명의 subject를 사용했지만, 본 논문은 preprocessing 이후에도 5289명의 subject를 사용했다고 보고한다. 또한 BHS와 AAMI standard 모두에서 Grade A 또는 기준 충족 결과를 제시했다.

그러나 한계도 중요하다. 첫째, 데이터는 ICU 환자 기반의 MIMIC II database에서 가져온 것이다. ICU 환자의 혈압과 PPG는 약물, 질환, ventilation, circulatory support 등의 영향을 받을 수 있으며, 일반 건강인 또는 일상 wearable 환경과 분포가 다를 수 있다.

둘째, abnormal BP range가 상당 부분 제거되었다. SBP와 DBP의 threshold filtering, correlation filtering, systolic peak detection failure 제거로 인해 hypertensive 또는 hypotensive case가 줄어든다. 논문도 이러한 과정이 dataset을 약 90%까지 줄였다고 언급한다. 따라서 실제 임상적으로 중요한 저혈압, 고혈압, 쇼크, 급격한 혈압 변화 상황에서의 성능은 충분히 입증되지 않았다.

셋째, train, validation, test split이 subject-independent인지 명확하지 않다. 혈압 추정 연구에서 동일 subject의 segment가 training과 test에 동시에 포함되면 모델이 subject-specific morphology를 학습하여 성능이 과대평가될 수 있다. 원문은 partition이 disjoint하다고 설명하지만, subject-level disjoint split이라고 명시하지 않는다. 이 점은 보고된 성능의 일반화 가능성을 해석할 때 가장 중요한 불확실성이다.

넷째, correlation threshold가 높게 설정되어 있어 모델이 상대적으로 쉬운 sample에서 평가되었을 가능성이 있다. 실제 wearable PPG는 motion, illumination, sensor pressure, skin condition 등으로 인해 ABP와의 morphology correlation이 낮아질 수 있다. 본 연구의 모델이 그러한 낮은 품질의 PPG에서도 안정적으로 동작하는지는 추가 검증이 필요하다.

다섯째, PPG-to-PPG reconstruction 후 encoder freezing이 최적의 transfer learning 방식인지에 대한 ablation study가 충분히 제시되지 않는다. 예를 들어 encoder를 freeze하지 않고 fine-tuning했을 때, derivative input을 제외했을 때, PPG만 사용했을 때, LSTM 대신 CNN이나 GRU를 사용했을 때의 비교가 본문에 충분히 상세하게 제시되지는 않는다.

여섯째, ABP waveform 자체의 평가가 SBP와 DBP 중심으로 이루어진다. 논문은 waveform translation을 강조하지만, waveform-level metric으로는 주로 qualitative example과 correlation 설명이 제시된다. ABP waveform의 전체 형태를 평가하려면 waveform MAE, dynamic time warping distance, beat-level morphology error, systolic upstroke error, dicrotic notch timing error 등의 더 세밀한 분석이 추가될 수 있다.

## 6. 결론

이 논문은 PPG만을 사용하여 continuous ABP waveform을 추정하는 LSTM-based signal-to-signal translation 방법을 제안한다. 제안 모델은 raw PPG, first derivative, second derivative를 입력으로 사용하고, LSTM-based autoencoder를 통해 PPG representation을 학습한 뒤 transfer learning으로 ABP waveform을 생성한다. 생성된 ABP waveform에서 SBP와 DBP를 추출하여 혈압 추정 성능을 평가한다.

실험 결과, 제안 모델은 5289명의 subject를 사용한 test에서 SBP MAE 4.05 mmHg, SBP RMSE 5.25 mmHg, DBP MAE 2.41 mmHg, DBP RMSE 3.17 mmHg를 달성했다. BHS 기준에서는 SBP와 DBP 모두 Grade A를 만족했고, AAMI 기준에서도 MAE와 STD 조건을 충족했다. 또한 predicted SBP와 observed SBP 사이의 correlation은 0.92, predicted DBP와 observed DBP 사이의 correlation은 0.93으로 보고되었다.

이 연구의 의미는 cuffless continuous BP monitoring을 위해 PPG 기반 접근을 한 단계 확장했다는 데 있다. 단순히 SBP와 DBP라는 숫자를 추정하는 대신 ABP waveform 전체를 생성하려 했으며, 복잡한 hand-crafted feature engineering 없이 raw signal 기반 sequence model을 사용했다. 이는 wearable device, mobile healthcare, continuous patient monitoring 분야에서 유망한 방향이다.

다만 실제 적용을 위해서는 abnormal BP case를 포함한 더 넓은 혈압 범위, subject-independent split 기반 평가, motion artifact가 많은 실제 wearable 환경, 낮은 signal quality 조건에서의 robustness 검증이 필요하다. 특히 train/test subject separation이 명확히 확인되어야 모델의 실제 generalization 성능을 신뢰할 수 있다. 종합적으로 이 논문은 PPG-only continuous ABP waveform estimation을 위한 의미 있는 signal-to-signal deep learning 접근을 제시했으며, 향후 cuffless BP monitoring 연구에서 중요한 참고점이 될 수 있다.
