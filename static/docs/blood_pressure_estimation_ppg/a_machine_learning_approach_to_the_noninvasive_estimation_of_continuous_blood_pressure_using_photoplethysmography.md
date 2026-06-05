# A Machine Learning Approach to the Non-Invasive Estimation of Continuous Blood Pressure Using Photoplethysmography

* **저자**: Basheq Tarifi, Aaron Fainman, Adam Pantanowitz, David M. Rubin
* **발표연도**: 2023

## 1. 논문 개요

이 논문은 photoplethysmography, 즉 PPG 신호만을 이용하여 continuous blood pressure를 cuffless, non-invasive 방식으로 추정할 수 있는지를 조사한 연구이다. 현재 임상에서 혈압을 측정하는 대표적인 방법은 cuff 기반 manual auscultation 또는 oscillometric measurement이며, 이는 비침습적이지만 연속 측정이 어렵고 arterial blood pressure, 즉 ABP waveform을 제공하지 않는다. 반대로 arterial cannulation은 연속 ABP waveform을 직접 측정할 수 있지만 침습적이며 출혈, 혈관 폐색, sepsis, 환자 불편감과 같은 위험이 있다. 이 논문은 이러한 두 방식의 한계를 줄이기 위해 PPG와 machine learning을 이용한 연속 혈압 추정 시스템을 제안한다.

논문의 연구 문제는 크게 두 가지이다. 첫째, PPG signal만으로 clinically useful한 discrete blood pressure value, 즉 systolic blood pressure, diastolic blood pressure, mean arterial pressure를 추정할 수 있는가이다. 둘째, 단순히 SBP, DBP, MAP 같은 discrete value만 예측하는 것이 아니라, ABP waveform shape까지 추정할 수 있는가이다. ABP waveform은 심부전, arterial compliance, valve damage 등과 관련된 진단적 정보를 포함하므로, 단순 수치 혈압보다 더 풍부한 cardiovascular information을 제공할 수 있다.

이 연구는 MIMIC-III Waveform Database에서 PPG와 ABP가 모두 포함된 환자 데이터를 사용한다. 연구진은 25,000명의 환자 record를 탐색하여 PPG와 ABP가 함께 존재하는 4027명의 record를 다운로드했고, filtering과 quality control을 거친 뒤 최종적으로 1669명의 환자, 890시간 분량의 대응 PPG-ABP waveform을 사용하였다. 논문은 이 dataset 규모가 검토한 기존 연구들보다 훨씬 크다고 설명한다.

제안된 방법은 하나의 model이 모든 것을 직접 예측하는 방식이 아니라, **discrete BP prediction**과 **waveform shape prediction**을 분리하는 two-stage approach이다. SBP, DBP, MAP은 feature extraction 기반 ANN 모델로 예측하고, 별도의 waveform neural network는 normalized ABP waveform shape를 예측한다. 이후 예측된 SBP, DBP, MAP을 이용해 waveform network의 출력을 shift 및 scale하여 최종 ABP waveform을 생성한다.

주요 결과는 feature-trained ANN이 SBP를 $5.26 \pm 6.53$ mmHg, DBP를 $2.96 \pm 3.31$ mmHg, MAP을 $3.27 \pm 3.55$ mmHg의 error 수준으로 예측했다는 것이다. 논문은 이 결과가 AAMI/ESH/ISO international clinical blood pressure measurement standard를 만족한다고 주장한다. Waveform prediction에서는 실제 ABP waveform과 예측 waveform 간 Pearson correlation $r = 0.864$를 달성하였다.

이 논문의 중요성은 PPG 기반 혈압 추정 연구에서 discrete BP value뿐 아니라 continuous ABP waveform까지 추정하려 했다는 점에 있다. 특히 저자들은 PPG가 이미 pulse oximeter와 wearable device에 널리 사용되는 저비용 센서라는 점을 강조하며, 이 방법이 임상 환경, 저자원 의료 환경, 가정 혈압 monitoring, wearable health device로 확장될 가능성을 제시한다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 **PPG signal에서 혈압의 절대 수치와 waveform shape를 분리해서 학습한 뒤, 두 결과를 결합하여 continuous ABP waveform을 생성하는 것**이다. 기존 연구 중 일부는 PPG waveform 전체를 deep learning model에 입력하여 SBP와 DBP를 직접 예측하거나, PPG-to-ABP waveform translation을 시도하였다. 그러나 이 논문은 waveform shape prediction과 amplitude calibration을 분리한다. 즉, waveform network는 ABP의 정확한 절대 pressure amplitude를 맞추기보다 waveform의 모양을 잘 예측하도록 학습하고, SBP, DBP, MAP prediction model이 그 waveform을 실제 혈압 scale에 맞게 조정한다.

이 설계는 매우 실용적인 직관에 기반한다. PPG와 ABP는 모두 cardiac cycle에 의해 생성되는 맥파 형태의 physiological waveform이지만, 센서 위치, 혈관 특성, 조직 광학 특성, 측정 지연 때문에 absolute amplitude가 직접 대응되지는 않는다. 따라서 waveform network가 ABP의 절대값까지 동시에 맞추려고 하면 학습이 어려울 수 있다. 반면 shape만 잘 맞춘 뒤, 별도로 예측된 SBP, DBP, MAP을 이용해 scaling하면 모델의 역할이 더 명확해진다.

논문에서 제시한 최종 ABP waveform 생성식은 다음과 같다.

$$
ABP =
\left(
\frac{ABP_0 - \overline{ABP_0}}
{\max(ABP_0) - \min(ABP_0)}
\right)
(SBP - DBP) + MAP
$$

여기서 $ABP_0$는 waveform neural network가 출력한 normalized waveform이고, $SBP$, $DBP$, $MAP$은 별도의 discrete blood pressure neural network들이 예측한 값이다. 이 식은 waveform network의 출력에서 평균을 제거하고 range로 normalize한 뒤, 예측된 pulse pressure인 $SBP - DBP$만큼 scale하고 $MAP$을 더해 전체 waveform을 pressure domain으로 이동시키는 방식이다.

또 다른 핵심 아이디어는 PPG 기반 discrete BP prediction에서 deep learning on raw waveform보다 handcrafted feature extraction 기반 model이 더 좋은 성능을 보였다는 점이다. 논문은 ANN, LSTM, CNN, regression SVM, binary decision tree 등 여러 모델을 비교했으며, 최종적으로 feature-trained ANN이 가장 좋은 discrete BP prediction 성능을 보였다. 이는 PPG waveform에서 systolic point, diastolic point, waveform foot, pulse width, area, heart rate, respiratory rate와 같은 physiological feature가 여전히 혈압 예측에 유용하다는 점을 보여준다.

기존 접근 방식과의 차별점은 세 가지로 정리할 수 있다. 첫째, PPG만 사용하고 ECG나 cuff-based measurement 같은 auxiliary biosignal을 배제했다. 이는 실용성과 저비용 구현 가능성을 높인다. 둘째, discrete BP prediction과 waveform shape prediction을 분리한 hybrid design을 사용했다. 셋째, MIMIC-III Waveform Database에서 1669명의 환자와 890시간의 데이터를 사용하여 기존 다수 연구보다 큰 규모로 평가하였다.

## 3. 상세 방법 설명

### 전체 파이프라인

논문의 전체 시스템은 입력 PPG signal로부터 최종 ABP waveform을 생성하는 구조이다. 먼저 PPG와 ABP가 함께 존재하는 MIMIC-III waveform record를 수집한다. 이후 signal filtering, poor-quality segment removal, phase shift correction, correlation-based quality filtering을 수행한다. 전처리된 데이터는 두 가지 학습 흐름으로 나뉜다.

첫 번째 흐름은 discrete BP prediction이다. 이 흐름에서는 PPG signal에서 time-domain feature와 frequency-domain feature를 추출하고, 이를 machine learning model에 입력하여 SBP, DBP, MAP을 각각 예측한다. 두 번째 흐름은 waveform prediction이다. 이 흐름에서는 10초 길이의 PPG waveform을 입력으로 사용하여 마지막 1초에 해당하는 ABP waveform shape를 예측한다. 최종적으로 discrete BP prediction 결과와 waveform shape prediction 결과를 결합하여 pressure scale을 갖는 ABP waveform을 생성한다.

### 데이터셋과 데이터 구성

연구는 MIMIC-III Waveform Database를 사용한다. 이 database는 ICU 환자의 anonymized physiological waveform을 포함한다. 본 연구는 MIMIC-III clinical database의 demographic 및 clinical variable은 사용하지 않고, waveform database의 PPG와 ABP signal만 사용하였다. 저자들은 clinical data를 사용하지 않은 점을 한계로 명시한다.

연구진은 25,000명의 patient record를 탐색했고, PPG와 ABP가 모두 존재하는 4027명의 record를 확보하였다. 이후 전처리와 품질 검사를 거쳐 1669명의 환자, 93,309개의 segment, 총 890시간의 대응 PPG-ABP waveform이 남았다.

논문은 데이터의 SBP 평균이 134.2 mmHg이고 표준편차가 18.65 mmHg라고 설명한다. 이는 포함된 평균 환자가 hypertensive에 가까울 가능성을 시사한다. 데이터가 ICU 환자로부터 수집되었기 때문에 healthy individual이나 outpatient population으로 일반화할 때 bias가 존재할 수 있다.

### 데이터 전처리

전처리는 PPG와 ABP waveform 모두에 적용되었다. 먼저 Hampel filter를 사용하여 outlying point를 제거했다. Hampel filter는 일정 window 안에서 outlier를 window median으로 대체하는 방식이다. 이후 두 signal 모두 6th order zero-phase lowpass Butterworth filter를 통과시켰고 cutoff frequency는 25 Hz로 설정하였다. Zero-phase filtering은 phase distortion을 줄이기 위한 선택으로 해석할 수 있다.

다음으로 poor-quality segment를 제거하였다. 제거 기준에는 flat signal, out-of-range value, distorted waveform이 포함된다. Out-of-range value는 SBP가 80 mmHg 미만 또는 180 mmHg 초과이거나, DBP가 60 mmHg 미만 또는 130 mmHg 초과인 경우로 정의하였다. Flat signal은 gradient가 약 0인 구간이 1.5초 이상 지속되는 경우로 탐지하였다.

ABP waveform의 integrity는 두 가지 방식으로 확인되었다. 첫째, WFDB library의 beat detection tool을 사용해 cardiac period, 특히 diastolic point index를 검출할 수 있는지 확인하였다. 둘째, 1.92초 window의 moving maximum을 적용하여 irregular pattern이 있는 구간과 정상 구간을 구분하였다. 논문은 이 window size가 empirical하게 선택되었다고 설명한다.

PPG와 ABP는 측정 위치가 다르므로 phase shift가 존재한다. 이를 제거하기 위해 cross-correlation function을 계산하였다.

$$
g(\tau) = \sum_t ABP[t]PPG[t+\tau]
$$

여기서 $g(\tau)$가 최대가 되는 $\tau$를 PPG와 ABP 사이의 time offset으로 보고 phase shift를 보정한다. 이후 Pearson correlation coefficient $r$을 계산하고, $r \leq 0.8$인 PPG-ABP segment pair는 제거하였다. 이는 PPG와 ABP가 충분히 유사한 morphology를 공유하는 segment만 학습에 사용하기 위한 절차이다.

### Discrete blood pressure prediction

Discrete BP prediction의 목표는 PPG signal에서 SBP, DBP, MAP을 각각 예측하는 것이다. 논문은 두 가지 접근을 비교하였다. 첫째는 PPG에서 handcrafted feature를 추출한 뒤 classical 또는 shallow machine learning model을 적용하는 방식이다. 둘째는 waveform 전체를 deep learning model에 직접 입력하는 방식이다.

#### Feature extraction

Feature extraction은 PPG waveform에서 systolic point, diastolic point, waveform foot을 안정적으로 찾는 것에서 시작한다. Systolic point는 peak로 정의하며, peak 사이 최소 간격은 0.24초로 설정하였다. 이는 최대 heart rate 250 bpm에 해당한다. Waveform foot은 systolic peak 이전의 local minimum으로 찾는다. Diastolic point는 systolic peak와 다음 foot 사이의 local maxima 중 amplitude, first derivative, second derivative를 고려하는 scoring matrix를 사용해 찾는다. 특히 dicrotic notch나 diastolic point가 잘 보이지 않는 stiff artery waveform에서도 feature extraction이 가능하도록 설계하였다.

추출된 feature는 크게 time-related feature, amplitude feature, time-amplitude feature, area feature, frequency feature로 구성된다.

Time-related feature에는 systolic upstroke duration, diastolic duration, systolic peak와 diastolic peak 사이 duration이 포함된다. Amplitude feature에는 systolic peak, diastolic peak, foot, augmentation index가 포함된다. Augmentation index는 systolic peak와 diastolic peak의 ratio로 정의된다. Time-amplitude feature에는 systolic width, diastolic width, total width, systolic-to-diastolic width ratio가 포함된다. 이때 pulse height의 25%, 33%, 50%, 66%, 75%, 90% 수준에서 width를 계산한다. Area feature에는 pulse area와 inflection point area가 포함된다. Frequency feature에는 respiratory rate와 heart rate가 포함된다.

Respiratory rate는 PPG signal의 low-frequency modulation을 이용하여 추정한다. 논문은 respiratory frequency range를 0.05 Hz에서 0.47 Hz로 가정하고, Fourier transform에서 이 band 안의 가장 큰 frequency를 respiratory rate로 사용하였다.

#### Discrete prediction models

Discrete BP prediction에는 ANN, LSTM, CNN, regression SVM, binary decision tree가 사용되었다. ANN은 feature extraction approach와 full waveform approach 모두에 적용되었다. Feature extraction approach에서는 2-layer, 3-layer, 4-layer ANN이 사용되었고, 각 hidden layer는 512 activation unit을 가진다. Full waveform approach에서는 2-layer와 3-layer ANN이 사용되었다.

LSTM은 sequential data에 적합하기 때문에 full waveform input에 적용되었다. 논문은 LSTM이 긴 sequence 처리에 어려움을 가질 수 있어, 일부 구조에서는 2초 PPG data로 마지막 1초 ABP 값을 예측하는 방식이 사용되었다고 설명한다. CNN은 waveform pattern을 찾는 데 적합하므로 1D CNN 형태로 적용되었다. CNN은 세 개의 convolutional layer를 포함하며, 각 convolutional layer 사이에는 batch normalization과 ReLU가 있다. 마지막에는 global average pooling layer와 regression output이 사용된다.

ANN, RNN, CNN 학습에는 Adam optimizer와 batch size 128이 사용되었다. Loss function은 mean squared error이다. Dataset은 training, validation, test set으로 70:15:15 비율로 random split되었다. 모든 feature와 waveform은 $-1$에서 $1$ 범위로 shift 및 scale되었다. 학습은 learning rate $10^{-4}$에서 시작했고, validation loss가 20 epochs 동안 개선되지 않을 때까지 진행되었다. 최종 learning rate는 $10^{-6}$까지 낮아질 수 있다.

이 논문에서 명시적으로 주의할 점은 subject-level split을 사용했다는 설명은 제공되지 않는다는 것이다. 데이터가 segment 단위로 70:15:15로 나뉘었다고 서술되어 있으므로, 동일 환자의 segment가 training과 test에 동시에 포함되었는지 여부는 제공된 텍스트만으로 확인할 수 없다. 이는 결과 해석에서 중요한 한계가 될 수 있다.

### Waveform prediction

Waveform prediction의 목표는 PPG로부터 ABP waveform shape를 예측하는 것이다. 입력은 10초 길이의 PPG waveform이고, 출력은 마지막 1초 구간, 즉 $t=9$ s에서 $t=10$ s까지의 ABP waveform이다. Sampling frequency는 125 Hz이므로 입력은 1250 point이고, 출력은 원래 125 point이다. 그러나 출력 ABP는 25 Hz로 down-sampling되어 25 point가 된다. 논문은 ABP bandwidth를 약 8 Hz로 보고, 25 Hz가 Nyquist criterion $F_s \geq 2F_b = 16$ Hz를 충분히 만족한다고 설명한다.

Waveform input과 output은 feature scaling된다. PPG scaling 식은 다음과 같다.

$$
PPG_{scaled} =
\frac{PPG - E_s[PPG]}
{E_s[\max(PPG)-\min(PPG)]}
$$

여기서 $E_s[PPG]$는 전체 data sample의 mean에 대한 expected value이고, denominator는 전체 sample range의 expected value이다. ABP에도 같은 방식의 scaling이 적용된다.

Waveform prediction을 위해 300,000개 이상의 observation이 생성되었다. 각 observation은 1250-point PPG input과 25-point ABP output으로 구성된다.

### Custom regression loss layer

Waveform prediction model은 absolute pressure value를 맞추는 것이 아니라 waveform shape를 잘 맞추는 것이 목표이다. 이를 위해 저자들은 MATLAB에서 custom regression layer를 만들고, Pearson correlation coefficient를 기반으로 loss function을 정의하였다.

$$
loss = 1 - |r|
$$

여기서 $r$은 predicted wave shape $A_0$와 actual wave shape $A_t$ 사이의 Pearson correlation coefficient이다. $|r|$이 1에 가까울수록 두 waveform shape가 강하게 correlated되어 있으므로, $1-|r|$을 minimize하면 waveform shape correlation을 maximize하게 된다.

이 설계는 discrete BP prediction과 waveform shape prediction을 분리한다는 논문의 전체 철학과 일치한다. 즉, waveform network는 amplitude error보다 shape similarity에 집중한다.

### Waveform neural network architecture

최종 waveform model은 sequence-to-sequence regression network이며, input layer, 세 개의 hidden layer, output layer로 구성된다. 각 hidden layer는 1024 neuron의 fully connected layer이고, 그 뒤에 batch normalization, ReLU activation, dropout layer가 붙는다. Dropout factor는 0.2이다. Output layer는 Pearson correlation loss를 사용하는 custom regression layer이다. Optimizer는 stochastic gradient descent with momentum, SGDM이다.

논문은 LSTM, bidirectional LSTM, fully connected ANN 등 여러 구조를 비교하였다. 최종 선택된 구조는 세 개의 fully connected 1024 layer와 Pearson regression layer를 사용하는 구조이며, test set에서 Pearson correlation $r = 0.864$를 달성했다. 다른 구조로는 2-layer biLSTM과 standard regression layer가 $r = 0.801$, 2-layer LSTM과 Pearson regression layer가 $r = 0.818$을 기록했다.

## 4. 실험 및 결과

### Discrete BP prediction 결과

Discrete blood pressure prediction에서는 총 9개의 machine learning model이 학습 및 평가되었다. 평가 metric으로는 MAE, error가 10 mmHg 미만인 prediction 비율, coefficient of determination $R^2$가 사용되었다. 논문은 AAMI/ESH/ISO standard에 따라 error가 10 mmHg 미만인 prediction이 최소 85% 이상이면 adequate하다고 설명한다.

가장 좋은 성능은 feature extraction 기반 ANN에서 나왔다. SBP의 경우 4-layer feature-trained ANN이 가장 좋은 결과를 보였으며, MAE 5.26 mmHg, 10 mmHg 미만 error 비율 86.1%, $R^2 = 0.798$을 기록했다. DBP의 경우 2-layer feature-trained ANN이 가장 좋았고, MAE 2.96 mmHg, 10 mmHg 미만 error 비율 95.8%, $R^2 = 0.692$를 기록했다. MAP 역시 2-layer feature-trained ANN이 가장 좋았고, MAE 3.27 mmHg, 10 mmHg 미만 error 비율 94.5%, $R^2 = 0.771$을 기록했다.

저자들이 최종적으로 제시한 error는 다음과 같다. SBP는 $5.26 \pm 6.53$ mmHg, DBP는 $2.96 \pm 3.31$ mmHg, MAP은 $3.27 \pm 3.55$ mmHg이다. 논문은 이 세 모델 모두 국제 clinical blood pressure measurement standard를 만족한다고 해석한다.

반면 full waveform을 직접 입력으로 사용하는 deep learning model들은 discrete BP prediction에서 좋은 성능을 내지 못했다. Full waveform ANN은 SBP MAE 약 13.8–14.0 mmHg, DBP MAE 약 6.54–6.62 mmHg 수준이었다. Waveform LSTM은 SBP MAE 22.5 mmHg, DBP MAE 22.2 mmHg로 매우 낮은 성능을 보였고, $R^2$도 음수 또는 낮은 값이었다. CNN도 SBP MAE 16.1 mmHg, DBP MAE 21.6 mmHg로 부적합했다. 논문은 CNN이 DBP와 MAP에서 constant value를 출력했다고 설명하며, 이는 underfitting을 의미한다고 해석한다.

Regression SVM은 feature dataset에 적용되었지만 SBP MAE 13.8 mmHg, DBP MAE 6.11 mmHg, MAP MAE 7.13 mmHg로 feature ANN보다 훨씬 낮은 성능을 보였다. Binary decision tree는 DBP와 MAP에서 10 mmHg 미만 error 비율이 각각 90.2%, 86.7%로 기준을 만족했지만, SBP는 77.0%로 기준에 미달했다.

이 결과는 논문에서 다소 흥미로운 결론을 낳는다. 일반적으로 deep learning이 waveform feature를 자동으로 추출할 것으로 기대할 수 있지만, 이 연구에서는 PPG waveform에서 명시적으로 feature를 추출한 뒤 ANN으로 학습하는 방식이 훨씬 더 효과적이었다. 저자들은 더 복잡한 ResNet 또는 U-Net 구조와 추가 hyperparameter tuning이 deep learning 성능을 개선할 가능성은 인정하지만, 본 연구의 실험 범위에서는 feature-trained ANN이 우수했다.

### Waveform prediction 결과

Waveform prediction의 주 metric은 Pearson correlation coefficient $r$이다. 최종 모델인 3-layer fully connected network with Pearson correlation regression layer는 test data에서 $r = 0.864$를 달성하였다. 이는 predicted waveform과 actual ABP waveform 사이에 strong correlation이 있음을 의미한다.

비교 모델로는 2-layer biLSTM, 2-layer LSTM, Pearson loss 또는 standard regression loss를 사용한 구조들이 실험되었다. Standard regression layer를 사용한 2-layer biLSTM은 $r = 0.801$, Adam optimizer를 사용한 biLSTM은 $r = 0.797$, Pearson loss를 사용한 biLSTM은 $r = 0.802$를 기록했다. 2-layer LSTM은 standard regression에서 $r = 0.781$, Pearson loss에서 $r = 0.818$을 기록했다. 이 결과를 바탕으로 저자들은 더 단순한 fully connected ANN이 LSTM 계열보다 계산적으로 효율적이면서도 더 높은 correlation을 제공한다고 판단했다.

논문은 normotensive sample과 hypertensive sample에 대한 generated ABP waveform 예시를 제시하였다. 제공된 텍스트에서는 figure의 세부 waveform 형태를 직접 분석할 수는 없지만, 저자들은 discrete BP model과 waveform shape model을 결합하여 pressure scale을 갖는 ABP waveform을 생성할 수 있음을 보였다.

### 기존 연구와의 비교

저자들은 기존 연구들과 비교하면서 본 연구의 강점을 강조한다. Kurylyak et al.은 PPG feature와 ANN을 사용하여 SBP MAE 3.8 mmHg, DBP MAE 2.2 mmHg를 보고했지만, 데이터 규모가 약 4시간 waveform으로 작고 환자 수가 명확하지 않았다. Chowdhury et al.은 219명의 data와 Gaussian process regression을 사용해 SBP MAE 3.02 mmHg, DBP MAE 1.74 mmHg를 보고했다. Slapničar et al.은 510명과 700시간 data를 사용한 deep learning 연구였지만 SBP MAE 9.4 mmHg, DBP MAE 6.9 mmHg로 성능이 낮았다.

본 연구는 1669명, 890시간의 데이터를 사용했다는 점에서 규모가 크며, SBP, DBP, MAP 모두에서 clinical standard를 만족한다고 주장한다. 다만 일부 기존 연구가 더 낮은 MAE를 보고한 경우도 있으므로, 이 논문의 차별점은 단순히 최저 error가 아니라 큰 dataset 규모와 waveform prediction 결합에 있다.

Waveform prediction에서 Athaya and Choi의 U-Net 연구는 Pearson correlation 0.993을 보고했지만, 그 연구는 100명의 subject를 사용했고 PPG와 ABP를 phase-matched한 뒤 학습했다. 본 연구는 1669명으로 훨씬 큰 dataset을 사용했으며, 더 단순한 ANN으로 $r = 0.864$를 얻었다. 저자들은 phase matching을 본 연구의 architecture에 적용하면 더 좋은 결과가 가능할 수 있다고 언급한다.

## 5. 강점, 한계

이 논문의 첫 번째 강점은 PPG만을 사용하여 SBP, DBP, MAP, ABP waveform shape를 모두 추정하려 했다는 점이다. ECG, cuff measurement, demographic feature를 사용하지 않고 PPG signal만으로 discrete pressure와 waveform을 모두 생성하려는 접근은 실제 wearable 또는 pulse oximeter 기반 구현 가능성을 높인다. 추가 sensor가 필요 없다는 점은 비용, 편의성, 착용성 측면에서 중요하다.

두 번째 강점은 discrete BP prediction과 waveform shape prediction을 분리한 architecture design이다. 이 설계는 waveform model이 absolute pressure amplitude를 직접 맞추는 어려운 문제를 피하고, shape correlation에 집중하도록 한다. 이후 feature-trained ANN이 예측한 SBP, DBP, MAP을 이용해 pressure scale을 부여한다. 이러한 decomposition은 physiological signal modeling에서 해석 가능하고 실용적인 선택이다.

세 번째 강점은 비교적 큰 데이터 규모이다. 최종 dataset은 1669명의 ICU 환자와 890시간의 waveform으로 구성되며, 저자들은 이 규모가 검토한 기존 연구들보다 크다고 설명한다. 큰 데이터셋은 모델의 일반화 가능성을 어느 정도 높이는 장점이 있다.

네 번째 강점은 feature extraction 기반 모델과 waveform 기반 deep learning 모델을 함께 비교했다는 점이다. 이 논문은 단순히 deep learning이 항상 우수하다고 가정하지 않고, handcrafted physiological feature의 효과를 실험적으로 보여준다. 실제 결과에서는 feature-trained ANN이 full waveform ANN, LSTM, CNN보다 명확히 우수했다.

다섯 번째 강점은 waveform prediction의 loss function을 task objective에 맞게 설계했다는 점이다. Pearson correlation 기반 loss는 absolute amplitude보다 waveform shape similarity를 최적화한다. 이는 이후 SBP, DBP, MAP으로 scale을 조정하는 전체 pipeline과 잘 맞는다.

그러나 한계도 명확하다. 가장 중요한 한계는 dataset이 ICU 환자 중심이라는 점이다. MIMIC-III Waveform Database는 중환자실 환자 데이터를 포함하므로, 평균 SBP가 높고 cardiovascular disease 또는 acute clinical condition이 반영되어 있을 가능성이 크다. 따라서 healthy individual, outpatient, 일반 wearable 사용자에게 같은 성능이 유지된다고 말할 수 없다. 저자들도 healthy individuals의 data와 다양한 population에서 추가 검증이 필요하다고 명시한다.

두 번째 한계는 clinical and demographic data를 사용하지 않았다는 점이다. 나이, 성별, 체중, 질병 상태, 약물 사용, skin tone, perfusion 상태 등은 PPG morphology와 혈압 관계에 영향을 줄 수 있다. 그러나 본 연구는 waveform database만 사용했고 MIMIC-III clinical database는 라이선스와 교육 절차 문제로 사용하지 않았다. 따라서 model bias를 demographic 또는 clinical subgroup별로 분석하지 못했다.

세 번째 한계는 data split strategy에 대한 불확실성이다. 논문은 dataset을 70% training, 15% validation, 15% test로 split했다고 설명하지만, 제공된 텍스트에서는 patient-level split을 수행했다는 명시가 없다. 만약 segment-level random split이 사용되었다면 동일 환자의 waveform segment가 training과 test에 동시에 포함될 수 있으며, 이는 성능을 과대평가할 위험이 있다. 논문이 실제로 이를 방지했는지는 제공 텍스트만으로 판단할 수 없으므로, 이 부분은 재현성과 임상적 일반화 측면에서 중요한 확인 사항이다.

네 번째 한계는 discrete BP prediction에서 deep learning model들이 underfitting했다는 점이다. CNN과 LSTM이 낮은 $R^2$ 또는 constant output을 보였다는 것은 architecture와 hyperparameter가 충분히 최적화되지 않았을 수 있음을 의미한다. 저자들도 ResNet, U-Net 같은 더 복잡한 구조를 적용하면 결과가 달라질 가능성을 인정한다.

다섯 번째 한계는 waveform prediction의 평가가 주로 Pearson correlation에 의존한다는 점이다. Pearson correlation은 waveform shape similarity를 측정하는 데 유용하지만, clinical decision에 필요한 systolic peak timing, dicrotic notch fidelity, pulse pressure dynamics, waveform-derived hemodynamic indices가 얼마나 정확한지는 별도로 평가되어야 한다. 이 논문은 waveform shape의 diagnostic utility를 직접 검증하지는 않았다.

여섯 번째 한계는 실제 hardware deployment가 아직 수행되지 않았다는 점이다. 논문은 pulse oximeter나 wearable device에 적용 가능성을 언급하지만, 실제 PPG sensor hardware, motion artifact, skin tone, low perfusion, ambient light, real-time processing, on-device computation 문제는 실험하지 않았다. 따라서 실제 제품화 또는 임상 적용까지는 추가 연구가 필요하다.

비판적으로 보면, 이 논문은 PPG 기반 continuous BP estimation의 가능성을 강하게 보여주지만, clinical-grade device로 바로 사용하기에는 아직 부족하다. 특히 ICU dataset 기반 성능, potential subject leakage, demographic bias 미분석, real-world PPG artifact 미검증은 향후 반드시 해결해야 할 부분이다. 그럼에도 discrete BP와 waveform을 결합한 구조는 연구적으로 가치 있는 방향이다.

## 6. 결론

이 논문은 PPG signal만을 이용하여 non-invasive, cuffless, continuous blood pressure measurement를 구현할 수 있는 machine learning pipeline을 제안하였다. 연구는 MIMIC-III Waveform Database에서 1669명의 환자와 890시간의 PPG-ABP waveform을 사용하여 SBP, DBP, MAP, ABP waveform shape를 예측하였다.

가장 좋은 discrete BP prediction은 feature extraction 기반 ANN에서 나왔다. SBP는 $5.26 \pm 6.53$ mmHg, DBP는 $2.96 \pm 3.31$ mmHg, MAP은 $3.27 \pm 3.55$ mmHg의 error를 기록했으며, 세 모델 모두 10 mmHg 미만 error 비율이 85% 이상이었다. 논문은 이를 AAMI/ESH/ISO clinical blood pressure measurement standard를 만족하는 결과로 해석한다. Waveform prediction에서는 Pearson correlation $r = 0.864$를 달성하여, PPG로부터 ABP waveform shape를 어느 정도 복원할 수 있음을 보였다.

이 연구의 주요 기여는 discrete BP value prediction과 waveform shape prediction을 분리하고, 다시 결합하여 pressure scale을 갖는 continuous ABP waveform을 생성한 점이다. 이 방식은 PPG의 저비용, 비침습, 연속 측정 가능성이라는 장점을 활용하면서, cuff-based measurement와 invasive arterial line 사이의 간극을 줄이려는 시도이다.

실제 적용 측면에서 이 연구는 병원 내 pulse oximeter 기반 monitoring, 저자원 의료 환경의 continuous BP monitoring, wearable health device 기반 home monitoring으로 확장될 가능성이 있다. 그러나 이를 위해서는 healthy population과 diverse demographic group에 대한 external validation, clinical feature 기반 bias analysis, motion artifact와 skin tone 영향 분석, 실제 hardware deployment, patient-level validation split 확인, medical device standard에 따른 prospective validation이 필요하다.

종합하면, 이 논문은 PPG 기반 cuffless continuous BP estimation이 임상적으로 의미 있는 방향임을 보여주는 실험적 근거를 제공한다. 특히 waveform prediction까지 포함한 two-stage framework는 향후 non-invasive cardiovascular monitoring 연구에서 유용한 출발점이 될 수 있다.
