# A PPG-Based Calibration-Free Cuffless Blood Pressure Estimation Method Using Cardiovascular Dynamics

* **저자**: Hamed Samimi, Hilmi R. Dajani
* **발표연도**: 2023

## 1. 논문 개요

이 논문은 단일 photoplethysmogram, 즉 PPG 신호만을 이용하여 cuff 없이 혈압을 추정하는 방법을 제안한다. 특히 기존 cuffless blood pressure estimation 방법에서 흔히 필요한 개인별 calibration 단계를 제거하고, PPG 신호의 morphology 정보와 cardiovascular dynamics 정보를 결합하여 calibration-free 방식으로 systolic blood pressure, 즉 SBP와 diastolic blood pressure, 즉 DBP를 추정하는 것이 핵심 목표이다.

전통적인 혈압 측정 방식인 cuff 기반 sphygmomanometer 또는 oscillometry는 비교적 안전하고 널리 사용되지만, 수면 중 사용하기 불편하고 연속 측정에 적합하지 않다. 침습적 arterial catheter 방식은 정확하지만 감염, 혈전, 출혈, ischemia 등의 위험이 있어 일반적인 가정 또는 일상 환경에서 사용할 수 없다. 따라서 논문은 wearable device 환경에서 사용할 수 있는 비침습적이고 연속적인 cuffless blood pressure monitoring 방법의 필요성을 배경으로 한다.

기존 cuffless 방법 중 대표적인 pulse transit time, 즉 PTT 기반 방식은 두 개 이상의 센서가 필요하고, 신호 간 시간 동기화가 정확해야 한다는 문제가 있다. 반면 PPG 기반 방식은 하나의 optical sensor만으로 혈액량 변화에 따른 파형을 얻을 수 있으므로 wearable 환경에 적합하다. 그러나 단일 PPG 기반 혈압 추정은 PTT 기반 방식에 비해 생리학적 해석과 안정성이 충분히 확립되지 않았고, morphology feature만으로는 충분한 정확도를 얻기 어렵다는 문제가 있다.

이 논문은 이러한 문제를 해결하기 위해 PPG 파형의 개별 pulse morphology뿐 아니라, peak와 trough 사이의 시간적 변동에서 얻어지는 cardiovascular dynamics 정보를 함께 사용한다. 또한 이전 연구에서 사용했던 calibrated mathematical model을 PPG morphology 기반 추정값으로 대체할 수 있는지 평가함으로써, 실제 사용 시 개인별 cuff calibration이 필요 없는 calibration-free 모델을 구성하고자 한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 PPG 신호 안에 혈압과 관련된 두 종류의 정보가 함께 들어 있다는 점에 있다. 첫 번째는 pulse waveform의 모양에서 얻어지는 morphology 정보이다. 예를 들어 cardiac period, systolic upstroke time, diastolic time, pulse height의 특정 비율 지점에서의 systolic width와 diastolic width 등이 이에 해당한다. 두 번째는 PPG peak와 trough의 시간 간격 변화, 즉 interbeat interval, IBI에서 얻어지는 cardiovascular dynamics 정보이다. 이 정보는 단일 pulse의 모양이 아니라 여러 박동 사이의 변동성과 동역학적 특성을 반영한다.

기존의 많은 PPG 기반 혈압 추정 연구는 pulse morphology feature를 직접 machine learning 또는 deep learning 모델에 입력하여 혈압을 예측했다. 반면 이 논문은 morphology feature만 사용하는 것이 아니라, morphology feature로부터 한 번 혈압을 추정한 뒤, 그 추정값을 cardiovascular dynamics feature와 결합한다. 즉 morphology 기반 모델은 최종 혈압 추정의 보조 정보 또는 calibration을 대체하는 정보로 사용된다.

논문에서 중요한 차별점은 이전 연구에서 사용했던 calibration stage를 PPG morphology 기반 estimation으로 대체하려 했다는 점이다. 저자들의 이전 연구에서는 PPG dynamics feature에 calibrated mathematical model의 출력값을 추가하여 혈압을 추정했다. 하지만 이 경우 실제 사용을 위해 개인별 calibration이 필요할 수 있다. 본 연구에서는 PPG morphology 기반 추정값과 calibrated mathematical model 기반 추정값 사이의 상관관계가 높다는 점을 보이고, morphology 기반 추정값이 calibration feature를 대체할 수 있음을 실험적으로 제시한다.

또 다른 핵심 아이디어는 one point-to-point, 즉 oPTP 방식 대신 mean point-to-point, 즉 mPTP 방식을 사용한 것이다. oPTP는 개별 pulse에서 feature를 추출하는 방식으로 잡음, pulse detection 오류, 일시적 waveform distortion에 민감할 수 있다. 논문은 일정 구간 동안 여러 pulse에서 feature를 추출한 뒤 평균을 내는 mPTP 방식을 사용하여 더 안정적인 morphology feature를 구성한다.

## 3. 상세 방법 설명

논문의 전체 파이프라인은 크게 세 단계로 이해할 수 있다. 첫째, PPG 신호에서 morphology feature와 cardiovascular dynamics feature를 각각 추출한다. 둘째, morphology feature를 이용하여 SBP와 DBP를 추정하는 neural network model을 학습한다. 셋째, morphology 기반 혈압 추정값을 cardiovascular dynamics feature와 결합하여 최종 SBP와 DBP를 추정한다.

### 3.1 사용 데이터셋

논문은 두 개의 공개 bio-signal dataset을 사용한다.

첫 번째는 University of Queensland Vital Signs Dataset이다. 이 데이터셋은 수술실에서 수집된 ECG, PPG, noninvasive arterial blood pressure waveform을 포함한다. 총 32명의 환자 데이터가 있으며, 이 논문에서는 시각적으로 품질을 검사한 뒤 30명의 환자 데이터를 사용했다. 샘플링 주파수는 100 Hz이며, 기록 길이는 13분에서 5시간까지 다양하고 중앙값은 105분이다. 이 데이터는 마취 유도와 회복 과정에서 vital sign이 빠르게 변할 수 있다는 점에서 일반적인 ICU 데이터와 다른 특성을 가진다.

두 번째는 UCI Machine Learning Repository에 있는 MIMIC II 기반 subset이다. 이 데이터셋은 ICU 환경에서 수집된 PPG와 arterial blood pressure, 즉 ABP 신호를 포함한다. 원본 MIMIC II는 PhysioNet에서 제공되는 대규모 multiparameter recording이다. UCI subset은 고정 길이 signal block으로 분할하고, smoothing filter를 적용하며, 비정상적인 혈압 또는 heart rate 값을 제거하고, PPG autocorrelation을 이용해 연속 pulse 간 변화가 큰 block을 제거하는 전처리 과정을 거친 데이터이다. 본 연구에서는 8분에서 10분 길이의 신호를 가진 200명 데이터를 주 학습 및 평가에 사용했고, 추가로 25명의 환자 데이터를 최종 never-seen test set으로 따로 분리했다.

혈압 reference 값은 데이터셋에 따라 다르게 계산되었다. UCI 데이터에서는 ABP waveform의 peak 평균을 SBP reference로, trough 평균을 DBP reference로 사용했다. University of Queensland 데이터에서는 제공된 beat-to-beat noninvasive SBP와 DBP 값을 신호 구간 전체에 대해 평균하여 reference 값을 만들었다.

### 3.2 PPG morphology feature 추출

논문은 morphology 기반 혈압 추정을 위해 두 종류의 feature set을 사용한다.

첫 번째는 feedforward artificial neural network model에 사용된 21개 morphology feature이다. 이 feature들은 선행 연구를 기반으로 하되, 개별 pulse 하나에서 값을 뽑는 oPTP가 아니라 전체 신호 구간에서 평균을 내는 mPTP 방식으로 계산된다. 사용된 feature는 cardiac period, systolic upstroke time, diastolic time, pulse height의 10%, 25%, 33%, 50%, 66%, 75% 지점에서의 diastolic width, 같은 높이 지점에서 systolic width와 diastolic width의 합, 그리고 같은 높이 지점에서 diastolic width와 systolic width의 비율이다.

예를 들어 diastolic width at 10%는 pulse peak amplitude의 10%에 해당하는 높이에서, peak와 같은 시간축에 있는 지점과 diastolic region에 위치한 같은 amplitude 지점 사이의 시간 차이로 계산된다. 이러한 width 계열 feature는 PPG pulse가 얼마나 빠르게 상승하고, 얼마나 천천히 하강하며, 말초혈관 및 arterial compliance와 관련된 waveform shape이 어떻게 변하는지를 반영할 수 있다.

두 번째는 deep learning model에 사용된 7개 morphology feature이다. 이 feature들은 cardiac period, diastolic time, pulse height의 25%와 75% 지점에서의 diastolic width, 33%와 75% 지점에서의 systolic width와 diastolic width의 합, 그리고 10% 지점에서의 diastolic width 대 systolic width 비율로 구성된다. 이 7개 feature 역시 mPTP 방식으로 계산된다.

### 3.3 PPG cardiovascular dynamics feature 추출

cardiovascular dynamics feature는 PPG signal의 peak와 trough를 검출하고, 이로부터 interbeat interval, 즉 IBI series를 구성하여 계산된다. 논문은 Pan and Tompkins의 QRS detection algorithm을 수정하여 PPG peak와 trough를 검출했다고 설명한다. SBP 추정에는 PPG peak의 IBI를 사용하고, DBP 추정에는 PPG trough의 IBI를 사용한다.

이 연구는 저자들의 이전 연구에서 사용한 IBI feature들을 재사용한다. University of Queensland 데이터셋에서는 calibration-free 모델에 대해 SBP 추정용으로 NNx, $\alpha_1$, LF, HF를 사용했고, DBP 추정용으로 SDNN, RMSSD, SD1, LF, $\alpha_1$을 사용했다. calibrated model에서는 SBP 추정에 SampleEn과 $\alpha_1$, DBP 추정에 SampleEn과 HF를 사용하고, 여기에 calibrated mathematical model의 혈압 추정값을 추가 feature로 사용했다.

UCI 데이터셋 기반 이전 연구에서는 SBP 추정에 mean IBI, NNx, pNNx, SD2, $\alpha_1$을 사용했고, DBP 추정에는 mean IBI, NNx, pNNx, PRVTi, SampleEn, IBI ratio of LF/HF를 사용했다. 최종 proposed model에서는 25명의 never-seen test 환자에 대해 SDNN, PRVTi, TINN, LF, HF, $\alpha_1$, $\alpha_2$라는 7개의 PPG dynamics feature와 morphology 기반 혈압 추정값을 함께 사용했다.

이러한 feature들은 단순한 pulse shape이 아니라 박동 간 변동성, 주파수 영역 특성, 비선형 동역학적 특성 등을 반영한다. 논문은 이 정보가 cardiovascular system의 dynamics와 관련되어 있으며, morphology feature만 사용할 때보다 혈압 추정 정확도를 높일 수 있다고 주장한다.

### 3.4 Morphology 기반 neural network model

논문은 morphology feature만으로 SBP와 DBP를 추정하기 위해 여러 neural network 구조를 비교한다.

21개 morphology feature를 사용하는 feedforward artificial neural network는 21개의 입력과 2개의 출력 neuron을 가진다. 출력은 SBP와 DBP이다. 이 구조는 두 개의 hidden layer를 포함하며, 첫 번째 hidden layer는 35개 neuron, 두 번째 hidden layer는 20개 neuron으로 구성된다. Hidden layer에는 sigmoid activation function을 사용하고, output layer에는 linear activation function을 사용한다.

7개 morphology feature를 사용하는 deep learning model로는 세 가지가 비교된다. 첫째, feedforward deep neural network는 70, 100, 150개의 neuron을 가진 세 개의 hidden layer로 구성된다. 둘째, LSTM model은 sequential time-domain data 처리를 위해 사용되며, 64개와 512개의 neuron을 가진 두 개의 hidden layer로 구성된다. 셋째, GRU model은 LSTM과 유사하지만 더 적은 parameter를 사용하며, 128, 256, 512개의 neuron을 가진 세 개의 hidden layer로 구성된다. Feedforward deep neural network에는 sigmoid activation function을 사용하고, LSTM과 GRU에는 ReLU activation function을 사용한다.

실험 결과, deep learning 기반 morphology estimation 중에서는 feedforward deep neural network가 가장 좋은 성능을 보였다. 이 결과를 바탕으로 최종 proposed model에서는 morphology feature 기반 feedforward deep neural network의 혈압 추정값을 calibration replacement feature로 사용한다.

### 3.5 최종 blood pressure estimation model

최종 혈압 추정 모델은 morphology 기반 혈압 추정값과 PPG dynamics feature를 결합한다. 논문에서 설명한 최종 구조는 세 개의 subnetwork로 이해할 수 있다. 왼쪽 subnetwork는 7개의 PPG morphology feature를 입력받아 SBP와 DBP를 먼저 추정한다. 이 subnetwork는 세 개의 hidden layer와 linear output layer를 가진다.

그 다음 오른쪽에는 SBP 추정용 subnetwork와 DBP 추정용 subnetwork가 각각 존재한다. 각 subnetwork는 8개의 입력을 가진다. 이 8개 입력은 7개의 cardiovascular dynamics feature와 앞 단계 morphology network에서 추정된 혈압값 하나로 구성된다. 예를 들어 최종 SBP network에는 7개의 dynamics feature와 morphology 기반 SBP estimate가 들어가고, 최종 DBP network에는 7개의 dynamics feature와 morphology 기반 DBP estimate가 들어간다. 각 subnetwork는 sigmoid hidden layer와 linear output layer를 가진다.

학습 및 평가에는 leave-one-out, 즉 LOO 방식이 사용된다. 각 반복에서 한 명의 환자 데이터를 test set으로 제외하고, 나머지 데이터를 training과 validation에 사용한다. Training data 중 15%는 validation set으로 분리되며, overfitting을 줄이기 위해 early stopping을 사용한다. 최종 testing 단계에서는 200명의 UCI 데이터 전체로 모델을 학습한 뒤, 이전에 모델이 본 적 없는 25명의 환자 데이터에 적용한다.

혈압 추정 성능은 mean error, ME, standard deviation of error, SDE, mean absolute error, MAE로 평가된다. 논문에서 ME는 다음과 같이 정의된다.

$$
ME = \frac{\sum_{i=1}^{n}(y_i - x_i)}{n}
$$

여기서 $y_i$는 모델의 예측값이고, $x_i$는 reference blood pressure 값이다. ME는 예측이 평균적으로 reference보다 높은지 낮은지를 나타내는 bias 성격의 지표이다.

MAE는 다음과 같이 정의된다.

$$
MAE = \frac{\sum_{i=1}^{n}|y_i - x_i|}{n}
$$

MAE는 오차의 방향을 무시하고 평균적인 절대 오차 크기를 나타낸다. 혈압 추정에서 실제 사용자가 체감할 수 있는 오차 크기를 이해하는 데 중요한 지표이다.

SDE는 다음과 같이 정의된다.

$$
SDE = \sqrt{\frac{\sum_{i=1}^{n}|e_i - \bar{e}|^2}{n}}
$$

여기서 $e_i = y_i - x_i$이고, $\bar{e}$는 error의 평균이다. SDE는 예측 오차가 얼마나 넓게 퍼져 있는지를 나타내며, Bland–Altman analysis에서 limits of agreement와 관련된 중요한 지표이다.

논문은 또한 sensor fusion 방식도 비교한다. 두 추정값 $x_1$, $x_2$와 각 noise variance $\sigma_1^2$, $\sigma_2^2$가 있을 때 결합 추정값 $x_3$는 다음과 같이 계산된다.

$$
x_3 = \sigma_3^2(\sigma_1^{-2}x_1 + \sigma_2^{-2}x_2)
$$

결합 추정의 variance는 다음과 같다.

$$
\sigma_3^2 = (\sigma_1^{-2} + \sigma_2^{-2})^{-1}
$$

그러나 실험 결과에서는 단순 fusion보다 morphology estimate를 dynamics feature와 함께 ANN feature로 넣는 feature combination 방식이 더 좋은 성능을 보였다.

## 4. 실험 및 결과

논문은 크게 세 가지 실험 단계를 수행한다. 첫 번째는 University of Queensland 데이터셋의 30명 환자에 대한 실험이다. 이 단계의 목적은 PPG morphology 기반 추정값이 calibrated mathematical model을 대체할 수 있는지를 검토하는 것이다. 두 번째는 UCI 데이터셋의 200명 환자에 대해 여러 morphology model과 dynamics feature 결합 방식을 비교하는 것이다. 세 번째는 200명 데이터로 학습한 최종 모델을 previously unseen 25명 환자 데이터에 적용하는 것이다.

### 4.1 University of Queensland 30명 데이터 결과

30명 환자 데이터에서 먼저 세 가지 단독 추정 방법이 비교되었다. 첫째는 PPG IBI 기반 dynamics feature만 사용하는 방법이다. 둘째는 PPG morphology feature만 사용하는 방법이다. 셋째는 calibrated mathematical model이다.

SBP의 경우 PPG IBI 방식은 ME -0.39 mmHg, SDE 22.16 mmHg, MAE 15.26 mmHg를 보였다. PPG morphology 방식은 ME 0.06 mmHg, SDE 14.22 mmHg, MAE 10.10 mmHg로 IBI 단독 방식보다 더 낮은 SDE와 MAE를 보였다. Calibrated mathematical model은 ME 3.18 mmHg, SDE 12.49 mmHg, MAE 9.11 mmHg로 세 단독 방법 중 SBP에서 가장 낮은 MAE를 보였다.

DBP의 경우 PPG IBI 방식은 ME 0.14 mmHg, SDE 10.97 mmHg, MAE 7.54 mmHg였다. PPG morphology 방식은 ME 0.01 mmHg, SDE 8.32 mmHg, MAE 6.16 mmHg였고, calibrated mathematical model은 ME 0.45 mmHg, SDE 8.36 mmHg, MAE 5.47 mmHg였다. DBP에서도 IBI 단독보다 morphology 또는 calibration 기반 방식이 더 좋은 결과를 보였다.

그 다음 논문은 fusion 방식과 feature combination 방식을 비교했다. Fusion 방식에서는 PPG IBI 추정값과 morphology 추정값, 또는 IBI 추정값과 calibrated mathematical model 추정값, 또는 세 방법의 추정값을 sensor fusion 공식으로 결합했다. 그러나 fusion 결과는 feature combination보다 낮은 정확도를 보였다. 예를 들어 세 방법을 모두 fusion한 경우 SBP MAE는 8.95 mmHg, DBP MAE는 6.09 mmHg였다.

Feature combination 방식에서는 morphology 기반 추정값 또는 calibrated mathematical model 기반 추정값을 IBI feature에 추가한 뒤 ANN으로 최종 혈압을 추정했다. 이 방식이 훨씬 더 좋은 성능을 보였다. PPG IBI와 PPG morphology를 결합한 경우 SBP는 ME -1.51 mmHg, SDE 11.23 mmHg, MAE 7.50 mmHg였고, DBP는 ME -0.42 mmHg, SDE 6.14 mmHg, MAE 4.94 mmHg였다. PPG IBI와 calibrated mathematical model을 결합한 경우 SBP MAE는 8.89 mmHg, DBP MAE는 4.92 mmHg였다. 세 정보를 모두 결합한 경우 SBP는 ME -1.15 mmHg, SDE 10.69 mmHg, MAE 7.41 mmHg였고, DBP는 ME -1.11 mmHg, SDE 6.07 mmHg, MAE 4.90 mmHg였다.

중요한 점은 morphology estimate만 추가한 경우와 calibrated mathematical model estimate를 추가한 경우의 성능이 유사했다는 것이다. 또한 morphology와 calibration을 둘 다 추가했을 때 성능 향상은 크지 않았다. 이 결과는 morphology estimate가 calibrated mathematical model과 유사한 역할을 할 수 있음을 시사한다.

이를 더 확인하기 위해 논문은 세 추정 방법 간 correlation coefficient를 계산했다. PPG IBI와 calibration 사이의 correlation은 SBP 0.23, DBP 0.45였다. PPG IBI와 morphology 사이의 correlation은 SBP 0.16, DBP 0.29였다. 반면 morphology와 calibration 사이의 correlation은 SBP 0.74, DBP 0.78로 상대적으로 높았다. 즉 morphology 기반 추정값과 calibrated mathematical model 기반 추정값은 서로 비슷한 정보를 담고 있으며, morphology estimate가 calibration feature를 대체할 가능성이 있다는 결론을 뒷받침한다.

### 4.2 UCI 200명 데이터 결과

UCI 데이터셋 200명에 대해서는 PPG IBI feature, PPG morphology feature, 그리고 두 정보를 결합한 방법들이 비교되었다. Morphology 기반 모델로는 feedforward neural network, feedforward deep neural network, LSTM, GRU가 사용되었다.

PPG IBI 단독 방식은 SBP에서 ME 0.09 mmHg, SDE 18.81 mmHg, MAE 14.49 mmHg를 보였고, DBP에서 ME 0.03 mmHg, SDE 7.91 mmHg, MAE 5.75 mmHg를 보였다.

Morphology만 사용한 모델 중 가장 좋은 성능은 feedforward deep neural network에서 나타났다. 이 모델은 SBP에서 ME 0.36 mmHg, SDE 13.81 mmHg, MAE 11.24 mmHg를 보였고, DBP에서 ME 0.12 mmHg, SDE 6.49 mmHg, MAE 4.75 mmHg를 보였다. 반면 LSTM과 GRU는 이 데이터와 feature 구성에서는 feedforward deep neural network보다 낮은 성능을 보였다. 예를 들어 LSTM의 SBP MAE는 15.20 mmHg, GRU의 SBP MAE는 15.30 mmHg였다.

가장 좋은 성능은 morphology 기반 feedforward deep neural network의 추정값을 PPG IBI feature와 결합했을 때 나타났다. 이 결합 모델은 SBP에서 ME 0.15 mmHg, SDE 12.40 mmHg, MAE 9.74 mmHg를 보였고, DBP에서 ME -0.01 mmHg, SDE 6.29 mmHg, MAE 4.65 mmHg를 보였다. 이는 IBI 단독 방식보다 SBP와 DBP 모두에서 성능이 개선된 결과이다.

논문은 이를 통해 PPG dynamics 정보가 morphology 기반 혈압 추정의 정확도를 높이는 데 실제로 기여한다고 해석한다. 특히 feedforward deep neural network 기반 morphology estimate와 IBI feature의 결합은 SBP에서 SDE와 MAE를 크게 줄였다.

### 4.3 Never-seen 25명 데이터 결과

최종 실험에서는 UCI 데이터셋의 200명 환자로 모델을 학습하고, 모델이 이전에 본 적 없는 25명 환자의 PPG 신호에 적용했다. 이때 최종 모델은 7개의 PPG dynamics feature, 즉 SDNN, PRVTi, TINN, LF, HF, $\alpha_1$, $\alpha_2$와 feedforward deep neural network로부터 얻은 morphology 기반 혈압 추정값을 함께 사용했다.

최종 test 결과는 SBP에서 ME -4.02 mmHg, SDE 10.40 mmHg, MAE 7.41 mmHg였고, DBP에서 ME -0.31 mmHg, SDE 4.89 mmHg, MAE 3.32 mmHg였다. DBP 추정 성능은 SBP보다 더 우수하게 나타났다. 논문은 이러한 현상이 이전 연구에서도 관찰되었으며, IBI dynamics와 DBP 사이의 관계가 SBP보다 더 뚜렷할 수 있다는 점, baseline DBP 값이 SBP보다 낮기 때문에 absolute error가 작게 나타날 수 있다는 점 등을 가능한 설명으로 제시한다.

Bland–Altman plot 분석과 관련하여, 논문은 일부 실험에서 mean blood pressure가 증가함에 따라 error가 증가하거나 감소하는 trend가 나타났다고 언급한다. 이는 다른 혈압 추정 연구에서도 보고된 현상이며, 두 측정 방법 간 interindividual variability 차이 때문일 수 있다고 설명한다. 다만 최종 구조에서는 SBP와 DBP 모두에서 이러한 trend가 심하지 않은 모델을 선택했다고 밝혔다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 단일 PPG 센서만으로 calibration-free cuffless blood pressure estimation을 시도했다는 점이다. 기존 PTT 기반 방식처럼 ECG와 PPG 등 두 개의 센서를 동기화할 필요가 없고, cuff 기반 calibration도 실제 사용 단계에서 요구하지 않는다. 이는 wearable device 또는 sleep monitoring 환경에서 실용적 장점을 가진다.

두 번째 강점은 morphology feature와 cardiovascular dynamics feature를 결합한 설계이다. Morphology feature는 PPG pulse shape에서 혈압 관련 정보를 추출하고, dynamics feature는 pulse 간 시간 변동성과 cardiovascular regulation 정보를 반영한다. 논문은 두 정보가 서로 완전히 중복되지 않으며, 결합했을 때 성능이 향상됨을 실험적으로 보였다. 특히 IBI feature와 morphology estimate 사이의 correlation이 낮았다는 결과는 두 정보가 상보적일 수 있음을 시사한다.

세 번째 강점은 calibration replacement라는 구체적인 문제를 실험적으로 검증했다는 점이다. University of Queensland 30명 데이터에서 morphology 기반 추정값과 calibrated mathematical model 추정값 사이의 correlation이 SBP 0.74, DBP 0.78로 나타났고, feature combination 실험에서도 morphology estimate를 calibration feature 대신 사용했을 때 유사한 성능을 보였다. 이는 논문의 calibration-free 주장에 중요한 근거가 된다.

네 번째 강점은 최종적으로 never-seen 25명 환자 데이터에서 모델을 평가했다는 점이다. 단순한 cross-validation 또는 LOO 평가뿐 아니라, 학습에 사용되지 않은 별도 환자 집합에서 성능을 확인했다는 점은 모델의 generalization 가능성을 평가하는 데 의미가 있다.

그러나 한계도 명확하다. 첫째, 데이터셋 크기가 deep learning 관점에서 충분히 크다고 보기 어렵다. 특히 University of Queensland 데이터셋의 calibration replacement 분석은 30명 환자에 기반한다. 논문 자체도 morphology estimate가 calibration을 대체할 수 있다는 결론은 제한된 데이터셋과 짧은 측정 구간에서 평가되었으며, 더 긴 신호 기간과 더 많은 sample에서 검증해야 한다고 언급한다.

둘째, 데이터셋에 race, age, gender, weight와 같은 demographic 및 anthropometric 정보가 포함되어 있지 않다. 혈압은 나이, 성별, 체중, 인종, 질환 상태, 약물 사용 등에 영향을 받기 때문에 이러한 정보가 없는 것은 모델 해석과 성능 향상에 한계가 된다. 논문은 이러한 정보가 추가되면 혈압 추정 정확도에 기여할 수 있다고 설명한다.

셋째, PPG signal quality를 수동으로 검사했다는 점도 한계이다. 실제 wearable 환경에서는 motion artifact, poor sensor contact, skin tone, ambient light, sensor placement 등의 영향이 크다. 수동 품질 검사는 연구 단계에서는 가능하지만, real-world deployment에서는 자동 signal quality assessment가 필요하다.

넷째, UCI 데이터셋과 University of Queensland 데이터셋은 서로 다른 조건에서 수집되었다. UCI 데이터는 ICU 환자에서 얻은 invasive ABP를 reference로 사용하고, University of Queensland 데이터는 수술 및 마취 환경의 noninvasive BP를 reference로 사용한다. Invasive와 noninvasive BP 사이에는 차이가 있을 수 있으며, ICU 환자는 약물이나 inotrope의 영향을 받을 수 있다. 이 차이는 feature selection과 모델 성능 차이에 영향을 줄 수 있다.

다섯째, SBP 성능은 DBP보다 낮다. 최종 test에서 SBP MAE는 7.41 mmHg이고 SDE는 10.40 mmHg인 반면, DBP MAE는 3.32 mmHg이고 SDE는 4.89 mmHg이다. DBP는 상당히 좋은 성능을 보였지만, SBP의 오차와 variability는 실제 의료기기 수준의 신뢰성을 주장하기에는 추가 검증이 필요하다. 논문은 성능 향상을 보였지만, 의료적 사용을 위한 표준 검증 프로토콜이나 다양한 population에서의 external validation은 제공된 텍스트 기준으로 확인되지 않는다.

비판적으로 보면, 이 연구는 calibration-free PPG 기반 혈압 추정의 가능성을 보여주는 중요한 실험적 접근이지만, 실제 임상 또는 소비자 wearable 제품에 바로 적용 가능한 완성된 방법이라기보다는, morphology estimate가 calibration 역할을 일부 대체할 수 있다는 proof-of-concept 성격이 강하다. 특히 morphology와 calibration의 높은 correlation이 모든 population, sensor type, 장기 측정 상황에서 유지되는지는 아직 검증되지 않았다.

## 6. 결론

이 논문은 단일 PPG 신호만을 이용하여 cuffless, calibration-free blood pressure estimation을 수행하는 방법을 제안했다. 핵심 기여는 PPG morphology 기반 혈압 추정값을 기존 calibrated mathematical model의 대체 feature로 사용하고, 이를 PPG cardiovascular dynamics feature와 결합하여 최종 SBP와 DBP를 추정했다는 점이다.

실험 결과, 30명 University of Queensland 데이터에서 PPG morphology 기반 추정값은 calibrated mathematical model 기반 추정값과 높은 correlation을 보였고, calibration feature를 morphology estimate로 대체해도 유사한 성능을 얻을 수 있었다. 200명 UCI 데이터에서는 morphology 기반 feedforward deep neural network와 PPG IBI dynamics feature를 결합한 방법이 가장 좋은 성능을 보였다. 최종적으로 25명의 never-seen 환자 데이터에서 SBP는 ME -4.02 mmHg, SDE 10.40 mmHg, MAE 7.41 mmHg, DBP는 ME -0.31 mmHg, SDE 4.89 mmHg, MAE 3.32 mmHg의 결과를 얻었다.

이 연구는 PPG morphology만으로는 부족할 수 있는 혈압 추정을 cardiovascular dynamics 정보로 보완할 수 있음을 보여준다. 또한 calibration-free 방식을 통해 cuff 기반 보정 없이도 혈압 추정이 가능할 수 있다는 방향을 제시한다. 향후 연구에서는 더 큰 데이터셋, 더 다양한 인구집단, 장시간 측정, 자동 signal quality assessment, 실제 wearable 환경에서의 검증이 필요하다. 이러한 검증이 이루어진다면 본 연구의 접근은 수면 중 혈압 모니터링, 가정용 건강관리, wearable continuous monitoring, 그리고 다른 PPG 또는 PTT 기반 혈압 추정 방법의 정확도 향상에 기여할 가능성이 있다.
