# Blood Pressure Estimation from Photoplythmography Using Hybrid Scattering–LSTM Networks

* **저자**: Osama A. Omer, Mostafa Salah, Ammar M. Hassan, Mohamed Abdel-Nasser, Norihiro Sugita, Yoshifumi Saijo
* **발표연도**: 2024

## 1. 논문 개요

이 논문은 PPG(Photoplethysmography) 신호만을 이용하여 beat-by-beat 방식으로 혈압을 추정하는 방법을 제안한다. 논문의 목표는 PPG beat의 형태적 특징과 혈압 사이의 복잡한 관계를 더 잘 학습하기 위해, 원시 time domain 신호를 그대로 deep learning 모델에 넣는 대신 Wavelet Scattering Transform(WST)을 사용하여 더 안정적이고 설명력 있는 feature domain으로 변환한 뒤 LSTM(Long Short-Term Memory) 네트워크로 혈압을 추정하는 것이다.

혈압은 심혈관 건강을 나타내는 핵심 지표이며, 특히 고혈압은 심근경색, 뇌졸중 등 심각한 질환의 위험을 높인다. 임상적으로 invasive arterial blood pressure(ABP)는 연속 혈압 측정의 gold standard로 간주되지만, arterial cannulation이 필요하므로 일상적 사용이나 일반적인 임상 환경에는 부담이 크다. 반면 cuff 기반 non-invasive 방식은 간편하지만 연속 측정에 적합하지 않고, 반복적인 cuff inflation은 불편함과 측정 오차를 유발할 수 있다. 따라서 웨어러블 센서로 쉽게 얻을 수 있는 PPG만으로 연속 혈압을 추정하는 연구가 중요해졌다.

이 논문이 다루는 핵심 연구 문제는 두 가지이다. 첫째, PPG beat로부터 해당 beat의 ABP waveform 전체를 복원할 수 있는가이다. 둘째, PPG beat로부터 SBP(systolic blood pressure)와 DBP(diastolic blood pressure)를 직접 추정할 수 있는가이다. 저자들은 이 두 문제를 각각 “beat-by-beat cPPG-to-ABP mapping”과 “beat-by-beat cPPG-to-SBP/DBP mapping”으로 정의한다.

문제의 난점은 PPG가 고차원 time series이고, beat마다 shift, scaling, local deformation, noise가 존재한다는 점이다. 또한 사용 가능한 학습 데이터는 엄격한 cleaning을 거치면 줄어들기 때문에, 큰 deep learning 모델이 time domain에서 직접 복잡한 관계를 학습하기 어렵다. 논문은 이러한 한계를 해결하기 위해 WST를 feature extraction 단계에 도입한다. WST는 shift invariance와 local deformation stability를 제공하므로, PPG beat의 불필요한 변형을 줄이고 LSTM이 혈압 관련 형태 정보를 더 안정적으로 학습하도록 돕는다고 설명한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 PPG beat를 원시 time domain에서 바로 LSTM에 넣는 것보다, WST를 통해 안정적이고 변형에 강한 feature representation으로 바꾼 뒤 LSTM에 입력하는 것이 혈압 추정에 더 효과적이라는 것이다. PPG 신호는 측정 환경, sensor contact, 움직임, beat segmentation 오류 등에 따라 형태가 쉽게 변한다. 같은 혈압 범위에 속하는 PPG라도 여러 형태를 가질 수 있고, peak와 foot 위치가 불명확한 경우도 있다. 이러한 상황에서 deep learning 모델이 원시 신호만으로 혈압과의 관계를 학습하려면 많은 데이터와 복잡한 네트워크가 필요하다.

WST는 wavelet transform, modulus nonlinearity, low-pass averaging을 반복적으로 적용하여 translation-invariant하고 local deformation에 안정적인 표현을 만든다. CNN과 비슷한 계층적 convolution 구조를 갖지만, filter가 학습되는 것이 아니라 사전에 정의된 wavelet filter를 사용한다. 따라서 학습 데이터가 제한적인 상황에서 feature extraction을 안정적으로 수행할 수 있다는 장점이 있다.

논문은 WST를 LSTM과 결합한 hybrid Scattering–LSTM 구조를 제안한다. ScatNet 또는 WST가 PPG beat의 feature를 먼저 추출하고, LSTM이 이 feature sequence를 이용해 ABP waveform 또는 SBP/DBP 값을 예측한다. 저자들은 WST를 time domain, DCT(Discrete Cosine Transform), DWT(Discrete Wavelet Transform)와 비교하여, WST가 두 가지 추정 시나리오 모두에서 가장 우수한 RMSE와 MAE를 보인다고 보고한다.

기존 연구와의 차별점은 세 가지로 정리할 수 있다. 첫째, ECG 없이 PPG만 사용한다. 많은 cuffless BP 연구는 ECG와 PPG를 함께 사용하여 PTT(Pulse Transit Time), PAT(Pulse Arrival Time), PWV(Pulse Wave Velocity) 등을 추출하지만, 이 방식은 두 개 이상의 sensor가 필요하다. 본 논문은 PPG only 설정에 집중한다. 둘째, signal-level이 아니라 beat-by-beat 단위로 학습한다. 이를 통해 연속 혈압 모니터링에 더 적합한 단위로 문제를 구성한다. 셋째, WST를 regression 기반 BP estimation에 사용하여 PPG beat의 deformation 문제를 완화하려고 한다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

제안 방법은 PPG 신호를 입력으로 받아 beat 단위로 분할하고, 유효한 beat만 선택한 뒤, 각 beat를 여러 feature domain으로 변환하여 LSTM 기반 regression 모델에 입력하는 구조이다. 논문은 네 가지 feature domain을 비교한다. 첫 번째는 time domain이고, 두 번째는 DCT domain, 세 번째는 DWT domain, 네 번째가 제안의 핵심인 WST domain이다.

전체 처리 과정은 다음과 같이 구성된다. 먼저 PPG 신호에 band-pass filtering을 적용하여 심박 관련 주파수 대역인 0.5–8 Hz를 보존하고 noise를 줄인다. 다음으로 local minimum detection을 통해 PPG 신호를 beat 단위로 나눈다. 이후 beat interval, skewness quality index, correlation quality index를 사용하여 비정상 beat를 제거한다. 선택된 beat는 amplitude normalization과 time normalization을 거쳐 고정 길이 120 sample로 맞춰진다. 그 후 각 beat를 time, DCT, DWT, WST domain 중 하나로 변환하고, LSTM 네트워크가 이를 입력받아 ABP beat 또는 SBP/DBP 값을 예측한다.

논문은 두 가지 learning scenario를 제안한다. 첫 번째 시나리오에서는 PPG beat 하나를 입력으로 받아 대응되는 ABP beat 전체를 예측한다. 출력 ABP waveform의 maximum value를 SBP, minimum value를 DBP로 사용할 수 있다. 두 번째 시나리오에서는 PPG beat 하나를 입력으로 받아 SBP와 DBP 두 값을 직접 예측한다.

### 3.2 데이터셋과 전처리

데이터는 PhysioNet의 MIMIC II dataset에서 가져온 PPG–ABP paired data를 사용한다. 원본 데이터는 ABP, PPG, ECG가 모두 포함되어 있고 sampling frequency는 125 Hz이다. 그러나 이 논문에서는 ECG를 사용하지 않고 PPG beat와 이에 대응되는 ABP label 또는 ABP beat만 사용한다. 원본 dataset은 12,000개 record로 구성되어 있으며, 각 record는 길이가 다를 수 있다. 효율적인 처리와 filtering을 위해 record를 1024 sample segment로 나누었고, 총 30,660개 signal segment가 사용 가능한 후보로 소개된다.

논문은 기존 연구에서 cleaning된 MIMIC II 기반 dataset을 사용했다고 설명한다. 다만 텍스트상으로는 cleaning dataset의 자세한 구성과 subject-level split 여부가 충분히 명확하지 않다. 논문은 beat basis로 training, validation, test를 나누었다고 설명하며, training phase에는 175,660 beats가 있고 이 중 90%인 158,094 beats를 training, 10%인 17,566 beats를 validation에 사용한다. Testing에는 17,566 beats가 사용된다.

PPG 신호는 0.5–8 Hz band-pass filter를 통과한다. 이 대역은 심박의 fundamental frequency와 일부 harmonic 성분을 포함한다고 설명된다. ABP 신호는 혈압 magnitude 자체가 label이므로, ABP waveform의 amplitude를 바꾸는 filtering은 적용하지 않는다. ABP beat는 time normalization은 수행하지만 amplitude normalization은 하지 않는다. 이는 ABP의 amplitude가 실제 혈압값을 나타내기 때문이다.

PPG beat는 다음 식으로 0–1 범위로 normalization된다.

$$
S_n=\frac{S-\min(S)}{\max(S)-\min(S)}
$$

여기서 $S$는 원래 beat이고, $S_n$은 normalized beat이다. 이후 beat 길이는 120 sample로 time normalization된다. Time domain feature를 사용할 때는 beat interval(BI)이 추가 feature로 사용된다.

### 3.3 Beat segmentation과 beat selection

PPG 신호는 local minimum 위치를 기준으로 beat 단위로 분할된다. PPG waveform에서 연속된 local minimum 사이의 구간을 하나의 beat로 간주한다. 이때 두 successive minimum 사이의 시간 간격을 beat interval(BI)이라고 한다. BI는 심박수와 직접적으로 관련되므로 중요한 feature로 간주된다.

모든 beat가 학습에 적합한 것은 아니기 때문에, 논문은 세 가지 criteria로 valid beat를 선택한다.

첫 번째 기준은 beat interval이다. 정상적인 heart rate 범위를 40–180 bpm으로 두면, beat interval은 대략 0.33–1.5초 범위에 있어야 한다. 따라서 논문은 $0.33 \le BI \le 1.5$ 범위의 beat만 사용한다. 너무 짧거나 긴 beat는 segmentation 오류나 noise 가능성이 높다.

두 번째 기준은 skewness quality index(SQI)이다. 정상적인 PPG beat는 보통 positive skewness를 갖는다고 설명된다. 논문은 skewness를 다음과 같이 계산한다.

$$
SQI=\frac{\sum_{i=1}^{N}(Y_i-\tilde{Y})^3/N}{S^3}
$$

여기서 $Y_i$는 beat의 $i$번째 amplitude, $\tilde{Y}$는 beat 평균, $S$는 표준편차, $N$은 beat sample 수이다. 논문은 $0 \le SQI \le 1$ 범위의 beat를 valid beat로 사용한다. 음수 SQI는 left-skewed beat를 의미하므로 제거하고, 너무 큰 positive SQI도 long-tailed abnormal beat일 수 있어 제외한다.

세 번째 기준은 beat correlation quality index(CQI)이다. 이는 beat가 standard beat와 얼마나 유사한지를 correlation으로 평가한다. 논문은 correlation이 너무 낮은 beat를 제거하며, CQI가 0.3보다 큰 beat를 사용한다고 설명한다. 이 기준은 형태가 심하게 왜곡된 beat가 학습을 방해하지 않도록 하는 역할을 한다.

이러한 beat selection 단계는 중요한 의미가 있다. PPG 기반 혈압 추정은 신호 품질에 매우 민감하므로, invalid beat를 제거하지 않으면 모델이 noise나 segmentation artifact를 혈압 관련 특징으로 잘못 학습할 수 있다.

### 3.4 Wavelet Scattering Transform

WST는 이 논문의 핵심 feature extractor이다. WST는 wavelet transform을 기반으로 하지만, 단순 wavelet transform보다 translation invariance와 local deformation stability를 더 잘 제공하도록 설계된 representation이다. 논문은 WST가 small time-warping deformation에 안정적이며, PPG beat의 shift와 scaling 문제를 완화하는 데 적합하다고 설명한다.

WST는 기본적으로 세 가지 연산을 반복한다. 첫째, 입력 신호를 wavelet filter와 convolution한다. 둘째, convolution 결과에 modulus nonlinearity를 적용한다. 셋째, low-pass filter 또는 scaling function으로 averaging을 수행한다. 이 과정을 여러 계층으로 반복하여 0차, 1차, 2차 이상의 scattering coefficients를 얻는다.

0차 scattering coefficient는 입력 신호를 scaling function $\phi_J$와 convolution하여 계산된다.

$$
S_0x=x*\phi_J(u)
$$

1차 계층에서는 먼저 wavelet convolution과 modulus를 적용한다.

$$
U_1x(u,\lambda)=|x*\psi_\lambda(u)|
$$

그 후 low-pass averaging을 적용하여 1차 scattering coefficient를 얻는다.

$$
S_1x(u,\lambda)=U_1x(u,\lambda)*\phi_J(u)=|x*\psi_\lambda(u)|*\phi_J(u)
$$

2차 계층에서는 1차 modulus output에 다시 wavelet convolution과 modulus를 적용한다.

$$
U_2x(u,\lambda_1,\lambda_2)=\left|U_1x(u,\lambda_1)*\psi_{\lambda_2}(u)\right|
$$

이를 풀어 쓰면 다음과 같다.

$$
U_2x(u,\lambda_1,\lambda_2)=\left||x*\psi_{\lambda_1}(u)|*\psi_{\lambda_2}(u)\right|
$$

일반적인 $m$차 scattering coefficient는 다음과 같이 표현된다.

$$
S_mx(u,\lambda_1,\ldots,\lambda_m)=\left|\cdots\left|x*\psi_{\lambda_1}\right|\cdots *\psi_{\lambda_m}\right|*\phi_J(u)
$$

그리고 다음 계층으로 전달되는 covariant component는 다음과 같이 표현된다.

$$
U_{m+1}x(u,\lambda_1,\ldots,\lambda_{m+1})=\left|\cdots\left|x*\psi_{\lambda_1}\right|\cdots *\psi_{\lambda_{m+1}}(u)\right|
$$

최종 scattering representation은 여러 order의 coefficient 집합이다.

$$
Sx={S_0x,S_1x,\ldots,S_mx}
$$

이 구조는 CNN과 유사하게 계층적 feature를 추출하지만, CNN과 달리 filter를 학습하지 않는다. 따라서 학습 데이터가 제한적인 경우에도 안정적 feature extraction을 제공할 수 있다. 논문은 WST가 PPG beat의 shift, scaling, local deformation에 덜 민감하므로 LSTM이 PPG와 ABP 또는 BP 사이의 관계를 더 쉽게 학습한다고 주장한다.

### 3.5 LSTM 기반 회귀 모델

WST 또는 다른 feature domain으로 변환된 PPG beat는 LSTM 네트워크에 입력된다. LSTM은 recurrent neural network의 한 종류로, input gate, forget gate, output gate, memory cell을 사용하여 sequential data의 장기 의존성을 학습한다. PPG beat는 시간적 순서를 갖는 생체 신호이므로 LSTM이 적합한 모델로 선택되었다.

논문에서 사용한 네트워크는 비교적 단순하다. 입력은 길이 120의 sequence이고, LSTM hidden unit은 20개이다. 그 뒤에 두 개의 fully connected layer가 있고, regression output layer는 mean-squared-error를 사용한다. Optimizer는 ADAM이며, learning rate는 0.005이다. 네트워크 specification에 따르면 training beat 수는 158,094개이고, epoch당 iteration 수는 191이다. Optimization function은 L2-norm으로 표기되어 있으나, 회귀 output에서는 mean-squared-error와 관련된 손실로 이해할 수 있다.

두 시나리오에서 input과 output size는 다르다. PPG-to-ABP scenario에서는 output이 길이 120의 ABP waveform이다. Time domain의 경우 input은 beat interval까지 포함하여 $121\times1$이고, DCT, DWT, WST domain에서는 $120\times1$이다. PPG-to-SBP/DBP scenario에서는 output이 $2\times1$이며, 이는 SBP와 DBP 두 값을 의미한다.

### 3.6 학습 시나리오 1: PPG-to-ABP waveform mapping

첫 번째 시나리오는 PPG beat 하나를 입력으로 받아 대응되는 ABP beat 전체를 예측한다. 이 경우 output은 120 sample 길이의 연속 waveform이다. 이후 예측된 ABP beat의 maximum value가 SBP, minimum value가 DBP로 해석될 수 있다.

이 시나리오에서는 여러 feature-domain 조합이 비교된다. 예를 들어 time domain PPG beat와 BI를 입력으로 하고 time domain ABP를 출력하는 TD+BI-TD 조합, DCT feature를 입력으로 하고 time-domain ABP를 출력하는 DCT-TD 조합, DCT feature를 입력으로 DCT-domain ABP를 출력한 뒤 inverse DCT로 복원하는 DCT-DCT 조합 등이 있다. DWT와 WST도 유사하게 비교된다. WST는 직접 invertible하지 않으므로, WST-DWT 조합에서는 PPG는 WST feature로 표현하고 ABP는 DWT domain으로 예측한 뒤 inverse DWT를 통해 waveform을 복원한다.

이 접근은 단순한 SBP/DBP 예측보다 더 어려운 문제이다. 왜냐하면 모델이 혈압의 두 scalar 값뿐 아니라 ABP waveform의 전체 morphology를 복원해야 하기 때문이다. 그러나 성공적으로 복원할 경우 beat-level continuous BP monitoring에 더 풍부한 정보를 제공할 수 있다.

### 3.7 학습 시나리오 2: PPG-to-SBP/DBP direct mapping

두 번째 시나리오는 PPG beat 하나를 입력으로 받아 해당 beat의 SBP와 DBP를 직접 예측한다. 이 경우 output은 두 개의 값으로 구성된다. ABP waveform 전체를 예측하는 것보다 output dimension은 작지만, waveform reconstruction 없이 혈압값을 바로 얻을 수 있으므로 실용적이다.

이 시나리오에서는 time, DCT, DWT, WST domain의 feature representation이 모두 동일한 LSTM architecture에 입력되어 비교된다. 성능 평가는 SBP와 DBP 각각에 대한 RMSE와 MAE로 수행된다.

## 4. 실험 및 결과

### 4.1 평가 설정

논문은 cleaned MIMIC II 기반 beat dataset을 사용한다. Training phase에는 총 175,660 beats가 사용되며, 이 중 158,094 beats가 training, 17,566 beats가 validation에 사용된다. Test에는 17,566 beats가 사용된다. 다만 텍스트에 따르면 “beat basis”로 split했다고 되어 있으며, patient-level 또는 subject-level split인지 명확히 기술되어 있지는 않다. 이는 성능 해석에서 중요한 한계로 볼 수 있다.

평가 지표는 RMSE와 MAE이다. MAE는 다음과 같이 예측값과 실제값의 절대 오차 평균으로 이해할 수 있다.

$$
MAE=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|
$$

RMSE는 squared error 평균의 제곱근이다.

$$
RMSE=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}
$$

RMSE는 큰 오차에 더 민감하고, MAE는 평균적인 예측 오차를 직관적으로 보여준다.

### 4.2 시나리오 1: ABP waveform 추정 결과

Table 3은 PPG beat에서 ABP beat를 추정하는 첫 번째 시나리오의 결과를 보여준다. 여러 feature domain 조합 중 WST 기반 방법이 가장 좋은 성능을 보인다.

Time domain 기반 TD+BI-TD의 RMSE는 11.1663, MAE는 9.8877이다. DCT-TD는 RMSE 11.3587, MAE 10.0606이고, DCT-DCT는 RMSE 11.5532, MAE 10.1669로 오히려 더 나쁘다. DWT-TD는 RMSE 10.9905, MAE 9.7415이고, DWT-DWT는 RMSE 10.8554, MAE 9.6080으로 DCT와 time domain보다 약간 개선된다.

WST-TD는 RMSE 9.2084, MAE 7.7671로 성능이 크게 개선된다. 가장 좋은 성능은 WST-DWT 조합에서 나오며, RMSE 8.9935, MAE 7.6257이다. 이는 WST feature가 PPG beat의 혈압 관련 형태 정보를 잘 보존하면서도 shift와 deformation에 덜 민감하기 때문이라고 해석된다.

논문은 Figure 9에서 high blood pressure, normal blood pressure, low blood pressure의 세 가지 예시를 통해 reconstructed ABP beat를 시각적으로 비교한다. WST-TD와 WST-DWT로 복원된 ABP beat는 ground truth ABP beat와 높은 상관성을 보인다고 설명된다. 이 결과는 WST가 단순한 scalar BP estimation뿐 아니라 waveform-level ABP reconstruction에도 유용하다는 점을 보여준다.

### 4.3 시나리오 2: SBP/DBP 직접 추정 결과

Table 4는 cPPG signal을 사용하여 SBP와 DBP를 직접 추정한 결과를 보여준다. 이 실험에서도 WST domain이 가장 우수하다.

Time domain의 DBP RMSE는 9.5636, MAE는 7.1212이고, SBP RMSE는 17.6580, MAE는 13.5720이다. DCT domain은 DBP RMSE 9.7477, MAE 7.2472, SBP RMSE 17.9762, MAE 13.9056으로 time domain보다 약간 나쁘다. DWT domain은 DBP RMSE 9.4865, MAE 7.0517, SBP RMSE 17.3914, MAE 13.3367로 time domain보다 약간 개선된다.

WST domain은 DBP RMSE 6.9164, MAE 5.0945, SBP RMSE 14.2079, MAE 10.8358로 가장 좋은 성능을 보인다. 특히 DBP 추정에서 개선 폭이 크다. 논문은 Figure 10에서 predicted SBP/DBP와 actual SBP/DBP의 scatter plot을 비교하며, WST domain에서 예측값이 ground truth 주변에 더 밀집한다고 설명한다.

이 결과는 WST가 PPG beat의 혈압 관련 특징을 더 잘 정리하여 LSTM의 회귀 학습을 돕는다는 논문의 핵심 주장을 뒷받침한다. 다만 SBP MAE가 10.8358로 여전히 상당히 큰 편이므로, clinical-grade 정확도를 달성했다고 보기는 어렵다.

### 4.4 rPPG를 사용한 평가 결과

논문은 궁극적으로 camera 기반 remote PPG(rPPG)를 이용한 혈압 추정 가능성에도 관심을 둔다. 이를 위해 public rPPG dataset을 사용하여 제안 시스템을 평가했다고 설명한다. rPPG 신호도 cPPG와 동일하게 beat segmentation, normalization, time normalization을 거친다.

Table 5는 rPPG beat에서 SBP와 DBP를 추정한 결과를 보여준다. Time domain에서는 DBP RMSE 11.1798, MAE 11.1295, SBP RMSE 17.8066, MAE 15.0720이다. DCT domain은 DBP RMSE 11.7560, MAE 11.5511, SBP RMSE 17.7062, MAE 16.7606이다. DWT domain은 DBP RMSE 11.2555, MAE 10.0244, SBP RMSE 16.9441, MAE 14.3486이다. WST domain은 DBP RMSE 11.2034, MAE 9.5390, SBP RMSE 15.4742, MAE 13.3852로 가장 좋은 결과를 보인다.

rPPG에서는 cPPG보다 전반적으로 오차가 크다. 이는 rPPG가 contact PPG보다 motion artifact, illumination variation, skin tone, camera noise 등에 더 취약하기 때문이다. 그럼에도 WST는 rPPG에서도 상대적으로 가장 낮은 error를 보이며, 특히 SBP와 DBP의 maximum/minimum tracking이 더 안정적이라고 설명된다.

다만 rPPG dataset의 구체적 출처, subject 수, train/test 방식, cPPG에서 학습한 모델을 rPPG에 그대로 적용한 것인지 또는 rPPG로 별도 학습한 것인지는 제공된 텍스트만으로는 충분히 명확하지 않다.

### 4.5 결과 해석

실험 전체에서 WST는 time domain, DCT, DWT보다 일관되게 좋은 결과를 낸다. 논문은 이를 PPG beat의 shift, scaling, local deformation에 대한 WST의 안정성 때문이라고 해석한다. Time domain feature는 beat interval과 PPG waveform 자체를 포함하므로 혈압 정보를 직접 담고 있지만, deep learning 모델이 raw waveform에서 안정적인 특징을 학습하려면 많은 데이터와 더 복잡한 모델이 필요하다. DCT는 신호를 주파수 성분으로 압축할 수 있지만, low-frequency component나 DC component 예측 오류가 ABP/BP 추정에 큰 영향을 줄 수 있다. DWT는 time-frequency feature를 제공하지만, PPG beat의 shift와 scaling에 민감할 수 있다.

반면 WST는 wavelet 기반 multi-scale representation을 만들면서도 translation invariance와 deformation stability를 제공한다. 이 때문에 PPG beat의 작은 시간 이동이나 형태 변형에 덜 민감하고, LSTM이 혈압 관련 정보를 더 쉽게 학습하도록 돕는다.

그러나 WST를 사용해도 SBP 추정 오차는 cPPG에서 MAE 10.8358, rPPG에서 MAE 13.3852로 보고된다. 이는 cuffless BP estimation에서 충분히 개선된 결과이지만, 임상적 대체 장치로 사용하기에는 추가 검증과 정확도 개선이 필요하다. 특히 SBP는 DBP보다 오차가 더 크게 나타나며, 이는 PPG morphology와 SBP 간 관계가 더 복잡하거나 ABP waveform의 peak를 정확히 예측하기 어렵기 때문일 수 있다.

## 5. 강점, 한계

### 5.1 강점

이 논문의 가장 큰 강점은 PPG 기반 혈압 추정에서 feature representation 문제를 명확히 다룬다는 점이다. 많은 deep learning 기반 연구는 raw PPG waveform을 직접 CNN, LSTM, U-Net 등에 입력하지만, 이 논문은 PPG의 shift, scaling, local deformation, noise 문제를 인식하고 WST를 통해 더 안정적인 representation을 만든다.

두 번째 강점은 beat-by-beat 접근이다. 신호 전체를 긴 segment로 처리하는 대신, 개별 PPG beat를 단위로 ABP waveform 또는 SBP/DBP를 예측한다. 이는 continuous BP monitoring이라는 목적과 잘 맞는다. beat 단위 추정이 가능하면 혈압 변화를 더 높은 시간 해상도로 추적할 수 있다.

세 번째 강점은 두 가지 추정 시나리오를 모두 검증했다는 점이다. PPG-to-ABP waveform mapping은 더 풍부한 정보를 제공하는 어려운 문제이고, PPG-to-SBP/DBP direct mapping은 실제 사용에서 더 간단한 문제이다. 두 시나리오에서 모두 WST의 우수성을 보였다는 점은 제안 방법의 일관성을 보여준다.

네 번째 강점은 feature domain 비교가 체계적이라는 점이다. Time domain, DCT, DWT, WST를 동일한 LSTM 구조에서 비교함으로써, 성능 향상이 LSTM 구조 자체보다는 feature representation 차이에서 비롯되었음을 비교적 명확히 보여준다.

다섯 번째 강점은 cPPG뿐 아니라 rPPG까지 평가했다는 점이다. rPPG 기반 혈압 추정은 contactless monitoring 가능성을 열어주므로, telemedicine, camera-based health monitoring, neonatal monitoring 등 다양한 응용 가능성이 있다. rPPG 결과는 cPPG보다 낮지만, WST가 여전히 가장 좋은 성능을 보인다는 점은 흥미롭다.

### 5.2 한계

첫 번째 한계는 dataset split 방식의 불명확성이다. 논문은 beat basis로 training, validation, test를 나누었다고 설명하지만, subject-level 또는 patient-level split인지 명확하지 않다. MIMIC II는 한 환자에서 많은 beat가 생성될 수 있으므로, 같은 환자의 beat가 training과 test에 동시에 포함되면 모델이 환자별 특성을 암묵적으로 학습하여 성능이 과대평가될 수 있다. 실제 적용에서는 unseen patient에 대한 generalization이 중요하므로, patient-independent split 검증이 필요하다.

두 번째 한계는 임상 기준과의 비교가 부족하다는 점이다. 논문은 RMSE와 MAE를 제시하지만, AAMI, BHS, ISO 81060-2 같은 혈압 측정 장치 검증 기준과 직접 비교하지 않는다. 특히 cPPG 기반 WST의 SBP MAE가 10.8358이고 rPPG 기반 SBP MAE가 13.3852이므로, 임상 장치 수준의 정확도라고 보기는 어렵다.

세 번째 한계는 WST parameter와 LSTM architecture에 대한 세부 설명이 제한적이라는 점이다. WST의 wavelet type, scattering order, scale parameter, averaging scale, coefficient selection 방식 등이 충분히 구체적으로 제시되지 않으면 재현성이 떨어질 수 있다. 제공된 텍스트에서는 WST의 일반식은 설명되지만 실제 구현 세부 설정은 제한적으로 보인다.

네 번째 한계는 baseline deep learning 모델이 단순하다는 점이다. 비교는 주로 동일 LSTM에서 feature domain만 바꾸는 방식으로 이루어진다. 이는 feature domain 효과를 보기에는 좋지만, 최신 CNN-LSTM, Transformer, U-Net, temporal convolutional network, attention model 등과의 직접 비교는 부족하다. 따라서 “WST-LSTM이 최신 deep learning 모델보다 우수하다”는 결론까지는 내릴 수 없다.

다섯 번째 한계는 rPPG 평가의 세부 정보가 부족하다는 점이다. 논문은 public rPPG dataset으로 평가했다고 설명하지만, 데이터셋 구성, subject 수, reference BP 획득 방식, 학습과 테스트 구분, cPPG와 rPPG 간 domain shift 처리 방식이 명확히 드러나지 않는다. rPPG는 조명, motion, camera quality에 매우 민감하기 때문에, 이 정보가 결과 해석에 중요하다.

여섯 번째 한계는 ABP waveform estimation의 생리학적 타당성 평가가 제한적이라는 점이다. waveform-level reconstruction에서는 단순 RMSE/MAE뿐 아니라 systolic peak timing, dicrotic notch preservation, pulse pressure, waveform morphology similarity, spectral similarity 등을 평가할 수 있다. 논문은 시각적 예시와 RMSE/MAE를 제공하지만, waveform morphology 보존에 대한 더 정밀한 분석은 부족하다.

일곱 번째 한계는 beat selection으로 인해 모델이 clean beat에 치우칠 가능성이 있다는 점이다. 논문은 SQI, CQI, BI 기준으로 invalid beat를 제거한다. 이는 학습 안정성을 높이지만, 실제 wearable 환경에서는 noisy beat가 빈번히 발생한다. 따라서 모델이 실제 환경의 low-quality beat에 얼마나 robust한지는 추가 검증이 필요하다.

### 5.3 비판적 해석

이 논문은 “더 큰 deep learning 모델”이 아니라 “더 좋은 signal representation”이 PPG 기반 혈압 추정에 중요할 수 있음을 보여주는 연구이다. WST는 학습 가능한 filter가 아니라 고정된 wavelet 기반 scattering coefficient를 사용하므로, 데이터가 제한적이고 noise가 많은 상황에서 안정적 feature extraction을 제공한다. 이러한 관점은 의료 신호 처리에서 여전히 중요하다. 생체 신호는 데이터 수가 제한적이고 labeling이 어렵기 때문에, 무조건 end-to-end 학습에 의존하기보다 domain-aware representation을 사용하는 전략이 실용적일 수 있다.

그러나 논문의 결과는 WST가 다른 feature domain보다 우수하다는 것을 보여줄 뿐, 실제 임상 적용 가능성을 충분히 입증하지는 않는다. 특히 SBP 오차가 여전히 크고, patient-independent evaluation이 명확하지 않으며, 최신 deep learning baseline과의 비교도 제한적이다. 따라서 이 연구는 “임상적으로 완성된 cuffless BP monitor”라기보다, PPG beat representation으로서 WST의 유용성을 검증한 methodological study로 보는 것이 적절하다.

또한 WST가 shift와 deformation에 강하다는 장점은 분명하지만, 혈압 추정에서 중요한 absolute amplitude 정보가 PPG normalization 과정에서 일부 제거될 수 있다는 점도 고려해야 한다. PPG amplitude는 sensor contact와 개인 차이에 민감하지만, 동시에 혈관 상태와 관련된 정보를 포함할 수 있다. 이 논문은 PPG beat를 0–1로 normalization하므로 amplitude 관련 정보는 제한될 수 있다. 향후 연구에서는 amplitude normalization 전후 feature를 함께 사용하는 방법도 고려할 수 있다.

## 6. 결론

이 논문은 PPG 신호만을 사용하여 beat-by-beat 혈압을 추정하기 위한 hybrid Scattering–LSTM framework를 제안한다. 핵심은 PPG beat를 원시 time domain에서 직접 학습하는 대신, Wavelet Scattering Transform을 통해 shift-invariant하고 local deformation에 안정적인 feature representation으로 변환한 후 LSTM 회귀 모델로 ABP waveform 또는 SBP/DBP를 예측하는 것이다.

제안 방법은 두 가지 시나리오에서 평가되었다. 첫 번째는 PPG beat에서 ABP beat 전체를 복원하는 PPG-to-ABP mapping이고, 두 번째는 PPG beat에서 SBP와 DBP를 직접 추정하는 PPG-to-SBP/DBP mapping이다. 실험 결과 WST는 time domain, DCT, DWT보다 일관되게 더 낮은 RMSE와 MAE를 보였다. ABP waveform estimation에서는 WST-DWT 조합이 RMSE 8.9935, MAE 7.6257로 가장 좋은 결과를 보였고, SBP/DBP direct estimation에서는 WST가 DBP RMSE 6.9164, MAE 5.0945, SBP RMSE 14.2079, MAE 10.8358을 기록했다. rPPG 평가에서도 WST가 다른 feature domain보다 우수한 성능을 보였다.

이 연구의 주요 기여는 PPG 기반 BP estimation에서 WST가 효과적인 feature extractor로 작동할 수 있음을 보인 점이다. 특히 제한된 dataset, beat deformation, shift, scaling 문제가 있는 상황에서 WST는 LSTM의 학습 부담을 줄이고 더 안정적인 혈압 추정을 가능하게 한다.

향후 연구에서는 patient-independent split을 통한 일반화 검증, 최신 deep learning baseline과의 비교, WST parameter의 체계적 분석, noisy real-world wearable signal에서의 robustness 평가, clinical validation standard와의 비교가 필요하다. 또한 LSTM 대신 Transformer, temporal convolutional network, attention-based recurrent model 등을 결합하거나, WST coefficient와 raw PPG feature를 함께 사용하는 hybrid representation도 유망한 후속 방향이 될 수 있다.
