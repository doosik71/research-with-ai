# A deep learning method for continuous noninvasive blood pressure monitoring using photoplethysmography

* **저자**: Hao Liang, Wei He, Zheng Xu
* **발표연도**: 2023

## 1. 논문 개요

이 논문은 단일 photoplethysmography, 즉 PPG 신호만을 입력으로 사용하여 연속적인 arterial blood pressure, ABP waveform을 추정하는 deep learning 기반 방법을 제안한다. 논문에서 제안하는 모델은 ABP-UNet으로, PPG 신호로부터 systolic blood pressure, SBP와 diastolic blood pressure, DBP의 단일 값만 직접 회귀하는 것이 아니라, 시간에 따라 변화하는 전체 혈압 파형을 생성한다. 이후 생성된 ABP waveform에서 최대값을 SBP, 최소값을 DBP로 추출한다.

연구의 핵심 목표는 cuff 기반 혈압 측정의 불편함과 간헐성, invasive arterial line의 위험성, 그리고 기존 PPG 기반 pulse wave analysis, PWA 방식의 수동 feature engineering 문제를 동시에 줄이는 것이다. 전통적인 cuff-based measurement는 비교적 널리 사용되지만 연속 측정이 어렵고, 장시간 반복 측정 시 혈관 압박으로 불편함을 유발한다. 반면 arterial line은 연속 ABP waveform을 정확히 제공하지만 감염과 출혈 위험이 있어 일상 사용이나 일반 병동, wearable 환경에는 적합하지 않다.

이 논문이 다루는 연구 문제는 다음과 같다. 단일 PPG 신호만으로 추가 ECG, IPG, arterial line 없이 연속 ABP waveform을 정확히 추정할 수 있는가이다. 또한 wave peak, valley, dicrotic notch 등 수동으로 정의한 PPG morphology feature를 사용하지 않고도 convolutional neural network가 혈압 추정에 필요한 signal feature를 자동으로 학습할 수 있는지가 핵심 질문이다.

문제의 중요성은 hypertension과 cardiovascular disease의 높은 질병 부담과 관련된다. 논문은 전 세계적으로 약 14억 명이 고혈압을 가지고 있으며, SBP를 5 mmHg 낮추면 주요 cardiovascular event 위험이 10% 감소한다는 근거를 제시한다. 즉, 혈압을 자주, 편리하게, 연속적으로 측정할 수 있다면 hypertension 관리, 심혈관 질환 조기 진단, 치료 효과 추적, 재활 관리에 실질적인 도움이 된다.

논문은 MIMIC II dataset을 중심으로 모델을 학습 및 평가하고, 추가로 cuff-based measurement를 포함하는 University of Queensland Vital Signs, UQVS dataset에서 별도 평가를 수행한다. MIMIC II 평가에서 제안 모델은 SBP에 대해 MAE 2.62 mmHg, SD 4.05 mmHg, DBP에 대해 MAE 1.71 mmHg, SD 2.84 mmHg를 달성했다. 전체 BP waveform 추정에서도 MAE 2.49 mmHg, SD 3.09 mmHg를 보고했다. 논문은 이 결과가 AAMI 기준과 BHS Grade A 기준을 만족한다고 주장한다.

논문의 성격은 PPG-to-ABP waveform translation에 가깝다. 즉, PPG window를 입력받아 동일 길이의 ABP waveform을 출력하는 end-to-end signal-to-signal model이다. 이 접근은 SBP/DBP scalar regression보다 더 풍부한 cardiovascular information을 제공할 수 있다는 장점이 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **PPG 신호와 ABP waveform 사이의 복잡한 비선형 관계를 수동 feature extraction 없이 1D UNet 구조로 직접 학습한다**는 것이다. 기존 PPG 기반 BP estimation은 크게 두 가지 흐름이 있었다. 하나는 PWV, PAT, PTT와 같이 ECG나 IPG 등 추가 생체신호와 PPG를 함께 사용하여 pulse wave의 도달 시간이나 속도를 계산하는 방식이다. 다른 하나는 PPG waveform만을 사용하되 peak, valley, slope, amplitude, time interval 같은 morphology feature를 사람이 설계하여 regression model에 넣는 PWA 방식이다.

이 논문은 두 방식의 한계를 동시에 지적한다. PWV 또는 PAT 기반 방식은 ECG 등 보조 sensor가 필요하므로 wearable device의 실용성과 편의성이 낮아진다. 또한 개인의 생리적 특성 차이로 인해 regression parameter shift가 발생할 수 있다. PWA 기반 방식은 단일 PPG만 사용할 수 있다는 장점이 있지만, waveform feature를 사람이 설계하고 추출해야 하므로 subjective factor가 개입되고, feature detection이 noise와 waveform distortion에 민감하다.

ABP-UNet의 차별점은 PPG waveform을 그대로 입력하여 ABP waveform을 직접 출력한다는 점이다. 모델은 convolutional layer를 통해 PPG의 local morphology와 frequency-related feature를 자동으로 학습한다. 논문은 내부 convolution layer의 intermediate output을 관찰했을 때, 입력에 가까운 layer들이 각 cardiac cycle의 waveform을 인식하는 경향을 보였다고 설명한다. 즉, 모델은 heart rate와 관련된 frequency feature를 명시적으로 입력하지 않아도 자동으로 활용한다.

또한 UNet 구조는 encoder-decoder와 skip connection을 사용하기 때문에, PPG 신호의 low-level local feature와 high-level global feature를 함께 보존할 수 있다. Encoder는 PPG 신호를 점차 down-sampling하면서 abstract feature를 추출하고, decoder는 up-sampling을 통해 ABP waveform을 복원한다. Skip connection은 encoder의 중간 feature를 decoder로 전달하여 waveform의 세부 형태가 손실되지 않도록 돕는다.

이 논문의 또 다른 핵심은 scalar BP value가 아니라 **continuous BP waveform**을 학습 목표로 삼는다는 점이다. SBP와 DBP만 직접 맞추는 모델은 혈압의 최고점과 최저점만 맞추도록 학습될 수 있지만, 이 논문은 전체 ABP waveform과 reference waveform의 차이를 MSE loss로 최소화한다. 따라서 모델은 혈압 파형 전체의 동적 변화와 형태를 맞추도록 학습된다. 이 점은 continuous monitoring 목적에 더 적합하다.

## 3. 상세 방법 설명

### 전체 파이프라인

논문에서 제안하는 전체 파이프라인은 데이터 수집, preprocessing, signal quality filtering, ABP-UNet 학습, ABP waveform 추정, SBP/DBP/MAP 산출, 성능 평가로 구성된다.

먼저 MIMIC II dataset에서 fingertip PPG와 invasive ABP 신호를 동기적으로 가져온다. 신호는 125 Hz sampling rate로 기록되어 있으며, 각 record의 길이는 수 초에서 500초까지 다양하다. 이후 모든 신호를 길이 1024 sample의 sequence로 자른다. 125 Hz 기준으로 1024 sample은 8.192초에 해당한다. 이 길이는 모델 입력과 출력의 표준 window가 된다.

PPG와 ABP 신호는 개인별·센서별 amplitude range가 다르므로 min-max normalization을 적용한다. 이 과정은 신호 값을 0에서 1 사이의 동일한 range로 mapping하여, 서로 다른 사람과 센서에서 측정된 pulse signal을 비교 가능하게 만든다.

다음으로 noise removal을 수행한다. 논문은 PPG acquisition 과정에서 motion artifact, respiration에 의한 baseline drift, high-frequency electromagnetic interference, sensor displacement 또는 sensor falling off로 인한 심한 distortion이 발생할 수 있다고 설명한다. 이를 제거하기 위해 discrete wavelet transform, DWT를 사용한다. Mother wavelet은 Daubechies 4, 즉 db4이며, 6-level decomposition을 수행한다. 0–0.4 Hz 범위의 저주파 성분은 baseline wandering과 motion artifact로 보고 제거하고, 15 Hz 이상의 고주파 성분은 electromagnetic interference로 보고 제거한다. 결과적으로 0.5–15 Hz 범위의 PPG 정보가 보존된다.

그러나 wavelet filtering만으로 sensor detachment 등으로 인한 심각한 waveform distortion은 제거하기 어렵다. 이를 해결하기 위해 signal quality detection을 추가로 수행한다. 논문은 approximate entropy와 fuzzy entropy를 사용하여 signal quality를 판단한다. 총 2696개의 수동 라벨링된 waveform quality data를 이용해 support vector machine classifier를 학습하고, 정상 신호와 왜곡 신호를 이진 분류하였다. Approximate entropy와 fuzzy entropy를 joint input으로 사용할 때 classifier가 98%의 가장 높은 classification accuracy를 달성했다고 보고한다. 이 classifier를 사용하여 심하게 오염된 signal segment를 제거한다.

이 preprocessing 후 MIMIC II에서 총 117,042개 record, 총 266.3시간 분량의 데이터가 남았다. 이 중 80%인 93,634개 record, 9349명 환자 데이터를 training에 사용하고, 나머지 20%인 23,408개 record, 2651명 환자 데이터를 testing에 사용하였다.

### ABP-UNet 아키텍처

ABP-UNet은 biomedical image segmentation에서 사용되는 UNet 구조를 1D time-series signal에 맞게 변형한 모델이다. 원래 UNet은 image segmentation을 위해 2D convolution을 사용하지만, 이 논문에서는 PPG가 1D time-series vector이므로 모든 convolution을 1D convolution으로 대체한다.

모델의 입력은 길이 1024의 PPG sequence이고, 출력은 동일 길이 1024의 ABP waveform이다. 따라서 모델은 다음과 같은 mapping을 학습한다.

$$
f_{\theta}: PPG_{1:1024} \rightarrow ABP_{1:1024}
$$

여기서 $f_{\theta}$는 학습 가능한 neural network이고, $\theta$는 모델 parameter이다.

모델은 encoder와 decoder로 구성된다. Encoder는 네 개의 down-sampling block, DS1–DS4를 포함한다. 각 down-sampling block은 pooling 또는 stride 기반 down-sampling 역할과 convolution submodule을 포함한다. 입력 길이 1024의 pulse wave는 encoder를 지나며 512, 256, 128, 64 길이의 multiscale feature vector로 변환된다. 즉, 모델은 점차 temporal resolution을 줄이면서 더 넓은 time context와 global pattern을 학습한다.

Decoder는 네 개의 up-sampling block, US1–US4를 포함한다. 각 up-sampling block은 bilinear interpolation 기반 up-sampling을 통해 feature length를 다시 늘리고, encoder 단계의 corresponding intermediate feature와 concat 방식으로 fusion한다. 이 skip connection은 UNet의 핵심 구조로, down-sampling 과정에서 사라질 수 있는 local waveform detail을 복원하는 데 도움을 준다.

논문은 원래 UNet 대비 몇 가지 구조적 개선을 적용한다. 첫째, PPG waveform의 crest와 pulse wave morphology를 더 정확히 인식하기 위해 convolution kernel size를 3에서 7로 늘렸다. PPG와 ABP는 image보다 훨씬 좁은 local pattern이 아니라 일정한 시간 폭을 가진 waveform morphology가 중요하므로, 더 큰 kernel은 waveform의 temporal structure를 더 잘 포착할 수 있다.

둘째, maximum pooling layer를 작은 convolution kernel, size 3, stride 2의 down-sampling 구조로 대체하였다. 이는 단순히 최대값만 취하는 pooling보다 learnable down-sampling 효과를 줄 수 있다.

셋째, 전통적인 convolution 대신 depthwise separable convolution을 사용하여 계산 복잡도와 parameter 수를 줄였다. Kernel size를 7로 늘리면 연산량이 증가하므로, depthwise separable convolution은 성능과 효율 사이의 균형을 맞추기 위한 선택이다.

넷째, convolution submodule은 reverse bottleneck 구조를 가진다. Down-sampling 후 layer normalization을 적용하고, convolution layer를 통해 channel을 4배로 확장한 뒤, GELU activation을 사용하고, 다시 filter 수가 적은 convolution으로 처리한다. 마지막으로 residual connection을 통해 module input과 output을 결합한다. 이 구조는 feature representation capacity를 높이면서 학습 안정성을 확보하려는 설계로 볼 수 있다.

다섯째, 논문은 일반적인 CNN처럼 모든 convolution 뒤에 normalization과 activation을 붙이지 않는다. 실험적으로 너무 많은 non-convolution operation은 model redundancy만 늘리고 prediction accuracy에는 뚜렷한 장점을 주지 않았다고 설명한다. 따라서 각 convolution submodule마다 하나의 normalization과 activation function만 유지했다.

### 학습 목표와 손실 함수

모델은 예측한 ABP waveform과 ground truth ABP waveform 사이의 mean squared error, MSE를 최소화하도록 학습된다. 논문에서 명시한 loss function은 전체 waveform 단위의 MSE이다.

$$
MSE = \frac{1}{N}\sum_{i=1}^{N}(y_p(i)-y_g(i))^2
$$

여기서 $y_p(i)$는 모델이 예측한 BP waveform의 $i$번째 sample 값이고, $y_g(i)$는 reference ABP waveform의 $i$번째 sample 값이다. $N$은 waveform sample 수이다.

이 손실 함수의 의미는 SBP나 DBP만 맞추는 것이 아니라, 전체 ABP waveform의 모든 time point를 맞추도록 모델을 학습한다는 것이다. 따라서 모델은 waveform shape, peak timing, valley timing, rising slope, falling slope 등 전체 동역학을 학습할 수 있다.

학습에는 Adam optimizer와 learning rate attenuation이 사용된다. 또한 training 초기에 learning rate를 점진적으로 증가시키는 warm-up strategy를 적용한다. 이는 학습 초기에 gradient가 불안정하게 커지는 것을 방지하고, 이후 dynamic learning rate adjustment와 결합하여 안정적인 최적화를 가능하게 한다.

### 평가 지표

모델의 예측 결과는 ME, MAE, SD, RMSE로 평가된다. 논문은 ground truth BP를 $y_g$, predicted BP를 $y_p$, test sample 수를 $N$으로 두고 다음 지표를 사용한다.

Mean Error, ME는 prediction error의 평균 방향성을 나타낸다.

$$
ME = \frac{1}{N}\sum_{i=1}^{N}(y_p(i)-y_g(i))
$$

ME가 0에 가까우면 전체적으로 overestimation 또는 underestimation bias가 작다는 의미이다.

Mean Absolute Error, MAE는 절대 오차의 평균이다.

$$
MAE = \frac{1}{N}\sum_{i=1}^{N}|y_p(i)-y_g(i)|
$$

MAE는 실제 혈압 추정에서 가장 직관적인 오차 지표로, 평균적으로 몇 mmHg 정도 틀리는지를 나타낸다.

Standard Deviation, SD는 오차의 변동성을 나타낸다.

$$
SD = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_p(i)-y_g(i)-ME)^2}
$$

SD가 작을수록 prediction error가 안정적이라는 의미이다. AAMI 기준에서는 ME가 5 mmHg 이하이고 SD가 8 mmHg 이하이어야 한다.

Root Mean Square Error, RMSE는 제곱 오차의 평균에 제곱근을 취한 값이다.

$$
RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_p(i)-y_g(i))^2}
$$

RMSE는 큰 error에 더 민감하므로 extreme prediction error가 있는 경우 MAE보다 크게 증가한다.

모델이 출력한 ABP waveform에서 SBP는 waveform의 maximum value, DBP는 minimum value로 정의된다. MAP도 계산할 수 있다고 언급하지만, MAP 계산식 자체는 본문 추출 텍스트에서 구체적으로 제시되지 않았다.

## 4. 실험 및 결과

### MIMIC II dataset 평가

주요 실험은 MIMIC II test dataset의 23,408개 record에서 수행되었다. 전체 성능은 SBP, MAP, DBP, waveform에 대해 각각 보고되었다.

SBP 추정 결과는 ME -0.37 mmHg, MAE 2.62 mmHg, SD 4.05 mmHg, RMSE 4.06 mmHg였다. DBP 추정 결과는 ME 0.35 mmHg, MAE 1.71 mmHg, SD 2.84 mmHg, RMSE 2.86 mmHg였다. MAP은 ME 0.28 mmHg, MAE 1.04 mmHg, SD 2.19 mmHg, RMSE 2.21 mmHg였다. 전체 waveform 추정은 ME -0.28 mmHg, MAE 2.49 mmHg, SD 3.09 mmHg, RMSE 3.38 mmHg로 보고되었다.

논문 초록에는 SBP 2.55 ± 3.92 mmHg, DBP 1.66 ± 2.76 mmHg, waveform 2.52 ± 3.02 mmHg로 제시되어 있는데, 본문 Table 2에서는 SBP MAE 2.62, SD 4.05, DBP MAE 1.71, SD 2.84, waveform MAE 2.49, SD 3.09로 제시된다. 제공된 텍스트만으로는 초록과 본문 수치가 약간 다른 이유는 명확하지 않다. 다만 두 결과 모두 매우 유사한 수준이며, 논문은 전체적으로 AAMI와 BHS 기준을 만족한다고 주장한다.

SBP error가 DBP error보다 큰 이유에 대해 논문은 physiologic waveform 특성을 든다. SBP는 pulse wave의 peak이며, systolic waveform의 peak는 변화 폭이 크고 slope가 steep하다. 따라서 neural network가 peak value를 정확히 찾는 것이 DBP의 trough를 찾는 것보다 어렵고, 이로 인해 SBP 예측 오차가 더 크게 나타난다.

### BHS protocol 평가

BHS standard는 절대 오차가 5 mmHg, 10 mmHg, 15 mmHg 이하인 비율을 기준으로 모델을 평가한다. Grade A 기준은 각각 60%, 85%, 95% 이상이다.

제안 모델은 SBP에서 absolute error가 5 mmHg 이하인 비율 86.7%, 10 mmHg 이하 96.1%, 15 mmHg 이하 98.8%를 달성했다. DBP에서는 각각 93.6%, 98.2%, 99.5%였다. MAP에서는 각각 96.1%, 99.1%, 99.7%였다. 모두 BHS Grade A 기준을 크게 초과한다. 이는 단순히 평균 오차만 낮은 것이 아니라, 대부분의 sample에서 오차가 임상적으로 허용 가능한 범위 안에 들어간다는 의미이다.

### Regression 및 Bland-Altman 분석

논문은 reference SBP/DBP와 estimated SBP/DBP 사이의 regression plot을 제시한다. Pearson correlation coefficient는 SBP에서 0.982, DBP에서 0.967로 보고되었다. 이는 모델 prediction과 reference value 사이에 매우 높은 linear correlation이 있음을 의미한다.

다만 논문은 regression slope가 정확히 1이 아닌 이유도 설명한다. 하나는 모델 입력이 fingertip PPG인 반면 training reference BP는 ascending aortic cannulation으로 얻은 BP라는 점이다. Peripheral PPG와 central arterial BP 사이에는 nonlinear relationship이 존재한다. 다른 하나는 preprocessing이 완벽하지 않아 일부 noise와 abnormal point가 sample에 남아 regression curve에 영향을 주었을 가능성이다.

Bland-Altman 분석에서는 SBP와 DBP에 대해 predicted BP와 standard BP 사이의 consistency를 평가하였다. SBP의 95% consistency interval은 [-8.30, 7.56] mmHg였고, DBP는 [-5.21, 5.90] mmHg였다. 이 interval 안에 포함된 비율은 SBP 94.8%, DBP 94.0%로, 이상적인 95%에 매우 근접한다. 논문은 최대 deviation이 약 30 mmHg까지 관찰되었다고 설명하며, 이는 invasive BP probe의 interference로 reference BP waveform 자체가 부정확했기 때문일 수 있다고 해석한다. 이 부분은 model error뿐 아니라 label noise 가능성을 언급했다는 점에서 중요하다.

### Continuous BP monitoring 사례 분석

논문은 test dataset에서 일부 individual case를 선택하여 장시간 monitoring 성능을 분석했다. Table 4에는 7–9분 범위의 여러 subject에 대해 SBP와 DBP MAE ± SD가 제시되어 있다. 예를 들어 subject 164는 SBP 1.72 ± 1.86 mmHg, DBP 1.20 ± 1.54 mmHg였고, subject 3918은 SBP 1.63 ± 1.78 mmHg, DBP 0.77 ± 0.93 mmHg였다. 일부 subject에서는 SBP 3.71 ± 4.80 mmHg, DBP 3.49 ± 3.14 mmHg처럼 더 큰 오차가 나타났지만, 전반적으로 individual continuous monitoring에서도 비교적 낮은 error를 보였다.

논문은 이러한 결과가 모델이 전체 test set뿐 아니라 individual case에서도 혈압 변화를 추적할 수 있음을 보여준다고 해석한다. 이는 continuous BP monitoring application에서 중요한 결과이다. 다만 제공된 텍스트 기준으로는 long-term이 수일 또는 수주가 아니라 7–9분 수준의 case monitoring에 해당한다. 따라서 진정한 장기간 monitoring 성능은 별도 검증이 필요하다.

### UQVS cuff-based dataset 평가

MIMIC II는 ICU 환자와 invasive ABP reference를 기반으로 한다. 실제 noninvasive BP monitoring에서는 cuff-based BP가 더 일반적인 reference로 사용된다. 이를 고려해 논문은 UQVS dataset에서 추가 평가를 수행하였다. UQVS dataset은 anesthesia patient의 vital signs를 포함하며, PPG는 100 Hz로 sampling되었고, 논문에서는 MIMIC II와 일관성을 위해 125 Hz로 resampling했다. 각 PPG segment에 대응하는 SBP와 DBP는 cuff-based measurement의 mean value로 사용되었다.

UQVS 평가에서 SBP는 ME -0.48 mmHg, MAE 4.13 mmHg, SD 6.51 mmHg, RMSE 6.62 mmHg였고, DBP는 ME -0.18 mmHg, MAE 3.32 mmHg, SD 4.86 mmHg, RMSE 4.86 mmHg였다. SBP의 absolute error가 5 mmHg 이하인 비율은 72.8%, 10 mmHg 이하 91.0%, 15 mmHg 이하 96.0%였고, DBP는 각각 81.2%, 94.8%, 98.3%였다. 논문은 DBP가 BHS Grade A를 달성하고, SBP는 Grade A에 거의 도달했으며, SBP와 DBP 모두 AAMI protocol을 만족한다고 설명한다.

이 결과는 모델이 invasive ABP reference로 학습된 MIMIC II뿐 아니라 cuff-based BP reference를 가진 다른 dataset에서도 어느 정도 일반화될 수 있음을 보여준다. 그러나 UQVS dataset은 anesthesia patient 데이터이므로 일반 일상생활 population과는 여전히 차이가 있다.

### 기존 연구와의 비교

논문은 여러 기존 방법과 비교한다. Kachuee et al.은 PPG와 ECG를 사용해 PAT parameter를 결정하는 방식이었고, SBP 성능이 DBP보다 상대적으로 낮아 BHS Grade B 수준이었다. Haddad et al.은 MIMIC I에서 PPG feature를 multilinear regression으로 처리했으며, SBP MAE 6.10 mmHg, SD 8.08 mmHg, DBP MAE 4.65 mmHg, SD 6.22 mmHg였다. Wang et al.은 PPG temporal signal을 visibility graph image로 변환하고 transfer learning으로 feature를 추출했으며, SBP SD 8.46 mmHg, DBP SD 5.36 mmHg를 보고했다.

Huang et al.은 PPG와 ECG를 모두 사용하는 MLP-Mixer 기반 framework를 제안했고, MIMIC II에서 SBP MAE 3.52 mmHg, SD 5.09 mmHg, DBP MAE 2.13 mmHg, SD 3.07 mmHg를 달성했다. 이 결과는 제안 모델과 가까운 수준이지만, 논문은 Huang et al.의 방법이 ECG channel을 추가로 필요로 하고, 여러 filter를 사용해 multichannel feature를 생성해야 하므로 convenience와 efficiency 측면에서 제안 모델보다 불리하다고 주장한다.

UNet 기반 기존 연구와의 비교도 중요하다. Ibtehaz et al.과 Athaya and Choi도 PPG-to-ABP waveform estimation에 UNet 구조를 사용했다. Athaya and Choi는 PPG와 BP signal의 phase correction 및 overlapping slicing을 사용하여 높은 정확도를 보였으나, neural network 구조 자체의 조정과 최적화는 본 논문만큼 다루지 않았다고 설명한다. 본 논문의 ABP-UNet은 kernel size 확장, depthwise separable convolution, reverse bottleneck 구조, normalization 및 activation 배치 조정 등을 통해 성능을 개선했다는 점을 강조한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 단일 PPG 신호만으로 continuous ABP waveform을 end-to-end로 추정한다는 점이다. ECG, IPG, SCG 등 추가 sensor를 요구하지 않기 때문에 wearable device와 smartphone 기반 측정 환경에 더 적합하다. 특히 cuff-less, noninvasive, continuous monitoring이라는 세 가지 요구를 동시에 만족하려는 방향성이 명확하다.

두 번째 강점은 수동 feature extraction을 제거했다는 점이다. 기존 PWA 방법은 PPG peak, valley, dicrotic notch, slope 등을 정확히 검출해야 하는데, 이는 noise와 motion artifact에 취약하다. ABP-UNet은 convolutional structure를 사용하여 feature를 자동으로 학습하므로, manual waveform analysis의 subjective factor를 줄인다. 이는 대규모 데이터 기반 모델링에서 중요한 장점이다.

세 번째 강점은 scalar regression이 아니라 waveform reconstruction을 수행한다는 점이다. 전체 ABP waveform을 출력하면 SBP와 DBP뿐 아니라 MAP, waveform shape, dynamic response 등 더 풍부한 cardiovascular information을 얻을 수 있다. 병동이나 재활 환경에서는 단일 BP value보다 시간에 따른 waveform trend가 임상적으로 더 유용할 수 있다.

네 번째 강점은 preprocessing이 비교적 체계적이라는 점이다. DWT 기반 filtering으로 baseline drift와 high-frequency noise를 제거하고, approximate entropy와 fuzzy entropy를 기반으로 심각한 distortion을 탐지하여 제거했다. 특히 signal quality classifier가 98% accuracy를 달성했다고 보고한 점은 데이터 품질 관리 측면에서 의미가 있다.

다섯 번째 강점은 MIMIC II뿐 아니라 UQVS dataset에서도 추가 평가를 수행했다는 점이다. MIMIC II는 invasive ABP reference를 포함하는 강력한 dataset이지만 ICU population에 제한된다. UQVS의 cuff-based BP 평가를 통해 모델이 다른 reference 환경에서도 어느 정도 성능을 유지할 수 있음을 보여준다.

여섯 번째 강점은 AAMI, BHS, regression analysis, Bland-Altman analysis 등 여러 관점에서 성능을 평가했다는 점이다. 단순히 MAE만 보고하지 않고, error distribution, 임상 기준 충족률, predicted-reference consistency를 함께 분석했다.

그러나 한계도 명확하다. 첫째, 주요 학습 및 평가 데이터가 MIMIC II의 ICU 환자 데이터이다. ICU 환자는 약물, 질병 상태, monitoring device, invasive procedure의 영향을 받으며, 일반 healthy population이나 home monitoring population과 생리적 조건이 다를 수 있다. 논문도 이 점을 인정하며, 기존 데이터가 주로 ICU 환자에서 왔기 때문에 모델이 high BP output에 치우칠 수 있다고 설명한다.

둘째, data split 방식에 대한 subject-independent 여부가 제공된 텍스트에서 충분히 명확하지 않다. 논문은 93,634 record from 9349 patients를 training, 23,408 record from 2651 patients를 testing에 사용했다고 설명하므로 환자 단위 분리가 이루어진 것처럼 보인다. 다만 구체적인 split protocol, 동일 patient record leakage 방지 방식, stratification 기준은 제공된 텍스트만으로는 완전히 확인할 수 없다.

셋째, MIMIC II에서 매우 높은 정확도가 보고되었지만, label quality 문제가 존재할 수 있다. 논문은 Bland-Altman 분석에서 최대 deviation 약 30 mmHg가 invasive BP probe interference로 인한 inaccurate BP waveform label 때문일 수 있다고 설명한다. 이는 반대로 전체 dataset에도 일부 label noise가 남아 있을 수 있음을 의미한다.

넷째, UQVS dataset 결과는 MIMIC II보다 오차가 증가했다. SBP MAE는 2.62 mmHg에서 4.13 mmHg로, DBP MAE는 1.71 mmHg에서 3.32 mmHg로 증가했다. 여전히 AAMI 기준은 만족하지만, domain shift가 존재함을 보여준다. 특히 UQVS는 anesthesia patient dataset이므로, 실제 wearable daily-life condition에서의 generalization은 아직 검증되지 않았다.

다섯째, 실제 hardware deployment나 low-power wearable implementation은 수행되지 않았다. 논문은 향후 low-power mobile device deployment와 cloud server communication 개선이 필요하다고 언급한다. 현재 ABP-UNet은 kernel size 7과 encoder-decoder 구조를 가진 relatively complex CNN이므로, smartwatch나 edge device에서 실시간 추론이 가능한지는 별도 검증이 필요하다.

여섯째, motion artifact가 실제 일상생활 수준으로 포함된 데이터에서 검증되지 않았다. Preprocessing은 motion artifact와 distortion 제거를 다루지만, MIMIC II와 UQVS는 병원 또는 마취 환경의 데이터이므로 walking, exercise, wrist motion, loose contact 같은 wearable condition을 충분히 반영하지 않을 수 있다.

일곱째, continuous monitoring 사례가 7–9분 수준으로 제시되어 있다. 논문 제목과 목표는 continuous noninvasive monitoring이지만, long-term stability, calibration drift, sensor position change, day-to-day variation은 평가되지 않았다. 따라서 장기 wearable monitoring 가능성은 아직 가능성 수준으로 해석해야 한다.

비판적으로 보면, 이 논문은 PPG-to-ABP waveform estimation에서 매우 우수한 benchmark 성능을 보고하지만, 실제 임상 및 일상 적용을 위해서는 더 넓은 population, non-ICU data, wearable sensor data, 장기 monitoring, subject-independent validation, computational efficiency 평가가 필요하다.

## 6. 결론

이 논문은 단일 PPG 신호에서 continuous ABP waveform을 추정하는 ABP-UNet을 제안하였다. 이 모델은 1D UNet 기반 encoder-decoder 구조를 사용하며, depthwise separable convolution, kernel size 확장, reverse bottleneck, skip connection 등을 통해 PPG waveform의 local morphology와 global temporal pattern을 학습한다. 모델은 수동 PWA feature extraction 없이 end-to-end 방식으로 ABP waveform을 출력하고, 이 waveform에서 SBP와 DBP를 계산한다.

MIMIC II dataset에서 제안 모델은 SBP MAE 2.62 mmHg, SD 4.05 mmHg, DBP MAE 1.71 mmHg, SD 2.84 mmHg를 달성했으며, 전체 waveform 추정에서도 MAE 2.49 mmHg, SD 3.09 mmHg를 기록했다. BHS protocol에서는 SBP, MAP, DBP 모두 Grade A 기준을 크게 초과했고, AAMI 기준도 만족했다. UQVS cuff-based dataset에서도 SBP MAE 4.13 mmHg, DBP MAE 3.32 mmHg로 비교적 우수한 성능을 유지했다.

이 연구의 주요 기여는 PPG-only continuous BP waveform estimation의 가능성을 강하게 보여주었다는 점이다. 특히 추가 ECG sensor 없이 PPG만으로 높은 정확도를 달성했다는 점은 wearable BP monitoring의 실용성을 높이는 방향이다. 또한 scalar BP estimation이 아니라 waveform reconstruction을 수행하므로, 향후 병동, home care, rehabilitation, hypertension management에서 더 풍부한 cardiovascular information을 제공할 수 있다.

다만 실제 적용까지는 중요한 과제가 남아 있다. 모델은 주로 ICU 기반 MIMIC II 데이터로 학습되었기 때문에 일반 population, 다양한 연령대, 다양한 지역, 다양한 질병 상태, 실제 wearable 환경에서의 검증이 필요하다. 또한 long-term monitoring, motion artifact robustness, hardware implementation, low-power inference, cloud-device communication, cuff-based real-world calibration과의 결합도 추가 연구가 필요하다.

종합하면, ABP-UNet은 PPG 기반 cuff-less BP estimation 분야에서 높은 정확도와 waveform-level monitoring 가능성을 보여주는 의미 있는 deep learning 접근이다. 그러나 현재 결과는 병원 기반 dataset에서의 성능을 중심으로 하므로, 실제 daily-life wearable medical device로 발전하기 위해서는 더 엄격하고 다양한 외부 검증이 필수적이다.
