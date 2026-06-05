# Systolic Blood Pressure Estimation from PPG Signal Using ANN

* **저자**: Benedetta C. Casadei, Alessandro Gumiero, Giorgio Tantillo, Luigi Della Torre, Gabriella Olmo
* **발표연도**: 2022

## 1. 논문 개요

이 논문은 photoplethysmography, 즉 PPG 신호만을 이용하여 systolic blood pressure, 즉 SP 또는 SBP를 추정하는 artificial neural network, 즉 ANN 기반 회귀 모델을 제안한다. 연구의 궁극적인 목표는 cuff 기반 혈압계 없이도 wearable device에서 지속적으로 수축기 혈압을 추정할 수 있는 compact한 모델을 만드는 것이다. 논문은 특히 telemonitoring과 remote monitoring 맥락에서, 환자가 일상생활 또는 가정 환경에서 지속적으로 혈압 상태를 관찰할 수 있는 기술의 필요성을 강조한다.

연구 문제는 “PPG signal의 한 beat morphology만을 입력으로 사용하여 ANN이 수축기 혈압을 충분히 정확하게 추정할 수 있는가”이다. 기존 cuff-based blood pressure measurement는 임상에서 널리 사용되지만, cuff inflation으로 인한 불편함이 있고 연속 측정이 어렵다. Invasive blood pressure, 즉 IBP는 beat-to-beat로 정확한 혈압을 제공하지만 ICU와 같은 병원 환경에서만 사용 가능하고 감염, 혈종, 혈전 등의 위험이 있다. 따라서 non-invasive, cuffless, continuous BP monitoring은 고혈압, cardiovascular diseases, neurodegenerative disease 환자의 자율신경계 이상 감시와 치료 조정에 중요한 의미를 갖는다.

이 논문은 특히 고혈압뿐 아니라 Parkinson’s Disease와 같은 neurodegenerative disease 환자에서 나타날 수 있는 autonomic control impairment를 언급한다. 정상적인 circadian arterial pressure cycle이 역전되면 cardiovascular risk와 fall risk가 증가할 수 있으므로, continuous pressure monitoring이 예후 평가 및 치료 조정에 도움이 될 수 있다는 문제의식을 제시한다.

방법론적으로 저자들은 두 개의 데이터셋을 사용한다. 첫 번째는 PhysioNet/MIMIC III에서 추출한 hospital environment 기반 PPG와 arterial blood pressure, 즉 ABP 신호로 구성된다. 이 데이터셋은 ANN 학습에 사용되며, 최종적으로 249,672개의 PPG period와 그에 대응하는 SP value를 포함한다. 두 번째는 STMicroelectronics s.r.l.에서 MORFEA3 wearable device로 수집한 자체 데이터셋이다. 이 데이터셋은 8명의 건강한 피험자로부터 수집되었고, cuff-based digital sphygmomanometer를 reference로 사용한다. 이 두 번째 데이터셋은 학습된 모델의 외부적 또는 실사용 유사 환경 test에 사용된다.

논문에서 제안한 ANN은 PPG 한 주기, 즉 one beat period를 100개 sample로 zero-padding하여 입력하고, 하나의 scalar SP 값을 출력하는 multilayer feed-forward neural network이다. 최적 구조는 $[100, 80, 100, 60]$ architecture로 보고되며, 최종 output neuron은 linear activation function을 사용한다. 모델은 MIMIC III dataset에서 학습되고, MIMIC III test set과 MORFEA3 dataset에서 평가된다.

주요 결과는 MIMIC III test set에서 MAE 2.99 mmHg, STD 3.37 mmHg이고, MORFEA3 dataset에서 MAE 3.85 mmHg, STD 4.29 mmHg이다. MORFEA3 test 결과는 AAMI 기준, 즉 평균 오차 5 mmHg 이하와 표준편차 8 mmHg 이하 조건을 만족하고, BHS 기준에서는 Grade A를 달성한다고 논문은 보고한다. 다만 자체 데이터셋은 8명으로 구성되어 있어 AAMI/BHS general population protocol에서 요구하는 85명 기준에는 크게 미달한다. 따라서 이 연구는 promising하지만 preliminary study로 해석해야 한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 PPG signal morphology가 arterial blood pressure, 특히 systolic pressure와 비선형적으로 관련되어 있으며, 이를 ANN이 학습할 수 있다는 것이다. PPG는 피부 표면에서 LED와 photodetector를 이용하여 blood volume fluctuation을 측정하는 optical non-invasive signal이다. PPG waveform의 AC component는 heartbeat에 따른 blood volume change를 반영하고, DC component는 respiration, thermoregulation, sympathetic nervous system activity 등에 의해 천천히 변화한다.

기존 cuffless BP estimation의 대표적인 접근은 ECG와 PPG를 동시에 사용하여 pulse transit time, 즉 PTT 또는 pulse wave velocity, 즉 PWV를 계산하는 방식이다. 이 방식은 혈압 상승에 따라 arterial stiffness가 증가하고 pressure wave propagation velocity가 변화한다는 생리학적 근거가 있다. 그러나 PTT/PWV 기반 방법은 적어도 두 개의 sensor가 필요하고, ECG와 PPG의 시간 동기화가 필요하며, sensor 간 거리와 calibration 문제가 발생한다. 이는 착용자의 움직임 자유도를 제한하고 wearable device 설계를 복잡하게 만든다.

이 논문은 이러한 한계를 줄이기 위해 PPG-only 방식을 선택한다. 즉 ECG 없이 PPG 한 신호만으로 SP를 추정하려 한다. 이 접근은 sensor 수를 줄이고 device를 compact하게 만들 수 있으며, continuous telemonitoring 또는 wearable remote monitoring에 유리하다.

두 번째 핵심 아이디어는 handcrafted feature를 별도로 계산하여 regressor에 입력하는 대신, PPG period 자체를 일정 길이의 vector로 만들어 ANN에 직접 입력하는 것이다. 기존 연구 중 일부는 PPG waveform에서 21개의 time-domain feature를 추출하여 ANN에 입력하거나, time-domain, frequency-domain, complexity feature를 계산하여 ensemble regressor에 입력한다. 반면 이 논문은 각 PPG period를 100개의 sample로 맞춘 뒤 그 자체를 ANN input으로 사용한다. 이는 명시적인 fiducial feature engineering을 줄이고, ANN이 waveform morphology와 SP 사이의 nonlinear relationship을 직접 학습하도록 하는 설계이다.

세 번째 핵심 아이디어는 모델의 embedded deployment 가능성을 고려한다는 점이다. 저자들은 최종 regression model을 STM32L4+ family의 Micro-Controller Unit, 즉 MCU에 embedding하는 것을 목표로 한다고 설명한다. 따라서 단순히 성능이 높은 모델이 아니라, memory capacity와 computational cost가 제한된 wearable system에 들어갈 수 있는 모델을 지향한다. 이 때문에 CNN이나 복잡한 recurrent model 대신 relatively simple multilayer feed-forward ANN을 사용한 것으로 해석된다.

네 번째 핵심 아이디어는 hospital-based dataset으로 학습한 모델을 자체 wearable device에서 수집한 데이터로 평가하는 것이다. MIMIC III는 ABP reference가 있는 대규모 병원 데이터이고, MORFEA3 dataset은 실제 prototype wearable device에서 수집된 PPG와 cuff reference를 포함한다. 두 데이터의 acquisition environment와 reference 방식이 다르기 때문에, MORFEA3 test는 모델이 완전히 동일한 데이터 조건에만 맞춰진 것은 아닌지 확인하는 의미가 있다. 다만 MORFEA3 데이터는 controlled environment와 static position에서 수집되었기 때문에, 실제 everyday-life motion artifact 환경까지 검증한 것은 아니다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

논문의 전체 시스템은 크게 data acquisition, preprocessing, PPG period segmentation, ANN regression model training, test 및 clinical standard evaluation으로 구성된다. 먼저 MIMIC III에서 PPG와 ABP를 추출한다. ABP signal에서 systolic pressure peak를 검출하여 reference SP value를 얻고, PPG signal은 filtering 후 beat 단위로 분할한다. 각 PPG beat period는 길이가 다르므로 zero-padding을 통해 100개 sample로 맞춘다. 이렇게 구성된 dataset의 각 row는 100개의 PPG sample과 마지막 column의 SP target으로 구성된다.

그다음 ANN regression model을 Keras로 layer-by-layer 구현한다. 모델은 100개의 input neuron을 가지며, 이는 각 PPG period의 sample 수와 같다. Hidden layer는 fully connected dense layer로 구성되고, output layer는 1개의 neuron과 linear activation function을 사용한다. Output은 해당 PPG period에 대응하는 SP 추정값이다.

모델 구조와 hyperparameter는 MIMIC III test set에서의 MAE를 기준으로 선택된다. 저자들은 단순한 model에서 출발하여 hidden node 수와 hidden layer 수를 늘려가며 성능 변화를 관찰했다. 너무 단순한 모델은 표현력이 부족하고, 너무 복잡한 모델은 overfitting으로 인해 test performance가 감소할 수 있다. 최종적으로 가장 좋은 generalization property를 보인 architecture와 hyperparameter 조합을 Grid Search로 선택하였다.

학습된 모델은 두 가지 데이터에서 평가된다. 첫째, MIMIC III test set에서 평가하여 기존 literature와 비교한다. 둘째, MORFEA3 wearable device로 수집한 자체 dataset에서 평가하여 실제 prototype sensor 기반 데이터에 대한 성능을 확인한다.

### 3.2 MIMIC III 데이터 구성 및 전처리

첫 번째 데이터셋은 MIMIC III에서 파생되었다. 논문은 ICU 환자 47명의 PPG와 ABP signal을 사용했다고 설명한다. 신호 sampling frequency는 125 Hz이다. MIMIC III는 임상 데이터와 physiological waveform을 포함하는 대규모 공개 데이터베이스이지만, 본 연구에서는 그중 PPG와 ABP를 사용할 수 있는 일부 환자 데이터만 추출한 것으로 보인다.

ABP signal은 reference SP를 얻기 위해 사용된다. ABP 전처리에는 4th order Butterworth low-pass filter가 사용되며, cutoff frequency는 6.6 Hz이다. 이 filtering은 ABP의 high-frequency noise를 제거하기 위한 것이다. 이후 Matlab의 `findpeaks` function으로 systolic pressure peak를 검출한다. 검출된 각 SP value는 corresponding PPG period와 연결된다.

PPG signal은 band-pass filter로 전처리된다. Bandwidth는 0.5 Hz에서 7 Hz이며, 논문은 PPG pulse wave frequency가 일반적으로 0.5–4.0 Hz 범위에 있기 때문에 이 대역을 선택한 것으로 설명한다. Filtering 후 PPG는 one beat period 단위로 segmentation된다. 각 beat period의 길이는 heart rate 변화에 따라 다르므로, ANN input dimension을 고정하기 위해 모든 period를 100 samples로 zero-padding한다. 저자들은 zero-padding을 통해 long period와 short period의 PPG feature를 보존하려 한다고 설명한다.

최종 MIMIC III dataset은 249,672개의 PPG period와 corresponding SP value로 구성된다. 논문은 이 데이터가 $80,mmHg < SP < 180,mmHg$ 범위를 포함하며, American Heart Association classification에서 hypotension을 제외한 Desired, Pre-Hypertension, Stage 1 Hypertension, Stage 2 Hypertension class를 포함한다고 설명한다.

### 3.3 MORFEA3 데이터 구성 및 전처리

두 번째 데이터셋은 STMicroelectronics s.r.l.에서 수집한 자체 데이터셋이다. 측정에는 MORFEA3 wearable prototype이 사용되었다. MORFEA3는 STMicroelectronics remote monitoring group에서 개발한 prototype board이며, multispectral PPG reading이 가능하다. 장치에는 3-axis accelerometer, 두 개의 VD6283 spectrometer, NTC thermistor, MCU 등이 포함되어 있고, Bluetooth connection으로 외부와 통신한다.

MORFEA3는 white LED를 light source로 사용하고, spectrometer로 white light beam을 component로 분리하여 green wavelength contribution을 측정한다. PPG는 오른손 손가락, 구체적으로 right hand index fingertip에서 측정된다. Finger-device contact을 안정적으로 유지하기 위해 손가락을 넣을 수 있는 case에 장치를 배치하였다.

자체 데이터는 8명의 healthy subjects에게서 수집되었다. 피험자는 남성 5명, 여성 3명이며 age range는 25–50세이다. 측정은 resting position에서 진행되었고, everyday-life activity 중이 아니라 controlled environment의 static position에서 이루어졌다. Reference BP는 왼팔에 감은 digital cuff-based sphygmomanometer로 동시에 측정되었고, acquisition time과 함께 Excel file에 기록되었다.

MORFEA3의 PPG sampling frequency는 62.5 Hz이다. 그러나 ANN model의 input neuron 수는 MIMIC III에서 학습할 때 100으로 고정되어 있고, MIMIC III PPG는 125 Hz sampling에 기반한다. 따라서 MORFEA3 PPG는 MIMIC III와 같은 처리 조건에 맞추기 위해 125 Hz로 resampling되었다. 이후 MIMIC III PPG와 동일한 4th order Butterworth filter를 사용해 filtering하고, beat period 단위로 segmentation한다.

MORFEA3 dataset은 총 6,460개의 PPG period로 구성되며, 각 period는 100 samples로 맞춰지고 corresponding SP target이 추가된다. 다만 reference sphygmomanometer는 continuous signal을 제공하지 않고 discrete measurement만 제공한다. 따라서 각 PPG period에 정확히 대응하는 순간 SP를 직접 얻을 수 없다. 논문은 두 SP acquisition instants 사이에서 consecutive SP records를 평균하여 각 PPG period에 SP value를 부여했다고 설명한다. 이는 practical solution이지만, true beat-level SP label과는 차이가 있을 수 있다.

또한 MORFEA3 dataset에는 hypotension, Stage 1 Hypertension, Stage 2 Hypertension 같은 pathological values가 포함되지 않는다. 건강한 피험자, resting position, controlled measurement 조건이기 때문에 Desired 또는 Pre-Hypertension에 가까운 제한된 range만 포함된 것으로 보인다. 따라서 pathological BP range에서의 성능은 이 dataset으로 평가할 수 없다.

### 3.4 ANN regression model 구조

이 연구의 ANN은 multilayer feed-forward neural network이다. 첫 번째 layer는 input layer이며 100개의 neuron으로 구성된다. 이는 zero-padded PPG period의 sample 수와 일치한다. Hidden layer는 dense layer로 구성된다. Dense layer는 한 layer의 모든 neuron이 다음 layer의 모든 neuron과 연결되는 fully connected layer이다. Output layer는 1개의 neuron을 가지며, linear activation function을 사용한다. 이 output neuron은 predicted SP value를 직접 출력한다.

저자들은 여러 network architecture를 실험하였다. 가장 처음 사용한 model은 기존 연구에서 영감을 받은 단순 구조였고, 이후 hidden node 수와 hidden layer 수를 증가시키며 MIMIC III test set 성능을 비교했다. 모델 복잡도를 늘리면 어느 지점까지는 성능이 향상되지만, 이후 overfitting 때문에 test error가 증가할 수 있다고 논문은 설명한다.

최종 선택된 architecture는 다음과 같다.

$$
[100, 80, 100, 60]
$$

이 표기는 input 100개, hidden layer 80개 neuron, hidden layer 100개 neuron, hidden layer 60개 neuron, 그리고 output 1개 neuron으로 구성된 구조를 의미하는 것으로 해석된다. 논문은 output layer를 별도로 1 neuron이라고 설명하므로, $[100,80,100,60]$은 output 이전까지의 주요 layer 크기를 나타내는 것으로 보는 것이 자연스럽다.

최종 hyperparameter 조합은 다음과 같다. Activation function은 Softsign, optimization algorithm은 stochastic gradient descent, 즉 SGD, initialization mode는 glorot uniform이다. Learning rate는 0.007, dropout rate는 0.2, batch size는 1024이다. 초기 실험에서는 sigmoid activation, Adam optimizer, batch size 512, epoch 500 등이 사용되었으나, 최종 최적 조합에서는 Softsign과 SGD가 선택되었다. 제공된 텍스트에서는 최종 epoch 수가 명확히 다시 제시되지 않으므로, 최종 model training에서 epoch 500을 그대로 사용했는지는 확정할 수 없다.

Dropout 0.2는 overfitting을 줄이기 위한 regularization으로 사용된다. Glorot uniform initialization은 dense layer의 weight를 안정적으로 초기화하여 gradient flow를 개선하는 역할을 한다. Softsign activation은 sigmoid보다 gradient saturation이 완화될 수 있는 smooth nonlinear activation이다.

### 3.5 손실 및 평가 지표

모델의 predictive performance는 Mean Absolute Error, 즉 MAE로 평가된다. 논문은 MAE를 다음과 같이 정의한다.

$$
MAE=\frac{\sum |y-y'|}{N}
$$

여기서 $y$는 actual value, $y'$는 predicted value, $N$은 sample point 수이다. MAE는 예측 오차의 절대값 평균으로, 단위가 mmHg로 유지되기 때문에 혈압 추정 성능을 직관적으로 해석할 수 있다. 또한 Root Mean Squared Error, 즉 RMSE보다 outlier에 덜 민감하다는 점에서 본 연구의 주요 평가 지표로 사용되었다.

논문은 AAMI와 BHS 기준도 함께 고려한다. AAMI 기준은 혈압 측정 장치가 평균 오차 5 mmHg 이하, 표준편차 8 mmHg 이하를 만족해야 한다는 조건으로 설명된다. 논문에서는 “MAE ≤ 5 mmHg with SD ≤ 8 mmHg”로 표현하지만, 엄밀히는 AAMI에서는 일반적으로 mean error와 standard deviation을 기준으로 평가한다. 논문은 자체 결과가 이 기준을 만족한다고 해석한다.

BHS 기준은 기준 장치와 시험 장치의 absolute difference가 5 mmHg 이내인 비율, 10 mmHg 이내인 비율, 15 mmHg 이내인 비율로 평가한다. Grade A 기준은 각각 60%, 85%, 95% 이상이어야 한다. 본 논문은 MIMIC III test set과 MORFEA3 dataset에서 이 세 threshold별 비율을 제시한다.

### 3.6 MCU embedding 관점

논문은 최종 ANN regression model을 STM32L4+ family MCU에 embedding하는 것을 목표로 한다고 명시한다. 따라서 모델의 computational cost와 memory capacity requirement를 평가하는 것이 중요하다. 이를 위해 STM32CubeMX graphical tool이 사용되었다. 논문 Figure 4는 ANN regression model이 차지하는 Flash와 RAM memory occupation percentage를 보여주지만, 제공된 텍스트에는 정확한 수치가 명시되어 있지 않다. 따라서 본 보고서에서는 memory 점유율의 정량값은 제시할 수 없다.

이 설계 관점은 본 연구가 단순한 offline analysis가 아니라 wearable remote monitoring device로의 적용을 염두에 둔 연구임을 보여준다. 다만 실제 MCU에서의 inference latency, energy consumption, 실시간 PPG preprocessing cost는 제공된 텍스트에서 상세히 보고되지 않는다.

## 4. 실험 및 결과

### 4.1 MIMIC III test set 결과

최종 ANN regression model은 MIMIC III training data로 학습된 후 MIMIC III test set에서 평가되었다. MIMIC III test set은 74,902개의 PPG period로 구성된다. 논문 Table 2에 따르면 제안 모델은 MIMIC III test set에서 MAE 2.99 mmHg, STD 3.37 mmHg를 기록하였다.

이는 기존 연구와 비교하여 우수한 수치로 제시된다. Kurylyak et al.의 ANN 기반 연구는 21개의 input parameter를 사용했고 MAE 3.80 mmHg, STD 3.46 mmHg를 기록하였다. Slapničar et al.의 ensemble of regression trees와 RReliefF selected feature subset 기반 연구는 MAE 4.90 mmHg, STD 6.59 mmHg를 기록하였다. 제안 모델은 이 두 결과보다 낮은 MAE와 낮거나 유사한 STD를 보인다.

MIMIC III test set에서 absolute error threshold별 성능도 매우 높다. 전체 SP 값 중 83.48%가 5 mmHg 미만의 absolute error로 예측되었고, 96.34%가 10 mmHg 미만, 98.88%가 15 mmHg 미만의 error로 예측되었다. 이는 BHS 기준으로 보면 Grade A 조건인 60%, 85%, 95%를 충분히 초과한다.

Figure 8은 MIMIC III test set에서 actual SP value와 predicted SP value를 비교한 예시를 보여준다. 논문은 예측값을 연속적인 방식으로 연결하여 표시했지만, 실제로는 PPG period별 scalar SP prediction이다. 즉 모델은 continuous ABP waveform을 생성하는 것이 아니라, 각 PPG beat period에 대해 하나의 SP 값을 추정한다.

### 4.2 MORFEA3 dataset 결과

MORFEA3 dataset은 자체 wearable device로 수집한 6,460개의 PPG period로 구성된다. 이 dataset에서 제안 모델은 MAE 3.85 mmHg, STD 4.29 mmHg를 달성하였다. 논문은 이를 Jozef Stefan Institute, 즉 JSI에서 수집한 everyday-life dataset 결과와 비교한다. JSI collected test set의 결과는 MAE 7.87 mmHg, STD 7.47 mmHg로 보고되어 있으며, 제안 모델은 MORFEA3 dataset에서 이보다 상당히 낮은 error를 보였다.

MORFEA3 dataset에서 absolute error threshold별 성능은 5 mmHg 미만 74.58%, 10 mmHg 미만 90.55%, 15 mmHg 미만 97.27%이다. 이는 BHS Grade A 기준인 60%, 85%, 95%를 모두 만족한다. 또한 MAE 3.85 mmHg와 STD 4.29 mmHg는 논문이 제시한 AAMI 기준인 MAE ≤ 5 mmHg와 SD ≤ 8 mmHg를 만족한다.

Figure 9는 MORFEA3 test set에서 actual SP와 predicted SP를 비교한 예시이다. 여기서 actual SP는 cuff-based sphygmomanometer에서 얻은 discrete measurement이다. 논문은 시각적 단순화를 위해 predicted SP와 actual SP를 continuous curve처럼 연결해서 나타냈다고 설명한다. 그러나 실제 reference는 continuous ABP가 아니라 간헐적 cuff measurement라는 점이 중요하다.

### 4.3 결과의 의미

실험 결과는 PPG one-beat waveform을 100-sample vector로 입력하는 relatively simple ANN이 SP estimation에서 의미 있는 정확도를 낼 수 있음을 보여준다. 특히 MIMIC III test set에서는 MAE 2.99 mmHg로 기존 feature-based ANN 및 ensemble regression tree 방식보다 좋은 성능을 보였다. 이는 raw 또는 quasi-raw PPG period morphology를 직접 ANN에 입력하는 방식이 handcrafted feature 기반 접근보다 유용할 수 있음을 시사한다.

MORFEA3 dataset에서 성능이 MAE 3.85 mmHg로 유지된 것은 hospital database에서 학습한 모델이 자체 wearable prototype에서 수집한 데이터에도 어느 정도 적용될 수 있음을 보여준다. 이는 모델의 실사용 가능성 측면에서 긍정적이다. 다만 MORFEA3 dataset은 건강한 8명, resting position, controlled environment에서 수집되었으므로, 진정한 everyday-life motion condition 또는 다양한 pathological BP range에서 일반화된다고 보기는 어렵다.

논문은 최종 결과가 “most clinical applications”에 compatible하다고 표현한다. 그러나 clinical application의 범위는 매우 넓고, 실제 혈압 측정기 인증에는 대상자 수, 혈압 범위, 성별 비율, 연령, measurement protocol 등이 엄격하게 요구된다. 논문 스스로도 8명만 테스트했다는 점과 pathological SP range가 부족하다는 점을 한계로 인정한다.

## 5. 강점, 한계

이 논문의 첫 번째 강점은 PPG-only 혈압 추정이라는 실용적인 문제 설정이다. ECG와 PPG를 함께 사용하는 PTT/PWV 방식은 동기화와 다중 sensor 배치가 필요하지만, PPG-only 방식은 wearable device를 더 compact하게 만들 수 있다. 특히 MORFEA3 같은 prototype device와 결합하여 실제 remote monitoring application을 염두에 두었다는 점은 응용 가능성을 높인다.

두 번째 강점은 모델이 비교적 단순하다는 점이다. CNN, LSTM, attention network 같은 복잡한 deep learning 구조를 사용하지 않고, 100-sample PPG period를 입력하는 multilayer feed-forward ANN을 사용했다. 이 단순성은 MCU embedding, memory requirement, low-power wearable system 측면에서 장점이 될 수 있다. 논문이 STM32CubeMX를 사용하여 memory occupation을 검토한 것도 이러한 embedded deployment 관점을 반영한다.

세 번째 강점은 MIMIC III 기반 학습 후 자체 wearable device dataset에서 평가했다는 점이다. 많은 연구가 public dataset 내부에서만 평가되는데, 이 논문은 MORFEA3 prototype으로 수집한 데이터를 별도로 사용하였다. 물론 통제된 환경이지만, signal acquisition device와 reference 방식이 다른 dataset에서 test했다는 점은 의미가 있다.

네 번째 강점은 BHS threshold별 error percentage를 제공했다는 점이다. 단순히 MAE만 보고하는 경우보다, 5, 10, 15 mmHg threshold 내 error 비율을 제시하면 임상적 기준과의 관계를 더 직접적으로 해석할 수 있다. MIMIC III와 MORFEA3 모두 BHS Grade A 조건을 만족한다는 결과는 모델의 예측 안정성을 보여준다.

다섯 번째 강점은 기존 literature와의 비교를 포함한다는 점이다. 제안 ANN은 MIMIC III test set에서 21 handcrafted feature 기반 ANN과 ensemble regression tree 방식보다 낮은 MAE를 보였다. 이는 PPG period sample 자체를 ANN에 입력하는 단순한 설계가 충분히 경쟁력 있음을 보여준다.

반면 한계도 명확하다. 첫째, MORFEA3 dataset의 subject 수가 매우 작다. 논문은 AAMI와 BHS general population study protocol에서 85명 이상의 subject가 요구된다고 설명하지만, 본 연구의 자체 test는 8명뿐이다. 따라서 MORFEA3 결과가 AAMI/BHS의 형식적 성능 조건을 만족하더라도, 엄밀한 validation protocol을 만족한다고 볼 수는 없다.

둘째, MORFEA3 dataset에는 pathological BP range가 포함되지 않는다. Hypotension, Stage 1 Hypertension, Stage 2 Hypertension 값이 representation되지 않았다고 논문은 명시한다. 따라서 실제 고혈압 환자 또는 저혈압 환자에게 모델이 정확히 작동할지는 검증되지 않았다. 특히 혈압 추정 모델은 정상 범위에서는 높은 성능을 보이더라도 극단 혈압 범위에서는 error가 커질 수 있다.

셋째, 자체 dataset은 everyday-life dataset이라기보다 controlled static environment dataset에 가깝다. Abstract에서는 PPG signal during everyday-life activity의 artifact가 어렵다고 설명하지만, 실제 MORFEA3 dataset은 motion artifact 문제 때문에 controlled environment in a static position에서 수집되었다. 따라서 보행, 운동, 손 움직임, sensor displacement, 피부 접촉 변화 같은 실제 wearable artifact 환경에서의 robustness는 검증되지 않았다.

넷째, reference label 문제도 한계이다. MIMIC III에서는 continuous ABP를 사용하므로 beat-level SP label을 얻을 수 있다. 반면 MORFEA3에서는 cuff-based sphygmomanometer가 discrete measurement만 제공한다. 저자들은 두 consecutive SP records를 평균하여 각 PPG period에 target을 부여했다. 하지만 이 방식은 period별 true SP variation을 반영하지 못하며, reference timing mismatch 또는 cuff measurement uncertainty가 발생할 수 있다.

다섯째, 모델은 SP만 추정하고 DBP는 추정하지 않는다. 혈압 monitoring에서는 SBP와 DBP가 모두 중요하며, pulse pressure나 mean arterial pressure도 임상적으로 의미가 있다. 기존 연구 중 일부는 SBP와 DBP를 동시에 예측하지만, 본 논문은 SP estimation에 집중한다. 따라서 완전한 BP monitoring system으로 확장하려면 DBP estimation도 추가되어야 한다.

여섯째, PPG period를 zero-padding하여 100 samples로 맞추는 방식은 단순하지만, period length 자체에 담긴 heart rate information을 왜곡하거나 padding pattern을 학습할 가능성이 있다. 물론 zero-padding은 다양한 period length를 하나의 fixed input size에 맞추기 위한 실용적 방법이지만, resampling, interpolation, time normalization 등 다른 방식과의 비교는 제공되지 않는다.

일곱째, train/test split과 subject independence에 대한 설명이 제한적이다. MIMIC III에서 249,672 periods를 사용했다고 하지만, training set과 test set이 subject-independent 방식으로 나뉘었는지 명확히 제시되지 않는다. 만약 같은 patient의 PPG period가 train과 test에 동시에 포함되었다면, 모델 성능이 subject-specific pattern에 의해 과대평가될 수 있다. 이 점은 PPG 기반 혈압 추정 연구에서 매우 중요한 평가 설계 문제이다.

마지막으로, wrist PPG에 대한 일반화가 되지 않는다. 논문은 finger acquisition에는 모델이 valid하지만, wrist PPG로 SP를 추정하려 하면 성능이 저하된다고 명시한다. Wrist wearable device는 실제 시장 적용에서 매우 중요하므로, wrist PPG training set을 별도로 구축해야 한다는 점은 실용적 한계이다.

## 6. 결론

이 논문은 PPG signal만을 이용하여 systolic blood pressure를 추정하는 ANN 기반 regression model을 제안하였다. 모델은 100-sample로 zero-padding된 PPG one-beat period를 입력으로 받고, 하나의 SP value를 출력하는 multilayer feed-forward ANN이다. 최종 architecture는 $[100,80,100,60]$으로 보고되며, Softsign activation, SGD optimizer, glorot uniform initialization, learning rate 0.007, dropout 0.2, batch size 1024가 최적 hyperparameter로 선택되었다.

실험에서는 MIMIC III에서 추출한 249,672개의 PPG period와 ABP-derived SP value를 사용하여 모델을 학습하고 평가하였다. MIMIC III test set에서 제안 모델은 MAE 2.99 mmHg, STD 3.37 mmHg를 달성하여 기존 feature-based ANN 및 ensemble regression tree 방식보다 좋은 성능을 보였다. 또한 자체 MORFEA3 wearable device dataset에서는 MAE 3.85 mmHg, STD 4.29 mmHg를 기록하였다. MORFEA3 결과는 논문이 제시한 AAMI 조건과 BHS Grade A 기준을 만족한다.

이 연구의 주요 기여는 PPG-only, low-complexity ANN으로도 SP estimation에서 promising한 성능을 얻을 수 있음을 보인 점이다. 특히 ECG 없이 PPG만 사용하고, 복잡한 handcrafted feature extraction 없이 PPG period sample을 직접 입력한다는 점에서 wearable remote monitoring device에 적합한 방향을 제시한다. MORFEA3 prototype을 사용한 자체 데이터 평가도 실제 device integration 가능성을 보여주는 요소이다.

그러나 이 연구는 preliminary study로 보는 것이 타당하다. 자체 dataset은 8명의 healthy subject만 포함하고, pathological SP range가 거의 없으며, controlled static condition에서 수집되었다. 또한 DBP estimation은 다루지 않고, MIMIC III split의 subject independence가 명확하지 않으며, wrist PPG에서는 성능이 저하된다고 보고된다. 실제 clinical validation을 위해서는 AAMI/BHS protocol에 맞는 충분한 subject 수, 다양한 혈압 범위, wrist 및 daily-life motion condition, 외부 기관 데이터, 그리고 DBP까지 포함한 전체 BP estimation 평가가 필요하다.

종합하면, 이 논문은 compact wearable device에서 PPG-only systolic pressure estimation을 구현하기 위한 실용적이고 단순한 ANN baseline을 제시한다. 성능은 유망하지만, 임상 적용과 제품화를 위해서는 더 큰 규모의 subject-independent validation과 실제 생활 환경에서의 artifact robustness 검증이 필수적이다.
