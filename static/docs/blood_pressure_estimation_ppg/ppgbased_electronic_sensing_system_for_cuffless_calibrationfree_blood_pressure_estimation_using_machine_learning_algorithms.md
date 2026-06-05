# PPG-Based Electronic Sensing System for Cuff-Less Calibration-Free Blood Pressure Estimation Using Machine Learning Algorithms

* **저자**: Chiara Botrugno, Francesco Dell’Olio
* **발표연도**: 2024

## 1. 논문 개요

이 논문은 Photoplethysmography, 즉 PPG 신호만을 이용하여 cuff-less, calibration-free 방식으로 systolic blood pressure, SBP와 diastolic blood pressure, DBP를 추정하는 전자 sensing system을 제안한다. 논문의 핵심 목표는 ECG나 별도의 두 번째 센서를 사용하지 않고, multi-wavelength PPG 신호와 machine learning 알고리즘을 결합하여 혈압을 연속적이고 비침습적으로 추정하는 것이다.

연구 문제는 기존 cuff 기반 혈압 측정의 불편함과, PTT, PWV, PAT와 같은 간접 추정 방식의 한계를 줄이는 데 있다. PTT나 PAT 기반 방식은 일반적으로 ECG와 PPG를 함께 사용하거나 서로 다른 위치의 센서가 필요하며, sensor 간 거리와 subject-specific calibration이 필요할 수 있다. 이 논문은 이러한 복잡성을 줄이고자 단일 PPG sensing system에서 얻은 신호만으로 혈압을 예측하려 한다.

문제의 중요성은 cardiovascular disease의 조기 탐지와 지속 관리에 있다. 혈압은 주요 vital sign이며, cardiovascular disease의 예방, 진단, 치료 시점 판단에 중요한 지표이다. 그러나 전통적인 cuff-based blood pressure measurement는 환자에게 불편하고 연속 모니터링에 적합하지 않다. 반면 PPG는 피부 아래 vascular bed의 blood volume 변화를 optical 방식으로 측정하므로 wearable device나 digital health system에 통합하기 쉽다. 이 논문은 PPG 기반 혈압 추정이 병원, 가정, care environment에서 continuous monitoring을 가능하게 할 수 있다고 본다.

제안 시스템은 MAX86916 Evaluation System을 사용하여 multi-wavelength PPG를 획득한다. 이 optical module은 infrared, red, green, blue 네 파장의 LED와 photodiode를 포함한다. 논문은 wavelength별 피부 침투 깊이와 motion artifact 민감도가 다르다는 점을 활용한다. 긴 파장인 infrared와 red는 심장 및 혈관 활동을 더 잘 반영하지만 motion artifact에 취약하고, 짧은 파장인 green과 blue는 깊은 cardiac activity 정보는 상대적으로 적지만 motion artifact에 덜 민감하다. 이러한 multi-wavelength 특성을 이용해 Independent Component Analysis, ICA를 수행하고, 가장 주기적인 component를 선택하여 혈압 예측에 사용한다.

혈압 예측 알고리즘은 Feed-Forward Artificial Neural Network, ANN이다. ANN 입력은 PPG에서 추출한 21개 feature이며, 이 중 15개는 time-domain morphology와 관련된 temporal feature이고, 6개는 power spectral density에서 얻은 spectral feature이다. 출력은 SBP와 DBP 두 값이다. 모델은 PPG-BP Database로 학습하고, 저자들이 MAX86916 Evaluation Kit로 새롭게 수집한 88명의 multi-wavelength PPG dataset으로 테스트한다. 최종 test set 결과는 SBP에서 MAE 5.08 &plusmn; 8.83 mmHg, DBP에서 MAE 4.37 &plusmn; 7.08 mmHg이다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 multi-wavelength PPG 신호를 이용해 motion artifact를 줄이고, temporal feature와 spectral feature를 함께 사용하여 calibration-free blood pressure estimation을 수행하는 것이다. 단일 wavelength PPG는 sensor placement, skin properties, biological factors, motion artifact에 민감하다. 저자들은 서로 다른 wavelength가 피부와 혈관에서 다르게 상호작용한다는 점을 이용하면 PPG 신호의 physiological component를 더 안정적으로 추출할 수 있다고 본다.

기존 PTT나 PAT 방식은 혈압과 pulse wave propagation time 사이의 관계를 이용하지만, ECG와 PPG를 함께 측정해야 하거나 두 PPG sensor를 일정 거리로 배치해야 한다. 이 경우 calibration stage가 필요하고 hardware setup이 복잡해진다. 반면 이 논문은 PPG only 접근을 택한다. 즉, 하나의 optical sensing system에서 얻은 PPG 신호만으로 SBP와 DBP를 직접 추정한다. 이는 wearable 또는 portable digital health device로 구현하기에 더 간단한 구조이다.

또 다른 핵심 아이디어는 ICA를 이용한 motion artifact reduction이다. Multi-wavelength PPG는 네 개의 관측 신호를 제공한다. 이 신호들은 cardiac pulsation component와 motion artifact component가 섞인 형태라고 볼 수 있다. ICA는 관측된 mixed signal을 서로 독립적인 component로 분리하는 방법이다. 논문은 heartbeat에 의한 pulsatile component가 주기적인 성질을 갖는다는 점에 근거하여, ICA 출력 중 periodicity가 가장 높은 component를 선택한다. 이 component가 혈압 예측에 사용할 physiological cardiac signal에 가장 가깝다고 판단한다.

모델링 측면에서 이 논문은 end-to-end deep learning보다는 handcrafted feature 기반 ANN을 사용한다. PPG 신호의 morphology를 반영하는 15개 temporal feature와 frequency-domain 정보를 반영하는 6개 spectral feature를 추출하고, 이를 feed-forward ANN에 입력한다. 이 접근은 복잡한 CNN, RNN, Transformer 모델보다 해석 가능성과 구현 단순성이 높으며, 상대적으로 작은 dataset에서도 학습이 가능하다는 장점이 있다.

논문이 강조하는 차별점은 calibration-free, ECG-free, multi-wavelength, feature-fusion 기반 ANN이라는 조합이다. 즉, 별도 calibration 없이, ECG 없이, multi-wavelength PPG에서 artifact를 줄이고, time-domain과 frequency-domain feature를 함께 사용하여 SBP와 DBP를 예측한다.

## 3. 상세 방법 설명

전체 시스템은 hardware acquisition, dataset 구성, PPG preprocessing, feature extraction, ANN 학습 및 평가로 구성된다. Hardware는 Analog Devices의 MAX86916 Integrated Optical Sensor Module 기반 Evaluation System이다. 이 module은 biodetection, proximity, color application을 위한 optical sensor platform이며, 내부 LED, photodetector, low-noise electronics, light rejection circuit을 포함한다. 전원은 sensor module에 1.8 V, LED에는 별도 5.5 V가 사용된다. 데이터 sampling frequency는 100 Hz이고, communication은 I2C-compatible interface를 통해 이루어진다.

MAX86916은 네 가지 wavelength를 사용한다. Infrared는 930–955 nm, red는 655–663 nm, green은 520–535 nm, blue는 455–466 nm 범위이다. Device Studio software를 이용해 각 time sample마다 IR, red, green, blue 네 PPG signal 값을 `.csv` 파일로 저장한다.

데이터는 학습용 dataset과 테스트용 dataset으로 나뉜다. 학습에는 공개 clinical database인 PPG-BP Database가 사용된다. 이 database는 중국 Guilin People’s Hospital에서 수집된 219명의 deidentified clinical data로 구성된다. 환자 나이는 21세부터 86세까지이며, 각 subject folder에는 2.1초 길이의 infrared PPG signal과 physiological data, SBP, DBP가 포함된다. 원래 waveform sampling rate는 1 kHz이고 A/D conversion precision은 12 bit이다. 이 database의 SBP는 80–182 mmHg 범위이며 평균은 127.95 mmHg, 표준편차는 20.38 mmHg이다. DBP는 42–107 mmHg 범위이며 평균은 71.84 mmHg, 표준편차는 11.11 mmHg이다.

테스트에는 저자들이 새롭게 수집한 dataset을 사용한다. 이 dataset은 MAX86916 Evaluation Kit로 획득한 88개의 multi-wavelength PPG signal과 reference sphygmomanometer로 측정한 SBP, DBP 값을 포함한다. 측정은 약 30초 동안 진행되며, 이 시간은 sphygmomanometer가 pressure value를 반환하는 데 필요한 시간이다. PPG는 오른쪽 검지를 optical sensor 위에 놓고 측정하고, 혈압은 왼팔에서 sphygmomanometer로 동시에 측정한다. 오른손과 왼팔을 나누어 사용한 이유는 PPG 측정 팔을 안정적으로 유지하여 motion artifact를 줄이기 위한 편의성 때문이다. 테스트 대상자는 남녀가 같은 수로 구성되어 있고, 나이는 19세부터 84세까지이다.

테스트 dataset의 SBP 범위는 95–171 mmHg, 평균은 119.68 mmHg, 표준편차는 15.96 mmHg이다. DBP 범위는 55–112 mmHg, 평균은 71.98 mmHg, 표준편차는 9.99 mmHg이다. 논문은 age group별 boxplot을 통해 두 dataset 모두에서 SBP가 나이와 함께 증가하고, DBP는 70대 이후 감소하는 경향을 보인다고 설명한다. 이는 aging에 따라 large artery가 더 rigid해지고, 같은 vascular resistance와 blood flow 조건에서 SBP는 높아지고 DBP는 낮아지는 현상과 일치한다. 저자들은 이를 두 dataset의 통계적 신뢰성을 뒷받침하는 근거로 사용한다.

PPG preprocessing은 세 단계로 구성된다. 첫째, filtering이다. 논문은 pulse wave frequency가 대략 0.5–4.0 Hz 범위에 있으므로, cut-off frequency 0.5 Hz와 5 Hz의 band-pass filter를 적용한다. 이 필터는 slow fluctuation과 high-frequency noise를 줄이고, heartbeat 관련 PPG component를 남기는 역할을 한다.

둘째, peaks removal이다. 특히 red와 infrared PPG signal에서 wide amplitude peak를 제거한다. 긴 wavelength인 red와 infrared는 피부를 더 깊이 침투하므로 heart와 blood artery activity를 더 잘 보여주지만, 긴 light path 때문에 motion artifact에도 더 민감하다. 논문은 Fig. 4에서 IR과 red signal에 빠르고 넓은 peak가 나타나며, 이를 제거해야 한다고 설명한다. 반면 green과 blue는 penetration depth가 짧아 cardiac activity 정보는 적지만 motion artifact에는 덜 민감하다.

셋째, Independent Component Analysis, ICA이다. Multi-wavelength PPG input은 네 개 signal이므로 ICA 출력도 네 개 independent component이다. ICA의 목적은 pulsation component와 motion artifact component가 섞인 관측 신호에서 physiological cardiac period와 가장 관련 있는 component를 추출하는 것이다. Heartbeat로 인한 blood volume change는 periodic component이므로, 각 independent component의 periodicity를 평가하여 가장 높은 periodicity를 가진 component를 선택한다. 이 선택된 independent component가 ANN feature extraction의 기반 signal로 사용된다.

ICA의 기본적인 관점은 관측 신호 $X$가 독립 source $S$들의 선형 혼합이라고 보는 것이다.

$$
X = AS
$$

여기서 $X$는 관측된 multi-channel PPG signal, $S$는 독립 source component, $A$는 mixing matrix이다. ICA는 $A$ 또는 그 역변환을 추정하여 $S$를 복원하려고 한다. 논문 자체에는 ICA 수식이 제시되어 있지 않지만, 방법의 의미는 네 wavelength PPG에서 공통 cardiac pulsation과 motion artifact가 섞여 있으므로, 이를 독립 component로 분해하고 heartbeat periodicity가 가장 강한 component를 선택하는 것이다.

Feature extraction은 temporal feature 15개와 spectral feature 6개로 구성된다. Temporal feature는 PPG signal의 time-domain morphology와 관련되며, single cardiac cycle 내부의 주요 interval을 고려하여 얻어진다. 논문은 Kurylyak 등의 기존 연구를 기반으로 15개 temporal feature를 사용한다고 설명하지만, 추출 텍스트에서는 15개 feature의 정확한 이름과 수식은 제시되지 않는다. 따라서 systolic peak 위치, diastolic peak 또는 notch, pulse interval, 상승 시간, 하강 시간 등과 관련될 가능성은 있지만, 구체 feature를 단정할 수는 없다.

Spectral feature는 power spectral density에서 직접 얻어진다. 총 6개 feature는 세 frequency band의 amplitude와 highest frequency를 고려하여 생성된다고 설명된다. 이 역시 각 band의 경계나 계산 방식은 제공된 텍스트에 명확히 제시되어 있지 않다. 다만 temporal morphology와 spectral distribution을 함께 사용함으로써 signal의 shape 정보와 frequency-domain 정보를 동시에 ANN에 제공한다는 점이 중요하다.

ANN 구조는 feed-forward multilayer network이다. 입력층은 21개 neuron으로 구성되며, 이는 15개 temporal feature와 6개 spectral feature에 대응한다. Hidden layer는 8개이고, 각 hidden layer의 neuron 수는 hyperparameter tuning을 통해 10개부터 60개 사이에서 선택된다. 출력층은 2개 neuron으로 구성되며, 각각 predicted SBP와 predicted DBP를 나타낸다. 학습은 error backpropagation algorithm을 사용한다.

모델의 목표는 실제 혈압값과 예측 혈압값 사이의 오차를 줄이는 것이다. 논문은 구체적인 training loss function을 명시하지 않지만, feed-forward ANN의 regression 학습에서 일반적으로 squared error 또는 mean squared error가 사용된다. 다만 제공된 텍스트에서는 손실 함수가 명시되어 있지 않으므로, MSE를 사용했다고 단정할 수는 없다. 평가는 Mean Absolute Error, MAE와 Standard Deviation, SD로 수행된다.

MAE는 다음과 같이 정의할 수 있다.

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

여기서 $y_i$는 reference sphygmomanometer로 측정한 실제 SBP 또는 DBP이고, $\hat{y}_i$는 ANN이 예측한 SBP 또는 DBP이다. MAE는 평균적으로 예측값이 실제값과 몇 mmHg 차이 나는지 보여주는 직관적 지표이다.

SD는 prediction error의 변동성을 나타낸다. 일반적으로 error $e_i = y_i - \hat{y}_i$와 mean error $\bar{e}$에 대해 다음과 같이 계산된다.

$$
\text{SD} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(e_i - \bar{e})^2}
$$

논문은 AAMI 기준을 목표로 한다고 설명한다. 제공된 텍스트에 따르면 AAMI requirement는 SBP와 DBP에 대해 MAE $\leq 5$ mmHg, SD $\leq 8$ mmHg이다. 실제 AAMI 기준은 보통 mean error와 standard deviation을 사용하지만, 이 논문은 MAE와 SD로 기준을 설명한다. 따라서 이 부분은 논문 내 표현을 따르되, 엄밀한 표준 해석에서는 추가 확인이 필요하다.

## 4. 실험 및 결과

실험은 공개 PPG-BP Database로 ANN을 학습하고, 새롭게 획득한 88명 test dataset으로 최종 검증하는 방식으로 진행된다. Training phase에서는 PPG-BP Database를 사용하며, validation set은 5-fold cross-validation 방식으로 구성된다. Hyperparameter tuning을 통해 hidden layer neuron 수를 선택한 뒤, 최종 network를 test dataset에 적용한다.

실험 결과는 Table I에 제시된다. Training set에서 SBP는 $5.33 \pm 10.16$ mmHg, DBP는 $4.26 \pm 7.65$ mmHg의 MAE ± SD를 보인다. Validation set에서는 SBP $6.98 \pm 9.30$ mmHg, DBP $6.03 \pm 8.55$ mmHg이다. Test set에서는 SBP $5.08 \pm 8.83$ mmHg, DBP $4.37 \pm 7.08$ mmHg이다.

이 결과를 해석하면, DBP는 test set에서 MAE와 SD 모두 논문이 언급한 AAMI 기준인 MAE 5 mmHg 이하, SD 8 mmHg 이하를 만족한다. DBP MAE는 4.37 mmHg이고 SD는 7.08 mmHg이다. 반면 SBP는 MAE가 5.08 mmHg로 기준보다 0.08 mmHg 높고, SD도 8.83 mmHg로 기준 8 mmHg를 초과한다. 따라서 논문이 말하듯 전체 결과는 promising하지만, 특히 SBP에서는 AAMI threshold에서 약간 벗어난다.

흥미로운 점은 training set과 test set의 결과가 단순히 training에서 가장 좋고 test에서 나쁜 형태가 아니라는 것이다. Training SBP는 $5.33 \pm 10.16$ mmHg이고 test SBP는 $5.08 \pm 8.83$ mmHg로, test가 오히려 약간 더 낮은 MAE와 SD를 보인다. DBP도 training $4.26 \pm 7.65$ mmHg, test $4.37 \pm 7.08$ mmHg로 유사하다. 반면 validation set에서는 SBP와 DBP 모두 오차가 더 크다. 이는 dataset 구성, fold split, sample distribution 차이, 또는 새 test dataset의 혈압 범위와 subject 특성 차이 때문일 수 있다. 다만 논문은 이에 대한 자세한 원인 분석을 제공하지 않으므로, 과도한 해석은 피해야 한다.

Regression analysis도 수행되었다. Fig. 5는 test set에서 actual value, 즉 target과 predicted value, 즉 output을 비교한 regression line을 SBP와 DBP 각각에 대해 보여준다. 제공된 텍스트에는 correlation coefficient, $R^2$, slope, intercept 값이 명시되어 있지 않으므로 정량적 선형성은 확인할 수 없다. 다만 논문은 reference device와 비교했을 때 promising한 결과가 관찰된다고 설명한다.

논문은 test set 결과가 AAMI threshold에서 약간 벗어난 이유로 training dataset과 test dataset의 acquisition protocol 차이를 제시한다. Training은 공개 PPG-BP Database의 infrared PPG signal을 사용하지만, test는 MAX86916 Board로 새롭게 수집한 multi-wavelength PPG signal이다. 센서, wavelength 구성, sampling frequency, measurement duration, acquisition setup, reference device, subject population이 다르기 때문에 domain shift가 발생할 수 있다. 이러한 protocol mismatch는 학습된 ANN이 test dataset에 적용될 때 성능 저하 또는 불안정성을 만들 수 있다.

또한 논문은 연구 대상자가 주로 healthy subjects aged between 20 and 70이라고 언급한다. 앞선 dataset 설명에서는 test 대상자의 전체 나이가 19–84세라고 되어 있지만, 결과 해석 부분에서는 건강한 20–70세 subject 중심이라고 설명한다. 이 표현에는 약간의 불일치가 있으므로, 실제 elderly subject나 pathology가 있는 subject에 대한 성능은 확실히 보장할 수 없다. 저자들도 elderly subjects suffering from pathologies에 대해 같은 성능을 보장할 수 없다고 명시한다.

실험 결과의 실제 의미는 다음과 같다. 제안 시스템은 calibration 없이 PPG-only로 혈압을 추정하면서, DBP에서는 기준에 가까운 성능을 달성했고, SBP에서도 기준에서 크게 벗어나지는 않았다. 그러나 SD가 특히 SBP에서 8.83 mmHg로 높기 때문에 개별 예측값의 변동성이 여전히 존재한다. Continuous monitoring이나 trend tracking에는 유용할 수 있지만, medical-grade diagnostic device로 인정받으려면 더 큰 dataset, 다양한 subject, 엄격한 validation protocol, 표준 기준에 맞춘 추가 평가가 필요하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 multi-wavelength PPG hardware와 machine learning algorithm을 결합한 실제 electronic sensing system을 제시했다는 점이다. 많은 연구가 공개 PPG dataset의 offline analysis에 머무르는 반면, 이 논문은 MAX86916 Evaluation Kit를 이용해 새로운 test dataset을 직접 수집하고 reference sphygmomanometer와 비교한다. 이는 실제 device-oriented 연구라는 점에서 의미가 있다.

두 번째 강점은 ECG를 사용하지 않는 PPG-only 접근이다. PTT나 PAT 방식은 ECG electrode나 두 개의 PPG sensor를 요구할 수 있고, calibration이 필요할 수 있다. 이 논문은 단일 optical sensor platform에서 얻은 PPG 신호만 사용하므로 hardware setup이 단순하고 wearable integration 가능성이 높다. 또한 calibration-free를 목표로 하므로, 사용자가 개인별 calibration을 반복해야 하는 부담을 줄일 수 있다.

세 번째 강점은 multi-wavelength 정보를 artifact reduction에 활용했다는 점이다. Red와 infrared는 deep penetration으로 cardiac activity를 잘 반영하지만 motion artifact에 취약하고, green과 blue는 motion artifact에는 덜 민감하다. ICA를 통해 네 wavelength signal에서 독립 component를 추출하고, 가장 주기적인 component를 선택하는 방식은 multi-channel PPG의 장점을 활용한 합리적 전처리 방법이다.

네 번째 강점은 temporal feature와 spectral feature를 함께 사용하는 것이다. PPG의 waveform morphology는 cardiovascular dynamics와 관련되고, spectral content는 pulse rhythm이나 frequency distribution과 관련될 수 있다. 15개 temporal feature와 6개 spectral feature를 결합하면 단순 time-domain feature만 사용할 때보다 더 풍부한 입력을 ANN에 제공할 수 있다.

다섯 번째 강점은 외부 test dataset을 사용했다는 점이다. 학습에 사용한 PPG-BP Database와 다른 acquisition setup에서 수집한 test dataset으로 평가했다는 것은 domain generalization을 어느 정도 확인하려는 시도이다. 많은 연구가 같은 dataset 내부 split만 사용하는 것과 비교하면 실용적 가치가 있다.

하지만 한계도 분명하다. 첫째, training dataset과 test dataset의 sensor 및 protocol 차이가 크다. Training은 PPG-BP Database의 2.1초 infrared PPG signal을 사용하고, test는 MAX86916으로 얻은 약 30초 multi-wavelength PPG signal을 사용한다. 모델이 실제로 multi-wavelength signal을 어떻게 training에서 학습했는지 명확하지 않다. PPG-BP Database는 infrared PPG만 포함한다고 설명되므로, training 단계에서 multi-wavelength ICA 기반 feature extraction을 동일하게 적용할 수 있었는지 불분명하다. 이 부분은 방법론의 중요한 재현성 문제이다.

둘째, test dataset 규모가 88명으로 작다. 남녀 수를 동일하게 구성하고 나이 범위를 넓혔다는 장점은 있지만, 혈압 추정 모델의 generalizability를 검증하기에는 제한적이다. 특히 hypertension, diabetes, vascular disease, arrhythmia 등 병리적 조건을 가진 subject가 충분히 포함되었는지 명확하지 않다. 논문은 healthy subjects 중심이므로 elderly pathological population에 대한 성능을 보장할 수 없다고 명시한다.

셋째, SBP 성능은 AAMI threshold를 만족하지 못한다. Test set에서 SBP MAE는 5.08 mmHg로 기준을 약간 초과하고, SD는 8.83 mmHg로 기준보다 더 높다. 이는 평균 오차뿐 아니라 개별 예측의 변동성이 임상 기준에는 아직 부족하다는 뜻이다. DBP는 기준을 만족하지만, 혈압 모니터링 장치로 사용하려면 SBP와 DBP 모두 안정적이어야 한다.

넷째, ANN architecture 설명이 충분히 구체적이지 않다. Hidden layer가 8개이고 neuron 수가 10–60개라고 되어 있지만, 각 layer의 최종 neuron 수, activation function, optimizer, learning rate, epoch 수, regularization, early stopping, loss function 등이 명시되지 않는다. Matlab 2023a에서 구현했다는 설명은 있지만, 재현 가능한 수준의 training detail은 부족하다.

다섯째, 21개 feature의 세부 정의가 제공되지 않는다. 15개 temporal feature와 6개 spectral feature를 사용했다고 설명하지만, 각 feature의 수식, cardiac cycle segmentation 방법, peak detection 방법, frequency band 경계, PSD 계산 방법이 추출 텍스트에 제시되지 않는다. Feature 기반 모델에서는 feature 정의가 핵심이므로, 이 정보가 부족하면 재현성과 해석 가능성이 제한된다.

여섯째, 평가 지표가 제한적이다. MAE와 SD가 제시되지만, mean error, Bland–Altman plot, BHS 기준의 5/10/15 mmHg threshold, correlation coefficient, calibration slope, subject-wise error distribution 등은 제시되지 않는다. 특히 혈압 추정에서는 평균 성능뿐 아니라 특정 혈압 범위, 나이 그룹, 성별, 고혈압 여부에 따른 오차 분석이 중요하다.

일곱째, 실제 continuous monitoring 성능은 아직 검증되지 않았다. Test measurement는 약 30초 동안 안정적인 자세에서 수행되며, sphygmomanometer 측정과 동시에 진행된다. 실제 wearable 환경에서는 사용자가 걷거나 움직이며 sensor pressure가 바뀌고, ambient light와 sweat, skin contact 변화가 발생한다. 이 논문은 motion artifact를 줄이기 위한 preprocessing을 제안하지만, 실제 daily-life motion condition에서의 robustness는 평가하지 않는다.

비판적으로 종합하면, 이 논문은 PPG-only, calibration-free, multi-wavelength sensing system을 제안하고 초기 test에서 promising한 성능을 보였지만, 아직 의료기기 수준의 검증에는 도달하지 않았다. 특히 training-test protocol mismatch, 작은 test dataset, feature 정의 부족, SBP SD 초과는 중요한 한계이다. 그럼에도 불구하고 device prototype과 새 test dataset을 결합했다는 점에서 실제 wearable 혈압 모니터링 시스템 개발의 초기 단계 연구로 의미가 있다.

## 6. 결론

이 논문은 MAX86916 multi-wavelength optical sensor module을 이용한 PPG-based electronic sensing system과 feed-forward ANN을 결합하여 cuff-less, calibration-free blood pressure estimation을 수행하는 방법을 제안했다. 제안 시스템은 red, infrared, green, blue 네 wavelength PPG signal을 획득하고, band-pass filtering, peak removal, ICA를 통해 motion artifact와 slow fluctuation을 줄인다. 이후 선택된 periodic independent component에서 15개 temporal feature와 6개 spectral feature를 추출하고, 21개 feature를 ANN에 입력하여 SBP와 DBP를 예측한다.

실험은 PPG-BP Database를 사용한 training 및 5-fold cross-validation 기반 hyperparameter tuning, 그리고 MAX86916 Evaluation Kit로 새롭게 수집한 88명 test dataset 평가로 구성된다. Test set 결과는 SBP에서 $5.08 \pm 8.83$ mmHg, DBP에서 $4.37 \pm 7.08$ mmHg이다. DBP는 논문이 언급한 AAMI 기준을 만족하지만, SBP는 MAE와 SD 모두 기준을 약간 초과한다. 따라서 결과는 유망하지만, clinical-grade device로 보기에는 추가 개선과 검증이 필요하다.

이 연구의 주요 기여는 PPG-only, ECG-free, calibration-free 혈압 추정을 위해 multi-wavelength PPG와 ICA 기반 artifact reduction, temporal-spectral feature 기반 ANN을 결합했다는 점이다. 또한 공개 training dataset과 별도 hardware로 수집한 test dataset을 함께 사용하여 prototype-level validation을 시도했다는 점도 의미가 있다.

향후 연구에서는 더 큰 규모의 multi-wavelength dataset 구축, training과 testing protocol의 일치, 병리적 subject와 elderly population 포함, feature 정의의 명확화, subject-wise validation, Bland–Altman 및 BHS/AAMI/ISO 기준 기반 평가, 실제 motion condition에서의 robustness 검증이 필요하다. 이러한 보완이 이루어진다면, 본 연구는 병원과 가정 환경에서 사용할 수 있는 non-invasive continuous blood pressure monitoring device 개발의 기초가 될 수 있다.
