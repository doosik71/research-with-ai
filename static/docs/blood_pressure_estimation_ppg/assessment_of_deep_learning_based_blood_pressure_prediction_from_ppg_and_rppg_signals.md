# Assessment of Deep Learning Based Blood Pressure Prediction from PPG and rPPG Signals

* **저자**: Fabian Schrumpf, Patrick Frenzel, Christoph Aust, Georg Osterhoff, Mirco Fuchs
* **발표연도**: 원문 발췌문에 명확히 표시되지 않음

## 1. 논문 개요

이 논문은 PPG 및 rPPG 신호를 이용한 deep learning 기반 blood pressure, 즉 혈압 예측 방법을 비판적으로 평가하는 연구이다. 논문의 목적은 새로운 최고 성능 모델을 제안하는 것이 아니라, 기존에 널리 쓰이는 neural network architecture들이 PPG 및 rPPG 기반 혈압 예측에서 실제로 얼마나 신뢰할 수 있는지, 그리고 reported mean error가 데이터 분포에 의해 얼마나 왜곡될 수 있는지를 분석하는 데 있다.

PPG는 finger clip sensor 등으로 쉽게 측정할 수 있는 optical pulse signal이고, rPPG는 일반 RGB camera로 얼굴이나 피부 영역의 미세한 색 변화에서 유도되는 remote photoplethysmography 신호이다. PPG와 rPPG는 모두 혈류량 변화와 관련된 waveform을 제공하므로, cuffless blood pressure estimation의 후보 신호로 주목받아 왔다. 특히 rPPG는 접촉 없이 camera만으로 측정할 수 있기 때문에, 임상 환경뿐 아니라 원격 진료, 비접촉 모니터링, 모바일 헬스케어에서 매력적이다.

하지만 저자들은 기존 연구들이 주로 전체 test set에 대한 평균 성능, 예를 들어 mean absolute error, 즉 MAE만 보고하는 관행을 문제로 지적한다. 혈압 데이터는 균등하게 분포하지 않고 특정 범위에 샘플이 몰려 있는 경우가 많다. 예를 들어 SBP가 100–120 mmHg, DBP가 56–60 mmHg 근처에 데이터가 많다면, 모델은 그 범위를 잘 예측하도록 학습될 수 있다. 이 경우 전체 평균 MAE는 낮아 보이지만, hypo- 또는 hypertensive range처럼 임상적으로 중요한 드문 혈압 범위에서는 error가 커질 수 있다.

따라서 이 논문의 연구 문제는 “PPG 또는 rPPG로 혈압을 예측할 수 있는가”를 넘어서, “딥러닝 기반 혈압 예측 모델이 혈압 분포 전체에서 균일하게 신뢰할 수 있는가”이다. 저자들은 이를 위해 PPG 입력 segment 구성 방식, window length, derivative 사용 여부, subject-based train/test split, BP bin별 error distribution, rPPG transfer learning, personalization 효과를 체계적으로 평가한다.

논문의 결론은 비교적 신중하다. PPG 기반 neural network는 전체 평균 성능에서는 mean regressor보다 일부 우수하지만, 효과 크기는 작고, 혈압 분포의 tail 영역에서는 error가 크게 증가한다. rPPG 기반 예측도 transfer learning으로 실험했지만, subject별 변동이 크고 mean regressor와 비교해 압도적으로 우수하다고 보기 어렵다. 저자들은 PPG 및 rPPG 기반 혈압 예측이 실제 임상 적용에 충분히 성숙했다고 보기 어렵고, 향후 연구에서는 전체 혈압 범위에 대한 세밀한 error analysis와 subject-independent evaluation이 필수적이라고 강조한다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 “평균 오차만으로 PPG/rPPG 기반 혈압 예측 모델을 평가하면 안 된다”는 것이다. 기존 연구들 중 상당수는 전체 test set에서 SBP와 DBP의 MAE가 낮다는 이유로 AAMI 또는 BHS 기준에 근접한다고 주장해 왔다. 그러나 혈압 데이터는 일반적으로 특정 정상 범위에 많이 몰려 있고, 저혈압 또는 고혈압 범위의 샘플은 상대적으로 적다. 이때 모델이 단순히 training set의 mode 근처를 예측하는 경향만 가져도 전체 평균 MAE는 낮아질 수 있다.

저자들은 이러한 문제를 확인하기 위해 mean regressor를 baseline으로 사용한다. Mean regressor는 training set의 평균 SBP와 평균 DBP만 항상 출력하는 매우 단순한 기준선이다. 만약 복잡한 neural network가 mean regressor보다 크게 우수하지 않다면, 그 모델이 PPG morphology에서 혈압 관련 정보를 충분히 학습했다고 보기 어렵다. 이 논문에서 AlexNet, ResNet, Slapničar et al. 모델은 일부 조건에서 mean regressor보다 낮은 MAE를 보였지만, 특히 DBP에서는 차이가 작았고, BP bin별 분석에서는 데이터가 많은 구간에서만 낮은 error를 보였다.

두 번째 핵심 아이디어는 PPG와 rPPG 모두에서 input segment 구성 방식이 성능에 영향을 준다는 점이다. 일반적으로 continuous PPG signal을 고정 시간 길이로 자르면 segment의 시작과 끝에서 heartbeat cycle이 잘릴 수 있다. 이는 waveform phase discontinuity를 만들고, neural network 학습에 불리할 수 있다. 저자들은 고정 시간 segment인 const time 방식과, 완전한 heartbeat cycle을 포함하도록 자르는 const HR 또는 const beats 방식을 비교했다. 그 결과 완전한 PPG cycle을 포함하도록 segment를 구성하는 방식이 유의하게 더 낮은 error를 보였다.

세 번째 핵심 아이디어는 rPPG 기반 혈압 예측을 PPG에서 학습한 network를 transfer learning하여 평가하는 것이다. rPPG 데이터는 수집하기 어렵고 noise가 많기 때문에 처음부터 deep network를 학습하기 어렵다. 저자들은 PPG 데이터로 pretraining된 모델의 대부분 layer를 freezing하고, 마지막 layer만 rPPG 데이터로 fine-tuning하였다. 이는 PPG와 rPPG waveform이 유사한 pulse morphology를 공유한다는 가정에 기반한다. 그러나 결과적으로 rPPG 기반 예측 성능은 subject별 차이가 컸고, 전체적으로 mean regressor와 매우 큰 차이를 보이지 않았다.

## 3. 상세 방법 설명

이 연구의 방법은 크게 PPG dataset 분석, neural network architecture 구성, PPG input segment parameterization 평가, PPG 기반 BP prediction, rPPG 기반 transfer learning, personalization 평가로 구성된다.

### 3.1 데이터셋 구성

논문은 두 종류의 PPG 데이터셋과 하나의 rPPG 데이터셋을 사용한다.

첫 번째 PPG 데이터셋은 MIMIC-A로 표기된다. 이는 Kaggle에 공개된 MIMIC-III subset으로, 12,000개의 record를 포함하며 PPG, ECG, ABP signal로 구성되어 있다. 원저자들이 이미 extensive preprocessing을 수행했기 때문에 compact하고 signal quality가 비교적 양호하다는 장점이 있다. 그러나 subject affiliation, 즉 각 record가 어떤 subject에 속하는지 알 수 없으므로 모델의 최종 성능 평가에는 적합하지 않다. 따라서 MIMIC-A는 input segment 길이와 cropping strategy를 결정하는 초기 실험에만 사용되었다.

두 번째 PPG 데이터셋은 MIMIC-B로 표기된다. 이는 Slapničar et al. 연구에서 제공한 script를 사용해 PhysioNet의 MIMIC-III database에서 다운로드한 더 큰 규모의 dataset이다. 총 4,000개의 PPG-ABP signal pair record로 구성되며, 성능 평가에 사용되었다. 중요한 점은 이 데이터셋에서는 subject-based split을 수행했다는 것이다. 즉 training, validation, test set에 동일 subject의 sample이 섞이지 않도록 나누었다. 이는 sample-level random split이 모델 성능을 과대평가하는 문제를 방지하기 위한 핵심 설계이다.

rPPG 데이터는 Leipzig University Hospital에서 임상 연구로 수집되었다. 연구는 University of Leipzig ethics committee의 승인을 받았고, subject들은 연구 내용을 고지받고 서면 동의했다. 총 50명의 수술 예정 환자가 등록되었으며, 수술 후 ICU로 이동한 뒤 face and upper body video가 촬영되었다. 촬영 장비는 industrial USB camera인 IDS UI-3040CP이며, frame rate는 32 fps이다. 각 영상은 약 2시간 길이이고, ground truth BP는 bedside monitor에서 1분 간격으로 기록되었다.

rPPG 데이터는 motion artifact, lighting 문제, 잦은 움직임이 있는 subject를 제외한 뒤 분석되었다. 최종적으로 14명의 subject가 rPPG 기반 fine-tuning 및 평가에 포함되었다.

### 3.2 Neural network architecture

저자들은 세 가지 neural network architecture를 사용했다.

첫 번째는 AlexNet 기반 모델이다. AlexNet은 원래 image classification용 CNN architecture이지만, 이 연구에서는 1D PPG time series를 입력으로 받아 SBP와 DBP를 출력하도록 수정되었다. 원래 classification layer는 두 개의 neuron을 갖는 regression layer로 교체되었고, linear activation function을 사용했다.

두 번째는 ResNet 기반 모델이다. ResNet은 deep CNN에서 발생하는 vanishing gradient 문제를 residual connection, 즉 skip connection으로 완화하는 구조이다. 이 연구에서는 ResNet을 1D time series regression에 맞게 수정하여 입력 PPG segment에서 SBP와 DBP를 예측하도록 했다. 입력은 univariate case에서는 $N_{samp} \times 1$이고, raw PPG와 1차 및 2차 derivative를 함께 사용하는 multivariate case에서는 $N_{samp} \times 3$이다.

세 번째는 Slapničar et al.이 제안한 spectrotemporal residual network이다. 이 모델은 PPG waveform과 그 1차 및 2차 derivative를 병렬적으로 처리하는 구조로, PPG 기반 혈압 예측 연구에서 이미 사용된 architecture이다. 저자들은 이 모델을 비교 대상으로 포함하여, 일반 CNN architecture인 AlexNet/ResNet과 PPG 특화 architecture 간 차이를 평가했다.

모든 모델은 SBP와 DBP 두 값을 동시에 예측하는 regression model로 사용되었다.

### 3.3 PPG signal processing 및 input parameterization

저자들은 먼저 PPG input segment를 어떻게 구성하는 것이 적절한지 평가했다. 두 가지 방식이 비교되었다.

첫 번째 방식은 const time 방식이다. PPG와 ABP signal을 1, 2, 5, 7, 9, 11, 13, 15, 17, 20초 길이의 고정 시간 segment로 나눈다. 이 방식은 구현이 간단하지만, segment boundary에서 heartbeat cycle이 잘릴 수 있다. 즉 하나의 PPG pulse가 segment 앞뒤에서 끊기며 phase discontinuity가 발생할 수 있다.

두 번째 방식은 const beats 또는 const HR 방식이다. 먼저 PPG signal에서 spectral component 중 amplitude가 가장 큰 주파수를 찾아 heart rate를 추정한다. 그다음 각 segment가 정수 개수의 complete PPG waves, 즉 완전한 heartbeat cycle을 포함하도록 자른다. 1, 2, 5, 7, 9, 11, 13, 15, 17, 20개의 PPG wave를 포함하는 segment가 생성된다. 각 segment는 동일 길이가 되도록 resampling된다. 이 방식은 absolute temporal information 일부를 제거하지만, pulse cycle을 중간에 끊지 않는다는 장점이 있다.

ABP signal도 같은 방식으로 처리하여 PPG-ABP pair를 구성한다. Ground truth SBP와 DBP는 ABP segment에서 peak detection algorithm을 사용하여 systolic peak와 diastolic peak를 검출하고, segment 내 peak들의 median으로 계산했다. 혈압값의 physiological plausibility check도 수행되었다. SBP는 75–165 mmHg, DBP는 40–80 mmHg 범위를 벗어나면 제외했고, heart rate도 50–140 bpm 범위를 벗어나면 제거했다.

저자들은 PPG derivative의 효과도 평가했다. 기존 연구들은 PPG의 1차 derivative와 2차 derivative가 cardiovascular state를 반영한다고 보고했기 때문에, raw PPG만 사용하는 univariate input과 raw PPG, first derivative, second derivative를 함께 사용하는 multivariate input을 비교했다. 그러나 실험 결과 derivative 사용은 일반적으로 성능을 개선하지 않았고, 이후 분석에서는 단순성을 위해 derivative를 사용하지 않았다.

### 3.4 PPG 기반 혈압 예측

MIMIC-B dataset은 앞 단계에서 결정한 최적 cropping strategy와 window length를 사용해 나누었다. MIMIC-B는 원본 PhysioNet 데이터에서 직접 얻은 것이므로 추가 preprocessing이 필요했다. 먼저 PPG signal에는 4th order Butterworth band-pass filter가 적용되었고, cutoff frequency는 0.5 Hz와 8 Hz로 설정되었다. 이후 각 signal window의 signal-to-noise ratio, 즉 SNR을 계산하고, SNR이 -7 dB 미만인 window는 제거했다. 모든 PPG window는 zero mean과 unit variance로 normalization되었다.

데이터셋은 subject-basis로 training, validation, test set으로 나뉘었다. Training에는 3,750명, validation과 test에는 각각 625명이 사용되었다. 이 subject들에서 training sample 1,000,000개, validation sample 250,000개, test sample 250,000개가 무작위로 추출되었다. 이처럼 subject-level split을 유지한 것은 논문의 중요한 방법론적 강점이다.

모델 학습은 TensorFlow 2.4와 Python 3.8로 구현되었으며, Adam optimizer, learning rate $\alpha = 0.001$, Euclidean loss, 60 epochs가 사용되었다. Validation set에서 MAE가 가장 낮은 모델을 최종 test에 사용했다.

손실 함수는 원문에서 Euclidean loss로 표현되며, 일반적으로 예측값과 정답값 사이의 제곱 오차 기반 거리로 이해할 수 있다. SBP와 DBP 예측값을 각각 $\hat{y}*{SBP}$, $\hat{y}*{DBP}$라 하고, 정답을 $y_{SBP}$, $y_{DBP}$라 하면 loss는 다음과 같은 형태로 해석할 수 있다.

$$
\mathcal{L} = \sqrt{(\hat{y}*{SBP}-y*{SBP})^2 + (\hat{y}*{DBP}-y*{DBP})^2}
$$

또는 구현에 따라 평균 제곱 오차와 유사한 형태일 수 있다. 원문은 정확한 수식은 제시하지 않았으므로, 여기서는 Euclidean loss의 개념적 의미만 설명하는 것이 적절하다.

평가에는 MAE가 사용되었다.

$$
MAE = \frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i-y_i|
$$

여기서 $\hat{y}_i$는 예측 혈압값이고, $y_i$는 reference 혈압값이다. 전체 dataset 평균 MAE뿐 아니라, 저자들은 혈압 범위를 10 mmHg 폭의 bin으로 나누어 각 bin별 MAE를 계산했다. 이 bin-wise analysis가 논문의 핵심 평가 방법이다.

### 3.5 rPPG 기반 혈압 예측 및 transfer learning

rPPG 분석에서는 먼저 subject의 forehead와 cheeks 영역을 수동으로 labeling했다. 이후 Plane-Orthogonal-to-Skin, 즉 POS algorithm을 사용하여 skin pixel에서 pulse wave를 추출했다. POS는 RGB signal에서 motion 및 illumination variation을 줄이고 blood volume pulse 성분을 추출하는 rPPG 알고리즘이다.

추출된 rPPG signal은 visual inspection을 거쳤다. 심한 motion artifact, frequent movement, insufficient lighting이 있는 25명의 subject는 분석에서 제외되었다. 남은 데이터는 heart rate 기반 window로 나뉘었으며, 각 window에는 7개의 heartbeat가 포함되었다. MIMIC-B 처리와 동일하게 resampling을 수행하고, SNR이 -7 dB 미만인 window는 제외했다. Ground truth BP는 bedside monitor에서 얻은 값을 사용했다.

rPPG 데이터는 양이 적기 때문에 neural network를 처음부터 학습하지 않고, PPG 데이터로 pretraining된 network를 fine-tuning했다. 이때 대부분의 network weight는 freezing하고 final layer만 학습했다. 이는 PPG와 rPPG waveform이 pulse morphology 측면에서 유사하므로, PPG에서 학습한 feature extractor를 rPPG에 재사용할 수 있다는 transfer learning 가정에 기반한다.

Fine-tuning은 leave-one-out cross-validation 방식으로 수행되었다. 14명 중 12명은 training, 1명은 validation, 1명은 testing에 사용되었다. 이후 test subject에 대한 성능을 평가했다.

### 3.6 Personalization

저자들은 personalization 효과도 평가했다. Personalization은 test subject의 일부 데이터를 training에 추가하여 subject-specific adjustment를 수행하는 방식이다. 구체적으로 test subject 데이터의 20%를 training에 포함하고, 나머지 80%를 validation/test에 사용했다. 이는 Slapničar et al.에서 사용한 subject-specific fine-tuning 전략과 유사하다.

Personalization은 실제 혈압 예측에서 중요한 의미를 가진다. PPG morphology는 나이, 혈관 상태, 질환, 약물, sensor contact pressure, 피부 특성 등 subject-specific factor에 크게 영향을 받는다. 따라서 population-level model이 모든 subject에 잘 일반화되기 어렵고, 일부 subject-specific calibration 또는 fine-tuning이 성능을 개선할 가능성이 있다.

## 4. 실험 및 결과

### 4.1 Input segment 구성 결과

첫 번째 실험에서는 PPG segment를 const time 방식으로 자를 때와 const HR 또는 const beats 방식으로 자를 때의 성능을 비교했다. Figure 2에 따르면, complete heartbeat cycle을 포함하도록 segment를 구성한 const HR 방식이 const time 방식보다 더 낮은 prediction error를 보였다. Paired t-test 결과 이 차이는 통계적으로 유의했다($p < 0.01$).

이는 PPG cycle이 segment boundary에서 잘리는 것이 neural network 학습에 불리하게 작용한다는 점을 보여준다. 혈압 예측에서 PPG waveform morphology는 pulse onset, systolic upstroke, peak, diastolic decay 등의 구조에 의존하는데, cycle이 중간에 잘리면 이러한 구조가 왜곡된다. 따라서 단순히 고정 시간 window를 사용하는 것보다 complete pulse cycle을 포함하도록 구성하는 것이 더 타당하다.

Derivative 사용은 전반적으로 성능 향상을 주지 않았다. AlexNet의 SBP MAE에서만 약간의 개선이 있었지만, 일반적인 improvement는 관찰되지 않았다. 따라서 저자들은 이후 분석에서 derivative를 제외하고 raw PPG 기반 univariate input을 사용했다.

Window length의 경우, PPG 기반 prediction error는 segment length에 따라 큰 차이를 보이지 않았다. 저자들은 원래 긴 segment일수록 여러 heartbeat morphology가 섞여 ambiguity가 증가할 것으로 예상했지만, empirical analysis에서는 이 효과가 명확히 확인되지 않았다. 다만 충분한 반복 학습 횟수가 없어 통계적 정당화는 제공하지 못한다고 명시했다.

rPPG에서는 segment length가 SNR과 사용 가능한 sample 수 사이의 trade-off를 만든다. 너무 짧은 segment는 SNR 계산이 불안정하고, 너무 긴 segment는 사용 가능한 sample 수가 줄어든다. Figure 3의 SNR 분석을 바탕으로 저자들은 이후 모든 분석에 7초 segment length를 선택했다.

### 4.2 PPG 기반 BP prediction 결과

MIMIC-B dataset에서 AlexNet, ResNet, Slapničar et al. 모델을 학습한 결과는 Figure 4에 제시된다. 전체적으로 DBP MAE가 SBP MAE보다 낮았다. 이는 DBP 값의 분포 범위가 SBP보다 좁기 때문에 상대적으로 예측이 쉬운 것으로 해석된다.

모델들은 mean regressor와 비교되었다. AlexNet과 ResNet은 SBP MAE에서 mean regressor보다 유의하게 낮은 error를 보였다. DBP에서도 AlexNet과 ResNet은 mean regressor보다 유의하게 낮은 error를 보였지만, 그 차이는 작았다. Slapničar et al. 모델은 DBP에서는 mean regressor보다 유의하게 낮은 성능을 보이지 못했다.

이 결과는 중요한 시사점을 가진다. 복잡한 deep neural network가 단순히 training set 평균값만 출력하는 mean regressor보다 조금 나은 정도라면, 모델이 PPG signal에서 혈압에 대한 강한 일반화 가능한 정보를 학습했다고 단정하기 어렵다. 특히 subject-based split을 적용했을 때 성능 향상이 제한적이라는 점은, 기존 연구에서 보고된 매우 낮은 MAE가 sample-level split 또는 subject leakage에 의해 과대평가되었을 가능성을 시사한다.

### 4.3 혈압 bin별 error distribution

이 논문의 가장 중요한 실험 결과는 Figure 5의 BP bin별 MAE 분석이다. 저자들은 ground truth BP 범위를 10 mmHg 폭의 bin으로 나누고, 각 bin에서 MAE를 별도로 계산했다. Figure 5에는 AlexNet 결과만 제시되지만, 저자들은 세 architecture에서 유사한 패턴이 나타났다고 설명한다.

결과적으로 error는 BP bin에 따라 크게 달라졌다. SBP의 경우 100–120 mmHg 범위에서, DBP의 경우 56–60 mmHg 범위에서 가장 낮은 MAE를 보였다. 이 구간은 test set에서 sample 수가 가장 많은 구간이다. 반대로 sample 수가 적은 낮은 혈압 또는 높은 혈압 bin에서는 error가 크게 증가했다.

이는 모델이 training distribution의 mode 근처를 예측하도록 학습된다는 것을 보여준다. Neural network는 전체 loss를 줄이는 방향으로 학습되기 때문에, 가장 많은 sample이 존재하는 구간에서 error를 줄이는 것이 전체 MAE 감소에 가장 효과적이다. 따라서 데이터가 적은 hypo- 또는 hypertensive range에서는 상대적으로 큰 error가 발생할 수 있다.

임상적으로 이는 매우 중요한 문제이다. 혈압 예측 모델은 정상 범위에서 몇 mmHg 정확한 것보다, 저혈압이나 고혈압처럼 intervention이 필요한 구간에서 안정적으로 동작해야 한다. 이 논문은 기존의 평균 MAE 중심 평가가 이러한 위험을 숨길 수 있음을 명확히 보여준다.

### 4.4 AAMI 및 BHS 기준에 대한 평가

저자들은 자신들의 mean performance를 기존 연구와 비교하기 위해 전체 평균 성능도 평가했다. 그러나 논문은 어떤 모델도 BHS 및 AAMI 기준을 만족하지 못했다고 명시한다. 관련 기준에서는 혈압 측정 장치가 허용 가능한 오차 범위, 예를 들어 $BP \leq 10$ mmHg error를 충분히 높은 확률로 만족해야 하며, BHS 기준에서는 10 mmHg 이하 error 비율이 85%를 넘어야 한다.

이 논문에서 관찰된 high MAE, 특히 낮은 혈압 및 높은 혈압 범위에서의 큰 error는 임상 적용에 문제가 된다. 저자들은 이 결과가 Slapničar et al.의 subject-specific split을 고려한 연구 결과와 일치한다고 설명한다. 반면 subject-based split을 명시하지 않은 일부 기존 연구는 훨씬 낮은 prediction error를 보고했는데, 저자들은 이것이 methodological improvement라기보다 data selection 또는 train/test contamination의 영향일 수 있다고 해석한다.

### 4.5 rPPG 기반 BP prediction 결과

rPPG 실험에서는 PPG로 pretraining된 neural network를 rPPG data로 fine-tuning했다. Table 1의 no personalization 결과에 따르면, 전체 MAE는 다음과 같다.

AlexNet은 SBP MAE 15.7 mmHg, DBP MAE 8.27 mmHg를 보였다. ResNet은 SBP MAE 13.02 mmHg, DBP MAE 9.81 mmHg를 보였다. Slapničar et al. 모델은 SBP MAE 14.04 mmHg, DBP MAE 8.64 mmHg였다. Mean regressor는 SBP MAE 13.4 mmHg, DBP MAE 8.9 mmHg였다.

이 결과에서 ResNet은 SBP에서 가장 낮은 MAE를 보였고, AlexNet은 DBP에서 가장 낮은 MAE를 보였다. 그러나 전체적으로 neural network들이 mean regressor를 압도적으로 능가하지는 않았다. Figure 6의 blue boxplot도 subject별 MAE 변동이 매우 크다는 점을 보여준다. 저자들은 세 neural network의 전체 성능이 서로 유사하고, mean regressor와도 비교적 가깝다고 설명한다.

rPPG 기반 혈압 예측이 어려운 이유는 명확하다. rPPG는 PPG보다 SNR이 낮고, motion, illumination change, skin tone, ROI selection, camera noise, compression artifact 등의 영향을 크게 받는다. 따라서 waveform morphology를 기반으로 혈압을 추론하는 모델은 rPPG 환경에서 더 불안정해질 수 있다.

### 4.6 Personalization 결과

Personalization을 적용했을 때 일부 모델의 error는 약간 감소했다. Table 1의 personalization 결과에 따르면, AlexNet의 SBP MAE는 15.7에서 15.2 mmHg로 0.5 mmHg 감소했다. ResNet의 SBP MAE는 13.02에서 12.51 mmHg로 0.51 mmHg 감소했고, DBP MAE는 9.81에서 8.3 mmHg로 1.29 mmHg 감소했다. Slapničar et al. 모델의 SBP MAE도 14.04에서 13.56 mmHg로 0.48 mmHg 감소했다.

그러나 personalization이 항상 강력한 개선을 제공한 것은 아니다. Mean regressor 역시 training/test split 변화에 따라 SBP MAE가 13.4에서 13.9 mmHg로 증가하고, DBP MAE가 8.9에서 8.5 mmHg로 감소했다. 따라서 personalization의 개선 효과는 존재하지만 제한적이며, 데이터 분할 변화의 영향과 구분해서 해석해야 한다.

저자들은 rPPG 데이터가 매우 제한적이었기 때문에 마지막 layer만 fine-tuning할 수 있었다고 설명한다. 더 많은 rPPG 데이터가 있다면 additional layer까지 fine-tuning하여 성능 향상을 기대할 수 있지만, 본 논문에서는 이를 검증하지 못했다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 PPG/rPPG 기반 혈압 예측 연구에서 흔히 간과되는 evaluation bias를 정면으로 다룬다는 점이다. 기존 논문들은 전체 평균 MAE만 보고하면서 모델 성능이 우수하다고 주장하는 경우가 많다. 그러나 이 연구는 BP bin별 error를 분석하여, 모델이 실제로는 데이터가 많은 정상 범위에 편향되어 있고 드문 혈압 범위에서는 error가 크게 증가한다는 점을 보여준다. 이는 임상 적용 가능성을 평가하는 데 매우 중요한 관점이다.

두 번째 강점은 subject-based split을 명확히 적용했다는 점이다. PPG signal은 subject-specific morphology가 강하기 때문에 같은 subject의 sample이 train과 test에 동시에 포함되면 모델 성능이 크게 과대평가될 수 있다. 저자들은 MIMIC-B에서 3,750명 training, 625명 validation, 625명 test로 subject를 나누어 contamination을 방지했다. 이는 PPG 기반 혈압 예측 연구에서 반드시 지켜야 할 평가 원칙이다.

세 번째 강점은 mean regressor baseline을 사용했다는 것이다. 복잡한 neural network가 단순 평균 예측보다 얼마나 우수한지를 확인하는 것은, 모델이 실제 생리학적 정보를 학습했는지 판단하는 기본적인 검증이다. 이 논문은 일부 neural network의 성능이 mean regressor와 크게 다르지 않음을 보여주며, 기존 연구의 과도한 성능 주장에 대한 중요한 반례를 제시한다.

네 번째 강점은 PPG와 rPPG를 함께 다룬다는 점이다. rPPG 기반 혈압 예측은 매우 매력적인 응용 분야이지만, 아직 신호 품질과 일반화 문제에 대한 의문이 크다. 저자들은 PPG로 pretraining한 network를 rPPG에 transfer하는 실험을 통해, rPPG 기반 혈압 추정의 현실적인 어려움을 보여준다.

하지만 한계도 분명하다. 첫째, 이 연구는 state-of-the-art model을 찾기 위한 hyperparameter tuning을 적극적으로 수행하지 않았다. 저자들도 자신들의 목적이 최고 성능 달성이 아니라 평가 방법의 문제를 분석하는 것이라고 명시한다. 따라서 여기서 사용된 AlexNet, ResNet, Slapničar et al. 모델의 성능이 PPG 기반 혈압 예측의 이론적 한계를 의미하지는 않는다.

둘째, rPPG 데이터 규모가 매우 작다. 초기 50명 subject 중 motion artifact, movement, lighting 문제로 많은 subject가 제외되었고, 최종 분석에는 14명만 사용되었다. 이 데이터로는 rPPG 기반 혈압 예측의 일반화 성능을 확정적으로 평가하기 어렵다. 특히 마지막 layer만 fine-tuning한 결과이므로, 충분한 rPPG 데이터가 있을 때 end-to-end fine-tuning 또는 rPPG-specific model이 더 나은 결과를 보일 가능성은 남아 있다.

셋째, rPPG ground truth BP는 bedside monitor에서 1분 temporal resolution으로 수집되었다. 반면 rPPG signal은 continuous video에서 window 단위로 추출된다. 혈압의 시간 변동과 ground truth temporal resolution mismatch가 예측 성능에 영향을 줄 수 있지만, 원문 발췌문에서는 이 문제를 상세히 분석하지 않는다.

넷째, MIMIC 기반 데이터는 ICU 환경에서 수집된 것이므로 sensor contact pressure, 장비 종류, patient condition, medication, comorbidity 등이 다양하다. 이는 일반화를 평가하는 데 장점이지만, 동시에 PPG morphology와 BP의 관계를 매우 복잡하게 만든다. 논문은 이러한 다양성이 모델 일반화를 방해할 수 있다고 해석하지만, 구체적으로 어떤 factor가 가장 큰 영향을 미치는지는 분석하지 않는다.

다섯째, bin-wise error가 증가하는 원인을 data imbalance로 주로 설명하지만, 각 bin에서 subject composition, disease severity, age, medication, measurement equipment 차이가 함께 작용했을 수 있다. 즉 단순히 sample 수가 적어서 error가 커진 것인지, 해당 혈압 범위의 생리적 특성이 더 복잡해서 error가 커진 것인지는 추가 분석이 필요하다.

## 6. 결론

이 논문은 PPG 및 rPPG 기반 deep learning 혈압 예측 연구에 대해 매우 중요한 평가적 관점을 제시한다. 저자들은 AlexNet, ResNet, Slapničar et al. 모델을 사용하여 PPG 기반 SBP/DBP 예측을 수행했고, rPPG 데이터에 대해서는 transfer learning과 personalization을 평가했다. 그 결과, complete heartbeat cycle을 포함하도록 PPG segment를 구성하는 것이 성능에 유리하며, derivative 사용은 일반적 개선을 제공하지 않는다는 점을 확인했다.

가장 중요한 결론은 neural network의 전체 평균 MAE만으로 혈압 예측 성능을 평가해서는 안 된다는 점이다. 모델은 데이터가 많은 정상 혈압 범위에서는 낮은 error를 보였지만, 낮은 혈압 및 높은 혈압 범위에서는 error가 크게 증가했다. 이는 임상적으로 중요한 hypo- 및 hypertensive range에서 모델이 신뢰하기 어렵다는 의미이다. 또한 복잡한 neural network가 mean regressor보다 크게 우수하지 않은 경우가 있었고, 이는 PPG morphology에서 일반화 가능한 혈압 정보를 학습하는 일이 생각보다 어렵다는 것을 시사한다.

rPPG 기반 혈압 예측은 더욱 도전적이었다. PPG로 pretraining한 network를 rPPG에 fine-tuning했지만, subject별 성능 변동이 컸고, 전체 성능은 mean regressor와 크게 차별화되지 않았다. Personalization은 일부 모델에서 error를 약간 줄였지만, 개선 폭은 제한적이었다.

이 연구의 실질적 기여는 새로운 모델 자체보다 평가 기준의 재정립에 있다. 저자들은 PPG/rPPG 기반 혈압 예측 연구가 임상적으로 의미 있으려면, 반드시 subject-independent split을 사용하고, 전체 혈압 범위에 대해 bin-wise error를 보고해야 하며, mean regressor와 같은 강력한 단순 baseline과 비교해야 한다고 주장한다. 또한 데이터 분포, subject 특성, sensor contact pressure, motion artifact, illumination, skin tone 등 실제 환경 변수를 더 면밀히 다루어야 한다.

종합하면, 이 논문은 PPG와 rPPG 기반 cuffless BP estimation의 가능성을 완전히 부정하지는 않지만, 현재의 deep learning 기반 접근들이 평균 성능만으로는 실제 임상 적용 가능성을 충분히 입증하지 못한다고 결론짓는다. 향후 연구는 더 균형 잡힌 혈압 분포, 대규모 subject-independent dataset, 혈압 범위별 성능 보고, domain-invariant feature learning, 그리고 현실적인 rPPG artifact 환경에서의 검증을 중심으로 진행되어야 한다.
