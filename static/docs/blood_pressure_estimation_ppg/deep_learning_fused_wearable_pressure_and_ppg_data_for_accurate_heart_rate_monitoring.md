# Deep Learning Fused Wearable Pressure and PPG Data for Accurate Heart Rate Monitoring

* **저자**: Philip Mehrgardt, Matloob Khushi, Simon Poon, Anusha Withana
* **발표연도**: 2021

## 1. 논문 개요

이 논문은 웨어러블 환경에서 PPG를 이용한 심박수 측정의 정확도와 신뢰성을 높이기 위해, PPG, inertial sensor, pressure sensor를 결합한 다중 모달 손가락 클립 장치와 이를 위한 딥러닝 기반 신호 융합 방법을 제안한다. PPG는 피부에 빛을 조사하고 반사되거나 투과된 빛의 변화를 측정하여 혈류량 변화를 추정하는 비침습적 방법이다. 스마트워치, 피트니스 밴드, 의료용 산소포화도 측정기 등에서 널리 사용되지만, 실제 사용 환경에서는 motion artifact, 피부 접촉 상태 변화, 센서 위치, 광학적 잡음, 전기적 잡음 등으로 인해 파형이 쉽게 오염된다.

논문의 핵심 연구 문제는 단순히 평균 심박수만 맞추는 것이 아니라, 개별 heartbeat의 timing과 phase를 ECG 기준 R-peak에 가깝게 예측할 수 있는가이다. 이는 heart rate variability, blood pressure, pulse arrival time, pulse transit time, pulse wave velocity와 같은 더 정밀한 심혈관 지표를 계산하는 데 중요하다. 예를 들어 HRV는 heartbeat 간 시간 간격의 미세한 변화에 민감하고, PTT나 PWV는 신체의 서로 다른 위치에서 측정된 파형 간의 매우 작은 시간 차이를 요구한다. 논문에서는 손가락 클립처럼 길이가 약 4에서 5 cm 수준인 장치에서 pulse transit time이 약 6.4 ms 정도로 매우 짧을 수 있다고 설명한다. 따라서 PPG 기반 장치가 단순 심박수 측정을 넘어 혈압이나 HRV 추정에 사용되려면, 파형의 위치와 위상을 매우 정확하게 잡아야 한다.

이 논문의 목표는 두 가지로 정리할 수 있다. 첫째, PPG, accelerometer, gyroscope, pressure sensing을 포함하는 총 17채널의 다중 모달 finger clip sensing device를 설계하고 데이터를 수집하는 것이다. 둘째, ECG R-peak를 기준 정답으로 삼아 다중 채널 시계열 데이터를 융합하여 heartbeat timing과 heart rate를 정확히 예측하는 neural network를 개발하는 것이다. 저자들은 특히 PPG 센서의 부착 압력 변화, 즉 attachment pressure가 PPG 기반 심박 측정 정확도에 기여할 수 있음을 보였다고 주장한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 PPG 신호만으로는 실제 환경의 잡음과 피부 접촉 변화를 충분히 설명하기 어렵기 때문에, PPG 신호를 다른 센서 모달리티와 함께 해석해야 한다는 것이다. 기존 연구들은 주로 accelerometer나 gyroscope를 활용하여 motion artifact를 줄이는 데 초점을 맞추었다. 그러나 이 논문은 움직임 자체뿐 아니라 센서와 피부 사이의 접촉 상태 변화가 PPG 신호 품질에 큰 영향을 미친다는 점에 주목한다. PPG는 피부에서 반사되거나 투과되는 빛을 측정하므로, 센서가 피부에 얼마나 안정적으로 밀착되어 있는지, 압력이 어떻게 변하는지가 신호의 baseline과 waveform에 영향을 줄 수 있다.

이 논문에서 가장 중요한 차별점은 pressure sensor, 구체적으로 load cell을 PPG 기반 심박 측정의 하나의 입력 모달리티로 포함했다는 점이다. 저자들은 문헌상 PPG 기반 heart rate sensing에 pressure modality를 통합한 첫 사례라고 주장한다. 논문에서 제안한 장치는 손가락의 distal phalanx와 proximal phalanx 두 위치에 PPG 및 pressure sensor를 배치하고, 한 위치에는 IMU도 포함하여 motion 정보를 함께 수집한다. 이 구조를 통해 어떤 센서, 어떤 파장, 어떤 위치가 심박 측정에 더 기여하는지 비교할 수 있도록 설계했다.

또 다른 핵심 아이디어는 고정된 handcrafted peak detection 알고리즘 대신, 여러 센서의 time-domain signal 전체를 neural network에 입력하여 ECG R-peak 위치를 직접 회귀하도록 하는 것이다. 기존의 많은 알고리즘은 PPG peak, valley, 또는 특정 waveform feature를 먼저 찾고 이를 기반으로 심박수나 phase를 추정한다. 반면 이 논문은 하나의 window 안에 포함된 여러 채널의 전체 파형 모양을 이용하여 해당 window의 ECG R-peak 위치를 예측한다. 따라서 신호의 특정 peak만 신뢰하는 방식보다 잡음이 많은 상황에서 더 유연하게 정보를 활용할 수 있다.

## 3. 상세 방법 설명

논문의 전체 시스템은 크게 다중 모달 손가락 클립 하드웨어, 데이터 수집 시스템, ECG 기반 정답 라벨링, preprocessing, neural network 학습 및 평가 절차로 구성된다.

하드웨어 측면에서 저자들은 손가락에 부착되는 3D 프린트 finger clip prototype을 제작했다. 이 장치는 손가락의 두 지점에서 데이터를 측정한다. 논문에서는 distal phalanx, 즉 손가락 끝에 가까운 첫 번째 마디 부위를 position 0으로, proximal phalanx, 즉 손가락의 기저부 쪽 마디를 position 1로 정의한다. 두 PPG 센서 중심 사이의 거리는 40 mm이다. position 0에는 Maxim Integrated MAX30101 PPG sensor, TDK InvenSense MPU-9250 IMU, TAL221 100 g miniature load cell이 들어간다. position 1에는 MAX30101 PPG sensor와 TAL221 load cell이 들어간다.

PPG sensor는 세 파장의 빛을 사용한다. infrared는 약 880 nm, red는 약 660 nm, green은 약 537 nm이다. 논문은 green wavelength가 red나 infrared보다 피부 침투 깊이가 낮기 때문에, 더 깊은 조직이나 힘줄, 혈관에서 발생하는 motion artifact에 상대적으로 덜 민감할 수 있다고 설명한다. 반면 red와 infrared는 더 깊이 침투할 수 있어 신호 특성은 다르지만 motion artifact에 더 취약할 수 있다.

Pressure sensing에는 load cell이 사용되었다. Load cell은 strain gauge 원리를 사용하며, 힘에 따른 저항 변화를 Wheatstone bridge를 통해 전기 신호로 변환한다. 논문에서는 microcontroller의 ADC 해상도만으로는 load cell의 약한 신호에 quantization error가 발생할 수 있어, HX711 24-bit load cell amplifier를 각 load cell에 연결해 SNR을 높였다고 설명한다. Pressure sensor의 역할은 단순히 착용 압력을 측정하는 것이 아니라, 움직임 중 피부 접촉이 변하면서 발생하는 PPG signal corruption을 설명할 수 있는 보조 정보를 제공하는 것이다.

데이터 수집 장치는 Teensy 3.6 ARM Cortex-M4 microcontroller를 중심으로 구성되었다. PPG sensor는 I2C bus, IMU는 SPI, load cell amplifier는 TwoWire bus를 통해 연결되었으며, 가능한 병렬적으로 읽도록 설계되었다. PPG, accelerometer, gyroscope는 500 Hz로 측정되었고, pressure channel은 80 Hz로 측정되었다. 논문에서는 원문 초록에서 17 channels를 언급하며, PPG, accelerometer, gyro, pressure, temperature 관련 채널이 포함된다고 설명한다. ECG는 별도의 lead를 통해 측정되었고, 이 ECG R-peak가 심박의 기준 정답으로 사용되었다.

데이터는 22명의 건강한 참가자에게서 수집되었으나, 한 남성 참가자가 unusual properties를 보여 outlier로 제거되었다. 최종 분석에는 21명의 참가자가 사용되었다. 참가자들은 stationary, walking, running 세 가지 활동을 수행했다. 각 활동은 8분 동안 진행되었고, 총 504분 이상의 데이터와 39,000개 이상의 heartbeat가 수집되었다.

데이터 라벨링은 ECG R-peak를 기준으로 수행되었다. 저자들은 먼저 Engelse and Zeelenberg, Two Moving Average, Pan and Tompkins 알고리즘을 비교했고, 세 명의 참가자 데이터에서 Engelse and Zeelenberg 알고리즘이 manual labeling과 비교했을 때 true positive와 true negative의 균형이 가장 좋다고 판단했다. 이후 모든 ECG peak를 전용 ECG toolkit으로 수동 검토하여 false positive나 false negative를 수정했다. 이 절차는 논문에서 매우 중요한데, neural network의 목표값이 ECG R-peak이므로 라벨 품질이 전체 성능 평가의 신뢰도를 결정하기 때문이다.

Preprocessing에서는 PPG와 load cell 신호의 drifting DC baseline을 제거하기 위해 centred rolling Gaussian mean window를 사용했다. 이는 긴 8분 recording window 동안 신호 baseline이 움직이는 문제를 줄이기 위한 detrending 과정이다. 또한 PPG, load cell, IMU 데이터는 interquartile range를 기준으로 scaling하여 neural network 입력의 크기를 비교 가능한 수준으로 맞추었다. 논문은 비정상적이고 중심이 맞지 않은 time series가 neural network의 수렴을 어렵게 할 수 있다고 설명한다.

Neural network의 학습 목표는 입력 window에서 가장 이른 ECG R-peak의 위치를 회귀하는 것이다. 입력은 여러 센서 채널에서 얻은 정규화된 time series이며, 출력은 해당 window 안에서 ECG R-peak가 위치하는 시간 또는 sample 위치에 해당한다. 논문은 이를 $X$에서 $Y$로의 mapping을 학습하는 문제로 정의한다. 모델은 하나의 window 안에 포함된 모든 data point와 signal shape를 활용하며, 특정 PPG peak만을 기반으로 판단하지 않는다.

Window 길이는 $w = 1024$ samples로 설정되었고, sample rate는 500 Hz이다. 따라서 한 window의 시간 길이는 약 2.048초이다. Window shift는 50 samples, 즉 100 ms로 설정되었다. 이 설정은 accuracy와 runtime 사이의 절충으로 선택되었다. 저자들은 더 작은 shift가 accuracy를 높일 수 있지만 runtime이 증가한다고 설명한다. Window 길이와 sample rate를 기준으로 검출 가능한 최소 심박수는 다음과 같이 설명된다.

$$
HR_{min} = \frac{500 Hz}{1024 samples} \approx 0.48 Hz \approx 29.3 bpm
$$

이는 관찰 가능한 매우 낮은 human heart rate 수준에 해당하며, 저자들은 이 window 크기가 적어도 하나의 peak를 포함하도록 하는 실용적 기준이라고 설명한다.

Neural network가 예측한 ECG R-peak 위치를 $\hat{y}$라고 할 때, 연속된 두 예측 R-peak 사이의 시간 차이를 이용해 heart rate를 계산한다. 논문에서 사용한 변환은 다음과 같다.

$$
HR_{pred}[Hz] = \frac{1}{T} = \frac{1}{\hat{y}(n+1) - \hat{y}(n)}
$$

그리고 bpm 단위의 심박수는 다음과 같이 계산된다.

$$
HR_{pred}[bpm] = HR_{pred}[Hz] \times 60s
$$

여기서 $T$는 heartbeat period이고, $\hat{y}(n)$은 neural network가 예측한 $n$번째 ECG R-peak의 시간 위치이다. 쉬운 말로 설명하면, 모델이 각 심장 박동이 언제 발생했는지를 예측하고, 인접한 두 박동 사이의 간격을 이용해 1분당 박동 수를 계산하는 방식이다.

모델 구조는 input stage, 네 개의 hidden layer, output layer로 구성되었다고 설명되어 있다. 그러나 제공된 텍스트에는 각 layer의 정확한 neuron 수, activation function, optimizer, learning rate, batch size, epoch 수와 같은 세부 hyperparameter는 명확히 제시되어 있지 않다. 저자들은 CNN, RNN, LSTM, Transformer와 비교 실험을 해 보았으며, 제시한 neural network가 accuracy와 runtime의 균형이 좋았다고 설명하지만, 이 비교의 상세 수치는 제공된 텍스트에 포함되어 있지 않다.

Evaluation은 leave-one-out cross-validation 방식으로 수행되었다. 즉, 21명 중 한 명을 test participant로 남겨두고 나머지 참가자 데이터로 학습한 뒤, 남겨둔 참가자에서 성능을 평가한다. 이 과정을 모든 참가자에 대해 반복했다. 논문은 15개의 sensor 또는 sensor combination과 세 가지 activity에 대해 이 과정을 수행하여 총 945번 neural network를 학습 및 테스트했다고 설명한다. 이 방식은 새로운 사용자에게 모델이 어느 정도 일반화될 수 있는지 평가하기 위한 설계이다.

비교 대상으로는 HeartPy라는 기존 PPG heart rate analysis toolkit을 사용했다. HeartPy는 noisy PPG data 처리를 위한 알고리즘 기반 도구이다. 저자들은 grid search를 통해 HeartPy에 대해 0.75에서 5 Hz의 bandpass filter를 최적 설정으로 적용했다. 이를 통해 제안한 deep learning fused method와 기존 algorithmic method의 성능을 비교했다.

## 4. 실험 및 결과

실험은 21명의 건강한 참가자 데이터를 기반으로 stationary, walking, running 세 활동에서 수행되었다. Ground truth는 수동 검토된 ECG R-peak이다. 평가 지표는 크게 heartbeat timing prediction error와 heart rate prediction error로 나뉜다. Heartbeat prediction error는 ECG R-peak 기준으로 모델이 예측한 heartbeat 위치가 얼마나 차이 나는지를 ms 단위로 평가한다. Heart rate error는 bpm 단위의 average absolute error로 평가한다.

논문은 heartbeat prediction에 대해 mean, standard deviation, variance를 sensor channel 또는 sensor combination별로 제시한다. 또한 root mean square error도 비교한다. RMSE는 다음과 같이 정의된다.

$$
RMSE_{predictedHB} =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i - \hat{y}_i)
}
$$

다만 일반적인 RMSE 정의에서는 제곱항이 포함되는 것이 보통이지만, 제공된 논문 텍스트의 식에는 $(y_i - \hat{y}_i)$ 형태로 표시되어 있다. 원문 수식의 추출 과정에서 제곱 표기가 누락되었을 가능성도 있으나, 제공된 텍스트만으로는 이를 확정할 수 없다. 이 보고서에서는 논문에 제시된 식의 의도를 heartbeat 위치 예측 오차의 root mean square 측정으로 이해한다.

주요 결과는 제안한 다중 모달 deep learning 방법이 세 활동 모두에서 매우 낮은 average absolute error를 달성했다는 점이다. 전체 성능은 stationary에서 0.47 bpm, walking에서 0.79 bpm, running에서 0.89 bpm의 AAE를 보였다. 이는 기존 HeartPy 알고리즘 기반 방법보다 훨씬 낮은 오차이다. HeartPy의 AAE는 예시적으로 stationary에서 17.5 bpm, walking에서 53.0 bpm, running에서 38.9 bpm으로 보고되었다. 논문은 HeartPy가 noisy period에서 HR tracking을 잃고 peak를 과검출하여 큰 오차가 발생했다고 해석한다.

Heartbeat timing 측면에서도 제안 방법은 의미 있는 성능을 보였다. 논문 초록과 결과에서는 single heartbeat standard deviation이 stationary에서 28.43 ms, walking에서 40.3 ms, running에서 34.14 ms라고 제시된다. 이는 ECG R-peak와 비교한 개별 heartbeat timing error의 분산 정도를 나타낸다. 활동 중에는 walking이 running보다 오히려 더 어려운 조건으로 나타났는데, 논문은 accelerometer spectrogram과 ECG heart rate를 비교했을 때 walking motion artifact의 주파수가 실제 heart rate와 유사하게 나타날 수 있음을 보였다. 이런 경우 motion artifact가 heartbeat peak로 잘못 검출될 위험이 커진다.

센서 위치 측면에서는 position 0, 즉 distal phalanx 쪽이 position 1, 즉 proximal phalanx보다 전반적으로 더 좋은 SNR과 낮은 오차를 보였다. 원시 데이터의 line density plot에서는 position 1에서 더 큰 spread와 낮은 overlay가 관찰되었다. 저자들은 device를 180도 돌려 센서 위치를 바꾸어도 noise가 site 1을 따라갔다고 설명하므로, 차이가 센서 개체의 문제가 아니라 측정 위치의 특성과 관련된 것으로 해석한다.

파장별로는 green wavelength가 walking과 running에서 좋은 indicator로 나타났다고 논문은 설명한다. Green PPG는 red나 infrared보다 피부 침투 깊이가 낮아 motion artifact의 영향을 상대적으로 덜 받을 수 있다는 설명과 연결된다. 다만 stationary 상황에서는 position 0에서 green wavelength가 가장 나쁜 결과를 보였다고 논문은 언급한다. 따라서 특정 파장이 모든 조건에서 일관되게 최선이라고 보기는 어렵고, activity와 sensor location에 따라 기여도가 달라진다.

Pressure sensor의 기여는 논문의 핵심 결과 중 하나이다. 저자들은 PPG0G에 LC0를 추가했을 때 stationary 조건에서 heartbeat detection error의 standard deviation이 39.37 ms에서 31.34 ms로 줄어들었다고 보고한다. 이는 8.03 ms 감소이며, 비율로는 25.6% 개선이다. Walking에서는 42.95 ms에서 41.91 ms로, running에서는 35.86 ms에서 35.03 ms로 줄어들었다. 즉 pressure sensing의 개선 폭은 stationary에서 가장 컸고, walking과 running에서는 상대적으로 작았지만 여전히 개선이 있었다.

IMU의 기여에 대해서는 조금 더 미묘한 결과가 제시된다. 기존 연구는 PPG와 IMU를 결합하면 motion 상태에서 HR detection이 개선될 수 있음을 보였지만, 이 논문에서는 여러 위치와 여러 센서가 이미 융합된 경우 IMU의 추가적 이득이 줄어드는 것으로 관찰되었다. Stationary에서는 IMU를 포함하지 않은 모델의 RMSE가 28.43 ms로, IMU를 포함한 경우의 30.04 ms보다 낮았다. Walking에서는 LC, PPG, IMU를 함께 융합한 모델이 가장 낮은 RMSE인 40.3 ms를 보였고, running에서는 IMU를 제외한 모델이 0.55 ms 더 낮은 RMSE를 보였다. 따라서 IMU는 특히 walking과 같은 motion artifact 조건에서 유용할 수 있지만, 항상 성능을 향상시키는 것은 아니며 다른 센서 조합과 상황에 따라 달라진다.

True HR과 predicted HR의 density plot에서는 stationary, walking, running 모두에서 Pearson correlation이 0.977에서 0.988 사이로 보고되었다. Pearson $\rho$는 0.000으로 제시되었으며, 논문은 이를 통계적으로 유의한 상관관계로 해석한다. Regression line은 $HR_{true} \pm 3 bpm$ band 안에 유지되었다고 설명된다. 이는 예측 심박수가 전체적으로 ECG 기반 실제 심박수와 강하게 일치한다는 것을 의미한다.

정성적 결과로는 모든 참가자와 모든 활동에 대한 line plot에서 ECG 기반 true HR과 predicted HR이 대체로 잘 겹치는 모습이 제시된다. 다만 논문은 절대 심박수가 더 높은 영역에서 예측이 불안정해지는 경향이 있었다고 설명한다. 저자들은 그 원인으로 training data의 HR distribution imbalance를 지적한다. 즉 높은 HR 영역의 데이터가 충분하지 않아 neural network가 그 구간에서 충분히 잘 수렴하지 못했을 가능성이 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 pressure sensing을 PPG 기반 heart rate monitoring에 통합하고, 그 기여를 정량적으로 분석했다는 점이다. PPG 신호 품질은 센서와 피부의 접촉 상태에 크게 의존하지만, 기존 연구는 주로 accelerometer나 gyroscope를 이용한 motion artifact reduction에 집중했다. 이 논문은 피부 접촉 압력 자체를 하나의 관측 가능한 modality로 측정하고, neural network가 이를 PPG와 함께 학습하도록 설계했다. 특히 stationary 조건에서 pressure sensor 추가로 heartbeat detection standard deviation이 8.03 ms, 25.6% 개선된 결과는 이 설계의 실질적 가치를 보여준다.

두 번째 강점은 센서, 파장, 위치, 활동 조건을 나누어 체계적으로 비교했다는 것이다. 단순히 모든 센서를 결합한 최종 성능만 제시하는 것이 아니라, distal phalanx와 proximal phalanx, red, infrared, green PPG, load cell, IMU 등의 상대적 기여를 분석했다. 이러한 분석은 향후 smart ring이나 finger clip 형태의 웨어러블 장치를 설계할 때 어느 위치에 어떤 센서를 배치해야 하는지에 대한 실용적 근거를 제공한다.

세 번째 강점은 ECG R-peak를 수동 검토하여 ground truth 품질을 높였다는 점이다. ECG도 motion, sweat, body hair, clothes 등의 영향을 받아 artifact가 생길 수 있기 때문에, 자동 검출 결과를 그대로 쓰면 모델 학습과 평가 모두에 오류가 전파될 수 있다. 저자들이 모든 ECG peak를 전용 도구로 manual review했다는 점은 데이터 라벨의 신뢰성을 높이는 중요한 절차이다.

네 번째 강점은 leave-one-out cross-validation을 통해 participant-independent evaluation을 수행했다는 점이다. 같은 사람의 일부 데이터를 학습하고 같은 사람의 나머지 데이터를 테스트하는 방식은 성능을 과대평가할 위험이 있다. 반면 이 논문은 한 명의 참가자를 완전히 제외하고 학습한 뒤 그 참가자에게 테스트하는 절차를 반복했으므로, 새로운 사용자에 대한 일반화 가능성을 더 엄격하게 평가했다.

그러나 한계도 분명하다. 첫째, 참가자 수가 21명으로 기술적 feasibility study로는 의미가 있지만, 실제 population-level generalization을 주장하기에는 제한적이다. 논문도 gender와 age balance가 개선될 필요가 있다고 인정한다. 특히 피부색 다양성 측면에서 Fitzpatrick scale type IV 이상으로 추정되는 참가자는 두 명뿐이었다. PPG는 melanin에 의한 light absorption의 영향을 받을 수 있으므로, 다양한 피부색에서의 성능 검증이 필요하다.

둘째, 활동 조건이 실제 생활의 walking과 running을 완전히 대변하지 못할 수 있다. 논문에서는 walking과 running이 제자리 또는 약간의 앞뒤 움직임으로 수행되었다고 설명한다. 실제 야외 보행이나 달리기에서는 팔과 손의 움직임, 충격, 피부 접촉 변화, 땀, 온도 변화가 더 복잡할 수 있다. 따라서 실제 생활 환경에서 같은 성능이 유지되는지는 추가 검증이 필요하다.

셋째, 하드웨어가 아직 실사용 웨어러블 수준으로 완성된 형태는 아니다. 논문은 finger clip prototype이 다소 intrusive하게 보일 수 있다고 언급하고, 향후 smart ring form factor로 통합될 수 있다고 제안한다. 또한 load cell의 form factor가 크고, 장치의 질량과 관성으로 인해 motion artifact가 증폭되었을 가능성이 있다고 설명한다. 이는 pressure sensor가 실제 상용 웨어러블에 들어갈 때 더 작고 가벼운 형태로 재설계되어야 함을 의미한다.

넷째, neural network의 세부 구조와 학습 설정이 제공된 텍스트만으로는 충분히 상세하게 확인되지 않는다. 논문은 input stage, 네 개의 hidden layer, output layer로 구성되었다고 설명하지만, 각 layer의 크기, activation function, optimizer, learning rate, epoch 수 등 재현성에 중요한 정보는 제공된 추출 텍스트에 명확히 나타나지 않는다. 원문 전체의 그림이나 표에 더 자세한 정보가 있을 가능성은 있지만, 제공된 텍스트만으로는 단정할 수 없다.

다섯째, high HR 영역에서 prediction instability가 관찰되었다. 논문은 이를 training data의 HR distribution imbalance와 연결한다. 한 참가자의 HR이 다른 참가자들보다 평균 33.3 bpm 높아 LOOCV 분석에서 outlier로 제거되었는데, 이는 모델이 훈련 데이터에 없는 HR 범위에 일반화하기 어렵다는 점을 보여준다. 실제 응용에서는 고심박 운동, 심혈관 질환, 부정맥 등 더 넓은 HR 분포를 포함해야 하므로, 균형 잡힌 대규모 데이터셋이 필요하다.

비판적으로 보면, 이 논문은 pressure sensing이 PPG 기반 심박 추정에 유용하다는 설득력 있는 초기 증거를 제시하지만, 그 효과가 모든 조건에서 크지는 않다. 특히 walking과 running에서는 pressure 추가로 인한 standard deviation 감소가 각각 2.5%, 2.4% 수준으로 stationary보다 훨씬 작았다. 따라서 pressure modality의 가장 큰 가치는 움직임이 거의 없는 조건에서 skin contact variation을 보정하는 데 있을 수 있으며, 강한 motion condition에서는 장치 질량, 센서 부착 안정성, IMU와의 상호 보완 설계가 함께 중요해질 가능성이 있다.

## 6. 결론

이 논문은 PPG 기반 웨어러블 heart rate monitoring에서 pressure sensing을 새로운 보조 모달리티로 도입하고, 이를 PPG, accelerometer, gyroscope와 함께 deep learning model로 융합하여 ECG 기준 심박수와 heartbeat timing을 정확하게 예측할 수 있음을 보였다. 제안 시스템은 stationary에서 0.47 bpm, walking에서 0.79 bpm, running에서 0.89 bpm의 average absolute error를 달성했으며, 기존 HeartPy algorithmic toolkit보다 훨씬 낮은 오차를 보였다.

논문의 주요 기여는 다중 모달 finger clip sensing device의 설계, pressure modality를 포함한 PPG 기반 심박 측정, ECG R-peak 기준 neural network regression 방법, 그리고 센서 위치와 채널별 기여도 분석이다. 특히 load cell을 통해 attachment pressure 변화를 측정하고 이를 PPG와 함께 사용하면 heartbeat detection error가 줄어든다는 결과는 향후 웨어러블 센서 설계에서 중요한 시사점을 제공한다.

실제 적용 측면에서 이 연구는 단순 fitness tracking보다 더 정밀한 심혈관 모니터링으로 확장될 가능성이 있다. Heart rate variability, pulse transit time, pulse wave velocity, cuffless blood pressure estimation 같은 응용은 모두 정확한 waveform timing과 phase detection이 필요하기 때문이다. 다만 실제 제품이나 의료 응용으로 이어지려면 더 다양한 참가자, 더 현실적인 활동 조건, 더 작은 하드웨어, 더 넓은 heart rate distribution, 그리고 재현 가능한 학습 세부 설정이 필요하다. 그럼에도 이 논문은 PPG 신호를 단일 광학 신호로만 보지 않고, 피부 접촉 압력과 motion 정보를 함께 해석하는 방향이 wearable physiological sensing의 정확도를 높이는 중요한 연구 방향임을 잘 보여준다.
