# The Machine Learnings Leading the Cuffless PPG Blood Pressure Sensors Into the Next Stage

* **저자**: Paul C.-P. Chao, Chih-Cheng Wu, Duc Huy Nguyen, Ba-Sy Nguyen, Pin-Chia Huang, Van-Hung Le
* **발표연도**: 2021

## 1. 논문 개요

이 논문은 cuffless blood pressure sensor, 특히 PPG waveform을 기반으로 혈압을 추정하는 machine learning 및 deep learning 연구들을 정리한 review 논문이다. 논문은 기존 pulse wave velocity, 즉 PWV 이론에 기반한 PTT 또는 PAT 중심 접근에서 출발하여, hand-crafted PPG features를 사용하는 intelligent regression 방법, 그리고 최근 raw PPG waveform에서 feature extraction과 BP regression을 통합적으로 수행하는 deep learning 방법으로 연구 흐름이 발전해 왔다고 설명한다.

연구의 중심 문제는 cuff 없이, 비침습적으로, 장시간 연속 혈압을 정확하게 추정할 수 있는가이다. 기존 cuff sphygmomanometer는 측정 정확도와 임상적 신뢰도 측면에서 널리 쓰이지만, cuff가 크고 무겁고 사용자가 불편하며 장시간 연속 측정에 적합하지 않다. 반면 cuffless PPG sensor는 wearable device 형태로 구현할 수 있고, 장기간 데이터를 수집할 수 있다는 장점이 있다. 그러나 PPG waveform은 개인별 혈관 특성, 피부와 조직의 optical property, 측정 위치, motion artifact, device design, calibration 상태에 크게 영향을 받는다. 따라서 단순한 PTT 또는 PAT만으로 혈압을 충분히 정확하게 설명하기 어렵다.

논문은 크게 두 가지 기술적 흐름을 비교한다. 첫 번째는 PTT, PAT, PPG morphology, second derivative PPG 등 미리 정의된 physiological 또는 non-physiological features를 추출한 뒤, MLR, SVM, regression tree, random forest, AdaBoost, ANN 같은 regression model을 적용하는 방식이다. 두 번째는 CNN, RNN, LSTM, Bi-LSTM, CNN-LSTM, U-Net 등 deep learning 모델을 사용하여 raw PPG 또는 ECG/PPG data에서 feature extraction과 BP estimation을 하나의 학습 시스템으로 처리하는 방식이다.

이 논문의 목표는 특정 새로운 모델을 제안하는 것이 아니라, 최근 cuffless PPG BP estimation 연구의 기술적 진화 방향을 정리하고, 향후 상용화를 위해 해결해야 할 문제를 도출하는 것이다. 저자들은 deep learning이 hand-crafted feature의 한계를 완화하고 단일 PPG sensor 기반 혈압 추정 정확도를 향상시킬 수 있는 기회를 제공한다고 평가한다. 그러나 동시에 실제 상용화를 위해서는 대규모 다기관 데이터 수집, real-time signal quality screening, lightweight model, motion artifact robustness, calibration 및 개인 상태 변화 문제를 해결해야 한다고 강조한다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 cuffless PPG 기반 혈압 추정 기술이 세 단계로 발전하고 있다는 해석이다. 첫 단계는 PWV theory 기반 접근이다. 이 방식은 ECG와 PPG를 동시에 측정하여 pulse transit time, pulse arrival time 같은 시간 지표를 추출하고, 이를 혈압과 연결한다. 그러나 PTT와 PAT는 혈압과 관련이 있지만, 실제 cardiovascular system의 복잡성을 충분히 설명하지 못한다.

두 번째 단계는 PPG waveform에서 더 많은 pre-determined features를 추출하고, machine learning regression을 적용하는 방식이다. 예를 들어 systolic rising time, pulse area, dicrotic notch 관련 시간, PPG의 second derivative에서 얻는 arterial stiffness 관련 feature 등을 사용한다. 이 방식은 PTT/PAT만 사용하는 것보다 더 많은 정보를 활용하지만, feature extraction algorithm이 waveform morphology에 의존한다는 문제가 있다. 특히 고령자나 CVD 환자에서는 dicrotic notch가 사라지거나 불명확할 수 있어, 미리 정의한 feature를 안정적으로 추출하기 어렵다.

세 번째 단계는 deep learning 기반 접근이다. Deep learning은 raw PPG waveform 또는 ECG/PPG sequence를 직접 입력으로 받아, feature extraction과 BP estimation을 동시에 학습한다. 저자들은 이 방향이 hand-crafted feature의 불충분성, 개인별 waveform variation, feature extraction algorithm의 오류 가능성을 줄일 수 있다고 본다. CNN은 waveform의 local pattern을 추출하고, RNN이나 LSTM은 시간적 변화와 long-term dependency를 모델링하며, U-Net이나 encoder-decoder 구조는 waveform-to-waveform estimation도 가능하게 한다.

이 논문의 중요한 비판적 관점은 PTT/PAT 기반 접근만으로는 cuffless BP sensor의 상용화가 어렵다는 점이다. PTT와 PAT는 ECG R-peak와 PPG rising edge 사이의 시간 차이를 이용하지만, 혈압은 arterial stiffness, vascular tone, blood volume, peripheral resistance, medication, stress, posture, sensor contact 등 다양한 요인에 의해 결정된다. 따라서 단일 시간 지표로 혈압을 안정적으로 설명하기 어렵고, 개인별 calibration이 자주 필요하다.

또한 이 논문은 deep learning이 단순히 더 복잡한 모델이라는 이유로 유리한 것이 아니라, feature extraction 자체를 데이터 기반으로 학습한다는 점에서 중요하다고 본다. 즉, PPG waveform 안에 포함된 BP-related information을 사람이 미리 정한 feature로 제한하지 않고, 모델이 필요한 representation을 학습할 수 있다는 것이다. 다만 저자들은 deep learning이 실제 상용화를 자동으로 보장하지는 않으며, 데이터 품질, 모델 크기, real-time preprocessing, 다양한 인구와 환경에서의 검증이 여전히 핵심 문제라고 지적한다.

## 3. 상세 방법 설명

이 논문은 review 논문이므로 자체적인 단일 training pipeline이나 새로운 loss function을 제안하지 않는다. 대신 cuffless PPG BP estimation에서 사용되어 온 주요 방법론을 이론적 배경과 연구 흐름에 따라 정리한다.

전통적 pipeline은 일반적으로 signal preprocessing, feature extraction, regression 또는 learning, BP estimation으로 구성된다. Preprocessing 단계에서는 ECG와 PPG signal에 filtering, segmentation, outlier removal, normalization을 적용한다. PPG signal은 baseline drift와 high-frequency noise가 많기 때문에, 많은 연구는 4차 이상의 high-order band-pass filter를 사용한다고 설명된다. Baseline drift는 주로 breathing이나 body movement 같은 motion artifact에서 발생한다. Static electricity 등으로 생기는 impulsive noise는 spline function 같은 방법으로 제거될 수 있다. 이후 ECG/PPG waveform은 보통 약 10초 segment 단위로 나뉘어 feature extraction과 model training에 사용된다.

PWV 기반 혈압 추정은 Bramwell-Hill equation에서 출발한다. 논문은 PWV를 다음과 같이 정의한다.

$$
PWV = \sqrt{ \frac{V}{\rho} \frac{\Delta P}{\Delta V} }
$$

여기서 $\rho$는 blood density, $V$는 artery 내부 blood volume, $\Delta P$는 SBP와 DBP의 차이, 즉 pulse pressure에 해당하며, $\Delta V$는 이에 대응하는 blood volume change이다. 개인 내부에서 $\rho$, $V$, $\Delta V$가 거의 일정하다고 가정하면, SBP와 DBP의 차이는 PWV의 제곱에 비례한다고 볼 수 있다.

Pulse wave velocity는 propagation distance $L$을 pulse transit time $PTT$로 나눈 값으로 표현할 수 있으므로, 논문은 다음 관계를 제시한다.

$$
SBP - DBP = \rho \frac{\Delta V}{V} (PWV)^2 = \rho \frac{\Delta V}{V} \left( \frac{L}{PTT} \right)^2 = K_a \cdot \frac{1}{PTT^2}
$$

여기서 $K_a$는 실험적으로 calibration해야 하는 subject-specific parameter이다. 이 식의 직관은 PTT가 짧을수록 pulse wave가 빠르게 전달된다는 뜻이고, 이는 일반적으로 arterial stiffness 또는 pressure 증가와 관련될 수 있다는 것이다.

이 관계를 사용하면 SBP는 다음과 같이 표현된다.

$$
SBP = DBP + K_a \cdot \frac{1}{PTT^2}
$$

또 다른 이론적 근거로 Moens-Korteweg equation이 제시된다.

$$
PWV = \sqrt{ \frac{E_{in} h}{2 \rho r} }
$$

여기서 $E_{in}$은 artery의 elastic modulus, $h$는 artery thickness, $r$은 artery radius, $\rho$는 blood density이다. 이 식은 혈관의 탄성과 구조적 특성이 pulse wave propagation speed에 영향을 준다는 점을 설명한다.

논문은 mean blood pressure, 즉 MBP를 다음과 같이 정의한다.

$$
MBP = \frac{1}{3}SBP + \frac{2}{3}DBP
$$

그리고 실험 결과 기반 관계식으로 다음을 제시한다.

$$
MBP = K_b + \frac{2}{0.031} \ln \left( \frac{K_c}{PTT} \right)
$$

여기서 $K_b$와 $K_c$ 역시 calibration parameter이다. 위 식들을 결합하면 DBP는 다음과 같이 표현된다.

$$
DBP = K_b + \frac{2}{0.031} \ln \left( \frac{K_c}{PTT} \right) - \frac{1}{3} \frac{K_a}{PTT^2}
$$

이론적으로는 PTT 또는 PAT를 ECG와 PPG에서 추출하고, reference BP monitor로 얻은 SBP와 DBP를 사용해 $K_a$, $K_b$, $K_c$를 calibration하면 cuffless BP sensor를 만들 수 있다. 그러나 논문은 이러한 식들이 실제 인간 cardiovascular dynamics를 충분히 설명하지 못한다고 평가한다. 특히 PTT/PAT와 BP 사이의 관계는 사람마다 다르고, 상태에 따라 변하며, periodic calibration이 필요할 수 있다.

Hand-crafted feature 기반 regression 방법은 PTT/PAT 외에 PPG waveform의 다양한 feature를 사용한다. 예를 들어 pulse area, pulse rising time, systolic time, diastolic time, dicrotic notch 관련 feature, PPG의 first derivative와 second derivative에서 추출한 feature 등이 사용된다. SDPPG는 aortic compliance와 stiffness에 관한 정보를 포함할 수 있어 혈압과 관련될 수 있다고 논문은 설명한다.

이러한 feature를 입력으로 사용하는 regression algorithm에는 MLR, SVM, regression tree, random forest, AdaBoost, ANN 등이 포함된다. MLR은 feature와 BP 사이의 선형 관계를 가정한다. SVM은 kernel function과 hyperplane을 사용하여 nonlinear decision을 수행하며, epsilon tolerance와 slack variable을 통해 residual error를 다룬다. Regression tree는 데이터를 여러 branch로 나누어 MSE를 줄이는 방향으로 BP를 예측한다. AdaBoost는 여러 weak estimator를 결합해 strong estimator를 만드는 방식이지만, 많은 estimator를 사용하면 계산량과 시간이 증가할 수 있다.

Deep learning 기반 방법은 pre-determined feature extraction을 줄이거나 제거하려는 방향이다. CNN은 convolution과 pooling을 통해 waveform에서 local morphology feature를 자동 추출한다. LSTM이나 RNN은 시간 축의 dependency와 장기적인 BP dynamics를 모델링한다. ANN-LSTM 또는 CNN-LSTM 구조는 하위 layer에서 feature를 추출하고 상위 recurrent layer에서 temporal variation을 반영한다. U-Net 방식은 PPG에서 ABP waveform을 직접 추정한 뒤, ABP waveform의 peak와 valley에서 SBP와 DBP를 도출하는 방식도 가능하다.

이 논문은 자체 손실 함수, 학습률, optimizer 등은 제시하지 않는다. 이는 review 논문이므로 각 인용 연구의 architecture와 성능을 비교하는 방식이다. 따라서 방법론의 핵심은 새로운 수식이나 모델 제안이 아니라, cuffless BP estimation 방법들을 PWV 기반, hand-crafted feature 기반 regression, deep learning 기반 feature learning으로 체계화한 데 있다.

## 4. 실험 및 결과

이 논문은 자체 실험 결과를 제시하기보다는 기존 연구들의 성능을 두 개의 큰 표로 정리한다. Table I은 pre-determined features를 사용한 regression 및 learning 기반 연구들을 요약하고, Table II는 pre-determined features 없이 raw 또는 near-raw waveform data에서 feature extraction까지 deep learning이 수행하는 연구들을 요약한다.

Pre-determined feature 기반 연구에서는 다양한 성능이 보고된다. 예를 들어 Kurylyak et al.은 21개 PPG features와 ANN을 사용했으나 mean error가 크다고 평가된다. Duan et al.은 57개 feature candidate에서 11개 feature set을 구성하고 SVR을 사용했으며, AAMI와 BHS 기준을 간신히 통과했다고 설명된다. Kachuee et al.은 PAT 기반 feature와 whole-based representation을 사용했지만 SBP와 DBP의 standard deviation이 각각 10.09 mmHg와 6.14 mmHg로 비교적 컸다.

Khalid et al.은 단일 PPG signal만으로 MLR, SVM, decision tree regression을 비교했고, pulse area와 pulse rising time 등 5개 features를 사용했다. 이 연구는 SBP 4.82 ± 4.31 mmHg, DBP 3.25 ± 4.17 mmHg의 비교적 작은 error를 보고했으며, regression tree가 다른 regression method보다 좋았다고 정리된다. 저자들은 regression tree가 noisy data와 slight nonlinearity에 강건하기 때문일 수 있다고 해석한다.

Wang et al.은 단일 PPG signal에서 morphological 및 spectral features를 추출하고 ANN을 사용하여 SBP 4.02 ± 2.79 mmHg, DBP 2.27 ± 1.82 mmHg를 보고했다. 그러나 논문은 mean error가 다른 연구에 비해 상대적으로 크다고 평가한다. Su et al.은 7개 pre-determined features를 BiLSTM에 입력하여 장기 BP dynamics를 모델링했으며, SBP와 DBP deviation이 각각 5.81 mmHg와 5.21 mmHg 수준으로 개선되었다고 설명된다. 이는 LSTM이 temporal variation을 다룰 수 있다는 점을 보여주는 초기 사례로 평가된다.

El-Hajj and Kyriacou는 22개 PPG characteristic features를 추출하고 Bi-GRU with attention을 사용해 BP를 추정했다. 이 연구는 SBP와 DBP standard deviation이 각각 4.22 mmHg와 2.07 mmHg로 AAMI threshold를 통과하는 성능을 보였다고 정리된다. 다만 논문은 MIMIC II에서 두 개의 500-record subset을 어떻게 선택했는지가 명확하지 않다고 지적한다. 이는 review 논문으로서 단순 수치 비교뿐 아니라 데이터 선택 방식의 투명성도 중요하게 본다는 점을 보여준다.

Deep learning 기반 feature extraction 연구에서는 더 다양한 architecture가 소개된다. Liang et al.은 PPG signal을 continuous wavelet transform으로 변환하고 pre-trained GoogLeNet CNN을 사용하여 hypertension classification을 수행했으며, superior hypertension classification accuracy 82.95%를 보고했다. 이 연구는 BP regression보다는 hypertension risk stratification에 가깝다.

Tanveer and Hasan은 ANN-LSTM 구조를 사용했다. 하위 ANN은 ECG와 PPG waveform에서 morphological features를 추출하고, 상위 LSTM은 시간 domain variation을 반영한다. 성능은 SBP 1.10 ± 1.56 mmHg, DBP 0.58 ± 0.85 mmHg로 매우 우수하게 보고되지만, 이 결과는 39 subjects에 기반하므로 population size가 작다는 한계가 지적된다.

Yan et al.은 multi-task learning convolutional neural network인 Deep-BP를 제안했다. ECG와 PPG signal에서 feature를 추출하고 SBP와 DBP를 추정하며, multi-task learning을 통해 representation sharing과 overfitting 감소를 도모했다. 성능은 SBP 3.09 ± 2.76 mmHg, DBP 2.11 ± 2.0 mmHg로 비교적 우수하지만, 논문은 604 subjects 기반이라는 점을 함께 언급한다.

Baek et al.은 end-to-end CNN 기반 BP prediction 모델을 제안했다. Time encoder와 frequency encoder를 사용해 ECG와 PPG signal을 처리하고, calibration-free 조건에서는 SBP 9.60 ± 9.53 mmHg, DBP 5.14 ± 5.10 mmHg, calibration-based 조건에서는 SBP 5.32 ± 5.54 mmHg, DBP 3.38 ± 3.82 mmHg를 보고했다. 이 연구는 MIMIC II와 Queensland database 모두에서 평가되었고, calibration 여부에 따른 성능 차이를 보여준다는 점에서 중요하다. 다만 CNN은 LSTM과 달리 장기 temporal variation을 직접 다루는 데 제한이 있을 수 있다고 논문은 평가한다.

Panwar et al.은 PP-Net이라는 CNN-LSTM 기반 lightweight framework를 제안했다. CNN은 feature extraction을 수행하고 LSTM은 BP estimation 또는 temporal modeling을 담당한다. MIMIC II의 1557 subjects를 사용했으며, SBP와 DBP standard deviation이 각각 5.65와 5.41로 보고된다. 이는 비교적 큰 dataset에서 얻은 결과라는 점에서 의미가 있다.

Athaya and Choi는 U-Net architecture를 사용하여 finger PPG에서 continuous non-invasive arterial blood pressure waveform을 추정했다. 추정된 ABP waveform에서 peak와 valley를 찾아 SBP와 DBP를 도출한다. SBP와 DBP standard deviation은 각각 4.42와 2.92로 보고된다. 그러나 사용된 record가 100 subjects 수준이므로 ABP waveform의 phase difference까지 고려해야 하는 문제에 비해 데이터 규모가 적을 수 있다고 논문은 지적한다.

논문의 종합적 결론은 deep learning 기반 방법이 hand-crafted feature 기반 방법보다 더 나은 가능성을 보인다는 것이다. 특히 raw PPG discrete data를 입력으로 사용하여 feature extraction을 learning machine 내부에서 수행하는 모델들이 최근 5년간 성능 개선을 이끌었다고 평가한다. 또한 LSTM은 BP dynamics의 temporal variation을 모델링하고 장기 사용 시 accuracy decay를 줄이는 데 유리한 접근으로 평가된다.

그러나 이 review는 단순히 deep learning 결과가 모두 우수하다고 말하지 않는다. 일부 연구는 sample 수가 너무 작고, 일부 연구는 MIMIC ICU data에 과도하게 의존하며, 일부 연구는 signal quality control이 강하게 적용된 조건에서만 성능을 보고한다. 따라서 성능 비교는 dataset size, subject diversity, calibration 여부, signal preprocessing, feature extraction 방식, 평가 기준을 함께 고려해야 한다고 해석할 수 있다.

## 5. 강점, 한계

이 논문의 첫 번째 강점은 cuffless PPG BP estimation 연구를 기술 발전 흐름에 따라 잘 구조화했다는 점이다. 단순히 여러 논문을 나열하는 것이 아니라, PWV/PTT 기반 이론, hand-crafted feature 기반 regression, deep learning 기반 feature extraction이라는 세 단계로 정리한다. 이를 통해 독자는 왜 PTT/PAT만으로는 부족하고, 왜 machine learning과 deep learning이 필요해졌는지 이해할 수 있다.

두 번째 강점은 PWV 기반 혈압 추정의 수학적 근거와 한계를 함께 설명한다는 점이다. Bramwell-Hill equation과 Moens-Korteweg equation을 통해 PTT와 BP 사이의 관계를 유도하지만, 실제 cardiovascular system에서는 개인별 특성과 생리 상태 변화 때문에 이 식들이 충분하지 않다고 비판한다. 이는 단순한 이론 소개보다 균형 잡힌 review로 볼 수 있다.

세 번째 강점은 다양한 regression 및 deep learning 연구들의 성능을 표로 요약하고, 각 연구의 dataset, feature, model, accuracy, 한계점을 함께 언급한다는 점이다. 특히 작은 subject 수에서 매우 좋은 성능을 보고한 연구에 대해 주의가 필요하다고 지적하고, MIMIC II subset selection이 불명확한 연구도 언급한다. 이는 reviewer 관점에서 중요한 비판적 검토이다.

네 번째 강점은 상용화 관점의 문제를 명확히 제시한다는 점이다. 논문은 알고리즘 성능만 다루지 않고, real-time signal quality screening, model size reduction, wearable firmware implementation, motion artifact, calibration, subject condition 변화, medication과 stress의 영향까지 논의한다. 이는 연구실 환경의 offline accuracy와 실제 cuffless BP sensor의 상용화 사이에 큰 간극이 있음을 잘 보여준다.

하지만 한계도 있다. 첫째, review 논문이므로 자체 실험 검증이 없다. 따라서 논문의 결론은 기존 연구들의 보고 성능에 의존한다. 각 연구의 데이터 분할 방식, subject overlap, calibration setting, preprocessing strength, validation protocol이 서로 다르기 때문에 표에 나열된 수치를 직접 비교하는 데는 한계가 있다.

둘째, 일부 논문에 대한 평가가 요약 중심이라 방법론의 깊은 재현 가능성까지 제공하지는 않는다. 예를 들어 CNN, LSTM, U-Net, attention model의 구체적 architecture나 training setting은 각 원 논문을 확인해야 한다. 이 review만으로는 개별 모델을 재현하기 어렵다.

셋째, deep learning이 hand-crafted feature보다 우수할 수 있다는 방향성은 타당하지만, deep learning 연구들 자체도 dataset bias와 overfitting 위험이 크다. MIMIC II와 Queensland database는 널리 사용되지만, ICU patient data 또는 특정 측정 환경에 편향되어 있을 수 있다. 이 논문도 이를 지적하지만, 어떤 benchmark protocol이 적절한지에 대한 구체적 제안은 제한적이다.

넷째, 상용화에 필요한 clinical validation 기준이 더 구체적으로 논의될 필요가 있다. AAMI와 BHS 기준은 언급되지만, wearable cuffless BP sensor가 실제 의료기기로 승인받기 위해 요구되는 calibration protocol, population stratification, longitudinal validation, motion condition validation 등은 더 세부적으로 다루어질 수 있었다.

다섯째, 논문은 deep learning 기반 feature extraction의 장점을 강조하지만, interpretability 문제는 상대적으로 덜 다룬다. 혈압 추정은 임상적으로 중요한 task이므로, 모델이 어떤 waveform component를 혈압 추정에 사용했는지, 개인별 calibration이 왜 필요한지, 특정 환자에서 실패하는 이유가 무엇인지 설명 가능성이 중요하다. Deep learning이 raw data에서 feature를 자동 추출한다는 점은 장점이지만, 동시에 black-box risk를 증가시킨다.

비판적으로 보면, 이 논문은 cuffless PPG BP estimation 연구의 현황을 이해하는 데 매우 유용한 review이지만, “deep learning이 다음 단계로 이끈다”는 방향성을 실제 상용화 가능성으로 연결하기 위해서는 더 엄격한 benchmark와 real-world validation이 필요하다. 특히 같은 MIMIC 기반 결과라도 subject-wise split인지, record-wise split인지, calibration 여부가 무엇인지에 따라 성능 해석이 크게 달라질 수 있다.

## 6. 결론

이 논문은 cuffless PPG blood pressure sensor를 위한 machine learning 및 deep learning 기술의 발전을 종합적으로 검토했다. 초기 연구들은 PWV theory에 기반하여 PTT와 PAT를 이용했지만, 실제 혈압은 다양한 생리적·광학적·환경적 요인의 영향을 받기 때문에 PTT/PAT만으로 정확한 추정이 어렵다. 이후 연구들은 PPG waveform에서 더 많은 hand-crafted features를 추출하고 MLR, SVM, regression tree, AdaBoost, ANN 등을 적용하여 성능을 개선했다. 최근에는 CNN, RNN, LSTM, CNN-LSTM, U-Net 등 deep learning 모델이 raw waveform에서 feature extraction과 BP estimation을 통합적으로 수행하면서 더 높은 정확도의 가능성을 보여주고 있다.

논문의 주요 기여는 이 기술 흐름을 체계적으로 정리하고, 각 접근의 이론적 근거와 한계를 비교하며, 향후 상용화를 위한 과제를 도출한 데 있다. 특히 deep learning은 pre-determined features의 부족함과 feature extraction algorithm의 불안정성을 완화할 수 있으며, LSTM 계열 모델은 BP dynamics의 temporal variation을 모델링하는 데 유리하다고 평가된다.

그러나 cuffless PPG BP sensor가 실제 daily life wearable device로 상용화되기 위해서는 아직 해결해야 할 문제가 많다. 대규모 다기관 데이터 수집, ICU가 아닌 일반 사용자 환경에서의 검증, motion artifact와 low-quality signal의 real-time screening, lightweight model 설계, firmware implementation, calibration 최소화, medication과 stress 등 개인 상태 변화에 대한 robustness가 필요하다.

결론적으로 이 논문은 PPG 기반 cuffless BP estimation 연구가 hand-crafted feature 중심에서 deep learning 기반 end-to-end feature learning으로 이동하고 있음을 잘 보여준다. 하지만 deep learning 자체가 문제를 완전히 해결하는 것은 아니며, 실제 의료기기 수준의 신뢰성과 상용화를 위해서는 데이터, 모델, 센서, 임상 검증이 함께 발전해야 한다. 이 review는 해당 분야의 연구 방향을 이해하고 후속 연구의 기술적 우선순위를 설정하는 데 유용한 기준점을 제공한다.
