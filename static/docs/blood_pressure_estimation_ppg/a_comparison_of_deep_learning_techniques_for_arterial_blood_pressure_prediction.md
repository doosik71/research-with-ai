# A Comparison of Deep Learning Techniques for Arterial Blood Pressure Prediction

* **저자**: Annunziata Paviglianiti, Vincenzo Randazzo, Stefano Villata, Giansalvo Cirrincione, Eros Pasero
* **발표연도**: 2021

## 1. 논문 개요

이 논문은 cuff를 사용하지 않고, 비침습적이며 연속적으로 arterial blood pressure, 즉 ABP를 예측하는 deep learning 기반 방법을 비교·분석한 연구이다. 전통적인 혈압 측정 방식은 크게 두 가지로 나뉜다. 하나는 동맥에 catheter를 삽입하는 invasive catheter system이고, 다른 하나는 일반적으로 사용되는 cuff-based sphygmomanometer이다. 전자는 정확하지만 고통스럽고 전문 의료진이 필요하며, 후자는 비교적 간단하지만 사용자가 측정 자세와 절차를 지켜야 하고 연속 측정이 어렵다. 또한 cuff 자체가 불편감을 유발하거나 white coat hypertension처럼 측정 상황에 따른 혈압 변화를 일으킬 수 있다.

논문의 핵심 연구 문제는 PPG와 ECG 신호를 이용하여 cuffless 방식으로 ABP, 특히 systolic blood pressure, 즉 SBP와 diastolic blood pressure, 즉 DBP를 얼마나 정확하게 예측할 수 있는지 평가하는 것이다. 저자들은 먼저 PPG만을 입력으로 사용하는 경우를 실험하고, 이후 PPG와 ECG를 함께 사용하는 경우를 실험하여 ECG 추가가 성능 향상에 기여하는지 분석하였다.

이 문제가 중요한 이유는 혈압이 cardiovascular disease와 hypertension의 핵심 위험 지표이기 때문이다. 특히 혈압의 시간적 변화, 즉 blood pressure variability는 위험한 심혈관 사건과 관련이 있기 때문에, 간헐적 측정보다 연속적이고 편안한 모니터링이 중요하다. 논문은 이러한 요구를 wearable medical device 환경과 연결하여, PPG와 ECG처럼 웨어러블 기기에 비교적 쉽게 통합 가능한 생체 신호로부터 혈압을 추정하는 방법을 제안하고 비교한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 PPG와 ECG raw signal로부터 사람이 직접 설계한 feature를 사용하지 않고, deep neural network가 혈압 예측에 필요한 temporal pattern과 signal feature를 자동으로 학습하도록 하는 것이다. 기존 연구에서는 pulse transit time, pulse wave velocity, pulse arrival time 등 ECG와 PPG에서 추출한 hand-crafted feature를 사용해 혈압을 추정하는 접근이 많았다. 그러나 이 논문은 ResNet, WaveNet, LSTM과 같은 deep learning architecture를 이용하여 raw 또는 전처리된 생체 신호에서 직접 ABP를 예측한다.

논문의 중요한 비교 축은 두 가지이다. 첫째, 입력 신호로 PPG만 사용하는 경우와 PPG에 ECG를 추가한 경우를 비교한다. PPG는 말초 혈류량 변화와 관련된 신호이므로 혈압과 연관성이 있지만, 혈관 탄성, 말초 조직 특성, 개인차 때문에 혈압과 단순한 선형 관계를 갖지 않는다. ECG는 심장의 전기적 활동을 반영하므로 심박 발생 시점과 혈류 변화 사이의 시간적 관계를 모델이 학습하는 데 도움을 줄 수 있다. 실험 결과, MIMIC 데이터셋에서는 PPG와 ECG를 함께 사용할 때 모든 제안 configuration에서 성능이 향상되었다.

둘째, 출력 방식으로 direct SBP/DBP prediction과 entire ABP signal prediction을 비교한다. Direct SBP/DBP prediction은 일정 길이의 입력 구간을 보고 SBP와 DBP 두 값을 직접 예측한다. Entire ABP signal prediction은 연속적인 ABP waveform 전체를 예측한 뒤, 그 waveform에서 peak와 valley를 찾아 SBP와 DBP를 얻는다. 논문에서는 상용 wearable healthcare device 관점에서는 SBP와 DBP 값만 직접 예측하는 방식이 실용적이며, 임상적으로는 waveform 전체를 예측하는 방식도 의미가 있다고 설명한다.

가장 좋은 성능은 ResNet 뒤에 세 개의 LSTM layer를 연결한 hybrid CNN-RNN 구조에서 얻어졌다. 이 구조는 convolutional layer를 통해 local signal feature와 downsampling을 수행하고, LSTM layer를 통해 시간적 의존성을 학습한다. 논문의 결과는 혈압 예측에서 공간적 또는 국소적 패턴 추출 능력과 temporal dependency modeling이 함께 필요하다는 점을 보여준다.

## 3. 상세 방법 설명

논문의 전체 파이프라인은 데이터 수집, 전처리, 입력 신호 구성, neural network 학습, 검증, 그리고 외부 데이터셋 테스트로 구성된다. 주요 데이터는 MIMIC database에서 추출한 ECG, PPG, ABP 신호이며, 추가 검증을 위해 Politecnico di Torino의 Neuronica Lab에서 구축한 Polito dataset이 사용되었다.

MIMIC database는 ICU 환자의 실제 생체 신호를 포함한다. ECG, PPG, ABP는 125 Hz sampling rate와 12-bit precision으로 기록되었다. 데이터는 10분 단위 segment로 구성되어 있으며, 연구진은 필요한 신호가 모두 존재하지 않는 recording을 제거하였다. 전처리 과정은 논문의 성능에 매우 중요한 역할을 한다. 먼저 NaN 값은 가장 가까운 이용 가능한 값으로 대체되었다. 이후 ABP 또는 PPG에 flat line이 있는 record, flat peak가 많은 record, 비정상적 ABP 또는 PPG를 포함하는 record가 제거되었다. ABP는 10분 recording 내에서 15 mmHg 이상 300 mmHg 이하 범위에 있어야 한다는 조건이 적용되었다. 또한 일정 시간 이상 단조 증가 또는 단조 감소하는 비정상 signal도 제거하였다.

PPG 신호에는 4th order Butterworth band-pass filter가 적용되었다. 주파수 범위는 0.5 Hz에서 8 Hz이며, 0.5 Hz 이하 성분은 baseline wandering으로, 8 Hz 이상 성분은 high-frequency noise로 간주하였다. 이후 PPG와 ABP에는 Hampel filter를 적용하여 outlier를 제거하였다. 너무 짧은 recording만 가진 환자는 제외하였고, recording이 긴 경우에는 계산 비용을 고려하여 처음 190분만 사용하였다. 마지막으로 PPG는 standardization하고 ABP는 normalization하였다. 출력이 normalize되었기 때문에, network prediction은 physiological ABP range로 되돌리기 위해 de-normalization이 필요하다. 저자들은 이 normalization 통계가 모든 test set에 항상 유효하다는 보장은 없으며, 실제 scenario에서 검증이 필요하다고 언급한다.

ECG를 포함하는 데이터셋도 별도로 구축되었다. 가장 많은 환자에게서 사용 가능한 ECG lead V를 사용하여 PPG, ECG lead V, ABP가 모두 있는 데이터를 선택하였다. ECG 신호에는 8th order Chebyshev type 1 passband filter가 적용되었고, cutoff frequency는 2 Hz와 59 Hz로 설정되었다. 이는 motion artifact와 alternating current artifact를 줄이기 위한 것이다. 전처리 후 PPG와 ECG를 모두 포함하는 MIMIC 데이터셋은 40명 환자로 구성되었다.

이 연구에서 사용된 주요 metric은 RMSE와 MAE이다. RMSE는 예측값과 실제값 차이의 제곱 평균에 루트를 취한 값으로, 큰 오차에 더 민감하다.

$$
RMSE(X,h)=\sqrt{\frac{1}{m}\sum_{i=1}^{m}(h(x^{(i)})-y^{(i)})^2}
$$

MAE는 예측값과 실제값 사이의 절대 오차 평균이다. outlier가 많을 때는 MAE가 모델 성능을 더 직관적으로 보여줄 수 있다.

$$
MAE(X,h)=\frac{1}{m}\sum_{i=1}^{m}|h(x^{(i)})-y^{(i)}|
$$

다만 저자들은 MAE를 neural network의 loss function으로 직접 사용하기 어렵다고 설명한다. 그 이유는 MAE의 gradient가 거의 일정하기 때문에 loss가 작은 영역에서도 gradient가 충분히 작아지지 않는 문제가 있기 때문이다. 따라서 학습에는 Huber loss를 사용하였다. Huber loss는 작은 오차에서는 MSE처럼 동작하고, 큰 오차에서는 MAE처럼 동작하여 outlier에 더 robust하다.

$$
L_{\delta}(y,f(x))=
\begin{cases}
\frac{1}{2}(y-f(x))^2, & |y-f(x)| \le \delta \
\delta |y-f(x)|-\frac{1}{2}\delta^2, & \text{otherwise}
\end{cases}
$$

학습에는 Adam optimizer가 사용되었고, learning rate는 $\eta=0.001$로 설정되었다. 구현은 TensorFlow 1.15 기반이며, 학습 그래프 시각화에는 TensorBoard가 사용되었다. Adam은 adaptive learning rate algorithm이므로 논문에서는 learning rate에 대한 많은 tuning이 필요하지 않았다고 설명한다.

논문에서 비교한 architecture는 ResNet, LSTM, WaveNet, fully connected network, 그리고 이들을 결합한 hybrid model이다. ResNet은 skip connection을 통해 깊은 network의 degradation 문제를 줄인다. 일반적인 deep network가 목표 mapping $h(x)$를 직접 학습하려 한다면, ResNet은 입력 $x$를 출력에 더함으로써 residual mapping $f(x)=h(x)-x$를 학습하게 한다. 이 구조는 signal이 network 전체를 쉽게 통과하도록 하며, 초기에는 identity function에 가까운 동작을 하므로 학습 안정성을 높인다. 논문에서 ResNet은 convolutional layer, Batch Normalization, ReLU activation을 포함한 residual block을 기반으로 한다.

WaveNet은 원래 raw audio waveform을 처리하기 위해 제안된 architecture이다. 이 논문에서는 time series signal인 PPG, ECG, ABP 처리에 WaveNet 구조를 적용하였다. WaveNet의 핵심은 causal convolution과 dilated convolution이다. Causal convolution은 현재 출력이 미래 입력을 보지 않도록 하며, dilated convolution은 filter 사이의 간격을 늘려 receptive field를 효율적으로 확장한다. 논문에서는 dilation rate를 1에서 8까지 두 배씩 증가시키는 block을 사용하였다. 이를 통해 낮은 layer는 short-term pattern을, 높은 layer는 long-term pattern을 처리할 수 있다.

LSTM은 recurrent neural network의 한 종류로, 장기 의존성 문제를 완화하기 위해 설계되었다. LSTM cell은 forget gate, input gate, output gate를 사용하여 어떤 정보를 기억하고 버릴지 조절한다. 논문에서는 forget gate를 다음과 같이 표현한다.

$$
f_t=\sigma(W_f[h_{t-1},x_t]+b_t)
$$

여기서 $W$는 weight vector, $b$는 bias, $\sigma$는 sigmoid function, $t$는 현재 시점을 의미한다. Output gate는 다음과 같이 표현된다.

$$
o_t=\sigma(W_o[h_{t-1},x_t]+b_o)
$$

Hidden state는 output gate와 cell state $c_t$를 사용해 다음과 같이 계산된다.

$$
h_t=o_t * \tanh(c_t)
$$

이 논문에서 LSTM은 혈압 신호의 시간적 의존성을 포착하는 데 중요한 역할을 한다. 혈압은 한 시점에서 갑자기 완전히 독립적으로 변하는 값이 아니며, 직전 시점의 상태와 연속적인 관계를 갖기 때문이다. 따라서 convolutional layer가 local feature를 추출하고 LSTM이 temporal dependency를 분석하는 ResNet + LSTM 구조가 가장 좋은 성능을 보였다.

논문은 두 가지 prediction setup을 사용하였다. Direct SBP/DBP prediction에서는 recording을 5초 chunk로 나누고, 각 chunk에서 Elgendi et al.의 algorithm을 사용해 SBP와 DBP를 추출한 뒤, 5초 구간 내 여러 cardiac cycle의 평균 SBP와 평균 DBP를 target으로 사용하였다. 이 방식의 network 출력은 SBP 값 하나와 DBP 값 하나이다. Entire ABP signal prediction에서는 network가 연속 혈압 signal을 예측한다. 이 경우 입력 sample은 2초 길이로 제한되었다. LSTM은 너무 긴 sequence를 처리할 때 gradient 계산과 장기 기억 측면에서 어려움이 있기 때문이다.

Direct SBP/DBP prediction에서는 ResNet과 ResNet + LSTM이 비교되었다. ResNet 단독 모델에서는 batch size를 650, 128, 32로 바꾸어 실험하였고, batch size 32가 더 좋은 결과를 보였다. 저자들은 작은 batch가 local minimum에 덜 갇히고 학습이 더 잘 이루어진 것으로 해석한다. 이후 ResNet 뒤에 세 개의 LSTM layer를 연결한 모델을 실험하였다. 첫 번째 LSTM layer는 bidirectional 구조였고, 각 LSTM layer는 128개 neuron으로 구성되었다. 이 모델은 direct SBP/DBP prediction에서 가장 좋은 성능을 냈다.

Entire ABP signal prediction에서는 fully connected network, LSTM, WaveNet, WaveNet + LSTM, ResNet + LSTM이 비교되었다. Fully connected network는 단순한 구조 때문에 좋은 성능을 내지 못했다. LSTM 단독 모델은 세 개의 stacked LSTM layer로 구성되었으며, 첫 번째 layer는 bidirectional이었다. WaveNet은 두 개 block으로 구성되었고 각 block에는 네 개 convolutional layer가 포함되었다. WaveNet + LSTM은 WaveNet이 feature를 추출하고 LSTM이 temporal pattern을 분석하는 방식이다. 마지막으로 ResNet + LSTM은 causal padding을 사용하고 max-pooling을 제거한 modified ResNet 뒤에 세 개 LSTM layer를 연결한 구조이다. 이 모델은 entire ABP signal prediction에서도 가장 좋은 성능을 보였다.

모델의 generalization 성능을 평가하기 위해 Leave-One-Out, 즉 LOO cross-validation이 수행되었다. LOO는 한 명의 환자를 validation으로 두고 나머지 환자로 학습하는 과정을 모든 환자에 대해 반복하는 방식이다. 논문에서는 LOO 결과가 같은 환자의 다른 recording을 test하는 경우보다 나빴다고 보고한다. 이는 personalization, 즉 개인별 calibration 또는 개인 내부 데이터가 혈압 예측 성능을 크게 향상시킨다는 점을 시사한다.

## 4. 실험 및 결과

실험은 크게 MIMIC database 기반 실험과 Polito dataset 기반 외부 검증으로 구성된다. MIMIC database는 ICU 환자의 실제 ECG, PPG, ABP 신호를 포함하므로, 다양한 병태생리와 급격한 혈압 변화를 포함하는 복잡한 데이터셋이다. PPG만 사용하는 데이터셋과 PPG + ECG를 사용하는 데이터셋이 각각 구성되었다. ECG 포함 데이터셋에서는 lead V가 주로 사용되었으며, 이는 MIMIC에서 가장 자주 기록된 ECG lead였기 때문이다.

주요 결과는 PPG와 ECG를 함께 사용할 때 모든 configuration에서 성능이 향상되었다는 점이다. Direct SBP/DBP prediction에서 ResNet 단독 모델은 PPG만 사용할 때 SBP MAE 9.556 mmHg, DBP MAE 4.217 mmHg를 기록하였다. 같은 ResNet이 PPG + ECG를 사용할 때는 SBP MAE 4.667 mmHg, DBP MAE 2.445 mmHg로 크게 개선되었다.

가장 좋은 direct prediction 성능은 ResNet + LSTM에서 얻어졌다. PPG만 사용할 때 ResNet + LSTM의 SBP MAE는 7.122 mmHg, DBP MAE는 3.534 mmHg였다. PPG + ECG를 사용할 때는 SBP MAE 4.118 mmHg, DBP MAE 2.228 mmHg로 개선되었다. RMSE 기준으로도 PPG + ECG ResNet + LSTM은 SBP RMSE 5.682 mmHg, DBP RMSE 2.986 mmHg를 기록하였다. 논문은 이 결과가 Association for the Advancement of Medical Instrumentation의 American National Standards 조건, 즉 평균 차이 5 ± 8 mmHg 기준에 부합한다고 설명한다.

Entire ABP signal prediction에서도 PPG + ECG 입력은 전반적으로 성능을 개선하였다. Fully connected network는 성능이 가장 낮았으며, PPG만 사용할 때 SBP MAE 36.559 mmHg, DBP MAE 10.602 mmHg였다. PPG + ECG를 사용해도 SBP MAE 29.753 mmHg, DBP MAE 12.759 mmHg로 충분히 좋은 결과를 내지 못했다. 이는 단순한 fully connected 구조가 time-series signal의 temporal dependency와 local waveform pattern을 적절히 처리하지 못하기 때문으로 해석할 수 있다.

LSTM 단독 모델은 PPG만 사용할 때 SBP MAE 12.118 mmHg, DBP MAE 5.018 mmHg를 기록했고, PPG + ECG에서는 SBP MAE 7.603 mmHg, DBP MAE 3.688 mmHg로 개선되었다. WaveNet은 PPG만 사용할 때 SBP MAE 18.539 mmHg, DBP MAE 8.154 mmHg였고, PPG + ECG에서는 SBP MAE 14.501 mmHg, DBP MAE 7.224 mmHg였다. WaveNet + LSTM은 PPG + ECG에서 SBP MAE 8.812 mmHg, DBP MAE 3.471 mmHg를 보였다.

Entire ABP signal prediction에서 가장 좋은 성능도 ResNet + LSTM에서 나왔다. PPG만 사용할 때 SBP MAE 8.660 mmHg, DBP MAE 3.843 mmHg였고, PPG + ECG를 사용할 때 SBP MAE 4.507 mmHg, DBP MAE 2.209 mmHg로 개선되었다. 또한 전체 pressure signal 자체에 대한 error에서도 ResNet + LSTM이 가장 우수했다. PPG만 사용할 때 entire BP signal MAE는 6.230 mmHg, RMSE는 8.883 mmHg였으며, PPG + ECG를 사용할 때 MAE는 3.282 mmHg, RMSE는 5.010 mmHg였다.

LOO cross-validation 결과는 validation set 결과보다 훨씬 나빴다. Direct SBP/DBP prediction에서 ResNet + LSTM을 사용할 때, PPG only 50명 데이터셋에서는 SBP MAE 23.5976 mmHg, DBP MAE 10.7459 mmHg였다. 같은 40명 기준으로 비교하면 PPG only는 SBP MAE 24.2227 mmHg, DBP MAE 11.1056 mmHg였고, PPG + ECG는 SBP MAE 20.3667 mmHg, DBP MAE 9.5484 mmHg였다. 즉 ECG는 LOO에서도 generalization을 개선했지만, 여전히 같은 환자 내 validation보다 error가 크게 증가했다. 이는 개인차가 혈압 예측에 큰 영향을 준다는 것을 의미한다.

논문은 LOO error가 평균 ABP와 관련이 있다고 분석한다. 데이터셋은 대부분 physiological blood pressure에 치우쳐 있었기 때문에, 평균 ABP가 높은 환자를 예측할 때 error가 커지는 경향이 나타났다. 또한 일부 PPG에는 2초 입력 window보다 긴 pattern이 존재하여, network가 이를 충분히 인식하지 못하는 경우가 있었다. 저자들은 ECG를 추가하면 validation set 성능뿐 아니라 LOO generalization도 개선되고, 예측이 평균 ABP에 덜 의존하게 된다고 설명한다.

Polito dataset은 Neuronica Lab에서 구축한 custom dataset이다. 9명의 건강한 젊은 피험자, 즉 남성 5명과 여성 4명이 참여했으며 평균 나이는 22.84 ± 1.07세였다. PPG, ECG, BP signal acquisition은 GE Healthcare B125 patient monitor를 사용해 수행되었다. 다만 Polito dataset에서는 CNAP system이나 invasive ABP measurement가 사용되지 않았고, sphygmomanometer로 혈압을 측정하였다. PPG는 300 Hz, ECG는 100 Hz로 sampling되었고, 둘 다 125 Hz로 resampling되었다. ECG는 wearable device 적용 가능성을 고려하여 lead I만 기록되었다.

Polito dataset 평가를 위해 MIMIC에서 PPG, ECG lead I, ABP가 모두 있는 별도 데이터셋이 구성되었다. 그러나 ECG lead I을 가진 MIMIC 환자는 12명에 불과했기 때문에 학습 데이터가 매우 작았다. 이 데이터로 ResNet + LSTM을 학습하여 Polito dataset에 적용한 결과, PPG + ECG 모델은 Polito dataset에서 SBP MAE 12.435 mmHg, DBP MAE 8.567 mmHg를 기록하였다. 흥미롭게도 PPG only 모델은 Polito dataset에서 SBP MAE 9.916 mmHg, DBP MAE 5.905 mmHg로 더 좋은 결과를 보였다. 즉 MIMIC 내부 validation에서는 ECG가 성능을 개선했지만, Polito 외부 데이터셋에서는 ECG 추가가 오히려 악영향을 주었다.

저자들은 이 원인을 작은 training set과 서로 다른 혈압 측정 방식에서 찾는다. MIMIC에서는 ABP가 invasive 방식으로 얻어진 반면, Polito dataset에서는 sphygmomanometer로 측정되었고, 이 장비는 약 5 mmHg의 uncertainty를 가진다. 따라서 epistemic uncertainty와 aleatoric uncertainty가 추가되어 성능 평가에 영향을 미쳤을 가능성이 있다.

모델 complexity 측면에서는 ResNet + LSTM이 가장 복잡한 모델이다. Fully connected와 LSTM의 complexity order는 $O(length \times (vector dimension)^2)$로 제시되었고, WaveNet, WaveNet + LSTM, ResNet + LSTM은 convolution kernel size가 추가되어 $O(length \times (vector dimension)^2 \times kernel size)$로 표현되었다. ResNet + LSTM은 PPG 입력에서 약 4375 FLOPs, PPG + ECG 입력에서 약 17,500 FLOPs로 가장 비용이 높지만, 동시에 가장 좋은 성능을 보였다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 cuffless ABP prediction 문제를 단일 architecture 제안으로 끝내지 않고, 여러 deep learning technique을 체계적으로 비교했다는 점이다. ResNet, WaveNet, LSTM, fully connected network, 그리고 hybrid 구조를 direct SBP/DBP prediction과 entire ABP signal prediction이라는 두 가지 setup에서 비교함으로써, 어떤 구조가 어떤 예측 방식에 적합한지 보여준다. 특히 ResNet + LSTM이 두 setup 모두에서 가장 좋은 성능을 보였다는 결과는 ABP prediction에서 convolutional feature extraction과 recurrent temporal modeling의 결합이 효과적임을 잘 뒷받침한다.

두 번째 강점은 PPG only와 PPG + ECG를 명확히 비교했다는 점이다. 논문은 ECG가 MIMIC 기반 실험에서 모든 configuration의 성능을 개선했다는 결과를 제시한다. 이는 PPG만으로도 혈압 예측이 가능하지만, ECG를 추가하면 심장 전기 활동과 말초 혈류 변화 사이의 시간적 관계를 network가 더 잘 학습할 수 있음을 시사한다.

세 번째 강점은 validation set 성능뿐 아니라 Leave-One-Out cross-validation을 수행해 generalization 문제를 분석했다는 것이다. 많은 생체 신호 기반 예측 연구에서 같은 subject의 다른 recording을 test로 사용할 경우 성능이 과대평가될 수 있다. 이 논문은 LOO 결과가 크게 나빠지는 현상을 명시적으로 보고하며, personalization의 중요성을 지적한다. 이는 실제 wearable blood pressure monitoring 시스템을 설계할 때 개인별 calibration이 필요할 수 있음을 보여주는 중요한 관찰이다.

네 번째 강점은 외부 custom dataset인 Polito dataset으로 검증을 시도했다는 점이다. 비록 성능이 MIMIC 내부 validation만큼 좋지는 않았지만, 서로 다른 장비와 다른 피험자 집단에서 모델을 테스트한 점은 실제 적용 가능성을 평가하려는 시도로 볼 수 있다. 또한 Polito 결과에서 ECG가 오히려 성능을 악화시킨 원인을 작은 lead I training dataset과 measurement uncertainty로 분석한 점은 연구의 한계를 비교적 솔직하게 드러낸다.

그러나 한계도 명확하다. 첫째, MIMIC 데이터셋의 혈압 분포가 physiological value에 치우쳐 있어 고혈압 또는 비정상 혈압 구간에 대한 예측력이 제한될 수 있다. 논문에서도 평균 ABP가 높은 환자에서 error가 커지는 경향을 보고한다. 이는 모델이 충분히 다양한 혈압 range를 학습하지 못했을 가능성을 의미한다.

둘째, LOO cross-validation 성능이 validation set 성능보다 크게 악화되었다. 이는 모델이 subject-specific pattern에 상당히 의존한다는 뜻이다. 실제 cuffless blood pressure device는 새로운 사용자에게도 안정적으로 동작해야 하므로, 이 문제는 상용화 또는 임상 적용에서 핵심적인 한계이다. 논문은 ECG가 generalization을 일부 개선한다고 보고하지만, LOO error는 여전히 높은 수준이다.

셋째, Polito dataset은 9명의 젊고 건강한 피험자로만 구성되어 있다. 따라서 elderly patient, cardiovascular pathology를 가진 환자, 혈압 변동성이 큰 환자에 대한 일반화 성능을 판단하기 어렵다. 논문도 향후 연구에서 elderly patients와 cardiovascular pathologies를 포함한 데이터베이스에서 검증할 필요가 있다고 언급한다.

넷째, Polito dataset의 혈압 측정 방식은 MIMIC의 invasive ABP와 다르다. Polito에서는 sphygmomanometer를 사용했기 때문에 ground truth 자체에 measurement uncertainty가 존재한다. 이로 인해 모델 평가가 noisy해졌을 수 있으며, MIMIC에서 학습한 continuous ABP prediction 모델과 Polito의 intermittent cuff-based measurement 사이에는 target definition mismatch도 존재할 수 있다.

다섯째, 논문은 transfer learning을 탐색하지 않았다. 저자들은 future work에서 이를 적용할 예정이라고 언급한다. ECG lead V와 lead I의 차이, MIMIC와 Polito의 장비 차이, 환자군 차이 등을 고려하면 transfer learning 또는 domain adaptation이 중요한 후속 연구 방향이 될 수 있다.

마지막으로, 논문은 neural network가 black-box라는 일반적 한계를 충분히 해결하지 않는다. 생체 신호 기반 의료 예측 모델에서는 예측 근거와 failure case 분석이 매우 중요하지만, 이 논문은 주로 성능 비교와 architecture 설명에 집중한다. 특히 어떤 waveform pattern이 SBP 또는 DBP 예측에 결정적이었는지, ECG가 어떤 생리적 정보를 통해 성능을 높였는지에 대한 explainability 분석은 부족하다.

## 6. 결론

이 논문은 PPG와 ECG를 이용한 cuffless arterial blood pressure prediction에서 다양한 deep learning architecture를 비교한 연구이다. 핵심 결론은 PPG만으로도 연속적이고 비침습적인 ABP 예측이 가능하지만, MIMIC 데이터셋에서는 ECG를 추가했을 때 모든 configuration에서 성능이 향상되었다는 것이다. 특히 ResNet + LSTM hybrid architecture가 direct SBP/DBP prediction과 entire ABP signal prediction 모두에서 가장 좋은 성능을 보였다.

가장 좋은 결과는 PPG + ECG를 입력으로 사용하고 ResNet 뒤에 세 개의 LSTM layer를 연결한 direct SBP/DBP prediction 모델에서 얻어졌다. 이 모델은 validation set에서 SBP MAE 4.118 mmHg, DBP MAE 2.228 mmHg를 달성했으며, RMSE는 각각 5.682 mmHg와 2.986 mmHg였다. 논문은 이 결과가 ANSI/AAMI 기준에 부합한다고 설명한다.

연구의 실질적 의미는 wearable portable device에 내장 가능한 cuffless blood pressure monitoring algorithm의 가능성을 보여준다는 점이다. PPG와 ECG는 웨어러블 기기에서 비교적 쉽게 수집할 수 있는 신호이므로, 이 접근은 hypertension이나 cardiovascular disease의 조기 감지, 연속 건강 모니터링, 고령자 또는 만성질환자의 자가 모니터링에 활용될 가능성이 있다.

다만 실제 적용을 위해서는 해결해야 할 문제가 남아 있다. 새로운 사용자에 대한 generalization 성능은 아직 제한적이며, LOO cross-validation에서 error가 크게 증가했다. 또한 데이터셋의 혈압 분포가 physiological range에 치우쳐 있고, Polito dataset은 젊고 건강한 소수 피험자로 구성되어 있어 다양한 임상 population에 대한 검증이 부족하다. 따라서 향후 연구에서는 MIMIC II, MIMIC III와 같은 더 큰 데이터셋, elderly patients와 cardiovascular pathologies를 포함한 데이터셋, 그리고 sudden blood pressure change를 감지하는 robust model이 필요하다.

종합하면, 이 논문은 cuffless ABP prediction에서 deep learning의 가능성을 실험적으로 보여주었으며, 특히 ResNet + LSTM 구조와 ECG 추가의 효과를 강조한다. 동시에 개인차, domain shift, 작은 외부 검증 데이터셋, ground truth uncertainty라는 현실적인 문제도 드러내어, 향후 wearable blood pressure monitoring 연구가 나아가야 할 방향을 제시한다.
