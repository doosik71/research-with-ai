# Non-Invasive Continuous Blood Pressure Estimation from Single-Channel PPG Based on a Temporal Convolutional Network Integrated with an Attention Mechanism

* **저자**: Dong Dai, Zhaohui Ji, Haiyan Wang
* **발표연도**: 2024

## 1. 논문 개요

이 논문은 single-channel Photoplethysmography, 즉 단일 채널 PPG 신호만을 이용하여 cuff-less, non-invasive, continuous blood pressure estimation을 수행하는 딥러닝 모델을 제안한다. 기존 cuff 기반 혈압 측정은 특정 시점의 혈압만 제공하므로 연속적인 혈압 변화를 추적하기 어렵고, 측정 환경, cuff 크기, 측정자 숙련도 등에 영향을 받을 수 있다. 반면 침습적 연속 혈압 측정은 catheter와 같은 장치를 사용해야 하므로 감염이나 합병증 위험이 있다. 이러한 문제를 해결하기 위해 논문은 PPG 신호 기반의 비침습적 연속 혈압 추정 방법을 연구한다.

논문의 핵심 목표는 PPG와 혈압 사이의 장기적인 시간 의존성을 잘 학습하면서, PPG 신호 내에서 혈압 추정에 중요한 feature에 더 집중할 수 있는 end-to-end 모델을 설계하는 것이다. 이를 위해 저자들은 Temporal Convolutional Network, TCN에 Convolutional Block Attention Module, CBAM을 결합한 TCN-CBAM 모델을 제안한다. 모델은 PPG, PPG의 1차 derivative인 PPG’, PPG의 2차 derivative인 PPG’’를 입력으로 사용하고, 최종적으로 systolic blood pressure, SBP와 diastolic blood pressure, DBP를 예측한다.

연구 문제는 크게 세 가지로 볼 수 있다. 첫째, single-channel PPG만으로 SBP와 DBP를 beat-to-beat 방식으로 얼마나 정확하게 추정할 수 있는가이다. 둘째, RNN, LSTM, GRU와 같은 recurrent architecture 대신 TCN의 dilated causal convolution을 사용하면 긴 시계열 의존성을 더 효율적으로 학습할 수 있는가이다. 셋째, CBAM attention mechanism을 통합하면 PPG 기반 혈압 추정에서 중요한 channel과 time position을 더 잘 강조하여 성능을 향상시킬 수 있는가이다.

이 문제는 임상적·실용적 중요성이 크다. 혈압은 심혈관계 건강 상태를 평가하는 핵심 지표이며, 고혈압은 전 세계적으로 매우 중요한 공중보건 문제이다. 논문은 2021년 기준 전 세계 성인 12.8억 명이 고혈압을 앓고 있으며, 고혈압이 매년 1천만 명 이상의 사망과 관련된다고 언급한다. 따라서 손쉽고 지속적인 혈압 모니터링 기술은 cardiovascular disease prevention, hypertension management, wearable healthcare, remote monitoring 분야에서 중요한 역할을 할 수 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 PPG 기반 혈압 추정을 time series regression 문제로 보고, 긴 시간 의존성을 학습하는 TCN과 중요한 feature를 선택적으로 강조하는 CBAM을 결합하는 것이다. PPG 신호는 혈관의 filling과 emptying 과정을 반영하며, 심박과 혈압 변동에 밀접하게 연결되어 있다. 따라서 특정 시점의 PPG 값만 보는 것이 아니라 일정 구간의 waveform 변화, 주기성, slope, peak 주변 형태, derivative pattern 등을 함께 고려해야 한다.

기존 접근 방식은 크게 두 부류로 나뉜다. 전통적 machine learning 방식은 PPG waveform에서 half-width, two-thirds width, systolic ascending time, diastolic time, peak, valley, dicrotic notch, inflection point 등 수작업 feature를 추출한 뒤 linear regression, support vector regression, ANN 등을 사용한다. 이 방식은 feature point를 정확히 찾아야 하므로 고품질 PPG 신호가 필요하고, wearable device에서 발생하는 noise와 motion artifact에 취약하다. 또한 feature 설계에 전문 지식과 반복적 실험이 필요하다.

Deep learning 방식은 raw signal 또는 전처리된 signal에서 feature를 자동으로 학습한다. CNN은 local waveform pattern을 잘 포착할 수 있고, RNN, LSTM, GRU는 temporal dependency를 학습할 수 있다. 그러나 RNN 계열 모델은 긴 sequence를 다룰 때 gradient vanishing 또는 exploding 문제가 발생할 수 있고, 순차 계산 특성 때문에 대규모 데이터에서 학습 효율이 제한될 수 있다. 논문은 이러한 한계를 보완하기 위해 TCN을 사용한다.

TCN의 핵심은 causal convolution과 dilated convolution이다. Causal convolution은 현재 시점의 출력을 계산할 때 미래 정보를 사용하지 않도록 제한한다. Dilated convolution은 convolution kernel 사이에 간격을 두어 receptive field를 넓힌다. 이 덕분에 모델은 매우 깊은 recurrent 구조 없이도 긴 시간 구간의 정보를 포착할 수 있다. 즉, PPG waveform의 국소 패턴과 장기적 시간 의존성을 동시에 다룰 수 있다.

CBAM의 역할은 feature map에서 중요한 channel과 중요한 time position에 더 큰 가중치를 부여하는 것이다. 이 논문에서는 PPG, PPG’, PPG’’가 서로 다른 channel처럼 사용된다. 이 세 신호는 혈압과의 관련성이 다를 수 있으므로, channel attention은 어떤 signal component가 더 중요한지 학습한다. 또한 SBP와 DBP는 waveform의 특정 위치나 구간과 더 강하게 관련될 수 있으므로, spatial attention은 time dimension을 하나의 spatial dimension처럼 간주하여 중요한 시간 위치를 강조한다.

따라서 이 논문의 차별점은 단일 PPG 기반 혈압 추정에서 TCN의 dilated causal convolution으로 장기 의존성을 학습하고, CBAM으로 channel-wise 및 temporal-position-wise 중요도를 학습한다는 점이다. 논문은 이를 통해 CNN, CNN-GRU, CNN-LSTM, TCN, traditional machine learning 모델보다 더 좋은 SBP 및 DBP 추정 성능을 달성했다고 주장한다.

## 3. 상세 방법 설명

전체 파이프라인은 데이터 선택, PPG baseline drift 제거, signal segmentation, label extraction, TCN-CBAM 모델 학습, 평가의 순서로 구성된다. 사용 데이터는 UCI의 “Cuff-Less Blood Pressure Estimation Data Set”이며, 이 데이터셋은 MIMIC-II online waveform database에서 추출된 subset이다. 각 record는 ICU 환자의 PPG, ECG, arterial blood pressure, ABP 신호를 포함하고, sampling rate는 125 Hz이며, 기록 길이는 8초에서 10분까지 다양하다.

논문은 단일 채널 PPG 기반 방법을 목표로 하므로, 실제 모델 입력에는 ECG를 사용하지 않는다. ECG와 PPG를 함께 사용하는 기존 방식은 data synchronization, signal fusion, sensor cost, implementation complexity, noise robustness 측면에서 불리할 수 있다고 설명한다. 반면 single-channel PPG는 데이터 수집과 처리 과정이 단순하고, fingertip에서 안정적으로 측정할 수 있어 장시간 혈압 모니터링에 적합하다고 본다.

전처리의 첫 단계는 baseline drift elimination이다. UCI 데이터셋에서는 원 신호에 moving average filtering과 abnormal segment 제거가 일부 수행되었지만, baseline wander 문제는 충분히 처리되지 않았다고 논문은 설명한다. Baseline wander는 호흡, 자세 변화, 온도 변화 등으로 인해 PPG 신호 전체가 위아래로 천천히 움직이는 현상이다. 이 현상은 서로 다른 PPG cycle을 비교하기 어렵게 만들고, signal-to-noise ratio를 낮추며, 혈압 추정 모델의 오차를 증가시킬 수 있다.

이를 해결하기 위해 논문은 Variational Mode Decomposition, VMD를 사용한다. VMD는 원 신호를 서로 다른 주파수 특성을 갖는 여러 Intrinsic Mode Function, IMF로 분해하는 방법이다. 낮은 주파수 IMF는 일반적으로 baseline wander와 같은 느리게 변하는 성분을 포함한다. 따라서 논문은 원래 PPG 신호에서 low-frequency IMF를 제거한 뒤 신호를 재구성하여 baseline drift를 줄인다.

논문에서 제시한 VMD 최적화 문제는 다음과 같다.

$$
\min \sum_i |A_{ni} \otimes u_n - x(t)|^2 + \lambda \sum_i \int (\partial_t u_p(t))^2 dt
$$

여기서 $x(t)$는 원 신호이고, $A_{ni}$는 각 IMF의 envelope function, $u_n$은 각 IMF의 phase function으로 설명된다. $\lambda$는 regularization parameter이다. 첫 번째 항은 분해된 mode들을 통해 원 신호를 얼마나 잘 재구성하는지를 나타내는 reconstruction error이고, 두 번째 항은 IMF가 지나치게 불안정하거나 거칠어지지 않도록 smoothness를 제한하는 regularization term이다. 논문은 VMD를 통해 얻은 low-frequency IMF가 원 신호의 baseline drift와 유사한 추세를 보이며, 이를 제거하면 PPG의 주기성이 더 뚜렷해진다고 설명한다.

그다음 signal segmentation을 수행한다. 데이터 균형을 위해 기록 길이가 8분보다 짧은 record는 제외한다. 이후 PPG와 ABP recording을 10초 window 단위로 나눈다. 각 ABP frame에서는 peak and valley detection algorithm을 사용하여 SBP와 DBP label을 추출한다. SBP는 ABP cycle에서 main peak의 amplitude로 정의되고, DBP는 main peak 이전에 인접한 trough의 amplitude로 정의된다.

극단적인 혈압값이 PPG와 혈압 사이의 일반적 관계 학습을 방해하지 않도록, 논문은 SBP가 80 mmHg에서 180 mmHg 사이이고 DBP가 60 mmHg에서 130 mmHg 사이인 record segment만 유지한다. 전처리 후 총 270,488개의 sample data point를 얻었다고 보고한다. 각 sample은 8초 길이의 PPG 신호와 대응되는 SBP 및 DBP 값을 포함한다. Sampling rate가 125 Hz이므로 8초 PPG는 1000개 time step에 해당한다.

모델 입력은 Table 1에서 size $3 \times 1000$으로 제시된다. 이는 PPG, PPG’, PPG’’의 세 channel과 1000개의 time step을 의미하는 것으로 해석할 수 있다. PPG’는 PPG의 1차 derivative이고, PPG’’는 2차 derivative로 볼 수 있다. 논문은 derivative 계산 방식의 세부 수식은 제공된 추출 텍스트에서 명확히 설명하지 않는다. 다만 CBAM 설명에서 PPG, PPG’, PPG’’가 서로 다른 channel처럼 사용된다는 점은 명시되어 있다.

제안 모델인 TCN-CBAM은 input layer, 네 개의 stacked dilated causal convolutional residual block, CBAM attention module, 두 개의 fully connected layer, output layer로 구성된다. 각 residual block은 one-dimensional dilated causal convolution layer를 포함하며, dilation rate는 $2^i$ 형태로 증가한다. Kernel size는 $1 \times 3$, convolution kernel 수는 64이다. Normalization은 batch normalization을 사용하고, activation function은 ReLU로 제시된다. 추출 텍스트의 Methods 설명 부분에서는 ELU activation도 언급되지만, Table 1에는 ReLU가 제시되어 있어 activation function 설명에 약간의 불일치가 있다. 최종 설정은 Table 1의 ReLU를 기준으로 보는 것이 타당하지만, 원문 전체 확인 없이는 정확한 구현을 단정하기 어렵다.

Dilated causal convolution의 1차원 연산은 다음과 같이 제시된다.

$$
y[t] = \sum_{k=0}^{K} x[t - r \cdot k] \cdot h[k]
$$

여기서 $y[t]$는 출력 sequence의 $t$번째 값, $x[t]$는 입력 sequence의 $t$번째 값, $h[k]$는 convolution kernel의 weight parameter, $r$은 dilation coefficient, $K$는 convolution kernel size를 의미한다. 이 식은 현재 출력 $y[t]$를 계산할 때 현재 또는 과거의 입력 위치 $x[t - r \cdot k]$만 사용한다는 점에서 causal하다. 또한 $r$이 커질수록 convolution이 입력을 촘촘히 보는 것이 아니라 간격을 두고 보게 되므로 더 넓은 시간 범위를 포괄할 수 있다. 예를 들어 dilation factor가 커지면 멀리 떨어진 PPG waveform의 과거 패턴까지 현재 예측에 반영할 수 있다.

Residual block은 깊은 network에서 gradient 흐름을 개선하고 학습 안정성을 높이는 데 사용된다. 논문의 Figure 3 구조에는 dilated causal convolution, weight normalization 또는 normalization, ReLU, dropout, residual addition, 1×1 convolution 등이 포함된 형태로 나타난다. 1×1 convolution은 residual connection에서 channel dimension을 맞추기 위한 projection 역할을 할 수 있다. 다만 제공된 텍스트만으로 각 residual block 내부 layer의 정확한 반복 횟수와 순서는 완전히 확정할 수 없다.

CBAM은 channel attention module과 spatial attention module로 구성된다. Channel attention module은 feature map의 channel dimension에서 global average pooling과 max pooling을 적용한 뒤, shared MLP를 통해 channel별 중요도 weight를 생성한다. 생성된 weight는 원 feature map에 곱해져 중요한 channel이 강조된 refined feature를 만든다. 이 논문 맥락에서 channel은 PPG, PPG’, PPG’’ 또는 convolution을 통해 생성된 feature channel을 의미할 수 있다. PPG 원신호와 derivative 신호들이 혈압 추정에 기여하는 정도가 다르기 때문에 channel attention은 이 차이를 학습한다.

Spatial attention module은 feature map의 위치별 중요도를 학습한다. 일반적인 image CNN에서는 spatial attention이 이미지의 위치별 중요도를 학습하지만, 이 논문에서는 time dimension을 spatial dimension처럼 해석한다. 즉, PPG waveform의 모든 time step이 혈압 추정에 동일하게 중요한 것은 아니므로, 특정 peak 주변, 상승 구간, 하강 구간, derivative 변화가 큰 구간 등에 더 큰 attention weight를 부여할 수 있다. Spatial attention layer의 kernel size는 $1 \times 7$로 제시된다.

최종적으로 attention이 적용된 feature는 flatten 또는 fully connected layer를 통과한다. Fully connected layer는 첫 번째 layer에서 input dimension 64, output dimension 128을 갖고, 두 번째 layer에서 input dimension 128, output dimension 1로 제시되어 있다. 그러나 output layer는 $2 \times 1$이라고 되어 있으므로 최종 출력은 SBP와 DBP 두 값이다. Table 1의 fully connected layer 설명과 output layer 설명이 다소 압축적으로 제시되어 있어, SBP와 DBP 각각에 별도 head를 두었는지, 하나의 shared representation에서 두 출력을 생성하는지는 추출 텍스트만으로 명확하지 않다.

학습에서는 Adam optimizer를 사용한다. Adam은 gradient의 1차 moment와 2차 moment를 이용하여 parameter별 adaptive learning rate를 적용하는 최적화 알고리즘이다. 초기 learning rate는 0.0001이고, batch size는 32이며, 총 1500 iterations 동안 학습한다. 논문은 model parameters가 training 및 validation set에서 convergence할 때까지 충분히 학습되었다고 설명한다.

학습 손실 함수는 Mean Squared Error, MSE이다. MSE는 예측 혈압과 실제 혈압의 차이를 제곱하여 평균한 값으로, 큰 오차에 더 큰 penalty를 부여한다.

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(BP_i - \widehat{BP}_i)^2
$$

여기서 $BP_i$는 실제 혈압값, $\widehat{BP}_i$는 모델이 예측한 혈압값, $n$은 sample 수이다. MSE는 regression 문제에서 널리 쓰이며, 혈압 추정처럼 연속값을 예측하는 작업에 적합하다.

평가 지표는 Mean Absolute Error, MAE, Standard Deviation, STD, Mean Absolute Percentage Error, MAPE이다. MAE는 예측값과 실제값 사이의 절대 오차 평균이다.

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

여기서 $y_i$는 실제 혈압값, $\hat{y}_i$는 예측 혈압값이다. MAE가 작을수록 평균적인 예측 오차가 작다는 의미이다.

STD는 예측 오차의 변동성을 나타낸다. 논문에서는 Mean Error, ME를 먼저 계산하고, 각 오차가 ME 주변에서 얼마나 퍼져 있는지를 STD로 계산한다.

$$
\text{ME} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)
$$

$$
\text{STD} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i - \text{ME})^2}
$$

ME는 모델이 평균적으로 혈압을 과대추정하는지 또는 과소추정하는지 보여주는 bias 지표이고, STD는 오차의 안정성을 보여준다. STD가 작으면 예측 오차가 일정 범위 안에 안정적으로 분포한다는 의미이다.

MAPE는 실제값 대비 오차의 비율을 백분율로 나타내는 지표이다.

$$
\text{MAPE} = \frac{1}{n}\sum_{i=1}^{n}\frac{y_i - \hat{y}_i}{y_i} \times 100%
$$

다만 논문이 제시한 식에는 절대값 기호가 명확히 포함되어 있지 않다. 일반적인 MAPE 정의는 절대값을 사용하므로, 논문 식이 표기상 누락인지 실제 구현이 그렇게 되었는지는 제공된 텍스트만으로 확인할 수 없다. 또한 실제값 $y_i$가 0에 가깝거나 0이면 MAPE가 매우 커지거나 계산이 불가능할 수 있는데, 혈압값은 일반적으로 0에 가깝지 않으므로 이 문제는 본 연구에서는 크지 않을 가능성이 있다.

## 4. 실험 및 결과

실험 데이터는 UCI “Cuff-Less Blood Pressure Estimation Data Set”에서 전처리한 270,488개 sample이다. 데이터는 7:1:2 비율로 나뉘며, 첫 70%는 training set, 다음 10%는 validation set, 마지막 20%는 test set으로 사용된다. 이 분할 방식은 시간 순서 또는 데이터 순서 기준으로 나눈 것으로 보이지만, subject-independent split인지 여부는 제공된 텍스트에서 명확히 제시되지 않는다. 혈압 추정에서 subject overlap은 성능을 과대평가할 수 있으므로 중요한 검토 지점이다.

실험 환경은 Windows 10, Python 3.8, PyTorch 13.1로 제시되어 있다. 다만 PyTorch의 실제 버전 표기상 “13.1”은 일반적인 PyTorch 버전 체계와 맞지 않으므로, 원문 또는 오탈자 확인이 필요하다. 수치 계산에는 NumPy 2.0, 데이터 처리에는 Pandas 2.0.0, 시각화에는 Matplotlib 3.4.3, 전통적 machine learning 모델 구현에는 Scikit-learn 0.24.2를 사용했다. 하드웨어는 Intel Core i9 processor와 NVIDIA GeForce 4070Ti GPU이다.

비교 대상은 총 10개 모델이다. 전통적 machine learning 모델로는 linear regression, support vector regression, multilayer perceptron, XGBoost, random forest, K-nearest neighbours regression이 사용된다. Deep learning 모델로는 CNN, CNN-GRU, CNN-LSTM, TCN이 사용된다. 제안 모델인 TCN-CBAM까지 포함하면 총 11개 모델의 성능을 비교한다. CNN 계열 모델의 convolution kernel 수와 크기는 TCN-CBAM과 일관되게 설정되었다고 설명한다. CNN-GRU는 hidden node 128개를 가진 two-layer GRU를 포함하고, CNN-LSTM은 hidden node 128개를 가진 two-layer LSTM을 포함한다.

정량 결과는 Table 3에 제시된다. SBP estimation에서 TCN-CBAM은 MAE 5.3482 mmHg, STD 8.3410 mmHg, MAPE 3.34%를 달성한다. DBP estimation에서는 MAE 2.1190 mmHg, STD 3.1795 mmHg, MAPE 2.40%를 달성한다. 이 값은 비교 모델 중 가장 좋은 성능이다. 특히 DBP에서는 TCN, CNN-LSTM, CNN-GRU 등 다른 deep learning 모델보다도 명확히 낮은 MAE와 STD를 보인다.

SBP 결과를 보면 linear regression은 MAE 14.0665 mmHg, SVR은 12.7897 mmHg, MLP는 13.5818 mmHg로 성능이 낮다. Ensemble 계열인 XGBoost는 8.4072 mmHg, random forest는 7.1597 mmHg, KNN은 7.3237 mmHg로 개선되지만, deep learning 모델보다는 낮은 성능을 보인다. CNN은 SBP MAE 8.2555 mmHg, CNN-GRU는 5.6089 mmHg, CNN-LSTM은 5.3837 mmHg, TCN은 5.8680 mmHg이다. TCN-CBAM은 이들보다 낮은 5.3482 mmHg를 보인다. 다만 SBP의 경우 CNN-LSTM과 TCN-CBAM의 MAE 차이는 약 0.0355 mmHg로 매우 작기 때문에, 통계적 유의성 검정 없이 큰 차이라고 단정하기는 어렵다.

DBP 결과에서는 TCN-CBAM의 우위가 더 분명하다. Linear regression, SVR, MLP는 약 4.3~4.5 mmHg의 MAE를 보이고, XGBoost, random forest, KNN은 약 2.68~2.99 mmHg의 MAE를 보인다. Deep learning 모델에서는 CNN 3.3077 mmHg, CNN-GRU 2.5339 mmHg, CNN-LSTM 2.5029 mmHg, TCN 2.3707 mmHg이며, TCN-CBAM은 2.1190 mmHg로 가장 낮다. 이는 CBAM attention이 DBP 추정에서 특히 도움이 되었을 가능성을 시사한다.

논문은 AAMI 기준과 비교하여 해석한다. AAMI 기준은 일반적으로 mean error가 5 mmHg 이내이고 standard deviation이 8 mmHg 이내일 것을 요구한다. 논문은 TCN-CBAM의 SBP 결과가 MAE 5.3482 mmHg, STD 8.3410 mmHg이므로 AAMI의 5 ± 8 mmHg 기준에 약간 미달한다고 설명한다. 반면 DBP 결과는 MAE 2.1190 mmHg, STD 3.1795 mmHg로 AAMI 기준을 충족한다. 따라서 이 모델은 DBP 추정에서는 임상 기준에 더 근접하지만, SBP 추정에서는 아직 개선이 필요하다.

BHS 기준 결과는 Table 2에 제시된다. BHS Grade A 기준은 예측 오차가 5 mmHg 이내인 비율이 60% 이상, 10 mmHg 이내가 85% 이상, 15 mmHg 이내가 95% 이상이어야 한다. TCN-CBAM의 SBP 오차 비율은 5 mmHg 이내 48.74%, 10 mmHg 이내 73.19%, 15 mmHg 이내 87.06%이다. 이는 Grade A, B, C 기준 모두를 완전히 충족하지 못한다. DBP의 경우 5 mmHg 이내 38.62%, 10 mmHg 이내 88.73%, 15 mmHg 이내 97.29%이다. DBP는 10 mmHg와 15 mmHg 기준에서는 Grade A 수준을 넘지만, 5 mmHg 기준에서는 Grade C 기준인 40%에도 약간 미달한다. 따라서 BHS 기준으로 보면 논문 모델은 특히 5 mmHg 이내의 정밀한 예측 비율에서 부족함이 있다.

Scatter plot과 Bland–Altman plot 분석도 포함된다. Figure 7에서 SBP와 DBP의 예측값과 실제값 사이에는 양의 상관관계가 나타난다. Pearson correlation coefficient는 SBP에서 0.8, DBP에서 0.6으로 보고된다. SBP의 상관계수가 DBP보다 높지만, 논문은 DBP regression line의 slope가 1에 더 가깝고 intercept의 절대값이 작아 전체 정확도가 더 높다고 설명한다. 이는 상관계수와 calibration이 서로 다른 측면을 보여준다는 점을 시사한다. 상관계수는 값의 변화 방향과 선형 관계를 보여주지만, 실제 예측 오차의 크기를 직접 의미하지는 않는다.

Figure 8은 모델별 error distribution을 보여준다. 논문은 linear regression, SVR, MLP처럼 구조가 단순한 모델은 SBP와 DBP 모두에서 error distribution이 넓고 성능이 낮다고 설명한다. Random forest와 XGBoost는 오차가 줄어들지만, CNN과 RNN 기반 구조에는 미치지 못한다. Deep learning 모델들은 time-series feature를 더 잘 포착하기 때문에 전반적으로 우수한 결과를 보이며, TCN-CBAM은 가장 낮은 error distribution을 보인다고 해석한다.

실험 결과의 전체적인 의미는 다음과 같다. TCN-CBAM은 단일 채널 PPG 기반 혈압 추정에서 기존 traditional machine learning 모델과 CNN-RNN 계열 모델보다 전반적으로 낮은 오차를 달성했다. 특히 DBP 추정에서는 AAMI 기준을 충족할 정도로 안정적인 결과를 보였다. 반면 SBP 추정에서는 비교 모델 중 최고 성능이지만 AAMI 기준에는 약간 미치지 못하고, BHS 기준에서도 충분한 등급을 달성하지 못한다. 따라서 논문은 제안 모델의 우수성을 보여주지만, 임상적으로 즉시 사용할 수 있는 수준의 완전한 검증에는 아직 도달하지 못했다고 보는 것이 적절하다.

## 5. 강점, 한계

이 논문의 가장 중요한 강점은 single-channel PPG만을 사용하는 실용적인 설정을 채택했다는 점이다. ECG와 PPG를 함께 사용하는 방식은 성능 측면에서 유리할 수 있지만, 실제 wearable 또는 장시간 모니터링 환경에서는 여러 센서를 동기화해야 하고 사용자의 착용 부담이 커진다. 이 논문은 single-channel PPG에 집중하여 데이터 수집과 시스템 구현의 복잡성을 낮추는 방향을 선택했다.

두 번째 강점은 TCN을 사용하여 RNN 계열 모델의 한계를 보완하려 했다는 점이다. PPG 기반 혈압 추정은 시간적 의존성을 포함하는 regression 문제이며, RNN, LSTM, GRU는 이러한 문제에 널리 사용되어 왔다. 그러나 recurrent architecture는 긴 sequence에서 계산 효율과 gradient 안정성 문제가 생길 수 있다. TCN은 dilated causal convolution을 통해 긴 receptive field를 확보하면서도 convolution 기반 병렬 처리가 가능하므로, 대규모 시계열 데이터에 적합하다.

세 번째 강점은 CBAM attention을 physiological time series에 적용했다는 점이다. CBAM은 원래 CNN feature map에서 channel과 spatial 위치의 중요도를 조정하는 모듈로 널리 사용된다. 이 논문은 이를 PPG, PPG’, PPG’’ channel과 time dimension에 적용하여, 혈압 추정에 중요한 신호 성분과 시간 구간을 강조한다. 특히 DBP estimation에서 TCN-CBAM이 TCN보다 좋은 성능을 보인 점은 attention mechanism의 효과를 뒷받침하는 결과로 해석할 수 있다.

네 번째 강점은 비교 실험의 범위가 비교적 넓다는 점이다. 논문은 linear regression, SVR, MLP, XGBoost, random forest, KNN 같은 전통적 regression 모델뿐 아니라 CNN, CNN-GRU, CNN-LSTM, TCN과도 비교한다. 이를 통해 제안 모델이 단순한 machine learning뿐 아니라 기존 deep learning baseline보다도 경쟁력이 있음을 보여주려 한다.

다만 한계도 분명하다. 첫째, SBP estimation 성능은 아직 임상 기준을 충분히 만족하지 못한다. TCN-CBAM은 SBP에서 비교 모델 중 가장 좋은 성능을 보였지만, MAE 5.3482 mmHg와 STD 8.3410 mmHg로 AAMI 기준인 5 ± 8 mmHg에 약간 미달한다. BHS 기준에서도 SBP는 5 mmHg 이내 48.74%, 10 mmHg 이내 73.19%, 15 mmHg 이내 87.06%로 Grade A, B, C 기준을 모두 충족하지 못한다. 따라서 “임상 적용 가능”보다는 “기존 모델 대비 개선 가능성을 보인 연구”로 해석하는 것이 더 정확하다.

둘째, 데이터 분할 방식에 대한 정보가 부족하다. 논문은 데이터를 첫 70%, 다음 10%, 마지막 20%로 나누었다고 설명하지만, subject-independent split인지 명확하지 않다. 혈압 추정에서는 같은 환자의 여러 segment가 training set과 test set에 동시에 들어갈 경우, 모델이 개인별 waveform 특성을 학습하여 실제 새로운 환자에 대한 일반화 성능보다 높은 성능을 보일 수 있다. 이 문제는 cuff-less BP estimation 연구에서 매우 중요한 검증 포인트이다.

셋째, extreme blood pressure 범위를 제외한 점은 모델 성능 해석에 영향을 줄 수 있다. 논문은 SBP 80~180 mmHg, DBP 60~130 mmHg 범위 밖의 segment를 제거했다. 이는 일반적인 PPG-BP 관계 학습에는 도움이 될 수 있지만, 실제 임상에서는 매우 낮거나 높은 혈압 상태를 감지하는 것이 중요할 수 있다. 따라서 모델이 hypotension이나 severe hypertension 같은 극단적 상황에서 잘 작동하는지는 확인되지 않았다.

넷째, CBAM의 실제 기여를 더 명확히 보이기 위한 세부 ablation이 제한적이다. TCN과 TCN-CBAM 비교는 제공되지만, channel attention만 사용한 경우, spatial attention만 사용한 경우, PPG만 사용한 경우, PPG+PPG’만 사용한 경우, PPG+PPG’+PPG’’를 사용한 경우 등 세밀한 분석은 추출 텍스트에 제시되지 않는다. 이러한 ablation이 있어야 어떤 구성 요소가 성능 향상에 가장 크게 기여했는지 더 명확히 알 수 있다.

다섯째, MAPE 수식 표기와 일부 구현 정보에 불명확성이 있다. 일반적인 MAPE는 절대값을 사용하지만, 논문에서 제시된 식에는 절대값 기호가 보이지 않는다. 또한 PyTorch 버전이 “13.1”로 표기되어 있고, activation function 설명에서 ELU와 ReLU가 모두 등장하는 등 일부 세부 사항이 일관되지 않다. 이러한 부분은 재현성 측면에서 원문 코드나 supplementary material 확인이 필요하다.

여섯째, real-world robustness 검증이 부족하다. 논문은 UCI/MIMIC-II 기반 public dataset을 사용하고, 전처리로 baseline wander를 제거한다. 그러나 실제 wearable 환경에서는 motion artifact, sensor pressure variation, skin tone, peripheral perfusion, ambient light, irregular sampling, device-specific noise 등 다양한 문제가 존재한다. 논문 결론에서도 noisy or irregularly sampled data에 대한 robustness 향상이 향후 연구로 제시된다. 따라서 실제 착용형 기기에 바로 적용하기 위해서는 추가 검증이 필요하다.

비판적으로 종합하면, 이 논문은 TCN-CBAM이라는 합리적인 구조를 통해 single-channel PPG 기반 혈압 추정 성능을 개선했지만, 임상 기준 충족 측면에서는 DBP에 비해 SBP가 부족하다. 또한 subject-wise external validation과 더 엄격한 ablation study가 필요하다. 그럼에도 TCN과 attention mechanism을 결합한 방향은 PPG 기반 continuous BP estimation에서 의미 있는 연구 방향으로 볼 수 있다.

## 6. 결론

이 논문은 single-channel PPG를 이용한 cuff-less beat-to-beat blood pressure estimation을 위해 TCN과 CBAM을 결합한 TCN-CBAM 모델을 제안했다. 모델은 PPG, PPG’, PPG’’를 입력으로 사용하고, dilated causal convolution을 통해 긴 시간 의존성을 학습하며, CBAM을 통해 중요한 channel과 time position에 더 집중한다. 최종적으로 fully connected layer와 output layer를 통해 SBP와 DBP를 예측한다.

실험은 UCI Cuff-Less Blood Pressure Estimation Data Set에서 수행되었으며, 전처리 후 270,488개의 sample을 사용했다. TCN-CBAM은 SBP estimation에서 MAE 5.3482 mmHg, STD 8.3410 mmHg, MAPE 3.34%를 기록했고, DBP estimation에서 MAE 2.1190 mmHg, STD 3.1795 mmHg, MAPE 2.40%를 기록했다. 비교 대상인 linear regression, SVR, MLP, XGBoost, random forest, KNN, CNN, CNN-GRU, CNN-LSTM, TCN보다 전반적으로 우수한 성능을 보였다.

이 연구의 주요 기여는 PPG 기반 연속 혈압 추정에서 TCN의 dilated causal convolution과 CBAM attention mechanism을 결합하여 time-series dependency와 feature importance를 동시에 학습한 점이다. 특히 single-channel PPG만 사용하기 때문에 wearable healthcare system으로 확장될 가능성이 있다. DBP estimation에서는 AAMI 기준을 만족할 정도의 성능을 보였다는 점도 의미가 있다.

그러나 SBP estimation은 아직 AAMI 기준에 약간 미달하고, BHS 기준에서도 충분한 등급을 달성하지 못한다. 또한 subject-independent validation 여부가 명확하지 않고, 실제 wearable 환경에서의 noise robustness 검증도 부족하다. 따라서 이 모델은 실제 임상 적용 단계라기보다는, single-channel PPG 기반 continuous BP estimation에서 TCN-attention 구조의 가능성을 보여준 연구로 보는 것이 타당하다.

향후 연구에서는 더 다양한 인구집단과 외부 데이터셋에서의 검증, subject-wise split을 사용한 엄격한 평가, motion artifact와 irregular sampling에 대한 robustness 향상, channel attention과 spatial attention의 개별 효과 분석, 추가 biometric parameter와의 결합 등이 필요하다. 이러한 후속 연구가 보완된다면, 본 논문의 접근법은 non-invasive continuous blood pressure monitoring과 wearable cardiovascular health management 분야에서 중요한 기반 기술로 발전할 가능성이 있다.
