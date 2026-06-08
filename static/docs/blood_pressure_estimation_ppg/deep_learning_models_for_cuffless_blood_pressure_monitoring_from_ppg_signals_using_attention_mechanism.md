# Deep learning models for cuffless blood pressure monitoring from PPG signals using attention mechanism

* **저자**: C El-Hajj, PA Kyriacou
* **발표연도**: 2021

## 1. 논문 개요

이 논문은 단일 PPG 신호만을 이용하여 cuffless, non-invasive, continuous blood pressure monitoring을 수행하기 위한 딥러닝 기반 혈압 추정 모델을 제안한다. 구체적으로, PPG waveform에서 추출한 time-domain features를 입력으로 사용하고, bidirectional recurrent neural network와 attention mechanism을 결합하여 systolic blood pressure, 즉 SBP와 diastolic blood pressure, 즉 DBP를 동시에 추정한다.

연구의 출발점은 고혈압이 cardiovascular disease의 주요 위험 요인이라는 점이다. Blood pressure는 심혈관계 상태를 평가하는 핵심 vital sign 중 하나이며, 고혈압은 장기 손상과 심혈관 질환의 위험을 증가시킨다. 그러나 임상에서 흔히 사용하는 cuff 기반 oscillometry나 auscultation 방식은 간헐적 측정만 가능하고, cuff 압박으로 인한 불편함이 있으며, 장기간 연속 모니터링에 적합하지 않다. Invasive arterial blood pressure 측정은 연속적이고 정확하지만 catheterization이 필요하므로 병원 환경에 제한된다.

이러한 한계를 해결하기 위해 cuffless BP estimation이 연구되어 왔다. 대표적인 방법으로 ECG와 PPG를 함께 사용하여 pulse transit time, 즉 PTT나 pulse arrival time, 즉 PAT를 계산하는 접근이 있다. 하지만 PTT와 PAT는 두 개 이상의 센서를 동시에 사용해야 하고, 신호 정렬과 전처리가 복잡하며, 개인별 생리적 특성에 따라 calibration이 필요하다는 문제가 있다. 이 논문은 이러한 복잡성을 줄이기 위해 ECG 없이 단일 PPG sensor만 사용하는 접근을 선택한다.

논문의 핵심 연구 문제는 다음과 같이 정리할 수 있다. PPG waveform의 형태적 특징만으로 SBP와 DBP를 충분히 정확하게 추정할 수 있는가, 그리고 PPG feature와 BP 사이의 복잡한 비선형 관계 및 시간적 변화를 recurrent neural network와 attention mechanism으로 더 잘 모델링할 수 있는가이다. 저자들은 기존의 multilinear regression, feedforward neural network, conventional LSTM, conventional GRU보다 proposed bidirectional RNN with attention 구조가 더 높은 정확도를 보인다고 주장한다.

문제의 중요성은 실제 웨어러블 헬스케어 응용과 직접 연결된다. 단일 PPG 센서만으로 혈압을 연속적으로 추정할 수 있다면, 기존 cuff 기반 혈압계보다 착용성이 높고, 비용이 낮으며, 장기간 모니터링이 가능하다. 이는 고혈압의 조기 발견, cardiovascular disease 예방, home healthcare, wearable monitoring system에 중요한 기반 기술이 될 수 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 PPG waveform에서 추출한 특징들이 혈압과 단순 선형 관계를 갖지는 않지만, 충분히 강한 생리적 정보를 포함하고 있으며, 이를 딥러닝 모델이 비선형적이고 시간적인 방식으로 학습할 수 있다는 것이다. PPG는 혈관 내 blood volume change를 반영하는 optical signal이며, waveform의 systolic upstroke time, diastolic time, pulse width, pulse area 등은 vascular tone, systemic vascular resistance, cardiac cycle duration과 관련될 수 있다. 따라서 이러한 특징들을 적절히 추출하면 SBP와 DBP 추정에 사용할 수 있다.

기존 PPG 기반 혈압 추정 연구들은 linear regression, support vector machine, random forest, feedforward neural network 등을 사용했다. 그러나 저자들은 이러한 접근이 세 가지 측면에서 제한적이라고 본다. 첫째, BP와 PPG feature 사이의 관계는 명확한 선형 관계가 아니므로 multilinear regression만으로는 충분하지 않다. 둘째, classical machine learning 모델들은 SBP와 DBP를 별도 모델로 추정하는 경우가 많지만, DBP와 SBP는 서로 관련되어 있으므로 하나의 모델 구조에서 동시에 추정하는 것이 더 타당할 수 있다. 셋째, feedforward neural network는 현재 입력 feature vector만 보고 예측하기 때문에, 시간에 따른 PPG feature 변화와 장기적 temporal dependency를 충분히 반영하지 못한다.

논문이 제안하는 차별점은 bidirectional recurrent structure와 attention mechanism을 PPG-only blood pressure estimation에 결합한 것이다. Bidirectional RNN은 입력 sequence를 앞에서 뒤로, 뒤에서 앞으로 모두 처리함으로써 과거 정보뿐 아니라 근접한 미래 정보도 함께 활용한다. PPG feature sequence가 7초 window 단위로 구성되기 때문에, 특정 time step의 정보만이 아니라 window 전체에서 혈압 추정에 중요한 패턴을 찾을 수 있다.

Attention mechanism은 RNN의 마지막 hidden state sequence 중 혈압 추정에 더 중요한 time step에 더 큰 가중치를 부여한다. 즉 모든 time step을 동일하게 취급하지 않고, SBP와 DBP 추정에 더 관련 있는 hidden state를 모델이 자동으로 선택하도록 한다. 저자들은 이러한 attention layer가 information search space를 줄이고, target output에 중요한 부분에 모델이 집중하도록 하여 정확도를 개선한다고 설명한다.

논문에서 명확히 제시된 주요 기여는 두 가지이다. 첫째, LSTM과 GRU unit을 사용한 bidirectional recurrent structures를 PPG 기반 cuffless BP estimation에 도입했다. 둘째, attention mechanism을 추가하여 모델 성능과 정확도를 더 개선했다. 저자들은 Bi-LSTM 또는 Bi-GRU와 attention mechanism의 결합이 단일 PPG sensor 기반 BP estimation에 적용된 것은 처음이라고 주장한다.

## 3. 상세 방법 설명

논문의 전체 파이프라인은 MIMIC II 데이터셋에서 PPG와 ABP signal을 수집하고, PPG signal을 전처리한 뒤, PPG waveform에서 22개의 time-domain features를 추출하고, feature reduction을 통해 7개 feature subset도 만든 다음, 여러 machine learning 및 deep learning regression model을 학습하고 평가하는 흐름으로 구성된다.

데이터셋은 Multiparameter Intelligent Monitoring in Intensive Care II, 즉 MIMIC II를 사용했다. 이 데이터셋은 ICU bedside에서 동시에 기록된 여러 physiological signals를 포함하며, 이 논문에서는 PPG와 invasive arterial blood pressure, 즉 ABP signal만 사용했다. 총 500개의 waveform set이 수집되었고, 모든 signal은 125 Hz로 sampling되었다. Reference SBP와 DBP는 ABP signal에서 추출되었다. SBP는 ABP pulse waveform의 systolic peak, DBP는 같은 waveform cycle의 end diastole 값에 해당한다. 최종 데이터셋의 SBP 범위는 83.2에서 182.7 mmHg, 평균은 144.5 mmHg, 표준편차는 14.1 mmHg이다. DBP 범위는 60.4에서 112.1 mmHg, 평균은 72.7 mmHg, 표준편차는 6.5 mmHg이다.

전처리 단계에서는 PPG signal의 품질을 높이고 feature extraction을 안정화하기 위해 여러 절차를 수행했다. 먼저 PPG denoising을 위해 4th order, 51 frame length의 Savitzky-Golay filter를 적용했다. 이 필터는 moving average와 polynomial fitting을 함께 사용하면서 sharp edge를 비교적 잘 보존하는 장점이 있다. 이후 low-frequency baseline wandering을 제거했다. 이는 PPG peak와 foot point를 명확하게 잡아야 segmentation과 feature extraction이 가능하기 때문이다. 다음으로 PPG와 BP signal 사이의 작은 delay를 제거하기 위해 두 신호를 synchronisation했다. 마지막으로 PPG amplitude를 min-max normalisation을 통해 $[0,1]$ 범위로 정규화했다.

전처리 후 PPG와 BP reference는 7초 window로 segmentation되었다. 7초 window를 사용한 이유는 각 segment가 충분한 peak 또는 heartbeat를 포함하여 recurrent neural network가 time series data를 처리하는 능력을 활용할 수 있도록 하기 위함이다. 이 단계에서 저자들은 manual check를 수행하여 motion artifact가 심하거나, irregular하거나, distorted된 segment, 또는 BP reference segment가 누락된 PPG segment를 제외했다. 또한 매우 높거나 낮은 BP와 heart rate를 가진 segment도 제거했다. 최종적으로 9000개 이상의 good quality segment가 feature extraction에 사용되었다.

Feature extraction은 PPG waveform contour에서 수행되었다. 논문은 PPG waveform의 dicrotic notch가 모든 개인에게 명확히 보이지 않는다는 점을 중요하게 다룬다. 이상적인 waveform에서는 dicrotic notch가 명확하여 관련 feature를 쉽게 추출할 수 있지만, 실제 데이터에서는 dicrotic notch가 흐릿하거나 완전히 보이지 않을 수 있다. 첫 번째 또는 두 번째 derivative를 사용하면 일부 inflection point를 추정할 수 있지만, 계산 복잡도가 증가하고 모든 환자에게 안정적으로 적용되기 어렵다. 따라서 이 논문은 PPG의 first derivative나 second derivative feature를 제외하고, waveform contour에서 직접 추출 가능한 time-domain feature를 사용한다.

저자들은 Kurylyak et al.에서 사용된 21개 time-domain PPG features에 pulse area를 추가하여 총 22개 features를 사용했다. 각 PPG cycle은 두 valley 사이에 하나의 peak를 포함하는 단위로 정의된다. 추출된 feature는 cardiac period, systolic time, diastolic time, pulse area, 그리고 여러 amplitude level에서의 systolic width와 diastolic width 관련 feature로 구성된다. 예를 들어 10%, 25%, 33%, 50%, 66%, 75% amplitude level에서 $DW$, $SW + DW$, $DW/SW$가 계산된다. 여기서 $SW$는 systolic width, $DW$는 diastolic width를 의미한다. Pulse area는 diastolic area와 systolic area의 비율로 정의되며 vascular tone change와 관련되어 BP와 직접적으로 관련될 수 있다고 설명된다.

정규화는 feature scale 차이를 줄이기 위해 수행되었다. 서로 다른 PPG features는 값의 범위가 크게 다를 수 있는데, 정규화하지 않으면 machine learning model이 값이 큰 feature를 더 중요하다고 잘못 해석할 수 있다. 논문에서는 전체 데이터셋에 대해 train, validation, test split 전에 min-max normalisation을 수행했다고 명시한다. 정규화 식은 다음과 같다.

$$
X_{norm} = \frac{x - min(x)}{max(x) - min(x)}
$$

이 식은 각 feature 값을 해당 feature의 최소값과 최대값을 기준으로 $[0,1]$ 범위에 매핑한다. 쉬운 말로 설명하면, 모든 입력 feature를 같은 숫자 범위로 맞춰 모델이 특정 feature scale에 편향되지 않게 하는 과정이다. 다만 전체 데이터셋을 분할 전에 정규화했다는 점은 엄밀한 machine learning 평가 관점에서는 data leakage 가능성이 있다. 논문은 이를 계산 효율성과 feature scale 정렬의 관점에서 설명하지만, test set 정보가 정규화 과정에 사용될 수 있다는 비판적 해석이 가능하다.

Feature dimensionality reduction을 위해 네 가지 방법이 사용되었다. Pearson’s correlation coefficient는 각 feature와 target 사이의 선형 상관을 평가하여 더 높은 값을 가진 feature를 유지한다. Random forest feature importance는 random forest regression model을 학습한 뒤 BP estimation에 더 크게 기여하는 feature를 추정한다. Forward feature selection은 feature를 하나씩 추가하면서 성능 개선이 더 이상 발생하지 않을 때 멈춘다. Recursive feature elimination은 학습 모델의 성능을 바탕으로 target output에 영향이 큰 feature subset을 선택한다. 이 과정을 종합적으로 분석한 결과, 7개 feature가 선택되었다. 선택된 feature는 cardiac period, diastolic time, $DW10/SW10$, $DW25$, $SW33 + DW33$, $DW75$, $SW75 + DW75$이다. 따라서 논문은 22-feature dataset과 7-feature dataset 두 가지를 만들어 실험했다.

모델 측면에서는 네 가지 proposed recurrent neural network model을 제안하고, 네 가지 기존 모델과 비교했다. 비교 모델은 multilinear regression, feedforward neural network 또는 MLP, conventional LSTM, conventional GRU이다. 제안 모델은 크게 두 가지 architecture로 구성된다. 첫 번째 architecture는 stacked Bi-RNN layers 뒤에 attention layer를 붙인 구조이다. 두 번째 architecture는 가장 아래에 하나의 Bi-RNN layer를 두고, 그 위에 하나 이상의 unidirectional RNN layer를 쌓은 뒤, 마지막에 attention layer를 적용하는 구조이다. 각 architecture에서 RNN unit은 LSTM 또는 GRU로 바꿔 실험했으므로, Bi-LSTM + attention, Bi-LSTM + LSTM + attention, Bi-GRU + attention, Bi-GRU + GRU + attention의 네 가지 proposed model이 된다.

Multilinear regression은 PPG feature와 BP 사이의 선형 관계를 가정한다. 논문에서 제시한 식은 다음과 같다.

$$
Y = X_1 \beta_1 + X_2 \beta_2 + \cdots + X_n \beta_n + \epsilon
$$

여기서 $Y$는 SBP 또는 DBP이고, $X_1, X_2, \ldots, X_n$은 PPG features이며, $\beta$는 각 feature에 대한 regression coefficient, $\epsilon$은 intercept이다. 이 모델은 SBP와 DBP를 각각 별도의 모델로 추정해야 한다. 논문은 BP와 PPG feature 관계가 강한 비선형성을 갖기 때문에 MLR이 특히 SBP 추정에서 낮은 성능을 보였다고 설명한다.

Feedforward neural network, 즉 MLP는 입력 feature가 hidden layer를 거쳐 output layer로 전달되는 구조이다. Hidden layer에서는 nonlinear activation function을 사용하고, output layer에서는 SBP와 DBP를 병렬로 추정한다. 각 neuron의 activation은 다음과 같이 계산된다.

$$
a = \sum_i x_i w_i + b
$$

여기서 $x_i$는 입력 feature, $w_i$는 weight, $b$는 bias이다. 논문에서는 sigmoid나 tanh보다 ReLU가 더 좋은 성능을 보였다고 하며, ReLU는 다음과 같이 정의된다.

$$
F(a) = max(0, a)
$$

MLP는 비선형 mapping을 학습할 수 있다는 장점이 있지만, sequence history를 직접 모델링하지 않기 때문에 PPG feature의 시간적 변화를 반영하는 데 제한이 있다.

LSTM은 recurrent neural network의 한 종류로, long-term dependency를 학습하기 위해 memory cell과 gate mechanism을 사용한다. 논문에서 LSTM의 hidden state는 다음 식들로 설명된다.

$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
$$

$$
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
$$

$$
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
$$

$$
\hat{c}*t = tanh(W_c x_t + U_c h*{t-1} + b_c)
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot \hat{c}_t
$$

$$
h_t = o_t \odot tanh(c_t)
$$

여기서 $f_t$, $i_t$, $o_t$는 각각 forget gate, input gate, output gate이다. Forget gate는 이전 정보를 얼마나 버릴지 결정하고, input gate는 새로운 정보를 얼마나 cell state에 추가할지 결정하며, output gate는 현재 cell state 중 얼마나 hidden state로 출력할지 결정한다. $\hat{c}_t$는 새로 추가될 candidate cell state이며, $\odot$는 element-wise multiplication을 의미한다. 쉬운 말로 설명하면, LSTM은 이전 시점의 정보를 기억하거나 잊는 과정을 gate로 조절하면서 PPG feature sequence의 시간적 변화를 학습한다.

GRU는 LSTM과 유사한 recurrent unit이지만 구조가 더 단순하다. 별도의 memory cell이 없고 reset gate와 update gate 두 개만 사용한다. 논문에서 GRU는 다음 식으로 설명된다.

$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$

$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$

$$
\hat{h}*t = tanh(W_h x_t + U_h(r_t \odot h*{t-1}) + b_h)
$$

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \hat{h}_t
$$

여기서 $r_t$는 reset gate, $z_t$는 update gate이다. Reset gate는 이전 hidden state를 얼마나 반영할지 조절하고, update gate는 이전 hidden state와 새로운 candidate activation 사이의 비율을 조절한다. 논문은 GRU가 LSTM보다 계산량이 적고 더 빠르게 학습될 수 있으며, 많은 task에서 경쟁력 있는 성능을 낸다고 설명한다.

Bidirectional RNN은 sequence를 forward direction과 backward direction으로 모두 처리한다. 일반적인 LSTM이나 GRU는 과거에서 현재 방향으로만 정보를 전달하므로 과거 정보만 활용한다. 그러나 7초 PPG feature sequence 전체가 이미 주어진 상황에서는 가까운 미래 time step의 정보도 현재 time step의 해석에 도움이 될 수 있다. Bidirectional structure는 forward hidden state와 backward hidden state를 결합하여 최종 hidden state를 만든다. 논문에서 제시한 식은 다음과 같다.

$$
h_t = W_f h_t^f + W_b h_t^b + b_h
$$

$$
h_t^f = f(W_h^f x_t + U_h^f h_{t-1}^f + b^f)
$$

$$
h_t^b = f(W_h^b x_t + U_h^b h_{t+1}^b + b^b)
$$

여기서 $h_t^f$는 forward hidden state, $h_t^b$는 backward hidden state이다. 함수 $f$는 Bi-LSTM의 경우 LSTM update equations, Bi-GRU의 경우 GRU update equations에 해당한다.

Attention mechanism은 RNN output hidden states 중 target estimation에 더 중요한 hidden state에 더 높은 weight를 부여한다. 논문에서 context vector $v_t$는 hidden states의 weighted sum으로 계산된다.

$$
v_t =
\sum_{i=1}^{t-1}
\alpha_{t,i} h_i
$$

Attention weight $\alpha_{t,i}$는 다음과 같이 softmax 형태로 계산된다.

$$
\alpha_{t,i} = align(y_t, x_i) =
\frac{exp(f(h_i, h_t))}
{\sum_{j=1}^{t-1} exp(f(h_j, h_t))}
$$

여기서 $\alpha_{t,i}$는 입력 sequence의 $i$번째 위치와 target output $t$ 사이의 중요도를 나타낸다. $f$는 tanh activation을 사용하는 single-layer feedforward neural network로 설명된다. 직관적으로 말하면, attention layer는 7초 window 안의 모든 time step을 같은 비중으로 보지 않고, SBP와 DBP 추정에 더 중요한 hidden state를 크게 반영하여 최종 context vector를 만든다. 이후 SBP와 DBP는 linear activation function을 통해 동시에 추정된다.

학습과 평가는 60% train, 20% validation, 20% test split으로 수행되었다. Validation set은 최적화된 모델 선택에 사용되었고, test set의 성능이 결과로 보고되었다. Batch size는 64이며, 모델은 early stopping을 사용하여 최대 200 epochs까지 학습되었다. Optimizer는 Adam을 사용했고, 논문은 Adam이 RMSE optimizer와 gradient descent optimizer보다 더 안정적이고 좋은 성능을 보였다고 설명한다. Loss function은 mean squared error, 즉 MSE이다. MSE는 예측 혈압과 reference 혈압의 차이를 제곱해 평균한 값으로, 큰 오차에 더 큰 penalty를 주는 회귀 손실 함수이다. 논문에 MSE의 명시적 수식은 제공되지 않았지만, 일반적으로 다음과 같이 이해할 수 있다.

$$
MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

여기서 $y_i$는 reference SBP 또는 DBP이고, $\hat{y}_i$는 모델의 추정값이다. 논문은 SBP와 DBP를 동시에 추정한다고 설명하지만, MSE가 두 출력에 대해 구체적으로 어떻게 합산되는지에 대한 세부 수식은 제공된 텍스트에 명확히 제시되어 있지 않다.

## 4. 실험 및 결과

실험은 22-feature dataset과 7-feature dataset 두 가지 입력 설정에서 수행되었다. 모든 모델은 SBP와 DBP를 추정했고, 성능은 mean absolute error, 즉 MAE와 error standard deviation, 즉 SD로 평가되었다. 제안 모델 중 가장 좋은 모델에 대해서는 error histogram과 Bland-Altman plot도 분석되었다. 또한 AAMI 기준과 비교하여 cuffless BP estimation 성능이 국제적 기준에 부합하는지 평가했다.

22-feature set 실험에서 multilinear regression은 가장 낮은 성능을 보였다. SBP MAE는 11 mmHg, DBP MAE는 4.04 mmHg였다. 이는 PPG feature와 BP 사이의 관계가 단순 선형으로 설명되기 어렵다는 저자들의 주장을 뒷받침한다. MLP는 SBP MAE 4.1 mmHg, DBP MAE 2.17 mmHg로 MLR보다 크게 개선되었다. LSTM은 SBP MAE 3.02 mmHg, DBP MAE 1.44 mmHg를 보였고, GRU는 SBP MAE 3.1 mmHg, DBP MAE 1.61 mmHg를 보였다. 이는 recurrent network가 PPG feature의 시간적 변화를 반영함으로써 feedforward network보다 더 좋은 성능을 낼 수 있음을 보여준다.

제안 모델들은 22-feature set에서 모든 baseline보다 더 좋은 결과를 보였다. Bi-LSTM + attention은 SBP MAE 2.66 mmHg, DBP MAE 1.26 mmHg를 기록했다. Bi-LSTM + LSTM + attention은 SBP MAE 2.73 mmHg, DBP MAE 1.51 mmHg였다. Bi-GRU + attention은 SBP MAE 2.62 mmHg, DBP MAE 1.31 mmHg를 보였다. 가장 좋은 모델은 Bi-GRU + GRU + attention으로, SBP MAE 2.58 mmHg, SD 3.35 mmHg, DBP MAE 1.26 mmHg, SD 1.63 mmHg를 달성했다. 이 모델은 하나의 Bi-GRU layer 뒤에 두 개의 unidirectional GRU layer와 attention layer를 결합한 구조이다.

22-feature set에서 error histogram은 SBP와 DBP error가 0을 중심으로 비교적 좁게 분포함을 보여준다. SBP error range는 대략 $[-10,10]$ mmHg, DBP error range는 대략 $[-5,5]$ mmHg로 설명된다. 논문은 SBP의 값 범위가 DBP보다 크기 때문에 SBP error distribution이 더 넓다고 해석한다. Bland-Altman plot에서는 SBP의 limits of agreement가 약 $[-8.79, 7.76]$ mmHg, DBP의 limits of agreement가 약 $[-4.72, 3.41]$ mmHg로 보고되었다. 대부분의 error가 95% confidence interval의 agreement limits 안에 있었고, mean error가 작아 bias가 낮다고 설명된다.

7-feature set 실험에서도 제안 모델은 baseline보다 우수했다. MLR은 SBP MAE 12.14 mmHg, DBP MAE 4.54 mmHg로 낮은 성능을 보였다. MLP는 SBP MAE 4.23 mmHg, DBP MAE 2.37 mmHg였다. LSTM은 SBP MAE 3.23 mmHg, DBP MAE 1.59 mmHg, GRU는 SBP MAE 3.25 mmHg, DBP MAE 1.43 mmHg였다. 제안 모델들은 모두 SBP MAE 약 3 mmHg 수준, DBP MAE 약 1.3에서 1.45 mmHg 수준을 보였다. 가장 좋은 모델은 Bi-GRU + attention으로, SBP MAE 2.9 mmHg, SD 3.94 mmHg, DBP MAE 1.31 mmHg, SD 1.76 mmHg를 기록했다.

7-feature set의 결과는 두 가지 중요한 의미를 가진다. 첫째, 22개의 전체 feature를 사용하면 가장 높은 정확도를 얻을 수 있지만, 7개의 선택된 feature만으로도 경쟁력 있는 성능을 달성할 수 있다. 둘째, dimensionality reduction을 통해 계산 복잡도를 줄이면서도 성능 손실을 제한할 수 있다. 이는 웨어러블 장치나 저전력 환경에서 실제 추론 비용을 줄이는 데 중요할 수 있다.

AAMI 기준과의 비교도 논문의 중요한 결과이다. AAMI 기준에 따르면 reference device와 estimation device 사이의 mean error difference는 5 mmHg 이하, standard deviation은 8 mmHg 이하이어야 한다. 논문에서 proposed models는 22-feature set에서 SBP mean error -0.52 mmHg, SD 4.22 mmHg, DBP mean error -0.66 mmHg, SD 2.07 mmHg를 보였다. 7-feature set에서는 SBP mean error -0.05 mmHg, SD 5.12 mmHg, DBP mean error -0.15 mmHg, SD 2.14 mmHg를 보였다. 따라서 두 dataset 모두 AAMI 기준의 mean error와 SD 조건을 만족한다고 보고된다.

관련 연구와의 비교에서 제안 모델은 PPG-only 접근 및 PTT/PAT 기반 접근들과 비교되었다. 논문의 best model인 Bi-GRU + GRU + attention은 DBP MAE 1.26 mmHg, SD 1.63 mmHg, SBP MAE 2.58 mmHg, SD 3.35 mmHg를 보였다. Ruiz-Rodriguez et al.의 PPG 기반 DBRBM은 SD가 AAMI 기준을 만족하지 못했고, Liu et al.의 PPG 기반 SVM 역시 SBP와 DBP의 SD가 크다고 보고되었다. Kurylyak et al.의 ANN 기반 PPG 접근은 좋은 결과를 보였지만, 이 논문은 recurrent structure와 attention을 통해 더 적은 feature set에서도 경쟁력 있는 성능을 보였다고 설명한다. PTT/PAT 기반 방법들은 일부 좋은 성능을 보였지만 ECG와 PPG 두 센서가 필요하고, 신호 alignment와 calibration 문제가 있다는 점에서 단일 PPG approach보다 실용성이 떨어질 수 있다고 논문은 해석한다.

## 5. 강점, 한계

이 논문의 첫 번째 강점은 단일 PPG sensor만으로 SBP와 DBP를 동시에 추정하는 deep learning architecture를 제안했다는 점이다. ECG와 PPG를 함께 사용하는 PTT 또는 PAT 기반 방법은 두 센서의 동기화와 정렬이 필요하고 착용성이 떨어질 수 있다. 반면 PPG-only 방식은 optical, inexpensive, wearable sensor에 쉽게 통합될 수 있어 cuffless continuous BP monitoring의 실제 적용 가능성이 높다.

두 번째 강점은 PPG feature와 BP 사이의 비선형성과 시간적 변화를 모두 고려했다는 점이다. MLR의 낮은 성능은 선형 가정이 충분하지 않음을 보여주고, LSTM과 GRU가 MLP보다 높은 성능을 보인 것은 sequence modeling이 혈압 추정에 도움이 된다는 근거를 제공한다. 또한 bidirectional connection과 attention mechanism을 결합한 proposed models가 conventional LSTM과 GRU보다 더 좋은 성능을 보였다는 점은, past와 future context 및 중요한 time step 선택이 PPG-based BP estimation에 유용할 수 있음을 보여준다.

세 번째 강점은 22-feature set과 7-feature set을 모두 평가하여 accuracy와 computational complexity 사이의 trade-off를 분석했다는 점이다. 전체 22개 feature를 사용하면 가장 높은 정확도를 얻지만, 7개 selected features만으로도 AAMI 기준을 만족하는 성능을 얻었다. 이는 실제 장치에서 연산 비용과 feature extraction 부담을 줄이는 데 유용하다.

네 번째 강점은 AAMI standard를 기준으로 모델 성능을 평가했다는 점이다. 단순히 MAE가 낮다는 결과만 제시하는 것이 아니라, mean error와 SD가 cuffless BP estimation에 요구되는 국제적 기준을 만족하는지 확인했다. 이는 의료기기적 관점에서 결과의 의미를 해석하는 데 중요하다.

그러나 한계도 분명하다. 첫째, 이 접근은 PPG feature extraction의 정확성에 크게 의존한다. 논문은 dicrotic notch가 모든 사람에게 명확히 나타나지 않으며, 일부 waveform에서는 관련 feature를 안정적으로 추출하기 어렵다고 설명한다. 이를 피하기 위해 derivative features를 제외했지만, 그만큼 arterial stiffness 등 일부 생리적 정보를 포기했을 가능성도 있다.

둘째, manual check에 의존한 signal quality filtering은 실제 시스템 적용에서 부담이 된다. 논문은 motion artifact가 있거나 distorted된 segment를 제외하기 위해 manual checks를 수행했다고 설명한다. 이는 연구 환경에서는 가능하지만, 연속 wearable monitoring 환경에서는 자동화되어야 한다. 만약 실제 환경에서 low-quality segment를 자동으로 잘 걸러내지 못하면 모델 성능이 크게 떨어질 수 있다.

셋째, 데이터셋의 BP 분포가 제한적이다. 논문은 MIMIC database에서 추출한 signal이 mostly normotensive and hypertensive BP categories를 포함하며, hypotensive BP values를 충분히 예측할 수 없다고 명시한다. 따라서 저혈압 환자나 혈압 범위가 더 넓은 실제 환경에 모델을 적용하기 위해서는 hypotensive signal을 추가한 학습과 검증이 필요하다.

넷째, demographic information이 충분히 사용되지 않았다. Age, gender, height, weight 같은 정보는 모든 환자에게 제공되지 않아 제외되었다. 그러나 이러한 개인 특성은 BP와 PPG waveform 관계에 영향을 줄 수 있다. 논문도 demographic information이 model performance와 reliability를 개선할 수 있다고 언급한다. 따라서 현재 모델은 개인별 생리적 차이를 PPG waveform feature만으로 간접적으로 반영해야 한다는 한계가 있다.

다섯째, train, validation, test split 전에 전체 데이터셋을 정규화했다는 점은 엄밀한 관점에서 비판 가능하다. 전체 데이터의 minimum과 maximum을 사용하면 test set의 분포 정보가 training preprocessing에 간접적으로 반영될 수 있다. 이는 성능을 약간 낙관적으로 만들 수 있는 data leakage 가능성을 가진다. 논문은 이를 문제로 다루지 않았지만, 재현 연구나 후속 연구에서는 train set 기준으로 normalisation parameter를 계산하고 validation/test set에 적용하는 방식이 더 적절하다.

여섯째, 모델은 classical machine learning model이나 conventional RNN보다 학습 시간이 더 길고 복잡하다. 논문은 제안 모델이 더 높은 성능을 내지만 training time이 더 필요하다고 인정한다. 특히 attention 기반 Bi-RNN 구조는 실시간 wearable device 내장 모델로 사용하려면 model compression, edge inference optimization, feature extraction automation이 추가로 필요할 수 있다.

비판적으로 보면, 이 논문은 PPG-only BP estimation에서 매우 좋은 수치 성능을 보고하지만, 연구 환경에서 정제된 good-quality segment와 manual filtering에 기반한 결과라는 점을 고려해야 한다. 실제 cuffless BP monitoring은 motion artifact, sensor displacement, skin tone, temperature, vascular disease, medication effect 등 훨씬 복잡한 조건을 포함한다. 따라서 이 논문의 결과는 PPG-only deep learning BP estimation의 가능성을 강하게 보여주지만, 임상적 일반화와 실사용 신뢰성을 보장하기 위해서는 더 엄격한 외부 검증과 실시간 환경 평가가 필요하다.

## 6. 결론

이 논문은 단일 PPG sensor에서 추출한 waveform features만으로 cuffless and continuous BP estimation을 수행하기 위해 Bi-LSTM, Bi-GRU, bidirectional connection, attention mechanism을 결합한 deep learning models를 제안했다. 제안 모델은 MIMIC II 기반 PPG와 ABP signal에서 평가되었으며, 22-feature set과 7-feature set 모두에서 기존 multilinear regression, MLP, conventional LSTM, conventional GRU보다 우수한 성능을 보였다.

가장 좋은 22-feature model은 Bi-GRU + GRU + attention 구조였고, SBP MAE 2.58 mmHg, SD 3.35 mmHg, DBP MAE 1.26 mmHg, SD 1.63 mmHg를 달성했다. 7-feature set에서는 Bi-GRU + attention이 SBP MAE 2.9 mmHg, SD 3.94 mmHg, DBP MAE 1.31 mmHg, SD 1.76 mmHg를 달성했다. 두 feature setting 모두 AAMI 기준의 mean error와 standard deviation 요구 조건을 만족했다.

이 연구의 주요 기여는 PPG-only BP estimation에서 recurrent temporal modeling, bidirectional context, attention-based hidden state selection이 성능 개선에 기여할 수 있음을 보인 것이다. 특히 7개의 selected features만으로도 좋은 성능을 얻었다는 점은 저전력 wearable system이나 간단한 optical sensor 기반 혈압 추정에 실용적 가능성을 제공한다.

향후 연구에서는 hypotensive signal을 포함한 더 균형 잡힌 데이터셋, demographic information의 통합, automatic signal quality assessment, convolutional layer를 활용한 short-term pattern learning, attention weight 분석을 통한 생리적 해석 가능성 강화가 필요하다. 또한 실제 웨어러블 환경에서 motion artifact와 sensor placement variation이 존재할 때도 성능이 유지되는지 검증해야 한다. 그럼에도 이 논문은 cuffless, non-invasive, continuous BP monitoring을 위한 PPG-only deep learning 접근의 중요한 가능성을 보여주는 연구로 평가할 수 있다.
