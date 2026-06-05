# An Efficient Hybrid LSTM-ANN Joint Classification-Regression Model for PPG Based Blood Pressure Monitoring

* **저자**: Noor Faris Ali, Mohamed Atef
* **발표연도**: 2023

## 1. 논문 개요

이 논문은 photoplethysmography, 즉 PPG 신호를 이용한 cuffless continuous Non-Invasive Blood Pressure, 즉 cNIBP 모니터링에서 혈압 회귀 추정 성능을 높이기 위해 classification을 regression 앞단 또는 공동 학습 구조에 결합하는 방법을 제안한다. 논문의 핵심 목표는 단순히 SBP와 DBP 추정 오차를 낮추는 것이 아니라, 실제 microcontroller unit, 즉 MCU와 같은 resource-constrained device에 탑재할 수 있을 정도로 memory usage, latency, model complexity를 제한하면서도 충분한 혈압 예측 정확도를 달성하는 것이다.

연구 문제는 PPG 기반 혈압 추정에서 성능과 효율성 사이의 균형을 어떻게 달성할 것인가이다. 기존 연구들은 CNN, LSTM, 복잡한 deep neural network 또는 수십 개에서 수백 개에 이르는 feature를 사용하여 정확도를 높이는 데 집중해 왔다. 그러나 wearable 또는 standalone cNIBP device에 실제로 탑재하려면 계산량, 메모리 점유율, 추론 지연시간이 매우 중요하다. 이 논문은 이러한 실제 구현 제약을 명시적으로 고려한다는 점에서, 단순한 offline accuracy 경쟁형 연구와 목적이 다르다.

혈압 monitoring 문제의 중요성은 hypertension이 심혈관 질환, stroke, kidney failure, peripheral arterial disease 등 심각한 질환과 관련되며, 초기에는 증상이 잘 드러나지 않는 silent killer라는 점에서 출발한다. 기존 cuff 기반 혈압계는 불편하고 연속 측정에 적합하지 않으며, invasive monitor는 정확하지만 일상적 사용이 어렵다. PPG는 optical, noninvasive, low-cost, lightweight, portable 기술이므로 continuous BP monitoring을 위한 유망한 신호원이다.

논문은 Physionet MIMIC II database에서 선택한 40명 subject의 PPG 및 ABP 기록을 사용하고, normotension, hypertension, hypotension의 세 가지 혈압 class를 고려한다. 제안 모델은 단 4개의 feature만 사용하면서 hybrid LSTM-ANN joint classification-regression 구조를 통해 SBP와 DBP에 대해 각각 MAE ± SD 3.39 ± 5.47 mmHg, 1.79 ± 3.72 mmHg를 달성했다고 보고한다. 또한 single-stage ANN과 two-stage classification-regression ANN을 STM32 MCU에 실제로 embedded하여 real-time BP prediction을 수행했고, real-time testing에서 two-stage model이 traditional single-stage ANN보다 MAE를 83.5% 줄였다고 보고한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 continuous BP 값을 직접 regression하는 대신, 먼저 혈압 상태를 discrete category로 분류하면 regression이 더 쉬워질 수 있다는 것이다. SBP와 DBP는 넓은 범위에서 변동하기 때문에, 하나의 regression model이 전체 혈압 범위를 한 번에 예측하려면 예측 공간이 넓고 불안정해질 수 있다. 반면 classification stage가 현재 혈압이 normotension, hypertension, hypotension 중 어느 범주에 속하는지 먼저 알려주면, regression stage는 더 제한된 범위 안에서 혈압 값을 예측하게 된다. 이로써 예측값이 넓은 interval에서 흔들리는 문제를 줄이고 regression accuracy를 높일 수 있다는 것이 논문의 직관이다.

이 접근은 joint classification-regression이라는 관점에서 설명된다. 연속형 target인 BP 값을 discrete BP category로 변환하고, 이 classification 정보를 regression task와 결합한다. 논문은 이러한 방식이 다른 분야, 예를 들어 remaining useful life prediction 등에서는 사용된 적이 있지만, PPG 기반 BP estimation에서는 기존 연구가 거의 없었다고 주장한다.

또 다른 핵심 아이디어는 hybrid LSTM-ANN 구조이다. PPG feature는 시간에 따라 변하는 time-series 성격을 가지므로 LSTM은 temporal dependency를 모델링하는 데 유리하다. 그러나 복잡한 LSTM architecture는 memory와 연산량이 크기 때문에 MCU 같은 작은 장치에 올리기 어렵다. 이 논문은 하나의 단순한 LSTM layer를 사용해 sequential PPG feature의 시간적 변화를 포착하고, 이후 classification과 regression branch는 비교적 가벼운 ANN 구조로 구성하여 efficiency를 유지하려 한다.

기존 연구와의 차별점은 세 가지로 정리할 수 있다. 첫째, 많은 feature를 사용하는 대신 6개 extracted feature 중 4개 selected feature만 사용하여 모델을 설계한다. 둘째, 단순 regression model뿐 아니라 single-stage, two-stage, joint learning, hybrid LSTM-ANN joint model 등 여러 topology를 비교하여 classification의 기여를 검증한다. 셋째, offline evaluation에 그치지 않고 selected ANN-based model을 STM32 microcontroller에 실제 배포하여 real-time prediction 성능과 resource requirement를 평가한다.

## 3. 상세 방법 설명

논문의 방법론은 크게 두 단계로 구성된다. 첫 번째 단계는 single-stage model, two-stage classification-regression model, joint classification-regression model, hybrid LSTM-ANN joint model을 설계하고 학습 및 평가하는 단계이다. 두 번째 단계는 two-stage ANN model과 single-stage ANN model을 STM32 MCU에 배포하여 real-time BP prediction을 검증하는 단계이다.

### 3.1 데이터베이스 선택과 라벨 구성

이 연구는 Physionet MIMIC II database에서 40명의 subject 데이터를 선택하여 사용한다. 각 subject에 대해 2분 길이의 PPG recording과 이에 대응하는 ABP recording이 제공된다. 논문은 high-quality PPG signal을 선택했다고 설명하지만, 제공된 텍스트에서는 signal quality selection 기준이나 자동/수동 판정 방식의 세부 내용은 확인되지 않는다.

ABP signal로부터 SBP와 DBP 값을 추출한다. Classification model을 학습하기 위해 추출된 SBP와 DBP 값은 세 개의 discrete class label로 mapping된다. Class 0은 normotension, class 1은 hypertension, class 2는 hypotension을 의미한다. 데이터의 class distribution은 각각 40%, 40%, 20%로 보고된다. 즉 normotension과 hypertension이 각각 전체의 40%이고, hypotension이 20%를 차지한다.

이 class label은 regression을 직접 대체하기 위한 것이 아니라, regression 정확도를 향상시키기 위한 보조 task 또는 선행 stage로 사용된다. 따라서 이 논문의 target은 두 종류이다. 하나는 혈압 상태 class를 예측하는 classification target이고, 다른 하나는 실제 SBP와 DBP 값을 예측하는 regression target이다.

### 3.2 PPG signal preprocessing

PPG 신호는 out-of-band noise에 취약하기 때문에 feature extraction 전에 filtering이 수행된다. 논문은 MATLAB filter design tool을 사용하여 16th order bandpass filter를 설계했다고 설명한다. High-pass cut-off frequency는 0.5 Hz이고, low-pass cut-off frequency는 5 Hz이다. 이 filter는 raw PPG signal에 적용되어 clean PPG signal을 얻는 데 사용된다.

다음으로 PPG amplitude normalization이 수행된다. 논문은 min-max normalization을 사용해 PPG 신호의 amplitude를 0부터 1까지의 predefined range로 정규화한다고 설명한다. 제공된 텍스트에서 수식은 중간에 끊겨 있지만, 일반적인 min-max normalization의 형태는 다음과 같이 해석할 수 있다.

$$
X_n = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
$$

여기서 $X$는 원래 PPG sample 또는 feature 값이고, $X_{\min}$과 $X_{\max}$는 해당 신호 구간의 최솟값과 최댓값이다. $X_n$은 0과 1 사이로 정규화된 값이다. 다만 이 수식의 정확한 표기는 제공된 텍스트에서 완전히 보이지 않으므로, 위 식은 문맥상 min-max normalization을 설명하기 위한 일반적 형태이다.

### 3.3 Feature extraction

논문은 PPG 신호에서 총 6개의 feature를 추출하고, 그중 4개의 selected feature를 모델 학습에 사용한다고 설명한다. 그러나 제공된 추출 텍스트는 feature extraction 절의 초반에서 끊겨 있어, 6개 feature의 정확한 이름과 4개 selected feature의 구체적 조합은 확인할 수 없다. 따라서 이 보고서에서는 논문이 abstract와 proposal에서 명시한 “4 selected features out of six extracted features”라는 사실까지만 확정적으로 기술한다.

관련 연구 검토에서 논문은 기존 연구들이 35개, 46개, 60개, 107개, 513개와 같이 많은 feature를 사용하는 경우가 많다고 지적한다. 많은 feature는 계산 비용을 증가시키고, embedded device에서 memory와 latency 문제를 유발하며, redundant feature로 인해 overfitting 가능성을 높인다. 이에 비해 본 연구는 feature 수를 4개로 제한함으로써 resource efficiency를 강조한다.

제공된 텍스트에서 feature가 temporal feature인지, morphological feature인지, frequency-domain feature인지의 정확한 목록은 보이지 않는다. 다만 논문은 PPG feature가 time-sequenced feature로 사용되며, hybrid LSTM-ANN 구조에서 LSTM layer가 sequential input의 temporal variation을 모델링한다고 설명한다. 따라서 최종 모델은 단일 시점 feature vector뿐 아니라 시간적으로 배열된 PPG feature sequence를 활용하는 것으로 해석할 수 있다.

### 3.4 Single-stage model

Single-stage model은 classification을 사용하지 않고 PPG feature에서 직접 SBP와 DBP를 regression하는 전통적 방식으로 이해할 수 있다. 논문은 ANN과 LSTM 기반 conventional single-stage model을 비교 대상으로 구성했다고 설명한다. 이 구조는 classification-regression 접근이 실제로 regression accuracy를 개선하는지 확인하기 위한 baseline 역할을 한다.

ANN 기반 single-stage model은 상대적으로 memory와 energy efficiency가 좋지만, 논문은 ANN 단독 구조가 performance 측면에서 부족할 수 있다고 지적한다. 반면 LSTM은 temporal dependency modeling에 강하지만, layer 수와 hidden unit 수가 많아지면 embedded deployment에 부담이 된다. 따라서 single-stage ANN과 LSTM은 각각 efficiency와 temporal modeling 성능 측면에서 비교 기준이 된다.

### 3.5 Two-stage classification-regression model

Two-stage classification-regression model은 두 개의 순차적 단계로 구성된다. 첫 번째 단계는 BP category를 분류하는 classification model이다. 이 모델은 입력 PPG feature를 바탕으로 현재 혈압 상태가 normotension, hypertension, hypotension 중 어디에 속하는지 예측한다. 두 번째 단계는 regression model로, classification 결과를 활용하여 SBP 또는 DBP 값을 예측한다.

이 구조의 핵심은 classification이 regression의 search space를 제한한다는 점이다. 예를 들어 classification stage가 hypertension으로 판단하면 regression stage는 hypotension 또는 normotension 범위까지 포함한 매우 넓은 범위를 자유롭게 예측하기보다, hypertension에 해당하는 혈압 범위 안에서 더 안정적으로 예측하도록 유도될 수 있다. 논문은 이를 “range of quantifiable BP values is restricted to a specific limit”라고 설명한다.

Two-stage model은 특히 MCU deployment에서 중요하게 다루어진다. 논문은 single-stage ANN model과 two-stage classification-regression ANN model을 STM32 MCU에 embedded하여 real-time BP prediction을 수행했다고 보고한다. Real-time testing은 6명의 subject에 대해 수행되었으며, two-stage classification-regression ANN은 MAE ± SD 1.41 ± 1.29 mmHg를 달성했고, traditional single-stage ANN 대비 MAE를 83.5% 줄였다고 설명한다.

### 3.6 Joint classification-regression model

Joint classification-regression model은 classification과 regression을 별도의 완전히 독립된 단계로만 처리하지 않고, 하나의 학습 구조 안에서 두 task를 함께 다루는 방식이다. 제공된 텍스트에서는 joint model의 세부 loss function이나 branch별 parameter sharing 방식이 완전히 제시되어 있지 않다. 그러나 논문은 “joint learning models”와 “hybrid LSTM-ANN joint classification-regression architecture”를 언급하며, classification branch와 regression branch를 연결한 구조를 사용한다고 설명한다.

일반적으로 joint classification-regression에서는 classification loss와 regression loss가 함께 최적화될 수 있다. 이 경우 classification branch는 혈압 category를 예측하고, regression branch는 SBP와 DBP의 연속값을 예측한다. 다만 제공된 텍스트에는 cross-entropy, mean squared error, mean absolute error 등의 구체적 loss function 명칭이나 결합 가중치가 명시되어 있지 않다. 따라서 이 논문에서 실제 학습 objective가 어떤 수식으로 정의되었는지는 제공된 텍스트만으로는 확정할 수 없다.

### 3.7 Proposed hybrid LSTM-ANN joint classification-regression model

제안된 핵심 모델은 efficient hybrid LSTM-ANN joint classification-regression architecture이다. 이 구조는 hierarchical joint-learning approach에 기반하며, 세 개의 successive branch로 구성된다.

첫 번째 branch는 LSTM layer이다. 이 LSTM layer는 PPG feature sequence의 temporal variation을 모델링하고, 이전 sequential input에 대한 정보를 보존한다. PPG는 time-series signal이기 때문에, 단순 feedforward ANN이 한 시점 또는 한 window의 feature만 보는 것보다 LSTM이 temporal dependency를 반영하는 데 유리하다.

두 번째 branch는 ANN-based classification branch이다. 이 branch는 LSTM에서 나온 temporal representation을 이용하여 BP category를 예측한다. 예측되는 category는 normotension, hypertension, hypotension이다. 이 classification branch는 regression 성능을 높이기 위한 보조 구조이면서, 동시에 사용자에게 현재 BP status를 제공할 수 있다는 실용적 장점도 가진다.

세 번째 branch는 regression branch이다. Regression branch는 최종적으로 SBP와 DBP 값을 추정하는 main task를 수행한다. 논문은 이 구조가 LSTM의 temporal modeling 능력, ANN의 resource efficiency, classification stage의 regression stabilization 효과를 동시에 활용한다고 주장한다.

이 hybrid 구조는 성능만을 위해 매우 깊고 복잡한 LSTM을 쌓는 방식과 다르다. 논문은 하나의 simple LSTM layer를 사용하여 temporal dependency를 처리하고, 이후 ANN 구조를 결합함으로써 memory usage, complexity, latency를 낮추는 것을 목표로 한다. 따라서 이 모델의 설계 철학은 “최대한 복잡한 모델”이 아니라 “embedded deployment가 가능한 수준에서 classification의 도움을 받아 regression accuracy를 높이는 모델”이다.

## 4. 실험 및 결과

논문은 offline model evaluation과 MCU real-time deployment evaluation을 모두 수행한다. 제공된 텍스트에서 전체 table이나 모든 topology의 세부 numerical result는 보이지 않지만, abstract와 introduction에 주요 결과가 제시되어 있다.

### 4.1 데이터셋과 task

실험 데이터는 Physionet MIMIC II에서 선택한 40명 subject의 PPG 및 ABP recording이다. 각 subject에 대해 2분 길이의 PPG와 corresponding ABP가 제공된다. ABP로부터 SBP와 DBP를 추출하고, 동시에 BP category label을 구성한다. Category는 normotension, hypertension, hypotension의 세 가지이며, class distribution은 40%, 40%, 20%이다.

Task는 두 가지이다. 첫 번째는 classification으로, PPG feature를 입력받아 BP category를 맞히는 것이다. 두 번째는 regression으로, SBP와 DBP의 연속값을 예측하는 것이다. 논문의 핵심은 classification을 regression과 결합했을 때 regression accuracy가 개선되는지 평가하는 것이다.

### 4.2 비교 모델

비교 대상은 conventional single-stage ANN, conventional single-stage LSTM, two-stage classification-regression model, joint classification-regression model, hybrid LSTM-ANN joint classification-regression model을 포함한다. 또한 ANN 기반 single-stage model과 two-stage classification-regression ANN model은 STM32 MCU에 실제로 embedded되어 real-time prediction 성능을 비교한다.

이 비교 설계는 두 가지 질문에 답하기 위한 것이다. 첫째, classification stage가 regression accuracy를 높이는가. 둘째, 그 성능 향상이 embedded device deployment를 고려할 때도 실용적인가. 따라서 논문은 단순 MAE만 보지 않고 memory utilization, latency, complexity도 중요한 평가 항목으로 본다.

### 4.3 평가 지표

주요 regression 성능 지표는 MAE ± SD이다. MAE는 mean absolute error로, 예측 혈압과 reference 혈압 사이의 절대 오차 평균이다. $y_i$를 예측값, $x_i$를 reference 값이라고 하면 일반적으로 다음과 같이 표현된다.

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - x_i|
$$

SD는 error의 standard deviation으로, 예측 오차가 평균 주변에서 얼마나 퍼져 있는지를 나타낸다. MAE가 작더라도 SD가 크면 subject나 sample에 따라 예측이 불안정할 수 있으므로, BP monitoring에서는 MAE와 SD를 함께 보는 것이 중요하다.

Classification 성능 지표는 제공된 텍스트 일부에서는 구체적인 수치가 제시되어 있지 않다. 다만 논문은 high-performing classifier의 robustness를 강조하고, classification이 regression accuracy 향상에 중요한 역할을 한다고 설명한다.

Embedded evaluation에서는 memory usage, latency 또는 inference speed, model complexity가 중요한 지표로 언급된다. 논문은 microcontroller에서 deep learning model을 평가할 때 이러한 resource-related metric이 핵심이라고 명시한다.

### 4.4 Offline evaluation 결과

제안된 hybrid LSTM-ANN joint classification-regression model은 단 4개의 feature만 사용하여 가장 좋은 성능을 달성했다고 보고된다. 성능은 SBP에 대해 MAE ± SD 3.39 ± 5.47 mmHg, DBP에 대해 1.79 ± 3.72 mmHg이다. 이 수치는 PPG 기반 BP estimation에서 feature 수와 모델 복잡도를 크게 늘리지 않고도 경쟁력 있는 정확도를 얻었다는 점에서 중요하다.

논문은 전체적으로 classification-regression 접근이 ANN 기반 모델과 LSTM 기반 모델 모두에서 regression accuracy를 개선한다고 주장한다. 즉 improvement가 특정 architecture에만 의존하는 것이 아니라, classification을 regression과 결합하는 strategy 자체에서 나온다는 해석을 제시한다.

관련 연구와 비교하면, 논문은 기존 CNN 또는 LSTM 기반 연구들이 많은 feature를 사용하거나 ECG와 PPG를 함께 사용하는 경우가 많다고 지적한다. 예를 들어 일부 연구는 raw PPG와 ECG 및 physical characteristic을 함께 사용하거나, 513-feature vector를 사용하거나, CNN-LSTM multi-stage network를 사용했다. 반면 본 연구는 PPG 기반의 제한된 feature set과 비교적 효율적인 hybrid LSTM-ANN 구조를 사용한다는 점을 강조한다.

### 4.5 STM32 MCU real-time deployment 결과

논문은 실용성을 검증하기 위해 single-stage ANN model과 two-stage classification-regression ANN model을 STM32 Microcontroller Unit에 embedded했다. 이는 cNIBP system이 실제 standalone device로 동작할 수 있는지 확인하기 위한 중요한 실험이다.

Real-time testing은 6명의 subject에 대해 수행되었다. Two-stage classification-regression ANN model은 MAE ± SD 1.41 ± 1.29 mmHg를 달성했다. 논문은 이 결과가 traditional single-stage ANN model과 비교해 MAE를 83.5% 줄인 것이라고 보고한다. 이는 classification stage가 offline evaluation뿐 아니라 real-time embedded setting에서도 regression accuracy와 robustness 향상에 기여했음을 보여준다.

다만 제공된 텍스트에서는 STM32 MCU의 정확한 model memory footprint, latency 수치, clock setting, quantization 여부, fixed-point 또는 floating-point implementation 여부, subject별 결과는 확인할 수 없다. 따라서 이 보고서에서는 MCU deployment가 수행되었고 two-stage model이 real-time setting에서 우수한 MAE ± SD를 보였다는 점까지만 확정적으로 서술한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 PPG 기반 혈압 추정에서 classification이 regression accuracy를 높일 수 있다는 아이디어를 여러 topology에서 검증했다는 점이다. 단순히 하나의 모델을 제안하는 데 그치지 않고, single-stage, two-stage, joint learning, hybrid LSTM-ANN 구조를 비교함으로써 classification-regression approach 자체의 효과를 보이려 한다.

두 번째 강점은 efficiency를 연구의 핵심 목표로 설정했다는 점이다. 기존 deep learning 기반 BP estimation 연구는 정확도 향상을 위해 CNN, BiLSTM, multi-layer LSTM, large feature vector 등을 사용하는 경우가 많다. 하지만 본 논문은 memory usage, latency, complexity를 명시적으로 고려하며, MCU deployment까지 수행한다. 이는 실제 wearable 또는 portable BP monitoring device 개발 관점에서 매우 중요한 장점이다.

세 번째 강점은 매우 적은 수의 feature를 사용했다는 점이다. 논문은 6개 extracted feature 중 4개 selected feature만 사용한다고 설명한다. 많은 feature를 추출하는 방식은 computation cost를 높이고 overfitting 위험을 증가시킬 수 있다. 이 연구는 feature 수를 줄이면서도 hybrid LSTM-ANN joint model에서 SBP MAE 3.39 mmHg, DBP MAE 1.79 mmHg를 달성했다고 보고하므로, compact model design 측면에서 의미가 있다.

네 번째 강점은 MCU 기반 real-time 검증이다. 많은 biomedical deep learning 연구가 offline dataset evaluation에 머무르는 반면, 이 논문은 STM32 MCU에 ANN model을 embedded하여 real-time BP prediction을 수행했다. 특히 two-stage classification-regression ANN이 single-stage ANN 대비 MAE를 크게 줄였다는 결과는 classification stage의 실용적 가치를 강조한다.

그러나 한계도 분명하다. 첫째, 데이터셋 규모가 작다. Offline model evaluation은 40명 subject를 대상으로 하고, real-time MCU testing은 6명 subject를 대상으로 한다. BP estimation model은 subject variability, sensor placement, motion artifact, age, sex, disease condition, medication 등에 민감할 수 있으므로, 이 정도 규모만으로 generalization을 강하게 주장하기는 어렵다.

둘째, 제공된 텍스트 기준으로 feature의 정확한 종류와 selection procedure가 완전히 확인되지 않는다. 논문은 6개 feature를 추출하고 4개를 선택했다고 하지만, 추출 텍스트가 중간에 끊겨 있어 feature 목록, selection criterion, feature normalization 방식, windowing 방식, sequence length 등 주요 세부 사항을 확인할 수 없다. 따라서 방법 재현성은 제공된 발췌문만으로는 제한적이다.

셋째, high-quality PPG signal을 선택했다고 되어 있으나, signal quality selection의 구체적 기준은 제공된 텍스트에서 확인되지 않는다. 실제 wearable 환경에서는 motion artifact, low perfusion, skin tone, sensor pressure, ambient light 등으로 PPG 품질이 크게 흔들린다. 깨끗한 PPG만 선택한 실험이라면 실제 환경에서 성능이 낮아질 수 있다.

넷째, classification label의 정의가 구체적으로 보이지 않는다. Normotension, hypertension, hypotension을 어떤 SBP/DBP threshold로 나누었는지, SBP와 DBP label이 충돌하는 경우 어떻게 처리했는지, subject 단위 또는 segment 단위로 class를 정의했는지 등이 제공된 텍스트에서 명확하지 않다. Classification-regression 접근의 핵심이 class boundary에 의존하므로, 이 부분은 중요한 재현성 요소이다.

다섯째, regression 성능이 매우 우수하게 보고되었지만, subject split 방식과 data leakage 가능성을 판단할 정보가 제공된 텍스트만으로는 충분하지 않다. PPG 기반 BP estimation에서는 같은 subject의 segment가 train과 test에 동시에 포함되면 성능이 과대평가될 수 있다. 제공된 텍스트에서는 40명 subject를 사용했다는 점은 확인되지만, subject-independent split인지, segment-level split인지, cross-validation 방식이 무엇인지는 보이지 않는다.

비판적으로 보면, 이 논문은 embedded BP monitoring을 목표로 classification-regression 구조를 제안한 점에서 실용적 가치가 크다. 그러나 제공된 발췌문 기준으로는 full methodology와 validation protocol의 세부 사항이 충분히 드러나지 않기 때문에, 보고된 높은 정확도를 해석할 때는 데이터 분할, subject independence, PPG quality filtering, class definition을 반드시 추가로 확인해야 한다.

## 6. 결론

이 논문은 PPG 기반 cuffless BP monitoring에서 classification-regression 접근이 regression accuracy를 향상시킬 수 있음을 보이고, 이를 효율적인 hybrid LSTM-ANN architecture로 구현한 연구이다. 제안 모델은 LSTM layer를 통해 PPG feature sequence의 temporal dependency를 반영하고, ANN-based classification branch를 통해 혈압 상태를 normotension, hypertension, hypotension으로 분류하며, regression branch에서 최종 SBP와 DBP를 추정한다.

논문의 주요 기여는 classification을 regression과 결합하여 BP prediction을 안정화하려 했다는 점, 단 4개의 feature만으로 competitive accuracy를 달성했다는 점, memory usage와 latency 및 model complexity를 고려했다는 점, 그리고 STM32 MCU에 ANN-based model을 실제 embedded하여 real-time prediction을 수행했다는 점이다.

제공된 텍스트 기준으로, hybrid LSTM-ANN joint classification-regression model은 40명 MIMIC II subject에서 SBP MAE ± SD 3.39 ± 5.47 mmHg, DBP MAE ± SD 1.79 ± 3.72 mmHg를 달성했다. 또한 STM32 MCU real-time test에서는 two-stage classification-regression ANN이 6명 subject에 대해 MAE ± SD 1.41 ± 1.29 mmHg를 달성하고, traditional single-stage ANN 대비 MAE를 83.5% 줄였다고 보고된다.

이 연구는 향후 wearable cNIBP device, portable health monitoring system, MCU 기반 biomedical AI model 설계에 중요한 방향을 제시한다. 특히 모델을 무조건 깊고 크게 만드는 대신, classification으로 regression 범위를 안정화하고, LSTM과 ANN을 적절히 결합하여 temporal modeling과 resource efficiency를 동시에 달성하려는 설계 철학은 실제 embedded healthcare application에 유용하다.

다만 실제 임상 또는 상용 wearable 적용을 위해서는 더 큰 subject-independent dataset, 다양한 혈압 범위와 인구집단, motion artifact가 포함된 실제 환경 데이터, 명확한 signal quality assessment, class threshold 정의, MCU memory 및 latency 수치, 장기간 continuous monitoring 검증이 추가로 필요하다. 따라서 이 논문은 완성된 임상용 BP monitoring solution이라기보다는, classification-regression 기반의 efficient PPG blood pressure estimation이 실제 embedded implementation까지 이어질 수 있음을 보여주는 중요한 proof-of-concept 연구로 평가할 수 있다.
