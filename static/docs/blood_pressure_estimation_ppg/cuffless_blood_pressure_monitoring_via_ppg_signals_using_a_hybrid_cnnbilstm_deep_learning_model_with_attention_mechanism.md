# Cuff-less blood pressure monitoring via PPG signals using a hybrid CNN-BiLSTM deep learning model with attention mechanism

* **저자**: Hanieh Mohammadi, Bahram Tarvirdizadeh, Khalil Alipour, Mohammad Ghamari
* **발표연도**: 2025

## 1. 논문 개요

이 논문은 PPG(Photoplethysmography) 신호만을 이용하여 커프 없이 혈압을 추정하는 비침습적 딥러닝 모델을 제안한다. 논문은 *Scientific Reports*에 2025년 게재되었으며, DOI는 10.1038/s41598-025-07087-2이다. 연구의 핵심 목표는 단일 PPG 신호로부터 SBP(Systolic Blood Pressure)와 DBP(Diastolic Blood Pressure)를 정확하게 예측하는 것이다.

혈압은 심혈관 건강 상태를 나타내는 핵심 생체 지표이다. 특히 고혈압은 자각 증상이 적고 장기간 방치될 경우 심장, 뇌, 신장 질환의 위험을 높이므로 지속적인 모니터링이 중요하다. 기존 커프 기반 혈압계는 비교적 신뢰할 수 있는 측정값을 제공하지만, 측정이 불연속적이고 반복적인 커프 팽창으로 인한 불편함이 있다. 이러한 한계 때문에 일상생활 중 장시간 혈압 변화를 추적하거나 웨어러블 기기에서 실시간으로 혈압을 측정하기에는 적합하지 않다.

이 논문은 이러한 문제를 해결하기 위해 PPG 기반 cuff-less blood pressure estimation을 다룬다. PPG는 피부에 빛을 조사하고 혈류량 변화에 따른 반사 또는 흡수 광량 변화를 측정하는 방식이다. 심장 박동에 따라 말초 혈관의 혈액량이 변하기 때문에 PPG 신호에는 혈압과 관련된 생리적 정보가 포함될 수 있다. 또한 PPG 센서는 스마트워치, 스마트밴드, 패치형 센서 등 웨어러블 기기에 쉽게 탑재될 수 있어 연속 모니터링에 적합하다.

연구 문제는 전처리된 PPG segment를 입력으로 받아 해당 구간의 평균 SBP와 DBP를 예측하는 회귀 문제이다. 저자들은 CNN, BiLSTM, attention mechanism을 결합한 hybrid deep learning architecture를 제안한다. CNN은 PPG 신호의 국소적 파형 패턴을 추출하고, BiLSTM은 시간적 의존성을 양방향으로 학습하며, attention mechanism은 혈압 추정에 더 중요한 시간 구간에 높은 가중치를 부여한다.

논문은 MIMIC-II 기반 공개 데이터셋에서 2064명의 환자 데이터를 사용했다는 점을 강조한다. 이는 비교 대상 연구들보다 상대적으로 큰 표본 규모이며, 저자들은 이를 통해 모델의 robustness와 generalizability가 강화된다고 주장한다. 최종 모델은 5-fold cross-validation에서 SBP MAE 1.88, DBP MAE 1.34를 달성했다고 보고한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 PPG 기반 혈압 추정에서 공간적 또는 국소적 파형 특징과 시간적 의존성을 함께 학습해야 한다는 것이다. PPG 신호는 단순한 1차원 시계열이지만, 그 안에는 여러 수준의 정보가 포함되어 있다. 짧은 시간 구간에서는 pulse peak, rising slope, falling slope, notch와 같은 waveform morphology가 중요하고, 긴 시간 구간에서는 여러 박동에 걸친 반복 패턴과 temporal dependency가 중요하다.

CNN은 convolution filter를 통해 국소 패턴을 자동으로 추출하는 데 강하다. PPG 신호에서 CNN은 특정 시간 구간의 파형 모양, 진폭 변화, 상승 및 하강 구간의 형태 같은 특징을 포착할 수 있다. 그러나 CNN만으로는 긴 시퀀스 전체의 시간적 흐름이나 과거와 미래 문맥을 충분히 반영하기 어렵다.

BiLSTM은 시퀀스를 앞 방향과 뒤 방향으로 동시에 처리한다. 일반 LSTM은 과거에서 현재 방향의 문맥을 학습하지만, BiLSTM은 현재 시점의 표현을 만들 때 이전 시점과 이후 시점의 정보를 모두 활용한다. PPG segment 전체가 이미 주어진 상태에서 혈압을 추정하는 문제에서는 미래 방향 정보도 활용할 수 있으므로 BiLSTM이 적합하다.

Attention mechanism은 BiLSTM이 출력한 각 time step의 hidden state에 중요도를 부여한다. 모든 시간 구간이 혈압 예측에 같은 정도로 중요하지는 않다. 예를 들어 특정 pulse cycle의 peak 주변, slope가 급격히 변하는 부분, 또는 혈압과 밀접한 morphology가 나타나는 구간이 더 중요한 정보를 포함할 수 있다. Attention은 이러한 구간을 학습 과정에서 자동으로 강조한다.

기존 접근 방식과의 차별점은 세 가지로 정리된다. 첫째, 이 논문은 PPG만 사용하므로 ECG, VPG, APG 등 추가 신호를 요구하지 않는다. 둘째, handcrafted feature를 설계하지 않고 raw 또는 전처리된 PPG segment에서 end-to-end 방식으로 특징을 학습한다. 셋째, CNN, BiLSTM, attention을 결합하여 local feature extraction, temporal dependency modeling, informative segment selection을 하나의 모델 안에서 수행한다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

논문의 전체 흐름은 다음과 같다.

먼저 MIMIC-II 기반 공개 데이터셋에서 PPG와 ABP 신호를 사용한다. ABP 신호는 정답 혈압 값을 만들기 위한 참조 신호로 사용되고, 모델 입력은 PPG 신호이다. 원본 신호는 detrending, 이상치 제거, 짧은 기록 제거, segmentation 등의 전처리를 거친다. 이후 각 8.192초 구간의 ABP 신호에서 systolic point와 diastolic point를 검출하고, 각각의 평균값을 해당 PPG segment의 SBP 및 DBP target으로 사용한다. 최종적으로 CNN-BiLSTM-Attention 모델을 학습하고 5-fold cross-validation으로 평가한다.

전체 구조는 다음과 같이 요약할 수 있다.

$$
PPG \rightarrow Preprocessing \rightarrow CNN \rightarrow BiLSTM \rightarrow Attention \rightarrow Regression \rightarrow (\hat{SBP}, \hat{DBP})
$$

이 연구에서 목표 변수는 SBP와 DBP이다. 논문은 MAP 예측을 별도로 다루지 않는다.

### 3.2 데이터셋과 입력 구성

논문은 MIMIC-II 기반 UCI 공개 데이터셋을 사용한다. 해당 데이터셋은 ICU 환자들의 PPG, ABP, ECG 신호를 포함하며, 원래 총 12,000명의 환자 기록을 포함한다고 설명된다. 모든 신호는 125 Hz sampling rate와 8-bit digital precision으로 기록되었다.

이 연구에서는 PPG와 ABP 신호만 사용한다. PPG는 모델 입력으로 사용되고, ABP는 target label을 만들기 위해 사용된다. ECG는 사용하지 않는다. 저자들은 ABP의 최대값이 200 mmHg를 초과하는 기록을 제거하고, 신호 길이가 8분 미만인 기록도 제거했다. 이 과정을 통해 최종 subject 수는 12,000명에서 2064명으로 줄었다.

각 신호는 8.192초 window로 분할된다. sampling rate가 125 Hz이므로 8.192초는 1024 samples에 해당한다.

$$
8.192 \times 125 = 1024
$$

Segmentation 과정에서는 75% overlap을 사용한다. 이는 인접 segment가 상당 부분 겹친다는 의미이다. Overlap을 사용하면 더 많은 training sample을 만들 수 있고, 신호의 연속적인 생리적 변화를 더 촘촘하게 포착할 수 있다. 그러나 overlapping segment가 cross-validation split 과정에서 어떻게 분리되었는지에 따라 data leakage 위험이 생길 수 있다. 논문은 dataset을 shuffling한 뒤 5-fold cross-validation을 수행했다고 설명하지만, subject-wise split을 사용했다고 명확히 말하지는 않는다. 따라서 동일 환자 또는 인접 overlapping window가 서로 다른 fold에 들어갔는지는 제공된 텍스트만으로 확인할 수 없다. 이는 평가 신뢰성 측면에서 중요한 한계이다.

### 3.3 신호 전처리

논문은 모델 학습 전 PPG와 ABP 신호 품질을 높이기 위해 여러 전처리 단계를 적용한다.

첫째, detrending을 수행한다. PPG 신호에는 센서 접촉 변화, 호흡, 움직임, 피부 상태, 저주파 drift로 인해 baseline이 천천히 변하는 문제가 있다. 저자들은 linear regression model을 이용해 PPG 신호의 baseline trend line을 추정하고, 이를 원 신호에서 제거한다. 원 신호를 $x(t)$, 추정된 trend를 $r(t)$라고 하면 detrended signal은 개념적으로 다음과 같이 표현할 수 있다.

$$
x_{detrended}(t) = x(t) - r(t)
$$

이 과정은 저주파 baseline shift를 줄이고, 심박과 관련된 주요 파형 정보를 더 명확히 만든다.

둘째, ABP 최대값이 200 mmHg를 초과하는 기록을 제거한다. 이는 극단적인 이상치가 모델 학습에 과도한 영향을 주는 것을 방지하기 위한 절차이다.

셋째, duration이 8분 미만인 기록을 제거한다. 저자들은 충분히 긴 신호만 사용해야 의미 있는 생리적 패턴을 포착할 수 있다고 설명한다. 이 기준을 적용한 뒤 최종 환자 수는 2064명이 된다.

넷째, 8.192초 window와 75% overlap을 사용해 segmentation을 수행한다. 이 길이는 여러 심박 주기를 포함할 수 있어 PPG 기반 혈압 추정에 필요한 temporal pattern을 포함할 가능성이 높다.

### 3.4 ABP에서 target label 생성

이 연구는 전체 ABP waveform을 직접 예측하는 것이 아니라, 각 PPG segment에 대응되는 단일 SBP 값과 단일 DBP 값을 예측한다. 이를 위해 각 8.192초 ABP segment에서 systolic point와 diastolic point를 검출한다.

Systolic point는 ABP waveform의 peak에 해당하며, diastolic point는 valley에 해당한다. 각 segment 안에는 여러 심박 주기가 포함될 수 있으므로 여러 개의 peak와 valley가 존재한다. 저자들은 해당 segment 안의 systolic points 평균을 SBP target으로 사용하고, diastolic points 평균을 DBP target으로 사용한다.

개념적으로 한 segment에 $m$개의 systolic point와 $m$개의 diastolic point가 있다고 하면 target은 다음과 같이 정의할 수 있다.

$$
SBP_{target} =
\frac{1}{m}
\sum_{j=1}^{m}
SBP_j
$$

$$
DBP_{target} =
\frac{1}{m}
\sum_{j=1}^{m}
DBP_j
$$

즉 모델은 8.192초 PPG waveform 전체를 입력으로 받고, 같은 구간에서 ABP로부터 추출한 평균 SBP와 평균 DBP를 출력하도록 학습된다.

### 3.5 CNN layer: 국소 특징 추출

모델의 첫 번째 핵심 구성 요소는 CNN이다. CNN은 PPG 신호의 local pattern을 추출하는 역할을 한다. PPG는 시간축을 따라 변화하는 1차원 신호이므로, 논문은 1D convolution을 사용한다.

논문에서 제시한 1D convolution 연산은 다음과 같다.

$$
f(x)=\sigma
\left(
\sum_{i=1}^{k}
w_i x_{t-i+1}+b
\right)
$$

여기서 $w_i$는 convolution filter의 학습 가능한 가중치, $k$는 filter size, $b$는 bias term, $\sigma$는 activation function이다. 이 논문에서는 ReLU를 사용한다.

ReLU는 다음과 같이 표현된다.

$$
ReLU(x)=\max(0,x)
$$

ReLU는 음수 값을 0으로 만들고 양수 값은 그대로 통과시킨다. 이 비선형 함수는 vanishing gradient 문제를 완화하고 학습 수렴을 빠르게 하는 장점이 있다. 저자들은 sigmoid와 tanh도 실험했지만, ReLU가 training stability와 accuracy 측면에서 더 좋은 성능을 보였다고 설명한다.

최종 모델은 세 개의 CNN layer를 사용한다. 논문은 1 CNN, 2 CNN, 3 CNN 구조를 비교했으며, 세 개의 CNN layer를 사용하는 구조가 가장 좋은 결과를 보였다. 이는 PPG의 얕은 국소 특징뿐 아니라 더 추상적인 waveform representation도 필요하다는 점을 시사한다.

### 3.6 BiLSTM layer: 시간 의존성 학습

CNN이 추출한 feature map은 BiLSTM으로 전달된다. BiLSTM은 PPG segment의 시간적 의존성을 학습한다. PPG 신호에서 한 시점의 값은 독립적이지 않고, 이전 및 이후 파형과 연속적인 관계를 가진다. 예를 들어 peak의 의미는 그 전의 상승 구간과 그 후의 하강 구간을 함께 봐야 더 잘 해석된다.

LSTM은 input gate, forget gate, output gate를 사용하여 장기 의존성을 학습한다. 전통적인 RNN은 긴 시퀀스를 처리할 때 gradient vanishing 문제가 발생하기 쉽지만, LSTM은 memory cell과 gating mechanism을 통해 필요한 정보를 장기간 보존할 수 있다.

BiLSTM은 forward LSTM과 backward LSTM을 동시에 사용한다. forward LSTM은 과거에서 미래 방향으로 신호를 처리하고, backward LSTM은 미래에서 과거 방향으로 신호를 처리한다. 두 방향의 hidden representation을 결합하면 각 time step에서 앞뒤 문맥을 모두 반영한 표현을 얻을 수 있다.

이 논문은 두 개의 BiLSTM layer를 사용한다. 첫 번째 BiLSTM의 출력은 양방향 구조 때문에 feature dimensionality가 확장되며, 이 출력이 두 번째 BiLSTM의 입력으로 들어간다. 이러한 계층적 구조는 더 깊은 temporal feature representation을 만들기 위한 것이다.

### 3.7 Attention mechanism: 중요한 시간 구간 강조

BiLSTM 출력은 attention mechanism으로 전달된다. Attention은 각 time step의 hidden state가 혈압 예측에 얼마나 중요한지 학습한다. 논문은 attention score, attention weight, context vector를 다음과 같이 설명한다.

먼저 각 time step $t$에 대해 BiLSTM hidden state $h_t$와 학습 가능한 parameter를 사용하여 score $e_t$를 계산한다.

$$
e_t = \tanh(W_a h_t + b_a)
$$

그 다음 softmax를 적용하여 attention weight $\alpha_t$를 얻는다.

$$
\alpha_t =
\frac{\exp(e_t)}
{\sum_{t'} \exp(e_{t'})}
$$

마지막으로 attention weight를 이용해 BiLSTM hidden states의 가중합을 계산한다.

$$
c =
\sum_t
\alpha_t h_t
$$

여기서 $c$는 context vector이다. 이 context vector는 최종 혈압 예측에 사용된다.

이 구조의 의미는 모든 time step을 동일하게 평균내지 않고, 혈압 추정에 더 중요한 time step을 더 강하게 반영한다는 것이다. PPG 신호에서는 특정 pulse cycle이나 특정 waveform morphology가 혈압 정보와 더 밀접할 수 있으므로 attention mechanism은 합리적인 설계이다.

### 3.8 실험한 모델 구조

저자들은 최종 구조를 선택하기 위해 여러 architecture를 비교했다. 실험한 구조는 다음과 같다.

| 구조                         | 설명                                 |
| ---------------------------- | ------------------------------------ |
| 1 CNN + 1 BiLSTM + Attention | 가장 얕은 비교 모델                  |
| 2 CNN + 1 BiLSTM + Attention | CNN feature extraction을 강화한 모델 |
| 2 CNN + 2 BiLSTM + Attention | temporal modeling을 강화한 모델      |
| 3 CNN + 2 BiLSTM + Attention | 최종 제안 모델                       |

최종적으로 3 CNN + 2 BiLSTM + Attention 구조가 SBP와 DBP 모두에서 가장 우수한 성능을 보였다. 이는 PPG 기반 혈압 추정에서 충분한 local feature extraction과 deep temporal modeling이 모두 중요하다는 것을 보여준다.

### 3.9 손실 함수와 평가 지표

논문은 모델 성능 평가 지표로 MSE(Mean Squared Error)와 MAE(Mean Absolute Error)를 사용한다. 텍스트에서는 MSE와 MAE를 평가 기준으로 설명한다. 손실 함수 그래프도 제시되지만, 실제 학습 손실로 MSE를 사용했는지 MAE를 사용했는지, 또는 두 지표 중 무엇을 최적화했는지는 제공된 텍스트만으로 완전히 명확하지 않다.

SBP에 대한 MSE는 다음과 같이 정의된다.

$$
MSE_{SBP} =
\frac{1}{n}
\sum_{i=1}^{n}
(SBP_{act}(i)-SBP_{pred}(i))^2
$$

DBP에 대한 MSE는 다음과 같다.

$$
MSE_{DBP} =
\frac{1}{n}
\sum_{i=1}^{n}
(DBP_{act}(i)-DBP_{pred}(i))^2
$$

제공된 원문 수식에서는 제곱 표시가 누락된 것처럼 보이지만, MSE의 정의상 오차 제곱이 포함되어야 한다. 따라서 위와 같이 해석하는 것이 타당하다.

MAE는 다음과 같다.

$$
MAE_{SBP} =
\frac{1}{n}
\sum_{i=1}^{n}
|SBP_{act}(i)-SBP_{pred}(i)|
$$

$$
MAE_{DBP} =
\frac{1}{n}
\sum_{i=1}^{n}
|DBP_{act}(i)-DBP_{pred}(i)|
$$

MSE는 큰 오차를 더 강하게 벌점화하므로 outlier에 민감하다. 반면 MAE는 평균적으로 얼마나 벗어나는지를 직관적으로 보여주며, 임상적 해석이 쉽다. 혈압 추정에서는 MAE가 “평균적으로 몇 mmHg 정도 틀리는가”를 직접 나타내기 때문에 중요한 지표이다.

### 3.10 학습 설정과 하이퍼파라미터

모델은 PyTorch로 구현되었고 NVIDIA GeForce RTX 3090에서 학습되었다. 저자들은 5-fold cross-validation을 사용했으며, 각 fold마다 모델을 처음부터 다시 학습했다고 명시한다. 이는 fold 간 학습 정보가 섞이지 않도록 하기 위한 적절한 절차이다.

하이퍼파라미터 탐색 결과 선택된 값은 다음과 같다.

| 항목          | 선택 값       |
| ------------- | ------------- |
| Learning rate | 0.001         |
| Optimizer     | Adam          |
| Patience      | 30            |
| Epochs        | 500           |
| Batch size    | 64            |
| Filters       | [32, 64, 128] |
| Kernel size   | 3             |
| Kernel stride | 1             |
| Pool method   | Max           |
| Pool size     | 2             |
| Pool stride   | 1             |
| LSTM units    | 128           |
| Dropout ratio | 0.2           |

논문은 learning rate, optimizer, patience, epoch 수, batch size, filter 수, kernel size, pooling 방식, LSTM unit 수, dropout ratio 등을 탐색했다고 설명한다. 최종 설정은 비교적 표준적인 딥러닝 학습 구성이다.

## 4. 실험 및 결과

### 4.1 평가 방식

모델은 5-fold cross-validation으로 평가되었다. 전체 dataset을 다섯 부분으로 나누고, 각 iteration에서 하나의 fold를 test set으로 사용하며 나머지 네 fold를 training set으로 사용한다. 논문은 training data 중 10%를 validation data로 사용했다고 설명한다. 각 fold의 결과를 평균하여 최종 성능을 보고한다.

이 방식은 단일 train-test split보다 안정적인 평가를 제공할 수 있다. 그러나 앞서 언급했듯이 이 논문은 subject-wise split을 명시하지 않는다. 데이터가 segmentation과 75% overlap을 거친 뒤 sample 단위로 shuffling되어 fold가 구성되었다면, 동일 환자의 segment 또는 서로 겹치는 segment가 train과 test에 동시에 존재할 가능성이 있다. 이 경우 성능이 과대평가될 수 있다. 제공된 텍스트만으로는 이 문제가 해결되었는지 확인할 수 없다.

### 4.2 Architecture 비교 결과

네 가지 architecture의 5-fold 평균 결과는 다음과 같다.

| Architecture                 | MSE(SBP) | MSE(DBP) | MAE(SBP) | MAE(DBP) |
| ---------------------------- | -------: | -------: | -------: | -------: |
| 1 CNN + 1 BiLSTM + Attention |    13.94 |    10.66 |     2.70 |     2.15 |
| 2 CNN + 1 BiLSTM + Attention |    12.44 |     6.76 |     2.22 |     1.64 |
| 2 CNN + 2 BiLSTM + Attention |     9.20 |     4.50 |     2.00 |     1.40 |
| 3 CNN + 2 BiLSTM + Attention |     8.62 |     4.39 |     1.88 |     1.34 |

결과는 CNN layer와 BiLSTM layer를 늘릴수록 성능이 개선되는 경향을 보여준다. 1 CNN + 1 BiLSTM 구조는 SBP MAE 2.70, DBP MAE 2.15로 가장 낮은 성능을 보인다. 2 CNN + 1 BiLSTM 구조는 특히 DBP에서 큰 개선을 보인다. 2 CNN + 2 BiLSTM 구조는 temporal modeling을 강화하면서 SBP와 DBP 모두 개선된다. 최종적으로 3 CNN + 2 BiLSTM + Attention 구조가 가장 낮은 MAE와 MSE를 달성한다.

이 결과는 PPG 기반 혈압 추정에서 단순한 shallow CNN이나 단일 BiLSTM만으로는 충분하지 않으며, 더 풍부한 local feature extraction과 deeper sequential modeling이 필요하다는 논문의 주장을 뒷받침한다.

### 4.3 최종 모델의 fold별 결과

최종 모델인 3 CNN + 2 BiLSTM + Attention의 SBP fold별 결과는 다음과 같다.

| Fold          |    MAE(SBP) |    MSE(SBP) |
| ------------- | ----------: | ----------: |
| Fold 1        |        1.67 |        7.00 |
| Fold 2        |        2.40 |       12.85 |
| Fold 3        |        1.75 |        7.76 |
| Fold 4        |        1.85 |        8.21 |
| Fold 5        |        1.69 |        7.26 |
| Average ± STD | 1.88 ± 0.27 | 8.62 ± 2.15 |

SBP의 평균 MAE는 1.88이며 fold 간 표준편차는 0.27이다. Fold 2에서 MAE와 MSE가 상대적으로 높지만, 전체적으로는 안정적인 성능을 보인다.

DBP fold별 결과는 다음과 같다.

| Fold          |    MAE(DBP) |    MSE(DBP) |
| ------------- | ----------: | ----------: |
| Fold 1        |        1.34 |        4.42 |
| Fold 2        |        1.34 |        4.26 |
| Fold 3        |        1.26 |        3.97 |
| Fold 4        |        1.47 |        5.33 |
| Fold 5        |        1.28 |        3.98 |
| Average ± STD | 1.34 ± 0.07 | 4.39 ± 0.49 |

DBP의 평균 MAE는 1.34이며 fold 간 표준편차는 0.07로 매우 작다. 이는 DBP 예측이 SBP보다 더 안정적임을 보여준다. 일반적으로 SBP는 혈관 탄성, 반사파, 심박출량 변화에 더 민감하므로 DBP보다 예측이 어려운 경향이 있으며, 이 논문의 결과도 그러한 경향과 일치한다.

### 4.4 학습 곡선과 산점도 분석

논문은 best fold에 대해 training 및 validation loss와 MAE 변화를 그래프로 제시한다. 설명에 따르면 학습이 진행될수록 loss와 MAE가 감소하고 일정한 값에 수렴한다. 이는 모델이 PPG signal feature를 학습하고 있으며, validation data에서도 성능이 안정화된다는 것을 의미한다.

또한 validation dataset에 대해 actual versus predicted scatter plot을 제시한다. SBP의 best fold는 Fold 1, DBP의 best fold는 Fold 3으로 보고된다. 산점도에서 대부분의 점이 이상적인 예측선 $y=x$에 가깝게 분포한다고 설명한다. 이는 예측값이 실제값과 높은 일치도를 보인다는 시각적 근거이다.

### 4.5 Bland–Altman 분석

논문은 best fold에 대해 Bland–Altman plot을 제시한다. Bland–Altman plot은 예측값과 실제값의 평균을 x축에, 두 값의 차이를 y축에 나타낸다. 이 분석은 단순 correlation과 달리 예측 오차가 혈압 값의 크기에 따라 체계적으로 변하는지, 그리고 예측과 실제 측정 사이 agreement가 어느 정도인지를 평가한다.

논문은 plot에서 파란 선을 limits of agreement로, 빨간 선을 mean difference로 표시했다고 설명한다. 그러나 제공된 텍스트에는 limits of agreement의 구체적인 수치가 포함되어 있지 않다. 따라서 정량적인 bias나 agreement 범위는 원문 figure를 직접 확인하지 않는 한 정확히 보고할 수 없다.

### 4.6 기존 연구와의 비교

논문은 기존 PPG 기반 혈압 추정 연구들과 성능을 비교한다. 비교 결과는 다음과 같다.

| 연구           | 데이터셋 및 subject 수 | 입력          | 방법                     | MAE(SBP) | MAE(DBP) |
| -------------- | ---------------------- | ------------- | ------------------------ | -------: | -------: |
| Baek et al.    | MIMIC-II, 942명        | PPG           | CNN                      |    10.86 |     5.95 |
| Baek et al.    | MIMIC-II, 942명        | PPG & ECG     | CNN                      |     9.30 |     5.12 |
| Wang et al.    | MIMIC, 90명            | PPG           | CNN + RNN                |     3.95 |     2.14 |
| Panwar et al.  | MIMIC-II, 1557명       | PPG           | CNN + LSTM               |     3.97 |     2.30 |
| Ibtehaz et al. | MIMIC-II, 942명        | PPG           | PPG2ABP                  |     5.73 |     3.45 |
| Cheng et al.   | MIMIC-II, 1627명       | PPG, VPG, APG | ABP-Net                  |     3.27 |     1.90 |
| Tang et al.    | MIMIC-II, 500명        | PPG, VPG, APG | W-Net                    |     2.62 |     1.56 |
| Tang et al.    | MIMIC-II, 500명        | PPG           | W-Net                    |     2.60 |     1.45 |
| Our study      | MIMIC-II, 2064명       | PPG           | CNN + BiLSTM + Attention |     1.88 |     1.34 |

이 비교에서 제안 모델은 SBP와 DBP 모두에서 가장 낮은 MAE를 달성한다. 특히 PPG 단일 신호만 사용하면서도 PPG, VPG, APG를 함께 사용하는 ABP-Net이나 W-Net보다 낮은 오차를 보인다는 점이 논문의 주요 주장이다. 저자들은 큰 표본 수, CNN-BiLSTM-Attention 구조, 세밀한 전처리, 5-fold cross-validation이 성능 향상에 기여했다고 해석한다.

다만 이 비교는 신중하게 해석해야 한다. 기존 연구들은 서로 다른 데이터셋 구성, subject 수, preprocessing, split 방식, evaluation protocol을 사용했을 가능성이 있다. 따라서 표에 나타난 성능 차이가 순수하게 모델 구조 차이 때문이라고 단정하기는 어렵다. 특히 이 논문의 cross-validation이 subject-wise인지 불명확하므로, 엄격한 cross-subject generalization 평가와 직접 비교가 추가로 필요하다.

## 5. 강점, 한계

### 5.1 강점

이 논문의 가장 큰 강점은 PPG 단일 신호만으로 매우 낮은 SBP 및 DBP 예측 오차를 보고했다는 점이다. ECG, VPG, APG와 같은 추가 신호를 사용하지 않으므로 시스템 구성이 단순하고 웨어러블 기기에 적용하기 쉽다. 이는 실제 사용성 측면에서 중요한 장점이다.

두 번째 강점은 CNN, BiLSTM, attention mechanism의 역할이 비교적 명확하다는 점이다. CNN은 PPG waveform의 국소 특징을 추출하고, BiLSTM은 양방향 시간 문맥을 학습하며, attention은 중요한 time step을 강조한다. 각 구성 요소가 생체신호 시계열 처리에서 합리적인 기능을 맡고 있다.

세 번째 강점은 2064명의 환자를 사용했다는 점이다. 논문이 비교한 여러 기존 연구보다 subject 수가 많아, 적어도 표본 규모 측면에서는 더 넓은 데이터 기반을 갖는다. 큰 데이터셋은 딥러닝 모델이 다양한 파형과 생리적 변동을 학습하는 데 유리하다.

네 번째 강점은 architecture ablation이 포함되어 있다는 점이다. 1 CNN + 1 BiLSTM부터 3 CNN + 2 BiLSTM까지 구조를 비교하여, 최종 구조가 경험적으로 가장 좋은 선택임을 보여준다. 이는 모델 설계가 단순한 임의 선택이 아니라 실험적으로 검증되었음을 의미한다.

다섯 번째 강점은 fold별 결과를 제공한다는 점이다. 평균 성능뿐 아니라 각 fold의 MAE와 MSE를 제시하여 성능의 변동성을 어느 정도 확인할 수 있다. 특히 DBP의 fold 간 표준편차가 낮아 안정적인 예측을 보인다는 점이 뚜렷하다.

### 5.2 한계

가장 중요한 한계는 data split 방식의 불명확성이다. 논문은 5-fold cross-validation 전에 dataset을 shuffling했다고 설명하지만, subject-wise split을 적용했다고 명시하지 않는다. 또한 segmentation에서 75% overlap을 사용했기 때문에 인접 segment들이 매우 유사할 수 있다. 만약 동일 환자 또는 overlapping segment가 training fold와 test fold에 동시에 포함되었다면, 모델 성능은 실제 새로운 환자에 대한 일반화 성능보다 과대평가될 수 있다. PPG 기반 혈압 추정 연구에서는 이러한 data leakage 문제가 매우 중요하다.

두 번째 한계는 ABP 기반 target extraction의 세부 정확성이 충분히 설명되지 않았다는 점이다. Systolic 및 diastolic peak detection을 수행했다고 하지만, peak detection algorithm의 구체적 방식, artifact 처리, 비정상 박동 처리, segment 내 peak 수가 부족한 경우의 처리 방법은 제공된 텍스트에 자세히 나타나지 않는다.

세 번째 한계는 모델의 실제 임상적 검증이 부족하다는 점이다. 실험은 MIMIC-II 기반 공개 ICU 데이터셋에서 수행되었다. ICU 데이터는 신호 품질과 환자 상태가 일반 웨어러블 환경과 다를 수 있다. 실제 스마트워치나 fitness tracker에서 측정되는 PPG는 움직임, 센서 접촉 불량, 피부색, 주변광, 체온, 활동 상태의 영향을 크게 받는다. 논문은 웨어러블 적용 가능성을 강조하지만, 실제 웨어러블 기기에서의 실시간 검증은 수행하지 않았다.

네 번째 한계는 임상 혈압계 검증 기준이 충분히 다뤄지지 않았다는 점이다. 논문은 MAE와 MSE를 주요 지표로 사용하고 Bland–Altman plot을 제시하지만, AAMI 또는 BHS 기준과 같은 국제 혈압계 검증 기준을 직접 적용한 결과는 제공하지 않는다. 이전의 다른 BP estimation 논문들이 AAMI/BHS 기준을 자주 사용한다는 점을 고려하면, 임상적 수용 가능성을 더 명확히 평가할 필요가 있다.

다섯 번째 한계는 demographic 정보가 사용되지 않았다는 점이다. 혈압과 PPG morphology는 나이, 성별, 체중, 혈관 탄성, 질환 상태, 약물 복용 여부 등과 밀접하게 관련될 수 있다. 이 논문은 PPG만 사용한다는 장점을 가지지만, 개인차를 보정하기 위한 personalized calibration 또는 demographic covariate를 사용하지 않는다. 따라서 다양한 인구 집단에서 성능이 유지되는지는 추가 검증이 필요하다.

여섯 번째 한계는 모델 설명가능성의 부족이다. Attention mechanism을 사용하지만, attention weight가 실제로 어떤 PPG 구간을 중요하게 보는지, 그 구간이 생리학적으로 어떤 의미를 갖는지에 대한 분석은 제공된 텍스트에서 충분히 확인되지 않는다. 의료 응용에서는 단순 성능뿐 아니라 모델이 어떤 신호 특성에 의존하는지 이해하는 것이 중요하다.

### 5.3 비판적 해석

이 논문은 PPG 기반 cuff-less BP estimation에서 hybrid CNN-BiLSTM-Attention 구조가 매우 높은 정확도를 달성할 수 있음을 보여준다. 구조 자체는 생체신호 처리에 적합하다. CNN은 파형의 local morphology를 추출하고, BiLSTM은 temporal context를 학습하며, attention은 중요한 구간을 강조한다. 이러한 조합은 PPG와 같은 1차원 생체 시계열 데이터에 자연스럽게 맞는다.

그러나 보고된 MAE가 매우 낮다는 점은 동시에 신중한 해석을 요구한다. SBP MAE 1.88, DBP MAE 1.34는 기존 문헌 대비 매우 우수한 수치이다. 이러한 성능이 실제 cross-subject, cross-device, real-world wearable 환경에서도 유지되는지는 논문 텍스트만으로 확인할 수 없다. 특히 subject-wise splitting과 overlap segment 관리가 명확히 설명되지 않은 점은 중요한 평가상 약점이다.

따라서 이 연구는 모델 구조와 성능 측면에서 강력한 결과를 제시하지만, 임상 또는 제품 적용 가능성을 확정하기 위해서는 더 엄격한 validation protocol이 필요하다. 예를 들어 subject-wise cross-validation, leave-one-subject-out evaluation, external dataset validation, motion artifact 환경 검증, wearable on-device inference 평가가 필요하다.

## 6. 결론

이 논문은 PPG 신호만을 이용한 비침습적 혈압 추정을 위해 CNN, BiLSTM, attention mechanism을 결합한 hybrid deep learning model을 제안하였다. 모델은 8.192초 길이의 PPG segment를 입력으로 받고, 대응되는 ABP segment에서 추출한 평균 SBP와 DBP를 target으로 학습한다.

최종 구조는 세 개의 CNN layer, 두 개의 BiLSTM layer, 하나의 attention layer로 구성된다. CNN은 PPG의 국소적 파형 특징을 추출하고, BiLSTM은 양방향 시간 의존성을 학습하며, attention은 혈압 추정에 중요한 time step을 강조한다. 이 조합은 PPG 기반 혈압 추정 문제에서 spatial feature extraction, temporal modeling, informative segment selection을 통합적으로 수행한다.

MIMIC-II 기반 2064명 데이터셋에서 5-fold cross-validation을 수행한 결과, 제안 모델은 SBP MAE 1.88, DBP MAE 1.34를 달성했다. 이는 논문에서 비교한 기존 PPG 기반 혈압 추정 방법들보다 우수한 성능이다. 특히 ECG나 추가 derivative signal 없이 PPG 단일 신호만 사용했다는 점에서 웨어러블 적용 가능성이 높다.

그럼에도 불구하고 이 연구의 결과를 실제 임상 또는 상용 웨어러블 환경으로 확장하기 위해서는 추가 검증이 필요하다. 가장 중요한 보완점은 subject-wise split 여부를 명확히 하고, overlapping segment에 의한 data leakage 가능성을 제거하는 것이다. 또한 외부 데이터셋, 다양한 인구 집단, 실제 활동 중 PPG, 장기 모니터링 환경에서 성능을 검증해야 한다.

종합적으로 이 논문은 PPG 기반 cuff-less blood pressure monitoring 분야에서 CNN-BiLSTM-Attention 구조의 강력한 가능성을 보여주는 연구이다. 특히 단일 PPG 신호만으로 높은 정확도를 달성했다는 점에서 웨어러블 혈압 모니터링 연구에 중요한 참고 모델로 볼 수 있다. 향후 엄격한 cross-subject 검증과 실제 기기 환경 평가가 추가된다면, 연속적이고 편안한 혈압 모니터링 기술 발전에 의미 있는 기반이 될 수 있다.
