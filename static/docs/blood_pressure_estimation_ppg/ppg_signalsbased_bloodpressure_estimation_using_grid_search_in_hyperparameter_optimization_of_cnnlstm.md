# PPG Signals-Based Blood-Pressure Estimation Using Grid Search in Hyperparameter Optimization of CNN–LSTM

* **저자**: Nurul Qashri Mahardika T, Yunendah Nur Fuadah, Da Un Jeong, Ki Moo Lim
* **발표연도**: 2023

## 1. 논문 개요

이 논문은 photoplethysmography, 즉 PPG 신호만을 이용하여 continuous non-invasive blood pressure, cNIBP를 추정하는 deep learning 기반 방법을 제안한다. 논문의 핵심 목표는 PPG 신호에서 혈압 추정에 유용한 형태적 및 시간적 정보를 자동으로 추출하고, grid search를 이용해 CNN–LSTM 모델의 hyperparameter를 최적화함으로써 systolic blood pressure, SBP와 diastolic blood pressure, DBP를 더 정확하게 예측하는 것이다.

연구 문제는 기존 PPG 기반 혈압 추정 연구들이 여전히 충분한 정확도와 안정성을 확보하지 못했고, 특히 deep learning 모델에서 hyperparameter를 수동으로 선택하는 과정이 overfitting, underfitting, 긴 계산 시간, 성능 불안정성을 유발할 수 있다는 점이다. 혈압 측정은 고혈압, 저혈압, 심혈관 질환, 중환자 관리 등에서 매우 중요한 생체 지표이며, 침습적 arterial blood pressure, ABP 측정은 정확하지만 감염, 출혈, 혈관 손상 등의 위험이 있다. 반면 cuff-based sphygmomanometer는 비교적 안전하지만 cuff 압박으로 불편하고 연속 측정에 적합하지 않다. 따라서 PPG 기반 cuffless BP estimation은 wearable device 또는 지속적 모니터링 시스템에서 중요한 연구 주제이다.

논문은 MIMIC III database에서 PPG와 ABP 신호를 추출하여 사용한다. PPG 신호는 deep learning 모델의 입력으로 사용되고, ABP 신호에서 얻은 SBP와 DBP가 reference label로 사용된다. 저자들은 low-quality waveform을 제거하고, PPG 신호를 peak-to-peak 기준으로 두 cycle 단위로 segmenting한 뒤, spline interpolation을 통해 모든 segment의 길이를 200 sample로 맞춘다. 이후 z-score standardization을 적용하여 환자별 PPG amplitude 차이를 줄인다.

제안 모델은 CNN–LSTM이다. CNN은 PPG waveform의 local morphological pattern을 추출하고, LSTM은 순차적 temporal dependency를 학습한다. 모델의 hyperparameter는 grid search와 five-fold cross-validation을 통해 선택된다. 비교 대상으로는 LSTM 단독 모델과 LSTM–autoencoder 모델이 사용된다.

최종적으로 CNN–LSTM이 LSTM 및 LSTM–autoencoder보다 훨씬 우수한 성능을 보였다. Table 2 기준으로 CNN–LSTM은 SBP에서 MAE 3.64 mmHg, SD 7.04 mmHg, DBP에서 MAE 2.39 mmHg, SD 3.79 mmHg를 달성했다. Table 3의 standard evaluation에서는 CNN–LSTM이 SBP에서 MAD 5.34 mmHg, ME 0.13 mmHg, SD 7.04 mmHg, BHS Grade B를 보였고, DBP에서 MAD 2.89 mmHg, ME 0.48 mmHg, SD 3.79 mmHg, BHS Grade A를 보였다. 본문과 abstract에는 성능 수치의 표기 방식이 일부 일관되지 않으므로, 본 보고서에서는 Table 2와 Table 3의 값을 구분하여 해석한다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 PPG waveform을 사람이 설계한 feature로 변환하지 않고, CNN–LSTM이 직접 waveform representation을 학습하도록 하면서, grid search로 모델 학습에 중요한 hyperparameter를 자동으로 최적화하는 것이다.

첫 번째 핵심 직관은 PPG 신호가 혈압 waveform과 관련된 morphological information을 포함한다는 점이다. PPG는 혈관 내 혈액량 변화에 따른 광 반사 강도 변화를 측정하며, cardiac cycle에 따라 주기적인 pulse waveform을 형성한다. 이 waveform의 peak, trough, pulse width, rising slope, falling slope 등은 arterial pulse의 systolic 및 diastolic 특성과 관련될 수 있다. 논문은 특히 PPG의 systolic peak-to-peak 구간이 arterial pulse의 systolic 및 diastolic component를 포괄적으로 반영한다고 보고, 두 개의 PPG pulse cycle을 입력 단위로 사용한다.

두 번째 핵심 직관은 CNN과 LSTM의 역할 분담이다. CNN은 convolution operation을 통해 waveform의 local pattern을 추출하는 데 강하다. 예를 들어 PPG pulse의 상승부, peak 주변의 형태, 하강부, notch 또는 곡률 변화와 같은 국소적 형태 정보를 포착할 수 있다. 반면 LSTM은 time-series data에서 장기 의존성 및 순차적 변화를 학습하는 데 적합하다. 따라서 CNN–LSTM은 PPG waveform의 형태적 정보와 시간적 정보를 함께 활용할 수 있다.

세 번째 핵심 아이디어는 hyperparameter optimization이다. Deep learning 모델은 optimizer, learning rate, batch size, dropout rate, layer 구성 등 여러 hyperparameter에 민감하다. 기존 연구에서는 이런 값을 수동으로 선택하는 경우가 많았고, 잘못된 설정은 overfitting 또는 underfitting을 유발할 수 있다. 이 논문은 grid search를 통해 optimizer, learning rate, batch size의 조합을 체계적으로 탐색하고, five-fold cross-validation을 통해 최적 조합을 선택한다. 이를 통해 수동 tuning의 부담을 줄이고 모델 성능을 안정화하려 한다.

기존 접근 방식과의 차별점은 두 가지로 정리할 수 있다. 첫째, PPG 신호만을 사용하여 ECG 없이 혈압을 추정하므로 sensor 구성 측면에서 단순하다. 둘째, CNN–LSTM 구조에 grid search를 결합하여 hyperparameter 선택 문제를 명시적으로 다루었다. 논문은 이를 통해 기존 LSTM, LSTM–autoencoder, SVM, SVR, ResNet, RNN, U-Net, AdaBoost, CNN–BiLSTM 기반 연구보다 경쟁력 있는 성능을 달성했다고 주장한다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

논문의 전체 파이프라인은 데이터 수집, 전처리, segmentation, standardization, model training, hyperparameter tuning, test evaluation 순서로 구성된다.

먼저 MIMIC III database에서 PPG와 ABP 신호를 추출한다. PPG는 입력 신호이고, ABP는 reference SBP와 DBP를 얻기 위한 ground truth 신호이다. 이후 quality가 낮은 waveform, flatline, flat peak, incomplete ABP 또는 PPG waveform을 자동으로 제거한다. 그런 다음 PPG artifact와 baseline drift를 줄이기 위해 discrete wavelet decomposition, DWT 기반 filtering을 적용한다.

전처리된 PPG 신호는 systolic peak-to-peak 기준으로 두 cycle씩 segmenting된다. 두 cycle 단위로 자르는 이유는 하나의 pulse만 사용하는 것보다 systolic 및 diastolic waveform 특성을 더 포괄적으로 담을 수 있기 때문이다. 각 segment의 길이는 심박수에 따라 달라질 수 있으므로 spline interpolation을 이용하여 모두 200 sample로 맞춘다. 이후 abnormal BP label을 제거하고 z-score standardization을 적용한다.

이렇게 만들어진 PPG segment는 LSTM, LSTM–autoencoder, CNN–LSTM 모델에 입력된다. Training 과정에서는 grid search와 five-fold cross-validation을 통해 optimizer, learning rate, batch size 조합을 선택한다. 최종 선택된 모델은 test set에서 SBP와 DBP를 예측하고, MAE, ME, SD, BHS, AAMI, IEEE 기준으로 평가된다.

### 3.2 데이터셋

논문은 MIMIC III waveform database를 사용한다. MIMIC III는 ICU 환자의 생체신호를 포함하는 공개 데이터베이스이며, 본 연구에서는 125 Hz sampling frequency와 8-bit precision으로 수집된 PPG 및 ABP 신호를 사용한다.

저자들은 먼저 저장 공간과 waveforms의 품질 문제를 고려하여 100명의 환자를 무작위로 선택했고, 그중 high-quality waveform을 가진 55명의 환자 데이터를 최종적으로 사용했다. 각 환자에서 37,500 sample, 즉 300초 길이의 신호를 무작위로 추출했다. 이후 preprocessing을 거쳐 총 75,226개 signal segment를 얻었다고 설명한다.

다만 논문 내 데이터 개수에는 약간의 불일치가 있다. Abstract에서는 75,226 segment 중 60,180개를 training, 12,030개를 validation, 15,045개를 test에 할당했다고 쓰여 있는데, 이 세 값을 합치면 87,255개로 75,226개와 맞지 않는다. Results section에서는 training data가 SBP 예측에 65,000개, DBP 예측에 65,235개, test data가 SBP에 14,763개, DBP에 15,423개였다고 설명한다. 따라서 실제 split 개수는 본문 내에서 일관되지 않다. 이 보고서에서는 논문이 최종적으로 55명 환자와 약 75,226개 segment를 사용했다는 점, 그리고 training/validation/test split 및 five-fold cross-validation을 수행했다는 점까지만 확정적으로 해석한다.

### 3.3 DWT 기반 PPG 전처리

PPG 신호에는 motion artifact와 baseline drift가 포함될 수 있다. 논문은 이를 줄이기 위해 one-dimensional discrete wavelet decomposition, 1D DWT를 사용했다. 사용한 wavelet은 Daubechies order four, db4이며, decomposition level은 8이다.

1D DWT는 신호를 low-frequency component와 high-frequency component로 나누며, 각각 approximation coefficient와 detail coefficient를 생성한다. 8-level decomposition을 수행하면 하나의 approximation sub-band와 여덟 개의 detail sub-band, 총 아홉 개의 sub-band가 생성된다. 논문은 125 Hz sampling frequency에서 8-level DWT가 PPG 신호의 구성 성분을 보기 충분한 bandwidth sub-band를 제공한다고 설명한다.

DWT를 사용하는 이유는 PPG waveform에서 baseline drift와 artifact를 줄이면서 pulse morphology에 필요한 성분을 보존하기 위해서이다. 다만 제공된 텍스트에서는 어떤 detail coefficient를 제거하거나 유지했는지, thresholding을 적용했는지, inverse DWT를 어떻게 수행했는지는 구체적으로 설명되지 않는다. 따라서 DWT filtering의 정확한 재현 절차는 제공된 텍스트만으로는 완전하지 않다.

### 3.4 Peak-to-peak segmentation과 spline interpolation

논문은 PPG 신호를 두 개의 systolic peak-to-peak cycle로 segmenting한다. PPG waveform에서 systolic peak는 cardiac circulation 중 혈액량이 최대가 되는 시점을 나타내며, arterial wall에 가해지는 압력 변화와 관련된다. Peak-to-peak measurement는 연속적인 pulse cycle 사이의 변화를 반영하므로 혈압 추정에 유용한 입력 단위로 간주된다.

본 연구는 Athaya et al.의 overlapping segmentation 접근을 참고하여, PPG segment를 두 cycle 단위로 구성하되 한 cycle은 overlap되도록 했다. 즉 인접 segment가 하나의 pulse cycle을 공유하는 방식으로 sample 수를 늘리고 waveform continuity를 유지하려 한 것으로 보인다.

하지만 두 cycle의 실제 sample 수는 individual heart rate에 따라 달라진다. 예를 들어 심박수가 빠르면 두 cycle의 길이가 짧고, 심박수가 느리면 길이가 길어진다. Deep learning 모델에 입력하려면 모든 segment의 길이가 같아야 하므로, 논문은 spline interpolation을 사용해 segment 길이를 200 sample로 통일했다. 가장 긴 sample length가 200이었고, 200보다 짧은 segment는 spline interpolation으로 새로운 point를 추가하여 길이를 맞췄다.

Spline interpolation은 기존 sample 사이에 부드러운 곡선을 맞춰 새로운 data point를 생성하는 방식이다. PPG waveform의 연속성과 smoothness를 유지하면서 일정한 입력 길이를 만들 수 있다는 장점이 있다. 그러나 interpolation은 실제로 존재하지 않는 point를 생성하므로, waveform의 미세한 형태를 왜곡할 가능성도 있다. 논문은 이 잠재적 왜곡에 대한 별도 분석은 제공하지 않는다.

### 3.5 Abnormal BP 제거와 z-score standardization

논문은 Chobanian et al.이 제시한 혈압 범위를 바탕으로 abnormal BP value를 제거했다. 제거 기준은 SBP ≥ 200 mmHg, DBP ≥ 120 mmHg, SBP ≤ 80 mmHg, DBP ≤ 40 mmHg이다. 전처리 후 최종 SBP 범위는 80.21–180.47 mmHg였고, DBP 범위는 40–80.78 mmHg였다.

이후 z-score standardization을 적용했다. PPG amplitude와 variability는 환자의 움직임, 건강 상태, 측정 장치, sensor contact 등에 따라 달라질 수 있다. 이를 줄이기 위해 각 PPG segment 또는 데이터 단위에서 평균과 표준편차를 이용해 정규화한다.

논문이 제시한 standardization 수식은 다음과 같다.

$$
X_{standardization} = \frac{X - mean(X)}{Standard\ Deviation(X)}
$$

여기서 $X$는 PPG signal이다. 표준편차는 다음과 같이 계산된다.

$$
\sigma =
\sqrt{
\frac{\sum (x-\mu)^2}{N}
}
$$

여기서 $\sigma$는 PPG signal의 standard deviation, $x$는 PPG signal 값, $\mu$는 평균값, $N$은 sample 수이다. 이 과정은 PPG pulse 값들이 평균에서 얼마나 떨어져 있는지를 기준으로 scale을 맞추는 역할을 한다.

### 3.6 Grid search 기반 hyperparameter tuning

논문은 deep learning 모델의 hyperparameter 선택을 위해 grid search를 사용했다. Grid search는 미리 정의한 후보 hyperparameter 조합을 모두 실험하고, cross-validation 결과가 가장 좋은 조합을 선택하는 방법이다. 이 연구에서 탐색한 hyperparameter는 optimizer, learning rate, batch size이다.

Optimizer 후보에는 Adam, Adadelta, RMSprop, SGD가 포함된다. Batch size 후보는 32, 64, 128이고, learning rate 후보는 0.001, 0.01, 0.005, 0.05이다. 논문은 이러한 후보들이 machine learning과 deep learning에서 자주 사용되고 효과가 검증된 값들이라고 설명한다.

SGD는 loss gradient 방향으로 parameter를 업데이트한다.

$$
W_{new} = W_{old} - \alpha \nabla L(W_{old}, x_i, y_i)
$$

여기서 $W_{new}$는 갱신된 weight, $W_{old}$는 이전 weight, $\alpha$는 learning rate, $\nabla L$은 loss function의 gradient이다.

RMSprop은 이전 gradient의 squared average를 이용해 parameter별 learning rate를 조정한다.

$$
W_{new} = W_{old} - \frac{\alpha}{\sqrt{MeanSquare(W,t)}} \nabla L(W_{old})
$$

$$
MeanSquare(W,t) = \rho MeanSquare(W,t-1) + (1-\rho)(\nabla L(W))^2
$$

Adam은 momentum과 RMSprop의 아이디어를 결합한 optimizer이다. 1차 moment와 2차 moment를 각각 다음과 같이 업데이트한다.

$$
m_t = \rho_1 m_{t-1} + (1-\rho_1)g_t
$$

$$
u_t = \rho_2 u_{t-1} + (1-\rho_2)g_t^2
$$

논문에는 두 번째 식에서 $\rho_1$로 표기된 부분이 있으나, Adam의 일반적 정의와 문맥상 $\rho_2$가 맞다. Bias-corrected moment는 다음과 같다.

$$
\hat{m}_t = \frac{m_t}{1-\rho_1^t}
$$

$$
\hat{u}_t = \frac{u_t}{1-\rho_2^t}
$$

최종 parameter update는 다음과 같다.

$$
W_{new} = W_{old} + \Delta W
$$

$$
\Delta W = -\alpha \frac{\hat{m}_t}{\sqrt{\hat{u}_t}+\epsilon}
$$

Adadelta는 AdaGrad의 확장형으로, accumulated gradient에 따라 update scale을 조절한다. 논문은 다음과 같은 형태로 제시한다.

$$
W_{i+1} = W_t - \frac{RMS[\Delta W]_{i-1}}{RMS[g]_t}g_t
$$

이러한 optimizer 후보와 learning rate, batch size 후보를 grid search로 조합하여, 각 모델 및 SBP/DBP 예측 task별 최적 hyperparameter를 선택한다.

### 3.7 LSTM architecture

비교 모델 중 하나인 LSTM은 time-series data의 long-term dependency를 학습하기 위한 recurrent architecture이다. 일반 RNN은 장기 정보를 유지하기 어렵지만, LSTM은 input gate, forget gate, output gate, cell state를 통해 오래된 정보 중 필요한 것은 유지하고 불필요한 것은 삭제할 수 있다.

논문이 제안한 LSTM architecture는 두 개의 LSTM layer로 구성된다. 첫 번째 LSTM layer는 25 unit, 두 번째 LSTM layer는 50 unit을 사용한다. 각 LSTM unit은 input gate와 forget gate를 통해 cell state에 어떤 정보를 추가하거나 제거할지 결정하고, output gate를 통해 hidden layer state를 결정한다. Overfitting을 줄이기 위해 dropout rate 0.2와 batch normalization이 사용된다.

LSTM 단독 모델은 PPG sequence의 temporal structure를 직접 학습하려 하지만, 실험 결과 CNN–LSTM보다 성능이 크게 낮았다. 이는 PPG waveform에서 혈압과 관련된 국소 morphology pattern을 먼저 추출하지 않으면 LSTM만으로는 충분한 representation을 얻기 어렵다는 해석이 가능하다.

### 3.8 LSTM–autoencoder architecture

두 번째 비교 모델은 LSTM–autoencoder이다. Autoencoder는 encoder와 decoder로 구성된다. Encoder는 입력 sequence를 압축된 latent representation으로 변환하고, decoder는 이 representation을 다시 원래 입력 형태와 유사하게 복원한다. 이 구조는 sequence representation learning에 사용할 수 있다.

논문은 encoder와 decoder 각각에 네 개의 LSTM layer를 사용했다고 설명한다. Encoder의 마지막 layer가 sequence를 반환하지 않기 때문에, decoder 입력으로 사용하기 위해 repeat vector를 적용하여 time-step 차원을 맞춘다. Dropout rate는 0.2이고, kernel initializer로 Glorot Normal을 사용하여 overfitting을 줄이려 했다.

LSTM–autoencoder는 LSTM 단독 모델보다 DBP에서는 일부 개선되었지만, SBP 성능은 여전히 좋지 않았다. 이는 reconstruction 기반 representation이 BP regression에 직접 최적화된 feature를 충분히 만들지 못했거나, 데이터 규모와 모델 복잡도 사이의 균형이 맞지 않았을 가능성을 시사한다.

### 3.9 CNN–LSTM architecture

제안된 핵심 모델은 1D CNN–LSTM이다. 이 모델은 다섯 개의 CNN layer, 하나의 LSTM layer, 두 개의 fully connected layer로 구성된다. CNN layer는 morphological feature extraction을 담당하고, LSTM layer는 temporal feature extraction을 담당한다.

CNN 부분은 다섯 개의 convolutional layer와 max-pooling layer로 구성되며, 각 convolutional layer는 ReLU activation function을 사용한다. Max pooling은 feature map dimension을 줄여 계산량을 줄이고, 중요한 feature를 압축하는 역할을 한다. Convolution과 pooling을 거친 feature map은 flattening을 통해 feature vector로 변환된다.

그 다음 LSTM layer가 이 feature vector sequence를 처리하여 temporal dependency를 학습한다. 마지막으로 두 개의 fully connected layer가 연결되어 SBP와 DBP 값을 예측한다. 논문의 설명에 따르면 SBP와 DBP에 대해 각각 최적 hyperparameter를 별도로 선택한 것으로 보이며, 최종 모델 구성은 abstract 기준으로 다섯 개 convolutional layer, 하나의 LSTM layer, 두 개 fully connected layer이다.

CNN–LSTM이 가장 좋은 성능을 보인 이유는 CNN이 PPG waveform의 morphology를 자동으로 추출하고, LSTM이 이 feature들의 순차적 변화를 추가적으로 학습하기 때문이라고 해석할 수 있다. 단순 LSTM은 raw PPG sequence에서 local morphology를 효과적으로 추출하지 못할 수 있고, LSTM–autoencoder는 reconstruction 목적과 BP regression 목적이 완전히 일치하지 않을 수 있다.

### 3.10 Loss function과 평가 기준

논문은 model evaluation metric으로 MSE를 사용했다고 설명한다. Training loss도 명시적으로 MSE 기반으로 해석할 수 있다. MSE는 예측값과 실제값의 squared error 평균이다.

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

최종 성능 평가는 IEEE, BHS, AAMI 기준으로 이루어진다.

IEEE standard에서는 mean absolute difference, MAD 또는 MAE가 사용된다. 논문은 다음 수식을 제시한다.

$$
MAD = \frac{\sum_{i=1}^{n}|p_i-y_i|}{n}
$$

여기서 $p_i$는 test measurement, $y_i$는 reference measurement의 평균이다. IEEE 기준에서는 MAD가 5 mmHg 이하이면 Grade A, 5–6이면 Grade B, 6–7이면 Grade C, 그보다 낮으면 D로 평가된다. 다만 논문 Table 1에는 IEEE Grade A 기준이 ≤5 mmHg라고 되어 있으나, Table 3에서는 “MAD ≤4 mmHg”라는 표기가 함께 나타난다. 이 부분은 표기상 불일치가 있다.

AAMI standard는 mean error, ME와 standard deviation, SD를 평가한다. ME는 다음과 같다.

$$
Mean\ error = \frac{1}{N}\sum_{i=1}^{n}(y_i-\hat{y}_i)
$$

AAMI 기준은 ME ≤ 5 mmHg, SD ≤ 8 mmHg이다. SD는 다음과 같이 제시된다.

$$
SD =
\sqrt{
\frac{1}{n-1}\sum_{i=1}^{n}(y_i-\bar{y})^2
}
$$

다만 이 수식은 일반적인 error standard deviation이라기보다 값 자체의 standard deviation처럼 표기되어 있다. AAMI 평가에서는 일반적으로 error의 standard deviation을 사용하는 것이 자연스럽다. 본문 결과 해석에서는 predicted error의 SD로 사용된 것으로 보인다.

BHS standard는 absolute error가 5, 10, 15 mmHg 이하인 sample의 cumulative percentage, CP를 기준으로 Grade A, B, C를 부여한다. Grade A 기준은 각각 60%, 85%, 95% 이상이다.

## 4. 실험 및 결과

### 4.1 비교 모델과 실험 구성

논문은 LSTM, LSTM–autoencoder, CNN–LSTM 세 모델을 비교한다. 모두 PPG 신호를 입력으로 사용하며, SBP와 DBP를 추정한다. Grid search와 five-fold cross-validation을 통해 각 모델의 optimizer, learning rate, batch size를 선택한다.

Grid search 결과, LSTM 모델에서는 SBP 예측에 Adadelta optimizer, learning rate 0.001, batch size 32가 선택되었고, DBP 예측에는 RMSprop, learning rate 0.001, batch size 64가 선택되었다. LSTM–autoencoder에서는 SBP 예측에 Adadelta, learning rate 0.001, batch size 64가 선택되었고, DBP 예측에는 Nadam, learning rate 0.001, batch size 32가 선택되었다. 다만 Methods에서 optimizer 후보로 Adam, Adadelta, RMSprop, SGD를 언급했는데, Results에서는 Nadam이 선택되었다고 되어 있어 후보 목록과 결과 사이에 불일치가 있다.

CNN–LSTM에서는 SBP 예측에 Adadelta, learning rate 0.001, batch size 64가 선택되었고, DBP 예측에는 RMSprop, learning rate 0.001, batch size 64가 선택되었다. 이 결과는 SBP와 DBP가 서로 다른 error landscape와 최적화 특성을 가질 수 있음을 보여준다.

### 4.2 Table 2 기준 성능 비교

Table 2에 따르면 LSTM 단독 모델은 SBP에서 MAE 14.2 mmHg, SD 20.7 mmHg, DBP에서 MAE 7.53 mmHg, SD 10.01 mmHg를 보였다. 이는 매우 큰 오차이며, 특히 SBP 추정에서 성능이 낮다.

LSTM–autoencoder는 LSTM보다 일부 개선되었다. SBP에서 MAE 13.45 mmHg, SD 19.01 mmHg, DBP에서 MAE 5.71 mmHg, SD 7.67 mmHg를 달성했다. DBP에서는 AAMI 기준의 SD 8 mmHg 이하에 근접하거나 만족할 수 있는 수준까지 개선되었지만, SBP에서는 여전히 큰 오차를 보였다.

CNN–LSTM은 세 모델 중 가장 좋은 성능을 보였다. Table 2 기준으로 SBP에서 MAE 3.64 mmHg, SD 7.04 mmHg, DBP에서 MAE 2.39 mmHg, SD 3.79 mmHg를 기록했다. 이는 LSTM 단독 및 LSTM–autoencoder와 비교할 때 큰 성능 향상이다. 특히 SBP MAE가 14.2에서 3.64로 크게 줄어든 것은 CNN 기반 local feature extraction이 PPG 혈압 추정에서 핵심적 역할을 한다는 점을 보여준다.

논문은 기존 연구들과도 비교한다. 예를 들어 Slapnicar et al.의 spectro-temporal ResNet은 MIMIC III에서 SBP MAE 9.43, DBP MAE 6.88을 보였고, RNN 기반 연구는 SBP MAE 12.08, DBP MAE 5.56을 보였다. U-Net 기반 PPG2ABP 연구는 SBP MAE 5.73, DBP MAE 3.45로 보고되었다. CNN–LSTM은 이러한 기존 연구보다 낮은 MAE를 보이는 것으로 제시된다. 다만 각 연구의 데이터 분할, patient 수, preprocessing, 평가 조건이 다르기 때문에 절대적으로 공정한 비교라고 보기는 어렵다.

### 4.3 LSTM 결과 해석

LSTM 모델의 Bland–Altman plot과 error histogram은 Figure 8에 제시된다. SBP의 pressure range는 75–180 mmHg, DBP는 40–90 mmHg로 나타난다. LSTM 모델은 SBP와 DBP 모두에서 error dispersion이 크며, 특히 SBP의 predicted error deviation이 DBP보다 훨씬 컸다.

Table 3 기준으로 LSTM의 SBP는 MAD 14.281 mmHg, ME -0.49 mmHg, SD 20.7 mmHg이고, BHS 기준 CP5 30.92%, CP10 53.07%, CP15 67.37%로 Grade D이다. DBP는 MAD 7.53 mmHg, ME -0.21 mmHg, SD 10.01 mmHg이며, CP5 45.14%, CP10 72.02%, CP15 86%로 BHS Grade C이다. LSTM은 IEEE, AAMI, BHS 기준을 만족하지 못한다.

이 결과는 단순 LSTM이 PPG waveform 기반 혈압 추정에 충분하지 않음을 보여준다. Raw 또는 standardized PPG segment를 순차적으로 처리하더라도, 혈압과 관련된 morphology feature를 명시적으로 추출하는 능력이 부족했을 가능성이 있다.

### 4.4 LSTM–autoencoder 결과 해석

LSTM–autoencoder의 결과는 Figure 9에 제시된다. SBP의 ME는 -0.94 mmHg이고 confidence interval은 약 -38.2에서 36.3 mmHg로 매우 넓다. DBP의 ME는 -0.56 mmHg이고 confidence interval은 -15.56에서 14.48 mmHg로 SBP보다 좁다.

Table 3 기준으로 LSTM–autoencoder의 SBP는 MAD 26.94로 표기되어 있는데, Table 2의 SBP MAE 13.45와 일치하지 않는다. Table 3의 “SBP 26.94”는 실제 MAD라기보다 CP5 값 또는 다른 지표가 잘못 배치된 것일 가능성이 있다. 그러나 같은 행에서 CP5는 27로 제시되어 있어 중복 또는 표기 오류가 의심된다. DBP는 MAD 5.71, ME -0.56, SD 7.67로 Table 2와 일치하며, BHS Grade B를 달성한 것으로 제시된다.

LSTM–autoencoder는 LSTM보다 DBP 추정에서는 개선되었지만, SBP 추정은 여전히 매우 낮은 성능을 보였다. 이는 autoencoder의 reconstruction-oriented representation이 혈압값 예측에 충분히 discriminative하지 않았거나, SBP가 DBP보다 PPG morphology에서 더 복잡한 정보를 요구하기 때문일 수 있다.

### 4.5 CNN–LSTM 결과 해석

CNN–LSTM은 가장 좋은 결과를 보인다. Figure 10의 Bland–Altman plot에서 SBP prediction error는 대략 -40에서 40 mmHg 사이에 분포하고, DBP prediction error는 -10에서 10 mmHg 사이에 분포한다. SBP의 ME는 -0.13 mmHg이고 confidence interval은 -15.6에서 15.32 mmHg이다. DBP의 ME는 0.45 mmHg이고 confidence interval은 -6.8에서 7.8 mmHg이다. Error histogram은 0 mmHg 근처에 집중되어 있다.

Table 3 기준으로 CNN–LSTM의 SBP는 MAD 5.34 mmHg, MAPD 0.04, ME 0.13 mmHg, SD 7.04 mmHg이다. BHS 기준 cumulative percentage는 CP5 63.4%, CP10 85.9%, CP15 92.78%로 Grade B이다. DBP는 MAD 2.89 mmHg, MAPD 0.05, ME 0.48 mmHg, SD 3.79 mmHg이다. BHS 기준 CP5 81.70%, CP10 98.28%, CP15 100%로 Grade A이다.

AAMI 기준으로 보면 CNN–LSTM은 SBP와 DBP 모두 ME ≤ 5 mmHg 및 SD ≤ 8 mmHg를 만족한다. IEEE 기준에서는 DBP가 Grade A이고, SBP는 Table 3 기준 Grade B이다. BHS 기준에서는 SBP Grade B, DBP Grade A이다. 따라서 “모든 기준을 완전히 최고 등급으로 만족했다”기보다는, AAMI는 통과했고, BHS에서는 SBP가 Grade B, DBP가 Grade A이며, IEEE에서는 SBP가 Grade B, DBP가 Grade A라고 해석하는 것이 정확하다.

### 4.6 기존 연구와의 비교

논문은 여러 기존 PPG 기반 혈압 추정 연구와 CNN–LSTM을 비교한다. SVR 기반 연구는 MIMIC II에서 SBP MAE 8.54, DBP MAE 4.34를 보였고, SVM 기반 연구는 Queensland dataset에서 SBP MAE 11.6, DBP MAE 7.6을 보였다. Spectro-temporal ResNet은 MIMIC III에서 SBP MAE 9.43, DBP MAE 6.88을 보였다. RNN은 MIMIC III에서 SBP MAE 12.08, DBP MAE 5.56을 보였고, U-Net 기반 PPG-to-ABP 방식은 SBP MAE 5.73, DBP MAE 3.45를 보였다.

Table 2 기준 CNN–LSTM의 SBP MAE 3.64와 DBP MAE 2.39는 이들보다 낮다. 따라서 논문은 CNN–LSTM이 PPG-only BP estimation에서 유망한 성능을 보인다고 주장한다. 다만 기존 연구들과의 비교는 데이터셋 버전, subject split, preprocessing, signal length, evaluation protocol이 다르기 때문에 엄밀한 apples-to-apples comparison은 아니다. 특히 이 논문은 55명의 환자만 사용했으므로, 더 큰 subject-independent dataset에서 같은 성능이 유지되는지는 추가 검증이 필요하다.

## 5. 강점, 한계

이 논문의 첫 번째 강점은 PPG-only 방식으로 비교적 좋은 혈압 추정 성능을 달성했다는 점이다. ECG를 함께 사용하는 PTT 또는 PAT 기반 방식은 두 개 이상의 sensor와 signal synchronization이 필요하다. 반면 PPG-only 방식은 wearable device 구현이 더 단순할 수 있다.

두 번째 강점은 CNN–LSTM 구조의 설계가 PPG 신호 특성과 잘 맞는다는 점이다. CNN은 PPG waveform의 local morphology를 자동으로 추출하고, LSTM은 두 pulse cycle 내의 temporal dependency를 학습한다. 실험 결과에서도 LSTM 단독 및 LSTM–autoencoder보다 CNN–LSTM이 월등히 좋은 성능을 보였으므로, 두 구성 요소의 결합이 효과적이었다고 볼 수 있다.

세 번째 강점은 hyperparameter tuning을 grid search로 체계화했다는 점이다. Deep learning 성능은 optimizer, learning rate, batch size에 민감한데, 본 연구는 이를 수동으로 정하지 않고 후보 조합을 탐색했다. 이는 모델 개발 과정의 임의성을 줄이고, 재현성을 높이는 방향이다.

네 번째 강점은 IEEE, BHS, AAMI라는 여러 기준으로 성능을 평가했다는 점이다. 단순히 MAE만 보고하지 않고, cumulative percentage와 ME, SD를 함께 제시하여 혈압 측정기 평가 관점에서 성능을 해석했다.

그러나 한계도 중요하다. 첫째, 데이터셋의 subject 수가 55명으로 작다. 논문은 IEEE 기준의 최소 subject 수는 만족하지만, AAMI와 BHS guideline은 85명 이상의 participant를 요구한다고 스스로 언급한다. 따라서 본 연구의 결과만으로 clinical feasibility를 충분히 입증했다고 보기 어렵다.

둘째, 데이터 분할이 subject-independent인지 명확하지 않다. 논문은 75,226 segment를 training, validation, test로 나누고 five-fold cross-validation을 수행했다고 설명하지만, 같은 환자의 segment가 train과 test에 동시에 포함되었는지는 명확하지 않다. PPG 기반 혈압 추정에서는 patient-specific waveform pattern을 모델이 학습할 수 있으므로, subject-level split이 아니면 성능이 과대평가될 가능성이 있다.

셋째, 논문 내 수치와 표기가 일부 일관되지 않는다. Abstract의 성능 수치, Table 2의 MAE/SD, Table 3의 MAD/SD 사이에 차이가 있으며, 특히 LSTM–autoencoder의 SBP MAD 값은 Table 2의 MAE와 맞지 않는다. 또한 training, validation, test segment 수 합계도 전체 segment 수와 맞지 않는 부분이 있다. Optimizer 후보에는 Nadam이 포함되지 않았는데 결과에서는 Nadam이 선택된 것으로 표시된다. 이러한 불일치는 논문의 재현성과 해석을 어렵게 만든다.

넷째, overlapping segmentation은 sample 수를 늘리는 데 도움이 되지만, adjacent segment 간 유사도가 높아 data leakage 위험을 증가시킬 수 있다. 논문은 두 cycle 중 한 cycle을 overlap한다고 설명한다. 만약 segment-level random split을 사용했다면 매우 유사한 PPG segment가 train과 test에 나뉘어 들어갈 수 있다. 이는 test performance를 실제보다 좋게 만들 수 있다.

다섯째, MIMIC III는 ICU 환자 기반 데이터이다. ICU 환자의 PPG와 ABP는 병원 환경에서 측정된 신호이며, wearable device의 일상 환경 PPG와는 noise, motion artifact, sensor placement, 피부 상태, 혈류 상태가 다를 수 있다. 따라서 본 연구의 성능이 실제 wearable cNIBP 환경에서도 유지된다고 단정할 수 없다.

여섯째, grid search는 수동 tuning 부담을 줄이지만 계산량이 많이 필요하다. 논문은 grid search가 computational time을 줄인다고 표현하지만, 엄밀히 말하면 grid search는 후보 조합을 모두 탐색하므로 계산 비용이 증가할 수 있다. 다만 manual trial-and-error에 비해 체계적이고 병렬화가 가능하다는 장점은 있다.

비판적으로 보면, 이 논문은 CNN–LSTM이 PPG waveform 기반 혈압 추정에 효과적임을 보여주는 실험적 가치가 있다. 그러나 작은 subject 수, segmentation overlap, 데이터 분할 방식의 불명확성, 수치 표기 불일치 때문에 reported performance를 그대로 임상적 성능으로 해석하기에는 주의가 필요하다. 향후 연구에서는 subject-independent split과 external validation을 반드시 수행해야 한다.

## 6. 결론

이 논문은 MIMIC III PPG 신호를 이용하여 SBP와 DBP를 추정하는 CNN–LSTM 기반 continuous non-invasive blood pressure estimation 방법을 제안했다. 제안 방법은 PPG 신호를 DWT로 전처리하고, 두 개의 systolic peak-to-peak cycle로 segmenting한 뒤, spline interpolation과 z-score standardization을 적용한다. 이후 CNN–LSTM 모델이 PPG waveform의 morphological feature와 temporal dependency를 함께 학습한다. Hyperparameter는 grid search와 five-fold cross-validation을 통해 optimizer, learning rate, batch size를 선택한다.

실험 결과, CNN–LSTM은 LSTM 및 LSTM–autoencoder보다 훨씬 우수한 성능을 보였다. Table 2 기준으로 CNN–LSTM은 SBP MAE 3.64 mmHg, SD 7.04 mmHg, DBP MAE 2.39 mmHg, SD 3.79 mmHg를 달성했다. Table 3 기준으로는 SBP가 IEEE Grade B, BHS Grade B, AAMI pass에 해당하고, DBP는 IEEE Grade A, BHS Grade A, AAMI pass에 해당한다. 따라서 DBP 성능은 매우 우수하게 나타났고, SBP도 AAMI 기준은 만족하지만 BHS 및 IEEE의 최고 등급에는 완전히 도달하지 못했다.

이 연구의 주요 기여는 PPG-only input으로 CNN–LSTM을 구성하고, grid search를 통해 hyperparameter tuning을 자동화하여 기존 LSTM 및 LSTM–autoencoder보다 높은 성능을 얻었다는 점이다. 또한 다양한 기존 연구 및 평가 기준과 비교하여 CNN–LSTM의 가능성을 제시했다.

향후 연구에서는 더 많은 subject를 포함한 데이터셋, subject-independent split, overlapping segment로 인한 leakage 방지, 외부 데이터셋 검증, 실제 wearable 환경에서의 motion artifact 평가, 장기 연속 측정 검증이 필요하다. 이러한 보완이 이루어진다면, CNN–LSTM과 grid-search 기반 hyperparameter optimization은 PPG 기반 cuffless BP monitoring을 위한 실용적인 방향이 될 수 있다.
