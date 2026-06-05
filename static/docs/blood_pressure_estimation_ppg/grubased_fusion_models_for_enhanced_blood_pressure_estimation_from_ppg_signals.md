# GRU-Based Fusion Models for Enhanced Blood Pressure Estimation From PPG Signals

* **저자**: Syamsul Rizal, Yuniarti Ana Rahma
* **발표연도**: 2024

## 1. 논문 개요

이 논문은 Photoplethysmogram, 즉 PPG 신호와 개인 생리·인구통계 정보를 함께 사용하여 수축기 혈압인 Systolic Blood Pressure, SBP와 이완기 혈압인 Diastolic Blood Pressure, DBP를 비침습적으로 추정하는 딥러닝 기반 방법을 제안한다. 제안 방법의 핵심은 두 개의 Gated Recurrent Unit, GRU 기반 처리 경로를 사용하는 dual-input fusion model이다. 하나의 GRU 경로는 시간에 따라 변화하는 PPG waveform의 동적 패턴을 학습하고, 다른 GRU 경로는 성별, 나이, 키, 몸무게, 심박수, BMI와 같은 개인 정보를 처리한다. 두 경로에서 추출된 특징은 concatenation 이후 dense layer들을 통과하여 최종적으로 SBP와 DBP를 동시에 예측한다.

논문의 연구 문제는 cuff를 사용하는 전통적 혈압 측정 방식이나 침습적 혈압 측정 방식의 한계를 줄이면서, wearable health technology와 telemedicine 환경에서 지속적이고 비침습적인 혈압 모니터링을 가능하게 하는 것이다. 혈압은 심혈관 질환의 위험 평가와 관리에 매우 중요한 생체 신호이지만, 기존 방식은 지속 측정에 불편함이 크거나, 침습적 방식의 경우 감염, 출혈, 혈관 손상 등의 위험을 동반할 수 있다. 이에 따라 PPG 기반 혈압 추정은 손가락, 손목, 귀 등에서 optical sensor를 통해 비교적 쉽게 얻을 수 있는 신호를 활용한다는 점에서 실용적 가치가 크다.

이 논문은 기존 PPG 기반 혈압 추정 연구들이 handcrafted feature extraction, 복잡한 hyperparameter tuning, ECG 신호의 artifact 문제, 일반화 성능 부족 등의 문제를 갖고 있다고 보고, PPG 신호와 개인 생리 정보를 결합하는 dual-GRU 구조를 통해 혈압 추정 성능을 높이고자 한다. 논문에서 보고한 주요 성능은 PPGBP 데이터셋 기준으로 SBP에 대해 MAE 1.459 mmHg, DBP에 대해 MAE 1.165 mmHg이며, IEEE, BHS, AAMI 기준에서 모두 우수한 등급을 달성했다고 설명한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 혈압이 단순히 PPG waveform의 모양만으로 결정되는 것이 아니라 개인의 생리적 특성에도 영향을 받는다는 점을 모델 구조에 반영하는 것이다. PPG 신호는 cardiac cycle에 따른 혈액량 변화를 반영하므로 혈압과 관련된 시간적 패턴을 포함한다. 그러나 같은 PPG 패턴이라도 나이, 성별, 체중, 키, BMI, 심박수와 같은 개인 특성에 따라 혈압과의 관계가 달라질 수 있다. 따라서 논문은 PPG 신호만을 사용하는 단일 입력 모델보다, PPG와 demographic 또는 physiological data를 함께 사용하는 모델이 개인별 혈압 변동성을 더 잘 포착할 수 있다고 본다.

기존 연구와 비교했을 때 이 논문의 차별점은 두 가지로 정리할 수 있다. 첫째, PPG time-series를 처리하는 GRU branch와 개인 정보를 처리하는 별도의 GRU branch를 구성한 뒤, 두 정보를 feature-level에서 결합한다. 둘째, 모델의 예측 성능을 단순한 MAE만으로 평가하지 않고 IEEE, BHS, AAMI라는 혈압 측정 장치 평가 기준에 맞춰 다면적으로 검증한다. 특히 BHS 기준에서는 오차가 5, 10, 15 mmHg 이내에 들어오는 예측 비율을 평가하고, AAMI 기준에서는 Mean Error와 Standard Deviation을 통해 bias와 변동성을 함께 확인한다.

논문은 GRU가 long sequence를 처리하면서도 vanishing gradient 문제를 완화할 수 있기 때문에 PPG와 같은 시계열 생체 신호에 적합하다고 설명한다. LSTM보다 상대적으로 구조가 단순한 GRU를 사용함으로써 시간적 의존성을 학습하면서도 계산 효율성을 확보하려는 의도가 있다. 다만 본문에서 실제 computational cost나 LSTM 대비 학습 시간 감소량을 정량적으로 제시하지는 않으므로, 계산 효율성에 대한 주장은 구조적 기대에 가깝고 실험적으로 충분히 입증되었다고 보기는 어렵다.

## 3. 상세 방법 설명

제안 방법은 데이터 수집, 전처리, dual-input GRU 모델 구성, 학습, 평가의 흐름으로 구성된다. 입력 데이터는 크게 두 종류이다. 첫 번째는 PPG waveform이며, 두 번째는 성별, 나이, 키, 몸무게, 심박수, BMI로 구성된 개인 정보이다. 모델은 이 두 입력을 별도의 GRU 경로에서 처리한 뒤 결합하여 SBP와 DBP를 예측한다.

데이터는 주로 Liang 등이 구축한 PPGBP Database를 사용한다. 이 데이터셋은 중국 Guilin People’s Hospital에서 수집된 것으로, 219명의 대상자로부터 얻은 657개의 data segment를 포함한다. 대상자에는 고혈압과 당뇨 등 다양한 조건을 가진 사람들이 포함되어 있으며, PPG waveform과 arterial blood pressure가 함께 수집되었다. 논문은 이 데이터셋이 비침습적 심혈관 질환 탐지와 혈압 추정 연구에 유용하다고 설명한다. 추가적으로, 비교 분석을 위해 MIMIC III matched subset과 clinical database에서 25명의 환자에 대한 PPG 신호와 환자 정보를 수집했다고 제시한다.

전처리 단계에서는 raw PPG waveform에 bandpass filtering을 적용한다. 이 필터는 high-pass Butterworth filter와 low-pass Butterworth filter를 결합한 형태이다. High-pass filter의 cutoff frequency는 0.5 Hz로 설정되어 baseline wander와 같은 저주파 잡음을 제거한다. Low-pass filter의 cutoff frequency는 15.0 Hz로 설정되어 electronic interference나 motion artifact와 같은 고주파 잡음을 줄인다. PPG sampling frequency는 1000 Hz로 제시되며, 이러한 필터 설정은 혈류 변화와 관련된 핵심 PPG 성분을 보존하면서 잡음을 줄이기 위한 것이다.

개인 정보는 서로 다른 단위를 가지므로 normalization을 수행한다. 논문에서 사용한 min-max normalization은 다음과 같다.

$$
X_{\text{Normalized}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
$$

이 식은 각 변수 $X$에서 해당 변수의 최소값 $X_{\min}$을 뺀 뒤, 전체 범위 $X_{\max} - X_{\min}$으로 나누어 값을 일정한 범위로 변환한다. 예를 들어 나이, 키, 몸무게, BMI는 값의 범위가 서로 다르므로 그대로 모델에 넣으면 특정 변수의 scale이 학습에 과도한 영향을 줄 수 있다. Normalization은 이러한 scale 차이를 줄여 모델이 각 변수를 더 안정적으로 학습하도록 돕는다.

전처리 후 데이터는 training, validation, testing set으로 나뉜다. 논문은 전체 데이터의 60%를 training set, 25%를 validation set, 15%를 testing set으로 사용한다고 설명한다. Training set은 모델 가중치 학습에 사용되고, validation set은 학습 중 성능 확인 및 overfitting 방지에 사용되며, testing set은 학습에 사용되지 않은 데이터에서 최종 성능을 평가하는 데 사용된다.

모델 아키텍처는 dual-input 구조이다. PPG 입력 경로는 크기 $(2100, 1)$의 1D input vector를 받는다. 여기서 2100은 time-step 길이를, 1은 각 time-step의 feature dimension을 의미하는 것으로 해석된다. 이 입력은 64 units를 가진 GRU layer 세 개를 연속적으로 통과한다. 각 GRU layer는 ReLU activation function을 사용한다고 논문은 설명한다. GRU는 순환 신경망의 한 종류로, 이전 time-step의 정보를 현재 상태에 반영하면서 sequence 전체의 시간적 패턴을 학습한다. PPG 신호에서는 systolic peak, diastolic peak, pulse shape, waveform slope와 같은 시간적 구조가 혈압과 관련될 수 있으므로 GRU를 사용하는 것이 자연스럽다.

개인 정보 입력 경로는 크기 $(6, 1)$의 input vector를 사용한다. 이 6개 입력은 sex, age, height, weight, heart rate, BMI이다. 논문은 이 입력이 32 units를 가진 5개의 GRU layer를 통과한다고 설명한다. 다만 개인 정보는 본질적으로 시간에 따른 sequence라기보다 static feature에 가깝기 때문에, 이를 GRU로 처리하는 설계가 왜 일반적인 dense layer보다 더 적합한지에 대해서는 논문에서 충분한 이론적 설명이나 ablation study가 제시되지 않는다. 이 점은 방법론상의 흥미로운 선택이지만 동시에 비판적으로 검토할 필요가 있는 부분이다.

두 GRU stream의 출력은 concatenation layer에서 결합된다. 결합된 feature representation은 dense layer들을 통과한다. Dense layer의 unit 수는 64, 32, 16, 8로 점진적으로 줄어든다. 이 구조는 두 입력 경로에서 추출된 고차원 특징을 단계적으로 압축하면서 혈압 예측에 필요한 결합 표현을 학습하는 역할을 한다. 마지막 output layer는 두 개의 unit을 가지며, 각각 SBP와 DBP를 나타낸다. 즉 모델은 하나의 입력 쌍에 대해 두 개의 연속값을 출력하는 multi-output regression 모델이다.

학습에서는 Adamax optimizer가 사용된다. Adamax는 Adam 계열의 optimizer로, sparse gradient가 있는 상황에서 안정적으로 동작할 수 있다고 알려져 있다. Learning rate는 0.005로 설정되었다. Loss function은 Mean Squared Error, MSE이다. MSE는 예측값과 실제값의 차이를 제곱한 뒤 평균을 내는 손실 함수이며, 큰 오차에 더 큰 penalty를 부여한다. 혈압 예측에서는 큰 예측 오차가 임상적으로 중요할 수 있기 때문에 MSE를 사용하는 것이 타당하다.

MSE는 일반적으로 다음과 같이 표현할 수 있다.

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

여기서 $y_i$는 실제 혈압값, $\hat{y}_i$는 모델이 예측한 혈압값, $n$은 sample 수이다. 논문은 주요 평가 지표로 MAE, MRE, RMSE, ME, SD, MAPD, CP 등을 사용하지만, 학습 손실은 MSE를 사용한다.

Batch size는 128이고, epoch 수는 500으로 설정되었다. 또한 validation loss를 모니터링하는 callback function을 사용하여 validation 성능이 가장 좋은 weight를 저장한다고 설명한다. 논문은 이 전략이 overfitting을 완화하는 데 도움이 된다고 설명하지만, early stopping이 실제로 몇 epoch에서 작동했는지, 또는 학습 곡선이 어떻게 변화했는지는 추출 텍스트만으로는 확인할 수 없다.

평가 지표는 여러 표준에 맞춰 정의된다. IEEE 기준에서 사용되는 Mean Absolute Difference 또는 MAE는 다음과 같이 표현된다.

$$
\text{MAD} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

논문에서는 MAD와 MAE를 사실상 같은 절대 오차 평균의 의미로 사용한다. MAPD는 실제값 대비 절대 오차의 비율을 백분율로 나타내며 다음과 같다.

$$
\text{MAPD} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

BHS 기준에서 사용하는 Cumulative Percentage, CP는 특정 오차 범위 안에 들어오는 예측의 비율을 나타낸다.

$$
\text{CP} = \frac{\text{Cumulative Frequency}}{n} \times 100
$$

AAMI 기준에서는 Mean Error, ME와 Standard Deviation, SD를 사용한다. ME는 예측 오차의 평균 bias를 나타낸다.

$$
\text{ME} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)
$$

SD는 오차가 ME 주변에서 얼마나 퍼져 있는지를 나타낸다.

$$
\text{SD} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i - \text{ME})^2}
$$

추가 지표인 Mean Relative Error, MRE는 다음과 같이 정의된다.

$$
\text{MRE} = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100
$$

Root Mean Squared Error, RMSE는 다음과 같다.

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

RMSE는 오차를 제곱한 뒤 평균을 내므로, 큰 오차에 더 민감하다. MAE가 평균적인 절대 오차를 직관적으로 보여준다면, RMSE는 큰 예측 실패가 존재하는지를 더 민감하게 반영한다.

## 4. 실험 및 결과

실험은 PPGBP 데이터셋을 중심으로 수행되었고, 추가적으로 MIMIC III 데이터셋을 이용한 비교 또는 일반화 평가가 포함된다. PPGBP 데이터셋은 219명으로부터 얻은 657개 segment로 구성되며, PPG waveform과 arterial blood pressure가 함께 포함된다. MIMIC III 데이터는 25명의 환자에 대한 PPG 신호와 환자 정보를 사용했다고 설명되어 있다. 다만 MIMIC III 실험에서 정확히 어떤 subset selection 기준, preprocessing, train-test split이 사용되었는지에 대한 세부 정보는 추출 텍스트만으로 충분히 확인되지 않는다.

비교 대상은 CNN-LSTM, SVM, ResNet, ANN, RNN, U-Net, AdaBoost 등 다양한 기존 machine learning 및 deep learning 기반 혈압 추정 모델이다. 논문은 Table 3에서 기존 연구들과 SBP 및 DBP의 MAE와 SD를 비교한다. 제안 모델은 dual GRU network와 ANN을 결합한 구조로 표시되며, 입력으로 PPG와 demographic data를 모두 사용한다.

PPGBP 데이터셋에서 제안 모델은 SBP에 대해 MAE 1.45 또는 1.459 mmHg, DBP에 대해 MAE 1.16 또는 1.165 mmHg를 달성했다고 보고된다. SD는 SBP 2.658 mmHg, DBP 1.628 mmHg로 제시된다. Table 4에서는 SBP MAE 1.459 mmHg, DBP MAE 1.165 mmHg, SBP MRE 1.114%, DBP MRE 1.687%, SBP RMSE 2.787 mmHg, DBP RMSE 1.628 mmHg가 보고된다. 이 수치는 평균적으로 예측값과 실제 혈압값 사이의 차이가 매우 작다는 것을 의미한다.

IEEE 기준에서 모델은 SBP와 DBP 모두 Grade A를 받는다. 논문에 따르면 IEEE 기준의 Grade A는 MAE가 5 mmHg 미만인 경우이며, 제안 모델은 SBP와 DBP 모두 이 기준을 충분히 만족한다. MAPD도 SBP 1.114%, DBP 1.687%로 낮다.

BHS 기준에서는 특정 오차 범위 안에 들어오는 예측 비율이 중요하다. 논문은 제안 모델이 SBP의 경우 5 mmHg 이내 예측 비율이 96% 이상, DBP의 경우 98% 이상이며, 15 mmHg 이내에서는 SBP와 DBP 모두 100%에 도달한다고 설명한다. 이 결과에 따라 BHS 기준에서도 SBP와 DBP 모두 Grade A를 받았다고 보고한다.

AAMI 기준에서는 ME가 5 mmHg 미만이고 SD가 8 mmHg 미만이어야 한다. 논문에서 PPGBP 데이터셋 기준 ME는 SBP 0.84 mmHg, DBP 0.041 mmHg로 매우 낮다. SD는 SBP 2.658 mmHg, DBP 1.628 mmHg로 AAMI 기준을 만족한다. 이는 모델이 평균적으로 특정 방향으로 크게 과대추정하거나 과소추정하지 않으며, 오차의 분산도 작다는 의미이다.

Figure 3은 실제 혈압값과 예측 혈압값의 scatter plot을 보여준다. SBP와 DBP 모두에서 데이터 포인트들이 대체로 45도에 가까운 trend line 주변에 분포한다고 설명되어 있다. 이는 예측값이 실제값과 강한 양의 상관관계를 보인다는 의미이다. 다만 추출 텍스트에서는 correlation coefficient나 R-squared 값이 제시되지 않으므로, 시각적 정렬만으로 정확한 상관 정도를 정량화할 수는 없다.

Figure 4는 Bland-Altman plot과 prediction error histogram을 사용해 모델의 bias와 agreement를 분석한다. SBP의 Bland-Altman 분석에서는 평균 차이가 0.84 mmHg, upper limit of agreement가 6.05 mmHg, lower limit of agreement가 -4.37 mmHg로 제시된다. DBP의 평균 차이는 0.04 mmHg이며, upper limit는 3.23 mmHg, lower limit는 -3.15 mmHg이다. DBP의 agreement range가 SBP보다 더 좁기 때문에, 논문은 DBP 예측이 더 정밀하다고 해석한다.

MIMIC III 데이터셋 결과는 텍스트 내에서 다소 혼재되어 제시된다. 한 부분에서는 GRU and Dense model이 MIMIC III 데이터셋에서 SBP MAE 5.74, DBP MAE 6.72, SBP SD 6.97, DBP SD 8.93을 보였다고 설명한다. 이는 PPGBP 데이터셋보다 성능이 낮으며, MIMIC III의 환자군 다양성이나 PPG signal pattern 차이 때문일 수 있다고 해석한다. 그러나 이어지는 부분에서는 MIMIC III 데이터셋에서 SBP MAE 1.42, DBP MAE 1.97, SBP SD 2.38, DBP SD 2.97을 달성했다고도 설명한다. Table 5에서는 MIMIC III 기준 SBP MAE 1.429, DBP MAE 2.385, SBP MAPD 1.523%, DBP MAPD 4.208%, SBP SD 2.38, DBP SD 2.97로 제시된다. 이처럼 MIMIC III 관련 수치가 문맥상 서로 완전히 일관되지는 않으므로, 논문 전체 표와 원본 문맥을 확인하지 않고는 어떤 값이 최종 대표 성능인지 단정하기 어렵다.

그럼에도 논문이 강조하는 전체 결과는 분명하다. 제안 모델은 PPGBP 데이터셋에서 매우 낮은 오차를 보였고, IEEE, BHS, AAMI 기준을 모두 만족했으며, MIMIC III 데이터에서도 일정 수준 이상의 일반화 가능성을 보였다고 주장한다. 특히 PPG와 개인 정보를 함께 사용하는 fusion 방식이 혈압 추정 정확도 향상에 기여했다고 해석한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 PPG time-series와 개인 생리 정보를 함께 사용하는 구조를 명확히 제안했다는 점이다. 혈압은 개인의 신체적 특성과 밀접하게 관련되므로, PPG waveform만으로 모든 개인의 혈압을 동일한 방식으로 추정하는 것은 한계가 있을 수 있다. 이 논문은 age, sex, height, weight, heart rate, BMI를 함께 사용하여 individual physiological variation을 모델에 반영하려고 한다. 이는 personalized blood pressure estimation 관점에서 타당한 설계 방향이다.

두 번째 강점은 평가 기준이 비교적 포괄적이라는 점이다. 단순히 MAE만 보고하는 것이 아니라 IEEE, BHS, AAMI 기준에 맞춰 MAPD, CP, ME, SD 등을 함께 제시한다. 혈압 추정 모델은 단순한 평균 오차뿐 아니라 clinical acceptability가 중요하므로, 여러 표준 기준에서 모델을 평가한 점은 실용적 의의가 있다.

세 번째 강점은 Bland-Altman plot과 histogram을 통해 예측값과 실제값의 agreement, bias, error distribution을 분석했다는 점이다. 의료 측정 모델에서는 예측값이 평균적으로 정확한 것뿐 아니라 특정 혈압 범위에서 체계적으로 과대 또는 과소 추정하지 않는지가 중요하다. Bland-Altman 분석은 이러한 점을 확인하는 데 적합한 도구이다.

그러나 한계도 명확하다. 첫째, 데이터셋 규모가 크지 않다. PPGBP 데이터셋은 219명, 657개 segment로 구성되어 있으며, MIMIC III 비교 분석은 25명의 환자를 사용했다고 제시된다. 딥러닝 모델, 특히 여러 GRU layer를 포함한 구조를 안정적으로 학습하고 일반화 성능을 검증하기에는 상대적으로 제한적인 규모일 수 있다. 논문도 larger and more diverse datasets에서 추가 검증이 필요하다고 인정한다.

둘째, 개인 정보를 처리하는 경로에 GRU를 사용하는 설계의 타당성이 충분히 설명되지 않는다. PPG는 명백한 time-series이므로 GRU 사용이 자연스럽지만, sex, age, height, weight, heart rate, BMI는 일반적으로 static tabular feature이다. 이를 $(6, 1)$ 형태의 sequence로 보고 5개의 GRU layer에 통과시키는 것이 dense layer나 multilayer perceptron보다 우수한지에 대한 ablation study가 필요하다. 추출 텍스트에는 PPG-only 모델, demographic-only 모델, dense-only fusion 모델, single-GRU 모델 등과의 체계적 비교가 명확히 제시되지 않는다.

셋째, MIMIC III 관련 결과 제시에 일관성 문제가 있다. 한 부분에서는 MIMIC III에서 MAE가 SBP 5.74, DBP 6.72로 성능 저하가 나타났다고 설명하고, 다른 부분에서는 SBP 1.42, DBP 1.97 또는 SBP 1.429, DBP 2.385로 매우 우수한 성능을 제시한다. 이 값들이 서로 다른 실험 설정, 다른 모델 변형, 다른 subset 또는 다른 evaluation protocol을 의미할 가능성이 있지만, 추출 텍스트만으로는 그 차이가 명확하지 않다. 따라서 MIMIC III 일반화 성능에 대해서는 조심스럽게 해석해야 한다.

넷째, 논문은 automated hyperparameter optimization을 언급하지만, 추출 텍스트에서는 어떤 최적화 방법을 사용했는지 명확하지 않다. Grid search, random search, Bayesian optimization, Keras Tuner 등의 구체적 방법이 제시되지 않으며, Table 1에 최종 hyperparameter만 제시되는 것으로 보인다. 따라서 hyperparameter tuning 문제를 해결했다는 주장은 일부만 뒷받침된다.

다섯째, clinical deployment 관점의 검증은 아직 부족하다. 논문은 controlled experimental conditions에서 수집된 고품질 PPG 신호를 사용한다. 그러나 wearable device 환경에서는 motion artifact, sensor placement variation, skin tone, peripheral perfusion, ambient light, device-specific noise 등 다양한 변수가 존재한다. 논문은 real-world situation에서 추가 검증이 필요하다고 언급하지만, 실제 생활 환경에서의 robust performance는 아직 입증되지 않았다.

비판적으로 보면, 논문은 매우 낮은 MAE를 보고하고 있어 성능이 인상적이지만, 데이터 분할 방식에서 subject-independent split이 보장되었는지 확인이 중요하다. 만약 같은 사람의 segment가 training과 testing에 동시에 포함된다면, 모델은 개인 특성을 기억하여 실제 새로운 사용자에 대한 일반화보다 높은 성능을 보일 수 있다. 추출 텍스트에는 subject-wise split 여부가 명확하게 제시되지 않으므로, 이 부분은 반드시 원문 또는 코드 확인이 필요한 중요한 검증 포인트이다.

## 6. 결론

이 논문은 PPG 신호와 개인 생리 정보를 결합한 dual-input GRU 기반 혈압 추정 모델을 제안한다. 제안 모델은 PPG waveform의 시간적 패턴과 sex, age, height, weight, heart rate, BMI 같은 개인 정보를 각각 별도의 GRU branch에서 처리한 뒤 dense layer를 통해 SBP와 DBP를 동시에 예측한다. PPGBP 데이터셋에서 SBP MAE 1.459 mmHg, DBP MAE 1.165 mmHg 수준의 낮은 오차를 보였고, IEEE, BHS, AAMI 기준에서 모두 우수한 등급을 달성했다고 보고한다.

이 연구의 주요 기여는 비침습적 혈압 추정을 위해 time-series signal과 static physiological information을 결합하는 fusion architecture를 제시했다는 점이다. 또한 단순 오차 지표뿐 아니라 의료기기 평가에 사용되는 표준 기준을 적용하여 모델의 clinical acceptability를 검토했다는 점도 의미가 있다. 이러한 접근은 향후 wearable health technology, remote patient monitoring, hypertension management, telemedicine 분야에서 연속적이고 비침습적인 혈압 모니터링 시스템 개발에 기여할 수 있다.

다만 실제 적용 가능성을 확정하기 위해서는 더 큰 규모의 subject-independent validation, 다양한 인구집단과 다양한 센서 환경에서의 검증, real-world motion artifact에 대한 robustness 평가, PPG-only 모델과 fusion 모델 간의 ablation study, static demographic feature에 GRU를 사용하는 설계의 타당성 검증이 필요하다. 특히 보고된 매우 낮은 오차가 실제 새로운 사용자에게도 유지되는지 확인하는 것이 중요하다.

종합하면, 이 논문은 PPG 기반 cuffless blood pressure estimation 분야에서 유망한 deep learning fusion approach를 제시한 연구이다. 제안 방법은 높은 성능을 보고하지만, clinical deployment를 위해서는 데이터 분할의 엄격성, 외부 데이터셋 검증, 실제 착용 환경에서의 안정성 평가가 추가로 요구된다.
