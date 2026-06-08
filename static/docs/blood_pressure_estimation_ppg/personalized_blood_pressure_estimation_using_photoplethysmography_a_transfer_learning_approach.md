# Personalized Blood Pressure Estimation Using Photoplethysmography: A Transfer Learning Approach

* **저자**: Jared Leitner, Po-Han Chiang, Sujit Dey
* **발표연도**: 2022

## 1. 논문 개요

이 논문은 PPG 신호만을 이용하여 개인화된 blood pressure, 즉 BP를 추정하는 deep learning 기반 방법을 제안한다. 구체적으로, 저자들은 raw PPG time series를 직접 입력으로 받아 systolic blood pressure, 즉 SBP와 diastolic blood pressure, 즉 DBP를 5초마다 추정하는 hybrid neural network architecture를 설계한다. 이 모델은 convolutional layers, recurrent layer, fully connected layers를 결합한 구조이며, 논문에서는 이를 BP-CRNN이라고 부른다.

논문의 핵심 문제는 PPG와 BP 사이의 관계가 개인마다 다르다는 점이다. PPG signal은 심장 활동, 혈관 이완, 미세순환 상태 등 다양한 생리적 요인이 섞인 신호이므로 정보량은 풍부하지만, 같은 혈압 변화라도 사람마다 PPG waveform의 반응이 다를 수 있다. 따라서 모든 사람에게 하나의 공통 모델을 적용하는 aggregate model은 정확도가 낮아질 수 있다. 반면 개인별 모델을 학습하려면 개인별 PPG-BP paired data가 많이 필요하다. PPG는 wearable sensor로 쉽게 모을 수 있지만, BP ground truth는 cuff나 arterial catheter가 필요하므로 일상 환경에서 대량으로 수집하기 어렵다.

이 논문은 이 문제를 transfer learning으로 해결하려 한다. 먼저 여러 source patients의 풍부한 PPG-BP 데이터를 사용하여 BP-CRNN을 pre-train한다. 이후 target patient, 즉 새 환자에 대해서는 소량의 개인 데이터만 사용하여 일부 layer만 fine-tuning한다. 이 접근은 source patients로부터 일반적인 PPG feature representation을 배우고, target patient의 소량 데이터로 개인 특화 high-level feature와 BP mapping을 보정하는 방식이다.

연구의 중요성은 wearable blood pressure monitoring의 현실적 제약과 관련된다. 기존 cuff 기반 측정은 불편하고 간헐적이며, arterial catheter는 gold standard이지만 침습적이므로 일상 사용이 불가능하다. PPG sensor는 대부분의 wrist wearable이나 fingertip sensor에 쉽게 탑재될 수 있다. 따라서 PPG만으로 정확한 개인화 혈압 추정이 가능하다면, 일상에서 continuous and automated BP monitoring을 구현할 수 있고, 고혈압의 조기 발견과 혈압 변동 추적에 큰 도움이 될 수 있다.

논문에서 제안한 BP-CRNN-Transfer는 MIMIC III 데이터셋에서 SBP MAE 3.52 mmHg, DBP MAE 2.20 mmHg를 달성했다. 또한 BHS와 AAMI 혈압 측정 기준을 모두 만족한다고 보고된다. 특히 저자들은 개인당 50개의 PPG-BP sample만으로도 transfer learning을 통해 개인화 모델을 학습할 수 있음을 보였다고 주장한다.

## 2. 핵심 아이디어

이 논문의 중심적인 직관은 PPG-BP 관계에는 두 종류의 정보가 공존한다는 것이다. 하나는 여러 사람에게 공통적으로 나타나는 physiological signal pattern이고, 다른 하나는 개인별 혈관 특성, 심혈관 상태, 피부 및 센서 조건 등으로 인해 달라지는 person-specific relationship이다. Deep learning model은 source patients의 대규모 데이터를 통해 공통적인 PPG feature extractor를 학습할 수 있고, transfer learning을 통해 target patient의 소량 데이터에 맞게 일부 layer만 조정할 수 있다.

기존 PPG 기반 혈압 추정 방법은 크게 hand-crafted feature 기반 방법과 deep learning 기반 방법으로 나눌 수 있다. Hand-crafted feature 기반 pulse wave analysis, 즉 PWA는 PPG cycle에서 morphology feature를 추출한다. 예를 들어 peak, valley, pulse width, time-domain 또는 frequency-domain feature를 추출한 뒤 regression model에 입력한다. 그러나 PPG는 noise에 민감하고, 사람마다 waveform morphology가 다르며, key point detection이 어렵다. 또한 사람이 설계한 feature는 redundant하거나 irrelevant할 수 있고, raw signal에 포함된 정보를 충분히 활용하지 못할 수 있다.

이 논문은 이러한 한계를 줄이기 위해 raw PPG time series를 직접 입력으로 사용한다. 즉, feature engineering을 사람이 수행하지 않고, convolutional layers가 PPG waveform에서 low-level 및 high-level feature를 자동으로 추출하도록 한다. 이후 GRU가 추출된 feature들의 temporal dependency를 모델링하고, fully connected layers가 이를 SBP와 DBP 추정값으로 변환한다. 이는 raw signal에서 feature learning과 regression을 end-to-end로 수행한다는 점에서 기존 hand-crafted PWA 방식과 차별화된다.

또 다른 핵심 차별점은 transfer learning에서 모든 layer를 업데이트하지 않고, 일부 layer만 개인화한다는 점이다. 저자들은 pre-trained BP-CRNN에서 마지막 convolutional layer인 Conv3와 마지막 fully connected layer인 FC2만 target patient 데이터로 fine-tuning하는 전략이 가장 안정적이라고 보고한다. Conv3를 fine-tuning하면 개인별 high-level PPG feature representation을 학습할 수 있고, FC2를 fine-tuning하면 추출된 feature와 target patient의 BP 사이 mapping을 개인화할 수 있다. 반면 GRU layer는 fine-tuning하지 않는 것이 더 좋았는데, 논문은 이를 PPG feature 간 temporal relationship이 개인 간에 어느 정도 transferable하기 때문일 수 있다고 해석한다.

이 논문의 중요한 메시지는 개인화가 필수적이지만, 개인 데이터를 많이 요구하지 않아도 된다는 점이다. 기존 개인화 모델은 수 시간 또는 수천 개의 개인 BP sample이 필요할 수 있는데, 이 논문은 source patients의 데이터를 활용한 transfer learning으로 개인 데이터 요구량을 크게 줄였다. 특히 transfer learning을 사용하면 360개의 개인 sample로, transfer learning 없이 3600개 sample을 사용한 개인 모델과 비슷한 성능을 달성한다. 저자들은 이를 10배 적은 개인 데이터로 동등한 성능을 얻은 결과로 해석한다.

## 3. 상세 방법 설명

전체 방법은 데이터 수집 및 전처리, BP-CRNN architecture 설계, source patients 기반 pre-training, target patient 기반 partial fine-tuning, 성능 평가로 구성된다.

데이터는 MIMIC III Matched Subset database에서 가져왔다. 이 데이터베이스는 ICU 환자들의 ECG, respiration, continuous blood pressure, PPG waveform 등을 포함한다. 각 waveform은 125 Hz로 sampling되었다. 이 논문에서 reference BP는 radial artery에 삽입된 invasive arterial catheter로 측정된 ABP signal에서 얻었다. Arterial catheter는 침습적이지만 정확도가 높아 gold-standard BP measurement method로 간주된다.

저자들은 preprocessing 이후 최소 10시간 이상의 high-quality PPG와 BP data를 가진 환자만 고려했다. 최종적으로 100명의 환자를 무작위로 선택했고, 이 중 남성은 56명, 여성은 44명이다. 환자 나이는 21세에서 82세 사이이며 평균은 58세이다. 논문은 demographic diversity를 어느 정도 확보했다고 설명하지만, 인종, 질병 상태, 약물 복용, ICU 상태 등 세부적인 임상적 다양성은 제공된 텍스트에서 자세히 설명되지 않는다.

Preprocessing의 첫 단계는 raw PPG signal을 5초 segment로 나누는 것이다. 이후 PPG sampling rate를 125 Hz에서 25 Hz로 downsampling한다. 저자들은 25 Hz가 중요한 frequency components를 포함하기에 충분하다고 설명한다. 5초 segment가 25 Hz로 downsampling되므로, 각 입력 segment는 시간축 길이 $5 \times 25 = 125$ samples를 갖는다고 이해할 수 있다.

각 PPG segment의 label은 해당 5초 동안의 평균 SBP와 평균 DBP이다. SBP와 DBP는 raw ABP series에서 peak detection algorithm을 사용하여 추출된다. SBP는 pulse waveform의 최대 압력, DBP는 최소 압력에 해당한다. 따라서 모델은 각 5초 PPG segment를 입력으로 받아 해당 segment 동안의 mean SBP와 mean DBP를 동시에 출력한다.

PPG signal에는 motion artifact나 sensor misplacement로 인해 corrupted segment가 포함될 수 있다. 이를 제거하기 위해 저자들은 autocorrelation filter를 사용했다. 정상적인 PPG segment는 주기성을 가져야 하므로, cycle length의 배수만큼 offset했을 때 autocorrelation이 높게 나타날 것으로 기대된다. 저자들은 maximum autocorrelation에 대해 empirical threshold 0.7을 설정하고, 이 기준을 만족하지 못하는 segment를 제거했다. 이후 filtered PPG segment는 zero mean과 unit variance로 정규화된다.

제안 network architecture는 BP-CRNN이라고 불린다. 이 구조는 convolutional layers, GRU, fully connected layers로 구성된다. Convolutional layers는 raw PPG input에서 feature를 추출하고, GRU는 추출된 feature sequence의 temporal relationship을 모델링하며, fully connected layers는 GRU output을 SBP와 DBP가 잘 추정되는 representation으로 변환한다.

Convolutional part는 세 개의 1-dimensional convolutional layers로 구성된다. 각 convolutional layer는 50개의 filter를 사용한다. 저자들은 convolutional layer 수를 1개에서 5개까지 바꾸어 실험했고, 3개 convolutional layers가 가장 좋은 성능을 보였다고 설명한다. 각 convolutional layer 뒤에는 ReLU activation function이 적용된다. 논문에서 convolutional layer의 output feature map은 다음 식으로 정의된다.

$$
x_j^l = ReLU \left( \left( \sum_i x_i^{l-1} * k_{ij}^l \right) + b_j^l \right)
$$

여기서 $x_j^l$는 $l$번째 convolutional layer에서 생성된 $j$번째 feature map이고, $x_i^{l-1}$는 이전 layer의 $i$번째 feature map이다. $k_{ij}^l$는 학습되는 convolution kernel이고, $b_j^l$는 bias이며, $*$는 convolution operation을 의미한다. 쉬운 말로 설명하면, convolutional layer는 PPG waveform의 국소적인 시간 패턴을 여러 filter로 훑으면서 혈압 추정에 유용한 waveform feature를 자동으로 추출한다.

BP-CRNN의 특징적인 설계는 첫 번째 convolutional layer의 output과 세 번째 convolutional layer의 output을 concatenate하여 GRU에 전달한다는 점이다. 첫 번째 layer의 output은 상대적으로 low-level feature를 담고, 세 번째 layer의 output은 더 추상적인 high-level feature를 담는다. 각 layer가 50개의 filter를 가지므로, 두 layer의 output을 합치면 총 100개의 feature series가 GRU에 입력된다. 각 feature map은 padding을 통해 입력 PPG sequence와 같은 길이를 유지한다. 따라서 GRU 입력은 $100 \times t_n$ 형태이며, 여기서 $t_n$은 input PPG segment의 길이이다.

GRU는 LSTM과 유사하게 sequential data를 처리하는 recurrent unit이지만, gate 수가 적어 계산이 더 간단하다. 논문에서 GRU의 동작은 다음 식으로 설명된다.

$$
z_t = \sigma(W^{(z)} x_t + U^{(z)} h_{t-1})
$$

$$
r_t = \sigma(W^{(r)} x_t + U^{(r)} h_{t-1})
$$

$$
h'*t = tanh(W^{(h)} x_t + U^{(h)}(r_t \odot h*{t-1}))
$$

$$
h_t = z_t \odot h_{t-1} + (1 - z_t) \odot h'_t
$$

여기서 $z_t$는 update gate이고, 이전 hidden state를 얼마나 유지할지 결정한다. $r_t$는 reset gate이고, candidate activation을 계산할 때 이전 hidden state를 얼마나 반영할지 조절한다. $h'_t$는 candidate activation이며, $h_t$는 최종 GRU activation이다. $\sigma$는 sigmoid function이고, $tanh$는 hyperbolic tangent function이며, $\odot$는 element-wise multiplication이다.

직관적으로 설명하면, GRU는 PPG feature sequence를 시간 순서대로 읽으면서 과거 feature pattern을 얼마나 기억할지, 새로운 feature pattern을 얼마나 반영할지 결정한다. 이 논문에서는 GRU activation size를 25로 설정했으며, 실험적으로 좋은 성능을 보였다고 설명한다. 따라서 GRU output은 $25 \times t_n$ 형태가 된다.

GRU의 activation은 flatten되어 fully connected layers로 전달된다. 마지막 두 network layers는 fully connected layers이며, 최종 output은 2-dimensional vector이다. 이 두 값은 각각 estimated SBP와 estimated DBP에 해당한다. 각 fully connected layer 뒤에도 ReLU activation function이 사용된다. 또한 batch normalization이 사용되어 각 layer의 input distribution을 안정화하고, internal covariate shift를 줄이며, 학습을 더 빠르게 만든다고 설명된다.

Transfer learning 절차는 이 논문의 핵심이다. 먼저 여러 source patients의 PPG-BP data를 사용하여 BP-CRNN을 pre-train한다. 이후 target patient에 대해 pre-trained network를 initialization으로 사용하고, 일부 layer만 fine-tuning한다. 저자들은 실험적으로 Conv3와 FC2를 fine-tuning하는 것이 가장 robust한 transfer learning 성능을 보였다고 보고한다. 또한 target patient의 data distribution을 반영하기 위해 batch normalization parameters도 업데이트한다.

BP-CRNN 전체는 약 250,000개의 trainable parameters를 가진다. 이 중 fine-tuning되는 Conv3와 FC2에는 약 18,000개의 parameters가 포함된다. 즉 전체 parameter의 약 7.2%만 target patient data로 업데이트된다. 이는 제한된 개인 데이터에 overfitting되는 것을 막는 데 중요하다. 너무 많은 layer를 fine-tuning하면 target patient의 적은 sample에 모델이 과적합될 수 있고, 너무 적은 layer를 fine-tuning하면 개인화가 충분하지 않을 수 있다.

학습 설정은 non-transfer approach와 transfer approach로 나뉜다. Non-transfer method인 BP-CRNN에서는 100명의 각 patient에 대해 별도의 personalized model을 처음부터 학습한다. 이때 model parameters는 random initialization되고 모든 layer가 업데이트된다. Learning rate는 0.01, batch size는 32이다.

Transfer method인 BP-CRNN-Transfer에서는 100명의 환자를 두 그룹으로 나누어 실험한다. 첫 50명의 target patients에 대해서는 마지막 50명의 data로 initial model을 pre-train하고, 반대로 마지막 50명의 target patients에 대해서는 첫 50명의 data로 initial model을 pre-train한다. 이렇게 하여 target patient의 데이터가 pretraining에 포함되지 않도록 patient-level separation을 유지한다. Pre-training 단계에서는 learning rate 0.001, batch size 256을 사용하고, fine-tuning 단계에서는 learning rate 0.01, batch size 32를 사용한다. 모든 training session에는 early stopping이 적용되어 validation error가 증가하기 시작하면 learned weights를 저장한다.

평가에는 5-fold cross-validation이 사용되며, 각 patient에 대해 별도로 수행된다. 각 patient의 10시간 데이터 중 validation set과 test set은 각각 1시간 PPG-BP data로 구성된다. Training sample 수는 50, 100, 360, 1800, 3600개로 바꾸어 실험한다. 5초 segment 기준으로 3600 samples는 5시간 데이터에 해당한다. 각 network는 convergence 차이를 고려하기 위해 5번 학습되고 결과가 평균된다.

성능 지표는 mean absolute error, 즉 MAE이다. 논문에서 MAE는 다음과 같이 정의된다.

$$
MAE =
\frac{
\sum_{i=1}^{n}
\left| BP_{pred}^{i} - BP_{actual}^{i} \right|
}{n}
$$

여기서 $BP_{pred}^{i}$는 모델이 추정한 BP이고, $BP_{actual}^{i}$는 arterial catheter로 측정한 reference BP이다. $n$은 sample 수이다. MAE는 평균적으로 추정값이 실제값에서 몇 mmHg 벗어나는지를 나타내므로, 혈압 추정 모델의 직관적인 정확도 지표이다.

## 4. 실험 및 결과

실험은 personalized BP estimation에서 transfer learning이 얼마나 성능을 개선하는지, 그리고 개인별 training sample 수를 얼마나 줄일 수 있는지 평가하는 데 초점을 맞춘다. 비교 대상은 Aggregate BP-CRNN, Mean Regressor, BP-CRNN, BP-CRNN-Transfer, 그리고 이전 연구들이다.

Aggregate BP-CRNN은 여러 환자의 데이터를 사용해 학습하지만, target patient에 대한 personalization이나 transfer learning을 적용하지 않은 모델이다. 이 모델의 estimation error가 높게 나타났다는 점은 PPG-BP relationship을 효과적으로 모델링하려면 personalization이 필요하다는 것을 보여준다. 즉, 모든 사람에게 하나의 공통 모델을 그대로 적용하는 방식은 충분하지 않다.

Mean Regressor는 target patient training set의 평균 SBP와 평균 DBP만을 항상 예측하는 dummy baseline이다. 이 baseline은 중요한 비교 대상이다. 만약 특정 patient의 BP가 거의 변하지 않는다면, 단순히 평균값만 예측해도 낮은 오차가 나올 수 있기 때문이다. 따라서 BP-CRNN이 Mean Regressor보다 충분히 낮은 MAE를 보여야만 PPG와 BP 사이의 의미 있는 관계를 학습했다고 볼 수 있다.

Non-transfer personalized model인 BP-CRNN은 SBP MAE 4.59 mmHg, DBP MAE 2.72 mmHg를 달성했다. Transfer learning을 적용한 BP-CRNN-Transfer는 SBP MAE 3.52 mmHg, DBP MAE 2.20 mmHg를 달성했다. 이는 transfer learning을 통해 SBP 성능이 23.3%, DBP 성능이 19.1% 개선되었음을 의미한다. 또한 BP-CRNN-Transfer는 Mean Regressor의 SBP MAE 9.07 mmHg, DBP MAE 4.58 mmHg보다 훨씬 낮은 오차를 보였다. 이는 모델이 단순히 개인 평균 혈압을 예측하는 것이 아니라, PPG segment의 변화와 BP 사이의 관계를 학습했음을 보여준다.

기존 방법과 비교했을 때도 BP-CRNN-Transfer는 높은 성능을 보였다. 논문에 따르면 제안 방법은 저자들의 이전 RF-wavelet 방법보다 SBP에서 27.9%, DBP에서 15.7% 개선되었다. Spectro-temporal neural network 기반 방법과 비교하면 SBP 62.7%, DBP 68% 개선되었고, Siamese network 기반 방법과 비교하면 SBP 40.8%, DBP 35.5% 개선되었다. 또한 가장 가까운 성능을 보인 이전 transfer learning 기반 CNN 방법과 비교해서는 SBP MAE를 13.3% 개선했고, DBP MAE는 동일한 수준이라고 보고된다. 저자들은 이 개선이 raw PPG를 직접 처리하는 hybrid architecture와 선택적 layer fine-tuning 전략 덕분이라고 해석한다.

BHS standard 평가에서도 좋은 결과를 보였다. BHS Grade A는 reference BP와 estimated BP의 absolute difference가 5 mmHg 이하인 sample 비율이 최소 60%, 10 mmHg 이하가 최소 85%, 15 mmHg 이하가 최소 95%이어야 한다. Non-transfer BP-CRNN은 SBP에서 각각 72%, 92%, 97%를 달성했고, transfer learning을 적용한 BP-CRNN-Transfer는 80%, 95%, 98%를 달성했다. DBP의 경우 non-transfer는 89%, 98%, 99%, transfer는 93%, 99%, 100%를 달성했다. 따라서 두 방법 모두 SBP와 DBP에서 BHS Grade A를 만족하지만, transfer learning이 더 높은 비율을 보인다.

AAMI standard에서도 두 방법 모두 기준을 만족한다. AAMI 기준은 estimated BP와 reference BP 사이의 mean error가 5 mmHg 이하이고, error standard deviation이 8 mmHg 이하이어야 한다. Non-transfer BP-CRNN은 SBP에서 mean error와 SD가 -0.07 ± 5.49 mmHg, DBP에서 -0.05 ± 3.24 mmHg였다. BP-CRNN-Transfer는 SBP에서 0.11 ± 4.56 mmHg, DBP에서 0.05 ± 2.82 mmHg였다. 두 방법 모두 mean error가 거의 0에 가깝고, SD도 8 mmHg보다 충분히 낮다. 특히 transfer learning을 적용하면 SBP error SD가 5.49에서 4.56 mmHg로, DBP error SD가 3.24에서 2.82 mmHg로 감소한다.

Fine-tuning할 layer 조합에 따른 성능 분석도 중요하다. 저자들은 target patient 10명에 대해 다양한 layer combination을 fine-tuning하며 비교했다. 결과적으로 Conv3와 FC2만 fine-tuning하는 것이 가장 좋은 transfer learning performance를 보였다. Conv3를 개인화하지 않으면 SBP MAE가 3.84에서 4.41 mmHg로, DBP MAE가 2.24에서 2.63 mmHg로 증가했다. 이는 마지막 convolutional layer가 개인별 high-level PPG feature를 학습하는 데 중요하다는 것을 의미한다. 반대로 GRU까지 fine-tuning하면 SBP MAE가 3.90 mmHg, DBP MAE가 2.28 mmHg로 약간 나빠졌다. 논문은 GRU가 feature 자체보다는 feature 간 temporal relationship을 모델링하며, 이러한 temporal relationship은 개인 간에 transfer될 수 있기 때문이라고 해석한다.

Source patient 수에 따른 pretraining 효과도 분석되었다. 10명, 30명, 50명, 70명, 90명의 source patients를 사용하여 pre-train한 뒤 transfer performance를 비교했다. 평균적으로 source patient 수가 늘어날수록 MAE가 감소하지만, 50명을 넘으면 성능이 거의 plateau에 도달했다. 예를 들어 50, 70, 90명의 source patients를 사용할 때 SBP MAE는 각각 3.84, 3.85, 3.85 mmHg였고, DBP MAE는 2.24, 2.24, 2.23 mmHg였다. 이는 약 50명의 source patients만으로도 transferable PPG-BP feature를 학습하기에 충분한 variability가 포함될 수 있음을 시사한다.

Training set size 실험은 이 논문의 가장 중요한 실용적 결과 중 하나이다. Transfer learning은 모든 training sample 수에서 non-transfer learning보다 좋은 성능을 보였다. Training sample 수가 줄어들수록 두 방법 모두 MAE가 증가하지만, transfer learning을 사용할 때 증가 폭이 더 작았다. Non-transfer BP-CRNN은 100 training samples에서 SBP MAE 8.15 mmHg, DBP MAE 4.48 mmHg까지 나빠졌고, 50 samples에서는 수렴하지 못했다. 반면 BP-CRNN-Transfer는 100 samples에서 SBP MAE 5.52 mmHg, DBP MAE 3.38 mmHg를 달성했고, 50 samples에서도 수렴하여 SBP MAE 5.86 mmHg, DBP MAE 3.59 mmHg를 보였다.

특히 360 samples를 사용한 BP-CRNN-Transfer는 3600 samples를 사용한 non-transfer BP-CRNN과 거의 같은 성능을 보였다. Non-transfer 3600 samples에서는 SBP MAE 4.59 mmHg, DBP MAE 2.72 mmHg였고, transfer 360 samples에서는 SBP MAE 4.56 mmHg, DBP MAE 2.80 mmHg였다. 이는 transfer learning을 통해 10배 적은 개인 PPG-BP data로도 abundant data를 사용해 처음부터 학습한 개인 모델과 동등한 성능을 얻을 수 있음을 보여준다.

Bland-Altman analysis와 Pearson correlation analysis도 수행되었다. Bland-Altman analysis에서는 invasive arterial catheter와 BP-CRNN-Transfer 추정값 간의 차이를 두 방법의 평균값에 대해 plot한다. 논문에서 difference와 mean은 다음과 같이 정의된다.

$$
BP_{diff} = BP_{catheter} - BP_{BP-CRNN}
$$

$$
BP_{mean} = \frac{BP_{catheter} + BP_{BP-CRNN}}{2}
$$

95% limits of agreement는 measurement difference의 평균에 $\pm 1.96$배 standard deviation을 더하고 뺀 값으로 정의된다. 전체 100명 중 86%는 SBP measurement에서, 93%는 DBP measurement에서 이 agreement 기준을 달성했다. 또한 한 명의 대표 patient에 대해서는 SBP difference의 95.1%, DBP difference의 95.6%가 Bland-Altman limits 안에 들어갔고, Pearson correlation은 SBP 0.9, DBP 0.85로 나타났다.

전체 patient에 대한 평균 Pearson correlation도 transfer learning으로 개선되었다. Non-transfer approach에서는 SBP Pearson-R이 0.83 ± 0.10, DBP Pearson-R이 0.73 ± 0.17이었다. Transfer approach에서는 SBP Pearson-R이 0.90 ± 0.06, DBP Pearson-R이 0.82 ± 0.12로 증가했다. 이는 transfer learning이 단순히 MAE만 줄이는 것이 아니라 reference BP와 estimated BP 사이의 선형적 일치도도 높인다는 것을 의미한다.

마지막으로 source patient selection에 대한 초기 분석도 제시된다. 평균적으로는 50명 source patients로 pretraining하는 것이 10명보다 좋지만, 특정 target patient에 대해서는 더 작은 source patient subset이 오히려 더 좋은 transfer performance를 낼 수 있었다. 예를 들어 Patient 2의 경우 10명의 source patients로 pre-trained된 특정 Model 3이 50명 source patients로 pre-trained된 Model 1보다 SBP에서 13.9%, DBP에서 11.6% 더 좋은 성능을 보였다. 저자들은 향후 intelligent source patient selection을 연구할 계획이라고 설명한다.

## 5. 강점, 한계

이 논문의 첫 번째 강점은 raw PPG time series를 직접 사용한다는 점이다. 기존 PWA 기반 방법은 PPG cycle에서 key point를 탐지하고 morphology feature를 수작업으로 추출해야 한다. 하지만 PPG는 noise와 개인차에 민감하므로 feature extraction이 불안정할 수 있다. BP-CRNN은 convolutional layers를 통해 raw signal에서 feature를 자동으로 학습하므로, hand-crafted feature의 누락이나 redundancy 문제를 줄일 수 있다.

두 번째 강점은 architecture가 목적에 맞게 구성되어 있다는 점이다. Convolutional layers는 local waveform pattern을 추출하고, GRU는 시간적 관계를 학습하며, fully connected layers는 SBP와 DBP 추정을 수행한다. 첫 번째와 세 번째 convolutional layer의 output을 함께 GRU에 전달하는 설계는 low-level feature와 high-level feature를 동시에 활용하려는 명확한 의도를 가진다.

세 번째 강점은 transfer learning을 매우 실용적인 방식으로 적용했다는 점이다. 전체 network를 fine-tuning하지 않고 Conv3와 FC2만 fine-tuning함으로써 개인화와 overfitting 방지 사이의 균형을 잡았다. 전체 parameter의 7.2%만 업데이트한다는 점은 target patient의 개인 데이터가 적을 때 특히 중요하다.

네 번째 강점은 patient-level data separation을 유지했다는 점이다. Transfer learning 실험에서 target patient의 data가 pretraining에 포함되지 않도록 첫 50명과 마지막 50명을 교차 사용했다. 이는 개인화 혈압 추정 연구에서 매우 중요한 평가 설계이며, 같은 사람의 데이터가 source와 target에 동시에 들어가는 leakage를 방지한다.

다섯 번째 강점은 개인 데이터 요구량을 체계적으로 분석했다는 점이다. 단순히 최종 성능만 보고한 것이 아니라, training sample 수를 50, 100, 360, 1800, 3600개로 바꾸어 transfer learning의 효과를 평가했다. 이를 통해 transfer learning이 특히 limited personal data setting에서 유용함을 명확히 보여준다.

여섯 번째 강점은 BHS, AAMI, Bland-Altman, Pearson correlation 등 여러 관점에서 결과를 평가했다는 점이다. MAE만 보고하면 모델이 실제 혈압 측정 기준을 만족하는지 판단하기 어렵다. 이 논문은 의료 측정 장치 평가에 사용되는 기준과 gold-standard arterial catheter와의 agreement 분석을 포함하여 결과의 임상적 의미를 강화했다.

그러나 한계도 존재한다. 첫째, 데이터가 ICU 환자에서 수집된 MIMIC III 기반이라는 점이다. ICU 환자의 PPG와 BP dynamics는 일반 사용자의 일상 생활 중 wearable PPG와 다를 수 있다. 환자는 침상에 누워 있고 sensor가 비교적 안정적으로 부착되어 있을 가능성이 높지만, 실제 wearable 환경에서는 움직임, 땀, 피부 접촉 변화, ambient light, sensor displacement가 훨씬 더 심하다. 따라서 이 모델이 일상 wearable 환경에서 동일하게 동작한다고 단정할 수 없다.

둘째, reference BP는 arterial catheter로 정확하지만, 이 때문에 데이터 수집 환경이 병원 ICU로 제한된다. 논문의 목표는 wearable BP monitoring이지만, 모델은 invasive reference가 있는 병원 데이터로 학습 및 평가되었다. 실제 환경에서는 cuff-based intermittent BP 또는 다른 reference를 사용해야 하므로, transfer learning을 위한 개인별 BP sample을 어떻게 현실적으로 얻을지가 여전히 중요한 문제로 남는다.

셋째, 5초 PPG segment마다 평균 SBP와 평균 DBP를 label로 사용한다는 점은 beat-to-beat BP variation을 완전히 반영하지 않을 수 있다. 모델은 5초 단위 평균 혈압을 추정하므로, 개별 pulse level의 급격한 혈압 변화나 beat-level variability를 정밀하게 추정하는 모델이라고 보기는 어렵다. 논문은 5초마다 BP estimation을 제공한다고 설명하지만, beat-to-beat continuous BP estimation과는 구분해야 한다.

넷째, autocorrelation threshold 0.7은 empirical threshold로 설정되었다. 이 기준은 MIMIC III의 PPG segment 품질 판별에는 유용할 수 있지만, 다른 sensor, 다른 sampling rate, 실제 wearable 환경에서도 동일한 threshold가 적절한지는 검증이 필요하다. 또한 corrupted segment를 제거한 후의 성능이므로, real-world noise가 많은 상황에서 coverage와 accuracy가 어떻게 변하는지는 추가 분석이 필요하다.

다섯째, 모델 inference time은 GPU 기반 PC 환경에서 0.32 ± 0.09초로 보고되었다. 이는 5초 단위 추정에는 충분히 빠르지만, wearable device에 직접 탑재하기에는 모델 크기, memory, energy consumption 문제가 남는다. 저자들도 future work에서 lightweight model과 wearable implementation을 연구하겠다고 밝힌다.

여섯째, BHS와 AAMI 기준을 만족하지만 Bland-Altman agreement는 전체 100명 중 SBP 86%, DBP 93%에서 달성되었다고 보고된다. 대표 patient에서는 95% 이상이 limits of agreement 안에 들어가지만, 전체 patient 기준으로는 모든 환자가 충분한 agreement를 보인 것은 아니다. 특히 SBP에서 개인별 agreement가 더 어려운 것으로 보인다. 이는 특정 환자군에서 모델이 여전히 부족할 수 있음을 시사한다.

비판적으로 보면, 이 논문은 PPG-only BP estimation에서 transfer learning의 실용성을 매우 설득력 있게 보여준다. 특히 개인별 BP data가 제한된 상황에서 source patient data를 활용하는 접근은 wearable BP monitoring의 핵심 bottleneck을 직접 겨냥한다. 다만 실제 상용 또는 임상 응용으로 가기 위해서는 ICU가 아닌 ambulatory environment에서의 validation, sensor variation에 대한 robustness, calibration sample acquisition protocol, privacy-preserving personalization, lightweight on-device inference가 추가로 필요하다.

## 6. 결론

이 논문은 raw PPG time series를 직접 입력으로 사용하여 개인화된 SBP와 DBP를 추정하는 BP-CRNN architecture와, 제한된 개인 데이터를 활용하기 위한 BP-CRNN-Transfer 방법을 제안했다. BP-CRNN은 convolutional layers, GRU, fully connected layers를 결합하여 waveform feature extraction, temporal dependency modeling, BP regression을 하나의 end-to-end 구조로 수행한다.

가장 중요한 기여는 transfer learning을 통해 개인별 데이터 요구량을 크게 줄였다는 점이다. 여러 source patients의 데이터를 사용해 pre-trained model을 만들고, target patient에 대해서는 Conv3와 FC2만 fine-tuning함으로써 약 7.2%의 parameter만 업데이트한다. 이 선택적 fine-tuning 전략은 개인화 성능을 높이면서도 limited target data에 대한 overfitting을 줄인다.

실험 결과, BP-CRNN-Transfer는 SBP MAE 3.52 mmHg, DBP MAE 2.20 mmHg를 달성했으며, non-transfer BP-CRNN보다 각각 23.3%, 19.1% 성능을 개선했다. 또한 BHS Grade A와 AAMI 기준을 모두 만족했고, Pearson correlation과 Bland-Altman analysis에서도 arterial catheter reference와 높은 일치성을 보였다. 특히 transfer learning을 사용하면 360개의 개인 sample만으로도 transfer learning 없이 3600개 sample을 사용한 개인 모델과 비슷한 성능을 달성했다. 50개의 개인 sample만으로도 모델이 수렴하여 SBP MAE 5.86 mmHg, DBP MAE 3.59 mmHg를 보인 점은 실제 개인화 wearable BP estimation에 중요한 가능성을 제시한다.

이 연구는 PPG-based cuffless BP monitoring에서 personalization과 transfer learning이 핵심적인 역할을 할 수 있음을 보여준다. 향후 연구에서는 실제 wearable 환경에서의 검증, source patient selection 최적화, lightweight model 설계, on-device inference, privacy-preserving learning이 중요해질 것이다. 제공된 결과만으로는 일상 환경에서의 완전한 임상적 신뢰성을 보장할 수 없지만, 이 논문은 raw PPG 기반 개인화 혈압 추정의 중요한 방향을 제시한 연구로 평가할 수 있다.
