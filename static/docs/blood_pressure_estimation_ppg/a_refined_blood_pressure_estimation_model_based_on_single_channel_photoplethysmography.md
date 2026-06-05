# A Refined Blood Pressure Estimation Model Based on Single Channel Photoplethysmography

* **저자**: Yiming Zhang, Xianglin Ren, Xiao Liang, Xuesong Ye, Congcong Zhou
* **발표연도**: 2022

## 1. 논문 개요

이 논문은 single-channel photoplethysmography, 즉 단일 채널 PPG 신호를 이용하여 long-term blood pressure, 즉 장기 혈압 변화를 추정하는 refined blood pressure estimation model을 제안한다. 핵심 목표는 모든 사람에게 하나의 혈압 예측 모델을 적용하는 대신, 먼저 사용자의 cardiovascular status를 분류하고, 이후 해당 집단에 맞는 혈압 추정 모델을 적용함으로써 BP estimation의 정확도와 안정성을 높이는 것이다.

논문이 다루는 주요 연구 문제는 두 가지이다. 첫째, 단일 PPG 신호와 demographic information, 즉 age와 gender만으로 normal, atrial fibrillation, coronary arteriosclerosis 세 집단을 구분할 수 있는가이다. 둘째, 이렇게 분류된 cardiovascular disease group별로 별도의 deep learning 기반 BP prediction model을 만들고, transfer learning을 이용해 individual-specific model로 fine-tuning하면 장기 혈압 추정 성능을 개선할 수 있는가이다.

이 문제가 중요한 이유는 cardiovascular diseases, 즉 CVDs가 전 세계 사망 원인의 큰 비중을 차지하며, blood pressure가 심혈관 상태를 반영하는 핵심 physiological parameter이기 때문이다. 혈압은 생리 주기, 감정, 외부 자극에 따라 변동하므로, discrete measurement보다 continuous monitoring이 더 많은 정보를 제공할 수 있다. 특히 BP의 circadian variation과 long-term trend는 sudden cardiovascular events 예방에 중요하다.

기존의 auscultation 또는 oscillometric BP measurement는 cuff를 필요로 하며, 연속적이고 장기적인 모니터링에는 불편하다. Invasive arterial catheter는 연속 혈압 측정이 가능하지만 침습적이기 때문에 일반적인 wearable 또는 home healthcare 환경에 적용하기 어렵다. 따라서 PPG 기반 cuffless BP estimation은 실용적 가치가 높다. 그러나 PPG morphology는 나이, 혈관 탄성, 질환 상태, 개인차에 따라 크게 달라진다. 이 논문은 바로 이러한 개인차와 질환별 차이를 무시한 기존 모델이 서로 다른 CVD population에서 큰 오차를 만들 수 있다고 보고, “질환 분류 후 혈압 추정”이라는 refined modeling strategy를 제안한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 BP estimation을 바로 수행하지 않고, 먼저 PPG morphology와 demographic characteristics를 이용해 cardiovascular status를 stratification한 뒤, 각 group에 맞는 BP prediction model을 적용하는 것이다. 논문에서 다룬 세 집단은 normal subjects, atrial fibrillation, 즉 AF subjects, 그리고 coronary arteriosclerosis, 즉 CA subjects이다.

기존 BP estimation model은 population 전체를 하나의 모델로 묶어 학습하는 경우가 많았다. 그러나 CVD는 heart function, vascular elasticity, arterial stiffness, pulse wave propagation, vascular resistance 등에 영향을 주며, 이는 PPG waveform의 형태를 바꾼다. 예를 들어 AF에서는 pulse-to-pulse interval이 불규칙해지고 PPG waveform shape가 크게 변할 수 있다. CA에서는 dicrotic wave가 줄어들거나 사라지는 현상이 나타날 수 있다. 따라서 같은 PPG signal이라도 질환군에 따라 BP와 PPG 사이의 mapping이 다를 수 있다.

논문은 이 문제를 해결하기 위해 먼저 Random Forest classifier로 disease classification을 수행한다. 입력에는 PPG morphological features와 age, gender가 포함된다. RF classifier는 normal, AF, CA 세 class를 분류하며, 92.19%의 accuracy를 달성하였다. demographic information을 포함하지 않았을 때보다 성능이 10.2% 향상되었다고 보고한다. 이는 PPG morphology뿐 아니라 age와 gender가 cardiovascular status screening에 유의미한 정보를 제공함을 보여준다.

두 번째 핵심 아이디어는 BiLSTM-At 모델을 사용하여 raw PPG sequence에서 long-term BP trend를 추정하는 것이다. BiLSTM은 forward와 backward direction의 temporal dependency를 모두 활용할 수 있고, attention mechanism은 긴 sequence에서 중요한 hidden state나 feature vector에 더 큰 weight를 부여한다. 따라서 PPG signal의 시간적 구조를 모델링하는 데 적합하다.

세 번째 핵심 아이디어는 transfer learning을 이용한 personalized modeling이다. Universal model을 먼저 여러 subject의 PPG와 BP data로 학습한 뒤, target patient의 소량 데이터로 마지막 fully connected layer만 fine-tuning한다. 이를 통해 새로운 사용자에 대해 필요한 training data와 computational cost를 줄이면서도 individual-specific PPG-BP relationship을 반영할 수 있다. 논문 결과에 따르면, universal model은 오차가 컸지만, transfer learning을 적용한 personalized model은 normal, AF, CA 세 집단 모두에서 AAMI와 BHS Class A 기준을 만족하였다.

## 3. 상세 방법 설명

논문의 전체 시스템은 preprocessing, disease classification, BP prediction model, transfer learning의 네 부분으로 구성된다. 먼저 MIMIC-III Waveform Database Matched Subset에서 PPG와 ABP waveform을 추출하고, MIMIC-III Clinical Database에서 age와 gender 및 ICD-9 code 기반 진단 정보를 가져온다. 이후 PPG와 ABP를 filtering하고 quality screening을 수행하여 clean segment를 구성한다. 다음으로 PPG morphology와 demographic features를 이용해 normal, AF, CA를 분류한다. 마지막으로 각 group에 대해 BiLSTM-At 기반 BP estimation model을 학습하고, transfer learning으로 개인별 모델을 fine-tuning한다.

### 3.1 데이터셋 구성

사용된 데이터는 MIMIC-III Waveform Database Matched Subset이다. 이 데이터셋은 10,282명의 ICU patient로부터 얻은 22,317 waveform record를 포함하며, ECG, ABP, respiration, PPG 등의 digitized signal이 포함되어 있다. sampling frequency는 125 Hz이다. 이 논문에서는 fingertip PPG와 beat-to-beat ABP signal을 가진 record를 추출하고, 해당 subject의 age와 gender 정보를 Clinical Database에서 가져왔다.

전처리 후에는 2,028명 subject의 29,289 records가 확보되었다. 이후 long-term estimation 성능을 평가하기 위해 최소 50개 segment를 포함하는 long record가 선택되었다. 연구 목적이 normal, AF, CA 세 집단에서 BP prediction 성능을 비교하는 것이므로, ICD-9 code를 기준으로 각 집단 85명씩 총 255명을 선택하였다. 최종적으로 1,387 records와 69,350개의 10초 segment가 분석에 포함되었다. 성별 구성은 남성 157명, 여성 98명이다.

### 3.2 Preprocessing

전처리는 signal quality를 확보하고 computational complexity를 줄이기 위해 수행된다. 먼저 PPG와 ABP가 최소 10분 이상 존재하지 않는 record는 삭제하였다. 긴 waveform segment는 BP fluctuation을 포함할 수 있으므로 long-term estimation 평가에 적합하다고 보았다.

PPG signal에는 4th order Butterworth band-pass filter가 적용되었다. cutoff frequency는 0.5–8 Hz이며, baseline drift와 high-frequency noise를 제거하는 것이 목적이다. ABP signal에는 spike noise 제거를 위해 Hampel filter가 적용되었다. Hampel filter는 각 ABP sample과 주변 6개 sample의 median을 계산하고, median에서 세 standard deviation 이상 벗어나는 sample을 median으로 대체한다.

또한 flat peak와 flat line이 있는 waveform은 saturation 또는 lack of signal을 의미하므로 제거되었다. 전체 signal segment의 30% 이상이 flat peak이거나, record duration의 10% 이상이 flat line인 record는 제외되었다. 이후 각 record는 연속적인 10초 segment로 나뉘었다.

각 segment에 대해서도 quality threshold가 적용되었다. PPG와 ABP에 대해 flat line, flat peak, skewness index, autocorrelation index가 사용되었다. Skewness는 PPG signal의 asymmetry 정도를 나타내며, threshold는 0.2로 설정되었다. 논문에서 skewness-related signal quality index는 다음과 같이 제시된다.

$$
SSQI=\frac{1}{N}\sum_{i=1}^{N}\left[\frac{x_i-\hat{\mu}_x}{\sigma}\right]^2
$$

여기서 $\hat{\mu}_x$와 $\sigma$는 각각 $x_i$의 empirical mean과 standard deviation이며, $N$은 PPG signal sample 수를 의미한다. 다만 일반적인 통계학의 skewness 정의와는 형태가 다르게 표기되어 있으므로, 제공된 텍스트 기준으로는 이 식이 논문에서 사용한 SSQI 표현이라고 이해하는 것이 적절하다.

Autocorrelation index는 uncorrupted PPG segment가 높은 periodicity를 가져야 한다는 가정에 기반한다. PPG autocorrelation signal의 peak value를 quality 판단에 사용하였고, threshold는 0.7로 설정되었다. 모든 PPG와 ABP signal은 이후 max-min normalization을 거쳐 subsequent model의 입력으로 사용되었다.

### 3.3 Disease classification

Disease classification 단계에서는 PPG morphological features와 demographic features를 결합하여 normal, AF, CA 세 class를 분류한다. 논문은 cardiovascular disease가 혈관의 기하학적 형태와 기계적 성질을 바꾸며, 이 변화가 pulse wave intensity, shape, rhythm, rate에 반영된다고 설명한다. 따라서 PPG morphology는 CVD screening에 사용할 수 있다.

사용된 PPG morphological feature는 총 7개이며, 구체적 feature representation은 기존 연구와 일치한다고 설명한다. 제공된 텍스트에서 명시적으로 설명된 feature는 rise time, descent time, photoplethysmography area, peak 등이다. Rise time은 heart contraction과 left ventricular function을 반영하고, descent time은 ventricular diastole과 관련된다. Photoplethysmography area는 total peripheral resistance와 blood vessel tension 변화와 관련되며, peak는 blood perfusion을 반영한다.

Classifier로는 SVM과 Random Forest가 비교되었다. SVM은 medical research에서 널리 사용되는 classifier이며, 이 연구에서는 rbf kernel을 사용하였다. misclassification penalty $C$는 0.9, tolerance termination criterion은 0.001로 설정되었고, 이는 grid search로 얻었다. Random Forest는 여러 decision tree의 ensemble로 구성되며, estimator 수는 25로 설정되었다.

Morphological feature는 전처리된 69,350개의 PPG segment에서 추출되었고, 각 segment feature value는 평균화되었다. Age는 같은 subject에서 반복되는 값이므로, $\pm 3$년 범위의 random Gaussian noise를 추가하여 분산되도록 하였다. Feature는 training set과 test set으로 8:2 비율로 나뉘었다. Training set에서 5-fold cross-validation을 통해 best model을 선택하고, test set에서 최종 성능을 평가하였다. Ground truth는 MIMIC Clinical Database에서 추출한 ICD-9 code이다.

### 3.4 BiLSTM-At BP prediction model

BP prediction model은 single-channel raw PPG signal로부터 SBP와 DBP를 추정하기 위해 설계된 Bidirectional Long Short-Term Memory-Attention Neural Network, 즉 BiLSTM-At이다. 이 모델은 각 disease group별로 universal model을 학습하고, 이후 transfer learning으로 individual model을 fine-tuning한다.

LSTM은 RNN의 vanishing gradient와 exploding gradient 문제를 완화하기 위해 설계된 구조이다. LSTM은 cell state $C_t$와 hidden state $h_t$라는 두 transmission state를 갖는다. $C_t$는 비교적 천천히 변하는 long-term memory에 해당하고, $h_t$는 각 time step마다 달라지는 short-term memory에 해당한다. LSTM 내부에는 forget gate, input gate, output gate가 있으며, 각각 어떤 정보를 버릴지, 어떤 정보를 유지할지, 어떤 정보를 출력할지 조절한다.

BiLSTM은 forward RNN layer와 backward RNN layer를 함께 사용한다. 즉 PPG sequence를 앞에서 뒤로, 뒤에서 앞으로 모두 읽고 각 time step의 결과를 concatenate한다. 이를 통해 past sequence information뿐 아니라 future sequence information도 feature representation에 반영할 수 있다. 이 구조는 10초 PPG segment 내에서 혈압 추정에 유용한 temporal pattern을 더 풍부하게 포착하려는 목적을 가진다.

Attention mechanism은 BiLSTM output 중 혈압 예측에 중요한 hidden state나 feature vector에 더 큰 weight를 부여한다. LSTM은 일반 RNN보다 긴 dependency를 잘 다루지만, 매우 긴 sequence에서는 모든 long-range dependency를 정확히 기억하지 못할 수 있다. Attention layer는 각 time step의 hidden state를 모두 고려하면서 target output과 관련성이 높은 부분에 집중하도록 한다.

논문은 Bahdanau attention을 사용하였다. Context vector $c_i$는 attention weight vector $a_i$와 hidden state $h_{t_i}$의 weighted sum으로 계산된다.

$$
c_i=\sum_{i=0}^{T}a_i h_{t_i}
$$

Attention weight $a_i$는 score function을 softmax로 normalize하여 계산된다.

$$
a_i=\frac{\exp(score(h_{t_i},h_s))}{\sum_{s'\in s}\exp(score(h_{t_i},h_{s'}))}
$$

Score function은 다음과 같이 정의된다.

$$
score(h_t,\bar{h}_s)=v_a^T\tanh(W_1h_t+W_1\bar{h}_s)
$$

여기서 $h_s$는 last time step의 output을 의미하고, $v_a$와 $W_1$은 attention mechanism에서 학습되는 parameter이다. 이 attention 구조는 PPG sequence 전체 중 SBP와 DBP 예측에 더 중요한 temporal feature를 강조한다.

BiLSTM-At의 output vector는 demographic information과 결합된 뒤 fully connected layer를 통과하여 SBP와 DBP를 추정한다. Universal model에서는 LSTM unit 수가 각 layer마다 128로 설정되었고, fully connected layer는 2개의 unit을 가지며 ReLU activation function을 사용한다. Loss function은 mean squared error, 즉 MSE이고, optimizer는 Adam이다. Learning rate는 0.001, batch size는 512이며, patience가 5 epochs에 도달하면 training을 중단하였다.

### 3.5 Universal model과 transfer model

Universal model은 각 disease group별로 여러 subject의 data를 사용해 학습된다. Dataset은 subject 수 기준으로 60% training, 20% validation, 20% testing으로 나뉜다. Validation set은 best-optimized model을 결정하는 데 사용되고, test set은 전체 population에 대한 generalization performance를 평가하는 데 사용된다. Universal model 학습 시에는 model parameter가 random initialization되고, 모든 layer가 update된다.

그러나 universal model은 높은 estimation error를 보였다. 논문은 그 원인을 PPG waveform이 person-to-person으로 크게 다르고, 여러 요인의 interference가 존재하기 때문이라고 해석한다. 즉 single-channel PPG alone으로 population-level universal mapping을 만드는 것은 충분하지 않으며, individual-specific modeling이 필요하다는 결론을 제시한다.

Transfer model은 이러한 문제를 해결하기 위해 도입되었다. 먼저 source patient pool의 대량 PPG와 BP data로 universal model을 pre-training한다. 이후 target patient의 소량 data를 사용해 individual model을 fine-tuning한다. 이때 shallow layer는 general feature representation을 저장한다고 보고 parameter weight를 고정하며, 마지막 fully connected layer만 update한다. 이렇게 하면 새로운 subject에 필요한 training data 수와 model parameter update 범위를 줄이면서 individual-specific PPG feature representation을 얻을 수 있다.

Target patient data는 transfer model 학습을 위해 6:2:2 비율로 training, validation, testing으로 나뉜다. 논문은 target patient의 data가 pre-training 과정에는 사용되지 않았다고 명시한다. 이는 data leakage를 방지하기 위한 설계이다. Transfer learning model에서도 MSE가 loss function으로 사용되었고, Adam optimizer가 적용되었다. Fine-tuning 시 learning rate는 0.001, batch size는 64로 설정되었다.

## 4. 실험 및 결과

실험은 disease classification, universal BP prediction, transfer BP prediction, long-term validation, related work comparison, demographic parameter 영향 분석으로 구성된다.

### 4.1 Disease classification 결과

SVM과 RF classifier를 비교한 결과, RF가 더 좋은 성능을 보였다. Test set에서 실제 label 수는 normal 4551개, AF 4316개, CA 5003개로 비교적 균형 잡혀 있었다. RF의 predicted label 수는 normal 4277개, AF 3882개, CA 4678개로 실제 분포와 큰 차이가 없었다.

정량적으로 SVM의 accuracy는 80.55%였고, RF의 accuracy는 92.19%였다. 따라서 이후 disease classification에는 RF가 사용되었다. 논문은 demographic information을 포함하면 classification accuracy가 10.2% 향상된다고 보고한다. 또한 RF의 variable importance score 분석에서 age가 0.22로 가장 큰 contribution을 보였고, gender는 더 작은 contribution을 보였다. 이는 vascular aging과 CVD risk가 age와 강하게 관련된다는 생리적 배경과 일관된다.

흥미롭게도 demographic parameter만 사용한 classification accuracy는 0.4423에 불과하였다. 즉 age와 gender만으로는 CVD classification이 충분하지 않으며, PPG morphological feature가 핵심 정보를 제공한다. 반대로 PPG feature만 사용할 때보다 demographic feature를 함께 사용할 때 성능이 더 좋으므로, 두 정보가 상호보완적이라고 해석할 수 있다.

### 4.2 Universal model 결과

Universal model은 각 disease population 내 여러 subject data를 사용해 학습되었다. 이 모델의 목적은 같은 disease group 내에서 population-level BP prediction이 가능한지 확인하는 것이다. Raw ABP signal에서 peak와 trough를 추출하여 각각 SBP와 DBP reference value로 사용하였다. Evaluation metric은 MAE와 RMSE이다.

Universal model의 성능은 상대적으로 낮았다. Normal population에서 SBP MAE는 13.110 mmHg, DBP MAE는 4.963 mmHg였다. AF population에서는 SBP MAE 12.796 mmHg, DBP MAE 3.989 mmHg였다. CA population에서는 SBP MAE 19.369 mmHg, DBP MAE 11.734 mmHg로 가장 큰 오차를 보였다.

이 결과는 같은 disease group으로 나누어도, 하나의 universal model이 모든 subject의 PPG-BP relationship을 충분히 설명하지 못한다는 것을 의미한다. 특히 CA group에서 오차가 큰 것은 coronary arteriosclerosis가 PPG morphology와 BP relationship에 더 복잡한 변화를 만들거나, 개인별 vascular condition 차이가 더 크기 때문일 수 있다. 다만 제공된 텍스트에서는 이 원인을 정량적으로 추가 분석하지 않으므로, 이는 논문 결과에 근거한 가능한 해석에 가깝다.

논문은 universal model의 높은 error를 PPG waveform의 개인차와 여러 interference factor 때문이라고 설명한다. 따라서 effective PPG-BP model은 individual-level modeling 또는 personalization이 필요하다고 결론짓는다.

### 4.3 Non-transfer model과 transfer model 결과

개인별 modeling의 효과를 보기 위해 non-transfer model과 transfer model이 비교되었다. Non-transfer model은 각 patient에 대해 random initialization model을 처음부터 학습하는 방식이다. Transfer model은 universal model을 기반으로 마지막 layer만 fine-tuning하는 방식이다.

Non-transfer model은 universal model보다 훨씬 좋은 성능을 보였다. Normal population에서 SBP MAE는 3.327 mmHg, DBP MAE는 2.019 mmHg였다. AF population에서는 SBP MAE 5.649 mmHg, DBP MAE 2.378 mmHg였고, CA population에서는 SBP MAE 8.533 mmHg, DBP MAE 4.709 mmHg였다. 이는 individual-specific modeling이 population-level universal modeling보다 훨씬 효과적임을 보여준다.

Transfer model은 non-transfer model보다도 더 좋은 성능을 보였다. Normal population에서 SBP MAE는 2.815 mmHg, DBP MAE는 1.876 mmHg였다. AF population에서는 SBP MAE 3.024 mmHg, DBP MAE 1.334 mmHg였다. CA population에서는 SBP MAE 4.444 mmHg, DBP MAE 2.549 mmHg였다.

논문은 transfer model이 non-transfer model 대비 성능을 개선했다고 보고한다. Normal population에서는 SBP와 DBP MAE가 각각 15.4%, 7.1% 개선되었다. AF population에서는 각각 46.5%, 43.9% 개선되었고, CA population에서는 각각 47.9%, 45.9% 개선되었다. 특히 diseased population에서 transfer learning의 성능 향상이 더 컸다는 점은, disease-specific universal representation을 학습한 후 개인화하는 전략이 질환군에서 유효함을 보여준다.

### 4.4 AAMI 및 BHS 기준 평가

AAMI standard는 estimated BP와 reference BP 사이의 mean error가 5 mmHg 이하이고, error의 standard deviation이 8 mmHg 이하일 것을 요구한다. 논문은 transfer learning model을 사용한 normal, AF, CA 세 group의 SBP와 DBP error distribution이 모두 AAMI criteria를 만족한다고 보고한다.

BHS standard는 BP measurement system의 cumulative error가 5 mmHg, 10 mmHg, 15 mmHg threshold 이하에 얼마나 들어오는지를 기준으로 grade를 부여한다. 논문에 따르면 세 group의 transfer learning model은 SBP와 DBP 모두 BHS Class A 기준을 만족하였다. 또한 AAMI와 BHS standard는 최소 85명의 subject를 요구하는데, 이 연구는 normal, AF, CA 각각 85명으로 이 요건을 충족한다고 명시한다.

### 4.5 Long-term validation 결과

논문은 기존 데이터셋을 기반으로 1–2일 수준의 long-term validation도 수행하였다. 각 group에서 가장 긴 subject record를 선택하여 continuous data segment로 나누었다. 사용된 subject와 duration은 normal subject P047814 17.5시간, AF patient P043738 46.0시간, CA patient P027162 20.6시간이다.

Transfer model을 calibration한 뒤, 이후 PPG data로 BP values를 추정하여 long-term performance를 평가하였다. Normal population에서는 SBP MAE 4.004 mmHg, DBP MAE 3.582 mmHg를 보였다. AF population에서는 SBP MAE 4.146 mmHg, DBP MAE 2.120 mmHg였고, CA population에서는 SBP MAE 4.631 mmHg, DBP MAE 2.324 mmHg였다.

논문은 시간이 지나면서 model performance가 약간 감소하지만, 결과는 여전히 acceptable하다고 설명한다. 그러나 수일, 1개월, 반년, 1년 이상의 더 긴 기간에 대한 성능 평가는 future work로 남겨두었다. 따라서 이 논문에서 검증된 long-term performance는 최대 46시간 수준으로 이해해야 하며, 실제 장기간 wearable monitoring에서 calibration drift가 얼마나 발생하는지는 아직 명확하지 않다.

### 4.6 기존 연구와의 비교

논문은 MIMIC 기반 single-channel PPG BP estimation 연구들과 비교하였다. 다만 raw data, experimental condition, evaluation metric, calibration strategy가 서로 다르기 때문에 직접 비교가 어렵다고 명시한다. 이 점은 중요한데, BP estimation 연구에서는 subject split, segment split, calibration 여부, preprocessing, ground truth alignment에 따라 결과가 크게 달라질 수 있기 때문이다.

비교 대상으로는 spectro-temporal ResNet, CNN with Siamese network, fusion neural network, seq2seq model with demographic information, LSTM-based signal-to-signal translation, CRNN-based transfer learning, visibility graph와 transfer learning을 결합한 방법 등이 언급된다. 논문은 제안한 BiLSTM-At transfer learning model이 더 간단한 fine-tuning 구조를 사용하면서도 normal, AF, CA group에서 높은 accuracy를 보였다고 주장한다.

특히 Leitner et al.의 transfer learning 기반 CRNN model은 subject당 최소 10시간 data가 필요하고, transfer 과정에서 convolutional layer와 fully connected layer를 fine-tuning해야 했다고 설명한다. 반면 본 논문은 마지막 fully connected layer만 update하므로 더 간단하고 computationally efficient하다고 주장한다. 다만 제공된 텍스트 기준으로는 모든 비교가 동일한 split과 동일한 protocol에서 수행된 것은 아니므로, “우수하다”는 결론은 논문이 제시한 비교표의 범위 안에서 해석해야 한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 BP estimation 전에 cardiovascular status를 먼저 분류하는 refined modeling strategy를 제안했다는 점이다. PPG morphology는 질환, 나이, 혈관 탄성, 심박 리듬 등에 따라 달라지므로, 모든 subject를 하나의 모델에 넣는 방식은 오차를 키울 수 있다. 이 논문은 normal, AF, CA group을 나누고 각 group별 BP model을 구성함으로써 질환별 PPG-BP relationship 차이를 반영하려 했다. 이는 기존 단일 population model보다 생리학적으로 설득력 있는 접근이다.

두 번째 강점은 single-channel PPG만을 continuous waveform input으로 사용한다는 점이다. PTT나 PAT 기반 방법은 ECG와 PPG 또는 여러 PPG sensor가 필요하지만, 이 논문은 단일 PPG sensor와 demographic information만으로 BP를 추정한다. 이는 wearable device, smartwatch, home healthcare system에 적용하기 쉬운 구조이다.

세 번째 강점은 disease classification과 BP estimation을 모두 실제 clinical database와 waveform database를 연결하여 수행했다는 점이다. MIMIC-III Waveform Database와 Clinical Database를 함께 사용하여 PPG, ABP, demographic data, ICD-9 diagnosis label을 결합하였다. 이를 통해 단순 waveform regression이 아니라 cardiovascular status-aware BP estimation이라는 더 복합적인 문제를 다루었다.

네 번째 강점은 transfer learning을 개인화에 활용했다는 점이다. Universal model은 오차가 크지만, transfer learning으로 마지막 layer만 fine-tuning하면 normal, AF, CA 모두에서 큰 성능 향상이 나타났다. 특히 diseased population인 AF와 CA에서 improvement가 컸다는 점은, source domain에서 학습한 disease-specific representation과 target individual calibration이 결합될 때 효과적임을 보여준다.

다섯 번째 강점은 AAMI와 BHS Class A 기준을 모두 만족했다는 결과를 제시한 점이다. BP estimation 연구는 단순 MAE만으로 평가하면 실제 의료기기 기준과의 관련성이 약해질 수 있다. 이 논문은 AAMI와 BHS라는 표준 기준으로 성능을 검증하고, 각 group별 최소 85명 subject 기준도 충족한다고 설명한다.

그러나 한계도 분명하다. 첫째, universal model의 성능이 낮다. Normal과 AF group에서도 SBP MAE가 약 13 mmHg 수준이고, CA group에서는 SBP MAE 19.369 mmHg, DBP MAE 11.734 mmHg로 상당히 크다. 이는 PPG-based BP estimation이 population-level model만으로는 어렵고, 개인화가 사실상 필수적임을 의미한다. 따라서 제안 기법의 좋은 성능은 transfer learning 기반 personalized calibration이 전제된 결과로 이해해야 한다.

둘째, transfer learning에는 target patient의 reference BP data가 필요하다. 논문은 transfer learning이 적은 data로도 individual model을 만들 수 있다고 설명하지만, target patient의 PPG와 reference BP가 어느 정도 필요하며, 이를 어떤 방식으로 실제 wearable 환경에서 얻을지에 대한 구체적 operational protocol은 충분히 설명되지 않는다. Cuffless BP monitoring의 실용성은 calibration 부담과 밀접하게 관련되므로, 이 부분은 중요한 미해결 문제이다.

셋째, long-term validation은 최대 46시간 수준이다. 논문은 1–2일 연속 데이터에서 성능이 약간 감소하지만 acceptable하다고 보고한다. 그러나 실제 장기 모니터링은 수일, 수주, 수개월 단위로 이루어질 수 있다. PPG sensor position, skin condition, vascular tone, medication, activity level, circadian rhythm 변화가 누적될 때 transfer model이 얼마나 오래 유지되는지는 아직 검증되지 않았다.

넷째, disease classification label은 ICD-9 code에 기반한다. ICD-9 code는 clinical diagnosis label로 유용하지만, waveform segment 단위의 disease state나 acute physiological state를 직접 반영하지는 않을 수 있다. 예를 들어 AF patient라 하더라도 특정 PPG segment에서 실제 AF episode가 발생 중인지, heart rate가 정상인지에 따라 waveform morphology가 다를 수 있다. 논문은 AF가 발생하면 pulse interval이 불규칙해진다고 설명하지만, segment-level rhythm annotation을 사용했는지는 제공된 텍스트에서 명확하지 않다.

다섯째, normal, AF, CA 세 group만 다룬다. 실제 CVD는 hypertension, heart failure, cardiomyopathy, peripheral arterial occlusive disease 등 매우 다양하다. 논문도 future work에서 더 많은 disease population을 포함하겠다고 언급한다. 따라서 현재 모델은 모든 cardiovascular condition에 일반화된다고 보기 어렵다.

여섯째, PPG signal quality와 motion artifact 문제는 상대적으로 깊이 다뤄지지 않는다. 이 논문은 preprocessing에서 filtering, flat line, flat peak, skewness, autocorrelation threshold를 사용하지만, ambulatory wearable 환경에서 큰 문제가 되는 motion artifact를 별도의 quality-aware model로 다루지는 않는다. 이전 논문들처럼 SQA를 명시적으로 BP prediction 앞에 넣는 구조와 비교하면, 실제 움직임이 많은 환경에서의 robust performance는 추가 검증이 필요하다.

마지막으로, 기존 연구와의 비교는 protocol 차이 때문에 제한적으로 해석해야 한다. 논문도 raw data, experimental conditions, evaluation metrics, calibration 정보가 서로 달라 직접 비교가 어렵다고 인정한다. 따라서 Table IV에서 제안 모델이 우수하게 보이더라도, 동일 조건의 benchmark에서 우월성이 엄밀히 증명되었다고 단정하기는 어렵다.

## 6. 결론

이 논문은 single-channel PPG 기반 BP estimation에서 cardiovascular status stratification과 transfer learning 기반 personalization을 결합한 refined modeling strategy를 제안하였다. 먼저 PPG morphological features와 demographic information을 이용해 normal, AF, CA 세 집단을 분류하고, 이후 각 group에 대해 BiLSTM-At 기반 BP prediction model을 학습한다. 마지막으로 target patient의 소량 데이터를 이용해 transfer learning으로 마지막 layer를 fine-tuning하여 individual-specific model을 만든다.

Disease classification에서는 Random Forest가 SVM보다 우수했으며, RF classifier는 92.19%의 accuracy를 달성하였다. Age와 gender를 포함한 demographic information은 classification 성능을 10.2% 향상시켰고, age는 RF variable importance score에서 가장 큰 contribution을 보였다. 이는 CVD screening에서 PPG morphology와 demographic factor를 함께 고려하는 것이 효과적임을 보여준다.

BP estimation에서는 universal model의 성능이 낮았지만, transfer learning 기반 personalized model은 크게 향상된 성능을 보였다. Transfer model의 MAE는 normal subjects에서 SBP 2.815 mmHg, DBP 1.876 mmHg, AF subjects에서 SBP 3.024 mmHg, DBP 1.334 mmHg, CA subjects에서 SBP 4.444 mmHg, DBP 2.549 mmHg였다. 세 group 모두 AAMI와 BHS Class A criteria를 만족한다고 보고되었다.

이 연구의 중요한 의미는 single-channel PPG만으로도 질환군별 stratification과 개인화를 결합하면 long-term BP trend monitoring이 가능하다는 점을 보여준 것이다. 특히 wearable device에서 ECG 없이 PPG sensor만으로 BP를 추정하려는 방향에 실용적인 기여를 한다. 또한 transfer learning을 통해 많은 개인별 training data 없이도 성능을 개선할 수 있다는 점은 real-world deployment에서 중요한 장점이다.

다만 실제 적용을 위해서는 더 긴 기간의 validation, 더 다양한 disease population, calibration protocol의 구체화, ambulatory motion artifact 환경에서의 robustness 검증, wearable device용 model compression과 real-time deployment 평가가 필요하다. 논문도 future work로 더 많은 disease population, 더 긴 기간의 평가, wearable device deployment를 위한 model compression을 제시한다.

종합하면, 이 논문은 PPG 기반 cuffless BP estimation 연구에서 단순 regression model을 넘어, cardiovascular status-aware modeling과 personalized transfer learning의 필요성을 강조한 연구이다. 질환별 PPG morphology 차이를 모델 설계에 반영했다는 점에서 학술적 가치가 있으며, 향후 wearable long-term BP monitoring과 CVD early warning system으로 확장될 가능성이 있다.
