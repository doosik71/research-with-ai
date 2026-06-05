# An Estimation Method of Continuous Non-Invasive Arterial Blood Pressure Waveform Using Photoplethysmography: A U-Net Architecture-Based Approach

* **저자**: Tasbiraha Athaya, Sunwoong Choi
* **발표연도**: 2021

## 1. 논문 개요

이 논문은 fingertip photoplethysmogram, 즉 손가락 PPG 신호만을 입력으로 사용하여 arterial blood pressure, 즉 ABP waveform을 연속적이고 비침습적으로 추정하는 U-Net 기반 deep learning 방법을 제안한다. 기존의 cuff 기반 혈압 측정 방식은 널리 사용되지만 불연속적이고, 측정 과정에서 팔을 압박해야 하므로 장시간 또는 연속 모니터링에 적합하지 않다. 반면 invasive ABP 측정은 arterial catheter를 삽입하여 beat-to-beat 혈압 파형을 직접 측정할 수 있으므로 임상적으로 gold standard에 가깝지만, 감염과 합병증 위험이 있어 일반적인 건강 관리나 wearable device 환경에는 적합하지 않다.

이 연구의 핵심 목표는 invasive ABP waveform을 직접 측정하지 않고, 비침습적으로 쉽게 획득할 수 있는 PPG 신호만으로 ABP waveform 전체를 추정하는 것이다. 기존 연구들은 PPG에서 handcrafted feature를 추출하여 SBP와 DBP만 예측하거나, PPG와 ECG를 함께 사용하여 pulse transit time, 즉 PTT를 계산하는 방식이 많았다. 그러나 이 방식들은 feature selection 문제, ECG 장비와 PPG 장비 간 동기화 문제, 개별 사용자 calibration 문제를 동반한다. 이 논문은 이러한 복잡성을 줄이고자 raw PPG window를 직접 deep learning model에 입력하여 normalized ABP waveform을 출력하도록 설계했다.

연구 문제는 단순히 “SBP와 DBP 값을 맞출 수 있는가”가 아니라, “PPG signal에서 ABP waveform 자체를 복원할 수 있는가”이다. ABP waveform은 SBP와 DBP뿐 아니라 mean arterial pressure, 즉 MAP, waveform morphology, arterial stiffness, cardiac output, stroke volume 등 추가적인 생리학적 정보를 포함할 수 있다. 따라서 ABP waveform을 추정하는 접근은 단일 혈압값 회귀보다 임상적 활용 가능성이 크다.

저자들은 MIMIC 및 MIMIC-III waveform database에서 PPG와 ABP가 동시에 존재하는 100명 subject 데이터를 사용했다. 제안한 modified U-Net은 256-sample PPG window를 입력받아 같은 길이의 ABP waveform window를 출력한다. 실험 결과, 예측 ABP waveform과 reference ABP waveform 사이의 평균 Pearson correlation coefficient는 0.993으로 매우 높았고, SBP, DBP, MAP의 MAE는 각각 3.68 mmHg, 1.97 mmHg, 2.17 mmHg로 보고되었다. 또한 AAMI 기준과 BHS Grade A 기준을 만족한다고 제시하였다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 PPG waveform과 ABP waveform 사이에 높은 형태적 유사성이 존재한다는 점이다. PPG는 손가락 말초 혈관의 blood volume 변화를 광학적으로 측정한 신호이고, ABP는 동맥 내 압력 변화를 직접 측정한 신호이다. 두 신호는 측정 위치와 물리량이 다르지만, 같은 cardiac cycle에 의해 생성되므로 systolic peak, diastolic phase, dicrotic notch 등 유사한 시간적 구조를 공유한다. 논문은 기존 연구들을 근거로 두 신호의 time-domain 및 frequency-domain similarity가 높고, 일부 연구에서 평균 Pearson correlation coefficient가 0.9 이상이었다고 설명한다.

이러한 유사성을 바탕으로 저자들은 PPG-to-ABP mapping을 signal-to-signal translation 문제로 본다. 즉, 입력은 1차원 PPG signal segment이고 출력은 동일한 시간 길이의 1차원 ABP signal segment이다. 이때 U-Net은 encoder-decoder 구조를 갖고 있어 입력 신호의 local pattern과 global context를 압축한 뒤, 다시 원래 길이의 신호로 복원하는 데 적합하다. 특히 skip connection은 contracting path에서 학습된 세부 정보를 expansive path로 전달하므로, waveform reconstruction에서 중요한 fine-grained temporal information을 보존할 수 있다.

기존 연구와의 중요한 차별점은 다음과 같다. 첫째, 이 방법은 ECG나 seismocardiogram 같은 추가 생체신호가 필요 없고 PPG만 사용한다. 둘째, PTT, PAT, morphology feature 등 handcrafted feature를 계산하지 않는다. 셋째, beat segmentation을 수행하지 않고 fixed-length window 단위로 학습한다. 넷째, SBP와 DBP 값을 직접 회귀하는 것이 아니라 ABP waveform을 먼저 추정하고, 이후 peak-valley detection을 통해 SBP와 DBP를 계산한다. 이 방식은 혈압의 점 추정값뿐 아니라 continuous waveform을 제공할 수 있다는 점에서 임상 정보량이 더 크다.

## 3. 상세 방법 설명

논문에서 제안한 전체 pipeline은 data collection, preprocessing, modified U-Net training, ABP waveform estimation, SBP/DBP/MAP calculation, performance evaluation으로 구성된다. 전체 구조는 원문 Figure 4의 flow chart로 요약되어 있으며, 핵심은 PPG와 ABP 신호를 정제하고 phase matching한 후, PPG window를 입력으로 ABP window를 출력하는 U-Net을 학습하는 것이다.

### 3.1 데이터 수집

사용 데이터는 MIMIC database와 MIMIC-III waveform database에서 가져왔다. 두 데이터베이스는 ICU 환자에서 수집된 다양한 생체신호를 포함하며, ABP와 fingertip PPG가 동시에 기록된 waveform을 제공한다. 샘플링 주파수는 125 Hz이다.

논문은 MIMIC database에서 45명, MIMIC-III waveform database에서 55명의 subject를 선택하여 총 100명의 recording을 사용했다. 각 subject는 PPG와 ABP 신호가 모두 존재해야 했다. 최종 preprocessing 이후 약 195시간의 데이터가 확보되었으며, subject당 평균 약 3.4시간의 데이터가 사용되었다고 보고한다.

이 데이터는 ICU 환자 기반 데이터이므로 일반 건강인보다 혈압 범위가 넓을 수 있다. Figure 7은 SBP와 DBP의 분포를 보여주며, 논문은 normal, pre-hypertension, stage-1 hypertension, stage-2 hypertension에 해당하는 값들이 모두 포함되어 있다고 설명한다. 이는 모델이 다양한 혈압 범위를 학습할 수 있다는 장점이 있지만, 동시에 ICU 환경 특성이 일반 wearable 환경과 다를 수 있다는 점도 고려해야 한다.

### 3.2 Preprocessing

Preprocessing은 이 연구에서 매우 중요한 단계로 제시된다. 먼저 PPG signal은 Equiripple FIR filter를 사용하여 0.5–8 Hz 대역으로 필터링되었다. 0.5 Hz 이하 성분은 baseline wandering으로 간주했고, 8 Hz 이상 성분은 high-frequency noise로 간주했다.

그 다음 PPG와 ABP 신호를 sequential window로 나누었다. 원래 window size는 350 samples이고, overlap은 100 samples이다. overlap을 적용한 이유는 window boundary에서 중요한 waveform 정보가 누락되는 것을 줄이기 위함이다. 이후 artifact가 포함된 window는 이전 연구에서 사용한 machine learning 기반 artifact detection model을 통해 제거했다. 이 artifact detection model의 구체적 구조와 성능은 본문에 상세히 제시되어 있지 않고, 저자들의 다른 연구를 인용하고 있다.

PPG와 ABP는 같은 cardiac event에서 유래하지만, 측정 위치가 다르기 때문에 phase difference가 발생한다. ABP는 brachial artery에서 측정되고, PPG는 digital artery에서 측정되므로 혈류가 이동하는 시간 차이가 존재한다. 이 phase mismatch를 해결하기 위해 cross-correlation을 사용했다. PPG window를 고정한 뒤 ABP window와의 cross-correlation을 계산하고, cross-correlation이 최대가 되는 지점을 time lag로 추정한다. 이후 ABP window를 해당 lag만큼 shift하여 PPG와 phase matching한다.

최종 U-Net 입력 크기가 256 samples이므로, 256 samples보다 짧은 window는 제외하고, 긴 window는 256 samples로 trimming했다. 125 Hz sampling rate에서 256 samples는 2.048초에 해당한다. 즉, 모델은 약 2초 길이의 PPG segment를 보고 같은 길이의 ABP waveform을 예측한다.

마지막으로 PPG와 ABP window는 min-max normalization을 수행했다.

$$
x_{normalized}(i)=\frac{x_i-x_{min}}{x_{max}-x_{min}}
$$

여기서 $x_i$는 $i$번째 signal window이고, $x_{max}$와 $x_{min}$은 전체 windowed signal에서의 최대값과 최소값이다. 학습 시에는 normalized PPG를 input으로, normalized ABP를 target으로 사용한다. 추론 후에는 저장해 둔 ABP의 최대값과 최소값을 사용하여 output waveform을 de-normalization한다.

### 3.3 Modified U-Net architecture

제안 모델은 원래 biomedical image segmentation에서 사용된 U-Net을 1D signal-to-signal regression 문제에 맞게 수정한 구조이다. 입력은 256 samples 길이의 1D PPG signal이고, 출력은 256 samples 길이의 1D ABP signal이다. 원래 U-Net은 2D convolution을 사용하지만, 이 연구에서는 waveform 처리를 위해 1D convolution을 사용한다.

모델은 크게 contracting path, bottleneck path, expansive path로 구성된다. Contracting path는 encoder 역할을 하며, PPG signal의 local waveform pattern을 점차 압축하면서 더 많은 feature channel로 표현한다. 각 contraction block은 두 개의 $3 \times 1$ convolution layer, 각 convolution 뒤의 Leaky ReLU activation, 그리고 $2 \times 1$ max pooling layer로 구성된다. Leaky ReLU는 일반 ReLU에서 발생할 수 있는 dying ReLU 문제를 줄이기 위해 사용되었다.

Contracting path에서는 block을 지날수록 feature channel 수가 두 배가 된다. 이는 신호 길이가 줄어드는 대신 더 많은 feature representation을 사용하여 복잡한 waveform 구조를 학습하기 위한 설계이다. Bottleneck path는 encoder와 decoder 사이의 latent feature space 역할을 하며, 두 개의 $3 \times 1$ convolution layer와 $2 \times 1$ up-sampling layer를 사용한다. Contracting path의 마지막 부분과 bottleneck path에는 50% dropout이 적용되었다. 이는 overfitting을 줄이기 위한 regularization이다.

Expansive path는 decoder 역할을 한다. 각 expansion block은 $2 \times 1$ convolution layer, 두 개의 $3 \times 1$ convolution layer, Leaky ReLU activation, 그리고 $2 \times 1$ up-sampling layer를 포함한다. Expansion path에서는 feature channel 수가 점차 절반으로 줄어들며, signal length는 다시 원래 크기로 복원된다. 중요한 점은 각 expansion block에서 대응되는 contracting path의 feature map을 concatenation한다는 것이다. 이 skip connection은 encoder에서 학습한 세부 waveform 정보를 decoder가 활용할 수 있게 하여, ABP waveform의 세밀한 형태를 복원하는 데 도움을 준다.

마지막 expansion path에서는 두 개의 추가 $3 \times 1$ convolution layer를 사용하여 64개의 feature vector를 입력과 동일한 차원의 ABP waveform으로 mapping한다. 결과적으로 이 네트워크는 256-sample PPG window를 256-sample ABP window로 변환하는 end-to-end regression model이다.

### 3.4 학습 목표와 추론 절차

모델 학습에서는 Adam optimizer와 mean squared error, 즉 MSE loss가 사용되었다. MSE는 예측 ABP waveform과 reference ABP waveform 사이의 sample-wise squared error를 최소화한다. 논문에서 loss function을 명시적 수식으로 쓰지는 않았지만, 설명에 따르면 학습 목표는 다음과 같이 이해할 수 있다.

$$
\mathcal{L}_{MSE}=\frac{1}{N}\sum_{i=1}^{N}\left(\hat{y}_i-y_i\right)^2
$$

여기서 $y_i$는 reference ABP waveform의 $i$번째 sample이고, $\hat{y}_i$는 U-Net이 예측한 ABP waveform의 $i$번째 sample이다. $N$은 window 내 sample 수, 즉 256이다. 이 손실을 줄이면 waveform 전체의 point-wise reconstruction error가 감소한다.

데이터는 training 70%, validation 15%, testing 15%로 나누었다. 논문은 세 dataset이 완전히 분리되었다고 설명하지만, subject-level split인지 window-level split인지는 원문 텍스트만으로 명확히 확인하기 어렵다. 이 점은 성능 해석에서 중요한 요소이다. 만약 동일 subject의 window가 train과 test에 모두 포함된다면, 새로운 subject에 대한 일반화 성능이 과대평가될 수 있다.

학습 hyperparameter는 learning rate $10^{-4}$, batch size 4이다. Early stopping은 validation loss가 5 epoch 연속 개선되지 않을 때 적용되었고, 최종적으로 51 epoch에서 학습이 중단되었다. 각 epoch 중 validation loss가 가장 낮은 model이 자동 저장되고, 학습 종료 후 그 model이 최종 model로 사용되었다. 학습 환경은 NVIDIA GTX 1080 Ti 10 GB GPU와 257 GB system memory를 갖춘 GPU server이며, 구현은 Python으로 수행되었다.

추론 단계에서는 normalized PPG window를 입력하고, U-Net output으로 normalized ABP waveform을 얻는다. 이후 preprocessing 단계에서 저장한 ABP maximum 및 minimum 값을 사용해 de-normalization한다. 이렇게 얻은 ABP waveform에서 standard peak and valley detection algorithm을 적용하여 SBP와 DBP를 계산한다. 각 window에서 peak들의 평균을 SBP, valley들의 평균을 DBP로 간주한다. MAP는 reference 및 predicted ABP window 각각의 arithmetic mean으로 계산했다.

## 4. 실험 및 결과

### 4.1 ABP waveform prediction 성능

제안 모델의 핵심 결과는 predicted ABP waveform이 reference invasive ABP waveform과 매우 높은 상관을 보였다는 점이다. Figure 9는 한 subject의 input PPG, reference ABP, predicted ABP를 보여주며, predicted ABP waveform이 reference waveform의 형태를 상당히 잘 따라가는 것으로 제시된다.

ABP waveform 전체의 유사성 평가는 Pearson correlation coefficient $r$로 수행되었다. 전체 test window에서 $r$ 분포를 나타낸 Figure 10에 따르면 대부분의 값이 0.9–1.0 사이에 위치한다. Table 1에 따르면 평균 $r$은 0.993이고, minimum $r$은 0.262, maximum $r$은 0.999이다. 25th percentile은 0.989, 75th percentile은 0.996이다. 평균 계산 전 Fisher-Z transformation을 수행하고, 평균 후 다시 retransformation을 수행했다고 설명한다.

이 결과는 대부분의 window에서 waveform morphology를 매우 잘 복원했다는 것을 의미한다. 다만 minimum $r=0.262$인 window가 존재한다는 점은 일부 window에서는 예측이 실패하거나 artifact, abnormal waveform, phase mismatch 등이 영향을 주었을 가능성을 시사한다. 논문은 이러한 low-correlation case에 대한 별도 정성 분석은 제공하지 않는다.

### 4.2 SBP, DBP, MAP 추정 성능

ABP waveform에서 계산한 SBP, DBP, MAP에 대해 MAE, STD, RMSE, Pearson correlation coefficient가 평가되었다. Table 2에 따르면 성능은 다음과 같다.

SBP의 MAE는 3.68 mmHg, STD는 4.42 mmHg, RMSE는 5.75 mmHg, correlation coefficient는 0.976이다. DBP의 MAE는 1.97 mmHg, STD는 2.92 mmHg, RMSE는 3.52 mmHg, correlation coefficient는 0.970이다. MAP의 MAE는 2.17 mmHg, STD는 3.06 mmHg, RMSE는 3.75 mmHg, correlation coefficient는 0.976이다.

DBP가 SBP보다 더 낮은 error를 보였는데, 논문은 SBP prediction error의 deviation이 DBP보다 거의 두 배 정도 크다고 설명한다. Figure 11의 error histogram에서도 SBP error 분포가 DBP보다 더 넓게 퍼져 있음을 확인할 수 있다. 이는 SBP가 DBP보다 waveform peak의 sharpness, artifact, phase alignment, amplitude scaling에 더 민감하기 때문일 수 있다. 다만 논문은 이 원인을 상세히 분석하지는 않는다.

### 4.3 혈압 범위별 성능

Figure 12는 normal, pre-hypertension, stage-1 hypertension, stage-2 hypertension의 네 혈압 범위에서 SBP와 DBP prediction accuracy를 분석한다. Normal range에서는 성능이 가장 좋으며, SBP prediction rate는 97.09%, DBP는 99.04%로 제시된다. Pre-hypertension range에서는 SBP 예측이 DBP보다 더 좋았다고 설명한다.

Stage-1 hypertension에서는 상대적으로 성능이 낮았다. 이 범위에서 35% 이상의 SBP 값과 22% 이상의 DBP 값이 10 mmHg 이상 deviation을 보였으며, 많은 stage-1 hypertension 값이 pre-hypertension 범위로 잘못 식별되었다고 설명한다. 반면 stage-2 hypertension에서는 비교적 좋은 성능을 보였다고 한다. 논문은 stage-2 hypertension subject가 stage-1 hypertension subject보다 dataset에 더 많이 포함되어 있었기 때문에 stage-2 성능이 더 좋았다고 discussion에서 설명한다.

이 결과는 모델 성능이 단순히 overall MAE만으로 평가될 수 없음을 보여준다. 실제 임상 활용에서는 정상 혈압보다 hypertension boundary 근처에서의 정확성이 매우 중요하다. 특히 stage-1 hypertension을 pre-hypertension으로 낮게 예측하는 것은 clinical screening에서 문제가 될 수 있다.

### 4.4 Regression plot 및 Bland-Altman plot

Figure 13은 SBP, DBP, MAP의 predicted value와 actual value 사이의 linear regression plot을 보여준다. 세 지표 모두 reference와 prediction 사이에 선형적 관계가 나타난다고 설명한다. 이는 모델이 혈압 변화 경향을 상당히 잘 추적한다는 의미이다.

Figure 14는 SBP와 DBP에 대한 Bland-Altman plot이다. x축은 actual BP와 predicted BP의 평균이고, y축은 두 값의 차이이다. 논문은 대부분의 SBP와 DBP error가 $-5$에서 $+5$ mmHg 사이에 위치한다고 설명한다. Bland-Altman plot은 두 측정 방식 간 agreement를 평가하는 데 사용되므로, 이 결과는 제안 모델이 reference ABP 기반 혈압값과 비교적 잘 일치함을 보여준다.

다만 Bland-Altman plot에서 outlier 또는 큰 error case가 어느 혈압 범위에서 발생하는지, artifact와 관련이 있는지, subject별 편향이 있는지에 대한 세부 분석은 충분히 제시되지 않는다.

### 4.5 AAMI 및 BHS 기준 충족 여부

논문은 제안 모델의 SBP 및 DBP 추정 결과가 AAMI 기준을 만족한다고 보고한다. AAMI 기준에서는 일반적으로 mean error 또는 mean difference가 5 mmHg 이하이고, standard deviation이 8 mmHg 이하이어야 한다. Table 3에서 제안 모델은 100명 subject 기준으로 SBP MAE 3.68 mmHg, STD 4.42 mmHg, DBP MAE 1.97 mmHg, STD 2.92 mmHg를 보인다. 따라서 error magnitude 측면에서 AAMI 기준을 만족한다.

BHS grading standard와 비교한 Table 4에서는 SBP의 cumulative error 비율이 5 mmHg 이하 76.21%, 10 mmHg 이하 93.66%, 15 mmHg 이하 97.71%로 나타났다. DBP는 각각 93.51%, 98.70%, 99.46%이다. BHS Grade A 기준은 5 mmHg 이하 60%, 10 mmHg 이하 85%, 15 mmHg 이하 95%이므로, SBP와 DBP 모두 Grade A에 해당한다고 제시된다.

여기서 주의할 점은 논문이 “AAMI와 BHS 기준을 만족한다”고 표현하지만, 실제 의료기기 validation standard는 단순 error threshold뿐 아니라 대상자 구성, 측정 절차, reference device, 반복 측정 조건, 독립 검증 설계 등 다양한 요구사항을 포함한다. 따라서 본 논문의 결과는 error metric 관점에서 기준을 만족한 것으로 해석하는 것이 적절하며, 완전한 의료기기 검증을 완료했다고 보기는 어렵다.

### 4.6 기존 연구와의 비교

Table 5는 제안 방법과 기존 연구를 비교한다. 제안 방법은 raw PPG만 사용하며, MIMIC I 및 MIMIC III waveform 데이터를 기반으로 testing data 29.3시간을 사용했다. MAE는 SBP 3.68 mmHg, DBP 1.97 mmHg로 보고되며, STD는 SBP 4.42 mmHg, DBP 2.92 mmHg이다.

논문은 ANN 기반 feature method, Random Forest 기반 feature method, CNN 기반 raw PPG method, PTT 기반 SCG+PPG method, LSTM 기반 PPG+ECG method, ResNet-GRU 기반 raw PPG method, CNN-LSTM 기반 raw PPG method와 비교한다. 저자들은 제안 방법이 PPG만 사용하고 beat segmentation이 필요 없으며, SBP와 DBP MAE 및 STD 측면에서 경쟁력 있는 결과를 보인다고 주장한다.

특히 Wang et al.의 ANN 연구는 유사한 MIMIC 데이터를 사용했으나 subject 수와 signal availability 설명에서 혼란스러운 부분이 있다고 지적한다. Li et al.의 LSTM 연구는 ECG와 PPG 두 신호를 필요로 하고 handcrafted feature를 사용한다. Slapničar et al.의 ResNet-GRU 연구는 대규모 MIMIC III waveform을 사용했지만 AAMI error range를 만족하지 못했다고 설명한다. 반면 제안 방법은 단일 PPG raw waveform만 사용하면서 높은 성능을 달성했다는 점을 강조한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 PPG-to-ABP waveform estimation을 end-to-end deep learning 문제로 정의했다는 점이다. 기존의 많은 연구는 PPG feature를 추출하고 SBP와 DBP를 직접 회귀하는 방식이었다. 그러나 이 논문은 waveform 전체를 복원하고, 그 waveform에서 SBP, DBP, MAP를 계산한다. 이는 임상적으로 더 풍부한 정보를 제공할 수 있으며, arterial stiffness, cardiac output, stroke volume 같은 추가 분석 가능성을 열어준다.

두 번째 강점은 PPG만 사용한다는 점이다. ECG, SCG, 다중 PPG 센서가 필요하지 않으므로 wearable device나 smartphone 기반 시스템으로 확장하기 쉽다. 또한 beat segmentation이나 handcrafted feature extraction이 필요 없기 때문에 pipeline이 상대적으로 단순하다. Feature engineering에 의존하지 않는다는 점은 subject나 측정 조건에 따라 feature detection이 불안정해지는 문제를 줄일 수 있다.

세 번째 강점은 U-Net 구조의 적절한 활용이다. U-Net은 encoder-decoder 구조와 skip connection을 통해 signal reconstruction에 적합하다. Contracting path는 PPG의 temporal pattern을 압축하고, expansive path는 ABP waveform으로 복원하며, skip connection은 waveform 세부 구조를 유지한다. 이는 단순 CNN regression보다 waveform morphology 보존에 유리할 수 있다.

네 번째 강점은 성능이 여러 지표에서 높게 보고되었다는 점이다. ABP waveform의 평균 correlation coefficient가 0.993으로 매우 높고, SBP와 DBP MAE도 각각 3.68 mmHg, 1.97 mmHg로 낮다. BHS Grade A 및 AAMI error range를 만족한다고 보고한 점도 실험적 강점이다.

그러나 한계도 명확하다. 첫째, subject 수가 100명으로 비교적 제한적이다. 전체 시간은 195시간으로 크지만, deep learning 모델의 실제 일반화 성능은 subject diversity에 크게 의존한다. 특히 ICU 데이터는 일반 사용자 wearable 환경과 생리적 조건이 다를 수 있다.

둘째, training, validation, test split이 subject-independent 방식인지 명확하지 않다. 논문은 dataset이 완전히 분리되었다고 설명하지만, 같은 subject의 window가 train과 test에 동시에 들어가지 않았는지는 원문 텍스트만으로 확정하기 어렵다. 만약 window-level random split이라면 모델이 subject-specific waveform pattern을 학습했을 가능성이 있으며, 신규 subject에 대한 성능이 낮아질 수 있다.

셋째, phase matching 과정이 실제 deployment에서 문제될 수 있다. 연구에서는 reference ABP와 PPG 사이의 cross-correlation을 사용하여 ABP window를 shift했다. 하지만 실제 wearable device에서는 reference ABP가 없으므로 학습 시 정렬된 target이 필요하고, 추론 시에는 phase-matched ABP를 output으로 얻는다고 해도 brachial artery ABP와 digital artery PPG 사이의 생리적 지연이 어떻게 임상적으로 해석될지 추가 검토가 필요하다.

넷째, artifact robustness가 제한적으로 검증되었다. 논문은 artifact window를 machine learning model로 제거한 뒤 학습했다. 이는 성능을 높이는 데 도움이 되지만, 실제 wearable 환경에서는 motion artifact, finger pressure 변화, skin thickness, skin color, subcutaneous tissue 특성 등이 강하게 작용한다. 저자들도 PPG가 movement, finger pressure, skin coloration에 민감하다고 언급한다. 따라서 artifact가 많은 현실 환경에서의 성능은 별도 검증이 필요하다.

다섯째, stage-1 hypertension 구간 성능이 상대적으로 낮다. 논문은 stage-1 hypertension 값들이 pre-hypertension으로 잘못 예측되는 경우가 있다고 설명한다. 혈압 모니터링에서 정상과 고혈압 경계 구간은 임상적으로 중요하므로, 이 구간의 성능 저하는 실제 screening 또는 monitoring 적용에서 중요한 한계가 될 수 있다.

여섯째, personalized calibration의 필요성이 남아 있다. 논문은 새로운 subject에서 정확도를 높이기 위해 personalized calibration technique이 적용될 수 있다고 discussion에서 언급한다. 이는 완전한 calibration-free 방법으로 보기에는 추가 검증이 필요하다는 의미이기도 하다.

## 6. 결론

이 논문은 fingertip PPG signal만을 사용하여 continuous non-invasive ABP waveform을 추정하는 modified U-Net 기반 방법을 제안했다. 제안 모델은 256-sample PPG window를 입력받아 같은 길이의 ABP waveform을 출력하며, waveform에서 SBP, DBP, MAP를 계산한다. 모델은 MIMIC 및 MIMIC-III waveform database의 100명 subject 데이터를 사용하여 학습 및 평가되었고, predicted ABP waveform은 reference invasive ABP waveform과 평균 Pearson correlation coefficient 0.993의 높은 유사성을 보였다.

혈압값 예측에서도 SBP MAE 3.68 mmHg, DBP MAE 1.97 mmHg, MAP MAE 2.17 mmHg로 우수한 결과를 보였으며, 논문은 AAMI 기준과 BHS Grade A 기준을 만족한다고 보고한다. 특히 ECG나 다른 sensor 없이 PPG raw waveform만 사용하고, beat segmentation이나 handcrafted feature extraction을 요구하지 않는다는 점은 wearable device 및 smartphone 기반 혈압 모니터링에 유리한 특징이다.

이 연구는 cuffless, continuous, non-invasive BP monitoring 분야에서 중요한 방향을 제시한다. 단순한 SBP/DBP 회귀가 아니라 ABP waveform 자체를 복원함으로써, 혈압 수치뿐 아니라 waveform morphology에 담긴 추가 생리 정보를 활용할 가능성을 보여주었다. 다만 실제 적용을 위해서는 subject-independent validation, 더 큰 규모의 외부 데이터셋 검증, motion artifact가 많은 현실 환경에서의 평가, stage-1 hypertension과 같은 임상 경계 구간의 성능 개선, personalized calibration 여부에 대한 체계적 검토가 필요하다.

종합하면, 이 논문은 PPG 기반 혈압 추정 연구를 waveform reconstruction 문제로 확장한 의미 있는 연구이며, U-Net 구조가 physiological signal translation에도 효과적으로 사용될 수 있음을 보여준다. 그러나 의료기기 수준의 신뢰성을 확보하기 위해서는 더 엄격하고 다양한 조건의 후속 검증이 필수적이다.
