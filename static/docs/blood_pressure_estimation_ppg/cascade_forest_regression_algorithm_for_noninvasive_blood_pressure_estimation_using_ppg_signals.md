# Cascade Forest Regression Algorithm for Non-Invasive Blood Pressure Estimation Using PPG Signals

* **저자**: Gengjia Zhang, Siho Shin, Jaehyo Jung
* **발표연도**: 2023

## 1. 논문 개요

이 논문은 photoplethysmogram, 즉 PPG 신호만을 이용하여 비침습적으로 blood pressure, 즉 혈압을 추정하는 방법을 제안한다. 핵심 모델은 cascade forest regression, CFR이며, 이는 deep forest 또는 gcForest 계열의 cascade structure를 회귀 문제에 맞게 변형한 모델이다. 논문은 deep learning 기반 혈압 추정 모델들이 높은 성능을 보일 수는 있지만, 많은 hyperparameter tuning이 필요하고, 구조가 복잡하며, 학습 과정에서 사용자의 개입과 human error에 취약하다는 문제의식에서 출발한다.

연구 문제는 wearable device에서 얻을 수 있는 PPG 신호로부터 systolic blood pressure, SBP와 diastolic blood pressure, DBP를 정확하고 안정적으로 추정하는 것이다. 기존 cuff-based oscillometric blood pressure measurement는 비교적 정확하지만 cuff 압박으로 인한 불편함이 있고, 반복 측정 사이에 휴식 시간이 필요하여 일상생활에서 continuous monitoring이 어렵다. invasive arterial catheterization은 정확하지만 외상과 감염 위험이 있어 장기적 또는 원격 모니터링에 적합하지 않다. 따라서 단일 PPG 기반 cuffless BP estimation은 smartwatch와 같은 wearable device에서 지속적 건강 모니터링을 가능하게 할 수 있다는 점에서 중요하다.

논문은 UCI Machine Learning Repository에 공개된 cuffless blood pressure estimation dataset을 사용한다. 이 데이터셋은 MIMIC-II waveform database를 filtering 및 processing한 버전이며, PPG와 invasive ambulatory blood pressure, ABP 신호를 포함한다. 저자들은 PPG와 ABP 신호를 5초 segment로 나누고, PPG에서 heart rate variability, HRV 관련 time-domain feature를 추출하여 CFR 모델의 입력으로 사용한다. ABP 신호에서는 peak와 valley를 검출하여 각각 SBP와 DBP reference 값을 계산한다.

논문이 주장하는 최종 성능은 매우 높다. 본문 Table 3 기준으로 CFR 모델은 DBP에서 $R^2=0.926$, MSE 3.033 mmHg, MAE 1.760 mmHg를 달성했고, SBP에서 $R^2=0.948$, MSE 4.625 mmHg, MAE 2.896 mmHg를 달성했다. 또한 AAMI 기준에서 ME와 STD가 모두 허용 범위 안에 들었고, BHS 기준에서도 SBP와 DBP 모두 Grade A를 달성했다고 보고한다. 다만 abstract에서는 “MAE 1.760 and 2.896 mmHg for SBP and DBP”라고 되어 있어, 본문 Table 3의 “DBP 1.760, SBP 2.896”과 순서가 맞지 않는다. 전체 결과표와 Discussion의 수치를 보면 DBP MAE가 1.760 mmHg, SBP MAE가 2.896 mmHg인 해석이 더 일관적이다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 deep neural network 대신 cascade forest regression을 사용하여 PPG 기반 혈압 추정의 nonlinear relationship을 학습하는 것이다. 저자들은 PPG에서 추출한 HRV feature와 혈압 사이의 관계가 명확히 선형적이지 않으며, 단순 linear model로는 충분한 성능을 얻기 어렵다고 본다. 반면 deep learning 모델은 비선형 관계를 잘 학습할 수 있지만, 구조가 복잡하고 hyperparameter 설정에 민감하며, 많은 데이터와 계산 자원이 필요하다. CFR은 이러한 두 접근 사이에서 nonlinear fitting 능력과 비교적 낮은 tuning 부담을 동시에 얻으려는 대안이다.

CFR의 직관은 layer-by-layer feature enhancement이다. 첫 번째 cascade layer는 입력 feature를 여러 regression forest에 넣어 예측값 또는 새 feature vector를 생성한다. 이 새 feature vector는 원래 입력 feature와 결합되어 다음 layer의 입력이 된다. 이런 방식으로 각 layer는 이전 layer가 만든 정보를 바탕으로 더 풍부한 representation을 구성한다. deep neural network의 hidden layer처럼 여러 단계의 feature transformation을 수행하지만, 각 layer가 random forest regressor와 extra-tree regressor로 구성된다는 점이 다르다.

논문의 또 다른 핵심 설계는 random forest regression, RFR과 extra-tree regression, ETR을 함께 사용하는 것이다. RFR은 bootstrap sampling을 통해 여러 decision tree를 만들고 평균 예측을 수행한다. ETR은 random forest와 유사하지만 node split이 더 무작위적이며, 일반적으로 학습이 빠르고 variance가 높은 다양한 tree를 만든다. 논문은 같은 종류의 regressor만 조합하는 것보다 RFR과 ETR을 함께 사용하는 것이 model diversity를 높이고 generalization 성능을 향상시킨다고 주장한다.

기존 ECG-PPG 기반 PTT 방식과의 차별점도 있다. PTT 또는 PAT 기반 방법은 ECG와 PPG 두 신호를 동시에 측정해야 하므로 최소 두 개 이상의 sensor가 필요하다. 이 논문은 PPG만 사용하므로 smartwatch와 같은 단일 optical sensor 기반 wearable 환경에 더 적합하다고 주장한다. 또한 raw waveform을 end-to-end deep learning에 넣는 대신, PPG에서 해석 가능한 HRV feature를 추출하여 사용한다. 따라서 feature interpretability 측면에서도 장점이 있다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

논문의 전체 파이프라인은 네 단계로 구성된다. 첫째, MIMIC-II 기반 UCI dataset에서 PPG와 ABP 신호를 수집한다. 둘째, PPG 신호의 noise를 제거하고 5초 segment로 나누는 preprocessing을 수행한다. 셋째, 각 5초 PPG segment에서 HRV 관련 time-domain feature를 추출하고, 같은 구간의 ABP에서 SBP와 DBP reference 값을 계산한다. 넷째, 추출된 feature와 SBP/DBP label을 이용해 CFR 모델을 학습하고, MAE, MSE, ME, STD, $R^2$ 등의 지표로 성능을 평가한다.

논문 Figure 1은 이 과정을 data collection, preprocessing, training process, model evaluation의 네 단계 flowchart로 제시한다. Figure 7은 proposed algorithm의 workflow를 더 구체적으로 보여주며, PPG noise removal, 5초 interval segmentation, feature extraction, CFR training, performance evaluation이 순차적으로 진행됨을 나타낸다.

### 3.2 데이터셋과 segmentation

사용된 데이터셋은 UCI Machine Learning Repository의 cuffless blood pressure estimation dataset이다. 이 데이터셋은 MIMIC-II waveform database를 기반으로 하며, ICU 환자에게서 측정된 PPG, ABP, ECG 등 physiological signal을 포함한다. 이 연구에서는 PPG와 ABP를 사용한다. 두 신호의 sampling rate는 125 Hz이다.

데이터는 `.mat` file 형식으로 저장되어 있으며, 각 cell은 4000개의 instance를 포함한다. 저자들은 각 file에서 처음 300개 instance를 무작위로 선택했다고 설명한다. 최종적으로 1200개 experimental instance를 preprocessing하였고, 이로부터 총 25,561개 segment를 만들었다. 전체 segment 중 70%, 즉 17,893개 segment가 training set으로 사용되었고, 30%, 즉 7,669개 segment가 independent test set으로 사용되었다. Training set 중 20%, 즉 3,579개 segment는 validation set으로 사용되었다.

PPG와 ABP 신호는 5초 sliding window로 나뉜다. Sampling rate가 125 Hz이므로 각 segment 길이는 625 sample이다. 논문은 PPG peak를 찾아 각 sequence가 두 heartbeat interval로 나뉘도록 했고, beat 사이의 distance interval을 기준으로 설정하여 sample 간 overlap이 매우 작다고 설명한다. 다만 “sliding window”라는 표현과 “little overlap”이라는 표현이 함께 사용되므로, 완전 non-overlapping인지 또는 heartbeat alignment 과정에서 매우 작은 overlap이 존재하는지는 제공 텍스트만으로는 다소 모호하다.

### 3.3 PPG signal preprocessing

PPG 신호는 움직임 등에 의해 baseline noise를 포함할 수 있으며, 이는 혈압 추정 성능을 낮출 수 있다. 논문은 이 noise를 줄이기 위해 high-pass filter를 설계했다고 설명한다. 여기서 cut-off frequency가 40 Hz라고 되어 있는데, 일반적인 PPG 생리 주파수 대역과 비교하면 다소 높은 값으로 보인다. 텍스트상으로는 “high-pass filter with a cut-off frequency of 40 Hz”라고 명시되어 있으므로 그대로 기술하지만, 실제로는 low-pass 또는 다른 filtering 조건의 오기일 가능성도 있다. 이 부분은 원문 그림 또는 구현 코드 확인이 필요하다.

Figure 3은 original PPG signal과 noise removal 후 PPG signal을 비교한다. 전처리의 목적은 baseline movement artifact를 줄이고 HRV feature extraction이 안정적으로 이루어지게 하는 것이다.

### 3.4 Feature extraction

이 논문은 PPG segment에서 HRV 기반 feature를 추출한다. ECG에서 HRV를 계산하는 것이 일반적이지만, PPG에서도 pulse peak 간 interval, 즉 inter-beat interval, IBI를 이용해 pulse rate variability 또는 HRV 유사 지표를 계산할 수 있다. Figure 4는 PPG 신호에서 peak, peak interval, adjacent interval difference를 계산하는 과정을 보여준다.

논문은 총 12개의 time-domain feature를 사용한다. Table 1에 따르면 feature는 BPM, SDNN, RMSSD, IBI, SDSD, SD1, SD2, S, SD, pNN20, pNN50, HR mad로 구성된다.

BPM은 heart rate를 의미하며, peak-to-peak interval의 평균인 $RR$을 이용해 계산된다.

$$
BPM = \frac{60000\ ms}{RR}
$$

SDNN은 inter-beat interval의 standard deviation이다.

$$
SDNN = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(RR_i - \overline{RR})^2}
$$

RMSSD는 successive difference의 root mean square이다.

$$
RMSSD = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(RRdiff_i)^2}
$$

IBI는 adjacent heartbeat interval의 평균이다.

$$
IBI = \frac{1}{n-1}\sum_{i=1}^{n}RR_i
$$

SDSD는 successive difference의 standard deviation이다.

$$
SDSD = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(RRdiff_i - \overline{RRdiff})^2}
$$

Poincaré plot 관련 feature인 SD1과 SD2는 interval variability의 단기 및 장기 성분을 나타낸다.

$$
SD1 = \sqrt{\frac{1}{2}SDSD^2}
$$

$$
SD2 = \sqrt{2SDNN^2 - \frac{1}{2}SDSD^2}
$$

SD는 SD1과 SD2의 ratio이며, S는 Poincaré ellipse의 area이다.

$$
SD = \frac{SD1}{SD2}
$$

$$
S = \pi \cdot SD1 \cdot SD2
$$

pNN20과 pNN50은 successive RR interval difference가 각각 20 ms 또는 50 ms보다 큰 비율을 나타낸다.

$$
pNNxx = \frac{RR_{xx}}{n}
$$

HR mad는 RR interval의 median absolute deviation이다.

$$
HRmad = median(|RR_i - median(RR)|)
$$

이 feature들은 모두 PPG waveform의 morphology 자체보다는 beat-to-beat interval variability를 반영한다. 따라서 논문은 PPG에서 얻은 HRV-like physiological parameters가 혈압 추정에 사용될 수 있음을 보이려 한다.

### 3.5 SBP와 DBP reference 계산

Reference SBP와 DBP는 ABP signal에서 계산된다. 논문은 detection peaks function을 사용하여 ABP의 peak와 valley를 검출한다. SBP는 5초 sequence 안에서 검출된 peak들의 평균값이고, DBP는 valley들의 평균값이다. 논문은 SBP peak 검출을 위해 80보다 큰 peak를, DBP valley 검출을 위해 65보다 큰 값 기준을 설정했다고 설명한다. Figure 5는 ABP signal에서 systolic blood pressure와 diastolic blood pressure feature point를 추출하는 과정을 보여준다.

이 방식은 sample-level label을 단일 peak가 아니라 segment 내 여러 beat의 평균으로 만들기 때문에 순간적 noise나 peak detection error의 영향을 일부 줄일 수 있다. 다만 segment가 5초로 짧기 때문에, heart rate가 낮은 경우 포함되는 beat 수가 제한적일 수 있다.

### 3.6 Cascade forest regressor 구조

CFR은 여러 cascade layer로 구성된다. 각 cascade layer는 네 개의 regression estimator를 포함한다. 이 네 estimator는 두 개의 random forest regressor와 두 개의 extra-tree regressor이다. 각 estimator는 입력 feature를 받아 하나의 prediction 또는 새 feature 값을 출력하고, 네 estimator의 출력이 4-dimensional feature vector를 형성한다. 이 새 feature vector는 원래의 12-dimensional HRV feature와 concatenation되어 다음 cascade layer의 입력이 된다.

초기 입력은 12-dimensional feature vector이다. 첫 번째 layer에서 네 개의 estimator가 4-dimensional feature vector를 생성한다. 이 벡터는 원래 12개 feature와 결합되어 16-dimensional feature vector가 된다. 두 번째 layer도 같은 방식으로 4-dimensional feature vector를 생성하고, 다시 원래 12개 feature와 결합하여 다음 layer에 전달된다. 마지막 layer에서는 네 estimator의 regression output 평균값을 최종 output으로 사용한다.

이 구조의 중요한 특징은 adaptive depth이다. 새로운 cascade layer가 추가될 때마다 validation set에서 전체 cascade의 성능을 평가하고, 성능 향상이 더 이상 나타나지 않으면 training을 종료한다. 논문 Figure 9에서는 R2 score를 기준으로 layer별 성능을 추적하며, 6번째 level layer에서 training이 자동 종료되었다고 설명한다. 이후 더 학습하면 overfitting이 발생할 수 있다고 해석한다.

CFR의 parameter는 scikit-learn 기반으로 구현되었다. Table 2에 따르면 forest 안의 tree 수, 즉 $n_estimators$는 100이고, 각 split의 최소 sample 수는 20이다. Split을 찾을 때 고려하는 feature 수는 sqrt 또는 log2이며, bootstrap sample을 사용하고, 병렬 처리를 위해 $n_jobs=-1$이 사용된다.

### 3.7 Random forest regression과 extra-trees regression

Random forest regression은 bootstrap sampling으로 여러 decision tree를 학습한 뒤, 각 tree의 regression output을 평균하여 최종 예측값을 얻는다. Feature vector는 모든 tree에 전달되고, 각 tree의 결과가 ensemble되어 robust한 prediction을 만든다. Random forest는 overfitting을 줄이고 nonlinear relationship을 학습하는 데 유용하다.

Extra-trees regression은 random forest와 유사하지만 split point를 더 무작위적으로 선택한다. 논문은 extra-trees가 일반 decision tree보다 구축 속도가 빠르고 prediction variance가 더 크다고 설명한다. CFR에서 RFR과 ETR을 함께 사용하는 이유는 model diversity를 확보하기 위해서이다. 같은 유형의 regressor만 사용하는 것보다 서로 다른 bias-variance 특성을 가진 regressor를 조합하면 ensemble generalization이 개선될 수 있다.

### 3.8 비교 모델

논문은 CFR의 성능을 검증하기 위해 여러 baseline model을 비교한다. 첫째, gcForest를 비교한다. gcForest는 multi-grained scanning module과 cascade forest module로 구성된다. 이 연구에서는 12개 feature vector를 window size 4, step size 1로 sliding하여 더 높은 차원의 feature vector를 생성하고, 이를 cascade forest regression에 입력한다. 그러나 결과적으로 gcForest는 CFR보다 성능과 자원 효율성이 낮게 나타났고, 저자들은 HRV feature 기반 BP estimation에서는 multi-grained scanning module이 적합하지 않다고 해석한다.

둘째, ANN을 비교한다. ANN은 input dimension 12를 가지며, hidden layer는 6개, 각 hidden layer는 256 neuron으로 구성된다. Dropout rate는 0.3이고, optimizer는 Adam이며 learning rate는 0.001이다. Validation accuracy가 개선되지 않으면 EarlyStopping을 사용하여 학습을 멈춘다.

셋째, gradient boosting regressor, GBR과 hist gradient boosting regressor, HGBR을 비교한다. GBR은 여러 decision tree regressor를 순차적으로 학습하면서 이전 regressor의 error를 줄이는 방식이다. 논문에서는 learning rate 0.1, estimator 100, maximum depth 3을 사용했다. HGBR은 큰 dataset에서 빠르게 작동하도록 feature dimensionality를 줄인 개선형 gradient boosting 방식이며, loss는 squared error, learning rate는 0.01, max iteration은 100이다.

넷째, CNN-LSTM with self-attention과 GRU-LSTM with self-attention을 비교한다. CNN-LSTM 모델은 1D convolution layer로 feature pattern을 추출하고, LSTM으로 time dependency를 모델링한 뒤, self-attention으로 feature importance를 반영한다. GRU-LSTM 모델은 GRU layer, LSTM layer, attention layer, dense output layer로 구성되며, long-term sequence information을 처리하기 위한 모델이다.

## 4. 실험 및 결과

### 4.1 평가 지표

논문은 $R^2$, MAE, MSE, ME, STD를 사용하여 regression performance를 평가한다. $R^2$는 모델이 실제 target variance를 얼마나 설명하는지 나타낸다. 다만 논문에 제시된 $R^2$ 수식은 일반적인 $R^2 = 1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$와 다르게 보이는 부분이 있다. 본문 수식에는 numerator가 $(\hat{y}_i-\bar{y})^2$처럼 표기되어 있어 오기 가능성이 있다. 결과 해석은 통상적인 coefficient of determination 의미로 이해하는 것이 자연스럽다.

MAE는 다음과 같다.

$$
MAE = \frac{\sum_{i=1}^{n}|y_i-\hat{y}_i|}{n}
$$

MSE는 다음과 같다.

$$
MSE = \frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{n}
$$

ME는 다음과 같이 제시된다.

$$
ME = \frac{\sum_{i=1}^{n}y_i-\hat{y}_i}{n}
$$

STD는 다음과 같이 제시된다.

$$
STD = \sqrt{\frac{\sum_{i=1}^{n}(y_i-\bar{y})^2}{n}}
$$

여기서 $y_i$는 actual value, $\hat{y}_i$는 predicted value, $\bar{y}$는 average value, $n$은 sample 수이다. 다만 STD 수식 역시 prediction error의 standard deviation이 아니라 actual value의 standard deviation처럼 표기되어 있어 일반적인 AAMI용 error STD와는 다소 다르게 보인다. 그러나 Discussion의 AAMI 비교에서는 predicted error의 mean과 standard deviation으로 해석되는 값을 사용한다.

### 4.2 Training 및 adaptive layer 분석

CFR 모델은 training 중 $R^2$ score를 이용해 cascade depth를 자동 조절한다. Figure 9는 layer가 추가될 때마다 SBP와 DBP의 $R^2$ score가 어떻게 변화하는지 보여준다. 성능이 더 이상 증가하지 않으면 layer 추가를 멈추며, 논문에서는 6번째 level layer에서 training이 종료되었다고 설명한다. 이는 CFR이 deep neural network처럼 고정된 layer 수를 사전에 정하는 것이 아니라, validation performance에 따라 depth를 결정한다는 점에서 adaptive model이라는 장점을 가진다.

### 4.3 Resource cost 비교

논문은 CFR과 gcForest 등 모델의 execution time과 memory usage를 비교한다. 실험 환경은 Windows 11, 8GB memory, 4-core dual CPU이다. Figure 10에 따르면 CFR은 가장 짧은 execution time을 보였고, memory overhead 측면에서는 세 알고리즘 중 두 번째로 좋은 결과를 보였다. 저자들은 CFR이 small dataset task에서 특히 유리하며, 복잡한 의료 데이터 수집의 어려움을 완화할 수 있다고 설명한다.

### 4.4 Hyperparameter grid search

논문은 CFR의 주요 hyperparameter로 tree maximum depth, cascade layer 안의 estimator 수, 각 estimator 안의 tree 수를 분석한다. GridSearchCV를 이용해 5-fold cross-validation을 수행하고, parameter 조합에 따른 성능 변화를 관찰했다. Figure 11은 tree maximum depth와 estimator 조합, tree 수의 관계를 보여준다. 논문은 tree maximum depth가 cascaded regression 성능에 중요한 역할을 하며, maximum depth가 10을 넘으면 모델 fitting 정도가 수렴하는 경향이 있다고 설명한다. 이는 CFR이 tree 수를 무조건 늘리기보다 적절한 tree depth를 선택하는 것이 중요하다는 것을 보여준다.

Figure 13에서는 여러 parameter 조합에 대한 $R^2$와 MAE 변화를 보여준다. 저자들은 주요 parameter를 크게 늘린다고 해서 성능이 계속 향상되는 것은 아니며, 오히려 더 많은 computing resource를 소모한다고 설명한다. CFR은 비교적 적은 parameter configuration으로도 높은 predictive performance를 달성하므로, deep learning보다 hyperparameter 의존성이 낮다는 결론을 제시한다.

### 4.5 Ablation study

논문은 regressor 조합과 feature 수에 대한 ablation study를 수행한다. Regressor 조합은 RFR01+RFR02, ETR03+ETR04, RFR01+ETR03, 그리고 proposed method인 RFR01+RFR02+ETR03+ETR04로 비교된다. Figure 12(a)에 따르면 같은 종류의 regressor만 조합한 경우보다 서로 다른 종류의 regressor를 섞은 경우가 더 좋은 성능을 보였다. 특히 네 개 regressor를 모두 사용하는 proposed combination이 가장 높은 accuracy를 기록했다. 이는 ensemble diversity가 generalization 성능 향상에 기여한다는 해석을 뒷받침한다.

Feature 수에 대한 ablation도 수행되었다. Figure 12(b)에 따르면 12개 physiological feature 전체를 사용하는 것뿐 아니라, 추출된 feature 중 일부만 선택했을 때도 높은 성능을 보였다. 텍스트에 따르면 첫 5개 feature를 선택했을 때 SBP의 $R^2$ score가 0.941, DBP의 $R^2$ score가 0.916까지 도달했다. 이는 모든 feature가 반드시 필요한 것은 아니며, 일부 핵심 HRV feature만으로도 상당한 예측 성능을 낼 수 있음을 시사한다. 다만 최종 main result는 전체 12개 feature 기반 CFR로 보고된다.

### 4.6 Main results

Table 3은 여러 모델의 blood pressure estimation performance를 비교한다. CFR은 모든 주요 baseline보다 우수한 성능을 보인다.

DBP에 대해 CFR은 $R^2=0.926$, MSE 3.033 mmHg, MAE 1.760 mmHg, 실행 시간 17초를 기록했다. SBP에 대해 CFR은 $R^2=0.948$, MSE 4.625 mmHg, MAE 2.896 mmHg, 실행 시간 18초를 기록했다. 이는 gcForest, gradient boosting regressor, hist gradient boosting regressor, ANN, CNN-LSTM with self-attention, GRU with self-attention보다 좋은 성능이다.

예를 들어 gcForest는 DBP MAE 7.760, SBP MAE 10.560으로 성능이 낮았다. ANN은 DBP MAE 5.300, SBP MAE 9.400이었으며, HGBR은 DBP MAE 2.941, SBP MAE 5.128로 비교적 좋았지만 CFR보다 낮았다. Attention 기반 time-series 모델인 CNN-LSTM과 GRU도 SBP에서는 각각 MAE 3.886, 3.439로 어느 정도 성능을 보였지만, DBP에서는 각각 6.63, 5.572로 CFR보다 훨씬 낮은 성능을 보였다.

흥미로운 점은 time-series deep learning 모델이 HRV feature 기반 입력에서는 우수성을 보이지 못했다는 해석이다. 저자들은 HRV feature vector에는 raw waveform sequence만큼 충분한 time dependence가 없기 때문에 attention time-series combination model이 강점을 발휘하지 못한다고 설명한다. 반면 CFR은 feature vector 간 nonlinear relationship을 ensemble tree 구조로 잘 학습한 것으로 해석된다.

### 4.7 SOTA 비교

Table 4는 기존 연구들과의 비교를 제공한다. CFR은 MIMIC dataset에서 SBP MAE 2.90 mmHg, DBP MAE 1.76 mmHg를 기록하여 비교 연구들보다 낮은 오차를 보였다. 예를 들어 Kachuee et al.의 AdaBoost는 SBP MAE 11.17, DBP MAE 5.35였고, Wang et al.의 ANN은 SBP MAE 4.02, DBP MAE 2.27이었다. Su et al.의 LSTM은 SBP MAE 3.9, DBP MAE 2.66으로 좋은 성능을 보였지만 CFR보다 다소 높다.

다만 이 비교는 주의해서 해석해야 한다. 일부 연구는 public dataset이 아니며, dataset size, preprocessing, train-test split, subject composition이 다를 수 있다. 논문도 완전히 공정한 비교가 어렵다고 언급한다. 그럼에도 같은 MIMIC 기반에서 일부 SOTA 방법을 재검증한 결과 CFR이 우수하게 나타난 점은 실험적으로 의미가 있다.

### 4.8 Generalization 분석

논문은 train-test split과 leave-one-group-out, LOGO 방법을 비교하여 generalization capability를 분석한다. Table 5에 따르면 train-test split에서는 test 기준 SBP $R^2=0.948$, MAE 2.896이고, DBP $R^2=0.926$, MAE 1.76이다. LOGO에서는 test 기준 SBP $R^2=0.908$, MAE 3.86이고, DBP $R^2=0.901$, MAE 2.31이다.

LOGO에서 성능이 떨어지기는 하지만 여전히 비교적 높은 $R^2$와 낮은 MAE를 유지한다. 저자들은 이를 바탕으로 CFR이 training data에 overfit되지 않고 unseen data에도 일반화 가능하다고 주장한다. 그러나 LOGO의 group 수가 5이고 split data size가 300이라고 되어 있어, 실제 subject-independent leave-one-subject-out과 동일한지는 명확하지 않다. 따라서 patient-level generalization을 완전히 입증했다고 보기는 어렵다.

### 4.9 Regression plot, histogram, Bland-Altman plot

Figure 14는 predicted BP와 true BP 사이의 regression plot 및 Pearson correlation coefficient를 보여준다. 논문은 SBP와 DBP 모두에서 predicted value가 actual value와 잘 맞고 높은 correlation을 보였다고 설명한다. 텍스트에는 SBP와 DBP의 $R^2$ 값이 0.942와 0.948이라고 적혀 있는데, Table 3의 SBP 0.948, DBP 0.926과 일부 불일치가 있다. 이는 figure 기반 correlation 또는 다른 split에서 계산한 값일 가능성이 있지만, 제공 텍스트만으로 정확한 원인은 확인할 수 없다.

Figure 15는 predicted value와 true value의 histogram을 비교한다. Green line은 actual value, red line은 model-predicted value를 나타내며, 두 분포가 상당히 overlap되어 error가 작음을 시각적으로 보여준다.

Figure 16은 Bland-Altman plot이다. Horizontal axis는 각 sample의 predicted value와 true value의 평균이고, vertical axis는 두 값의 차이이다. Blue dotted line은 95% confidence interval, 즉 $1.96$ times standard deviation에 해당하는 upper/lower limits를 나타내며, gray horizontal line은 difference의 mean을 나타낸다. Bland-Altman plot은 예측 오차가 혈압 수준에 따라 systematic bias를 보이는지 확인하는 데 유용하다. 논문은 CFR 기반 추정이 reference와 잘 일치한다고 설명한다.

### 4.10 BHS 및 AAMI 기준 평가

BHS 기준에서 CFR은 SBP와 DBP 모두 Grade A를 달성했다. Table 6에 따르면 DBP는 error가 5 mmHg 이하인 비율이 93.71%, 10 mmHg 이하가 98.88%, 15 mmHg 이하가 99.57%이다. SBP는 5 mmHg 이하가 83.49%, 10 mmHg 이하가 95.48%, 15 mmHg 이하가 98.55%이다. BHS Grade A 기준은 각각 60%, 85%, 95%이므로, 두 혈압 모두 충분히 Grade A 기준을 넘는다.

AAMI 기준에서도 좋은 결과를 보였다. Table 7에 따르면 DBP의 ME는 1.800 mmHg, STD는 2.529 mmHg이고, SBP의 ME는 2.903 mmHg, STD는 3.526 mmHg이다. AAMI 기준은 일반적으로 ME가 5 mmHg 이하, STD가 8 mmHg 이하인지를 보므로, 두 값 모두 기준을 만족한다. 총 test sample은 7,669개이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 PPG 단일 신호만으로 높은 혈압 추정 성능을 보고했다는 점이다. ECG와 PPG를 함께 사용하는 PTT 기반 방법은 두 sensor의 동기화가 필요하지만, 이 연구는 PPG에서 추출한 HRV feature만으로 SBP와 DBP를 추정한다. 이는 smartwatch와 같은 wearable device에 적용하기 쉬운 방향이다.

두 번째 강점은 deep learning에 비해 tuning 부담이 낮은 CFR 구조를 제안했다는 점이다. Deep neural network는 layer 수, hidden unit 수, learning rate, optimizer, dropout, batch size 등 많은 hyperparameter에 민감하다. 반면 CFR은 random forest와 extra-trees 기반 ensemble을 cascade 방식으로 쌓고, validation performance에 따라 자동으로 depth를 조절한다. 논문은 이를 통해 hyperparameter configuration에 대한 의존도를 낮출 수 있다고 주장한다.

세 번째 강점은 feature interpretability이다. 입력 feature는 BPM, SDNN, RMSSD, IBI, SDSD, SD1, SD2, pNN20, pNN50 등 HRV와 관련된 생리적 지표이다. Raw waveform을 deep neural network에 직접 넣는 방식보다 각 feature가 무엇을 의미하는지 해석하기 쉽다. 이는 의료 분야에서 모델 설명 가능성을 높이는 장점이 있다.

네 번째 강점은 다양한 비교 실험이다. 논문은 gcForest, ANN, gradient boosting, hist gradient boosting, CNN-LSTM with self-attention, GRU with self-attention 등 다양한 baseline과 비교했고, regressor 조합 및 feature 수에 대한 ablation study도 수행했다. 또한 BHS와 AAMI 기준을 모두 사용해 의료기기 평가 관점의 성능도 제시했다.

그러나 한계도 명확하다. 첫째, 데이터 분할 방식이 subject-independent인지 명확하지 않다. 총 25,561개 segment를 train/test로 나누었다고 되어 있지만, 같은 환자의 서로 다른 segment가 train과 test에 동시에 포함될 가능성이 있다. 혈압 추정에서는 patient-specific pattern이 강하게 작용할 수 있으므로, segment-level split은 성능을 과대평가할 수 있다. LOGO 분석이 추가되었지만, group 구성이 subject 단위인지 명확하지 않다.

둘째, 논문 내 일부 수치와 표기가 일관되지 않다. Abstract에서는 MAE 1.760과 2.896을 각각 SBP와 DBP로 설명하지만, Table 3과 Discussion에서는 DBP MAE 1.760, SBP MAE 2.896으로 나타난다. 또한 $R^2$와 STD 수식에도 일반적 정의와 다른 표기가 보인다. 이런 수식 및 수치 표기 오류는 재현성과 신뢰성 측면에서 아쉬운 부분이다.

셋째, PPG preprocessing 설명에 의문점이 있다. 논문은 high-pass filter cut-off frequency를 40 Hz로 설계했다고 설명하는데, PPG의 주요 생리적 성분을 고려하면 이 설정은 직관적이지 않다. 실제로는 low-pass 또는 다른 filtering parameter를 의미했을 수 있지만, 제공 텍스트 기준으로는 확정할 수 없다. 이는 구현 재현에 중요한 정보이다.

넷째, 입력 feature가 5초 segment에서 계산된 HRV feature라는 점은 실용성과 한계를 동시에 가진다. 짧은 5초 구간에서는 충분한 heartbeat 수가 확보되지 않을 수 있으며, SDNN, RMSSD, pNN50 등 일부 HRV feature는 일반적으로 더 긴 구간에서 안정적으로 계산된다. 따라서 5초 segment에서 이러한 feature가 얼마나 안정적인지에 대한 추가 검증이 필요하다.

다섯째, 실제 wearable 환경 검증은 수행되지 않았다. 논문은 wearable device 적용 가능성을 강조하지만, 실험 데이터는 ICU 기반 MIMIC-II dataset이다. 실제 smartwatch 환경에서는 motion artifact, sensor contact 변화, 피부색, 온도, 운동 상태 등 다양한 문제가 발생한다. 따라서 MIMIC-II 기반 offline 성능이 실제 일상 환경에서도 유지된다고 단정할 수 없다.

비판적으로 보면, 이 연구는 PPG 기반 혈압 추정에서 deep learning 외의 강력한 ensemble alternative를 제시했다는 점에서 의미가 크다. 특히 CFR이 작은 feature set에서도 높은 성능을 보인 것은 주목할 만하다. 그러나 매우 낮은 MAE 수치는 subject-independent external validation이 부족할 경우 과대평가 가능성이 있으므로, 실제 임상적 신뢰성을 판단하려면 환자 단위 분할과 외부 데이터셋 검증이 추가로 필요하다.

## 6. 결론

이 논문은 PPG 신호에서 추출한 HRV feature를 이용하여 SBP와 DBP를 추정하는 cascade forest regression 기반 방법을 제안했다. CFR은 두 개의 random forest regressor와 두 개의 extra-tree regressor를 각 cascade layer에 배치하고, 각 layer에서 생성한 4-dimensional feature vector를 원래 12-dimensional HRV feature와 결합하여 다음 layer로 전달한다. Validation performance가 더 이상 향상되지 않으면 cascade depth를 자동으로 멈추기 때문에, deep learning보다 hyperparameter tuning 부담이 낮고 adaptive한 구조를 가진다.

실험 결과, 본문 Table 3 기준으로 CFR은 SBP에서 MAE 2.896 mmHg, $R^2=0.948$, DBP에서 MAE 1.760 mmHg, $R^2=0.926$을 달성했다. BHS 기준에서는 SBP와 DBP 모두 Grade A를 달성했고, AAMI 기준에서도 ME와 STD가 기준 범위 안에 들었다. 다양한 baseline과 비교했을 때 CFR은 gcForest, ANN, gradient boosting, hist gradient boosting, CNN-LSTM, GRU 기반 모델보다 좋은 성능을 보였다.

이 연구의 주요 기여는 단일 PPG 신호만으로 해석 가능한 HRV feature를 추출하고, 이를 cascade forest regression으로 학습하여 높은 혈압 추정 정확도를 달성했다는 점이다. 또한 deep learning에 비해 tuning 부담과 학습 복잡도를 줄이면서도 nonlinear relationship을 잘 모델링할 수 있음을 보였다.

향후 연구에서는 subject-independent split, external validation dataset, 실제 wearable 환경에서의 motion artifact 포함 실험, 장시간 연속 측정 검증, 짧은 5초 segment에서 HRV feature 안정성 분석이 필요하다. 이러한 검증이 보완된다면, CFR 기반 PPG blood pressure estimation은 smartwatch나 wearable healthcare device에서 cuffless BP monitoring을 구현하는 데 유용한 방향이 될 수 있다.
