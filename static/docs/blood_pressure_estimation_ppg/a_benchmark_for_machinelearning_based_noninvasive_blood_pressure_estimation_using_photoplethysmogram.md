# A benchmark for machine-learning based non-invasive blood pressure estimation using photoplethysmogram

* **저자**: Sergio González, Wan-Ting Hsieh, Trista Pei-Chun Chen
* **발표연도**: 2023

## 1. 논문 개요

이 논문은 photoplethysmography, 즉 PPG 신호를 이용한 machine learning 기반 non-invasive blood pressure estimation 연구들을 공정하게 비교하기 위한 표준 benchmark를 제안한다. 기존 연구들은 서로 다른 dataset, 서로 다른 preprocessing, 서로 다른 train-validation-test split, 서로 다른 evaluation metric을 사용해 왔기 때문에, 어떤 모델이 실제로 더 좋은지 판단하기 어려웠다. 이 논문은 이러한 문제를 해결하기 위해 네 개의 공개 dataset, 통일된 preprocessing 절차, subject information leakage를 방지하는 validation strategy, 그리고 표준화된 evaluation metric을 갖춘 benchmark를 설계하였다.

이 연구의 핵심 문제는 단순히 새로운 BP estimation model을 제안하는 것이 아니라, **PPG 기반 BP estimation 연구를 어떻게 공정하게 평가할 것인가**이다. 특히 BP estimation은 일반적인 regression 문제와 다르게, 동일 subject에서 여러 segment가 나오며 SBP와 DBP label distribution이 skewed distribution을 갖는 경우가 많다. 따라서 단순 random split을 사용하면 같은 subject의 데이터가 training set과 test set에 동시에 들어가 subject information leakage가 발생할 수 있고, 이로 인해 모델 성능이 비현실적으로 높게 평가된다.

논문이 제안하는 benchmark는 네 개의 공개 dataset, 즉 Sensors, UCI, BCG, PPGBP를 사용한다. 이 dataset들은 subject 수, segment 수, segment continuity, BP distribution, demographic information에서 서로 다른 특성을 가진다. 이를 통해 특정 dataset에만 적합한 모델이 아니라, 다양한 조건에서 ML/DL 기반 BP estimation 방법을 비교할 수 있다.

논문은 BP estimation 모델을 세 가지 category로 나눈다. 첫째, **Feat2Lab**은 PPG waveform에서 handcrafted feature를 추출한 뒤 SBP와 DBP label을 예측하는 방식이다. 둘째, **Sig2Lab**은 raw PPG signal을 직접 입력으로 받아 SBP와 DBP label을 예측하는 deep learning 방식이다. 셋째, **Sig2Sig**는 PPG signal을 입력으로 받아 continuous ABP signal을 생성한 뒤, 이 ABP waveform에서 SBP와 DBP를 추출하는 방식이다.

논문은 총 11개의 representative ML/DL algorithm을 비교한다. Feat2Lab에는 LightGBM, SVR, Random Forest, MLP, AdaBoost가 포함된다. Sig2Lab에는 ResNet, SpectroResNet, MLP-BP가 포함된다. Sig2Sig에는 U-Net, PPG2IABP, V-Net이 포함된다. 이 비교는 단일 모델의 성능을 주장하기보다, 같은 preprocessing과 validation 조건에서 각 접근 방식이 어떤 dataset에서 강한지를 보여준다.

이 연구의 중요성은 PPG 기반 cuff-less BP estimation이 wearable healthcare에서 매우 중요한 응용임에도, 기존 논문들의 결과가 서로 직접 비교되기 어려웠다는 데 있다. 논문은 open dataset과 code를 제공하여 reproducibility를 높이고, 향후 연구자가 새로운 모델을 제안할 때 동일 benchmark 위에서 성능을 비교할 수 있도록 한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **PPG 기반 BP estimation 연구에서 모델 성능보다 먼저 평가 체계의 신뢰성이 확보되어야 한다**는 것이다. 기존 연구에서는 특정 dataset에서 높은 accuracy를 보고하는 경우가 많았지만, 그 결과가 preprocessing, data split, subject leakage, BP distribution imbalance에 의해 과대평가되었을 가능성이 있었다. 이 논문은 이러한 문제를 명시적으로 분석하고, benchmark design 자체를 주요 기여로 삼는다.

첫 번째 핵심 설계는 네 개의 dataset을 하나의 공통 pipeline으로 전처리하는 것이다. PPG와 ABP waveform은 측정 위치가 다르기 때문에 시간 지연이 존재할 수 있다. 논문은 cross-correlation을 이용해 PPG와 ABP를 align하고, 이후 5초 길이의 non-overlapping segment로 나눈다. 그 다음 ABP와 PPG의 품질이 낮은 segment를 제거하고, baseline wander를 cubic spline interpolation으로 보정한다. 이렇게 dataset마다 다른 전처리 관행을 제거하여, 모델 비교의 공정성을 높인다.

두 번째 핵심 설계는 validation strategy이다. 일반적인 random split은 BP estimation에서 위험하다. 동일 subject의 segment가 training set과 test set에 나뉘어 들어가면, 모델은 실제 생리적 관계를 학습했다기보다 subject-specific pattern을 기억할 수 있다. 특히 UCI나 BCG처럼 같은 subject 또는 record에서 연속 segment가 많이 생성되는 dataset에서는 leakage가 성능을 크게 부풀린다. 논문은 이를 방지하기 위해 subject 단위로 fold를 분리하고, 동시에 SBP/DBP distribution을 유지하는 stratified partitioning을 사용한다.

세 번째 핵심 아이디어는 evaluation metric으로 Mean Absolute Scaled Error, MASE를 BP estimation에 도입한 것이다. MAE는 직관적인 metric이지만, BP range가 좁은 dataset에서는 자연스럽게 더 낮은 MAE가 나올 수 있다. 따라서 dataset 간 비교에서는 MAE만으로 성능을 판단하기 어렵다. MASE는 모델 MAE를 naive predictor의 MAE로 나눈 값이다. Naive predictor는 training set의 mean SBP 또는 mean DBP를 항상 예측하는 모델이다. 이 metric은 dataset의 BP range 차이를 어느 정도 보정하여, 서로 다른 dataset 간 비교를 더 해석 가능하게 만든다.

네 번째 핵심 아이디어는 PPG 기반 BP estimation 방법을 input-output 형식에 따라 체계적으로 분류한 것이다. Feat2Lab은 전문가가 설계한 PPG feature에 의존하지만 작은 dataset에서 강하다. Sig2Lab은 feature engineering 없이 raw PPG를 직접 사용하지만 충분한 데이터가 필요하다. Sig2Sig은 continuous ABP waveform까지 추정할 수 있어 정보가 풍부하지만, training을 위해 invasive ABP waveform이 필요하다는 부담이 있다. 논문은 이 세 category를 같은 조건에서 비교하여, 각 접근 방식의 실제 장단점을 드러낸다.

## 3. 상세 방법 설명

### 전체 benchmark pipeline

논문의 benchmark pipeline은 크게 dataset collection, signal alignment, segmentation, signal cleaning, feature extraction 또는 raw signal preparation, validation split, model training, evaluation으로 구성된다. 모든 dataset은 가능한 한 동일한 전처리 기준을 적용받는다.

먼저 PPG와 ABP signal이 함께 있는 dataset에서는 두 waveform을 cross-correlation 기반으로 align한다. PPG signal은 측정 위치 차이 때문에 ABP보다 지연될 수 있으므로, maximum cross-correlation magnitude가 되는 shift를 찾아 보정한다. 다만 과도하게 비현실적인 shift를 막기 위해 최대 1초까지만 허용한다. 이후 각 record를 5초 길이의 non-overlapping chunk로 나눈다. PPGBP처럼 ABP waveform이 없는 dataset은 Sig2Sig model training이 불가능하므로, label 기반 방식만 적용된다.

### Data preprocessing

전처리는 ABP와 PPG 모두의 품질을 보장하기 위해 여러 단계로 구성된다. 첫째, extremely abnormal ABP segment를 제거한다. ABP는 invasive measurement이므로 gold standard에 가깝지만, 실제 데이터에는 error나 disturbance가 존재할 수 있다. 논문은 cardiac cycle을 식별할 수 없거나, amplitude가 30–220 mmHg 범위를 벗어나거나, pulse pressure가 10 mmHg 이하이거나, resting heart rate가 35–140 BPM 범위를 벗어나는 segment를 제거한다.

둘째, PPG cycle identification을 수행한다. PPG cardiac cycle은 initial valley, systolic peak, second valley로 구성된다. Valley나 peak가 없거나 과도하게 많은 segment는 제거된다. 또한 heart rate가 adult resting range인 35–140 BPM을 벗어나면 제거된다.

셋째, distorted PPG waveform을 제거한다. 논문은 peak-to-peak interval, valley-to-valley interval, peak amplitude, valley amplitude의 standard deviation이 지나치게 큰 segment를 distortion으로 간주한다. Threshold는 waveform과 cumulative percentage plot을 검사하여 설정하였다.

넷째, baseline wander를 제거한다. Baseline wander는 respiration이나 movement로 인해 발생하는 low-frequency artifact이다. 논문은 cardiac cycle의 valley point를 이용해 cubic spline interpolation으로 baseline을 추정하고, 원 signal에서 이 baseline을 빼서 corrected waveform을 만든다.

다섯째, preprocessing을 반복적으로 정제하고 SQI를 적용한다. Feature extraction이 실패하는 segment는 제거하고, skewness 기반 signal quality index가 0보다 낮은 segment도 제외한다. 이 과정을 거쳐 각 dataset의 최종 segment 수가 결정된다.

최종적으로 Sensors dataset은 1195 subjects, 11102 segments, UCI dataset은 10793 records, 410596 segments, BCG dataset은 40 subjects, 3063 segments, PPGBP dataset은 218 subjects, 619 segments로 정리되었다.

### 세 가지 모델 category

#### Feat2Lab: PPG feature에서 BP label로

Feat2Lab은 PPG waveform에서 handcrafted feature를 추출하고, 이를 ML regressor에 입력하여 SBP와 DBP를 예측한다. 이 논문은 기존 연구에서 자주 사용된 PPG feature를 세 그룹으로 정리한다.

첫째, points-of-interest와 time-based feature이다. PPG waveform과 그 derivative에서 특정 point를 찾는다. 예를 들어 PPG의 systolic peak, first derivative인 VPG의 $w$, $y$, $z$, second derivative인 APG의 $a$, $b$, $c$, $d$, $e$ 등이 사용된다. 이 point들에서 amplitude, elapsed time, area under curve 등을 계산한다. 특히 systolic peak에서 dicrotic notch 또는 diastolic rise까지의 elapsed time이 중요한 feature로 평가되었다.

둘째, frequency-based feature이다. Fast Fourier Transform, FFT를 통해 dominant frequency, 그 magnitude, 주변 평균 magnitude 등을 추출한다.

셋째, operational/statistical feature이다. PPG, VPG, APG의 systolic phase와 diastolic phase에 대해 histogram feature를 계산하고, slope deviation curve, SQI skewness와 kurtosis, Aging Index 및 여러 index feature를 포함한다.

Feature 수가 많기 때문에 논문은 tree-based ensemble을 이용해 feature selection을 수행한다. Fully-grown Random Forest와 Extra-Trees를 각각 SBP와 DBP에 대해 학습하고, Gini impurity 감소량의 평균을 feature importance로 사용한다. 중요도 순으로 feature를 정렬한 뒤, 선택할 feature 비율을 hyperparameter로 조정한다.

Feat2Lab의 regression model로는 LightGBM, SVR, MLP, AdaBoost, Random Forest를 사용한다.

#### Sig2Lab: PPG signal에서 BP label로

Sig2Lab은 handcrafted feature 없이 raw PPG signal을 입력으로 받아 SBP와 DBP label을 직접 예측한다. 이 방식은 CNN을 automatic feature extractor로 사용하여 signal morphology를 학습한다.

논문은 Sig2Lab 대표 모델로 ResNet, SpectroResNet, MLP-BP를 사용한다. ResNet은 residual block을 통해 깊은 network를 안정적으로 학습할 수 있고, PPG와 같은 1D signal에서도 automatic feature extraction에 적합하다. SpectroResNet은 ResNet-GRU 구조를 사용하여 temporal information과 spectro-temporal information을 함께 추출한다. MLP-BP는 MLP-Mixer를 BP estimation에 맞게 변형한 model이다.

#### Sig2Sig: PPG signal에서 ABP signal로

Sig2Sig은 PPG waveform을 입력으로 받아 continuous ABP waveform을 출력한다. 이렇게 생성된 ABP waveform에서 systolic peak와 valley를 detection하여 SBP와 DBP를 추출한다.

논문은 Sig2Sig 대표 모델로 U-Net, PPG2IABP, V-Net을 사용한다. U-Net은 contracting path와 expansive path를 가지며, skip connection을 통해 low-level feature와 high-level feature를 함께 활용할 수 있다. PPG-to-ABP translation에서 U-Net이 자주 사용되는 이유는 signal-to-signal mapping에 적합하기 때문이다. PPG2IABP는 GRU encoder-decoder와 attention mechanism을 이용해 ABP mean cycle을 추정한다. V-Net은 ABP waveform estimation을 위한 또 다른 deep architecture로 포함된다.

PPGBP dataset은 continuous ABP signal이 없고 SBP/DBP label만 제공하므로, Sig2Sig model은 이 dataset에서 학습할 수 없다.

### Validation strategy

이 논문에서 가장 중요한 방법론적 기여 중 하나는 validation strategy이다. BP dataset은 일반 regression dataset과 다르다. 동일 subject의 여러 segment는 서로 독립적이지 않고, SBP와 DBP는 skewed distribution을 가진다. 따라서 random split은 두 가지 문제를 일으킨다.

첫째, subject information leakage가 발생한다. 동일 subject의 segment가 training과 test에 동시에 들어가면, 모델은 일반화 가능한 PPG-BP 관계가 아니라 subject-specific characteristic을 학습할 수 있다. 특히 연속 segment는 혈압 값이 크게 변하지 않기 때문에 leakage 효과가 더욱 커진다.

둘째, extreme BP label이 특정 fold에서 부족하거나 빠질 수 있다. SBP와 DBP는 정상 범위 sample이 많고 고혈압 또는 저혈압 sample은 적은 imbalance regression 문제이다. Random split은 validation set과 test set의 BP distribution을 다르게 만들 수 있고, 이 경우 validation 성능이 test 성능으로 잘 이어지지 않는다.

이를 해결하기 위해 논문은 subject 단위로 fold를 나누면서 SBP와 DBP distribution을 유지하는 stratified partitioning을 사용한다. 구체적으로 SBP는 네 class로 나눈다: 100 mmHg 미만, 100–140 mmHg, 140–160 mmHg, 160 mmHg 초과. DBP도 네 class로 나눈다: 60 mmHg 미만, 60–80 mmHg, 80–100 mmHg, 100 mmHg 초과. 두 label의 조합은 총 16개 class가 된다. 각 subject에 대해 이 class combination frequency를 계산한 뒤, iterative stratification for multi-label data를 사용해 K fold로 subject를 분할한다.

Sensors, BCG, PPGBP dataset에는 5-fold cross-validation을 적용한다. UCI dataset은 subject identification이 없고 sample 수가 매우 많기 때문에 Hold-One-Set-Out, HOO strategy를 사용한다.

### Evaluation metrics

논문은 BP estimation 연구에서 MAE, ME, SD를 함께 보고할 것을 권장한다. ME는 error의 평균 bias를 나타낸다.

$$
ME = \frac{1}{n}\sum_{i=1}^{n} Diff_i
$$

여기서 $Diff_i = PRED_i - REF_i$이다. 즉, 예측 혈압에서 reference 혈압을 뺀 값이다.

SD는 error의 분산 또는 안정성을 나타낸다.

$$
SD = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(Diff_i - ME)^2}
$$

MAE는 절대 오차의 평균이다.

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|Diff_i|
$$

논문은 여기에 MASE를 추가로 제안한다. MASE는 모델의 MAE를 naive prediction의 MAE로 나눈 값이다.

$$
MASE = \frac{MAE}{MAE_{Naive}}
$$

Naive prediction은 training set의 mean SBP 또는 mean DBP를 항상 예측하는 방식이다. MASE가 100%이면 naive baseline과 같은 수준이고, 100%보다 작으면 naive보다 좋은 성능이다. 예를 들어 어떤 dataset에서 MAE가 낮더라도 BP range가 좁아서 낮게 나온 것일 수 있다. MASE는 이러한 scale 차이를 보정하여 dataset 간 비교를 쉽게 만든다.

### Hyperparameter tuning

ML model은 grid search로 hyperparameter를 조정한다. 예를 들어 SVR은 RBF kernel, $C$, gamma, epsilon을 탐색하고, RF와 LightGBM은 tree 수, depth, minimum samples per leaf, sampling rate 등을 조정한다. DL model은 Mean Squared Error, MSE를 loss function으로 사용하고 Adam optimizer로 학습하며, validation loss에 대해 patience 15 epochs의 early stopping을 사용한다. Hyperparameter search에는 Optuna Toolkit을 사용한다.

## 4. 실험 및 결과

### Dataset 특성

Sensors dataset은 MIMIC-III subset으로, 최종적으로 1195 subjects와 11102 segments를 포함한다. Segment 길이는 5초이고, validation은 5-fold CV이다. SBP 평균은 134.36 mmHg, DBP 평균은 65.37 mmHg이다. ICU 환자 data이므로 다양한 BP range를 포함한다.

UCI dataset은 MIMIC-II Waveform Dataset의 subset이며, subject 정보가 없다. 최종적으로 410596 segments를 포함하는 가장 큰 dataset이다. Segment 길이는 5초이고, validation은 HOO 방식이다. SBP 평균은 131.57 mmHg, DBP 평균은 66.79 mmHg이다.

BCG dataset은 40 subjects, 3063 segments로 구성되며, subject 수는 작지만 subject당 segment 수가 많다. Segment 길이는 5초이고 validation은 5-fold CV이다. SBP 평균은 120.99 mmHg, DBP 평균은 67.23 mmHg이다. BP distribution이 상대적으로 좁다.

PPGBP dataset은 218 subjects, 619 segments로 가장 작은 dataset이다. 각 segment는 2.1초이며, continuous ABP가 아니라 SBP/DBP label만 제공한다. 따라서 Sig2Sig model은 적용되지 않는다. SBP 평균은 128.02 mmHg, DBP 평균은 71.91 mmHg이다.

### Feat2Lab 결과

Feat2Lab category에서는 전반적으로 SVR과 LightGBM이 가장 강한 성능을 보였다. Sensors dataset에서는 SVR이 SBP MAE 15.60, DBP MAE 7.50으로 우수했고, LightGBM도 SBP MAE 15.63, DBP MAE 7.61로 근접했다. UCI dataset에서는 LightGBM이 SBP에서 16.85, SVR이 DBP에서 8.07로 각각 좋은 결과를 보였다.

BCG dataset에서는 AdaBoost가 SBP MAE 11.42로 가장 좋았고, MLP가 DBP MAE 7.14로 가장 좋았다. PPGBP dataset에서는 LightGBM이 SBP MAE 13.06, AdaBoost와 SVR이 DBP MAE 8.04로 가장 좋은 수준이었다.

논문은 Feat2Lab이 특히 작은 dataset과 medium-sized dataset에서 강하다고 해석한다. Handcrafted feature는 domain knowledge를 반영하므로 데이터가 적을 때 deep learning보다 안정적일 수 있다. 다만 feature extraction이 복잡하고 noise에 취약하며, 새로운 sensor나 population에 대해 feature engineering을 다시 검토해야 할 수 있다.

### Sig2Lab 결과

Sig2Lab category에서는 ResNet이 가장 좋은 모델로 평가된다. ResNet은 대부분의 dataset에서 Sig2Lab 모델 중 가장 낮은 MASE를 기록했다. UCI dataset에서는 ResNet이 SBP MAE 16.59로 전체 category 중 가장 좋은 SBP 성능을 보였다. PPGBP dataset에서도 ResNet은 SBP MAE 13.62, DBP MAE 8.61로 SpectroResNet과 MLPBP보다 우수했다.

SpectroResNet은 일부 dataset에서 성능이 낮았다. 특히 UCI에서 SBP MAE 19.88, DBP MAE 9.00으로 악화되었고, PPGBP에서도 SBP MAE 18.87, DBP MAE 11.38로 매우 낮은 성능을 보였다. MLPBP는 대체로 naive baseline과 비슷한 수준을 보이는 경우가 많았다.

논문은 raw PPG를 직접 학습하는 deep learning 방식이 large dataset에서는 Feat2Lab을 능가할 수 있지만, dataset이 작으면 handcrafted feature 기반 방법이 여전히 경쟁력이 있다고 해석한다.

### Sig2Sig 결과

Sig2Sig category에서는 U-Net이 가장 일관되게 좋은 모델로 평가된다. Sensors dataset에서 U-Net은 SBP MAE 15.64, DBP MAE 7.66을 기록하여 Feat2Lab의 best model과 유사한 수준을 보였다. UCI dataset에서는 U-Net이 DBP MAE 7.88로 전체 모델 중 가장 좋은 DBP 성능을 보였다. PPGIABP는 UCI DBP에서 8.07, Sensors DBP에서 7.99로 중간 수준이었고, V-Net은 dataset에 따라 성능 변동이 컸다.

BCG dataset에서는 V-Net이 SBP MAE 11.42로 매우 좋은 성능을 보였고, PPGIABP가 DBP MAE 7.78로 U-Net보다 약간 우수했다. 하지만 전체적으로는 U-Net이 Sig2Sig 대표 모델로 적합하다고 결론낸다.

Sig2Sig 방식의 중요한 장점은 continuous ABP waveform을 추정할 수 있다는 것이다. 이는 SBP/DBP label만 출력하는 모델보다 더 많은 cardiovascular information을 제공한다. 그러나 training에 invasive ABP waveform이 필요하므로 데이터 확보가 어렵고, PPGBP처럼 ABP가 없는 dataset에는 적용할 수 없다.

### Category 간 비교

논문은 dataset 크기에 따라 우수한 approach가 달라진다고 분석한다. 작은 dataset인 BCG와 PPGBP에서는 Feat2Lab이 강했다. 특히 PPGBP에서는 LightGBM, SVR, AdaBoost 등 classical ML model이 ResNet보다 더 좋은 성능을 보였다.

중간 규모인 Sensors dataset에서는 SVR과 U-Net이 SBP에서 비슷한 성능을 보였고, DBP에서도 SVR이 강했다. 즉, Feat2Lab과 Sig2Sig이 경쟁하는 형태였다.

가장 큰 UCI dataset에서는 deep learning 기반 모델이 더 강했다. ResNet이 SBP에서 가장 좋은 성능을 냈고, U-Net이 DBP에서 가장 좋은 성능을 냈다. 이는 충분한 데이터가 주어졌을 때 automatic feature extraction 또는 signal-to-signal learning이 handcrafted feature보다 더 나은 가능성을 가진다는 점을 보여준다.

### Feature importance 분석

Feat2Lab feature importance 분석에서 가장 중요한 feature는 PPG의 point-of-interest 사이 elapsed time이었다. 특히 $T_{s_e}$, $T_{s_z}$, $vpg_z$가 상위 feature로 나타났다. $T_{s_e}$는 systolic peak에서 dicrotic notch까지의 elapsed time이고, $T_{s_z}$는 systolic peak에서 diastolic rise point까지의 elapsed time이다. $vpg_z$는 first derivative signal에서 $z$ point의 amplitude이다.

Feature group 관점에서는 time-based feature가 SBP와 DBP 모두에서 중요했다. Area-based feature와 SDC feature도 유용한 것으로 나타났다. 반면 frequency-based feature와 width-based feature는 상대적으로 낮은 relevance를 보였다. Frequency feature가 낮게 평가된 이유는 PPG waveform의 BP 관련 정보가 dominant frequency보다는 cardiac cycle morphology와 time interval에 더 강하게 반영되기 때문으로 해석할 수 있다.

### Data splitting의 영향

논문은 subject leakage가 성능을 얼마나 부풀리는지 실험적으로 보여준다. Random split을 사용하여 같은 subject의 segment가 training과 test에 동시에 들어가는 leaked scenario와, subject 단위로 분리한 no-leak scenario를 비교했다. 결과적으로 모든 category의 모델이 leaked scenario에서 더 좋은 성능을 보였다.

특히 UCI와 BCG처럼 연속 segment가 많고 subject 또는 record당 segment 수가 큰 dataset에서 leakage 효과가 컸다. 일부 경우 SBP와 DBP의 MASE가 98–92% 수준에서 60% 이하로 급격히 낮아졌다. 이는 성능이 실제로 좋아진 것이 아니라, test set에 training과 같은 subject pattern이 들어가 모델이 이를 이용했기 때문이다.

또한 논문은 BP distribution을 유지하지 않고 fold를 나누면 validation 성능이 test 성능으로 잘 transfer되지 않는다는 점도 보였다. Sensors dataset에서 distribution-aware split 전후를 비교했을 때, distribution을 유지하지 않은 경우 validation-test MASE difference의 mean과 standard deviation이 더 컸다. 이는 extreme BP sample이 fold마다 불균형하게 배치되어 평가 안정성이 떨어지는 것을 의미한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 PPG 기반 BP estimation 분야에서 반복적으로 발생하던 비교 불가능성 문제를 정면으로 다루었다는 점이다. 기존 연구들은 서로 다른 dataset, preprocessing, split, metric을 사용하면서 자신들의 모델 성능을 주장했지만, 이 논문은 같은 조건에서 11개 모델을 비교할 수 있는 framework를 제시했다. 이는 새로운 모델을 만드는 것만큼이나 중요한 기여이다.

두 번째 강점은 subject information leakage 문제를 실험적으로 명확히 보여주었다는 것이다. BP estimation 연구에서 같은 subject의 segment가 train과 test에 동시에 들어가는 것은 매우 흔한 오류이며, 특히 continuous waveform dataset에서는 성능을 크게 부풀릴 수 있다. 논문은 이 문제를 단순히 지적하는 수준이 아니라, leaked split과 no-leak split의 결과 차이를 보여줌으로써 validation design의 중요성을 설득력 있게 제시한다.

세 번째 강점은 MASE를 도입해 dataset 간 성능 비교를 더 해석 가능하게 만들었다는 점이다. 예를 들어 BCG dataset은 BP range가 좁기 때문에 MAE가 낮게 나올 수 있다. 단순 MAE만 보면 BCG에서 모델이 더 좋은 것처럼 보일 수 있지만, MASE를 보면 실제 naive baseline 대비 개선 정도는 다를 수 있다. 이 metric은 benchmark의 공정성을 높이는 중요한 장치이다.

네 번째 강점은 Feat2Lab, Sig2Lab, Sig2Sig이라는 input-output 기반 taxonomy를 제시했다는 것이다. 이 분류는 연구자들이 자신의 모델이 어떤 category에 속하는지, 어떤 dataset 조건에서 유리한지, 어떤 label과 sensor requirement를 갖는지 이해하는 데 도움을 준다.

다섯 번째 강점은 open dataset split과 code를 제공한다는 점이다. Scientific Data 논문으로서 reproducibility와 reusability를 중시하며, 향후 연구자가 동일 split과 preprocessing으로 새로운 모델을 평가할 수 있도록 했다.

그러나 한계도 있다. 첫째, benchmark가 사용하는 dataset들은 모두 공개 dataset이지만, 대부분 clinical environment 또는 ICU 기반 data이다. 실제 wearable daily-life 환경의 motion artifact, sensor displacement, skin tone variability, device-specific noise가 충분히 반영되지 않았을 수 있다. 따라서 benchmark에서 좋은 성능을 낸 모델이 실제 smartwatch나 ring 기반 BP monitoring에서도 동일하게 작동한다고 보장할 수 없다.

둘째, UCI dataset은 가장 크지만 subject information이 없다. 논문은 이 때문에 HOO strategy를 사용했지만, subject-level leakage를 완전히 확인할 수는 없다. UCI dataset이 매우 큰 장점을 제공하는 동시에, subject identity 부재는 validation 신뢰성 측면에서 구조적 한계로 남는다.

셋째, Sig2Sig 모델은 continuous ABP waveform이 있는 dataset에서만 학습 가능하다. ABP는 invasive arterial line에서 얻어야 하므로 대규모 일반 population dataset 확보가 어렵다. 논문은 Sig2Sig이 정보량 측면에서 선호될 수 있다고 말하지만, 실제 구현과 training data 확보 측면에서는 매우 큰 장벽이 있다.

넷째, benchmark는 algorithm 비교에 초점을 맞추며, 실제 medical device validation을 대체하지 않는다. 논문도 명확히 언급하듯이, benchmark에서 좋은 성능을 얻은 ML algorithm이라도 실제 certification을 위해서는 ANSI/AAMI/ISO, BHS, IEEE 등 표준 protocol에 따른 별도 validation이 필요하다.

다섯째, 모델의 computational cost, memory footprint, energy consumption, latency 등 wearable deployment 관점의 효율성은 핵심 평가 대상이 아니다. PPG 기반 BP estimation이 wearable application을 지향한다면, 모델 accuracy뿐 아니라 on-device inference 가능성도 중요하다. 이 benchmark는 주로 estimation performance와 validation fairness에 집중한다.

비판적으로 보면, 이 논문은 PPG 기반 BP estimation 분야에서 “무엇이 state-of-the-art인가”보다 “어떻게 state-of-the-art를 공정하게 측정할 것인가”에 더 큰 가치를 둔다. 따라서 특정 모델 성능만 보려는 독자에게는 직접적인 algorithmic novelty가 부족해 보일 수 있다. 하지만 연구 분야의 재현성과 공정성을 높인다는 점에서 학술적 기여는 매우 크다.

## 6. 결론

이 논문은 PPG 기반 non-invasive blood pressure estimation 연구를 공정하게 비교하기 위한 benchmark를 제안하였다. 네 개의 공개 dataset, 통일된 preprocessing, subject leakage를 방지하는 validation strategy, BP distribution을 유지하는 stratified split, 그리고 MAE, ME, SD, MASE를 포함한 evaluation metric을 결합하여 reproducible evaluation framework를 구축했다.

실험 결과, 작은 dataset에서는 handcrafted feature를 사용하는 Feat2Lab 방식이 여전히 강력했다. SVR과 LightGBM은 여러 dataset에서 안정적인 성능을 보였고, 특히 PPGBP와 BCG처럼 데이터가 적은 환경에서 경쟁력이 높았다. 반면 large dataset인 UCI에서는 deep learning 기반 Sig2Lab과 Sig2Sig이 더 나은 성능을 보였다. ResNet은 SBP estimation에서, U-Net은 DBP 및 ABP waveform 기반 estimation에서 강점을 보였다.

Feature importance 분석에서는 PPG cardiac cycle의 elapsed time feature, 특히 dicrotic notch와 diastolic rise 관련 time feature가 SBP와 DBP 추정에 중요하다는 결론을 얻었다. Frequency-based feature는 상대적으로 낮은 중요도를 보였다.

이 연구의 가장 중요한 메시지는 BP estimation에서 validation design이 모델 구조만큼 중요하다는 것이다. Subject information leakage와 skewed BP distribution을 무시하면 성능이 크게 과대평가될 수 있다. 따라서 향후 PPG 기반 BP estimation 연구는 반드시 subject-level split, BP distribution-aware partitioning, 표준 metric 보고를 따라야 한다.

종합하면, 이 논문은 새로운 BP estimation model 자체를 제안한 논문이라기보다, 앞으로의 PPG 기반 BP estimation 연구가 따라야 할 평가 기반을 제공한 benchmark 논문이다. 실제 medical-grade wearable BP monitoring을 위해서는 여전히 정확도 향상, external validation, subject calibration, real-world wearable data 검증, medical standard validation이 필요하지만, 이 논문은 그 이전 단계에서 연구 결과를 공정하게 비교할 수 있는 중요한 기준점을 제공한다.
