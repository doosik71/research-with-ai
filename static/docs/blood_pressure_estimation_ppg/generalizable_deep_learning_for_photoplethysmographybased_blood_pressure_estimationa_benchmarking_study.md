# Generalizable deep learning for photoplethysmography-based blood pressure estimation—A benchmarking study

* **저자**: Mohammad Moulaeifard, Peter H. Charlton, Nils Strodthoff
* **발표연도**: 2025

## 1. 논문 개요

이 논문은 PPG(Photoplethysmography) 신호만을 이용한 혈압 추정 딥러닝 모델들이 실제로 얼마나 일반화될 수 있는지를 체계적으로 평가한 benchmarking study이다. 기존 PPG 기반 혈압 추정 연구들은 주로 같은 데이터 분포에서 학습하고 평가하는 in-distribution, 즉 ID 평가에 집중해 왔다. 그러나 실제 환경에서는 센서 종류, 측정 조건, 피험자 특성, 혈압 분포, 신호 품질이 학습 데이터와 다른 경우가 많다. 이 논문은 이러한 현실적인 차이를 out-of-distribution, 즉 OOD 일반화 문제로 정의하고, 여러 딥러닝 모델이 외부 데이터셋에서 얼마나 성능을 유지하는지를 분석한다.

연구의 핵심 목표는 새로운 최고 성능 모델을 제안하는 것이 아니다. 오히려 저자들은 PulseDB라는 대규모 데이터셋에서 여러 딥러닝 모델을 학습시키고, 내부 테스트뿐 아니라 Sensors, UCI, BCG, PPGBP 같은 외부 데이터셋에서 평가함으로써 PPG 기반 혈압 추정 모델의 일반화 한계를 드러내고자 한다. 또한 단순한 sample-based domain adaptation 방법을 적용하여, 학습 데이터와 테스트 데이터의 혈압 label distribution 차이를 줄이면 OOD 성능이 개선되는지 실험한다.

논문의 연구 문제는 다음과 같이 정리할 수 있다. PPG waveform만을 입력으로 사용하여 SBP(Systolic Blood Pressure)와 DBP(Diastolic Blood Pressure)를 예측하는 딥러닝 모델이 있을 때, 이 모델은 학습 데이터와 같은 분포의 테스트셋에서는 어느 정도 성능을 내는가? 그리고 학습 데이터와 다른 외부 데이터셋에서는 성능이 얼마나 저하되는가? 이러한 성능 저하는 모델 구조의 문제인가, 아니면 데이터셋 간 혈압 분포 차이와 같은 domain shift의 영향인가? 마지막으로 label distribution 기반 importance weighting을 사용하면 OOD 성능을 개선할 수 있는가?

이 문제는 매우 중요하다. PPG 기반 cuffless BP estimation은 웨어러블 기기에서 연속 혈압 모니터링을 가능하게 할 수 있는 유망한 기술이다. 그러나 실제 의료기기나 소비자용 웨어러블에 적용하려면 특정 데이터셋에서만 좋은 성능을 내는 모델이 아니라, 새로운 병원, 새로운 센서, 새로운 환자군에서도 안정적으로 동작하는 모델이 필요하다. 이 논문은 기존 연구의 낙관적인 ID 성능이 실제 일반화 성능을 과대평가할 수 있음을 보여주며, 향후 연구에서 반드시 외부 데이터셋 평가와 domain adaptation을 고려해야 한다고 주장한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 PPG 기반 혈압 추정에서 “좋은 ID 성능”이 “좋은 실제 일반화 성능”을 의미하지 않는다는 것이다. 딥러닝 모델은 학습 데이터에 존재하는 통계적 패턴을 매우 잘 학습할 수 있다. 그러나 혈압 추정 문제에서는 데이터셋마다 혈압 분포, 환자군, 센서, 신호 품질, 전처리 방법이 다르기 때문에 학습 데이터에서 잘 맞는 패턴이 외부 데이터셋에서는 깨질 수 있다.

저자들은 이를 검증하기 위해 PulseDB를 중심 데이터셋으로 사용한다. PulseDB는 MIMIC-III와 VitalDB에서 유래한 대규모 PPG, ECG, ABP waveform 데이터셋이며, 5361명의 subject와 524만 개 이상의 10초 segment를 포함한다. 이 논문은 PulseDB를 다양한 subset으로 나누어 학습하고, 내부 테스트와 외부 테스트를 모두 수행한다. 특히 PulseDB는 Calib, CalibFree, AAMI라는 서로 다른 평가 시나리오를 제공한다. Calib는 같은 subject의 일부 sample이 training과 test에 모두 존재하는 calibration 기반 시나리오이고, CalibFree는 train과 test subject가 겹치지 않는 calibration-free 시나리오이다. AAMI는 calibration-free 조건이면서 혈압 분포의 tail까지 더 엄격히 반영하는 의료기기 검증 지향 시나리오이다.

두 번째 핵심 아이디어는 OOD 성능 저하의 중요한 원인 중 하나가 label distribution shift, 즉 혈압값 분포의 차이라는 점이다. 예를 들어 학습 데이터에는 정상 혈압 구간이 많고 테스트 데이터에는 고혈압 구간이 많다면, 모델은 테스트 데이터에서 체계적인 bias를 보일 수 있다. 논문은 earth mover’s distance, 즉 EMD를 사용하여 학습 데이터와 테스트 데이터의 SBP 분포 차이를 측정하고, OOD MAE와의 상관성을 분석한다. 그 결과 데이터셋 간 혈압 분포 차이가 클수록 OOD 오차가 커지는 경향을 확인한다.

세 번째 핵심 아이디어는 label distribution 기반 importance weighting을 이용한 간단한 domain adaptation이다. 저자들은 target domain의 개별 label을 사용하지 않고, target dataset의 전체 label distribution만 사용한다고 설명한다. 학습 sample이 속한 혈압 bin에서 target distribution이 source distribution보다 더 높으면 해당 sample에 더 큰 weight를 주고, 반대로 target distribution에서 드문 영역이면 상대적으로 낮은 weight를 준다. 이렇게 하면 학습 과정에서 source dataset의 혈압 분포를 target dataset의 혈압 분포와 더 비슷하게 만들 수 있다.

이 논문의 차별점은 모델 성능을 단순히 한 데이터셋의 테스트셋에서 보고하는 것이 아니라, 여러 학습 데이터 구성, 여러 모델 구조, 내부 및 외부 데이터셋, domain adaptation까지 포함해 PPG 기반 BP estimation의 일반화 문제를 정면으로 분석했다는 점이다. 이는 “새 모델 제안 논문”보다는 “방법론적 경고와 benchmark 논문”에 가깝다.

## 3. 상세 방법 설명

### 3.1 전체 실험 파이프라인

논문의 전체 방법은 PPG 신호에서 SBP와 DBP를 예측하는 회귀 문제로 구성된다. 모든 모델은 PPG waveform만을 입력으로 사용하며, 출력 node는 두 개이다. 하나는 SBP 예측값, 다른 하나는 DBP 예측값이다. ECG, ABP waveform, demographic metadata는 모델 입력으로 사용하지 않는다. ABP는 reference BP를 만들기 위한 기준 신호로 사용된다.

전체 흐름은 다음과 같이 표현할 수 있다.

$$
PPG \ waveform \rightarrow DL \ model \rightarrow (\hat{SBP}, \hat{DBP})
$$

학습 손실은 MSE(Mean Squared Error)이다. 모델은 SBP와 DBP를 동시에 예측하도록 학습된다. 두 출력값을 동시에 학습하는 이유는 SBP와 DBP가 서로 독립적이지 않고, 같은 PPG waveform에 포함된 공통 생리 정보를 공유하기 때문이다.

개념적으로 loss는 다음과 같이 쓸 수 있다.

$$
L =
\frac{1}{n}
\sum_{i=1}^{n}
\left[
(SBP_i - \hat{SBP}_i)^2
+
(DBP_i - \hat{DBP}_i)^2
\right]
$$

논문은 모델 성능의 주요 지표로 MAE를 사용한다. MAE는 다음과 같이 정의된다.

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|Predicted_i - Reference_i|
$$

여기서 $Predicted_i$는 모델 예측값, $Reference_i$는 참조 혈압값이다. MAE는 평균적으로 몇 mmHg 정도 틀리는지를 직접 보여주기 때문에 혈압 추정 연구에서 해석이 쉽다.

### 3.2 데이터셋 구성

#### PulseDB

PulseDB는 이 연구의 핵심 training 및 ID/OOD 평가 데이터셋이다. PulseDB는 MIMIC-III와 VitalDB에서 선택·전처리된 PPG, ECG, ABP waveform segment로 구성된다. 전체적으로 5361명의 subject와 5,245,454개의 10초 segment를 포함한다. 데이터셋에는 age, gender, weight, height, BMI 같은 demographic 정보도 포함되어 있지만, 이 논문에서는 PPG waveform만 모델 입력으로 사용한다.

PulseDB는 크게 MIMIC source와 VitalDB source로 나뉜다. MIMIC은 ICU 환자의 fingertip PPG를 포함하고, VitalDB는 수술 환자 데이터를 포함한다. 두 source는 환자군, 혈압 분포, 측정 환경이 다르기 때문에 OOD 일반화 분석에 유용하다.

논문은 PulseDB를 source와 scenario 조합에 따라 9개 subset으로 생성한다. Source는 Combined, Vital, MIMIC이고, scenario는 Calib, CalibFree, AAMI이다. 따라서 Combined-Calib, Combined-CalibFree, Combined-AAMI, Vital-Calib, Vital-CalibFree, Vital-AAMI, MIMIC-Calib, MIMIC-CalibFree, MIMIC-AAMI의 9개 설정이 만들어진다.

Calib scenario는 같은 subject가 training과 test에 모두 나타날 수 있는 calibration 기반 설정이다. 이 경우 모델은 test subject의 일부 signal pattern을 학습 중에 본 적이 있으므로 subject-specific feature를 활용할 수 있다. CalibFree scenario는 training과 test set이 subject를 공유하지 않는다. 따라서 완전히 새로운 subject에 대한 일반화 성능을 평가한다. AAMI scenario도 calibration-free에 해당하지만, 혈압 분포의 tail을 더 강조하여 의료기기 검증 기준에 가까운 엄격한 설정으로 구성된다.

#### 외부 데이터셋

논문은 PulseDB 외부의 네 데이터셋을 OOD 평가에 사용한다.

Sensors dataset은 MIMIC-III에서 유래한 ICU 환자 PPG와 ABP waveform 데이터이다. 1195명의 ICU patient를 포함하며, 전처리 후 record당 15초 segment 두 개가 포함된다.

UCI dataset은 MIMIC-II에서 유래한 데이터셋이며, 외부 데이터셋 중 가장 큰 규모로 제시된다. 원문 표에서는 subject 수가 10,793명으로 나타난다. Segment는 410,596개이고 길이는 5초이다.

BCG dataset은 주로 건강한 40명의 subject로 구성된 작은 데이터셋이다. 전체 duration은 약 4시간이고, segment 수는 3063개이다. Subject 수는 작지만 subject당 segment 수가 많다.

PPGBP dataset은 cardiovascular condition이 있는 219명의 subject를 포함한다. 전처리 후 218명, 613개의 2.1초 segment가 남아 전체 duration이 1시간 미만으로 매우 짧다. 외부 데이터셋 중 총 duration 기준으로 가장 작다.

이 네 데이터셋은 PulseDB와 비교했을 때 sample size, signal quality, patient population, BP distribution이 다르기 때문에 OOD generalization 평가에 적합하다.

### 3.3 모델 구조

논문은 다섯 가지 딥러닝 time-series 모델을 평가한다. 모델의 목적은 모두 PPG waveform에서 SBP와 DBP를 직접 예측하는 것이다.

첫 번째는 **LeNet1D**이다. 이는 고전적인 feed-forward CNN 구조를 1D time series에 맞게 변형한 모델이다. 구조가 비교적 단순하지만, 특정 calibration-free나 AAMI 설정에서 ResNet이나 Inception과 비슷한 수준의 성능을 보인다고 보고된다.

두 번째와 세 번째는 **XResNet1d50**과 **XResNet1d101**이다. ResNet 계열 모델은 skip connection을 통해 gradient flow를 개선하고 깊은 CNN을 안정적으로 학습할 수 있게 한다. 1D ResNet은 PPG waveform의 local pattern과 hierarchical temporal feature를 추출하는 데 사용된다. XResNet1d101은 전체 분석에서 경쟁력 있는 성능을 보였기 때문에 이후 OOD 분석의 대표 모델로 선택된다.

네 번째는 **Inception1D**이다. Inception 구조는 여러 kernel size의 convolution filter를 병렬로 적용하여 다양한 temporal scale의 feature를 추출한다. PPG waveform에서는 짧은 pulse morphology와 조금 더 긴 temporal pattern이 모두 중요하므로, multi-scale feature extraction이 유리할 수 있다.

다섯 번째는 **S4(Structured State Space Sequence) model**이다. S4는 긴 sequence의 long-range dependency를 효율적으로 모델링하기 위해 제안된 structured state space model이다. ECG와 EEG 같은 다른 생리 시계열에서 좋은 성능을 보인 바 있지만, 이 논문에서는 PPG 기반 혈압 추정에서 ResNet이나 Inception만큼 뛰어난 성능을 보이지는 못했다고 보고된다.

### 3.4 학습 절차

모든 실험에서 effective batch size는 512이다. 이를 위해 gradient accumulation을 사용한다. S4 모델은 learning rate finder를 사용해 learning rate를 찾고, 다른 모델들은 learning rate를 0.001로 설정한다. Optimizer는 AdamW이며, 모든 모델은 50 epoch 동안 학습된다.

입력 sampling frequency는 모든 데이터셋에서 125 Hz로 통일되어 있다. PulseDB의 segment 길이는 10초이므로 1250 time steps를 가진다.

$$
10 \times 125 = 1250
$$

BCG, UCI, Sensors는 5초 segment를 사용하므로 625 time steps이다.

$$
5 \times 125 = 625
$$

PPGBP는 2.1초 segment를 사용하므로 262 time steps이다.

$$
2.1 \times 125 \approx 262
$$

모델들은 global average pooling을 포함하므로, PulseDB의 10초 입력으로 학습한 모델도 inference 시 5초 또는 2.1초 입력을 처리할 수 있다. 이는 외부 데이터셋마다 segment 길이가 다른 상황에서 중요한 설계적 장점이다.

Overfitting을 줄이기 위해 validation set score를 기준으로 model selection을 수행한다. 즉 학습 중 validation score를 추적하고, 가장 좋은 validation score를 보인 checkpoint를 최종 평가에 사용한다.

### 3.5 통계적 유의성 평가

논문은 모델 간 성능 차이가 test set 구성의 우연한 변동 때문인지 평가하기 위해 empirical bootstrapping을 사용한다. 각 실험 scenario에서 가장 낮은 MAE를 달성한 모델을 reference model로 선택하고, 다른 모델과의 performance difference에 대해 1000회 bootstrap을 수행한다.

이후 95% confidence interval을 계산한다. 원문에는 “estimated 95% confidence intervals contain 0”일 때 statistically significantly worse라고 쓰여 있으나, 일반적인 통계 해석에서는 차이의 신뢰구간이 0을 포함하면 유의한 차이가 없다고 해석한다. 따라서 이 문장은 원문 표현상 오류 또는 부정확한 기술일 가능성이 있다. 본 보고서에서는 원문이 의도한 바를 “bootstrap으로 모델 간 차이의 통계적 유의성을 평가했다”는 수준으로 해석한다.

### 3.6 Bland–Altman 분석

MAE는 평균적인 오차 크기를 보여주지만, 예측값이 체계적으로 높거나 낮은 bias를 보이는지는 알기 어렵다. 이를 위해 논문은 Bland–Altman analysis를 수행한다. 이 분석에서는 prediction과 reference의 차이를 계산하고, 그 평균인 bias와 limits of agreement, 즉 LoA를 보고한다.

오차를 $d_i = Predicted_i - Reference_i$ 또는 반대로 정의할 수 있으며, 논문에서는 prediction과 reference의 difference 평균을 bias로 사용한다. 일반적으로 bias는 다음과 같이 표현할 수 있다.

$$
bias =
\frac{1}{n}
\sum_{i=1}^{n}
d_i
$$

LoA는 오차 표준편차의 1.96배를 사용한다.

$$
LoA =
bias \pm 1.96 \cdot SD(d)
$$

Bias는 예측이 평균적으로 얼마나 치우쳤는지를 보여준다. LoA는 개별 예측 오차가 얼마나 넓게 퍼져 있는지를 보여준다. 논문은 특히 AAMI task에서 negative bias가 크게 나타난다고 보고한다. 이는 AAMI test set의 혈압 분포가 training distribution과 다르기 때문일 가능성이 높다고 해석한다.

### 3.7 Domain adaptation: label distribution 기반 importance weighting

논문은 OOD 성능 개선을 위해 간단한 sample-based domain adaptation 방법을 사용한다. 핵심 아이디어는 source domain의 label distribution을 target domain label distribution과 더 비슷하게 만들도록 training sample에 weight를 부여하는 것이다.

먼저 학습 데이터와 테스트 데이터의 SBP 또는 DBP label distribution을 histogram으로 만든다. 어떤 training sample이 bin $i$에 속한다고 하자. 이때 source train distribution에서 해당 bin의 확률을 $h_{train,i}$, target test distribution에서 해당 bin의 확률을 $h_{test,i}$라고 한다. 그러면 sample weight는 다음과 같이 정의된다.

$$
w_i =
\begin{cases}
\max\left(\tau, \frac{h_{test,i}}{h_{train,i}}\right), & \text{if } h_{train,i} > 0 \
\tau, & \text{if } h_{train,i} = 0
\end{cases}
$$

여기서 $\tau$는 너무 작은 weight 때문에 sample이 사실상 학습에서 제외되는 것을 막기 위한 hyperparameter이다. 이 논문에서는 $\tau = 1$로 고정한다.

이 weight는 loss 계산에 사용된다. SBP loss에는 SBP distribution에서 유도된 weight를 사용하고, DBP loss에는 DBP distribution에서 유도된 weight를 사용한다. 최종 importance-weighted loss는 두 값을 합친다.

개념적으로는 다음과 같이 쓸 수 있다.

$$
L_{weighted} = \frac{1}{n} \sum_{i=1}^{n} \left[ w^{SBP}_i(SBP_i-\hat{SBP}_i)^2 + w^{DBP}_i(DBP_i-\hat{DBP}_i)^2 \right]
$$

이 방식은 target domain의 개별 sample label을 사용하지 않고, target domain 전체의 label distribution만 사용한다. 저자들은 이것이 실제 상황에서 비교적 얻기 쉬운 정보라고 가정한다. 예를 들어 특정 병원 또는 cohort의 대략적인 혈압 분포는 알 수 있지만, 개별 PPG sample에 대한 정확한 혈압 label은 없을 수 있다.

## 4. 실험 및 결과

### 4.1 PulseDB 내부 모델 비교

논문은 9개 PulseDB subset 각각에서 LeNet1D, XResNet1d50, XResNet1d101, Inception1D, S4를 학습하고 평가한다. 주요 결과는 Table 3에 제시된다.

전체적으로 Calib scenario에서 오차가 가장 낮고, CalibFree와 AAMI로 갈수록 오차가 커진다. 이는 예상된 결과이다. Calib에서는 같은 subject의 일부 sample이 training set에 포함되므로 모델이 patient-specific signal pattern을 기억하거나 활용할 수 있다. 반면 CalibFree는 완전히 새로운 subject에 대해 평가하므로 더 어렵다. AAMI는 calibration-free에 더해 혈압 분포 tail을 강조하므로 가장 어려운 task가 된다.

Combined-Calib에서 XResNet1d101은 SBP/DBP MAE 9.43/5.98 mmHg를 달성한다. Vital-Calib에서는 XResNet1d101이 9.09/6.09 mmHg를 기록한다. MIMIC-Calib에서는 XResNet1d101이 9.52/6.64 mmHg를 기록한다. 반면 AAMI scenario에서는 오차가 훨씬 커진다. Combined-AAMI에서 XResNet1d101은 19.38/14.04 mmHg, Vital-AAMI에서는 19.31/12.33 mmHg, MIMIC-AAMI에서는 18.35/15.65 mmHg 수준이다.

모델 간 비교에서는 XResNet1d50, XResNet1d101, Inception1D가 전반적으로 좋은 성능을 보인다. 흥미롭게도 LeNet1D도 CalibFree와 일부 AAMI 설정에서는 상당히 경쟁력 있는 성능을 보인다. 이는 복잡한 모델이 항상 더 좋은 일반화 성능을 내는 것은 아니며, 특히 patient-specific memorization이 덜 중요한 setting에서는 단순 모델도 robust할 수 있음을 시사한다.

S4는 ECG나 EEG 영역에서 알려진 강력한 long-range modeling 능력에도 불구하고, 이 연구의 PPG 기반 혈압 추정에서는 두드러진 우위를 보이지 못한다. 이는 BP estimation 문제가 단순히 long-range dependency만으로 해결되는 것이 아니라, waveform morphology, label distribution, dataset shift의 영향을 강하게 받는 문제임을 보여준다.

### 4.2 PulseDB 내 ID 및 OOD 평가

논문은 XResNet1d101을 대표 모델로 선택하여 PulseDB 내부의 다양한 source와 scenario 사이에서 교차 평가를 수행한다. Table 4는 training subset과 test subset을 모두 9개로 두고, 각 조합의 MAE를 보여준다.

같은 source와 같은 scenario에서 train/test가 이루어진 경우, 즉 ID 조건에서는 상대적으로 낮은 오차가 나온다. 예를 들어 Combined-Calib에서 train하고 Combined-Calib에서 test하면 9.43/5.98 mmHg이다. Vital-Calib에서 train하고 Vital-Calib에서 test하면 9.09/6.09 mmHg이다. MIMIC-Calib에서 train하고 MIMIC-Calib에서 test하면 9.52/6.64 mmHg이다.

하지만 다른 source 또는 다른 scenario로 넘어가면 성능이 크게 저하된다. Vital-Calib에서 학습한 모델을 MIMIC-Calib에 적용하면 20.67/13.63 mmHg로 오차가 크게 증가한다. 반대로 MIMIC-Calib에서 학습한 모델을 Vital-Calib에 적용하면 16.08/10.49 mmHg이다. 이는 같은 PulseDB 안에서도 Vital과 MIMIC source 사이의 domain shift가 크다는 것을 보여준다.

논문은 MIMIC-based model보다 Vital-based model이 외부 dataset에서 더 좋은 generalization을 보이는 경향을 발견한다. 이는 기존 연구에서 MIMIC을 주된 학습 데이터셋으로 사용하는 관행에 의문을 제기한다. MIMIC은 ICU 환경의 특정 환자군과 신호 조건에 편향되어 있을 수 있기 때문이다.

AAMI test set은 전반적으로 가장 큰 오차를 보인다. 이는 AAMI가 unseen patient뿐 아니라 더 넓은 혈압 분포, 특히 tail 값을 포함하도록 설계되었기 때문이다. 따라서 AAMI 성능은 실제 의료기기 수준의 generalization을 더 엄격히 평가하는 지표로 해석할 수 있다.

### 4.3 외부 데이터셋 OOD 평가

Table 5는 PulseDB subset에서 학습한 XResNet1d101 모델을 Sensors, UCI, PPGBP, BCG 외부 데이터셋에 적용한 결과를 보여준다.

외부 데이터셋에서의 OOD 성능은 ID 성능보다 훨씬 불안정하고, training dataset에 따라 큰 차이를 보인다. 예를 들어 Vital-AAMI에서 학습한 모델은 Sensors에서 16.27/10.65 mmHg, UCI에서 19.70/10.35 mmHg, PPGBP에서 26.82/11.67 mmHg, BCG에서 14.33/7.66 mmHg를 기록한다. Vital-CalibFree는 Sensors에서 18.45/8.61 mmHg, BCG에서 10.05/6.93 mmHg로 비교적 좋은 결과를 보인다.

반면 MIMIC-based 모델은 외부 데이터셋에서 매우 나쁜 성능을 보이는 경우가 많다. MIMIC-Calib 모델은 Sensors에서 32.86/23.77 mmHg, UCI에서 43.72/28.31 mmHg, PPGBP에서 33.33/15.65 mmHg, BCG에서 26.95/12.33 mmHg를 보인다. MIMIC-AAMI도 Sensors에서 40.93/15.63 mmHg, UCI에서 44.92/16.26 mmHg로 높은 오차를 보인다.

이 결과는 OOD generalization이 단순히 학습 데이터 규모나 모델 복잡도만으로 결정되지 않는다는 것을 보여준다. 어떤 source에서 학습했는지, 그 source의 혈압 분포와 외부 데이터셋의 분포가 얼마나 유사한지가 중요하다. 논문은 CalibFree Vital을 기준으로 외부 데이터셋과의 SBP distribution EMD를 계산하고, EMD가 클수록 OOD MAE가 커지는 경향을 확인한다. 이는 label distribution mismatch가 domain shift의 주요 원인 중 하나임을 보여준다.

### 4.4 Importance weighting의 효과

논문은 label distribution 기반 importance weighting을 적용하여 OOD generalization이 개선되는지 평가한다.

PulseDB 내부 평가에서는 153개 비교 중 81개, 즉 53%에서 importance weighting이 성능을 개선했다. 평균 개선폭은 SBP 0.36 mmHg, DBP 0.27 mmHg로 비교적 작다. 하지만 AAMI Vital test set에서는 최대 4 mmHg까지 개선되었다고 보고된다. 이는 AAMI처럼 train distribution과 test distribution의 mismatch가 큰 setting에서 importance weighting이 더 효과적일 수 있음을 의미한다.

외부 데이터셋 OOD 평가에서는 72개 경우 중 40개, 즉 55%에서 성능이 개선되었다. 평균 개선폭은 SBP 2.66 mmHg, DBP 0.86 mmHg로 PulseDB 내부보다 훨씬 크다. 이는 외부 데이터셋에서는 label distribution shift가 더 크고, 이를 보정하는 weighting이 더 큰 효과를 낼 수 있음을 보여준다.

예를 들어 MIMIC-Calib 모델은 unweighted 상태에서 Sensors 32.86/23.77 mmHg, UCI 43.72/28.31 mmHg, PPGBP 33.33/15.65 mmHg, BCG 26.95/12.33 mmHg였으나, weighting 후 Sensors 19.79/9.25, UCI 20.39/11.44, PPGBP 22.73/10.78, BCG 15.06/7.74로 크게 개선된다. 특히 MIMIC-Calib의 외부 성능 개선은 매우 크다.

하지만 importance weighting이 항상 좋아지는 것은 아니다. Combined-AAMI 모델은 외부 데이터셋 일부에서 오히려 성능이 크게 나빠진다. 예를 들어 Combined-AAMI는 unweighted 상태에서 Sensors 28.67/11.40이었지만 weighted 상태에서는 39.92/12.30으로 악화된다. 이는 label distribution만 맞춘다고 모든 domain shift가 해결되지는 않음을 보여준다. 데이터셋 차이는 혈압 분포뿐 아니라 sensor hardware, signal quality, patient physiology, preprocessing 방식에도 존재하기 때문이다.

### 4.5 기존 ID benchmark와의 비교

논문은 외부 데이터셋에서 기존 benchmark 연구의 ID 성능과 자신들의 OOD 성능을 비교한다. 예를 들어 기존 연구에서 ResNet은 Sensors에서 17.46/8.33, UCI에서 16.59/8.30, PPGBP에서 13.62/8.61, BCG에서 12.20/7.76을 기록했다. SpectroResNet은 Sensors에서 17.29/9.73, UCI에서 21.92/10.21, PPGBP에서 11.01/8.46, BCG에서 9.89/6.29를 기록했다.

AAMI Vital에서 학습한 모델은 OOD 상태에서도 Sensors 16.27/10.65, UCI 19.70/10.35, BCG 14.33/7.66으로 일부 dataset에서 ID 성능과 비슷한 수준에 도달한다. Importance weighting을 적용한 AAMI Vital 모델은 PPGBP에서 17.18/8.16, BCG에서 10.01/7.51을 기록하여 일부 ID benchmark와 경쟁 가능한 수준이다.

이 결과는 적절한 training dataset과 scenario를 선택하면, PulseDB에서 학습한 모델이 외부 데이터셋에서도 어느 정도 경쟁력 있는 성능을 낼 수 있음을 보여준다. 그러나 모든 외부 데이터셋에서 임상적으로 충분한 성능을 달성한 것은 아니며, 여전히 상당한 오차가 존재한다.

### 4.6 임상 기준 관점의 해석

논문은 IEEE standard for wearable cuffless BP measuring devices를 언급하며, MAE가 7 mmHg를 초과하면 grade D로 간주되어 clinical use에 부적합하다고 설명한다. 이 관점에서 보면 대부분의 결과는 아직 임상적으로 충분하지 않다. ID에서 best model이 PulseDB Calib 조건에서 SBP MAE 9.0 mmHg, DBP MAE 5.8 mmHg 수준을 보였고, calibration-free에서는 SBP 13.9, DBP 8.5 mmHg 수준으로 악화된다. 외부 데이터셋에서는 SBP 10.0–18.6 mmHg, DBP 5.9–10.3 mmHg 범위로 보고된다.

즉 이 연구의 가장 중요한 메시지는 “모델이 좋은 성능을 보였다”가 아니라 “PPG 단독 혈압 추정은 아직 어렵고, 특히 OOD 일반화는 더 어렵다”이다. 논문은 일부 sample은 acceptable grade A-B에 들어가지만, 또 다른 상당수 sample은 grade D에 해당할 수 있다고 설명한다. 향후 연구에서는 어떤 sample이 잘 예측되고 어떤 sample이 실패하는지 clinical metadata나 signal quality를 통해 구분하는 것이 중요하다고 제안한다.

## 5. 강점, 한계

### 5.1 강점

이 논문의 가장 큰 강점은 PPG 기반 BP estimation 분야에서 자주 간과된 OOD generalization 문제를 정면으로 다룬다는 점이다. 많은 기존 연구는 같은 데이터셋에서 random split 또는 내부 test set으로 성능을 보고하며, 이 경우 실제 외부 데이터셋 성능이 과대평가될 수 있다. 이 논문은 다양한 PulseDB subset과 네 개의 외부 데이터셋을 사용해 이 문제를 체계적으로 보여준다.

두 번째 강점은 대규모 데이터셋인 PulseDB를 중심으로 다양한 학습 시나리오를 구성했다는 점이다. Calib, CalibFree, AAMI는 각각 calibration-based, calibration-free, 의료기기 검증 지향 조건을 나타낸다. 이를 통해 같은 모델이라도 평가 시나리오에 따라 성능이 크게 달라진다는 점을 명확히 보여준다.

세 번째 강점은 여러 모델 구조를 공정하게 비교했다는 점이다. LeNet1D, XResNet1d50, XResNet1d101, Inception1D, S4를 동일한 절차로 평가함으로써, 모델 구조 자체의 영향과 데이터셋 영향 모두를 분석한다. 특히 단순한 LeNet1D도 일부 calibration-free setting에서 좋은 성능을 보인다는 결과는 모델 복잡도만 강조하는 경향에 대한 중요한 반례이다.

네 번째 강점은 label distribution shift를 정량적으로 분석했다는 점이다. EMD를 이용하여 training dataset과 test dataset의 SBP distribution 차이를 측정하고, OOD MAE와의 상관성을 보였다. 이는 성능 저하가 단순히 “외부 데이터라서” 발생한 것이 아니라, 혈압 분포 차이라는 구체적 요인과 관련될 수 있음을 보여준다.

다섯 번째 강점은 domain adaptation을 단순하고 해석 가능한 방식으로 실험했다는 점이다. Importance weighting은 복잡한 adversarial adaptation이나 representation alignment 없이도 성능 개선 가능성을 확인할 수 있는 방법이다. 특히 외부 데이터셋에서 평균 SBP 2.66 mmHg, DBP 0.86 mmHg 개선은 의미 있는 결과이다.

여섯 번째 강점은 코드와 전처리 절차를 공개했다는 점이다. 논문은 Zenodo와 GitHub를 통해 preprocessing과 model training source code를 공개했다고 설명한다. 이는 benchmark 연구에서 매우 중요한 재현성 요소이다.

### 5.2 한계

첫 번째 한계는 PPG signal only 설정을 의도적으로 선택했기 때문에, 실제 혈압 추정에 도움이 될 수 있는 정보가 제외되었다는 점이다. 나이, 성별, BMI, 혈관 상태, 약물, sensor 위치, signal quality index 같은 metadata는 BP estimation에 중요한 정보를 제공할 수 있다. 논문은 PPG 단독의 predictive capacity를 평가하기 위해 이러한 정보를 제외했지만, 임상적으로 실용적인 모델을 만들려면 추가 정보의 활용이 필요할 수 있다.

두 번째 한계는 importance weighting이 target label distribution을 사용한다는 점이다. 저자들은 개별 label이 아니라 전체 label distribution만 사용한다고 설명하지만, 실제 deployment에서 target population의 혈압 분포를 사전에 정확히 아는 것은 항상 가능하지 않다. 또한 새로운 사용자나 새로운 환경에서는 target distribution 자체가 시간이 지남에 따라 변할 수 있다.

세 번째 한계는 domain adaptation이 label distribution shift만 보정한다는 점이다. 데이터셋 차이는 혈압 분포뿐 아니라 sensor hardware, sampling condition, signal quality, patient physiology, preprocessing pipeline, 병원 환경 등 다양한 요인에서 발생한다. 따라서 label histogram을 맞추는 것만으로는 모든 OOD 문제를 해결할 수 없다. 실제로 importance weighting이 일부 설정에서는 성능을 악화시킨다.

네 번째 한계는 외부 데이터셋의 규모와 특성이 매우 불균형하다는 점이다. PPGBP는 613개의 2.1초 segment만 포함하고, BCG는 subject 수가 40명으로 작다. 이러한 작은 데이터셋에서는 성능 추정의 통계적 변동성이 크다. 논문은 bootstrap을 사용해 불확실성을 일부 평가하지만, 외부 데이터셋 자체의 제한은 여전히 남는다.

다섯 번째 한계는 absolute performance가 아직 임상적으로 충분하지 않다는 점이다. 논문 스스로도 IEEE standard 기준에서 MAE가 7 mmHg를 넘으면 grade D에 해당한다고 언급한다. 많은 calibration-free 및 OOD 결과에서 SBP MAE는 10 mmHg 이상이고, 일부는 30–40 mmHg에 이른다. 따라서 이 연구는 임상 적용 가능한 모델을 완성했다기보다, 현재 모델들의 한계를 드러내는 연구로 해석해야 한다.

여섯 번째 한계는 모델 interpretability가 제한적이라는 점이다. 논문은 Bland–Altman 분석과 distribution shift 분석은 제공하지만, 모델이 PPG waveform의 어떤 morphology나 physiological feature에 의존하는지는 깊이 분석하지 않는다. 혈압 추정이 실제 생리적 관계를 학습한 것인지, 데이터셋별 shortcut을 학습한 것인지 더 구체적인 해석이 필요하다.

### 5.3 비판적 해석

이 논문은 PPG 기반 혈압 추정 연구에서 매우 중요한 균형추 역할을 한다. 최근 많은 논문들이 특정 데이터셋에서 낮은 MAE를 보고하지만, 실제 외부 데이터셋으로 가면 성능이 크게 떨어질 수 있다. 이 논문은 바로 그 문제를 정량적으로 보여준다. 특히 ID 평가가 일반화 성능을 과대평가한다는 점을 다양한 실험으로 입증한다.

또한 MIMIC 기반 모델의 OOD 성능이 좋지 않다는 결과는 중요하다. 많은 기존 연구가 MIMIC 또는 MIMIC-derived dataset을 중심으로 수행되었기 때문이다. 이 논문은 MIMIC이 크고 널리 쓰이는 데이터셋이라고 해서 반드시 좋은 일반화 성능을 주는 것은 아니며, VitalDB 기반 subset이나 AAMI Vital scenario가 더 나은 외부 일반화를 보일 수 있음을 보여준다.

다만 이 논문을 “PPG 기반 혈압 추정은 불가능하다”는 결론으로 해석해서는 안 된다. 논문은 PPG 단독 모델이 여전히 어렵지만, dataset selection, distribution-aware training, domain adaptation, metadata integration, self-supervised pretraining, foundation model 등을 통해 개선 여지가 있다고 본다. 즉 이 논문은 비판적이지만 건설적인 benchmark 연구이다.

## 6. 결론

이 논문은 PPG 기반 혈압 추정 딥러닝 모델의 일반화 성능을 체계적으로 평가한 benchmark 연구이다. 저자들은 PulseDB에서 여러 딥러닝 모델을 학습하고, 내부 테스트뿐 아니라 Sensors, UCI, BCG, PPGBP 같은 외부 데이터셋에서 OOD 성능을 평가했다. 그 결과 ID 성능은 실제 외부 일반화 성능을 과대평가하는 경향이 매우 크다는 점을 확인했다.

모델 비교에서는 XResNet1d50, XResNet1d101, Inception1D가 전반적으로 좋은 성능을 보였고, 단순한 LeNet1D도 일부 calibration-free 및 AAMI setting에서 경쟁력 있는 성능을 보였다. S4 모델은 다른 생리 시계열 분야에서의 강력한 성능과 달리 이 문제에서는 두드러진 우위를 보이지 못했다. 이후 분석에서는 XResNet1d101이 대표 모델로 사용되었다.

핵심 실험 결과는 다음과 같다. PulseDB에서 subject-specific calibration이 있는 ID 설정에서는 best model이 SBP MAE 9.0 mmHg, DBP MAE 5.8 mmHg 수준에 도달했지만, calibration-free에서는 SBP 13.9 mmHg, DBP 8.5 mmHg로 악화되었다. 외부 데이터셋에서 calibration 없이 평가한 경우 SBP MAE는 10.0–18.6 mmHg, DBP MAE는 5.9–10.3 mmHg 범위로 나타났다. 일부 설정에서는 훨씬 큰 오류도 관찰되었다.

논문은 이러한 OOD 성능 저하의 주요 원인 중 하나가 혈압 label distribution shift임을 보였다. EMD로 측정한 SBP distribution dissimilarity와 OOD MAE 사이에 상관성이 있었고, 이를 바탕으로 label distribution 기반 importance weighting을 적용했다. Importance weighting은 PulseDB 내부에서는 평균 SBP 0.36 mmHg, DBP 0.27 mmHg 개선을 보였고, 외부 데이터셋에서는 평균 SBP 2.66 mmHg, DBP 0.86 mmHg 개선을 보였다. 그러나 모든 설정에서 개선된 것은 아니므로, label distribution 보정만으로 OOD 문제를 완전히 해결할 수는 없다.

이 연구의 가장 중요한 기여는 PPG 기반 혈압 추정에서 외부 데이터셋 평가와 OOD generalization을 필수적으로 고려해야 한다는 점을 명확히 보여준 것이다. 또한 어떤 training dataset과 scenario가 일반화에 유리한지에 대한 실용적인 시사점을 제공한다. 특히 Vital 및 AAMI Vital 기반 학습이 외부 일반화에서 강한 후보로 나타났고, MIMIC 중심 학습의 일반화 한계가 드러났다.

향후 연구에서는 PPG 단독 입력을 넘어 clinical metadata, signal quality control, self-supervised pretraining, domain adaptation, foundation model, fixed population label distribution 기반 reweighting 등을 결합해야 할 가능성이 높다. 종합적으로 이 논문은 새로운 최고 성능 모델을 제시하는 논문이라기보다, PPG 기반 cuffless BP estimation 분야가 실제 임상 및 웨어러블 적용으로 나아가기 위해 반드시 해결해야 할 일반화 문제를 정교하게 드러낸 benchmark 연구로 평가할 수 있다.
