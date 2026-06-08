# Enhancement of blood pressure estimation method via machine learning

* **저자**: Nashat Maher, G.A. Elsheikh, W.R. Anis, Tamer Emara
* **발표연도**: 2021

## 1. 논문 개요

이 논문은 PPG 신호만을 이용하여 cuffless, calibration-free blood pressure estimation을 수행하는 machine learning 기반 방법을 제안한다. 연구의 목표는 기존 cuff 기반 혈압 측정 방식의 불편함과 간헐적 측정 한계를 줄이고, ECG 없이 단일 PPG sensor에서 추출한 특징만으로 systolic pressure, 즉 SP와 diastolic pressure, 즉 DP를 추정하는 것이다.

혈압은 시간에 따라 변동하며, nutrition, behavioral condition, tension 등 다양한 요인의 영향을 받는다. 특히 hypertension은 심혈관 질환과 여러 장기 손상의 위험을 증가시키므로, 한 시점에서의 단발성 측정보다 지속적이고 간편한 모니터링이 중요하다. 기존 sphygmomanometer는 정확하고 널리 쓰이지만 cuff를 팔에 감아야 하므로 불편하고 continuous measurement에 적합하지 않다. Intra-arterial blood pressure monitoring은 연속 측정이 가능하지만 cannula를 artery에 삽입해야 하는 invasive method이므로 ICU나 수술실처럼 제한된 환경에서만 사용된다.

이 논문이 다루는 핵심 연구 문제는 단일 PPG signal에서 추출한 physiological features와 machine learning regression model만으로 SP와 DP를 ISO 및 BHS 기준에 부합하는 수준으로 추정할 수 있는가이다. 특히 저자들은 기존 PTT 또는 PAT 기반 방법처럼 ECG와 PPG를 동시에 측정하지 않고, PPG만으로 calibration-free estimation을 달성하려고 한다.

논문은 두 단계 구조를 제안한다. 첫 번째 단계에서는 전체 데이터를 이용한 coarse BP estimation model을 사용하여 입력 PPG signal이 low, normal, pre-high, high BP 중 어느 범위에 속하는지 분류한다. 두 번째 단계에서는 해당 BP level에 맞는 별도 regression model을 적용하여 더 정확한 BP를 추정한다. 또한 기존 PPG features에 LASI를 곱한 crossed features를 추가하여 feature representation을 강화한다.

문제의 중요성은 wearable blood pressure monitoring과 직접 연결된다. PPG는 optical sensor 기반이므로 비교적 저렴하고, 비침습적이며, wearable device에 통합하기 쉽다. 이 논문이 주장하는 방식이 충분히 안정적이라면, cuff나 ECG electrode 없이도 간단한 PPG device를 이용해 혈압을 추정할 수 있으므로 장기 모니터링과 개인 건강관리 장치에 활용될 수 있다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 PPG waveform에서 추출한 몇 가지 physiological features만으로는 혈압과의 관계를 충분히 표현하기 어렵기 때문에, feature crossing과 BP range별 model specialization을 결합하여 추정 정확도를 높이는 것이다.

첫 번째 아이디어는 LASI, 즉 Large Artery Stiffness Index를 다른 PPG features와 곱해 crossed features를 만드는 것이다. 논문은 기존 feature들과 BP 출력 사이의 correlation이 LASI와의 product operation을 통해 증가한다고 설명한다. 사용된 기본 feature는 heart rate, augmentation index, LASI, 그리고 PPG curve의 area feature인 S1, S2, S3, S4이다. 이후 LASI를 각각의 feature와 곱하여 $LASI \times HR$, $LASI \times AI$, $LASI \times LASI$, $LASI \times S1$, $LASI \times S2$, $LASI \times S3$, $LASI \times S4$와 같은 crossed features를 만든다. 논문에서는 이 feature crossing이 전체적으로 결과를 약 17%에서 20% 정도 향상시켰다고 설명한다.

두 번째 아이디어는 전체 혈압 범위를 하나의 regression model로 처리하지 않고, BP level에 따라 데이터를 나눈 뒤 각 범위에 특화된 모델을 학습하는 것이다. 논문은 BP와 PWV 사이에 exponential relationship이 있어 high BP 영역에서 non-linearity가 커지고, 하나의 전체 모델로는 고혈압 영역의 추정 오차가 커질 수 있다고 설명한다. 이를 해결하기 위해 전체 데이터를 hypotensive, normal, pre-high, hypertensive 네 그룹으로 나누고, 각 그룹마다 별도의 regression model을 학습한다.

세 번째 아이디어는 coarse-to-fine estimation 구조이다. 실제 적용에서는 입력 PPG signal의 true BP level을 알 수 없으므로, 먼저 전체 데이터로 학습한 main model을 사용해 BP range를 거칠게 추정한다. 그 다음 해당 range에 대응하는 specialized model을 적용하여 정밀 추정을 수행한다. 즉, 첫 번째 모델은 classifier처럼 작동하고, 두 번째 모델은 regression accuracy를 높이는 역할을 한다.

기존 접근 방식과 비교했을 때 이 논문의 차별점은 ECG를 사용하지 않는다는 점, individual calibration을 요구하지 않는다는 점, 그리고 PPG feature crossing과 BP level별 모델 분할을 결합한다는 점이다. 기존 PAT 또는 PTT 기반 방식은 ECG와 PPG를 동시에 측정해야 하며, 두 장치의 filtering 및 processing delay가 PAT 계산에 영향을 줄 수 있다. 반면 이 논문은 PPG만 사용하므로 하드웨어 구성이 더 단순하고, 사용자 입장에서 착용 부담이 줄어든다.

## 3. 상세 방법 설명

논문에서 제안한 전체 파이프라인은 여섯 단계로 구성된다. 첫째, PPG와 BP database를 선택한다. 둘째, raw signal에서 unreliable signal을 제거하고 smoothing을 수행한다. 셋째, PPG waveform에서 physiological features를 추출한다. 넷째, 데이터를 training, validation, test set으로 나눈다. 다섯째, linear 및 non-linear machine learning regression model을 학습한다. 여섯째, 학습된 모델로 BP를 추정하고 MAE, STD, MAPE, SMAPE 등의 지표로 성능을 평가한다.

데이터셋은 PhysioNet에서 제공되는 MIMIC II database를 사용했다. 이 데이터베이스는 여러 병원에서 수집된 physiological signals를 포함하며, 신호는 125 Hz sampling rate와 8-bit accuracy로 저장되어 있다고 설명된다. 논문에서는 PPG signal과 continuous arterial blood pressure, 즉 ABP signal을 사용한다. ABP는 SP와 DP의 reference target으로 사용된다. 데이터 전처리 이후 2000개 이상의 records가 확보되었고, 일부 실험에서는 820개의 perfect samples가 사용되었다고 제시된다. BP level별 세부 모델에서는 각 범위마다 sample 수가 다르게 나타난다.

Sample selection and filtration 단계에서는 왜곡되거나 신뢰할 수 없는 신호를 제거한다. 첫 번째로 simple averaging filter를 이용하여 모든 신호를 smoothing한다. 두 번째로 비정상적이거나 허용할 수 없는 BP 값을 제거한다. 세 번째로 허용 범위를 벗어난 heart rate를 가진 신호를 제거한다. 네 번째로 smoothing filter로도 극복하기 어려운 심한 disturbance가 있는 신호를 제거한다. 논문은 이러한 filtering의 구체적 threshold나 algorithmic rule을 상세히 제시하지는 않는다. 따라서 어떤 기준으로 “unacceptable” signal을 정의했는지는 제공된 텍스트만으로 완전히 재현하기 어렵다.

PPG feature extraction에서는 다섯 종류의 feature 그룹을 사용한다. 첫 번째는 heart rate이다. 이는 PPG signal의 peak-to-peak time interval을 이용해 계산된다. 두 번째는 augmentation index, 즉 AI이다. AI는 artery에서 wave reflection을 나타내는 지표로 설명되며, diastolic peak와 systolic peak의 비율로 계산된다. 세 번째는 LASI이다. LASI는 arterial stiffness를 나타내는 지표로, systolic peak와 diastolic peak 사이의 time interval로 계산된다. 네 번째는 inflection point area ratio, 즉 IPA이다. 이는 PPG curve 아래의 area를 S1, S2, S3, S4라는 구간으로 나누어 계산한 ratio로 설명된다. 다섯 번째는 crossed features이다. 이는 기존 feature에 LASI를 곱한 feature들이다.

Feature crossing의 핵심은 혈압과 PPG feature 사이의 비선형 관계를 더 잘 표현하기 위한 feature engineering이다. 논문에서 사용한 crossed features는 다음과 같이 정리할 수 있다.

$$
LASI \times HR
$$

$$
LASI \times AI
$$

$$
LASI \times LASI
$$

$$
LASI \times S1
$$

$$
LASI \times S2
$$

$$
LASI \times S3
$$

$$
LASI \times S4
$$

기본 feature 7개만 사용하는 경우와, LASI를 곱한 crossed features를 포함해 14개 feature를 사용하는 경우가 비교되었다. 논문은 14 crossed features setting이 7 features setting보다 전반적으로 더 좋은 결과를 보인다고 보고한다.

Data partition 단계에서는 처리된 dataset을 arbitrary split 방식으로 70% training, 15% validation, 15% testing sample로 나누었다. 그러나 논문 텍스트에서는 subject-independent split인지, record-level random split인지 명확하지 않다. MIMIC II처럼 한 환자에게 여러 segment가 있을 수 있는 데이터에서 record-level random split을 사용하면 같은 환자의 유사한 신호가 train과 test에 동시에 들어갈 가능성이 있다. 제공된 텍스트만으로는 이 문제가 통제되었는지 확인할 수 없다.

Machine learning model로는 SVM, ANN, 여러 regression algorithm이 비교되었다. Table 1에는 linear regression, interaction linear, robust linear, stepwise linear, regression tree, SVM variants, ensemble tree, Gaussian process regression, ANN with Levenberg-Marquardt, ANN with Bayesian algorithm 등이 포함된다. 논문에서 최종적으로 가장 강조하는 모델은 ANN regression using Bayesian algorithm이다.

ANN 구조는 세 층으로 구성된다. 첫 번째는 input layer이고, 크기는 feature vector의 크기이다. 따라서 7 features를 사용하는 경우 input dimension은 7이고, 14 crossed features를 사용하는 경우 input dimension은 14로 해석된다. 두 번째는 hidden layer이며 hidden neuron 수는 15이다. 세 번째는 output layer이고, 두 개의 neuron을 가진다. 이 두 output neuron은 각각 SP와 DP를 추정한다. 논문은 MATLAB R2017b의 “nnstart”와 “anntool”을 사용했다고 명시한다.

ANN 학습에서는 Bayesian algorithm과 Levenberg-Marquardt back propagation algorithm이 사용되었고, cost function은 mean square error, 즉 MSE이다. MSE는 추정값과 실제값의 차이를 제곱하여 평균한 값으로, 큰 오차에 더 큰 penalty를 부여한다. 일반적인 형태는 다음과 같이 쓸 수 있다.

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

여기서 $y_i$는 실제 SP 또는 DP이고, $\hat{y}_i$는 모델이 추정한 SP 또는 DP이다. 제공된 텍스트에는 MSE의 명시적 수식은 없지만, “mean square error is used as the cost function”이라고 제시되어 있다. SBP와 DBP 두 출력에 대해 MSE가 어떻게 결합되는지의 세부 정의는 명확하게 제시되어 있지 않다.

성능 평가는 MAE, STD, MAPE, SMAPE를 사용한다. 논문에서 MAE는 다음과 같이 정의된다.

$$
MAE = \frac{\sum |d|}{n}
$$

여기서 $d$는 estimated output과 actual target output 사이의 error이고, $n$은 sample 수이다. 쉬운 말로 하면, MAE는 추정 혈압이 실제 혈압에서 평균적으로 몇 mmHg 정도 벗어나는지를 나타낸다.

STD는 논문에서 다음과 같이 제시된다.

$$
STD = \sqrt{\frac{\sum d^2}{n}}
$$

일반적인 통계적 standard deviation 식과는 달리, 평균 오차를 빼는 항이 포함되어 있지 않아 root mean square error와 유사한 형태로 보인다. 그러나 논문에서는 이를 STD of estimation errors로 사용한다. 이 보고서에서는 논문 표기와 용어를 따라 STD라고 부르되, 수식상 일반적인 표준편차 정의와 차이가 있음을 유의해야 한다.

MAPE는 다음과 같이 제시된다.

$$
MAPE = \frac{1}{n} \sum_{t=1}^{n} \frac{A_t - F_t}{A_t}
$$

SMAPE는 다음과 같이 정의된다.

$$
SMAPE = \frac{1}{n} \sum_{t=1}^{n} \frac{|F_t - A_t|}{|A_t| + |F_t|}
$$

여기서 $A_t$는 actual value이고, $F_t$는 forecast value이다. 다만 논문에 제시된 MAPE 식에는 절댓값 표기가 명확히 보이지 않는다. 일반적인 MAPE는 절대 percentage error를 사용하지만, 제공된 텍스트만으로는 저자들이 실제 계산에서 절댓값을 사용했는지 확인할 수 없다.

BP level별 enhancement method는 논문의 핵심 절차이다. 전체 데이터로 학습한 모델은 BP range를 coarse하게 분류하는 데 사용된다. 이후 입력이 hypotensive, normal, pre-high, hypertensive 중 어느 그룹에 속하는지 판단되면, 해당 그룹의 전용 model이 fine estimation을 수행한다. Hypotensive는 SP가 90 mmHg보다 작고 DP가 60 mmHg보다 작은 경우로 설명된다. Normal BP는 mean BP가 70에서 100 mmHg 사이, pre-high BP는 mean BP가 100에서 115 mmHg 사이로 설정된다. Hypertensive 그룹은 high BP range에 해당하지만, 제공된 텍스트에서는 hypertensive 분할 기준이 normal 및 pre-high처럼 명확한 mean BP threshold 형태로 완전히 정리되어 있지는 않다.

## 4. 실험 및 결과

실험은 크게 전체 데이터 모델, feature crossing 효과, BP level별 specialized model, 그리고 실시간 구현 검증으로 나뉜다. 전체 데이터 모델에서는 7 features와 14 crossed features를 사용한 다양한 machine learning regression algorithm을 비교했다. 그 결과 가장 강조되는 성능은 ANN with Bayesian algorithm과 14 crossed features 조합에서 얻어졌다.

Table 1에 따르면 ANN with Bayesian algorithm은 14 crossed features를 사용할 때 SP에서 STD 5.296 mmHg, MAE 4.235 mmHg를 달성했고, DP에서 STD 6.373 mmHg, MAE 4.457 mmHg를 달성했다. 같은 모델에서 7 features만 사용할 때는 SP STD 6.397 mmHg, MAE 4.805 mmHg였고, DP STD 6.547 mmHg, MAE 4.682 mmHg였다. 따라서 14 crossed features는 7 features보다 SP와 DP 모두에서 오차를 줄였다. 논문의 결론부에서는 LASI feature crossing이 systolic STD error를 17.15%, diastolic STD error를 3.43%, systolic MAE를 15.85%, diastolic MAE를 6.65% 개선했다고 설명한다.

다양한 regression algorithm 비교에서도 Gaussian process regression이나 SVM 계열이 좋은 결과를 보였지만, 최종적으로 ANN with Bayesian algorithm이 좋은 정확도와 짧은 training time을 함께 보인 것으로 제시된다. 예를 들어 Exponential GPR과 Rational Quadratic GPR은 SP와 DP에서 낮은 MAE를 보이지만 training time이 매우 길다. 반면 ANN with Bayesian algorithm은 training time이 약 1.8초로 표기되어 계산 효율이 높다. 다만 training time은 MATLAB 환경과 데이터 크기에 의존하므로 일반화에는 주의가 필요하다.

BHS standard와의 비교에서는 SP와 DP 모두 Grade A를 달성했다고 보고된다. BHS Grade A는 error가 5 mmHg 이하인 비율이 60% 이상, 10 mmHg 이하인 비율이 85% 이상, 15 mmHg 이하인 비율이 95% 이상이어야 한다. 논문 결과는 SP에서 각각 65.85%, 94.51%, 99.51%, DP에서 67.93%, 91.59%, 97.5%로 제시된다. 따라서 SP와 DP 모두 Grade A 기준을 만족한다.

ISO 기준과 관련해서 논문은 non-invasive BP device의 요구 조건으로 average difference가 5 mmHg 이하, standard deviation이 8 mmHg 이하이어야 한다고 설명한다. 전체 데이터 모델의 대표 결과로 제시된 SP MAE 4.235 mmHg, STD 5.296 mmHg, DP MAE 4.457 mmHg, STD 6.373 mmHg는 이 기준을 만족한다. 초록에서는 SP STD 약 5.3 mmHg, DP STD 약 6.4 mmHg, SP MAE 약 4.2 mmHg, DP MAE 약 4.5 mmHg로 요약된다. 다만 Introduction에는 SP MAE 4.77 mmHg, STD 6.0 mmHg, DP MAE 4.8 mmHg, STD 6.5 mmHg라는 조금 다른 값도 제시되어 있다. 이는 실험 설정 또는 작성 과정에서의 요약 값 차이로 보이며, 제공된 텍스트만으로 어느 값이 최종 대표값인지 완전히 확정하기는 어렵다.

BP level별 specialized model에서는 전체 모델보다 더 큰 성능 향상이 보고된다. Hypotensive model의 경우 sample 수가 68개인 실험에서는 ANN이 14 features 기준으로 SBP STD 1.267 mmHg, MAE 0.668 mmHg, DBP STD 1.295 mmHg, MAE 0.627 mmHg를 달성했다. 그러나 이 결과는 sample 수가 85개 미만으로 매우 작다는 한계가 있다. 이후 mean BP를 70 mmHg 이하로 선택하여 sample 수를 385개로 늘린 경우, ANN은 14 features 기준으로 SBP STD 4.855 mmHg, MAE 3.624 mmHg, DBP STD 3.974 mmHg, MAE 3.0093 mmHg를 보였다.

Normal BP model에서는 여러 설정이 제시된다. sample 수 215개인 ANN Bayesian 설정에서 14 features를 사용했을 때 SBP STD 3.224 mmHg, MAE 2.152 mmHg, DBP STD 3.508 mmHg, MAE 2.10 mmHg를 달성했다. 논문은 normal BP 영역에서 전체 모델 대비 SBP STD가 약 72%, DBP STD가 약 52% 향상되었다고 설명한다.

Pre-high BP model에서는 sample 수 600개인 SVM이 14 features 기준으로 SBP STD 4.483 mmHg, MAE 3.428 mmHg, DBP STD 4.94 mmHg, MAE 3.189 mmHg를 보인다. 또 다른 ANN 설정에서는 sample 수 444개에서 SBP STD 5.474 mmHg, MAE 4.098 mmHg, DBP STD 3.904 mmHg, MAE 2.9828 mmHg를 보인다. 본문에는 pre-high BP에서 STD error가 SBP와 DP 모두 약 51% 또는 52% 개선되었다고 설명되지만, 구체 수치와 표의 여러 설정이 혼재되어 있어 어떤 모델을 최종 대표 모델로 선택했는지는 다소 불명확하다.

Hypertensive model에서는 high BP 영역의 비선형성을 다루기 위해 별도 모델을 학습한다. Table 6에는 여러 ANN 및 SVM 설정이 제시된다. 본문에서는 Bayesian algorithm과 14 features를 사용한 경우 SBP STD 6.445 mmHg, MAE 3.778 mmHg, DBP STD 4.809 mmHg, MAE 3.656 mmHg가 best result라고 설명한다. 그러나 Table 6에는 ANN Bay. 설정에서 SBP STD 5.2964 mmHg, MAE 2.9747 mmHg, DBP STD 5.6196 mmHg, MAE 3.4659 mmHg도 제시된다. 이처럼 hypertensive 결과에서는 본문 설명과 표의 최저 수치가 완전히 일치하지 않는 부분이 있다. 제공된 텍스트만으로는 어떤 기준에서 best result를 선택했는지 명확하지 않다.

문헌 비교에서는 제안 방법이 기존 연구보다 개선되었다고 주장한다. Table 8에서는 Kachuee et al. 2017의 SP STD 16.17 mmHg, MAE 12.38 mmHg, DP STD 8.45 mmHg, MAE 6.34 mmHg와 비교하여, proposed work가 SP STD 5.059 mmHg, MAE 4.235 mmHg, DP STD 6.373 mmHg, MAE 4.81 mmHg를 달성했다고 제시된다. 또한 MAP에 대해서는 STD 4.45 mmHg, MAE 3.02 mmHg로 보고된다. 이 비교에서 proposed work는 SP와 DP 모두에서 큰 폭의 개선을 보인다.

실시간 구현도 논문에 포함되어 있다. 저자들은 hardware-in-the-loop, 즉 HIL configuration을 구성하여 PPG sensor로 신호를 측정하고, microcontroller와 Wi-Fi module을 통해 computer로 데이터를 전송한 뒤, 제안 feature를 추출하고 모델에 입력하는 실험을 수행했다. 결과는 Omron BP device의 cuff measurement와 비교되었다. 다만 제공된 텍스트에서는 real-time implementation의 정량적 성능 수치가 충분히 자세히 제시되어 있지 않다. 따라서 이 부분은 모델이 실제 환경에서 구현 가능하다는 시연으로 볼 수 있지만, 임상적 정확도 검증으로 보기에는 정보가 부족하다.

## 5. 강점, 한계

이 논문의 첫 번째 강점은 ECG 없이 단일 PPG signal만으로 BP estimation을 수행하려는 단순한 hardware 구성이다. 기존 PTT 또는 PAT 기반 방법은 ECG와 PPG를 동시에 측정해야 하며, sensor synchronization, filtering delay, propagation delay 계산이 성능에 큰 영향을 미친다. 이 논문은 PPG만을 사용하기 때문에 구현이 간단하고, wearable device나 저비용 health monitoring system으로 확장하기 쉽다.

두 번째 강점은 feature crossing을 통해 단순 PPG feature의 표현력을 높였다는 점이다. LASI는 arterial stiffness와 관련된 feature이며, 이를 HR, AI, area features와 곱함으로써 혈압과 관련된 비선형적 상호작용을 feature level에서 표현하려고 했다. 결과적으로 14 crossed features가 7 features보다 더 좋은 성능을 보인다는 점은 이 feature engineering 전략이 어느 정도 효과적임을 보여준다.

세 번째 강점은 혈압 범위별로 별도 모델을 학습하는 구조이다. 전체 BP range를 하나의 모델이 처리하면 high BP 영역의 non-linearity 또는 low BP 영역의 data imbalance로 인해 오차가 커질 수 있다. 논문은 coarse classifier model과 range-specific regression model을 결합하여 이 문제를 줄이려 했다. 특히 normal, hypotensive, hypertensive 영역에서 STD가 기존 문헌 대비 크게 개선되었다는 결과를 제시한다.

네 번째 강점은 여러 regression algorithm을 폭넓게 비교했다는 점이다. Linear regression, SVM, tree-based model, ensemble, Gaussian process regression, ANN 등 다양한 방법이 동일한 feature set에서 비교되었다. 이를 통해 ANN with Bayesian algorithm이 정확도와 training time의 균형에서 유리하다는 결론을 제시한다.

하지만 한계도 상당히 명확하다. 첫째, 데이터 분할 방식이 subject-independent인지 명확하지 않다. MIMIC II 데이터에서 같은 환자 또는 유사한 waveform segment가 train과 test에 동시에 포함되면, 모델 성능이 과대평가될 수 있다. 논문은 arbitrary split을 사용했다고만 설명하므로, 환자 단위 분리 검증이 이루어졌는지는 확인할 수 없다.

둘째, sample filtering 기준이 충분히 재현 가능하게 설명되어 있지 않다. “unacceptable BP”, “unacceptable heart rate”, “extreme distraction” 같은 표현은 있지만, 구체적 threshold나 자동 판별 기준이 제시되지 않는다. 실제 wearable environment에서는 PPG signal quality가 크게 변동하므로, 어떤 신호를 제거하고 어떤 신호를 사용할지의 기준이 매우 중요하다.

셋째, BP level별 모델에서 일부 그룹의 sample 수가 작다. 특히 hypotensive의 68 samples 결과는 매우 낮은 오차를 보이지만, sample 수가 적어 일반화 성능을 판단하기 어렵다. 논문도 이 한계를 인정하고 sample 수를 385개로 늘린 재실험을 언급한다. 하지만 여전히 저혈압 영역의 충분한 다양성을 확보했는지는 명확하지 않다.

넷째, 결과 값이 본문, 초록, 표 사이에서 일부 일관되지 않다. 예를 들어 초록에서는 SP MAE 약 4.2 mmHg, DP MAE 약 4.5 mmHg로 제시되지만, Introduction에서는 SP MAE 4.77 mmHg, DP MAE 4.8 mmHg가 언급된다. Table 8에서는 proposed work의 DP MAE가 4.81 mmHg로 나타난다. Hypertensive best result도 본문 설명과 Table 6의 수치가 완전히 명확하게 정렬되지 않는다. 이는 논문 해석과 재현성 측면에서 아쉬운 부분이다.

다섯째, real-time implementation은 포함되어 있지만, Omron device와의 비교에서 충분한 정량 결과가 제공되지 않는다. HIL setup과 PPG sensor, microcontroller, Wi-Fi transmission은 실제 구현 가능성을 보여주지만, real-time 환경에서의 MAE, STD, delay, robustness, subject 수가 명확히 제시되지 않는다. 따라서 실시간 성능 검증은 예비적 demonstration에 가깝다.

여섯째, calibration-free라는 주장은 추가 검증이 필요하다. 논문은 individual calibration을 요구하지 않는다고 주장하지만, 데이터가 MIMIC II 기반이고, 모델이 특정 데이터 분포에서 학습되었다. 새로운 sensor, 새로운 population, 새로운 measurement condition에서도 calibration 없이 동일 성능을 내는지는 제공된 텍스트만으로 확인할 수 없다.

비판적으로 보면, 이 논문은 복잡한 deep learning architecture보다는 feature engineering과 data partitioning으로 성능을 끌어올리는 연구에 가깝다. 방법 자체는 실용적이고 간단하지만, 평가 설계가 subject-independent generalization을 충분히 입증하지 못한다면 실제 임상 적용 가능성은 제한적으로 해석해야 한다. 특히 MIMIC II segment 기반 random split, manual filtering, BP range별 sample imbalance는 후속 연구에서 더 엄격히 다루어야 한다.

## 6. 결론

이 논문은 PPG signal만을 이용한 cuffless, calibration-free BP estimation 방법을 제안했다. 제안 방법은 PPG waveform에서 HR, AI, LASI, IPA 관련 area features를 추출하고, LASI와 다른 feature를 곱한 crossed features를 추가한 뒤, ANN 및 여러 machine learning regression model을 사용하여 SP와 DP를 추정한다. 또한 전체 BP range를 하나의 모델로 처리하지 않고, hypotensive, normal, pre-high, hypertensive range별로 별도 모델을 학습하여 정확도를 높이는 coarse-to-fine estimation 구조를 제안한다.

주요 결과로는 ANN with Bayesian algorithm과 14 crossed features를 사용했을 때 전체 데이터에서 SP STD 약 5.296 mmHg, MAE 4.235 mmHg, DP STD 약 6.373 mmHg, MAE 4.457 mmHg를 달성했다. 이는 ISO 기준에서 요구하는 평균 오차 5 mmHg 이하 및 표준편차 8 mmHg 이하 조건을 만족한다고 제시된다. 또한 BHS standard 기준에서도 SP와 DP 모두 Grade A를 달성했다. BP level별 모델을 사용하면 특정 range에서 추가적인 성능 향상이 가능하며, 논문은 특히 low, normal, high BP 영역에서 기존 문헌 대비 큰 폭의 STD 개선을 보고한다.

이 연구의 주요 기여는 단일 PPG sensor 기반 혈압 추정에서 LASI-crossed feature engineering과 BP level별 specialized regression model이 성능을 향상시킬 수 있음을 보인 것이다. 이는 ECG 없이 간단한 optical sensor만으로 continuous BP monitoring을 구현하려는 wearable healthcare 연구에 실용적 시사점을 제공한다.

다만 실제 적용을 위해서는 더 엄격한 subject-independent validation, 명확한 signal quality filtering 기준, 더 큰 hypotensive dataset, real-time implementation의 정량 평가, 외부 데이터셋 검증이 필요하다. 특히 calibration-free wearable BP estimation을 주장하려면 다양한 사용자, 다양한 센서, 다양한 활동 조건에서 모델이 안정적으로 동작하는지 검증해야 한다. 그럼에도 이 논문은 PPG-only BP estimation에서 feature crossing과 range-specific modeling이 유용한 방향임을 보여주는 실용적 연구로 평가할 수 있다.
