# Machine Learning Algorithm for Non-invasive Blood Pressure Estimation Using PPG Signals

* **저자**: Gengjia Zhang, Siho Shin, Jaehyo Jung, Meina Li, Youn Tae Kim
* **발표연도**: 2022

## 1. 논문 개요

이 논문은 photoplethysmography, 즉 PPG 신호만을 이용하여 cuff 없이 systolic blood pressure, 즉 SBP와 diastolic blood pressure, 즉 DBP를 추정하는 machine learning 기반 혈압 추정 알고리즘을 제안한다. 제안 방법의 핵심 모델은 Gradient Boosting Regressor, 즉 GBR이며, MIMIC II database에서 얻은 PPG 데이터를 일정 길이의 구간으로 나누고, 각 구간에서 feature를 추출한 뒤, 이 feature를 이용해 SBP와 DBP를 예측한다.

논문의 연구 문제는 비교적 간단하고 명확하다. 기존 cuff 기반 혈압 측정은 병원에서 널리 사용되고 정확도가 높지만, cuff가 팔을 압박하기 때문에 노약자나 취약 환자에게 부담을 줄 수 있고, 일상생활에서 주기적·장기적으로 측정하기 어렵다. 또한 기존 deep learning 기반 혈압 추정 알고리즘은 높은 계산량과 memory usage를 요구할 수 있어, 장시간 동작해야 하는 healthcare device에는 적합하지 않을 수 있다고 논문은 문제를 제기한다. 이에 따라 본 연구는 복잡한 deep learning 대신 상대적으로 계산 부담이 낮은 machine learning regressor를 사용하여 PPG 기반 혈압 추정 가능성을 확인하려 한다.

문제의 중요성은 hypertension 관리와 관련된다. 고혈압은 stroke, heart failure, heart attack, kidney disease 등 다양한 심혈관 및 신장 질환의 위험 요인이며, 증상이 뚜렷하지 않은 경우가 많기 때문에 정기적인 혈압 확인이 중요하다. 특히 고령자는 질병에 취약하고 혈압 관리 필요성이 크므로, cuffless, non-invasive, lightweight한 혈압 추정 알고리즘은 개인 건강관리 및 wearable healthcare system에 유용할 수 있다.

논문은 MIMIC II database에서 PPG signal을 가져와 전처리하고, NeuroKit2를 이용해 heart rate 관련 feature를 추출한 뒤, Gradient Boosting Regressor로 SBP와 DBP를 예측한다. 성능은 $R^2$, MSE, MAE, training time으로 평가되며, Random Forest Regressor와 Decision Tree Regressor가 비교 대상으로 사용된다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 PPG signal에서 복잡한 deep learning feature를 직접 학습하기보다, PPG waveform을 일정 구간으로 나누고 그 구간에서 heart rate와 waveform-related feature를 추출한 뒤, Gradient Boosting Regressor로 혈압을 추정하는 것이다. Gradient Boosting Regressor는 여러 decision tree를 순차적으로 학습하면서 이전 모델의 예측 오차를 다음 모델이 보정하는 ensemble learning 방식이다. 따라서 단일 decision tree보다 더 강한 regression 성능을 낼 수 있고, deep neural network보다 구현과 학습이 비교적 간단할 수 있다.

기존 연구 중 상당수는 ECG와 PPG를 함께 사용하여 pulse transit time, pulse arrival time 등의 feature를 계산한다. 그러나 ECG와 PPG를 동시에 사용하려면 두 신호의 동기화, sampling rate 정합, 추가 sensor 부착이 필요하다. 논문은 이러한 계산 복잡도와 memory usage 증가를 문제로 보고, PPG만 사용하는 단순한 구조를 제안한다. 이 점이 기존 ECG+PPG 기반 방법과의 주요 차별점이다.

또 다른 설계 아이디어는 PPG를 일정 길이로 segmentation하여 학습 데이터를 확보하는 것이다. 논문은 PPG 데이터를 2초 간격으로 나누고, 각 segment가 적어도 하나의 cardiac cycle을 포함하도록 random selection을 적용했다고 설명한다. 이 방식은 제한된 recording으로부터 더 많은 학습 sample을 만들기 위한 전략으로 이해할 수 있다.

다만 이 논문은 deep learning 모델처럼 raw PPG waveform에서 end-to-end로 SBP와 DBP를 추정하는 접근은 아니다. NeuroKit2를 이용해 peak를 검출하고, peak 간 interval, heart rate, minimum/maximum heart rate, standard deviation, curvature 등 hand-crafted feature를 계산한 뒤, 이 feature를 machine learning regressor에 입력한다. 따라서 이 연구는 feature engineering 기반 machine learning approach에 속한다.

## 3. 상세 방법 설명

논문의 전체 파이프라인은 data collection, preprocessing, feature extraction, BP estimation, model evaluation의 네 단계로 구성된다. 먼저 MIMIC II database에서 PPG 데이터를 수집하고, signal filtering과 segmentation을 수행한다. 이후 NeuroKit2를 사용하여 PPG peak와 heart rate 관련 feature를 추출한다. 마지막으로 Gradient Boosting Regressor를 이용해 SBP와 DBP를 예측하고, Random Forest 및 Decision Tree와 성능을 비교한다.

### 3.1 데이터베이스

연구에 사용된 데이터는 MIMIC II database이다. MIMIC II는 ICU patient monitor에서 수집된 physiological signals와 vital sign time series, 그리고 병원 medical information system에서 얻은 clinical data를 포함하는 공개 데이터베이스이다. 논문에 따르면 데이터 sampling rate는 125 Hz이다.

논문은 12,000 data points 중 2,842 samples를 random selection하였다고 설명한다. 이후 dataset은 training과 validation으로 8:2 비율로 분할되었다. 다만 제공된 텍스트만으로는 이 2,842 samples가 subject 단위인지, segment 단위인지, recording 단위인지 명확하지 않다. 또한 subject-independent split인지 segment-level random split인지도 명시되어 있지 않다. 이 점은 성능 해석에서 중요한 한계이다.

### 3.2 PPG signal preprocessing

PPG data는 2초 간격으로 나뉜다. 논문은 모든 segment가 최소 하나의 cardiac cycle을 포함하도록 random selection을 적용했다고 설명한다. 이후 SBP와 DBP를 추출한다고 되어 있는데, 제공된 텍스트에는 “SBP and DBP are extracted from the PPG using the maximum and minimum values of the signal”이라고 되어 있다. 일반적으로 SBP와 DBP의 ground truth는 ABP waveform이나 cuff measurement에서 얻어야 하며, PPG의 최대·최소값을 곧바로 SBP와 DBP로 볼 수는 없다. 따라서 이 부분은 논문 서술이 부정확하거나, 실제로는 ABP label을 사용했으나 추출 텍스트에서 PPG로 잘못 표현되었을 가능성이 있다. 제공된 텍스트만으로는 ground truth label 생성 방식이 충분히 명확하지 않다.

PPG에는 breathing 등에 의한 baseline noise가 포함될 수 있다. 논문은 이 noise가 혈압 추정 정확도를 낮출 수 있으므로 filtering을 수행한다고 설명한다. Sampling frequency는 125 Hz이고 cutoff frequency는 40 Hz로 설정되었다. 텍스트에서는 high pass filter라고도 표현되어 있으나, cutoff frequency 40 Hz와 PPG smoothing 설명을 함께 고려하면 filter type이 정확히 무엇인지 명확하지 않다. 논문은 filtering 후 signal이 flat and smooth해져 혈압 추정 정확도 향상에 유리하다고 설명한다.

### 3.3 Feature extraction

Feature extraction에는 NeuroKit2가 사용되었다. NeuroKit2는 neurophysiological signal processing을 위한 Python toolbox이며, 이 논문에서는 PPG peak detection과 heart rate 관련 parameter 계산에 사용된다.

먼저 PPG에서 peak point를 검출하고, 인접한 peak 사이의 interval을 계산한다. Heart rate는 단위 시간 내 peak 수 또는 평균 peak interval을 기반으로 계산된다. 논문에서 heart rate 계산식은 다음과 같다.

$$
HR=\frac{60}{period}
$$

여기서 $period$는 peak 사이 평균 거리, 즉 average distance between peaks를 의미한다. 만약 $period$가 초 단위라면 이 식은 분당 박동수, beats per minute을 계산하는 표준적인 방식이다.

논문은 heart rate 외에도 minimum heart rate, maximum heart rate, standard deviation, curvature 등을 feature로 계산한다고 설명한다. 그러나 feature vector의 전체 차원, 각 feature의 정확한 정의, curvature 계산 방식, feature normalization 여부는 제공된 텍스트에서 구체적으로 제시되지 않는다. 따라서 방법론 재현성 측면에서는 feature extraction 설명이 제한적이다.

### 3.4 Gradient Boosting Regressor 기반 혈압 추정

제안 모델은 Gradient Boosting Regressor이다. GBR은 여러 regression tree를 순차적으로 학습하는 ensemble model이다. 첫 번째 tree가 예측을 수행한 뒤, 다음 tree는 이전 모델이 남긴 residual error를 줄이는 방향으로 학습된다. 이 과정이 반복되면서 전체 모델은 점진적으로 오차를 보정한다.

논문은 GBR이 여러 decision tree를 이용해 하나의 강력한 모델을 생성한다고 설명한다. Hyperparameter는 sklearn package와 GridSearchCV를 사용해 조정하였다. 최종 설정은 learning rate 0.01, boosting value 100, regression estimator의 maximum depth 3이다. 여기서 boosting value는 일반적으로 number of estimators, 즉 boosting stage 수를 의미하는 것으로 해석된다.

비교 대상으로는 Random Forest Regressor와 Decision Tree Regressor가 사용되었다. Decision Tree는 단일 tree 기반 모델이고, Random Forest는 여러 decision tree를 병렬적으로 학습하여 평균을 내는 bagging ensemble 방식이다. GBR은 순차적으로 residual을 보정하는 boosting ensemble이므로, Random Forest와 학습 방식이 다르다.

### 3.5 평가 지표

모델 성능은 $R^2$, MAE, MSE, training time으로 평가된다. 논문에서 $R^2$는 다음과 같이 제시되어 있다.

$$
R^2=1-\frac{\sum_{i=1}^{n}(\hat{y}*i-\bar{y})^2}{\sum*{i=1}^{n}(y_i-\bar{y})^2}
$$

다만 일반적인 coefficient of determination의 정의는 예측 오차 제곱합 $\sum_i(y_i-\hat{y}_i)^2$를 분자에 사용한다. 제공된 텍스트의 식은 $\hat{y}_i-\bar{y}$를 분자에 사용하고 있어 일반적인 $R^2$ 식과 다르다. 이는 논문 추출 과정의 오류이거나 원문 식 표기의 오류일 수 있다. 성능표에 제시된 $R^2$ 값은 일반적인 sklearn의 $R^2$ score를 사용했을 가능성이 있지만, 제공된 텍스트만으로는 확인할 수 없다.

MAE는 다음과 같이 제시된다.

$$
MAE=\frac{\sum_{i=1}^{n}|y_i-\hat{y}_i|}{n}
$$

MAE는 실제값과 예측값 사이의 절대 오차 평균이며, 혈압 추정에서 직관적으로 해석하기 쉬운 지표이다.

MSE는 텍스트에서 다음 형태로 제시된다.

$$
MSE=\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)}{n}
$$

그러나 일반적인 MSE는 오차를 제곱해야 하므로 다음과 같이 정의된다.

$$
MSE=\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{n}
$$

따라서 제공된 텍스트의 MSE 식은 제곱이 누락된 것으로 보인다. 또한 논문에서 MSE 단위를 mmHg로 표기하고 있는데, 엄밀하게는 MSE의 단위는 $(mmHg)^2$가 되어야 한다. 표의 값이 실제 MSE인지 RMSE인지도 제공된 텍스트만으로는 완전히 명확하지 않다.

## 4. 실험 및 결과

실험에서는 제안한 GBR 모델을 Random Forest Regressor 및 Decision Tree Regressor와 비교하였다. 평가 대상은 SBP와 DBP 각각에 대한 regression 성능이며, 사용 지표는 $R^2$, MSE, MAE, training time이다.

제공된 표에 따르면 DBP 예측에서 Gradient Boosting Regressor는 $R^2=0.58$, MSE 4.18 mmHg, MAE 2.54 mmHg, training time 0.27초를 기록하였다. Random Forest Regressor는 $R^2=0.57$, MSE 4.42 mmHg, MAE 3.56 mmHg, training time 0.74초였고, Decision Tree Regressor는 $R^2=0.38$, MSE 6.03 mmHg, MAE 2.99 mmHg, training time 0.022초였다. DBP에서는 GBR이 $R^2$, MSE, MAE 측면에서 전반적으로 가장 좋은 성능을 보인다. Decision Tree는 학습 시간은 가장 짧지만 $R^2$가 낮고 MSE가 크다.

SBP 예측에서는 Gradient Boosting Regressor가 $R^2=0.87$, MSE 7.07 mmHg, MAE 4.33 mmHg, training time 0.46초를 기록하였다. Random Forest Regressor는 $R^2=0.87$, MSE 7.65 mmHg, MAE 4.2 mmHg, training time 1.36초였고, Decision Tree Regressor는 $R^2=0.80$, MSE 9.29 mmHg, MAE 4.78 mmHg, training time 0.013초였다. SBP에서는 GBR과 Random Forest의 $R^2$가 동일하게 0.87이며, MSE는 GBR이 더 낮고 MAE는 Random Forest가 약간 더 낮다. 논문은 종합적으로 GBR이 가장 좋은 성능을 보인다고 해석한다.

다만 abstract에는 “MSE of SBP is 7.07 mmHg, MAE is 4.33 mmHg, and $R^2$ is 0.58. In addition, MSE of DBP is 4.18 mmHg, MAE is 2.54 mmHg, and $R^2$ is 0.87”이라고 되어 있어, 표의 $R^2$ 값과 SBP/DBP가 서로 바뀐 것으로 보인다. 표에서는 SBP의 $R^2$가 0.87이고 DBP의 $R^2$가 0.58이다. 일반적으로 표가 더 구체적인 결과를 제공하므로, 보고서에서는 표 기준으로 해석하는 것이 더 타당하지만, 논문 내부에 수치 불일치가 존재한다는 점은 명확한 한계이다.

논문은 GBR이 기존 혈압 예측 모델보다 SBP의 MSE를 1.85 mmHg, DBP의 MSE를 2.22 mmHg 개선했다고 설명한다. 그러나 어떤 “existing blood pressure prediction model”을 직접 기준으로 삼았는지는 문맥상 비교 모델 중 하나인지, 선행 연구인지 명확하지 않다. 표 기준으로 보면 Decision Tree 대비 SBP MSE는 9.29에서 7.07로 2.22 감소하고, DBP MSE는 6.03에서 4.18로 1.85 감소한다. 논문 문장에서는 SBP와 DBP의 개선량이 서로 바뀐 것으로 보인다.

기존 연구와의 비교에서는 ECG+PPG 기반 Random Forest, PPG 기반 PTT-CP Analysis, PPG 기반 SVR, ECG+PPG 기반 SVM, ECG+PPG 기반 AdaBoost 등과 비교하였다. 제안 방법은 PPG만 사용하면서 SBP MAE 4.33 mmHg, DBP MAE 2.54 mmHg를 기록하여 표에 제시된 모든 prior work보다 낮은 MAE를 보였다. 예를 들어 ECG+PPG Random Forest 연구는 SBP MAE 9.54 mmHg, DBP MAE 5.48 mmHg였고, PPG 기반 SVR 연구는 SBP MAE 8.54 mmHg, DBP MAE 4.34 mmHg였다. 논문은 이를 근거로 제안 방법이 성능과 계산 시간 측면에서 우수하다고 주장한다.

그러나 이 비교는 주의해서 해석해야 한다. 선행 연구들은 서로 다른 dataset, preprocessing, subject split, evaluation protocol을 사용했을 가능성이 높다. 제공된 텍스트에는 동일한 조건에서 재실험했다는 설명이 없다. 따라서 Table III의 비교는 대략적인 참고로 볼 수 있지만, 엄밀한 benchmark comparison으로 보기는 어렵다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 단순하고 계산량이 낮은 machine learning approach로 PPG 기반 혈압 추정을 수행하려 했다는 점이다. Deep learning 모델은 높은 성능을 낼 수 있지만, wearable device나 long-term healthcare product에서는 memory usage, computational cost, battery consumption이 문제가 될 수 있다. GBR은 상대적으로 가볍고 sklearn 기반 구현이 쉬우며, 이 논문에서는 Random Forest보다 짧은 training time을 보였다.

두 번째 강점은 ECG 없이 PPG만 사용한다는 점이다. ECG와 PPG를 함께 사용하면 PTT나 PAT 같은 생리적 feature를 계산할 수 있지만, sensor 수가 늘어나고 synchronization 문제가 발생한다. PPG-only 방식은 optical wearable sensor 하나로 구현 가능성이 있으므로, cuffless healthcare device 측면에서 실용성이 있다.

세 번째 강점은 GBR, Random Forest, Decision Tree를 같은 데이터 조건에서 비교했다는 점이다. 세 모델 모두 tree-based regression 계열이지만, 단일 tree, bagging ensemble, boosting ensemble이라는 차이를 갖는다. 실험 결과 GBR은 전반적으로 좋은 accuracy와 상대적으로 짧은 training time을 보여, 이 문제에서 boosting 방식의 유효성을 확인했다.

네 번째 강점은 NeuroKit2를 이용하여 PPG peak 및 heart rate 관련 feature를 추출한 점이다. 공개 toolbox를 사용하면 peak detection과 feature extraction 구현의 신뢰성과 재현 가능성을 어느 정도 높일 수 있다. 다만 실제 feature list가 충분히 상세하게 제시되지는 않았다.

반면 한계는 상당히 명확하다. 첫째, ground truth label 생성 방식이 불명확하다. 논문은 SBP와 DBP를 PPG의 maximum 및 minimum value에서 추출한다고 설명하는데, 이는 생리학적으로 타당하지 않다. 혈압 label은 ABP waveform 또는 cuff measurement에서 얻어야 한다. 만약 실제 구현에서는 ABP를 사용했지만 텍스트에 PPG로 잘못 표기된 것이라면 단순 서술 오류이지만, 그렇지 않다면 방법론에 심각한 문제가 된다. 제공된 추출 텍스트만으로는 이를 확인할 수 없다.

둘째, 수식과 결과표에 오류 또는 불일치가 있다. Abstract의 SBP/DBP $R^2$ 값은 표와 서로 바뀐 것으로 보이며, MSE 개선량 설명도 표와 맞지 않는다. 또한 MSE 식에는 제곱항이 빠져 있고, $R^2$ 식도 일반적인 정의와 다르게 표기되어 있다. 이러한 오류는 논문의 신뢰성과 재현성에 부정적인 영향을 준다.

셋째, dataset split이 subject-independent인지 명확하지 않다. PPG segment를 2초 단위로 나누고 random selection한 뒤 8:2로 training/validation split을 수행했다면, 같은 subject 또는 같은 recording에서 나온 segment가 training과 validation에 동시에 포함될 수 있다. 이 경우 모델이 subject-specific pattern을 학습하여 validation 성능이 과대평가될 수 있다. 혈압 추정 연구에서는 subject-independent validation이 매우 중요하므로, 이 부분의 부재는 큰 한계이다.

넷째, 데이터 규모가 제한적이다. 논문은 12,000 data points 중 2,842 samples를 random selection했다고 설명한다. 이는 deep learning 연구와 비교하면 작을 수 있고, tree-based model에서도 다양한 혈압 범위와 subject variability를 충분히 반영하기 어려울 수 있다. 또한 선택된 sample의 혈압 분포, age distribution, disease status, hypertension/hypotension 비율이 제공되지 않는다.

다섯째, feature extraction 설명이 부족하다. Heart rate, minimum/maximum heart rate, standard deviation, curvature 등을 계산했다고 하지만, 전체 feature set의 차원, 각 feature의 수학적 정의, feature scaling 방식, missing value 처리, peak detection 실패 시 처리 방법이 명확하지 않다. 이는 다른 연구자가 같은 방법을 재현하기 어렵게 만든다.

여섯째, filtering 설명도 모호하다. 텍스트에는 high pass filter로 보이는 설명과 cutoff frequency 40 Hz가 함께 등장하며, filtering 후 signal이 flat and smooth해진다고 설명한다. PPG에서 일반적으로 중요한 주파수 대역과 baseline drift 제거 목적을 고려하면 filter design이 더 명확히 제시되어야 한다. Filter order, type, passband, stopband가 충분히 설명되지 않는다.

마지막으로, 임상적 기준 평가가 없다. 논문은 MAE, MSE, $R^2$, time을 평가하지만, AAMI 또는 BHS 기준과 같은 혈압 측정기 평가 standard를 적용하지 않는다. 또한 Bland–Altman plot, mean error와 standard deviation, error distribution, hypertension classification 성능 등이 제공되지 않는다. 따라서 실제 blood pressure monitoring device로서의 임상적 적합성은 이 논문만으로 판단하기 어렵다.

## 6. 결론

이 논문은 PPG signal만을 사용하여 non-invasive blood pressure estimation을 수행하는 Gradient Boosting Regressor 기반 machine learning 알고리즘을 제안하였다. MIMIC II database에서 얻은 PPG 데이터를 2초 단위로 segmentation하고, NeuroKit2를 사용하여 peak와 heart rate 관련 feature를 추출한 뒤, GBR을 이용해 SBP와 DBP를 예측하였다.

실험 결과, 표 기준으로 GBR은 DBP에서 $R^2=0.58$, MSE 4.18 mmHg, MAE 2.54 mmHg를 기록했고, SBP에서 $R^2=0.87$, MSE 7.07 mmHg, MAE 4.33 mmHg를 기록하였다. Random Forest 및 Decision Tree와 비교했을 때 GBR은 전반적으로 낮은 MSE와 짧은 training time을 보여, 간단한 tree-based boosting model이 PPG 기반 혈압 추정에 유용할 수 있음을 보였다. 또한 prior work와의 비교표에서는 제안 방법이 PPG만 사용하면서도 더 낮은 MAE를 달성한 것으로 제시된다.

이 연구의 주요 기여는 복잡한 ECG+PPG 조합이나 deep learning 구조 없이, PPG-only feature와 Gradient Boosting Regressor만으로 비교적 좋은 혈압 추정 성능을 보고했다는 점이다. 이는 저전력, 저메모리 healthcare device 또는 초기 wearable prototype에서 실용적인 baseline으로 활용될 수 있다.

그러나 논문의 방법론적 명확성은 제한적이다. Ground truth SBP/DBP 생성 방식, subject-independent split 여부, feature 정의, filtering 설계, MSE 및 $R^2$ 수식, abstract와 table 간 수치 불일치가 모두 명확히 해결되어야 한다. 특히 PPG의 최대·최소값으로 SBP와 DBP를 추출했다는 서술은 생리학적으로 타당하지 않으므로, 실제 label이 ABP에서 추출되었는지 확인이 필요하다. 또한 AAMI/BHS 기준, Bland–Altman analysis, 외부 validation, 다양한 혈압 범위에서의 성능 검증이 부족하다.

종합하면, 이 논문은 PPG 기반 cuffless blood pressure estimation을 위한 간단하고 빠른 machine learning baseline으로서 의미가 있다. 하지만 학술적·임상적으로 강한 결론을 내리기 위해서는 데이터 구성과 label 정의를 명확히 하고, subject-independent 검증 및 표준 임상 평가 기준을 추가해야 한다. 향후 연구에서는 ECG feature 추가, 더 큰 subject-level dataset, robust preprocessing, external validation을 통해 모델의 신뢰성과 실제 적용 가능성을 높일 필요가 있다.
