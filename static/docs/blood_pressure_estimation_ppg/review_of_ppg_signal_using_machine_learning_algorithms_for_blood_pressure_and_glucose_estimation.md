# Review of PPG signal using Machine Learning Algorithms for Blood Pressure and Glucose Estimation

* **저자**: R Gayathri Priyadarshini, M Kalimuthu, S Nikesh, M Bhuvaneshwari
* **발표연도**: 2021

## 1. 논문 개요

이 논문은 Photoplethysmography, 즉 PPG 신호를 이용하여 blood pressure와 blood glucose level을 non-invasive 방식으로 추정하는 machine learning 기반 연구들을 검토하고, PPG feature와 여러 regression algorithm의 활용 가능성을 정리한 review 성격의 논문이다. 논문 제목과 초록은 review paper임을 명확히 밝히지만, 본문에는 특정 workflow, population statistics, algorithm performance table, Clarke error grid analysis 결과도 포함되어 있어 단순 문헌 요약뿐 아니라 PPG 기반 BP 및 BGL 추정 방법의 일반적 파이프라인을 제시하려는 목적도 가진다.

논문의 주요 관심 대상은 systolic blood pressure, diastolic blood pressure, blood glucose level이다. 저자들은 일반적인 정상 범위로 BGL 70-100 mg/dl, BP 120/80 mmHg를 언급하고, hypertension과 diabetes 관련 지표를 지속적으로 모니터링할 필요성을 강조한다. 기존 혈당 측정은 손가락을 찌르는 invasive method가 일반적이어서 통증과 감염 위험이 있고, 기존 혈압 측정은 cuff 기반 측정이 불편하며 연속 측정에 적합하지 않다. 따라서 저자들은 PPG가 저비용, 비침습, 연속 모니터링 가능성 측면에서 중요한 대안이 될 수 있다고 본다.

연구 문제는 PPG signal에서 추출한 waveform features를 이용해 BP와 BGL을 어느 정도 정확하게 추정할 수 있는지, 그리고 이를 위해 어떤 machine learning regression algorithms가 유리한지 검토하는 것이다. 논문은 특히 linear regression보다 nonlinear regression algorithms가 PPG와 생리 지표 사이의 복잡한 관계를 더 잘 설명한다고 주장한다. Decision tree regression, support vector machine, random forest regression, neural network 등이 언급되며, 성능 비교에서는 random forest가 BGL, SBP, DBP regression에서 가장 높은 $R^2$를 보이는 것으로 제시된다.

이 문제의 중요성은 wearable healthcare와 continuous monitoring에 있다. PPG는 pulse oximeter, finger sensor, wrist wearable 등에 쉽게 적용될 수 있으며, 하드웨어가 비교적 간단하다. 만약 PPG만으로 혈압과 혈당을 안정적으로 추정할 수 있다면, 고혈압 환자, 당뇨 환자, 임신 중 혈압 관리가 필요한 사람, 장기 건강 모니터링이 필요한 사용자에게 실용적 가치가 크다. 다만 이 논문은 review 성격이 강하고, 자체 실험 설계와 데이터 출처가 제한적으로 설명되어 있어, 실제 임상적 신뢰성을 주장하기에는 근거가 충분하지 않은 부분도 존재한다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 PPG waveform이 혈관의 blood volume change를 반영하므로, 그 형태적 특징을 분석하면 BP와 BGL 관련 생리 정보를 추정할 수 있다는 것이다. PPG는 피부에 빛을 조사하고, 혈액량 변화에 따른 light absorption 변화를 측정하는 optical sensing method이다. 심장 박동에 따라 혈관이 팽창하고 수축하면 빛의 흡수량이 달라지고, 이 변화가 PPG waveform으로 기록된다.

혈압 추정 측면에서 PPG waveform은 systolic peak, diastolic peak, slope, amplitude, frequency, pulse interval, pulse area 등 다양한 특징을 제공한다. 혈압이 혈관 탄성, arterial stiffness, pulse wave propagation과 관련되어 있으므로, PPG waveform의 시간적 및 형태적 변화가 BP와 상관될 수 있다는 것이 기본 직관이다. 논문은 PPG와 ECG를 함께 사용하면 추가적인 parameter를 얻을 수 있다고 언급하지만, PPG 단독 방식이 더 간단하고 wearable sensor로 구현하기 쉽다는 점을 강조한다.

혈당 추정 측면에서는 PPG의 optical absorption 특성과 NIR light를 이용한다. 논문은 glucose가 특정 wavelength의 near-infrared light를 흡수하고, 혈관 변화와 viscosity 변화가 PPG 분석에 영향을 줄 수 있다고 설명한다. 다만 PPG만으로 BGL을 정확히 측정하기는 어렵고, noise artifact contamination이 큰 문제라고 언급한다. 따라서 waveform morphology와 frequency를 함께 고려하고, machine learning 모델로 PPG feature와 BGL 사이의 관계를 학습해야 한다고 본다.

기존 접근 방식과의 차별점으로 논문은 PPG 기반 non-invasive estimation의 가능성과 nonlinear machine learning의 필요성을 강조한다. Linear regression은 PPG signal과 BP 또는 BGL 사이의 복잡한 관계를 잘 설명하지 못하며, nonlinear regression algorithms가 더 적합하다고 주장한다. 특히 random forest regression은 여러 decision tree의 평균을 사용하기 때문에 nonlinear relationship과 feature interaction을 잘 포착할 수 있고, 논문에 제시된 성능 표에서도 가장 높은 $R^2$를 보인다.

다만 이 논문에서 제시하는 핵심 아이디어는 새로운 딥러닝 아키텍처나 새로운 손실 함수라기보다는, PPG feature extraction과 여러 machine learning regression algorithms의 활용 가능성을 검토하는 데 가깝다. 논문 제목에 “Review”가 포함되어 있으며, 제공된 텍스트에는 독창적인 end-to-end neural network 구조나 엄밀한 대규모 실험 설계는 명확히 제시되어 있지 않다.

## 3. 상세 방법 설명

논문에서 제시된 전체 방법론은 PPG signal acquisition, activity detection and signal processing, feature extraction, machine learning regression, 성능 평가로 구성된다. 이 과정은 blood pressure와 blood glucose level을 PPG 기반으로 추정하기 위한 일반적 pipeline으로 설명된다.

먼저 PPG signal은 optical sensor를 통해 수집된다. 기본적인 PPG system은 light emitting diode와 photodetector로 구성된다. LED가 피부 조직에 빛을 조사하면, photodetector가 혈액량 변화에 따른 반사 또는 투과 광량 변화를 감지한다. 혈관 내 혈액량은 심장 박동에 따라 주기적으로 변하므로 PPG waveform은 cardiac pulse와 관련된 주기적 신호를 갖는다.

혈압 추정에서는 fingertip PPG signal이 주로 언급된다. Cardiac cycle이 발생하면 pressure pulse가 손가락의 subcutaneous tissue와 arterioles, arteries에 도달하고, 이로 인한 volume change가 PPG 신호에 반영된다. 논문은 PPG measurement가 arterial tonometer보다 sensor displacement에 덜 민감할 수 있고, cuff sphygmomanometer보다 continuous monitoring에 적합하며, ECG처럼 여러 전극을 부착할 필요가 없다는 장점을 언급한다.

혈당 추정에서는 NIR LED와 photodiode를 이용한 PPG sensor가 언급된다. 논문은 선택된 NIR wavelength가 물에는 상대적으로 투명하고 glucose에 의해 흡수될 수 있다고 설명한다. 이 원리를 이용하면 PPG waveform에서 blood glucose level 관련 정보를 추출할 수 있다는 것이다. 하지만 논문은 기본 PPG signal만으로는 noise artifact 때문에 정확한 BGL 추정이 어렵다고 인정한다.

Signal processing 단계에서는 activity detection module과 signal processing module이 제시된다. Activity detection module의 목적은 noise, signal loss 등으로 corrupted된 parameter를 제거하는 것이다. 논문은 이 모듈이 음성 처리에서 voice activity detection에 사용되는 기법과 유사하다고 설명한다. Finite State Automation, 즉 FSA를 사용해 PPG signal을 세 상태로 나눈다. S1은 spurious 또는 lack of signal 상태, S2는 PPG-in 상태, S3는 PPG-out 상태로 설명된다. FSA가 S2와 S3 상태에서 일정 시간 유지되면 sample을 frame과 variable로 복사하고 적절한 output을 생성한다고 설명한다.

또한 spectral entropy와 Teager energy도 사용된다고 언급된다. Spectral entropy는 signal의 frequency distribution이 얼마나 불규칙하거나 복잡한지를 나타낼 수 있는 지표이다. Teager energy는 신호의 instantaneous energy 성격을 반영하는 특징으로, 생체 신호의 급격한 변화나 활성 구간을 탐지하는 데 활용될 수 있다. Signal processing module에서는 spectral entropy statistics가 중요한 역할을 하며, PPG signal의 correlation과 coherence를 계산하여 수신 신호의 통계적 품질을 평가한다고 설명된다. 다만 구체적인 수식, threshold, 구현 세부 사항은 제공된 텍스트에 충분히 명시되어 있지 않다.

Feature extraction 측면에서 논문은 BP와 BGL 추정에 사용할 수 있는 여러 PPG feature를 언급한다. 혈압과 관련해서는 frequency, amplitude, slope, waveform shape, systolic peak와 diastolic peak 사이의 시간 차이, pulse interval 등이 중요하다고 설명한다. Addison이 제안한 slope transit time, 즉 STT도 언급되며, 이는 systolic waveform의 foot에서 peak까지 상승하는 slope parameter로 BP와 관련될 수 있다고 설명된다.

혈당 추정에서는 pulse area와 pulse interval이 중요하게 언급된다. Pulse area는 cardiac cycle에서 최소점과 최대점 사이의 PPG curve 아래 면적으로 계산되며, trapezoidal integration을 사용해 계산된다고 설명된다. Pulse interval은 PPG waveform의 시작과 끝의 거리 또는 consecutive systolic peaks 사이의 interval로 측정된다. 이후 이러한 feature들의 mean value가 machine learning model의 입력으로 사용된다.

논문에 포함된 study population statistics는 glucose, SBP, DBP에 대해 제공된다. Glucose는 최소 49 mg/dl, 최대 393 mg/dl, 평균 139 mg/dl, 표준편차 66 mg/dl, range 343 mg/dl로 제시된다. SBP는 최소 90 mmHg, 최대 180 mmHg, 평균 123 mmHg, 표준편차 21 mmHg, range 90 mmHg이다. DBP는 최소 60 mmHg, 최대 120 mmHg, 평균 78 mmHg, 표준편차 16 mmHg, range 60 mmHg이다. 다만 이 population이 어디에서 수집되었는지, sample 수가 얼마인지, train/test split이 어떻게 되었는지는 제공된 텍스트에 명확하지 않다.

Machine learning module에서는 네 가지 nonlinear regression algorithms가 주요하게 언급된다. 첫 번째는 decision tree regression이다. Decision tree는 데이터를 여러 node와 branch로 나누어 예측 규칙을 구성하는 모델이며, regression에서는 최종 leaf node가 real number를 출력한다. 이 논문에서는 decision tree가 basic data를 정의하고 분류하며, 최종 outcome이 real number라고 설명된다.

두 번째는 support vector machine이다. 논문은 support vector regression이 structural risk minimization에 기반하며, 일반적인 linear regression처럼 단순히 error magnitude만 최소화하지 않는다는 점을 강조한다. 또한 nonlinear kernel, 특히 radial basis function을 사용할 수 있어 nonlinear regression problem에 적용할 수 있다고 설명한다.

세 번째는 random forest regression이다. Random forest는 여러 decision tree를 학습하고, 각 tree의 결과를 평균하여 최종 regression output을 만드는 ensemble method이다. 논문은 standard tree에서는 각 node가 최적 분할점에서 나뉘지만, random forest에서는 각 node에서 무작위로 선택된 feature subset을 기반으로 최적 분할을 찾는다고 설명한다. 여러 weak estimator를 평균하기 때문에 일반화 성능이 높아질 수 있으나, 최종 모델 저장에 많은 memory가 필요하다고 언급한다.

네 번째는 neural network이다. 논문에는 neural network의 구체적 architecture, layer 수, activation function, optimizer, loss function이 명확히 제시되어 있지 않다. 다만 table에서는 neural network가 linear model보다 높은 성능을 보이며, nonlinear algorithm 중 하나로 사용되었다고 설명된다.

성능 비교에는 coefficient of determination, 즉 $R^2$가 사용된다. $R^2$는 모델이 target variable의 variance를 얼마나 설명하는지를 나타내며, 일반적으로 값이 1에 가까울수록 예측력이 높다. 일반적인 형태는 다음과 같이 이해할 수 있다.

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

여기서 $y_i$는 실제값, $\hat{y}_i$는 예측값, $\bar{y}$는 실제값의 평균이다. 논문 자체에는 $R^2$ 수식이 명시되어 있지 않지만, 성능 table에서 BGL, SBP, DBP 각각에 대한 regression score로 사용된다.

BGL 평가에는 Clarke error grid analysis도 사용된다. Clarke error grid는 reference blood glucose와 predicted blood glucose를 비교하여 임상적으로 안전한 예측인지 평가하는 방법이다. Scatter plot을 여러 region으로 나누고, 각 region은 예측 오류가 임상 의사결정에 미치는 영향을 나타낸다. 논문은 region A가 정확한 예측 또는 임상적으로 허용 가능한 예측을 의미하고, region B는 20% 이내 또는 부적절한 치료로 이어지지 않는 예측을 의미한다고 설명한다. Region C, D, E는 더 위험한 오류를 나타낸다.

## 4. 실험 및 결과

논문에 제시된 정량 결과는 크게 machine learning algorithm별 $R^2$ 비교와 Clarke error grid analysis로 구성된다.

먼저 regression algorithm 비교에서는 linear model, neural network, support vector machine, random forest가 BGL, SBP, DBP 추정에 대해 비교된다. Table 2에 따르면 BGL regression의 $R^2$는 linear model 0.52, neural network 0.54, SVM 0.64, random forest 0.88이다. SBP regression의 $R^2$는 linear model 0.59, neural network 0.65, SVM 0.72, random forest 0.90이다. DBP regression의 $R^2$는 linear model 0.53, neural network 0.63, SVM 0.68, random forest 0.86이다.

이 결과는 random forest regression이 세 가지 target 모두에서 가장 높은 설명력을 보였음을 의미한다. 특히 SBP에서 $R^2 = 0.90$, BGL에서 $R^2 = 0.88$, DBP에서 $R^2 = 0.86$으로 나타나, PPG features와 target physiological variables 사이의 nonlinear relationship을 가장 잘 포착한 모델로 제시된다. SVM도 linear model이나 neural network보다 높은 성능을 보이지만, random forest에는 미치지 못한다. Linear model은 모든 target에서 가장 낮은 성능을 보이며, 이는 PPG feature와 BP 또는 BGL 사이의 관계가 단순 선형이 아니라는 논문의 주장을 뒷받침한다.

다만 neural network의 성능이 random forest보다 낮게 제시된 점은 주의 깊게 해석해야 한다. 현대 deep learning 모델이 항상 우수한 것은 아니며, 제한된 sample size, feature engineering 기반 입력, 작은 데이터셋에서는 random forest 같은 ensemble tree model이 더 안정적으로 작동할 수 있다. 그러나 논문에는 neural network 구조와 학습 설정이 명확히 제공되지 않기 때문에, neural network 자체의 한계라기보다는 해당 실험 설정에서의 결과로 보는 것이 적절하다.

BGL 추정에 대해서는 Clarke error grid analysis 결과가 제시된다. 논문에 따르면 region A에는 87.7%의 points가 포함되었고, region B에는 10.3%의 data가 포함되었다. Region B는 reference sensor와 비교했을 때 오차가 있더라도 부적절한 치료로 이어지지 않는 영역으로 설명된다. Region C에는 points가 없었고, 이는 false positive disease prediction이 없었다는 의미로 해석된다. Region D에는 2%의 points가 포함되었으며, 이는 hypoglycemia 또는 hyperglycemia detection을 놓칠 수 있는 위험한 영역을 의미한다. Region E에는 points가 없었다.

이 Clarke grid 결과는 대부분의 BGL 예측이 임상적으로 안전한 영역인 A와 B에 들어갔음을 보여준다. 특히 A와 B를 합하면 98%의 points가 포함된다. 그러나 region D에 2%가 존재한다는 점은 실제 혈당 모니터링 제품으로 사용하기에는 중요한 위험 요소이다. Hypoglycemia나 hyperglycemia를 놓치는 것은 임상적으로 심각할 수 있기 때문이다.

논문은 BGL coefficient $R^2$의 approximate final value가 0.90이고, RSS가 sample variance의 10%로 측정되었다고 설명한다. 그러나 이 값은 Table 2의 random forest BGL $R^2 = 0.88$과 약간 다르다. 제공된 텍스트만으로는 이 차이가 다른 실험 설정, 다른 모델, 또는 반올림 때문인지 확인할 수 없다.

논문은 mean error와 standard deviation이 낮은 모델이 가장 좋은 estimator라고 설명한다. 그러나 제공된 텍스트에는 BP와 BGL 각각에 대한 ME와 SD의 구체적 수치 table이 충분히 제시되어 있지 않다. 따라서 본 보고서에서는 $R^2$와 Clarke grid analysis 중심으로 결과를 해석한다.

전반적으로 실험 결과는 PPG feature 기반 estimation에서 nonlinear regression이 linear regression보다 유리하며, 특히 random forest가 가장 높은 성능을 보였다는 결론으로 정리된다. 다만 데이터셋 크기, 수집 조건, train/test split, cross-validation 세부 방식, 외부 검증 여부가 불명확하여 결과의 일반화 가능성은 제한적으로 해석해야 한다.

## 5. 강점, 한계

이 논문의 첫 번째 강점은 PPG를 이용한 non-invasive BP와 BGL estimation을 함께 다룬다는 점이다. 대부분의 연구는 혈압 또는 혈당 중 하나에 집중하는 경우가 많지만, 이 논문은 두 생리 지표 모두에 대해 PPG 기반 추정 가능성을 검토한다. 이는 PPG가 단순 heart rate measurement를 넘어 다양한 physiological monitoring에 활용될 수 있다는 관점을 제공한다.

두 번째 강점은 여러 machine learning regression algorithms를 비교했다는 점이다. Linear model, neural network, SVM, random forest를 BGL, SBP, DBP 각각에 대해 비교하여 nonlinear model의 필요성을 보여준다. 특히 random forest가 모든 target에서 가장 높은 $R^2$를 보였다는 결과는 PPG feature 기반 소규모 데이터 환경에서 ensemble tree method가 실용적일 수 있음을 시사한다.

세 번째 강점은 Clarke error grid analysis를 사용해 BGL 추정의 임상적 안전성을 평가하려 했다는 점이다. 단순 correlation이나 $R^2$만으로는 혈당 예측이 실제 의사결정에 안전한지 판단하기 어렵다. Clarke grid는 예측 오류가 치료 결정에 미치는 영향을 구분하기 때문에, BGL estimation 연구에서 중요한 평가 방법이다.

네 번째 강점은 PPG signal quality 문제를 인식하고 activity detection, FSA, spectral entropy, Teager energy 등의 preprocessing 개념을 포함했다는 점이다. PPG는 motion artifact와 signal loss에 매우 취약하므로, corrupted segment를 탐지하고 제거하는 과정은 실제 시스템에서 필수적이다.

그러나 이 논문에는 한계가 많다. 첫째, review paper와 experimental paper의 경계가 불명확하다. 제목과 초록은 review paper를 표방하지만, 본문에는 자체 dataset처럼 보이는 population table과 algorithm performance table이 포함되어 있다. 이 데이터가 저자들이 직접 수집한 것인지, 기존 문헌에서 가져온 것인지, 또는 특정 reference의 결과를 재정리한 것인지 명확하지 않다.

둘째, 데이터셋 설명이 부족하다. Population statistics는 제공되지만, sample 수, participant 수, 측정 장비, PPG sensor wavelength, sampling rate, measurement protocol, train/test split, cross-validation 방식이 충분히 설명되지 않는다. 이러한 정보가 없으면 결과를 재현하거나 성능의 신뢰성을 평가하기 어렵다.

셋째, machine learning model의 구체적 학습 설정이 부족하다. Decision tree, SVM, random forest, neural network가 언급되지만, hyperparameter, kernel setting, tree 수, maximum depth, neural network architecture, optimizer, loss function 등이 명확히 제시되지 않는다. 따라서 어떤 조건에서 random forest가 가장 좋았는지 판단하기 어렵다.

넷째, blood glucose estimation에서 PPG 기반 optical absorption 설명이 단순화되어 있다. 실제 non-invasive glucose monitoring은 매우 어려운 문제이며, 피부, 조직, 혈류, 온도, 수분, 압력, motion artifact 등 다양한 confounder의 영향을 받는다. 논문은 NIR absorption과 glucose의 관계를 언급하지만, PPG만으로 glucose를 정확히 추정할 때 발생하는 생리적 및 광학적 난점을 충분히 깊게 다루지는 않는다.

다섯째, Clarke error grid 결과에서 region D에 2%의 points가 있다는 점은 안전성 측면에서 중요한 문제이다. 논문은 전체적으로 좋은 결과를 강조하지만, hypoglycemia 또는 hyperglycemia detection을 놓칠 수 있는 오류는 실제 의료 응용에서 매우 위험하다. 이 부분에 대한 원인 분석이나 개선 방향이 충분히 제시되지 않는다.

여섯째, 결과 수치의 일관성과 맥락이 부족하다. Table 2의 BGL $R^2$는 random forest에서 0.88로 제시되지만, 결과 설명에서는 final $R^2$가 약 0.90이라고 언급된다. 이 정도 차이는 크지 않지만, 어떤 모델과 어떤 데이터 분할에서 나온 값인지 명확하지 않다. 또한 ME와 SD가 가장 중요한 평가 지표라고 설명하면서도, 구체적인 ME와 SD table은 제공된 텍스트에 나타나지 않는다.

비판적으로 보면, 이 논문은 PPG 기반 BP 및 BGL estimation의 가능성을 개괄적으로 소개하는 데는 유용하지만, 엄밀한 실험 논문으로 보기에는 방법론과 평가 설명이 부족하다. 특히 혈당 추정은 임상적으로 매우 민감한 문제이므로, 단순한 $R^2$와 제한적 Clarke grid 결과만으로 실제 적용 가능성을 강하게 주장하기는 어렵다. 후속 연구에서는 명확한 데이터 수집 프로토콜, 독립 test set, subject-wise validation, sensor calibration, artifact robustness 분석, 임상 기준 기반 평가가 필요하다.

## 6. 결론

이 논문은 PPG signal을 이용한 non-invasive blood pressure와 blood glucose level estimation의 가능성을 검토하고, 여러 machine learning regression algorithms의 성능을 비교했다. PPG는 optical sensing 기반으로 혈액량 변화를 측정할 수 있으며, waveform의 amplitude, frequency, slope, pulse interval, pulse area 등 다양한 feature를 통해 SBP, DBP, BGL과 관련된 정보를 제공할 수 있다.

제시된 결과에서는 random forest regression이 BGL, SBP, DBP 추정에서 각각 높은 $R^2$를 보였고, linear regression보다 nonlinear algorithms가 더 적합하다는 결론을 뒷받침했다. Clarke error grid analysis에서는 BGL 예측 points의 대부분이 region A와 B에 포함되어 임상적으로 허용 가능한 예측이 많았다고 설명된다. 그러나 region D에 일부 points가 존재하고, 데이터셋 및 학습 설정이 명확하지 않다는 점은 중요한 한계이다.

이 연구의 주요 기여는 PPG 기반 BP 및 BGL estimation을 위한 feature extraction, signal processing, nonlinear regression pipeline을 개괄적으로 정리하고, random forest와 같은 machine learning algorithm의 유용성을 강조한 것이다. 실제 적용 측면에서는 finger-based PPG뿐 아니라 wrist-based PPG와 smart watch 형태의 wearable system으로 확장 가능성을 제안한다.

다만 이 논문은 새로운 딥러닝 모델을 엄밀히 제안하고 검증한 연구라기보다는, PPG 기반 physiological parameter estimation의 가능성과 관련 방법을 소개하는 review 성격이 강하다. 따라서 실제 제품화나 임상 적용을 위해서는 더 큰 규모의 데이터, 명확한 실험 설계, 외부 검증, 개인별 차이 분석, motion artifact robustness, 혈당 추정의 임상 안전성 평가가 반드시 필요하다. 그럼에도 이 논문은 PPG를 이용한 비침습 혈압 및 혈당 추정 연구의 기본 아이디어와 machine learning 활용 방향을 이해하는 입문적 자료로 활용할 수 있다.
