# Cuffless blood pressure estimation from PPG signals and its derivatives using deep learning models

* **저자**: C. El-Hajj, P.A. Kyriacou
* **발표연도**: 2021

## 1. 논문 개요

이 논문은 하나의 Photoplethysmography, 즉 PPG 센서에서 얻은 신호만을 사용하여 cuff 없이 systolic blood pressure, 즉 SBP와 diastolic blood pressure, 즉 DBP를 추정하는 딥러닝 기반 방법을 제안한다. 기존의 혈압 측정은 cuff 기반 oscillometric 방식이 널리 쓰이지만, 측정이 간헐적이고 불편하며, 장시간 연속 모니터링에는 적합하지 않다. 또한 병원 환경에서 긴장 때문에 혈압이 높게 측정되는 white coat hypertension이나, 실제로는 혈압이 높지만 측정 시 정상으로 나타나는 masked hypertension 같은 문제가 발생할 수 있다. 따라서 논문은 비침습적이고 연속적이며 사용자가 일상적으로 착용할 수 있는 cuffless blood pressure monitoring 방법의 필요성에서 출발한다.

연구 문제는 PPG 파형과 그 1차 미분 PPG’, 2차 미분 PPG’’에서 추출한 morphological feature들이 SBP와 DBP를 얼마나 정확하게 예측할 수 있는지, 그리고 temporal dependency를 학습할 수 있는 recurrent deep learning model이 이 문제에 효과적인지를 검증하는 것이다. 특히 이 논문은 ECG와 PPG를 함께 사용하는 Pulse Transit Time, PTT 또는 Pulse Arrival Time, PAT 기반 접근과 달리, 단일 PPG 신호만을 사용한다는 점을 강조한다. PTT/PAT 방식은 일반적으로 두 개 이상의 센서가 필요하고, 센서 간 동기화가 중요하며, 움직임 잡음에 민감하다는 단점이 있다. 반면 단일 PPG 기반 접근은 웨어러블 기기에 적용하기 쉽고, 장기적 혈압 모니터링 시스템으로 발전할 가능성이 높다.

논문은 MIMIC II 데이터셋에서 유래한 942명의 피험자 데이터를 사용한다. PPG와 invasive arterial blood pressure, 즉 ABP 신호가 사용되며, ABP에서 추출한 SBP와 DBP가 ground truth 역할을 한다. 저자들은 PPG 파형과 그 derivative에서 총 52개의 feature를 추출하고, feature selection을 통해 24개 feature로 축소한 뒤, LSTM 또는 GRU 기반 bidirectional recurrent layer, stacked recurrent layers, attention layer를 포함한 deep recurrent architecture를 평가한다.

## 2. 핵심 아이디어

논문의 핵심 아이디어는 혈압이 PPG 파형의 모양, 즉 morphology에 반영된다는 가정에 기반한다. PPG는 피부의 혈관에 빛을 조사하고 반사 또는 흡수되는 빛의 변화를 측정하여 혈액량 변화를 간접적으로 포착한다. 심장이 수축하면 혈압이 상승하고 말초 혈관으로 pulse wave가 전달되며, 이 과정은 PPG 파형의 systolic peak, pulse width, systolic upstroke time, diastolic time, area under curve 등으로 나타난다. 혈관 탄성, 말초 저항, 나이 또는 질병으로 인한 혈관 변화 역시 PPG morphology에 영향을 줄 수 있으므로, PPG feature와 혈압 사이에는 유용한 상관관계가 존재할 수 있다.

기존 접근과의 중요한 차별점은 세 가지이다. 첫째, 논문은 ECG와 PPG를 함께 사용하는 PTT/PAT 기반 방법이 아니라 단일 PPG 센서만을 사용한다. 이는 실제 wearable device 적용 측면에서 장점이 크다. 둘째, 원시 PPG 신호를 그대로 사용하는 end-to-end 방식이 아니라, PPG, PPG’, PPG’’에서 해석 가능한 morphology feature를 추출한 뒤 recurrent neural network에 입력한다. 셋째, 단순한 feedforward neural network나 classical machine learning model이 아니라, 시간 순서에 따라 변화하는 연속 cardiac cycle feature들의 dependency를 학습하기 위해 BiLSTM/BiGRU와 attention mechanism을 결합한다.

논문이 보는 중요한 직관은 혈압 추정이 단일 cardiac cycle의 독립적 회귀 문제가 아니라, 연속된 PPG cycle들 사이의 시간적 변화까지 반영해야 하는 sequential regression 문제라는 점이다. LSTM과 GRU는 과거 시점의 정보를 저장하고 활용할 수 있으며, bidirectional layer는 forward direction과 backward direction의 정보를 함께 사용한다. Attention mechanism은 각 time step의 hidden state 중 혈압 추정에 더 중요한 정보를 가진 부분에 더 큰 가중치를 부여한다. 따라서 모델은 단순히 feature vector 하나를 혈압으로 매핑하는 것이 아니라, 시간에 따른 feature sequence에서 혈압과 관련된 패턴을 학습한다.

## 3. 상세 방법 설명

논문의 전체 파이프라인은 데이터 준비, 전처리, feature extraction, feature selection, deep recurrent model 학습 및 평가로 구성된다. 먼저 MIMIC II 기반의 전처리된 데이터셋에서 PPG와 ABP 신호를 사용한다. PPG와 ABP는 125 Hz로 샘플링되어 있으며, 저자들은 10초 길이의 segment로 신호를 나누어 각 segment 단위로 SBP와 DBP를 추정한다. Ground truth SBP와 DBP는 ABP 신호의 peak와 valley에서 얻는다. 한 cardiac cycle에서 peak는 SBP에 대응하고 valley는 end-diastolic value, 즉 DBP에 대응한다. 10초 segment 안에서 검출된 peak와 valley의 평균값이 해당 sequence의 SBP와 DBP label로 사용된다.

전처리는 Kachuee et al.이 공개한 MIMIC II 기반 전처리 데이터셋을 사용한다. 원 데이터는 ICU 환경에서 수집되었기 때문에 잡음과 손상된 신호가 많다. 논문에서 설명한 전처리 절차는 discrete wavelet decomposition, DWT를 사용하며, Daubechies 8, 즉 db8 wavelet을 mother wavelet로 사용한다. 매우 낮은 주파수 성분과 높은 주파수 성분을 제거한 뒤 wavelet denoising을 적용한다. 이후 PPG amplitude는 min-max normalization으로 $[0,1]$ 범위로 정규화된다. 이는 피험자마다 PPG amplitude scale이 크게 다르기 때문에 peak detection과 feature extraction의 안정성을 높이기 위한 과정이다.

Feature extraction 단계에서는 PPG, PPG’, PPG’’에서 총 52개의 feature를 추출한다. PPG에서 추출한 feature는 systolic peak amplitude, 여러 amplitude level에서의 pulse width, systolic width와 diastolic width의 ratio, systolic area, diastolic area, pulse interval, heart rate, systolic upstroke time, diastolic time, PPG intensity ratio 등이다. PPG’에서는 첫 번째 peak amplitude, peak time, valley time 및 이들의 pulse interval 대비 ratio를 사용한다. PPG’’에서는 a-wave, b-wave, e-wave와 관련된 amplitude와 time interval, 그리고 $A_b/A_a$, $A_e/A_a$ 같은 ratio feature를 추출한다.

논문은 dicrotic notch와 diastolic peak 관련 feature를 제외한다. 그 이유는 MIMIC II 데이터가 ICU 환자에게서 수집되었고, 고령, 질병, 약물 영향 등으로 인해 dicrotic notch와 diastolic peak가 명확하지 않은 경우가 많기 때문이다. 따라서 저자들은 모든 cycle에서 안정적으로 검출할 수 있어야 하는 두 개의 foot point와 하나의 systolic peak를 중심으로 feature를 구성한다. 논문 page 3의 Fig. 1은 전형적인 PPG cycle에서 systolic peak, diastolic peak, dicrotic notch, foot point를 보여주며, page 7의 Fig. 3은 PPG와 그 1차 및 2차 미분에서 시간 및 amplitude feature가 어떻게 정의되는지 시각적으로 설명한다.

Feature selection은 세 단계로 진행된다. 첫째, 모든 feature를 $[0,1]$ 범위로 정규화한다. 논문에서 사용한 min-max normalization은 다음과 같다.

$$
x' = \frac{x - \min(x)}{\max(x) - \min(x)}
$$

여기서 $x$는 원래 feature 값이고, $x'$는 정규화된 feature 값이다. 이 정규화는 feature scale 차이 때문에 특정 feature가 과도하게 큰 영향을 갖는 문제를 줄이고, outlier의 영향을 완화하는 목적을 가진다.

둘째, Pearson’s correlation coefficient를 사용하여 feature들 사이의 선형 collinearity를 평가한다. 논문에서는 두 feature의 Pearson correlation coefficient가 0.9 이상이면 강한 상관관계가 있다고 보고, 중복 feature 후보로 간주한다. 동시에 maximum information coefficient, MIC를 사용하여 각 feature와 혈압 사이의 비선형 dependency를 평가한다. 강하게 상관된 feature pair 중 혈압과의 MIC가 낮은 feature를 제거함으로써 collinearity와 redundancy를 줄인다.

셋째, recursive feature elimination, RFE를 사용한다. RFE는 특정 learner model을 내부에 두고 feature importance를 계산한 뒤, 중요도가 낮은 feature를 반복적으로 제거하는 wrapper-based feature selection 방법이다. 이 논문에서는 random forest를 RFE 내부 learner로 사용한다. 그 결과 SBP에는 23개, DBP에는 22개 feature가 선택되었고, 두 집합을 합쳐 최종 24개 feature set을 구성한다. page 7의 Table 3은 SBP와 DBP에 대해 선택된 feature와 중요도 순서를 보여준다. 가장 중요한 feature로는 $A_e/A_a$, $DW25/SW25$, $DT$, $t_{slope}/t_{pi}$ 등이 제시된다.

모델 구조는 baseline인 multilinear regression, MLR과 두 종류의 deep recurrent model로 구성된다. MLR은 feature와 혈압 사이의 선형 관계만을 모델링하는 비교용 baseline이다. 논문에서 제시한 MLR 식은 다음과 같다.

$$
Y = \beta + X_0\beta_0 + X_1\beta_1 + X_2\beta_2 + X_3\beta_3 \cdots X_n\beta_n + \epsilon
$$

여기서 $Y$는 예측된 SBP 또는 DBP이고, $X_0, X_1, \ldots, X_n$은 PPG 기반 input feature이다. $\beta_0, \beta_1, \ldots, \beta_n$은 회귀 계수이며, $\epsilon$은 예측값과 실제값의 차이를 나타내는 error term이다. 계수는 mean squared error, MSE를 최소화하도록 최적화된다.

Deep recurrent model은 하나의 bidirectional recurrent layer를 첫 번째 layer로 사용하고, 그 뒤에 여러 개의 unidirectional recurrent layer를 쌓은 다음 attention layer를 적용한다. Recurrent unit으로는 LSTM과 GRU를 각각 사용하여 두 모델을 비교한다. LSTM은 input gate, forget gate, output gate와 cell state를 이용하여 장기 의존성을 학습하고 vanishing gradient 문제를 완화한다. GRU는 update gate와 reset gate를 사용하며, LSTM보다 parameter 수가 적어 계산 효율이 높을 수 있다.

Bidirectional recurrent layer는 sequence를 정방향과 역방향으로 동시에 처리한다. 정방향 layer는 과거 정보를 반영하고, 역방향 layer는 가까운 미래 정보를 반영한다. 두 hidden representation을 concatenate하여 각 time step의 출력으로 사용한다. page 8의 Fig. 5는 forward layer와 backward layer가 time step마다 결합되는 bidirectional structure를 보여준다.

Attention mechanism은 각 time step의 hidden state가 혈압 예측에 얼마나 중요한지를 학습한다. 논문은 self-attention 형태를 사용하며, hidden state마다 score를 계산한 뒤 softmax로 정규화하여 attention weight를 만든다. 최종 context vector $V$는 attention weight와 hidden state의 weighted sum으로 계산된다.

$$
V = \sum_i^n \alpha_i h_i
$$

여기서 $h_i$는 $i$번째 time step의 hidden state이고, $\alpha_i$는 해당 hidden state에 부여된 attention weight이다. 직관적으로 말하면, 모델은 모든 time step을 동일하게 취급하지 않고 혈압 추정에 더 유용한 temporal feature에 더 큰 비중을 둔다.

학습 설정은 train 70%, validation 15%, test 15% 분할을 사용한다. Test set은 최종 평가용으로 training data와 완전히 분리된다. 모델 선택은 validation error가 가장 낮은 모델을 기준으로 한다. Loss function은 MSE이고 optimizer는 Adam이다. Epoch 수는 300, batch size는 128로 설정된다. 논문은 calibration-free setting에서 평가하며, 이는 개인별 cuff calibration 없이 일반화된 모델을 평가한다는 점에서 실제 적용 측면의 의미가 있다.

## 4. 실험 및 결과

실험은 MIMIC II에서 유래한 942명 subject의 PPG와 ABP 데이터를 사용한다. 최종 데이터셋의 10초 평균 SBP 범위는 약 80.09 mmHg에서 179.99 mmHg이고, 평균은 134.30 mmHg, 표준편차는 19.80 mmHg이다. DBP 범위는 60.00 mmHg에서 129.58 mmHg이고, 평균은 73.48 mmHg, 표준편차는 10.04 mmHg이다. page 4의 Table 1과 page 5의 Fig. 2는 SBP와 DBP의 분포를 보여주며, SBP가 DBP보다 더 넓은 범위를 갖는다는 점이 실험 결과 해석에서 중요하다.

평가 지표는 mean absolute error, MAE와 standard deviation, SD이다. MAE는 예측값과 실제값의 절대 오차 평균이며, 혈압 추정 문제에서 직관적으로 해석하기 쉽다. SD는 오차의 변동성을 나타낸다. 추가적으로 논문은 Pearson correlation coefficient와 Bland-Altman plot을 사용하여 예측값과 reference value의 상관성 및 agreement를 분석한다.

52개 feature 전체를 사용한 실험에서 MLR은 SBP에 대해 MAE 14.86 mmHg, SD 10.88 mmHg, DBP에 대해 MAE 7.14 mmHg, SD 6.30 mmHg를 기록한다. 이는 deep learning model보다 훨씬 나쁜 성능이며, PPG feature와 혈압 사이의 관계가 단순 선형 모델로 충분히 설명되지 않음을 시사한다.

반면 BiLSTM + stacked LSTM + attention 모델은 52개 feature set에서 가장 좋은 성능을 보인다. 구체적으로 512 hidden unit, 4개의 unidirectional LSTM layer, learning rate 0.0001 설정에서 SBP MAE 4.51 mmHg, SD 7.81 mmHg, DBP MAE 2.60 mmHg, SD 4.41 mmHg를 달성한다. BiGRU + stacked GRU + attention 모델도 유사한 성능을 보이며, SBP MAE 4.69 mmHg, SD 7.76 mmHg, DBP MAE 2.68 mmHg, SD 4.39 mmHg를 기록한다. 이 결과는 recurrent deep learning model이 단일 PPG 기반 혈압 추정에 효과적임을 보여준다.

52개 feature set에서 best model의 regression analysis는 SBP에 대해 Pearson correlation coefficient $R = 0.89$, DBP에 대해 $R = 0.86$을 보인다. 이는 reference 혈압과 예측 혈압 사이에 강한 양의 상관관계가 있음을 의미한다. Bland-Altman analysis에서는 SBP mean error가 -0.48 mmHg, DBP mean error가 -0.49 mmHg로 나타난다. Limits of agreement는 SBP에서 약 [-18.42, 17.45], DBP에서 [-10.50, 9.52]이다. page 9의 Fig. 6은 이 regression plot과 Bland-Altman plot을 보여준다.

24개 feature로 축소한 실험에서도 성능 저하는 크지 않다. MLR은 여전히 낮은 성능을 보이며, SBP MAE 15.11 mmHg, DBP MAE 7.42 mmHg이다. Deep recurrent model의 경우 BiGRU + GRU + attention 모델이 가장 좋은 성능을 보인다. 512 hidden unit, 2개의 unidirectional GRU layer, learning rate 0.001 설정에서 SBP MAE 4.79 mmHg, SD 8.08 mmHg, DBP MAE 2.77 mmHg, SD 4.72 mmHg를 달성한다. 52개 feature를 사용한 최선의 BiLSTM 모델보다 약간 낮은 성능이지만, feature 수가 절반 이하로 줄고 GRU의 계산 효율이 높다는 점을 고려하면 실용적인 장점이 있다.

24개 feature set에서 correlation coefficient는 SBP $R = 0.88$, DBP $R = 0.84$로 보고된다. Bland-Altman plot에서는 SBP mean error -0.91 mmHg, DBP mean error -0.44 mmHg이며, limits of agreement는 SBP에서 [-19.22, 17.41], DBP에서 [-11.13, 10.25]이다. page 10의 Fig. 7은 축소 feature set에 대한 regression 및 Bland-Altman 결과를 보여준다.

논문은 AAMI standard와의 비교도 수행한다. AAMI 기준은 최소 85명 이상의 subject에서 mean error가 5 mmHg 이하이고 SD가 8 mmHg 이하이어야 한다는 조건을 제시한다. 논문 결과에서 DBP는 mean error와 SD 모두 기준을 만족한다. SBP는 mean error는 기준을 만족하지만, Table 6에 제시된 SD는 52개 feature best model에서 9.15 mmHg, 24개 feature best model에서 9.34 mmHg로 AAMI의 8 mmHg 기준을 약간 초과한다. 따라서 논문의 abstract에서 “international standard를 충족한다”고 표현하지만, 본문 Table 6을 엄밀히 보면 DBP는 충분히 충족하고 SBP는 mean error 기준은 만족하되 SD 기준은 약간 초과한다는 점을 구분해서 이해해야 한다.

관련 연구와의 비교에서 논문은 Kachuee et al.의 PTT/PAT 기반 calibration-free 방법, Tanveer et al.의 ECG+PPG raw signal 기반 ANN-LSTM, Kurylyak et al.의 PPG feature 기반 ANN, Slapnicar et al.의 raw PPG derivative 기반 ResNet과 비교한다. 특히 Kachuee et al.의 942명 MIMIC II 기반 Adaboost 방법은 SBP MAE 11.17 mmHg, DBP MAE 5.35 mmHg로 보고되며, 본 논문의 52개 feature BiLSTM 모델은 이보다 더 낮은 오차를 보인다. 그러나 논문도 직접 비교가 어렵다는 점을 인정한다. 데이터셋의 subject 수, 전처리 방식, calibration 여부, window length, evaluation metric이 연구마다 다르기 때문이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 단일 PPG sensor만을 사용하면서도 비교적 큰 규모인 942명 subject에서 실험을 수행했다는 점이다. cuffless BP estimation 분야에서는 적은 수의 subject에서 매우 낮은 오차를 보고하는 연구가 많지만, subject 수가 적으면 일반화 성능을 판단하기 어렵다. 본 논문은 MIMIC II 기반 대규모 데이터에서 calibration-free setting을 사용하여 실험했다는 점에서 실용적 의미가 있다.

두 번째 강점은 PPG, PPG’, PPG’’의 feature를 체계적으로 탐색하고 feature selection을 수행했다는 점이다. 단순히 많은 feature를 넣는 데 그치지 않고, Pearson correlation, MIC, RFE를 순차적으로 사용해 collinearity와 redundancy를 줄이려 했다. 특히 52개 feature에서 24개 feature로 축소해도 성능 저하가 크지 않다는 결과는 계산 효율과 wearable implementation 관점에서 유용하다.

세 번째 강점은 temporal dependency를 명시적으로 모델링했다는 점이다. MLR과 deep recurrent model 사이의 큰 성능 차이는 PPG feature와 혈압 사이 관계가 비선형적이며, 시간적 정보가 중요하다는 논문의 주장을 뒷받침한다. LSTM과 GRU는 연속된 cardiac cycle에서 나타나는 feature 변화의 흐름을 학습할 수 있고, attention mechanism은 혈압 추정에 중요한 time step에 더 집중하도록 돕는다.

그러나 한계도 분명하다. 첫째, 데이터가 ICU 환자에게서 수집되었다는 점은 일반 인구에 대한 일반화 가능성을 제한한다. ICU 환자는 질병, 약물, 고령, 생리적 불안정성의 영향을 받을 수 있으며, 이는 PPG morphology와 혈압 분포를 일반적인 wearable 사용자 환경과 다르게 만들 수 있다. 논문 역시 MIMIC II의 BP range가 일반 population을 완전히 대표하지 않는다고 언급한다.

둘째, PPG signal quality가 낮고 dicrotic notch 및 diastolic peak가 명확하지 않은 경우가 많아 일부 중요한 morphology feature를 제외해야 했다. 이는 실제 wearable 환경에서도 중요한 문제다. 손가락, 손목, 귀 등 측정 위치에 따라 PPG 품질이 달라지고, motion artifact가 심하면 feature extraction이 실패할 수 있다. 논문은 feature extraction이 가능한 segment를 중심으로 평가했기 때문에, 실제 noisy environment에서의 robustness는 추가 검증이 필요하다.

셋째, 환자의 demographic information, 예를 들어 age, gender, height, weight 같은 변수를 사용하지 못했다. 혈관 탄성, 혈압, PPG morphology는 나이와 성별, 체격, 건강 상태와 관련이 있을 수 있으므로, 이러한 정보가 있으면 성능이 향상될 가능성이 있다. 논문은 MIMIC II에서 이러한 정보가 충분히 제공되지 않는 점을 한계로 제시한다.

넷째, AAMI 기준 해석에는 주의가 필요하다. DBP는 기준을 명확히 만족하지만, SBP의 SD는 Table 6에서 8 mmHg를 초과한다. 따라서 “전체적으로 국제 기준을 만족한다”는 표현은 다소 완화해서 해석해야 한다. 엄밀하게는 SBP mean error는 기준을 만족하지만, SBP error variability는 기준보다 약간 크다.

다섯째, 모델은 calibration-free 방식으로 평가되었지만, 실제 장기 사용 상황에서 시간 경과, 센서 위치 변화, 피부 상태, 운동, 온도 변화에 대해 얼마나 안정적인지는 논문에서 직접 검증되지 않았다. 또한 train/validation/test split이 subject-independent 방식인지, 즉 같은 subject의 segment가 train과 test에 동시에 들어가지 않았는지에 대한 설명은 원문에서 명확히 충분히 드러나지 않는다. 만약 subject-level 분리가 엄격하지 않았다면, 개인 고유의 PPG 특성이 test 성능에 영향을 주었을 가능성이 있다. 이 부분은 논문에 명확히 제시되지 않은 것으로 보이며, 후속 검증에서 중요하게 확인해야 한다.

## 6. 결론

이 논문은 단일 PPG sensor 기반 cuffless blood pressure estimation의 가능성을 deep recurrent learning 관점에서 체계적으로 검증한 연구이다. 저자들은 PPG와 그 1차 및 2차 미분에서 총 52개 morphology feature를 추출하고, feature selection을 통해 24개 feature set도 구성하였다. 이후 BiLSTM/BiGRU, stacked recurrent layers, attention mechanism을 결합한 모델을 사용하여 SBP와 DBP를 예측했다.

가장 좋은 결과는 52개 feature를 사용한 BiLSTM + LSTM + attention 모델에서 나타났으며, SBP MAE 4.51 mmHg, DBP MAE 2.60 mmHg를 달성했다. 이는 MLR baseline보다 크게 우수하며, PPG feature와 혈압 사이 관계가 비선형적이고 temporal dependency를 포함한다는 점을 뒷받침한다. 24개 feature set을 사용한 BiGRU 모델도 비슷한 수준의 성능을 보여, 계산 효율을 고려한 practical system 설계 가능성을 보여준다.

실제 적용 측면에서 이 연구는 cuff, ECG, 다중 센서 동기화 없이 하나의 PPG sensor만으로 혈압을 연속적으로 추정할 수 있다는 방향성을 제시한다. 이는 스마트워치, 손가락 센서, 웨어러블 헬스케어 디바이스에서 장기 혈압 모니터링을 구현하는 데 중요한 기반이 될 수 있다. 다만 ICU 데이터 기반 학습, noisy PPG에서의 feature extraction 문제, demographic 정보 부재, SBP error variability, 실제 생활 환경 검증 부족은 향후 연구에서 해결해야 할 핵심 과제이다. 특히 일반 인구 대상의 고품질 longitudinal dataset, subject-independent evaluation, motion artifact robustness, 개인별 calibration 여부에 따른 성능 비교가 추가된다면 이 접근의 임상적 신뢰성과 실용성이 더 명확히 평가될 수 있을 것이다.
