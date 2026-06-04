# A continuous cuffless blood pressure measurement from optimal PPG characteristic features using machine learning algorithms

* **저자**: Araf Nishan, S. M. Taslim Uddin Raju, Md Imran Hossain, Safin Ahmed Dipto, S. M. Tanvir Uddin, Asif Sijan, Md Abu Shahid Chowdhury, Ashfaq Ahmad, Md Mahamudul Hasan Khan
* **발표연도**: 2024

## 1. 논문 개요

이 논문은 photoplethysmogram(PPG) 신호의 characteristic features와 machine learning regression algorithm을 이용하여 cuff 없이 비침습적으로 혈압을 추정하는 방법을 제안한다. 추정 대상은 systolic blood pressure(SBP), mean arterial pressure(MAP), diastolic blood pressure(DBP) 세 가지이다. 논문의 목표는 PPG 신호 하나만으로 연속적이고 편리한 혈압 추정이 가능한 모델을 구축하고, AAMI 및 BHS 임상 기준을 만족하는 수준의 성능을 보이는지 검증하는 것이다.

혈압은 혈액이 혈관 벽에 가하는 압력이며, SBP는 심장이 수축할 때의 최고 압력, DBP는 심장이 이완할 때의 최저 압력을 의미한다. MAP는 한 심장 주기 동안의 평균 동맥압을 의미하며, 논문은 다음 식으로 MAP를 정의한다.

$$
MAP=\frac{2DBP+SBP}{3}
$$

고혈압과 저혈압은 모두 임상적으로 중요하다. 고혈압은 심장, 뇌, 신장 등 여러 장기에 장기적 손상을 줄 수 있으며 심혈관 질환의 주요 위험 요인이다. 저혈압 역시 빈혈, 심인성 쇼크, 어지러움, 장기 기능 이상 등 다양한 건강 문제와 관련될 수 있다. 따라서 혈압을 정확하고 반복적으로 측정하는 것은 심혈관 질환의 예방, 진단, 관리에 중요하다.

기존의 cuff 기반 혈압 측정은 병원과 가정에서 널리 사용되지만, 팔을 압박해야 하므로 불편하고 연속 측정에 적합하지 않다. 침습적 혈압 측정은 연속적인 혈압 파형을 얻을 수 있지만, 동맥에 cannula를 삽입해야 하므로 감염 위험과 환자 부담이 크다. 이 때문에 웨어러블 센서와 PPG를 활용한 cuffless, non-invasive, continuous BP monitoring이 중요한 연구 주제가 되었다.

이 논문은 PPG 신호에서 시간 영역, 주파수 영역, derivative 기반 feature를 추출하고, age와 gender까지 포함한 feature set을 구성한 뒤, feature selection과 nonlinear regression model을 결합한다. 핵심 연구 문제는 “PPG characteristic feature를 적절히 선택하면 단일 PPG 신호만으로 SBP, MAP, DBP를 임상 기준에 맞게 추정할 수 있는가?”이다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 PPG waveform이 혈관 내 혈류 변화와 혈압 관련 생리 정보를 반영하므로, waveform의 characteristic points와 derivative, frequency-domain feature를 추출하면 혈압을 추정할 수 있다는 것이다. PPG 신호는 일반적으로 systolic peak, dicrotic notch, diastolic peak를 포함한다. 이러한 지점들은 심장 수축, 대동맥판 폐쇄, 혈관 반사파, 혈관 탄성 등과 관련된 정보를 포함할 수 있다. 논문은 이 waveform 구조를 활용해 혈압과 관련될 가능성이 높은 feature를 명시적으로 설계한다.

기존 연구와의 차별점은 크게 세 가지로 정리할 수 있다. 첫째, ECG 없이 PPG 신호만 사용한다. PTT나 PAT 기반 방법은 보통 ECG와 PPG를 함께 사용해야 하지만, 이 논문은 단일 PPG 채널만을 사용해 센서 구성을 단순화한다. 둘째, PPG의 원 신호뿐 아니라 1차 derivative, 2차 derivative, Fast Fourier Transform(FFT) 기반 주파수 feature를 함께 사용한다. 셋째, 전체 feature를 그대로 사용하는 것이 아니라 CFS와 ReliefF feature selection을 통해 중요한 feature만 선택하고, nonlinear regression model에 입력한다.

논문의 가장 좋은 조합은 ReliefF 기반 feature selection과 support vector regression(SVR)이다. 이 조합은 SBP, MAP, DBP 추정에서 각각 MAE 2.49 mmHg, 1.62 mmHg, 1.69 mmHg를 달성했다고 보고한다. 다만 abstract에는 DBP MAE가 1.43 mmHg로 기재되어 있고, 본문 Table 8과 Table 11에는 1.69 mmHg로 제시되어 있다. 제공된 본문 표를 기준으로 하면 DBP MAE는 1.69 mmHg이며, abstract와 표 사이에 불일치가 존재한다.

이 연구는 deep learning end-to-end 방식이 아니라 handcrafted feature와 classical machine learning을 결합한다. 따라서 모델 구조는 상대적으로 단순하지만, feature가 어떤 PPG 형태 요소에서 추출되었는지 해석하기 쉽다. 또한 feature selection을 통해 과적합 가능성을 줄이고, 계산량을 낮추며, 예측 성능을 높이는 것을 목표로 한다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

제안 방법은 다섯 단계로 구성된다. 첫째, PPG-BP database에서 PPG 신호와 reference BP를 수집하고 signal quality를 평가한다. 둘째, normalization과 filtering을 통해 PPG 신호를 전처리한다. 셋째, 연속 PPG waveform에서 가장 좋은 단일 PPG cycle을 자동으로 검출하고 선택한다. 넷째, 선택된 PPG cycle, 1차 derivative, 2차 derivative, FFT 결과에서 feature를 추출하고, age와 gender를 추가한다. 다섯째, CFS와 ReliefF로 feature selection을 수행한 뒤 SVR, random forest regression(RFR), decision tree regression(DTR), K-nearest neighbor regression(KNR)을 학습하여 SBP, MAP, DBP를 추정한다.

전체 흐름은 다음과 같이 표현할 수 있다.

$$
PPG\ signal
\rightarrow
quality\ assessment
\rightarrow
preprocessing
\rightarrow
best\ PPG\ cycle\ selection
\rightarrow
feature\ extraction
\rightarrow
feature\ selection
\rightarrow
BP\ regression
$$

이 논문은 raw waveform을 neural network에 직접 넣는 방식이 아니라, PPG waveform의 characteristic point를 먼저 검출하고, 그 점들로부터 feature를 구성한 뒤 regression model에 입력한다.

### 3.2 데이터셋

사용한 데이터셋은 Guilin People’s Hospital에서 수집된 PPG-BP database이다. 데이터는 customized portable hardware system을 통해 수집되었다. 하드웨어는 PPG sensor probe, MSP430FG4618 microcontroller, Android application으로 구성되며, PPG sensor는 660 nm red light와 905 nm infrared LED를 사용한다. sampling rate는 1 kHz이고, ADC는 12-bit이며, hardware bandpass filter 범위는 0.5–12 Hz이다.

Reference BP는 Omron HEM-7201 upper arm monitor를 사용해 측정되었다. PPG는 왼손 검지에서 수집되었고, 병원 환경에서 간호사가 혈압을 측정하였다. 원래 데이터는 219명의 피험자로부터 657개 PPG signal을 포함하며, 각 피험자당 3개의 PPG segment가 수집되었다. 각 PPG signal은 2100 sampling points, 즉 2.1초 길이이며 sampling rate는 1 kHz이다.

논문은 skewness-based signal quality index(SQI)를 사용해 signal quality를 평가했고, 최종적으로 125명의 subject에서 218개 signal을 사용했다고 설명한다. 여기서 주의할 점은 논문 내에 “219 participants”라는 표현과 “218 recordings, 125 subjects”라는 표현이 함께 등장한다는 것이다. 원 데이터는 219명에서 시작했지만, 품질 평가 후 최종 모델링에는 125명의 218개 recording이 사용된 것으로 해석하는 것이 타당하다.

최종 데이터의 통계는 다음과 같다. 평균 나이는 $57 \pm 15$세, 남성은 104명으로 48%이다. 평균 height는 $161 \pm 8$ cm, 평균 weight는 $60 \pm 11$ kg이다. 평균 SBP는 $127 \pm 20$ mmHg, 평균 MAP는 $87 \pm 13$ mmHg, 평균 DBP는 $71 \pm 11$ mmHg이다. 본문에는 SBP 범위가 80–174 mmHg, MAP 범위가 58–128 mmHg, DBP 범위가 42–104 mmHg라고 제시되어 있다.

### 3.3 Signal quality assessment

PPG signal에는 적합한 waveform과 부적합한 waveform이 존재한다. 적합한 waveform은 systolic peak, diastolic peak, dicrotic notch가 명확하게 관찰된다. 반면 부적합 waveform은 dicrotic notch나 주요 peak 구조가 뚜렷하지 않아 feature extraction이 어렵다. 논문은 skewness-based SQI를 이용해 부적합한 PPG waveform을 제거했다.

이 과정은 매우 중요하다. 제안 방법은 peak, notch, derivative point 등 morphology feature에 크게 의존하기 때문에 waveform quality가 낮으면 feature가 잘못 계산될 수 있다. 따라서 signal quality control은 regression model 자체만큼 중요한 전처리 단계이다.

### 3.4 PPG normalization

전처리의 첫 단계는 min-max normalization이다. raw PPG signal을 $PPG_o$, normalized PPG signal을 $PPG_n$이라고 할 때, 논문은 다음 식을 사용한다.

$$
PPG_n=\frac{PPG_o-\min(PPG_o)}{\max(PPG_o)-\min(PPG_o)}
$$

이 normalization은 신호 amplitude를 0과 1 사이로 변환한다. 피험자별 센서 접촉 상태, 피부 특성, 손가락 위치, 광 흡수 정도 등에 따라 PPG amplitude scale이 달라질 수 있으므로, min-max normalization은 feature extraction과 후속 모델링을 안정화하는 역할을 한다.

### 3.5 Signal filtration

Raw PPG signal에는 power frequency interference, motion artifact, baseline drift, high-frequency noise가 포함될 수 있다. 논문은 여러 filtering 기법을 비교했으며, 7th order low-pass Butterworth filter, moving average, FIR, DWT 등을 검토했다. 그림 설명에 따르면 Butterworth filter가 high-frequency noise를 줄이고 systolic peak와 diastolic peak를 더 뚜렷하게 만들어 feature extraction에 적합한 것으로 판단된다.

본문은 최종적으로 어떤 filter를 모든 실험에 사용했는지 완전히 엄밀하게 수식으로 정리하지는 않지만, Fig. 6 설명과 본문 내용상 Butterworth low-pass filter가 주요 전처리 필터로 사용된 것으로 해석된다.

### 3.6 Best single PPG cycle 선택

각 PPG signal은 2.1초 길이이며 두 개 이상의 cardiac cycle을 포함할 수 있다. 논문은 이 중 하나의 단일 PPG cycle을 선택하여 feature extraction에 사용한다. 선택 기준은 “maximum systolic amplitude”를 갖는 PPG cycle이며, 이를 best PPG cycle($PPG_B$)이라고 부른다.

Cycle detection algorithm은 다음 논리를 따른다. 먼저 연속 PPG signal($PPG_S$)에서 가능한 PPG cycle($PPG_C$)을 찾는다. 각 cycle은 시작점 $P_s$, dicrotic notch $z$, ending point $P_e$를 consecutive minima로 보고, systolic peak $x$와 diastolic peak $y$를 consecutive maxima로 본다. Python NumPy의 `find_peak` 기능을 이용해 peak 탐색 시간을 줄인다. 유효한 cycle은 systolic peak가 diastolic peak보다 크고, dicrotic notch가 시작점과 끝점보다 큰 조건을 만족해야 한다.

조건을 만족하는 PPG cycle들이 candidate list에 저장되고, 그중 systolic peak $x$가 가장 큰 cycle이 최종 $PPG_B$로 선택된다.

이 방법은 feature extraction을 단순화한다는 장점이 있다. 그러나 PPG signal 내 여러 cycle의 평균 정보를 사용하지 않고 하나의 best cycle만 사용하기 때문에, 선택된 cycle이 우연히 noise나 비정상적 waveform을 포함하면 결과가 민감해질 수 있다. 논문은 SQI와 filtering으로 이 문제를 줄이려 하지만, cycle-level robustness에 대한 추가 분석은 제한적이다.

### 3.7 Feature extraction

논문은 최종 선택된 $PPG_B$와 derivative 및 FFT에서 총 46개의 signal feature를 추출한다. 여기에 age와 gender를 추가하여 최종 feature 수는 48개가 된다.

PPG 원 신호에서는 21개 feature를 추출한다. 여기에는 systolic peak $x$, diastolic peak $y$, dicrotic notch $z$, pulse interval $t_{pi}$, augmentation index $y/x$, alternative augmentation index $(x-y)/x$, dicrotic notch와 systolic peak의 ratio $z/x$, negative relative augmentation index $(y-x)/x$, systolic peak time $t_1$, dicrotic notch time $t_2$, diastolic peak time $t_3$, systolic-diastolic peak time difference $\Delta T=t_3-t_1$, full width at half systolic peak $w$, inflection point area ratio, dicrotic notch 전후 area ratio, systolic peak rising slope $t_1/x$, diastolic peak falling slope $y/(t_{pi}-t_3)$, 그리고 여러 time ratio가 포함된다.

1차 derivative($PPG'$)에서는 8개 feature를 추출한다. 여기에는 derivative waveform의 주요 peak 및 valley와 관련된 elapsed time, interval time, 그리고 pulse interval 대비 ratio가 포함된다.

2차 derivative($PPG''$)에서는 11개 feature를 추출한다. 여기에는 $b_2/a_2$, $e_2/a_2$, $(b_2+e_2)/a_2$와 같은 amplitude ratio와 $t_{a2}$, $t_{b2}$, $t_{a2}/t_{pi}$, $t_{b2}/t_{pi}$, derivative timing combination ratio가 포함된다.

주파수 영역에서는 FFT 기반으로 6개 feature를 추출한다. 주요 component frequency $f_{base}$, primary component magnitude $|s_{base}|$, second component frequency $f_{2nd}$, second component magnitude $|s_{2nd}|$, third component frequency $f_{3rd}$, third component magnitude $|s_{3rd}|$가 포함된다.

이 feature set은 PPG waveform의 morphology, timing, amplitude ratio, derivative dynamics, spectral characteristics를 폭넓게 포함한다. Age와 gender를 추가한 점은 혈압이 demographic factor와도 관련된다는 현실적 고려를 반영한다.

### 3.8 Feature selection: CFS

Correlation-based Feature Selection(CFS)은 target과 관련성이 높고 feature들끼리 중복성이 낮은 feature를 선택하는 방법이다. 논문은 feature set과 response value 간의 거리 기반 correlation을 계산하여 feature subset의 적합도를 평가한다.

두 response value $y_i$, $y_j$의 차이는 다음과 같이 정의된다.

$$
E_y=y_i-y_j
$$

해당 feature vector 간 거리는 다음과 같이 정의된다.

$$
E_X=
\begin{cases}
\sqrt{\frac{\sum_{k=1}^{n}(X_{i,k}-X_{j,k})^2}{n}}, & \text{if } E_y \ge 0 \
-\sqrt{\frac{\sum_{k=1}^{n}(X_{i,k}-X_{j,k})^2}{n}}, & \text{if } E_y < 0
\end{cases}
$$

그 다음 $E_X$와 $E_y$ 사이의 correlation coefficient $R_{cfs}$를 계산한다.

$$
R_{cfs}=
\frac{
\frac{\sum_i(E_{X_i}-\bar{E}*X)(E*{y_i}-\bar{E}*y)}{n-1}
}{
\sqrt{
\frac{\sum_i(E*{X_i}-\bar{E}*X)^2}{n-1}
\frac{\sum_i(E*{y_i}-\bar{E}_y)^2}{n-1}
}
}
$$

목표는 $R_{cfs}$를 최대화하는 feature subset을 선택하는 것이다. CFS 적용 후 feature 수는 SBP에 대해 15개, MAP와 DBP에 대해 각각 16개로 줄어든다.

CFS가 선택한 feature에는 age, gender, $f_{3rd}$, $f_{2nd}$ 또는 $|s_{2nd}|$, $f_{base}$ 또는 $|s_{base}|$, derivative timing feature, area ratio, $\Delta T$, pulse interval, dicrotic notch, diastolic peak 등이 포함된다. 이는 demographic feature와 frequency-domain feature가 주요 feature로 반복 선택되었음을 보여준다.

### 3.9 Feature selection: ReliefF

ReliefF는 instance 기반 feature weighting 방법이다. 무작위로 선택한 target instance에 대해 nearest hit와 nearest miss를 비교하며, target 구분 또는 regression response 차이를 잘 설명하는 feature에 높은 weight를 부여한다. 논문은 ReliefF를 continuous 및 binary data에 적용 가능한 통계 기반 feature selection 방법으로 설명한다.

모든 feature weight는 처음에 0으로 초기화된다.

$$
W_{F_i}(t)=0.0
$$

반복 과정에서 feature weight는 다음과 같이 갱신된다.

$$
W_{F_i}(t)=W_{F_i}(t-1)-\frac{\Delta F_i(R_t,H)}{m}+\frac{\Delta F_i(R_t,M)}{m}
$$

여기서 $R_t$는 무작위로 선택된 target instance, $H$는 nearest hit, $M$은 nearest miss이다. feature value difference는 다음과 같이 정의된다.

$$
\Delta F_i(I_1,I_2)=
\begin{cases}
0, & \text{if value}(F_i,I_1)=\text{value}(F_i,I_2) \
1, & \text{otherwise}
\end{cases}
$$

ReliefF 적용 후 feature 수는 SBP와 DBP에서 15개, MAP에서 16개로 줄어든다. ReliefF가 선택한 feature 역시 age와 gender를 포함하며, frequency magnitude, derivative ratio, systolic peak rising slope, augmentation-related feature, dicrotic notch, diastolic peak 등을 포함한다.

논문 결과에서는 CFS보다 ReliefF가 더 좋은 feature subset을 제공하였다. 최종 최고 성능은 ReliefF-selected features와 SVR의 조합에서 나왔다.

### 3.10 Regression model

논문은 네 가지 nonlinear regression model을 비교한다. Linear regression은 PPG feature와 BP 사이의 관계를 충분히 설명하지 못하고 AAMI/BHS 기준을 만족하지 못한다고 판단하여 주요 비교 대상에서 제외되었다.

첫 번째 모델은 Support Vector Regression(SVR)이다. SVR은 nonlinear mapping을 통해 feature를 high-dimensional space로 변환하고, RBF kernel을 이용해 nonlinear relationship을 모델링한다. 논문의 SVR 식은 다음과 같다.

$$
BP_e=\sum_{i=1}^{m}\omega_i\varphi_i(F)+b
$$

여기서 $F$는 input feature set, $\varphi_i(F)$는 nonlinear transformation, $\omega_i$는 regression coefficient, $b$는 bias이다. 실험에서 SVR의 주요 hyperparameter는 $C=100$, $\epsilon=0.1$, kernel은 RBF이다.

두 번째 모델은 Random Forest Regression(RFR)이다. RFR은 여러 decision tree의 예측값을 평균내어 최종 혈압을 예측한다.

$$
BP_e=\frac{1}{B}\sum_{b=1}^{B}T_b(x)
$$

여기서 $T_b$는 $b$번째 regression tree이고, $T_b(x)$는 해당 tree의 예측값이다. 논문에서 RFR은 n_estimators 150, criterion은 MAE, max depth는 13으로 설정된다.

세 번째 모델은 Decision Tree Regression(DTR)이다. DTR은 feature 값을 기준으로 데이터를 반복적으로 분할하여 leaf node에서 연속값을 예측한다. 논문에서는 criterion MAE, min samples leaf 17, max depth 12, min samples split 5를 사용한다.

네 번째 모델은 K-nearest Neighbor Regression(KNR)이다. KNR은 query point와 가장 가까운 $K$개의 sample을 찾고, 그 target 값의 평균을 예측값으로 사용한다. Euclidean distance는 다음과 같이 정의된다.

$$
E_d=\sqrt{\sum_{i=1}^{K}(x_i-y_i)^2}
$$

KNR의 예측값은 다음과 같이 계산된다.

$$
BP_e=\frac{1}{K}\sum_{i=1}^{K}y_i
$$

논문에서는 number of neighbours 51, leaf size 45, distance는 Euclidean으로 설정한다.

### 3.11 Validation 및 평가 지표

데이터는 ID 기준으로 training 80%, test 20%로 분할된다. 각 recorded signal에는 unique ID가 있으며, 논문은 이를 이용해 training set과 test set 간 subject overlap을 방지했다고 설명한다. 모델 검증에는 10-fold cross-validation이 사용되며, hyperparameter는 grid search로 최적화하였다.

평가 지표는 $R^2$, MAE, RMSE, MSE, ME, STD이다. 논문에서 제시한 주요 식은 다음과 같다.

$$
R^2=1-\frac{\sum_n(BP_a-BP_e)^2}{\sum_n(BP_a-\bar{BP})^2}
$$

$$
MAE=\frac{1}{n}\sum_n |BP_a-BP_e|
$$

$$
MSE=\frac{1}{n}\sum_n(BP_a-BP_e)^2
$$

$$
RMSE=\sqrt{MSE}
$$

$$
ME=\frac{1}{n}\sum_n(BP_a-BP_e)
$$

$$
STD=\sqrt{\frac{\sum_n(BP_a-BP_e-ME)^2}{n-1}}
$$

여기서 $BP_a$는 reference BP, $BP_e$는 estimated BP, $\bar{BP}$는 reference BP의 평균이다.

## 4. 실험 및 결과

### 4.1 전체 feature 사용 결과

먼저 feature selection 없이 48개 전체 feature를 사용해 SVR, RFR, DTR, KNR을 비교했다. Table 6에 따르면 전체 feature 사용 시 SVR이 가장 좋은 성능을 보였다.

SBP 추정에서 SVR은 $R^2=0.71$, MAE 11.78 mmHg, RMSE 14.85 mmHg, MSE 239.87을 기록했다. MAP 추정에서는 $R^2=0.72$, MAE 5.53 mmHg, RMSE 9.72 mmHg를 기록했다. DBP 추정에서는 $R^2=0.61$, MAE 7.89 mmHg, RMSE 9.78 mmHg를 기록했다.

전체 feature만 사용할 경우 성능은 임상 기준을 만족하기 어렵다. 특히 SBP MAE 11.78 mmHg는 높은 편이며, 이는 불필요하거나 중복된 feature가 모델에 포함되어 과적합 또는 noise sensitivity를 유발할 수 있음을 시사한다.

### 4.2 CFS feature selection 결과

CFS 적용 후 feature 수는 SBP 15개, MAP 16개, DBP 16개로 줄어들었다. Table 7에 따르면 CFS-selected features를 사용했을 때 모든 모델 성능이 전체 feature 사용보다 크게 개선되었으며, 이 경우에도 SVR이 가장 좋은 성능을 보였다.

SBP에서 SVR은 $R^2=0.90$, MAE 3.82 mmHg, RMSE 5.28 mmHg를 기록했다. MAP에서는 $R^2=0.91$, MAE 2.62 mmHg, RMSE 3.89 mmHg를 기록했다. DBP에서는 $R^2=0.92$, MAE 2.97 mmHg, RMSE 4.84 mmHg를 기록했다.

논문은 전체 feature 사용 SVR과 비교했을 때, CFS-selected features를 사용한 SVR의 $R^2$가 SBP에서 26.76%, MAP에서 26.38%, DBP에서 48.38% 증가했다고 보고한다. 이는 feature selection이 단순히 feature 수를 줄이는 것뿐 아니라 예측 성능에도 결정적인 영향을 미친다는 점을 보여준다.

### 4.3 ReliefF feature selection 결과

ReliefF를 적용한 경우 feature 수는 SBP 15개, MAP 16개, DBP 15개로 줄었다. Table 8에 따르면 ReliefF-selected features와 SVR의 조합이 전체 실험 중 최고 성능을 보였다.

SBP 추정에서 SVR은 $R^2=0.93$, MAE 2.49 mmHg, RMSE 3.45 mmHg, MSE 12.78을 달성했다. MAP 추정에서는 $R^2=0.95$, MAE 1.62 mmHg, RMSE 2.52 mmHg, MSE 6.38을 달성했다. DBP 추정에서는 $R^2=0.95$, MAE 1.69 mmHg, RMSE 3.60 mmHg, MSE 12.78을 달성했다.

다른 모델과 비교하면 DTR도 비교적 좋은 성능을 보였지만 SVR에는 미치지 못했다. 예를 들어 ReliefF 사용 시 DTR은 SBP MAE 3.32, MAP MAE 2.26, DBP MAE 2.80을 기록했다. RFR과 KNR은 상대적으로 성능이 낮았다. 이는 해당 데이터셋과 feature set에서 RBF kernel SVR이 nonlinear relationship을 가장 잘 포착했음을 의미한다.

논문은 전체 feature 사용 SVR과 비교했을 때 ReliefF-selected features를 사용한 SVR의 $R^2$가 SBP에서 30.98%, MAP에서 31.94%, DBP에서 55.73% 증가했다고 보고한다. ReliefF가 CFS보다 더 좋은 성능을 보인 이유는 nearest neighbor 구조를 통해 feature의 지역적 구분력을 더 잘 반영했기 때문일 수 있다. 다만 논문은 왜 ReliefF가 CFS보다 우수했는지에 대한 정량적 feature importance 분석을 깊게 제공하지는 않는다.

### 4.4 Error histogram, regression plot, Bland–Altman plot

Fig. 13은 ReliefF + SVR 모델의 SBP, MAP, DBP estimation error histogram을 보여준다. 세 혈압 지표 모두 error가 대체로 0 주변에 분포한다. 이는 모델의 평균 bias가 크지 않다는 것을 의미한다.

Fig. 14의 regression plot은 estimated BP와 reference BP 사이의 관계를 보여준다. ReliefF + SVR 모델은 SBP, MAP, DBP 모두에서 reference value와 높은 선형 관계를 보인다. 특히 Table 8에서 $R^2$가 0.93–0.95 수준이므로, test set에서 상당한 설명력을 갖는 것으로 보고된다.

Bland–Altman plot은 estimated BP와 reference BP 사이의 agreement를 평가한다. 논문에 따르면 95% confidence interval에서 limits of agreement는 SBP의 경우 [-0.287 mmHg, 6.98 mmHg], MAP의 경우 [-5.31 mmHg, 4.83 mmHg], DBP의 경우 [-5.16 mmHg, 4.98 mmHg]로 제시된다. 다만 SBP의 limits가 평균차 중심으로 비대칭적으로 보이는 점은 일반적인 $md \pm 1.96sd$ 표현과 다소 어색할 수 있다. 제공 텍스트만으로는 이 값들이 정확히 어떻게 계산되었는지 추가 확인이 어렵다.

논문은 극단적으로 낮거나 높은 혈압값에서는 추정이 상대적으로 부정확하다고 명시한다. 이는 training dataset에 extreme BP sample이 적기 때문이라고 설명한다. 이 한계는 실제 임상 적용에서 중요하다. 왜냐하면 cuffless BP monitoring이 가장 필요한 대상 중 하나가 고혈압 또는 저혈압 위험군이기 때문이다.

### 4.5 AAMI 기준 평가

AAMI 기준에 따르면 BP measurement device는 ME가 5 mmHg 이하, STD가 8 mmHg 이하이어야 하며, 최소 85명 이상의 피험자가 필요하다. Table 9에 따르면 ReliefF + SVR 모델은 125명의 subject 기준으로 AAMI 기준을 만족한다.

SBP의 ME는 0.533 mmHg, STD는 4.175 mmHg이다. MAP의 ME는 -0.299 mmHg, STD는 2.441 mmHg이다. DBP의 ME는 -0.049 mmHg, STD는 2.188 mmHg이다. 세 지표 모두 ME와 STD가 AAMI 허용 기준보다 낮으므로 논문은 제안 방법이 AAMI standard를 만족한다고 결론낸다.

다만 AAMI standard는 실제 의료기기 검증에서 특정 protocol과 population distribution, validation procedure를 요구한다. 이 논문은 AAMI의 numerical threshold와 subject 수 조건을 만족한다고 보고하지만, 의료기기 인증 수준의 정식 protocol을 모두 수행했다고 보기는 어렵다. 이 점은 해석 시 주의해야 한다.

### 4.6 BHS 기준 평가

BHS 기준은 추정 오차가 5, 10, 15 mmHg 이내에 들어오는 sample 비율을 기준으로 등급을 부여한다. Grade A는 각각 60%, 85%, 95% 이상을 요구한다.

Table 10에 따르면 제안 방법의 누적 오차 비율은 SBP에서 5 mmHg 이하 91%, 10 mmHg 이하 95%, 15 mmHg 이하 99%이다. MAP에서는 각각 93%, 96%, 100%이고, DBP에서는 각각 94%, 96%, 100%이다. 따라서 SBP, MAP, DBP 모두 BHS Grade A를 달성한다.

이 결과는 제안 방법이 평균 오차뿐 아니라 일정 threshold 내 예측 비율 측면에서도 매우 우수하다고 주장하는 근거가 된다. 그러나 앞서 언급했듯 extreme BP에서 성능이 떨어질 수 있으므로, Grade A 결과가 모든 혈압 범위에서 균일한 정확도를 보장하는 것은 아니다.

### 4.7 기존 연구와 비교

Table 11은 여러 기존 PPG 기반 cuffless BP estimation 연구와 제안 방법을 비교한다. 논문은 데이터셋, sample 수, validation method, metric이 서로 다르기 때문에 직접적인 공정 비교는 어렵다고 명시한다.

제안 방법은 218 recordings, 125 subjects를 사용하고, time 및 frequency domain features of PPG, $PPG'$, $PPG''$, ReliefF, CFS, SVR을 결합한다. 보고된 성능은 Table 11 기준으로 SBP $2.49 \pm 7.82$ mmHg, MAP $1.62 \pm 5.47$ mmHg, DBP $1.69 \pm 4.02$ mmHg이다. 여기서 앞의 값은 MAE이고 뒤의 값은 STD로 해석된다. 그러나 Table 9의 STD 값과 Table 11의 STD 값이 서로 다르게 제시된다. Table 9에서는 SBP STD 4.175, MAP STD 2.441, DBP STD 2.188이고, Table 11에서는 SBP STD 7.82, MAP STD 5.47, DBP STD 4.02이다. 이 불일치는 논문 내부의 중요한 보고 문제로 보인다. 제공된 텍스트만으로는 어떤 STD 값이 최종 기준인지 확정하기 어렵다.

기존 연구 중 Mousavi et al.은 MIMIC-II 441 subjects에서 FFT, PCA, AdaBoostR을 사용해 SBP $3.97 \pm 7.99$, DBP $2.43 \pm 3.37$을 보고했다. Chowdhury et al.은 126 subjects에서 ReliefF, CFS, GPR을 사용해 SBP $3.02 \pm 9.29$, DBP $1.74 \pm 5.54$를 보고했다. El-Hajj et al.은 MIMIC-II 942 subjects에서 DWT, BiLSTM, LSTM, Attention을 사용해 SBP $4.51 \pm 7.81$, DBP $2.6 \pm 4.41$을 보고했다. 제안 방법은 MAE 기준으로 이들보다 더 낮은 값을 보고한다.

하지만 이러한 비교는 주의해야 한다. 일부 연구는 MIMIC-II나 MIMIC-III처럼 ICU 기반 대규모 데이터를 사용하고, 일부는 서로 다른 protocol과 train-test split을 사용한다. 제안 방법은 최종 subject 수가 125명으로 상대적으로 작고, 특정 PPG-BP database에서 검증되었다. 따라서 낮은 MAE가 범용 일반화 성능을 의미한다고 단정하기는 어렵다.

## 5. 강점, 한계

### 5.1 강점

이 논문의 첫 번째 강점은 PPG 단일 신호만을 이용한다는 점이다. ECG, arterial line, multiple sensor가 필요하지 않으므로 실제 웨어러블 또는 스마트폰 기반 건강 모니터링 시스템으로 확장하기 쉽다. 단일 PPG 채널 기반 모델은 장치 비용과 사용자 부담을 줄일 수 있다.

두 번째 강점은 feature extraction 범위가 넓다는 점이다. 원 PPG waveform뿐 아니라 1차 derivative, 2차 derivative, FFT 기반 frequency-domain feature, demographic feature를 모두 포함한다. 이로 인해 시간적 morphology, 미분 기반 waveform dynamics, spectral component, 인구학적 요인을 동시에 반영한다.

세 번째 강점은 feature selection의 효과를 명확히 검증했다는 점이다. 전체 48개 feature를 사용했을 때보다 CFS와 ReliefF로 feature를 줄였을 때 성능이 크게 향상되었다. 특히 ReliefF + SVR 조합은 모든 혈압 지표에서 가장 좋은 결과를 보였고, 이는 feature selection이 과적합 감소와 중요한 feature 강조에 효과적임을 보여준다.

네 번째 강점은 다양한 nonlinear regression model을 비교했다는 점이다. SVR, RFR, DTR, KNR을 같은 feature 조건에서 평가함으로써 어떤 regression model이 해당 문제에 적합한지 비교했다. 실험 결과 SVR이 일관되게 가장 우수했으며, 이는 PPG feature와 BP 사이의 nonlinear relationship을 kernel method가 잘 포착했음을 시사한다.

다섯 번째 강점은 AAMI와 BHS 기준을 모두 사용했다는 점이다. 단순히 MAE나 $R^2$만 보고하지 않고, 임상 혈압 측정 장치 평가에 자주 언급되는 기준을 적용하여 결과를 해석했다. 이는 연구의 임상적 의미를 강조하는 데 도움이 된다.

여섯 번째 강점은 데이터셋과 코드 또는 feature dataset link를 공개했다는 점이다. Kaggle 및 GitHub link가 제공되어 있어 재현 가능성이 상대적으로 높다. 이는 자체 수집 데이터가 공개되지 않는 많은 의료 AI 연구와 비교할 때 장점이다.

### 5.2 한계

가장 중요한 한계는 최종 사용 데이터 규모가 작다는 점이다. 원 데이터는 219 participants로 설명되지만, quality assessment 후 최종적으로 125 subjects, 218 recordings가 사용된다. 이 정도 규모는 machine learning model의 일반화 성능을 평가하기에는 제한적이다. 특히 극단적 고혈압·저혈압 sample이 부족해 extreme BP에서 성능이 낮다고 논문 스스로 인정한다.

두 번째 한계는 PPG cycle 선택 방식이다. 논문은 2.1초 신호에서 maximum systolic peak를 가진 단일 cycle을 best cycle로 선택한다. 이는 구현이 단순하지만, 하나의 cycle이 전체 생리 상태를 대표한다고 가정한다. 여러 cycle 평균이나 cycle-level uncertainty를 반영하지 않기 때문에 noise나 peak detection error에 민감할 수 있다.

세 번째 한계는 subject-level split이 실제로 얼마나 엄밀히 수행되었는지 설명이 충분하지 않다는 점이다. 본문은 unique ID를 사용해 training과 testing set 간 subject overlap을 방지했다고 말한다. 그러나 10-fold cross-validation이 어떤 수준에서 수행되었는지, 즉 recording-level인지 subject-level인지가 완전히 명확하지 않다. PPG-BP 연구에서는 같은 subject의 유사한 signal이 train과 test에 동시에 들어가면 성능이 과대평가될 수 있으므로, 이 부분은 매우 중요하다.

네 번째 한계는 clinical standard 해석의 한계이다. AAMI와 BHS의 numerical criteria를 만족한다고 해서 의료기기 수준의 정식 validation을 완료한 것은 아니다. 실제 AAMI/ISO protocol은 피험자의 혈압 분포, 측정 반복, reference observer, device comparison 절차 등 더 구체적인 조건을 요구한다. 논문은 threshold-based comparison을 수행했지만, full clinical validation study라고 보기는 어렵다.

다섯 번째 한계는 논문 내부의 수치 불일치이다. Abstract에서는 DBP MAE가 1.43 mmHg라고 제시되지만, Table 8과 Table 11에서는 DBP MAE가 1.69 mmHg로 제시된다. 또한 Table 9의 STD 값과 Table 11의 STD 값이 서로 다르다. 예를 들어 SBP STD는 Table 9에서 4.175 mmHg인데 Table 11에서는 7.82 mmHg로 제시된다. 이러한 불일치는 결과 해석의 신뢰성을 약화시킨다.

여섯 번째 한계는 calibration-free라고 주장하지만, 실제 사용 조건에서의 calibration-free 의미가 제한적이라는 점이다. 논문은 cuff나 calibration 없이 PPG characteristic feature만으로 BP를 추정한다고 주장한다. 그러나 reference BP label로 학습된 supervised model이고, age와 gender 같은 subject feature를 사용한다. 또한 새로운 환경, 다른 센서, 다른 인구집단에서 별도 calibration 없이 같은 성능이 유지되는지는 검증되지 않았다.

일곱 번째 한계는 외부 데이터셋 검증이 없다는 점이다. 제안 방법은 하나의 PPG-BP database에서만 평가되었다. PPG 기반 혈압 추정은 센서, 측정 부위, 인구집단, waveform quality, 혈압 분포에 따라 성능이 크게 달라질 수 있다. 따라서 외부 데이터셋에서의 out-of-distribution generalization 검증이 필요하다.

여덟 번째 한계는 비교 대상 deep learning 모델이 제한적이라는 점이다. 논문은 기존 연구와 table comparison은 제공하지만, 같은 데이터 split에서 CNN, LSTM, Transformer, modern time-series model과 직접 비교하지 않는다. 따라서 classical ML feature-based approach가 최신 end-to-end model보다 우수하다고 결론내릴 수는 없다.

### 5.3 비판적 해석

이 논문은 PPG 기반 혈압 추정을 위한 실용적인 feature-based machine learning pipeline을 제시한다. 특히 ReliefF feature selection과 SVR의 조합이 매우 좋은 결과를 보였다는 점은, 작은 데이터셋에서는 복잡한 deep learning 모델보다 잘 설계된 feature와 kernel regression이 더 안정적일 수 있음을 보여준다.

하지만 보고된 성능이 매우 높기 때문에 검증 방식과 데이터 분할의 엄밀성이 특히 중요하다. 최종 데이터가 125 subjects, 218 recordings로 작고, PPG signal이 2.1초로 짧으며, feature selection과 hyperparameter tuning이 같은 데이터셋 내에서 수행되었다. 이러한 조건에서는 test set이 완전히 독립적이지 않거나, feature selection 과정에서 data leakage가 발생하면 성능이 과대평가될 수 있다. 논문은 ID 기반 split을 했다고 설명하지만, feature selection과 cross-validation이 어떤 순서로 test set과 분리되었는지 구체적으로 충분히 설명하지 않는다.

또한 extreme BP에서 성능이 낮다는 점은 실제 사용에서 중요한 문제다. 혈압 모니터링 시스템은 정상 혈압보다 위험 혈압 구간에서 더 높은 신뢰성이 필요하다. 그러나 데이터 분포가 정상 또는 중간 혈압에 치우쳐 있으면 모델은 평균적인 범위에서만 잘 작동하고, 고위험 구간에서 underestimation 또는 overestimation을 보일 수 있다.

그럼에도 이 연구는 단일 PPG signal에서 meaningful morphology feature를 추출하고, feature selection으로 성능을 개선한 pipeline을 명확히 보여준다. 공개 데이터와 비교 가능한 구조를 제공한다는 점에서 후속 연구의 baseline으로 활용할 수 있다.

## 6. 결론

이 논문은 PPG characteristic feature와 nonlinear regression model을 이용한 cuffless, non-invasive BP estimation 방법을 제안했다. PPG-BP database에서 수집된 PPG 신호를 quality assessment, normalization, filtering 과정을 거쳐 전처리하고, maximum systolic peak를 가진 best single PPG cycle을 선택하여 feature extraction을 수행하였다. 원 PPG, 1차 derivative, 2차 derivative, FFT에서 총 46개 feature를 추출하고, age와 gender를 추가하여 48개 feature를 구성하였다.

Feature selection에는 CFS와 ReliefF를 사용했으며, regression model로는 SVR, RFR, DTR, KNR을 비교하였다. 실험 결과 ReliefF-selected features와 SVR의 조합이 가장 높은 성능을 보였다. Table 8 기준으로 SBP는 $R^2=0.93$, MAE 2.49 mmHg, RMSE 3.45 mmHg를 달성했고, MAP는 $R^2=0.95$, MAE 1.62 mmHg, RMSE 2.52 mmHg를 달성했으며, DBP는 $R^2=0.95$, MAE 1.69 mmHg, RMSE 3.60 mmHg를 달성하였다. 또한 AAMI numerical criteria와 BHS Grade A 기준을 만족한다고 보고하였다.

이 연구의 주요 기여는 단일 PPG 신호에서 다양한 morphology, derivative, frequency-domain feature를 추출하고, feature selection과 nonlinear regression을 결합하여 높은 혈압 추정 성능을 보였다는 점이다. 특히 ReliefF가 중요한 feature를 효과적으로 선택하고, SVR이 nonlinear relationship을 잘 모델링한 것으로 나타났다.

향후 연구에서는 더 큰 규모의 데이터셋, 다양한 인구집단, 고혈압 및 저혈압 위험군, 장기 real-world monitoring 환경에서 검증이 필요하다. 또한 외부 데이터셋 검증, subject-level cross-validation의 명확한 설계, extreme BP 구간 성능 분석, deep learning 모델과의 동일 조건 비교가 요구된다. 논문 내부의 일부 수치 불일치도 후속 검증에서 명확히 정리되어야 한다. 그럼에도 이 논문은 PPG feature 기반 cuffless BP estimation 연구에서 재현 가능한 feature-selection-regression pipeline을 제시했다는 점에서 의미 있는 연구로 평가할 수 있다.
