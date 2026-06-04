# A blood pressure estimation approach based on single-channel photoplethysmography differential features

* **저자**: Qin Chen, Xuezhi Yang, Yawei Chen, Xuesong Han, Zheng Gong, Dingliang Wang, Jie Zhang
* **발표연도**: 2024

## 1. 논문 개요

이 논문은 단일 채널 photoplethysmography(PPG) 신호만을 이용하여 비침습적으로 혈압을 추정하는 방법을 제안한다. 핵심 목표는 기존의 PPG feature를 혈압값에 직접 매핑하는 방식이 아니라, 개인별 baseline을 기준으로 PPG feature의 변화량, 즉 differential feature(DF)를 계산하고 이를 혈압 변화량과 연결함으로써 장기간 혈압 변화를 더 정확하게 추정하는 것이다.

기존 PPG 기반 혈압 추정 연구는 크게 PTT(Pulse Transit Time), PAT(Pulse Arrival Time), PWA(Pulse Wave Analysis), deep learning 기반 end-to-end 방법으로 나눌 수 있다. PTT와 PAT 기반 방법은 혈압과 맥파 전달 시간 사이의 생리적 관계를 이용하지만, 보통 ECG와 PPG처럼 두 개 이상의 생체신호를 동시에 측정해야 한다. PWA 기반 방법은 단일 PPG waveform에서 morphology feature를 추출하여 혈압을 추정할 수 있다는 장점이 있지만, 어떤 feature가 혈압 변화와 실제로 어떤 생리적 관계를 갖는지 명확하지 않은 경우가 많다. Deep learning 기반 방법은 PPG waveform을 직접 입력으로 사용해 feature를 자동으로 학습하지만, 모델 내부 해석이 어렵고 feature 변화가 혈압 변화에 미치는 영향을 명확히 설명하기 어렵다.

이 논문이 다루는 연구 문제는 “PPG의 절대 feature 값보다, 개인 baseline 대비 PPG feature 변화량이 혈압 변화량을 더 잘 설명할 수 있는가?”이다. 저자들은 90명의 건강한 참가자를 대상으로 약 3개월 동안 PPG와 cuff 기반 혈압을 반복 측정하였고, 각 참가자의 첫 측정값을 baseline으로 설정하였다. 이후 측정된 PPG feature에서 baseline feature를 뺀 값을 differential feature로 정의하고, 이후 측정 혈압에서 baseline 혈압을 뺀 값을 blood pressure differential(BPD)로 정의하였다. 모델은 DF를 입력으로 하여 SBP와 DBP의 변화량을 먼저 예측하고, 마지막에 개인 baseline BP를 더해 최종 혈압값을 산출한다.

문제의 중요성은 명확하다. 혈압은 시간, 스트레스, 식이, 자세, 생리 상태, 혈관 탄성 등 다양한 요인에 따라 변동한다. cuff 기반 혈압계는 반복적인 압박이 필요해 일상적·장기적 모니터링에 불편하다. 반면 PPG는 손가락이나 웨어러블 센서로 쉽게 측정 가능하므로 연속 또는 반복 혈압 모니터링에 적합하다. 다만 개인별 생리 차이가 크기 때문에 PPG feature를 혈압값에 직접 매핑하면 일반화가 어려울 수 있다. 이 논문은 baseline 대비 변화량을 사용함으로써 개인차를 일부 제거하고 장기 추적 혈압 추정의 정확도와 해석 가능성을 높이려 한다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 절대적인 PPG feature보다 “개인 내부에서의 PPG feature 변화”가 혈압 변화와 더 직접적으로 연결될 수 있다는 것이다. 같은 PPG waveform morphology라도 사람마다 혈관 탄성, 나이, 체중, 말초혈관저항, 심박 특성이 다르기 때문에 feature의 절대값은 개인차를 크게 반영한다. 반면 동일한 사람에서 baseline 대비 feature가 어떻게 변했는지를 보면, 개인 고유 특성의 영향을 줄이고 혈압 변화와 관련된 동적 정보를 더 잘 포착할 수 있다.

저자들은 이를 위해 첫 번째 측정 시점의 혈압과 PPG feature를 개인 baseline으로 설정한다. 이후 측정 시점마다 PPG feature 변화량을 계산하고, 이를 이용해 baseline 대비 혈압 변화량을 추정한다. 최종 혈압은 추정된 변화량에 baseline 혈압을 더해 얻는다. 즉, 모델이 직접 $BP$를 예측하는 것이 아니라 먼저 $\Delta BP$를 예측한다는 점이 중요하다.

이 접근법은 다음과 같은 차별점을 갖는다. 첫째, ECG 없이 단일 채널 PPG만 사용하므로 센서 구성이 단순하다. 둘째, deep learning end-to-end 방식과 달리 명시적인 morphology feature와 differential feature를 사용하기 때문에 feature 해석이 가능하다. 셋째, 단순 feature 기반 추정이 아니라 baseline feature와 후속 feature의 차이를 사용하므로 개인 생리적 편차를 줄이는 calibration-based 전략을 취한다. 넷째, feature selection을 통해 혈압 변화와 관련성이 높고 중복성이 낮은 DF subset을 선택함으로써 모델 입력의 효율성과 해석 가능성을 높인다.

논문은 특히 PPG DFs와 BP changes 사이의 상관성이 기존 PPG features와 BP 사이의 상관성보다 더 높다는 결과를 제시한다. 예를 들어 선택된 11개 feature 중 8개는 원 feature보다 differential feature로 변환했을 때 혈압 변화와 더 높은 상관성을 보였다. 이는 논문의 중심 가정, 즉 “혈압 추정에는 feature 자체보다 feature 변화가 중요하다”는 주장을 뒷받침한다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

제안 방법은 크게 네 단계로 구성된다. 첫째, 참가자로부터 cuff 기반 reference BP와 finger PPG pulse signal을 수집한다. 둘째, PPG 신호를 filtering, normalization, quality-based segmentation을 통해 전처리한다. 셋째, PPG waveform과 그 derivative에서 총 68개 feature를 추출하고, baseline 대비 differential feature를 계산한 뒤 feature selection을 수행한다. 넷째, 선택된 DF를 입력으로 machine learning regression model을 학습하여 SBP와 DBP의 변화량을 추정하고, baseline BP를 더해 최종 혈압을 계산한다.

전체 개념은 다음과 같이 요약할 수 있다.

$$
PPG_{baseline}, BP_{baseline}, PPG_{follow-up}
\rightarrow
\Delta feature
\rightarrow
\widehat{\Delta BP}
\rightarrow
\widehat{BP}=\widehat{\Delta BP}+BP_{baseline}
$$

여기서 모델의 핵심 입력은 후속 측정 시점의 feature 자체가 아니라 baseline feature와의 차이인 $\Delta f$이다.

### 3.2 데이터 수집

저자들은 90명의 건강한 참가자를 대상으로 3개월 동안 데이터를 수집하였다. 참가자는 여성 45명, 남성 45명이며, 나이는 22세에서 72세까지 분포한다. 각 참가자는 최소 26개의 데이터 sample을 제공하였다. 측정 간격은 4일을 넘지 않도록 관리되었고, 논문 discussion에서는 BP variation 관찰을 위해 최소 2일 이상의 간격을 유지했다고 설명한다.

각 측정은 다음 순서로 진행된다. 참가자가 5분 동안 앉아서 안정 상태를 유지한 뒤, 왼팔에서 cuff-type electronic monitor를 이용해 혈압을 측정한다. 이후 1분 간격을 두고 fingertip PPG sensor를 이용해 1분 동안 PPG 신호를 기록한다. PPG sampling frequency는 200 Hz이다. 총 2340개의 PPG 신호와 대응하는 BP 데이터가 수집되었다.

이 데이터셋은 공개 데이터셋이 아니라 자체 수집 데이터이며, 논문은 데이터가 confidential하다고 명시한다. 또한 참가자는 모두 건강한 사람으로 구성되었기 때문에 고혈압 환자나 심혈관 질환자의 PPG morphology 변화까지 충분히 포함했다고 보기는 어렵다.

### 3.3 Blood pressure differential 생성

각 개인의 첫 번째 측정 혈압을 baseline BP로 정의한다. 이후 측정된 혈압에서 baseline BP를 빼서 blood pressure differential(BPD)을 계산한다. 논문에서 제시한 식은 다음과 같다.

$$
\Delta BP_{i,j}=BP_{i,j+1}-BP_{i,base}
$$

여기서 $i$는 $i$번째 subject를 의미하고, $j$는 측정 index를 의미한다. $\Delta BP_{i,j}$는 해당 subject의 baseline 대비 $j$번째 후속 측정 혈압 변화량이다.

논문에 따르면 전체 SBP 범위는 85–165 mmHg이고, DBP 범위는 56–113 mmHg이다. baseline 대비 SBP 변화량은 -25 mmHg에서 39 mmHg까지, DBP 변화량은 -17 mmHg에서 32 mmHg까지 분포한다. 이는 같은 사람의 혈압도 장기간 추적 시 상당히 변할 수 있음을 보여준다.

### 3.4 PPG filtering 및 normalization

PPG 신호는 움직임, 센서 접촉 상태, baseline drift, 외부 광 간섭 등의 영향을 받을 수 있다. 저자들은 우선 raw PPG에 0.8–4 Hz bandpass filter를 적용한다. 이 범위는 human heartbeat frequency range와 관련된 구간으로 설정되었다. 이후 subject나 측정 시점에 따른 amplitude scale 차이를 줄이기 위해 z-score 형태의 normalization을 수행한다.

$$
P_n=\frac{P-\bar{P}}{SD_p}
$$

여기서 $P$는 원래 PPG 신호, $\bar{P}$는 PPG 신호 평균, $SD_p$는 원래 PPG 신호의 표준편차이다. 전처리 후 signal-to-noise ratio(SNR)가 3.32 dB에서 6.79 dB로 증가했다고 보고한다. 이는 filtering이 noise frequency component를 억제하고 심박 관련 signal component를 보존했음을 의미한다.

논문은 SNR을 다음과 같이 정의한다.

$$
SNR=
\frac{
\int_{B_1}^{B_2} S(f)df
}{
\int_{0}^{B_1} S(f)df+\int_{B_2}^{\infty}S(f)df
}
$$

여기서 $S(f)$는 PPG segment의 power spectral density이고, $B_1=0.8$ Hz, $B_2=4.0$ Hz이다. 분자는 심박 관련 주파수 범위의 신호 에너지이고, 분모는 저주파 및 고주파 noise 에너지이다.

### 3.5 고품질 PPG segment 추출

저자들은 1분 PPG 신호에서 고품질 10초 segment를 추출하기 위해 fixed-length rectangular window를 사용한다. window 길이는 2000 sampling points이고, sampling rate가 200 Hz이므로 10초에 해당한다. window interval은 500 points이므로 adjacent segment 간 overlap은 75%이다.

논문의 segment 식은 다음과 같이 제시된다.

$$
y_i=x([n(i-1)\times500+1:n\times i+2000])
$$

다만 이 식은 표기상 약간 혼동이 있다. 본문 설명에 따르면 핵심은 2000-point window를 500-point 간격으로 이동시키며 segment를 추출하는 것이다.

각 segment의 품질은 SNR과 peak variance(PV)를 사용해 평가한다. 품질 점수는 다음과 같이 정의된다.

$$
QS_i=\alpha SNR_i-PV_i
$$

$$
\alpha=sgn(SNR_i-\overline{SNR})
$$

여기서 $\overline{SNR}$는 해당 1분 PPG 신호에서 추출된 모든 10초 segment의 평균 SNR이다. $\alpha=1$이면 해당 segment의 SNR이 평균보다 높아 acceptable signal quality로 간주된다. $PV_i$는 peak amplitude fluctuation을 나타내며, 작을수록 안정적인 segment로 해석된다. 최종적으로 $QS_i$가 가장 큰 3개의 PPG segment를 선택한다.

이 과정의 목적은 단순 filtering으로 제거하기 어려운 내부 waveform anomaly를 줄이고, feature extraction이 안정적으로 수행될 수 있는 segment만 사용하는 것이다.

### 3.6 PPG feature extraction

저자들은 PPG waveform과 그 derivative에서 총 68개의 feature를 추출한다. feature는 크게 time feature, amplitude/intensity feature, area feature, slope feature, derivative feature, width feature, ratio feature 등으로 구성된다.

예를 들어 time feature에는 ascending branch time(ABT), descending branch time(DBT), systolic time(ST), diastolic time(DT), cycle time(CT), beginning point to diastolic peak time(BDT) 등이 포함된다. Area feature에는 single period pulse wave area(SPA), ascending branch area(ABA), descent branch area(DBA), systolic area(SA), diastolic area(DA)가 포함된다. Slope feature에는 ascending branch slope(ASL), maximum ascending branch slope(ASLmax), descending branch slope(DSL), maximum descending branch slope(DSLmax) 등이 포함된다. 또한 first derivative인 VPPG와 second derivative 관련 intensity feature도 포함된다.

각 10초 segment는 pulse cycle 단위로 나뉘고, 각 cycle에서 feature를 추출한 뒤, 유효 feature extraction이 성공한 cycle들의 feature 평균을 해당 segment의 PPG feature로 사용한다. peak detection은 amplitude threshold와 frequency threshold를 결합한다. main peak와 valley는 baseline calibrated PPG signal의 최대·최소값 대비 일정 기준을 만족해야 하며, adjacent peak 또는 trough 간 시간 간격이 0.8–4.0 Hz 범위의 human heart rate에 해당해야 한다.

### 3.7 Differential feature processing

이 논문의 가장 중요한 방법론은 differential feature processing이다. 각 subject의 baseline PPG feature를 기준으로 후속 측정 feature와의 차이를 계산한다.

$$
\Delta f^k_{i,j}=f^k_{i,j+1}-f^k_{i,base}
$$

여기서 $k$는 $k$번째 feature, $i$는 subject, $j$는 후속 측정 index이다. $\Delta f^k_{i,j}$는 $i$번째 subject의 $k$번째 feature가 baseline 대비 얼마나 변했는지를 나타낸다.

DF의 장점은 개인별 나이, 키, 체중, 혈관 상태 등으로 인한 절대 feature 차이를 줄인다는 것이다. 예를 들어 어떤 사람은 원래 pulse wave amplitude가 크고, 다른 사람은 작을 수 있다. 이 절대값 차이를 그대로 모델에 넣으면 개인차가 혈압 추정에 혼입된다. 반면 같은 사람 안에서 amplitude나 time interval이 baseline 대비 어떻게 변했는지를 보면, 혈압 변화와 더 직접적인 관계를 볼 수 있다.

DF 계산 후에는 feature 간 unit과 scale 차이를 제거하기 위해 normalization을 적용한다. 논문은 모델 특성에 따라 제공되는 normalization metric을 사용했다고 설명하지만, 구체적인 normalization 방식은 feature별로 상세히 제시되지는 않는다.

### 3.8 Feature selection

총 68개의 DF 중 혈압 변화 추정에 효과적인 feature subset을 선택하기 위해 combined filter-wrapper feature selection을 수행한다. 이 방법은 두 단계로 구성된다.

첫 번째 단계는 filter-based feature retention이다. Pearson correlation coefficient(PCC)를 사용하여 각 DF와 BP change 사이의 상관성을 계산하고, correlation이 0.1 미만인 feature를 제거한다. 이 결과 SBP change에 대해서는 18개의 DF가 유지되고, DBP change에 대해서는 23개의 DF가 유지된다.

두 번째 단계는 wrapper-based optimal subset search이다. 저자들은 exhaustive feature selection을 사용하여 가능한 모든 feature subset을 탐색하고, Multiple Linear Regression(MLR)을 기반으로 MAE가 가장 낮은 subset을 선택한다. 최소 feature 수는 1개, 최대 feature 수는 filter 단계에서 남은 전체 feature 수로 설정한다. 모든 가능한 조합에 대해 MAE를 계산하고, 가장 낮은 MAE를 내는 feature subset을 최종 선택한다.

최종적으로 SBP change estimation에는 7개의 DF가 선택된다.

$$
BDT,\ SPA,\ ABA,\ DBA,\ DA,\ ASLmax,\ STT
$$

DBP change estimation에는 5개의 DF가 선택된다.

$$
DT,\ SPA,\ SA,\ ASL,\ DSLmax
$$

SBP와 DBP를 합치면 총 11개의 DF가 사용된다. 논문은 exhaustive search가 computational cost는 크지만, 제한된 feature 수와 연구 목적상 가능한 모든 조합을 고려해 global optimal subset을 찾는 데 가치가 있다고 설명한다.

### 3.9 혈압 추정 모델

혈압 추정은 2단계 전략으로 수행된다. 첫 번째 단계에서는 선택된 normalized differential features를 사용하여 baseline 대비 혈압 변화량을 추정한다.

$$
\Delta BP_{i,j}
\sim
{\Delta \tilde{f}^{{i,j}}_1,\Delta \tilde{f}^{{i,j}}_2,\ldots,\Delta \tilde{f}^{{i,j}}_k}
$$

두 번째 단계에서는 추정된 혈압 변화량에 baseline blood pressure를 더해 최종 혈압값을 계산한다.

$$
BP_{i,j}
\sim
\Delta BP_{i,j}+BP_{i,base}
$$

여기서 $BP_{i,base}$는 $i$번째 subject의 baseline 혈압이다. 이 방식은 baseline BP를 calibration information으로 사용하는 calibrated BP estimation이다.

저자들은 네 가지 machine learning model을 비교한다. Multiple Linear Regression(MLR), LASSO, Random Forest(RF), XGBoost가 사용되었다. MLR은 feature coefficient를 통해 해석 가능성이 높고, LASSO는 sparse regularization으로 feature selection 및 overfitting 방지에 유리하다. RF는 여러 decision tree를 결합하여 robustness와 nonlinear relationship modeling에 강점이 있다. XGBoost는 gradient boosting 기반 모델로 regularization과 missing feature handling, nonlinear modeling에 강점이 있다.

하이퍼파라미터는 grid search와 10-fold cross-validation을 결합하여 최적화하였다. 최종 성능은 70% train, 30% test split으로 평가하며, 각 subject의 sample은 train 또는 test 중 하나에만 포함되도록 하여 subject overlap을 방지했다.

## 4. 실험 및 결과

### 4.1 Differential feature와 혈압 변화의 상관성

논문은 먼저 선택된 DF가 실제로 BP change와 얼마나 관련이 있는지 분석한다. Fig. 7과 Table 2에 따르면 선택된 time DFs는 BP change와 0.5 이상의 높은 correlation을 보인다. 또한 최종 선택된 11개 feature 중 area DFs가 5개로 가장 큰 비중을 차지하고, slope DFs도 중요한 역할을 한다.

Table 2에서는 원래 feature와 BP 사이의 correlation, 그리고 differential feature와 BP differential 사이의 correlation을 비교한다. 11개 feature 중 8개에서 DF가 원 feature보다 더 높은 correlation을 보였다. 예를 들어 BDT의 경우 SBP와의 correlation은 0.16이지만, BDT differential과 SBP differential의 correlation은 0.52로 크게 높다. SPA는 SBP와 0.26, SBP_diff와 0.41이며, DBP와 0.43, DBP_diff와 0.52이다. DT는 DBP와 0.32, DBP_diff와 0.56이다.

이 결과는 baseline 대비 feature 변화가 혈압 변화와 더 밀접하게 연결된다는 논문의 핵심 가정을 뒷받침한다.

### 4.2 선택된 feature의 생리적 해석

저자들은 MLR 기반 feature selection 과정에서 얻은 coefficient weight를 사용해 differential feature의 생리적 의미를 해석한다. SBP change estimation에서 ASLmax의 weight는 3.71로 매우 크며, SPA는 0.75, DA는 0.67, DBA는 -0.36이다. DBP change estimation에서는 ASL의 weight가 -3.25, DSLmax가 -2.80, SA가 0.78, SPA가 0.24이다.

논문은 혈압에 영향을 주는 주요 생리 요인으로 cardiac output(CO), peripheral vascular resistance(PVR), arterial wall elasticity(AWE)를 제시한다. Slope DFs는 모델에서 가장 큰 weight를 가지며, BP 변화에 가장 중요한 feature로 해석된다. 심박출량이 증가하면 심장이 밀어내는 혈액량이 증가하고, 말초혈관저항은 혈액이 말초로 빠르게 흐르는 것을 방해한다. 이 두 요인의 결합은 PPG waveform의 상승 기울기와 하강 기울기에 영향을 줄 수 있다.

Area DFs와 time DFs는 secondary feature로 해석된다. 혈압 pulsatility가 증가하면 cardiac output이 증가하고, 이는 pulse wave upstroke를 더 가파르게 만들 수 있다. arterial wall elasticity가 감소하면 ejection time과 waveform shape가 변할 수 있으며, 말초저항 증가와 결합하여 downstroke 및 diastolic area에 영향을 준다. 저자들은 특히 SBP change model에서 diastolic period에서 추출된 DA differential의 weight가 upstroke에서 추출된 ABA differential보다 큰 점을 흥미로운 현상으로 언급한다. 가능한 설명으로는 diastolic feature가 혈압이 감소하는 방식과 관련되거나, downstroke duration이 upstroke보다 길어 더 많은 정보를 포함할 수 있다는 점을 제시한다. 다만 이는 가설적 해석이며, 논문에서 직접 생리 실험으로 검증한 것은 아니다.

### 4.3 Machine learning model 비교

Table 4는 네 가지 model의 SBP와 DBP estimation 성능을 비교한다. 평가 지표는 RMSE, STD, PCC이다. 여기서 STD는 error의 standard deviation이고, PCC는 예측값과 실제값 간 Pearson correlation coefficient이다.

MLR은 SBP RMSE 9.37, STD 9.35, PCC 0.78을 기록했고, DBP RMSE 6.80, STD 6.78, PCC 0.75를 기록했다. LASSO는 SBP RMSE 9.54, DBP RMSE 6.91로 MLR보다 약간 낮은 성능을 보였다. XGBoost는 SBP RMSE 8.63, DBP RMSE 7.23이며, DBP에서는 성능이 상대적으로 좋지 않았다.

가장 좋은 성능은 Random Forest에서 나왔다. RF는 SBP RMSE 7.15, STD 7.12, PCC 0.90을 달성했고, DBP RMSE 5.30, STD 5.27, PCC 0.86을 달성했다. 논문 abstract에서는 “STD of the error of 7.15 mmHg for SBP and 5.30 mmHg for DBP”라고 표현하지만, Table 4에서는 이 값들이 RMSE이고 STD는 각각 7.12와 5.27로 제시되어 있다. 따라서 본 보고서에서는 Table 4 기준으로 RF의 RMSE는 7.15/5.30, STD는 7.12/5.27로 정리한다.

RF가 가장 좋은 이유는 DF와 BP change 사이의 관계가 완전한 선형이 아닐 가능성이 높기 때문이다. RF는 decision tree ensemble을 통해 nonlinear interaction을 포착하면서도 overfitting에 비교적 강하다. 반면 MLR과 LASSO는 해석 가능성은 높지만 복잡한 nonlinear 관계를 충분히 반영하기 어렵다. XGBoost는 강력한 모델이지만, 데이터 규모나 하이퍼파라미터 조건에서 RF보다 안정적이지 않았을 수 있다.

### 4.4 Regression plot과 Bland–Altman 분석

Fig. 10은 estimated BP와 true BP의 regression plot, 그리고 estimated BP differential과 true BP differential의 regression plot을 제시한다. 최종 BP estimation의 PCC는 SBP 0.90, DBP 0.86으로 높다. 반면 BP differential estimation만 놓고 보면 PCC는 SBP 0.78, DBP 0.77로 낮다.

이 차이는 baseline BP를 더하는 calibration step 때문이다. baseline BP는 개인의 안정적인 기준값 역할을 하며, 추정된 변화량의 bias를 보정한다. 따라서 $\widehat{\Delta BP}$ 자체의 correlation보다 최종 $\widehat{BP}$의 correlation이 더 높게 나타난다. 이는 baseline calibration이 모델 성능에 중요한 역할을 한다는 점을 보여준다.

Fig. 11의 Bland–Altman plot은 estimated BP와 true BP 간 agreement를 시각화한다. SBP의 upper/lower limit은 $0.31 \pm 1.96 \times 7.12$이고, DBP는 $-0.43 \pm 1.96 \times 5.27$이다. 차이값의 95.63%와 95.32%가 이 범위 안에 포함된다고 보고한다. 이는 예측값과 실제값 사이에 비교적 일관된 agreement가 있음을 의미한다. 다만 Bland–Altman plot은 평균적인 agreement를 보여줄 뿐, 특정 고혈압 환자나 질환군에서의 성능을 보장하지는 않는다.

### 4.5 Differential feature와 feature selection의 효과

Table 5는 feature selection(FS)과 differential feature(DF)의 효과를 비교한다. 네 가지 설정을 비교한다.

첫째, FS와 DF를 모두 사용하는 경우가 최상의 성능을 보인다. SBP RMSE는 7.15, STD는 7.12, PCC는 0.90이고, DBP RMSE는 5.30, STD는 5.27, PCC는 0.86이다.

둘째, FS는 사용하지만 DF를 사용하지 않고 원 feature를 사용하는 경우 SBP RMSE는 10.90, DBP RMSE는 7.38이다. 셋째, DF는 사용하지만 FS를 사용하지 않는 경우 SBP RMSE는 8.52, DBP RMSE는 6.67이다. 넷째, FS와 DF를 모두 사용하지 않는 경우가 가장 낮은 성능을 보이며, SBP RMSE 12.82, DBP RMSE 8.74, PCC도 각각 0.44, 0.43에 그친다.

이 결과는 두 가지를 명확히 보여준다. 먼저 DF 자체가 성능 향상에 큰 영향을 준다. FS 없이도 DF를 사용하면 SBP RMSE가 12.82에서 8.52로 크게 감소한다. 둘째, feature selection도 추가적인 개선을 제공한다. DF를 사용한 상태에서 FS를 적용하면 SBP RMSE가 8.52에서 7.15로, DBP RMSE가 6.67에서 5.30으로 감소한다.

논문은 forward feature selection과도 비교한다. Forward selection에서는 SBP에 대해 상위 10개 feature를 선택했을 때 MAE 7.5 mmHg, DBP에 대해 22개 feature를 선택했을 때 MAE 5.8 mmHg가 최적이었다. 반면 exhaustive feature selection은 SBP와 DBP에서 더 좋은 subset을 찾아냈다. 하지만 exhaustive search는 computational cost가 크므로 feature 수가 더 많아지는 경우 확장성이 제한될 수 있다.

### 4.6 기존 연구와 비교

Table 6은 기존 연구와 제안 방법을 ME와 STD 기준으로 비교한다. Haddad et al.은 MLR과 calibration을 사용해 SBP ME 1.04, STD 7.65, DBP ME 0.81, STD 5.68을 기록했다. Aguet et al.은 LASSO를 사용해 SBP STD 10.77, DBP STD 7.62를 기록했다. Natarajan et al.은 MLR 기반 feature change approach를 사용해 SBP ME 2.30, STD 9.30, DBP ME 0.50, STD 7.00을 보고했다. Finnegan et al.은 RF를 사용해 SBP ME 0.41, STD 7.21, DBP ME 0.74, STD 6.32를 기록했다.

이 논문의 RF 기반 방법은 SBP ME 0.31, STD 7.12, DBP ME -0.43, STD 5.27로 가장 낮은 또는 매우 경쟁력 있는 STD를 보인다. 특히 DBP에서 STD 5.27은 비교 대상보다 낮다. 저자들은 이러한 결과를 PPG differential feature가 individual physiological factor의 영향을 줄이고, exhaustive feature selection이 최적 feature subset을 효과적으로 찾았기 때문으로 해석한다.

다만 비교 연구들은 데이터셋, 측정 조건, 참가자 특성, calibration 방식, feature 종류가 서로 다르므로 직접적인 공정 비교에는 한계가 있다. 특히 이 논문은 건강한 참가자 90명을 대상으로 한 자체 수집 데이터이므로, 환자군이나 공개 ICU dataset에서의 성능과 직접 비교하기는 어렵다.

## 5. 강점, 한계

### 5.1 강점

이 논문의 가장 큰 강점은 baseline 대비 differential feature라는 명확하고 해석 가능한 관점을 제시했다는 점이다. PPG 기반 혈압 추정에서 개인차는 매우 큰 문제인데, 이 논문은 각 subject의 첫 측정값을 기준으로 feature 변화량과 혈압 변화량을 연결함으로써 개인 간 생리적 차이를 줄이려 한다. 이는 calibration-based long-term monitoring 시나리오에 적합한 접근이다.

두 번째 강점은 feature engineering 과정이 체계적이라는 점이다. 저자들은 총 68개의 PPG feature를 추출하고, PCC 기반 filtering과 exhaustive wrapper search를 결합하여 SBP 및 DBP 변화 추정에 가장 적합한 feature subset을 선택하였다. 이 과정은 feature effectiveness와 redundancy를 동시에 고려한다.

세 번째 강점은 생리적 해석을 시도했다는 점이다. 단순히 machine learning 모델의 성능을 제시하는 데 그치지 않고, 선택된 DFs가 cardiac output, peripheral vascular resistance, arterial wall elasticity와 어떻게 관련될 수 있는지 설명한다. 특히 slope, area, time feature가 혈압 변화와 어떤 관계를 가질 수 있는지 논의한 점은 PPG feature 기반 연구에서 중요한 장점이다.

네 번째 강점은 subject-level split을 사용했다는 점이다. 논문은 각 subject의 sample이 train 또는 test 중 하나에만 포함되도록 하여 train-test leakage를 줄였다. 이는 같은 subject의 데이터가 train과 test에 동시에 들어가 성능이 과대평가되는 문제를 방지하는 데 중요하다.

다섯 번째 강점은 자체 장기 추적 데이터를 수집했다는 점이다. 3개월 동안 90명의 PPG와 BP를 반복 측정했기 때문에 baseline 대비 혈압 변화량을 분석할 수 있었다. 대부분의 공개 dataset은 단일 시점 또는 병원 환경의 waveform을 포함하는 경우가 많아 개인 내 장기 변화 분석에 제한이 있다.

### 5.2 한계

가장 큰 한계는 데이터셋이 건강한 참가자만으로 구성되어 있다는 점이다. 저자들은 hypertension이나 cardiovascular disease가 PPG waveform에 미치는 영향을 줄이기 위해 healthy participants만 포함했다고 설명한다. 하지만 혈압 추정 시스템의 실제 주요 대상은 고혈압, 저혈압, 심혈관 질환자, 고령자 등이다. 건강한 참가자에서 좋은 성능이 환자군에서도 유지된다고 보장할 수 없다.

두 번째 한계는 baseline calibration이 필수적이라는 점이다. 제안 방법은 각 subject의 첫 cuff BP와 baseline PPG feature를 알아야 작동한다. 따라서 완전한 calibration-free 방법은 아니다. 실제 장기 모니터링에서는 baseline cuff measurement를 주기적으로 갱신해야 할 가능성이 있으며, baseline이 시간이 지나도 안정적인 기준으로 남는지 검증이 필요하다.

세 번째 한계는 reference BP가 cuff-type electronic monitor로 측정되었다는 점이다. cuff BP는 임상적으로 널리 사용되지만, beat-to-beat ABP waveform과 달리 측정 오차와 시간 지연이 존재한다. 또한 PPG는 cuff BP 측정 후 1분 뒤에 기록되므로, BP와 PPG가 정확히 동시 측정된 것은 아니다. 혈압은 짧은 시간에도 변동할 수 있기 때문에 이 시간 차이가 label noise를 유발할 수 있다.

네 번째 한계는 exhaustive feature selection의 계산 비용이다. 현재는 filter 단계 후 남은 18개 또는 23개 feature에 대해 exhaustive search를 수행했지만, feature 수가 더 많아지거나 multi-channel signal을 포함할 경우 가능한 조합 수가 급격히 증가한다. 따라서 더 큰 feature space에서는 실용성이 떨어질 수 있다.

다섯 번째 한계는 생리적 해석이 일부 가설적이라는 점이다. 논문은 coefficient weight를 기반으로 CO, PVR, AWE와의 관계를 설명하지만, 실제로 CO, PVR, AWE를 동시에 측정하여 검증한 것은 아니다. 따라서 feature와 생리 요인 간 관계는 문헌과 waveform physiology에 기반한 합리적 해석이지만, 직접 검증된 causal mechanism은 아니다.

여섯 번째 한계는 deep learning 기반 end-to-end 방법과의 직접 비교가 제한적이라는 점이다. 논문은 MLR, LASSO, RF, XGBoost를 비교하고 기존 연구와 table comparison을 제공하지만, 동일 데이터셋에서 CNN, LSTM, Transformer 등 현대 deep learning 모델과 비교하지는 않는다. 따라서 DF 기반 RF가 동일 조건의 deep learning 모델보다 우수한지는 알 수 없다.

일곱 번째 한계는 데이터 공개가 되지 않는다는 점이다. Data availability에서 사용 데이터가 confidential하다고 명시되어 있다. 이는 재현성과 외부 검증 측면에서 한계가 된다.

### 5.3 비판적 해석

이 논문은 최근 PPG-BP 연구에서 흔히 나타나는 deep learning end-to-end 경쟁과 다른 방향을 취한다. 모델 복잡도를 높이기보다, 개인 baseline 대비 feature 변화량이라는 feature representation을 개선함으로써 성능과 해석 가능성을 동시에 얻으려 한다. 이는 실용적이고 설득력 있는 접근이다.

특히 “혈압은 사람마다 다르므로 PPG feature의 절대값보다 개인 내 변화가 중요하다”는 관점은 장기 모니터링에 매우 적합하다. 웨어러블 기기가 처음 cuff BP calibration을 수행한 뒤, 이후 PPG 변화량을 통해 BP 변화를 추적하는 시나리오를 생각하면 이 논문의 방법론은 현실적인 활용 가능성이 있다.

그러나 이 방법은 calibration-dependent하다. 즉, baseline BP가 없으면 최종 BP를 추정할 수 없다. 또한 baseline이 오래되면 개인의 생리 상태 변화, 체중 변화, 약물 복용, 질병 진행 등으로 인해 baseline relationship이 변할 수 있다. 논문은 3개월 추적 데이터를 사용했지만, baseline update 주기나 calibration drift 문제를 충분히 다루지는 않는다.

또한 건강한 사람만 대상으로 했기 때문에 혈압 범위와 PPG waveform variability가 실제 임상 환경보다 제한적일 수 있다. 고혈압 환자, 당뇨 환자, 혈관 경직이 높은 고령자, 부정맥 환자 등에서 PPG morphology와 BP 변화의 관계는 더 복잡해질 수 있다. 따라서 이 논문은 “DF가 혈압 변화 추정에 유용하다”는 근거를 제공하지만, 범용 혈압 추정 알고리즘으로 보기에는 추가 검증이 필요하다.

## 6. 결론

이 논문은 단일 채널 PPG 기반 혈압 추정에서 differential feature를 활용하는 방법을 제안한다. 개인별 첫 측정값을 baseline으로 설정하고, 이후 PPG feature 변화량과 혈압 변화량을 연결함으로써 개인 생리적 차이를 줄이고 장기 혈압 모니터링의 정확도를 높이고자 했다.

방법론적으로는 1분 PPG 신호에서 filtering, normalization, SNR 및 peak variance 기반 segment selection을 수행한 뒤, 총 68개의 PPG feature를 추출한다. 이후 baseline 대비 differential feature를 계산하고, PCC filter와 exhaustive wrapper search를 결합하여 SBP에는 7개, DBP에는 5개의 DF를 선택한다. 선택된 DF를 입력으로 MLR, LASSO, RF, XGBoost를 비교했으며, Random Forest가 가장 좋은 성능을 보였다.

주요 결과는 RF 기준 SBP RMSE 7.15 mmHg, STD 7.12 mmHg, PCC 0.90, DBP RMSE 5.30 mmHg, STD 5.27 mmHg, PCC 0.86이다. FS와 DF를 모두 사용한 경우가 가장 우수했고, feature를 직접 사용하는 방식보다 DF 기반 방식이 큰 성능 향상을 보였다. 논문은 PPG DFs가 BP changes와 더 높은 상관성을 갖는다는 결과를 통해 differential feature의 유효성을 입증한다.

이 연구의 주요 기여는 PPG feature의 절대값이 아니라 baseline 대비 변화량을 혈압 변화 추정에 활용했다는 점, 그리고 선택된 feature의 생리적 의미를 해석하려 했다는 점이다. 특히 단일 채널 PPG만으로 calibration-based long-term BP monitoring을 수행할 수 있는 가능성을 보여준다.

향후 연구에서는 고혈압 및 심혈관 질환자를 포함한 더 다양한 population에서 검증해야 하며, baseline drift와 recalibration 문제를 다루어야 한다. 또한 동일 데이터셋에서 deep learning 기반 모델과 비교하고, multi-channel PPG 또는 ECG, demographic information과 결합했을 때의 성능 변화도 평가할 필요가 있다. 그럼에도 이 논문은 PPG 기반 혈압 추정에서 개인 내 feature 변화라는 해석 가능한 방향을 제시했다는 점에서 의미 있는 연구로 평가할 수 있다.
