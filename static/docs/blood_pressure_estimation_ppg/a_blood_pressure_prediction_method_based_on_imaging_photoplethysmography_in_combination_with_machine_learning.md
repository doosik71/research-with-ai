# A Blood Pressure Prediction Method Based on Imaging Photoplethysmography in combination with Machine Learning

* **저자**: Meng Rong, Kaiyang Li
* **발표연도**: 2021

## 1. 논문 개요

이 논문은 얼굴 영상을 이용해 비접촉 방식으로 혈압을 추정하는 non-contact blood pressure, 즉 NCBP 시스템을 제안한다. 기존의 cuff 기반 혈압계는 비교적 정확하지만 팔을 압박해야 하므로 불편하고, 연속적인 혈압 모니터링에 적합하지 않다. 또한 일반적인 PPG 기반 비침습 혈압 추정 방법은 손가락, 손목, 귀 등 신체에 센서를 부착해야 하며, 일부 방법은 ECG와 PPG를 동시에 측정해야 하므로 장치 구성과 동기화가 복잡하다. 이 논문은 이러한 문제를 줄이기 위해 일반 웹캠으로 촬영한 얼굴 영상에서 imaging photoplethysmography, 즉 IPPG 신호를 추출하고, 이 신호에서 혈압 관련 feature를 계산한 뒤 machine learning 회귀 모델로 SBP와 DBP를 예측한다.

연구 문제는 “얼굴 영상에서 얻은 IPPG 신호만으로 cuff를 사용하지 않고 혈압을 어느 정도 정확하게 추정할 수 있는가”이다. 저자들은 ambient light, 즉 별도의 보조 광원이 없는 일반 조명 환경에서 Logitech C922 PRO 웹캠을 사용해 얼굴 영상을 촬영하고, 이를 기반으로 혈압 추정 모델을 학습 및 평가했다. 이 연구의 의의는 혈압 측정 장치를 단순화하고, 사용자가 센서를 착용하거나 팔을 압박하지 않아도 되는 방식으로 혈압 모니터링 가능성을 탐색했다는 점에 있다.

논문은 총 191명의 volunteer 데이터를 수집했고, screening 이후 189개의 유효 사례를 분석에 사용했다. 이 중 고혈압 17명, 저혈압 10명이 포함되어 있어 정상 혈압뿐 아니라 일부 abnormal blood pressure 사례도 포함되었다. 다만 abnormal case의 수가 적기 때문에 저자들은 고혈압·저혈압 구간에서의 일반화 성능이 충분하지 않을 수 있음을 한계로 명시했다.

## 2. 핵심 아이디어

논문의 핵심 아이디어는 얼굴 영상의 RGB channel 중 green channel에서 혈류 변화에 따른 미세한 밝기 변화를 추출하면, 이 변화가 PPG와 유사한 pulse wave 정보를 포함한다는 점이다. 혈액 속 hemoglobin은 특정 파장대에서 빛을 흡수하며, 논문은 500–600 nm 부근의 green channel이 혈액량 변화에 가장 민감하다고 설명한다. 따라서 얼굴 영상의 ROI에서 green channel 평균값을 시간에 따라 계산하면 IPPG 신호를 얻을 수 있고, 이 신호의 waveform morphology를 혈압 추정에 활용할 수 있다.

기존 NCBP 연구 중 일부는 얼굴과 손 등 두 부위를 동시에 촬영하여 pulse transit time, 즉 PTT를 계산하거나, 고가 장비와 외부 광원을 사용했다. 이 논문은 단일 웹캠과 ambient light만으로 혈압을 추정하려 한다는 점에서 시스템 구성을 단순화한다. 또한 저자들은 단순히 기존 PPG feature를 적용하는 데 그치지 않고, IPPG 신호 품질을 높이기 위해 두 가지 절차를 제안한다. 첫째는 colormap 기반 ROI 선택 방법이다. 얼굴 영역의 light intensity 분포를 관찰하여 forehead와 cheek 부근에서 강한 신호가 나타남을 확인했지만, 실제 측정에서는 머리카락이 이마를 가릴 수 있으므로 cheek과 nose 영역을 ROI로 선택했다. 둘째는 robust peak extraction algorithm이다. thresholding과 difference operation을 이용해 noisy IPPG signal에서도 peak와 valley를 안정적으로 찾는 절차를 제안했다.

혈압 예측 모델 측면에서는 26개의 original feature를 추출한 뒤, Pearson correlation coefficient 기반 feature selection으로 collinear feature를 제거하고 16개 feature를 최종 사용했다. 이후 multiple linear regression, support vector regression, random forest, multi-layer perceptron을 비교했고, 실험 결과 SVR이 가장 좋은 성능을 보였다. 이는 표본 수가 크지 않고 feature와 혈압 사이의 관계가 비선형적일 가능성이 높은 문제에서 SVR이 적합했기 때문이라고 해석할 수 있다.

## 3. 상세 방법 설명

논문에서 제안한 전체 pipeline은 video acquisition, IPPG signal extraction, signal filtering, peak and valley extraction, feature extraction, feature selection, machine learning regression으로 구성된다. Fig. 3의 system block diagram은 camera에서 영상을 수집한 뒤 IPPG signal을 추출하고, signal processing을 거쳐 feature extraction과 feature selection을 수행한 후 machine learning module에서 BP를 예측하는 흐름을 보여준다.

### 3.1 데이터 수집 및 실험 장치

실험에는 Logitech C922 PRO webcam이 사용되었다. 영상은 30 FPS, 해상도 1280×720으로 촬영되었고, 각 volunteer는 촬영 전 5분간 휴식하여 생리적 상태를 안정화했다. 이후 약 40초 동안 얼굴 영상을 촬영했다. volunteer는 IPPG 신호의 안정성을 위해 머리를 움직이지 않도록 안내받았다. 혈압 reference는 OMRON HEM-1020 sphygmomanometer로 동시에 측정했다. 논문 Fig. 1은 웹캠, 혈압계, 컴퓨터로 구성된 실험 장치를 보여주며, 전체 장치가 비교적 단순하다는 점을 강조한다.

총 191명의 volunteer가 수집되었고, screening 이후 189개의 유효 사례가 사용되었다. 성별 구성은 남성 141명, 여성 50명이고, 연령 범위는 20–61세이다. 혈압 분포는 Fig. 2에 제시되어 있으며, SBP는 92–148 mmHg, DBP는 51–102 mmHg 범위를 갖는다. 이 분포는 정상 혈압을 중심으로 하지만, 일부 hypertension 및 hypotension 사례를 포함한다.

### 3.2 IPPG 신호 추출

IPPG는 Lambert-Beer law와 light scattering theory에 기반한다. 논문은 단색광이 흡수 물질을 통과할 때 입사광과 투과광 사이의 관계를 다음과 같이 설명한다.

$$
I = I_0 e^{-\varepsilon(\lambda)CL}
$$

여기서 $I$는 transmitted light intensity, $I_0$는 emitted light intensity, $\varepsilon(\lambda)$는 wavelength $\lambda$에서의 absorption coefficient, $C$는 medium concentration, $L$은 light travel distance이다. 얼굴 피부에 빛이 조사되면 혈관 내 blood volume이 심장 박동에 따라 주기적으로 변하고, 이로 인해 반사광의 intensity도 주기적으로 변한다. 이 반사광의 시간 변화에 physiological information이 포함된다는 것이 IPPG의 기본 가정이다.

각 frame에서 ROI 영역의 pixel value를 평균하여 IPPG signal을 계산한다.

$$
S_{ippg}(t)=\frac{\sum_{i=1}^{M}\sum_{j=1}^{N}s_{ippg}(i,j,t)}{M \times N}
$$

여기서 $t$는 frame sequence, $M$과 $N$은 ROI의 height와 width이다. $s_{ippg}(i,j,t)$는 $t$번째 frame에서 ROI 내부 pixel 위치 $(i,j)$의 intensity를 의미한다. 얼굴 영상은 RGB channel로 분해되고, red, green, blue 각각에서 시간 신호를 만들 수 있다. 저자들은 hemoglobin의 optical absorption 특성과 green channel의 pulse-related variation이 가장 뚜렷하다는 이유로 green channel signal을 최종 IPPG signal로 선택했다.

### 3.3 ROI 선택

일반적으로 얼굴 전체를 ROI로 사용할 수 있지만, 이 논문은 얼굴 전체가 항상 최적은 아니라고 본다. 눈 깜박임, 코 찡그림, 머리 움직임 같은 motion artifact와 얼굴 부위별 light intensity 차이, 머리카락 등 appearance factor가 IPPG 품질에 영향을 주기 때문이다.

저자들은 6명의 얼굴 colormap을 관찰하여 얼굴 부위별 light intensity distribution을 분석했다. Fig. 5의 colormap에서 forehead와 cheek 영역은 붉은색·흰색 계열로 나타나 비교적 강한 light intensity를 보인다. 그러나 여성 volunteer의 경우 머리카락이 forehead를 가릴 수 있으므로, 논문은 forehead 대신 cheek과 nose area를 ROI로 선택한다. Fig. 4에서 red rectangle로 표시된 cheek-nose 영역이 최종 ROI이다. 이 선택은 신호 강도와 실사용 가능성을 함께 고려한 practical design으로 볼 수 있다.

### 3.4 IPPG 신호 필터링

비접촉 얼굴 영상 기반 IPPG에는 여러 noise가 포함된다. 예를 들어 미세한 head shaking, fluorescent lamp에 의한 power frequency noise, baseline drift 등이 있다. 논문은 wavelet transform과 Butterworth band-pass filtering을 조합하여 IPPG signal을 정제한다.

먼저 baseline drift 제거를 위해 discrete wavelet transform이 사용된다. 논문은 1차원 IPPG signal $S_{ippg}(t)$의 wavelet transform을 다음과 같이 나타낸다.

$$
F_{\psi}^{s}(x,y)=x_0^{-\frac{j}{2}}\int_{-\infty}^{+\infty}S_{ippg}(t)\psi(x_0^{-j}t-y_0k)dt,\quad j,k \in Z
$$

실제 적용에서는 $x_0=2$, $y_0=1$로 두어 binary wavelet 형태를 사용한다.

$$
\psi_{j,k}(t)=2^{-\frac{j}{2}}\psi(2^{-j}t-k),\quad j,k \in Z
$$

저자들은 Sym6 mother wavelet을 사용해 IPPG signal을 5-layer로 분해하고, 5번째 layer의 low-frequency component를 baseline drift signal로 간주하여 원 신호에서 제거했다. 이후 heart rate frequency range가 0.7–4 Hz임을 고려해 0.7–4 Hz Butterworth band-pass filter를 적용했다. Fig. 6은 original IPPG signal, baseline drift 제거 후 signal, band-pass filtering 후 signal을 순서대로 보여주며, 필터링 후 waveform이 더 주기적이고 매끄럽게 정리되는 것을 시각적으로 확인할 수 있다.

### 3.5 Peak 및 valley 검출 알고리즘

IPPG waveform에서 한 주기를 구분하려면 peak와 valley를 정확히 찾아야 한다. 논문은 noisy waveform에서도 peak와 valley를 찾기 위한 threshold-based algorithm을 제안한다. 절차는 다음과 같다.

먼저 원 IPPG signal $S_{ippg}^{I}(t)$를 정규화한 뒤, signal midline 근처의 threshold $\tau$를 설정한다. threshold 이상이면 1, 미만이면 0으로 변환하여 binary signal $S_{ippg}^{II}(t)$를 만든다.

$$
S_{ippg}^{II}(t)=
\begin{cases}
0, & S_{ippg}^{I}(t)<\tau \
1, & S_{ippg}^{I}(t)\ge \tau
\end{cases}
,\quad t \in (0,1,\cdots,N_{frame})
$$

이후 point-by-point difference를 수행해 다음 signal을 만든다.

$$
S_{ippg}^{III}(t)=S_{ippg}^{II}(t+1)-S_{ippg}^{II}(t)
$$

즉, binary signal이 0에서 1로 바뀌거나 1에서 0으로 바뀌는 지점을 찾아 waveform 구간을 나눈다. 논문에서는 $S_{ippg}^{III}(t)=-1$에 해당하는 abscissa set을 $T=[t_1,t_2,\cdots,t_n]$으로 두고, 인접한 boundary 사이의 최대값을 peak로 정의한다.

$$
S_{ippg}^{MAX}(i)=\max(S_{ippg}(T(i)):S_{ippg}(T(i+1))),\quad i \in (0,N_T-1)
$$

valley는 유사하게 $S_{ippg}^{III}(t)=1$을 기준으로 구할 수 있다. Fig. 7은 이 알고리즘이 IPPG waveform에서 peak point를 찾는 과정을 보여준다. 논문은 이 방법이 waveform에 noise가 많아도 peak와 valley를 비교적 안정적으로 찾는다고 주장한다.

### 3.6 Feature extraction

저자들은 IPPG signal에서 총 26개의 original feature를 추출했다. feature는 time-domain feature, Kaiser-Teager energy, K value, energy profile, heart rate로 구성된다.

Time-domain feature는 총 22개이며, wave height, time, area, slope, rising branch width, descending branch width 관련 값으로 구성된다. Table 1은 $F1$부터 $F26$까지 feature의 의미를 정리한다. 예를 들어 $F1$은 waveform height, $F4$와 $F5$는 rising 및 falling branch time, $F9$와 $F10$은 rising 및 falling branch area, $F15$와 $F16$은 waveform slope를 의미한다. $F17$–$F22$는 peak height의 25%, 50%, 75%에서 rising branch와 falling branch의 width를 나타낸다.

Kaiser-Teager energy, 즉 KTE는 waveform의 instantaneous energy를 추적하기 위한 feature이다. 논문은 KTE를 다음과 같이 계산한다.

$$
S_{ippg}^{KTE}(t)=\left(S_{ippg}^{I}\right)^2(t)-S_{ippg}^{I}(t-1)\times S_{ippg}^{I}(t+1)
$$

K value는 pulse waveform characteristic quantity로, arterial blood pressure와 관련성이 있다고 알려진 feature이다. 논문은 이를 다음과 같이 계산한다.

$$
S_{ippg}^{K}=\frac{S_{ippg}^{oo'}-S_{ippg}^{I}(t_o)}{S_{ippg}^{I}(t_o)-S_{ippg}^{I}(t_{o'})}
$$

여기서 $o$와 $o'$는 각 IPPG period의 start point와 end point이며, $S_{ippg}^{oo'}$는 해당 period의 평균값이다.

$$
S_{ippg}^{oo'}=\frac{1}{t_{o'}-t_o}\sum_{t=t_o}^{t_{o'}}S_{ippg}^{I}(t)
$$

Energy profile, 즉 EN은 각 cardiac cycle의 periodic energy를 나타낸다.

$$
\log S_{ippg}^{EN}=\log\left(\sum_{t=t_o}^{t_{o'}}\left(S_{ippg}^{I}\right)^2(t)\right)
$$

마지막으로 heart rate, 즉 HR도 feature로 포함된다. HR은 인체의 중요한 physiological parameter이며, 여러 문헌에서 혈압과 관련성이 보고되었기 때문이다.

### 3.7 Feature selection

원래 26개의 feature에는 collinear feature나 redundant feature가 포함될 수 있다. feature 수가 과도하면 limited sample condition에서 sample space가 sparse해지고 model generalization이 떨어질 수 있다. 논문은 Pearson correlation coefficient를 사용해 feature 간 correlation을 계산하고, heat map으로 이를 시각화했다. Fig. 9의 correlation heat map에서 descending point width 관련 feature들이 높은 상관성을 보였으며, 저자들은 strong correlation threshold를 0.7로 설정했다.

그 결과 $F6$, $F9$, $F10$, $F11$, $F12$, $F18$, $F19$, $F20$, $F21$, $F22$가 제거되었고, 최종적으로 다음 16개 feature가 machine learning training에 사용되었다.

$F1$, $F2$, $F3$, $F4$, $F5$, $F7$, $F8$, $F13$, $F14$, $F15$, $F16$, $F17$, $F23$, $F24$, $F25$, $F26$

이 feature selection은 모델 복잡도를 줄이고 collinearity로 인한 성능 저하를 완화하기 위한 절차이다.

### 3.8 Machine learning regression models

논문은 네 가지 회귀 모델을 비교한다. 첫째, multiple linear regression, 즉 MLR은 feature와 혈압 사이의 선형 관계를 모델링한다. 모델은 다음과 같은 형태로 정의된다.

$$
Y^m=\alpha_0^m+\sum_{i=1}^{n}\alpha_i^m X_F(i),\quad m={SBP,DBP}
$$

여기서 $X_F$는 feature selection 이후의 feature vector이고, $n$은 feature 수이다. MLR은 해석이 쉽지만 비선형 관계를 충분히 표현하기 어렵다.

둘째, support vector regression, 즉 SVR은 small sample과 nonlinear regression 문제에 강점이 있는 supervised learning 모델이다. 논문은 Gaussian kernel을 사용했고, grid search로 hyperparameter $C$와 $\gamma$를 최적화했다. SBP에서는 $C=1$, $\gamma=30$이 최적이었고, DBP에서는 $C=0.5$, $\gamma=20$이 최적이었다.

셋째, random forest, 즉 RF는 decision tree ensemble 기반 회귀 모델이다. 논문은 grid search를 통해 tree 수와 max features를 최적화했다. SBP에서는 $n_estimators=22$, $max_features=5$, DBP에서는 $n_estimators=16$, $max_features=5$가 사용되었다.

넷째, multi-layer perceptron, 즉 MLP는 nonlinear mapping 능력을 가진 neural network이다. 논문은 one hidden layer와 output layer를 갖는 비교적 단순한 구조를 사용했으며, activation function으로 ReLU를 선택했다. Table 2에는 MLP의 hidden layer size가 $(12,1)$로 제시되어 있으나, 논문 표기상 solver가 “adma”로 기재되어 있어 Adam의 오탈자인지 확정할 수 없다.

데이터는 70% training set, 30% testing set으로 random split되었다. 각 모델은 300 iterations로 학습되었고, 모델 일반화 성능을 높이기 위해 반복 예측값의 평균을 최종 결과로 사용했다. 구현은 Scikit-learn library 기반으로 이루어졌다.

## 4. 실험 및 결과

### 4.1 모델 비교 결과

모델 평가는 standard deviation, 즉 STD, mean absolute error, 즉 MAE, mean bias, 즉 ME를 사용했다. 모든 결과는 30% test set, 총 56개 사례를 기준으로 계산되었다.

Table 3에 따르면 SVR이 네 모델 중 가장 우수했다. SBP 예측에서 MLR은 STD 7.92 mmHg, MAE 12.05 mmHg, mean bias 6.04 mmHg를 보이며 가장 낮은 성능을 보였다. RF와 MLP는 중간 수준의 성능을 보였고, SVR은 STD 3.35 mmHg, MAE 9.97 mmHg, mean bias 2.10 mmHg로 가장 좋은 결과를 냈다. DBP에서도 SVR이 가장 좋았으며, STD 2.58 mmHg, MAE 7.59 mmHg, mean bias 0.79 mmHg를 기록했다.

여기서 주의할 점은 abstract와 result 서술 일부에서 “STD and MAE”의 순서 또는 “MAE, STD”의 순서가 혼재되어 보인다는 점이다. Table 3의 column heading을 기준으로 해석하면 SVR의 SBP 결과는 STD 3.35 mmHg, MAE 9.97 mmHg이고, DBP 결과는 STD 2.58 mmHg, MAE 7.59 mmHg이다.

SVR이 좋은 성능을 보인 이유는 IPPG waveform feature와 혈압 사이의 관계가 단순 선형이 아니라 복잡한 nonlinear relationship일 가능성이 크기 때문이다. SVR은 kernel function을 통해 high-dimensional feature space에서 nonlinear regression을 수행할 수 있고, sample 수가 제한적인 조건에서도 비교적 안정적으로 작동한다.

### 4.2 BHS 및 AAMI 기준 평가

논문은 SVR 결과를 BHS와 AAMI 기준으로 평가했다. BHS 기준은 예측 혈압과 reference 혈압의 absolute difference가 5, 10, 15 mmHg 이하인 비율에 따라 grade를 부여한다.

Table 4에 따르면 SBP의 경우 error가 5 mmHg 이하인 비율은 48.2%, 10 mmHg 이하인 비율은 78.6%, 15 mmHg 이하인 비율은 94.6%였다. DBP의 경우 각각 55.4%, 85.7%, 98.2%였다. 저자들은 이 결과가 BHS 기준의 허용 범위에 들어간다고 해석했다. 구체적으로 SBP는 5 mmHg 기준에서는 Grade C, 10 mmHg와 15 mmHg 기준에서는 Grade B 수준으로 볼 수 있고, DBP는 5 mmHg 기준에서는 Grade B, 10 mmHg와 15 mmHg 기준에서는 Grade A 수준으로 볼 수 있다.

AAMI 기준에서는 ME가 5 mmHg 미만이고 STD가 8 mmHg 미만이어야 한다. Table 5에 따르면 SVR의 SBP ME는 2.10 mmHg, STD는 3.35 mmHg이며, DBP ME는 0.79 mmHg, STD는 2.58 mmHg이다. 따라서 논문은 SBP와 DBP 모두 AAMI 기준을 만족한다고 주장한다.

다만 여기서 비판적으로 보아야 할 점은 AAMI와 BHS standard는 단순히 error threshold만이 아니라 sample composition, validation protocol, subject 수, reference method 등과 관련된 엄격한 요구사항을 포함할 수 있다는 점이다. 이 논문은 test set 56 cases를 기준으로 비교했으므로, device validation standard를 완전한 임상 검증으로 만족했다고 보기보다는, error metric 측면에서 해당 기준과 부합한다고 해석하는 것이 더 적절하다.

### 4.3 Regression scatter plot 및 Bland-Altman 분석

Fig. 10은 predicted BP와 reference BP 사이의 regression scatter plot을 보여준다. SBP의 correlation coefficient는 0.70, DBP의 correlation coefficient는 0.76으로 보고되었다. 이는 IPPG 기반 feature와 reference BP 사이에 유의미한 상관이 있음을 보여준다. 다만 correlation이 1에 가까운 수준은 아니므로, 실제 혈압값을 정밀하게 대체하기에는 추가 검증이 필요하다.

Fig. 11의 Bland-Altman plot에서는 SBP의 98.8%, DBP의 98.2%가 95% agreement interval 안에 들어간다고 보고된다. 저자들은 이를 근거로 제안 시스템이 sphygmomanometer reference와 strong consistency를 보인다고 해석한다. 그러나 그림 caption과 discussion에서 고혈압 및 저혈압 일부 point는 confidence interval 밖에 위치한다고 언급한다. 이는 abnormal BP sample 수가 적어 해당 구간의 예측이 충분히 학습되지 않았기 때문이라고 설명된다.

### 4.4 조도 실험

NCBP 시스템에서는 light intensity가 매우 중요하다. 논문은 600 lx, 250 lx, 20 lx의 세 가지 light intensity에서 실험을 수행했다. 600 lx는 additional light source를 사용한 밝은 환경, 250 lx는 external light source 없는 indoor ambient light, 20 lx는 dark room에서 약한 external light source를 사용한 환경이다. Fig. 12는 세 조도 조건의 얼굴 영상을 보여준다.

Table 6에 따르면 SBP MAE는 20 lx에서 7.31 mmHg, 250 lx에서 6.69 mmHg, 600 lx에서 7.03 mmHg였고, DBP MAE는 각각 5.81 mmHg, 5.84 mmHg, 6.21 mmHg였다. STD와 mean bias도 조도에 따라 큰 변화가 없었다. 저자들은 stable ambient light 조건에서는 light intensity 변화가 실험 결과에 큰 영향을 주지 않으며, 충분한 ambient light가 있으면 별도의 external light source 없이도 NCBP 측정이 가능하다고 결론 내린다.

다만 이 실험은 20명의 healthy people을 대상으로 수행되었고, 다양한 피부색, 환경 온도, 실제 생활 중 motion artifact 등은 충분히 평가되지 않았다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 비접촉 방식의 혈압 추정 시스템을 비교적 단순한 장비로 구현했다는 점이다. 고속 카메라나 특수 조명 없이 일반 웹캠과 ambient light를 사용했다는 점은 실사용 가능성을 높인다. 또한 단일 부위 얼굴 영상만 사용했기 때문에, 두 부위를 동시에 촬영하여 PTT를 계산하는 방식보다 사용 편의성이 높다.

두 번째 강점은 IPPG pipeline의 실용적 요소를 구체적으로 다루었다는 점이다. 논문은 ROI 선택을 단순히 얼굴 전체로 두지 않고, colormap을 통해 light intensity가 강한 영역을 분석했다. 그 결과 forehead와 cheek이 강한 후보임을 확인했지만, 머리카락 가림 문제를 고려하여 cheek-nose 영역을 선택했다. 이는 실제 사용자 환경을 어느 정도 고려한 설계이다.

세 번째 강점은 peak detection 문제를 별도로 다루었다는 점이다. IPPG signal은 contact PPG보다 noise와 motion artifact에 취약하므로, peak와 valley 검출이 중요한 병목이 된다. 논문은 thresholding과 difference operation 기반 알고리즘을 제안하여 noisy waveform에서도 pulse period를 나눌 수 있도록 했다.

네 번째 강점은 여러 machine learning model을 비교했다는 점이다. 단일 모델만 제시하지 않고 MLR, SVR, RF, MLP를 비교함으로써, small sample condition에서 SVR이 가장 적합하다는 실험적 근거를 제시했다.

그러나 한계도 분명하다. 첫째, dataset 규모가 작다. 최종 유효 사례는 189개이고 test set은 56개에 불과하다. 특히 hypertension은 17명, hypotension은 10명으로 abnormal BP sample이 매우 적다. Fig. 11에서도 abnormal BP point에서 error가 커지는 경향이 언급되므로, 실제 임상적 위험군에서의 성능은 충분히 검증되었다고 보기 어렵다.

둘째, reference BP가 automatic sphygmomanometer로 측정되었다. 논문은 이 장비가 international gold standard에 맞춰 calibration되었다고 설명하지만, 저자 스스로도 future work에서는 manual sphygmomanometer를 reference로 사용해야 더 설득력 있다고 언급한다. 실제 혈압 추정 모델을 평가하려면 reference device의 정확성과 측정 시점 동기화가 매우 중요하다.

셋째, 사용 환경의 다양성이 부족하다. 논문은 stable ambient light 조건에서 외부 광원이 필요 없음을 보였지만, 다양한 temperature, skin color, 실제 생활 환경의 movement, 표정 변화, 카메라 거리, 배경 조명 변화 등에 대한 검증은 수행하지 않았다. 특히 얼굴 영상 기반 IPPG는 피부색, 주변광 spectrum, camera auto-exposure, compression artifact에 영향을 받을 수 있다.

넷째, temporal generalization과 subject-independent robustness에 대한 검증이 제한적이다. 논문은 데이터를 random split하여 70% training, 30% testing으로 나누었다고 설명하지만, subject 단위의 엄격한 분리 여부는 명확하지 않다. 본 연구의 데이터는 volunteer별 하나의 대표 사례처럼 보이나, 이 부분은 원문만으로 완전히 확정하기 어렵다. 만약 동일 subject에서 여러 sample이 train과 test에 동시에 들어간다면, 실제 신규 사용자에 대한 성능은 과대평가될 수 있다.

다섯째, deep learning이나 end-to-end 접근과 비교하지 않았다. MLP는 비교적 단순한 neural network이며, IPPG waveform 자체를 입력하는 CNN, RNN, Transformer 계열 모델과의 비교는 없다. 이 논문은 handcrafted feature 기반 machine learning 접근으로서 의미가 있지만, feature extraction과 peak detection에 성능이 크게 의존한다.

## 6. 결론

이 논문은 얼굴 영상 기반 IPPG와 machine learning을 결합하여 비접촉 혈압 추정 시스템을 구현한 연구이다. 제안 시스템은 웹캠으로 얼굴 영상을 촬영하고, cheek-nose ROI에서 green channel IPPG signal을 추출한 뒤, wavelet transform과 band-pass filter로 noise를 제거한다. 이후 robust peak detection으로 pulse period를 나누고, time-domain feature, KTE, K value, energy profile, HR 등 총 26개 feature를 추출한다. Pearson correlation 기반 feature selection을 거쳐 16개 feature를 선택한 뒤, MLR, SVR, RF, MLP를 비교하여 SVR이 가장 좋은 성능을 보임을 확인했다.

실험 결과 SVR은 SBP에서 STD 3.35 mmHg, MAE 9.97 mmHg, mean bias 2.10 mmHg를 기록했고, DBP에서 STD 2.58 mmHg, MAE 7.59 mmHg, mean bias 0.79 mmHg를 기록했다. AAMI error criterion 측면에서는 SBP와 DBP 모두 기준을 만족한다고 보고되었고, BHS 기준에서도 허용 가능한 수준의 결과를 보였다. 또한 조도 실험을 통해 stable ambient light 조건에서는 별도 external light source 없이도 혈압 추정이 가능하다는 점을 보였다.

이 연구는 cuffless, contactless, camera-based BP monitoring의 가능성을 보여준다는 점에서 의미가 있다. 특히 웹캠 하나로 얼굴 영상에서 혈압 정보를 추정하려는 접근은 telemedicine, home healthcare, kiosk-based screening, remote monitoring 등에서 활용 가능성이 있다. 그러나 실제 의료기기 수준의 적용을 위해서는 훨씬 더 큰 규모의 subject-independent validation, 다양한 혈압 범위와 피부색·조명·움직임 조건에서의 검증, 더 엄격한 reference BP 측정, abnormal BP group에 대한 충분한 데이터 확보가 필요하다. 따라서 이 논문은 완성된 임상 혈압계라기보다는, IPPG 기반 비접촉 혈압 추정이 가능함을 보인 초기 단계의 실험적 연구로 평가하는 것이 적절하다.
