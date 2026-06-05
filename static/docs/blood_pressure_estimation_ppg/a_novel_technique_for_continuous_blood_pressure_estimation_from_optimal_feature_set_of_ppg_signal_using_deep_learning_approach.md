# A Novel Technique for Continuous Blood Pressure Estimation from Optimal Feature Set of PPG Signal Using Deep Learning Approach

* **저자**: S. M. Taslim Uddin Raju, Safin Ahmed Dipto, Md Imran Hossain, Md. Abu Shahid Chowdhury, Fabliha Haque, Ayesha Tun Nashrah, Araf Nishan, Ashfaq Ahmad, Mostafa Zaman Chowdhury, M. M. A. Hashem
* **발표연도**: 2023

## 1. 논문 개요

이 논문은 cuff 기반 혈압 측정의 불편함을 줄이고, wearable 또는 mobile healthcare 환경에서 연속적인 혈압 모니터링을 가능하게 하기 위해 PPG, 즉 photoplethysmography 신호만을 사용하여 systolic blood pressure, SBP와 diastolic blood pressure, DBP를 추정하는 방법을 제안한다. 기존의 sphygmomanometry나 oscillometry 기반 혈압 측정 방식은 inflatable cuff를 사용하기 때문에 반복 측정 시 불편하고, 환자의 스트레스나 불안이 측정값에 영향을 줄 수 있다. 따라서 논문은 비침습적이고 cuff-less이며 연속적인 혈압 추정 방법의 필요성을 연구 문제로 설정한다.

논문의 핵심 목표는 PPG 신호에서 의미 있는 특징을 추출하고, 여러 feature selection 방법을 결합한 ensemble feature selection을 통해 최적의 feature set을 만든 뒤, 이를 feed-forward deep neural network, DNN에 입력하여 SBP와 DBP를 정확하게 추정하는 것이다. 저자들은 PPG-BP Database의 데이터를 사용하며, 원래 데이터는 219명으로부터 얻은 657개의 PPG record로 구성되어 있다. 그러나 신호 품질 평가와 전처리를 거친 뒤 최종적으로는 125명의 subject에 해당하는 218개 record를 사용한다.

이 문제가 중요한 이유는 혈압이 hypertension, cardiovascular disease, stroke, kidney disease 등과 직접적으로 관련된 핵심 생체 지표이기 때문이다. 특히 고혈압 환자나 심혈관 질환 위험군에서는 단발성 측정보다 시간에 따른 혈압 변화를 지속적으로 관찰하는 것이 중요하다. PPG는 저비용, 소형, 착용형 센서에 쉽게 통합될 수 있는 optical signal이므로, PPG만으로 혈압을 안정적으로 추정할 수 있다면 스마트워치, 스마트폰 기반 헬스케어, 원격 환자 모니터링 등에서 실용적 가치가 크다.

다만 이 논문은 ResearchSquare preprint 형식이며, 표지에 peer review를 거치지 않았다는 주의 문구가 있다. 따라서 논문의 결과는 임상적으로 확정된 성능이라기보다는 제안 방법의 초기 검증 결과로 이해하는 것이 적절하다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 “PPG 신호 하나만으로도 혈압과 관련된 충분한 형태학적, 시간 영역, 주파수 영역 특징을 얻을 수 있으며, 이 중 가장 유용한 feature subset을 선택하면 DNN 기반 혈압 추정 성능을 높일 수 있다”는 것이다.

기존 연구들 중 일부는 PPG와 ECG를 함께 사용하여 pulse transit time, PTT를 계산하거나, hand-crafted feature와 전통적인 machine learning regression model을 사용했다. 반면 이 논문은 ECG 없이 PPG 신호만 사용한다는 점을 강조한다. 또한 단일 feature selection 방법에 의존하지 않고, CFS, ReliefF, FSCMRMR, RFE라는 네 가지 feature selection 기법의 결과를 majority voting 방식으로 결합한다. 이 ensemble feature selection이 논문에서 제안하는 주요 차별점이다.

이 방식의 직관은 간단하다. 어떤 feature가 여러 feature selection 알고리즘에서 반복적으로 선택된다면, 그 feature는 특정 방법의 편향에 의해 우연히 선택된 것이 아니라 혈압 추정에 안정적으로 유용할 가능성이 높다. 논문은 네 가지 방법 중 적어도 두 가지 방법에서 선택된 feature를 최종 feature set에 포함한다. 즉, 각 feature가 받은 vote score를 계산하고, vote score가 threshold 이상인 feature를 선택한다.

또 다른 중요한 설계 아이디어는 전체 PPG waveform을 그대로 DNN에 넣는 것이 아니라, 먼저 신호를 정규화하고, filtering과 baseline correction을 수행한 뒤, peak detection을 통해 가장 적절한 single PPG cycle을 선택한다는 점이다. 이 single PPG cycle에서 systolic peak, diastolic peak, dicrotic notch, pulse interval, derivative signal, Fourier component 등 다양한 feature를 추출한다. 따라서 이 논문은 end-to-end raw waveform learning보다는 signal processing, feature engineering, feature selection, DNN regression을 결합한 hybrid approach에 가깝다.

## 3. 상세 방법 설명

논문이 제안하는 전체 파이프라인은 데이터 수집, PPG signal quality assessment, preprocessing, PPG cycle detection, feature extraction, ensemble feature selection, DNN-based BP estimation, 성능 평가의 순서로 구성된다. 논문 page 5의 Figure 2는 이 전체 과정을 block diagram으로 보여주며, public PPG database에서 입력 신호를 가져와 전처리, cycle detection, feature extraction, feature selection, DNN training과 testing을 거쳐 최종적으로 BP를 추정하는 흐름을 제시한다.

### 3.1 혈압 추정의 생리학적 배경

논문은 PPG 기반 혈압 추정의 이론적 배경으로 arterial pulse wave propagation과 pulse transit time, PTT의 관계를 설명한다. 혈관을 탄성 원통형 tube로 보고, 혈관벽의 탄성, 압력, pulse wave velocity가 서로 관련되어 있다고 가정한다. 중앙 동맥에서 tube의 elastic modulus $E$는 pressure $P$와 지수 관계를 갖는 것으로 표현된다.

$$
E = E_0 e^{\alpha(P - P_0)}
$$

여기서 $E_0$는 기준 압력 $P_0$에서의 elastic modulus이고, $\alpha$는 correction factor이다. 이 식은 혈압이 높아질수록 혈관의 탄성 특성이 달라지고, 결과적으로 pulse wave가 전달되는 속도와 시간이 변한다는 생리학적 직관을 담고 있다.

혈관의 compliance $C$는 압력 변화에 따른 단면적 변화율로 정의되며, 논문은 이를 다음과 같이 나타낸다.

$$
C(P) =
\frac{A_m}
{\pi P_1 \left[1 + \left(\frac{P - P_0}{P_1}\right)^2\right]}
$$

여기서 $A_m$, $P_0$, $P_1$은 subject마다 달라지는 값이다. 이후 pressure wave propagation은 다음과 같이 표현된다.

$$
P(x,t) = f\left(x \pm \frac{t}{\sqrt{LC(P)}}\right)
$$

이 식으로부터 pulse wave velocity가 $1/\sqrt{LC(P)}$와 관련된다는 점을 설명하고, 길이 $K$의 경로를 pulse wave가 이동하는 데 걸리는 시간인 PTT는 다음과 같이 표현된다.

$$
PTT = K\sqrt{LC(P)}
$$

이를 더 구체화하면 다음과 같다.

$$
PTT =
K
\sqrt{
\frac{\rho A_m}
{\pi A P_1 \left[1 + \left(\frac{P - P_0}{P_1}\right)^2\right]}
}
$$

논문은 이 식을 통해 PTT와 blood pressure 사이에 역관계가 있음을 설명한다. 즉, 혈압이 증가하면 arterial stiffness가 증가하고 pulse wave가 더 빠르게 전달되므로 PTT는 감소하는 경향을 보인다.

또한 pulse pressure $PP$는 PTT의 제곱에 반비례한다고 설명한다.

$$
PP = PP_0 \left(\frac{PTT_0}{PTT}\right)^2
$$

여기서 $PP_0$와 $PTT_0$는 초기 calibration 값이다. DBP는 PPG intensity ratio, PIR의 역수와 비례 관계를 갖는다고 제시된다.

$$
DBP \propto \frac{1}{PIR}
$$

초기 calibration 값을 사용하면 DBP는 다음과 같이 표현된다.

$$
DBP = DBP_0 \frac{PIR_0}{PIR}
$$

마지막으로 $PP = SBP - DBP$ 관계를 사용하여 SBP는 다음과 같이 유도된다.

$$
SBP =
DBP_0 \frac{PIR_0}{PIR}
+
PP_0 \left(\frac{PTT_0}{PTT}\right)^2
$$

다만 실제 제안 모델은 이 물리식을 직접 계산 모델로만 사용하는 것이 아니라, PPG에서 추출한 다양한 특징을 DNN에 입력하여 SBP와 DBP를 regression 방식으로 추정한다. 따라서 위 식들은 PPG waveform과 혈압 사이에 생리학적 관련성이 있음을 뒷받침하는 배경 이론으로 이해하는 것이 적절하다.

### 3.2 데이터셋과 신호 품질 평가

논문은 PPG-BP Database를 사용한다. 이 데이터셋은 219명의 subject에서 수집한 657개의 PPG recording으로 구성되어 있다. PPG 신호는 subject의 왼손 검지에서 customized probe를 통해 측정되었으며, sampling rate는 1 kHz이고 각 signal의 길이는 2.1초이다.

논문은 모든 PPG 신호를 그대로 사용하지 않는다. 일부 PPG signal은 noise가 많거나 systolic peak, diastolic peak, dicrotic notch 같은 정보성 특징이 명확하지 않기 때문에 feature extraction에 부적합하다. 이를 제거하기 위해 skewness quality index, SQI를 사용하여 PPG signal을 appropriate signal과 inappropriate signal로 나눈다. 적절한 신호는 systolic feature, diastolic feature, dicrotic notch가 명확히 보이는 waveform이고, 부적절한 신호는 이러한 주요 형태학적 특징이 부족한 waveform이다.

품질 평가와 전처리를 거친 최종 데이터는 125명의 subject에 해당하는 218개 record이다. 논문은 각 recording의 unique ID를 사용하여 training set과 test set에 같은 subject가 중복 포함되는 것을 방지했다고 설명한다. 이 점은 subject leakage를 줄이기 위한 중요한 실험 설계 요소이다.

### 3.3 PPG 신호 전처리

전처리는 normalization, signal filtration, baseline correction으로 구성된다.

첫째, normalization 단계에서는 z-score normalization을 사용한다. 원래 PPG signal $PPG_O$를 평균과 표준편차로 정규화하여 $PPG_n$을 만든다.

$$
PPG_n =
\frac{PPG_O - \mu(PPG_O)}
{\sigma(PPG_O)}
$$

여기서 $\mu(PPG_O)$는 원 신호의 평균이고, $\sigma(PPG_O)$는 원 신호의 표준편차이다. 이 과정은 subject마다 signal amplitude scale이 다를 수 있는 문제를 줄이고, 모델이 amplitude 범위 차이보다 waveform의 상대적 형태를 더 잘 활용하게 한다.

둘째, signal filtration 단계에서는 여러 filtering 방법을 비교한 뒤 7th order low-pass Butterworth filter를 선택한다. 논문은 Butterworth filter가 다른 방법들보다 phase response, systolic peak와 diastolic peak의 시각적 보존, noise reduction 측면에서 유리했다고 설명한다. 최종적으로 cutoff frequency는 15 Hz로 설정되었다. page 6의 Figure 4는 raw PPG signal 위에 Butterworth IIR zero-phase filtered signal을 겹쳐 보여주며, filtering 후 고주파 잡음이 줄어든 것을 시각적으로 확인할 수 있다.

셋째, baseline correction은 baseline wandering, BW를 제거하기 위한 단계이다. BW는 주로 subject movement와 respiration으로 인해 발생하는 저주파 artifact이며, 논문은 0.5 Hz 이하의 성분을 baseline wandering과 관련된 것으로 설명한다. 이 논문에서는 4차 polynomial을 signal에 fitting하여 trend를 추정하고, 이 trend를 원 신호에서 빼서 baseline-corrected signal을 얻는다. page 6의 Figure 5는 baseline wandering이 있는 filtered signal과 4th degree polynomial trend, 그리고 detrending 이후 residual signal을 보여준다.

### 3.4 PPG cycle detection과 single cycle 선택

논문은 전체 2.1초 PPG signal에서 하나의 single PPG cycle을 선택하여 feature를 추출한다. 일반적으로 2.1초 signal에는 두 개 이상의 PPG cycle이 포함될 수 있으므로, Algorithm 1을 통해 valid cycle을 탐지하고 그중 systolic peak amplitude가 가장 큰 cycle을 best single PPG cycle, $B_{PPG}$로 선택한다.

cycle detection 과정은 consecutive minima와 consecutive maxima를 이용한다. 시작점 $S_p$, dicrotic notch $z$, 끝점 $E_p$는 consecutive minima로 보고, systolic peak $x$와 diastolic peak $y$는 consecutive maxima로 본다. PPG cycle이 valid하려면 systolic peak, dicrotic notch, diastolic peak와 같은 전형적인 critical feature를 포함해야 한다. 또한 systolic peak $x$가 diastolic peak $y$보다 커야 하고, dicrotic notch $z$가 시작점과 끝점보다 커야 한다. 조건을 만족하지 않는 cycle은 버려진다.

page 7의 Figure 6은 continuous PPG waveform에서 maxima와 minima를 탐지하여 cycle을 분리하는 과정과, 최종적으로 선택된 single PPG cycle을 보여준다. 이 단계는 이후 feature extraction의 품질을 좌우한다. 잘못된 cycle을 선택하면 peak time, dicrotic notch, area ratio, derivative feature 등이 모두 왜곡될 수 있기 때문이다.

### 3.5 Feature extraction

논문은 선택된 single PPG cycle $B_{PPG}$, 그 1차 derivative, 2차 derivative, 그리고 Fourier transform 결과에서 총 46개의 signal feature를 추출한다. 여기에 age와 gender를 추가하여 전체 feature 수는 48개가 된다.

feature는 크게 네 그룹으로 볼 수 있다.

첫 번째 그룹은 원래 PPG cycle에서 얻은 morphology 및 time-domain feature이다. 여기에는 systolic peak $x$, diastolic peak $y$, dicrotic notch $z$, pulse interval $t_{pi}$, augmentation index $y/x$, alternative augmentation index $(x-y)/x$, dicrotic notch와 systolic peak의 비율 $z/x$, negative relative augmentation index $(y-x)/x$, systolic peak time $t_1$, dicrotic notch time $t_2$, diastolic peak time $t_3$, systolic peak와 diastolic peak 사이의 시간 차이 $\Delta T = t_3 - t_1$, full width at half systolic peak $w$, inflection point area ratio $A_2/A_1$, stress-induced vascular response index $sVRI = V_2/V_1$ 등이 포함된다.

두 번째 그룹은 PPG의 1차 derivative에서 얻은 시간 간격 feature이다. 예를 들어 $t_{a1}$, $t_{b1}$, $t_{e1}$, $t_{l1}$와 이들을 pulse interval $t_{pi}$로 나눈 ratio feature가 포함된다. derivative feature는 원 신호에서 직접 보이지 않는 상승 기울기, 변화율, inflection 관련 정보를 반영한다.

세 번째 그룹은 PPG의 2차 derivative, 즉 acceleration plethysmogram과 관련된 feature이다. 여기에는 $b_2/a_2$, $e_2/a_2$, $(b_2+e_2)/a_2$, $t_{a2}$, $t_{b2}$, $t_{a2}/t_{pi}$, $t_{b2}/t_{pi}$ 등이 포함된다. 2차 derivative는 혈관 탄성, waveform curvature, pulse contour 변화와 관련된 정보를 제공할 수 있다.

네 번째 그룹은 Fourier transform된 $B_{PPG}$에서 얻은 frequency-domain feature이다. primary component frequency $f_{base}$와 magnitude $|s_{base}|$, second component frequency와 magnitude, third component frequency와 magnitude가 포함된다.

마지막으로 demographic feature인 age와 gender가 추가된다. Table 3에서 age는 $F47$, gender는 $F48$로 정의된다. 실제 feature selection 결과에서 age $F47$은 SBP와 DBP 모두에서 가장 강하게 선택된 feature로 나타나며, 이는 혈압 추정에서 나이 정보가 매우 중요한 역할을 했음을 시사한다.

### 3.6 Ensemble feature selection

논문은 전체 48개 feature를 모두 사용하는 대신, feature selection을 통해 relevant하고 non-redundant한 feature subset을 선택한다. 사용된 feature selection 방법은 CFS, ReliefF, FSCMRMR, RFE의 네 가지이다.

CFS, 즉 correlation-based feature selection은 target과 관련성이 높으면서 feature들끼리는 중복성이 낮은 feature를 선호한다. 논문은 correlation coefficient의 p-value를 사용하여 feature subset을 구성했다고 설명한다.

ReliefF는 nearest neighbor 개념을 이용해 feature의 중요도를 계산하는 filter-based feature selection 방법이다. feature 간 상호작용을 간접적으로 포착할 수 있다는 장점이 있다.

FSCMRMR은 minimum redundancy maximum relevance 기준을 사용하는 방법이다. 즉, target과의 관련성은 높고 feature들 사이의 redundancy는 낮은 조합을 찾는다. 이 방법은 feature를 하나씩 독립적으로 평가하기보다 relevance와 redundancy 사이의 균형을 고려한다.

RFE, 즉 recursive feature elimination은 모델 예측에 덜 중요한 feature를 반복적으로 제거하면서 최종 feature subset을 구성한다. collinearity를 줄이고 target prediction에 중요한 feature를 남기는 목적을 가진다.

논문의 ensemble feature selection은 이 네 방법의 결과를 majority voting 방식으로 결합한다. 각 feature에 대해 어떤 feature selection method가 그 feature를 선택하면 1표를 부여한다. 네 방법 중 적어도 두 방법에서 선택된 feature만 최종 feature set에 포함된다. 이를 수식적으로 표현하면 각 feature $F_t$의 voting score는 다음과 같은 개념으로 이해할 수 있다.

$$
VScore_t = \sum_{k=1}^{K} I(F_t \in FS_k)
$$

여기서 $K=4$이고, $FS_k$는 $k$번째 feature selection method가 선택한 feature subset이다. $I(\cdot)$는 조건이 참이면 1, 거짓이면 0이 되는 indicator function이다. 논문은 threshold $\theta = 2$를 사용하므로 최종 선택 조건은 다음과 같다.

$$
F_t \in List_V \quad \text{if} \quad VScore_t \geq 2
$$

SBP 추정에서는 $F47$, $F28$, $F25$, $F45$, $F14$, $F26$, $F2$, $F4$, $F6$, $F3$, $F41$, $F48$ 등이 선택되었다. DBP 추정에서는 $F47$, $F45$, $F25$, $F14$, $F3$, $F41$, $F6$, $F12$, $F26$, $F40$, $F27$, $F28$, $F4$, $F1$, $F2$, $F48$ 등이 선택되었다. 특히 $F47$, 즉 age는 SBP와 DBP 모두에서 네 개 방법 모두의 vote를 받아 voting score 4를 기록했다. 이는 이 데이터셋에서 age가 혈압 추정에 매우 강한 설명력을 가졌다는 의미이다.

### 3.7 DNN 모델 구조와 학습

논문은 feed-forward deep neural network를 사용하여 SBP와 DBP를 추정한다. page 9의 Figure 7은 입력 feature가 네 개 hidden layer를 거쳐 두 개의 output, 즉 $\hat{y}*{SBP}$와 $\hat{y}*{DBP}$로 이어지는 구조를 보여준다.

입력층의 neuron 수는 전체 feature를 사용할 때 48개이고, feature selection을 사용할 때는 선택된 feature 수에 따라 달라진다. hidden layer는 총 네 개이며, 각 layer의 neuron 수는 50, 100, 150, 200이다. 두 번째 hidden layer에는 dropout 0.25가 적용되고, 네 번째 hidden layer에는 dropout 0.5가 적용된다. activation function은 hidden layer에서 ReLU를 사용하고, output layer에서는 linear activation을 사용한다.

한 hidden unit의 선형 결합은 다음과 같이 표현된다.

$$
BP_i = \sum_i \omega_i F_i + \beta
$$

여기서 $F_i$는 입력 feature, $\omega_i$는 weight, $\beta$는 bias이다. 이후 ReLU activation은 다음과 같이 적용된다.

$$
\phi_{Re}(BP) = \max(0, BP)
$$

output layer에서는 regression 문제에 적합하도록 linear activation을 사용한다.

$$
\phi_{Li}(BP) = BP'
$$

모델은 Adam optimizer로 학습되며, epoch 수는 100, batch size는 32, learning rate는 0.01이다. 10-fold cross-validation이 사용되며, 논문은 subject의 unique ID를 활용해 training set과 test set 사이의 subject overlap을 방지했다고 설명한다.

### 3.8 성능 평가 지표

논문은 ME, STD, MAPE, MSE, RMSE, MAE, $R^2$를 사용하여 성능을 평가한다. 핵심적으로 보고되는 지표는 MAE와 $R^2$이다. MAE는 실제 혈압과 예측 혈압 사이의 절대 오차 평균이다.

$$
MAE = \frac{1}{n} \sum_n |BP_a - BP_e|
$$

여기서 $BP_a$는 reference blood pressure이고, $BP_e$는 estimated blood pressure이다. $R^2$는 모델이 reference value의 분산을 얼마나 설명하는지를 나타낸다.

$$
R^2 =
1 -
\frac{
\sum_n (BP_a - BP_e)^2
}{
\sum_n (BP_a - \overline{BP})^2
}
$$

$R^2$가 1에 가까울수록 예측값이 실제값을 잘 설명한다. 다만 작은 데이터셋에서 cross-validation이 사용된 경우에도, 데이터 분할 방식이나 subject-level 독립성이 충분히 엄격하지 않으면 실제 일반화 성능보다 높게 보일 수 있으므로 해석에 주의가 필요하다.

## 4. 실험 및 결과

논문은 PPG-BP Database에서 품질 평가를 통과한 218개 record와 125명 subject를 대상으로 실험을 수행했다. task는 PPG feature를 입력으로 하여 SBP와 DBP를 추정하는 regression이다. 기준선은 전체 feature를 사용한 DNN, 그리고 CFS, ReliefF, FSCMRMR, RFE 각각의 feature selection 결과를 사용한 DNN이다. 제안 방법은 네 feature selection 방법의 majority voting으로 얻은 ensemble feature subset을 DNN에 입력하는 방식이다.

Table 5는 각 feature selection 방법이 선택한 feature 목록을 보여준다. SBP의 경우 CFS와 ReliefF는 각각 16개 feature를 선택했고, FSCMRMR은 5개, RFE는 6개를 선택했다. DBP의 경우 CFS는 17개, ReliefF는 16개, FSCMRMR은 6개, RFE는 9개를 선택했다. Table 6은 ensemble voting score를 보여주며, SBP와 DBP 모두에서 age feature $F47$이 voting score 4로 가장 중요하게 나타났다.

Table 7의 결과가 논문의 핵심 실험 결과이다. 전체 feature를 사용한 DNN은 SBP에서 $R^2 = 0.867$, MAE = 3.631 mmHg를 기록했고, DBP에서 $R^2 = 0.805$, MAE = 2.387 mmHg를 기록했다. 이는 feature selection 없이 모든 feature를 사용하는 것이 항상 최선은 아님을 보여준다.

개별 feature selection 방법 중에서는 SBP의 경우 CFS가 가장 좋은 성능을 보였으며, $R^2 = 0.952$, MAE = 2.804 mmHg를 기록했다. DBP의 경우 ReliefF가 상대적으로 좋은 성능을 보여 $R^2 = 0.936$, MAE = 1.796 mmHg를 기록했다.

제안한 ensemble feature selection과 DNN의 조합은 가장 높은 성능을 냈다. SBP에서는 $R^2 = 0.962$, MAE = 2.480 mmHg, MSE = 13.760, RMSE = 3.709, MAPE = 3.187을 기록했다. DBP에서는 $R^2 = 0.955$, MAE = 1.499 mmHg, MSE = 4.869, RMSE = 2.206, MAPE = 2.721을 기록했다. 즉, feature selection을 통해 불필요하거나 중복된 feature를 제거하고 DNN에 더 적절한 입력을 제공했을 때 성능이 향상되었다.

page 11의 Figure 8은 SBP와 DBP 추정 error의 histogram을 보여준다. 두 경우 모두 error가 0 근처에 집중되어 있으며, 논문은 error distribution이 대체로 normal distribution 형태를 보인다고 설명한다. 다만 SBP error의 분산이 DBP보다 큰데, 이는 SBP target value 자체의 분산이 DBP보다 크기 때문이라고 해석한다.

page 12의 Figure 9는 correlation plot과 Bland-Altman plot을 보여준다. SBP와 DBP 모두 estimated value가 reference value와 높은 상관을 보이며, 대부분의 sample이 Bland-Altman plot의 agreement limit 안에 들어간다. 그러나 매우 높거나 낮은 blood pressure sample에서는 예측 정확도가 상대적으로 떨어지는 경향이 나타난다. 논문은 이를 training set에 extreme high BP 또는 extreme low BP subject가 적기 때문이라고 설명한다.

논문은 제안 방법을 AAMI standard와 BHS standard로도 평가한다. AAMI 기준에서는 mean error가 5 mmHg 이하이고 standard deviation이 8 mmHg 이하이어야 하며, subject 수는 최소 85명 이상이어야 한다. 제안 방법은 125명 subject를 사용했고, SBP의 ME는 -0.471 mmHg, STD는 4.105 mmHg이며, DBP의 ME는 -0.049 mmHg, STD는 2.194 mmHg이다. 따라서 논문은 제안 방법이 AAMI 기준을 만족한다고 주장한다.

BHS standard에서는 누적 오차 비율이 5, 10, 15 mmHg 이하에 얼마나 들어가는지에 따라 grade가 결정된다. 제안 방법은 SBP에서 각각 91%, 95%, 99%를 기록했고, DBP에서 각각 96%, 99%, 100%를 기록했다. 논문은 이를 바탕으로 SBP와 DBP 모두 Grade A를 달성했다고 보고한다.

Table 10은 기존 연구와의 비교를 보여준다. 제안 방법은 PPG-BP dataset에서 DNN을 사용하여 SBP MAE ± STD = 2.48 ± 7.83 mmHg, DBP MAE ± STD = 1.49 ± 4.02 mmHg를 기록했다. 논문은 이 결과가 MIMIC II, University of Queensland, PPG-BP dataset을 사용한 여러 기존 PPG-only 방법보다 우수하다고 주장한다. 다만 비교 대상 연구들은 dataset, subject 수, preprocessing, validation protocol이 서로 다르기 때문에 절대적인 성능 우열로 해석하기에는 한계가 있다.

## 5. 강점, 한계

이 논문의 첫 번째 강점은 PPG signal만을 사용하여 SBP와 DBP를 모두 추정하려 했다는 점이다. ECG나 다른 reference physiological signal 없이 PPG만 사용하면 wearable device나 mobile device에 적용하기 훨씬 쉽다. 이는 실제 healthcare application에서 중요한 장점이다.

두 번째 강점은 signal preprocessing부터 feature engineering, feature selection, DNN regression까지 비교적 체계적인 pipeline을 구성했다는 점이다. 특히 normalization, low-pass Butterworth filtering, baseline correction, single PPG cycle detection을 통해 feature extraction 이전에 signal quality를 개선하려 했다. PPG는 motion artifact와 noise에 취약하기 때문에 이러한 preprocessing은 필수적인 단계이다.

세 번째 강점은 ensemble feature selection을 사용했다는 점이다. 단일 feature selection method는 특정 기준에 편향될 수 있다. 예를 들어 CFS는 correlation 구조에 민감하고, ReliefF는 nearest neighbor 구조에 의존하며, RFE는 사용된 모델과 학습 데이터에 영향을 받을 수 있다. 논문은 네 방법을 majority voting으로 결합하여 보다 안정적인 feature subset을 얻으려 했다. 실제 실험에서도 ensemble feature selection이 개별 방법 및 전체 feature 사용보다 더 좋은 성능을 보였다.

네 번째 강점은 AAMI와 BHS 같은 혈압 측정 장치 평가 기준을 함께 보고했다는 점이다. 단순히 MAE나 $R^2$만 제시하는 것보다, 의료기기 평가에서 사용되는 기준과 비교함으로써 결과의 실용적 의미를 더 명확히 전달한다.

그러나 한계도 분명하다. 첫째, 최종 학습과 평가에 사용된 데이터 규모가 작다. 원래 데이터셋은 219명, 657 record였지만, 품질 평가 이후 최종적으로 125명, 218 record만 사용되었다. DNN을 학습하기에는 상당히 작은 규모이며, 특히 extreme high BP나 extreme low BP sample이 적어 해당 구간에서 성능이 떨어진다는 점을 논문 스스로 인정한다.

둘째, 논문은 10-fold cross-validation과 unique ID 기반 subject overlap 방지를 언급하지만, 데이터 분할의 세부 구현이 충분히 자세히 설명되어 있지는 않다. 예를 들어 같은 subject의 여러 record가 완전히 같은 fold에만 배정되었는지, preprocessing과 feature selection이 cross-validation 내부에서만 수행되었는지, 혹은 전체 데이터에서 feature selection을 먼저 수행한 뒤 cross-validation을 했는지 명확하지 않다. 만약 feature selection이 전체 데이터에 대해 먼저 수행되었다면 test fold 정보가 feature selection에 간접적으로 반영되는 data leakage 가능성이 있다. 논문 본문만으로는 이 부분을 확정할 수 없다.

셋째, demographic feature인 age가 가장 높은 voting score를 얻었다는 점은 장점이자 한계이다. age는 혈압과 강한 관련이 있으므로 모델 성능을 높일 수 있지만, 이는 PPG waveform 자체의 정보만으로 혈압을 추정했다기보다는 subject-level demographic prior를 크게 활용한 것일 수 있다. 실제 wearable 환경에서 개인별 calibration 없이 새로운 사용자에게 적용할 경우, age와 gender만으로 설명되는 population-level correlation이 얼마나 안정적으로 작동할지는 추가 검증이 필요하다.

넷째, 비교 실험의 공정성에 한계가 있다. Table 10의 기존 연구들은 서로 다른 dataset, subject 수, sampling protocol, validation method를 사용한다. 따라서 제안 방법이 수치상 더 좋은 MAE를 보였다고 해서 모든 조건에서 기존 방법보다 우수하다고 단정하기는 어렵다. 동일한 dataset과 동일한 subject-independent split에서 재현 비교가 필요하다.

다섯째, 논문은 clinical application 가능성을 언급하지만, 실제 clinical deployment를 위해 필요한 외부 검증, 다양한 연령대와 질환군, 장기간 측정, motion-heavy 상황, device variation, sensor placement variation에 대한 검증은 충분하지 않다. 또한 ResearchSquare preprint이므로 peer review를 통한 검증이 아직 완료되지 않았다는 점도 결과 해석 시 고려해야 한다.

## 6. 결론

이 논문은 PPG signal만을 이용한 cuff-less continuous blood pressure estimation을 위해 signal preprocessing, morphology 및 frequency-domain feature extraction, ensemble feature selection, DNN regression을 결합한 방법을 제안한다. 핵심 기여는 네 가지 feature selection 방법, 즉 CFS, ReliefF, FSCMRMR, RFE의 결과를 majority voting으로 통합하여 SBP와 DBP 추정에 유용한 optimal feature set을 구성한 점이다.

실험 결과에서 제안 방법은 SBP에 대해 $R^2 = 0.962$, MAE = 2.480 mmHg를 기록했고, DBP에 대해 $R^2 = 0.955$, MAE = 1.499 mmHg를 기록했다. 또한 AAMI 기준을 만족하고 BHS 기준에서 SBP와 DBP 모두 Grade A를 달성했다고 보고한다. 이는 feature selection과 DNN을 결합한 PPG-only 혈압 추정 방식이 높은 잠재력을 가질 수 있음을 보여준다.

실제 적용 측면에서 이 연구는 wearable healthcare device, smartphone-based monitoring, remote patient monitoring 등에 응용될 가능성이 있다. 특히 cuff 없이 연속적으로 혈압을 추정할 수 있다면 고혈압 관리와 조기 위험 감지에 유용할 수 있다. 그러나 최종 데이터 규모가 작고, extreme BP range에서 성능이 약하며, 외부 데이터셋 검증과 엄격한 subject-independent evaluation이 충분히 제시되지 않았다는 한계가 있다. 따라서 이 연구는 임상적으로 확정된 혈압 측정 솔루션이라기보다는, PPG feature engineering과 ensemble feature selection이 DNN 기반 혈압 추정 성능을 향상시킬 수 있음을 보여주는 유망한 초기 연구로 보는 것이 적절하다.
