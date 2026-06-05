# Crisp-BP: Continuous Wrist PPG-based Blood Pressure Measurement

* **저자**: Yetong Cao, Huijie Chen, Fan Li, Yu Wang
* **발표연도**: 2022

## 1. 논문 개요

이 논문은 손목 착용형 기기에서 얻을 수 있는 photoplethysmography, 즉 PPG 신호만을 이용하여 arterial blood pressure, 즉 ABP를 연속적으로 추정하는 시스템인 **Crisp-BP**를 제안한다. 논문의 목표는 기존 wearable blood pressure monitoring 방식이 가진 여러 한계, 즉 불연속 측정, 사용자 행동 요구, 낮은 정확도, 사용자별 calibration data 수집 부담을 줄이면서도 손목 기반 PPG 센서만으로 정확하고 user-independent한 혈압 측정을 가능하게 하는 것이다.

기존 cuff-based blood pressure monitor는 비교적 정확하지만 혈관을 압박해야 하므로 반복적 또는 장기적 측정에 불편함이 크다. ECG와 PPG를 함께 사용하는 smartwatch 기반 방식은 pulse arrival time, 즉 PAT 또는 pulse transit time, 즉 PTT를 계산할 수 있지만, 사용자가 ECG pad에 손가락을 대는 등 적극적인 동작을 해야 한다. 또 다른 wearable 방식들은 glasses, compression shorts, in-ear balloon 등 특수한 장치를 요구하거나 특정 상황에서만 동작한다. 이 논문은 상용 wrist-worn device에 이미 탑재될 수 있는 단일 PPG 센서를 활용하여 이러한 사용성 문제를 완화하려 한다.

Crisp-BP의 핵심 연구 문제는 다음과 같이 정리할 수 있다. 첫째, 손목의 단일 PPG 센서만으로 혈압과 관련된 충분한 생리적 정보를 얻을 수 있는가. 둘째, PPG 신호가 contact pressure, motion artifact, capillary pulse 등의 영향을 크게 받는 상황에서 reliable한 arterial pulse를 추출할 수 있는가. 셋째, 사용자별 training data를 요구하지 않고 새로운 사용자에게도 적용 가능한 user-independent blood pressure estimation model을 만들 수 있는가.

이 문제는 임상적, 실용적 중요성이 크다. Hypertension은 증상이 거의 없어 조기 진단이 어렵고, 지속적인 혈압 모니터링은 고혈압 관리와 심혈관 질환 예방에 직접적으로 연결된다. 하지만 일상 환경에서 불편함 없이 장시간 혈압을 측정하는 기술은 아직 제한적이다. Crisp-BP는 손목 착용 기기 기반의 continuous, comfortable, convenient, accurate ABP monitoring을 목표로 한다는 점에서 wearable healthcare 분야의 중요한 문제를 다룬다.

## 2. 핵심 아이디어

Crisp-BP의 중심 아이디어는 단일 PPG 센서로 **reflected wave transit time**, 즉 RWTT를 측정하고, 이 RWTT와 보조적인 vascular feature를 이용해 DBP와 SBP를 추정하는 것이다. RWTT는 혈액의 forward blood flow와 arterial tree의 분기점 등에서 반사되어 되돌아오는 backward blood flow가 같은 측정 지점에 도달하는 시간 차이를 의미한다. 혈압이 높거나 혈관의 탄성이 변하면 pulse wave velocity, 즉 PWV가 변하고, 이에 따라 반사파가 돌아오는 시간도 달라진다. 따라서 RWTT는 ABP와 상관관계를 가질 수 있다.

기존의 PTT 또는 PAT 방식은 일반적으로 ECG와 PPG 또는 두 개의 PPG 센서를 필요로 한다. 반면 Crisp-BP는 단일 wrist PPG 센서에서 관측되는 waveform 내부의 forward wave와 reflected wave의 시간적 구조를 분석한다. 이를 위해 원래 PPG waveform보다 시간적 fiducial point가 더 뚜렷하게 나타나는 **acceleration photoplethysmography**, 즉 APG를 사용한다. APG는 PPG의 second derivative이며, 논문은 APG의 fiducial point 중 systolic slope와 backward blood flow의 augmentation point 사이의 시간 간격을 RWTT로 정의한다.

두 번째 핵심 아이디어는 두 파장의 PPG, 즉 green PPG와 infrared PPG를 활용해 capillary pulse interference를 제거하고 arterial pulse를 분리하는 것이다. Green light는 주로 superficial capillary에 도달하고, infrared light는 더 깊게 침투하여 artery까지 도달할 수 있다. 따라서 green PPG에는 capillary pulse가 강하게 포함되고, IR PPG에는 capillary pulse와 arterial pulse가 혼합되어 있다고 본다. Crisp-BP는 modified Beer-Lambert law를 기반으로 두 신호의 차이를 조합하여 capillary contribution을 줄이고 arterial pulse를 추정한다.

세 번째 핵심 아이디어는 contact pressure를 별도로 추정하고 제어한다는 점이다. PPG waveform은 센서와 피부 사이의 압력에 매우 민감하다. 압력이 너무 낮으면 motion artifact가 커지고, 너무 높으면 혈관이 눌려 blood volume change가 잘 관측되지 않는다. Crisp-BP는 LSSVM 기반 contact pressure estimation을 통해 사용자가 wristband를 더 조이거나 느슨하게 하도록 안내하여 optimal contact pressure 근처에서 데이터를 수집하도록 한다.

마지막으로, Crisp-BP는 user-independent monitoring을 위해 BLSTM 기반 hybrid neural network를 사용하고, MIMIC III의 online PPG/ABP 데이터로부터 general knowledge를 학습한 뒤 Crisp-BP target domain에 transfer learning을 적용한다. 이는 사용자별 calibration data를 요구하는 기존 방식과 차별화되는 중요한 설계이다.

## 3. 상세 방법 설명

### 3.1 전체 시스템 구조

Crisp-BP는 크게 네 가지 모듈로 구성된다. 첫째, **Two-Wavelength Light Sensing** 모듈은 green LED와 infrared LED를 번갈아 켜고, 반사된 빛의 intensity를 photodetector로 측정한다. 둘째, **Device State Identification** 모듈은 PPG sensor와 skin 사이의 contact pressure가 적절한지 판단한다. 셋째, **Arterial Pulse Profiling** 모듈은 PPG signal에서 noise를 제거하고, modified Beer-Lambert law에 기반하여 capillary interference를 제거한 arterial pulse를 추출한다. 넷째, **Continuous ABP Monitoring** 모듈은 APG에서 RWTT와 여러 vascular feature를 추출하고, BLSTM-based hybrid neural network를 사용해 SBP와 DBP를 추정한다.

이 시스템은 proof-of-concept prototype으로 구현되었다. 논문에 따르면 상용 smartwatch 제조사는 일반적으로 raw PPG data에 직접 접근할 수 있도록 하지 않기 때문에, 연구진은 commodity PPG sensor, adjustable wrist band, embedded evaluation board로 구성된 prototype을 직접 제작했다.

### 3.2 RWTT와 APG 기반 혈압 추정

혈관을 elastic tube로 볼 때, pulse wave는 forward blood flow와 backward blood flow의 조합으로 나타난다. 혈액이 arterial tree의 bifurcation 등에서 resistance change를 만나면 일부 blood flow가 backward direction으로 반사된다. 이 reflected wave가 측정 지점에 돌아오면 혈관 내 volume change와 pressure waveform에 추가적인 변화를 만든다.

PPG는 skin/tissue에 빛을 조사하고 반사 또는 흡수 변화를 측정함으로써 blood volume change를 감지한다. Wrist reflective PPG의 waveform은 혈관 확장과 수축에 따른 light absorption change를 반영하며, pulse wave와 유사한 형태를 가진다. 그러나 원래 PPG waveform에서는 fiducial point가 불명확할 수 있으므로 Crisp-BP는 PPG의 second derivative인 APG를 사용한다.

논문은 blood volume $V$를 혈관 단면적 $S$와 blood-contained portion의 길이 $L$의 곱으로 표현한다.

$$
V = S \cdot L
$$

이를 시간 $t$에 대해 두 번 미분하면 다음과 같다.

$$
\frac{d^2V}{dt^2} = 2 \cdot \frac{dS}{dt} \cdot \frac{dL}{dt} + L \cdot \frac{d^2S}{dt^2} + S \cdot \frac{d^2L}{dt^2}
$$

이 식의 각 항은 혈류 속도와 혈관 확장 속도의 곱, 혈관 확장 가속도, 혈류 가속도와 관련된다. 혈압은 혈류 속도와 혈류 가속도에 영향을 주므로, $V$의 second derivative는 arterial pressure change와 관련된 정보를 포함한다. APG는 blood volume acceleration의 변화를 보여주므로 RWTT를 추출하는 데 적합하다고 논문은 설명한다.

APG waveform에는 일반적으로 $a$, $b$, $c$, $d$, $e$의 fiducial point가 나타난다. Crisp-BP는 RWTT를 APG waveform에서 $a$와 $e$ 사이의 시간 간격으로 측정한다. 여기서 $a$는 forward blood flow와 관련된 systolic slope 지점으로, $e$는 backward blood flow가 도달하는 augmentation point로 해석된다.

### 3.3 Contact pressure estimation

PPG waveform은 contact pressure에 따라 크게 변한다. 센서가 너무 느슨하게 접촉하면 피부와 센서 사이의 상대 움직임이 커져 motion artifact가 증가한다. 반대로 너무 강하게 누르면 vessel occlusion이 발생하여 혈액량 변화가 잘 감지되지 않는다. 따라서 Crisp-BP는 적절한 contact pressure를 유지하는 것을 핵심 조건으로 본다.

논문은 contact pressure가 증가함에 따라 PPG AC component의 amplitude가 처음에는 증가하다가, optimal contact pressure, 즉 OCP를 지난 뒤 다시 감소한다고 설명한다. OCP는 AC component amplitude가 가장 큰 압력으로 정의된다.

Contact pressure estimation을 위해 Crisp-BP는 vascular elasticity와 vascular resistance를 반영하는 feature 후보들을 고려한다. 예를 들어 pulse width, systolic phase와 diastolic phase의 시간 비율, pulse propagation time, diastolic area와 systolic area의 비율, systolic peak amplitude와 diastolic peak amplitude의 비율 등이 후보가 될 수 있다. 그러나 feature를 너무 많이 사용하면 multicollinearity로 인해 모델이 불안정해질 수 있다. 따라서 논문은 Variance Inflation Factor, 즉 VIF를 비교하여 가장 낮은 VIF를 갖는 두 feature를 선택한다.

선택된 feature는 다음과 같다. 첫 번째 feature $F_1$은 systolic phase와 diastolic phase 사이의 time interval ratio이다. APG에서 start point부터 $b$까지의 interval과 $b$부터 end point까지의 interval의 비율로 측정된다. 두 번째 feature $F_2$는 systolic peak amplitude와 diastolic peak amplitude의 비율이며, APG의 $a$와 $b$가 발생하는 시점에서의 PPG amplitude ratio로 측정된다.

이 두 feature를 입력으로 하여 Least-Squares Support Vector Machine, 즉 LSSVM을 학습한다. LSSVM은 적은 training data에서도 ill-posed problem을 다루는 데 유용하다고 설명된다. Training은 offline으로 수행되고, 사용자가 Crisp-BP를 사용할 때는 pre-trained LSSVM이 beat-to-beat contact pressure를 추정한다.

OCP를 찾기 위해 calibration 절차도 수행한다. 사용자가 smartwatch를 착용할 때 기기를 천천히 눌러 contact pressure를 증가시켜 PPG AC component가 완전히 사라지게 하고, 다시 천천히 release하여 접촉이 느슨해질 때까지 압력을 낮춘다. 이 과정에서 AC component amplitude가 최대가 되는 압력을 OCP로 추정한다. 현재 contact pressure가 OCP보다 높으면 wristband를 느슨하게 하고, OCP보다 낮으면 wristband를 조이도록 안내한다. 원문에는 안내 방향 표현에 다소 혼동될 수 있는 문장이 있으나, 전체 맥락상 목표는 OCP 근처의 contact pressure를 유지하는 것이다.

### 3.4 Arterial pulse profiling과 noise reduction

Arterial Pulse Profiling은 PPG data에서 ABP 추정에 필요한 arterial pulse를 얻는 단계이다. 먼저 noise reduction을 위해 fourth-order Butterworth filter를 적용하며 cutoff frequency는 $[0.5, 8]$ Hz이다. 이 범위는 일반적인 heart rate가 60–100 bpm이라는 점을 고려하여, pulse 관련 frequency component를 남기고 baseline drift와 high-frequency noise를 줄이기 위한 설정이다.

이후 beat-to-beat sequence를 대략적으로 segment하고, percentage change method를 사용하여 motion artifact와 outlier pulse를 제거한다. 구체적으로 interbeat interval이 이전 네 개 accepted interval의 평균에서 30% 이상 벗어나면 outlier로 간주한다.

### 3.5 Modified Beer-Lambert law 기반 arterial pulse extraction

Crisp-BP의 중요한 기술적 기여 중 하나는 capillary pulse interference를 제거하는 arterial pulse extraction이다. 손목 PPG에는 artery뿐 아니라 superficial capillary bed의 volume change도 포함된다. Capillary pulse는 artery pulse와 도달 시간이 다르므로 RWTT 추정에 오류를 유발할 수 있다.

Crisp-BP는 두 파장 PPG의 penetration depth 차이를 이용한다. Green light는 상대적으로 얕은 capillary layer에 주로 도달하고, infrared light는 더 깊은 subcutaneous tissue의 artery까지 도달한다. 따라서 green PPG는 capillary pulse를 주로 반영하고, IR PPG는 capillary pulse와 arterial pulse가 섞인 신호로 모델링한다.

광학 밀도 변화는 detected light intensity $I(t)$와 incident light intensity $I_0$의 비율로 표현된다.

$$
\Delta OD(t) = -\log\left(\frac{I(t)}{I_0}\right)
$$

논문은 modified Beer-Lambert law에 따라 이를 다음과 같이 근사한다.

$$
\Delta OD(t)
\approx
\langle L \rangle \Delta \mu_a(t) +
\left(\frac{\mu_a^0}{\mu_s^{\prime 0}}\right)
\langle L \rangle \Delta \mu_s'(t)
\approx
\langle L \rangle \Delta \mu_a(t)
$$

여기서 $\langle L \rangle$은 differential path length이고, $\Delta \mu_a(t)$는 absorption change이다. Tissue scattering change인 $\Delta \mu_s'(t)$는 hemodynamic concentration variation에 비해 무시할 수 있다고 가정한다.

피부와 혈관 구조를 두 layer의 homogeneous medium으로 단순화한다. 첫 번째 layer는 capillary를, 두 번째 layer는 arteriole과 artery를 나타낸다. Green wavelength를 $\lambda_g$, infrared wavelength를 $\lambda_{IR}$라고 할 때 optical density change는 다음과 같이 표현된다.

$$
\Delta OD_{\lambda_g}(t) = \langle L_1^{\lambda_g} \rangle \cdot \epsilon_1^{\lambda_g} \cdot \Delta C_1(t)
$$

$$
\Delta OD_{\lambda_{IR}}(t) = \langle L_1^{\lambda_{IR}} \rangle \cdot \epsilon_1^{\lambda_{IR}} \cdot \Delta C_1(t) + \langle L_2^{\lambda_{IR}} \rangle \cdot \epsilon_2^{\lambda_{IR}} \cdot \Delta C_2(t)
$$

여기서 $\Delta C_1(t)$는 capillary layer의 blood volume fraction change이고, $\Delta C_2(t)$는 artery layer의 blood volume fraction change이다. 목표는 $\Delta C_2(t)$, 즉 arterial pulse를 추정하는 것이다.

논문은 이를 다음과 같이 정리한다.

$$
\Delta C_2(t) = \frac{ \Delta OD_{\lambda_{IR}}(t) - \frac{ \langle L_1^{\lambda_{IR}} \rangle \epsilon_1^{\lambda_{IR}} }{ \langle L_1^{\lambda_g} \rangle \epsilon_1^{\lambda_g} } \Delta OD_{\lambda_g}(t) }{ \langle L_2^{\lambda_{IR}} \rangle \epsilon_2^{\lambda_{IR}} }
$$

그리고 실제 구현에서는 scaling coefficient $\alpha$와 capillary removal coefficient $\beta$를 사용하여 다음과 같이 단순화한다.

$$
\Delta C_2(t) = \frac{ \Delta OD_{\lambda_{IR}}(t) - \beta \cdot \Delta OD_{\lambda_g}(t) }{\alpha}
$$

논문은 $\alpha$를 waveform amplitude scaling coefficient로 보고, RWTT는 time interval descriptor이므로 핵심은 $\beta$를 찾는 것이라고 설명한다. $\beta$가 적절하면 $\Delta OD_{\lambda_{IR}}(t)-\beta \Delta OD_{\lambda_g}(t)$에서 capillary component가 줄어들고 arterial component가 지배적이 된다.

$\beta$를 찾기 위해 beat-to-beat duration correlation을 사용한다. 이는 $\Delta d(\beta,t)$의 beat-to-beat duration sequence와 original green PPG data의 beat-to-beat duration sequence 사이의 Pearson correlation coefficient로 정의된다. $\beta$가 증가함에 따라 capillary contribution이 줄어들고, arterial component가 지배적인 stable range가 나타난다. 논문은 이 stable range의 midpoint를 $\beta$로 선택한다.

### 3.6 Basic ABP estimation model

RWTT는 APG waveform에서 $a$와 $e$ 사이의 시간 간격으로 측정된다. RWTT는 PWV와 역관계를 가지며, PWV는 ABP와 강한 상관관계를 가진다. 논문은 기본적인 ABP estimation을 다음 식으로 표현한다.

$$
ABP = \gamma \cdot PWV + \eta
$$

여기서 $\gamma$와 $\eta$는 user-specific parameter이다. 이 식은 기본 모델이 개인별 혈관 특성에 강하게 의존한다는 점을 보여준다. 이러한 user-specific parameter를 정확히 얻으려면 calibration data가 필요하지만, Crisp-BP는 이를 줄이기 위해 user-independent neural network model을 설계한다.

### 3.7 BLSTM 기반 Hybrid Neural Network

Crisp-BP는 user-independent ABP monitoring을 위해 BLSTM-based Hybrid Neural Network, 즉 HNN을 제안한다. 입력은 beat-to-beat RWTT와 vascular characteristic feature들이다. 후보 feature 중에서 multicollinearity와 reference ABP와의 Pearson correlation을 고려하여 6가지 보조 feature를 선택한다.

논문에 제시된 supplement feature는 다음과 같다. $t_{a,b}$는 APG fiducial point $a$와 $b$ 사이의 time interval이고, $t_{b,a+1}$은 $b$와 다음 beat의 $a$ 사이의 time interval이다. HR은 heart rate이다. $A_{e,a+1}/A_{a,e}$는 PPG pulse 아래에서 $e$부터 다음 $a$까지의 area와 $a$부터 $e$까지의 area 비율이다. $H_{a,b}$는 $a$와 $b$ 사이의 amplitude difference이고, $H_{a,e}$는 $a$와 $e$ 사이의 amplitude difference이다. 논문은 Equ. 5의 $\alpha$를 1로 설정한다고 명시한다.

이 feature들은 min-max scaling으로 normalization된 뒤 HNN에 입력된다. Hidden layer에는 두 개의 BLSTM layer가 사용된다. BLSTM은 forward LSTM과 backward LSTM을 함께 사용하여 feature sequence의 과거와 미래 방향 정보를 모두 반영한다. 논문은 두 개의 BLSTM layer가 complexity와 accuracy 사이의 최적 균형을 제공한다고 보고한다.

개인 생리 요인의 영향을 반영하기 위해 BLSTM 이후 personal information calibration layer를 추가한다. 개인 정보 vector는 원문에서 $PI=[gender, age]^T$로 정의된다. $i$번째 timestep에서 이 layer의 출력은 다음과 같다.

$$
O_s^i = W_P \cdot PI + O_{BLSTM}^i
$$

여기서 $W_P$는 weight이고, $O_{BLSTM}^i$는 BLSTM의 출력이다. 논문은 training dataset에서 gender와 age를 사용할 수 있어 이를 사용했지만, weight, height, BMI, cardiac output 등의 biometric information도 이용 가능하다면 $PI$에 통합할 수 있다고 설명한다.

Output layer에는 두 개의 fully connected layer가 있으며, 최종 출력은 DBP와 SBP이다.

$$
y = W_{FC} \cdot O_s^i + b_{FC}
$$

여기서 $y=[DBP, SBP]^T$이고, $W_{FC}$와 $b_{FC}$는 각각 fully connected layer의 weight와 bias이다. 네트워크는 predicted ABP와 reference ABP 사이의 차이를 최소화하도록 학습된다.

### 3.8 Transfer learning과 model boosting

User-independent model은 다양한 age와 gender를 포함하는 충분한 training data가 필요하다. 하지만 실제 target device 환경에서 이를 대규모로 수집하기는 어렵다. 이를 보완하기 위해 Crisp-BP는 source domain의 online PPG data에서 general knowledge를 학습하고 target domain으로 전이하는 transfer learning을 사용한다.

Source domain으로는 MIMIC III database가 사용된다. MIMIC III는 ICU 환자의 physiologic signal과 vital sign time series를 포함하며, 원문은 30,000명 이상의 환자와 67,830개의 record set을 포함한다고 설명한다. Crisp-BP는 이 중 high-quality fingertip PPG와 ABP를 가진 subset을 사용하여 hybrid neural network를 사전 학습한다.

Transfer learning을 위해 원래 HNN 구조에 adaptation layer를 추가한다. 먼저 source domain data로 model을 학습한다. 이후 target domain인 Crisp-BP domain의 새로운 network를 source model의 weight와 bias로 초기화한다. 그 다음 hidden layer와 personal information calibration layer를 freeze하고, output layer의 fully connected layer weight를 fine-tuning하여 Crisp-BP domain에 맞춘다.

Adaptation layer의 핵심은 domain loss, 즉 DL이다. 논문은 DL을 다음과 같이 정의한다.

$$
DL = \sum MAE_h \cdot \theta_h
$$

여기서 $\theta_h$는 $h$번째 adaptation layer의 weight이며 softmax layer의 출력이고, $MAE_h$는 MIMIC III feature series와 Crisp-BP feature set 사이의 mean absolute error를 의미한다.

Crisp-BP의 전체 loss는 domain loss와 SBP, DBP의 mean squared error 합으로 구성된다.

$$
Loss =  DL + \frac{1}{n} \sum_{i=1}^{n} [ (SBP_{pi}-SBP_{li})^2 + (DBP_{pi}-DBP_{li})^2 ]
$$

여기서 $n$은 input sequence의 길이이며, $SBP_{pi}$와 $DBP_{pi}$는 predicted result, $SBP_{li}$와 $DBP_{li}$는 true ABP record이다. 이 loss는 source domain과 target domain 사이의 feature-level 차이를 줄이면서 동시에 혈압 예측 오차를 줄이도록 설계되어 있다.

## 4. 실험 및 결과

### 4.1 구현 및 데이터 수집

Crisp-BP는 commodity PPG sensor, adjustable wrist band, embedded evaluation board로 구성된 prototype으로 구현되었다. Machine learning pipeline은 TensorFlow, Keras, Sklearn을 사용했다. Neural network는 2개의 BLSTM layer, layer당 14 neuron, embedding dropout 0.2, batch size 32를 사용하며, optimizer는 Adam이다.

데이터는 연구기관의 ethical review board 승인을 받은 protocol에 따라 수집되었다. 총 35명의 참가자가 모집되었고, 남성 17명, 여성 18명이며 나이는 19세에서 50세 사이이다. 참가자는 연구기관의 학생과 faculty로 구성되었고, 평가와 관련된 known medical condition은 없다고 설명된다. 실험은 조용하고 쾌적한 온도의 laboratory에서 수행되었다.

Ground truth ABP는 FDA-approved arm-cuff ABP measurement device인 Omron U30으로 측정했다. 표준 validation procedure에 따라 참가자는 등을 지지하고 앉고, 다리를 꼬지 않으며, arm cuff를 heart level에 위치시켰다. Crisp-BP wristband와 cuff device 측정은 60초 간격으로 번갈아 수행되었고, 한 session은 약 10분이 소요되었다. 각 참가자는 네 달 동안 최소 10개의 session을 제공했으며, session 사이에는 10분 휴식을 취했다. 전체적으로 51,750분 이상의 PPG recording이 수집되었다.

### 4.2 평가 지표

논문은 mean error, standard deviation of mean error, sample Pearson correlation coefficient를 평가 지표로 사용한다. Reference ABP를 $r_i$, Crisp-BP의 predicted ABP를 $p_i$, sample 수를 $n$이라고 할 때 mean error는 다음과 같다.

$$
ME = \frac{\sum_{i=1}^{n}(p_i-r_i)}{n}
$$

Standard deviation은 다음과 같다.

$$
STD = \sqrt{ \frac{\sum_{i=1}^{n}(p_i-r_i-ME)^2}{n} }
$$

Pearson correlation coefficient는 다음과 같다.

$$
P = \frac{ \sum_{i=1}^{n}(r_i-\bar{r})(p_i-\bar{p}) }{ \sqrt{\sum_{i=1}^{n}(r_i-\bar{r})^2} \sqrt{\sum_{i=1}^{n}(p_i-\bar{p})^2} }
$$

여기서 $\bar{r}$와 $\bar{p}$는 각각 reference ABP와 predicted ABP의 평균이다.

### 4.3 전체 성능

User-independent 성능을 평가하기 위해 leave-one-participant-out validation을 수행했다. 즉 한 참가자의 데이터를 test에 사용하고, 나머지 참가자의 데이터를 training에 사용한다. 이 설정은 새로운 사용자에게 모델을 적용할 수 있는지를 평가한다는 점에서 중요하다.

전체 결과에서 DBP의 mean error는 0.86 mmHg, SBP의 mean error는 1.67 mmHg이다. DBP와 SBP의 standard deviation은 각각 6.55 mmHg와 7.31 mmHg이다. 따라서 결과는 다음과 같이 정리된다.

| 항목 |        ME |       STD |
| ---- | --------: | --------: |
| DBP  | 0.86 mmHg | 6.55 mmHg |
| SBP  | 1.67 mmHg | 7.31 mmHg |

AAMI standard는 SBP와 DBP에 대해 평균 오차가 5 mmHg 이하이고 standard deviation이 8 mmHg 이하일 것을 요구한다. Crisp-BP는 DBP와 SBP 모두 이 기준을 충족한다.

BHS standard에서도 Crisp-BP는 Grade A를 달성했다. 누적 오차 비율은 다음과 같다.

| 항목             | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg |
| ---------------- | ------: | -------: | -------: |
| DBP              |  76.73% |   91.43% |   97.96% |
| SBP              |  65.66% |   87.17% |   96.23% |
| BHS Grade A 기준 |     60% |      85% |      95% |

SBP와 DBP 모두 Grade A 기준인 5 mmHg 이하 60%, 10 mmHg 이하 85%, 15 mmHg 이하 95%를 만족한다.

Correlation analysis에서도 reference value와 estimated value는 높은 상관을 보였다. Pearson correlation coefficient는 SBP가 0.88, DBP가 0.82이다. Bland-Altman plot에서는 95% 이상의 point가 limits of agreement 안에 들어간다고 보고되어, practical usage에서의 agreement가 양호함을 보여준다.

### 4.4 24시간 성능

Crisp-BP가 continuous monitoring에 적합한지 확인하기 위해 24-hour experiment가 수행되었다. 다만 정확한 ground truth ABP는 controlled condition에서 측정해야 하므로, 24시간 동안 beat-to-beat ground truth를 직접 얻는 것은 어렵다. 대신 참가자들은 daytime에는 30분마다, night에는 1시간마다 prototype과 arm-cuff device를 번갈아 착용했다.

24시간 결과에서 Crisp-BP는 낮 시간 동안 더 좋은 성능을 보였고, DBP와 SBP의 estimation error는 3.2 mmHg 이하에서 변동한다고 보고된다. 그러나 새벽 3시에서 5시 사이 DBP error가 상대적으로 높았다. 논문은 네 명의 참가자가 이 시간대 error에 크게 기여했으며, 수면 중 interruption과 수면 부족이 감정 상태 및 heart rate에 영향을 주었을 가능성이 있다고 설명한다. 이 사례는 future work로 남겨졌다.

### 4.5 Contact pressure estimation의 효과

Crisp-BP는 contact pressure가 ABP estimation accuracy에 큰 영향을 미친다고 보고한다. 실험에서는 contact pressure를 OCP보다 낮은 경우, OCP와 같은 경우, OCP보다 높은 경우로 나누어 ABP estimation error를 비교했다. 결과적으로 OCP와 같은 경우가 DBP와 SBP 모두에서 가장 낮은 error를 보였다. OCP보다 낮은 경우는 상대적으로 큰 error를 보였으며, 이는 PPG sensor와 skin 사이의 relative motion이 noise를 증가시켰기 때문으로 해석된다.

Contact pressure estimation 자체의 정확도를 검증하기 위해, PPG sensor 옆에 7.6 mm round head thin film pressure sensor를 부착하여 ground truth contact pressure를 측정했다. 제안된 contact pressure estimation method는 mean error 6.12 g, standard deviation 11.63 g, Pearson correlation coefficient 0.65를 보였다. 또한 contact pressure를 lower than OCP, equal to OCP, higher than OCP의 세 class로 나누었을 때 96.81%의 sample을 올바르게 분류했다. 이는 contact pressure control이 Crisp-BP의 전체 성능을 뒷받침하는 핵심 요소임을 보여준다.

### 4.6 Transfer learning의 효과

Transfer learning의 효과는 매우 크게 나타났다. Transfer learning을 사용하지 않은 경우와 사용한 경우의 성능은 다음과 같다.

| 설정                      |    DBP ME |    DBP STD |    SBP ME |    SBP STD |
| ------------------------- | --------: | ---------: | --------: | ---------: |
| Without Transfer Learning | 4.88 mmHg | 12.24 mmHg | 7.30 mmHg | 18.59 mmHg |
| With Transfer Learning    | 0.86 mmHg |  6.55 mmHg | 1.67 mmHg |  7.31 mmHg |

Transfer learning 없이 학습한 모델은 AAMI standard의 STD 기준을 충족하지 못한다. 특히 SBP STD가 18.59 mmHg로 매우 크다. 반면 transfer learning을 적용하면 DBP와 SBP 모두에서 ME와 STD가 크게 감소하고 AAMI standard를 만족한다. 이는 target domain의 limited data만으로 user-independent model을 학습하기 어렵고, MIMIC III와 같은 source domain으로부터 general knowledge를 가져오는 것이 성능 향상에 중요하다는 것을 의미한다.

### 4.7 Sensor diversity와 sampling frequency

Crisp-BP는 세 가지 commodity PPG sensor, 즉 MAX30105, MAX30101, AS7026GG로 수집한 데이터를 사용하여 sensor diversity의 영향을 평가했다. 한 device에서 수집한 데이터를 test로 사용하고 나머지 device 데이터를 training에 사용하는 방식과 mixed data 기반 cross-validation을 수행했다. 결과적으로 sensor diversity는 system performance에 약한 영향만 주었다. 논문은 sensor 차이가 주로 amplitude와 area-related vascular feature에 영향을 주지만, time interval-related feature와 time-resolved RWTT는 상대적으로 높은 품질을 유지한다고 해석한다.

Sampling frequency의 영향도 평가했다. Frequency를 50 Hz에서 200 Hz까지 변화시켰을 때 sampling frequency가 증가할수록 error가 감소했다. 그러나 off-the-shelf smartwatch와 fitness tracker도 LED를 초당 수백 번 flash할 수 있으므로, 실험상 받아들일 수 있는 결과를 얻을 수 있다고 설명한다. 이 결과는 Crisp-BP가 다양한 sampling frequency를 지원하는 device와 호환될 가능성을 보여준다.

### 4.8 관련 연구와의 비교

논문은 Glabella, SeismoWatch, Seismo, Naptics, eBP 등 portable ABP monitoring system과 Crisp-BP를 비교한다. 기존 방법들은 대부분 PTT 또는 CPA를 사용하며, glasses, wristband, phone, shorts, in-ear device 등 다양한 device type을 가진다. 그러나 일부는 accelerometer, electrode, oximeter, balloon 등의 auxiliary tool을 요구하거나, 낮 또는 밤과 같은 제한된 scenario에서만 동작한다.

Crisp-BP는 단일 wristband PPG sensor만 사용하며, passive sensing이 가능하고 daytime과 night 모두에 적용할 수 있다고 주장한다. 또한 comfort level이 높고, SBP와 DBP에서 AAMI 기준을 만족하는 정확도를 보인다. 논문 관점에서는 Crisp-BP가 continuousness, comfort, convenience, accuracy, user-independence를 동시에 제공하는 첫 시스템으로 제시된다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 손목의 단일 PPG 센서만으로 continuous ABP monitoring을 시도한다는 점이다. ECG pad, 여러 센서, cuff inflation, camera setup, in-ear balloon 등 추가 장치를 요구하지 않고, smartwatch 또는 fitness tracker 형태의 device에 적용 가능한 구조를 목표로 한다. 이는 실제 사용자 경험과 장기 모니터링 가능성 측면에서 큰 장점이다.

두 번째 강점은 생리학적 직관과 signal processing을 결합한 점이다. 단순히 deep learning model에 raw PPG를 넣는 방식이 아니라, RWTT라는 혈압 관련 시간 특징을 정의하고, APG를 사용해 fiducial point를 찾으며, modified Beer-Lambert law를 통해 capillary pulse를 제거하고 arterial pulse를 추출한다. 이러한 구조는 모델이 black-box feature만 학습하는 것이 아니라 혈류역학적 원리에 기반한 feature를 활용하게 한다.

세 번째 강점은 contact pressure 문제를 명시적으로 다룬 점이다. PPG 기반 혈압 추정에서 contact pressure는 매우 중요한 confounder이다. 센서 압력이 다르면 waveform amplitude와 morphology가 바뀌고, 이는 혈압 변화로 잘못 해석될 수 있다. Crisp-BP는 LSSVM 기반 contact pressure estimation과 OCP calibration을 통해 이 문제를 완화하려고 한다.

네 번째 강점은 user-independent evaluation을 수행했다는 점이다. Leave-one-participant-out validation은 새로운 사용자에 대한 일반화 능력을 평가하는 데 적합한 설정이다. 또한 transfer learning 전후 비교를 통해 MIMIC III 기반 source training이 target domain 성능을 크게 개선함을 보여주었다.

다섯 번째 강점은 AAMI와 BHS 기준을 모두 만족하는 정량 결과를 보고했다는 점이다. DBP는 $0.86 \pm 6.55$ mmHg, SBP는 $1.67 \pm 7.31$ mmHg의 error를 보였고, BHS Grade A도 달성했다. 또한 24-hour monitoring, sensor diversity, sampling frequency, contact pressure, transfer learning 등 다양한 조건에서 system robustness를 평가했다.

그러나 한계도 분명하다. 첫째, 참가자 수가 35명으로 제한적이며, 나이는 19세에서 50세 사이이고 known medical condition이 없는 학생과 faculty 중심이다. 실제 고혈압 환자, 고령자, 심혈관 질환자, 혈관 stiffness가 큰 집단, 다양한 skin tone 및 BMI를 포함하지 않았기 때문에 임상적으로 넓은 population에 대한 일반화는 아직 충분히 입증되지 않았다.

둘째, ground truth는 invasive ABP가 아니라 FDA-approved arm-cuff device이다. Cuff device는 일반적인 validation에는 유용하지만 beat-to-beat continuous ABP waveform의 gold standard는 아니다. Crisp-BP가 continuous ABP monitoring을 주장하지만, ground truth는 60초 간격 또는 24시간 실험에서는 30분/1시간 간격으로 얻어진 cuff measurement이다. 따라서 beat-to-beat continuous accuracy는 직접적으로 검증되지 않았다.

셋째, 24시간 실험도 엄밀한 의미의 continuous ground truth와 비교한 것은 아니다. 참가자가 prototype과 cuff device를 번갈아 착용했으며, 밤에는 1시간마다 ground truth를 측정했다. 따라서 24시간 동안 발생할 수 있는 급격한 혈압 변화, 운동, 자세 변화, 수면 중 움직임 등에 대해 연속적으로 정확히 추적했는지는 제한적으로만 확인된다.

넷째, wrist position의 영향을 시스템 내부에서 자동 보정하지 못한다. 논문은 손목과 심장의 상대 위치가 blood flow velocity와 RWTT에 영향을 줄 수 있다고 설명하고, 사용자가 손목을 heart level에 가깝게 두어야 한다고 제안한다. 이는 실제 daily life 환경에서 중요한 제약이다. 사용자가 걷거나 팔을 움직이거나 누워 있는 상황에서는 wrist position이 계속 변할 수 있다.

다섯째, skin tone과 ambient light의 영향은 실험적으로 충분히 검증되지 않고 discussion에서 주로 언급된다. 논문은 darker skin이 green light를 더 많이 흡수하여 arterial pulse extraction을 어렵게 할 수 있고, 향후 yellow light와 infrared light 조합이 도움이 될 수 있다고 제안한다. 그러나 Crisp-BP prototype이 다양한 피부색에서 동일하게 동작하는지는 제공된 텍스트만으로 확인할 수 없다.

여섯째, transfer learning에 사용한 MIMIC III source domain은 fingertip PPG이고, target domain은 wrist two-wavelength PPG이다. 두 domain의 sensor position, wavelength, population, acquisition condition이 다르기 때문에 domain shift가 존재한다. 논문은 adaptation layer와 domain loss를 사용하지만, source-target domain mismatch가 어떤 feature에 어떻게 영향을 주는지에 대한 분석은 제한적이다.

일곱째, contact pressure calibration은 사용자가 매번 smartwatch를 착용할 때 기기를 누르고 release하는 절차를 요구한다. 논문은 사용자 부담이 적다고 암시하지만, “no behavior changes”라는 abstract의 주장과 비교하면 초기 calibration은 일정한 user action을 필요로 한다. 또한 사용 중 contact pressure가 땀, 움직임, band 위치 변화로 바뀔 때 얼마나 자주 재보정해야 하는지는 명확하지 않다.

종합적으로 이 논문은 wrist PPG 기반 continuous BP monitoring에서 매우 설계가 풍부한 시스템을 제안하지만, 실제 임상 적용을 위해서는 더 큰 규모의 diverse cohort, hypertensive population, invasive reference 또는 beat-to-beat reference, free-living condition에서의 장기 검증이 필요하다.

## 6. 결론

Crisp-BP는 손목 착용형 기기의 단일 PPG 센서를 이용하여 continuous, comfortable, convenient, accurate, user-independent ABP monitoring을 목표로 한 시스템이다. 핵심은 APG에서 RWTT를 추출하고, modified Beer-Lambert law 기반으로 capillary interference를 제거하며, LSSVM으로 contact pressure를 추정하고, BLSTM-based hybrid neural network와 transfer learning을 통해 DBP와 SBP를 추정하는 것이다.

실험 결과 Crisp-BP는 leave-one-participant-out validation에서 DBP $0.86 \pm 6.55$ mmHg, SBP $1.67 \pm 7.31$ mmHg의 추정 오차를 달성했다. 이는 AAMI standard의 허용 범위인 평균 오차 5 mmHg 이하, standard deviation 8 mmHg 이하를 만족한다. 또한 BHS standard에서도 DBP와 SBP 모두 Grade A를 달성했다. Transfer learning을 적용하지 않았을 때보다 적용했을 때 성능이 크게 향상되었으며, contact pressure control도 오차 감소에 중요한 역할을 하는 것으로 나타났다.

이 연구의 주요 기여는 손목 기반 PPG 센서만으로 RWTT를 측정하고, 이를 user-independent blood pressure estimation에 활용했다는 점이다. 특히 capillary pulse 제거, contact pressure estimation, BLSTM 기반 HNN, transfer learning을 통합한 end-to-end system design은 wearable healthcare 연구에서 중요한 참고점이 된다.

다만 논문의 결과는 35명의 비교적 건강한 성인 참가자와 cuff-based reference를 기반으로 하므로, 임상적 적용 가능성을 확정하기에는 추가 검증이 필요하다. 특히 고혈압 환자, 고령자, 다양한 피부색과 체형, 실제 일상 활동 중 움직임과 자세 변화, continuous invasive ABP reference와의 비교가 향후 연구에서 중요하다. 그럼에도 Crisp-BP는 wrist-worn device를 이용한 cuffless continuous blood pressure monitoring의 가능성을 구체적인 시스템과 실험으로 보여준 의미 있는 연구로 평가할 수 있다.
