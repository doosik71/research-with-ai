# Central Blood Pressure Estimation From Distal PPG Measurement Using Semiclassical Signal Analysis Features

* **저자**: Peihao Li, Taous-Meriem Laleg-Kirati
* **발표연도**: 2021

## 1. 논문 개요

이 논문은 손가락에서 측정한 distal photoplethysmography, 즉 PPG 신호를 이용하여 arterial blood pressure, 특히 central artery 수준의 혈압을 비침습적으로 추정하는 방법을 제안한다. 논문의 핵심 목표는 cuffless, calibration-free, continuous blood pressure estimation을 구현하는 것이며, 이를 위해 semi-classical signal analysis, 즉 SCSA를 이용하여 PPG 신호에서 새로운 feature를 추출한다.

혈압은 심혈관계 활동을 반영하는 핵심 생체 지표이다. 특히 hypertension은 cardiovascular disease의 주요 위험 인자로 알려져 있으며, 장기간 방치될 경우 뇌졸중, 심근경색, 신장 손상, 혈관 손상과 같은 심각한 결과를 초래할 수 있다. 따라서 혈압을 정확하고 지속적으로 측정하는 기술은 예방의학, 만성질환 관리, 중환자 모니터링, 웨어러블 헬스케어에서 매우 중요하다.

기존의 invasive arterial blood pressure 측정은 정확하고 연속적이지만, catheter 삽입이 필요하므로 출혈, 감염, 혈관 손상 위험이 있다. 반면 cuff-based sphygmomanometer는 널리 사용되지만 cuff inflation으로 인해 불편하고, 측정이 intermittent하므로 연속 혈압 모니터링에는 적합하지 않다. ECG와 PPG를 함께 사용하는 PTT 또는 PAT 기반 방법도 많이 연구되었지만, 두 개 이상의 sensor를 착용해야 하고 signal synchronization 문제가 발생한다.

이 논문은 이러한 한계를 줄이기 위해 PPG 단일 신호를 중심으로 혈압을 추정하는 방법을 제안한다. 특히 일반적인 PPG morphology 기반 방법은 dicrotic notch, systolic peak, inflection point 같은 waveform landmark를 정확히 찾아야 하는데, ICU 환자나 약물 영향을 받은 환자에서는 이러한 landmark가 불명확할 수 있다. 저자들은 이를 “inappropriate PPG” 문제로 보고, SCSA를 통해 PPG waveform을 Schrödinger operator의 spectrum으로 분해하면 이러한 문제를 완화할 수 있다고 주장한다.

연구 문제는 다음과 같이 요약할 수 있다. 첫째, PPG 신호만으로 SBP, DBP, MAP를 신뢰성 있게 추정할 수 있는가. 둘째, dicrotic notch가 불명확한 PPG에서도 안정적으로 feature를 추출할 수 있는가. 셋째, SCSA 기반 feature가 기존 PPG morphology feature 또는 ECG-PPG 기반 feature에 비해 얼마나 경쟁력 있는 성능을 보이는가. 넷째, 전통적 machine learning과 feed-forward neural network 중 어느 방식이 AAMI 및 BHS 기준을 더 잘 만족하는가.

## 2. 핵심 아이디어

논문의 핵심 아이디어는 PPG 신호를 단순한 시간 영역 waveform landmark의 집합으로 보지 않고, pulse-shaped signal을 Schrödinger operator의 eigenvalue와 eigenfunction으로 분해할 수 있는 신호로 해석하는 것이다. 이때 사용되는 방법이 semi-classical signal analysis, 즉 SCSA이다.

일반적인 PPG 기반 혈압 추정에서는 waveform의 systolic peak, diastolic peak, dicrotic notch, inflection point, pulse width, area 등을 feature로 사용한다. 하지만 이러한 feature는 waveform이 정상적으로 보일 때만 안정적으로 추출된다. ICU 환자의 PPG처럼 약물, 질환, 혈관 상태, sensor noise의 영향을 받은 경우에는 dicrotic notch가 보이지 않거나 peak와 valley가 불분명할 수 있다. 이 논문은 이러한 상황에서도 사용할 수 있는 feature extraction 방법으로 SCSA를 도입한다.

SCSA의 직관은 PPG waveform을 여러 개의 soliton-like component로 분해하는 것이다. 큰 eigenvalue는 빠르고 강한 peak 성분을 나타내며, 작은 eigenvalue 또는 나머지 component는 더 느린 변화나 세부적인 waveform 구조를 반영한다. 저자들은 가장 큰 eigenvalue들로 구성된 partial sum을 systolic dynamics로 보고, 나머지 component로 구성된 partial sum을 diastolic dynamics로 본다. 이렇게 하면 dicrotic notch를 직접 검출하지 않고도 PPG 신호를 systolic phase와 diastolic phase로 나눌 수 있다.

논문은 두 개의 estimation framework를 제안한다. 첫 번째 framework는 SCSA feature를 추출한 뒤 multiple linear regression, support vector machine, decision tree regression을 사용하여 SBP, DBP, MAP를 추정한다. 이 framework는 SCSA feature의 유효성을 확인하기 위한 전통적 supervised learning 접근이다. 두 번째 framework는 SCSA feature를 feed-forward neural network, 즉 FFNN에 입력하여 SBP, DBP, MAP를 동시에 추정한다. 이 방식은 특히 SBP 추정에서 standard deviation error를 줄여 AAMI와 BHS 기준을 만족시키기 위해 제안되었다.

기존 접근 방식과의 차별점은 세 가지이다. 첫째, ECG 없이 PPG만 사용하는 것을 주요 장점으로 내세운다. 둘째, PPG waveform landmark가 불명확한 경우에도 SCSA feature를 추출할 수 있다. 셋째, 단순 morphology feature가 아니라 Schrödinger spectrum에서 나온 eigenvalue와 invariant를 feature로 사용한다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

논문의 전체 절차는 signal preprocessing, SCSA feature extraction, machine learning regression, neural network regression, standard-based evaluation으로 구성된다.

데이터는 PhysioNet의 MIMIC II waveform database에서 가져왔다. 이 database에는 ICU 환자들로부터 동시에 기록된 PPG와 ABP signal이 포함되어 있다. 논문에서는 ABP signal을 reference BP value 추출에 사용하고, PPG signal을 SCSA feature extraction에 사용한다. 신호는 125 Hz sampling rate와 8-bit precision으로 기록되었다고 설명된다. ABP는 invasive 방식으로 기록되었고, PPG는 fingertip에서 기록되었다.

각 40초 signal segment에서 ABP의 minimum value를 DBP, maximum value를 SBP로 사용한다. 논문은 MAP를 다음과 같이 계산한다고 명시한다.

$$
MAP = \frac{2 \times SBP + DBP}{3}
$$

다만 일반 임상에서 자주 쓰이는 근사식은 $MAP = \frac{SBP + 2DBP}{3}$인 경우가 많다. 본 보고서는 원문에 제시된 식을 그대로 따르지만, 이 부분은 재현 연구에서 확인이 필요한 지점이다.

데이터베이스에는 707,567개의 signal segment가 포함되며, 약 8000개의 individual record에 속한다고 설명된다. Noise와 artifact의 영향을 줄이기 위해 기존 연구의 preprocessing procedure를 따른다고 되어 있으나, 본문에 구체적인 filter parameter와 artifact rejection 조건이 충분히 상세히 제시되지는 않는다.

### 3.2 SCSA의 기본 수식과 의미

SCSA는 real positive signal $y(t)$를 Schrödinger operator의 discrete spectrum을 이용해 squared eigenfunction의 합으로 표현한다. 논문에서 reconstructed signal은 다음과 같다.

$$
y_h(t) = 4h \sum_{n=1}^{N_h} \kappa_{nh}\psi_{nh}^{2}(t), \quad t \in R
$$

여기서 $\lambda_{nh}=-\kappa_{nh}^{2}$는 negative eigenvalue이고, $\psi_{nh}(t)$는 해당 eigenvalue에 대응하는 normalized eigenfunction이다. 이 eigenfunction은 다음 Schrödinger equation을 만족한다.

$$
-h^{2}\frac{d^{2}\psi(t)}{dt^{2}} - y(t)\psi(t) = \lambda \psi(t)
$$

$h$는 semi-classical constant이고, $N_h$는 negative eigenfunction의 개수이다. $h$가 작아질수록 $N_h$는 증가하고, $N_h$가 충분히 커지면 reconstructed signal $y_h(t)$는 원래 signal $y(t)$에 가까워진다. 논문에서는 PPG signal reconstruction에 적절한 $N_h$를 개별 case마다 선택하며, MIMIC II database에서는 보통 $N_h$가 5에서 12 사이일 때 PPG가 잘 재구성된다고 설명한다.

이 수식의 실질적 의미는 PPG waveform을 여러 개의 basis-like component로 분해한다는 것이다. 큰 eigenvalue에 대응하는 eigenfunction은 주요 peak profile을 설명하고, 나머지 eigenfunction은 waveform의 세부 구조나 noise-like detail을 설명한다. 따라서 SCSA feature는 단순히 peak 위치를 찾는 것이 아니라, waveform 전체의 구조를 spectrum 형태로 요약한다.

### 3.3 Systolic dynamics와 diastolic dynamics 분해

PPG waveform은 생리적으로 systolic phase와 diastolic phase로 나눌 수 있다. Systolic phase는 심장 수축과 관련되고, diastolic phase는 심장 이완 및 혈관 반사파와 관련된다. 일반적으로는 dicrotic notch를 기준으로 두 phase를 나누지만, 논문이 다루는 ICU data에서는 dicrotic notch가 명확하지 않은 경우가 많다.

저자들은 SCSA decomposition에서 가장 큰 eigenvalue 일부를 systolic dynamics로 사용하고, 나머지 eigenvalue component를 diastolic dynamics로 사용한다. $N_s$는 다음 범위에서 선택된다.

$$
N_s = 1,2,\cdots,\min(3,[N_h/2])
$$

Systolic dynamics $P_s$는 다음과 같다.

$$
P_s = 4h \sum_{n=1}^{N_s}\kappa_{nh}\psi_{nh}^{2}(t)
$$

Diastolic dynamics $P_d$는 다음과 같다.

$$
P_d = 4h \sum_{n=N_s+1}^{N_h}\kappa_{nh}\psi_{nh}^{2}(t)
$$

여기서 $P_s$는 빠른 systolic phenomenon을 설명하고, $P_d$는 느린 diastolic phenomenon을 설명한다. 이 접근의 장점은 PPG waveform의 dicrotic notch를 직접 검출하지 않아도 두 cardiovascular dynamics를 분리할 수 있다는 점이다. 따라서 dicrotic notch가 보이지 않는 inappropriate PPG에서도 feature extraction이 가능하다.

### 3.4 SCSA feature

논문에서 사용하는 SCSA feature는 크게 eigenvalue, systolic invariant, diastolic invariant, eigenvalue summation으로 구성된다.

첫째, SCSA eigenvalues는 다음과 같이 사용된다.

$$
\kappa_1,\kappa_2,\cdots,\kappa_{N_h-1},\kappa_{N_h}
$$

이 값들은 PPG waveform을 구성하는 spectral component의 크기를 나타낸다. 큰 $\kappa$는 빠르고 강한 waveform component를 반영하므로 systolic peak와 관련된 정보를 포함할 수 있다.

둘째, systolic invariant는 다음과 같다.

$$
INVS_1 = 4h \sum_{n=1}^{N_s}\kappa_{nh}
$$

$$
INVS_2 = \frac{16h}{3}\sum_{n=1}^{N_s}\kappa_{nh}^{3}
$$

$INVS_1$은 systolic component의 전체 크기를 요약하고, $INVS_2$는 큰 eigenvalue에 더 큰 비중을 주는 cubic invariant이다. 따라서 systolic pressure와 관련된 강한 peak dynamic을 반영할 수 있다.

셋째, diastolic invariant는 다음과 같다.

$$
INVD_1 = 4h \sum_{n=N_s+1}^{N_h}\kappa_{nh}
$$

$$
INVD_2 = \frac{16h}{3}\sum_{n=N_s+1}^{N_h}\kappa_{nh}^{3}
$$

이 feature들은 diastolic component의 전체 구조와 강도를 요약한다. DBP 및 MAP 추정에 유용한 정보를 제공할 수 있다.

넷째, 전체 eigenvalue summation은 다음과 같다.

$$
4h \sum_{n=1}^{N_h}\kappa_{nh}
$$

이 값은 전체 PPG spectrum의 크기 또는 waveform energy에 해당하는 요약 feature로 볼 수 있다. 논문은 이러한 SCSA feature가 PPG waveform의 systolic 및 diastolic peak component를 localization하는 데 도움을 주며, 기존 PPG feature보다 더 많은 정보를 추출한다고 주장한다.

### 3.5 비교 대상 feature

논문은 기존 연구에서 많이 사용된 PPG 및 ECG feature도 비교 대상으로 설명한다. Heart rate는 ECG의 R peak 간격 $L$과 sampling frequency를 이용해 계산한다고 되어 있다.

$$
HR = \frac{L}{freq}
$$

다만 일반적인 heart rate 계산은 시간 간격의 역수 형태로 해석되는 경우가 많기 때문에, 원문 식의 표기 또는 변수 정의에는 주의가 필요하다.

Inflection Point Area Ratio, 즉 IPAR는 PPG waveform에서 네 개의 pulse area $S_1$, $S_2$, $S_3$, $S_4$의 ratio를 이용한다. 이는 total peripheral resistance와 관련이 있다고 알려져 있다. Large Artery Stiffness Index, 즉 LASI는 inflection point와 그 직전 systolic peak 사이의 시간 간격에 반비례하는 arterial stiffness index이다. Augmentation Index, 즉 AI는 arterial wall의 wave reflection과 관련된 지표이며 다음과 같이 계산된다.

$$
AI = \frac{x}{y}
$$

PAT feature는 ECG R peak와 PPG의 특정 point 사이의 시간 거리로 계산된다. PATp는 ECG R peak와 PPG systolic peak 사이의 거리이고, PATd는 ECG R peak와 PPG diastolic peak 사이의 거리이다. 또 다른 PAT feature는 ECG R peak와 PPG first derivative의 maximum point 사이의 거리로 정의된다.

이러한 feature들은 혈압과 생리적으로 관련이 있지만, ECG와 PPG의 동시 측정이 필요하거나 PPG landmark detection이 필요하다는 단점이 있다. 논문의 SCSA feature는 이와 달리 PPG 단일 신호의 spectrum decomposition을 사용한다.

### 3.6 Machine learning regression

첫 번째 framework에서는 SCSA feature vector를 입력으로 하고 SBP, DBP, MAP를 target으로 하는 supervised regression을 수행한다. 사용된 모델은 multiple linear regression, support vector machine regression, decision tree regression이다.

Multiple linear regression은 feature와 target 사이의 선형 관계를 가정한다. 논문에서는 각 SCSA feature에 coefficient $\theta$를 대응시키고, least square 방식으로 regression error를 줄인다고 설명한다. 원문에 제시된 hypothesis는 다음과 같은 형태이다.

$$
h_{\theta}(x)=4h\theta_1\lambda_1+4h\theta_2\lambda_2+\cdots+4h\theta_n\lambda_n
+4h\theta_{n+1}\sum_{i=1}^{N_s}\lambda_i
+4h\theta_{n+2}\sum_{i=N_s+1}^{N_h}\lambda_i
+4h\theta_{n+3}\sum_{i=1}^{N_h}\lambda_i+\epsilon
$$

여기서 $\epsilon$은 bias coefficient 및 random error를 포함하는 항으로 설명된다. 원문에 목적 함수 식이 일부 불완전하게 추출되어 있지만, 설명상 squared error를 줄이는 regression으로 이해된다.

Support vector machine regression은 kernel function을 이용하여 feature와 BP target 사이의 관계를 모델링한다. SVM은 support vector 주변의 margin을 고려하고 overfitting을 완화할 수 있다는 장점이 있다. 논문에서 SVM은 세 전통적 모델 중 가장 좋은 성능을 보인다.

Decision tree regression은 feature space를 여러 subset으로 나누어 각 node 또는 leaf에서 regression output을 생성하는 방식이다. 학습이 빠르고 normalization 부담이 적지만, tree 구조에 따라 overfitting 가능성이 있다.

### 3.7 FFNN 기반 neural network estimation

두 번째 framework에서는 SCSA feature를 feed-forward neural network에 입력하여 SBP, DBP, MAP를 동시에 추정한다. 이 모델은 14개의 input neuron과 3개의 output neuron을 가진다. Output은 각각 SBP, DBP, MAP에 대응한다. Hidden layer node 수는 조정되며, 논문은 일반적으로 10개 정도를 사용한다고 설명한다.

Loss function은 원문에서 다음과 같이 정의된다.

$$
L(t)=\frac{1}{2m}\sum_{i=1}^{m}(y_i-\hat{y}_i)
$$

여기서 $m$은 sample 수, $\hat{y}$는 predicted output, $y$는 reference output이다. 다만 neural network regression에서 Levenberg-Marquardt training을 사용하는 경우 보통 squared error 기반 loss를 사용하므로, 원문 식에는 제곱 항이 누락되었을 가능성이 있다. 따라서 재현을 위해서는 실제 code 또는 저자 제공 구현을 확인해야 한다.

FFNN의 back-propagation training에는 Levenberg-Marquardt algorithm이 사용된다. Approximate Hessian은 다음과 같다.

$$
H = J^T J + \mu I
$$

여기서 $J$는 loss function의 Jacobian matrix이고, $\mu>0$는 damping parameter이다. Weight update는 다음과 같이 이루어진다.

$$
W_{k+1}=W_k-[J^TJ+\mu I]^{-1}J^Te
$$

이 방식은 small and middle scale regression problem에서 빠른 수렴을 기대할 수 있다. 논문은 큰 network architecture가 아니라 SCSA feature 기반의 비교적 작은 FFNN을 사용하여 SBP, DBP, MAP를 동시에 예측한다.

## 4. 실험 및 결과

실험은 MIMIC II waveform database를 기반으로 수행되었다. 논문은 약 8000명의 individual record를 사용했으며, 최종적으로 707,567개의 signal segment를 포함한다고 설명한다. 각 segment는 40초 길이이며, ABP signal에서 reference SBP, DBP, MAP를 얻고 PPG signal에서 SCSA feature를 추출한다.

첫 번째 실험에서는 SCSA feature를 사용하여 MLR, SVM, decision tree regression을 비교한다. 데이터는 70% training segment와 30% prediction segment로 나뉜다. 세 모델 모두 일정 수준의 추정 성능을 보였지만, SVM이 가장 좋은 결과를 냈다. 원문 설명에 따르면 SVM은 SBP 추정에서 MAE 7.44 mmHg를 달성했고, STD 역시 세 모델 중 가장 낮은 값으로 보고된다. 또한 DBP와 MAP에서도 SVM이 전반적으로 우수한 성능을 보였다고 설명된다.

SVM 기반 error histogram에서는 error가 0 주변에 분포한다. 그러나 SBP error의 분포가 DBP보다 더 넓으며, 이는 SBP 추정이 DBP보다 어렵다는 것을 보여준다. SBP는 systolic peak의 크기, arterial stiffness, reflected wave, peripheral vascular condition 등의 영향을 크게 받기 때문에 PPG만으로 안정적으로 예측하기가 더 어려울 수 있다.

AAMI 기준 평가에서 SVM 기반 method는 mean error가 거의 0에 가까워 mean error 기준은 만족한다. DBP와 MAP는 STD 8 mmHg 이하 기준을 만족하지만, SBP의 STD는 10.22 mmHg로 AAMI 기준인 8 mmHg를 초과한다. 즉 첫 번째 framework는 DBP와 MAP에는 비교적 안정적이지만 SBP에서는 기준을 완전히 만족하지 못한다.

BHS 기준에서는 SCSA feature와 SVM을 사용한 method가 DBP와 MAP 추정에서 Grade B를 달성한다고 보고된다. 특히 MAP는 15 mmHg 이하 cumulative error percentage에서 92.77%를 기록하여 Grade A 기준인 95%에 근접한다. 하지만 SBP에 대해서는 충분한 Grade A 결과를 제시하지 못한 것으로 해석된다.

기존 연구와의 비교에서는 SCSA feature가 기존 PPG morphology feature 및 PAT-only feature보다 DBP와 MAP에서 더 나은 성능을 보인다고 설명된다. ECG와 PPG를 모두 사용하는 feature set이 더 풍부한 정보를 제공할 수 있지만, 본 논문은 PPG-only 방식으로도 경쟁력 있는 성능을 보였다는 점을 강조한다. 이는 wearable 또는 mobile healthcare 환경에서 sensor complexity를 낮추는 데 중요한 장점이다.

두 번째 실험에서는 FFNN을 사용하여 SBP 추정 성능을 개선한다. FFNN은 SCSA feature를 입력으로 받아 SBP, DBP, MAP를 동시에 출력한다. 결과적으로 DBP의 mean difference와 standard deviation은 다음과 같이 보고된다.

$$
-0.0252 \pm 4.8569 \text{ mmHg}
$$

SBP의 mean difference와 standard deviation은 다음과 같다.

$$
0.0349 \pm 6.4477 \text{ mmHg}
$$

이 결과는 AAMI 기준인 mean error 5 mmHg 이하, STD 8 mmHg 이하를 모두 만족한다. 또한 BHS 기준에서도 DBP와 SBP estimation 모두 Grade A를 달성한다고 보고된다. 따라서 FFNN framework는 SVM framework의 주요 한계였던 SBP STD 문제를 해결한 것으로 제시된다.

다만 논문에서 table의 모든 수치가 텍스트로 명확히 추출되지는 않았고, 일부 결과는 본문 설명에 근거한다. 또한 train/test split이 subject-independent인지 segment-level random split인지 명확하지 않기 때문에, reported performance가 실제 unseen subject에서 그대로 유지되는지는 추가 검증이 필요하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 SCSA라는 독창적인 signal analysis 방법을 PPG 기반 혈압 추정에 적용했다는 점이다. 기존 morphology feature는 dicrotic notch나 inflection point가 정확히 검출되어야 하지만, SCSA는 waveform 전체를 Schrödinger spectrum으로 표현하기 때문에 landmark detection에 덜 의존한다. 특히 ICU 환자의 inappropriate PPG처럼 waveform이 정상적이지 않은 경우에도 feature를 추출할 수 있다는 주장은 이 논문의 중요한 기여이다.

두 번째 강점은 PPG-only 기반 접근이다. ECG와 PPG를 함께 사용하는 PAT 또는 PTT 기반 방법은 생리학적으로 유용하지만, 실제 wearable device에서는 sensor 수, 착용 위치, synchronization 문제가 발생한다. 이 논문은 finger PPG 하나만으로 SBP, DBP, MAP를 추정하려 하므로 실제 device integration 측면에서 장점이 있다.

세 번째 강점은 전통적 machine learning과 FFNN을 단계적으로 비교했다는 점이다. 먼저 MLR, SVM, decision tree regression을 통해 SCSA feature의 기본 성능을 확인하고, 이후 FFNN으로 nonlinear regression 성능을 개선한다. 이 과정은 단순히 deep learning을 적용한 것이 아니라, feature extraction method와 regression model의 역할을 비교적 명확히 구분한다.

네 번째 강점은 AAMI와 BHS라는 표준 기준을 사용하여 결과를 평가했다는 점이다. 단순히 MAE만 제시하는 것이 아니라 혈압 측정 장치 평가에서 자주 언급되는 기준과 비교했다는 점은 논문의 실용적 가치를 높인다.

그러나 한계도 명확하다. 첫째, MIMIC II database는 ICU 환자 중심 데이터이다. ICU 환자는 약물, 질환, 중증도, 처치 상태의 영향을 받기 때문에 일반 건강인이나 일상적 wearable 환경과는 다르다. 저자도 향후 healthy subject data를 추가하여 검증해야 한다고 언급한다.

둘째, preprocessing의 세부 정보가 부족하다. 논문은 기존 연구 [25]의 preprocessing을 따른다고 설명하지만, filter 조건, artifact rejection 기준, segment exclusion 기준이 충분히 구체적으로 제시되지 않는다. PPG 기반 BP estimation은 preprocessing에 매우 민감하므로 재현성 측면에서 아쉬움이 있다.

셋째, subject-independent split 여부가 불명확하다. 논문은 많은 subject와 segment를 사용했다고 설명하지만, 동일 subject의 segment가 training과 test에 동시에 포함되었는지 여부가 명확하지 않다. 만약 segment-level random split이 사용되었다면, 모델이 subject-specific waveform pattern을 학습하여 성능이 과대평가되었을 가능성이 있다. 혈압 추정 연구에서는 subject-level split이 매우 중요하다.

넷째, 일부 수식 표기에 의문이 있다. MAP 계산식은 일반적으로 널리 쓰이는 근사식과 다르게 제시되어 있으며, FFNN loss function에는 squared error의 제곱 항이 누락된 것처럼 보인다. 이는 논문의 구현 세부사항을 확인해야 하는 부분이다.

다섯째, central blood pressure estimation이라는 제목과 reference ABP 위치에 대한 해석이 엄밀히 검증되어야 한다. 원문은 ABP가 aorta에서 invasively recorded되었다고 설명하지만, MIMIC II의 ABP recording site가 모든 subject에서 동일한 central artery인지 본문만으로 완전히 확인하기 어렵다.

여섯째, SCSA parameter selection이 실제 device 환경에서 얼마나 자동적이고 안정적으로 수행되는지 명확하지 않다. 논문에서는 $N_h$가 5에서 12 사이에서 적절한 reconstruction을 보인다고 설명하지만, noise가 많은 real-world PPG나 motion artifact가 많은 wearable PPG에서 동일하게 동작할지는 추가 실험이 필요하다.

## 6. 결론

이 논문은 distal fingertip PPG를 이용하여 arterial blood pressure를 비침습적, cuffless, calibration-free 방식으로 추정하는 SCSA 기반 framework를 제안한다. 논문의 핵심 기여는 PPG waveform을 Schrödinger operator의 eigenspectrum으로 분해하고, 그로부터 eigenvalue, systolic invariant, diastolic invariant, eigenvalue summation을 feature로 추출했다는 점이다.

첫 번째 framework에서는 SCSA feature와 MLR, SVM, decision tree regression을 비교했고, SVM이 가장 좋은 성능을 보였다. 이 방식은 DBP와 MAP 추정에서 좋은 결과를 보였지만, SBP의 STD가 AAMI 기준을 초과하는 한계가 있었다. 두 번째 framework에서는 SCSA feature를 FFNN에 입력하여 SBP, DBP, MAP를 동시에 추정했고, DBP에서 $-0.0252 \pm 4.8569$ mmHg, SBP에서 $0.0349 \pm 6.4477$ mmHg의 결과를 얻어 AAMI 기준을 만족했다. 또한 BHS 기준에서도 Grade A를 달성했다고 보고한다.

이 연구는 PPG-only blood pressure estimation에서 기존 landmark 기반 feature extraction의 약점을 보완하려는 의미 있는 시도이다. 특히 dicrotic notch가 검출되지 않는 inappropriate PPG에서도 SCSA feature를 추출할 수 있다는 점은 임상 데이터와 wearable signal 모두에서 중요한 가능성을 가진다.

다만 실제 임상 적용을 위해서는 subject-independent validation, healthy population 및 다양한 BP category에 대한 검증, real-world wearable noise에 대한 robustness 분석, preprocessing 및 parameter selection의 재현성 확보가 필요하다. 종합적으로 이 논문은 SCSA라는 수학적 signal decomposition을 PPG 기반 혈압 추정에 도입한 독창적 연구이며, 향후 cuffless continuous BP monitoring device 개발에 활용될 수 있는 유망한 방향을 제시한다.
