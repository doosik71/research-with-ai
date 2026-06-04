# Advancing PPG-Based Continuous Blood Pressure Monitoring from a Generative Perspective

* **저자**: Hui Ji, Pengfei Zhou
* **발표연도**: 2024

## 1. 논문 개요

이 논문은 PPG(Photoplethysmography) 신호만을 사용하여 ECG(Electrocardiogram) 수준의 혈압 추정 성능을 달성하려는 연구이다. 기존의 cuff 기반 혈압 측정은 비교적 정확하지만 간헐적이고 불편하며, ECG 기반 cuffless 혈압 추정은 심장 전기 활동에 대한 풍부한 정보를 제공하지만 straps와 patches를 착용해야 하므로 일상적 연속 모니터링에는 불편하다. 반면 PPG는 손목형 웨어러블이나 스마트밴드에 쉽게 통합될 수 있고 사용자 경험이 좋지만, ECG에 비해 혈압과 관련된 심장 전기 활동 정보가 부족하여 정확도가 낮은 문제가 있다.

논문의 핵심 질문은 “저비용이고 편리한 PPG만으로 ECG를 사용한 것과 유사한 수준의 continuous blood pressure monitoring을 달성할 수 있는가?”이다. 저자들은 이 문제를 PPG-to-ECG generation 문제로 접근한다. 즉, PPG 신호를 조건(condition)으로 사용하여 ECG 신호를 생성하고, 원래 ECG 대신 생성된 ECG와 PPG를 함께 이용해 SBP(systolic blood pressure)와 DBP(diastolic blood pressure)를 추정한다.

연구 문제의 중요성은 명확하다. 고혈압은 cardiovascular diseases(CVDs)의 주요 위험 요인이고, 일상생활 속에서 혈압을 연속적으로 모니터링할 수 있다면 조기 진단과 예방에 도움이 된다. 그러나 cuff 기반 장치는 continuous monitoring에 부적합하고, ECG 기반 장치는 착용 편의성이 떨어진다. 따라서 PPG만으로 ECG의 정보를 보완할 수 있다면, 웨어러블 기반 혈압 모니터링에서 정확도와 사용자 편의성을 동시에 개선할 수 있다.

이 논문은 PPGG라는 PPG-conditional generative framework를 제안한다. PPGG는 diffusion model을 기반으로 하며, ECG 생성 과정에서 QRS complex, R-peak 위치, amplitude range, R-peak frequency를 명시적으로 정렬하도록 설계되었다. 또한 생성된 ECG와 원 PPG를 BiLSTM 기반 혈압 추정기에 입력하고, ECG generation loss와 BP estimation loss를 함께 학습하는 end-to-end 구조를 가진다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 PPG 자체로는 혈압 예측에 충분한 심장 전기 정보를 제공하지 못하지만, PPG를 조건으로 ECG와 유사한 신호를 생성하면 ECG의 진단적 장점을 활용할 수 있다는 것이다. 단순히 PPG에서 혈압을 직접 회귀하는 것이 아니라, 중간 표현으로 ECG를 생성하고 이를 혈압 예측에 사용하는 generative perspective가 이 논문의 가장 중요한 특징이다.

저자들은 기존 PPG-to-ECG generative model의 한계를 먼저 분석한다. 특히 SOTA 모델로 언급된 RDDM(Region-Disentangled Diffusion Model)은 ECG의 R-peak 위치를 정확히 맞추지 못하고, amplitude range가 원 ECG와 다르며, QRS waveform을 충실히 복원하지 못하는 문제가 있다고 설명한다. ECG에서 QRS complex는 ventricular depolarization을 나타내며, 심실 수축 및 혈압 변화와 밀접하게 관련된다. R-peak의 위치와 빈도는 심박 주기 및 혈압 추정에 중요한 단서가 되며, amplitude range와 QRS morphology 역시 심장 기능과 관련된 정보를 제공한다.

이 관찰을 바탕으로 논문은 세 가지 정렬 및 탐색 모듈을 제안한다. 첫째, forward process에서 QRS adaptive search module을 사용하여 ECG의 QRS 구간을 고정 window가 아니라 waveform 변화에 맞춰 동적으로 찾는다. 둘째, reverse process에서 scale alignment module을 사용하여 생성 ECG의 R-peak 위치와 amplitude range를 원 ECG와 맞춘다. 셋째, frequency alignment module을 사용하여 생성 ECG의 R-peak 개수 또는 heart rate가 원 ECG와 일치하도록 유도한다.

기존 접근 방식과의 차별점은 분명하다. 기존 PPG 기반 혈압 추정은 PPG feature extraction, PTT/PAT, CNN, LSTM, U-Net 등을 이용해 PPG에서 직접 BP를 예측하는 경우가 많았다. 반면 이 논문은 PPG-to-ECG conversion을 혈압 추정에 특화된 생성 문제로 정의하고, diffusion model의 forward/reverse process에 ECG의 생리학적으로 중요한 구조를 반영한다. 또한 ECG 생성 품질만 높이는 것이 아니라, BP estimation loss를 전체 학습 목표에 포함하여 생성된 ECG가 혈압 예측에 유용하도록 직접 최적화한다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

제안 프레임워크 PPGG는 크게 네 부분으로 구성된다. 첫 번째는 ECG에 noise를 추가하는 diffusion forward process이며, 여기에는 QRS adaptive search module이 포함된다. 두 번째는 PPG를 조건으로 ECG를 복원하는 reverse process이며, 여기에는 denoising U-Net과 alignment modules가 포함된다. 세 번째는 생성된 ECG와 원 PPG를 입력으로 사용하는 BiLSTM 기반 BP estimator이다. 네 번째는 ECG generation loss, alignment loss, BP estimation loss를 합친 end-to-end training objective이다.

학습 단계에서는 원 ECG가 forward process에 입력되어 점진적으로 noise가 추가된다. 이때 QRS 영역과 non-QRS 영역을 구분하여 선택적으로 noise를 추가한다. 이후 reverse process에서는 PPG를 condition으로 사용하여 noisy ECG로부터 ECG를 복원한다. 복원된 ECG와 원 PPG는 BiLSTM 기반 혈압 추정기로 전달되어 SBP와 DBP를 예측한다. 추론 단계에서는 원 ECG가 필요하지 않고 PPG만 입력으로 사용된다.

즉, 학습 시에는 ECG가 supervision으로 사용되지만, 실제 사용 시에는 PPG만으로 ECG를 생성하고 혈압을 추정한다.

### 3.2 Diffusion model의 기본 구조

논문은 diffusion model을 forward process와 reverse process로 설명한다. Forward process는 clean signal $x_0$에 Gaussian noise를 여러 timestep에 걸쳐 점진적으로 추가하는 과정이다. 기본 forward process는 다음과 같이 표현된다.

$$
q(x_T|x_0)=\prod_{t=1}^{T}q(x_t|x_{t-1})
$$

각 timestep에서의 transition은 다음과 같은 Gaussian 형태를 따른다.

$$
q(x_t|x_{t-1})\sim \mathcal{N}(x_t;\sqrt{1-\delta_t}x_{t-1},\delta_t I)
$$

여기서 $x_0$는 깨끗한 ECG 신호이고, $\delta_t$는 noise schedule에서 얻는 작은 양의 상수이다. 논문은 $\beta_t$ 관련 표기에서 일부 OCR 또는 수식 추출 오류가 있는 것으로 보이지만, 핵심적으로는 다음과 같이 임의 timestep $t$의 noisy sample을 표현한다.

$$
x_t=\sqrt{\bar{\beta}_t}x_0+\sqrt{1-\bar{\beta}_t}\epsilon,\quad \epsilon\sim \mathcal{N}(0,I)
$$

Reverse process는 noisy signal에서 원 신호를 복원하는 denoising 과정이다. 이 과정은 parameterized Gaussian transition으로 모델링된다.

$$
p_\theta(x_{t-1}|x_t)\sim \mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\sigma_t^2I)
$$

조건부 diffusion에서는 PPG 신호 $c$가 reverse process에 condition으로 들어간다. 기본 diffusion loss는 다음과 같이 noise prediction error를 최소화한다.

$$
L(\theta)=\left|\epsilon-\epsilon_\theta\left(\sqrt{\bar{\beta}_t}x_0+\sqrt{1-\bar{\beta}_t}\epsilon,c,t\right)\right|^2
$$

PPGG에서는 $c$가 PPG 신호이고, 생성 대상 $x_0$는 ECG 신호이다. 따라서 모델은 PPG 조건을 사용하여 ECG를 생성하도록 학습된다.

### 3.3 QRS adaptive search module

QRS adaptive search는 이 논문의 가장 중요한 설계 중 하나이다. RDDM 같은 기존 방법은 고정 길이 window를 사용하여 ROI(region of interest)를 설정하지만, 사람마다 QRS complex의 폭과 형태가 다르고 noise나 생리적 변화에 따라 QRS 구간이 달라질 수 있다. 고정 window는 이러한 변동을 반영하지 못하므로 R-peak misalignment나 QRS distortion을 유발할 수 있다.

PPGG는 Pan-Tompkins algorithm을 사용하여 ECG에서 R-peak를 검출한다. Pan-Tompkins algorithm은 ECG의 derivative, squaring, moving average integration, adaptive thresholding을 이용해 QRS complex와 R-peak를 검출하는 고전적이고 실용적인 방법이다. R-peak가 검출되면, R-peak 이전의 local minimum을 Q-peak로, 이후의 local minimum을 S-peak로 찾아 QRS 구간을 정의한다.

그 다음 ECG 신호와 같은 길이의 binary mask $m$을 만든다.

$$
m[i]=
\begin{cases}
1, & \text{if } i_q \le i \le i_s \
0, & \text{otherwise}
\end{cases}
$$

여기서 $i_q$는 Q-peak 위치이고, $i_s$는 S-peak 위치이다. $m[i]=1$인 구간은 QRS complex이며, $m[i]=0$인 구간은 non-QRS 영역이다.

PPGG는 forward process에서 QRS와 non-QRS 영역에 선택적으로 noise를 추가한다. 먼저 $0$부터 $T/2$까지의 timestep에서는 QRS 영역에 noise를 추가한다.

$$
x_t=\sqrt{\bar{\beta}_t}x_0+\sqrt{1-\bar{\beta}_t}(m\cdot \epsilon),\quad \epsilon\sim \mathcal{N}(0,I)
$$

이후 $T/2$부터 $T$까지는 non-QRS mask $\tilde{m}$을 사용하여 non-QRS 영역에 noise를 추가한다.

$$
x_t=\sqrt{\bar{\beta}*t}x*{T/2}+\sqrt{1-\bar{\beta}_t}(\tilde{m}\cdot \epsilon),\quad \epsilon\sim \mathcal{N}(0,I)
$$

이 방식은 ECG에서 혈압 예측에 중요한 QRS 영역을 명시적으로 분리하고, generation 과정에서 QRS 형태를 더 잘 보존하도록 유도한다. 저자들은 fixed window보다 adaptive search가 cardiac activity 변화에 더 잘 대응한다고 주장한다.

### 3.4 Reverse process와 QRS-aware denoising loss

Reverse process에서는 PPG condition $c$를 사용하여 noisy ECG를 denoising한다. 논문은 QRS 영역과 non-QRS 영역을 구분하는 loss $L_q$를 제안한다.

$$
L_q=\lambda_1\left|(m\cdot \epsilon)-\epsilon_\theta(x_t,c,t)\right|^2+\lambda_2\left|(x_T-x_t)-x_t^{[p]}\right|^2
$$

첫 번째 항은 QRS 영역의 noise를 정확히 예측하도록 유도한다. 즉, QRS mask가 적용된 noise $m\cdot \epsilon$와 모델이 예측한 noise 사이의 차이를 줄인다. 이 항은 QRS morphology와 fine-grained ECG detail을 보존하는 데 중요하다.

두 번째 항은 non-QRS 영역의 복원을 담당한다. $x_t^{[p]}=p_\theta(x_t,c,t)$는 모델이 복원한 non-QRS 부분으로 해석된다. 이 항은 전체 ECG 신호의 integrity를 유지하도록 돕는다.

논문은 $\lambda_1=100$, $\lambda_2=1$로 설정했다고 설명한다. 이는 QRS segment denoising을 non-QRS보다 훨씬 더 강하게 강조한다는 의미이다. ECG에서 QRS complex가 혈압 추정과 심박 타이밍에 특히 중요하다는 생리적 가정을 반영한 설계이다.

### 3.5 Scale alignment module

Scale alignment module은 생성 ECG와 원 ECG 사이의 R-peak 위치와 amplitude range를 맞추기 위한 모듈이다. 저자들은 RDDM이 R-peak misalignment와 amplitude variability 문제를 보였고, 이러한 문제가 BP estimation 성능을 저하시킨다고 관찰했다.

R-peak position alignment loss는 생성 ECG의 R-peak 위치와 true ECG의 R-peak 위치 사이의 mean absolute error로 정의된다.

$$
L_{position}=\frac{1}{N_{min}}\sum_{i=1}^{N_{min}}|p_{g_i}-p_{t_i}|
$$

여기서 $N_{min}$은 생성 ECG와 true ECG에서 검출된 R-peak 수 중 더 작은 값이고, $p_{g_i}$는 생성 ECG의 $i$번째 R-peak 위치, $p_{t_i}$는 true ECG의 $i$번째 R-peak 위치이다.

Amplitude alignment loss는 생성 ECG와 true ECG의 amplitude range 차이를 줄인다.

$$
L_{amplitude}=|E_g-E_t|
$$

여기서 $E_g$는 생성 ECG의 maximum amplitude difference이고, $E_t$는 true ECG의 해당 amplitude difference이다.

Scale alignment loss는 두 항의 합이다.

$$
L_{scale}=L_{position}+L_{amplitude}
$$

이 모듈은 ECG의 시간적 정렬과 신호 크기 정렬을 동시에 고려한다. 혈압 예측에서는 R-peak timing, heartbeat interval, ECG amplitude 변화가 모두 관련될 수 있으므로, 단순 waveform reconstruction loss보다 목적 지향적인 정렬 loss라고 볼 수 있다.

### 3.6 Frequency alignment module

Frequency alignment module은 생성 ECG와 true ECG의 heart rate 또는 R-peak frequency를 맞추기 위한 모듈이다. 논문은 noisy environment에서 R-peak 개수 자체가 달라질 수 있으며, position alignment만으로는 첫 $N_{min}$개의 R-peak만 맞출 뿐 전체 심박 빈도를 보장하지 못한다고 설명한다.

Frequency alignment loss는 생성 ECG의 R-peak 기반 heart rate $N_g$와 true ECG의 heart rate $N_t$ 사이의 절대 차이로 정의된다.

$$
L_{freq}=|N_g-N_t|
$$

따라서 전체 alignment loss는 다음과 같다.

$$
L_a=L_{scale}+L_{freq}
$$

이 모듈은 ECG 생성 결과의 temporal rhythm을 보존하는 데 기여한다. 혈압은 심박수 및 cardiac cycle dynamics와 관련되므로, R-peak frequency의 일관성은 BP estimation에 중요한 정보를 제공한다.

### 3.7 BiLSTM 기반 혈압 추정기

생성된 ECG와 원 PPG는 BP estimator에 입력된다. 저자들은 U-Net, LSTM, BiLSTM을 비교한 뒤 BiLSTM을 채택했다. BiLSTM은 forward direction과 backward direction의 LSTM을 동시에 사용하므로, 현재 시점의 BP를 예측할 때 이전 신호뿐 아니라 이후 신호 segment의 정보도 함께 활용할 수 있다.

ECG와 PPG 같은 physiological signal은 시간적으로 연속적이며, 특정 순간의 혈압은 단일 point가 아니라 주변 cardiac cycle의 패턴과 관련된다. BiLSTM은 이러한 장기 의존성과 양방향 temporal context를 포착하는 데 적합하다.

BiLSTM output은 dense layer로 전달되며, dense layer는 두 개의 neuron을 사용해 SBP와 DBP를 각각 예측한다. BP estimation loss는 mean squared error 형태로 정의된다.

$$
L_{bp}=\frac{1}{N}\sum_{i=1}^{N}(y_{real,i}-y_{pred,i})^2
$$

여기서 $y_{real,i}$는 실제 혈압값이고, $y_{pred,i}$는 모델 예측 혈압값이다.

### 3.8 전체 학습 목표

PPGG의 전체 loss는 ECG generation을 위한 QRS-aware diffusion loss, ECG 정렬을 위한 alignment loss, 혈압 예측 loss를 합친 형태이다.

$$
L_{overall}=L_q+L_a+L_{bp}
$$

이 구조의 장점은 ECG 생성과 혈압 추정을 별도로 최적화하지 않는다는 점이다. 생성된 ECG가 단순히 visually plausible한 신호가 아니라 BP estimation에 실제로 유용한 신호가 되도록 학습된다. 즉, PPGG는 “ECG reconstruction quality”와 “BP prediction accuracy”를 동시에 고려하는 end-to-end generative BP monitoring framework이다.

## 4. 실험 및 결과

### 4.1 데이터셋

논문은 두 개의 공개 데이터셋과 하나의 self-collected dataset을 사용한다.

첫 번째 공개 데이터셋은 PTT-PPG dataset이다. 이 데이터셋은 2022년에 공개되었으며 University of Sydney에서 22명의 healthy subjects를 대상으로 수집되었다. 피험자들은 sitting, stationary, walking, running 같은 활동을 수행했고, PPG, inertial data, ECG, BP, SpO2가 포함된다. 평균 나이는 28.5세이고, 남성 16명, 여성 6명이다. PPG sensor는 MAX30101이며 sampling rate는 500 Hz이다.

두 번째 공개 데이터셋은 MIMIC II dataset이다. MIMIC II는 ICU 환자의 anonymized records를 포함하며, 2001년부터 2008년 사이 여러 병원에서 수집된 PPG, ECG, BP signal을 포함한다. 논문 표에 따르면 MIMIC II는 age range가 15–101세로 넓고 평균 나이는 65.5세이다. 남성은 17,857명, 여성은 9,013명으로 제시되어 있다. PPG와 ECG는 Philips IntelliVue 또는 GE Healthcare bedside monitors에서 수집되었고 sampling rate는 125 Hz이다.

세 번째 데이터셋은 self-collected dataset이다. 이 데이터는 2명의 피험자, 즉 29세 남성 1명과 26세 여성 1명에서 수집되었다. 혈압은 Omron cuff BP monitor로 10분마다 측정했고, ECG는 Polar device, PPG는 Mi4 Smart Band로 연속 수집했다. 식사 전후 30분 구간을 중심으로 혈압 변화를 평가했다. 이 데이터셋은 대규모 검증용이라기보다는 실제 사용 가능성을 확인하기 위한 real-world case study 성격이 강하다.

### 4.2 전처리 및 학습 설정

ECG와 PPG는 모두 125 Hz로 resampling된다. ECG에는 threshold frequency 0.5 Hz의 Butterworth highpass filter가 적용되고, PPG에는 0.5–8 Hz bandpass Butterworth filter가 적용된다. Subject 간 차이를 줄이기 위해 per-subject z-score normalization을 수행한다. 이후 신호는 4초 window로 segment된다.

각 데이터셋은 training set 80%, test set 20%로 무작위 분할된다. PPGG는 2개의 NVIDIA A100 GPU에서 batch size 512로 학습되며, optimizer는 AdamW이다. 학습은 400 epochs 동안 수행된다. Diffusion variance scheduler는 linear schedule을 사용하며 $\beta$ 범위는 0.0001부터 0.2까지이다. Diffusion step $T$는 50으로 설정된다.

비교 모델에는 CycleGAN, GDCAE, DDPM, RDDM이 포함된다. CycleGAN은 PPG에서 ECG를 생성하는 cycle generative adversarial network 기반 모델이다. GDCAE는 generative deep convolutional autoencoder 계열 모델이다. DDPM은 기본 denoising diffusion probabilistic model이다. RDDM은 ECG의 temporal dynamics를 반영하기 위해 region-disentangled diffusion을 사용하는 SOTA PPG-to-ECG 모델로 소개된다.

### 4.3 평가 지표

평가 지표는 MAE와 RMSE이다. SBP와 DBP에 대해 별도로 계산된다. MAE는 평균 절대 오차로, 예측값과 실제값의 평균적인 차이를 직접 나타낸다.

$$
MAE=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|
$$

RMSE는 squared error의 평균에 square root를 취한 값이며, 큰 오차에 더 민감하다.

$$
RMSE=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}
$$

여기서 $y_i$는 실제 혈압값, $\hat{y}_i$는 예측 혈압값, $N$은 sample 수이다.

### 4.4 ECG 생성 품질의 정성적 결과

Figure 7(a)는 MIMIC II와 PTT-PPG 데이터셋에서 원 PPG, 원 ECG, DDPM 생성 ECG, RDDM 생성 ECG, PPGG 생성 ECG를 시각적으로 비교한다. 논문은 PPGG가 원 ECG와 가장 유사한 waveform을 생성한다고 설명한다. DDPM은 noise가 많고 QRS wave의 복잡한 구조와 frequency를 재현하지 못한다. RDDM은 QRS signal의 상당 부분을 보존하지만, 특히 MIMIC II에서 R-peak misalignment가 발생하고 amplitude 차이도 크다. 반면 PPGG는 temporal domain과 frequency domain 모두에서 ECG 특성을 더 잘 복원한다.

Figure 7(b)는 alignment module을 적용하기 전후의 ECG 생성 결과를 비교한다. Scale alignment와 frequency alignment가 peak restoration과 signal detail refinement에 기여한다고 보고된다. 이는 단순히 diffusion model을 적용하는 것보다 ECG의 생리학적 구조를 반영한 alignment loss가 중요하다는 주장을 뒷받침한다.

### 4.5 Generative model 비교 결과

Table 2는 PTT-PPG와 MIMIC II에서 여러 generative model을 비교한다. PTT-PPG 데이터셋에서 PPGG는 RMSESBP 3.22, RMSEDBP 2.49, MAESBP 2.16, MAEDBP 1.85를 달성했다. RDDM의 경우 RMSESBP 4.82, RMSEDBP 3.06, MAESBP 3.28, MAEDBP 2.19이므로, PPGG가 모든 지표에서 더 우수하다. DDPM, GDCAE, CycleGAN과의 차이는 더 크다.

MIMIC II 데이터셋에서도 PPGG는 RMSESBP 2.92, RMSEDBP 1.40, MAESBP 2.18, MAEDBP 1.27을 기록했다. RDDM은 RMSESBP 4.94, RMSEDBP 2.88, MAESBP 3.57, MAEDBP 2.14이다. 특히 MIMIC II에서 DBP RMSE 1.40과 MAE 1.27은 매우 낮은 수치로 보고된다. 저자들은 MIMIC II의 original signal quality가 높아 더 정확한 예측이 가능했다고 해석한다.

논문은 평균적으로 PPGG가 SOTA 모델 대비 SBP estimation MAE를 36.5%, DBP estimation MAE를 28.1% 줄였다고 주장한다. 이 결과는 제안한 QRS adaptive search와 alignment modules가 ECG 생성 품질뿐 아니라 BP estimation 성능에도 직접적인 영향을 준다는 점을 보여준다.

### 4.6 End-to-end BP estimation 모델 비교

Table 3은 MIMIC II에서 여러 end-to-end BP estimation method와 PPGG를 비교한다. PPG와 ECG를 모두 사용하는 기존 모델에는 CNN, MLR, U-Net 등이 포함된다. PPG만 사용하는 모델에는 SVM, NN, U-Net, LSTM, BiLSTM 등이 포함된다.

PPG만 사용하는 기존 모델들의 MAE는 상대적으로 높다. 예를 들어 SVM은 MAESBP 11.64, MAEDBP 7.62이고, NN은 MAESBP 13.40, MAEDBP 6.98이다. PPG-only U-Net은 MAESBP 8.39, MAEDBP 5.87이며, PPG-only BiLSTM은 MAESBP 7.85, MAEDBP 4.42이다.

PPGG는 test modality가 PPG로 표시되어 있으며, MAESBP 2.18, MAEDBP 1.27을 달성한다. 이는 PPG만 사용하는 기존 방법보다 큰 성능 향상을 보인다. PPG와 ECG를 모두 사용하는 U-Net은 MAESBP 2.12, MAEDBP 1.79로 SBP에서는 PPGG보다 약간 낮은 MAE를 보이지만, DBP에서는 PPGG가 더 낮다. 중요한 점은 PPGG는 inference 단계에서 실제 ECG를 요구하지 않고 PPG만 사용한다는 것이다.

### 4.7 BP estimator 비교

Table 4는 BP estimator로 U-Net, LSTM, BiLSTM을 비교한다. 원 PPG만 U-Net에 입력한 경우 MAESBP 3.78, MAEDBP 2.40이다. 원 PPG와 생성 ECG를 U-Net에 함께 입력하면 MAESBP 2.94, MAEDBP 1.61로 개선된다. 원 PPG와 생성 ECG를 LSTM에 입력하면 MAESBP 3.26, MAEDBP 2.05이다. 가장 좋은 결과는 원 PPG와 생성 ECG를 BiLSTM에 입력했을 때이며, MAESBP 2.18, MAEDBP 1.27을 기록한다.

이 결과는 생성 ECG 자체가 PPG-only 예측보다 유용한 추가 정보를 제공하며, BiLSTM이 ECG/PPG의 양방향 temporal dependency를 더 잘 포착한다는 점을 시사한다. 저자들은 혈압이 과거와 미래 신호 segment의 패턴 모두에 영향을 받기 때문에 BiLSTM이 적합하다고 설명한다.

### 4.8 Ablation study

Ablation study는 QRS adaptive search, scale alignment, frequency alignment의 효과를 분석한다.

QRS adaptive search의 효과를 확인하기 위해 RDDM의 fixed ROI window를 PPGG의 adaptive QRS search로 대체했다. MIMIC II에서 평균 RMSESBP는 4.94에서 4.68로, RMSEDBP는 2.88에서 2.73으로 감소했다. 이는 QRS 영역을 고정 길이로 자르는 것보다, 실제 Q-peak와 S-peak를 찾아 동적으로 설정하는 것이 더 정확하다는 것을 보여준다.

Table 5는 PTT-PPG 데이터셋에서 scale alignment(SA)와 frequency alignment(FA)를 제거한 결과를 보여준다. 전체 PPGG는 RMSESBP 3.22, RMSEDBP 2.49, MAESBP 2.16, MAEDBP 1.85이다. Scale alignment를 제거하면 RMSESBP 4.15, RMSEDBP 3.17, MAESBP 3.03, MAEDBP 2.18로 성능이 크게 떨어진다. Frequency alignment를 제거하면 RMSESBP 3.48, RMSEDBP 2.62, MAESBP 2.38, MAEDBP 1.96으로 성능이 저하된다. 두 모듈을 모두 제거하면 RMSESBP 4.79, RMSEDBP 3.39, MAESBP 3.16, MAEDBP 2.32가 되어 가장 나쁘다.

이 결과는 scale alignment가 특히 중요한 역할을 하며, frequency alignment도 noisy environment에서 추가적인 안정성을 제공한다는 점을 보여준다.

### 4.9 Cross-dataset validation

Cross-dataset validation은 모델이 특정 데이터셋의 특성에 과적합되지 않았는지 평가하기 위한 실험이다. 저자들은 MIMIC II로 학습한 모델을 PPGG-M, PTT-PPG로 학습한 모델을 PPGG-P라고 부른다. MIMIC II와 PTT-PPG에서 각각 학습에 사용되지 않은 100개 sample을 추출하고, 10개 validation submission으로 나누어 평가한다.

PPGG-M은 MIMIC II에서 평균 RMSESBP 3.43을 기록했고, unseen PTT-PPG에서는 3.92로 약간 증가했다. 이는 MIMIC II로 학습한 모델이 다른 데이터셋에도 비교적 잘 일반화됨을 의미한다. 반면 PPGG-P는 PTT-PPG에서 평균 RMSESBP 3.89를 기록했지만, unseen MIMIC II에서는 6.56으로 크게 증가했다.

저자들은 이 차이를 데이터셋의 규모와 품질 차이로 해석한다. MIMIC II는 더 크고 다양한 clinical dataset이므로, 이를 학습한 모델이 더 좋은 generalizability와 transferability를 보인다. 반면 PTT-PPG는 작고 homogeneous한 healthy subject 중심 데이터이기 때문에, 이를 학습한 모델은 MIMIC II처럼 다양한 환자 데이터를 잘 처리하지 못한다.

### 4.10 Generated ECG와 Original ECG 비교

논문은 생성 ECG가 실제 ECG에 비해 BP estimation에서 얼마나 유용한지 평가한다. MIMIC II로 학습한 PPGG를 사용해 test PPG에서 ECG를 생성하고, 원래 test ECG와 비교한다. 이후 SVM, NN, U-Net, LSTM, BiLSTM을 사용해 generated ECG와 original ECG 각각으로 SBP를 예측한다.

Figure 9에 따르면 MIMIC II에서는 generated ECG가 original ECG와 유사한 수준의 BP estimation 성능을 보인다. PTT-PPG에서는 generated ECG가 오히려 original ECG보다 더 좋은 성능을 내는 경우도 있다. 저자들은 PTT-PPG가 더 noisy한 데이터셋이므로, PPGG가 생성한 ECG가 denoising된 ECG representation처럼 작동했을 가능성을 제시한다.

이 결과는 매우 흥미롭지만 신중히 해석해야 한다. 생성 ECG가 원 ECG보다 항상 생리학적으로 더 정확하다는 의미는 아니다. BP estimation task에 최적화된 synthetic ECG representation이 특정 noisy dataset에서 original ECG보다 더 예측에 유리했을 수 있다. 즉, 생성 ECG는 “진단용 ECG”라기보다 “BP estimation에 유용한 ECG-like representation”으로 보는 것이 더 적절하다.

### 4.11 QRS에서 P-QRS-T로 ROI 확장 실험

논문은 QRS뿐 아니라 P wave와 T wave도 혈압과 관련될 수 있음을 인정한다. 이에 따라 QRS adaptive search를 P-QRS-T adaptive search로 확장하는 실험을 수행했다. MIMIC II에서 P wave와 T wave가 명확히 구분되는 200개 sequence를 training에 사용하고, 4개 sequence를 validation에 사용했다.

Table 6에 따르면 P-QRS-T를 ROI로 포함했을 때 RMSESBP가 QRS만 사용한 경우보다 더 낮아진다. 예를 들어 submission s1에서는 QRS 3.48, P-QRS-T 3.11이고, s2에서는 QRS 3.29, P-QRS-T 2.95이다. 이는 P wave와 T wave까지 포함하면 혈압 예측에 더 많은 정보를 제공할 수 있음을 시사한다.

그러나 저자들은 MIMIC II training dataset 13,312개 sequence 중 P-QRS-T waveform이 명확히 식별되는 sequence가 1.53%뿐이라고 설명한다. 따라서 P-QRS-T 전체를 안정적으로 활용하기에는 데이터 품질 문제가 크며, 이 적은 subset만으로 학습하면 과적합 위험이 높다. 실제 ECG에서는 noise, waveform overlap, tachycardia, 장비 한계, 개인차 때문에 P wave와 T wave를 명확히 구분하기 어렵다. 따라서 본 논문에서는 실용성을 고려해 QRS 중심 접근을 주요 방법으로 채택한다.

### 4.12 Real-world case study

실제 환경 검증을 위해 두 명의 healthy volunteer를 대상으로 식사 전후 혈압 변화를 측정했다. 식후 30분 내 혈압은 소화 시스템 활성화와 혈액 재분배 때문에 변할 수 있다. 실험에서는 Omron cuff BP monitor로 10분마다 혈압을 측정하고, Polar device로 ECG, Mi4 Smart Band로 PPG를 연속 수집했다.

여성 피험자는 3일 연속 lunch 전후 데이터를 수집했고, 남성 피험자는 하루 데이터를 수집했다. 모델은 두 공개 데이터셋으로 학습한 뒤, self-collected PPG와 ECG 데이터의 5%만 사용해 fine-tuning되었다.

Table 7에 따르면 여성 피험자의 3일간 MAE는 SBP 기준 Day1 4.93, Day2 4.98, Day3 4.87이고, DBP 기준 Day1 3.86, Day2 3.83, Day3 3.85이다. Figure 11에 따르면 남성 피험자의 하루 MAE는 SBP 4.91, DBP 3.83이다. 저자들은 이 성능이 ISO 81060-2 및 British standard 기준을 만족한다고 주장한다.

다만 이 real-world case study는 피험자가 2명뿐이고, 한 명은 1일, 다른 한 명은 3일만 수집되었으므로 임상 검증이라고 보기는 어렵다. 논문도 이 실험의 목적을 “실제 환경에서 continuous monitoring feasibility를 확인하는 것”으로 제한한다.

### 4.13 Model size와 overhead

PPGG는 168,245,600개의 parameter를 가지며 모델 크기는 725.3 MB이다. Inference memory usage는 약 1,442 MB이다. MacBook Pro 2023, Apple M2 Pro, 16 GB memory 환경에서 500회 inference 평균 시간은 instance당 304 ms이다. Samsung S23 Ultra에서는 200회 iteration 기준 평균 inference 시간이 1.45초이다.

이 결과는 mobile platform에서 실시간 또는 준실시간 혈압 추정 가능성을 보여준다. 그러나 모델 크기가 크고 memory usage가 높기 때문에, 실제 웨어러블 기기나 스마트밴드에서 직접 실행하려면 model compression, quantization, distillation, edge deployment optimization이 필요하다.

## 5. 강점, 한계

### 5.1 강점

이 논문의 가장 큰 강점은 PPG 기반 혈압 추정 문제를 generative modeling 관점에서 재정의했다는 점이다. 기존 연구가 PPG에서 직접 BP를 예측하거나 handcrafted feature를 사용하는 데 집중했다면, 이 논문은 PPG를 통해 ECG-like signal을 생성하고 이를 혈압 추정에 활용한다. 이는 PPG의 편의성과 ECG의 정보성을 결합하려는 독창적인 접근이다.

두 번째 강점은 ECG 생성 과정에 생리학적으로 중요한 구조를 반영했다는 점이다. QRS complex, R-peak position, amplitude range, R-peak frequency는 ECG에서 심박 타이밍과 심장 수축 정보를 담고 있으며 혈압 추정과 관련된다. PPGG는 이러한 요소를 QRS adaptive search, scale alignment, frequency alignment로 명시적으로 반영한다. 이는 단순한 black-box generative model보다 task-specific하고 physiologically guided된 설계이다.

세 번째 강점은 end-to-end training objective이다. ECG generation loss와 BP estimation loss를 함께 사용함으로써, 생성 ECG가 단순히 원 ECG와 비슷해 보이는 것을 넘어 혈압 추정에 유용하도록 학습된다. 이는 의료 신호 생성에서 중요한 관점이다. 생성된 신호의 시각적 유사성보다 downstream clinical task 성능이 더 중요할 수 있기 때문이다.

네 번째 강점은 다양한 비교 실험이다. 논문은 CycleGAN, GDCAE, DDPM, RDDM과 비교하고, end-to-end BP estimation baseline들과도 비교한다. 또한 ablation study, cross-dataset validation, generated ECG vs original ECG 비교, real-world case study까지 수행한다. 이러한 실험 구성은 제안 방법의 여러 측면을 검증하려는 시도로 볼 수 있다.

다섯 번째 강점은 cross-dataset validation을 통해 generalization 문제를 다루었다는 점이다. 특히 MIMIC II로 학습한 모델이 PTT-PPG에도 비교적 잘 작동하지만, PTT-PPG로 학습한 모델은 MIMIC II에서 성능이 크게 떨어진다는 결과는 데이터셋 규모와 다양성의 중요성을 잘 보여준다.

여섯 번째 강점은 실제 mobile device에서 inference time을 측정했다는 점이다. Samsung S23 Ultra에서 instance당 1.45초라는 수치를 보고함으로써, 완전한 제품화는 아니더라도 mobile deployment 가능성을 논의할 수 있는 근거를 제공한다.

### 5.2 한계

첫 번째 한계는 ECG 생성이 실제 임상 진단용 ECG를 대체할 수 있는지 불명확하다는 점이다. 논문은 generated ECG가 BP estimation에 매우 유용하다고 보여주지만, 이 ECG가 arrhythmia diagnosis, ischemia detection, conduction abnormality diagnosis 등 임상 ECG 판독에 사용할 수 있을 만큼 생리학적으로 정확한지는 검증하지 않는다. 따라서 generated ECG는 clinical ECG라기보다 BP estimation을 위한 ECG-like latent signal 또는 synthetic representation으로 해석하는 것이 안전하다.

두 번째 한계는 일부 성능 수치가 지나치게 낮아 검증 방식에 대한 추가 설명이 필요하다는 점이다. MIMIC II에서 PPGG의 MAEDBP 1.27, RMSEDBP 1.40은 매우 우수한 수치이다. 그러나 MIMIC 계열 데이터는 환자별 segment가 많고 유사한 신호가 train/test에 섞일 경우 성능이 과대평가될 수 있다. 논문은 random 80/20 split을 사용했다고 설명하지만, subject-level split 또는 patient-level split인지 명확하지 않다. 만약 같은 환자의 다른 segment가 train과 test에 모두 들어갔다면, 실제 new subject generalization 성능은 낮아질 수 있다.

세 번째 한계는 self-collected dataset 검증 규모가 매우 작다는 점이다. 실제 환경 실험은 2명의 피험자만 포함하며, 여성 3일, 남성 1일의 식사 전후 데이터에 국한된다. 이 실험은 feasibility demonstration으로는 의미가 있지만, long-term real-world continuous BP monitoring의 일반적 성능을 주장하기에는 부족하다.

네 번째 한계는 PPG-to-ECG 변환의 entropy reduction 문제를 강조하지만, 이를 정보이론적으로 엄밀하게 정량화하지는 않는다는 점이다. 논문은 PPG에서 ECG로 변환하는 과정이 inherent entropy reduction을 포함한다고 설명하지만, 실제 entropy measure, mutual information, uncertainty quantification 분석은 제공하지 않는다. 따라서 이 표현은 주로 직관적 설명에 가깝다.

다섯 번째 한계는 P-QRS-T 전체 waveform 활용 가능성이 제한적이라는 점이다. 논문은 P-QRS-T ROI가 QRS만 사용하는 것보다 더 좋은 결과를 낼 수 있음을 보였지만, MIMIC II에서 명확한 P-QRS-T waveform을 가진 sequence가 1.53%에 불과하다고 설명한다. 이는 ECG quality가 낮거나 noise가 많은 환경에서는 QRS 외 wave를 안정적으로 활용하기 어렵다는 것을 의미한다.

여섯 번째 한계는 모델 크기와 계산량이다. PPGG는 168M parameter, 725.3 MB 모델 크기, 1.4 GB 이상의 inference memory를 사용한다. 스마트폰에서는 1.45초 inference가 가능하다고 보고하지만, 실제 smartwatch나 low-power wearable device에서는 부담이 클 수 있다. 실시간 continuous monitoring에서는 battery consumption과 thermal issue도 중요하지만, 논문은 이 부분을 정량적으로 분석하지 않는다.

일곱 번째 한계는 혈압 reference의 다양성과 정확성 문제이다. MIMIC II의 BP data는 AOBP, NBP, PAWP 등 다양한 source에서 온다고 설명되어 있다. 이러한 reference 측정 방식 간 차이는 label noise를 만들 수 있다. 논문은 이를 충분히 분리하여 분석하지 않는다.

여덟 번째 한계는 ECG 생성 품질 평가가 주로 downstream BP estimation 성능과 시각화에 의존한다는 점이다. ECG signal quality를 평가하기 위한 DTW distance, morphology similarity, R-peak F1 score, heart rate error, spectral coherence 같은 정량 지표가 더 체계적으로 제시되었다면 생성 모델의 특성을 더 명확히 판단할 수 있었을 것이다.

### 5.3 비판적 해석

이 논문은 PPG 기반 혈압 추정 분야에서 매우 창의적인 방향을 제시한다. 특히 “PPG만으로 ECG-level BP monitoring을 달성한다”는 목표는 웨어러블 헬스케어 관점에서 매력적이다. QRS adaptive search와 alignment module은 단순히 deep generative model을 적용하는 수준을 넘어, ECG 신호의 생리학적 구조를 반영하려는 설계라는 점에서 강점이 있다.

그러나 결과 해석에는 주의가 필요하다. 생성된 ECG가 original ECG보다 PTT-PPG에서 더 좋은 BP estimation 성능을 보였다는 결과는 흥미롭지만, 이는 생성 모델이 ECG를 정확히 복원했다는 뜻이라기보다 BP estimation에 유리한 denoised 또는 task-optimized representation을 생성했다는 의미일 수 있다. 즉, PPGG의 생성 ECG는 “의학적으로 실제 ECG와 동일한 신호”라기보다는 “BP prediction을 위해 ECG의 일부 유용한 특성을 모사한 신호”로 이해하는 것이 더 적절하다.

또한 random split 기반 성능이 실제 patient-independent scenario에서 유지되는지는 명확하지 않다. 혈압 추정 모델은 개인별 혈관 특성, 피부색, sensor placement, motion artifact, age, disease state에 매우 민감하다. 실제 적용을 위해서는 subject-independent split, external validation, 장기 추적 데이터, 다양한 질환군 평가가 필요하다.

그럼에도 불구하고 이 논문은 단순 PPG-to-BP regression을 넘어, PPG-to-ECG generation과 BP estimation을 결합하는 새로운 연구 방향을 제안했다는 점에서 가치가 크다. 특히 ECG 생성 과정에서 혈압 추정에 중요한 신호 구조를 loss와 mask로 강제한 점은 후속 연구에서 확장 가능성이 높다.

## 6. 결론

이 논문은 PPG 기반 continuous blood pressure monitoring을 generative modeling 관점에서 재구성한 연구이다. 제안된 PPGG는 PPG를 condition으로 사용하여 ECG-like signal을 생성하고, 생성 ECG와 PPG를 BiLSTM 기반 estimator에 입력하여 SBP와 DBP를 추정한다. Diffusion model의 forward process에는 QRS adaptive search를 도입하여 ECG의 핵심 구간인 QRS complex를 동적으로 식별하고, reverse process에는 scale alignment와 frequency alignment를 추가하여 R-peak 위치, amplitude range, R-peak frequency를 정렬한다. 전체 모델은 ECG generation loss, alignment loss, BP estimation loss를 함께 최적화한다.

실험 결과 PPGG는 PTT-PPG와 MIMIC II 데이터셋에서 CycleGAN, GDCAE, DDPM, RDDM보다 우수한 성능을 보였다. MIMIC II에서는 MAESBP 2.18, MAEDBP 1.27을 보고했으며, PTT-PPG에서는 MAESBP 2.16, MAEDBP 1.85를 기록했다. Ablation study는 scale alignment와 frequency alignment가 성능 개선에 기여함을 보여주었고, QRS adaptive search 역시 fixed ROI window보다 더 좋은 성능을 보였다. Cross-dataset validation에서는 MIMIC II로 학습한 모델이 PTT-PPG에도 비교적 잘 일반화되는 반면, PTT-PPG로 학습한 모델은 MIMIC II에서 성능이 크게 떨어져 데이터 다양성의 중요성을 확인했다.

이 연구의 주요 기여는 PPG만을 입력으로 하면서도 ECG의 정보를 생성적으로 보완하여 혈압 추정 성능을 높인 점이다. 특히 QRS complex와 R-peak alignment를 명시적으로 반영한 diffusion model 설계는 생리학적 지식을 활용한 generative health sensing의 좋은 사례이다.

향후 연구에서는 patient-level split과 external validation을 통한 엄밀한 일반화 평가, 더 큰 real-world dataset에서의 장기 검증, generated ECG의 임상적 의미에 대한 체계적 평가, model compression을 통한 wearable deployment 최적화가 필요하다. 이러한 과제가 해결된다면 PPGG와 같은 접근은 저비용, 고편의성, 연속 혈압 모니터링을 위한 유망한 방향이 될 수 있다.
