# PPG-Based Blood Pressure Estimation Can Benefit From Scalable Multi-Scale Fusion Neural Networks and Multi-Task Learning

* **저자**: Qihan Hu, Daomiao Wang, Cuiwei Yang
* **발표연도**: 2022

## 1. 논문 개요

이 논문은 photoplethysmography, 즉 PPG 신호만을 이용하여 cuffless continuous blood pressure estimation을 수행하는 end-to-end deep learning 모델인 MSF-MTLNet을 제안한다. 논문의 핵심 목표는 8초 길이의 PPG segment와 그 derivative인 VPG, APG를 입력으로 사용하여 systolic blood pressure, 즉 SBP와 diastolic blood pressure, 즉 DBP를 동시에 추정하는 것이다. 저자들은 PPG 기반 혈압 추정에서 multi-scale feature fusion과 multi-task learning이 성능 향상에 기여할 수 있음을 실험적으로 보이고자 한다.

연구 문제는 “단일 PPG 신호에서 서로 다른 시간 scale의 waveform 정보를 adaptive하게 융합하고, SBP와 DBP라는 관련된 두 regression task를 함께 학습하면 혈압 추정 정확도를 높일 수 있는가”로 요약할 수 있다. 기존의 cuff 기반 혈압계는 비교적 신뢰할 수 있는 혈압 측정을 제공하지만, cuff 팽창과 압박이 필요하므로 장시간 연속 측정에는 적합하지 않다. 특히 hypertension은 cardiovascular diseases, 즉 CVD의 중요한 위험 요인이므로, 혈압을 연속적이고 편리하게 측정할 수 있는 기술은 조기 진단과 예방 측면에서 중요하다.

기존 PPG 또는 ECG 기반 non-invasive BP estimation 방법은 크게 feature extraction과 regressor의 두 단계로 구성되는 경우가 많았다. 전통적인 방법은 pulse transit time, pulse arrival time, PPG intensity ratio, heart rate, PPG morphology feature 등 전문가 지식에 기반한 handcrafted feature를 추출한 뒤 machine learning 또는 shallow neural network에 입력한다. 이러한 방식은 생리학적 해석 가능성이 있지만, fiducial point detection에 의존하고, waveform delineation이 복잡하며, 원신호에 포함된 잠재 정보를 놓칠 수 있다.

최근 end-to-end deep learning 방법은 raw physiological signal에서 feature를 자동으로 학습할 수 있다는 장점 때문에 PPG 기반 BP estimation에 널리 적용되고 있다. 그러나 저자들은 기존 end-to-end CNN 또는 CNN-RNN 기반 연구들이 주로 single-scale convolution kernel을 사용하여 feature를 추출했기 때문에, PPG waveform에 존재하는 다양한 scale의 정보를 충분히 활용하지 못했다고 지적한다. 작은 kernel은 forward wave나 dicrotic notch 같은 local waveform feature를 포착하는 데 유리하고, 큰 receptive field는 pulse interval이나 morphology 간 관계 같은 더 넓은 temporal structure를 포착하는 데 유리하다. 따라서 multi-scale feature extraction과 adaptive fusion이 필요하다는 것이 논문의 출발점이다.

또한 기존 end-to-end network는 SBP와 DBP를 동시에 출력하더라도 두 task의 차이를 명시적으로 반영하지 않는 경우가 많았다. SBP와 DBP는 같은 PPG signal에서 추정되지만, 각 task에 중요한 feature의 종류와 중요도는 다를 수 있다. 저자들은 이러한 공통성과 차이를 활용하기 위해 hard parameter sharing 기반 multi-task learning, 즉 MTL을 도입한다. Shared backbone은 공통된 multi-scale PPG representation을 학습하고, task-specific subnetwork는 SBP와 DBP 각각에 적합한 feature를 정제한다.

실험에는 UCI Machine Learning Repository dataset이 사용되었다. 이 데이터는 MIMIC database에서 파생된 것으로, ICU 환자의 ECG lead II, fingertip PPG, invasive arterial blood pressure, 즉 ABP를 포함한다. 최종적으로 1,825명의 환자로부터 277,600개의 8초 segment가 사용되었고, training, validation, test set은 subject가 겹치지 않도록 inter-subject manner로 8:1:1 비율로 분할되었다. 이는 동일 환자의 segment가 학습과 평가에 동시에 포함되어 성능이 과대평가되는 문제를 줄이는 중요한 설계이다.

논문의 주요 결과는 universal model 기준으로 SBP estimation error가 $0.83 \pm 8.80$ mmHg, DBP estimation error가 $0.43 \pm 4.13$ mmHg였다는 것이다. 초록에는 SBP와 DBP error가 각각 $0.97 \pm 8.87$ mmHg, $0.55 \pm 4.23$ mmHg로 제시되는데, 본문 Table 7에서는 최종 검증 수치가 $0.83 \pm 8.80$ mmHg, $0.43 \pm 4.13$ mmHg로 제시된다. 두 수치 모두 같은 결론을 지지한다. DBP는 AAMI standard를 만족하고 BHS Grade A를 달성했으며, SBP는 standard deviation이 AAMI 기준인 8 mmHg를 약간 초과하여 AAMI를 완전히 만족하지는 못하지만 BHS Grade B를 달성하였다.

## 2. 핵심 아이디어

이 논문의 첫 번째 핵심 아이디어는 PPG 기반 혈압 추정에서 multi-scale information이 중요하다는 것이다. PPG waveform은 단순한 주기 신호가 아니라 cardiac ejection, vascular compliance, peripheral resistance, reflected wave 등의 영향을 받은 복합적인 생리 신호이다. 짧은 시간 scale에서는 systolic upstroke, forward wave, dicrotic notch, diastolic wave 같은 국소 형태가 중요하고, 긴 시간 scale에서는 pulse interval, waveform component 간 상대적 위치, 반복되는 pulse pattern 등이 중요할 수 있다. 따라서 하나의 fixed convolution kernel만으로는 BP-related information을 충분히 표현하기 어렵다.

이를 해결하기 위해 저자들은 scalable Multi-Scale Fusion block, 즉 MSF block을 설계한다. MSF block은 MobileNet의 depthwise separable convolution과 SKNet의 selective kernel idea를 참고한 구조이다. 입력 feature를 pointwise convolution으로 확장한 뒤, 서로 다른 receptive field를 갖는 depthwise convolution branch로 multi-scale feature를 추출하고, attention mechanism으로 각 scale의 중요도를 adaptive하게 조정한다. 이 구조는 단순히 여러 convolution branch를 concatenate하는 것보다 유연하다. 특정 입력 구간이나 channel에서는 local morphology가 중요할 수 있고, 다른 channel에서는 longer temporal pattern이 중요할 수 있기 때문이다.

두 번째 핵심 아이디어는 PPG 원신호뿐 아니라 VPG와 APG를 함께 입력으로 사용하는 것이다. VPG는 PPG의 1차 difference이며 waveform의 변화율을 반영하고, APG는 2차 difference로 waveform의 곡률과 acceleration 정보를 반영한다. PPG derivative는 cardiovascular system의 subtle morphology change를 드러낼 수 있으므로, handcrafted feature를 직접 계산하지 않더라도 network가 derivative signal에서 유용한 feature를 학습할 수 있다. 최종 입력은 PPG, VPG, APG로 구성된 3-channel 8초 segment이다.

세 번째 핵심 아이디어는 SBP estimation과 DBP estimation을 multi-task learning 구조로 학습하는 것이다. Conventional machine learning에서는 DBP와 SBP를 별도의 모델로 학습하는 경우가 많았고, 일반적인 end-to-end deep learning에서는 동일한 backbone이 두 값을 동시에 출력하지만 task-specific difference를 충분히 반영하지 못하는 경우가 있었다. MSF-MTLNet은 shared MSF backbone 뒤에 SBP-specific subnetwork와 DBP-specific subnetwork를 따로 둔다. 각 subnetwork에는 bottleneck attention module, 즉 BAM이 포함되어 있으며, shared feature 중 각 task에 더 중요한 channel과 temporal location을 강조한다.

네 번째 핵심 아이디어는 모델의 성능뿐 아니라 계산 효율성도 고려한다는 점이다. Multi-scale convolution은 성능을 높일 수 있지만 계산량을 크게 늘릴 위험이 있다. 저자들은 depthwise separable convolution과 scale factor $\alpha$를 사용하여 parameter 수와 MAdds를 조절할 수 있도록 설계하였다. 원래 MSF-MTLNet은 970,940개 parameter와 10.45M MAdds를 사용하며, 단일 inference 시간은 local workstation에서 약 0.34 ms로 보고된다. 저자들은 이를 근거로 mobile device에서 real-time 실행 가능성이 있다고 주장한다.

## 3. 상세 방법 설명

### 3.1 문제 정의

논문은 PPG-based BP estimation을 time-series regression problem으로 정의한다. 데이터셋은 다음과 같이 표현된다.

$$
D={(X_1,Y_1^d,Y_1^s),\ldots,(X_M,Y_M^d,Y_M^s)}
$$

여기서 $M$은 sample 수, $X$는 입력 segment, $Y^d$는 reference DBP, $Y^s$는 reference SBP이다. Raw PPG signal은 difference operation을 통해 3-channel data로 확장된다. 각 segment는 다음과 같이 정의된다.

$$
X=[x_1,x_2,\ldots,x_n]
$$

$$
x_n=[x_n^{PPG},x_n^{VPG},x_n^{APG}]
$$

여기서 $n$은 segment length이다. 이 논문에서는 PPG를 50 Hz로 resampling하고 8초 window를 사용하므로, 입력 sequence length는 $50 \times 8=400$이다. 따라서 network input은 $3 \times 400$ 형태가 된다.

제안 모델은 SBP와 DBP를 각각 다음과 같이 추정한다.

$$
\hat{Y}^s=F_s(X,\theta_c,\theta_s)
$$

$$
\hat{Y}^d=F_d(X,\theta_c,\theta_d)
$$

여기서 $F_s$와 $F_d$는 각각 SBP와 DBP를 추정하는 nonlinear mapping function이다. $\theta_c$는 shared backbone의 parameter로 multi-scale feature를 추출하는 데 사용된다. $\theta_s$와 $\theta_d$는 각각 SBP-specific subnetwork와 DBP-specific subnetwork의 parameter이다. 학습 objective는 estimated value와 reference value 사이의 mean squared error, 즉 MSE를 최소화하는 것이다.

### 3.2 전체 파이프라인

전체 pipeline은 database preprocessing, 3-channel input construction, MSF backbone feature extraction, task-specific feature refinement, SBP/DBP regression으로 구성된다.

먼저 UCI dataset에서 PPG와 ABP signal을 가져온다. PPG에는 baseline wandering, powerline interference, signal loss, motion artifact 등이 포함될 수 있으므로 noise suppression과 artifact removal을 수행한다. 이후 PPG와 synchronized ABP를 8초 window로 segmentation하며, 75% overlap을 적용한다. 각 ABP segment에서 peak와 bottom을 찾아 여러 SBP와 DBP 값을 얻고, 이들의 평균을 해당 8초 PPG segment의 reference SBP와 reference DBP로 사용한다.

전처리된 PPG에서 1차 difference로 VPG를 만들고, 2차 difference로 APG를 만든다. 이 세 신호는 각각 zero-mean normalization을 거쳐 하나의 3-channel input vector를 구성한다. 이 입력은 MSF-MTLNet에 전달된다.

Backbone network는 stacked scalable MSF blocks로 구성된다. 각 MSF block은 서로 다른 scale의 feature를 depthwise convolution으로 추출하고 attention으로 융합한다. Backbone output은 두 task-specific subnetworks로 분기된다. 각 subnetwork는 BAM을 통해 task-relevant feature를 강조하고, convolution과 global average pooling을 거쳐 SBP 또는 DBP를 출력한다.

### 3.3 MSF-MTLNet architecture

논문 Table 1에 따르면 MSF-MTLNet의 입력은 $3 \times 400$이다. 초기 convolution은 $[Conv1,16,3,hswish]$로 표현되며, 이후 feature map은 $16 \times 200$이 된다. 그다음 여러 MSF block이 stacked되어 channel 수를 늘리고 temporal length를 줄인다. 구조는 대략 $16 \times 200$, $32 \times 100$, $32 \times 50$, $64 \times 25$, $128 \times 13$ 순서로 진행된다. 각 stage의 첫 번째 block은 stride 2를 사용하여 downsampling한다.

Task-specific subnetwork는 $128 \times 13$ feature map을 입력으로 받는다. DBP subnetwork와 SBP subnetwork는 같은 구조를 가지지만 서로 다른 weight를 갖는다. 각 subnetwork는 BAM, $1 \times 1$ convolution with 768 filters, global average pooling, $1 \times 1$ convolution with 128 filters, final $1 \times 1$ convolution으로 구성된다. 최종 output layer는 SBP와 DBP 두 값을 제공한다.

### 3.4 Scalable Multi-Scale Fusion block

MSF block은 입력 embedding $X \in R^{C \times W}$를 받아 multi-scale representation을 생성한다. 먼저 pointwise convolution으로 channel dimension을 확장한다.

$$
X'=\sigma(BN(f_1(X)))
$$

여기서 $f_1$은 kernel size 1의 convolution, $BN$은 batch normalization, $\sigma$는 hard-swish activation이다. Hard-swish는 다음과 같이 정의된다.

$$
\sigma(x)=x\cdot\frac{RELU6(x+3)}{6}
$$

이 expansion step은 입력을 고차원 공간으로 보내 후속 feature extraction의 표현력을 높이기 위한 것이다. 논문은 expansion factor $\beta$를 첫 번째 MSF block을 제외하고 6으로 설정한다.

이후 서로 다른 scale의 feature를 추출하기 위해 depthwise convolution branch를 사용한다. 두 scale 설정에서는 dilation value가 $[1,2]$인 Conv3 branch가 사용된다. 즉 같은 kernel size라도 dilation을 다르게 하여 receptive field를 다르게 만든다. 각 branch는 depthwise convolution, batch normalization, hard-swish activation으로 구성된다.

두 single-scale transformation은 다음과 같이 표현된다.

$$
F_1:X'\rightarrow U_1
$$

$$
F_2:X'\rightarrow U_2
$$

여기서 $U_1$과 $U_2$는 서로 다른 scale에서 추출된 feature이다. 이후 element-wise summation으로 통합 feature $U$를 만든다. 이 통합 feature에 global average pooling을 적용하여 channel-wise global information $s$를 얻고, fully connected layer를 통해 scale fusion을 guide하는 feature $z$를 만든다.

$$
z=\sigma(BN(Ws))
$$

여기서 $W$는 fully connected layer parameter이고, $\sigma$는 ReLU이다. 다음으로 softmax를 사용하여 scale별 attention weight를 계산한다.

$$
a_i=\frac{e^{A_i z}}{e^{A_i z}+e^{B_i z}}
$$

$$
b_i=\frac{e^{B_i z}}{e^{A_i z}+e^{B_i z}}
$$

여기서 $a_i$와 $b_i$는 각각 $U_1$과 $U_2$ branch의 attention weight이다. 최종 multi-scale feature map은 다음과 같이 얻는다.

$$
V_i=a_i\cdot U_i^1+b_i\cdot U_i^2
$$

즉 각 channel에서 서로 다른 scale의 feature를 attention weight로 가중합한다. 이 방식은 fixed fusion이나 단순 concatenation과 달리 입력 signal의 상태에 따라 어떤 scale을 더 중요하게 볼지 학습할 수 있다.

마지막으로 pointwise linear convolution을 사용하여 channel 정보를 결합하고 dimension을 줄인다.

$$
Y=BN(f_1(V))
$$

여기서 $Y \in R^{\alpha C'' \times W'}$이며, $\alpha$는 scale factor이다. $\alpha$는 network width를 조절하는 hyperparameter로, hardware resource에 따라 모델 크기를 줄이거나 늘릴 수 있다.

### 3.5 BAM 기반 task-specific subnetwork

각 task-specific subnetwork는 shared backbone output을 SBP 또는 DBP에 적합한 feature로 정제한다. 핵심 구성 요소는 bottleneck attention module, 즉 BAM이다. BAM은 channel attention과 spatial attention을 결합하여 informative feature를 강조하고 less useful feature를 억제한다.

Backbone output을 $F \in R^{C \times L}$이라고 할 때, channel attention은 GAP와 두 개의 fully connected layer를 통해 계산된다.

$$
M_c(F)=BN(W_1(\sigma(W_0\cdot Avg(F)+b_0))+b_1)
$$

여기서 $Avg$는 global average pooling, $\sigma$는 ReLU, $W_0$, $W_1$, $b_0$, $b_1$은 학습 가능한 parameter이다. Channel attention은 어떤 channel이 해당 task에 중요한지를 학습한다.

Spatial attention은 convolution을 통해 temporal dimension에서 중요한 위치를 찾는다.

$$
M_s(F)=BN(f_1(f_3(f_3(f_1(F)))))
$$

여기서 $f_x$는 kernel size $x$인 convolution layer이다. 이 구조는 먼저 channel reduction을 수행하고, dilated convolution으로 contextual information을 학습한 뒤, 다시 dimension을 줄인다.

최종 refined feature map은 다음과 같다.

$$
F'=F+F\times(\sigma(M_c(F)\times M_s(F)))
$$

여기서 $\sigma$는 sigmoid function이다. 이 식은 residual 방식으로 원래 feature를 유지하면서 attention-weighted feature를 추가하는 구조이다. DBP subnetwork와 SBP subnetwork는 서로 다른 BAM을 가지므로, 두 task가 같은 shared feature에서도 서로 다른 channel과 temporal location을 강조할 수 있다.

### 3.6 데이터 전처리

실험 데이터는 UCI Machine Learning Repository dataset이다. 이 dataset은 MIMIC database에서 파생되었으며, ICU 환자의 ECG lead II, fingertip PPG, invasive ABP를 포함한다. 원신호의 sampling rate는 125 Hz이다.

저자들은 subject별 기여도를 균형 있게 만들기 위해 data duration이 8분 미만인 subject를 제외하였다. 최종적으로 1,825명의 subject가 남았고, 평균 recording duration은 9.15분, standard deviation은 0.55분이다. 최종 데이터베이스의 DBP 범위는 50–127 mmHg, SBP 범위는 79–197 mmHg, heart rate 범위는 47–184 BPM이다. 평균은 DBP 70 mmHg, SBP 135 mmHg, HR 92 BPM이며, 표준편차는 각각 10, 19, 19이다.

Noise suppression에는 7-layer discrete wavelet transform, 즉 DWT가 사용되었다. Order-eight Daubechies wavelet을 적용하고, low-frequency baseline wandering에 해당하는 first-layer approximation component와 powerline interference에 해당하는 6–7번째 detail component를 zeroing한 뒤 reconstruction을 수행한다.

Segmentation에서는 PPG와 ABP를 모두 8초 window로 나누고 75% overlap을 적용한다. 이후 artifact removal을 위해 skewness를 사용한다. 각 8초 segment는 여덟 개의 1초 segment로 나뉘며, 각 1초 PPG와 ABP segment의 skewness를 계산한다. Skewness는 다음과 같이 정의된다.

$$
Skew_k=\frac{1}{N-1}\sum_{i=1}^{N}\frac{(x(i)-x^{mean})^3}{x^{std}}
$$

여기서 $Skew_k$는 $k$번째 1초 segment의 skewness, $x(i)$는 $i$번째 sample, $x^{mean}$은 평균, $x^{std}$는 표준편차이다. 어떤 1초 segment라도 skewness index가 0보다 작으면 해당 8초 segment는 제거된다. 이 quality assessment 후 약 40%의 segment가 low quality로 제거되었고, 277,600개 segment가 보존되었다.

Normalization 단계에서는 PPG를 50 Hz로 resampling한다. 논문은 PPG spectrum이 0.02–25 Hz에 집중되어 있으므로 50 Hz resampling이 계산량 절감에 적절하다고 설명한다. PPG에는 zero-mean normalization을 적용한다. VPG와 APG는 전처리된 PPG의 1차·2차 difference로 생성되며, three-point moving average로 noise를 줄인 뒤 zero-mean normalization을 적용한다.

Reference SBP와 DBP는 ABP segment에서 peak와 bottom을 탐색하여 얻는다. 한 8초 ABP segment 안에는 여러 heartbeat가 포함될 수 있으므로, 여러 SBP와 DBP 값을 얻고 이들을 평균하여 해당 segment의 reference SBP와 DBP로 사용한다.

### 3.7 Baseline model과 ablation 설계

저자들은 MSF-MTLNet의 구성 요소가 실제로 성능에 기여하는지 확인하기 위해 다양한 baseline과 ablation model을 구현했다. Ensemble ML baseline은 Kurylyak feature set 등 handcrafted feature를 사용하며, Bayesian optimization으로 hyperparameter를 조정하였다. Deep learning baseline으로는 VGGNet, ResNet, ResNet+GRU, ECNet, MSFNet-1S, MSFNet-2S, MSFNet-3S, MSF-BAM, MSF-MTL이 비교되었다.

VGGNet은 plain convolution block을 사용한 single-scale CNN baseline이다. ResNet은 residual block을 사용한 single-scale architecture이다. ResNet+GRU는 convolutional feature 뒤에 bidirectional GRU를 붙여 temporal feature를 처리하는 CNN-RNN baseline이다. ECNet은 multi-scale convolution branch를 단순 convolution 기반으로 결합하는 Extraction-Concentration block을 사용한다. MSFNet은 MSF block을 사용하지만 MTL 구조는 사용하지 않는 모델이다. MSF-BAM은 MSF backbone 뒤에 BAM을 추가한 shared subnetwork 구조이다. MSF-MTL은 두 task-specific subnetwork를 가진 최종 제안 모델이다.

또한 MSF block의 scale 수에 따른 영향을 확인하기 위해 dilation 설정 $[1]$, $[1,2]$, $[1,2,3]$을 비교하였다. 각각 MSFNet-1S, MSFNet-2S, MSFNet-3S에 해당한다. 최종 모델은 성능과 계산량의 균형을 고려하여 two-scale setting, 즉 $[1,2]$를 선택하였다.

### 3.8 학습 설정과 평가 지표

모든 deep learning model은 공정한 비교를 위해 같은 training setting을 사용하였다. Convolution layer와 fully connected layer의 weight는 Kaiming initialization으로 초기화하였다. Optimizer는 Adam이며, epoch 수는 150, mini-batch size는 256, learning rate는 0.0001이다. 구현은 PyTorch 1.1.0 기반이며, 실험 환경은 2-core Intel Xeon CPU와 NVIDIA GeForce RTX 2080 Ti 11 GB GPU이다. Random seed도 고정하여 network initialization과 data loading의 randomness를 줄였다.

Dataset은 1,825명의 subject를 training 80%, validation 10%, test 10%로 나누었다. 즉 training set은 1,460명, validation set은 182명, test set은 183명이다. 이 분할은 inter-subject manner로 수행되었으므로 동일 subject의 데이터가 세 subset에 동시에 들어가지 않는다. 또한 논문은 intra-subject manner로 individual model도 구현하였다. Individual model은 각 subject의 데이터를 8:1:1로 나누어 개인별 모델을 평가하는 방식이다.

평가 지표는 mean error, 즉 ME, mean absolute error, 즉 MAE, standard deviation, 즉 STD이다. 또한 AAMI와 BHS standard를 사용한다. AAMI standard는 85명 이상의 subject, ME 5 mmHg 이하, STD 8 mmHg 이하를 요구한다. BHS standard는 error가 $\pm 5$, $\pm 10$, $\pm 15$ mmHg 이내에 들어오는 cumulative percentage에 따라 Grade A, B, C를 부여한다. 모델 복잡도는 parameter 수와 multiply-adds, 즉 MAdds로 평가한다.

## 4. 실험 및 결과

### 4.1 Architecture ablation 결과

Table 5는 architecture별 성능을 비교한다. Ensemble ML은 handcrafted feature를 사용했음에도 DBP $7.83 \pm 9.18$ mmHg, SBP $12.17 \pm 16.98$ mmHg로 가장 큰 error를 보였다. 이는 end-to-end deep learning이 raw PPG signal에서 더 유용한 representation을 학습할 수 있음을 시사한다.

VGGNet은 DBP $3.25 \pm 4.88$ mmHg, SBP $6.60 \pm 10.65$ mmHg를 기록하였다. ResNet은 DBP $3.09 \pm 4.87$ mmHg, SBP $6.42 \pm 10.52$ mmHg로 VGGNet보다 약간 개선되었다. ResNet+GRU는 DBP $3.01 \pm 4.77$ mmHg, SBP $6.57 \pm 11.10$ mmHg로, GRU를 추가했음에도 SBP 측면에서 ResNet보다 나아지지 않았다. 저자들은 CNN-RNN 구조가 항상 유리한 것은 아니며, RNN 계열의 sequential nature가 학습을 어렵게 만들 수 있다고 해석한다.

ECNet은 multi-scale 구조를 포함하지만 DBP $4.98 \pm 6.44$ mmHg, SBP $10.44 \pm 13.85$ mmHg로 기대보다 낮은 성능을 보였다. 이는 multi-scale branch를 단순히 구성하는 것만으로는 충분하지 않으며, network configuration과 fusion 방식이 중요함을 보여준다. 특히 ECNet은 653,520 parameter와 28.45M MAdds로 계산량이 크지만 성능은 좋지 않았다.

MSFNet 계열은 baseline보다 명확히 좋은 결과를 보였다. MSFNet-1S는 DBP $2.78 \pm 4.48$ mmHg, SBP $6.12 \pm 10.29$ mmHg이고, MSFNet-2S는 DBP $2.82 \pm 4.46$ mmHg, SBP $5.82 \pm 9.64$ mmHg이다. MSFNet-3S는 DBP $2.77 \pm 4.55$ mmHg, SBP $5.80 \pm 9.58$ mmHg로 2S보다 약간 좋지만, parameter와 계산량이 증가한다. 저자들은 3-scale의 추가 이득이 제한적이므로 성능과 complexity의 균형을 고려해 2-scale setting을 최종 선택하였다.

MSF-BAM은 DBP $2.63 \pm 4.22$ mmHg, SBP $5.34 \pm 8.98$ mmHg로 MSFNet-2S보다 개선되었다. 이는 BAM이 shared feature에서 informative feature를 강조하고 불필요한 feature를 억제하는 데 도움이 된다는 것을 보여준다.

최종 MSF-MTL(2S)는 DBP $2.58 \pm 4.13$ mmHg, SBP $5.22 \pm 8.80$ mmHg로 가장 좋은 universal model 성능을 보였다. Parameter 수는 970,940개, 계산량은 10.45M MAdds이다. MSF-BAM보다 parameter 수는 증가하지만, DBP와 SBP 모두 추가 개선이 나타났다. 이는 SBP와 DBP에 별도 task-specific subnetwork를 두는 MTL 구조가 실제로 도움이 되었음을 의미한다.

### 4.2 Multi-scale attention 분석

논문은 MSF-MTLNet의 attention weight를 시각화하여 scale별 중요도가 channel마다 다름을 보였다. Two-scale MSF-MTLNet에서는 Conv3와 Conv5에 해당하는 두 scale의 weight가 channel index에 따라 달라진다. Three-scale MSF-MTLNet에서는 Conv3, Conv5, Conv7에 해당하는 세 scale의 weight가 서로 다른 분포를 보인다.

이 결과는 서로 다른 scale의 feature가 동일한 channel 안에서도 complementary role을 수행한다는 것을 의미한다. 특정 channel에서는 local morphology를 포착하는 작은 receptive field가 더 중요할 수 있고, 다른 channel에서는 pulse interval이나 broader temporal pattern을 포착하는 큰 receptive field가 더 중요할 수 있다. Attention mechanism은 이러한 중요도를 동적으로 조정한다.

### 4.3 Scale factor $\alpha$ 분석

Scale factor $\alpha$는 backbone network width를 조절하기 위한 hyperparameter이다. Table 6은 $\alpha$ 값에 따른 성능과 복잡도를 보여준다. $\alpha=1.5$인 MSF-MTL-1.5는 DBP $2.39 \pm 4.13$ mmHg, SBP $4.76 \pm 8.65$ mmHg로 가장 좋은 성능을 보였지만, parameter 수가 2,162,776개, 계산량이 23.13M MAdds로 원래 모델보다 약 두 배 이상 크다.

원래 모델인 MSF-MTL-1.0은 DBP $2.58 \pm 4.13$ mmHg, SBP $5.22 \pm 8.80$ mmHg, parameter 970,940개, 10.45M MAdds를 사용한다. $\alpha=0.8$은 parameter 596,392개, 6.30M MAdds로 줄어들며 DBP $2.66 \pm 4.30$ mmHg, SBP $5.44 \pm 9.04$ mmHg를 기록한다. $\alpha=0.4$까지 줄이면 parameter는 162,684개, 2.08M MAdds로 작아지지만 DBP $3.18 \pm 4.73$ mmHg, SBP $6.51 \pm 9.87$ mmHg로 성능이 저하된다.

이 결과는 MSF-MTLNet이 hardware constraint에 맞게 scalable하게 조정될 수 있음을 보여준다. Mobile 또는 wearable device에서는 $\alpha$를 낮춰 계산량과 memory를 줄일 수 있고, 더 높은 성능이 필요한 환경에서는 $\alpha$를 높일 수 있다.

### 4.4 MTL과 BAM의 효과

논문은 BAM의 spatial and channel attention heat map을 DBP-specific subnetwork와 SBP-specific subnetwork에 대해 시각화하였다. 두 heat map은 일부 위치에서 공통적인 강조 영역을 보이지만, 동시에 상당한 차이도 보인다. 이는 DBP와 SBP estimation이 공통 feature를 공유하면서도 서로 다른 task-specific feature를 필요로 한다는 논문의 가설을 뒷받침한다.

MSF-BAM이 MSFNet-2S보다 좋은 성능을 보였고, MSF-MTL이 MSF-BAM보다 더 좋은 성능을 보였다는 점은 중요하다. 이는 첫째, attention-based feature refinement가 효과적이며, 둘째, SBP와 DBP를 같은 subnetwork에서 동시에 처리하는 것보다 각각 task-specific subnetwork에서 처리하는 것이 더 적절하다는 것을 의미한다. 따라서 이 논문에서 MTL은 단순히 output을 두 개로 만드는 수준이 아니라, task-specific feature learning을 가능하게 하는 구조적 기여를 한다.

### 4.5 Universal model 성능

Universal MSF-MTLNet의 regression plot에서는 DBP와 SBP 모두 reference와 estimation이 대체로 직선 주변에 분포한다. Correlation coefficient는 DBP 0.89, SBP 0.88이다. 이는 inter-subject split과 대규모 segment 조건에서도 모델이 reference BP와 높은 상관을 갖는 예측을 수행했음을 의미한다.

Bland-Altman analysis와 error distribution에서는 estimation error가 대체로 0 주변에 분포한다. DBP error는 대부분 10 mmHg 이내에 위치하고, SBP error는 대부분 20 mmHg 이내에 위치한다. 논문은 SBP target value의 STD가 DBP의 약 두 배이므로 SBP estimation이 더 어렵다고 설명한다. Table 3에서 SBP STD는 19 mmHg, DBP STD는 10 mmHg이다.

Training loss curve에서는 training MSE와 validation MSE가 점진적으로 수렴하며, 저자들은 이를 overfitting이 발생하지 않았다는 근거로 제시한다. 다만 training curve와 validation curve 사이에는 차이가 있는데, 논문은 MSE의 square characteristic, 대규모 dataset, inter-subject split이 이 차이를 시각적으로 확대한다고 설명한다.

### 4.6 AAMI 및 BHS standard 검증

Table 7은 MSF-MTLNet을 AAMI와 BHS 기준으로 검증한 결과를 보여준다. Universal model 기준으로 SBP는 ME 0.83 mmHg, STD 8.80 mmHg이며, DBP는 ME 0.43 mmHg, STD 4.13 mmHg이다. Test subject 수는 183명으로 AAMI의 최소 subject 수 조건인 85명을 만족한다. DBP는 ME와 STD 조건을 모두 만족하지만, SBP는 STD가 8 mmHg를 초과하므로 AAMI 기준을 완전히 만족하지 못한다.

BHS 기준에서는 universal model이 SBP Grade B, DBP Grade A를 달성한다. Universal model의 cumulative percentage는 SBP에서 error가 5 mmHg 이내인 비율 82.14%, 10 mmHg 이내 90.13%, 15 mmHg 이내 94.33%이다. DBP에서는 5 mmHg 이내 90.17%, 10 mmHg 이내 97.94%, 15 mmHg 이내 99.47%이다. DBP는 매우 높은 누적 비율을 보여 Grade A를 달성한다.

Individual model은 성능이 훨씬 좋다. Individual model 기준으로 SBP는 ME 0.31 mmHg, STD 2.59 mmHg이고, DBP는 ME 0.13 mmHg, STD 1.45 mmHg이다. BHS 기준에서도 SBP와 DBP 모두 Grade A를 달성한다. 이는 개인별 calibration 또는 개인 데이터를 활용한 fine-tuning이 PPG 기반 BP estimation에서 매우 강력한 성능 개선을 가져올 수 있음을 보여준다. 그러나 individual model 성능은 현실적으로 새 사용자에게 얼마만큼의 개인 데이터가 필요한지, calibration 부담이 어느 정도인지와 함께 해석해야 한다.

### 4.7 Tenfold cross-validation robustness

논문은 MSF-MTLNet의 robustness를 평가하기 위해 tenfold cross-validation도 수행하였다. 이때 dataset은 intra-subject manner로 10개 부분으로 나뉘어 data balance를 보장하였다. 9개 part로 training하고 나머지 1개 part로 evaluation하는 방식이다.

Tenfold cross-validation 결과, DBP estimation의 MAE는 $1.00 \pm 0.37$, DBP estimation의 STD는 $1.51 \pm 0.82$, SBP estimation의 MAE는 $1.88 \pm 0.45$, SBP estimation의 STD는 $2.75 \pm 0.11$로 보고되었다. 저자들은 이를 근거로 제안 모델이 validation set이 달라져도 robust한 성능을 보인다고 주장한다. 다만 이 cross-validation은 intra-subject manner이므로 universal inter-subject generalization보다 더 쉬운 설정이라는 점을 함께 고려해야 한다.

### 4.8 기존 연구와의 비교

Table 8은 최근 MIMIC 기반 연구들과 제안 모델을 비교한다. 비교는 완전히 공정하기 어렵다. 논문도 연구마다 data selection criteria, preprocessing method, signal quality filtering 방식, subject split 방식이 다르다고 명시한다. 예를 들어 어떤 연구는 SNR index, 어떤 연구는 skewness index, 어떤 연구는 manual check로 poor quality signal을 제거한다. 따라서 Table 8은 엄밀한 동일 조건 benchmark라기보다, 모델 성능과 설정을 함께 보여주는 비교표로 해석해야 한다.

Universal setting에서 제안 모델은 PPG만 사용하고 8초 segment를 입력으로 하며, 1,825명의 subject를 포함한다. 성능은 DBP $2.58 \pm 4.13$ mmHg, SBP $5.22 \pm 8.80$ mmHg이다. 이는 PPG-only end-to-end model로서는 경쟁력 있는 결과이다. Individual setting에서는 DBP $0.97 \pm 1.45$ mmHg, SBP $1.82 \pm 2.59$ mmHg로 매우 낮은 error를 보인다.

비교 대상 중 일부는 ECG와 PPG를 함께 사용하거나, handcrafted feature를 사용하거나, intra-subject setting으로 평가된다. 논문은 intra-subject 또는 individual model이 inter-subject universal model보다 일반적으로 error가 작다고 지적한다. 따라서 제안 모델이 universal과 individual 두 설정을 모두 제시한 것은 결과 해석에 도움이 된다. 특히 universal model이 inter-subject split에서 1,825명이라는 비교적 큰 subject 수를 사용했다는 점은 강점이다.

### 4.9 Complexity 분석

MSF-MTLNet의 backbone network에는 535,928개의 parameter가 있고, 각 BAM에는 18,977개, 각 task-specific network에는 198,529개의 parameter가 있다. 전체 architecture의 parameter 수는 970,940개이며, float32 기준 memory는 약 3.9 MB이다. 저자들은 deep learning model이 mobile deployment 전에 int8 quantization을 적용받는 경우가 많으므로, memory requirement를 원래의 1/4 수준까지 줄일 수 있다고 설명한다.

Local workstation에서 single inference time은 약 0.34 ms이고, computation은 10.45M MAdds이다. 저자들은 2018년 mobile device가 MobileNet 기반 image classification에 100–200 ms와 약 1.4 GFLOPs를 사용했다는 점을 언급하며, 1-D signal을 다루는 MSF-MTLNet은 현재 mobile device에서 real-time으로 실행될 가능성이 있다고 주장한다. 다만 실제 wearable device나 microcontroller에서의 전력 소모, sensor pipeline, preprocessing cost까지 포함한 end-to-end latency는 별도로 평가되지 않았다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 PPG-based BP estimation에서 multi-scale feature fusion의 필요성을 명확히 제기하고, 이를 scalable MSF block이라는 구체적 architecture로 구현했다는 점이다. PPG waveform은 local morphology와 broader temporal relationship을 모두 포함하므로, single-scale convolution보다 multi-scale structure가 더 적절할 수 있다. 실험에서도 MSFNet 계열은 VGGNet, ResNet, ResNet+GRU, ECNet보다 더 낮은 error를 보였다.

두 번째 강점은 attention mechanism을 단순한 성능 향상 장치로만 사용하지 않고, scale fusion과 task-specific feature refinement라는 두 수준에서 사용했다는 점이다. MSF block에서는 scale별 attention을 통해 서로 다른 receptive field의 feature를 adaptive하게 융합하고, BAM에서는 channel attention과 spatial attention으로 SBP 또는 DBP에 중요한 feature를 강조한다. Figure 5와 Figure 6의 시각화는 scale별 중요도와 task-specific attention이 실제로 서로 다르게 작동함을 보여준다.

세 번째 강점은 multi-task learning을 PPG 기반 SBP/DBP estimation에 명시적으로 도입했다는 점이다. SBP와 DBP는 같은 cardiovascular system에서 발생하는 관련 task이지만 feature importance가 완전히 같지 않다. MSF-MTLNet은 shared backbone과 task-specific subnetwork를 결합하여 parameter sharing의 효율성과 task-specific modeling의 장점을 동시에 얻는다. Table 5에서 MSF-MTL이 MSF-BAM보다 더 낮은 error를 보인 것은 이 설계가 효과적임을 뒷받침한다.

네 번째 강점은 inter-subject split을 사용한 평가이다. PPG 기반 BP estimation 연구에서 segment-level random split은 동일 subject의 유사 waveform이 train과 test에 동시에 포함될 수 있어 성능을 과대평가할 위험이 있다. 이 논문은 1,825명을 subject 기준으로 training, validation, test set에 분리하여 universal model을 평가했다. 이는 실제 새로운 사용자에 대한 일반화 가능성을 평가하는 데 더 적절하다.

다섯 번째 강점은 ablation study가 비교적 체계적이라는 점이다. 저자들은 ML baseline, single-scale CNN, residual CNN, CNN-RNN, EC block, MSF block scale 수, BAM, MTL, scale factor $\alpha$를 모두 비교한다. 이를 통해 성능 개선이 단순히 network depth나 parameter 수 증가 때문이 아니라, MSF structure와 MTL design에서 비롯되었음을 설득하려 한다.

여섯 번째 강점은 complexity analysis를 포함했다는 점이다. 제안 모델은 multi-scale architecture이지만 depthwise separable convolution을 사용해 10.45M MAdds 수준으로 계산량을 제한한다. 또한 scale factor $\alpha$를 통해 모델 크기를 조절할 수 있으므로, 성능과 resource 사이의 trade-off를 설계할 수 있다.

그러나 한계도 분명하다. 첫째, universal model의 SBP estimation은 AAMI standard를 완전히 만족하지 못한다. SBP ME는 낮지만 STD가 8.80 mmHg로 AAMI 기준인 8 mmHg를 초과한다. 논문은 “almost meets”라고 표현하지만, 실제 의료기기 validation 관점에서는 이 차이가 중요할 수 있다. DBP는 충분히 우수하지만, SBP의 높은 variation은 여전히 해결해야 할 문제이다.

둘째, 데이터는 UCI dataset, 즉 MIMIC subset 기반이며 ICU 환경의 fingertip PPG에 제한된다. 실제 wearable 환경에서는 wrist PPG, motion artifact, sensor pressure variation, skin tone, temperature, posture, daily activity 등의 변수가 훨씬 다양하다. 논문도 long-term BP measurement와 wristwatch prevalence가 MIMIC setting과 충돌한다고 한계로 인정한다. 따라서 실제 wearable device 적용 가능성은 외부 데이터와 장기 데이터에서 추가 검증이 필요하다.

셋째, preprocessing과 artifact removal이 비교적 단순하다. Skewness threshold로 low-quality segment를 제거하여 약 40%의 segment를 discard했는데, 이는 clean segment 중심의 성능을 높일 수 있다. 실제 연속 모니터링에서는 low-quality signal을 단순히 제거하기 어려운 경우가 많으며, 제거된 구간의 BP를 어떻게 처리할지도 중요하다. 또한 motion artifact가 다양한 형태로 나타나는 wearable PPG에서는 skewness만으로 충분하지 않을 수 있다.

넷째, individual model의 성능은 매우 좋지만, 이 성능은 intra-subject data split에 기반한다. 즉 같은 사람의 일부 데이터를 학습에 사용하고 나머지를 평가하는 설정이다. 이는 personalization이나 calibration이 가능한 환경에서는 유용하지만, 완전히 새로운 사용자에게 calibration 없이 적용하는 성능과는 다르다. 실제 적용에서는 얼마나 많은 개인 데이터가 필요한지, calibration frequency가 어느 정도인지, 장기적인 drift에 어떻게 대응할지가 중요하다.

다섯째, 모델은 SBP와 DBP scalar value를 추정할 뿐 continuous ABP waveform을 재구성하지 않는다. 논문은 continuous BP estimation이라고 표현하지만, 이는 8초 segment 단위로 SBP/DBP를 연속 추정한다는 의미에 가깝다. Pulse waveform morphology, beat-to-beat ABP waveform, dicrotic notch pressure, pulse pressure waveform 등을 직접 제공하지는 않는다. 임상적으로 waveform-level information이 필요한 상황에서는 별도 모델이 필요하다.

여섯째, 비교 연구는 동일 조건에서 완전히 재현된 benchmark가 아니다. 저자도 Table 8의 비교가 preprocessing, subject selection, split 방식 차이 때문에 완전히 공정하기 어렵다고 인정한다. 따라서 “cutting-edge method보다 우수하다”는 주장은 동일 데이터와 동일 protocol에서의 비교를 통해 더 강하게 검증될 필요가 있다.

마지막으로, 논문은 mobile deployment 가능성을 complexity 관점에서 논의하지만, 실제 device implementation, quantization 후 성능 변화, preprocessing 포함 latency, power consumption, sensor acquisition pipeline은 평가하지 않았다. 따라서 real-time wearable deployment 가능성은 promising하지만 아직 실증된 것은 아니다.

## 6. 결론

이 논문은 PPG 기반 cuffless blood pressure estimation을 위해 end-to-end MSF-MTLNet을 제안하였다. 모델은 단일 PPG sensor에서 얻은 8초 PPG segment와 그 derivative인 VPG, APG를 3-channel input으로 사용하고, scalable MSF backbone을 통해 multi-scale feature를 추출한다. 이후 SBP-specific subnetwork와 DBP-specific subnetwork가 각각 BAM을 사용하여 task-specific feature를 정제하고 SBP와 DBP를 추정한다.

논문의 주요 기여는 세 가지이다. 첫째, PPG waveform의 BP-related information이 다양한 time scale에 존재한다는 점에 주목하고, attention 기반 multi-scale fusion block을 설계하였다. 둘째, SBP와 DBP estimation의 공통성과 차이를 반영하기 위해 hard parameter sharing 기반 multi-task learning을 도입하였다. 셋째, 1,825명, 277,600 segment 규모의 public dataset에서 inter-subject split을 사용하여 비교적 엄격한 generalization 평가를 수행하였다.

실험 결과, universal MSF-MTLNet은 DBP $2.58 \pm 4.13$ mmHg, SBP $5.22 \pm 8.80$ mmHg의 MAE±STD 수준 성능을 보였고, AAMI/BHS 검증에서는 DBP가 AAMI 기준과 BHS Grade A를 만족했으며 SBP는 BHS Grade B를 달성하였다. Individual model에서는 DBP $0.97 \pm 1.45$ mmHg, SBP $1.82 \pm 2.59$ mmHg로 훨씬 낮은 error를 보였고, SBP와 DBP 모두 BHS Grade A를 달성하였다. Ablation study는 MSF block, BAM, MTL이 각각 성능 향상에 기여함을 보여준다.

이 연구는 PPG-only BP estimation에서 architecture design의 중요한 방향을 제시한다. 단순 handcrafted feature나 single-scale CNN보다, multi-scale waveform pattern을 adaptive하게 융합하고 SBP/DBP task-specific representation을 학습하는 구조가 더 유망하다는 점을 실험적으로 보였다. 또한 depthwise separable convolution과 scale factor를 사용하여 계산 효율성을 고려한 점은 wearable deployment 가능성을 높이는 요소이다.

다만 실제 임상 및 wearable 적용을 위해서는 SBP universal model의 AAMI 기준 미충족 문제, wrist PPG 및 장기 데이터에서의 외부 검증, motion artifact에 대한 robustness, personalization 필요성, 실제 device deployment 평가가 추가로 필요하다. 이러한 보완이 이루어진다면 MSF-MTLNet과 같은 multi-scale multi-task architecture는 cuffless continuous BP monitoring을 위한 중요한 기반 기술로 발전할 수 있다.
