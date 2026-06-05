# TransfoRhythm: A Transformer Architecture Conductive to Blood Pressure Estimation through Solo PPG Signal Capturing

* **저자**: Amir Arjomand, Amin Boudesh, Farnoush Bayatmakou, Georgiy Krylov, Kenneth B. Kent, Arash Mohammadi
* **발표연도**: 2025

## 1. 논문 개요

이 논문은 단일 Photoplethysmography, PPG 신호만을 이용하여 cuff-less blood pressure estimation을 수행하는 Transformer 기반 deep neural network인 **TransfoRhythm**을 제안한다. 논문의 핵심 목표는 ECG와 같은 보조 센서 없이 PPG 단독 신호에서 systolic blood pressure, SBP와 diastolic blood pressure, DBP를 정확하게 추정하는 것이다.

기존의 cuff 기반 혈압 측정은 병원과 가정에서 널리 사용되지만, 연속 측정이 어렵고 cuff inflation과 deflation이 사용자의 불편함을 유발한다. 또한 야간 혈압 변화, 갑작스러운 hypotension, nocturnal dipping이나 surge처럼 시간에 따라 변하는 중요한 혈압 패턴을 놓칠 수 있다. 논문은 continuous BP monitoring이 hypertension 관리와 cardiovascular disease 위험 평가에 중요하다고 설명한다.

최근 cuff-less BP estimation 연구에서는 ECG와 PPG를 함께 사용하는 방식이 많이 활용되었다. ECG와 PPG를 결합하면 Pulse Transit Time, PTT 또는 Pulse Arrival Time, PAT 같은 feature를 얻을 수 있어 혈압 추정에 도움이 된다. 그러나 이 방식은 여러 종류의 센서가 필요하고, 센서 간 거리 유지, 높은 전력 소모, 재보정 문제, 착용 불편성 등의 한계가 있다. 이에 대한 대안으로 단일 PPG 기반 혈압 추정이 주목받지만, ECG가 제공하던 timing reference가 사라지기 때문에 PPG waveform morphology에서 혈압 관련 정보를 더 정교하게 추출해야 한다.

이 논문은 이러한 문제를 해결하기 위해 Transformer architecture의 Multi-Head Attention, MHA를 활용한다. TransfoRhythm은 PPG에서 추출한 morphological feature sequence를 입력으로 받아, time sequence 내부의 dependency와 segment 간 similarity를 학습한다. 논문은 이 모델이 sequence order를 유지하면서도 parallel computation을 가능하게 하는 regressive time series transformer라고 설명한다.

또한 이 연구는 MIMIC-IV Waveform Dataset을 cuff-less BP estimation에 사용한 첫 번째 연구라고 주장한다. 기존 연구들은 MIMIC-II나 MIMIC-III를 주로 사용했지만, 이 논문은 최신 ICU waveform database인 MIMIC-IV를 사용한다. 추출된 텍스트 기준으로 총 198명의 de-identified patient signal을 포함하고, preprocessing 후 360개 purified records가 식별되며, 약 126,000개의 observation을 사용한 것으로 설명된다.

초록에서 제시된 최종 성능은 매우 높다. TransfoRhythm은 SBP와 DBP 각각에 대해 RMSE $[2.21, 1.84]$ mmHg, MAE $[1.37, 1.06]$ mmHg를 달성했다고 보고한다. 이는 PPG-only cuff-less BP estimation 연구에서 매우 낮은 오차에 해당한다. 다만 제공된 텍스트는 Evaluation Metrics 수식 중간에서 끊겨 있어, 전체 실험 표, 비교 모델별 수치, ablation study, fold별 결과, subject-wise split 여부 등은 확인할 수 없다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 PPG 신호에서 beat-level 또는 cycle-level morphological feature를 추출한 뒤, 이를 Transformer 기반 time series regression model에 입력하여 SBP와 DBP를 추정하는 것이다. 단순히 raw PPG waveform 전체를 CNN이나 RNN에 넣는 방식이 아니라, PPG cycle과 second derivative PPG, SDPPG에서 혈압과 관련된 feature를 추출하고, 이 feature sequence에 embedding과 positional encoding을 더해 Transformer 입력으로 구성한다.

기존 PPG-only 혈압 추정의 어려움은 PPG가 여러 생리적 요인의 혼합 결과라는 데 있다. PPG는 혈관 내 blood volume 변화, arterial stiffness, pulse morphology, sensor contact, motion artifact, high-frequency noise의 영향을 동시에 받는다. ECG가 없는 경우 PTT와 같은 timing feature를 직접 계산할 수 없기 때문에, PPG waveform 자체의 형태와 반복 주기에서 혈압 관련 정보를 찾아야 한다. 논문은 이 문제를 “morphological feature를 신중하게 고려해야 하는 문제”로 정의한다.

TransfoRhythm의 중심 설계는 **Multi-Head Attention**이다. MHA는 입력 feature sequence의 서로 다른 위치가 서로 얼마나 관련되는지를 학습한다. PPG 기반 혈압 추정에서는 한 cycle 내부의 rise time, peak, dicrotic notch, fall time, integration area뿐 아니라 여러 frame에 걸친 temporal dependency가 중요할 수 있다. MHA는 이러한 위치 간 관계를 병렬적으로 학습하여, 단일 frame 또는 단일 feature만으로는 포착하기 어려운 패턴을 활용할 수 있다.

또 다른 핵심은 **positional encoding**이다. Transformer의 self-attention은 기본적으로 token의 순서를 직접 알지 못한다. 따라서 feature sequence의 시간적 순서를 보존하려면 positional encoding이 필요하다. 논문은 NLP에서 사용되는 방식과 유사하게 sinusoidal positional encoding을 feature embedding에 더하여, 모델이 각 frame의 content information과 temporal position information을 함께 사용할 수 있도록 한다.

이 논문이 기존 방법과 차별화되는 지점은 크게 세 가지이다. 첫째, ECG 없이 solo PPG signal만 사용한다. 둘째, MIMIC-IV Waveform Dataset을 사용한다. 셋째, handcrafted morphological feature와 Transformer 기반 attention mechanism을 결합한다. 즉, 완전한 raw signal end-to-end 모델이라기보다는, PPG와 SDPPG에서 의미 있는 physiological feature를 추출하고, Transformer가 이 feature들의 시간적 관계를 학습하도록 설계한 hybrid data-driven approach이다.

## 3. 상세 방법 설명

논문의 전체 파이프라인은 dataset formation, preprocessing, feature extraction, stacked frame 구성, embedding 확장, positional encoding 추가, Multi-Head Attention 기반 Transformer block, position-wise feed-forward processing, time frame compressor, flattening 및 SBP/DBP regression으로 구성된다.

먼저 데이터는 **MIMIC-IV Waveform Dataset**에서 가져온다. 이 dataset은 PhysioNet repository에서 제공되며, modern ICU bedside monitoring device에서 수집된 waveform recording을 포함한다. 사용된 signal에는 ECG, PPG, invasive ABP가 포함되어 있지만, 이 논문의 모델 입력은 solo PPG signal이다. ABP는 SBP와 DBP label을 얻기 위한 reference로 사용된 것으로 이해된다. Sampling rate는 62.4 Hz이며, 전체 dataset에는 198명의 de-identified individual이 포함된다.

데이터 기록 길이는 환자별로 다양하다. 대부분의 환자는 1.5시간 미만의 recording을 가지지만, 일부 환자는 15시간이 넘는 recording도 포함한다. 논문은 record duration distribution을 제시하고, patient data의 SBP와 DBP distribution도 보여준다. 제공된 설명에 따르면 대부분의 관측값은 normal blood pressure range에 집중되어 있다.

### Preprocessing

Preprocessing은 data cleaning과 signal filtering으로 구성된다. Data cleaning 단계에서는 amplitude가 자주 range를 벗어나는 signal, 최소 시간 기준인 15분을 만족하지 못하는 record, long record 내부에서 flattened-line 또는 extremely noisy segment를 포함하는 frame을 제거한다. 이 과정을 거친 뒤 총 360개의 purified records가 식별되었다고 설명된다.

Signal filtering 단계에서는 5th order Butterworth bandpass filter와 5th order Moving Average Filter, MAF를 사용한다. Butterworth filter의 frequency range는 0.7–10 Hz이다. 이 filter는 wandering baseline과 high-frequency noise를 제거하기 위해 사용된다. 논문은 power-line interference인 50–60 Hz noise도 제거 대상이라고 설명한다. 이후 MAF는 signal smoothing을 위해 사용되며, 부분 구간의 평균을 사용해 overshoot와 fluctuation을 줄인다. 논문은 이 두 필터의 결합으로 peak와 foot이 잘 보존된 smooth PPG signal을 얻었다고 설명한다.

Butterworth bandpass filtering은 다음과 같은 목적을 가진다. 낮은 주파수 영역에서는 baseline wander가 제거되고, 높은 주파수 영역에서는 sensor noise와 전원 간섭이 줄어든다. MAF는 다음처럼 간단한 moving average 형태로 이해할 수 있다.

$$
\tilde{x}[t] = \frac{1}{M}\sum_{j=0}^{M-1}x[t-j]
$$

여기서 $x[t]$는 원래 PPG signal, $\tilde{x}[t]$는 smoothing된 signal, $M$은 averaging window 길이이다. 논문은 MAF를 5th order라고 표현하지만, 정확한 window 구현 세부 사항은 제공된 텍스트만으로는 확인할 수 없다.

### Feature Extraction

Feature extraction은 이 논문의 중요한 부분이다. 논문은 complete 10-second frame 전체를 그대로 분석하면 variability와 noise가 증가하여 subtle physiological marker가 가려질 수 있다고 설명한다. 따라서 개별 PPG cycle을 식별하고 분리하는 것이 중요하다고 본다.

Cycle extraction은 PPG signal과 second derivative PPG, SDPPG에서 peak와 trough, 즉 valley point를 동시에 찾는 robust algorithm으로 수행된다. 이 알고리즘은 adjustable peak-to-peak distance parameter를 사용하여 peak height와 amplitude 변화에 대응한다. Cycle extraction 전에 amplitude thresholding을 적용하여 sensor contact가 나쁘거나 motion artifact가 큰 frame을 제거한다.

최종적으로 선택된 feature는 12개이다. 이 feature들은 correlation analysis에 근거하여 선택되었다고 설명된다. Table 1에서 제공된 feature는 다음과 같다.

PPG_Cycle_Duration은 cycle의 전체 길이이며, $TD1 = T_{end} - T_{start}$로 정의된다. 이는 한 heartbeat cycle의 duration을 나타낸다. PPG_Rise_Half_Peak, $T_{rhp}$는 cycle start에서 rise mode의 half peak amplitude에 도달하는 데 걸리는 시간이다. PPG_Peak_To_Notch는 peak에서 dicrotic notch까지의 시간이며, $TD2 = T_{dn} - T_p$로 정의된다. PPG_Rise_Peak, $T_p$는 start에서 peak까지의 시간이다. PPG_Fall_Half, $T_{fh}$는 peak에서 amplitude의 절반 수준까지 떨어지는 데 걸리는 시간이다. PPG_Fall_Peak은 peak에서 foot 또는 end까지의 시간이며, $TD3 = T_{end} - T_p$로 정의된다.

PPG_BPM_Frame, PBF는 frame당 cycle 수를 나타낸다. PPG_Integration, PPGI는 PPG cycle 아래 green area의 적분값이다. 이는 cycle area, 즉 waveform amplitude와 duration을 함께 반영하는 feature이다. SDPPG_Fall_Half_Foot, SDPPG_Tfhf는 SDPPG에서 start부터 fall mode half amplitude까지의 duration이다. SDPPG_Extremum_Amp, SDPPG_AMP는 SDPPG의 maximum amplitude이다. SDPPG_Integration, SDPPGI는 SDPPG upper curve area이다. SDPPG_Foot_Peak은 SDPPG에서 foot에서 peak까지의 duration이며, $TD4 = T_{peak} - T_{foot}$로 정의된다.

이 feature들은 PPG waveform의 timing, shape, area, SDPPG curvature 정보를 포함한다. 혈압은 arterial stiffness, pulse wave reflection, vascular compliance와 관련되므로, rise time, peak-to-notch interval, fall time, integration area 같은 feature들이 BP estimation에 유용할 수 있다.

### Stacked Frame 및 Embedding

Feature extraction 후 전체 dataset은 12개 final feature를 기반으로 구성된다. 모델 입력은 stacked frames로 정의된다. 논문은 shuffled frame 기반으로 $N_i = 20,000$ frames, time frame sequence $T = 48$, input feature length $L_{in}=12$를 사용한다고 설명한다.

Stacked Frame, $SF$는 다음과 같이 정의된다.

$$
SF = [F_T^{L,1}, F_T^{L,2}, \ldots, F_T^{L,n}]
$$

여기서 $n$은 frame 수를 의미하고, $F_T^{L,i}$는 input feature length $L$을 가지며 time frame sequence $T$에 걸친 개별 frame을 의미한다. 제공된 텍스트에 따르면 $L_{in}=12$이고, $T=48$이다.

이후 1D convolution layer를 사용해 feature dimensionality를 확장한다. 논문은 feature length를 12에서 128로 확장한다고 설명한다. 제공된 수식은 다음과 같다.

$$
ST_{Extended}(N_i, L_{out}) =
Bias(L_{out}) +
\sum_{k=0}^{L_{in}} weight(L_{out}, k)input(N_i,k)
$$

여기서 $ST_{Extended} \in \mathbb{R}^{N_i \times L_{out} \times T}$이고, $weight \in \mathbb{R}^{L_{out} \times L_{in} \times KS}$, $input \in \mathbb{R}^{N_i \times L_{in} \times T}$이다. Kernel size는 $KS=1$이고, $L_{in}=12$, $L_{out}=128$이다. 쉽게 말하면, 각 time frame의 12차원 feature vector를 128차원 embedding으로 선형 변환하는 1D convolution을 적용한 것이다. Kernel size가 1이므로 시간 방향의 주변 context를 섞기보다는 feature channel dimension을 확장하는 역할이 크다.

### Positional Encoding

Transformer는 self-attention만으로는 sequence order를 알 수 없기 때문에 positional encoding이 필요하다. 논문은 sinusoidal positional encoding을 사용한다고 설명한다. 일반적인 sinusoidal positional encoding은 다음과 같은 형태이다.

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

여기서 $pos$는 sequence position, $i$는 embedding dimension index, $d$는 embedding dimension이다. 논문에는 이 수식이 직접 제시되지는 않지만, sinusoidal function of different frequencies를 더한다고 설명한다. Positional encoding array는 $ST_{Extended}$에 element-wise로 더해져 embedded input을 만든다.

제공된 텍스트에는 “feature-length of 48 and time frame sequence length of 128”이라는 표현이 나오는데, 앞서 feature length를 128로 확장하고 time frame sequence가 48이라고 설명한 부분과 표현이 다소 뒤바뀐 것처럼 보인다. 따라서 정확한 tensor dimension 표기는 원문 전체 또는 code가 필요하다. 다만 핵심은 12개 feature를 1D convolution으로 고차원 embedding으로 확장한 뒤, positional encoding을 더해 Transformer input matrix를 만든다는 점이다.

### Multi-Head Attention

TransfoRhythm의 중심 구성 요소는 Multi-Head Attention이다. Self-attention은 각 frame 또는 token이 다른 frame과 어떤 관계를 갖는지 학습한다. 이를 위해 input embedding에서 Key, Query, Value matrix를 만든다. 논문에는 “Key, Queue, and Value”라고 되어 있으나, 문맥상 “Query”를 의미하는 것으로 보인다.

Query, Key, Value는 각각 learnable weight $W_Q$, $W_K$, $W_V$를 input embedding에 곱해 얻는다. Self-attention은 일반적으로 다음과 같이 계산된다.

$$
Attention(Q,K,V) =
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

여기서 $Q$는 Query, $K$는 Key, $V$는 Value, $d_k$는 Key vector dimension이다. 이 연산은 각 token이 다른 token을 얼마나 참고해야 하는지를 계산한다.

Multi-Head Attention은 이러한 attention을 여러 head에서 병렬로 수행한 뒤 concatenate한다.

$$
MHA(Q,K,V) = Concat(head_1, head_2, \ldots, head_h)W_O
$$

$$
head_j = Attention(QW_Q^j, KW_K^j, VW_V^j)
$$

이 논문에서는 head 수를 14로 설정한다. 논문은 head 수 같은 architecture parameter가 experiment를 통해 설정되었다고 설명한다. MHA의 마지막에는 linear layer를 추가하여 non-linearity 또는 feature transformation을 수행한다. 이 과정은 input feature 사이의 복잡한 관계와 dependency를 학습하여 BP prediction 성능을 높이기 위한 것이다.

### Position-Wise Feed-Forward Network

MHA 출력은 position-wise feed-forward block으로 전달된다. 이 block은 1D convolution layer와 ReLU activation function을 사용한다. Position-wise feed-forward layer는 각 position의 representation을 독립적으로 변환하면서, attention이 만든 contextual representation을 더 풍부하게 만드는 역할을 한다. 논문은 normalization layer와 dropout도 사용하여 generalization을 높이고 overfitting을 줄였다고 설명한다.

### Time Frame Compressor와 Flattening

초기 실험에서 final flattening layer가 높은 computational overhead를 유발했다고 설명된다. 이를 해결하기 위해 Time Frame Compressor unit을 도입한다. 이 모듈은 learnable transformer가 아니라 non-learnable transformer이며, temporal order를 더 짧은 condensed representation으로 줄인다. 즉, 긴 time frame sequence를 그대로 flatten하지 않고 시간 차원을 압축하여 계산량을 줄인다.

마지막 단계에서는 temporally compressed signal을 flattening layer로 변환하고, ReLU function을 통해 SBP와 DBP 값을 산출한다. 제공된 텍스트에서는 final regression head의 정확한 layer 수, hidden dimension, output activation 세부 설정은 명시되지 않는다. 따라서 모델의 마지막 회귀 구조는 “time compressor 후 flattening과 ReLU를 통해 SBP/DBP를 예측한다”는 수준으로만 확인할 수 있다.

### Cross-Validation 및 평가 지표

논문은 5-fold cross-validation을 사용한다. 전체 dataset은 약 126,000 observation으로 구성되며, 무작위로 shuffle한 뒤 5개 subset으로 나눈다. 각 fold는 전체의 20%, 약 25,200 records를 포함한다. 각 iteration에서 80%는 training, 20%는 testing에 사용된다. 다만 제공된 텍스트에는 “80% of the data (20,160 records per fold)”라는 표현이 있는데, fold 하나의 80%를 의미하는 듯하나 전체 cross-validation 설명과 숫자가 약간 혼동될 수 있다. 일반적인 5-fold cross-validation이라면 각 iteration에서 약 100,800개 observation을 training에, 약 25,200개 observation을 test에 사용하는 것이 자연스럽다. 따라서 이 수치 표현은 원문 전체 확인이 필요하다.

평가 지표로는 R-squared, $R^2$, Mean Error, ME, Mean Absolute Error, MAE, Root Mean Square Error, RMSE를 사용한다고 설명된다. 그러나 제공된 텍스트는 $R^2$ 수식 중간에서 끊겨 있어 전체 수식과 결과 표는 확인할 수 없다. 일반적으로 MAE와 RMSE는 다음과 같다.

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|T_i - P_i|
$$

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(T_i - P_i)^2}
$$

여기서 $T_i$는 실제 혈압값, $P_i$는 예측 혈압값이다. ME는 평균 오차로 다음과 같이 쓸 수 있다.

$$
ME = \frac{1}{n}\sum_{i=1}^{n}(T_i - P_i)
$$

$R^2$는 일반적으로 prediction이 target variance를 얼마나 설명하는지를 나타낸다. 다만 논문에서 제시한 정확한 $R^2$ 수식은 텍스트가 끊겨 있어 확인할 수 없다.

## 4. 실험 및 결과

제공된 텍스트에서 확인 가능한 실험 결과는 주로 초록에 제시되어 있다. TransfoRhythm은 SBP와 DBP에 대해 각각 RMSE $[2.21, 1.84]$ mmHg, MAE $[1.37, 1.06]$ mmHg를 달성했다고 보고한다. 즉, SBP RMSE는 2.21 mmHg, DBP RMSE는 1.84 mmHg이며, SBP MAE는 1.37 mmHg, DBP MAE는 1.06 mmHg이다.

이 수치는 PPG-only BP estimation 연구에서 매우 낮은 오차이다. 일반적으로 cuff-less BP estimation에서는 SBP가 DBP보다 더 어렵고, SBP 오차가 더 높게 나타나는 경우가 많다. 이 논문에서도 SBP의 RMSE와 MAE가 DBP보다 높지만, 두 값 모두 매우 낮다. 특히 MAE가 1–1.5 mmHg 수준이라는 것은 모델이 평균적으로 reference blood pressure와 거의 일치하는 예측을 수행했다는 의미이다.

논문은 TransfoRhythm이 state-of-the-art counterpart보다 우수하다고 주장한다. Introduction에서는 비교 대상으로 여러 기존 연구를 언급한다. 예를 들어 ECG와 PPG를 함께 사용한 hybrid CNN-LSTM 연구는 SBP MAE 4.41 mmHg, DBP MAE 2.91 mmHg를 보고했고, MIMIC I 및 III 기반 TCN 연구는 SBP RMSE 3.03 mmHg, DBP RMSE 1.58 mmHg를 보고했다. MLP-Mixer 기반 연구는 MIMIC II dataset에서 SBP RMSE 5.10 mmHg, DBP RMSE 3.13 mmHg를 보고했다. 이들과 비교하면 TransfoRhythm의 SBP RMSE 2.21, DBP RMSE 1.84, MAE 1.37/1.06은 상당히 경쟁력 있는 결과이다.

다만 제공된 텍스트에는 실제 Results section의 상세 비교 표가 포함되어 있지 않다. 따라서 어떤 baseline을 동일한 MIMIC-IV preprocessing, 동일한 cross-validation split, 동일한 feature extraction 조건에서 재현했는지 확인할 수 없다. 기존 연구의 수치와 직접 비교한 것인지, 아니면 동일 조건에서 재학습한 baseline과 비교한 것인지는 텍스트만으로 명확하지 않다. 이 점은 결과 해석에서 중요하다.

Dataset 측면에서 논문은 MIMIC-IV Waveform Dataset을 사용했다는 점을 강조한다. 총 198명의 환자와 ICU에서 수집된 ECG, PPG, invasive ABP signal이 포함되며, sampling rate는 62.4 Hz이다. Preprocessing 후 360 purified records와 약 126,000 observations를 사용한다고 설명된다. 데이터는 5-fold cross-validation으로 평가되며, 모든 observation이 training과 testing에 모두 사용되도록 설계했다고 설명한다.

그러나 여기서 중요한 한계가 있다. 제공된 텍스트에서는 cross-validation이 **subject-wise**로 수행되었는지, 아니면 observation-wise random shuffle로 수행되었는지 명확하지 않다. 논문은 “approximately 126,000 observations was randomly shuffled and split into five equal subsets”라고 설명한다. 이 표현만 보면 같은 환자에서 나온 여러 observations가 training fold와 test fold에 동시에 들어갈 가능성이 있다. 만약 그렇다면 모델이 환자별 PPG morphology와 혈압 범위를 학습할 수 있어, 새로운 환자에 대한 generalization 성능보다 높은 성능이 보고될 수 있다. 혈압 추정 연구에서는 subject-independent split이 매우 중요하다.

또한 실험 결과가 추출 텍스트상 초록에만 제시되므로, fold별 표준편차, Bland–Altman analysis, AAMI/BHS/ISO 기준, error distribution, 고혈압 범위별 성능, 연령별 성능, motion artifact robustness 등은 확인할 수 없다. 따라서 보고된 MAE/RMSE는 인상적이지만, 임상 적용 가능성을 판단하려면 추가 평가 정보가 필요하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 단일 PPG 기반 혈압 추정에 Transformer architecture를 체계적으로 적용했다는 점이다. ECG 없이 PPG만 사용하면 hardware가 단순해지고 wearable device에 적용하기 쉬워진다. TransfoRhythm은 PPG와 SDPPG에서 추출한 morphological feature sequence를 attention mechanism으로 처리하여, ECG가 없는 상황에서도 혈압 관련 temporal dependency를 학습하려 한다.

두 번째 강점은 MIMIC-IV Waveform Dataset을 사용했다는 점이다. 기존 연구들은 MIMIC-II나 MIMIC-III에 많이 의존했지만, 이 논문은 최신 ICU waveform dataset인 MIMIC-IV를 사용한다. 논문은 이것이 cuff-less BP prediction에서 MIMIC-IV를 적용한 첫 연구라고 주장한다. 최신 dataset을 활용했다는 점은 연구의 시의성과 확장성 측면에서 의미가 있다.

세 번째 강점은 raw signal을 그대로 사용하는 것이 아니라 PPG cycle과 SDPPG에서 physiologically meaningful feature를 추출한다는 점이다. PPG_Cycle_Duration, peak-to-notch interval, rise time, fall time, integration area, SDPPG amplitude와 integration 등은 waveform morphology와 vascular dynamics를 반영할 가능성이 있다. Transformer는 이 feature sequence 사이의 관계를 학습하므로, 완전한 black-box raw signal model보다 physiological interpretability가 조금 더 높다.

네 번째 강점은 preprocessing이 비교적 명확하게 구성되어 있다는 점이다. Data cleaning, Butterworth bandpass filtering, Moving Average Filter, amplitude thresholding, peak/trough detection, cycle extraction이 순차적으로 적용된다. 논문은 flattened-line이나 extremely noisy segment를 제거하고, peak와 foot을 보존하는 filtering을 수행한다고 설명한다. 이는 PPG 기반 모델에서 매우 중요한 단계이다.

다섯 번째 강점은 reported performance가 매우 높다는 점이다. 초록의 수치만 보면 SBP MAE 1.37 mmHg, DBP MAE 1.06 mmHg는 state-of-the-art 수준 또는 그 이상으로 보인다. 특히 solo PPG만 사용했다는 점을 고려하면 매우 강력한 결과이다.

그러나 한계도 분명하다. 첫째, 제공된 텍스트 기준으로 결과 검증 방식이 subject-wise인지 불명확하다. Observation-wise random shuffle을 사용했다면 같은 환자 데이터가 train과 test에 동시에 포함될 수 있다. 이 경우 모델은 subject-specific distribution을 학습하여 test performance가 과대평가될 수 있다. 실제 cuff-less BP device는 처음 보는 사용자에게 적용되어야 하므로, subject-independent validation이 필수적이다.

둘째, reported error가 매우 낮기 때문에 더욱 엄격한 검증이 필요하다. PPG-only BP estimation에서 MAE 1 mmHg 수준은 매우 뛰어난 결과이지만, preprocessing, label extraction, split 방식, frame overlap, 동일 record 내 segment leakage에 따라 성능이 크게 달라질 수 있다. 제공된 텍스트만으로는 126,000 observations가 서로 독립적인지, overlapping segment가 있는지, 같은 record의 인접 frame이 train/test에 나뉘었는지 확인할 수 없다.

셋째, MIMIC-IV는 ICU dataset이다. ICU 환자는 다양한 질환, 약물, hemodynamic intervention, sensor placement condition을 가진다. 이는 모델이 임상 중환자 환경에서는 잘 작동할 수 있지만, 일반 healthy population이나 wearable daily-life setting으로 바로 일반화된다고 보장할 수 없다는 뜻이다. 또한 ICU의 invasive ABP reference는 강력한 label을 제공하지만, consumer wearable 환경과는 signal quality와 motion condition이 다르다.

넷째, feature extraction에 의존한다. 논문은 12개 feature를 correlation analysis로 선택했다고 설명하지만, correlation analysis의 기준, candidate feature pool, feature selection 과정, train-test separation 내에서 feature selection이 수행되었는지는 제공된 텍스트만으로 확인할 수 없다. Feature selection이 전체 데이터에 대해 먼저 수행되었다면 data leakage 가능성도 검토해야 한다.

다섯째, 모델 구조의 일부 세부 사항이 불명확하다. 1D convolution embedding, MHA 14 heads, position-wise block, dropout, normalization, time compressor가 설명되지만, layer 수, hidden dimension의 일관된 tensor shape, optimizer, learning rate, batch size, epoch 수, loss function, early stopping 기준은 제공된 텍스트에 포함되어 있지 않다. 특히 “feature-length of 48 and time frame sequence length of 128”이라는 설명은 앞의 $L_{out}=128$, $T=48$ 설명과 혼동될 수 있다.

여섯째, baseline 비교가 충분히 확인되지 않는다. 논문은 state-of-the-art counterparts보다 우수하다고 주장하지만, 제공된 텍스트에는 동일한 MIMIC-IV split에서 비교 모델들을 재학습한 표가 없다. 서로 다른 dataset과 protocol에서 나온 기존 연구 수치와 비교하면 공정성이 떨어질 수 있다.

일곱째, 임상 표준 기반 평가가 제공된 텍스트에서는 확인되지 않는다. AAMI, BHS, ISO 기준은 cuff-less BP estimation에서 중요하다. 그러나 현재 제공된 부분에서는 RMSE, MAE, ME, $R^2$만 언급되며, Bland–Altman plot, error threshold within 5/10/15 mmHg, mean error와 standard deviation, hypertensive range performance 등이 보이지 않는다.

비판적으로 종합하면, TransfoRhythm은 solo PPG 기반 BP estimation에서 매우 흥미롭고 강력한 Transformer framework이다. 그러나 보고된 성능이 매우 높기 때문에, subject-independent split, record-level separation, external validation, overlap 제거, clinical standard evaluation이 반드시 확인되어야 한다. 이러한 검증이 충분하다면 매우 중요한 연구가 될 수 있지만, 제공된 텍스트만으로는 임상 적용 가능성을 확정하기 어렵다.

## 6. 결론

이 논문은 단일 PPG 신호만을 이용한 cuff-less blood pressure estimation을 위해 Transformer 기반 모델인 TransfoRhythm을 제안한다. 이 모델은 MIMIC-IV Waveform Dataset에서 PPG signal을 정제하고, PPG 및 SDPPG cycle에서 12개 morphological feature를 추출한 뒤, 1D convolution embedding과 positional encoding을 통해 Transformer input을 구성한다. 이후 Multi-Head Attention, position-wise feed-forward block, time frame compressor, flattening block을 거쳐 SBP와 DBP를 예측한다.

논문이 보고한 성능은 SBP/DBP에 대해 RMSE $[2.21, 1.84]$ mmHg, MAE $[1.37, 1.06]$ mmHg이다. 이는 solo PPG 기반 혈압 추정에서 매우 낮은 오차이며, 논문은 TransfoRhythm이 기존 state-of-the-art 모델보다 우수하다고 주장한다. 특히 ECG 없이 PPG만 사용하고, 최신 MIMIC-IV dataset을 활용했다는 점은 중요한 기여이다.

이 연구의 주요 기여는 PPG-only BP estimation 문제를 Transformer 기반 time series regression framework로 구성하고, physiological feature extraction과 attention mechanism을 결합했다는 점이다. 또한 MIMIC-IV를 사용했다는 점에서 기존 MIMIC-II/MIMIC-III 중심 연구를 확장한다.

다만 제공된 텍스트가 평가 지표 수식 중간에서 끊겨 있어, 전체 실험 결과와 비교 표를 확인할 수 없다. 특히 subject-wise cross-validation 여부, observation leakage 가능성, feature selection 절차, baseline 재학습 여부, external validation, AAMI/BHS/ISO 기준 평가는 확인이 필요하다. 이러한 정보가 보완된다면 TransfoRhythm은 wearable 및 ICU 환경에서 continuous cuff-less BP monitoring을 위한 중요한 연구 기반이 될 수 있다. 현재 제공된 텍스트만으로는 매우 유망한 결과를 보인 연구로 평가할 수 있지만, 임상 적용 가능성을 확정하기 위해서는 더 엄격한 검증이 필요하다.
