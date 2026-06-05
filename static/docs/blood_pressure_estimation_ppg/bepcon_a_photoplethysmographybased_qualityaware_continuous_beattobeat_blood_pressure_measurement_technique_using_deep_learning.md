# BePCon: A Photoplethysmography-Based Quality-Aware Continuous Beat-to-Beat Blood Pressure Measurement Technique Using Deep Learning

* **저자**: Monalisa Singha Roy, Rajarshi Gupta, Kaushik Das Sharma
* **발표연도**: 2022

## 1. 논문 개요

이 논문은 단일 photoplethysmogram, 즉 PPG 신호만을 이용하여 beat-to-beat, 즉 BtB 방식으로 systolic blood pressure, 즉 SBP와 diastolic blood pressure, 즉 DBP를 연속 측정하는 deep learning 기반 기법인 BePCon을 제안한다. 논문의 핵심 목표는 ambulatory health monitoring 환경에서 움직임으로 인해 PPG가 손상되는 motion artifact, 즉 MA 문제를 고려하면서도, 실시간에 가까운 혈압 예측을 수행할 수 있는 품질 인식형 continuous BtB BP measurement 모델을 개발하는 것이다.

기존의 cuff 기반 oscillometric blood pressure measurement는 임상적으로 널리 사용되는 noninvasive blood pressure, 즉 NIBP 측정 방식이지만, 장시간 착용이 불편하고 연속적인 ambulatory monitoring에는 적합하지 않다. 이 때문에 ECG와 PPG를 함께 사용하는 pulse transit time, 즉 PTT, 또는 pulse arrival time, 즉 PAT 기반 방법이 많이 연구되어 왔다. 그러나 이러한 방법은 여러 sensor를 부착해야 하며, 움직임이 있는 환경에서는 PPG의 fiducial point detection이 불확실해지고 calibration 문제도 남는다.

이 논문이 다루는 연구 문제는 단일 PPG sensor만으로 cuffless, continuous, beat-to-beat BP measurement를 수행할 수 있는가, 그리고 PPG 신호가 motion artifact로 오염된 상황에서 이를 먼저 감지하고 제외함으로써 혈압 예측의 신뢰도를 높일 수 있는가이다. 논문은 이 문제를 해결하기 위해 signal quality assessment, feature extraction, feature selection, temporal convolutional network, 그리고 on-device implementation을 하나의 파이프라인으로 결합한다.

문제의 중요성은 명확하다. BtB BP measurement는 단순한 평균 혈압 측정보다 더 많은 정보를 제공한다. 특히 BP variability는 acute hypertensive patients, ICU 환자, poststroke rehabilitation 환자의 상태 평가에 중요하다. 또한 PPG 기반 혈압 추정은 smart watch와 같은 wearable device에서 단일 optical sensor만으로 구현될 수 있으므로, 장기적인 개인 건강 모니터링에 실용적 가치가 있다. 다만 PPG는 움직임, sensor disconnection, saturation, poor contact 등에 민감하므로, 실제 적용을 위해서는 신호 품질을 고려한 예측 구조가 필요하다. BePCon은 바로 이 지점을 연구의 출발점으로 삼는다.

## 2. 핵심 아이디어

BePCon의 중심 아이디어는 “좋은 품질의 PPG beat만 사용하고, 현재 beat뿐 아니라 직전 beat들의 정보를 함께 사용하여, Temporal Convolutional Network로 현재 beat의 SBP와 DBP를 예측한다”는 것이다. 기존 PPG 기반 BtB BP 연구에서는 많은 경우 수작업으로 고른 좋은 품질의 pulse만 사용하거나, 현재 PPG cycle의 morphological feature에만 의존하였다. 그러나 실제 ambulatory measurement에서는 motion artifact가 자주 발생하므로, 수동으로 clean pulse를 선택하는 방식은 실시간 적용에 적합하지 않다.

논문은 이를 해결하기 위해 먼저 self-organizing map, 즉 SOM 기반 signal quality assessment를 수행한다. SOM은 3초 길이의 PPG segment가 clean인지 corrupted인지 이진 분류한다. 이 단계에서 motion artifact로 오염된 segment를 제거하고, 이후 최소 10개의 consecutive clean segment, 즉 약 30개 beat가 확보되면 beat delineation과 feature extraction을 수행한다. 이 구조는 혈압 예측 모델이 오염된 PPG를 입력으로 받아 잘못된 혈압을 출력하는 상황을 줄이기 위한 장치이다.

두 번째 핵심 아이디어는 current beat만 사용하지 않고, 현재 beat $k$와 직전 세 beat인 $k-1$, $k-2$, $k-3$의 feature를 함께 사용한다는 것이다. 저자들은 pulse wave generation과 arterial network를 통한 propagation 과정 때문에 현재 beat의 BP가 앞선 PPG cycle들과도 상관관계를 가질 수 있다고 보았다. 논문에 따르면, 이런 preceding beat history를 BtB BP prediction에 명시적으로 활용한 연구는 기존에 충분히 탐색되지 않았다.

세 번째 핵심 아이디어는 feature set을 다양하게 구성하되, 실시간 구현을 위해 recursive feature elimination, 즉 RFE로 feature 수를 줄이는 것이다. 각 PPG beat에서 time-domain feature, wavelet-based feature, statistical feature, stacked denoising autoencoder feature를 합쳐 총 50개 feature를 추출한다. 네 beat를 사용하므로 초기 combined feature set은 총 200개이다. 이후 RFE와 random forest regressor를 결합하여 각 beat당 20개, 총 80개의 optimal feature를 선택한다. 이 과정을 통해 예측 정확도와 계산 효율성을 동시에 확보하려 한다.

마지막 핵심 아이디어는 sequence modeling에 Temporal Convolutional Network, 즉 TCN을 사용한다는 것이다. TCN은 dilated causal convolution과 residual block을 이용하여 과거 beat 정보를 효율적으로 처리한다. LSTM 같은 recurrent architecture와 달리 convolution 기반이므로 병렬화와 안정적인 학습에 유리하고, dilation factor를 통해 넓은 temporal receptive field를 확보할 수 있다. 이 논문은 단일 PPG sensor, quality-aware filtering, current plus preceding beat feature, RFE, TCN, 그리고 standalone hardware implementation을 결합한 점을 주요 novelty로 제시한다.

## 3. 상세 방법 설명

BePCon의 전체 파이프라인은 PPG preprocessing, PPG segment quality evaluation, beat delineation, feature extraction, feature selection, TCN 기반 BtB BP prediction, 그리고 on-device implementation으로 구성된다. 논문에서 제시된 signal processing flow에 따르면, raw PPG는 먼저 filtering을 거친다. 이후 3초 segment 단위로 clean 또는 corrupted 여부를 판정한다. clean segment가 충분히 누적되면 PPG beat를 분리하고, 현재 beat와 직전 세 beat에서 feature를 추출한다. RFE를 통해 feature를 줄인 뒤, reduced feature set을 TCN에 입력하여 현재 beat의 SBP와 DBP를 예측한다.

### 3.1 PPG preprocessing

raw PPG data는 먼저 third-order Butterworth low-pass filter를 통과한다. cutoff frequency는 4 Hz로 설정되어 있으며, 목적은 high-frequency noise를 제거하는 것이다. 논문은 filter의 amplitude response를 다음과 같이 제시한다.

$$
H[z]=\frac{0.0008391+0.002517z^{-1}+0.002517z^{-2}+0.0008391z^{-3}}{1-2.599z^{-1}+2.274z^{-2}-0.6684z^{-3}}
$$

이 filtering 단계는 PPG waveform의 고주파 잡음을 줄여 이후 signal quality assessment와 beat delineation이 안정적으로 수행되도록 한다. 논문은 이 filter를 통해 모든 motion artifact가 제거된다고 주장하지는 않는다. motion artifact detection은 별도의 SQA 단계에서 수행된다.

### 3.2 PPG segment quality evaluation

BePCon은 PPG quality assessment를 혈압 예측 전에 수행한다. 3초 PPG segment마다 네 가지 feature를 추출하여 self-organizing map, 즉 SOM에 입력한다. 사용된 feature는 approximate entropy, spectral entropy, Hjorth complexity, Higuchi fractal dimension이다. 이 feature들은 PPG의 irregularity, complexity, fractal-like behavior, spectral disorder를 반영하며, clean signal과 motion-corrupted signal을 구분하기 위해 사용된다.

SOM은 unsupervised learning 기반의 fully connected 구조로 설명된다. 각 iteration $c$에서 input vector $I$와 SOM node weight vector $w$ 사이의 거리 $d_j$를 계산하고, 이 거리를 최소화하도록 weight를 조정한다. 논문에서 제시된 식은 다음과 같다.

$$
d_j(c)=\sum_{i=1}^{4}(I_i-w_{ij}(c))^2
$$

여기서 $I_i$는 네 개 quality feature 중 $i$번째 feature이고, $w_{ij}(c)$는 iteration $c$에서 node $j$에 대응하는 weight이다. 입력과 가장 가까운 output neuron이 winning neuron으로 선택된다. 이 과정을 통해 PPG segment는 clean 또는 corrupted class로 재구성된다. 이 단계의 목적은 motion artifact가 포함된 segment를 BP measurement model에 넣지 않도록 하여 예측 신뢰도를 높이는 것이다.

### 3.3 PPG beat delineation

beat delineation은 최소 10개의 consecutive clean PPG segment가 stack buffer에 누적되었을 때 시작된다. 논문은 3초 segment 10개를 약 30 beats로 보고 있으며, 이렇게 모은 clean PPG data에서 systolic peak와 foot을 찾아 PPG beat를 분리한다. 이를 위해 empirical mode decomposition 이후 second derivative-based approach가 사용되었다고 설명한다.

혈압 예측에는 현재 timestamp $k$의 PPG beat뿐 아니라, 직전 세 beat인 $k-1$, $k-2$, $k-3$이 함께 사용된다. 이 네 beat의 feature를 이용하여 현재 beat의 SBP $bp_s$와 DBP $bp_d$를 예측한다. 즉 모델은 단일 beat의 모양만 보는 것이 아니라, 짧은 beat sequence의 context를 함께 보는 구조이다.

### 3.4 Feature extraction

각 PPG beat에서는 네 종류의 feature가 추출된다. 첫 번째는 time-domain feature이다. second-order derivative 기반 접근으로 PPG cycle의 fiducial point를 찾고, systolic peak height, diastolic peak height, dicrotic notch height, pulse interval, time interval between peaks의 다섯 feature를 계산한다. 이 feature들은 PPG beat의 기본적인 morphology와 timing 정보를 반영한다.

두 번째는 wavelet-based feature이다. 각 PPG beat는 Daubechies 2, 즉 Db2를 사용한 discrete wavelet transform, 즉 DWT로 6단계까지 분해된다. 이후 subband별 relative energy contribution을 계산한다. 논문에서 제시한 식은 다음과 같다.

$$
E_k=\left(\frac{\sum_k c_k^2}{\sum_j\sum_k c_{kj}^2}\right)
$$

여기서 $E_k$는 $k$번째 subband의 energy contribution이고, $c$는 wavelet coefficient이며, $j$에 대한 summation은 subband 전체에 대한 합을 의미한다. 논문에서는 maximum energy가 subband 3부터 6 사이에서 발견되었다고 설명한다. 이후 coefficient array를 amplitude 기준으로 내림차순 정렬하고, 처음 15개 값을 선택하여 wavelet feature set을 구성한다. 이 feature는 PPG beat의 time-frequency 특성을 반영한다.

세 번째는 statistical feature이다. 각 PPG beat array $x_p$에서 standard deviation, mean absolute deviation, skewness, kurtosis, interquartile range, approximate entropy, spectral entropy, Hjorth complexity, Higuchi fractal dimension, detrended fluctuation analysis를 추출한다. 앞의 다섯 feature는 waveform의 통계적 형태와 분포를 나타내고, 뒤의 feature들은 signal irregularity와 complexity를 나타낸다.

네 번째는 stacked denoising autoencoder, 즉 SDAE 기반 feature이다. SDAE는 PPG pulse의 nonlinear representation을 학습하기 위해 사용된다. 일반 autoencoder와 달리 denoising autoencoder는 입력에 stochastic noise를 추가한 corrupted input $\tilde{x}$를 사용하며, 원래 입력을 복원하도록 학습한다. 논문은 encoder의 hidden representation과 decoder의 reconstruction을 다음과 같이 표현한다.

$$
f_k=\phi_e\left(\sum_k w_e \tilde{x}_k+b_e\right)
$$

$$
z_k=\phi_d\left(\sum_k w_d f_k+b_d\right)
$$

여기서 $e$와 $d$는 각각 encoder와 decoder를 의미하고, $\phi(\cdot)$는 sigmoid nonlinear activation function을 의미한다. $f_k$는 $k$번째 beat의 compressed representation이며, reconstruction error, 즉 RMSE를 최소화하도록 weight $w$와 bias $b$가 조정된다. 각 beat에서 SDAE feature는 20개 추출된다.

따라서 한 beat에서 추출되는 feature 수는 time-domain 5개, wavelet 15개, statistical 10개, SDAE 20개로 총 50개이다. 현재 beat와 직전 세 beat를 합치면 총 200개 feature가 된다. 논문은 combined feature set을 다음과 같이 표현한다.

$$
F_{comb}=[{f_t,f_w,f_s,f_{ae}}*k,\ldots,{f_t,f_w,f_s,f*{ae}}_{k-3}]
$$

여기서 $k$는 current timestamp를 나타낸다. 각 feature type은 전체 data length에 대해 0부터 1 사이로 normalized된다.

### 3.5 Recursive feature elimination을 이용한 feature selection

BePCon은 실시간 구현을 염두에 둔 모델이므로, 200개 feature를 그대로 사용하는 대신 RFE를 통해 feature 수를 줄인다. RFE는 wrapper-based feature selection 방법으로, 모델 성능에 덜 중요한 feature를 반복적으로 제거하면서 최적의 feature subset을 찾는다. 이 논문에서는 random forest regressor를 estimator로 사용하여 RFE를 수행한다.

논문은 각 PPG beat당 최종적으로 20개 feature를 선택한다고 설명한다. 네 beat를 사용하므로 최종 입력 feature 수는 총 80개가 된다. RFE 과정은 subset feature를 선택하고, regressor를 학습한 뒤, RMSE가 개선되거나 유지되는지를 확인하면서 가장 덜 중요한 feature를 제거하는 방식으로 설명된다. 선택된 중요한 feature들의 index는 significance map에 기록된다.

이 단계의 의미는 두 가지이다. 첫째, 불필요하거나 noise에 취약한 feature를 줄여 prediction accuracy를 높일 수 있다. 둘째, on-device implementation에서 latency와 memory usage를 줄이는 데 도움이 된다. 논문 결과에서도 beat당 5개, 10개, 15개, 20개 feature를 비교했을 때, 20개 feature를 사용할 때 SBP와 DBP의 SD가 가장 낮아 성능이 가장 좋았다고 보고한다.

### 3.6 TCN 기반 beat-to-beat BP prediction

최종 reduced feature set은 1-D Temporal Convolutional Network에 입력된다. TCN은 sequence modeling을 위해 사용되며, 논문은 최근 연구에서 TCN이 LSTM 같은 recurrent architecture보다 더 좋은 sequence modeling 성능을 보일 수 있다고 언급한다.

BePCon의 TCN 구조는 1-D convolutional layer, 네 개 stacked residual block, 두 개 fully connected layer로 구성된다. 각 residual block에는 두 개의 dilated causal convolution layer와 nonlinearity가 포함된다. 각 dilated convolution 이후에는 dropout이 추가되어 training step마다 regularization을 수행한다. residual block의 input과 output width가 다를 수 있기 때문에, width를 맞추기 위해 $1 \times 1$ convolution이 추가된다.

TCN의 핵심은 dilated causal convolution이다. causal convolution은 현재 시점의 예측이 미래 입력에 의존하지 않도록 하는 구조이다. 즉 현재 beat의 BP를 예측할 때 이후 beat 정보를 사용하지 않는다. dilated convolution은 convolution kernel 사이에 간격을 두어, 적은 layer 수로도 더 넓은 temporal context를 볼 수 있게 한다. 논문에서 dilated causal convolution은 다음과 같이 표현된다.

$$
C(k)=\sum_{q=0}^{n-1}l(q)F_{k-d\cdot q}
$$

여기서 $l$은 convolution kernel, $n$은 kernel size, $d$는 dilation factor, $F$는 input feature vector이다. dilation factor $d$는 2의 지수 형태로 증가한다. 이것은 adjacent filter tap 사이의 fixed time step이 점점 커지는 것과 같으며, TCN이 가까운 beat뿐 아니라 더 넓은 beat history를 고려할 수 있게 한다.

전체 TCN model은 다음과 같이 표현된다.

$$
bp_s,bp_d=\Phi[F_{k-3},F_{k-2},F_{k-1},F_k]
$$

여기서 $F_k=[f_1,f_2,\ldots,f_{20}]$는 $k$번째 PPG beat에서 선택된 20개 optimal feature이고, $\Phi$는 TCN 전체 model function이다. 이 식은 현재 beat의 SBP와 DBP가 현재 beat와 직전 세 beat의 feature sequence로부터 예측된다는 것을 명확히 보여준다.

## 4. 실험 및 결과

실험에는 PhysioNet의 MIMIC-II/III waveform database에서 가져온 single-channel PPG record 150개가 사용되었다. sampling frequency는 125 Hz이며, 총 duration은 65,000분이다. 이 데이터셋은 다양한 나이, 성별, 혈압 범위를 가진 subject의 physiological signal을 포함한다. TCN 학습의 ground truth로는 같은 데이터셋의 invasive catheter-based continuous ABP signal이 사용되었다. 논문은 MIMIC-II/III waveform database의 technical limitation으로 interchannel delay가 존재한다고 언급하지만, 이 delay가 결과에 미친 영향을 정량적으로 별도 분석하지는 않는다.

데이터는 train과 test를 50:50 비율로 나누었고, algorithm validation에는 tenfold cross validation이 사용되었다. 논문은 blind test result를 제시한다고 설명한다. 평가 지표로는 RMSE, standard deviation, 즉 SD, mean error, 즉 ME, 그리고 MAE가 사용되었다. 특히 AAMI standard에서는 ME와 SD가 중요하게 사용된다.

먼저 PPG signal quality assessment 성능을 평가하였다. 약 50명의 subject에서 임의로 선택한 약 2분 길이의 representative PPG duration을 사용하였고, ground truth annotation은 두 명의 qualified cardiologist가 수행하였다. 두 annotator 사이에 conflict가 있는 segment는 연구에서 제외되었다. SQA binary classifier는 motion artifact corrupted segment를 검출하는 데 sensitivity 100%, positive predictive value 92.86%, accuracy 93.8%를 달성하였다. 이 결과는 BePCon의 전처리 단계가 motion-corrupted segment를 효과적으로 제거할 수 있음을 보여준다.

혈압 예측 성능에서는 BePCon이 전체적으로 매우 낮은 error를 보였다. 논문 abstract와 conclusion에 따르면, 150개 record에 대한 BePCon의 전체 성능은 SBP에 대해 SD 3.24 mmHg, MAE 2.38 mmHg, RMSE 3.32 mmHg이며, DBP에 대해 SD 1.73 mmHg, MAE 1.23 mmHg, RMSE 1.78 mmHg이다. 이는 beat-to-beat BP measurement에서 상당히 낮은 오차이며, 논문은 BePCon이 AAMI standard와 BHS Grade A standard를 모두 만족한다고 주장한다.

혈압 범위별 분석도 수행되었다. 논문은 SBP와 DBP를 healthy, prehypertension, hypertension 세 범위로 나누어 performance를 제시하였다. healthy는 $SBP \le 120$ 및 $DBP \le 80$, prehypertension은 $120 < SBP < 140$ 및 $80 < DBP < 90$, hypertension은 $SBP > 140$ 및 $DBP > 90$으로 정의된다. 결과적으로 error는 normotensive subject에서 가장 낮고, hypertensive group에서 가장 높았다. 이는 고혈압 범위에서 PPG morphology와 BP relationship이 더 복잡하거나, 데이터 분포 또는 sample 수의 영향으로 예측이 어려워질 수 있음을 시사한다. 다만 논문은 이 현상의 정확한 생리적 원인을 추가 실험으로 입증하지는 않는다.

Bland–Altman plot을 이용한 분석도 제시되었다. 논문은 best performance volunteer와 worst performance volunteer에 대해 SBP 및 DBP의 actual value와 estimated value 사이의 difference를 평균값에 대해 표시하였다. Best performance case에서는 SBP의 MAE, SD, correlation coefficient가 각각 1.99 mmHg, 2.69 mmHg, 0.82였고, DBP는 각각 0.93 mmHg, 1.23 mmHg, 0.78이었다. Worst performance case에서는 SBP의 MAE, SD, correlation coefficient가 각각 2.37 mmHg, 6.23 mmHg, 0.57이었고, DBP는 각각 1.24 mmHg, 2.93 mmHg, 0.61이었다. 논문은 limits of agreement가 95% confidence interval 내에 있으며, mean difference ±1.96 SD 기준에서 AAMI standard를 만족한다고 설명한다.

SQA의 기여를 확인하기 위한 ablation 성격의 비교도 수행되었다. 논문은 MIMIC-II/III waveform database에서 임의로 선택한 10개 patient record의 2분 segment를 사용하여, SQA를 사용하는 경우와 사용하지 않는 경우를 비교하였다. 이 segment들은 clean zone과 motion artifact corrupted zone을 모두 포함한다. SQA 없이 예측할 때 평균 SD는 SBP 9.55 mmHg, DBP 8.32 mmHg였지만, SQA를 사용하면 각각 4.00 mmHg와 2.08 mmHg로 줄어들었다. 논문은 SQA 사용으로 SBP accuracy가 19.56%, DBP accuracy가 24.61% 향상되었다고 보고한다. 이 결과는 BePCon의 quality-aware design이 단순한 부가 기능이 아니라 실제 BP prediction accuracy 개선에 중요한 역할을 한다는 것을 보여준다.

Feature selection의 효과도 분석되었다. 논문은 RFE로 beat당 5개, 10개, 15개, 20개 feature subset을 구성하여 성능을 비교하였다. 그 결과 beat당 20개 optimal feature를 사용할 때 SBP와 DBP의 SD가 가장 작았고, 최종 BePCon 설정으로 선택되었다. 또한 세 개보다 많은 preceding PPG beat history를 사용하는 것은 유의미한 성능 개선을 만들지 못했다고 보고한다. 따라서 논문은 current beat와 preceding three beats를 사용하는 구조를 선택하였다.

기존 연구와의 비교에서 BePCon은 MIMIC-II/III waveform database를 사용한 PPG-based BtB BP measurement 연구들보다 SD, MAE, RMSE 측면에서 우수한 성능을 보인다고 주장한다. 저자들은 기존 연구 중 다수가 morphological feature에 의존했고, motion artifact가 있는 경우 measurement error가 커질 수 있다고 지적한다. 또한 기존 연구들은 SQA를 BP estimation 전에 수행하지 않았거나, manual checking에 의존했으며, hardware implementation까지 제시한 경우가 제한적이었다고 설명한다. BePCon은 SQA, compact feature set, TCN, 그리고 standalone target device implementation을 함께 제공한다는 점에서 차별화된다.

On-device implementation도 중요한 실험 결과이다. BePCon은 Raspberry Pi Zero W 환경에서 검증되었다. 이 장치는 single-core 1-GHz ARM v6 controller와 512-MB RAM을 가진다. 구현은 Python 환경에서 PyCharm IDE, TensorFlow, Keras 등을 사용하여 수행되었다. PPG data는 on-board SD card에서 직접 읽고, predicted BtB BP values도 같은 장치에 저장하였다. 실행 시간은 SBP와 DBP를 예측하는 데 약 2.5초 per beat였고, memory engagement는 약 32.22 kB per beat였다. 논문은 이 결과를 통해 BePCon이 real-time ambulatory BtB BP measurement에 잠재력이 있음을 주장한다. 다만 latency는 더 높은 사양의 controller를 사용하면 개선될 수 있지만, power consumption이 증가할 수 있다고 언급한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 motion artifact 문제를 BP prediction pipeline의 핵심 요소로 다루었다는 점이다. 많은 PPG 기반 혈압 추정 연구는 clean signal만을 사용하거나 수동으로 quality control을 수행하는 경우가 많다. 그러나 실제 wearable 또는 ambulatory 환경에서는 움직임으로 인해 PPG가 쉽게 오염된다. BePCon은 SOM 기반 SQA를 통해 corrupted segment를 먼저 제거하고, clean segment만을 사용해 BP를 예측함으로써 실제 사용 환경에 더 가까운 문제 설정을 다룬다. SQA 사용 시 SBP와 DBP의 accuracy가 각각 19.56%, 24.61% 향상되었다는 결과는 이 설계의 타당성을 뒷받침한다.

두 번째 강점은 단일 PPG sensor만 사용하는 구조이다. ECG–PPG 기반 PTT 또는 PAT 방법은 성능상 장점이 있을 수 있지만, 여러 sensor attachment가 필요하다. BePCon은 single-channel PPG만을 사용하므로 smartwatch나 wearable optical sensor 기반 시스템에 더 쉽게 적용될 수 있다. 특히 on-device implementation까지 제시했다는 점은 단순한 offline algorithm 연구를 넘어 embedded health monitoring application을 염두에 둔 연구라는 점에서 중요하다.

세 번째 강점은 feature engineering과 deep learning을 균형 있게 결합했다는 점이다. BePCon은 end-to-end raw signal model은 아니지만, time-domain, wavelet, statistical, SDAE feature를 함께 사용하여 PPG beat의 morphology, frequency structure, statistical irregularity, nonlinear representation을 폭넓게 반영한다. 이후 RFE로 feature 수를 줄이고, TCN으로 temporal relationship을 모델링한다. 이는 제한된 embedded device 환경에서 정확도와 계산 효율을 동시에 고려한 설계로 볼 수 있다.

네 번째 강점은 현재 beat뿐 아니라 preceding three beats를 포함했다는 점이다. 혈압과 PPG waveform은 beat 단위로 완전히 독립적이지 않다. 혈관 탄성, pulse wave propagation, hemodynamic state는 시간적으로 연속적이기 때문에 직전 beat 정보가 현재 beat BP prediction에 도움이 될 수 있다. BePCon은 이 정보를 TCN 구조로 처리함으로써 BtB BP measurement에 적합한 temporal modeling을 수행한다.

다섯 번째 강점은 AAMI와 BHS라는 국제적 평가 기준을 사용했다는 점이다. 논문은 BePCon이 AAMI standard를 만족하고 BHS Grade A에 해당한다고 보고한다. 또한 Bland–Altman plot, BP range별 performance, SQA ablation, feature subset 비교, 기존 연구 비교, on-device latency와 memory evaluation까지 다양한 관점에서 결과를 제시하였다.

그러나 한계도 존재한다. 첫째, BePCon은 minimum four consecutive good quality PPG beats가 필요하다. 논문에서는 실제 segment measurement를 위해 약 30개 consecutive good quality beats를 사용했다고 설명한다. 따라서 motion artifact가 지속적으로 발생하거나 clean PPG가 충분히 확보되지 않는 상황에서는 beat-to-beat measurement가 지연되거나 중단될 수 있다. 이는 ambulatory 환경에서 중요한 제약이다.

둘째, 논문은 BePCon이 personalized model이라고 명시한다. PPG morphology는 다양한 physiological factor에 따라 크게 달라질 수 있다. 개인마다 혈관 탄성, 피부 특성, sensor contact, 혈류 반응, 질환 상태가 다르기 때문에, 단일 모델이 모든 사용자에게 동일하게 잘 동작한다고 보기 어렵다. 논문은 personalization 문제를 언급하지만, 개인별 calibration이 얼마나 필요한지, 새로운 subject에 대한 generalization 성능이 어느 정도인지에 대한 상세한 분석은 제한적이다.

셋째, 데이터셋은 MIMIC-II/III waveform database에 기반하며, ground truth로 invasive ABP signal을 사용한다는 장점이 있지만, MIMIC 데이터에는 interchannel delay라는 technical limitation이 존재한다고 논문이 직접 언급한다. PPG와 ABP 사이의 delay는 beat alignment와 BP label assignment에 영향을 줄 수 있다. 논문은 이 문제를 언급하지만, delay correction이 어떻게 수행되었는지 또는 delay가 성능에 어떤 영향을 미쳤는지에 대한 상세한 정량 분석은 제공하지 않는다.

넷째, SQA ground truth annotation에서 두 cardiologist 사이에 conflict가 있는 segment는 제외되었다. 이는 classifier training과 evaluation을 더 명확하게 만들 수 있지만, 실제 ambiguous quality segment를 어떻게 처리할지는 별도 문제이다. 실제 wearable 환경에서는 clean과 corrupted 사이의 경계가 모호한 segment가 빈번할 수 있으므로, conflict segment를 제외한 평가가 실제 난이도를 다소 낮게 만들었을 가능성이 있다.

다섯째, BePCon은 feature extraction에 상당히 많은 단계가 필요하다. Time-domain fiducial point detection, DWT, statistical complexity measures, SDAE feature extraction, RFE-selected feature mapping, TCN prediction이 모두 포함된다. 논문은 on-device implementation에서 낮은 memory와 2.5초 per beat latency를 보였다고 보고하지만, 다양한 wearable processor, battery-powered environment, real-time sensor streaming 환경에서의 안정성은 추가 검증이 필요하다.

여섯째, end-to-end learning 방식과의 직접 비교가 충분하지 않다. BePCon은 handcrafted feature와 learned SDAE feature를 결합한 hybrid feature-based TCN 모델이다. 이 접근은 해석 가능성과 계산 효율에서 장점이 있지만, raw PPG waveform을 직접 입력으로 사용하는 modern deep learning model과 공정하게 비교했는지는 제공된 텍스트만으로는 확인하기 어렵다. 논문은 published works와 비교하지만, 동일한 train-test split, 동일한 preprocessing, 동일한 subject split 조건에서의 비교인지 여부는 명확하지 않다.

마지막으로, blood pressure range별 결과에서 hypertensive group의 error가 가장 높았다. 실제 임상적으로 중요한 대상은 고혈압 환자나 급격한 혈압 변화가 있는 환자일 가능성이 크다. 따라서 BePCon이 정상 혈압 범위에서는 매우 우수한 성능을 보이더라도, hypertension 또는 unstable hemodynamic condition에서의 성능 개선은 향후 연구에서 더 중요하게 다루어져야 한다.

## 6. 결론

이 논문은 단일 PPG sensor 기반의 quality-aware continuous beat-to-beat blood pressure measurement 기법인 BePCon을 제안하였다. BePCon은 PPG signal quality assessment, beat delineation, 다양한 feature extraction, RFE 기반 feature selection, TCN 기반 sequential BP prediction을 결합한 구조이다. 특히 motion artifact가 포함된 PPG segment를 SOM 기반 SQA로 먼저 걸러내고, 현재 beat와 직전 세 beat의 feature를 이용하여 현재 beat의 SBP와 DBP를 예측한다는 점이 핵심 기여이다.

실험 결과 BePCon은 MIMIC-II/III waveform database의 150개 record에서 SBP에 대해 SD 3.24 mmHg, MAE 2.38 mmHg, RMSE 3.32 mmHg를 달성했고, DBP에 대해 SD 1.73 mmHg, MAE 1.23 mmHg, RMSE 1.78 mmHg를 달성하였다. 또한 AAMI standard와 BHS Grade A standard를 만족한다고 보고되었다. SQA를 사용하면 motion-corrupted PPG에서 BP measurement accuracy가 SBP 19.56%, DBP 24.61% 향상되었다. 이는 PPG 기반 혈압 예측에서 signal quality control이 성능에 큰 영향을 미친다는 점을 강하게 보여준다.

실제 적용 측면에서도 BePCon은 의미 있는 결과를 제시한다. Raspberry Pi Zero W와 같은 single-core 1-GHz ARM v6 controller, 512-MB RAM 환경에서 약 2.5초 per beat latency와 약 32.22 kB per beat memory requirement로 동작하였다. 이는 deep learning 기반 기법임에도 embedded device에서 구현 가능성이 있음을 보여준다. 따라서 BePCon은 smartwatch, wearable monitor, portable healthcare device에서 cuffless continuous BP monitoring을 수행하기 위한 후보 기술로 볼 수 있다.

다만 실제 임상 및 상용 적용을 위해서는 추가 검증이 필요하다. BePCon은 consecutive good quality beats가 확보되어야 하며, PPG morphology의 개인차 때문에 personalized model이라는 한계를 가진다. 또한 MIMIC waveform의 interchannel delay, ambiguous quality segment 처리, hypertensive group에서 증가하는 error, 다양한 외부 데이터셋에서의 generalization 성능은 더 깊이 검증되어야 한다.

종합하면, 이 논문은 PPG 기반 BtB BP measurement에서 motion artifact quality assessment와 TCN-based temporal modeling을 결합한 실용적이고 성능 중심적인 연구이다. 단일 PPG sensor만으로 높은 정확도를 달성하고, on-device implementation까지 제시했다는 점에서 wearable continuous blood pressure monitoring 연구에 중요한 기여를 한다. 향후에는 더 다양한 환자군, 실제 ambulatory setting, 장기 착용 환경, 개인화 calibration 전략, 그리고 domain generalization을 포함한 검증이 이루어진다면, BePCon과 같은 접근은 cuffless BP monitoring의 실제 적용 가능성을 더욱 높일 수 있을 것이다.
