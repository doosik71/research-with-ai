# OPTIMIZATION OF DEEP NEURAL NETWORKS FOR PPG-BASED BLOOD PRESSURE ESTIMATION ON EDGE DEVICES

* **저자**: Francesco Carlucci
* **발표연도**: 2024

## 1. 논문 개요

이 논문은 Photoplethysmography, 즉 PPG 신호를 이용한 non-invasive continuous blood pressure estimation을 edge device, 특히 smartwatch와 같은 ultra-low-power wearable device에서 실행 가능하도록 Deep Neural Network, DNN을 자동 최적화하는 연구이다. 논문의 핵심 목표는 혈압 추정 정확도를 유지하거나 개선하면서도, DNN의 memory footprint, parameter 수, operation 수, latency, energy consumption을 줄여 GAP8과 같은 저전력 multicore System-on-Chip, SoC에 배포 가능한 모델을 만드는 것이다.

연구 문제는 단순히 PPG로 혈압을 예측하는 것에 그치지 않는다. 이미 여러 deep learning 모델들이 PPG-based BP estimation에서 기존 machine learning 방법보다 좋은 성능을 보였지만, 이 모델들은 parameter 수와 연산량이 커서 실제 wearable hardware에 탑재하기 어렵다. 따라서 이 논문의 중심 질문은 “state-of-the-art 수준의 정확도를 유지하면서, ultra-low-power edge device의 제한된 메모리와 에너지 예산 안에서 동작할 수 있는 DNN을 자동으로 설계할 수 있는가”이다.

이 문제는 실제 의료·헬스케어 응용에서 매우 중요하다. Blood pressure, BP는 hypertension, stroke, heart failure, atrial fibrillation, dementia 등 여러 cardiovascular diseases, CVD의 핵심 위험 지표이다. 특히 hypertension은 증상이 거의 없어 “silent killer”라고 불리며, 조기 진단과 지속적인 모니터링이 중요하다. 기존 cuff-based sphygmomanometer는 정확하지만 연속 측정이 어렵고, arterial cannulation은 연속적이고 정확하지만 침습적이다. 반면 PPG는 LED와 photodiode만으로 혈액량 변화를 측정할 수 있어 wearable device에 적합하다. 하지만 PPG에서 SBP와 DBP를 정확히 추정하는 것은 physiological complexity, 개인차, motion artifact, signal variability 때문에 여전히 어려운 문제이다.

논문은 이러한 배경에서 automated DNN optimization pipeline을 제안한다. 이 pipeline은 HW-aware Neural Architecture Search, NAS, Pruning In Time, PIT, Quantization Aware Training, QAT를 포함한다. 구체적으로 PLiNIO library의 SuperNet, PIT, Mixed Precision Search, MPS를 사용하여 기존 state-of-the-art seed model인 ResNet, UNet, 그리고 추가적으로 TEMPONet을 최적화한다. 최종적으로 일부 모델은 GAP8 SoC에 배포되어 memory, latency, energy consumption을 평가한다.

논문이 다루는 출력 목표 변수는 systolic blood pressure, SBP와 diastolic blood pressure, DBP이다. 접근 방식은 크게 두 가지이다. 첫 번째는 PPG signal window에서 SBP와 DBP scalar 값을 직접 예측하는 signal-to-label, Sig2Lab regression이다. 두 번째는 PPG signal에서 arterial blood pressure, ABP waveform 전체를 복원한 뒤 peak detection을 통해 SBP와 DBP를 추출하는 signal-to-signal, Sig2Sig reconstruction이다. 이 논문은 두 접근법 모두를 고려하고, 각 모델 계열을 edge deployment 관점에서 최적화한다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 PPG-based blood pressure estimation을 “정확도 문제”와 “edge deployment 문제”를 동시에 만족해야 하는 multi-objective optimization 문제로 다루는 것이다. 기존 연구들은 주로 Mean Absolute Error, MAE를 낮추는 데 집중했지만, 실제 wearable device에서는 모델이 메모리에 들어가야 하고, 낮은 latency와 낮은 energy consumption으로 반복 실행되어야 한다. 따라서 정확도만 높은 모델은 실용적이지 않을 수 있다.

논문은 이 문제를 해결하기 위해 gradient-based NAS를 사용한다. SuperNet은 기존 seed architecture의 각 convolutional layer를 여러 후보 operation의 조합으로 바꾼다. 후보는 original 1D convolution, depthwise-separable convolution, identity operation이다. Original convolution은 표현력이 높지만 parameter와 연산량이 크다. Depthwise-separable convolution은 MobileNet 계열에서 널리 사용된 경량 convolution으로, standard convolution보다 훨씬 적은 parameter로 유사한 feature extraction을 수행할 수 있다. Identity operation은 layer를 제거하거나 skip하는 효과를 가지므로 network depth를 줄일 수 있다. 이 후보들 중 어떤 것을 선택할지 architecture parameter $\theta$를 통해 학습한다.

이 논문의 두 번째 핵심 아이디어는 SuperNet으로 coarse architecture search를 수행한 뒤, Pruning In Time으로 더 세밀한 구조 최적화를 수행하는 것이다. SuperNet은 layer 단위에서 어떤 operation을 선택할지 결정한다. 반면 PIT는 convolutional layer 내부의 number of features, kernel size, dilation 등을 mask 기반으로 최적화한다. 즉, SuperNet이 “큰 구조”를 바꾸는 단계라면, PIT는 “선택된 구조 안에서 불필요한 channel과 temporal receptive field를 줄이는 단계”라고 볼 수 있다. 이를 통해 SuperNet만으로는 도달하지 못한 더 작은 모델 또는 더 정확한 모델을 만들 수 있다.

세 번째 핵심 아이디어는 Quantization Aware Training을 통해 모델을 int8 형식으로 변환하고, 실제 GAP8 edge hardware에서 실행 가능성을 확인하는 것이다. Edge device에서는 32-bit floating point 연산이 비싸거나 지원되지 않을 수 있다. GAP8은 Floating Point Unit이 없기 때문에 non-quantized fp32 model은 직접 배포할 수 없다. 따라서 8-bit integer quantization은 단순한 압축 기술이 아니라 실제 deployment를 위한 필수 단계이다.

논문은 또한 PPG와 ABP의 관계를 생리학적으로 설명한다. PPG는 혈관 내 blood volume 변화와 관련되고, ABP는 arterial pressure waveform이다. 두 신호는 형태적으로 유사하지만 완전히 동일하지 않다. 혈관의 dynamic compliance, hysteresis, arterial stiffness, peripheral resistance, reflected wave 등의 영향으로 PPG에서 ABP를 추정하는 것은 단순한 선형 변환이 아니다. 따라서 논문은 handcrafted feature나 단순 mathematical model보다 data-driven deep learning이 더 유망하다고 본다. 하지만 deep learning 모델이 실제 wearable로 가려면 반드시 경량화가 필요하다는 것이 이 논문의 중심 문제의식이다.

기존 접근과 비교했을 때 이 논문의 차별점은 다음과 같다. 첫째, PPG-based BP estimation에서 model footprint와 hardware deployment를 정량적으로 고려한다. 둘째, SuperNet, PIT, QAT를 결합한 자동 pipeline으로 여러 Pareto-optimal architecture를 생성한다. 셋째, 네 개의 공개 benchmark dataset을 사용하여 공정한 비교를 수행하고, state-of-the-art 모델들과 동일한 preprocessing 및 validation protocol을 따른다. 넷째, 일부 모델은 실제 GAP8 SoC에 배포하여 latency와 energy까지 측정한다.

## 3. 상세 방법 설명

논문의 전체 방법은 데이터셋 선택과 benchmark 정렬, seed architecture 설정, data augmentation 실험, SuperNet 기반 NAS, PIT 기반 구조 refinement, MPS 기반 quantization, GAP8 deployment의 순서로 구성된다.

먼저 데이터셋은 PPGBP, BCG, SENSORS, UCI 네 가지를 사용한다. 이들은 모두 PPG와 혈압 관련 label을 포함하는 공개 benchmark dataset이다. PPGBP는 218명 또는 219명의 피험자에서 얻은 619개 segment를 포함하는 가장 작은 데이터셋이며, 각 segment 길이는 2.1초이다. 이 데이터셋은 continuous ABP waveform이 아니라 scalar SBP와 DBP 값만 제공하므로 Sig2Sig 모델을 학습할 수 없다. BCG는 40명의 피험자에서 약 4시간 동안 수집된 bed-based ballistocardiography dataset이며, 5초 window를 사용한다. SENSORS는 MIMIC-III subset으로 1195명의 ICU 환자와 11,102개 segment를 포함한다. UCI는 MIMIC-II waveform subset으로 약 410,596개 segment를 포함하는 가장 큰 데이터셋이며, subject identity가 제공되지 않는다.

논문은 benchmark의 preprocessing과 data split protocol을 따른다. PPGBP, BCG, SENSORS는 subject information을 고려한 5-fold cross-validation을 사용한다. UCI는 subject identity가 없기 때문에 blood pressure distribution을 유지하는 Hold-One-Out 또는 held-out validation/test split을 사용한다. 이 점은 중요한데, 혈압 추정에서는 같은 환자의 segment가 training과 test에 동시에 들어가면 모델이 subject-specific pattern을 기억할 수 있어 성능이 과대평가된다. 논문은 이러한 leakage 문제가 기존 연구에서 자주 발생했다고 지적하고, benchmark protocol을 사용함으로써 비교의 공정성을 확보하려 한다.

입력과 출력 형식은 세 가지로 구분된다. Feat2Lab은 handcrafted feature에서 SBP와 DBP 값을 예측하는 방식이며, traditional ML 모델인 SVR이나 Random Forest에서 주로 사용된다. Sig2Lab은 raw PPG signal sample에서 SBP와 DBP scalar 값을 직접 예측하는 방식이며, ResNet과 TEMPONet 같은 CNN 계열 모델이 이에 해당한다. Sig2Sig은 raw PPG signal에서 continuous ABP waveform을 복원하고, 이후 peak detection을 통해 SBP와 DBP를 추출하는 방식이며, UNet, VNet, PPG2ABP 등이 이에 해당한다. 이 논문은 deep learning 모델에 집중하므로 Sig2Lab과 Sig2Sig을 주요 대상으로 삼는다.

Seed architecture는 ResNet, UNet, TEMPONet이다. ResNet은 1D convolution 기반 regression network이다. 입력은 univariate PPG signal이고, 첫 convolution, Batch Normalization, ReLU, max pooling을 거친 뒤 residual block을 반복한다. Residual block에는 convolutional layer, Batch Normalization, ReLU, skip connection이 포함되며, squeeze-and-excitation component도 포함된다. Squeeze-and-excitation은 channel 간 중요도를 학습하는 구조이다. Average pooling으로 각 channel을 하나의 값으로 압축하는 squeeze를 수행하고, fully connected layer와 ReLU, Sigmoid를 거쳐 channel weight를 만든 뒤 원래 feature map에 곱한다. 마지막에는 linear layer가 SBP와 DBP 두 값을 예측한다.

UNet은 1D temporal signal에 맞게 변형된 encoder-decoder architecture이다. Encoder는 convolution을 통해 입력 PPG의 temporal dimension을 점차 줄이면서 low-dimensional embedding을 만들고, decoder는 upsampling을 통해 다시 원래 길이의 ABP waveform을 복원한다. Encoder와 decoder의 대응 layer 사이에는 skip connection이 있어, 고해상도 waveform 정보를 decoder로 전달한다. UNet은 fully connected layer 없이 fully convolutional 구조를 사용하며, Instance Normalization과 PReLU activation을 사용한다. 이 모델은 Sig2Sig 방식이므로 continuous ABP label이 없는 PPGBP에는 사용할 수 없다.

TEMPONet은 Temporal Convolutional Network 계열의 1D CNN이다. PPG-based heart rate estimation에서 사용된 경험이 있으며, temporal data에서 meaningful embedding을 추출할 수 있다는 점에서 후보 seed로 추가되었다. TEMPONet은 repeated block으로 구성되며, 각 block은 1D convolution, ReLU, Batch Normalization을 포함한다. 마지막에는 flattened convolution output을 fully connected layer로 전달하여 SBP와 DBP를 예측한다. 다만 논문 실험에서 TEMPONet은 전반적으로 ResNet보다 낮은 성능을 보였다.

Data augmentation 실험도 수행되었다. 사용한 변환은 jittering, scaling, magnitude warping, time warping이다. Jittering은 각 time step에 Gaussian noise를 더하는 방식이다.

$$
X^{aug}_t = X_t + \mathcal{N}(0,\sigma)
$$

여기서 $X_t$는 원래 signal의 $t$번째 값이고, $\mathcal{N}(0,\sigma)$는 평균 0, 표준편차 $\sigma$의 Gaussian noise이다. 이 변환은 sensor noise에 대한 robustness를 높이려는 목적을 갖는다.

Scaling은 전체 signal amplitude에 Gaussian random factor를 곱한다.

$$
X^{aug} = X \times \mathcal{N}(1,\sigma)
$$

이는 PPG amplitude가 센서 압력, 피부 상태, 측정 위치 등에 따라 달라질 수 있다는 점을 반영한다.

Magnitude warping은 signal 전체에 시간에 따라 변하는 smooth random curve를 곱한다.

$$
X^{aug} = X \times \text{CubicSpline}(0...T,\mathcal{N}(1,\sigma))
$$

이는 signal amplitude가 시간에 따라 완만하게 달라지는 상황을 모사한다.

Time warping은 시간축 자체를 변형한다.

$$
X^{aug} = \text{Interp}(\text{Cumulative}(\text{CubicSpline}(0...T,\mathcal{N}(1,\sigma))), X)
$$

이 변환은 signal의 일부 구간을 압축하거나 늘려 heart rate variation 또는 temporal distortion을 흉내 낸다. 그러나 실험 결과 이러한 data augmentation은 성능을 일관되게 크게 개선하지 못했으며, UCI처럼 큰 데이터셋에는 추가 적용하지 않았다.

핵심 최적화 단계는 PLiNIO 기반 pipeline이다. SuperNet은 각 convolution layer를 여러 alternative branch의 ensemble로 바꾼다. 각 alternative는 같은 입력을 받고, 출력은 architecture parameter $\theta_i$의 softmax weight로 가중합된다. 개념적으로 한 layer의 출력은 다음과 같이 표현할 수 있다.

$$
Y = \sum_i \text{softmax}(\theta_i) O_i(X)
$$

여기서 $O_i$는 original convolution, depthwise-separable convolution, identity operation 중 하나이다. 학습이 끝난 뒤에는 가장 큰 $\theta_i$를 가진 operation만 남기고 나머지는 제거한다.

SuperNet 학습의 목적 함수는 다음과 같다.

$$
\min_{W,\theta} \mathcal{L}(W;\theta) + \lambda \mathcal{R}(\theta)
$$

여기서 $W$는 일반 network weight이고, $\theta$는 architecture parameter이다. $\mathcal{L}(W;\theta)$는 task loss이며, 논문에서는 Mean Squared Error, MSE를 사용한다. $\mathcal{R}(\theta)$는 architecture cost이며, 이 연구에서는 주로 total number of parameters를 의미한다. $\lambda$는 cost penalty의 strength로, 더 큰 $\lambda$는 더 작은 모델을 선호하게 만든다. 따라서 이 목적 함수는 단순히 예측 오차를 줄이는 것이 아니라, 정확도와 모델 크기 사이의 trade-off를 최적화한다.

MSE는 다음과 같이 이해할 수 있다.

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

여기서 $y_i$는 실제 SBP, DBP 또는 ABP waveform label이고, $\hat{y}_i$는 모델 예측값이다. Regression 방식에서는 $y_i$가 scalar SBP 또는 DBP이고, Sig2Sig 방식에서는 시간축 전체의 ABP sample이다.

SuperNet 이후 일부 Pareto-optimal model은 Pruning In Time을 통해 추가 최적화된다. PIT는 mask-based differentiable NAS이다. SuperNet이 operation branch를 고르는 path-based NAS라면, PIT는 convolutional layer 내부의 channel, kernel size, dilation 등에 mask를 적용하고, 이 mask를 gradient descent로 학습한다. 학습 후 mask를 binarization하여 threshold 이상인 channel이나 temporal component만 유지한다. 즉 PIT는 structural pruning을 수행하여 불필요한 channel과 convolutional geometry를 제거한다.

PIT의 장점은 SuperNet보다 더 세밀한 search를 수행할 수 있다는 점이다. SuperNet은 정해진 후보 operation 중 선택하지만, PIT는 convolutional layer의 receptive field와 dilation까지 조정할 수 있다. 따라서 temporal convolution에서 어느 정도의 과거 context가 필요한지, 얼마나 많은 channel이 필요한지를 더 정밀하게 줄일 수 있다.

Quantization은 PLiNIO의 Mixed Precision Search, MPS를 통해 수행된다. 이 연구에서는 실제 mixed precision search를 수행했다기보다, 모든 weight와 activation을 8-bit integer로 변환하는 Quantization Aware Training을 수행했다. Weight에는 standard min-max affine quantization이 적용되고, activation에는 Parametrized Clipping Activation, PaCT가 사용된다. Accumulation과 bias는 target inference library인 PULP-NN에서 지원하는 32-bit로 유지된다. QAT는 quantization noise를 학습 중에 모사하여, 단순 post-training quantization보다 accuracy drop을 줄이는 방식이다.

학습 절차는 비교의 공정성을 위해 대부분 고정된다. 모든 PLiNIO experiment는 20 epoch pretraining으로 시작한다. 이후 architecture parameter 또는 pruning mask를 validation set에서 최적화하고, model weight는 training set에서 최적화한다. Weight optimizer와 NAS parameter optimizer는 모두 Adam을 사용하지만 learning rate가 다르다. Network weight의 learning rate는 0.001이고, NAS parameter의 learning rate는 0.01이다. NAS training은 최소 50 epoch, 최대 160 epoch로 설정되며, patience 40의 early stopping이 사용된다. NAS 후 export된 architecture는 200 epoch fine-tuning을 거쳐 최종 평가된다. Strength $\lambda$는 일반적으로 $10^{-11}$부터 $10^{-7}$까지 log scale로 9개 또는 18개 값을 사용한다.

평가 지표는 주로 Mean Absolute Error, MAE이다.

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

MAE는 실제 혈압과 예측 혈압 사이의 평균 절대 오차를 의미한다. 논문은 또한 model footprint를 평가하기 위해 number of parameters와 number of operations를 함께 사용한다. 일부 benchmark에서는 Mean Error와 Standard Deviation, 그리고 Mean Absolute Scaled Error도 언급되지만, 이 논문의 최적화 결과 분석은 MAE와 parameter 수를 중심으로 이루어진다.

## 4. 실험 및 결과

실험은 먼저 state-of-the-art model footprint evaluation에서 시작한다. 기존 benchmark의 deep learning 모델과 traditional machine learning 모델에 대해 parameter 수와 operation 수를 추가로 측정한다. 이 과정에서 ResNet과 UNet이 네 데이터셋에서 일관되게 좋은 성능을 보였기 때문에 주요 NAS seed로 선택된다. MLPBP는 성능이 평균적이지만 parameter 수가 매우 커서 이후 plot에서는 제외된다.

기존 모델 비교에서 traditional ML인 SVR과 Random Forest는 작은 데이터셋인 PPGBP와 BCG에서 매우 강한 성능을 보인다. 예를 들어 PPGBP에서 SVR은 SBP MAE 13.15, DBP MAE 8.04를 기록하고, Random Forest는 SBP MAE 13.17, DBP MAE 8.12를 기록한다. ResNet은 같은 데이터셋에서 SBP MAE 13.402, DBP MAE 8.451이다. 이는 작은 데이터셋에서는 handcrafted feature와 classical regressor가 여전히 DNN보다 유리할 수 있음을 보여준다.

BCG에서도 SVR은 SBP MAE 11.45, DBP MAE 7.34를 기록한다. ResNet은 SBP 11.945, DBP 7.895이고, UNet은 SBP 12.3, DBP 7.98이다. 다만 VNet은 SBP 11.42로 낮은 SBP 오차를 보인다. 이 결과는 작은 데이터셋에서 deep learning이 항상 우세하지 않으며, 데이터 규모와 다양성이 중요함을 보여준다.

SENSORS와 UCI처럼 더 큰 데이터셋에서는 deep learning의 장점이 커진다. SENSORS에서 SVR은 SBP MAE 15.60, DBP MAE 7.50이고, UNet은 SBP 15.64, DBP 7.66으로 비슷한 수준이다. UCI에서는 ResNet이 SBP MAE 16.588로 가장 좋은 SBP deep model이고, UNet이 DBP MAE 7.88로 가장 좋은 DBP deep model이다. Traditional ML은 UCI에서 parameter 규모가 커지고 성능 우위가 줄어든다. 논문은 이를 통해 large dataset에서는 DNN이 classical method를 능가할 가능성이 높다고 해석한다.

TEMPONet 실험에서는 TEMPONet이 PPGBP에서 비교적 좋은 결과를 보였지만, 전체적으로 ResNet을 이기지 못했다. PPGBP에서 TEMPONet은 SBP MAE 15.782, DBP MAE 9.399를 기록했고, ResNet은 SBP 13.402, DBP 8.451로 더 우수했다. BCG에서는 TEMPONet이 특히 DBP task에서 어려움을 보였고, SENSORS와 UCI에서는 평균적인 성능을 보였다. 논문은 TEMPONet이 상대적으로 shallow architecture이기 때문에 작은 데이터셋에서는 어느 정도 작동하지만, 더 복잡한 signal relation을 학습하기에는 ResNet보다 부족하다고 해석한다.

TEMPONet variant R1, R2, R3도 실험되었다. 이들은 더 작은 모델을 만들기 위해 block 수, channel 수, dilation, receptive field 등을 조정한 모델이다. Dropout 없이 R1, R2, R3는 original TEMPONet보다 낮은 성능을 보였다. Dropout 0.2를 마지막 linear layer 이전에 추가했을 때 variant들의 성능은 개선되었고, 일부 경우 TEMPONet에 근접하거나 능가했다. 예를 들어 PPGBP에서 R2는 dropout 적용 후 SBP MAE 15.697, DBP MAE 9.426을 기록하여 original TEMPONet의 SBP 15.782보다 약간 나은 SBP 결과를 보였다. 그러나 여전히 ResNet보다는 낮은 성능이다. 이 결과는 TEMPONet 계열이 overfitting에 취약하며, regularization이 어느 정도 도움을 줄 수 있음을 보여준다.

Data augmentation 실험에서는 PPGBP, BCG, SENSORS에 대해 6× 및 9× augmentation을 적용했다. 사용한 변환은 jittering, scaling, time warping, magnitude warping이다. 결과는 일관되게 크지 않았다. PPGBP에서는 ResNet의 9× augmentation이 SBP MAE를 13.402에서 13.283으로 약간 개선했지만, DBP는 8.451에서 8.46으로 거의 변화가 없었다. BCG에서는 ResNet 9× augmentation이 DBP를 7.895에서 7.682로 개선했지만, SBP는 11.945에서 12.094로 악화되었다. SENSORS에서는 augmentation이 ResNet 성능을 더 뚜렷하게 개선하여 6×에서 SBP 15.865, DBP 7.625, 9×에서 SBP 15.846, DBP 7.602를 기록했다. 그러나 computational cost 대비 효과가 제한적이어서 UCI에는 적용하지 않았다.

SuperNet 결과는 논문의 첫 번째 주요 성과이다. 모든 데이터셋에서 SuperNet은 seed를 지배하거나 memory-error Pareto front 위에 있는 모델을 생성했다. PPGBP에서는 ResNet에서 출발해 rich Pareto curve를 얻었으며, seed size를 16% 줄이면서 DBP MAE는 3.9%, SBP MAE는 3.5%만 증가시켰다. 이는 accuracy를 약간 희생하고 memory footprint를 줄이는 실용적 trade-off를 보여준다. 그러나 PPGBP에서는 여전히 SVR과 Random Forest가 가장 좋은 모델이다.

BCG에서는 SuperNet이 두 seed model을 모두 Pareto-dominate했다. 가장 좋은 UNet-derived model은 SBP MAE 11.139, DBP MAE 7.52를 달성했다. 이는 best seed인 ResNet 대비 SBP에서 6.7%, DBP에서 4.7% 낮은 error이고, parameter 수는 3.8배 줄어든다. 다만 BCG에서도 classical ML 모델은 여전히 강하며, SVR은 DBP MAE 7.34를 달성한다.

SENSORS에서는 SuperNet-optimized DNN이 classical ML을 넘어선다. SVR은 SBP MAE 15.60, DBP MAE 7.50을 기록했지만, UNet NAS model은 SBP MAE 15.51, DBP MAE 7.50을 기록하면서 parameter 수를 최대 40배 줄였다. 이는 정확도를 유지하면서 model size를 크게 줄인 중요한 결과이다.

UCI에서는 SuperNet의 장점이 더 뚜렷하다. UCI는 가장 큰 데이터셋이며, classic ML은 deep learning seed보다 낮은 성능을 보인다. ResNet seed는 SBP MAE 16.59를 기록하고, UNet seed는 DBP MAE 7.88을 기록한다. SuperNet으로 찾은 가장 정확한 모델들은 약 149.8k 또는 156.3k parameters만으로 SBP MAE 16.655와 DBP MAE 7.86을 달성한다. ResNet seed는 SBP에서 더 낮은 MAE를 갖지만 792k parameters를 사용하여 quantization 후에도 GAP8의 512 kB memory에 들어가지 않는다. 따라서 deployment 관점에서는 SuperNet 모델이 훨씬 유리하다.

Quantization Aware Training은 UCI에만 적용되었다. Floating point ResNet은 SBP MAE 16.59, DBP MAE 8.3, size 3.17 MB이다. Floating point UNet은 SBP MAE 16.93, DBP MAE 7.88, size 118.9 kB이다. int8 quantization 후 ResNet은 size 791.8 kB가 되어도 GAP8 memory를 초과하여 out-of-memory가 발생한다. 반면 int8 UNet은 size 29.8 kB, latency 7.04 ms, energy 0.36 mJ로 배포 가능하다.

SuperNet-derived quantized models는 모두 GAP8에 들어간다. ResNet-B는 SBP MAE 17.83, DBP MAE 8.44, size 156.3 kB, latency 7.12 ms, energy 0.36 mJ이다. ResNet-S는 SBP MAE 17.48, DBP MAE 8.08, size 149.8 kB, latency 7.27 ms, energy 0.37 mJ이다. UNet-S는 SBP MAE 17.2, DBP MAE 8.26, size 23.4 kB, latency 8.91 ms, energy 0.45 mJ이다. DBP에서는 ResNet-S가 가장 좋고, SBP에서는 UNet-S가 가장 좋은 quantized deployable model이다. 다만 depthwise-pointwise convolution이 많은 UNet-S는 parameter 수는 작지만 latency와 energy가 더 높게 나타난다. 이는 parameter 수가 항상 latency와 energy의 완전한 proxy가 아님을 보여준다.

PIT 결과는 논문의 두 번째 주요 성과이다. PIT는 SuperNet에서 얻은 모델 또는 기존 best DL model을 seed로 사용하여 더 세밀한 pruning과 temporal geometry optimization을 수행했다. 결과적으로 PIT는 모든 데이터셋에서 Pareto front를 개선했고, BCG와 UCI에서는 새로운 best model을 만들었다.

PPGBP에서는 PIT가 original ResNet을 seed로 사용했다. PIT는 ResNet보다 훨씬 작은 여러 Pareto model을 생성했지만, 정확도 자체는 ResNet을 넘지 못했다. SBP에서는 parameter를 55% 줄이면서 MAE가 10.07% 증가한 모델이 있었고, DBP에서는 parameter를 52% 줄이면서 MAE가 2.73% 증가한 모델이 있었다. 작은 데이터셋인 PPGBP에서는 여전히 SVR과 Random Forest가 가장 우수하다.

BCG에서는 PIT가 모든 기존 neural network를 지배하며 Pareto front를 크게 개선했다. PIT 모델은 SBP에서 best model인 VNet 대비 45.9% 적은 parameter로 4.676% 낮은 error를 달성했고, DBP에서는 best model인 ResNet 대비 86.68% 적은 parameter로 7.99% 낮은 MAE를 달성했다. 또 다른 모델들은 parameter를 96.12% 줄이면서도 SBP error를 1.05% 낮추거나, parameter를 92.47% 줄이면서 DBP MAE를 1.77% 낮추었다. 특히 BCG에서는 SuperNet이 넘지 못했던 classical ML 모델까지 PIT가 능가했다고 설명된다.

SENSORS에서는 PIT가 매우 작은 모델들을 만들었지만, 많은 경우 error가 증가했다. 예를 들어 ResNet이나 SpectroResNet 대비 SBP와 DBP에서 각각 98.86% 또는 89.15% parameter reduction을 달성하면서 MAE는 1.74%, 1.72%만 증가한 모델들이 있다. 그러나 UNet과 VNet 같은 Sig2Sig model이 여전히 더 좋은 성능을 보이므로, PIT 모델이 최고 정확도를 달성하지는 못했다. 논문은 SENSORS가 Sig2Sig approach에 특히 잘 맞는 데이터셋으로 보인다고 해석한다.

UCI에서는 PIT가 가장 중요한 결과를 낸다. PIT는 SBP와 DBP 모두에서 새로운 최저 MAE를 달성했다. SBP에서는 기존 best인 ResNet 대비 99.3% fewer weights를 사용하면서 1.59% accuracy improvement를 달성했다. DBP에서는 기존 best DBP neural network인 UNet 대비 71.67% 더 효율적인 모델로 2.31% lower error를 달성했다. 최종 best MAE는 SBP 16.324 mmHg, DBP 7.698 mmHg이다. 또한 일부 candidate는 best CNN과 비교해 유사한 정확도를 유지하면서 SBP에서는 99.67%, DBP에서는 91.7% 더 가벼운 모델을 만든다.

전체 결과를 요약하면, SuperNet은 최대 4.99% lower error 또는 iso-error에서 73.36% parameter reduction을 달성했고, PIT는 최대 8.4% lower MAE 또는 iso-error에서 97.5% parameter reduction을 달성했다. Quantization 후 SuperNet 모델들은 모두 GAP8 memory에 들어가며, latency와 energy consumption도 wearable-class device에서 실행 가능한 수준으로 낮다. 다만 논문은 모든 모델의 error가 여전히 clinical standard와는 거리가 있다고 명확히 인정한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 PPG-based BP estimation을 실제 edge deployment 문제로 확장했다는 점이다. 많은 연구들이 MAE만 보고하지만, 이 논문은 parameter 수, operation 수, memory footprint, quantization, latency, energy consumption까지 함께 고려한다. 특히 GAP8이라는 실제 ultra-low-power multicore SoC를 대상으로 배포 결과를 제시한 점은 매우 실용적이다.

두 번째 강점은 공정한 benchmark protocol을 사용했다는 점이다. 논문은 기존 연구들의 문제점으로 custom dataset, 불공정한 preprocessing, random split에 의한 subject leakage, metric inconsistency 등을 지적한다. 그리고 PPGBP, BCG, SENSORS, UCI 네 공개 dataset과 benchmark preprocessing 및 validation protocol을 사용하여 기존 SoA 모델과 비교한다. 이는 결과의 신뢰성을 높인다.

세 번째 강점은 SuperNet, PIT, QAT를 연결한 end-to-end optimization pipeline이다. SuperNet은 coarse layer-level search를 수행하고, PIT는 fine-grained pruning과 temporal convolution geometry optimization을 수행하며, MPS는 quantization-aware training을 통해 int8 deployment를 준비한다. 이 pipeline은 architecture search에서 deployment까지 이어지므로, 실제 적용 가능성이 높다.

네 번째 강점은 다양한 model family를 고려했다는 점이다. ResNet-like Sig2Lab, UNet-like Sig2Sig, TEMPONet-like TCN을 모두 검토하고, traditional ML 모델인 SVR과 Random Forest와도 비교한다. 또한 작은 데이터셋에서는 classical ML이 여전히 강하고, 큰 데이터셋에서는 DNN이 유리하다는 결과를 구분해 해석한다. 이는 단순히 “DNN이 항상 좋다”는 식의 과장된 주장을 피한다.

다섯 번째 강점은 최종 결과를 Pareto front 관점에서 제시한다는 점이다. Edge AI에서는 단일 best accuracy 모델보다, application constraint에 따라 선택 가능한 여러 accuracy-complexity trade-off 모델이 더 중요할 수 있다. 이 논문은 여러 $\lambda$ strength 값을 사용해 다양한 모델을 생성하고, 사용자가 memory나 accuracy 요구조건에 따라 선택할 수 있는 모델군을 제공한다.

그러나 한계도 분명하다. 첫째, 혈압 추정 정확도는 아직 clinical standard를 만족하지 못한다. 최종 PIT 결과에서도 UCI에서 SBP MAE 16.324 mmHg, DBP MAE 7.698 mmHg 수준이다. 이는 trend monitoring이나 개인 건강 insight에는 유용할 수 있지만, medical-grade diagnosis나 clinical decision에 바로 사용하기에는 부족하다. 논문도 이 점을 명확히 인정한다.

둘째, UCI 데이터셋은 subject identity가 unknown이므로 subject leakage를 완전히 배제할 수 없다. 논문은 UCI의 크기 자체가 어느 정도 일반화를 보장할 수 있다고 설명하지만, 혈압 추정에서는 patient-specific pattern이 매우 강할 수 있다. 따라서 UCI 결과는 크고 중요한 benchmark이지만, subject-independent validation으로 완전히 검증되었다고 보기는 어렵다.

셋째, 모든 데이터셋은 주로 clinical resting condition에서 수집되었다. 논문은 이 때문에 motion artifact removal을 생략할 수 있다고 보지만, 실제 wearable 사용 환경에서는 사용자가 움직이고, sensor pressure가 변하며, ambient light와 sweat, skin contact variation이 발생한다. 따라서 본 논문의 모델이 실제 daily-life continuous BP monitoring에서 얼마나 robust한지는 아직 검증되지 않았다.

넷째, NAS cost가 주로 parameter 수에 기반한다. Parameter 수는 memory footprint와 관련이 크지만, latency와 energy를 완전히 설명하지는 않는다. 실제로 UNet-S는 parameter 수가 작지만 depthwise-pointwise convolution이 많아 latency와 energy가 seed UNet보다 증가했다. 이는 operation type, memory access pattern, DMA scheduling, hardware-specific kernel efficiency가 중요하다는 것을 보여준다. 향후에는 실제 hardware latency나 energy model을 NAS cost에 직접 포함하는 것이 더 적절할 수 있다.

다섯째, data augmentation 효과는 제한적이었다. Jittering, scaling, magnitude warping, time warping은 일부 조건에서 약간의 개선을 보였지만, 전반적으로 큰 성능 향상을 만들지 못했다. 이는 PPG-based BP estimation에서 단순 signal transformation만으로는 generalization 문제를 해결하기 어렵다는 점을 시사한다. 특히 혈압 label과 PPG waveform 사이의 physiological relation을 보존하는 augmentation 설계가 필요하다.

여섯째, TEMPONet 탐색은 제한적인 성공만 보였다. TEMPONet과 그 변형들은 overfitting에 취약했고, ResNet을 능가하지 못했다. 논문은 TCN 계열의 가능성을 일부 검토했지만, 더 현대적인 TCN 또는 attention-based time-series architecture에 대한 본격적인 탐색은 수행하지 않았다. 결론에서 transformer-based architecture와 small language model adaptation을 향후 연구로 제안한 것은 이 한계를 인식한 것으로 볼 수 있다.

일곱째, personalized calibration이 아직 포함되지 않았다. 논문은 subject-specific fine-tuning이 clinical standard에 가까워지기 위해 중요하다고 설명하지만, 실제 실험 pipeline에는 포함하지 않는다. PPG와 BP의 관계는 arterial stiffness, vascular compliance, age, tissue properties 등 개인차에 크게 좌우되므로, general model만으로 충분한 정확도를 얻기 어렵다. 따라서 few-shot learning, transfer learning, knowledge distillation 기반 개인화가 향후 핵심 과제로 남아 있다.

비판적으로 종합하면, 이 논문은 “PPG만으로 의료기기급 혈압 추정을 완성한 연구”가 아니라, “PPG-based BP estimation DNN을 edge device에 올릴 수 있도록 자동 최적화한 연구”이다. 정확도 측면에서는 아직 임상 요구 수준과 거리가 있지만, system deployment와 model efficiency 측면에서는 매우 실질적인 기여를 한다.

## 6. 결론

이 논문은 PPG-based blood pressure estimation을 ultra-low-power edge device에서 실행 가능하게 만들기 위한 automated DNN optimization pipeline을 제안했다. 연구는 ResNet, UNet, TEMPONet과 같은 seed architecture에서 출발하여, PLiNIO의 SuperNet으로 coarse NAS를 수행하고, PIT로 fine-grained pruning과 temporal convolution geometry optimization을 수행하며, MPS 기반 QAT로 int8 quantization을 적용한다. 최종적으로 일부 모델은 GAP8 SoC에 배포되어 latency와 energy consumption까지 평가된다.

주요 결과는 의미 있다. SuperNet은 최대 4.99% 낮은 error 또는 iso-error에서 73.36% parameter reduction을 달성했다. PIT는 더 나아가 최대 8.4% 낮은 MAE 또는 iso-error에서 97.5% parameter reduction을 달성했다. UCI 데이터셋에서는 PIT가 SBP MAE 16.324 mmHg, DBP MAE 7.698 mmHg의 새로운 best result를 만들었다. Quantization 이후 SuperNet 모델들은 모두 GAP8의 512 kB memory limit 안에 들어갔고, 일부 모델은 약 7~9 ms의 latency와 0.36~0.45 mJ 수준의 energy consumption을 보였다.

이 연구의 주요 기여는 PPG 기반 혈압 추정 모델을 정확도 중심의 offline benchmark에서 벗어나, 실제 wearable deployment까지 고려한 edge AI optimization 문제로 다루었다는 점이다. 또한 SuperNet, PIT, QAT를 결합하여 다양한 accuracy-complexity trade-off를 가진 Pareto-optimal model set을 자동 생성했다는 점도 중요하다.

다만 모델의 절대 혈압 추정 정확도는 아직 clinical standard를 만족하지 못한다. 따라서 현 단계의 모델은 medical-grade diagnosis보다는 blood pressure trend monitoring, preventive health insight, low-power on-device inference 가능성 검증에 더 적합하다. 실제 의료 적용을 위해서는 subject-specific fine-tuning, few-shot calibration, motion artifact robustness, 더 다양한 healthy population과 real-world wearable dataset에서의 검증이 필요하다.

향후 연구 방향으로는 transformer-based time-series model, sequence-to-sequence PPG-to-ABP translation, personalized fine-tuning, knowledge distillation, theory-guided machine learning, hardware latency-aware NAS가 중요하다. 특히 physical model과 data-driven model을 결합하여 학습 공간을 제한하고, 개인별 calibration을 적은 측정으로 수행할 수 있다면, 이 논문의 lightweight DNN deployment pipeline은 실제 wearable continuous blood pressure monitoring system 개발의 핵심 기반이 될 수 있다.
