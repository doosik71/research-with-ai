# Optimization and Deployment of Deep Neural Networks for PPG-based Blood Pressure Estimation Targeting Low-power Wearables

* **저자**: Alessio Burrello, Francesco Carlucci, Giovanni Pollo, Xiaying Wang, Massimo Poncino, Enrico Macii, Luca Benini, Daniele Jahier Pagliari
* **발표연도**: 2024

## 1. 논문 개요

이 논문은 PPG 기반 혈압 추정용 Deep Neural Network, DNN을 저전력 wearable device에 실제로 배포할 수 있도록 최적화하는 방법을 제안한다. 기존 연구들은 Photoplethysmography, PPG 신호를 이용해 Systolic Blood Pressure, SBP와 Diastolic Blood Pressure, DBP를 추정하는 딥러닝 모델을 개발해 왔다. 그러나 state-of-the-art 모델들은 정확도는 높지만 parameter 수와 연산량이 커서 smartwatch와 같은 저전력 wearable hardware에 탑재하기 어렵다는 문제가 있다. 이 논문은 바로 이 “정확한 PPG-based BP estimation 모델을 실제 edge hardware에서 실행 가능한 크기와 에너지 수준으로 줄이는 문제”를 다룬다.

논문의 목표는 단순히 혈압 추정 오차를 낮추는 것이 아니라, model size, memory footprint, latency, energy consumption까지 고려한 hardware-aware DNN optimization pipeline을 구축하는 것이다. 이를 위해 저자들은 Neural Architecture Search, NAS와 Quantization을 결합한 자동화 설계 흐름을 사용한다. 먼저 기존 state-of-the-art regression 모델과 signal-to-signal 모델을 seed network로 삼고, gradient-based NAS를 통해 각 convolution layer를 standard convolution, depthwise-separable convolution, identity operation 중 하나로 대체하거나 유지하도록 탐색한다. 이후 Pareto-optimal 모델을 int8로 quantization하고, DORY compiler를 통해 GreenWaves GAP8 ultra-low-power multicore System-on-Chip, SoC에서 실행 가능한 C code로 변환한다.

연구 문제는 세 가지로 정리할 수 있다. 첫째, 기존 PPG 기반 혈압 추정 DNN의 정확도를 유지하거나 개선하면서 parameter 수를 줄일 수 있는가이다. 둘째, 최적화된 모델이 GAP8과 같은 wearable-class low-power hardware의 내부 메모리에 실제로 들어갈 수 있는가이다. 셋째, 배포된 모델이 continuous BP monitoring에 필요한 낮은 latency와 낮은 energy consumption을 달성할 수 있는가이다.

이 문제는 실제 적용 측면에서 매우 중요하다. 혈압은 hypertension, cardiomyopathy, heart failure 등 심혈관 질환과 밀접하게 관련된 핵심 생체 지표이다. PPG는 LED와 photodiode를 이용해 피부에서 반사되는 빛의 세기 변화를 측정하고, 혈액량 변화에 따른 waveform을 얻는 방식이므로 wearable device에서 비교적 쉽게 측정할 수 있다. 하지만 continuous monitoring을 위해서는 모델이 기기 내부에서 반복적으로 실행되어야 하므로, 단순한 offline accuracy만으로는 충분하지 않다. 모델이 wearable device의 memory limit, battery budget, real-time latency constraint를 만족해야 한다. 이 논문은 이러한 실용적 병목을 직접 겨냥한다는 점에서 일반적인 혈압 추정 모델 논문과 구별된다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 PPG-based BP estimation을 위한 DNN을 hardware-aware NAS와 quantization으로 자동 최적화하여, 정확도와 배포 가능성을 동시에 달성하는 것이다. 기존 연구의 중심 관심이 주로 MAE를 낮추는 데 있었다면, 이 논문은 “정확한 모델이 실제 wearable hardware에서 실행 가능한가”를 핵심 질문으로 삼는다.

논문에서 출발점으로 삼는 seed model은 두 종류이다. 첫 번째는 scalar regression 방식의 ResNet-like 1D CNN이다. 이 방식은 일정한 PPG window를 입력받아 SBP 또는 DBP 같은 scalar blood pressure value를 직접 예측한다. 두 번째는 signal-to-signal, sig2sig 방식의 UNet-like 1D CNN이다. 이 방식은 PPG time series를 입력받아 전체 blood pressure time series를 재구성하는 방식이다. 기존 benchmark 연구에서는 ResNet-like regression model과 UNet-like sig2sig model이 각각 해당 범주의 우수한 모델로 사용되었다.

논문의 차별점은 이 seed model들을 단순히 직접 배포하는 것이 아니라, cost-aware NAS를 통해 layer-level architecture를 자동으로 다시 설계한다는 점이다. 각 convolution layer는 standard 1D convolution, depthwise-separable block, identity operation 후보 중 하나로 선택될 수 있다. Standard convolution은 표현력이 강하지만 parameter와 연산량이 크다. Depthwise-separable convolution은 MobileNet에서 널리 사용된 경량 convolution 방식으로, standard convolution보다 parameter 수와 연산량을 크게 줄이면서 유사한 표현력을 제공할 수 있다. Identity operation은 해당 layer를 사실상 skip하여 network depth를 줄이는 역할을 한다. 따라서 NAS는 모델의 각 위치에서 “정확도를 위해 복잡한 연산을 유지할지, 경량화를 위해 더 작은 연산으로 바꿀지, 혹은 layer를 생략할지”를 학습한다.

또 다른 핵심 아이디어는 최적화 목표에 model size를 포함한다는 점이다. 일반적인 NAS가 validation accuracy 또는 task loss만 최적화할 수 있다면, 이 논문은 loss function에 expected cost regularization을 추가한다. 이를 통해 NAS는 혈압 추정 오차와 parameter 수 사이의 trade-off를 학습한다. 그 결과 단일 모델 하나가 아니라, MAE와 model size의 Pareto frontier에 놓인 여러 모델을 얻는다. 사용자는 가장 정확한 모델, 가장 작은 모델, 또는 특정 hardware constraint를 만족하는 모델을 선택할 수 있다.

마지막으로 quantization과 deployment까지 포함한 end-to-end flow가 중요하다. NAS로 얻은 floating-point 모델은 다시 int8 precision으로 quantization된다. Weight와 activation을 8-bit로 줄이면 memory footprint와 data movement가 감소하고, edge hardware에서 latency와 energy consumption도 줄일 수 있다. 이후 DORY compiler가 quantized DNN을 GAP8에서 실행 가능한 optimized C code로 변환한다. 이 논문은 NAS 자체보다, NAS, quantization, compiler-based deployment를 하나의 pipeline으로 묶어 실제 low-power wearable-class hardware에서 검증했다는 점이 강점이다.

## 3. 상세 방법 설명

논문의 전체 방법은 데이터셋 및 seed model 선택, gradient-based NAS, quantization, GAP8 deployment의 네 단계로 구성된다. 입력 데이터는 PPG signal이며, 목표 변수는 SBP와 DBP이다. 데이터셋에 따라 scalar BP label만 존재하는 경우에는 regression model만 사용할 수 있고, full BP time series ground truth가 있는 경우에는 signal-to-signal 모델도 사용할 수 있다.

첫 번째 단계에서는 네 개의 공개 데이터셋을 사용한다. 모든 데이터셋은 125 Hz로 resampling된다. PPGBP는 가장 작은 데이터셋으로, 218명의 환자에서 얻은 총 619개의 PPG segment를 포함하며 각 segment 길이는 2.1초이다. 이 데이터셋은 scalar SBP와 DBP 값만 제공하므로 sig2sig 모델을 학습할 수 없다. BCG는 bed-based ballistocardiography dataset이며, 40명에게서 약 4시간의 누적 측정치를 얻고 이를 5초 window로 나눈 데이터셋이다. Sensors는 MIMIC III의 subset으로, 1195명의 환자에게서 얻은 11,102개의 non-overlapping 5초 segment를 포함한다. UCI는 MIMIC II waveform의 subset이며, 약 411,000개의 segment를 포함하는 가장 큰 데이터셋이다. UCI의 subject 수는 논문 텍스트에서 unknown으로 명시된다.

평가 방식은 SBP와 DBP에 대해 test set Mean Absolute Error, MAE를 각각 계산하는 것이다. PPGBP, BCG, Sensors는 5-fold per-subject cross-validation을 사용한다. UCI는 데이터 규모가 크기 때문에 single-held-out validation set과 test set을 사용한다. 논문은 이 protocol에서 cross-patient inference를 수행하면 의료기기 기준을 만족하기 어려운 높은 오차가 나온다고 명확히 언급한다. 즉, 본 논문의 목표는 의료기기 수준의 절대 정확도를 달성하는 것이 아니라, 기존 state-of-the-art DNN과 유사하거나 더 나은 정확도를 유지하면서 wearable hardware deployment를 가능하게 하는 것이다. 저자들은 patient-specific fine-tuning을 적용하면 정확도 개선이 가능할 수 있지만, 이는 본 연구의 범위를 벗어난다고 설명한다.

두 번째 단계는 Neural Architecture Search이다. 저자들은 PLiNIO라는 open-source library를 사용하여 gradient-based NAS와 quantization을 수행한다. NAS의 seed network는 기존 benchmark 연구에서 가장 성능이 좋았던 두 모델이다. 하나는 ResNet-derived scalar regressor이고, 다른 하나는 UNet-like sig2sig model이다.

NAS는 SuperNet 방식으로 수행된다. Seed network의 각 convolutional layer를 하나의 고정된 layer로 두지 않고, 여러 candidate operation이 병렬로 존재하는 pool로 대체한다. 후보 operation은 standard 1D convolution, depthwise-separable convolution block, identity operation이다. 각 후보의 출력은 trainable architecture parameter $\theta_i$에 softmax를 적용한 weight로 가중합된다. 따라서 특정 layer의 출력은 개념적으로 다음과 같이 표현할 수 있다.

$$
Y = \sum_i \text{softmax}(\theta_i) \cdot O_i(X)
$$

여기서 $X$는 해당 layer의 입력, $O_i$는 candidate operation, $\theta_i$는 각 operation의 선택 가능성을 나타내는 trainable parameter이다. 학습 초기에는 여러 operation이 동시에 섞여 사용되지만, 학습이 진행되면서 특정 operation의 $\theta_i$가 커지고 다른 operation은 작아진다. 학습이 끝난 뒤에는 각 layer에서 가장 큰 $\theta_i$를 가진 operation을 선택하여 최종 discrete architecture를 만든다.

NAS 학습의 loss function은 task loss와 cost regularization의 합으로 구성된다.

$$
\mathcal{L}_{\text{NAS}}(W,\theta) = \mathcal{L}(W,\theta) + \lambda \mathcal{R}(\theta)
$$

여기서 $W$는 일반적인 network weight이고, $\theta$는 architecture selection parameter이다. $\mathcal{L}(W,\theta)$는 task loss이며, 이 논문에서는 모델 출력과 ground truth 사이의 Mean Squared Error, MSE이다. $\mathcal{R}(\theta)$는 expected cost이며, 이 논문에서는 model size를 cost metric으로 사용한다. $\lambda$는 accuracy와 size 사이의 trade-off를 조절하는 regularization strength이다.

Task loss인 MSE는 다음과 같이 이해할 수 있다.

$$
\mathcal{L}(W,\theta) = \text{MSE}(f(W,X), \hat{Y})
$$

여기서 $f(W,X)$는 입력 PPG $X$에 대한 network output이고, $\hat{Y}$는 ground truth BP label 또는 BP time series이다. Regression model의 경우 $\hat{Y}$는 SBP 또는 DBP scalar이고, sig2sig model의 경우 $\hat{Y}$는 blood pressure time series이다.

Expected cost $\mathcal{R}(\theta)$는 각 layer에서 candidate operation이 선택될 확률과 해당 operation의 cost를 곱해 더한 값으로 볼 수 있다. 예를 들어 어떤 layer에 standard convolution, depthwise-separable convolution, identity operation이 있고 각각의 softmax weight가 $\theta_C$, $\theta_{DW}$, $\theta_{ID}$라면, 해당 layer의 expected cost는 다음과 같은 형태로 해석할 수 있다.

$$
\mathcal{R}*{\text{layer}}(\theta) =
\theta_C \cdot \text{Cost}(C) +
\theta*{DW} \cdot \text{Cost}(DW) +
\theta_{ID} \cdot \text{Cost}(ID)
$$

Identity operation은 별도의 parameter를 거의 추가하지 않으므로 cost가 0에 가깝다. 전체 network의 expected cost는 각 layer의 expected cost를 모두 더해 계산된다. 이 구조 덕분에 NAS는 accuracy만 높이는 것이 아니라, model size가 작은 architecture를 선호하도록 유도된다.

학습 과정에서는 network weight와 NAS parameter를 별도의 Adam optimizer로 최적화한다. Network weight는 learning rate 0.001로 training set에서 업데이트되고, NAS parameter $\theta$는 learning rate 0.01로 validation set에서 업데이트된다. 모든 데이터셋과 ResNet, UNet seed에 대해 $\lambda$ 값을 $10^{-11}$부터 $10^{-7}$까지 log scale로 균등하게 배치한 18개 값으로 실험한다. $\lambda$가 작으면 accuracy 중심의 모델이 나오고, $\lambda$가 크면 더 작은 모델을 선호하게 된다. 이 과정을 통해 다양한 error-size trade-off를 가진 Pareto-optimal architecture를 얻는다.

세 번째 단계는 quantization이다. NAS로 찾은 Pareto-optimal DNN 중 일부를 선택하여 int8 precision으로 변환한다. 논문은 PLiNIO의 Quantization-Aware Training, QAT 기능을 사용한다. Weight에는 standard min-max affine quantization을 적용하고, layer input과 output activation에는 Parametrized Clipping Activation, PaCT 방식을 사용한다. Accumulation과 bias는 target inference library가 지원하는 방식에 따라 32-bit로 유지된다.

Quantization을 적용하면 fp32 weight를 int8 weight로 줄일 수 있으므로 model size가 대략 4분의 1로 감소한다. 또한 저전력 processor에서 integer arithmetic을 사용할 수 있어 memory bandwidth, latency, energy 측면에서 유리하다. 다만 quantization은 표현 정밀도를 낮추므로 MAE가 약간 증가할 수 있다. 이 논문에서도 quantization 이후 최대 9.8%의 MAE 증가가 나타난다고 보고한다.

네 번째 단계는 deployment이다. Target hardware는 GreenWaves GAP8이다. GAP8은 RISC-V 기반 ultra-low-power multicore IoT processor이며, signal processing task를 위해 설계되었다. 8개의 general-purpose core로 구성된 cluster를 통해 compute-intensive workload를 가속할 수 있다. Memory는 512 kB main memory와 64 kB last-level cache 구조를 가진다. 512 kB main memory는 application code와 DNN weight를 저장하는 데 사용되므로, 모델이 이 범위를 초과하면 onboard memory에 배포할 수 없다.

DNN을 GAP8에서 실행하기 위해 저자들은 DORY compiler를 사용한다. DORY는 quantized DNN을 입력으로 받아 C code를 자동 생성하며, inference 과정 전체를 처리한다. 여기에는 memory management, DMA transfer scheduling, optimized AI primitive invocation이 포함된다. 성능 측정은 GAP8 evaluation board에서 수행되며, latency는 internal performance counter로 측정하고, power는 Nordic Power Profiler Kit II로 측정한다.

이 논문의 방법론에서 중요한 점은 최적화가 논문상 모델 압축에 그치지 않고 실제 hardware profiling까지 이어진다는 것이다. 많은 DNN compression 연구는 parameter 수나 MAC 수만 보고하지만, 이 논문은 실제 SoC에서 inference latency와 energy per inference를 측정한다. 따라서 wearable deployment 가능성을 더 직접적으로 평가한다.

## 4. 실험 및 결과

실험은 네 개의 공개 데이터셋인 PPGBP, BCG, Sensors, UCI에서 수행된다. 비교 대상은 기존 benchmark 연구에서 사용된 ResNet seed, UNet seed, Random Forest, Support Vector Regression이다. NAS 결과는 MAE와 model size, 즉 parameter 수의 trade-off로 분석되며, quantization 및 deployment 결과는 가장 크고 복잡한 UCI 데이터셋에 대해 상세히 보고된다.

Pareto analysis에서는 NAS로 얻은 모델들이 seed network를 지배하거나, 적어도 memory-error Pareto frontier 위에 위치함을 보여준다. 여기서 어떤 모델이 다른 모델을 지배한다는 것은 더 낮은 MAE와 더 작은 size를 동시에 달성한다는 의미이다. 이는 hardware-aware NAS가 단순 경량화가 아니라, accuracy-size trade-off를 효과적으로 개선했음을 의미한다.

PPGBP는 가장 작은 데이터셋이며, scalar BP label만 제공하므로 UNet seed는 사용할 수 없다. ResNet seed에서 출발한 NAS는 다양한 Pareto architecture를 생성한다. 저자들은 seed size를 16% 줄이면서 DBP MAE는 3.9%, SBP MAE는 3.5%만 증가시키는 모델을 얻었다고 보고한다. 즉, 작은 성능 손실로 모델 크기를 줄일 수 있었다. 그러나 이 데이터셋에서는 classic ML model인 SVR이 여전히 가장 좋은 DBP MAE 8.04 mmHg와 SBP MAE 13.15 mmHg를 달성한다고 설명한다. 이는 작은 데이터셋에서는 DNN보다 classical ML이 더 강할 수 있음을 시사한다.

BCG 데이터셋에서는 NAS가 ResNet과 UNet seed 모두를 Pareto-dominate한다. 가장 좋은 UNet-derived model은 SBP prediction에서 MAE 11.139 mmHg, DBP prediction에서 MAE 7.52 mmHg를 달성한다. 이는 가장 좋은 seed인 ResNet보다 SBP에서 6.7%, DBP에서 4.7% 더 좋은 결과이며, 동시에 parameter 수를 3.8배 줄인다. 그러나 PPGBP와 BCG 같은 작은 데이터셋에서는 classical ML method가 DNN보다 더 좋은 경우가 여전히 존재한다. 예를 들어 BCG에서 SVR은 DBP MAE 7.34 mmHg를 달성하고, SBP는 11.45 mmHg로 UNet-derived model보다 약간 낮거나 비슷한 수준이다. 이 결과는 데이터 규모가 작을수록 DNN의 장점이 제한될 수 있음을 보여준다.

Sensors는 더 큰 데이터셋이다. 이 데이터셋에서는 classical ML model이 seed network보다 약간 나은 성능을 보이지만, NAS-optimized DNN이 이를 넘어선다. 논문은 SVR이 SBP MAE 15.60 mmHg, DBP MAE 7.50 mmHg를 달성했지만, UNet NAS model은 SBP MAE 15.51 mmHg와 DBP MAE 7.50 mmHg를 달성하면서 parameter 수를 최대 40배 줄였다고 보고한다. 즉, accuracy는 같거나 더 좋고 model size는 훨씬 작다. 이는 low-power deployment 관점에서 매우 중요한 결과이다.

UCI는 가장 큰 데이터셋으로, 약 411,000개의 segment를 포함한다. 이 데이터셋에서는 classical ML model이 DNN seed보다도 낮은 성능을 보인다. 예를 들어 가장 좋은 classical model 중 RF는 SBP MAE 16.85 mmHg를 달성하지만, ResNet seed는 16.59 mmHg를 달성한다. DBP에서는 SVR이 8.07 mmHg, UNet seed가 7.88 mmHg로 UNet이 더 좋다. 또한 UCI처럼 복잡한 데이터셋에서는 SVR이나 RF의 parameter 또는 support vector 규모가 크게 증가하여 model size가 매우 커진다. 논문은 UCI에서 SVR이 best NAS output보다 998배 크다고 보고한다. 이 결과는 대규모 데이터에서는 DNN 최적화의 장점이 더 뚜렷해짐을 보여준다.

UCI에서 NAS로 찾은 가장 정확한 모델은 SBP estimation에서 16.655 mmHg에 가까운 MAE를 보이고, DBP estimation에서는 전체 최저인 7.86 mmHg MAE를 달성한다. 이 모델들은 각각 약 149.8k 또는 156.3k parameters를 사용한다. 반면 seed ResNet은 SBP에서 더 낮은 MAE를 달성할 수 있지만 parameter 수가 792k로 많아, quantization을 적용해도 GAP8의 512 kB internal memory에 들어가지 않는다. 따라서 순수 accuracy만 보면 seed ResNet이 일부 지표에서 우수할 수 있지만, deployment 가능성을 고려하면 NAS 모델이 더 실용적이다.

Quantization 및 deployment 결과는 UCI dataset을 중심으로 Table I에 보고된다. Floating point seed 모델의 경우 ResNet은 SBP MAE 16.59, DBP MAE 8.3, size 3.17 MB이다. UNet은 SBP MAE 16.93, DBP MAE 7.88, size 118.9 kB이다. GAP8은 floating-point unit이 없으므로 이 fp32 모델들은 직접 배포 대상이 아니다.

Quantized int8 모델을 보면, seed ResNet은 size가 791.8 kB로 줄었지만 여전히 GAP8의 onboard memory를 초과하여 out-of-memory가 발생한다. 즉, quantization만으로는 가장 큰 seed model을 배포할 수 없다. 반면 seed UNet은 int8에서 size 29.8 kB, latency 7.04 ms, energy 0.36 mJ로 배포 가능하다. 하지만 MAE는 SBP 17.63, DBP 8.19로 floating point보다 증가한다.

NAS로 생성된 ResNet-B, ResNet-S, UNet-S는 모두 GAP8에 배포 가능하다. ResNet-B는 SBP MAE 17.83, DBP MAE 8.44, size 156.3 kB, latency 7.12 ms, energy 0.36 mJ이다. ResNet-S는 SBP MAE 17.48, DBP MAE 8.08, size 149.8 kB, latency 7.27 ms, energy 0.37 mJ이다. UNet-S는 SBP MAE 17.2, DBP MAE 8.26, size 23.4 kB, latency 8.91 ms, energy 0.45 mJ이다.

이 결과에서 DBP 기준으로 가장 좋은 배포 모델은 ResNet-S이며, DBP MAE 8.08을 달성하면서 0.37 mJ만 소비한다. SBP 기준으로는 UNet-S가 MAE 17.2로 가장 좋고, size도 23.4 kB로 매우 작다. 다만 UNet-S는 depthwise-separable layer가 많아 size는 작지만 latency와 energy는 seed UNet보다 약간 증가한다. 이는 depthwise-separable convolution이 parameter 수는 줄이지만 특정 hardware에서는 항상 가장 빠른 연산은 아닐 수 있음을 보여준다.

논문 전체의 실험 결과를 종합하면, NAS는 기존 seed DNN 대비 최대 4.99% 낮은 error를 달성하거나, 같은 error 수준에서 최대 73.36% 작은 size를 달성한다. 또한 가장 큰 UCI dataset에서는 기존 가장 정확한 모델이 GAP8 memory에 들어가지 못하는 반면, NAS로 최적화한 모든 모델은 배포 가능하다. Inference latency는 7.12~8.91 ms, energy는 0.36~0.45 mJ 수준으로 보고된다. 이는 continuous monitoring 관점에서 매우 낮은 에너지와 빠른 응답 시간이다.

다만 혈압 추정 정확도 자체는 의료기기 기준과 비교하면 아직 높지 않은 오차를 보인다. 예를 들어 UCI에서 SBP MAE가 16 mmHg 이상이고 DBP MAE가 약 8 mmHg 수준이므로, 임상용 cuffless BP monitor로 바로 사용할 수 있는 수준이라고 보기는 어렵다. 논문도 cross-patient inference protocol에서는 medical-grade device requirement보다 훨씬 높은 error가 나온다고 언급한다. 따라서 이 논문의 성과는 “의료기기급 정확도 달성”이 아니라 “기존 DNN 수준의 성능을 저전력 wearable hardware에 배포 가능하게 최적화”한 데 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 PPG-based BP estimation 연구를 실제 hardware deployment 관점에서 다룬다는 점이다. 많은 혈압 추정 논문은 offline dataset에서 MAE만 비교하지만, 이 논문은 model size, quantization, memory limit, latency, energy consumption을 함께 평가한다. 특히 target hardware인 GAP8의 512 kB memory constraint를 고려하여, 실제 wearable-class SoC에 들어가는지를 검증했다는 점이 매우 실용적이다.

두 번째 강점은 NAS와 quantization을 결합한 완전 자동화된 optimization pipeline이다. 기존 모델을 사람이 수동으로 줄이는 것이 아니라, SuperNet 기반 gradient-based NAS가 각 layer의 operation과 depth를 자동으로 선택한다. 또한 $\lambda$ 값을 바꾸며 여러 Pareto-optimal 모델을 생성하므로, application constraint에 따라 가장 적절한 모델을 선택할 수 있다. 이는 wearable device처럼 memory, latency, energy budget이 제품마다 다른 상황에서 유용하다.

세 번째 강점은 네 개의 공개 데이터셋에서 평가했다는 점이다. PPGBP, BCG, Sensors, UCI는 규모와 특성이 서로 다르며, 논문은 작은 데이터셋에서는 classic ML이 여전히 강할 수 있고, 큰 데이터셋에서는 NAS-optimized DNN이 더 유리해진다는 점을 보여준다. 이처럼 dataset size에 따른 model family의 장단점을 분석한 점은 의미가 있다.

네 번째 강점은 signal-to-signal model과 scalar regression model을 모두 고려했다는 점이다. 논문은 ResNet-like regression model과 UNet-like sig2sig model을 seed로 사용하여 두 계열의 모델을 모두 최적화한다. 이는 PPG-based BP estimation에서 널리 쓰이는 두 접근법을 포괄한다. 또한 PPGBP처럼 full BP time series가 없는 데이터셋에서는 sig2sig model을 학습할 수 없다는 제약을 명확히 설명한다.

다섯 번째 강점은 실제 deployment 결과를 수치로 제시했다는 점이다. 단순히 parameter 수가 줄었다고 주장하지 않고, GAP8 evaluation board에서 latency와 energy를 측정했다. ResNet-S가 DBP MAE 8.08을 달성하면서 0.37 mJ만 소비하고, UNet-S가 SBP MAE 17.2를 달성하면서 0.45 mJ를 소비한다는 결과는 wearable deployment 가능성을 직접적으로 보여준다.

그러나 한계도 분명하다. 첫째, 혈압 추정 정확도 자체는 medical-grade requirement를 만족하지 못한다. 논문은 cross-patient inference protocol에서 오차가 높으며, personalized fine-tuning이 필요할 수 있다고 언급한다. UCI에서 SBP MAE가 16~17 mmHg 수준이라는 점은 임상적 혈압 측정기로 사용하기에는 매우 큰 오차이다. 따라서 본 논문은 clinical BP estimation algorithm의 완성이라기보다, existing BP estimation DNN의 efficient deployment 연구로 보는 것이 정확하다.

둘째, motion artifact가 제한적으로만 고려된다. 논문은 모든 데이터셋이 resting patients in a clinical setting에서 측정되었기 때문에 acceleration data 기반 motion artifact removal을 생략할 수 있다고 설명한다. 그러나 실제 wearable 환경에서는 사용자가 걷거나 운동하거나 손목 위치가 바뀌는 상황이 일반적이다. 이 경우 PPG signal은 motion artifact, sensor-skin contact variation, ambient light, sweat, skin tone, peripheral perfusion 등의 영향을 크게 받을 수 있다. 따라서 clinical resting dataset에서의 deployment 가능성이 실제 생활 환경의 continuous monitoring 성능을 보장하지는 않는다.

셋째, UCI 데이터셋의 subject 수가 unknown으로 제시되고, UCI에서는 5-fold per-subject cross-validation이 아닌 single-held-out validation/test split을 사용한다. 논문이 기존 benchmark protocol을 따른다고 설명하지만, subject-wise independence가 충분히 보장되는지 여부는 제공된 텍스트만으로 명확하지 않다. BP estimation에서는 같은 환자의 segment가 train과 test에 동시에 포함되면 일반화 성능이 과대평가될 수 있으므로 이 부분은 중요한 검토 지점이다.

넷째, NAS의 cost metric이 model size에 집중되어 있다. 실제 hardware에서는 parameter 수뿐 아니라 activation memory, peak memory, data movement, operation type별 latency, DMA scheduling, memory access pattern 등이 중요하다. 논문은 최종 deployment에서 latency와 energy를 측정하지만, NAS 단계의 regularization cost는 model size를 사용한다. 그 결과 UNet-S처럼 depthwise-separable layer가 많아 size는 작지만 latency와 energy가 오히려 증가하는 사례가 나타난다. 더 정교한 hardware-aware NAS라면 GAP8에서의 measured latency 또는 analytic latency model을 직접 cost로 사용할 수도 있다.

다섯째, quantization으로 인한 accuracy degradation이 일부 모델에서 존재한다. 논문은 quantization 이후 MAE가 최대 9.8% 증가한다고 보고한다. 특히 ResNet 계열이 quantization degradation에 더 민감하다고 설명한다. 이는 int8 QAT가 효과적이지만 모든 architecture에 동일하게 안정적인 것은 아니며, quantization-friendly architecture search가 추가로 필요할 수 있음을 시사한다.

여섯째, seed model과 preprocessing protocol을 기존 benchmark 연구에 의존한다. 이는 공정 비교를 가능하게 하는 장점이 있지만, 논문 자체에서 preprocessing의 세부적인 재검증이나 혈압 추정 모델의 생리학적 해석을 깊게 다루지는 않는다. 본 논문의 중심은 model optimization과 deployment이므로 이는 자연스러운 범위 설정이지만, 혈압 추정 자체의 physiological validity나 calibration 문제는 제한적으로만 논의된다.

비판적으로 보면, 이 논문은 “혈압 추정 정확도를 획기적으로 임상 수준까지 끌어올린 연구”가 아니다. 오히려 이미 존재하는 state-of-the-art DNN을 low-power hardware에 올릴 수 있도록 경량화하고, 그 과정에서 정확도 손실을 최소화하거나 일부 경우에는 성능까지 개선한 연구이다. 따라서 이 논문의 학술적 가치는 biomedical signal estimation 모델의 deployment-aware optimization에 있으며, PPG-based BP estimation이 실제 wearable 제품으로 가기 위해 필요한 시스템적 문제를 다룬다는 데 있다.

## 6. 결론

이 논문은 PPG-based blood pressure estimation DNN을 low-power wearable device에 배포하기 위한 automated optimization and deployment pipeline을 제안했다. 기존 state-of-the-art ResNet-like regression model과 UNet-like signal-to-signal model을 seed로 사용하고, SuperNet 기반 gradient-based NAS를 통해 각 convolution layer를 standard convolution, depthwise-separable convolution, identity operation 중에서 선택하도록 최적화했다. 이후 Pareto-optimal 모델을 int8로 quantization하고, DORY compiler를 이용해 GAP8 ultra-low-power SoC에서 실행 가능한 C code로 변환했다.

실험 결과, 제안된 최적화는 기존 state-of-the-art DNN 대비 최대 4.99% 낮은 error를 달성하거나, 같은 error 수준에서 최대 73.36% 작은 model size를 달성했다. 특히 가장 큰 UCI 데이터셋에서 seed ResNet은 quantization 후에도 GAP8 memory에 들어가지 못했지만, NAS로 생성된 모든 모델은 배포 가능했다. GAP8에서의 inference latency는 약 7.12~8.91 ms이고, energy consumption은 약 0.36~0.45 mJ로 보고되었다. DBP estimation에서 ResNet-S는 MAE 8.08을 0.37 mJ의 에너지로 달성했고, SBP estimation에서 UNet-S는 MAE 17.2를 0.45 mJ로 달성했다.

이 연구의 주요 기여는 PPG 기반 혈압 추정 모델을 정확도 중심의 offline benchmark에서 벗어나, wearable hardware deployment까지 고려한 end-to-end optimization 문제로 정식화했다는 점이다. NAS와 quantization을 결합하여 memory-constrained SoC에 실제로 탑재 가능한 DNN을 자동으로 설계했고, 이를 실제 GAP8 board에서 latency와 energy까지 측정했다는 점이 중요하다.

다만 이 연구의 모델은 아직 medical-grade BP monitoring 기준을 만족하는 수준의 정확도를 달성하지 못한다. 특히 cross-patient inference에서는 SBP와 DBP 모두 임상 적용을 위해 추가 개선이 필요하며, 저자들도 future work로 patient-specific fine-tuning을 제안한다. 또한 실제 wearable 환경에서 발생하는 motion artifact와 sensor variability에 대한 검증은 부족하다.

종합하면, 이 논문은 PPG-based cuffless BP estimation의 정확도 향상 논문이라기보다, 이미 강력한 DNN 모델을 low-power wearable-class hardware에 배포하기 위한 model compression, NAS, quantization, compiler deployment 연구로 평가하는 것이 적절하다. 향후 patient-specific calibration, motion robustness, hardware latency-aware NAS, quantization-friendly architecture search가 결합된다면, 이 연구는 실제 wearable continuous blood pressure monitoring system 개발에 중요한 기반이 될 수 있다.
