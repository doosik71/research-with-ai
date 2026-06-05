# BPNet: A Multi-Modal Fusion Neural Network for Blood Pressure Estimation Using ECG and PPG

* **저자**: Weicai Long, Xingjun Wang
* **발표연도**: 2023

## 1. 논문 개요

이 논문은 ECG와 PPG 신호를 동시에 사용하여 continuous blood pressure, 즉 연속 혈압을 추정하는 deep learning 모델인 BPNet을 제안한다. 목표는 ICU 환자 데이터에서 수집된 ECG, PPG, invasive arterial blood pressure, 즉 ABP를 이용하여 systolic blood pressure, SBP와 diastolic blood pressure, DBP를 정확하게 추정하는 것이다. 논문은 단순히 새로운 neural network를 제안하는 데 그치지 않고, 기존 cuffless blood pressure estimation 연구에서 자주 발생하는 세 가지 방법론적 문제를 함께 다룬다. 첫째는 skewed label distribution 때문에 regression model이 특정 혈압 범위에 치우쳐 학습되는 문제이고, 둘째는 overlapping sample 생성으로 인한 data leakage 문제이며, 셋째는 ECG와 PPG를 단순 concatenate하여 두 physiological modality 사이의 관계를 충분히 활용하지 못하는 문제이다.

혈압 추정은 임상적으로 중요한 문제이다. 고혈압은 심장, 뇌, 신장 등 주요 장기에 영향을 미치는 위험 요인이며, 조기 발견과 지속적 모니터링이 중요하다. 기존 invasive method는 정확하지만 동맥에 transducer를 연결해야 하므로 임상 중환자 또는 수술 상황에 제한된다. Cuff-based non-invasive method는 널리 사용되지만 불편하고 연속 측정에 적합하지 않다. 따라서 ECG와 PPG 같은 비침습 생체신호를 이용한 cuffless continuous BP estimation은 wearable device와 장기 모니터링 시스템에서 중요한 연구 주제이다.

이 논문은 MIMIC-II와 MIMIC-III 데이터셋을 사용한다. MIMIC-II에서는 942명 환자, 245,638개 sample을 사용했고, MIMIC-III에서는 833명 환자, 총 833,000개 sample을 사용했다. 각 sample은 10초 길이이며, sampling rate는 125 Hz이다. 모델의 입력은 ECG와 PPG segment이고, target label은 해당 segment 안의 ABP peak와 valley로부터 계산한 평균 SBP와 DBP이다.

논문이 보고한 최종 성능은 상당히 높다. MIMIC-II에서 BPNet은 SBP에 대해 ME -0.17 mmHg, MAE 3.98 mmHg, STD 4.62 mmHg, correlation coefficient $r=0.9564$를 달성했고, DBP에 대해 ME -0.24 mmHg, MAE 2.33 mmHg, STD 2.95 mmHg, $r=0.9340$을 달성했다. MIMIC-III에서는 SBP에 대해 ME -0.30 mmHg, MAE 5.59 mmHg, STD 5.78 mmHg, $r=0.9295$를 보였고, DBP에 대해 ME -0.25 mmHg, MAE 3.35 mmHg, STD 3.80 mmHg, $r=0.9265$를 보였다. 또한 AAMI 기준을 두 데이터셋 모두에서 만족했고, BHS 기준에서는 MIMIC-II의 SBP와 DBP가 모두 Grade A, MIMIC-III에서는 SBP Grade B, DBP Grade A를 달성했다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 세 가지이다. 첫 번째는 혈압 label distribution을 regression 관점에서 다시 보는 것이다. 저자들은 BP estimation이 regression task이기 때문에 일반적으로 MSE loss를 사용하는데, MSE는 residual이 normal distribution을 따른다는 가정과 연결된다고 설명한다. 만약 label distribution이 크게 skewed되어 있으면, 모델은 빈도가 높은 혈압 범위에 집중하고 드문 혈압 범위에서는 일반화 성능이 떨어질 수 있다. 이를 해결하기 위해 논문은 Box-Cox transformation을 적용하여 skewed label distribution을 normal distribution에 가깝게 변환한다.

두 번째 아이디어는 non-overlapping sample 구성이다. 생체신호 연구에서 긴 waveform을 window로 자를 때 adjacent sample을 overlap시키면 sample 수는 늘어나지만, train set과 test set 사이에 거의 같은 waveform 구간이 들어갈 수 있다. 이 경우 모델은 실제 generalization을 한 것이 아니라 test set과 유사한 train segment를 기억해서 높은 성능을 보일 수 있다. 이 논문은 10초 segment를 overlap 없이 구성하여 data leakage를 줄이려 한다.

세 번째 아이디어는 ECG와 PPG의 cross-modal multi-level fusion이다. 기존 일부 연구는 ECG와 PPG를 단순히 channel 방향으로 붙여서 neural network에 입력했다. 그러나 ECG는 심장의 electrical activity를 반영하고, PPG는 혈액이 혈관을 통해 전파되는 pulse wave를 반영한다. 즉 두 신호는 서로 다른 생리적 정보를 담고 있으며, 특히 PAT 또는 PTT와 같이 두 신호 사이의 시간적 관계가 혈압과 관련된다. BPNet은 ECG branch와 PPG branch를 별도로 구성하고, 여러 CNN level에서 feature를 추출한 뒤, FPN을 이용해 multi-level feature를 융합한다. 이를 통해 단순 concatenation보다 ECG와 PPG 사이의 implicit relationship을 더 잘 포착하려 한다.

이 논문의 차별점은 architecture 자체뿐 아니라 실험 설계에도 있다. 저자들은 Box-Cox transformation, cross-modal fusion, multi-task learning 각각을 ablation study로 검증한다. 또한 ResNet, SEResNet, LSTM 같은 비교 모델뿐 아니라 기존 연구들과도 MIMIC-II 및 MIMIC-III 기준으로 비교한다. 특히 논문은 MIMIC-III를 구성할 때 각 환자에서 동일한 수의 sample을 사용했다고 강조하는데, 이는 특정 환자의 sample이 과도하게 많아 모델 평가를 왜곡하는 문제를 줄이려는 의도로 해석된다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

BPNet의 전체 파이프라인은 입력, 전처리, BPNet, 출력의 네 단계로 구성된다. 입력은 ECG segment와 PPG segment이다. 전처리 단계에서는 signal quality assessment, signal filtering, peak-valley detection을 수행한다. 이후 전처리된 ECG와 PPG가 BPNet에 입력된다. BPNet은 크게 pre-feature extraction, cross-modal fusion, post-feature extraction, multi-task structure의 네 구성 요소를 가진다. 최종 출력은 SBP와 DBP 두 값이다.

### 3.2 Regression task 재검토와 Box-Cox transformation

논문은 먼저 regression task의 통계적 가정을 설명한다. 입력을 $x$, 혈압 label을 $y$라고 할 때, 단순화된 regression 관계는 다음과 같이 표현된다.

$$
y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}
$$

여기서 $\theta$는 regressor parameter이고, $\epsilon^{(i)}$는 residual term이다. 일반적인 linear regression에서는 residual이 평균 0, 분산 $\sigma^2$인 normal distribution을 따른다고 가정한다.

$$
p(\epsilon^{(i)}) = N(0, \sigma^2)
$$

이 가정에서 예측값을 $y_{pred}^{(i)}$라고 하면 conditional distribution은 다음과 같이 쓸 수 있다.

$$
p(y^{(i)}|x;\theta) = N(y^{(i)}; y_{pred}^{(i)}, \sigma^2)
$$

이 likelihood를 최대화하는 것은 negative log likelihood를 최소화하는 것과 같고, 결과적으로 MSE loss를 최소화하는 문제로 이어진다. 논문은 이를 통해 MSE를 사용하는 regression에서는 label distribution과 residual distribution의 통계적 성격을 고려해야 한다고 설명한다. 즉 혈압 label이 skewed되어 있으면 모델이 frequent label에 편향될 수 있다.

이를 해결하기 위해 Box-Cox transformation을 사용한다.

$$
\hat{y} =
\begin{cases}
\frac{y^\lambda - 1}{\lambda}, & \lambda \neq 0 \
\ln y, & \lambda = 0
\end{cases}
$$

여기서 $y$는 원래의 continuous blood pressure label이고, $\hat{y}$는 변환된 label이며, $\lambda$는 maximum likelihood estimation으로 구하는 transformation parameter이다. 이 변환의 목적은 skewed distribution을 normal distribution에 가깝게 만들어 MSE loss 기반 학습이 더 타당하게 작동하도록 하는 것이다. 논문은 ablation study에서 Box-Cox를 제거한 경우보다 Box-Cox를 사용한 최종 모델이 더 좋은 MAE와 STD를 보인다고 보고한다.

### 3.3 데이터 전처리

논문은 MIMIC-II의 compiled version과 직접 구성한 MIMIC-III를 사용한다. 두 데이터셋 모두 sampling rate는 125 Hz이다. MIMIC-II는 이미 compiled version이 있지만 blank segment나 missing segment 같은 anomaly가 남아 있을 수 있고, MIMIC-III는 완전히 compiled되어 있지 않기 때문에 추가 전처리가 필요하다.

첫 번째 전처리 단계는 quality assessment이다. ECG, PPG, ABP는 blank signal, baseline drift, high-frequency electromyography, abnormal impulse 등 여러 noise를 포함할 수 있다. 단순 filtering만으로 blank 또는 abnormal signal을 제거하기 어렵기 때문에, 논문은 segment removal 기반 quality assessment를 수행한다. 구체적으로 3000 sample point 크기의 window를 고정하여 ECG, PPG, ABP segment를 평가한다. 평가 기준은 signal amplitude range와 derivative가 0인 point의 수이다. 정상 신호에서는 amplitude range가 제한적이어야 하며, derivative가 0인 point가 과도하게 많으면 blank signal이 포함된 것으로 볼 수 있다.

두 번째 단계는 signal filtering이다. PPG의 주요 noise는 baseline drift이므로 db8 wavelet을 사용한다. 원 신호를 discrete wavelet transformation으로 8 level로 분해하고, Sureshrink threshold rule을 사용하여 작은 wavelet coefficient를 0으로 만든다. 이는 noise를 줄이면서 유용한 signal component를 보존하기 위한 방법이다. ECG의 경우 baseline drift와 EMG interference를 제거하기 위해 passband 1 Hz에서 35 Hz인 bandpass filter를 사용한다.

세 번째 단계는 ABP peak-valley detection이다. SBP와 DBP label을 얻기 위해 ABP signal의 peak와 valley를 검출한다. 논문은 slope sum function, SSF를 사용한다. SSF는 waveform의 rising slope를 강조하고 falling slope를 억제하기 위한 함수이다. 시간 $i$에서 window weighting $z_i$는 다음과 같이 정의된다.

$$
z_i = \sum_{k=i-w}^{i} \Delta u_k,\quad
\Delta u_k =
\begin{cases}
\Delta y_k, & \Delta y_k > 0 \
0, & \Delta y_k \leq 0
\end{cases}
$$

여기서 $\Delta y_k = y_k - y_{k-1}$이고, $w$는 window length이다. $w$는 일반적으로 signal rising slope의 길이로 선택된다. ABP에서 peak는 SBP 후보, valley는 DBP 후보가 되며, segment 내 peak와 valley 값들의 평균이 해당 10초 sample의 true SBP와 true DBP로 사용된다.

$$
SBP = mean(SBP_i)
$$

$$
DBP = mean(DBP_i)
$$

### 3.4 BPNet architecture

BPNet은 ECG와 PPG를 위한 두 개의 별도 branch를 가진다. 각 branch는 SEResNet 기반 CNN 구조를 사용하여 signal feature map을 추출한다. 논문 그림 기준으로 각 branch는 SEResNet block 이후 Res1, Res2, Res3, Res4 단계의 feature를 생성한다. 즉 ECG와 PPG 각각에서 여러 level의 representation을 얻는다.

특정 CNN layer에서 추출된 feature map은 $V \in R^{C \times L}$로 표현된다. 여기서 $C$는 channel dimension이고, $L$은 signal length dimension이다. PPG와 ECG feature vector를 각각 $V_{ppg}$, $V_{ecg}$라고 할 때, 특정 level에서의 cross-modal feature는 다음과 같이 정의된다.

$$
f_p = Concat\left(\frac{V_{ppg}}{|V_{ppg}|*2}, \frac{V*{ecg}}{|V_{ecg}|_2}\right)
$$

여기서 $|\cdot|_2$는 L2 norm이고, $Concat(\cdot)$은 channel dimension에서의 concatenation이다. 중요한 점은 단순히 raw ECG와 raw PPG를 붙이는 것이 아니라, 각 modality에서 CNN feature를 추출한 뒤 normalize하고 level별로 fusion한다는 것이다. 이 방식은 ECG와 PPG의 feature scale 차이를 줄이고, modality-specific feature를 공유 공간으로 옮겨 결합하려는 설계이다.

BPNet은 네 개의 joint multi-modal feature-sharing subspace를 구성한다. 즉 여러 CNN stage에서 나온 ECG와 PPG feature가 각각 Concat1, Concat2, Concat3, Concat4로 결합된다. 이후 FPN, feature pyramid network를 사용하여 multi-level feature를 통합한다. FPN은 deep layer의 semantic information과 shallow layer의 positional information을 결합하는 구조로, 원래 object detection에서 자주 사용되지만 이 논문에서는 ECG와 PPG waveform의 multi-scale morphology를 포착하기 위해 사용된다.

논문은 기존 FPN을 약간 수정하여 fused feature map의 마지막 layer만 사용한다고 설명한다. 이 feature map은 가장 크며 더 작은 waveform detail에 집중할 수 있다고 주장한다. FPN에서 1x1 convolution은 channel 수를 맞추는 데 사용되고, upsample과 element-wise summation을 통해 상위 level feature와 하위 level feature가 결합된다.

### 3.5 Post-feature extraction

FPN으로 융합된 feature는 shared space에 있으므로, 이를 곧바로 SBP와 DBP estimation에 사용하면 parameter sharing이 과도해질 수 있다. 이를 완화하기 위해 post-feature extraction module을 추가한다. 이 module은 먼저 convolution layer로 feature를 smoothing한 뒤, 두 개의 convolution layer를 추가로 사용해 feature를 추출한다. 논문 그림에서는 white convolution module이 stride 1의 3x3 convolution이고, light blue module이 stride 2의 3x3 convolution이다.

이 과정은 두 가지 목적을 가진다. 첫째, multi-level fusion으로 얻은 feature가 불안정할 수 있으므로 이를 안정화한다. 둘째, SBP와 DBP task가 shared representation을 사용하되, 각 task에 필요한 feature를 더 적절히 분리할 수 있게 한다.

### 3.6 Multi-task structure와 loss function

BPNet은 SBP와 DBP를 별도 network로 독립적으로 학습하지 않고, 하나의 multi-task structure에서 동시에 추정한다. 기존 연구에서는 SBP와 DBP를 각각 별도 network로 추정하는 경우가 많았지만, 이 논문은 두 task가 생리적으로 관련되어 있으므로 shared representation을 사용하는 것이 accuracy와 efficiency 측면에서 유리하다고 본다.

최종 SBP branch와 DBP branch는 각각 별도의 MLP module로 구성된다. 각 MLP는 두 개의 linear layer로 이루어져 최종 혈압값을 출력한다. 다만 multi-task learning에서는 두 task의 loss scale이 크게 다르면 전체 model이 한 task에 치우쳐 학습될 수 있다. 이를 해결하기 위해 논문은 다음 loss function을 설계한다.

$$
Loss = \frac{1}{n}\sum_i^n \left[(SBP_{pi} - SBP_{ti})^2 + corrcoef \cdot (DBP_{pi} - DBP_{ti})^2\right]
$$

여기서 $SBP_{pi}$와 $DBP_{pi}$는 predicted SBP와 predicted DBP이고, $SBP_{ti}$와 $DBP_{ti}$는 true SBP와 true DBP이다. $corrcoef$는 scaling factor이며 다음과 같이 정의된다.

$$
corrcoef = \frac{(SBP_{pi} - SBP_{ti})^2}{(DBP_{pi} - DBP_{ti})^2}
$$

이 설계의 의도는 SBP task와 DBP task의 loss value scale을 맞추어, 전체 모델이 특정 task에 과도하게 치우치지 않도록 하는 것이다. 다만 비판적으로 보면, 이 $corrcoef$ 정의는 현재 batch 또는 sample의 error ratio에 직접 의존하므로 numerical stability, zero division 방지, gradient behavior 등에 대한 추가 설명이 필요하다. 원문에는 이 부분의 안정화 기법이 자세히 설명되어 있지 않다.

## 4. 실험 및 결과

### 4.1 데이터셋 및 실험 설정

논문은 MIMIC-II와 MIMIC-III를 각각 별도로 사용한다. MIMIC-II는 Kachuee 등이 compiled한 version을 사용했고, 942명 환자와 245,638개 sample로 구성된다. MIMIC-III는 저자들이 직접 compiled했으며 833명 환자, 총 833,000개 sample로 구성된다. 각 환자에서 1000개의 sample을 추출하여 환자별 sample 수를 동일하게 맞추었다. 논문 표에는 MIMIC-III total samples가 8,330,000으로 적힌 부분이 있으나, 본문 implementation detail과 train, validation, test sample 수를 합산하면 833,000개가 일관된다. 따라서 표의 8,330,000은 오기일 가능성이 있다.

각 sample의 길이는 10초, 즉 1250 sample point이며 overlap은 없다. 데이터 분할은 sample 단위로 64% train, 16% validation, 20% test로 수행된다. MIMIC-II에서는 train 157,208개, validation 39,303개, test 49,127개 sample을 사용한다. MIMIC-III에서는 train 533,120개, validation 133,280개, test 166,600개 sample을 사용한다.

학습은 PyTorch로 수행되며, Ubuntu 16.04.1 LTS 환경에서 24GB memory를 가진 4개의 RTX 3090 GPU를 사용한다. CPU는 10-core Xeon Silver 4210이고 RAM은 128GB이다. 초기 learning rate는 $1e^{-3}$이며, evaluation metric이 2 epoch 동안 개선되지 않으면 learning rate를 기존의 0.2배로 줄인다. 최소 learning rate는 $1e^{-5}$로 설정된다.

### 4.2 평가 지표

논문은 Pearson correlation coefficient $r$, mean error, ME, mean absolute error, MAE, standard deviation, STD를 사용한다.

Pearson correlation coefficient는 추정값과 true value의 선형 상관성을 측정한다.

$$
r =
\frac{\sum_{i=1}^{n}(y_i-\bar{y})(\hat{y}*i-\bar{\hat{y}})}
{\sqrt{\sum*{i=1}^{n}(y_i-\bar{y})^2}\sqrt{\sum_{i=1}^{n}(\hat{y}_i-\bar{\hat{y}})^2}}
$$

ME는 예측값이 평균적으로 true value보다 얼마나 높거나 낮은지 나타내는 bias 지표이다.

$$
ME = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)
$$

MAE는 오차의 방향을 무시한 평균 절대 오차이다.

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|\hat{y}_i-y_i|
$$

STD는 ME를 제거한 뒤 error의 분산 정도를 나타낸다.

$$
STD =
\sqrt{
\frac{\sum_{i=1}^{n}(\hat{y}_i-y_i-ME)^2}{n-1}
}
$$

또한 의료기기 평가 관점에서 BHS standard와 AAMI standard를 사용한다. BHS standard는 error가 5 mmHg, 10 mmHg, 15 mmHg 이내에 들어오는 누적 비율을 기준으로 Grade A, B, C를 부여한다. AAMI standard는 최소 85명 이상의 subject를 대상으로 ME가 5 mmHg 이하이고 STD가 8 mmHg 이하인지를 본다.

### 4.3 Ablation study

논문은 MIMIC-II 데이터셋에서 ablation study를 수행하여 Box-Cox transformation, cross-modal fusion, multi-task structure의 기여를 평가한다.

Box-Cox를 사용하지 않은 모델은 SBP에서 ME -0.16, MAE 4.15, STD 4.68, $r=0.9540$이고, DBP에서 ME 0.23, MAE 2.57, STD 2.99, $r=0.9273$이다. 최종 모델은 SBP에서 MAE 3.98, STD 4.62, $r=0.9564$이고, DBP에서 MAE 2.33, STD 2.95, $r=0.9340$이다. 개선폭이 아주 크지는 않지만, Box-Cox가 특히 DBP MAE와 correlation에 긍정적으로 작용한 것으로 보인다.

Cross-modal fusion을 제거한 no fusion 모델은 SBP에서 ME -0.63, MAE 4.65, STD 5.37, $r=0.9407$이고, DBP에서 ME -0.71, MAE 2.69, STD 3.38, $r=0.9137$이다. 최종 모델과 비교하면 multi-level fusion이 SBP와 DBP 모두에서 MAE, STD, correlation을 확실히 개선한다. 이는 논문이 주장하는 “단순 ECG-PPG concatenation보다 cross-modal multi-level fusion이 더 효과적”이라는 결론을 뒷받침한다.

Single-task 모델은 SBP에서 MAE 4.38, STD 4.77, $r=0.9506$이고, DBP에서 MAE 2.57, STD 3.14, $r=0.9229$이다. 2D regression 모델은 SBP에서 MAE 4.71, STD 4.92, $r=0.9464$이고, DBP에서 MAE 2.68, STD 3.27, $r=0.9165$이다. 최종 multi-task 구조는 이들보다 더 좋은 성능을 보인다. 이는 SBP와 DBP를 완전히 분리하거나 단순한 2D regression으로 처리하는 것보다, shared feature와 task-specific MLP head를 결합하는 방식이 더 효과적임을 보여준다.

### 4.4 MIMIC-II 결과

MIMIC-II에서 BPNet의 최종 성능은 SBP와 DBP 모두 매우 우수하다. SBP는 ME -0.17 mmHg, MAE 3.98 mmHg, STD 4.62 mmHg, $r=0.9564$를 달성했다. DBP는 ME -0.24 mmHg, MAE 2.33 mmHg, STD 2.95 mmHg, $r=0.9340$을 달성했다.

BHS standard 기준으로 MIMIC-II SBP는 5 mmHg 이내 74.47%, 10 mmHg 이내 92.52%, 15 mmHg 이내 97.05%를 기록하여 Grade A를 달성했다. DBP는 5 mmHg 이내 89.57%, 10 mmHg 이내 97.70%, 15 mmHg 이내 99.13%로 역시 Grade A이다. AAMI standard 기준으로도 SBP의 ME -0.17, STD 4.62와 DBP의 ME -0.24, STD 2.95는 모두 기준을 만족한다.

논문은 MIMIC-II에 대해 correlation plot과 Bland-Altman plot을 제공한다. Correlation plot에서는 추정값이 reference line 주변에 밀집하여 높은 상관성을 보인다. Bland-Altman plot에서는 대부분의 point가 95% limit of agreement 안에 들어가며, 이는 추정 오차가 비교적 안정적으로 분포함을 보여준다.

### 4.5 MIMIC-III 결과

MIMIC-III에서도 BPNet은 좋은 성능을 보였다. SBP는 ME -0.30 mmHg, MAE 5.59 mmHg, STD 5.78 mmHg, $r=0.9295$이고, DBP는 ME -0.25 mmHg, MAE 3.35 mmHg, STD 3.80 mmHg, $r=0.9265$이다.

BHS standard 기준으로 MIMIC-III SBP는 5 mmHg 이내 59.47%, 10 mmHg 이내 84.90%, 15 mmHg 이내 93.73%로 Grade B이다. DBP는 5 mmHg 이내 79.90%, 10 mmHg 이내 95.01%, 15 mmHg 이내 98.24%로 Grade A이다. AAMI 기준은 SBP와 DBP 모두 만족한다. 특히 AAMI 기준은 ME와 STD를 중심으로 보기 때문에, MIMIC-III SBP의 STD 5.78과 DBP의 STD 3.80은 기준인 8 mmHg보다 낮다.

MIMIC-II보다 MIMIC-III에서 SBP 성능이 다소 낮아진 것은 데이터 구성, preprocessing 난이도, patient variability의 차이 때문일 수 있다. 논문은 MIMIC-III compiled version을 직접 만들었고, cleaning이 필요했다고 명시한다. 따라서 MIMIC-III는 MIMIC-II보다 더 어려운 평가 환경으로 볼 수 있다.

### 4.6 Segment length 비교

논문은 segment length 변화에 대한 robustess도 확인한다. MIMIC-II에서 5초, 8초, 10초, 12초, 15초 segment를 비교했으며, 10초를 최종 설정으로 사용한다. 10초 segment는 SBP MAE 3.98, STD 4.62, $r=0.9564$이고, DBP MAE 2.33, STD 2.95, $r=0.9340$으로 가장 균형 잡힌 결과를 보였다. 8초 segment의 SBP MAE는 3.93으로 조금 낮지만 DBP MAE는 2.38로 10초보다 높다. 12초와 15초에서는 전반적으로 성능이 떨어진다. 논문은 segment length 변화가 모델에 큰 영향을 주지는 않는다고 설명하지만, 결과표를 보면 10초가 특히 DBP에서 가장 우수하다.

### 4.7 다른 deep learning 모델과의 비교

MIMIC-II에서 BPNet은 ResNet18, SEResNet18, LSTM보다 우수했다. ResNet18은 SBP MAE 5.12, DBP MAE 2.93을 보였고, SEResNet18은 SBP MAE 4.60, DBP MAE 2.68을 보였다. LSTM은 SBP MAE 13.45, DBP MAE 6.73으로 매우 낮은 성능을 보였다. BPNet은 SBP MAE 3.98, DBP MAE 2.33으로 가장 좋았다.

MIMIC-III에서도 같은 경향이 나타난다. ResNet34는 SBP MAE 6.67, DBP MAE 4.02이고, SEResNet34는 SBP MAE 5.82, DBP MAE 3.50이다. LSTM은 SBP MAE 15.73, DBP MAE 9.30으로 매우 낮은 성능이다. BPNet은 SBP MAE 5.59, DBP MAE 3.35로 가장 좋은 결과를 보였다. 이는 이 task에서 단순 temporal model인 LSTM보다 CNN 기반 waveform morphology extraction과 multi-modal fusion이 더 효과적이었다는 논문의 설계를 뒷받침한다.

### 4.8 기존 연구와의 비교

MIMIC-II에서 BPNet은 기존 연구들과 비교해 거의 모든 지표에서 우수했다. 예를 들어 한 비교 연구는 SBP MAE 5.16, DBP MAE 2.89를 보였고, 다른 연구는 SBP MAE 5.73, DBP MAE 3.45를 보였다. BPNet은 SBP MAE 3.98, DBP MAE 2.33으로 더 낮은 오차를 기록했다.

MIMIC-III에서도 BPNet은 비교적 우수한 성능을 보였다. 기존 연구 중 하나는 SBP MAE 7.10, DBP MAE 4.61을 보고했고, 다른 PPG 기반 연구는 SBP MAE 9.43, DBP MAE 6.88을 보고했다. BPNet은 SBP MAE 5.59, DBP MAE 3.35로 더 낮은 오차를 달성했다.

다만 비교 연구들의 subject 수, preprocessing 방식, train-test split 방식, input modality, target definition이 서로 다를 수 있으므로, 이러한 table 비교를 절대적인 superiority 증거로만 해석하는 것은 조심해야 한다. 그럼에도 논문 내부에서 동일 데이터셋과 동일 실험 설정으로 비교한 ResNet, SEResNet, LSTM 대비 우수한 결과는 BPNet architecture의 효과를 비교적 설득력 있게 보여준다.

### 4.9 실행 시간

BPNet은 inference time 측면에서도 실용성을 보여준다. MIMIC-II test set 49,127 sample에서 single GPU inference는 총 103초, sample당 약 0.002초가 걸렸다. CPU inference는 총 1250초, sample당 약 0.03초가 걸렸다. MIMIC-III test set 166,600 sample에서는 GPU 총 499초, sample당 0.003초, CPU 총 4998초, sample당 0.03초가 걸렸다.

이 결과는 BPNet이 매우 큰 모델이라기보다는 비교적 빠른 inference가 가능한 모델임을 보여준다. 다만 이 논문은 실제 microcontroller나 wearable hardware에 배포한 결과를 제공하지는 않는다. CPU/GPU server 환경에서의 inference time이 wearable device에서의 latency와 memory requirement를 직접 보장하지는 않는다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 혈압 추정 문제를 단순한 neural network regression 문제가 아니라, label distribution, data leakage, modality fusion이라는 세 가지 핵심 문제와 연결하여 다룬 점이다. 특히 Box-Cox transformation을 통해 skewed label distribution 문제를 regression assumption 관점에서 해석한 부분은 이 분야에서 중요한 방법론적 기여로 볼 수 있다.

두 번째 강점은 non-overlapping sample 구성이다. 생체신호 deep learning에서 overlapping window는 성능을 과대평가할 위험이 크다. 이 논문은 이 문제를 명시적으로 지적하고, 10초 non-overlapping segment를 사용한다. 이는 기존 연구와 비교할 때 더 엄격한 평가를 시도한 것으로 볼 수 있다.

세 번째 강점은 BPNet architecture의 설계가 비교적 잘 정당화되어 있다는 점이다. ECG와 PPG는 각각 다른 생리적 정보를 가지므로, 별도 CNN branch를 통해 modality-specific feature를 추출하고, multi-level에서 fusion하는 방식은 타당하다. FPN을 사용하여 deep semantic information과 shallow positional information을 결합한 것도 waveform morphology를 다루는 데 적절한 선택이다. 또한 SBP와 DBP를 동시에 추정하는 multi-task structure는 두 혈압 값 사이의 생리적 관련성을 고려한다는 점에서 설득력이 있다.

네 번째 강점은 실험의 폭이다. 논문은 MIMIC-II와 MIMIC-III 두 데이터셋에서 평가하고, ablation study, 다른 segment length 비교, ResNet/SEResNet/LSTM과의 비교, 기존 연구와의 비교, BHS와 AAMI standard 평가를 모두 제공한다. 특히 MIMIC-II와 MIMIC-III에서 각각 AAMI 기준을 만족한 점은 임상적 측정 기준 관점에서 의미가 있다.

그러나 한계도 분명하다. 첫째, 데이터 분할이 sample-level random split로 보인다. 논문은 64%, 16%, 20% sample을 train, validation, test로 random selection한다고 설명한다. Non-overlapping segment를 사용해 adjacent window leakage는 줄였지만, 같은 환자의 다른 segment가 train과 test에 동시에 포함될 가능성이 있다. BP estimation에서는 subject-specific waveform pattern을 모델이 학습할 수 있으므로, subject-independent split이 아니면 실제 unseen patient에 대한 generalization 성능이 과대평가될 수 있다. 이 점은 논문의 가장 중요한 한계이다.

둘째, Box-Cox transformation의 적용 세부가 충분히 설명되지 않는다. $\lambda$를 maximum likelihood로 구한다고 설명하지만, SBP와 DBP 각각에 별도 $\lambda$를 적용했는지, train set에서만 $\lambda$를 추정했는지, validation/test label 변환과 inverse transform을 어떻게 처리했는지는 명확하지 않다. 이러한 세부 사항은 reproducibility와 data leakage 여부 판단에 중요하다.

셋째, loss function의 scaling factor 정의가 다소 불안정해 보인다. $corrcoef$가 SBP squared error를 DBP squared error로 나눈 형태라면, DBP error가 매우 작을 때 값이 커질 수 있다. 논문은 이로써 두 task의 loss value를 같게 만든다고 설명하지만, numerical stability를 위한 epsilon 추가 여부나 batch-level averaging 여부는 명시되어 있지 않다.

넷째, BPNet은 server GPU/CPU 환경에서 inference time을 보고하지만, 실제 wearable 또는 edge device 배포 결과는 없다. 논문 결론에서는 wearable device의 clinical application 가능성을 언급하지만, memory footprint, power consumption, embedded latency는 실험적으로 검증되지 않았다. 따라서 실제 wearable 적용 가능성은 아직 추정 수준이다.

다섯째, MIMIC 데이터는 ICU 환자 기반 데이터이다. ICU 환자의 혈압, 약물 상태, 질환 상태, signal quality는 일반 wearable 사용자의 일상 환경과 다를 수 있다. 따라서 MIMIC에서의 높은 성능이 ambulatory monitoring, 운동 중 측정, 수면 중 측정, 다양한 sensor placement 환경에서도 유지될지는 추가 검증이 필요하다.

비판적으로 평가하면, BPNet은 ECG와 PPG를 활용한 BP estimation 연구에서 매우 체계적인 architecture와 실험을 제시한 강한 논문이다. 특히 multi-modal fusion과 data leakage 문제의식은 타당하다. 그러나 sample-level split과 실제 unseen-subject generalization 검증 부족은 결과 해석에서 주의해야 할 지점이다. 이 모델이 clinical setting에서 정말 robust하게 작동하려면 subject-independent external validation과 실제 device 환경 검증이 필요하다.

## 6. 결론

이 논문은 ECG와 PPG를 이용한 continuous blood pressure estimation을 위해 BPNet이라는 multi-modal fusion neural network를 제안했다. BPNet은 PPG branch와 ECG branch에서 각각 SEResNet 기반 feature를 추출하고, 여러 level에서 cross-modal feature를 결합한 뒤, FPN으로 multi-scale feature를 통합한다. 이후 post-feature extraction과 multi-task MLP branch를 통해 SBP와 DBP를 동시에 추정한다.

논문의 주요 기여는 세 가지로 요약된다. 첫째, Box-Cox transformation을 사용하여 skewed BP label distribution을 보정하고 MSE 기반 regression의 통계적 가정을 고려했다. 둘째, non-overlapping segment를 사용하여 overlapping sample로 인한 data leakage를 줄이려 했다. 셋째, ECG와 PPG를 단순 concatenation하지 않고 cross-modal multi-level fusion을 통해 두 physiological signal의 관계를 더 효과적으로 학습했다.

실험적으로 BPNet은 MIMIC-II에서 SBP MAE 3.98 mmHg, DBP MAE 2.33 mmHg를 달성했고, MIMIC-III에서 SBP MAE 5.59 mmHg, DBP MAE 3.35 mmHg를 달성했다. AAMI standard는 두 데이터셋 모두에서 만족했으며, BHS standard에서는 MIMIC-II의 SBP와 DBP가 모두 Grade A, MIMIC-III에서는 SBP Grade B와 DBP Grade A를 달성했다. Ablation study는 Box-Cox transformation, cross-modal fusion, multi-task structure가 각각 최종 성능 향상에 기여함을 보여준다.

이 연구는 향후 cuffless BP monitoring, ICU signal analysis, ECG-PPG 기반 wearable health monitoring에서 중요한 참고점이 될 수 있다. 특히 modality fusion과 label distribution correction을 함께 고려한 점은 다른 biomedical regression task에도 적용 가능성이 있다. 다만 실제 임상 및 wearable 적용을 위해서는 subject-independent split, external dataset validation, ambulatory 환경 테스트, embedded device 배포 실험이 추가로 필요하다. 따라서 BPNet은 실제 제품 수준의 완결된 시스템이라기보다는, ECG와 PPG 기반 혈압 추정을 더 엄격하고 구조적으로 수행하기 위한 강력한 연구적 기반으로 평가할 수 있다.
