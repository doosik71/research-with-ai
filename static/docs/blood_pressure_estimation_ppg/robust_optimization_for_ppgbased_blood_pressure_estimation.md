# Robust optimization for PPG-based blood pressure estimation

* **저자**: Sungjun Lim, Taero Kim, Hyeonjeong Lee, Yewon Kim, Minhoi Park, Kwang-Yong Kim, Minseong Kim, Kyu Hyung Kim, Jiyoung Jung, Kyungwoo Song
* **발표연도**: 2025

## 1. 논문 개요

이 논문은 PPG(Photoplethysmography) 신호만을 사용하여 cuff-less 방식으로 혈압을 추정할 때, 평균 성능뿐 아니라 특정 혈압군, 특히 고위험 또는 소수 혈압군에서의 성능 저하를 줄이기 위한 robust optimization 프레임워크를 제안한다. 기존 PPG 기반 혈압 추정 연구들은 주로 전체 테스트셋의 평균 MAE나 RMSE를 낮추는 데 집중했지만, 실제 의료 환경에서는 Hypotension이나 Hypertensive crisis처럼 데이터 수가 적고 임상적으로 중요한 집단에서 모델이 크게 실패할 수 있다. 이 논문은 이러한 집단을 “worst group”으로 정의하고, 혈압군 간 성능 격차를 줄이는 것을 주요 연구 문제로 삼는다.

논문은 BP 그룹을 Hypo, Normal, Prehyper, Hyper2, Crisis로 나눈다. 기준은 SBP와 DBP 범위에 따라 정의된다. 예를 들어 Hypo는 SBP 80–90 mmHg, DBP 40–60 mmHg에 해당하고, Crisis는 SBP 180–200 mmHg, DBP 120–130 mmHg에 해당한다. 데이터 분포는 매우 불균형하다. MIMIC-III 기반 처리 데이터에서 training set 기준 Normal은 7696개, Prehyper는 6274개, Hyper2는 4821개인 반면, Hypo는 537개, Crisis는 276개에 불과하다. 따라서 단순 ERM(Empirical Risk Minimization)으로 학습하면 데이터가 많은 Normal, Prehyper, Hyper2에 맞춰지고, Hypo나 Crisis 같은 소수 고위험군에서는 큰 오차가 발생할 수 있다.

논문은 이러한 문제를 data, model, loss의 세 관점에서 해결한다. 데이터 관점에서는 같은 혈압군 안에서 두 PPG sequence를 시간축으로 잘라 붙이는 Time-CutMix(TC)를 제안한다. 모델 관점에서는 convolution layer와 Transformer Encoder를 결합한 ConvTransformer를 사용하여 PPG 신호의 local context와 global context를 동시에 포착한다. 손실 함수 관점에서는 그룹별 데이터 수와 label distribution의 차이를 반영하는 C-REx, D-REx, CD-REx라는 새로운 robust regularization 기법을 제안한다.

논문의 핵심 목표는 단순히 전체 평균 MAE를 낮추는 것이 아니라, 혈압군별 성능 격차를 줄이고, 특히 소수이면서 임상적으로 중요한 Hypo와 Crisis 그룹에서의 성능을 개선하는 것이다. 제공된 실험 결과에 따르면 ConvTransformer는 일반 Transformer보다 전체 평균 성능과 그룹별 성능에서 크게 우수하며, CD-REx와 Time-CutMix를 결합한 방법은 full data와 small data 설정 모두에서 group average loss와 worst group loss를 가장 효과적으로 낮춘다. 또한 PPGBP benchmark dataset에서도 다양한 backbone에 제안 방법을 적용했을 때 group average 및 worst group 성능이 대체로 개선됨을 보인다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 PPG 기반 혈압 추정 문제가 단순한 회귀 문제가 아니라, 혈압군 간 데이터 불균형과 label distribution shift를 포함하는 imbalanced regression 문제라는 점이다. 일반적인 ERM은 전체 데이터 평균 손실을 최소화하므로 데이터 수가 많은 그룹의 성능에 치우치기 쉽다. 하지만 실제 의료 응용에서는 평균 성능보다도 위험도가 높은 소수 그룹에서의 안정적인 예측이 중요하다. 예를 들어 Hypertensive crisis 환자에서 혈압을 크게 낮게 예측하거나 Hypotension 환자에서 혈압을 높게 예측하면 임상적으로 위험한 판단을 유발할 수 있다.

논문은 기존 worst-group optimization을 혈압 회귀 문제에 맞게 확장한다. 기존 GDRO나 V-REx는 주로 classification 또는 domain robustness 문제에서 각 그룹 손실을 균형 있게 다루는 데 사용되었다. 그러나 혈압 추정은 regression task이기 때문에 class label 간 순서가 없는 classification과 다르다. 혈압값은 연속적이고 순서가 있으며, 평균제곱오차(MSE)를 사용하면 모델이 label distribution의 평균 부근에 잘 맞춰지는 경향이 있다. 이 때문에 극단적인 혈압 구간, 즉 Hypo와 Crisis에서는 예측 성능이 더 나빠질 수 있다.

이 논문은 이러한 회귀 문제의 특성을 반영하여 두 가지 정보를 robust regularization에 포함한다. 첫째, 각 그룹의 데이터 수이다. 데이터가 적은 그룹은 학습 중 충분히 대표되지 않으므로 더 강하게 regularization해야 한다. 이를 반영한 것이 C-REx(Count-aware Risk Extrapolation)이다. 둘째, 각 그룹의 label distribution이 전체 label distribution에서 얼마나 멀리 떨어져 있는지이다. 극단적인 혈압군은 전체 데이터 평균에서 멀리 떨어져 있으므로 모델이 평균값에 과도하게 맞춰질 때 손해를 본다. 이를 반영한 것이 D-REx(Divergence-aware Risk Extrapolation)이다. 두 요소를 결합한 것이 CD-REx이다.

모델 측면의 핵심 아이디어는 PPG 신호에는 peak, foot, dicrotic notch와 같은 local morphology 정보뿐 아니라 신호 전반의 global temporal relationship도 중요하다는 것이다. Transformer는 global context를 잘 포착할 수 있지만, raw sequence에서 local physiological pattern을 충분히 포착하지 못할 수 있다. 반면 convolution layer는 국소 파형 구조를 잘 잡는다. ConvTransformer는 convolution layer를 통해 local context를 먼저 추출하고, Transformer Encoder로 global context를 통합한다.

데이터 측면의 핵심 아이디어는 같은 혈압군 안에서만 sequence를 섞는 것이다. Time-CutMix는 서로 같은 BP group에 속하는 두 PPG segment를 시간축 기준으로 자르고 붙인다. label도 자른 비율에 따라 선형 결합한다. 이렇게 하면 소수 그룹 안에서도 다양한 학습 sample을 만들 수 있으며, 다른 혈압군끼리 섞어서 label 의미를 흐리는 문제를 피한다. 논문은 이 방식이 mixed sample data augmentation의 특수한 경우이며, 특정 가정하에서 generalization bound를 더 타이트하게 만들 수 있음을 이론적으로 분석한다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

이 논문이 제안하는 PPG 기반 혈압 추정 시스템은 다음과 같이 구성된다. 먼저 PPG와 ABP가 동시에 존재하는 데이터를 사용한다. PPG는 모델 입력이고, ABP에서 추출한 SBP와 DBP는 정답 label이다. MIMIC-III 데이터의 경우 8초 길이, 즉 $T=1000$ sample의 PPG segment를 사용한다. sampling rate는 125 Hz이므로 1000 sample은 8초에 해당한다. PPG는 min-max normalization 후 모델에 입력된다. 모델은 하나의 PPG sequence $x \in \mathbb{R}^{T}$를 받아 SBP와 DBP 예측값 $\hat{y}*{SBP}$, $\hat{y}*{DBP}$를 출력한다.

논문의 전체 접근은 다음 세 축으로 요약된다.

첫째, backbone model로 ConvTransformer를 사용한다. 이는 convolution layer와 Transformer Encoder를 결합하여 PPG의 local morphology와 global sequence dependency를 함께 학습한다.

둘째, 학습 objective에는 ERM뿐 아니라 group-aware robust regularization을 적용한다. C-REx는 group count imbalance를 고려하고, D-REx는 label distribution divergence를 고려한다. CD-REx는 두 정보를 모두 결합한다.

셋째, training data에는 Time-CutMix augmentation을 적용할 수 있다. 이는 같은 혈압군에 속한 두 sequence를 시간축으로 결합하고 label도 비율에 맞게 결합하는 방식이다.

### 3.2 ConvTransformer 구조

ConvTransformer는 PPG sequence에서 local context와 global context를 동시에 추출하기 위해 설계된 모델이다. 입력은 단일 PPG time sequence $x \in \mathbb{R}^{T}$이다. 먼저 linear layer를 통해 입력을 $\mathbb{R}^{D \times T}$ 형태의 hidden representation으로 변환한다. 이후 convolution layer를 적용하여 local context를 추출한다.

논문은 convolution filter size로 3, 5, 7, 9를 사용하고, 각 filter size마다 8개의 filter를 사용한다. 여러 kernel size를 사용하는 이유는 PPG waveform의 국소 패턴이 다양한 시간 폭에서 나타날 수 있기 때문이다. 예를 들어 peak 주변의 sharp한 변화는 작은 kernel에서 잘 포착될 수 있고, pulse cycle의 더 넓은 형태는 큰 kernel에서 유리할 수 있다.

Convolution output은 Transformer Encoder layer로 전달된다. Transformer는 self-attention을 통해 time step 간 global dependency를 학습한다. 논문은 일반 Transformer의 positional encoder 대신 convolutional layer output을 Transformer 입력으로 사용한다. 이는 PPG에서 단순 위치 정보보다 local waveform structure가 더 중요하다는 가정에 기반한다.

마지막으로 Transformer output에 max-pooling을 적용하고 output layer를 통해 SBP와 DBP를 예측한다. 전체 흐름은 다음과 같이 표현할 수 있다.

$$
x
\rightarrow Linear
\rightarrow Multi\text{-}scale\ Conv1D
\rightarrow Transformer\ Encoder
\rightarrow MaxPooling
\rightarrow Output
\rightarrow (\hat{y}_{SBP}, \hat{y}_{DBP})
$$

실험에서 ConvTransformer는 Transformer-only 모델보다 크게 좋은 성능을 보인다. Table 2에 따르면 data average MAE는 Transformer 26.88 mmHg에서 ConvTransformer 20.08 mmHg로 개선되고, group average MAE는 40.03에서 25.52로 개선되며, worst group MAE는 74.36에서 43.44로 개선된다. 특히 Hypo와 Crisis 그룹에서 SBP와 DBP 오차가 대략 30–50% 수준으로 감소한다. 이는 convolution layer가 PPG의 peak와 foot point 같은 국소 생리 구조를 포착하는 데 중요한 역할을 함을 시사한다.

### 3.3 기본 회귀 손실과 그룹별 위험

기본 회귀 손실은 각 BP group $g$에 대해 MSE로 정의된다. 그룹 $g$의 정답 label을 $y_{g,i}$, 예측값을 $\hat{y}*{g,i}=f*{\theta}(x_{g,i})$, 해당 그룹 sample 수를 $N_g$라고 할 때 그룹별 risk는 다음과 같다.

$$
R_g(\theta) = \frac{1}{N_g} \sum_{i=1}^{N_g} (y_{g,i}-\hat{y}_{g,i})^2
$$

전체 ERM은 보통 모든 sample에 대한 평균 손실을 최소화한다. 하지만 이 방식은 그룹 크기 $N_g$가 큰 그룹에 더 큰 영향을 받는다. 이 논문은 모델 선택 및 평가에서 다음 세 기준을 구분한다.

데이터 평균 손실은 전체 sample 기준 평균 MSE이다.

$$
R_{Data} := \frac{1}{N} \sum_{i=1}^{N} (y_{\cdot,i}-\hat{y}_{\cdot,i})^2
$$

그룹 평균 손실은 각 그룹별 평균 손실을 동일한 비중으로 평균낸다.

$$
R_{Group} := \frac{1}{G} \sum_{g=1}^{G} \frac{1}{N_g} \sum_{i=1}^{N_g} (y_{g,i}-\hat{y}_{g,i})^2
$$

Worst group loss는 가장 손실이 큰 그룹의 손실이다.

$$
R_{Worst} := \max_{g \in G} \frac{1}{N_g} \sum_{i=1}^{N_g} (y_{g,i}-\hat{y}_{g,i})^2
$$

이 구분이 중요한 이유는 ERM은 $R_{Data}$를 낮추는 데 유리하지만, $R_{Group}$이나 $R_{Worst}$를 보장하지 못하기 때문이다. 의료 응용에서는 $R_{Worst}$가 매우 중요할 수 있다.

### 3.4 C-REx: Count-aware Risk Extrapolation

C-REx는 각 그룹의 데이터 수를 반영하는 robust regularization이다. 직관적으로, 데이터 수가 적은 그룹일수록 더 강하게 학습되도록 regularization term에 count-aware penalty를 추가한다. 전체 dataset 크기를 $N$, 그룹 $g$의 sample 수를 $N_g$라고 할 때, 논문은 $\sqrt{N/N_g}$를 그룹별 count imbalance 정도로 사용한다. $N_g$가 작을수록 이 값은 커진다.

C-REx objective는 다음과 같다.

$$
R_C(\theta) = \sum_{g=1}^{G} R_g(\theta) + \delta Var \left( R_1(\theta)+\alpha\sqrt{\frac{N}{N_1}}, \ldots, R_G(\theta)+\alpha\sqrt{\frac{N}{N_G}} \right)
$$

여기서 $\alpha$와 $\delta$는 regularization 강도를 조절하는 hyperparameter이다. $Var(\cdot)$는 그룹별 조정 손실의 분산을 의미한다. 이 항은 그룹 간 손실 차이가 커지는 것을 억제한다. 데이터 수가 적은 그룹은 count penalty가 커지므로, 해당 그룹의 손실이 더 중요하게 반영된다. 만약 모든 그룹의 데이터 수가 같다면 $N_1=\cdots=N_G$가 되어 C-REx는 V-REx와 유사한 형태로 퇴화한다.

### 3.5 D-REx: Divergence-aware Risk Extrapolation

D-REx는 각 그룹의 label distribution이 전체 label distribution과 얼마나 다른지를 반영한다. 회귀 문제에서는 classification과 달리 label 값의 순서와 위치가 중요하다. MSE 최적해는 평균값에 가까운 예측을 유도하는 경향이 있으므로, 전체 평균에서 멀리 떨어진 extreme group은 더 큰 손실을 겪을 수 있다. D-REx는 이를 완화하기 위해 각 그룹 label distribution과 전체 label distribution 사이의 divergence를 계산하고, divergence가 큰 그룹에 더 강한 regularization을 적용한다.

먼저 논문은 Tukey’s Ladder of Power Transformation을 사용하여 label distribution의 skewness를 줄이고 Gaussian-like distribution에 가깝게 만든다.

$$
\tilde{y} = \begin{cases} y^{\lambda}, & \lambda \neq 0 \ \log y, & \lambda = 0 \end{cases}
$$

논문에서는 $\lambda=0.5$를 사용한다. 전체 label $y=(y_{SBP},y_{DBP})$와 각 그룹 label $y_g=(y^g_{SBP},y^g_{DBP})$에 대해 변환을 적용하고, 변환된 label이 Gaussian distribution을 따른다고 가정한다. 전체 distribution은 $p_{total}=N(\mu(\tilde{y}),\Sigma(\tilde{y}))$, 그룹 distribution은 $p_g=N(\mu(\tilde{y}_g),\Sigma(\tilde{y}_g))$로 둔다.

각 그룹의 divergence는 symmetric KL divergence를 정규화하여 계산한다.

$$
Div_g = \frac{ \left( KLD(p_{total}\Vert p_g) + KLD(p_g\Vert p_{total}) \right)/2 }{ \sum_g \left( KLD(p_{total}\Vert p_g) + KLD(p_g\Vert p_{total}) \right)/2 }
$$

D-REx objective는 다음과 같다.

$$
R_D(\theta) = \sum_{g=1}^{G} R_g(\theta) + \delta Var \left( R_1(\theta)+\beta Div_1, \ldots, R_G(\theta)+\beta Div_G \right)
$$

여기서 $\beta=(\beta_{SBP},\beta_{DBP})$는 divergence regularization의 scale hyperparameter이다. 이 방식은 전체 label distribution에서 멀리 떨어진 Hypo나 Crisis 같은 extreme group의 손실을 더 중요하게 반영한다.

### 3.6 CD-REx: Count and Divergence-aware Risk Extrapolation

CD-REx는 C-REx와 D-REx를 결합한 objective이다. 즉, 데이터 수가 적은 그룹과 label distribution이 전체와 다른 그룹을 동시에 더 중요하게 고려한다. objective는 다음과 같다.

$$
R_{CD}(\theta) = \sum_{g=1}^{G} R_g(\theta) + \delta Var \left( R_1(\theta) + \alpha\sqrt{\frac{N}{N_1}} + \beta Div_1, \ldots, R_G(\theta) + \alpha\sqrt{\frac{N}{N_G}} + \beta Div_G \right)
$$

CD-REx의 장점은 worst group이 단순히 데이터 수가 적어서만 생기는 것이 아니라, label distribution의 극단성 때문에도 생길 수 있다는 점을 동시에 반영한다는 것이다. 혈압 추정에서 Hypo와 Crisis는 둘 다 데이터 수가 적고 label distribution상 극단에 위치하므로, CD-REx가 특히 적합하다.

### 3.7 Time-CutMix

Time-CutMix는 sequence data에 맞게 설계된 in-group augmentation 방법이다. 두 개의 입력 sequence $x_i,x_j \in \mathbb{R}^{T}$와 label $y_i,y_j$를 같은 BP group 안에서 선택한다. 이후 $\gamma$를 uniform distribution에서 샘플링한다.

$$
\gamma \sim Uniform(0,1)
$$

그다음 $x_i$의 앞쪽 $\gamma$ 비율 segment와 $x_j$의 뒤쪽 $1-\gamma$ 비율 segment를 이어 붙인다.

$$
\tilde{x}_{i,j} = Concat(x_{i,1:\gamma T},x_{j,\gamma T:T})
$$

label은 시간 비율에 따라 선형 결합한다.

$$
\tilde{y}_{i,j} = \gamma y_i+(1-\gamma)y_j
$$

Time-CutMix의 핵심은 서로 다른 혈압군끼리 섞지 않고, 같은 혈압군 내부에서만 섞는다는 점이다. 이 방식은 소수 그룹에서 다양한 training sample을 생성하면서도 혈압 label의 의미를 크게 훼손하지 않는다. 예를 들어 Hypo 그룹 안의 두 PPG segment를 섞으면 여전히 Hypo에 가까운 label을 갖는 자연스러운 augmented sequence를 만들 수 있다.

논문은 Time-CutMix가 random noise augmentation보다 자연스러운 sequence를 생성하며, TC를 CD-REx와 결합했을 때 full data와 small data에서 group average loss와 worst group loss가 모두 개선된다고 보고한다.

### 3.8 Time-CutMix 이론 분석

논문은 Time-CutMix가 mixed sample data augmentation(MSDA)의 특수한 경우라고 보고, generalized linear model(GLM) 설정에서 generalization bound를 분석한다. 기본 아이디어는 augmentation이 경험적 손실에 특정 regularization 효과를 유도하고, 같은 group 안에서 augmentation할 경우 covariance structure가 더 작아져 generalization bound가 더 타이트해진다는 것이다.

Augmented sample은 다음과 같이 표현된다.

$$
\tilde{x}_{i,j}(\gamma) = M(\gamma)\odot x_i + M(1-\gamma)\odot x_j
$$

$$
\tilde{y}_{i,j}(\gamma) = \gamma y_i+(1-\gamma)y_j
$$

GLM loss는 다음과 같이 정의된다.

$$
R_{GLM}(\theta,(x,y)) = A(\theta^T x)-y\theta^T x
$$

여기서 $A(\cdot)$는 log-partition function이다. 논문은 Taylor expansion과 Rademacher complexity를 사용하여 Time-CutMix의 generalization error bound를 제시한다. 그룹 정보를 사용하지 않는 TC의 generalization bound를 $GE_{w/o}$라고 하고, in-group TC의 generalization bound를 $GE_{w/}$라고 할 때, 특정 가정하에서 다음 관계가 성립한다고 주장한다.

$$
GE_{w/}
\le
GE_{w/o}
$$

핵심 가정은 같은 group 안의 데이터가 더 많은 공통 feature를 공유하므로, group별 covariance를 가중 평균한 $\tilde{\Sigma}$의 Frobenius norm이 전체 covariance $\Sigma$보다 작거나 같다는 것이다.

$$
|\tilde{\Sigma}|_F
\le
|\Sigma|_F
$$

이론 분석은 현실의 deep neural network를 완전히 설명하지는 않지만, 같은 혈압군 안에서 sequence를 섞는 것이 무작위 전체 mixing보다 더 안정적인 augmentation일 수 있음을 뒷받침한다.

## 4. 실험 및 결과

### 4.1 데이터셋

논문은 두 개의 데이터셋을 사용한다. 첫 번째는 MIMIC-III Waveform Database Matched Subset에서 자체 처리한 데이터셋이다. 이 데이터셋은 ICU 환자의 동기화된 ABP, PPG, ECG 신호를 포함하지만, 이 연구에서는 PPG만 입력으로 사용하고 ABP에서 SBP와 DBP label을 추출한다. sampling rate는 125 Hz이고, PPG segment 길이는 8초, 즉 1000 sample이다. 논문은 각 환자당 최대 100 segment만 사용하여 특정 환자에 과적합되는 것을 줄이려 한다. 단, train/validation/test split은 patient-wise가 아니라 uniform random sampling으로 수행된다. 저자들은 기존 환자와 신규 환자 모두에서 잘 예측하는 모델을 목표로 하기 때문에 patient-wise split을 사용하지 않았다고 명시한다.

전처리 과정은 비교적 정교하다. 8초 segment로 분할한 뒤 length check, value check, quality check, validity check를 수행한다. Flat line, flat peak, missing peak가 있는 segment를 제거하거나 짧은 결측 구간은 interpolation한다. PSDR(Power Spectrum Density Ratio), skewness, ABP mean morphology를 사용하여 품질이 낮은 segment를 제거한다. 마지막으로 ABP segment에서 peak와 foot point를 검출하여 SBP와 DBP를 계산하고, 생리적으로 부적절한 값은 제거한다. 이 과정을 통해 326명, 8763 segment의 MIMIC-III 처리 데이터셋을 구성한다.

두 번째 데이터셋은 PPGBP benchmark dataset이다. 이 데이터셋은 218명, 619 segment를 포함하며 sampling rate는 1000 Hz, segment length는 2.1초이다. 논문은 PPGBP의 기존 preprocessing을 그대로 사용하고, 추가 전처리는 하지 않는다. 이 데이터셋에서는 MLP-BP, SpectroResNet, ResNet, ConvTransformer 등 다양한 backbone에 제안 방법을 적용하여 일반성을 확인한다.

### 4.2 BP group 정의와 데이터 불균형

MIMIC-III 처리 데이터셋에서 BP group은 SBP와 DBP 기준으로 나뉜다. Training set 기준 데이터 수는 Hypo 537개(2.7%), Normal 7696개(39.3%), Prehyper 6274개(32.0%), Hyper2 4821개(24.6%), Crisis 276개(1.4%)이다. Validation과 test set에서도 비슷한 비율을 유지한다. 즉, Hypo와 Crisis는 명확한 minority group이며, 동시에 임상적으로 중요한 high-risk group이다.

이 불균형은 모델 성능에 직접적인 영향을 준다. ERM은 Normal, Prehyper, Hyper2처럼 많은 데이터를 가진 그룹에 성능이 편향되고, Hypo와 Crisis에서 높은 오차를 낸다. 논문은 이러한 문제를 평균 성능만으로는 파악하기 어렵다고 보고, data average loss, group average loss, worst group loss를 모두 평가한다.

### 4.3 Transformer와 ConvTransformer 비교

Table 2는 Transformer-only와 ConvTransformer를 비교한다. ConvTransformer는 거의 모든 지표에서 Transformer보다 우수하다. Data average MAE는 Transformer 26.88 mmHg에서 ConvTransformer 20.08 mmHg로 25.3% 개선된다. Group average MAE는 40.03에서 25.52로 36.2% 개선된다. Worst group MAE는 74.36에서 43.44로 41.6% 개선된다.

그룹별로 보면 Hypo SBP MAE는 35.61에서 17.31로 51.4% 개선되고, Hypo DBP MAE는 11.56에서 6.44로 44.3% 개선된다. Crisis SBP MAE는 53.47에서 29.73으로 44.4% 개선되고, Crisis DBP MAE는 20.88에서 13.72로 34.3% 개선된다. 이는 ConvTransformer가 특히 minority high-risk group에서 유리함을 보여준다.

Prehyper SBP의 경우에는 Transformer가 9.39이고 ConvTransformer가 10.29로 약간 나빠진다. 그러나 Prehyper는 데이터가 많은 majority group이고, 전체적으로는 ConvTransformer가 훨씬 안정적인 그룹별 성능을 보인다. 논문은 이 결과를 통해 PPG 기반 혈압 추정에서 local context, 즉 peak와 foot point 주변의 국소 파형 정보가 매우 중요하다고 해석한다.

### 4.4 Full data에서 robust optimization 결과

Table 3은 MIMIC-III full data에서 여러 학습 방법을 비교한다. ERM은 data average loss에서 가장 좋은 성능을 보인다. Data average MAE는 20.08, RMSE는 25.98이다. 하지만 group average MAE는 25.52, worst group MAE는 43.44로, 소수 그룹 성능은 충분히 보장되지 않는다.

Undersampling은 모든 지표에서 성능을 크게 악화시킨다. Data average MAE는 29.77로 ERM보다 48.3% 나빠지고, group average MAE도 33.23으로 악화된다. 이는 majority group 데이터를 줄이는 방식이 전체 정보 손실을 크게 유발함을 보여준다.

Oversampling은 worst group MAE를 39.04로 개선하지만, data average MAE는 22.30으로 나빠지고 group average MAE도 25.90으로 ERM보다 약간 나쁘다. 즉, worst group 성능은 개선되지만 다른 그룹 성능이 희생되는 trade-off가 발생한다.

기존 robust optimization인 GDRO와 V-REx는 worst group 성능을 개선한다. GDRO는 worst group MAE 35.50, V-REx는 38.16을 달성한다. 그러나 group average loss에서는 ERM과 큰 차이가 없거나 약간만 개선된다.

제안 방법인 C-REx, D-REx, CD-REx는 기존 robust optimization보다 더 균형 잡힌 결과를 보인다. 특히 CD-REx는 group average MAE 24.77, worst group MAE 34.73을 기록한다. 여기에 Time-CutMix를 결합한 CD-REx + TC는 가장 좋은 group average 및 worst group 성능을 보인다. CD-REx + TC는 group average MAE 24.38, group average RMSE 32.20, worst group MAE 33.35, worst group RMSE 45.87을 달성한다. ERM 대비 worst group MAE는 23.2%, worst group RMSE는 17.4% 개선된다. 다만 data average MAE는 22.10으로 ERM보다 10.1% 나쁘다. 이는 전체 평균 성능과 worst group robustness 사이의 trade-off가 여전히 존재함을 의미한다.

### 4.5 Small data 결과

Table 4는 training data를 50%만 사용한 small data 설정 결과이다. ERM은 data average MAE 22.51, group average MAE 28.55, worst group MAE 45.97을 보인다. 데이터가 줄어들면서 전체적으로 성능이 악화된다.

이 설정에서도 undersampling과 oversampling은 좋지 않다. Undersampling은 data average MAE 30.82로 크게 악화되고, oversampling도 group average와 worst group에서 개선을 거의 보이지 못한다. GDRO와 V-REx는 worst group 성능을 어느 정도 개선하지만, full data만큼 명확하지 않다.

C-REx, D-REx, CD-REx는 worst group 성능에서 GDRO와 유사하거나 더 좋은 결과를 보이지만, 가장 강한 성능은 CD-REx + TC에서 나온다. CD-REx + TC는 data average MAE 22.63으로 ERM과 거의 비슷하고, group average MAE는 26.79로 ERM 대비 6.2% 개선되며, worst group MAE는 39.40으로 ERM 대비 14.3% 개선된다. 이는 Time-CutMix가 데이터가 줄어든 상황에서도 그룹 내 sample 다양성을 늘려 robust learning에 도움이 됨을 보여준다.

### 4.6 그룹별 성능 분석

논문은 Figs. 4와 5에서 group-specific MAE와 RMSE를 분석한다. ERM은 Normal과 Prehyper 같은 majority group에 편향된 성능을 보인다. 반대로 Hypo와 Crisis에서는 underfitting이 발생한다. Oversampling과 GDRO, V-REx는 minority group 성능을 개선하지만, majority group 성능을 희생하는 경향이 있다.

C-REx, D-REx, CD-REx는 Hypo와 Crisis의 SBP loss를 줄이는 데 효과적이다. DBP의 경우 Crisis에서 일부 설정은 GDRO보다 나쁠 때도 있으나, Time-CutMix를 결합하면 이 문제가 완화된다. 전반적으로 제안 방법은 majority group의 성능을 과도하게 희생하지 않으면서 minority group 성능 저하를 줄이는 방향으로 작동한다.

### 4.7 PPGBP benchmark 결과

Table 5는 PPGBP dataset에서 다양한 backbone에 제안 방법을 적용한 결과이다. 비교 backbone은 ResNet, SpectroResNet, MLP-BP, ConvTransformer이다. 여기서 “+ Ours”는 제안한 CD-REx와 Time-CutMix 계열 방법을 적용한 경우로 해석된다.

ResNet에서는 data average SBP MAE가 13.33에서 14.11로 나빠지지만, group average SBP는 24.42에서 23.48로 개선되고, worst group SBP는 44.39에서 42.90으로 개선된다. SpectroResNet에서는 data average SBP가 18.87에서 17.14로, DBP가 11.38에서 9.84로 크게 개선되며, worst group DBP도 18.21에서 14.73으로 크게 좋아진다. MLP-BP에서는 개선 폭이 작지만 group average와 worst group SBP가 약간 개선된다. ConvTransformer에서는 data average SBP가 14.82에서 14.07로, DBP가 9.17에서 8.82로 개선되고, group average와 worst group SBP도 개선된다.

이 결과는 제안 방법이 특정 backbone에만 의존하지 않고, 여러 모델 구조에서 group robustness를 개선할 수 있음을 보여준다. 다만 PPGBP는 전체 segment 수가 619개로 작고, Hypo 11개, Crisis 6개처럼 극단적으로 작은 그룹을 포함하므로 결과의 통계적 안정성에는 주의가 필요하다.

### 4.8 Gradient norm 기반 해석

논문은 ConvTransformer가 왜 Transformer보다 좋은지 이해하기 위해 gradient norm analysis를 수행한다. 입력 PPG sequence의 각 time step에 대한 gradient norm을 계산하여 모델이 어느 시점에 집중하는지 확인한다. Fig. 6에 따르면 ConvTransformer는 PPG의 peak와 foot point 주변에서 더 큰 gradient norm을 보인다.

이는 기존 PPG 기반 혈압 추정 연구에서 peak, foot point, dicrotic notch 등의 morphology feature를 수작업으로 추출하던 것과 연결된다. ConvTransformer는 end-to-end 방식으로 이러한 생리적으로 중요한 지점을 자동으로 포착하는 것으로 해석된다. Transformer-only 모델보다 convolution layer를 앞단에 둔 구조가 local waveform pattern을 잘 포착하고, 이후 Transformer가 global context를 결합하면서 성능이 향상된다는 설명을 뒷받침한다.

## 5. 강점, 한계

### 5.1 강점

이 논문의 가장 큰 강점은 PPG 기반 혈압 추정에서 평균 성능이 아니라 그룹별 robustness를 명시적으로 다룬다는 점이다. 기존 연구들은 전체 MAE 또는 RMSE를 낮추는 데 초점을 맞추는 경우가 많았지만, 이 논문은 Hypo와 Crisis 같은 고위험 소수 그룹에서 모델이 실패하는 문제를 정면으로 다룬다. 의료 AI에서 worst-group performance는 실제 안전성과 직접 연결되므로 매우 중요한 관점이다.

두 번째 강점은 data, model, loss의 세 관점에서 문제를 종합적으로 접근했다는 점이다. ConvTransformer는 local-global feature modeling을 담당하고, C-REx와 D-REx는 group imbalance와 label distribution extremeness를 반영하며, Time-CutMix는 소수 그룹의 data diversity를 늘린다. 단일 기법이 아니라 여러 요소를 결합하여 robustness를 개선했다는 점에서 설계가 체계적이다.

세 번째 강점은 회귀 문제에 맞는 robust optimization을 제안했다는 점이다. Classification에서는 class imbalance나 group robustness 연구가 많지만, 혈압 추정처럼 연속 label을 예측하는 imbalanced regression에서는 label의 순서성과 distribution distance가 중요하다. D-REx는 이러한 회귀 문제의 특성을 반영하여 전체 label distribution과 각 group label distribution의 divergence를 사용한다.

네 번째 강점은 전처리 과정이 비교적 상세하다는 점이다. MIMIC-III에서 PPG와 ABP의 길이, flat line, flat peak, PSDR, skewness, ABP morphology 등을 점검하여 품질이 낮은 segment를 제거한다. 이는 PPG 기반 혈압 추정에서 label quality와 signal quality가 매우 중요하다는 점을 잘 반영한다.

다섯 번째 강점은 모델 해석을 시도했다는 점이다. Gradient norm analysis를 통해 ConvTransformer가 PPG의 peak와 foot point에 집중한다는 것을 보여주며, 성능 향상이 단순히 black-box적인 결과가 아니라 생리학적으로 중요한 waveform landmark와 관련될 수 있음을 제시한다.

여섯 번째 강점은 PPGBP benchmark dataset에서도 추가 실험을 수행했다는 점이다. MIMIC-III 자체 처리 데이터에만 의존하지 않고, 별도의 benchmark dataset에서 여러 backbone에 제안 방법을 적용하여 일반성을 확인하려 했다.

### 5.2 한계

가장 중요한 한계는 MIMIC-III 데이터 split이 patient-wise가 아니라는 점이다. 논문은 “기존 환자와 신규 환자 모두에서 잘 예측하는 모델”을 목표로 하기 때문에 patient-wise split을 사용하지 않았다고 설명한다. 그러나 PPG 기반 혈압 추정에서는 같은 환자의 segment가 train과 test에 동시에 들어갈 경우 subject-specific pattern leakage가 발생할 수 있다. 특히 혈압 추정 모델은 개인별 혈관 특성, 센서 부착 조건, 신호 morphology를 학습할 수 있으므로, random segment split은 실제 신규 환자 일반화 성능을 과대평가할 위험이 있다.

두 번째 한계는 평가 결과의 절대 성능이 여전히 임상 적용에는 부족할 수 있다는 점이다. ConvTransformer와 robust optimization은 worst group 성능을 개선하지만, full data에서도 CD-REx + TC의 worst group MAE는 33.35 mmHg이고 RMSE는 45.87 mmHg이다. 이는 high-risk group에서 여전히 매우 큰 오차가 남아 있음을 의미한다. 논문도 real-world deployment 전에는 추가 개선이 필요하다고 인정한다.

세 번째 한계는 BP group 정의가 SBP와 DBP 범위의 결합 조건에 기반하지만, 실제로 한 sample이 어떤 방식으로 그룹에 할당되는지 세부 규칙이 완전히 명확하지 않다. 예를 들어 SBP는 Prehyper 범위지만 DBP는 Normal 범위일 경우 어떤 그룹으로 분류되는지 명시적으로 설명되지 않는다. 혈압 guideline에서는 SBP와 DBP 중 더 높은 risk category를 사용하는 경우가 많으므로, group labeling rule이 결과에 영향을 줄 수 있다.

네 번째 한계는 Time-CutMix의 생리적 타당성에 대한 검증이 제한적이라는 점이다. 같은 BP group 안에서 두 PPG segment를 시간축으로 잘라 붙이면 label은 비율에 따라 섞을 수 있지만, 실제 PPG waveform의 phase continuity나 cardiac cycle boundary가 깨질 가능성이 있다. 논문은 이를 “relatively natural sequence”라고 주장하지만, cardiac cycle alignment를 고려하지 않은 단순 concatenation이 항상 생리적으로 자연스러운지는 추가 검증이 필요하다.

다섯 번째 한계는 D-REx의 distribution assumption이다. D-REx는 Tukey transformation 후 label distribution을 Gaussian으로 가정하고 symmetric KL divergence를 계산한다. 이 가정은 실용적인 approximation이지만, 실제 SBP/DBP distribution이 다봉성(multimodal)이거나 그룹별로 비정규적일 수 있다. 특히 group sample 수가 작은 Hypo와 Crisis에서는 평균과 공분산 추정이 불안정할 수 있다.

여섯 번째 한계는 PPGBP 실험에서 minority group sample 수가 매우 작다는 점이다. PPGBP에서 Hypo는 11개, Crisis는 6개뿐이다. 이 경우 worst group 성능은 fold 구성이나 몇 개 sample의 오차에 매우 민감할 수 있다. 따라서 PPGBP에서의 worst group 개선은 방향성은 의미 있지만, 통계적으로 매우 안정적이라고 보기는 어렵다.

일곱 번째 한계는 추가적인 생리 정보나 demographic information을 사용하지 않는다는 점이다. 논문은 PPG만으로 robust estimation을 수행한다는 점을 강점으로 제시하지만, 실제 혈압은 나이, 성별, 체중, 혈관 탄성, 약물, 질환 상태 등에 영향을 받는다. PPG only 접근은 wearable 적용에는 단순하다는 장점이 있지만, 극단 혈압군에서의 정확도를 높이는 데 한계가 있을 수 있다.

### 5.3 비판적 해석

이 논문은 PPG 기반 혈압 추정 연구에서 매우 중요한 질문을 제기한다. 전체 평균 MAE가 낮더라도 고위험 혈압군에서 모델이 실패한다면 실제 의료적 가치는 제한된다. 이 논문은 worst group robustness를 혈압 추정에 도입하고, regression task에 맞는 C-REx, D-REx, CD-REx를 제안했다는 점에서 기여가 분명하다.

다만 논문의 결과를 해석할 때는 “robustness 개선”과 “임상적으로 충분한 정확도”를 구분해야 한다. 제안 방법은 ERM 대비 worst group loss를 상당히 낮추지만, 절대적인 worst group 오차는 여전히 크다. 따라서 이 연구는 완성된 혈압 측정 알고리즘이라기보다, 혈압군 불균형을 고려하는 robust training framework의 출발점으로 보는 것이 적절하다.

또한 patient-wise split을 사용하지 않은 점은 이 분야에서 매우 중요한 평가상 약점이다. 논문은 group imbalance와 worst-group 문제를 강조하지만, 실제 deployment 관점에서는 subject shift 역시 매우 큰 문제이다. 향후 연구에서는 BP group robustness와 subject-level generalization을 동시에 평가해야 한다. 예를 들어 patient-wise split 상태에서 Hypo와 Crisis의 sample이 더 적어질 경우, 제안 방법이 여전히 효과적인지 확인해야 한다.

그럼에도 이 논문은 평균 성능 중심의 PPG-BP 연구 관행을 비판하고, high-risk group performance를 명시적으로 최적화했다는 점에서 중요한 의미가 있다. 특히 ConvTransformer의 gradient norm 분석은 모델이 PPG의 생리학적 landmark에 주목한다는 해석 가능성을 제공하며, robust optimization과 생리 신호 모델링을 연결하려는 시도가 돋보인다.

## 6. 결론

이 논문은 PPG 기반 cuff-less 혈압 추정에서 혈압군별 성능 격차를 줄이기 위한 robust optimization framework를 제안한다. 연구의 핵심은 Hypo와 Crisis처럼 데이터가 적고 임상적으로 중요한 high-risk BP group에서 모델 성능이 악화되는 문제를 “worst group” 문제로 정의하고, 이를 data, model, loss 관점에서 해결하려는 것이다.

모델 관점에서는 ConvTransformer를 제안하여 convolution layer로 PPG의 local morphology를 포착하고 Transformer Encoder로 global context를 학습한다. 실험 결과 ConvTransformer는 Transformer-only 모델보다 data average, group average, worst group 성능에서 모두 크게 우수하며, 특히 Hypo와 Crisis 그룹에서 큰 개선을 보인다.

Loss 관점에서는 C-REx, D-REx, CD-REx를 제안한다. C-REx는 그룹별 데이터 수를 고려하여 소수 그룹을 더 강하게 반영하고, D-REx는 각 그룹 label distribution이 전체 distribution에서 얼마나 벗어나는지를 반영한다. CD-REx는 두 요소를 결합한다. 실험 결과 CD-REx는 기존 GDRO, V-REx보다 group average와 worst group 성능을 더 효과적으로 개선한다.

Data 관점에서는 Time-CutMix를 제안한다. 이는 같은 BP group 안에서 두 PPG sequence를 시간축으로 잘라 붙이고, label을 비율에 따라 섞는 augmentation이다. CD-REx와 Time-CutMix를 결합한 CD-REx + TC는 full data와 small data 설정에서 가장 우수한 group average 및 worst group 성능을 보인다. 특히 small data에서도 ERM과 유사한 data average 성능을 유지하면서 worst group 성능을 개선한다.

이 연구의 주요 기여는 PPG 기반 혈압 추정에서 처음으로 BP group disparity와 worst-group robustness를 본격적으로 다루었다는 점이다. 또한 회귀 문제의 특성을 반영한 count-aware 및 divergence-aware regularization을 제안하고, ConvTransformer와 Time-CutMix를 결합하여 실험적·이론적 근거를 제시했다.

실제 적용 가능성 측면에서 이 연구는 연속 혈압 모니터링 시스템이 평균적으로 잘 맞는 것만으로는 충분하지 않으며, 고위험 혈압군에서 안정적으로 작동해야 한다는 중요한 방향을 제시한다. 다만 현재 결과만으로 즉시 임상 적용이 가능하다고 보기는 어렵다. Worst group의 절대 오차가 여전히 크고, patient-wise generalization이 충분히 검증되지 않았으며, Time-CutMix의 생리적 자연성도 추가 검증이 필요하다.

향후 연구에서는 subject-wise split, external dataset validation, 더 다양한 환자군, demographic 및 physiological information 결합, signal quality-aware modeling, real wearable 환경 검증이 필요하다. 그럼에도 이 논문은 PPG 기반 혈압 추정의 평가와 학습 목표를 평균 성능에서 group robustness로 확장했다는 점에서 중요한 연구로 평가할 수 있다.
