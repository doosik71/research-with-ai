# A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder

* **저자**: Daehyung Park, Yuuna Hoshi, Charles C. Kemp
* **발표연도**: 2017
* **arXiv**: [https://arxiv.org/abs/1609.03894](https://arxiv.org/abs/1609.03894)

## 1. 논문 개요

이 논문은 robot-assisted feeding 과정에서 발생하는 이상 상황(anomaly)을 실시간으로 탐지하기 위한 멀티모달 이상 탐지기를 제안한다. 핵심 문제는 보조 로봇이 사람에게 음식을 먹여 주는 과정에서 다양한 종류의 실패나 위험 상황이 발생할 수 있다는 점이다. 예를 들어 사용자가 갑자기 얼굴을 움직이거나, 숟가락이 얼굴이나 환경과 충돌하거나, 시스템이 멈추거나, 소음이나 가림(occlusion) 같은 외부 요인이 개입할 수 있다. 이런 상황을 빠르게 감지하지 못하면 로봇 보조의 안전성과 신뢰성이 크게 떨어진다.

논문은 특히 다음과 같은 어려움을 다룬다. 첫째, 이상 상황은 하나의 센서만으로는 포착하기 어려워서 시각, 힘/토크, 관절, 음향 등 여러 modality를 함께 활용해야 한다. 둘째, 이들 센서는 차원도 높고 성질도 서로 다르기 때문에 단순 결합이 어렵다. 셋째, feeding은 시간에 따라 진행되는 sequential task이므로 한 시점의 관측만 보는 것이 아니라 temporal dependency를 모델링해야 한다. 넷째, anomaly detector는 정상 데이터만으로 학습되는 one-class setting이어야 하며, 정상 variation은 허용하면서도 실제 이상은 민감하게 잡아야 한다.

이 문제의 중요성은 assistive robotics의 안전성에 직접 연결된다는 점에 있다. 단순히 오분류를 줄이는 수준이 아니라, 실제 사람과 접촉하는 로봇이 잠재적 hazard를 미리 감지하고 동작을 중단할 수 있어야 한다. 따라서 이 논문은 고차원 멀티모달 시계열을 hand-engineered feature 없이도 직접 처리하면서, online anomaly detection이 가능한 구조를 제시했다는 점에서 의의가 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 LSTM과 VAE를 결합한 LSTM-VAE를 사용하여 멀티모달 시계열의 정상 실행 분포를 학습하고, 현재 관측이 그 분포에서 얼마나 벗어나는지를 reconstruction-based anomaly score로 계산하는 것이다. 일반적인 autoencoder는 입력을 압축했다가 복원하면서 reconstruction error를 사용할 수 있지만, 이 논문은 여기서 한 단계 더 나아가 VAE의 확률적 재구성을 사용한다. 즉, 단순히 복원값 하나를 예측하는 것이 아니라, 각 시점의 입력이 어떤 확률분포로 재구성되는지를 모델링하고, 현재 관측의 negative log-likelihood를 anomaly score로 사용한다.

이 설계의 장점은 세 가지로 요약할 수 있다. 첫째, 고차원 heterogeneous signal을 feature engineering 없이 직접 다룰 수 있다. 기존 연구에서는 PCA나 handcrafted feature를 통해 차원을 줄인 뒤 HMM이나 GP 같은 모델에 넣는 방식이 많았는데, 이 과정은 정보 손실과 엔지니어링 비용을 수반한다. 둘째, LSTM을 통해 temporal dependency를 반영한다. 즉, 현재 관측만 보는 것이 아니라 task progression에 따라 정상 패턴이 어떻게 변하는지를 내부 상태에 담는다. 셋째, threshold를 고정값으로 두지 않고 latent state에 따라 달라지는 state-based threshold를 도입해 false alarm은 줄이고 sensitivity는 높였다.

기존 접근 방식과의 차별점도 명확하다. HMM-GP처럼 likelihood-based model은 handcrafted feature에 의존했고, OSVM이나 PCA 기반 방법은 고차원 데이터에서 표현력이 부족할 수 있다. EncDec-AD 같은 LSTM autoencoder 계열은 reconstruction error를 쓰지만, 이 논문은 VAE 기반 확률적 reconstruction과 progress-based prior를 도입하여 latent space 자체에 task progression을 반영한다. 또한 anomaly score의 threshold를 latent state 기반의 SVR로 예측함으로써, task의 특정 구간에서 정상적으로 나타나는 score 상승을 유연하게 처리한다.

## 3. 상세 방법 설명

전체 구조는 크게 두 부분으로 나뉜다. 첫 번째는 정상 실행 데이터의 분포를 학습하는 LSTM-VAE이고, 두 번째는 latent state에 따라 anomaly score threshold를 추정하는 state-based threshold estimator이다. 입력은 시점 $t$에서의 멀티모달 관측 벡터 $x_t \in \mathbb{R}^D$이고, 잠재변수는 $z_t \in \mathbb{R}^K$이다. 이 논문에서는 입력 차원 $D$가 17 또는 4이며, latent dimension $K$는 3으로 설정되었다.

### 3.1 LSTM-VAE의 기본 구조

encoder는 시계열 입력 $x_t$를 받아 LSTM을 통과시킨 뒤, 두 개의 linear module을 사용해 잠재분포의 평균 $\mu_{z_t}$와 공분산 $\Sigma_{z_t}$를 추정한다. 즉, 각 시점마다 posterior approximation을 Gaussian으로 둔다. 그 후 이 posterior에서 샘플링한 $z_t$를 decoder의 LSTM에 넣어 reconstruction distribution의 평균 $\mu_{x_t}$와 공분산 $\Sigma_{x_t}$를 출력한다. 따라서 decoder는 입력을 한 점으로 복원하는 것이 아니라, 현재 시점의 정상 입력이 따라야 할 확률분포를 예측한다.

논문은 VAE의 기본 objective를 먼저 다음과 같이 설명한다.

$$
\mathcal{L}_{vae} = - D_{KL}(q_\phi(z|x),|,p_\theta(z)) - \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)].
$$

여기서 첫 번째 항은 posterior approximation과 prior 사이의 차이를 줄이는 regularization 항이고, 두 번째 항은 reconstruction likelihood를 높이는 항이다. 쉽게 말하면, latent variable이 너무 제멋대로 흩어지지 않게 하면서도 입력을 잘 설명하도록 학습하는 구조다.

### 3.2 Denoising criterion

저자들은 LSTM-VAE가 단순 identity mapping을 학습하는 것을 막고 representation capability를 높이기 위해 denoising autoencoding criterion을 도입했다. 입력에 Gaussian noise를 더해 corrupted input $\tilde{x}=x+\epsilon$를 만들고, $\epsilon \sim \mathcal{N}(0,\sigma_{noise})$로 둔다. 그런 다음 원래 입력 $x_t$를 복원하도록 학습한다.

이때 사용되는 objective는 denoising variational lower bound이다.

$$
\mathcal{L}_{dvae} = - D_{KL}(\tilde{q}_\phi(z_t|x_t),|,p_\theta(z_t)) - \mathbb{E}_{\tilde{q}_\phi(z_t|x_t)}[\log p_\theta(x_t|z_t)].
$$

논문은 엄밀히는 $\tilde{q}_\phi(z_t|x_t)$가 corruption distribution을 고려한 mixture of Gaussians가 될 수 있다고 설명하지만, 계산 편의를 위해 이를 단일 Gaussian으로 근사하여 $\tilde{q}_\phi(z|x) \approx q_\phi(z|\tilde{x})$를 사용한다. 즉, noisy input을 encoder에 넣고 clean target을 복원하게 한다고 이해하면 된다.

### 3.3 Progress-based prior

이 논문의 중요한 차별점 중 하나는 static prior 대신 progress-based prior를 사용한 점이다. 일반적인 VAE는 latent prior를 $p(z)=\mathcal{N}(0,1)$ 같은 고정된 정규분포로 둔다. 하지만 feeding task는 시간에 따라 상태가 변하는 sequential process이므로, 모든 시점의 정상 데이터가 동일한 latent prior를 가져야 한다고 보는 것은 비효율적일 수 있다.

그래서 이 논문은 prior를 $p(z_t)=\mathcal{N}(\mu_p,\Sigma_p)$로 두고, task progress에 따라 prior의 중심 $\mu_p$를 선형적으로 바꾼다. 시작 시점에서는 $p_1$, 마지막 시점에서는 $p_T$에 해당하는 중심을 가지도록 점진적으로 이동시킨다. 공분산은 단순화를 위해 isotropic Gaussian으로 두어 $\Sigma_p = I$를 사용한다.

이 설정 아래 KL term은 다음과 같이 계산된다.

$$
\begin{aligned}
D_{KL}(\tilde{q}_\phi(z_t|x_t),|,p_\theta(z_t)) \approx D_{KL}(\mathcal{N}(\mu_{z_t}, \Sigma_{z_t}) ,|, \mathcal{N}(\mu_p, 1)) \\ =
\frac{1}{2} \left( \operatorname{tr}(\Sigma_{z_t}) + (\mu_p-\mu_{z_t})^T(\mu_p-\mu_{z_t}) - D \log |\Sigma_{z_t}| \right).
\end{aligned}
$$

이 식의 의미는 간단하다. latent posterior의 평균과 분산이 현재 progress에 맞는 prior 중심 근처에 오도록 유도한다는 것이다. 따라서 sequence의 초반과 후반이 latent space에서 자연스럽게 다른 위치를 차지하게 되고, task progression 정보가 내재화된다.

### 3.4 Reconstruction term

입력이 연속값인 고차원 멀티모달 신호이므로, reconstruction likelihood는 diagonal covariance를 갖는 multivariate Gaussian으로 모델링한다. reconstruction term은 다음과 같이 주어진다.

$$
\mathbb{E}_{\tilde{q}_\phi(z_t|x_t)}[\log p_\theta(x_t|z_t)] = -\frac{1}{2} \left( \log |\Sigma_{x_t}| + (x_t-\mu_{x_t})^T \Sigma_{x_t}^{-1}(x_t-\mu_{x_t}) + D \log(2\pi) \right).
$$

이 식은 현재 관측 $x_t$가 decoder가 예측한 정상 분포 $\mathcal{N}(\mu_{x_t},\Sigma_{x_t})$ 아래에서 얼마나 가능성 높은 값인지를 나타낸다. 관측이 평균에서 멀리 떨어지거나 분산 구조와 맞지 않으면 likelihood가 낮아지고, 곧 anomaly score는 높아진다.

### 3.5 Anomaly score

논문은 anomaly detection rule을 다음처럼 정의한다.

$$
\text{anomaly if } f_s(x_t,\phi,\theta) > \eta
$$

여기서 anomaly score는 reconstructed distribution에 대한 negative log-likelihood이다.

$$
f_s(x_t,\phi,\theta) = -\log p(x_t;\mu_{x_t},\Sigma_{x_t}).
$$

즉, 입력이 정상 데이터에서 학습한 reconstruction distribution으로 잘 설명되지 않으면 anomaly score가 커진다. 일반적인 AE의 reconstruction error보다 더 풍부한 정보가 들어가는 이유는 평균 오차뿐 아니라 uncertainty 구조까지 함께 반영하기 때문이다.

### 3.6 State-based thresholding

고정 threshold는 실제 시계열 작업에서 자주 부적절하다. 어떤 task phase에서는 정상 실행도 재구성이 어렵기 때문에 score가 다소 높을 수 있고, 다른 phase에서는 작은 deviation만 있어도 이상일 수 있다. 이를 해결하기 위해 논문은 latent state $z$를 입력으로 받아 expected anomaly score를 예측하는 함수 $\hat{f}_s : z \rightarrow s$를 학습한다.

구체적으로 non-anomalous validation dataset에서 각 시점의 latent state $Z$와 anomaly score $S$를 추출한 뒤, support vector regression(SVR) with RBF kernel로 상태에서 기대 score를 회귀한다. 최종 threshold는 다음과 같이 정의된다.

$$
\eta = \hat{f}_s(z) + c
$$

여기서 $c$는 sensitivity를 조절하는 상수다. 이 구조는 정상 데이터에서도 원래 score가 높은 구간에는 threshold를 느슨하게 두고, 원래 score가 낮은 구간에는 threshold를 더 엄격하게 두는 효과를 낸다.

### 3.7 학습 및 테스트 절차

학습 단계에서는 먼저 training/validation sequence를 동일 길이 $T$로 resampling하고, 각 modality를 $[0,1]$ 범위로 정규화한다. 이후 LSTM-VAE를 $\mathcal{L}_{dvae}$를 최대화하도록 학습한다. 논문은 validation 성능이 4 epochs 동안 증가하지 않으면 학습을 멈춘다고 설명한다. 그리고 validation set을 encoder-decoder에 통과시켜 각 시점의 latent vector와 anomaly score를 수집하고, 이를 사용해 SVR threshold estimator를 학습한다. sequence 시작 시에는 LSTM state를 reset한다.

테스트 단계에서는 매 시점마다 현재 multimodal input을 받아 같은 방식으로 scaling하고, encoder로 latent state $z_t$를 구한 뒤, decoder가 예측한 $(\mu_{x_t}, \Sigma_{x_t})$를 얻는다. 이후 anomaly score $f_s(x_t)$를 계산하고, 이것이 $\hat{f}_s(z_t)+c$보다 크면 anomaly로 판정한다. 논문은 이 과정을 online으로 수행할 수 있다고 주장한다.

### 3.8 구현 세부사항

논문에 명시된 구현 세부사항은 다음과 같다. encoder와 decoder 모두 tanh activation을 사용하는 LSTM을 썼고, Keras로 stateful LSTM 모델을 구현했다. latent variable dimension은 3, optimizer는 Adam, learning rate는 0.001이다. sliding window는 사용하지 않았지만, 적용 가능성은 열어 두었다고 언급한다.

## 4. 실험 및 결과

### 4.1 실험 환경과 데이터

실험 플랫폼은 Willow Garage의 PR2 mobile manipulator이다. 두 개의 7-DOF arms, powered grippers, omni-directional base를 갖고 있으며, safety를 위해 low-gain PID controller와 50 Hz의 mid-level model predictive controller를 사용했다. 논문은 haptic feedback 없이 동작했다고 명시한다.

센서는 총 5종류를 사용했다. 오른쪽 손목의 RGB-D camera with microphone(Intel SR300), utensil handle의 force/torque sensor(ATI Nano25), joint encoders, current sensors 등이 포함된다. 이들로부터 mouth position, sound, force on utensil, spoon position, joint torque를 측정했다.

전체 데이터는 24명의 able-bodied participant로부터 수집한 1,555회의 feeding execution이다. 이 중 1,203회는 새롭게 수집한 non-anomalous execution이며, training/testing dataset은 기존 연구의 352회 실행으로 구성된다. 이 352회는 160 anomalous, 192 non-anomalous execution을 포함하고, 8명의 participant가 yogurt와 silicone spoon을 사용했다.

추가적인 pre-training dataset은 16명의 새 participant로부터 수집한 1,203회의 non-anomalous execution으로 구성된다. 이들은 food와 utensil의 종류가 다양했고, 일부는 experimenter self-study data도 포함된다. 저자들은 이 데이터로 pre-training을 수행한 뒤, target dataset으로 fine-tuning했다고 설명한다.

### 4.2 태스크와 anomaly 유형

feeding system은 세 가지 subtask를 지원한다. scooping/stabbing, clean spoon, feeding이다. 일반적인 실행은 scooping 또는 stabbing 후 feeding으로 이어진다. participant는 상체를 움직이지 않고 입술만으로 utensil의 음식을 먹도록 지시받았다.

논문은 fault tree analysis를 통해 12가지 representative anomaly를 정의했다. 예를 들어 touch by user, aggressive eating, utensil collision by user, sound from user, face occlusion, utensil miss by user, unreachable location, environmental collision, environmental noise, utensil miss by system fault, utensil collision by system fault, system freeze가 포함된다. 이 구성은 anomaly 유형이 사용자, 환경, 시스템 모두에서 발생할 수 있음을 보여 준다.

### 4.3 입력 신호와 전처리

각 실행마다 17차원 sensory signal을 수집했다. 구성은 sound energy 1차원, force 3차원, joint torque 7차원, spoon position 3차원, mouth position 3차원이다. 각 signal은 초기값을 0으로 맞추고, 실제 anomaly checking frequency인 20 Hz에 맞게 resampling했다. 정상 데이터 기준으로 모든 signal을 $[0,1]$ 범위로 scaling했고, anomalous dataset에도 동일한 scale을 적용했다.

비교를 위해 저자들은 기존 연구에서 사용하던 4차원 hand-engineered feature도 추출했다. 그것은 sound energy, first joint torque, accumulated force, spoon-mouth distance이다. auditory anomaly를 놓치지 않기 위해 raw audio waveform이 아니라 sound energy를 사용했다고 명시한다.

### 4.4 비교 대상과 평가 방식

비교 baseline은 총 5개다. RANDOM, OSVM(one-class SVM), HMM-GP, AE, EncDec-AD가 사용되었다. OSVM과 AE, EncDec-AD는 sliding window를 활용했고, HMM-GP는 저자들의 이전 likelihood-based method이다. LSTM-VAE는 이들과 비교되는 제안 방법이다.

평가 방식은 leave-one-person-out cross-validation이다. 8명 중 7명의 데이터를 훈련에 사용하고, 나머지 1명으로 테스트하는 절차를 반복했다. 성능 지표는 ROC curve의 area under the curve(AUC)이다. 이 지표는 threshold를 변화시켰을 때 true positive rate와 false positive rate의 trade-off를 종합적으로 보여 준다.

### 4.5 정성적 결과

논문은 Figure 6을 통해 정상 실행과 이상 실행에서의 reconstruction behavior를 시각적으로 보여 준다. 정상 실행에서는 observed feature와 reconstructed mean이 시간에 따라 유사한 패턴을 보인다. 반면 이상 실행에서는, 예를 들어 face-spoon collision이 발생했을 때 accumulated force의 실제 관측과 재구성 분포 사이의 차이가 커진다. 이 deviation 이후 anomaly score가 점차 상승한다. 이것은 모델이 정상 데이터에서 학습한 패턴으로는 해당 충돌 상황을 잘 설명하지 못한다는 의미다.

또한 Figure 7에서는 한 participant의 anomalous 24회와 non-anomalous 20회에 대해 시간에 따른 anomaly score 분포를 비교한다. 정상 실행의 score는 평균과 분산이 더 작고 일정한 패턴을 가지는 반면, 이상 실행의 score는 더 높고 분산도 크다. 따라서 anomaly score 자체가 정상/비정상을 구분하는 데 효과적이라는 점을 보여 준다.

### 4.6 정량적 결과

가장 중요한 결과는 Table I에 제시된 AUC 비교다.

4개의 hand-engineered feature를 사용할 때 성능은 다음과 같다.

* RANDOM: 0.5121
* OSVM: 0.7427
* HMM-GP: 0.8121
* AE: 0.8123
* EncDec-AD: 0.7995
* **LSTM-VAE: 0.8564**

이 결과에서 LSTM-VAE는 가장 좋은 baseline인 HMM-GP보다 0.0443 높은 AUC를 보인다. 저자들도 이를 0.044 improvement라고 요약한다. 즉, handcrafted feature만 써도 제안한 probabilistic sequence autoencoding 구조가 기존 방법보다 더 우수했다.

17개의 raw sensory signal을 사용할 때 성능은 다음과 같다.

* RANDOM: 0.5052
* OSVM: 0.7376
* HMM-GP: N/A
* AE: 0.8012
* EncDec-AD: 0.8075
* **LSTM-VAE: 0.8710**

이 설정에서 LSTM-VAE는 EncDec-AD보다 0.0635 높은 AUC를 보이며, 논문은 이를 0.064 improvement라고 요약한다. 특히 raw multimodal signal을 그대로 사용했는데도 handcrafted feature 기반 결과보다 더 좋았다는 점이 중요하다. 이는 제안 모델이 고차원 heterogeneous signal을 직접 모델링할 수 있음을 뒷받침한다. 한편 HMM-GP는 high-dimensional input에서 underflow error로 학습에 실패했다고 명시되어 있다. 이 역시 고차원 멀티모달 확률모델링에서 기존 방법의 한계를 보여 준다.

### 4.7 State-based threshold의 효과

Figure 8은 fixed threshold와 proposed state-based threshold의 ROC curve를 비교한다. 제안된 state-based threshold를 사용했을 때 동일한 false positive rate에서 더 높은 true positive rate를 얻었다. 이것은 task state에 따라 정상 score의 기대값이 달라지는 점을 반영했기 때문으로 해석할 수 있다. 즉, threshold를 시간 또는 latent state에 맞추어 적응적으로 조절함으로써 false alarm을 줄이면서 더 민감한 detection이 가능해졌다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 멀티모달 고차원 시계열 anomaly detection을 하나의 통합된 probabilistic sequence model로 해결했다는 점이다. 기존에는 handcrafted feature selection이 필수적이거나, 차원 축소 후 별도 classifier를 적용해야 하는 경우가 많았다. 그러나 이 논문은 17차원 raw sensory signal을 그대로 입력으로 사용해도 기존 방법보다 높은 AUC를 달성했다. 이는 실제 로봇 시스템에 새로운 센서가 추가되거나 신호 구성이 바뀌더라도 feature engineering 비용을 줄일 수 있다는 뜻이다.

또 다른 강점은 reconstruction error가 아니라 reconstruction distribution의 negative log-likelihood를 anomaly score로 썼다는 점이다. 이 방식은 단순 평균 오차보다 풍부한 정보를 활용한다. 예를 들어 어떤 차원은 본래 variation이 큰 반면 다른 차원은 매우 안정적일 수 있는데, 분산까지 모델링하면 이런 차이를 반영할 수 있다. 특히 safety-critical robotics에서는 “얼마나 벗어났는가”뿐 아니라 “원래 얼마나 불확실한 구간인가”도 중요하므로 적절한 선택이다.

세 번째 강점은 progress-based prior와 state-based thresholding이다. 전자는 latent space에 task progression을 자연스럽게 반영하고, 후자는 정상 시계열의 stage-dependent variability를 threshold 수준에서 보정한다. 이 둘은 모두 temporal structure를 더 정교하게 이용하려는 설계로 볼 수 있다. 특히 state-based threshold는 실제 deployment에서 false alarm 문제를 줄이는 데 실용적 가치가 높다.

실험 설계도 비교적 설득력이 있다. 1,555 executions라는 비교적 큰 규모의 feeding dataset을 사용했고, anomaly 유형도 12가지로 다양하다. leave-one-person-out cross-validation을 사용하여 사람 간 generalization을 보려 했다는 점도 긍정적이다.

반면 한계도 분명하다. 첫째, 데이터셋이 robot-assisted feeding이라는 매우 특정한 task에 한정되어 있다. 논문은 이 방법이 일반적인 manipulation anomaly detection에 유용할 수 있음을 암시하지만, 실제로 다른 task에서 동일한 성능이 나오는지는 검증하지 않았다. 둘째, participant는 able-bodied adults이며 실제 target population인 motor impairment 사용자와는 차이가 있다. 따라서 실제 assistive setting에서의 신호 variation이 어떻게 달라질지는 이 논문만으로는 판단하기 어렵다.

셋째, progress-based prior는 task progress가 비교적 monotonic하게 흐르는 상황에 잘 맞는다. feeding처럼 명확한 시작과 끝이 있는 task에서는 유용하지만, 분기나 반복이 많은 작업, 혹은 task phase가 명확하지 않은 환경에서는 단순한 linearly moving prior가 충분하지 않을 수 있다. 논문도 Solch 등의 RNN prior나 Karl 등의 transition prior와 구분되는 단순화된 prior를 택했다고 설명하지만, 그 선택이 모든 sequence 유형에 적합하다고 보기는 어렵다.

넷째, threshold estimator로 SVR을 별도로 학습하는 2-stage pipeline은 구조가 다소 분리되어 있다. 다시 말해 representation learning과 threshold calibration이 end-to-end로 최적화되지 않는다. latent state 품질이 threshold estimator 성능을 좌우하는데, 두 부분이 공동 학습되지 않는다는 점은 개선 여지가 있다.

다섯째, detection latency에 대한 정량 분석은 제한적이다. 논문은 online detection이 가능하다고 말하고 Figure 6에서 detection timing 예시를 보여 주지만, anomaly onset 이후 평균 몇 시점 또는 몇 초 내에 탐지되는지에 대한 종합 통계는 제시하지 않았다. 실제 안전 시스템에서는 AUC뿐 아니라 detection delay도 매우 중요하다.

비판적으로 보면, 이 논문은 당시 시점에서는 매우 타당한 접근이지만, posterior와 reconstruction covariance 모두 diagonal Gaussian으로 제한되어 있어 modality 간 상관관계를 완전히 표현하지는 못한다. 멀티모달 신호의 핵심은 modality 간 관계일 수 있는데, 이 구조는 그 관계를 LSTM hidden state에는 간접적으로 담더라도 likelihood 자체는 diagonal covariance로 단순화한다. 이는 계산 안정성을 위해 현실적인 선택이지만, 표현력 측면의 제약이기도 하다.

## 6. 결론

이 논문은 robot-assisted feeding의 안전성을 높이기 위해 LSTM-VAE 기반의 멀티모달 anomaly detector를 제안했다. 제안 방법은 멀티모달 시계열을 latent space로 인코딩하고, 각 시점의 정상 입력 분포를 reconstruction distribution으로 예측한 뒤, 그 분포에 대한 negative log-likelihood를 anomaly score로 사용한다. 여기에 denoising training, progress-based prior, state-based thresholding을 결합해 sequence modeling과 threshold calibration을 동시에 개선했다.

실험적으로는 12가지 anomaly가 포함된 1,555회의 feeding execution 데이터에서 기존 baseline들보다 더 높은 AUC를 기록했다. 특히 17개의 raw sensory signal을 직접 사용했을 때 AUC 0.8710을 달성했고, 이는 handcrafted feature 기반 방법보다도 우수했다. 이 결과는 제안 모델이 feature engineering 의존도를 줄이면서도 고차원 heterogeneous sensory fusion을 효과적으로 수행할 수 있음을 보여 준다.

실제 적용 측면에서 이 연구는 assistive robot이 사람과 상호작용하는 상황에서 안전 모니터링 모듈로 활용될 가능성이 크다. 또한 broader robotics 관점에서도, 다중 센서 기반 execution monitoring과 online anomaly detection 문제에 대한 유의미한 설계 원칙을 제공한다. 향후 연구에서는 더 다양한 task와 사용자군, 더 복잡한 temporal prior, end-to-end threshold learning, detection delay 최적화 등으로 확장될 수 있을 것이다. 전체적으로 이 논문은 assistive robotics와 multimodal anomaly detection의 교차점에서 실용성과 방법론적 기여를 모두 갖춘 탄탄한 연구라고 평가할 수 있다.
