# A Generative Appearance Model for End-to-end Video Object Segmentation

- **저자**: Joakim Johnander, Martin Danelljan, Emil Brissman, Fahad Shahbaz Khan, Michael Felsberg
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1811.11611

## 1. 논문 개요

이 논문은 semi-supervised video object segmentation(VOS) 문제를 다룬다. 구체적으로는 첫 프레임에서만 ground-truth mask가 주어지고, 이후 프레임들에서 같은 객체를 계속 분할해야 하는 설정이다. 이 문제의 핵심 어려움은 시간이 지나면서 target object의 appearance가 크게 바뀔 수 있고, 빠른 motion, occlusion, 그리고 target과 비슷하게 생긴 distractor object가 함께 등장할 수 있다는 점이다.

기존의 강력한 방법들은 첫 프레임에서 convolutional neural network를 온라인 fine-tuning하여 target appearance를 학습하는 전략을 많이 사용했다. 하지만 이 방식은 계산 비용이 매우 크고, 실시간 처리에 부적합하며, 무엇보다 online fine-tuning이 offline training 과정 안에 포함되지 않기 때문에 엄밀한 의미의 end-to-end training이 아니다. 반대로 fine-tuning을 피하는 feedforward 방식들은 속도는 빠르지만, target appearance를 명시적으로 모델링하지 못하거나, 비미분적인 matching 절차에 의존하여 end-to-end 학습이 어렵다는 한계가 있다.

이 논문의 목표는 이 두 부류의 단점을 동시에 피하는 것이다. 즉, 첫 프레임에서 한 번의 forward pass만으로 target과 background의 appearance를 학습하고, 이후 각 프레임에서 이를 이용해 분할을 수행하되, 전체 파이프라인이 완전히 differentiable하여 진정한 end-to-end training이 가능하도록 하는 것이다. 저자들은 이를 위해 feature space에서 foreground와 background의 분포를 generative probabilistic model, 정확히는 class-conditional Gaussian mixture로 모델링하는 appearance module을 제안한다.

문제의 중요성은 분명하다. 비디오 객체 분할은 video editing, robotics, autonomous systems, video understanding 같은 다양한 응용의 기본 문제이며, 실제 사용을 위해서는 정확도뿐 아니라 속도와 일반화 성능이 함께 필요하다. 논문은 이 점에서, online fine-tuning 없이도 strong appearance cue를 제공하는 경량 generative module을 제시했다는 데 의의가 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 target appearance를 “저장된 exemplar feature의 단순 결합”이나 “온라인 학습된 분류기”로 다루지 않고, deep feature space 위의 generative distribution으로 직접 모델링하는 것이다. 저자들은 각 위치의 feature vector가 foreground 또는 background에 조건부인 Gaussian mixture에서 나왔다고 가정하고, 현재 프레임의 feature가 어느 mixture component에 속할 posterior를 계산한다. 이 posterior 또는 그에 대응하는 score는 segmentation network에 매우 강한 판별 신호를 제공한다.

핵심 직관은 다음과 같다. 좋은 video object segmentation은 단순히 이전 프레임 mask를 조금씩 보정하는 것만으로는 부족하다. target이 일시적으로 사라지거나, 비슷한 객체가 등장하거나, 모양이 크게 변하면 공간적 prior만으로는 실패하기 쉽다. 따라서 현재 프레임의 feature가 “foreground답게 보이는지, background답게 보이는지”를 appearance 차원에서 지속적으로 평가해야 한다. 저자들은 이 역할을 generative appearance model이 맡도록 했다.

기존 접근과의 차별점도 명확하다. online fine-tuning 계열은 강력하지만 느리고 end-to-end가 아니며, KNN matching 계열은 target appearance를 다루지만 저장 비용이 크고 nearest-neighbor 검색이 비미분적이다. 반면 이 논문의 appearance module은 폐형식(closed-form)에 가까운 계산으로 mixture parameter를 추정하고 posterior score를 내며, learning과 inference 모두 differentiable하다. 즉, “appearance를 명시적으로 모델링하면서도 전체 네트워크 안에 자연스럽게 넣을 수 있다”는 점이 가장 큰 차별점이다.

또 하나 중요한 설계는 foreground와 background를 각각 하나의 Gaussian만으로 두지 않고, hard examples를 위한 추가 component를 둔 것이다. 저자들은 실제 배경이 종종 multi-modal이고, 특히 target과 비슷한 distractor를 따로 설명할 수 있어야 좋은 분할이 가능하다고 본다. 그래서 foreground와 background 각각에 기본 component 하나, residual/hard-example component 하나를 두어 총 4개 Gaussian으로 모델을 구성한다. 이 설계는 논문의 정량 실험에서도 실제 성능 향상으로 이어진다.

## 3. 상세 방법 설명

전체 구조는 다섯 부분으로 이루어진다. 첫째는 backbone feature extractor, 둘째는 generative appearance module, 셋째는 mask propagation branch, 넷째는 두 정보를 합치는 fusion module, 다섯째는 coarse representation을 세밀한 segmentation으로 복원하는 upsampling and prediction module이다. 첫 프레임에서는 image와 정답 mask를 이용해 appearance model을 초기화하고, 이후 프레임에서는 현재 이미지 feature, 이전 coarse mask, 그리고 누적된 appearance parameter를 함께 사용해 segmentation을 예측한다.

backbone은 dilated convolution이 적용된 ResNet101이다. deepest layer의 stride를 32에서 16으로 줄였고, ImageNet pretrained weight를 사용하며 `layer4` 이전은 freeze한다. 이 backbone은 각 spatial location $p$에서 $D$차원 feature vector $x_p$를 만든다.

### Generative Appearance Module

appearance module의 핵심은 feature들의 분포를 mixture model로 보는 것이다. 논문은 각 위치의 feature를 다음과 같이 모델링한다.

$$
p(x_p)=\sum_{k=1}^{K} p(z_p=k)\, p(x_p \mid z_p=k)
$$

여기서 $z_p$는 어떤 mixture component에 속하는지를 나타내는 이산 변수이고, 각 component의 조건부 분포는 Gaussian이다.

$$
p(x_p \mid z_p=k)=\mathcal{N}(x_p \mid \mu_k, \Sigma_k)
$$

저자들은 prior $p(z_p=k)=1/K$를 균등하게 두고, component마다 평균 $\mu_k$와 공분산 $\Sigma_k$를 유지한다. 실제 구현에서는 계산 효율을 위해 공분산을 diagonal covariance로 제한한다. 즉, 각 feature channel의 분산만 추정하고 channel 간 공분산은 무시한다.

### 초기화와 업데이트

첫 프레임에서는 ground-truth binary mask가 있으므로, 각 feature를 foreground 또는 background의 base component에 직접 할당할 수 있다. 이후 프레임에서는 ground truth가 없기 때문에, 네트워크가 예측한 coarse segmentation $\tilde{y}_p$를 soft label처럼 사용한다. 일반적으로 프레임 $i$에서 soft assignment $\alpha^i_{pk}\in[0,1]$가 주어지면, component의 새로운 평균과 분산 추정치는 다음과 같이 계산된다.

$$
\tilde{\mu}_k^i=
\frac{\sum_p \alpha^i_{pk} x_p^i}{\sum_p \alpha^i_{pk}}
$$

$$
\tilde{\Sigma}_k^i=
\frac{\sum_p \alpha^i_{pk}\,\mathrm{diag}\left\{(x_p^i-\tilde{\mu}_k^i)^2+r_k\right\}}{\sum_p \alpha^i_{pk}}
$$

여기서 $r_k$는 singularity를 피하기 위한 trainable regularization vector이다. 즉, 분산이 0으로 무너지는 것을 막기 위한 역할을 한다.

첫 프레임에서는 이 추정치를 그대로 초기 parameter로 쓰고, 이후 프레임에서는 이전 parameter와 현재 추정치를 learning rate $\lambda$로 지수 평균 형태로 섞어 업데이트한다.

$$
\mu_k^i=(1-\lambda)\mu_k^{i-1}+\lambda \tilde{\mu}_k^i
$$

$$
\Sigma_k^i=(1-\lambda)\Sigma_k^{i-1}+\lambda \tilde{\Sigma}_k^i
$$

이 업데이트는 target과 background appearance가 시간에 따라 바뀌는 상황을 반영하기 위한 것이다.

### Assignment 변수의 정의

첫 프레임에서는 정답 mask $y_p$를 사용해 foreground와 background base component에 hard assignment를 준다.

$$
\alpha_{p0}^0 = 1-y_p,\qquad \alpha_{p1}^0 = y_p
$$

이후 프레임에서는 네트워크의 최종 예측 확률 $\tilde{y}_p(I_i,\theta_{i-1},\Phi)$를 이용해 soft assignment를 만든다.

$$
\alpha_{p0}^i = 1-\tilde{y}_p(I_i,\theta_{i-1},\Phi)
$$

$$
\alpha_{p1}^i = \tilde{y}_p(I_i,\theta_{i-1},\Phi)
$$

여기서 $\Phi$는 신경망 파라미터, $\theta_{i-1}$는 이전 mixture model parameter이다. 즉, 현재 segmentation prediction이 다음 시점 appearance model update의 supervision 역할을 한다.

### Hard Example용 추가 Gaussian

foreground 하나, background 하나만 두면 각 클래스 분포가 unimodal하다고 가정하는 셈인데, 실제 배경은 여러 모드로 이루어진 경우가 많다. 특히 target과 비슷한 distractor는 base background Gaussian 하나로 잘 설명되지 않을 수 있다. 그래서 저자들은 background의 hard negative를 위한 component $k=2$, foreground의 hard positive를 위한 component $k=3$를 추가한다.

이들의 assignment는 base component가 설명하지 못한 오차를 이용해 계산된다.

$$
\alpha_{p2}^i = \mathrm{ReLU}\big(p(z_p^i=0 \mid x_p^i,\mu_0^i,\Sigma_0^i)-\alpha_{p0}^i\big)
$$

$$
\alpha_{p3}^i = \mathrm{ReLU}\big(p(z_p^i=1 \mid x_p^i,\mu_1^i,\Sigma_1^i)-\alpha_{p1}^i\big)
$$

직관적으로 말하면, base background component가 어떤 픽셀을 background라고 강하게 믿지만 segmentation 측면에서는 그렇지 않은 경우, 그 차이를 residual component가 흡수하도록 만든다. 이 설계 덕분에 “비슷하게 생긴 다른 객체”나 “어려운 경계 부근”을 별도 component로 설명할 수 있다.

### Appearance Module의 출력

현재 프레임 feature $x_p^i$와 이전 프레임까지의 appearance parameter $\theta_{i-1}$가 주어지면, 각 component posterior는 Bayes rule로 계산된다.

$$
p(z_p^i=k \mid x_p^i,\theta_{i-1})=
\frac{p(z_p^i=k)\,p(x_p^i \mid z_p^i=k)}
{\sum_k p(z_p^i=k)\,p(x_p^i \mid z_p^i=k)}
$$

하지만 실제 네트워크에는 posterior probability 자체보다 log-probability score를 넣는 것이 더 좋았다고 보고한다. 저자들은 상수항을 제외한 score를 다음과 같이 계산한다.

$$
s_{pk}^i=
-\ln |\Sigma_k^{i-1}|+
\frac{(x_p^i-\mu_k^{i-1})^T(\Sigma_k^{i-1})^{-1}(x_p^i-\mu_k^{i-1})}{2}
$$

논문 설명에 따르면 softmax를 취하면 posterior를 복원할 수 있고, 따라서 이 score는 foreground/background assignment 정보를 담은 discriminative representation으로 해석할 수 있다. 실제 ablation에서 posterior probability를 직접 쓰는 것보다 이 score를 쓰는 편이 훨씬 좋았다.

### Segmentation Architecture

appearance module 외에 mask propagation module도 있다. 이 모듈은 이전 프레임의 predicted mask, 현재 프레임 feature, 초기 프레임의 feature와 정답 mask를 함께 사용해 현재 타깃의 대략적인 위치를 예측한다. 구조는 3개의 convolution layer이며, 가운데는 dilation pyramid이다. 이 부분은 RGMP 계열 아이디어를 이어받은 것으로 보인다.

appearance module 출력과 mask propagation module 출력은 concatenate된 뒤 fusion module로 들어간다. fusion module은 두 개의 convolution layer로 구성되며 coarse mask encoding을 출력한다. 여기서 두 갈래가 나온다. 하나는 predictor를 거쳐 coarse segmentation $\tilde{y}_p$를 만든다. 이 coarse prediction은 다음 시점의 appearance update와 mask propagation recurrent connection에 들어간다. 다른 하나는 upsampling module로 가서 얕은 layer의 low-level feature와 결합되며 최종 refined segmentation $\hat{y}_p$를 만든다.

이렇게 coarse branch와 refined branch를 나눈 이유는 recurrent path를 짧게 만들어 temporal optimization을 쉽게 하기 위함으로 해석된다. 저자들도 coarse mask만으로도 이전 타깃 정보를 전달하기에 충분했다고 적고 있다.

여러 객체가 있는 경우에는 객체별로 한 번씩 이 방법을 실행하고, 마지막에는 RGMP와 동일한 softmax aggregation으로 합친다. 그리고 이 aggregated soft segmentation이 recurrent connection에서 coarse segmentation을 대신한다.

### 학습 절차

학습은 recurrent 방식으로 end-to-end 수행된다. 각 비디오 샘플은 길이 $n$의 snippet이며, 첫 프레임에만 ground-truth mask가 주어진다. 네트워크는 각 프레임의 segmentation을 예측하고, 최종 refined segmentation과 coarse segmentation 모두에 cross-entropy loss를 둔다. 즉, auxiliary loss가 함께 사용된다.

데이터셋은 세 가지다.

DAVIS2017은 60개 training video를 포함하고, 각 비디오는 25~100 프레임 길이이며 single 또는 multiple object를 담는다.  
YouTube-VOS는 3471개 video로 이루어진 대규모 데이터셋이며, 20~180 프레임 길이이고 매 5번째 프레임만 label이 있다. 학습에는 label이 있는 프레임만 사용한다.  
SynthVOS는 MSRA10k의 salient object를 VOC2012 이미지 위에 붙여 인공 비디오를 만드는 synthetic dataset이다. 한 이미지에 1~5개의 객체를 붙이고 움직여 synthetic sequence를 만든다.

학습은 두 단계다. 초기 학습에서는 세 데이터셋 모두를 사용해 80 epoch 동안 half resolution $(240\times432)$에서 학습한다. batch size는 4 snippet, snippet 길이는 8 프레임이다. learning rate는 $10^{-4}$, epoch당 exponential decay는 0.95, weight decay는 $10^{-5}$이다.

이후 finetuning 단계에서는 DAVIS2017과 YouTube-VOS만 사용하고 full resolution에서 100 epoch 추가 학습한다. batch size는 2, snippet 길이는 14 프레임으로 늘린다. learning rate는 $10^{-5}$, decay는 0.985, weight decay는 $10^{-6}$이다. YouTube-VOS training set에서 떼어낸 300개 sequence held-out set으로 early stopping을 수행한다.

## 4. 실험 및 결과

논문은 먼저 YouTube-VOS에서 ablation study를 수행하고, 이어서 YouTube-VOS, DAVIS2017, DAVIS2016 세 벤치마크에서 state-of-the-art와 비교한다. 구현은 PyTorch이며, 학습은 단일 Nvidia V100 GPU에서 수행했다고 명시되어 있다.

### 평가 지표와 데이터셋

YouTube-VOS validation set은 474개 video로 구성되며, 여러 객체가 포함될 수 있다. 이 데이터셋은 seen / unseen class를 구분해 평가한다. 지표는 mean Jaccard index $J$, 즉 IoU와 mean contour accuracy $F$이다. seen class와 unseen class 각각에 대해 $J$와 $F$를 계산하고, 전체 점수 $G$는 이 네 값을 평균한 것이다.

DAVIS2017 validation set은 30개 비디오이며 single 또는 multiple object를 포함한다. 표에서는 주로 $J$ 성능을 사용한다. DAVIS2016은 DAVIS2017의 부분집합으로 20개 비디오, single object setting이다.

### Ablation Study

가장 중요한 결과는 appearance module의 기여도다. 전체 모델은 YouTube-VOS에서 $G=66.0$을 기록했다. 그런데 appearance module을 제거하면 $G=50.0$으로 급락한다. 특히 seen class의 $J$는 66.9에서 57.8로, unseen class의 $J$는 61.2에서 40.6으로 떨어진다. 즉 unseen class에서 무려 20.6%p 감소가 발생한다. 이 결과는 제안한 generative appearance model이 class-agnostic generalization에 핵심적이라는 주장을 강하게 뒷받침한다.

mask propagation module을 제거한 경우 성능은 $G=64.0$이다. 즉 전체 모델 대비 2.0%p 하락이다. 이는 이전 mask를 이용한 spatial prior가 분명 도움이 되지만, 논문 전체에서 더 본질적인 요소는 appearance modeling이라는 점을 보여준다.

foreground/background를 각각 하나의 Gaussian으로만 모델링하는 unimodal appearance 버전은 $G=64.4$를 기록했다. 전체 모델 대비 1.6%p 감소이며, 저자들은 이것이 hard examples와 distractor modeling의 중요성을 보여준다고 해석한다.

appearance model update를 끄고 첫 프레임에서만 mixture를 계산하는 No update 버전은 $G=64.9$이다. 감소폭은 1.1%p로 크지는 않지만, target과 background appearance가 시간에 따라 변하므로 online update가 실질적으로 도움이 된다는 점을 확인할 수 있다.

appearance module의 출력으로 log-probability score 대신 posterior probability를 직접 넣는 Appearance SoftMax 버전은 $G=55.8$로 크게 나빠진다. 전체 모델 대비 10.2%p 감소다. 이는 중간 표현을 너무 일찍 확률로 압축하는 것이 정보 손실을 유발한다는 해석이 가능하며, 저자들도 segmentation/classification 일반 원칙과 부합한다고 설명한다.

마지막으로, appearance model learning 과정에서 backpropagation을 끊어 end-to-end differentiability를 제거한 No end-to-end 버전은 $G=58.8$이다. 전체 모델보다 7.2%p 낮다. 이는 이 논문의 주요 주장 중 하나인 “appearance module의 end-to-end differentiable integration”이 실제로 성능에 중요하다는 강한 근거다.

### YouTube-VOS 비교

YouTube-VOS에서 제안 방법은 $G=66.0$을 기록했다. 표에 따르면 이는 온라인 fine-tuning을 하지 않는 기존 방법들뿐 아니라, 일부 online fine-tuning 기반 방법보다도 높은 점수다. 예를 들어 OSMN은 51.2, RGMP는 53.8, fine-tuning 없는 S2S는 57.6이었다. online fine-tuning을 쓰는 OSVOS는 58.8, OnAVOS는 55.2였다. 심지어 fine-tuning을 사용하는 S2S 변형 64.4보다도 높다.

특히 unseen class에서의 $J$가 61.2로 높다는 점을 저자들은 강조한다. 이는 학습 중 보지 못한 객체 category에서도 target-specific, class-agnostic appearance model이 잘 작동한다는 증거로 제시된다.

### DAVIS2017 비교

DAVIS2017 validation에서는 제안 방법이 $J=67.2$를 기록했다. 이는 표에서 causal하면서 online fine-tuning을 사용하지 않는 방법들 중 최고 성능이다. RGMP는 64.8, VideoMatch는 56.5, FAVOS는 54.6, OSMN은 52.5였다.

더 흥미로운 점은 online fine-tuning 또는 non-causal processing을 허용한 방법들과도 거의 동급이라는 것이다. 예를 들어 DyeNet은 non-causal 방식으로 67.3을 기록했고, CINM은 online fine-tuning 기반으로 67.2였다. 제안 방법은 이런 방법들과 비슷한 수준에 도달하면서도 causal이고 fine-tuning이 없다. 논문은 이것을 “online fine-tuning 기반 방법과의 성능 격차를 사실상 메웠다”고 해석한다.

### DAVIS2016 비교

DAVIS2016 validation에서는 제안 방법이 $J=82.0$을 기록했다. 최고 점수는 OnAVOS의 86.1, OSVOS-S의 85.6, DyeNet의 84.7 등으로 더 높지만, 이 데이터셋은 단일 객체이고 크기가 작아 이미 상당히 saturated되어 있다고 저자들은 설명한다. 또한 논문은 DAVIS2016에서 강한 일부 방법들이 더 크고 다양한 YouTube-VOS나 DAVIS2017에서는 일반화가 잘 되지 않는 반면, 제안 방법은 더 큰 데이터셋에서도 강하다고 주장한다.

속도 측면도 중요하다. DAVIS2016 표에서 제안 방법은 프레임당 0.07초, 즉 약 14~15 FPS 수준이다. 이는 RGMP의 0.13초보다도 빠르고, fine-tuning 기반 접근들과 비교하면 훨씬 빠르다. 논문 초록에서 언급한 “single GPU에서 15 FPS”라는 주장과 일관된다.

### 정성 평가

정성 비교에서는 RGMP, CINM, FAVOS와의 비교 그림이 제시된다. 저자들에 따르면 RGMP는 객체 일부를 잃거나 서로 다른 객체를 잘 구분하지 못하는 경향이 있고, CINM은 일부 경우 세밀한 마스크를 만들지만 특정 사례에서 실패 모드가 나타난다. FAVOS는 target 구분과 세부 경계 표현에서 약점을 보인다. 반면 제안 방법은 occlusion 상황에서도 안정적으로 target을 구분하고, 서로 비슷한 객체들 사이를 잘 판별하며, 여러 사례에서 두 객체 모두를 정확히 분할했다고 보고한다.

또한 appearance module 자체의 시각화에서는 foreground component가 target을 잘 강조하고, secondary component가 distractor나 비슷한 객체를 따로 강조하는 모습을 보여 준다고 설명한다. 이는 hard-example component 설계의 역할을 직관적으로 드러내는 부분이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 appearance modeling을 명시적이고 수학적으로 해석 가능한 형태로 network 안에 넣었다는 점이다. 많은 VOS 방법이 appearance를 다루지만, 이 논문은 이를 Gaussian mixture라는 간결한 generative model로 표현하고, inference와 update를 모두 differentiable한 closed-form 계산으로 구성했다. 덕분에 speed, interpretability, end-to-end training이라는 세 요소를 동시에 확보했다.

두 번째 강점은 ablation이 매우 설득력 있다는 점이다. appearance module 제거, unimodal variant, no update, no end-to-end 등 핵심 설계 하나씩을 제거한 결과가 체계적으로 제시되어 있고, 특히 unseen class generalization에서 appearance module의 효과가 매우 크게 나타난다. 논문의 핵심 주장과 실험 결과가 잘 맞물린다.

세 번째 강점은 accuracy-speed tradeoff가 우수하다는 점이다. 제안 방법은 online fine-tuning이 없고 causal이면서도 DAVIS2017에서 fine-tuning/non-causal 최고 수준과 거의 같은 성능을 낸다. 실제 응용 관점에서는 매우 의미 있는 결과다.

하지만 한계도 있다. 먼저 appearance model이 diagonal Gaussian mixture에 기반하기 때문에 feature channel 간 상관관계를 모델링하지 못한다. 이는 계산 효율을 위한 선택이지만, 더 복잡한 분포 구조는 포착하지 못할 수 있다. 논문은 이 한계가 얼마나 큰지는 별도로 분석하지 않았다.

또한 background와 foreground 각각에 두 개 component만 사용하는 설계가 모든 multi-modal 상황을 충분히 다룬다고 보장되지는 않는다. 저자들은 hard examples를 위한 residual component가 유용하다는 것은 보여 주었지만, component 수가 더 많을 때 어떤 tradeoff가 있는지는 제시하지 않았다.

학습과 추론이 모두 이전 prediction에 의존한다는 점도 잠재적 위험이다. 이후 프레임의 appearance update는 coarse segmentation $\tilde{y}_p$를 사용하므로, 초기 예측 오류가 누적되면 model drift가 발생할 가능성이 있다. 논문은 update의 이점을 보였지만, 장기 시퀀스에서 error accumulation이 어느 정도인지 정량적으로 분석하지는 않았다.

또한 multi-object의 경우 객체별로 독립 실행 후 softmax aggregation을 하는데, 객체들 사이의 상호작용이나 joint reasoning을 직접 모델링하지는 않는다. 이 역시 practical한 해결책이지만, 복잡한 multi-object occlusion에서는 한계가 있을 수 있다.

비판적으로 보면, mask propagation branch는 기존 RGMP 계열 요소를 상당 부분 차용하고 있고, novelty의 핵심은 appearance module에 집중되어 있다. 이는 약점이라기보다 기여의 범위를 분명히 해 주는 점이다. 즉 이 논문은 전체 VOS 구조를 완전히 새로 정의했다기보다, 기존 propagation 기반 segmentation architecture에 매우 효과적인 differentiable generative appearance model을 삽입한 연구라고 보는 것이 정확하다.

## 6. 결론

이 논문은 semi-supervised video object segmentation에서 가장 중요한 요소 중 하나인 target/background appearance representation 문제를 정면으로 다룬다. 저자들은 foreground와 background의 deep feature distribution을 class-conditional Gaussian mixture로 모델링하고, 이를 한 번의 forward pass로 초기화하며 이후 프레임마다 효율적으로 업데이트하는 appearance module을 제안했다. 이 모듈은 posterior class probability에 대응하는 강한 discriminative cue를 제공하고, learning과 inference 모두 differentiable하여 전체 segmentation pipeline을 end-to-end로 학습할 수 있게 한다.

실험적으로는 YouTube-VOS에서 당시 공개된 방법들 중 최고 성능을 기록했고, DAVIS2017에서는 causal이며 online fine-tuning이 없는 방법들 중 최고 수준에 도달했다. 또한 약 15 FPS 수준의 속도를 달성해 실용성도 확보했다. 특히 unseen class에서의 강한 성능은 이 방법이 특정 semantic category에 과도하게 의존하지 않고, 첫 프레임에서 주어진 object appearance를 class-agnostic하게 잘 포착한다는 점을 보여 준다.

따라서 이 연구의 핵심 기여는 “빠르면서도 강력한 appearance model을 end-to-end VOS 네트워크 안에 넣는 방법”을 제시했다는 데 있다. 실제 응용 측면에서는 실시간 또는 온라인 video understanding 시스템에 적합한 설계를 제시했고, 향후 연구 측면에서는 differentiable probabilistic model을 neural network 안의 모듈로 통합하는 방향의 좋은 사례로 볼 수 있다. 특히 appearance modeling, temporal adaptation, structured probabilistic reasoning을 결합하는 후속 연구에 중요한 출발점이 될 가능성이 크다.
