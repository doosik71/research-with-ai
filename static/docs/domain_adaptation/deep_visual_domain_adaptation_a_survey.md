# Deep Visual Domain Adaptation: A Survey

* **저자**: Mei Wang, Weihong Deng
* **발표연도**: 2018
* **arXiv**: [https://arxiv.org/abs/1802.03601](https://arxiv.org/abs/1802.03601)

## 1. 논문 개요

이 논문은 computer vision에서의 **deep domain adaptation (deep DA)** 연구를 체계적으로 정리한 survey이다. 출발점은 매우 현실적이다. 딥러닝이 강력한 성능을 보이더라도, 새로운 도메인마다 대규모 라벨 데이터를 다시 수집하고 주석하는 일은 비용이 크고 시간이 오래 걸린다. 반면 다른 도메인에는 이미 풍부한 데이터가 존재하는 경우가 많다. 따라서 잘 라벨된 source domain의 지식을, 라벨이 거의 없거나 전혀 없는 target domain으로 옮길 수 있다면 실제 응용에서 큰 가치가 있다.

논문이 다루는 핵심 연구 문제는 다음과 같다. source와 target이 같은 task를 수행하더라도, 조명, 자세, 해상도, 이미지 품질, 센서 차이, 스타일 차이 등으로 인해 두 도메인 사이에 **distribution shift** 혹은 **domain shift**가 생긴다. 이때 source에서 잘 학습된 모델은 target에서 성능이 급격히 떨어질 수 있다. 전통적인 shallow DA는 instance reweighting이나 shared subspace 학습으로 이 문제를 완화했지만, 딥러닝 시대에는 표현 자체를 더 전이 가능하게 만들 필요가 있다. 이 논문은 바로 그 지점에서, deep network 안에 adaptation을 직접 내장하는 방식의 방법들을 정리한다.

이 survey의 중요성은 단순한 논문 나열에 있지 않다. 저자들은 deep DA를 다음 네 축으로 정리하려고 한다. 첫째, domain divergence의 성격에 따라 시나리오를 분류한다. 둘째, training loss 관점에서 방법론을 재분류한다. 셋째, image classification을 넘어 face recognition, semantic segmentation, object detection, style translation, person re-identification까지 응용 범위를 확장해 정리한다. 넷째, 당시 방법들의 결핍과 미래 방향을 짚는다. 즉, 이 논문은 deep DA를 “문제 설정–방법론–응용–한계”의 흐름으로 이해하게 해 주는 지형도 역할을 한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 deep domain adaptation을 단일 알고리즘으로 보지 않고, **문제 설정과 학습 기준(loss/criterion)에 따라 구조적으로 분해해서 이해해야 한다**는 점이다. 저자들은 먼저 domain adaptation을 transfer learning의 한 하위 범주로 놓고, source와 target의 차이가 어디에서 오는지를 기준으로 전체 공간을 정리한다. 그 위에서 deep DA 방법들을 “무엇을 맞추려 하는가”라는 관점에서 다시 분류한다. 예를 들어 어떤 방법은 클래스 정보로 정렬하고, 어떤 방법은 통계적 분포를 맞추고, 어떤 방법은 도메인 판별기를 속이도록 학습하며, 어떤 방법은 reconstruction을 통해 공통 표현과 도메인 고유 표현을 동시에 다룬다.

기존 shallow survey들과 비교했을 때 이 논문의 차별점은 크게 세 가지다. 첫째, 단순히 shallow DA를 딥 특징으로 대체한 수준이 아니라, **back-propagation으로 직접 적응 기준을 최적화하는 narrow sense의 deep DA**에 초점을 둔다. 둘째, one-step adaptation뿐 아니라 source와 target이 너무 멀리 떨어진 경우를 위한 **multi-step (transitive) DA**를 별도 범주로 다룬다. 셋째, classification 중심 논의를 넘어서 다양한 visual task에 deep DA가 어떻게 응용되는지를 폭넓게 정리한다.

또 하나의 중요한 관점은, 딥 특징이 자동으로 domain invariant하지는 않다는 인식이다. 논문은 Donahue 등과 Yosinski 등의 선행 관찰을 바탕으로, deep feature도 상위 층으로 갈수록 source-specific해질 수 있고 domain shift의 영향을 여전히 크게 받는다고 본다. 따라서 deep network를 쓰는 것만으로는 부족하며, **표현이 semantic하면서도 domain invariant하도록 만드는 추가 학습 기준**이 필요하다고 주장한다. 이 관점이 이후 논문 전개의 핵심 축이 된다.

## 3. 상세 방법 설명

### 3.1 기본 정의와 문제 설정

논문은 먼저 domain과 task를 형식적으로 정의한다. 하나의 domain $\mathcal{D}$는 feature space $\mathcal{X}$와 marginal distribution $P(X)$로 구성된다. task $\mathcal{T}$는 label space $\mathcal{Y}$와 예측 함수 $f(\cdot)$ 또는 조건부분포 $P(Y|X)$로 표현된다.

source domain은 $\mathcal{D}^s={\mathcal{X}^s, P(X)^s}$, target domain은 $\mathcal{D}^t={\mathcal{X}^t, P(X)^t}$로 둔다. task는 DA에서 동일하다고 가정하므로 $\mathcal{T}^s=\mathcal{T}^t$이다. 즉 domain adaptation은 label space나 task가 다르기보다, **입력 분포 혹은 feature space의 차이** 때문에 발생하는 문제로 정리된다.

이때 설정은 크게 두 방향으로 나뉜다.

첫째는 **homogeneous DA**이다. 여기서는 source와 target의 feature space가 같고 차원도 같다. 즉 $\mathcal{X}^s=\mathcal{X}^t$이고, 차이의 핵심은 $P(X)^s \neq P(X)^t$라는 분포 이동이다.

둘째는 **heterogeneous DA**이다. 여기서는 feature space 자체가 다르며, 차원도 다를 수 있다. 예를 들어 RGB와 depth, VIS와 NIR, photo와 sketch, text와 image 같은 경우가 여기에 해당한다.

그리고 각 설정 안에서 target label의 유무에 따라 supervised, semi-supervised, unsupervised DA로 나뉜다. 이 survey는 semi-supervised는 supervised와 unsupervised의 결합으로 보고, 설명의 중심은 supervised와 unsupervised에 둔다.

마지막으로 source와 target 사이의 거리에 따라 **one-step DA**와 **multi-step DA**를 구분한다. source와 target이 직접 연결될 수 있으면 one-step으로 충분하지만, 거리가 너무 멀면 intermediate domain들을 거쳐 점진적으로 지식을 옮겨야 한다.

### 3.2 One-step DA의 세 가지 큰 축

논문은 one-step deep DA를 크게 **discrepancy-based**, **adversarial-based**, **reconstruction-based** 세 가지로 분류한다. 이 분류는 사실상 “어떤 loss 혹은 criterion으로 source와 target을 가깝게 만들 것인가”를 기준으로 한다.

#### 3.2.1 Discrepancy-based approaches

이 계열은 source와 target 사이의 차이를 어떤 식으로든 수치화한 뒤, 그 차이를 줄이도록 deep network를 fine-tuning하는 방식이다. 논문은 이를 다시 class criterion, statistic criterion, architecture criterion, geometric criterion으로 나눈다.

##### (1) Class criterion

가장 기본적인 경우는 target 쪽에 일부 라벨이 있는 supervised DA이다. 이때는 보통 softmax 기반 cross-entropy를 사용한다.

$$
\mathcal{L} = -\sum_{i=0}^{N} y_i \log \hat{y}_i
$$

여기서 $\hat{y}_i$는 softmax 출력 확률이다. 즉 target의 소량 라벨을 활용해 source로 pretrained된 네트워크를 fine-tune한다.

논문은 여기서 더 나아가 **soft label loss**를 소개한다. Hinton의 지식 증류 관점에서 softmax temperature $T$를 크게 두면 클래스 간 상대적 유사성이 살아 있는 부드러운 분포를 얻을 수 있다.

$$
q_i=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

이 아이디어는 단순히 정답 클래스 하나만 맞추는 것이 아니라, 예를 들어 숫자 2가 3과 7 중 무엇에 더 비슷한지를 같이 전달해 준다. Tzeng 등은 domain confusion loss와 soft label loss를 함께 써서 클래스 관계를 도메인 간에도 보존하려고 했다. Gebru 등은 fine-grained recognition에서 class-level soft loss뿐 아니라 attribute-level soft loss도 함께 사용했다.

또 다른 흐름은 **metric learning**이다. 같은 클래스지만 도메인이 다른 샘플은 가까이, 다른 클래스 샘플은 멀어지도록 학습한다. Hu 등은 intra-class compactness $S_c$, inter-class separability $S_b$, 그리고 source-target 간 MMD 항을 결합한 목적함수를 제안한다.

$$
\min \mathcal{J} = S_c^{(M)} - \alpha S_b^{(M)} + \beta D_{ts}^{(M)}(\mathcal{X}^s,\mathcal{X}^t) - \gamma \sum_{m=1}^M \left(|W^{(m)}|_F^2 + |b^{(m)}|_2^2\right)
$$

이 식의 의미는 분명하다. 같은 클래스는 뭉치고, 다른 클래스는 벌어지며, 동시에 source와 target 표현 분포도 가까워져야 한다는 것이다.

한편 target 라벨이 없을 때는 pseudo label이나 attribute를 대체 정보로 쓴다. pseudo label은 현재 모델의 posterior가 가장 큰 클래스를 임시 정답으로 쓰는 방식이다.

$$
\hat{y}_j^t = \arg\max_c p(y_j^t=c|x_j^t)
$$

이 방식은 완전히 정답이 없는 target에서도 조건부분포 정렬이나 entropy minimization에 활용될 수 있다. 다만 pseudo label은 초기에 오분류가 많으면 오류를 증폭시킬 수 있다는 점이 구조적 한계다. 논문도 이 위험을 암시하지만 구체적 안정화 전략을 깊게 파고들지는 않는다.

또 attribute-based transfer도 소개된다. 이는 unseen class recognition과 비슷한 아이디어로, 클래스 자체보다 속성(attribute) 벡터를 매개로 target을 해석하는 접근이다. 핵심 식은 클래스 posterior를 attribute posterior의 곱으로 전개하는 것이다.

$$
p(y|x)=\frac{p(y)}{p(a^y)} \prod_{m=1}^{M} p(a_m^y|x)
$$

이 식은 “이 샘플이 어떤 속성 조합을 보이는가”를 통해 간접적으로 클래스를 추정하는 구조를 보여 준다.

##### (2) Statistic criterion

이 범주는 라벨이 거의 없는 unsupervised DA에서 특히 중요하다. 핵심은 source와 target의 feature distribution을 맞추는 것이다.

가장 대표적인 척도는 **MMD (Maximum Mean Discrepancy)**이다. 두 분포의 RKHS 상 평균 임베딩 차이를 재는 방식으로, 정의는 다음과 같다.

$$
\text{MMD}^2(s,t)=\sup_{|\phi|_{\mathcal{H}}\le 1}\left| \mathbb{E}_{x^s\sim s}[\phi(x^s)]-\mathbb{E}_{x^t\sim t}[\phi(x^t)] \right|_{\mathcal{H}}^2
$$

실제 데이터에 대해선 경험적 추정량을 사용한다.

$$
\text{MMD}^2(D_s,D_t)=\left|\frac{1}{M}\sum_{i=1}^M \phi(x_i^s)-\frac{1}{N}\sum_{j=1}^N \phi(x_j^t)\right|_H^2
$$

DDC는 CNN의 adaptation layer에서 MMD를 추가했고, 목적함수는 classification loss와 MMD penalty의 합으로 표현된다.

$$
\mathcal{L}= \mathcal{L}_C(X^L,y)+\lambda \text{MMD}^2(X^s,X^t)
$$

DAN은 한 층이 아니라 여러 층에 MMD를 걸고 multiple kernel을 사용해 보다 강한 정렬을 시도한다. JAN은 더 나아가 feature와 output label의 **joint distribution**을 맞추는 JMMD를 제안한다. RTN은 MMD로 feature를 맞추는 것에 더해 classifier 자체도 residual 방식으로 적응시킨다. weighted MMD는 클래스 비중이 source와 target에서 다를 수 있다는 현실적 문제를 반영한다.

또 다른 대표 기법은 **CORAL**이다. 이는 covariance 같은 2차 통계를 맞추는 방식이며, deep CORAL loss는 다음과 같다.

$$
\mathcal{L}_{CORAL}=\frac{1}{4d^2}|C_S-C_T|_F^2
$$

즉 평균만 맞추는 것이 아니라 feature 간 공분산 구조까지 정렬하려는 것이다.

MMD가 적절한 kernel을 쓰면 고차 모멘트까지 간접적으로 본다는 해석에서 출발해, CMD는 아예 여러 차수의 central moment를 직접 맞춘다.

$$
CMD_K(X^s,X^t)=\frac{1}{b-a}|E(X^s)-E(X^t)|_2 + \sum_{k=2}^{K}\frac{1}{|b-a|^k}|C_k(X^s)-C_k(X^t)|_2
$$

이 식의 직관은 평균뿐 아니라 분포의 모양까지 단계적으로 맞추겠다는 것이다.

##### (3) Architecture criterion

이 범주는 loss를 추가하는 대신, 네트워크 구조를 바꾸어 더 transferable한 표현을 만들려는 접근이다.

대표 예로 Rozantsev 등은 source와 target이 완전히 같은 가중치를 공유해야 한다고 보지 않는다. 대신 두 스트림의 가중치가 **related but not identical** 하도록 regularizer를 둔다.

$$
r_w(\theta_j^s,\theta_j^t)=\exp(|\theta_j^s-\theta_j^t|^2)-1
$$

혹은 조금 더 유연하게 선형변환까지 허용한다.

$$
r_w(\theta_j^s,\theta_j^t)=\exp(|a_j\theta_j^s+b_j-\theta_j^t|^2)-1
$$

이는 source와 target이 비슷하지만 완전히 같지는 않다는 현실을 잘 반영한다.

또 하나의 유명한 흐름은 **Adaptive Batch Normalization (AdaBN)**이다. 논문은 class-related knowledge는 주로 weight matrix에, domain-related knowledge는 BN 통계량에 저장된다고 해석한다. 따라서 target 도메인에서 평균과 표준편차를 다시 계산하면 어느 정도 정렬이 가능하다고 본다.

$$
BN(X^t)=\lambda \left(\frac{x-\mu(X^t)}{\sigma(X^t)}\right)+\beta
$$

여기서 핵심은 BN 통계값을 domain-specific하게 다루는 것이다. 이후 AutoDIAL, Instance Normalization 등도 이 흐름의 연장선으로 소개된다.

또 domain-guided dropout은 특정 도메인에 불필요한 뉴런을 끄는 방식이다. 뉴런 하나를 제거했을 때 loss가 얼마나 나빠지는지를 기반으로 중요도를 정의한다.

$$
s_i = \mathcal{L}(g(x)_{\backslash i})-\mathcal{L}(g(x))
$$

즉 어떤 뉴런이 특정 도메인에는 거의 도움이 안 된다면 적극적으로 억제하겠다는 아이디어다.

##### (4) Geometric criterion

이 범주는 source와 target 사이를 기하학적으로 연결하려는 접근이다. Grassmann manifold 위에서 source subspace와 target subspace를 잇는 geodesic 경로를 만들고, 그 사이 중간 subspace들을 통해 도메인 간 상관관계를 찾는다.

DLID는 이 아이디어를 딥러닝 쪽으로 가져온 사례다. source 데이터에서 target 데이터를 조금씩 섞어가며 intermediate dataset들을 생성하고, 이 연속적인 경로를 따라 representation을 학습한다. 즉 source에서 target으로 갑자기 점프하지 않고, 점진적으로 바뀌는 중간 환경들을 통해 적응을 부드럽게 만든다.

#### 3.2.2 Adversarial-based approaches

이 계열은 GAN의 아이디어를 domain adaptation으로 가져온 것이다. 기본 철학은 간단하다. feature extractor가 만든 표현을 보고 domain discriminator가 source/target을 구분하지 못하게 만들면, 그 표현은 domain invariant에 가까울 것이다.

GAN의 기본 목적함수는 다음과 같다.

$$
\min_G \max_D V(D,G)= \mathbb{E}_{x\sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z\sim p_z(z)}[\log(1-D(G(z)))]
$$

DA에서는 generator가 꼭 픽셀 이미지를 만들 필요는 없고, feature mapping 자체가 생성자 역할을 할 수 있다.

##### (1) Generative models

이 범주는 source 정보를 유지하면서 target처럼 보이는 synthetic sample을 생성하는 방식이다.

CoGAN은 두 개의 GAN을 두어 source와 target 이미지를 각각 생성하되, 일부 생성층과 판별층의 가중치를 공유한다. 이렇게 하면 명시적 paired supervision 없이도 공통 high-level semantics를 공유하는 두 도메인의 샘플을 동시에 생성할 수 있다.

다른 흐름은 source image를 target-like image로 변환하는 것이다. Bousmalis 등은 adversarial loss, task loss, content-similarity loss를 함께 쓴다.

$$
\min_{G,T}\max_D V(D,G)= \alpha \mathcal{L}_d(D,G)+\beta \mathcal{L}_t(T,G)+\gamma \mathcal{L}_c(G)
$$

여기서 $\mathcal{L}_d$는 target과 비슷하게 보이게 하는 adversarial loss, $\mathcal{L}_t$는 생성된 이미지에도 원래 source label이 유지되도록 하는 분류 loss, $\mathcal{L}_c$는 원본 내용이 너무 많이 바뀌지 않게 하는 content similarity loss다. 이 구조는 synthetic-to-real adaptation에서 매우 자연스럽다.

##### (2) Non-generative models

이 범주는 이미지를 생성하지 않고 feature representation 자체를 adversarial하게 맞춘다. 가장 대표적인 것이 **DANN**이다. DANN은 shared feature extractor와 label predictor, domain classifier로 이루어진다. 핵심 장치는 **GRL (gradient reversal layer)** 이다. 순전파 때는 identity지만, 역전파 때는 gradient의 부호를 뒤집는다. 따라서 feature extractor는 label prediction에는 유리하면서 domain classification에는 불리한 방향으로 학습된다. 결과적으로 source와 target feature가 섞이게 된다.

ADDA는 DANN과 달리 source mapping과 target mapping의 가중치를 분리한다. 먼저 source encoder와 classifier를 source labeled data로 학습한 뒤, target encoder를 adversarial objective로 source encoder의 feature space에 맞춘다. 논문에 제시된 목적함수는 세 부분이다.

먼저 source 분류기 학습:

$$
\min_{M^s,C}\mathcal{L}_{cls}(X^s,Y^s)
= -\mathbb{E}_{(x^s,y^s)\sim(X^s,Y^s)} \sum_{k=1}^{K}\mathbb{1}_{[k=y^s]}\log C(M^s(x^s))
$$

다음으로 discriminator 학습:

$$
\min_D \mathcal{L}_{advD}(X^s,X^t,M^s,M^t)
= -\mathbb{E}_{x^s\sim X^s}[\log D(M^s(x^s))]
-\mathbb{E}_{x^t\sim X^t}[\log(1-D(M^t(x^t)))]
$$

마지막으로 target mapping을 속이도록 학습:

$$
\min_{M^s,M^t}\mathcal{L}_{advM}(M^s,M^t)
= -\mathbb{E}_{x^t\sim X^t}[\log D(M^t(x^t))]
$$

이 구조는 source와 target의 입력 특성이 꽤 다를 때 더 유연할 수 있다.

또 논문은 SAN, Wasserstein distance 기반 적응, classifier discrepancy 기반 적응 등도 언급한다. 중요한 점은 adversarial 방법들이 단순 이진 domain 분류를 넘어서, 부분 라벨 공유(partial transfer), Wasserstein metric, 다중 classifier disagreement 같은 더 정교한 정렬 기준으로 확장되고 있다는 것이다.

#### 3.2.3 Reconstruction-based approaches

이 계열은 source와 target을 동시에 잘 설명하는 표현을 얻기 위해 reconstruction을 보조 과제로 둔다. 핵심 가정은 “복원까지 가능하려면 표현이 도메인의 중요한 구조를 담고 있어야 하며, 동시에 적절히 공유/분리된 표현을 배울 수 있다”는 것이다.

##### (1) Encoder-decoder reconstruction

DRCN은 shared encoder 위에 두 개의 가지를 둔다. 하나는 source label을 이용한 supervised classification branch이고, 다른 하나는 target 입력을 복원하는 reconstruction branch이다. 목적함수는 다음과 같다.

$$
\min \lambda \mathcal{L}_c({\theta_{enc},\theta_{lab}}) + (1-\lambda)\mathcal{L}_r({\theta_{enc},\theta_{dec}})
$$

여기서 $\mathcal{L}_c$는 분류 cross-entropy, $\mathcal{L}_r$는 reconstruction squared loss이다.

$$
\mathcal{L}_r = |x-f_r(x)|_2^2
$$

이 구조의 직관은 명확하다. encoder는 source 분류에 유용하면서도 target 재구성도 가능해야 하므로, 양쪽 도메인에 모두 의미 있는 표현을 배우게 된다.

DSN은 이 아이디어를 더 정교화한다. shared encoder는 공통 표현을, private encoder는 도메인 특화 표현을 담당한다. decoder는 shared와 private를 합쳐 입력을 복원한다. 그러면 classifier는 shared representation 위에서 학습되어 도메인 특수성에 덜 오염된 표현을 사용할 수 있다. 이 survey는 DSN이 reconstruction과 discrepancy/adversarial loss를 결합할 수 있는 hybrid 구조라는 점도 강조한다.

##### (2) Adversarial reconstruction

CycleGAN, DualGAN, DiscoGAN은 원래 image-to-image translation으로 잘 알려져 있지만, 이 survey는 이를 DA 관점에서도 본다. 특히 paired example이 없어도 두 도메인 사이의 왕복 매핑을 학습할 수 있다는 점이 중요하다.

CycleGAN의 핵심은 adversarial loss와 cycle consistency loss의 결합이다.

$$
\mathcal{L}_{GAN}(G,D_Y,X,Y)
= \mathbb{E}_{y\sim p_{data}(y)}[\log D_Y(y)]

* \mathbb{E}_{x\sim p_{data}(x)}[\log(1-D_Y(G(x)))]
  $

$$
\mathcal{L}_{cyc}(G,F)
= \mathbb{E}_{x\sim data(x)}[|F(G(x))-x|_1]

* \mathbb{E}_{y\sim data(y)}[|G(F(y))-y|_1]
  $

첫 번째 항은 $G(X)$가 target domain처럼 보이게 만들고, 두 번째 항은 source를 target으로 갔다가 다시 source로 돌아왔을 때 원래 입력이 복원되도록 한다. 즉 단순 스타일 변환이 아니라, 의미를 보존한 도메인 이동을 강제하는 셈이다.

### 3.3 Hybrid approaches

논문은 실제 강한 방법들이 위 범주를 섞는 경우가 많다고 본다. 예를 들어 Tzeng 등의 방법은 domain confusion loss와 soft label loss를 함께 쓰고, RTN은 MMD와 residual classifier adaptation을 동시에 사용하며, DSN은 reconstruction과 MMD 혹은 adversarial loss를 결합한다. 이 관찰은 중요하다. 왜냐하면 deep DA의 본질이 “한 가지 loss로 모든 것을 해결”하는 데 있지 않고, **분류 가능성, 분포 정렬, 구조 보존, 도메인 특수성 분리**를 동시에 다뤄야 하는 다목적 최적화 문제라는 점을 보여 주기 때문이다.

### 3.4 Heterogeneous DA와 Multi-step DA

heterogeneous DA에서는 source와 target의 feature space가 다르므로 homogeneous DA처럼 같은 입력 구조를 가정하기 어렵다. 논문은 이를 두 시나리오로 나눈다. 하나는 둘 다 이미지지만 modality나 style이 다른 경우이고, 다른 하나는 text-image처럼 매체 자체가 다른 경우다.

첫 번째 경우는 이미지를 같은 해상도로 맞춰 CNN에 넣을 수 있어 class/statistic criterion이 어느 정도 작동한다. 예를 들어 RGB-to-depth, VIS-to-NIR, photo-to-sketch 적응이 여기에 속한다. 두 번째 경우는 훨씬 어렵다. Weakly-shared deep transfer networks나 transfer neural trees 같은 구조가 소개되지만, 저자들 스스로도 **heterogeneous deep DA는 아직 충분히 성숙하지 않았다**고 평가한다.

multi-step DA는 source와 target이 직접 이어지지 않을 때 intermediate domain을 활용한다. 방법은 세 가지로 정리된다.

첫째, **hand-crafted**는 사람이 중간 도메인을 정한다.
둘째, **instance-based**는 보조 데이터 중 일부 샘플을 골라 intermediate를 구성한다. DDTL은 reconstruction error를 기준으로 점진적으로 관련 샘플을 선택한다. 논문에 제시된 목적함수는 선택된 source, intermediate, target reconstruction 오차의 가중합이다.

$$
\begin{aligned}
\mathcal{J}_1(f_e,f_d,v_S,v_T) = & \frac{1}{n_S}\sum_{i=1}^{n_S} v_S^i |\hat{x}_S^i-x_S^i|_2^2 \\
&+\frac{1}{n_I}\sum_{i=1}^{n_I} v_I^i |\hat{x}_I^i-x_I^i|_2^2 \\
&+\frac{1}{n_T}\sum_{i=1}^{n_T}|\hat{x}_T^i-x_T^i|_2^2 \\
&+R(v_S,v_T)
\end{aligned}
$$

여기서 $v_S^i$, $v_I^i$는 어떤 샘플을 intermediate 경로에 포함할지 선택하는 indicator이다.

셋째, **representation-based**는 이전 네트워크를 freeze하고 그 intermediate representation을 다음 네트워크 입력으로 넘긴다. progressive networks가 대표 사례다. 이런 구조는 이전 도메인의 지식을 보존하면서 새로운 도메인으로 확장할 수 있다는 장점이 있다.

## 4. 실험 및 결과

이 논문은 survey이므로 하나의 통일된 실험을 수행한 것이 아니라, 기존 논문들의 대표 결과를 묶어서 보여 준다. 따라서 여기서 중요한 것은 “어떤 방법이 왜 어느 설정에서 유리했는가”를 읽는 것이다.

### 4.1 이미지 분류

가장 중심이 되는 응용은 image classification이다. 논문은 Office-31, MNIST, USPS, SVHN 같은 고전적 benchmark를 사용한 결과를 표로 정리한다.

Office-31은 Amazon(A), DSLR(D), Webcam(W) 세 도메인으로 이루어지고, 31개 object class를 포함한다. 이 데이터셋은 시각적으로 비슷한 물체라도 촬영 조건과 해상도가 달라 domain shift가 분명하다. 논문이 인용한 결과에서, AlexNet baseline 대비 DDC, DAN, RTN, JAN, DANN 같은 적응 방법들은 대체로 성능을 높인다.

특히 [74]의 표에서는 Office-31 평균 정확도가 AlexNet 70.1에서 DDC 70.6, DAN 72.9, RTN 73.7, JAN 76.3, DANN 74.3으로 향상된다. 이 결과는 단순 feature extractor보다 분포 정렬을 넣은 방법이 유리하며, 특히 **joint distribution을 맞추는 JAN**이 강하다는 메시지를 준다.

또 [134]의 표에서는 CMD가 deep CORAL, AdaBN, DANN 대비 강한 평균 성능을 보이며, Office-31 평균이 79.9로 보고된다. 이는 higher-order moment matching이 효과적일 수 있음을 시사한다.

[118]의 supervised setting 결과에서는 domain confusion, soft labels, 그리고 둘의 결합이 비교된다. 평균 정확도는 baseline 66.2에서 결합 모델이 82.22까지 올라간다. 특히 A→W, A→D 같은 task에서 큰 폭 개선이 나타난다. 이는 클래스 간 관계를 유지하는 soft label이 supervised adaptation에서 매우 강력한 보조 신호가 될 수 있음을 보여 준다.

digits benchmark에서는 M→U, U→M, S→M 같은 작업에서 adversarial 계열의 이점이 두드러진다. [119]의 표에서 VGG-16 baseline 대비 DANN, CoGAN, ADDA가 전반적으로 향상되며, 예를 들어 U→M에서는 baseline 57.1에서 DANN 73.0, CoGAN 89.1, ADDA 90.1로 크게 증가한다. 이는 픽셀 스타일이나 저수준 통계 차이가 큰 숫자 데이터셋에서 adversarial alignment가 매우 효과적임을 보여 준다.

다만 이 survey도 분명히 말하듯이, 서로 다른 논문은 backbone, preprocessing, tuning strategy, protocol이 다르기 때문에 **절대적인 공정 비교는 어렵다**. 이 점은 독자가 반드시 염두에 두어야 한다.

### 4.2 얼굴 인식

face recognition에서는 pose, ethnicity, sensor, illumination, age variation 등이 domain shift의 원인이 된다. 논문은 BAE, SSPP-DAN, unlabeled video adaptation, adult-to-infant smile detection 같은 사례를 소개한다. 여기서 중요한 메시지는 classification에서 잘 먹히던 adaptation이 face domain에서도 유용하지만, 얼굴 분야는 identity 보존이 특히 중요하므로 단순 분포 정렬만으로는 부족할 수 있다는 점이다.

### 4.3 객체 검출

object detection에서는 classifier뿐 아니라 bounding box annotation의 부족이 큰 문제다. 논문은 LSDA, semantic relatedness 기반 transfer, Faster R-CNN에 adversarial adaptation을 넣는 방식 등을 소개한다. 특히 detection은 classification보다 구조가 복잡하므로, image-level adaptation과 instance-level adaptation을 동시에 고려해야 한다는 흐름이 보인다.

### 4.4 의미 분할

semantic segmentation은 pixel-level prediction이 필요하므로 domain shift에 더 민감하다. 논문은 FCNs in the Wild, cross-city adaptation, curriculum adaptation, target-guided distillation 등을 언급한다. segmentation에서는 feature-level adversarial loss만으로 충분하지 않고, spatial layout, class-wise alignment, pseudo labeling, projected image-space alignment 등 **공간 구조를 반영하는 적응 전략**이 중요하다는 점이 강조된다.

### 4.5 이미지-이미지 변환, person re-ID, captioning

style transfer와 image-to-image translation은 사실상 heterogeneous DA의 대표 응용으로 읽힌다. pix2pix, CycleGAN, DualGAN, DiscoGAN, DGCAN 등이 소개된다.
person re-ID에서는 domain-guided dropout, SPGAN 같은 방법이 언급된다.
image captioning에서는 paired source data와 unpaired target data를 연결하기 위해 adversarial training과 dual learning이 쓰인다.

즉 이 survey가 전하려는 실험적 메시지는 명확하다. **deep DA는 단순 분류 문제를 넘어서 다양한 visual task에 이미 확장되고 있으며, 각 task에 맞는 구조적 제약과 loss 설계가 성능을 좌우한다**는 것이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 정리 체계가 매우 명료하다는 점이다. domain adaptation을 먼저 homogeneous/heterogeneous, supervised/semi-supervised/unsupervised, one-step/multi-step으로 나눈 뒤, 방법론을 discrepancy/adversarial/reconstruction으로 재구성한다. 이 덕분에 독자는 개별 논문을 외우지 않아도, “이 방법은 어떤 설정에서 어떤 기준으로 도메인 차이를 줄이는가?”라는 질문으로 전체 분야를 이해할 수 있다.

또 survey임에도 단순히 분류만 하지 않고 수식, 대표 모델, 응용 과제를 충분히 소개한다는 점도 장점이다. 특히 MMD, CORAL, adversarial loss, reconstruction objective 같은 핵심 공식을 제시해 각 방법의 작동 원리를 비교할 수 있게 한다. 그리고 object detection, semantic segmentation, person re-identification, image captioning까지 포함해 “deep DA가 실제로 어디에 쓰이는가”를 폭넓게 보여 준다.

그러나 한계도 분명하다.

첫째, 이 논문은 어디까지나 **survey**이므로 새로운 통합 이론이나 새로운 benchmark를 제시하지 않는다. 방법들을 잘 정리하지만, 언제 어떤 방법이 본질적으로 더 우월한지에 대한 엄밀한 통합 분석은 제한적이다.

둘째, 실험 비교의 공정성이 약하다. 논문 스스로도 서로 다른 논문들의 backbone과 protocol이 달라 직접 비교가 어렵다고 인정한다. 따라서 표에 있는 수치를 그대로 순위표처럼 받아들이면 위험하다.

셋째, 방법론 분류는 유용하지만 경계가 완전히 분리되지는 않는다. 실제 많은 강한 모델은 discrepancy, adversarial, reconstruction을 동시에 섞는다. 따라서 이 분류는 이해를 돕는 데는 탁월하지만, 최신 혹은 복합 모델을 엄밀히 분리하기에는 다소 인위적일 수 있다.

넷째, 당시 시점의 한계이기도 하지만, **heterogeneous deep DA**에 대한 논의는 상대적으로 약하다. 저자들도 이 부분이 아직 초기 단계라고 인정한다. 즉 논문은 이 문제를 중요하게 제기하지만, 실질적으로는 homogeneous unsupervised DA에 더 큰 비중을 둔다.

다섯째, label space mismatch 문제, 즉 source와 target이 완전히 같은 클래스 집합을 갖지 않는 현실적 상황은 결론부에서 중요하다고 언급되지만 본문 핵심 분류에는 아직 충분히 녹아 있지 않다. 이는 이후 partial DA, open-set DA, universal DA로 발전할 중요한 문제인데, 이 survey에서는 시작점만 제시된다.

비판적으로 보면, 이 논문은 “representation을 domain invariant하게 만드는 것”을 deep DA의 핵심 목표로 본다. 이는 매우 강력한 관점이지만, 동시에 너무 강한 invariance가 class-discriminative structure를 훼손하거나 negative transfer를 일으킬 수 있다는 문제는 상대적으로 덜 다뤄진다. 즉 alignment 자체보다 **무엇을 보존하면서 무엇만 제거할 것인가**의 문제가 이후 연구에서 더 중요해질 여지가 크다.

## 6. 결론

이 논문은 deep visual domain adaptation 분야를 체계적으로 정리한 대표적 survey로, deep DA를 **문제 설정**, **학습 기준**, **적용 방식**, **응용 과제**라는 네 축에서 이해하게 해 준다. 핵심 기여는 다음과 같이 요약할 수 있다.
첫째, homogeneous/heterogeneous, supervised/semi-supervised/unsupervised, one-step/multi-step이라는 설정별 분류를 제시했다.
둘째, one-step deep DA를 discrepancy-based, adversarial-based, reconstruction-based로 정리하고, 각 범주 안의 대표 하위 기법을 비교했다.
셋째, multi-step DA를 intermediate domain 선택 방식에 따라 hand-crafted, instance-based, representation-based로 설명했다.
넷째, image classification을 넘어 face recognition, detection, segmentation, style translation, re-ID, captioning까지 폭넓은 응용을 연결했다.

실제 적용 측면에서 이 survey가 주는 메시지는 분명하다. 라벨이 부족한 현실 세계의 비전 문제에서, source의 지식을 target으로 옮기는 기술은 필수적이며, 그 중심에는 **semantic하면서도 domain-invariant한 representation을 학습하는 것**이 있다. 다만 이 representation은 단순히 분포를 맞추는 것만으로 충분하지 않고, 클래스 구조 보존, 도메인 특수성 처리, intermediate bridge 활용, label space mismatch 대응까지 함께 고려해야 한다.

향후 연구의 중요 방향도 논문이 잘 짚고 있다. heterogenous DA, adaptation beyond classification, 그리고 shared label space 가정이 깨지는 현실적 시나리오가 그 핵심이다. 결과적으로 이 논문은 당시까지의 성과를 정리하는 데 그치지 않고, 이후 deep domain adaptation 연구가 어디로 가야 하는지를 비교적 정확하게 제시한 survey라고 볼 수 있다.
