# Deep Anomaly Detection with Deviation Networks

- **저자**: Guansong Pang, Chunhua Shen, Anton van den Hengel (The University of Adelaide)
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1911.08623

## 1. 논문 개요

이 논문은 anomaly detection에서 deep learning을 어떻게 더 직접적이고 데이터 효율적으로 사용할 수 있는지를 다룬다. 기존의 deep anomaly detection 방법은 대체로 먼저 representation을 학습한 뒤, 그 representation 위에서 reconstruction error나 distance 같은 별도의 anomaly measure를 적용하는 2단계 구조를 사용했다. 저자들은 이런 방식이 anomaly score 자체를 직접 최적화하지 못하므로, 학습 데이터 사용이 비효율적이고 최종 anomaly score의 품질도 최적이 아닐 수 있다고 지적한다.

이 논문이 다루는 핵심 연구 문제는 다음과 같다. 대규모 labeled anomaly를 얻기 어려운 현실적인 환경에서, 매우 적은 수의 labeled anomaly만을 활용하여 anomaly score를 end-to-end로 직접 학습할 수 있는가 하는 점이다. 이 문제는 실제 보안, 금융, 의료 같은 분야에서 매우 중요하다. 예를 들어 침입 탐지, 사기 거래 탐지, 질병 탐지에서는 정상 데이터는 많지만 명시적으로 라벨링된 이상 사례는 극히 적은 경우가 흔하다. 또한 anomaly는 서로 매우 이질적일 수 있어서, 일반적인 supervised classification처럼 anomaly class 내부의 유사성을 가정하기도 어렵다.

저자들은 이를 해결하기 위해 anomaly score를 직접 출력하는 neural framework를 제안하고, 이를 DevNet(Deviation Networks)으로 구체화한다. DevNet은 소수의 labeled anomalies와 Gaussian prior를 이용해, 정상 객체의 score 근처에는 normal이 모이도록 하고 anomaly는 그 분포의 upper tail에서 통계적으로 유의미하게 벗어나도록 학습한다. 논문에 따르면 이 방식은 representation을 우회적으로 학습하는 기존 deep anomaly detection 방법보다 더 높은 정확도와 더 좋은 data efficiency를 보인다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 “좋은 representation을 먼저 만든 다음 anomaly score를 계산하자”가 아니라, “anomaly score 자체를 신경망의 직접적인 학습 목표로 삼자”는 데 있다. 즉 입력 $x$가 들어왔을 때 중간 표현을 거쳐 최종적으로 scalar anomaly score를 출력하도록 하고, 그 score가 정상 데이터에 대해서는 어떤 기준점 주변에 모이고 anomaly에 대해서는 그 기준점보다 충분히 크게 떨어지도록 loss를 설계한다.

이 아이디어에서 중요한 부분은 reference score의 도입이다. 저자들은 정상 데이터의 anomaly score 평균을 기준점으로 삼고 싶어 하지만, 실제로는 labeled normal data가 없다. 이를 해결하기 위해 정상 데이터 score의 기준 분포를 Gaussian prior로 대체한다. 구체적으로 $\mathcal{N}(0,1)$에서 샘플링한 값들의 평균과 표준편차를 사용해 normal score의 기준을 만든다. 이후 각 입력의 anomaly score가 이 기준에 대해 얼마나 벗어났는지를 Z-score 형태로 계산하고, anomaly는 크게 양의 방향으로 벗어나고 normal은 기준점 근처에 머물도록 deviation loss를 정의한다.

기존 접근과의 차별점은 세 가지로 정리할 수 있다. 첫째, anomaly score를 직접 최적화한다는 점이다. 기존 autoencoder, GAN, deep SVDD, REPEN류 방법은 feature learning이 중심이고 scoring은 그 결과에 의존하는 간접 목표였다. 둘째, 아주 적은 labeled anomaly를 prior knowledge로 적극 활용한다는 점이다. 셋째, anomaly들이 서로 비슷하지 않아도 된다는 점이다. 이 방법은 anomaly끼리 가까워지도록 강제하지 않고, 정상 score 분포에서 멀어지기만 하면 되므로 서로 다른 종류의 anomaly도 다룰 수 있게 설계되어 있다.

## 3. 상세 방법 설명

### 전체 파이프라인

논문이 제안하는 framework는 세 모듈로 구성된다.

첫째, anomaly scoring network $\phi$가 입력 $x$를 받아 scalar anomaly score를 출력한다.
둘째, reference score generator가 normal object들의 anomaly score 평균에 해당하는 기준값 $\mu_{\mathcal{R}}$와 그에 대응하는 표준편차 $\sigma_{\mathcal{R}}$를 생성한다.
셋째, deviation loss $L$이 $\phi(x)$, $\mu_{\mathcal{R}}$, $\sigma_{\mathcal{R}}$를 사용해 학습을 진행한다.

논문의 목적은 anomaly에 대해서는 높은 score를, normal에 대해서는 기준점 주변 score를 주는 scoring function $\phi:\mathcal{X}\mapsto \mathbb{R}$를 학습하는 것이다. 문제 설정은 다음과 같다. 전체 학습 데이터는 unlabeled data $\mathcal{U}$와 매우 적은 수의 labeled anomalies $\mathcal{K}$로 이루어진다. 즉,

$$
\mathcal{X} = \mathcal{U} \cup \mathcal{K}
$$

이며 $\mathcal{K}$의 크기 $K$는 $\mathcal{U}$의 크기 $N$에 비해 매우 작다. 목표는 anomaly인 $x_i$와 normal인 $x_j$에 대해

$$
\phi(x_i) > \phi(x_j)
$$

가 되도록 학습하는 것이다.

### anomaly scoring network

저자들은 anomaly score learner를 representation learner와 score learner의 결합으로 정의한다. 먼저 feature learner $\psi(\cdot;\Theta_t)$가 입력을 중간 표현 공간 $\mathcal{Q}$로 보낸다.

$$
q = \psi(x; \Theta_t)
$$

여기서 $q \in \mathcal{Q}$이다. 이 feature learner는 데이터 종류에 따라 MLP, CNN, RNN 등으로 구현할 수 있다고 설명한다. 본 논문의 실험에서는 unordered multidimensional data를 다루므로 MLP를 사용했다.

그 다음 score learner $\eta(\cdot;\Theta_s)$가 중간 표현 $q$를 scalar anomaly score로 바꾼다. 이는 출력층의 단일 linear unit으로 정의된다.

$$
\eta(q; \Theta_s) = \sum_{i=1}^{M} w_i^o q_i + w_{M+1}^o
$$

즉 최종 anomaly scoring network는

$$
\phi(x; \Theta) = \eta(\psi(x; \Theta_t); \Theta_s)
$$

가 된다. 중요한 점은, 이 구조가 입력에서 최종 scalar score까지 한 번에 연결되어 있어 end-to-end 학습이 가능하다는 것이다.

### Gaussian prior-based reference score

정상 데이터의 score 기준점으로 $\mu_{\mathcal{R}}$를 사용하려면 원래는 normal object들의 score 평균이 필요하다. 하지만 labeled normal이 없기 때문에, 논문은 prior-driven approach를 채택한다. 구체적으로 Gaussian prior에서 $l$개의 score를 샘플링한다.

$$
r_1, r_2, \dots, r_l \sim \mathcal{N}(\mu, \sigma^2)
$$

그리고 그 평균을 reference score로 둔다.

$$
\mu_{\mathcal{R}} = \frac{1}{l}\sum_{i=1}^{l} r_i
$$

논문에서는 $\mu = 0$, $\sigma = 1$, $l = 5000$을 사용했다. 저자들은 $\sigma$가 너무 크지만 않으면 성능이 크게 민감하지 않았고, $l$도 충분히 크면 central limit theorem 때문에 큰 차이가 없었다고 보고한다. 이 설정 덕분에 reference score를 매번 안정적으로 만들 수 있고, anomaly score를 Z-score 해석이 가능한 형태로 만들 수 있다.

### Z-Score-based deviation loss

핵심은 deviation을 Z-score로 정의하는 것이다.

$$
dev(x) = \frac{\phi(x; \Theta) - \mu_{\mathcal{R}}}{\sigma_{\mathcal{R}}}
$$

여기서 $\sigma_{\mathcal{R}}$는 prior에서 샘플링한 $r_1,\dots,r_l$의 표준편차다. 이 값은 현재 입력의 anomaly score가 normal score 기준 분포에서 얼마나 떨어져 있는지를 표준편차 단위로 나타낸다.

이 deviation을 사용해 loss를 다음처럼 정의한다.

$$
L(\phi(x;\Theta), \mu_{\mathcal{R}}, \sigma_{\mathcal{R}}) = (1-y)|dev(x)| + y \max(0, a - dev(x))
$$

여기서 $y=1$이면 anomaly, $y=0$이면 normal이다. 또한 $a$는 Z-score margin 역할을 하는 confidence interval parameter다.

이 식의 의미를 쉽게 풀면 다음과 같다.

정상 샘플($y=0$)에 대해서는 loss가 $|dev(x)|$가 된다. 따라서 정상 데이터의 score는 기준점 $\mu_{\mathcal{R}}$ 근처에 오도록 학습된다.

이상 샘플($y=1$)에 대해서는 loss가 $\max(0, a-dev(x))$가 된다. 즉 anomaly의 deviation이 $a$보다 작으면 벌점을 받고, $a$ 이상이면 loss가 0이 된다. 다시 말해 anomaly score는 기준점보다 적어도 $a$ 표준편차 이상 큰 값을 갖도록 강제된다.

논문에서는 $a=5$를 사용했다. 저자들은 이를 매우 높은 significance level에 해당한다고 설명한다. 또한 anomaly인데 $dev(x)$가 음수이면 loss가 특히 커지므로, anomaly는 반드시 upper tail 방향으로 크게 밀어내는 효과가 있다.

### labeled normal이 없을 때의 학습 전략

이 방법의 현실적 난점은 labeled normal data가 없다는 점이다. 저자들은 이를 단순한 전략으로 처리한다. unlabeled set $\mathcal{U}$의 모든 데이터를 normal처럼 취급하는 것이다. 물론 여기에는 실제 anomaly가 일부 섞여 있을 수 있다. 그러나 anomaly contamination이 있더라도 anomaly는 희귀하므로 SGD 기반 최적화에서 영향이 제한적이라고 주장한다. 실제 실험에서도 이 단순 전략이 잘 작동했다고 보고한다.

### 학습 알고리즘

DevNet의 학습은 stochastic gradient descent 방식으로 진행된다. 알고리즘의 핵심 절차는 다음과 같다.

각 epoch와 batch마다, mini-batch $\mathcal{B}$를 stratified random sampling으로 구성하는데, 절반은 labeled anomalies $\mathcal{K}$에서, 나머지 절반은 unlabeled data $\mathcal{U}$에서 뽑는다. 동시에 Gaussian prior $\mathcal{N}(\mu,\sigma^2)$에서 $l$개의 score를 샘플링해 $\mu_{\mathcal{R}}$와 $\sigma_{\mathcal{R}}$를 계산한다. 이후 mini-batch의 각 샘플에 대해 anomaly score를 forward propagation으로 계산하고 deviation loss를 평균내어 gradient descent를 수행한다.

논문은 DevNet의 핵심 연산이 결국 network의 forward/backward propagation이므로, MLP 사용 시 학습 시간 복잡도는 hidden layer 크기에 선형적으로 비례한다고 분석한다. 따라서 데이터 크기와 차원이 증가해도 선형적으로 확장된다고 주장한다.

### anomaly score의 해석 가능성

이 논문의 흥미로운 장점 중 하나는 anomaly score의 interpretability다. 많은 anomaly detector는 score를 주더라도 그 수치가 어떤 확률적 의미를 갖는지 해석하기 어렵다. 반면 DevNet은 Gaussian prior와 Z-score 기반 loss를 사용하므로 score를 통계적으로 해석할 수 있다.

논문은 proposition 형태로, $\phi(x)$가 $\mu \pm z_p \sigma$ 범위 밖에 있을 확률이 $2(1-p)$라고 설명한다. 특히 upper tail만 보면 $\mu + z_p \sigma$보다 클 확률은 $(1-p)$다. 예를 들어 $p=0.95$이면 $z_{0.95}=1.96$이므로, anomaly score가 1.96을 넘는 객체는 normal generating mechanism으로부터 나올 확률이 0.05 정도라고 볼 수 있다. 따라서 사용자는 원하는 confidence level에 따라 threshold를 정하기 쉽다.

## 4. 실험 및 결과

### 데이터셋과 설정

실험은 9개의 공개 real-world 데이터셋에서 수행되었다. 도메인은 기부 프로젝트, 소득, 신용카드 fraud, 얼굴 attribute, 네트워크 backdoor attack, malicious URL, 은행 마케팅, 뉴스 텍스트, 갑상선 질환 등으로 다양하다. 데이터 규모도 상당히 크다. 예를 들어 donors는 약 61만 개, fraud는 약 28만 개, URL은 약 8.9만 개지만 차원 수가 323만 개, news20은 약 1만 개지만 차원 수가 135만 개에 달한다.

실험 시나리오는 현실적인 semi-supervised anomaly detection을 모사한다. 각 데이터셋을 train 80%, test 20%로 나누고, unlabeled training set $\mathcal{U}$에는 anomaly contamination을 2%로 맞춘다. 그리고 anomaly class에서 30개만 뽑아 labeled anomaly set $\mathcal{K}$로 사용한다. 이 30개는 전체 training data의 0.005%에서 1%, 전체 anomaly의 0.08%에서 6% 정도에 해당한다. 즉 정말 적은 수의 anomaly label만 사용하는 셈이다.

### 비교 방법

비교 대상은 네 가지다. REPEN, adaptive Deep SVDD(DSVDD), prototypical networks 기반 FSNet, 그리고 비지도 방식인 iForest다. 저자들은 DSVDD를 공정 비교를 위해 수정해 labeled anomalies를 사용할 수 있게 만들었다고 설명한다. FSNet은 원래 few-shot classification용 방법이지만, anomaly detection에 맞게 unlabeled와 anomaly를 episode 구성에 사용하도록 변형했다.

평가 지표는 AUC-ROC와 AUC-PR를 모두 사용했다. 저자들은 anomaly detection처럼 positive class가 매우 희귀한 문제에서는 AUC-PR가 특히 중요하다고 강조한다. 모든 결과는 10번 독립 실행 평균이며, paired Wilcoxon signed-rank test로 통계적 유의성도 확인했다.

### 주요 정량 결과

논문의 핵심 결과는 Table 1에 제시되어 있다. 평균 성능을 보면 DevNet은 AUC-ROC에서 $0.916 \pm 0.004$, AUC-PR에서 $0.574 \pm 0.008$을 기록했다. 이는 REPEN의 AUC-ROC $0.838$, AUC-PR $0.263$, DSVDD의 AUC-ROC $0.888$, AUC-PR $0.473$, FSNet의 AUC-ROC $0.750$, AUC-PR $0.270$, iForest의 AUC-ROC $0.708$, AUC-PR $0.140$보다 전반적으로 우수하다.

저자들이 정리한 평균 개선폭은 다음과 같다. DevNet은 AUC-ROC 기준으로 REPEN 대비 9%, DSVDD 대비 3%, FSNet 대비 22%, iForest 대비 29% 향상되었다. AUC-PR 기준으로는 REPEN 대비 118%, DSVDD 대비 21%, FSNet 대비 113%, iForest 대비 309% 향상되었다. 특히 AUC-PR 향상폭이 매우 크다는 점은 실제 anomaly retrieval 품질이 좋아졌다는 의미로 해석할 수 있다.

개별 데이터셋 수준에서도 DevNet은 매우 강한 성능을 보였다. AUC-ROC에서는 9개 중 8개 데이터셋에서 최고 또는 거의 최고 수준이었고, AUC-PR에서는 9개 모두에서 최고 성능을 기록했다. 예를 들어 donors에서는 AUC-ROC와 AUC-PR 모두 1.000에 도달했고, backdoor에서는 AUC-PR 0.883, URL에서는 AUC-PR 0.681, news20에서는 AUC-PR 0.653으로 다른 방법들을 크게 앞섰다.

### data efficiency 실험

저자들은 labeled anomalies의 수를 5개에서 120개까지 바꾸며 data efficiency를 분석했다. 결과적으로 모든 deep 방법은 보통 label이 늘수록 성능이 좋아졌지만, 경쟁 방법은 어떤 경우 오히려 성능이 떨어지기도 했다. 이는 anomaly가 서로 이질적이어서 새로 추가된 labeled anomaly가 기존 anomaly와 다른 방향의 정보를 줄 수 있기 때문이라고 해석한다.

반면 DevNet은 전반적으로 더 안정적이었고, 가장 data-efficient한 방법으로 보고되었다. 논문은 DevNet이 경쟁 방법과 비슷하거나 더 좋은 성능을 내기 위해 75%에서 88% 적은 labeled data만 필요했다고 주장한다. 예를 들어 donors에서는 최고 경쟁자인 FSNet과 비슷한 성능을 위해 83% 적은 label이 필요했고, news20과 thyroid에서는 각각 88%, 75% 적은 label로 최고 경쟁자보다 더 좋은 성능을 냈다고 설명한다.

또한 iForest와 비교하면, labeled anomaly가 5개 또는 15개처럼 극히 적어도 prior knowledge-driven deep methods, 특히 DevNet과 DSVDD의 성능 향상이 매우 컸다. 논문은 어떤 경우 DevNet과 DSVDD가 iForest 대비 평균 400% 이상의 향상을 보였다고 언급한다.

### anomaly contamination에 대한 robustness

또 다른 중요한 실험은 unlabeled data에 anomaly contamination이 많아질 때의 robustness다. contamination 비율을 0%에서 20%까지 늘리고 labeled anomaly는 30개로 고정했다. 모든 deep detector는 contamination이 올라갈수록 성능이 하락했다. 이는 unlabeled를 normal로 가정했기 때문에, contamination이 커질수록 mini-batch에서 anomaly를 normal처럼 잘못 샘플링할 가능성이 높아지기 때문이다.

그럼에도 DevNet은 오염 수준 전반에서 다른 deep 방법보다 일관되게 더 높은 AUC-PR를 기록했다. 논문은 contamination이 다양한 조건에서 DevNet이 REPEN보다 평균 200%, DSVDD보다 28%, FSNet보다 336% 더 나은 AUC-PR를 보였다고 보고한다. 또한 iForest와 비교해도 DevNet과 DSVDD는 평균적으로 각각 800%, 600% 이상의 AUC-PR 향상을 보였다고 주장한다. 이는 소수의 labeled anomalies가 noisy unlabeled training 환경에서도 강한 prior knowledge로 작동한다는 논문의 주장을 뒷받침한다.

### ablation study

저자들은 DevNet의 핵심 구성 요소가 실제로 필요한지 확인하기 위해 세 가지 변형을 비교했다.

DevNet-Rep는 output layer를 제거하고 deviation loss로 representation만 학습한다.
DevNet-Linear는 hidden layer 없이 입력에서 anomaly score로 바로 linear mapping한다.
DevNet-3HL은 hidden layer를 세 층으로 깊게 만든 버전이다.

결과를 보면 기본 DevNet(Def)이 평균 AUC-ROC 0.916, AUC-PR 0.574로 가장 좋다. Rep는 각각 0.899, 0.550으로 약간 낮고, Linear는 0.865, 0.442로 더 크게 떨어진다. 3HL은 0.853, 0.520으로 깊은 네트워크가 항상 좋은 것은 아님을 보여준다.

이 결과는 세 가지를 시사한다. 첫째, representation만 학습하는 것보다 score 자체를 end-to-end로 학습하는 것이 유리하다. 둘째, non-linear feature learning은 필요하다. 단순 linear mapping은 확실히 성능이 낮다. 셋째, labeled anomalies가 매우 적기 때문에 모델을 너무 깊게 만들면 오히려 학습이 어렵다. 즉 이 문제에서는 “깊을수록 좋다”가 성립하지 않는다.

### scalability

저자들은 synthetic data를 이용해 데이터 크기와 차원 수를 증가시키며 runtime을 평가했다. 결론은 DevNet의 runtime이 데이터 수와 차원 수에 대해 선형적으로 증가한다는 것이다. 이는 앞서 제시한 complexity analysis와 일치한다. 또한 대규모 데이터에서는 REPEN, FSNet, iForest보다 10배에서 20배 빠르다고 주장한다. 다만 고차원 데이터에서는 bottom layer의 projection 비용이 지배적이어서 FSNet이 약간 더 빠를 수 있다고 설명한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 anomaly detection의 목표 자체에 더 직접적으로 맞춘 학습 objective를 제안했다는 점이다. 기존 deep anomaly detection이 “representation을 잘 만들면 결과적으로 anomaly detection도 잘 되겠지”라는 간접 전략에 머물렀다면, 이 논문은 anomaly score를 바로 학습 대상으로 삼는다. 문제 정의와 손실 설계가 매우 일관적이며, 소수의 labeled anomaly를 활용해야 하는 현실적 setting과도 잘 맞는다.

둘째 강점은 deviation loss의 설계가 단순하면서도 목적에 부합한다는 점이다. normal score는 기준점 주변으로 모으고 anomaly는 upper tail 쪽으로 일정 margin 이상 밀어내는 구조는 anomaly detection의 ranking 목적과 자연스럽게 연결된다. 특히 anomaly끼리 유사할 필요가 없다는 점에서 heterogeneous anomaly를 다루기 쉽다.

셋째 강점은 interpretability다. 많은 anomaly detector는 score가 왜 큰지, 어느 threshold를 써야 하는지 설명하기 어렵다. 이 논문은 Gaussian prior와 Z-score 해석을 통해 threshold를 통계적 confidence와 연결하려고 한다. 실제 현업 적용에서 이 부분은 꽤 실용적일 수 있다.

넷째 강점은 empirical validation이 비교적 충실하다는 점이다. 데이터셋이 9개로 다양하고, AUC-ROC와 AUC-PR를 모두 보고하며, data efficiency, contamination robustness, ablation, scalability까지 다룬다. 특히 AUC-PR 개선폭을 강조한 것은 anomaly detection의 class imbalance 특성을 잘 이해한 평가 방식으로 보인다.

반면 한계도 분명하다. 첫째, Gaussian prior 가정이 얼마나 일반적으로 타당한지는 논문이 충분히 이론적으로 증명하지 않는다. 저자들은 기존 연구에서 anomaly score가 Gaussian에 잘 맞는 경우가 많다고 인용하지만, 이 논문에서 학습되는 score가 실제로 그런 분포를 따르는지에 대한 강한 보장은 없다. 따라서 interpretability 주장은 설계상 장점이긴 하지만, 실제 모든 데이터셋에서 엄밀히 성립한다고 보기는 어렵다.

둘째, labeled normal 없이 unlabeled를 전부 normal처럼 취급하는 전략은 실용적이지만, contamination이 더 높거나 anomaly가 구조적으로 normal과 많이 섞인 경우에는 취약할 수 있다. 논문은 20% contamination까지 실험했지만, 그 이상 또는 anomaly prevalence가 높은 환경에서는 어떻게 될지 명확하지 않다.

셋째, 본 논문 실험은 결국 unordered multidimensional data에 대해 MLP를 사용한 설정에 집중되어 있다. 저자들은 image와 sequence data로 확장 중이라고 결론에서 언급하지만, 본문 실험만 놓고 보면 CNN이나 RNN 기반 DevNet의 효과는 아직 검증되지 않았다. 따라서 제안의 일반성은 아이디어 차원에서는 넓지만, 실험적 근거는 탭уляр/벡터 데이터에 더 치우쳐 있다.

넷째, 비교 대상 중 일부는 anomaly detection에 완전히 최적화된 구조라기보다 문제를 변형해서 적용한 방식이다. 예를 들어 FSNet은 원래 few-shot classification용이며, DSVDD도 공정 비교를 위해 수정되었다. 따라서 “모든 종류의 관련 방법을 충분히 대표하는가”에 대해서는 다소 조심스럽게 볼 필요가 있다.

다섯째, anomaly score의 해석 가능성을 강조하지만, 실제로 calibration quality를 정량 평가하는 별도 실험은 없다. 예를 들어 특정 score threshold가 정말 제시된 confidence와 맞는지를 empirical calibration 관점에서 검증하지는 않는다.

종합하면, 이 논문은 방법론적 아이디어가 명확하고 실제 성능 개선도 강력하게 보여 주지만, prior 가정의 보편성, contamination 가정, 데이터 타입 확장성, score calibration 검증 등은 앞으로 더 보완될 여지가 있다.

## 6. 결론

이 논문은 소수의 labeled anomalies를 활용해 anomaly score를 end-to-end로 직접 학습하는 새로운 framework를 제안하고, 이를 DevNet으로 구현했다. DevNet의 핵심은 Gaussian prior 기반 reference score와 Z-score 기반 deviation loss를 결합해, normal object는 기준 score 근처에 모으고 anomaly는 upper tail에서 통계적으로 유의미하게 떨어지도록 학습하는 것이다. 이 구조는 기존 representation 중심 방법과 달리 anomaly score 자체를 직접 최적화하므로, 더 높은 data efficiency와 더 좋은 anomaly ranking 성능을 제공한다.

실험적으로도 DevNet은 9개 real-world 데이터셋에서 AUC-ROC와 특히 AUC-PR 측면에서 강한 성능을 보였고, 적은 수의 labeled anomaly만으로도 경쟁 방법을 능가했다. 또한 contamination이 존재하는 noisy unlabeled data 환경에서도 강한 robustness를 보였다. ablation study는 direct score learning, deviation loss, non-linear feature learning이 모두 중요하다는 점을 뒷받침한다.

실제 적용 측면에서 이 연구는 매우 가치가 있다. 많은 anomaly detection 시스템은 충분한 anomaly label을 확보하기 어렵고, anomaly 유형도 계속 바뀐다. DevNet은 이런 조건에서 소수의 확정된 anomaly 사례를 prior knowledge로 사용해 더 현실적인 탐지기를 학습하는 방향을 제시한다. 향후 image, sequence data로의 확장과 reference score를 더 정교하게 만드는 hybrid approach가 더해진다면, 보안, 금융, 의료, 제조 이상 탐지 등 다양한 분야에서 실질적인 영향력을 가질 가능성이 높다.
