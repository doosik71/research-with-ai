# Toward Supervised Anomaly Detection

- **저자**: Nico Gornitz, Marius Kloft, Konrad Rieck, Ulf Brefeld
- **발표연도**: 2013
- **arXiv**: http://arxiv.org/abs/1401.6424v1

## 1. 논문 개요

이 논문은 anomaly detection을 단순한 비지도학습 문제로만 다루는 기존 관행을 넘어, 소량의 라벨 정보를 활용하면서도 **새롭게 등장하는 이상(anomaly)** 을 놓치지 않는 **semi-supervised anomaly detection** 프레임워크를 제안한다. 저자들은 특히 네트워크 침입 탐지(network intrusion detection) 같은 환경에서는 훈련 시점에 존재하지 않던 공격 유형이 테스트 시점에 나타나는 것이 본질적인 특성이라고 본다. 따라서 일반적인 semi-supervised classification처럼 “라벨이 있는 클래스들을 더 잘 구분하는 문제”로 바꾸어버리면, 알려지지 않은 새로운 이상 패턴을 탐지하는 능력이 오히려 약해질 수 있다고 지적한다.

논문의 핵심 문제의식은 다음과 같다. 순수한 unsupervised anomaly detection은 새로운 공격이나 드문 이상 현상을 다루는 데 자연스럽지만, 실제 운영 환경에서는 성능이 부족할 수 있다. 반대로 supervised 혹은 전통적인 semi-supervised classification은 일부 라벨을 활용해 성능을 끌어올릴 수 있지만, 학습 데이터에 없던 새로운 anomaly class가 테스트 시점에 등장하면 쉽게 실패한다. 이 논문은 이 두 문제를 동시에 해결하고자 한다. 즉, **학습의 철학은 unsupervised anomaly detection에 두되, 라벨 정보는 그 위에 부가적으로 얹는 방식** 을 설계하는 것이 목적이다. 원문 초록과 서론은 바로 이 점을 이 논문의 가장 중요한 출발점으로 제시한다.

문제의 중요성은 매우 실용적이다. 의료영상, 센서 모니터링, 보안, 금융 이상거래 탐지 등 많은 응용에서 이상은 본질적으로 희귀하고, 미래의 이상 유형은 과거 데이터에 모두 포함되어 있지 않다. 저자들은 네트워크 보안에서 “단 하나의 놓친 attack도 시스템을 장악하기에 충분할 수 있다”는 맥락을 강조하며, 이런 환경에서는 단순 분류 성능보다 **미지의 공격을 버틸 수 있는 generalization** 이 더 중요하다고 본다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **semi-supervised anomaly detection은 supervised learning의 확장이 아니라 unsupervised anomaly detection의 확장이어야 한다** 는 것이다. 저자들은 이 차이를 Figure 1, Figure 2, Figure 3의 장난감 실험(toy experiment)으로 직관적으로 보여준다. 일반적인 supervised/semi-supervised classifier는 훈련 시점에 관찰된 클래스 경계를 잘 학습하지만, 테스트 시점에 새로운 anomaly cluster가 나타나면 그것을 기존 정상 영역으로 흡수해 버릴 수 있다. 반면 anomaly detection 관점은 애초에 “정상 데이터의 영역”을 기술하고, 그 바깥을 이상으로 보는 방식이므로 새로운 anomaly cluster가 생겨도 상대적으로 더 견고하다.

이를 위해 저자들은 Support Vector Data Description(SVDD)을 출발점으로 삼는다. SVDD는 정상 데이터를 둘러싸는 hypersphere를 학습하고, 그 구 밖에 있는 점을 anomaly로 간주하는 대표적인 one-class 방법이다. 논문은 이 SVDD에 라벨 정보를 직접 주입해, 양의 라벨(normal)은 hypersphere 안으로, 음의 라벨(anomaly)은 hypersphere 밖으로 밀어내는 방식의 새로운 목적함수를 만든다. 이 방식은 다음과 같은 차별점을 가진다.

첫째, supervised classifier처럼 클래스 경계를 직접 배우는 것이 아니라 정상성(normality)의 영역을 먼저 정의한다. 둘째, anomaly 라벨뿐 아니라 normal 라벨도 활용하므로, 기존의 SVDDneg 같은 방법보다 라벨 정보를 더 폭넓게 쓴다. 셋째, active learning을 결합해 경계 근처의 애매한 점뿐 아니라 **새로운 anomaly cluster가 있을 법한 영역** 도 우선적으로 질의하도록 설계한다.

결국 저자들이 제안하는 SSAD(Semi-Supervised Anomaly Detection)는 “one-class description + partial labels + active querying”의 결합이다. 이 조합이 중요한 이유는, anomaly detection의 본질인 **novelty robustness** 를 해치지 않으면서도 적은 수의 라벨로 성능을 크게 높일 수 있기 때문이다.

## 3. 상세 방법 설명

### 3.1 기본 SVDD 복습

논문은 먼저 일반적인 anomaly score를 다음처럼 정의한다.

$$
f(x) = \|\phi(x) - c\|^2 - R^2
$$

여기서 $\phi(x)$는 feature space로의 매핑, $c$는 hypersphere의 중심, $R$은 반지름이다. $f(x) > 0$이면 구 밖에 있으므로 anomaly, $f(x) < 0$이면 정상으로 본다. 이 점수는 정상 데이터를 둘러싸는 compact한 영역을 학습한다는 점에서, 다중 클래스 분류와는 다른 철학을 가진다.

기본 SVDD는 다음과 같이 radius를 줄이면서, 구 밖에 놓이는 unlabeled sample에 대해서 slack을 부여하는 convex 최적화 문제다.

$$
\min_{R,c,\xi} \; R^2 + \eta_u \sum_{i=1}^{n} \xi_i
$$

subject to

$$
\|\phi(x_i)-c\|^2 \le R^2 + \xi_i, \qquad \xi_i \ge 0
$$

여기서 $\eta_u$는 반지름 최소화와 slack penalty 사이의 trade-off를 조절한다. 저자들은 이 $\eta_u$가 training set에서 outlier 비율에 대한 추정치 역할도 한다고 설명한다.

### 3.2 제안 방법: SSAD 목적함수

이제 논문은 unlabeled 데이터 $x_1,\dots,x_n$에 더해 labeled 데이터 $(x_1^*, y_1^*), \dots, (x_m^*, y_m^*)$를 사용한다. 라벨은 $y^* \in \{+1,-1\}$로 두고, $+1$은 nominal data, $-1$은 anomaly를 뜻한다. 제안하는 제약은 직관적이다.

- $y^*=+1$인 점은 hypersphere 안에 있어야 한다.
- $y^*=-1$인 점은 hypersphere 밖에 있어야 한다.

이를 반영한 직관적 목적은 다음과 같이 쓸 수 있다.

$$
\min_{R,\gamma,c,\xi} \; R^2 - \kappa \gamma + \eta_u \sum_{i=1}^{n} \xi_i + \eta_l \sum_{j=n+1}^{n+m} \xi_j^*
$$

여기서 $\gamma$는 labeled sample에 대한 margin이고, $\kappa$, $\eta_u$, $\eta_l$는 trade-off parameter다. 직관적으로 보면 $R^2$를 줄여 정상 영역을 compact하게 유지하면서, $\gamma$를 키워 labeled example에 대한 분리를 더 명확히 하고, 위반 정도는 slack으로 벌점 준다. 문제는 음의 라벨을 포함하면 이 최적화가 non-convex가 된다는 점이다.

저자들은 이를 unconstrained form으로 바꾸기 위해 slack을 loss function으로 흡수한다. unlabeled data용 slack과 labeled data용 slack은 각각 다음처럼 정의된다.

$$
\xi_i = \ell\big(R^2 - \|\phi(x_i)-c\|^2\big)
$$

$$
\xi_j^* = \ell\big(y_j^*(R^2 - \|\phi(x_j^*)-c\|^2) - \gamma\big)
$$

이렇게 하면 labeled anomaly는 구 밖으로, labeled normal은 구 안으로 배치되도록 loss가 작동한다. 만약 $\ell(t)=\max\{-t,0\}$인 hinge loss를 쓰면 원래 제약식 형태를 회복한다.

또한 representer theorem을 적용해 중심 $c$를 training sample들의 선형결합으로 표현할 수 있다.

$$
c = \sum_{i=1}^{n} \alpha_i \phi(x_i) + \sum_{j=n+1}^{n+m} \alpha_j y_j^* \phi(x_j^*)
$$

이 표현은 kernel trick을 가능하게 하며, 결과적으로 입력 공간에서 직접 구를 다루지 않고 kernel matrix만으로 학습할 수 있게 한다. 논문은 이를 이용해 SSAD의 kernelized unconstrained objective를 제시하고, 그 식을 Equation (5)로 정리한다. 이 식은 $R$, $\gamma$, $\alpha$를 변수로 가지며, unlabeled term과 labeled term이 모두 kernel 표현 안에 들어간다.

### 3.3 손실 함수와 최적화

문제 자체는 non-convex이지만, unconstrained form으로 바꾸면 gradient-based optimization을 쓰기 쉬워진다. 저자들은 비매끄러운 hinge loss 대신 **Huber loss** 를 추천한다. Huber loss는 중심 근방에서는 quadratic, 멀리서는 linear하게 동작하기 때문에 differentiable하면서도 이상치에 대한 강건성을 유지한다. 이 선택 덕분에 conjugate gradient나 Newton류의 off-the-shelf optimizer를 적용할 수 있다. 논문은 Appendix C에서 gradient를 상세히 유도한다.

### 3.4 Convex equivalent formulation

Section 3.2의 또 다른 중요한 기여는, **translation-invariant kernel** 또는 feature space에서 unit norm이 되는 경우(예: RBF kernel)에는 위의 non-convex 문제를 등가의 convex 문제로 바꿀 수 있다는 점이다. 저자들은 Fenchel-Legendre conjugate와 duality를 사용해 이를 보이며, 이 과정에서 one-class SVM이 더 일반적인 density level set estimation 틀의 special case라는 해석도 제시한다. 다만 저자들 스스로도 convex model의 직관적 해석은 normalized kernel일 때 더 자연스럽다고 설명한다. 즉, 이론적 convex reformulation은 강점이지만 모든 커널에서 똑같이 직관적인 것은 아니다.

### 3.5 Active learning 전략

논문은 단순히 SSAD 모델만 제안하지 않고, 제한된 labeling budget에서 어떤 샘플을 물어봐야 하는지도 함께 설계한다. active learning은 다음 두 요소를 결합한다.

첫 번째는 **margin strategy** 다. 즉, 현재 hypersphere 경계에 가장 가까운 점을 질의한다.

$$
x^* = \arg\min_{x \in \{x_1,\dots,x_n\}} |f(x)|
$$

이 전략은 경계 근처의 low-confidence point를 빠르게 정정하는 데 유리하다.

두 번째는 **cluster strategy** 다. 저자들은 $k$-nearest-neighbor graph의 adjacency matrix $A=(a_{ij})$를 만들고, 라벨이 거의 없는 영역이나 기존에 드물게 라벨된 영역을 질의하도록 한다. 이는 decision boundary와 멀리 떨어져 있어도 **새로운 anomaly cluster** 가 있을 수 있는 영역을 탐색하게 해 준다. 논문의 핵심은 이 둘을 합치는 것이다. 최종 질의 규칙은 경계 근접성과 희소 라벨 클러스터 탐색을 가중합으로 결합한다.

$$
x^* = \arg\min_{x_i \in \{x_1,\dots,x_n\}} \left( \delta \frac{|f(x_i)|}{c} + \frac{1-\delta}{2k} \sum_{j=1}^{n+m} (\bar y_j + 1)a_{ij} \right)
$$

여기서 $\delta \in [0,1]$는 두 전략의 균형을 조정한다. 이 전략은 “경계에 가깝지만 동시에 이상 클러스터일 가능성도 높은 점”을 우선적으로 질의하게 만든다. 저자들은 이것이 anomaly detection에서 특히 중요하다고 본다. 단순 margin strategy만 쓰면 경계 근처의 기존 구조만 다듬고, 완전히 새로운 anomaly region을 놓칠 수 있기 때문이다.

## 4. 실험 및 결과

### 4.1 장난감 데이터에서의 학습 패러다임 비교

Section 5에서는 2차원 synthetic data를 사용한 controlled experiment를 수행한다. 훈련/검증 데이터는 두 개의 정상 Gaussian cluster와 하나의 anomaly cluster로 만들고, 테스트 시점에는 여기에 **두 개의 새로운 anomaly cluster** 를 추가한다. 이 설정은 “테스트 시 새로운 이상 분포가 등장한다”는 anomaly detection의 본질을 의도적으로 재현한 것이다. 비교 대상은 supervised SVM, transductive semi-supervised LDS, unsupervised SVDD, LPUE 계열의 SVDDneg, 그리고 제안법 SSAD다. 평가는 false-positive interval $[0, 0.01]$ 에서의 ROC AUC, 즉 $\mathrm{AUC}_{0.01}$ 로 수행한다. 각 반복에서 $\eta_u, \eta_l$은 validation set에서 $[10^{-2}, 10^2]$ 범위로 튜닝했다.

결과는 매우 분명하다. supervised learning paradigm에서 출발한 SVM과 LDS는 novel anomaly cluster를 처리하지 못해 모든 라벨 비율에서 낮은 성능을 보인다. 오히려 라벨을 전혀 쓰지 않는 SVDD보다도 못한 경우가 있다. 반면 unsupervised paradigm 위에 세워진 SVDDneg와 SSAD는 모든 경쟁법을 이긴다. 특히 SSAD는 anomaly 라벨뿐 아니라 normal 라벨도 활용하기 때문에 적은 수의 라벨만으로도 빠르게 성능이 포화된다. 논문은 약 15% 수준의 labeled data만으로도 SSAD가 거의 최적 수준에 도달한다고 보고한다. 또한 Figure 7은 fully supervised, semi-supervised, unsupervised 세 경우의 contour를 비교해, 적은 라벨만으로도 semi-supervised 해가 거의 완전한 분리에 도달할 수 있음을 시각적으로 보여준다.

실행 시간 측면에서는 SVM이 가장 빠르지만, 이는 unlabeled data를 무시하기 때문이다. SSAD, SVDD, SVDDneg는 비슷한 수준의 비용을 보이며, LDS는 transductive 특성 때문에 가장 느리다. 저자들은 이 결과를 바탕으로 anomaly detection 시나리오에서는 supervised paradigm에서 출발한 방법이 부적절하고, unsupervised paradigm을 유지하는 semi-supervised 방법이 가장 적절하다고 결론 내린다.

### 4.2 실제 네트워크 침입 탐지 실험

실제 응용은 HTTP 네트워크 트래픽에 대한 intrusion detection이다. 정상 데이터는 Fraunhofer Institute FIRST에서 10일간 수집한 145,069개의 정상 연결로 이루어져 있으며, 평균 길이는 489 bytes다. 각 payload는 3-gram bag-of-features 방식으로 임베딩된다. 즉, 가능한 모든 길이 3의 byte string을 특징으로 보고, payload 안에 해당 3-gram이 있으면 1, 아니면 0인 sparse vector로 바꾼다. 이 표현은 차원 수가 $256^3$으로 매우 크지만, 실제 payload는 sparse하기 때문에 효율적 처리 가능하다고 설명한다. 악성 데이터는 Metasploit을 사용해 만든 27개 실제 attack class이며, buffer overflow, code injection, HTTP tunnel, cross-site scripting 등이 포함된다. 또한 cloaked pool이라는 난도 높은 조건을 만들기 위해, 공격 payload에 benign한 HTTP header를 덧붙여 feature space에서 정상처럼 보이게 만드는 obfuscation도 적용한다.

탐지 성능 실험에서는 두 가지 시나리오를 비교한다. 하나는 normal vs. malicious, 다른 하나는 normal vs. cloaked다. 매 반복마다 정상 966개와 공격 34개를 training set으로 사용하고, holdout/test set은 정상 795개와 공격 27개로 구성한다. 특히 동일 attack class가 train과 test에 동시에 들어가지 않도록 하여, truly novel attack generalization을 평가한다. 평가 지표는 again $\mathrm{AUC}_{0.01}$ 이다.

normal vs. malicious에서는 사실상 모든 방법이 잘 동작한다. 이는 원본 악성 트래픽이 3-gram 표현만으로도 충분히 구분되기 때문이다. 그러나 더 현실적인 normal vs. cloaked에서는 차이가 크게 난다. unsupervised SVDD는 약 70% 수준까지 떨어지고, SVDDneg도 소폭 개선만 보인다. 반면 SSAD는 모든 labeled data를 활용하기 때문에 훨씬 큰 폭의 향상을 보이며, 5% 정도의 labeled data만으로도 최고의 baseline을 쉽게 넘어선다. 무작위 라벨링으로 15%의 데이터를 라벨링하면 거의 완벽에 가까운 분리가 가능하다고 저자들은 보고한다. Figure 9의 설명에서는 cloaked data에서 SSAD가 baseline보다 **최대 30% 높은 accuracy** 를 보인다고 정리한다.

active learning을 적용하면 효과는 더 크다. cloaked setting에서 SSAD는 active learning으로 **약 3%의 labeled data만으로 거의 완벽한 분리** 를 달성하는 반면, random labeling은 같은 수준에 도달하려면 약 25%가 필요하다고 논문은 보고한다. 또한 Figure 10은 combined strategy가 margin-only strategy나 random sampling보다 novel attack을 더 빨리 찾아낸다는 것을 보여준다. 즉, proposed querying rule은 단순히 모델을 조금 더 다듬는 수준이 아니라, 실제로 새로운 anomaly class를 더 빨리 발굴하는 역할을 한다.

### 4.3 Threshold adaptation

Section 6.2에서는 active learning이 SSAD뿐 아니라 기존 anomaly detector의 threshold calibration에도 유용하다는 점을 보인다. 저자들은 vanilla SVDD에 대해, 질의된 labeled sample들의 거리 정보를 이용해 새로운 threshold $\hat R$를 추정한다. 라벨된 positive와 negative의 존재 여부에 따라 $\hat R$를 다르게 정하는 piecewise rule을 제안하며, 실제로 active learning 기반 threshold는 random sampling보다 훨씬 더 합리적인 ROC operating point를 찾아낸다. 원문은 5%의 labeled data만 있어도 active learning이 reasonable한 radius를 찾는 반면, random sampling과 vanilla SVDD는 false-positive rate가 각각 0.5와 1 수준으로 사실상 실패한다고 보고한다. 이는 active learning이 “모델 자체의 학습”뿐 아니라 “운용 임계값 설정”에도 실질적으로 도움이 된다는 의미다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 anomaly detection의 문제 설정을 매우 정확하게 다시 규정했다는 점이다. 많은 연구가 semi-supervised라는 이름 아래 사실상 classification 문제를 풀고 있었는데, 이 논문은 anomaly detection에서는 훈련과 테스트의 anomaly distribution이 다를 수 있다는 점을 정면으로 문제 삼는다. 그리고 그에 맞는 수학적 모델을 실제로 제안한다. 다시 말해, 단순한 성능 개선이 아니라 **문제 정의 자체를 올바르게 세운 뒤 그에 맞는 학습 원리를 설계한 논문** 이라는 점이 중요하다.

두 번째 강점은 방법론이 깔끔하다는 것이다. SVDD라는 잘 알려진 one-class framework를 기반으로 normal/anomaly 라벨을 모두 반영하는 목적함수를 만들고, representer theorem으로 kernel화하며, Huber loss로 매끄럽게 최적화하고, 특정 커널 조건에서는 convex equivalent까지 보인다. 즉, 아이디어가 직관적일 뿐 아니라 수학적으로도 정리되어 있다. 특히 “non-convex primal formulation”과 “mild assumptions 하의 convex equivalent”를 함께 제시한 점은 이론적 완성도를 높인다.

세 번째 강점은 active learning이 anomaly detection의 성격에 맞게 설계되어 있다는 점이다. 경계 근처의 불확실성만 보는 것이 아니라, 드물게 라벨된 영역이나 새로운 anomaly cluster 후보를 탐색하도록 만든 것은 실제 보안 운영에 매우 설득력이 있다. 원문 결과에서도 이것이 단순 random sampling보다 훨씬 적은 라벨로 더 높은 성능을 얻는 것으로 확인된다.

반면 한계도 분명하다. 첫째, 실험의 중심 응용이 네트워크 intrusion detection에 맞춰져 있어 다른 도메인으로의 일반화는 논문이 직접 충분히 보여주지 않는다. 의료, 제조, 이미지 이상 탐지 등 다른 형태의 feature geometry에서도 동일한 장점이 유지되는지는 이 논문만으로는 확정할 수 없다. 둘째, convex equivalent의 해석은 unit norm feature space 또는 translation-invariant kernel 같은 조건에서 가장 자연스럽다. 따라서 모든 설정에서 동일한 이론적 이점을 직관적으로 설명하기는 어렵다. 셋째, active learning은 여전히 domain expert의 라벨링을 필요로 하므로, labeling cost를 줄여주기는 하지만 완전히 없애지는 못한다. 넷째, 비교 대상은 당시 기준으로 타당하지만 현대의 representation learning 기반 anomaly detection이나 deep metric learning 계열과 비교한 것은 아니므로, 오늘날 기준의 절대적 최강 성능을 말하는 논문으로 읽으면 안 된다. 이 마지막 지점은 시대적 맥락에 따른 한계다.

또 하나의 비판적 해석은, SSAD가 효과적인 이유가 단순히 “라벨을 조금 썼기 때문”이 아니라 “정상성의 형상을 먼저 배우고 라벨은 그 형상을 조정하는 데 사용했기 때문”이라는 데 있다. 즉, 이 논문의 메시지는 라벨을 더 많이 모으자는 것이 아니라, **어떤 inductive bias 위에 라벨을 얹을 것인가가 더 중요하다** 는 점이다. 이 통찰은 이후의 weakly-supervised anomaly detection 연구를 이해할 때도 매우 중요하다.

## 6. 결론

이 논문은 anomaly detection을 semi-supervised로 확장할 때, supervised classification의 관점으로 가져가면 본질을 잃는다는 점을 명확히 보여준다. 이를 대신해 저자들은 SVDD 기반의 **SSAD** 를 제안했고, 이 방법은 정상성의 hypersphere를 유지하면서도 labeled normal/anomaly 샘플을 함께 사용해 더 강한 탐지 경계를 만든다. 또한 Huber loss를 이용한 gradient-based optimization, 특정 조건하의 convex equivalent formulation, 그리고 anomaly-specific active learning strategy까지 포함해 하나의 완성된 프레임워크를 제시했다.

실험적으로도 메시지는 일관된다. novelty가 없는 쉬운 조건에서는 큰 차이가 없지만, 테스트 시 새로운 anomaly cluster가 등장하거나 공격이 cloaking으로 위장된 더 현실적인 조건에서는 SSAD가 기존 supervised/semi-supervised baseline보다 훨씬 견고하다. 특히 소량의 라벨만으로 성능이 크게 향상되고, active learning으로 그 비용을 더 줄일 수 있다는 점은 실제 보안 시스템 운용에 매우 실용적이다.

요약하면 이 논문의 주요 기여는 다음 세 가지로 정리할 수 있다. 첫째, semi-supervised anomaly detection의 올바른 학습 패러다임을 제시했다. 둘째, SVDD 기반 SSAD라는 구체적 알고리즘과 그 이론적 정당화를 제시했다. 셋째, active learning을 통해 적은 라벨로도 실제적인 성능 향상을 얻을 수 있음을 보였다. 따라서 이 연구는 이후의 weakly-supervised anomaly detection, PU-style anomaly learning, 그리고 사람이 개입하는 human-in-the-loop anomaly detection의 중요한 이론적 선행 연구로 볼 수 있다.
