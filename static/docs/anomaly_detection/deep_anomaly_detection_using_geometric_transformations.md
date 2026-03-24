# Deep Anomaly Detection Using Geometric Transformations

- **저자**: Izhak Golan, Ran El-Yaniv
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1805.10917

## 1. 논문 개요

이 논문은 이미지 anomaly detection, 보다 정확히는 **오직 정상 클래스의 샘플만 주어진 one-class / pure anomaly detection 문제**를 다룬다. 예를 들어 학습 데이터가 전부 dog 이미지일 때, 테스트 시 dog가 아닌 이미지들을 out-of-distribution 또는 anomaly로 감지하는 것이 목표다. 저자들은 이 문제를 해결하기 위해 reconstruction 기반 접근을 버리고, 정상 이미지에 다양한 geometric transformation을 적용한 뒤, 어떤 변환이 적용되었는지를 맞히는 **self-labeled multi-class classification 문제**로 바꾸는 새로운 방법을 제안한다. 이 접근은 결과적으로 정상 클래스의 구조를 잘 반영하는 특징을 학습하게 만들고, 테스트 이미지가 정상인지 아닌지를 classifier의 softmax 반응 패턴으로 판단하게 한다. 논문은 이 방법이 CIFAR-10, CIFAR-100, Fashion-MNIST, CatsVsDogs 등 여러 데이터셋에서 기존 방법들을 큰 폭으로 앞선다고 보고한다.

연구 문제의 중요성은 매우 분명하다. 자율주행, 로봇, 안전-critical vision system에서는 학습 시 보지 못한 입력이 들어왔을 때 이를 정상으로 오인하지 않는 능력이 필수적이다. 기존 deep anomaly detection 연구는 주로 autoencoder나 GAN을 이용한 reconstruction error에 의존해 왔는데, 논문은 이런 계열이 이미지 문제에서 충분히 강력하지 않을 수 있음을 지적한다. 따라서 “정상 데이터를 얼마나 잘 복원하느냐”가 아니라, “정상 데이터의 구조적 특징을 얼마나 잘 식별하느냐”를 중심에 둔 discriminative formulation이 핵심 문제의식이다. 이 점이 논문의 출발점이다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 다음과 같다. **정상 이미지에 여러 기하학적 변환을 가했을 때, 그 변환의 종류를 구분하도록 학습된 분류기는 정상 클래스의 모양, 배치, 구조적 단서에 민감한 feature를 배우게 된다.** 예를 들어 dog 클래스만 가지고 학습한다면, 네트워크는 dog 이미지가 뒤집히거나 회전하거나 이동했을 때 어떤 변화가 생기는지를 안정적으로 구분하기 위해 dog의 형태적 특징을 내부적으로 학습하게 된다. 반면 abnormal 이미지에 같은 변환을 적용하면, 그 softmax 패턴은 정상 이미지들에서 관측된 패턴과 다르게 나타날 가능성이 높다.

기존 접근과의 가장 큰 차별점은 **생성적 복원 문제를 풀지 않는다는 점**이다. 논문은 autoencoder, GAN, density estimation on latent representation 같은 흐름과 달리, 처음부터 끝까지 discriminative classifier만 사용한다. 즉, 생성 모델의 품질이나 reconstruction fidelity에 의존하지 않는다. 학습은 일반적인 다중분류와 크게 다르지 않으며, 테스트 시에는 여러 변환에 대한 softmax 출력만 모아 normality score를 계산한다. 이런 설계는 구현이 단순하고, 학습 안정성이 상대적으로 높으며, 실험에서는 성능도 강력하다는 점이 강조된다.

또 하나 중요한 아이디어는, 단순히 “정답 transformation의 softmax 확률이 큰가”만 보는 것이 아니라, **정상 데이터에서 transformation별 softmax 벡터가 어떤 분포를 이루는지**를 모델링한다는 점이다. 저자들은 각 transformation 조건에서의 softmax 벡터 분포를 Dirichlet distribution으로 근사하고, 테스트 이미지의 변환별 softmax 벡터들이 그 분포 아래에서 얼마나 그럴듯한지를 합산해 normality score를 만든다. 이 때문에 최종 점수는 단일 confidence 값보다 더 풍부한 통계 정보를 담는다.

## 3. 상세 방법 설명

전체 파이프라인은 비교적 명료하다. 먼저 정상 이미지 집합 $S$와 geometric transformation 집합 $\mathcal{T} = {T_0, T_1, \dots, T_{k-1}}$가 주어진다. 여기서 $T_0$는 identity transformation이다. 각 정상 이미지 $x \in S$에 대해 모든 변환 $T_j$를 적용하고, 그 결과 이미지 $T_j(x)$에 레이블 $j$를 붙인다. 그러면 다음과 같은 self-labeled 데이터셋이 만들어진다.

$$
S_{\mathcal{T}} \triangleq {(T_j(x), j) : x \in S, ; T_j \in \mathcal{T}}.
$$

즉, 원래 one-class 문제였던 데이터를 “어떤 transformation이 적용되었는가”를 맞히는 $k$-class classification 문제로 바꾸는 것이다. 이렇게 생성된 데이터셋의 크기는 $|S| \times |\mathcal{T}|$가 된다.

이 self-labeled 데이터셋 위에서 분류기 $f_\theta$를 **표준 cross-entropy loss**로 학습한다. 논문은 특정 손실을 새로 설계하지 않는다. 핵심은 label의 의미를 사람이 준 semantic class가 아니라 transformation index로 바꾼 것이다. 따라서 training objective 자체는 아주 단순하다. 다만 그 단순한 auxiliary task가 anomaly detection에 유용한 representation을 유도한다는 점이 논문의 포인트다.

학습이 끝난 뒤 테스트 이미지 $x$가 들어오면, 모든 transformation $T_i$를 다시 적용한다. 각 변환된 이미지에 대해 분류기의 softmax 출력 벡터를 얻는다.

$$
\mathbf{y}(x) \triangleq \text{softmax}(f_\theta(x)).
$$

그러면 각 $T_i(x)$에 대해 $\mathbf{y}(T_i(x)) \in \mathbb{R}^k$라는 softmax 벡터가 생긴다. 논문은 정상 이미지에서 이 벡터들이 transformation별로 일정한 통계적 패턴을 가진다고 본다. 그래서 조건부분포 $\mathbf{y}(T_i(x)) \mid T_i$를 Dirichlet distribution으로 근사한다.

$$
\mathbf{y}(T_i(x)) \mid T_i \sim \text{Dir}(\alpha_i).
$$

여기서 $\alpha_i \in \mathbb{R}_+^k$는 transformation $T_i$ 조건에서의 Dirichlet 파라미터다. 각 $\alpha_i$는 정상 훈련셋 $S$에 있는 이미지들만 사용하여 추정한다. 구체적으로는

$$
S_i = {\mathbf{y}(T_i(x)) \mid x \in S}
$$

를 만들고, 이 샘플들로부터 maximum likelihood 방식으로 $\tilde{\alpha}_i$를 구한다. 논문은 초기화는 Wicker et al.의 방법, 추정은 Minka의 fixed-point iteration을 사용한다고 명시한다. 즉, 제안법의 중요한 통계적 뒷받침은 “softmax 벡터는 simplex 위에 있으므로 Dirichlet로 근사하기 자연스럽다”는 점이다.

최종 normality score는 변환별 softmax 벡터가 해당 Dirichlet 분포에서 얼마나 높은 likelihood를 가지는지 합산한 값이다.

$$
n_S(x) \triangleq \sum_{i=0}^{k-1} \log p(\mathbf{y}(T_i(x)) \mid T_i).
$$

Dirichlet log-likelihood를 대입하면,

$$
n_S(x) = \sum_{i=0}^{k-1} \left[ \log \Gamma \left(\sum_{j=0}^{k-1} [\tilde{\alpha}_i]_j \right) - \sum_{j=0}^{k-1} \log \Gamma([\tilde{\alpha}_i]_j) + \sum_{j=0}^{k-1} ([\tilde{\alpha}_i]_j - 1)\log \mathbf{y}(T_i(x))_j \right].
$$

여기서 첫 두 항은 $x$와 무관한 상수이므로 ranking 관점에서는 생략 가능하다. 따라서 논문은 다음의 단순화된 score를 사용한다.

$$
n_S(x) = \sum_{i=0}^{k-1}\sum_{j=0}^{k-1} ([\tilde{\alpha}_i]_j - 1)\log \mathbf{y}(T_i(x))_j = \sum_{i=0}^{k-1} (\tilde{\alpha}_i - 1)\cdot \log \mathbf{y}(T_i(x)).
$$

이 식의 의미를 쉽게 풀어 말하면, 각 transformation에 대해 정상 데이터가 보이던 softmax 패턴과 테스트 이미지의 softmax 패턴이 얼마나 잘 맞는지를 모두 더한 값이다. score가 높을수록 정상 이미지일 가능성이 높다. 논문은 threshold selection 자체는 다루지 않고, 이 score의 ranking quality를 AUROC로 평가한다.

또한 논문은 보다 단순한 초기 버전의 점수도 소개한다.

$$
\hat{n}_S(x) \triangleq \frac{1}{k}\sum_{j=0}^{k-1} [\mathbf{y}(T_j(x))]_j.
$$

이는 각 transformation을 적용했을 때, “그 transformation class에 해당하는 softmax 좌표”를 평균내는 방식이다. 구현은 훨씬 쉽고 Dirichlet fitting도 필요 없지만, 성능은 full Dirichlet score보다 약간 낮다고 설명한다. 흥미로운 점은 이후 multi-class OOD 실험에서는 이 simplified score를 사용했다는 것이다.

변환 집합의 설계도 중요하다. 본 실험에서 사용한 transformation은 총 72개이며, horizontal flip 여부 2가지, 가로/세로 방향 translation 각각 3가지씩, 그리고 $90^\circ$ 단위 rotation 4가지를 조합한다. 따라서 총 개수는 $2 \times 3 \times 3 \times 4 = 72$개다. translation은 이미지 높이와 너비의 0.25배만큼 이동하며, 비어 있는 픽셀은 reflection으로 채운다. 저자들은 Gaussian blur나 sharpening, gamma correction 같은 non-geometric transformation도 시도했지만 성능이 떨어져 제외했다고 밝힌다. 이 부분은 논문의 귀중한 실험적 통찰이다. 즉, anomaly detection에 도움이 되는 auxiliary task는 아무 task나 좋은 것이 아니라, **정상 클래스의 공간적 구조를 보존하면서도 구별 가능한 task**여야 한다는 뜻이다.

## 4. 실험 및 결과

실험은 one-vs-all protocol로 진행된다. 데이터셋에 클래스가 $C$개 있으면, 각 클래스 $c$를 정상 클래스로 두는 실험을 하나씩 만든다. 학습에는 오직 해당 정상 클래스의 training image만 사용하고, 테스트에서는 정상 클래스와 나머지 모든 클래스 이미지를 함께 넣어 anomaly detection 성능을 평가한다. 이 설정은 pure single-class anomaly detection의 전형적인 형태이며, 논문은 threshold를 정하는 binary decision 문제 대신 score ranking 문제에 집중하여 **AUROC**를 기본 지표로 사용한다. 보조적으로 supplementary material에는 AUPR-In과 AUPR-Out도 제시한다.

비교 대상은 상당히 강하다. 전통적 방법으로 RAW-OC-SVM, convolutional autoencoder bottleneck 위의 CAE-OC-SVM이 있고, deep one-class 계열의 E2E-OC-SVM(Deep SVDD), deep structured energy-based model(DSEBM), Deep Autoencoding Gaussian Mixture Model(DAGMM), 그리고 GAN 기반의 ADGAN을 포함한다. 특히 RAW-OC-SVM과 CAE-OC-SVM의 하이퍼파라미터 $\nu, \gamma$는 AUROC를 최대화하도록 hindsight로 최적화했다고 논문이 명시한다. 즉, baseline에게 다소 유리한 설정을 준 셈인데도 제안법이 우세하다는 점을 강조한다.

데이터셋은 CIFAR-10, CIFAR-100, Fashion-MNIST, CatsVsDogs 네 가지다. CIFAR-100은 base class 100개 대신 20개의 superclass를 사용한다. 이는 클래스별 데이터 수가 너무 적은 문제를 피하기 위한 설계다. 32×32 데이터셋에는 WRN depth 10, width 4를, CatsVsDogs 64×64에는 WRN depth 16, width 8을 사용했다. optimizer는 Adam, batch size는 128, epoch 수는 대부분 200이다. self-labeled dataset은 transformation 수만큼 데이터가 늘어나므로, WRN은 $ \lceil 200 / |\mathcal{T}| \rceil $ epoch만 학습하여 대략 원래 200 epoch와 비슷한 parameter update 수를 맞췄다.

핵심 결과는 매우 강력하다. Table 1의 평균 AUROC를 보면 CIFAR-10에서 baseline들의 평균은 대략 53.1~64.8 수준인데, 제안법은 **86.0**을 기록한다. 특히 normal class가 automobile(class 1)일 때 기존 최고 baseline이 65.9 수준인데 제안법은 **95.7**이고, horse(class 7)에서도 baseline 최고가 67.3 수준인데 제안법은 **95.5**다. 논문이 말하는 “baseline이 힘들어하는 경우에 오히려 더 강하다”는 주장을 잘 보여준다.

CIFAR-100에서도 평균 AUROC는 baseline들이 대체로 50.5~63.1 부근인 반면 제안법은 **78.7**이다. 물론 여기서는 모든 클래스에서 완벽하지는 않다. 예를 들어 class 13(non-insect invertebrates)은 **58.0**, class 5(household electrical devices)는 **59.1**, class 7(insects)는 **65.0**으로 상대적으로 낮다. 논문은 이를 정상 클래스 내부의 다양성이 큰 경우로 해석한다. 즉, 단일 “정상 클래스”가 실제로는 시각적으로 매우 이질적인 하위 분포들의 집합이면, transformation discrimination으로 학습한 representation의 응집도가 낮아질 수 있다는 시사점이 있다.

Fashion-MNIST에서는 모든 방법이 대체로 높은 성능을 보인다. 평균 AUROC는 RAW-OC-SVM 92.8, CAE-OC-SVM 91.7, DSEBM 86.6, ADGAN 88.4, 제안법 **93.5**이다. 여기서는 제안법의 우세가 존재하되, CIFAR-10이나 CatsVsDogs만큼 극적이지는 않다. 이는 데이터셋 자체가 상대적으로 쉬운 anomaly detection 환경이기 때문으로 읽힌다.

가장 인상적인 결과는 CatsVsDogs다. baseline들은 거의 random guessing에 가까운 수준이다. 평균 AUROC가 RAW-OC-SVM 51.7, CAE-OC-SVM 52.5, DSEBM 51.6, ADGAN 49.4인 반면, 제안법은 **88.8**이다. 논문 본문에서도 dog를 정상으로 둔 경우 best baseline 0.561에 비해 제안법이 0.888이라고 직접 예시를 든다. 큰 해상도 자연 이미지에서 reconstruction 계열이 잘 버티지 못하는 반면, transformation discrimination은 훨씬 더 robust하다는 메시지가 강하게 전달된다.

보조 실험으로 labeled multi-class dataset의 out-of-distribution detection도 다룬다. 여기서는 category 분류용 head와 transformation 분류용 head를 함께 둔 two-headed WRN을 사용하고, 테스트 때는 transformation head만 사용한다. CIFAR-10을 정상 분포로, resized Tiny-ImageNet을 OOD로 둔 설정에서 ODIN의 AUROC/AUPR-In/AUPR-Out 92.1/89.0/93.6을 제안법이 **95.7/96.1/95.4**로 개선했다고 보고한다. 다만 저자들은 이 실험이 주된 초점은 아니며, 자신들의 방법은 class label이 전혀 없는 pure single-class setting에도 적용 가능하다는 점이 더 중요하다고 말한다.

논문 6장은 정량 실험 외에 방법의 직관을 검증하는 소규모 분석도 제시한다. 예를 들어 MNIST 숫자 ‘8’을 정상으로 두고 horizontal flip만 구분하는 문제에서는 ‘8’이 좌우 반전에 거의 invariant하므로 transformation classifier가 충분한 discriminative feature를 못 배우고, anomaly detection 성능도 낮아 AUROC 0.646에 그친다. 반대로 정상이 ‘3’이고 anomaly가 ‘8’인 경우에는 flip 구분이 쉬워서 AUROC 0.957을 얻는다. 또 ‘8’을 정상으로 두되 horizontal flip 대신 translation을 쓰면 AUROC가 0.919까지 올라간다. 이 관찰은 “좋은 transformation이란 정상 클래스의 구조는 유지하면서도 서로 구별 가능해야 한다”는 저자들의 가설을 강하게 뒷받침한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 재정의 자체가 매우 영리하다**는 점이다. anomaly detection을 직접 풀려 하지 않고, 정상 클래스에서만 만들 수 있는 self-supervised classification task로 우회한다. 이 아이디어는 간단하지만 강력하며, reconstruction 품질에 집착하던 당시 흐름에서 분명한 전환점을 만든다. 또한 구현이 복잡한 generative model 없이도 WRN + cross-entropy만으로 강한 성능을 낸다는 점에서 실용성도 높다.

둘째 강점은 **실험적 우위가 일관되고 큰 폭**이라는 점이다. 단순히 평균 성능이 조금 좋은 수준이 아니라, 특히 어려운 설정에서 baseline이 거의 무너질 때도 제안법이 강한 성능을 유지한다. CatsVsDogs 결과는 이 논문의 설득력을 크게 높인다. 또한 OC-SVM에 hindsight hyperparameter tuning이라는 유리한 조건을 부여했음에도 우세했다는 점도 인상적이다.

셋째 강점은 방법이 어느 정도 해석 가능하다는 점이다. transformation classifier가 무엇을 배우는지에 대해 MNIST 예시와 gradient ascent visualization을 통해 직관을 제시한다. 특히 정상 점수를 높이도록 ‘0’을 최적화하자 ‘3’처럼 보이게 변형되는 결과는, 모델이 정상 클래스 특유의 feature를 실제로 포착했다는 설득력 있는 사례다.

하지만 한계도 분명하다. 첫째, **Dirichlet independence assumption**은 명백히 근사적이다. 논문도 여러 transformation에 대한 softmax 벡터들이 서로 독립이라는 가정이 “naive”하고 “typically incorrect”하다고 사실상 인정한다. 그럼에도 실용적으로 잘 작동한다는 것이지, 이론적으로 엄밀한 정당화가 제공된 것은 아니다.

둘째, 변환 집합 $\mathcal{T}$의 선택은 매우 중요하지만, **어떤 데이터셋에 어떤 변환이 최적인지에 대한 체계적 이론은 없다**. 논문은 경험적으로 72개 geometric transformation이 잘 된다고 보여주지만, 이는 문제별 tuning에 가깝다. 특히 정상 클래스가 회전이나 좌우 반전에 본질적으로 invariant하면 auxiliary task가 무력해질 수 있다. MNIST ‘8’ 예시는 바로 그 취약점을 드러낸다.

셋째, 정상 클래스 내부 다양성이 큰 경우 성능이 떨어질 수 있다. CIFAR-100 superclass 일부에서 성능이 낮은 것은 이 점을 보여준다. 즉, “정상”이라는 개념이 하나의 응집된 시각 패턴이 아니라 넓은 분포일 때는 transformation discrimination만으로 충분하지 않을 수 있다.

넷째, 논문은 threshold selection과 calibrated decision rule을 다루지 않는다. AUROC 중심 평가는 ranking 품질을 보여주지만, 실제 시스템에 넣으려면 false positive budget 하에서 어떤 threshold를 잡을지, domain shift 상황에서 calibration이 유지되는지 같은 운영 측면의 문제가 남는다. 이 부분은 후속 연구가 필요하다.

비판적으로 보면, 제안법은 완전히 “anomaly”를 직접 모델링한다기보다는 “정상 클래스의 transformation semantics를 얼마나 잘 유지하는가”를 간접적으로 측정한다. 이는 매우 강력한 우회 전략이지만, 만약 anomaly가 정상 클래스와 형태적으로 매우 유사하면서 transformation response까지 비슷하게 보인다면 한계가 있을 수 있다. 논문은 이 케이스를 깊게 분석하지는 않는다. 따라서 본 방법은 특히 **정상 클래스 특유의 구조적 cue가 명확한 시각 도메인**에서 강력하고, 그렇지 않은 경우에는 성능 편차가 생길 가능성이 있다.

## 6. 결론

이 논문은 이미지 anomaly detection에서 매우 중요한 기여를 한다. 핵심 기여는 세 가지로 정리할 수 있다. 첫째, pure single-class anomaly detection을 위해 **geometric transformation discrimination**이라는 새로운 self-supervised surrogate task를 제안했다. 둘째, transformation별 softmax 벡터를 Dirichlet distribution으로 모델링하여 normality score를 정의하는 간단하면서도 효과적인 scoring framework를 만들었다. 셋째, 여러 대표 이미지 데이터셋에서 당시 state-of-the-art를 큰 폭으로 앞서는 결과를 보였다.

실제 적용 측면에서도 의미가 크다. 정상 데이터만 상대적으로 많이 확보 가능한 산업 비전, 이상 탐지, 품질 검사, 의료 영상 사전 필터링 같은 환경에서 매우 유용할 가능성이 있다. 특히 생성 모델 없이 분류기만으로 구현 가능하다는 점은 엔지니어링 부담을 줄여 준다. 더 넓게 보면, 이 논문은 self-supervised learning이 anomaly detection에 어떻게 접목될 수 있는지를 보여 준 초기의 대표 사례로 읽을 수 있다.

향후 연구 방향도 자연스럽다. transformation set를 자동으로 선택하거나 학습하는 방법, 클래스 내부 다양성이 큰 경우를 위한 보완 설계, 더 일반적인 OOD detection 및 uncertainty estimation과의 결합, 그리고 score calibration까지 이어질 수 있다. 요약하면, 이 논문은 단순한 성능 향상을 넘어, anomaly detection 문제를 바라보는 관점을 “복원 중심”에서 “구조적 판별 중심”으로 옮긴 매우 영향력 있는 작업이다.
