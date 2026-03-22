# Batch Active Learning via Coordinated Matching

이 논문은 **batch active learning** 에서 가장 중요한 난제인 “한 번에 여러 개의 샘플을 선택할 때, informativeness와 diversity를 어떻게 동시에 만족시킬 것인가”를 다룬다. 저자들은 기존 sequential active learning 정책이 대체로 더 example-efficient하다는 점에 주목하고, batch 방법을 처음부터 새로 설계하기보다 **좋은 sequential policy가 $k$ 단계 동안 보일 행동을 모사하는 batch 선택**을 만들자는 관점을 제안한다. 이를 위해 Monte Carlo simulation으로 sequential policy의 $k$-step 선택 분포를 추정하고, 그 분포를 가장 잘 근사하는 $k$개 샘플을 고르는 문제를 **Bounded Coordinated Matching (BCM)**으로 정식화한다. BCM은 NP-hard이지만, 논문은 이를 위해 **supermodular minimization 기반 greedy 알고리즘**을 제안하고 근사 보장을 제공한다. 실험적으로도 8개 UCI 이진 분류 데이터셋에서 기존 batch active learning baseline보다 일관되게 강한 성능을 보였다고 보고한다.

## 1. Paper Overview

이 논문의 연구 문제는 매우 실용적이다. 일반적인 active learning은 unlabeled pool에서 가장 informative한 샘플을 골라 라벨링하는 문제인데, 많은 기존 연구는 **한 번에 하나씩 고르는 sequential setting** 에 집중해 왔다. 하지만 현실에서는 라벨링이 병렬로 가능하거나, wet lab experiment처럼 라벨 하나하나의 생성 시간이 길어 **한 번에 batch로 질의하는 방식**이 더 자연스러운 경우가 많다. 문제는 batch로 단순히 sequential rule을 $k$번 반복하면, 서로 비슷하고 중복된 샘플이 함께 선택될 가능성이 높아 성능이 떨어진다는 점이다. 논문은 바로 이 지점을 겨냥한다.

저자들의 기본 문제의식은 다음과 같다. 좋은 sequential active learning 정책은 이미 많이 존재하고, 이론적으로도 최적 sequential 전략은 최적 batch 전략보다 나쁠 수 없다. 따라서 batch active learning의 목표를 “batch 전용 점수함수 만들기”로 보기보다, **좋은 sequential policy의 기대 행동을 최대한 잘 흉내 내는 batch를 만드는 것**으로 재정의한다. 이 점이 논문의 핵심 출발점이다.

## 2. Core Idea

논문의 핵심 아이디어는 **simulation matching** 이다. 구체적으로는, 어떤 sequential active learning 정책 $\pi$ 가 현재 labeled set $D_l$ 와 unlabeled pool $D_u$ 에서 앞으로 $k$번 연속 선택을 한다고 생각하자. 이때 매 단계에서 아직 관측되지 않은 라벨이 이후 선택에 영향을 미치므로, 실제로는 $\pi$ 가 만들어 내는 $k$개 샘플 집합은 확률변수다. 논문은 이를 $S_\pi^k$ 로 놓고, 이 분포를 직접 최적화하기보다 Monte Carlo로 샘플링한 뒤, 그 샘플 집합 분포를 잘 근사하는 batch를 고른다.

여기서 novelty는 단순 시뮬레이션 자체가 아니라, “잘 근사한다”를 **coordinated matching** 이라는 조합 최적화 문제로 정식화했다는 데 있다. 저자들은 일반적인 i.i.d. mixture model이 아니라, **각 mixture component가 정확히 하나의 점을 생성하도록 강제하는 $k$-Matching Mixture Model (k-MMM)**을 도입한다. 이 구조는 batch 내부 샘플들 간의 의존성을 부분적으로 반영하여, 동일한 component에서 중복 샘플이 몰리는 현상을 줄여 준다. 결과적으로 기존의 uncertainty-only selection보다 batch diversity를 자연스럽게 반영할 수 있다.

즉 이 논문의 핵심 기여는 다음 세 줄로 요약된다.

* 좋은 sequential policy의 $k$-step 행동을 Monte Carlo simulation으로 추정한다.
* 그 분포를 근사하는 모델을 $k$-MMM으로 둔다.
* 최적 batch 선택을 BCM이라는 조합 최적화 문제로 풀고, greedy 근사해를 제시한다.

## 3. Detailed Method Explanation

### 3.1 Sequential policy simulation

논문은 먼저 sequential active learning 정책 $\pi$ 를 가정한다. $\pi$ 는 현재의 labeled data와 unlabeled data를 입력받아 다음에 라벨링할 샘플 하나를 선택한다. 그런데 batch를 만들려면 $\pi$ 를 $k$번 실행했을 때 어떤 샘플 집합이 생기는지 알아야 한다. 문제는 첫 번째 샘플의 실제 라벨을 아직 모르므로, 두 번째 선택 이후가 결정되지 않는다는 점이다. 이를 위해 논문은 **확률적 분류기(probabilistic classifier)** 로부터 샘플의 label posterior를 얻고, 이를 사용해 라벨을 샘플링하며 시뮬레이션을 진행한다.

즉, 한 trajectory는 다음처럼 만들어진다.

1. 현재 $D_l$ 에서 정책 $\pi$ 로 첫 샘플 $x_1$ 선택
2. $x_1$ 의 posterior label 분포에서 $y_1$ 샘플링
3. $(x_1, y_1)$ 을 $D_l$ 에 추가
4. 이 과정을 총 $k$단계 반복

이렇게 하면 sequential policy가 미래 라벨 결과에 따라 어떤 집합을 선택할지를 Monte Carlo로 근사할 수 있다. 논문은 이때 얻어지는 $k$개 샘플 집합의 확률변수를 $S_\pi^k$ 라고 정의한다.

### 3.2 왜 일반 mixture model이 아니라 $k$-MMM인가

논문은 $S_\pi^k$ 의 분포를 단순 Gaussian mixture model로 근사하는 것은 좋지 않다고 본다. 이유는 GMM이 **i.i.d.로 $k$개를 생성**하므로, 가장 강한 component 주변 샘플이 반복 선택되어 batch 내부 redundancy를 반영하지 못하기 때문이다. 반면 실제 sequential policy는 이미 선택된 샘플과 유사한 점을 반복 선택하지 않으므로, 선택된 샘플들은 독립적이지 않고 강한 의존성을 가진다.

이를 해결하기 위해 제안된 것이 **$k$-Matching Mixture Model** 이다. 이 모델은 $k$개의 Gaussian component를 두고, 각 component가 정확히 하나의 샘플을 생성하도록 강제한다. 즉 집합 $S={x_1,\dots,x_k}$ 의 생성 확률은 가능한 모든 matching에 대해 합을 취한 형태로 정의된다. 논문 식 (1)은 이를 다음처럼 표현한다.

$$
Q^k(S)=\frac{1}{k!}\sum_{m\in M}\prod_{i=1}^k f(x_i;\mu_{m(i)},\Sigma_{m(i)})
$$

여기서 $M$ 은 가능한 모든 matching 집합이고, $f$ 는 Gaussian PDF다. 이 구조 덕분에 batch 내부 샘플들이 서로 다른 component에 대응되며, **다양한 위치를 고르게 커버하는 집합**에 더 높은 확률을 줄 수 있다.

### 3.3 Cross-entropy minimization과 BCM objective

이제 문제는 sequential policy의 시뮬레이션으로 얻은 target distribution $P_\pi^k$ 를 가장 잘 근사하는 $Q_\mu^k$ 를 찾는 것이다. 저자들은 component mean $\mu={\mu_1,\dots,\mu_k}$ 를 unlabeled pool $D_u$ 에서 선택하고, covariance는 고정한다고 두어 문제를 단순화한다. 그다음 최적화 목표를 다음의 KL divergence 최소화, 즉 cross-entropy 최소화로 둔다.

$$
\min_{\mu\in U^k} H(P_\pi^k,Q_\mu^k)
$$

하지만 $P_\pi^k$ 를 닫힌 형태로 알 수 없기 때문에, Monte Carlo simulation으로 얻은 sample trajectory 집합 $\mathcal{S}={S_1,\dots,S_N}$ 로 expectation을 근사한다. 이때 추가 근사를 통해 최종 objective는 다음 형태로 정리된다.

$$
\arg\min_{\mu\in U^k}\sum_{i=1}^N \min_{m\in M}\sum_{j=1}^k d_\Sigma(x_{ij},\mu_{m(j)})
$$

이 식이 바로 **Bounded Coordinated Matching (BCM)**이다. 직관적으로는, 각 simulation sample set $S_i$ 와 candidate batch $\mu$ 사이의 minimum-cost matching을 계산하고, 그 총합이 가장 작은 $\mu$ 를 찾는 문제다. 결국 batch $\mu$ 가 simulation trajectories 전체를 얼마나 잘 대표하는지를 matching cost로 측정하는 셈이다.

### 3.4 BCM의 성질과 greedy 최적화

논문은 BCM이 **NP-complete** 임을 보인다. 부록에서는 3-Dimensional Matching으로부터의 reduction을 통해 이를 증명한다. 따라서 exact solution은 어렵고, 저자들은 set function 관점에서 BCM objective를 재해석한다.

핵심은 BCM objective가 다음과 같은 set function $g(\mu)$ 로 표현된다는 점이다.

$$
g(\mu)=\sum_{i=1}^N \min_{m\in M}\sum_{j=1}^k d_\Sigma(x_{ij},\mu_{m(j)})
$$

논문은 이 함수가 **non-increasing supermodular function** 임을 보이고, 따라서 supermodular minimization에 대한 기존 결과를 적용할 수 있다고 말한다. 이에 따라 greedy 알고리즘은 처음에 전체 candidate set에서 시작해, 매 단계마다 제거했을 때 objective 증가가 가장 작은 원소를 하나씩 삭제해 최종적으로 $k$개만 남긴다. 이 알고리즘은 Theorem 2에 따라 steepness parameter $t$ 에 의존하는 근사 보장을 갖는다.

여기서 중요한 점은 보통의 submodular maximization greedy처럼 원소를 추가하는 방식이 아니라, **전체 집합에서 원소를 제거해 나가는 greedy descent** 라는 것이다. 논문은 이것이 이 문제 구조에 더 맞는 형태라고 설명한다.

### 3.5 가속화 기법

논문은 naive greedy가 비용이 크다는 점도 인식하고, 세 가지 가속화 아이디어를 제안한다.

첫째, 초기 candidate set $\mu$ 를 전체 unlabeled pool이 아니라 **simulation에서 실제 등장한 점들의 합집합 $\mu_0=\bigcup_i S_i$** 로 줄인다. 이렇게 하면 greedy 최적화의 시간복잡도가 전체 pool 크기 $|D_u|$ 가 아니라 최대 $N\cdot k$ 규모에 의존하게 된다.

둘째, supermodularity를 이용해 매 iteration에서 모든 incremental difference를 다시 계산하지 않고, 이전 계산 결과를 재활용해 많은 재계산을 생략한다. 이는 submodular maximization에서 lazy evaluation을 쓰는 것과 유사한 가속이다.

셋째, matching 계산도 매번 Hungarian algorithm을 처음부터 돌리지 않고, 기존 matching을 저장한 뒤 필요한 경우만 업데이트한다. 논문은 이를 통해 적어도 factor $k$ 정도의 시간 절감을 기대할 수 있다고 말한다.

### 3.6 Scalability

논문은 전체 알고리즘의 계산을 두 단계로 나눈다.

1. sequential policy simulation
2. BCM 최적화

첫 단계는 base sequential policy가 unlabeled pool 전체를 봐야 하므로 대체로 $|D_u|$ 에 선형으로 증가한다. 하지만 simulation trajectory들은 서로 독립이라 병렬화가 쉽다. 두 번째 단계는 앞서 설명한 초기화 덕분에 $|D_u|$ 와 독립적이고, 대략 $N\cdot k$ 에 의존한다. 따라서 전체적으로는 **기본 sequential policy와 비슷한 수준의 scalability** 를 가진다고 주장한다.

## 4. Experiments and Findings

논문은 8개의 UCI 이진 분류 데이터셋에서 실험한다. 데이터셋은 Breast, Ionosphere, Pima, German, Haberman, Sonar, EF, MN이며, EF와 MN은 letter dataset의 이진 부분집합이다. 분류기는 RBF kernel을 쓴 kernel logistic regression이고, 각 데이터셋을 70% train, 30% test로 나누고, 초기 labeled set은 클래스당 5개 랜덤 샘플로 설정한다. 각 실험은 50회 반복 평균이다. Batch size는 10과 20, simulation trajectory 수는 $N=20$ 이다.

비교 baseline은 다음 네 가지다.

* Fisher Information 기반 batch active learning
* Maximum Uncertain: entropy가 가장 높은 top-$k$ 선택
* Random
* Sequential maximum entropy policy

제안 방법은 sequential entropy policy를 base policy로 삼아 이를 simulation matching으로 batch화한 것이다.

결과는 Figure 1에 정리되어 있다. 논문 본문과 8페이지 그래프를 함께 보면, **BCM은 대부분의 데이터셋에서 다른 batch 방법들보다 learning curve가 우세**하다. 특히 batch size가 20일 때 이 우위가 더 두드러진다. 본문은 Fisher Information과 Maximum Uncertain가 Pima와 German 같은 데이터셋에서는 오히려 Random보다 일관되게 나쁜 경우도 있었다고 지적한다. 즉, batch active learning에서 naive uncertainty/diversity 설계가 random보다도 나쁠 수 있음을 보여 준다. 이에 비해 BCM은 비교군 중 유일하게 **일관되게 robust** 한 batch 성능을 보였다고 저자들은 주장한다.

또한 sequential 정책은 대체로 batch 방법보다 더 좋았지만, BCM은 그 sequential 성능에 비교적 가깝게 따라갔고, 일부 데이터셋(Ionosphere, Breast)에서는 오히려 sequential을 능가하기도 했다고 논문은 설명한다. 저자들은 이것이 여러 simulation outcome을 aggregate함으로써 base sequential policy의 variance를 줄이는 효과 때문일 수 있다고 해석한다.

계산 시간도 보고된다. 가장 큰 데이터셋 중 하나인 MN에서 batch size 20을 고르는 데, 최적화되지 않은 Matlab 구현으로도 표준 데스크톱에서 **3분 이하**가 걸렸다고 한다. 이는 batch active learning의 일반적 사용 맥락에서 충분히 실용적이라는 것이 논문의 주장이다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 batch active learning을 직접 휴리스틱으로 만들기보다, **검증된 sequential policy를 모사하는 principled한 틀**로 재구성했다는 점이다. 기존 batch active learning은 대개 uncertainty와 diversity 사이의 trade-off를 수작업으로 설계했는데, 이 논문은 그 대신 sequential policy가 실제로 어떤 batch-like behavior를 유도하는지를 simulation으로 추정한다. 이 발상 자체가 매우 깔끔하다.

둘째, BCM 정식화가 단순 직관에 머물지 않고 **조합 최적화 + supermodular minimization** 의 형태로 떨어진다는 점이 강하다. 즉, heuristic처럼 보이지만 수학적 구조를 잘 끌어내 근사 보장까지 제시한다. batch active learning 논문 중 이런 이론적 정리가 있는 경우는 상대적으로 드물다.

셋째, 실험 결과가 robust하다. 논문이 강조하듯 random보다 consistently 좋은 batch 방법을 만드는 것조차 쉽지 않은데, BCM은 여러 데이터셋과 두 batch size에서 안정적인 우위를 보였다. 특히 page 8의 Figure 1을 보면 많은 데이터셋에서 BCM curve가 batch baseline들 위쪽에 위치하며, sequential curve와도 꽤 가까운 경우가 많다.

### 한계

한계도 분명하다.

첫째, 이 방법은 **base sequential policy의 품질에 강하게 의존**한다. 논문의 목적 자체가 sequential policy를 모사하는 것이므로, base policy가 부적절하면 batch 결과도 좋아지기 어렵다. 다시 말해 BCM은 “좋은 batch objective를 직접 학습”하는 방법이 아니라, sequential policy의 wrapper에 가깝다.

둘째, Monte Carlo simulation과 matching 최적화가 들어가므로, 단순 top-$k$ uncertainty보다 구현이 복잡하다. 논문은 가속화 기법과 병렬화를 제시하지만, 여전히 매우 대규모 modern deep active learning setting에 바로 적용하기에는 비용 구조를 더 따져봐야 한다. 이 논문은 2012년의 kernel logistic regression 기반 setting에서 검증되었다는 점을 감안해야 한다.

셋째, 실험이 모두 소규모 UCI 이진 분류 데이터셋에 제한되어 있다. 따라서 이론적 아이디어는 흥미롭지만, 고차원 이미지나 deep model 기반 active learning에서 동일한 이점이 유지되는지는 이 논문만으로는 판단할 수 없다. 실제로 이후 딥 active learning 문헌에서는 diversity/coreset/gradient embedding 류 방법이 더 많이 발전했다. 이 논문은 그런 흐름의 개념적 선행 작업으로 읽는 편이 적절하다.

### 해석

비판적으로 보면, 이 논문의 진짜 공헌은 BCM 알고리즘 그 자체보다도 **batch active learning을 sequential policy approximation 문제로 재정의한 관점**에 있다. 즉, batch active learning이 본질적으로 sequential learning보다 열세라는 점을 인정하고, 그 차이를 줄이는 방향으로 설계한 것이다. 이 사고방식은 이후 batch Bayesian optimization, imitation-style batch design, sequential-to-batch reduction 같은 다양한 맥락과도 통한다.

## 6. Conclusion

이 논문은 batch active learning에서 redundancy 문제를 해결하기 위해, 좋은 sequential policy의 $k$-step 행동을 Monte Carlo simulation으로 추정하고 이를 가장 잘 근사하는 batch를 선택하는 **simulation matching** 접근을 제안했다. 이를 위해 $k$-Matching Mixture Model과 **Bounded Coordinated Matching (BCM)**이라는 새 조합 최적화 문제를 도입했고, BCM이 NP-hard임을 보이면서도 supermodular minimization 기반 greedy 근사 알고리즘과 가속화 기법을 제공했다. 실험에서는 8개 UCI 데이터셋에서 Fisher Information, Maximum Uncertain, Random보다 일관되게 우수한 batch 성능을 보였고, 많은 경우 sequential policy에 근접하는 성능까지 달성했다.

실무적 의미도 분명하다. 이 논문은 batch active learning에서 흔히 말하는 “uncertainty + diversity”를 직접 설계하는 대신, **좋은 sequential rule을 얼마나 잘 imitation할 수 있는가**를 중심 문제로 삼는다. 그래서 이 논문은 단순 성능 개선 논문이 아니라, batch active learning을 바라보는 또 하나의 기본 프레임을 제시한 논문으로 볼 수 있다.
