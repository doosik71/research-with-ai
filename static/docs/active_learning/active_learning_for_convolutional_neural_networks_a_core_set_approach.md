# Active Learning for Convolutional Neural Networks: A Core-Set Approach

이 논문은 CNN 기반 이미지 분류에서 **batch active learning** 이 왜 기존 불확실성 기반 휴리스틱으로 잘 작동하지 않는지 분석하고, 이를 **core-set selection** 문제로 재정의한 뒤, 실제 선택 문제를 **$k$-Center** 최적화로 푸는 방법을 제안한다. 저자들의 출발점은 현실적이다. CNN은 한 번에 한 샘플씩 질의하는 고전적 active learning 설정과 달리, 매 iteration마다 큰 배치를 골라 재학습해야 하므로 샘플 간 상관성이 커지고, 그 결과 uncertainty sampling류 방법이 기대만큼 효과적이지 않다는 것이다. 이에 따라 저자들은 “불확실한 샘플을 고르는 문제”보다 “작은 subset으로 전체 데이터를 잘 대표하는 문제”가 CNN batch active learning의 본질에 더 가깝다고 본다. 그 결과물이 바로 core-set 접근이며, 논문은 이 접근이 기존 방법보다 세 데이터셋에서 크게 우수하다고 주장한다.

## 1. Paper Overview

논문의 핵심 문제는 다음과 같다. CNN은 성능을 높이기 위해 많은 라벨 데이터를 요구하지만, 이미지 라벨링 비용은 매우 크다. 따라서 고정된 labeling budget 아래에서 어떤 샘플에만 라벨을 붙일지 잘 고르는 것이 중요하다. 이것이 active learning의 기본 문제다. 다만 저자들은, CNN에서는 기존 active learning 문헌에서 자주 쓰이는 entropy, Bayesian uncertainty, decision boundary distance 같은 방법들이 **batch acquisition** 때문에 효율이 급격히 떨어진다고 본다. 이유는 CNN이 한 번에 하나가 아니라 큰 묶음을 선택해야 하고, 그렇게 뽑힌 샘플들은 서로 비슷한 영역에 몰려 상관성이 커지기 때문이다.  

이 문제가 중요한 이유는 명확하다. CNN의 정확도는 데이터 양에 쉽게 포화되지 않기 때문에, 더 많은 데이터를 모으는 유인이 계속 존재한다. 하지만 비용 제약이 있는 실제 환경에서는 모든 샘플을 라벨링할 수 없다. 따라서 CNN에 맞는 active learning 전략이 필요하다. 저자들은 바로 이 점에서 기존 이론과 실험 모두가 충분치 않다고 보고, CNN 전용의 batch active learning 원리를 새로 세운다.

## 2. Core Idea

이 논문의 중심 아이디어는 active learning을 **core-set selection** 으로 재해석하는 데 있다. core-set selection은 원래 “큰 데이터셋의 작은 부분집합만으로도 전체 데이터셋으로 학습한 모델과 비슷한 성능을 내게 만드는 대표 subset을 고르는 문제”다. 저자들은 라벨이 없는 pool에서도 이 문제를 geometric coverage 관점에서 다룰 수 있다고 본다. 즉, 가장 informative한 점을 uncertainty로 찾는 대신, **feature space에서 전체 데이터를 잘 덮는 점들**을 고르면 CNN 학습에도 유리하다는 주장이다.

이 아이디어는 결국 다음 연결로 이어진다.

* 좋은 active set은 전체 데이터를 잘 대표하는 core-set이어야 하고,
* 이 대표성은 데이터 포인트 간 거리 구조로 근사할 수 있으며,
* 그 bound를 줄이는 문제는 **$k$-Center** 문제와 동치가 된다.

즉, 이 논문의 novelty는 “새 uncertainty score”를 제안한 것이 아니라, **CNN batch active learning의 목적함수 자체를 대표성 기반 core-set loss로 다시 세우고**, 그것을 조합 최적화 문제로 풀었다는 데 있다. 또한 fully supervised뿐 아니라 weakly supervised setting도 함께 다루며, CNN에 대해 이론적 정당화를 제공하려 했다는 점을 저자들이 직접 강조한다.

## 3. Detailed Method Explanation

### 3.1 문제 정의

논문은 batch active learning을 다음처럼 정식화한다. 초기 라벨 집합 $\mathbf{s}^0$ 가 있고, 추가로 budget $b$ 만큼의 샘플 $\mathbf{s}^1$ 을 골라 oracle에게 질의할 수 있다. 목표는 이 선택 후 학습된 모델의 기대 loss를 최소화하는 것이다. 핵심은 선택 시점에는 대부분의 라벨이 없기 때문에, 직접 supervised risk를 최소화할 수 없다는 점이다. 그래서 저자들은 이를 직접 최적화하지 않고, **subset이 전체 데이터를 얼마나 잘 덮는가**로 바꾸어 접근한다.

### 3.2 Core-set bound

논문 4장에서는 임의의 subset이 전체 데이터셋 loss를 얼마나 잘 대표하는지에 대한 bound를 제시한다. 큰 그림은 다음과 같다.

* loss가 입력에 대해 Lipschitz이고
* training error가 0에 가깝다고 가정하면
* subset과 전체 데이터 사이의 성능 차이는 결국 **각 데이터가 가장 가까운 selected point로부터 얼마나 떨어져 있는가**에 의해 제어된다.  

저자들은 CNN의 loss가 이 bound를 만족하도록 만들기 위해, 이론 전개에서는 cross-entropy가 아니라 **$l_2$ loss** 를 사용한다. 즉, desired class probability와 softmax 출력 사이의 $l_2$ distance를 쓴다. 실험은 실제 관례대로 cross-entropy로 수행하지만, 이론 파트는 $l_2$ 기반 Lipschitz continuity를 증명하는 방향으로 간다. 저자들 스스로도 이론이 cross-entropy까지 직접 확장되지는 않는다고 밝힌다.  

### 3.3 CNN에 대한 Lipschitz 성질

논문은 ReLU와 max-pool을 사용하는 CNN의 loss가 입력에 대해 Lipschitz라는 lemma를 제시한다. 특히 class probability와 network parameter가 고정되었을 때, softmax 출력 기반 $l_2$ loss는 입력에 대한 Lipschitz 함수가 되며, Lipschitz 상수는 네트워크 깊이와 weight magnitude에 의해 제어된다. 여기서 저자들은 $\alpha$ 를 neuron별 입력 가중치 합의 상한으로 두고, 이 값이 작아질수록 bound가 더 좋게 해석된다고 설명한다.

이 부분의 의미는 단순 수학 장식이 아니다. 저자들이 말하고 싶은 것은, CNN에서도 “representative subset을 잘 고르면 전체 loss도 통제된다”는 논리를 세울 수 있다는 것이다. 물론 zero training error 가정은 현실적이지 않지만, 논문은 실험상 이 upper bound가 실제로 꽤 effective하다고 해석한다.

### 3.4 $k$-Center로의 환원

이론 bound를 computational하게 풀기 위해, 저자들은 선택 문제를 사실상 다음 형태로 바꾼다.

$$
\min_{\mathbf{s}^1}\max_i \min_{j\in \mathbf{s}^0 \cup \mathbf{s}^1}\Delta(\mathbf{x}_i,\mathbf{x}_j)
$$

즉, 모든 데이터 포인트가 선택된 포인트 집합으로부터 갖는 **최대 최근접 거리**를 최소화한다. 이것이 전형적인 **$k$-Center** 문제다. 직관적으로는, selected set이 feature space 전체를 고르게 덮도록 만드는 것이다.

논문은 이를 위해 두 단계의 해법을 사용한다.

* 빠른 근사 해법인 **greedy $k$-Center**
* 더 정확한 해를 위한 **mixed integer programming(MIP)** 기반 solver

Greedy는 scalable하고, MIP는 더 좋은 해를 줄 수 있다. Figure 6과 관련 설명에 따르면, 2-OPT/greedy 근사만 써도 다른 baseline들을 이기지만, MIP를 쓰면 accuracy가 조금 더 좋아진다. 또한 50k 이미지 규모에서도 runtime이 practical하다고 주장한다.  

### 3.5 왜 uncertainty보다 coverage가 중요한가

논문은 Figure 5를 통해 uncertainty-based oracle과 제안 방법의 선택 패턴을 비교한다. uncertainty oracle은 feature space의 일부 영역에 쿼리가 몰려 전체 공간을 잘 덮지 못하는 반면, 제안 방법은 훨씬 고르게 공간을 커버한다. 저자들의 해석은 명확하다. CNN batch active learning에서 중요한 것은 “가장 헷갈리는 점들”을 모으는 것보다, **representation space를 넓게 커버하는 diverse representative subset** 을 고르는 것이다.

## 4. Experiments and Findings

논문은 세 데이터셋에서 image classification 실험을 수행했고, fully supervised와 weakly supervised 모델을 모두 평가했다. 서술상 baseline에는 random selection, uncertainty-based methods, Bayesian active learning류, 그리고 Wang & Ye 계열 방법이 포함된다. 저자들은 uncertainty 및 Bayesian 계열이 소규모에서는 의미가 있을 수 있으나, 대규모 CNN batch setting에서는 상관된 샘플을 뽑게 되어 성능이 떨어진다고 본다.

핵심 결과는 다음과 같다.

* 제안 방법은 **모든 실험에서 baseline을 능가**했다.
* 특히 **weakly supervised 모델에서 큰 격차**를 보였다.
* 저자들은 이것이 더 좋은 feature learning과 더 정확한 geometry 덕분이라고 해석한다.

또한 Figure 6 비교에서는 optimal $k$-Center와 greedy/2-OPT 근사해를 비교한다. MIP 해가 약간 더 높은 정확도를 보이지만, 근사해도 여전히 다른 baseline보다 낫다. 따라서 데이터셋 규모가 매우 크지 않다면 optimal solver가 더 좋고, 매우 커도 greedy 방식으로 충분히 실용적이라는 것이 논문의 결론이다.

정량 수치가 현재 첨부 HTML의 검색 스니펫에 완전하게 노출되지는 않았지만, 논문이 직접 서술하는 실험 메시지는 꽤 분명하다. **batch CNN active learning에서는 representative coverage가 uncertainty보다 중요하며, core-set/$k$-Center 접근이 state-of-the-art를 큰 폭으로 앞선다**는 것이다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 문제 설정을 정확히 짚었다는 점이다. 많은 active learning 방법이 “불확실한 점을 고르면 된다”는 전제를 두지만, 저자들은 CNN batch training에서는 이 전제가 깨진다고 본다. 이 지적은 이후 딥 active learning 연구 흐름에서 diversity, coverage, coreset 계열 방법이 중요해지는 방향과도 잘 맞는다.

둘째, 방법이 단순하면서도 설득력이 있다. core-set → geometric bound → $k$-Center 환원이라는 흐름은 직관적이고 구현 가능하다. 실제로 greedy와 MIP 모두 practical하다는 점도 장점이다.

셋째, weakly supervised setting까지 함께 다룬 점이 좋다. 단지 fully labeled classification만 본 것이 아니라, representation quality가 선택 성능에 어떤 영향을 주는지도 보여 준다.  

### 한계

가장 뚜렷한 한계는 이론과 실제 학습 loss 사이의 간극이다. 이론 파트는 $l_2$ loss와 zero training error 가정 위에 세워져 있지만, 실제 분류 실험은 cross-entropy로 진행된다. 저자들도 이를 인정하며, 이론이 실험 세팅을 완전히 포괄하지는 못한다.

둘째, 거리 기반 대표성은 feature space quality에 크게 의존한다. 즉, backbone이 나쁜 representation을 만들면 geometric coverage 자체가 의미를 잃을 수 있다. 논문이 weakly supervised 쪽에서 더 잘 된다고 해석한 것도, 반대로 말하면 representation quality에 민감하다는 뜻이다.

셋째, 이 방법은 기본적으로 **representativeness** 를 중시하므로, class imbalance나 rare-but-important class를 직접 겨냥하는 uncertainty/exploitation 효과는 상대적으로 약할 수 있다. 논문은 coverage의 장점을 잘 보였지만, “어떤 상황에서 uncertainty가 더 유리할 수 있는가”까지는 깊게 다루지 않는다.

### 해석

비판적으로 보면, 이 논문의 진짜 공헌은 하나의 알고리즘보다도 **deep active learning의 관점을 바꾼 것**에 있다. 즉, CNN의 active learning은 더 이상 “모델이 헷갈리는 점 찾기”가 아니라, “현재 feature space를 가장 잘 대표하는 subset 찾기”로 봐야 한다는 프레임 전환이다. 이후 BADGE, VAAL, coreset류 방법들이 대표성과 diversity를 적극적으로 끌어들이는 흐름을 생각하면, 이 논문은 그 출발점 가운데 하나라고 볼 수 있다.

## 6. Conclusion

이 논문은 CNN batch active learning에서 기존 uncertainty-based heuristic이 잘 작동하지 않는 이유를 batch correlation으로 설명하고, 이를 해결하기 위해 active learning을 core-set selection으로 재정의했다. 이론적으로는 CNN loss와 데이터 geometry를 연결하는 bound를 제시하고, 실제 선택 문제를 $k$-Center로 환원해 greedy/MIP 방식으로 해결한다. 실험적으로는 fully supervised와 weakly supervised image classification 모두에서 기존 방법보다 우수한 성능을 보였고, 특히 representation이 좋아질수록 geometric selection의 이점이 커진다고 해석한다.  

실무적으로 이 연구가 중요한 이유는, 대규모 CNN active learning에서 단순 uncertainty가 아니라 **coverage와 diversity가 핵심 설계 원리**라는 점을 비교적 이른 시기에 명확히 보여 주었기 때문이다. 그래서 이 논문은 단지 한 편의 성능 개선 논문이 아니라, 이후 딥 active learning 연구의 기본 문제 설정을 정리한 논문으로 읽는 것이 더 적절하다.
