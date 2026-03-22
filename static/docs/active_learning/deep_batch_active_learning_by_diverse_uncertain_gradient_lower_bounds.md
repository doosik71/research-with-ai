# Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds

## 1. Paper Overview

이 논문은 deep neural network를 사용하는 **batch active learning** 문제를 다룬다. 핵심 질문은 간단하다. “라벨링 예산이 제한된 상황에서, unlabeled pool에서 어떤 샘플들을 한 번에 골라야 가장 효율적으로 성능을 올릴 수 있는가?” 저자들은 기존 방법들이 대체로 **uncertainty만 보거나**, 혹은 **diversity만 보는** 식으로 설계되어 있어서 실제 batch selection에서는 한쪽이 쉽게 무너진다고 지적한다. uncertainty만 따르면 서로 거의 비슷한 샘플을 한 배치에 여러 개 뽑기 쉽고, diversity만 따르면 모델에 별로 도움이 안 되는 샘플을 고를 수 있다.  

이를 해결하기 위해 제안된 방법이 **BADGE (Batch Active learning by Diverse Gradient Embeddings)**이다. BADGE는 각 unlabeled sample을 “hallucinated gradient” 공간에 임베딩한 뒤, 그 gradient의 **크기(magnitude)**로 uncertainty를 반영하고, **방향 및 거리(directional spread / distance)**로 diversity를 반영한다. 그리고 이 임베딩들 위에서 **k-means++ initialization** 을 이용해 한 배치를 고른다. 저자들의 주장은 분명하다. BADGE는 uncertainty와 diversity를 동시에 반영하면서도, 별도의 hand-tuned hyperparameter 없이 실전적으로 잘 작동한다는 것이다.  

논문의 실질적 기여는 단순히 “새 acquisition score 하나 만들었다”가 아니다. 오히려 **batch active learning에서 uncertainty–diversity trade-off를 gradient embedding 하나로 묶었다**는 점이 중요하다. 초록과 서론에 따르면, BADGE는 다양한 architecture, batch size, dataset에서 기존 강한 baseline들과 비교해 대체로 같거나 더 좋은 성능을 보이며, 실제 현업 active learning에서 쓸 만한 general-purpose option을 목표로 한다.  

## 2. Core Idea

이 논문의 핵심 아이디어는 다음 한 문장으로 요약할 수 있다.

**“불확실한 샘플이면서 동시에 서로 다른 방향의 정보를 주는 샘플들을, gradient embedding 공간에서 한 번에 고르자.”**  

기존 active learning에서는 보통 least confidence, entropy 같은 uncertainty criterion이 널리 쓰인다. 하지만 batch setting에서는 이런 방식이 매우 비효율적일 수 있다. 모델이 불확실해하는 샘플들이 서로 거의 같은 이미지라면, 한 번에 여러 장을 라벨링해도 정보 중복이 크다. 반대로 diversity-based selection은 서로 다른 샘플을 고를 수는 있지만, 정작 decision boundary를 개선하는 데 중요한 샘플이 아닐 수 있다. BADGE는 이 둘을 따로 최적화하지 않고, **gradient embedding** 이라는 단일 표현으로 합쳐 버린다.

여기서 gradient magnitude는 “이 샘플이 현재 모델을 얼마나 크게 흔들 수 있는가”를 나타내므로 uncertainty와 연결된다. 반면 여러 샘플의 gradient 방향이 서로 다르면, 이 샘플들은 모델을 서로 다른 방식으로 업데이트하게 되므로 diversity를 반영한다고 볼 수 있다. BADGE는 이 성질을 이용해, **큰 gradient를 가지되 서로 멀리 떨어진** 샘플들을 batch로 선택한다.  

## 3. Detailed Method Explanation

### 3.1 문제 설정

논문은 pool-based active learning 설정을 사용한다. 모델은 unlabeled pool (U) 를 가지고 있고, 이 중 일부 샘플의 label을 query할 수 있다. 목표는 가능한 적은 query로 낮은 expected error를 갖는 classifier를 얻는 것이다. 논문은 multiclass classification을 전제로 하며, classifier는 fixed architecture의 neural network (f(x;\theta)) 로 parameterized된다. 예측은 다음처럼 주어진다.

$$
h_{\theta}(x)=\arg\max_{y\in[K]} f(x;\theta)_y
$$

또한 supervised training은 cross-entropy loss를 최소화하는 방식으로 이루어진다. 이 정식화는 BADGE가 특별한 모델 family를 요구하는 것이 아니라, 일반적인 neural multiclass classifier 위에 얹히는 acquisition method임을 보여준다.  

### 3.2 uncertainty-only와 diversity-only의 한계

저자들은 먼저 왜 기존 접근이 불충분한지 설명한다. uncertainty-only 방식은 batch AL에서 병적인 상황을 만든다. 모델이 비슷한 샘플들에 동시에 불확실할 경우, 선택된 배치 전체가 거의 duplicate처럼 될 수 있다. 반대로 diversity-only selection은 모델이 현재 가장 헷갈려하는 영역을 놓칠 수 있다. 특히 저자들은 architecture, batch size, dataset에 따라 어떤 기준이 잘 듣는지가 달라지므로, 실제 상황에서 “어떤 active learning algorithm이 맞는지” 미리 알기 어렵다고 지적한다. 더구나 active learning에서 hyperparameter sweep은 라벨 비용을 실제로 더 쓰게 만들기 때문에, 알고리즘은 fixed hyperparameter로도 “그냥 잘 작동해야 한다”고 강조한다.

### 3.3 hallucinated gradient embedding

BADGE의 가장 중요한 구성요소는 **hallucinated gradient** 다. 각 unlabeled point (x) 에 대해, 먼저 현재 모델의 예측 label (\hat y) 를 사용한다. 즉, 아직 정답 label은 모르지만, 모델이 가장 가능성이 높다고 보는 class를 임시 label처럼 놓는다. 그다음 이 pseudo-label에 대한 loss의 gradient를 계산해, 그 샘플이 만들어낼 업데이트 방향을 근사한다. 서론에서 저자들은 uncertainty를 **final output layer parameter에 대한 gradient magnitude** 로 측정한다고 설명한다.

직관적으로 보면 이 gradient embedding은 두 정보를 동시에 담는다.

첫째, **gradient의 크기** 가 크다는 것은 그 샘플이 현재 모델에 대해 더 “영향력 있는” 샘플이라는 뜻이다. 보통 이는 모델이 그 샘플에 대해 아직 충분히 확신하지 못한다는 신호로 해석된다.

둘째, **gradient의 방향** 이 다르다는 것은 그 샘플들이 모델을 서로 다른 방향으로 업데이트한다는 뜻이다. 그래서 서로 멀리 떨어진 gradient embedding들을 함께 고르면 batch diversity를 자연스럽게 확보할 수 있다.  

### 3.4 k-means++ initialization을 이용한 batch selection

BADGE는 이 hallucinated gradient 공간 위에서 **k-means++ initialization** 을 수행해 batch를 만든다. 중요한 점은 논문이 전체 k-means clustering을 끝까지 수행하는 것이 아니라, initialization step 자체를 selection mechanism으로 활용한다는 점이다. k-means++는 이미 선택된 중심들과 멀리 떨어진 포인트를 더 뽑을 가능성이 높기 때문에 diversity를 확보한다. 동시에 gradient norm이 큰 샘플들은 embedding 공간에서 더 중요한 후보가 되므로 uncertainty도 반영된다. 저자들은 이를 통해 BADGE가 batch 안에서 uncertainty와 diversity를 동시에 trade off한다고 설명한다.  

이 설계의 미덕은, uncertainty score 하나 뽑고 나중에 diversity re-ranking을 하는 식의 2-stage heuristic보다 훨씬 더 통합적이라는 점이다. BADGE에서는 selection criterion 자체가 “큰 gradient이면서 서로 다른 gradient”인 샘플을 선호하도록 짜여 있다. 그리고 초록에 따르면 이 과정은 별도의 hand-tuned hyperparameter를 요구하지 않는다.

### 3.5 왜 gradient embedding이 유효한가

이 논문의 직관은 매우 설득력 있다. 최종적으로 supervised learning은 gradient update의 축적을 통해 이루어진다. 그렇다면 active learning에서 “어떤 샘플이 유용한가?”라는 질문은, “어떤 샘플이 모델 파라미터를 가장 유의미하게 바꿀 수 있는가?”라는 질문으로 옮겨갈 수 있다. BADGE는 바로 이 관점을 택한다.

특히 final-layer gradient는 계산이 비교적 간단하고, 분류 모델의 class-level uncertainty를 직접 반영한다. 모델이 특정 샘플을 애매하게 보고 있다면 predicted label을 기준으로 한 gradient magnitude가 커질 수 있고, 샘플 간 representation이 다르면 gradient direction도 달라진다. 즉, BADGE는 feature space diversity가 아니라 **parameter update space diversity** 를 이용한다는 점에서 더 task-aware한 selection으로 볼 수 있다. 이 해석은 논문 서론의 설명에 근거한 합리적 정리다.

## 4. Experiments and Findings

논문 초록과 서론에서 명시적으로 드러나는 실험 메시지는 분명하다. BADGE는 **architecture choice, batch size, dataset** 이 달라져도 전반적으로 안정적으로 잘 작동하며, 실험 전반에서 “best baseline과 같거나 더 좋은” 결과를 보인다고 주장한다. 즉, 특정 환경에서만 우연히 잘 되는 specialized trick이 아니라, 꽤 robust한 batch active learning 방법이라는 것이 실험의 핵심 결론이다.  

이 논문에서 특히 중요한 실험적 포인트는 성능 숫자 하나보다도 **일관성(consistency)** 이다. 저자들은 uncertainty-only 혹은 diversity-only 방법이 데이터셋이나 배치 크기, 모델 구조에 따라 성능이 크게 흔들린다고 문제 삼는다. 따라서 BADGE의 실험은 “어떤 조건에서 최고냐”보다 “조건이 바뀌어도 무너지지 않느냐”를 보여주는 방향으로 읽는 것이 맞다. 초록의 표현도 “sometimes succeed”하는 기존 방법과 대비해, BADGE가 “consistently performs as well as or better”하다고 말한다.

다만 현재 대화에서 확보된 파일 조각은 주로 초록, 서론, 설정 부분이어서, 개별 데이터셋 이름이나 세부 baseline, 정확한 수치까지는 전부 확인되지는 않는다. 그래서 이 보고서에서는 실험 결과를 과장하지 않고, **논문이 명시적으로 강조한 high-level finding** 위주로 정리한다. 즉,

* batch active learning에서 uncertainty-only의 redundancy 문제를 줄이고,
* diversity-only의 비정보성 문제를 피하며,
* 추가 hyperparameter tuning 없이도,
* 여러 환경에서 robust한 성능을 보였다는 점이 핵심이다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **문제 정의와 방법 설계가 정확히 맞물린다**는 점이다. batch active learning의 본질적 문제는 uncertainty와 diversity를 함께 다뤄야 한다는 데 있다. BADGE는 이를 gradient embedding 하나로 자연스럽게 합쳤다. 단순히 두 개의 heuristic을 이어붙인 것이 아니라, 모델 업데이트 관점에서 두 기준을 통합한 점이 이 논문의 가장 우아한 부분이다.  

두 번째 강점은 **hyperparameter-free 성격** 이다. 논문은 BADGE가 uncertainty–diversity trade-off를 별도 hand tuning 없이 수행한다고 강조한다. active learning은 hyperparameter tuning 자체가 라벨 비용을 증가시킬 수 있으므로, 이 점은 실용적으로 매우 중요하다. 실제 서비스나 산업 현장에서는 “무난하게 잘 되는 default”가 큰 가치가 있다.  

세 번째 강점은 **gradient space** 를 사용한다는 점이다. 많은 diversity method는 feature space geometry에 의존하는데, BADGE는 “이 샘플이 모델을 어떻게 바꿀 것인가”라는 더 직접적인 관점을 취한다. 따라서 task relevance가 더 높다고 해석할 수 있다. 이 점에서 BADGE는 representation learning과 active learning을 보다 밀접하게 연결하는 방법으로 볼 수 있다.

### 한계

한계도 분명하다. 우선 BADGE는 **모델의 현재 예측 label을 pseudo-label로 사용한 gradient** 에 의존한다. 즉, hallucinated gradient는 진짜 정답이 아니라 모델의 현재 belief를 바탕으로 계산된다. 모델이 초기에 크게 틀려 있으면, gradient embedding 자체도 왜곡될 가능성이 있다. 논문 제목의 “Lower Bounds” 표현도 이런 근사적 성격과 관련 있는 해석이 가능하다. 이 부분은 BADGE의 효율성과 동시에 근사적 한계이기도 하다.  

또한 gradient embedding 기반 선택은 final-layer gradient를 중심으로 설계되므로, deeper representation 전체의 불확실성을 완전히 포착한다고 보기는 어렵다. 계산 효율과 실용성을 위해 좋은 절충을 택한 것이지만, 더 복잡한 Bayesian uncertainty estimation이나 full-network sensitivity와 비교하면 근사라는 성격이 남는다. 이 평가는 논문 구조와 방법 설명에 근거한 해석이다.

마지막으로, BADGE가 robust하다고는 해도 모든 상황에서 최적이라는 보장은 없다. 논문 자체도 기존 방법들이 특정 batch size나 architecture에서는 잘 될 수 있다고 인정한다. 따라서 BADGE는 “언제나 압도적 1등”이라기보다, **환경이 불확실할 때 믿고 쓸 수 있는 강한 기본 선택지** 로 이해하는 편이 더 정확하다.

### 해석

비판적으로 보면, 이 논문의 진짜 공헌은 새로운 acquisition heuristic 그 자체보다, **batch active learning을 parameter update geometry 관점에서 재정의했다**는 데 있다. uncertainty는 gradient norm으로, diversity는 gradient direction dispersion으로 본다는 관점은 매우 직관적이면서도 일반적이다. 그래서 BADGE는 이후 deep active learning 문헌에서도 자주 인용되는 기준점 역할을 하게 된다. 이 해석은 논문의 문제의식과 방법의 구조를 가장 잘 드러낸다.

## 6. Conclusion

이 논문은 batch active learning에서 오래된 난제인 **uncertainty와 diversity의 동시 고려** 를 매우 간결하게 해결하려는 시도다. BADGE는 unlabeled sample을 hallucinated gradient embedding으로 표현하고, 그 위에서 k-means++ initialization을 수행함으로써, 정보량이 크면서도 서로 중복되지 않는 샘플을 선택한다. 이 접근은 이론적으로도 납득 가능하고, 실용적으로도 하이퍼파라미터 부담이 적다는 장점이 있다.  

결국 BADGE의 핵심 메시지는 명확하다.
**좋은 batch active learning은 단순히 “가장 헷갈리는 샘플”을 모으는 것이 아니라, “모델을 서로 다른 방향으로 많이 움직일 샘플들”을 고르는 것이다.** 이 점에서 BADGE는 deep active learning 분야에서 매우 설득력 있고 실전적인 기준점으로 읽을 수 있다.
