# Dying ReLU and Initialization: Theory and Numerical Examples

## 1. Paper Overview

이 논문은 ReLU 네트워크에서 잘 알려진 **dying ReLU** 현상을 경험적 직관이 아니라 **이론적으로 분석**하려는 논문이다. 저자들은 dying ReLU를 “일부 뉴런이 0만 출력하는 문제”보다 더 강한 형태로 다룬다. 즉, 네트워크 전체가 상수 함수로 붕괴하는 **dying ReLU neural network**를 정의하고, 대칭적인 초기화(symmetric initialization)를 쓰는 깊은 ReLU 네트워크는 깊이가 충분히 커질수록 높은 확률로 결국 죽게 된다는 점을 엄밀히 보인다. 이어서 이를 완화하기 위해 **RAI(Randomized Asymmetric Initialization)**라는 새로운 초기화 방법을 제안하고, 이론과 수치 실험으로 효과를 검증한다.  

이 문제가 중요한 이유는 분명하다. ReLU는 단순하고 강력하지만, 깊은 네트워크에서는 gradient가 사라지며 학습 자체가 붕괴할 수 있다. 기존 해결책은 activation 수정, batch normalization 같은 추가 기법, 혹은 구조 변경이 많았는데, 이 논문은 가장 단순한 개입인 **초기화만 바꾸는 방법**에 집중한다. 특히 “왜 아주 깊은 ReLU 네트워크는 구조적으로 학습이 어려운가”를 확률적 관점에서 설명하고, deep-and-narrow 네트워크를 실제로 살릴 수 있는 초기화 규칙을 제시한다는 점이 핵심 공헌이다.  

## 2. Core Idea

논문의 핵심 아이디어는 두 단계다.

첫째, **대칭 초기화는 깊은 ReLU 네트워크를 죽게 만들기 쉽다**는 것을 이론적으로 보인다. 가중치와 bias를 0을 중심으로 하는 symmetric distribution에서 뽑으면, 각 층에서 activation이 전부 음수 영역으로 밀려 전체가 0이 되는 사건이 반복적으로 발생할 수 있다. 이 현상은 폭이 좁고 깊이가 큰 네트워크에서 특히 심각하며, 논문은 깊이가 무한대로 갈 때 dying probability가 1로 간다는 점까지 보인다.

둘째, 이를 막기 위해 **비대칭성을 의도적으로 심은 초기화**를 사용한다. 저자들이 제안한 RAI는 각 뉴런의 파라미터들 중 하나를 양의 값을 갖는 분포에서 뽑고, 나머지는 기존 symmetric Gaussian으로 두는 식의 randomized asymmetric scheme이다. 이 작은 비대칭 덕분에 모든 뉴런이 동시에 음수 영역으로 떨어져 네트워크가 통째로 죽는 확률이 크게 줄어든다. 동시에 second moment analysis를 통해 exploding gradient가 생기지 않도록 hyperparameter도 이론적으로 설계한다.

## 3. Detailed Method Explanation

### 3.1 dying ReLU의 정의

논문은 단순히 “어떤 뉴런 하나가 죽는다”가 아니라, **네트워크 전체가 상수 함수가 되는 경우**를 worst case로 정의한다. 이 경우 이후 학습에서도 gradient 정보가 사실상 전달되지 않아 성공적인 학습을 기대할 수 없다. 저자들은 이를 born dead 혹은 dying ReLU network 관점으로 분석한다.  

논문 초반의 예시는 매우 직관적이다. 함수 $f(x)=|x|$ 는 폭 2의 2-layer ReLU 네트워크로 정확히 표현 가능하다. 실제로

$$
|x| = \mathrm{ReLU}(x) + \mathrm{ReLU}(-x)
$$

이기 때문이다. 그런데 10-layer, width 2 네트워크로 이를 학습하면 1000회 독립 실험 중 90% 이상이 상수 함수로 붕괴했다고 보고한다. 즉, **표현 가능성은 충분한데도 initialization 때문에 학습이 망가질 수 있다**는 것이 논문의 출발점이다.  

### 3.2 이론적 설정

논문은 fully connected feed-forward ReLU network를 층별 너비 $N_\ell$ 를 갖는 함수 $\mathcal N^L$ 로 정의하고, 각 층의 affine transform 뒤에 element-wise ReLU를 적용하는 표준 구조를 사용한다. 분석의 핵심은 **각 층을 지나며 activation이 전부 0이 되는 사건의 확률**을 추적하는 것이다.

이 틀에서 저자들은 symmetric initialization 하에서 born-dead probability(BDP)의 upper/lower bound를 유도하고, 깊이 $L$ 와 폭 $N$ 이 이 확률에 어떻게 작용하는지 분석한다. 주요 직관은 다음과 같다.

* 폭이 좁으면 층 하나가 완전히 죽을 확률이 커진다.
* 이런 사건이 깊이를 따라 누적되면 전체 네트워크가 죽는다.
* 따라서 깊이가 커질수록 dying probability는 증가한다.
* 반대로 폭을 키우면 같은 깊이에서 죽을 확률은 줄어든다.  

### 3.3 주요 이론 결과

논문이 가장 강하게 말하는 결과는 다음이다.

* **깊은 fully connected ReLU network는 결국 확률적으로 죽는다.**
* symmetric initialization 하에서는 depth가 무한대로 갈 때 born-dead probability가 1에 가까워진다.

또 practical guide로 매우 유용한 corollary도 제시한다. width가 $N$ 인 네트워크에 대해 dying probability를 $1-\delta$ 이하로 유지하려면 depth는 대략

$$
L = \Omega(\log_2(N/\delta))
$$

규모를 만족해야 한다고 설명한다. 즉, **안전한 depth는 width에 대해 로그 수준으로만 증가**한다. 이 결과는 왜 실제로는 deep-and-wide 네트워크가 선호되는지 설명해 준다. 하지만 폭을 키우는 것은 파라미터 수와 계산량을 크게 증가시키므로, 깊고 좁은 네트워크에는 별도 대책이 필요하다.

논문은 이를 그림으로도 해석한다. 예를 들어 width 10의 10-layer 네트워크는 collapse probability가 1% 미만이지만, width 5면 10% 이상, width 3이면 약 60% 수준까지 올라갈 수 있다고 정리한다. 이 수치는 깊고 좁은 네트워크가 왜 특히 위험한지를 직관적으로 보여 준다.

### 3.4 RAI(Randomized Asymmetric Initialization)

이 논문의 방법적 핵심은 RAI다. 기존 He initialization 같은 표준 방법은 가중치와 bias를 대부분 0 중심 대칭분포에서 뽑는다. RAI는 여기서 벗어나, **각 뉴런의 파라미터 중 하나를 비대칭적이고 양의 값을 갖는 분포에서 선택**한다. 나머지 파라미터는 Gaussian 계열을 유지한다. 이로 인해 각 뉴런은 초기부터 완전히 symmetric하지 않게 되고, 전체 층이 한 번에 음수 쪽으로 넘어가 0만 출력하는 확률이 줄어든다.  

이 아이디어는 단순하지만, 논문은 감으로 정하지 않는다. Section 4.2에서 second moment analysis를 통해 어떤 분포와 스케일을 써야 exploding gradient를 피할 수 있는지 분석한다. 즉, RAI는 “positive bias를 크게 넣자” 같은 순진한 처방이 아니라, **dying ReLU와 exploding gradient를 동시에 고려한 설계**다.  

### 3.5 기존 초기화와의 관계

논문은 orthogonal initialization, He initialization, 그리고 큰 positive bias를 넣은 He initialization도 비교한다. 결과적으로 orthogonal initialization은 He와 매우 비슷하거나 약간 나은 정도에 그쳐 dying ReLU를 근본적으로 막지 못한다. 큰 positive bias는 BDP를 줄일 수는 있지만 exploding gradient 위험이 커진다. 반면 RAI는 dying probability를 훨씬 강하게 낮추면서도 second moment 조건을 만족하도록 설계된다.

## 4. Experiments and Findings

### 4.1 1D function approximation

논문은 먼저 간단한 1차원 함수 근사에서 deep narrow ReLU network가 얼마나 자주 붕괴하는지 보여 준다. $f(x)=|x|$ 예제에서 10-layer width-2 네트워크는 1000회 독립 실험 중 90% 이상이 상수 함수로 collapse되었다. 이 결과는 “표현력 부족”이 아니라 “초기화와 optimization dynamics 문제”라는 점을 잘 보여 준다.

### 4.2 smooth function examples에서 RAI의 효과

더 어려운 smooth target function 실험에서도 RAI는 collapse를 크게 줄였다. 논문에 따르면 한 예제에서는 symmetric initialization으로 학습한 결과의 **91.9%** 가 collapse된 반면, RAI는 **29.2%** 로 줄였다. 또 일부 경우에는 initialization 시점에서는 죽지 않았지만 학습 후 부분 붕괴(partial collapse)로 가는 경우도 관찰되었다. 저자들은 이 partial collapse가 별도의 중요한 연구 주제라고 해석한다.

또 다른 함수 실험에서는 He initialization의 collapse 비율이 **93.8%**, RAI는 **32.6%** 였다. 논문은 이 차이를 “60.3 percentage point 감소”라고 강조하며, 특히 deep and narrow ReLU network에서 RAI가 dying ReLU를 효과적으로 피하게 한다고 주장한다.

### 4.3 다차원 입력/출력 함수 실험

다차원 함수 $f_4(\mathbf x)$ 실험에서도 같은 경향이 나온다. 1000회 독립 실험에서 He initialization은 **76.8%**, RAI는 **9.6%** 가 collapse되었다. 이 결과는 RAI의 효과가 단순한 1D toy problem이 아니라 다차원 입력/출력 설정에서도 유지된다는 점을 보여 준다.  

### 4.4 MNIST

마지막으로 MNIST 분류 실험을 수행한다. 얕고 넓은 네트워크(depth 2, width 1024)에서는 He와 RAI가 비슷한 test accuracy를 보였다. 반면 깊고 좁은 네트워크(depth 50, width 10)에서는 **RAI가 더 높은 test accuracy** 를 달성했다. 즉, RAI는 단순히 born-dead probability를 줄이는 것뿐 아니라, 실제 generalization 성능 면에서도 deep narrow regime에서 이점이 있음을 보인다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **dying ReLU를 처음으로 이론적으로 정식화하고 분석했다**는 점이다. ReLU가 죽는다는 경험적 관찰은 오래 있었지만, 이 논문은 깊이가 커질수록 왜 구조적으로 그런 일이 발생하는지를 확률적으로 설명한다. 또한 단순히 현상을 지적하는 데 그치지 않고, 바로 이어서 RAI라는 구체적 해결책을 제시하고 theoretical bound와 수치 실험 둘 다 제공한다.  

또 하나의 강점은 **deep-and-narrow network** 를 정면으로 다룬다는 점이다. 많은 실무 시스템은 폭을 크게 늘려 문제를 우회하지만, 이 논문은 폭 확장이 비싸고 이론적으로도 narrow network가 중요하다는 점을 분명히 한다. 그 결과, 안전한 작동 구간(safe operating region)을 width-depth 관계로 제시한 부분이 매우 실용적이다.  

### 한계

한계도 있다. 첫째, 논문이 다루는 dying ReLU는 **네트워크 전체가 상수 함수가 되는 worst case** 에 집중한다. 실제 학습에서는 일부 층 또는 일부 채널만 부분적으로 죽는 경우가 더 흔한데, 논문도 partial collapse가 자주 관찰되었다고 하면서 이를 future work로 남긴다. 즉, 현실의 모든 dying 현상을 완전히 설명한 것은 아니다.  

둘째, 분석 대상은 주로 fully connected ReLU network다. CNN, residual connection, batch normalization, modern optimizer까지 포함한 오늘날의 대형 네트워크에 결과를 그대로 이식하려면 추가 해석이 필요하다.

셋째, RAI는 dying probability를 크게 줄이지만 완전히 0으로 만들지는 않는다. 예를 들어 다차원 실험에서도 9.6% collapse가 남아 있으며, initialization 때 안 죽었더라도 학습 도중 부분 붕괴가 생길 수 있다. 따라서 RAI는 강한 개선책이지만 완전 해법은 아니다.

### 해석

비판적으로 보면, 이 논문의 가장 큰 의미는 “새 초기화 하나를 제안했다”보다도 **ReLU 네트워크의 학습 난제를 width-depth-probability의 구조적 관계로 설명했다**는 데 있다. 이후 관점에서 보면 residual connection, normalization, activation 변형 등이 모두 비슷한 문제를 다른 방식으로 해결해 왔는데, 이 논문은 그중 initialization 관점의 가장 명확한 설명 중 하나다.

## 6. Conclusion

이 논문은 dying ReLU를 단순한 경험적 현상이 아니라 **깊은 ReLU 네트워크에서 확률적으로 필연적인 현상**으로 분석했다. symmetric initialization 하에서는 depth가 충분히 커질수록 네트워크가 born dead 혹은 fully dead 상태로 갈 확률이 증가하며, 특히 deep-and-narrow regime에서 문제가 심각하다는 점을 이론적으로 보였다. 이를 해결하기 위해 제안한 **RAI(Randomized Asymmetric Initialization)**는 의도적으로 작은 비대칭성을 주입해 dying probability를 크게 낮추며, second moment analysis를 통해 exploding gradient도 함께 제어한다.  

실험적으로도 RAI는 여러 함수 근사 문제에서 He initialization 대비 collapse 확률을 크게 낮췄고, MNIST의 깊고 좁은 네트워크에서는 더 나은 test accuracy를 보였다. 따라서 이 논문은 “왜 깊은 ReLU 네트워크가 죽는가?”에 대한 이론적 답과 “그럼 어떻게 초기화해야 하는가?”에 대한 실용적 답을 함께 준 논문으로 볼 수 있다.
