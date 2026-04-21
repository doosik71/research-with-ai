# Meta-SGD: Learning to Learn Quickly for Few-Shot Learning

- **저자**: Zhenguo Li, Fengwei Zhou, Fei Chen, Hang Li
- **발표연도**: 2017
- **arXiv**: https://arxiv.org/abs/1707.09835

## 1. 논문 개요

이 논문은 few-shot learning 문제를 위해 설계된 meta-learner인 **Meta-SGD**를 제안한다. 목표는 새로운 태스크가 주어졌을 때, 매우 적은 수의 예제만으로도 빠르고 정확하게 적응할 수 있는 학습 전략을 메타 수준에서 학습하는 것이다. 저자들은 기존의 일반적인 딥러닝 방식이 각 태스크를 독립적으로, 그리고 처음부터 다시 학습한다는 점을 문제로 본다. 이런 방식은 많은 데이터와 많은 업데이트 스텝을 필요로 하므로, 데이터가 적거나 빠른 적응이 필요한 환경에서는 비효율적이다.

논문이 다루는 핵심 연구 문제는 다음과 같다. 새로운 태스크에 대해 학습기를 어떻게 초기화할지, 어떤 방향으로 업데이트할지, 그리고 얼마나 큰 step size로 업데이트할지를 few-shot 환경에 맞게 **자동으로 학습**할 수 있는가이다. 저자들은 이 세 요소를 사람이 수동으로 정하는 대신, 태스크들의 분포로부터 end-to-end로 학습하는 meta-learning 접근이 필요하다고 주장한다.

이 문제가 중요한 이유는 분명하다. few-shot learning은 이미지 분류, 회귀, 강화학습처럼 서로 다른 문제 영역에서 공통적으로 등장하며, 특히 자율주행, 로보틱스, 동적 환경 적응처럼 **빠른 학습과 적은 데이터**가 필수인 상황에서 중요하다. 따라서 적은 데이터로도 빠르게 일반화 가능한 학습 알고리즘을 설계하는 것은 실용성과 학문적 가치가 모두 크다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 단순하다. **SGD처럼 보이지만, SGD의 핵심 요소들을 모두 meta-learning으로 학습하자**는 것이다. 보통 SGD에서는 초기 파라미터는 랜덤하게 정하고, 업데이트 방향은 gradient가 결정하며, learning rate는 사람이 하이퍼파라미터로 정한다. 반면 Meta-SGD는 이 세 가지를 모두 학습한다.

구체적으로, Meta-SGD는 learner의 초기 파라미터인 $\theta$와, 같은 크기의 벡터 $\alpha$를 함께 학습한다. 여기서 $\alpha$는 단순 learning rate 스칼라가 아니라 **파라미터별(element-wise) 업데이트를 제어하는 벡터**다. 따라서 $\alpha$는 업데이트 크기뿐 아니라 실제 업데이트 방향까지도 바꿀 수 있다. 즉, gradient를 그대로 따르지 않고, 메타 학습을 통해 few-shot 일반화에 더 유리한 방향으로 조정한다.

기존 접근과의 차별점은 논문에서 분명히 제시된다.

MAML은 learner의 초기값만 meta-learning으로 학습하고, 실제 적응은 일반 gradient descent에 맡긴다. 따라서 update direction과 learning rate는 사실상 hand-designed되어 있다. 반면 Meta-SGD는 초기값, 업데이트 방향, learning rate를 모두 학습하므로 더 높은 capacity를 가진다고 주장한다.

Meta-LSTM은 이 세 요소를 모두 학습한다는 점에서는 비슷하지만, LSTM 기반 meta-learner라서 구조가 더 복잡하고 학습이 어렵다. 논문은 특히 Meta-LSTM이 실제로는 learner의 각 파라미터를 독립적으로 갱신하는 식으로 구현되어 복잡도는 높은데 잠재력은 제한될 수 있다고 지적한다. Meta-SGD는 이보다 개념적으로 단순하고 구현이 쉽고, 학습도 더 효율적이라는 점을 장점으로 내세운다.

## 3. 상세 방법 설명

Meta-SGD는 supervised learning과 reinforcement learning 모두에 적용 가능한 meta-learner로 제안된다. 논문의 핵심 식은 다음과 같다.

일반적인 gradient descent는 다음과 같이 learner를 반복적으로 갱신한다.

$$
\theta_t = \theta_{t-1} - \alpha \nabla L_T(\theta_{t-1})
$$

여기서 $L_T(\theta)$는 태스크 $T$에서의 empirical loss이고, $\alpha$는 보통 사람이 정하는 learning rate다.

논문이 제안하는 Meta-SGD의 적응 규칙은 다음과 같다.

$$
\theta' = \theta - \alpha \circ \nabla L_T(\theta)
$$

여기서 $\circ$는 element-wise product를 의미한다. 이 식의 의미는 중요하다.

- $\theta$는 단순한 현재 파라미터가 아니라, 새로운 태스크에 대해 learner를 시작시키는 **meta-learned initialization**이다.
- $\alpha$는 각 파라미터별 스텝 크기를 조절하는 벡터이며, 동시에 gradient 성분별 가중을 통해 **업데이트 방향 자체도 바꾸는 역할**을 한다.
- 따라서 $\alpha \circ \nabla L_T(\theta)$는 단순한 learning rate scaling이 아니라, few-shot 일반화에 적합하도록 학습된 적응 규칙이다.

논문은 이 점을 특히 강조한다. 업데이트 벡터의 방향이 일반 gradient $\nabla L_T(\theta)$와 다를 수 있으므로, Meta-SGD는 SGD처럼 gradient에 의존하면서도 실제로는 **gradient descent와 다른 방향으로 learner를 이동시킬 수 있다**. 이 때문에 few-shot 상황에서 empirical fitting보다 generalization에 더 적합한 적응이 가능하다는 것이 저자들의 주장이다.

### 메타 학습 목표

메타 학습에서는 각 태스크 $T$가 `train(T)`와 `test(T)`로 나뉜다. Meta-SGD는 `train(T)`를 사용해 learner를 한 번 적응시키고, `test(T)`에서의 loss가 작아지도록 $\theta$와 $\alpha$를 학습한다. 목적함수는 다음과 같다.

$$
\min_{\theta,\alpha} \; \mathbb{E}_{T \sim p(T)} \left[L_{\text{test}(T)}(\theta')\right]
= \mathbb{E}_{T \sim p(T)} \left[L_{\text{test}(T)}\left(\theta - \alpha \circ \nabla L_{\text{train}(T)}(\theta)\right)\right]
$$

즉, 태스크 분포 $p(T)$에서 샘플된 여러 태스크에 대해, `train` set으로 한 번 적응한 뒤 `test` set에서 잘 일반화하도록 meta-parameters를 학습한다. 중요한 점은 이 목적함수가 $\theta$와 $\alpha$에 대해 미분 가능하므로, 메타 수준에서도 SGD로 최적화할 수 있다는 점이다.

### Supervised learning에서의 알고리즘 흐름

논문에 제시된 Algorithm 1을 풀어 쓰면 다음과 같다.

1. $\theta$와 $\alpha$를 초기화한다.
2. 태스크 분포에서 여러 태스크 $T_i$를 샘플링한다.
3. 각 태스크에 대해 `train(T_i)`에서 loss를 계산한다.
4. 그 gradient로 한 번만 적응하여 $\theta_i'$를 만든다.
5. 적응된 $\theta_i'$를 `test(T_i)`에서 평가한다.
6. 여러 태스크의 test loss를 합친 뒤, 이를 줄이도록 $\theta$와 $\alpha$를 업데이트한다.

즉, inner loop는 learner adaptation이고, outer loop는 meta-learner training이다. 논문 Figure 1과 Figure 2는 이 두 단계 구조를 시각화한다. 태스크 내에서는 빠른 학습이 일어나고, 태스크들 전체에 걸쳐서는 그 빠른 학습 전략 자체를 천천히 개선한다.

### Reinforcement learning으로의 확장

강화학습에서는 하나의 태스크를 MDP로 본다. 태스크 $T$는 $(S, A, q, q_0, T, r, \gamma)$로 구성되며, learner $f_\theta$는 stochastic policy다. 이때 loss는 discounted return의 음수로 정의된다.

$$
L_T(\theta) =
- \mathbb{E}_{s_t, a_t \sim f_\theta, q, q_0}
\left[
\sum_{t=0}^{T} \gamma^t r(s_t, a_t)
\right]
$$

Meta-SGD의 갱신 식 자체는 supervised learning과 동일하다.

$$
\theta' = \theta - \alpha \circ \nabla L_T(\theta)
$$

차이는 gradient를 계산하는 방식이다. 강화학습에서는 trajectory를 샘플링한 뒤 policy gradient로 $\nabla L_T(\theta)$를 추정한다. 이후 업데이트된 policy $f_{\theta'}$로 다시 trajectory를 샘플링하고, 그 성능을 바탕으로 meta-parameters를 최적화한다. 논문은 이 설정에서도 Meta-SGD가 적용 가능함을 보이며, supervised learning뿐 아니라 RL까지 포괄하는 meta-learner라는 점을 장점으로 제시한다.

## 4. 실험 및 결과

논문은 회귀, 분류, 강화학습의 세 영역에서 Meta-SGD를 평가한다. 모든 실험에서 공통적으로 강조되는 포인트는 **one-step adaptation**이다. 즉, 새로운 태스크에 대해 단 한 번의 gradient-based update만으로도 좋은 성능을 낸다는 점을 검증하려 한다.

### 회귀 실험

회귀 실험은 sine wave regression이다. 목표 함수는 $y(x) = A \sin(\omega x + b)$이며, amplitude $A$, frequency $\omega$, phase $b$를 각각 특정 구간의 uniform distribution에서 샘플링한다. 입력 $x$의 범위는 $[-5, 5]$다. 각 태스크는 하나의 sine curve를 맞추는 문제이며, $K \in \{5, 10, 20\}$ shot 설정을 사용한다.

모델은 입력 1차원, hidden layer 두 개(각 40 unit, ReLU), 출력 1차원인 작은 MLP를 사용한다. 성능 지표는 MSE다. MAML은 learning rate를 고정된 $\alpha = 0.01$로 사용했고, Meta-SGD는 $\alpha$의 모든 원소를 같은 초기값으로 두되 $[0.005, 0.1]$ 구간에서 랜덤 초기화했다. 두 모델 모두 one-step adaptation이며, 60000 iteration 동안 meta-training했다.

결과는 Table 1에 제시되어 있다. 모든 meta-training shot 수와 meta-testing shot 수 조합에서 Meta-SGD가 MAML보다 낮은 MSE를 기록한다. 예를 들어 10-shot meta-training 후 20-shot meta-testing에서는 MAML이 $0.56 \pm 0.08$인 반면, Meta-SGD는 $0.35 \pm 0.06$이다. 20-shot meta-training 후 20-shot meta-testing에서도 MAML은 $0.48 \pm 0.08$, Meta-SGD는 $0.31 \pm 0.05$다.

이 결과는 단순히 initialization만 배우는 것보다, update direction과 learning rate까지 같이 배우는 것이 회귀 문제의 구조를 더 잘 포착한다는 논문의 주장과 일치한다. 논문은 MAML의 learning rate를 $0.01$에서 $0.1$로 바꾸면 성능이 더 나빠진다고도 보고한다. 이는 few-shot 환경에서 learning rate를 사람이 고정하는 방식이 민감하고 불안정할 수 있음을 보여주는 근거로 사용된다.

Figure 3의 정성적 결과도 같은 메시지를 준다. 5-shot만 보고도 Meta-SGD는 MAML보다 더 빠르게 sine curve의 형태에 적응한다. 특히 예제가 입력 구간의 절반에만 있어도 더 잘 일반화한다는 점을 보여준다.

### 분류 실험

분류에서는 Omniglot과 MiniImagenet 두 벤치마크를 사용한다.

Omniglot은 50개 알파벳의 1623개 문자 클래스로 이루어진 데이터셋이며, 1200개 문자를 meta-training에, 나머지를 meta-testing에 사용했다. 5-way와 20-way 분류를 각각 1-shot, 5-shot 설정으로 평가했다.

MiniImagenet은 100개 클래스, 클래스당 600장 이미지로 구성되며, 64/16/20 클래스를 meta-training/meta-validation/meta-testing으로 나눴다. 마찬가지로 5-way와 20-way, 1-shot과 5-shot 설정을 고려했다.

모델은 [7]을 따라 4개의 convolution module을 사용한다. 각 모듈은 $3 \times 3$ convolution, batch normalization, ReLU, $2 \times 2$ max-pooling으로 구성된다. Omniglot은 $28 \times 28$로 축소하고 64 filters를 사용하며, convolution 뒤에 32차원 fully connected layer를 추가했다. MiniImagenet은 $84 \times 84$로 축소하고 32 filters를 사용했다. Meta-SGD는 분류에서도 one-step adaptation만 사용한다.

#### Omniglot 결과

Table 2에 따르면 Meta-SGD는 거의 모든 설정에서 기존 최고 수준과 비슷하거나 조금 더 좋다. 예를 들면 5-way 1-shot에서 $99.53 \pm 0.26\%$, 5-way 5-shot에서 $99.93 \pm 0.09\%$를 기록한다. 20-way 1-shot은 $95.93 \pm 0.38\%$, 20-way 5-shot은 $98.97 \pm 0.19\%$다. MAML과 비교하면 소폭 우위이며, Siamese Nets와 Matching Nets도 넘어선다.

흥미롭게도 저자들은 5-shot 분류 성능이 반드시 5-shot meta-training에서 가장 좋은 것은 아니라고 보고한다. Omniglot에서는 오히려 1-shot meta-training으로 학습한 모델이 5-shot meta-testing에서도 더 나았다고 적고 있다. 이는 메타 학습에서 task sampling 방식이 결과에 중요한 영향을 줄 수 있음을 시사한다.

#### MiniImagenet 결과

MiniImagenet에서는 Meta-SGD의 이점이 더 뚜렷하다. Table 3에서 5-way 1-shot은 $50.47 \pm 1.87\%$, 5-way 5-shot은 $64.03 \pm 0.94\%$로, Matching Nets, Meta-LSTM, MAML보다 높다. 더 어려운 20-way 설정에서도 1-shot은 $17.56 \pm 0.64\%$, 5-shot은 $28.92 \pm 0.35\%$로 가장 높은 수치를 기록한다.

특히 MAML은 20-way classification에서 one-step adaptation일 때 성능이 Matching Nets나 Meta-LSTM보다도 낮아진다고 논문은 보고한다. 반면 Meta-SGD는 one-step adaptation만으로도 가장 높은 정확도를 달성한다. 이는 이 논문의 핵심 주장인 “빠르면서도 성능이 좋다”를 뒷받침하는 중요한 결과다.

### 강화학습 실험

강화학습 실험은 2D navigation task다. 에이전트는 2차원 평면에서 시작 위치에서 목표 위치까지 이동해야 한다. 두 종류의 태스크 세트를 사용한다.

- 첫 번째는 시작 위치를 원점 $(0,0)$으로 고정하고 목표만 랜덤하게 바꾸는 설정
- 두 번째는 시작 위치와 목표 위치를 모두 unit square 안에서 랜덤하게 바꾸는 설정

상태는 현재 2D 위치, 행동은 다음 스텝의 속도다. 정책은 Gaussian distribution을 출력하는 신경망으로 parameterize된다. 평균은 입력 2차원, hidden layer 두 개(각 100 unit, ReLU), 출력 2차원인 작은 MLP로 계산하고, 분산은 학습 가능한 diagonal log variance 파라미터를 사용한다. 보상은 현재 상태와 목표 사이 거리의 음수다.

meta-training에서는 매 iteration마다 20개 태스크를 샘플링하고, 각 태스크에서 20개 trajectory를 수집해 vanilla policy gradient로 gradient를 계산한 뒤, Meta-SGD 규칙으로 한 번 업데이트한다. 이후 업데이트된 정책으로 다시 20개 trajectory를 모으고, 전체 태스크에 대해 TRPO로 $\theta$와 $\alpha$를 업데이트한다. 총 100 iteration 학습한다.

meta-testing에서는 600개 태스크를 샘플링하고, 각 태스크에서 적응 후 20개의 새 trajectory를 수집해 평균 return을 계산한다. Table 4에 따르면 fixed start position에서는 MAML이 $-9.12 \pm 0.66$, Meta-SGD가 $-8.64 \pm 0.68$이다. varying start position에서는 MAML이 $-10.71 \pm 0.76$, Meta-SGD가 $-10.15 \pm 0.62$다. return은 덜 음수일수록 더 좋으므로 Meta-SGD가 두 설정 모두에서 우세하다.

Figure 4의 정성적 결과에서도 적응 후 Meta-SGD가 목표 위치를 더 잘 향하는 경향이 보인다고 논문은 설명한다. 즉, supervised few-shot뿐 아니라 RL에서도 Meta-SGD의 업데이트 전략이 단순 gradient descent보다 효과적이라는 점을 주장한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 아이디어가 매우 단순하면서도 설계 논리가 명확하다는 점이다. Meta-learning에서 optimizer-like meta-learner를 생각할 때 필요한 세 요소인 initialization, update direction, learning rate를 정확히 짚고, 이를 하나의 간단한 수식으로 통합했다. 식 자체는 SGD와 거의 비슷하지만, 실제 의미는 훨씬 강력하다. 이런 단순성은 구현 난이도와 학습 안정성 면에서 실질적 장점이 된다.

두 번째 강점은 **one-step adaptation**이라는 매우 강한 제약 아래에서도 회귀, 분류, 강화학습 전반에 걸쳐 경쟁력 있는 결과를 보였다는 점이다. 특히 MiniImagenet과 2D navigation 결과는 논문의 핵심 주장을 잘 지지한다. MAML처럼 initialization만 배우는 것보다 더 높은 capacity를 가진다는 논리도 실험적으로 어느 정도 설득력 있게 제시된다.

세 번째 강점은 supervised learning과 reinforcement learning을 동일한 틀에서 설명했다는 점이다. 메타 학습 목적함수와 적응 구조가 두 영역에서 일관되게 유지되므로, 방법론의 범용성이 드러난다.

반면 한계도 분명하다. 첫째, 논문은 Meta-SGD가 왜 특정 태스크 분포에서 좋은 $\alpha$를 학습하는지 경험적으로는 보여주지만, **이론적 분석**은 제공하지 않는다. 예를 들어 learned $\alpha$가 실제로 어느 정도 update direction을 바꾸는지, 어떤 조건에서 일반 gradient descent보다 유리한지에 대한 정량적 해석은 부족하다.

둘째, 비교 실험은 주로 MAML, Meta-LSTM, Matching Nets 중심이며, 메타 학습의 더 넓은 스펙트럼과의 비교는 제한적이다. 물론 논문 시점에서는 자연스러운 선택일 수 있지만, 결과 해석은 당시 기준에 묶여 있다.

셋째, 계산 비용 측면에서 “Meta-LSTM보다 쉽다”고는 하지만, 메타 학습 자체가 일반 학습보다 훨씬 비싼 구조라는 점은 논문 결론부에서도 인정한다. 실제로 많은 태스크를 반복적으로 샘플링하고 inner adaptation까지 수행해야 하므로, 대규모 모델이나 더 복잡한 태스크로 확장할 때 비용 문제가 심각해질 수 있다.

넷째, 분류 실험에서 regularization term을 추가했다고 되어 있지만, 제공된 텍스트에는 그 구체적인 형태가 명시되어 있지 않다. 따라서 분류 성능 향상 중 일부가 어떤 regularization 설계에서 왔는지를 이 텍스트만으로는 정확히 분리해 해석할 수 없다. 마찬가지로 일부 구현 세부사항은 인용 논문 [7], [18], [25]를 따른다고만 되어 있어, 완전한 재현에 필요한 정보가 모두 본문에 직접 제시되지는 않는다.

비판적으로 보면, Meta-SGD의 성능 향상이 “초기화만 배우는 MAML보다 더 많은 자유도를 학습했기 때문”이라는 설명은 타당하지만, 자유도가 늘어난 만큼 overfitting 위험이나 task distribution shift에 대한 취약성은 어떤지 본문에서 충분히 다뤄지지 않는다. 또한 태스크 분포가 바뀌었을 때 learned $\theta$와 $\alpha$의 전이 가능성이 얼마나 되는지도 열린 문제로 남아 있다.

## 6. 결론

이 논문은 few-shot learning을 위한 간단하고 강력한 meta-learner인 Meta-SGD를 제안했다. 핵심 기여는 learner의 **initialization, update direction, learning rate**를 모두 메타 수준에서 공동 학습하는 SGD-like 규칙을 제시한 것이다. 그 결과, 새로운 태스크에 대해 단 한 번의 업데이트만으로도 회귀, 분류, 강화학습에서 강한 성능을 보였다.

실제 적용 측면에서 이 연구는 빠른 적응이 필요한 문제에 특히 중요하다. 적은 데이터, 적은 업데이트, 빠른 일반화가 필요한 환경에서는 매우 실용적인 방향을 제시한다. 향후 연구로는 논문이 직접 언급하듯이 대규모 meta-learning, unseen task/domain에 대한 일반화, 그리고 더 범용적인 meta-learner 설계가 중요할 것이다. 전체적으로 이 논문은 MAML 이후의 optimizer-based meta-learning을 한 단계 확장한 작업으로 평가할 수 있으며, few-shot learning에서 “무엇을 학습할 것인가”를 넘어 “어떻게 업데이트할 것인가”까지 본격적으로 학습 대상으로 삼았다는 점에서 의미가 크다.
