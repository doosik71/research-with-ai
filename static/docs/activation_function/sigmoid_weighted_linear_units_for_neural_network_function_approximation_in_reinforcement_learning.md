# Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning

## 1. Paper Overview

이 논문은 강화학습에서 neural network를 function approximator로 사용할 때, 어떤 activation function과 학습 방식이 실제로 더 잘 작동하는지를 다룬다. 저자들의 목표는 두 가지다. 첫째, **SiLU (sigmoid-weighted linear unit)**와 그 도함수 기반 activation인 **dSiLU** 를 제안하는 것이다. 둘째, experience replay와 target network에 의존하는 DQN류 접근 대신, **on-policy 학습 + eligibility traces + softmax action selection** 같은 더 전통적인 조합도 충분히 경쟁력이 있을 수 있음을 보이는 것이다. 논문은 이 조합을 stochastic SZ-Tetris, 10×10 Tetris, Atari 2600에 적용해 강한 결과를 보고한다.  

이 문제가 중요한 이유는 명확하다. 당시 deep reinforcement learning의 표준은 DQN이었고, experience replay와 separate target network가 거의 필수 구성처럼 여겨졌다. 저자들은 여기에 도전해, **activation 선택과 on-policy 업데이트 구조만 바꿔도** 강한 성능을 낼 수 있다고 주장한다. 즉, 이 논문은 단순 activation proposal이 아니라, 강화학습에서 “어떤 activation이 value approximation에 잘 맞는가”와 “DQN류의 복잡한 안정화 장치가 꼭 필요한가”를 동시에 묻는 논문이다.  

## 2. Core Idea

핵심 아이디어는 두 축으로 나뉜다.

첫째, **SiLU와 dSiLU** 다. SiLU는 입력에 sigmoid gate를 곱한 형태로, ReLU의 연속적이고 부드러운 변형처럼 동작한다. dSiLU는 SiLU의 도함수를 activation처럼 사용하는 함수이며, 저자들은 이를 더 가파르고 약간 overshooting된 sigmoid형 함수로 해석한다. 논문은 특히 강화학습의 value function approximation에서 이 두 함수가 ReLU나 sigmoid보다 더 적합할 수 있다고 본다.  

둘째, **전통적 on-policy 강화학습의 재평가** 다. 저자들은 TD($\lambda$)와 Sarsa($\lambda$)를 neural network와 결합하고, 행동 선택은 simple annealing이 있는 softmax로 수행한다. 이 방식은 DQN처럼 replay buffer나 separate target network를 쓰지 않지만, max operator 기반 Q-learning이 가진 overestimation 문제를 피할 수 있고, 실제로 특정 환경에서는 더 안정적으로 잘 동작한다고 해석한다.  

## 3. Detailed Method Explanation

### 3.1 강화학습 알고리즘

논문은 두 알고리즘을 사용한다.

* **TD($\lambda$)**: state-value function $V^\pi$ 를 학습
* **Sarsa($\lambda$)**: action-value function $Q^\pi$ 를 학습

파라미터 업데이트는 일반적인 eligibility trace 기반 gradient descent 형태다.

$$
\boldsymbol{\theta}_{t+1}
=========================

\boldsymbol{\theta}_t
+
\alpha \delta_t \boldsymbol{e}_t
$$

TD($\lambda$)에서 TD error는

$$
\delta_t
========

r_t + \gamma V_t(s_{t+1}) - V_t(s_t)
$$

Sarsa($\lambda$)에서는

$$
\delta_t
========

r_t + \gamma Q_t(s_{t+1}, a_{t+1}) - Q_t(s_t, a_t)
$$

가 된다. 즉, DQN식 bootstrapped max target이 아니라, 현재 policy를 따르는 on-policy target을 사용한다.  

### 3.2 SiLU

논문에서 SiLU activation은 hidden pre-activation $z_k$ 에 대해

$$
a_k(\mathbf{s}) = z_k \sigma(z_k)
$$

로 정의된다. 여기서 $\sigma(\cdot)$ 는 sigmoid다. 저자들은 이 함수가 ReLU의 연속적이면서 약간 “undershooting”한 버전처럼 보인다고 설명한다. 큰 절댓값 영역에서는 ReLU와 비슷한 경향을 보이지만, 0 부근에서는 훨씬 부드럽게 변화한다.  

### 3.3 dSiLU

dSiLU는 SiLU의 도함수를 activation처럼 사용하는 함수다. 논문은 이를

$$
a_k(\mathbf{s})
===============

\sigma(z_k)\left(1 + z_k(1-\sigma(z_k))\right)
$$

로 제시한다. 저자들 설명에 따르면 dSiLU는 더 가파르고 약간 overshooting한 sigmoid처럼 보이며, 최대값은 약 1.1, 최소값은 약 -0.1 정도다. 즉, 단순 sigmoid보다 sharper하고, ReLU보다 부드럽지만 완전히 같은 family는 아니다.

### 3.4 행동 선택과 학습 철학

이 논문의 중요한 부분은 activation보다도 **학습 방식의 선택** 이다. 저자들은 experience replay 대신 on-policy learning과 eligibility traces를 사용하고, 행동 선택은 **softmax action selection with simple annealing** 으로 수행한다. 이는 TD-Gammon 계열의 고전적 접근을 현대 deep RL에 다시 적용한 형태다. 저자들은 특히 Q-learning 계열의 max target이 과대추정(overestimation)을 만들 수 있는 반면, TD($\lambda$)와 Sarsa($\lambda$)는 이런 max operator를 target 계산에 직접 사용하지 않는다는 점을 장점으로 해석한다.  

### 3.5 네트워크 구조

실험마다 구조가 다르다.

* **stochastic SZ-Tetris의 shallow setting**: hidden layer 1개, hidden unit 50개, linear output layer
* **Atari deep setting**: convolutional layers 뒤에 512개의 dSiLU fully connected hidden unit과 linear output layer가 붙는다. valid action 수에 따라 출력은 4~18개였다.

또한 Atari에서는 convolutional layers에 SiLU, fully connected layer에 dSiLU를 넣은 **SiLU-dSiLU** 조합을 사용했다.

## 4. Experiments and Findings

### 4.1 Stochastic SZ-Tetris

이 논문에서 가장 먼저 강조되는 결과는 stochastic SZ-Tetris다. 저자들은 shallow network로 SiLU, ReLU, dSiLU, sigmoid를 비교했고, **dSiLU network agent가 이전 state-of-the-art 평균 점수를 20% 향상**시켰다고 말한다. 또한 shallow agent들 간 final average score 차이는 **$p < 0.0001$** 수준으로 유의미했다고 보고한다.  

여기서 중요한 메시지는 단순히 “새 activation이 조금 좋다”가 아니다. reinforcement learning의 bootstrapped value approximation 환경에서, **dSiLU가 ReLU와 sigmoid보다 더 잘 맞는 inductive bias**를 가질 수 있다는 실증적 근거로 제시된다.

### 4.2 Deep stochastic SZ-Tetris

저자들은 raw board configuration을 상태로 사용하는 deep network agents도 학습시켰고, 이 경우 **convolution 층에 SiLU, fully connected 층에 dSiLU를 넣은 SiLU-dSiLU 조합**이 기존 최고 평균 점수를 넘었다고 주장한다. 즉, SiLU와 dSiLU를 층 역할에 따라 다르게 배치하는 것이 유효하다는 실험적 포인트도 있다.

### 4.3 10×10 Tetris

표준 Tetris의 20-high board는 학습 시간이 너무 길어 직접 적용이 어렵다고 솔직히 말하지만, 더 작은 **10×10 board**에서는 dSiLU agent가 state-of-the-art를 달성했다고 주장한다. 즉, 이 논문은 Tetris류 combinatorial control problem에서 activation choice가 실제로 큰 차이를 만든다고 본다.  

### 4.4 Atari 2600

Atari에서는 12개 게임을 unbiasedly selected subset으로 골라, SiLU-dSiLU deep Sarsa($\lambda$) agent를 **200,000 episodes** 학습시켰고, 각 실험은 2회 반복되었다. episode는 최대 30개의 no-op으로 시작할 수 있고, 최대 18,000 frames까지 진행되었다.  

결과 요약은 강하다. 논문 초록/서론 기준으로 이 방법은 **DQN normalized mean score 대비 232%**, **double DQN 대비 161%** 향상을 보였다고 주장한다. 즉, 적어도 이 12개 Atari subset에서는 replay와 target network가 없는 on-policy deep Sarsa($\lambda$) + SiLU/dSiLU 조합이 DQN류보다 더 강력했다고 본다.

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 두 가지다.

첫째, **activation proposal이 강화학습 문맥과 밀접하게 결합**되어 있다는 점이다. SiLU/dSiLU는 단순한 범용 activation 후보로 제시된 것이 아니라, bootstrapped value approximation과 on-policy trace-based learning에서 잘 맞는 함수로 제안되었다. 실제로 Tetris와 Atari 모두에서 강한 결과를 보인다는 점이 이를 뒷받침한다.

둘째, **DQN 대안으로서의 on-policy deep RL** 을 진지하게 보여 줬다는 점이다. 논문은 max-based Q-learning이 과대추정을 유발할 수 있다는 기존 지적을 활용해, TD($\lambda$)/Sarsa($\lambda$) 계열이 특정 설정에서는 더 나을 수 있다고 해석한다. 이건 activation보다도 오히려 더 큰 개념적 메시지다.

### 한계

한계도 있다.

첫째, Atari 비교는 **12개 게임 subset** 기준이다. 따라서 “모든 Atari에서 DQN보다 우월하다”로 일반화하기는 어렵다. 논문도 12 unbiasedly selected games라는 점을 명시한다.

둘째, 이 방법은 Tetris의 표준 20-high board에는 학습 시간이 너무 길어 적용하기 어렵다고 직접 인정한다. 즉, 성능 잠재력은 높지만 sample efficiency나 wall-clock 효율성 면에서는 제약이 있다.

셋째, SiLU/dSiLU의 강점이 activation 자체 때문인지, on-policy trace learning과의 상호작용 때문인지 완전히 분리해 설명하진 않는다. 실험은 설득력 있지만, mechanistic explanation은 비교적 제한적이다.

### 해석

비판적으로 읽으면, 이 논문의 가장 큰 의미는 **SiLU를 처음 체계적으로 실용 activation으로 보여 준 초기 사례**라는 점과, **deep RL의 표준 recipe가 반드시 experience replay + target network여야 하는 것은 아니다**라는 점을 함께 드러낸 데 있다. 이후 SiLU/Swish 계열 activation이 널리 퍼진 것을 생각하면, 이 논문은 강화학습 맥락에서 그 잠재력을 먼저 실증한 흥미로운 선행 작업으로 볼 수 있다. 이 마지막 문장은 논문의 결과를 바탕으로 한 해석이다.

## 6. Conclusion

이 논문은 강화학습용 neural function approximation을 위해 **SiLU** 와 **dSiLU** 를 제안하고, 동시에 **on-policy TD($\lambda$)/Sarsa($\lambda$) + softmax annealing** 이 DQN류의 복잡한 안정화 장치 없이도 충분히 경쟁력 있을 수 있음을 보였다. SiLU는

$$
a_k(\mathbf{s}) = z_k \sigma(z_k)
$$

형태의 self-gated linear unit이고, dSiLU는 그 도함수 기반 activation이다. 실험적으로는 stochastic SZ-Tetris, 10×10 Tetris, Atari 2600 subset에서 강한 결과를 내며, 특히 dSiLU와 SiLU-dSiLU 조합이 ReLU, sigmoid, DQN류 baseline을 능가한다고 주장한다.

정리하면, 이 논문은 activation function 연구와 deep reinforcement learning 연구를 함께 밀어붙인 논문이다. 하나는 **SiLU 계열 activation의 초기 실증**, 다른 하나는 **전통적 on-policy trace learning의 현대적 재평가**다. 두 메시지 모두 후속 연구에 의미 있는 출발점을 제공했다.
