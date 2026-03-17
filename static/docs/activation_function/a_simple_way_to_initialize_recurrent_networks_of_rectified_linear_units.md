# A Simple Way to Initialize Recurrent Networks of Rectified Linear Units

## 1. Paper Overview

이 논문은 RNN이 긴 시계열 의존성을 학습할 때 겪는 vanishing gradient, exploding gradient 문제를 더 단순한 방식으로 완화할 수 있는지 묻는다. 저자들은 복잡한 gated 구조를 도입하는 대신, **ReLU 기반 RNN의 recurrent weight를 identity matrix 또는 scaled identity로 초기화**하는 매우 간단한 방법을 제안한다. 핵심 주장은 이 초기화만으로도 hidden state와 gradient가 시간축을 따라 안정적으로 전달되어, 장기 의존성 문제에서 LSTM에 근접하거나 일부 toy task에서는 더 나은 성능까지 낼 수 있다는 것이다.  

이 문제가 중요한 이유는 분명하다. 당시 장기 의존성 학습을 잘 하려면 Hessian-Free optimization 같은 복잡한 최적화 기법이나 LSTM 같은 구조적 장치가 사실상 필요하다고 여겨졌는데, 저자들은 “정말로 그렇게 복잡해야만 하는가?”를 실험적으로 반박하려 한다. 즉, LSTM의 강점 중 일부는 복잡한 gating 자체보다도 **시간축에서 정보와 gradient를 보존하는 메커니즘**에 있을 수 있으며, 그 효과를 더 단순한 초기화로 상당 부분 재현할 수 있다는 문제의식을 가진 논문이다.

## 2. Core Idea

이 논문의 핵심 아이디어는 매우 직관적이다.

* hidden unit으로 tanh 대신 **ReLU**를 쓴다.
* recurrent weight matrix $W_{hh}$ 를 **identity matrix $I$** 로 초기화한다.
* bias는 0으로 둔다.

이렇게 하면 입력이 없을 때 hidden state는 대체로 이전 state를 그대로 유지한다. 즉, 시점 $t$ 의 hidden state가 시점 $t-1$ 의 hidden state를 복사한 뒤 현재 입력의 영향을 더하고, 음수는 ReLU로 잘려 0이 된다. 저자들은 이런 모델을 **IRNN**이라고 부른다. 이때 중요한 점은 backpropagation through time에서 error derivative가 시간축을 따라 전달될 때, 추가적인 왜곡이 없으면 크기가 쉽게 줄어들거나 커지지 않고 비교적 일정하게 유지된다는 것이다. 논문은 이것이 “forget gate가 거의 decay를 만들지 않는 LSTM”과 유사한 동작이라고 해석한다.

또 하나의 중요한 아이디어는 **scaled identity**다. 모든 문제에 완전한 기억 유지가 필요한 것은 아니다. 짧은 문맥만 필요하면 과거를 빨리 잊는 것이 오히려 유리하다. 그래서 저자들은 $W_{hh} = \alpha I$ 와 같이 작은 스칼라 $\alpha$ 를 곱한 identity 초기화를 사용하면 장기 기억을 약화시키고 짧은 문맥 중심으로 동작하게 만들 수 있다고 본다. 이것은 LSTM에서 forget gate를 크게 열지 않고 memory decay를 빠르게 만드는 것과 비슷한 역할이다.

## 3. Detailed Method Explanation

### 3.1 모델 구조

논문이 제안하는 모델은 구조적으로는 매우 단순한 vanilla RNN에 가깝다. 차이는 hidden activation을 ReLU로 두고 recurrent matrix 초기화를 identity 계열로 잡는다는 점이다. 일반적인 형태로 쓰면 hidden update는 대략 다음과 같이 이해할 수 있다.

$$
h_t = \mathrm{ReLU}(W_{xh}x_t + W_{hh}h_{t-1} + b)
$$

여기서 제안법의 핵심은

$$
W_{hh} \leftarrow I \quad \text{or} \quad \alpha I,\qquad b \leftarrow 0
$$

라는 초기화다. non-recurrent weight는 평균 0, 표준편차 0.001의 Gaussian으로 초기화한다.

### 3.2 왜 identity initialization이 중요한가

identity initialization의 효과는 두 가지로 볼 수 있다.

첫째, **forward dynamics** 측면에서 hidden state가 쉽게 보존된다. 입력이 없으면 $h_t \approx h_{t-1}$ 이므로 내부 상태가 오래 유지된다. 이는 장기 의존성을 기억해야 하는 task에서 유리하다.

둘째, **backward dynamics** 측면에서 gradient가 시간축을 따라 지나갈 때 급격히 소실되거나 폭주할 가능성이 줄어든다. 논문은 이 점을 LSTM memory cell과 비슷한 관점에서 설명한다. LSTM은 gate가 열리고 닫히면서 memory를 통제하지만, IRNN은 구조는 단순해도 적절한 초기화만으로 “gradient를 오래 보존하는 통로”를 만든다. 즉, 이 논문의 메시지는 **복잡한 recurrent architecture 없이도 initialization만으로 상당한 효과를 얻을 수 있다**는 것이다.

### 3.3 학습 방식

모델들은 모두 BPTT로 학습하며, 최적화는 **fixed learning rate를 갖는 SGD + gradient clipping**으로 수행한다. 저자들은 learning rate를 ${10^{-9},10^{-8},...,10^{-1}}$ 범위에서, gradient clipping 값은 ${1, 10, 100, 1000}$ 범위에서 grid search했다. LSTM에 대해서는 forget gate bias도 ${1.0, 4.0, 10.0, 20.0}$ 범위에서 탐색했다. batch size는 16이다.

이 설정은 중요하다. 논문은 “IRNN이 단순한 초기화 트릭만으로 잘 된다”고 말하지만, 동시에 **학습률과 clipping은 꽤 중요하며 grid search를 수행했다**는 점도 분명하다. 따라서 이 논문을 재현하거나 해석할 때 “튜닝이 전혀 필요 없는 magic trick”으로 받아들이면 과장이다. 다만 저자들은 그럼에도 불구하고 구조 복잡도는 LSTM보다 훨씬 낮다고 주장한다.

## 4. Experiments and Findings

논문은 네 가지 benchmark를 사용한다.

1. Adding Problem
2. Pixel-by-pixel MNIST
3. Large-scale language modeling
4. TIMIT speech recognition

전체적으로 저자들의 결론은 다음과 같다.

* 장기 의존성이 매우 중요한 문제에서는 IRNN이 강력하다.
* 특히 toy long-range task에서는 LSTM과 대등하거나 더 낫다.
* 실제 과제에서도 대체로 LSTM에 근접한 성능을 보인다.
* 다만 짧은 문맥 위주 문제에서는 full identity보다 **scaled identity**가 더 적합하다.  

### 4.1 Adding Problem

이 task는 두 개의 입력 채널 중 하나가 random signal, 다른 하나가 mask이며, mask가 1인 두 시점의 숫자를 합산하는 문제다. sequence 길이 $T$ 가 커질수록 의존성이 멀어지므로 long-term dependency를 테스트하기 좋다. baseline은 항상 1을 예측하는 방식으로, MSE 약 0.1767이다. 저자들은 train 100,000개, test 10,000개 예제를 사용했고, hidden unit 수는 모두 100으로 고정했다. 이 경우 LSTM은 parameter 수와 timestep당 계산량이 IRNN보다 약 4배 크다.

논문에 따르면 $T \approx 150$ 부근부터 LSTM과 일반 RNN이 힘들어지기 시작했고, 이후 $T=150,200,300,400$ 에서 비교를 집중적으로 수행했다. 결과 해석의 핵심은 다음과 같다.

* IRNN의 convergence는 LSTM만큼 좋다.
* sequence 길이 400에서도 두 방법 모두 어려움을 겪지만, IRNN은 충분히 경쟁력 있다.
* 계산량 측면에서는 LSTM보다 훨씬 단순하다.  

즉, 이 실험은 “identity initialization이 실제로 장기 의존성을 유지하는 데 도움이 된다”는 논문의 직접적 증거다.

### 4.2 Pixel-by-pixel MNIST

이 task에서는 28×28 이미지를 784개의 pixel sequence로 펼쳐서 한 픽셀씩 읽고, 마지막에 숫자를 분류한다. permutation MNIST도 같이 실험해 더 어렵게 만들었다. 이는 784 timestep에 걸친 long-range dependency 문제다. 모든 네트워크는 hidden unit 100개를 사용했고, 최대 1,000,000 iteration까지 학습했다.

논문 요약 부분에서 저자들은 **IRNN이 LSTM보다 더 좋았고, test error 3% 대 34%** 수준의 큰 차이를 보였다고 강조한다. 이 수치는 특히 sequential MNIST에서 LSTM이 당시 설정상 매우 불리했음을 의미한다. 다만 저자들 스스로도 더 잘 튜닝된 LSTM이 더 나을 가능성은 열어둔다. 즉, 이 결과는 “IRNN이 언제나 LSTM보다 우월하다”가 아니라, **아주 긴 의존성을 직접 통과시켜야 하는 설정에서 identity-initialized ReLU recurrence가 매우 강력할 수 있다**는 증거로 보는 것이 타당하다.

### 4.3 Language Modeling

저자들은 large language modeling에서도 IRNN과 LSTM을 비교했다. 비교 방식이 흥미로운데, LSTM의 memory cell 하나는 ReLU hidden unit보다 내부 구조가 더 복잡하고 parameter도 많기 때문에, 단순히 같은 hidden width로 맞추는 것이 공정하지 않다고 본다. 그래서 다음과 같은 형태로 비교했다.

* LSTM: $N$ memory cells
* IRNN: 4 layers × $N$ hidden units
* 또는 1 layer × $2N$ hidden units

결론은 IRNN이 “equivalent LSTM”과 **비슷한 결과**를 낸다는 것이다. 여기서 논문의 포인트는 절대적인 SOTA가 아니라, **단순한 RNN도 적절한 초기화만 있으면 실용적 경쟁력을 가질 수 있다**는 점이다.

다만 제공된 첨부본의 발췌 범위에서는 language modeling의 세부 수치표가 완전히 드러나지 않아, 정확한 perplexity나 세부 아키텍처 수치를 여기서 단정적으로 적는 것은 피해야 한다. 논문 본문이 보여 주는 수준에서 확실한 결론은 “IRNN이 comparable”하다는 것이다.

### 4.4 TIMIT Speech Recognition

TIMIT는 장기 기억보다 **짧은 문맥 정보**가 더 중요한 acoustic modeling task로 설정된다. 여기서 저자들은 full identity 대신 **$0.01I$** 초기화를 사용하고 이를 **iRNN**이라 부른다. full identity는 과거 입력을 너무 오래 축적해 느린 수렴, 낮은 성능, 심지어 발산까지 유도할 수 있었다고 설명한다. 저자들의 해석은 speech task에서는 인접 frame들이 서로 비슷하므로, 과거를 너무 강하게 유지하면 현재 frame에 집중하기 어렵다는 것이다.

실험 결과도 이 해석을 뒷받침한다.

* 2-layer LSTM (250 cells): 34.5 / 35.4
* 2-layer iRNN (500 neurons): 34.3 / 35.5
* 5-layer LSTM: 35.0 / 36.2
* 5-layer iRNN: 33.0 / 33.8
* 5-layer bidirectional LSTM: 28.5 / 29.1
* 5-layer bidirectional iRNN: 28.9 / 29.7

즉, iRNN은 tanh RNN보다 확실히 낫고, 여러 설정에서 LSTM에 매우 근접한다. 최고 성능은 5-layer bidirectional LSTM이지만, 5-layer bidirectional iRNN도 바짝 따라간다.  

이 실험은 이 논문의 중요한 보정점을 제공한다. **identity initialization은 무조건 클수록 좋은 것이 아니고, task의 memory horizon에 맞게 스케일 조절이 필요하다.**

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 아이디어가 극도로 단순하면서도 통찰이 분명하다는 점이다. 복잡한 recurrent design의 성공 요인을 구조 전체에서 찾기보다, **정보와 gradient를 오래 보존하는 초기 상태**라는 핵심 요소를 분리해 실험한다. 그래서 논문 자체가 일종의 “ablation-like scientific argument” 역할을 한다.

또한 실험 설계도 메시지와 잘 맞는다. Adding problem, sequential MNIST처럼 long-term dependency를 직접 시험하는 task에서 IRNN의 장점을 뚜렷하게 보여 주고, speech처럼 short-term task에서는 scaled identity가 필요함을 보여 준다. 즉, 논문의 주장은 단순한 one-off result가 아니라 **memory timescale과 initialization scale의 관계**라는 일반적 설계 원리까지 제시한다.  

### 한계

첫째, 이 방법은 여전히 **초기화에 민감하고 학습 튜닝이 필요하다**. 논문도 learning rate, clipping, forget bias를 grid search했다고 명시한다. 따라서 “초기화만 바꾸면 끝”은 아니다.

둘째, 논문 스스로도 인정하듯이 일부 toy task에서 LSTM이 약하게 나온 것은 **튜닝 부족 또는 architecture mismatch** 때문일 수 있다. 따라서 sequential MNIST에서 IRNN이 LSTM을 크게 이긴 결과를 일반화할 때는 조심해야 한다.

셋째, ReLU recurrent dynamics는 잘못 다루면 activation explosion이나 지나친 accumulation 문제를 만들 수 있다. speech task에서 full identity가 오히려 해로웠다는 점은 이 방법이 “항상 identity가 정답”이 아님을 잘 보여 준다. 즉, **task별 memory horizon에 맞는 forgetting control이 필요**하다.

### 해석

비판적으로 보면, 이 논문의 가장 큰 가치는 “LSTM을 대체했다”보다도 **RNN 학습 난제의 본질을 구조가 아니라 dynamics 관점에서 설명했다**는 데 있다. 실제로 이후 recurrent model이나 residual/orthogonal initialization, gated/residual sequence model 설계에서도 “gradient path를 어떻게 보존할 것인가”는 핵심 주제가 된다. 이 논문은 그 흐름의 초기이자 매우 설득력 있는 사례로 읽을 수 있다.

## 6. Conclusion

이 논문은 ReLU 기반 RNN의 recurrent matrix를 identity 또는 scaled identity로 초기화하는 단순한 방법만으로도, 장기 의존성 학습을 상당히 안정화할 수 있음을 보였다. toy long-range task에서는 LSTM에 필적하거나 더 나은 결과를, language modeling과 speech recognition 같은 실제 문제에서는 대체로 comparable한 성능을 보고했다. 핵심 메시지는 복잡한 gating 이전에, **시간축을 따라 state와 gradient를 보존하는 initialization/dynamics가 얼마나 중요한가**를 보여 주는 데 있다.  

실무적으로는 다음처럼 이해하면 된다. 아주 긴 dependency가 필요하면 full identity에 가까운 초기화가 유리할 수 있고, 짧은 문맥만 중요하면 $0.01I$ 같은 scaled identity가 더 적합하다. 즉 이 논문은 하나의 기법 제안이면서 동시에, recurrent network를 설계할 때 **기억 유지와 forgetting의 균형을 어떻게 초기화로 조절할 것인가**라는 설계 철학을 제공한다.
