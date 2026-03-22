# Three Decades of Activations: A Comprehensive Survey of 400 Activation Functions for Neural Networks

## 1. Paper Overview

이 논문은 신경망의 activation function(AF)에 관한 **대규모 문헌 조사(survey)**이다. 저자들의 핵심 문제의식은, 지난 30여 년 동안 매우 많은 activation function이 제안되었음에도 이를 한곳에 체계적으로 정리한 포괄적 자료가 부족하다는 점이다. 그 결과 기존 함수가 다시 제안되거나, 유사한 함수가 “새로운 방법”처럼 반복적으로 등장하는 비효율이 발생한다. 이 논문은 이런 문제를 줄이기 위해 **총 400개의 activation function을 모아 체계적으로 분류하고, 각 함수의 원 논문까지 연결하는 참고 자원**을 제공하는 것을 목표로 한다.

이 문제가 중요한 이유는 activation function이 단순한 구현 선택지가 아니라, 학습 안정성·gradient 흐름·계산 효율성·표현력에 직접적인 영향을 주기 때문이다. 논문은 특히 deep learning 시대에 activation function 선택이 성능과 학습 속도에 미치는 영향이 매우 커졌음을 강조한다. 따라서 이 논문의 기여는 “새 activation을 제안하는 것”이 아니라, **연구 지형을 정리해 중복 연구를 줄이고, 후속 연구자들이 더 정확한 출발점 위에서 연구할 수 있도록 돕는 것**에 있다.

## 2. Core Idea

이 논문의 중심 아이디어는 activation function 연구를 “좋은 함수 몇 개를 소개하는 수준”이 아니라, **역사적·구조적 관점에서 전수 조사에 가깝게 조직화**하는 데 있다. 기존 survey들이 대체로 잘 알려진 함수만 다루거나 70개 내외 수준에서 멈췄다면, 이 논문은 그보다 훨씬 큰 규모로 문헌을 수집해 400개를 정리한다.

가장 중요한 설계 선택은 activation function을 크게 두 부류로 나누는 것이다.

1. **Fixed activation functions**: 함수 형태가 미리 정해져 있고 학습 중 바뀌지 않는 함수
2. **Adaptive activation functions (AAFs)**: 기울기, 이동량, 곡률, 혼합 비율 등 일부 파라미터를 학습하는 함수

저자들은 swish와 SiLU처럼 본질적으로 유사하지만 “파라미터를 학습하느냐” 여부에 따라 다른 범주에 놓일 수 있는 사례를 지적하면서도, 이 구분이 여전히 유의미하다고 본다. 왜냐하면 adaptive 함수는 단순한 모양 차이를 넘어, 학습 중 데이터에 맞추어 함수 형태가 변하는 추가 유연성을 제공하기 때문이다.

즉, 이 논문의 새로움은 하나의 새 activation을 내놓는 데 있지 않고, **activation function 연구 전체를 지도처럼 정리해 주는 메타 수준의 기여**에 있다. 연구자 입장에서는 “ReLU vs GELU vs Mish” 수준의 비교를 넘어서, sigmoid 계열·ReLU 변형·self-gated 계열·square-based 함수·chaotic 함수·adaptive spline/rational 계열 등 넓은 공간을 체계적으로 탐색할 수 있게 된다.

## 3. Detailed Method Explanation

### 3.1 논문의 전체 구조

논문 구조는 전형적인 실험 논문과 다르다. 새로운 모델을 설계해 성능을 검증하는 대신, 다음 흐름으로 구성된다.

- activation function 연구 필요성과 survey 공백 제시
- 기존 survey/benchmark 문헌 리뷰
- **고정형 activation function** 대규모 정리
- **적응형 activation function** 대규모 정리
- 최종적으로 전체 흐름을 요약하고 한계를 정리

즉, “방법(method)”은 새로운 알고리즘이 아니라 **분류 체계와 정리 방식 자체**라고 보는 편이 정확하다.

### 3.2 분류 원리: Fixed vs Adaptive

논문의 가장 핵심적인 방법론은 activation function을 아래와 같이 이분화하는 것이다.

#### Fixed activation functions

입력 $z$가 들어왔을 때, 출력 $f(z)$의 형태가 학습 중 변하지 않는 함수들이다. 예를 들어 sigmoid, tanh, ReLU처럼 네트워크 전체 혹은 레이어 전체에 동일한 규칙을 적용한다. 이 범주는 역사적으로 초기 신경망과 현대 딥러닝 모두에서 널리 쓰인 함수들을 포함한다.

#### Adaptive activation functions

함수 모양을 결정하는 파라미터를 학습하는 계열이다. 예를 들어 PReLU처럼 음수 구간의 slope를 학습하거나, PELU처럼 곡선을 조정하거나, TAAF처럼 스케일·이동을 학습하는 방식이 여기에 들어간다. 저자들은 adaptive 함수가 데이터에 맞춰 형태를 바꿀 수 있기 때문에 더 높은 유연성을 가진다고 해석한다.

이 분류는 단순하지만 강력하다. 왜냐하면 activation function 연구에서 반복적으로 등장하는 질문—“새 함수가 fixed design인지, 아니면 trainable shape인지”—를 바로 정리해 주기 때문이다.

### 3.3 고정형 activation function의 내부 체계

고정형 함수는 다시 여러 하위 계열로 나뉜다. 논문은 단지 이름을 나열하지 않고, 수식과 역사적 맥락, 그리고 다른 함수와의 관계까지 함께 설명한다.

#### (a) Binary / step 계열

가장 원초적인 activation으로, 입력의 부호에 따라 0 또는 1을 출력한다. 수식은 다음과 같다.

$$
f(z) =
\begin{cases}
1, & z \ge 0 \\
0, & z < 0
\end{cases}
$$

이 계열은 계산은 매우 쉽지만, 거의 모든 구간에서 미분 정보가 없어서 backpropagation 기반 최적화와 잘 맞지 않는다. 논문은 이를 역사적으로 중요한 함수로 보되, 현대 gradient learning에는 부적합하다고 정리한다.

#### (b) Sigmoid family

대표적으로 logistic sigmoid와 tanh가 있다. 예를 들어 logistic sigmoid는

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

이며, tanh는

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z)-1
$$

이다.

논문은 sigmoid 계열의 장점으로 확률적 해석 가능성, 매끄러운 곡선, 전통적 사용성을 들면서도, **saturation**으로 인해 gradient가 사라지는 vanishing gradient 문제가 크다고 설명한다. 이어서 shifted/scaled sigmoid, arctan, softsign, hard sigmoid, multistate sigmoid 같은 수많은 변형을 소개한다. 중요한 포인트는, 많은 “새로운 함수”가 사실 sigmoid 구조의 변형 혹은 조합이라는 점이다.

#### (c) Sigmoid-weighted linear unit 계열

이 범주는 최근 딥러닝에서 특히 중요한 흐름이다. 일반형은 다음과 같이 정리된다.

$$
f(z) = z \cdot s(z)
$$

여기서 $s(z)$는 sigmoid류의 squashing 함수다. 대표 예시는 SiLU:

$$
f(z) = z\sigma(z)
$$

이다. 이 계열에서 GELU, Mish, Logish, Phish, SinSig 등 다양한 함수가 나온다.

- **GELU**는 Gaussian CDF를 사용해 입력을 soft하게 통과시킨다.
- **Mish**는 $z \cdot \tanh(\ln(1+e^z))$ 형태로, self-gating과 smoothness를 결합한다.
- **SwiGLU / GEGLU / ReGLU** 같은 GLU 계열은 transformer나 sequence model과도 연결된다.

논문은 이 계열이 현대 딥러닝에서 중요한 이유를, ReLU의 단순성은 유지하면서도 gate-like weighting을 통해 더 섬세한 입력 제어를 하기 때문으로 보여준다.

#### (d) ReLU family

ReLU는 현대 feedforward network의 기본 선택지로 제시된다.

$$
f(z) = \max(0, z)
$$

장점은 단순함, 계산 효율성, 빠른 수렴이다. 그러나 음수 영역에서 gradient가 0이 되므로 “dying ReLU” 문제가 있다. 논문은 이 문제를 해결하거나 완화하려는 엄청나게 많은 파생형을 정리한다.

- **Leaky ReLU**: 음수 구간에도 작은 기울기 허용
- **RReLU**: leakiness를 stochastic하게 샘플링
- **PReLU**: leakiness를 학습
- **ELU / SELU**: 음수 영역에 exponential 곡선을 사용
- **Hard variants**: hard sigmoid, hard tanh, hard swish
- **대칭/삼각/이동형 변형**: vReLU, SoftModulusT, DisReLU, SignReLU 등

이 부분에서 논문이 특히 잘 보여주는 것은, ReLU 하나가 단순한 함수가 아니라 **거대한 설계 계열(family)**의 중심이라는 점이다. 실제로 많은 후속 제안은 “ReLU의 양수 구간은 유지하고, 음수 구간을 어떻게 바꾸느냐”의 문제로 귀결된다.

#### (e) 계산 효율 중심 계열

square-based activation, inverse square root 계열, polynomial 계열 등도 정리한다. 이 함수들은 정확히 더 좋은 일반화를 노리기보다, **exp 연산을 줄이거나 하드웨어 친화적 구현을 돕는 방향**으로 설계된다. 예를 들어 SQNL, SQLU, square swish 등은 저전력 장치나 hardware acceleration 문맥에서 중요하다.

#### (f) 특수/비전형 계열

논문 후반부의 고정형 분류에는 sine, cosine, wave, chaotic activation, k-winner-take-all 같은 비전형 함수들도 포함된다. 이는 저자들이 “주류 함수만 모은 survey”가 아니라, activation이라는 개념 아래 실제 문헌에 등장한 폭넓은 제안을 가능한 많이 포착하려고 했음을 보여준다.

### 3.4 적응형 activation function의 내부 체계

적응형 함수 파트는 논문 4장에서 다루며, 핵심 메시지는 “activation을 더 이상 고정된 함수로만 볼 필요가 없다”는 것이다.

대표적 개념은 다음과 같다.

#### (a) PReLU류

가장 익숙한 adaptive activation의 출발점은 음수 구간 기울기를 학습하는 PReLU다. ReLU의 문제를 데이터 기반으로 보정한다는 점에서 중요한 전환점이다.

#### (b) TAAF (Transformative Adaptive Activation Function)

이 논문이 adaptive 파트를 조직할 때 중요한 상위 틀로 제시하는 것이 TAAF다. 개념적으로는 임의의 기반 함수 $f$에 대해 스케일과 이동을 학습하는 일반형이다.

$$
g(f, z_i) = \alpha_i \cdot f(\beta_i z_i + \gamma_i) + \delta_i
$$

여기서 $\alpha_i, \beta_i, \gamma_i, \delta_i$는 뉴런별 학습 파라미터다. 이 식이 중요한 이유는, 많은 adaptive activation이 사실상 “기존 함수에 trainable scaling/shift를 더한 것”으로 해석될 수 있기 때문이다. 즉 TAAF는 개별 함수 하나라기보다, 여러 adaptive 방법을 포괄하는 **상위 프레임워크**에 가깝다.

#### (c) shape-learning 계열

adaptive spline, rational activation, 혼합형 activation, trainable polynomial/rational 함수 등은 함수 자체의 곡률과 형태를 더 자유롭게 학습한다. 이런 계열은 단순 slope 조정이 아니라, activation 자체를 함수 근사 문제처럼 취급한다.

#### (d) blended / combinational 계열

여러 activation을 섞어서 학습하거나, 입력 조건에 따라 다른 activation을 고르게 하는 방식도 포함된다. 이는 activation 선택 문제를 사람이 수작업으로 고르는 대신, 네트워크가 내부적으로 조합하게 만든다는 점에서 중요하다.

### 3.5 이 논문이 실제로 하는 일과 하지 않는 일

이 논문을 읽을 때 가장 중요한 해석 포인트는, 이것이 “새로운 activation 설계 논문”도 아니고 “엄밀한 benchmark 논문”도 아니라는 점이다.

이 논문이 하는 일:
- activation function 문헌을 폭넓게 수집
- 수식과 출처를 정리
- 함수 간 유사성/중복성/역사적 선후를 드러냄
- fixed vs adaptive 축으로 조직화

이 논문이 하지 않는 일:
- 동일 조건에서 대규모 통합 benchmark 수행
- 특정 함수가 항상 최선이라고 결론 내리기
- 하나의 새로운 universal activation 제안

따라서 이 survey의 가치는 “정답 함수 제시”가 아니라, **탐색 공간을 정확히 보여주는 지도 제공**에 있다.

## 4. Experiments and Findings

이 논문은 실험 논문이 아니라 survey이므로, 독자적인 benchmark 실험은 거의 수행하지 않는다. 오히려 저자들은 기존 benchmark 연구들이 대체로 소수의 유명 activation만 비교한다는 점을 비판적으로 요약한다. 즉, 실험 파트의 핵심 발견은 “어떤 activation이 절대적으로 최고다”가 아니라, **지금까지의 실험 문화 자체가 activation 공간의 아주 일부만 보고 있었다**는 점이다.

그래도 논문은 개별 activation을 설명하는 과정에서 여러 실증적 경향을 전달한다.

- **ReLU**는 여전히 기본값으로 강력하다. 계산이 싸고 수렴이 빠르다.
- 하지만 **dying ReLU** 문제 때문에 LReLU, ELU, PReLU 같은 파생형이 다수 제안되었다.
- **GELU, Mish, swish/SiLU 계열**은 현대 deep model에서 자주 더 나은 성능을 보인다고 정리된다.
- **SELU**는 self-normalizing 특성을 목표로 설계되었다.
- **Mish, SinSig, EANAF** 등 일부 함수는 특정 비전 과제나 deep architecture에서 유리했다고 소개된다.
- **square-based / inverse square root 계열**은 정확도뿐 아니라 계산 효율성 측면에서 의미가 있다.

중요한 것은, 논문이 이런 결과를 일관된 통합 실험으로 주장하지 않는다는 점이다. 각 함수의 “좋았다”는 평가는 대부분 원 논문이 보고한 결과를 요약한 것이다. 따라서 독자는 이를 **재현된 보편 법칙**이 아니라, “문헌상 보고된 경향”으로 이해해야 한다.

또 하나의 중요한 발견은 **중복 제안**의 빈도다. 저자들은 RePU, DPReLU, TRec, ReLU-Swish, BReLU 같은 사례를 들어, 이미 존재하던 함수가 다시 제안되는 문제가 실제로 자주 발생함을 지적한다. 이 점은 이 survey의 실질적 가치를 잘 보여준다. 단순 정리 작업처럼 보여도, 연구 생태계에서는 중복과 재발명을 줄이는 기능이 크기 때문이다.

## 5. Strengths, Limitations, and Interpretation

### 강점

첫째, **규모 자체가 강점**이다. 400개 activation function을 한 논문 안에서 구조화한 자료는 매우 드물다. 특히 함수 이름만 나열한 것이 아니라, 수식·기원 논문·유사 함수 관계를 함께 정리한다.

둘째, **fixed vs adaptive라는 큰 축이 명확**하다. 이 분류 덕분에 activation function 연구를 단순 나열이 아니라 설계 철학의 차이로 이해할 수 있다.

셋째, **중복 제안 문제를 직접 드러낸다**. activation 연구는 이름만 다른 유사 함수가 많고, 함수 형태가 조금만 달라도 새 논문처럼 보이기 쉽다. 이 논문은 그런 현상을 문헌 수준에서 정리해 준다.

넷째, **실용적 참고서 역할**을 한다. 새로운 모델을 설계할 때 activation을 더 넓게 탐색하고 싶은 연구자에게 매우 유용하다.

### 한계

첫째, 저자들도 명시하듯이 **통합 benchmark가 없다**. 따라서 “어떤 함수가 실제로 가장 좋은가?”라는 질문에는 답하지 않는다.

둘째, **real-valued activation function 중심**이다. complex-valued, quaternion, photonic, fuzzy, quantum activation 등은 범위 밖이다.

셋째, activation function을 폭넓게 모으는 대신, 개별 함수에 대한 깊은 이론적 분석은 제한적이다. 예를 들어 어떤 함수가 왜 특정 데이터셋에서 잘 되는지에 대한 엄밀한 일반 이론은 제공하지 않는다.

넷째, 이런 survey는 본질적으로 시간이 지나면 다시 불완전해진다. activation 연구는 계속 새 함수가 등장하므로, 이 논문도 “완결판”이라기보다 2024년 시점의 가장 넓은 정리에 가깝다.

### 해석

비판적으로 보면, 이 논문은 새로운 실험 결과나 이론보다 **문헌 정리의 가치**에 거의 전적으로 기대고 있다. 하지만 activation function 분야처럼 이름 중복, 유사 함수 남발, 제한적 benchmark 관행이 심한 영역에서는 이런 정리 작업이 매우 중요하다. 즉 이 논문의 기여는 flashy하지 않지만, 연구 효율성과 재현 가능성 측면에서는 상당히 실질적이다.

또한 이 survey는 activation 연구의 큰 흐름을 잘 보여준다.

- 초기에는 sigmoid/tanh 중심
- 이후 ReLU와 그 파생형이 주도
- 최근에는 self-gated smooth activation과 adaptive activation으로 확장
- 동시에 hardware efficiency, robustness, task-specific design으로 세분화

이 흐름을 한 논문에서 한눈에 볼 수 있다는 점이 이 논문의 가장 큰 의미다.

## 6. Conclusion

이 논문은 activation function 분야에서 “새 함수 제안”이 아니라 **연구 지형 전체를 체계적으로 재정리한 기준점** 역할을 한다. 저자들은 총 400개의 activation function을 정리하면서, 이를 고정형과 적응형으로 나누고, 각 계열 안에서 수식·출처·관계성을 설명한다.

실무적으로 이 논문은 다음과 같은 상황에서 특히 가치가 크다.

- 새로운 네트워크를 설계하면서 ReLU 외 대안을 폭넓게 검토하고 싶을 때
- activation function 관련 연구를 시작하며 중복 제안을 피하고 싶을 때
- 특정 함수가 기존 계열의 변형인지, 정말 새로운 설계인지 검토할 때
- adaptive activation이나 hardware-friendly activation 같은 세부 분야를 빠르게 훑고 싶을 때

정리하면, 이 논문은 “최고의 activation이 무엇인가”를 답하는 논문은 아니다. 대신 **activation function 연구의 전체 검색 공간을 넓고 정돈된 형태로 보여주는 논문**이다. 앞으로 activation 연구를 하는 사람들에게는 새로운 출발점이자 체크리스트 역할을 할 가능성이 크다.
