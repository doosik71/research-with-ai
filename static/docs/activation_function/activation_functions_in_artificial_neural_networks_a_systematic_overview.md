# Activation Functions in Artificial Neural Networks: A Systematic Overview

## 1. Paper Overview

이 논문은 인공신경망에서 activation function이 어떤 역할을 하며, 대표적인 activation들이 어떤 수학적 성질과 실무적 함의를 갖는지를 체계적으로 정리한 overview 논문이다. 저자는 activation function이 뉴런의 출력 형태를 결정하므로, neural network 전반과 deep learning에서 핵심적 구성요소라고 본다. 그런데 logistic, relu처럼 오래된 함수에 더해 deep learning 붐 이후 수많은 새 activation이 등장하면서, 이론과 실무 모두에서 선택 기준이 흐려졌고, 이 논문은 바로 그 혼란을 줄이기 위해 “인기 있는 activation function과 그 성질”을 analytic하면서도 최신 관점으로 정리한다. 초록은 이 논문이 neural networks를 연구하거나 적용하는 사람에게 시의적절한 참고자료가 되도록 쓰였다고 밝힌다.

논문은 서론에서 뉴런을
$$
\mathfrak{n}_{\beta,\boldsymbol{\theta},\mathfrak{f}}(\boldsymbol{x})
= \mathfrak{f}!\left(\beta + \boldsymbol{\theta}^{\top}\boldsymbol{x}\right)
\mathfrak{f}!\left(\beta + \sum_{j=1}^{d}\theta_j x_j\right)
$$
형태의 함수로 두고, 여기서 $\beta$는 bias, $\boldsymbol{\theta}$는 입력별 민감도, $\mathfrak{f}$는 activation pattern을 결정하는 함수라고 설명한다. 즉, activation function은 단순한 부가 요소가 아니라 “입력의 선형 결합을 어떤 비선형 출력으로 바꿀 것인가”를 정하는 핵심 설계 요소다.

또한 저자는 artificial neurons를 결합해 더 큰 함수 $\mathbb{R}^d \to \mathbb{R}$를 만들 수 있다는 점에서, activation choice가 네트워크의 표현력과 학습 특성 전체에 영향을 미친다고 본다. 이 논문의 문제의식은 결국 “어떤 activation이 어떤 상황에서 더 적절한가”를 감이 아니라 구조적으로 이해하자는 데 있다.

## 2. Core Idea

논문의 핵심 아이디어는 새로운 activation을 제안하는 것이 아니라, 기존 activation들을 세 범주로 묶어 비교 가능하게 만드는 데 있다. 실제 섹션 구조도 이를 그대로 반영한다. Section 2는 크게 sigmoid functions, piecewise-linear functions, other functions로 나뉘고, 각각의 하위 항목으로 logistic, arctan, tanh, softsign, linear, relu, leakyrelu, softplus, elu/selu, swish, learning activation functions를 다룬다. 이 자체가 논문의 핵심 프레임이다.

이 프레임에서 저자가 일관되게 보는 비교 기준은 다음과 같다. 각 activation의 출력 범위, 미분 가능성, 도함수 형태, 계산 단순성, saturation 여부, expressivity, 그리고 optimization에서의 문제점 또는 장점이다. 예를 들어 sigmoid 계열은 bounded하고 smooth하지만 gradient가 약해지기 쉽고, ReLU 계열은 계산이 매우 단순하고 실용적이지만 dying-ReLU 같은 문제가 있다. other functions는 이런 trade-off를 보완하거나 확장하려는 시도로 제시된다. 이 구조는 Section 2 전체와 Section 3의 “practical implications”에서 명확히 드러난다.  

따라서 이 논문의 새로움은 “best activation”을 선언하는 데 있지 않다. 오히려 함수 형태가 비슷하면 충분히 잘 최적화된 뒤 empirical result도 비슷할 수 있으므로, 실제 선택은 계산 단순성, 이론적 정당성, 자동 탐색 가능성, 데이터 적응성까지 포함해 보아야 한다는 관점을 제공하는 데 있다. 이 결론은 Section 3에서 softsign, relu, selu, swish, data-adaptive schemes를 연결해 정리된다.  

## 3. Detailed Method Explanation

이 논문은 실험 논문이 아니라 분석적 survey이므로, “방법론”은 모델 학습 파이프라인이 아니라 activation들을 해부하는 분석 절차다. 서론은 뉴런과 네트워크를 수학적으로 정의하고, Section 2는 activation family별 속성을 비교하며, Appendix A는 각 함수의 미분과 성질을 보강하는 수학적 세부를 제공한다. 즉, 본문은 직관과 실무 해석을, 부록은 엄밀한 미분 및 성질 증명을 담당하는 이중 구조다.  

### 3.1 Overall pipeline

논문의 분석 흐름은 대략 다음과 같다.

첫째, activation function이 뉴런의 출력 구조를 어떻게 바꾸는지 설명한다.
둘째, activation들을 몇 개의 family로 묶는다.
셋째, 각 함수의 정의역과 치역, smoothness, derivative, shape를 비교한다.
넷째, 이 차이가 expressivity와 optimization에 어떤 차이를 만드는지 해석한다.
다섯째, practical implication을 정리하고, 미래 방향으로 theory-driven design, automated search, adaptive learning을 제안한다.  

### 3.2 Sigmoid family: logistic, arctan, tanh, softsign

#### logistic

logistic은
$$
\mathfrak{f}_{\log}(z)=\frac{1}{1+e^{-z}}
$$
로 정의되며 출력 범위는 $(0,1)$이다. 논문은 logistic을 perceptron의 binary/step activation의 smooth version으로 해석한다. 즉, hard threshold를 부드럽게 만든 함수라는 역사적 맥락을 부여한다. 이는 logistic이 고전적이면서도 여전히 중요한 이유를 설명한다.

#### arctan

arctan은
$$
\mathfrak{f}_{\operatorname{arctan}}(z)=\arctan(z)
$$
형태로 소개되며, 출력 범위는 $\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$이다. 논문은 이를 tangent의 inverse로 위치시키며, logistic이나 tanh와 유사한 sigmoid-like behavior를 가지지만 출력 범위와 기울기 구조가 다름을 보여준다. 이는 같은 “sigmoid 계열” 안에서도 세부 수학적 성질이 다를 수 있음을 보여주는 예다.

#### tanh

tanh는
$$
\mathfrak{f}_{\tanh}(z)=\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$
로 주어지고 출력 범위는 $(-1,1)$이다. 논문은 tanh가 entire domain에서 infinitely differentiable하며, 첫째 미분이
$$
\dot{\mathfrak{f}}_{\tanh}(z)=1-\bigl(\mathfrak{f}_{\tanh}(z)\bigr)^2
$$
꼴임을 강조한다. tanh는 zero-centered 출력 때문에 logistic보다 해석상 편한 면이 있지만, 여전히 bounded sigmoid 계열의 특성을 공유하므로 saturation과 gradient attenuation 문제를 완전히 피하지는 못한다.

#### softsign

softsign은
$$
\mathfrak{f}_{\operatorname{soft}}(z)=\frac{z}{1+|z|}
$$
로 정의되며 역시 출력 범위는 $(-1,1)$이다. 논문은 softsign을 특히 단순한 sigmoid function으로 강조한다. 도함수는
$$
\dot{\mathfrak{f}}_{\operatorname{soft}}(z)=\bigl(1-|\mathfrak{f}_{\operatorname{soft}}(z)|\bigr)^2 \in (0,1)
$$
형태로 제시되고, 계산 구조가 매우 단순하다는 점이 practical implication에서 재차 강조된다. 저자는 Section 3에서 바로 이 softsign이 더 많은 관심을 받아야 한다고 주장한다.  

### 3.3 Piecewise-linear family: linear, relu, leakyrelu

#### linear

linear activation은 identity function이다. 논문은 constant derivative 때문에 optimization 관점에서 종종 과소평가되지만, 진짜 약점은 업데이트가 아니라 낮은 expressivity라고 지적한다. 모든 activation이 linear인 네트워크는 결국 전체적으로 선형 함수에 머무르므로, 복잡한 비선형 문제를 다루기 어렵다. 그래서 linear activation은 보통 회귀 output layer처럼 제한된 위치에서만 적절하다고 본다.

#### relu

relu는 piecewise-linear family의 중심이다. 논문은 relu를 계산적으로 매우 단순한 함수로 보고, practical implication에서도 competitor들보다 relu가 computational simplicity 측면에서 두드러진다고 말한다. Section 3는 “지금 시점에서는 relu와 softsign이 practice의 standard activation이 되어야 한다”고까지 정리한다. 이는 논문이 relu를 단순 인기 함수가 아니라 현실적인 기준에서 가장 강한 baseline으로 간주함을 보여준다.  

다만 relu의 대표적 약점은 dying-ReLU phenomenon이다. Appendix에 별도 subsection이 있을 정도로 이 현상은 중요하게 취급된다. 즉, 음수 영역에서 gradient가 사라져 일부 unit이 사실상 죽을 수 있다는 문제다. 논문은 relu의 실용적 우위를 인정하면서도, 이 약점을 명확히 남겨둔다.

#### leakyrelu

leakyrelu는 relu를 모방하되 dying-ReLU를 피하기 위해 음수 영역에도 작은 기울기를 주는 함수로 설명된다. 즉, positive region에서는 relu와 같고 negative region에서 완전한 0 slope 대신 작은 leak를 남긴다. 논문은 이를 relu의 직접적인 보완형으로 소개하지만, practical implication에서는 여전히 relu 자체가 simplicity 면에서 더 강하게 지지된다. 다시 말해 leakyrelu는 중요한 변형이지만, 이 논문은 “기본 선택”으로는 relu에 더 무게를 둔다.  

### 3.4 Other functions: softplus, elu/selu, swish, learnable activations

#### softplus

softplus는
$$
\mathfrak{f}_{\operatorname{soft+}}(z)=\log(1+e^z)
$$
로 정의된다. 논문은 softplus를 logistic과 직접 비교하기보다, 오히려 relu의 smooth version으로 봐야 한다고 말한다. 첫째 미분이 logistic이고 둘째 미분은 logistic의 derivative가 되므로, softplus는 relu류의 형태를 부드럽게 만든 activation으로 해석된다. 이 점은 “smoothness”를 중시할 때 softplus의 위치를 이해하는 핵심이다.

#### elu and selu

ELU는 parameter $a \in [0,\infty)$를 갖는 함수로, 양수 영역에서는 identity, 음수 영역에서는 exponential tail을 가진다. 출력 범위는 $(-a,\infty)$다. 논문은 ELU/SELU를 theory-driven design의 사례로 연결하고, Section 3에서는 앞으로 더 나은 activation을 찾는 방향 중 하나로 theoretical considerations를 제시하면서 SELU를 예로 든다. 즉, 단순 empirical tweak가 아니라 네트워크의 안정성이나 정규화 성질을 고려한 설계라는 의미다.  

#### swish

swish는 parameter $a$에 따라 relu와 linear 사이를 연속적으로 오가는 activation으로 설명된다. 논문은 $a$가 커질수록 swish가 relu를 닮고, 작아질수록 scaled linear를 닮는다고 말한다. 또한 swish는 entire domain에서 infinitely differentiable하며, 도함수와 출력 범위의 성질이 Appendix에서 자세히 다뤄진다. 이 함수는 Section 3에서 automated searches의 대표 예로 언급되는데, 즉 사람이 손으로 만든 함수라기보다 search-driven design의 산물이라는 점이 중요하다.  

#### Learning activation functions / maxout

논문은 activation을 고정 함수로만 보지 않는다. 미리 정한 activation family 안에서 선택하거나, activation의 parameter를 학습하거나, polynomial activation을 맞추거나, PReLU/PELU처럼 기존 함수의 parameter를 조정하는 방식들을 소개한다. swish도 parameter fitting의 사례다. 관련하여 maxout도 다루는데, 이는 뉴런 구조 자체를 바꾸는 방식이며 parameter space를 키워 overfitting과 computational burden 위험을 동반한다고 지적한다. 논문은 이런 adaptive schemes가 흥미롭지만, 아직 theoretical and empirical support가 매우 제한적이라고 본다.

### 3.5 수식 해석의 핵심

이 논문은 activation을 단순히 “곡선 모양”으로 비교하지 않고, derivative를 매우 중요하게 본다. logistic, tanh, softsign, softplus, swish 모두 도함수 성질을 본문과 Appendix에서 반복적으로 분석한다. 이유는 분명하다. backpropagation은 derivative를 통해 흐르므로, activation의 미분 가능성, 도함수의 범위, 음수/양수 영역에서의 기울기 구조가 optimization behavior를 좌우하기 때문이다.

예를 들어 linear는 derivative가 상수여서 gradient 전파는 단순하지만 expressivity가 부족하다. softplus는 relu보다 더 smooth하지만 계산적 단순성에서는 밀린다. relu는 단순하지만 negative region에서 죽을 수 있다. softsign은 sigmoid family 안에서 특히 단순하다. swish는 relu와 linear 사이를 parameter로 잇는다. 이런 수식 기반 비교가 논문의 분석 축이다.

## 4. Experiments and Findings

이 논문은 benchmark 실험을 중심에 둔 empirical paper가 아니다. 원문 구조상 Section 1은 Introduction, Section 2는 activation들의 properties, Section 3은 Practical Implications이며, 별도의 “Experiments” 섹션이 없다. 따라서 이 논문에서 말하는 findings는 실험 표를 통해 얻은 결과라기보다, 함수 비교를 통해 도출한 분석적 결론에 가깝다.

핵심 findings는 세 가지 practical implication으로 정리된다.

첫째, sigmoid family 안에서는 softsign이 특히 유망하다는 것이다. 저자는 비슷한 그래프를 가지는 activation들은 충분히 잘 최적화되면 비슷한 empirical result를 낼 수 있다고 보고, 그 경우 더 계산이 단순한 activation을 택하는 것이 낫다고 해석한다. 그 결과, softsign은 “particularly simple sigmoid”이므로 더 많은 관심을 받아야 한다고 주장한다.

둘째, piecewise-linear 계열에서는 relu가 여전히 강한 실용적 기본값이라는 것이다. 경쟁자들에 대한 empirical/theoretical evidence는 제한적이며, relu는 computational simplicity에서 두드러진다고 논문은 정리한다. 그래서 현 시점 practice의 standard activation으로는 softsign과 relu를 추천한다.  

셋째, 이 둘을 넘어서는 개선 방향으로 세 가지를 제시한다. theory-driven design의 예로 selu, automated search의 예로 swish, data-adaptive selection schemes의 예로 learnable activations를 든다. 즉, 미래의 activation 연구는 임의의 새 함수 제안보다 더 구조적인 탐색 방식으로 갈 가능성이 크다는 것이 이 논문의 전망이다.  

## 5. Strengths, Limitations, and Interpretation

이 논문의 강점은 activation function을 “공식 모음집”이 아니라 설계 원리의 관점에서 정리한다는 점이다. 실제로 본문은 activation을 family별로 조직하고, 각 함수의 정의, 출력 범위, derivative, smoothness, expressivity, practical meaning을 함께 설명한다. 부록의 Mathematical Details는 이 분석에 엄밀성을 더한다. 그래서 초심자에게는 체계적인 입문서이고, 연구자에게는 activation 비교의 기준표 역할을 한다.

또 다른 강점은 결론이 지나치게 과장되지 않는다는 점이다. 논문은 특정 activation 하나를 절대적으로 최고라고 밀어붙이지 않는다. 오히려 비슷한 activation은 성능도 비슷할 수 있으며, 그럴수록 computational simplicity나 theoretical support가 중요해진다고 본다. 이는 실무적으로 꽤 설득력 있다. 괜히 복잡한 새 activation을 쓰기보다, 충분한 근거가 없는 한 relu나 softsign 같은 단순한 대안을 우선 고려하라는 메시지이기 때문이다.

한계도 분명하다. 첫째, 이 논문은 실험 논문이 아니어서 특정 task나 dataset에서 어떤 activation이 얼마나 더 낫다는 강한 empirical conclusion을 주지 않는다. 둘째, practical implication이 분명하지만, 그 추천은 어디까지나 이론적·분석적 비교를 강하게 반영한 것이지, 모든 현대 아키텍처에서의 보편적 승자를 선언하는 것은 아니다. 셋째, learnable activation과 maxout 같은 adaptive approaches에 대해서도 잠재력은 인정하지만, 지지 증거는 아직 제한적이라고 논문 스스로 말한다.

내 해석으로는, 이 논문은 activation 연구의 “지도” 같은 논문이다. logistic, tanh, relu 정도만 어렴풋이 알고 있는 독자에게는 softsign, softplus, swish, selu, maxout을 서로 어떤 관계로 봐야 하는지 정리해 준다. 특히 softplus를 logistic이 아니라 smooth relu로 봐야 한다는 설명, swish를 relu-linear interpolation 관점에서 보는 설명, softsign을 단순한 sigmoid의 유력 후보로 끌어올리는 설명은 이 논문의 해석적 가치가 큰 부분이다.

## 6. Conclusion

이 논문은 activation functions를 세 큰 범주로 나누어 비교하고, 각 함수의 수학적 성질과 practical meaning을 연결해 설명하는 체계적인 overview다. 서론은 뉴런을 activation을 포함한 함수로 수학화하고, Section 2는 sigmoid, piecewise-linear, other functions를 비교하며, Section 3은 그 비교를 실무적 추천으로 압축한다. 그 결론은 현재 실무 기준으로는 softsign과 relu가 강한 기본 선택지이며, 그 이상을 찾는 방향은 selu 같은 theory-based design, swish 같은 automated search, 그리고 adaptive activation learning이라는 것이다.  

실제로 이 논문이 주는 가장 큰 기여는 “activation 선택은 취향 문제가 아니라 trade-off 문제”라는 점을 분명히 한 데 있다. boundedness, differentiability, derivative shape, expressivity, simplicity는 서로 긴장 관계에 있고, 좋은 activation은 이들 사이에서 어떤 균형을 택하느냐에 따라 달라진다. 그 의미에서 이 논문은 특정 함수 추천서라기보다, activation을 고르는 사고방식을 정리한 논문으로 읽는 것이 가장 적절하다.  
