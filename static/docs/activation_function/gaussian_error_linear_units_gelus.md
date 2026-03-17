# Gaussian Error Linear Units (GELUs)

## 1. Paper Overview

이 논문은 신경망의 activation function으로 **Gaussian Error Linear Unit (GELU)** 를 제안한다. 저자들은 ReLU가 입력의 부호(sign)에 따라 hard gating을 수행하는 반면, GELU는 입력을 단순히 통과시키거나 자르는 대신 **입력값의 크기에 따라 연속적으로 가중(weighting)** 한다고 설명한다. 수식으로는 GELU가 $x\Phi(x)$ 로 정의되며, 여기서 $\Phi(x)$ 는 표준 정규분포의 누적분포함수(CDF)이다. 논문의 핵심 주장은 GELU가 ReLU, ELU보다 더 자연스러운 확률적 해석을 가지면서도, 컴퓨터 비전·자연어처리·음성 과제 전반에서 더 나은 성능을 보였다는 점이다.  

이 논문이 중요한 이유는 activation function을 단순한 비선형 함수가 아니라, **정보를 얼마나 통과시킬지 결정하는 soft stochastic gate** 로 다시 해석했다는 데 있다. ReLU는 $x>0$ 이면 모두 통과, 아니면 모두 차단하는 매우 거친 규칙을 쓰지만, GELU는 입력이 클수록 더 많이 통과시키고 작을수록 덜 통과시키는 방식을 취한다. 이 관점은 dropout류 regularization과 activation을 더 가깝게 연결한다는 점에서도 의미가 있다.

## 2. Core Idea

GELU의 핵심 아이디어는 **입력 의존적 stochastic masking의 기대값(expected transformation)** 을 activation으로 사용하자는 것이다. 저자들은 neuron input $x$ 에 대해 Bernoulli mask $m \sim \text{Bernoulli}(\Phi(x))$ 를 곱하는 상황을 생각한다. 그러면 입력이 클수록 통과될 확률이 높고, 작을수록 0이 될 확률이 높아진다. 이 stochastic process의 기대값을 취하면 바로 GELU가 된다.

즉,

$$
\mathbb{E}[mx] = x\Phi(x)
$$

이 되고, 이것이 GELU의 정의다. 논문은 이를 통해 GELU가 dropout, zoneout, ReLU의 성질을 연결하는 activation이라고 설명한다. ReLU가 입력의 sign만 보고 hard decision을 내리는 반면, GELU는 **입력의 상대적 크기** 에 따라 soft decision을 내린다.

저자들의 표현을 빌리면, GELU는 입력을 sign 기준으로 gate하는 ReLU와 달리, **입력값 자체에 비례해 weighting** 한다. 이 차이가 작은 입력과 큰 입력을 더 섬세하게 구분하게 만들고, 그 결과 optimization과 generalization 모두에서 이득을 줄 수 있다는 것이 논문의 중심 메시지다.

## 3. Detailed Method Explanation

### 3.1 GELU의 정의

논문은 GELU를 다음과 같이 정의한다.

$$
\text{GELU}(x)=xP(X\le x)=x\Phi(x)
$$

여기서 $X \sim \mathcal{N}(0,1)$ 이고, $\Phi(x)$ 는 표준 정규분포의 누적분포함수다. 이를 error function으로 쓰면

$$
\text{GELU}(x)=x\cdot \frac{1}{2}\left[1+\text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

가 된다.

이 식의 의미는 직관적이다. 큰 양수 입력은 $\Phi(x)\approx 1$ 이므로 거의 그대로 통과한다. 큰 음수 입력은 $\Phi(x)\approx 0$ 이므로 거의 0에 가까워진다. 하지만 ReLU처럼 정확히 0으로 딱 잘라내지는 않고, 중간 영역에서는 부드럽게 transition한다. 그래서 GELU는 ReLU보다 smooth하고, ELU처럼 음수 영역을 전부 saturation시키는 방식과도 다르다.  

### 3.2 확률적 해석

논문은 GELU를 단순한 smooth activation으로 소개하는 데 그치지 않고, **Adaptive Dropout의 기대값** 으로 해석한다. 구체적으로 입력 $x$ 를 Bernoulli mask로 살리거나 죽이되, 그 mask probability를 $\Phi(x)$ 로 두면 activation의 출력 기대값은 $x\Phi(x)$ 가 된다. 이 때문에 GELU는 stochastic regularization과 activation을 분리된 두 요소로 보기보다, 하나의 연속적 관점에서 이해하게 만든다.  

이 해석은 ReLU와의 차이를 더 분명히 보여 준다. ReLU는 사실상 $x\mathbf{1}_{x>0}$ 로 쓸 수 있어 sign-based gating이다. 반면 GELU는 입력이 얼마나 “유의미하게 큰지”에 따라 통과 비율이 달라진다. 그래서 작은 양수도 무조건 살리지 않고, 약한 음수도 완전히 버리지 않는다. 이는 noisy feature나 애매한 intermediate representation을 다룰 때 더 부드러운 inductive bias를 제공한다는 의미로 읽을 수 있다.

### 3.3 근사식

정확한 GELU는 CDF 또는 erf 계산이 필요하므로, 논문은 더 빠른 근사도 제안한다. 대표적인 근사식은 다음과 같다.

$$
0.5x\left(1+\tanh\left[\sqrt{\frac{2}{\pi}}\left(x+0.044715x^3\right)\right]\right)
$$

또는 더 빠른 feedforward speed를 원할 때

$$
x\sigma(1.702x)
$$

도 사용할 수 있다고 설명한다.

실제로 오늘날 널리 알려진 GELU 구현은 첫 번째 tanh 근사식인 경우가 많다. 이 논문 시점에서 이미 exact form과 fast approximation을 함께 제시했다는 점이 실용적으로 중요하다. 즉, GELU는 수학적으로는 Gaussian CDF 기반이지만, 구현 측면에서는 충분히 빠르게 사용할 수 있도록 설계되었다.

### 3.4 관련 activation과의 차이

논문은 GELU를 ReLU, ELU와 직접 비교한다. ReLU는 부호 기반 hard gating이고, ELU는 음수 영역을 smooth하게 열어 두는 activation이다. GELU는 이 둘과 달리 **확률적 soft gating** 이라는 해석을 갖는다. 또 논문은 logistic CDF를 사용하면 $x\sigma(x)$ 형태의 **SiLU** 도 얻을 수 있다고 언급하지만, 본 논문에서는 $\mu=0, \sigma=1$ 의 Gaussian CDF를 사용하는 GELU를 중심으로 실험한다. 추가적인 hyperparameter도 도입하지 않는다.

## 4. Experiments and Findings

### 4.1 실험 과제 구성

논문은 GELU, ELU, ReLU를 다음 과제에서 비교한다.

* MNIST classification
* MNIST autoencoding
* Twitter POS tagging
* TIMIT frame recognition/classification
* CIFAR-10/100 classification

즉, 단순 이미지 분류뿐 아니라 **비전, NLP, speech** 까지 걸쳐 activation의 일반성을 확인하려는 구성이다. 논문 abstract에서도 저자들은 GELU가 이들 전반에서 개선을 보였다고 요약한다.

### 4.2 MNIST Classification

MNIST 분류 실험에서 논문은 GELU가 dropout 유무와 관계없이 **가장 낮은 median training log loss를 보이는 경향** 이 있다고 말한다. 또한 입력에 uniform noise를 추가한 robustness 실험에서도 GELU가 ReLU, ELU와 비슷하거나 그 이상 수준의 강건성을 보였다고 설명한다. 즉, GELU는 dropout과도 잘 맞고 noisy input에 대해서도 안정적이라는 메시지를 준다.

이 결과는 “smooth activation은 dropout과 잘 안 맞을 수 있다”는 우려를 반박하는 쪽으로 읽힌다. 오히려 GELU는 stochastic regularizer에서 유도된 activation답게 dropout 환경에서도 자연스럽게 작동했다고 해석할 수 있다.

### 4.3 MNIST Autoencoder

MNIST autoencoder에서는 논문이 더 강한 표현을 쓴다. 저자들은 GELU가 **different learning rates를 잘 수용하며**, ReLU와 ELU보다 **significantly outperforms** 한다고 서술한다. 특히 learning rate를 바꿔 가며 비교했을 때 ELU는 큰 학습률에서 발산했고, GELU와 ReLU는 상대적으로 안정적이었지만 최종적으로는 GELU가 가장 좋은 결과를 냈다고 적고 있다.

이 부분은 GELU의 장점이 단순히 classifier accuracy 하나에 국한되지 않고, representation learning이나 reconstruction 계열에서도 나타날 수 있음을 보여 준다. 즉, GELU는 optimization landscape 자체를 좀 더 다루기 쉽게 만들어 줄 가능성이 있다.

### 4.4 Twitter POS Tagging

Twitter POS tagging 실험에서는 논문이 구체적인 수치를 제시한다. median test error는

* GELU: **12.57%**
* ReLU: **12.67%**
* ELU: **12.91%**

였다. 차이는 크지 않지만, 적은 데이터의 NLP setting에서도 GELU가 가장 낮은 에러를 기록했다는 점이 중요하다. 저자들은 이를 통해 GELU가 small-data generalization에도 잘 작동한다고 주장한다.

### 4.5 TIMIT Frame Classification

TIMIT phone/frame classification에서도 GELU가 가장 좋은 수치를 보였다. 논문에 따르면 median test error는

* GELU: **29.3%**
* ReLU: **29.5%**
* ELU: **29.6%**

였다. 여기서도 개선 폭은 크지 않지만 일관적이다. 즉, 음성 인식 계열의 깊은 fully connected classifier에서도 GELU가 우세했다.

### 4.6 CIFAR-10 / CIFAR-100

CIFAR-10/100 section의 확장된 문서 조각에서는, GELU가 shallow CNN과 deep CNN에서도 **again outperforms other nonlinearities** 라는 서술이 확인된다. 다만 현재 업로드된 HTML에서 확장 가능한 조각은 중간에 잘려 있어, 최종 수치 테이블 전체를 이 대화 내에서 완전하게 확인할 수는 없었다. 따라서 정확한 CIFAR 최종 error 수치를 여기서 단정적으로 적기는 어렵다. 다만 논문 본문이 분명히 전달하는 메시지는 **보다 복잡한 convolutional architecture에서도 GELU가 ReLU/ELU 대비 우수했다** 는 점이다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 activation을 **확률적 기대값** 으로 재해석했다는 점이다. 많은 activation 논문이 함수 모양의 경험적 개선에 집중하는 반면, GELU는 dropout/zoneout/ReLU를 하나의 관점으로 묶으면서 왜 이런 형태가 자연스러운지 설명한다. 즉, 단순히 “smooth해서 좋다”가 아니라 “입력 의존적 stochastic gate의 expectation이라서 좋다”는 해석을 준다.

또 하나의 강점은 실험 범위다. 비전, NLP, speech에 걸쳐 일관된 개선을 보였고, 특히 Twitter POS와 TIMIT처럼 구조가 다른 과제들에서도 우세했다. 이는 GELU가 특정 도메인 전용 trick이 아니라 꽤 일반적인 activation 개선일 수 있음을 시사한다.  

### 한계

한편 이 논문의 한계도 있다. 첫째, GELU의 성능 향상 폭은 일부 과제에서 매우 크지 않다. Twitter POS, TIMIT에서는 improvement가 일관적이지만 margin은 작다. 둘째, 정확한 GELU는 Gaussian CDF 계산을 포함하므로 ReLU보다 계산이 무겁다. 물론 논문은 tanh approximation과 sigmoid approximation을 제시해 이를 완화한다. 셋째, 당시 기준으로는 Transformer 같은 이후 대규모 구조에 대한 검증은 아직 없었다. 따라서 이 논문 자체만 보면 GELU의 강점은 주로 당시의 CNN/MLP 계열 실험에 의해 뒷받침된다.  

### 해석

비판적으로 해석하면, GELU의 진짜 공헌은 “ReLU보다 조금 더 smooth한 함수”를 제안한 것이 아니라, **activation을 gating probability와 연결하는 사고방식** 을 제시한 데 있다. 이후 modern deep learning에서 GELU가 크게 자리 잡은 이유도, 단순히 empirical win 때문만이 아니라 이런 soft selection bias가 깊은 모델의 intermediate representation과 잘 맞았기 때문이라고 볼 수 있다. 이 논문은 그 출발점을 제공한 셈이다. 이 마지막 문장은 논문 결과를 바탕으로 한 해석이다.  

## 6. Conclusion

이 논문은 **Gaussian Error Linear Unit (GELU)** 를 제안하며, 이를

$$
\text{GELU}(x)=x\Phi(x)
$$

로 정의했다. GELU는 ReLU처럼 입력을 hard-threshold하지 않고, 입력의 크기에 따라 연속적으로 가중하는 activation이다. 또한 Bernoulli mask 기반 stochastic regularizer의 기대값으로 해석할 수 있어, dropout과 activation 사이의 개념적 연결도 제공한다.  

실험적으로는 MNIST, Twitter POS, TIMIT, CIFAR 등 다양한 과제에서 ReLU와 ELU보다 일관되게 좋거나 최소한 동등 이상의 성능을 보였다. 특히 논문에 명시된 수치 기준으로 Twitter POS와 TIMIT에서는 GELU가 가장 낮은 test error를 기록했다. 따라서 이 논문은 현대 activation 설계에서 매우 중요한 전환점으로 볼 수 있다.
