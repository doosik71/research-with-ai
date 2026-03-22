# Continuously Differentiable Exponential Linear Units

## 1. Paper Overview

이 논문은 ELU(Exponential Linear Unit)가 가진 장점을 유지하면서도, 기존 ELU의 수학적 결함 하나를 제거하는 매우 짧고 집중된 논문이다. 저자는 기존 ELU가 $\alpha \neq 1$ 일 때 입력 $x$ 에 대해 **continuous는 맞지만 derivative가 $x=0$ 에서 불연속**이라는 점을 지적한다. 이는 activation을 해석하고 튜닝할 때 불편함을 만들고, 특히 큰 $\alpha$ 에서는 음의 작은 입력 근처에서 gradient가 커질 수 있어 학습 안정성을 해칠 수 있다. 이를 해결하기 위해 저자는 ELU를 재매개변수화한 **CELU(Continuously Differentiable ELU)**를 제안한다. CELU는 모든 $\alpha$ 값에 대해 $C^1$ 연속이며, derivative가 bounded이고, linear function과 ReLU를 special case 또는 극한 형태로 포함하며, $\alpha$ 에 대해 scale-similar하다는 성질을 갖는다.  

이 문제가 중요한 이유는 activation function이 단지 경험적 성능뿐 아니라 optimization dynamics, gradient stability, 해석 가능성에도 큰 영향을 주기 때문이다. ELU는 mean activation을 0 근처로 유지하고 vanishing gradient 문제를 완화하는 장점 때문에 널리 관심을 받았지만, 저자는 그 parametrization 자체가 더 잘 정리될 수 있다고 본다. 즉, 이 논문은 “새 activation을 완전히 새로 만들자”가 아니라, **기존에 이미 유용한 ELU를 더 수학적으로 깔끔하고 튜닝하기 쉬운 형태로 다시 쓰자**는 문제의식을 가진다.

## 2. Core Idea

핵심 아이디어는 한 줄로 요약된다. 기존 ELU의 음수 구간

$$
\alpha(\exp(x)-1)
$$

을 그대로 쓰지 말고,

$$
\alpha\left(\exp\left(\frac{x}{\alpha}\right)-1\right)
$$

로 바꾸자는 것이다. 이렇게 하면 $x=0$ 에서 음수 쪽 derivative가 항상 1이 되어 양수 쪽 derivative와 정확히 이어진다. 그 결과 CELU는 모든 $\alpha$ 에 대해 $C^1$ continuous가 된다.

이 재매개변수화가 좋은 이유는 단순히 미분 가능성이 좋아졌다는 데서 끝나지 않는다. 저자는 CELU가 다음 성질을 추가로 가진다고 정리한다.

1. $x$ 에 대한 derivative가 bounded하다.
2. linear transfer function과 ReLU를 special case 또는 limit로 포함한다.
3. $\alpha$ 에 대해 scale-similar하다.

즉, CELU는 ELU의 practical intuition은 유지하면서도, 수학적 성질은 더 정제된 activation이다.  

## 3. Detailed Method Explanation

### 3.1 기존 ELU의 정의와 문제

논문은 먼저 기존 ELU를 다음과 같이 둔다.

$$
\operatorname{ELU}(x,\alpha)=
\begin{cases}
x & \text{if } x \ge 0 \
\alpha(\exp(x)-1) & \text{otherwise}
\end{cases}
$$

그리고 그 derivative는

$$
\frac{d}{dx}\operatorname{ELU}(x,\alpha)=
\begin{cases}
1 & \text{if } x \ge 0 \
\alpha\exp(x) & \text{otherwise}
\end{cases}
$$

이다. 여기서 문제는 $x=0$ 에서 나타난다. 양수 쪽 derivative는 1인데, 음수 쪽에서 $x \to 0^-$ 로 가면 derivative는 $\alpha$ 로 간다. 따라서 $\alpha \neq 1$ 이면 derivative가 끊긴다. 즉, ELU는 function 자체는 이어져 있어도 input에 대해 continuously differentiable하지 않다.

또한 저자는 큰 $\alpha$ 값을 쓰면, 작은 음수 입력에서 gradient가 커져 exploding gradient 비슷한 현상이 생길 수 있다고 지적한다. 이는 튜닝을 어렵게 만들 수 있다.

### 3.2 CELU의 정의

이를 해결하기 위해 제안하는 CELU는 다음과 같다.

$$
\operatorname{CELU}(x,\alpha)=
\begin{cases}
x & \text{if } x \ge 0 \
\alpha\left(\exp\left(\frac{x}{\alpha}\right)-1\right) & \text{otherwise}
\end{cases}
$$

여기서 핵심은 exponent 안에 $x/\alpha$ 가 들어간다는 점이다. 이 변화 하나로 음수 구간의 local slope가 조절되어, $x=0$ 에서 derivative가 항상 1로 맞춰진다. 저자는 이를 통해 CELU가 모든 $\alpha$ 에 대해 $C^1$ continuous라고 주장한다.

또한 $\alpha=1$ 이면 ELU와 CELU는 완전히 동일하다.

$$
\forall_x\ \operatorname{ELU}(x,1)=\operatorname{CELU}(x,1)
$$

즉, CELU는 ELU를 버리는 것이 아니라 **ELU의 자연스러운 일반화**라고 볼 수 있다.

### 3.3 CELU의 미분

논문은 CELU의 입력 $x$ 와 shape parameter $\alpha$ 에 대한 derivative도 제시한다.

입력에 대한 derivative는

$$
\frac{d}{dx}\operatorname{CELU}(x,\alpha)=
\begin{cases}
1 & \text{if } x \ge 0 \
\exp\left(\frac{x}{\alpha}\right) & \text{otherwise}
\end{cases}
$$

이다. 이 식에서 중요한 점은 음수 구간 derivative에 **앞의 $\alpha$ 가 사라진다**는 것이다. 기존 ELU에서는 $\alpha \exp(x)$ 였기 때문에 $\alpha$ 가 클수록 음수 경계 근처 slope도 커졌지만, CELU에서는 최대값이 1이다. 따라서 derivative가 bounded되고 큰 $\alpha$ 로 인한 급격한 gradient 증가가 방지된다.  

$\alpha$ 에 대한 derivative는

$$
\frac{d}{d\alpha}\operatorname{CELU}(x,\alpha)=
\begin{cases}
0 & \text{if } x \ge 0 \
\exp\left(\frac{x}{\alpha}\right)\left(1-\frac{x}{\alpha}\right)-1 & \text{otherwise}
\end{cases}
$$

로 주어진다. 이 식은 $\alpha$ 를 learnable parameter로 둘 가능성까지 열어 준다. 논문은 이를 깊게 확장하지는 않지만, 적어도 CELU가 parameterized activation으로 다루기 더 깔끔한 형태임을 보여 준다.

### 3.4 중요한 성질들

#### 3.4.1 Bounded derivative

CELU의 음수 구간 derivative는 $\exp(x/\alpha)$ 이므로, $x<0$ 에서 항상 0과 1 사이에 있다. 양수 구간에서는 1이다. 따라서 전체적으로 derivative가 bounded된다. 이는 기존 ELU의 “큰 $\alpha$ 에서 gradient가 커질 수 있음” 문제와 대비된다.  

#### 3.4.2 Linear function과 ReLU와의 관계

논문은 CELU가 linear transfer function과 ReLU를 special case로 포함한다고 설명한다. 정확히는 다음처럼 해석하는 것이 적절하다.

* $\alpha \to \infty$ 이면 음수 구간에서 $\exp(x/\alpha)\approx 1 + x/\alpha$ 이므로
  $$
  \alpha\left(\exp(x/\alpha)-1\right)\approx x
  $$
  가 되어 전체 함수가 거의 linear에 가까워진다.

* $\alpha \to 0^+$ 이면 음수 구간 출력이 0으로 수렴하여 ReLU에 가까워진다.

즉, CELU는 parameter 하나로 “ReLU에 가까운 sharp rectifier”부터 “거의 linear한 함수”까지 부드럽게 연결하는 family로 볼 수 있다.  

#### 3.4.3 Scale-similarity

논문은 CELU가 $\alpha$ 에 대해 scale-similar하다고 말한다. 직관적으로는 $\alpha$ 가 activation의 형태를 바꾸더라도, 단순히 amplitude만 바꾸는 것이 아니라 x축과 y축이 함께 조정되는 식으로 family 전체가 self-similar하게 움직인다는 뜻이다. 기존 ELU는 $\alpha$ 가 커질수록 음수 saturation level과 boundary slope가 함께 불균형하게 바뀌지만, CELU는 더 정돈된 방식으로 shape를 조절한다. 이 점이 tuning을 쉽게 만든다고 저자는 본다.

### 3.5 Shifted ReLU로의 확장 가능성

논문 말미에서는 CELU를 x축과 y축 방향으로 조금 shift하면, 작은 $\alpha$ 에서도 음수 activation을 허용하는 arbitrary shifted ReLU로 수렴하게 만들 수 있다고 언급한다. 이는 CELU가 단순히 하나의 activation이 아니라, 더 넓은 rectifier family의 기반 형태로도 해석될 수 있음을 보여 준다. 다만 이 부분은 제안의 본체라기보다 부가적 관찰에 가깝다.

## 4. Experiments and Findings

이 논문은 일반적인 benchmark 논문처럼 대규모 실험을 수행하지 않는다. 별도의 데이터셋, 네트워크, 정량 성능 비교 표가 제시되지 않는다. 대신 **수학적 성질과 함수/gradient의 shape**를 비교하는 것이 핵심이다. 이 점은 논문 해석에서 매우 중요하다. 이 논문은 “CELU가 ImageNet에서 몇 % 더 좋다”를 주장하는 empirical paper가 아니라, **ELU parametrization을 더 좋은 형태로 정리한 short note**에 가깝다.

실질적으로 논문이 보여 주는 “findings”는 Figure 1에 요약된다.

* 기존 ELU는 $\alpha \neq 1$ 에서 $x=0$ 에서 derivative가 불연속이다.
* 큰 $\alpha$ 에서는 음수의 작은 입력 근처에서 gradient가 커질 수 있다.
* CELU는 ELU의 장점을 유지하면서도 continuously differentiable하다.
* CELU는 scale-similar하고, bounded derivative를 가지며, linear와 ReLU를 포함하는 더 정돈된 family다.  

따라서 이 논문에서 “실험 결과”는 benchmark 성능 향상이라기보다, **기존 ELU와 새 parametrization의 함수형 비교를 통해 이론적/구조적 우월성을 보이는 것**이라고 이해해야 한다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 문제를 매우 정확하게 겨냥한다는 점이다. ELU는 이미 유용한 activation인데, 저자는 그 장점을 부정하지 않는다. 대신 **parametrization의 불연속 derivative 문제**를 딱 집어내고, 이를 최소한의 수정으로 해결한다. 수학적으로는 훨씬 더 깔끔해지고, 실용적으로는 $\alpha$ 튜닝이 쉬워지며, bounded derivative 덕분에 안정성도 좋아진다. 짧은 논문이지만 아이디어의 밀도가 높다.  

또한 CELU가 ELU를 완전히 대체하는 낯선 함수가 아니라, $\alpha=1$ 에서 ELU와 동일하고 $\alpha$ 의 변화에 따라 ReLU와 linear에 자연스럽게 연결된다는 점도 설계상 우아하다.

### 한계

한계도 명확하다. 첫째, 이 논문은 **대규모 empirical validation이 없다**. 즉, CELU가 실제 다양한 네트워크와 데이터셋에서 ELU보다 항상 더 낫다고 이 논문만으로 말할 수는 없다. 논문의 주장은 주로 수학적 성질의 개선에 기반한다.

둘째, activation function의 실제 성능은 initialization, normalization, optimizer, architecture와 상호작용하므로, 수학적으로 더 예쁜 parametrization이 항상 최종 성능 향상으로 이어진다고 단정할 수는 없다. 이 논문은 그 부분을 거의 다루지 않는다.

### 해석

비판적으로 보면, 이 논문의 가치는 “새로운 성능 혁명”보다도 **activation design에서 parametrization 자체가 중요하다**는 점을 보여 준 데 있다. 같은 직관을 가진 activation이라도, parameter가 gradient와 shape에 어떻게 개입하느냐에 따라 수학적 성질과 optimization behavior가 달라질 수 있다. CELU는 바로 그 점을 매우 명쾌하게 보여 주는 사례다.

## 6. Conclusion

이 논문은 기존 ELU가 $\alpha \neq 1$ 일 때 $x=0$ 에서 continuously differentiable하지 않다는 문제를 지적하고, 이를 해결하는 재매개변수화인 **CELU**를 제안했다. CELU는

* 모든 $\alpha$ 에 대해 $C^1$ continuous이고,
* 입력에 대한 derivative가 bounded되며,
* linear function과 ReLU를 포함하는 family를 이루고,
* $\alpha$ 에 대해 scale-similar하다.

즉, ELU의 장점을 보존하면서 더 수학적으로 정돈되고 튜닝하기 쉬운 activation으로 바뀐 것이다.  

실무적으로 보면, 이 논문은 activation을 새로 발명했다기보다 **기존 ELU를 더 “well-behaved”하게 만든 논문**으로 이해하는 것이 가장 정확하다. 아주 큰 empirical paper는 아니지만, activation parametrization을 설계할 때 무엇이 중요한지 보여 주는 좋은 예다.
