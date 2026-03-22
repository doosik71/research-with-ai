# A Study on ReLU and Softmax in Transformer

## 1. Paper Overview

이 논문은 Transformer의 두 핵심 구성요소인 **FFN(Feed-Forward Network)**과 **SAN(Self-Attention Network)**을 “key-value memory” 관점에서 다시 해석하면서, 그 차이를 만들어내는 본질적 요인이 **activation function**, 즉 ReLU와 Softmax의 차이에 있다고 주장한다. 기존 연구는 FFN과 메모리 네트워크가 형식적으로 유사하다는 점은 밝혔지만, FFN은 ReLU를 쓰고 메모리/attention은 Softmax를 쓰기 때문에 두 모듈이 실제로 완전히 동일하다고 보기 어렵다는 문제의식에서 출발한다. 저자들은 이 차이를 이론·실험적으로 분석해, **Softmax에 추가적인 layer normalization을 붙이면 FFN과 key-value memory가 사실상 동등해질 수 있다**고 보인다. 또한 slot 수가 많아질수록 ReLU가 Softmax보다 더 유리하며, 이 성질을 self-attention까지 확장한 **ReLUFormer**를 제안해 긴 시퀀스 번역에서 Transformer보다 더 나은 성능을 보였다고 보고한다.

이 문제가 중요한 이유는, Transformer 내부의 FFN과 attention을 단순히 “다른 블록”으로 보는 대신 **모두 메모리 연산의 변형**으로 통합해 이해할 수 있게 해 주기 때문이다. 특히 긴 입력에서 Softmax attention이 왜 비효율적일 수 있는지, FFN의 ReLU가 왜 큰 hidden dimension에서 강한지, 그리고 activation choice가 모델의 표현력과 안정성에 어떤 영향을 미치는지 설명하려는 시도라는 점에서 의미가 있다.  

## 2. Core Idea

논문의 핵심 아이디어는 두 단계로 요약할 수 있다.

첫째, **FFN과 key-value memory의 차이는 구조 자체보다 activation/normalization 차이에서 온다**는 점이다.
FFN은 대략

$$
H=\mathrm{ReLU}(XW_1^T+b_1)W_2+b_2
$$

형태이고, key-value memory는

$$
H=\mathrm{Softmax}(XK^T)V
$$

형태인데, 기존에는 이 둘이 비슷하다고만 보았다. 저자들은 실제 차이는 ReLU가 **정규화 없이 양수 부분만 통과**시키는 반면, Softmax는 **모든 slot에 대해 exponential normalization**을 수행한다는 데 있다고 지적한다. 이 차이 때문에 Softmax 출력은 분산이 작고, 상위 몇 개 slot에 과도하게 집중되는 경향을 갖는다.

둘째, ReLU의 장점을 self-attention에까지 확장하려면 단순 치환으로는 안 되고, **분산 제어 장치가 필요하다**는 점이다.
FFN이나 memory에서는 ReLU가 큰 slot 수에서 더 잘 작동하지만, self-attention에서는 sequence length에 따라 ReLU attention의 분산이 커지므로 그대로 쓰면 학습이 불안정해진다. 따라서 저자들은 **variance reduction factor**와 **regularization loss**를 도입해 이를 안정화하고, 그 결과 FFN과 SAN 모두 ReLU 기반으로 구성된 **ReLUFormer**를 제안한다.

## 3. Detailed Method Explanation

### 3.1 FFN과 key-value memory의 재해석

논문은 Transformer의 FFN을 key-value memory와 비교한다. FFN에서 $W_1$은 key, $W_2$는 value처럼 해석될 수 있다. 입력 $X$가 각 hidden slot과 상호작용해 activation을 만들고, 그 activation으로 value를 가중합한다는 점에서 memory lookup과 유사하다. 다만 FFN은 ReLU를 사용하고, 전통적인 key-value memory는 Softmax를 사용한다. 저자들의 문제제기는 “이 activation 차이를 무시한 상태에서 둘을 동일하다고 볼 수 있는가?”이다.

### 3.2 ReLU와 Softmax의 차이: variance와 normalization

저자들은 두 activation의 차이를 **variance**와 **normalization**이라는 두 축에서 설명한다.

Softmax는 모든 element를 합이 1이 되도록 정규화하므로, 출력값이 작아지고 분산도 작아진다. 이 경우 residual connection이 FFN 출력보다 상대적으로 더 지배적이 되어, 현재 layer의 표현력이 충분히 활용되지 못할 수 있다. 반면 ReLU는 normalization을 하지 않기 때문에 더 큰 분산을 가질 수 있고, 이 때문에 표현력이 높아질 수 있다. 논문은 Softmax로 ReLU를 대체했을 때 BLEU가 34.22에서 33.08로 떨어졌다고 보고하며, 이 하락의 핵심 원인을 activation 차이로 해석한다.

또한 Softmax는 exponential normalization을 하기 때문에, slot 수가 많아질수록 몇 개의 큰 score에 확률질량이 집중된다. 그 결과 대부분의 slot은 사실상 활용되지 못하고, 메모리 공간이 덜 다양하게 사용된다. ReLU는 이런 강한 집중화가 없기 때문에 더 많은 slot 정보를 활용할 수 있다.  

### 3.3 FFN과 key-value memory를 연결하는 방법

Softmax가 성능이 떨어지는 주된 이유가 분산 축소와 과도한 집중화라면, 이를 보정하면 FFN과 key-value memory는 더 가까워질 수 있다. 저자들은 바로 이 지점에서 **Softmax 뒤에 layer normalization(LN)**을 추가한다. 그러면 작은 분산 문제와 과도한 집중화가 상당 부분 완화되고, Softmax 기반 FFN이 ReLU 기반 FFN과 비슷한 성능을 낼 수 있다고 주장한다. 즉, 논문의 중요한 결론 하나는 다음과 같다.

$$
H=\mathrm{LN}(\mathrm{Softmax}(XK^T)V)
$$

처럼 LN을 추가하면, FFN과 key-value memory는 기능적으로 거의 동등하게 볼 수 있다. 이는 기존의 “FFN은 메모리처럼 보인다” 수준의 서술을 넘어서, **적절한 정규화 조건하에서는 둘이 동등하다**는 좀 더 강한 주장이다.  

### 3.4 큰 slot 수에서 ReLU가 유리한 이유

논문은 memory slot 수 또는 FFN hidden dimension이 커질수록 ReLU가 더 좋은 이유도 분석한다. 실험에서 hidden size를 32부터 4096까지 증가시키며 ReLU, Softmax, Softmax+LN을 비교한 결과, ReLU는 전 구간에서 Softmax보다 낫고, Softmax+LN은 많이 따라오지만 아주 큰 크기(예: 3072, 4096)에서는 ReLU가 다시 우세해진다. 저자들은 그 이유를 Softmax의 **centralized distribution**에서 찾는다. 즉, slot이 많아질수록 Softmax는 실제로는 소수의 slot만 보게 되고, 나머지 컨텍스트를 활용하지 못한다.  

### 3.5 Self-attention에 ReLU를 적용하는 문제와 해결책

이제 저자들은 FFN에서 확인한 ReLU의 장점을 self-attention으로 확장한다. SAN은 원래도 key-value computation 형태이므로 ReLU를 쓰면 긴 sequence에서 더 유리할 수 있을 것처럼 보인다. 하지만 실제로는 **Softmax를 ReLU로 바로 바꾸면 수렴하지 않는다**. 이유는 ReLU attention의 출력 분산이 입력 길이 $n$에 따라 커지는 동적인 성질을 가지기 때문이다. 긴 시퀀스일수록 분산이 커져 variance exploding이 발생하고, 학습이 불안정해진다.  

이를 해결하기 위해 논문은 두 가지를 도입한다.

* **attention scale factor**: 길이에 따른 분산 폭증을 억제
* **regularization loss / normalization loss**: attention 분포가 지나치게 나빠지는 것을 방지

ablation 결과에서 scale factor를 제거하면 모델이 아예 수렴하지 않았고, normalization loss를 제거하면 BLEU가 1.37 하락하고 entropy도 1.52 감소했다. 이는 단순한 ReLU 치환이 아니라, **ReLU attention을 안정적으로 작동시키기 위한 분산 보정과 분포 regularization이 핵심 구성요소**임을 뜻한다.  

### 3.6 ReLUFormer

이렇게 해서 저자들은 FFN과 SAN 모두에 ReLU를 적용한 **ReLUFormer**를 제안한다. 이 모델은 FFN을 **global key-value memory**, SAN을 **local key-value memory**로 보는 통합적 해석 위에 서 있으며, Transformer 전체를 memory network 관점에서 재구성하는 시도다.  

## 4. Experiments and Findings

### 4.1 FFN/Memory equivalence 실험

저자들은 먼저 activation을 바꿨을 때 FFN 성능이 어떻게 달라지는지 본다. Softmax로 FFN을 대체하면 성능이 유의미하게 하락한다. 하지만 LN을 추가하면 성능과 variance ratio가 크게 회복되어 ReLU와 유사한 수준에 도달한다. 이 실험은 “FFN과 memory가 구조적으로 닮았다”는 기존 서술을 넘어서, **Softmax+LN이 붙은 memory가 FFN과 거의 동등하게 행동할 수 있다**는 논문의 핵심 실험적 근거다.  

### 4.2 slot 수 증가 실험

hidden dimension 또는 slot 수를 32에서 4096까지 변화시키는 실험에서 ReLU는 consistently Softmax보다 우수했다. Softmax+LN은 상당 부분 격차를 줄였지만, slot 수가 아주 커지면 ReLU가 다시 더 좋아졌다. 이는 큰 메모리 공간에서 ReLU가 다양한 slot 정보를 더 잘 활용하는 반면, Softmax는 상위 slot에만 집중해 정보 이용이 비효율적이기 때문이라는 해석을 뒷받침한다.

### 4.3 sentence-level translation에서의 self-attention 실험

논문은 self-attention에서 ReLU를 안정적으로 쓰기 위한 제안들이 실제로 효과가 있는지 sentence-level translation에서도 검증한다. ablation 결과, scale factor가 없으면 학습이 붕괴하고, normalization loss를 제거하면 BLEU와 entropy가 모두 유의미하게 떨어진다. 즉, ReLU self-attention의 장점은 단순 activation 치환이 아니라 **분산 안정화 설계 전체**에 의해 실현된다.  

### 4.4 document-level translation

가장 중요한 실험은 긴 입력·출력을 갖는 document translation이다. 저자들은 Europarl7 En-De 기반의 긴 문서를 구성하고, 길이를 ${128,256,512,1024,2048}$ 로 바꾸며 ReLUFormer, vanilla Transformer, Sparsemax를 비교한다. 결과는 짧은 길이(128, 256)에서는 ReLUFormer가 baseline들과 대체로 비슷하지만, 긴 길이(512, 1024, 2048)에서는 **일관되게 더 우수**하다는 것이다. 특히 길이 1024에서 ReLUFormer는 Europarl7에서 **1.15 BLEU 향상**을 보였고, Sparsemax는 긴 시퀀스에서 수렴하지 못했다. 이는 논문의 핵심 주장, 즉 “긴 시퀀스에서는 Softmax의 과도한 집중화가 불리하고 ReLU가 더 효과적”이라는 점을 직접적으로 보여 준다.

### 4.5 시각화 분석

시각화 결과도 저자들의 해석을 뒷받침한다. document translation에서 Softmax는 attention이 더 중앙집중적이고, ReLU는 더 넓은 컨텍스트를 활용한다. 예시로 길이 1024의 Europarl7 샘플에서 ReLU attention은 “Madam”, “President” 같은 더 먼 상관관계까지 포착했지만, Softmax는 그 가중치가 작았다. 또한 ReLU는 쉼표나 “this” 같은 stop word에 더 작은 attention을 부여해 noise가 적다고 보고된다. 이런 결과는 ReLU가 long-range correlation을 더 잘 포착한다는 정성적 근거다.

## 5. Strengths, Limitations, and Interpretation

이 논문의 강점은 Transformer의 두 모듈을 하나의 memory 관점으로 통합하면서도, 그 차이를 단순 구조가 아니라 **activation-induced dynamics**로 설명했다는 점이다. 특히 ReLU와 Softmax를 variance와 normalization 차원에서 분리해 해석하고, FFN·memory·attention에 공통 논리를 적용한 점이 설득력 있다. 또한 긴 sequence라는 실제로 중요한 setting에서 document translation 실험으로 주장을 검증했다는 점도 강하다.  

한계도 있다. 첫째, self-attention에서는 ReLU를 그대로 쓰면 안 되고 scale factor와 regularization이 필요하므로, “ReLU가 Softmax보다 본질적으로 우월하다”기보다는 **적절한 안정화 설계가 들어간 ReLU attention이 긴 시퀀스에서 유리하다**고 보는 편이 정확하다. 둘째, 논문은 주로 translation, 특히 long document translation 중심으로 검증했기 때문에 이 결과가 모든 attention 기반 과제에 곧바로 일반화된다고 말하기는 어렵다. 셋째, Softmax의 집중성이 항상 나쁜 것은 아니다. 짧은 시퀀스나 핵심 토큰 몇 개에 강하게 집중하는 문제가 더 유리한 상황에서는 Softmax가 여전히 적합할 수 있다. 실제로 짧은 길이에서는 ReLUFormer가 baseline 대비 큰 우위를 보이지 않는다.  

비판적으로 해석하면, 이 논문의 더 큰 의미는 “Transformer에서 attention=Softmax”라는 관습을 흔들고, **activation choice 자체가 메모리 활용 방식과 long-context 성능을 크게 바꾼다**는 점을 보여 준 데 있다. 이후 linear attention, kernelized attention, sparse attention 계열을 이해할 때도 이 논문의 문제의식은 여전히 유효하다.

## 6. Conclusion

이 논문은 Transformer의 FFN과 self-attention을 모두 key-value memory 관점에서 재해석하면서, ReLU와 Softmax의 차이를 **variance**와 **normalization** 측면에서 체계적으로 분석했다. 핵심 결론은 세 가지다.

1. Softmax에 layer normalization을 추가하면 FFN과 key-value memory는 사실상 동등해질 수 있다.
2. slot 수가 많을수록 ReLU는 Softmax보다 더 다양한 정보를 활용할 수 있어 유리하다.
3. self-attention에서는 분산 폭증 문제를 해결하는 추가 설계가 필요하지만, 이를 갖춘 ReLUFormer는 긴 시퀀스 번역에서 baseline Transformer보다 더 좋다.

실용적으로 보면, 이 논문은 “긴 컨텍스트에서 왜 Softmax attention이 비효율적일 수 있는가”에 대한 하나의 명료한 설명을 제공한다. 또한 Transformer 내부를 메모리 네트워크로 통합적으로 해석할 수 있게 해 준다는 점에서도 의미가 크다.
