# GLU Variants Improve Transformer

이 논문은 Transformer의 FFN(Feed-Forward Network) 서브레이어에서 보통 쓰는 ReLU나 GELU 대신, **GLU 계열의 gated activation** 을 쓰면 성능이 더 좋아질 수 있는지를 실험적으로 검증한 짧고 명확한 논문이다. 저자는 기존 Transformer의 FFN을 단순한 두 개의 선형변환 + activation으로 보지 않고, **입력을 두 경로로 투영한 뒤 element-wise 곱으로 결합하는 gating 구조**로 바꾸면 더 나은 표현력을 얻을 수 있다고 본다. 실제로 T5 스타일의 text-to-text transfer setup에서 여러 GLU 변형을 비교한 결과, **GEGLU와 SwiGLU가 가장 좋은 perplexity와 전반적으로 더 좋은 downstream 성능**을 보였다고 보고한다.

## 1. Paper Overview

이 논문의 문제의식은 매우 구체적이다. Transformer는 multi-head attention과 position-wise FFN을 번갈아 쌓는데, 당시 FFN의 기본 activation은 주로 ReLU였고 후속 연구에서는 GELU나 Swish 같은 대안도 제안되었다. 하지만 저자는 FFN의 비선형성을 “한 번의 activation”으로 처리하는 대신, **한 경로가 다른 경로를 gate하는 multiplicative 구조**를 쓰는 편이 더 적절할 수 있다고 본다. GLU 자체는 원래 gated convolutional networks에서 소개된 개념이지만, 이 논문은 그것을 Transformer의 FFN에 옮겨와 체계적으로 비교한다.  

왜 이 문제가 중요한가도 분명하다. Transformer의 FFN은 attention만큼 주목받지는 않지만, 전체 파라미터와 계산량에서 매우 큰 비중을 차지하며 각 위치의 hidden state를 재가공하는 핵심 모듈이다. 따라서 FFN activation의 작은 변화도 모델 전체 성능에 꽤 큰 영향을 줄 수 있다. 이 논문은 바로 그 지점을 겨냥해, **Transformer 개선이 attention 설계에만 있지 않고 FFN activation 구조에도 있다**는 메시지를 전달한다.

## 2. Core Idea

기존 FFN은 bias 없는 T5 스타일로 쓰면 대략 다음과 같다.

$$
\mathrm{FFN}_{\mathrm{ReLU}}(x,W_1,W_2)=\max(xW_1,0)W_2
$$

또는 GELU/Swish를 쓰면

$$
\mathrm{FFN}_{\mathrm{GELU}}(x,W_1,W_2)=\mathrm{GELU}(xW_1)W_2
$$

$$
\mathrm{FFN}_{\mathrm{Swish}}(x,W_1,W_2)=\mathrm{Swish}_1(xW_1)W_2
$$

가 된다. 즉, 하나의 projection에 activation을 적용하고 다시 projection하는 구조다.  

저자가 제안하는 핵심은 이를 **두 개의 projection + 곱셈 게이트**로 바꾸는 것이다. 기본 GLU는

$$
\mathrm{GLU}(x,W,V,b,c)=\sigma(xW+b)\otimes(xV+c)
$$

이고, activation을 빼면 bilinear layer가 된다. 여기서 $\otimes$ 는 element-wise product다. 이 구조의 의미는 직관적이다. 한 projection은 “content”를 만들고, 다른 projection은 그것을 얼마나 통과시킬지 gate 역할을 한다. 논문은 sigmoid 대신 다른 비선형을 쓸 수 있다고 보고, 여러 GLU variants를 정의한다.  

핵심적으로 비교한 변형들은 다음과 같은 계열이다.

* **Bilinear**: activation 없이 두 선형 투영을 바로 곱함
* **GLU**: sigmoid gate
* **ReGLU**: ReLU gate
* **GEGLU**: GELU gate
* **SwiGLU**: Swish gate

즉, 이 논문의 메시지는 “ReLU vs GELU” 같은 단일 activation 비교를 넘어서, **곱셈적 gating 자체가 더 강한 inductive bias일 수 있으며, 그중 GELU/Swish 기반 gate가 특히 좋다**는 것이다.

## 3. Detailed Method Explanation

### 3.1 Transformer FFN의 재설계

논문은 원래 FFN이 다음처럼 point-wise로 동작한다고 본다.

* 입력 hidden state $x$ 를 intermediate dimension으로 올린다.
* activation을 적용한다.
* 다시 model dimension으로 내린다.

하지만 GLU 계열에서는 intermediate 표현을 하나만 만들지 않고 **두 개 만든 뒤 곱한다**. 이는 단순 activation보다 더 풍부한 interaction을 제공한다. 하나의 경로가 다른 경로를 scaling/gating하기 때문에, 네트워크는 additive nonlinearity보다 더 정교한 feature selection을 할 수 있다. 이 해석은 논문이 직접 길게 설명하진 않지만, 제시된 수식 구조 자체가 보여 주는 설계 의도다.

### 3.2 각 변형의 형태

논문 본문에서 핵심적인 GLU family는 다음 식들로 이해할 수 있다.

기본 GLU:
$$
\sigma(xW)\otimes xV
$$

ReGLU:
$$
\mathrm{ReLU}(xW)\otimes xV
$$

GEGLU:
$$
\mathrm{GELU}(xW)\otimes xV
$$

SwiGLU:
$$
\mathrm{Swish}_1(xW)\otimes xV
$$

Bilinear:
$$
(xW)\otimes xV
$$

즉, 구조는 같고 **gate 쪽 activation만 바뀐다**. 이 덕분에 비교가 깔끔하다. 논문은 바로 이 family를 Transformer FFN 자리에 넣고 성능을 비교한다.

### 3.3 파라미터 수와 hidden dimension 보정

중요한 구현 포인트는, GLU 계열은 원래 FFN보다 projection matrix가 하나 더 필요하므로 그대로 쓰면 파라미터 수가 늘어난다는 점이다. 이를 공정하게 비교하기 위해 논문은 **hidden dimension을 기존 대비 $2/3$로 줄인다**고 명시한다. 이렇게 하면 추가된 세 번째 matrix를 감안해도 전체 FFN 파라미터/연산량 규모를 기존과 비슷하게 맞출 수 있다.

이 부분은 논문 해석에서 중요하다. 성능 향상이 단순히 파라미터 수 증가 때문이 아니라, **같은 예산 안에서 구조가 더 효율적이었기 때문**이라는 주장을 가능하게 해 준다.

### 3.4 실험 설정

논문은 T5 transfer-learning setup을 따른다. encoder-decoder Transformer를 denoising objective로 pre-training한 뒤, SQuAD와 GLUE, SuperGLUE 혼합 태스크로 fine-tuning한다. pre-training에서는 Adafactor와 inverse-square-root learning-rate schedule을 사용했고, **pre-training 중 dropout은 쓰지 않았으며**, 이는 더 나은 결과를 준다고 보고한다. training quality는 held-out C4 shard의 log-perplexity로 평가했다. 또한 batch는 128 examples, 입력 길이 512 token, 출력 길이 114 token이며, 각 training step은 TPUv2 32-core cluster에서 약 0.15초가 걸렸다고 적는다. fine-tuning은 131072 step, learning rate $10^{-3}$, dropout 0.1을 사용한다.  

## 4. Experiments and Findings

### 4.1 Pre-training perplexity

논문의 가장 직접적인 결과는 pre-training perplexity다. held-out C4 shard의 log-perplexity를 비교한 결과, **GEGLU와 SwiGLU가 가장 좋은 perplexity**를 기록했다고 명시한다. 이는 단순히 downstream fine-tuning noise 때문이 아니라, pre-training objective 자체에서 이미 GLU variants가 더 좋은 optimization/representation을 보였다는 뜻이다.  

### 4.2 Downstream language understanding

fine-tuning에서는 SQuAD, GLUE, SuperGLUE development set 결과를 비교한다. 논문은 결과가 다소 noisy하다고 인정하지만, **새로운 GLU variants가 대부분의 task에서 가장 좋은 결과를 냈다**고 요약한다. 즉, improvement가 한두 task의 우연이 아니라 전반적 경향이라는 것이 저자의 해석이다.

### 4.3 어떤 변형이 가장 좋은가

논문 전체를 관통하는 결론은 다음이다.

* 단순 ReLU/GELU FFN보다 GLU family가 더 좋다.
* 그중에서도 **GEGLU와 SwiGLU가 가장 안정적으로 강하다**.
* Bilinear나 기본 GLU도 의미 있지만, 최고 성능은 GELU/Swish gate 쪽에서 나왔다.  

이 결과는 흥미롭다. 단순 activation 교체만으로는 GELU/Swish가 ReLU보다 약간 좋을 수 있지만, **gated multiplicative structure와 결합될 때 그 장점이 더 크게 드러난다**는 해석이 가능하다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 아이디어가 매우 단순하면서도 결과가 일관적이라는 점이다. Transformer 전체 구조를 바꾸지 않고, FFN activation만 gating 구조로 바꿨는데도 pre-training perplexity와 downstream task 성능이 모두 개선되었다. 게다가 hidden dimension을 $2/3$로 줄여 비교했기 때문에 “그냥 더 큰 모델이라서 좋았다”는 반론도 어느 정도 막는다.  

또 하나의 강점은 이후 영향력이다. 논문 자체는 짧지만, **SwiGLU/GEGLU 계열 FFN**은 이후 대형 Transformer나 LLM 계열에서 매우 널리 채택되는 설계가 된다. 이건 논문 바깥의 역사적 해석이지만, 논문 안에서도 이미 그 설계의 실용성과 단순성을 강하게 보여 준다. 논문 자체는 “simple to implement, no apparent computational drawbacks”라고 정리한다.

### 한계

한계도 분명하다. 저자는 왜 이 구조가 잘 작동하는지에 대해 **거의 설명하지 않는다**. 결론 문단에서도 농담조로 “why these architectures seem to work”에 대해 설명하지 못한다고 말한다. 즉, 이 논문은 강한 empirical paper이지, mechanistic explanation paper는 아니다.

또한 실험은 주로 T5-style text-to-text transfer setup에 집중되어 있어, vision이나 다른 modality에 일반화되는지까지는 이 논문만으로 말하기 어렵다. 물론 이후 역사를 보면 널리 일반화되지만, 이 논문 자체의 증거는 Transformer/T5 문맥에 한정된다.

### 해석

비판적으로 보면, 이 논문의 진짜 기여는 “새 activation 하나를 발명했다”보다 **Transformer FFN을 additive nonlinearity에서 multiplicative gating으로 재해석했다**는 데 있다. ReLU/GELU FFN은 정보를 한 경로로만 가공하지만, GLU 계열은 “내용”과 “통과 정도”를 분리한다. 이 차이가 모델의 표현력을 실제로 더 효율적으로 만든다는 것이 이 논문의 핵심 통찰이다. 이 해석은 논문의 수식 구조와 실험 결과를 바탕으로 한 것이다.

## 6. Conclusion

이 논문은 Transformer의 FFN에 GLU family를 적용해 **ReGLU, GEGLU, SwiGLU, Bilinear** 같은 변형을 비교했고, 그 결과 **GEGLU와 SwiGLU가 ReLU/GELU 기반 FFN보다 더 좋은 perplexity와 더 나은 downstream 결과**를 보였다고 결론낸다. 구현은 단순하고, hidden dimension을 $2/3$로 조절해도 성능 이득이 유지되며, 특별한 계산상 불이익도 없다고 주장한다.

실무적으로 이 논문은 매우 중요한 포인트를 준다. Transformer 성능 개선이 attention 설계나 normalization에만 있는 것이 아니라, **FFN 내부 activation/gating 구조**에도 크게 달려 있다는 것이다. 이후 많은 모델이 SwiGLU/GEGLU를 채택하게 된 배경을 생각하면, 이 논문은 짧지만 영향력 있는 설계 제안 논문으로 볼 수 있다.
