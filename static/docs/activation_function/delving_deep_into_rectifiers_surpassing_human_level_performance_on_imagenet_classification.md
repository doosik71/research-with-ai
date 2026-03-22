# Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

## 1. Paper Overview

이 논문은 deep CNN의 성능 향상을 위해 두 가지를 동시에 다룬다. 첫째는 기존 ReLU를 일반화한 **PReLU (Parametric ReLU)**를 제안하는 것이고, 둘째는 **rectifier 비선형성을 고려한 initialization 방법**을 이론적으로 유도하는 것이다. 저자들의 문제의식은 분명하다. ReLU가 딥러닝 성공의 핵심 요소임에도, 당시 많은 연구가 네트워크 깊이·너비·구조에는 집중했지만 정작 **rectifier의 성질 자체**와, 그것이 매우 깊은 네트워크 학습에 미치는 영향은 충분히 분석하지 않았다는 것이다. 논문은 이 두 축을 통해 더 깊거나 더 넓은 CNN을 scratch부터 안정적으로 학습시키고, 결국 ImageNet 2012에서 top-5 test error 4.94%를 달성해 당시 보고된 인간 성능 5.1%를 처음 넘어섰다고 주장한다.

이 문제가 중요한 이유는 activation과 initialization이 단순 구현 디테일이 아니라, 학습 가능 깊이와 최종 성능을 직접 제한하기 때문이다. 깊은 rectifier network는 theoretically powerful하지만, 초기화가 맞지 않으면 gradient가 layer를 지나며 점점 사라지거나 커져 학습이 멈춘다. 저자들은 이 문제를 구조적 트릭이 아니라 **rectifier의 forward/backward variance propagation** 관점에서 분석하고, 그에 맞는 원리 기반 초기화를 제시한다. 이 점 때문에 이 논문은 단순히 PReLU를 소개한 논문이 아니라, 이후 널리 알려진 **He initialization**의 출발점으로도 중요하다.  

## 2. Core Idea

핵심 아이디어는 두 가지다.

첫째, ReLU의 음수 구간을 완전히 0으로 막아 두지 말고, 그 기울기를 학습 가능하게 만들자는 것이다. 이때 activation은

$$
f(y_i)=
\begin{cases}
y_i, & y_i>0 \
a_i y_i, & y_i\le 0
\end{cases}
$$

로 정의된다. 여기서 $a_i$는 음수 구간 slope를 조절하는 learnable parameter다. $a_i=0$이면 ReLU, 작은 상수면 Leaky ReLU가 되며, PReLU는 이 값을 데이터와 함께 end-to-end로 학습한다. 저자들의 직관은 “negative response도 완전히 버리지 말고, task와 layer에 맞게 적응적으로 살리자”는 것이다.

둘째, rectifier network의 initialization은 Xavier처럼 선형/대칭 activation 가정에 기대면 충분하지 않다는 점이다. ReLU류에서는 활성의 절반 정도가 0이 되므로, forward와 backward에서 variance가 layer마다 절반씩 줄어드는 효과가 있다. 저자들은 이를 보정하기 위해 weight variance를

$$
\mathrm{Var}[w] = \frac{2}{n}
$$

꼴로 두어야 한다고 유도한다. 여기서 $n$은 fan-in 또는 해당 layer의 입력 연결 수다. 이 초기화 덕분에 매우 깊은 rectifier 모델도 scratch부터 수렴할 수 있다고 보인다.

## 3. Detailed Method Explanation

### 3.1 PReLU의 정의와 의미

PReLU는 ReLU의 음수 구간 slope를 고정하지 않고 학습하는 activation이다. 수식으로 다시 쓰면

$$
f(y_i)=\max(0,y_i)+a_i\min(0,y_i)
$$

와 같다. 여기서 $a_i$를 channel마다 따로 두는 **channel-wise** 버전과, layer 전체에서 하나를 공유하는 **channel-shared** 버전을 모두 고려한다. 추가 파라미터 수는 총 channel 수와 같거나 layer 수 정도이므로 전체 weight 수에 비하면 매우 작아 overfitting 위험도 작다고 본다.  

이 설계의 포인트는 단순히 dead ReLU를 피하는 데 있지 않다. 논문은 Leaky ReLU처럼 음수 slope를 고정하면 정확도 향상이 거의 없었다는 기존 결과를 언급하면서, **고정된 작은 slope가 아니라 task-specific, channel-specific slope를 학습하는 것**이 중요하다고 주장한다. 즉, PReLU는 “negative activation을 허용하자”보다 “negative activation의 정도를 모델이 직접 결정하게 하자”에 가깝다.

### 3.2 PReLU의 최적화

PReLU의 $a_i$는 일반적인 backpropagation으로 다른 weight들과 함께 학습된다. 논문은 chain rule로 $a_i$의 gradient를 쉽게 구할 수 있음을 보이며, 이 파라미터 역시 SGD로 최적화한다. 실험에서는 $a_i=0.25$로 초기화했다고 밝힌다.

중요한 점은, PReLU가 네트워크를 구조적으로 크게 바꾸지 않는다는 것이다. convolution, pooling, FC 구조는 그대로 유지하고 activation만 바꾸므로, 성능 차이를 activation 효과로 비교하기 좋다.

### 3.3 Rectifier-aware initialization의 유도

논문의 두 번째 큰 기여는 initialization이다. 저자들은 layer를 통과할 때 activation의 분산이 어떻게 변하는지 분석한다. ReLU의 경우 입력이 대칭분포라고 보면 절반 정도가 0으로 잘리므로, 단순 선형 layer의 분산 보존식에 $1/2$ 요인이 추가된다. 그 결과 forward signal의 variance가 유지되려면 대략

$$
\frac{1}{2}n_l \mathrm{Var}[w_l] = 1
$$

이 되어야 하며, 따라서

$$
\mathrm{Var}[w_l] = \frac{2}{n_l}
$$

를 얻는다. backward propagation 쪽에서도 유사한 분석을 하며, 결국 같은 형태의 조건이 나온다. 논문은 forward 식만 써도 충분하고 backward 식만 써도 충분하다고 설명한다.  

이 초기화는 Xavier initialization과 중요한 차이가 있다. Xavier는 sigmoid/tanh나 선형 가정에 더 가깝고, rectifier가 만드는 비대칭성과 zeroing 효과를 직접 반영하지 않는다. 논문은 깊은 rectifier model에서는 그 차이가 실제 수렴 여부로 이어진다고 본다.  

### 3.4 아키텍처 설계 선택

논문은 초기화 덕분에 30-layer까지 수렴시킬 수 있음을 보이지만, 실제 ImageNet 최종 성능 향상은 “무조건 더 깊게”가 아니라 **더 넓게** 만드는 방향에서 더 컸다고 말한다. model A는 baseline large model, B는 더 깊은 버전, C는 더 넓은 버전이며, 저자들은 depth 증가가 accuracy saturation이나 degradation을 보일 수 있어 width 확장을 선택했다고 설명한다.  

이 부분은 이 논문의 중요한 현실 감각이다. initialization이 좋아졌다고 해서 깊이 증가가 곧바로 성능 향상으로 이어지지는 않는다는 점을 솔직하게 인정한다.

## 4. Experiments and Findings

### 4.1 ReLU 대비 PReLU의 효과

작은 ImageNet model에서 channel-wise PReLU는 10-view testing 기준 **top-1 32.64%, top-5 12.75%**를 기록해 ReLU보다 더 좋은 성능을 보였다. 논문은 small model과 large model 모두에서 PReLU가 일관되게 개선을 보였다고 정리한다.  

large model A에 대해서도 dense testing 비교 결과, multi-scale combination에서 PReLU가 ReLU 대비 **top-1 error 1.05%, top-5 error 0.23%**를 줄였다고 보고한다. 이 성능 향상은 “거의 추가 계산비용 없이” 얻어졌다고 저자들은 강조한다.  

### 4.2 Initialization 비교

22-layer model에서는 Xavier와 저자들의 initialization 모두 수렴하긴 했지만, 제안 초기화가 더 빨리 error를 줄였다. 정확도 차이는 크지 않아, 해당 실험에서 ReLU 모델은 Xavier로 33.90/13.44, 제안 초기화로 33.82/13.34 top-1/top-5를 기록했다. 즉, 중간 깊이에서는 “수렴 속도 개선”이 더 분명하고, 최종 정확도 차이는 크지 않았다.

하지만 30-layer 극심한 deep model에서는 차이가 훨씬 크다. 저자들의 초기화는 이 모델을 수렴시켰지만, Xavier는 학습이 거의 멈추고 gradient가 diminishing되었다고 보고한다. 이는 이 초기화가 정말로 “rectifier 깊은 모델의 학습 가능성”을 넓혀 준다는 핵심 근거다.  

### 4.3 그러나 더 깊다고 더 좋은 것은 아니다

흥미롭게도, 저자들은 30-layer model을 수렴시키는 데는 성공했지만, 그 모델의 최종 정확도는 14-layer model보다 오히려 나빴다고 말한다. 30-layer 모델은 **38.56/16.59**, 14-layer 모델은 **33.82/13.34** top-1/top-5 error였다. 즉, initialization은 optimization 문제를 해결했지만, architecture depth 자체의 generalization or degradation 문제까지 자동으로 해결하지는 못했다.

이 실험은 매우 중요하다. 이 논문은 흔히 “He initialization + PReLU”로 기억되지만, 실제 논문 메시지는 더 미묘하다. **optimization barrier를 낮추는 것**과 **더 깊은 모델이 실제로 더 낫다는 것**은 별개의 문제라는 점을 분명히 보여 준다.

### 4.4 최종 ImageNet 결과

논문의 최종 결과는 강렬하다. single-model PReLU-net은 **5.71% top-5 test error**를 달성해 기존 multi-model 결과들을 넘어섰고, multi-model ensemble은 **4.94% top-5 test error**를 기록했다. 이는 ILSVRC 2014 우승 GoogLeNet의 6.66%보다 26% 상대 개선이며, 당시 보고된 human-level 5.1%보다 낮아 처음으로 이를 넘어섰다고 주장한다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 empirical trick과 theoretical reasoning이 잘 결합되어 있다는 점이다. PReLU는 매우 단순하고 실용적이며, initialization은 단순 heuristic이 아니라 rectifier의 분산 전파를 분석해 유도된다. 그래서 논문 전체가 “작동한다”에서 끝나지 않고, **왜 작동하는가**까지 설명한다.  

또 하나의 강점은 결과의 규모다. ImageNet 대형 실험에서 single-model과 multi-model 모두 당시 최고 수준을 달성했고, 특히 human-level performance를 넘어섰다는 메시지는 논문의 역사적 임팩트를 크게 만들었다.

### 한계

하지만 한계도 명확하다. 첫째, PReLU의 개선 폭은 일관되지만 아주 압도적이라기보다는 **작지만 안정적인 개선**에 가깝다. 둘째, initialization이 deep optimization을 도와도, 깊이 증가 자체가 accuracy 향상으로 이어지지 않는다는 점을 논문이 스스로 인정한다. 즉, 이 논문은 depth 문제를 완전히 해결한 것이 아니다.  

셋째, 오늘날 관점에서 보면 이 논문은 batch normalization 이전/초기 시기의 work이기 때문에, 이후 등장한 BN, residual connection, modern optimizer와 함께 볼 필요가 있다. 실제로 후속 연구에서는 initialization alone보다 residual design이 더 큰 역할을 하게 된다.

### 해석

비판적으로 보면, 이 논문의 더 큰 의미는 PReLU 자체보다도 **rectifier network를 위한 principled initialization**에 있다. PReLU는 실용적 activation 개선이고, initialization은 이후 거의 모든 ReLU 계열 네트워크의 기본 규칙이 되었다. 따라서 이 논문은 activation paper이면서 동시에 optimization paper이고, 더 넓게는 modern CNN training recipe를 정립한 논문으로 읽는 것이 맞다.

## 6. Conclusion

이 논문은 두 가지 중요한 기여를 남겼다. 하나는 ReLU의 음수 기울기를 학습 가능하게 만든 **PReLU**이고, 다른 하나는 rectifier 비선형성을 반영한 **He initialization**이다. PReLU는 거의 비용 증가 없이 small/large ImageNet 모델 모두에서 ReLU보다 더 나은 성능을 보였고, initialization은 30-layer rectifier model까지 scratch부터 수렴시킬 수 있게 했다. 최종적으로 PReLU-net ensemble은 ImageNet 2012에서 4.94% top-5 error를 달성해 당시 human-level 성능을 넘어섰다.  

실무적으로는, 이 논문은 “activation을 약간 더 유연하게 만들 수 있다”는 점보다도, **깊은 ReLU network를 안정적으로 학습시키는 초기화 원리**를 제공했다는 점에서 더 오래 남았다. 이후 CNN의 표준 관행에 가장 깊게 스며든 것은 바로 그 부분이다.
