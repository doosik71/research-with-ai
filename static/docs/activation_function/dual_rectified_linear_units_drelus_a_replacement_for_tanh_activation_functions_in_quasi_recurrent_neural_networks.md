# Dual Rectified Linear Units (DReLUs): A Replacement for Tanh Activation Functions in Quasi-Recurrent Neural Networks

## 1. Paper Overview

이 논문은 순환신경망 계열, 특히 **Quasi-Recurrent Neural Networks (QRNNs)** 에서 널리 쓰이던 `tanh` 활성함수를 대체할 수 있는 새로운 활성화 단위인 **DReLU (Dual Rectified Linear Unit)** 를 제안한다. 저자들의 문제의식은 분명하다. ReLU는 feed-forward 비전 모델에서 강력한 성능을 보였지만, 순환 구조에서는 출력이 양수 영역에만 존재하기 때문에 hidden state를 “빼는 방향”으로 갱신하기 어렵다. 반면 `tanh`는 양수와 음수 출력을 모두 줄 수 있지만, 포화(saturation)와 vanishing gradient 문제에 취약하다. 이 논문은 바로 이 틈을 겨냥한다.  

핵심 제안은 간단하다. **두 개의 ReLU 출력을 서로 빼서 signed output을 만들자**는 것이다. 이렇게 하면 ReLU의 장점인 희소성(sparsity), vanishing gradient 완화, 계산 단순성을 어느 정도 유지하면서도 `tanh`처럼 양수와 음수 값을 모두 표현할 수 있다. 저자들은 이 구조가 QRNN의 recurrent step에서 `tanh`를 대체하는 **drop-in replacement** 로 동작할 수 있다고 주장한다.

논문은 단순한 아이디어 제안에 그치지 않고, 이를 **sentiment classification**, **word-level language modeling**, **character-level language modeling** 에 적용해 원래의 `tanh` 기반 QRNN 및 LSTM과 비교한다. 특히 **DReLU 기반 QRNN을 최대 8층까지 skip connection 없이 쌓을 수 있었다**는 점을 중요한 실험 결과로 내세운다.

## 2. Core Idea

이 논문의 핵심은 “ReLU의 장점은 유지하되, 출력 부호(sign)를 복원하자”로 요약할 수 있다.

표준 ReLU는 다음과 같다.

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

문제는 이 함수의 출력이 항상 0 이상이라는 점이다. 순환 모델의 hidden update에서는 새로운 정보가 기존 state에 더해질 뿐 아니라, 때로는 **감산적인 역할**도 필요하다. `tanh`는 이를 자연스럽게 처리하지만, 포화 구간에서 gradient가 약해지고 깊게 쌓을수록 학습이 불안정해질 수 있다.

저자들이 제안한 DReLU는 개념적으로 두 ReLU의 차이로 정의된다.

$$
\mathrm{DReLU}(a,b)=\max(0,a)-\max(0,b)
$$

이 정의는 매우 중요하다. 왜냐하면:

* 첫 번째 ReLU가 양의 방향 기여를 담당하고
* 두 번째 ReLU가 음의 방향 기여를 담당하며
* 둘 다 0이면 정확히 0을 만들 수 있기 때문이다.

즉, DReLU는 **ReLU처럼 정확한 zero activation** 을 만들 수 있으면서도, 전체 출력은 양수도 음수도 가능하다. 저자들은 이것이 `tanh`와 ReLU의 장점을 동시에 일부 갖는 구조라고 해석한다.

논문은 또 하나의 변형으로 **DELU (Dual Exponential Linear Unit)** 도 제안한다. 이는 ReLU 대신 ELU를 두 개 사용해 signed output을 만드는 방식이다. 하지만 논문의 중심 메시지는 DReLU 쪽에 있다.

## 3. Detailed Method Explanation

### 3.1 왜 RNN/QRNN에서 ReLU가 바로 쓰이기 어려운가

순환 구조는 시점마다 hidden state를 갱신해야 한다. 이때 candidate hidden state가 항상 비음수이면, state 갱신이 편향된 방향으로만 이루어질 수 있다. 논문은 바로 이 점을 지적한다. **ReLU는 strictly negative value를 만들 수 없기 때문에 hidden state에서 subtraction 역할을 수행하기 어렵다**는 것이다. 이것이 순환 구조에서 `tanh`가 여전히 많이 사용된 이유다.

### 3.2 QRNN이 왜 적합한 배경인가

QRNN은 전통적인 RNN/LSTM과 달리 hidden-to-hidden matrix multiplication을 크게 줄이고, convolution과 recurrent pooling을 결합한 하이브리드 구조다. 저자들은 이러한 특성 때문에 QRNN이 일반 RNN보다 rectified activation을 도입하기에 유리하다고 본다. 즉, recurrent computation이 비교적 단순하므로 activation 대체 효과를 더 직접적으로 검증할 수 있다.

고수준에서 보면 QRNN은 각 시점의 입력 시퀀스에 대해 convolution으로 candidate state 및 gate들을 만들고, recurrent pooling이 이를 누적한다. 이 논문에서 DReLU는 바로 그 **candidate hidden representation** 을 만드는 부분의 `tanh`를 대체한다.

### 3.3 DReLU의 구조적 성질

DReLU는 다음 성질을 갖는다.

첫째, **양의 출력과 음의 출력이 모두 가능**하다.
둘째, **정확히 0이 될 수 있다**.
셋째, ReLU처럼 **비포화 영역에서 gradient 흐름이 비교적 좋다**.
넷째, **희소 activation** 을 유도한다.  

이 네 번째 항목이 흥미롭다. `tanh`는 값이 작아질 수는 있어도 정확히 0이 되는 경우가 상대적으로 덜 자연스럽다. 저자들은 이것이 noise를 유발할 수 있다고 본다. 반면 DReLU는 두 ReLU가 모두 꺼질 수 있기 때문에 exact zero를 만들 수 있다. 이 점을 논문은 “noise robustness”와 연결한다.

### 3.4 DReLU와 tanh의 기능적 비교

논문이 전달하는 메시지는 DReLU가 `tanh`를 완전히 복제한다는 것이 아니다. 오히려 다음과 같은 절충이다.

* `tanh`의 장점: signed output
* ReLU의 장점: vanishing gradient 완화, sparse activation, 단순성
* DReLU의 목표: 두 장점을 QRNN의 recurrent step에서 결합

그래서 DReLU는 “새로운 비선형성”이라기보다, **순환 구조에 맞게 재구성된 signed rectifier** 로 이해하는 편이 적절하다.

### 3.5 DELU

논문은 DReLU 외에도 DELU를 소개한다. 이는 ELU의 음수 영역 saturation 특성을 이용해 보다 부드러운 dual activation을 구성하는 방식이다. 다만 가시적인 초반 설명과 기여 요약에서 중심은 DReLU이며, DELU는 보조적 확장으로 제시된다.

## 4. Experiments and Findings

논문이 명시적으로 다루는 실험 영역은 세 가지다.

### 4.1 Sentiment Classification

저자들은 DReLU 기반 QRNN을 감성분류 문제에 적용해 원래의 `tanh` 기반 QRNN, 그리고 LSTM과 비교했다고 설명한다. 이 실험의 목적은 DReLU가 단순히 언어모델링 같은 생성 태스크뿐 아니라 분류 태스크에서도 유효한지 확인하는 데 있다.

### 4.2 Word-level Language Modeling

두 번째 실험 축은 단어 단위 언어모델링이다. 여기서도 비교 기준은 `tanh`-QRNN과 LSTM이다. 논문의 visible abstract와 introduction만으로도, 저자들이 단순 제안이 아니라 **기존 QRNN 실험을 독립 재현한 뒤 activation만 바꾸어 비교** 하려 했다는 점이 드러난다. 즉, 공정한 대조를 지향한 실험 설계다.  

### 4.3 Character-level Language Modeling

가장 강조되는 실험은 character-level language modeling이다. 저자들은 이 설정에서 **최대 8개의 DReLU 기반 QRNN layer를 skip connection 없이 적층**할 수 있었다고 말하며, 이를 통해 shallow LSTM 기반 구조보다 더 좋은 결과, 나아가 당시 SOTA를 개선했다고 주장한다.  

이 메시지는 중요하다. DReLU의 의미는 단순히 `tanh`를 대체하는 데 그치지 않고, **더 깊은 QRNN stack을 학습 가능하게 만드는 activation 설계**에 있다는 것이다.

### 4.4 실험 결과 해석

현재 대화에 제공된 추출본에서는 주요 표와 수치가 중간에서 많이 잘려 있어, 각 벤치마크의 정확한 perplexity/error 수치를 신뢰성 있게 모두 재인용할 수는 없다. 다만 visible text 기준으로 분명한 결론은 다음과 같다.

* DReLU는 QRNN recurrent step에서 `tanh`를 대체하도록 설계되었다.
* 저자들은 이를 sentiment classification, word-level LM, character-level LM에 적용했다.
* 특히 깊은 character-level QRNN 적층에서 유리함을 강조했다.
* LSTM 및 원래 QRNN과의 비교를 통해 실용성을 주장했다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 장점은 **아이디어가 매우 단순하면서도 문제 정의가 정확하다**는 점이다. 순환 구조에서 ReLU를 못 쓰는 이유를 “출력 부호 부족”으로 요약하고, 이를 두 ReLU의 차로 해결했다. 이런 종류의 설계는 구현 난이도가 낮고, 기존 모델에 직접 넣어 보기 쉽다.

또 다른 장점은 **exact zero activation** 을 강조한 부분이다. `tanh`는 signed output은 되지만 sparse activation에는 불리하다. DReLU는 signed output과 sparsity를 동시에 어느 정도 만족시키므로, recurrent representation을 더 선택적으로 만들 가능성이 있다. 저자들이 말하는 noise robustness도 이 맥락에서 이해할 수 있다.

세 번째 장점은 **QRNN 깊이 확장성**이다. 논문은 DReLU를 통해 8층 적층이 가능했다고 주장하며, 이는 activation 함수 하나가 optimization landscape에 실제 영향을 줄 수 있음을 보여 준다.

### 한계

첫째, 논문의 주장은 상당 부분 **QRNN이라는 특정 구조**에 기대고 있다. 즉, DReLU가 모든 RNN, LSTM, GRU, Transformer 계열로 일반화되는지는 이 논문만으로는 충분히 말하기 어렵다.

둘째, DReLU는 개념적으로 단순하지만 실제 구현에서는 하나의 activation 대신 **두 개의 pre-activation 경로**가 필요하다. 계산량 증가가 아주 크진 않더라도, 표준 `tanh` 대비 완전히 공짜인 대체는 아니다.

셋째, visible snippet 기준으로는 왜 DReLU가 특정 태스크에서 더 잘 작동했는지에 대한 이론적 분석은 제한적이다. 논문은 성질을 직관적으로 설명하지만, PReLU나 He initialization 논문처럼 깊은 분산 전파 분석까지 제시하는 방향은 아니다.

### 해석

이 논문은 넓게 보면 “활성화 함수 연구”이지만, 더 정확히는 **순환 구조에서의 signed rectifier 설계**에 관한 논문이다. DReLU는 ReLU를 recurrent model에 억지로 넣은 것이 아니라, 순환 갱신에 필요한 음수 표현을 유지하면서 rectifier의 장점을 보존하려는 설계다.

그래서 이 논문의 의미는 “ReLU가 tanh보다 항상 낫다”가 아니라,
**“순환 구조에 맞는 형태로 rectifier를 재설계하면 tanh의 역할을 대체할 수 있다”**
는 데 있다.

## 6. Conclusion

이 논문은 QRNN의 recurrent step에서 `tanh`를 대체할 수 있는 새로운 activation인 **DReLU**를 제안했다. DReLU는 두 ReLU의 차로 구성되어, ReLU의 희소성과 gradient 흐름 장점을 유지하면서도 `tanh`처럼 양수/음수 출력을 모두 표현할 수 있다. 또한 exact zero activation이 가능해 noise robustness 측면의 장점도 주장한다.

실험적으로는 sentiment classification, word-level language modeling, character-level language modeling에서 tanh-based QRNN 및 LSTM과 비교되었고, 특히 **8층 DReLU-QRNN 적층**과 character-level modeling 성과가 핵심 메시지로 제시된다. 수치 표 전체는 현재 제공된 추출본에서 완전히 확인되지는 않았지만, 논문의 중심 기여와 방향성은 충분히 분명하다.  

종합하면, 이 논문은 순환신경망에 rectifier 계열 activation을 더 자연스럽게 도입하기 위한 실용적이고 영리한 시도이며, activation 설계가 sequence modeling의 depth/optimization 문제에 실제 영향을 줄 수 있음을 보여 주는 사례로 읽을 수 있다.
