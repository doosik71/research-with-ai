# GELU Activation Function in Deep Learning: A Comprehensive Mathematical Analysis and Performance

## 1. Paper Overview

이 논문은 **GELU (Gaussian Error Linear Unit)** activation function을 수학적·실험적으로 종합 분석하는 성격의 논문이다. 핵심 목표는 단순히 “GELU가 잘 된다”는 경험적 사실을 반복하는 것이 아니라, 왜 GELU가 deep learning에서 유리한지에 대해 **수학적 성질**과 **실험 성능**을 함께 정리하는 데 있다. 초록 기준으로 저자는 GELU의 **differentiability, boundedness, stationarity, smoothness** 를 분석하고, residual convolutional network를 사용해 **CIFAR-10, CIFAR-100, STL-10** 에서 다양한 activation과 비교 실험을 수행했다고 밝힌다. 또한 그 결과 GELU가 다른 activation들보다 우수한 성능을 보였다고 주장한다.

이 논문의 위치를 더 정확히 말하면, 2016년 Hendrycks와 Gimpel의 원래 GELU 제안 논문처럼 “새 activation을 처음 제안하는 논문”이라기보다, **이미 널리 쓰이는 GELU를 재조명하고 그 성질을 체계적으로 정리하는 분석형 논문** 에 가깝다. 특히 Introduction에서는 GELU가 BERT, ViT, GPT 같은 대표 모델들에 사용되었다고 언급하며, GELU가 modern deep learning에서 사실상 표준 activation 중 하나가 되었다는 배경을 강조한다.

즉, 이 논문의 기여는 “새 activation 제안”보다도, **GELU의 이론적 해석과 실전 성능을 하나의 문맥에서 정리해 practitioner가 activation 선택을 더 잘 하게 돕는 것**에 있다. 그런 점에서 survey와 analysis paper의 중간 성격을 가진다고 볼 수 있다.

## 2. Core Idea

이 논문의 핵심 아이디어는 간단하다. GELU는 이미 널리 쓰이고 있지만, 실제로는 많은 사용자가 “ReLU보다 부드럽다” 정도의 직관만 갖고 쓰는 경우가 많다. 저자는 여기서 한 걸음 더 나아가, GELU를 다음 두 관점에서 다시 정리한다.

첫째, **수학적 관점**이다. GELU가 differentiable하고 smooth하다는 것은 잘 알려져 있지만, 논문은 여기서 더 나아가 boundedness와 stationarity 같은 특성까지 함께 다루겠다고 말한다. 이는 activation function을 단순히 모양(shape)만으로 평가하지 않고, optimization 관점에서 해석하려는 시도다.

둘째, **실험적 관점**이다. 논문은 residual convolutional network를 testbed로 사용해 CIFAR-10, CIFAR-100, STL-10에서 다양한 activation들과 GELU를 비교하고, GELU가 전반적으로 더 나은 결과를 보인다고 보고한다. 즉, 이론과 실험을 따로 두지 않고 “수학적으로도 매끄럽고, 실제 성능도 좋다”는 이중 메시지를 전달하려는 구성이다.

또한 Introduction에서는 ReLU의 dying ReLU 문제를 다시 상기시키면서, GELU의 매끄러운 비선형성이 이런 문제를 완화할 수 있는 대안이라는 맥락을 준다. 동시에 GELU가 BERT, ViT, GPT 같은 대형 모델에 채택된 사례를 제시해, 이 논의가 단순한 이론적 취향의 문제가 아니라 실제 architecture design과 연결된다는 점도 강조한다.

## 3. Detailed Method Explanation

### 3.1 GELU의 기본 형태와 의미

GELU는 일반적으로 다음과 같이 알려져 있다.

$$
\mathrm{GELU}(x)=x\Phi(x)
$$

여기서 $\Phi(x)$ 는 표준 정규분포의 누적분포함수(CDF)다. 이 식의 직관은 입력을 ReLU처럼 hard-threshold로 자르지 않고, **입력의 크기에 따라 부드럽게 가중(weighting)** 한다는 것이다. 큰 양수는 거의 그대로 통과하고, 큰 음수는 강하게 억제되며, 0 근처에서는 연속적으로 변화한다.

이 논문은 바로 이 GELU의 성질을 deep learning 관점에서 다시 해석한다. 특히 smoothness와 differentiability는 ReLU와의 가장 직접적인 차이점이다. ReLU는 구현이 단순하고 계산 효율이 좋지만 0에서 미분 불가능하고, 음수 영역 전체를 일괄적으로 죽인다. 반면 GELU는 훨씬 부드러운 transition을 제공하므로, gradient flow와 optimization stability 측면에서 장점이 있을 수 있다. 초록과 서론에서 강조되는 이 논문의 출발점도 바로 여기에 있다.  

### 3.2 논문이 강조하는 수학적 분석 축

초록에 따르면 저자는 GELU에 대해 다음 성질들을 중점적으로 분석한다.

* differentiability
* boundedness
* stationarity
* smoothness

이 중 differentiability와 smoothness는 비교적 직관적이다. GELU는 연속적이고 미분 가능한 activation이므로, gradient-based optimization과 잘 맞는다. 이는 특히 layer가 깊어질수록 activation의 기울기 특성이 training dynamics에 큰 영향을 준다는 점에서 중요하다.

boundedness와 stationarity는 더 미묘한 포인트다. 업로드된 발췌본에서는 해당 정리나 증명 전개가 충분히 보이지 않아 논문이 각각을 어떤 엄밀한 정의로 다루는지는 여기서 단정하기 어렵다. 다만 문맥상 보면, 저자는 GELU의 함수값 또는 도함수의 거동, 극값 구조, 그리고 입력 구간별 변화율을 분석하면서 **optimization landscape에서 GELU가 어떤 inductive bias를 주는지** 설명하려는 것으로 읽힌다. 이 문장은 현재 보이는 초록·서론의 정보를 바탕으로 한 해석이다.

### 3.3 ReLU와의 비교 맥락

논문 서론은 ReLU가 여전히 가장 널리 쓰이는 activation이라고 인정하면서도, **dying ReLU problem** 을 중요한 한계로 지적한다. 즉, 음수 영역의 뉴런이 비활성 상태에 고착되면 learning이 방해될 수 있다는 오래된 문제가 여전히 relevance가 있다는 것이다.

이 맥락에서 GELU는 다음 같은 장점을 갖는 후보로 제시된다.

* 음수 영역을 무조건 0으로 만드는 hard gating이 아니다.
* 부드럽고 미분 가능하다.
* ReLU를 어느 정도 근사하면서도 더 정교한 비선형성을 제공한다.
* 이미 BERT, ViT, GPT 같은 현대 모델에서 실제로 사용된다.

따라서 이 논문은 “GELU가 왜 modern architecture에서 자주 채택되는가?”를 activation shape, 수학적 성질, 실험 결과를 통해 정리하는 구조라고 볼 수 있다.

### 3.4 실험 방법

초록에 따르면 실험은 **residual convolutional network** 를 공통 backbone/testbed로 사용하고, 데이터셋으로 **CIFAR-10, CIFAR-100, STL-10** 을 사용한다. 또한 “a broad range of alternative activation functions”와의 비교를 수행했다고 되어 있으므로, 실험 설계 의도는 단일 baseline 비교가 아니라 **activation family 전반에서 GELU의 상대적 위치를 평가**하려는 데 있다.

다만 현재 대화에서 확인 가능한 업로드 본문은 앞부분 위주로 잘려 있어, 비교된 activation의 정확한 목록, optimizer 세팅, epoch 수, 정규화 방식, 최종 표(table)의 상세 수치까지는 이 대화 내에서 확정적으로 읽어낼 수 없었다. 그래서 아래 실험 해석에서는 초록 수준에서 확인 가능한 주장과, 그로부터 합리적으로 해석 가능한 범위만 다룬다.

## 4. Experiments and Findings

이 논문이 실험적으로 가장 강하게 주장하는 것은, GELU가 residual convolutional network 환경에서 **CIFAR-10, CIFAR-100, STL-10** 전반에 걸쳐 다른 activation들보다 더 나은 성능을 보였다는 점이다. 초록은 이를 매우 직접적으로 말하며, GELU의 “superior performance”를 논문의 핵심 결론 중 하나로 둔다.

이 결과가 의미 있는 이유는 두 가지다.

첫째, 데이터셋 구성이 단일 benchmark에 국한되지 않는다. CIFAR-10과 CIFAR-100은 클래스 granularity가 다르고, STL-10은 상대적으로 데이터 특성이 또 다르다. 따라서 한 데이터셋에서만 우연히 좋은 것이 아니라, **서로 다른 이미지 분류 조건에서 비교적 일관된 우세**를 보였다는 메시지를 전달하려는 것으로 보인다. 이 해석은 초록의 서술을 기반으로 한다.

둘째, backbone이 residual convolutional network라는 점도 중요하다. 이는 단순 MLP toy example이 아니라 실제 vision backbone에서 activation의 차이를 평가했다는 뜻이므로, practitioner 관점에서 더 실용적이다. 특히 modern vision architecture에서도 activation 선택이 여전히 성능 차이를 만들 수 있음을 시사한다.

다만 중요한 한계도 있다. 현재 업로드된 본문 조각만으로는 **정확한 수치 테이블, standard deviation, 통계적 유의성, 비교 대상 activation 목록** 을 끝까지 확인할 수 없었다. 따라서 “몇 %포인트 더 좋았다” 같은 정량적 비교를 여기서 단정적으로 쓰는 것은 적절하지 않다. 현 시점에서 안전하게 말할 수 있는 것은, **논문이 초록 수준에서 GELU의 우수성을 강하게 주장한다**는 점까지다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **이론과 실험을 함께 묶었다는 점**이다. GELU를 둘러싼 많은 논의는 보통 둘 중 하나로 치우친다. 어떤 글은 “GELU는 BERT에 쓰인다” 같은 경험적 사실만 말하고, 어떤 글은 수식만 다룬다. 반면 이 논문은 적어도 의도 수준에서는 GELU의 수학적 특성과 benchmark 성능을 함께 설명하려고 한다. practitioner에게는 이런 정리가 꽤 유용하다.

또 다른 강점은 **현대 모델과의 연결성** 을 분명히 해 준다는 점이다. GELU가 BERT, ViT, GPT 같은 모델들에 채택되었다는 점을 명시함으로써, activation 함수 논의가 단순한 교과서적 이슈가 아니라 실제 state-of-the-art 모델 설계의 일부라는 점을 잘 보여 준다.

### 한계

반면 한계도 분명하다. 첫째, 이 논문은 원천적으로 **새 activation을 제안하는 논문이 아니라 GELU를 분석·정당화하는 논문** 이다. 따라서 novelty는 “완전히 새로운 아이디어”보다는 “정리와 해석” 쪽에 있다. 연구적 참신성만 놓고 보면 2016년 원논문보다 강한 인상을 주기는 어렵다.

둘째, 서론의 일부 표현은 다소 포괄적이고 일반적이다. 예를 들어 GELU와 normalization의 상호작용까지 언급하지만, 현재 확인 가능한 발췌본만으로는 이 부분이 실험과 이론에서 얼마나 깊게 다뤄지는지 분명하지 않다. 즉, 문제 제기는 넓지만 실제 기여의 중심은 결국 **GELU의 성질 요약 + vision benchmark 비교** 쪽일 가능성이 높다. 이 평가는 현재 보이는 본문 범위를 바탕으로 한 해석이다.

셋째, 지금 대화에서 확보된 본문이 앞부분에 치우쳐 있어, 논문의 정밀한 가치 판단에 필요한 **후반부 정리, proof details, full result tables** 를 확인하지 못했다. 따라서 이 보고서는 논문의 핵심 방향과 주요 주장에는 충실하지만, 세부 수학 증명과 정량 결과까지 완전히 복원한 보고서는 아니다.

### 해석

비판적으로 해석하면, 이 논문의 진짜 가치는 “GELU가 최고다”를 새로 선언하는 데 있지 않다. 오히려 **왜 GELU가 modern deep learning에서 사실상 표준으로 자리 잡았는지를 이론과 실험의 언어로 정리한다**는 데 있다. 즉, activation 선택을 경험적 관습이 아니라 좀 더 구조적인 판단으로 끌어올리려는 시도라고 볼 수 있다.

또한 이 논문은 GELU를 단지 ReLU의 smooth replacement로 설명하는 데서 멈추지 않고, activation function selection 자체를 더 넓은 설계 선택의 일부로 보게 만든다. 이 점은 실제 모델 튜닝에서 꽤 중요하다. activation은 흔히 “기본값 그대로 두는 옵션”처럼 취급되지만, 이 논문은 그것이 여전히 성능과 학습 안정성에 영향을 주는 핵심 요소임을 상기시킨다.

## 6. Conclusion

이 논문은 GELU를 deep learning의 대표 activation 중 하나로 놓고, 그 **수학적 성질**과 **실험적 성능**을 함께 분석하는 종합 연구다. 초록과 서론 기준으로 보면, 저자는 GELU의 differentiability, boundedness, stationarity, smoothness를 다루고, residual convolutional network를 활용해 CIFAR-10, CIFAR-100, STL-10에서 여러 activation과 비교 실험을 수행했다. 그 결과 GELU가 전반적으로 더 우수한 성능을 보였다고 주장한다.

이 논문을 한 문장으로 요약하면, **“GELU는 단지 많이 쓰이는 activation이 아니라, 이론적으로도 설명 가능하고 실험적으로도 강한 activation이다”** 라는 메시지를 정리한 논문이다. 다만 현재 확인 가능한 업로드 범위만으로는 후반부 세부 증명과 결과표를 모두 읽을 수 없어, 정량 수치 중심의 평가는 보류해야 한다. 그럼에도 불구하고 이 논문은 GELU를 이해하고 activation selection의 의미를 다시 생각해 보려는 독자에게 충분히 유용한 분석 자료라고 볼 수 있다.
