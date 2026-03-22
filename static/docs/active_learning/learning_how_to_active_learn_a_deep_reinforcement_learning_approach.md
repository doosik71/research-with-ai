# Learning how to Active Learn: A Deep Reinforcement Learning Approach

## 1. Paper Overview

이 논문은 active learning에서 흔히 쓰이는 uncertainty sampling 같은 **고정 heuristic** 대신, **데이터 선택 전략 자체를 학습**하자는 문제를 다룬다. 저자들은 active learning을 reinforcement learning 문제로 재정의하고, 어떤 unlabeled instance를 라벨링할지 결정하는 정책(policy)을 **deep Q-network**로 학습하는 **PAL (Policy-based Active Learning)**을 제안한다. 특히 이 논문의 초점은 일반적인 이미지 분류가 아니라 **cross-lingual named entity recognition (NER)**이며, 한 언어에서 시뮬레이션으로 학습한 active learning policy를 다른 언어로 **전이(transfer)**할 수 있다는 점을 핵심 기여로 내세운다. 논문 초록은 기존 heuristic의 효과가 데이터셋마다 크게 달라진다는 한계를 지적하고, 제안 방식이 전통적 active learning보다 일관된 개선을 보였다고 요약한다.  

이 문제가 중요한 이유는 low-resource language에서 annotation 비용이 특히 크기 때문이다. 기존 active learning은 보통 어느 정도 seed set이 있어야 uncertainty 같은 신호를 안정적으로 계산할 수 있는데, 저자들은 저자원 언어에서는 그 가정 자체가 비현실적일 수 있다고 본다. 따라서 “무엇을 고를지”를 hand-crafted rule로 정하는 대신, 더 데이터 의존적이고 적응적인 selection policy를 학습해 전이하는 것이 이 논문의 문제의식이다.

## 2. Core Idea

핵심 아이디어는 active learning을 **Markov Decision Process (MDP)**로 보는 것이다. 즉, 현재까지 선택되어 라벨링된 데이터와 현재 candidate instance, 그리고 그 데이터로 학습된 supervised model의 상태를 관찰한 뒤, 에이전트가 “이 샘플을 라벨링할지 말지”를 binary action으로 결정한다. 이렇게 보면 기존 heuristic은 사실상 고정 정책이며, 그러면 정책도 학습할 수 있어야 한다는 것이 저자들의 출발점이다.

PAL의 novelty는 두 층위에 있다. 첫째, active learning policy를 **deep reinforcement learning** 으로 학습한다는 점이다. 둘째, 그 policy를 단일 언어 안에서만 쓰지 않고, **cross-lingual word embeddings** 를 통해 다른 언어로 이식할 수 있게 설계했다는 점이다. 저자들은 policy가 문장 내용(content), supervised model의 예측(classification), confidence를 함께 관찰하므로, uncertainty heuristic보다 더 풍부한 의사결정이 가능하다고 주장한다.  

## 3. Detailed Method Explanation

### 3.1 Active learning의 MDP 정식화

논문은 stream-based active learning을 가정한다. 즉, unlabeled sentence들이 순차적으로 도착하고, 각 시점마다 agent가 그 문장을 라벨링 요청할지 결정한다. 이때 상태 $s_i$는 현재 candidate sentence와 지금까지 축적된 labeled set, 그리고 그 labeled set으로 학습된 모델 $p_\phi$를 포함한다. 행동 집합은 단순히 $A={0,1}$ 이며, 1이면 라벨링 요청, 0이면 건너뛴다. 라벨링된 문장은 training set에 추가되고, supervised model은 그에 따라 업데이트된다. 이 과정을 annotation budget이 소진되거나 데이터가 끝날 때까지 반복한다.

논문은 이를 MDP 튜플 $\langle S, A, Pr(s_{i+1}\mid s_i, a), R\rangle$ 로 정식화한다. 이 정식화의 장점은 active learning을 더 이상 one-shot scoring 문제가 아니라, **과거 선택의 누적 효과가 미래 선택과 모델 품질에 영향을 미치는 sequential decision problem** 으로 다룰 수 있다는 점이다. 이것이 기존 uncertainty sampling과 가장 큰 차이다.

### 3.2 State representation

상태 표현은 논문에서 매우 중요한 부분이다. 저자들은 state를 continuous vector로 만들기 위해 세 가지 정보를 결합한다.

첫째, **문장 내용(content representation)**이다. 문장 $\mathbf{x}_i$ 는 CNN 기반 sentence encoder로 고정 길이 벡터로 변환된다. 이 부분은 Kim (2014) 스타일의 wide convolution과 max-pooling 구조를 따른다. 둘째, supervised model의 **predictive marginals** 를 직접 사용한다. 저자들은 entropy 같은 요약 통계만 쓰기보다, 예측 분포 자체를 넣는 편이 더 일반적이고 풍부한 신호를 줄 수 있다고 본다. 셋째, 모델의 **confidence 정보** 를 함께 넣는다. 이렇게 해서 policy network는 문장 의미와 현재 모델의 불확실성/예측 패턴을 동시에 볼 수 있다.

이 설계는 논문의 핵심 의도와 맞닿아 있다. 즉, 기존 active learning이 주로 uncertainty라는 단일 heuristic에 의존했다면, PAL은 문장 내용과 예측 분포의 구조까지 함께 보면서 더 복합적인 selection policy를 학습한다는 것이다.

### 3.3 Reward design

강화학습에서 reward 설계는 매우 중요하다. 가장 단순한 방법은 episode가 끝난 뒤 최종 held-out accuracy를 reward로 주는 것이지만, 저자들은 이것이 너무 delayed reward라 학습이 어렵다고 본다. 그래서 **reward shaping** 을 사용한다. 각 step의 intermediate reward를

$$
R(s_{i-1}, a) = \mathrm{Acc}(\phi_i) - \mathrm{Acc}(\phi_{i-1})
$$

처럼, 새 샘플 선택 후 모델의 held-out 성능이 얼마나 개선되었는지로 정의한다. 즉, 선택한 샘플이 실제로 모델 품질을 얼마나 높였는지를 즉시 피드백으로 주는 것이다. 이는 long-horizon credit assignment 문제를 완화하려는 설계다.

또한 policy 학습 시에는 여러 simulated active learning episode를 반복 수행한다. 각 episode마다 데이터를 섞고, 라벨은 요청될 때만 공개하며, held-out set은 reward 계산용으로 고정한다. episode가 끝나면 supervised model은 초기화되고, 데이터 순서와 evolving policy만 바뀐다. 이 구조는 policy가 특정 데이터 순서나 단일 run에 과적합되지 않도록 한다.  

### 3.4 Cross-lingual policy transfer

이 논문의 독특한 부분은 policy를 다른 언어로 옮기는 절차다. 저자들은 source language에서 학습한 policy를 target language에 적용하기 위해 **cross-lingual word embeddings** 를 사용한다. 이렇게 하면 서로 다른 언어의 문장도 공통 feature space에 매핑되므로, source language에서 배운 selection rule이 target language에서도 의미를 갖는다.  

transfer 단계의 알고리즘은 학습 단계와 비슷하지만 두 가지 차이가 있다. 첫째, target low-resource language에서는 annotation 비용이 크므로 **여러 pass가 아니라 한 번만 훑는다**. 둘째, source에서 가져온 초기 policy를 target episode에서 held-out 성능 기준으로 **fine-tune** 할 수 있다. 이 설정이 bilingual / multilingual transfer 실험의 기반이다. 논문은 추가로 이 방법이 전통적인 batch setting으로도 확장 가능하다고 짧게 언급한다.  

### 3.5 Cold-start transfer

저자들은 더 어려운 **cold-start** 시나리오도 제안한다. 이 설정에서는 held-out evaluation data도 없고, annotation 결과를 받아 policy를 다시 업데이트할 수도 없다. 즉, target language에서 **딱 한 번만 batch를 선택**해야 한다. 이를 위해 논문은 policy뿐 아니라 supervised model도 다른 언어에서 미리 학습해 target으로 옮긴다. 저자 설명에 따르면, 이 setting에서는 cross-lingual embeddings를 이용해 **policy와 model을 모두 transfer** 하며, 이를 통해 feedback이 전혀 없는 상태에서도 어느 정도 informative selection이 가능하게 만든다.  

이 점은 논문의 practical value를 높인다. 단순히 held-out set이 충분한 실험실 환경이 아니라, 실제 field linguistics처럼 supervision과 통신이 모두 제한된 상황까지 고려했다는 뜻이기 때문이다. 동시에 이 설정은 논문의 방법이 기존 uncertainty sampling보다 더 robust한 prior policy를 제공해야 한다는 강한 검증이기도 하다.

## 4. Experiments and Findings

### 4.1 실험 설정

실험은 CoNLL2002/2003 NER corpora를 사용하며, 언어는 **English (en), German (de), Spanish (es), Dutch (nl)**이다. 논문은 IOB1 라벨을 IO 라벨로 변환해 사용한다. 기존 corpus split를 그대로 사용해 `train` 을 policy 학습용, `testb` 를 reward 계산용 held-out, `testa` 를 최종 평가용으로 쓴다. 성능은 최종적으로 **F1 score** 로 보고된다.

설정은 세 가지다. **bilingual** 은 영어에서 policy를 학습하고 다른 한 언어를 target으로 쓴다. **multilingual** 은 target을 제외한 여러 언어를 source로 함께 써 policy를 학습한다. **cold-start** 는 영어 pretrained NER tagger를 사용해 source에서 policy를 학습하고, 이를 별도의 target language에 적용하는 더 어려운 설정이다. 실험 보고 시 target set에 budget 200 문장을 선택해 training set을 구성한다. Table 2 설명도 각 방법이 **200개 문장으로 training set을 구성**한다고 명시한다.

### 4.2 학습 세부사항

논문 조각에서 확인되는 하이퍼파라미터로는, policy 학습에서 discount factor $\gamma = 0.99$ 를 사용하고, optimizer로 **Adam**, mini-batch size 32를 사용한다. 이는 RL policy 학습이 비교적 표준적 deep Q-learning 세팅 위에 구현되었음을 보여준다. 다만 업로드된 HTML이 길이 제한으로 일부 표와 세부 네트워크 설정은 전부 보이지 않아, 본 보고서는 확인 가능한 범위만 반영한다.  

### 4.3 주요 결과: bilingual / multilingual

Figure 3과 관련 설명에 따르면, **PAL-b** 는 bilingual setting에서 세 target language 전부에 대해 **Random** 과 **Uncertainty** baseline을 일관되게 앞선다. 저자들은 uncertainty sampling이 특히 초기 단계에서 비효율적인데, 그 이유를 uncertainty 기반 방법이 충분히 좋은 초기 모델에 크게 의존하기 때문이라고 해석한다. 반면 PAL은 sentence content까지 보기 때문에, 초기 모델이 약한 상황에서도 더 나은 선택을 할 수 있다는 논리다.

또한 multilingual policy learning인 **PAL-m** 은 PAL-b를 포함한 모든 방법보다 더 좋은 성능을 보인다. 특히 Spanish와 Dutch에서 초기 단계의 우위가 두드러졌다고 서술하는데, 이는 여러 언어에서 학습한 policy가 문장 내용 기반 선택 패턴을 더 잘 익혔음을 시사한다고 해석한다. 즉, 이 논문은 단순 “영어에서 배운 정책도 다른 언어에서 된다”를 넘어서, **여러 언어에서 policy를 jointly 학습하면 더 좋은 transferable policy가 된다**는 점까지 보여준다.

### 4.4 Cold-start 결과

cold-start setting은 더 인상적인 실험이다. 여기서는 held-out data가 없으므로 policy와 model을 업데이트하지 못하고, 모든 annotation이 batch로 도착한다. 그럼에도 논문 conclusion과 실험 설명은 PAL 계열이 이 매우 어려운 setting에서도 baseline보다 우수했다고 정리한다. Figure 4 설명에 따르면, 모든 방법은 pretrained cross-lingual NER tagger 덕분에 약 **40%** 정도에서 시작하지만, PAL은 그 위에서 더 나은 샘플을 골라 성능을 끌어올린다.  

논문 결론은 이 결과를 두고, **평가 데이터도 없고 annotation에 반응할 수도 없는 환경**에서도 learned policy transfer가 작동한다고 강조한다. 이는 전통적 heuristic보다 learned policy가 더 강한 prior를 제공한다는 실험적 근거로 읽을 수 있다.

### 4.5 실험이 실제로 보여주는 것

이 논문의 실험은 절대적인 수치 하나보다도 **일관성** 이 중요하다. abstract, Figure 3/4 설명, conclusion이 모두 공통적으로 말하는 것은 PAL이 “uniform improvements”, “consistently outperforms”, “consistent and sizeable improvements”를 보였다는 점이다. 즉, 특정 언어쌍에서만 우연히 이긴 것이 아니라, bilingual, multilingual, cold-start 전반에서 heuristic baseline보다 안정적으로 좋았다는 것이 핵심 메시지다.

다만 업로드된 HTML이 일부 표 내부 수치를 완전히 노출하지는 않기 때문에, 각 언어별 정확한 F1 숫자와 relative cost reduction 값을 모두 복원할 수는 없었다. 다행히 Table 2 caption과 본문 설명만으로도, 200문장 budget, F1 기준 평가, random/uncertainty 대비 일관된 개선이라는 결론은 충분히 확인된다.

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 active learning을 **policy learning 문제** 로 재정의했다는 점이다. 기존 연구가 주로 heuristic을 설계했다면, 이 논문은 heuristic 자체를 학습 대상으로 바꿨다. 이는 문제 정의 차원의 기여다.

둘째, **cross-lingual transfer** 를 전면에 내세운 점이 강하다. low-resource language에서는 직접 policy를 학습할 데이터도 부족한데, 이 논문은 high-resource language에서 반복 시뮬레이션으로 policy를 익힌 뒤 다른 언어로 옮길 수 있게 만들었다. 이는 active learning과 transfer learning을 자연스럽게 결합한 설계다.  

셋째, **cold-start** 설정을 따로 다룬 점도 실용적이다. held-out set도 없고 online update도 불가능한 환경을 별도 시나리오로 놓고 정책과 모델을 함께 transfer하는 방식은, 실제 저자원 언어 annotation 환경을 더 잘 반영한다.  

### 한계

가장 큰 한계는 논문이 **stream-based selection** 을 기본 가정으로 둔다는 점이다. 저자도 각주에서 traditional batch setting으로 확장 가능하다고 언급하지만, 본문 중심 방법은 순차 도착 가정 하에서 더 자연스럽게 설계되어 있다. 따라서 일반적인 pool-based AL과 완전히 동일한 조건에서의 비교는 다소 제한적일 수 있다.  

둘째, reward가 held-out performance change에 의존하기 때문에, 원래 transfer 알고리즘은 **held-out evaluation data** 가 필요하다. 저자들도 이것이 low-resource setting에서는 비현실적일 수 있다고 인정하고, 그래서 별도로 cold-start variant를 제안했다. 즉, 기본 알고리즘 자체는 현실의 가장 극단적인 저자원 상황을 바로 만족하지 못한다.

셋째, 실험 도메인이 **cross-lingual NER** 에 집중되어 있다. 아이디어는 더 일반적이지만, 실제로 광범위하게 검증된 것은 NER과 언어 전이 시나리오다. 다른 NLP task나 비언어 도메인에서도 같은 정도의 우위가 유지되는지는 이 논문만으로는 알 수 없다. 이는 논문의 명시적 범위를 바탕으로 한 해석이다.

### 해석

비판적으로 보면, 이 논문의 진짜 공헌은 “RL을 써서 active learning을 했다”보다, **active learning policy도 transferable object가 될 수 있다**는 점을 보여준 데 있다. 즉, classifier 파라미터만 전이하는 것이 아니라, “어떤 데이터를 골라야 하는가”에 대한 전략도 학습·전이될 수 있다는 주장이다. 이 점은 이후 meta-active learning, learned acquisition literature로 이어지는 흐름과 맞닿아 있다.

## 6. Conclusion

이 논문은 active learning을 강화학습 문제로 재정의하고, deep Q-network로 데이터 선택 정책을 학습하는 **PAL** 을 제안한다. 정책은 문장 내용, 현재 supervised model의 예측 분포, confidence를 함께 관찰하며, 한 언어에서 시뮬레이션으로 학습된 뒤 다른 언어로 전이될 수 있다. 실험적으로는 CoNLL NER 데이터에서 bilingual, multilingual, cold-start 설정 전반에 걸쳐 random 및 uncertainty baseline보다 일관된 개선을 보였다고 보고한다.

실무적으로는 저자원 언어 annotation처럼 라벨 비용이 큰 환경에서, “고정 heuristic 대신 학습된 selection policy”라는 방향을 제시했다는 점이 중요하다. 연구적으로는 active learning heuristic을 더 이상 손으로 설계하는 규칙이 아니라, **학습되고 전이되는 정책** 으로 본다는 점에서 의미 있는 초기 작업이다. 다만 held-out reward 의존성과 stream-based 가정은 남는 제약이며, 그 한계를 완화하기 위한 cold-start 변형이 함께 제안되었다고 이해하는 것이 적절하다.
