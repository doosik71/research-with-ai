# Active learning for data streams: a survey

Davide Cacciarelli와 Murat Kulahci의 이 논문은 **data stream 환경에서의 online active learning** 연구를 체계적으로 정리한 survey다. 논문의 문제의식은 분명하다. 실제 산업·서비스 환경에서는 데이터가 연속적으로 유입되지만, 각 샘플의 라벨을 모두 즉시 확보하기는 비용과 시간 측면에서 어렵다. 따라서 모델은 모든 데이터를 저장해 두고 나중에 고르는 방식이 아니라, **흘러오는 데이터 중 어떤 샘플을 지금 라벨링할지 실시간으로 판단**해야 한다. 저자들은 이런 문제를 다루는 방법들을 정리하고, query strategy, query timing, model update, scalability, evaluation이라는 다섯 가지 핵심 질문을 중심으로 분야 전체를 재구성한다.  

## 1. Paper Overview

이 논문이 다루는 핵심 연구 문제는 “스트리밍 데이터 환경에서 제한된 라벨링 예산으로 어떻게 가장 유익한 샘플만 선택해 학습 성능을 높일 것인가”이다. 기존 active learning survey들은 주로 **pool-based active learning**, 즉 정적인 unlabeled pool에서 샘플을 고르는 설정에 집중해 왔다. 하지만 현실의 많은 문제는 데이터가 순차적으로 도착하고, 저장 공간이나 의사결정 지연 허용 범위가 제한적이기 때문에 이 설정이 잘 맞지 않는다. 저자들은 이러한 간극을 지적하며, 특히 Lughofer(2017) 이후 등장한 다양한 online/stream-based active learning 방법을 포괄적으로 다시 정리할 필요가 있다고 주장한다. 또한 이 논문은 단순 문헌 나열이 아니라, 온라인 active learning을 평가하기 위해 무엇을 봐야 하는지, 실제 적용 시 어떤 제약이 중요한지, 앞으로 어떤 방향이 유망한지를 함께 제시한다.

## 2. Core Idea

이 논문의 중심 아이디어는 새로운 알고리즘을 제안하는 것이 아니라, **online active learning 분야를 공통 구조와 설계 축으로 재해석하는 것**이다. 저자들은 먼저 active learning의 기본 인스턴스 선택 원리를 uncertainty, expected error/variance minimization, expected model change, disagreement, diversity/density, hybrid strategy로 나눈다. 그런 다음 online setting에서는 단순히 “무엇을 고를 것인가” 외에도, **언제 고를 것인가**, **개념 변화(concept drift)에 어떻게 적응할 것인가**, **모델을 증분 학습할지 재학습할지**, **라벨 지연(verification latency)을 어떻게 처리할지**가 함께 설계되어야 함을 강조한다.

가장 중요한 기여는 taxonomy다. 저자들은 online active learning을 네 부류로 정리한다.

1. stationary data stream classification
2. drifting data stream classification
3. evolving fuzzy system approaches
4. experimental design and bandit approaches

이 분류는 단순 주제별 정리가 아니라, **데이터 분포가 안정적인지 변화하는지**, **문제가 분류인지 회귀인지**, **모델이 single model인지 ensemble인지**, **single-pass인지 batch/window-based인지**를 모두 함께 고려하는 구조라는 점에서 유용하다. 논문 9페이지의 Figure 5도 이를 잘 보여 주는데, online active learning은 대체로 data stream, instance evaluation, instance selection, model update, drift detection, drift adaptation이 연결된 공통 프레임워크 위에서 이해할 수 있다고 설명한다.

## 3. Detailed Method Explanation

### 3.1 Active learning의 기본 선택 기준

논문은 먼저 active learning이 어떤 샘플을 라벨링할지 결정하는 일반적 원리를 정리한다.

* **Uncertainty-based query strategies**: 현재 모델이 가장 확신하지 못하는 샘플을 고른다.
* **Expected error/variance minimization**: 새 라벨이 들어왔을 때 미래 오차나 분산이 얼마나 줄어드는지를 직접 본다.
* **Expected model change maximization**: 샘플 하나가 현재 파라미터를 얼마나 많이 바꿀지를 기준으로 삼는다.
* **Disagreement-based strategies**: 여러 모델 또는 전문가 집단이 서로 다른 예측을 내는 샘플을 선택한다.
* **Diversity / density-based strategies**: 현재까지 본 labeled set과 멀리 떨어져 있거나, 데이터 구조상 대표성이 큰 샘플을 고른다.
* **Hybrid strategies**: 위 기준들을 조합한다.

이 정리는 후반부 taxonomy의 토대가 된다. 즉 online active learning의 다양한 방법들은 결국 이들 원리를 스트리밍 환경 제약에 맞춰 구현한 것으로 볼 수 있다.

### 3.2 Active learning 시나리오와 online setting의 특수성

논문은 active learning 시나리오를 membership query synthesis, pool-based, online active learning으로 나눈다. 여기서 online setting의 가장 큰 차이는 **정적인 pool이 없고, 각 샘플을 본 직후 바로 querying 여부를 결정해야 할 수 있다**는 점이다. 저자들은 이를 secretary problem과 유사한 상황으로 설명한다.

또한 online active learning 내부에서도 두 가지 처리 방식이 나온다.

* **single-pass**: 샘플이 도착하자마자 즉시 평가하고 버리거나 라벨 요청
* **window-based / batch-based**: 일정 크기의 buffer나 window를 모아 그 안에서 top-$k$를 선택

논문 5~6페이지의 Figure 3, Figure 4는 이 차이를 시각적으로 보여 준다. single-pass는 latency가 매우 작은 환경에 적합하고, window-based는 다소 시간이 허용되는 대신 더 정교한 비교가 가능하다.

### 3.3 Online setting에서 추가로 중요해지는 요소

논문이 survey임에도 꽤 설계론적으로 좋은 이유는, online active learning을 단순 query rule 문제가 아니라 **시스템 설계 문제**로 본다는 점이다. 저자들이 특히 강조하는 요소는 다음과 같다.

첫째, **stationary vs drifting stream**이다. 데이터 분포가 변하지 않는다면 기존 uncertainty-based 정책이 잘 작동할 수 있지만, concept drift가 발생하면 과거 기준으로 유익하던 샘플이 더 이상 중요하지 않을 수 있다.

둘째, **verification latency**다. 라벨 요청 후 오라클이 즉시 답하지 않으면, 아직 라벨이 돌아오지 않은 샘플과 유사한 샘플을 중복 질의할 위험이 있다. 논문은 null, intermediate, extreme latency로 이를 구분한다.

셋째, **incremental training vs complete retraining**이다. 새 라벨이 들어올 때마다 모델을 미세 조정할지, 전체 labeled set으로 다시 학습할지에 따라 시스템 비용과 적응성이 크게 달라진다.

### 3.4 온라인 active learning 방법들의 taxonomy와 대표 예시

#### 3.4.1 Stationary classification

이 범주에서는 margin이나 confidence 기반 thresholding이 대표적이다. 예를 들어 selective sampling perceptron은 현재 margin $p_t^b = w_{t-1}^\top x_t$를 계산하고, 다음 확률로 라벨 요청 여부를 정한다.

$$
P_t = \frac{b}{b + |p_t^b|}
$$

margin이 0에 가까울수록 샘플이 decision boundary 근처에 있으므로, 라벨을 요청할 확률이 높아진다. 이 방식은 uncertainty sampling을 확률적 querying으로 구현한 전형적인 예다. 논문은 이 알고리즘의 절차를 Algorithm 1로 제시한다.

또한 passive-aggressive variant는 perceptron보다 더 공격적으로 파라미터를 갱신하며, 최근 연구들은 early-stage model uncertainty, class imbalance, differential privacy, reject option classifier 등도 함께 고려한다.

#### 3.4.2 Drifting classification

drifting stream에서는 covariate shift, real concept drift, label distribution shift를 구분해야 하며, drift의 양상도 abrupt, gradual, incremental, recurring concept로 나뉜다. 논문은 이것이 sampling policy뿐 아니라 model replacement, labeling-rate scheduling까지 바꿔야 하는 문제라고 본다.

이 범주의 핵심은 **drift detector와 active learning의 결합**이다. 예를 들어 ADWIN 같은 drift detector를 사용해 drift warning이나 drift detected 상태를 판단하고, 그 시점에 labeling rate를 늘리거나 새로운 classifier를 병행 학습한 뒤 교체하는 식의 일반 프레임워크가 Algorithm 3으로 제시된다. 이는 “drift가 의심될수록 더 공격적으로 라벨을 모아야 한다”는 매우 실용적인 통찰을 반영한다.

또한 verification latency를 다루는 forgetting/simulating 전략, class imbalance를 고려한 class-specific threshold, stable classifier와 dynamic classifier를 결합한 paired/ensemble framework, clustering 기반 sampling 등이 상세히 소개된다. 저자들은 gradual drift를 처리하려면 단순 최신 window만 보는 reactive model보다, **여러 시간 스케일을 함께 보는 ensemble 구조**가 더 적합하다고 설명한다.

#### 3.4.3 Evolving fuzzy systems

이 범주는 online active learning을 fuzzy rule-based adaptive system과 결합한다. 논문은 EFS가 rule generation, merging, pruning, parameter updating을 통해 새 데이터에 맞춰 구조와 파라미터를 함께 바꿀 수 있다고 설명한다. 23페이지 Figure 11은 structure evolving과 parameters updating이 핵심 축이라는 점을 보여 준다.

이 계열의 방법은 단순 uncertainty뿐 아니라 **conflict**, **ignorance**, **novelty content**, **overlap degree**, **parameter uncertainty** 등을 함께 고려한다. 즉 “현재 분류가 애매한가?”뿐 아니라 “이 샘플이 아직 학습되지 않은 공간에 있나?”, “새 규칙이 필요할 정도로 새로운 패턴인가?”를 본다. 이는 stream 환경의 novelty 대응 측면에서 일반 분류기 기반 접근과 구별되는 장점이다.

#### 3.4.4 Experimental design and bandits

이 부분은 survey 가운데서도 비교적 독특하다. 저자들은 online active learning을 실험계획법과 bandit의 관점으로도 읽을 수 있다고 본다. 예를 들어 선형 회귀 설정에서 A-, D-, E-optimality, G-optimality는 **어떤 샘플을 선택하면 파라미터 추정 또는 예측분산 측면에서 가장 이득이 큰가**를 수학적으로 정당화해 준다.

특히 stream-based linear regression에서는 thresholding 방식으로 $x_t^\top (X^\top X)^{-1} x_t$ 같은 값을 계산해, 현재 설계에서 정보량이 큰 샘플만 선택하는 방식이 소개된다. 이는 uncertainty sampling과 닮았지만, 훨씬 더 **설계-최적화적 해석**을 갖는다. 또한 multi-armed bandit과 reinforcement learning 관점에서는 exploration과 exploitation의 trade-off를 통해 어떤 샘플을 질의해야 장기적 보상이 큰지 논의한다. 이 부분은 active learning을 단순 supervised labeling 문제보다 더 넓은 sequential decision-making 틀로 연결한다는 점에서 의미가 있다.

## 4. Experiments and Findings

이 논문은 survey이므로 새로운 실험 결과를 제시하기보다, 기존 연구들을 **평가 방식과 적용 영역 기준으로 비교·정리**한다는 점이 중요하다. 저자들은 active learning 평가에서 learning curve와 통계 검정의 필요성을 강조하며, 특히 online stream 환경에서는 고정 holdout test set보다 **prequential evaluation(test-then-train)**이 concept drift를 더 잘 반영할 수 있다고 설명한다. 즉 각 시점마다 먼저 예측하고, 그다음 샘플을 학습에 반영하는 평가 방식이 streaming 현실에 더 가깝다는 것이다.

또한 Table 1에서는 어떤 연구가 holdout 기반 평가를 썼고, 어떤 연구가 prequential 방식을 썼는지를 정리한다. 저자들의 해석은 명확하다. **drifting stream을 다루는 방법은 대체로 prequential evaluation이 더 적합**하고, concept drift를 직접 다루지 않는 방법은 holdout test set을 더 자주 사용한다. 즉 평가 프로토콜 자체가 방법의 문제 설정과 맞아야 한다는 점을 강조한다.

Table 2는 survey의 또 다른 핵심 산출물이다. 여기서는 online active learning 전략을 data processing(single-pass vs batch), data stream 가정(stationary, drifting, evolving), task(classification, regression, object detection), model(single model, ensemble) 기준으로 요약한다. 이 표 덕분에 독자는 개별 논문을 하나하나 읽지 않고도, 어떤 영역이 많이 연구되었고 어떤 조합이 상대적으로 덜 탐색되었는지 빠르게 파악할 수 있다. 33페이지 표와 결론 바로 앞 부분은 이 survey의 “지도(map)” 역할을 한다.

응용 측면에서는 spam filtering, network traffic protocol identification, computer vision, object detection, autonomous driving, human activity recognition, industrial process monitoring, compressor optimization 등이 예시로 제시된다. 이 중 특히 computer vision에서는 deep active learning의 강력한 방법들이 pairwise similarity나 clustering에 의존하는 경우가 많아 single-pass stream setting으로 옮기기 어렵다는 지적이 나온다. 저자들은 그래서 실제 streaming vision 문제에서는 여전히 비교적 단순한 uncertainty-based streaming classifier가 많이 쓰인다고 분석한다.  

## 5. Strengths, Limitations, and Interpretation

이 논문의 강점은 세 가지다.

첫째, **문헌 정리를 단순 시대순 나열이 아니라 설계 축 중심으로 재구성**했다는 점이다. uncertainty, drift, latency, update regime, evaluation 같은 요소를 함께 봐야 한다는 프레임은 실제 연구자와 실무자 모두에게 유용하다.

둘째, **online active learning을 넓게 정의**한다. 단순 분류기 기반 sampling뿐 아니라 fuzzy system, experimental design, bandit, reinforcement learning까지 포함함으로써, 이 분야가 단지 “uncertainty sampling의 스트리밍 버전”이 아니라는 점을 보여 준다.

셋째, **평가와 응용을 함께 다룬다**. survey 논문이 종종 알고리즘 분류에만 머무는 반면, 이 논문은 evaluation protocol, real-world applications, practical challenges까지 포함해 실제 적용 관점의 그림을 준다.

한계도 있다.

첫째, survey라는 성격상 개별 알고리즘의 수학적 정당화나 empirical superiority를 깊게 비교하지는 않는다. 즉 “어떤 방법이 항상 더 좋다” 같은 결론은 의도적으로 피한다. 이것은 장점이자 한계다.

둘째, taxonomy가 매우 포괄적이지만, 그만큼 서로 다른 범주의 방법들 사이에서 **직접적인 apples-to-apples 비교**는 어렵다. 예를 들어 drifting classification의 ensemble 방식과 experimental design 기반 regression 방법은 문제 설정 자체가 다르므로, 하나의 통합 성능 기준으로 보기는 힘들다.

셋째, deep learning 기반 streaming active learning은 중요하지만 아직 성숙하지 않았고, 논문도 이 부분에서 “유망하지만 제약이 많다”는 수준의 정리에 머문다. 즉 현대 대규모 foundation model 시대의 관점에서 보면, 이 영역은 survey 이후 추가 정리가 더 필요해 보인다. 이 점은 논문 자체도 future direction의 형태로 암시한다.  

비판적으로 해석하면, 이 논문이 가장 설득력 있게 보여 주는 메시지는 다음이다. **online active learning의 본질은 불확실성만 보는 것이 아니라, 시간성과 예산성을 함께 다루는 적응적 의사결정 문제**라는 것이다. 샘플 선택, drift 대응, 라벨 지연 처리, 모델 갱신 주기, 평가 방법은 따로따로가 아니라 하나의 통합 시스템으로 설계되어야 한다. 이 점에서 이 survey는 단순한 문헌 요약을 넘어 분야의 설계 원칙을 정리한 논문으로 읽을 수 있다.

## 6. Conclusion

이 논문은 data stream 환경의 online active learning을 폭넓고 체계적으로 정리한 survey다. 핵심 기여는 세 가지다. 첫째, pool-based active learning 중심의 기존 survey가 놓친 **stream-based / online setting** 을 본격적으로 정리했다. 둘째, 방법론을 stationary, drifting, evolving fuzzy, experimental design and bandits라는 네 범주로 묶고, single-pass와 batch 처리, single model과 ensemble, classification과 regression의 차이까지 함께 보여 주었다. 셋째, evaluation strategy, application, challenge, future direction까지 포함해 연구와 실무를 연결했다.

논문 결론에서 저자들은 online active learning이 빠르게 성장 중인 분야이며, 특히 의료, 자율주행, 산업 생산처럼 데이터는 많지만 라벨은 비싼 영역에서 중요성이 더 커질 것이라고 본다. 또한 uncertainty, diversity, query by committee, reinforcement learning 등 다양한 전략이 이미 시도되었지만, 앞으로는 **model-agnostic** 이고 **single-pass** 에 강한 방법이 특히 중요할 것이라고 전망한다. 이 메시지는 survey 전체를 관통하는 결론이기도 하다.  
