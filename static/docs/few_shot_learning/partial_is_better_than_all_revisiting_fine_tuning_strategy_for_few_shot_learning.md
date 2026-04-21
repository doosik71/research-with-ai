# Partial Is Better Than All: Revisiting Fine-tuning Strategy for Few-shot Learning

- **저자**: Zhiqiang Shen, Zechun Liu, Jie Qin, Marios Savvides, Kwang-Ting Cheng
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2102.03983

## 1. 논문 개요

이 논문은 few-shot classification에서 널리 쓰이는 “base classes에서 충분히 학습한 뒤, novel classes에는 그 지식을 그대로 옮긴다”는 관행을 다시 점검한다. 기존 방법은 대체로 backbone feature extractor를 base 데이터로 학습한 뒤, novel 클래스에서는 backbone을 고정(freeze)하고 마지막 classifier만 학습하거나, meta-learning 방식으로 base에서 학습된 표현을 거의 그대로 사용한다. 저자들은 이런 전체 지식 이전(full transfer)이 항상 최선은 아니라고 주장한다.

핵심 문제는 base classes와 novel classes가 서로 겹치지 않는다는 점이다. 즉, base에서 유용했던 표현이 novel에서도 그대로 유효할 수는 있지만, 일부는 편향되었거나 심지어 novel 클래스 적응에 해로울 수 있다. few-shot 환경에서는 novel 데이터가 매우 적기 때문에, 반대로 모든 층을 전부 fine-tuning하면 쉽게 overfitting이 발생한다. 따라서 “전부 고정”과 “전부 업데이트” 사이에서 더 섬세한 절충이 필요하다는 것이 이 논문의 출발점이다.

이 문제는 중요하다. few-shot learning은 의료 영상처럼 데이터 수집이 비싸거나 희귀 클래스가 많은 문제에서 특히 필요하다. 이런 상황에서 소량의 novel data만으로도 좋은 일반화를 얻으려면, 기존 지식을 얼마나 유지하고 어느 정도만 적응시킬지 결정하는 transfer strategy 자체가 성능의 핵심이 된다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **Partial Transfer (P-Transfer)** 이다. 즉, base 모델의 모든 층을 동일하게 취급하지 않고, 어떤 층은 완전히 고정하고 어떤 층은 fine-tuning하며, fine-tuning되는 층들조차도 층마다 서로 다른 learning rate를 부여한다. 저자들은 이것이 “지식을 일부만 선택적으로 이전한다”는 의미에서 partial transfer라고 설명한다.

이 아이디어의 직관은 명확하다. 낮은 층의 특징은 일반적인 edge, texture 같은 보편적 표현일 가능성이 높고, 높은 층의 특징은 특정 데이터 분포나 클래스에 더 특화되어 있을 수 있다. novel 클래스가 base 클래스와 충분히 비슷하면 적은 수의 층만 조정해도 되지만, 차이가 커질수록 더 많은 층을 적응시켜야 할 수 있다. 따라서 backbone 전체를 일괄적으로 freeze하는 방식은 지나치게 경직되어 있고, 반대로 전체를 fine-tune하는 방식은 few-shot 환경에서 불안정하다.

기존 접근과의 차별점은 두 가지다. 첫째, meta-learning이든 non-meta baseline이든 상관없이 적용 가능한 일반적인 transfer strategy라는 점이다. 둘째, 어떤 층을 조정할지와 각 층의 learning rate를 사람이 수동으로 정하지 않고, **evolutionary search**로 자동 탐색한다는 점이다. 저자들은 fixed transfer는 자기들 search space의 특수한 경우에 불과하다고 본다. 즉, 모든 층의 learning rate를 0으로 두면 기존의 “전부 freeze”와 동일해진다.

## 3. 상세 방법 설명

문제 설정은 표준 few-shot classification이다. base classes 집합을 $L_b$, novel classes 집합을 $L_n$이라 하고, 두 집합은 서로 겹치지 않아 $L_b \cap L_n = \emptyset$ 이다. novel task는 일반적인 $N$-way $K$-shot 설정으로 정의된다. 즉, novel support set에는 $N$개 클래스 각각에 대해 $K$개의 라벨된 샘플이 있고, query set에는 같은 $N$개 클래스의 미지 샘플이 들어 있다.

논문의 목표는 층별 learning rate 벡터 $V_{lr} = [v_1, v_2, \dots, v_L]$ 를 찾아서, 해당 전략으로 fine-tuning했을 때 정확도가 최대가 되도록 하는 것이다. 이를 식으로 쓰면 다음과 같다.

$$
V^*_{lr} = \arg\max \text{Acc}(W, V_{lr})
$$

여기서 $W$는 네트워크 파라미터, $L$은 전체 층 수이다. 각 $v_l$은 해당 층의 fine-tuning learning rate를 뜻하며, $v_l = 0$이면 그 층을 freeze한다는 의미다. 따라서 이 식은 “어떤 층을 얼리고, 어떤 층을 얼마나 빠르게 업데이트할 것인가”를 jointly 최적화하는 문제로 볼 수 있다.

전체 프레임워크는 세 단계로 구성된다.

첫째, **base class pre-training** 단계다. non-meta 방법에서는 base dataset에서 일반적인 cross-entropy loss로 feature extractor를 학습한다. meta-learning 방법에서는 episode 기반으로 support/query를 구성하여, base episode에서의 $N$-way prediction loss를 최소화하도록 학습한다. 이 단계는 기존 few-shot literature와 크게 다르지 않다.

둘째, **evolutionary search** 단계다. 이 단계가 논문의 핵심이다. search 대상은 아키텍처 자체가 아니라, 각 층의 fine-tuning 여부와 learning rate다. 예를 들어 learning rate 후보 집합을 $\{0, 0.01, 0.1, 1.0\}$ 로 두면, 각 층은 이 중 하나를 고른다. Conv6처럼 6개 층이면 search space는 $4^6$개 전략이 된다. ResNet-12처럼 더 깊어지면 경우의 수는 훨씬 커진다.

저자들은 genetic algorithm 형태의 evolutionary search를 사용한다. 첫번째 여러 개의 전략 벡터를 무작위로 초기화한다. 각 전략 벡터는 층별 learning rate assignment를 담고 있다. 그런 다음 각 전략에 대해 다음 과정을 수행한다. base에서 학습된 feature extractor를 불러오고, 전략에서 $v_l = 0$인 층은 gradient를 막아 freeze하며, 나머지 층은 해당 learning rate로 mini fine-tuning을 수행한다. 그 뒤 validation set에서 정확도를 측정한다. 성능이 좋은 상위 전략들을 부모로 선택하고, mutation과 crossover를 통해 새로운 전략들을 생성한다. 이 과정을 반복해 최종적으로 validation accuracy가 가장 높은 전략을 고른다.

논문에 제시된 Algorithm 1의 요지는 다음과 같다. `miniEval(v)`는 주어진 전략 $v$를 실제로 적용해 support set으로 짧게 fine-tuning하고 validation 정확도를 반환하는 함수다. 초기에는 random sampling으로 여러 전략을 평가하고, 이후 세대마다 mutation과 crossover를 통해 offspring 전략을 만들고 다시 평가한다. 매 세대마다 top-$K$ 전략만 유지하고, 마지막에 최고 성능 전략 $v^*$를 반환한다.

셋째, **searched strategy를 이용한 partial transfer** 단계다. search에서 찾은 최적 전략을 novel support set에 적용하여 일부 층만 fine-tuning하고, 나머지는 freeze한 뒤 query set에서 평가한다. 이 과정은 meta-learning에도, baseline++ 같은 non-meta 방법에도 모두 적용된다.

논문은 이 방법을 Baseline++와 ProtoNet에 구체적으로 결합한다. Baseline++에서는 cosine similarity 기반 classifier를 사용하면서 backbone $f_\theta(x)$를 층별 learning rate에 따라 end-to-end로 조정한다. 기존 baseline++는 backbone을 고정하고 weight vector 쪽만 조정하는 경향이 있는데, P-Transfer는 backbone 일부까지 함께 적응시켜 더 조화로운 representation을 얻으려는 것이다. ProtoNet 같은 meta method에서는 episode 기반 support/query 구조는 유지하되, search 단계에서 backbone을 부분적으로 fine-tuning하는 전략을 찾는다. 즉, metric-based meta-learning 위에 transfer strategy optimization을 얹는 구조다.

이 논문에는 별도의 새로운 손실 함수가 제안되지는 않는다. 오히려 기여는 loss design보다 **transfer configuration search**에 있다. non-meta에서는 standard cross-entropy 기반 training을, meta에서는 episode-based prediction loss를 그대로 사용하면서, fine-tuning 과정의 자유도를 층별 learning rate로 확장한 것이다.

## 4. 실험 및 결과

실험은 주로 `mini-ImageNet`과 `CUB200-2011`에서 수행된다. `mini-ImageNet`은 100개 클래스, 클래스당 600장 이미지로 구성되며, 64개 base, 16개 validation, 20개 novel 클래스로 나뉜다. `CUB200-2011`은 새(bird) 이미지의 fine-grained dataset으로, 100개 base, 50개 validation, 50개 novel 클래스로 분할한다. 논문은 일반 few-shot classification뿐 아니라, `mini-ImageNet`에서 학습하고 CUB로 옮기는 cross-domain setting도 실험한다.

구현 측면에서 meta methods는 5-way episode를 사용하고, 각 클래스에서 support는 $k$개, query는 15개 샘플을 사용한다. 1-shot에서는 60,000 episodes, 5-shot에서는 40,000 episodes를 학습했다. search 단계에서는 validation dataset에서 20 episodes를 샘플링하고, support set으로 100 iterations fine-tuning한 뒤 query 성능을 본다. 최종 평가는 600 episodes 평균 정확도와 95% confidence interval로 보고한다. non-meta 방법은 base dataset에서 400 epochs 학습하고, fine-tuning 시 fully connected layer에는 SGD learning rate 0.01을 사용하며 backbone 층들에는 searched learning rate를 사용한다. evolutionary algorithm의 하이퍼파라미터는 population size $P=20$, max iterations $I=20$, 그리고 random sampling, mutation, crossover 횟수를 각각 50으로 두었다.

먼저 ablation study에서 fixed strategy, 사람이 설계한 manual strategy, searched strategy를 비교한다. Conv6 backbone을 쓴 non-meta 실험에서 CUB 5-shot은 Fixed 79.48, Manual 79.26, Searched 80.48로 searched가 가장 좋다. `mini-ImageNet`에서도 1-shot은 Fixed 52.52, Manual 53.22, Searched 53.55로 향상된다. 다만 CUB 1-shot에서는 Fixed 66.75가 Searched 65.82보다 약간 높다. 즉, searched strategy가 전반적으로 우세하지만 모든 셀에서 항상 절대 우세한 것은 아니다. 이 점은 논문을 읽을 때 과장 없이 해석해야 한다.

ResNet-12 실험에서는 개선 폭이 더 분명하다. Baseline++ 기준으로 CUB에서 1-shot은 70.72에서 73.88로, 5-shot은 85.59에서 87.81로 향상된다. `mini-ImageNet`에서는 1-shot이 59.35에서 64.21로, 5-shot이 77.51에서 80.38로 크게 오른다. ProtoNet에서도 개선이 나타나는데, `mini-ImageNet` 5-shot은 74.87에서 76.59로, CUB 5-shot은 87.28에서 88.32로 상승한다. 다만 CUB 1-shot에서는 ProtoNet Fixed 73.82가 Searched 73.16보다 약간 높다. 따라서 저자들의 주장은 “대부분의 경우 더 낫다”이지 “모든 경우 완전히 지배한다”는 것은 아니다.

논문은 “모든 층을 fine-tune”하는 단순 전략이 왜 좋지 않은지도 보여준다. `mini-ImageNet`에서 모든 층을 fine-tuning하면 1-shot 40.84%, 5-shot 50.95%에 그쳐 매우 낮은 성능을 보였다고 보고한다. 이는 few-shot에서 full fine-tuning이 overfitting과 불안정성 문제를 가진다는 저자들의 문제의식을 뒷받침한다.

cross-domain transfer 결과도 흥미롭다. `mini-ImageNet`에서 학습한 뒤 CUB로 옮길 때, 일반적으로 same-domain보다 더 많은 층을 fine-tune해야 한다는 패턴이 Figure 4에 시각화되어 있다. 또한 작은 batch size 문제로 BatchNorm이 불리할 수 있기 때문에 GroupNorm도 시험한다. 예를 들어 Baseline++ + ResNet-12의 5-shot cross-domain에서는 BatchNorm 기준 Fixed 63.25, Searched 68.30인데, GroupNorm을 쓰면 Fixed 64.02, Searched 74.22로 더 크게 향상된다. 이는 domain gap이 커질수록 표현 적응과 normalization 선택이 더 중요해진다는 점을 시사한다.

최종 state-of-the-art 비교에서 `mini-ImageNet`의 P-Transfer는 ResNet-12 backbone으로 1-shot 64.21 ± 0.77, 5-shot 80.38 ± 0.59를 기록한다. 이는 같은 표에 있는 Meta-Baseline의 63.17 / 79.26보다 높고, DropBlock이나 label smoothing 같은 추가 훈련 기법을 사용한 MetaOptNet†와 비교해도 경쟁력 있다. 다만 confidence interval이 겹칠 가능성이나 세부 실험 설정 차이는 표만으로 완전히 판단하기 어렵다.

논문은 전통적인 transfer learning으로도 확장한다. ImageNet에서 사전학습한 Inception V3를 CUB로 옮길 때, 모든 가중치를 단순 상속하는 baseline보다 일부 층을 재초기화하고 fine-tune하는 partial transfer가 더 좋았다고 보고한다. 표 6에서 top-1 accuracy는 baseline 82.9%, partial transfer 83.8%다. 개선 폭은 크지 않지만, few-shot 밖의 일반 transfer learning에도 아이디어가 확장될 수 있음을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 few-shot learning의 성능 향상을 “새로운 metric”이나 “새로운 meta-optimizer”가 아니라 **transfer strategy 자체의 재설계**로 풀었다는 점이다. 많은 연구가 classifier나 distance function에 집중하는 반면, 이 논문은 backbone을 어떻게 옮길지가 사실상 더 근본적인 문제일 수 있다고 본다. 그리고 이 문제를 freeze-or-finetune의 이분법이 아니라, 층별 learning rate를 포함한 세밀한 정책으로 정식화했다는 점이 설득력 있다.

또 다른 강점은 범용성이다. Baseline++ 같은 non-meta 방법과 ProtoNet 같은 meta-learning 방법 모두에 적용 가능하며, 실제로 둘 다에서 개선을 보였다. 즉, 특정 알고리즘에 종속된 테크닉이 아니라 “base에서 novel로 넘어갈 때 backbone을 어떻게 다룰 것인가”라는 보다 일반적인 설계 원리로 제시된다.

실험적으로도 강점이 있다. same-domain, cross-domain, non-meta, meta, 그리고 전통적 transfer learning까지 다양한 설정을 다룬다. 또한 Figure 4를 통해 domain gap이 커질수록 더 많은 층을 fine-tune해야 한다는 구조적 패턴을 관찰한 점은 단순 성능 보고보다 더 의미 있다. 검색된 전략이 임의적인 블랙박스가 아니라 어느 정도 해석 가능한 패턴을 가진다는 뜻이기 때문이다.

반면 한계도 분명하다. 첫째, evolutionary search는 효율적이라고 주장되지만, 여전히 추가 계산 비용이 필요하다. 논문은 Conv6는 V100 한 장으로 약 6시간, ResNet-12는 약 하루라고 말한다. few-shot 벤치마크 연구에서는 감당 가능한 수준일 수 있으나, 실제 대규모 환경이나 더 깊은 backbone에서는 부담이 커질 가능성이 있다.

둘째, search space가 learning rate zoo와 layer granularity에 의존한다. 예를 들어 $\{0, 0.01, 0.1, 1.0\}$ 같은 learning rate 후보 집합이 최적인지, 또는 block 단위보다 더 세밀한 parameter group 설계가 필요한지는 논문이 깊게 다루지 않는다. 즉, “partial transfer가 좋다”는 메시지는 강하지만, 구체적 탐색 공간 설계의 일반성은 아직 제한적이다.

셋째, 결과가 모든 경우에서 일관되게 우세한 것은 아니다. 예를 들어 CUB 1-shot에서는 Fixed가 Searched보다 약간 좋은 경우가 있다. 따라서 “partial is always better than all”이라고 문자 그대로 받아들이기보다는, 대체로 더 좋은 방향이며 특히 deeper network나 domain shift가 있을 때 더 효과적이라고 해석하는 편이 정확하다.

넷째, 왜 특정 층 조합이 선택되는지에 대한 분석은 아직 제한적이다. Figure 4는 유의미한 패턴을 보여주지만, 각 층의 semantic role이나 gradient stability, representation drift 같은 관점에서 정교한 해석까지는 제공하지 않는다. 저자들 스스로도 future work에서 더 많은 분석을 하겠다고 밝힌다.

다섯째, 논문은 윤리적 영향에서 의료영상 등 고위험 응용을 언급하지만, few-shot 시스템 실패 시 어떤 안전장치가 필요한지까지는 구체적으로 다루지 않는다. 이는 방법론 논문으로서는 자연스럽지만, 실제 적용에서는 중요한 미해결 문제다.

## 6. 결론

이 논문은 few-shot learning에서 base-to-novel transfer를 “전부 고정” 또는 “전부 fine-tuning”으로 다루는 기존 관행을 비판하고, 층별로 freeze 여부와 learning rate를 달리하는 **P-Transfer**를 제안했다. 핵심 기여는 partial transfer를 위한 layer-wise search space를 설계하고, evolutionary search로 최적 fine-tuning configuration을 자동으로 찾는 방법을 제시한 데 있다.

실험 결과를 보면, 이 방법은 Baseline++와 ProtoNet 모두에서 대체로 성능을 끌어올렸고, 특히 ResNet-12와 `mini-ImageNet` 같은 설정에서 꽤 큰 개선을 보였다. 또한 cross-domain 상황에서는 더 많은 층을 적응시켜야 한다는 직관을 실험적으로 확인했고, 전통적인 pre-training + fine-tuning에도 확장 가능성을 보여주었다.

실제 적용 측면에서 이 연구는 “few-shot 문제는 단지 좋은 embedding이나 metric만의 문제가 아니라, **무엇을 얼마나 옮길지**의 문제이기도 하다”는 점을 분명히 한다. 향후에는 detection, segmentation 같은 다른 few-shot task로의 확장, 더 효율적인 search, 그리고 partial transfer가 왜 효과적인지에 대한 representation-level 분석이 중요한 후속 연구가 될 가능성이 크다.
