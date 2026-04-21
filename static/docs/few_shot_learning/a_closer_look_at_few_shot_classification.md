# A Closer Look at Few-shot Classification

- **저자**: Wei-Yu Chen, Yen-Cheng Liu, Zsolt Kira, Yu-Chiang Frank Wang, Jia-Bin Huang
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1904.04232

## 1. 논문 개요

이 논문은 few-shot classification 분야에서 널리 쓰이는 여러 방법들을 같은 조건에서 다시 비교해 보면 무엇이 실제로 중요한지 달라 보인다는 문제의식에서 출발한다. 저자들은 기존 문헌에서 성능 향상으로 보고된 많은 결과가 알고리즘 자체의 우월성 때문인지, 아니면 backbone 깊이, data augmentation, optimizer, meta-training 설정 같은 구현 차이 때문인지 명확하지 않다고 지적한다.

연구 문제는 두 가지로 요약된다. 첫째, 대표적인 few-shot 방법들을 공정한 실험 조건에서 비교했을 때 실제 성능 차이가 얼마나 남는가이다. 둘째, 기존 평가가 base class와 novel class를 같은 데이터셋 안에서 나누는 방식에 치우쳐 있어 현실적인 domain shift를 반영하지 못하는데, 이런 더 어려운 환경에서도 meta-learning 방법들이 여전히 강한가이다.

이 문제는 중요하다. few-shot learning은 라벨이 매우 적은 새로운 클래스를 빠르게 인식해야 하는 실제 응용과 직접 연결되기 때문이다. 예를 들어 희귀종 분류처럼 데이터가 부족한 상황에서는, “적은 샘플로 일반화” 자체뿐 아니라 “훈련 때 보지 못한 다른 도메인으로도 적응”하는 능력이 중요하다. 논문은 바로 이 지점에서 기존 few-shot 평가가 다소 낙관적일 수 있음을 보여준다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 새롭고 복잡한 few-shot 알고리즘을 제안하는 것이 아니라, 기존 방법들을 더 엄밀하게 다시 비교하고, 단순한 baseline이 실제로 얼마나 강한지 드러내는 데 있다. 특히 저자들은 표준 transfer learning 방식의 baseline에 distance-based classifier를 붙인 `Baseline++`가, 놀랍게도 당시 state-of-the-art meta-learning 방법들과 비슷하거나 더 나은 성능을 내는 경우가 많다고 보인다.

논문이 강조하는 직관은 다음과 같다. shallow backbone에서는 feature의 intra-class variation, 즉 같은 클래스 안에서도 feature가 많이 퍼지는 문제가 크다. 이때 cosine similarity 기반 분류기처럼 class prototype 방향으로 feature를 더 정렬시키는 설계가 도움이 된다. 하지만 backbone이 깊어지면 feature 자체가 더 잘 정리되므로, 알고리즘 간 차이가 줄어들 수 있다.

기존 접근과의 차별점은 “더 복잡한 meta-learning이 항상 더 낫다”는 암묵적 전제를 실험적으로 재검토했다는 점이다. 논문은 특히 다음 두 가지를 강하게 주장한다. 하나는 성능 향상의 일부가 알고리즘 혁신보다 backbone과 구현 디테일에서 올 수 있다는 점이고, 다른 하나는 cross-domain 상황에서는 오히려 단순한 baseline fine-tuning이 더 유리할 수 있다는 점이다.

## 3. 상세 방법 설명

논문은 먼저 가장 단순한 `Baseline`을 정의한다. 이 방법은 전형적인 pre-training + fine-tuning 절차를 따른다. base class 데이터 $X_b$가 충분히 있을 때 feature extractor $f_\theta$와 classifier $C(\cdot \mid W_b)$를 cross-entropy loss로 학습한다. 여기서 $W_b \in \mathbb{R}^{d \times c}$는 분류기 가중치이고, $d$는 feature 차원, $c$는 base class 수이다. 분류기는 기본적으로 선형층 $W_b^\top f_\theta(x_i)$ 뒤에 softmax를 붙인 형태다.

즉, training stage에서는 base class에 대해 일반적인 supervised classification을 한다. 그 다음 fine-tuning stage에서는 feature extractor의 파라미터 $\theta$를 고정하고, novel class의 소수 샘플 $X_n$만 사용해 새 classifier $C(\cdot \mid W_n)$를 학습한다. 핵심은 backbone은 그대로 두고 마지막 classifier만 새 클래스에 맞게 다시 학습한다는 점이다.

`Baseline++`는 이 구조를 거의 그대로 유지하되 classifier를 바꾼다. 선형 분류기 대신 각 클래스의 weight vector와 입력 feature 사이의 cosine similarity를 사용한다. 클래스 $j$의 weight를 $w_j$라 하면, 입력 $x_i$에 대한 score는 다음과 같다.

$$
s_{i,j} = \frac{f_\theta(x_i)^\top w_j}{\|f_\theta(x_i)\| \, \|w_j\|}
$$

이 score들을 softmax로 정규화해 class probability를 만든다. 논문은 이 방식이 각 클래스 weight vector를 prototype처럼 작동하게 하며, 결과적으로 같은 클래스 feature들이 더 비슷한 방향으로 모이도록 유도한다고 설명한다. 즉, intra-class variation을 줄이는 효과를 기대할 수 있다. 저자들은 이것이 자신들의 독창적 발명은 아니며, 기존 metric learning 및 few-shot literature에 이미 있던 아이디어라고 분명히 밝힌다.

meta-learning 방법들의 공통 구조도 정리한다. 이들은 support set $S$에 조건부인 분류기 $M(\cdot \mid S)$를 학습한다. meta-training에서는 base class 중 임의의 $N$개 클래스를 뽑고, 각 클래스에서 few-shot support set $S_b$와 query set $Q_b$를 샘플링해 하나의 episode를 만든다. 그리고 query set에서의 $N$-way classification loss $L_{N\text{-}way}$를 최소화하도록 학습한다. meta-testing에서는 novel class support set $S_n$이 주어졌을 때 그 support set에 조건부인 분류기를 만들어 novel query를 분류한다.

각 방법의 차이는 support set을 어떻게 사용하는지에 있다. MatchingNet은 query feature와 각 support feature의 cosine similarity를 활용한다. ProtoNet은 support feature들의 class mean, 즉 prototype을 만들고 query와의 Euclidean distance를 본다. RelationNet은 고정된 거리 함수 대신 learnable relation module을 둔다. MAML은 support set으로 몇 번 gradient update를 해서 task-specific 파라미터로 적응하고, 그 적응 후 query loss가 좋은 초기 파라미터를 학습한다.

실험 절차도 비교를 위해 통일했다. Baseline과 Baseline++는 training에서 400 epoch, batch size 16으로 학습한다. meta-learning 계열은 1-shot에서 60,000 episodes, 5-shot에서 40,000 episodes를 사용한다. 모든 방법은 Adam optimizer와 초기 learning rate $10^{-3}$를 쓴다. random crop, horizontal flip, color jitter 같은 표준 data augmentation도 적용한다. 이 통일 자체가 논문 메시지의 중요한 일부다. 왜냐하면 저자들이 보기에 기존 성능 차이 중 일부는 이 설정 불일치에서 비롯되기 때문이다.

## 4. 실험 및 결과

논문은 세 가지 시나리오를 다룬다. 첫째는 generic object recognition으로 `mini-ImageNet`이다. 이 데이터셋은 100개 클래스, 클래스당 600장 이미지로 구성되며, 64 base, 16 validation, 20 novel class 분할을 따른다. 둘째는 fine-grained classification인 `CUB-200-2011`이다. 여기서는 100 base, 50 validation, 50 novel class로 나눈다. 셋째는 cross-domain setting으로 `mini-ImageNet → CUB`이다. 즉, base는 일반 객체 이미지, novel은 새 도메인의 세밀한 bird class다.

표준 5-way few-shot 설정에서 Conv-4 backbone으로 실험한 결과가 먼저 제시된다. `mini-ImageNet`에서 1-shot은 Baseline이 $42.11 \pm 0.71$, Baseline++가 $48.24 \pm 0.75$이고, MatchingNet $48.14 \pm 0.78$, ProtoNet $44.42 \pm 0.84$, MAML $46.47 \pm 0.82$, RelationNet $49.31 \pm 0.85$였다. 5-shot에서는 Baseline $62.53 \pm 0.69$, Baseline++ $66.43 \pm 0.63$, MatchingNet $63.48 \pm 0.66$, ProtoNet $64.24 \pm 0.72$, MAML $62.71 \pm 0.71$, RelationNet $66.60 \pm 0.69$였다. 즉 Baseline++는 당시 강력한 meta-learning 방법들과 거의 대등했다.

`CUB`에서는 이 경향이 더 분명하다. 1-shot에서 Baseline $47.12 \pm 0.74$에 비해 Baseline++는 $60.53 \pm 0.83$까지 올라가며, MatchingNet $60.52 \pm 0.88$, RelationNet $62.34 \pm 0.94$와 비슷한 수준이다. 5-shot에서는 Baseline++가 $79.34 \pm 0.61$로 매우 강했고, ProtoNet $76.39 \pm 0.64$, MAML $75.75 \pm 0.76$보다 높다. 이 결과는 shallow backbone 환경에서는 intra-class variation을 줄이는 것이 매우 중요하다는 주장을 뒷받침한다.

저자들은 Baseline 성능이 과거에 과소평가되었다고도 말한다. 실제로 data augmentation이 없는 Baseline*는 mini-ImageNet에서 1-shot $36.35 \pm 0.64$, 5-shot $54.50 \pm 0.66$으로 낮다. 반면 augmentation을 넣은 Baseline은 같은 backbone에서도 각각 $42.11$, $62.53$으로 크게 개선된다. 즉, 단순 baseline이 약하다고 여겨진 이유 중 일부는 알고리즘 자체보다 training recipe의 차이였다.

이후 backbone 깊이를 Conv-4, Conv-6, ResNet-10, ResNet-18, ResNet-34로 늘려본다. `CUB`에서는 backbone이 깊어질수록 모든 방법의 성능이 올라가고, 방법 간 격차가 줄어든다. 예를 들어 CUB 5-shot에서 Baseline은 Conv-4의 $64.16$에서 ResNet-34의 $84.27$까지 올라간다. ProtoNet도 $76.39$에서 $87.86$까지 오른다. 이 결과는 feature extractor가 강해질수록 기존 meta-learning 방법의 상대적 이점이 줄 수 있음을 시사한다.

`mini-ImageNet`에서는 양상이 더 복잡하지만, 특히 5-shot에서 deep backbone을 쓰면 Baseline이나 Baseline++가 일부 meta-learning 방법을 앞선다. 예를 들어 ResNet-18 기준 mini-ImageNet 5-shot에서 Baseline은 $74.27 \pm 0.63$, Baseline++는 $75.68 \pm 0.63$인데, MatchingNet은 $68.88 \pm 0.69$, MAML은 $65.72 \pm 0.77$, RelationNet은 $69.83 \pm 0.68$이다. 이는 깊은 backbone이 “학습 알고리즘의 sophistication”보다 더 큰 영향을 줄 수 있음을 보여준다.

논문의 가장 중요한 결과는 cross-domain setting이다. ResNet-18 backbone에서 `mini-ImageNet → CUB` 5-shot accuracy는 Baseline이 $65.57 \pm 0.70$으로 가장 높았다. Baseline++는 $62.04 \pm 0.76$, ProtoNet은 $62.02 \pm 0.70$, RelationNet은 $57.71 \pm 0.73$, MatchingNet은 $53.07 \pm 0.74$, MAML은 $51.34 \pm 0.72$였다. 저자들은 meta-learning 방법이 base dataset 내부의 episode들로 “learn to learn”했지만, 전혀 다른 도메인으로 넘어가면 그 적응 방식이 충분히 일반화되지 않는다고 해석한다. 반면 Baseline은 새 classifier를 직접 다시 학습하므로 domain shift에 덜 취약하다는 것이다.

추가로 저자들은 meta-learning 방법에도 further adaptation을 적용해 본다. MatchingNet, ProtoNet은 feature를 고정하고 softmax classifier를 다시 학습하는 식으로, MAML은 gradient update 횟수를 늘리는 식으로, RelationNet은 relation module을 추가 fine-tuning하는 식으로 적응을 더 수행한다. 그 결과 MatchingNet과 MAML은 특히 cross-domain setting에서 개선되지만, ProtoNet은 domain difference가 작은 경우 오히려 성능이 떨어지기도 한다. 논문은 이를 바탕으로 “meta-training 단계에서부터 adaptation 자체를 배우는 방향”이 미래 연구 과제라고 정리한다.

부록도 이 논지와 잘 맞는다. 예를 들어 Davies-Bouldin index로 측정한 intra-class variation은 backbone이 깊어질수록 감소했다. 또한 N-way meta-testing을 10-way, 20-way로 늘리면 Baseline++가 비교적 강한 성능을 유지했다. 이는 cosine-based classifier가 class separation 측면에서 더 유리할 수 있음을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 실험적 정직성이다. 새로운 모델을 과장하기보다, 비교 기준 자체를 재정비해 field의 혼란을 줄이려는 목적이 분명하다. 같은 optimizer, augmentation, backbone, training protocol 아래에서 여러 방법을 재구현하고 비교했다는 점은, few-shot literature에서 드문 높은 가치의 기여다.

또 다른 강점은 Baseline++와 cross-domain evaluation이라는 두 가지 실용적 메시지다. Baseline++는 구조적으로 매우 단순하면서도 strong baseline 역할을 하며, 이후 연구에서 “단순 baseline을 반드시 이겨야 한다”는 기준을 더 엄격하게 만드는 데 기여했다. 또한 `mini-ImageNet → CUB` 실험은 few-shot 성능이 같은 데이터셋 내부에서만 좋고 도메인이 달라지면 쉽게 무너질 수 있음을 보여주며, 평가 프로토콜의 현실성을 끌어올렸다.

방법론적으로도 장점이 있다. 논문은 “intra-class variation 감소”라는 비교적 명확한 설명 틀로 shallow backbone과 distance-based classifier의 효과를 해석한다. 단순히 숫자만 나열한 것이 아니라, 왜 Baseline++가 잘 되는지, 왜 deep backbone에서 격차가 줄어드는지, 왜 domain shift에서 Baseline이 강해지는지를 일관된 관점에서 설명하려 한다.

한계도 있다. 첫째, 논문은 이론적 분석보다 경험적 비교에 집중한다. 따라서 왜 특정 meta-learning 방법이 domain shift에 약한지에 대한 보다 깊은 원인 분석은 제한적이다. 둘째, Baseline++의 classifier 설계는 저자들의 신규 제안이 아니라 기존 아이디어의 단순화된 활용이다. 논문도 이를 명시한다. 즉, novelty는 알고리즘 자체보다 비교 연구와 재해석에 있다.

셋째, 실험 대상이 대표적이긴 하지만 모든 few-shot 방법을 포괄하지는 않는다. hallucination-based 방법은 비교의 복잡성을 이유로 제외되었다. 따라서 “few-shot meta-learning 전체가 baseline보다 못하다”라고 일반화하면 과도하다. 논문이 실제로 보여준 것은, 자신들이 선택한 대표 방법들과 특정 설정에서는 단순 baseline이 매우 강력하다는 점이다.

넷째, cross-domain 설정도 매우 중요한 제안이지만, domain shift의 종류가 하나뿐이다. `mini-ImageNet → CUB`는 일반 객체에서 fine-grained bird로 가는 특정 시나리오이므로, 다른 종류의 시각적 도메인 이동에서도 같은 결론이 유지되는지는 이 논문만으로 확정할 수 없다.

비판적으로 보면, 이 논문은 “meta-learning이 별로 필요 없다”기보다 “기존 few-shot benchmarking이 meta-learning의 이점을 과장했을 수 있다”는 주장으로 읽는 것이 정확하다. 실제로 further adaptation 실험에서 일부 meta-learning 방법은 개선된다. 이는 meta-learning의 잠재력을 부정하기보다, 훈련과 테스트 사이의 adaptation mismatch가 문제임을 시사한다.

## 6. 결론

이 논문은 few-shot classification에서 널리 받아들여지던 성능 서열을 다시 점검한 실험 중심 연구다. 핵심 기여는 세 가지다. 첫째, 대표적인 few-shot 방법들을 통일된 조건에서 비교하는 testbed를 제공했다. 둘째, cosine similarity 기반의 단순한 `Baseline++`가 shallow backbone 환경에서 매우 강력한 baseline임을 보였다. 셋째, base와 novel class 사이에 domain shift가 있는 현실적 설정에서는 복잡한 meta-learning보다 표준 fine-tuning baseline이 더 강할 수 있음을 보였다.

실제 적용 관점에서 이 연구는 매우 중요하다. 새로운 few-shot 알고리즘을 설계할 때는, 단순 baseline과 backbone choice, augmentation, evaluation protocol을 먼저 엄격히 통제해야 한다는 교훈을 준다. 또한 실제 응용에서는 같은 데이터셋 내부의 novel class보다 “다른 도메인의 새로운 클래스”를 다루는 경우가 많기 때문에, 향후 연구는 단순한 episodic training을 넘어 domain shift 하에서의 adaptation을 직접 학습하는 방향으로 발전할 필요가 있다.

전체적으로 이 논문은 flashy한 새 알고리즘을 내세우기보다, few-shot learning 연구가 어디에서 실제 진전을 만들고 어디에서 착시가 발생하는지 냉정하게 보여준다는 점에서 영향력이 크다. 특히 strong baseline의 중요성과 cross-domain generalization의 필요성을 분명히 했다는 점에서, 이후 few-shot 및 meta-learning 연구의 평가 기준을 더 엄격하게 만드는 데 의미 있는 역할을 한 논문이다.
