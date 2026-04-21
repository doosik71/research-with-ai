# Concept Learners for Few-Shot Learning

- **저자**: Kaidi Cao, Maria Brbić, Jure Leskovec
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2007.07375

## 1. 논문 개요

이 논문은 few-shot learning, 즉 클래스마다 극히 적은 수의 labeled example만 주어진 상황에서 새로운 분류 작업에 빠르게 일반화하는 방법을 다룬다. 저자들은 기존 meta-learning 방법들이 여러 과거 task로부터 “학습하는 방법을 학습”한다는 점에서는 성공적이지만, 사람이 새로운 개념을 배울 때 사용하는 방식처럼 지식을 구조화된 concept 단위로 다루지 않는다는 한계를 지적한다. 기존 방법은 대체로 하나의 큰 unstructured embedding space를 학습하고 그 안에서 모든 정보를 처리하는데, 저자들은 이 방식이 일반화와 해석 가능성 모두에 제약이 된다고 본다.

논문의 핵심 문제의식은 명확하다. 사람은 새로운 새 종을 배울 때 이미지 전체를 한 덩어리로 보지 않고, beak, wing, feather 같은 재사용 가능한 고수준 concept를 통해 차이를 파악한다. 반면 기존 few-shot meta-learning 모델은 이런 구조를 명시적으로 활용하지 않는다. 저자들은 바로 이 지점이 성능과 신뢰성의 병목이라고 주장한다.

이 문제는 중요하다. few-shot setting에서는 데이터가 매우 적어서 모델이 쉽게 overfit하거나, 반대로 지나치게 단순해질 수 있다. 더구나 예측 근거를 설명하기 어렵기 때문에 실제 적용에서 신뢰를 얻기 힘들다. 특히 biology처럼 고비용 라벨링이 필요한 분야에서는 적은 예제로도 잘 일반화하고, 동시에 왜 그런 예측을 했는지 설명할 수 있는 모델이 실질적으로 중요하다. 이 논문은 정확도와 interpretability를 동시에 겨냥한다는 점에서 의미가 크다.

## 2. 핵심 아이디어

논문이 제안하는 방법은 COMET이다. 이름 그대로, 이 방법은 전체 입력에 대해 하나의 공통 metric space를 학습하는 대신, 사람이 이해할 수 있는 여러 concept dimension마다 별도의 concept learner를 둔다. 각 concept learner는 해당 concept와 관련된 입력 부분만 보고 embedding을 만들고, 그 embedding 공간 안에서 class prototype을 계산한다. 최종 분류는 여러 concept 공간에서 계산된 class별 거리 정보를 합쳐서 수행한다.

중심 직관은 “복잡한 분류 문제를 여러 해석 가능한 부분 문제로 나누면 일반화가 쉬워진다”는 것이다. 예를 들어 새 분류에서는 beak만 보는 learner, wing만 보는 learner, tail만 보는 learner를 따로 두고, 이들의 판단을 합친다. 그러면 전체 이미지 공간에서 바로 판별하는 것보다 각 concept 공간에서 class 간 차이가 더 단순하고 안정적으로 드러날 수 있다.

저자들이 강조하는 COMET의 장점은 세 가지다. 첫째, semi-structured representation learning이다. 완전히 hand-crafted한 symbolic system은 아니지만, 전혀 구조 없는 embedding도 아니다. 둘째, concept-specific metric space와 concept prototype을 사용한다. 즉 각 concept마다 따로 class 대표점을 만든다. 셋째, 다수 concept learner의 결합을 통해 ensemble 효과를 얻는다. 이 조합이 일반화 성능을 높인다고 본다.

기존 Prototypical Networks와의 차이도 분명하다. ProtoNet은 하나의 embedding function $f_\theta$만 학습하고, 그 공간 안에서 class prototype을 만든다. 반면 COMET은 concept별로 별도의 embedding function $f_\theta^{(j)}$와 prototype $p_k^{(j)}$를 둔다. 따라서 prototype이 단순히 “클래스 전체의 평균 표현”이 아니라, “특정 concept 관점에서 본 클래스의 대표 표현”이 된다. 이 점이 구조적이고 해석 가능한 reasoning을 가능하게 한다.

또 하나 중요한 차별점은 concept가 완벽하지 않아도 된다는 주장이다. 저자들은 concept 집합이 noisy, incomplete, overlapping, redundant해도 COMET이 유용하게 작동한다고 실험으로 보인다. 즉 concept annotation이 약하거나 자동 추출된 경우에도 성능 향상을 기대할 수 있다는 것이 논문의 실용적 포인트다.

## 3. 상세 방법 설명

문제 설정은 표준 few-shot classification이다. 훈련 시에는 기존 class들로 구성된 training tasks를 episodic training 방식으로 사용하고, 테스트 시에는 training class와 겹치지 않는 새로운 class들에 대해 support set 몇 개만 보고 query를 분류한다. 각 episode는 보통 $N$-way, $k$-shot 형태로 샘플링되며, 이 논문에서는 주로 5-way 1-shot, 5-way 5-shot을 평가한다.

논문의 출발점은 Prototypical Networks이다. ProtoNet에서는 embedding function $f_\theta : \mathbb{R}^D \to \mathbb{R}^M$를 학습하고, 각 class $k$의 prototype을 support example 평균으로 만든다.

$$
p_k = \frac{1}{|S_k|} \sum_{(x_i, y_i)\in S_k} f_\theta(x_i)
$$

이후 query $x_q$는 embedding된 뒤 각 class prototype과의 거리로 분류된다.

$$
p_\theta(y=k \mid x_q)=
\frac{\exp(-d(f_\theta(x_q), p_k))}
{\sum_{k'} \exp(-d(f_\theta(x_q), p_{k'}))}
$$

COMET은 여기서 입력 feature를 concept별 subspace로 나누는 방향으로 확장한다. concept 집합을 $C=\{c^{(j)}\}_{j=1}^N$이라 두며, 각 concept $c^{(j)} \in \{0,1\}^D$는 어떤 입력 차원이 그 concept를 설명하는 데 쓰이는지를 나타내는 binary mask이다. 여기서 중요한 점은 concept들이 서로 겹쳐도 되고, 완전하지 않아도 되며, redundancy가 있어도 허용된다는 것이다.

각 concept $j$에 대해 별도의 concept learner $f_\theta^{(j)}$를 둔다. 이 learner는 입력 전체가 아니라 concept mask가 적용된 입력 $x_i \circ c^{(j)}$를 사용해 embedding을 만든다. 여기서 $\circ$는 Hadamard product, 즉 element-wise product이다. 그러면 class $k$의 concept prototype은 다음과 같이 계산된다.

$$
p_k^{(j)}=
\frac{1}{|S_k|}
\sum_{(x_i,y_i)\in S_k}
f_\theta^{(j)}(x_i \circ c^{(j)})
$$

즉 각 class는 하나의 prototype이 아니라, concept마다 하나씩 총 $N$개의 prototype $\{p_k^{(j)}\}_{j=1}^N$으로 표현된다.

query sample $x_q$가 들어오면, 각 concept learner가 concept-specific embedding $f_\theta^{(j)}(x_q \circ c^{(j)})$를 만든다. 그런 다음 class $k$에 대해 각 concept 공간에서 query embedding과 class concept prototype 사이 거리를 계산하고, 그것들을 모두 더한다. 최종 class posterior는 다음과 같다.

$$
p_\theta(y=k \mid x_q)=
\frac{
\exp\left(
-\sum_j d\left(f_\theta^{(j)}(x_q \circ c^{(j)}), p_k^{(j)}\right)
\right)
}{
\sum_{k'}
\exp\left(
-\sum_j d\left(f_\theta^{(j)}(x_q \circ c^{(j)}), p_{k'}^{(j)}\right)
\right)
}
$$

직관적으로 보면, 각 concept learner가 “이 query는 class $k$의 beak prototype과 얼마나 가까운가”, “wing prototype과는 얼마나 가까운가”를 따로 판단하고, 마지막에 이것을 종합해 가장 그럴듯한 class를 선택하는 구조다. 이는 하나의 거대한 embedding space에서 한 번에 판단하는 것보다 더 구조적인 추론 방식이다.

학습 목표는 true class에 대한 negative log-likelihood이다.

$$
L_\theta = -\log p_\theta(y=k \mid x_q)
$$

훈련은 episodic fashion으로 이루어진다. 즉 각 episode에서 support set으로 prototype을 만들고, query set에 대한 loss를 계산해 업데이트한다. 논문은 distance function으로 Euclidean distance를 사용하며, appendix의 ablation에서 cosine distance보다 consistently better하다고 보고한다. 또한 서로 다른 concept 공간의 거리값을 비교 가능하게 만들기 위해 batch normalization이 중요하다고 명시한다.

이 논문의 interpretability 메커니즘은 모델 구조 안에 직접 들어 있다. 특정 query와 class에 대해 concept $j$의 local importance는 query의 concept embedding과 class concept prototype 사이 거리의 inverse로 정의된다. 거리가 작을수록 그 concept가 해당 class 예측에 더 크게 기여했다고 본다. 따라서 prediction explanation은 post-hoc attribution이 아니라, 실제 분류 과정에서 사용된 concept-level similarity에서 직접 나온다.

global explanation도 유사하다. 특정 class 또는 관심 있는 query 집합 전체에 대해, concept prototype과 query들의 concept embedding 사이 평균 거리를 계산하고, 그 inverse를 global concept importance로 사용한다. 이 값으로 class 수준에서 어떤 concept가 중요한지를 ranking할 수 있다.

또 하나 흥미로운 기능은 locally similar examples 검색이다. concept $j$를 고정한 상태에서, 각 sample의 concept embedding이 class prototype $p_k^{(j)}$와 얼마나 가까운지로 정렬하면, 같은 class 안에서 그 concept를 잘 드러내는 예시, 혹은 반대로 잘 드러내지 못하는 예시를 찾을 수 있다. 논문은 이를 misannotation이나 concept visibility 문제를 파악하는 데도 쓸 수 있다고 말한다.

구현 측면에서 데이터셋마다 backbone이 다르다. CUB 이미지 분류에는 Conv-4 backbone을 기본으로 사용하고, Tabula Muris 생물 데이터에는 batch normalization, ReLU, dropout이 포함된 2-layer fully connected network를 사용한다. CUB에서는 training speed를 높이기 위해 concept learner 간 weight를 공유한다. 구체적으로는 이미지 전체를 convolutional network에 통과시켜 spatial feature embedding을 만든 뒤, 그 feature에 concept mask를 적용하는 방식이다. 저자들은 convolution의 local property 때문에 입력 초기에 mask를 적용하는 것과 유사한 성능을 얻는다고 설명한다.

## 4. 실험 및 결과

실험은 computer vision, NLP, biology의 세 도메인, 네 개 데이터셋에서 수행되었다. 이미지 분야에서는 CUB-200-2011과 Flowers-102를 사용했고, NLP에서는 Reuters 문서 분류 데이터셋을 사용했다. biology에서는 저자들이 새롭게 구성한 Tabula Muris 기반 single-cell transcriptomic few-shot dataset을 제안했다. 이 데이터셋은 mouse 23개 organ에서 수집한 105,960개 cell, 124개 cell type을 포함하고, 23,341개 gene 중 2,866개 high-dispersion gene을 feature로 선택했다. concept는 Gene Ontology level 3 term 중 최소 64개 gene이 할당된 190개 term으로 정의했다. 훈련/검증/테스트를 서로 다른 organ 기준으로 나눠, 보지 못한 organ과 cell type에 대한 generalization을 평가하는 설계가 특히 중요하다.

평가 프로토콜은 5-way classification이며, 각 episode에서 class 5개를 샘플링하고, class당 $k$개의 support example을 사용한다. query set은 class당이 아니라 전체적으로 class들에 속하는 unlabeled sample 16개로 구성된다고 서술되어 있다. 최종 성능은 test split에서 600개의 랜덤 episode 평균 accuracy와 standard deviation으로 보고한다.

비교 대상은 매우 강하다. FineTune/Baseline++, Matching Networks, MAML, Relation Networks, MetaOptNet, DeepEMD, ProtoNet이 포함된다. 즉 단순 baseline부터 대표적인 optimization-based, metric-based meta-learning 방법까지 폭넓게 비교했다.

핵심 정량 결과는 Table 1이다. COMET은 CUB, Tabula Muris, Reuters에서 모두 최고 성능을 기록했다. CUB에서는 1-shot에서 $67.9 \pm 0.9$, 5-shot에서 $85.3 \pm 0.5$를 달성했고, 가장 강한 baseline인 DeepEMD의 $64.0 \pm 1.0$, $81.1 \pm 0.7$보다 높다. Tabula Muris에서는 1-shot $79.4 \pm 0.9$, 5-shot $91.7 \pm 0.5$로, MetaOptNet의 $73.6 \pm 1.1$, $85.4 \pm 0.9$를 크게 앞선다. Reuters에서도 1-shot $71.5 \pm 0.7$, 5-shot $89.8 \pm 0.3$로 최고 성능이다.

저자들은 COMET이 가장 좋은 baseline 대비 평균적으로 1-shot에서 9.5%, 5-shot에서 9.3% 개선되었다고 요약한다. 특히 ProtoNet 대비 1-shot task에서 19~23% improvement를 강조한다. 이 수치는 COMET이 사실상 ProtoNet의 구조적 확장이라는 점을 생각하면 매우 중요하다. 단순히 backbone이나 training trick 차이가 아니라, concept decomposition 자체가 실질적인 성능 향상을 낳았다는 주장을 뒷받침한다.

추가로 COMET의 향상이 단순한 parameter 증가나 ensemble 효과 때문이 아니라는 점을 검증한다. Table 2에서 ensemble of ProtoNets와 COMET shared weights, full COMET을 비교했는데, COMET은 weight를 concept 간 공유한 경우에도 ProtoNet ensemble보다 훨씬 높다. 예를 들어 Tabula Muris 1-shot에서 ProtoNetEns는 $67.2 \pm 0.8$인데, COMET shared weights는 $78.2 \pm 1.0$이다. 따라서 “여러 모델을 합쳐서 좋아진 것”만으로는 설명되지 않고, concept prototype 구조가 중요한 역할을 한다는 해석이 타당하다.

concept 개수의 효과도 인상적이다. Figure 2에 따르면 concept 수를 늘릴수록 CUB와 Tabula Muris에서 성능이 일관되게 상승한다. CUB에서는 전체 이미지를 의미하는 global concept 하나만 쓰는 ProtoNet-like setting에서 시작해, 가장 자주 보이는 bird part 하나인 beak concept만 추가해도 1-shot에서 10%, 5-shot에서 5% improvement를 얻는다. Tabula Muris에서는 단 8개 concept만으로도 baseline들을 넘어서며, 더 많은 concept를 넣어도 성능이 계속 좋아진다. 저자들은 Gene Ontology의 여러 계층을 모두 포함해 1,500개 concept까지 확장했을 때도 성능이 약간 더 좋아졌다고 보고한다. 이는 noisy, overlapping, redundant concept 집합에도 COMET이 robust하다는 주장의 핵심 증거다.

자동 concept 추출 실험도 중요하다. 이미지에서는 unsupervised landmark discovery 방법으로 30개 landmark를 추출하고, 이를 concept mask로 바꾸었다. 이렇게 자동으로 얻은 noisy concept를 사용해도 COMET은 CUB 1-shot $64.8 \pm 1.0$, 5-shot $82.0 \pm 0.5$, Flowers 1-shot $70.4 \pm 0.9$, 5-shot $86.7 \pm 0.6$을 기록하며 모든 baseline을 이겼다. Flowers에서는 best baseline 대비 각각 4.8%, 4.6% improvement다. 이는 사람이 검증한 concept가 없더라도 COMET이 여전히 실용적일 수 있음을 보여준다.

Tabula Muris와 Reuters에서는 prior knowledge 없이 random mask subset을 concept처럼 정의하고, validation set에서 importance score가 높은 concept를 선택하는 방식도 실험했다. 이 경우 human-defined concept보다 accuracy가 Tabula Muris에서 약 2%, Reuters에서 약 1% 정도만 낮았다. 즉 COMET의 구조적 장점은 concept quality가 완벽하지 않아도 상당 부분 유지된다.

interpretability 평가도 단순한 시각화 수준에 그치지 않는다. Tabula Muris에서는 각 cell type에 대해 differential expression gene set을 구하고, 그 gene set에 유의하게 enriched된 Gene Ontology term을 ground-truth concept explanation으로 사용했다. 그 후 COMET의 global importance ranking이 top 20 안에서 얼마나 많은 relevant term을 회수하는지 측정했는데, average recall@20이 0.71이었다. 이는 class-level explanation이 biological ground truth와 상당히 잘 일치함을 의미한다. CUB에서는 beak, belly, forehead 등이 자주 중요한 concept로 선택되었고, 개별 species의 알려진 형태적 특징과도 잘 맞는다고 서술한다.

또한 local similarity 분석에서는 특정 concept에 대해 prototype에 가장 가까운 이미지와 가장 먼 이미지를 정렬해 보여준다. 예를 들어 chipping sparrow의 belly concept에서 prototype에 가까운 이미지들은 belly가 잘 보이고 prototypical appearance를 갖는 반면, 먼 이미지들은 belly가 잘 보이지 않거나 annotation이 부정확한 경우가 많았다. 이는 COMET의 concept-level distance가 의미 있는 local pattern similarity를 포착한다는 정성적 증거다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 interpretability를 정확도 손실 없이 구조 안에 통합했다는 점이다. 많은 설명 가능한 AI 방법은 post-hoc explanation을 붙이거나, 해석 가능성을 높이는 대신 성능이 떨어지는 경우가 많다. 반면 COMET은 예측 자체가 concept-level prototype 비교를 통해 이루어지므로, 설명이 모델 동작과 직접 연결되어 있다. 게다가 저자들은 실제로 기존 강력한 few-shot baseline들보다 성능이 높다는 결과를 제시했다.

두 번째 강점은 domain-agnostic하다는 점이다. 이미지, 문서, single-cell biology에 모두 적용되었고, 특히 biology 데이터셋에서는 concept를 Gene Ontology로 정의해 domain knowledge와 잘 결합했다. 기존 compositional few-shot 접근들이 주로 시각적 part나 attribute annotation에 의존했던 것과 달리, COMET은 입력 차원을 concept mask로 정의할 수만 있으면 다양한 modality에 적용 가능하다는 장점이 있다.

세 번째 강점은 concept의 질에 대한 강건성이다. 논문은 concept가 incomplete, noisy, overlapping, redundant해도 성능이 유지되거나 개선될 수 있음을 보여준다. 이는 실제 환경에서 매우 중요하다. 현실의 ontology나 자동 추출 concept는 거의 항상 불완전하기 때문이다.

네 번째 강점은 새로운 biology meta-learning benchmark를 제안했다는 점이다. Tabula Muris 기반 cross-organ cell type classification setting은 few-shot learning의 활용 가능성을 biomedical domain으로 넓혔고, 단순 부록 수준이 아니라 논문의 중요한 기여 중 하나다.

한계도 분명하다. 첫째, COMET은 concept 집합이 어느 정도는 주어지거나 추출 가능하다는 가정을 둔다. 저자들은 무감독 landmark나 random mask 실험으로 이 의존성을 완화하려 했지만, concept 정의 방식이 성능과 해석 가능성의 핵심 입력이라는 점은 변하지 않는다. 즉 concept가 정말 의미 없는 경우에도 얼마나 안정적인지에 대해서는 이 논문만으로 완전히 판단하기 어렵다.

둘째, concept importance score가 거리 기반 inverse로 정의되는데, 이것이 항상 인간이 기대하는 “인과적 중요도”와 일치한다고 보장되지는 않는다. 논문은 relevance ranking과 biological enrichment를 통해 설득력 있는 근거를 제시하지만, importance가 어디까지나 embedding-space geometry에서 나온 값이라는 점은 유의해야 한다.

셋째, concept learner 수가 많아질수록 계산량과 메모리 비용이 커질 수 있다. 논문은 CUB에서 weight sharing으로 이를 완화했고, ensemble 비교에서도 shared weights 설정을 사용했다. 하지만 다양한 모달리티에서 매우 많은 concept를 독립 network로 둘 때의 실제 비용 문제는 완전히 해소되었다고 보기는 어렵다.

넷째, 논문은 주로 classification accuracy 중심으로 평가하며, concept annotation 품질과 explanation의 사용자 수준 usefulness를 직접 측정하지는 않는다. Tabula Muris의 recall@20 평가는 좋은 시작이지만, 인간 사용자가 이 설명을 실제 의사결정에 어떻게 활용할 수 있는지까지는 다루지 않는다.

다섯째, COMET이 왜 어떤 task에서 더 크게 이득을 보는지에 대한 이론적 분석은 제한적이다. 저자들은 semi-structured representation, concept-specific prototype, ensembling을 주요 원인으로 제시하지만, 각각의 기여를 완전히 분리한 정교한 분석까지는 본문에 충분히 제시되지 않는다. 다만 ProtoNet ensemble 및 shared-weight ablation은 최소한 단순 ensemble 효과 이상의 구조적 이점이 있음을 보여준다.

## 6. 결론

이 논문은 few-shot learning에서 “새 task를 적은 예제로 빨리 배우는 능력”을 높이기 위해, 사람의 개념적 추론 방식에 가까운 concept-based meta-learning 구조를 제안했다. COMET은 각 concept마다 별도의 metric space와 prototype을 학습하고, 이들을 결합해 최종 분류를 수행한다. 그 결과 기존 few-shot baselines보다 더 높은 정확도를 달성하면서도, 어떤 concept가 예측에 기여했는지를 local/global 수준에서 설명할 수 있다.

실험 결과를 보면 이 방법의 기여는 단순히 해석 가능성을 덧붙인 수준이 아니다. CUB, Reuters, Tabula Muris, Flowers 등 다양한 도메인에서 일관된 성능 향상을 보였고, 특히 가장 어려운 1-shot setting에서 improvement가 크게 나타났다. 또한 자동 추출 concept나 noisy concept에서도 효과가 유지되어, 실제 적용 가능성도 높다.

향후 연구 관점에서 이 논문은 두 가지 방향에서 중요하다. 하나는 few-shot learning이 단순히 좋은 embedding을 찾는 문제를 넘어서, 어떤 구조적 decomposition이 generalization에 유리한지 탐구하게 만든다는 점이다. 다른 하나는 interpretable meta-learning이라는 거의 비어 있던 영역에 실질적인 출발점을 제공했다는 점이다. 특히 biomedical AI처럼 데이터가 적고 설명 가능성이 중요한 분야에서, COMET류 접근은 실제 활용 가치가 클 가능성이 높다.
