# FlexKBQA: A Flexible LLM-Powered Framework for Few-Shot Knowledge Base Question Answering

- **저자**: Zhenyu Li, Sunqi Fan, Yu Gu, Xiuxing Li, Zhichao Duan, Bowen Dong, Ning Liu, Jianyong Wang
- **발표연도**: 2024
- **arXiv**: https://arxiv.org/abs/2308.12060

## 1. 논문 개요

이 논문은 **few-shot knowledge base question answering (KBQA)** 문제를 다룬다. KBQA는 사용자의 자연어 질문을 knowledge base의 실행 가능한 질의 형태, 예를 들어 **SPARQL**이나 **S-expression**으로 변환한 뒤, 이를 KB에 실행하여 정답을 얻는 문제다. 기존 KBQA 시스템은 대체로 대규모 수작업 annotation에 의존하는데, 실제 환경에서는 이 가정이 잘 성립하지 않는다. 이유는 KB가 매우 크고 계속 변하며, 서로 다른 KB마다 schema와 query language가 달라서 데이터 구축과 모델 적응 비용이 매우 크기 때문이다.

논문의 목표는 **소수의 annotation만으로도 다양한 KB와 query language에 적용 가능한 유연한 KBQA 프레임워크**를 만드는 것이다. 이를 위해 저자들은 **LLM을 직접 추론 엔진으로 쓰기보다, program을 자연어 질문으로 번역하는 데이터 생성기(program translator)** 로 사용한다. 이렇게 생성한 synthetic data로 lightweight KBQA 모델을 학습시키고, 이후 실제 사용자 질문에 대해 self-training을 수행해 synthetic-real 간 분포 차이를 줄인다.

이 문제가 중요한 이유는 매우 분명하다. KBQA는 해석 가능하고 구조화된 질의가 가능하다는 장점이 있지만, 실제 서비스 수준에서 필요한 annotation 비용이 높다. 만약 적은 label만으로도 KBQA 시스템을 만들 수 있다면, 새로운 도메인이나 새로운 KB에 대한 진입 장벽이 크게 낮아진다. 논문은 바로 이 지점을 겨냥해, **LLM의 생성 능력과 경량 모델의 배포 용이성**을 결합하는 방향을 제안한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 다음과 같다. 기존 few-shot KBQA 연구는 LLM에게 질문을 직접 program으로 바꾸게 하는 **in-context learning** 방식을 주로 사용했다. 하지만 저자들은 이 방향이 context window 한계, 높은 추론 비용, domain-specific KB에 대한 grounding 오류 때문에 확장성이 낮다고 본다. 대신 이 논문은 방향을 뒤집어, **KB에서 먼저 실행 가능한 program을 대량으로 만들고, 이를 LLM이 자연어 질문으로 번역하게 한다**. 즉, 어려운 “질문 → program” 문제를 LLM에게 직접 맡기지 않고, 더 자연스러운 “program → 질문” 번역 문제로 바꾼다.

이 설계의 핵심 직관은 두 가지다. 첫째, program은 KB에서 직접 샘플링하므로 실행 가능성이 보장된다. 둘째, LLM은 자연어 생성에는 강하므로, program을 자연어 질문으로 옮기는 작업에서 더 안정적으로 synthetic training pair를 만들 수 있다. 이렇게 생성된 question-program pair를 이용해 작은 KBQA 모델을 fine-tuning하면, 실제 추론은 LLM 없이도 효율적으로 수행할 수 있다.

하지만 synthetic question은 실제 사용자 질문과 분포가 다를 수 있다. 이를 해결하기 위해 논문은 **Execution-Guided Self-Training (EGST)** 를 도입한다. 이 방법은 unlabeled real user question에 대해 현재 모델이 pseudo program을 예측하고, 그 program을 실제 KB에 실행해 오류 여부와 의미적 타당성을 검사한 뒤, 통과한 샘플만 다시 학습에 사용한다. 즉, synthetic data로 시작하되 점차 real question 분포에 적응하도록 만든다.

추가로 저자들은 **Inherent Reasoning (IR)** 이라는 개념을 사용한다. 이는 LLM이 파라미터 내부에 내재한 지식을 이용해 질문에 직접 답을 생성하는 방식이다. 논문에서는 IR을 주된 KBQA 방법으로 쓰지 않고, self-training 중 pseudo-label 정제와 최종 answer 보완에 사용하는 **보조 신호**로 활용한다. 이 점이 중요하다. 저자들은 해석 가능성과 실행 가능성을 유지하기 위해 semantic parsing을 중심에 두고, IR은 보조 역할로 제한했다.

## 3. 상세 방법 설명

FlexKBQA는 크게 네 단계로 이해할 수 있다.

### 3.1 전체 파이프라인

전체 흐름은 다음과 같다.

1. 소수의 labeled question-program pair를 준비한다.
2. 이 소량의 데이터에서 program template를 수집한다.
3. template를 KB 위에서 grounding하여 많은 실행 가능한 program을 자동 생성한다.
4. LLM이 각 program을 자연어 질문으로 번역하여 synthetic dataset을 만든다.
5. synthetic data와 소량의 real labeled data로 lightweight model을 학습한다.
6. unlabeled user question에 대해 pseudo-label을 만들고, 실행 결과 기반 filtering을 거쳐 다시 학습한다.
7. 필요하면 IR 결과를 filtering과 최종 응답 보완에 사용한다.

논문 Figure 2가 이 구조를 요약한다. 중요한 점은 LLM이 직접 서비스 단계에서 모든 질문을 푸는 것이 아니라, **데이터 생성과 보조 추론 역할**에 쓰이고, 실제 배포는 가벼운 모델이 담당한다는 것이다.

### 3.2 KBQA의 기본 학습 목표

논문은 KB를 다음과 같이 표현한다.

$$
K \in E \times R \times (E \cup L \cup C)
$$

여기서 $E$는 entity 집합, $R$은 relation 집합, $C$는 class 집합이다. KBQA의 목표는 자연어 질문을 program으로 변환하고, 이를 KB에 실행해 답을 얻는 것이다.

논문은 underlying model로 ranking model과 generation model을 사용하는 일반적 semantic parsing 틀을 설명한다.

랭킹 모델의 loss는 다음과 같다.

$$
L_{ranker} = - \frac{e^{s(q,p)}}{e^{s(q,p)} + \sum_{p_i \in P \land p_i \neq p} e^{s(q,p_i)}}
$$

여기서 $p$는 정답 program, $P$는 candidate program 집합, $s(q,p_i)$는 질문 $q$와 후보 program $p_i$ 사이의 similarity score다. 직관적으로는 정답 program의 점수를 다른 후보보다 높이도록 학습하는 목적이다.

생성 모델의 loss는 다음과 같다.

$$
L_{gen} = - \sum_{t=1}^{n} \log \big( probability(p_t \mid p_{<t}; q; r) \big)
$$

여기서 $p_t$는 program의 $t$번째 토큰, $q$는 질문, $r$은 ranking 결과 같은 추가 정보다. 즉, generation model은 질문과 보조 정보를 보고 정답 program을 순차적으로 생성한다.

### 3.3 Automatic Program Sampling

이 단계의 목표는 **실행 가능한(valid)** program을 대량 확보하는 것이다. 논문은 이를 두 단계로 나눈다.

첫째는 **template collection**이다. 예를 들어 SPARQL query 안의 entity와 relation을 변수로 치환하면 구조만 남은 template를 얻을 수 있다. 예를 들어 `ent_0`, `rel_0`, `ent_1` 같은 placeholder를 넣는 방식이다. 이런 template는 질문 유형의 논리 구조를 담고 있으므로, 소수의 annotated example에서 몇 개만 수집해도 다양한 형태를 포괄할 수 있다고 본다. Appendix A에 따르면 실험에서는 few-shot annotated sample의 program에서 template를 만들었다.

둘째는 **step-wise grounding**이다. template 안의 변수들에 실제 entity와 relation 값을 채워 넣어 executable program을 만든다. 논문은 한 번에 모든 변수를 질의하면 대규모 KB에서 실행이 너무 느리거나 오류가 나기 쉽다고 지적한다. 그래서 변수 값을 한 번에 다 찾지 않고, 예를 들어 `ent_0 → rel_0 → ent_1` 순서처럼 **반복적으로 하나씩 grounding**한다. 저자들은 이 방식이 search space를 줄이면서도 program 다양성을 유지한다고 주장한다.

이 단계의 의미는 매우 크다. synthetic question 생성 전에 먼저 executable program을 확보하므로, 학습 데이터의 정답 측 program은 품질이 높다. 즉, noisy natural language는 있을 수 있어도, symbolic side는 비교적 깨끗하다.

### 3.4 Low-Resource Program Translation

이 논문의 가장 핵심적인 설계다. LLM은 각 sampled program $p_i^s$를 자연어 질문 $q_i^s$로 변환한다. 논문은 이를 다음과 같이 쓴다.

$$
q_i^s \leftarrow Translator(Inst; (p_1^f, q_1^f), \dots, (p_N^f, q_N^f); p_i^s)
$$

여기서 `Inst`는 번역 지시문, $(p_j^f, q_j^f)$는 few-shot demonstration, $N$은 shot 수다. $N=0$이면 zero-shot setting이다.

저자들의 논리는 분명하다. LLM은 원래 자연어를 잘 다루므로, domain-specific KB에 grounding된 program을 직접 생성하게 하는 것보다, 이미 올바른 program을 자연어 질문으로 바꾸게 하는 편이 쉽고 안정적이라는 것이다. Appendix F의 보조 실험도 이 점을 뒷받침한다. “real program, synthetic question” 설정에서는 real data로 학습한 경우보다 성능이 조금만 낮았고, 이는 **LLM이 symbolic language를 자연어로 번역하는 능력은 꽤 강함**을 뜻한다.

Appendix G에서는 in-context example 수를 10, 25, 60개로 바꿔도 최종 성능 차이가 작았다고 보고한다. 저자들은 LLM이 이미 SPARQL이나 S-expression 같은 program language에 대한 지식을 파라미터에 상당 부분 내장하고 있기 때문이라고 해석한다.

### 3.5 Execution-Guided Self-Training (EGST)

synthetic data만으로는 실제 사용자 질문 분포를 충분히 반영하지 못한다. 논문은 이를 주요 문제로 본다. 특히 entity와 relation 분포가 다를 때 성능 저하가 크며, 이전 연구에서는 GrailQA로 학습한 모델이 WebQSP에 직접 학습한 모델 대비 약 65% 수준 성능만 낼 수 있다고 언급한다.

EGST는 이를 해결하기 위한 iterative teacher-student self-training이다. 알고리즘은 다음과 같다.

1. synthetic data $D_s$와 소량의 labeled data $D_f$로 teacher model $\theta_{tea}$를 학습한다.
2. teacher가 unlabeled user question $q_i^u$에 대해 pseudo program $p_i^u$를 생성한다.
3. filtering을 통해 신뢰할 수 있는 pseudo pair만 남긴다.
4. synthetic data, few-shot labeled data, filtered pseudo-labeled data를 합쳐 student model $\theta_{stu}$를 학습한다.
5. student를 다음 iteration의 teacher로 갱신한다.
6. 수렴할 때까지 반복한다.

논문 Algorithm 1이 이를 제시한다.

이때 filtering은 세 종류다.

- **Error Filtering**: 예측한 SPARQL이 실행 오류를 내거나 답을 반환하지 않으면 제거한다.
- **Semantic Filtering**: 질문과 예측 program 속 relation 사이의 semantic similarity를 계산해 너무 낮으면 제거한다.
- **Inherent Reasoning Filtering**: pseudo answer가 IR 결과와 일치하지 않으면 제거한다.

Appendix C에 따르면 semantic filtering에는 `all-MiniLM-L6-v2` sentence-transformer를 사용했고, cosine similarity 평균이 0.2 미만인 샘플을 걸렀다. 즉, 질문과 relation이 의미적으로 너무 동떨어진 pseudo label을 제거하려는 것이다.

이 설계는 KBQA에서 특히 합리적이다. 일반 self-training은 pseudo-label이 틀리면 오류가 증폭될 수 있는데, FlexKBQA는 **program을 실제 KB에서 실행할 수 있다**는 점을 활용해 오류를 강하게 제어한다.

### 3.6 Inherent Reasoning Augmentation

IR은 LLM이 KB를 명시적으로 조회하지 않고도 내재 지식으로 답을 생성하는 능력을 뜻한다. 논문은 이를 semantic parsing의 대체가 아니라 보완으로 사용한다.

첫째, self-training 과정에서 pseudo-labeled sample의 answer가 IR answer와 맞는 경우만 남겨 training purity를 높인다.

둘째, semantic parsing이 실패하는 경우, 예를 들어 candidate ranking을 못 하거나 program execution error가 나는 경우, IR의 답을 최종 답으로 사용할 수 있다고 제안한다.

특히 KQA Pro에서는 entity linking 단계가 없기 때문에, 관계나 엔티티를 정확히 기억하지 못하면 executable SPARQL 생성이 어려워진다. 이때 IR이 큰 보완 효과를 보였다고 보고한다. 다만 이 방식은 interpretability가 떨어질 수 있으므로, 논문도 IR만으로 충분하다고 주장하지는 않는다.

### 3.7 Underlying Models

FlexKBQA는 model-agnostic이라고 하지만, 실험에서는 데이터셋별로 서로 다른 underlying model을 사용했다.

- GrailQA, WebQSP: **RnG-KBQA**
- KQA Pro: **BART-SPARQL**

Appendix D에 따르면 RnG-KBQA는 ranking-then-generate 구조다. 질문에서 topic entity를 찾고 candidate program을 검색한 뒤, ranker가 상위 후보를 고르고, generator가 질문과 top-k candidate를 바탕으로 최종 answer를 생성한다. ranker는 `BERT-base-uncased`, generator는 `T5-base`로 초기화했다.

반면 BART-SPARQL은 질문에서 직접 SPARQL을 end-to-end로 생성한다. 이 방식은 entity linking이 없는 KQA Pro에서 KB의 relation과 entity를 모델이 내부적으로 더 많이 기억해야 하므로 unseen question에 더 민감하다고 설명한다.

## 4. 실험 및 결과

### 4.1 데이터셋과 설정

논문은 세 가지 대표 데이터셋으로 평가한다.

- **GrailQA**: Freebase 기반, 64,331개 질문, program type은 S-expression. i.i.d., compositional, zero-shot generalization split을 제공한다.
- **WebQSP**: Freebase 기반, 4,737개 질문, program type은 SPARQL. Google query logs 기반이라 실제 사용자 질문 성격이 강하다.
- **KQA Pro**: Wikidata 기반, 117,970개 질문, program type은 SPARQL. multi-hop, comparison, set operation 등 복합 reasoning이 포함되며 entity linking stage가 없다.

실험에서 원래 training set은 unlabeled real user question으로 취급했고, synthetic pair는 Freebase에서 6,184개, Wikidata에서 5,017개를 만들었다고 한다. program translation에 사용한 LLM은 **gpt-3.5-turbo**다. GrailQA에서는 TIARA의 entity linking 결과, WebQSP에서는 ELQ를 사용했다.

### 4.2 비교 대상

few-shot baseline으로는 주로 두 모델과 비교한다.

- **Pangu**
- **KB-BINDER**

KQA Pro에는 이들 결과가 없어, 저자들이 직접 **LLM-ICL** baseline을 구현해 비교했다. 또 supervised model들과도 비교하여 few-shot임에도 어디까지 따라가는지 보여준다.

### 4.3 GrailQA 결과

GrailQA에서 25-shot FlexKBQA는 test set 기준 **EM 62.8, F1 69.4**를 기록했다. 이는 이전 few-shot SOTA였던 Pangu의 F1 62.7보다 **6.7 포인트 높다**. 더 흥미로운 점은 fully supervised RnG-KBQA의 F1 74.4와 비교했을 때, FlexKBQA가 그 성능의 약 **93% 수준**에 도달했다는 점이다.

세부 split에서도 일관되게 강했다.

- I.I.D.: EM 71.3, F1 75.8
- Compositional: EM 59.1, F1 65.4
- Zero-shot split: EM 60.6, F1 68.3

zero-shot setting, 즉 labeled example이 아예 없는 경우에도 **EM 61.9, F1 68.9**를 기록했다. 이는 few-shot 25-shot 결과와 큰 차이가 나지 않을 정도로 높다. 물론 이 값은 synthetic data와 self-training의 기여가 매우 큰 결과이며, 일반적인 zero-shot semantic parsing과 직접 비교할 때는 설정 차이를 조심해서 봐야 한다.

### 4.4 WebQSP 결과

WebQSP에서는 100-shot FlexKBQA가 **F1 60.6**으로, Pangu의 54.5보다 **6.1 포인트 높다**. supervised 최고 성능 DecAF의 78.8에는 못 미치지만, few-shot setting이라는 점을 고려하면 차이가 상당히 줄어든다.

zero-shot에서는 **F1 46.2**를 기록했다. GrailQA보다 절대 성능은 낮지만, 실제 사용자 질의 성격이 강한 데이터에서 synthetic + self-training 조합이 어느 정도 효과가 있음을 보여준다.

### 4.5 KQA Pro 결과

KQA Pro에서는 100-shot FlexKBQA가 **Accuracy 46.83**을 기록해, few-shot baseline LLM-ICL의 31.75보다 크게 높다. 다만 supervised BART+SPARQL의 89.68과는 매우 큰 격차가 있다.

논문은 그 이유를 명시적으로 설명한다. KQA Pro에는 entity linking stage가 없기 때문에, test set에 등장한 relation이나 entity가 학습 중 충분히 보이지 않으면 모델이 **semantic하게는 맞지만 실행 불가능한 SPARQL**을 만들기 쉽다. 즉, 이 데이터셋은 특히 program executability 측면에서 어렵다. 이 해석은 설득력이 있다.

zero-shot에서는 Accuracy 33.28이다. 절대적으로 높다고 보기 어렵지만, IR과 EGST가 없을 때 성능이 더 크게 떨어지는 것을 보면, FlexKBQA 구성 요소들이 실제로 유효하다는 점은 확인된다.

### 4.6 EGST와 IR의 기여

ablation 결과는 논문의 핵심 주장을 잘 뒷받침한다.

GrailQA에서 25-shot 기준:
- FlexKBQA: F1 69.4
- `-w/o IR`: F1 68.0
- `-w/o EGST`: F1 57.7

즉, **EGST가 11.7 포인트에 가까운 큰 차이**를 만든다. 본문에서는 10.3 포인트 증가로 요약한다. WebQSP에서도 EGST 제거 시 F1이 60.6에서 51.1로 떨어진다. KQA Pro에서는 Accuracy가 46.83에서 23.10으로 크게 감소한다. 따라서 논문 전체에서 가장 중요한 모듈은 EGST라고 보는 것이 타당하다.

IR의 효과는 데이터셋마다 다르다. GrailQA에서는 F1 1.4, WebQSP에서는 2.4 정도 증가로 비교적 작지만, KQA Pro에서는 **13.5 포인트**의 큰 정확도 향상을 보인다. 이는 KQA Pro에서 executable program 생성 실패가 많고, 그 경우 LLM의 direct answer가 강한 보완책이 되기 때문이라고 논문은 해석한다.

### 4.7 Synthetic vs Real Data 분석

Appendix F는 흥미롭다. WebQSP에서 동일한 양의 데이터로 학습했을 때:

- Real Data: EM 61.4 / F1 68.2
- Synthetic Data: EM 32.7 / F1 35.5
- Real Program, Synthetic Question: EM 50.9 / F1 56.3

이 결과는 두 가지를 보여준다. 첫째, synthetic question 자체가 치명적인 문제는 아니다. real program에 synthetic question을 붙인 경우 성능이 꽤 높다. 둘째, 더 큰 문제는 **sampled program distribution이 실제 test set 분포와 다르다**는 점이다. 결국 EGST가 중요한 이유도 여기에 있다. synthetic pair만으로 충분하지 않고, real unlabeled question을 통해 분포를 맞춰야 한다.

### 4.8 Beyond Few-Shot

논문 Figure 3에 따르면 FlexKBQA는 few-shot을 넘어 더 많은 real labeled sample이 주어져도 계속 이득을 준다. 예를 들어 real sample이 1000개일 때도, synthetic pretraining을 한 모델이 real data만으로 학습한 모델보다 **8포인트 높은 성능**을 유지했다고 한다. 즉, 이 방법은 단순한 few-shot 기법이 아니라 **data augmentation strategy**로도 작동할 수 있다는 주장이다.

### 4.9 Case Study

Table 5는 FlexKBQA가 Pangu보다 왜 강한지 직관적으로 보여준다.

첫 번째 예시는 non-i.i.d. relation이 등장한 질문이다. Pangu는 in-context example에 없는 relation 때문에 틀린 relation으로 grounding했지만, FlexKBQA는 synthetic data에 해당 relation이 포함되어 있어 정답 relation을 맞췄다.

두 번째 예시는 EGST의 효과를 보여준다. 실제 사용자 질문에서 pseudo-labeled된 유사 패턴을 학습하면서, 모호하거나 복잡한 test question에도 적절한 relation으로 연결할 수 있게 되었다. 즉, synthetic data는 relation coverage를 넓히고, EGST는 실제 표현 방식에 적응하게 만든다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 설정과 해결 방향이 매우 실용적**이라는 점이다. LLM을 직접 deploy-time parser로 쓰는 대신, 학습 데이터 생성기와 보조 reasoning 모듈로 활용했다. 이 설계 덕분에 context window 제약과 추론 비용 문제를 피하면서도, few-shot KBQA에서 강한 성능을 얻었다. 특히 GrailQA에서 supervised 모델에 근접한 결과를 낸 것은 인상적이다.

또 다른 강점은 **실행 가능성(executability)을 적극 활용한 self-training**이다. 보통 pseudo-label 기반 self-training은 noise가 큰데, 이 논문은 KB 실행 결과, semantic similarity, IR agreement를 이용해 pseudo-label을 걸러낸다. KBQA라는 과제 특성을 잘 이용한 설계다. 단순히 LLM이 생성한 텍스트를 믿는 것이 아니라, structured environment에서 검증 가능한 정보로 필터링한다는 점이 설득력 있다.

세 번째 강점은 **유연성(flexibility)** 에 대한 논문의 주장이 단순한 수사가 아니라 실제 구성에서 드러난다는 점이다. Freebase 기반 GrailQA/WebQSP뿐 아니라 Wikidata 기반 KQA Pro까지 실험했고, S-expression과 SPARQL 모두 다뤘다. 또한 zero-shot까지 포함해 평가했다. 논문이 주장하는 data-efficient, domain-agnostic, deployable이라는 세 가지 flexibility는 실험 설계와 어느 정도 연결된다.

하지만 한계도 분명하다. 첫째, synthetic data의 품질이 충분히 높지 않다. Appendix F에서 synthetic data만으로 학습한 성능은 real data보다 크게 낮다. 즉, 논문의 성공은 synthetic generation 자체만이 아니라 **EGST를 통한 분포 보정**에 크게 의존한다. 다시 말해, unlabeled real question을 충분히 모을 수 없는 환경에서는 성능이 제한될 가능성이 있다.

둘째, template coverage에 대한 의존성이 있다. 논문은 few-shot annotated sample에서 template를 만든다고 설명하지만, 만약 초기 few-shot sample이 구조적으로 편향되어 있으면 생성 가능한 program space도 제한될 수 있다. 실제로 논문도 synthetic data coverage가 충분하면 non-i.i.d. 문제를 줄일 수 있다고 말하지만, coverage를 어떻게 체계적으로 보장할지는 아직 해결되지 않았다.

셋째, KQA Pro 결과는 한계를 잘 드러낸다. entity linking이 없는 setting에서는 unseen relation/entity를 정확히 기억하고 executable SPARQL로 만드는 문제가 훨씬 어려워진다. 이 경우 FlexKBQA는 baseline보다 강하지만 supervised와는 큰 격차가 남아 있다. 즉, 이 프레임워크가 모든 KBQA setting에서 annotation scarcity 문제를 거의 해결했다고 보기는 어렵다.

넷째, IR 사용 방식에는 해석 가능성 측면의 긴장이 있다. 논문은 IR을 보조적으로만 사용한다고 하지만, 실제로 final answer fallback에 IR을 쓰는 순간, 그 답은 KB execution 기반이 아니라 LLM 내부 지식 기반이 된다. 이는 KBQA의 장점인 traceability를 약화시킬 수 있다. 논문도 IR의 interpretability 부족을 인정하고 있다.

다섯째, baseline 비교 폭은 제한적이다. 저자들도 few-shot KBQA 자체가 새로운 주제라 baseline 선택이 제한된다고 언급한다. 따라서 “모든 이전 방법을 크게 앞선다”는 주장은 few-shot KBQA 맥락에서는 맞지만, broader semantic parsing literature 전체와 폭넓게 비교했다고 보기는 어렵다.

## 6. 결론

이 논문은 few-shot KBQA를 위해 **LLM과 lightweight model의 역할을 분리하는 새로운 프레임워크 FlexKBQA**를 제안했다. 핵심은 KB에서 실행 가능한 program을 자동 샘플링하고, 이를 LLM이 자연어 질문으로 번역해 synthetic data를 만든 뒤, execution-guided self-training으로 실제 사용자 질문 분포에 적응시키는 것이다. 여기에 IR을 보조적으로 결합해 self-training 정제와 답변 보완까지 수행한다.

주요 기여는 세 가지로 요약할 수 있다. 첫째, KBQA에서 LLM을 직접 parser로 쓰기보다 **program translator**로 활용하는 실용적 설계를 제안했다. 둘째, **EGST**를 통해 synthetic-real distribution shift를 줄이는 구체적 메커니즘을 제시했다. 셋째, GrailQA, WebQSP, KQA Pro에서 few-shot과 zero-shot 모두 의미 있는 결과를 보이며, 특히 GrailQA에서는 supervised 수준에 상당히 근접했다.

실제 적용 측면에서 이 연구는 annotation이 부족한 도메인 KB, 빠르게 바뀌는 KB, 또는 배포 비용이 민감한 환경에서 특히 중요할 가능성이 있다. 향후 연구로는 더 강한 template coverage 확보, synthetic program distribution 개선, entity linking이 없는 설정에서의 robust grounding, 그리고 IR과 symbolic execution의 결합 방식을 더 정교하게 설계하는 방향이 자연스럽다. 논문이 주장하듯, 이 연구의 가치는 단지 few-shot KBQA 성능 향상 자체보다도, **대형 모델과 경량 모델의 협업 구조를 KBQA에 맞게 설계했다는 점**에 있다.
