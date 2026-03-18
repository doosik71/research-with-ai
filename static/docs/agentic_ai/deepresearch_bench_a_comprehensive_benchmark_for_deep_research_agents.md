# DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents

## 1. Paper Overview

이 논문은 Deep Research Agent(DRA)를 체계적으로 평가하기 위한 전용 벤치마크인 **DeepResearch Bench**를 제안한다. 저자들은 기존 에이전트 벤치마크가 웹 탐색, 정보 검색, 혹은 생성 능력 중 일부만 따로 평가하는 경향이 있어, 실제 사용자가 기대하는 “깊이 있는 조사와 종합 보고서 작성” 능력을 종단간(end-to-end)으로 측정하기 어렵다고 본다. 이를 해결하기 위해 22개 분야에 걸친 100개의 PhD급 연구 과제를 구축하고, 생성된 보고서의 품질을 평가하는 **RACE**와, 인용 및 정보 수집 능력을 평가하는 **FACT**라는 두 개의 평가 프레임워크를 함께 제시한다. 논문의 핵심 문제의식은 “Deep Research Agent의 진짜 성능은 최종 보고서와 그 근거 인용의 품질에서 드러나는데, 이를 사람 판단과 잘 정렬되게 평가할 방법이 없다”는 점이다.

이 문제가 중요한 이유는, Deep Research Agent가 단순 검색 보조를 넘어 다단계 웹 탐색, 정보 통합, 고차원적 분석을 통해 사람의 리서치 업무를 대체하거나 보완하는 방향으로 빠르게 발전하고 있기 때문이다. 따라서 실제 업무와 유사한 난도의 과제, 그리고 보고서 품질과 사실 기반성을 동시에 보는 평가 기준이 필요하다. 저자들은 이런 요구를 반영하기 위해 실제 사용자 질의 96,147개를 분석해 주제 분포를 추정하고, 이를 바탕으로 벤치마크의 주제 구성을 정했다.

## 2. Core Idea

이 논문의 중심 아이디어는 크게 두 가지다.

첫째, **벤치마크 자체를 실제 수요 분포에 맞춰 설계**했다는 점이다. 저자들은 웹 검색 기능이 있는 LLM 챗봇 상호작용으로부터 수집한 96,147개의 실제 질의를 바탕으로, “여러 차례 검색하고, 정보를 수집하고, 분석해 고품질 보고서를 작성해야 하는 과제”를 deep research task로 정의했다. 이후 44,019개의 deep research 질의를 22개 도메인으로 분류하고, 그 분포를 압축해 100개의 평가 과제를 만들었다. 즉, 단순히 연구자가 임의로 만든 문제셋이 아니라 **현실적 사용자 수요의 축소판**을 만들려 했다는 점이 핵심이다.

둘째, **평가를 두 축으로 분리**했다는 점이다.
하나는 최종 보고서의 품질을 보는 **RACE (Reference-based Adaptive Criteria-driven Evaluation with Dynamic Weighting)**, 다른 하나는 보고서가 실제 웹 근거를 얼마나 정확하고 풍부하게 활용했는지를 보는 **FACT (Factual Abundance and Citation Trustworthiness)** 이다. 이 분리는 매우 합리적이다. 어떤 에이전트는 글을 매끄럽게 잘 쓰더라도 근거 인용이 약할 수 있고, 반대로 많은 근거를 모아도 분석과 구조화 능력이 떨어질 수 있기 때문이다. 저자들은 이 두 축을 함께 측정해야 DRA의 실질 능력을 더 잘 파악할 수 있다고 본다.

이 논문의 상대적 신선함은 단순히 “벤치마크를 만들었다”는 데 있지 않다. 보고서 평가에서 흔히 발생하는 문제, 즉 고정된 rubric이나 checklist가 과제별 특수성을 반영하지 못하고, LLM judge가 보고서를 단독으로 보면 점수를 지나치게 높게 주는 경향을 보완하기 위해, **과제별 가중치 생성 + 기준 생성 + reference 기반 상대평가**를 결합했다는 점이 더 중요하다. 또한 FACT는 단순 citation count가 아니라 **실제로 문장과 URL이 지원 관계인지 판단**하여 유효한 인용만 세는 구조라는 점에서 기존의 표면적 인용 수 평가보다 정교하다.

## 3. Detailed Method Explanation

### 3.1 DeepResearch Bench 구축 방식

논문은 먼저 deep research task의 실세계 분포를 파악하기 위해 실제 사용자 질의를 분석한다. 원시 질의 96,147개에서 개인정보를 익명화한 뒤, DeepSeek-V3-0324를 사용해 deep research 정의에 맞는 질의를 필터링했고, 최종적으로 44,019개의 관련 질의를 얻었다. 이후 WebOrganizer의 taxonomy를 바탕으로 22개 주제 도메인으로 분류했다. 이렇게 얻은 주제 분포는 벤치마크 내 각 분야의 문제 수를 결정하는 기준이 된다.

그 다음 단계는 실제 과제 작성이다. 저자들은 각 분야에서 박사급 혹은 5년 이상 경력의 시니어 실무자를 초청해 후보 과제를 작성하게 했고, 이를 수동 검수하여 품질, 명확성, 복잡성, deep research 정의와의 정합성을 확인했다. 이 과정을 거쳐 최종적으로 100개의 고품질 과제를 구성했으며, 언어는 중국어 50개와 영어 50개로 구성했다. 논문 3페이지의 그림 2(a)는 이 전체 파이프라인을 시각화하고 있고, 4페이지의 그림 3은 22개 도메인별 질의 분포를 도넛 차트와 막대 그래프로 보여준다. 여기서 Science & Technology, Finance & Business, Software Development 비중이 큰 편임을 확인할 수 있다.

### 3.2 RACE: 보고서 품질 평가 프레임워크

RACE는 보고서 품질을 평가하는 프레임워크다. 핵심은 세 단계다.

#### 3.2.1 상위 평가 차원 정의

먼저 네 개의 상위 차원을 둔다.

* **Comprehensiveness (COMP)**: 핵심 영역을 빠짐없이 다뤘는가
* **Insight/Depth (DEPTH)**: 원인, 영향, 추세에 대한 깊이 있는 분석이 있는가
* **Instruction-Following (INST)**: 과제 요구를 정확히 따랐는가
* **Readability (READ)**: 구조와 문장이 명확하고 읽기 쉬운가

이 네 차원은 Appendix B에서 정의되어 있으며, 논문은 이를 “직교적(top-level orthogonal)” 평가 차원으로 본다.

#### 3.2.2 Dynamic Weight 생성

같은 보고서라도 과제 특성에 따라 중요한 평가 항목이 다르다. 예를 들어 투자 전략 분석 과제는 Insight 비중이 커야 하고, 폭넓은 시장 맵핑 과제는 Comprehensiveness 비중이 더 커질 수 있다. 그래서 Judge LLM이 과제 $t$에 대해 여러 번 시도한 차원별 가중치 $w_d^{(j)}$를 평균내어 최종 가중치 $W_d$를 만든다.

$$
W_d = \frac{1}{T} \sum_{j=1}^{T} w_d^{(j)}
$$

즉, 한 번의 임의적 판단이 아니라 여러 trial의 평균으로 차원 가중치를 안정화한다. 이후 각 차원 안에서도 과제 특화된 세부 criterion 집합 ${c_{d,k}}$와 그 가중치 ${w_{d,k}}$를 생성한다. 이 설계의 장점은 고정된 rubric보다 **과제 특수성에 더 민감한 평가**가 가능하다는 점이다.

#### 3.2.3 Reference-based Scoring

저자들의 예비 실험에 따르면, 보고서를 독립적으로 단독 평가하면 LLM judge가 대체로 높은 점수를 주어 모델 간 차이가 잘 드러나지 않았다. 이를 보완하기 위해 RACE는 **reference report**를 도입한다. 각 task마다 고품질 참조 보고서 $R_{ref}$를 하나 정하고, 평가 대상 보고서 $R_{tgt}$와 reference를 **같은 criterion 집합**으로 비교 평가한다.

Judge LLM은 모든 criterion에 대해 두 보고서의 점수 리스트를 출력한다.

$$
({s_{tgt,c}}*{c \in C_t}, {s*{ref,c}}*{c \in C_t}) = \text{JudgeLLM}(t, R*{tgt}, R_{ref}, C_t)
$$

그 후 criterion 가중치와 dimension 가중치를 반영해 intermediate score를 계산하고, 최종 점수는 reference 대비 상대 점수로 정한다.

$$
S_{final}(R_{tgt}) = \frac{S_{int}(R_{tgt})}{S_{int}(R_{tgt}) + S_{int}(R_{ref})}
$$

이 식의 의미는 절대점수보다 **상대 품질 비율**을 사용한다는 것이다. 따라서 점수값 자체보다는 모델 간 순위와 차이를 해석하는 것이 더 중요하다고 저자들은 설명한다. 실제로 본문에서도 RACE 점수는 인간 점수와 매우 선형적으로 상관되지만, 스케일은 다를 수 있다고 말한다.

### 3.3 FACT: 인용 및 정보 수집 평가 프레임워크

FACT는 에이전트가 웹에서 얼마나 유효한 정보를 찾아와 보고서에 반영했는지를 본다. 단계는 비교적 직관적이다.

#### 3.3.1 Statement-URL 추출 및 중복 제거

먼저 보고서에서 Judge LLM이 개별 factual statement와 대응 citation URL을 추출한다. 그리고 같은 URL에 연결된 여러 문장이 사실상 동일한 내용을 말하는 경우 하나만 남겨 중복을 제거한다. 이렇게 해서 각 task에 대해 고유한 statement-URL pair 집합 $U_t$를 만든다.

#### 3.3.2 Support Judgment

각 statement-URL pair에 대해, 해당 URL의 웹페이지 텍스트를 Jina Reader API로 가져온 뒤 Judge LLM이 “이 페이지가 이 문장을 실제로 뒷받침하는가”를 이진 분류한다. 결과는 support 또는 not support다. 이 단계가 FACT의 핵심이다. 단순히 citation이 달려 있다고 가산점 주는 것이 아니라, **인용이 실제 문장 근거인지 검증**한다.

#### 3.3.3 Citation Metrics 계산

논문은 두 개의 핵심 지표를 정의한다.

첫째, **Citation Accuracy (C. Acc.)** 는 task별로 support 판정을 받은 비율을 평균한 값이다.

과제 $t$에서 고유 pair 수를 $N_{u,t}$, support된 pair 수를 $N_{s,t}$라 할 때,

$$
Acc_t =
\begin{cases}
\frac{N_{s,t}}{N_{u,t}} & \text{if } N_{u,t} > 0 \
0 & \text{if } N_{u,t} = 0
\end{cases}
$$

전체 Citation Accuracy는

$$
C.Acc. = \frac{1}{|T|}\sum_{t \in T} Acc_t
$$

이다. 즉, citation precision에 가까운 개념이다.

둘째, **Average Effective Citations per Task (E. Cit.)** 는 task당 평균적으로 몇 개의 “실제로 support되는 유효 citation”을 생성했는지를 본다.

$$
E.Cit. = \frac{\sum_{t \in T} N_{s,t}}{|T|}
$$

이 값은 단순 citation 개수가 아니라 **검증된 정보량**을 나타낸다. 따라서 많이 인용하되 틀리는 모델과, 적게 인용하지만 정확한 모델을 구분할 수 있다.

### 3.4 Human Consistency 검증

RACE의 타당성을 보이기 위해 저자들은 50개의 중국어 task에 대해 4개 agent가 생성한 보고서를 수집하고, 70명 이상의 석사급 이상 annotator를 모집해 사람 평가를 수행했다. 각 task마다 3명의 관련 분야 annotator가 네 개 차원과 overall을 평가했다. 각 annotator는 최대 3개의 query만 평가하게 해 편향을 줄였다.

평가 정합성은 다음 네 지표로 본다.

* **Pairwise Agreement Rate (PAR)**: 보고서 쌍 비교에서 자동평가와 사람평가 선호가 얼마나 일치하는가
* **Overall Pearson Correlation (OPC)**: 모델 평균 점수와 사람 평균 점수의 선형 상관
* **Filtered Average Pearson (FAP)**: 사람 간 신뢰도 낮은 task를 제외한 per-task Pearson 평균
* **Filtered Average Spearman (FAS)**: 같은 필터링 후 모델 순위 일치도

특히 필터링에는 ICC(1,1) < 0 인 task를 제거하는 방식이 사용되며, 실험에서는 50개 중 37개 task가 남았다. 이 부분은 평가 자체의 통계적 신뢰성을 고려했다는 점에서 논문 완성도가 높다.

## 4. Experiments and Findings

### 4.1 실험 설정

RACE에서는 citation formatting이 Judge LLM의 점수에 악영향을 줄 수 있어, 먼저 보고서에서 citation 표현을 정리하는 전처리를 한다. RACE의 Judge LLM은 **Gemini-2.5-pro**, FACT의 statement extraction과 support judgment에는 비용 효율을 고려해 **Gemini-2.5-flash**를 사용했다. RACE의 reference report는 2025년 4월 시점의 Gemini-2.5-pro 기반 Deep Research가 생성한 보고서 중에서 선택했다.

평가 대상은 크게 두 그룹이다.

* **Deep Research Agents**

  * Gemini-2.5-Pro Deep Research
  * OpenAI Deep Research
  * Grok Deeper Search
  * Perplexity Deep Research

* **LLMs with Search Tools**

  * Claude 3.7/3.5 Sonnet with Search
  * Perplexity Sonar 계열
  * Gemini grounding 계열
  * GPT-4o/4.1 search 계열 등

그리고 search tool을 가진 LLM들은 가능한 한 공정하게 비교하기 위해 search context size를 high로 통일하는 등 설정을 표준화했다. Appendix G.2에는 thinking budget, 최대 search iteration, 출력 길이, citation formatting 통일까지 상세히 적혀 있다.

### 4.2 메인 결과: RACE

표 1에 따르면, Deep Research Agent 그룹에서 **Gemini-2.5-Pro Deep Research**가 RACE overall 48.88로 가장 높다. 세부적으로 Comprehensiveness 48.53, Depth 48.50, Instruction-Following 49.18, Readability 49.44를 기록했다. **OpenAI Deep Research**는 overall 46.98로 근소하게 뒤따르지만, Instruction-Following에서는 49.27로 Gemini를 약간 앞선다. 이는 차원별 역량이 완전히 결합되어 있지 않음을 보여준다. **Perplexity Deep Research**와 **Grok Deeper Search**는 각각 42.25, 40.24 수준이다.

논문 1페이지 Figure 1의 왼쪽 막대그래프와 6페이지 Table 1은 이런 경향을 시각적으로 잘 보여준다. Gemini와 OpenAI가 전반적으로 상위권이며, 특히 Gemini는 Comprehensiveness와 Depth에서 강하고, OpenAI는 Instruction-Following이 강하다. 저자들은 RACE 점수 절대값보다 순위와 비율 차이를 보는 것이 더 중요하다고 설명한다.

흥미로운 점은 Search Tool LLM 중 **Claude-3.7-Sonnet w/Search**가 overall 40.67로 Grok Deeper Search를 넘는 성능을 보였다는 것이다. 이는 deep research 전용 에이전트가 아니어도 멀티턴 웹 탐색이 허용되면 상당한 수준까지 근접할 수 있음을 시사한다. 또한 Figure 5(7페이지)는 주제별·언어별 heatmap을 통해 모델 성능이 대체로 안정적이지만, 중국어 transportation task는 전 모델에게 상대적으로 어려웠다고 분석한다.

### 4.3 메인 결과: FACT

FACT 결과는 RACE와 다른 면을 드러낸다.
**Gemini-2.5-Pro Deep Research**는 task당 평균 **111.21 effective citations**로 압도적 1위다. 이는 아주 많은 유효 정보 조각을 수집·제시한다는 뜻이며, Comprehensiveness 1위와도 잘 맞아떨어진다. 반면 **Citation Accuracy**는 81.44로, **Perplexity Deep Research의 90.24**보다 낮다. 즉, Gemini는 엄청나게 많이 인용하지만 정확도 면에서는 Perplexity가 더 강하다. **OpenAI Deep Research**는 Citation Accuracy 77.96, E. Cit. 40.79다.

이 결과는 매우 흥미롭다.

* Gemini 계열은 **많이 찾고 많이 싣는 모델**
* Perplexity는 **상대적으로 정확하게 연결하는 모델**
* OpenAI Deep Research는 **보고서 품질은 높지만 citation grounding에서는 다소 보수적 혹은 덜 정밀한 모델**

로 해석할 수 있다. 저자들도 Perplexity Deep Research가 retrieved text에서 관련 내용을 정확히 재호출하는 능력이 강하다고 분석한다.

### 4.4 Human Consistency 결과

표 2에 따르면 **RACE(Full)** 는 PAR 71.33, OPC 99.54, FAP 60.24, FAS 59.12로 가장 강한 전반적 정합성을 보인다. Vanilla Prompt는 overall 60.46에 그치며, reference 제거, static criteria 사용, weight 제거 등 여러 ablation은 모두 RACE full보다 성능이 떨어진다. 특히 **No Reference** 변형의 성능 하락은 reference-based scoring이 실제로 중요하다는 강한 근거다.

더 인상적인 부분은, RACE의 Pairwise Agreement Rate 71.33이 인간 annotator 간 inter-agreement 68.44보다 높다는 점이다. 이는 적어도 “보고서 A와 B 중 무엇이 더 낫나”라는 비교 문제에서, RACE가 사람 평균 판단을 상당히 안정적으로 근사한다는 것을 뜻한다. 물론 이것이 자동평가가 사람보다 “더 옳다”는 의미는 아니지만, practical evaluation tool로 쓰기에는 충분히 설득력 있는 결과다.

표 3에서는 Judge LLM도 비교하는데, **Gemini 2.5 Pro Preview**가 overall 72.56으로 가장 좋고 평균 비용도 $0.13$ 수준이어서 성능-비용 균형이 우수했다. o4-mini는 비용은 더 저렴하지만 성능은 약간 낮았고, Claude 3.7 Sonnet도 성능은 높지만 비용이 비쌌다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **문제 설정이 현실적**이라는 점이다. 벤치마크를 만들 때 실제 사용자 질의 분포를 참고해 주제 비율을 정하고, 박사급 전문가가 과제를 작성했다는 점은 데이터셋 설계의 설득력을 높인다. 단순히 “어려운 문제”를 만든 것이 아니라, “현실에서 많이 필요한 deep research task”를 반영하려 했다는 점이 좋다.

둘째, **평가의 다면성**이 우수하다. 최종 보고서 품질만 보면 factual grounding을 놓치기 쉽고, citation만 보면 분석력과 구조화 능력을 놓치기 쉽다. RACE와 FACT를 함께 사용하면 이 두 축을 분리해 해석할 수 있다. 실제 결과에서도 Gemini와 Perplexity의 장단점이 분명히 다르게 드러난다.

셋째, **human consistency 검증이 상대적으로 탄탄**하다. 많은 자동평가 논문이 “그럴듯한 프롬프트”를 제시하는 데 그치는데, 이 논문은 여러 ablation과 ICC 기반 필터링, pairwise agreement까지 포함해 비교적 정교하게 검증했다. 이는 제안 기법의 실용성을 높인다.

### 한계

논문 자체가 Appendix A에서 인정하듯, **벤치마크 규모가 100 task로 크지 않다**. 저자들은 고품질 task 작성 비용 때문에 규모 확장이 어렵다고 설명한다. 실제로 22개 도메인을 100개 문제로 덮는 것은 대표성 측면에서 한계가 있다. 특히 각 분야 내 세부 하위 task 다양성을 충분히 반영했는지는 더 검증이 필요하다.

또한 **도메인 커버리지 편향** 문제도 있다. 실제 사용자 질의 기반이라고 해도, 질의 수집 출처가 특정 챗봇 사용자군에 편중되어 있을 수 있고, task 작성 과정 역시 저자 네트워크와 초빙 전문가 풀의 편향에서 완전히 자유롭지 않다. 저자들도 외부 리뷰어 확충이 필요하다고 적고 있다.

인간 평가 역시 제한적이다. 50개 중국어 task, 4개 agent, 각 task당 3명 평가자 구조는 validation 목적으론 충분할 수 있지만, 모든 언어와 모든 task에 대한 광범위한 정답성 검증이라고 보기는 어렵다. 논문에 따르면 전체 human evaluation effort는 225 person-hours이고, 전문가 1명당 query 하나 평가에 평균 1.5시간이 걸렸다. 비용상 이해되지만, 통계적 강건성 면에서는 더 확장이 필요하다.

### 비판적 해석

개인적으로 이 논문에서 가장 중요한 공헌은 “Gemini가 더 좋다 / OpenAI가 더 좋다” 같은 leaderboard가 아니라, **deep research라는 사용 시나리오를 평가 가능한 객체로 정식화했다는 점**이다. 특히 RACE의 reference-relative scoring은 long-form generation evaluation 전반에 응용 가능성이 있다. 다만 reference report를 Gemini 계열 보고서에서 가져왔다는 점은 잠재적으로 특정 스타일에 유리한 기준을 만들 수 있다. 저자들이 strong linear correlation with humans를 제시하긴 했지만, reference provenance 자체에 대한 추가 분석이 있으면 더 좋았을 것이다.

또 하나는 FACT의 support judgment가 웹페이지 텍스트와 statement 간의 support만 본다는 점이다. 이것은 매우 유용하지만, citation이 문장 전체의 해석을 충분히 보장하는지, 혹은 partial support와 overclaim를 얼마나 정교하게 구분하는지는 추가 연구 여지가 있다. 즉, FACT는 좋은 시작이지만 “citation faithfulness”의 완전한 해법은 아니다.

## 6. Conclusion

이 논문은 Deep Research Agent 평가의 공백을 메우기 위해 **DeepResearch Bench**, **RACE**, **FACT**를 함께 제안한 작업이다. 벤치마크는 22개 분야, 100개 과제로 구성되며, 실제 사용자 질의 분포를 반영하려 했다. RACE는 과제별 동적 기준과 reference 기반 비교평가를 통해 긴 보고서 품질을 사람 판단과 잘 맞추려 했고, FACT는 statement-URL support 검증을 통해 citation 정확도와 유효 정보량을 측정한다. 실험 결과 Gemini-2.5-Pro Deep Research와 OpenAI Deep Research가 보고서 품질에서 강하고, Perplexity Deep Research는 citation accuracy에서 강점을 보였다. 또한 RACE는 human consistency 측면에서 vanilla prompt나 여러 ablation보다 우수했다.

실무적으로 이 연구는 앞으로 “에이전트가 검색을 잘하느냐”보다 “실제 analyst-grade report를 얼마나 잘 작성하고 근거를 얼마나 정확히 제시하느냐”를 평가하는 기준이 필요하다는 흐름을 강화할 가능성이 크다. 연구적으로도 long-form report evaluation, reference-based LLM-as-a-judge, citation-grounded evaluation의 교차점에서 의미 있는 출발점으로 볼 수 있다.

## 부록 수준 추가 메모

논문 17페이지 Appendix A는 저자들이 직접 인정한 한계를 정리한다. 핵심은 benchmark scale, domain coverage bias, human evaluation throughput이다. 또 18페이지 Appendix C는 FACT용 Judge LLM으로 Gemini-2.5-Flash를 택한 이유를 설명하는데, 무작위 100개 statement-URL pair에 대해 인간 판단과 support 96%, not support 92% 수준의 일치를 보였고, Gemini-2.5-Pro와 정확도 차이가 크지 않으면서 비용이 낮았다고 한다. 18페이지 Appendix D는 상용 모델 결과 수집 기간을 정리하는데, OpenAI Deep Research는 2025년 4월 1일~5월 8일, Gemini 2.5 Pro Deep Research는 4월 27일~29일 등으로 다소 차이가 있다. 따라서 leaderboard 해석 시점에 약간의 버전 차이 가능성은 염두에 둘 필요가 있다.
