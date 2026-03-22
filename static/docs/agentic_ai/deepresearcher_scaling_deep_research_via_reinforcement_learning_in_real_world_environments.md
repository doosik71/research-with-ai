# DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments

이 논문은 **웹 검색이 가능한 LLM research agent를 실제 웹 환경에서 end-to-end RL로 훈련**시키는 최초의 종합적 프레임워크를 제안한다는 점을 핵심 기여로 내세운다. 저자들의 문제의식은 분명하다. 기존 deep research 계열 시스템은 대개 사람이 짜 놓은 프롬프트 워크플로에 의존하거나, RL을 쓰더라도 정적인 local corpus 위의 RAG 환경에서만 학습되기 때문에, 실제 웹의 **잡음, 비정형성, 검색 품질 변동성, 정보 부재 가능성**을 제대로 다루지 못한다. DeepResearcher는 이 한계를 넘기 위해, agent가 실제 검색엔진과 웹페이지를 직접 다루는 환경에서 RL을 수행하도록 설계되었다.  

## 1. Paper Overview

이 논문이 다루는 문제는 “deep research를 잘하는 agent를 어떻게 만들 것인가”가 아니라, 더 정확히 말하면 **현실적인 웹 환경에서 견고하게 작동하는 deep research agent를 어떻게 학습시킬 것인가**다. 기존 prompt-engineering 기반 agent는 사람이 미리 설계한 단계에 따라 움직이므로 적응성이 약하고, 새로운 과제나 복잡한 multi-step research 상황에서 brittle한 성향을 보인다고 저자들은 비판한다. 반면 기존 RL 기반 search agent들도 주로 Wikipedia나 고정된 텍스트 저장소 같은 **local RAG environment**에 갇혀 있어서, 실제 웹 검색이 가지는 불확실성과 복잡성을 충분히 반영하지 못한다고 본다.

이 문제가 중요한 이유는 실제 deep research가 결코 “정답이 고정된 corpus 안에 있다”는 가정을 따르지 않기 때문이다. 현실의 웹에서는 정보가 없을 수도 있고, 오래되었을 수도 있고, 여러 출처를 종합해야만 답이 나올 수도 있다. 따라서 검색 agent를 정적 retrieval 환경에서만 훈련시키면, 실제 응용에서 취약해질 가능성이 높다. 저자들은 이 점을 논문의 중심 메시지로 밀고 있으며, **real-world web environment에서의 end-to-end RL이 단순 구현 선택이 아니라 핵심 요구조건**이라고 주장한다.

## 2. Core Idea

이 논문의 핵심 아이디어는 세 가지로 요약할 수 있다.

첫째, **실제 웹 검색 환경을 RL 학습 환경으로 사용**한다는 점이다. 기존 RAG 기반 RL 연구는 미리 구축된 corpus에 대해 retriever가 문서를 찾고 LLM이 답하는 구조였지만, DeepResearcher는 agent가 실제 검색 API를 호출하고, 결과 URL을 선택하고, 웹페이지를 탐색하며 정보를 모은다. 즉, retrieval이 아니라 **real-world search interaction 자체**를 학습 대상에 포함시켰다.

둘째, **multi-agent browsing architecture**를 둔다는 점이다. 메인 연구 agent가 질문에 대해 생각하고 검색·브라우징 tool을 호출하면, 별도의 browsing agent가 웹페이지의 segment를 순차적으로 읽으면서 relevant information을 short-term memory에 축적한다. 이는 단순 snippet retrieval이나 fixed-passage RAG와 다르다. 논문은 이를 통해 긴 웹페이지와 다양한 구조의 문서를 더 안정적으로 다룰 수 있다고 본다.

셋째, RL이 단순 성능 향상뿐 아니라 **planning, cross-validation, reflection, honesty** 같은 행동을 자연스럽게 emergence하게 만든다고 주장한다. 저자들은 explicit planning SFT나 rule-based workflow 없이도 이런 behavior가 나타난다고 보고하며, 이것을 real-world end-to-end RL의 중요한 부산물로 해석한다.

## 3. Detailed Method Explanation

### 3.1 Deep Research trajectory

논문은 DeepResearcher의 trajectory를 iterative reasoning-and-tool-use 과정으로 정의한다. 에이전트는 질문과 현재 observation을 바탕으로 먼저 reasoning을 수행하고, 이후 search 또는 browse tool을 선택한다. reasoning은 `<think>` 태그 안에서 수행되도록 제한되며, 이는 DeepSeek-R1 스타일의 explicit reasoning format을 따른다. 즉, “생각 없이 바로 tool 호출”이 아니라, **생각하고 행동하는 루프**를 구조적으로 강제한다.

### 3.2 Web search tool과 browsing agent

search tool은 JSON 형식으로 `web_search`를 호출하고, 검색 결과는 title, URL, snippet 형태의 구조화된 결과로 반환된다. 현재 구현은 top-k를 고정값으로 두지만, 논문은 향후 동적 search parameter 최적화를 탐색할 수 있다고 언급한다.

그 다음 핵심이 browsing agent다. browsing agent는 `web_browse` 요청을 받으면 URL의 첫 segment부터 순차적으로 읽고, query와 기존 memory, 새로 읽은 내용을 보고 두 가지를 판단한다. 하나는 **계속 읽을지 멈출지**, 다른 하나는 **어떤 내용을 short-term memory에 추가할지**다. 이는 사람이 긴 웹페이지를 앞에서부터 훑어보며 “이 페이지가 쓸모 있는가”를 빠르게 판단하는 방식과 유사하다. 논문은 특히 초기 segment가 대부분 무관하면 그 페이지 전체가 비생산적일 가능성이 높다고 가정하고, 이 heuristic으로 browsing 효율을 높인다.

### 3.3 Real-world environment에서의 구현 난제

논문은 실제 웹 환경에서 RL을 돌릴 때 생기는 공학적 난제를 꽤 전면에 내세운다.

첫째, **대규모 동시성 문제**다. GRPO rollout 때문에 한 번에 수천 건의 search/crawl 요청이 발생할 수 있어 지연이 커진다. 이를 해결하기 위해 저자들은 **50-node distributed CPU cluster**를 구축해 tool request를 분산 처리했다고 설명한다.

둘째, **rate limit, anti-crawling, network latency** 문제다. 검색 API나 웹서버는 요청 제한이나 크롤링 방어를 걸 수 있고, 실패 응답이나 무의미한 페이지가 반환될 수도 있다. 이를 위해 retry mechanism과 search result caching을 구현했으며, 동일한 query가 일정 기간 안에 다시 들어오면 캐시를 재사용하게 했다.

셋째, **웹페이지 구조 다양성** 문제다. 같은 “문서 검색”이라도 실제 웹은 페이지 길이, 포맷, 텍스트 밀도, 잡음이 제각각이므로, fixed-passage retrieval보다 더 복잡한 reading policy가 필요하다. browsing agent는 이 문제를 해결하기 위한 전용 component로 설계되었다.

### 3.4 RL training framework: GRPO

학습 알고리즘으로는 **GRPO (Group Relative Policy Optimization)**를 사용한다. 각 입력 질문 $x$에 대해 여러 rollout을 생성하고, 별도의 critic을 학습하는 대신 group 내 rollout들을 이용해 baseline을 추정한다. 목적함수는 PPO류의 clipped ratio term과 reference policy에 대한 KL penalty를 포함한다. 논문 본문에 제시된 형태는 대략 다음과 같다.

$$
\mathcal{J}(\theta)
===================

\mathbb{E}_{x \sim \mathcal{D}, {y_i}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(\cdot \mid x)}
\frac{1}{G}\sum_{i=1}^{G}
\left[
\min\left(
\frac{\pi_{\theta}(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)} A_i,
\operatorname{clip}\left(
\frac{\pi_{\theta}(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)},
1-\epsilon, 1+\epsilon
\right) A_i
\right)
-------

\beta D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\theta_{\text{ref}}})
\right]
$$

즉, 최근 reasoning-model RL에서 널리 쓰이는 policy optimization 계열 기법을 real web search agent에 이식한 것이다.

또한 tool output은 model이 생성해야 하는 정답이 아니라 observation이기 때문에, 논문은 **observation masking**을 적용해 tool 응답 텍스트가 직접 학습 대상이 되지 않도록 했다. 오직 model response만 loss에 기여하게 한다는 점이 중요하다.

### 3.5 Reward 설계

reward는 surprisingly simple하다. 논문은 open-domain QA 데이터셋을 사용하므로, 기본 reward를 **word-level F1**로 둔다. 다만 출력 format이 틀리면 보상은 즉시 **-1 penalty**가 된다. 따라서 보상 규칙은 다음과 같다.

$$
\text{reward} =
\begin{cases}
-1 & \text{if format is incorrect} \
\text{F1 score} & \text{if format is correct}
\end{cases}
$$

흥미로운 점은 긴 설명문 품질을 직접 보상하지 않고, 비교적 단순한 short-answer QA reward를 사용했다는 점이다. 이는 학습 안정성에는 유리하지만, 논문이 말하는 “deep research”의 풍부한 장기 보고서 품질을 직접 최적화하는 것은 아니라는 한계도 내포한다. 저자들도 미래에는 long-form reward가 더 필요할 수 있다고 인정한다.

### 3.6 Training data와 contamination control

논문은 training data 설계에도 상당히 공을 들인다. Deep research 전용 공개 학습셋이 없기 때문에, 저자들은 **NaturalQuestions, TriviaQA, HotpotQA, 2WikiMultiHopQA**를 조합해 학습 corpus를 만든다. 이 중 multi-hop 데이터 비율을 높게 잡아, 최종적으로 **80,000개 예제, 비율 1:1:3:3 (NQ:TQ:HotpotQA:2Wiki)**의 셋을 구성한다. 즉, 전체의 75%를 multi-hop 문제로 채워 복합 탐색 행동을 유도하려는 설계다.

여기서 가장 중요한 포인트는 **contamination detection**이다. base model이 검색 없이도 답을 맞히는 질문은 search tool 사용 학습에 도움이 안 되므로 제거한다. 이를 위해 각 질문마다 base model 응답을 10개 샘플링하고(pass@10), 이 중 하나라도 정답을 포함하면 training set에서 제외한다. 이 설계는 논문의 목적이 “모델의 내장 기억으로 정답 맞히기”가 아니라 **실제로 search behavior를 배우게 만드는 것**이라는 점을 잘 보여준다.

## 4. Experiments and Findings

### 4.1 실험 설정

backbone은 **Qwen2.5-7B-Instruct**다. 매 step마다 256개의 prompt를 샘플링하고, 각 prompt마다 16 rollout을 생성한다. 각 rollout은 최대 10번의 tool call 뒤에 final answer를 내도록 구성된다. mini-batch size는 4,096이다. 즉, 생각보다 꽤 큰 rollout budget을 쓰는 RL scaling 실험이다.

평가는 in-domain 4종(NQ, TQ, HotpotQA, 2Wiki)과 OOD 3종(MuSiQue, Bamboogle, PopQA)으로 구성된다. 각 dev set에서 512개씩, Bamboogle은 125개 전체를 사용한다. metric은 두 가지다. 하나는 **F1**, 다른 하나는 **MBE (model-based evaluation)**로, GPT-4o-mini를 judge로 써 정답 여부를 평가한다. 논문은 long-form 응답에는 F1만으로 부족하므로 MBE가 더 믿을 만하다고 본다.

### 4.2 In-domain 결과

Table 1에 따르면 DeepResearcher는 **4개 in-domain 데이터셋 모두에서 MBE 기준 최고 성능**을 기록한다. 구체적으로 NQ에서 F1 39.6 / MBE 61.9, TQ에서 78.4 / 85.0, HotpotQA에서 52.8 / 64.3, 2Wiki에서 59.7 / 66.6을 달성한다. 저자들은 특히 **TQ와 2Wiki에서 큰 우위**를 강조한다.

흥미로운 비교는 Search-r1-base다. NQ와 HotpotQA 일부에서는 MBE가 꽤 경쟁적이지만, 이 모델은 local RAG 환경에서 관련 Wikipedia corpus에 직접 접근하며 훈련·평가된 모델이다. 반면 DeepResearcher는 훨씬 어려운 설정인 **전체 웹 탐색** 환경에서 이 성능을 냈다는 점이 논문이 강조하는 포인트다. 즉, 절대 점수만이 아니라 **환경 난이도 차이**를 함께 읽어야 한다.

### 4.3 OOD 결과와 real-world environment의 의미

Table 2는 DeepResearcher의 메시지를 더 강하게 보여준다. MuSiQue, Bamboogle, PopQA 세 OOD benchmark에서 DeepResearcher는 각각 **27.1/29.3, 71.0/72.8, 48.5/52.7 (F1/MBE)**를 기록해 모든 baseline을 앞선다. 논문은 이를 통해 모델이 단순히 training distribution에 overfit된 것이 아니라, **reasoning, searching, synthesis skill 자체를 일반화**했다고 해석한다.

특히 **Bamboogle**가 중요하다. 이 데이터셋은 Wikipedia 바깥 지식을 요구하므로 local RAG 기반 학습의 한계를 드러내기 쉽다. 논문은 Bamboogle에서 DeepResearcher가 local RAG 기반 방법뿐 아니라, R1-Searcher에 실제 웹 검색을 허용한 경우보다도 더 잘한다는 점을 들어, **훈련 단계에서부터 real-world environment를 경험하는 것 자체가 중요하다**고 주장한다. 즉, inference 때 웹을 쓸 수 있게 해주는 것만으로는 부족하고, **웹 환경에서 RL로 policy를 길러야 한다**는 것이다.

### 4.4 Training dynamics

분석 섹션도 흥미롭다. Figure 4(a)에 따르면 F1은 training step이 진행되며 **0.375에서 약 0.55까지 꾸준히 상승**한다. 이는 RL scaling이 단순한 초기 bump가 아니라 지속적 성능 향상을 가져온다는 근거로 제시된다.

또한 harder question일수록 tool call 수가 더 늘어나며, 특히 **4-hop 문제는 34 step 이후에도 계속 tool call 수가 증가**한다. 이는 모델이 어려운 문제일수록 더 많은 정보 수집이 필요하다는 사실을 학습하고 있음을 시사한다. response length 역시 reasoning complexity에 따라 증가하며, saturation 없이 계속 늘어나는 경향을 보인다. 저자들은 이를 통해 모델이 double-check, refinement, planning 같은 더 긴 reasoning behavior를 점차 습득한다고 해석한다.

### 4.5 Emergent behavior 분석

논문은 RL 이후 나타나는 네 가지 behavior를 강조한다.

첫째, **planning**이다. multi-hop 문제를 풀 때 단계별 계획을 세우고, 필요하면 중간에 step을 합치거나 수정한다. 저자들은 explicit planning SFT 없이도 이런 능력이 emergence했다고 본다.

둘째, **cross-validation**이다. 첫 tool call에서 이미 정답 후보를 찾았더라도, 곧바로 답하지 않고 추가 검색으로 확인한다. 이는 answer reliability를 높이는 행동으로 해석된다.

셋째, **reflection**이다. 현재 검색 결과가 질문과 잘 맞지 않는다고 판단하면 query를 수정해 다시 탐색한다. 즉, 실패한 search trajectory에서 빠져나오는 self-correction behavior가 나타난다.

넷째, **honesty**다. 적절한 답을 찾지 못했을 때 억지로 답을 꾸며내기보다, 찾지 못했다고 인정하는 경향이 관찰된다고 저자들은 말한다. 다만 현재 QA 평가 지표는 이런 honesty를 충분히 반영하지 못한다고 지적한다.

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **문제를 정확히 재정의했다는 점**이다. 기존 search-RL 연구가 retrieval quality나 prompt workflow 개선에 머물렀다면, 이 논문은 “진짜 deep research agent를 만들려면 진짜 웹에서 학습해야 한다”는 강한 명제를 던진다. 이는 단지 engineering claim이 아니라, benchmark 결과와 behavior analysis로까지 연결되어 있어 설득력이 있다.

두 번째 강점은 **학습 데이터 contamination 통제**다. search behavior를 배우는 척하면서 사실은 pretrained memory를 쓰는 문제를 pass@10 필터링으로 통제한 점은 매우 중요하다. 이 설계 덕분에 논문의 실험은 “정말 검색을 배웠는가”에 대해 조금 더 믿을 만한 답을 준다.

세 번째 강점은 **행동 분석의 질**이다. planning, cross-validation, reflection, honesty를 단순 수사로 제시한 것이 아니라, case study와 training dynamics를 통해 RL이 어떤 종류의 search policy를 유도하는지 보여주려 했다. 이는 단순 benchmark paper보다 해석 가치가 높다.

### 한계

첫 번째 한계는 reward mismatch다. 시스템은 “deep research”를 표방하지만, 실제 학습 reward는 **짧은 정답형 QA용 F1**에 크게 의존한다. format penalty와 F1만으로는 장문의 리서치 보고서 품질, 출처 신뢰성, 논리적 구성, 설명력 같은 deep research의 핵심 속성을 직접 최적화하기 어렵다. 저자들 스스로도 long-form reward의 필요성을 인정한다.

두 번째 한계는 공학 비용이다. 50-node cluster, caching, retry, anti-crawling 대응, 대량 rollout 등은 매우 높은 시스템 복잡도를 의미한다. 연구로서는 인상적이지만, 많은 연구팀이 그대로 재현하기는 쉽지 않다. 즉, “open-source”라는 점과 별개로 **재현 비용**은 상당히 높아 보인다. 이 평가는 논문이 기술한 시스템 구성 자체에서 직접 추론할 수 있다.

세 번째 한계는 평가 설정이다. MBE는 F1보다 낫지만, GPT-4o-mini judge에 의존하는 **model-based evaluation** 역시 완전한 진실 판정은 아니다. 또한 honesty behavior처럼 실제로 중요한 품질이 현행 metric에 잘 반영되지 않는다는 점도 논문이 스스로 지적한다.

### 해석

비판적으로 보면, 이 논문의 가장 본질적인 기여는 “search tool을 붙인 reasoning model”을 만든 데 있는 것이 아니라, **환경 realism이 학습 결과를 바꾼다**는 점을 실증적으로 밀어붙인 데 있다. 즉, RAG 환경에서 학습한 agent에게 inference 시 웹을 허용하는 것과, 애초에 웹에서 policy를 학습한 것은 다르다는 것이다. 오늘날 agent 연구에서 environment design이 얼마나 중요한지를 보여주는 사례로 읽을 수 있다.

## 6. Conclusion

이 논문은 DeepResearcher를 통해 **real-world web environment에서의 end-to-end RL이 deep research agent 학습에 핵심적**이라고 주장한다. 구체적으로는 실제 웹 검색과 브라우징을 RL trajectory에 포함시키고, multi-agent browsing 구조와 GRPO 학습, contamination-controlled QA 데이터를 결합해, 기존 prompt-based 방법과 local-RAG RL 방법을 모두 능가하는 성능을 보였다고 보고한다. 또한 planning, cross-validation, reflection, honesty 같은 behavior가 자연스럽게 emergence한다는 점을 사례 분석으로 보여준다.

실무적으로는 “웹을 붙인 LLM”과 “웹에서 훈련된 LLM agent” 사이의 차이를 분명히 보여준 논문으로 볼 수 있다. 연구적으로는 search agent RL의 진짜 병목이 reward만이 아니라 **environment fidelity**에 있다는 점을 강하게 제기한 작업이다. 앞으로 long-form reward, 더 정교한 evaluation, 비용 효율적 training이 보완된다면 이 계열 연구의 기반 논문으로 오래 인용될 가능성이 크다.
