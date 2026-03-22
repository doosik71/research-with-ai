# Toolformer: Language Models Can Teach Themselves to Use Tools

## 1. Paper Overview

이 논문은 대규모 언어 모델(Language Model, LM)이 단순히 텍스트를 생성하는 데 그치지 않고, **외부 도구를 언제, 어떻게 호출할지 스스로 학습**할 수 있는가를 다룬다. 저자들은 기존 LM이 few-shot/zero-shot 일반화 능력은 뛰어나지만, 산술 계산, 최신 정보 접근, 사실 조회, 시간 개념 처리 같은 기본 기능에서는 오히려 훨씬 단순한 전용 시스템보다 약하다는 문제를 출발점으로 삼는다. 이를 해결하기 위해 제안된 것이 **Toolformer**이며, 이 모델은 계산기, 질의응답 시스템, 검색 엔진, 번역기, 캘린더 같은 외부 API를 텍스트 생성 과정에 통합해 사용한다. 핵심은 이런 도구 사용 능력을 대규모 인간 주석 없이, **소수의 시연 예시만으로 self-supervised 방식으로 학습**한다는 점이다. 논문은 Toolformer가 다양한 다운스트림 과제에서 zero-shot 성능을 크게 개선하며, 더 큰 GPT-3 계열 모델과 경쟁 가능한 수준에 도달한다고 주장한다.

이 연구가 중요한 이유는, 언어 모델의 한계를 “모델 크기 증가”만으로 해결하려는 접근에서 벗어나, **외부 연산 및 지식 시스템과의 결합**을 LM 자체의 행동으로 학습하게 만든다는 점에 있다. 즉, Toolformer는 단순한 prompt engineering이나 hand-crafted tool pipeline이 아니라, LM이 자기 예측 성능을 높이는 방향으로 도구 호출 패턴을 선택하게 한다. 이는 이후의 agent, function calling, tool-use LLM 연구 흐름의 매우 초기이면서도 상징적인 선행 작업으로 볼 수 있다.

## 2. Core Idea

이 논문의 중심 아이디어는 다음 한 문장으로 요약할 수 있다.
**“언어 모델이 미래 토큰 예측에 실제로 도움이 되는 API 호출만 남기도록 self-supervised filtering을 적용하면, 도구 사용 능력을 데이터 주석 없이도 학습시킬 수 있다.”**

좀 더 풀어 설명하면, 저자들은 각 도구(API)에 대해 몇 개의 인간 작성 예시만 준비한 뒤, 원래의 언어 모델이 대규모 텍스트 코퍼스에 대해 **“이 위치에서 어떤 API를 부르면 도움이 될까?”** 를 in-context learning으로 스스로 제안하게 한다. 그다음 실제 API를 실행해 결과를 얻고, 이 결과를 문장에 삽입했을 때 **다음 토큰 예측 loss가 감소하는지**를 계산한다. loss를 줄이는 호출만 남겨 finetuning 데이터로 사용하면, 모델은 점차 어떤 상황에서 어떤 도구를 호출해야 하는지를 학습한다.

기존 접근 대비 차별점은 크게 세 가지다.

첫째, **대규모 human annotation이 필요 없다.**
둘째, **특정 task 전용 파이프라인이 아니라 일반 언어 모델 위에 얹히는 형태**다.
셋째, **도구 사용 여부 자체를 모델이 결정**한다. 즉, 외부 시스템이 강제로 tool call을 삽입하는 것이 아니라, 모델이 문맥상 필요할 때 직접 호출하는 방향으로 학습된다.

이 논문은 오늘날 function calling이나 browsing agent처럼 상호작용적 구조까지는 가지 않지만, “도구 사용을 LM 내부의 행동 선택 문제로 본다”는 점에서 매우 선구적이다. 저자들도 Toolformer가 여러 도구를 제어하고, 어느 시점에 어떤 도구를 쓸지를 스스로 정할 수 있게 된다고 설명한다.

## 3. Detailed Method Explanation

### 3.1 문제 설정과 API 표현

논문은 각 API 호출을 텍스트 시퀀스로 선형화한다. 하나의 API 호출은
$c = (a_c, i_c)$
형태로 표현되며, 여기서 $a_c$는 API 이름, $i_c$는 입력 인자다. 실행 결과를 $r$라고 하면, 호출은 다음과 같은 텍스트 형태로 문장에 삽입된다.

호출만 있는 형태:
$$
e(c) = \texttt{<API>} , a_c(i_c) , \texttt{</API>}
$$

호출과 결과가 포함된 형태:
$$
e(c, r) = \texttt{<API>} , a_c(i_c) \rightarrow r , \texttt{</API>}
$$

실제 구현에서는 별도 vocabulary 수정 없이 `[` `]` `->` 같은 토큰 시퀀스를 사용했다고 설명한다. 이 표현 방식의 장점은 도구 호출을 **일반 텍스트의 일부처럼 취급할 수 있다**는 점이다. 즉, LM은 텍스트 생성 중간에 자연스럽게 API call 토큰을 생성하고, 결과도 문맥 안에 통합할 수 있다.

### 3.2 전체 파이프라인

논문의 핵심 파이프라인은 크게 세 단계다.

#### (1) Sampling API Calls

원본 텍스트 데이터셋 $\mathcal{C}$ 에 대해, 각 API마다 few-shot prompt를 구성한다. 이 prompt는 모델에게 “어떤 위치에서 어떤 API를 어떻게 호출하면 좋을지”를 제안하게 만든다. 모델은 각 위치 $i$에 대해 API 시작 토큰이 나올 확률을 계산하고, 확률이 높은 위치를 후보로 선택한다. 이어서 그 위치마다 여러 개의 API 호출 후보를 샘플링한다. 즉, 모델이 먼저 **도구 사용 후보를 대량 생성**한다.

이 단계에서 중요한 점은, 모델이 인간이 직접 표시한 정답 호출을 학습하는 것이 아니라, **기존 LM의 in-context learning 능력을 이용해 자기 스스로 pseudo-label을 만든다**는 것이다. 이것이 self-supervised 데이터 생성의 출발점이다.

#### (2) Execute and Filter by Loss Reduction

생성된 후보 API 호출은 실제로 실행된다. 예를 들어 계산기면 계산 결과를, 검색이면 검색 결과를, 번역이면 번역문을 얻는다. 그런 뒤 저자들은 이 결과를 문맥에 삽입했을 때 **미래 토큰 예측 loss가 줄어드는지**를 평가한다. 논문 개요 그림 설명에 따르면, 각 위치에서 여러 후보 호출 $c_i^1, c_i^2, \dots, c_i^k$ 를 생성한 뒤, 실행 결과를 반영해 **다음 토큰 예측 손실 $L_i$를 감소시키는 호출만 유지**한다.

이 부분이 Toolformer의 가장 중요한 기계학습적 아이디어다.
도구 호출의 유용성을 “사람이 보기 좋은 호출인가”가 아니라,
**“LM의 언어 모델링 목적함수에 실제로 도움이 되는가”**
로 정의했기 때문이다.

즉, 어떤 API 호출이 남는 조건은:

* 해당 위치에서 문맥상 관련이 있어야 하고
* API 결과가 이후 텍스트를 예측하는 데 실제로 도움을 줘야 하며
* 그렇지 않으면 제거된다

이로써 Toolformer는 인간 선호가 아니라 **모델 자신에게 유익한 도구 사용 패턴**을 학습한다. 논문 서론에서도 “인간이 유용하다고 생각하는 것과 모델이 유용하다고 생각하는 것은 다를 수 있다”고 명시한다.

#### (3) Finetuning on Augmented Corpus

필터링 후 살아남은 API 호출들을 원본 코퍼스에 삽입해 증강 데이터셋 $\mathcal{C}^*$ 를 만든다. 그 후 언어 모델 $M$을 이 데이터셋에 대해 finetune한다. 저자들은 이 방식이 원래 pretraining에 사용한 것과 동일한 형태의 데이터셋 위에서 동작하므로, 모델의 **일반 언어 모델링 능력을 유지**하는 데 유리하다고 설명한다.

즉, Toolformer는 별도의 모듈형 planner를 붙이는 방식이 아니라,
**언어 모델 자체를 “도구 호출 토큰까지 포함한 다음 토큰 예측기”로 재학습**하는 접근이다.

### 3.3 사용된 도구들

논문 초록과 예시에서 Toolformer는 다음 도구들을 포함한다.

* calculator
* question answering system
* search engine
* translation system
* calendar

Figure 1 설명에 따르면, Toolformer는 텍스트를 완성하는 과정에서 위에서부터 순서대로 질문응답 시스템, 계산기, 기계번역 시스템, 위키피디아 검색 엔진 등을 자율적으로 호출한다. 즉, 단일 도구만 배우는 것이 아니라 **서로 다른 성질의 도구를 하나의 LM 안에 통합**했다는 점도 중요하다.

### 3.4 왜 이 방법이 효과적인가

이 방법이 잘 작동하는 이유는 크게 두 가지로 해석할 수 있다.

첫째, **API 결과를 텍스트 예측 문제로 환원**했기 때문이다.
LM은 외부 도구를 별도 symbolic module처럼 다루는 것이 아니라, 결과 텍스트를 받아 이어지는 문맥 예측 정확도를 높이는 방식으로 배운다.

둘째, **유용한 호출만 남기는 loss-based filtering** 때문이다.
만약 모든 호출을 그대로 학습하면 noisy supervision이 너무 많아질 것이다. 하지만 Toolformer는 실제로 예측에 기여하는 호출만 선별하므로, 학습 데이터 품질을 자동으로 통제한다. 논문 말미에서도 Toolformer가 “perplexity-based filtering step 이후 자기 자신의 예측에 대해 학습된다”고 설명한다.

## 4. Experiments and Findings

### 4.1 실험 설정

Toolformer는 **GPT-J 기반 6.7B 파라미터 모델**을 바탕으로 한다. 저자들은 이 모델에 도구 사용 능력을 추가한 뒤, 다양한 downstream task에서 zero-shot 성능을 평가한다. 서론은 Toolformer가 더 큰 GPT-3와 여러 baseline을 다양한 과제에서 능가한다고 요약한다.

논문의 실험이 특히 인상적인 이유는, Toolformer가 “도구를 학습했다”는 사실만 보이는 데 그치지 않고, 각 태스크에서 실제로 적절한 도구를 사용하는지를 함께 보여준다는 점이다. 예를 들어 question answering에서는 검색 도구, multilingual QA에서는 번역 도구, arithmetic 류에서는 계산기가 자연스럽게 대응된다.

### 4.2 Question Answering

표 5 설명에 따르면, 여러 질의응답 데이터셋에서 Toolformer는 **Wikipedia search tool**을 주로 사용하며, 같은 크기의 baseline보다 분명히 좋은 성능을 보인다. 다만 GPT-3 175B보다는 낮다고 저자들은 명시한다. 즉, Toolformer의 핵심 메시지는 “항상 가장 큰 모델을 이긴다”가 아니라, **상대적으로 작은 모델이 적절한 도구 사용을 통해 훨씬 큰 모델에 접근하거나 일부 조건에서 경쟁 가능해진다**는 것이다.

이 결과는 Toolformer의 철학과 잘 맞는다. 단순히 파라미터 수를 늘리는 대신, 필요한 외부 지식을 검색으로 보완하면 더 효율적으로 성능을 끌어올릴 수 있다는 점을 실험적으로 보여준다.

### 4.3 Multilingual Question Answering

MLQA 실험에서는 영어 문맥과 다양한 언어의 질문을 함께 다뤄야 하므로, 번역 도구 사용 능력이 중요하다. 논문은 Toolformer가 **모든 언어에서 API 호출을 사용하면 성능이 일관되게 향상**되며, 이는 모델이 machine translation tool을 실제로 활용했음을 시사한다고 말한다. 언어에 따라 번역 도구 사용 비율은 63.8%~94.9%에 달했고, Hindi만 예외적으로 7.3%에 그쳤다.

다만 여기서도 저자들은 솔직하게 한계를 언급한다. Toolformer가 항상 vanilla GPT-J를 능가한 것은 아니며, 일부 언어에서는 CCNet finetuning이 성능을 악화시켰을 수 있다고 분석한다. 즉, 도구 사용 학습이 항상 순수 LM 성능 개선으로 직결되는 것은 아니고, **학습 데이터 분포 이동(distribution shift)** 같은 문제가 개입할 수 있다.

### 4.4 종합 해석

실험 전반의 메시지는 다음과 같다.

* Toolformer는 적절한 도구 사용을 학습한다.
* 이로 인해 zero-shot 성능이 여러 태스크에서 좋아진다.
* 특히 작은/중간 규모 모델이 도구 사용을 통해 큰 모델과 경쟁력을 확보할 수 있다.
* 하지만 모든 태스크와 모든 언어에서 무조건 우세한 것은 아니며, 데이터와 도구 특성에 따라 편차가 있다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **도구 사용 학습을 self-supervised objective에 직접 연결**했다는 점이다. 이후의 많은 연구가 tool use를 agent orchestration이나 instruction tuning 관점에서 확장했지만, Toolformer는 비교적 단순한 구조로 “도구 호출이 예측에 도움이 되는가”라는 기준을 세웠다. 이 기준은 매우 자연스럽고, 언어 모델의 본래 목적함수와 정합적이다.

둘째, **일반성 유지**를 중요한 설계 목표로 둔 점이 좋다. 논문은 원래 pretraining 데이터셋과 동일한 형태의 코퍼스를 증강해 사용하므로, 모델이 본래의 general language modeling ability를 잃지 않도록 한다고 강조한다. 이는 task-specific tool pipeline과 구분되는 장점이다.

셋째, **다양한 도구를 하나의 프레임워크에 통합**했다는 점이다. 계산기, QA, 검색, 번역, 캘린더처럼 성질이 다른 도구들을 동일한 linearized API 형식으로 다루는 것은 이후 function-calling paradigm의 전조처럼 보인다.

### 한계

논문 7장 Limitations에서 저자들은 몇 가지 분명한 한계를 인정한다.

첫째, Toolformer는 **tool chaining**을 하지 못한다. 즉, 한 도구의 출력을 다른 도구의 입력으로 연쇄적으로 쓰는 복합 도구 사용이 불가능하다. 그 이유는 각 도구의 API 호출이 독립적으로 생성되기 때문에, finetuning 데이터셋 안에 chained tool-use 예시가 존재하지 않기 때문이다.

둘째, **interactive tool use**가 불가능하다. 특히 검색 엔진처럼 수많은 결과를 반환하는 도구에 대해, 결과를 탐색하거나 질의를 refinement하는 방식의 상호작용을 지원하지 않는다. 오늘날 browsing agent가 하는 multi-step 탐색과 비교하면, Toolformer는 기본적으로 **단발성 호출(single-shot tool use)**에 가깝다.

셋째, 실험 결과가 보여주듯, 도구 사용 학습이 항상 순수 LM 성능 향상으로 이어지지는 않는다. multilingual QA에서 일부 언어는 vanilla GPT-J를 안정적으로 넘지 못했고, finetuning 데이터 분포 문제가 원인일 수 있다고 분석된다. 즉, Toolformer의 이득은 task와 데이터에 따라 다르며, 단순히 “tool-use를 붙이면 항상 좋아진다” 수준은 아니다.

### 비판적 해석

이 논문은 오늘 기준으로 보면 agent 연구의 관점에서는 제한적이다. planning, memory, multi-hop retrieval, browser interaction 같은 요소가 없다. 그러나 그럼에도 불구하고 가치가 큰 이유는, Toolformer가 **“LM이 외부 도구를 호출하는 행위 자체를 학습할 수 있다”** 는 문제를 매우 깔끔한 형태로 정식화했기 때문이다.

특히 이 논문은 나중에 등장한 function calling LLM, ReAct류의 도구 사용, tool-augmented agents와 비교할 때 다음과 같은 중간 고리에 있다.

* 전통적 LM: 내부 파라미터만 사용
* Toolformer: 외부 도구 호출을 LM 행동으로 학습
* 현대 Agent: 다단계 계획 + 상호작용 + 도구 생태계

따라서 Toolformer는 오늘날 agent 기준으로는 단순하지만, **개념적으로는 매우 중요한 전환점**이다.

## 6. Conclusion

Toolformer는 언어 모델이 외부 도구를 활용하는 능력을 인간 주석 없이도 배울 수 있음을 보여준 논문이다. 소수의 API 사용 예시만 제공하고, 모델이 스스로 대규모 코퍼스에 API 호출 후보를 생성하게 한 뒤, 실제로 미래 토큰 예측 loss를 줄이는 호출만 남겨 finetuning함으로써 도구 사용 행동을 학습한다. 이 방식은 계산기, 검색, QA, 번역, 캘린더 등 다양한 도구에 적용되며, 여러 zero-shot downstream task에서 성능 향상을 보였다. 특히 GPT-J 6.7B 기반 모델이 더 큰 GPT-3 계열과 경쟁 가능한 수준의 결과를 보였다는 점이 인상적이다.

실무적으로 이 연구는 “모든 능력을 파라미터에 넣는 대신, 필요한 기능은 외부 시스템에서 호출하자”는 현대 LLM 시스템 설계 철학의 초기 형태를 제시했다. 연구적으로도 self-supervised tool-use learning, API linearization, loss-based usefulness filtering 같은 개념은 이후 agent 연구와 function-calling 패러다임에 중요한 영향을 준다. 다만 chaining과 interactive browsing이 불가능하다는 한계 때문에, 오늘날의 복합 에이전트 문제를 직접 해결하는 구조는 아니다. 그럼에도 Toolformer는 **LM이 도구를 쓴다**는 발상을 본격적으로 열어젖힌 대표 논문으로 평가할 수 있다.
