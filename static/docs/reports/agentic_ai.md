# Agentic AI

## 서론

### 1. 연구 배경

본 보고서는 Agentic AI 분야에서 최근 주목받고 있는 다중 에이전트 시스템의 설계, 학습, 평가, 안전성, 방법론 전반에 대한 체계적 분석을 목적으로 한다. 특히 LLM 기반 능동적 에이전트 시스템이 단순한 도구 수준을 넘어 독립적 의사결정과 협업을 가능하게 하는 복합 체계로 진화하고 있는 현상을 중심으로, 관련 문헌과 실험 결과를 종합하여 Agentic AI 의 연구 지형을 정리한다.

### 2. 문제의식 및 분석 필요성

다중 에이전트 시스템은 단일 에이전트와 달리 협업 체계, 역할 분화, 실패 모드, 일반화 능력 등 새로운 차원의 문제가 발생한다. 단일 LLM 도구를 넘어 memory, planning, action, coordination, self-improvement 를 통합한 시스템 설계 문제로 재정의된 LLM 기반 에이전트 연구는 기존 AI 안전성 평가의 한계를 드러내며, multi-agent 환경에서 발생하는 실패 모드와 위험 요인 또한 별도 체계적 분석이 필요한 영역이다.

### 3. 보고서의 분석 관점

본 보고서는 Agentic AI 연구 체계론적 분류, 방법론 계열 및 설계 패턴 분석, 실험 결과의 종합 정리라는 세 가지 관점으로 문헌을 정리한다. 이를 위해 연구 대상, 문제 정의, 접근 관점, 분류 키워드 네 가지 차원을 기준으로 논문을 체계적으로 분류하고, 여러 방법론 계열과 설계 패턴의 공통성과 차이점을 비교 분석한다.

### 4. 보고서 구성

본 보고서는 연구 체계 분류와 방법론 분석, 실험 결과 분석의 세 장으로 구성된다.

1 장은 다중 에이전트 시스템 설계, RL 기반 학습 전략, 안전성 평가, Survey 와 방법론이라는 네 가지 대분류와 그 하위 범주로 Agentic AI 연구의 분류 체계를 수립한다.

2 장은 Agentic AI 방법론이 공유하는 공통 문제 설정, 방법론 계열 분류, 핵심 설계 패턴, 학습 방식, 도구 활용 전략, 협업 메커니즘 등을 종합하여 방법론적 특징과 진화 경향을 분석한다.

3 장은 다양한 데이터셋과 실험 환경에 따른 평가 지표와 성능 결과, multi-agent 와 single-agent 의 성능 비교, 실험의 한계와 주의점, 결과 해석의 경향 등을 정리하여 Agentic AI 의 실험적 성취와 과제를 종합한다.

## 1 장. 연구체계 분류

### 1. 연구 분류 체계 수립 기준

본 연구 분류 체계는 제공된 논문 요약들을 **연구 대상 (연구 대상 시스템/도구의 유형)**,**문제 정의 (해결하고자 하는 핵심 과제의 성격)**,**접근 관점 (해결하려는 문제를 바라보는 시각)**,**분류 키워드** 네 가지 차원에서 종합하여 수립하였다.

- **연구 대상**: 논문이 다루는 시스템, 도구, 프레임워크, 벤치마크 등의 유형 (예: 멀티 에이전트 프레임워크, RL 학습 에이전트, 벤치마크 환경, survey 등)
- **문제 정의**: 해결하려는 핵심 과제의 성격 (예: 협업 체계 설계, 성능 평가, 새로운 위험 분석, 실제 환경 적용 등)
- **접근 관점**: 해법을 설계하고 적용하는 방식 (예: multi-agent 협업, hybrid architecture, benchmark 환경 구축, RL 학습 등)
- **분류 키워드**: 논문 요약문에 명시된 키워드를 참조하여 범주화에 활용

이 네 가지 차원은 상호 보완적이며, 각 논문은 이 기준들을 종합하여 분류된다. 하나의 논문이 여러 범주에 걸릴 수 있으나, 본 보고서에서는 **가장 대표적인 1 개 범주**에 배치하며, 나머지는 간접적 관련으로 고려하였다.

### 2. 연구 분류 체계

#### 2.1 대분류 A: Agent 시스템 설계 및 프레임워크

이 대분류는 **다중 에이전트 시스템의 설계, 프레임워크 구축, 협력 체계 수립**에 주력하는 연구들을 포괄한다. 핵심 문제는 단일 에이전트에서 넘어선**복합적 협업**,**역할 분화**,**워크플로 구성**,**시스템 아키텍처 설계** 등이며, 해법은 multi-agent 상호작용 구조, 역할 기반 협업, 대화 기반 오케스트레이션, 절차론적 협업 등에 기반한다.

##### 2.1.1 하위 범주 A-1: Multi-Agent 협업 프레임워크

다중 에이전트 시스템 전반을 아우르는 범용 프레임워크, 협업 메커니즘, 시스템 아키텍처 설계 연구들.

| 분류                     | 논문명                                                                                     | 분류 근거                                                                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 대분류 A > 하위 범주 A-1 | AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors (2023) | 인간 팀 문제 해결 절차를 LLM agent 시스템으로 번역한 adaptive closed-loop framework 로, task 특성에 따라 의사결정 구조를 동적으로 구성하며 feedback 기반 재구성 가능        |
| 대분류 A > 하위 범주 A-1 | AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation (2023)            | 프레임워크 논문으로서 다양한 응용 사례에서 범용성을 시연하며, conversation programming 패러다임을 통해 LLM·human·tool 을 통일된 대화 인터페이스로 통합하는 시스템 설계 기법 |
| 대분류 A > 하위 범주 A-1 | AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML (2025)                  | 단일 LLM 기반 생성이 아닌 Retrieval-Augmented Planning 과 역할별 계획 분해, 다단계 검증을 결합한 전체 파이프라인 자동화 멀티에이전트 프레임워크                             |
| 대분류 A > 하위 범주 A-1 | MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework (2023)                 | 자연어 대화보다 구조화된 문서와 명시적 절차에 기반한 협업 과정을 강조한 multi-agent framework                                                                               |
| 대분류 A > 하위 범주 A-1 | OpenAgents: An Open Platform for Language Agents in the Wild (2023)                        | 새로운 agent algorithm 을 제시하기보다, UI/백엔드/도구 인터페이스/실행 환경을 포함한 실제 소프트웨어 시스템으로서 language agent 를 설계하고 평가하는 platform paper        |

##### 2.1.2 하위 범주 A-2: Agent 설계 및 계획 (Planning)

에이전트의 내부 계획 (planning) 능력과 도구 사용 (tool use), 하이브리드 계획 구조 등을 다룬 연구들.

| 분류                     | 논문명                                                                      | 분류 근거                                                                                                                                                                                                                     |
| ------------------------ | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 대분류 A > 하위 범주 A-2 | AutoGPT+P: Affordance-based Task Planning with Large Language Models (2024) | classical planner 와 LLM 을 역할 분담한 hybrid 구조로, LLM 은 natural language 이해와 tool selection 에만 활용하고 planning 핵심은 symbolic domain 에서 수행되며, affordance-based representation 을 통해 substitution reason |
| 대분류 A > 하위 범주 A-2 | Toolformer: Language Models Can Teach Themselves to Use Tools (2023)        | LM 의 자기 예측 성능을 목적함수로 도구 유용성을 정의하며, 인간 주석 없이 도구 사용 패턴을自主学习하는 self-supervised tool-use paradigm                                                                                       |

##### 2.1.3 하위 범주 A-3: Agent 벤치마크 및 평가 환경

에이전트 학습/성능을 측정하기 위한 벤치마크 환경, 평가 프로토콜, 플랫폼 구축 연구들.

| 분류                     | 논문명                                                                        | 분류 근거                                                                                                                                                                      |
| ------------------------ | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 대분류 A > 하위 범주 A-3 | Pommerman: A Multi-Agent Playground (2018)                                    | 4 인 general-sum 환경과 폭탄 전략/부분 관측/통신 체계를 결합한 멀티 에이전트 학습용 벤치마크 환경 설계 연구                                                                    |
| 대분류 A > 하위 범주 A-3 | DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents (2025) | 실제 사용자 수요를 반영한 Deep Research Task 벤치마크와 보고서 품질·인용 정확도를 평가하는 두 가지 프레임워크 (RACE, FACT) 를 결합한 종단간 평가 체계를 제안하는 벤치마크 연구 |

#### 2.2 대분류 B: Agent 학습 및 진화 전략

이 대분류는 에이전트의 **학습 알고리즘**,**RL 기반 최적화**,**자기 진화**,**동적 구성**에 중점을 둔 연구들을 포괄한다. 핵심 문제에는**환경 충실도**,**generalization**,**robustness**,**emergence**,**오류로부터의 자기 개선** 등이 있으며, 해법은 RL 최적화 (GRPO 등), 자기 감독 학습, multi-agent 상호작용 데이터 활용, 진화적 설계 등을 포함한다.

##### 2.2.1 하위 범주 B-1: RL 기반 에이전트 학습

강화학습 기반 에이전트 학습 전략, 환경 충실도 향상, 실제 환경에서의 학습 등.

| 분류                     | 논문명                                                                                             | 분류 근거                                                                                                                                                                       |
| ------------------------ | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 대분류 B > 하위 범주 B-1 | DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments (2025) | 실제 웹 상호작용과 multi-agent browsing 구조를 통해 환경 충실도를 높인 GRPO RL 학습 프레임워크이며, contamination 통제를 통한 genuine search behavior 학습을 주장하는 실증 연구 |

##### 2.2.2 하위 범주 B-2: Agent 진화 및 협업 메커니즘

에이전트의 자기 진화, 역할별 협력, 동적 구조 적응, 진화적 동력 등.

| 분류                     | 논문명                                                                                  | 분류 근거                                                                                                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 대분류 B > 하위 범주 B-2 | Large Language Model Agent: A Survey on Methodology, Applications and Challenges (2025) | 이 논문은 LLM agent 를 단순 prompt engineering 의 연장이 아니라 memory, planning, action, coordination, self-improvement 를 포함한 복합 시스템 설계 문제로 재정의하는 통합 survey |

#### 2.3 대분류 C: Agent 시스템 평가 및 안전성 분석

이 대분류는 에이전트 시스템의 **성능 평가**,**벤치마크 설계**,**위험 분석**,**안전성 평가**,**제한 요소 규명** 등을 핵심으로 하는 연구들을 포괄한다. 핵심 문제에는**평가 메트릭 부재**,**실제 환경 성능 한계**,**multi-agent 환경 위험**,**도구 사용 제약** 등이 있으며, 해법은 새로운 벤치마크, 평가 프레임워크, 위험 분류 체계 제시 등.

##### 2.3.1 하위 범주 C-1: Agent 안전성 및 위험 분석

Multi-agent 환경에서 발생하는 위험 (miscoordination, conflict, collusion 등) 을 체계적으로 분석하고 완화하는 연구들.

| 분류                     | 논문명                                    | 분류 근거                                                                                                                                                                                              |
| ------------------------ | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 대분류 C > 하위 범주 C-1 | Multi-Agent Risks from Advanced AI (2025) | 이 보고서는 single-agent AI safety 연구의 한계를 지적하고, 상호작용하는 multi-agent 환경에서 발생하는 새로운 실패 양상을 three failure modes 와 seven risk factors 로 분리한 체계적 분류틀로 제시한다. |

#### 2.4 대분류 D: Agent 연구 방법론 및 Survey

이 대분류는 **Agent 연구 전반을 survey**,**taxonomy**,**종합 분석**하여 분야 전체를 이해하고 체계화하는 연구들이다.

##### 2.4.1 하위 범주 D-1: Survey 및 방법론

Agent 연구의 전반적인 방법론, 응용, 한계, 분야 동향을 survey 한 연구들.

| 분류                     | 논문명                                                                                  | 분류 근거                                                                                                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 대분류 D > 하위 범주 D-1 | Agentic Large Language Models, a survey (2025)                                          | agentic LLM 연구를 reason/act/interact 기능으로 정의하며 세 범주가 상호보완적 구조를 형성, 단순 LLM 도구가 아니라 새 데이터 생성 메커니즘임                                       |
| 대분류 D > 하위 범주 D-1 | Large Language Model Agent: A Survey on Methodology, Applications and Challenges (2025) | 이 논문은 LLM agent 를 단순 prompt engineering 의 연장이 아니라 memory, planning, action, coordination, self-improvement 를 포함한 복합 시스템 설계 문제로 재정의하는 통합 survey |

### 3. 종합 정리

본 연구 분류 체계는 Agentic AI 연구의 **시스템 설계 (다중 에이전트 협업 프레임워크)**,**학습 및 진화 (RL 기반 최적화 및 자기 진화)**,**평가 및 안전성 (벤치마크, 위험 분석)**,**방법론 및 Survey (분야 전체 종합)** 네 가지 대분류로 구성되어 있으며, 각 대분류는 상호 배타적이지 않고 연구의 측면 (예: 프레임워크 설계 논문이 동시에 안전성 이슈를 다루는 경우) 을 반영하여 유연하게 적용된다. 특히 multi-agent 협업, RL 학습, 벤치마크 설계, survey 등 명확한 연구 유형과 문제 정의에 따라 논문을 체계적으로 배치하고, 분류 근거는 각 논문의 핵심 키워드와 요약문 내용을 엄격히 반영하였다.

## 2 장. 방법론 분석

## 1. 공통 문제 설정 및 접근 구조

## 1.1. 문제 정의의 공통성

제공된 문서에 포함된 논문들은 모두 **LLM 기반 능동적 에이전트 시스템**을 다루며, 다음 공통 문제 설정을 공유한다:

| 차원     | 공통 요소                    | 설명                                                              |
| -------- | ---------------------------- | ----------------------------------------------------------------- |
| **입력** | 자연어 요구사항, 초기 상태   | 사용자 질의, 목표 $g \in \mathcal{G}$, 초기 상태 $s_{\text{old}}$ |
| **처리** | 추론 → 도구 선택 → 행동 실행 | Reasoning → Tool Selection → Action Execution 순환                |
| **출력** | 구조화된 답변/계획/코드      | 답변 텍스트, 계획/Partial Plan, 배포 가능한 코드                  |

## 1.2. 방법론 관점의 공통 구조

세 가지 상호보완적 축으로 체계화된다:

```text
┌──────────────────────────────────────────────────────────┐
│                  Agentic LLM Architecture                │
├──────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │   Reasoning  │  │    Acting    │  │  Interacting  │   │
│  │  (추론)      │  │   (행동)     │  │   (상호작용)  │   │
│  └──────────────┘  └──────────────┘  └───────────────┘   │
│           └──────────────┬──────────────────┘            │
│                          ▼                               │
│           ┌──────────────────────────────┐               │
│           │    Virtuous Cycle            │               │
│           │  (생성된 데이터 → 재학습)    │               │
│           └──────────────────────────────┘               │
└──────────────────────────────────────────────────────────┘
```

## 2. 방법론 계열 분류

제공된 논문들을 다음과 같은 방법론 계열로 그룹화한다:

## (계열 1) 다중 에이전트 대화 기반 협업 프레임워크

**계열 정의**: 메시지 기반 통신과 중앙/분산 의사결정 구조를 통한 multi-agent 시스템 설계

**공통 특징**:

- 메시지 passing 을 통한 decentralised workflow orchestration
- Auto-reply 메커니즘을 통한 연쇄 호출
- natural language + tool calling 혼합 제어
- termination condition 기반 자동 종료

**해당 논문**:

| 방법론 계열    | 논문명                                                                                     | 핵심 특징                                                        |
| -------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| 대화 기반 협업 | AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation (2023)            | ConversableAgent, GroupChatManager, dynamic speaker selection    |
|                | AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors (2023) | Expert recruitment, Collaborative decision-making, Feedback loop |

## (계열 2) 절차 중심 구조화된 협업

**계uel 정의**: 인간 조직의 SOP(Standardized Operating Procedures)를 agent 협업에 내재화하여 구조화된 산출물 형식 강제

**공통 특징**:

- 역할별 책임과 제약 정의
- ReAct 행동 원리 적용
- structured communication 프로토콜
- executable feedback mechanism 기반 iterative self-correction
- publish-subscribe 통신 패턴

**해당 논문**:

| 방법론 계열    | 논문명                                                                     | 핵심 특징                                                                   |
| -------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| 절차 중심 협업 | MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework (2023) | Product Manager~QA Engineer 역할, structured output, subscription mechanism |
|                | AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML (2025)  | Agent Manager, RAP, Plan Decomposition, Multi-stage Verification            |

## (계uel 3) 하이브리드 계획 아키텍처

**계열 정의**: 고전적 planner 와 LLM 기반 도구 선택을 결합한 hybrid 접근

**공통 특징**:

- Object Affordance Mapping 을 통한 symbolic scene representation
- PDDL syntax 기반 계획 생성
- LLM 도구 선택 피드백 루프
- missing object 대응을 위한 대체 탐색

**해당 논문**:

| 방법론 계열     | 논문명                                                                      | 핵심 특징                                                   |
| --------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------- |
| 하이브리드 계획 | AutoGPT+P: Affordance-based Task Planning with Large Language Models (2024) | Object detector, OAM, classical planner, LLM tool selection |

## (계uel 4) 강화학습 기반 실제 환경 학습

**계열 정의**: 실제 웹/검색 환경에서 end-to-end 강화학습을 통한 에이전트 학습

**공통 특징**:

- 실제 환경에서의 reward 기반 학습
- multi-step tool chain 처리
- observation masking 및 cache/retry mechanism
- contamination control

**해당 논문**:

| 방법론 계열  | 논문명                                                                                             | 핵심 특징                                         |
| ------------ | -------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| RL 기반 학습 | DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments (2025) | GRPO rollout, word-level F1 reward, real-world RL |

## (계uel 5) 도구 호출 자동 학습

**계uel 정의**: 대규모 언어 모델이 스스로 도구/API 호출 패턴을 학습하는 self-supervised 접근

**공통 특징**:

- API 텍스트 linearization
- in-context generation 기반 도구 후보 생성
- execution result 기반 loss filtering
- fine-tuning 을 통한 zero-shot 도구 사용

**해당 논문**:

| 방법론 계열    | 논문명                                                               | 핵심 특징                                                         |
| -------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 도구 자동 학습 | Toolformer: Language Models Can Teach Themselves to Use Tools (2023) | pseudo-label, loss-based filtering, self-supervised tool learning |

## 3. 핵심 설계 패턴 분석

## 3.1. 설계 패턴별 분류

### 패턴 A: Closed-loop Feedback 시스템

여러 논문에서 반복적으로 등장하는 구조:

| 논문                | 피드백 루프 구조                                                                                                                                                       |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AgentVerse (2023)   | 목표 $g$ → 전문가 구성 → 집단 결정 → 행동 실행 → 상태 전이 $s_{\text{new}} = \mathcal{T}(s_{\text{old}}, A)$ → 평가 $r = \mathcal{R}(s_{\text{new}}, g)$ → 다음 라운드 |
| AutoML-Agent (2025) | Request Verification → Execution Verification → Implementation Verification                                                                                            |
| Toolformer (2023)   | API 호출 → 실행 → 손실 감소 확인 → 유익한 호출 증강 → fine-tuning                                                                                                      |
| MetaGPT (2023)      | 요구사항 → PRD → 설계 → 구현 → 테스트 → 수정/반복 (최대 3 회 retry)                                                                                                    |

### 패턴 B: Multi-tier Architecture

| 논문                                     | 계층 구조                                                                             |
| ---------------------------------------- | ------------------------------------------------------------------------------------- |
| OpenAgents (2023)                        | Frontend website → Backend server → Agent logic (DataAgent, Plugins Agent, Web Agent) |
| AutoML-Agent (2025)                      | Agent Manager (전체 통제) + Ap/Ad/Am/Ao (각 역할 에이전트)                            |
| Large Language Model Agent Survey (2025) | 四层 구조 (methodology, evaluation tools, real-world issues, applications)            |

### 패턴 C: Multi-stage Verification

| 논문                      | 검증 단계                                                                           |
| ------------------------- | ----------------------------------------------------------------------------------- |
| AutoML-Agent (2025)       | Request Verification → Execution Verification → Implementation Verification         |
| DeepResearch Bench (2025) | Statement-URL 추출 → Jina Reader 텍스트 가져오기 → support 이진 분류 → metrics 집계 |

### 패턴 D: Dynamic Team Reconstitution

| 논문                | 적응 메커니즘                                                      |
| ------------------- | ------------------------------------------------------------------ |
| AgentVerse (2023)   | 실패 후 팀 자체 재구성, verbal feedback 기반 동적 전문가 설명 생성 |
| AutoML-Agent (2025) | RAP 기반 계획 탐색, 실패한 계획은 다른 계획 집합으로 환류          |

## 3.2. 학습 방식 분석

제공된 자료에 따른 학습/최적화 방식 분류:

```text
┌────────────────────────────────────────────────────────────────┐
│                   학습 방식 분류                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Training-free / Prompting                    │  │
│  │  - MetaGPT, AutoML-Agent, AgentVerse (verbal feedback)   │  │
│  │  - OpenAgents (prompt engineering + output formatting)    │  │
│  │  - AutoGen (built-in LLM 추론 기반)                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Self-supervised Learning                │  │
│  │  - Toolformer (pseudo-label, loss-based filtering)        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    RL-based Learning                      │  │
│  │  - DeepResearcher (GRPO, word-level F1 reward)            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Pretraining + Fine-tuning + Inference-time        │  │
│  │  - Agentic LLM Survey (Agentic behavior → 재투입)         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## 4. 방법론 비교 분석

## 4.1. 계열 간 차이점

### 문제 접근 방식

| 차원         | 대화 기반 협업           | 절차 중심 협업       | 하이브리드 계획   | RL 기반 학습          |
| ------------ | ------------------------ | -------------------- | ----------------- | --------------------- |
| **초점**     | 자연어 기반 의사결정     | 문서/프로cedure 중심 | symbolic planning | reward 최적화         |
| **의사결정** | 집단 판단 ($f(a_{m_i})$) | 역할별 책임          | planner 결정      | policy gradient       |
| **유연성**   | 동적 전문가 구성         | 고정 역할 프로파일   | hybrid            | environment dependent |

### 구조/모델 차이

```text
┌───────────────────────────────────────────────────────────┐
│                     구조적 특징 비교                      │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │         대화 기반 협업 (AutoGen, AgentVerse)        │  │
│  │  • Message-driven computation                       │  │
│  │  • Decentralised control plane                      │  │
│  │  • LLM = reasoning+acting+interacting               │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │         절차 중심 협업 (MetaGPT, AutoML-Agent)      │  │
│  │  • Structured output/format enforcement             │  │
│  │  • SOP 내재화                                       │  │
│  │  • Manager-controlled decomposition                 │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │         하이브리드 계획 (AutoGPT+P)                 │  │
│  │  • Symbolic planning (PDDL) + LLM tool selection    │  │
│  │  • Affordance-based scene representation            │  │
│  │  • Classical planner + LLM orchestration            │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │         RL 기반 학습 (DeepResearcher)               │  │
│  │  • GRPO rollout                                     │  │
│  │  • Word-level F1 reward                             │  │
│  │  • Real-world environment learning                  │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### 적용 대상 차이

| 계열            | 적합한 시나리오                                         | 제한사항                                                  |
| --------------- | ------------------------------------------------------- | --------------------------------------------------------- |
| 대화 기반 협업  | 자연어 응답, 창의적 작업, emergent behavior 탐구        | Zero-shot 설정, verbal feedback에 의존                    |
| 절차 중심 협업  | 소프트웨어 개발, ML 파이프라인, 구조화된 작업           | Strong backbone LLM 필요, skeleton code 부재 시 추가 작업 |
| 하이브리드 계획 | 로봇 계획, 누락 물체 대응, substituted reasoning        | Deterministic mapping, closed-world assumption            |
| RL 기반 학습    | 실제 검색 환경, web research, real-world generalization | 50-node cluster 필요, QA 학습셋 필요                      |

## 4.2. 복잡도 및 확장성

| 차원       | 대화 기반 협업                                          | 절차 중심 협업               | RL 기반 학습                   |
| ---------- | ------------------------------------------------------- | ---------------------------- | ------------------------------ |
| **복잡도** | message count 증가, team size 증가에 선형적             | 역할 개수, retry 횟수에 비례 | rollout step, environment size |
| **확장성** | dynamic speaker selection, horizontal/vertical topology | Manager 병렬 제어, plan 분해 | distributed tool parallelism   |
| **비용**   | API 호출 빈도, iteration 수                             | verification 단계 수         | cluster resources (50-node)    |

## 5. 방법론 흐름 및 진화

## 5.1. 초기 접근 방식

**2018~2023: 기본 인프라 및 벤치마크 구축**

| 연도 | 주요 발전                       | 대표 논문                                         |
| ---- | ------------------------------- | ------------------------------------------------- |
| 2018 | Multi-agent benchmark 환경 설계 | Pommerman: A Multi-Agent Playground (2018)        |
| 2023 | 대화 기반 multi-agent 시스템    | AutoGen (2023), AgentVerse (2023), MetaGPT (2023) |
| 2023 | 도구 자동 학습                  | Toolformer (2023)                                 |
| 2023 | 오픈 플랫폼 설계                | OpenAgents (2023)                                 |

**초기 접근 특징**:

- message passing 및 structured communication 도입
- SOP 와 structured output 형식화
- self-supervised 도구 학습 방법론 제안

## 5.2. 발전된 구조 (2024~2025)

**2024~2025: 하이브리드 아키텍처 및 RL 학습**

| 연도 | 주요 발전                    | 특징                                             |
| ---- | ---------------------------- | ------------------------------------------------ |
| 2024 | Hybrid planning architecture | AutoGPT+P: affordance + planner + LLM            |
| 2025 | Full-pipeline AutoML         | AutoML-Agent: RAP + Multi-stage Verification     |
| 2025 | Real-world RL 학습           | DeepResearcher: GRPO + real web environment      |
| 2025 | Comprehensive taxonomy       | Agentic LLM Survey: reasoning–acting–interacting |
| 2025 | Evaluation framework         | DeepResearch Bench, Multi-Agent Risks            |

**발전된 구조 특징**:

- hybrid 설계 (planner + LLM + affordance mapping)
- multi-stage verification 시스템
- real-world 환경에서의 end-to-end RL
- comprehensive evaluation taxonmy

## 5.3. 최근 경향

**2025 년의 주요 방향성**:

1. **Evaluation 중심**: RACE, FACT 같은 평가 프레임워크 등장
2. **Real-world generalization**: 실제 웹 환경에서의 RL 학습
3. **Hybrid approach**: 고전적 planner 와 LLM 의 결합
4. **Comprehensive taxonomy**: Agentic LLM을 3 축으로 체계화
5. **Risk 분석**: multi-agent risk taxonomy 및 대응 권고

## 6. 종합 정리

## 6.1. 전체 방법론 지형

제공된 문헌을 분석한 결과, Agentic AI 방법론은 다음 네 가지 차원에서 구조화된다:

### 차원 1: 구조적 조직화 방식

| 차원                   | 유형                                                    |
| ---------------------- | ------------------------------------------------------- |
| Communication topology | Horizontal (민주적 통합), Vertical (계층적)             |
| Control flow           | Centralized (Manager), Decentralized (message-driven)   |
| Output format          | Natural language, Structured documents, Executable code |

### 차원 2: 학습/최적화 메커니즘

```text
┌───────────────────────────────────────────────────┐
│                 학습 방식의 축                    │
├───────────────────────────────────────────────────┤
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │  Training-free / Prompting                  │  │
│  │  ┌────────────────┐  ┌────────────────┐     │  │
│  │  │  In-context    │  │  Verbal        │     │  │
│  │  │  Reasoning     │  │  Feedback      │     │  │
│  │  └────────────────┘  └────────────────┘     │  │
│  └─────────────────────────────────────────────┘  │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │  Self-supervised Learning                   │  │
│  │  ┌────────────────┐  ┌────────────────┐     │  │
│  │  │  Pseudo-label  │  │  Loss-based    │     │  │
│  │  │  generation    │  │  filtering     │     │  │
│  │  └────────────────┘  └────────────────┘     │  │
│  └─────────────────────────────────────────────┘  │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │  Reinforcement Learning                     │  │
│  │  ┌────────────────┐  ┌────────────────┐     │  │
│  │  │  GRPO rollout  │  │  Word-level    │     │  │
│  │  │  + masking     │  │  F1 reward     │     │  │
│  │  └────────────────┘  └────────────────┘     │  │
│  └─────────────────────────────────────────────┘  │
│                                                   │
└───────────────────────────────────────────────────┘
```

### 차원 3: 도구 활용 전략

| 전략                              | 대표 논문                                         |
| --------------------------------- | ------------------------------------------------- |
| **Self-selected tools**           | Toolformer: 모델이 스스로 유익한 도구 선택        |
| **Fixed tool set**                | AutoML-Agent, DeepResearcher: 고정 도구 집합 사용 |
| **Affordance-based substitution** | AutoGPT+P: 누락 물체 대응을 위한 대체 탐색        |

### 차원 4: 협업 메커니즘

| 메커니즘                      | 적용 논문                                      |
| ----------------------------- | ---------------------------------------------- |
| **Dynamic recruitment**       | AgentVerse: 목표 기반 동적 전문가 구성         |
| **Role specialization**       | MetaGPT, AutoML-Agent: 역할별 책임 정의        |
| **Multi-agent collaboration** | AutoGen, DeepResearcher: multi-agent 병렬 실행 |
| **Social simulation**         | Agentic LLM Survey: social capabilities        |

## 6.2. 방법론의 통합 관점

Agentic AI 방법론은 **"construction–collaboration–evolution"**이라는 세 단계로 진화한다:

```text
┌────────────────────────────────────────────────────────────┐
│                      Agentic Lifecycle                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   ┌──────────────┐  ┌───────────────┐  ┌───────────┐       │
│   │ Construction │  │ Collaboration │  │ Evolution │       │
│   │ (구성)       │  │ (협업)        │  │ (진화)    │       │
│   └──────────────┘  └───────────────┘  └───────────┘       │
│         │                  │                  │            │
│         ▼                  ▼                  ▼            │
│   profile           memory             multi-agent         │
│   memory            planning           self-learning       │
│   planning          action             external resources  │
│   action            collaboration      co-evolution        │
│                                                            │
│     └──────────────────────────────────────────┘           │
│                        Self-improvement Loop               │
└────────────────────────────────────────────────────────────┘
```

이 구조는 개별 논문들의 방법론이 공유하는 진화적 경향을 반영하며, 초기 대화 기반 시스템 → 절차 중심 구조화 → 하이브리드 아키텍처 및 RL 학습으로 발전한 흐름을 보여준다.

## 3장. 실험결과 분석

### 1. 평가 구조 및 공통 실험 설정

제공된 자료에 기반한 평가 구조를 정리에 따르면, Agentic AI 연구는 다음과 같은 공통 실험 환경을 사용하며, 여러 데이터셋 유형을 다변량 평가 방식으로 구성하였다.

#### 1.1 주요 데이터셋 유형

| 데이터셋 범주                 | 대표 사례                                                        | 사용 논문                                                                                  |
| ----------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 벤치마크 최적화               | MATH, HumanEval, MBPP, MGSM                                      | Agentic LLM survey(2025), AutoGen(2023), AgentVerse(2023), MetaGPT(2023), Toolformer(2023) |
| QA 및 지식 기반               | NaturalQuestions, TriviaQA, HotpotQA, 2WikiMultiHopQA            | DeepResearcher(2025)                                                                       |
| 도메인 전문 데이터셋          | Medical diagnosis, logistics, financial market, science research | Agentic LLM survey(2025)                                                                   |
| 시뮬레이션/게임 환경          | Minecraft, ALFWorld, OptiGuide, Pommerman(11×11 격자 맵)         | AgentVerse(2023), AutoGen(2023), AutoGPT+P(2024), Pommerman(2018)                          |
| 실제 환경                     | Controlled environment, wild environment, 실제 웹 환경           | OpenAgents(2023), DeepResearcher(2025)                                                     |
| 실제 검색 엔진 및 웹 브라우징 | 실제 검색엔진 및 웹 브라우징 환경                                | DeepResearcher(2025)                                                                       |

#### 1.2 평가 환경 유형

| 환경 유형               | 실험 설정                                  | 적용 사례                         |
| ----------------------- | ------------------------------------------ | --------------------------------- |
| 시뮬레이션 환경         | Minecraft 시뮬레이션, 11×11 격자 맵        | AgentVerse(2023), Pommerman(2018) |
| 실제 웹 환경            | 실제 검색엔진 및 웹 브라우징 환경          | DeepResearcher(2025)              |
| 실제 사용자 요청        | 실제 사용자 질의 96,147 개                 | DeepResearch Bench(2025)          |
| controlled vs wild 환경 | Controlled environment 와 wild environment | OpenAgents(2023)                  |

#### 1.3 비교 방식

| 비교 방식            | 상세                                                    | 적용 사례                                 |
| -------------------- | ------------------------------------------------------- | ----------------------------------------- |
| Baseline 대비        | vanilla GPT, single-agent, existing methods             | 대부분의 논문                             |
| SOTA 비교            | SoTA 코드 생성, SoTA 성능                               | MetaGPT(2023)                             |
| Multi-agent baseline | Dynamic group chat, solo vs group, multi-agent baseline | AutoGen(2023), AgentVerse(2023)           |
| 기존 모델 대비       | 기존 retrieval/search 방법, 기존 LLM-based planners     | Agentic LLM survey(2025), AutoGPT+P(2024) |

#### 1.4 주요 평가 지표

| 지표 유형   | 지표명                                                                  | 사용 빈도                                                        |
| ----------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 정확도 기반 | accuracy, Pass@1, Pass@k, F1 score, success rate, task success rate     | 모든 연구                                                        |
| 성능 점수   | benchmark score, normalized performance score, RACE, NPS, MBE           | 일부 연구                                                        |
| 비용/효율   | Token usage, cost ($), LLM call 횟수, 탐색 시간                         | AutoML-Agent(2025), MetaGPT(2023)                                |
| 안전성/품질 | Safety, Citation Accuracy, Human revision cost, Emergent behavior       | Multi-Agent Risks(2025), MetaGPT(2023), DeepResearch Bench(2025) |
| 평가 지표   | emergent behavior, user value, FACT, Instruction-Following, Readability | Agentic LLM survey(2025), DeepResearch Bench(2025)               |

### 2. 주요 실험 결과 정렬

#### 2.1 코드 생성 및 소프트웨어 엔지니어링 태스크

| 논문명            | 데이터셋/환경                                                | 비교 대상                                          | 평가 지표                                  | 핵심 결과                                                                                           |
| ----------------- | ------------------------------------------------------------ | -------------------------------------------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| MetaGPT (2023)    | HumanEval(164 tasks), MBPP(427 tasks), SoftwareDev(70 tasks) | GPT-4 단독, AutoGPT, AgentVerse, ChatDev           | Pass@1, Executability, Human revision cost | HumanEval 85.9% Pass@1, MBPP 87.7% Pass@1, SoftwareDev Executability 3.75, Human revision cost 0.83 |
| AgentVerse (2023) | HumanEval, MBPP                                              | Solo (단일 agent), Group                           | accuracy, pass@1                           | HumanEval Group 최고 성능 89.0, coding 영역에서 multi-agent discussion 이 이점                      |
| AutoGen (2023)    | OptiGuide 코드 작업, 100 개 safe/unsafe coding task          | vanilla GPT-4, Multi-Agent Debate, LangChain ReAct | code 줄 수, F1 score, success rate         | 코드 줄 수 430→100 줄 (3.3 배 감소), unsafe code 식별 F1: GPT-4 대비 8%, GPT-3.5 대비 35% 향상      |

#### 2.2 도구 사용 및 오토메이션 태스크

| 논문명            | 데이터셋/환경                                      | 비교 대상                                             | 평가 지표                                      | 핵심 결과                                                                                                     |
| ----------------- | -------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| AgentVerse (2023) | 도구 사용 과제 10 개 (최소 두 종류 이상 도구 필요) | CoT, Solo, Group                                      | accuracy, 도구 사용 성공률, 요구사항 커버리지  | 10 개 중 9 개 해결 (단일 ReAct 는 3 개), GPT-4 기반에서는 협업 의사결정이 일관되게 우수                       |
| AutoGPT+P (2024)  | SayCan instruction set, 150 scenarios              | SayCan (81%), 기존 LLM-based planners                 | task success rate                              | 98% 성공률 (기존 81% 대비 17%p 개선), 150 scenarios 에서 79% 성공률                                           |
| Toolformer (2023) | QA 데이터셋, MLQA (63 개 언어)                     | vanilla GPT-J, GPT-3 175B, CCNet fine-tuning          | 토큰 예측 loss, downstream task zero-shot 성능 | GPT-J 6.7B 기반 모델이 더 큰 GPT-3 175B 보다 나은 성능, MLQA 모든 언어에서 API 호출 시 성능 향상 (63.8~94.9%) |
| OpenAgents (2023) | 200 개 이상 plugin, 웹 브라우징 기능               | AutoGPT, BMTools, BabyAGI, Gentopia, Open Interpreter | 온라인 배포 가능성, 도구 수                    | 200 개 이상 plugin 통합 및 Real-time streaming generation 구현                                                |

#### 2.3 연구 및 조사 태스크

| 논문명                    | 데이터셋/환경                                       | 비교 대상                                                         | 평가 지표                    | 핵심 결과                                                                                                                              |
| ------------------------- | --------------------------------------------------- | ----------------------------------------------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| DeepResearch Bench (2025) | 100 개 PhD 급 연구 과제 (중국어 50, 영어 50)        | Gemini-2.5-Pro DR, OpenAI Deep Research, Grok, Perplexity, Claude | RACE, FACT                   | Gemini-2.5-Pro DR RACE 48.88(최고), OpenAI 46.98, Perplexity 42.25, Grok 40.24, Claude 40.67; Perplexity Citation Accuracy 90.24(최고) |
| DeepResearcher (2025)     | in-domain 4 종, OOD 3 종, 실제 웹 환경              | Search-r1-base, local RAG 기반 RL, R1-Searcher                    | F1, MBE, pass@10             | in-domain 4 데이터셋 모두 MBE 최고 성능 (NQ 61.9, TQ 85.0, HotpotQA 64.3, 2Wiki 66.6); OOD 모두 최고 성능                              |
| AutoML-Agent (2025)       | 14 개 데이터셋, 5 개 모달리티, 7 개 downstream task | Human Models, AutoGluon, GPT-3.5, GPT-4, DS-Agent                 | SR, NPS, CS, 탐색 시간, 비용 | 평균 SR 87.1%, NPS 와 CS 에서 모든 baseline 우위, SELA 대비 8 배 빠른 탐색 시간                                                        |

#### 2.4 다중 에이전트 및 협업 태스크

| 논문명            | 데이터셋/환경                                                 | 비교 대상                                              | 평가 지표                                        | 핵심 결과                                                                                                                          |
| ----------------- | ------------------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| AgentVerse (2023) | FED, Commongen Challenge, MGSM, Logic Grid Puzzles, Minecraft | CoT, Solo, Group                                       | accuracy, emergent behavior                      | MGSM 95.2→96.0→95.2, Logic Grid Puzzles 59.5→64.0→66.5, Minecraft 에서 volunteer/conformity/destroyive behavior 관찰               |
| AutoGen (2023)    | ALFWorld 환경, 12 개 수작업 태스크                            | vanilla GPT-4, Multi-Agent Debate, 2-agent static chat | success rate, 상호작용 횟수                      | ALFWorld grounding agent 추가 시 평균 15% 성능 개선, Dynamic group chat 에서 role-play prompt 가 task-based 보다 높은 success rate |
| MetaGPT (2023)    | SoftwareDev(70 tasks)                                         | GPT-4 단독, 다른 multi-agent 프레임워크                | Productivity, Executability, Human revision cost | Engineer 만 3.0→다중 역할 3.75, executable feedback: HumanEval +4.2%, MBPP +5.4% Pass@1                                            |

#### 2.5 안전성 및 위험 분석

| 논문명                   | 데이터셋/환경                                           | 비교 대상                                     | 평가 지표                           | 핵심 결과                                                                                                                       |
| ------------------------ | ------------------------------------------------------- | --------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Multi-Agent Risks (2025) | 실제 사례 (trading, 군사 지휘, critical infrastructure) | single-agent AI, existing AI safety framework | failure mode, risk factor, taxonomy | multi-agent risk 는 single-agent risk 의 단순 합이 아님, 실패 모드: miscoordination/conflict/collusion, 위험 요소 7 가지 체계화 |
| AutoGen (2023)           | 100 개 safe/unsafe coding task                          | vanilla GPT-4, LangChain ReAct                | success rate, F1 score              | unsafe code 식별 F1: GPT-4 대비 8%, GPT-3.5 대비 35% 향상                                                                       |

#### 2.6 게임 및 강화학습 태스크

| 논문명           | 데이터셋/환경                                     | 비교 대상                                             | 평가 지표 | 핵심 결과                                                                            |
| ---------------- | ------------------------------------------------- | ----------------------------------------------------- | --------- | ------------------------------------------------------------------------------------ |
| Pommerman (2018) | 11×11 격자 맵, 4 명 에이전트 FFA 및 team variants | SimpleAgent, Deep Q-Learning, PPO, 초기 제출 에이전트 | 승률      | DAgger 사용 시 FFA 에서 SimpleAgent 승률 약 20%, 상위 에이전트 22 승/35 경기 (62.9%) |

### 3. 성능 패턴 및 경향 분석

#### 3.1 Multi-agent vs Single-agent 비교 결과

여러 논문을 종합한 multi-agent 와 single-agent 의 성능 비교 경향을 정리하면 다음과 같다.

**일관적으로 multi-agent 가 우세한 영역:**

- 코드 생성: MetaGPT(85.9%) 와 HumanEval 에서 Group 이 Solo 를 능가 (89.0%), MBPP 에서 +5.4% Pass@1 향상
- 도구 사용: AgentVerse 에서 10 개 중 9 개 해결 (단일 ReAct 는 3 개)
- 협업 의사결정: GPT-4 기반에서 일관되게 우수
- ALFWorld: grounding agent 추가 시 평균 15% 성능 개선

**모델에 따른 상충되는 결과:**

- AgentVerse 에서 GPT-3.5 는 일부 task 에서 Group 이 Solo 에 뒤처짐 (MGSM 오류의 10% 가 잘못된 방향 수정에서 발생)
- AutoML-Agent 는 소형 모델에서 복잡한 planning/code generation 취약

**데이터셋 의존성:**

- DeepResearcher 는 실제 웹 환경 훈련 시 OOD 벤치마크에서도 월등한 성능 (MBE 최고)
- DeepResearch Bench 는 Gemini(Comprehensiveness 강) 와 Perplexity(Citation Accuracy 강) 의 상보적 강점

#### 3.2 평가 환경에 따른 성능 차이

| 환경 유형          | 성능 패턴                    | 관찰 사례                                                              |
| ------------------ | ---------------------------- | ---------------------------------------------------------------------- |
| 실제 웹 환경       | 일반화 능력 월등히 우수      | DeepResearcher(2025): in-domain/OOD 모두 최고 성능                     |
| 시뮬레이션 환경    | 전략적 탐색 가능성 높음      | Pommerman(2018): 초보 인간이 시도하지 않는 폭탄 투사체 전략 발견       |
| Controlled vs Wild | Wild 환경이 실제 사용성 반영 | OpenAgents(2023): UX 메트릭 (지연, 실패 처리) 이 실제 사용성에 더 중요 |

#### 3.3 비용/효율성 패턴

| 지표        | 관측된 패턴                                                                       |
| ----------- | --------------------------------------------------------------------------------- |
| 토큰 효율성 | MetaGPT: 124.3 tokens/line, 실행 가능한 피드백으로 효율성 개선                    |
| 코드 최적화 | AutoGen: 430 줄→100 줄 (3.3 배 감소)                                              |
| 탐색 비용   | AutoML-Agent: SELA 대비 8 배 빠른 탐색 시간, 단일 모델 탐색 비용 525 초 0.30 달러 |

#### 3.4 실패 모드의 다중 에이전트 특이성

Multi-Agent Risks(2025)에 따르면, multi-agent environment 는 single-agent와 상이한 실패 모드를 가진다:

- **Miscoordination**: 에이전트 간 조율 실패
- **Conflict**: 상호작용 갈등
- **Collusion**: 비윤리적 협력

이러한 실패 모드는 single-agent risk 의 단순 합이 아닌, 상호작용에서 emergent 한 위험으로, static safety evaluation 보다 배치 후 진화적 압력이 중요하다.

### 4. 추가 실험 및 검증 패턴

제공된 자료에서 관측된 추가 실험 및 검증 패턴은 다음과 같다.

#### 4.1 Ablation Study

| 논문명                    | ablation 대상                | 결과                                                               |
| ------------------------- | ---------------------------- | ------------------------------------------------------------------ |
| MetaGPT (2023)            | 역할 수, executable feedback | Engineer 만 3.0→다중 역할 3.75, HumanEval +4.2%, MBPP +5.4% Pass@1 |
| AutoML-Agent (2025)       | RAP, plan decomposition      | RAP alone → 성능 저하, downstream 성능↑, code verification 부족    |
| DeepResearch Bench (2025) | Reference report             | No Reference 변형 성능 하락, PAR 71.33 vs 인간 간 68.44            |

#### 4.2 민감도 분석

| 논문명              | 민감도 분석 대상              | 결과                                                      |
| ------------------- | ----------------------------- | --------------------------------------------------------- |
| AgentVerse (2023)   | 기저 모델 (GPT-3.5 vs GPT-4)  | GPT-4 기반에서는 일관된 우수, GPT-3.5 는 상충 정보에 취약 |
| AutoGPT+P (2024)    | missing object, implicit 의도 | missing object 상황 대응, implicit 의도 해석 테스트       |
| AutoML-Agent (2025) | $P$ (RAP 계획 수)             | $P=3$에서 다양성/비용 균형                                |

#### 4.3 조건 변화 실험

| 논문명                    | 조건 변화                             | 결과                                             |
| ------------------------- | ------------------------------------- | ------------------------------------------------ |
| AutoGen (2023)            | role-play prompt vs task-based prompt | role-play 가 conversation context 반영도 더 높음 |
| DeepResearch Bench (2025) | Judge LLM 비용                        | Gemini Pro $0.13/최고 성능                       |
| DeepResearcher (2025)     | harder question                       | harder question 일수록 tool call 증가            |

#### 4.4 Emergent Behavior 관찰

| 논문명                | Emergent Behavior 유형                          | 관찰 환경            |
| --------------------- | ----------------------------------------------- | -------------------- |
| AgentVerse (2023)     | volunteer, conformity, destructive behavior     | Minecraft 시뮬레이션 |
| DeepResearcher (2025) | planning, cross-validation, reflection, honesty | RL 학습 과정         |

### 5. 실험 설계의 한계 및 비교상의 주의점

#### 5.1 비교 조건의 불일치

| 한계                  | 설명                                                                                             | 영향                          |
| --------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------- |
| 엄격한 공정 비교 부재 | AutoGen(2023): 각 응용마다 metric 과 환경 제각각                                                 | 방법론 간 직접 비교 어려움    |
| evaluation 불안정     | Large Language Model Agent survey(2025): agent evaluation 이 여전히 불안정, standardization 부족 | 결과 재현성 문제              |
| base model 의존성     | 여러 논문: 결과 base model 품질에 크게 의존                                                      | 모델 변경 시 결과 재검토 필요 |

#### 5.2 데이터셋 의존성

| 데이터셋             | 한계                                                                       |
| -------------------- | -------------------------------------------------------------------------- |
| HumanEval/MBPP       | software engineering 도메인에 집중되어 일반화 가능성 불명확 (MetaGPT 2023) |
| Benchmarks           | benchmark 편향 가능성 (Agentic LLM survey 2025)                            |
| 도메인 전문 데이터셋 | 도메인 편향 가능성 (DeepResearch Bench 2025)                               |

#### 5.3 일반화 한계

| 제한 사항                 | 설명                                                           |
| ------------------------- | -------------------------------------------------------------- |
| tool chaining 불가능      | Toolformer(2023): single-shot tool use 에 근접                 |
| uncertainty modeling 부족 | AutoGPT+P(2024): deterministic OAM (uncertainty modeling 부족) |
| novel classes 지원 안됨   | AutoGPT+P(2024): OAM 자동화 한계, novel classes 지원 안됨      |
| skeleton code 부재        | AutoML-Agent(2025): skeleton code 부재 시 hallucination 위험   |

#### 5.4 평가 지표의 한계

| 지표              | 한계                                                                                            |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| word-level F1     | DeepResearcher(2025): reward 는 word-level F1 에 의존 (장문 연구 보고서 품질 직접 최적화 안 됨) |
| human consistency | DeepResearch Bench(2025): 인간 평가 225 person-hours 로 확장성 부족                             |
| MBE 한계          | DeepResearcher(2025): GPT-4o-mini 기반 MBE 만의 한계 존재                                       |

### 6. 결과 해석의 경향

#### 6.1 저자들의 공통 해석 경향

| 경향                 | 설명                                                                                                                                           |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 구조적 접근          | Agentic LLM survey(2025): reasoning-acting-interacting 은 독립 기술이 아니라 상호보완적 구조                                                   |
| 시스템 공학화        | Large Language Model Agent survey(2025): 단일 프롬프트 래퍼를 넘어 memory/planning/action/coordination을 포함한 복합 시스템 설계 문제로 재정의 |
| 환경 충실도 중요성   | DeepResearcher(2025): environment fidelity 가 성능의 핵심 결정인자                                                                             |
| 검증 메커니즘 결정적 | MetaGPT(2023): multi-agent 자체보다 역할 설계와 검증 메커니즘이 성능에 결정적                                                                  |
| hybrid 구조 우수성   | AutoGPT+P(2024): hybrid 구성이 LLM 의 자연어 능력은 살리고 planner 의 symbolic reasoning 는 보완                                               |

#### 6.2 해석과 관찰 결과의 구분

| 관찰 결과                                            | 저자 해석                                                                                                 |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Multi-agent 가 coding 영역에서 우수 (HumanEval 89.0) | LLM 의 사전학습으로 인한 내성, multi-agent 가 요구사항 커버리지와 작업 지속성 개선                        |
| GPT-3.5는 잘못된 피드백에 취약 (MGSM 오류 10%)       | GPT-4 기반에서는 협업 의사결정이 일관되게 우수, GPT-3.5는 상충 정보에 취약                                |
| 실제 웹 환경 RL agent 가 월등한 성능 (MBE 최고)      | real-world environment 에서 RL 수행한 agent 가 단순 inference 시 웹 허용 모델보다 일반화 능력 월등히 우수 |

#### 6.3 "우수하다" 표현 대신 구체적 결과 사용

| 예시                          | 해석 표현      | 구체적 결과                                                         |
| ----------------------------- | -------------- | ------------------------------------------------------------------- |
| AutoGPT+P가 "우수함"          | 17%p 성능 개선 | SayCan 대비 98% 성공률 (기존 81% 대비 17%p 개선)                    |
| DeepResearch Bench가 "우수함" | RACE 점수 차이 | Gemini-2.5-Pro DR 48.88(최고), OpenAI 46.98, Perplexity 42.25       |
| Multi-agent가 "효과적"        | 조건 의존성    | GPT-4 기반에서는 일관되게 우수, GPT-3.5에서는 일부 task 에서 뒤처짐 |

### 7. 종합 정리

Agentic AI 연구에서 여러 논문의 실험 결과는 multi-agent 시스템이 코드 생성, 도구 사용, 협업 의사결정에서 single-agent 를 능가한다는 일관된 경향을 보여주며, 이는 모델 (GPT-4 기반) 과 환경 (실제 웹 환경) 에 크게 의존한다. 코드 생성 태스크에서는 executable feedback 과 역할 설계가 성능에 결정적이고, 도구 사용 태스크에서는 hybrid 구조가 missing object 와 implicit 의도 해석에서도 robust 한 성능을 달성한다. 연구 및 조사 태스크에서는 실제 웹 환경에서 RL 로 훈련된 agent 가 일반화 능력이 월등히 우수하며, Gemini 와 Perplexity 는 각각 Comprehensiveness 와 Citation Accuracy 에서 상보적 강점을 가진다. 다만, 모든 태스크에서 multi-agent 가 우세한 것은 아니며, 소형 모델과 GPT-3.5 기반에서는 성능이 제한되고, tool chaining 과 uncertainty modeling 은 아직 해결되지 않은 문제이다. 안전성 측면에서는 single-agent risk 보다 multi-agent environment 에서 emergent 한 실패 모드 (miscoordination, conflict, collusion) 가 추가 평가 대상이 되어야 하며, evaluation 표준화가 필요한 분야로 나타났다.
