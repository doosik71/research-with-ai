# OpenAgents: An Open Platform for Language Agents in the Wild

## 1. Paper Overview

이 논문은 **OpenAgents**라는 오픈소스 플랫폼을 제안한다. 목표는 LLM 기반 language agent를 연구용 프로토타입이나 개발자용 CLI 수준에 머물지 않고, **일반 사용자도 실제 생활에서 사용할 수 있는 형태로 배포·운영 가능한 플랫폼**으로 만드는 것이다. 저자들은 기존 agent framework가 proof-of-concept 구현에는 집중했지만, 비전문 사용자의 접근성, 실제 서비스 수준의 UI/백엔드, 장애 처리, 실시간 스트리밍, 도구 확장성 같은 **application-level design**을 충분히 다루지 못했다고 본다. 이에 OpenAgents는 세 가지 대표 agent, 즉 **Data Agent**, **Plugins Agent**, **Web Agent**를 통합한 웹 기반 플랫폼을 설계하고, 사용자·개발자·연구자 각각에게 의미 있는 사용 환경을 제공하려 한다.

연구 문제는 단순히 “LLM agent를 잘 만들 수 있는가”가 아니라, **“LLM agent를 현실 세계의 사용자와 환경 속에서 실제로 동작하게 만들려면 무엇이 필요한가”**에 가깝다. 논문은 이 문제를 중요하게 보는 이유로, 기존 벤치마크가 통제된 환경에서 특정 기능만 평가하는 경우가 많아, 실제 서비스 환경에서 발생하는 지연, API 실패, UI 문제, 사용자 개입, 웹 환경의 불안정성 같은 요소를 충분히 반영하지 못한다고 지적한다. OpenAgents는 이런 간극을 줄이기 위한 플랫폼적 시도다.

## 2. Core Idea

이 논문의 핵심 아이디어는 다음 한 문장으로 요약할 수 있다.

**Language agent를 하나의 “모델 프롬프트”가 아니라, UI·백엔드·툴 인터페이스·실행 환경을 포함한 실제 소프트웨어 시스템으로 다뤄야 한다.**

즉, OpenAgents의 공헌은 새로운 reasoning algorithm 자체를 제시하는 데 있지 않다. 대신 다음 세 가지를 결합한 **실전형 agent platform**을 만든 데 있다.

첫째, **일반 사용자를 위한 웹 UI**를 제공한다. 기존 agent 프레임워크가 CLI나 개발자 중심 사용성을 가졌던 것과 달리, OpenAgents는 웹 인터페이스를 통해 비개발자도 agent 기능을 사용할 수 있게 한다.

둘째, **개발자와 연구자를 위한 배포 가능한 오픈 플랫폼**이다. 프론트엔드, 백엔드, 에이전트 로직, 실행 환경을 포함해 로컬 배포와 확장이 가능하도록 설계했다. 이 점에서 단순 데모가 아니라 실험·개발 기반(codebase)까지 제공하는 플랫폼을 지향한다.

셋째, **in-the-wild evaluation**의 기반을 제공한다. 사용자가 실제 요청을 하고, agent가 실제 도구·브라우저·코드 실행 환경에서 이를 처리하는 과정과 피드백을 수집할 수 있으므로, 논문은 OpenAgents를 현실 환경 기반의 human-in-the-loop 평가 플랫폼으로도 본다.

기존 연구와 비교했을 때 novelty는 “더 강한 단일 agent”라기보다, **실제 배포 가능한 language agent ecosystem**을 공개했다는 데 있다. 표 1에서는 OpenAgents가 온라인 배포 가능, 웹 UI 제공, human feedback 지원, 200개 이상의 도구, 웹 브라우징 기능, 그리고 controlled와 wild 환경을 모두 다룬다는 점을 강조한다. 이는 기존 AutoGPT, BMTools, BabyAGI, Gentopia, Open Interpreter 등과 구분되는 지점이다.

## 3. Detailed Method Explanation

### 3.1 전체 시스템 구조

OpenAgents의 아키텍처는 크게 두 부분으로 나뉜다.

1. **User Interface**
2. **Language Agent**

논문 4페이지의 Figure 2는 이 구조를 도식화한다. 사용자 요청은 프론트엔드/백엔드를 거쳐 language agent로 전달되고, language agent는 language model, tool interface, environment를 활용해 행동을 수행한다. 즉, agent는 단순 텍스트 생성기가 아니라, 외부 환경과 연결된 실행 주체로 설계된다.

### 3.2 User Interface 설계

논문은 UI를 주변 요소가 아니라 핵심 구성요소로 본다. 여기에는 다음이 포함된다.

* frontend website
* backend server
* streaming
* error handling
* database/user system

저자들은 기존 agent 연구가 에이전트 reasoning이나 action planning 자체에 집중한 반면, 실제 서비스로 만들기 위해 필요한 **backend operation, streaming, failure handling, data storage** 등의 재사용 가능한 business logic을 충분히 다루지 않았다고 본다. 그래서 OpenAgents는 사용자와 에이전트 사이의 다리 역할을 하는 UI/백엔드를 체계적으로 구축한다.

### 3.3 Language Agent 설계

Language Agent는 세 구성요소로 이루어진다.

* **language model**
* **tool interface**
* **environment**

에이전트는 ReAct 계열의 흐름을 따라 매 턴마다 대체로

**Observation $\rightarrow$ Deliberation $\rightarrow$ Action**

의 순서로 동작한다. 모델은 파싱 가능한 형식의 텍스트를 출력하도록 프롬프트되며, tool interface는 이를 실제 코드 실행, API 호출, 브라우저 조작 같은 실행 가능한 액션으로 변환한다. 이후 환경이 그 액션을 수행하고, 그 결과가 다시 관측으로 들어간다.

여기서 중요한 점은 논문이 agent를 추상 reasoning entity로만 보지 않고, **“출력 파싱 가능성”과 “실행 가능성”**을 매우 중시한다는 것이다. 즉, agent의 응답은 사람이 읽기 위한 텍스트이면서 동시에 시스템이 해석해 행동으로 전환할 수 있는 형태여야 한다.

### 3.4 Practical Implementation Challenges

논문 5페이지부터는 실제 구현에서 마주친 핵심 문제들을 정리한다. 이는 이 논문의 가장 실무적인 기여 중 하나다.

#### 3.4.1 Adaptive Data Mapping / DataModel

OpenAgents는 문자열 하나로 모든 데이터를 다루지 않는다. 대신 **DataModel**이라는 추상화를 도입해 텍스트, 코드, 이미지, 테이블, 데이터베이스 구조 등 다양한 raw data를 사람·프론트엔드·LLM·저장소가 각각 다루기 적합한 형태로 변환한다. 예를 들어 LLM은 대용량 테이블 전체를 처리하기 어렵기 때문에 일부 row를 선형화한 문자열이 필요할 수 있고, 반면 사람은 인터랙티브 표나 시각화가 더 적합하다. DataModel은 이 다중 표현 문제를 해결하기 위한 계층이다.

#### 3.4.2 Strategic Data Storage

다중 사용자 환경을 지원하기 위해 데이터 저장을 계층화했다.

* 임시 변수: 메모리
* 전역 변수: Redis
* 사용자별 대화/데이터: MongoDB

이 설계는 단순 데모를 넘어서 실제 multi-user application처럼 동작하도록 하기 위한 것이다.

#### 3.4.3 User-Centric Interface

각 agent 유형에 따라 UI도 다르게 설계된다.

* **Data Agent**: 코드, 콘솔 출력, 이미지, 인터랙티브 시각화를 notebook처럼 섞어 보여줌
* **Plugins Agent**: 카드 UI 등으로 API 결과를 보기 좋게 구조화
* **Web Agent**: 브라우저 확장 프로그램을 통해 실행 계획과 단계를 사용자가 추적·개입 가능하게 표시

이 부분은 agent usability 연구 측면에서 중요하다. 단순히 정답률이 높은 것이 아니라, 사용자가 agent가 무엇을 하는지 이해하고 개입할 수 있어야 실서비스 가치가 커진다는 관점이다.

#### 3.4.4 Real-time Response Streaming

긴 응답을 한 번에 반환하면 사용자 체감 지연이 크다. 이를 해결하기 위해 OpenAgents는 **streaming generation**을 도입하고, 생성되는 토큰의 역할을 실시간으로 구분해 렌더링하는 문제를 다룬다. 논문은 이 과정을 automata 이론, 특히 pushdown automata와 연결지어 설명한다. 생성 중인 토큰이 일반 응답인지, 툴 호출인지, special marker인지 구분하는 문제를 상태 전이 관점으로 해석한다.

이 부분은 알고리즘 자체보다 시스템 구현 관점에서 흥미롭다. 많은 agent 논문이 streaming UX 문제를 거의 다루지 않는데, 이 논문은 이를 현실 사용성의 중요한 요소로 본다.

#### 3.4.5 System Robustness

논문은 robustness를 세 축으로 본다.

* **failure handling**
* **in-time response**
* **token overflow**

예를 들어 외부 API 실패 시 retry/terminate 결정을 해야 하고, 여러 LLM key pool을 활용해 rate limit 부담을 줄이며, 사용자가 generation을 중단/재시도할 수 있게 한다. 또한 대화 이력이 길어질 때는 `MessageDataModel`을 통해 오래된 메시지를 잘라내고 필요한 액션/도구 응답만 유지한다. 이 설계는 “agent가 잘 추론하느냐”보다 “서비스가 망가지지 않고 돌아가느냐”에 가까운 문제를 다룬다.

#### 3.4.6 Chrome Extension 기반 Web Control

Web Agent는 Chrome extension과 Chrome Debugger API를 사용해 **사용자 브라우저를 직접 조작**한다. 이는 서버 측 비가시적 browsing이 아니라, 사용자의 브라우저에서 agent action이 눈에 보이고 중단도 가능하다는 점에서 차별적이다. 논문은 이런 설계가 웹 보조 에이전트를 더 현실적이고 통제 가능하게 만든다고 주장한다.

### 3.5 세 가지 에이전트

#### 3.5.1 Data Agent

Data Agent는 Python과 SQL 코드 생성/실행을 지원하며, Kaggle Data Search, Data Profiling, ECharts 같은 데이터 도구를 통합한다. 흥미로운 설계는, 데이터 작업의 경우 코딩 비중이 높기 때문에 에이전트가 직접 복잡한 코드를 길게 생성하기보다, **tool 내부에 language model을 넣어 tool이 코드를 생성**하게 했다는 점이다. 즉, Python/SQL/ECharts tool이 코드 생성 능력을 가지며, agent는 이를 orchestration한다.

논문 6페이지 Figure 3은 Data Agent pipeline과 데모를 보여준다. 파이프라인은 대체로 **Data Agent → Program 생성 → Execute → Data Tools와 상호작용** 구조다. 이 agent는 데이터 질의, 시각화, 테이블 조작, 이미지 처리 등 광범위한 데이터 작업을 목표로 한다.

#### 3.5.2 Plugins Agent

Plugins Agent는 쇼핑, 검색, 뉴스, 날씨, 여행 등 일상적인 작업을 위해 **200개 이상의 plugin/API**를 통합한다. 사용자는 하나 또는 여러 plugin을 직접 고를 수 있고, 적절한 plugin을 모르더라도 **automatic tool selection** 기능이 이를 대신한다. 논문은 RapidAPI와 OpenAI plugin store 등에서 API provider 정보를 수집하고, 실제 LLM 기반 사용 가능성을 검증하며 도구를 확장했다고 설명한다.

Figure 4(7페이지)는 Plugins Agent의 pipeline과 예시를 보여준다. 구조는 **Plugins Agent → Retriever/API Call → Plugins** 형태로 제시된다. 핵심은 수많은 도구 중 무엇을 써야 할지 사용자 대신 추론하는 기능이다.

#### 3.5.3 Web Agent

Web Agent는 웹 브라우징을 담당한다. 여기서 중요한 설계는 **chat agent와 browse agent의 분리**다. 사용자의 복잡한 요구를 chat agent가 먼저 정리하고, 필요 시 browse agent에게 세부 sub-task를 넘긴다. 이로써 웹 작업을 더 명확한 하위 단계로 분해할 수 있고, chat과 browsing이 교차하는 multi-turn interaction도 가능해진다.

Figure 5(7페이지)는 Web Agent pipeline을 보여주며, 실제 브라우저 조작이 포함된다. 이 구조는 “web browsing module”과 “chat reasoning module”을 분리해 각자의 발전을 독립적으로 가능하게 한다는 점도 장점으로 제시된다.

### 3.6 Prompting as Core Mechanism

논문 부록 C는 각 agent의 핵심 prompt를 상당히 길게 공개한다. 이를 보면 OpenAgents의 중심 기술이 새로운 신경망 구조가 아니라, **정교하게 설계된 prompting + output formatting + execution control**임을 알 수 있다. 예를 들어 Data Agent prompt는 도구 호출 시 JSON schema를 강제하고, 악의적 코드나 보안상 위험한 작업을 거부하도록 지시한다. Plugins Agent와 Web Agent도 마찬가지로, 도구 호출 형식과 응답 포맷, 추가 plugin 사용 조건 등을 상세히 규정한다.

즉, 이 논문은 LLM agent 품질이 단순 모델 능력뿐 아니라, **prompt engineering과 application logic의 축적**에 크게 좌우된다는 점을 실증적으로 보여준다.

## 4. Experiments and Findings

이 논문은 전통적인 benchmark 점수 중심의 대규모 정량 실험보다는, **플랫폼 제안과 사례 중심 분석**에 가깝다. 그럼에도 몇 가지 중요한 주장과 비교가 있다.

### 4.1 비교 관점

표 1은 OpenAgents를 다른 기존 프레임워크들과 비교한다. 여기서 강조되는 비교 축은 다음과 같다.

* online 배포 가능 여부
* human feedback 지원 여부
* UI 제공 여부
* coding/tool/web capability
* tool 수
* wild vs controlled environment

이 비교에서 OpenAgents는 웹 UI를 제공하고, 온라인 호스팅 가능하며, 200개 이상의 도구와 웹 브라우징 기능을 갖춘, 상대적으로 **실서비스 지향적인 종합 플랫폼**으로 위치된다. 특히 controlled와 wild 환경을 모두 지원한다는 점이 강조된다.

### 4.2 논문이 실제로 보여주는 것

논문은 세 agent의 사용 예시를 다수 제공한다.

* Data Agent: Kaggle dataset search, Python code execution, SQL query, interactive ECharts visualization, image operations
* Plugins Agent: 쇼핑, 날씨, 개념 시각화, Wolfram Alpha 질의, Zapier workflow, 여행 planning, automatic plugin selection
* Web Agent: 항공편 검색, IMDb 코멘트 요약, Google Maps 경로 탐색, Twitter 포스팅, Google Form 작성

부록 B와 각 figure는 OpenAgents가 단일 유형의 task만이 아니라, **데이터 작업 / API 도구 작업 / 웹 인터랙션 작업**이라는 세 범주의 실제 사용자 과업을 포괄하려는 플랫폼임을 보여준다.

### 4.3 Research-to-Deployment 관찰 결과

논문 8페이지는 실제 배포 과정에서 드러난 중요한 관찰을 정리한다.

첫째, **실제 앱을 위한 prompting은 매우 길고 복잡해진다.** 출력 형식, 미적 구성, 안전 제약 등을 모두 prompt에 넣다 보면 수백 토큰에 달하며, 이는 instruction following과 context length에 부담을 준다.

둘째, **현실 세계의 불안정성**이 agent 성능을 크게 좌우한다. API 서버 장애, 사용자의 중간 개입, CAPTCHA, 광고에 의한 웹 구조 변화 등은 기존 benchmark가 충분히 모델링하지 못한 요소다.

셋째, **정확도 외의 메트릭**도 중요하다. streaming, 예쁜 응답 형식, 응답 속도, 실패 시 사용자 경험 등은 실제 사용성에 큰 영향을 준다. 논문은 정확도만 높은데 체감 UX가 나쁜 시스템은 실무적으로 한계가 있다고 본다.

넷째, **평가 복잡성**이 커진다. 실패가 LLM 자체의 한계인지, 애플리케이션 로직 문제인지 구분하기 어려워진다. 예를 들어 파일 다운로드 실패는 agent 지능의 부족이 아니라 시스템 미지원 문제일 수 있다. 이는 application-level evaluation의 난점을 잘 보여준다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 **language agent 연구를 소프트웨어 플랫폼 문제로 확장**했다는 점이다. 많은 연구가 prompting, reasoning, tool-use accuracy에 초점을 맞출 때, OpenAgents는 UI, 데이터 저장, 스트리밍, 장애 처리, 브라우저 조작, 사용자 개입 가능성까지 묶어서 다룬다. 이 덕분에 “에이전트를 실제 서비스로 만들려면 무엇이 필요한가”를 비교적 구체적으로 보여준다.

또 다른 강점은 **세 가지 agent를 한 플랫폼 안에 통합**했다는 점이다. Data/Plugins/Web는 사용 시나리오가 꽤 다른데, 이를 공통 인터페이스와 공통 인프라 위에서 제공함으로써 연구용 testbed로서도 의미를 가진다.

또한 부록에서 prompt와 executor 설계를 공개해, 단지 결과만 보여주는 것이 아니라 **재현 가능하고 확장 가능한 오픈 플랫폼**으로서의 성격을 분명히 한다는 점도 장점이다.

### Limitations

한계도 분명하다.

첫째, 논문은 플랫폼 설계와 사례 제시에 강하지만, **엄밀한 정량 평가나 ablation**은 상대적으로 약하다. 예를 들어 Data Agent, Plugins Agent, Web Agent 각각의 성공률, 오류 유형, 비용/지연 trade-off를 체계적으로 수치화한 실험은 제한적이다. 이 논문은 benchmark paper보다는 system/demo paper에 더 가깝다.

둘째, 핵심 동작이 상당 부분 **prompt engineering**에 의존한다. 부록 C를 보면 프롬프트가 매우 길고 세부 규칙이 많다. 이는 유연성과 빠른 개발에는 유리하지만, 모델이 바뀌거나 환경이 달라졌을 때 유지보수 부담이 크고, 동작 안정성이 prompt 품질에 크게 좌우될 수 있다.

셋째, 200개 이상의 plugin 확장과 실제 웹 환경 제어는 강력하지만, 동시에 **실제 서비스 인프라 품질**에 매우 민감하다. 논문 스스로도 API 불안정성, 웹페이지 변화, 사용자 개입, rate limit 등을 주요 어려움으로 인정한다. 즉, OpenAgents는 문제를 해결했다기보다, 그 문제들을 명확히 드러낸 플랫폼이라고 보는 편이 정확하다.

### Brief Critical Interpretation

비판적으로 읽으면, 이 논문의 진짜 공헌은 “새로운 agent algorithm”이 아니라 **agent engineering methodology**다. 다시 말해 OpenAgents는 연구 커뮤니티에 다음 메시지를 던진다.

**Agent를 실제로 쓰게 만들고 싶다면, 모델 reasoning만 보지 말고 UI, execution, storage, streaming, failure recovery, evaluation design을 함께 보라.**

이 관점은 이후 등장한 다수의 agent 플랫폼과 제품형 AI 시스템을 이해하는 데도 유용하다. 그래서 이 논문은 성능 SoTA 논문이라기보다, **실전형 LLM agent platform의 초창기 설계 청사진**으로 읽는 것이 적절하다.

## 6. Conclusion

OpenAgents는 LLM 기반 language agent를 실제 사용자 환경으로 끌어오기 위한 오픈소스 플랫폼이다. 논문은 세 가지 대표 에이전트인 Data Agent, Plugins Agent, Web Agent를 제안하고, 이를 웹 UI, 백엔드, tool interface, 실행 환경, 스트리밍, 오류 처리, 데이터 저장, 브라우저 확장 등과 결합해 **현실 배포 가능성**을 보여준다.

이 논문이 중요한 이유는, language agent 연구가 단순한 benchmark 성능 경쟁을 넘어 **실제 서비스 환경의 복잡성**을 마주해야 한다는 점을 분명히 했기 때문이다. 따라서 OpenAgents는 앞으로의 agent 연구에서 특히 다음 영역에 의미가 크다.

* real-world agent platform design
* human-in-the-loop evaluation
* tool-augmented LLM systems
* deployable web/data/plugin agents

실무적으로는 “일반 사용자가 쓸 수 있는 agent product”를 만들고자 할 때 참고할 만한 초기 설계서이고, 연구적으로는 “controlled benchmark를 넘는 in-the-wild evaluation”의 필요성을 제기한 시스템 논문으로 볼 수 있다.
