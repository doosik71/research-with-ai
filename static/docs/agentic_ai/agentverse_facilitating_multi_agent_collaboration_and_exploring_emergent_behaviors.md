# AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors

## 1. Paper Overview

이 논문은 **LLM 기반 autonomous agent를 “여럿이 협업하는 시스템”으로 확장했을 때 어떤 구조가 효과적인지**를 다루는 연구다. 저자들은 단일 agent가 이미 강력한 언어 이해, 추론, 코딩, 도구 사용 능력을 보이지만, 실제 세계의 복잡한 문제는 종종 여러 역할의 협업을 필요로 한다는 점에서 출발한다. 기존 multi-agent 연구는 특정 과제에만 초점을 맞추거나, agent 역할이 고정되어 있어 일반화와 적응성이 부족했다. 이를 해결하기 위해 저자들은 인간 집단의 문제 해결 과정을 모사하는 범용 multi-agent 프레임워크 **AgentVerse**를 제안한다. 이 프레임워크는 단순히 agent를 여러 개 두는 것이 아니라, 문제 상태에 따라 팀 구성을 바꾸고, 집단 의사결정을 수행하고, 실행 결과를 평가하여 다음 라운드에 반영하는 순환 구조를 갖는다. 논문은 이 구조가 텍스트 이해, 추론, 코딩, tool use, embodied AI 전반에서 단일 agent보다 우수한 성능을 보이며, 더 나아가 협업 과정에서 volunteer, conformity, destructive behavior 같은 emergent behavior도 나타난다고 주장한다.

이 연구가 중요한 이유는 두 가지다. 첫째, “LLM을 하나 더 크게 만드는 것”이 아니라 **여러 agent의 사회적 상호작용 자체를 시스템 설계 요소로 끌어들였다**는 점이다. 둘째, 단순 benchmark 향상에 그치지 않고, 협업 과정에서 발생하는 바람직한 행동과 위험한 행동을 함께 분석했다는 점에서 이후 multi-agent AI 시스템 설계의 출발점 역할을 할 수 있다.  

---

## 2. Core Idea

이 논문의 핵심 직관은 다음과 같다.

> 복잡한 문제를 잘 풀기 위해서는 “똑똑한 agent 하나”보다, 상황에 맞는 전문가 집단을 구성하고 그들이 토론하고 실행하고 피드백을 받는 구조가 더 효과적일 수 있다.

즉, AgentVerse의 본질은 **human team problem-solving process를 LLM agent 시스템으로 번역한 것**이다. 저자들은 문제 해결 과정을 네 단계로 나눈다.

1. **Expert Recruitment**: 현재 목표와 진행 상황에 맞는 전문가 agent를 구성
2. **Collaborative Decision-Making**: agent들이 토론하거나 계층적으로 검토하며 의사결정
3. **Action Execution**: 실제 행동이나 도구 사용 수행
4. **Evaluation**: 현재 상태와 목표의 차이를 평가하고 다음 라운드에 반영  

이 아이디어의 참신성은 두 부분에 있다.

첫째, 기존 역할 기반 multi-agent 시스템과 달리 **역할을 고정하지 않고 동적으로 생성한다**. 즉, 특정 task에 필요한 전문가 설명을 recruiter가 생성하고, 그에 따라 팀이 구성된다. 둘째, 실패했을 때 같은 팀이 반복해서 같은 실수를 하는 것이 아니라, **evaluation의 feedback을 다음 recruitment 단계로 다시 흘려 보내 팀 자체를 재구성**한다. 이 점에서 AgentVerse는 단순 ensemble이나 voting이 아니라, “구성-토론-실행-반성”이 반복되는 적응형 집단 시스템이다.  

---

## 3. Detailed Method Explanation

### 3.1 전체 구조: 인간 집단 문제 해결 절차의 모사

AgentVerse는 전체 과정을 하나의 Markov decision process로 본다. 논문은 이를

$$
(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \mathcal{G})
$$

로 표현한다. 여기서 $\mathcal{S}$는 agent와 환경을 포함한 상태 공간, $\mathcal{A}$는 해결책 또는 행동 공간, $\mathcal{T}$는 상태 전이 함수, $\mathcal{R}$은 평가/피드백 함수, $\mathcal{G}$는 목표 공간이다. 이 수식은 강화학습을 하겠다는 뜻이라기보다, **문제 해결 과정을 상태-행동-피드백 순환 구조로 정식화**하기 위한 것이다. 즉, AgentVerse는 매 라운드마다 현재 상태를 보고, 적절한 집단 결정을 만든 뒤, 그것을 실행하고, 목표와의 차이를 다시 측정한다.

이 관점이 중요한 이유는, multi-agent 협업을 단순 대화가 아니라 **환경과 상호작용하며 반복적으로 개선하는 closed-loop system**으로 본다는 데 있다.

---

### 3.2 Expert Recruitment: 어떤 agent들이 팀에 들어와야 하는가

논문이 강조하는 첫 번째 핵심 모듈은 **Expert Recruitment**다. 저자들은 팀 구성 자체가 성능의 상한을 결정한다고 본다. 인간 집단 연구에서 다양성이 성능에 영향을 주는 것처럼, multi-agent 시스템도 서로 다른 역할과 관점을 가진 agent 조합이 더 나은 문제 해결을 가능하게 한다고 본다. 기존 방법들은 agent 역할을 미리 수동으로 정해 두는 경우가 많았는데, 이는 task를 미리 잘 이해하고 있어야 하므로 확장성이 떨어진다.

AgentVerse에서는 특정 목표 $g \in \mathcal{G}$가 주어지면 recruiter 역할을 맡은 agent $M_r$가 이 목표를 보고 필요한 전문가 설명들을 생성한다. 논문은 이를 다음과 같이 쓴다.

$$
\mathcal{M} = M_r(g)
$$

즉, 목표 $g$에 따라 전문가 집합 $\mathcal{M}$이 생성된다는 뜻이다. 중요한 점은 **전문가 역할을 사전 정의된 고정 목록에서 꺼내오는 것이 아니라, 목표 자체로부터 생성한다**는 것이다. 예를 들어 어떤 task가 코드 작성이라면 programmer, tester, UI designer 같은 역할이 필요할 수 있고, 어떤 task가 상담이라면 domain expert와 evaluator가 필요할 수 있다. 논문은 이 과정을 사람이 직접 설계하는 대신 agent가 수행하게 만들어 더 다양한 문제에 대응하도록 한다.  

또한 recruitment는 한 번으로 끝나지 않는다. evaluation 단계에서 얻은 feedback이 다음 라운드 recruitment로 다시 들어가므로, **현재 상태에 적합한 새로운 팀으로 동적으로 재편성**될 수 있다. 이 점이 AgentVerse의 적응성을 만든다.

---

### 3.3 Collaborative Decision-Making: 여러 agent가 어떻게 의사결정을 만드는가

AgentVerse는 협업 구조를 하나로 고정하지 않고, 두 가지 전형적인 communication structure를 제시한다.

#### (1) Horizontal Structure

Horizontal structure는 민주적 구조에 가깝다. 각 agent $m_i \in \mathcal{M}$가 자신의 판단 $a_{m_i}$를 제시하고, 최종 집단 결정 $A$는 이를 통합하는 함수 $f$를 통해 얻는다.

$$
A = f({a_{m_i}}_i) \in \mathcal{A}
$$

여기서 $f$는 예를 들어 summarization이나 ensemble에 해당할 수 있다. 중요한 것은 **여러 agent가 동등한 위치에서 의견을 내고, 그 집합으로부터 최종 결론을 도출한다**는 점이다. 논문은 이 구조가 consulting이나 tool-use처럼 여러 관점과 정보 조각을 합치는 작업에 적합하다고 설명한다.

#### (2) Vertical Structure

Vertical structure는 solver와 reviewer가 나뉘는 계층적 구조다. solver agent $m^*$가 초기 해답 $a_0^*$를 만들고, 다른 agent들은 reviewer로서 피드백을 준다. 이후 solver는 이를 반영해 여러 차례 refinement를 수행하고, 최종적으로

$$
A = a_k^* \in \mathcal{A}
$$

를 출력한다. 여기서 $k$는 refinement 횟수다. 이 구조는 수학 문제 풀이, 소프트웨어 개발처럼 **최종적으로 하나의 정교한 해답을 계속 다듬어야 하는 과제**에 적합하다.

두 구조를 함께 제시한 것은 중요하다. 논문의 메시지는 “multi-agent면 무조건 토론하면 된다”가 아니라, **task 특성에 따라 의사결정 topology 자체를 선택해야 한다**는 것이다.

---

### 3.4 Action Execution: 계획을 실제 행동으로 옮기기

Collaborative Decision-Making 단계에서 만들어진 집단 결정 $A$는 실제 환경에서 실행된다. 논문은 이 과정을 상태 전이 함수로

$$
s_{\text{new}} = \mathcal{T}(s_{\text{old}}, A)
$$

로 표현한다. 즉, 현재 상태 $s_{\text{old}}$에서 집단이 합의한 행동 $A$를 수행하면 새로운 상태 $s_{\text{new}}$가 만들어진다. 여기서 행동은 텍스트 답변 생성일 수도 있고, 코드를 작성하고 수정하는 것일 수도 있으며, 브라우저나 계산기 같은 외부 도구를 호출하는 것일 수도 있다. 일부 agent는 decision에만 참여하고 execution은 하지 않을 수도 있다는 점도 논문이 명시한다.

이 모듈은 AgentVerse가 단순한 “대화 시뮬레이션”이 아니라 **환경과 상호작용하는 agent system**임을 보여준다.

---

### 3.5 Evaluation: 성능을 측정하고 다음 라운드를 준비하기

Evaluation 단계는 AgentVerse의 closed-loop를 완성하는 핵심이다. 논문은 평가를

$$
r = \mathcal{R}(s_{\text{new}}, g)
$$

로 표현한다. 즉, 새로운 상태 $s_{\text{new}}$와 목표 $g$의 차이를 평가해 feedback $r$를 만든다. 이 feedback은 단순 scalar reward라기보다, **무엇이 부족했고 어떻게 개선할 수 있는지에 대한 verbal feedback**일 수 있다. 논문은 이 evaluator가 human-in-the-loop일 수도 있고, 자동화된 agent일 수도 있다고 말한다.

목표가 아직 달성되지 않았다면, 이 feedback은 다시 recruitment 단계로 돌아가고, 다음 라운드에서는 목표뿐 아니라 현재의 실패 양상까지 고려해 팀 구성이 달라진다. 이 구조 때문에 AgentVerse는 “한 번의 다중 토론”이 아니라, **집단 구성 자체를 수정해 가며 문제를 풀어 가는 iterative adaptive system**이 된다.

---

## 4. Experiments and Findings

### 4.1 실험 설정

논문은 네 가지 축에서 AgentVerse를 평가한다.

* **General understanding and reasoning**
* **Coding**
* **Tool utilization**
* **Embodied AI**

모든 정량 실험은 zero-shot setting에서 수행되며, GPT-3.5-Turbo-0613과 GPT-4-0613을 사용한다. 비교 대상은 세 가지다.

* **CoT**: chain-of-thought를 사용하는 단일 agent
* **Solo**: AgentVerse 구조는 쓰되, decision-making에 agent 한 명만 쓰는 경우
* **Group**: 여러 agent가 decision-making에 실제로 협업하는 경우

이 설정은 매우 중요하다. 왜냐하면 논문은 단순히 “AgentVerse 전체가 CoT보다 낫다”가 아니라, **구조적 모듈의 효과(Solo)와 multi-agent 협업 자체의 추가 이득(Group)**을 분리해서 보여주려 하기 때문이다.

---

### 4.2 General Understanding and Reasoning

이 영역에서 논문은 FED, Commongen Challenge, MGSM, Logic Grid Puzzles를 사용한다. 처음 두 개는 텍스트 이해와 creative writing, 뒤 두 개는 수학 및 논리 추론을 평가한다.  

결과적으로, **AgentVerse 기반 설정(Solo, Group)은 전반적으로 standalone CoT보다 일관되게 우수**했다. 특히 GPT-4 계열에서는 협업 의사결정이 더 강하게 작동했다. 예를 들어 MGSM에서는 GPT-4 기준 CoT 95.2, Solo 96.0, Group 95.2가 아니라? 스니펫상 95.2, 96.0, 95.2가 보이지만 맥락상 표 전체 해석은 제한적이다. 다만 Logic Grid Puzzles에서는 GPT-4 기준 59.5 → 64.0 → 66.5로 Group이 가장 높았고, MGSM에서는 최고값이 96.0으로 나타난다. 따라서 적어도 일부 reasoning task에서 multi-agent 또는 AgentVerse 구조가 실제 정량 향상을 보였다고 볼 수 있다. 다만 제공된 스니펫이 표 전체를 완전하게 보여주지는 않아 일부 값의 열 대응은 조심해서 읽어야 한다.

흥미로운 점은 GPT-3.5에서는 항상 Group이 좋은 것이 아니라는 점이다. 논문은 GPT-3.5-Turbo가 일부 경우 Solo보다 못한 이유를 **잘못된 피드백에 대한 취약성**으로 해석한다. 한 agent가 맞는 답을 시작했더라도 다른 agent의 틀린 피드백에 설득되어 잘못된 방향으로 수정되는 일이 있었고, MGSM 오류의 약 10%가 이런 현상에서 비롯됐다고 분석한다. GPT-4 기반 agent에서는 이런 현상이 두드러지지 않았다. 이는 multi-agent 협업이 “언제나 더 낫다”가 아니라, **기저 모델이 상충 정보에 얼마나 강건한가**에 따라 협업 효과가 달라짐을 시사한다.  

---

### 4.3 Coding Capabilities

코딩 실험에서 논문은 Group setup의 이점을 강조한다. 제공된 표 스니펫에서는 성능이 73.8, 83.5에서 Solo 74.4, 87.2, Group 75.6, 89.0으로 증가하는 양상이 보이며, Group이 최고치를 기록한다. 저자들은 reasoning에서는 GPT-3.5가 잘못된 피드백에 흔들렸지만, coding에서는 오히려 multi-agent discussion이 이점을 보여 준다고 말한다. 그 이유로 LLM이 코드에 대해 더 많이 사전학습되어 있어, 코딩 영역에서는 잘못된 정보에 대한 내성이 더 크기 때문일 수 있다고 추정한다.

또한 논문은 단순 정답률(pass@1) 이상의 질적 향상을 강조한다. HumanEval에서 Group setup이 생성한 코드는 단지 “맞는 코드”가 아니라, **더 효율적이고, 더 견고하며, 더 안전한 알고리즘**으로 다듬어졌다고 설명한다. 즉, reviewer 역할의 agent들이 성능, robustness, UI/UX, 테스트 관점에서 서로 다른 피드백을 주면서 결과물의 품질을 끌어올린다는 것이다. 사례 연구로 Python GUI calculator 개발 예시를 제시하며, Group 결과물은 Solo와 달리 색상 구분, 키보드 입력, 더 나은 사용자 경험, 더 견고한 코드 구조를 갖췄다고 분석한다. 여기서 multi-agent의 장점은 “정답을 맞히는가”를 넘어, **실제 소프트웨어 개발처럼 다면적 품질을 개선하는 협업**에 있다는 점이다.

---

### 4.4 Tool Utilization

Tool use 실험은 AgentVerse의 강점을 가장 직관적으로 보여 주는 부분이다. 논문은 계산기, 웹 브라우저, 코드 인터프리터, task-specific API 등 여러 도구를 agent들에게 제공하고, **최소 두 종류 이상의 도구를 함께 써야 하는 10개의 복잡한 과제**를 설계한다. 여기서 Group 기반 AgentVerse는 10개 중 9개를 해결하지만, 단일 ReAct agent는 3개만 해결한다. ReAct가 실패한 7개 중 6개는 과제의 여러 요구사항 중 일부를 놓치거나 너무 일찍 종료했기 때문이라고 설명한다.  

이 결과가 의미하는 바는 단순히 “더 많은 agent가 있으니 검색을 더 잘한다”가 아니다. 논문은 AgentVerse가 **큰 문제를 manageable sub-task로 분해하고, 각 sub-task에 적합한 tool을 배치하며, 결과를 다시 통합하는 orchestration 능력**을 가진다고 본다. 예시로 제시된 사례에서는 Group이 24-point game의 규칙과 solver code, test case뿐 아니라 유사 게임 요약까지 제공한 반면, 단일 ReAct agent는 일부 요구사항만 만족했다. 즉, multi-agent 구조는 tool use에서 특히 중요한 **요구사항 커버리지와 작업 지속성**을 개선한다.

---

### 4.5 Embodied AI와 Emergent Behaviors

논문은 Minecraft를 사용해 embodied AI 맥락에서 emergent behavior를 분석한다. 여기서 agent들은 단순히 답변을 생성하는 것이 아니라, 종이, 그림, 책, 책장 등을 함께 제작하며 계획·협업·환경 적응을 수행한다. 저자들은 이 환경이 현실 세계의 사회적 상호작용과 유사한 복잡성을 제공하므로, multi-agent의 집단 행동을 관찰하기 적합하다고 본다.  

논문이 보고한 대표적 emergent behavior는 세 가지다.

#### Volunteer behaviors

다른 agent의 효용을 높이기 위해 자발적으로 도움을 주는 행동이다. 예를 들어 Bob이 놀고 있는 시간을 발견하고, 혼자 기다리지 말고 함께 sugar cane을 모으자고 제안해 전체 작업 시간을 줄이는 사례가 제시된다. 이는 시스템이 명시적으로 “이타적 행동”을 프로그래밍하지 않았더라도, 협업 효율을 높이는 방향으로 자발적 기여가 생길 수 있음을 뜻한다.

#### Conformity behaviors

다른 agent들의 비판을 받고 자신의 벗어난 행동을 팀 목표에 맞게 수정하는 행동이다. 이는 협업 과정에서 social correction 메커니즘이 자연스럽게 발생할 수 있음을 보여 준다.

#### Destructive behaviors

반대로 협업이 항상 긍정적이지는 않다. 잘못된 상호작용이나 부정확한 피드백은 오히려 성능 저하나 원치 않는 결과를 낳을 수 있다. reasoning 실험에서 GPT-3.5가 잘못된 피드백에 설득되는 문제는 이러한 destructive tendency의 정량적 예라고 볼 수 있다.  

이 부분은 논문의 중요한 차별점이다. 많은 시스템 논문이 성능 향상만 보고하는 데 반해, 이 논문은 **협업의 부작용과 위험까지 함께 보고**한다.

---

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 **multi-agent 협업을 범용 프레임워크 수준으로 추상화했다는 점**이다. recruitment, decision, execution, evaluation으로 나눈 구조는 task 종류가 달라도 재사용 가능하다. 또한 단순히 여러 agent를 병렬 호출하는 것이 아니라, 역할 생성과 feedback loop를 통해 **동적 적응성**을 확보했다. 텍스트 이해, 추론, 코드, 도구 사용, embodied AI까지 폭넓게 실험했다는 점도 설득력을 높인다.  

또 다른 강점은 **질적 분석의 깊이**다. 특히 coding과 Minecraft 사례에서, 정확도 숫자만으로 포착되지 않는 협업의 실제 이점을 잘 보여 준다. 예를 들어 코드의 robustness, UI 품질, 요구사항 충족 범위 같은 요소는 실제 응용에서 매우 중요한데, 논문은 이런 부분을 multi-agent의 강점으로 해석한다.

### Limitations

하지만 한계도 분명하다. 첫째, multi-agent 협업은 비용과 latency를 증가시킬 가능성이 높다. 논문은 구조의 효용을 강조하지만, 여러 agent의 반복적 토론과 재구성은 실제 시스템 운영비 측면에서 부담이 될 수 있다. 이 점은 논문에서 중심적으로 정량화되지는 않는다.

둘째, **협업은 기저 LLM의 robustness에 크게 의존**한다. GPT-3.5에서 나타난 erroneous feedback 문제는, multi-agent 구조가 약한 모델 위에서는 오히려 취약점을 증폭시킬 수 있음을 보여 준다. 즉, 협업은 자동으로 성능을 올려 주는 만능 해법이 아니다.

셋째, evaluation과 recruitment가 대부분 자연어 기반 피드백에 의존하기 때문에, 이 feedback이 편향되거나 잘못되면 다음 라운드의 팀 구성 자체가 왜곡될 수 있다. 논문은 이 위험을 destructive behavior와 연결해 암시하지만, 이를 제어하는 엄격한 이론적 장치는 아직 충분히 제시하지 않는다.

### Interpretation

비판적으로 보면, 이 논문은 “성능 향상 논문”인 동시에 “LLM society 설계의 초기 청사진”에 가깝다. 진짜 기여는 개별 benchmark 수치보다, **agent 집단이 어떤 구조에서 어떻게 더 똑똑해지거나 더 위험해질 수 있는가**를 보여 준 데 있다. 향후 이 방향은 agent orchestration, AI teamwork, distributed cognition 연구로 이어질 가능성이 크다.

---

## 6. Conclusion

이 논문은 AgentVerse라는 multi-agent 프레임워크를 제안하여, LLM agent 협업을 인간 팀의 문제 해결 절차처럼 설계할 수 있음을 보여 준다. 핵심은 **동적 expert recruitment, task에 맞는 collaborative decision-making 구조, 실제 action execution, 그리고 feedback 기반 evaluation loop**다. 이 구조는 일반 이해·추론, 코딩, tool use, embodied AI에서 단일 agent를 능가하는 성능을 보였고, 동시에 volunteer, conformity, destructive behavior 같은 emergent social behavior도 드러냈다. 저자들은 이를 통해 multi-agent interaction이 앞으로 AGI 시대에 점점 더 중요해질 것이라고 주장한다. 논문 전체를 관통하는 메시지는 명확하다. 미래의 지능 시스템은 “더 큰 모델 하나”가 아니라, **상황에 맞게 구성되고 서로 교정하며 함께 행동하는 agent 집단**이 될 수 있다는 것이다.
