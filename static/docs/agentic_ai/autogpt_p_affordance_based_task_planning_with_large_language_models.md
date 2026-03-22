# AutoGPT+P: Affordance-based Task Planning with Large Language Models

## 1. Paper Overview

이 논문은 로봇 task planning에서 LLM을 활용하되, LLM의 약한 추론 능력을 그대로 믿지 않고 **affordance-based scene representation + classical planner + LLM-based tool selection**을 결합한 **AutoGPT+P**를 제안한다. 문제의식은 분명하다. 기존 LLM 기반 로봇 planning은 자연어 명령을 더 일반적으로 다룰 수 있게 해 주지만, 실제 장면의 초기 상태를 동적으로 반영하기 어렵고, 누락된 물체나 불완전한 정보가 있을 때 취약하다. 저자들은 이를 해결하기 위해 장면을 단순한 object list가 아니라 **객체와 affordance의 쌍**으로 표현하고, 이 표현으로부터 planning domain과 initial state를 구성해 symbolic planning을 수행한다. 그 결과, 닫힌 세계(closed-world)뿐 아니라 **missing objects**가 있는 상황에서도 탐색, 대체물 제안, partial plan 생성까지 지원한다. 초록 기준으로 SayCan instruction set에서 **98%** 성공률을 기록해 SayCan의 **81%**를 넘어섰고, 저자들이 만든 150개 시나리오 데이터셋에서는 **79%** 성공률을 보였다.  

이 문제가 중요한 이유는, 실제 서비스 로봇 환경에서는 사용자가 “우유 한 잔 가져다줘”처럼 자연어로 요청하더라도, 필요한 물체가 장면에 없거나 다른 대체물이 있을 수 있기 때문이다. 논문은 예로 컵 대신 유리잔이 없을 때 컵을 대체물로 제안하고, 우유병을 여는 것이 어려우면 사람에게 도움을 요청하는 상황까지 언급한다. 즉, 단순한 plan generation이 아니라 **현실 환경의 불완전성과 기능적 대체 가능성**을 planning에 통합하려는 시도다.

## 2. Core Idea

핵심 아이디어는 **“객체 이름”이 아니라 “객체가 무엇을 할 수 있는가”를 planning의 중심 표현으로 삼자**는 것이다. AutoGPT+P는 scene을 object-affordance-pair의 집합으로 표현하고, 이로부터 planning에 필요한 action 가능성과 substitution 가능성을 추론한다. 예를 들어 knife는 cutting, grasping, stirring 같은 affordance를 가진다. 이런 표현을 사용하면 planner가 특정 object identity에 덜 묶이고, **같은 affordance를 공유하는 다른 객체로 대체**하는 reasoning이 가능해진다.  

논문의 novelty는 세 가지 층위에 있다. 첫째, **Object Affordance Mapping (OAM)**을 ChatGPT로 자동 생성해 object class와 affordance를 연결한다. 둘째, 장면에서 감지된 객체와 OAM을 결합해 **affordance-based symbolic scene representation**을 만든다. 셋째, LLM을 planner 자체로 쓰기보다, **tool selection**과 **goal generation/correction**에 활용하고, 실제 plan 생성은 classical planner가 수행하도록 하는 hybrid 구성을 택한다. 저자들은 이를 통해 LLM의 번역/도구선택 능력은 활용하고, 약한 논리 추론은 planner와 외부 checker로 보완한다.

## 3. Detailed Method Explanation

### 3.1 전체 구조

AutoGPT+P는 크게 두 단계로 구성된다.

첫 번째 단계는 **환경 인지와 affordance 추출**이다. RGB 이미지에서 object detection을 수행하고, 감지된 object classes를 OAM에 통과시켜 각 객체의 affordance를 부여한다. 두 번째 단계는 **task planning**이다. 사용자 자연어 요청을 받아 goal을 만들고, scene representation을 바탕으로 planner가 plan을 찾는다. 여기에 LLM은 직접 전체 plan을 생성하기보다, 상황에 따라 어떤 도구를 써야 할지를 선택하는 상위 제어기로 동작한다.  

논문은 이 접근을 기존 taxonomy에서 **Step-By-Step Autoregressive Plan Generation**과 **LLM With Planner**의 hybrid라고 명시한다. 즉, tool selection은 step-by-step한 LLM feedback loop로 처리하고, 핵심 planning은 planner 기반으로 수행한다. 이 구성이 AutoGPT+P의 설계 철학을 잘 보여준다. LLM 단독 planner의 불안정성을 피하면서도 자연어 이해와 유연한 제어는 유지하려는 것이다.

### 3.2 Affordance-based scene representation

논문은 장면 $S$를 object-affordance-pairs의 집합으로 정의한다. 각 원소는 대략 다음과 같은 형태다.

$$
S={p_1,\dots,p_n}, \qquad
p_i=(o_i, k_i, a_i, b_i)
$$

여기서 $o_i$는 object class, $k_i$는 instance index, $a_i$는 affordance, $b_i$는 bounding box다. 즉, 하나의 장면은 “무엇이 있는가”만이 아니라 “각 객체가 어떤 affordance를 가지는가”까지 포함한 symbolic state로 바뀐다. 저자들은 scene space를 이 tuple들의 power set으로 정의하고, Object Affordance Detection을 이미지 공간에서 이 scene space로의 사상으로 formalize한다.

이 표현의 의미는 크다. 기존 grounding 방식처럼 단순히 “scene에 컵, 병, 칼이 있다”를 prompt에 붙이는 것이 아니라, planner가 직접 사용할 수 있는 **기능 중심 state representation**을 만든다. 그래서 같은 affordance를 가진 물체는 implicit substitution 후보가 되고, planning domain도 더 유연해진다. 논문 Table I도 이 점을 강조한다. 저자들은 자신들의 방법이 **16개의 affordance**, **explicit and implicit substitution**, **long horizon tasks**를 동시에 다룬다고 비교한다.  

### 3.3 Object detection과 OAM

Object Affordance Detection은 두 단계다. 먼저 object detector가 scene에서 객체들을 찾는다. 이후 **Object Affordance Mapping (OAM)**이 object class를 대응하는 affordance 집합으로 매핑한다. 저자들은 이 OAM을 ChatGPT로 자동 생성한다고 설명한다. 즉, 각 object class에 대해 사람이 직접 affordance 테이블을 수작업으로 만들지 않고, LLM을 사용해 자동으로 매핑을 구축한다. 논문 contribution에도 “automatic object-affordance mapping using ChatGPT”가 명시돼 있다.

이 방식은 장점과 한계를 동시에 가진다. 장점은 새로운 everyday objects에 대해 빠르게 affordance dictionary를 넓힐 수 있다는 점이다. 단순 수작업 ontology보다 확장성이 좋다. 반면 정확도는 OAM 품질에 의존하며, 저자들도 후반부에서 probabilistic OAM으로 확장해야 한다고 말한다. 즉, 현재는 deterministic symbolic mapping이지만, 미래에는 confidence-aware representation이 필요하다고 본다.

### 3.4 Tool selection feedback loop

AutoGPT+P의 LLM은 자연어 메모리와 현재 상태를 보고 어떤 도구를 쓸지 선택한다. visible snippet에 따르면 주요 도구는 다음과 같다.

* **Plan**: 일반적인 planning 수행
* **Partial Plan**: 현재 장면 제약하에서 가능한 최선의 부분 계획 생성
* **Suggest Alternative Tool**: 누락된 물체 대신 쓸 대체물 제안
* 그 외 탐색/실행 관련 도구들

즉, 시스템은 처음부터 “plan만 세운다”가 아니라, 상황에 따라 **계획 / 부분 계획 / 대체 제안 / 탐색** 중 무엇을 할지 LLM이 고른다. 이 구조 덕분에 missing object 상황에서 유연하게 대응할 수 있다.  

### 3.5 Planner와 self-correction

논문에서 특히 중요한 것은 LLM을 planner의 대체재가 아니라 **goal generator + correction agent**로 쓴다는 점이다. 저자들은 사용자 자연어 요청으로부터 전체 problem을 생성하지 않고, 주로 **goal state를 PDDL syntax로 생성**한다. 초기 상태는 장면 표현으로부터 직접 유도되므로, 기존 LLM+P류처럼 문제 전체를 언어에서 생성할 필요가 없다.

생성된 goal은 바로 planner에 넘기지 않고, **syntactic and semantic correctness**를 검사한다. 오류가 있으면 LLM에 error message를 다시 입력해 self-correction을 수행한다. 논문은 core planning tool이 기존 작업을 확장해 **semantic 및 syntactic 오류를 자동 수정**한다고 설명한다. 이는 LLM이 PDDL goal을 생성할 때 생기기 쉬운 형식 오류와 의미 오류를 외부 checker로 보정하는 구조다. 저자들은 이 외부 피드백 루프가 planning 성공률 향상에 중요하다고 본다.  

### 3.6 Missing objects와 substitution

AutoGPT+P의 차별점은 누락된 물체가 있을 때의 동작이다. 닫힌 세계 가정의 planner는 보통 필요한 object가 없으면 실패한다. 그러나 이 논문은 affordance를 이용해 두 종류의 대체를 다룬다.

첫째는 **implicit substitution**이다. 같은 affordance를 공유하는 object로 자동 대체하는 방식이다. 예를 들어 glass가 없으면 cup으로 대체할 수 있다.
둘째는 **explicit substitution**이다. 사용자가 요구한 물체와 정확히 다른 물체를 기능적으로 바꾸는 planning reasoning이다.

저자들은 자신들의 방법이 everyday long-horizon task에서 implicit와 explicit substitution을 동시에 다루는 첫 접근이라고 주장한다.  

## 4. Experiments and Findings

### 4.1 평가 구성

논문은 평가를 세 단계로 나눈다.

첫째, **OAM 자체의 품질** 평가
둘째, **Suggest Alternative Tool** 평가
셋째, **Plan Tool** 평가 및 전체 AutoGPT+P 시스템 평가

OAM 평가는 독립된 30 object class train set으로 prompt를 최적화한 뒤, 70 object class test set으로 수행한다. 이들은 40개의 제안 affordance에 대해 labeled되어 있고, same set이 Plan tool과 Suggest Alternative Tool 평가에도 사용된다. 즉, OAM은 단순 qualitative demo가 아니라 별도 held-out object class로 평가된다.  

### 4.2 Plan Tool 평가

Plan Tool은 두 종류의 시나리오에서 평가된다. 첫 번째는 **SayCan instruction set**으로, 기존 SOTA와 직접 비교하기 위한 것이다. 두 번째는 저자들이 만든 자체 시나리오 세트로, LLM의 사용자 의도 이해 한계를 분석하기 위한 것이다. 이 자체 세트는 5개의 subset으로 구성되며 각 subset은 30개 prompt를 포함한다.

* **Simple Task**
* **Simple Goal**
* **Complex Goal**
* **Knowledge**
* **Implicit**

여기서 Knowledge는 commonsense knowledge가 필요한 목표 이해를, Implicit는 “I am thirsty”처럼 직접 task가 표현되지 않은 사용자 의도를 다룬다. 이 설계는 단순 plan success만이 아니라, **goal interpretation difficulty**를 세분화해 분석하려는 의도를 보여준다.

### 4.3 주요 결과

초록의 핵심 수치는 매우 강하다.
AutoGPT+P는 **SayCan instruction set에서 98% 성공률**을 기록해 **SayCan의 81%**를 넘어섰다. 또한 저자들이 만든 **150 scenarios** 데이터셋에서는 **79% 성공률**을 기록했다. 이 데이터셋은 pick-and-place, handover, pouring, chopping, heating, wiping, sorting 등 폭넓은 task와 missing object 상황을 포함한다.

이 결과가 의미 있는 이유는 두 가지다. 첫째, 기존 대표 LLM-based planner와의 직접 비교에서 이겼다. 둘째, 단순 closed-world benchmark가 아니라 missing objects와 long-horizon tasks가 있는 더 어려운 자체 세트에서도 꽤 높은 성능을 보였다. 특히 논문은 단순한 short manipulation이 아니라 **everyday long-horizon tasks**에 초점을 둔다.  

### 4.4 실험이 실제로 보여주는 것

이 논문이 실험으로 보여주는 것은 “LLM을 planner로 직접 쓰지 않는 것이 더 낫다”는 메시지에 가깝다. LLM은 자연어 goal extraction, tool selection, self-correction에는 유용하지만, long-horizon symbolic reasoning은 planner가 더 안정적이라는 것이다. 또한 affordance representation이 없으면 missing object 상황에서 실패하기 쉬운데, AutoGPT+P는 partial plan이나 alternative suggestion으로 이를 우회한다. 따라서 성능 향상의 핵심은 단순 prompt engineering이 아니라, **representation + hybrid planning architecture + feedback correction**의 조합이라고 보는 것이 맞다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **affordance-based grounding**이다. 많은 LLM planning 연구가 scene object list를 prompt에 붙이는 수준에 머무르는데, 이 논문은 장면을 planner가 직접 쓸 수 있는 symbolic affordance state로 바꾼다. 이 덕분에 substitution reasoning과 incomplete information handling이 가능해진다.

두 번째 강점은 **hybrid architecture**다. LLM의 자연어 이해와 도구선택 능력은 살리되, 약한 추론 능력은 classical planner와 checker로 보완한다. 논문 결과가 SayCan보다 높은 것은 단순 모델 성능 때문이 아니라, 구조적으로 더 안정적인 역할 분담 덕분이라고 해석할 수 있다.  

세 번째 강점은 **missing objects에 대한 실용적 대응**이다. plan 실패로 끝나는 것이 아니라, alternative suggestion, partial planning, human help request로 이어진다. 실제 로봇 시스템에서는 이 유연성이 매우 중요하다.

### 한계

가장 명확한 한계는 **tool selection**이다. 결론 snippet에서 저자들은 어려움이 주로 LLM이 잘못된 tool을 선택하는 데서 발생한다고 인정한다. 즉, planner 자체보다 상위 orchestration이 병목이다. 이는 시스템이 유연해질수록 “어떤 모드를 실행할 것인가”가 더 어려워진다는 의미다.

둘째, 현재 OAM과 scene representation은 본질적으로 **deterministic**이다. 저자들은 future work로 probabilistic OAM, object detection confidence를 포함한 probabilistic scene representation, 그리고 success probability를 최적화하는 planner의 필요성을 언급한다. 이는 현재 시스템이 perception uncertainty와 execution uncertainty를 충분히 모델링하지 못함을 뜻한다.

셋째, Table I에서 AutoGPT+P는 long horizon과 substitution에서는 강하지만 **novel classes는 지원하지 않는다**고 표시된다. 즉, unseen object classes를 인식하고 바로 affordance reasoning으로 일반화하는 수준까지는 아직 아니다. OAM 자동화가 확장성은 주지만, object detector와 class ontology의 범위를 벗어나면 한계가 생긴다.

### 해석

비판적으로 보면, 이 논문의 진짜 기여는 “LLM이 planning을 잘한다”가 아니다. 오히려 **LLM이 planning 파이프라인에서 어디에 써야 하는지**를 잘 보여준 논문이다. 자연어 요청 해석, goal generation, self-correction, tool routing에는 유용하지만, long-horizon symbolic search는 planner가 더 적합하다는 점을 설계 수준에서 드러낸다. 따라서 이 논문은 end-to-end LLM robotics보다 **structured neuro-symbolic orchestration**에 가까운 작업으로 이해하는 편이 정확하다.

## 6. Conclusion

이 논문은 **AutoGPT+P**를 통해 affordance-based scene representation과 classical planning, 그리고 LLM-based tool selection을 결합한 hybrid robotic task planning 시스템을 제안한다. 핵심은 object detection과 ChatGPT 기반 OAM으로부터 장면의 affordance 구조를 만들고, 그 위에서 planner가 goal-directed plan을 생성하며, missing object 상황에서는 alternative suggestion, exploration, partial plan으로 대응하는 것이다. 실험적으로는 SayCan instruction set에서 **98%** 성공률로 SayCan의 **81%**를 넘었고, 저자들이 만든 150개 복합 시나리오에서도 **79%** 성공률을 보였다.

실무적으로는, everyday service robot이 불완전한 환경에서 사용자 자연어 명령을 처리하는 데 중요한 방향을 제시한다. 연구적으로는 LLM을 planner 자체로 쓰는 대신, **affordance grounding + symbolic planning + LLM orchestration**이라는 역할 분담이 더 강력할 수 있음을 보여준다. 다만 tool selection 오류와 probabilistic uncertainty modeling 부족은 남아 있으며, 저자들도 이를 미래 과제로 명확히 인정한다.
