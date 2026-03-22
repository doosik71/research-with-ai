# Active Learning of Inverse Models with Intrinsically Motivated Goal Exploration in Robots

이 논문은 로봇이 고차원·중복(redundant) sensorimotor space에서 **inverse model** 을 더 효율적으로 학습하려면, 저수준 actuator space에서 무작위로 motor babbling을 하는 대신 **task space에서 goal을 능동적으로 생성하고 선택해야 한다**고 주장한다. 이를 위해 저자들은 **SAGG-RIAC (Self-Adaptive Goal Generation - Robust Intelligent Adaptive Curiosity)**라는 다층 active learning 아키텍처를 제안한다. 핵심은 로봇이 스스로 목표(goal)를 만들어 보고, 그 목표에 대한 **competence progress** 가 큰 영역을 더 자주 탐색하게 만드는 것이다. 논문은 이 방식이 단순한 random goal selection이나 actuator-space 탐색보다 더 빠르게 inverse model을 익히고, 로봇이 **무엇을 할 수 있고 무엇을 할 수 없는지**, 즉 reachable space의 경계를 스스로 발견하게 만든다고 보고한다.

## 1. Paper Overview

이 논문이 다루는 문제는 로봇의 **inverse model learning** 이다. forward model이 “이 행동을 하면 어떤 결과가 나오는가”를 학습하는 것이라면, inverse model은 “원하는 결과를 얻으려면 어떤 행동 정책을 써야 하는가”를 학습하는 것이다. 실제 로봇에서는 이 문제가 특히 어렵다. sensorimotor space가 매우 크고, 연속적이며, redundancy가 존재하고, 실제 세계에서 데이터를 수집하는 비용도 높기 때문이다. 저자들은 이런 상황에서 단순 random exploration만으로는 충분한 품질의 데이터를 얻기 어렵고, 따라서 **탐색 자체를 능동적으로 조직하는 메커니즘** 이 필요하다고 본다.

논문의 문제의식은 기존 active learning이나 curiosity-driven learning과도 연결된다. 하지만 저자들은 기존 uncertainty maximization이나 prediction error 기반 active learning이 개방형 로봇 환경에서는 비효율적일 수 있다고 본다. 이유는 로봇의 세계가 너무 커서 전부 배울 수 없고, 어떤 부분은 본질적으로 unlearnable하거나 unreachable하며, noise도 균일하지 않기 때문이다. 따라서 이 논문은 “가장 불확실한 곳을 찾는 것”보다 **학습이 실제로 진전되는 곳**, 즉 competence progress가 큰 task 영역을 찾는 것이 더 적절하다고 주장한다.

## 2. Core Idea

논문의 핵심 아이디어는 두 가지다.

첫째, **actuator space가 아니라 task space에서 탐색하라**는 것이다. 로봇은 보통 매우 많은 motor parameter를 가지지만, 실제로 달성하고 싶은 것은 end-effector 위치, 이동 방향, 낚싯줄 도달 위치 같은 상대적으로 저차원의 task다. 따라서 task space에서 goal을 고르면, 같은 효과를 내는 수많은 redundant action variation을 불필요하게 모두 탐색하지 않아도 된다. 논문은 이것이 “같은 목표를 수행하는 많은 방법”을 배우기보다, “더 많은 서로 다른 목표”를 더 빨리 배우게 해 준다고 설명한다.

둘째, goal selection의 기준을 **competence progress** 로 삼는 것이다. 단순히 현재 실패가 큰 곳이나 uncertainty가 큰 곳이 아니라, 최근 시도들에서 **도달 능력이 얼마나 개선되고 있는지**를 측정해서 그 값이 큰 영역을 더 흥미로운 영역으로 본다. 이 방식은 로봇이 너무 쉬운 곳이나 너무 어려운 곳에 머무르지 않고, 점점 학습 가능한 난이도의 목표들로 이동하게 만든다. 저자들은 이것이 developmental trajectory를 형성한다고 해석한다.  

이 아이디어는 abstract에서도 직접 요약된다. SAGG-RIAC는 high-dimensional redundant robot에서 inverse model을 능동적으로 학습하게 하며, parameterized task를 task space에서 샘플링하고, 각 goal은 motor policy parameter를 찾는 저수준 goal-directed learning을 촉발한다. 또한 regression을 이용해 이미 학습한 task-policy correspondence를 바탕으로 새 목표에 대한 policy parameter를 일반화한다.

## 3. Detailed Method Explanation

### 3.1 문제 설정: inverse model과 task space

논문은 로봇 시스템을 state/context space $S$, action 또는 motor policy space $A$, 그리고 task/effect space $Y$ 로 본다. forward model은 대체로 $(S,\pi_\theta)\rightarrow Y$ 형태이고, inverse model은 원하는 결과 $Y$ 와 현재 상태 $S$ 가 주어졌을 때 그것을 달성할 policy $\pi_\theta$ 를 찾는 문제다. 저자들이 특히 강조하는 점은, 실제 로봇에서는 inverse model을 직접 global하게 배우기 어렵고, redundancy 때문에 동일한 task를 달성하는 여러 action이 존재한다는 것이다. 그래서 전체 actuator space를 무작정 덮는 방식은 sample inefficient하다.

### 3.2 SAGG-RIAC의 2계층 구조

논문은 SAGG-RIAC를 **두 개의 time scale** 을 가진 active learning 구조로 설명한다.

1. 상위 계층은 **goal/task를 스스로 생성하고 선택**한다.
2. 하위 계층은 선택된 goal을 실제로 달성하기 위해 **저수준 action을 탐색**한다.

상위 계층의 역할은 “다음에 무엇을 시도할 것인가”를 정하는 것이고, 하위 계층의 역할은 “그 목표를 어떻게 달성할 것인가”를 배우는 것이다. 이 분리는 논문 전체의 핵심이다. 기존 RIAC가 action-policy 공간에서 forward model 학습을 유도했다면, SAGG-RIAC는 이를 task space 목표 생성 문제로 옮겨 **inverse model learning에 맞게 재구성**했다.

### 3.3 왜 task-space exploration이 중요한가

저자들은 knowledge-based exploration이나 actuator-space active learning이 redundant robot에서는 비효율적이라고 본다. 이유는 같은 효과를 내는 action variation이 너무 많아, 알고리즘이 같은 결과를 내는 여러 action에 시간을 낭비하게 되기 때문이다. 논문은 이를 “공을 앞으로 미는 10가지 방법을 배우는 대신, 공을 10방향으로 미는 법을 배우는 편이 낫다”는 식으로 설명한다. 이 때문에 goal을 task space 수준에서 정의하고, 그 goal을 중심으로 exploration을 조직하는 것이 더 효율적이다.

### 3.4 Competence progress 기반 관심도(interestingness)

SAGG-RIAC에서 가장 중요한 선택 기준은 **competence progress** 다. 특정 goal 근방의 competence가 빠르게 증가하고 있으면, 그 지역은 현재 학습하기에 적절한 난이도와 정보량을 가진 영역으로 본다. 반대로 이미 충분히 mastered된 영역이나 전혀 진전이 없는 영역은 덜 흥미로운 영역이 된다. 논문은 이것이 RIAC의 prediction error reduction 아이디어를 **task space에서 goal competence improvement** 로 옮긴 것이라고 설명한다. 또한 competence 자체는 보통 목표 $y_g$ 와 실제 도달 결과 $y_f$ 의 차이로 정의된다. 제공된 텍스트 조각에는 예시로 다음과 같은 형태가 나온다.

$$
C(y_g, y_f, \emptyset) = - |y_g - y_f|^2
$$

즉, 목표와 실제 결과가 가까울수록 competence가 높고, 그 competence의 시간적 변화가 progress가 된다.

이 기준이 중요한 이유는 전통적인 uncertainty-driven active learning과 달리, **영원히 어려운 영역에 매달리지 않게 해 준다**는 데 있다. 논문은 실제 로봇 환경에는 본질적으로 unreachable하거나 noisy한 영역이 존재하므로, 단순 uncertainty나 prediction error는 쉽게 함정에 빠질 수 있다고 지적한다. competence progress는 이 문제를 완화한다.

### 3.5 Regression과 local inversion

SAGG-RIAC는 모든 목표에 대해 완전한 analytic inverse model을 구하는 것이 아니라, 이미 경험한 goal-policy correspondence를 축적하고, regression 기법을 이용해 **새로운 목표에 대한 motor policy parameter를 추론**한다. 즉, active exploration으로 task space의 유용한 부분을 채워 나가고, 거기서 local inverse/forward structure를 형성한다. 그래서 논문은 이 방법이 “complete forward model”을 배우지 않아도 reachable task 대부분을 달성하는 데 충분한 sub-part를 익힐 수 있다고 설명한다.

### 3.6 실험 비교군

논문은 실험에서 task-space와 actuator-space를 각각 random/active 방식으로 탐색하는 여러 방법을 비교한다. 본문 조각상 대표 비교군은 다음 흐름으로 이해할 수 있다.

* **ACTUATOR-RANDOM**: actuator space에서 random motor babbling
* **ACTUATOR-RIAC**: actuator space에서 RIAC-like active exploration
* **SAGG-RANDOM**: task space에서 random goal generation
* **SAGG-RIAC**: task space에서 competence progress 기반 goal generation

이 비교 설계 덕분에 논문은 “task-space exploration 자체의 효과”와 “competence-progress 기반 active selection의 추가 효과”를 분리해 보여 준다.  

## 4. Experiments and Findings

논문은 세 가지 로봇 설정에서 실험한다.

1. **고중복 로봇 팔의 inverse kinematics 학습**
2. **사족보행 로봇의 omnidirectional locomotion 학습**
3. **유연한 낚싯줄이 달린 팔의 fishing control 학습**

### 4.1 Experiment 1: Redundant arm inverse kinematics

이 실험의 핵심 메시지는 명확하다. **goal/operational space에서의 babbling이 actuator space에서의 babbling보다 훨씬 효율적**이라는 것이다. 논문은 highly-redundant arm의 inverse kinematics를 배울 때, ACTUATOR-RANDOM이나 ACTUATOR-RIAC보다 SAGG 계열이 전반적으로 더 빠르고 더 좋은 inverse model을 만든다고 보고한다. 특히 ACTUATOR-RANDOM이 ACTUATOR-RIAC보다 나은 경우까지 나타나는데, 이는 원래 RIAC가 high-dimensional actuator space에서 inverse model 학습을 위해 설계된 알고리즘이 아님을 보여 준다고 해석한다.

또한 같은 task-space 탐색 안에서도 **SAGG-RIAC가 SAGG-RANDOM보다 더 빠르고 더 좋은 generalization** 을 보인다. 저자들은 이것을 SAGG-RIAC가 competence progress가 큰 영역을 점진적으로 식별하고 거기에 집중할 수 있기 때문이라고 해석한다.

흥미로운 점은 task space가 reachable region보다 훨씬 더 크게 설정된 경우다. 이때 많은 candidate goal이 물리적으로 unreachable해지는데, 논문은 **SAGG-RIAC만이 이런 high-volume task space에서도 효율적인 학습을 유지했다**고 말한다. 반면 SAGG-RANDOM은 unreachable goal을 시도하는 데 많은 시간을 소모한다. 이는 competence progress 기반 goal discrimination이 실제로 reachability discovery에 도움이 된다는 강한 증거다.  

또한 arm DOF가 7, 15, 30으로 증가하는 비교에서도 SAGG-RIAC는 일관되게 강한 성능을 보였고, 특히 7 DOF에서도 SAGG-RIAC가 가장 낮은 reaching error를 보였다고 보고된다. 저자들은 이 결과를 통해 redundancy와 geometry가 달라져도 이 접근이 유효하다고 주장한다.

### 4.2 Experiment 2: Quadruped locomotion

제공된 조각에는 이 실험의 세부 수치가 완전하게 드러나진 않지만, abstract와 본문 구조를 보면 저자들의 메시지는 일관된다. quadruped에서도 task-space goal exploration과 competence-progress 기반 선택이 random selection보다 더 효율적이며, developmental trajectory를 형성한다는 것이다. 이 실험은 특히 motor synergies를 이용하는 고차원 control space에서도, task space가 상대적으로 저차원이라면 SAGG-RIAC가 여전히 유효하다는 점을 뒷받침한다.  

### 4.3 Experiment 3: Fishing rod control

낚싯대와 유연한 줄을 다루는 실험은 논문에서 가장 인상적인 사례 중 하나다. 이 설정은 geometry가 비대칭이고 compliant/soft dynamics가 있어 analytic model을 만들기 어렵다. 논문은 10,000번의 “water touched” trial 후 도달한 float 위치의 histogram을 비교했을 때, **SAGG-RIAC가 ACTUATOR-RANDOM보다 reachable space 전체를 더 고르게 탐색**했다고 설명한다. 저자들은 이것이 복잡하고 비대칭적인 reachable structure를 더 잘 드러낸다고 해석한다.  

정량적으로도 SAGG-RIAC가 더 좋다. Figure 21 설명에 따르면, 10개의 random seed로 반복한 결과 **1000 successful trials 이후 SAGG-RIAC가 유의미하게 더 효율적**이었다. 6000 trial 이후 reaching error가 약간 증가하는 구간이 있는데, 저자들은 이를 새로운 motor synergy redundancy를 발견하는 과정에서 inverse model이 잠시 모호해졌기 때문이라고 설명한다. 즉, 새로운 방식으로 이미 mastered된 goal에 도달하는 경우가 생기면, 그 local inverse mapping이 충분히 분리되어 학습되기 전까지 잠시 generalization이 흔들릴 수 있다는 것이다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **문제 설정을 정확히 바꿨다**는 점이다. 기존 active learning이 주로 “어떤 action을 시도할 것인가”에 초점을 맞췄다면, 이 논문은 redundant robot에서 진짜 중요한 것은 “어떤 goal을 시도할 것인가”라고 본다. 이 전환은 매우 설득력 있고, 실제 실험에서도 actuator-space exploration보다 task-space exploration이 훨씬 효율적이라는 결과로 뒷받침된다.  

둘째, **competence progress** 라는 기준이 매우 실용적이다. 단순 prediction error나 uncertainty는 로봇이 forever-unreachable 영역에 빠질 수 있지만, competence progress는 학습이 실제로 진전되는 구간을 찾게 해 준다. 이 때문에 로봇이 스스로 적절한 난이도의 과제로 이동하는 developmental trajectory가 생긴다.  

셋째, 실험이 단순 inverse kinematics에 그치지 않고 quadruped locomotion과 fishing control까지 포함한다. 즉, rigid arm뿐 아니라 motor synergy, compliant dynamics, asymmetric reachable set 같은 복잡한 조건에서도 아이디어가 통한다는 점을 보여 준다.  

### 한계

한계도 있다.

첫째, 접근이 **task space 정의에 의존**한다. 즉, 로봇에게 어떤 effect space를 task space로 둘 것인지가 어느 정도는 설계자의 선택이며, 이 정의가 적절하지 않으면 goal exploration의 효율도 떨어질 수 있다. 논문은 다양한 예시를 보이지만, task parameterization 자체를 자동으로 배우는 문제까지 풀지는 않는다. 이 점은 이후 representation learning 기반 goal discovery 연구로 이어질 여지가 있다.

둘째, competence progress는 매우 강력한 휴리스틱이지만, 그 계산이 결국 지역별 competence 추정 품질에 달려 있다. 저자들도 RIAC 계열이 sampling density와 차원 저주 문제에 취약하다고 지적하는데, SAGG-RIAC는 task space로 차원을 낮춰 이를 완화하지만, task space 자체가 커지면 여전히 discrimination 문제는 남는다. 다만 논문은 큰 task space에서도 상당한 robustness를 보였다고 주장한다.  

셋째, 이 논문은 현대 딥러닝 이전의 developmental robotics 맥락에 있으므로, function approximator 자체는 최근의 deep latent policy/goal-conditioned RL보다 비교적 고전적이다. 따라서 오늘날 관점에서는 “representation learning 없는 goal exploration”이라는 한계가 있다. 그러나 그 개념적 통찰은 여전히 강하다.

### 해석

비판적으로 해석하면, 이 논문의 진짜 기여는 특정 알고리즘보다 **탐색의 수준(level)을 바꿨다**는 데 있다. 로봇 학습에서 정말 중요한 것은 actuator-level novelty가 아니라, **task-level capability growth** 라는 것이다. 그래서 이 논문은 modern goal-conditioned RL, curriculum learning, intrinsically motivated exploration, skill discovery의 여러 흐름에 선구적인 아이디어를 제공한 논문으로 읽을 수 있다.

## 6. Conclusion

이 논문은 SAGG-RIAC라는 intrinsically motivated active learning 아키텍처를 제안해, high-dimensional redundant robot에서 inverse model을 효율적으로 학습하는 방법을 제시한다. 핵심은 **task space에서 goal을 능동적으로 생성**하고, 그중에서도 **competence progress가 큰 영역**을 더 자주 탐색하는 것이다. 이를 통해 로봇은 actuator space에서 같은 효과를 내는 수많은 redundant action을 헤매기보다, 더 다양한 task를 더 빨리 배우게 된다. 또한 이 과정에서 reachable/unreachable 영역을 스스로 구분하고, 점진적으로 더 복잡한 과제로 이동하는 developmental trajectory를 형성한다.

실험적으로도 저자들은 redundant arm inverse kinematics, quadruped locomotion, fishing rod control에서 일관되게 SAGG-RIAC의 장점을 보였다고 보고한다. 특히 large task space나 asymmetric reachable set에서도 이 방법이 random actuator exploration보다 훨씬 효율적이었다는 점은 중요하다. 따라서 이 논문은 단순한 robotics active learning 논문이 아니라, **goal-directed autonomous exploration** 의 설계 원리를 정리한 foundational paper로 볼 수 있다.  
