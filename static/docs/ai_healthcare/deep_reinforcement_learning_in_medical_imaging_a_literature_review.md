# Deep reinforcement learning in medical imaging: A literature review

이 보고서는 사용자가 첨부한 ar5iv HTML 원문을 바탕으로 작성했다. 이 논문은 DRL(Deep Reinforcement Learning)의 기본 이론을 먼저 정리한 뒤, medical imaging에서의 활용을 크게 세 부류로 나눠 리뷰한다. 첫째는 landmark detection, object/lesion detection, registration, view plane localization 같은 **parametric medical image analysis**, 둘째는 hyperparameter tuning, augmentation strategy selection, neural architecture search 같은 **optimization 문제 해결**, 셋째는 surgical gesture segmentation, personalized mobile health intervention, computational model personalization 같은 **miscellaneous application**이다. 논문은 최종적으로 이 분야의 future perspective와 challenge도 함께 정리한다.  

## 1. Paper Overview

이 논문의 목표는 의료 영상 분야에서 DRL이 어디에, 왜, 어떻게 쓰이고 있는지를 체계적으로 정리하는 것이다. 저자들은 단순히 응용 사례를 나열하지 않고, 먼저 RL과 DRL의 개념적 기초를 설명한 뒤, 실제 의료 영상 문제에 DRL을 적용할 때 어떤 formulation이 사용되는지, 기존 방식 대비 어떤 장점이 있는지, 그리고 아직 어떤 한계가 남아 있는지를 연결해서 설명한다. 논문은 특히 medical imaging에서 DRL이 아직 완전히 개척되지 않은 영역이라고 보고, 체계적 이해 부족이 적용 확산을 막는 원인 중 하나라고 본다.

연구 문제가 중요한 이유도 분명하다. 의료 영상은 보통 3D 혹은 고차원 데이터이고, 검색 공간이 크며, 비분화(non-differentiable) 목적함수나 sequential decision 문제를 자주 포함한다. 예를 들어 landmark를 찾기 위해 전체 볼륨을 exhaustive scan하는 것은 비효율적이고, registration에서는 non-convex optimization이 자주 등장한다. DRL은 이런 상황에서 “어떤 순서로 어떤 행동을 취해야 최종 성능이 좋아지는가”를 학습할 수 있기 때문에 기존 supervised learning이나 hand-crafted optimization으로 다루기 어려운 문제를 풀 수 있는 잠재력이 있다.  

또한 이 논문은 단순 응용 survey가 아니라 tutorial 성격도 강하다. 논문 구조 자체가 RL basics, DRL algorithms, medical imaging applications, future challenges 순으로 되어 있어, 의료영상 연구자가 DRL에 입문하면서 동시에 응용 사례를 훑을 수 있도록 설계되어 있다. 이런 점에서 “리뷰 논문”이면서도 “분야 입문서” 역할을 같이 한다고 볼 수 있다.

## 2. Core Idea

논문의 핵심 아이디어는 medical imaging 문제를 단순 prediction/classification 문제가 아니라 **sequential decision-making problem**으로 재해석할 수 있다는 점이다. 즉, 의료 영상 분석에서 많은 작업은 한 번에 정답을 예측하는 대신, 현재 상태를 보고 다음 행동을 결정하고, 그 누적 결과로 최종 목표에 도달하는 과정으로 볼 수 있다. landmark localization이라면 현재 voxel 위치에서 어느 방향으로 한 칸 이동할지 결정하는 문제이고, registration이라면 현재 transformation parameter에서 어떤 조정 action을 취할지 결정하는 문제다.  

이 논문이 강조하는 또 하나의 중심 관점은 DRL이 의료 영상에서 특히 유리한 이유가 세 가지라는 점이다. 첫째, delayed reward를 다룰 수 있다. 둘째, non-differentiable objective도 optimization 대상으로 삼을 수 있다. 셋째, 고차원 의료 영상을 전부 동시에 처리하지 않고 작은 region 혹은 현재 관심 상태를 중심으로 탐색하게 만들어 memory burden을 줄일 수 있다. 논문 서론은 이런 특성을 DRL의 본질적 강점으로 제시한다.

새로움은 survey의 범위 설정에 있다. 저자들은 기존 DRL 자체를 새로 제안하지는 않지만, medical imaging에서의 DRL 응용을 **parametric image analysis / optimization / miscellaneous**라는 세 범주로 조직해 공통 구조를 보여준다. 또한 RL 이론부터 model-free/model-based 알고리즘까지 함께 설명해, 단지 사례 모음집이 아니라 응용을 이해하기 위한 개념 틀을 제공한다.  

## 3. Detailed Method Explanation

이 논문은 survey이므로 하나의 새로운 알고리즘이나 손실함수를 제안하지 않는다. 대신 DRL의 기본 수학적 프레임과 의료 영상 응용에서 반복적으로 나타나는 problem formulation을 정리한다.

### 3.1 RL과 MDP의 기본 틀

논문은 RL을 MDP(Markov Decision Process) 위에서 설명한다. MDP는 상태 공간 $S$, 행동 공간 $\mathcal{A}$, 전이 확률 $T(s_{t+1}\mid s_t,a_t)$, 보상 함수 $R$, discount factor $\gamma$로 정의된다. 에이전트는 상태 $s_t$에서 정책 $\pi(a_t\mid s_t)$에 따라 행동 $a_t$를 선택하고, 환경은 보상 $r_{t+1}$와 다음 상태 $s_{t+1}$를 반환한다. 이 반복을 통해 trajectory가 형성되고, 에이전트는 누적 보상을 최대화하는 정책을 학습한다. 논문은 이를 DRL 이해의 기본 골격으로 둔다.

개념적으로 return은 보통 다음처럼 쓸 수 있다.

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

그리고 목표는 기대 누적 보상을 최대화하는 정책 $\pi$를 찾는 것이다. 논문은 Bellman equation, policy, transition, trajectory, episode 같은 RL 핵심 용어를 정리하며 독자가 이후 응용 섹션을 이해할 수 있도록 만든다.

### 3.2 Model-based vs. Model-free RL

논문은 RL 접근을 크게 model-based와 model-free로 나눈다. Figure 2는 RL을 model-based와 model-free로 나누고, model-based에는 value functions, transition models, policy search, return functions를, model-free에는 policy gradient, value-based, actor-critic을 배치한다. 또한 Table 1은 두 계열을 비교한다. survey 전반에서 실제 medical imaging 응용은 대체로 **model-free learning**을 더 많이 사용한다고 명시한다.

이 점은 중요하다. 의료 영상 문제는 환경 모델을 명시적으로 잘 정의하기 어려운 경우가 많고, 대신 샘플 기반 trial-and-error로 policy 혹은 value를 직접 학습하는 편이 실용적이기 때문이다. 따라서 논문이 model-based도 설명하긴 하지만, 실제 응용 파트의 중심은 DQN, DDQN, actor-critic, policy-gradient 류의 model-free DRL에 가깝다.  

### 3.3 DRL for parametric medical image analysis

논문에서 가장 중심적인 method 파트는 Section 4.1이다. 여기서 저자들은 landmark detection, image registration, view plane localization 등 여러 문제를 공통적으로 “현재 state를 보고 다음 action을 택해 목표 위치 혹은 최적 parameter에 도달하는 sequential search”로 모델링한다.

예를 들어 anatomical landmark detection에서는 landmark를 3D point로 보고, 행동은 left, right, up, down, forward, backward 같은 voxel 이동으로 구성된다. 멀티스케일 구조에서는 coarse scale에서 global context를 이용해 대략적인 방향을 잡고, fine scale로 갈수록 local refinement를 수행한다. 이때 각 scale마다 Q-function을 학습해 exhaustive volumetric scanning 대신 효율적 navigation을 수행한다. 논문은 trajectory가 작은 oscillatory cycle로 수렴하면 탐색을 종료하고 해당 위치를 detection 결과로 기록하는 방식도 설명한다.

image registration의 경우도 유사하다. rigid registration은 보통 6개 정도의 transformation parameter를 최적화해야 하는데, 전통적 방법은 normalized correlation coefficient나 mutual information 같은 matching metric을 직접 최적화한다. 그러나 이 문제는 비볼록적이라 local optimum가 많다. 논문이 소개하는 DRL 접근은 registration을 “최적 motion action sequence를 찾는 과정”으로 바꾸고, 입력으로는 3D raw image와 현재 registration parameter 추정을 넣고, 출력은 다음 최적 action을 예측하게 한다. path supervision을 사용한 end-to-end training으로 기존 문제 특화 hand-crafted optimization보다 image quality나 artifact에 덜 민감하다는 장점을 제시한다.

vessel centerline extraction 같은 사례에서는 DDQN과 3D dilated CNN을 결합한 구조가 소개된다. 한 branch는 다음 action을 예측하는 tracker 역할을 하고, 다른 branch는 artery branch point와 radius를 추정한다. 하나의 seed만으로 전체 coronary tree를 추출하며, CAT08 challenge에서 state-of-the-art 수준 성능과 빠른 추론 속도를 보였다고 정리한다.

### 3.4 DRL for optimization tasks

논문은 DRL이 non-differentiable metric을 다룰 수 있다는 점을 활용해 optimization task에도 많이 쓰인다고 설명한다. 여기에는 radiotherapy planning hyperparameter tuning, image augmentation selection, neural architecture search, acquisition strategy learning이 포함된다. 핵심은 exhaustive grid search 대신, 에이전트가 “좋은 탐색 정책”을 배우도록 만드는 것이다.  

이 formulation의 의미는 크다. 예를 들어 architecture search는 discrete design choice의 조합으로 이뤄져 gradient-based optimization이 직접적이지 않다. augmentation selection 또한 조합 공간이 넓다. DRL은 이들을 일련의 action 선택 문제로 바꾸어 sequential optimization을 수행하게 한다. 논문은 medical imaging 응용 대부분이 parametric analysis에 집중되어 있지만, optimization 영역도 빠르게 확장되고 있다고 본다.

### 3.5 Miscellaneous applications

miscellaneous 카테고리에는 surgical gesture segmentation, personalized mobile health intervention, computational model personalization 등이 포함된다. 특히 computational model personalization에서는 agent가 offline에서 모델의 parameter 변화에 따른 동작을 탐색하며 decision process를 학습하고, online personalization에서 더 robust하고 빠른 성능을 낸 사례가 소개된다. cardiac electrophysiology inverse problem 및 whole-body circulation personalization에서 표준 방법과 동등하거나 더 나은 결과를 보였고, 최대 11% 높은 success rate와 최대 7배 빠른 속도를 보고했다고 논문은 정리한다.

## 4. Experiments and Findings

이 논문은 survey이므로 자체적으로 하나의 통일된 실험 세트를 제시하지는 않는다. 대신 다양한 응용 논문들의 reported result를 묶어 보여준다. 따라서 실험 결과 해석은 “논문 자체의 benchmark”가 아니라 “리뷰된 사례들에서 반복적으로 관찰되는 패턴”으로 이해해야 한다.

가장 뚜렷한 finding은 DRL이 의료 영상의 **search/navigation 기반 문제**에서 강점을 보인다는 점이다. landmark detection에서는 멀티스케일 Q-function 기반 탐색이 exhaustive search보다 효율적으로 동작하고, landmark가 FOV 밖에 있을 경우 trajectory가 image space를 벗어나는 행동을 통해 “missing landmark”를 간접적으로 표현할 수 있다는 흥미로운 관찰도 소개된다. 이는 DRL이 단순 위치 예측기를 넘어 search process 자체를 모델링한다는 것을 보여준다.

registration 분야에서는 spine와 heart dataset에서 DRL 기반 registration agent를 평가했다고 정리한다. 특히 CT와 CBCT 정합처럼 local optimum가 많은 환경에서 DRL agent는 최적 motion sequence를 학습함으로써 기존 수동 설계 최적화 방식의 약점을 줄인다. 논문은 image quality 및 artifact에 민감한 기존 기법 대비 학습된 agent의 장점을 강조한다.

optimization 응용에서도 의미 있는 수치가 보인다. skin disease classification 사례에서는 CNN 분류기와 RL 기반 QA agent를 결합해 CNN-only 방식 대비 20% 이상 accuracy를 높였고, 평균 질문 수 측면에서도 decision-tree 기반 QA보다 빠르게 diagnosis를 좁혔다고 한다. 이는 DRL이 이미지 자체뿐 아니라 진단 과정 전체의 sequential decision에도 기여할 수 있음을 보여준다.

computational model personalization에서는 기존 표준 방법과 비슷한 결과를 더 빠르고 robust하게 달성했으며, 최대 11% 높은 success rate, 최대 7배 빠른 속도를 보고했다. vessel centerline extraction에서는 DDQN + 3D dilated CNN 구조가 CAT08 challenge에서 state-of-the-art 성능과 약 7초 inference를 보였다고 정리된다. 이런 사례들은 DRL이 accuracy뿐 아니라 efficiency 측면에서도 경쟁력이 있음을 시사한다.  

또 하나의 중요한 finding은 분야가 빠르게 성장하고 있다는 점이다. 논문은 Table 3, 4, 5에 top journal 및 conference 중심의 49개 참고문헌을 정리하고, Figure 8을 통해 medical imaging의 DRL 논문 수가 해마다 증가하는 trend를 보여준다고 설명한다. 동시에 대부분의 논문이 model-free algorithm을 사용한다고 강조한다.

## 5. Strengths, Limitations, and Interpretation

이 논문의 강점은 세 가지다. 첫째, RL basics부터 medical imaging application까지 단계적으로 설명해 입문자와 실무자 모두에게 유용하다. 둘째, 응용을 세 카테고리로 나누어 분야 지형을 명확하게 보여준다. 셋째, 단순 사례 나열이 아니라 왜 DRL이 medical imaging에 맞는지, 특히 sequential decision, delayed reward, non-differentiable optimization 측면에서 어떤 구조적 이점이 있는지를 잘 설명한다.  

한계도 분명하다. 우선 survey 특성상 각 방법의 실험 조건, 데이터셋 차이, 알고리즘 비교가 완전히 통일되어 있지 않다. 따라서 “어떤 DRL 알고리즘이 항상 우월하다”는 식의 결론을 내리긴 어렵다. 또한 논문 자체도 future challenge로 reward function design, high-dimensional continuous action에서의 Q-learning difficulty 등을 지적한다. 보상이 너무 delayed되면 학습이 어렵고, 각 step마다 intermediate reward를 설계하는 것도 쉽지 않다. 또한 action space가 고차원일수록 Q-function 학습이 어렵다고 정리한다.  

비판적으로 해석하면, 이 논문은 DRL을 의료 영상의 범용 만능해법으로 보지 않는다. 오히려 DRL이 특히 잘 맞는 문제 구조가 무엇인지 보여준다. landmark localization, registration, NAS처럼 탐색과 정책 학습이 중요한 문제에는 매우 적합하지만, reward design이 불명확하거나 action space가 지나치게 크면 적용이 어렵다. 따라서 이 논문이 주는 가장 실용적인 교훈은 “의료 영상 문제를 DRL로 풀 수 있느냐”가 아니라, “이 문제가 truly sequential decision problem으로 잘 표현되는가”를 먼저 판단해야 한다는 점이다.  

## 6. Conclusion

이 논문은 의료 영상에서의 DRL을 체계적으로 정리한 고품질 literature review다. DRL의 이론적 기반을 RL/MDP 관점에서 설명하고, model-free와 model-based 알고리즘을 소개한 뒤, 실제 응용을 parametric image analysis, optimization, miscellaneous application으로 나눠 폭넓게 검토한다. 특히 landmark detection, registration, view plane localization 같은 search/navigation 문제와 hyperparameter tuning, augmentation selection, NAS 같은 non-differentiable optimization 문제에서 DRL이 유망하다는 메시지가 분명하다.  

실무적 관점에서 이 논문은 “어떤 의료 영상 문제에 DRL을 고려할 만한가”를 가늠하는 지도를 제공한다. 또한 future challenge로 reward design과 high-dimensional action 문제를 지적함으로써, 단순 성공 사례 소개를 넘어 향후 연구 방향도 제시한다. 따라서 이 논문은 medical imaging 연구자가 DRL을 새로 배우거나, 자신의 문제를 DRL로 formulation할 수 있는지 판단하려 할 때 매우 유용한 출발점이다.  
