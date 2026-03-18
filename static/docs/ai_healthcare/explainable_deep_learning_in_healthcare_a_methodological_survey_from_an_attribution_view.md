# Explainable Deep Learning in Healthcare: A Methodological Survey from an Attribution View [Advanced Review]

이 논문은 healthcare에서 deep learning이 실제 임상 환경에 널리 채택되지 못하는 핵심 이유를 “black-box nature”로 보고, 이를 완화하기 위한 explainability / interpretability 방법들을 **attribution 관점**에서 체계적으로 정리한 방법론 중심 survey다. 저자들은 단순히 XAI의 필요성을 강조하는 데 그치지 않고, attribution method를 back-propagation, attention, feature perturbation, model distillation, game theory, example-based, generative 방식으로 나누어 비교하며, 각각의 장단점과 적합한 사용 시나리오, 그리고 healthcare 문제에 어떻게 적응되어 왔는지를 정리한다. 또한 마지막에는 model-agnosticity, credibility/trustworthiness, method evaluation이라는 실제 적용 단계의 핵심 쟁점을 따로 다룬다.  

## 1. Paper Overview

이 논문의 목표는 healthcare에서 사용되는 deep learning 모델의 interpretability를 방법론적으로 정리하고, 연구자나 임상의가 자신의 문제에 맞는 설명 기법을 선택할 수 있도록 실질적인 가이드를 제공하는 것이다. 논문 서두는 EHR, 의료영상, 임상 시계열 데이터의 증가와 deep learning 기술의 발전이 diagnosis, prognosis, treatment를 지원하는 clinical decision support system 개발을 촉진했지만, 실제 의료 현장에서는 모델의 판단 근거가 불투명하다는 점이 큰 장애라고 설명한다. 특히 readmission prediction, disease diagnosis, drug prescription recommendation 같은 high-stakes setting에서는 오판 비용이 매우 크기 때문에, 단순히 높은 accuracy만으로는 deployment가 어렵다고 본다.

저자들이 보는 연구 문제는 “딥러닝이 왜 그렇게 예측했는지 사람에게 납득 가능한 형태로 설명할 수 있는가”이다. 그리고 이 문제는 단지 기술적 curiosity가 아니라 실제 임상 의사결정의 신뢰성, bias 점검, 모델 디버깅, 규제 대응과 직접 연결된다. 논문은 interpretability가 algorithm designer에게는 모델을 interrogate, debug, improve하는 수단이고, end user인 clinician에게는 예측을 수용할지 거부할지 판단하는 근거라고 설명한다. 또한 GDPR의 “right to explanation” 같은 규제 맥락도 언급하며, 설명 가능성의 부재가 법적·운영상 비용으로 이어질 수 있다고 지적한다.

이 논문이 중요한 이유는 healthcare XAI literature에서 흔한 “고수준 overview”를 넘어서, 실제 explanation method를 비교·선택하는 데 필요한 세부 메커니즘과 사용 조건을 자세히 다룬다는 점이다. 저자들은 기존 survey가 definition, concept, application overview에 치우쳤다고 보고, 본 논문은 각 방법의 detail, advantage/disadvantage, suitable scenario, healthcare adaptation까지 포함하는 methodological reference를 지향한다고 분명히 말한다.  

## 2. Core Idea

논문의 핵심 아이디어는 healthcare의 explainable deep learning을 “설명 가능한 모델” 대 “설명 불가능한 모델”의 이분법으로 보지 않고, **입력 feature가 특정 예측에 얼마나 기여했는지를 추정하는 attribution problem**으로 재구성하는 데 있다. Section 2의 시작에서 저자들은 입력 $x=[x_1,\dots,x_N]$를 받아 출력 $S(x)=[S_1(x),\dots,S_C(x)]$를 만드는 DNN을 생각하고, 특정 target neuron $c$에 대해 각 입력 feature의 기여도 $R^c=[R_1^c,\dots,R_N^c]$를 구하는 것이 attribution method의 목표라고 정리한다. 이 framing 덕분에 서로 다른 explanation 방법들을 공통 언어 위에서 비교할 수 있다.

즉, 이 논문은 explainability를 “왜 이 예측이 나왔는가?”라는 질문에 대한 다양한 계산적 근사법들의 집합으로 본다. 가장 흔한 형태는 saliency heatmap처럼 input element별 중요도를 보여주는 local explanation이고, 여기에 example-based explanation이나 generative explanation도 보완적으로 포함된다. 이 관점의 장점은 이미지, 텍스트, 시계열, structured EHR 등 다양한 modality에 비교적 통일된 방식으로 접근할 수 있다는 데 있다.

또 다른 핵심 아이디어는 explanation method를 단일 우월 해법으로 보지 않는 점이다. 저자들은 method마다 class-discriminativeness, faithfulness, computational cost, architecture restriction, baseline dependence, perturbation realism 등이 다르며, 따라서 어떤 explanation이 “최선”인지는 task와 modality, 사용 목적에 따라 달라진다고 본다. 이 때문에 논문은 각 방법의 원리 설명뿐 아니라 언제 무엇을 써야 하는지에 큰 비중을 둔다.  

## 3. Detailed Method Explanation

이 논문은 새로운 단일 알고리즘을 제안하는 논문이 아니라 survey이므로, “method”는 저자들이 제안한 새 모델이라기보다 **interpretability method의 taxonomy와 그 계산 원리**를 뜻한다.

### 3.1 Attribution의 기본 정의

저자들은 먼저 attribution method를 각 input feature에 relevance 혹은 contribution score를 할당하는 방법으로 정의한다. DNN 출력 벡터 $S(x)$에서 특정 class 또는 target output $S_c$가 주어졌을 때, 목표는 각 feature $x_i$가 해당 출력에 기여한 정도 $R_i^c$를 계산하는 것이다. 보통 이 결과는 heatmap으로 시각화되며, 어떤 색은 positive contribution, 다른 색은 suppressing effect를 나타낸다. 이 정의는 이후 모든 방법을 설명하는 공통 토대가 된다.

### 3.2 Back-propagation based methods

논문에서 가장 먼저 다루는 것은 back-propagation 계열이다. 가장 기본적인 Saliency Map은 target output $S_c(x)$를 입력 $x_i$로 미분한 절댓값

$$
\left|\frac{\partial S_c(x)}{\partial x_i}\right|
$$

을 사용한다. 직관적으로는 “어떤 입력 feature를 조금만 바꿔도 출력이 많이 바뀌는가”를 보는 방식이다. 하지만 절댓값을 사용하면 positive evidence와 negative evidence를 구분하기 어렵다는 문제가 있다. 이를 개선하기 위해 Deconvolution과 Guided Back-propagation이 제안되며, 세 방법의 차이는 주로 ReLU를 통과하는 gradient를 어떻게 mask하느냐에 있다. 즉, 이 계열은 gradient flow를 조작해 더 선명하거나 더 해상도 높은 explanation map을 얻으려는 접근이다.

이후 Gradient * Input은 signed gradient에 input 자체를 곱해 attribution map의 sharpness를 높이고, Integrated Gradients는 baseline $\tilde{x}$에서 실제 입력 $x$까지 선형 경로를 따라 average gradient를 적분함으로써 attribution을 계산한다. Integrated Gradients의 핵심은 특정 한 점의 gradient가 아니라 입력이 baseline에서 현재 값으로 이동하는 동안의 누적 변화를 반영한다는 점이다. 이 때문에 단순 gradient보다 saturation 문제를 완화할 수 있다. 다만 baseline choice가 결과에 영향을 주는 구조적 한계도 함께 갖는다.

논문은 pixel-space gradient visualization과 localization 계열도 구분한다. Guided Back-prop이나 Deconv는 고해상도이지만 class-discriminative하지 않을 수 있다. 반면 CAM은 class-specific feature map을 제공하지만 CNN 구조에 제약이 있다. 이를 일반화한 Grad-CAM은 마지막 convolutional layer로 들어오는 gradient를 사용해 특정 decision에 중요한 neuron을 파악하며, Guided Grad-CAM은 high-resolution과 class-discriminativeness를 동시에 얻으려는 절충안이다. 이 구분은 의료영상처럼 “병변의 위치를 특정하는 것”이 중요한 경우에 특히 중요하다.

또 하나의 중요한 분기는 activation-value propagation이다. LRP는 출력층에서 target neuron의 relevance를 시작점으로 삼고 relevance를 이전 층으로 재분배한다. 논문에 따르면 출력층 초기화는 식 (1)처럼 target neuron에는 그 자체의 output을 부여하고 나머지는 0으로 둔다. 이후 $\epsilon$-rule을 통해 relevance를 이전 층으로 재귀적으로 전파한다. DeepLIFT는 baseline input을 통과시킨 reference activation을 이용해 relevance를 보정한다. 즉, “현재 입력이 baseline 대비 얼마나 출력을 바꿨는가”를 설명의 기준으로 삼는다. 이 계열은 gradient 자체보다 activation 차이나 relevance conservation 개념에 더 가깝다.

### 3.3 Attention, perturbation, distillation, game-theoretic methods

논문은 attribution method를 back-propagation으로만 한정하지 않는다. Section 2 전체 taxonomy에 따르면 설명 방법은 back-propagation, attention-based, feature perturbation-based, model distillation-based, game theory-based로 구분되며, completeness를 위해 example-based, generative-based explanation도 포함한다. 이 taxonomy 자체가 논문의 중요한 methodological contribution이다.

Feature perturbation 계열은 특정 feature를 가리거나 바꿨을 때 예측이 얼마나 변하는지를 본다. 직관성은 좋지만 perturbation이 data manifold 밖으로 벗어나 unrealistic sample을 만들 수 있고, 계산 비용이 크다는 문제가 있다. Game theory 계열의 대표인 SHAP는 feature contribution을 Shapley value로 해석한다. 논문은 SHAP가 널리 쓰이지만, neural network용 KernelSHAP는 model linearity 가정에 기대고 있으며 non-linear deep model에서는 approximation 문제가 있다고 지적한다. 이를 완화하기 위해 DASP 같은 polynomial-time approximation이나 data sparsity에 더 잘 대응하는 Baseline Shapley가 제안된다고 정리한다.

Example-based method는 feature importance heatmap 대신 “어떤 training example이 이 예측에 영향을 미쳤는가”를 설명한다. 논문은 influence function, example-level feature selection, contextual decomposition 같은 계열을 예로 든다. 이 접근은 입력별 relevance map보다 사례 기반 reasoning에 익숙한 clinician에게 더 직관적일 수 있지만, deep network의 실제 causal influence를 충실하게 반영하는지는 별개 문제다.

### 3.4 Healthcare adaptation과 evaluation 관점

논문은 단순히 general-domain explanation 방법을 나열하지 않고, 이들이 healthcare에서 어떻게 쓰이는지도 논의한다. 결론 부분 요약에 따르면, Section 3에서는 일반 도메인에서 제안된 방법들이 healthcare problem에 어떻게 적응되는지 다루고, Section 4에서는 세 가지 질문을 중심으로 적용상의 핵심 이슈를 다룬다. 그 세 질문은 다음과 같다.

1. 해당 interpretation method는 model-agnostic한가
2. 그 explanation의 credibility와 trustworthiness는 얼마나 되는가
3. 서로 다른 explanation method의 성능을 어떻게 비교할 것인가

이 구조는 의료 AI에서 explanation을 “보기 좋은 그림”이 아니라 evaluation 대상이 되는 artifact로 취급한다는 점에서 중요하다. 즉, explanation 자체도 신뢰성과 적합성을 따져야 하며, 의료 환경에서는 이것이 prediction 성능만큼 중요할 수 있다.

## 4. Experiments and Findings

이 논문은 survey이므로 하나의 실험 시스템을 구축해 benchmark를 제시하는 형태는 아니다. 대신 다양한 explanation method와 healthcare adaptation 사례를 종합하고, method 평가 논문들이 무엇을 보여주는지를 정리한다. 따라서 “experiments”는 저자 자신이 수행한 실험이라기보다, 문헌 전반에서 도출한 비교적 일관된 finding의 요약으로 이해해야 한다.

첫 번째 핵심 finding은 **설명 방법마다 장단점이 크게 다르다**는 점이다. 논문은 benchmarking evaluation을 통해 서로 다른 interpretation method가 sensitivity, faithfulness, complexity 등에서 상당한 차이를 보인다고 정리한다. 이는 “한 가지 XAI 방법만 적용하면 충분하다”는 접근이 위험할 수 있음을 뜻한다. 오히려 explanation method의 조합이나 ensemble이 필요할 수 있다는 해석으로 이어진다.

둘째, explanation method의 **신뢰성 평가는 쉽지 않다**. 논문은 일부 feature-importance annotation이 human annotation이 아니라 synthetic하게 생성된 것이라 설득력이 약하다고 지적한다. 또한 sensitivity, faithfulness, complexity를 정량화하는 evaluation criteria가 제안되어 있지만, explanation quality를 완전히 객관적으로 정의하기는 어렵다. 즉, explanation method는 본질적으로 model output 위에 올려진 2차 해석이기 때문에, 그 자체의 타당성 검증이 별도 연구 주제가 된다.

셋째, explanation aggregation이 유망한 방향으로 제시된다. 논문은 서로 다른 method가 서로의 약점을 보완할 수 있기 때문에 두 종류 이상의 interpretation method를 합치는 접근이 제안된다고 소개한다. 예를 들어 여러 explanation function을 결합해 complexity는 낮추고 sensitivity는 줄이는 방향이 논의된다. 이는 실무적으로도 의미가 큰데, 의료 현장에서는 단일 heatmap보다 여러 관점이 일치하는 설명이 더 신뢰받을 가능성이 높기 때문이다.

넷째, healthcare에서는 explanation이 robustness/security 점검에도 쓰인다. 논문 일부는 smart healthcare system에서 evasion attack, FGSM, randomized gradient-free attack, zeroth-order optimization attack 등을 사용해 모델의 취약점을 드러내는 예를 소개한다. 또한 ECG arrhythmia classification을 오도하는 adversarial perturbation 사례도 언급한다. 이는 explanation/perturbation method가 단순 해석 도구를 넘어, 의료 모델의 failure mode를 탐지하는 진단 도구로도 기능할 수 있음을 보여준다.

## 5. Strengths, Limitations, and Interpretation

이 논문의 가장 큰 강점은 **방법론 중심의 깊이**다. 많은 survey가 explainable AI의 필요성과 개념적 구분만 다루는 반면, 이 논문은 attribution method를 수학적 정의 수준에서 설명하고, method별 장단점과 적용 맥락을 함께 정리한다. 특히 저자들은 기존 survey와 달리 high-level overview가 아니라 methodological guidance를 제공하는 것이 목표라고 명시한다. 이 점에서 본 논문은 healthcare XAI 입문서이면서 동시에 method selection guide 역할을 한다.  

둘째 강점은 **taxonomy가 잘 설계되어 있다는 점**이다. back-propagation, attention, perturbation, distillation, game theory, example-based, generative-based라는 구분은 넓은 XAI literature를 구조적으로 파악하게 해 준다. 특히 attribution을 공통 프레임으로 설정해 서로 다른 방법들을 비교 가능한 대상으로 만든 점이 좋다.

셋째 강점은 **실제 적용 질문을 다룬다**는 점이다. 이 논문은 “모델이 설명 가능한가”보다 “이 explanation을 믿어도 되는가”, “서로 다른 explanation을 어떻게 비교할 것인가”, “어떤 상황에서 어떤 방법을 선택해야 하는가”를 더 중요하게 본다. 이것은 의료 AI의 현실적 요구와 잘 맞는다.

반면 한계도 있다. 첫째, survey 범위가 매우 넓어서 개별 방법의 수학적 증명이나 의료 데이터별 empirical comparison은 상대적으로 얕을 수밖에 없다. 예를 들어 SHAP, LRP, IG, CAM 계열의 faithfulness를 동일 데이터셋과 동일 task에서 정밀 비교하는 식의 일관된 benchmark는 논문 자체가 제공하지 않는다. 둘째, explanation의 정의 자체가 community 내에서 통일되지 않았다는 점을 저자들도 인정한다. 즉, 어떤 complex model에 대해 simple explanation은 본질적으로 불완전할 수밖에 없고, local explanation과 global explanation의 긴장도 남아 있다. 셋째, explanation method literature가 너무 빠르게 성장하므로, 저자들도 이 review가 completion date에 묶여 있음을 명시한다.

비판적으로 해석하면, 이 논문은 “설명 가능한 딥러닝”을 완성된 해결책으로 제시하지 않는다. 오히려 explanation은 언제나 근사이고, explanation method 선택 자체가 새로운 모델링 문제라고 보여준다. 하지만 바로 그 점 때문에 이 논문은 가치가 있다. 의료 AI에서 explanation은 accessory가 아니라 trust calibration mechanism이며, 이 논문은 그 설계 공간을 꽤 정교하게 보여준다.  

## 6. Conclusion

이 논문은 healthcare에서의 explainable deep learning을 attribution 중심으로 체계화한 survey다. 의료 영역에서 DL adoption을 가로막는 black-box 문제를 출발점으로 삼아, interpretability의 개념적 필요성, attribution의 수학적 정의, back-propagation/attention/perturbation/distillation/game theory/example-based/generative-based explanation의 taxonomy, 그리고 healthcare adaptation 및 evaluation 이슈를 폭넓게 다룬다. 특히 저자들은 explanation method를 단순 visualization 기법이 아니라, clinician의 신뢰 형성, bias 점검, 모델 디버깅, 규제 대응을 위한 핵심 도구로 본다.  

실무적으로 이 논문은 “어떤 XAI 방법이 제일 좋은가”에 대한 단일 답을 주지는 않는다. 대신 더 유용한 질문, 즉 “내 문제는 어떤 explanation을 요구하는가”, “faithfulness와 interpretability 중 무엇을 더 중시하는가”, “모델 구조나 데이터 modality가 어떤 제한을 주는가”, “설명을 어떻게 평가할 것인가”를 던지게 만든다. 그래서 이 논문은 설명 가능한 의료 AI를 설계하려는 연구자와 실무자에게 매우 좋은 출발점이 된다.  
