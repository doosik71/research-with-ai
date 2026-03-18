# Secure and Robust Machine Learning for Healthcare Applications: A Survey

이 논문은 healthcare에서 ML/DL이 빠르게 도입되고 있음에도 불구하고, 실제 임상 환경에서의 **security, privacy, robustness** 문제가 여전히 해결되지 않았다는 점을 중심에 두는 survey다. 저자들은 먼저 prognosis, diagnosis, treatment, clinical workflow 등 의료 응용 전반에서 ML/DL이 어떻게 활용되는지 정리한 뒤, 의료용 ML 파이프라인의 각 단계에서 어떤 취약점이 발생하는지 구조적으로 분석한다. 이어서 adversarial attack, poisoning, privacy leakage, data bias, distribution shift, regulation 같은 문제를 정리하고, homomorphic encryption, secure aggregation, poisoning defense 같은 대응 방향을 소개한다. 즉, 이 논문은 “의료 AI가 얼마나 잘 맞추는가”보다 “그 성능을 실제 의료 현장에서 믿고 쓸 수 있는가”를 묻는 survey라고 볼 수 있다.  

## 1. Paper Overview

이 논문의 목표는 healthcare에서 ML/DL이 쓰이는 대표 응용을 개괄하고, 그 성능을 실제 임상적 가치로 연결하는 과정에서 발생하는 **보안·프라이버시·강건성 문제**를 체계적으로 정리하는 것이다. 논문 초록과 서론은 최근 ML/DL이 심장 이상 예측, medical image 기반 CADx, radiology, pathology, ophthalmology, dermatology 등에서 뛰어난 성능을 보였지만, adversarial attack과 data/model poisoning, privacy breach 가능성 때문에 life-critical domain인 healthcare에 그대로 배치하기 어렵다고 지적한다. 즉, 정확도 향상만으로는 임상 도입이 정당화되지 않으며, integrity와 security가 함께 확보되어야 한다는 것이 논문의 출발점이다.  

논문이 제기하는 연구 문제는 다음처럼 정리할 수 있다. 의료 AI 시스템은 데이터 수집, 라벨링, 모델 학습, 배치, 추론의 긴 파이프라인 위에 구축되는데, 각 단계마다 노이즈, 라벨 오류, 데이터 편향, 분포 이동, adversarial manipulation, privacy leakage, 해석 오류가 발생할 수 있다. 따라서 “좋은 모델”이 아니라 “secure, private, robust한 모델링 및 시스템 파이프라인”이 필요하다. 이 관점에서 저자들은 predictive healthcare용 ML pipeline을 공식적인 분석 단위로 삼고, 단계별 vulnerability source를 정리한다.  

왜 이 문제가 중요한가도 분명하다. healthcare는 error cost가 매우 높다. false negative는 환자를 놓치게 만들고, false positive는 불필요한 검사·시술·불안을 초래한다. 여기에 개인정보 유출, 알고리즘 편향, cross-site generalization failure, 규제 부적합성까지 더해지면, ML 시스템의 단순 benchmark 성능은 실제 가치와 크게 어긋날 수 있다. 논문은 lab 환경의 높은 성능이 곧 safety를 의미하지 않으며, hidden strata, rare case, subtle case, outlier까지 안전하게 다룰 수 있어야 임상적 의미가 있다고 본다.  

서론에서 저자들이 명시한 기여는 다섯 가지다. 첫째, healthcare에서의 ML/DL 응용을 정리한다. 둘째, predictive healthcare를 위한 ML pipeline과 단계별 취약점을 구조화한다. 셋째, conventional security/privacy challenge와 ML adoption으로 인해 새로 생기는 challenge를 함께 논의한다. 넷째, secure and privacy-preserving ML을 위한 잠재적 해법을 소개한다. 다섯째, open research issue를 제시한다. 이 기여 구조가 논문 전체의 서술 흐름을 그대로 이끈다.

## 2. Core Idea

이 논문의 핵심 아이디어는 의료 AI의 문제를 **개별 알고리즘의 정확도 문제**가 아니라 **end-to-end pipeline robustness 문제**로 재정의하는 데 있다. 논문은 Figure 1과 Figure 4를 통해 ML 기반 healthcare system development와 predictive clinical care pipeline을 도식화하고, 이 파이프라인의 각 단계마다 고유한 vulnerability가 존재한다고 본다. 즉, 센서와 데이터 수집에서 생긴 artifact, annotation 단계의 ambiguity, training 단계의 poisoning이나 privacy leak, deployment 단계의 distribution shift, testing 단계의 false interpretation이 모두 최종 시스템 위험으로 연결된다.  

이 관점의 중요한 점은 “adversarial example만 막으면 안전하다”는 식의 협소한 보안 이해를 거부한다는 것이다. 저자들은 security threat를 influence, security violation, attack specificity라는 세 축으로 분류한다. influence 차원에서는 causative attack과 exploratory attack을 나누고, security violation 차원에서는 integrity attack, availability attack, privacy violation attack을 구분하며, attack specificity 차원에서는 targeted와 indiscriminate attack을 구별한다. 이 taxonomy는 의료 AI가 단지 inference-time adversarial perturbation뿐 아니라 training-time poisoning, privacy leakage, service degradation까지 포함한 훨씬 넓은 위협 공간을 가진다는 사실을 강조한다.

또 다른 핵심 아이디어는 healthcare 특유의 조건이 robustness 문제를 더 어렵게 만든다는 점이다. 의료 데이터는 naturally limited, imbalanced, sparse하며, ground truth조차 전문가 간 불일치가 있을 수 있다. 더구나 병원마다 장비, 프로토콜, 환자군, 인프라가 달라 distribution shift가 빈번하다. 따라서 일반 도메인에서의 ML robustness보다 healthcare robustness가 더 어렵고 더 중요하다고 본다. 저자들은 이 점에서 high benchmark performance와 safe clinical deployment를 명확히 구분한다.  

정리하면, 이 논문의 core idea는 다음과 같다.
의료 ML의 문제는 “정확한 예측 모델을 만드는 것”이 아니라, **heterogeneous data, fragile annotation, adversarial manipulation, privacy requirement, causal ambiguity, institutional heterogeneity, policy/regulation**까지 견딜 수 있는 전체 시스템을 설계하는 것이다. 이 framing 덕분에 survey는 단순 공격 사례 모음이 아니라, secure healthcare ML의 설계 지도처럼 읽힌다.  

## 3. Detailed Method Explanation

이 논문은 새로운 단일 알고리즘을 제안하는 연구가 아니라 survey이므로, 여기서의 “method”는 저자들이 제시한 **systematic framework와 taxonomy**를 뜻한다. 즉, 구체적 기술은 healthcare application taxonomy, vulnerability analysis, security threat taxonomy, solution landscape로 구성된다.

### 3.1 ML for healthcare: application structure

논문은 먼저 healthcare에서 ML이 어디에 쓰이는지 정리한다. 큰 범주는 prognosis, diagnosis, treatment, clinical workflow 네 가지다. prognosis에서는 multi-modal patient data를 활용한 disease progression, recurrence, survivability prediction이 다뤄진다. diagnosis에서는 EHR와 medical imaging이 핵심인데, imaging 쪽은 enhancement, detection, classification, segmentation, reconstruction, registration, retrieval 등 세부 task로 나뉜다. treatment에서는 image interpretation/report generation과 real-time health monitoring이 언급된다. clinical workflow에서는 disease prediction, CADx/CADe, clinical reinforcement learning, time-series modeling, clinical NLP, clinical speech/audio processing까지 포함한다. 이 구조는 논문이 healthcare ML을 단순 의료영상 분류 문제로 보지 않고, 병원 운영과 의사결정 전반으로 확장해서 본다는 점을 보여준다.  

또한 저자들은 healthcare ML의 big picture 안에서 unsupervised, supervised, semi-supervised, reinforcement learning을 모두 소개한다. 특히 semi-supervised learning은 의료에서 라벨이 부족하다는 현실 때문에 중요하며, RL은 sepsis treatment 같은 sequential clinical decision 문제에 잠재력이 있다고 설명한다. 이 부분은 뒤의 robustness 문제와도 연결된다. 학습 패러다임이 달라질수록 vulnerability surface도 달라지기 때문이다.

### 3.2 Predictive healthcare ML pipeline and vulnerabilities

논문의 방법론적 중심은 Figure 4로 상징되는 pipeline analysis다. 저자들은 predictive healthcare pipeline을 대략 데이터 수집, 데이터 annotation, 모델 학습, 배치, 테스트/해석 단계로 본다. 각 단계에서 발생하는 문제는 다음과 같이 정리된다.

데이터 수집 단계에서는 instrumental and environmental noise, 예를 들어 MRI motion artifact처럼 원천적 signal corruption이 발생할 수 있다. 또한 qualified personnel 부족도 문제다. 즉, 데이터가 처음부터 clinical-grade ML input이 아닐 수 있다.

데이터 annotation 단계에서는 ambiguous ground truth와 improper annotation이 핵심이다. 의료 데이터는 전문가 간에도 해석이 갈릴 수 있고, 희귀 질환일수록 fine-grained labeling이 어렵다. 여기에 trainee staff나 자동 라벨링이 투입되면 coarse label, class imbalance, label misspecification이 생긴다. 결국 supervised ML의 전제인 “정확한 label”이 healthcare에서는 자주 무너진다.

이와 관련된 효율 문제로 limited and imbalanced datasets, sparsity가 함께 지적된다. 많은 life-threatening condition은 자연적으로 rare하고, missing value도 흔하다. 그래서 표준 ERM 학습은 편향되기 쉽고, minority/rare class 안전성이 낮아질 수 있다. 이는 실제 환자 안전 문제로 직접 이어질 수 있다.

모델 학습 단계에서는 improper training setting 외에도 adversarial attack, model stealing, model poisoning, data poisoning, privacy breach가 등장한다. 논문은 특히 training-time poisoning과 inference-time adversarial vulnerability를 모두 중요한 위험으로 다룬다. 즉, 학습이 끝난 뒤에만 공격받는 것이 아니라, 학습 자체가 오염될 수 있다는 관점이다.

배치 단계에서는 distribution shift와 incomplete data가 큰 위험이다. 병원, 장비, 프로토콜, 시기, 환자군이 달라지면 training distribution과 deployment distribution이 달라지며, public benchmark에서 잘 되던 모델이 실제 임상에서 성능이 급격히 떨어질 수 있다. 이는 단순 정확도 하락이 아니라 safety hazard다.  

테스트 및 해석 단계에서는 false positive, false negative, misinterpretation이 문제다. 저자들이 말하는 “ML empowered healthcare의 진정한 본질은 단순히 크랭크를 돌리는 것이 아니다”라는 표현은, 결과를 임상적으로 조심스럽게 읽어야 한다는 점을 강조한다.

### 3.3 Security threat taxonomy

논문은 ML security를 세 차원의 taxonomy로 정리한다. 이는 survey의 중요한 methodological device다.

첫 번째는 **influence**다. 공격이 training data를 건드리면 causative attack이고, 학습은 건드리지 않은 채 test-time weakness를 노리면 exploratory attack이다.
두 번째는 **security violation**이다. harmful input을 benign처럼 보이게 만들어 false negative를 높이면 integrity attack, benign input을 harmful처럼 보이게 만들어 false positive를 늘리면 availability attack, training data나 model의 sensitive information을 드러내면 privacy violation attack이다.
세 번째는 **attack specificity**다. 특정 sample이나 subgroup을 노리면 targeted attack, 시스템을 전반적으로 망가뜨리면 indiscriminate attack이다.

이 taxonomy는 healthcare 문맥에서 특히 유용하다. 예를 들어 cancer detection 시스템에서 악성 종양을 놓치게 하면 integrity attack이고, 정상 환자를 계속 양성으로 보내면 availability attack에 가깝다. 또 환자 재식별이나 민감 속성 복원은 privacy violation으로 해석된다. 논문은 이 taxonomy를 통해 일반 ML security 개념을 healthcare consequence에 직접 매핑한다.

### 3.4 Adversarial machine learning in healthcare

논문은 adversarial attack을 크게 poisoning attack과 evasion attack으로 나눈다. poisoning attack은 training data를 조작해 모델이 잘못 학습하도록 만드는 방식이고, evasion attack은 inference 시점에서 입력을 조작해 잘못된 예측을 유도하는 방식이다. 의료에서는 poisoning도 매우 현실적이라고 보는데, 기존 데이터를 직접 바꾸기보다 새로운 sample을 추가하는 방식으로도 진단 체계를 어지럽힐 수 있기 때문이다. hypothyroid diagnosis용 conventional ML 모델들에 대한 systematic poisoning 사례가 그 예로 언급된다.

또 흥미로운 점은 논문이 “adversarial patients”라는 개념을 언급한다는 것이다. 이는 공격자가 의도적으로 만든 adversarial example뿐 아니라, 모델 관점에서 취약하게 작동하는 환자 subgroup이 실제로 존재할 수 있음을 뜻한다. 즉, 환자 자체가 모델의 failure case가 될 수 있으며, identical predictive feature를 가져도 treatment effect가 다를 수 있다는 점은 의료 ML의 윤리적 문제와도 연결된다.

실제 공격 예로는 fundoscopy, dermoscopy, chest X-ray에 대한 white-box/black-box adversarial attack이 소개된다. 이는 논문이 보안을 추상적으로만 논하지 않고, 이미 주요 의료영상 application에서 공격 가능성이 실증되었다고 본다는 뜻이다.

### 3.5 Secure, private, and robust ML solutions

Section IV 전체는 완전히 펼쳐 보이지는 않았지만, 논문은 secure/private/robust ML을 위한 해결책들의 taxonomy와 Table I을 제시한다. 여기서 확인 가능한 해결책은 다음과 같다.

첫째, **cryptographic privacy/security method**다. commodity-based cryptography, homomorphic encryption, Paillier homomorphic encryption, secure logistic regression, multiparty random masking, polynomial aggregation 같은 방식이 제시된다. 핵심 아이디어는 민감 의료 데이터를 평문으로 노출하지 않고도 학습이나 추론을 가능하게 하는 것이다. Table I에는 secure logistic regression, encrypted DNN, encrypted classical ML model 사례가 포함된다.  

둘째, **poisoning defense**다. Table I에서 Jagielski 등의 TRIM 알고리즘이 poisoning attack 방어 방법으로 소개된다. 즉, 이 survey는 privacy만이 아니라 robustness against malicious data도 solutions 축에 포함한다.

셋째, **secure/private/robust ML taxonomy** 자체다. Figure 6은 commonly used approaches의 taxonomy를 보여준다고 설명되며, Section IV 첫 문단은 secure, private, and robust ML methods를 개괄한다고 밝힌다. 다만 제공된 내용이 부분적이므로, 논문이 상세히 다루는 세부 하위 기법을 여기서 모두 단정하는 것은 어렵다. 안전하게 말하면, 이 논문은 cryptographic protection, privacy-preserving learning, poisoning/adversarial robustness, 그리고 secure ML deployment를 아우르는 해법 공간을 survey한다.

### 3.6 Causality, fairness, and regulation as methodological requirements

이 논문에서 중요한 점은 해결책을 순수 기술 문제로만 보지 않는다는 것이다. causality, fairness, regulation도 사실상 “robust healthcare ML”의 구성 요소로 다룬다. 예를 들어 의료에서는 “treatment $A$ 대신 $B$를 주면 어떻게 되는가?” 같은 counterfactual question이 중요하지만, correlation-driven DL은 이런 질문에 직접 답하지 못한다. 논문은 이런 문제 때문에 causal model이 필요하다고 보고, fairness 역시 causal reasoning의 관점에서 더 잘 다룰 수 있다고 본다.

또 regulatory challenge도 핵심이다. evolving ML/DL-based software as a medical device는 기존 정적 인증 체계와 잘 맞지 않는다. 따라서 objective clinical evaluation과 지속적 reassessment가 필요하며, 병원에도 data scientist와 AI engineer가 상시적으로 안전성을 점검해야 한다고 주장한다. 이는 이 논문이 “모델을 개발하는 법”보다 “모델을 의료시스템으로 운영하는 법”에 관심이 크다는 점을 보여준다.

## 4. Experiments and Findings

이 논문은 survey이므로 하나의 통일된 benchmark나 새로운 실험 시스템을 제안하지 않는다. 따라서 이 섹션에서 말하는 “findings”는 저자들의 단일 실험 결과가 아니라, 문헌 전반을 종합하며 도출한 패턴이다.

첫 번째 핵심 finding은 ML/DL이 healthcare application 전반에서 이미 강한 성능을 보이고 있다는 점이다. 저자들은 pathology, radiology, ophthalmology, dermatology 등에서 human-level 또는 그 이상의 성능 보고가 있었고, FDA 승인을 받은 지능형 진단 시스템 사례까지 언급한다. 즉, 이 논문은 healthcare AI의 유망성을 부정하지 않는다. 오히려 성능이 충분히 좋기 때문에, 이제는 robustness와 security가 더 중요한 병목이 되었다고 본다.

두 번째 finding은 **prediction performance degradation under shift**다. EHR example에서 historical data로 학습한 모델이 미래 데이터에서 mortality prediction과 length-of-stay 성능이 떨어진다고 소개된다. 이는 의료에서 temporal shift와 institutional shift가 실제로 무시할 수 없는 문제임을 보여준다. 즉, standard train/test split에서 얻은 높은 score는 deployment reliability를 보장하지 않는다.  

세 번째 finding은 의료영상과 clinical workflow가 adversarial vulnerability에서 예외가 아니라는 점이다. 저자들은 chest X-ray, dermoscopy, fundoscopy 같은 대표 medical imaging application에 대해 white-box/black-box adversarial attack이 실증되었다고 정리한다. 이는 “의료영상은 사람이 보니 괜찮을 것”이라는 순진한 기대가 틀릴 수 있음을 보여준다. 또한 poisoning attack이 hypothyroid diagnosis 같은 structured-data medical model에도 적용될 수 있다고 소개한다.

네 번째 finding은 데이터 문제의 심각성이다. 논문은 healthcare data가 limited, imbalanced, sparse하며, subjectivity, redundancy, bias를 내포한다고 반복해서 강조한다. 특히 보험 미가입자 집단처럼 기존 병원 운영 관행 자체가 데이터에 bias를 남기고, AI가 그 bias를 재학습할 수 있다는 사례는 매우 중요하다. 이는 단지 성능 문제를 넘어 fairness와 ethics 문제까지 연결된다.  

다섯 번째 finding은 secure/privacy-preserving technique의 필요성이다. Table I에 요약된 여러 연구들은 homomorphic encryption, secure aggregation/masking, secure regression, poisoning defense 등 다양한 방법을 사용해 의료 데이터를 보호하거나 학습 시스템을 더 안전하게 만들려 한다. 논문 자체가 이들의 성능을 하나의 벤치마크로 재평가하진 않지만, 적어도 “secure ML for healthcare”가 이미 독립된 연구 축으로 자라나고 있음을 보여준다.  

## 5. Strengths, Limitations, and Interpretation

이 논문의 가장 큰 강점은 **문제 설정의 넓이와 구조화 능력**이다. 많은 논문이 adversarial attack 하나, privacy leakage 하나만 다루는 반면, 이 논문은 healthcare ML pipeline 전체를 분석 단위로 놓고 데이터 수집부터 임상 배치까지 vulnerability를 연결한다. 덕분에 독자는 secure healthcare ML을 “공격 방어 알고리즘”이 아니라 “시스템 설계 문제”로 이해하게 된다.  

두 번째 강점은 **응용과 보안을 함께 다룬다**는 점이다. 저자들은 prognosis, diagnosis, treatment, clinical workflow 등 실제 응용을 먼저 정리한 뒤 보안 논의로 넘어간다. 그래서 security 논의가 추상적이지 않고, 왜 이 문제가 실제 의료에서 중요한지 자연스럽게 드러난다. 예를 들어 EHR, medical imaging, report generation, real-time monitoring, clinical RL 같은 응용은 각기 다른 데이터 타입과 threat surface를 가진다는 사실을 독자가 이해하게 된다.

세 번째 강점은 **기술적 문제를 사회기술적 문제와 연결한다**는 것이다. safety, privacy, ethics, causality, regulation, infrastructure modernization까지 포함하는 범위는 매우 인상적이다. 특히 “lab performance is not evidence of safety”, “causality is challenging”, “updating hospital infrastructure is hard” 같은 논점은 오늘날에도 여전히 중요한 메시지다.  

반면 한계도 있다. 첫째, survey 범위가 넓은 대신 각 방어 기법의 수학적 세부 비교는 깊지 않다. 예를 들어 adversarial defense나 privacy-preserving learning의 어떤 방법이 어떤 threat model에서 가장 낫다는 식의 통합 benchmark 분석은 제공하지 않는다. 둘째, 제공된 사례 중 일부는 고전적이거나 point solution 수준에 머문다. 즉, secure healthcare ML의 완성된 engineering blueprint보다는 문제 지도와 사례 모음에 가깝다. 셋째, 논문 시점이 2020년 초이기 때문에, 이후 급격히 발전한 federated learning, diffusion-based attack/defense, large multimodal model security, foundation model regulation 논의는 반영되어 있지 않다. 다만 이는 출간 시점을 고려하면 자연스러운 한계다.

비판적으로 해석하면, 이 논문은 “robust healthcare AI를 만들 수 있다”기보다 “왜 그것이 어려운지”를 설득력 있게 보여주는 문서다. 즉, 단순한 accuracy leaderboard 경쟁이 얼마나 실제 의료 현실과 거리가 먼지를 드러낸다는 점이 핵심 가치다. 실제로 이 논문을 읽고 나면, secure ML을 위해서는 더 나은 모델뿐 아니라 더 좋은 데이터 거버넌스, 더 나은 annotation protocol, causal reasoning, privacy-preserving computation, interoperability standard, regulatory design이 모두 필요하다는 결론에 이르게 된다.  

## 6. Conclusion

이 논문은 healthcare에서의 ML/DL 응용을 폭넓게 정리하면서, 그 실제 deployment를 가로막는 security, privacy, robustness 문제를 체계적으로 분석한 survey다. 저자들은 prognosis, diagnosis, treatment, clinical workflow 전반에서 ML이 이미 큰 가능성을 보여주고 있음을 인정하면서도, 의료용 ML pipeline의 각 단계가 instrumental noise, annotation ambiguity, limited and biased data, poisoning, adversarial attack, distribution shift, incomplete data, misinterpretation 같은 취약점을 가진다고 본다. 이어서 security threat taxonomy를 정리하고, privacy-preserving cryptographic method와 poisoning defense 같은 해결 방향을 제시하며, safety, fairness, causality, regulation, interoperability, infrastructure modernization까지 open issue로 끌어올린다.

실무적으로 이 논문이 주는 가장 중요한 교훈은, 의료 AI는 단지 “잘 맞추는 모델”로는 충분하지 않다는 것이다. 실제로 필요한 것은 **secure, private, robust, fair, clinically governable system**이다. 따라서 향후 연구와 개발은 adversarial robustness, privacy-preserving learning, better data standardization, distribution shift handling, causal reasoning, clinical evaluation protocol을 함께 다뤄야 하며, 병원 인프라와 규제 체계 역시 이를 수용하도록 바뀌어야 한다. 이런 의미에서 이 논문은 healthcare ML의 기술적 가능성과 사회기술적 과제를 동시에 보여주는 중요한 survey다.  
