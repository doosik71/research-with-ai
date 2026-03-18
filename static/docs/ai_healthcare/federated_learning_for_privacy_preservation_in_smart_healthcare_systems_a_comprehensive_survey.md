# Federated Learning for Privacy Preservation in Smart Healthcare Systems: A Comprehensive Survey

이 논문은 IoMT(Internet of Medical Things) 기반 스마트 헬스케어에서 privacy preservation을 위해 Federated Learning(FL)이 어떤 역할을 할 수 있는지, 기존 보안·프라이버시 위협은 무엇인지, 그리고 DRL, digital twin, GAN 등을 결합한 확장 아키텍처와 응용 사례, 향후 연구 과제를 종합적으로 정리한 survey이다.

## 1. Paper Overview

이 논문은 스마트 헬스케어 시스템에서 IoMT 디바이스가 생성하는 민감한 의료 데이터를 중앙 서버로 모아 학습하는 기존 centralized AI/ML 방식이 갖는 privacy 및 security 한계를 짚고, 이를 완화하는 대표적 distributed AI 패러다임으로서 FL을 체계적으로 정리하는 survey다. 저자들은 단순히 “FL이 데이터를 직접 모으지 않으므로 안전하다”는 수준을 넘어서, IoMT 환경의 프라이버시 위협 유형, FL 기반 보호 메커니즘, 확장형 FL 아키텍처, 실제 healthcare 응용, 그리고 남아 있는 open challenge까지 폭넓게 다룬다.

논문의 핵심 문제의식은 다음과 같다. 스마트 헬스케어는 원격 모니터링, disease prediction, medical image processing, COVID-19 detection 같은 고도화된 서비스를 필요로 하지만, 의료 데이터는 본질적으로 고도의 민감 정보를 포함한다. 중앙 집중형 학습은 scalability 문제뿐 아니라 single point of failure와 정보 유출 위험을 키운다. 반면 FL은 원시 데이터를 외부로 내보내지 않고 local update만 공유함으로써 privacy preservation의 가능성을 제공한다. 그러나 저자들은 FL 역시 gradient leakage, inference attack, poisoning attack, Byzantine attack 등 새로운 위협을 낳는다고 보고, “FL 도입 자체”가 아니라 “FL을 어떻게 설계·보강할 것인가”를 논문의 중심 축으로 둔다.

왜 이 문제가 중요한가도 분명하다. 헬스케어는 잘못된 보안 설계가 곧 환자 안전 문제로 이어질 수 있는 영역이며, 스마트 펌프·웨어러블·원격 진단 장치·병원 정보 시스템 등 다양한 엔드포인트가 연결되어 attack surface가 넓다. 따라서 privacy-preserving learning은 단지 규제 준수 문제가 아니라, 실제 임상 운영과 환자 보호를 위한 핵심 인프라 문제로 다뤄진다.

## 2. Core Idea

이 논문의 중심 아이디어는 “스마트 헬스케어 IoMT에서 FL을 privacy-preserving learning의 기본 골격으로 삼되, 실제 환경에서 발생하는 다양한 공격·자원 제약·참여 유인 부족·이질적 데이터 문제를 해결하기 위해 보강 아키텍처가 필요하다”는 것이다. 즉, FL을 만능 해법으로 제시하지 않고, FL을 출발점으로 한 종합 프레임워크를 survey 형태로 제시한다.

논문이 기존 survey와 구분된다고 주장하는 지점도 여기 있다. 저자들은 기존 문헌들이 대체로 다음 중 하나에 치우쳐 있다고 본다: IoT 전반의 보안/프라이버시, healthcare에서의 FL 일반론, digital health의 기술 요구사항, 혹은 IIoT 중심 논의. 반면 이 논문은 IoMT를 대상으로 privacy preservation 관점에서 FL을 정리하면서, DRL, digital twin, GAN과 같은 최신 AI 기법이 privacy threat detection 및 mitigation에 어떻게 결합되는지를 함께 다룬다는 점을 자기 기여로 제시한다. 특히 Table I은 기존 survey와 본 논문의 차별점을 “privacy preservation 중심의 holistic taxonomy”라는 형태로 요약한다.

요약하면, 이 논문의 core idea는 세 층으로 정리할 수 있다.

첫째, **IoMT의 privacy/security 문제 정의**: PII, 익명성, pseudonym, k-anonymity, edge processing, data minimization 같은 기본 개념을 정리한다.
둘째, **FL의 역할과 한계**: 중앙집중형 학습 대비 장점과 함께, FL 내부에서도 발생하는 leakage와 공격을 분석한다.
셋째, **확장 FL 설계 공간 제시**: differential privacy, homomorphic encryption, blockchain, DRL, GAN, digital twin, incentive mechanism 등을 결합한 설계 흐름을 survey한다.

## 3. Detailed Method Explanation

이 논문은 survey이므로 새로운 단일 알고리즘이나 하나의 loss function을 제안하지 않는다. 대신, 스마트 헬스케어 IoMT에서 FL을 이해하기 위한 구조적 설명을 제공한다. 따라서 “method”는 개별 모델의 세부 구현이라기보다, 논문이 제시하는 시스템 구조와 설계 요소의 분해로 이해하는 것이 맞다.

### 3.1 IoMT의 privacy 문제 정식화

논문은 먼저 IoMT 환경에서 privacy를 단순한 암호화 문제로 보지 않는다. PII(Personal Identifiable Information)를 기준으로 데이터를 sensitive personal data, general data, statistical data로 나누고, PII owner와 processor의 역할을 설명한다. 이때 핵심은 healthcare 데이터가 단순히 “한 번 유출되면 곤란한 정보”가 아니라, 치료 이력, 상태, 생체 정보, 위치·행동 패턴까지 포함해 개인을 재식별할 수 있는 정보라는 점이다.

또한 prevention mechanism 차원에서 데이터 최소 수집, 짧은 저장 기간, edge processing, anonymization 같은 운영 원칙을 제시한다. 이는 학습 모델 이전 단계의 privacy-by-design 원칙으로 볼 수 있다. 즉, FL이 있더라도 데이터 수집·보관·처리 단계에서 privacy exposure를 줄이는 것이 선행되어야 한다는 관점이다.

### 3.2 중앙집중형 ML 대비 FL의 기본 구조

논문이 설명하는 FL 기반 healthcare architecture는 전형적인 parameter-server 방식이다. 흐름은 다음과 같다. 먼저 중앙 서버가 task를 정의하고, prediction/classification 목적과 관련 hyperparameter를 선택하며, 참여 client를 결정한다. 그 다음 초기 global model을 각 end node에 배포한다. 각 node는 자신의 local data로 모델을 학습한 뒤 local model update를 서버에 보내고, 중앙 서버는 이를 aggregate하여 새로운 global model을 만든다. 이 과정을 원하는 정확도에 도달할 때까지 반복한다. aggregation 예시로는 data size 기반 가중치를 두는 federated averaging이 언급된다.

이를 간단한 알고리즘 흐름으로 쓰면 다음과 같다.

1. 중앙 서버가 초기 모델 $\theta^{(0)}$와 학습 설정을 결정한다.
2. 각 client $k$는 자신의 로컬 데이터 $D_k$로 $\theta^{(t)}$를 학습해 local update $\Delta_k^{(t)}$ 또는 local parameter $\theta_k^{(t+1)}$를 만든다.
3. 서버는 이를 데이터 양 혹은 지정된 weight로 집계해 새로운 global model $\theta^{(t+1)}$를 생성한다.
4. 수렴할 때까지 반복한다.

논문은 엄밀한 수식 전개를 제공하지는 않지만, 설명상 핵심 집계 아이디어는 FedAvg 계열이다. 개념적으로는

$$
\theta^{(t+1)} = \sum_{k=1}^{K} w_k , \theta_k^{(t+1)}
$$

처럼 이해할 수 있으며, 여기서 $w_k$는 각 client의 데이터 크기나 중요도를 반영하는 aggregation weight다. 이 수식은 논문의 핵심 설명을 명확히 하기 위해 정리한 표현이며, 논문 본문은 이를 “weights are assigned to local model parameters based on the data size availability” 수준으로 설명한다.

### 3.3 FL이 제공하는 장점

논문이 보는 FL의 직접적 이점은 세 가지다.

첫째, 원시 데이터를 공유하지 않으므로 privacy leakage 가능성을 줄인다.
둘째, 여러 기관·디바이스의 데이터를 활용해 generalization을 높인다.
셋째, 대규모 IoMT 네트워크에서 전체 데이터를 업로드하지 않고 gradient/model update만 보내므로 communication burden을 줄인다.

하지만 이 장점은 조건부다. 논문은 global model이나 central server가 compromise되면 전체 FL framework의 효율이 무너질 수 있다고 지적한다. 또한 IoT 디바이스의 power/resource 제약, heterogeneous data, unstable network는 FL 운영을 어렵게 만든다. 즉, privacy-preserving learning이 가능해지는 대신, system robustness와 distributed optimization 문제가 새롭게 등장한다.

### 3.4 FL 내부의 공격 표면과 보호 설계

논문의 가장 중요한 method-oriented 부분은 Section IV의 “Featured FL design for IoMT”이다. 여기서 저자들은 FL을 privacy-enabled FL, incentive-enabled FL, FL-enabled digital twin 등으로 나누어 설명한다.

#### (a) Information leakage

FL에서는 raw data를 보내지 않더라도 local update 자체가 training data에 대한 단서를 담을 수 있다. 논문은 reconstruction attack이나 leakage 문제를 지적하며, 이에 대한 보호책으로 Paillier homomorphic encryption(PHE), Shamir’s threshold secret sharing(TSS)를 사용하는 EaSTFLy 같은 접근을 소개한다. 핵심 아이디어는 aggregation에 필요한 계산은 가능하게 하되, 개별 update의 내용을 직접 해독할 수 없도록 만드는 것이다.

#### (b) Poisoning attack

논문은 poisoning을 data poisoning과 model poisoning으로 구분한다. 전자는 training sample 자체를 변조하는 경우이고, 후자는 local update를 조작하는 경우다. 이를 막기 위해 GAN 기반 auditing이나 malicious user 탐지 기법을 survey한다. 여기서 GAN은 생성 모델로서 공격 패턴을 흉내 내거나 이상한 업데이트를 구분하는 데 활용된다.

#### (c) Byzantine attack

Byzantine client는 아예 허위 model parameter를 보내 convergence와 accuracy를 망친다. 논문은 이에 대한 대응으로 blockchain-incorporated FL과 digital twin 기반 탐지 구조를 언급한다. blockchain은 update의 무결성 및 추적 가능성 측면에서, digital twin은 실제 운영 전 혹은 병렬 환경에서 anomaly를 감지하는 측면에서 기여한다.

#### (d) Privacy data leakage attack / Inference attack

논문은 differential privacy(DP)를 가장 널리 쓰이는 privacy enhancement 기법 중 하나로 다룬다. DP의 핵심은 local update나 dataset에 noise를 추가해 특정 개인 데이터의 존재 여부나 속성이 역으로 추론되지 않도록 하는 것이다. survey 대상 논문들에서는 SGD 기반 differential privacy, homomorphic encryption과 결합된 gradient boosting, cost-privacy tradeoff를 다루는 incentive mechanism 등이 소개된다. 동시에 privacy를 강화하면 accuracy가 떨어질 수 있다는 tradeoff도 분명히 지적한다.

이를 개념적으로 쓰면 DP는 어떤 query 혹은 update 결과 $M(D)$에 noise를 더해 인접 데이터셋 $D$와 $D'$에 대해 출력 분포가 크게 달라지지 않도록 만든다. 논문은 형식적 $\epsilon$-DP 정의를 쓰지는 않지만, 실질적으로는 “privacy 증가 ↔ accuracy 감소”의 tension을 survey의 주요 메시지로 제시한다.

### 3.5 Incentive-enabled FL

IoMT 디바이스는 자원 제약이 크고, privacy나 trust 문제 때문에 훈련에 적극적으로 참여하지 않을 수 있다. 그래서 논문은 incentive mechanism을 별도의 설계 축으로 본다. 데이터 가치, 디바이스 참여도, 자원 기여량을 기준으로 보상하는 방식이 소개되며, Stackelberg game, DRL, Shapley value 같은 도구가 사용된다.

이 부분의 의미는 크다. 헬스케어 FL은 병원, 웨어러블 디바이스, 환자 개인 장치, third-party platform 등 다양한 참여자가 얽혀 있으며, 실제 현장에서는 “기술적으로 가능한가?”보다 “누가 왜 참여해야 하는가?”가 더 큰 장애물이 될 수 있다. 논문은 이 문제를 자원 할당 및 사회적 복지 문제로 본다. 다만 survey 수준이라 incentive mechanism의 정교한 수학 모델을 깊게 파고들지는 않는다.

### 3.6 Digital Twin + FL

논문은 digital twin(DT)을 physical process의 디지털 복제물로 설명하고, IoMT에서는 환자 상태나 의료 운영을 가상 환경에서 먼저 시험하는 수단으로 본다. DT와 FL을 결합하면, 실제 민감 데이터를 직접 노출하지 않으면서 anomaly detection과 remote patient monitoring을 강화할 수 있다는 그림을 제시한다. 소개된 예시에서는 LSTM 기반 anomaly detection과 FL-based DT가 함께 언급된다.

다만 이 부분은 survey 전체에서 상대적으로 개념적 설명 비중이 크고, 구체적 시스템 설계나 정량 비교보다는 “유망한 방향성”으로 소개되는 성격이 강하다. 따라서 독자는 DT+FL을 완성된 표준 해법이라기보다 research frontier로 이해하는 것이 적절하다.

## 4. Experiments and Findings

이 논문 자체는 survey이기 때문에 단일 benchmark 실험을 수행하지 않는다. 대신 여러 선행 연구를 정리하면서, 어떤 task에서 FL이 어떻게 사용되었고 어떤 tradeoff가 보고되었는지를 요약한다. 즉, 실험 결과 섹션은 “저자들의 하나의 실험”이 아니라 “survey된 대표 응용 및 reported finding의 정리”로 이해해야 한다.

### 4.1 Electronic Health Record management

EHR 관리에서는 여러 기관이 협력해야 하지만 환자 정보를 직접 공유하기 어렵다. 논문은 FL 기반 EHR 분석이 privacy와 resource usage를 함께 고려할 수 있는 솔루션으로 소개되며, training data perturbation을 통해 memorization attack을 줄이는 아이디어도 언급한다. 핵심 finding은 “원본 EHR를 중앙에 모으지 않고도 다기관 협력 분석이 가능하다”는 점이다.

### 4.2 Medical image processing

의료 영상에서는 병원 간 데이터 공유가 특히 어렵다. 논문은 FL 기반 image processing과 GAN을 활용한 privacy-preserving 이미지 생성/공유 아이디어를 소개하고, 특정 cancer dataset에서 97% accuracy를 달성하며 non-FL 방식보다 우수했다고 보고된 사례를 든다. 이 결과가 시사하는 바는 두 가지다. 하나는 privacy constraint가 강한 medical imaging에서도 FL이 실용적일 수 있다는 것, 다른 하나는 synthetic/raw imprint 공유 같은 보조 전략이 성능과 privacy를 함께 잡는 방향으로 탐색되고 있다는 점이다.

### 4.3 Brain imaging

brain tumor 관련 brain imaging에서는 각 client가 DNN을 local data로 학습하고, update를 중앙에 공유한다. 이때 communication 중 model data leakage 위험을 줄이기 위해 differential privacy가 사용된다. 논문이 강조하는 실험적 메시지는, DP noise를 통한 보호가 가능하지만 그만큼 model utility와의 균형이 중요하다는 점이다. 즉, 성능 저하 없는 privacy 강화는 쉽지 않다.

### 4.4 COVID-19 detection

COVID-19 detection 응용에서는 병원별 chest X-ray 데이터를 각자 보유한 채 FL로 CNN 기반 진단 모델을 학습하는 그림이 소개된다. 또한 dynamic FL처럼 client participation과 selection을 iteration마다 조정하는 방식도 언급된다. 여기서 실험적 시사점은, 실제 의료기관 환경에서는 모든 client가 항상 같은 시점에 안정적으로 참여하지 않으므로, client selection과 timing-aware aggregation이 중요하다는 것이다. 이는 healthcare FL이 단순한 정적 분산 학습이 아니라 운영 최적화 문제이기도 하다는 점을 보여준다.

### 4.5 Table II가 보여주는 종합적 finding

논문의 Table II는 advanced FL architecture를 theme, FL type, client node, aggregator type, contribution, limitation으로 요약한다. 여기서 드러나는 반복 패턴은 다음과 같다.

* privacy 강화를 위해 DP, DNN, encryption, blockchain이 자주 결합된다.
* 대부분의 사례가 Horizontal FL(HFL) 중심이다.
* cloud/data center/centralized server 기반 aggregation이 여전히 많다.
* 거의 모든 접근이 고유한 limitation을 가진다. 예를 들어 lightweight encryption 필요성, convergence issue 미해결, IID 가정 의존, unseen threat 대응 부족 등이 반복적으로 등장한다.

즉, survey의 종합 결론은 “FL은 분명 유망하지만, 아직 production-grade healthcare privacy architecture로 완성되었다고 보기 어렵다”는 쪽에 가깝다.

## 5. Strengths, Limitations, and Interpretation

### 5.1 Strengths

이 논문의 가장 큰 강점은 범위 설정이 명확하다는 점이다. 단순히 “헬스케어에서 FL이 중요하다”는 일반론이 아니라, IoMT와 privacy preservation을 중심 축으로 삼아 관련 위협, 보호 기법, 응용, 연구 과제를 연결한다. Survey 논문으로서 독자가 분야 지형을 빠르게 파악하는 데 유용하다. 특히 Table I과 Table II가 기존 survey 대비 차별점과 advanced FL design space를 구조적으로 보여준다.

또 다른 강점은 FL의 장점뿐 아니라 한계를 함께 전면에 둔다는 점이다. 많은 입문적 글이 FL을 privacy silver bullet처럼 다루는 반면, 이 논문은 central server compromise, inference attack, poisoning, Byzantine attack, privacy-accuracy tradeoff, 참여 유인 문제를 분명히 언급한다. 이는 실무자나 연구자가 “어디서부터 추가 설계를 해야 하는가”를 파악하는 데 도움이 된다.

응용 관점에서도 EHR, medical imaging, COVID-19, remote monitoring 등 구체적인 healthcare use case를 포함해, survey가 추상적인 개념 정리에 머무르지 않게 만든다. 또한 DRL, GAN, digital twin 같은 주변 기술과 FL의 결합 가능성을 함께 다뤄 향후 연구 방향 탐색에 좋은 출발점을 제공한다.

### 5.2 Limitations

반면 한계도 분명하다. 첫째, survey 특성상 개별 방법의 수학적·실험적 비교가 깊지 않다. 예를 들어 differential privacy나 encryption 계열 기법이 어떤 threat model에서 얼마나 효과적이고 어떤 비용을 초래하는지 정교한 비교표나 통일된 벤치마크 분석은 부족하다.

둘째, 논문이 자주 언급하는 최신 결합 기법들(DRL, DT, GAN)은 흥미롭지만, 일부는 아직 개념적 소개 비중이 크고 실전 deployment 관점의 검증이 충분히 정리되지는 않는다. 따라서 독자가 “이 논문만 읽고 바로 어떤 설계를 채택할지” 결정하기는 어렵다.

셋째, healthcare FL의 중요한 현실 문제인 non-IID data, missing modality, label noise, cross-silo vs cross-device 차이, regulatory workflow, auditability 같은 주제는 부분적으로만 드러난다. 논문도 heterogeneous data와 universal architecture 부족을 open problem으로 제시하지만, survey 범위상 이를 깊게 해부하지는 않는다.

### 5.3 Critical Interpretation

비판적으로 보면, 이 논문은 “FL이 privacy를 보장한다”기보다 “FL은 중앙집중형보다 더 나은 privacy-preserving 출발점이지만, 실제 보호 수준은 추가 메커니즘 설계에 달려 있다”는 사실을 잘 드러낸다. 이 점이 오히려 이 논문의 가치다. 헬스케어에서 privacy는 data locality만으로 끝나지 않으며, update leakage, aggregation trust, incentive, network instability, heterogeneity까지 함께 봐야 한다는 통합적 관점을 제공한다.

실무적 해석으로는, 병원 간 FL 협업 시스템을 설계할 때 이 논문을 “설계 체크리스트”처럼 사용할 수 있다. 예를 들어 다음 질문이 자연스럽게 나온다: 어떤 threat model을 상정하는가? DP를 쓸 것인가, encryption을 쓸 것인가? aggregator는 신뢰 가능한가? client 참여 유인은 어떻게 설계할 것인가? non-IID data를 어떻게 다룰 것인가? 이런 질문을 구조화해 준다는 점에서 survey로서 효용이 높다.

## 6. Conclusion

이 논문은 스마트 헬스케어 IoMT에서 privacy preservation을 위한 FL의 역할을 종합적으로 정리한 survey다. 기존 centralized ML의 privacy/security/scalability 한계를 출발점으로, FL의 기본 구조와 장점, FL 내부에서 발생하는 다양한 공격, 그리고 differential privacy, homomorphic encryption, blockchain, DRL, GAN, digital twin, incentive mechanism 등을 결합한 확장 설계들을 폭넓게 다룬다. 또한 EHR, medical imaging, COVID-19 detection 등 실제 응용과 communication/network, universal architecture, heterogeneous dataset, next-generation network 같은 open research direction도 함께 제시한다.

이 작업이 실무와 후속 연구에 중요한 이유는, 헬스케어 FL을 단순한 분산 학습 기술이 아니라 privacy engineering, system design, incentive design, secure networking이 만나는 교차 영역으로 보여주기 때문이다. 따라서 이 논문은 새로운 알고리즘을 배우기 위한 문서라기보다, IoMT privacy-preserving FL 분야의 지도(map)를 얻기 위한 문서에 가깝다. 특히 헬스케어 데이터 협업 시스템, 의료 영상 연합학습, 병원 간 privacy-preserving AI 플랫폼을 설계하거나 연구하려는 사람에게 좋은 출발점이 된다.
