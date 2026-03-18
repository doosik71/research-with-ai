# Reliable and Resilient AI and IoT-based Personalised Healthcare Services: A Survey

## 1. Paper Overview

이 논문은 **Healthcare 5.0 환경에서 AI/ML과 IoT 기반 personalized healthcare service를 어떻게 더 신뢰성 있게(reliable), 복원력 있게(resilient), 그리고 진정한 의미에서 개인화된(personalized) 형태로 설계할 것인가**를 다루는 서베이 논문이다. 저자들은 기존 personalized healthcare가 대체로 특정 질환, 특정 장치, 특정 환경에만 초점을 맞추는 경향이 있어, 실제 환자가 동시에 지니는 여러 상호연관된 건강 상태를 충분히 반영하지 못한다고 본다. 이 때문에 잘못된 진단이나 부적절한 제어가 발생할 수 있으며, 장기적 건강성과 지속가능성에도 문제가 생긴다는 것이 논문의 문제의식이다.  

이를 해결하기 위해 논문은 **CPHS(Comprehensive Personalised Healthcare Services)** 라는 개념을 중심에 둔다. CPHS는 단일 건강 조건만 따로 맞춤화하는 것이 아니라, 서로 영향을 주고받는 여러 건강 조건을 함께 고려해 최적화하는 서비스다. 논문은 이를 위해 (1) Healthcare 5.0의 배경과 personalized healthcare의 재정의, (2) Things–Communication–Application의 3계층 HIoT 참조 아키텍처, (3) 계층별 reliability·resilience·personalization 요구사항과 기존 AI/비AI 접근의 장단점, (4) IoT 계층별 보안 위협, (5) 이를 종합한 제안 방법론을 순서대로 정리한다.  

## 2. Core Idea

이 논문의 핵심 아이디어는 **“개인화”를 단일 질환 중심 customization이 아니라, 상호의존적인 다중 건강 조건의 determinant-based optimization으로 재정의하는 것**이다. 기존 personalized healthcare는 유전, 생활습관, 특정 생체신호 등 일부 공통 특징을 기반으로 한 개별 맞춤치료를 의미하는 경우가 많지만, 저자들은 이것만으로는 실제 환자의 전체 건강을 설명하기 어렵다고 지적한다. 대신 유전, 행동, 환경, 물리적 영향, 의료적 개입, 사회적 요인 등 여러 determinant와 건강 조건 간 관계를 함께 고려해야 진정한 personalization이 가능하다고 주장한다.

이 관점을 실현하기 위해 논문은 Healthcare 5.0를 **connectedness + integrity + personalization**의 시대로 해석한다. 4.0이 디지털화와 smart health 비즈니스 모델에 가까웠다면, 5.0은 5G 등 진보한 통신 환경 위에서 더 긴밀한 연결성과 통합성을 바탕으로 고객 모델, lifelong partnership, digital wellness를 추구한다. 즉, 본 논문의 novelty는 새로운 딥러닝 모델 하나를 제안하는 데 있지 않고, **신뢰성·복원력·개인화를 함께 만족하는 CPHS를 위한 시스템 수준의 프레임**을 제시하는 데 있다.

## 3. Detailed Method Explanation

이 논문은 실험 중심 모델 논문이 아니라 **survey + architecture/methodology proposal** 성격의 논문이므로, “방법”은 특정 loss function이나 training recipe보다 **개념적 아키텍처와 설계 절차**로 이해하는 것이 맞다.

### 3.1 Personalized healthcare의 재정의: CPHS

저자들은 personalized healthcare를 단순히 한 건강 상태에 대한 맞춤 설정으로 보지 않는다. 대신 **환자의 여러 건강 조건을 장기적으로 함께 최적화하는 comprehensive service**로 재정의한다. 이 최적화는 각 건강 조건의 개별 상태뿐 아니라 건강 조건들 사이의 biological relationship과 side effect를 함께 고려한다. 예를 들어 고혈압이 고콜레스테롤의 부작용일 수 있고, 당뇨가 심혈관질환을 유발할 수 있으며, 한 질환의 치료가 다른 질환을 악화시킬 수도 있다는 식이다. 따라서 personalization은 “한 장치의 파라미터 조정”이 아니라, **복수의 건강 조건 간 dependency를 분석해 전체 건강을 더 나은 방향으로 제어하는 문제**가 된다.

### 3.2 Example Use Case: 건강 조건 간 의존성 분석

논문은 자동 혈압계, 혈당 모니터, 심박수 모니터, 자동 insulin pump 같은 여러 모니터링 장치를 예시로 든다. 현재 시스템은 이런 장치들을 대체로 독립적으로 모니터링하고 제어하지만, 저자들은 이것이 현실을 충분히 반영하지 못한다고 본다. 실제 환자에게는 질환 간 상호작용이 존재하기 때문이다. 논문은 이를 **one-to-many**와 **many-to-many** dependency로 설명한다. 예를 들어 어떤 질환이 여러 다른 질환에 영향을 줄 수 있고, 당뇨·관절염·간질처럼 여러 질환이 서로 복합적으로 영향을 줄 수도 있다. 이 use case는 CPHS가 왜 필요한지를 보여주는 핵심 예시다.

### 3.3 3계층 참조 아키텍처

논문은 HIoT 기반 healthcare system을 **Things layer → Communication layer → Application layer**의 3계층 구조로 본다. 또한 IoT device는 기본적으로 세 가지 성질을 가져야 한다고 말한다:
첫째, **Ability to Sense** — 혈당, 심박, 체온, 콜레스테롤 같은 신호를 감지하고 결합할 수 있어야 한다.
둘째, **Communicable** — sensing 이후 데이터를 3G/4G/5G, WLAN, WSN, MANET 등 다양한 매체로 전송할 수 있어야 한다.
셋째, **Actionable** — 수집된 데이터가 단순 표시로 끝나지 않고 처리되어, 이상 상태를 탐지하고 자동 alert나 제어로 이어져야 한다.

#### (a) Things Layer

Things layer는 biosensor, pressure sensor, wearable, medical devices 등 실제 환자 상태를 수집하는 물리적 계층이다. 논문은 biosensor의 중요성을 강조하며, 작은 크기, 저비용, 빠른 결과, 재사용성, 오염 회피 등의 장점을 설명한다. 이 계층의 personalization은 더 정밀한 sensing과 condition-specific device customization에 가깝지만, 저자들의 시각에서는 여기서 끝나면 진정한 CPHS가 아니다.

#### (b) Communication Layer

Communication layer는 수집된 데이터를 다른 계층으로 전달하는 backbone이다. WiFi, Bluetooth, RFID, 5G, Ethernet, USB 등 다양한 매체가 사용될 수 있으며, 이 계층의 핵심 요구는 energy efficiency, range, cost, reliability, security, scalability다. 논문은 Healthcare 4.0의 한계가 주로 communication layer에서 드러난다고 보고, data loss 최소화, 혼잡 없는 채널, 비용 효율성, 실시간 데이터 회수, M2M/D2D 통신 같은 문제를 강조한다. 5G의 등장은 이런 한계를 상당 부분 완화시키며 Healthcare 5.0의 기반이 된다고 해석한다.  

#### (c) Application Layer

Application layer는 monitoring, diagnosis, control, decision-making이 일어나는 계층이다. 저자들은 기존 smart healthcare application들이 종종 서로 협력하지 못하고, 동일 환자의 서로 다른 건강 조건을 맥락적으로 이해하지 못하며, 이질적인 소프트웨어/하드웨어 위에서 안전하게 정보를 교환하지 못한다고 비판한다. 또한 건강 조건 간 효과와 관계를 제대로 모델링하지 못한다는 점을 문제로 지적한다. 이것이 바로 CPHS 제안의 직접적 동기다.

### 3.4 세 가지 핵심 요구사항: Reliability, Resilience, Personalization

논문은 CPHS의 핵심 요구를 세 가지로 정리한다.

**Reliability**는 stated conditions에서 일정 시간 동안 요구 기능을 일관되게 수행하는 능력이다. Healthcare 5.0은 자동 모니터링과 자동 제어를 수행하므로, 정상 상황에서 예상대로 동작하지 않으면 환자 생명에 직접적 위험을 줄 수 있다. 저자들은 battery, memory, computational power, QoS 개선, blockchain 기반 기술 등이 reliability 향상과 관련된 요소라고 설명한다.

**Resilience**는 fault, error, bug, cyber threat 같은 adverse condition이 생겨도 서비스를 계속 유지하고, 문제를 탐지하고, 영향을 완화하며, 기능을 복구하는 능력이다. 여기에는 하드웨어·소프트웨어 결함, 클라우드 서비스 문제, 통신 저하, 과부하, 노후화, 보안 공격 등이 포함된다. 논문은 MTD(Moving Target Defense)나 blockchain 기반 복구 전략도 예시로 든다.

**Personalization**은 저자들이 가장 강하게 재정의한 개념이다. 단일 건강 조건에 대한 strict customization이 아니라, genetics, behavior, environment, physical influences, medical care, social factors를 고려하여 **multiple health conditions를 함께 최적화하는 능력**이다. 여기서 중요한 것은 단순 예측이 아니라, 부작용을 줄이고 장기적 건강을 개선하는 방향의 optimization이라는 점이다.

### 3.5 Proposed Methodology의 의미

논문은 구체적인 수학식 형태로 알고리즘을 쓰지는 않지만, 구조적으로 보면 제안 방법론은 다음 흐름을 가진다.

1. 다양한 device와 sensor에서 건강 데이터를 수집한다.
2. communication layer를 통해 데이터를 안전하고 신뢰성 있게 전달한다.
3. application layer에서 health condition 간 dependency를 분석한다.
4. reliability, resilience, personalization 요구를 동시에 고려한 제어 및 monitoring logic을 구성한다.
5. security threats까지 포함해 계층별 weaknesses를 보완하는 CPHS를 설계한다.  

즉, 저자들의 방법론은 “좋은 모델 하나”보다 **계층 전체를 종단 간으로 설계하는 healthcare system engineering**에 가깝다.

## 4. Experiments and Findings

이 논문은 전형적인 benchmark 실험 논문이 아니다. 따라서 datasets, baselines, metrics를 통일된 조건에서 비교한 정량 실험은 없다. 대신 **systematic literature review 방식의 survey methodology**를 취한다. 저자들은 Kitchenham 가이드라인을 따라 관련 문헌을 체계적으로 수집했으며, 초기 웹 검색에서 3만 편 이상을 찾고, 관련 키워드 기반 정제 후 약 200편 수준으로 압축했으며, 비과학적이거나 지나치게 이론적인 자료를 제거해 최종 분석 대상으로 삼았다고 설명한다.

이 survey가 실질적으로 보여주는 발견은 다음과 같다.

첫째, 기존 healthcare survey 상당수는 **특정 질환, 특정 환경, 특정 서비스, 특정 agent**에 집중한다. 예를 들어 심혈관, 호흡기, 당뇨 등 특정 health condition이나 smart home, assisted living, elderly home 같은 특정 환경 중심 연구가 많다. 저자들은 이것이 comprehensive personalization을 방해한다고 본다.

둘째, 현재 접근들은 reliability·resilience·personalization 중 일부만 부분적으로 지원하며, **multiple health conditions를 함께 고려하는 personalization은 매우 부족**하다고 분석한다. 논문 중간 비교표 설명에서도 기존 AI/ML/DL 기반 접근 대부분이 환자의 복수 건강 조건을 함께 다루는 personalization을 충분히 지원하지 못한다고 정리한다.

셋째, 보안 위협을 무시한 상태에서의 계층 분석은 현실성이 부족하므로, **security를 CPHS의 integral requirement**로 포함해야 한다고 본다. 이는 health IoT를 단순 기능 시스템이 아니라 공격과 장애를 전제로 한 resilient system으로 봐야 한다는 뜻이다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 **personalized healthcare의 범위를 넓혀 재정의했다는 점**이다. 일반적인 personalization이 “개인별 단일 조건 맞춤화”였다면, 이 논문은 “상호연관된 다중 건강 조건의 최적화”로 문제를 다시 세운다. 이 덕분에 의료 IoT 시스템을 더 현실적인 임상 시나리오에서 생각할 수 있게 된다.

또 다른 강점은 **system-level thinking**이다. 저자들은 장치, 네트워크, 애플리케이션을 따로 보지 않고, Things–Communication–Application 전 계층에서 요구사항과 약점을 분석한다. 여기에 security threats까지 결합해 reliability와 resilience를 함께 논의한다는 점도 강하다. 의료 AI를 모델 단위가 아니라 인프라와 운영까지 포함한 시스템 문제로 본다는 점에서 유용하다.  

셋째, use case를 통해 질환 간 dependency 문제를 구체적으로 설명한 점이 좋다. 이는 추상적 주장에 머무르지 않고, 실제로 왜 독립적인 device monitoring만으로는 부족한지 설득력을 높여 준다.

### Limitations

한편 한계도 분명하다.

첫째, 이 논문은 **survey 및 conceptual framework 중심**이어서, 구체적인 알고리즘 성능 비교나 실험적 검증은 제한적이다. 따라서 “어떤 모델이 어떤 조건에서 최고 성능을 내는가”를 알고 싶은 독자에게는 다소 추상적일 수 있다.

둘째, CPHS 방법론은 설계 철학으로는 설득력이 있지만, 실제 구현에서 필요한 세부 사항—예를 들어 데이터 표준화 방식, cross-condition inference engine, optimization objective, real-time control loop, validation protocol—은 상대적으로 덜 구체적이다. 다시 말해 시스템 blueprint는 제시하지만, 구현 recipe는 상세하지 않다. 이 평가는 논문 내용에 근거한 해석이다.

셋째, Healthcare 5.0와 personalization에 대한 서술은 다소 비전 지향적이어서, 실제 임상 도입 과정의 규제·책임·workflow integration 문제는 제한적으로 다뤄진다. 다만 이는 서베이 논문의 범위를 고려하면 어느 정도 자연스러운 한계다.

### Critical Interpretation

비판적으로 보면, 이 논문은 AI 자체의 정교한 방법론보다 **healthcare IoT를 위한 requirements engineering 논문**에 더 가깝다. 즉, “어떻게 더 정확히 예측할 것인가”보다 “어떤 요구를 동시에 만족하는 시스템이어야 하는가”를 묻는다. 그만큼 실험적 ML 논문보다는 architecture, CPS, dependable systems, digital health platform을 설계하는 연구자나 실무자에게 더 가치가 크다.

## 6. Conclusion

이 논문은 Healthcare 5.0 시대의 AI/IoT 기반 personalized healthcare를 단일 질환 맞춤화 수준에서 넘어, **reliable, resilient, and comprehensive personalized healthcare service(CPHS)**라는 관점으로 재구성한 포괄적 서베이다. 핵심 기여는 personalized healthcare를 다중 건강 조건의 determinant-based optimization으로 재정의하고, 이를 Things–Communication–Application의 3계층 HIoT 아키텍처 위에서 reliability·resilience·personalization 요구와 security threats까지 함께 분석했다는 점이다.

실제로 이 논문은 새로운 SOTA 모델을 제안하는 논문은 아니지만, **의료 IoT 시스템을 더 현실적이고 안전하며 통합적인 방향으로 설계하려는 사람에게 매우 유용한 프레임**을 제공한다. 특히 여러 만성질환이 공존하는 환자, 다양한 벤더 장치가 혼재하는 환경, 연속 모니터링과 자동 제어가 필요한 상황에서는 이 논문이 제시한 CPHS 관점이 중요한 설계 기준이 될 수 있다.  
