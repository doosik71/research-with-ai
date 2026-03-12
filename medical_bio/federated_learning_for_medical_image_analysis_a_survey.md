# Federated Learning for Medical Image Analysis: A Survey

## 논문 메타데이터

- 제목: Federated Learning for Medical Image Analysis: A Survey
- 저자: Hao Guan, Pew-Thian Yap, Andrea Bozoki, Mingxia Liu
- 발표 형태: arXiv preprint
- arXiv: [2306.05980](https://arxiv.org/abs/2306.05980)
- 버전 기준: v4
- 날짜: 2024년 7월 7일
- 키워드: federated learning, machine learning, medical image analysis, data privacy

## 연구 배경 및 문제 정의

의료 영상 인공지능은 여전히 근본적인 `small-sample-size` 문제를 안고 있다. 한 기관 내부 데이터만으로는 환자군의 다양성과 장비 차이를 충분히 반영하기 어렵고, 그 결과 학습된 모델의 일반화가 제한된다. 가장 직관적인 해결책은 여러 병원의 데이터를 합쳐 대규모 학습셋을 만드는 것이지만, 실제로는 HIPAA, GDPR 같은 규제와 기관별 정책 때문에 의료 영상 원본을 자유롭게 공유할 수 없다.

이 논문은 바로 이 딜레마를 해결하는 대표 패러다임으로 federated learning(FL)을 다룬다. FL은 각 기관이 데이터를 로컬에 유지한 채 모델 업데이트만 교환함으로써 협업 학습을 수행한다. 저자들은 이를 의료 영상 분석에 적용한 연구들을 체계적으로 정리하면서, 단순 알고리즘 나열이 아니라 FL 시스템을 `client end`, `server end`, `client-server communication` 세 부분으로 나누어 해석한다.

## 논문의 핵심 기여

- 2017년부터 2023년 10월까지의 의료영상 FL 연구를 폭넓게 정리한다.
- FL 방법을 시스템 관점에서 `client-end`, `server-end`, `communication`으로 분류하는 독특한 taxonomy를 제시한다.
- 각 범주에서 의료영상 특유의 문제, 예를 들어 client shift, limited labels, corrupted clients, data leakage를 중심으로 방법을 정리한다.
- 의료영상 FL용 benchmark dataset과 software platform을 함께 요약한다.
- 대표 FL 방법(FedAvg, FedSGD, FedProx 등)에 대한 실험 비교를 추가해 survey 이상의 실증 정보를 제공한다.
- 향후 과제로 federated domain adaptation, unseen client generalization, multi-modal FL, blockchain, medical video FL 등을 제안한다.

## FL의 기본 개념과 의료영상에서의 의미

논문은 전형적인 client-server 기반 FL 프로세스를 먼저 정리한다.

1. 서버가 client를 선택하고 global model을 초기화한다.
2. 각 client는 자신의 로컬 데이터로 모델을 학습한다.
3. client는 raw image 대신 model update만 서버로 보낸다.
4. 서버는 업데이트를 aggregation해 새 global model을 만든다.
5. global model을 다시 각 client에 broadcast한다.

이 구조는 단순해 보이지만, 의료영상 환경에서는 몇 가지 중요한 의미를 가진다.

- 데이터 원본을 중앙에 모으지 않아도 다기관 협업이 가능하다.
- 기관별 데이터 다양성을 간접적으로 반영할 수 있다.
- 다만 데이터가 안 보인다고 해서 privacy가 자동으로 보장되는 것은 아니며, gradient leakage 같은 새로운 공격면이 생긴다.

즉, FL은 privacy와 collaborative learning 사이의 절충안이지, 문제를 완전히 제거하는 만능 해법은 아니라는 점이 이 논문의 기본 입장이다.

## FL 유형 정리

논문은 FL의 기본 유형도 소개한다.

### 1. Horizontal FL

여러 기관이 유사한 feature space를 가진 데이터를 보유하고 있을 때의 FL이다. 예를 들어 여러 병원의 chest X-ray처럼 modality와 feature 정의가 비슷한 경우가 여기에 속한다. 의료영상 FL에서 가장 일반적인 형태다.

### 2. Vertical FL

같은 환자 집합에 대해 서로 다른 종류의 feature를 기관별로 나눠 가진 경우다. 예를 들어 한 기관은 영상 데이터를, 다른 기관은 유전체 또는 EHR 데이터를 보유한 경우다. 의료영상에서는 아직 덜 일반적이지만 multimodal precision medicine으로 갈수록 중요성이 커질 수 있다.

## 이 논문의 taxonomy: Client-End, Server-End, Communication

이 survey의 가장 큰 특징은 FL을 단순 알고리즘 목록이 아니라 시스템 구조 관점에서 분해한다는 점이다.

### 1. Client-End Learning

client 내부 데이터와 학습 과정에서 발생하는 문제를 다룬다.

- client shift / domain shift
- limited data and labels
- heterogeneous computation resources and data scales

### 2. Server-End Learning

서버가 여러 client 업데이트를 어떻게 모으고 제어하는지를 다룬다.

- weight aggregation
- server-side handling of client shift
- client corruption / anomaly detection

### 3. Client-Server Communication

통신 과정에서 발생하는 보안과 효율성 문제를 다룬다.

- data leakage and attack
- differential privacy
- communication efficiency

이 구조는 실제 FL 시스템 설계에 매우 유용하다. 문제를 "어느 위치에서 생기는가"에 따라 구분하기 때문에, 방법의 목적과 trade-off를 더 명확히 볼 수 있다.

## Client-End Learning 정리

### 1. Client Shift: 기관 간 분포 차이

의료영상 FL에서 가장 중요한 문제는 client shift다. 병원마다 스캐너, 프로토콜, 환자군, 이미지 품질, 병변 분포가 달라 같은 global model이 모든 client에 잘 맞지 않을 수 있다.

논문은 이를 해결하는 대표 방향으로 다음을 소개한다.

- personalized FL: shared encoder + client-specific decoder 또는 local adaptation
- local GNN, local discriminator 같은 client-specific module
- federated domain adaptation
- batch normalization 기반 적응
- image harmonization, generative replay, cycleGAN 기반 스타일 정렬
- frequency-domain harmonization

핵심 아이디어는 "모든 client에 완전히 같은 모델을 강제하지 말고, 공유할 부분과 로컬에 남길 부분을 분리하자"는 것이다. 이는 의료영상처럼 기관별 style 차이가 큰 분야에서 특히 중요하다.

### 2. Limited Data and Labels

각 client 내부 데이터가 적고 라벨은 더 적기 때문에, local model 자체가 불안정하게 학습될 수 있다. 이를 보완하기 위해 논문은 다음 흐름을 소개한다.

- self-supervised 또는 representation learning 기반 pretraining
- federated multi-task learning
- teacher-student distillation
- unlabeled data 활용
- weakly supervised / semi-supervised learning
- virtual adversarial training 및 synthetic sample generation

이 부분은 의료영상 FL이 단순 분산 학습이 아니라, 소량 라벨 환경에서의 표현학습 문제와 강하게 결합되어 있음을 보여 준다.

### 3. Heterogeneous Environments

병원마다 GPU 성능, 네트워크 대역폭, 데이터 양이 다르기 때문에, 같은 local epoch 수를 강제하면 전체 학습 속도가 비효율적이 된다. 논문은 semi-synchronous training 같은 접근을 예시로 들며, 계산 자원이 큰 client는 더 많은 update를 수행하고 느린 client는 기다림을 줄이는 방식이 중요하다고 설명한다.

이는 의료기관 현실과 잘 맞는 문제 설정이다. 실제 FL은 수학적 aggregation보다 운영 레벨의 synchronization 병목에서 먼저 무너질 수 있기 때문이다.

## Server-End Learning 정리

### 1. Weight Aggregation

서버는 각 client 업데이트를 조합해 global model을 만든다. 논문은 이 과정을 단순 평균으로만 보지 않고, aggregation 자체가 성능과 공정성의 핵심이라고 본다.

대표 예시는 다음과 같다.

- Progressive Fourier Aggregation: 저주파 파라미터 성분만 집계
- client loss 또는 성능 기반 가중 aggregation
- 불균형 데이터 상황에서 성능이 낮은 client 가중치를 조절하는 방식

이 흐름은 의료영상 FL에서 server가 단순 합산기가 아니라, noisy client를 걸러내고 중요한 client를 더 반영하는 조정자 역할을 해야 함을 보여 준다.

### 2. Server-Side Handling of Domain Shift

client 간 분포 차이로 global model이 특정 기관에 치우치게 되는 문제를 server objective 설계로 해결하려는 연구도 소개된다.

- 성능이 낮은 client에 더 큰 비중을 주는 fairness-oriented optimization
- label distribution 정보를 활용하는 aggregation 또는 weighted loss
- global gradient를 안정화하는 guided-gradient 방식

이 계열은 personalization을 client 쪽에서 하는 대신, 서버가 보다 공정한 global objective를 설계해 편향을 줄이려는 시도라고 볼 수 있다.

### 3. Client Corruption / Anomaly Detection

FL는 정상적인 client만 참여한다고 가정하지만, 실제로는 noisy label, poor quality scan, malicious attack이 섞일 수 있다. 논문은 outlier score 기반 client suppression 같은 기법을 예로 들며, 이상 client의 aggregation weight를 낮추는 전략을 설명한다.

의료영상에서는 잘못 라벨된 데이터나 품질이 낮은 영상이 실제로 잦기 때문에, 이 문제는 단순한 security 문제가 아니라 data quality management 문제이기도 하다.

## Client-Server Communication 정리

### 1. Data Leakage and Attack

논문은 "데이터를 공유하지 않으니 안전하다"는 순진한 가정을 경계한다. gradient나 intermediate representation만으로도 원본 영상이 복원될 수 있다는 연구들을 언급하며, 실제 의료영상 FL에서도 privacy leakage가 심각한 문제라고 지적한다.

대응 방법으로는 다음이 소개된다.

- partial weight sharing: 모델 일부만 공유
- differential privacy: gradient에 Gaussian noise 추가
- attack/defense 실험: gradient inversion, leakage visualization

특히 batch normalization 통계나 gradient 정보로부터 이미지를 재구성할 수 있다는 지적은 이 survey의 중요한 경고다.

### 2. Communication Efficiency

FL는 반복적으로 모델을 업로드하고 브로드캐스트해야 하므로 통신 비용이 크다. 특히 병원 네트워크 환경이 균일하지 않으면 communication bottleneck이 심해진다. 논문은 성능과 training time을 고려한 dynamic client selection 같은 방법을 소개하며, 일정 시간 안에 응답하지 못한 client를 aggregation에서 제외하는 전략도 설명한다.

이 문제는 의료기관 인프라 수준과 직접 맞닿아 있기 때문에, 실제 배포 관점에서 매우 중요하다.

## Software Platform과 Dataset 정리

이 논문은 survey로서 드물게 software platform을 꽤 비중 있게 다룬다. 대표적으로 다음이 소개된다.

- PySyft
- OpenFL
- PriMIA
- Fed-BioMed

또한 benchmark dataset도 기관/장기별로 정리한다.

- Brain: ADNI, ABIDE, BraTS, RSNA Brain CT, UK Biobank, IXI
- Chest/Lung/Heart: CheXpert, ChestX-ray14, COVID-19 CXR, COVIDx, ACDC, M&M
- Skin: HAM10000, ISIC
- Eye: Kaggle Diabetic Retinopathy
- Abdomen: PROMISE12
- Histology: TCGA
- Others: fastMRI, MedMNIST

이 정리는 의료영상 FL 연구자가 어떤 데이터로 어떤 task를 재현할 수 있는지 빠르게 파악하게 해 준다는 점에서 실용적 가치가 크다.

## 실험 연구의 의미

논문은 ADNI를 사용해 `Cross`, `Single`, `Mix`, `FedAvg`, `FedSGD`, `FedProx`를 비교한다. 핵심 결과는 다음과 같다.

- `Mix`가 가장 좋다. 모든 데이터를 직접 모았기 때문이다.
- `Cross`가 가장 나쁘다. domain shift 때문이다.
- FL 방법은 raw data sharing 없이도 `Cross`와 `Single`보다 좋은 성능을 낸다.
- weight aggregation 계열(FedAvg, FedProx)이 gradient aggregation(FedSGD)보다 유리하게 나타난다.

이 실험은 federated learning이 centralized pooling을 완전히 대체하지는 못하지만, privacy 제약하에서 실용적인 타협점이 될 수 있다는 점을 보여 준다.

## 핵심 도전 과제

논문은 FL for medical imaging의 도전 과제를 네 가지로 요약한다.

### 1. Data Heterogeneity Among Clients

스캐너, 프로토콜, 환자군, 품질 차이로 인한 client shift는 FL의 핵심 난제다. 의료영상 FL의 상당수 방법이 사실상 이 문제를 해결하려고 설계되어 있다.

### 2. Privacy Leakage / Poisoning Attacks

gradient reconstruction, malicious update, poisoned local training은 FL의 핵심 리스크다. 의료 데이터에서는 이 리스크가 기술적 문제를 넘어 법적 문제로 직결된다.

### 3. Technological Limitations

고성능 연산 자원, 네트워크 안정성, 동기화, 병원 내 인프라와의 통합이 실제 배포의 큰 걸림돌이다.

### 4. Long-Term Viability

client가 중간에 빠지거나 새로 들어오는 문제, 운영 지속 가능성, 제도 변화, 사용자 채택 문제까지 고려해야 한다. 저자들은 FL을 단순 알고리즘이 아니라 장기적 시스템 엔지니어링 문제로 본다.

## 미래 연구 방향

논문이 제시하는 방향 중 중요한 것은 다음과 같다.

### 1. Federated Domain Adaptation / Personalization

client shift를 더 잘 다루기 위해 personalized FL과 federated domain adaptation이 핵심이 될 것이라고 본다.

### 2. Multi-Modality FL

MRI, fMRI, CT, PET 등 여러 modality를 함께 쓰는 FL은 아직 초기 단계이며, 실제 임상 가치가 크기 때문에 더 연구가 필요하다고 본다.

### 3. Generalization to Unseen Clients

기존 federation 내부 client뿐 아니라, 학습 시 보지 못한 새로운 병원 데이터에도 잘 작동하는지 평가하고 향상시키는 문제를 중요한 open problem으로 제시한다. 이는 domain generalization과 test-time adaptation과 연결된다.

### 4. Weakly-Supervised FL

의료 데이터의 불완전 라벨을 고려하면 weakly supervised, noisy label learning과 FL의 결합이 중요하다고 본다.

### 5. Security, Blockchain, Decentralization

중앙 서버 의존성을 줄이기 위해 blockchain이나 decentralized FL을 탐색하는 방향도 제안한다.

### 6. FL for Medical Video

지금까지는 2D/3D 이미지 중심이었지만, 수술 비디오나 동영상 기반 의료 데이터로 FL이 확장될 가능성이 크다고 전망한다.

## 비판적 평가

이 논문은 의료영상 FL 연구를 비교적 성숙한 관점에서 정리한 survey다. 기존 survey보다 시스템 관점 분류가 분명하고, dataset 및 software platform, 실험 비교까지 넣어 실용성이 높다.

강점은 다음과 같다.

- client/server/communication 관점의 taxonomy가 명확하다.
- privacy, heterogeneity, efficiency를 한 프레임에서 함께 본다.
- platform과 benchmark dataset 정보를 정리해 재현성과 실용성을 높인다.
- 실험 섹션이 있어 survey 이상의 감각을 제공한다.

한계도 있다.

- 2024년 이전 연구 중심이므로 이후 빠르게 발전하는 federated foundation model, parameter-efficient FL, medical multimodal LLM 연계는 충분히 반영되지 않는다.
- personalization 방법들 사이의 정량적 공정 비교는 제한적이다.
- blockchain, unseen client generalization 등은 방향 제시 수준이고 아직 정교한 통합 틀까지는 아니다.

## 연구적 시사점

이 논문이 주는 핵심 메시지는 명확하다.

- 의료영상 FL의 본질은 privacy-preserving distributed optimization이 아니라, `heterogeneous, privacy-sensitive, low-label` 환경에서의 협업 학습이다.
- client shift 문제를 해결하지 못하면 FL는 단순한 분산 평균에 머물 수 있다.
- aggregation 설계, personalization, privacy defense, communication efficiency가 동등하게 중요하다.
- 앞으로는 federation 내부 성능보다, unseen client generalization과 long-term operability가 더 중요해질 가능성이 크다.

## 종합 평가

`Federated Learning for Medical Image Analysis: A Survey`는 의료영상 FL 분야를 가장 체계적으로 조망하는 문헌 중 하나다. 특히 방법론을 알고리즘 이름이 아니라 시스템 구성요소 기준으로 정리했다는 점이 강점이다. 이는 실제 의료기관 연합 환경에서 어떤 문제가 어느 계층에서 발생하는지 이해하는 데 직접 도움이 된다.

의료영상에서 FL을 연구하거나 적용하려면, 이 논문은 단순 입문서를 넘어서 설계 체크리스트 역할을 한다. client heterogeneity, aggregation fairness, privacy leakage, infrastructure constraints, unseen-site generalization까지 함께 봐야 한다는 점을 분명히 해 주기 때문이다.
