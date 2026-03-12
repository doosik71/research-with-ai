# Towards Foundation Models Learned from Anatomy in Medical Imaging via Self-Supervision

## 논문 메타데이터

- **제목**: Towards Foundation Models Learned from Anatomy in Medical Imaging via Self-Supervision
- **저자**: Mohammad Reza Hosseinzadeh Taher, Michael B. Gotway, Jianming Liang
- **학회/워크숍**: MICCAI 2023 Workshop on Domain Adaptation and Representation Transfer (DART 2023)
- **출판 연도**: 2024
- **시리즈**: Lecture Notes in Computer Science 14293
- **페이지**: 94-104
- **DOI**: 10.1007/978-3-031-45857-6_10
- **PMID**: 38752223
- **PMCID**: PMC11095552
- **arXiv ID**: 2309.15358
- **arXiv URL**: https://arxiv.org/abs/2309.15358
- **PMCID URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11095552/
- **코드 저장소**: https://github.com/JLiangLab/Eden

## 연구 배경 및 문제 정의

이 논문은 의료영상 분야의 foundation model이 왜 NLP나 일반 비전만큼 빠르게 성숙하지 못했는지에 대한 문제의식에서 출발한다. 저자들은 기존 self-supervised learning(SSL)이 이미지 수준의 불변성이나 patch 수준 복원에 집중했지만, 의료영상의 본질적 토대인 `인체 해부학` 자체를 학습 대상으로 삼지 못했다고 본다.

논문이 제시하는 핵심 전제는 명확하다.

- 의료영상의 기초는 해부학이다.
- 해부학은 `locality`와 `compositionality`라는 두 속성을 가진다.
- 따라서 의료영상 foundation model은 해부학 구조를 coarse-to-fine하게 이해하도록 학습되어야 한다.

여기서 locality는 서로 다른 해부학 구조가 형태적으로 구별 가능하다는 뜻이고, compositionality는 작은 구조가 상위 구조의 일부로 조합된다는 뜻이다. 저자들은 바로 이 두 속성을 embedding space에 반영하는 SSL 전략이 의료영상용 foundation model의 핵심이라고 주장한다.

## 핵심 기여

논문은 다음 세 가지 기여를 중심으로 구성된다.

1. 해부학의 계층 구조를 반영한 새로운 self-supervised training strategy를 제안했다.
2. embedding space가 실제로 해부학의 locality와 compositionality를 보존하는지 평가하는 해석 프레임을 제시했다.
3. 9개 다운스트림 과제와 few-shot 실험을 통해, 제안한 pretrained model `Adam`과 embedding `Eve`가 기존 fully supervised 및 SSL baseline보다 더 일반화된 표현을 학습함을 보였다.

이 논문은 단순히 "의료영상 SSL 성능을 조금 높였다"는 수준을 넘어, 의료영상 foundation model이 무엇을 학습해야 하는지에 대한 설계 원리를 제안한다는 점에서 의미가 있다.

## 방법론 요약

## 1. 전체 개념: Adam과 Eve

저자들은 사전학습된 모델을 `Adam`, 이 모델이 생성하는 dense embedding을 `Eve`라고 명명한다.

- **Adam**: autodidactic dense anatomical model
- **Eve**: semantic richness를 가진 embedding vectors

이 명명은 단순한 브랜딩이 아니라, 논문의 철학을 드러낸다. 즉 Adam은 라벨 없이 해부학을 스스로 학습하고, Eve는 그 결과로 형성된 조밀하고 의미론적인 표현 공간을 뜻한다.

## 2. Anatomy Decomposer

첫 번째 핵심 모듈은 `Anatomy Decomposer (AD)`다. AD는 입력 영상을 재귀적으로 분할해 서로 다른 granularity의 해부학 단위를 만든다.

동작은 다음과 같다.

- 입력 이미지를 먼저 수직으로 둘로 분할
- 이후 수평/수직 분할을 번갈아 적용
- granularity level `n`에 따라 더 세밀한 patch 집합 생성
- 그중 하나를 랜덤 샘플링해 학습 anchor로 사용

이 구조는 모델이 처음에는 큰 해부학 구조를, 이후에는 더 세밀한 하위 구조를 학습하도록 만든다. 즉, coarse-to-fine curriculum이 해부학 계층성과 직접 연결된다.

## 3. Purposive Pruner

두 번째 핵심 모듈은 `Purposive Pruner (PP)`다. 일반 contrastive learning에서는 anchor와 의미적으로 비슷한 구조가 negative sample로 들어오면 semantic collision이 발생할 수 있다. 예를 들어 서로 다른 환자의 동일 해부학 부위를 억지로 멀어지게 만들 수 있다.

PP는 이를 막기 위해 다음 절차를 쓴다.

- anchor feature와 memory bank 내 표본 간 cosine similarity 계산
- similarity가 임계값 이상인 표본 제거
- pruned memory bank만으로 InfoNCE loss 계산

즉, "비슷한 해부학 구조는 가까워야 한다"는 상식을 contrastive objective에 반영한 장치다.

## 4. 학습 전략의 핵심 논리

이 논문의 학습 전략은 세 문장으로 요약할 수 있다.

1. 해부학은 계층적이므로, 학습도 coarse-to-fine하게 진행해야 한다.
2. 같은 해부학 구조는 환자가 달라도 비슷한 embedding을 가져야 한다.
3. embedding space는 개별 구조의 구분성(locality)과 전체-부분 관계(compositionality)를 함께 가져야 한다.

이 점에서 논문은 단순한 patch-level MIM이나 일반 instance discrimination과 다르다. 설계 중심축이 "이미지 복원"이나 "전역 불변성"이 아니라 "해부학 이해"에 있다.

## 실험 설정

저자들은 ChestX-ray14와 EyePACS의 비라벨 데이터를 사전학습에 사용했고, ResNet-50을 backbone으로 채택했다. 사전학습 설정은 다음과 같다.

- optimizer: SGD
- initial learning rate: 0.03
- weight decay: 1e-4
- momentum: 0.9
- batch size: 256
- 입력 크기: `224 x 224`
- data granularity level: 최대 4

비교 대상은 MoCo-v2, TransVW, VICRegL, DenseCL, PCRL, DiRA, Medical-MAE, SimMIM과 ImageNet/ChestX-ray14 supervised pretraining이다.

다운스트림 평가는 분류, 분할, 탐지 등 9개 과제를 포함한다.

## 실험 결과와 해석

## 1. 다양한 다운스트림 과제에서의 일반화

논문은 Adam이 기존 SSL 방법들보다 일관되게 우수하며, fully supervised baseline과 비교해도 우수하거나 비슷한 성능을 보인다고 보고한다. 특히 저자들은 다음 비교를 강조한다.

- DenseCL, VICRegL 같은 dense SSL보다 우수
- PCRL, DiRA 같은 의료영상 SSL보다 우수
- Medical-MAE, SimMIM 같은 ViT 기반 masked modeling보다 우수하거나 경쟁적
- 해부학 반복 패턴을 학습하는 TransVW보다 큰 폭으로 우수

저자들의 해석은 분명하다. 단순 local feature나 patch dependency만으로는 부족하고, 해부학적 관계를 계층적으로 학습해야 진짜 전이 가능한 표현이 만들어진다는 것이다.

## 2. few-shot segmentation에서의 주목할 만한 이점

이 논문의 가장 설득력 있는 결과는 적은 라벨 상황이다. SCR-Heart와 SCR-Clavicle 분할에서 3, 6, 12, 24-shot 설정을 비교했을 때 Adam은 모든 구간에서 baseline보다 큰 폭으로 앞선다.

대표 수치는 다음과 같다.

- SCR-Heart 3-shot Dice: `84.35`
- SCR-Heart 24-shot Dice: `90.45`
- SCR-Clavicle 3-shot Dice: `66.69`
- SCR-Clavicle 24-shot Dice: `84.76`

논문은 SSL baseline 대비 분할 성능 향상이 대략 `9%~30%` 범위라고 요약한다. 이는 해부학 중심 pretraining이 annotation efficiency를 실제로 끌어올린다는 근거다.

## 3. embedding space의 locality 검증

저자들은 단순 성능 비교에 그치지 않고, 모델이 해부학을 정말 "이해"했는지 점검하려 한다. 이를 위해 ChestX-ray14 이미지 1000장에 대해 전문가가 10개 해부학 landmark를 수동 표시한 뒤, landmark 주변 patch feature를 추출해 t-SNE로 시각화한다.

결과적으로 Adam은 서로 다른 해부학 구조를 더 잘 분리된 클러스터로 배치했다. 이는 서로 다른 구조는 멀고, 같은 구조는 환자가 달라도 가깝게 모이는 locality 특성이 embedding space에 반영되었음을 시사한다.

## 4. compositionality 검증

compositionality 실험에서는 하나의 patch를 2, 3, 4개의 sub-patch로 분해한 뒤, 전체 patch embedding과 부분 embedding의 집계 결과 사이 cosine similarity를 비교한다. Adam은 baseline보다 분포가 더 좁고 평균 유사도가 1에 더 가깝다.

즉, Adam의 embedding space에서는 "전체 구조의 표현이 부분 구조 표현들의 조합과 잘 정렬된다"는 점이 관찰된다. 이 부분이 논문 제목의 "learned from anatomy"를 가장 직접적으로 뒷받침한다.

## 5. ablation 결과

ablation도 논문의 주장과 잘 맞물린다.

- granularity level을 점진적으로 늘릴수록 전 과제 성능이 일관되게 상승
- Purposive Pruner를 제거하면 성능이 유의하게 하락
- EyePACS 같은 다른 모달리티 사전학습에도 확장 가능성 확인

특히 EyePACS 사전학습 후 혈관 분할에서 top SSL 대비 `1.4%` 향상을 보였다는 점은, 제안 전략이 흉부 X-ray에만 묶이지 않는다는 근거다.

## 강점

## 1. 의료영상 foundation model의 설계 원리를 제안한다

많은 논문이 성능 개선에 집중하지만, 이 논문은 "의료영상에서 foundation model은 무엇을 학습해야 하는가"라는 더 근본적인 질문을 다룬다.

## 2. 해석 가능성 있는 평가 축을 제시한다

locality와 compositionality는 단순 accuracy보다 표현의 질을 더 잘 드러내는 지표다. 특히 의료영상처럼 의미 구조가 분명한 도메인에서 설득력이 있다.

## 3. few-shot 효율성이 강하다

실제 임상에서는 라벨이 비싸기 때문에, few-shot 개선은 실험실 수준의 성능 향상보다 더 실용적일 수 있다.

## 한계와 비판적 검토

## 1. foundation model이라는 표현은 다소 선제적이다

논문 제목에 foundation models가 들어가지만, 실제 모델 규모나 데이터 범위는 오늘날의 대규모 foundation model과는 거리가 있다. 보다 정확히는 "foundation model을 향한 해부학 기반 SSL 전략"에 가깝다.

## 2. backbone이 ResNet-50에 머문다

학습 철학은 새롭지만, 백본은 비교적 전통적이다. 따라서 최신 대규모 ViT, multimodal encoder, 3D foundation architecture로 확장했을 때도 같은 이점이 유지되는지는 별도 검증이 필요하다.

## 3. 해부학 분해가 실제 해부학 ontology를 직접 쓰는 것은 아니다

Anatomy Decomposer는 실제 장기 마스크나 해부학 atlas가 아니라 규칙적 이미지 분할을 이용한다. 즉, "해부학적"이라는 개념은 완전한 의미의 해부학 지식 주입이라기보다 계층적 공간 분해의 귀납편향에 더 가깝다.

## 4. 주된 검증 모달리티가 제한적이다

Chest X-ray 중심 검증이 강하고, 다른 모달리티 확장은 일부 실험으로만 제시된다. CT, MRI, 초음파 등 전반으로 일반화된다고 단정하기엔 아직 이르다.

## 실무적 및 연구적 인사이트

이 논문은 의료영상 SSL에서 단순히 더 큰 데이터나 더 큰 모델만이 답이 아니라는 점을 보여준다. 중요한 것은 도메인의 구조를 objective에 반영하는 것이다. 해부학은 의료영상의 공통 기반이므로, 해부학적 locality와 compositionality를 보존하는 표현 학습은 segmentation, detection, registration, landmark matching 같은 여러 과제로 자연스럽게 이어질 수 있다.

후속 연구 방향도 분명하다.

- 2D chest X-ray 중심 전략을 3D CT/MRI로 확장
- 실제 anatomy atlas나 organ hierarchy와의 결합
- vision-language 또는 report-grounded 의료 foundation model과 통합
- registration, landmark detection, surgical planning 같은 구조 중심 과제로 확장

## 종합 평가

`Towards Foundation Models Learned from Anatomy in Medical Imaging via Self-Supervision`은 의료영상 foundation model을 단순 규모의 문제가 아니라 `무엇을 학습하느냐`의 문제로 재정의한 논문이다. 제안한 Adam/Eve 프레임워크는 거대한 범용 모델이라기보다 해부학 중심 self-supervised representation learning의 설계 원형에 가깝지만, locality와 compositionality를 핵심 개념으로 명시하고 이를 실제 전이 성능과 few-shot 효율성으로 연결했다는 점에서 학술적 가치가 크다.

따라서 이 논문은 최신 의료 foundation model의 완성형을 제시한 문헌이라기보다, 의료영상 SSL이 해부학 이해를 중심으로 어떻게 재설계될 수 있는지를 보여주는 방향 제시형 논문으로 읽는 것이 가장 적절하다.
