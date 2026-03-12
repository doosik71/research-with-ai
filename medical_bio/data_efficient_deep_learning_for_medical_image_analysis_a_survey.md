# Data Efficient Deep Learning for Medical Image Analysis: A Survey

## 논문 메타데이터

- 제목: Data efficient deep learning for medical image analysis: A survey
- 저자: Suruchi Kumari, Pravendra Singh
- 발표 형태: arXiv preprint
- arXiv: [2310.06557](https://arxiv.org/abs/2310.06557)
- 날짜: 2023년 10월 10일
- 키워드: data efficient deep learning, medical image analysis, inexact supervision, incomplete supervision, inaccurate supervision, limited supervision, no supervision

## 연구 배경 및 문제 정의

의료영상 딥러닝은 분류, 탐지, 분할, 등록 등에서 큰 성과를 냈지만, 성능 향상의 가장 큰 병목은 여전히 데이터다. 자연영상과 달리 의료 데이터는 규모가 작고, 개인정보 보호와 기관 정책 때문에 수집 자체가 어렵다. 게다가 실제로는 전체 데이터 중 극히 일부만 전문가가 라벨링한 상태이거나, 라벨의 품질도 완전하지 않은 경우가 많다.

이 논문은 이 문제를 단순히 `small data`라는 한 문장으로 다루지 않고, 어떤 종류의 supervision 결핍이 존재하는가에 따라 data-efficient learning을 체계적으로 분해한다. 즉 의료영상에서 데이터가 부족하다는 것은 단지 샘플 수가 적다는 뜻이 아니라, 라벨이 없거나, 부정확하거나, 거칠거나, 일부만 존재한다는 복합적 문제라는 점을 분명히 한다.

## 논문의 핵심 기여

- 의료영상 분야의 data-efficient deep learning 방법을 폭넓게 정리한 대규모 survey를 제공한다.
- 방법을 supervision 수준에 따라 `no supervision`, `inexact supervision`, `incomplete supervision`, `inaccurate supervision`, `only limited supervision`의 다섯 축으로 분류한다.
- 각 축을 다시 predictive/generative/contrastive SSL, MIL, weak annotation, semi-supervised learning, active learning, domain-adaptive learning, robust loss, few-shot learning, transfer learning 등으로 세분화한다.
- 의료영상에서 자주 쓰이는 benchmark dataset을 장기별로 폭넓게 정리한다.
- 미래 방향으로 continual learning, domain knowledge integration, vision transformer, NAS, federated learning, text supervision 등을 제안한다.

## 논문의 전체 taxonomy

이 survey의 핵심은 데이터 효율성을 supervision 수준의 관점에서 다시 정의했다는 점이다. 저자들은 Figure 1에서 data-efficient deep learning을 다음 다섯 범주로 나눈다.

### 1. No Supervision

라벨이 전혀 없는 상태에서 self-supervised learning으로 표현을 학습하는 경우다.

### 2. Inexact Supervision

정확한 픽셀 단위 라벨 대신 bag label, image-level label, point, scribble, bounding box 같은 거친 감독을 쓰는 경우다.

### 3. Incomplete Supervision

일부 데이터만 라벨이 있고 나머지는 unlabeled인 경우다. semi-supervised learning, active learning, domain-adaptive learning이 여기에 포함된다.

### 4. Inaccurate Supervision

라벨이 존재하지만 noisy하거나 틀릴 수 있는 경우다.

### 5. Only Limited Supervision

라벨 수 자체가 매우 적어서 augmentation, few-shot learning, transfer learning 같은 방법으로 보완하는 경우다.

이 분류의 장점은 의료영상 실무와 잘 맞는다는 점이다. 실제 데이터셋은 종종 이 범주들이 섞여 있지만, 어떤 종류의 supervision 문제가 중심인지에 따라 적절한 알고리즘 선택 기준을 제공한다.

## No Supervision: Self-Supervised Learning

논문은 no supervision 범주를 사실상 self-supervised learning(SSL)로 본다. 핵심은 pretext task를 잘 설계해 downstream task에 유용한 representation을 학습하는 것이다.

### 1. Predictive self-supervision

pretext task를 분류 또는 회귀로 정의하는 방식이다.

- anatomical position prediction
- jigsaw puzzle solving
- rotation prediction
- context restoration
- Rubik’s cube recovery 같은 3D 구조 기반 task

의료영상에서는 일반 자연영상용 pretext task를 그대로 쓰기보다, 해부학적 위치나 volume 구조를 반영하는 task가 더 효과적이라고 논문은 설명한다.

### 2. Generative self-supervision

입력을 복원하거나 재구성하는 pretext task를 사용한다.

- context restoration
- masked region reconstruction
- multimodal reconstruction
- image denoising / corruption recovery
- GAN 또는 reconstruction 기반 pretext

이 계열은 픽셀 수준 구조를 잘 보존할 수 있다는 장점이 있지만, semantic representation이 contrastive SSL보다 약할 수 있다는 trade-off가 있다.

### 3. Contrastive self-supervision

positive pair와 negative pair를 구분하도록 학습해 표현을 얻는 방식이다. SimCLR, MoCo류 아이디어를 의료영상에 맞게 adaptation한 연구들이 소개된다.

특히 논문은 의료영상에서는 augmentation 설계가 자연영상과 다르게 중요하다고 본다. 잘못된 augmentation은 병변 구조를 파괴하거나 의미를 훼손할 수 있기 때문이다.

### 4. Multi-self supervision

predictive, generative, contrastive task를 하나로 결합하는 방식이다. 단일 pretext task가 task-specific feature에 치우칠 수 있다는 문제를 줄이기 위해 여러 SSL 목적을 동시에 사용한다.

## Inexact Supervision

이 범주는 라벨이 아예 없는 것은 아니지만, 정밀도가 떨어지는 경우를 다룬다. 의료영상 annotation 비용이 높은 현실을 가장 잘 반영하는 범주 중 하나다.

### 1. Multiple Instance Learning (MIL)

MIL은 bag 단위 라벨만 있고 instance 단위 라벨은 없는 경우를 다룬다. 병리 whole-slide image처럼 큰 이미지에 cancer/non-cancer 라벨만 있는 상황이 대표적이다.

논문은 deep MIL을 크게 다음으로 나눈다.

- instance-based methods
- bag-based methods

attention-based MIL, clustering-based MIL, graph-based MIL 등이 소개되며, 특히 computational pathology에서 핵심 전략으로 자리잡고 있음을 보여 준다.

### 2. Learning with weak annotations

약한 라벨은 다음처럼 다양한 형태를 가진다.

- image-level annotation
- point annotation
- scribble-level supervision
- box-level supervision

point, scribble, box supervision은 full mask보다 훨씬 싸게 얻을 수 있기 때문에 segmentation에서 특히 중요하다. 논문은 이 범주에서 pseudo-label expansion, CRF refinement, seed region growing, box-tightness constraint 같은 아이디어를 정리한다.

핵심 메시지는, 약한 annotation은 supervision gap이 크지만, 해부학적 priors와 structured regularization을 이용하면 상당한 성능을 끌어낼 수 있다는 점이다.

## Incomplete Supervision

이 범주는 일부만 라벨이 있고 나머지 데이터는 unlabeled인 상황을 다룬다. survey는 이를 semi-supervised learning, active learning, domain-adaptive learning으로 나눈다.

### 1. Semi-Supervised Learning

논문은 semi-supervised learning을 네 가지로 분류한다.

- consistency regularization
- generative methods
- pseudo-labeling
- hybrid methods

consistency regularization에서는 π-model, temporal ensembling, mean teacher, uncertainty-aware mean teacher 같은 흐름이 정리된다. generative methods에서는 GAN과 VAE를 이용해 unlabeled data를 regularization 또는 data synthesis에 활용한다. pseudo-labeling에서는 confidence-aware pseudo labels, anti-curriculum pseudo labeling, teacher-student refinement가 핵심으로 소개된다.

의료영상에서는 3D volume, class imbalance, boundary ambiguity 때문에 일반 semi-supervised learning보다 uncertainty estimation과 multi-view consistency가 더 중요하다는 점이 반복적으로 강조된다.

### 2. Active Learning

active learning은 제한된 annotation budget 아래에서 어떤 샘플을 우선 라벨링할지 결정하는 방법이다. 논문은 uncertainty, representativeness, disagreement, loss prediction, suggestive annotation 등 다양한 기준을 소개한다.

의료영상에서는 단순 classification보다 segmentation annotation 비용이 높기 때문에, 어떤 샘플 하나를 고르는 문제의 가치가 특히 크다. 또한 cold start problem과 class imbalance가 핵심 이슈로 다뤄진다.

### 3. Domain-Adaptive Learning

여기서 말하는 domain-adaptive learning은 labeled source와 unlabeled target 간 domain shift를 줄이는 label-efficient adaptation이다. 논문은 이를 다음 흐름으로 정리한다.

- discrepancy minimization
- adversarial learning
- image translation
- disentangled representation
- pseudo-labeling based UDA
- self-supervision 활용 UDA

이 범주는 앞서 따로 정리한 medical domain adaptation survey와도 연결된다. 이 논문에서는 그것을 data-efficient 관점에서 재배치한다는 점이 특징이다.

## Inaccurate Supervision

라벨이 존재하더라도 noisy label이 섞여 있으면 supervised training은 쉽게 무너진다. 논문은 이를 세 가지로 분류한다.

### 1. Robust Loss Design

loss function 자체를 noise-tolerant하게 만드는 접근이다. generalized cross entropy, symmetric loss, uncertainty-aware loss 등이 소개된다.

### 2. Data Re-weighting

의심스러운 샘플의 영향을 줄이고, 신뢰도 높은 샘플에 더 큰 비중을 주는 방식이다. sample weighting, example reweighting, curriculum/anti-curriculum 접근이 여기에 포함된다.

### 3. Training Procedures

dual-branch consistency, mutual teaching, prototype refinement, label correction, clean/noisy split learning 같은 절차 중심의 접근이다.

이 범주는 의료영상 데이터가 항상 깨끗하다고 가정할 수 없다는 현실을 잘 반영한다. 실제로 crowd annotation, pseudo label, weak label이 섞이면 noisy supervision 문제는 더 중요해진다.

## Only Limited Supervision

이 범주는 라벨 수가 매우 적지만 품질은 상대적으로 괜찮은 경우를 다룬다.

### 1. Data Augmentation

논문은 augmentation을 크게 세 부분으로 정리한다.

- transformation of original data
- generation of artificial data
- learnable / task-specific augmentation

전통적인 affine, elastic, pixel-level transformation뿐 아니라, GAN 기반 synthetic sample generation, copy-paste 계열(TumorCP, InsMix, SelfMix, CarveMix, TensorMixup), 3D augmentation, learnable augmentation(AutoAugment류)까지 폭넓게 다룬다.

특히 의료영상에서는 병변 모양과 해부학 구조를 유지하는 augmentation이 중요하며, 자연영상용 augmentation을 그대로 적용하면 오히려 성능이 떨어질 수 있다는 함의가 있다.

### 2. Few-Shot Learning

few-shot learning은 support set과 query set을 기반으로 적은 예시만으로 새로운 task를 해결하는 방식이다. 논문은 이를 크게 다음으로 나눈다.

- training data enlargement
- metric-learning based methods
- meta-learning based methods
- others

Prototypical Network, MAML 계열, adaptive prototype extraction, self-supervised few-shot segmentation 등이 소개된다. 의료영상 few-shot 문제는 foreground/background 불균형, 기관 간 차이, 3D 구조라는 이유로 자연영상 few-shot보다 더 까다롭다고 논문은 시사한다.

### 3. Transfer Learning

transfer learning은 여전히 가장 실용적인 data-efficient baseline이다. 자연영상 pretrained CNN을 medical task에 fine-tune하는 고전적 전략부터, 2D→3D weight transfer, dimensional transfer learning, two-stage transfer, segmentation용 transfer까지 정리한다.

논문은 transfer learning이 classification에서는 강력하지만, segmentation과 volumetric medical imaging에서는 구조적 한계가 있을 수 있다고 본다. 그럼에도 적은 데이터 환경에서 여전히 매우 중요한 도구라는 점은 분명히 한다.

## Dataset 정리의 의미

이 논문은 dataset 표를 통해 data-efficient learning 연구가 실제로 어떤 장기와 modality에 집중되어 있는지를 보여 준다. ADNI, BraTS, MM-WHS, ACDC, DRIVE, ISIC, CAMELYON, CHAOS, KiTS, PROMISE12 등 다양한 benchmark가 정리되어 있다.

이 정리는 단순 참고 자료를 넘어, 어떤 종류의 supervision 문제가 어떤 장기에서 활발히 연구되는지 파악하는 데 유용하다. 예를 들어 pathology와 WSI는 MIL과 weak supervision, cardiac/abdominal segmentation은 semi-supervised와 domain adaptation, skin/fundus는 transfer learning과 active learning이 많이 쓰인다는 흐름을 읽을 수 있다.

## 미래 연구 방향

논문은 future scope에서 여러 흥미로운 방향을 제안한다.

### 1. Continual / Lifelong Learning

의료 시스템은 시간이 지나면서 새로운 질환과 새로운 데이터가 계속 들어오기 때문에, 기존 지식을 잃지 않으면서 새 지식을 흡수하는 continual learning이 중요하다고 본다.

### 2. Incorporating Domain Knowledge

해부학 정보, radiomics, patient metadata, textual report 같은 medical domain knowledge를 data-efficient learning에 통합하는 방향이 중요하다고 강조한다.

### 3. Label-Efficient Learning by Vision Transformers

transformer 기반 모델이 의료영상에서도 강력하지만 데이터 요구량이 큰 만큼, self-supervised transformer와 label-efficient pretraining이 중요하다고 본다.

### 4. Flexible Target Model Design / NAS

수작업 모델 설계 대신 NAS 같은 자동화된 구조 탐색도 향후 의미 있는 방향으로 제시된다.

### 5. Federated Learning

데이터 사일로와 privacy 문제를 고려하면 data-efficient learning과 federated learning의 결합이 자연스러운 미래 방향이라는 점을 시사한다.

### 6. Text Supervision

의료영상과 함께 존재하는 판독문, 임상 보고서 등을 supervision으로 활용하는 방향도 미래 가능성으로 제시된다.

## 비판적 평가

이 논문은 의료영상 data-efficient learning을 굉장히 넓은 범위에서 정리한다는 점이 강점이다. supervision quality 자체를 축으로 삼았기 때문에, SSL부터 noisy label, weak annotation, few-shot, transfer learning까지 하나의 프레임에서 볼 수 있다.

강점은 다음과 같다.

- 분류 체계가 명확하고 포괄적이다.
- 250편 이상을 다뤄 coverage가 넓다.
- dataset 정리가 실용적이다.
- 각 범주를 finer subcategory까지 세분화해 연구자 입장에서 탐색 경로가 좋다.

한계도 있다.

- 범위가 넓은 만큼, 각 세부 방법의 정량 비교나 실패 사례 분석은 깊지 않다.
- 2023년 이후 급격히 증가한 medical foundation model, multimodal VLM, large-scale text supervision 흐름은 제한적으로만 반영된다.
- supervision 범주들이 현실에서는 서로 섞여 있는데, taxonomy가 깔끔한 대신 이런 중첩 관계는 다소 단순화된다.

## 연구적 시사점

이 논문이 주는 핵심 메시지는 다음과 같다.

- 의료영상에서 데이터 효율성 문제는 단순 샘플 부족이 아니라 supervision의 결핍과 불완전성 문제다.
- 가장 유망한 방향은 하나의 기법보다, SSL + weak supervision + pseudo labeling + transfer learning처럼 여러 전략의 결합이다.
- 의료영상에서는 해부학 구조, 3D context, class imbalance, annotation cost를 반영한 task-specific 설계가 필수다.
- 앞으로는 domain knowledge, transformer pretraining, federated setting, text supervision이 data-efficient learning의 중심축이 될 가능성이 높다.

## 종합 평가

`Data efficient deep learning for medical image analysis: A survey`는 의료영상에서 `적은 데이터로 어떻게 학습할 것인가`라는 질문을 가장 넓고 체계적으로 정리한 문헌 중 하나다. 특히 supervision 수준을 중심으로 한 taxonomy는, 서로 다른 연구 흐름을 하나의 좌표계 안에 놓고 비교하게 해 준다는 점에서 가치가 크다.

의료영상 연구자가 self-supervised learning, weak supervision, semi-supervised learning, active learning, noisy label learning, few-shot learning, transfer learning 중 무엇을 택해야 할지 고민할 때, 이 논문은 좋은 출발점이자 설계 지도로 기능한다.
