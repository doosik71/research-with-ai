# Domain Adaptation for Medical Image Analysis: A Survey

## 논문 메타데이터

- 제목: Domain Adaptation for Medical Image Analysis: A Survey
- 저자: Hao Guan, Mingxia Liu
- 발표 형태: arXiv preprint
- arXiv: [2102.09508](https://arxiv.org/abs/2102.09508)
- 날짜: 2021년 2월 18일
- 키워드: domain adaptation, domain shift, machine learning, deep learning, medical image analysis

## 연구 배경 및 문제 정의

의료 영상 딥러닝 모델은 대개 학습 데이터와 테스트 데이터가 같은 분포를 따른다고 가정하지만, 실제 임상 환경에서는 이 가정이 거의 성립하지 않는다. 병원마다 스캐너 제조사와 촬영 프로토콜이 다르고, 환자군 구성과 해상도, 대조도, 노이즈 특성도 달라서 모델 성능이 쉽게 저하된다. 논문은 이 문제를 `domain shift`로 규정하고, 이를 줄이는 핵심 방법으로 domain adaptation(DA)을 다룬다.

저자들은 특히 의료 영상에서 DA가 더 중요한 이유를 두 가지로 본다. 첫째, 라벨링 비용이 매우 높아 매번 새 도메인에 맞춘 충분한 정답 데이터를 확보하기 어렵다. 둘째, 자연영상과 달리 의료 영상은 modality 자체가 매우 다양하고 3D/4D 구조를 가지므로, 단순 fine-tuning만으로는 일반화가 충분하지 않은 경우가 많다.

## 논문의 핵심 기여

- 의료 영상 분야의 domain adaptation 연구를 체계적으로 정리하는 survey를 제공한다.
- 방법론을 `shallow DA`와 `deep DA`로 나누고, 각각을 `supervised`, `semi-supervised`, `unsupervised`로 다시 구분한다.
- DA를 이해하기 위한 문제 설정 축으로 `label availability`, `modality difference`, `number of sources`, `adaptation steps`를 제시한다.
- 뇌, 폐, 심장, 안저, 유방, 복부, 병리 등 의료 영상 분야별 benchmark dataset을 정리한다.
- 의료 영상 DA의 구조적 난제와 미래 연구 방향을 3D/4D, unsupervised, multimodality, multi-source 관점에서 요약한다.

## DA의 기본 개념과 문제 설정

논문은 transfer learning과 domain adaptation의 관계부터 정리한다. transfer learning은 도메인 또는 태스크가 바뀌는 더 넓은 범주이고, domain adaptation은 태스크는 같지만 source domain과 target domain의 marginal distribution이 다른 경우에 초점을 맞춘다.

저자들이 제시한 DA 분류 축은 다음 네 가지다.

### 1. Label Availability

- supervised DA: target domain에 소량의 라벨이 존재
- semi-supervised DA: 소량의 라벨과 다수의 unlabeled target data가 존재
- unsupervised DA: target domain에는 unlabeled data만 존재

의료 영상에서는 라벨 확보가 가장 비싸기 때문에, 저자들은 unsupervised DA가 특히 중요하고 빠르게 성장하는 영역이라고 본다.

### 2. Modality Difference

- single-modality DA: source와 target이 같은 modality 내부에서 다름
- multi-modality DA: source와 target modality 자체가 다름

예를 들어 MRI vendor 간 적응은 single-modality DA이고, MRI에서 CT로의 적응은 cross-modality DA에 해당한다.

### 3. Number of Sources

- single-source DA
- multi-source DA

현실 의료 데이터는 여러 병원과 여러 스캐너에서 들어오므로 multi-source setting이 자연스럽지만, 실제 연구의 다수는 single-source에 머물러 있다고 정리한다.

### 4. Adaptation Steps

- one-step DA
- multi-step DA

source와 target의 차이가 매우 큰 경우에는 중간 도메인을 거치는 transitive adaptation이 필요할 수 있다는 점을 논문은 강조한다.

## 전체 taxonomy: Shallow DA와 Deep DA

이 survey의 가장 중요한 구조는 DA 방법을 `shallow`와 `deep`으로 나누는 것이다.

### 1. Shallow DA

shallow DA는 사람 손으로 설계한 특징이나 전통적 머신러닝 표현 위에서 수행되는 적응을 뜻한다. 대표 전략은 다음 두 가지다.

- instance weighting
- feature transformation

instance weighting은 target과 유사한 source 샘플에 더 높은 가중치를 주어 source-target 간 분포 차이를 줄이는 접근이다. feature transformation은 source와 target을 공통 latent space로 사상해 분포 간 간격을 줄이는 방식이다.

이 계열은 해석이 비교적 쉽고 데이터가 적을 때 유리할 수 있지만, 복잡한 3D 의료 영상의 표현을 학습하는 데는 한계가 있다.

### 2. Deep DA

deep DA는 CNN 등 딥러닝 표현학습과 적응을 end-to-end로 결합한다. 논문 시점 기준으로 의료 영상 DA의 중심축은 이미 deep DA로 이동해 있으며, 특히 adversarial learning, image translation, consistency learning이 핵심 도구로 자리잡고 있다고 본다.

## Shallow DA 정리

논문은 shallow DA도 여전히 의료영상에서 의미가 있다고 본다. 이유는 소량 데이터 환경에서 handcrafted feature와 간단한 적응 모델이 안정적으로 작동할 수 있기 때문이다.

대표 예시는 다음과 같다.

- instance weighting 기반 AD 분류
- PCA, subspace alignment, low-rank regularization 기반 뇌질환 분류
- feature matching 후 random forest 또는 SVM을 사용하는 접근
- multi-source latent space alignment 기반 ASD classification

이 흐름은 비교적 초기 의료영상 DA의 전형으로, 뇌 MRI나 fMRI 분류 문제에서 많이 활용됐다. 다만 survey의 전반적 메시지는 분명하다. 최근의 핵심 발전은 shallow보다 deep DA에서 일어나고 있다.

## Supervised Deep DA

supervised deep DA는 target domain의 소량 라벨을 활용하는 설정이다. 논문은 다음과 같은 흐름을 소개한다.

- ImageNet 사전학습 모델을 feature extractor로 사용한 뒤 shallow DA를 얹는 방법
- source domain에서 학습한 CNN의 상위층만 target 라벨로 fine-tuning하는 방법
- 3D CNN이나 3D U-Net을 source에서 학습한 뒤 target에 맞게 일부 층을 조정하는 방법

brain MRI classification, lesion segmentation, prostate segmentation, tumor segmentation 등의 예시에서 이런 접근이 쓰인다. 저자들은 supervised deep DA가 실용적이긴 하지만, target 라벨이 여전히 필요하다는 점에서 확장성 한계가 있다고 본다.

또 하나 중요한 지적은 자연영상용 2D CNN 사전학습이 의료영상에 항상 최선은 아니라는 점이다. 의료 영상은 3D 구조와 기관별 특성이 강하기 때문에, task-specific 3D backbone이 더 적합할 수 있다.

## Semi-Supervised Deep DA

semi-supervised DA에서는 적은 수의 labeled target data와 더 많은 unlabeled target data를 함께 사용한다. 논문에서 언급된 대표 방향은 다음과 같다.

- segmentation decoder와 reconstruction decoder를 함께 두는 Y-Net 계열
- labeled source, labeled target, generated sample을 함께 사용하는 semi-supervised GAN
- reconstruction loss를 활용해 domain-invariant representation을 학습하는 구조

이 설정은 supervised와 unsupervised 사이의 절충안으로 이해할 수 있다. 실제 임상에서는 완전 무라벨 target보다 소량의 anchor label을 확보하는 경우가 있으므로, survey는 이 설정이 실용적으로 의미 있다고 본다.

## Unsupervised Deep DA

이 논문의 핵심 비중은 unsupervised deep DA에 있다. 의료 영상 라벨 부족 문제를 고려하면 가장 중요한 연구 축이기 때문이다. 논문은 여러 세부 전략을 소개하지만, 큰 흐름은 다음과 같이 요약할 수 있다.

### 1. Adversarial feature alignment

가장 대표적인 구조는 DANN이다. feature extractor가 domain classifier를 속이도록 학습함으로써 source와 target이 구분되지 않는 domain-invariant representation을 학습한다.

의료영상에서는 다음과 같은 확장이 소개된다.

- multi-connected adversarial network
- segmentation network와 domain discriminator의 공동 학습
- low-level layer만 적응시키는 cross-modality adaptation
- edge detector나 ROI proposal을 추가해 보다 task-aware하게 정렬하는 방식

이 계열의 장점은 label 없이도 표현 수준 정렬이 가능하다는 점이지만, 지나친 정렬이 task-discriminative information까지 희석시킬 위험이 있다.

### 2. Image alignment / image translation

CycleGAN 기반 image-to-image translation은 의료영상 DA에서 매우 중요한 위치를 차지한다. source 이미지를 target 스타일로 바꾸거나, 반대로 real target을 synthetic source 스타일로 바꾸어 domain gap을 줄인다.

논문에서 강조하는 응용은 다음과 같다.

- MRI-CT 간 cross-modality segmentation
- whole-slide pathology image adaptation
- OCT denoising을 도메인 변환 문제로 재해석한 접근
- synthetic tumor image를 real-looking image로 변환해 segmentation 학습을 돕는 접근

이 흐름은 의료영상에서 style gap이 강할 때 매우 직관적이지만, 생성된 이미지가 실제 진단 신호를 왜곡하지 않는지 항상 주의가 필요하다.

### 3. Image + feature alignment 결합

단순 image translation만으로는 충분하지 않기 때문에, 일부 연구는 translated image와 real target image를 함께 feature-level adversarial learning에 넣는다. 논문은 이를 cross-modality cardiac segmentation과 vendor adaptation 사례로 소개한다.

이 전략은 image-level과 representation-level adaptation을 동시에 수행한다는 점에서 survey가 높게 평가하는 방향 중 하나다.

### 4. Disentangled representation

source와 target 이미지를 공통 content space와 domain-specific style space로 나누어 표현하는 접근이다. CT와 MRI처럼 modality가 다른 경우, 공통 해부학 정보는 content에 두고 modality appearance는 style에 두는 방식이 특히 유용하다.

이 논문은 disentangled representation이 multi-modality DA의 유망한 방향이라고 본다. 단순 adversarial alignment보다 구조적 해석이 더 명확하기 때문이다.

### 5. Self-ensembling / consistency learning

teacher-student 구조를 사용해 unlabeled target 예측의 일관성을 강제하는 방식이다. teacher는 student 파라미터의 exponential moving average로 유지되고, 두 네트워크의 target prediction 차이를 줄이는 consistency loss가 핵심이다.

이 방법은 spinal cord segmentation, brain tumor segmentation 같은 문제에서 소개되며, adversarial loss와 결합되기도 한다. 논문은 이 계열이 안정성과 성능 측면에서 실용적이라고 본다.

### 6. Soft labels and pseudo supervision

일부 연구는 source와 target 간 구조적 유사성을 찾아 heatmap이나 soft label을 만들고, 이를 target 학습 신호로 사용한다. 이는 완전한 pseudo-labeling의 초기 형태로 볼 수 있으며, 의료영상처럼 강한 구조 prior가 있는 경우 의미가 있다.

## Multi-Target / Lifelong Adaptation

논문 말미에 소개되는 multi-target deep DA와 lifelong adaptation도 중요하다. batch normalization 파라미터를 도메인별로 분리해 여러 스캐너와 프로토콜에 순차적으로 적응하는 접근은, 실제 병원 시스템 운영에 더 가까운 설정이다.

저자들은 이를 의료영상 DA의 실용적 확장 방향으로 본다. 새로운 병원이 들어올 때 기존 성능을 잃지 않으면서 빠르게 적응해야 하기 때문이다.

## Benchmark Dataset 정리

논문은 DA 연구를 지원하는 benchmark dataset을 장기적으로 정리해 둔 점도 강점이다. 대표적으로 다음이 포함된다.

- Brain: ADNI, AIBL, CADDementia, IXI, ABIDE, ISBI2015, BraTS, MICCAI WMH, HCP
- Lung: NIH ChestXray14, DLCST, COPDGene
- Heart: MM-WHS, NIH PLCO, NIH Chest
- Eye: DRIVE, STARE, SINA
- Breast: CBIS-DDSM, InBreast, CAMELYON
- Abdomen: PROMISE12, BWH, LiTS
- Histology and Microscopy: NKI, VGH, IHC

이 목록은 어떤 기관/장기/모달리티에서 DA 연구가 성숙했고, 어디가 아직 부족한지를 보여 주는 지도 역할을 한다.

## 의료영상 DA의 핵심 난제

논문은 도전 과제를 세 가지로 강하게 정리한다.

### 1. 3D/4D volumetric representation

의료영상은 본질적으로 3D 또는 4D인 경우가 많다. 하지만 많은 DA 방법은 2D 자연영상에서 출발했기 때문에 slice-level adaptation에 머무르거나, 공간-시간 구조를 충분히 활용하지 못한다. 저자들은 이 점을 가장 중요한 구조적 한계 중 하나로 본다.

### 2. Limited training data

의료 데이터는 절대량도 적고 라벨 데이터는 더 적다. 따라서 대규모 natural image pretraining의 이점을 그대로 가져오기 어렵고, deep DA는 쉽게 과적합되거나 불안정해질 수 있다.

### 3. Inter-modality heterogeneity

CT, structural MRI, fMRI, PET는 서로 시각 표현이 매우 다르다. 같은 환자라 해도 modality 간 격차가 너무 크기 때문에 단순한 feature alignment는 충분하지 않다. 특히 cross-modality DA는 여전히 어려운 열린 문제로 남아 있다.

## 미래 연구 방향

저자들은 다음 네 가지 방향을 제안한다.

### 1. Task-specific 3D/4D DA 모델

ImageNet 기반 2D CNN을 의료영상에 그대로 전이하는 대신, 3D/4D 구조와 ROI prior를 반영한 task-specific backbone이 필요하다고 본다.

### 2. Unsupervised DA와 그 이후

라벨 없는 target adaptation은 계속 중요해질 것이며, 더 나아가 target data조차 사용하지 않는 domain generalization과 zero-shot learning도 중요해질 것이라고 전망한다.

### 3. Multi-modality DA

MRI와 CT, PET 등 이질적인 modality 간 adaptation은 단일 modality보다 훨씬 어렵고, CycleGAN이나 disentanglement 같은 구조가 계속 발전해야 한다고 본다.

### 4. Multi-source / Multi-target DA

실제 의료 환경은 단일 병원보다 다기관 환경이 일반적이므로, 여러 source를 통합하고 여러 target으로 확장 가능한 DA가 장기적으로 더 중요하다고 본다.

## 비판적 평가

이 논문은 2021년 시점의 의료영상 DA 지형을 파악하는 데 매우 유용하다. 특히 문제 설정을 여러 축으로 나누고, shallow와 deep를 분리해 정리한 구조가 명확하다. benchmark dataset 정리도 실제 연구 설계에 직접 도움이 된다.

강점은 다음과 같다.

- medical DA 문제를 조직적으로 분류한다.
- supervised, semi-supervised, unsupervised를 균형 있게 다룬다.
- cross-modality와 multi-source 같은 현실적 설정을 별도 축으로 드러낸다.
- future direction이 비교적 실무적이다.

한계도 있다.

- 2021년 이후 빠르게 발전한 self-supervised pretraining, foundation model, test-time adaptation, diffusion 기반 adaptation 흐름은 반영되지 않는다.
- 방법 수가 많아 개별 접근의 정량 비교나 실패 사례 분석은 상대적으로 얕다.
- survey 시점상 adversarial DA 비중이 높고, 이후 더 중요해진 normalization-based adaptation이나 prompt-based adaptation은 다뤄지지 않는다.

## 연구적 시사점

이 논문이 주는 핵심 메시지는 다음과 같다.

- 의료영상 DA는 단순 fine-tuning 문제가 아니라, 스캐너·센터·모달리티 차이에서 오는 구조적 distribution shift 문제다.
- shallow DA는 여전히 의미가 있지만, 연구의 중심은 deep unsupervised DA로 이동하고 있다.
- cross-modality adaptation과 3D/4D volumetric adaptation이 향후 핵심 난제다.
- 실제 임상 배포를 생각하면 multi-source, lifelong, domain generalization까지 연결해서 봐야 한다.

## 종합 평가

`Domain Adaptation for Medical Image Analysis: A Survey`는 의료영상 DA 분야의 초창기 표준 정리 문헌에 가깝다. 분류 체계가 명확하고, 문제 설정과 데이터셋, 방법론, 미래 과제를 모두 한 프레임으로 연결한다는 점에서 가치가 크다.

특히 이 논문은 의료영상에서 domain shift가 얼마나 본질적인 문제인지, 그리고 왜 단순 전이학습만으로는 충분하지 않은지를 분명하게 보여 준다. 이후 등장한 foundation model 기반 적응 기법들을 읽을 때도, 이 논문이 정리한 supervised/semi-supervised/unsupervised, single-modality/cross-modality, single-source/multi-source 축은 여전히 유효한 기준점으로 작동한다.
