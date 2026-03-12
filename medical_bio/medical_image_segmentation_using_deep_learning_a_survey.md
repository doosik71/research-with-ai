# Medical Image Segmentation Using Deep Learning: A Survey

## 논문 메타데이터

- **제목**: Medical Image Segmentation Using Deep Learning: A Survey
- **저자**: Risheng Wang, Yu Meng, Zongyuan Ge, Jingke Ma, Jianfeng Zheng, Mingyang Ren, Yefeng Zheng
- **출판 연도**: 2021
- **저널**: IET Image Processing
- **arXiv ID**: 2009.13120
- **DOI**: 10.1049/ipr2.12419
- **arXiv URL**: https://arxiv.org/abs/2009.13120
- **PDF URL**: https://arxiv.org/pdf/2009.13120v3

## 연구 배경 및 문제 정의

이 논문은 의료영상 분할을 단순히 "U-Net 계열 모델 모음"으로 정리하지 않는다. 저자들은 의료영상 분할이 해부학적 구조, 병변, 장기 경계를 픽셀 또는 복셀 단위로 예측하는 문제이며, 진단 보조, 수술 계획, 방사선 치료 계획, 추적 관찰 정량화의 핵심 기반이라고 본다.

논문이 겨냥하는 핵심 문제는 다음과 같다.

- 의료영상은 CT, MRI, PET, 초음파, X-ray 등 modality가 다양하고 신호 특성이 서로 다르다.
- 병변은 작고 경계가 불명확하며 클래스 불균형이 심하다.
- 3D volumetric 데이터는 계산량과 메모리 비용이 크다.
- 픽셀 단위 annotation은 비싸고, weak label 또는 부분 라벨이 흔하다.
- 단순 supervised segmentation만으로는 실제 임상 데이터 분포 변화와 데이터 부족 문제를 해결하기 어렵다.

따라서 이 논문은 "어떤 모델이 최고인가"를 묻기보다, 의료영상 분할을 위한 딥러닝 방법론을 supervised learning, weakly supervised learning, advanced methods, datasets/challenges/future directions라는 축으로 재구성하는 데 초점을 둔다.

## 논문의 핵심 기여

이 논문의 기여는 새 아키텍처 제안이 아니라 survey의 분류 체계와 관점에 있다.

1. 의료영상 분할 딥러닝 연구를 `supervised`, `weakly supervised`, `advanced methods`, `datasets/challenges`로 나누어 체계화했다.
2. supervised segmentation을 다시 `network blocks`, `loss function`, `training strategy`, `post-processing` 관점으로 분해해 실무적인 설계 요소를 정리했다.
3. weak supervision을 `data augmentation`, `transfer learning`, `semi-supervised learning`으로 정리해 annotation scarcity 문제를 별도 축으로 다뤘다.
4. 당시 신흥 주제였던 `NAS`, `GCN`, `multi-modality fusion`, `medical transformer`를 차세대 방법군으로 묶어 소개했다.
5. 공개 데이터셋, 공통 난제, 향후 연구 방향을 함께 제시해 방법론 survey에 그치지 않고 연구 지도 역할을 하도록 구성했다.

## 방법론 구조 요약

## 1. Supervised learning 기반 분할

논문은 supervised segmentation 설계를 네 가지 블록으로 나눈다.

### 1.1 Network blocks

저자들은 encoder-decoder 구조를 기본 틀로 두고, 성능 차이를 만드는 세부 설계 요소를 정리한다.

- **Backbone**: FCN, U-Net, V-Net 같은 fully convolutional 계열이 중심이다.
- **Skip connection**: 고해상도 spatial detail을 decoder로 복원하기 위한 핵심 장치다.
- **Residual/Dense block**: 깊은 네트워크 학습 안정성과 feature reuse를 개선한다.
- **Attention block**: 병변 또는 장기 영역에 선택적으로 집중하도록 만든다.
- **Multi-scale context block**: dilated convolution, pyramid pooling 등으로 넓은 receptive field를 확보한다.
- **2D/3D convolution**: 2D는 효율적이고, 3D는 volumetric context를 직접 활용한다.

이 정리는 중요한데, 의료영상 분할 성능은 backbone 이름보다도 경계 복원, 다중 스케일 문맥, 3D 문맥 활용의 조합에 더 크게 좌우된다는 메시지를 준다.

### 1.2 Loss function

논문은 의료영상 분할에서 손실 함수가 단순 부속 요소가 아니라 class imbalance와 boundary sensitivity를 제어하는 핵심이라고 본다.

- **Cross-entropy loss**: 가장 기본적인 픽셀 단위 분류 손실
- **Dice loss**: foreground가 작을 때 overlap 중심 최적화에 유리
- **Weighted loss**: 클래스 불균형 완화
- **Hybrid loss**: Dice + CE, boundary-aware term 결합 등

특히 저자들은 작은 병변 분할이나 장기 대 배경 비율이 극단적인 문제에서 Dice 계열 손실이 매우 중요하다고 정리한다.

### 1.3 Training strategy

훈련 전략은 단순 optimizer 선택이 아니라 데이터 제약을 보완하는 기법군으로 설명된다.

- patch-based training
- hard sample mining
- cascaded/coarse-to-fine training
- multi-stage training
- deep supervision

이 전략들은 고해상도 3D 데이터를 한 번에 처리하기 어렵고, 작은 병변을 놓치기 쉬운 의료영상 환경에 맞춘 실용적 해법으로 제시된다.

### 1.4 Post-processing

후처리는 예측 결과를 해부학적으로 더 타당하게 보정하는 단계로 다뤄진다.

- CRF 기반 refinement
- connected component filtering
- morphology operation
- shape prior 또는 anatomical constraint 반영

논문은 segmentation 성능이 네트워크 자체만으로 결정되지 않으며, 임상적으로 말이 되는 mask를 만들기 위한 후처리가 여전히 유효하다고 본다.

## 2. Weakly supervised learning

이 논문은 weak supervision을 별도 장으로 다룬다. 이는 의료영상에서 완전한 mask annotation이 비싸기 때문이다.

### 2.1 Data augmentation

단순 회전, 이동, flipping을 넘어서 intensity variation, deformation, adversarial augmentation, synthetic sample generation까지 포함한다. 논문은 augmentation이 supervised baseline의 일반화 개선뿐 아니라 weak-label 환경에서 사실상 필수라고 본다.

### 2.2 Transfer learning

저자들은 자연영상 사전학습과 의료영상 도메인 전이를 모두 포함해 transfer learning을 설명한다. 다만 segmentation은 classification보다 구조적 정보가 중요하므로, 자연영상 pretrained model의 효과가 제한적일 수 있다는 점도 함께 언급한다.

### 2.3 Semi-supervised learning

pseudo-labeling, consistency regularization, teacher-student, adversarial semi-supervised learning 등이 소개된다. 논문의 시각은 명확하다. 의료영상 분할에서는 적은 수의 fully labeled sample과 많은 unlabeled sample이 흔하기 때문에, semi-supervised learning은 "옵션"이 아니라 실용적 표준 후보라는 것이다.

## 3. Advanced methods

논문은 2020년 전후의 신흥 흐름을 네 가지로 정리한다.

### 3.1 Neural architecture search

NAS는 수동 설계 대신 segmentation architecture를 자동 탐색하는 방향으로 소개된다. 저자들은 NAS가 성능 최적화 가능성은 높지만, 탐색 비용이 크고 의료 데이터셋 규모가 작아 과적합 위험이 있다고 본다.

### 3.2 Graph convolutional networks

GCN은 장기 간 관계, 혈관 구조, 해부학적 연결성처럼 CNN의 grid representation만으로 잡기 어려운 구조 정보를 반영하는 수단으로 설명된다. 이 논문은 GCN을 boundary refinement나 relational reasoning에 유용한 보조 모듈로 본다.

### 3.3 Multi-modality data fusion

CT-MRI, PET-CT, multiparametric MRI 같은 환경에서 modality 간 상보 정보를 융합하는 전략이 중요하다고 정리한다. 핵심 메시지는 단순 early fusion만으로 충분하지 않으며, modality별 표현 차이와 alignment 문제를 고려한 중간 또는 후기 융합이 필요하다는 점이다.

### 3.4 Medical transformer

이 논문이 나온 시점은 의료영상 segmentation에 Transformer가 본격적으로 확산되기 직전이다. 따라서 이 섹션은 후대 survey처럼 방대한 taxonomy는 아니고, self-attention이 장거리 의존성과 전역 문맥 모델링에 유리하다는 가능성을 짚는 예고편에 가깝다.

이 점은 중요하다. 지금 관점에서 보면 이 논문은 Transformer 시대 이전의 survey이면서도, 이미 다음 세대 구조 전환을 포착하고 있었다.

## 데이터셋, 평가지표, 정량 결과 해석

이 논문은 새로운 benchmark 실험을 수행하는 논문이 아니다. 대신 대표 데이터셋과 기존 연구 결과를 정리해 분할 연구의 평가 관행을 구조화한다.

### 1. 자주 언급되는 데이터셋 축

- **Brain MRI**: BraTS, iSEG, MRBrainS 등
- **Cardiac imaging**: ACDC 등
- **Abdominal / multi-organ CT**: BTCV 계열, LiTS 등
- **Retinal / fundus**: vessel, optic disc/cup segmentation 데이터셋
- **Skin / pathology / ultrasound**: 경계가 불명확하거나 texture 중심인 사례

논문은 특정 장기 하나보다 "어떤 modality와 구조적 난제를 가지는가"가 모델 설계 선택을 더 잘 설명한다고 본다.

### 2. 핵심 평가지표

- **Dice coefficient**
- **Jaccard / IoU**
- **Sensitivity / Specificity**
- **Hausdorff distance**
- **Average surface distance**

저자들은 overlap 지표만으로는 boundary quality를 충분히 설명하지 못하므로, surface distance 계열 지표가 함께 필요하다고 본다. 이는 의료영상 분할이 단순 픽셀 정확도보다 경계 위치 오차에 훨씬 민감하기 때문이다.

### 3. 결과 해석 방식

논문은 여러 표를 통해 대표 모델과 데이터셋별 성능을 비교하지만, 메타 분석 수준으로 통제된 공정 비교는 아니다. 즉, 실험 프로토콜, 전처리, 데이터 분할, 후처리, 2D/3D 설정이 서로 달라 숫자를 직접 가로비교하기는 어렵다.

이 한계에도 불구하고 논문이 주는 정성적 결론은 비교적 분명하다.

- encoder-decoder 계열이 여전히 기본 골격이다.
- Dice 기반 loss와 skip connection은 사실상 표준 요소다.
- 3D 문맥 활용은 중요하지만 계산 비용이 높다.
- weak supervision과 transfer learning은 데이터 부족 문제에 실질적으로 기여한다.
- multi-modality fusion과 transformer는 이후의 주류 확장 방향으로 보인다.

## 논문의 한계와 저자들이 제시한 미래 방향

논문은 미래 과제를 비교적 명확하게 정리한다.

### 1. 데이터 부족과 annotation 비용

완전한 segmentation mask 구축은 고비용이며 전문가 의존적이다. 따라서 weak supervision, semi-supervised learning, transfer learning이 계속 중요해질 것이라고 본다.

### 2. 클래스 불균형과 작은 병변 문제

배경 대비 병변 비율이 매우 작고 경계가 흐릿한 경우가 많아, loss 설계와 hard example 처리 전략이 중요하다.

### 3. 3D volumetric 계산 비용

3D CNN은 성능상 이점이 있지만 GPU 메모리와 추론 시간 측면의 제약이 크다. 이 때문에 2D, 2.5D, 3D hybrid가 현실적인 절충안으로 제시된다.

### 4. 도메인 이동과 일반화

장비, 병원, 프로토콜, 환자군 차이로 인한 분포 이동이 커서, 한 데이터셋에서 좋은 성능이 임상 전반으로 일반화되지 않을 수 있다.

### 5. 설명 가능성과 임상 신뢰성

분할 결과가 임상 의사결정에 직접 쓰이는 만큼, 단순 Dice 향상보다 오류 양상과 신뢰성 평가가 중요하다고 시사한다. 다만 이 논문은 explainability 자체를 깊게 다루지는 않는다.

## 실무적 관점의 해설

### 1. 이 논문은 "세대 전환 직전"의 survey다

이 논문은 U-Net 계열과 supervised segmentation이 중심이던 시기의 지형을 잘 요약한다. 동시에 NAS, GCN, transformer를 advanced methods로 묶어 다음 흐름을 예고한다. 따라서 의료영상 분할 연구사의 중간 지점을 이해하는 데 적합하다.

### 2. taxonomy가 실용적이다

많은 survey가 모델 이름 나열에 그치는데, 이 논문은 network block, loss, training strategy, post-processing으로 분해한다. 이 구조는 실제 구현과 실험 설계에 바로 연결되므로 연구 입문자에게 특히 유용하다.

### 3. weak supervision을 본론에 포함시킨 점이 강점이다

의료영상 segmentation의 현실 문제는 라벨 부족이다. 이 논문은 weak supervision을 부록 수준이 아니라 독립 장으로 다룬다. 이는 의료 도메인 특성을 정확히 짚은 구성이다.

### 4. 반대로 최신 관점에서는 범위가 빠르게 낡았다

2021년 이후의 핵심 흐름인 foundation model, promptable segmentation, SAM 계열 adaptation, diffusion-based segmentation, large-scale medical pretraining, domain generalization benchmark는 포함되지 않는다. 따라서 지금 읽을 때는 최신 survey라기보다 기반 survey로 보는 것이 정확하다.

## 후속 연구와의 연결

이 논문 이후 의료영상 분할은 대략 다음 방향으로 확장됐다.

- CNN 중심 구조에서 Transformer 및 hybrid 구조로 이동
- fully supervised 중심에서 semi-supervised, weakly supervised, few-shot, promptable setting으로 확장
- 단일 장기/단일 modality 중심에서 universal segmentation, foundation model 적응으로 확장
- 성능 중심 비교에서 robustness, generalization, efficiency, clinical reliability 평가로 확장

즉, 이 논문이 정리한 문제들 자체는 아직 유효하지만, 해법의 규모와 추상화 수준이 더 커졌다고 보는 편이 맞다.

## 종합 평가

`Medical Image Segmentation Using Deep Learning: A Survey`는 의료영상 분할 딥러닝을 구조적으로 이해하기 위한 좋은 기준점이다. 특히 supervised segmentation을 세부 설계 요소로 분해하고, weak supervision과 advanced methods를 한 문서 안에서 연결했다는 점이 강점이다.

다만 최신 연구 지형을 반영하는 survey로 읽기에는 한계가 있다. 오늘 기준에서는 foundation model 이전 시대의 의료영상 segmentation 지도를 제공하는 문서로 읽는 것이 가장 정확하다. 그럼에도 데이터 부족, 클래스 불균형, 3D 계산 비용, 도메인 이동 같은 핵심 문제 설정은 여전히 현재적이며, 후속 연구가 왜 그런 방향으로 발전했는지 이해하는 데 유용한 출발점이다.
