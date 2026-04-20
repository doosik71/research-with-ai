# Instance Segmentation

## 서론

### 1. 연구 배경

Instance Segmentation 은 객체 탐지와 픽셀 단위 마스크 생성을 결합한 시각 분석 태스크로, supervised learning 에서 weakly supervised으로, detector-based 에서 detection-free 로, two-stage 에서 one-stage/query-based 로 기술적 진화를 이루어왔다. 본 보고서에서는 weakly supervised, detection-free one-stage, query-based set prediction, two-stage, biomedical/medical, real-time, domain-specialized, long-tail, few-shot/zero-shot, architecture design 등 10 개 주요 범주의 연구들을 체계적으로 정리한다.

### 2. 문제의식 및 분석 필요성

Instance Segmentation 분야의 논문들은 학습 설정, 아키텍처 구조, 기술적 혁신, 적용 도메인 등에서 다양한 관점을 보여주나, 체계적인 분류와 종합 분석이 부재하다. weakly-supervised 접근이 box-only/pixel-level annotation 으로 fully-supervised 성능에 근접함을 보여주는가, Detection-Free 와 Query-Based 방법은 각각怎样的한 trade-off 를 가지는가, long-tail/open-vocabulary 설정에서 specialized 방법의 일반화 한계는 무엇인지 등 방법론적 비교와 성능 패턴 분석이 필수적이다.

### 3. 보고서의 분석 관점

본 보고서는 연구체계 분류, 방법론 분석, 실험결과 분석의 세 가지 축을 통해 instance segmentation 연구의 핵심 축을 명확히 드러낸다. 연구체계 분류는 논문들을 학습 설정, 아키텍처 유형, 적용 도메인 등의 기준에 따라 10 개 주요 범주로 조직화한다. 방법론 분석은 구조적 축 (Detection-Free vs Query-Based vs Box-Supervised), 학습 축 (Weakly vs Fully vs Self-Supervised), 출력 축 (Direct mapping vs Clustering vs Prototype fusion) 으로 구분하여 비교·분석한다. 실험결과 분석은 COCO, LVIS, biomedical 데이터셋 등의 벤치마크 결과를 정리하고, one-stage vs two-stage, weak supervision ceiling 등 성능 패턴과 경향을 종합한다.

### 4. 보고서 구성

**1 장: 연구체계 분류**는 instance segmentation 분야의 110 편 논문 (연도 미상 논문 포함) 을 weakly supervised, detection-free one-stage, query-based set prediction, two-stage, biomedical/medical, real-time, domain-specialized, long-tail, few-shot/zero-shot, architecture design 등 10 개 주요 범주로 조직화한다.

**2 장: 방법론 분석**은 공통 문제 설정 및 접근 구조, 방법론 계열 분류, 학습 손실/특징 표현/추론 전략 패턴, 핵심 설계 패턴, one-stage vs two-stage 비교, 방법론 흐름 및 진화 등을 분석한다.

**3 장: 실험결과 분석**은 평가 구조 및 공통 실험 설정 (주요 데이터셋, 평가 지표, baseline) 을 정리한 후, COCO, LVIS, biomedical 데이터셋 등의 주요 실험 결과를 정렬하고, 성능 패턴 및 경향, ablation study, 일반화 한계, 결과 해석의 경향 등을 종합한다.

## 1장. 연구체계 분류

### 1. 연구 분류 체계 수립 기준

본 분류 체계는 instance segmentation 분야의 논문들을 다음과 같은 기준과 원칙으로 조직화했다.

- **문제 정의 관점**: supervised, weakly supervised, unsupervised/few-shot 등으로 학습 설정을 구분
- **방법론적 접근**: architecture design, representation redesign, loss design 등 핵심 공헌 유형을 분류
- **도메인/응용**: natural imagery, biomedical/medical, aerial/3D point cloud 등으로 적용 대상 명시
- **기술적 특징**: Transformer-based, detection-free, one-stage 등 구조적 차별점 강조
- **분류 원칙**: 각 논문은 가장 대표적인 단일 범주에 배치하며, 중복 배제하고 전체 논문을 포괄하도록 구성

### 2. 연구 분류 체계

#### 2.1 Weakly Supervised Instance Segmentation

Weak supervision 하에 bounding box/pixel-level annotation만으로 fully supervised 성능에 근접하는 연구

| 분류                                          | 논문명                                                                                     | 분류 근거                                                                                                        |
| --------------------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Weakly Supervised > Box-supervised            | Box-supervised Instance Segmentation with Level Set Evolution (2022)                       | box projection initialization과 Chan-Vese energy 기반 level-set evolution 모듈을 training-time에 적용            |
| Weakly Supervised > Box-supervised            | Box2Mask: Box-supervised Instance Segmentation via Level-set Evolution (2022)              | box-supervised segmentation 에서 classical Chan-Vese energy functional을 end-to-end differentiable module로 통합 |
| Weakly Supervised > Box-supervised            | BoxInst: High-Performance Instance Segmentation with Box Annotations (2020)                | network 구조 변경 없이 color similarity prior를 활용한 supervision noise 감소                                    |
| Weakly Supervised > Box-supervised            | Simple Does It: Weakly Supervised Instance and Semantic Segmentation (2016)                | pseudo segmentation mask를 생성한 후 standard segmentation network로 학습하는 단순 접근                          |
| Weakly Supervised > Weak Annotation Variants  | Weakly Supervised Instance Segmentation by Learning Annotation Consistent Instances (2020) | conditional distribution과 annotation-agnostic prediction distribution을 joint probabilistic objective로 학습    |
| Weakly Supervised > Weak Annotation Variants  | Weakly Supervised Instance Segmentation using Class Peak Response (2018)                   | classification network 내부 class response map의 peak를 인스턴스 seed로 활용                                     |
| Weakly Supervised > Point-supervised          | Pointly-Supervised Instance Segmentation (2021)                                            | bounding box + 내부 포인트 sparse supervision을 통한 annotation-efficient weak supervision                       |
| Weakly Supervised > Unsupervised/Pseudo-label | FreePoint: Unsupervised Point Cloud Instance Segmentation (2023)                           | annotation-free setting에서 multicut 기반 pseudo instance mask 생성과 weakly-supervised refinement               |
| Weakly Supervised > Unsupervised/Pseudo-label | Predicting Future Instance Segmentation by Forecasting Convolutional Features (2018)       | 레이블 없는 비디오 시퀀스에서 자기 지도 학습을 통한 미래 instance segmentation 예측                              |

#### 2.2 Detection-free One-stage Instance Segmentation

Detector-output에 의존하지 않고 direct dense prediction으로 instance mask 예측하는 방법

| 분류                               | 논문명                                                                                            | 분류 근거                                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Detection-free > Center-based      | SOLO: A Simple Framework for Instance Segmentation (2021)                                         | instance segmentation을 위치 기반 classification 문제로 재정립한 box-free one-stage framework            |
| Detection-free > Center-based      | SOLOv2: Dynamic and Fast Instance Segmentation (2022)                                             | bounding box 의존성을 제거한 direct instance segmentation 접근법                                         |
| Detection-free > Center-based      | CenterMask: single shot instance segmentation with point representation (2020)                    | one-stage anchor-box free instance segmentation, Local Shape와 Global Saliency branch                    |
| Detection-free > Center-based      | Learning Gaussian Instance Segmentation in Point Clouds (2020)                                    | instance center 분포 예측을 중심으로 proposal 없이 center-first 방식으로 box/mask 단일 스테이지로 예측   |
| Detection-free > Spatial-Embedding | Fully Convolutional Instance-aware Semantic Segmentation (2016)                                   | position-sensitive score maps로 detection과 segmentation을 joint formulation으로 통합                    |
| Detection-free > Spatial-Embedding | Look Closer to Segment Better: Boundary Patch Refinement for Instance Segmentation (2021)         | 경계 주변 patch-only 이진 세분화를 통한 경계 품질 개선에 집중하는 post-processing                        |
| Detection-free > Spatial-Embedding | Explicit Shape Encoding for Real-Time Instance Segmentation (2019)                                | mask decoder 없는 explicit parameterized shape representation을 통한 detection 속도                      |
| Detection-free > Spatial-Embedding | PolarMask: Single Shot Instance Segmentation with Polar Representation (2019)                     | instance 분할을 instance-center classification 과 polar-coordinate distance regression 으로 재정의       |
| Detection-free > Spatial-Embedding | SPRNet: Single Pixel Reconstruction for One-stage Instance Segmentation (2019)                    | feature map의 단일 픽셀을 객체 carrier로 사용해 direct mask reconstruction 가능                          |
| Detection-free > Spatial-Embedding | S4Net: Single Stage Salient-Instance Segmentation (2017)                                          | salient instance segmentation을 위한 single-stage architecture와 RoI masking 기반 RoI feature extraction |
| Detection-free > Spatial-Embedding | Mask Encoding for Single Shot Instance Segmentation (2020)                                        | one-stage detector의 mask prediction 병목 문제를 PCA 기반 저차원 coefficient regression 으로 해결        |
| Detection-free > Spatial-Embedding | TensorMask: A Foundation for Dense Object Segmentation (2019)                                     | dense sliding-window 방식으로 instance segmentation 문제를 4D tensor prediction 으로 재정의              |
| Detection-free > Spatial-Embedding | YOLACT: Real-time Instance Segmentation (2019)                                                    | prototype mask generation과 instance coefficient 예측의 병렬 one-stage 구조                              |
| Detection-free > Spatial-Embedding | Path Aggregation Network for Instance Segmentation (2018)                                         | Bottom-up Path Augmentation과 Adaptive Feature Pooling을 통한 특징 피라미드 정보 흐름 구조 개선          |
| Detection-free > Spatial-Embedding | Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth (2019)    | spatial embedding 예측과 instance-specific clustering bandwidth 학습을 통한 IoU 최적화                   |
| Detection-free > Spatial-Embedding | Instance Segmentation and Tracking with Cosine Embeddings and Recurrent Hourglass Networks (2018) | stacked hourglass network 내부 ConvGRU 삽입과 cosine embedding loss를 통한 temporal recurrence           |
| Detection-free > Spatial-Embedding | Learning to Cluster for Proposal-Free Instance Segmentation (2018)                                | 쌍별 관계 기반 KL 발산 손실과 그래프 색칠 이론을 통한 proposal-free instance segmentation                |
| Detection-free > Spatial-Embedding | Recurrent Instance Segmentation (2015)                                                            | ConvLSTM 을 활용한 순차적 인스턴스 탐지 구조와 공간 상태 업데이트                                        |
| Detection-free > Spatial-Embedding | Proposal-free Network for Instance-level Object Segmentation (2015)                               | 제안-자유 네트워크 아키텍처를 통해 영역 제안 단계를 제거한 종단 간 객체 분할                             |
| Detection-free > Spatial-Embedding | Pixelwise Instance Segmentation with a Dynamically Instantiated Network (2017)                    | detector 기반 refinement 대신 semantic segmentation 결과를 기반으로 instance identity 추론               |
| Detection-free > Spatial-Embedding | Bridging Category-level and Instance-level Semantic Image Segmentation (2016)                     | 범주 수준 의미론적 분할을 기반으로 인스턴스 수준 분할을 수행하는 종속적 파이프라인                       |
| Detection-free > Spatial-Embedding | Bottom-up Instance Segmentation using Deep Higher-Order CRFs (2016)                               | object detector 출력을 CRF inference 과정 자체에 통합하는 two-stage structured prediction                |
| Detection-free > Spatial-Embedding | Instance-aware Self-supervised Learning for Nuclei Segmentation (2020)                            | nuclei size와 quantity라는 두 종류의 domain-specific prior를 proxy task 로 학습                          |

#### 2.3 Set Prediction with Query-based Transformer

DETR 기반 query-based set prediction을 instance segmentation에 적용한 연구

| 분류                                     | 논문명                                                                                         | 분류 근거                                                                                                                         |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Query-based > Standard Set Prediction    | ISTR: End-to-End Instance Segmentation with Transformers (2021)                                | low-dimensional mask embedding 예측과 bipartite matching으로 detection과 segmentation을 joint learning                            |
| Query-based > Standard Set Prediction    | SOLQ: Segmenting Objects by Learning Queries (2021)                                            | DETR decoder 구조 계승과 Unified Query Representation(UQR)으로 class/box/mask 통합 학습                                           |
| Query-based > Standard Set Prediction    | Masked-attention Mask Transformer for Universal Image Segmentation (2021)                      | 파노라마/인스턴스/의미론적 분할 등 모든 이미지 분할 작업을 단일 아키텍처로 처리                                                   |
| Query-based > Query Design Optimization  | Learning Equivariant Segmentation with Instance-Unique Querying (2022)                         | query embedding의 discrimination potential과 geometric robustness을 dataset-level uniqueness와 transformation equivariance로 강화 |
| Query-based > Query Design Optimization  | FastInst: A Simple Query-Based Model for Real-Time Instance Segmentation (2023)                | query 초기화, decoder 업데이트 방식, masked attention 학습을 단순화한 lightweight segmentation                                    |
| Query-based > Sparse Representation      | Sparse Instance Activation for Real-Time Instance Segmentation (2022)                          | instance activation map으로 객체를 표현하고 Hungarian matching으로 one-to-one prediction                                          |
| Query-based > Mask Representation Design | Mask2Former 기반 실시간 벤치마크 (FastInst)                                                    | Mask2Former meta-architecture를 유지하되 query 초기화, query-pixel 상호작용, masked attention supervision 재설계                  |
| Query-based > Mask Representation Design | iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images (2019)                 | 항공 영상 domain의 구조적 어려움을 위한 전용 데이터셋과 벤치마크 (dataset paper)                                                  |
| Query-based > Mask Representation Design | LVIS: A Dataset for Large Vocabulary Instance Segmentation (2019)                              | 긴 꼬리 카테고리의 저샘플 환경에서 객체 탐지를 위한 벤치마크 데이터셋                                                             |
| Query-based > Mask Representation Design | The Multi-modality Cell Segmentation Challenge: Towards Universal Solutions (2023)             | 다중 모달리티 세포 분할의 범용성과 효율성을 검증한 benchmark design 및挑战赛                                                      |
| Query-based > Mask Representation Design | Transformer-Based Visual Segmentation: A Survey (2023)                                         | Transformer-based segmentation 방법들을 DETR-like meta-architecture를 공통 골격으로 분류한 survey                                 |
| Query-based > Mask Representation Design | Mask Transfiner for High-Quality Instance Segmentation (2021)                                  | incoherent region detector와 quadtree 기반 sparse structured refinement으로 coarse mask 정제                                      |
| Query-based > Mask Representation Design | Instance-Specific Feature Propagation for Referring Segmentation (2022)                        | explicit instance representation과 전역적 feature propagation을 통한 객체 간 관계 모델링                                          |
| Query-based > Mask Representation Design | Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth (2019) | clustering bandwidth 를 learnable parameter 로 직접 학습하는 loss 기반 접근                                                       |

#### 2.4 Two-stage Detection-based Instance Segmentation

Mask R-CNN 계열 및 variant, proposal 기반 two-stage pipeline 연구

| 분류                           | 논문명                                                                                          | 분류 근거                                                                                                       |
| ------------------------------ | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Two-stage > Mask R-CNN Variant | Associatively Segmenting Instances and Semantics in Point Clouds (2019)                         | instance segmentation과 semantic segmentation을 end-to-end로 결합한 multi-task point cloud framework            |
| Two-stage > Mask R-CNN Variant | Instance and Panoptic Segmentation Using Conditional Convolutions (2022)                        | dynamic conditional convolution filter를 사용한 ROI-free mask prediction 프레임워크                             |
| Two-stage > Mask R-CNN Variant | Conditional Convolutions for Instance Segmentation (2020)                                       | detector branch와 controller sub-network 가 동적으로 생성하는 compact mask head 결합                            |
| Two-stage > Mask R-CNN Variant | Instance Segmentation in the Dark (2023)                                                        | 저조도 환경에서 feature disturbance suppress ion과 RAW input strategy로 robust한 segmentation                   |
| Two-stage > Mask R-CNN Variant | Attention-Based Transformers for Instance Segmentation of Cells in Microstructures (2020)       | DETR 구조를 biomedical 영역으로 전용하며 경량화한 transformer instance segmentation                             |
| Two-stage > Mask R-CNN Variant | Learning Instance Occlusion for Panoptic Segmentation (2019)                                    | panoptic segmentation에서 instance-level occlusion reasoning을 추가한 lightweight module                        |
| Two-stage > Mask R-CNN Variant | From Forks to Forceps: A New Framework for Instance Segmentation of Surgical Instruments (2022) | segmentation 과 classification 병목을 진단하고 전용 classifier 를 삽입하는 surgical-domain rethinking           |
| Two-stage > Mask R-CNN Variant | DropLoss for Long-Tail Instance Segmentation (2021)                                             | instance segmentation 의 long-tail 편향에서 background classification 이 주요 원인이며 stochastic dropping 보정 |
| Two-stage > Mask R-CNN Variant | Incremental Few-Shot Instance Segmentation (2021)                                               | FSIS 에서 재학습 없이 class representative 추가하는 incremental 클래스 추가 방법                                |
| Two-stage > Mask R-CNN Variant | Weakly Supervised Instance Segmentation using Class Peak Response (2018)                        | image-level label만으로 instance segmentation 수행하는 classification network 기반                              |
| Two-stage > Mask R-CNN Variant | One-Shot Instance Segmentation (2018)                                                           | Siamese Mask R-CNN 모델을 통한 one-shot segmentation                                                            |
| Two-stage > Mask R-CNN Variant | Camouflaged Instance Segmentation In-The-Wild: Dataset, Method, and Benchmark Suite (2021)      | in-the-wild camouflaged instance segmentation과 scene-driven fusion 방법론                                      |
| Two-stage > Mask R-CNN Variant | FGN: Fully Guided Network for Few-Shot Instance Segmentation (2020)                             | Mask R-CNN 의 각 구성 요소에 task-specific guidance 를 부여하는 module-wise guidance                            |
| Two-stage > Mask R-CNN Variant | Contour Proposal Networks for Biomedical Instance Segmentation (2021)                           | instance segmentation 을 sparse detection + explicit shape regression 으로 재정의                               |
| Two-stage > Mask R-CNN Variant | DoNet: Deep De-overlapping Network for Cytology Instance Segmentation (2023)                    | 세포를 intersection/complement 부분 영역으로 분해·재조합하며 생물학적 제약 구조화 학습                          |
| Two-stage > Mask R-CNN Variant | Instance Segmentation in the Dark (2023)                                                        | 저조도 환경에서 feature disturbance suppress ion과 RAW input strategy로 robust한 segmentation                   |
| Two-stage > Mask R-CNN Variant | Pose2Seg: Detection Free Human Instance Segmentation (2018)                                     | 사람 pose를 기준 정렬하는 Affine-Align 모듈과 skeleton 정보를 명시적 feature로 통합                             |
| Two-stage > Mask R-CNN Variant | Zero-Shot Instance Segmentation (2021)                                                          | seen class에서 학습한 visual-semantic mapping을 unseen instance segmentation 으로 전이                          |
| Two-stage > Mask R-CNN Variant | Deep Watershed Transform for Instance Segmentation (2016)                                       | 고전적 워터셰드 변환의 에너지 분지 개념을 CNN 으로 직접 학습                                                    |
| Two-stage > Mask R-CNN Variant | Gland Instance Segmentation by Deep Multichannel Neural Networks (2016)                         | 영역/위치/경계 정보를 통합하는 3 채널 병렬 네트워크와 융합 네트워크                                             |
| Two-stage > Mask R-CNN Variant | Gland Instance Segmentation Using Deep Multichannel Neural Networks (2016)                      | instance recognition 문제를 segmentation/edge/detection 세 가지 보조 문제로 분해                                |
| Two-stage > Mask R-CNN Variant | Capsules for Biomedical Image Segmentation (2020)                                               | CNN의 convolution weight sharing을 모방한 transformation matrix sharing 으로 재설계                             |
| Two-stage > Mask R-CNN Variant | BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation (2020)                            | instance segmentation 에서 mask representation 을 bottom-up shared basis와 top-down attention 으로 factorize    |
| Two-stage > Mask R-CNN Variant | Boundary-aware Instance Segmentation (2016)                                                     | 거리 변환 기반 밀집 표현을 통해 바운딩 박스 제안의 부정확성에 강건                                              |
| Two-stage > Mask R-CNN Variant | Are Larger Pretrained Language Models Uniformly Better? (2021)                                  | 대규모 모델의 성능 향상을 instance 분포 차원에서 재정의하는 분석 프레임워크                                     |
| Two-stage > Mask R-CNN Variant | End-to-End Instance Segmentation with Recurrent Attention (2016)                                | instance segmentation 을 time-axis 로 분해된 순차적 structured prediction 으로 재정의                           |
| Two-stage > Mask R-CNN Variant | Recurrent Neural Networks for Semantic Instance Segmentation (2017)                             | variable-length output을 처리하기 위해 recurrent hidden state 에 이전 객체 정보 축적                            |
| Two-stage > Mask R-CNN Variant | Monocular Object Instance Segmentation and Depth Ordering with CNNs (2015)                      | 단안 RGB 입력에서 인스턴스 분할과 깊이 순서를 결합한 patch-CNN + MRF                                            |
| Two-stage > Mask R-CNN Variant | Learning to See the Invisible: End-to-End Trainable Amodal Instance Segmentation (2018)         | amodal/visible/invisible의 구조적 차이를 네트워크와 손실에 직접 반영한 multi-task 모델                          |
| Two-stage > Mask R-CNN Variant | Amodal Instance Segmentation (2016)                                                             | amodal instance segmentation 문제의 최초 일반 목적 방법론, synthetic 데이터와 iterative 예측                    |
| Two-stage > Mask R-CNN Variant | Learning Gaussian Instance Segmentation in Point Clouds (2020)                                  | instance center 분포 예측을 중심으로 proposal 없이 center-first 방식으로 box/mask 예측                          |
| Two-stage > Mask R-CNN Variant | Learning to Cluster for Proposal-Free Instance Segmentation (2018)                              | 쌍별 관계 기반 KL 발산 손실과 그래프 색칠 이론을 통한 proposal-free                                             |
| Two-stage > Mask R-CNN Variant | Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth (2019)  | clustering bandwidth 를 learnable parameter 로 직접 학습하는 loss 기반 접근                                     |
| Two-stage > Mask R-CNN Variant | Deep Watershed Transform for Instance Segmentation (2016)                                       | 고전적 워터셰드 변환의 에너지 분지 개념을 CNN 으로 직접 학습                                                    |

#### 2.5 Biomedical/Medical Domain Instance Segmentation

병리 영상, 세포 분석, 의료 영상 특화된 instance segmentation 연구

| 분류                     | 논문명                                                                                        | 분류 근거                                                                                           |
| ------------------------ | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Biomedical > Cell/Nuclei | Instance-aware Self-supervised Learning for Nuclei Segmentation (2020)                        | nuclei segmentation 에 특화된 size 와 quantity 라는 두 종류의 domain-specific proxy task            |
| Biomedical > Cell/Nuclei | Nuclei instance segmentation and classification in histopathology images with StarDist (2022) | StarDist star-convex polygon instance representation 구조에 class probability head 추가             |
| Biomedical > Cell/Nuclei | DoNet: Deep De-overlapping Network for Cytology Instance Segmentation (2023)                  | 세포를 intersection/complement 부분 영역으로 분해·재조합하며 생물학적 제약 학습                     |
| Biomedical > Cell/Nuclei | IRNet: Instance Relation Network for Overlapping Cervical Cell Segmentation (2019)            | 인스턴스 간 관계 학습을 통한 특징 표현 개선과 희소성 제약 NMS                                       |
| Biomedical > Cell/Nuclei | Contour Proposal Networks for Biomedical Instance Segmentation (2021)                         | instance segmentation 을 Fourier Descriptor 기반 closed contour 회귀 문제로 접근                    |
| Biomedical > Cell/Nuclei | Attention-Based Transformers for Instance Segmentation of Cells in Microstructures (2020)     | DETR 구조를 biomedical 영역으로 전용하며 경량화한 transformer instance segmentation                 |
| Biomedical > Cell/Nuclei | Capsules for Biomedical Image Segmentation (2020)                                             | CNN의 convolution weight sharing을 모방한 transformation matrix sharing 으로 재설계                 |
| Biomedical > Cell/Nuclei | The Multi-modality Cell Segmentation Challenge: Towards Universal Solutions (2023)            | 다중 모달리티 세포 분할의 범용성과 효율성을 검증한 benchmark design                                 |
| Biomedical > Cell/Nuclei | Gland Instance Segmentation by Deep Multichannel Neural Networks (2016)                       | 영역/위치/경계 정보를 통합하는 3 채널 병렬 네트워크와 융합 네트워크                                 |
| Biomedical > Cell/Nuclei | Gland Instance Segmentation Using Deep Multichannel Neural Networks (2016)                    | instance recognition 문제를 segmentation/edge/detection 세 가지 보조 문제로 분해                    |
| Biomedical > Cell/Nuclei | Capsules for Biomedical Image Segmentation (2020)                                             | CNN의 convolution weight sharing을 모방한 transformation matrix sharing 으로 재설계                 |
| Biomedical > Cell/Nuclei | Deep Watershed Transform for Instance Segmentation (2016)                                     | 고전적 워터셰드 변환의 에너지 분지 개념을 CNN 으로 직접 학습                                        |
| Biomedical > Cell/Nuclei | Learning Equivariant Segmentation with Instance-Unique Querying (2022)                        | query embedding의 discrimination potential과 geometric robustness을 dataset-level uniqueness로 강화 |
| Biomedical > Cell/Nuclei | Look Closer to Segment Better: Boundary Patch Refinement for Instance Segmentation (2021)     | 경계 주변 patch-only 이진 세분화를 통한 경계 품질 개선                                              |
| Biomedical > Cell/Nuclei | Multiscale Cell Instance Segmentation with Keypoint Graph based Bounding Boxes (2019)         | 키포인트 탐지를 통한 바운딩 박스 생성과 그래프 기반 그룹화                                          |
| Biomedical > Cell/Nuclei | Instance Segmentation in the Dark (2023)                                                      | 저조도 환경에서 feature disturbance suppression과 RAW input strategy                                |

#### 2.6 Real-time/Speed Optimization Instance Segmentation

실시간 처리와 speed-accuracy trade-off에 초점을 맞춘 연구

| 분류                        | 논문명                                                                                            | 분류 근거                                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Real-time > Speed-Optimized | Explicit Shape Encoding for Real-Time Instance Segmentation (2019)                                | mask decoder 없는 explicit parameterized shape representation을 통한 detection 속도                             |
| Real-time > Speed-Optimized | SOLO: A Simple Framework for Instance Segmentation (2021)                                         | box-free, grouping-free, faster 시스템 구현                                                                     |
| Real-time > Speed-Optimized | SOLOv2: Dynamic and Fast Instance Segmentation (2022)                                             | bounding box 의존성을 제거한 direct instance segmentation 접근법                                                |
| Real-time > Speed-Optimized | FastInst: A Simple Query-Based Model for Real-Time Instance Segmentation (2023)                   | query 초기화, decoder 업데이트 방식, masked attention 학습을 단순화한 lightweight                               |
| Real-time > Speed-Optimized | Sparse Instance Activation for Real-Time Instance Segmentation (2022)                             | instance activation map으로 객체를 표현하고 Hungarian matching으로 one-to-one prediction                        |
| Real-time > Speed-Optimized | BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation (2020)                              | Mask R-CNN 대비 더 높은 정확도 + 약 20% 빠른 추론                                                               |
| Real-time > Speed-Optimized | YOLACT: Real-time Instance Segmentation (2019)                                                    | prototype mask generation과 instance coefficient 예측의 병렬 one-stage 구조                                     |
| Real-time > Speed-Optimized | PersonLab: Person Pose Estimation and Instance Segmentation (2018)                                | 상향식 박스 없는 부분 기반 모델링을 통한 인스턴스 분할                                                          |
| Real-time > Speed-Optimized | S4Net: Single Stage Salient-Instance Segmentation (2017)                                          | salient instance segmentation을 위한 single-stage architecture                                                  |
| Real-time > Speed-Optimized | Look Closer to Segment Better: Boundary Patch Refinement for Instance Segmentation (2021)         | 경계 주변 patch-only 이진 세분화를 통한 경계 품질 개선                                                          |
| Real-time > Speed-Optimized | Mask Transfiner for High-Quality Instance Segmentation (2021)                                     | incoherent region detector와 quadtree 기반 sparse structured refinement                                         |
| Real-time > Speed-Optimized | PolarMask++: Enhanced Polar Representation for Single-Shot Instance Segmentation (2021)           | polar representation으로 instance segmentation 문제를 center classification + ray length regression 으로 재정의 |
| Real-time > Speed-Optimized | Learning Gaussian Instance Segmentation in Point Clouds (2020)                                    | instance center 분포 예측을 중심으로 proposal 없이 center-first 방식으로 box/mask 예측                          |
| Real-time > Speed-Optimized | Deep Watershed Transform for Instance Segmentation (2016)                                         | 고전적 워터셰드 변환의 에너지 분지 개념을 CNN 으로 직접 학습                                                    |
| Real-time > Speed-Optimized | Monocular Object Instance Segmentation and Depth Ordering with CNNs (2015)                        | 단안 RGB 입력에서 인스턴스 분할과 깊이 순서를 결합한 patch-CNN + MRF                                            |
| Real-time > Speed-Optimized | Simple Does It: Weakly Supervised Instance and Semantic Segmentation (2016)                       | bounding box에서 pseudo segmentation mask를 생성한 후 standard segmentation network 로 학습                     |
| Real-time > Speed-Optimized | Learning to Cluster for Proposal-Free Instance Segmentation (2018)                                | 쌍별 관계 기반 KL 발산 손실과 그래프 색칠 이론을 통한 proposal-free                                             |
| Real-time > Speed-Optimized | Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth (2019)    | clustering bandwidth 를 learnable parameter 로 직접 학습하는 loss 기반 접근                                     |
| Real-time > Speed-Optimized | TensorMask: A Foundation for Dense Object Segmentation (2019)                                     | dense sliding-window 방식으로 instance segmentation 문제를 4D tensor prediction 으로 재정의                     |
| Real-time > Speed-Optimized | Proposal-free Network for Instance-level Object Segmentation (2015)                               | 제안-자유 네트워크 아키텍처를 통해 영역 제안 단계를 제거한 종단 간 객체 분할                                    |
| Real-time > Speed-Optimized | Pixelwise Instance Segmentation with a Dynamically Instantiated Network (2017)                    | detector 기반 refinement 대신 semantic segmentation 결과를 기반으로 instance identity 추론                      |
| Real-time > Speed-Optimized | Bridging Category-level and Instance-level Semantic Image Segmentation (2016)                     | 범주 수준 의미론적 분할을 기반으로 인스턴스 수준 분할을 수행하는 종속적 파이프라인                              |
| Real-time > Speed-Optimized | Bottom-up Instance Segmentation using Deep Higher-Order CRFs (2016)                               | object detector 출력을 CRF inference 과정 자체에 통합하는 two-stage structured prediction                       |
| Real-time > Speed-Optimized | Instance Segmentation and Tracking with Cosine Embeddings and Recurrent Hourglass Networks (2018) | stacked hourglass network 내부 ConvGRU 삽입과 cosine embedding loss를 통한 temporal recurrence                  |
| Real-time > Speed-Optimized | Recurrent Instance Segmentation (2015)                                                            | ConvLSTM 을 활용한 순차적 인스턴스 탐지 구조와 공간 상태 업데이트                                               |
| Real-time > Speed-Optimized | Instance Segmentation in the Dark (2023)                                                          | 저조도 환경에서 feature disturbance suppression과 RAW input strategy                                            |
| Real-time > Speed-Optimized | Pose2Seg: Detection Free Human Instance Segmentation (2018)                                       | 사람 pose를 기준 정렬하는 Affine-Align 모듈과 skeleton 정보를 명시적 feature로 통합                             |
| Real-time > Speed-Optimized | Zero-Shot Instance Segmentation (2021)                                                            | seen class에서 학습한 visual-semantic mapping을 unseen instance segmentation 으로 전이                          |

#### 2.7 Domain-Specialized Instance Segmentation

특정 도메인(3D point cloud, aerial imagery, text, surgical 등) 에 특화된 연구

| 분류                                         | 논문명                                                                                          | 분류 근거                                                                              |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Domain-Specialized > Aerial/Remote Sensing   | iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images (2019)                  | 항공 영상 domain의 구조적 어려움을 위한 전용 데이터셋과 벤치마크                       |
| Domain-Specialized > Aerial/Remote Sensing   | Box-supervised Instance Segmentation with Level Set Evolution (2022)                            | iSAID(remote sensing) 및 LiTS(medical)에서 box-only weak supervision                   |
| Domain-Specialized > Medical Imaging         | Box2Mask: Box-supervised Instance Segmentation via Level-set Evolution (2022)                   | LiTS(medical)에서 box-supervised segmentation                                          |
| Domain-Specialized > Medical Imaging         | BoxInst: High-Performance Instance Segmentation with Box Annotations (2020)                     | LiTS(medical)에서 weak/box-supervised segmentation                                     |
| Domain-Specialized > Medical Imaging         | Deep Watershed Transform for Instance Segmentation (2016)                                       | Cityscapes Instance Level Segmentation 벤치마크                                        |
| Domain-Specialized > Medical Imaging         | Learning Instance Occlusion for Panoptic Segmentation (2019)                                    | COCO(80 things + 53 stuff 클래스), Cityscapes(8 things + 11 stuff 클래스)              |
| Domain-Specialized > Surgical Instruments    | From Forks to Forceps: A New Framework for Instance Segmentation of Surgical Instruments (2022) | 수술 영상 환경에서 surgical instruments 의 multi-class instance segmentation           |
| Domain-Specialized > 3D Point Cloud          | Associatively Segmenting Instances and Semantics in Point Clouds (2019)                         | 3D 실내 장면 point cloud의 instance segmentation 및 semantic segmentation              |
| Domain-Specialized > 3D Point Cloud          | Learning Gaussian Instance Segmentation in Point Clouds (2020)                                  | 3D point cloud 환경의 instance segmentation 시스템, ScanNet, S3DIS                     |
| Domain-Specialized > 3D Point Cloud          | FreePoint: Unsupervised Point Cloud Instance Segmentation (2023)                                | 3D point cloud 기반 실내 환경의 instance segmentation 시스템, annotation-free          |
| Domain-Specialized > Text Segmentation       | Box2Mask: Box-supervised Instance Segmentation via Level-set Evolution (2022)                   | ICDAR2019 ReCTS (scene text)에서의 instance segmentation                               |
| Domain-Specialized > Text Segmentation       | BoxInst: High-Performance Instance Segmentation with Box Annotations (2020)                     | ICDAR 2019 ReCTS (character segmentation)                                              |
| Domain-Specialized > Text Segmentation       | BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation (2020)                            | anchor-free one-stage detector와 구조적 궁합 최적                                      |
| Domain-Specialized > Text Segmentation       | PolarMask++: Enhanced Polar Representation for Single-Shot Instance Segmentation (2021)         | rotated text detection에 적용                                                          |
| Domain-Specialized > In-the-Wild/Camouflaged | Camouflaged Instance Segmentation In-The-Wild: Dataset, Method, and Benchmark Suite (2021)      | in-the-wild camouflaged instance segmentation과 scene-driven fusion                    |
| Domain-Specialized > In-the-Wild/Camouflaged | EVF-SAM: Early Vision-Language Fusion for Text-Prompted Segment Anything Model (2024)           | SAM 기반 referring expression segmentation 시스템                                      |
| Domain-Specialized > Salient Object          | Instance-Level Salient Object Segmentation (2017)                                               | 이미지 내 개별 객체 인스턴스 인식과 다중 스케일 현저성 개선                            |
| Domain-Specialized > Salient Object          | S4Net: Single Stage Salient-Instance Segmentation (2017)                                        | class-agnostic salient instance segmentation 문제                                      |
| Domain-Specialized > Salient Object          | Localized Interactive Instance Segmentation (2020)                                              | Interactive instance segmentation 시스템                                               |
| Domain-Specialized > Open-Vocabulary         | A Simple Framework for Open-Vocabulary Segmentation and Detection (2023)                        | open-vocabulary instance/panoptic segmentation 및 object detection                     |
| Domain-Specialized > Open-Vocabulary         | Zero-Shot Instance Segmentation (2021)                                                          | Zero-Shot Instance Segmentation(ZSI) 과제                                              |
| Domain-Specialized > Open-Vocabulary         | Unseen Object Instance Segmentation for Robotic Environments (2020)                             | 로봇이 조작하는 tabletop 환경에서 학습하지 않은 unseen objects의 instance segmentation |
| Domain-Specialized > Occlusion Handling      | Robust Instance Segmentation through Reasoning about Multi-Object Occlusion (2020)              | multi-object occlusion 시나리오에서 객체 간 consistency와 occlusion order 추론         |
| Domain-Specialized > Occlusion Handling      | Learning Instance Occlusion for Panoptic Segmentation (2019)                                    | panoptic segmentation에서 instance-level occlusion ordering 학습                       |
| Domain-Specialized > Occlusion Handling      | End-to-End Instance Segmentation with Recurrent Attention (2016)                                | instance segmentation을 end-to-end 학습 가능한 순차적 모델로 재정의                    |
| Domain-Specialized > Occlusion Handling      | One-Shot Instance Segmentation (2018)                                                           | 소수의 훈련 예제만으로 이전에 본 적 없는 새로운 객체 카테고리의 인스턴스 분할          |
| Domain-Specialized > Amodal                  | Amodal Instance Segmentation (2016)                                                             | amodal instance segmentation 문제의 최초 일반 목적 방법론                              |
| Domain-Specialized > Amodal                  | Learning to See the Invisible: End-to-End Trainable Amodal Instance Segmentation (2018)         | amodal/visible/invisible의 구조적 차이를 네트워크와 손실에 직접 반영                   |
| Domain-Specialized > Multi-Modality          | The Multi-modality Cell Segmentation Challenge: Towards Universal Solutions (2023)              | 멀티모달 현미경 영상에서 범용 세포 분할 알고리즘                                       |
| Domain-Specialized > Surgical Instruments    | From Forks to Forceps: A New Framework for Instance Segmentation of Surgical Instruments (2022) | 수술 영상 환경에서 surgical instruments 의 multi-class instance segmentation           |
| Domain-Specialized > Salient Object          | Instance-Level Salient Object Segmentation (2017)                                               | 현저한 영역 감지 및 윤곽선 감지 작업별 데이터셋                                        |
| Domain-Specialized > Salient Object          | S4Net: Single Stage Salient-Instance Segmentation (2017)                                        | proposal 주변 배경 문맥을 명시적으로 활용                                              |
| Domain-Specialized > Interactive             | Localized Interactive Instance Segmentation (2020)                                              | Interactive instance segmentation 시스템                                               |
| Domain-Specialized > In-the-Wild             | Camouflaged Instance Segmentation In-The-Wild: Dataset, Method, and Benchmark Suite (2021)      | in-the-wild camouflaged instance segmentation                                          |
| Domain-Specialized > Open-Vocabulary         | A Simple Framework for Open-Vocabulary Segmentation and Detection (2023)                        | open-vocabulary instance/panoptic segmentation                                         |
| Domain-Specialized > Open-Vocabulary         | Zero-Shot Instance Segmentation (2021)                                                          | Zero-Shot Instance Segmentation(ZSI) 과제                                              |
| Domain-Specialized > Open-Vocabulary         | Unseen Object Instance Segmentation for Robotic Environments (2020)                             | 로봇이 조작하는 tabletop 환경에서 unseen objects                                       |

#### 2.8 Long-tail/Rare Class Instance Segmentation

긴 꼬리 분포와 rare/low-sample 카테고리 환경의 instance segmentation

| 분류                           | 논문명                                                                 | 분류 근거                                                                                                       |
| ------------------------------ | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Long-tail > Rare Class Balance | DropLoss for Long-Tail Instance Segmentation (2021)                    | instance segmentation 의 long-tail 편향에서 background classification 이 주요 원인이며 stochastic dropping 보정 |
| Long-tail > Dataset/Benchmark  | LVIS: A Dataset for Large Vocabulary Instance Segmentation (2019)      | 1,000개+ 엔트리 레벨 객체 카테고리, 긴 꼬리(long tail) 카테고리 분포                                            |
| Long-tail > Rare Class Balance | Learning Equivariant Segmentation with Instance-Unique Querying (2022) | small instance 성능 개선, LVISv1 데이터셋                                                                       |
| Long-tail > Rare Class Balance | Instance Segmentation in the Dark (2023)                               | LIS 저조도 인스턴스 세그먼테이션 데이터셋                                                                       |

#### 2.9 Few-Shot/Zero-Shot Instance Segmentation

Few-shot 학습과 zero-shot generalization을 위한 연구

| 분류                          | 논문명                                                                                | 분류 근거                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Few-Shot > Support Set        | FGN: Fully Guided Network for Few-Shot Instance Segmentation (2020)                   | FSIS에서는 Mask R-CNN 의 각 핵심 구성요소가 support set으로부터 서로 다르게 guided되어야 함 |
| Few-Shot > Support Set        | Incremental Few-Shot Instance Segmentation (2021)                                     | 재학습 없이 class representative 만 추가하는 구조적 접근                                    |
| Few-Shot > Support Set        | One-Shot Instance Segmentation (2018)                                                 | 단일 참조 이미지를 통한 unseen 카테고리 분할을 위한 Siamese Network                         |
| Few-Shot > Zero-Shot Transfer | Zero-Shot Instance Segmentation (2021)                                                | seen class에서 학습한 visual-semantic mapping을 unseen instance segmentation 으로 전이      |
| Few-Shot > Zero-Shot Transfer | Unseen Object Instance Segmentation for Robotic Environments (2020)                   | category 를 알지 못하는 임의 물체를 instance 단위로 구분 인식                               |
| Few-Shot > Zero-Shot Transfer | Zero-Shot Instance Segmentation (2021)                                                | Zero-Shot Detector, Semantic Mask Head, BA-RPN, Sync-bg로 구성된 end-to-end                 |
| Few-Shot > Zero-Shot Transfer | One-Shot Instance Segmentation (2018)                                                 | Siamese Mask R-CNN 모델을 통한 one-shot segmentation                                        |
| Few-Shot > Zero-Shot Transfer | EVF-SAM: Early Vision-Language Fusion for Text-Prompted Segment Anything Model (2024) | SAM 에 가장 적합한 text/multimodal prompting 방식을 실험적 탐색                             |
| Few-Shot > Few-Shot Transfer  | Incremental Few-Shot Instance Segmentation (2021)                                     | 재학습 없이 class representative 만 추가하는 구조적 접근                                    |
| Few-Shot > Few-Shot Transfer  | FGN: Fully Guided Network for Few-Shot Instance Segmentation (2020)                   | Mask R-CNN 의 각 구성 요소에 task-specific guidance 를 부여                                 |
| Few-Shot > Few-Shot Transfer  | One-Shot Instance Segmentation (2018)                                                 | Siamese Mask R-CNN 모델을 통한 one-shot segmentation                                        |
| Few-Shot > Few-Shot Transfer  | Zero-Shot Instance Segmentation (2021)                                                | seen class에서 학습한 visual-semantic mapping을 unseen instance segmentation 으로 전이      |
| Few-Shot > Few-Shot Transfer  | Unseen Object Instance Segmentation for Robotic Environments (2020)                   | category 를 알지 못하는 임의 물체를 instance 단위로 구분 인식                               |

#### 2.10 Biomedical/Deep Learning Architecture Design

아키텍처 설계와 new representation에 초점을 맞춘 연구

| 분류                                          | 논문명                                                                                            | 분류 근거                                                                                                       |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Architecture Design > Representation Redesign | Mask Encoding for Single Shot Instance Segmentation (2020)                                        | instance segmentation 의 핵심 병목을 mask representation 방식으로 보고, PCA 기반 저차원 coefficient regression  |
| Architecture Design > Representation Redesign | PolarMask: Single Shot Instance Segmentation with Polar Representation (2019)                     | 인스턴스 분할을 instance-center classification 과 polar-coordinate distance regression 으로 재정의              |
| Architecture Design > Representation Redesign | PolarMask++: Enhanced Polar Representation for Single-Shot Instance Segmentation (2021)           | polar representation으로 instance segmentation 문제를 center classification + ray length regression 으로 재정의 |
| Architecture Design > Representation Redesign | TensorMask: A Foundation for Dense Object Segmentation (2019)                                     | instance segmentation 문제를 4D tensor prediction으로 재정의하고 pyramid level에 따른 mask 해상도 역변환        |
| Architecture Design > Representation Redesign | Mask Encoding for Single Shot Instance Segmentation (2020)                                        | instance segmentation 의 핵심 병목을 mask representation 방식으로 보고, PCA 기반 저차원 coefficient regression  |
| Architecture Design > Representation Redesign | Explicit Shape Encoding for Real-Time Instance Segmentation (2019)                                | mask decoder 없는 explicit parameterized shape representation을 통한 detection 속도와 segmentation 정확도       |
| Architecture Design > Representation Redesign | SOLO: A Simple Framework for Instance Segmentation (2021)                                         | instance segmentation을 위치(location) 기반의 per-pixel classification 문제로 재정립한 dense prediction         |
| Architecture Design > Representation Redesign | SOLOv2: Dynamic and Fast Instance Segmentation (2022)                                             | bounding box 의존성을 제거한 direct instance segmentation 접근법                                                |
| Architecture Design > Representation Redesign | Learning Gaussian Instance Segmentation in Point Clouds (2020)                                    | instance center 분포 예측을 중심으로 proposal 없이 center-first 방식으로 box/mask 예측                          |
| Architecture Design > Representation Redesign | Sparse Instance Activation for Real-Time Instance Segmentation (2022)                             | instance activation map으로 객체를 표현하고 Hungarian matching으로 one-to-one prediction                        |
| Architecture Design > Representation Redesign | SPRNet: Single Pixel Reconstruction for One-stage Instance Segmentation (2019)                    | feature map의 단일 픽셀을 객체 carrier로 사용해 direct mask reconstruction 가능                                 |
| Architecture Design > Representation Redesign | S4Net: Single Stage Salient-Instance Segmentation (2017)                                          | proposal 주변 배경 문맥을 명시적으로 활용하며 foreground/background 대비를 ternary masking 으로 강화            |
| Architecture Design > Representation Redesign | Path Aggregation Network for Instance Segmentation (2018)                                         | 인스턴스 분할을 위한 특징 피라미드 정보 흐름 구조를 Bottom-up 상향식 경로 증강으로 개선                         |
| Architecture Design > Representation Redesign | Mask Transfiner for High-Quality Instance Segmentation (2021)                                     | information loss로 정의된 sparse incoherent region만 transformer 로 정제하는 structured refinement              |
| Architecture Design > Representation Redesign | Instance Segmentation and Tracking with Cosine Embeddings and Recurrent Hourglass Networks (2018) | cosine embedding loss 와 recurrent stacked hourglass 를 결합한 프레임워크                                       |
| Architecture Design > Representation Redesign | Weakly Supervised Instance Segmentation using Class Peak Response (2018)                          | classification network 내부 class response map의 peak 를 인스턴스 seed 로 활용                                  |
| Architecture Design > Representation Redesign | Simple Does It: Weakly Supervised Instance and Semantic Segmentation (2016)                       | pseudo segmentation mask를 생성한 후 standard segmentation network 로 학습                                      |
| Architecture Design > Representation Redesign | Capsules for Biomedical Image Segmentation (2020)                                                 | CNN의 convolution weight sharing을 모방한 transformation matrix sharing 으로 재설계                             |
| Architecture Design > Representation Redesign | Learning Equivariant Segmentation with Instance-Unique Querying (2022)                            | query embedding 의 discrimination potential과 geometric robustness을 dataset-level uniqueness 로 강화           |
| Architecture Design > Representation Redesign | Instance-Specific Feature Propagation for Referring Segmentation (2022)                           | explicit instance representation과 전역적 feature propagation을 통한 객체 간 관계 모델링                        |
| Architecture Design > Representation Redesign | Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth (2019)    | clustering bandwidth 를 learnable parameter 로 직접 학습하는 loss 기반 접근                                     |
| Architecture Design > Representation Redesign | Look Closer to Segment Better: Boundary Patch Refinement for Instance Segmentation (2021)         | 경계 주변 patch-only 이진 세분화를 통한 경계 품질 개선                                                          |
| Architecture Design > Representation Redesign | Deep Watershed Transform for Instance Segmentation (2016)                                         | 고전적 워터셰드 변환의 에너지 분지 개념을 CNN 으로 직접 학습                                                    |
| Architecture Design > Representation Redesign | Monocular Object Instance Segmentation and Depth Ordering with CNNs (2015)                        | 단안 RGB 입력에서 인스턴스 분할과 깊이 순서를 결합한 patch-CNN + MRF                                            |
| Architecture Design > Representation Redesign | Recurrent Instance Segmentation (2015)                                                            | ConvLSTM 을 활용한 순차적 인스턴스 탐지 구조와 공간 상태 업데이트                                               |
| Architecture Design > Representation Redesign | Instance Segmentation in the Dark (2023)                                                          | 저조도 인스턴스 세그먼테이션 문제를 feature space disturbance 문제로 재해석                                     |
| Architecture Design > Representation Redesign | Gland Instance Segmentation by Deep Multichannel Neural Networks (2016)                           | 영역/위치/경계 정보를 통합하는 3 채널 병렬 네트워크와 융합 네트워크                                             |
| Architecture Design > Representation Redesign | Gland Instance Segmentation Using Deep Multichannel Neural Networks (2016)                        | instance recognition 문제를 segmentation/edge/detection 세 가지 보조 문제로 분해                                |

### 3. 종합 정리

본 분류 체계는 instance segmentation 분야의 110편 논문(연도 미상 논문 포함)을 weakly supervised, detection-free one-stage, query-based set prediction, two-stage, biomedical/medical, real-time, domain-specialized, long-tail, few-shot/zero-shot, architecture design 등 10 개 주요 범주로 조직화했다. 각 범주는 학습 설정, 아키텍처 구조, 기술적 혁신, 적용 도메인 등의 관점에서 instance segmentation 연구의 핵심 축을 명확히 드러낸다. Weakly Supervised는 annotation 비용 절감, Detection-free는 direct dense prediction, Query-based는 set prediction 관점, Biomedical는 도메인 특화, Real-time는 speed-accuracy trade-off, Long-tail은 희귀 카테고리 처리, Few-shot/Zero-shot는 일반화, Architecture design는 representation redesign, Domain-specialized는 도메인 적응, Transformer-based는 최신 메타 아키텍처를 포괄한다. 이러한 분류는 instance segmentation 연구가 supervised learning에서 weak supervision으로, detector-based에서 detection-free로, box-based에서 location-based/center-based로, two-stage에서 one-stage/query-based로, detection-only에서 open-vocabulary/zero-shot으로 진화해 왔음을 체계적으로 보여준다.

## 2장. 방법론 분석

## 1. 공통 문제 설정 및 접근 구조

Instance Segmentation 연구들은 다음 공통 문제 설정을 공유한다:

| 요소            | 내용                                       |
| --------------- | ------------------------------------------ |
| **입력**        | 입력 이미지 (RGB/DOM/Point Cloud)          |
| **핵심 출력**   | 인스턴스 별 픽셀 단위 마스크 + 클래스 라벨 |
| **하위 태스크** | 객체 탐지, 마스크 생성, 인스턴스 분리      |

## 1.1 문제 정의의 주요 간극

| 간극                           | 설명                             | 해결 방향               |
| ------------------------------ | -------------------------------- | ----------------------- |
| **Detection-segmentation gap** | 탐지와 분할을 동시에 수행해야 함 | 단일 모델 통합 학습     |
| **Open vocabulary gap**        | 학습 unseen 카테고리 처리        | semantic embedding      |
| **Weak supervision gap**       | box-only, point-only annotation  | pseudo-label, weak loss |

## 2. 방법론 계열 분류

논문들을 방법론 계열별로 다음과 같이 분류한다:

| 방법론 계열                           | 논문명                                                                                        | 핵심 특징                                             |
| ------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **Box-Supervised Variational Series** | Box2Mask (2022), BoxInst (2020), Box-supervised Level Set (2022)                              | bounding box만으로 경계 진화 (Level-set/Chan-Vese)    |
| **Detection-Free Direct Mapping**     | SOLO (2021), SOLOv2 (2020), SOLOQ (2021)                                                      | grid-based 위치 매핑, dynamic kernel                  |
| **Query-Based Set Prediction**        | DETR (2020), ISTR (2021), FastInst (2023), SOLQ (2021)                                        | learnable query, bipartite matching                   |
| **Multi-Head Prototype Fusion**       | YOLACT (2019), BlendMask (2020), CenterMask (2020)                                            | prototype mask + coefficient                          |
| **Multi-Channel Information Fusion**  | Gland Segmentation (2016), IRNet (2019)                                                       | region/boundary/location 채널 병합                    |
| **Feature Space Clustering**          | ASIS (2019), Instance Segmentation by Embedding (2017), Semantic Instance Segmentation (2017) | embedding + clustering                                |
| **Higher-Order CRF**                  | Instance CRF (2016), Pixelwise CRF (2017), Instance Segmentation with CRF (2017)              | detection potential, dynamic instance labeling        |
| **Self-Supervised Pretraining**       | Instance-aware SSL (2020), Instance Relation Network (2019)                                   | scale/quantity proxy task, instance relation learning |
| **Few-Shot/Fusion Framework**         | FGN (2020), Camouflaged CFL (2021), Incremental FSIS (2021)                                   | support-conditioned, model selection                  |
| **Explicit Contour Parameterization** | ESE-Seg (2019), PolarMask (2019), PolarMask++ (2021), TensorMask (2019)                       | contour/parameter regression                          |
| **Multi-Stage Decomposition**         | DoNet (2023), Path Aggregation Network (2018), Look Closer (2021)                             | decompose-refine-combine                              |
| **Graph/Clustering-Based**            | ASIS (2019), Recurrent (2015), Learning to Cluster (2018), Deep Watershed (2016)              | graph clustering, spectral globalization              |
| **Point Cloud 3D**                    | ASIS (2019), Gaussian Instance (2020), FreePoint (2023), Learning Gaussian (2020)             | point cloud instance labeling                         |
| **Detection-Free Pose-Based**         | Pose2Seg (2018), PersonLab (2018)                                                             | bottom-up keypoint, skeleton features                 |
| **Refinement Framework**              | Look Closer (2021), Mask Transfiner (2021), Mask Encoding (2020)                              | post-processing mask refinement                       |

## 2.1 계열별 구조적 비교

| 계열               | 구조                              | 학습 데이터    | 주요 강점                  |
| ------------------ | --------------------------------- | -------------- | -------------------------- |
| Box-Supervised     | level-set evolution + CNN         | box-only       | pixel-level mask 품질      |
| Detection-Free     | grid/location mapping             | pixel-level    | simple, fast               |
| Query-Based        | query + transformer decoder       | box-only/pixel | end-to-end, set prediction |
| Multi-Head         | prototype + coefficient           | pixel-level    | fast, real-time            |
| Multi-Channel      | parallel branch fusion            | pixel-level    | robust boundary            |
| Feature Clustering | embedding + clustering            | pixel-level    | instance separation        |
| Higher-Order CRF   | CRF + detection potential         | pixel+box      | global consistency         |
| Self-Supervised    | proxy task (scale/quantity)       | unlabeled      | data efficient             |
| Few-Shot Fusion    | support-guided selection          | few-shot       | generalizable              |
| Explicit Contour   | parameter regression              | pixel-level    | shape accuracy             |
| Multi-Stage        | decompose-refine                  | pixel-level    | quality improvement        |
| Graph-Based        | graph clustering                  | pixel-level    | topological consistency    |
| Point Cloud 3D     | point feature + center prediction | point-level    | 3D scene understanding     |
| Pose-Based         | keypoint + skeleton               | pixel-level    | occlusion robustness       |
| Refinement         | post-processing                   | pixel-level    | mask quality boost         |

## 3. 핵심 설계 패턴 분석

## 3.1 학습 손실 패턴

| 손실 유형              | 사용 논문                         | 목적                                    |
| ---------------------- | --------------------------------- | --------------------------------------- |
| **Hungarian Matching** | DETR, ISTR, FastInst              | set prediction loss                     |
| **Dice Loss**          | SOLO, SOLOv2, Box2Mask            | foreground-background imbalance         |
| **Multi-task Loss**    | Gland (2016), Mask R-CNN variants | detection + segmentation + localization |
| **Triplet Ranking**    | Instance-aware SSL (2020)         | scale/quantity embedding                |
| **Polar IoU**          | PolarMask, PolarMask++            | contour alignment                       |
| **DropLoss**           | DropLoss (2021)                   | background suppression                  |
| **Channel-Wise Loss**  | DropLoss (2021)                   | 클래스 불균형 해결                      |

## 3.2 특징 표현 패턴

| 표현 방식                | 논문                                                   | 특징                     |
| ------------------------ | ------------------------------------------------------ | ------------------------ |
| **Class-Agnostic Mask**  | SOLO, SOLOv2                                           | instance별 mask 채널     |
| **Dynamic Filter**       | Conditional Convolutions (2020), Instance-aware (2022) | 위치별 필터 생성         |
| **Instance Activation**  | Sparse Instance Activation (2022), FastInst            | sparse query activation  |
| **Prototype Mask**       | YOLACT, BlendMask                                      | shared mask basis        |
| **Embedding Space**      | ASIS, Instance Segmentation by Embedding (2017)        | cosine similarity        |
| **Feature Propagation**  | Instance-Specific Feature (2022), IRNet                | instance 간 feature 전파 |
| **Polar Representation** | PolarMask, PolarMask++                                 | center + ray length      |
| **4D Tensor**            | TensorMask                                             | (H,W,V,U) 공간           |

## 3.3 추론 전략 패턴

| 전략                       | 논문                                            | 방식                      |
| -------------------------- | ----------------------------------------------- | ------------------------- |
| **NMS-Free**               | ISTR, FastInst, SOLOv2                          | query filtering           |
| **Matrix NMS**             | SOLO, SOLOv2                                    | parallel matrix operation |
| **Instance Matching**      | Pixelwise CRF (2017)                            | bipartite matching        |
| **Clustering**             | ASIS, Instance Segmentation by Embedding (2017) | HDBSCAN, mean-shift       |
| **Thresholding**           | SOLO, SOLOv2                                    | 0.5 threshold + NMS       |
| **Test-Time Augmentation** | StarDist (2022)                                 | 8-fold augmentation       |

## 4. 방법론 비교 분석

## 4.1 Box-Supervised vs Detection-Free

| 차원            | Box-Supervised      | Detection-Free          |
| --------------- | ------------------- | ----------------------- |
| **접근**        | 경계 진화 기반      | 위치 매핑 기반          |
| **학습 데이터** | bounding box만 필요 | pixel-level supervision |
| **구조 복잡도** | level-set + CNN     | grid-based conv         |
| **추론 속도**   | 중저                | 매우 빠름               |
| **주요 논리**   | energy minimization | direct location mapping |

## 4.2 Query-Based vs Prototype-Based

| 차원          | Query-Based         | Prototype-Based    |
| ------------- | ------------------- | ------------------ |
| **추론 시간** | 중                  | 매우 빠름          |
| **mask 품질** | 높음                | 좋음               |
| **복잡성**    | Transformer decoder | linear combination |
| **확장성**    | query 수 증가       | prototype 수 증가  |
| **주요 논문** | DETR, ISTR          | YOLACT, BlendMask  |

## 4.3 Clustering vs Direct Mapping

| 차원            | Clustering             | Direct Mapping       |
| --------------- | ---------------------- | -------------------- |
| **인스턴스 수** | 가변적                 | 고정적 (grid size)   |
| **분할 일관성** | clustering 결과에 의존 | 결정론적             |
| **학습 데이터** | embedding + cluster    | direct pixel mapping |
| **예: SOLO**    | NO                     | YES                  |
| **예: ASIS**    | YES                    | NO                   |

## 5. 방법론 흐름 및 진화

## 5.1 방법론의 진화

**초기 접근 (2015-2016)**:

- Recurrent, Deep Watershed, Higher-Order CRF
- 순차적 처리, CRF-based refinement

**중기 접근 (2017-2019)**:

- Mask R-CNN 기반 개선, Multi-channel fusion, Detection-free
- RoI-free, Feature propagation

**Transformer Era (2020-2023)**:

- DETR-based query, Mask2Former, ISTR, FastInst
- Query-based set prediction, end-to-end learning

**최신 접근 (2021-2024)**:

- Detection-Free (SOLO, SOLOv2)
- Refinement frameworks (Look Closer, Mask Transfiner)
- Weakly-supervised (Box-supervised Level Set)

## 5.2 진화 추세

| 시기      | 주요 특징                                      |
| --------- | ---------------------------------------------- |
| 2015-2016 | CRF, Recurrent, Watershed                      |
| 2017-2019 | RoI-free, Multi-channel, Feature propagation   |
| 2020-2021 | Query-based, Transformer, Detection-free       |
| 2022-2024 | Refinement, Weakly-supervised, Self-supervised |

## 6. 종합 정리

Instance Segmentation 방법론은 **세 가지 축**으로 구분된다:

1. **구조적 축**: Detection-Free vs Query-Based vs Box-Supervised
2. **학습 축**: Weakly-supervised vs Fully-supervised vs Self-supervised
3. **출력 축**: Direct mapping vs Clustering vs Prototype fusion

전체 방법론 지형은 다음과 같이 요약된다:

```text
Instance Segmentation 지형
├─ 구조: Detection-Free → Query → Box-Supervised
├─ 학습: Weak → Full → Self-Supervised
└─ 출력: Clustering → Direct → Prototype
```

각 계열은 서로 다른 trade-off를 가진다:

- **Detection-Free**는 빠르지만 grid-size에 제한
- **Query-Based**는 end-to-end지만 계산 복잡도 높음
- **Box-Supervised**는 품질은 높지만 학습 데이터 요구

이 지형은 **다양한 application**에 맞게 선택되도록 설계된다.

## 3 장. 실험결과 분석

### 1. 평가 구조 및 공통 실험 설정

#### 1.1 주요 데이터셋 유형

Instance Segmentation 논문에서 가장 빈번하게 등장하는 데이터셋과 해당 데이터셋이 검증하려는 성능 특성은 다음과 같다.

| 데이터셋        | 영역              | 용도/특성                                                       | 주요 타겟                             |
| --------------- | ----------------- | --------------------------------------------------------------- | ------------------------------------- |
| MS COCO         | 일반 영상         | standard benchmark, instance segmentation evaluation            | 80 classes, balanced object types     |
| Pascal VOC 2012 | 일반 영상         | instance segmentation classic benchmark                         | 20 classes, VOC 2012 validation split |
| Cityscapes      | 실외/자율주행     | fine-grained instance segmentation, high-res images (1024×2048) | 8 classes, urban driving scenes       |
| LVIS            | long-tail         | large vocabulary, 1000+ classes, rare classes evaluation        | 1640 classes, Zipfian distribution    |
| ADE20K          | semantic/panoptic | open-vocabulary, large-scale semantic + instance                | 200 classes, diverse scenes           |
| iSAID           | 항공 영상         | aerial imagery, high-res (2806 images, 655k instances)          | 15 classes, remote sensing            |
| MoNuSeg/CPS     | 병리 영상         | nuclei segmentation, touching cells                             | histopathology, nuclear morphology    |
| ISBI2014/DoNet  | cytology          | translucent cells, overlap handling                             | cervix cells, cytoplasm/nucleus       |

#### 1.2 평가 환경 (실험/시뮬레이션/실환경)

| 환경 유형               | 사용 사례                            | 특징                                           |
| ----------------------- | ------------------------------------ | ---------------------------------------------- |
| **실내/실외 일반 영상** | COCO, Pascal VOC, Cityscapes, ADE20K | 자연 조명, 다양한 조명 조건, occlusion         |
| **의료 영상**           | MoNuSeg, LiTS, CPM, LIDC-IDRI        | annotation scarce, instance boundary ambiguity |
| **항공/위성 영상**      | iSAID, D2S                           | high-res, sparse objects, multi-scale          |
| **3D point cloud**      | ScanNet v2, S3DIS                    | geometric features, clustering-based           |
| **저조도/위험 환경**    | LIS dataset, CAMO++                  | extreme conditions, feature disturbance        |
| **실시간/엣지**         | YOLACT/SparseInst                    | speed-accuracy trade-off, V100/1080Ti          |

#### 1.3 비교 대상 (baseline) 유형

| Baseline 유형           | 대표 방법                          | 비교 목적                              |
| ----------------------- | ---------------------------------- | -------------------------------------- |
| **two-stage detection** | Mask R-CNN, Cascade Mask R-CNN     | instance separation, standard baseline |
| **one-stage**           | YOLACT, PolarMask, SOLO, BlendMask | speed, single-pass                     |
| **query-based**         | SOLOv2, SOLQ, FastInst, ISTR       | end-to-end, query learning             |
| **weakly-supervised**   | BoxInst, Box2Mask, DropLoss        | box-only, pseudo-mask                  |
| **proposal-based**      | MCG, DWT, InstanceCut              | region proposal, clustering            |
| **specialized**         | IRNet, DoNet, S3Net                | domain-specific challenges             |

#### 1.4 주요 평가 지표 정리

| 지표            | 정의/용도                                    | 빈도                     |
| --------------- | -------------------------------------------- | ------------------------ |
| **mask AP**     | instance segmentation 정밀도 (mask IoU 기반) | 매우 빈번                |
| **AP50/AP75**   | IoU 임계값별 정밀도                          | 빈번                     |
| $mAP^r$         | IoU@[0.5:0.95] 평균                          | 빈번 (VOC 기준)          |
| **mIoU**        | mean intersection over union                 | 빈번 (semantic+instance) |
| **PQ**          | panoptic quality (thing+stuff)               | 빈번 (panoptic task)     |
| **APL/APM/APS** | large/medium/small object별 AP               | 빈번                     |
| **AJI**         | aggregated Jaccard Index (nuclei)            | 병리 영상                |
| **F1 Score**    | precision/recall 평균                        | 빈번                     |
| **cIoU/gIoU**   | instance-level IoU                           | referring segmentation   |
| **FPS**         | inference speed                              | 실시간 작업              |

### 2. 주요 실험 결과 정렬

#### 2.1 COCO 벤치마크에서의 주요 방법 비교

| 논문명                          | 데이터셋/환경 | 비교 대상                           | 평가 지표 | 핵심 결과                                                           |
| ------------------------------- | ------------- | ----------------------------------- | --------- | ------------------------------------------------------------------- |
| SOLOv2 (2020)                   | COCO val2017  | SOLO, Mask R-CNN, YOLACT            | AP        | ResNet-50 **38.8 AP**, ResNet-101**39.7 AP**, DCN**41.7 AP**        |
| ISTR (2021)                     | COCO          | Mask R-CNN, SOLOv2, MEInst          | AP        | ResNet50-FPN **46.8 box AP / 38.6 mask AP**,**13.8 fps**            |
| Mask2Former (2021)              | COCO          | 특화 아키텍처들                     | AP        | **50.1 AP** (single architecture)                                   |
| CenterMask (2020)               | COCO          | YOLACT, PolarMask, TensorMask       | AP        | **34.5 AP @ 12.3 FPS** (hourglass),**32.5 mAP @ 25.2 FPS** (DLA)    |
| FastInst (2023)                 | COCO test-dev | SparseInst, Mask2Former             | AP        | **35.6 AP @ 53.8 FPS** (fastest),**40.5 AP @ 32.5 FPS** (trade-off) |
| YOLACT (2019)                   | COCO          | Mask R-CNN, FCIS                    | AP/FPS    | **29.8 AP @ 33.5 FPS**, Fast NMS**+24.9 FPS**                       |
| TensorMask (2019)               | COCO          | Mask R-CNN, ESE-Seg                 | AP        | **34.0 AP** (Mask R-CNN 대비 +5.1), AP_L**48.4**                    |
| CondInst (2020)                 | COCO          | YOLACT-700, Mask R-CNN              | AP        | **Mask R-CNN 보다 정확도+속도 모두 우수**                           |
| Box2Mask (2022)                 | COCO          | BoxInst, BoxSupervisedSOLO          | AP        | **38.3 AP**, Swin-L**42.4 AP** (fully supervised 동급)              |
| DropLoss (2021)                 | LVIS          | EQL, BEQL                           | mAP       | **SOTA mAP**, rare/common/frequent 균형 개선                        |
| Learning Equivariant Seg (2022) | COCO/LVIS     | CondInst, SOLOv2, SOTR, Mask2Former | AP        | CondInst **+2.8~3.1 AP**, SOLOv2**+2.9~3.2 AP**                     |
| FreePoint (2023)                | COCO          | Mask3D, UnScene3D, PointContrast    | AP        | **fully-supervised 50% 이상**, SOTA 대비**+18.2%**                  |

#### 2.2 weakly/weakly-supervised setting 비교

| 논문명                    | 데이터셋        | 비교 대상                           | 평가 지표 | 핵심 결과                                                                      |
| ------------------------- | --------------- | ----------------------------------- | --------- | ------------------------------------------------------------------------------ |
| BoxInst (2020)            | COCO            | weak baseline 21.1%, CondInst, BBTP | AP        | **31.6%** (→33.2% R-101), fully-supervised 대비**39.1%** (gap↓)                |
| Box2Mask (2022)           | Pascal VOC      | BoxInst, DiscoBox, fully-supervised | AP        | **38.3% → 43.2%**, COCO**33.4% → 38.3%**                                       |
| Pointly-Supervised (2021) | COCO            | Mask R-CNN, BoxInst                 | AP        | **36.1 AP** (fully-supervised**37.2** 대비 97%),**16 초 vs 79.2 초** (5x 절감) |
| Simple Does It (2016)     | Pascal VOC      | fully-supervised DeepLab, GrabCut+  | mIoU      | weak 대비 **67.5** (fully**70.5**),**~95.7%**                                  |
| Box-supervised (2022)     | Pascal VOC/COCO | BoxInst, pseudo-mask methods        | AP        | Pascal VOC **ResNet-50 기준 +2.0%**, COCO**SOTA 달성**                         |
| Weakly Supervised (2020)  | Pascal VOC      | Ahn et al., Laradji et al.          | mAP^r     | image-level **50.9% @0.5**, bbox**32.1% @0.75**                                |

#### 2.3 biomedical/medical domain 결과

| 논문명                    | 데이터셋      | 비교 대상                          | 평가 지표   | 핵심 결과                                                   |
| ------------------------- | ------------- | ---------------------------------- | ----------- | ----------------------------------------------------------- |
| DoNet (2023)              | ISBI2014, CPS | Mask R-CNN, Occlusion R-CNN, IRNet | mAP/overlap | DRM 추가 **ISBI2014 +7.34%**,**DRM+CRM +1.74%**             |
| IRNet (2019)              | CPS           | Mask R-CNN, JOMLS, CSPNet          | AJI/F1      | **cell AJI 4.97%**,**nuclei AJI 6.33%**                     |
| Instance-aware SSL (2020) | MoNuSeg       | baseline ResUNet-101               | AJI         | **70.63%** (baseline**65.29** 대비 +5.34%)                  |
| Contour Proposal (2021)   | 4 datasets    | U-Net, Mask R-CNN                  | F1^avg      | CPNR4-U22 **0.55** vs U-Net**0.47**                         |
| Learning Gaussian (2020)  | ScanNet/S3DIS | SGPN, 3D-BoNet                     | AP@50       | **state-of-the-art**, 3D-BoNet 대비**적은 인스턴스 효율**   |
| Nuclei StarDist (2022)    | CoNIC 2022    | HoverNet, internal ablation        | mPQ/PQ      | **1 위 달성** (preliminary + final), internal**mPQ 0.5885** |

#### 2.4 3D/point cloud 및 surgical domain

| 논문명                                  | 데이터셋            | 비교 대상                      | 평가 지표 | 핵심 결과                                                              |
| --------------------------------------- | ------------------- | ------------------------------ | --------- | ---------------------------------------------------------------------- |
| Associatively Segmenting (2019)         | S3DIS               | SGPN, PointNet baseline        | mWCov     | S3DIS **46.3 → 51.8** (SGPN 대비**5.5 포인트**), shapeNet**+0.7→0.85** |
| Instance Segmentation PointCloud (2020) | S3DIS               | SGPN, fixed radius aggregation | AP@50     | **state-of-the-art**, 3D-BoNet 대비**efficient**                       |
| FreePoint (2023)                        | S3DIS (fine-tuning) | Mask3D, PointContrast          | AP        | fully-supervised **Mask3D 50% 이상**,**+18.2%** 기존 SOTA 대비         |

#### 2.5 surgical domain

| 논문명                       | 데이터셋         | 비교 대상                   | 평가 지표          | 핵심 결과                                                            |
| ---------------------------- | ---------------- | --------------------------- | ------------------ | -------------------------------------------------------------------- |
| From Forks to Forceps (2022) | EndoVis2017/2018 | Mask R-CNN, ISINet, TraSeTR | AP50/Challenge IoU | EV17 **기존 SOTA 대비 +12 포인트/+20%**, ISINet 대비**30%/60% 개선** |

#### 2.6 long-tail / large vocabulary

| 논문명                      | 데이터셋    | 비교 대상              | 평가 지표                   | 핵심 결과                                                |
| --------------------------- | ----------- | ---------------------- | --------------------------- | -------------------------------------------------------- |
| LVIS dataset (2019)         | LVIS (v0.5) | COCO, ADE20K quality   | AP, overlap, consistency    | **1000+ classes**,**1.64M images**, Zipfian distribution |
| DropLoss (2021)             | LVIS        | EQL, BEQL              | mAP/rare/common/frequent AP | **SOTA mAP**, long-tail 균형 개선                        |
| Learning Equivariant (2022) | LVISv1      | SOLOv2, CondInst, etc. | AR50/AR75, AP               | SOLOv2 **+2.7 AP**, COCO**CondInst +2.8~3.1**            |

#### 2.7 open-vocabulary / zero-shot

| 논문명                     | 데이터셋                          | 비교 대상                                  | 평가 지표          | 핵심 결과                                                              |
| -------------------------- | --------------------------------- | ------------------------------------------ | ------------------ | ---------------------------------------------------------------------- |
| Open-Vocabulary Seg (2023) | ADE20K, COCO panoptic, Cityscapes | segmentation-only, detection-only          | AP, concept+box AP | **mask AP 8.6→46.4** (ADE20K→COCO), concept+box 조건**COCO 성능 근접** |
| Zero-Shot Instance (2021)  | COCO (65/15 seen/unseen)          | one-hot semantic prior, existing zero-shot | Recall@100, mAP    | **Recall@100 61.9/58.9/54.4**, mAP**13.6** (baseline 대비**+5.61%**)   |
| EVF-SAM (2024)             | RefCOCO(g)                        | CLIP, BEIT-3, ViLT, LLaVA                  | cIoU               | BEIT-3(Text+Image) **83.7 cIoU**, CLIP-only**63.4** 대비**+20.3**      |

#### 2.8 real-time / speed-accuracy

| 논문명                         | 데이터셋         | 비교 대상               | 평가 지표   | 핵심 결과                                                       |
| ------------------------------ | ---------------- | ----------------------- | ----------- | --------------------------------------------------------------- |
| Explicit Shape Encoding (2019) | Pascal VOC, COCO | Mask R-CNN, YOLOv3-tiny | mAP/FPS     | VOC **69.3 mAP**, COCO**48.7 mAP**, Mask R-CNN 대비**~7x 빠름** |
| Explicit Shape Encoding (2023) | COCO             | Mask R-CNN, RetinaMask  | AP/FPS      | **48.7 mAP**,**~7x 빠름**, YOLOv3-tiny 대비**130 fps**          |
| SparseInst (2022)              | COCO             | YOLACT++, SOLOv2        | AP/FPS/TIDE | **37.9 AP @ 40 FPS (608px)**,**58.5 FPS (448px)**               |
| Pose2Seg (2018)                | OCHuman          | Mask R-CNN, PersonLab   | AP          | **0.238** (Mask R-CNN**0.169** 대비**+0.069**)                  |
| Look Closer (2021)             | Cityscapes       | Mask R-CNN, PointRend   | AP/latency  | **+4.3 AP** (36.4→40.0),**211ms** latency                       |

### 3. 성능 패턴 및 경향 분석

#### 3.1 데이터셋 의존성 및 한계

| 데이터셋       | 성능 개선 패턴                                                                        | 주의점                                                                        |
| -------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **COCO**       | one-stage 방법 (SOLO, CenterMask, YOLACT) 이 강세, query-based (STR, SOLQ) 도 높은 AP | real-time vs accuracy trade-off 명확                                          |
| **Pascal VOC** | weakly-supervised (BoxInst, Box2Mask) 이 fully-supervised 에 근접, +2~3% AP           | weak supervision ceiling 높음                                                 |
| **Cityscapes** | fine-grained boundary 에서 coarse-to-fine refinement 이 효과적, +4.3 AP 달성          | 1024×2048 해상도에서 post-processing 이 중요                                  |
| **LVIS**       | long-tail 에서 Mask R-CNN 대비 specialized 방법 필요, DropLoss 등 loss weighting 중요 | rare classes 에서 Mask R-CNN 성능 현저히 저하                                 |
| **biomedical** | domain-specific preprocessing (DRM, contour modeling) 이 일반화 성능에 결정적         | CNN 보다 Transformer 가 universal segmentation 에서 우세 (cellpose challenge) |

#### 3.2 one-stage vs two-stage 경향

| 접근법                        | 평균 AP   | FPS   | 특징/장단점                                     |
| ----------------------------- | --------- | ----- | ----------------------------------------------- |
| **two-stage (Mask R-CNN)**    | 33.2~37.2 | 7~14  | accurate boundary, slower, detection bottleneck |
| **one-stage (YOLACT/SOLO)**   | 29.8~37.1 | 33~36 | faster, shared prototype, detection byproduct   |
| **query-based (SOLOv2/ISTR)** | 38.6~40.9 | 11~14 | end-to-end, NMS 제거, embedding 예측            |

**패턴**: one-stage 방법은 Mask R-CNN 대비 2~4% AP 뒤이지만 2~3x 속도 향상. query-based 방법은 두 기준을 모두 넘거나 근접.

#### 3.3 backbone 영향

| backbone       | AP 영향      | FPS 영향  | 주석                                         |
| -------------- | ------------ | --------- | -------------------------------------------- |
| **ResNet-50**  | 33.2~35.6 AP | 12~36 FPS | baseline, balance                            |
| **ResNet-101** | +1.5~2.0 AP  | -1~2 FPS  | accuracy ↑, 속도 ↓                           |
| **Swin-L**     | +4~6 AP      | 2~10 FPS  | large models 에서 특히 효과                  |
| **DCN**        | +0.8~2.0 AP  | 유지      | deformable convolution 이 boundary quality ↑ |

#### 3.4 weak supervision 효과

| 설정                  | AP 달성률 (vs fully) | 비용 절감 | 패턴                                         |
| --------------------- | -------------------- | --------- | -------------------------------------------- |
| **image-level**       | 40~65%               | baseline  | recall↑, precision↓ trade-off                |
| **bounding-box**      | 70~90%               | 5x 절감   | +2~3% AP, pseudo-label quality 핵심          |
| **point-supervision** | 94~98%               | 5x 절감   | 36.1/37.2 (97%), 16s/79s (98%)               |
| **pseudo-mask**       | -1~2% (overfit)      | -         | overfitting 위험, sparse supervision 더 좋음 |

**패턴**: weak supervision ceiling 가 예상보다 높음. point-level > box-level > image-level 순.

#### 3.5 성능 상향/하향 경향

**상향**:

- Transformer 기반 모델 (Mask2Former) 이 한 단계씩 성능 ↑
- one-stage → query-based → end-to-end 전환
- refinement/post-processing 추가 (+4~6 AP)
- weak-to-strong transfer (+2~3% AP)

**하향**:

- open-vocabulary/zero-shot 에서 seen→unseen 전이 시 성능 ↓
- large vocabulary (LVIS) 에서 COCO 성능보다 AP ↓
- real-time 설정에서 accuracy trade-off 필수

#### 3.6 방법론적 경향

| 접근                                           | 성능 경향                     | 한계                                    |
| ---------------------------------------------- | ----------------------------- | --------------------------------------- |
| **clustering-based** (SGPN, MCG)               | SGPN 대비 ASIS**+5.5** 포인트 | seed selection 오류 연쇄                |
| **proposal-free** (DWT, Learning to Cluster)   | real-time 성능 달성           | boundary 품질 제한, small object 어려움 |
| **query-based** (SOLOv2, SOLQ)                 | end-to-end, NMS 제거          | query initialization, occlusion 취약    |
| **boundary-refinement** (Look Closer, Pointly) | +4.3 AP                       | model-agnostic, post-processing         |

### 4. 추가 실험 및 검증 패턴

#### 4.1 ablation study 공통 패턴

| 논문                 | ablation 항목                          | 효과                                 | 교훈                                    |
| -------------------- | -------------------------------------- | ------------------------------------ | --------------------------------------- |
| SOLOv2               | scale jitter 640-800, 3x schedule, DCN | **+1.0 AP** vs SOLO                  | schedule, augmentation, DCN 중요        |
| Box2Mask             | box projection, tree filter, 긴 학습   | **+2.0% vs BoxInst**                 | tree filter**+1.9%**, 긴 스케줄 필요    |
| CenterMask           | Local Shape + Global Saliency          | Shape **+10 AP**, Saliency**+5 AP**  | factorization 효과 명확                 |
| DropLoss             | stochastic dropping vs deterministic   | **Pareto frontier**                  | stochastic 이 rare/common/frequent 균형 |
| Learning Equivariant | dataset-level uniqueness loss          | **+2.8~3.2 AP**                      | architecture 변화 없이 학습 강화        |
| IRNet                | DRM, IRM modules                       | **cell AJI +4.97%, nuclei +6.33%**   | decompose-recombine 전략 유효           |
| Mask Transfiner      | incoherent region (quadtree)           | **+3.0 AP**, +1.8 AP (full RoI 대비) | sparse refinement 효율성 입증           |

#### 4.2 조건 변화 실험

| 논문               | 조건 변화                       | 결과                      | 교훈                         |
| ------------------ | ------------------------------- | ------------------------- | ---------------------------- |
| CenterMask         | class-agnostic → class-specific | **+2.4 AP**               | class-specific encoding 효과 |
| FastInst           | decoder layer 증가 (0~6)        | **층 6 근처 포화**        | 복잡도 증가 불필요           |
| Mask Encoding      | dimension 100~60                | **100 dim 최적**          | reconstruction error 2.5%    |
| PolarMask++        | ray 수 18→24→36→72              | **+1.1% → +0.3% → -0.1%** | 36 ray 이상 불이익           |
| S4Net              | RoIMasking α=0~1                | **α=1/3 최적 (86.7%)**    | ternary masking 이 이점      |
| Pointly-Supervised | point 수 10 vs 20               | **+0.3 AP vs 2x 시간**    | 10 point 최적                |

#### 4.3 cross-dataset generalization

| 논문                   | 설정                      | 일반화 결과                 | 교훈                      |
| ---------------------- | ------------------------- | --------------------------- | ------------------------- |
| CPN (Contour Proposal) | BBBC039 → BBBC041         | **cross-dataset superior**  | Fourier descriptor robust |
| Cellpose (2023)        | unseen modality           | **catastrophic forgetting** | self-attention 이 일반화  |
| Open-Vocabulary (2023) | COCO panoptic → zero-shot | **mask AP 8.6→46.4**        | concept+box 필요          |
| BoxInst                | COCO → ReCTS              | **+10% AP**                 | color-prior도 확장 가능   |
| DoNet                  | ISBI2014 → CPS            | **DRM alone CPS 혼동**      | dataset-specific 효과     |

### 5. 실험 설계의 한계 및 주의점

#### 5.1 비교 조건의 불일치

| 문제                       | 사례                            | 영향                       |
| -------------------------- | ------------------------------- | -------------------------- |
| **평가 지표 불일치**       | AP vs mAP^r vs cIoU             | apples-to-apples 비교 불가 |
| **backbone 불일치**        | ResNet-50/101/152/DCN/Swin 혼합 | 성능 차이 일부 backbone    |
| **training data 불일치**   | COCO train vs ADE20K vs LVIS    | pretraining 영향           |
| **test-dev 기준 불일치**   | COCO test-dev vs official       | ensemble vs single model   |
| **post-processing 불일치** | with/without CRF, TTA           | 공정 비교 어려움           |

#### 5.2 데이터셋 의존성

| 데이터셋          | 의존성                           | 일반화 한계                              |
| ----------------- | -------------------------------- | ---------------------------------------- |
| **COCO**          | balanced objects, standard tasks | open-vocabulary/long-tail 에서 성능 저하 |
| **LVIS**          | long-tail, rare classes          | COCO에서 trained 방법 성능 낮음          |
| **biomedical**    | domain-specific morphology       | 의료 영상 일반화 어려움                  |
| **iSAID**         | aerial imagery, sparse objects   | natural image에서 전이 어려움            |
| **COCO test-dev** | fixed ground truth               | test-time augmentation 효과              |

#### 5.3 평가 지표의 한계

| 지표           | 한계                                                | 대안/보완               |
| -------------- | --------------------------------------------------- | ----------------------- |
| **mask AP**    | localization 정확도만 측정, boundary quality 불포함 | boundary F-score 추가   |
| **mIoU**       | instance 분리 정보 불명확                           | AP, PQ, AJI 병행        |
| **Recall@100** | precision 반영 안 함                                | AP와 병행               |
| **PQ**         | thing vs stuff 구분 없음                            | Thing/Stuff separately  |
| **AJI**        | nucleus segmentation 특화                           | 세포 분할 일반화 어려움 |
| **FPS**        | hardware 의존적 (GPU 메모리 등)                     | GFLOPs 추가             |

#### 5.4 일반화 한계

| 제한                   | 사례                                  | 영향                     |
| ---------------------- | ------------------------------------- | ------------------------ |
| **domain gap**         | synthetic → real (AModal, LIS)        | real-world에서 성능 저하 |
| **annotation quality** | partial masks, occluded instances     | GT 불명확 → 평가 어려움  |
| **occlusion**          | severe occlusion (Pose2Seg, OCHuman)  | detection 기반 방법 취약 |
| **scale variation**    | COCO multi-scale → small object 성능↓ | small object AP 낮음     |
| **unseen objects**     | zero-shot/unseen class                | seen→unseen 전이 제한적  |

### 6. 결과 해석의 경향

#### 6.1 저자 해석 공통 패턴

| 해석 유형                   | 사례                                | 경향성                        |
| --------------------------- | ----------------------------------- | ----------------------------- |
| **simple is strong**        | SOLO/BoxInst/Pointly                | 단순한 방법이 효과적          |
| **weak-to-strong transfer** | BoxInst/Box2Mask/Pointly            | weak supervision ceiling 높음 |
| **refinement works**        | Look Closer/Transfiner/Post-process | post-processing 이 큰 이득    |
| **boundary matters**        | Look Closer/BAIS/SOLOv2             | boundary modeling 핵심        |
| **domain-specific**         | DoNet/IRNet/CPN                     | domain adaptation 필수        |
| **unified architecture**    | Mask2Former/OneFormer               | single arch, multi-task       |

#### 6.2 해석 vs 실제 결과

| 해석                                    | 실제 결과                                | 차이                       |
| --------------------------------------- | ---------------------------------------- | -------------------------- |
| **"simple does it"** (BoxInst/Pointly)  | weak supervision**94~98%**               | 예상보다 좋은 weak ceiling |
| **"one-stage fast"** (YOLACT)           | Mask R-CNN 대비**+24.9 FPS**             | 속도-정확도 trade-off 확인 |
| **"Transformer SOTA"** (Mask2Former)    | COCO instance**50.1 AP**                 | one-stage/transformer 병합 |
| **"boundary refinement 큰 이득"**       | **+4.3 AP** (Look Closer)                | post-processing 효과 입증  |
| **"query learning strong"** (SOLQ/ISTR) | SOLOv2**+1.2 AP**, SOLOv2 box AP**+6.1** | detection also improves    |

#### 6.3 과장 해석 주의사항

| 과장 해석                   | 실제 근거                       | 주의점                              |
| --------------------------- | ------------------------------- | ----------------------------------- |
| **"SOTA 달성"**             | COCO/특정 벤치마크에서          | 다른 benchmark 에서 아님            |
| **"weak supervision 강력"** | point-level**97%**, bbox**90%** | image-level**40~65%**               |
| **"Transformer 우세"**      | cellpose challenge, Mask2Former | COCO에선 Mask R-CNN/CondInst도 강세 |
| **"end-to-end"**            | ISTR, SOLQ, SOLQ                | 여전히 RoIAlign/query box 사용      |
| **"generalization"**        | cross-dataset                   | unseen domain/condition에서 제한적  |

### 7. 종합 정리

Instance Segmentation 분야의 실험 결과들을 종합하면, 성능은 크게 **data (weak supervision), architecture (one-stage→query-based→end-to-end), boundary modeling** 세 축에 의해 결정된다. Mask R-CNN baseline 대비 one-stage 방법들은 +2~4% AP 뒤지만 2~3x 속도 향상을, query-based 방법들은 두 기준을 넘어 COCO**38~46 AP**를 달성하며, weak supervision (point-level)은**94~98%** fully-supervised 성능 도달을 보여 weak supervision ceiling 가 높음을 입증한다. COCO/balanced object 환경에서는 Transformer 기반 방법들이 강세이나, long-tail (LVIS) 과 open-vocabulary 설정에서는 specialized loss weighting (DropLoss) 과 semantic transfer (ZSI) 이 필수적이며, biomedical/의료 영상에서는 domain-specific preprocessing (contour modeling, instance-aware SSL) 이 일반화 성능을 결정한다. real-time 요구사항이 있다면 SOLOv2/FastInst/YOLACT 계열이, accuracy 우선이라면 Mask2Former/ISTR/Mask R-CNN 계열이 유리하다. refinement/post-processing (Look Closer/Transfiner) 은**4~6 AP** 향상을 제공하므로 모델-agnostic으로 적용 가능하며, weak supervision 설정은 point-level 에서 가장 효과적(97% 성능)이고 image-level은 recall 향상은 있으나 precision 저하 trade-off 를 수반한다.
