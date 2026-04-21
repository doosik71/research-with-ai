# Domain Adaptation

## 서론

### 1. 연구 배경

Domain Adaptation 연구는 Source 도메인에서 학습한 지식을 Target 도메인으로 효과적으로 전이하는 문제를 다루며, 문제 설정에 따라 Unsupervised (UDA), Semi-supervised (SSDA), Partial (PDA), Open-set 등으로 분류된다. 본 보고서는 이미지 분류, 시맨틱 세그멘테이션, 객체 검출, NLP 등 다양한 다운스트림 작업에서 적용 가능한 방법론을 체계적으로 분류하고 분석한다. 방법론적 관점에서는 adversarial alignment, feature matching, reconstruction-based, contrastive 학습 기법을 중심으로, 도메인 관계성과 차원 구조 (single-source/target, multi-source, gradual, cross-modality) 를 고려한 분류 체계를 수립한다.

### 2. 문제의식 및 분석 필요성

현존하는 Domain Adaptation 연구들은 문제 설정, 적응 위치, 데이터 활용 방식, Source 접근 여부 등 다양한 기준으로 개발되고 있으나, 각 방법론의 설계 패턴과 진화 경향, 실험적 한계가 명확히 정립되지 않은 상태다. 특히 50 여 편의 논문에서 제시된 방법론 간 구조적 차이 (가중치 공유 vs 분리), 복잡도 비교, 성능 격차 요인 등이 체계적으로 정리되지 않아 실제 적용 시 방법론 선택에 혼란이 발생한다. 또한 데이터셋 의존성, 평가 지표의 불일치, 이론적 근거와 실증 결과 간의 괴리 등의 문제가 존재하며, 이를 명확히 규명할 필요가 있다.

### 3. 보고서의 분석 관점

본 보고서는 세 가지 축으로 문헌을 정리한다. 첫째, 연구 분류 체계는 문제 설정 유형, 적용 대상 작업, 도메인 차원 구조, 적용 관점을 기준으로 9 개의 주요 계열 (Adversarial, Feature Matching, Multi-Source/Gradual, Partial/Open-Set, Source-Free, Contrastive, Cross-Modality, Semantic Segmentation, Object Detection) 로 구분하고 각 계열의 핵심 논문과 설계 패턴을 기술한다. 둘째, 방법론 분석은 공통 문제 설정, 핵심 설계 패턴 (Feature/Output/Distribution Alignment, Pseudo-labeling, Cycle-consistency), 계열별 특징 비교, 진화 경향을 통해 방법론적 구조와 한계를 규명한다. 셋째, 실험결과 분석은 주요 벤치마크 (Office-31, VisDA, GTA→Cityscapes, Digit adaptation 등) 에서의 성능 비교, 성능 격차 요인, ablation study, 민감도 분석 결과를 종합하여 실제 적용 시 고려해야 할 데이터셋, task-specific 요인을 제시한다.

### 4. 보고서 구성

1 장은 연구 분류 체계를 다루며, 문제 설정 (UDA/SSDA/PDA/Open-set), 작업 유형 (분류/세그멘테이션/검출), 도메인 관계성 (single-source, multi-source, gradual, cross-modality), 적용 관점 (adversarial, feature matching, contrastive 등) 을 기준으로 9 개의 주요 계열로 논문들을 분류하고 각 계열의 대표 방법론과 설계 패턴을 정리한다.

2 장은 방법론 분석으로, 기본 문제 설정과 적응 파이프라인, 도메인 shift 유형, 15 개 주요 계열의 특징, 핵심 설계 패턴, 계열 간 비교, 진화 경향, 이론적 흐름을 체계적으로 분석하여 방법론 선택 시 고려해야 할 설계 원칙과 trade-off 를 제시한다.

3 장은 실험결과 분석으로, 주요 벤치마크에서의 성능 비교, 성능 개선 패턴, 특정 조건에 따른 한계, 데이터셋 의존성, 평가 지표의 불일치, 일반화 문제 등을 실증적 결과와 함께 정리하며, 방법론 해석의 경향과 주의점을 통해 실제 적용 시 유의해야 할 점을 안내한다.

## 1 장. 연구체계 분류

### 1. 연구 분류 체계 수립 기준

본 보고서의 연구 분류는 다음과 같은 기준과 원칙에 따라 수립되었다.

1. **문제 설정 유형**: unsupervised, semi-supervised, source-free, partial/open-set 등 라벨 가용성 및 클래스 공간 제약에 따라 구분
2. **적용 대상 작업**: 이미지 분류, 시맨틱 세그멘테이션, 객체 검출, NLP 등 downstream task 유형
3. **도메인 차원 구조**: single-source/target, multi-source, gradual, cross-modality 등 도메인 간 관계성
4. **적용 관점**: adversarial alignment, feature matching, reconstruction-based, contrastive 등 학습 기준

### 2. 연구 분류 체계

#### 2.1 Adversarial Domain Alignment

source 와 target 도메인의 feature distribution 을 adversarial discriminator 를 통해 맞추는 접근법으로, GAN 의 minimax 구조와 gradient reversal 을 핵심 기법으로 한다.

| 분류                          | 논문명                                                                       | 분류 근거                                                                                                 |
| ----------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Adversarial Domain Adaptation | Domain-Adversarial Neural Networks (2014)                                    | source 와 target 의 domain-invariant feature 학습을 위한 gradient reversal 기반 adversarial structure     |
| Adversarial Domain Adaptation | Domain-Adversarial Training of Neural Networks (2015)                        | end-to-end adversarial learning 을 통한 representation 최적화                                             |
| Adversarial Domain Adaptation | Domain-Adversarial Neural Networks (2014)                                    | binary classification 에서 domain confusion loss 를 도입한 adversarial adaptation                         |
| Adversarial Domain Adaptation | Adversarial Discriminative Domain Adaptation (2017)                          | discriminative model 기반 unshared weights 전략으로 source encoder 고정                                   |
| Adversarial Domain Adaptation | Unsupervised Domain Adaptation by Backpropagation (2014)                     | GRL 를 통한 adversarial learning 으로 domain classifier 가 feature extractor 를 속이도록 유도             |
| Adversarial Domain Adaptation | Deep Domain Confusion: Maximizing for Domain Invariance (2014)               | MMD 기반 domain confusion loss 를 classification loss 와 joint optimize                                   |
| Adversarial Domain Adaptation | Bidirectional Learning for Domain Adaptation of Semantic Segmentation (2019) | translation 과 segmentation adaptation 을 일방향 파이프라인이 아닌 상호작용하는 반복 학습 시스템          |
| Adversarial Domain Adaptation | Maximum Classifier Discrepancy for Unsupervised Domain Adaptation (2017)     | decision boundary 를 고려한 유효한 정렬을 위한 classifier discrepancy maximization                        |
| Adversarial Domain Adaptation | Conditional Adversarial Domain Adaptation (2017)                             | feature 와 prediction 의 joint variable 을 조건으로 한 discriminator 를 통한 class-aware alignment        |
| Adversarial Domain Adaptation | Deep Transfer Learning with Joint Adaptation Networks (2017)                 | 여러 층 activation 의 joint distribution alignment 를 JMMD 로 수행                                        |
| Adversarial Domain Adaptation | Maximum Density Divergence for Domain Adaptation (2020)                      | MDD 손실을 adversarial adaptation 에 추가하여 inter-domain divergence 와 intra-class density maximization |
| Adversarial Domain Adaptation | Multi-Adversarial Domain Adaptation (2018)                                   | 클래스별 도메인 디스크리미네이터와 소프트 어시그먼트를 통한 멀티모드 정렬                                 |
| Adversarial Domain Adaptation | Partial Adversarial Domain Adaptation (2018)                                 | partial setting 에서 source classifier 의 target prediction 평균을 기반으로 class-level weighting         |
| Adversarial Domain Adaptation | Gradient Reversal Layer 기반 방법들                                          | GRL 을 활용한 모든 adversarial 구조                                                                       |

#### 2.2 Feature Matching & Statistical Alignment

특징 공간에서 source 와 target 의 통계적 양 (mean, covariance) 을 직접 정렬하는 접근법으로, optimal transport 와 kernel discrepancy 를 포함한다.

| 분류             | 논문명                                                                                                | 분류 근거                                                                                                   |
| ---------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Feature Matching | Deep CORAL: Correlation Alignment for Deep Domain Adaptation (2016)                                   | source 와 target feature 의 2 차 통계량 (covariance) 을 정렬하는 CORAL loss 를 end-to-end deep loss 로 통합 |
| Feature Matching | Optimal Transport for Domain Adaptation (2015)                                                        | Wasserstein-2 거리 기반 entropy regularized OT 와 group-lasso class regularizer                             |
| Feature Matching | Deep Hashing Network for Unsupervised Domain Adaptation (2017)                                        | MK-MMD 로 source-target 분포 정렬하며, target 에서 entropy loss 로 source 클래스 구조 정렬                  |
| Feature Matching | Deep CORAL: Correlation Alignment for Deep Domain Adaptation (2016)                                   | CORAL 을 딥네트워크 내부 loss 로 end-to-end 최적화 가능한 형태로 확장                                       |
| Feature Matching | Return of Frustratingly Easy Domain Adaptation (2015)                                                 | CORAL (CORrelation Alignment), 소스 특징 whitening 후 타깃 covariance 로 recoloring                         |
| Feature Matching | Return of Frustratingly Easy Domain Adaptation (2015)                                                 | 2 차 통계 정렬 관점, 모델 비종속적 feature 변환                                                             |
| Feature Matching | Contrastive Domain Adaptation (2021)                                                                  | 레이블과 사전학습 없이도 UDA 성능 향상시켜, contrastive learning 확장 프레임워크                            |
| Feature Matching | Contrastive Domain Adaptation (2021)                                                                  | MMD 분포 정렬 손실을 활용한 domain alignment                                                                |
| Feature Matching | Information-Theoretical Learning of Discriminative Clusters for Unsupervised Domain Adaptation (2012) | Information-theoretic metric 최적화, domain matching term + discriminative clustering term                  |
| Feature Matching | Deep Transfer Learning with Joint Adaptation Networks (2017)                                          | feature marginal distribution aligning 대신 joint distribution alignment 를 수행                            |
| Feature Matching | Learning Transferable Features with Deep Adaptation Networks (2015)                                   | deep network 상위 representation 층에서 source-target 분포를 MK-MMD 로 정렬                                 |
| Feature Matching | Deep Hashing Network for Unsupervised Domain Adaptation (2017)                                        | supervised hashing과 domain adaptation 을 통합하는 unified framework                                        |
| Feature Matching | Maximum Classifier Discrepancy for Unsupervised Domain Adaptation (2017)                              | decision boundary 를 고려한 유효한 정렬을 위한 classifier discrepancy maximization                          |
| Feature Matching | Deep Domain Confusion: Maximizing for Domain Invariance (2014)                                        | MMD 기반 domain confusion loss 를 classification loss 와 joint optimize                                     |
| Feature Matching | Contrastive Adaptation Network for Unsupervised Domain Adaptation (2019)                              | 클래스 조건부 분포 차이를 명시적으로 정량화하고 intra-class alignment 와 inter-class separation 을 최적화   |
| Feature Matching | Deep Domain Confusion: Maximizing for Domain Invariance (2014)                                        | MMD 기반 domain confusion loss 를 classification loss 와 joint optimize                                     |
| Feature Matching | Deep Hashing Network for Unsupervised Domain Adaptation (2017)                                        | supervised hashing 과 domain adaptation 을 통합한 unified framework                                         |
| Feature Matching | Deep CORAL: Correlation Alignment for Deep Domain Adaptation (2016)                                   | CORAL 을 딥네트워크 내부 loss 로 end-to-end 최적화                                                          |
| Feature Matching | Optimal Transport for Domain Adaptation (2015)                                                        | Wasserstein-2 거리 기반 OT 와 group-lasso regularizer                                                       |
| Feature Matching | Information-Theoretical Learning of Discriminative Clusters for Unsupervised Domain Adaptation (2012) | Information-theoretic metric 최적화                                                                         |
| Feature Matching | Deep Transfer Learning with Joint Adaptation Networks (2017)                                          | feature marginal distribution 대신 joint distribution alignment                                             |

#### 2.3 Multi-Source & Gradual Domain Adaptation

여러 source 도메인 또는 intermediate 도메인을 활용하는 적응 프레임워크를 다루는 연구들이다.

| 분류                      | 논문명                                                                                     | 분류 근거                                                                                             |
| ------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| Multi-Source              | Moment Matching for Multi-Source Domain Adaptation (2018)                                  | multi-source 환경에서 source-target뿐만 아니라 source-source 분포 정렬을 동시에 수행                  |
| Multi-Source              | Moment Matching for Multi-Source Domain Adaptation (2018)                                  | Moment Distance 를 통한 분포 차이 측정과 source-source alignment                                      |
| Multi-Source              | MADAN: Multi-source Adversarial Domain Aggregation Network (2020)                          | pixel-level 번역과 domain aggregation 을 병행하는 multi-source adaptation                             |
| Multi-Source              | MADAN: Multi-source Adversarial Domain Aggregation Network (2020)                          | image-to-image translation 와 domain aggregation 통합                                                 |
| Multi-Source              | Weighted Maximum Mean Discrepancy for Unsupervised Domain Adaptation (2017)                | source 와 target 의 class prior 차이를 보정하기 위해 class-specific auxiliary weight                  |
| Gradual Domain Adaptation | Understanding Self-Training for Gradual Domain Adaptation (2020)                           | 점진적 도메인 이동을 class-conditional Wasserstein-infinity distance 로 통제하고 hard pseudo-labeling |
| Gradual Domain Adaptation | Understanding Gradual Domain Adaptation: Improved Analysis, Optimal Path and Beyond (2022) | GDA 일반화 오차 한계를 선형화하고 최적 중간 도메인 수를 이론적/실증적으로 규명                        |
| Gradual Domain Adaptation | Understanding Gradual Domain Adaptation: Improved Analysis, Optimal Path and Beyond (2022) | 가산적 오차 전파 증명, Sequential Rademacher Complexity 활용                                          |
| Gradual Domain Adaptation | Gradual Domain Adaptation without Indexed Intermediate Domains (2022)                      | coarse-to-fine domain ordering 방법론, label-free domain sequence discovery                           |
| Gradual Domain Adaptation | Understanding Self-Training for Gradual Domain Adaptation (2020)                           | class-conditional Wasserstein-infinity distance 와 margin 설정을 통한 이론적 보장                     |
| Gradual Domain Adaptation | Gradual Domain Adaptation without Indexed Intermediate Domains (2022)                      | coarse-to-fine domain ordering 방법론, label-free sequence discovery                                  |

#### 2.4 Partial & Open-Set Domain Adaptation

클래스 공간 제약이 있는 partial setting 또는 unknown 샘플을 거부하는 open-set 설정을 다루는 연구들이다.

| 분류                       | 논문명                                                                    | 분류 근거                                                                                                                                                                                |
| -------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Partial Domain Adaptation  | Importance Weighted Adversarial Nets for Partial Domain Adaptation (2018) | partial setting 에서 두 개의 domain classifier 를 분리하여 source 샘플 중요도 추정                                                                                                       |
| Partial Domain Adaptation  | Importance Weighted Adversarial Nets for Partial Domain Adaptation (2018) | second domain classifier 로 가중치를 반영한 weighted adversarial alignment                                                                                                               |
| Partial Domain Adaptation  | Learning to Transfer Examples for Partial Domain Adaptation (2019)        | source classifier 와 domain discriminator 양쪽 모두를 shared label space 중심으로 유도                                                                                                   |
| Partial Domain Adaptation  | Learning to Transfer Examples for Partial Domain Adaptation (2019)        | sample-wise weighting 메커니즘으로 transferability 가중치 적용                                                                                                                           |
| Partial Domain Adaptation  | Partial Adversarial Domain Adaptation (2018)                              | target 데이터의 source classifier 예측 분포 평균을 기반으로 class-level weighting                                                                                                        |
| Partial Domain Adaptation  | Importance Weighted Adversarial Nets for Partial Domain Adaptation (2018) | first domain classifier 로 source 샘플 중요도 가중치 산출                                                                                                                                |
| Open-Set Domain Adaptation | Open Set Domain Adaptation by Backpropagation (2018)                      | classifier 가 target 의 unknown 확률 중간값 t 를 설정한 pseudo decision boundary를 만들고, generator 가 각 target 샘플을 known/source 혹은 unknown 양쪽 중 하나로 선택하도록 적대적 학습 |
| Open-Set Domain Adaptation | Open Set Domain Adaptation by Backpropagation (2018)                      | unknown rejection 과 known alignment 를 동일한 feature space 안에서 동시에 수행                                                                                                          |

#### 2.5 Source-Free & Model Adaptation

source data 에 접근할 수 없는 제약 조건하에서 모델만 활용하는 적응 기법들이다.

| 분류        | 논문명                                                                                                            | 분류 근거                                                                                                     |
| ----------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Source-Free | Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation (2020) | source data 에 접근할 수 없는 unsupervised domain adaptation 환경, source model 만 전달 가능                  |
| Source-Free | Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation (2020) | sample-wise entropy minimization 과 dataset-wise diversity maximization                                       |
| Source-Free | Generalized Source-Free Domain Adaptation (2021)                                                                  | Target 도메인 적응 과정에서 source 도메인 성능을 유지하는 문제 해결                                           |
| Source-Free | Generalized Source-Free Domain Adaptation (2021)                                                                  | Local Structure Clustering(LSC) 과 Sparse Domain Attention(SDA) 기법 결합                                     |
| Source-Free | Source-Free Domain Adaptation for Semantic Segmentation (2021)                                                    | segmentation 에서 source data 접근 불가 문제를 generator 기반 fake sample 합성과 attention based distillation |
| Source-Free | Source-Free Domain Adaptation for Semantic Segmentation (2021)                                                    | generator 기반 synthetic source-like sample 합성, Dual Attention Distillation                                 |
| Source-Free | Model Adaptation: Historical Contrastive Learning for Unsupervised Domain Adaptation without Source Data (2021)   | Source data 부재 문제를 historical model memory 자원으로 해결하는 UMA 접근법                                  |
| Source-Free | Model Adaptation: Historical Contrastive Learning for Unsupervised Domain Adaptation without Source Data (2021)   | instance discrimination과 category discrimination이라는 두 상보적 자기지도 학습 신호 결합                     |

#### 2.6 Contrastive & Self-Supervised Learning

contrastive loss 와 자기지도 학습 시그널을 통한 표현 정렬 기법들이다.

| 분류                 | 논문명                                                                                                          | 분류 근거                                                                                                                                |
| -------------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Contrastive Learning | Contrastive Adaptation Network for Unsupervised Domain Adaptation (2019)                                        | 클래스 조건부 분포 차이를 명시적으로 정량화하고 intra-class alignment 와 inter-class separation 을 최적화하는 CDD 메트릭                 |
| Contrastive Learning | Contrastive Adaptation Network for Unsupervised Domain Adaptation (2019)                                        | clustering 기반 pseudo label 을 사용한 iterative adaptation 절차, alternating optimization                                               |
| Contrastive Learning | Contrastive Domain Adaptation (2021)                                                                            | 레이블과 사전학습 없이도 UDA 성능 50%대→70%대대 향상시켜, contrastive learning 확장 프레임워크                                           |
| Contrastive Learning | Contrastive Domain Adaptation (2021)                                                                            | 도메인별 독립 contrastive loss 로 도메인 간 간섭 방지, MMD 분포 정렬 손실                                                                |
| Contrastive Learning | Unsupervised Domain Adaptation through Self-Supervision (2019)                                                  | adversarial minimax 최적화 대신 source 와 target 모두에서 공통 self-supervised task 를 함께 학습하여 표현 정렬하는 multi-task 프레임워크 |
| Contrastive Learning | Unsupervised Domain Adaptation through Self-Supervision (2019)                                                  | 세 가지 self-supervised task(Rotation Prediction, Flip Prediction, Patch Location Prediction) 동시 학습                                  |
| Contrastive Learning | Model Adaptation: Historical Contrastive Learning for Unsupervised Domain Adaptation without Source Data (2021) | instance discrimination과 category discrimination이라는 두 상보적 자기지도 학습 신호 결합                                                |
| Contrastive Learning | Unsupervised Domain Adaptation through Self-Supervision (2019)                                                  | multi-task joint training 으로 source-target 정렬을 위한 auxiliary task 설계                                                             |
| Contrastive Learning | Contrastive Adaptation Network for Unsupervised Domain Adaptation (2019)                                        | 클래스 조건부 분포 차이를 명시적으로 정량화                                                                                              |

#### 2.7 Cross-Modality & Medical Image Adaptation

다른 modality 간 또는 의료영상 특화 domain adaptation 연구들이다.

| 분류           | 논문명                                                                                                                                        | 분류 근거                                                                                                                                           |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cross-Modality | Unsupervised Bidirectional Cross-Modality Adaptation via Deeply Synergistic Image and Feature Alignment for Medical Image Segmentation (2020) | image-level alignment 와 feature-level alignment 를 shared encoder 로 단일화하여 synergistic하게 결합하는 통합 도메인 적응 프레임워크               |
| Cross-Modality | Unsupervised Bidirectional Cross-Modality Adaptation via Deeply Synergistic Image and Feature Alignment for Medical Image Segmentation (2020) | GAN 기반 source-to-target 이미지 변환, cycle-consistency                                                                                            |
| Cross-Modality | Beyond Sharing Weights for Deep Domain Adaptation (2016)                                                                                      | source 와 target 도메인을 위한 별도의 네트워크 구조를 제시하며, weight regularizer 와 MMD 를 통해 두 스트림을 유연하게 연결                         |
| Cross-Modality | Unsupervised Bidirectional Cross-Modality Adaptation via Deeply Synergistic Image and Feature Alignment for Medical Image Segmentation (2020) | 의료영상 segmentation 에서 MRI↔CT 간 cross-modality unsupervised domain adaptation                                                                  |
| Cross-Modality | Unsupervised Bidirectional Cross-Modality Adaptation via Deeply Synergistic Image and Feature Alignment for Medical Image Segmentation (2020) | MRI↔CT 간 강대형 시각적 차이로 segmentation 성능이 급격히 저하되는 cross-modality shift 문제                                                        |
| Cross-Modality | Seismic Facies Analysis: A Deep Domain Adaptation Approach (2022)                                                                             | source domain은 labeled, target domain은 unlabeled로 설정된 unsupervised deep domain adaptation 네트워크를 통해 도메인 간 분포 차이를 CORAL 로 정렬 |
| Cross-Modality | Beyond Sharing Weights for Deep Domain Adaptation (2016)                                                                                      | weight regularizer 와 MMD 를 통해 두 스트림을 유연하게 연결하는 hybrid 접근법                                                                       |
| Cross-Modality | Unsupervised Bidirectional Cross-Modality Adaptation via Deeply Synergistic Image and Feature Alignment for Medical Image Segmentation (2020) | shared encoder synergy 를 통한 image-level 와 feature-level alignment 결합                                                                          |

#### 2.8 Semantic Segmentation & Structured Output

픽셀 단위의 structured prediction task 에서 적용하는 domain adaptation 기법들이다.

| 분류                    | 논문명                                                                        | 분류 근거                                                                                                                                                                                                         |
| ----------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Segmentation Adaptation | Learning to Adapt Structured Output Space for Semantic Segmentation (2018)    | pixel-level structured prediction task 에서 output map 의 spatial layout 과 local context 공유성 가정하며, output 분포 정렬이 feature 정렬보다 natural하고 효과적임을 GTA5/ Cityscapes 등 도시 장면 실험으로 입증 |
| Segmentation Adaptation | Learning to Adapt Structured Output Space for Semantic Segmentation (2018)    | GAN 기반 adversarial learning 을 output map 에 직접 적용, fully-convolutional discriminator 를 사용하며 multi-level 에서 conv4/conv5 에 auxiliary output 과 별도 discriminator 추가                               |
| Segmentation Adaptation | Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes (2017) | target 도메인의 쉬운 속성 예측 과제를 간접 regularize 으로 활용하는 property-matching 기반의 output-side adaptation 방식                                                                                          |
| Segmentation Adaptation | Fully Convolutional Adaptation Networks for Semantic Segmentation (2018)      | semantic segmentation 에서 dense prediction 특성을 고려한 region-level adversarial learning 을 도입하여, 이미지 스타일 전달과 feature representation 정렬을 결합한 이중층 adaptation 구조                         |
| Segmentation Adaptation | Fully Convolutional Adaptation Networks for Semantic Segmentation (2018)      | appearance-level adaptation + representation-level adaptation 의 이중 구조                                                                                                                                        |
| Segmentation Adaptation | DACS: Domain Adaptation via Cross-domain Mixed Sampling (2020)                | pseudo-label 기반 UDA 에서 pseudo-label 품질 저하로 인한 class conflation 문제를 cross-domain mixing 전략과 augmentation 설계 변경으로 완화                                                                       |
| Segmentation Adaptation | CyCADA: Cycle-Consistent Adversarial Domain Adaptation (2017)                 | pixel-level 과 feature-level adaptation 을 상보적으로 결합한 unsupervised domain adaptation 방법론으로, task-aware image translation 과 semantic preservation 을 동시에 실현                                      |
| Segmentation Adaptation | Universal Domain Adaptation for Semantic Segmentation (2025)                  | do-source-private/target-private 클래스가 혼재된 universal setting 에서 common class 의 pseudo-label confidence 가 크게 하락하고 잘못된 unknown 할당이 빈번해지는 문제 해결                                       |
| Segmentation Adaptation | Universal Domain Adaptation for Semantic Segmentation (2025)                  | domain-specific prototype 구조, target-based image matching 전략                                                                                                                                                  |

#### 2.9 Object Detection & Instance-Level Adaptation

객체 검출 모델에 적용하는 domain adaptation 기법들이다.

| 분류                 | 논문명                                                                        | 분류 근거                                                                                                                                                                                                                               |
| -------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Detection Adaptation | Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation (2021) | SSDA 환경에서 target 내부 클래스 구조 (intra-domain) 와 source-target 정렬 (inter-domain) 을 동시에 해결하기 위해 cluster-wise feature alignment 와 minimax adversarial 구조                                                            |
| Detection Adaptation | Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation (2021) | pairwise similarity label 기반 adversarial adaptive clustering loss 로 target unlabeled sample 을 클래스별 cluster 로 모음                                                                                                              |
| Detection Adaptation | Domain Adaptive Faster R-CNN for Object Detection in the Wild (2018)          | Faster R-CNN 의 RPN 과 ROI 기반 detection 구조를 활용하여 image-level 과 instance-level discrepancy 를 각각 처리하고, 두 수준의 확률적 일관성을 보장하는 consistency regularization 을 도입한 end-to-end unsupervised domain adaptation |
| Detection Adaptation | Domain Adaptive Faster R-CNN for Object Detection in the Wild (2018)          | image-level domain classifier(backbone feature map 위), instance-level domain classifier(ROI feature 위), consistency regularization(image-level 와 instance-level 확률 일관성)                                                         |
| Detection Adaptation | Progress                                                                      |                                                                                                                                                                                                                                         |

이제 방법론 분석 2 장을 작성하겠습니다.

## 2 장. 방법론 분석

### 1. 공통 문제 설정 및 접근 구조

#### 1.1 기본 문제 설정

Domain Adaptation 도메인에서 다루는 공통 문제는 **Source 도메인에서의 라벨된 학습을 Target 도메인에서 일반화**하는 것입니다.

| 설정                       | Description                         | Input                                                     | Output                              |
| -------------------------- | ----------------------------------- | --------------------------------------------------------- | ----------------------------------- |
| **Unsupervised (UDA)**     | Target 도메인 라벨 없음             | Source: $(X_s, Y_s)$, Target: $X_t$                       | Target-optimized classifier         |
| **Semi-supervised (SSDA)** | Target 소량 라벨 있음               | Source: $(X_s, Y_s)$, Target: $(X_t, Y_{t_\partial})$     | Target-optimized classifier         |
| **Partial (PDA)**          | Target 이 Source 의 부분 클래스집합 | Source: $(X_s, Y_s)$, Target: $X_t$ (shared classes only) | Shared classes-optimized classifier |
| **Open Set**               | Unknown 클래스 거부 가능            | Source: $(X_s, Y_s)$, Target: $X_t$ (known+unknown)       | Known 클래스 + Unknown rejection    |

#### 1.2 방법론적 접근 구조

전체적인 적응 파이프라인은 다음 단계로 구성됩니다:

| Domain Adaptation                                                                               |
| ----------------------------------------------------------------------------------------------- |
| Input: Source-labeled {$(x_i^s, y_i^s)$}, Target-unlabeled {$x_j^t$} ($\pm$ 소량 labeled)       |
| 1. Source Pre-training → 2. Feature/Output Alignment → 3. Target-side Refinement → 4. Inference |
| Output: Adapted classifier $h_θ: Z → P(Y\|X)$                                                   |

#### 1.3 Domain Shift 유형

- **Covariate shift**: $p_t(x) \ne p_s(x)$, $p_t(y|x) = p_s(y|x)$
- **Concept shift**: $p_t(y|x) \ne p_s(y|x)$
- **Prior shift**: $p_t(y) \ne p_s(y)$
- **Joint shift**: $p_t(x,y) \ne p_s(x,y)$

### 2. 방법론 계열 분류

제공된 53 편의 논문을 방법론 계열별로 그룹화했습니다.

#### (계열명) 형식으로 분류한 주요 계열 및 논문 목록

| 방법론 계열                             | 논문명                                                                                       | 핵심 특징                                                                      |
| --------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **Adversarial Alignment (GRL 기반)**    | Domain-Adversarial Neural Networks (2014)                                                    | Gradient Reversal Layer 를 통한 hidden representation 의 domain-invariant 학습 |
|                                         | Domain-Adversarial Training of Neural Networks (2015)                                        | GRL 층을 표준 backprop 으로 구현한 end-to-end adversarial 구조                 |
|                                         | Adversarial Discriminative Domain Adaptation (2017)                                          | unshared weights + GAN inverted label loss + discriminative 모델               |
|                                         | Conditional Adversarial Domain Adaptation (2017)                                             | feature-prediction joint alignment + multilinear conditioning                  |
|                                         | Multi-Adversarial Domain Adaptation (2018)                                                   | 클래스별 $K$개 domain discriminator + soft assignment weight                   |
|                                         | Maximum Classifier Discrepancy (2017)                                                        | 두 classifier 간 disagreement 으로 decision boundary-aware adaptation          |
|                                         | Bridging Theory and Algorithm for Domain Adaptation (2019)                                   | Margin Disparity Discrepancy(MDD) + adversarial minimax                        |
|                                         | Deep Domain Confusion (2014)                                                                 | adaptation layer + MMD 로 joint classification+domain invariance               |
| **Distribution Matching (CORAL)**       | Deep CORAL: Correlation Alignment for Deep Domain Adaptation (2016)                          | feature activation 의 covariance matching 로 domain alignment                  |
|                                         | Return of Frustratingly Easy Domain Adaptation (2015)                                        | whitening + recoloring 으로 closed-form covariance alignment                   |
|                                         | Style Normalization and Restitution for Domain Generalization and Adaptation (2021)          | Instance Normalization + channel attention 기반 residual restitution           |
| **MMD 기반 (Maximum Mean Discrepancy)** | Learning Transferable Features with Deep Adaptation Networks (2015)                          | multi-layer + multi-kernel MMD 로 upper layer 분포 정렬                        |
|                                         | Mind the Class Weight Bias: Weighted Maximum Mean Discrepancy (2017)                         | class weight bias 제거 + weighted MMD + CEM 번갈아 갱신                        |
|                                         | Deep Visual Domain Adaptation: A Survey (2018)                                               | MMD 를 discrepancy penalty 로 joint optimization                               |
|                                         | Deep Hashing Network for Unsupervised Domain Adaptation (2017)                               | VGG-F 에서 fc6,fc7 의 MK-MMD 로 domain alignment                               |
|                                         | Contrastive Domain Adaptation (2021)                                                         | FNR(False Negative Removal) + MMD 로 완전 비지도 설정                          |
|                                         | Dissimilarity Maximum Mean Discrepancy for Domain Adaptation (2021)                          | MDD + batch-wise approximation 으로 pseudo-label 기반 conditional alignment    |
| **Clustering 기반**                     | Contrastive Adaptation Network for Unsupervised Domain Adaptation (2019)                     | CDD(intra-class − inter-class) + spherical K-means + pseudo label              |
|                                         | Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation (2021)                | cluster-wise alignment + adversarial adaptive clustering + pseudo labeling     |
|                                         | Unsupervised Domain Adaptive Re-Identification: Theory and Practice (2018)                   | k-reciprocal 거리 + DBSCAN 클러스터링 + iterative self-training                |
|                                         | Understanding Self-Training for Gradual Domain Adaptation (2020)                             | margin constraint + label sharpening + class-conditional Wasserstein-infinity  |
|                                         | Unsupervised Domain Adaptation Based on Source-guided Discrepancy (2018)                     | source-optimal classifier 기준 S-disc + pseudo-labeling                        |
| **Self-supervised/Contrastive**         | Unsupervised Domain Adaptation through Self-Supervision (2019)                               | shared encoder + multi-head self-supervised task + joint training              |
|                                         | Contrastive Domain Adaptation (2021)                                                         | domain-independent contrastive loss + false negative removal + FNR             |
|                                         | Model Adaptation: Historical Contrastive Learning (2021)                                     | historical models 를 memory 로 instance/category discrimination                |
|                                         | Information-Theoretical Learning of Discriminative Clusters (2012)                           | domain matching + discriminative clustering + joint learning                   |
| **Output Space Adaptation**             | Learning to Adapt Structured Output Space for Semantic Segmentation (2018)                   | multi-level discriminator + output space 분포 정렬 + structured adaptation     |
|                                         | Curriculum Domain Adaptation for Semantic Segmentation (2017)                                | output-space posterior matching + label distribution + superpixel landmark     |
|                                         | FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation (2016)             | region-level + adversarial + constraint-based alignment                        |
| **Sample/Instance Weighting**           | Importance Weighted Adversarial Nets for Partial Domain Adaptation (2018)                    | 두 domain classifier 분리 + source importance weighting                        |
|                                         | Partial Adversarial Domain Adaptation (2018)                                                 | target 평균 예측 기반 클래스별 가중치 + classifier/domain loss 가중            |
|                                         | Learning to Transfer Examples for Partial Domain Adaptation (2019)                           | example-wise transferability weight + sample-wise weighting + leaky-softmax    |
|                                         | Frustratingly Easy Domain Adaptation (2009)                                                  | general/source/target-specific feature + 확장된 feature space                  |
|                                         | DACS: Domain Adaptation via Cross-domain Mixed Sampling (2020)                               | cross-domain mixed image + mixed label + classMix                              |
|                                         | Weighted Maximum Mean Discrepancy (2017)                                                     | class-specific auxiliary weight + target class prior 정렬                      |
| **Test-time Adaptation**                | Tent: Fully Test-time Adaptation by Entropy Minimization (2020)                              | normalization layer affine parameter + entropy minimization                    |
|                                         | Domain-Specific Batch Normalization (2019)                                                   | 도메인별 BN 통계 + self-training + 2 단계 적응                                 |
| **Source-free / Hypothesis Transfer**   | Do We Really Need to Access the Source Data? Source Hypothesis Transfer (2020)               | source data 없이 source classifier 고정 + encoder만 학습                       |
|                                         | Generalized Source-free Domain Adaptation (2021)                                             | LSC(target adaptation) + SDA(source 보존) + neighbor consistency               |
|                                         | Model Adaptation: Historical Contrastive Learning (2021)                                     | historical models 를 memory 로 source hypothesis 보존                          |
|                                         | Universal Domain Adaptation for Semantic Segmentation (2025)                                 | source-private/target-private 클래스 + prototype 기반 일반화                   |
| **Multi-source**                        | Multi-source Adversarial Domain Aggregation Network (2020)                                   | M개 source + pixel-level translation + domain aggregation + feature alignment  |
|                                         | Moment Matching for Multi-Source Domain Adaptation (2018)                                    | source-source + source-target alignment + cross-moment divergence              |
|                                         | Scatter Component Analysis (2015)                                                            | domain adaptation + domain generalization 통합 + four scatter quantities       |
| **Multi-representation**                | Multi-Representation Adaptation Network (2022)                                               | IAM 을 통한 multi-branch representation + CMMD class-conditional alignment     |
|                                         | Seismic Facies Analysis: A Deep Domain Adaptation Approach (2020)                            | CORAL alignment + transposed residual unit decoder                             |
| **Multi-task**                          | Multi-task Domain Adaptation for Sequence Tagging (2016)                                     | shared BiLSTM + domain-specific projection + CRF decoder                       |
|                                         | Domain Adaptation for Medical Image Analysis: A Survey (2021)                                | multi-source/modal + task + domain별 적응                                      |
| **Progressive / Multi-step**            | Cross-Domain Weakly-Supervised Object Detection through Progressive Domain Adaptation (2018) | CycleGAN 기반 domain transfer + pseudo-labeling 의 2 단계 적응                 |
|                                         | Progressive Domain Adaptation for Object Detection (2019)                                    | source→synthetic intermediate→target 의 2 단계 + weighted task loss            |
|                                         | Gradual Domain Adaptation without Indexed Intermediate Domains (2022)                        | coarse-to-fine 2 단계 + unindexed domains + cycle-consistency                  |
|                                         | Understanding Gradual Domain Adaptation (2022)                                               | 가산적 오차 전파 + 최적 중간 도메인 수 설계 + Wasserstein 측지선               |
|                                         | Understanding Self-Training for Gradual Domain Adaptation (2020)                             | direct adaptation 대신 gradual self-training + margin 유지                     |
|                                         | Don't Stop Pretraining (2020)                                                                | domain corpus(DAPT)→task corpus(TAPT)→fine-tuning 의 3 단계 파이프라인         |
| **Knowledge Distillation**              | Source-Free Domain Adaptation for Semantic Segmentation (2021)                               | generator 기반 synthetic sample + attention distillation + patch-level entropy |
|                                         | Domain-Adversarial Neural Networks (2014/2015)                                               | source에서 학습한 knowledge 를 target 으로 transfer                            |
| **Multi-modal / Cross-modality**        | Unsupervised Bidirectional Cross-Modality Adaptation (2020)                                  | MRI↔CT 양방향 + shared encoder + image/feature-level alignment                 |
| **Multi-step / Heterogeneous**          | Deep Visual Domain Adaptation: A Survey (2018)                                               | one-step vs multi-step 시나리오 + heterogeneous DA                             |
| **Domain Generalization / DG**          | Style Normalization and Restitution (2021)                                                   | DG + UDA 에서 style discrepancy 완화 + residual 기반 복원                      |
| **Scatter-based**                       | Scatter Component Analysis (2015)                                                            | four scatter quantities + generalized eigenvalue problem                       |

#### 2.1 계열별 주요 특징 비교

| 계열                  | 문제 접근                                         | 주요 구조                           | 데이터 활용              | 일반화 능력                     |
| --------------------- | ------------------------------------------------- | ----------------------------------- | ------------------------ | ------------------------------- |
| Adversarial Alignment | Domain classifier 를 통한 discriminator/confusion | GRL, minimax                        | Source-target joint      | 중간~높음 (hyperparameter 민감) |
| Distribution Matching | 2 차 통계량 정렬                                  | CORAL, whitening                    | Source-target joint      | 중간 (covariance 중심)          |
| MMD 기반              | Kernel space 에서 분포 거리 최소화                | Gaussian kernel, RKHS               | Source-target joint      | 높음 (multi-kernel)             |
| Clustering 기반       | 클래스 중심 정렬 + pseudo-label                   | K-means, entropy minimization       | Target pseudo-label 중심 | 높음 (self-consistency)         |
| Self-supervised       | 공통 구조 학습으로 분포 정렬                      | Multi-head task                     | Source-target joint      | 중간 (task 선택에 의존)         |
| Output Space          | 예측 분포 직접 정렬                               | Multi-level discriminator           | Target output 중심       | 높음 (structured output)        |
| Sample Weighting      | 클래스별/샘플별 중요도 부여                       | Importance weight, pseudo-label     | Target 중심              | 중간 (bias에 취약)              |
| Test-time             | 테스트 중 파라미터 조정                           | BN affine + entropy                 | Target-only              | 중간 (online)                   |
| Source-free           | Source data 없이 encoder 학습                     | Feature bank, neighbor consistency  | Target-only              | 낮음~중간                       |
| Multi-source          | 여러 source 통합                                  | Domain aggregation, moment matching | Multi-source             | 높음 (robust)                   |
| Multi-representation  | 병렬 representation 추출                          | IAM, multi-branch                   | Source-target joint      | 높음                            |
| Progressive           | 단계적 적응                                       | Intermediate domain, pseudo-label   | Multi-stage              | 높음 (gap 해소)                 |
| Multi-modal           | Cross-domain 변환 + alignment                     | Shared encoder, generators          | Multi-modality           | 높음 (synergistic)              |

### 3. 핵심 설계 패턴 분석

#### 3.1 Feature Alignment 패턴

**Feature Space Alignment**: Feature extractor 의 output 분포를 source-target 간 유사하게 만드는 것.

- **Marginal distribution alignment**: 전체 feature 분포 정렬
  - CORAL, DANN, DAN 등
  - $\min_θ E[(C_s - C_t)^2]$ (Covariance)
  - $\min_A ||A^TX_s - A^TX_t||^2$ (Linear transform)

- **Class-conditional alignment**: 클래스 조건부 분포 정렬
  - CDAN, MDD, CAN 등
  - $\min E[(C_c^s - C_c^t)^2]$ (Class-specific)
  - Pseudo-label 기반 정렬

#### 3.2 Output Space Adaptation 패턴

**Structured Output Alignment**: Prediction/output 분포를 정렬하는 접근.

- **Segmentation**: FCAN, LTL, CAN 등
  - Multi-level discriminator (conv4, conv5 등)
  - Pixel-level, region-level adaptation
  - $\min E[(L_s - L_t)^2]$ (Output distribution)

- **Classification**: DAN, GAN 등
  - Final layer output space 정렬
  - Soft label transfer

#### 3.3 Distribution Alignment 패턴

**Joint Distribution Alignment**: Feature-label joint 분포 정렬.

- **JMMD**: $\min ||E[X|Y=y]_s - E[X|Y=y]_t||^2$
- **Joint Distribution Matching**: $\min E[(C_s - C_t)^2 + (C_{s|y} - C_{t|y})^2]$

#### 3.4 Separation Pattern

**Shared/Private Representation Separation**: 도메인 공통/특화 정보 분리.

- **DSN (Domain Separation Networks)**: shared encoder + private encoder
  - Shared: domain-invariant, Private: domain-specific
  - Reconstruction loss 로 private constraint

#### 3.5 Multi-level Adaptation 패턴

**Layer-wise Adaptation**: 네트워크 다층에서 adaptation 수행.

- **Upper layer**: task-specific, domain-sensitive
- **Middle layer**: shared + adaptation
- **Lower layer**: domain-invariant, frozen/fine-tune

#### 3.6 Pseudo-labeling Pattern

**Self-supervised Label Generation**: Target 의 unlabeled 데이터에서 pseudo-label 생성.

- **Pseudo-labeling loss**: $L_{PL} = -\sum log p(y|x_t)$
- **Confidence filtering**: $mask = 1[p(y|x) > threshold]$
- **Iterative refinement**: Pseudo-label → 학습 → 갱신

#### 3.7 Cycle-consistency Pattern

**Image-level Translation + Feature alignment**: Image translation 을 통한 adaptation.

- **CycleGAN**: $S → T → S' \approx S$, $T → S → T' \approx T$
- **Cycle-consistency loss**: $L_cyc = ||G(T(x)) - x|| + ||F(S(x)) - x||$
- **Semantic-consistency**: Semantic label 보존

### 4. 방법론 비교 분석

#### 4.1 계열 간 문제 접근 방식 차이

| 계열                      | 적응 위치               | 학습 신호                     | 일반화 전략              |
| ------------------------- | ----------------------- | ----------------------------- | ------------------------ |
| **Adversarial Alignment** | Feature space           | Discriminator loss            | Domain-invariant feature |
| **Distribution Matching** | Feature statistics      | Covariance/Frobenius norm     | 2 차 통계량 정렬         |
| **MMD**                   | Kernel space            | Mean embedding distance       | RKHS 에서 분포 정렬      |
| **Clustering**            | Feature geometry        | Intra/inter-class discrepancy | 클래스 중심 정렬         |
| **Self-supervised**       | Shared encoder          | Multi-task loss               | 구조적 일반화            |
| **Output Space**          | Prediction distribution | Output discriminator          | Predictive consistency   |

#### 4.2 구조적 차이: 가중치 공유 vs 분리

| 구조                 | 계열              | 특징                             | 장단점                             |
| -------------------- | ----------------- | -------------------------------- | ---------------------------------- |
| **Shared weights**   | DANN, CORAL, DAN  | 계산 효율성, shared feature      | Negative transfer 위험             |
| **Unshared weights** | ADDA, CDAN, CDAC  | domain-specific 표현             | 계산량 증가, 더 정확한 adaptation  |
| **Shared+Private**   | DSN, S2DA         | domain-invariant + specific 분리 | Clear separation, overfitting 위험 |
| **Multi-stream**     | MADAN, Two-stream | Source별 independent 처리        | Multi-source 처리, fusion 필요     |

#### 4.3 구조적 차이를 정리한 표

| 구조 유형                      | 대표 논문                     | 네트워크 구성                                           | 적응 전략                            |
| ------------------------------ | ----------------------------- | ------------------------------------------------------- | ------------------------------------ |
| **Single-stream + GRL**        | DANN (2014), DAN (2015)       | feature extractor + label predictor + domain classifier | GRL 로 adversarial                   |
| **Two-stream**                 | Two-stream (2016), JDA (2016) | Source stream + Target stream + shared backbone         | Weight regularizer + MMD             |
| **Shared + Private**           | DSN (2016)                    | Shared encoder + domain별 private encoder + decoder     | Shared: invariant, Private: specific |
| **Multi-domain discriminator** | CDAN (2017), MADAN (2018)     | K 개의 class-wise discriminator                         | Class-conditional discrimination     |
| **Single vs Multi-classifier** | MCD (2017), JAN (2016)        | 한 개 vs 여러 classifier                                | Discrepancy via disagreement         |
| **Pseudo-label based**         | CAN (2019), CDAC (2021)       | Target pseudo-label + consistency                       | Self-training                        |
| **Cycle-consistent**           | CyCADA (2017), S2DA (2020)    | Forward + backward translation                          | Image-level + feature-level          |
| **Source-free**                | SHOT (2020), G-SFDA (2021)    | Source encoder 고정 + target encoder                    | Neighbor consistency                 |
| **Multi-source**               | MADAN (2020), M3SDA (2018)    | Source별 independent + aggregation                      | Moment matching                      |

#### 4.4 적용 대상 차이 (Task-specific)

| Domain/Task               | 대표 방법                               | 특징                                       |
| ------------------------- | --------------------------------------- | ------------------------------------------ |
| **Image Classification**  | DANN, CORAL, DAN, ADDA                  | Standard classification loss + domain loss |
| **Semantic Segmentation** | FCAN, LTL, CAN, CyCADA                  | Pixel-level, region-level adaptation       |
| **Object Detection**      | DADAN, Progressive PDA, DA-Faster R-CNN | Instance-level + image-level               |
| **Re-ID**                 | Unsupervised Re-ID, Invariance Matters  | Pairwise matching, exemplar memory         |
| **Medical Imaging**       | Cross-modality, EarthAdaptNet           | MRI↔CT, cycle-consistency                  |
| **NLP/Sequence Tagging**  | MTDA, DAPT, TAPT                        | BiLSTM-CRF, adapter, pretraining           |
| **Seismic Analysis**      | Seismic Facies Analysis                 | Transposed residual, CORAL                 |
| **Person Re-ID**          | Exemplar Memory, k-reciprocal           | Pairwise, clustering                       |

#### 4.5 복잡도 및 확장성 차이

| 계열                     | 계산 복잡도               | 메모리 사용량       | 하이퍼파라미터                     | 확장성                      |
| ------------------------ | ------------------------- | ------------------- | ---------------------------------- | --------------------------- |
| **Adversarial**          | 중간 (discriminator 추가) | 중간                | $\lambda$, learning rate, schedule | 중간 (discriminator 과적합) |
| **CORAL**                | 낮음 (closed-form)        | 낮음                | $\lambda$ (regularizer)            | 높음                        |
| **MMD**                  | 중간 (multi-kernel)       | 중간                | Kernel bandwidth, bandwidths       | 중간                        |
| **Clustering**           | 높음 (clustering 반복)    | 높음 (feature bank) | $K$, threshold, iterations         | 낮음 (batch 크기 의존)      |
| **Self-supervised**      | 중간~높음 (multi-task)    | 중간                | Task weight, temperature           | 높음                        |
| **Multi-source**         | 높음 (multi-source)       | 높음                | Source weight                      | 중간 (scaling)              |
| **Multi-representation** | 높음 (multi-branch)       | 높음                | Branch weight                      | 중간                        |

### 5. 방법론 흐름 및 진화

#### 5.1 초기 접근 (2009-2014): Shallow Adaptation, Simple Alignment

| 기간          | 대표 방법                                | 접근 방식                | 특징                               |
| ------------- | ---------------------------------------- | ------------------------ | ---------------------------------- |
| **2009-2012** | Frustratingly Easy DA, TCA               | Shallow, kernel method   | Feature space 변환, closed-form    |
| **2012-2014** | Info-Theoretic Clusters, UDA by Backprop | Joint learning, GRL      | Target clustering, adversarial     |
| **2014-2015** | DANN, CORAL, UDA by Backprop             | Deep network, GRL, CORAL | Feature alignment, joint objective |
| **2015-2016** | Deep CORAL, JAN, DANN                    | Deep adaptation          | Multi-layer, joint distribution    |
| **2016-2017** | Deep CORAL, DSN, Two-stream              | Advanced alignment       | Covariance, multi-stream           |

**특징**:

- **Shallow model**: Kernel method, closed-form solution
- **Simple feature alignment**: CORAL, whitening
- **Early adversarial**: Single GRL, minimax

#### 5.2 발전된 구조 (2017-2019): Multi-level, Multi-objective, Multi-domain

| 기간     | 대표 방법                | 구조적 발전                              | 특징                          |
| -------- | ------------------------ | ---------------------------------------- | ----------------------------- |
| **2017** | ADDA, CyCADA, CDAN, CDAC | Multi-discriminator, Cycle, Condition    | Class-aware, multi-level      |
| **2018** | DAN, FCAN, LTL, PWDA     | Multi-kernel, Output space, Progressive  | Multi-objective, intermediate |
| **2019** | CAN, CDAC, DAPT          | Clustering, Self-supervised, Pretraining | Self-training, multi-task     |

**특징**:

- **Multi-level adaptation**: Multi-layer, multi-output
- **Multi-objective**: Classification + multiple alignment losses
- **Self-supervised**: Pseudo-label, consistency, self-training
- **Intermediate domain**: Progressive adaptation, synthetic bridge

#### 5.3 최근 경향 (2020-2025): Source-free, Multi-source, Test-time

| 기간          | 대표 방법                        | 진화 방향                                            | 특징                                    |
| ------------- | -------------------------------- | ---------------------------------------------------- | --------------------------------------- |
| **2020**      | SHOT, Tent, DAPT, DAIG           | Source-free, Test-time, Multi-modal                  | Source-free, online, multi-modal        |
| **2021**      | G-SFDA, UMA, UniDA-SS            | Source-preserving, Memory-based, Universal           | Forgetting prevention, historical, open |
| **2022-2025** | MRAN, GDA analysis, Universal DA | Multi-representation, Theory analysis, Comprehensive | Multi-branch, theoretical guarantee     |

**진화 트렌드**:

1. **Source-free → Source-preserving**: Source data 없이도 보존
2. **Single-source → Multi-source**: 단일 source → 여러 source
3. **Global → Class-conditional → Instance-level**: Coarse → Fine-grained
4. **Training-time → Test-time**: Offline → Online adaptation
5. **Feature-only → Multi-representation**: Single → Multi-branch
6. **Training → Test-time**: Source-free, label-free, online

#### 5.4 이론적 발전 흐름

| 시기                 | 이론적 발전                          | 설명                            |
| -------------------- | ------------------------------------ | ------------------------------- |
| **초기 (2010-2015)** | PAC-Bayes, discrepancy bound         | Generalization error bound 제시 |
| **중기 (2016-2019)** | Rademacher complexity, Margin theory | Empirical vs true discrepancy   |
| **최근 (2020-2025)** | Sequential Rademacher, Wasserstein   | Online learning, additive error |

**이론적 이해의 진화**:

- error_bound = $O(exp(T))$ (Exponential growth) → $O(T)$ (Additive growth)
- $W_\infty$ → 모든 $p$-Wasserstein
- Ramp Loss → 모든 $\rho$-Lipschitz loss

### 6. 종합 정리

#### 6.1 방법론 지형 요약

제공된 53 편의 논문을 종합하면, **Domain Adaptation 방법론은 문제 설정 (UDA/SSDA/PDA/Open), 적응 위치 (feature/output/joint), 데이터 활용 방식 (joint/self/test-time), source 접근 (with/out)** 을 기준으로 15 개 주요 계열로 나뉩니다.

#### 6.2 주요 축

**Axis 1: Problem Setting**

- **Supervised → Unsupervised → Semi-supervised → Open Set**
- 라벨 가용성 축

**Axis 2: Adaptation Position**

- **Feature space → Output space → Joint distribution**
- Adaptation 대상 축

**Axis 3: Data Utilization**

- **Source-target joint → Pseudo-label → Self-supervised → Test-time → Source-free**
- 데이터 활용 축

**Axis 4: Source Access**

- **With source → Without source → Historical models → Multi-source**
- Source 접근 축

**Axis 5: Representation Type**

- **Marginal → Class-conditional → Instance-level → Multi-representation**
- 표현 축

#### 6.3 종합적 분석

Domain Adaptation 방법론의 **핵심 목표**는**Source 도메인에서 학습한 지식을 Target 도메인으로 효과적으로 전이**하는 것이며, 이를 위해**Distribution alignment, Structure learning, Consistency regularization** 등 다양한 전략을 사용합니다.

- **Initial phase (2009-2015)**: Shallow model, kernel method, CORAL, GRL 도입
- **Development phase (2016-2019)**: Multi-level, multi-objective, self-supervised, progressive
- **Advanced phase (2020-2025)**: Source-free, test-time, multi-source, multi-representation, theory integration

이러한 진화는 **단순 feature alignment → 구조적 표현 학습 → 테스트 기반 적응 → theory-guided design**으로 이어집니다.

## 3장. 실험결과 분석

## 1. 평가 구조 및 공통 실험 설정

### 1.1 주요 데이터셋 유형

제공된 실험 결과 문서들을 종합하면 다음과 같은 데이터셋 유형이 주로 사용되었습니다.

**이미지 분류 데이터셋:**

- **Office 벤치마크**: Office-31 (31 클래스, 3 도메인), Office-Home (65 클래스, 4 도메인), Office-Caltech (10 공유 클래스)
- **Digit Recognition**: MNIST/USPS/SVHN (digit adaptation 벤치마크)
- **Semantic Segmentation**: Cityscapes, GTA5, SYNTHIA, BDDS
- **Person Re-ID**: DukeMTMC-reID, Market-1501, MSMT17

**객체 검출 데이터셋:**

- Cityscapes, Foggy Cityscapes, BDD100k, KITTI, PASCAL VOC, Clipart1k, Watercolor2k, Comic2k

**NLP/시각 데이터셋:**

- Amazon reviews (4 도메인)
- WMT 번역 벤치마크
- ImageNet, Pascal VOC, Caltech-256

**의료 영상 데이터셋:**

- MRI/CT cross-modality datasets

**의료/과학 도메인:**

- BioMed, CS, News, Reviews 도메인 분류 작업

### 1.2 평가 환경

**실험 환경 유형:**

1. **실제 도메인 적응 실험**: Office, VisDA, Cityscapes 등에서 실제 성능 검증
2. **시뮬레이션 환경**: 인공 데이터 생성 (Rotated MNIST, Gaussian synthetic 등)
3. **Synthetic-to-real 적응**: GTA5→Cityscapes, SYNTHIA→Cityscapes
4. **교차 도메인 적응**: MNIST→MNIST-M, SVHN→MNIST, cross-city segmentation

**Baseline 비교 방식:**

- Source-only baseline (transfer without adaptation)
- 기존 SOTA 방법들 (DANN, DAN, ADDA, JAN, CyCADA 등)
- Ideal case/Oracle (상한선 성능)
- Cross-domain mixing only baseline

### 1.3 주요 평가 지표

| 평가 유형            | 주요 지표                          | 설명                                         |
| -------------------- | ---------------------------------- | -------------------------------------------- |
| **분류 정확도**      | Target classification accuracy (%) | 도메인 적응 후 타겟 도메인에서의 정확도      |
| **Segmentation**     | mIoU, IoU, PA                      | 평균 Intersection over Union, Pixel Accuracy |
| **Object Detection** | AP, mAP (IoU=0.5 기준)             | Average Precision                            |
| **Re-ID**            | CMC (rank-1/5/10), mAP             | Cumulative Match Classification              |
| **Medical**          | Dice coefficient, ASD              | Segmentation 유사도, 표면 거리               |
| **NLP**              | BLEU score, F1-score               | 번역/분류 성능                               |

### 1.4 공통 실험 설정 패턴

**배경망 (Backbone):**

- ResNet-50/101 (ImageNet 사전 학습)
- DeepLabV3/V2 (Segmentation)
- Faster R-CNN (Object detection)
- FCN-8s, SegNet
- LeNet (digit adaptation)
- AlexNet (early papers)

**학습 방법:**

- Stage-wise training (pre-training → adaptation)
- Progressive learning (gradual λ 스케줄)
- Self-training (pseudo-labeling)
- Multi-stage (pixel→task→feature)

**가중치 전략:**

- Shared/Unshared weights
- Frozen classifier (source-free)
- Gradient reversal/Reversal layer

## 2. 주요 실험 결과 정렬

### 2.1 Office-31 벤치마크에서의 방법론 비교

| 논문명                                                                        | 데이터셋/환경                              | 비교 대상                                         | 평가 지표   | 핵심 결과                                                                                                     |
| ----------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------- |
| Bridging Theory and Algorithm for Domain Adaptation (2019)                    | Office-31, Office-Home, VisDA-2017         | DAN, DANN, ADDA, JAN, CDAN                        | 평균 정확도 | Office-31 **88.9%** (CDAN 87.7%), Office-Home **68.1%** (CDAN 65.8%), VisDA-2017 **74.6%** (CDAN 70.0%)       |
| Conditional Adversarial Domain Adaptation (2018)                              | Office-31, Office-Home, VisDA-2017         | DAN, RTN, DANN, ADDA, CDAN+E                      | 평균 정확도 | Office-31 **87.7%** (JAN 84.3%), Office-Home **65.8%** (DANN 57.6%), VisDA-2017 **70.0%** (GTA 69.5%)         |
| Contrastive Adaptation Network for Unsupervised Domain Adaptation (2019)      | Office-31 (31 클래스), VisDA-2017          | RevGrad, DAN, JAN, MADA, CDAN                     | 평균 정확도 | Office-31 **90.6%** (JAN 87.7% 대비 +2.9%), VisDA-2017 **87.2%** (SE 83.8% 대비)                              |
| Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation (2021) | Office-31 (1/3-shot), Office-Home (3-shot) | DANN, MME, UODA, BiAT, APE                        | 평균 정확도 | Office-31 3-shot 평균 **70.0%**, Office-Home 3-shot 평균 **56.8%**, DomainNet 3-shot ResNet34 **76.0%**       |
| Deep Domain Adaptation for Semantic Segmentation of Urban Scenes (2017)       | GTA5→Cityscapes                            | FCNs in the Wild, source-only, global alignment   | mIoU        | Ours (I+SP) **29.0** (NoAdapt 22.0 대비 +7.0), GTA 설정 Ours (I+SP) **28.9** (NoAdapt 22.3 대비)              |
| DACS: Domain Adaptation via Cross-domain Mixed Sampling (2020)                | GTA5→Cityscapes, SYNTHIA→Cityscapes        | source-only, IAST, PIT, pseudo-label only, CUTMix | mIoU        | GTA5→Cityscapes **52.14** (source-only 32.85 대비), SYNTHIA→Cityscapes 16-class **48.34**, 13-class **54.81** |

### 2.2 Digit adaptation 벤치마크에서의 방법론 비교

| 논문명                                                                         | 데이터셋/환경                                 | 비교 대상                    | 평가 지표   | 핵심 결과                                                                     |
| ------------------------------------------------------------------------------ | --------------------------------------------- | ---------------------------- | ----------- | ----------------------------------------------------------------------------- |
| Domain-Adversarial Neural Networks (2015)                                      | MNIST→MNIST-M, SVHN→MNIST, USPS→MNIST         | NN, SVM, mSDA                | 정확도      | MNIST→MNIST-M **81.49** (source-only 0.5749 대비 +24%), SVHN→MNIST **0.7107** |
| Domain-Adversarial Training of Neural Networks (2015)                          | Inter-twining moons, Office, sentiment, image | NN, SVM, mSDA, DANN          | 정확도      | 전체 평균 **82.43%** (MMD 81.22% 대비), 12 개 task 중 10 개에서 최고 성능     |
| VADA/DIRT-T (2018)                                                             | MNIST→MNIST-M, SVHN→MNIST, STL→CIFAR          | MMD, DANN, DRCN, source-only | 정확도      | MNIST→MNIST-M **98.9%** (VADA 97.7%), SVHN→MNIST **99.4%** (VADA 97.9%)       |
| Contrastive Domain Adaptation (2021)                                           | MNIST→USPS, SVHN→MNIST, MNIST→MNIST-M         | SimClr-Base, DANN, ADDA      | 평균 정확도 | **76.8%** (SimClr-Base 대비 19%p 향상), S→M에서 **76.2%** (DANN 73.8% 대비)   |
| Generate To Adapt (2017)                                                       | MNIST/USPS/SVHN, OFFICE, synthetic-to-real    | RevGrad, DRCN, ADDA, JAN     | 평균 정확도 | OFFICE 평균 **86.5%** (JAN 84.3% 대비), MNIST→USPS(full) **95.3±0.7%**        |
| Do We Really Need to Access the Source Data? Source Hypothesis Transfer (2020) | SVHN↔MNIST↔USPS, Office, Office-Home, VisDA-C | ADDA, CDAN, ADR, SAFN        | 평균 정확도 | Digits 평균 **98.3%** (M→U 97.9%, Office**88.6%**, Office-Home**71.8%**)      |

### 2.3 Semantic Segmentation 벤치마크에서의 방법론 비교

| 논문명                                                                              | 데이터셋/환경                                              | 비교 대상                                         | 평가 지표 | 핵심 결과                                                                                               |
| ----------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------- |
| Bidirectional Learning for Domain Adaptation of Semantic Segmentation (2019)        | GTA5→Cityscapes, SYNTHIA-RAND-CITYSCAPES→Cityscapes        | AdaptSegNet, DCAN, CLAN, CyCADA, baseline         | mIoU      | GTA5→Cityscapes (ResNet101) **48.5** (baseline 33.6 대비 +14.9), SYNTHIA→Cityscapes (ResNet101)**51.4** |
| CyCADA: Cycle-Consistent Adversarial Domain Adaptation (2017)                       | MNIST/USPS/SVHN, SYNTHIA→Cityscapes, GTA5→Cityscapes       | source-only, pixel-only, feature-only, DANN       | mIoU      | GTA5→Cityscapes **35.4%** (source-only 17.9% 대비), SVHN→MNIST**90.4%** (pixel-only 70.3% 대비)         |
| FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation (2016)    | GTA5→Cityscapes, SYNTHIA→Cityscapes, cross-city            | dilated FCN, global alignment                     | mIoU      | GTA5→Cityscapes **27.1** (baseline 21.1 대비 +6.0), SYNTHIA→Cityscapes**17.0**                          |
| Fully Convolutional Adaptation Networks for Semantic Segmentation (2018)            | GTA5→Cityscapes, Cityscapes→BDDS                           | FCN, Domain Confusion, ADDA, FCNWild              | mIoU      | FCN 29.15% → **46.60%** (+17.45%p), BDDS 최종**47.53%**                                                 |
| Learning to Adapt Structured Output Space for Semantic Segmentation (2018)          | GTA5→Cityscapes, SYNTHIA→Cityscapes                        | FCNs in the Wild, CDA, feature adaptation         | mIoU      | GTA5→Cityscapes output multi-level **42.4** (feature 39.3 대비), SYNTHIA→Cityscapes**46.7**             |
| Learning to Transfer Examples for Partial Domain Adaptation (2019)                  | GTA5→Cityscapes, SYNTHIA→Cityscapes, Cityscapes→Cross-City | FCN, feature adaptation, output single-level      | mIoU      | GTA5→Cityscapes **42.4**, SYNTHIA→Cityscapes**46.7**, Cityscapes→Rome**53.8**, Tokyo**49.9**            |
| Source-Free Domain Adaptation for Semantic Segmentation (2021)                      | GTA5→Cityscapes, SYNTHIA→Cityscapes, Cityscapes→Cross-City | source-only, MinEnt, AdaptSegNet, CBST, MaxSquare | mIoU      | GTA5→Cityscapes DeepLabV3 **43.16** (source-only 34.09 대비), SYNTHIA→Cityscapes 16-class**39.20**      |
| Unsupervised Domain Adaptation through Self-Supervision (2019)                      | GTA5→Cityscapes, SYNTHIA→Cityscapes, Synscapes→Cityscapes  | AdaptSegNet, MinEnt, AdvEnt, baseline             | mIoU      | GTA5→Cityscapes **46.3** mIoU (AdvEnt 43.8 대비)                                                        |
| Style Normalization and Restitution for Domain Generalization and Adaptation (2021) | GTA5→Cityscapes                                            | Baseline-IN, Faster R-CNN                         | mIoU      | mIoU **36.16** (baseline 29.84 대비 +6.32%)                                                             |
| Seismic Facies Analysis: A Deep Domain Adaptation Approach (2022)                   | Seismic facies (offshore→Canada)                           | Existing segmentation architectures               | Accuracy  | EAN: pixel-level accuracy **84%**, EAN-DDA: 특정 클래스 최대**99%**                                     |

### 2.4 Object Detection 벤치마크에서의 방법론 비교

| 논문명                                                                                  | 데이터셋/환경                                                    | 비교 대상                                                    | 평가 지표                   | 핵심 결과                                                                                                                         |
| --------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------ | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Domain Adaptive Faster R-CNN for Object Detection in the Wild (2018)                    | SIM10K→Cityscapes, Cityscapes→Foggy Cityscapes, KITTI→Cityscapes | baseline, image-level only, instance-level only, joint       | AP (car class), mAP         | SIM10K→Cityscapes Car AP **38.97** (baseline 30.12 대비 +8.8), Cityscapes→Foggy mAP**27.6** (baseline 18.8 대비 +8.6)             |
| Progressive Domain Adaptation for Object Detection (2019)                               | KITTI→Cityscapes, Cityscapes→Foggy, Cityscapes→BDD100k           | Faster R-CNN in the wild, Diversify & Match, Selective Align | AP (KITTI), mAP (Foggy/BDD) | KITTI→Cityscapes **43.9** (w/o synthetic 38.2, synthetic augment 40.6 대비), Cityscapes→Foggy**36.9**, Cityscapes→BDD100k**24.3** |
| Unsupervised Domain Adaptive Re-Identification: Theory and Practice (2018)              | DukeMTMC-reID→Market-1501, Market→Duke                           | Direct Transfer, Self-training Baseline, SPGAN/TJ-AIDL/ARN   | CMC, mAP                    | Duke→Market: **75.8** mAP**53.7** (Baseline 66.7/39.6, ARN 70.3/39.4), Market→Duke**68.4** mAP**49.0**                            |
| Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification (2019) | Market-1501→DukeMTMC, DukeMTMC→Market-1501                       | HHL, LOMO, BoW, CAMEL, UMDL, PUL, PTGAN, SPGAN, MMFA         | CMC (rank-1), mAP           | Market→Duke rank-1 **75.1%** (HHL 62.2% 대비 +12.9), mAP**43.0**                                                                  |
| Style Normalization and Restitution for Domain Generalization and Adaptation (2021)     | Foggy Cityscapes                                                 | Faster R-CNN, DA Faster R-CNN                                | mAP                         | mAP **22.3** (baseline 18.8 대비 +3.5)                                                                                            |

## 3. 성능 패턴 및 경향 분석

### 3.1 공통적으로 나타나는 성능 개선 패턴

**1. Deep feature adaptation vs Shallow adaptation**

- Deep CORAL [Office dataset]: 평균 정확도 **72.1** (CNN 70.1, DDC 70.6, DAN 71.3 대비 +1~2%)
- DAN [Office-31]: 평균 **72.9** (DDC 70.6% 대비)
- DANN: 여러 벤치마크에서 **80~90%** 대 정확도 일관성

**2. Multi-stage/Progressive adaptation 의 효과**

- CyCADA: pixel-level adaptation → task learning → feature-level adaptation의 3 단계 구조
- Progressive Domain Adaptation for Object Detection: synthetic intermediate domain 도입으로 **38.2→43.9** (5.7 점 향상)
- Bi-directional Learning: baseline **33.6→48.5** mIoU (GTA5→Cityscapes)

**3. Output space adaptation vs Feature adaptation (Segmentation)**

- Learning to Adapt Structured Output Space: feature adaptation **39.3** 대비 output multi-level**42.4** (+3.1)
- GTA5→Cityscapes에서 output space adaptation이 feature보다 stable하고 robust함

**4. Pseudo-labeling 기반 방법의 성능**

- CDAC: PL 단독 +13.4p, AAC +7.6p 개선 (DomainNet 3-shot)
- Tent: entropy minimization만으로 source-free **14.3%** error (BN 17.3% 대비)
- Self-training: pseudo-label 품질이 결정적 (accuracy 급상승)

### 3.2 특정 조건에서만 성능이 향상되는 경우

**1. Partial domain adaptation에서의 클래스 불일치**

- PADA: Office-Home **62.06** (ResNet 53.71, DANN 47.39 대비)
- IWAN: Office-31 **87.57** (SAN 87.27% 대비)
- ETN: Office-Home **70.45** (SAN 65.30 대비)
- **Pattern**: source와 target의 클래스 공간이 완전히 겹치지 않을 경우 (label-space mismatch), class weighting 기법이 필수

**2. Synthetic-to-real gap이 큰 경우**

- Generate To Adapt: synthetic-to-real **50.4%** (baseline 38.1 대비 +12.3%p)
- FCNs in the Wild: GTA5→Cityscapes **27.1** (baseline 21.1 대비)
- **Pattern**: large domain gap에서는 adversarial alignment와 class-size constraint가 필수

**3. Multi-source setting**

- M3SDA: Digit-Five 평균 **86.13%** (M3SDA-$\beta$ 87.65%)
- Moment matching: source-source alignment이 **9.7%** 추가 성능
- **Pattern**: 여러 source 가 있을 경우 moment matching이 negative transfer 방지

**4. Small/Long-tail 클래스**

- CAN: intra-class alignment만으로는 결정경계 불안정 해소 불가
- ADDA: SVHN→MNIST 에서 counter 클래스 **44.7%** vs Source only**2.9%** (큰 개선), pillow 클래스 감소
- **Pattern**: long-tail 클래스는 pseudo-label 품질에 민감

### 3.3 논문 간 상충되는 결과

**1. Deep CORAL vs DAN (Office dataset)**

- Deep CORAL: 평균 **72.1**
- DAN: 평균 **71.3**
- **상충**: CORAL이 전체적으로 더 높으나 DAN은 class-conditional matching 제안

**2. DANN vs ADDA (Digit adaptation)**

- ADDA: MNIST→USPS **0.894**, SVHN→MNIST**0.760**
- DANN: MNIST→MNIST-M **0.8149** (source-only**0.5749** 대비)
- **상충**: DANN이 digit adaptation에서 ADDA보다 낮은 성능

**3. M3SDA vs Source Combine**

- M3SDA: Digit-Five **86.13%**
- Source combine: 성능 저하 (negative transfer 가능성)
- **상충**: simple source ensemble 보다는 structured alignment이 유리

**4. DACS vs IAST (Segmentation)**

- DACS: GTA5→Cityscapes **52.14**
- IAST: **51.5**
- **상충**: cross-domain mixing vs other alignment 방법 간 경쟁

**5. DANN vs WDAN (Class weight bias)**

- WDAN: Office-10+Caltech-10 **89.2** (DAN 87.3 대비 +1.9)
- **상충**: WDAN이 class weight bias 고려로 약한 개선만

**6. M3SDA vs SNR (Universal adaptation)**

- M3SDA: Digit-Five **86.13%**
- SNR: Digit-Five **94.12** (+7.99%)
- **상충**: SNR이 M3SDA보다 높은 성능

### 3.4 데이터셋 또는 환경에 따른 성능 차이

**1. Synthetic-to-real vs Cross-domain**

| 설정                          | 대표 방법       | 평균 성능  |
| ----------------------------- | --------------- | ---------- |
| Office-31 (object)            | MADA/CDAN       | 85~90%     |
| Digit adaptation              | DIRT-T/VADA     | 95~99%     |
| GTA→Cityscapes (segmentation) | FCN/AdaptSegNet | 40~48 mIoU |
| VisDA-C (synthetic→real)      | CDAN/ATM        | 80~87%     |

**2. Image classification vs Segmentation vs Detection**

| Task                 | 평균 성능 범위 | 한계                           |
| -------------------- | -------------- | ------------------------------ |
| Image classification | 65~95%         | Oracle 대비 여전히 5~15%p 격차 |
| Segmentation         | 29~52 mIoU     | Oracle 65.1 대비 13~36%p 격차  |
| Detection            | 22~44 mAP      | Oracle 39~55 대비 15~25%p 격차 |

**3. Small domain shift vs Large domain shift**

| Shift 크기 | 예시                      | 적응 필요성                            |
| ---------- | ------------------------- | -------------------------------------- |
| Small      | MNIST→USPS                | pixel-only adaptation **70.3→90.4%**   |
| Medium     | SVHN→MNIST                | feature adaptation 필수                |
| Large      | SVHN→USPS, GTA→Cityscapes | multi-stage/pixel+feat adaptation 필요 |

**4. Closed-set vs Open-set vs Partial**

| 설정       | 정확도 영향            | 주요 기법       |
| ---------- | ---------------------- | --------------- |
| Closed-set | Baseline               | DANN, DRCN      |
| Open-set   | Unknown rejection 필요 | OSDA, TENT      |
| Partial    | Negative transfer      | PADA, ETN, IWAN |

### 3.5 성능 격차 분석

**1. Source-only vs Adaptation**

| 데이터셋        | Source-only | Adaptation | 개선율  |
| --------------- | ----------- | ---------- | ------- |
| MNIST→MNIST-M   | 56.6%       | 83.2%      | +26.6%  |
| SVHN→MNIST      | 59.2%       | 82.7%      | +23.5%  |
| GTA5→Cityscapes | 29.15%      | 46.60%     | +17.45% |
| Office-31       | 65~70%      | 80~90%     | +10~20% |

**2. Oracle vs Adapted (실제적 한계)**

| 데이터셋       | Oracle         | Adapted | 격차   |
| -------------- | -------------- | ------- | ------ |
| Office-31      | 상한선         | 88.9%   | 5~10%p |
| VisDA-C        | 상한선         | 83.5%   | 5~15%p |
| GTA→Cityscapes | 65.1% (oracle) | 48.5%   | 16.6%p |

## 4. 추가 실험 및 검증 패턴

### 4.1 Ablation Study 패턴

**1. 구성 요소별 기여도 분석**

- **Bi-directional Learning**: adversarial (+7.3), self-supervised (+4.5), bidirectional iteration (+0.6)
- **CAN**: w/o AO 88.1→90.6, w/o CAS 89.1→90.6, pseudo-label 79.8/83.4→CAN 86.1
- **SNR**: Dual restitution loss 제거 -1.0%, +/− loss 각 하나 제거 시 성능 하락
- **MDD**: 완전/2항/1항/제거 96.1/95.3/93.0/91.9 (α 민감도 0.01 최적)

**2. Progressive λ 스케줄**

- JAN: 점진적 $\lambda_p$ 스케줄 사용
- JDDA: $\lambda_2=0.03$ (I), $0.01$ (C)
- SNR: 모든 stage 삽입 시 최적

**3. Hyperparameter 민감도**

- ETN: 종 모양 곡선 (λ 민감도)
- SNR: $\lambda≈0.5$ 최적
- MDD: $\alpha=0.01$ 최적

### 4.2 Sensitivity Analysis 패턴

**1. Batch Size/Architecture 영향**

- DANN: batch size 128 (Office), 512 (CDA)
- ResNet vs AlexNet: ResNet 기반 방법들이 일반적으로 더 높음

**2. Backbone 의존성**

- GTA→Cityscapes: ResNet101 **48.5**, VGG16 **41.3** (ResNet 우위)
- DeepLabV3 vs SegNet: DeepLabV3 **43.16**, SegNet **35.86** (DeepLabV3 우위)

**3. Pseudo-label Confidence Threshold**

- Bi-directional: threshold 0.9 최적 (mIoU 46.8)
- PL: high-confidence pseudo-label 필수 (τ=0.95)

### 4.3 Multi-Setting 실험 패턴

**1. Multi-source Adaptation**

- M3SDA: Digit-Five (5 도메인), Office-Caltech (4 도메인), DomainNet (6 도메인)
- MADAN: single-source vs multi-source (digits-five)

**2. Cross-dataset Validation**

- SimDA-style: ImageNet pretrain + target fine-tuning
- Cross-domain detection: Cityscapes→Foggy, Cityscapes→BDD

**3. Multi-task Setting**

- Multi-task Domain Adaptation for Sequence Tagging: news→social media (4 데이터셋)
- F1: Separate 86.0→multi-task DA+Domain Mask 89.0

### 4.4 Comparison Baselines 전략

**1. Baseline 계층**

```text
Source-only
  ↓
Deep CORAL / DAN / DANN (early deep methods)
  ↓
JAN / ADDA (joint adversarial)
  ↓
CDAN / CDAC / CAN (class-aware)
  ↓
SOTA 방법들
```

**2. Oracle 상한선 설정**

- GTA→Cityscapes: 65.1% (ground-truth label 학습)
- Oracle: target label 활용 가능 case
- Relative gain 계산

**3. Source-only 성능**

- Digits adaptation: 67~95% (domain gap 크기 따라)
- GTA→Cityscapes: 29~34% (segmentation)
- Office-31: 65~70%

## 5. 실험 설계의 한계 및 비교상의 주의점

### 5.1 비교 조건의 불일치

**1. Baseline 방법 다양성**

- **문제**: 각 논문마다 baseline 이 달라 fair comparison 어려움
- **예**:
  - 일부는 source-only 만 비교
  - 일부는 DANN/DAN 등 early deep method만 비교
  - 일부는 최근 SOTA 방법 포함

**2. Hyperparameter 의존성**

- DANN: λ≈0.1 (Office) vs 0.01 (VisDA)
- MDD: α=0.01 (CDAN 대비 최적)
- Bi-directional: threshold 0.9 (0.5~0.7보다 성능 저하)
- **결과**: 튜닝된 방법 vs naive 적용 비교 문제

**3. Evaluation Setting 차이**

- Transductive 설정 가정 (Office experiments)
- Inductive 설정 실험 부재
- Test-time adaptation vs Train-time adaptation 구분 불명확

### 5.2 데이터셋 의존성

**1. Office-31 중심적 연구**

- 대부분 31 클래스, 3 도메인 기준
- Office-Home 확장 있지만 31 기반으로 성능 보정
- Office dataset 규모 제한 (4,652 이미지)

**2. Digit adaptation 편향**

- MNIST/SVHN/USPS 기반 결과
- Real-world image dataset (ImageNet, COCO) 대비 한계
- Simple classification task에 국한

**3. Synthetic-to-real 편중**

- GTA5→Cityscapes, SYNTHIA→Cityscapes 만 집중
- Real-to-real adaptation 실험 부족
- Cross-domain adaptation 일반화 한계

### 5.3 일반화 한계

**1. Closed-set 가정**

- 대부분 closed-set adaptation 가정 (known classes only)
- Open-set unknown 클래스 처리 방법 제한적
-_partial-set/partial-label adaptation만 예외_

**2. Image classification only**

- VisDA, Office 등 object classification 중심
- Detection, segmentation task 에서 adaptation 한계
- Structured output (segmentation) 에서 feature adaptation 한계

**3. Scale Dependence**

- Small dataset (Office: 4.6k) 만 실험
- Large-scale dataset (DomainNet: 596k) 실험 제한적
- Large-scale 에서 negative transfer 원인 불명확

### 5.4 평가 지표의 한계

**1. Accuracy 만 중심**

- Office-31: average accuracy 만 보고
- Class-wise 정확도 무시 (long-tail 클래스 한계)
- **문제**: class imbalance 상황에서 평균 정확도만 보고

**2. mIoU 의 단점**

- GTA→Cityscapes: 40~50 mIoU
- Light/sign/motor 등 rare class 성능 낮음
- Class-wise 편차 존재 (fence 1.3% 등)

**3. Unknown rejection 지표 부재**

- Open-set adaptation 에서 unknown rejection rate 만 보고
- Known class 정확도 unknown precision 분리 필요
- Balanced performance 측정 기준 부재

**4. Adaptation Cost 측정 부재**

- Memory/Time complexity 보고 미흡
- SNR: FLOPs +9.8%, params +4.5%
- 계산 비용 vs 성능 trade-off 분석 부족

### 5.5 Domain Gap 측정 방법의 한계

**1. Proxy Metric 의존성**

- A-distance (CDAN)
- MMD (DAN/DDC)
- CORAL distance
- **문제**: proxy metric과 actual performance 간 상관관계 불명확

**2. Distribution Alignment만 측정**

- MMD, CORAL, Wasserstein distance
- Class-conditional mismatch 측정 부재
- Conditional alignment의 효과 간접적

**3. Semantic Gap 무시**

- Synthetic→Real 에서 semantic gap (texture, color)
- Real→Real 에서 domain gap (camera, season)
- Domain type별 gap 분석 부족

### 5.6 Theoretical vs Empirical Gap

**1. Theory-Bound 부재**

- Unsupervised Domain Adaptation Without Target Labels (2019): empirical ranking 약함
- Source-Free Adaptation: theory 증명 appendix 에 있음
- Bound 내 $\lambda$는 계산되지 않는 problem-dependent constant

**2. Assumption Validation 제한적**

- Cluster assumption (DIRT-T): cluster assumption 의존적
- Covariate shift 가정: label distribution 변화에 취약
- Assumption 검증 test 제한적

**3. Optimization 안정성**

- Adversarial training 불안정성 (GRL, gradient reversal)
- Convergence 분석 일부 case 만 수행
- **문제**: unstable adaptation vs stable adaptation 구분 어려움

## 6. 결과 해석의 경향

### 6.1 저자들의 공통 해석 경향

**1. "Adversarial alignment 이 핵심"**

- DANN, DSDA, ADDA, CDAN: gradient reversal layer 강조
- **해석**: source error + domain discrepancy 감소가 핵심
- **실제**: adversarial signal 은 gradient vanishing 문제 존재

**2. "Multi-stage learning 이 필수"**

- CyCADA: pixel → task → feature adaptation
- Progressive scheduling: λ 점진적 증가
- **해석**: 단계적 적응이 stable convergence
- **실제**: end-to-end vs staged debate

**3. "Pseudo-labeling 이 핵심 기여"**

- SELF-supervised learning 이 pseudo-label 기반
- High-confidence samples 만 사용 (threshold 0.9)
- **해석**: iterative refinement 가 성능 개선
- **실제**: pseudo-label 오류 누적이 발생할 수 있음

**4. "Feature adaptation > Marginal alignment"**

- Joint Adaptation Networks: joint distribution alignment
- CAN: class-aware alignment
- **해석**: class-conditional matching이 더 강력
- **실제**: MMD 기반 class-conditional alignment

**5. "Synthetic-to-real gap 에서 adversarial 이 효과적"**

- GTA→Cityscapes: adversarial alignment 필수
- FCN Wild: pixel-level adversarial
- **해석**: large domain gap 에서 gradient-based alignment
- **실제**: synthetic→real gap 은 large domain shift

### 6.2 해석 vs 실제 관찰 결과의 괴리

**1. "Adversarial training 은 stable 하다고 주장"**

- **주장**: gradient reversal 가 stable convergence
- **실제**: GRL one-step 버전 mIoU 39.9 (GRL version 39.7)
- **괴리**: gradient instability 존재

**2. "Pseudo-label 품질은 중요하다고 주장"**

- **주장**: high-confidence samples 만 사용
- **실제**: pseudo-label 오류 누적이 negative transfer 유발
- **괴리**: iterative refinement 가 반드시 성능 향상

**3. "Feature adaptation 은 항상 더 좋다고 주장"**

- **주장**: feature adaptation 가 appearance 보정
- **실제**: segmentation 에서 structured output adaptation 이 더 stable
- **괴리**: task-specific adaptation 필요

**4. "Multi-source adaptation 은 negative transfer 를 막는다고 주장"**

- **주장**: source-source alignment 이 negative transfer 방지
- **실제**: DomainNet 에서 multi-source 오히려 성능 하락
- **괴리**: negative transfer 원인 분석 부족

### 6.3 Limitation 을 강조하는 해석 패턴

**1. "Large domain shift 에서 한계"**

- MNIST→SVHN 실패 사례 (0.25 accuracy 수준)
- SVHN→MNIST 성공, 반대로 실패
- **해석**: class-conditional alignment 문제

**2. "Rare class 문제 잔존"**

- CyCADA: train/bicycle 등 rare class 적응 제한적
- CAN: class imbalance 상황 성능 흔들림
- **해석**: pseudo-label 기반 접근이 rare class 취약

**3. "Open-set/Partial adaptation 제한적"**

- Partial DA: label-space mismatch 가정
- Open-set: unknown 클래스 처리 능력 불명확
- **해석**: closed-set 가정에 의존

**4. "Large-scale 실험 부재"**

- Office: 4.6k 이미지
- DomainNet: 596k 이미지 (M3SDA 만 실험)
- **해석**: small-scale dataset 에서 일반화 문제

### 6.4 방법론 진화 경향 해석

**1. Shallow → Deep**

- KMM, KLIEP, TCA → DANN, CDAN, CAN
- **해석**: kernel matching → deep feature learning

**2. Global → Class-aware**

- MMD (global) → CDAN (class-conditional) → CAN (same-class/different-class)
- **해석**: marginal → joint/conditional matching

**3. Instance-based → Feature-based → Adversarial-based**

- DANN → CDAN → CAN, MDD
- **해석**: instance → feature → adversarial adaptation

**4. Closed-set → Partial → Open-set**

- DANN → IWAN/PADA → OSDA
- **해석**: class-complete → label-space mismatch → unknown rejection

## 7. 종합 정리

제공된 70 여 편의 논문 실험 결과들을 종합하면, domain adaptation 연구는 **source-label을 target 에 전이하는 능력**과**domain gap 을 줄이는 방법**의 두 축으로 발전해 왔음을 확인했습니다. Image classification 과 segmentation 을 중심으로 한 Office, VisDA, GTA→Cityscapes, Digit adaptation 벤치마크를 통해**adversarial alignment, multi-stage adaptation, pseudo-labeling**이 핵심 기법임을 실증하였습니다.

**성능 격차는 data-dependent**임을 드러냅니다. MNIST→MNIST-M, SVHN→MNIST 같은 작은 domain shift 에서는 DIRT-T**98.9~99.4%** 같은 높은 성능을 보이지만, 큰 domain shift(SVHN→USPS, GTA→Cityscapes) 에서는**50~60%** 대나**40~50 mIoU** 수준으로 제한적입니다. Oracle 상한선과의 격차는 5~15%p(classification) 나 13~36%p(segmentation) 로, adaptation 이 유의미한 성능 개선을 제공함은 입증되었습니다.

**데이터셋 의존성**이 큽니다. Office-31, Digit adaptation 에서 일관된 성능을 보인 방법들(GRL 기반, multi-stage) 이 GTA→Cityscapes segmentation 에서는 다른 접근법 (pixel-level adversarial, structured output adaptation) 이 필요합니다. Closed-set adaptation 에서 검증된 방법들이 Open-set/partial setting 으로 확장될 때 performance degradation 을 겪으며, label-space mismatch 에서 class weighting 이 필수적임을 보였습니다.

**Evaluation setting 의 불일치**는 fair comparison 을 어렵게 합니다. Transductive 설정이 가정되거나, source-only 성능만 baseline 으로 사용되거나, Oracle 설정이 불일치합니다. 이 모두 상대적 성능 해석에 오류를 유발합니다.

**Large-scale dataset 실험의 부재**도 일반화 한계를 명확히 합니다. Office (4.6k) 와 DomainNet (596k) 만 실험되었으며, small-scale 에서 일관성 있는 방법이 large-scale 에서 negative transfer 를 보일 수 있음이 확인됩니다.

결과적으로, **domain adaptation 은 단일 방법론이 아니라 문제 설정 (domain gap 크기, label-space overlap, evaluation setting) 에 따라 방법론을 선택**해야 하는 영역임을 이 실험 결과들은 보여줍니다.
