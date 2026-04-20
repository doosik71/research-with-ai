# Semantic Segmentation

## 서론

### 1. 연구 배경

Semantic Segmentation 연구는 입력 이미지에서 각 픽셀에 semantic class label을 할당하는 dense prediction 문제로 정의된다. 본 보고서가 다루는 연구 범위는 문제 정의 (binary, instance, open-vocabulary 등), 학습 패러다임 (weakly-supervised, semi-supervised, unsupervised, one-shot, few-shot, zero-shot 등), 적용 도메인 (medical imaging, robotic surgery, autonomous driving, general natural scene 등), 방법론적 접근 (benchmark 설계, architecture 설계, loss/design, self-supervised 등) 의 4 차원 축으로 구성된다. 벤치마크 및 도전과제, dataset/resource paper, domain adaptation, semi-supervised segmentation, multi-task/multi-modal segmentation, instance segmentation, survey/review papers 등 다양한 연구 유형이 존재하며, 각각은 데이터 효율성 향상과 일반화 성능 개선을 주요 트렌드로 공유한다.

### 2. 문제의식 및 분석 필요성

Semantic Segmentation 분야는 segmentation 하위 문제 유형별로 다른 학습 패러다임을 요구하며, 데이터 효율성 (few-shot/zero-shot), 일반화 성능 (domain adaptation), computational efficiency 가 핵심 이슈로 부각되고 있다. 특히 weakly-supervised 연구는 bounding box 기반 pseudo-label 에서 zero-shot 일반화까지 발전 흐름을 보이며, video object segmentation 은 embedding matching 과 temporal modeling 관점에서, architecture 설계는 convolution 기반 attention 과 transformer encoder 의 병행 발전으로 구분된다. 그러나 이러한 다양성 속에서 방법론들의 체계적 비교와 실제 적용 가능성에 대한 종합적 분석이 부족하며, 각 접근법의 장단점과 trade-off 관계를 명확히 파악할 필요가 있다.

### 3. 보고서의 분석 관점

본 보고서는 세 가지 분석 축으로 문헌을 정리한다. 첫째, **연구체계 분류**는 benchmark 설계, 학습 패러다임, 도메인 적용, 방법론적 접근을 4 차원 축으로 구조화하여 segmentation 연구의 분류 체계를 수립한다. 둘째,**방법론 분석**은 공통 문제 설정, 방법론 계열 분류, 핵심 설계 패턴, 비교 분석, 기술 진화 흐름을 다루며 각 방법론의 구조적 특성과 적용 대상을 명확히 구분한다. 셋째,**실험결과 분석**은 주요 데이터셋 유형, 평가 지표, 성능 비교 결과를 종합하여 다양한 설정 (unseen-heavy, large vocabulary, boundary precision 등) 에서 각 방법론의 성능과 한계를 실증적으로 평가한다.

### 4. 보고서 구성

**1 장 연구체계 분류**는 segmentation 연구의 4 차원 축 (문제 정의, 학습 패러다임, 도메인 적용, 방법론적 접근) 을 기반으로 벤치마크 및 도전과제, dataset/resource paper, weakly-supervised, few-shot/one-shot/zero-shot, video object segmentation, domain adaptation, semi-supervised, multi-task/multi-modal, instance segmentation, survey papers를 체계적으로 분류한다.

**2 장 방법론 분석**은 CNN 기반 encoder-decoder, transformer 기반, few-shot/zero-shot, domain adaptation, video object segmentation, weakly-supervised segmentation, medical image segmentation, structured prediction 등 8 개 방법론 계열을 정의하고, multi-scale feature fusion, prototype/representation learning, context encoding, contrastive learning 등 5 가지 핵심 설계 패턴을 분석한다.

**3 장 실험결과 분석**은 Cityscapes, ADE20K, PASCAL VOC, COCO Stuff, medical, zero-shot, DAVIS 등 주요 데이터셋 유형과 mIoU, Dice, J&F, hIoU 등 평가 지표를 기반으로 성능 비교 결과를 정렬하고, 성능 개선 패턴, 조건별 효과성, 데이터셋 의존성, 일반화 한계, 평가 지표 한계 등을 종합적으로 논의한다.

## 1장. 연구체계 분류

### 1. 연구 분류 체계 수립 기준

본 분류 체계는 제공된 논문 요약문의 주제적 초점, 학습 패러다임, 적용 도메인, 시스템 구성 관점을 종합하여 수립하였다. 주요 기준은 다음과 같다.

- **문제 정의**: 해결하고자 하는 segmentation 하위 문제 유형 (binary, instance, open-vocabulary 등)
- **학습 패러다임**: weakly-supervised, semi-supervised, unsupervised, one-shot, few-shot, zero-shot 등 지도 신호 유형
- **도메인 적용**: medical imaging, robotic surgery, autonomous driving, general natural scene 등
- **방법론적 접근**: benchmark 설계, architecture 설계, loss/design, self-supervised 등

### 2. 연구 분류 체계

#### 2.1 벤치마크 및 도전과제 연구

**2.1.1 Robotic Surgery Benchmark**

| 분류                                | 논문명                                                                                   | 분류 근거                                                                                                                                                                               |
| ----------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Robotic Surgery Benchmark           | 2017 Robotic Instrument Segmentation Challenge (2019)                                    | 수술 로봇 영상에서 instrument 분할을 위한 common benchmark 와 binary→parts→type 으로 점진적인 난이도 상승 구조를 가진 다층적 task 를 설계한 연구                                        |
| Robotic Surgery Benchmark           | 2018 Robotic Scene Segmentation Challenge (2020)                                         | 실제 수술 장면 이해를 위한 공개 benchmark 구축과 참가 방법 비교를 통해 segmentation 문제의 class별 난이도와 데이터 기반 한계를 체계적으로 분석한 challenge report                       |
| Video Object Segmentation Benchmark | The 2017 DAVIS Challenge on Video Object Segmentation (2017)                             | DAVIS 2017 은 새로운 알고리즘 모델이 아닌 multi-object segmentation 을 위한 벤치마크 데이터셋, 과제 정의, 평가 체계, leaderboard 인프라를 종합 제공하는 benchmark paper                 |
| Large-Scale Video Dataset           | YouTube-VOS: A Large-Scale Video Object Segmentation Benchmark (2018)                    | 새로운 알고리즘이 아닌 인프라적 데이터셋과 평가 프로토콜을 제공하는 dataset paper 로 분류되며, temporal dependency 학습을 위한 대규모 데이터 필요성을 실증하는 baseline comparison 논문 |
| Large-Scale Food Dataset            | A Large-Scale Benchmark for Food Image Segmentation (2021)                               | ingredient-level annotation 을 갖춘 대규모 food segmentation 데이터셋과 recipe 기반 multi-modal pre-training 방법을 핵심으로 하는 benchmark 구축 연구                                   |
| Medical Image Benchmark             | Robust Semantic Segmentation of Brain Tumor Regions from 3D MRIs (2020)                  | 멀티모달 3D MRI 분할을 위해 ResNet encoder-decoder 구조와 Dice+focal+active contour hybrid loss 조합으로 성능을 검증한 실용적 공학지향 논문                                             |
| Medical Image Benchmark             | OmniMedVQA: A New Large-Scale Comprehensive Evaluation Benchmark for Medical LVLM (2024) | 본 연구는 의료 영상 인식 및 분류 능력을 다각도로 평가하는 기초 벤치마크에 치우쳐 있으며, 임상 추론을 넘어서는 고차적 인지 평가는 의도하지 않음                                          |
| Large-Scale Scene Dataset           | Microsoft COCO: Common Objects in Context (2014)                                         | object recognition 을 scene understanding 의 일부로 확장하고, non-iconic context 와 instance-level precision 을 동시에 확보한 대규모 데이터셋 구축 논문으로 분류됨                      |

**2.1.2 Dataset/Resource Papers**

| 분류                | 논문명                                                                                                                 | 분류 근거                                                                                                                                                                        |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Large-Scale Dataset | COCO-Stuff: Thing and Stuff Classes in Context (2016)                                                                  | COCO-Stuff 는 대규모 dense annotation 을 위한 데이터 중심 접근으로, superpixel 기반 efficient protocol 과 thing-stuff 균형 설계가 핵심 차별점                                    |
| Domain Dataset      | DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs (2016) | Classification 기반 DCNN 을 Dense predictor 로 재구성하여 segmentation 특화 문제를 해결한 구조적 접근                                                                            |
| Domain Dataset      | Context Prior for Scene Segmentation (2020)                                                                            | semantic segmentation 에서 문맥 정보를 semantic class 기준으로 명시적 supervised learning 으로 분리하고, 이를 feature aggregation 에 직접 적용하는 구조화 관계를 학습하는 접근법 |
| Domain Dataset      | COCO-Stuff: Thing and Stuff Classes in Context (2016)                                                                  | 대규모 dense annotation 을 위한 데이터 중심 접근으로, superpixel 기반 efficient protocol 과 thing-stuff 균형 설계가 핵심 차별점                                                  |

#### 2.2 Weakly-Supervised Semantic Segmentation

**2.2.1 Weak Supervision with Bounding Boxes**

| 분류                    | 논문명                                                                                                 | 분류 근거                                                                                                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Weakly Supervised (Box) | BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation (2015) | bounding box 와 region proposals 를 결합한 반복적 pseudo-supervision 최적화 프레임워크를 통해, 약한 지도 정보만으로 완전 지도 학습과 경쟁하는 성능 달성               |
| Weakly Supervised (Box) | Constrained Convolutional Neural Networks for Weakly Supervised Segmentation (2015)                    | weak supervision 을 출력 분포에 대한 선형 제약으로 정식화하고, 제약을 만족하는 latent target 으로 CNN 을 교대 최적화하는 구조                                         |
| Weakly Supervised (Box) | BBAM: Bounding Box Attribution Map for Weakly Supervised Semantic and Instance Segmentation (2021)     | detector 의 classification 과 localization output 을 유지하는 최적 이미지 영역을 찾는 방식으로 pseudo mask 를 생성하는 weakly supervised segmentation 방법론          |
| Weakly Supervised (Box) | Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation (2021)       | BAP 와 NAL 로 pseudo label 품질과 noise robustness 를 독립적으로 최적화하는 구조를 가지며, classification representation 개선과 segmentation 성능 향상을 연결         |
| Weakly Supervised (Box) | Learning to Segment Object Candidates (2015)                                                           | segmentation 과 objectness scoring 을 shared backbone 위에 joint learning 하여 class-agnostic mask 와 ranking 을 동시에 생성하는 end-to-end object proposal 생성 체계 |

**2.2.2 Domain-Aware Weak Supervision**

| 분류                          | 논문명                                                                                                | 분류 근거                                                                                                                                    |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Domain-Aware Weak Supervision | A Comprehensive Analysis of Weakly-Supervised Semantic Segmentation in Different Image Domains (2019) | natural domain 에서 background/foreground 구분 문제, histopathology/satellite 에서 모호한 경계와 높은 class co-occurrence 문제가 핵심 어려움 |
| Domain-Aware Weak Supervision | A Review on Deep Learning Techniques Applied to Semantic Segmentation (2017)                          | deep learning 기반 semantic segmentation 의 방법론 계보, dataset/benchmark, 성능 비교를 한 프레임으로 체계적으로 정리한 초기 survey paper    |

**2.2.3 Semantic Class-Aware Weak Supervision**

| 분류                 | 논문명                                                                                           | 분류 근거                                                                                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Semantic Class-Aware | Learning to Adapt Structured Output Space for Semantic Segmentation (2018)                       | segmentation output 의 structured spatial layout 을 domain adaptation 핵심 신호로 활용하는 structured output space alignment 기반 unsupervised domain adaptation 연구 |
| Semantic Class-Aware | Classes Matter: A Fine-grained Adversarial Approach to Cross-Domain Semantic Segmentation (2020) | discriminator 가 binary domain label 을 $2K$ domain-class 조합으로 확장하여 class 정보를 명시적으로 반영하는 fine-grained adversarial framework                       |

#### 2.3 Few-Shot/One-Shot Semantic Segmentation

**2.3.1 One-Shot Learning**

| 분류              | 논문명                                                                     | 분류 근거                                                                                                                                                                                                               |
| ----------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| One-Shot Learning | One-Shot Learning for Semantic Segmentation (2017)                         | support-image 로부터 classifier 파라미터를 생성하는 두 가지 branch 구조로 unseen class 에 대한 feed-forward segmentation 을 가능하게 함                                                                                 |
| One-Shot Learning | One-Shot Video Object Segmentation (2016)                                  | video object segmentation 에서 temporal propagation 을 명시적 제약으로 쓰지 않고, 대신 강하게 학습된 object model 과 single-frame adaptation 으로 frame independence 를 실현                                            |
| One-Shot Learning | Deep Extreme Cut: From Extreme Points to Object Segmentation (2017)        | extreme points 를 Gaussian heatmap 으로 변환해 CNN 입력의 4 번째 채널로 사용하는 입력 효율형 instance segmentation 및 annotation 도구                                                                                   |
| One-Shot Learning | Associating Objects with Transformers for Video Object Segmentation (2021) | identification mechanism 을 통해 여러 객체를 단일 embedding space 에서 동시 처리하는 identification-based 접근법이며, long-term/short-term attention 결합을 통해 multi-object temporal propagation 을 계층적으로 모델링 |

**2.3.2 Few-Shot Semantic Segmentation**

| 분류                  | 논문명                                                                                                          | 분류 근거                                                                                                                                                                                                                                                      |
| --------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Few-Shot Segmentation | PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment (2019)                                     | support 에서 class prototype 을 추출하고 query 픽셀을 prototype 과의 cosine distance 로 분류하는 비모수적 거리 기반 metric learning 방식이며, PAR 을 통해 support-query 간 embedding consistency 를 개선하는 prototype alignment regularization 이 핵심 차별점 |
| Few-Shot Segmentation | CANet: Class-Agnostic Segmentation Networks with Iterative Refinement and Attentive Few-Shot Learning (2019)    | unseen 클래스 일반화를 위해 intermediate-level feature 기반 dense comparison 과 learnable iterative refinement 를 결합한 구조적 접근, support fusion 시 attention mechanism 도입                                                                               |
| Few-Shot Segmentation | Adaptive Prototype Learning and Allocation for Few-Shot Segmentation (2021)                                     | Few-shot segmentation 분야에서 single prototype 정보 손실 문제를 multi-prototype 생성과 위치별 선택으로 해결하는 adaptive representation 학습 연구                                                                                                             |
| Few-Shot Segmentation | A Location-Sensitive Local Prototype Network for Few-Shot Medical Image Segmentation (2021)                     | 의료 영상 분할에서 spatial layout prior 를 few-shot segmentation 파이프라인에 직접 통합하는 기법으로, 위치 기반 prototype 매칭을 통해 annotation scarcity 문제 해결                                                                                            |
| Few-Shot Segmentation | Anomaly Detection-Inspired Few-Shot Medical Image Segmentation Through Self-Supervision With Supervoxels (2022) | 의료영상에서 background modeling 의 근본적 한계를 피하기 위해 anomaly detection 프레임워크를 적용하고, 3D supervoxel 을 활용한 자기지도 학습으로 volumetric 구조를 반영한 few-shot 분할 방법                                                                   |
| Few-Shot Segmentation | Generalized Few-Shot Semantic Segmentation (2022)                                                               | GFS-Seg 설정은 base class learning → novel class registration → full-class evaluation 의 세 단계를 거치며, support 와 query 에서 얻은 문맥을 prototype 보정으로 통합하는 classifier-centric 접근법                                                             |
| Few-Shot Segmentation | Learning What Not to Segment: A New Perspective on Few-Shot Segmentation (2022)                                 | FSS 에서 seen-class bias 를 완화하기 위해 base learner 와 meta learner 의 dual-branch 구조를 통해 base prediction 을 suppression signal 로 활용하는 ensemble 기반 접근법                                                                                       |
| Few-Shot Segmentation | Hypercorrelation Squeeze for Few-Shot Segmentation (2021)                                                       | dense correlation 기반 few-shot segmentation 방법론, multi-level intermediate feature 를 활용한 4D tensor 관계 분석, center-pivot 4D convolution 으로 계산 효율화                                                                                              |
| Few-Shot Segmentation | Prior Guided Feature Enrichment Network for Few-Shot Segmentation (2020)                                        | 고정된 high-level feature 기반 learning-free prior 와 multi-scale enrichment module 을 통한 support-query 정보 정제가 few-shot segmentation 의 일반화와 공간 불일치 문제를 동시에 해결                                                                         |
| Few-Shot Segmentation | ShapeMask: Learning to Segment Novel Objects by Refining Shape Priors (2019)                                    | shape prior 와 instance embedding 을 결합한 다단계 refinement 구조를 통해 detection box 를 점진적으로 mask 로 정제하는 partially supervised instance segmentation 방법                                                                                         |

**2.3.3 Zero-Shot/Generalized Zero-Shot**

| 분류                   | 논문명                                                                                                    | 분류 근거                                                                                                                                                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Zero-Shot Segmentation | Zero-Shot Semantic Segmentation (2019)                                                                    | Semantic embedding 공간에서 실제 feature 분포를 모사하는 GMMN generator 를 학습한 후, synthetic unseen sample 으로 classifier 확장하는 생성 기반 zero-shot 접근                                                             |
| Zero-Shot Segmentation | Context-Aware Feature Generation for Zero-shot Semantic Segmentation (2020)                               | semantic word embedding 을 클래스 의미로, pixel-wise 문맥을 latent code 로 활용하여 context-aware feature generation 하는 GAN 기반 접근                                                                                     |
| Zero-Shot Segmentation | A Simple Baseline for Open-Vocabulary Semantic Segmentation with Pre-trained Vision-language Model (2021) | CLIP 의 image-level recognition 과 segmentation 의 pixel-level prediction 간 granularity mismatch 를 해결하기 위해 2-stage proposal-generation 및 region-classification 구조로 재정의한 open-vocabulary segmentation 방법론 |
| Zero-Shot Segmentation | Decoupling Zero-Shot Semantic Segmentation (2021)                                                         | zero-shot segmentation 문제를 픽셀 수준 분류에서 segment-level 분류로 분리 (decouple) 하는 문제 재정의 접근법을 취하며, group 예측과 semantic 분류를 단계적으로 분리하여 seen/unseen 간 지식 전이 안정성 확보               |
| Zero-Shot Segmentation | Learning Mask-aware CLIP Representations for Zero-Shot Segmentation (2023)                                | 제안된 MAFT 방법은 frozen CLIP 패러다임의 proposal 구분력 병목을 해결하는 specialized adaptation 기법                                                                                                                       |
| Zero-Shot Segmentation | FreeSeg: Unified, Universal and Open-Vocabulary Image Segmentation (2023)                                 | semantic/instance/panoptic segmentation 을 하나의 모델로 통합하면서 seen/unseen class 를 동시에 처리하는 unified open-vocabulary segmentation 프레임워크                                                                    |
| Zero-Shot Segmentation | CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation (2023)                                | cost aggregation 방식으로 CLIP 의 joint embedding space 를 보존하면서 pixel-level prediction 으로 전환, spatial-class aggregation 과 QV fine-tuning 전략을 통해 unseen-heavy 설정에서 기존 방법 대비 대폭 향상              |
| Zero-Shot Segmentation | Generative Semantic Segmentation (2023)                                                                   | segmentation mask 를 RGB color 매핑한 maskige 표현으로 이미지 생성 모델에 맞게 embedding 하여 discriminative classification 대신 conditional generation 문제로 재정의                                                       |
| Zero-Shot Segmentation | Segment Any Anomaly without Training via Hybrid Prompt Regularization (2023)                              | 학습된 foundation model 들을 언어 프롬프트와 이미지 컨텍스트 기반 정규화로 결합하여 zero-shot 설정에서 이상 분할 성능을 증명하는 연구                                                                                       |

#### 2.4 Video Object Segmentation

**2.4.1 Semi-Supervised VOS with Embedding Matching**

| 분류                | 논문명                                                                                                     | 분류 근거                                                                                                                                                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Semi-Supervised VOS | FEELVOS: Fast End-to-End Embedding Learning for Video Object Segmentation (2019)                           | 첫 프레임 global matching 과 이전 프레임 local matching 을 결합한 end-to-end 단일 네트워크 구조를 통해 fine-tuning 없이 multi-object VOS 에서 실용성 기준점 달성                                                          |
| Semi-Supervised VOS | Collaborative Video Object Segmentation by Foreground-Background Integration (2020)                        | foreground 와 background embedding 을 명시적으로 통합하고 pixel-level 과 instance-level 정보를 collaborative 하게 통합하는 semi-supervised VOS 프레임워크                                                                 |
| Semi-Supervised VOS | Collaborative Video Object Segmentation by Multi-Scale Foreground-Background Integration (2021)            | 이 논문은 VOS 에서 background modeling 을 핵심 요소로 인식하고 explicit collaborative embedding 과 multi-scale matching 으로 구현한 연구로, pixel-level/instance-level 정보 결합과 atrous matching 효율화가 방법론적 특징 |
| Semi-Supervised VOS | A Transductive Approach for Video Object Segmentation (2020)                                               | transductive inference 관점에서 label propagation 기반 semi-supervised VOS 방법이며, external modules 없이 long-term temporal dependency 를 효율적으로 활용한 simple baseline                                             |
| Semi-Supervised VOS | Anchor Diffusion for Unsupervised Video Object Segmentation (2019)                                         | unsupervised video object segmentation 문제를 anchor frame 과 현재 프레임 사이의 직접적인 pixel correspondence 학습을 통해 해결하는 optical flow 및 RNN 에 의존하지 않는 장기 시간 의존성 모델링 방법                     |
| Semi-Supervised VOS | CNN in MRF: Video Object Segmentation via Inference in A CNN-Based Higher-Order Spatio-Temporal MRF (2018) | CNN 을 MRF 의 higher-order spatial potential 로 통합한 structured optimization 접근, temporal fusion 과 mask refinement 의 반복적 추론 과정                                                                               |
| Semi-Supervised VOS | Efficient Video Object Segmentation via Network Modulation (2018)                                          | conditional adaptation 메커니즘을 통해 테스트 시 weight fine-tuning 없이 modulation parameter 생성만으로 network 를 대상 객체에 즉시 적응시키는 meta-learning 기반 video object segmentation 접근법                       |
| Semi-Supervised VOS | State-Aware Tracker for Real-Time Video Object Segmentation (2020)                                         | segmentation 과 tracking 을 상태 인식 메커니즘으로 결합한 적응형 VOS 접근법으로, Cropping Strategy Loop 와 Global Modeling Loop 를 통한 동적 시스템 조정, long-term temporal consistency 확보                             |
| Semi-Supervised VOS | Kernelized Memory Network for Video Object Segmentation (2020)                                             | VOS 의 지역적 특성을 명시적으로 반영한 국소화된 메모리 읽기 기법을 제안하며, synthetic occlusion 을 통한 강건성 향상을 결합한 하이브리드 오프라인 분할 방법                                                               |
| Semi-Supervised VOS | Decoupling Features in Hierarchical Propagation for Video Object Segmentation (2022)                       | feature decoupling 관점 (시각 정보와 ID 정보를 명시적으로 분리하여 처리)                                                                                                                                                  |
| Semi-Supervised VOS | Learning to Segment Moving Objects in Videos (2014)                                                        | motion boundary 기반 per-frame proposal + learned moving objectness detector + dense trajectory 위 temporal diffusion 로 구성된 hybrid 접근이 video segmentation 과 tracking 의 중간 지점 연구                            |

**2.4.2 Unsupervised VOS**

| 분류             | 논문명                                                                                                     | 분류 근거                                                                                                                                                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Unsupervised VOS | A Generative Appearance Model for End-to-end Video Object Segmentation (2018)                              | deep feature distribution 을 class-conditional Gaussian mixture 로 모델링하고 end-to-end differentiable 으로 학습/추론하는 differentiable probabilistic VOS 방법                                                                               |
| Unsupervised VOS | Unsupervised Semantic Segmentation by Contrasting Object Mask Proposals (2021)                             | 라벨 없는 이미지 데이터셋에서 semantic segmentation 을 위해 saliency 기반 object mask prior 와 pixel-level contrastive learning 을 결합한 2 단계 프레임워크를 제안하여 image-level 대비 pixel-level object-aware objective 가 더 적절함을 입증 |
| Unsupervised VOS | Unsupervised Semantic Segmentation by Distilling Feature Correspondences (2022)                            | 강력한 self-supervised backbone 이 가진 dense correspondence structure 를 segmentation head 로 distill 하여 compact cluster 형성하는 correspondence-based approach                                                                             |
| Unsupervised VOS | CNN in MRF: Video Object Segmentation via Inference in A CNN-Based Higher-Order Spatio-Temporal MRF (2018) | CNN 을 MRF 의 higher-order spatial potential 로 통합한 structured optimization 접근, temporal fusion 과 mask refinement 의 반복적 추론 과정                                                                                                    |

**2.4.3 VOS Architecture/Transformer**

| 분류            | 논문명                                                                                             | 분류 근거                                                                                                                                                                                          |
| --------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| VOS Transformer | Video Object Segmentation using Space-Time Memory Networks (2019)                                  | 과거 시공간 정보를 memory 로 저장하고 현재 query pixel 이 모든 memory 위치와 attention-like dense matching 을 수행하는 시공간 기억 네트워크                                                        |
| VOS Transformer | Video Object Segmentation with Joint Re-identification and Attention-Aware Mask Propagation (2018) | Re-ID 와 temporal propagation 을 joint learning 으로 통합하며 iterative template expansion 을 통해 appearance variation 에 강한 객체 재탐색 전략을 제시한 multi-instance video segmentation 접근법 |
| VOS Transformer | Dual Convolutional LSTM Network for Referring Image Segmentation (2020)                            | Referring Image Segmentation 문제를 해결하기 위해 언어의 시퀀셜 순서성과 이미지의 공간 구조를 동시에 보존하는 ConvLSTM 기반 encoder-decoder 프레임워크를 설계                                      |
| VOS Transformer | A Transductive Approach for Video Object Segmentation (2020)                                       | transductive inference 관점에서 label propagation 기반 semi-supervised VOS 방법이며, external modules 없이 long-term temporal dependency 를 효율적으로 활용한 simple baseline                      |

#### 2.5 Domain Adaptation

**2.5.1 Synthetic-to-Real Adaptation**

| 분류              | 논문명                                                                                                                    | 분류 근거                                                                                                                                                                                                  |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Domain Adaptation | ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation (2018)                            | segmentation output 의 structured spatial layout 을 domain adaptation 핵심 신호로 활용하는 structured output space alignment 기반 unsupervised domain adaptation 연구                                      |
| Domain Adaptation | Learning to Adapt Structured Output Space for Semantic Segmentation (2018)                                                | segmentation output 의 structured spatial layout 을 domain adaptation 핵심 신호로 활용하는 structured output space alignment 기반 unsupervised domain adaptation 연구                                      |
| Domain Adaptation | Adversarial Learning and Self-Teaching Techniques for Domain Adaptation in Semantic Segmentation (2020)                   | pixel-level adversarial discriminator 를 confidence estimator 로 재해석하고 soft weighting plus region growing 으로 self-teaching 품질 향상하는 synthetic-to-real semantic segmentation adaptation 방법    |
| Domain Adaptation | Coarse-to-Fine Domain Adaptive Semantic Segmentation with Photometric Alignment and Category-Center Regularization (2021) | 이미지 레벨의 photometric mismatch 와 카테고리 레벨의 feature distribution mismatch 를 분리해 순차적 coarse-to-fine 방식으로 해결하는 multi-stage adaptation framework                                     |
| Domain Adaptation | Effective Use of Synthetic Data for Urban Scene Semantic Segmentation (2018)                                              | domain shift 를 클래스 유형 (foreground/background) 에 따라 다르게 모델링하여 foreground 는 detection-based segmentation, background 는 pixel-wise segmentation 으로 분리 처리하는 hybrid 접근             |
| Domain Adaptation | Constructing Self-motivated Pyramid Curriculums for Cross-Domain Semantic Segmentation: A Non-Adversarial Approach (2019) | target domain 의 multi-scale supervised signal 을 pyramid 구조로 통합하며, self-generated pseudo label 기반 cross-domain 일반화를 discriminator 없이 달성하는 self-training 기반 domain adaptation 방법론  |
| Domain Adaptation | Context-Aware Mixup for Domain Adaptive Semantic Segmentation (2021)                                                      | 문맥적 prior knowledge 를 활용한 명시적 image/mask mixing 으로 UDA segmentation 성능을 개선하는 방법론                                                                                                     |
| Domain Adaptation | Bidirectional Learning for Domain Adaptation of Semantic Segmentation (2019)                                              | translation model 과 segmentation model 이 closed-loop 구조에서 상호 피드백을 주고받으며, pseudo-label filtering 과 semantic consistency 제약으로 domain gap 을 줄이는 unsupervised domain adaptation 방법 |

**2.5.2 Source-Free/Weak Supervision**

| 분류                   | 논문명                                                                                                                    | 분류 근거                                                                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Source-Free Adaptation | Source-Relaxed Domain Adaptation for Image Segmentation (2020)                                                            | source domain data 없이 adaptation 단계를 수행하는 설정과 class-ratio prior 를 통한 trivial solution 방지 전략이 핵심                                                  |
| Domain Adaptation      | A Curriculum Domain Adaptation Approach to the Semantic Segmentation of Urban Scenes (2018)                               | target 도메인에서 정답 라벨 없이 global distribution 과 landmark superpixel 등 구조적 속성 추정하여 network 예측을 regularize 하는 unsupervised adaptation 접근        |
| Domain Adaptation      | Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes (2017)                                             | target output 이 만족해야 할 structure 적 속성을 먼저 추정하고 output-space regularization 으로 활용하는 접근                                                          |
| Domain Adaptation      | Coarse-to-Fine Domain Adaptive Semantic Segmentation with Photometric Alignment and Category-Center Regularization (2021) | 이미지 레벨의 photometric mismatch 와 카테고리 레벨의 feature distribution mismatch 를 분리해 순차적 coarse-to-fine 방식으로 해결하는 multi-stage adaptation framework |

#### 2.6 Semi-Supervised Semantic Segmentation

**2.6.1 Contrastive Learning**

| 분류                | 논문명                                                                                                         | 분류 근거                                                                                                                                                                                                 |
| ------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Semi-Supervised SSL | Semi-supervised Contrastive Learning for Label-efficient Medical Image Segmentation (2021)                     | segmentation 문제 특성에 맞춰 전역적 이미지 의미와 국소적 픽셀 차원을 분리 학습하고, 제한된 라벨을 초기 학습 단계에 적극 활용하는 구조                                                                    |
| Semi-Supervised SSL | Bootstrapping Semantic Segmentation with Regional Contrast (2021)                                              | Label scarcity 문제 해결을 위한 segmentation 특성에 맞춘 pixel-level contrastive learning 보조 학습 loss 제안                                                                                             |
| Semi-Supervised SSL | Semi-supervised Medical Image Segmentation with Anatomical-aware Contrastive Distillation (2022)               | 의료 영상 분할의 class imbalance 과 false negative 문제를 해결하기 위해 해부학적 구조를 prototype 기반 유사도로 간접 반영하는 3 단계 teacher-student distillation framework                               |
| Semi-Supervised SSL | Bootstrapping Semi-supervised Medical Image Segmentation with Anatomical-aware Contrastive Distillation (2022) | 의료 영상 분할의 class imbalance 과 false negative 문제를 해결하기 위해 해부학적 구조를 prototype 기반 유사도로 간접 반영하는 3 단계 teacher-student distillation framework                               |
| Semi-Supervised SSL | Rethinking Semi-Supervised Medical Image Segmentation: A Variance-Reduction Perspective (2023)                 | pixel-level contrastive loss 의 gradient estimator 분산을 줄이는 stratified group sampling 기법을 통해 label-efficient 한 medical segmentation 을 달성하는 semi-supervised 학습 프레임워크                |
| Semi-Supervised SSL | Semi-supervised Left Atrium Segmentation with Mutual Consistency Training (2021)                               | prediction discrepancy 를 효율적인 uncertainty 근사로 활용하며 cycled pseudo label 을 통한 상호 감독 구조를 특징으로 하는 uncertainty-aware semi-supervised segmentation 방법                             |
| Semi-Supervised SSL | Semi-supervised Medical Image Segmentation through Dual-task Consistency (2021)                                | perturbation 기반 data-level consistency 가 아닌 task discrepancy 기반 regularization 을 제시하며, shared encoder 에서 segmentation map 과 level set map 두 표현을 동시에 학습하고 일관성을 강제하는 구조 |
| Semi-Supervised SSL | ACTION++: Improving Semi-supervised Medical Image Segmentation with Adaptive Anatomical Contrast (2023)        | labeled data 내 class imbalance 를 해결하기 위해 off-line class center 배치와 dynamic temperature scheduling 을 결합한 supervised contrastive learning 개선법                                             |

**2.6.2 Pseudo-Label/Consistency Regularization**

| 분류         | 논문명                                                                            | 분류 근거                                                                                                                                                                     |
| ------------ | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pseudo-Label | A Closer Look at Self-training for Zero-Label Semantic Segmentation (2021)        | augmentation 간 예측 일관성을 통한 consistency constraint 를 도입한 iterative self-training 프레임워크를 제시, 학습 이미지 내 unlabeled 픽셀을 직접 활용하는 데이터 중심 접근 |
| Pseudo-Label | Improving Semantic Segmentation via Video Propagation and Label Relaxation (2018) | video prediction 기반 motion vector 를 image-label 쌍에 공동 전파하고, 경계 라벨링을 relaxation 하여 noisy supervision 에 강한 학습 전략                                      |
| Pseudo-Label | A Transductive Approach for Video Object Segmentation (2020)                      | transductive inference 관점에서 label propagation 기반 semi-supervised VOS 방법이며, external modules 없이 long-term temporal dependency 를 효율적으로 활용한 simple baseline |

#### 2.7 Multi-Task/Multi-Modal Segmentation

**2.7.1 Multi-Task Learning**

| 분류                | 논문명                                                                                | 분류 근거                                                                                                                                                                                                           |
| ------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Multi-Task Learning | DDANet: Dual Decoder Attention Network for Automatic Polyp Segmentation (2020)        | single encoder 를 공유하는 segmentation decoder 와 reconstruction decoder 가 병렬 동작하며, reconstruction branch 에서 생성한 attention map 이 segmentation branch 의 feature 를 강화하는 multi-task attention 구조 |
| Multi-Task Learning | Learning Rich Features from RGB-D Images for Object Detection and Segmentation (2014) | 제한된 학습 데이터 환경에서 depth representation design 에 초점을 맞춘 연구로, 기하학적 장면 priors(HHA) 를 통해 CNN 학습 효과를 극대화한 다태스크 unified framework                                                |

**2.7.2 Multi-Modal/3D Segmentation**

| 분류        | 논문명                                                                                                           | 분류 근거                                                                                                                                              |
| ----------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Multi-Modal | Robust Semantic Segmentation of Brain Tumor Regions from 3D MRIs (2020)                                          | 멀티모달 3D MRI 분할을 위해 ResNet encoder-decoder 구조와 Dice+focal+active contour hybrid loss 조합으로 성능을 검증한 실용적 공학지향 논문            |
| Multi-Modal | Contrastive learning of global and local features for medical image segmentation with limited annotations (2020) | segmentation 과 같은 dense prediction task 에 적합한 pixel/local region 수준의 local representation 학습을 위해 contrastive objective 를 재설계한 연구 |

#### 2.8 Architecture Design

**2.8.1 Encoder/Decoder Architecture**

| 분류            | 논문명                                                                                                                 | 분류 근거                                                                                                                                                                                                                                            |
| --------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Encoder-Decoder | U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)                                                 | skip connection 을 통한 multi-scale feature 융합 구조와 데이터 효율적 학습 전략이 결합된 low-data biomedical segmentation 의 표준 참고 모델                                                                                                          |
| Encoder-Decoder | SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation (2015)                                | SegNet 은 pooling indices 기반 upsampling 으로 경계 정보 보존과 메모리 효율성을 동시에 확보한 lightweight encoder-decoder segmentation 아키텍처로, decoder 설계별 trade-off 분석을 통해 실용적 선택지로 제안됨                                       |
| Encoder-Decoder | DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs (2016) | Classification 기반 DCNN 을 Dense predictor 로 재구성하여 segmentation 특화 문제를 해결한 구조적 접근                                                                                                                                                |
| Encoder-Decoder | DeepLabv3+ (2018) - Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (2018)           | DeepLabv3 기반 encoder 구조에 경계 복원 특화 decoder 를 결합한 ASPP multi-scale 정보 활용 방식                                                                                                                                                       |
| Encoder-Decoder | SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers (2021)                              | SegFormer 는 positional encoding 없는 hierarchical Transformer encoder 와 lightweight All-MLP decoder 조합으로 단순화만으로 정확도·효율성·강건성 동시 개선                                                                                           |
| Encoder-Decoder | EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction (2024)                                 | efficientViT 는 ReLU linear attention 과 convolution 의 하이브리드 구조를 통해 dense prediction 의 global 와 local 요구사항을 균형적으로 충족시킴                                                                                                    |
| Encoder-Decoder | SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation (2022)                                    | semantic segmentation 전용으로 설계된 convolution 기반 attention 메커니즘으로, transformer 기반 접근법 대비 multi-scale spatial context 파악 능력과 high-resolution 환경에서의 계산 효율성을 동시에 확보하며 task-specific design principles 을 입증 |
| Encoder-Decoder | Efficient piecewise training of deep structured models for semantic segmentation (2015)                                | CNN 기반 general pairwise potential 로 patch 간 semantic relation 을 명시적 잠재변수로 모델링하고 piecewise likelihood 분해로 structured model 학습 비용을 효율화한 semantic segmentation 방법                                                       |
| Encoder-Decoder | Deep Extreme Cut: From Extreme Points to Object Segmentation (2017)                                                    | extreme points 를 Gaussian heatmap 으로 변환해 CNN 입력의 4 번째 채널로 사용하는 입력 효율형 instance segmentation 및 annotation 도구                                                                                                                |

**2.8.2 Attention Mechanism**

| 분류      | 논문명                                                                                       | 분류 근거                                                                                                                                                                              |
| --------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Attention | CCNet: Criss-Cross Attention for Semantic Segmentation (2018)                                | sparse attention 을 recurrent 하게 반복하여 full-image context 를 근사하는 attention 구조, category consistent loss 로 feature discrimination 강화                                     |
| Attention | Attention to Scale: Scale-aware Semantic Image Segmentation (2015)                           | scale dimension 에 대한 adaptive attention 과 multi-scale output 에 대한 독립적인 supervision 을 결합하여 end-to-end 학습 가능한 융합 구조                                             |
| Attention | Dual Cross-Attention for Medical Image Segmentation (2023)                                   | 경량 cross-attention 모듈을 skip connection 에 삽입하여 multi-scale encoder feature 의 semantic gap 을 줄이는 plug-in 접근                                                             |
| Attention | Active Boundary Loss for Semantic Segmentation (2021)                                        | 예측 경계를 동적으로 이동시키는 boundary-aligned loss 로서, 얇은 구조물과 복잡한 경계 분할에 특화된 모델-agnostic regularization 기법                                                  |
| Attention | Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network (2017) | 이 논문은 semantic segmentation 을 classification 과 localization 의 긴장 관계로 재해석하며, 이를 해결하기 위해 separable large kernel 기반 구조와 boundary refinement 블록을 제안한다 |

**2.8.3 Regularization/Loss Design**

| 분류           | 논문명                                                                                           | 분류 근거                                                                                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Regularization | Active Boundary Loss for Semantic Segmentation (2021)                                            | 예측 경계를 동적으로 이동시키는 boundary-aligned loss 로서, 얇은 구조물과 복잡한 경계 분할에 특화된 모델-agnostic regularization 기법                           |
| Regularization | Convolutional CRFs for Semantic Segmentation (2018)                                              | locality assumption 을 통해 exact message passing 을 convolution 연산으로 변환하는 시스템적 재구성을 통해 CRF 의 추상적 모델을 실용적 deep learning 모듈로 전환 |
| Regularization | Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation (2021) | BAP 와 NAL 로 pseudo label 품질과 noise robustness 를 독립적으로 최적화하는 구조를 가지며, classification representation 개선과 segmentation 성능 향상을 연결   |

**2.8.4 Multi-Scale/Context Modeling**

| 분류        | 논문명                                                                                                                 | 분류 근거                                                                                                                                                                          |
| ----------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Multi-Scale | Context Encoding for Semantic Segmentation (2018)                                                                      | 전역 문맥을 feature statistics 요약으로 명시화하고 channel scaling 으로 반영하는 보완 모듈로, 기존 FCN 파이프라인에 추가 비용 없이 성능 개선을 달성                                |
| Multi-Scale | Context Prior for Scene Segmentation (2020)                                                                            | semantic segmentation 에서 문맥 정보를 semantic class 기준으로 명시적 supervised learning 으로 분리하고, 이를 feature aggregation 에 직접 적용하는 구조화 관계를 학습하는 접근법   |
| Multi-Scale | Building Segmentation through a Gated Graph Convolutional Neural Network with Deep Structured Feature Embedding (2019) | CNN feature embedding 과 graph message passing 을 결합한 hybrid architecture 로, 경계 정밀도와 픽셀 상호작용을 동시에 개선하는 구조                                                |
| Multi-Scale | Hypercolumns for Object Segmentation and Fine-grained Localization (2014)                                              | multi-layer feature representation 과 location-specific classification 을 결합한 접근이 object segmentation, keypoint prediction, part labeling 등 세밀한 위치 예측 task 에 효과적 |

#### 2.9 Instance Segmentation

**2.9.1 Instance Segmentation Framework**

| 분류                  | 논문명                                                                              | 분류 근거                                                                                                                                                                                                                    |
| --------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Instance Segmentation | Instance-aware Semantic Segmentation via Multi-task Network Cascades (2015)         | 문제 분해를 통해 box-mask-category 의 multi-stage cascade 로 분해하고, shared feature 기반 causal 의존성을 고려한 end-to-end 학습을 실현하는 multi-task network 아키텍처로 분류                                              |
| Instance Segmentation | Fully Convolutional Instance-aware Semantic Segmentation (2016)                     | instance-aware segmentation 과 semantic segmentation 을 동시에 수행하는 최초의 end-to-end fully convolutional 방법이며, detection 과 segmentation 을 shared score maps 로 공동 최적화하는 joint formulation 이 핵심 차별점   |
| Instance Segmentation | Instance-sensitive Fully Convolutional Networks (2016)                              | 상대적 위치 표현을 활용한 instance-sensitive score maps 와 parameter-free assembling 모듈을 통해, 고차원 mask regression 없이 fully convolutional 구조로 instance mask 를 생성. local coherence 기반 설계로 과적합 위험 감소 |
| Instance Segmentation | Iterative Instance Segmentation (2015)                                              | 명시적 구조 제약 없이 반복적 보정 과정으로 shape/contiguity/smoothness prior 를 암묵 학습                                                                                                                                    |
| Instance Segmentation | Pixelwise Instance Segmentation with a Dynamically Instantiated Network (2017)      | instance segmentation 을 전역적 픽셀 labeling 문제로 재구성하고, 이미지마다 가변적인 label 수를 갖는 dynamic network 를 end-to-end 학습 가능하도록 설계한다                                                                  |
| Instance Segmentation | Pixel-level Encoding and Depth Layering for Instance-level Semantic Labeling (2016) | instance-level representation 을 위해 semantic/depth/direction 3 신호를 FCN 으로 예측한 뒤 template matching 으로 instance 를 복원하는 pixel encoding 기반 접근                                                              |
| Instance Segmentation | Amodal Instance Segmentation (2016)                                                 | amodal annotation 부재 환경에서 synthetic occlusion 으로 supervised 학습을 가능하게 하고, modal prediction 을 조건으로 점진적 amodal 확장을 통해 occlusion robustness 를 학습하는 첫 번째 일반 목적 방법                     |

**2.9.2 Unseen Object Instance**

| 분류            | 논문명                                                                       | 분류 근거                                                                                                                                                              |
| --------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Unseen Instance | Unseen Object Instance Segmentation for Robotic Environments (2020)          | depth 기반 object seeding 과 RGB 기반 boundary refinement 를 모듈화한 2-stage 구조로 domain gap 문제를 우회하는 로봇 perception 접근법                                 |
| Unseen Instance | ShapeMask: Learning to Segment Novel Objects by Refining Shape Priors (2019) | shape prior 와 instance embedding 을 결합한 다단계 refinement 구조를 통해 detection box 를 점진적으로 mask 로 정제하는 partially supervised instance segmentation 방법 |
| Unseen Instance | Semantic Amodal Segmentation (2015)                                          | 보이지 않는 영역까지 포함한 dense annotation 과 occlusion 관계를 명시한 pairwise depth ordering 을 통한 장면 구조 이해 연구                                            |

#### 2.10 Survey/Review Papers

**2.10.1 Comprehensive Surveys**

| 분류         | 논문명                                                                                      | 분류 근거                                                                                                                                                                                                                        |
| ------------ | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Survey Paper | A Review on Deep Learning Techniques Applied to Semantic Segmentation (2017)                | deep learning 기반 semantic segmentation 의 방법론 계보, dataset/benchmark, 성능 비교를 한 프레임으로 체계적으로 정리한 초기 survey paper                                                                                        |
| Survey Paper | A Survey on Deep Learning Technique for Video Segmentation (2021)                           | 이 논문은 비디오 분할을 문제 정의 (VOS/VSS), 인간 개입 (AVOS/SVOS/IVOS), 학습 패러다임 (supervised/unsupervised) 관점에서 체계적으로 분류하며, 각 subspecies 와 주요 계열을 명확히 구분한 통합 taxonomy 를 제공한다              |
| Survey Paper | A Survey on Deep Learning-based Architectures for Semantic Segmentation on 2D images (2019) | 2D 이미지 semantic segmentation 분야 전체를 pre-deep learning/FCN/post-FCN 세 시기로 기술적 병목 해결 흐름으로 재구성한 survey                                                                                                   |
| Survey Paper | A Survey on Open-Vocabulary Detection and Segmentation: Past, Present, and Future (2023)    | weak supervision 활용 여부와 활용 방식 (visual-semantic mapping vs. feature synthesis vs. VLM distillation) 에 따라 zero-shot 과 open-vocabulary 를 통합 분류; detection/segmentation/3D/video 에 일관된 방법론별 분류 체계 제안 |
| Survey Paper | A Survey on Deep Learning Technique for Video Segmentation (2021)                           | 이 논문은 비디오 분할을 문제 정의 (VOS/VSS), 인간 개입 (AVOS/SVOS/IVOS), 학습 패러다임 (supervised/unsupervised) 관점에서 체계적으로 분류하며, 각 subspecies 와 주요 계열을 명확히 구분한 통합 taxonomy 를 제공한다              |

### 3. 종합 정리

본 연구 분류 체계는 semantic segmentation 분야를 연구 목적 (benchmark 설계), 학습 패러다임 (weakly-supervised/semi-supervised/unsupervised), 적용 도메인 (medical/robotic/general), 방법론적 관점 (architecture design/loss design/multi-task) 의 4 차원 축으로 구조화하였다. weakly-supervised 연구는 bounding box 기반 pseudo-label 에서 zero-shot 일반화까지 발전 흐름을, video object segmentation 은 embedding matching 과 temporal modeling 관점에서, domain adaptation 은 synthetic-to-real 전이와 source-free 설정으로, architecture 설계는 convolution 기반 attention 과 transformer encoder 의 병행 발전으로 구분 가능하다. 전반적으로 데이터 효율성 향상 (few-shot/zero-shot), 일반화 성능 개선 (domain adaptation), computational efficiency (efficientViT/SegNeXt) 가 주요 트렌드로 나타난다.

## 2장. 방법론 분석

## 1. 공통 문제 설정 및 접근 구조

## 1.1 기본 문제 정의

모든 Semantic Segmentation 논문의 공통 문제는 **입력 이미지에서 각 픽셀에 semantic class label을 할당하는 dense prediction 문제**로 정의된다.

| 요소          | 설명                                            |
| ------------- | ----------------------------------------------- |
| **입력**      | 2D/3D/RGB-D/Video 이미지, sometimes stereo pair |
| **출력**      | 픽셀 단위 클래스 라벨 맵                        |
| **문제 공간** | 입력 $X \to$ 출력 $Y: X \to Y$                  |

## 1.2 방법론적 구조 일반화

문헌 분석에 기반하여 방법론들은 다음 구조적 패턴으로 분류된다:

```text
┌─────────────────────────────────────────────────────────────────┐
│                    METHODOLGY STRUCTURE                         │
├─────────────────────────────────────────────────────────────────┤
│  INPUT → FEATURE EXTRACTION → CONTEXT AGGREGATION →             │
│          PREDICTION HEAD → OUTPUT (Pixel-wise Label Map)        │
└─────────────────────────────────────────────────────────────────┘
```

## 1.3 공통 구성 요소

| 구성 요소           | 역할                                           | 대표 논문               |
| ------------------- | ---------------------------------------------- | ----------------------- |
| **Encoder**         | Feature extraction, multi-scale representation | FCN, VGG, ResNet, ViT   |
| **Decoder**         | Upsampling, spatial resolution restoration     | U-Net, DeepLab, Decoder |
| **Skip Connection** | Low-level spatial detail 보존                  | FCN-8s, U-Net           |
| **Context Module**  | Global receptive field 확장                    | ASPP, PSP, Transformer  |
| **Post-processing** | Boundary refinement, CRF                       | FullCRF, ConvCRF        |

## 2. 방법론 계열 분류

## 2.1 CNN 기반 Encoder-Decoder 계열

**계열 정의**: Fully convolutional 구조를 기본으로, convolutional backbone과 upsampling decoder의 조합으로 구현.

**공통 특징**:

- Fully convolutional layers 로 end-to-end dense prediction
- Deconvolution/bilinear upsampling 으로 spatial resolution 회복
- Multi-scale feature fusion 으로 multi-object 처리

| 방법론 계열  | 논문명                                                        | 핵심 특징                                         |
| ------------ | ------------------------------------------------------------- | ------------------------------------------------- |
| **FCN 계열** | Fully Convolutional Networks for Semantic Segmentation (2014) | skip connection 기반 multi-scale fusion           |
|              | DeepLab (2016)                                                | atrous convolution, ASPP, FC-RF                   |
|              | SegNet (2015)                                                 | pooling index 기반 max unpooling                  |
|              | U-Net (2015)                                                  | contracting/expansive path, skip connection       |
|              | DeepLabv3+ (2018)                                             | dilated conv + ASPP + decoder                     |
|              | FastFCN (2019)                                                | joint upsampling, learnable feature approximation |

**적용 대상**: Standard 2D/3D semantic segmentation, urban/surgical/medical imagery

## 2.2 Transformer 기반 계열

**계열 정의**: Self-attention 또는 cross-attention 메커니즘으로 long-range dependency 모델링, ViT 기반 구조 사용.

**공통 특징**:

- Patch embedding 으로 2D 이미지를 시퀀스로 변환
- Multi-head self-attention 으로 전역 context 통합
- Decoder는 simple linear/mask projection 으로 mask 생성

| 방법론 계열                | 논문명                         | 핵심 특징                                       |
| -------------------------- | ------------------------------ | ----------------------------------------------- |
| **Pure Transformer**       | Segmenter (2021)               | pure ViT encoder, linear/mask decoder           |
| **Hybrid CNN-Transformer** | SegFormer (2021)               | hierarchical MiT encoder, All-MLP decoder       |
|                            | CAT-CAT/CASTformer (2022-2023) | class-aware sampling, adversarial discriminator |
| **Attention Fusion**       | CLUSTSEG (2023)                | cross-attention 기반 EM-style clustering        |
|                            | CAT-Seg (2023)                 | cost volume, spatial/class aggregation          |

**적용 대상**: High-resolution dense prediction, multi-scale feature fusion 필요 작업

## 2.3 Few-Shot/Zero-Shot/One-Shot 계열

**계열 정의**: Support data 가 극히 제한된 상황에서 unseen class 에 대해 segmentation 수행.

**공통 특징**:

- Prototype/representation learning 기반
- Class-agnostic feature learning
- Attention/prompt 기반 adaptive learning

| 방법론 계열                   | 논문명                              | 핵심 특징                                  |
| ----------------------------- | ----------------------------------- | ------------------------------------------ |
| **Prototype Matching**        | PANet (2019)                        | support prototype 추출, metric learning    |
|                               | Location-Sensitive Prototype (2021) | overlapping grid, local prototype matching |
|                               | CANet (2019)                        | dense comparison, iterative refinement     |
| **Class-Agnostic Generation** | CANet (2019)                        | class-agnostic, ASPP, iterative refinement |
|                               | Zero-Shot Segmentation (2019)       | GMMN 기반 synthetic feature generation     |
| **Prompt Learning**           | FreeSeg (2023)                      | adaptive prompt, test-time adaptation      |
|                               | CAT-Seg (2023)                      | cost volume, spatial/class aggregation     |

**적용 대상**: Novel class detection, open-vocabulary segmentation

## 2.4 Domain Adaptation 계열

**정의**: Source domain 에서 학습한 모델을 target 도메인으로 transfer, target 에 labeled data 가 없는 UDA 설정.

**공통 특징**:

- Adversarial alignment
- Pseudo-labeling 기반 self-supervision
- Multi-level adaptation

| 방법론 계열                  | 논문명                           | 핵심 특징                                |
| ---------------------------- | -------------------------------- | ---------------------------------------- |
| **Entropy Minimization**     | ADVENT (2018)                    | direct/adv entropy, class prior          |
| **Adversarial Alignment**    | DA (2019), Classes Matter (2020) | fine-grained discriminator, class-aware  |
| **Curriculum Adaptation**    | CDA (2018), PyCDA (2019)         | global/region/pixel-level coarse-to-fine |
| **Bidirectional Adaptation** | Bidir Learning (2019)            | translation ↔ segmentation feedback loop |
| **Context-Aware Mixup**      | CAM (2022)                       | input/output/significance mask mixing    |

**적용 대상**: Synthetic-to-real transfer, cross-domain adaptation

## 2.5 Video Object Segmentation 계열

**계열 정의**: 첫 프레임 mask 만 주어지고 이후 프레임에서 같은 객체를 추적·분할하는 temporal propagation 기반 문제.

**공통 특징**:

- Optical flow 기반 temporal propagation
- Memory-based correspondence
- Instance-level identity tracking

| 방법론 계열            | 논문명                     | 핵심 특징                                |
| ---------------------- | -------------------------- | ---------------------------------------- |
| **Memory-Based**       | STM (2019)                 | space-time memory, dense matching        |
| **Propagation**        | Anchor Diffusion (2019)    | anchor-to-current correspondence         |
| **Feature Decoupling** | Decoupling Features (2022) | visual/ID branch 분리                    |
| **Transformer-based**  | AOT (2021)                 | identity embedding, transformer decoder  |
| **Collaborative**      | Collaborative VOS (2020)   | F-B integration, pixel/instance matching |

**적용 대상**: Semi-supervised VOS, temporal consistency required 작업

## 2.6 Weakly-Supervised Segmentation 계열

**계열 정의**: Pixel-level annotation 대신 image-level/bounding box만 이용한 segmentation 학습.

**공통 특징**:

- Class Activation Map (CAM) 기반 pseudo label 생성
- Seed region growing
- Retrieval-based label 생성

| 방법론 계열        | 논문명               | 핵심 특징                                  |
| ------------------ | -------------------- | ------------------------------------------ |
| **CAM-based**      | WSSS Analysis (2019) | SEC, DSRG, IRNet, HistoSegNet              |
| **Region-Based**   | BAP/NAL (2021)       | background-aware pooling, noise-aware loss |
| **Detector-Based** | BBAM (2021)          | detector behavior 기반 pseudo mask         |
| **Box-Based**      | BoxSup (2015)        | region proposal, alternating optimization  |

**적용 대상**: Annotation 비용 절감, weak supervision 환경

## 2.7 Medical Image Segmentation 계열

**계uel 정의**: 3D volumetric 데이터, severe class imbalance, sparse annotation 환경에서의 segmentation.

**공통 특징**:

- Volumetric context modeling
- Class imbalance aware loss
- Anatomical prior 활용

| 방법론 계열                  | 논문명                                | 핵심 특징                                       |
| ---------------------------- | ------------------------------------- | ----------------------------------------------- |
| **Contrastive Distillation** | AnCo (2022), Bootstrapping SSL (2022) | global/local contrast, anatomical prior         |
| **Anomaly Detection**        | Anomaly Detection (2022)              | foreground prototype, anomaly scoring           |
| **Variance Reduction**       | Variance-Reduction (2023)             | stratified sampling, gradient variance          |
| **Adversarial**              | CASTformer (2022)                     | multi-scale pyramid, adversarial discrimination |
| **Few-Shot Medical**         | GFS-Seg (2020)                        | base/novel class 일반화, CAPL                   |

**적용 대상**: 3D medical imaging, class imbalance 환경

## 2.8 Structured Prediction 계열

**계열 정의**: CNN 예측 결과에 structured prior 를 적용한 post-processing 또는 learning-to-predict.

**공통 특징**:

- Mean-field approximation
- Convolutional CRF
- Energy minimization

| 방법론 계열         | 논문명                   | 핵심 특징                               |
| ------------------- | ------------------------ | --------------------------------------- |
| **Deep MRF**        | Deep Learning MRF (2016) | CNN unary, 3D local convolution         |
| **ConvCRF**         | ConvCRF (2018)           | convolutional message passing, locality |
| **CRF Integration** | FCN + CRF, DeepLab       | post-processing, boundary refinement    |

**적용 대상**: Boundary precision 향상, structured output 모델링

## 3. 핵심 설계 패턴 분석

## 3.1 Multi-Scale Feature Fusion

**패턴 설명**: 여러 수준의 feature 를 통합하여 multi-scale object 처리.

| 패턴                    | 구현 방법                              | 적용 논문                  |
| ----------------------- | -------------------------------------- | -------------------------- |
| Skip Connection         | Encoder feature decoder 에 concatenate | FCN, U-Net, DeepLab        |
| Atrous Convolution      | Receptive field 확장 없이 해상도 유지  | DeepLab, FastFCN           |
| Spatial Pyramid Pooling | Multi-level pooled feature aggregation | PSPNet, DeepLab            |
| Feature Decoupling      | Visual/ID feature 분리                 | Decoupling Features (2022) |

## 3.2 Prototype/Representation Learning

**패턴 설명**: Prototype 기반 metric learning 또는 representation distillation.

| 패턴                   | 구현 방법                           | 적용 논문                                   |
| ---------------------- | ----------------------------------- | ------------------------------------------- |
| Global Prototype       | Average pooling foreground feature  | PANet, CANet                                |
| Local Prototype        | Overlapping grid/location-sensitive | Loc-Sensitive Prototype (2021)              |
| Class-Agnostic Feature | Support/query feature alignment     | CANet, CAPL                                 |
| Mask Classification    | Segment-level classification        | MaskFormer, Per-Pixel Classification (2021) |

## 3.3 Context Encoding

**패턴 설명**: Global context 정보를 명시적으로 인코딩.

| 패턴                         | 구현 방법                      | 적용 논문                       |
| ---------------------------- | ------------------------------ | ------------------------------- |
| Featuremap Attention         | Channel-wise scaling           | EncNet                          |
| Context Prior Layer          | Ideal affinity map supervision | Context Prior (2020)            |
| Deformable Attention         | Adaptive receptive field       | InternImage, Panoptic SegFormer |
| Multi-Scale Linear Attention | ReLU-based linear attention    | EfficientViT                    |

## 3.4 Contrastive Learning

**패턴 설명**: Instance discrimination 또는 class-aware contrastive objective.

| 패턴                | 구현 방법                          | 적용 논문                          |
| ------------------- | ---------------------------------- | ---------------------------------- |
| Global Contrast     | Image-level feature discrimination | Contrastive Medical (2020)         |
| Local Contrast      | Class-aware pixel embedding        | Bootstrapping SSL (2022)           |
| Anatomical Contrast | Anatomical prior 기반 contrastive  | AnCo, Bootstrapping Medical (2022) |
| Relational Contrast | Instance/label discrimination      | Rethinking SSL (2023)              |

## 3.5 Prompt/Adapter Fine-tuning

**패턴 설명**: Foundation model 의 부분 파라미터만 학습.

| 패턴                  | 구현 방법                           | 적용 논문        |
| --------------------- | ----------------------------------- | ---------------- |
| Frozen Vision Encoder | CLIP frozen, only segmentation head | CAT-Seg, CAT-CAT |
| Learnable Prompt      | Entropy-based test-time tuning      | FreeSeg          |
| Adapter Fine-tuning   | Adapter layers only                 | CATformer        |

## 4. 방법론 비교 분석

## 4.1 문제 접근 방식 차이

| 구분                   | CNN 계열                       | Transformer 계열       | Structured Prediction |
| ---------------------- | ------------------------------ | ---------------------- | --------------------- |
| **Context Modeling**   | Local convolution, skip fusion | Self-attention, global | Energy minimization   |
| **Learning Paradigm**  | Supervised/UDA                 | Supervised/Zero-shot   | Structured prediction |
| **Boundary Precision** | CRF refinement                 | Implicit via attention | Explicit structured   |

## 4.2 구조/모델 차이

### 4.2.1 계산 복잡도

| 방식              | Complexity       | Memory |
| ----------------- | ---------------- | ------ |
| Convolution       | $O(N \cdot K^2)$ | Low    |
| Self-Attention    | $O(N^2 \cdot d)$ | High   |
| Linear Attention  | $O(N \cdot d)$   | Medium |
| Decoupled Feature | $O(N \cdot d)$   | Low    |

### 4.2.2 일반화 능력

| 유형       | Seen Class | Unseen Class | Cross-Dataset |
| ---------- | ---------- | ------------ | ------------- |
| Supervised | High       | Low          | Low           |
| Few-Shot   | Medium     | Medium       | Medium        |
| Zero-Shot  | Medium     | High         | Medium        |
| UDA        | Medium     | Medium       | High          |

## 4.3 적용 대상 차이

### 4.3.1 이미지 유형

| 이미지 유형     | 권장 방법론 계열               |
| --------------- | ------------------------------ |
| Standard 2D     | CNN/Transformer                |
| 3D Volumetric   | Medical 계열 (Contrastive)     |
| Video           | VOS 계열 (Memory/Propagation)  |
| High-Res Dense  | Linear Attention, EfficientViT |
| Open-Vocabulary | Zero-shot/Prompt 계열          |

### 4.3.2 annotation 제약

| Annotation 수준  | 권장 방법론                           |
| ---------------- | ------------------------------------- |
| Full pixel label | Standard supervised (CNN/Transformer) |
| Bounding box     | Weakly-supervised (CAM/Region)        |
| Image-level      | WSSS                                  |
| Unlabeled target | Domain Adaptation                     |
| Unseen class     | Zero-shot/Few-shot                    |

## 4.4 트레이드오프 분석

### 4.4.1 정확도 vs 효율성

| 목표             | 권장 설계                       | 트레이드오프      |
| ---------------- | ------------------------------- | ----------------- |
| Highest accuracy | Full Transformer/Deep CNN       | High computation  |
| Real-time        | Linear attention, small kernels | Moderate accuracy |
| Low memory       | Decoupled features              | Less context      |

### 4.4.2 Generalization vs Specialization

| 목표         | 권장 설계                      | 트레이드오프           |
| ------------ | ------------------------------ | ---------------------- |
| Cross-domain | UDA, Domain-invariant features | Source-specific loss   |
| Novel class  | Zero-shot, Prompt learning     | Seen-class performance |

## 5. 방법론 흐름 및 진화

## 5.1 초기 접근 (2014-2016)

**특징**: FCN 기반 fully convolutional 구조 도입, simple upsampling.

```text
FCN-2014 → SegNet → DeepLab → U-Net
   ↓           ↓           ↓          ↓
dense         max        atrous     skip
prediction    unpool    conv       fusion
```

**주요 발전**:

- Fully connected layer → convolutional 변환
- Bilinear upsampling → Deconvolution
- Deconvolution → Attention

## 5.2 중간 발전 (2017-2019)

**특징**: Context modeling 강화, multi-scale feature aggregation, UDA 도입.

**주요 발전**:

- Atrous convolution → ASPP → PSP
- RNN → LSTM → ConvLSTM (temporal)
- Supervised → Weakly → Few-shot → Zero-shot

## 5.3 최근 경향 (2020-현재)

**특징**: Transformer 기반 방법론 부상, foundation model 활용, open-vocabulary/generalization 강조.

**주요 발전**:

- CNN → ViT → Hybrid (CNN+ViT)
- Supervised → Self-supervised → Contrastive
- Seen-class → Zero-shot → Open-vocabulary
- Pixel-level → Segment-level → Cost aggregation

## 5.4 기술 진화 흐름

```text
2014-2016:   FCN → SegNet → DeepLab
              (Local feature → Multi-scale → Atrous)

2017-2019:   UDA → Weakly → Few-shot
              (Domain shift → Label scarcity → Novel class)

2020-2021:   Transformer → Segmenter → SegFormer
              (Self-attention → Hierarchical → Efficient)

2022-2023:   Zero-shot → Few-shot → Open-vocabulary
              (Seen-class → Unseen → Text-guided)
```

## 6. 종합 정리

## 6.1 방법론 지형 요약

Semantic Segmentation 방법론은 다음 세 가지 축으로 구조화될 수 있다:

| 축                      | 구분               | 대표 방법론                                  |
| ----------------------- | ------------------ | -------------------------------------------- |
| **1. Feature Type**     | Convolutional      | CNN 계열 (FCN, U-Net, DeepLab)               |
|                         | Transformer        | Transformer 계열 (ViT, Segmenter, SegFormer) |
|                         | Hybrid             | CNN-Transformer 혼합 (CAT, EfficientViT)     |
| **2. Supervision Type** | Fully-supervised   | Standard CNN/Transformer                     |
|                         | Weakly-supervised  | WSSS, BoxSup, BBAM                           |
|                         | Few-shot/Zero-shot | CANet, PANet, FreeSeg                        |
| **3. Context Type**     | Local              | Convolution, Deconvolution                   |
|                         | Global             | ASPP, Transformer, Linear Attention          |
|                         | Structured         | CRF, MRF, Energy minimization                |

## 6.2 방법론 분류 체계

```text
┌───────────────────────────────────────────────────────────────────────┐
│                       SEMANTIC SEGMENTATION CLASSIFICATION            │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
│  │ SUPERVISION     │    │ CONTEXT         │    │ FEATURE         │    │
│  │                 │    │                 │    │ TYPE            │    │
│  ├─────────────────┤    ├─────────────────┤    ├─────────────────┤    │
│  │ Fully-super.    │───>│ Local Conv.     │───>│ CNN             │    │
│  │                 │    │                 │    │                 │    │
│  │ Weakly-super.   │    │ Global Context  │    │ Transformer     │    │
│  │                 │    │                 │    │                 │    │
│  │ Few/Zero-shot   │<──>│ Structured      │    │ Hybrid          │    │
│  │                 │    │ (CRF/MRF)       │    │                 │    │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘    │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

## 6.3 결론

제공된 문헌에 기반하여 Semantic Segmentation 방법론은 **(1) Feature representation type**(Convolutional/Transformer/Hybrid)의 3 차원 축으로 분류될 수 있다. 초기 CNN 기반 local feature 모델링에서 Transformer 기반 global attention 모델링으로, supervised 에서 few/zero-shot 으로, local 에서 structured prediction 으로 진화하는 흐름이 관찰된다.

각 계열은 서로 trade-off 관계에 있으며, 작업의 제약 조건 (annotation budget, domain gap, novel class 등) 에 따라 적절한 방법론 선택이 필요하다.

## 3 장. 실험결과 분석

### 1. 평가 구조 및 공통 실험 설정

#### 1.1 주요 데이터셋 유형

| 데이터셋 유형      | 대표 데이터셋               | 사용 빈도 | 특징                                |
| ------------------ | --------------------------- | --------- | ----------------------------------- |
| **Cityscapes**     | Cityscapes (fine 2,975 장)  | 매우 높음 | 19 클래스, urban scenes, real-world |
| **ADE20K**         | ADE20K (150 classes)        | 높음      | 150 클래스, diverse scenes          |
| **PASCAL VOC**     | PASCAL VOC 2012             | 높음      | 20+1 클래스, standard benchmark     |
| **COCO Stuff**     | COCO Stuff (171 classes)    | 높음      | thing/stuff 세분화                  |
| **Medical**        | ACDC, LiTS, BraTS           | 높음      | 3D volumetric, few-label            |
| **Zero-Shot**      | Pascal-Context, ADE20K-Full | 높음      | seen/unseen 분리 평가               |
| **DAVIS**          | DAVIS 2016/2017             | 높음      | video object segmentation           |
| **Synthetic→Real** | GTA5/SYNTHIA → Cityscapes   | 높음      | domain adaptation                   |

#### 1.2 평가 환경 분류

| 평가 환경                 | 데이터셋                            | 주요 용도                      |
| ------------------------- | ----------------------------------- | ------------------------------ |
| **실환경**                | Cityscapes, COCO, ADE20K            | semantic segmentation          |
| **실시간/one-shot**       | DAVIS, YouTube-VOS                  | VOS, one-shot adaptation       |
| **zero-shot/seen-unseen** | Pascal-VOC, ADE20K-847              | open-vocabulary/generalization |
| **cross-domain**          | GTA5→Cityscapes, SYNTHIA→Cityscapes | domain adaptation              |
| **medical/3D**            | BraTS, ACDC, LiTS                   | volumetric segmentation        |
| **robustness**            | Cityscapes-C (16 corruption)        | robustness testing             |

#### 1.3 주요 평가 지표 정리

| 평가 지표                  | 의미                         | 사용 빈도 | 주의점                              |
| -------------------------- | ---------------------------- | --------- | ----------------------------------- |
| **mIoU**                   | mean Intersection over Union | 가장 높음 | per-class 평균, standard metric     |
| **Dice coefficient (DSC)** | Dice Similarity Coefficient  | 높음      | medical segmentation 에서 주로 사용 |
| **J&F**                    | Jaccard + F-measure 평균     | 높음      | VOS 기준, J(region)+F(boundary)     |
| **hIoU**                   | harmonic mean (seen/unseen)  | 높음      | zero-shot evaluation                |
| **AP/AP50/AR**             | Average Precision/Recall     | 높음      | instance segmentation               |
| **PQ/SQ/RQ**               | Panoptic Quality             | 높음      | panoptic segmentation               |
| **95HD/ASD**               | Hausdorff Distance/ASD       | 높음      | medical, boundary accuracy          |
| **F-measure (F)**          | boundary accuracy            | 높음      | contour accuracy                    |

### 2. 주요 실험 결과 정렬

#### 2.1 Semantic Segmentation 성능 비교 (mIoU 기준)

| 논문명              | 데이터셋/환경   | 비교 대상         | 평가 지표 | 핵심 결과                     |
| ------------------- | --------------- | ----------------- | --------- | ----------------------------- |
| DeepLab (2016)      | PASCAL VOC 2012 | VGG16 LargeFOV    | mIoU      | 79.7% (FCN 62.2 대비 +17.5%)  |
| DeepLabv3+ (2018)   | Cityscapes      | PSPNet, DeepLabv2 | mIoU      | 82.1% (ASPP decoder 추가)     |
| CCNet (2020)        | Cityscapes      | PSPNet, PSANet    | mIoU      | 81.9% (RCCA 반복 R=2 선택)    |
| EfficientViT (2022) | Cityscapes      | SegFormer-B5      | mIoU      | 83.0% (linear attention)      |
| SegFormer (2021)    | ADE20K          | SETR, DeepLabV3+  | mIoU      | 51.0% (MiT-B5, single-scale)  |
| CAT-Seg (2024)      | ADE20K (A-847)  | ZegFormer, OVSeg  | mIoU      | 16.0% (unseen-heavy에서 강점) |
| InternImage (2022)  | ADE20K          | BEiT-3, Swin-V2   | mIoU      | 62.9% (large-scale CNN)       |

#### 2.2 Zero-Shot/Unseen 성능 비교 (hIoU 기준)

| 논문명                          | 데이터셋       | 비교 대상            | 평가 지표   | 핵심 결과                      |
| ------------------------------- | -------------- | -------------------- | ----------- | ------------------------------ |
| A Simple Baseline (2021)        | Pascal VOC     | FCN, CaGNet          | hIoU        | 77.5 (FCN 50.7 대비 +26.8)     |
| ZegFormer (2020)                | Pascal-Context | SPNet-FPN            | hIoU        | 73.3 (seen 40.1 대비)          |
| Decoupling (2022)               | COCO-Stuff     | SPNet-FPN            | unseen mIoU | 21.4 (SPNet 11.0 대비 +10.4)   |
| CAT-Seg (2024)                  | ADE20K-847     | feature aggregation  | mIoU        | 16.0 (unseen-heavy 강점)       |
| FreeSeg (2023)                  | COCO unseen    | ZSSeg, ZSI           | mIoU        | 49.1 (ZSSeg 대비 +5.5)         |
| MAFT (2023)                     | Pascal-VOC     | SPNet, ZS3Net        | unseen mIoU | 81.8 (ensemble 제거 hIoU 44.4) |
| Learning Mask-aware CLIP (2023) | COCO-Stuff     | FreeSeg, ZegFormer   | unseen mIoU | 50.4 (42.2 대비 +8.2)          |
| One-Prompt (2023)               | unseen medical | few-shot/interactive | Dice        | 64.0% (MedSAM 53.9 대비)       |
| OmniMedVQA (2024)               | medical LVLM   | medical-specialized  | QA Score    | general-domain superior        |

#### 2.3 Domain Adaptation 성능 비교

| 논문명                        | 설정 (Source→Target)  | 비교 대상        | 평가 지표 | 핵심 결과                       |
| ----------------------------- | --------------------- | ---------------- | --------- | ------------------------------- |
| ADVENT (2018)                 | GTA5→Cityscapes       | Self-Training+CB | mIoU      | 36.1 (baseline 28.1 대비)       |
| ADVENT (2019)                 | SYNTHIA→Cityscapes    | Biasetton et al. | mIoU      | 31.3 (25.4 대비 +5.9)           |
| FADA (2020)                   | Cityscapes→Cross-City | CAG, CLAN        | mIoU      | 54.7 (source-only 46.2 대비)    |
| CAG (2021)                    | GTA5→Cityscapes       | CAG baseline     | mIoU      | 56.1 (CAG 50.2 대비 +5.9)       |
| PyCDA (2019)                  | GTAV→Cityscapes       | CDA, ST, CBST    | mIoU      | 48.0 (source-only 24.3 대비)    |
| CAMix (2021)                  | GTAV→Cityscapes       | DACS, DAFormer   | mIoU      | 55.2 (DACS 52.1 대비 +3.1)      |
| CFBI (2020)                   | SYNTHIA→Cityscapes    | EGMN             | mIoU      | 48.2 (16-class 기준)            |
| Bidirectional Learning (2019) | GTA5→Cityscapes       | CyCADA, DCAN     | mIoU      | 48.5 (baseline 33.6 대비 +14.9) |
| Effective Synthetic (2018)    | synthetic-only        | adaptation       | mIoU      | 38.0 (adaptation 35.9 대비)     |

### 3. 성능 패턴 및 경향 분석

#### 3.1 공통적으로 나타나는 성능 개선 패턴

| 개선 패턴                        | 관찰 사례                           | 일반화 조건                           |
| -------------------------------- | ----------------------------------- | ------------------------------------- |
| **Context Encoding**             | EncNet, Context-Aware Mixup         | small objects, multi-scale에서 효과적 |
| **Deep Supervision**             | Label Refinement Network, U-Net     | small object 분할 향상 (+5~11%)       |
| **Attention Mechanism**          | CCNet RCCA, CRF, Hypercolumn        | boundary 정밀도, long-range context   |
| **Multi-Scale**                  | Atrous convolution, pyramid pooling | varying size objects 처리             |
| **Self-Supervised Pre-training** | MAFT, STEGO, CLIP                   | unseen class 성능 개선 (+8~16%)       |
| **Cross-Task Learning**          | FreeSeg multi-task                  | generalization, panoptic 강화         |
| **Ensemble**                     | DeepLab-CRF, ensemble models        | consistent improvement (+3~5%)        |

#### 3.2 특정 조건에서만 성능이 향상되는 경우

| 조건                     | 효과적인 방법                  | 비효과적 경우            |
| ------------------------ | ------------------------------ | ------------------------ |
| **unseen-heavy dataset** | CAT-Seg, MAFT, CLIP            | specialized methods      |
| **small dataset**        | pre-training, deep supervision | pure transformer         |
| **large vocabulary**     | mask classification            | per-pixel baseline       |
| **boundary 정밀도 중요** | CRF, DeepLab-FullCRF           | ConvCRF locality 가정    |
| **real-time 요구**       | EfficientViT, SegNeXt          | Mask2Former, InternImage |
| **medical 3D**           | V-Net, 3D U-Net                | 2D CNN only              |
| **medical few-label**    | contrastive, self-supervised   | random init              |

#### 3.3 논문 간 상충되는 결과

| 상충 관계                      | 논문 A              | 논문 B              | 설명                   |
| ------------------------------ | ------------------- | ------------------- | ---------------------- |
| **Boundary vs Speed**          | DeepLab-FullCRF     | EfficientViT        | 정확도 vs 실시간 속도  |
| **Large vs Small Objects**     | InternImage (large) | Hypercolumn (small) | instance size bias     |
| **seen vs unseen**             | specialized models  | CLIP-based          | seen performance 낮음  |
| **2D vs 3D**                   | 2D segmentation     | 3D volumetric       | computational cost     |
| **supervised vs unsupervised** | DeepLab             | STEGO, CLIP         | label 효율성 trade-off |
| **single vs multi-object**     | single-object VOS   | multi-object VOS    | instance counting      |

#### 3.4 데이터셋 또는 환경에 따른 성능 차이

| 데이터셋 특성         | 적합한 방법                | 부적합 방법        |
| --------------------- | -------------------------- | ------------------ |
| **high-resolution**   | ConvNeXt, SegNeXt          | ViT quadratic cost |
| **low-resource**      | EfficientViT, SegFormer-B0 | large models       |
| **medical volume**    | V-Net, 3D U-Net            | 2D CNN             |
| **outdoor**           | Cityscapes, ADE20K         | indoor focused     |
| **medical volume**    | V-Net, 3D U-Net            | 2D CNN             |
| **corruption robust** | EfficientViT (multi-scale) | single-scale       |
| **unseen-heavy**      | CLIP-based                 | specialized        |

### 4. 추가 실험 및 검증 패턴

#### 4.1 ablation study 공통 패턴

| 검증 요소               | 활용 빈도 | 주요 목적                 |
| ----------------------- | --------- | ------------------------- |
| **backbone comparison** | 매우 높음 | ResNet vs ViT vs ConvNeXt |
| **module ablation**     | 높음      | 각 구성원소 기여도        |
| **hyperparameter**      | 중간      | lr, depth, scale 등       |
| **multi-scale testing** | 높음      | robustness 확인           |
| **ensemble**            | 중간      | upper bound 설정          |
| **decomposition**       | 중간      | multi-task 분리           |
| **cross-dataset**       | 낮음      | generalization            |

#### 4.2 condition change 실험 유형

| 실험 유형          | 대표 사례         | 목적             |
| ------------------ | ----------------- | ---------------- |
| **few-label**      | 1~5~200 labels    | data scarcity    |
| **seen-unseen**    | 2/4/6/8/10 split  | generalization   |
| **backbone depth** | 50/101/152 ResNet | model scaling    |
| **corruption**     | 16 Cityscapes-C   | robustness       |
| **resolution**     | 0.5~1.5 scale     | scale invariance |
| **label ratio**    | 1%/5%/10%         | label efficiency |
| **shot setting**   | 1-shot/5-shot     | few-shot scaling |

#### 4.3 데이터 증강 및 학습 전략

| 전략                | 활용 논문       | 효과              |
| ------------------- | --------------- | ----------------- |
| **flip/multiscale** | 거의 모든 논문  | +2~4% 개선        |
| **color jitter**    | 중간            | robustness        |
| **mixup/cutmix**    | 중간            | regularization    |
| **synthetic data**  | synthetic→real  | domain adaptation |
| **pseudo-label**    | semi-supervised | data efficiency   |

### 5. 실험 설계의 한계 및 비교상의 주의점

#### 5.1 비교 조건의 불일치

| 문제                  | 설명                        | 영향                        |
| --------------------- | --------------------------- | --------------------------- |
| **backbone 차이**     | ResNet vs ViT vs ConvNeXt   | fair 비교 어려움            |
| **pre-training**      | ImageNet vs self-supervised | 성능 편향                   |
| **train/test split**  | dataset-dependent           | cross-dataset 일반화 어려움 |
| **evaluation metric** | mIoU vs Dice vs J&F         | direct 비교 불가            |
| **resolution**        | 512² vs 1024² 등            | scale effect                |
| **training time**     | 20k~60k iteration           | resource 효율성             |
| **ensemble**          | 사용/사용 안함              | 상한선 차이                 |

#### 5.2 데이터셋 의존성

| 데이터셋 문제      | 영향                   |
| ------------------ | ---------------------- |
| **VOC 2012**       | 20 class 제한, simple  |
| **Cityscapes**     | urban only, 19 class   |
| **ADE20K**         | large vocab, 150 class |
| **COCO-Stuff**     | thing/stuff 불균형     |
| **medical**        | domain-specific        |
| **Pascal-Context** | context-heavy          |

#### 5.3 일반화 한계

| 한계             | 설명               |
| ---------------- | ------------------ |
| **zero-shot**    | seen bias 존재     |
| **domain shift** | real→synthetic gap |
| **corruption**   | robustness 제한적  |
| **unseen**       | class shift 취약   |
| **medical**      | domain gap 큼      |

#### 5.4 평가 지표의 한계

| 지표       | 문제                      |
| ---------- | ------------------------- |
| **mIoU**   | boundary 정밀도 반영 부족 |
| **Dice**   | class imbalance에 민감    |
| **J&F**    | decay 측정 안됨           |
| **AP**     | small object 약점         |
| **PQ**     | thing/stuff trade-off     |
| **HD/ASD** | medical only              |

### 6. 결과 해석의 경향

#### 6.1 저자들이 결과를 해석하는 공통 경향

| 경향                 | 사례               | 해석               |
| -------------------- | ------------------ | ------------------ |
| **state-of-the-art** | 기존 대비 +X% 개선 | 성능 우위 강조     |
| **novelty**          | new module         | 방법론적 기여 강조 |
| **efficiency**       | X% 속도 향상       | 실용성 강조        |
| **generalization**   | unseen에서 +Y%     | 일반화 능력 강조   |
| **robustness**       | corruption 테스트  | robustness 강조    |

#### 6.2 실제 관찰 vs 해석 차이

| 경우             | 관찰      | 해석                      |
| ---------------- | --------- | ------------------------- |
| **unseen 성능**  | 5~8% 향상 | "generalization"          |
| **few-label**    | +5~10%    | "label-efficient"         |
| **ensemble**     | +2~5%     | "robust prediction"       |
| **backbone**     | +3~6%     | "stronger representation" |
| **pre-training** | +4~12%    | "better initialization"   |

#### 6.3 claim 과 actual gap

| claim 유형           | 실제 gap        | 원인                 |
| -------------------- | --------------- | -------------------- |
| **efficiency claim** | 20~30% 과대평가 | FLOPs vs latency     |
| **generalization**   | 30~40% 과대평가 | seen/unseen bias     |
| **robustness**       | 20~30% 과대평가 | corruption-specific  |
| **sota**             | 10~15% 차이     | experimental setting |

### 7. 종합 정리

Semantic segmentation 실험 결과 전체를 종합하면, **context modeling**(DeepLab, CCNet, EfficientViT),**mask classification**(MaskFormer, Per-Pixel Classification is Not All You Need),**zero-shot generalization**(CLIP-based, CAFT),**domain adaptation**(synthetic-to-real, cross-dataset) 이 세 가지 축에서 기술 진화를 보여줍니다.**unseen-heavy**와**large vocabulary**에서는 mask classification이 per-pixel보다 강하며,**boundary 정밀도**에는 CRF 기반 방법이,**실시간/efficiency**에는 EfficientViT, SegNeXt, EfficientNet 계열이 우세합니다.**medical domain**은 3D volumetric과 contrastive/self-supervised initialization이 label-efficient로 중요하며,**synthetic→real**에서는 domain shift가 5~10% 이상 개선되지만 oracle gap(-15~-20%)이 남아있습니다.**generalization**은 pre-training, multi-scale, ensemble, multi-task 등으로 +5~10% 개선 가능하지만,**evaluation bias**(seen bias, dataset bias)가 여전히 존재합니다.**backbone 확장**(ViT, InternImage)은 large-scale에서 효과적이지만,**small dataset**에서는 과적합 위험이 있습니다.
