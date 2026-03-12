# Vision Foundation Models in Medical Image Analysis: Advances and Challenges

## 논문 메타데이터

- **제목**: Vision Foundation Models in Medical Image Analysis: Advances and Challenges
- **저자**: Pengchen Liang, Bin Pu, Haishan Huang, Yiwei Li, Hualiang Wang, Weibo Ma, Qing Chang
- **발표 연도**: 2025
- **출판 형태**: arXiv preprint
- **arXiv ID**: 2502.14584
- **버전**: v2
- **DOI**: 10.48550/arXiv.2502.14584
- **arXiv URL**: https://arxiv.org/abs/2502.14584
- **PDF URL**: https://arxiv.org/pdf/2502.14584v2

## 연구 배경 및 문제 정의

이 논문은 최근 의료영상 분석에서 Vision Foundation Models(VFMs), 특히 Vision Transformer(ViT)와 Segment Anything Model(SAM)이 빠르게 확산되고 있다는 점을 배경으로 한다. 저자들은 이들 모델이 장거리 의존성 포착과 일반화 능력에서 기존 CNN보다 유리하지만, 자연영상 기반 대규모 사전학습 모델을 의료영상에 그대로 적용하는 데는 구조적 제약이 있다고 본다.

논문이 설정한 핵심 문제는 다음과 같다.

- 의료영상과 자연영상 사이의 도메인 차이가 크다.
- 의료영상 데이터는 대규모 라벨 수집이 어렵고 데이터셋 규모가 작다.
- 임상 환경은 연산 자원이 제한적이어서 경량화와 효율적 적응이 필요하다.
- 의료기관 간 데이터 공유가 어렵기 때문에 privacy-preserving 학습이 중요하다.

이 문제의식에 따라 논문은 의료영상 분할을 중심으로 VFM 적응 연구를 정리하고, 특히 `도메인 적응`, `모델 압축`, `지식 증류`, `연합학습`을 주요 축으로 리뷰한다.

## 핵심 기여

이 논문은 새로운 알고리즘을 제안하는 연구라기보다, 의료영상용 VFM 적응 흐름을 정리하는 survey 성격의 문헌이다. 기여는 다음 네 가지로 요약할 수 있다.

1. 의료영상 분석에서 ViT 계열이 어떤 방식으로 CNN의 한계를 보완했는지 개괄한다.
2. SAM 중심의 foundation model 적응 연구를 adapter, PEFT, 3D 확장, few-shot 적응 관점에서 정리한다.
3. 모델 압축과 지식 증류를 edge deployment 문제와 연결해 설명한다.
4. 의료 데이터 프라이버시 제약을 반영해 federated learning과 foundation model의 결합 가능성을 별도 축으로 논의한다.

즉, 이 논문은 "의료영상용 foundation model이 무엇인가"를 정의하기보다는, 이미 등장한 자연영상 기반 VFM을 의료 도메인에 어떻게 실용적으로 맞출 것인가를 정리한 로드맵에 가깝다.

## 방법론 및 논문 구조 요약

## 1. Vision Transformer의 의료영상 도입

논문은 먼저 ViT가 의료영상 분석에 도입된 이유를 설명한다. CNN은 지역적 receptive field에 강하지만 장거리 spatial relationship을 직접 모델링하는 데 한계가 있고, 의료영상은 장기 간 관계나 병변과 주변 조직의 전역 문맥이 중요하기 때문에 self-attention 기반 구조가 유리하다는 것이다.

저자들은 TransUNet, Swin-UNet 같은 모델을 예로 들며 다음 흐름을 제시한다.

- U-Net류의 정밀한 localization 능력 유지
- Transformer의 전역 attention 도입
- segmentation 중심 의료영상 과제에서 성능 개선

다만 이 단계의 ViT 계열 모델은 여전히 의료 데이터 부족 문제 때문에 대규모 사전학습의 이점을 충분히 누리기 어렵다고 지적한다.

## 2. 대규모 모델 중심 패러다임으로의 전환

논문은 의료영상 분석이 "전통적 task-specific model"에서 "large model-driven paradigm"으로 이동하고 있다고 본다. 그 상징적 전환점으로 SAM을 제시한다.

저자들의 정리는 다음과 같다.

- SAM은 자연영상에서 학습된 범용 분할 모델로 strong zero-shot segmentation capability를 보였다.
- 그러나 의료영상은 멀티모달, 복잡한 해부학 구조, 높은 도메인 특수성 때문에 직접 적용 시 성능이 제한된다.
- 따라서 의료 도메인 적응이 핵심 연구 주제가 되었다.

이 논문 전체의 초점도 바로 여기에서 형성된다. 즉 "foundation model 자체"보다 "foundation model adaptation"이 중심이다.

## 3. SAM 적응 방법 정리

논문에서 가장 큰 비중을 차지하는 부분은 SAM 기반 적응 연구다. 저자들은 이를 크게 다음 방향으로 분류한다.

- adapter-based improvement
- 3D medical adaptation
- parameter-efficient fine-tuning
- low-rank adaptation
- few-shot adaptation
- prompt quality 문제 완화

본문에서 언급한 대표 흐름은 다음과 같다.

- `Med-SA`: Space-Depth Transpose와 Hyper-Prompting Adapter를 이용한 의료 도메인 적응
- `3DMedSAM`: 2D SAM을 3D 의료영상으로 확장하기 위한 3D patch embedding과 multi-scale 3D mask decoder
- `Trans-SAM`: PEFT 기반 adapter 설계
- `LoRASAM`: low-rank adaptation으로 학습 파라미터를 99% 이상 줄이는 접근
- `DeSAM`: poor prompt의 영향을 줄이기 위한 decoupling 설계

논문은 이러한 연구들이 효과적이지만, 여전히 multi-scale context modeling 부족, domain-specific parameter update 전략 부족, 이론보다는 heuristic에 의존한 설계라는 한계가 남아 있다고 평가한다.

## 4. 모델 압축과 지식 증류

논문은 의료 edge device 배포 문제를 별도 축으로 다룬다. 임상 환경에서는 고성능 서버만 있는 것이 아니므로, foundation model의 추론 비용과 메모리 부담을 줄이는 것이 중요하다는 관점이다.

정리된 핵심 방향은 다음과 같다.

- 대형 teacher 모델의 표현을 소형 student로 전달하는 knowledge distillation
- 데이터 비의존적 증류
- self-distillation과 cross-distillation
- lightweight ViT를 위한 two-stage distillation
- SAM의 경량화를 위한 MobileSAM, TinySAM, SAM-Lightening, EfficientViT-SAM, PQ-SAM

이 섹션의 논지는 명확하다. 의료영상용 foundation model의 실제 도입을 위해서는 단순 정확도 경쟁보다 `효율적 배치 가능성`이 중요하며, 지식 증류와 압축은 이를 위한 핵심 수단이라는 것이다.

## 5. 연합학습과 foundation model 결합

논문은 federated learning(FL)을 VFM 의료 적용의 중요한 확장 방향으로 본다. 의료 데이터는 기관별 silo에 묶여 있고 개인정보보호 요구가 강하기 때문에, 중앙집중식 대규모 사전학습이나 공동 미세조정이 어렵다.

저자들이 제시하는 주요 포인트는 다음과 같다.

- FL은 raw data 공유 없이 다기관 협업 학습을 가능하게 한다.
- foundation model은 성능이 강하지만 파라미터 수가 커서 통신 비용이 높다.
- 따라서 FL 환경에서는 PEFT, adapter sharing, sparse activation, prompt-based personalization이 중요하다.

논문은 FedPFT, FedPIA, dual-personalization adapter, prompt-based FL 같은 방향을 언급하며, 앞으로의 과제로 통신 효율, robustness, privacy, heterogeneous data 대응을 제시한다.

## 실험 및 결과 해석

이 논문은 survey이므로 자체 정량 실험을 수행하지 않는다. 대신 대표 방법들을 정리하면서 현재 연구 흐름에서 무엇이 성과를 내고 있는지 서술적으로 분석한다.

실질적으로 드러나는 메시지는 다음과 같다.

- 의료영상 VFM 적응에서 현재 가장 활발한 영역은 segmentation이다.
- 순수 zero-shot 적용보다 adapter 또는 PEFT 기반 domain adaptation이 성능과 비용의 균형 측면에서 유리하다.
- 2D 자연영상 기반 foundation model을 3D 의료영상으로 옮기는 문제가 여전히 중요하다.
- 모델 압축과 증류는 성능 유지뿐 아니라 임상 배치 가능성 측면에서 점점 더 중요해지고 있다.
- privacy-preserving 다기관 학습은 향후 foundation model 확장의 핵심 기반이 될 가능성이 높다.

즉, 이 논문은 의료영상에서 foundation model 연구가 단순 backbone 경쟁을 넘어 `적응`, `효율`, `배포`, `협업 학습` 문제로 이동하고 있음을 보여준다.

## 강점

## 1. 기술 축이 비교적 명확하다

많은 survey가 모델 이름 나열에 그치는 반면, 이 논문은 domain adaptation, compression, distillation, federated learning이라는 비교적 선명한 축으로 논의를 전개한다.

## 2. 의료영상의 실무적 제약을 반영한다

작은 데이터셋, 프라이버시, edge device, 다기관 협업 같은 현실 문제를 함께 다룬다는 점이 실용적이다.

## 3. SAM 이후의 적응 연구를 빠르게 묶어낸다

2023년 이후 급증한 MedSAM 계열, LoRA 계열, PEFT 계열 흐름을 빠르게 훑기 위한 입문 survey로는 유용하다.

## 한계와 비판적 검토

## 1. 제목에 비해 범위가 좁다

제목은 `Vision Foundation Models in Medical Image Analysis` 전반을 다루는 듯하지만, 실제 중심은 의료영상 `segmentation` 적응이다. 분류, 탐지, 보고서 생성, registration, multimodal fusion, pathology foundation model 등은 상대적으로 거의 다뤄지지 않는다.

## 2. survey 깊이가 제한적이다

17페이지 분량의 짧은 리뷰라서 개별 방법론의 설계 차이, 벤치마크 조건, 실패 사례, 재현성 문제를 깊게 파고들지는 않는다. 넓게 정리하지만 분석의 밀도는 중간 수준이다.

## 3. foundation model의 정의가 다소 느슨하다

논문은 ViT, SAM, CLIP 계열을 포괄적으로 VFMs라고 부르지만, 실제로 어떤 조건을 충족해야 의료영상용 foundation model이라 할 수 있는지에 대한 엄밀한 기준은 제시하지 않는다.

## 4. 최신 의료 특화 foundation model과의 연결이 약하다

의료영상 자체로 사전학습한 domain-native backbone이나 vision-language medical foundation model과의 비교는 약하다. 따라서 "의료영상용 foundation model 자체의 발전사"를 보려면 보완 문헌이 더 필요하다.

## 5. 정량적 종합표의 정보량이 제한적이다

survey 논문의 가치 중 하나는 체계적인 표와 taxonomy인데, 이 논문은 핵심 방향을 설명하는 데 집중한 대신 구조화된 비교표나 세밀한 benchmark synthesis는 상대적으로 부족하다.

## 실무적 및 연구적 인사이트

이 논문이 주는 가장 중요한 통찰은 의료영상에서 foundation model의 승부처가 더 이상 "그대로 가져다 쓰는가"가 아니라 "어떻게 적응시키는가"라는 점이다. 자연영상에서 학습된 거대 모델이 강력한 출발점이 되더라도, 의료영상에서는 해부학 구조, 3D 볼륨 특성, modality heterogeneity, 데이터 프라이버시, 임상 추론 비용을 함께 다뤄야 한다.

후속 연구 방향도 비교적 분명하다.

- 이론적으로 정당화된 medical-specific adaptation 설계
- segmentation 이외 task로의 확장
- compression과 privacy-preserving learning의 공동 최적화
- 2D 자연영상 중심 foundation model을 3D/멀티모달 의료영상으로 확장
- adapter, prompt, federated personalization을 결합한 다기관 실전 배치형 파이프라인 개발

## 종합 평가

`Vision Foundation Models in Medical Image Analysis: Advances and Challenges`는 의료영상용 VFM 연구의 완성된 지형도를 주는 대형 survey라기보다, SAM 이후의 의료영상 적응 연구를 빠르게 구조화한 짧은 обзор 문헌에 가깝다. 특히 domain adaptation, model compression, knowledge distillation, federated learning을 한 흐름으로 묶어 설명한다는 점이 특징이다.

따라서 이 논문은 의료영상 foundation model 분야를 처음 훑는 독자에게는 최근 적응 이슈를 빠르게 파악하는 데 유용하지만, 세부 방법 비교나 의료 특화 foundation model의 전체 계보를 이해하려면 더 넓고 깊은 survey와 함께 읽는 것이 적절하다.
