# Transformers in Medical Imaging: A Survey

## 논문 메타데이터

- **제목**: Transformers in Medical Imaging: A Survey
- **저자**: Fahad Shamshad, Salman Khan, Syed Waqas Zamir, Muhammad Haris Khan, Munawar Hayat, Fahad Shahbaz Khan, Huazhu Fu
- **arXiv 공개 연도**: 2022
- **정식 출판 연도**: 2023
- **저널**: Medical Image Analysis
- **권/논문 번호**: Volume 88, Article 102802
- **DOI**: 10.1016/j.media.2023.102802
- **PMID**: 37315483
- **arXiv ID**: 2201.09873
- **arXiv URL**: https://arxiv.org/abs/2201.09873
- **PDF URL**: https://arxiv.org/pdf/2201.09873v1

## 연구 배경 및 문제 정의

이 논문은 Vision Transformer(ViT) 계열 모델이 자연어 처리와 일반 컴퓨터 비전에서 성공을 거둔 직후, 그 흐름이 의료영상 분석으로 어떻게 확장되고 있는지를 정리한 대규모 survey다. 저자들은 CNN이 의료영상 분야의 사실상 표준으로 자리 잡았지만, convolution의 지역적 receptive field와 입력 적응성이 낮은 고정 필터 구조 때문에 전역 문맥과 장거리 상호작용을 포착하는 데 구조적 한계가 있다고 본다.

논문이 설정한 핵심 문제는 다음과 같다.

- 의료영상은 CT, MRI, X-ray, ultrasound, pathology, PET 등 modality가 매우 다양하다.
- 과제도 segmentation, classification, detection, reconstruction, synthesis, registration, report generation 등으로 넓게 퍼져 있다.
- Transformer는 전역 self-attention으로 장거리 의존성을 잘 모델링하지만, 의료영상에서는 데이터 부족, 3D 입력, 계산량, 해석 가능성, 분포 이동 같은 별도 문제가 크다.

즉, 이 논문은 "Transformer가 의료영상에서 실제로 어디까지 쓰이고 있는가", "어떤 task에서 어떤 구조가 유리한가", "CNN 대비 장점과 한계가 무엇인가"를 체계적으로 정리하는 것을 목표로 한다.

## 논문의 핵심 기여

이 논문의 기여는 새로운 모델 제안이 아니라, Transformer 기반 의료영상 연구를 본격적으로 분류하고 지형도를 제시했다는 점에 있다.

핵심 기여는 다음과 같다.

1. 당시 기준 125편이 넘는 Transformer 기반 의료영상 논문을 폭넓게 정리했다.
2. segmentation, classification, detection, reconstruction, synthesis, registration, clinical report generation, 기타 응용으로 나누는 task taxonomy를 제시했다.
3. 각 task별로 architectural design, dataset, challenge, trend를 묶어 비교 가능한 survey 구조를 만들었다.
4. hand-crafted method, CNN, ViT로 이어지는 기술적 배경을 간단히 연결해 Transformer의 위치를 설명했다.
5. pre-training, interpretability, adversarial robustness, edge deployment, federated learning, domain adaptation/OOD detection을 핵심 open challenge로 제시했다.

## 방법론 요약

### 1. 논문의 기본 구조

이 논문은 일반적인 "문헌 나열형 survey"보다 더 구조적이다. 먼저 의료영상 분야의 배경을 hand-crafted, CNN, ViT 기반 접근으로 나누어 설명하고, 이후 task별로 Transformer 응용을 세부 taxonomy 아래 정리한다.

주요 섹션 구조는 다음과 같다.

- **Segmentation**
- **Classification**
- **Detection**
- **Reconstruction**
- **Synthesis**
- **Registration**
- **Clinical Report Generation**
- **Other Applications**
- **Open Challenges and Future Directions**

즉, 모델 아키텍처 자체보다 "어떤 문제에서 Transformer가 어떤 방식으로 쓰였는가"에 초점을 둔다.

### 2. Transformer 배경 설명

논문은 self-attention이 Transformer 성공의 중심 메커니즘이라고 본다. 입력 시퀀스 전체의 관계를 직접 계산해 장거리 의존성을 학습할 수 있고, multi-head self-attention으로 다양한 관계 패턴을 동시에 포착할 수 있다는 점이 핵심이다.

의료영상 문맥에서 저자들이 강조하는 장점은 다음과 같다.

- CNN보다 넓은 전역 문맥 활용
- 장기 간 관계나 멀리 떨어진 병변 상호작용 포착 가능
- task에 따라 hybrid CNN-Transformer 설계가 가능
- segmentation, report generation처럼 서로 다른 구조의 문제에도 확장 가능

반면 동시에 지적하는 약점도 분명하다.

- quadratic attention cost
- 대규모 pre-training 필요성
- medical domain에서 충분한 데이터 부족
- 3D 영상에서 메모리/연산량 급증

### 3. 논문이 채택한 정리 방식

이 survey는 각 응용 과제마다 다음 패턴으로 정리한다.

- 문제 정의와 해당 과제의 임상적 중요성 설명
- Transformer 기반 방법의 세부 taxonomy 제시
- 대표 논문 소개
- 데이터셋과 성능 비교
- task-specific challenge와 해결 방향 정리

이 구조 덕분에 단순 bibliography가 아니라 "어떤 설계가 왜 등장했는지"를 추적할 수 있다.

## 응용 분야별 정리

### 1. Medical Image Segmentation

논문에서 가장 큰 비중을 차지하는 영역이 segmentation이다. 저자들은 1년 남짓한 기간 동안 50편이 넘는 관련 논문이 등장했다고 정리하며, Transformer가 가장 빠르게 침투한 분야로 본다.

세부 분류는 다음과 같다.

- **Organ-specific segmentation**
- **Multi-organ segmentation**
- **2D segmentation**
- **3D segmentation**
- **Pure Transformer**
- **Hybrid ViT+CNN**

논문은 의료영상 segmentation에서 Transformer가 특히 유리한 이유를, 멀리 떨어진 spatial region 간 관계와 흩어진 foreground/background 구조를 잘 모델링할 수 있기 때문이라고 설명한다. 다만 실제로는 pure ViT보다 CNN encoder-decoder와 Transformer block을 결합한 hybrid 구조가 많이 채택된다. 이는 local texture와 global context를 동시에 취하기 위한 절충으로 해석된다.

대표 예시로는 TransUNet, CoTr, UNETR, Swin UNETR 등이 소개된다. 특히 Swin UNETR는 domain-specific CT pre-training을 통해 라벨 효율성과 성능을 동시에 개선한 사례로 강조된다.

### 2. Medical Image Classification

classification 섹션은 COVID-19, tumor, retinal disease를 중심으로 정리된다. 저자들은 classification을 단순한 accuracy 경쟁으로 보지 않고, explainability 수준에 따라 **black-box model**과 **interpretable model**로 다시 나눈다.

COVID-19 진단에서는 X-ray, ultrasound, CT 세 modality가 주로 다뤄진다. 여기서는 경량 ViT, transfer learning, federated learning, self-supervised pretraining이 함께 논의된다. 특히 해석 가능성 확보를 위해 saliency map과 Grad-CAM 기반 분석을 함께 사용하는 사례가 정리된다.

tumor classification에서는 MRI, pathology whole-slide image, breast ultrasound 등 다양한 설정이 포함된다. 여기서 중요한 흐름은 단순 2D image classification보다 weakly supervised MIL과 Transformer를 결합해 whole-slide pathology에서 morphology와 spatial relation을 함께 읽는 방향이다. TransMIL 같은 모델이 대표적이다.

retinal disease classification에서는 fundus image와 diabetic retinopathy grading에 대해 lesion-aware transformer 같은 구조가 등장한다. 저자들은 이 영역에서 Transformer가 병변 수준 localization과 disease grading을 동시에 다루는 방향으로 진화한다고 본다.

### 3. Detection

detection 분야는 segmentation이나 classification보다 논문 수는 적지만, DETR 계열 구조의 의료 도메인 확장 가능성을 보여준다. polyp detection, lymph node detection 등 몇몇 응용이 소개되며, 논문은 이 영역이 아직 초기 단계라고 평가한다.

즉, detection은 당시 survey 시점에서는 Transformer의 강점이 아직 충분히 구조화되지 않은 분야였고, 데이터셋과 benchmark의 부족도 제약으로 언급된다.

### 4. Reconstruction

재구성 분야는 이 survey에서 특히 흥미로운 부분이다. 저자들은 reconstruction을 단순 inverse problem이 아니라 Transformer가 global dependency와 prior modeling을 활용할 수 있는 분야로 본다.

다루는 대표 과제는 다음과 같다.

- **LDCT enhancement**
- **LDPET enhancement**
- **undersampled MRI reconstruction**
- **sparse-view CT reconstruction**
- **endoscopic video reconstruction**

이 영역에서 논문이 강조하는 포인트는 두 가지다.

1. Transformer가 전역 정보를 활용해 artifact 제거와 구조 복원을 더 잘할 수 있다.
2. data-poor setting에서는 zero-shot 혹은 untrained prior 기반 접근도 유망하다.

특히 MRI reconstruction에서 ImageNet-pretrained ViT가 low-data regime에서도 sharp reconstruction과 anatomy shift robustness를 보인다는 점이 중요한 관찰로 제시된다.

### 5. Synthesis

synthesis는 intra-modality와 inter-modality로 나누어 정리된다. MRI super-resolution, CT-to-CT enhancement, modality translation 같은 문제가 포함된다. 대부분 adversarial loss와 결합된 구조가 많고, paired data 부족 때문에 semi-supervised 또는 unsupervised 설정이 중요하다고 설명한다.

저자들은 synthesis가 고품질 생성뿐 아니라 data augmentation, missing modality 보완, downstream task 보조의 역할도 할 수 있다고 본다.

### 6. Registration

registration은 survey 시점에서 아직 초기 단계로 평가된다. 다만 Transformer의 long-range correspondence modeling 능력이 deformable registration 같은 문제에 잠재적으로 유리하다고 본다. 현재는 강한 결론을 내리기보다 연구 여지가 큰 분야로 위치시킨다.

### 7. Clinical Report Generation

이 논문은 비전 Transformer만이 아니라 language modeling 기반 report generation까지 의료영상 응용 범위에 포함한다. 여기서는 image encoder와 Transformer decoder를 조합해 chest X-ray나 radiology report를 생성하는 흐름이 소개된다.

이 부분의 의미는 중요하다. 저자들은 의료영상 Transformer 연구가 결국 pure vision task를 넘어 multimodal reasoning과 clinical language generation으로 확장될 것이라고 본다.

## 실험 설정과 결과

이 논문은 survey이므로 자체 실험을 수행하지 않는다. 대신 task별 대표 모델의 데이터셋, 메트릭, 구조적 특징을 표로 정리한다. 따라서 결과 해석의 포인트는 "새로운 최고 성능"보다 "어떤 문제에서 Transformer가 빠르게 표준 후보가 되었는가"에 있다.

### 1. Segmentation

논문은 BTCV, BraTS 등 대표 benchmark에서 Transformer 기반 segmentation 모델이 CNN baseline을 자주 능가하거나 경쟁력 있는 성능을 보인다고 정리한다. 특히 hybrid 구조가 pure ViT보다 더 안정적인 경우가 많다.

또한 segmentation 영역에서는 data availability가 상대적으로 좋아서 Transformer 확산 속도가 가장 빨랐다고 해석한다.

### 2. Classification

classification에서는 breast ultrasound, pathology WSI, retinal disease, COVID-19 CXR/CT 등 다양한 modality에서 ViT 기반 구조가 높은 정확도를 보인다. 그러나 저자들은 단순 accuracy 자체보다 다음 포인트를 더 강조한다.

- transfer learning 활용 빈도가 높다.
- self-supervised pretraining의 효과가 크다.
- saliency와 Grad-CAM 같은 해석 도구가 함께 요구된다.

### 3. Reconstruction

reconstruction에서는 undersampled MRI와 sparse-view CT에서 Transformer가 구조 보존과 전역 consistency 측면에서 강점을 보이는 사례가 소개된다. 저자들은 특히 pretraining과 zero-shot prior 기반 접근이 이 분야에서 중요한 연구 축이 될 수 있다고 본다.

## 한계 및 향후 연구 가능성

이 논문의 강점은 open challenge를 꽤 구체적으로 정리했다는 점이다.

### 1. Pre-training

ViT는 CNN보다 inductive bias가 약하기 때문에 적절한 pre-training 없이는 데이터 효율성이 떨어진다. 저자들은 의료영상 전용 pre-training이 중요하며, domain-specific self-supervised learning이 핵심 과제가 될 것이라고 본다.

### 2. Interpretability

의료영상에서는 black-box 모델 배포가 특히 위험하다. self-attention이 해석 가능성의 단서를 줄 수는 있지만, 실제 clinical interpretability framework는 아직 미성숙하다고 평가한다. 단순 attention map만으로는 충분하지 않다는 문제의식도 깔려 있다.

### 3. Adversarial Robustness

저자들은 의료 시스템이 금전적 인센티브와 결합된 고위험 환경이기 때문에 adversarial attack 연구가 중요하다고 본다. ViT가 자연영상에서는 CNN보다 견고할 수 있다는 초기 결과가 있지만, 의료영상에서는 아직 체계적 검증이 부족하다고 지적한다.

### 4. Edge Deployment

실제 의료 현장에서는 edge device, portable scanner, point-of-care setting이 중요하지만 ViT는 연산과 메모리 요구량이 크다. 따라서 compression, architecture search, hardware-aware design이 필요하다고 본다.

### 5. Decentralized / Federated Learning

의료 데이터는 병원 간 공유가 어렵기 때문에 federated learning과 privacy-preserving learning이 중요하다. 논문은 ViT의 구조적 장점이 multi-task federated setting과 결합될 수 있다고 보지만, 아직 proof-of-concept 수준이라고 평가한다.

### 6. Domain Adaptation and OOD Detection

병원, 장비, 환자군, acquisition protocol 차이로 인한 distribution shift는 의료영상의 핵심 문제다. 저자들은 ViT가 표현 품질 측면에서 OOD detection에 유리할 수 있지만, 의료영상에서는 benchmark와 rigorous protocol이 아직 부족하다고 본다.

## 실무적 또는 연구적 인사이트

### 1. 이 논문은 "초기 대형 roadmap" 역할을 한다

이 survey는 Transformer가 의료영상의 일부 하위 과제에 시도되는 수준을 넘어, 거의 모든 핵심 task로 확장되기 시작한 시점을 기록한다. 따라서 최신 세부 알고리즘을 찾는 문서라기보다, 연구 방향의 전체 맵을 잡는 문서로 읽는 것이 맞다.

### 2. 가장 큰 메시지는 pure ViT보다 hybrid 전략의 실용성이다

논문을 전체적으로 보면, 당시 의료영상에서는 pure Transformer보다 CNN과 Transformer를 섞는 hybrid 구조가 훨씬 많다. 이는 의료영상이 여전히 local texture, edge, anatomy prior에 크게 의존하기 때문이다. 즉, "Transformer가 CNN을 즉시 대체한다"기보다 "CNN의 local bias와 Transformer의 global modeling을 결합한다"가 실제 주류 전략이었다.

### 3. segmentation이 가장 먼저 변하고, detection과 registration은 느리다

이 survey는 분야별 성숙도 차이를 잘 보여준다.

- segmentation: 가장 빠르게 발전
- classification: 응용은 넓지만 explainability 이슈 큼
- reconstruction/synthesis: inverse problem과 잘 결합됨
- detection/registration: 아직 초기 단계

이 구도는 이후 몇 년간의 발전 방향과도 크게 어긋나지 않는다.

### 4. 오늘 시점에서 보면 매우 예측력이 좋다

논문이 open challenge로 든 항목들, 즉 pre-training, self-supervised learning, domain adaptation, OOD detection, edge deployment, federated learning, interpretability는 이후 실제로 의료 AI의 중심 과제가 되었다. 이 점에서 단순 정리 이상의 통찰을 가진 survey라고 볼 수 있다.

### 5. 동시에 오늘 기준으로는 업데이트가 필요하다

논문의 한계도 분명하다.

- foundation model, medical VLM, SAM 계열, diffusion 기반 medical generation의 후속 급팽창은 반영하지 못한다.
- 2022 arXiv 기준 문헌이라 2023년 이후의 대형 모델 흐름은 빠져 있다.
- benchmark 비교는 task별 protocol이 완전히 통일된 메타분석 수준은 아니다.

따라서 현재 연구자가 이 논문을 볼 때는 "최신 완결판 survey"가 아니라 "Transformer 의료영상 연구의 초기 체계화 문헌"으로 위치시키는 것이 적절하다.

## 종합 평가

이 논문은 Transformer 기반 의료영상 분석 연구를 처음으로 본격적인 task taxonomy 수준에서 정리한 대표 survey다. segmentation, classification, reconstruction, synthesis, registration, report generation까지 폭넓게 포괄하고, 각 분야의 구조적 문제와 향후 과제를 함께 제시한다는 점이 가장 큰 강점이다.

연구 입문자에게는 "Transformer가 의료영상에서 어디에 적용되는가"를 이해하게 해 주고, 이미 이 분야를 연구하는 사람에게는 "어떤 과제가 상대적으로 성숙했고 어디에 공백이 있는가"를 보여주는 로드맵 역할을 한다. 현재 시점에서는 최신 foundation-model 흐름을 보완하는 추가 survey와 함께 읽는 것이 가장 적절하지만, 의료영상 Transformer 연구의 출발점과 초기 설계 원리를 이해하는 기준 문헌으로서는 여전히 가치가 높다.
