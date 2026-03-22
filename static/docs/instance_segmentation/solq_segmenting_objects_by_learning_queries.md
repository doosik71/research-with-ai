# SOLQ: Segmenting Objects by Learning Queries

이 논문은 instance segmentation을 **진정한 end-to-end 방식**으로 풀기 위해, DETR의 object query를 그대로 확장한 **Unified Query Representation (UQR)**을 제안합니다. 핵심 문제의식은 기존 two-stage instance segmentation이 detection branch에 segmentation branch가 종속되어 joint learning이 약하고, DETR 계열에 마스크를 붙이려 해도 query embedding을 억지로 spatial mask로 reshape하는 방식은 공간 정보를 잘 담지 못한다는 점입니다. 이를 해결하기 위해 저자들은 각 query가 **class, box, mask를 동시에 담는 하나의 unified vector**가 되도록 만들고, 고해상도 spatial mask는 직접 예측하지 않고 **compression coding으로 저차원 mask vector**로 바꿔 회귀하게 합니다. 이로써 SOLQ는 box와 mask를 함께 **parallel regression**으로 출력하는 end-to-end instance segmentation 프레임워크가 됩니다.  

## 1. Paper Overview

이 논문이 해결하려는 핵심 문제는 “DETR의 end-to-end detection 철학을 instance segmentation까지 자연스럽게 확장할 수 있는가”입니다. 기존 instance segmentation 방법들은 크게 two-stage(top-down), bottom-up, directly-predict 계열로 나뉘지만, strong baseline들은 대체로 detect-then-segment 구조를 따릅니다. 이런 구조는 optimization은 쉽지만, segmentation branch가 detection branch에 강하게 의존하기 때문에 multi-task를 더 잘 함께 학습하기 어렵습니다. 한편 SOLO/CondInst 같은 direct 또는 anchor-free 계열은 RoI cropping 문제를 줄였지만, 여전히 proposal 기반 conditioning이나 NMS 같은 hand-crafted post-processing에 기대는 부분이 남아 있습니다. 논문은 결국 **“진짜 end-to-end instance segmentation은 아직 남은 문제”**라고 규정합니다.  

특히 저자들이 중요하게 보는 지점은 DETR류 query가 detection에는 잘 맞지만, segmentation에서는 그대로 쓰기 어렵다는 점입니다. DETR를 panoptic segmentation으로 확장한 naive 방식은 query embedding을 spatial domain으로 reshape하고 FPN-style mask branch를 붙이는데, 논문은 Transformer encoder/decoder가 spatial mask를 직접 생성하기에 충분히 공간 정보를 잘 모델링하지 못한다고 비판합니다. 게다가 고해상도 2D mask를 그대로 supervision하면 계산 비용이 커지고, 결과적으로 detection을 먼저 학습한 뒤 segmentation branch를 따로 학습하는 식의 분리된 학습으로 흐르기 쉽습니다.

이 문제가 중요한 이유는, instance segmentation이 단순 detection보다 훨씬 richer한 output space를 가지기 때문입니다. box뿐 아니라 pixel-level mask까지 동시에 예측해야 하며, 이 둘을 truly unified representation 아래에서 함께 학습할 수 있다면 detection과 segmentation이 서로 보조적으로 작용할 수 있습니다. 실제로 논문은 UQR을 쓰면 segmentation이 detection도 향상시킨다고 주장합니다.  

## 2. Core Idea

논문의 핵심 아이디어는 다음과 같습니다.

> **object query를 detection용 벡터로만 쓰지 말고, class, location, mask를 함께 담는 unified query로 학습하자.**

즉 DETR의 각 query는 원래 “하나의 object를 대표하는 learnable embedding”인데, SOLQ는 이 표현을 더 밀어붙여 classification, box regression, mask encoding까지 모두 하나의 query representation에서 병렬로 뽑아냅니다. 논문은 이를 **Unified Query Representation (UQR)**이라고 부릅니다.  

하지만 여기서 핵심 난점은 mask입니다. class나 box는 벡터 회귀가 자연스럽지만, mask는 원래 2D spatial structure입니다. 그래서 저자들은 “mask 자체를 2D로 직접 예측하지 말고, **compression coding으로 저차원 embedding으로 투영**한 뒤 그 벡터를 예측하자”고 제안합니다. 훈련 시에는 GT spatial mask를 low-dimensional mask embedding으로 바꿔 supervision하고, 추론 시에는 예측된 mask vector를 inverse transform으로 다시 spatial mask로 복원합니다. 즉, segmentation도 query space 안에서 regression task로 통일됩니다.  

이 설계의 novelty는 세 가지로 정리할 수 있습니다.

첫째, DETR 스타일 set prediction을 유지하면서 instance segmentation을 붙였다는 점입니다.
둘째, mask representation을 spatial domain이 아니라 **embedding domain**으로 옮겨 query와 표현 형식을 맞췄다는 점입니다.
셋째, 이 결과 detection과 segmentation이 **shared query representation 위에서 joint learning**되며, 실제로 detection AP도 향상된다는 점입니다.  

## 3. Detailed Method Explanation

### 3.1 DETR에서 출발하는 기본 틀

SOLQ는 기본적으로 DETR의 object detection formulation을 그대로 계승합니다. 즉 object detection을 set prediction 문제로 보고, fixed number의 learnable query가 이미지 feature와 Transformer decoder에서 상호작용한 뒤 object-level prediction을 냅니다. box와 class에 대해서는 DETR의 bipartite matching loss와 classification/L1/GIoU 손실 구성을 그대로 따릅니다. 이 부분은 SOLQ가 완전히 새로운 detector가 아니라, **DETR 위에 segmentation-compatible representation을 올리는 방식**임을 보여줍니다.

### 3.2 Network Architecture

논문은 SOLQ를 세 부분으로 설명합니다.

* feature extraction network
* Transformer decoder
* unified query representation (UQR)

이미지 feature는 backbone에서 추출되고, learnable queries는 Transformer decoder에서 image feature와 상호작용하며 **instance-aware query embeddings**로 업데이트됩니다. 이후 이 query embeddings는 세 개의 branch로 들어가며, 각각 linear projection을 통해 class vector, box vector, mask vector를 생성합니다. 중요한 점은 세 sub-task가 모두 **동일한 query representation을 공유**하고, 모두 regression-like prediction으로 통일된다는 것입니다.  

즉, SOLQ에서 instance segmentation은 “detector 뒤에 별도 mask head를 붙이는 것”이 아니라, **query 하나가 곧 object 하나이고, 그 query가 box와 mask를 동시에 책임지는 구조**입니다. 이 점이 Mask R-CNN 계열이나 naive DETR+mask branch와 가장 크게 다릅니다.

### 3.3 왜 naive spatial mask prediction이 부족한가

논문은 naive DETR-style segmentation의 문제를 꽤 분명하게 말합니다. query embedding을 억지로 spatial domain으로 reshape해 FPN-style CNN으로 mask를 만들면, detection branch의 representation form과 segmentation branch의 representation form이 달라집니다. 그 결과 query 자체는 class/box용 벡터처럼 작동하지만, mask는 별도의 spatial decoder에 크게 의존하게 되어 representation consistency가 깨집니다. 게다가 mask supervision이 2D high-resolution label이라 computation cost가 커지고, 분리 학습으로 흐르기 쉽습니다.  

SOLQ는 이 문제를 “representation mismatch”로 보고, 이를 compression-coded mask vector로 해결합니다. 즉 mask도 query가 직접 예측할 수 있는 **vector target**이 되면 query의 unified nature를 유지할 수 있습니다.

### 3.4 Mask Compression Coding

이 논문의 가장 중요한 기술적 요소입니다. 저자들은 좋은 mask representation이 세 조건을 만족해야 한다고 말합니다.

1. spatial mask를 embedding domain으로 자연스럽게 바꿀 수 있어야 한다.
2. 그 변환이 가역적이어야 한다.
3. 낮은 차원에서도 spatial mask의 principal information을 유지해야 한다.

이 요구를 만족하는 후보로 저자들은 classical compression coding을 택합니다. 논문이 비교한 것은 직접 flattening, Sparse Coding, PCA, DCT 등입니다. 훈련 단계에서는 GT binary mask를 먼저 compression coding으로 저차원 벡터로 변환하고, 모델은 이 mask vector를 예측하도록 supervision을 받습니다. 추론 단계에서는 예측된 mask vector를 inverse transform하여 최종 spatial mask를 복원합니다.  

이 설계는 ISTR류 PCA-mask embedding 계열과 비슷해 보일 수 있지만, SOLQ에서는 이것이 **DETR query representation과의 정합성**을 위해 도입되었다는 점이 다릅니다. 즉 compression coding은 단순한 mask compression trick이 아니라, query-based end-to-end learning의 일부입니다.

### 3.5 어떤 compression coding이 가장 좋은가

Ablation에서 저자들은 flat vector supervision도 box AP를 약간 높이지만, segmentation 성능 측면에서는 **DCT가 가장 좋다**고 보고합니다. 논문 설명에 따르면 DCT가 Sparse Coding이나 PCA보다 compression loss가 작고, 무엇보다 **online 방식**으로 각 이미지에 바로 적용 가능하다는 장점이 있습니다. 반면 Sparse Coding/PCA는 전체 training set에 대해 offline하게 dictionary나 principal components를 먼저 구해야 합니다. 그래서 논문은 DCT를 default compression coding method로 선택합니다.

이 부분은 실용적으로도 중요합니다. end-to-end detector에 붙일 mask encoding 방식은 정확도뿐 아니라 train/inference pipeline의 간결성도 중요하기 때문입니다. SOLQ가 DCT를 선택한 것은 단지 약간 더 좋은 성능 때문만이 아니라, **workflow simplicity**도 고려한 결과로 읽힙니다.

### 3.6 Unified Query Representation vs Separate Query Representation

논문이 강하게 밀고 있는 또 하나의 핵심은 UQR이 단순한 mask head 설계가 아니라, **representation-level design choice**라는 점입니다. 저자들은 DETR-style detection query와 별도의 segmentation representation을 쓰는 방식은 **Separate Query Representation (SQR)**으로 두고, 자신들의 UQR과 직접 비교합니다. 결과적으로 UQR은 SQR보다 segmentation도 훨씬 좋고, detection AP도 더 크게 향상시킵니다.

이 비교는 꽤 중요합니다. 왜냐하면 단순히 “DETR에 segmentation branch를 추가했다”가 아니라, **모든 task를 unified representation으로 함께 학습시키는 것 자체가 성능 이득의 원천**이라는 주장을 뒷받침하기 때문입니다.

## 4. Experiments and Findings

### 4.1 주요 성능

논문은 COCO에서 strong result를 보고합니다. Introduction에서 저자들은 **ResNet-101 backbone 기준 40.9 mask AP, 48.7 box AP**를 기록했다고 밝히며, 이는 SOLOv2보다 **+1.2 mask AP, +6.1 box AP** 높은 수치라고 주장합니다. 또한 DETR 대비 box AP가 약 **+2.0** 개선된다고 강조합니다.

이 결과의 의미는 두 가지입니다.
첫째, SOLQ는 단순히 segmentation을 DETR에 “붙인” 수준이 아니라 당시 strong baseline을 넘는 competitive framework입니다.
둘째, segmentation을 같이 배우는 것이 detection까지 좋아지게 만들었다는 점에서 **multi-task learning with unified representation**의 효용을 보여줍니다.  

### 4.2 UQR가 detection도 왜 좋아지나

논문에서 가장 흥미로운 실험 중 하나는 UQR vs SQR입니다. 저자들은 UQR이 DETR의 detection 성능을 큰 폭으로 높인다고 보고합니다. 구체적으로 **ResNet-50에서 +2.3 box AP, ResNet-101에서 +2.0 box AP**가 향상되며, 반면 SQR은 box AP를 거의 개선하지 못하고 segmentation AP도 낮다고 설명합니다.  

이는 논문 전체의 핵심 메시지와 맞닿아 있습니다. detection과 segmentation이 서로 다른 형식의 representation으로 분리되어 학습되면 상호 이득이 제한되지만, UQR에서는 한 query가 object의 semantic, geometry, shape를 함께 담기 때문에 더 풍부한 object representation을 형성할 수 있다는 해석이 가능합니다. 이 부분은 논문의 결과에 기반한 해석입니다.  

### 4.3 Compression coding ablation

압축 방식 비교에서도 중요한 메시지가 나옵니다. 직접 flatten한 mask vector도 baseline보다 그럭저럭 괜찮지만, **DCT가 segmentation과 detection 모두에서 가장 좋다**고 보고합니다. 저자들은 그 이유를 “mask compression으로 인한 정보 손실이 상대적으로 작기 때문”이라고 설명합니다. 즉, SOLQ의 성공은 단순히 query를 쓰는 것뿐 아니라, **mask embedding이 spatial mask를 충분히 잘 보존해야 한다**는 사실을 보여줍니다.

### 4.4 정성적 결과와 failure case

논문은 UQR이 SQR보다 더 fine-grained한 mask를 만든다고 시각화 결과를 통해 말합니다. 또한 decoder attention visualization에서 DETR는 주로 object extremities를 보는 반면, SOLQ는 **object outline** 쪽을 더 본다고 설명합니다. 이 점은 query가 단순 box localization을 넘어서 shape-aware representation으로 진화했음을 시사합니다.  

실패 사례도 흥미롭습니다. 논문 부록은 **occluded scenes**에서 failure가 자주 생긴다고 말합니다. 이유는 각 object query가 특정 region을 담당하기 때문에, overlap이 큰 객체들은 동일 query를 공유하거나 인접 query로 할당되어 **siamese mask** 같은 오류를 낼 수 있기 때문입니다. 즉, set prediction framework의 장점이 있는 반면 고중첩 장면에서는 query assignment ambiguity가 남아 있습니다.

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **representation design의 일관성**입니다. SOLQ는 detection과 segmentation을 따로 놀게 두지 않고, query 하나에 class/box/mask를 함께 담습니다. 이는 DETR의 철학과도 잘 맞고, 단순한 architecture trick보다 더 본질적인 장점입니다.  

둘째, mask를 compression-coded vector로 바꾸는 발상이 매우 실용적입니다. 고해상도 2D supervision을 직접 다루지 않으면서도 spatial mask를 복원할 수 있기 때문에, query-based end-to-end learning과 mask prediction 사이의 간극을 잘 메웁니다.

셋째, 실험적으로도 detection과 segmentation을 동시에 개선했다는 점이 강합니다. 보통 segmentation branch는 detection accuracy를 해치거나 별 이득이 없을 수도 있는데, SOLQ는 오히려 box AP까지 높입니다.

### 한계

첫째, mask representation의 품질이 compression coding에 크게 의존합니다. 논문이 DCT/PCA/Sparse Coding을 비교한 것도 이 때문입니다. 즉, mask embedding이 충분히 expressive하지 않으면 query-based segmentation의 장점이 곧바로 제한될 수 있습니다.

둘째, occlusion과 overlap이 큰 장면에서 failure가 나타납니다. 저자들 설명대로 object query가 특정 region을 담당하기 때문에, heavily overlapped objects에서는 query assignment가 꼬여 잘못된 mask가 생길 수 있습니다.

셋째, 오늘 관점에서 보면 SOLQ는 이후의 mask2former류처럼 mask를 더 직접적으로 다루는 query-based segmentation으로 가기 전 단계의 중요한 bridge처럼 읽힙니다. 즉, query-based end-to-end segmentation이라는 큰 방향은 맞지만, 여전히 mask를 compression vector로 우회해 다루고 있다는 점에서 과도기적 성격도 있습니다. 이 평가는 논문 구조에 기반한 해석입니다.

### 비판적 해석

제 해석으로 이 논문의 진짜 공헌은 “instance segmentation을 query representation problem으로 완전히 밀어붙였다”는 데 있습니다. 단순히 DETR에 mask head를 추가한 것이 아니라, **mask도 query space 안에서 다뤄야 한다**는 원칙을 세운 점이 중요합니다.

또한 UQR vs SQR 결과는 꽤 인상적입니다. 이 논문은 segmentation을 같이 배우면 detection이 왜 좋아질 수 있는지를, 단순 multi-task learning이 아니라 **unified representation learning**의 효과로 설명합니다. 이 메시지는 이후 query-based perception 연구 전반에도 중요한 통찰로 이어집니다.  

## 6. Conclusion

SOLQ는 DETR 기반의 **end-to-end instance segmentation framework**로, 각 object query가 class, box, mask를 동시에 담는 **Unified Query Representation**을 학습합니다. mask는 high-resolution spatial map을 직접 예측하지 않고, compression coding으로 만든 low-dimensional mask vector를 회귀한 뒤 복원합니다. 이 설계 덕분에 segmentation이 query representation과 형식적으로 정렬되고, detection과 segmentation이 truly joint하게 학습됩니다. 논문은 COCO에서 strong mask AP와 box AP를 보고하며, 특히 UQR이 DETR의 detection 성능까지 끌어올린다고 보여줍니다.  

연구적으로는 query-based instance segmentation의 중요한 기준점이며, 실무적으로는 “query 하나로 object의 semantic, geometry, shape를 함께 다루는 방식”이 얼마나 강력한지를 보여준 논문입니다. 특히 DETR 계열을 segmentation으로 확장할 때 **mask representation을 어떻게 query space에 맞출 것인가**라는 질문에 대해 아주 명확한 답을 제시한 작업입니다.
