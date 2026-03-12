# Transformers in Medical Image Analysis: A Review

## 논문 메타데이터

- **제목**: Transformers in Medical Image Analysis: A Review
- **저자**: Kelei He, Chen Gan, Zhuoyuan Li, Islem Rekik, Zihao Yin, Wen Ji, Yang Gao, Qian Wang, Junfeng Zhang, Dinggang Shen
- **arXiv 공개 연도**: 2022
- **정식 출판 연도**: 2023
- **저널**: Intelligent Medicine
- **권/호/페이지**: 3(1), 59-78
- **DOI**: 10.1016/j.imed.2022.07.002
- **arXiv ID**: 2202.12165
- **arXiv URL**: https://arxiv.org/abs/2202.12165
- **PDF URL**: https://arxiv.org/pdf/2202.12165v3

## 연구 배경 및 문제 정의

이 논문은 Transformer가 의료영상 분석 전반으로 빠르게 확산되던 시점에 작성된 review다. 저자들은 Transformer가 NLP에서 long-range dependency modeling으로 큰 성공을 거둔 뒤, Vision Transformer(ViT)를 계기로 컴퓨터 비전과 의료영상 분석까지 본격적으로 들어왔다고 본다.

문제의식은 비교적 분명하다.

- CNN은 의료영상에서 매우 강력하지만, local receptive field 중심 구조라 비지역적 상호작용을 직접적으로 모델링하는 데 제약이 있다.
- 의료영상 분석은 classification, segmentation, detection, registration, synthesis/reconstruction 등 임상 전 과정을 포괄한다.
- Transformer는 전역 self-attention으로 새로운 가능성을 보여주지만, 의료영상에 맞춘 구조 설계, 데이터 효율성, 계산량, 해석 가능성 등은 아직 정리되지 않았다.

따라서 이 논문은 Transformer의 핵심 원리를 소개하고, 의료영상 분석에서 실제로 어떤 방식으로 쓰이고 있으며 어떤 문제가 남아 있는지를 체계적으로 요약하는 것을 목표로 한다.

## 논문의 핵심 기여

이 논문의 기여는 새로운 알고리즘 제안보다, 의료영상 분석에서 Transformer 응용을 task와 학습 설정 중심으로 구조화했다는 데 있다.

핵심 기여는 다음과 같다.

1. 의료영상 분석 전반에서 Transformer 응용을 classification, segmentation, image-to-image translation, detection, registration, video-based application으로 정리했다.
2. self-attention, multi-head attention, encoder-decoder, ViT, DeiT, Swin Transformer 같은 배경 개념을 의료영상 독자 관점에서 다시 설명했다.
3. pure Transformer와 hybrid Transformer를 구분해 실제 의료 응용 구조를 비교했다.
4. weakly-supervised learning, multi-task learning, multi-modal learning 같은 실제 임상 데이터 설정과 Transformer를 연결했다.
5. 모델 효율성, 사전학습, 복잡한 학습 시나리오, 해석성과 안전성 등 향후 방향을 명시적으로 짚었다.

## 방법론 요약

### 1. Transformer 기초 설명

논문은 먼저 self-attention을 element-wise와 matrix-wise 방식으로 설명하고, 이를 기반으로 multi-head self-attention, Transformer encoder-decoder, ViT 계열 구조를 정리한다.

배경 설명의 핵심은 다음과 같다.

- **Self-attention**: 입력 토큰들 사이의 관계를 직접 계산해 long-range dependency를 모델링
- **Multi-head attention**: 서로 다른 표현 subspace에서 관계를 병렬적으로 포착
- **ViT**: 이미지를 patch sequence로 바꿔 sequence prediction 문제처럼 다룸
- **DeiT**: data-efficient training과 distillation을 통해 ViT의 데이터 요구량을 낮춤
- **Swin Transformer**: window attention과 shifted window로 계산량을 줄이고 계층적 표현을 만듦

저자들은 이 배경 설명을 통해 의료영상에서 Transformer가 왜 의미 있는지, 그리고 왜 pure ViT보다는 변형 구조가 중요해졌는지를 설명한다.

### 2. 논문의 분류 관점

이 리뷰는 task taxonomy뿐 아니라 구조적 관점도 함께 쓴다.

- **pure Transformer**
- **hybrid Transformer**
- **Transformer + CNN**
- **Transformer + graph**

즉, 단지 어디에 적용됐는지가 아니라 "어떤 기존 inductive bias와 결합했는가"까지 정리하려는 성격이 강하다.

## 응용 분야별 정리

### 1. Classification

논문은 classification을 가장 먼저 다루며, 크게 세 방향으로 정리한다.

1. ViT를 의료영상에 직접 적용하는 pure Transformer
2. CNN과 ViT를 결합해 local feature와 global dependency를 동시에 활용하는 hybrid 구조
3. graph representation과 Transformer를 결합해 복잡한 관계형 의료 데이터를 다루는 방식

이 분야에서 다뤄지는 modality와 task는 매우 넓다. CT 기반 COVID-19, X-ray, MRI, ultrasound, OCT, histopathology, fundus image, graph-based brain analysis 등이 모두 포함된다.

저자들이 특히 강조하는 흐름은 다음과 같다.

- COVID-19 진단처럼 빠르게 등장한 응용에서 pure ViT가 경쟁력 있는 결과를 보였다.
- 실제 성능과 안정성 면에서는 CNN과 Transformer를 결합한 hybrid 구조가 많이 채택됐다.
- histopathology나 brain graph처럼 관계 구조가 중요한 문제에서는 Transformer가 aggregation 도구로 유리했다.

논문 말미에서 classification에 대해 내리는 결론도 분명하다.

- 많은 과제에서 CNN 대비 comparable 혹은 better performance
- 대규모 데이터 요구량이 여전히 크다
- 계산량 문제 때문에 경량화가 중요하다
- hybrid Transformer가 점점 더 주목받는다

### 2. Segmentation

segmentation은 이 논문에서도 매우 큰 비중을 차지한다. abdominal multi-organ, cardiac, pancreas, brain tumor/tissue, polyp, liver lesion, kidney tumor, skin lesion, prostate, gland, nucleus, cell, retinal vessel 등 매우 다양한 segmentation task가 정리된다.

논문은 segmentation을 크게 두 축으로 나눈다.

- **Hybrid Transformers**
- **Pure Transformers**

hybrid 계열에서는 U-Net 류 encoder-decoder와 Transformer를 결합하는 방식이 가장 두드러진다. 저자들은 이 방향이 자연스러운 이유를 local feature extraction은 convolution이 잘하고, 전역 문맥과 장거리 상호작용은 Transformer가 잘하기 때문이라고 본다.

이 분야에서 논문이 제시하는 핵심 포인트는 다음과 같다.

- 의료 segmentation은 local boundary와 global context가 동시에 중요하다.
- Transformer는 non-local relation modeling에 강점이 있다.
- 그러나 의료영상의 고해상도와 3D 구조 때문에 full attention은 계산량이 커서 실제론 hybrid 또는 windowed 구조가 많이 쓰인다.
- 여러 benchmark에서 Dice가 경쟁력 있게 올라가며 빠르게 확산됐다.

### 3. Image-to-Image Translation

논문은 image-to-image translation을 synthesis, super-resolution, denoising 등을 포함하는 더 넓은 범주로 다룬다. 여기에는 multi-modal synthesis, MRI reconstruction, image super-resolution, denoising 등이 포함된다.

저자들은 이 영역에서 Transformer가 전역 구조 보존과 feature relation modeling 측면에서 잠재력이 크다고 본다. 특히 MRI synthesis나 multi-modal synthesis에서는 modality 간 대응 관계를 학습하는 데 attention이 유용하다고 해석한다.

동시에 paired data 부족과 inter-subject variability를 대표적 어려움으로 든다. 그래서 unsupervised 또는 zero-shot transformer 기반 접근도 주목할 만한 방향으로 언급된다.

### 4. Detection

이 논문에서 detection은 기술적으로 다루지만, segmentation이나 classification만큼 폭넓게 정리되지는 않는다. 의료 문맥에서 detection은 단순 존재 판단, 위치 추정, lesion detection이 섞여 있기 때문에 용어 자체가 더 넓다고 설명한다.

여기서의 핵심은 DETR 계열 아이디어가 의료 detection에 옮겨오고 있다는 점이다. 다만 survey 전체 톤을 보면 detection은 아직 초기 단계이자 상대적으로 데이터셋과 구조 탐색이 부족한 분야로 평가된다.

### 5. Registration

저자들은 registration에서 Transformer의 self-attention이 moving image와 fixed image 사이의 더 정교한 spatial mapping을 가능하게 한다고 본다. 특히 deformable registration에서 장거리 correspondence를 포착하는 점이 장점으로 제시된다.

다만 이 분야 역시 survey 시점에서는 아직 활발한 초창기 단계에 가깝고, 확실한 표준 구조가 자리 잡은 상태는 아니라고 해석된다.

### 6. Video-based Applications

이 논문은 medical imaging review이지만 video-based application도 별도 범주로 다룬다. surgical phase recognition, surgical tool detection, ultrasound video analysis, trajectory forecasting 같은 문제들이 포함된다.

이 부분은 Transformer가 image analysis를 넘어 temporal modeling에도 자연스럽게 확장된다는 점을 보여준다.

## 다양한 학습 패러다임에 대한 논의

이 논문의 특징은 단순 task 분류를 넘어, Transformer를 실제 임상 데이터 환경과 연결해 보는 섹션이 따로 있다는 점이다.

### 1. Multi-task learning

의료영상에서는 classification과 segmentation, detection을 분리해서 풀기보다 함께 푸는 것이 더 실용적인 경우가 많다. 논문은 MT-TransUNet 같은 사례를 들어 Transformer가 여러 task token을 매개로 multi-task setting에도 잘 확장될 수 있다고 본다.

### 2. Multi-modal learning

의료 진단은 보통 한 종류의 영상만 쓰지 않는다. OCT + VF, multi-phase MRI, imaging + clinical variable 같은 조합이 흔하다. 저자들은 Transformer가 modality 간 관계를 attention으로 모델링하기 좋기 때문에 multi-modal fusion에 특히 적합하다고 본다.

### 3. Weakly-supervised learning

ROI annotation이 비싸고 patch-level label만 있는 pathology 같은 문제에서는 weak supervision이 중요하다. 논문은 Transformer가 instance 사이 관계를 모델링해 weakly supervised multiple instance learning에 유리하다고 본다.

이 부분은 survey 전체에서 꽤 중요한 메시지다. Transformer의 가치가 단지 architecture novelty가 아니라, annotation이 불완전한 임상 현실과도 잘 맞는다는 주장으로 이어지기 때문이다.

## 실험 설정과 결과

이 논문은 survey라 자체 실험을 수행하지 않는다. 대신 여러 table을 통해 대표 모델, 데이터셋, 성능, highlight를 정리한다. 특히 classification과 segmentation에 대한 표가 비교적 촘촘하다.

### 1. Classification 결과

classification table은 CT, X-ray, MRI, histology, skin, eye, ultrasound 등 다양한 modality에서 Transformer 계열 모델이 사용되었음을 보여준다. 일부 과제에서는 90%대 accuracy가 반복적으로 나타나며, COVID-19, breast ultrasound, leukemia, gastric pathology, melanoma, fundus disease 등 폭넓은 응용이 제시된다.

이 표의 의미는 단일 SOTA 수치보다, Transformer가 modality-specific model이 아니라 범용 backbone 또는 fusion mechanism으로 쓰인다는 데 있다.

### 2. Segmentation 결과

segmentation 표는 abdominal multi-organ, cardiac, brain tumor, polyp, skin lesion 등에서 Transformer 기반 구조가 높은 Dice를 기록함을 보여준다. TransUNet, CoTr, UNETR, Swin UNETR 같은 구조가 등장하며, CNN 기반 U-shape backbone과 결합된 hybrid 구조가 특히 많다.

결과적으로 이 논문은 "Transformer가 segmentation에서 실제로 바로 임팩트를 냈다"는 메시지를 강하게 전달한다.

## 한계 및 향후 연구 가능성

논문 후반부는 미래 방향을 직접 한 장으로 정리하기보다는 여러 섹션에 분산된 논의를 통해 남은 문제를 드러낸다.

### 1. 데이터 요구량

저자들은 Transformer가 CNN보다 대규모 데이터에 더 의존한다고 본다. 의료영상처럼 표본 수가 적고 라벨 비용이 큰 분야에서는 이 점이 핵심 병목이다. 따라서 pretraining, self-supervised learning, transfer learning이 중요해진다.

### 2. 계산 효율성

의료영상은 원본 해상도가 크고 3D 데이터가 많다. 따라서 attention의 계산 비용이 더 직접적인 문제로 드러난다. 논문은 경량 모델, attention complexity reduction, token reduction 같은 방향의 중요성을 강조한다.

### 3. 해석 가능성과 안전성

직접적인 별도 대형 섹션은 아니지만, classification과 임상 적용 논의 전반에서 interpretability, quantification, safety가 아직 부족하다고 반복해서 언급된다.

### 4. 고급 학습 시나리오 부족

저자들은 weakly-supervised, multi-modal, multi-task learning 같은 고급 설정이 중요하지만, 당시까지는 아직 사례 수가 적다고 본다. 즉, 구조 자체보다 "임상 현실을 반영한 학습 문제"로의 확장이 더 필요하다는 의미다.

## 실무적 또는 연구적 인사이트

### 1. 이 논문은 task보다 학습 시나리오를 잘 드러낸다

`Transformers in Medical Imaging: A Survey`가 훨씬 넓고 대규모라면, 이 논문은 좀 더 "의료영상 분석 관점"에 맞춰 task와 learning paradigm을 함께 본다는 점이 특징이다. 그래서 실제 연구 설계를 고민할 때 더 직접적인 통찰을 주는 부분이 있다.

### 2. hybrid가 사실상의 기본값으로 보인다

이 논문 전체를 읽으면 pure ViT도 소개되지만, 실제 의료영상에서는 CNN과 결합한 hybrid 구조가 압도적으로 많다는 점이 명확하다. 이는 의료영상이 여전히 local texture, boundary, fine-scale structure에 민감하기 때문이며, Transformer만으로 대체하기보다 보완적으로 쓰이는 경우가 많다는 뜻이다.

### 3. segmentation과 classification이 주도한다

survey 시점에서 Transformer 확산은 segmentation과 classification이 가장 앞서 있었다. detection, registration, video, multi-task 등은 가능성은 보이지만 아직 성숙하지 않았다는 인상이 강하다.

### 4. 논문이 제안한 미래 방향은 여전히 유효하다

데이터 효율성, 사전학습, 모델 경량화, 해석 가능성, weak supervision, multimodal fusion은 지금도 핵심 연구 과제다. 즉, 기술 이름은 바뀌어도 문제 구조는 크게 달라지지 않았다.

### 5. 현재 시점에서는 보완 독해가 필요하다

이 논문은 2022년 기준 survey라서 이후의 foundation model, medical VLM, CLIP 계열 대형 모델, SAM 기반 분할, diffusion 중심 생성 모델의 급팽창은 반영하지 못한다. 따라서 현재 연구자가 사용할 때는 "Transformer 의료영상 응용의 초기 구조화 문헌"으로 보는 편이 맞다.

## 종합 평가

이 논문은 Transformer를 의료영상 분석에 적용하는 흐름을 비교적 이른 시점에 정리한 review로서, classification, segmentation, translation, detection, registration, video 분석까지 폭넓게 포괄한다. 특히 task taxonomy뿐 아니라 multi-task, multi-modal, weakly-supervised learning 같은 실제 의료 AI 문제 설정을 함께 다룬다는 점이 강점이다.

오늘 기준으로는 더 최신의 survey가 필요하지만, 왜 의료영상에서 pure ViT보다 hybrid 설계가 많았는지, 왜 데이터 효율성과 효율성 문제가 핵심이었는지, 그리고 Transformer가 임상 문제의 어떤 층위까지 확장되기 시작했는지를 이해하는 데 여전히 유용하다. 따라서 이 문서는 최신 방법 총람이라기보다, 의료영상 Transformer 연구의 초기 구조와 설계 논리를 정리하는 기준 문헌으로 보는 것이 적절하다.
