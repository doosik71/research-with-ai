---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Transformers in Medical Imaging: A Survey

- Fahad Shamshad et al.
- 2022 arXiv / 2023 Medical Image Analysis
- 의료영상 Transformer 연구 전반을 크게 정리한 대규모 survey

---

## 문제 배경

- Vision Transformer 계열 모델이 자연영상에서 성공한 뒤 의료영상으로 빠르게 확장됐다.
- 의료영상은 CT, MRI, X-ray, ultrasound, pathology, PET 등 modality가 매우 다양하다.
- 과제도 segmentation, classification, detection, reconstruction, synthesis, registration, report generation까지 넓다.
- Transformer는 global self-attention으로 장점을 보이지만, 의료영상에서는 데이터 부족, 3D 입력, 계산 비용, 해석 가능성 문제가 더 크다.
- 이 survey는 Transformer가 의료영상 어디까지 확장됐는지를 넓게 정리한다.

---

## 이 survey의 핵심 기여

- 당시 기준 125편 이상의 Transformer 기반 의료영상 논문을 폭넓게 정리한다.
- `segmentation`, `classification`, `detection`, `reconstruction`, `synthesis`, `registration`, `clinical report generation`으로 taxonomy를 제시한다.
- hand-crafted method -> CNN -> ViT로 이어지는 기술 배경을  정리한다.
- task별 설계, dataset, challenge, trend를 구조적으로 비교한다.
- `pre-training`, `interpretability`, `adversarial robustness`, `edge deployment`, `federated learning`, `domain adaptation/OOD`를 open challenge로 제시한다.

---

## 이 논문의 관점

- 이 survey의 초점은 개별 모델 제안이 아니다.
- 질문은 다음에 가깝다.
- 의료영상에서 Transformer가 실제로 어느 task에 퍼졌는가?
- 어떤 구조가 유리했는가?
- CNN 대비 장점과 한계는 무엇인가?
- 즉, 이 문서는 초기 medical Transformer 생태계의 로드맵이다.

---

## Transformer의 장점과 병목

- 장점:
  - global context modeling
  - long-range dependency capture
  - modality 간 관계와 넓은 spatial relation 처리
- 병목:
  - quadratic attention cost
  - large-scale pre-training 필요성
  - 3D volume에서 메모리 부담
  - clinical interpretability 부족

---

## 1. Segmentation

- 이 survey에서 가장 큰 비중을 차지하는 분야다.
- organ-specific, multi-organ, 2D, 3D segmentation을 모두 포괄한다.
- 대표 구조:
  - `TransUNet`, `CoTr`, `UNETR`, `Swin UNETR`
- 핵심 메시지는 분명하다.
  - 의료 segmentation에서는 pure ViT보다
    CNN encoder-decoder와 Transformer를 결합한 hybrid가 훨씬 많이 쓰였다.

---

## 2. Classification

- COVID-19, tumor, retinal disease, pathology, breast ultrasound 등 다양한 응용을 다룬다.
- classification은 단순 정확도 경쟁만이 아니라
  black-box model과 interpretable model 구도도 함께 본다.
- pathology WSI에서는 weakly supervised MIL + Transformer가 특히 중요하다.
- retinal disease에서는 lesion-aware transformer가 주목된다.
- 즉, classification에서도 Transformer는 backbone 그 이상으로 relation modeling 도구로 쓰이기 시작했다.

---

## 3. Detection

- DETR 계열 아이디어가 polyp detection, lymph node detection 등으로 확장되기 시작한 시기다.
- 그러나 survey 기준으로 detection은 아직 초기 단계다.
- dataset과 benchmark 성숙도가 segmentation만큼 높지 않았다.
- 이 분야는 가능성은 보였지만 당시엔 아직 탐색기였다.

---

## 4. Reconstruction

- reconstruction은 이 survey에서 의외로 중요한 영역이다.
- 예시:
  - low-dose CT enhancement
  - low-dose PET enhancement
  - undersampled MRI reconstruction
  - sparse-view CT reconstruction
- Transformer는 전역 prior와 구조 보존 측면에서 장점을 보인다.
- 특히 low-data regime에서 pretraining 또는 zero-shot prior가 중요한 축으로 제시된다.

---

## 5. Synthesis

- synthesis는 intra-modality와 inter-modality로 나눠 다룬다.
- super-resolution, enhancement, modality translation이 포함된다.
- paired data 부족 때문에 semi-supervised, unsupervised 설정이 중요하다.
- 논문은 synthesis를 단순 생성 문제가 아니라
  missing modality 보완과 downstream task 지원 도구로 해석한다.

---

## 6. Registration과 Report Generation

- registration은 long-range correspondence modeling 측면에서 Transformer가 유망하다.
- 하지만 review 시점에는 아직 초기 단계다.
- report generation은 image encoder + Transformer decoder로 radiology report를 생성하는 흐름을 다룬다.
- 이 부분은 Transformer가 pure vision을 넘어
  multimodal reasoning과 clinical language generation으로 확장된다는 점에서 중요하다.

---

## 핵심 해석 1: Hybrid가 실용적 기본값

- survey 전체를 보면 pure Transformer보다 hybrid 구조가 훨씬 많다.
- 이유는 의료영상이 local texture, boundary, anatomy prior에 민감하기 때문이다.
- 따라서 실제 adoption은 `CNN을 없애는 방향`이 아니라
  `CNN의 local bias와 Transformer의 global modeling을 결합하는 방향`이었다.

---

## 핵심 해석 2: Segmentation이 가장 빠르게 성장

- segmentation은 데이터와 benchmark가 비교적 정리되어 있었고,
  local-global tradeoff가 분명해서 Transformer의 효과를 보여주기 좋았다.
- classification도 넓게 확산됐지만,
  detection과 registration은 아직 연구 공간이 더 컸다.
- reconstruction과 synthesis는 inverse problem과 Transformer prior의 결합으로 주목받았다.

---

## Open Challenges

- `Pre-training`: domain-specific SSL과 medical pretraining의 필요성
- `Interpretability`: attention map만으로는 clinical 설명 가능성이 충분하지 않음
- `Adversarial robustness`: 의료영상에서 체계적 검증 부족
- `Edge deployment`: portable/point-of-care 환경에서 경량화 필요
- `Federated learning`: 병원 간 데이터 공유 제한과 privacy 문제
- `Domain adaptation / OOD detection`: 병원, 장비, 프로토콜 차이에 대한 강건성 필요

---

## 현재 시점에서의 해석

- 이 survey는 foundation model, SAM, medical VLM, diffusion 이후 흐름은 반영하지 못한다.
- 대신 초기 transformer medical imaging 연구의 전체 지도를 제공한다.
- 특히 흥미로운 점은 이 논문이 제시한 open challenge 대부분이
  이후 실제 핵심 연구 주제가 되었다는 것이다.
- 그래서 최신 survey는 아니지만,
  초기 구조와 병목을 이해하는 데 여전히 유효하다.

---

## 발표용 핵심 메시지

- 이 논문은 medical Transformer 연구의 초기 대형 로드맵이다.
- pure ViT보다 hybrid 구조가 실용적 중심이었다.
- segmentation이 가장 빠르게 확산됐고,
  classification, reconstruction, synthesis가 뒤를 이었다.
- open challenge로 제시한 `pretraining`, `interpretability`, `FL`, `OOD`, `edge deployment`는 지금도 핵심 과제다.

---

## 결론

- `Transformers in Medical Imaging: A Survey`는
  Transformer 기반 의료영상 연구를 처음으로 대규모 task taxonomy 아래 정리한 문헌이다.
- 강점은 폭넓은 coverage와 open challenge 정리에 있다.
- 현재 기준 최신 문헌은 아니지만,
  medical Transformer 연구가 어디서 시작됐고 어떤 방향으로 확장됐는지 이해하는 데 가치가 크다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/transformers_in_medical_imaging_a_survey_slide.md>
