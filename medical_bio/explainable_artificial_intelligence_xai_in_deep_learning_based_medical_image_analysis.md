# Explainable Artificial Intelligence (XAI) in Deep Learning-Based Medical Image Analysis

## 논문 메타데이터

- 제목: Explainable Artificial Intelligence (XAI) in deep learning-based medical image analysis
- 저자: Bas H.M. van der Velden, Hugo J. Kuijf, Kenneth G.A. Gilhuijs, Max A. Viergever
- 소속: Image Sciences Institute, University Medical Center Utrecht, Utrecht University
- 논문 성격: Survey
- arXiv: [2107.10912](https://arxiv.org/abs/2107.10912)
- 핵심 키워드: XAI, medical image analysis, deep learning, survey

## 연구 배경 및 문제 정의

딥러닝은 의료 영상 분류, 탐지, 분할 등에서 빠르게 성능을 높였지만, 모델이 왜 그런 결정을 내렸는지를 설명하기 어렵다는 문제가 계속 제기되어 왔다. 특히 의료 분야는 진단과 치료가 직접 연결되는 고위험 영역이기 때문에, 단순한 예측 정확도만으로는 임상 현장에서 신뢰를 얻기 어렵다. 저자들은 이 지점을 출발점으로 삼아, 의료 영상 딥러닝에서 사용되는 XAI 기법들을 체계적으로 정리하고자 한다.

이 논문은 단순히 saliency map 몇 가지를 소개하는 수준의 개관이 아니라, 의료 영상 분야에서 XAI를 어떤 기준으로 분류하고, 어떤 해부학적 부위와 영상 modality에서 주로 사용해 왔으며, 실제로 설명 품질은 어떻게 평가할 수 있는지를 함께 다룬다. 즉 "어떤 설명 기법이 존재하는가"뿐 아니라 "무엇을 설명이라고 부를 것인가"와 "그 설명을 어떻게 검증할 것인가"까지 한 프레임으로 묶는다.

## 논문의 핵심 기여

- 의료 영상 딥러닝용 XAI를 체계적으로 정리하는 survey 프레임워크를 제시한다.
- XAI 기법을 `model-based vs post hoc`, `model-specific vs model-agnostic`, `global vs local`의 세 축으로 분류한다.
- 실제 응용 논문들을 `visual explanation`, `textual explanation`, `example-based explanation`으로 나누어 정리한다.
- 해부학적 부위와 영상 modality 기준으로 적용 현황을 함께 정리해, 분야 편중 양상을 보여 준다.
- XAI 평가를 `application-grounded`, `human-grounded`, `functionally-grounded`로 구분해 설명 품질 검증 문제를 명시한다.
- XAI에 대한 비판과 향후 연구 방향까지 포함해, 단순 기술 목록이 아니라 연구 의제를 제시한다.

## 조사 범위와 전체 구성

저자들은 systematic review 절차, 동료 논의, snowballing을 통해 논문을 수집했고, 최종적으로 223편의 논문을 정리했다고 밝힌다. 조사 대상은 딥러닝 기반 의료 영상 분석에 한정되며, 설명 기법은 크게 세 범주로 정리된다.

- visual explanation
- textual explanation
- example-based explanation

이 분류는 단순 인터페이스 차원이 아니라, 설명이 어떤 형태로 사용자에게 전달되는지를 기준으로 한 실용적 taxonomy다. 의료 현장에서는 heatmap, 텍스트 보고서, 유사 사례 제시가 모두 다른 방식의 신뢰 형성에 기여하기 때문에 이 구분이 유용하다.

## XAI 프레임워크

논문이 가장 먼저 제시하는 핵심은 XAI를 분류하는 세 가지 기준이다.

### 1. Model-based vs Post hoc

- model-based explanation은 모델 구조 자체가 해석 가능하도록 설계된 경우를 뜻한다.
- post hoc explanation은 이미 학습된 모델을 사후적으로 분석해 설명을 생성하는 경우를 뜻한다.

의료 영상 딥러닝에서는 대부분의 모델이 복잡한 CNN 또는 유사 딥러닝 구조이기 때문에, 실제 응용의 대부분은 post hoc explanation에 속한다. 저자들도 surveyed papers의 다수가 이 범주에 속한다고 정리한다.

### 2. Model-specific vs Model-agnostic

- model-specific explanation은 특정 구조, 예를 들어 CNN의 gradient나 feature map을 직접 활용한다.
- model-agnostic explanation은 모델 내부 구조를 크게 가정하지 않고 입력-출력 관계를 통해 설명을 만든다.

Grad-CAM, CAM, guided backpropagation 같은 기법은 전형적인 model-specific 접근이고, LIME이나 SHAP 계열은 상대적으로 model-agnostic 성격이 강하다.

### 3. Global vs Local

- global explanation은 모델이 데이터셋 수준에서 어떤 규칙을 학습했는지를 설명한다.
- local explanation은 특정 환자 또는 특정 샘플에 대해 왜 그런 예측이 나왔는지를 설명한다.

의료 영상 응용에서는 환자 단위 의사결정이 중요하기 때문에 local explanation이 훨씬 많이 사용된다. 이 논문도 실제 surveyed papers의 주류가 local explanation이라고 요약한다.

## Visual Explanation

visual explanation은 의료 영상 XAI에서 가장 널리 쓰이는 형태이며, saliency mapping이 중심을 이룬다. 저자들은 이를 gradient-based와 perturbation-based 계열로 나누어 설명한다.

### 1. Gradient-based visual explanation

주요 기법은 다음과 같다.

- guided backpropagation / deconvolution
- CAM
- Grad-CAM
- Deep SHAP
- attention 기반 시각화

이 중 Grad-CAM은 survey 전반에서 가장 자주 등장하는 대표 기법이다. 분류기가 특정 질환 또는 이상 소견을 판단할 때 영상의 어느 영역을 중시했는지를 heatmap으로 보여 줄 수 있기 때문이다. 실제 논문 표를 보면 brain MRI, chest X-ray, CT, fundus, histology, ultrasound 등 매우 다양한 modality에서 Grad-CAM이 반복적으로 사용된다.

논문이 지적하는 중요한 점은, 의료 영상에서 visual explanation은 단순 시각화가 아니라 "의학적으로 타당한 영역을 모델이 보고 있는가"를 확인하는 도구로 사용된다는 것이다. 예를 들어 뇌종양 분류에서 종양 주변부를 가리키는지, 흉부 X-ray에서 병변 영역을 강조하는지, 병리 영상에서 암세포 집합을 강조하는지 등이 핵심이다.

### 2. Perturbation-based visual explanation

이 범주에는 다음이 포함된다.

- occlusion sensitivity
- LIME
- meaningful perturbation
- prediction difference analysis

이들 방법은 입력의 일부를 가리거나 변형해 출력 변화량을 측정함으로써 중요 영역을 추정한다. gradient-based 방법보다 계산량이 크지만, 특정 구현에서는 모델 내부 gradient에 직접 의존하지 않기 때문에 보완적 의미를 가진다.

저자들은 특히 prediction difference analysis와 multiscale supervoxel 기반 접근이 의료영상 구조와 더 잘 맞을 수 있음을 언급한다. 의료 영상은 해부학적 경계와 병변 형태가 중요하므로, 픽셀 단위보다 region 또는 supervoxel 단위 설명이 더 자연스러울 수 있기 때문이다.

## Textual Explanation

textual explanation은 설명을 사람이 읽을 수 있는 문장, 개념, 보고서 형태로 제공한다는 점에서 visual explanation과 구별된다. 논문은 이를 세 가지로 나눈다.

### 1. Image Captioning

영상 인코더와 LSTM 기반 텍스트 디코더를 결합해 의료 영상을 자연어 문장으로 설명하는 방식이다. chest X-ray 보고서 생성이 대표적이며, BLEU, METEOR, CIDEr, ROUGE 같은 자연어 생성 평가 지표가 주로 사용된다.

이 접근은 모델이 어떤 소견을 언어적으로 표현하는지 보여 준다는 점에서 단순 heatmap보다 임상 친화적이다. 다만 텍스트 품질이 좋아 보여도 실제 진단 논리와 정확히 맞는지는 별도 검증이 필요하다.

### 2. Image Captioning with Visual Explanation

일부 연구는 텍스트 생성과 시각적 attention map을 결합한다. 이렇게 하면 생성된 문장의 각 부분이 영상의 어느 영역과 대응되는지를 함께 제시할 수 있다. 논문은 histology, chest X-ray, mammography 등에서 이런 복합형 설명이 사용됐다고 정리한다.

이 범주는 의료 현장에서 특히 유망하다. radiology report와 영상 영역을 연결할 수 있기 때문에 "이 문장은 어디를 근거로 생성되었는가"를 더 직접적으로 확인할 수 있기 때문이다.

### 3. Concept-based explanation: TCAV

Testing with Concept Activation Vectors(TCAV)는 사람이 이해 가능한 고수준 개념을 통해 모델 반응을 측정한다. 예를 들어 fundus 영상에서 `microaneurysm`, 병리 영상에서 `nuclei area`, 심장 MRI에서 특정 기능 지표 같은 개념이 실제 분류에 얼마나 기여했는지를 볼 수 있다.

이 논문은 TCAV의 장점을 "heatmap보다 더 직접적으로 인간 개념과 연결된다"는 점에서 본다. 특히 regression concept vector처럼 연속값 개념을 다루는 확장은 의료 영상처럼 크기, 조영, 세포 밀도 같은 연속 특성이 중요한 분야에 적합하다.

### 4. Other textual explanation

일부 연구는 영상에서 radiologist가 사용하는 서술형 특성 자체를 직접 예측하게 만든다. 예를 들어 폐결절 악성도 예측과 함께 `spiculation`, `texture`, `margin` 같은 설명 가능한 중간 서술자를 예측하는 방식이다. 이는 완전한 free-form 텍스트 생성보다 더 구조적이고 임상 용어 중심이라는 장점이 있다.

## Example-Based Explanation

example-based explanation은 현재 입력과 유사하거나 대조적인 사례를 제시해 설명을 제공한다. 의사가 "이 환자는 이전에 본 어떤 사례와 비슷하다"라고 판단하는 방식과 유사하기 때문에 임상적 직관과 잘 맞는다.

### 1. Triplet network

triplet loss로 latent space를 학습해 유사 사례를 가까이, 비유사 사례를 멀리 배치한다. 이후 nearest neighbor 검색을 통해 설명 사례를 제공한다. 병리, 피부, 전신 CT lesion retrieval 등에서 활용된다.

### 2. Influence functions

특정 예측이 훈련 데이터의 어떤 샘플들에 의해 가장 큰 영향을 받았는지를 근사적으로 계산한다. 이는 단순한 유사 사례 제시를 넘어, 모델이 실제로 어떤 training example을 근거로 의사결정했는지를 해석하는 방식이다.

### 3. Prototypes

대표 사례(prototype)를 모델 내부에 직접 포함해 `this looks like that` 형태의 설명을 제공한다. 논문은 이 방식을 특히 중요하게 다룬다. prototype 설명은 post hoc approximation이 아니라 모델의 실제 계산 과정 일부라는 점에서, Rudin식 비판에 대응하는 model-based 설명의 성격을 부분적으로 갖기 때문이다.

### 4. Latent space example / disentangled representation

VAE, ladder VAE, capsule network 등을 사용해 latent space를 사람이 해석 가능한 축으로 구조화하고, 그 공간에서 유사 사례나 생성적 변형을 보여 주는 방식이다. 이 접근은 단순 retrieval보다 더 풍부한 dataset-level explanation을 줄 수 있다.

## 의료 영상 분야 적용 경향

논문이 정리한 분포를 보면, XAI 연구는 특정 부위와 modality에 집중되어 있다.

- 해부학적 위치는 chest와 brain에 편중되어 있다.
- modality는 X-ray와 MRI 비중이 높다.
- visual explanation, 특히 Grad-CAM 계열이 압도적으로 많이 쓰인다.

이는 의료영상 딥러닝 일반 동향과도 유사하다. 공개 데이터가 많고 벤치마크가 성숙한 흉부 X-ray와 brain MRI에서 먼저 XAI 적용이 활발해졌고, 설명 방법도 구현이 쉬운 saliency map 중심으로 발전했다는 해석이 가능하다.

## XAI 평가 프레임워크

이 논문의 또 다른 핵심은 "설명을 어떻게 평가할 것인가"를 별도 섹션으로 정리했다는 점이다. 저자들은 Doshi-Velez와 Kim의 틀을 따라 세 가지 평가 방식을 소개한다.

### 1. Application-grounded evaluation

실제 임상 맥락에서 전문가가 설명을 평가하는 방식이다. 예를 들어 영상의학과 전문의가 saliency map이나 유사 사례 설명을 보고, 실제 임상적으로 타당한지 판단한다. 가장 직접적이지만 비용이 높다.

### 2. Human-grounded evaluation

전문가 대신 일반 사용자 또는 단순화된 과업을 통해 설명을 평가한다. 비용은 줄어들지만, 실제 임상 타당성을 완전히 대체하지는 못한다.

### 3. Functionally-grounded evaluation

인간 실험 대신 proxy metric으로 설명 품질을 평가한다. 예를 들어 saliency map을 병변 분할 마스크와 비교하거나, localization 성능으로 설명 품질을 근사한다. 저자들은 이 방식이 저렴해 보일 수 있지만, 의료 영상에서는 정답 annotation 자체가 비싸다는 점도 함께 지적한다.

이 구분은 중요하다. 많은 의료영상 XAI 논문이 시각적으로 그럴듯한 결과만 제시하지만, 실제로는 어떤 수준의 평가가 수행되었는지 명확하지 않은 경우가 많기 때문이다.

## 비판과 한계

저자들은 XAI를 낙관적으로만 보지 않는다. 특히 Rudin(2019)과 Adebayo et al.(2018)의 비판을 바탕으로, 설명 가능한 것처럼 보이는 시각화가 실제 모델의 의사결정과 충실하게 연결되어 있지 않을 수 있음을 강조한다.

핵심 비판은 다음과 같다.

- post hoc explanation은 모델이 실제 계산한 것과 완전히 동일하지 않을 수 있다.
- saliency map은 시각적으로 설득력 있어 보여도, 모델 파라미터나 라벨이 바뀌어도 비슷하게 나올 수 있다.
- explanation fidelity와 clinical usefulness는 별개의 문제다.
- 일부 방법은 단순히 edge를 강조하거나 입력 구조를 반영할 뿐, 진짜 의학적 reasoning을 드러내지 않을 수 있다.

특히 guided backpropagation과 guided Grad-CAM에 대한 randomization test 논의는 이 survey의 중요한 경고다. 의료 영상 분야에서도 설명 시각화를 그대로 신뢰해서는 안 되며, robustness와 faithfulness를 별도로 검증해야 한다는 메시지를 준다.

## 향후 연구 방향

저자들은 몇 가지 유망한 방향을 제시한다.

### 1. 다중 설명 형태의 결합

텍스트와 시각 설명을 결합하거나, example-based explanation과 saliency map을 결합하는 식의 holistic XAI가 늘어날 것으로 본다. 실제로 의료 현장에서는 하나의 heatmap보다, 보고서 문장과 근거 영역, 유사 사례가 함께 제시되는 쪽이 더 유용할 가능성이 높다.

### 2. Biological explanation

영상 특징과 분자 아형, 유전적 경로, 생물학적 기전 간 연결을 설명하는 방향이 중요하다고 본다. 현재는 biological target을 예측하는 수준이 많지만, 장기적으로는 영상 phenotype와 생물학적 pathway 간 관계를 설명하는 수준으로 확장되어야 한다는 제안이다.

### 3. Causality와 XAI의 결합

현재 의료영상 딥러닝은 상관관계를 주로 학습한다. 그러나 임상 신뢰성과 일반화를 높이려면 causal reasoning을 설명 체계에 통합할 필요가 있다. 이는 데이터 편향, 센터 간 분포 차이, 기기 차이 문제를 다루는 데도 중요하다.

## 비판적 평가

이 논문은 2021년 시점의 의료영상 XAI 분야를 정리하는 데 매우 유용하다. 특히 많은 survey가 saliency map 사례 중심으로 끝나는 반면, 이 논문은 분류 프레임워크, 예시 논문 목록, 평가 방법, 비판, 미래 방향까지 포함해 연구 지형을 더 넓게 보여 준다.

강점은 다음과 같다.

- XAI를 형태별로 체계적으로 나누어 설명한다.
- 단순 visual explanation에 머물지 않고 textual, example-based explanation까지 함께 다룬다.
- 평가 문제를 별도 섹션으로 다뤄, 설명의 검증 가능성을 강조한다.
- XAI 비판을 survey 내부에 포함해 과도한 낙관론을 피한다.

한계도 있다.

- 2021년 이전 연구가 중심이어서 이후 급격히 늘어난 foundation model, multimodal LLM 기반 의료 XAI 흐름은 반영되지 않는다.
- 223편을 폭넓게 정리한 대신, 각 방법의 정량 비교는 상대적으로 얕다.
- 분할, 탐지, 생성, 보고서 작성 등 과업별 성숙도 차이를 깊게 비교하기보다는 기법 taxonomy 중심으로 정리한다.

## 연구적 시사점

이 논문이 주는 핵심 메시지는 명확하다.

- 의료영상 XAI의 주류는 여전히 post hoc, local, model-specific 설명이다.
- 하지만 실제 임상 신뢰를 위해서는 explanation fidelity, robustness, evaluation protocol이 더 중요해질 것이다.
- saliency map은 출발점일 뿐이며, 개념 기반 설명, 텍스트 설명, 사례 기반 설명으로 확장되어야 한다.
- 궁극적으로는 의료진의 추론 방식과 닮은 설명, 즉 근거 영역, 언어적 서술, 유사 사례, 생물학적 맥락이 함께 제공되는 형태가 유망하다.

## 종합 평가

`Explainable Artificial Intelligence (XAI) in deep learning-based medical image analysis`는 의료영상 XAI 분야의 초기 기준점 역할을 하는 survey다. 어떤 설명 기법이 있는지 정리하는 데 그치지 않고, 설명의 범주화, 적용 현황, 평가 방식, 비판적 쟁점을 하나의 구조로 연결했다는 점이 강점이다.

특히 이 논문은 의료영상에서 XAI를 단순 시각화 도구가 아니라, 임상 신뢰성 검증의 한 요소로 바라보게 만든다. 이후 의료용 foundation model이나 multimodal model의 설명 가능성을 논할 때도, 이 논문이 제시한 local/global, post hoc/model-based, fidelity/evaluation 문제는 여전히 유효한 기준으로 남는다.
