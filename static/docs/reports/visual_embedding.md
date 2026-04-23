# Visual Embedding

## 서론

### 1. 연구 배경

本报告서는 **"Visual Embedding"** 분야의 연구 동향을 체계적으로 분석하고 정리한다. 시각적 의미 표현 (visual embeddings) 과 공간 정렬 (spatial alignment) 을 동시에 학습하는 범용 vision encoder 를 구축하는 것이 연구의 핵심 주제이며, 이미지/비디오 분류, 검색, MLLM, detection, tracking, depth 추정 등 다양한 다운스트림 작업에서의 활용 가능성을 확보하는 것이 목표이다.

본 보고서가 다루는 연구 범위는 단일 사전학습 목적 함수 하에서 표현의 품질과 일반성을 분석하고, intermediate layer 기반 정렬 전략을 통해 표현 품질을 최적화하는 연구들을 포함한다.

### 2. 문제의식 및 분석 필요성

현재 "Visual Embedding" 분야에서 다양한 다운스트림 작업에서 **단일 범용 vision encoder**를 어떻게 구축할 것인가라는 공통 문제가 존재한다. 복잡한 multi-objective pretraining 으로 각 작업을 위한 목적 함수를 조합하는 방식 대신, 단순한 contrastive learning 으로 다양한 작업 커버 가능성을 입증해야 한다.

또한, 시각적 의미 표현 및 spatial alignment feature 를 효과적으로 학습하고, 언어 정렬 (LLM next-token prediction), frozen layer 정렬 (layer 41 cosine similarity), SAM mask logits 정렬 등 alignment tuning 전략을 적용하여 vision 표현의 downstream 확장성을 증진하는 체계적인 분석이 필요해졌다.

### 3. 보고서의 분석 관점

본 보고서는 다음 세 가지 관점에서 문헌을 정리한다:

- **연구체계 분류**: 연구의 문제 정의와 목적, 접근 방식과 방법론, 시스템 구성과 분석 관점, 적용 대상과 범용성이라는 네 가지 기준에 따라 연구들을 분류
- **방법론 분석**: 단일 목적 함수 기반 표현 분석 계열, intermediate layer 활용 패턴, alignment tuning 기반 확장 패턴, progressive resolution training 패턴 등을 분석하고 비교
- **실험결과 분석**: Zero-shot, fine-tuning 방식으로 다양한 작업에 적용 가능한 성능과 트레이드오프를 종합적으로 평가

### 4. 보고서 구성

보고서는 다음 세 장으로 구성된다:

- **1 장 (연구체계 분류)**: 연구 분류를 문제 정의와 목적, 접근 방식과 방법론, 시스템 구성과 분석 관점, 적용 대상과 범용성이라는 기준에 따라 체계화한다. 표현 분석 기반 연구와 정렬 전략 및 최적화 연구로 구분하여, 각 연구의 분류 근거와 특징을 정리한다.

- **2 장 (방법론 분석)**: 공통 문제 설정과 접근 구조, 방법론 계열 분류, 핵심 설계 패턴 분석을 통해 방법론적 특징을 분석한다. 단일 목적 함수 기반 접근의 문제 접근 방식, 구조/모델 차이, 적용 대상 차이, 트레이드오프를 비교하고, 방법론의 흐름과 진화 경향을 제시한다.

- **3 장 (실험결과 분석)**: 각 연구가 달성한 실험 결과와 성능을 분석한다. Progressive resolution 를 통한 robust core pretraining, teacher-student distillation 로 성능 강화, various downstream 작업에서의 적용 가능성 등을 종합적으로 평가한다.

논문 요약문을 분석하여 1장. 연구체계 분류를 작성하겠습니다.

## 1장. 연구체계 분류

### 1. 연구 분류 체계 수립 기준

본 보고서의 연구 분류는 다음 원칙에 기반하여 수립됨.

1. **문제 정의와 연구 목적**: 각 연구가 해결하려는 핵심 문제와 달성하려는 목표를 기준으로 분류
2. **접근 방식과 방법론**: 연구자가 사용한 주요 접근법 (단일 목적 함수 중심 vs multi-objective 등) 을 기준으로 분류
3. **시스템 구성과 분석 관점**: 연구가 시스템을 바라보는 관점 (표층 분석, 표현 레이어 분석, 시스템 구성 등) 을 기준으로 분류
4. **적용 대상과 범용성**: 연구 대상이 특정 작업에 특화되었는지, 범용적인 접근인지에 따라 분류

### 2. 연구 분류 체계

#### 2.1 표현 분석 기반 연구

단일 사전학습 목적 함수 하에서 다양한 downstream 작업에 대한 표현의 품질과 일반성을 분석하고, intermediate layer 기반 정렬 전략을 통해 표현 품질을 최적화하는 연구가 배치됨.

| 분류                                           | 논문명                                                                                     | 분류 근거                                                                                                                                                                                                                        |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 표현 분석 기반 연구 > 단일 목적 함수 중심 접근 | Perception Encoder: The best visual embeddings are not at the output of the network (2025) | 단일 contrastive pretraining 목적 함수에 intermediate layer 분석과 spatial-language alignment 정렬 전략을 결합하여, 복잡한 multi-objective pretraining 없이도 다양한 downstream 작업에서 범용 vision 표현의 일반성을 입증한 연구 |

#### 2.2 정렬 전략 및 최적화 연구

학습된 표현을 downstream 작업에 효과적으로 활용할 수 있도록 출력 레이어 정렬 (alignment tuning) 과 space-time alignment 전략을 적용하여 표현의 downstream 확장성을 증진하는 연구가 배치됨.

| 분류                                                  | 논문명                                                                                     | 분류 근거                                                                                                                                                                                       |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 정렬 전략 및 최적화 연구 > alignment tuning 기반 확장 | Perception Encoder: The best visual embeddings are not at the output of the network (2025) | 언어 정렬 (LLM next-token prediction), frozen layer 정렬 (layer 41 cosine similarity), SAM mask logits 정렬 등 alignment tuning 전략을 적용하여 vision 표현의 downstream 확장성을 최적화한 연구 |

### 3. 종합 정리

제공된 논문 요약에 따르면, 현재 "Visual Embedding" 분야의 연구는 **단일 목적 함수 기반의 contrastive pretraining**과**alignment tuning** 전략을 중심으로 이루어지고 있음. 특히 복잡한 multi-objective 사전학습 없이도 다양한 downstream 작업에 강건한 표현을 학습할 수 있음이 입증되었고,**intermediate layer 기반 정렬 전략** (layer 41 semantic 보존, layer 50→41 정렬, spatial alignment 등) 을 통해 표현의 일반성을 증진시키는 연구가 활발히 진행되고 있음. 이러한 연구 흐름은 vision foundation model 의 효율적인 사전학습과 downstream 활용을 위한 이론적, 실용적 기초를 제공함.

저장 대신, 분석 내용을 stdout 으로만 출력하겠습니다.

## 2 장. 방법론 분석

### 1. 공통 문제 설정 및 접근 구조

#### 1.1 공통 문제 설정

"Visual Embedding" 분야 논문들은 이미지·비디오 분류, 검색, MLLM, detection, tracking, depth 추정 등 다양한 다운스트림 작업에서 **단일 범용 vision encoder**를 어떻게 구축할 것인가를 공통 문제로 삼는다.

- **입력**: 이미지/비디오 이미지 임베딩과 텍스트 캡션
- **출력**: 시각적 의미 표현 및 spatial alignment feature
- **목표**: 다양한 작업에 강건한 범용 vision 표현 학습

#### 1.2 공통 접근 구조

전체 논문들은 다음 3 단계 처리 절차를 공유한다:

1. **Core Pretraining**: 이미지·비디오 대비 데이터로 core encoder 학습
2. **Alignment Tuning**: 언어/공간 작업에 맞게 intermediate layer 정렬
3. **Downstream 적용**: Zero-shot, fine-tuning 방식으로 다양한 작업 적용

```text
입력 (이미지/비디오 + 텍스트) 
    ↓
[Core Pretraining] → Intermediate Layer 학습
    ↓
[Alignment Tuning] → 작업별 정렬 (language/spatial)
    ↓
출력 (시각적 의미 표현) → Zero-shot / Fine-tuning downstream 작업
```

### 2. 방법론 계열 분류

제공된 자료에 기반하여 방법론적 접근을 하나의 계열로 분류한다.

#### 2.1 단일 목적 함수 기반 표현 분석 계열

**계열 정의**:  
단일 contrastive vision-language pretraining objective 에 intermediate layer 기반 정렬 전략을 결합하여, 복잡한 multi-objective pretraining 없이도 다양한 downstream 작업에 강건한 범용 vision 표현을 학습하는 접근법.

**공통 특징**:

- **구조**: ViT 기반 encoder, 50 개 층 중 특정 층을 각 작업용으로 할당 (layer 47: 언어, layer 41: 공간)
- **절차**:  
  1. Progressive resolution 를 통한 robust core pretraining  
  2. 언어/공간 정렬 (alignment tuning)  
  3. Teacher-student distillation 로 성능 강화
- **접근 방식**:  
  - 표현 레이어 분석 관점  
  - task별 표현 품질 최적화 관점  
  - alignment-based downstream 확장 관점  
  - 단일 사전학습 목적 함수 중심 접근 관점

**해당 논문 목록**:

| 방법론 계열                   | 논문명                                                                                     | 핵심 특징                                                                                                      |
| ----------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| 단일 목적 함수 기반 표현 분석 | Perception Encoder: The best visual embeddings are not at the output of the network (2025) | 단일 contrastive pretraining 목적 함수에 intermediate layer 분석과 spatial-language alignment 정렬 전략을 결합 |

### 3. 핵심 설계 패턴 분석

#### 3.1 패턴 1: Intermediate Layer 활용 패턴

**정의**:  
Contrastive learning 으로 학습된 core encoder 의 중간층 (intermediate layer) 에서 다양한 다운스트림 작업에 적합한 표현을 추출하고, alignment tuning 으로 각 작업에 맞게 정렬하는 패턴.

**특징**:

- ViT 기반 encoder 에서 50 개 층 중 특정 층을 각 작업용으로 할당:
  - Layer 47: 언어 작업용 (PE lang)
  - Layer 41: 공간 작업용 (PE spatial)
  - Layer 50→41: 정렬 대상

**적용 논문**:

- Perception Encoder (2025)

#### 3.2 패턴 2: Alignment Tuning 기반 확장 패턴

**정의**:  
LLM 기반 next-token prediction, frozen feature cosine similarity, SAM 기반 mask logits matching 등 alignment loss 를 활용하여 vision 표현의 downstream 확장성을 최적화하는 패턴.

**주요 정렬 방식**:

| 정렬 유형           | 방법                                             | 특징                      |
| ------------------- | ------------------------------------------------ | ------------------------- |
| 언어 정렬           | LLM next-token prediction                        | PE lang 학습              |
| frozen feature 정렬 | Cosine similarity 기반 alignment loss (layer 41) | Semantic 보존             |
| 공간 정렬           | SAM mask logits pairwise cosine matching         | PE spatial 학습           |
| distillation        | KL-divergence distillation                       | Teacher-student 지식 전이 |

**적용 논문**:

- Perception Encoder (2025)

#### 3.3 패턴 3: Progressive Resolution Training 패턴

**정의**:  
Robust core pretraining 을 위해 progressive resolution 전략을 적용하여 다양한 해상도의 이미지/비디오에서 강건한 표현 학습하는 패턴.

**특징**:

- 데이터: 5.4B 이미지·텍스트 페어, 265K 비디오 (120K human-refined caption)
- Progressive resolution 를 통한 robust pretraining
- Teacher-student distillation 로 성능 강화

**적용 논문**:

- Perception Encoder (2025)

### 4. 방법론 비교 분석

#### 4.1 문제 접근 방식

"Visual Embedding" 분야의 방법론은 **단일 목적 함수 기반 접근**이라는 점에서 일관성을 보인다. 복잡한 multi-objective pretraining 으로 각 작업을 위한 목적 함수를 조합하는 방식 대신, 단순한 contrastive learning 으로 다양한 작업 커버 가능성을 입증한다.

| 비교 요소   | 단일 목적 함수 기반 접근                                       |
| ----------- | -------------------------------------------------------------- |
| 목적 함수   | 단일 contrastive vision-language learning                      |
| 데이터 요구 | 대규모 이미지·텍스트 대비 데이터, 잘 정렬된 비디오 캡션 데이터 |
| 출력        | 다양한 downstream 작업에 강건한 표현                           |

#### 4.2 구조/모델 차이

- **모델 구조**: ViT 기반 encoder 가 공통으로 사용됨
- **정렬 전략**: 언어/공간 정렬을 위해 특정 층을 각 작업용으로 할당
- **학습 전략**: Contrastive learning → alignment tuning → distillation

#### 4.3 적용 대상 차이

| 작업 유형                      | 적용 가능성 |
| ------------------------------ | ----------- |
| 이미지/비디오 분류             | Zero-shot   |
| 검색                           | Zero-shot   |
| MLLM (OCR, Chart, Doc QA, VQA) | Fine-tuning |
| Detection                      | Fine-tuning |
| Segmentation                   | Fine-tuning |
| Tracking                       | Fine-tuning |
| Depth estimation               | Fine-tuning |

#### 4.4 트레이드오프

- **장점**:  
  - 단일 목적 함수로 다양한 작업 커버
  - 복잡한 multi-objective pretraining 없이 효율적인 학습
  - Zero-shot, fine-tuning 방식으로 다양한 작업 확장 가능

- **한계**:  
  - 대규모 데이터 요구 (5.4B 이미지·텍스트 페어, 265K 비디오)
  - Teacher-student distillation 로 성능 향상 가능하지만 재현 비용 높음 (G scale 모델 1.88B 파라미터)
  - 단일 layer 기준 best layer 는 잘 scale 하지만 last layer 만 보면 정체됨

### 5. 방법론 흐름 및 진화

제공된 자료의 범위는 2025 년 한 논문으로 제한되므로, 엄밀한 시간 흐름에 따른 방법론 진화를 기술할 수 없다. 다만, 제공된 논문의 방법론적 특징들을 통해 다음 흐름을 관찰할 수 있다:

**초기 접근**:  
단순한 contrastive pretraining 으로 core encoder 학습, intermediate layer 에서 표현 품질 분석

**발전된 구조**:  
언어/공간 정렬 전략 도입, frozen feature 정렬, SAM 기반 공간 정렬 등 alignment tuning 기술 다변화

**최근 경향**:  
단일 objective 에서 multi-task 성능 최적화, intermediate layer 활용 전략, teacher-student distillation 로 성능 강화

### 6. 종합 정리

"Visual Embedding" 분야의 방법론은 **단일 목적 함수 기반의 contrastive pretraining**과**alignment tuning** 전략으로 구분될 수 있다. 복잡한 multi-objective pretraining 으로 각 작업을 위한 목적 함수를 조합하는 방식 대신, 단순한 contrastive learning 으로 다양한 다운스트림 작업에 강건한 표현을 학습하는 접근법이 중심을 이룬다.

단일 contrastive pretraining 으로 학습된 시각 모델의 중간층 (intermediate layer) 에서 다양한 작업에 유용한 표현을 추출하고, language/spatial alignment 로 각 작업에 맞게 정렬하는 범용 vision encoder 접근법이 주요 흐름이다. ViT 기반 encoder 에서 특정 층을 각 작업용으로 할당하여 언어·공간 작업을 최적화하고, Progressive resolution, teacher-student distillation 등 학습 전략을 통해 성능을 강화한다.

이러한 방법론적 지지는 vision foundation model 의 효율적인 사전학습과 downstream 활용을 위한 이론적, 실용적 기초를 제공하며, 복잡한 multi-objective 사전학습 없이도 다양한 downstream 작업에서 범용 vision 표현의 일반성을 입증하는 연구 흐름을 보여준다.

제공된 문서에 따르면 첫 번째 논문인 "Perception Encoder: The best visual embeddings are not at the output of the network (2025)"에 대한 정보만 있습니다. 전체 실험 결과를 비교·정렬하기 위해 추가 논문 자료를 필요합니다.

추가로 어떤 논문들의 실험 결과 문서가 있는지 알려주시면, 통합 분석을 진행하겠습니다.
