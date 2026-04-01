# Surgical-VQLA: Transformer with Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery

Long Bai, Mobarakol Islam, Lalithkumar Seenivasan and Hongliang Ren

## 🧩 Problem to Solve

로봇 수술 영상에서 질문에 답하고 관련 영역을 정확히 찾아내는 기존 시각 질의 응답(VQA) 시스템의 한계를 해결하는 것이 이 연구의 주요 목표입니다. 구체적인 문제점은 다음과 같습니다:

* **전문가의 부담:** 수련의들은 수술 절차에 대한 질문에 대해 전문가에게 크게 의존하지만, 전문가들은 임상 및 학업 업무로 과부하 상태입니다.
* **기존 VQA 모델의 한계:**
  * 객체 감지기와 영역 기반 특징 추출기가 필요하지만, 수술 데이터셋은 작고 바운딩 박스 주석이 부족하여 수술 객체 감지 모델을 구축하기 어렵습니다.
  * 텍스트와 이미지 같은 이종 모달리티 간의 특징 융합 전략이 비효율적이고 단순합니다.
  * 복잡한 수술 시나리오에서 매우 중요한 **위치 기반 답변(localized answering)** 기능이 부족합니다. 이는 "무엇(what)?"과 "어디(where)?"에 답하여 "왜(why)?"를 추론하는 데 필수적입니다.
  * 대규모 주석 데이터셋의 필요성, 모델의 엔드-투-엔드 학습의 어려움, 높은 연산 비용으로 실시간 적용에 제약이 있습니다.

## ✨ Key Contributions

* 주어진 질문과 수술 장면을 기반으로 위치 기반 답변을 예측할 수 있는 Surgical Visual Question Localized-Answering (Surgical-VQLA) 모델을 설계하고 제안했습니다.
* 새로운 GVLE(Gated Vision-Language Embedding) 기법을 사용하여 이종 특징(시각 및 텍스트)을 효과적으로 융합하는 detection-free GVLE-LViT 모델을 VQLA 작업에 제안했습니다.
* 질의 응답 모델의 예측 및 위치 파악 성능을 모두 향상시키기 위해 크로스-엔트로피(CE) 손실 및 $L_1$ 손실과 함께 GIoU(Generalized Intersection over Union) 손실을 통합했습니다.
* 광범위한 검증을 통해 다음과 같은 사실을 발견했습니다:
  * Surgical-VQLA는 답변이 수술 상호작용과 관련될 때에도 문맥을 정확히 파악할 수 있습니다.
  * 제안된 detection-free VQLA는 연산 비용이 비싸고 오류 발생 가능성이 있는 감지 모듈을 피함으로써 더 나은 특징 학습을 보여주며, 수술 질의 위치 기반 답변 시스템의 엔드-투-엔드 실시간 적용을 용이하게 합니다.
  * 제안된 GVLE는 시각 및 단어 임베딩의 이종 모달리티를 효과적으로 융합하며 기존 접근 방식보다 우수한 성능을 보입니다.

## 📎 Related Works

* **의료 VQA (Medical VQA):** MedFuseNet [1]과 Surgical-VQA [2]는 의료 VQA의 가능성을 열었지만, 주로 "무엇(what)?"에 초점을 맞추고 "왜(why)?"에 대한 답변은 부족했습니다. 특히 Surgical-VQA [2]는 문장 기반 개방형 VQA를 제안했지만, 의료 분야의 주석 데이터셋 부족으로 견고한 모델 개발이 어려웠습니다.
* **Transformer 기반 VQA:** VisualBERT [9]와 VisualBERT ResMLP [2]는 트랜스포머 인코더 모델을 사용하여 비전-언어 작업 성능을 향상시켰습니다. 그러나 이들은 객체 감지 모델에 의존하며, 시각 특징 추출과 텍스트 임베딩의 융합 방식이 단순했습니다.
* **Vision Transformer (ViT):** ViT [21]는 이미지를 패치로 분할하여 언어 작업의 트랜스포머 성능을 시각 작업으로 확장했습니다.
* **손실 함수:** GIoU [17]는 바운딩 박스 회귀를 위한 손실 함수로, 겹치는 영역뿐만 아니라 겹치지 않는 영역까지 고려하여 성능을 향상시키는 데 기여했습니다.
* **특징 융합(Feature Fusion):** 기존 모델들은 단순한 연결(concatenation), 덧셈, 평균 등을 사용했으며, Attentional Feature Fusion (AFF) 및 iterative AFF (iAFF) [16]와 같은 방법이 제안되었습니다.

## 🛠️ Methodology

본 논문은 효율적인 임베딩을 통해 Surgical-VQLA를 수행하기 위한 GVLE(Gated Vision-Language Embedding) 시스템을 제안하는 Language-Vision Transformer (GVLE-LViT)를 개발했습니다.

* **GVLE-LViT 아키텍처:**
  * **Visual Feature Extractor:** ImageNet [24]에서 사전 훈련된 ResNet18 [23]을 사용하여 시각적 특징을 추출합니다. 기존 VQA와 달리 객체 제안(object proposals)에서 특징을 추출하지 않고 전체 이미지에서 특징을 추출하여 전역 장면 이해를 가능하게 합니다.
  * **Tokenizer:** 수술 관련 데이터셋으로 학습된 맞춤형 토크나이저를 사용하여 단어 임베딩을 생성합니다.
  * **Gated Vision-Language Embedding (GVLE):**
    * 기존의 단순한 시각 및 단어 임베딩 연결 방식을 대체합니다.
    * 순환 신경망의 흐름 제어 개념에서 영감을 받아 시각 및 단어 임베딩 간의 최적의 중간 상태를 찾는 데 중점을 둡니다.
    * 각 모달리티의 특징 임베딩은 $\tanh$ 활성화 함수를 통과하여 내부 표현을 인코딩합니다.
    * 게이트 노드 $\alpha$는 $\tanh$ 활성화 함수에서 전달된 정보를 받아 해당 임베딩 정보의 유용성을 결정합니다.
    * 시각 특징 $f$와 단어 임베딩 $e$를 융합하는 수식은 다음과 같습니다:
            $$ \omega = \alpha(\theta_{\omega} \cdot [f \parallel e]) $$
            $$ \Upsilon = \omega \ast \tanh (\theta_{f} \cdot f) + (1-\omega) \ast \tanh (\theta_{e} \cdot e) $$
            여기서 $\theta_{\omega}, \theta_{f}, \theta_{e}$는 학습 가능한 매개변수이며, $[ \cdot \parallel \cdot ]$는 연결 연산, $\Upsilon$는 GVLE 모듈의 최종 출력입니다.
  * **Vision Transformer (ViT):** GVLE의 출력은 Layer-Normalization과 함께 사전 훈련된 ViT [21] Transformer 인코더를 통과합니다.
  * **Prediction Head:**
    * **Classification Head:** ViT 블록의 출력을 Softmax와 함께 선형 예측 레이어로 전파하여 분류 예측을 수행합니다.
    * **Localization Head:** FFN(Feed-Forward Network)으로 구성되며, ReLU 활성화 함수가 있는 3계층 퍼셉트론과 선형 투영 레이어를 포함합니다. 바운딩 박스의 정규화된 좌표(높이, 너비, 중심 좌표)를 출력합니다.
* **Loss Function:** 엔드-투-엔드 공동 학습을 위해 분류 손실과 감지 손실을 결합합니다:
    $$ L = L_{CE} + (L_{GIoU} + L_1) $$
  * $L_{CE}$: 분류를 위한 크로스-엔트로피 손실.
  * $L_1$: 감지를 위한 $L_1$ 손실.
  * $L_{GIoU}$: GIoU(Generalized Intersection over Union) 손실 [17]은 바운딩 박스 회귀를 위해 겹치는 영역과 겹치지 않는 영역 모두에 초점을 맞춥니다.
        $$ L_{GIoU} = 1 - \left( \frac{|b_g \cap b_p|}{|b_g \cup b_p|} - \frac{|B(b_g, b_p) \setminus b_g \cup b_p|}{|B(b_g, b_p)|} \right) $$
        여기서 $b_g$는 실제 바운딩 박스, $b_p$는 예측 바운딩 박스, $|\cdot|$는 면적을 나타내고, $B(\cdot, \cdot)$는 $b_g$와 $b_p$를 모두 포함하는 가장 큰 박스를 의미합니다.

## 📊 Results

* **정량적 성능:**
  * 제안된 GVLE-LViT 모델은 EndoVis-18-VQLA 및 EndoVis-17-VQLA 데이터셋 모두에서 VisualBERT [9] 및 VisualBERT ResMLP [2]와 같은 기존 SOTA 모델보다 우수한 성능을 보였습니다 (Table I).
  * 전체 이미지에서 특징을 추출하는 detection-free 접근 방식은 객체 감지 모델의 출력에서 특징을 사용하는 방식보다 일관되게 우수했습니다. 이는 전역 장면 이해 능력과 엔드-투-엔드 모델 학습의 최적 수렴 덕분입니다.
  * 객체 감지 네트워크의 필요성을 제거함으로써 모델 처리 속도가 8배 이상 향상되어 150.6 FPS를 달성했으며, 실시간 애플리케이션에 적합함을 입증했습니다.
* **정성적 성능:**
  * 그림 3에 제시된 바와 같이, GVLE-LViT는 답변 예측과 위치 파악(Ground-truth 바운딩 박스에 근접) 모두에서 기준 모델보다 뛰어난 성능을 보였습니다.
* **K-fold 교차 검증:**
  * 3가지 다른 K-fold 분할 방식에 대한 연구에서도 GVLE-LViT 모델이 모든 폴드와 데이터셋에서 기준 트랜스포머 기반 모델보다 일반적으로 우수함을 입증했습니다 (Table II).
* **Ablation Studies:**
  * **손실 함수 조합:** 크로스-엔트로피(CE) 손실과 $L_1$ 손실 외에 GIoU [17] 손실을 통합하는 것이 답변 예측 및 위치 파악 성능을 모두 크게 향상시켰습니다 (Table III).
  * **이종 특징 융합 기법:** 제안된 GVLE 특징 융합 기법은 ConCAT [9], AFF [16], iAFF [16]와 같은 다른 특징 융합 기법들보다 우수한 성능을 보였습니다 (Table IV).

## 🧠 Insights & Discussion

* **"Why?" 추론의 용이성:** Surgical-VQLA는 "무엇?"과 "어디?"에 답함으로써 학생들이 "왜?"를 추론하기 쉽게 하여 수술 훈련에서 중요한 보조 도구가 될 수 있습니다. 이는 복잡한 의료 진단 및 수술 장면을 더 잘 이해하는 데 기여합니다.
* **효율적인 특징 융합:** GVLE(Gated Vision-Language Embedding) 기법은 시각 및 텍스트와 같은 이종 모달리티 특징을 효과적으로 융합하여 기존의 단순한 방법론보다 우수한 성능을 달성했습니다.
* **엔드-투-엔드 학습의 장점:** 객체 감지 모듈 없이 전체 이미지에서 특징을 추출하고 엔드-투-엔드로 학습함으로써, 연산 비용을 줄이고 모델의 전역 장면 이해 능력을 향상시키며 실시간 적용 가능성을 높였습니다.
* **GIoU 손실의 중요성:** GIoU 손실의 통합은 답변 예측과 위치 파악 성능을 동시에 향상시켜 모델의 견고성을 강화하는 데 핵심적인 역할을 했습니다.
* **제한 및 향후 연구:**
  * 현재 모델은 답변 위치 파악을 통해 예측의 신뢰도를 부분적으로 정량화할 수 있지만, 위치 정보를 사용하여 예측 신뢰도를 직접 예측하는 것은 향후 연구 방향이 될 수 있습니다.
  * 더 복잡한 데이터셋과 도전적인 질의응답 쌍을 활용하여 Surgical-VQLA 시스템의 잠재력을 더욱 확장할 수 있습니다.
  * 의료 진단과 같은 새로운 응용 분야에 대한 가능성을 열었습니다.

## 📌 TL;DR

본 논문은 로봇 수술 영상에서 질문에 대한 위치 기반 답변을 제공하는 **Surgical-VQLA** 모델을 제안합니다. 기존 VQA 모델이 가진 수술 데이터셋의 특징 주석 부족, 이종 모달리티 융합의 비효율성, 위치 기반 답변 부재 등의 문제를 해결하고자 합니다. 이를 위해, 객체 감지 모듈 없이 엔드-투-엔드 학습이 가능하고, 시각 및 텍스트 특징을 효율적으로 융합하는 **GVLE (Gated Vision-Language Embedding)** 기법을 포함한 **GVLE-LViT** 트랜스포머 모델을 개발했습니다. 또한, 답변 예측 및 위치 파악 성능 향상을 위해 크로스-엔트로피, $L_1$, **GIoU 손실**을 결합하여 사용했습니다. 실험 결과, Surgical-VQLA는 기존 SOTA 모델들 대비 뛰어난 성능을 보였으며, 실시간 적용 가능한 빠른 처리 속도를 달성했습니다. 이 시스템은 수술 훈련에서 "무엇?", "어디?" 질문에 답하여 "왜?"를 추론하는 데 도움을 주는 중요한 보조 도구가 될 수 있습니다.
