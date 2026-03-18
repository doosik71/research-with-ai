# A Comprehensive Survey of Foundation Models in Medicine

## 1. Paper Overview

이 논문은 의료 분야에서의 **Foundation Models(FMs)** 를 포괄적으로 정리한 survey paper다. 저자들은 기존 의료 AI 설문 논문들이 대개 특정 modality나 특정 계열 모델에만 집중해 왔고, 특히 **LLM, vision, graph, biology, omics, audio, protein** 등을 아우르는 통합 taxonomy가 부족하다고 지적한다. 이에 따라 본 논문은 의료 영역에서의 FMs를 역사적 맥락, 핵심 학습 전략, 대표 모델군, 의료 특화 모델, 실제 응용 분야, 기회 요인, 한계와 미래 방향까지 한 흐름으로 정리하는 것을 목표로 한다.  

연구 문제는 “의료 분야에서 foundation model을 어떻게 이해하고 분류할 것인가”에 가깝다. 단일 모델을 제안하는 논문이 아니라, 의료 AI 전반에서 FMs가 어떤 방식으로 발전해 왔고, 어떤 데이터·학습 방식·응용 시나리오와 연결되는지 정리한다. 특히 일반-purpose FMs와 의료 특화 FMs를 구분하고, 임상 텍스트, 의료 영상, graph learning, biology/omics 등 이질적인 영역을 한 프레임에서 묶어 설명하려는 점이 중요하다.

이 논문이 중요한 이유는 의료 분야에서 FMs를 논할 때 단순히 “의료 LLM”만 보는 것이 충분하지 않기 때문이다. 실제 의료 AI는 임상 기록, 영상, 분자생물학 데이터, 지식 그래프, 멀티모달 데이터가 섞여 있으며, 각 영역에서 foundation model의 역할이 다르다. 본 논문은 바로 이 복합성을 드러내며, 의료용 FMs의 잠재력과 위험을 함께 다룬다.

## 2. Core Idea

이 survey의 핵심 아이디어는 다음과 같다.

**의료 분야의 foundation model은 단일한 모델 계열이 아니라, 다양한 modality와 다양한 학습 패러다임 위에 형성된 광범위한 생태계이며, 이를 통합 taxonomy로 이해해야 한다.**

논문은 foundation model을 대규모 비지도/자기지도 학습으로 사전학습된 뒤 다양한 downstream task에 적응되는 모델로 보고, 의료 분야에서도 이 개념이 NLP, medical computer vision, graph learning, biology, omics로 확장되었다고 본다. 특히 BERT/GPT 계열 같은 language model뿐 아니라, CLIP 계열, diffusion model, protein foundation model, graph FM까지 포함해 설명한다는 점이 이 survey의 중심적 기획이다.

이 논문의 차별점은 “의료 분야의 foundation model survey”라는 말에 걸맞게 범위를 넓게 잡았다는 데 있다. 논문 초반의 비교표에서는 기존 survey들이 의료 분야에 집중하지 않거나, LLM만 다루거나, vision이나 protein, graph를 빠뜨리는 경우가 많다고 설명한다. 반면 이 논문은 **Healthcare, LLMs, Vision FMs, Protein, Audio, Graph FMs** 를 모두 포괄하는 survey라는 점을 스스로의 기여로 제시한다.

## 3. Detailed Method Explanation

이 논문은 새로운 알고리즘을 제안하는 논문이 아니라 survey이므로, 여기서의 “방법”은 **의료용 foundation model을 분류하고 설명하는 분석 프레임워크**를 뜻한다.

### 3.1 전체 구성

논문은 크게 다음 순서로 전개된다.

1. AI와 deep learning, transformer, SSL 등 배경 설명
2. foundation model의 일반적 구조와 학습 메커니즘 정리
3. BERT, GPT, CLIP, diffusion 등 대표 flagship models 설명
4. 의료 특화 foundation models 정리
5. 의료 응용 분야 taxonomy 제시
6. 기회(opportunities), 도전 과제(challenges), 미래 방향 논의

즉, 이 논문은 “기초 개념 → 일반 FM 계열 → 의료 특화 모델 → 의료 응용 → 한계와 전망”으로 이어지는 구조를 취한다. 독자가 FM 자체를 잘 모르는 상태에서도 읽을 수 있도록 background를 충분히 두고 시작한다는 점이 특징이다.

### 3.2 Foundation Model의 정의와 배경

논문은 foundation model을 대규모 데이터로 사전학습된 후 다양한 downstream task에 적응 가능한 대형 모델로 설명한다. 이때 핵심 enabling factor로 **training data, base model, transfer learning, scale** 를 든다. 또 전통적인 지도학습과 달리 많은 FMs가 **self-supervised learning(SSL)** 에 기반하며, 사전학습 과제가 데이터 자체로부터 만들어진다고 설명한다.

배경 설명에서 특히 강조하는 요소는 다음이다.

* **Transformer**: self-attention을 통해 sequence를 병렬 처리하고 장거리 의존성을 잘 포착하는 구조
* **Attention**: query-key-value 기반의 weighted aggregation
* **SSL**: 대량의 unlabeled data를 활용하는 contrastive, generative, adversarial learning
* **Human feedback RL**: 일부 language model 계열에서 alignment와 성능 향상에 쓰이는 접근
* **CLIP**: 자연어 supervision을 사용하는 image-text foundation model

즉, 논문은 의료 FMs를 이해하려면 단지 의료 데이터만 볼 것이 아니라, 그 기반이 되는 transformer/attention/SSL/contrastive multimodal learning을 함께 이해해야 한다는 입장이다.

### 3.3 학습 아키텍처 분류

논문은 FM의 기본 아키텍처를 주로 transformer 계열로 보고, 다음 세 가지를 설명한다.

* **encoder-decoder**
* **encoder-only**
* **decoder-only**

이 구분은 BERT 계열과 GPT 계열, 그리고 다른 생성/이해 모델들을 나누는 기본 틀이다. 예를 들어 BERT는 bidirectional encoding 중심이고, GPT는 autoregressive generation 중심이라는 점이 이후 의료 모델 분류에도 연결된다.

### 3.4 Flagship Foundation Models

논문은 의료 특화 모델을 설명하기 전에, 그 기반이 되는 대표 모델군을 먼저 설명한다. 핵심적으로는 다음 계열이 중심이다.

* **BERT family**
* **GPT models**
* **CLIP**
* **stable diffusion**
* 산업 규모의 대형 사전학습 모델들

여기서 논문의 메시지는 분명하다. 의료용 foundation model은 완전히 독립적으로 등장한 것이 아니라, 일반 도메인에서 성공한 foundation model 패러다임이 의료 데이터와 의료 태스크에 맞게 재구성된 결과라는 것이다.

### 3.5 의료 특화 Foundation Models

논문은 Section 5에서 의료 특화 FM을 따로 정리한다. 앞부분에서 언급되는 예시만 보더라도 다음과 같은 흐름이 보인다.

* **clinical large language models**
  예: GatorTronGPT는 GPT-3 기반으로 200억 파라미터, 2770억 단어로 학습되었고 그중 820억 단어가 200만 명 환자에서 나온 clinical text라고 소개된다. 이 모델은 NER, biomedical QA, relation extraction, NLI, semantic similarity 등에서 강점을 보였다고 설명된다.

* **medical image foundation models**
  예: MedSAM은 약 150만 medical image로 학습된 segmentation 모델로 소개되며, 일반 모델보다 우수한 성능을 보였다고 요약된다.

* **omics / protein / genomics 계열 모델**
  논문은 의료 FMs를 텍스트와 영상에 한정하지 않고 omics 데이터까지 포함해 다룬다.  

### 3.6 CLIP 계열과 의료 멀티모달 확장

논문이 비교적 자세히 설명하는 대표 사례 중 하나가 CLIP 계열이다. CLIP은 image encoder와 text encoder를 따로 두고, paired image-text의 embedding similarity를 높이는 contrastive objective로 학습된다. 일반 도메인에서는 대규모 웹 데이터로 zero-shot transfer가 가능하지만, 의료 분야에서는 image-text 데이터 규모가 작고 false negative 문제가 발생하기 쉽다고 지적한다.

이를 보완하기 위해 논문은 다음과 같은 의료 특화 CLIP 계열 모델들을 소개한다.

* **MedCLIP**: 의료 지식 기반 semantic matching loss를 사용해 false negative를 줄이려는 접근
* **BiomedCLIP**: biomedical domain에서 원본 CLIP보다 나은 성능을 보인 모델
* **PMC-CLIP**: 대규모 biomedical image-text pair를 기반으로 pretraining된 CLIP 기반 모델

이 부분은 의료 FMs의 중요한 일반 원리를 보여준다. 즉, 일반-domain FM을 그대로 가져오는 것만으로는 부족하고, **데이터 구조, 라벨 특성, semantic similarity, 멀티모달 쌍의 희소성** 같은 의료 고유 문제를 반영해야 한다는 것이다.

## 4. Experiments and Findings

이 논문은 survey이므로 보통의 실험 논문처럼 단일 benchmark 결과표를 중심으로 전개되지 않는다. 대신 다양한 모델군과 응용 분야를 비교·정리하고, 이를 통해 어떤 흐름이 나타나는지 보여준다.

### 4.1 논문이 실제로 보여주는 것

논문이 가장 강하게 보여주는 실증적 메시지는 다음과 같다.

첫째, foundation model은 이미 의료 전반에 확산되어 있으며, 적용 영역이 **clinical NLP, medical computer vision, healthcare graph learning, biology and omics** 로 매우 넓다.

둘째, 의료 특화 foundation model은 대체로 일반-purpose FM보다 의료 데이터에서 더 강한 성능을 보이는 방향으로 발전해 왔다. 논문은 clinical NLP, medical image analysis, omics data용 특화 모델들이 일반 모델 대비 우수하다고 요약한다.  

셋째, 응용 관점에서 FMs는 의료 분야에서 다음과 같은 기능을 지원한다고 정리된다.

* clinical decision support
* medical image analysis
* drug discovery
* personalized medicine
* clinical report generation
* medical text information extraction
* translation and communication support
* image generation / classification / segmentation / modality translation

### 4.2 응용 taxonomy

논문이 제시하는 핵심 응용 taxonomy는 다음 다섯 축이다.

* **Clinical NLP**
* **Medical Computer Vision**
* **Graph Learning**
* **Biology and Omics**
* **Other Applications**

이 taxonomy 자체가 이 논문의 중요한 결과물이다. 기존 survey들이 일부 영역만 다뤘다면, 이 논문은 의료용 foundation model의 응용 범위를 다차원적으로 재구성한다.

### 4.3 논문의 종합적 결론

Discussion 부분에서 저자들은 FMs가 healthcare에서 진단 정밀도 향상, personalized treatment, patient outcome 개선에 기여할 잠재력이 크다고 본다. 동시에 clinical report generation, information extraction, language translation, image reconstruction, segmentation, modality synthesis, augmented reality guidance 등으로 활용 범위를 넓게 본다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 **coverage**다. 의료 분야의 foundation model을 LLM에 한정하지 않고, vision, graph, biology, omics, protein, audio 등으로 확장해 설명한다. 논문 초반 비교표가 보여주듯, 저자들은 기존 survey가 놓친 범위를 메우려 했고, 실제로 이 점이 본 논문의 핵심 기여다.

또 다른 강점은 **구조화된 taxonomy**다. 단순히 모델 이름을 나열하는 것이 아니라, background → flagship models → medical-specific FMs → applications → opportunities → challenges → future directions라는 흐름으로 정리해 독자가 큰 그림을 이해하기 쉽게 만든다.

셋째, 논문은 낙관론만 제시하지 않는다. 의료 분야에서 FM의 잠재력을 크게 보면서도, 비용, 해석가능성, 검증, privacy/security, social bias, scalability, legal/ethical issue를 함께 논의한다는 점에서 균형감이 있다.

### Limitations

한계도 분명하다.

첫째, 이 논문은 survey이기 때문에 **통합 비교 기준이 완전히 엄밀하지는 않다**. 서로 다른 modality, 서로 다른 benchmark, 서로 다른 데이터셋에서 나온 결과를 넓게 정리하다 보니, “어떤 모델이 절대적으로 가장 좋다”는 식의 직접 비교는 어렵다.

둘째, coverage가 넓은 대신 각 세부 분야에 대한 **깊이는 불균등**할 가능성이 있다. 예를 들어 clinical NLP, imaging, protein FM은 각각 독립된 survey가 가능할 정도로 방대한데, 하나의 논문 안에서 모두 다루면 개별 subfield의 최신 쟁점을 충분히 파고들기 어렵다.

셋째, foundation model의 발전 속도가 매우 빠르기 때문에, survey의 시점 한계가 있다. 저자들도 자신들의 기여를 “early 2024까지의 최근 연구를 포함”한 것으로 설명한다. 따라서 이 논문은 2024년 상반기 기준 지형도를 잘 보여주지만, 이후의 급격한 발전까지 반영하지는 못한다.

### Brief Critical Interpretation

비판적으로 보면, 이 논문의 진짜 공헌은 새로운 모델이 아니라 **의료 AI에서 foundation model을 바라보는 시야를 넓혔다**는 점이다. 특히 의료 FMs를 “의료 LLM”으로 축소하지 않고, multimodal·multiscale·multidata 관점에서 본다. 이는 이후 의료 AI 연구에서 매우 중요한 관점이다.

동시에 이 논문은 foundation model의 가능성을 비교적 긍정적으로 보지만, 실제 임상 도입을 위해 필요한 검증, 신뢰성, 편향 통제, 법적 책임 문제는 아직 훨씬 더 엄격한 논의가 필요하다는 점도 드러낸다. 따라서 이 논문은 “의료 FM의 성공 선언”이라기보다는, **의료 FM 연구 지형도와 과제 목록을 정리한 기준 문서**로 읽는 것이 적절하다.

## 6. Conclusion

이 논문은 의료 분야에서의 foundation model을 포괄적으로 정리한 survey로서, transformer와 SSL 기반의 일반 FM 발전사에서 출발해, BERT/GPT/CLIP/diffusion 같은 flagship models, 의료 특화 foundation models, 그리고 clinical NLP·medical computer vision·graph learning·biology/omics 등 다양한 응용 분야를 한 체계 안에서 설명한다.

핵심 메시지는 분명하다. 의료 분야의 foundation model은 이미 다양한 영역에서 강력한 가능성을 보였고, 앞으로도 clinical decision support, image analysis, drug discovery, personalized medicine에 큰 영향을 줄 수 있다. 그러나 비용, 해석가능성, 검증, 프라이버시, 편향, 확장성, 법·윤리 문제를 해결하지 않으면 실제 의료 현장으로의 안전한 확산은 어렵다.

따라서 이 논문은 연구자에게는 의료 FMs의 전체 지형을 파악하는 입문서이자 참고 지도이며, 실무자에게는 “무엇이 가능하고 무엇이 아직 위험한가”를 함께 보여주는 survey라고 할 수 있다.
