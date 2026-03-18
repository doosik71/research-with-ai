# Foundation Model for Advancing Healthcare: Challenges, Opportunities and Future Directions

## 1. Paper Overview

이 논문은 healthcare foundation model(HFM)에 관한 **종합 survey**다. 즉, 새로운 단일 모델이나 실험 프레임워크를 제안하는 연구라기보다, 의료 분야에서 foundation model이 어떤 방식으로 발전해 왔고, 어떤 하위 분야(language, vision, bioinformatics, multimodal)에서 어떤 기술 흐름이 형성되었으며, 실제 적용과 한계가 무엇인지를 정리하는 리뷰 논문이다. 저자들은 의료 AI가 오랫동안 task-specific specialist model 중심으로 발전해 왔지만, 실제 의료 현장은 매우 다양한 데이터 유형과 과업으로 구성되어 있어 범용적 일반화 능력을 갖춘 foundation model이 필요하다고 본다. 이 논문은 바로 그 문제의식에서 출발해, HFM의 현재 진전, 핵심 도전 과제, 그리고 미래 방향을 체계적으로 정리한다.  

논문이 중요하게 다루는 연구 문제는 세 가지다. 첫째, 의료 분야에서 foundation model은 현재 어디까지 와 있는가. 둘째, 실제 의료 환경에 폭넓게 적용되기 위해 어떤 병목이 존재하는가. 셋째, 앞으로 어떤 연구 방향이 유망한가. 저자들은 이를 위해 2018년부터 2024년 초까지의 HFM 관련 연구를 검토하며, 방법론, 데이터셋, 응용, 도전 과제, 미래 방향까지 하나의 큰 지도처럼 정리한다. 특히 이 논문은 HFM을 language, vision, bioinformatics, multimodal의 네 축으로 나눠 보는 것이 핵심적이다.

## 2. Core Idea

이 논문의 중심 아이디어는 “의료용 foundation model”을 특정 모달리티나 특정 모델 계열로 좁게 보지 않고, **의료 데이터 전반을 대상으로 pre-training과 adaptation을 통해 다양한 다운스트림 task에 전이 가능한 일반 기반 모델**로 파악한다는 점이다. 저자들은 의료 분야의 foundation model을 단순히 “큰 모델”로 정의하지 않고, **넓은 데이터로 pre-train되고 다양한 task에 adapt 가능한 모델**이라는 foundation model의 본래 정의에 충실하게 다룬다.

논문의 핵심 기여는 단순한 literature listing이 아니다. 저자들은 다음 다섯 가지 기여를 명확히 내세운다.
첫째, 2018–2024년의 관련 기술 논문 200편을 바탕으로 HFM 방법론을 체계적으로 taxonomy화했다.
둘째, HFM 학습에 잠재적으로 활용 가능한 대규모 데이터셋/데이터베이스 114개를 정리했다.
셋째, 16개의 잠재 의료 응용 영역을 개관했다.
넷째, 데이터·알고리즘·컴퓨팅 인프라 차원의 핵심 도전 과제를 논의했다.
다섯째, 미래의 역할, 구현, 응용, 강조점 측면에서 future directions를 제시했다.

상대적으로 새로운 점은, 기존 survey들이 의료 LLM, medical imaging FM, multimodal FM 등 일부 하위 분야만 다루는 경향이 있는 반면, 이 논문은 **language, vision, bioinformatics, multimodal을 하나의 HFM 프레임으로 통합**해 서술한다는 데 있다. 즉 “의료 foundation model”을 네 갈래로 쪼개 설명하되, 궁극적으로는 일반 의료 AI로 수렴하는 흐름으로 해석한다.

## 3. Detailed Method Explanation

이 논문은 survey이므로, 여기서의 “method”는 하나의 새 모델 구조가 아니라 **의료 foundation model을 분류하고 이해하는 분석 프레임워크**를 뜻한다.

### 3.1 전체 프레임: Pre-training + Adaptation

논문은 HFM을 “대규모·다양한 의료 데이터에서 representation을 학습하고, 이후 여러 의료 application으로 적응하는 모델”로 정의한다. 이를 위해 방법론을 두 단계로 나눈다.

* **Pre-training**: 대규모 데이터에서 범용 표현 학습
* **Adaptation**: 특정 의료 과업이나 도메인에 맞게 미세 조정 또는 프롬프트 기반 적응

또한 pre-training 방법을 네 범주로 분류한다.

* **Generative Learning (GL)**: 데이터를 생성적으로 모델링하는 방식
* **Contrastive Learning (CL)**: 유사/비유사 샘플 간 표현 거리를 학습
* **Hybrid Learning (HL)**: 여러 학습 패러다임의 결합
* **Supervised Learning (SL)**: 라벨 기반 지도학습

그리고 adaptation 방법은 세 범주로 나눈다.

* **Fine-tuning (FT)**: 기존 파라미터 조정
* **Adapter Tuning (AT)**: 추가 adapter 파라미터만 학습
* **Prompt Engineering (PE)**: 설계된 prompt 또는 instruction으로 task 수행 유도

이 분류 체계는 논문 전체를 관통하는 분석 축이며, 각 하위 분야별 모델들을 이 축 위에 배치하는 식으로 survey가 전개된다.

### 3.2 네 가지 하위 분야 분류

논문은 HFM을 네 하위 분야로 나눈다.

#### 3.2.1 Language Foundation Models (LFM)

LFM은 의료 텍스트, 임상 기록, 질의응답, 대화, 요약, 정보검색 등에 쓰이는 foundation model이다. 논문에 따르면 의료 LFM에서는 GL 기반 pre-training이 가장 흔하며, adaptation은 주로 FT와 PE가 활용된다. 예시로 GatorTronGPT는 medical + general text를 활용한 next-token prediction 기반 GPT 계열 모델이고, PMC-LLaMA는 LLaMA 기반 의학 지식 주입형 모델로 소개된다. 반면 BioBERT, PubMedBERT, ClinicalBERT는 MLM 및 NSP를 활용하는 BERT 계열 HL 사례로 정리된다. 또 MedCPT는 biomedical retrieval에 특화된 contrastive learning 사례다.

이 분류는 의료 LLM을 단순 챗봇으로 보는 것이 아니라, pre-training objective와 adaptation 방식의 관점에서 이해하게 만든다. 논문은 특히 recent medical LLM 계열이 instruction tuning, LoRA, prompt engineering을 통해 빠르게 확장되고 있음을 보여준다.

#### 3.2.2 Vision Foundation Models (VFM)

서론에서 저자들은 VFM이 modality-specific, organ-specific, task-specific 방향에서 의료 영상으로 확장되어 왔다고 정리한다. 이는 범용 vision backbone이 의료영상에 적응하는 흐름과, SAM 같은 universal vision model의 영향을 의료영상 segmentation과 general medical perception으로 끌어오는 흐름을 뜻한다. 논문은 이러한 흐름을 바탕으로 의료 VFM이 검사 modality, 장기, task를 가로지르는 일반화 능력을 확보하려는 움직임이라고 본다.

#### 3.2.3 Bioinformatics Foundation Models (BFM)

BFM은 단백질, DNA, RNA 등 생명정보학 데이터에서 foundation model을 구축하는 흐름이다. 서론에서는 AlphaFold2의 영향이 단백질 구조 예측을 넘어 RNA, DNA, protein 전반의 foundation model 연구를 촉진했다고 정리한다. 이 영역은 언어/영상보다 데이터 표현이 더 생물학적이고 구조적이기 때문에, sequence와 structure를 모두 고려하는 표현 학습이 중요하다는 함의를 가진다.

#### 3.2.4 Multimodal Foundation Models (MFM)

MFM은 의료의 본질적 멀티모달성—이미지, 텍스트, omics, 임상 기록, 바이오시그널—을 통합하는 방향이다. 논문은 CLIP류의 vision-language pretraining이 의료로 빠르게 전이되었고, 이를 통해 generalist HFM의 유력한 구현 경로가 열렸다고 본다. 최근에는 강한 vision encoder와 open-source LLM을 결합한 visual condition language model 혹은 vision-language assistant 흐름도 소개한다. 예시로 PathAsst, PathChat, Qilin-Med-VL, XrayGPT 등이 언급된다.

### 3.3 역사적 전개에 대한 해석

논문은 2018년 BERT를 foundation model 시대의 분기점으로 보고, 의료 분야에서는 2019년 BioBERT, 2020년 AlphaFold2, 2021년 CLIP, 2022년 ChatGPT가 중요한 가속 이벤트였다고 설명한다. 이는 의료 HFM의 진화가 한 번에 이뤄진 것이 아니라, NLP, vision, structural biology, multimodal AI의 일반 foundation model 혁신이 의료로 흘러들어오며 형성되었다는 해석이다. 즉 의료 HFM은 독립된 기술 섬이 아니라, broader AI ecosystem의 의료화된 형태다.

## 4. Experiments and Findings

이 논문은 원저자들이 하나의 모델을 학습해 특정 benchmark에서 실험한 논문이 아니다. 따라서 “실험 결과”도 일반적인 machine learning paper처럼 단일 method의 성능표가 아니라, **survey가 수집·요약한 연구 동향과 정량적 집계** 형태로 나타난다. 이 점을 구분해 읽는 것이 중요하다.

### 4.1 Survey 규모와 범위

논문은 방법론 측면에서 **200편의 기술 논문**, 데이터 측면에서 **114개의 대규모 dataset/database**, 응용 측면에서 **16개 의료 application**을 포괄했다고 명시한다. 이 숫자는 단순 부록 나열이 아니라, 저자들이 survey의 폭과 체계성을 강조하기 위해 제시한 핵심 수치다.

### 4.2 의료 HFM의 기술적 흐름

논문의 주요 발견은 의료 HFM이 네 하위 분야 모두에서 빠르게 성장하고 있다는 것이다. LFM은 clinical NLP와 dialogue에서, VFM은 다양한 이미지 modality와 segmentation/classification/report generation에서, BFM은 생체 분자 수준의 구조 및 sequence 문제에서, MFM은 의료 데이터 통합 해석에서 중요한 진전을 보였다고 정리한다.

특히 의료 LFM에서는 GL 기반 pre-training과 FT/PE 기반 adaptation이 주류이며, PMC-LLaMA, GatorTronGPT, MedPaLM 계열 같은 모델들이 이 흐름을 대표한다. 이는 의료 LFM이 general-purpose LLM을 의학 도메인 데이터로 further pretrain하거나 instruction tune하는 방향으로 발전하고 있음을 보여준다.

### 4.3 핵심 병목

논문이 반복해서 강조하는 병목은 세 가지다.

첫째, **데이터 문제**다. 의료 데이터는 윤리, 프라이버시, 다양성, 이질성, 수집 비용 때문에 충분히 크고 일반화 가능한 학습 데이터를 만들기 어렵다.
둘째, **알고리즘 문제**다. 실제 의료 적용에는 단순 정확도뿐 아니라 adaptability, capacity, reliability, responsibility가 필요하다.
셋째, **컴퓨팅 인프라 문제**다. 3D CT, whole slide image 같은 고차원 대용량 의료 데이터는 다른 분야보다 훨씬 큰 계산 비용과 환경 비용을 요구한다.

이 세 병목은 단순 기술적 디테일이 아니라, HFM이 의료 분야에서 “바로 실전 배치되기 어려운 이유”를 설명하는 구조적 요인이다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **통합성**이다. 기존 survey가 의료 LLM, 의료 영상 FM, multimodal 모델 등 특정 축에 치우치는 경우가 많은 반면, 이 논문은 language, vision, bioinformatics, multimodal을 하나의 공통 프레임으로 묶는다. 덕분에 독자는 의료 foundation model 전체 지형을 한 번에 볼 수 있다.

둘째, **구조화된 taxonomy**가 유용하다. pre-training을 GL/CL/HL/SL로, adaptation을 FT/AT/PE로 나눈 틀은 이질적인 모델들을 비교 가능하게 만든다. 실제 의료 FM 연구가 너무 빠르게 확장되면서 용어와 범주가 혼재되는 문제를 생각하면, 이 taxonomy는 상당히 실용적이다.

셋째, **데이터와 응용까지 포함한 survey**라는 점도 강점이다. 많은 리뷰가 모델만 나열하고 끝나는 반면, 이 논문은 114개 데이터 자원과 16개 응용 시나리오를 함께 정리하여 “모델이 어디서 왔고 어디로 가는가”를 같이 보여준다.

### 한계

첫 번째 한계는 survey 논문 특유의 문제로, **깊이와 폭의 trade-off**다. 네 하위 분야를 모두 커버하는 대신, 각 분야의 개별 모델에 대한 비판적 비교나 benchmark-level 세밀 분석은 상대적으로 얕을 수밖에 없다. 예를 들어 각 모델의 평가 프로토콜 차이, 데이터 누수 위험, 임상적 유효성 수준 같은 문제는 개별 분야 survey만큼 깊게 들어가기 어렵다. 이 점은 논문의 결함이라기보다 범위 설정의 결과다.

둘째, 저자들은 HFM을 상당히 넓게 정의하기 때문에, **foundation model의 경계가 다소 느슨해질 수 있다**. 실제로 일부 사례는 엄밀한 의미의 large-scale foundation model이라기보다는 domain-adapted pretrained model 혹은 prompt-based system에 가깝다. 따라서 독자는 “foundation model”이라는 용어가 이 논문에서 비교적 포괄적으로 사용된다는 점을 감안해야 한다.

셋째, 이 논문은 미래 방향을 풍부하게 제시하지만, 그것이 대부분 **전망(prospect)** 의 형태라는 점도 기억해야 한다. 즉, 어떤 미래 방향이 실제로 가장 실현 가능하고 임상적으로 효과적인지는 후속 실증 연구가 더 필요하다.

### 비판적 해석

이 논문을 가장 잘 읽는 방법은 “최고 성능의 의료 foundation model을 고르는 논문”으로 보는 것이 아니라, **의료 AI가 specialist model 시대에서 foundation model 시대로 어떻게 전환되고 있는지 설명하는 지도**로 보는 것이다. 그런 의미에서 이 논문의 가치가 크다. 특히 EHR/clinical text, medical image, omics, multimodal clinical reasoning이 결국 하나의 범용 의료 AI 플랫폼으로 수렴할 수 있다는 장기 비전을 잘 드러낸다.

반면 실무적 관점에서는 이 논문이 제시하는 장밋빛 전망을 그대로 받아들이기보다는, 데이터 거버넌스, 신뢰성, 책임성, 계산 비용이라는 병목을 훨씬 더 심각하게 봐야 한다. 저자들도 그 점을 명시하고 있다. 특히 의료는 일반 소비자 AI보다 실패 비용이 훨씬 크기 때문에, foundation model의 “범용성”만으로는 충분하지 않다. 이 모델이 언제, 어떤 조건에서, 누구에게, 어느 정도 신뢰도로 쓰일 수 있는가가 더 중요하다.

## 6. Conclusion

이 논문은 의료용 foundation model 분야를 종합적으로 정리한 대규모 survey로서, HFM을 language, vision, bioinformatics, multimodal의 네 축으로 나누고, 각 영역에서의 pre-training과 adaptation 전략, 데이터 자원, 응용 분야, 그리고 구조적 병목을 정리한다. 특히 200편의 기술 논문, 114개의 데이터 자원, 16개의 응용 시나리오를 아우르며, 의료 AI가 specialist model 중심 패러다임에서 generalist foundation model 패러다임으로 이동하고 있음을 보여준다.

실무적으로 이 논문은 의료 foundation model을 처음 전체적으로 파악하려는 연구자에게 좋은 출발점이다. 연구적으로는 “의료 HFM이 어떤 하위 분야를 통해 형성되고, 무엇이 아직 해결되지 않았는가”를 구조적으로 이해하게 해 준다. 다만 survey인 만큼, 특정 모델의 우월성이나 임상적 실전성에 대한 직접 증거를 제공하는 논문은 아니다. 따라서 이 논문은 **분야 지도를 제공하는 문헌**으로 읽는 것이 가장 적절하다.
