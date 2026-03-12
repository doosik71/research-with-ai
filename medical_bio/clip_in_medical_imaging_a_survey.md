# CLIP in Medical Imaging: A Survey

## 논문 메타데이터

- 제목: CLIP in Medical Imaging: A Survey
- 저자: Zihao Zhao, Yuxiao Liu, Han Wu, Mei Wang, Yonghao Li, Sheng Wang, Lin Teng, Disheng Liu, Zhiming Cui, Qian Wang, Dinggang Shen
- 발표 형태: arXiv preprint
- arXiv: [2312.07353](https://arxiv.org/abs/2312.07353)
- 버전 기준: v6
- 날짜: 2025년 3월 26일
- 키워드: CLIP, medical image analysis, image-text alignment, vision-language model

## 연구 배경 및 문제 정의

의료 영상 AI는 분류, 탐지, 분할 등에서 빠르게 발전했지만, 기존의 순수 비전 중심 학습은 해석 가능성, 분포 이동 대응력, 사람의 진단 논리와의 정렬 측면에서 한계를 보여 왔다. 반면 의료 현장에는 영상과 함께 방사선 판독문, 임상 기술문, 병리 설명 등 전문가가 작성한 텍스트가 이미 대규모로 축적되어 있으며, 이는 시각 정보만으로는 포착하기 어려운 의미 정보를 제공한다.

이 논문은 이러한 맥락에서 CLIP(Contrastive Language-Image Pre-training)이 의료 영상 분야에서 어떤 방식으로 확장되고 활용되어 왔는지를 체계적으로 정리한다. 저자들은 단순히 의료용 CLIP 사전학습 모델을 나열하는 수준을 넘어서, 의료 영상이라는 도메인이 일반 이미지와 다른 지점이 무엇인지, 그 차이가 CLIP 설계에 어떤 수정 요구를 만드는지, 그리고 사전학습된 CLIP이 실제 의료 과제에서 어떤 역할을 수행하는지를 함께 분석한다.

## 논문의 핵심 기여

- 의료 영상 분야에서 CLIP을 전면적으로 다룬 초기 대규모 survey로서, 관련 연구 224편을 정리한다.
- 연구 흐름을 `refined CLIP pre-training`과 `CLIP-driven applications`의 두 축으로 구분하는 명확한 taxonomy를 제시한다.
- 의료 영상 CLIP 사전학습의 핵심 난제를 `multi-scale features`, `data scarcity`, `specialized knowledge demands`로 정리한다.
- 분류, dense prediction, cross-modal task까지 포함해 CLIP이 실제 임상 과제에 어떻게 적용되는지 폭넓게 요약한다.
- 공정성, 데이터 편향, 3D 영상, 멀티모달 통합, 평가 체계 등 향후 연구 과제를 구체적으로 제안한다.

## 데이터셋 및 조사 범위

저자들은 Google Scholar, DBLP, arXiv, IEEE Xplore 등을 기반으로 "CLIP", "image-text alignment", "medical imaging" 등의 키워드로 문헌을 수집했다. 단순 baseline으로 CLIP을 사용하는 논문, 의료 영상이 핵심이 아닌 논문, diffusion model 자체를 중심으로 하는 논문은 제외했다. 그 결과 총 224편을 채택했으며, modality 분포는 X-ray가 가장 큰 비중을 차지한다.

논문은 공개 의료 image-text 데이터셋도 함께 정리한다. ROCO, MedICaT, PMC-OA 같은 논문 기반 figure-caption 데이터셋, MIMIC-CXR와 PadChest 같은 임상 보고서 기반 데이터셋, OpenPath와 Quilt-1M 같은 병리 이미지 데이터셋, CT-RATE와 AbdomenAtlas 3.0 같은 CT 데이터셋을 폭넓게 다룬다. 이 정리는 의료용 CLIP 연구가 왜 흉부 X-ray에 편중되었는지, 그리고 최근 CT, 병리, 안저, 초음파로 왜 확장되는지를 설명하는 근거로 기능한다.

## CLIP 기본 원리와 의료 영상에서의 의미

논문은 먼저 일반 CLIP의 구조를 간단히 요약한다. 비전 인코더와 텍스트 인코더를 공동 학습해 대응되는 이미지-텍스트 쌍이 공통 임베딩 공간에서 가깝게 위치하도록 만들고, 이를 통해 zero-shot 분류와 강한 일반화 능력을 확보한다. 의료 영상 관점에서 중요한 점은 텍스트가 단순 caption이 아니라 전문지식이 압축된 판독문 또는 임상 보고서라는 점이다.

저자들이 강조하는 CLIP의 의료적 장점은 세 가지다.

- 텍스트 감독을 통해 모델이 사람의 진단 개념과 더 잘 정렬된다.
- zero-shot 또는 prompt-based 추론을 통해 해석 가능한 의사결정 경로를 만들 수 있다.
- 분류를 넘어서 분할, 탐지, 보고서 생성, VQA, 검색 같은 cross-modal task까지 확장 가능하다.

즉, CLIP은 단순한 표현학습 모델이 아니라 의료 영상에서 시각 정보와 전문가 언어를 연결하는 기반 모델로 해석된다.

## 의료 영상용 CLIP 사전학습의 핵심 난제

### 1. Multi-scale feature 문제

의료 영상에서는 병변이 매우 작은 영역에 존재할 수 있고, 텍스트 역시 여러 문장으로 구성되어 문장별 중요도가 다르다. 일반 CLIP처럼 이미지 전체와 텍스트 전체만 맞추는 global contrast만으로는 작은 병변, 국소 소견, 문장 단위 의미 정렬을 충분히 다루기 어렵다.

논문은 이를 의료 영상과 자연 이미지의 가장 중요한 차이 중 하나로 본다. 예를 들어 흉부 X-ray 판독문은 특정 위치의 결절, 삼출, 무기폐, 장비 삽입 여부 등을 문장별로 나누어 설명하기 때문에, 단어 또는 문장 수준과 영상의 국소 영역 간 정렬이 중요해진다.

### 2. Data scarcity 문제

일반 CLIP은 웹 규모 데이터에 의존하지만, 의료 분야에서는 개인정보, 저작권, 표준화 문제로 인해 대규모 image-text 쌍을 구하기 어렵다. 공개 데이터셋 수와 규모가 제한적이기 때문에, 의료용 CLIP은 더 적은 데이터로도 잘 작동하는 data-efficient 설계가 필요하다.

### 3. Specialized knowledge 문제

의료 텍스트는 일반 언어보다 훨씬 전문적이며, 개념 간 위계와 인과 구조가 복잡하다. 예를 들어 특정 소견은 특정 장기 구조, 질환, 장비, 병태생리와 얽혀 있다. 단순 contrastive objective만으로는 이런 도메인 지식을 충분히 반영하기 어렵기 때문에, 지식 그래프, 엔티티 관계, 프롬프트 기반 지식 주입이 중요해진다.

## Refined CLIP Pre-training 정리

논문은 의료 영상용 CLIP 사전학습 개선 연구를 크게 multi-scale contrast, data-efficient contrast, knowledge enhancement, 기타 확장으로 나눈다.

### 1. Multi-scale contrast

대표 사례는 GLoRIA, LocalMI, LoVT, PRIOR 등이다.

- GLoRIA는 word-level 또는 local-region 수준 정렬을 도입해 global contrast의 한계를 보완한다.
- LoVT는 문장 단위와 영상 지역 간 양방향 정렬을 강화하고, transformer 기반 local interaction을 활용한다.
- PRIOR 등 후속 연구는 conditional reconstruction이나 prototype memory를 추가해 더 세밀한 cross-modal 상호작용을 유도한다.
- CT 같은 volumetric imaging에서는 해부학 구조 단위의 정렬이 중요해지며, fVLM, BrgSA 같은 연구가 anatomy-aware alignment로 확장한다.

이 흐름의 핵심은 "의료 보고서는 문장 구조를 가지며 의료 영상은 해부학적 구조를 가진다"는 사실을 CLIP 학습 목표에 반영하는 것이다.

### 2. Data-efficient contrast

논문은 데이터 효율화 전략을 `correlation-driven contrast`와 `data mining`으로 구분한다.

- MedCLIP은 보고서 간 semantic correlation을 soft target으로 써서 false negative 문제를 줄인다.
- SAT는 pair를 positive, neutral, negative로 세분화해 더 정교하게 contrastive learning을 수행한다.
- MGCA는 질병 수준 semantic cluster를 도입해 disease-level 정렬을 강화한다.
- BioViL, BioViL-T, CXR-CLIP 등은 Findings/Impression 활용, 문장 섞기, uncertainty 활용, 시간 축 정보 활용 등으로 적은 데이터의 정보를 더 풍부하게 사용한다.

의료 데이터에서는 서로 다른 환자라도 "정상" 또는 유사 소견을 공유하는 경우가 많아, 원래 CLIP의 in-batch negative 가정이 잘 맞지 않는다. 이 논문은 바로 이 지점이 의료용 CLIP 설계에서 매우 중요하다고 지적한다.

### 3. Explicit knowledge enhancement

저자들은 knowledge enhancement를 subject-level과 domain-level로 나눈다.

- subject-level에서는 개별 보고서에서 엔티티와 관계를 추출해 entity-relation graph를 만들고, 이를 통해 한 환자 내부의 의미 구조를 보강한다.
- domain-level에서는 UMLS, RadGraph, 지식 프롬프트, 도메인 그래프 등을 사용해 영상-텍스트 정렬에 외부 의학 지식을 주입한다.

이는 의료 CLIP이 단순 co-occurrence 학습을 넘어서 임상 개념의 구조를 반영하도록 만드는 방향이다. 특히 분포 이동, 희귀질환, 장비 차이 같은 실제 임상 문제에 대응하려면 이 축이 중요하다는 것이 저자들의 관점이다.

## CLIP-driven Applications 정리

논문은 사전학습된 CLIP이 실제 임상 과제에서 어떻게 활용되는지 분류, dense prediction, cross-modal tasks로 정리한다.

### 1. Classification

분류는 zero-shot classification과 context optimization으로 나뉜다.

- zero-shot 분류에서는 질환명, 정상/비정상 설명, 해부학적 표현을 prompt로 만들어 분류를 수행한다.
- Xplainer 같은 연구는 질병명을 직접 맞히기보다 관찰 가능한 소견을 먼저 예측한 뒤 이를 결합해 최종 진단을 만든다.
- 이는 단순 정확도 향상보다 설명 가능성 측면에서 특히 중요하다.
- context optimization 계열은 의료 도메인 특성에 맞는 learnable prompt를 설계해 성능을 높인다.

저자들은 CLIP 기반 분류의 장점을 "사람이 읽을 수 있는 텍스트 개념을 통해 분류 논리를 노출할 수 있다"는 점으로 본다.

### 2. Dense prediction

dense prediction에는 탐지, 이상 탐지, 2D 분할, 3D 분할, 키포인트 위치 추정 등이 포함된다.

- GLIP 계열은 탐지에서 텍스트 설명과 박스 예측을 연결한다.
- AnomalyCLIP, MediCLIP 등은 정상/비정상 prompt를 이용해 zero-shot anomaly detection을 수행한다.
- 2D 분할에서는 CLIP 이미지 인코더 또는 image-text encoder를 fine-tuning해 segmentation backbone으로 활용한다.
- 3D 분할에서는 CLIP text embedding을 task representation으로 사용해 partially labeled multi-organ segmentation 문제를 개선한다.
- CLIP-driven universal segmentation은 one-hot task label 대신 의미가 있는 text embedding을 써서 organ-tumor 관계를 더 잘 반영한다.

이 부분은 CLIP이 단지 classification backbone이 아니라, task condition이나 label semantics를 제공하는 모듈로도 강력하다는 점을 보여 준다.

### 3. Cross-modal tasks

cross-modal task에는 보고서 생성, 이미지 생성, MedVQA, image-text retrieval이 포함된다.

- 보고서 생성에서는 CLIP의 semantic-aware visual representation을 이용해 중요한 소견을 더 잘 추출한다.
- MRI synthesis 같은 생성 과제에서는 텍스트 조건을 통해 원하는 영상 속성이나 acquisition setting을 반영한다.
- MedVQA에서는 이미지와 질문의 정렬을 강화하는 역할로 CLIP이 사용된다.
- retrieval에서는 의료 영상의 국소 특징을 반영하는 CLIP 기반 검색이 downstream 분류, 리포트 검색, 이상 탐지에 도움을 준다.

특히 저자들은 retrieval과 report generation을 통해 CLIP이 진단 보조 시스템의 실용적 인터페이스로 이어질 가능성을 높게 평가한다.

## 비교 분석과 해석

이 논문의 좋은 점은 단순한 논문 목록 정리에 머물지 않고, CLIP 기반 접근이 기존 의료 영상 방법과 어디서 본질적으로 다른지를 분명히 설명한다는 점이다.

- 기존 image-only 모델은 높은 성능을 내더라도 왜 그런 결론에 도달했는지 설명하기 어렵다.
- CLIP 기반 모델은 텍스트 개념을 매개로 사용하기 때문에 zero-shot, explainability, retrieval augmentation에서 구조적 이점이 있다.
- 반면 CLIP이 항상 우월한 것은 아니며, modality 특화 구조가 필요한 3D 영상, 세밀한 병변 경계, 데이터 부족 환경에서는 추가 설계가 필수적이다.

즉, 이 survey는 CLIP을 만능 해법으로 보지 않고, "의료 영상에서 언어 감독이 특히 큰 가치를 가지는 지점"을 구분해 보여 준다.

## 한계와 향후 연구 방향

저자들은 향후 과제를 비교적 구체적으로 제시한다.

### 1. 3D 및 고해상도 의료 영상에 대한 확장

현재 많은 연구가 흉부 X-ray에 편중되어 있고, CLIP 구조 자체도 2D 자연 이미지에서 출발했다. CT, MRI, PET처럼 3D 구조와 긴 문맥을 다루는 분야에서는 메모리 문제, 해부학적 장거리 의존성, volume-level alignment가 핵심 과제로 남는다.

### 2. 데이터와 언어 편향

공개 데이터는 특정 국가, 기관, 언어, 장비, 임상 워크플로우에 편중되어 있다. 보고서 언어도 영어뿐 아니라 스페인어, 중국어 등이 섞여 있어 cross-lingual bias 문제가 발생할 수 있다. 저자들은 공정성과 일반화 문제를 앞으로 더 본격적으로 다뤄야 한다고 본다.

### 3. 평가 체계의 미성숙

현재 많은 연구가 특정 다운스트림 task 성능으로만 의료용 CLIP의 품질을 평가한다. 그러나 foundation model 관점에서는 zero-shot 성능, calibration, explanation fidelity, retrieval quality, robustness, fairness를 함께 평가하는 더 넓은 벤치마크가 필요하다.

### 4. 임상 지식 통합의 고도화

현재의 knowledge enhancement는 아직 보조적 수준에 머무는 경우가 많다. 향후에는 지식 그래프, 의료 온톨로지, LLM 기반 보고서 구조화, 다기관 임상 프로토콜 정보를 함께 사용하는 방향이 중요하다.

### 5. 멀티모달 의료 AI로의 확장

영상과 판독문만이 아니라 EHR, 검사 수치, 유전체, 병리, 음성, 수술 비디오 등을 함께 다루는 generalist medical AI로 확장될 가능성이 크다. 이 survey는 CLIP이 그 출발점이 될 수 있다고 보지만, 단순 이미지-텍스트 contrast만으로는 부족하며 더 풍부한 융합 구조가 필요하다고 시사한다.

## 비판적 평가

이 논문은 의료 영상 CLIP 분야의 지형을 이해하는 데 매우 유용한 survey다. 특히 taxonomy가 분명하고, 단순 pre-training 기법뿐 아니라 downstream clinical tasks까지 이어지는 흐름을 잘 보여 준다. CLIP 연구를 막 시작하는 연구자에게는 어떤 문제를 풀어야 하는지 지도를 제공하는 역할을 한다.

다만 몇 가지 한계도 있다.

- survey의 폭이 매우 넓은 대신, 각 대표 논문의 실험 조건과 성능 비교는 깊게 통일되어 있지 않다.
- modality별 불균형이 커서 chest X-ray 중심 시야가 상대적으로 강하다.
- 2025년 기준 최신 논문까지 폭넓게 넣었지만, 개별 방법 간 공정 비교를 위한 정량 메타분석까지는 제공하지 않는다.
- CLIP 이후 등장한 더 큰 vision-language foundation model이나 의료 특화 multimodal LLM과의 관계는 제한적으로만 다룬다.

그럼에도 불구하고 이 논문의 가치는 명확하다. 의료 영상에서 CLIP을 "자연 이미지용 모델의 단순 이식"이 아니라, 의료 텍스트와 영상의 구조적 특성을 반영해 재설계해야 하는 연구 분야로 정의했다는 점이다.

## 연구적 시사점

이 논문이 주는 가장 중요한 메시지는 다음과 같다.

- 의료 영상에서 CLIP의 성패는 단순한 모델 크기보다, 의료 보고서 구조와 해부학적 구조를 얼마나 잘 반영하느냐에 달려 있다.
- 의료 도메인에서는 false negative, report structure, domain knowledge 같은 문제가 자연 이미지보다 훨씬 중요하다.
- downstream 성능뿐 아니라 explainability와 인간-모델 정렬이 CLIP의 본질적 장점이다.
- 앞으로의 핵심 방향은 3D, multilingual, multimodal, clinically grounded evaluation이다.

따라서 이 survey는 의료 vision-language 연구를 시작할 때 읽어야 할 입문 문헌일 뿐 아니라, 새로운 연구 주제를 설계할 때도 직접적인 아이디어 소스로 활용할 수 있다.

## 종합 평가

`CLIP in Medical Imaging: A Survey`는 의료 영상 분야에서 CLIP 계열 연구를 가장 체계적으로 정리한 문헌 중 하나다. CLIP 사전학습 적응 전략, 실제 임상 과제 적용, 한계와 미래 과제를 하나의 프레임으로 연결했다는 점에서 가치가 크다. 특히 multi-scale alignment, data-efficient contrast, knowledge enhancement라는 세 가지 축은 이후 의료용 vision-language 모델 연구를 설계할 때도 유효한 기준점으로 작동한다.

의료 영상에서 CLIP의 역할과 한계를 균형 있게 이해하려면, 이 논문은 매우 좋은 출발점이다.
