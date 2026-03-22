# EVF-SAM: Early Vision-Language Fusion for Text-Prompted Segment Anything Model

이 논문은 Segment Anything Model(SAM)의 강력한 **visual prompt 기반 분할 능력**을 **text prompt 기반 referring expression segmentation(RES)**으로 확장하는 문제를 다룬다. 저자들의 핵심 질문은 단순하다. “SAM에 텍스트를 넣으려면 어떤 종류의 텍스트/멀티모달 인코더가 가장 적합한가?” 그리고 그 답으로, **입력 이미지와 텍스트를 함께 넣고, 인코더 내부에서 일찍 융합하는 early vision-language fusion 방식**이 text-only encoder나 LLM 기반 prompting보다 더 효과적이라는 실험적 결론을 제시한다. 이를 바탕으로 제안된 모델이 **EVF-SAM**이며, BEIT-3 같은 early-fusion 멀티모달 인코더와 SAM을 단순한 projector로 연결한 구조다. 저자들은 이 방식이 RefCOCO/+/g에서 SOTA급 성능을 내면서도, 기존 LMM/LLM 기반 SAM 방식보다 훨씬 가볍고 안정적이라고 주장한다.  

## 1. Paper Overview

이 논문이 풀고자 하는 문제는 **SAM의 text-prompted segmentation 능력 부재**다. SAM은 point, box, mask 같은 기하학적 prompt에 대해서는 매우 강하지만, “빨간 셔츠를 입은 사람”처럼 언어로 대상 객체를 지정하는 referring expression segmentation에는 직접 대응하지 못한다. 기존에는 Grounded-SAM처럼 텍스트로 박스를 먼저 찾고 그 박스를 SAM에 넣는 2-stage 구조, CLIP류 text encoder를 그대로 붙이는 방식, 혹은 LLaVA/LISA 같은 LLM/LMM을 사용해 토큰 기반으로 프롬프트 임베딩을 생성하는 방식이 시도되었다. 그러나 이런 방법들은 각각 **비-end-to-end 구조**, **semantic gap**, **높은 계산량과 템플릿 의존성**이라는 한계를 가진다.

저자들은 이 문제가 중요한 이유를, SAM이 이미 강력한 segmentation foundation model이기 때문이라고 본다. 즉 문제는 “좋은 segmentation backbone을 새로 만들자”가 아니라, **기존의 강력한 SAM을 어떻게 언어 지시를 따르는 모델로 바꿀 것인가**이다. 그래서 논문은 새로운 대규모 end-to-end segmentation 모델보다, **SAM에 가장 잘 맞는 text/multimodal prompting 방식** 자체를 실험적으로 탐색한다.

## 2. Core Idea

이 논문의 핵심 아이디어는 두 가지다.

첫째, **text-only prompt보다 image+text를 함께 쓰는 multimodal prompt가 낫다**는 점이다. 저자들은 단순 텍스트 임베딩만으로는 텍스트 표현과 실제 이미지 영역을 충분히 정렬하기 어렵다고 본다. 특히 segmentation처럼 fine-grained localization이 필요한 작업에서는 입력 이미지가 함께 들어가야 어떤 단어가 어떤 영역을 가리키는지 더 정확히 정렬할 수 있다고 주장한다. 실제로 RefCOCO testA에서 CLIP text-only는 63.4 cIoU, CLIP text+image는 67.9 cIoU, BEIT-3 text-only는 65.1 cIoU, BEIT-3 text+image는 83.7 cIoU로 보고되어 멀티모달 입력의 이점이 매우 크게 나타난다.

둘째, **late fusion보다 early fusion이 낫다**는 점이다. 저자들은 CLIP이나 LLaVA처럼 text/image를 따로 인코딩한 후 뒤에서 합치는 late fusion보다, ViLT나 BEIT-3처럼 인코더 내부 self-attention 단계에서 텍스트와 이미지를 일찍 섞는 early fusion이 prompt 품질을 더 잘 만든다고 해석한다. 이 early fusion은 텍스트 표현을 더 풍부하게 만들 뿐 아니라, 이미지 쪽도 텍스트에 맞는 영역을 더 잘 모으게 해 SAM에 넣을 prompt embedding 품질을 높인다는 것이 저자들의 설명이다.

정리하면 이 논문의 novelty는 “SAM+LLM”이 아니라 오히려 그 반대에 가깝다. 즉, **무거운 autoregressive LLM 대신, 가벼운 encoder-based early-fusion vision-language model이 SAM prompting에 더 적합하다**는 점을 실험으로 보여준 것이 핵심이다.

## 3. Detailed Method Explanation

### 3.1 문제 재정의: SAM을 RES로 확장하기

논문은 text-prompted SAM 문제를 실질적으로 **Referring Expression Segmentation(RES)** 문제로 정식화한다. 사용자가 텍스트 설명을 주면, 모델은 그 설명에 해당하는 객체의 pixel-wise mask를 예측해야 한다. 이때 기존 SAM은 point/box/mask prompt만 처리하므로, 텍스트를 **SAM prompt encoder가 이해할 수 있는 sparse embedding**으로 바꾸는 것이 핵심이 된다.

### 3.2 EVF-SAM의 전체 구조

EVF-SAM은 크게 세 모듈로 구성된다.

* **Multimodal Encoder**
* **Projector**
* **Segment Anything Model (SAM)**

Multimodal Encoder는 입력 이미지와 텍스트를 함께 받아 fused multimodal embedding을 만든다. Projector는 이 임베딩 차원을 SAM이 요구하는 prompt embedding 차원으로 변환한다. 마지막으로 SAM은 원래 구조를 거의 유지한 채, prompt encoder에서 이 멀티모달 프롬프트를 추가로 받아 mask decoder로 segmentation mask를 생성한다. 저자들은 이 구조가 복잡한 추가 모듈 없이도 잘 작동한다고 강조한다.

### 3.3 Multimodal Encoder: BEIT-3 기반 early fusion

논문에서 기본 Multimodal Encoder는 **BEIT-3**다. 텍스트는 XLM-Roberta tokenizer로 토큰화되고, 이미지는 $224 \times 224$로 resize된 뒤 patch embedding으로 들어간다. 중요한 점은 텍스트 토큰과 이미지 토큰이 **attention block 내부에서 상호작용**한다는 것이다. 즉 이 모델은 텍스트와 이미지를 따로 encoding한 뒤 뒤에서 이어붙이는 late fusion이 아니라, 인코더 내부에서 cross-modal fusion을 반복하는 early fusion 구조다. 저자들은 최종적으로 `[CLS]` 토큰을 multimodal embedding으로 사용하고, 이것을 SAM용 prompt로 투사한다.

### 3.4 Projector: 단순한 2-layer MLP

서로 다른 foundation model은 임베딩 차원이 다르다. 예를 들어 BEIT-3-Large는 1024차원, Base는 768차원, SAM mask decoder는 256차원을 사용한다. 그래서 저자들은 중간에 **2-layer MLP + ReLU projector**를 둔다. 이 projector의 역할은 단순하지만 중요하다. 저자들은 일부러 elaborate module을 넣지 않았는데, 이유는 다음과 같다.

* 단순 MLP만으로도 충분히 효과적이다.
* 학습/추론 효율이 좋다.
* foundation model의 사전학습 지식을 덜 훼손한다.

즉 이 논문은 복잡한 adapter 설계보다 **좋은 multimodal encoder 선택 자체가 더 중요하다**는 입장이다.

### 3.5 SAM adaptation: image encoder는 유지, prompt encoder만 확장

SAM은 원래 다음 세 모듈을 가진다.

* Image Encoder
* Prompt Encoder
* Mask Decoder

EVF-SAM에서는 **SAM image encoder는 그대로 유지하고 frozen**한다. 텍스트와 저해상도 이미지를 본 Multimodal Encoder가 만든 embedding을 projector를 통해 SAM prompt encoder에 넣고, 이것을 기존 sparse embedding과 concatenate하여 mask decoder에 전달한다. 논문에 따르면 원래 point/box prompt는 $R^{B \times N \times D}$ 형태의 sparse embedding인데, EVF-SAM에서는 multimodal embedding $R^{B \times 1 \times D}$를 여기에 붙여 넣는 식으로 확장한다. 즉 구조적으로는 **SAM의 분할 능력을 해치지 않으면서 prompt만 바꾸는 방식**이다.

### 3.6 Training strategy

학습 전략도 이 논문의 중요한 포인트다.

첫째, **template-free**다. LISA 같은 LLM 기반 방식은 “Can you segment {object} in the picture?” 같은 instruction template와 “It is [SEG].” 같은 answer template에 의존한다. 하지만 EVF-SAM은 이런 질의응답 템플릿이 아예 필요 없다. 사용자의 표현 문장 자체를 그대로 입력으로 쓴다. 저자들은 이것이 학습과 추론을 단순화하고, 사용자 질문 문법 변화에 덜 민감하다고 본다.

둘째, **trainable modules**는 Multimodal Encoder 전체와, SAM의 prompt encoder 및 mask decoder다. 반면 SAM image encoder는 freeze한다. 흥미롭게도 저자들은 prompt encoder와 mask decoder를 얼려도 성능 저하가 크지 않다고 언급하는데, 이는 SAM의 원래 segmentation prior가 꽤 강하다는 점을 시사한다.

셋째, 구현 세부는 비교적 간단하다. 기본 설정은 SAM-ViT-Huge와 BEIT-3-Large를 초기화에 사용하고, 4개의 NVIDIA L40s GPU에서 mixed precision + DeepSpeed ZeRO-2로 학습한다. 배치 크기는 총 128, optimizer는 AdamW, learning rate는 1e-4 선형 감쇠, 총 15k iteration, loss는 BCE와 Dice loss를 동일 가중치로 사용한다.

## 4. Experiments and Findings

### 4.1 데이터셋과 지표

실험은 RefCLEF, RefCOCO, RefCOCO+, RefCOCOg에서 수행된다. 이 중 RefCOCOg는 긴 수식 표현(longer expressions)이 많고, RefCOCO의 testA는 human-centric, testB는 common object 중심이다. 지표는 gIoU와 cIoU를 사용하며, 논문은 기본적으로 **cIoU를 주 지표**로 보고한다.

### 4.2 핵심 비교: 어떤 prompt encoder가 SAM에 가장 잘 맞는가

논문 초반의 동기 분석에서 이미 강한 결과가 나온다. RefCOCO 기준 cIoU는 다음과 같다.

* CLIP (Text): **63.4**
* CLIP (Text+Image): **67.9**
* BEIT-3 (Text): **65.1**
* BEIT-3 (Text+Image): **83.7**
* LLaVA (Text+Image): **79.1**

이 표는 논문의 핵심 주장을 아주 직접적으로 뒷받침한다. 단순 text-only encoder는 SAM과 semantic gap이 크고, image+text multimodal input이 더 좋으며, 그중에서도 **early-fusion 계열인 BEIT-3가 late-fusion/LLM 계열보다 더 우수**하다는 것이다. 특히 BEIT-3 text+image가 LLaVA보다도 높다는 점은 “큰 autoregressive LLM이 항상 더 낫지 않다”는 중요한 메시지다.

### 4.3 전체 benchmark 결과

논문은 EVF-SAM이 RefCOCO/+/g에서 높은 평균 성능을 달성하며, 특히 **RefCOCOg 같은 긴 표현에서 강점**을 보인다고 말한다. 저자들은 Table II를 요약하며, 기존 text-encoder 기반 전통 모델들은 가볍지만 성능이 상대적으로 낮거나 대량 데이터가 필요하고, 최근 LMM 기반 모델들은 성능은 좋지만 계산량이 매우 크다고 정리한다. 이에 비해 EVF-SAM은 **제한된 학습 데이터와 관리 가능한 계산량으로 전체 benchmark 평균 cIoU 최고 수준**을 달성했다고 주장한다.

논문이 특히 강조하는 부분은 RefCOCOg다. 저자들 스스로 RefCOCOg 성능이 높다는 점을 “긴 텍스트를 이해하는 능력이 강하다는 반직관적 결과”로 해석한다. 즉 LLaVA/LISA 같은 거대 LLM 기반 방법이 긴 표현에 더 유리할 것 같지만, 실제로는 **lightweight early-fusion vision-language encoder가 더 나은 prompt를 만들어 SAM에 전달**할 수 있다는 것이다.

### 4.4 Ablation: multimodal input과 early fusion의 효과

논문에서 가장 설득력 있는 부분은 ablation study다. 저자들은 여러 CLIP/ViLT/BEIT-3 설정과 late fusion/early fusion을 비교한다. 결과 요약은 다음과 같다.

* text-only encoder는 성능이 제한적이다.
* image+text multimodal encoder는 큰 폭의 성능 향상을 준다.
* early fusion은 late fusion보다 더 좋다.
* 특히 ViLT와 BEIT-3 같은 early-fusion 모델에서 improvement가 크다.

구체적으로 논문은 multimodal input이 CLIP-Large(OpenAI), CLIP-Large(OpenCLIP), CLIP-Huge, ViLT, BEIT-3에서 각각 **4.5, 4.6, 4.0, 1.0, 4.5 cIoU** 향상을 가져왔다고 설명한다. 또 ViLT early fusion은 text-only baseline 대비 **11.1 cIoU** 개선을 보였고, BEIT-3도 former 12 layers만 fusion하거나 all 24 layers를 fusion하는 설정 모두 late fusion/text-only보다 확실히 더 좋았다고 한다.  

이 결과는 단순히 “멀티모달이면 좋다”가 아니라, **어디서 어떻게 융합하느냐가 중요하다**는 점을 보여준다. 저자들의 해석대로라면, segmentation prompt는 최종 출력 직전에 억지로 붙이는 late fusion보다, encoder 내부 단계에서부터 텍스트와 이미지가 반복적으로 상호작용하면서 만들어져야 SAM에 유리한 형태가 된다.

### 4.5 Foundation model 선택의 효과

저자들은 foundation model 교체도 실험한다. CLIP-Large(text only), ViLT, BEIT-3-Large, BEIT-3-Base, 그리고 SAM-H 대신 Efficient-SAM-S를 조합해 본다. 여기서 두 가지 메시지가 나온다.

첫째, **더 좋은 multimodal encoder가 더 좋은 SAM prompt를 만든다**. 특히 BEIT-3-Base는 성능이 크게 떨어졌고, 이는 SAM adaptation에서 encoder 품질이 매우 중요함을 뜻한다. 둘째, **SAM backbone 자체를 가볍게 바꿔도 EVF-SAM의 강점은 유지된다**. 논문은 Efficient-SAM-S와 SAM-H 사이 차이가 매우 작다고 보고하며, 이는 향후 text-prompted SAM 연구에서 “SAM만 키우는 것보다 멀티모달 인코더를 더 좋게 만드는 것이 더 중요하다”는 통찰로 이어진다고 말한다.  

### 4.6 정성적 결과

정성적 비교에서도 EVF-SAM은 RefCOCO의 짧은 표현에서는 경계가 더 정확하고, RefCOCOg의 긴 표현에서는 vanilla CLIP-prompted SAM이나 LISA보다 더 안정적인 분할을 보인다고 설명된다. 논문은 특히 “the umbrella closest to the camera”처럼 **공간관계가 포함된 긴 표현**도 EVF-SAM이 잘 이해한다고 서술한다. 이는 EVF-SAM이 단순 객체명 매칭이 아니라, 어느 정도 **fine-grained referential grounding**을 수행함을 시사한다.  

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 **문제 설정이 명확하고, 답도 단순하며 설득력 있다는 점**이다. “SAM에 텍스트를 어떻게 넣는가?”라는 질문에 대해, 저자들은 복잡한 LLM pipeline 대신 **early-fusion 멀티모달 encoder + simple projector**라는 간단한 구조를 제시한다. 그리고 이 단순함이 실제로 강한 성능과 효율성으로 이어졌다는 점이 논문의 미덕이다.

또 다른 강점은 **모듈성**이다. EVF-SAM은 SAM을 통째로 뜯어고치지 않고 prompt path만 확장한다. 그래서 SAM variant 교체나 더 나은 multimodal encoder로의 확장이 상대적으로 쉽다. 저자들이 foundation model ablation을 별도로 수행한 것도 이 구조적 장점을 보여주기 위함이다.

세 번째 강점은 **효율성과 실용성**이다. EVF-SAM은 1.32B 파라미터로 기존 LLM/LMM 기반 SAM 방법 대비 약 **82% 파라미터 감소**를 달성하면서도 더 높은 성능을 보였다고 주장한다. 게다가 template-free라 instruction engineering 부담이 적고, encoder-based라 sequence length가 안정적이라 학습/추론이 단순하다.

### Limitations

한계도 분명하다.

첫째, 이 논문은 본질적으로 **RES benchmark 중심**이다. 즉 text-prompted SAM의 일반적 가능성을 보여주지만, open-vocabulary semantic segmentation이나 more free-form instruction following 전체로 바로 일반화되지는 않는다. 저자들이 다루는 텍스트는 주로 referring expression이고, 이는 대화형 멀티모달 에이전트가 처리하는 훨씬 자유로운 언어 지시와는 다르다.

둘째, image encoder는 얼리고 prompt path 위주로 adaptation하는 구조이기 때문에, **SAM 내부 표현 자체가 텍스트 조건에 맞게 재조직되는 정도는 제한적**일 수 있다. 즉 현재 방식은 강한 foundation prior를 이용한 lightweight adaptation이지만, 더 깊은 language-grounded segmentation으로 가려면 image encoder까지 건드리는 방식이 필요할 가능성도 있다.

셋째, early fusion의 우수성이 RefCOCO 계열에서 매우 강하게 나오지만, 그것이 다른 데이터셋이나 더 복잡한 multimodal instruction setting에서도 동일하게 유지될지는 추가 검증이 필요하다.

### Interpretation

비판적으로 보면, 이 논문의 진짜 공헌은 “새로운 gigantic model”이 아니다. 오히려 **SAM adaptation에서는 LLM보다 더 적절한 inductive bias를 가진 멀티모달 encoder가 중요하다**는 점을 보여준 데 있다. 이는 최근 멀티모달 연구에서 종종 나타나는 “모든 문제를 autoregressive LLM으로 푸는 접근”에 대한 좋은 반례다. 이 논문은 segmentation prompt 생성이라는 하위 문제에서는, 크기와 범용성보다 **정렬 방식(alignment mechanism)**과 **fusion 위치**가 더 중요할 수 있음을 보여준다.

## 6. Conclusion

EVF-SAM은 SAM의 text-prompted segmentation 능력을 확장하기 위해, **입력 이미지와 텍스트를 함께 처리하는 early vision-language fusion encoder**를 prompt generator로 사용하는 간단한 프레임워크다. 핵심 발견은 다음 두 가지다.

* **멀티모달 입력(text+image)**이 text-only 입력보다 더 좋다.
* **early fusion** 이 late fusion이나 autoregressive LLM 기반 prompting보다 더 좋다.

이를 바탕으로 EVF-SAM은 RefCOCO/+/g에서 강한 성능을 달성했고, 특히 긴 표현이 많은 RefCOCOg에서도 우수함을 보였다. 동시에 기존 LMM/LLM 기반 SAM 방법보다 훨씬 적은 파라미터와 더 낮은 템플릿 의존성을 유지했다. 따라서 이 논문은 “SAM에 텍스트를 붙이는 가장 효과적인 길 중 하나는 거대한 생성형 LLM이 아니라, **잘 정렬된 early-fusion encoder**”라는 점을 설득력 있게 보여준다고 평가할 수 있다.  
