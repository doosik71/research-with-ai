# OmniMedVQA: A New Large-Scale Comprehensive Evaluation Benchmark for Medical LVLM

- **저자**: Yutao Hu, Tianbin Li, Quanfeng Lu, Wenqi Shao, Junjun He, Yu Qiao, Ping Luo
- **발표연도**: 2024
- **arXiv**: https://arxiv.org/abs/2402.09181

## 1. 논문 개요

이 논문은 의료 분야에서 Large Vision-Language Models(LVLMs)를 제대로 평가할 수 있는 대규모 벤치마크가 부족하다는 문제의식에서 출발한다. 저자들은 기존 medical VQA 데이터셋들이 이미지 수가 적고, modality 수가 제한적이며, 인체 해부학적 범위도 좁아서, 실제 의료 환경을 반영하는 종합 평가에는 적합하지 않다고 본다. 이를 해결하기 위해 제안된 것이 **OmniMedVQA**이며, 이는 의료 영상 기반 질의응답 능력을 평가하기 위한 대규모 benchmark이다.

논문이 다루는 핵심 연구 문제는 다음과 같다. 현재의 general-domain LVLM과 medical-specialized LVLM이 실제 의료 영상에 대해 얼마나 폭넓고 안정적으로 이해하고 답변할 수 있는가? 이를 검증하려면 MRI, CT, X-Ray, fundus photography, histopathology, ultrasound 등 서로 매우 다른 modality와 다양한 anatomical region을 포괄하는 benchmark가 필요하다. 저자들은 바로 이 점이 빠져 있었기 때문에, 지금까지 제안된 의료 LVLM의 실제 성능을 충분히 이해하기 어려웠다고 주장한다.

이 문제가 중요한 이유는 의료 영상은 일반 이미지와 분포가 크게 다르고, modality마다 시각적 패턴과 해석 방식이 매우 다르기 때문이다. 예를 들어 CT와 fundus photography는 시각적 특성이 전혀 다르며, 병변의 표현 방식도 다르다. 따라서 의료 LVLM이 실제로 의료적 지식을 학습했는지, 아니면 일부 제한된 데이터셋에서만 잘 작동하는지 확인하려면, 넓은 범위의 real medical images를 사용한 평가가 필수적이다. OmniMedVQA는 바로 이런 필요를 충족시키려는 시도다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **기존 classification dataset들을 VQA 형식으로 변환해, 대규모이면서도 실제 의료 영상을 기반으로 한 종합 VQA benchmark를 만들자**는 것이다. 의료 분야에서는 이미지-텍스트 paired data를 대규모로 모으기 어렵기 때문에, 이미 존재하는 73개의 의료 classification dataset을 활용하고, 그 안에 들어 있는 class label, modality, anatomy, lesion severity, 기타 biological attribute를 이용해 질문과 정답을 구성한다.

이 접근의 핵심 직관은, 비록 classification label이 간결해 보여도 그것이 의료 LVLM의 기초 능력을 평가하는 데는 매우 유용하다는 점이다. 예를 들어 어떤 영상이 CT인지 MRI인지 구분하는 능력, 어느 anatomical region인지 식별하는 능력, 질병을 판별하는 능력, 병변의 severity를 구분하는 능력은 모두 의료 multimodal understanding의 기본 요소다. 저자들은 이런 기본 능력조차 현존 모델들이 충분히 갖추지 못했다고 본다.

기존 접근과의 차별점은 크게 세 가지다. 첫째, **규모**가 크다. 논문에 따르면 OmniMedVQA는 118,010장의 이미지와 127,995개의 QA item을 포함한다. 둘째, **다양성**이 크다. 12개 modality와 20개 이상의 anatomical region을 포괄한다. 셋째, **실제 의료 이미지(real medical images)**를 사용한다는 점이다. 저자들은 PMC-VQA처럼 논문에서 추출한 이미지와 텍스트를 사용하는 방식은 이미지 압축이나 실제 임상 데이터와의 괴리가 발생할 수 있다고 지적한다. OmniMedVQA는 이런 간극을 줄이려 한다.

또 하나의 중요한 아이디어는 evaluation 설계에 있다. 모델들이 multiple-choice 형식의 instruction을 잘 따르지 못할 수 있기 때문에, 저자들은 단순 응답 정확도뿐 아니라 **Prefix-based Score**도 함께 사용한다. 이는 모델이 각 선택지를 얼마나 그럴듯한 답으로 보는지를 likelihood 수준에서 평가하려는 시도다. 즉, 모델이 답을 “잘 말하지는 못해도”, 내부적으로는 어느 정도 맞는 지식을 가지고 있는지를 따로 측정하려는 것이다.

## 3. 상세 방법 설명

전체 파이프라인은 크게 네 단계로 구성된다.

첫 번째 단계는 **원본 데이터 수집(original dataset preparing)**이다. 저자들은 73개의 의료 classification dataset을 수집했고, 이들 데이터는 12가지 modality와 20개 이상의 anatomical region에 걸쳐 있다. 본문과 부록에 따르면 modality에는 Colposcopy, CT, Digital Photography, Fundus Photography, Infrared Reflectance Imaging, MR, OCT, Dermoscopy, Endoscopy, Microscopy Images, X-Ray, Ultrasound가 포함된다. 해부학적 범위도 lung, mammary gland, eye, uterus, intestine, skin, kidney, liver, pelvic, oral cavity, knee, foot 등 매우 넓다.

두 번째 단계는 **QA template 설계**다. 각 classification dataset이 가진 label이나 속성을 기반으로 질문-정답 템플릿을 만든다. 예를 들어 특정 세포 이미지 분류 데이터셋이라면 “이 이미지에 있는 암세포의 구체적 진단은 무엇인가?” 같은 질문을 만들 수 있고, CT 데이터셋이라면 “이 영상의 modality는 무엇인가?” 혹은 “이 영상에서 이상이 있는 기관은 어디인가?” 같은 질문을 만들 수 있다. 논문은 이 QA들을 다섯 개 유형으로 정리한다.

1. Modality Recognition  
2. Anatomy Identification  
3. Disease Diagnosis  
4. Lesion Grading  
5. Other Biological Attributes

이 다섯 유형은 의료 LVLM의 서로 다른 능력을 평가하도록 설계되었다. 예를 들어 Lesion Grading은 병변의 심각도를 평가하는 능력을, Other Biological Attributes는 세포 종류, 암 상태, 촬영 방향 등 다양한 속성 이해 능력을 본다. Table 3에 따르면 QA item 수는 Disease Diagnosis가 73,455개로 가장 많고, Modality Recognition 19,427개, Anatomy Identification 20,330개, Lesion Grading 2,621개, Other Biological Attributes 12,162개다.

세 번째 단계는 **데이터 균형화와 QA 정제(refine QA pairs)**다. 저자들은 특정 템플릿이나 특정 대형 데이터셋이 과도하게 많은 샘플을 차지하지 않도록 **Inverse Proportional Sampling** 전략을 사용했다고 설명한다. 즉, 어떤 템플릿에 연결된 원본 이미지 수가 많을수록 샘플링 비율을 더 낮춰 전체 분포를 균형 있게 맞춘다. 논문은 이를 통해 category bias와 중복된 QA 편향을 줄이려 했다고 말한다.

이후 ChatGPT-3.5 API를 사용해 각 질문을 **의미는 유지하면서 다른 표현으로 재구성**하고, 동시에 **오답 선택지(incorrect options)**도 생성한다. 이렇게 해서 각 QA item을 2개에서 4개의 보기로 이루어진 multiple-choice 형태로 변환한다. 질문 표현을 다양화한 이유는 LVLM의 언어적 표현 변화에 대한 적응력을 평가하기 위해서다.

논문이 제시한 multi-choice 생성 프롬프트의 핵심은 다음과 같다. system prompt에서는 모델에게 “원래 medical QA pair를 받고, 같은 의미의 질문으로 바꾸고, 세 개의 오답 선택지를 생성하라”고 지시한다. user prompt에는 원래 question과 correct answer가 들어간다. 이 설계는 QA 생성 자체는 LLM을 활용하되, 최종 품질은 human double check로 보완했다는 점을 강조한다.

평가 방법은 두 가지다.

첫째는 **Question-answering Score**다. 입력 이미지와 질문, 그리고 선택지를 합쳐 하나의 prompt를 구성한 뒤 LVLM에 넣고 응답을 생성하게 한다. 프롬프트는 대략 “이것은 여러 옵션이 있는 의료 질문이며, 정답은 하나뿐이다. 질문에 대한 정답 옵션 하나만 선택하라”는 형식이다. 모델이 생성한 응답이 보기 중 어느 것과 가장 유사한지를 계산해 최종 선택지를 정하고, 이를 정답과 비교해 정확도를 계산한다.

둘째는 **Prefix-based Score**다. 이 방법은 생성된 자유 응답 자체보다, 각 선택지가 모델 내부에서 얼마나 높은 likelihood를 받는지를 측정한다. 논문 설명에 따르면 이미지의 visual feature와 텍스트 임베딩을 추출한 뒤, visual feature를 text embedding 앞에 prefix처럼 붙여 LLM에 넣고 likelihood score를 계산한다. 각 후보 선택지에 대해 이 점수를 계산해 가장 높은 점수를 받은 선택지를 정답으로 본다. 이를 수식으로 명시하지는 않았지만, 개념적으로는 다음과 같이 이해할 수 있다.

$$
\hat{a} = \arg\max_{a \in \mathcal{A}} \; p(a \mid I, q)
$$

여기서 $I$는 이미지, $q$는 질문, $\mathcal{A}$는 후보 선택지 집합, $\hat{a}$는 최종 예측 답이다. Prefix-based Score는 실제 생성 문장과 정확히 동일한 것은 아니지만, 각 보기를 모델이 얼마나 정답으로 간주하는지 반영하는 지표라고 저자들은 설명한다.

실험 설정에서는 총 12개의 LVLM을 zero-shot으로 평가한다. 일반 모델 8개는 BLIP2, LLaVA, LLaMA Adapter v2, MiniGPT-4, mPLUG-Owl, Otter, InstructBLIP, VPGTrans이며, 의료 특화 모델 4개는 Med-Flamingo, RadFM, MedVInT, LLaVA-Med이다. MedVInT는 open-ended와 multi-choice에 서로 다른 변형이 있어서, Question-answering Score에는 MedVInT-TD, Prefix-based Score에는 MedVInT-TE를 사용했다고 적혀 있다.

## 4. 실험 및 결과

실험은 전체 benchmark와 modality별 분석으로 나뉜다. 전체 benchmark에서는 다섯 개 question type에 대해 각 모델의 정확도를 보고하고, modality별 분석에서는 CT, MR, X-Ray, Fundus Photography, OCT 등 각 modality 단위로 성능을 따로 비교한다.

먼저 전체 성능(Table 5)을 보면, 이 벤치마크가 매우 어렵다는 점이 분명하다. random guess의 overall accuracy는 28.28인데, 많은 모델이 이보다 약간 높은 수준에 머문다. Question-answering Score 기준 overall accuracy는 BLIP-2가 50.69로 가장 높고, InstructBLIP가 42.49, MedVInT가 41.50, Med-Flamingo가 36.17 정도다. Prefix-based Score 기준 overall accuracy는 BLIP-2가 33.43으로 가장 높고, RadFM 29.00, mPLUG-Owl 29.89, LLaMA Adapter v2 29.88, InstructBLIP 28.71 등으로 전반적으로 낮다.

질문 유형별로 보면 BLIP-2는 Modality Recognition 57.51, Anatomy Identification 49.19, Disease Diagnosis 46.24, Other Biological Attributes 73.52로 특히 강하다. InstructBLIP는 Modality Recognition 70.62, Lesion Grading 54.60에서 눈에 띄는 수치를 보인다. 반면 MedVInT는 Modality Recognition 59.79, Anatomy Identification 41.36으로 강하지만 Lesion Grading은 15.49로 크게 떨어진다. RadFM은 Question-answering Score에서는 overall 26.82로 낮지만, Prefix-based Score에서는 CT 45.47, X-Ray 55.21로 강한 modality-specific 특성을 보여준다.

논문이 강조하는 가장 중요한 결과는 **general-domain LVLM이 의료 특화 LVLM보다 전체적으로 더 강하거나, 적어도 일관되게 뒤지지 않는다는 점**이다. 저자들은 특히 BLIP-2가 의료 모델들을 큰 폭으로 앞서는 점에 주목한다. 이는 단순히 의료 데이터를 추가로 instruction tuning했다고 해서 강한 medical LVLM이 자동으로 나오지는 않는다는 해석으로 이어진다.

Table 6과 Table 7의 modality별 분석은 더 흥미롭다. Question-answering Score 기준으로 BLIP-2는 CT 56.74, Fundus Photography 57.66, Infrared Reflectance Imaging 66.18, OCT 68.08, X-Ray 70.55 등 다수 modality에서 가장 강하다. 반면 MedVInT는 Digital Photography 43.89, MR 42.84, Ultrasound 41.26 등 일부 modality에서 강한 편이고, Med-Flamingo는 CT 38.47, MR 40.01, Microscopy 46.60에서 나쁘지 않다. Prefix-based Score 기준으로는 RadFM이 CT 45.47, X-Ray 55.21에서 최고 성능을 보이며, Med-Flamingo는 OCT 58.57, Endoscopy 46.61 등에서 상대적으로 강하다.

저자들은 여기서 두 가지 해석을 제시한다. 첫째, 의료 특화 LVLM은 전체 평균으로 보면 약하지만, **CT, MRI, X-Ray처럼 일반 이미지와 분포 차이가 큰 modality에서는 상대적으로 강할 수 있다.** 둘째, 특정 modality에 대한 대규모 학습 데이터가 있으면 그 영역에서는 성능 이득이 생긴다. 예를 들어 RadFM은 radiology image-text pair를 많이 사용했기 때문에 CT와 X-Ray에서 두드러진다. 그러나 fundus photography나 infrared reflectance imaging처럼 다양한 modality를 모두 잘 다루지는 못한다.

논문은 open-access subset에 대한 별도 결과도 부록에 제공한다. 이는 restricted dataset을 직접 내려받지 않아도 비교 실험을 일부 재현할 수 있도록 하기 위함이다. 다만 본문의 핵심 주장과 해석은 전체 benchmark 결과에 기반한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 benchmark 자체의 **포괄성**이다. 73개 데이터셋, 12개 modality, 20개 이상의 anatomical region, 118,010개 이미지, 127,995개 QA item은 기존 medical VQA benchmark보다 훨씬 넓은 범위를 커버한다. 특히 기존 VQA-RAD, SLAKE, Path-VQA, VQA-Med와 비교해 규모와 다양성 모두 크게 확장되었다는 점이 분명하다.

또 다른 강점은 **실제 의료 이미지 중심의 평가**라는 점이다. 저자들은 논문에서 추출한 figure 중심 데이터셋은 실제 의료 환경과 차이가 있을 수 있다고 보고, authentic medical scenarios에서 온 이미지들을 강조한다. 이 점은 benchmark의 실제성(realism)을 높인다.

평가 설계도 장점이다. 단순히 모델의 자유 생성 답변만 보지 않고, **Question-answering Score와 Prefix-based Score를 함께 사용**해 instruction-following 실패와 실제 지식 부족을 어느 정도 분리하려 한 점은 합리적이다. 의료 LVLM이 multiple-choice 포맷을 잘 따르지 못하더라도 내부적으로 어느 선택지를 더 가능성 높게 보는지 평가할 수 있기 때문이다.

그러나 한계도 분명하다. 첫째, benchmark의 질문 대부분은 classification attribute를 기반으로 생성되므로, **의료 추론의 깊이**를 평가하는 데는 제한이 있다. 예를 들어 장문의 임상적 reasoning, 병력과 영상의 종합 판단, 다단계 진단 추론 같은 것은 이 benchmark 범위에 명확히 포함되지 않는다. 저자들 역시 이 점을 인정하면서, 대신 foundational capability를 보는 기본 benchmark라고 위치시킨다.

둘째, QA와 오답 선택지 생성에 ChatGPT-3.5 API를 사용했다. human double check를 했다고는 하지만, 질문 표현이나 distractor 품질이 일정하지 않을 가능성은 남아 있다. 논문은 품질 관리 절차를 서술하지만, inter-annotator agreement나 구체적 오류율 같은 정량적 검증은 본문에 명시되어 있지 않다.

셋째, 본문 서술에 약간의 **표현상 불일치**가 있다. 초록과 결론, 그리고 다수의 결과 해석에서는 medical-specialized LVLM이 general-domain 모델보다 전반적으로 열세라고 말한다. 그런데 서론 중간에는 “medical-specialized LVLMs exhibit superior performance compared to general-domain LVLMs”라는 문장이 나타난다. 하지만 바로 이어지는 문맥과 표의 결과는 오히려 그 반대를 지지한다. 따라서 이 부분은 논문 서술상의 실수 또는 부정확한 표현으로 보이며, 보고서에서는 표와 나머지 본문 흐름에 근거해 해석하는 것이 타당하다.

넷째, Prefix-based Score는 모델의 내재 지식을 반영하려는 좋은 시도지만, 실제 사용자 관점에서 중요한 것은 결국 생성 응답의 정확성과 신뢰성이다. 따라서 이 지표는 “모델이 알고 있는가”를 부분적으로 보여줄 수는 있어도, 실제 임상 보조 시스템으로서의 사용 가능성을 직접 보장하지는 않는다.

비판적으로 보면, 이 논문은 “좋은 benchmark를 만들고 현재 모델들의 약점을 보여준다”는 목적에는 매우 충실하지만, benchmark 자체가 다루는 문제의 난이도는 주로 recognition/classification 기반이다. 그래서 이 결과만으로 의료 LVLM 전반의 고차 reasoning 능력을 결론 내리기는 어렵다. 그럼에도 불구하고, 적어도 현재 모델들이 의료 영상의 기본 분류적 이해조차 충분하지 않다는 사실은 꽤 설득력 있게 보여준다.

## 6. 결론

이 논문의 핵심 기여는 **OmniMedVQA라는 대규모 종합 의료 VQA benchmark를 구축하고, 이를 이용해 12개의 대표적 LVLM을 체계적으로 평가했다는 점**이다. 데이터셋은 12개 modality와 20개 이상의 anatomical region을 포괄하며, 실제 의료 이미지를 기반으로 한다. 이 규모와 범위는 의료 LVLM 평가의 기반을 넓혀 준다.

실험 결과는 다소 충격적이다. 의료 특화 모델들이 전반적으로 일반 모델보다 확실히 우월하지 않았고, 오히려 BLIP-2와 InstructBLIP 같은 general-domain LVLM이 더 강한 경우가 많았다. 의료 모델은 CT, MRI, X-Ray 같은 특정 modality에서는 장점을 보였지만, 다양한 의료 modality 전반에 걸쳐 robust한 성능을 보이지 못했다. 이는 단순한 domain adaptation이나 소규모 instruction tuning만으로는 충분하지 않으며, 의료 분야에서도 **강한 image-text alignment**, **고품질 caption 또는 instruction data**, **다양한 modality에 대한 폭넓은 사전학습**이 중요하다는 점을 시사한다.

실제 적용 측면에서 이 연구는 당장 더 좋은 의료 LVLM을 제안한 것은 아니지만, 어떤 능력이 부족한지, 어떤 modality에서 취약한지, 왜 현재 접근이 충분하지 않은지를 구체적으로 드러낸다. 향후 연구에서는 보다 넓은 의료 modality를 포괄하는 high-quality image-text pair 구축, 강력한 medical-domain pretraining, 그리고 단순 분류를 넘어서는 deeper clinical reasoning benchmark로의 확장이 중요할 가능성이 크다. OmniMedVQA는 그 출발점으로서 의미가 큰 작업이다.
