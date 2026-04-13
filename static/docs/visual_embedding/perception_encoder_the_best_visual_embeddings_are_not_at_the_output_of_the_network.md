# Perception Encoder: The best visual embeddings are not at the output of the network

- **저자**: Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, Junke Wang, Marco Monteiro, Hu Xu, Shiyu Dong, Nikhila Ravi, Daniel Li, Piotr Dollár, Christoph Feichtenhofer
- **발표연도**: 2025
- **arXiv**: https://arxiv.org/abs/2504.13181

## 1. 논문 개요

이 논문은 이미지와 비디오를 모두 다룰 수 있는 범용 vision encoder인 **Perception Encoder (PE)**를 제안한다. 논문의 핵심 문제의식은 분명하다. 기존 비전 사전학습은 목적 함수가 작업별로 나뉘어 있었다. 예를 들어 contrastive vision-language learning은 zero-shot classification과 retrieval에 강하고, captioning 기반 학습은 multimodal language model에 유리하며, spatial self-supervised learning은 detection이나 depth 같은 dense prediction에 강하다고 여겨져 왔다. 그래서 좋은 범용 encoder를 만들려면 여러 목적을 복잡하게 섞어야 한다는 인식이 강했다.

이 논문은 그 통념에 도전한다. 저자들은 **잘 설계된 대규모 contrastive vision-language training만으로도** 분류, 검색, MLLM, detection, tracking, depth estimation까지 폭넓은 작업에 유용한 표현을 학습할 수 있다고 주장한다. 다만 중요한 단서가 하나 있다. 가장 좋은 표현이 네트워크의 마지막 출력이 아니라 **중간층(intermediate layers)** 안에 숨어 있다는 점이다. 따라서 좋은 범용 표현을 얻는 핵심은 더 복잡한 사전학습 목적을 추가하는 것이 아니라, 이미 내부에 존재하는 표현을 **끝단으로 정렬(alignment)** 하는 것이라고 본다.

이 문제는 중요하다. 만약 단일하고 단순한 pretraining recipe만으로 다양한 downstream task를 커버할 수 있다면, 대규모 비전 모델 개발의 복잡도와 확장 비용을 크게 줄일 수 있기 때문이다. 이 논문은 그 가능성을 이미지와 비디오, 언어 이해, spatial understanding 전반에서 실험적으로 보여주려 한다.

## 2. 핵심 아이디어

논문의 중심 직관은 다음 한 문장으로 요약된다. **가장 좋은 visual embedding은 네트워크 출력에 있지 않을 수 있다.** 저자들은 강력한 contrastive pretraining을 수행한 뒤 layer-by-layer로 분석해 보니, OCR, VQA, grounding, detection, depth, tracking에 유용한 특징들이 각기 다른 중간층에 형성되어 있음을 발견했다. 즉, 기존에는 contrastive 모델이 이런 작업에 약하다고 생각했지만, 실제로는 “못 배운 것”이 아니라 “출력으로 잘 드러나지 않는 것”일 수 있다는 해석이다.

기존 접근과의 차별점은 두 가지다. 첫째, 여러 pretraining objective를 조합하지 않고 **contrastive learning 하나**를 끝까지 밀어붙인다. 둘째, downstream 성능 향상을 위해 사전학습 자체를 복잡하게 바꾸기보다, 이미 학습된 내부 표현을 task에 맞게 끌어올리는 **alignment tuning**을 사용한다. 언어 작업에는 **language alignment**, spatial 작업에는 **spatial alignment**를 적용한다.

이 아이디어는 단순한 경험적 관찰에 그치지 않는다. 논문은 intermediate layer가 왜 중요한지 분석하고, 특히 큰 ViT에서 후반부에 형성되는 **global tokens**가 spatial coherence를 무너뜨리는 일종의 “decoder 구간”을 만든다고 해석한다. 그래서 semantic 정보는 후반부에 강하지만, tracking 같은 순수 correspondence 기반 task는 오히려 그 직전 층에서 최고 성능을 낸다.

## 3. 상세 방법 설명

전체 구조는 크게 세 부분으로 나뉜다. 첫째는 **PE core**, 둘째는 **PE lang**, 셋째는 **PE spatial**이다.

### PE core: 강한 contrastive image-video encoder

PE core는 먼저 이미지-텍스트 contrastive pretraining으로 시작하고, 그 다음 비디오-텍스트 finetuning을 수행해 이미지와 비디오를 모두 다루는 unified encoder를 만든다. 논문은 이미지와 비디오를 처음부터 함께 학습하지 않고 두 단계로 나눈다. 이유는 이미지-텍스트 데이터는 풍부하고 학습 효율도 높지만, 비디오-텍스트 데이터는 희소하고 비용이 크기 때문이다.

#### Robust image pretraining recipe

저자들은 vanilla CLIP 위에 여러 학습 기법을 누적 적용한다.

- progressive resolution
- 더 큰 batch size
- AdamW 대신 LAMB optimizer
- 마지막 stage에서 더 높은 해상도
- 2D RoPE
- attention pooling
- 강한 data augmentation
- mask regularization

이 조합의 목적은 단순히 ImageNet 점수를 높이는 것이 아니라, **robustness와 generality**를 높이는 것이다. 특히 progressive resolution은 해상도가 계속 바뀌므로 모델이 특정 위치의 “global token” 구조에 과적합하기 어렵게 만들어 spatially useful feature 학습에 도움을 준다고 해석한다.

#### Video data engine

비디오 학습에서 가장 중요한 부분은 **잘 정렬된(aligned) 비디오 캡션 데이터 생성**이다. 논문은 이를 위해 3단계 video data engine을 만든다.

1. 초기 PLM(Perception Language Model) 기반 video captioner를 만든다.
2. 265K 비디오에 대해 caption을 생성한 뒤 인간 annotator가 이를 수정한 데이터로 captioner를 강화한다.
3. 최종적으로 비디오 metadata(title, description), video caption, frame captions를 text-only LLM에 넣어 **짧고 밀도 높은 aligned caption**으로 요약한다.

비디오 인코딩은 의외로 단순하다. 각 비디오에서 $N=8$개의 프레임을 균일 샘플링하고, 각 프레임 임베딩을 뽑은 뒤 평균내어 비디오 임베딩으로 사용한다. 즉, 복잡한 temporal attention 없이 **frame embedding average pooling**만 사용한다. 논문은 이 단순한 방식이 매우 효과적이라고 보고한다.

#### 작은 모델 distillation

가장 큰 G-scale 모델을 teacher로 두고, 작은 B/L 모델은 image-text similarity distribution과 text-image similarity distribution을 teacher에 맞추도록 **KL-divergence distillation**을 수행한다. 특히 teacher의 softmax temperature를 student보다 더 작게 두는 것이 효과적이었다고 보고한다.

### Intermediate layer 분석과 alignment 문제

논문 3장의 핵심은 PE core의 마지막 층이 아니라 중간층이 language task와 spatial task에서 더 좋다는 사실이다. 저자들은 AIMv2-3B, DINOv2-g와 비교했을 때, PE core의 중간층이 language task에서는 AIMv2에, spatial task에서는 DINOv2에 근접하거나 일부는 능가하는 성능을 보인다고 주장한다. 그러나 이런 성능은 마지막 층에 가면 크게 떨어진다. 저자들은 이를 **alignment problem**이라고 부른다. 즉, 좋은 일반 표현은 학습되었지만 출력으로 정렬되지 않은 상태라는 것이다.

### PE lang: language alignment

PE lang는 PE core를 decoder-only LLM에 맞게 정렬한 버전이다. 방법은 PLM의 midtraining과 유사하다. 먼저 1M image-text 데이터로 projector만 학습하는 warmup을 수행하고, 이후 약 70M 샘플에 대해 autoregressive next-token prediction loss로 vision encoder, projector, LLM을 함께 학습한다.

중요한 설계 선택은 다음과 같다.

- LLM은 Llama 3.2 3B
- 2-layer MLP projector 사용
- PE core G의 **47번째 층** 출력을 projector 입력으로 사용
- alignment 중 LayerScale과 DropPath regularization 적용
- 마지막 3개 층은 버리고 layer 47을 사실상 출력단처럼 사용

논문은 layer 47이 마지막 층보다 language task에 더 좋았다고 보고한다. alignment 이후에는 OCR Q&A, captioning, VQA, grounding 모두에서 **마지막 층이 최적층**이 되며, 즉 중간층의 좋은 특성이 출력단으로 끌어올려졌다고 해석한다.

### PE spatial: spatial alignment

spatial alignment는 더 흥미롭다. spatial task에는 detection, segmentation, depth, tracking처럼 성격이 다른 문제가 섞여 있기 때문이다. 논문은 tracking에 좋은 층과 detection/depth에 좋은 층이 다르다는 점을 분석하고, 그 원인을 후반부에서 생기는 **global token 기반 decoder 구간**으로 설명한다.

이를 해결하기 위해 두 종류의 teacher를 동시에 사용한다.

첫째는 **PE core 자체의 frozen layer 41**이다. 이것은 semantic 정보를 유지하기 위한 teacher다. 학생 모델의 마지막 층 $S_{50}$와 teacher의 layer 41 표현 $T_{41}$ 사이의 cosine similarity 기반 정렬 loss를 사용한다. 논문에 따르면 손실은 다음과 같이 쓴다.

$$
L_{\text{core}} = \frac{1}{n_{\text{tok}}} \sum \left( \frac{(S_{50})(T_{41})^T}{\|S_{50}\|\cdot\|T_{41}\|} \right)
$$

둘째는 **SAM 2.1의 mask logits**이다. 여기서 저자들의 아이디어가 독특하다. SAM의 raw feature도 global token 문제를 갖고 있으므로 feature 자체를 teacher로 쓰지 않고, 32×32 grid의 point query에 대한 SAM mask logits를 모아서 $H \times W \times 1024$ 형태의 feature처럼 사용한다. 그리고 학생과 teacher 각각에 대해 token 간 pairwise cosine similarity matrix를 만든 뒤, 이 spatial correspondence를 MSE로 맞춘다.

$$
L_{\text{loc}} = \frac{1}{n_{\text{tok}}^2}\sum \left( \frac{(S_{50})(S_{50})^T}{\|S_{50}\|^2} - \frac{(T_{\text{SAM}})(T_{\text{SAM}})^T}{\|T_{\text{SAM}}\|^2} \right)^2
$$

최종 spatial alignment loss는

$$
L_{\text{spatial}} = L_{\text{core}} + L_{\text{loc}}
$$

이다.

해석하면, $L_{\text{core}}$는 semantic richness를 보존하고, $L_{\text{loc}}$는 local correspondence와 spatial coherence를 강제한다. 논문은 둘 중 하나만 쓰면 각각 semantic 또는 locality 한쪽이 부족하고, 둘을 함께 써야 가장 균형 잡힌 spatial feature가 나온다고 주장한다.

## 4. 실험 및 결과

이 논문은 실험 범위가 매우 넓다. 이미지 zero-shot classification/retrieval, 비디오 zero-shot classification/retrieval, MLLM 기반 문서/차트/OCR/VQA/비디오 이해, detection, segmentation, tracking, depth estimation까지 다룬다.

### 이미지와 비디오 zero-shot 성능

PE core G는 이미지 zero-shot에서 매우 강한 결과를 보인다. 논문에 따르면 평균 zero-shot ImageNet robustness는 **86.6**, zero-shot video classification Kinetics-400은 **76.9**다. 이미지 쪽에서는 SigLIP2-g-opt와 경쟁하며, 비디오 쪽에서는 InternVideo2 같은 native video model과 비교해도 classification에서 더 강하거나 비슷한 수준을 보인다.

중요한 점은 비디오 finetuning이 비디오 성능만 올린 것이 아니라 **이미지 성능도 함께 올렸다**는 것이다. 저자들은 22M recaptioned videos로 학습한 뒤 general image classification, fine-grained classification, retrieval 모두가 향상되었다고 보고한다. 이는 잘 정렬된 video-text 데이터가 encoder의 전반적 표현 품질을 높였다는 주장으로 이어진다.

### PE Video Dataset (PVD)

논문은 자체 비디오 데이터셋인 **PVD**도 공개한다. 약 1M 비디오와 120K human-refined caption을 포함하며, 총 길이는 4625시간이다. 이 중 15K는 video retrieval benchmark로 사용된다. 데이터 필터링은 motion feature, DINOv2/SigLIP feature, MLLM 기반 yes/no 질의 특징, random forest 품질 점수, k-means 기반 중복 억제 과정을 거친다. 이 부분은 데이터 구축 기여로도 의미가 있다.

### Frozen feature 분석

가장 인상적인 실험은 layerwise frozen feature 분석이다. PE core G의 중간층은 detection, depth, OCR Q&A, visual Q&A, grounding, tracking 등에서 매우 높은 점수를 보인다. 그러나 마지막 층 성능은 어떤 작업에서는 급격히 나빠진다. 논문은 이것이 prior work가 “CLIP은 downstream task에 잘 scale되지 않는다”고 본 이유일 수 있다고 해석한다. 실제로 best layer 기준으로는 scale이 잘 되지만, last layer만 보면 정체되기 때문이다.

### MLLM 성능

PE lang는 Llama 3.1 8B와 Qwen 2.5 7B 둘 다와 결합했을 때 강한 성능을 보인다. 예를 들어 Llama 3.1 8B 기준으로 PE lang G는 OCR/Chart/Doc QA 평균 **72.4**, grounding 평균 **71.3**, video 평균 **56.0**을 기록한다. 논문은 이를 기존 vision encoder 대비 평균 **+3.5 포인트** 정도 우세하다고 요약한다.

시스템 수준 비교에서도 PE lang G를 사용한 **PLM-8B**는 DocVQA **94.6**, InfographicVQA **80.9**, PerceptionTest **82.7** 등 강한 성능을 낸다. 이 결과는 vision encoder 자체의 품질이 전체 MLLM 성능에 크게 기여한다는 논문의 주장과 맞닿아 있다.

### Spatial task 성능

PE spatial G는 dense prediction에서 매우 강하다. frozen feature 기준으로 DAVIS tracking J&F는 **61.5**, ADE20K semantic segmentation mIoU는 **49.3**, NYU depth RMSE는 **0.262**를 기록한다. detection full finetuning에서도 COCO box AP **57.8**, LVIS box AP **54.2**로 강하다.

논문이 특히 강조하는 결과는 system-level detection이다. 단순한 DETR-style decoder인 DETA와 결합해 COCO에서 **66.0 box mAP**를 달성했다고 보고한다. 이는 당시 absolute SOTA라고 주장한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 재정의**다. “contrastive learning은 classification용, captioning은 language용, self-supervised는 spatial용”이라는 고정관념을 깨고, 사실 좋은 일반 표현은 이미 contrastive model 내부에 존재할 수 있다고 보여준다. 이 관찰은 단순한 직관이 아니라 대규모 layerwise 실험으로 뒷받침된다.

둘째 강점은 **방법의 단순성과 확장성**이다. pretraining objective를 복잡하게 섞지 않고 contrastive learning 하나를 중심에 두며, downstream 확장은 alignment tuning으로 처리한다. 이는 대규모 foundation model 개발에서 매우 실용적인 방향이다.

셋째 강점은 **실험 범위의 넓이**다. 이미지, 비디오, 문서, OCR, grounding, tracking, detection, depth까지 아우르며, PE core, PE lang, PE spatial이 서로 다른 작업군에서 일관된 서사를 만든다. PVD 공개 역시 재현성과 후속 연구 측면에서 가치가 있다.

한계도 있다. 우선 이 논문의 핵심 주장은 매우 강력하지만, 그만큼 **계산 자원 의존성**이 크다. G scale은 vision tower만 1.88B parameter이며, 사전학습 데이터도 5.4B image-text pair 수준이다. 따라서 “단순한 방법”이라는 개념이 알고리즘 측면에서는 맞더라도, 실제 재현 비용은 매우 높다.

또한 contrastive loss 자체의 수식과 일부 세부 선택은 본문 발췌에서 완전하게 드러나지 않는다. 예를 들어 core 학습의 기본 contrastive objective는 CLIP 스타일이라고 설명되지만, 본문에 명시적인 식은 없다. 따라서 이 보고서도 그 부분은 추측하지 않는다.

비판적으로 보면, 논문의 주장은 “contrastive learning alone”이라고 말하지만, 실제 시스템은 robust recipe, 비디오 recaptioning engine, LLM summarization, language alignment, SAM-based spatial alignment 등 상당한 후처리와 정렬 단계를 포함한다. 즉, **사전학습 목적 함수는 단순하지만 전체 시스템은 결코 단순하지 않다**. 다만 저자들의 진짜 요지는 “여러 pretraining objective를 섞지 않아도 된다”는 데 있으므로, 이 점은 주장 자체를 무너뜨리기보다는 해석 시 주의할 부분이다.

## 6. 결론

이 논문은 Perception Encoder라는 이름 아래 세 가지 모델군을 제시한다. **PE core**는 강력한 image-video contrastive encoder이고, **PE lang**는 이를 MLLM에 적합하게 정렬한 버전이며, **PE spatial**은 dense prediction에 맞게 spatially aligned한 버전이다. 핵심 기여는 좋은 visual representation이 마지막 출력이 아니라 중간층에 숨어 있을 수 있다는 점을 체계적으로 보이고, 이를 alignment tuning으로 끌어올리면 여러 비전 작업에서 최고 수준 성능을 낼 수 있음을 입증한 것이다.

향후 연구 측면에서도 의미가 크다. 이 논문은 비전 foundation model 연구에서 “더 많은 pretraining objective를 추가할 것인가”보다 “이미 학습된 표현을 어떻게 읽어내고 정렬할 것인가”가 더 중요한 질문일 수 있음을 보여준다. 실제 적용 관점에서도, 하나의 강한 core encoder를 만들고 이를 언어 또는 spatial task에 맞게 정렬하는 방식은 범용 perception stack 설계에 매우 유력한 방향으로 보인다.
