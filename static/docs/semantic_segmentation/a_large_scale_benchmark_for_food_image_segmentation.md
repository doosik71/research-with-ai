# A Large-Scale Benchmark for Food Image Segmentation

- **저자**: Xiongwei Wu, Xin Fu, Ying Liu, Ee-Peng Lim, Steven C.H. Hoi, Qianru Sun
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2105.05409

## 1. 논문 개요

이 논문은 음식 이미지에서 개별 재료(ingredient)를 픽셀 단위로 분할하는 문제를 다룬다. 저자들은 기존 음식 이미지 segmentation 연구가 충분히 발전하지 못한 핵심 이유를 두 가지로 본다. 첫째, 재료 수준의 세밀한 라벨과 pixel-wise mask를 갖춘 대규모 데이터셋이 거의 없었다. 둘째, 음식은 같은 재료라도 조리 방식, 배치, 다른 재료와의 혼합 여부에 따라 외형이 매우 크게 변하고, 반대로 서로 다른 재료가 매우 비슷하게 보이기도 해서 segmentation이 본질적으로 어렵다.

이 문제는 건강 관련 응용에서 중요하다. 논문은 음식의 calorie나 nutrient를 추정하려면 단순히 “이 음식은 햄버거다”라고 분류하는 수준을 넘어서, 이미지 안에 어떤 재료가 어디에 얼마나 있는지를 알아야 한다고 강조한다. 따라서 dish-level classification보다 ingredient-level segmentation이 더 직접적으로 실용적이다.

이를 위해 저자들은 두 가지 기여를 제시한다. 하나는 대규모 ingredient-level segmentation 데이터셋인 **FoodSeg103**과 확장판 **FoodSeg154**를 구축한 것이다. 다른 하나는 recipe의 텍스트 정보를 활용해 시각 표현을 더 잘 학습시키는 multi-modal pre-training 방법 **ReLeM**을 제안한 것이다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 “음식 이미지만 봐서는 헷갈리는 재료를 recipe라는 언어 정보로 보완하자”는 것이다. 같은 pineapple이라도 과일 접시에 있을 때와 고기 요리에 들어갔을 때 외형이 달라질 수 있다. 반대로 고기와 함께 조리된 pineapple은 potato처럼 보일 수도 있다. 저자들은 이런 높은 intra-class variation과 inter-class similarity를 줄이기 위해, 이미지 표현과 recipe 표현을 연결하는 학습을 도입한다.

기존 segmentation 접근은 대체로 이미지 자체만 사용한다. 반면 이 논문은 Recipe1M 계열 데이터에 포함된 재료명과 조리 지시문(cooking instructions)을 함께 사용해, 동일한 음식 이미지의 visual embedding과 text embedding이 의미적으로 가까워지도록 학습한다. 논문 표현대로라면, 서로 다른 요리에서 등장한 동일 재료의 시각 표현을 공통 language embedding을 통해 feature space에서 “연결”하려는 것이다.

또 하나의 차별점은 dataset 자체에 있다. 기존 공개 음식 segmentation 데이터셋인 UECFoodPix 계열은 주로 dish 단위 mask를 제공한다. 즉 접시 전체 혹은 음식 덩어리 단위 annotation이다. 반면 이 논문은 tomato, lettuce, beef처럼 **ingredient-wise mask**를 제공한다. 이는 nutrition analysis나 calorie estimation에 훨씬 직접적이다.

## 3. 상세 방법 설명

전체 프레임워크는 크게 두 모듈로 구성된다. 첫 번째는 **Recipe Learning Module (ReLeM)** 이고, 두 번째는 일반적인 **encoder-decoder 기반 image segmentation module**이다. 먼저 ReLeM으로 vision encoder를 pre-train한 뒤, 이 encoder를 segmentation 모델의 encoder 초기값으로 사용한다. decoder는 랜덤 초기화 후 segmentation mask로 학습한다.

### ReLeM의 목적

ReLeM의 목표는 조리 방식 차이 때문에 같은 재료의 시각적 특징이 크게 달라지는 문제를 완화하는 것이다. 논문은 이를 직관적으로 다음과 같이 설명한다. 서로 다른 두 이미지에서 같은 재료가 서로 다른 방식으로 조리되면, visual space에서의 표현 $v_1$, $v_2$는 멀리 떨어질 수 있다. 하지만 각 이미지에 대응되는 recipe 정보 $r_1$, $r_2$를 함께 고려하면, 이 차이를 줄이는 방향으로 feature를 학습할 수 있다는 것이다.

논문은 이를 다음 식으로 표현한다.

$$
|\phi(v_1 \mid r_1) - \phi(v_2 \mid r_2)| < |\phi(v_1) - \phi(v_2)|
$$

여기서 $\phi$는 논문 설명상 segmentation 쪽 decoder와 연결되는 표현 함수로 쓰였지만, 핵심 의미는 “recipe 조건이 들어간 representation이 recipe 없는 representation보다 같은 재료끼리 더 가깝게 정렬되도록 하자”에 있다. 식 자체는 직관을 나타내는 용도에 가깝고, 실제 최적화는 아래의 loss로 수행된다.

### ReLeM의 손실 함수

ReLeM은 두 손실을 사용한다. 하나는 **cosine similarity loss**, 다른 하나는 **semantic loss**이다.

cosine loss는 이미지 표현 $v$와 텍스트 표현 $t$가 같은 recipe에서 왔는지 여부 $y$에 따라 정의된다.

$$
L_{\text{cosine}}((v,t), y)=
\begin{cases}
1-\cosine(v,t), & y=1 \\
\max(0, \cosine(v,t)-\alpha), & y=-1
\end{cases}
$$

여기서 $y=1$이면 같은 recipe 쌍, $y=-1$이면 다른 recipe 쌍이다. margin $\alpha$는 0.1로 설정했다. 같은 recipe인 경우 image-text embedding이 가까워지도록 하고, 다른 recipe인 경우에는 일정 margin 이상 가까워지지 못하게 한다.

semantic loss는 visual representation과 text representation이 각각 semantic class를 잘 맞히도록 하는 cross-entropy loss이다.

$$
L_{\text{semantic}}((v,t), u_v, u_t) = CE(v, u_v) + CE(t, u_t)
$$

여기서 $u_v$, $u_t$는 각각 시각 표현과 텍스트 표현에 대응되는 semantic class다. Recipe1M에는 명시적 dish label이 없기 때문에, 저자들은 recipe title에서 자주 등장하는 dish name을 바탕으로 **2,000개의 semantic label**을 새로 정의했다. 즉, 완전히 supervised semantic class가 원래 주어진 것은 아니고, title 기반으로 만든 라벨을 사용했다는 점이 중요하다.

### 텍스트 전처리와 인코더

각 recipe는 ingredient 리스트와 cooking instructions를 가진다. 저자들은 먼저 불필요한 단어를 제거해 유용한 텍스트만 추출한다. ingredient 텍스트는 bi-directional LSTM과 word2vec 표현을 사용해 인코딩한다. instructions는 길이가 길어 일반 LSTM으로 인코딩하기 어렵다고 보고, 이전 연구를 따라 **skip-instructions** 표현을 사용한다.

텍스트 인코더는 두 가지 버전이 있다.

첫째, **LSTM-based encoder**다. ingredient feature는 bi-directional LSTM, instruction feature는 LSTM으로 인코딩한다.

둘째, **transformer-based encoder**다. 2개의 lightweight transformer를 사용하며, 각 transformer는 2개 layer와 4-head self-attention으로 구성된다.

### 비전 인코더와 Segmenter

ReLeM에서 사용하는 vision encoder는 이후 segmentation 모델 encoder 초기화에 사용된다. 논문은 두 종류의 vision encoder를 쓴다.

- CNN 기반: **ResNet-50**
- Transformer 기반: **ViT-16/B**

Segmentation module은 표준 semantic segmentation 구조를 따른다. 입력 이미지를 encoder가 feature로 변환하고, decoder가 pixel-wise prediction을 수행한다. 논문은 세 가지 대표 구조를 실험한다.

- **Dilation based**: CCNet
- **FPN based**: Sem-FPN
- **Transformer based**: SeTR

Dilation based 모델은 dilated convolution으로 receptive field를 키우면서 해상도를 유지한다. FPN based 모델은 여러 해상도의 feature map을 lateral connection으로 통합한다. Transformer based 모델은 이미지를 patch sequence로 바꿔 attention으로 인코딩한 뒤 segmentation을 수행한다.

최종 segmentation 학습에는 표준 **pixel-wise cross-entropy loss**를 사용한다.

## 4. 실험 및 결과

### 데이터셋 구성

FoodSeg103은 7,118장의 서양 음식 이미지와 103개 ingredient class로 구성된다. 전체 마스크 수는 42,097개다. 이 데이터는 Recipe1M에서 가져온 이미지를 기반으로 만들었다. 이미지당 평균 ingredient label 수는 초록(Abstract) 기준 약 6개다.

FoodSeg154는 여기에 아시아 음식 이미지 2,372장을 추가한 확장판이다. 전체 9,490장, 154개 ingredient class, 59,773개 mask를 포함한다. 다만 논문 본문에 따르면 아시아 음식 세트는 confidentiality 문제로 당시 공개하지 못했다고 명시되어 있다.

훈련/테스트 분할은 FoodSeg103에서 7:3 비율로 이루어졌다.

- 훈련: 4,983장, 29,530 masks
- 테스트: 2,135장, 12,567 masks

아시아 음식 세트는 50:50으로 나누어 cross-domain 평가에 사용했다.

### 데이터셋 annotation 특징과 난이도

annotation 단계에서 작은 영역은 무시했다. 구체적으로 이미지 전체의 5% 미만 면적을 차지하는 tiny region은 annotation에서 제외했다. 이후 refinement 과정에서 오라벨 수정, 5장 미만의 희귀 라벨 삭제, visually similar class 병합을 수행하여 초기 125개 카테고리를 최종 103개로 정리했다.

기존 UECFoodPix 계열과 비교했을 때, 이 데이터셋은 ingredient-level annotation을 제공한다는 점이 가장 중요하다. 또한 DeepLabV3+로 평가했을 때 FoodSeg103의 mIoU는 34.2이고, UECFoodPix는 41.6, UECFoodPixComplete는 55.5였다. 저자들은 이 수치를 “FoodSeg103이 더 어렵다”는 증거로 해석한다. 즉 성능이 낮다는 것은 benchmark가 더 challenging하다는 뜻이다.

### 기본 성능과 ReLeM 효과

FoodSeg103에서의 주요 결과는 다음과 같다.

- CCNet: mIoU 35.5, mAcc 45.3
- ReLeM-CCNet (LSTM): mIoU 36.8, mAcc 47.4
- ReLeM-CCNet (Transformer): mIoU 36.0, mAcc 46.5

- FPN: mIoU 27.8, mAcc 38.2
- ReLeM-FPN (LSTM): mIoU 29.1, mAcc 39.8
- ReLeM-FPN (Transformer): mIoU 28.9, mAcc 39.7

- SeTR: mIoU 41.3, mAcc 52.7
- ReLeM-SeTR (LSTM): mIoU 43.9, mAcc 57.0
- ReLeM-SeTR (Transformer): mIoU 43.2, mAcc 55.7

즉 ReLeM은 세 가지 baseline 모두에 일관되게 성능 향상을 준다. 논문이 요약한 개선 폭은 대략 CCNet에서 1.3%p, FPN에서 1.3%p, SeTR에서 2.6%p 수준의 mIoU 향상이다. 또한 모든 설정에서 transformer-based text encoder보다 **LSTM-based ReLeM**이 더 낫게 나왔다.

이 결과는 두 가지를 시사한다. 첫째, recipe 정보는 실제로 segmentation encoder pre-training에 도움이 된다. 둘째, 당시 설정에서는 복잡한 transformer text encoder보다 LSTM 기반 텍스트 인코딩이 더 안정적이거나 task에 더 잘 맞았던 것으로 보인다. 논문은 이 이유를 깊게 분석하지는 않는다.

### 일반 segmentation benchmark와의 비교

저자들은 FoodSeg103의 난이도를 보여주기 위해 Cityscapes와 비교했다. 결과는 다음과 같다.

- CCNet: Cityscapes 79.0, FoodSeg103 35.0
- Sem-FPN: Cityscapes 74.5, FoodSeg103 27.8
- SeTR: Cityscapes 77.9, FoodSeg103 41.3

모든 모델이 Cityscapes에서는 높은 성능을 내지만 FoodSeg103에서는 큰 폭으로 하락한다. 논문은 이를 통해 음식 segmentation이 일반 객체 segmentation보다 훨씬 어렵다고 주장한다. 이 비교는 절대적 공정성보다 task difficulty를 직관적으로 전달하는 데 목적이 있다.

### Cross-domain 평가

FoodSeg154의 아시아 음식 세트를 이용해 domain adaptation 성능도 평가했다. FoodSeg103으로 학습한 모델을 아시아 음식 세트에 적용하고, 일부는 fine-tuning도 수행했다.

주요 결과는 다음과 같다.

- CCNet: mIoU 28.6
- ReLeM-CCNet: mIoU 29.2
- CCNet-Finetune: mIoU 41.3
- ReLeM-CCNet-Finetune: mIoU 47.1

- FPN: mIoU 21.9
- ReLeM-FPN: mIoU 22.9
- FPN-Finetune: mIoU 27.1
- ReLeM-FPN-Finetune: mIoU 30.8

즉 domain shift가 있는 상황에서도 ReLeM은 baseline보다 낫고, fine-tuning 후에도 그 이점이 유지된다. 특히 ReLeM-CCNet-Finetune은 mIoU 47.1로 가장 강한 결과를 보인다. 논문은 이를 recipe 기반 pre-training이 domain adaptation에도 도움이 된다고 해석한다.

### 추가 transformer 분석

부록에서는 더 다양한 transformer encoder도 시험했다. PVT-S, ViT-16/B, ViT-16/L, Swin-S, Swin-B 등을 비교했는데, 몇 가지 흥미로운 결과가 나온다.

- ReLeM-PVT-S: 31.3에서 32.0으로 개선
- ReLeM-ViT-16/B Naive: 41.3에서 43.9로 개선
- 하지만 ViT-16/B MLA에서는 baseline 45.1이 ReLeM 43.3보다 더 좋음

저자들은 MLA decoder가 여러 transformer layer의 feature를 통합하는 반면, ReLeM은 마지막 feature map만 활용하기 때문에 multi-level representation 학습과 잘 맞지 않았을 수 있다고 해석한다. 또 backbone이 더 크다고 항상 좋아지지 않았고, 다른 vision task에서 강한 Swin이 여기서는 ViT보다 약한 결과를 보였다. 이는 food image segmentation이 단순히 backbone 크기만 키운다고 해결되지 않는 어려운 문제임을 시사한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 데이터셋 기여다. FoodSeg103/FoodSeg154는 ingredient-level segmentation을 위한 드문 대규모 benchmark이며, 실제 응용 가치가 높다. 특히 dish-level이 아니라 ingredient-level annotation을 제공한다는 점은 nutrition analysis, calorie estimation, health monitoring에 직접 연결된다. annotation과 refinement에 1년 가까운 시간이 들었다는 설명도 데이터 품질 확보에 공을 들였음을 보여준다.

두 번째 강점은 ReLeM의 문제 설정이 설득력 있다는 점이다. 음식은 시각적으로만 보면 class ambiguity가 심하고, recipe는 이런 ambiguity를 풀어주는 좋은 보조 정보다. 논문은 recipe를 segmentation 학습에 직접 활용하는 방향을 제시했고, 세 종류의 대표 segmentation framework에 꽂아 성능 향상을 보였기 때문에 방법의 범용성도 어느 정도 입증했다.

세 번째 강점은 실험 설계가 비교적 충실하다는 점이다. 저자들은 in-domain 성능뿐 아니라 cross-domain 평가, Cityscapes와의 난이도 비교, qualitative visualization, transformer 계열 추가 실험까지 제시했다. 따라서 단순히 “새 데이터셋을 만들었다”에 그치지 않고, benchmark로서의 성격을 분명히 보여준다.

반면 한계도 분명하다. 첫째, FoodSeg154의 아시아 음식 세트는 본문 기준 공개되지 않았다. 따라서 “154-class 확장 benchmark”의 재현성과 활용성은 FoodSeg103보다 제한적이다. 둘째, class distribution이 매우 long-tailed인데, 이 문제를 정면으로 해결하는 별도 학습 전략은 제시되지 않았다. 논문은 hard negative mining도 사용하지 않았다고 밝힌다. 셋째, tiny region을 5% 미만이면 무시했기 때문에 실제 응용에서 중요한 작은 재료들은 충분히 반영되지 않을 수 있다.

또한 ReLeM의 설명에는 다소 모호한 부분이 있다. 예를 들어 식 (1)의 $\phi$ 설명은 본문에서 segmenter decoder와 연결되는데, 실제 representation learning 과정과의 관계가 엄밀하게 전개되지는 않는다. 그리고 semantic label 2,000개를 recipe title 빈도 기반으로 만든 방식은 실용적이지만, label noise가 어느 정도 섞였을 가능성이 있다. 논문은 이 노이즈의 영향을 별도로 분석하지 않는다.

비판적으로 보면, ReLeM의 성능 향상은 일관되지만 절대적으로 매우 큰 폭은 아니다. 가장 큰 향상은 SeTR에서 2.6%p 수준이며, 일부 transformer-decoder 조합에서는 오히려 baseline보다 떨어진다. 따라서 ReLeM은 강력한 일반 해법이라기보다, 특정 구조에서는 잘 작동하는 유용한 pre-training 기법으로 보는 편이 더 정확하다.

## 6. 결론

이 논문은 음식 이미지 segmentation 분야에서 중요한 기반 작업이다. 저자들은 ingredient-level mask를 갖춘 대규모 benchmark인 FoodSeg103과 FoodSeg154를 구축했고, recipe 텍스트를 활용한 multi-modal pre-training 기법 ReLeM을 제안했다. 실험 결과 ReLeM은 CCNet, FPN, SeTR 같은 대표 segmentation 모델에 대체로 일관된 성능 향상을 제공했으며, 특히 음식 segmentation이 일반 semantic segmentation보다 훨씬 어려운 문제임도 명확히 보여주었다.

실제 적용 측면에서는, 이 연구가 nutrition estimation, calorie tracking, smart food logging 같은 health-related application의 기반 기술로 이어질 가능성이 크다. 연구 측면에서는 long-tailed ingredient recognition, cross-domain food understanding, multi-level feature와 결합된 더 정교한 multi-modal pre-training, food-aware decoder 설계 같은 후속 과제를 자연스럽게 제시한다. 즉 이 논문은 최종 해결책이라기보다, 음식 재료 수준 장면 이해를 위한 본격적인 benchmark와 출발점을 제공한 작업으로 보는 것이 적절하다.
