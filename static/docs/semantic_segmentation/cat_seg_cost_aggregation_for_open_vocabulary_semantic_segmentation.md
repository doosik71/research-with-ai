# CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation

- **저자**: Seokju Cho, Heeseong Shin, Sunghwan Hong, Anurag Arnab, Paul Hongsuck Seo, Seungryong Kim
- **발표연도**: 2024
- **arXiv**: https://arxiv.org/abs/2303.11797

## 1. 논문 개요

이 논문은 open-vocabulary semantic segmentation 문제를 다룬다. 이 문제는 이미지의 각 픽셀에 대해, 고정된 소수 클래스가 아니라 텍스트로 주어지는 매우 넓은 범위의 class description 중 하나를 할당하는 과제다. 핵심 어려움은 CLIP 같은 vision-language foundation model이 본래 image-level supervision으로 학습되었기 때문에, pixel-level dense prediction으로 바로 옮기면 구조적 불일치가 생긴다는 점이다.

기존 연구들은 이 간극을 줄이기 위해 region proposal이나 class-agnostic mask generator를 활용해 region-to-text matching으로 문제를 우회하는 경우가 많았다. 하지만 이런 접근은 추가 모듈에 크게 의존하고, 결국 CLIP의 표현을 segmentation에 맞게 충분히 적응시키는 데 한계가 있다. 특히 CLIP encoder를 직접 fine-tuning하면 seen class에는 과적합되고 unseen class에 대한 정렬(alignment)이 무너지는 문제가 반복적으로 보고되었다.

이 논문은 바로 그 지점을 정면으로 겨냥한다. 저자들은 feature 자체를 decoder로 변형해 쓰는 대신, 이미지 임베딩과 텍스트 임베딩 사이의 cosine similarity, 즉 cost volume을 직접 만들고 이것을 aggregation하는 방식이 CLIP을 semantic segmentation에 더 안정적으로 적응시킨다고 주장한다. 이 아이디어를 바탕으로 제안한 모델이 CAT-Seg이며, 표준 benchmark와 multi-domain benchmark에서 모두 강한 성능을 보였다고 보고한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 분명하다. 기존 방식처럼 CLIP image feature를 별도의 segmentation decoder가 강하게 변형하도록 두면, CLIP이 원래 가지고 있던 image-text joint embedding space가 손상되기 쉽다. 그러면 seen class에는 맞추더라도 unseen class generalization이 급격히 떨어진다. 반면 이 논문은 이미지 feature와 텍스트 feature 사이의 유사도 자체를 cost로 보고, 그 cost를 refinement하는 방향으로 segmentation을 풀면 CLIP의 정렬 구조를 보존하면서 dense prediction으로 옮길 수 있다고 본다.

저자들은 이를 visual correspondence 문헌의 cost aggregation 개념과 연결한다. 원래 cost aggregation은 두 이미지 사이의 matching cost를 다루는 기법인데, 이 논문은 그것을 image-to-text cost volume에 맞게 재해석한다. 다만 여기서는 입력이 단일 modality가 아니라 image와 text의 multi-modal cost volume이므로, 기존 stereo나 correspondence 방식 그대로는 부족하다고 본다. 그래서 spatial aggregation과 class aggregation을 분리해 설계한다.

이 논문의 차별점은 세 가지로 요약할 수 있다. 첫째, feature aggregation이 아니라 cost aggregation을 전면에 둔다. 둘째, image-text cost volume의 구조를 반영해 spatial 축과 class 축을 나눠 reasoning한다. 셋째, 이 구조 덕분에 이전 연구들이 실패하거나 회피했던 CLIP fine-tuning을 seen과 unseen을 모두 살리는 방향으로 수행할 수 있다고 보인다.

## 3. 상세 방법 설명

전체 파이프라인은 다음과 같다. 입력 이미지 $I$와 클래스 집합 $C = \{T^{(n)}\}$가 주어지면, CLIP image encoder와 text encoder를 통해 dense image embedding과 text embedding을 얻는다. 그 다음 각 spatial position과 각 class 사이 cosine similarity를 계산하여 cost volume을 만들고, 이것을 spatial aggregation과 class aggregation으로 번갈아 정제한 뒤, upsampling decoder를 통해 최종 segmentation mask를 예측한다.

먼저 dense image embedding은 $D_V = \Phi_V(I) \in \mathbb{R}^{(H \times W) \times d}$, text embedding은 $D_L = \Phi_L(T) \in \mathbb{R}^{N_C \times d}$로 둔다. 여기서 $N_C$는 클래스 수다. 이미지 encoder는 CLIP의 마지막 attention layer를 수정하여 pooling 효과를 제거하고 dense token을 유지한다. 이후 각 위치 $i$와 클래스 $n$에 대해 cost volume $C$를 cosine similarity로 정의한다.

$$
C(i,n) = \frac{D_V(i) \cdot D_L(n)}{\|D_V(i)\| \|D_L(n)\|}
$$

이 식의 의미는 간단하다. 각 픽셀이 각 텍스트 클래스와 얼마나 잘 맞는지를 직접 점수화한 것이다. 논문은 이 raw cost volume이 이미 “거친 semantic mask”처럼 동작한다고 해석한다. 즉, refinement 이전에도 어느 정도 class-grounded heatmap 역할을 한다는 것이다.

그 다음 이 cost volume을 바로 쓰지 않고, 각 class slice를 독립적으로 1개 convolution layer에 통과시켜 초기 cost embedding $F \in \mathbb{R}^{(H \times W) \times N_C \times d_F}$를 만든다. 논문에서는 $d_F = 128$을 사용한다.

### Spatial Cost Aggregation

spatial aggregation은 이미지의 공간적 구조를 반영하는 단계다. 핵심은 각 클래스별로 따로 spatial refinement를 한다는 점이다. 즉, “chair에 대한 cost map”, “sofa에 대한 cost map”을 각각 독립된 spatial map으로 보고 다듬는다. 수식은 다음과 같다.

$$
F'(:,n) = T_{sa}(F(:,n))
$$

여기서 $T_{sa}$는 두 개의 연속된 Swin Transformer block이다. 첫 블록은 local window self-attention, 두 번째는 shifted window self-attention을 사용한다. 저자들이 CNN보다 Transformer, 그중에서도 Swin Transformer를 택한 이유는 전역 또는 semi-global receptive field를 효율적으로 얻기 위해서다.

이 단계의 직관은 segmentation mask refinement와 유사하다. 예를 들어 “sofa”에 대한 raw cost map이 배경에 잡음을 포함하더라도, 공간적 문맥을 이용해 실제 sofa 영역은 강화하고 배경 응답은 억제한다.

### Class Cost Aggregation

그 다음은 class aggregation이다. spatial aggregation이 이미지 내부 구조를 다뤘다면, class aggregation은 서로 다른 클래스들 사이 관계를 모델링한다. 예를 들어 어떤 픽셀에서 “chair”와 “sofa”가 동시에 높게 나오더라도, 두 클래스의 관계와 문맥을 보면 더 일관된 결정을 내릴 수 있다. 수식은 다음과 같다.

$$
F''(i,:) = T_{ca}(F'(i,:))
$$

여기서 $T_{ca}$는 position embedding이 없는 transformer block이다. 저자들이 position embedding을 제거한 이유는 open-vocabulary setting에서 클래스 수가 inference마다 달라질 수 있고, 클래스 순서에도 invariant해야 하기 때문이다. 이 요구사항은 class token sequence에 절대적 위치 정보를 넣지 않는 설계와 잘 맞는다.

또한 class aggregation에는 linear transformer를 사용한다. spatial 구조를 다루는 것이 아니므로 일반 self-attention보다 더 가볍게 클래스 축 관계를 모델링할 수 있기 때문이다.

### Spatial-Class Aggregation의 교차 적용

논문은 spatial aggregation과 class aggregation을 한 번만 쓰는 것이 아니라 번갈아 적용한다고 설명한다. 이는 image-text cost volume이 본질적으로 2차원 spatial 구조와 class 집합 구조를 동시에 가지기 때문이다. spatial만 쓰면 클래스 간 구분이 약해지고, class만 쓰면 공간적 매끄러움이 부족할 수 있다. 따라서 둘을 교차시키는 것이 multi-modal cost volume에 더 적합하다는 것이 저자들의 설계 철학이다.

### Upsampling Decoder

aggregation된 저해상도 cost volume만으로는 세밀한 경계 예측이 어렵다. 이를 보완하기 위해 FPN과 유사한 upsampling decoder를 둔다. bilinear upsampling으로 해상도를 올리고, CLIP image encoder 중간 층에서 뽑은 higher-resolution feature map과 concatenate한 뒤, $3 \times 3$ convolution으로 결합한다. 이 과정을 $N_U$번 반복하고 최종 prediction head로 segmentation 결과를 만든다.

중요한 점은 별도 무거운 backbone을 추가하지 않는다는 것이다. CLIP ViT 중간층 특징을 재활용해 U-Net 비슷한 구조로 세부 정보를 복원한다. supplementary에 따르면 ViT-B/16에서는 8번째 층과 4번째 층 출력을, ViT-L/14에서는 16번째 층과 8번째 층 출력을 decoder에 사용한다.

### Embedding Guidance

저자들은 cost aggregation만으로 끝내지 않고, 원래 CLIP embedding 자체를 aggregation의 guide로 활용한다. 직관은 “비슷한 시각적 또는 의미적 토큰은 비슷한 matching cost를 가질 가능성이 높다”는 것이다. 이를 식으로 쓰면 다음과 같다.

$$
F'(:,n) = T_{sa}([F(:,n); P_V(D_V)])
$$

$$
F''(i,:) = T_{ca}([F'(i,:); P_L(D_L)])
$$

여기서 $[\cdot]$는 concatenation, $P_V$와 $P_L$는 linear projection이다. 저자들은 embedding guidance를 query와 key에만 주는 것이 충분했다고 보고한다. 즉, value까지 모두 교란하기보다 attention이 어떤 토큰끼리 관계를 맺을지 결정하는 쪽을 돕는 설계다.

### Fine-tuning 전략

논문에서 중요한 또 하나의 축은 CLIP fine-tuning 방식이다. 저자들은 full fine-tuning뿐 아니라 prompt tuning, attention만 fine-tuning, QK만 fine-tuning, KV만 fine-tuning, QV만 fine-tuning 등을 비교한다. 결론적으로 image encoder와 text encoder 양쪽에서 query와 value projection만 fine-tuning하는 방식이 성능과 효율 측면에서 가장 좋았다고 보고한다.

이 결과는 흥미롭다. 단순히 “더 많이 학습하면 더 좋다”가 아니라, CLIP의 표현 공간을 지나치게 훼손하지 않으면서 segmentation에 필요한 적응만 일으키는 것이 중요하다는 해석이 가능하다. 논문은 이 점을 cost aggregation의 안정성과 연결한다.

## 4. 실험 및 결과

학습 데이터는 COCO-Stuff이며, 118k training image와 171 category를 사용한다. 평가 지표는 전부 mIoU다. 표준 benchmark로는 ADE20K의 A-150과 A-847, PASCAL-Context의 PC-59와 PC-459, PASCAL VOC의 PAS-20 및 PAS-20b를 사용한다. 추가로 multi-domain 일반화 능력을 보기 위해 MESS benchmark도 사용했다. MESS는 의료, 공학, 농업, 해양, 야간 주행 등 22개 데이터셋을 포괄하는 매우 강한 stress test다.

구현 세부사항을 보면, loss는 per-pixel binary cross-entropy를 사용한다. optimizer는 AdamW이고, learning rate는 모델 본체에 $2 \times 10^{-4}$, CLIP에는 $2 \times 10^{-6}$를 사용한다. batch size는 4, 총 80k iteration 학습이다. 훈련 해상도는 $384 \times 384$이며 cost volume 해상도는 $H=W=24$다.

### 표준 benchmark 결과

CAT-Seg는 ViT-B/16 기반에서도 기존 방법들을 일관되게 앞선다. 예를 들어 A-847에서 12.0, PC-459에서 19.0, A-150에서 31.8, PC-59에서 57.5, PAS-20에서 94.6, PAS-20b에서 77.3을 기록했다. 특히 unseen class가 많은 A-847, PC-459, PAS-20b에서 향상이 크다.

ViT-L/14 기반에서는 더 강하다. A-847에서 16.0, PC-459에서 23.8, A-150에서 37.9, PC-59에서 63.3, PAS-20에서 97.0, PAS-20b에서 82.5를 기록했다. 논문은 특히 A-847에서 이전 SOTA 대비 +3.6 mIoU, PC-459에서 +8.1 mIoU 향상을 강조한다. 이는 open-vocabulary segmentation에서 가장 까다로운 long-tail, unseen-heavy setting에서 큰 의미가 있다.

### Multi-domain 결과

MESS benchmark에서 ViT-B 기반 CAT-Seg는 평균 31.96, ViT-L 기반 CAT-Seg는 평균 34.70을 기록했다. 이는 기존 강한 baseline들보다 높은 평균이다. 일반 도메인과 농업/생물 분야에서 특히 강했고, 공학 분야에서도 일부 개선을 보였다.

다만 의료 분야는 결과가 불안정하다. ViT-B 기반에서는 의료 평균이 28.09, ViT-L 기반에서는 24.70으로 오히려 낮아졌고, 저자들도 이 영역에서는 random baseline과 크게 차이나지 않는 경우가 있다고 인정한다. 논문은 CLIP이 이런 특수 도메인에 대한 지식을 충분히 갖지 못했을 가능성을 언급하지만, 이는 어디까지나 저자들의 해석이지 엄밀한 원인 분석은 아니다.

### Feature aggregation과 Cost aggregation 비교

이 논문의 가장 핵심적인 실험 중 하나다. feature aggregation + fine-tuning은 A-847에서 5.6, PC-459에서 12.8이었다. 반면 cost aggregation + fine-tuning은 A-847에서 14.7, PC-459에서 23.2를 기록했다. 성능 차이가 매우 크다.

이 비교는 논문의 주장을 직접 뒷받침한다. fine-tuning 자체가 중요한 것이 아니라, 무엇을 aggregation하느냐가 중요하다는 것이다. feature를 decoder로 직접 변형하면 seen class로 쏠리는 과적합이 심해지고 unseen alignment가 깨진다. cost를 aggregation하면 CLIP의 image-text 정렬을 덜 손상시키면서 segmentation에 필요한 refinement를 수행할 수 있다.

### Ablation study

구성요소 분석에서도 설계 선택이 비교적 일관되게 검증된다. cost aggregation baseline 위에 spatial aggregation만 넣으면 일부 향상, class aggregation만 넣으면 다른 지표에서 향상, 둘을 함께 넣으면 전반적으로 더 좋아진다. 최종적으로 embedding guidance까지 넣은 모델이 가장 좋다. 예를 들어 ViT-L 기준 최종 모델은 A-847 16.0, PC-459 23.8을 기록했다.

upsampling decoder의 기여도도 분명하다. ViT-B 기준 decoder를 제거하면 A-847이 9.9, PC-459가 16.1로 떨어지고, 전체 CAT-Seg는 각각 12.0, 19.0이다. 즉 coarse cost refinement만으로는 충분하지 않고, 중간 feature를 이용한 high-resolution 복원이 실제로 중요하다는 뜻이다.

### Fine-tuning 방식 분석

freeze, prompt tuning, full fine-tuning, attention-only fine-tuning, QK/KV/QV fine-tuning을 비교한 결과, 양 encoder 모두에서 QV projection만 fine-tuning하는 방식이 가장 좋은 종합 성능을 보였다. 최종 방식은 A-847 16.0, PC-459 23.8을 기록했고, full fine-tuning보다 더 좋은 성능을 내면서 학습 파라미터 수도 줄였다.

full fine-tuning은 393.2M learnable parameter와 26.8 GiB 메모리가 필요했지만, 제안 방식은 70.3M parameter와 20.9 GiB 메모리로 더 가볍다. 즉 성능뿐 아니라 효율성 측면에서도 설계가 실용적이다.

### 소규모 학습 데이터셋 실험과 효율성

저자들은 COCO-Stuff보다 작은 A-150, PC-59로 학습했을 때도 CAT-Seg가 기존 방법보다 잘 일반화한다고 보였다. 예를 들어 A-150로 학습했을 때도 A-847에서 14.4, PC-459에서 16.2를 기록해 비교 방법들보다 우수했다. 이는 cost aggregation이 단순한 dataset memorization보다 더 강한 transfer 성질을 가질 수 있음을 시사한다.

효율성 비교에서는 inference time이 특히 눈에 띈다. 단일 RTX 3090 기준 CAT-Seg의 inference time은 0.54초로, ZegFormer 2.70초, ZSseg 2.73초, OVSeg 2.00초보다 훨씬 빠르다. GFLOPs도 2121.1로 경쟁 방법들보다 크게 낮다. 저자들은 region generator 같은 별도 mask module이 없기 때문이라고 설명한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 방법 설계가 잘 연결된다는 점이다. “왜 기존 fine-tuning이 unseen class에서 망가지는가”를 joint embedding space 붕괴 관점에서 설명하고, 그 대안으로 feature 대신 cost를 aggregation한다는 설계가 실험적으로도 강하게 지지된다. 특히 seen-heavy 데이터셋보다 unseen-heavy 벤치마크에서 성능 향상이 더 크다는 점은 논문의 주장을 설득력 있게 만든다.

둘째, spatial aggregation과 class aggregation의 분리는 단순한 모듈 추가가 아니라, image-text cost volume의 구조를 반영한 설계다. spatial 축과 class 축을 분리해 다루는 것은 open-vocabulary segmentation의 요구사항, 특히 variable class set과 permutation invariance를 잘 반영한다.

셋째, 성능과 효율을 동시에 잡았다는 점도 중요하다. region proposal 기반 방법보다 빠르고, full fine-tuning보다 메모리 효율적이며, 다양한 VLM으로도 확장 가능하다는 점이 supplementary에서 추가로 제시된다. EVA-CLIP-L/14나 SigLIP-L/16에 적용했을 때도 성능이 더 올라갔다는 결과는 framework의 범용성을 보여준다.

한계도 분명하다. 첫째, 의료 분야 같은 특수 도메인에서는 성능이 매우 제한적이다. 논문도 이를 인정하며, CLIP 자체의 도메인 지식 부족 가능성을 언급한다. 즉 이 방법은 CLIP의 일반적 open-vocabulary 능력을 잘 보존하고 확장하지만, foundation model이 원래 잘 모르는 영역까지 해결해주지는 못한다.

둘째, 평가는 다른 segmentation dataset의 ground truth를 open-vocabulary evaluation에 그대로 활용하는데, supplementary limitation 섹션에서 저자들 스스로 annotation ambiguity 문제를 지적한다. 즉 benchmark 수치가 유용하긴 하지만, open-vocabulary setting에서 완전히 이상적인 평가라고 보기 어렵다.

셋째, patch inference 전략은 실용적이지만, 이는 곧 기본 ViT backbone이 고해상도 dense prediction에 여전히 제약이 있음을 뜻한다. 논문은 overlapping patch를 잘 병합해 성능을 끌어올리지만, 이는 inference complexity를 완전히 없애는 해결책은 아니다.

넷째, 왜 QV fine-tuning이 최선인지에 대한 이론적 설명은 충분히 깊지 않다. 실험적으로는 강하지만, attention 내부 어떤 변화가 unseen generalization을 보존하는지까지는 명확히 해부하지 않는다. 이는 후속 연구 여지가 있는 부분이다.

## 6. 결론

이 논문은 open-vocabulary semantic segmentation에서 CLIP을 어떻게 dense prediction에 적응시킬 것인가에 대해, 매우 직접적이고 실용적인 답을 제시한다. 핵심은 feature를 세게 변형하는 대신 image-text cosine similarity로 만든 cost volume을 aggregation하는 것이다. 여기에 spatial aggregation, class aggregation, embedding guidance, 그리고 효율적인 CLIP fine-tuning 전략을 결합해 CAT-Seg를 구성했다.

실험 결과는 이 접근이 단순한 구조적 변형이 아니라 실제로 강력한 이점이 있음을 보여준다. 표준 benchmark에서는 새로운 state-of-the-art를 달성했고, multi-domain benchmark에서도 평균적으로 가장 강한 일반화 성능을 보였다. 특히 unseen class가 많은 설정에서 성능 차이가 크게 벌어진다는 점은, 이 방법이 open-vocabulary segmentation의 본질적 문제를 잘 건드렸다는 신호로 볼 수 있다.

실제 적용 측면에서도 의미가 있다. 별도 region generator 없이 빠르게 추론할 수 있고, 다양한 VLM으로 이식 가능하며, 작은 학습 데이터셋에서도 상대적으로 좋은 일반화를 보인다. 따라서 이 연구는 open-vocabulary segmentation을 더 실용적이고 확장 가능한 방향으로 밀어준 작업으로 평가할 수 있다. 다만 의료 같은 극단적 특수 도메인에서는 foundation model 자체의 사전 지식 한계가 여전히 병목으로 남아 있으며, 평가 데이터셋의 신뢰성 문제도 후속 연구에서 더 정교하게 다뤄질 필요가 있다.
