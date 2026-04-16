# ZegFormer: Decoupling Zero-Shot Semantic Segmentation

- **저자**: Jian Ding, Nan Xue, Gui-Song Xia, Dengxin Dai
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2112.07910

## 1. 논문 개요

이 논문은 zero-shot semantic segmentation (ZS3), 특히 generalized zero-shot semantic segmentation (GZS3) 문제를 다룬다. 목표는 학습 때 본 적 없는 unseen class까지 포함하여 이미지의 각 픽셀에 의미 있는 semantic label을 부여하는 것이다. 기존 연구들은 이 문제를 대체로 “픽셀마다 zero-shot classification을 수행하는 문제”로 다뤘다. 즉, 픽셀 feature를 semantic embedding 공간에 맞추고, seen class에서 학습한 지식을 unseen class로 전이하려 했다.

저자들은 이 접근이 근본적으로 부자연스럽다고 본다. 사람은 보통 개별 픽셀을 보고 의미를 붙이지 않고, 먼저 물체나 영역 단위로 묶은 뒤 그 segment에 이름을 붙인다. 또한 기존 방식은 주로 text-only language model에서 얻은 semantic embedding에 의존하는데, 최근 성능이 매우 좋은 vision-language model, 예를 들어 CLIP처럼 image-text pair로 사전학습된 모델을 픽셀 수준 ZS3에 직접 통합하기가 어렵다.

이 문제의 중요성은 분명하다. 실제 환경에서는 segmentation 대상 클래스가 고정되어 있지 않으며, 모든 클래스를 충분한 pixel annotation과 함께 수집하는 것은 거의 불가능하다. 따라서 새로운 클래스에 대해 재학습 없이 일반화할 수 있는 segmentation 모델은 open-vocabulary perception으로 가는 중요한 단계다. 이 논문은 바로 이 지점을 겨냥해, ZS3를 더 자연스럽고 확장성 있게 푸는 새로운 문제 정식화를 제안한다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 ZS3를 두 개의 하위 문제로 분리(decoupling)하는 것이다.

첫째는 **class-agnostic grouping**이다. 여기서는 어떤 픽셀들이 같은 segment에 속하는지만 예측한다. 즉, “이 픽셀들이 하나의 물체/영역인가?”를 묻는 문제이며, 클래스 이름 자체는 필요 없다. 저자들은 이 단계가 semantic category와 직접적으로 연결되지 않으므로 seen class에서 배운 grouping 능력이 unseen class에도 비교적 잘 전이된다고 본다.

둘째는 **segment-level zero-shot classification**이다. 첫 단계에서 얻은 각 segment에 대해 semantic label을 부여한다. 이때 분류 단위가 픽셀이 아니라 segment이므로, CLIP 같은 vision-language model이 학습한 표현과 훨씬 잘 맞는다. 언어는 원래 픽셀을 설명하기보다 사물, 장면의 부분, 영역을 설명하기 때문이다.

기존 접근과의 가장 큰 차별점은 바로 이 분리다. 기존 ZS3는 사실상 픽셀마다 unseen 분류를 수행하려고 했고, 그 결과 pixel-level feature와 text embedding 사이의 대응이 약하고 불안정했다. 반면 이 논문은 먼저 segment를 만들고, 그 다음 segment 단위에서 zero-shot 분류를 수행한다. 저자들은 이것이 더 인간적인 절차이며, 더 강건하고, CLIP의 image encoder와 text encoder를 자연스럽게 활용할 수 있게 만든다고 주장한다.

## 3. 상세 방법 설명

### 3.1 문제 정식화

이미지 $I$가 주어졌을 때 semantic segmentation은 두 매핑 $(R, L)$을 찾는 문제로 정의된다. $R$은 이미지 영역 $\Omega$를 서로 겹치지 않는 $N$개의 segment $\{R_i\}_{i=1}^N$로 나누고, $L$은 각 segment $R_i$에 클래스 label $c \in C$를 붙인다.

논문은 클래스 집합 관점에서 세 가지 설정을 정리한다.

- fully supervised semantic segmentation: 테스트 클래스 집합 $E \subseteq S$
- zero-shot semantic segmentation (ZS3): $S \cap E = \varnothing$
- generalized ZS3 (GZS3): $S \subset E$

여기서 $S$는 학습 중 본 seen class 집합이고, $U = E - E \cap S$는 unseen class 집합이다. 이 논문은 주로 GZS3를 대상으로 평가한다.

저자들은 기존 pixel-level zero-shot classification이 사실 자신들의 decoupled formulation의 특수한 경우라고 말한다. 극단적으로 각 픽셀을 하나의 segment로 보면 기존 방식이 되기 때문이다. 그러나 그런 방식은 class-agnostic grouping이라는 일반화가 쉬운 부분과, semantic transfer라는 어려운 부분을 섞어버린다.

### 3.2 전체 구조: ZegFormer

ZegFormer는 MaskFormer 계열 구조를 바탕으로 한다. 입력 이미지는 backbone과 pixel decoder를 거쳐 feature map으로 변환되고, transformer decoder는 $N$개의 query를 사용해 $N$개의 **segment-level embedding**을 만든다. 이 embedding은 이후 두 갈래로 나뉜다.

하나는 **mask projection**이다. 각 query embedding으로부터 mask embedding을 만들고, 이것을 고해상도 feature map과 내적하여 binary mask를 예측한다. 다른 하나는 **semantic projection**이다. 각 query embedding을 text embedding과 비교 가능한 semantic embedding으로 사상하여 segment-level classification을 수행한다.

또한 full model에서는 CLIP의 **image encoder**도 사용한다. 각 segment mask로부터 sub-image를 구성하고, 이를 image encoder에 넣어 image embedding을 얻는다. 이 embedding을 text embedding과 비교해 추가적인 classification score를 얻는다. 최종적으로 segment embedding 기반 score와 image embedding 기반 score를 fusion한다.

### 3.3 Segment embedding과 class-agnostic grouping

transformer decoder가 출력한 각 query에 대해 두 종류의 embedding이 정의된다.

- semantic segment embedding: $G_q \in \mathbb{R}^d$
- mask embedding: $B_q \in \mathbb{R}^d$

pixel decoder 출력 feature map을 $F(I) \in \mathbb{R}^{d \times H \times W}$라 하면, query $q$의 binary mask prediction은 다음과 같다.

$$
m_q = \sigma(B_q \cdot F(I)) \in [0,1]^{H \times W}
$$

여기서 $\sigma$는 sigmoid 함수다. 이 식의 의미는 간단하다. query 하나가 하나의 잠재적 segment를 나타내고, 그 query에서 나온 mask embedding이 전체 feature map의 각 위치와 얼마나 잘 맞는지를 계산해 mask를 얻는 것이다.

중요한 점은 이 grouping 단계가 class label에 직접 의존하지 않는다는 것이다. 그래서 seen/unseen 구분 없이 “같은 영역을 묶는 능력”을 전이할 수 있다고 본다.

### 3.4 Segment-level classification with text embedding

각 클래스 이름은 prompt template에 삽입된 뒤 pre-trained text encoder로 들어간다. 예를 들어 `"A photo of a {class name} in the scene"` 같은 형태다. 그러면 각 클래스 $c$에 대한 text embedding $T_c \in \mathbb{R}^d$를 얻는다. 학습 때는 seen class만 사용하고, 추론 때는 seen + unseen class를 모두 사용한다.

또한 “no object”를 위한 learnable embedding $T_0$를 따로 둔다. 이것은 어떤 query가 ground-truth segment와 충분히 맞지 않을 때 배경성 query로 처리하기 위해 필요하다.

query $q$에 대한 class probability는 cosine similarity 기반 softmax로 계산된다.

$$
p_q(c) =
\frac{\exp\left(\frac{1}{\tau}s_c(T_i, G_q)\right)}
{\sum_{i=0}^{|C|}\exp\left(\frac{1}{\tau}s_c(T_i, G_q)\right)}
$$

여기서 $s_c(e,e')=\frac{e \cdot e'}{|e||e'|}$는 cosine similarity이고, $\tau$는 temperature다. 직관적으로는 query가 표현하는 segment가 어떤 클래스 이름의 text embedding과 가장 비슷한지를 보는 것이다.

### 3.5 Segment-level classification with image embedding

저자들은 segment embedding 기반 분류만으로도 standalone ZS3가 가능하다고 말한다. 하지만 데이터가 작으면 일반화가 약할 수 있으므로, CLIP의 image encoder를 활용해 segment classification을 보강한다.

예측된 mask $m_q$와 입력 이미지 $I$로부터 sub-image $I_q = f(m_q, I)$를 만든다. 여기서 $f$는 crop, mask, crop-and-mask 같은 전처리 함수다. 이 sub-image를 pre-trained image encoder에 넣어 image embedding $A_q$를 얻고, 이를 text embedding과 비교해 또 하나의 class probability $p'_q(c)$를 계산한다.

이 단계의 핵심은 decoupling 덕분에 segment 단위 영상 조각을 만들 수 있다는 점이다. 기존 pixel-level formulation에서는 이런 방식으로 CLIP image encoder를 자연스럽게 쓰기 어렵다.

### 3.6 학습 절차

학습은 seen class pixel label만 사용한다. predicted mask와 ground-truth mask 사이에는 bipartite matching을 수행한다. 즉, 각 query가 어떤 ground-truth segment를 담당할지 일대일 대응을 찾는다.

분류 loss는 matched segment에 대해 negative log-likelihood 형태다.

$$
-\log(p_q(c_q^{gt}))
$$

여기서 $c_q^{gt}$는 query $q$가 매칭된 ground-truth의 클래스이며, 매칭되지 않으면 “no object”다.

mask loss는 ground-truth segment $R_q^{gt}$와 predicted mask $m_q$ 사이에 적용되며, 논문은 **dice loss와 focal loss의 조합**을 사용한다고 명시한다. 정확한 결합식은 제공된 본문에 명시되어 있지 않지만, 두 손실을 함께 사용했다는 점은 분명하다.

또한 appendix에 따르면 transformer decoder는 6개 층을 사용하고, 각 층 뒤에 동일한 loss를 적용한다.

### 3.7 추론 절차와 세 가지 변형

추론 시에는 query별 mask와 class score를 결합해 최종 per-pixel prediction을 만든다.

#### ZegFormer-seg

segment embedding 기반 score $p_q(c)$만 사용한다. 픽셀 $(h,w)$에서 클래스 $c$의 score는 각 query의 class score와 그 query mask 값의 합으로 계산된다.

$$
\sum_{q=1}^{N} p_q(c)\cdot m_q[h,w]
$$

하지만 GZS3에서는 seen class 쏠림 문제가 있으므로 calibration factor $\gamma$를 사용해 seen class 점수를 깎는다. 최종 예측은 다음과 같다.

$$
\arg\max_{c \in S+U}
\sum_{q=1}^{N} p_q(c)\cdot m_q[h,w] - \gamma \cdot I[c \in S]
$$

#### ZegFormer-img

위 식에서 $p_q(c)$ 대신 image embedding 기반 분포 $p'_q(c)$를 쓴다.

#### ZegFormer

full model에서는 두 분포를 fusion한다. unseen class에 대해서는 $p_q(c)$와 $p'_q(c)$의 기하평균을 사용하고, seen class에 대해서는 $p'_q(c)$가 seen class 내부 구분에 과도하게 영향을 주지 않도록 별도의 평균항 $p_{q,\text{avg}}$와 결합한다.

$$
p_{q,\text{fusion}}(c) =
\begin{cases}
p_q(c)^\lambda \cdot p_{q,\text{avg}}^{(1-\lambda)} & \text{if } c \in S \\
p_q(c)^{(1-\lambda)} \cdot p'_q(c)^\lambda & \text{if } c \in U
\end{cases}
$$

여기서

$$
p_{q,\text{avg}} = \sum_{j \in S} p'_q(j) / |S|
$$

이다. 이 설계는 seen/unseen score 스케일을 맞추면서, seen class 내부 구별은 주로 segment embedding 기반 분류가 담당하게 하려는 의도다. 논문이 강조하는 포인트는 $p_q(c)$와 $p'_q(c)$가 상보적(complementary)이라는 점이다.

## 4. 실험 및 결과

### 4.1 데이터셋과 평가 지표

논문은 COCO-Stuff, PASCAL-VOC, ADE20k-Full 세 데이터셋을 사용한다.

COCO-Stuff에서는 총 171개 valid class 중 156개를 seen, 15개를 unseen으로 사용한다. 118,287장 train, 5,000장 val을 사용한다. PASCAL-VOC에서는 기존 설정을 따라 15 seen, 5 unseen으로 나누고, 10,582장 학습 이미지와 1,449장 테스트 이미지를 사용한다.

ADE20k-Full은 이 논문이 새롭게 제안한 더 어려운 GZS3 benchmark다. 원래 3,000개 이상 open-vocabulary category가 있지만, train과 validation에 모두 존재하는 847개 클래스를 평가 대상으로 삼는다. 이 중 10장 초과로 등장하는 572개를 seen, 10장 미만으로 등장하는 275개를 unseen으로 둔다. unseen class 수가 매우 많아 기존 benchmark보다 훨씬 어렵다.

평가는 class-related metric으로 seen mIoU, unseen mIoU, harmonic mean을 보고, class-agnostic grouping 자체는 boundary precision/recall/F-score인 $P_b$, $R_b$, $F_b$로 평가한다.

### 4.2 구현 세부사항

구현은 Detectron2 기반이다. 주된 backbone은 ResNet-50이고, SOTA 비교에는 ResNet-101도 사용한다. pixel decoder는 FPN, vision-language model은 ViT-B/16 CLIP의 text encoder와 image encoder를 사용한다. query 수는 기본 100개, transformer 및 query embedding 차원은 256, text embedding 차원이 512라서 projection layer를 둔다. temperature $\tau$는 0.01이다. batch size는 32이며, COCO-Stuff는 60k iteration, PASCAL VOC는 10k iteration 학습한다.

appendix에 따르면 출력 stride는 4, transformer decoder는 6층, 학습 시 crop 크기는 COCO 640×640, ADE20k-Full과 VOC는 512×512다.

### 4.3 decoupling formulation 대 pixel-level baseline 비교

논문은 SPNet를 같은 코드베이스와 공통 설정으로 재구현한 SPNet-FPN을 baseline으로 둔다. 이 비교는 논문의 핵심 주장인 “decoupling이 진짜 효과가 있는가?”를 검증하는 데 중요하다.

COCO-Stuff에서 CLIP text embedding을 사용했을 때, SPNet-FPN은 unseen mIoU가 11.0인 반면, ZegFormer-seg는 21.4를 기록한다. harmonic mean도 16.4에서 27.2로 크게 상승한다. fastText+word2vec를 쓴 경우에도 ZegFormer-seg가 우수하다. 특히 text embedding을 ft+w2v에서 CLIP text로 바꿨을 때, SPNet-FPN의 unseen mIoU 향상은 +4.1이지만 ZegFormer-seg는 +10.6이다. 이는 segment-level feature가 CLIP text representation과 더 잘 정렬된다는 저자 주장과 맞아떨어진다.

class-agnostic grouping 측면에서도 차이가 크다. COCO-Stuff에서 $F_b$ 기준으로 SPNet-FPN은 40.3 또는 42.6 수준인데, ZegFormer-seg는 49.4 또는 50.4를 기록한다. 즉, decoupled formulation은 unseen class를 포함한 grouping 자체도 더 잘한다.

### 4.4 전체 모델 성능과 ablation

sub-image 전처리 방식은 full model 성능에 큰 영향을 준다. 단순 crop은 unseen mIoU 19.7로 ZegFormer-seg의 21.4보다도 낮다. mask만 사용하면 31.0, crop+mask는 33.1로 가장 좋다. 이는 segment classification용 sub-image에는 배경 억제와 적절한 local crop이 모두 필요함을 보여준다.

ZegFormer-seg와 ZegFormer-img를 비교하면, 저자들은 image-embedding branch가 thing class에는 더 강하고, segment-embedding branch가 stuff class에는 더 강하다고 분석한다. 예를 들어 giraffe, suitcase, carrot 같은 물체 클래스는 ZegFormer-img가 강하지만, grass, playing-field, river 같은 stuff 클래스는 ZegFormer-seg가 낫다. 이 상보성이 fusion의 근거가 된다.

### 4.5 기존 SOTA와 비교

PASCAL VOC에서 ZegFormer는 seen 86.4, unseen 63.6, harmonic 73.3을 기록한다. 이는 기존 방법들보다 매우 큰 폭의 향상이다. 논문은 Joint보다 unseen mIoU 기준 31포인트, self-training 포함 기준으로도 SIGN보다 22포인트 높다고 강조한다.

COCO-Stuff에서는 ZegFormer가 seen 36.6, unseen 33.2, harmonic 34.8이다. self-training을 포함한 이전 최고 수준인 STRICT의 unseen mIoU 30.3보다 약 3포인트 높다. 개선 폭은 VOC보다 작지만, 이 데이터셋이 더 크고 어려운 점을 고려하면 의미 있는 향상이다.

중요한 점은 ZegFormer가 **discriminative** 방식이며, generative method나 self-training 방법처럼 복잡한 multi-stage 학습이나 새 클래스 등장 시 재학습이 필요 없다는 것이다. 즉, 성능뿐 아니라 실용성 면에서도 장점이 있다.

### 4.6 ADE20k-Full 결과

ADE20k-Full은 unseen class가 275개로 많아 훨씬 도전적이다. 여기서 SPNet-FPN은 unseen mIoU 0.9, ZegFormer는 5.3을 기록한다. 수치 자체는 낮지만, fully supervised model이 unseen 포함 평가에서 5.6에 그친다는 점을 생각하면 ZegFormer의 5.3은 매우 인상적이다. 논문도 이를 근거로 “우리 방법이 fully supervised MaskFormer에 근접한다”고 주장한다.

다만 이 결과는 동시에 이 benchmark가 매우 어렵다는 점도 보여준다. supervised model조차 unseen mIoU가 5.6에 불과하므로, open-vocabulary segmentation이 아직 해결되지 않았음을 의미한다.

### 4.7 ZS3 setting, 속도 분석, prompt ensemble

appendix에서 ZS3 설정도 보고한다. COCO-Stuff에서 unseen class만 대상으로 평가하면 ZegFormer는 61.5 mIoU, ZegFormer-seg는 48.8, SPNet-FPN은 41.3이다. ADE20k-Full에서도 ZegFormer는 18.7, ZegFormer-seg는 9.4, SPNet-FPN은 7.4다.

속도 분석도 흥미롭다. 논문은 pixel-level zero-shot classification의 복잡도가 $O(H \times W \times C \times K)$인 반면, decoupled formulation은 $O(N \times C \times K)$라고 설명한다. 여기서 $N$은 segment 수이고 보통 $H \times W$보다 훨씬 작다. 실제로 COCO-Stuff에서 ZegFormer-seg는 25.5 FPS, SPNet-FPN은 17.0 FPS다. ADE20k-Full처럼 class 수 $K$가 커지면 SPNet-FPN 속도는 더 크게 불리해진다.

prompt ensemble도 시험했는데, 단일 prompt보다 여러 prompt를 평균한 것이 harmonic mean에서 34.0에서 34.4로 소폭 향상된다. 개선은 작지만 일관된 이득으로 보인다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제를 새롭게 정의하는 방식이 매우 설득력 있다는 점이다. 단순히 기존 pipeline의 세부 요소를 바꾼 것이 아니라, ZS3 자체를 “grouping + segment classification”으로 재구성했다. 이 정식화는 인간의 인지 절차와도 더 잘 맞고, CLIP 같은 vision-language model을 자연스럽게 끌어들인다.

두 번째 강점은 실험적 근거가 강하다는 점이다. 저자들은 단순히 최종 mIoU만 보고하지 않고, class-agnostic grouping metric, sub-image preprocessing, segment-vs-image branch 비교, 속도 분석, 많은 unseen class가 있을 때의 안정성까지 폭넓게 검증했다. 특히 847개 class를 넣었을 때 pixel-level baseline이 매우 불안정해지는 반면, decoupled approach는 비교적 견고하다는 시각화와 속도 분석은 논문의 주장을 잘 뒷받침한다.

세 번째 강점은 실용성이다. self-training이나 generative feature synthesis 없이도 강한 성능을 내고, 새 클래스에 대해 on-the-fly로 적용 가능하다는 점은 실제 시스템에서 중요하다.

반면 한계도 분명하다. 저자들 스스로 인정하듯, ZegFormer-seg는 학습 데이터 규모가 작을 때 성능이 충분히 좋지 않을 수 있다. transformer 기반 구조가 데이터 효율성이 낮을 가능성이 있다. 또한 full model의 성능 향상을 위해 CLIP image encoder branch가 필요한데, 이 branch는 속도를 상당히 낮춘다. COCO-Stuff에서 ZegFormer-seg는 25.5 FPS지만 full ZegFormer는 6.0 FPS다. 따라서 정확도와 효율의 trade-off가 존재한다.

또 하나의 해석상 주의점은 CLIP text embedding 사용이 “순수한” GZS3보다 완전히 엄격한 설정은 아닐 수 있다는 것이다. 논문도 객체 검출 분야의 관행을 따라, CLIP text encoder 사용을 완전한 제약의 zero-shot보다 다소 완화된 설정이라고 인정한다. 하지만 현실적 가치가 더 높다고 본다. 이 점은 성능 해석에서 공정하게 짚고 넘어가야 한다.

마지막으로, 논문은 mask loss의 정확한 결합 형태나 fusion 하이퍼파라미터 $\lambda$, calibration factor $\gamma$의 상세 튜닝 전략을 제공된 본문에서 충분히 설명하지 않는다. 따라서 재현에는 코드나 appendix 추가 정보가 필요할 수 있다.

## 6. 결론

이 논문은 zero-shot semantic segmentation을 픽셀 단위 분류 문제로 보던 기존 관점을 버리고, **class-agnostic grouping**과 **segment-level zero-shot classification**으로 분해하는 새로운 정식화를 제안한다. 그 위에 구축된 ZegFormer는 transformer decoder로 segment embedding을 생성하고, 이를 mask prediction과 semantic classification에 동시에 활용하며, 더 나아가 CLIP image encoder까지 결합해 성능을 끌어올린다.

실험적으로 ZegFormer는 PASCAL VOC와 COCO-Stuff에서 기존 방법을 큰 폭으로 능가하고, 특히 unseen class 성능에서 매우 강하다. 또한 ADE20k-Full처럼 unseen class 수가 많은 어려운 설정에서도 pixel-level baseline보다 훨씬 안정적이며, fully supervised 모델에 가까운 수준까지 도달한다. 이 결과는 open-vocabulary segmentation과 실제 환경에서의 확장 가능한 semantic segmentation을 향한 중요한 진전으로 볼 수 있다.

종합하면, 이 논문의 핵심 기여는 단순히 성능 향상 자체보다도, ZS3를 어떻게 모델링해야 하는지에 대한 방향을 바꿨다는 데 있다. 향후 few-shot semantic segmentation이나 더 넓은 open-vocabulary dense prediction 문제에도 이 아이디어가 영향을 줄 가능성이 크다.
