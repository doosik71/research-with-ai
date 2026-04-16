# Segmenter: Transformer for Semantic Segmentation

- **저자**: Robin Strudel, Ricardo Garcia, Ivan Laptev, Cordelia Schmid
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2105.05633

## 1. 논문 개요

이 논문은 semantic segmentation을 위해 convolution 없이 transformer만으로 구성된 모델인 **Segmenter**를 제안한다. semantic segmentation의 목표는 이미지의 각 pixel에 semantic class를 할당하는 것이다. 저자들은 이 문제가 단순히 작은 지역 patch만 보고 결정하기 어렵고, 장면 전체의 문맥(global context)이 매우 중요하다고 본다. 예를 들어 어떤 patch가 사람의 옷인지, 보도인지, 하늘인지 판단하려면 주변 물체와 장면 구조를 함께 봐야 하는 경우가 많다.

기존의 강력한 segmentation 방법들은 대체로 Fully Convolutional Network(FCN) 기반 encoder-decoder 구조를 사용한다. 하지만 convolution는 본질적으로 local 연산이기 때문에, 넓은 문맥을 보려면 dilated convolution, pyramid pooling, attention module 같은 별도 장치를 계속 추가해야 한다. 저자들은 이런 점이 convolution 기반 접근의 구조적 한계라고 본다.

이를 해결하기 위해 이 논문은 semantic segmentation을 **sequence-to-sequence 문제**로 보고, 이미지를 patch들의 시퀀스로 바꾼 뒤 transformer encoder로 전역 상호작용을 처음부터 끝까지 모델링한다. 그리고 encoder가 만든 patch 표현으로부터 segmentation map을 복원한다. 핵심 주장은, segmentation에서는 global context가 특히 중요하므로 transformer의 전역 self-attention이 구조적으로 유리하다는 것이다.

논문의 중요성은 두 가지 측면에서 크다. 첫째, 당시 segmentation 분야의 주류였던 CNN/FCN 중심 흐름에서 벗어나 **pure transformer segmentation**의 가능성을 강하게 보여주었다. 둘째, 단순한 linear decoder만으로도 강한 성능을 내고, 추가로 제안한 **mask transformer decoder**를 사용하면 더 좋아진다는 점을 통해, segmentation용 transformer 설계 방향을 구체적으로 제시했다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 명확하다. **이미지를 patch token 시퀀스로 바꾸고, ViT 스타일 encoder로 각 patch가 전체 장면을 참조하며 의미론적으로 풍부한 표현을 갖게 만든 뒤, 이를 segmentation mask로 디코딩한다**는 것이다.

기존 convolution 기반 segmentation은 초반 layer에서 local receptive field만 보게 되므로, 전역 문맥은 뒤쪽 layer나 별도 모듈을 통해 우회적으로 넣는다. 반면 Segmenter는 첫 layer부터 self-attention을 사용하므로, patch 간 장거리 관계를 곧바로 반영할 수 있다. 논문 Figure 1에서도 첫 layer attention map만으로도 비슷한 semantic 영역이 일찍 묶이는 모습을 보여준다.

이 논문이 제시하는 차별점은 다음과 같다.

첫째, **encoder뿐 아니라 전체 segmentation 파이프라인을 transformer 관점에서 단순하게 재구성**했다는 점이다. 당시 SETR는 ViT backbone 위에 CNN decoder를 얹었고, Swin Transformer는 local window 구조와 FCN류 decoder를 사용했다. 반면 Segmenter는 논문 기준으로 convolution-free encoder-decoder를 지향한다.

둘째, **decoder를 두 가지 방식으로 체계적으로 비교**한다. 하나는 간단한 patch-wise linear classifier이고, 다른 하나는 learnable class embedding을 이용해 class mask를 생성하는 **mask transformer**이다. 특히 후자는 class embedding과 patch embedding의 내적을 통해 클래스별 mask를 만들며, 단순 분류 이상의 구조를 디코딩에 도입한다.

셋째, **patch size를 정확도와 속도 사이의 핵심 trade-off 수단으로 본다**. patch를 작게 하면 더 세밀한 경계와 작은 객체를 잡을 수 있지만, 시퀀스 길이가 길어져 계산량이 크게 늘어난다. 이 논문은 이 trade-off를 segmentation 맥락에서 매우 설득력 있게 실험으로 보여준다.

## 3. 상세 방법 설명

### 전체 구조

Segmenter는 transformer 기반 encoder-decoder 구조이다. 입력 이미지 $x \in \mathbb{R}^{H \times W \times C}$를 $(P,P)$ 크기의 patch로 나누고, 각 patch를 flatten한 뒤 선형 투영하여 token embedding으로 바꾼다. patch 개수는

$$
N = \frac{HW}{P^2}
$$

이다. 각 patch $x_i$에 대해 embedding을 만들면

$$
x_0 = [Ex_1, \ldots, Ex_N] \in \mathbb{R}^{N \times D}
$$

가 된다. 여기서 $E \in \mathbb{R}^{D \times (P^2C)}$는 learnable projection matrix이고, $D$는 token dimension이다.

그 다음 위치 정보를 넣기 위해 learnable positional embedding $pos \in \mathbb{R}^{N \times D}$를 더해

$$
z_0 = x_0 + pos
$$

를 만든다. 이 시퀀스를 transformer encoder에 넣어 contextualized patch representation $z_L$를 얻는다.

### Encoder

encoder는 표준 transformer layer를 $L$번 쌓은 구조다. 각 layer는 multi-head self-attention(MSA)와 MLP block으로 구성되며, 각각 앞에 layer normalization이 있고 뒤에 residual connection이 있다. 논문은 이를 다음처럼 적는다.

$$
a_{i-1} = \text{MSA}(\text{LN}(z_{i-1})) + z_{i-1}
$$

$$
z_i = \text{MLP}(\text{LN}(a_{i-1})) + a_{i-1}
$$

여기서 $i \in \{1, \ldots, L\}$이다.

self-attention은 query, key, value를 선형변환으로 만든 뒤 다음과 같이 계산한다.

$$
\text{MSA}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

이 식의 의미는, 각 patch token이 다른 모든 patch token과의 관련도를 계산하고, 그 가중합으로 자신의 표현을 업데이트한다는 것이다. 따라서 특정 patch가 자기 주변뿐 아니라 이미지 반대편 patch까지 참고할 수 있다. segmentation처럼 “이 부분이 무엇인가”를 주변 전체 문맥으로 판단해야 하는 문제에 잘 맞는다.

최종 encoder 출력은

$$
z_L = [z_{L,1}, \ldots, z_{L,N}]
$$

형태의 patch sequence이다. 이 각 patch embedding은 이미 global context를 반영한 semantic representation이다.

### Decoder 1: Linear Decoder

가장 단순한 baseline은 linear decoder이다. encoder 출력 $z_L \in \mathbb{R}^{N \times D}$에 point-wise linear layer를 적용해 각 patch의 class logits를 만든다.

$$
z_{\text{lin}} \in \mathbb{R}^{N \times K}
$$

여기서 $K$는 class 수다. 이 시퀀스를 다시 $(H/P) \times (W/P) \times K$ 형태의 2D grid로 reshape한 뒤, bilinear interpolation으로 원래 해상도 $H \times W \times K$까지 upsample한다. 마지막으로 class dimension에 softmax를 적용해 각 pixel의 class score를 얻는다.

이 decoder의 장점은 매우 단순하다는 점이다. encoder가 patch를 충분히 잘 표현하면, 단순한 선형 분류만으로도 강한 segmentation 성능이 나온다는 것이 이 논문의 중요한 관찰이다.

### Decoder 2: Mask Transformer

논문의 핵심 decoder는 **mask transformer**이다. 저자들은 각 class마다 learnable embedding 하나를 둔다.

$$
cls = [cls_1, \ldots, cls_K] \in \mathbb{R}^{K \times D}
$$

각 class embedding은 하나의 semantic class에 대응하도록 학습된다. 이 class embedding들과 encoder의 patch encoding $z_L$을 함께 decoder transformer에 넣는다. decoder는 $M$개 layer의 transformer encoder 구조로 구현된다.

decoder를 통과한 뒤, patch embedding $z'_M \in \mathbb{R}^{N \times D}$와 class embedding $c \in \mathbb{R}^{K \times D}$를 얻고, 이 둘을 L2-normalize한 후 scalar product를 계산해 class mask를 만든다.

$$
\text{Masks}(z'_M, c) = z'_M c^T
$$

결과는 $\mathbb{R}^{N \times K}$ 크기의 patch-level class mask이다. 즉, 각 patch가 각 class와 얼마나 잘 맞는지를 내적으로 계산한다. 이를 2D mask로 reshape하고 bilinear upsampling 후 softmax를 적용하면 최종 segmentation map이 된다.

이 방식의 직관은 다음과 같다. linear decoder는 “각 patch를 바로 class로 분류”하는 방식인데, mask transformer는 “각 class의 prototype 같은 embedding”을 만들고 patch와 class의 상호작용으로 mask를 생성한다. 저자들은 이것이 **dynamic filter**처럼 동작한다고 설명한다. 또한 이 구조는 semantic segmentation에만 국한되지 않고, class embedding을 object embedding으로 바꾸면 panoptic segmentation으로도 확장 가능하다고 주장한다.

### 학습 목표와 추론

학습은 end-to-end로 수행되며, 손실 함수는 **pixel-wise cross-entropy loss**이다. 논문에는 별도의 class rebalancing은 사용하지 않았다고 명시되어 있다. 추론 시에는 upsample된 pixel-level score에 대해 argmax를 취해 각 pixel의 최종 class를 결정한다.

학습 스케줄은 데이터셋마다 다르지만, 공통적으로 SGD optimizer와 poly learning rate decay를 사용한다.

$$
\gamma = \gamma_0 \left(1 - \frac{N_{\text{iter}}}{N_{\text{total}}}\right)^{0.9}
$$

여기서 $\gamma_0$는 초기 learning rate이고, $N_{\text{iter}}$는 현재 iteration, $N_{\text{total}}$는 전체 iteration 수다.

### 구현 및 설정상 중요한 점

이 논문은 encoder backbone으로 ViT-Tiny, Small, Base, Large와 DeiT-B를 사용한다. patch size는 $8 \times 8$, $16 \times 16$, $32 \times 32$를 비교한다. 모델 이름 예를 들어 `Seg-B/16`은 Base backbone에 patch size 16을 뜻한다.

중요한 실용적 포인트는 **ImageNet pretraining**이다. 저자들은 ViT를 ImageNet-21k 혹은 DeiT를 ImageNet-1k에서 사전학습한 뒤 segmentation으로 fine-tuning한다. positional embedding은 입력 해상도가 달라질 때 bilinear interpolation으로 조정한다.

데이터 증강은 random resize, random flip, crop/pad를 사용한다. 손실은 표준 cross-entropy이고, 대형 모델은 더 큰 입력 해상도로 학습했다. inference에서는 sliding-window와 multi-scale inference도 사용한다.

## 4. 실험 및 결과

### 데이터셋과 평가 지표

실험은 세 가지 대표 semantic segmentation benchmark에서 수행된다.

ADE20K는 150개 semantic class를 가진 매우 어려운 장면 파싱 데이터셋이다. 학습 20,210장, 검증 2,000장, 테스트 3,352장으로 구성된다.

Pascal Context는 59개 semantic class와 background를 포함하며, 학습 4,996장, 검증 5,104장이다.

Cityscapes는 도시 주행 장면 데이터셋으로 19개 semantic class를 가지며, 학습 2,975장, 검증 500장, 테스트 1,525장이다.

평가 지표는 모든 클래스에 대한 평균 Intersection over Union, 즉 **mIoU**이다.

### Ablation 1: Regularization

Seg-S/16 on ADE20K validation에서 dropout과 stochastic depth를 비교했다. 결과는 stochastic depth 0.1이 가장 좋았고, dropout은 일관되게 성능을 떨어뜨렸다. 예를 들어 dropout 없이 stochastic depth 0.1을 쓰면 mIoU 45.37%였고, 아무 regularization이 없으면 45.01%였다. 반면 dropout 0.1이나 0.2는 성능이 크게 감소했다.

이 결과는 segmentation에서도 transformer regularization이 CNN과 다르게 작동할 수 있음을 보여준다. 저자들은 이후 모든 모델에서 dropout 없이 stochastic depth 0.1을 사용한다.

### Ablation 2: Model Size

patch size 16으로 고정했을 때, backbone이 커질수록 성능이 좋아졌다. ADE20K validation single-scale 기준으로 Seg-S/16은 45.37%, Seg-B/16은 48.06%, Seg-L/16은 50.71%를 기록했다. token dimension 증가와 layer 수 증가 모두 성능 향상에 기여했다.

이는 segmentation에서도 더 큰 transformer가 더 표현력이 좋다는 점을 보여준다. 저자들은 이런 결과를 통해 task-specific convolution layer 없이도 transformer가 segmentation에 잘 맞는 표현을 학습한다고 해석한다.

### Ablation 3: Patch Size

이 논문에서 가장 중요한 실험 중 하나다. 같은 backbone이라도 patch size가 작아질수록 성능이 좋아졌다. 예를 들어 Seg-B는 patch size 32에서 43.07%, 16에서 48.06%, 8에서 49.54%였다. patch size를 32에서 16으로 줄이면 무려 약 5%p 개선된다.

이유는 segmentation이 경계와 작은 객체를 잘 다뤄야 하기 때문이다. patch가 크면 한 patch 내부에 여러 semantic region이 섞이기 쉬워지고, 결과적으로 경계가 뭉개진다. Figure 3 설명에서도 patch size 32에서는 사람 둘이 하나의 blob처럼 예측되지만, patch size를 줄이면 경계가 훨씬 선명해진다고 분석한다. 가느다란 streetlight pole 같은 작은 구조는 patch size 8에서만 제대로 잡힌다고 서술한다.

다만 patch가 작아질수록 시퀀스 길이가 길어져 self-attention 비용이 커진다. 예를 들어 Seg-B/8은 49.54%로 강하지만 초당 7장으로 매우 느리다. 따라서 patch size는 단순한 하이퍼파라미터가 아니라 segmentation 정확도와 계산량을 좌우하는 핵심 설계 요소다.

### Ablation 4: Decoder Variants

mask transformer는 linear decoder보다 일관되게 좋았다. ADE20K validation에서 Seg-B†/16은 linear 47.10%, mask 48.70%로 1.6%p 향상되었고, Seg-L/16은 linear 50.71%, mask 51.30%로 0.6%p 향상되었다. 저자들은 특히 큰 객체와 작은 객체 모두에서 개선이 나타나며, 대체로 큰 객체에서 이득이 더 크다고 보고한다.

저자들의 해석은 class embedding과 patch embedding을 함께 처리하는 구조가 단순 class 분류보다 더 유연한 mask 생성을 가능하게 하기 때문이라는 것이다. 또한 appendix에서는 학습된 class embedding을 2D로 투영했을 때, transportation, household object, outdoor category 등 의미적으로 비슷한 클래스가 군집을 이룬다고 보여준다.

### Ablation 5: Transformer vs FCN

DeepLabv3+ ResNeSt backbone과 비교했을 때, Segmenter는 특히 **큰 객체(large instances)**에서 강한 성능을 보였다. ADE20K validation에서 object size별 mIoU를 비교한 Table 4에 따르면, Seg-L-Mask/16은 large object에서 57.06%를 기록했고, DeepLab ResNeSt-101은 50.67%였다. 이는 약 6.39%p 차이이다.

반면 작은 객체나 인접한 사람 경계처럼 세밀한 boundary localization에서는 DeepLabv3+가 더 날카로운 경우도 있었다. qualitative result에서도 저자들은 DeepLabv3+가 object boundary는 더 sharp한 경향이 있고, Segmenter는 large instance에서 label consistency가 더 좋고 partial occlusion을 더 잘 다룬다고 적는다.

### Ablation 6: Dataset Size와 Pretraining

transformer가 segmentation에서도 데이터 의존성이 크다는 점도 명확히 보여준다. Seg-S/16을 ADE20K 부분집합으로 학습했을 때, 4k 이미지에서는 38.31%, 8k에서는 41.87%, 전체 20k에서는 45.37%였다. 8k 이하로 내려가면 성능 저하가 특히 크다.

appendix의 pretraining 실험은 더 강력하다. Seg-S/16을 랜덤 초기화로 학습하면 SGD 기준 12.51%, AdamW 기준 4.42%밖에 나오지 않는다. 반면 ImageNet-21k pretrained 모델을 SGD로 fine-tuning하면 45.37%를 얻는다. 즉, 이 논문이 제시하는 강한 성능은 **대규모 사전학습이 사실상 필수**임을 보여준다. 이 점은 논문의 중요한 전제다.

### SOTA 비교

ADE20K validation multi-scale 기준으로, Seg-L-Mask/16은 **53.63% mIoU**를 기록했다. 이는 논문 표 기준 DeepLabv3+ ResNeSt-200의 48.36%, SETR-L MLA의 50.28%, Swin-L UperNet의 53.50%보다 높다. 저자들은 이를 SOTA라고 주장한다.

Pascal Context validation에서는 Seg-L-Mask/16이 **59.0% mIoU**를 기록했다. 이는 OCR의 56.2%, SETR-L MLA의 55.8%를 넘는다. linear decoder만 사용한 Seg-L/16도 56.5%로 이미 매우 강하다.

Cityscapes validation에서는 Seg-L-Mask/16이 **81.3% mIoU**를 기록했다. 이는 CCNet 81.3%와 같은 수준이며, DeepLabv3+ ResNeSt-200 82.7%보다는 낮다. 따라서 이 데이터셋에서는 “압도적 SOTA”라기보다 **competitive**하다고 보는 것이 논문 서술과 일치한다. 또한 이 실험에서는 메모리 한계 때문에 Seg-L-Mask/16의 decoder layer를 2개가 아니라 1개만 사용했다고 명시한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 semantic segmentation에서 transformer가 왜 유리한지를 단순한 주장에 그치지 않고, 구조와 실험으로 설득력 있게 보여준 점이다. 전역 문맥을 첫 layer부터 활용할 수 있다는 transformer의 장점이 segmentation에 실제로 중요한지, 그리고 그것이 large object와 scene consistency 개선으로 이어지는지를 다양한 표와 qualitative 결과로 입증한다.

또한 모델 설계가 비교적 단순하다. encoder는 ViT, decoder는 linear 또는 mask transformer라는 명확한 구조를 가지며, 불필요하게 복잡한 multi-branch convolution 모듈이나 hand-crafted context module 없이도 강한 성능을 낸다. 특히 linear decoder만으로도 이미 상당히 강하다는 점은 encoder 표현력 자체가 높다는 것을 보여준다.

mask transformer decoder도 의미 있는 기여다. class embedding을 통해 class별 mask를 생성하는 구조는 segmentation을 단순 patch classification이 아니라 class-mask matching 문제로 다시 보게 한다. 저자들이 말하듯 이 아이디어는 semantic segmentation을 넘어 instance/panoptic segmentation으로도 확장될 가능성이 있다.

하지만 한계도 분명하다.

첫째, **계산량과 메모리 비용**이 크다. transformer self-attention은 시퀀스 길이에 대해 quadratic cost를 갖기 때문에, patch size를 줄여 더 좋은 성능을 얻으려면 계산량이 급증한다. Seg-B/8이 좋은 성능을 내지만 초당 7장이라는 결과는 이 한계를 잘 보여준다. 저자들도 더 작은 patch를 처리할 수 있는 효율적 transformer가 유망한 방향이라고 적고 있다.

둘째, **대규모 pretraining 의존성**이 강하다. scratch 학습 성능이 매우 낮았기 때문에, 이 접근은 사실상 ImageNet-scale pretraining을 전제로 한다고 봐야 한다. 이는 데이터와 계산 자원이 제한된 환경에서는 적용 장벽이 될 수 있다.

셋째, **boundary localization은 CNN이 더 나은 경우가 있다**. qualitative 비교에서 DeepLabv3+가 사람 간 경계나 세밀한 윤곽을 더 잘 분리하는 경우가 나타난다. 즉, global context에는 강하지만 local detail 복원은 상대적으로 약할 수 있다. patch representation과 bilinear upsampling을 사용하는 구조상 자연스러운 한계로 볼 수 있다.

넷째, Cityscapes에서는 최고 성능이 아니었다. ADE20K와 Pascal Context에서는 매우 강력하지만, Cityscapes에서는 competitive 수준이다. 따라서 이 논문의 장점이 모든 segmentation benchmark에서 동일하게 압도적이라고 말할 수는 없다.

다섯째, 논문은 mask transformer의 개선 이유를 dynamic filter와 semantic class embedding 관점에서 설명하지만, 그 내부 동작이 정확히 어떤 조건에서 가장 이득을 주는지까지는 충분히 해명하지 않는다. 예를 들어 어떤 클래스 분포, 객체 크기 분포, 장면 구조에서 더 유리한지는 부분적으로만 드러난다. 이는 향후 분석 여지가 있는 부분이다.

## 6. 결론

이 논문은 semantic segmentation을 위한 **pure transformer encoder-decoder 모델**인 Segmenter를 제안하고, ViT 기반 patch representation이 segmentation에서도 매우 강력하다는 점을 보여준다. 핵심은 patch token들이 self-attention을 통해 전역 문맥을 처음부터 지속적으로 반영한다는 것이고, 단순 linear decoder만으로도 높은 성능을 내며, class embedding 기반의 mask transformer decoder를 사용하면 추가 개선이 가능하다는 점이다.

실험적으로는 ADE20K와 Pascal Context에서 매우 강한 결과를 달성했고, Cityscapes에서도 경쟁력 있는 성능을 보였다. 또한 모델 크기, patch 크기, regularization, decoder 설계, pretraining의 영향까지 폭넓게 분석해 이후 연구가 참고할 수 있는 실증적 가이드를 제공한다.

실제 적용 측면에서 보면, Segmenter는 scene-level consistency가 중요한 segmentation 문제에서 특히 가치가 크다. 반면 작은 patch를 사용할수록 계산량이 급격히 증가하고, 대규모 사전학습이 거의 필수라는 점은 현실적 제약이다. 그럼에도 이 연구는 semantic segmentation, instance segmentation, panoptic segmentation을 하나의 transformer 프레임으로 통합하려는 방향의 초기이자 중요한 발판으로 볼 수 있다.
