# SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

- **저자**: Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2105.15203

## 1. 논문 개요

이 논문은 semantic segmentation을 위해 Transformer encoder와 매우 가벼운 MLP decoder를 결합한 `SegFormer`를 제안한다. 저자들의 목표는 정확도만 높은 모델이 아니라, 실제로는 함께 고려되어야 하는 효율성, 계산량, 파라미터 수, 그리고 테스트 환경 변화에 대한 강건성까지 동시에 만족하는 segmentation framework를 만드는 것이다.

연구 문제는 명확하다. 기존 Transformer 기반 segmentation 방법, 특히 ViT 계열을 사용하는 방식은 semantic segmentation에 바로 쓰기에는 몇 가지 구조적 한계가 있었다. 대표적으로 단일 해상도의 저해상도 feature만 생성하는 점, 큰 해상도 입력에서 self-attention 비용이 매우 큰 점, 그리고 positional encoding이 학습 해상도와 테스트 해상도가 다를 때 성능 저하를 유발할 수 있다는 점이다. 반대로 CNN 기반 방법들은 효율적일 수는 있지만 전역 문맥을 충분히 포착하기 위해 ASPP 같은 추가 모듈을 붙이면서 구조가 무거워지는 경우가 많았다.

이 문제는 중요하다. Semantic segmentation은 scene parsing, autonomous driving 같은 downstream application의 기반이며, 특히 픽셀 단위 예측이 필요하므로 local detail과 global context를 동시에 잘 다뤄야 한다. 따라서 backbone과 decoder를 함께 재설계하여 더 단순하면서도 강력한 구조를 만드는 것은 학술적으로도, 실제 응용 측면에서도 가치가 크다.

## 2. 핵심 아이디어

SegFormer의 핵심 아이디어는 두 가지다. 첫째, positional encoding 없이도 동작하는 hierarchical Transformer encoder를 설계해 multi-scale feature를 직접 생성한다. 둘째, 복잡한 convolutional decoder나 context module 없이, 여러 단계의 feature를 단순한 MLP 기반 decoder로 합쳐도 충분히 강력한 segmentation representation을 만들 수 있다는 점을 보인다.

중심적인 직관은 Transformer의 각 stage가 서로 다른 범위의 attention을 자연스럽게 형성한다는 데 있다. 낮은 단계의 feature는 상대적으로 local attention을, 높은 단계의 feature는 non-local attention을 갖는다. 저자들은 이 성질을 이용해 여러 stage의 feature를 함께 모으면 CNN에서 별도 모듈로 억지로 만들던 local-global context 결합을 더 단순하게 구현할 수 있다고 본다.

기존 접근과의 차별점도 분명하다. SETR은 ViT backbone 위에 무거운 decoder를 붙여 segmentation을 수행하지만, SegFormer는 encoder 자체를 dense prediction에 맞게 hierarchical하게 바꾸고 decoder는 오히려 더 단순화했다. 또한 fixed positional embedding을 제거하여 테스트 해상도가 달라질 때 발생하는 positional code interpolation 문제를 피한다. 논문은 이 단순함이 단지 구현 편의가 아니라, 정확도와 효율성 모두에 기여하는 설계라고 주장한다.

## 3. 상세 방법 설명

전체 파이프라인은 다음과 같다. 입력 이미지 크기가 $H \times W \times 3$일 때, 먼저 $4 \times 4$ 단위의 더 작은 patch를 기반으로 hierarchical Transformer encoder에 넣는다. 이 encoder는 입력으로부터 해상도가 각각 원본의 $1/4$, $1/8$, $1/16$, $1/32$인 네 단계의 multi-level feature를 생성한다. 이후 All-MLP decoder가 이 feature들을 같은 해상도로 정렬하고 합쳐 segmentation mask를 예측한다.

### Hierarchical Transformer Encoder

저자들은 encoder를 `Mix Transformer (MiT)`라고 부르며, B0부터 B5까지 같은 구조를 크기만 다르게 확장한다. 이 encoder의 핵심은 세 가지다.

첫째는 **hierarchical feature representation**이다. ViT는 단일 해상도의 feature map만 생성하지만, MiT는 CNN처럼 여러 해상도의 feature map을 만든다. 각 stage의 출력 feature를 $F_i$라고 하면, 해상도는 대략 $H / 2^{i+1} \times W / 2^{i+1}$이고 채널 수 $C_i$는 stage가 깊어질수록 증가한다. segmentation에서는 경계와 질감 같은 세부 정보도 필요하고, 동시에 넓은 문맥도 필요하므로 이런 multi-scale 구조가 중요하다.

둘째는 **overlapped patch merging**이다. 일반적인 ViT류 patch embedding은 non-overlapping patch를 사용해 지역적 연속성을 잘 보존하지 못할 수 있다. 이를 해결하기 위해 SegFormer는 overlap이 있는 patch merging을 쓴다. 첫 stage에서는 $K=7$, $S=4$, $P=3$을, 이후 stage에서는 $K=3$, $S=2$, $P=1$을 사용한다. 여기서 $K$는 kernel 혹은 patch size, $S$는 stride, $P$는 padding이다. 이렇게 하면 출력 해상도는 유지하면서도 인접 지역 간 연속성을 더 잘 반영할 수 있다.

셋째는 **efficient self-attention**이다. 일반 self-attention은 sequence length를 $N=H \times W$라고 하면 연산량이 $O(N^2)$이어서 고해상도 이미지에 비싸다. 원래 attention은

$$
\mathrm{Attention}(Q,K,V)=\mathrm{Softmax}\left(\frac{QK^T}{\sqrt{d_{\text{head}}}}\right)V
$$

로 계산된다. 여기서 $Q$, $K$, $V$는 query, key, value이고, $d_{\text{head}}$는 head 차원이다.

SegFormer는 PVT의 sequence reduction을 사용해 $K$의 길이를 줄인다. 논문 식은 다음과 같다.

$$
\hat{K}=\mathrm{Reshape}\left(\frac{N}{R}, C \cdot R\right)(K)
$$

$$
K=\mathrm{Linear}(C \cdot R, C)(\hat{K})
$$

즉, 원래 길이 $N$인 sequence를 reduction ratio $R$을 이용해 길이 $N/R$로 줄인 뒤 attention을 수행한다. 이렇게 하면 self-attention의 계산 복잡도는 $O(N^2)$에서 $O(N^2/R)$로 감소한다. 각 stage에서 $R$은 $[64,16,4,1]$로 설정되어, 앞단의 고해상도 feature에서는 강하게 줄이고 뒷단으로 갈수록 덜 줄인다.

### Mix-FFN과 Positional Encoding 제거

SegFormer의 또 다른 핵심은 positional encoding을 완전히 제거한 점이다. 기존 Transformer는 위치 정보를 넣기 위해 positional embedding을 사용하지만, 입력 해상도가 바뀌면 이를 보간해야 하고 이것이 성능 하락을 부를 수 있다. 저자들은 semantic segmentation에서는 positional encoding이 꼭 필요하지 않다고 주장하고, 대신 FFN 안에 $3 \times 3$ convolution을 삽입한 `Mix-FFN`을 제안한다.

식은 다음과 같다.

$$
x_{\text{out}}=\mathrm{MLP}(\mathrm{GELU}(\mathrm{Conv}_{3\times3}(\mathrm{MLP}(x_{\text{in}})))) + x_{\text{in}}
$$

여기서 $x_{\text{in}}$은 self-attention 뒤의 feature다. 구조적으로는 MLP 내부에 depth-wise $3 \times 3$ convolution을 넣어 local spatial bias를 주는 방식이다. 저자들은 zero padding이 위치 정보를 간접적으로 새어 나오게 만든다고 보고, 이 작은 convolution만으로도 positional information을 충분히 줄 수 있다고 설명한다. 이 설계 덕분에 test resolution이 training resolution과 달라도 positional embedding interpolation이 필요 없다.

### Lightweight All-MLP Decoder

Decoder는 매우 단순하다. 각 stage의 encoder feature $F_i$를 먼저 linear layer로 동일한 채널 수 $C$로 맞춘다. 그 다음 모든 feature를 $1/4$ 해상도로 upsample하고, 채널 방향으로 concatenate한다. 이어서 하나의 linear layer로 fusion한 뒤, 마지막 linear layer로 class 수 $N_{\text{cls}}$ 차원의 segmentation mask를 만든다.

논문 식은 다음과 같다.

$$
\hat{F}_i=\mathrm{Linear}(C_i, C)(F_i), \quad \forall i
$$

$$
\hat{F}_i=\mathrm{Upsample}\left(\frac{H}{4}, \frac{W}{4}\right)(\hat{F}_i), \quad \forall i
$$

$$
F=\mathrm{Linear}(4C, C)(\mathrm{Concat}(\hat{F}_i))
$$

$$
M=\mathrm{Linear}(C, N_{\text{cls}})(F)
$$

여기서 $M$은 최종 segmentation mask다. 핵심은 복잡한 ASPP나 여러 개의 convolution block 없이도, encoder가 이미 강한 표현을 가지고 있기 때문에 MLP만으로 충분하다는 주장이다.

### Effective Receptive Field 관점의 해석

저자들은 왜 이렇게 단순한 decoder가 잘 작동하는지를 ERF(Effective Receptive Field) 분석으로 설명한다. 분석에 따르면 DeepLabv3+ 같은 CNN 기반 모델은 깊은 stage에서도 ERF가 상대적으로 작다. 반면 SegFormer encoder는 낮은 stage에서는 convolution 비슷한 local attention을 보이고, 높은 stage에서는 훨씬 넓은 non-local attention을 형성한다. 또한 decoder head는 stage-4의 global context뿐 아니라 더 강한 local attention도 함께 갖게 된다. 따라서 여러 stage의 feature를 합치는 MLP decoder가 local detail과 global context를 동시에 살릴 수 있다는 것이 저자들의 해석이다.

## 4. 실험 및 결과

실험은 ADE20K, Cityscapes, COCO-Stuff 세 데이터셋에서 수행되었다. ADE20K는 150개 클래스의 scene parsing 데이터셋이고, Cityscapes는 19개 클래스의 urban driving segmentation 데이터셋이며, COCO-Stuff는 172개 label을 포함하는 대규모 segmentation 데이터셋이다. 구현은 `mmsegmentation` 기반이며, encoder는 ImageNet-1K로 pre-train하고 decoder는 random initialization을 사용했다. optimizer는 AdamW, 학습 스케줄은 poly LR이며, auxiliary loss, OHEM, class balance loss 같은 흔한 기법은 사용하지 않았다고 명시한다.

### Ablation Study

모델 크기 실험에서는 B0부터 B5까지 encoder 크기를 키울수록 세 데이터셋 모두에서 성능이 꾸준히 증가했다. 예를 들어 ADE20K에서 single-scale 기준 mIoU는 B0의 37.4에서 B5의 51.0까지 증가했고, multi-scale에서는 51.8까지 도달했다. Cityscapes에서는 B5가 validation set에서 84.0% mIoU를 기록했다. 흥미로운 점은 decoder가 매우 작다는 것이다. B0에서는 decoder가 0.4M 파라미터뿐이며, B5에서도 전체 파라미터의 4% 정도만 차지한다.

Decoder 내부 채널 차원 $C$에 대한 실험에서는, $C$를 키우면 성능은 약간 좋아지지만 FLOPs와 파라미터가 증가한다. ADE20K 기준 $C=256$일 때 44.9 mIoU, $C=768$일 때 45.4 mIoU였다. 768 이상에서는 성능 향상이 거의 포화되었다. 그래서 저자들은 실시간 모델 B0, B1에는 $C=256$을, 나머지에는 $C=768$을 사용했다.

Positional encoding과 Mix-FFN 비교 실험은 이 논문의 중요한 근거다. Cityscapes에서 positional encoding을 쓰면 inference resolution이 $768 \times 768$일 때 77.3 mIoU였고, $1024 \times 2048$로 바뀌면 74.0으로 떨어졌다. 반면 Mix-FFN을 쓰면 각각 80.5와 79.8이었다. 즉, 단순히 성능이 높을 뿐 아니라 해상도 변화에 대한 민감도도 훨씬 작았다. 저자들의 주장대로 positional embedding 제거가 robustness에 실질적으로 도움이 된다는 결과다.

CNN encoder에 MLP decoder를 붙인 실험도 중요하다. ResNet50, ResNet101, ResNeXt101에 같은 MLP decoder를 붙였을 때 ADE20K mIoU는 각각 34.7, 38.7, 39.8에 그쳤다. 반면 MiT-B2는 stage-4만 써도 43.1, 모든 stage를 쓰면 45.4를 기록했다. 이는 MLP decoder의 성공이 decoder 자체의 마법이라기보다, Transformer encoder가 이미 충분히 넓은 receptive field와 좋은 multi-scale 표현을 제공하기 때문임을 보여준다.

### State-of-the-Art 비교

ADE20K에서 SegFormer-B5는 51.8% mIoU를 달성했다. 이는 논문이 비교한 기존 최고 성능인 SETR의 50.2보다 1.6% 높고, 파라미터는 318.3M 대비 84.7M으로 훨씬 작다. FLOPs와 속도 측면에서도 유리하다. B4조차 51.1% mIoU로 매우 강력하며, 64.1M 파라미터 수준이다.

실시간 측면에서도 결과가 좋다. ADE20K에서 SegFormer-B0는 3.8M 파라미터, 8.4G FLOPs로 37.4% mIoU를 기록했다. 이는 MobileNetV2 기반 DeepLabV3+보다 더 높은 정확도를 보이면서도 더 가볍다. Cityscapes에서는 short side를 512로 줄였을 때 B0가 47.6 FPS와 71.9% mIoU를 달성해 ICNet보다 더 빠르고 정확했다.

Cityscapes validation에서 SegFormer-B5는 84.0% mIoU를 기록해 비교 대상들보다 우수했다. 논문은 SETR보다 1.8% 높고 5배 빠르며 4배 작다고 강조한다. Cityscapes test set에서도 ImageNet-1K pre-training만으로 82.2% mIoU를 얻었고, 추가로 Mapillary Vistas를 사용하면 83.1%까지 올라간다. 비교표에 따르면 이는 ImageNet-22K와 coarse data를 사용한 SETR보다도 높은 수치다.

COCO-Stuff에서도 SegFormer-B5는 46.7% mIoU를 기록했다. 이는 저자들이 재현한 SETR의 45.8보다 0.9% 높고, 파라미터 수는 훨씬 적다. 따라서 세 데이터셋 모두에서 성능-효율 균형이 뛰어나다는 결론이 가능하다.

### Robustness to Corruptions

논문은 Cityscapes-C를 만들어 natural corruption과 perturbation에 대한 zero-shot robustness도 평가했다. Gaussian noise, blur, weather, digital artifact 등 16종류 corruption에 대해 비교한 결과, SegFormer-B5는 clean 성능 82.4% mIoU뿐 아니라 corruption 상황에서도 기존 방법들보다 월등히 높았다. 예를 들어 Gaussian noise에서 기존 모델들이 한 자릿수 혹은 낮은 두 자릿수 mIoU를 보일 때 SegFormer-B5는 57.8을 기록했다. Snow에서도 40.7로 매우 큰 차이를 보인다. 논문은 이를 안전이 중요한 응용, 특히 autonomous driving에 유리한 특성으로 해석한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 구조적 단순성과 성능 향상을 동시에 달성했다는 점이다. 많은 segmentation 연구가 복잡한 context module이나 handcrafted component를 추가하면서 좋아지는 반면, SegFormer는 encoder와 decoder를 더 단순한 방향으로 재설계해 오히려 더 좋은 결과를 얻었다. 특히 positional encoding 제거, hierarchical Transformer, All-MLP decoder라는 세 설계가 서로 잘 맞물린다.

또 다른 강점은 실험 설득력이다. 단순히 최고 성능만 제시하지 않고, 모델 크기, decoder channel width, positional encoding 제거 효과, CNN encoder와의 비교, ERF 분석, corruption robustness까지 다양한 각도에서 설계를 검증한다. 특히 “왜 MLP decoder가 가능한가”에 대해 ERF와 stage-wise attention 성질을 통해 설명하려는 시도는 이 논문의 핵심 설계를 이해하는 데 도움이 된다.

실용성도 강점이다. B0부터 B5까지 폭넓은 스펙트럼을 제공해 real-time부터 high-performance까지 대응하며, ImageNet-22K가 아닌 ImageNet-1K pre-training만으로도 강한 결과를 낸다. 이는 대규모 사전학습 의존도를 낮춘다는 점에서 의미가 있다.

한계도 있다. 논문이 직접 인정하듯, 가장 작은 3.7M 모델조차 극단적으로 작은 edge device, 예를 들어 메모리가 100k 수준인 칩에서 잘 동작할지는 불분명하다. 즉, “가볍다”는 것이 모바일 혹은 임베디드 전체 범위를 포괄한다고 보기는 어렵다. 또한 robustness 결과는 강력하지만, 왜 Transformer 기반 설계가 이런 corruption에 더 강한지에 대한 원인 분석은 상대적으로 제한적이다. ERF와 attention 특성으로 어느 정도 해석은 하지만, robustness 메커니즘을 정량적으로 분해한 수준은 아니다.

비판적으로 보면, 논문은 decoder 단순화의 이점을 잘 보여주지만, decoder가 극단적으로 단순해도 충분한 이유가 결국 encoder 품질에 크게 의존한다. 다시 말해 이 성과는 “MLP decoder가 항상 좋다”라기보다 “강한 hierarchical Transformer encoder와 결합된 MLP decoder가 좋다”로 읽는 편이 정확하다. 실제로 CNN encoder와 결합했을 때 성능이 크게 떨어지는 결과가 이를 뒷받침한다.

## 6. 결론

이 논문은 semantic segmentation을 위한 간단하지만 강력한 Transformer 기반 프레임워크 `SegFormer`를 제안한다. 핵심 기여는 positional-encoding-free hierarchical Transformer encoder와 lightweight All-MLP decoder의 조합이며, 이를 통해 정확도, 효율성, 강건성을 동시에 개선했다는 점이다.

실험적으로 SegFormer는 ADE20K, Cityscapes, COCO-Stuff에서 매우 강한 성능을 보였고, 특히 Cityscapes-C에서 zero-shot robustness까지 입증했다. 따라서 이 연구는 단순히 새로운 backbone 하나를 제안한 것이 아니라, semantic segmentation에서 “복잡한 decoder가 꼭 필요한가”라는 기존 관행을 다시 생각하게 만든다. 향후 연구에서는 이 구조를 더 작은 edge 환경에 맞게 경량화하거나, 다른 dense prediction task로 확장하는 방향이 유망할 것으로 보인다.
