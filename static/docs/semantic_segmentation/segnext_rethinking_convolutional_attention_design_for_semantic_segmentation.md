# SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation

- **저자**: Meng-Hao Guo, Cheng-Ze Lu, Qibin Hou, Zheng-Ning Liu, Ming-Ming Cheng, Shi-Min Hu
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2209.08575

## 1. 논문 개요

이 논문은 semantic segmentation을 위해 설계된 새로운 CNN 기반 아키텍처인 **SegNeXt**를 제안한다. 당시 semantic segmentation에서는 transformer 기반 모델이 self-attention을 통해 공간적 문맥 정보를 잘 다룬다는 이유로 강세를 보이고 있었는데, 저자들은 이 흐름을 그대로 따르지 않고 “정말 self-attention이 가장 효율적인가?”라는 질문을 던진다. 그 결과, segmentation에 중요한 특성을 다시 정리해 보면, 반드시 transformer일 필요는 없고, 오히려 잘 설계된 **convolutional attention**이 더 효율적이고 더 효과적일 수 있다고 주장한다.

논문이 다루는 핵심 연구 문제는 다음과 같다. semantic segmentation 모델은 픽셀 단위 예측을 해야 하므로, 단순한 이미지 분류보다 훨씬 강한 공간 정보 처리 능력, 다양한 크기의 객체를 다룰 수 있는 multi-scale 처리, 중요한 영역을 강조하는 spatial attention, 그리고 고해상도 입력을 감당할 수 있는 낮은 계산 복잡도가 필요하다. 기존 transformer 기반 모델은 문맥 포착 능력은 강하지만, 입력 해상도가 커질수록 self-attention의 계산량이 커지는 문제가 있다. 특히 Cityscapes 같은 고해상도 데이터에서는 이 비용이 매우 커진다.

이 문제의 중요성은 분명하다. semantic segmentation은 자율주행, 원격탐사, 로보틱스, 의료영상 등 실제 응용에서 고해상도 이미지와 세밀한 경계 처리가 중요한 경우가 많다. 따라서 단순히 정확도만 높은 것이 아니라, **정확도와 계산 효율의 균형**이 매우 중요하다. 이 논문은 바로 այդ 지점을 겨냥해, segmentation에 맞춘 CNN 설계가 transformer를 다시 앞설 수 있음을 실험적으로 보여준다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 semantic segmentation에서 중요한 것은 “transformer라는 형식”이 아니라, **강한 encoder, multi-scale interaction, spatial attention, low complexity**라는 성질 자체라는 점이다. 저자들은 기존 성공적인 segmentation 모델들을 비교해, 좋은 모델이 공통적으로 가지는 특성을 정리한다. 그리고 이 특성들을 만족시키는 새로운 CNN encoder-decoder 구조를 설계한다.

가장 핵심적인 설계는 encoder 안의 **MSCA (Multi-Scale Convolutional Attention)** 이다. 이 모듈은 여러 크기의 depth-wise convolution branch를 사용해 서로 다른 receptive field에서 문맥 정보를 추출하고, 이를 attention map처럼 사용해 입력 feature를 다시 가중한다. 즉, self-attention처럼 모든 위치 쌍을 직접 비교하지 않고도, convolution만으로 공간적 중요도를 학습하도록 만든다.

기존 접근과의 차별점은 두 가지로 요약할 수 있다. 첫째, SegFormer 같은 transformer 기반 segmentation 모델은 강한 transformer encoder와 비교적 단순한 decoder를 쓰는 반면, 이 논문은 그 반대로 **convolution 기반 encoder 자체를 매우 강하게 설계**한다. 둘째, 단순히 큰 kernel convolution 하나를 쓰는 것이 아니라, **여러 스케일의 convolution 출력을 attention으로 결합**해 segmentation에 필요한 multi-scale context를 더 직접적으로 반영한다. 저자들은 특히 segmentation에서는 이미지 분류와 달리 작은 물체, 가늘고 긴 구조물, 경계 세부 정보가 중요하기 때문에 multi-scale feature aggregation이 매우 중요하다고 본다.

또 하나의 중요한 포인트는 decoder 설계다. 저자들은 encoder에서 충분히 강한 표현을 만들 수 있다면, decoder는 무겁지 않아도 된다고 보고, 마지막 3개 stage의 feature만 모아서 **Hamburger (Ham)** 모듈로 전역 문맥을 보강하는 가벼운 decoder를 사용한다. 이를 통해 local-to-global 문맥을 효율적으로 결합한다.

## 3. 상세 방법 설명

SegNeXt는 전형적인 **encoder-decoder 구조**를 따른다. encoder는 **MSCAN (Multi-Scale Convolutional Attention Network)**, decoder는 다중 stage feature를 모은 뒤 Hamburger 모듈을 거치는 가벼운 구조다.

### Convolutional Encoder: MSCAN

MSCAN은 4-stage의 계층적 피라미드 구조를 가진다. 각 stage의 출력 해상도는 각각 $H/4 \times W/4$, $H/8 \times W/8$, $H/16 \times W/16$, $H/32 \times W/32$이다. 각 stage는 stride 2의 $3 \times 3$ convolution과 Batch Normalization으로 downsampling을 수행한 뒤, 여러 개의 building block을 쌓는다.

이 building block은 transformer block과 겉모양은 비슷하지만, self-attention 대신 **MSCA**를 사용한다. MSCA는 다음 세 부분으로 이루어진다.

1. 입력 feature의 local information을 모으는 **depth-wise convolution**
2. 여러 branch의 **multi-scale depth-wise strip convolution**
3. 채널 간 관계를 섞는 **$1 \times 1$ convolution**

여기서 중요한 점은 multi-scale branch의 출력들을 합친 후, $1 \times 1$ convolution을 통과시켜 attention map을 만들고, 이를 입력 feature에 element-wise multiplication으로 곱한다는 것이다. 논문은 이를 다음과 같이 쓴다.

$$
Att = Conv_{1\times1}\left(\sum_{i=0}^{3} Scale_i(DW\text{-}Conv(F))\right)
$$

$$
Out = Att \otimes F
$$

여기서 $F$는 입력 feature이고, $Att$는 attention map, $Out$은 출력 feature이다. $\otimes$는 element-wise multiplication이다. $DW\text{-}Conv$는 depth-wise convolution이고, $Scale_i$는 서로 다른 branch를 의미한다. $Scale_0$는 identity connection이다.

이 식의 의미를 쉽게 말하면, 입력 feature $F$에서 먼저 depth-wise convolution으로 기본적인 지역 정보를 모은 뒤, 서로 다른 스케일의 branch들이 각기 다른 receptive field에서 문맥을 뽑아낸다. 이들을 합쳐서 “어디를 더 강조할지”를 나타내는 attention map $Att$를 만들고, 그 attention map으로 원래 feature $F$를 다시 가중한다. 즉, convolution 결과를 단순 feature로 쓰는 것이 아니라, **attention weight**로 사용한다는 점이 핵심이다.

### 왜 strip convolution을 쓰는가

각 branch는 큰 정사각 kernel 대신, 예를 들어 $7 \times 7$ convolution을 직접 쓰지 않고, $1 \times 7$과 $7 \times 1$의 두 개 **depth-wise strip convolution**으로 근사한다. 논문은 kernel 크기로 7, 11, 21을 사용한다. 이렇게 하면 계산량이 줄어들고, 사람이나 전봇대처럼 길쭉한 구조를 더 잘 포착할 수 있다고 설명한다. 즉, strip convolution은 계산 효율성과 형상 적합성 두 측면에서 장점이 있다.

### Batch Normalization 사용

MSCAN의 각 block에서는 Layer Normalization 대신 **Batch Normalization**을 사용한다. 저자들은 segmentation 성능 측면에서 BatchNorm이 더 유리하다고 관찰했다고 밝힌다. 이 부분은 transformer 계열과 구별되는 구현상의 중요한 차이다.

### 모델 크기 변형

논문은 SegNeXt-T, SegNeXt-S, SegNeXt-B, SegNeXt-L 네 가지 크기를 제시한다. 가장 작은 T는 약 4.3M parameter, 가장 큰 L은 약 48.9M parameter를 가진다. stage별 채널 수와 block 개수는 모델 크기에 따라 달라지며, decoder 차원도 T/S는 256, B는 512, L은 1024로 커진다.

### Decoder

decoder는 세 가지 후보 구조를 비교한 뒤 최종 구조를 선택했다. SegFormer식 MLP decoder, CNN 계열에서 흔한 heavy decoder head 방식, 그리고 논문이 제안한 **마지막 3개 stage feature만 모아 처리하는 decoder**를 비교했다. 최종적으로 저자들은 stage 2, 3, 4의 출력을 모아서 concat한 뒤 MLP를 거치고, 그 위에 **Hamburger (Ham)** 모듈을 올리는 구조를 사용한다.

여기서 stage 1 feature를 일부러 제외한 이유도 분명히 설명한다. SegNeXt는 convolution 기반이라 stage 1에는 너무 많은 low-level information이 남아 있고, 이를 decoder에 넣으면 오히려 성능이 떨어지며 계산량만 크게 늘어난다고 본다. 즉, segmentation에서 항상 더 많은 해상도 정보를 넣는 것이 좋은 것이 아니라, **유용한 수준의 semantic abstraction이 있는 multi-level feature만 선택적으로 쓰는 것이 낫다**는 실험적 결론을 제시한다.

### 학습 절차

encoder는 모두 **ImageNet-1K pretrained**를 사용한다. 분류 pretraining은 DeiT와 같은 설정을 따른다. segmentation 학습에서는 random horizontal flip, random scaling (0.5~2), random crop을 사용하고, optimizer는 **AdamW**, 초기 learning rate는 0.00006, learning rate decay는 **poly policy**를 사용한다. ADE20K, Cityscapes, iSAID는 160K iteration, COCO-Stuff, Pascal VOC, Pascal Context는 80K iteration 학습한다. 평가는 single-scale과 multi-scale flip test 두 가지를 모두 사용한다.

## 4. 실험 및 결과

논문은 매우 폭넓은 벤치마크에서 성능을 검증한다. 사용된 데이터셋은 ImageNet-1K, ADE20K, Cityscapes, Pascal VOC, Pascal Context, COCO-Stuff, iSAID 총 7개다. classification 성능은 Top-1 accuracy, segmentation 성능은 mIoU로 측정한다.

### Encoder 자체 성능: ImageNet

Segmentation encoder로 쓰기 전에 MSCAN 자체를 ImageNet 분류에서 평가한 결과, MSCAN은 MiT, Swin Transformer, ConvNeXt, VAN과 비교해 경쟁력 있는 성능을 보인다. 예를 들어 MSCAN-B는 26.8M parameter로 **83.0% Top-1 accuracy**를 기록해, MiT-B2의 81.6%, Swin-T의 81.3%, ConvNeXt-T의 82.1%, VAN-Base의 82.8%보다 높다. 이 결과는 encoder 자체의 표현력이 충분히 강하다는 근거로 제시된다.

### ADE20K, Cityscapes, COCO-Stuff

주요 비교에서 SegNeXt는 transformer 계열을 전반적으로 앞선다. ADE20K에서 SegNeXt-T는 4.3M parameter와 6.6 GFLOPs로 **41.1 / 42.2 mIoU (SS/MS)** 를 기록해 SegFormer-B0의 37.4 / 38.0보다 크게 높다. SegNeXt-S도 같은 15.9 GFLOPs 수준에서 SegFormer-B1과 HRFormer-S보다 높다. 특히 SegNeXt-B는 27.6M parameter, 34.9 GFLOPs로 **48.5 / 49.9**를 기록해 SegFormer-B2의 46.5 / 47.5보다 약 2.0 mIoU 높고 계산량도 훨씬 적다.

가장 큰 모델인 SegNeXt-L은 ADE20K에서 **51.0 / 52.1 mIoU**를 달성한다. 이는 표에서 Mask2Former (Swin-T backbone)의 47.7 / 49.6보다 높고, HRFormer-B의 48.7 / 50.0보다도 높다. 논문은 특히 “유사하거나 더 적은 계산량으로 더 높은 성능”이라는 점을 강조한다.

Cityscapes에서는 고해상도 입력 때문에 계산 효율 차이가 더 극적으로 드러난다. 예를 들어 SegNeXt-S는 124.6 GFLOPs에서 **81.3 / 82.7 mIoU**를 기록하는데, SegFormer-B1은 243.7 GFLOPs에서 78.5 / 80.0이다. SegNeXt-B는 275.7 GFLOPs로 **82.6 / 83.8**을 기록하며, SegFormer-B2의 81.0 / 82.2보다 높다. 저자들은 self-attention의 quadratic complexity가 고해상도 환경에서 불리하다는 점을 이 결과로 뒷받침한다.

COCO-Stuff에서도 SegNeXt는 우수하다. SegNeXt-L은 **46.5 / 47.2 mIoU**를 기록하며, 같은 표의 다른 강한 baseline들보다 높은 수치를 보인다.

### Pascal VOC, Pascal Context, iSAID

Pascal VOC에서는 SegNeXt-L이 COCO pretraining을 사용한 설정에서 **90.6 mIoU**를 기록한다. 논문은 이 수치가 EfficientNet-L2 with NAS-FPN의 90.5를 넘으며, 그 모델이 485M parameter인 반면 SegNeXt-L은 약 48.7M parameter라는 점을 강조한다. 즉, 파라미터 수가 약 1/10 수준인데도 더 높은 성능을 냈다는 것이다.

Pascal Context에서는 SegNeXt-L이 **58.7 / 60.3**, ADE20K pretraining을 추가하면 **59.2 / 60.9**를 기록한다. 이는 HRNet(OCR), HamNet, HRFormer-B, DPT-Hybrid 등 강한 CNN/transformer baseline들과 비교해 매우 경쟁력 있거나 더 높은 수준이다.

원격탐사 데이터셋 iSAID에서는 SegNeXt-T가 이미 **68.3 mIoU**를 달성하며, SegNeXt-L은 **70.3**까지 올라간다. 기존 ResNet50 기반 방법들, Swin-T 기반 UperNet 등을 모두 앞선다. 이는 논문이 강조한 “고해상도 및 복잡한 장면에서 계산 효율이 중요한 환경”에서 SegNeXt의 장점이 잘 드러나는 사례다.

### 실시간 성능

논문은 정확도뿐 아니라 real-time 측면도 평가한다. Cityscapes test set에서 SegNeXt-T는 입력 크기 $768 \times 1536$에서 **78.0 mIoU**를 기록하며, 단일 RTX 3090 GPU 기준 **25 FPS**를 달성했다고 보고한다. 논문은 별도 최적화 없이도 real-time deployment 요구를 만족한다고 주장한다. 이는 lightweight model이 실용성까지 갖췄다는 점을 보여준다.

### Ablation Study

이 논문의 ablation은 설계 타당성을 보여주는 핵심 근거다.

첫째, **MSCA 구성요소별 ablation**에서 7x7, 11x11, 21x21 branch, $1 \times 1$ conv, attention 연산 각각이 성능에 기여한다. 모든 요소를 포함한 경우 ImageNet Top-1은 75.9, ADE20K mIoU는 41.1로 가장 높다. 특히 attention 연산을 넣었을 때 성능이 분명히 좋아진다.

둘째, **decoder의 global context 모듈** 비교에서 CCNet, EMA, Non-local, Ham을 테스트했는데, Ham이 성능과 계산량의 균형이 가장 좋았다. SegNeXt-B with Ham은 27.6M parameter, 34.9 GFLOPs에서 48.5 / 49.9 mIoU를 기록한다.

셋째, **decoder 구조 비교**에서 논문이 제안한 구조가 가장 좋았다. SegNeXt-T 기준으로 (c) 구조는 6.6 GFLOPs에서 41.1 / 42.2를 기록했고, stage 1까지 추가하면 FLOPs는 12.1로 크게 늘지만 성능은 오히려 좋아지지 않았다. 이는 stage 1 feature 제외 결정의 근거가 된다.

넷째, **MSCA의 중요성**을 보기 위해 multi-branch 구조를 없애고 단일 large-kernel branch만 남긴 실험을 수행했다. SegNeXt-T에서는 MSCA 없이 39.5 / 40.9, MSCA 포함 시 41.0 / 42.5였고, SegNeXt-S도 43.5 / 45.2에서 44.3 / 45.8로 개선되었다. 즉, segmentation에서는 단순 large-kernel attention보다 **multi-scale aggregation**이 더 중요하다는 결론이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정이 매우 명확하다는 점이다. 저자들은 단순히 새로운 블록을 제안하는 데 그치지 않고, 먼저 좋은 segmentation 모델의 조건을 정리한 뒤, 그 조건에 맞춰 구조를 설계한다. 따라서 방법론의 설계 논리가 비교적 일관되고 설득력이 있다.

또 다른 강점은 **효율성과 성능을 동시에 입증했다**는 점이다. 많은 논문이 최고 성능만 강조하거나, 반대로 경량성만 강조하는데, SegNeXt는 두 축을 함께 보여준다. 특히 Cityscapes처럼 고해상도 환경에서 FLOPs 대비 성능 이점이 분명하다. 이는 self-attention의 quadratic complexity가 실제 dense prediction에서 부담이 된다는 점을 정면으로 보여주는 결과다.

방법 자체도 비교적 단순하고 구현 친화적이다. MSCA는 복잡한 token mixing이나 sparse attention 같은 까다로운 기법이 아니라, depth-wise convolution, strip convolution, $1 \times 1$ convolution, element-wise multiplication으로 구성된다. 따라서 기존 CNN 인프라 위에서 적용하기 쉽고, 산업적 활용 가능성도 높다.

실험 범위가 넓다는 점도 강점이다. ADE20K, Cityscapes, COCO-Stuff, Pascal VOC, Pascal Context, iSAID, ImageNet까지 다루면서 classification backbone으로서의 성능과 segmentation 모델로서의 성능을 모두 보였다. Ablation도 핵심 설계를 검증하는 데 충분히 사용되었다.

한편 한계도 있다. 논문 스스로 인정하듯, 이 방법이 **100M+ 규모의 더 큰 모델**로 확장될 때도 동일한 장점을 유지하는지는 검증되지 않았다. 또한 이 구조가 semantic segmentation 외의 다른 dense prediction task나, 더 나아가 NLP 같은 다른 도메인에서도 잘 작동하는지는 다루지 않는다.

비판적으로 보면, 논문은 convolutional attention이 self-attention보다 효율적이라고 주장하지만, 이는 주로 semantic segmentation benchmark에 한정된 실험 결과를 근거로 한다. 따라서 이를 “일반적으로 더 낫다”라고 해석해서는 안 된다. 정확히는, **semantic segmentation에서 요구되는 multi-scale spatial context와 고해상도 효율성 측면에서는 매우 강력한 대안**이라고 보는 것이 타당하다.

또 하나의 제한은 decoder에 Hamburger 모듈이 들어간다는 점이다. 논문은 “거의 convolution으로 이루어진 구조”라고 말하지만, decoder는 완전히 순수 CNN은 아니다. 물론 이는 약점이라기보다는 pragmatic한 설계 선택에 가깝다. 다만 SegNeXt의 성능 향상이 encoder만의 효과인지, 아니면 decoder의 Ham 기여도 포함된 것인지는 해석 시 구분할 필요가 있다.

## 6. 결론

이 논문은 semantic segmentation에서 transformer의 우세가 절대적인 것이 아니며, segmentation에 맞춘 **적절한 convolutional attention 설계**만으로도 더 높은 정확도와 더 좋은 계산 효율을 얻을 수 있음을 보여준다. 핵심 기여는 multi-scale convolution feature를 attention weight로 활용하는 **MSCA**, 이를 기반으로 한 **MSCAN encoder**, 그리고 가벼운 decoder와 결합한 **SegNeXt 전체 구조**에 있다.

실험적으로는 ADE20K, Cityscapes, COCO-Stuff, Pascal VOC, Pascal Context, iSAID 등 다양한 벤치마크에서 당시 state-of-the-art 수준 혹은 그 이상을 달성했고, 특히 고해상도 환경에서 FLOPs 대비 우수한 성능을 입증했다. 이는 실제 응용 환경에서 매우 중요한 장점이다.

향후 연구 관점에서 이 논문은 두 가지 의미가 있다. 하나는 CNN이 더 이상 낡은 선택지가 아니라, 적절한 구조 설계를 통해 transformer와 경쟁하거나 앞설 수 있다는 점이다. 다른 하나는 semantic segmentation에서 중요한 것은 특정 유행 아키텍처가 아니라, **multi-scale context, spatial adaptivity, efficiency** 같은 task-specific design principle이라는 점을 분명히 했다는 것이다. 이런 의미에서 SegNeXt는 단순한 하나의 모델 제안이 아니라, segmentation backbone 설계 방향 자체를 다시 생각하게 만든 연구라고 평가할 수 있다.
