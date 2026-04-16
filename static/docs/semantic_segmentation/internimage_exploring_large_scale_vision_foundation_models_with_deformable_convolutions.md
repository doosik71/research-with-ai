# InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions

- **저자**: Wenhai Wang, Jifeng Dai, Zhe Chen, Zhenhang Huang, Zhiqi Li, Xizhou Zhu, Xiaowei Hu, Tong Lu, Lewei Lu, Hongsheng Li, Xiaogang Wang, Yu Qiao
- **발표연도**: 2023
- **arXiv**: https://arxiv.org/abs/2211.05778

## 1. 논문 개요

이 논문은 대규모 비전 foundation model이 반드시 Vision Transformer(ViT) 계열일 필요는 없으며, CNN도 적절한 연산자와 구조를 갖추면 대규모 파라미터와 대규모 데이터의 이점을 충분히 누릴 수 있다는 문제의식에서 출발한다. 저자들은 이를 위해 deformable convolution을 핵심 연산자로 사용하는 새로운 CNN backbone인 **InternImage**를 제안한다.

연구 문제는 비교적 명확하다. 최근 대규모 비전 모델의 성능 향상은 대부분 ViT 중심으로 이루어졌고, CNN은 상대적으로 “초기 단계”에 머물러 있다는 점이다. 저자들은 그 원인을 단순히 “CNN이라서 안 된다”로 보지 않고, 기존 CNN이 갖는 operator-level 한계와 architecture-level 한계를 함께 지적한다. operator 수준에서는 일반 convolution이 긴 거리 의존성(long-range dependencies)과 입력 조건에 따라 달라지는 adaptive spatial aggregation이 약하다는 점이 문제이며, architecture 수준에서는 LayerNorm, FFN, GELU 같은 현대적 구성요소가 transformer 쪽에서 더 체계적으로 활용되어 왔다는 점을 지적한다.

이 문제가 중요한 이유는 분명하다. 객체 검출, 인스턴스 분할, 의미 분할 같은 dense prediction task에서는 높은 해상도의 feature map을 다뤄야 하므로, global attention은 계산량과 메모리 사용량이 매우 커진다. 반면 CNN은 계산 효율 면에서 여전히 강점이 있다. 따라서 CNN이 ViT 수준의 표현력과 스케일링 특성을 확보할 수 있다면, 정확도와 효율을 동시에 만족하는 대규모 비전 backbone을 만들 가능성이 열린다. 이 논문은 바로 그 가능성을 deformable convolution 기반 설계로 실증하려는 시도라고 볼 수 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 **deformable convolution이 regular convolution과 self-attention 사이를 잇는 좋은 절충점**이라는 점이다. 저자들은 MHSA가 갖는 핵심 장점을 두 가지로 요약한다. 하나는 긴 거리 의존성을 포착할 수 있다는 점이고, 다른 하나는 입력에 따라 aggregation 방식이 달라지는 adaptive spatial aggregation을 수행한다는 점이다. 반면 일반적인 $3 \times 3$ convolution은 고정된 지역 윈도우와 정적인 가중치를 사용하므로 이러한 특성이 약하다.

기존 CNN 개선 방향 중 하나는 매우 큰 커널을 쓰는 large-kernel convolution이었다. 그러나 저자들은 이것이 receptive field를 넓히는 데는 도움을 주지만, 입력에 조건화된 adaptive aggregation을 제공하지는 못한다고 본다. 또한 큰 dense kernel은 최적화와 비용 측면의 부담도 크다. InternImage는 이와 달리 **작은 $3 \times 3$ 커널을 유지하면서도, offset을 학습해 실제 receptive field를 유연하게 확장하는 deformable convolution**을 택한다. 즉, 형태는 convolution이지만 동작은 더 유연하고 데이터 의존적이다.

이 논문의 차별점은 단순히 기존 DCN을 backbone에 넣은 것이 아니라, **DCNv2를 foundation model에 맞게 다시 설계해 DCNv3를 만들고**, 여기에 transformer 스타일의 block design, stacking rule, scaling rule을 결합해 “대규모 CNN”으로 체계화했다는 데 있다. 저자들은 이 조합 덕분에 CNN도 10억 개 이상의 파라미터와 수억 장의 학습 데이터로 확장 가능하다고 주장하며, ImageNet, COCO, ADE20K에서 이를 실험적으로 입증한다.

## 3. 상세 방법 설명

InternImage의 핵심은 **DCNv3를 중심 연산자로 삼은 계층형 4-stage backbone**이다. 전체 구조는 CNN과 transformer의 특징을 결합한 형태다. 입력 이미지는 stem에서 두 번의 stride 2 convolution을 거쳐 해상도가 4배 줄어든 뒤, 네 개의 stage를 순차적으로 통과한다. 각 stage 사이에는 downsampling layer가 들어가고, 각 stage 안에는 여러 개의 basic block이 쌓인다. 출력 feature는 분류, 검출, 분할 등 다양한 downstream task에 사용된다.

### 3.1 Deformable Convolution v2에서 v3로의 확장

논문은 먼저 DCNv2를 다음과 같이 제시한다.

$$
y(p_0) = \sum_{k=1}^{K} w_k m_k x(p_0 + p_k + \Delta p_k)
$$

여기서 $p_0$는 현재 위치, $p_k$는 정해진 grid 상의 sampling 위치, $\Delta p_k$는 그 위치에 대한 learned offset이다. $m_k$는 modulation scalar이고, $w_k$는 각 sampling point에 대응하는 projection weight이다. 이 식이 의미하는 바는 분명하다. regular convolution은 정해진 $3 \times 3$ 위치만 보지만, DCNv2는 $\Delta p_k$를 통해 각 위치를 유연하게 이동시킬 수 있다. 따라서 실제 receptive field가 데이터에 따라 달라질 수 있다. 또한 $m_k$가 각 위치의 중요도를 조절하므로 spatial aggregation도 입력 의존적으로 바뀐다.

저자들은 DCNv2가 잠재적으로 좋은 연산자이지만, 기존 방식은 주로 regular convolution을 fine-tuning으로 대체하는 용도로 쓰였고, 대규모 foundation model을 처음부터 학습하는 상황에는 적합하지 않다고 본다. 그래서 세 가지 수정을 가한다.

첫째, **sampling point마다 별도 weight를 두지 않고 공유한다**. 원래 DCNv2에서는 각 sampling point마다 독립적인 $w_k$가 있어 파라미터 수와 메모리 비용이 커진다. 이를 줄이기 위해 separable convolution 아이디어를 빌려, 공간 위치별 차이는 modulation 쪽이 담당하고 projection weight는 공유하도록 바꾼다. 이 수정은 대형 모델에서 특히 중요하다.

둘째, **multi-group mechanism**을 도입한다. 이는 transformer의 multi-head attention과 유사한 발상이다. 채널을 여러 group으로 나누고, 각 group이 독립적으로 offset과 modulation을 학습한다. 그러면 하나의 위치를 보더라도 group마다 다른 영역을 참고할 수 있어 더 풍부한 표현을 학습할 수 있다.

셋째, **modulation scalar 정규화를 sigmoid에서 softmax로 바꾼다**. 원래 sigmoid는 각 sampling point를 독립적으로 $[0,1]$ 범위로 제한하지만, 전체 합은 안정적이지 않다. 저자들은 이것이 대규모 모델 학습 시 gradient instability를 유발한다고 보고, sampling point 차원에서 softmax를 적용해 합이 1이 되도록 만든다. 이로써 학습 안정성을 높인다.

수정된 DCNv3는 다음과 같이 표현된다.

$$
y(p_0) = \sum_{g=1}^{G} \sum_{k=1}^{K} w_g \, m_{gk} \, x_g(p_0 + p_k + \Delta p_{gk})
$$

여기서 $G$는 group 수이고, $x_g$는 해당 group의 입력 채널 부분이다. $w_g$는 group별 shared projection weight이며, $m_{gk}$는 각 group 내 sampling point에 대한 softmax-normalized modulation scalar다.

이 식을 쉽게 말하면, InternImage의 한 convolution layer는 단순히 주변 9개 픽셀을 고정적으로 섞는 것이 아니라, 여러 group이 각자 다른 위치를 유연하게 골라 보고, 그 중요도를 정규화된 가중치로 합치는 구조라고 볼 수 있다. 그래서 작은 커널로도 긴 범위를 볼 수 있고, attention처럼 입력 의존적인 aggregation을 어느 정도 흉내 낸다.

### 3.2 Basic block 설계

InternImage의 basic block은 전통적인 ResNet bottleneck보다 transformer block에 더 가깝다. 블록 안에는 **Layer Normalization(LN)**, **DCNv3**, **Feed-Forward Network(FFN)**, **GELU**가 포함된다. 즉, 핵심 spatial mixing은 DCNv3가 담당하고, 채널 방향의 비선형 변환은 FFN이 담당하는 구조다. 논문은 기본적으로 **post-normalization** 설정을 사용한다고 밝힌다.

또한 sampling offset과 modulation scalar는 입력 feature $x$를 separable convolution, 즉 **$3 \times 3$ depth-wise convolution 뒤 선형 projection**에 통과시켜 예측한다. 이 부분은 DCNv3가 단순 고정 필터가 아니라 입력에 따라 동적으로 sampling strategy를 바꾸는 핵심 경로다.

### 3.3 Stem, downsampling, 계층 구조

stem은 두 개의 $3 \times 3$ convolution으로 구성되며 각각 stride 2, padding 1을 사용해 해상도를 총 4배 줄인다. 그 사이에 LN과 GELU가 들어간다. stage 사이의 downsampling layer도 $3 \times 3$ stride 2 convolution 뒤 LN을 붙이는 단순한 형태다. 따라서 전체 backbone은 해상도를 단계적으로 줄이며 더 깊고 넓은 feature representation을 만든다.

### 3.4 Stacking rule과 scaling rule

InternImage는 4개의 stage를 가지므로, 원래는 채널 수 $C_i$, group 수 $G_i$, block 수 $L_i$ 등 많은 hyperparameter를 정해야 한다. 저자들은 search space를 줄이기 위해 몇 가지 규칙을 둔다.

채널은 stage가 깊어질수록 2배씩 증가하고, group 수는 채널 수와 비례하도록 설계한다. block 수는 “AABA” 패턴을 따른다. 즉, 1, 2, 4번째 stage의 깊이는 같고, 3번째 stage는 그보다 같거나 더 깊게 만든다. 이 규칙 덕분에 모델은 사실상 $(C_1, C', L_1, L_3)$ 네 개의 핵심 hyperparameter만으로 결정된다.

이후 EfficientNet류의 scaling 아이디어를 가져와 depth와 width를 함께 키운다. 깊이 $D$와 첫 stage 채널 수 $C_1$에 대해 다음 규칙을 쓴다.

$$
D' = \alpha^\phi D, \quad C_1' = \beta^\phi C_1
$$

여기서 $\alpha \ge 1$, $\beta \ge 1$이고, 논문은 실험적으로 $\alpha = 1.09$, $\beta = 1.36$이 가장 좋았다고 보고한다. 이 규칙으로 InternImage-T/S/B/L/XL/H를 정의한다. Table 1에 따르면 InternImage-H는 약 1.08B 파라미터를 갖는다.

### 3.5 왜 $3 \times 3$ 커널인가

저자들은 deformable convolution의 유연한 sampling 덕분에 굳이 큰 커널이 필요 없다고 주장한다. appendix의 kernel size ablation에 따르면, $5 \times 5$는 정확도가 거의 늘지 않으면서 파라미터와 FLOPs만 증가하고, $7 \times 7$은 오히려 성능이 떨어진다. 논문의 해석은 명확하다. deformable operator에서는 sampling 위치 자체가 유연하므로, 커널 크기를 늘려 dense하게 보는 것보다 작은 커널을 동적으로 이동시키는 편이 최적화와 효율 측면에서 낫다는 것이다.

## 4. 실험 및 결과

## 4.1 이미지 분류

ImageNet에서 InternImage-T/S/B는 ImageNet-1K로 300 epoch 학습했고, L/XL은 ImageNet-22K로 사전학습 후 ImageNet-1K로 fine-tuning했다. 가장 큰 H 모델은 LAION-400M, YFCC-15M, CC12M을 합친 4.27억 장 규모의 공개 joint dataset 위에서 M3I pre-training을 30 epoch 수행한 뒤 ImageNet-1K에 fine-tuning했다.

결과는 매우 강하다. InternImage-T는 top-1 83.5%로 ConvNeXt-T의 82.1%, HorNet-T의 83.0%를 넘는다. InternImage-S는 84.2%, InternImage-B는 84.9%를 기록해 동급 CNN 및 여러 transformer 계열을 앞선다. 더 큰 모델에서는 InternImage-XL이 88.0%, InternImage-H가 입력 해상도 640 기준 89.6%를 달성했다. 저자들은 이 수치가 대형 ViT와의 격차를 약 1포인트 수준까지 줄였다고 해석한다. 특히 H 모델은 1.08B 파라미터로, CNN도 10억 파라미터 이상에서 유의미하게 확장될 수 있음을 보여준다.

## 4.2 객체 검출과 인스턴스 분할

COCO에서는 Mask R-CNN과 Cascade Mask R-CNN을 사용했다. backbone은 분류 사전학습 가중치로 초기화하고, 일반적인 $1\times$ 또는 $3\times$ schedule로 학습했다.

Mask R-CNN 기준에서 InternImage는 같은 규모의 모델들보다 매우 큰 개선폭을 보인다. 예를 들어 $1\times$ schedule에서 InternImage-T는 box AP 47.2를 기록해 Swin-T의 42.7보다 4.5포인트, ConvNeXt-T의 44.2보다 3.0포인트 높다. mask AP도 42.5로 Swin-T 39.3, ConvNeXt-T 40.1보다 높다. InternImage-B는 $3\times$ multi-scale schedule에서 box AP 50.3, mask AP 44.8을 기록한다.

더 큰 설정에서는 DINO detector와 Objects365 pre-training을 결합해 system-level 성능을 측정했다. 여기서 InternImage-H는 **COCO test-dev에서 65.4 box AP**를 달성한다. 이는 표에 나온 FD-SwinV2-G의 64.2, BEiT-3의 63.7, SwinV2-G의 63.1보다 높다. 논문은 특히 이 성능이 27% 적은 파라미터로 달성되었고, 복잡한 distillation 없이 얻은 결과라는 점을 강조한다.

## 4.3 의미 분할

ADE20K에서는 UperNet을 기본 segmentation framework로 사용했고, 최고 성능 설정에서는 Mask2Former를 결합했다. InternImage는 모든 규모에서 일관되게 강하다. InternImage-B는 50.8 mIoU(single-scale), 51.3 mIoU(multi-scale)로 ConvNeXt-B 49.1/49.9, RepLKNet-31B 49.9/50.6보다 높다. InternImage-XL은 55.0/55.3 mIoU를 기록한다.

가장 큰 결과는 InternImage-H + Mask2Former 조합이다. 이 설정은 **ADE20K에서 62.9 mIoU**를 달성하며, 표 기준 BEiT-3의 62.8을 근소하게 넘는다. UperNet만 사용한 InternImage-H도 multi-scale 기준 60.3 mIoU로 SwinV2-G의 59.9를 넘는다. 따라서 저자들의 주장대로, CNN 기반 foundation model도 대규모 사전학습과 강력한 dense prediction head를 결합했을 때 SOTA 수준에 도달할 수 있음을 보여준다.

## 4.4 어블레이션과 추가 분석

가장 중요한 어블레이션은 DCNv3를 구성하는 세 수정의 효과를 분리해 본 것이다.

첫째, **weight sharing**은 정확도를 거의 유지하면서 비용을 크게 줄인다. Table 6에서 shared weight를 제거하면 ImageNet top-1은 83.5에서 83.6으로 거의 같고 COCO AP도 47.2에서 47.4로 유사하지만, 논문 본문에 따르면 -H 규모에서는 파라미터 42.0%, GPU 메모리 84.2%를 절약할 수 있다. 즉, 대규모 모델에서 사실상 필수적인 설계다.

둘째, **multi-group spatial aggregation**은 실제 성능에 중요하다. group을 제거하면 top-1이 83.5에서 82.3으로, COCO box AP가 47.2에서 43.8로 크게 떨어진다. 저자들은 visualization을 통해 서로 다른 group이 서로 다른 공간 위치에 집중함을 보이며, 이것이 transformer의 multi-head와 유사한 역할을 한다고 해석한다.

셋째, **softmax normalization**은 학습 안정성에 결정적이다. 이를 제거하면 top-1이 65.7, COCO box AP가 38.7로 크게 무너진다. 이는 modulation scalar의 총합을 안정적으로 제어하는 것이 대규모 학습에서 매우 중요하다는 뜻이다.

또한 effective receptive field 분석에서, InternImage는 학습 후 특히 3, 4 stage에서 거의 전역적인 receptive field를 형성하는 반면, ResNet은 학습 후에도 활성화 주변에 더 제한된 반응을 보인다. 이는 InternImage가 작은 커널을 쓰면서도 실제로는 훨씬 넓은 문맥을 본다는 저자들의 주장을 뒷받침한다.

추가적으로 appendix에는 여러 downstream benchmark 결과도 제시된다. InternImage-H는 iNaturalist 2018, Places205, Places365, LVIS, Pascal VOC, OpenImages, CrowdHuman, BDD100K, COCO-Stuff, Pascal Context, Cityscapes, NYU Depth V2 등 다양한 데이터셋에서 기존 최고 기록을 갱신하거나 경쟁력 있는 성능을 보였다. 다만 이 부분은 본문 핵심 결과보다는 확장 검증에 가깝고, 각 실험의 세부 비교 조건이 본문만큼 상세하게 설명되지는 않는다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **CNN이 대규모 foundation model로서 다시 경쟁력을 가질 수 있음을 설계와 실험으로 설득력 있게 보였다는 점**이다. 단순히 “CNN도 된다”는 선언이 아니라, 왜 기존 CNN이 대규모 환경에서 밀렸는지 operator-level과 architecture-level에서 분석하고, deformable convolution을 중심으로 이를 해결하는 구체적 설계를 제시했다. 특히 DCNv3의 세 수정은 각각 목적이 분명하고, 어블레이션으로 그 필요성을 비교적 잘 입증했다.

또 다른 강점은 **dense prediction에서의 실질적 성능**이다. Image classification에서도 강하지만, COCO와 ADE20K 같은 검출·분할 과제에서 특히 강한 결과를 보인다. 이는 긴 receptive field와 adaptive spatial aggregation이 dense prediction에 중요하다는 논문의 설계 철학과 잘 맞는다. 또한 작은 $3 \times 3$ 커널로 long-range behavior를 얻는다는 점은, large-kernel CNN 대비 효율적이라는 메시지도 준다.

한편 한계도 분명하다. 논문 스스로도 **latency 문제**를 한계로 인정한다. deformable convolution 기반 연산자는 이론적 FLOPs만으로 설명되지 않는 구현 복잡성과 실제 추론 속도 문제를 갖는다. appendix의 throughput 비교에서도 InternImage-B는 224 해상도에서 ConvNeXt-B보다 느리고, 고해상도에서의 장점이 일부 있지만 여전히 DCN 기반 연산자의 효율 문제가 완전히 해결되지는 않았다.

또한 대형 모델의 최상위 결과는 ImageNet-22K, 4.27억 공개 데이터, Objects365, COCO-Stuff 등 여러 단계의 사전학습과 강력한 detector/segmenter 조합에 의존한다. 이것은 공정한 SOTA 비교에서는 흔한 설정이지만, “backbone 자체의 순수 기여”와 “대규모 시스템 조합의 기여”를 분리해서 해석할 필요가 있다. 논문은 이를 어느 정도 구분해 제시하지만, 최종 SOTA 수치는 시스템 수준 최적화가 함께 들어간 결과다.

비판적으로 보면, 논문은 왜 deformable convolution이 attention보다 본질적으로 더 나은지까지 주장하지는 않는다. 오히려 “효율과 inductive bias 측면에서 좋은 대안”이라는 수준에 머문다. 이 점은 오히려 정직한 서술에 가깝다. 다만 robustness나 data-scale 분석 일부는 appendix에 있고, 본문에서는 핵심 결론 위주로 제시되므로 독자가 일반화 범위를 신중히 판단할 필요는 있다.

## 6. 결론

이 논문은 deformable convolution을 핵심 연산자로 재설계한 **DCNv3**와, 이를 중심으로 한 **InternImage** 아키텍처를 제안함으로써 대규모 CNN 기반 비전 foundation model의 가능성을 강하게 보여준다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, CNN도 적절한 연산자와 구조를 갖추면 10억 파라미터 이상, 수억 장 데이터 규모로 확장 가능하다는 점을 보였다. 둘째, deformable convolution을 단순 보조 연산이 아니라 foundation model의 중심 operator로 재해석하고, 이를 위해 필요한 구조적 수정들을 제안했다. 셋째, ImageNet, COCO, ADE20K 등 대표 벤치마크에서 CNN과 ViT 모두를 상대로 매우 강력한 성능을 실증했다.

실제 적용 관점에서 보면, InternImage는 특히 객체 검출과 의미 분할처럼 고해상도 입력과 넓은 문맥 정보가 중요한 과제에서 유의미한 backbone 선택지가 될 가능성이 크다. 향후 연구 측면에서는 두 방향이 중요해 보인다. 하나는 DCN 기반 연산자의 실제 추론 지연을 더 줄여 실용성을 높이는 것이고, 다른 하나는 InternImage류의 CNN foundation model을 멀티모달 사전학습이나 더 광범위한 비전-언어 설정으로 확장하는 것이다. 논문이 직접 주장하는 수준을 넘어서 추측하자면, 이 연구의 가장 큰 의미는 “대규모 비전 모델의 표준이 반드시 transformer 하나로 수렴하지는 않는다”는 점을 다시 열어 두었다는 데 있다.
