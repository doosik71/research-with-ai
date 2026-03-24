# Transformer-Based Visual Segmentation: A Survey

- **저자**: Xiangtai Li, Henghui Ding, Haobo Yuan, Wenwei Zhang, Jiangmiao Pang, Guangliang Cheng, Kai Chen, Ziwei Liu, Chen Change Loy
- **발표연도**: 2023
- **arXiv**: <https://arxiv.org/abs/2304.09854>

## 1. 논문 개요

이 논문은 transformer 기반 visual segmentation 분야를 폭넓게 정리한 survey 논문이다. 다루는 범위는 단순한 semantic segmentation에 그치지 않고, instance segmentation, panoptic segmentation, video segmentation, point cloud segmentation, referring segmentation, domain adaptation, open-vocabulary segmentation, medical segmentation까지 확장된다. 즉, 특정한 하나의 새 모델을 제안하는 논문이 아니라, 최근 transformer 기반 segmentation 방법들을 공통된 관점에서 재구성하고 정리하는 데 목적이 있다.

논문이 제기하는 핵심 문제는 다음과 같다. 최근 computer vision에서 transformer, 특히 ViT와 DETR 계열 구조가 segmentation 성능과 설계 단순성 측면에서 기존 CNN 중심 접근을 빠르게 대체하고 있는데, 관련 연구가 매우 빠르게 증가하면서 분야 전체를 체계적으로 이해하기가 어려워졌다는 점이다. 기존 survey들은 vision transformer 일반론이나 넓은 vision task 전반을 다루는 경우가 많았고, segmentation 자체를 transformer 관점에서 일관되게 정리한 자료는 부족하다고 저자들은 본다.

이 문제가 중요한 이유는 segmentation이 자율주행, 로보틱스, 의료영상, 영상 편집, 감시 시스템 등 실제 응용과 직접 연결된 핵심 기술이기 때문이다. 특히 transformer 기반 방법은 전통적인 CNN 기반 segmentation보다 더 강한 global context modeling 능력을 갖고, query 기반 설계를 통해 detection과 segmentation을 하나의 통일된 틀로 다루기 쉬우며, 이미지와 비디오, 심지어 point cloud까지 확장 가능한 설계 패턴을 보여준다. 따라서 이 논문은 단순한 문헌 나열이 아니라, 향후 연구자들이 어떤 설계 축을 중심으로 방법들을 이해해야 하는지에 대한 구조적 지도를 제공하려는 의도가 강하다.

또한 이 survey의 중요한 특징은 transformer 기반 segmentation 방법들을 task별로만 나누지 않고, 공통되는 meta-architecture를 먼저 제시한 뒤 그 구조의 어떤 부분을 바꾸었는가에 따라 방법들을 분류한다는 점이다. 이는 단순한 taxonomy를 넘어, 왜 어떤 계열의 방법이 등장했는지, 서로 다른 task의 방법들이 사실은 어떤 공통 메커니즘을 공유하는지를 드러내는 데 유리하다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 transformer 기반 segmentation 방법들의 공통 골격을 DETR-like meta-architecture로 정리할 수 있다는 주장이다. 저자들은 최근의 많은 segmentation 방법들이 세부 구현은 달라도 크게 보면 feature extractor, object query, transformer decoder라는 공통 구조를 갖는다고 본다. 그리고 실제 차별화는 backbone 표현력, decoder의 cross-attention 설계, object query 초기화와 학습 방식, query를 이용한 association, 그리고 조건부 query 생성 방식에서 나타난다고 설명한다.

이 관점의 장점은 매우 크다. 예를 들어 semantic segmentation, instance segmentation, panoptic segmentation, video segmentation은 전통적으로 서로 다른 문제처럼 보이지만, 이 논문은 query가 무엇을 표현하느냐에 따라 사실상 하나의 프레임워크 안에서 이해할 수 있다고 본다. semantic segmentation에서는 query가 class query 또는 category-level representation처럼 작동할 수 있고, instance-aware task에서는 각 query가 하나의 object instance를 담당한다. video task에서는 각 query가 시간축을 따라 추적되는 object tracklet 또는 tube를 표현할 수 있다. 즉, query는 detection과 segmentation, image와 video, even multi-modal setting을 잇는 핵심 추상화로 제시된다.

기존 접근과의 차별점도 여기에 있다. 전통적 CNN 기반 segmentation은 대체로 dense pixel classification, FPN 기반 multi-scale fusion, 혹은 detection 결과와 segmentation 결과를 복잡하게 결합하는 공학적 파이프라인에 의존했다. 특히 panoptic segmentation이나 video segmentation에서는 여러 모듈을 따로 설계하고 후처리나 association 로직을 많이 붙이는 경향이 강했다. 반면 transformer 기반 접근은 query와 attention 메커니즘을 통해 object-level reasoning과 global context aggregation을 하나의 end-to-end 구조 안으로 가져오며, 경우에 따라 bounding box head조차 제거하고 순수 mask-based 접근으로 단순화할 수 있다.

또 하나의 중요한 아이디어는 이 논문이 단순히 대표 모델을 소개하는 데 그치지 않고, 왜 최근 방법들이 box-based 접근에서 pure mask-based 접근으로 이동했는지, 왜 cross-attention을 sparse하거나 masked하게 바꾸는지, 왜 query에 extra supervision이나 positional prior를 넣는지 등을 구조적으로 설명한다는 점이다. 즉, 이 survey는 “어떤 모델이 나왔다”가 아니라 “transformer segmentation 설계가 어떤 방향으로 진화했는가”를 설명하는 논문이다.

## 3. 상세 방법 설명

이 논문은 survey이므로 하나의 단일 알고리즘을 제안하지는 않는다. 대신 transformer-based segmentation의 공통 시스템 구조를 메타 아키텍처 형태로 제시한다. 그 핵심 구성요소는 backbone, neck, object query, transformer decoder, mask representation, bipartite matching 및 loss function이다.

먼저 backbone은 입력 이미지나 비디오, 혹은 point cloud로부터 feature를 추출하는 역할을 한다. 초기에는 ResNet 같은 CNN backbone이 널리 쓰였고, 이후 ViT가 등장하면서 patch sequence를 직접 transformer encoder에 넣는 방식이 보편화되었다. ViT에서는 입력 이미지 $I \in \mathbb{R}^{H \times W \times 3}$를 $P \times P$ patch로 나누고, 이를 flatten하여 patch token sequence로 만든다. 이후 linear embedding과 positional embedding을 더해 transformer encoder에 넣는다. classification에서는 별도의 $CLS$ token을 두지만, segmentation에서는 최종 token representation을 다시 dense feature map으로 해석하여 픽셀 수준 예측에 활용한다. 이 논문은 backbone의 진화를 세 갈래로 본다. 첫째, DeiT, MViT, Pyramid ViT처럼 transformer 자체를 더 잘 설계하는 흐름이 있고, 둘째, Swin, SegFormer, ConvNeXt, SegNeXt처럼 CNN과 transformer 또는 MLP를 혼합하여 local bias와 효율성을 확보하는 흐름이 있으며, 셋째, MAE, BEiT, DINO 같은 self-supervised learning을 통해 더 강한 feature representation을 획득하는 흐름이 있다.

neck은 보통 FPN과 같은 multi-scale feature aggregation 모듈이다. segmentation은 작은 물체와 큰 물체를 모두 다뤄야 하므로 단일 해상도 feature만으로는 한계가 있다. 논문은 FPN이 transformer 기반 방법에서도 여전히 중요하며, 특히 object query가 여러 해상도의 feature를 참조해 refinement될 때 성능이 향상된다고 설명한다. Deformable DETR 계열에서는 deformable attention을 통해 여러 scale의 feature에서 중요한 위치만 샘플링해 cross-scale fusion을 효율적으로 수행한다.

object query는 이 survey 전체에서 가장 핵심적인 개념이다. DETR에서 도입된 learnable embedding $Q_{obj} \in \mathbb{R}^{N_{ins} \times d}$는 최대 인스턴스 수 $N_{ins}$만큼의 query 벡터로 구성되며, 각 query는 하나의 object instance를 표현하도록 학습된다. 이 설계의 의미는 기존 detector에서 hand-crafted anchor나 NMS 같은 복잡한 구성요소를 대체하고, one-to-one set prediction으로 문제를 푼다는 데 있다. inference에서는 높은 score를 갖는 query만 선택하면 되므로 파이프라인이 간결해진다. 논문은 이후 많은 연구들이 이 query를 단순한 detection token이 아니라 tracking, multi-task fusion, language-conditioned selection, cross-image correspondence modeling 등 더 넓은 역할로 확장했다고 본다.

transformer decoder는 object query와 이미지 feature 사이의 cross-attention을 반복적으로 수행하면서 query를 정제한다. 여기서 query는 $Q_{obj}$이고, feature map $F$는 key와 value 역할을 한다. 정제된 query $Q_{out}$은 예측용 FFN으로 들어가 class, box, mask 등을 출력한다. segmentation에서는 refined query와 feature $F$ 사이의 dot product를 통해 binary mask logits를 형성하는 방식이 자주 쓰인다. 초기 DETR 스타일 구조에서는 이 decoder가 비교적 dense하고 일반적인 cross-attention 구조를 사용했지만, 이후 많은 연구가 이를 개선했다. Deformable DETR은 sparse sampling 기반 deformable attention을 도입하여 느린 학습과 높은 연산량 문제를 줄였다. Mask2Former는 masked cross-attention을 도입해 query가 이전 단계에서 예측된 mask 영역에만 attention하도록 제한했다. 이는 object query가 배경 전체를 보지 않고 자신이 담당하는 영역에 집중하게 해 더 효율적이고 정확한 refinement를 가능하게 한다.

mask representation은 두 형태로 나뉜다. 하나는 FCN처럼 pixel-wise dense prediction을 하는 방식이고, 다른 하나는 query마다 하나의 mask를 출력하는 per-mask prediction 방식이다. 전자는 semantic segmentation처럼 class-aware dense labeling에 적합하고, 후자는 instance segmentation, panoptic segmentation, video instance segmentation처럼 object-aware setting에 적합하다. 이 논문은 최근 흐름이 특히 후자의 mask-based query prediction으로 이동하고 있음을 강조한다. Max-DeepLab, K-Net, MaskFormer, Mask2Former가 대표적이다.

학습에서는 bipartite matching이 중요한 역할을 한다. object query와 ground truth mask 또는 object 사이에 one-to-one 대응을 만들기 위해 Hungarian algorithm을 사용한다. matching cost는 class label, box, mask 차이 등을 기준으로 계산된다. 이렇게 매칭된 뒤 각 query는 자신에게 할당된 ground truth에 대해서만 loss를 받는다. instance-aware segmentation의 경우 classification loss와 segmentation loss를 함께 사용하며, segmentation loss는 일반적으로 binary cross-entropy loss와 dice loss를 포함한다. dice loss는 예측 마스크와 정답 마스크의 overlap을 직접 반영하므로 class imbalance가 심한 경우에도 유용하다.

논문은 transformer의 기본 연산도 간단히 정리한다. 입력 token이 $X=[x_1,\dots,x_N] \in \mathbb{R}^{N \times c}$일 때 positional encoding을 더한 입력을 $I=X+P$라 두고, 선형 변환을 통해 다음과 같이 Query, Key, Value를 만든다.

$$
Q = IW_q,\quad K = IW_k,\quad V = IW_v
$$

그리고 self-attention은 다음과 같이 정의된다.

$$
O = SA(Q,K,V) = Softmax(QK^\top)V
$$

논문은 원문 식에서 $\frac{1}{\sqrt{d}}$ scaling을 명시적으로 적지는 않았고, 여기서도 그 표기를 따르지 않는다. 직관적으로는 각 token이 다른 모든 token과의 유사도를 계산해 global context를 집계하는 연산이다. multi-head self-attention은 이 연산을 여러 head에서 병렬로 수행한 뒤 concat하고 projection하는 방식이다.

이 survey의 방법론 서술에서 특히 중요한 부분은 “방법 분류” 자체가 곧 분석 프레임이라는 점이다. 저자들은 transformer segmentation 연구를 다음 다섯 축으로 나눈다. 첫째는 strong representation 학습이다. backbone 품질이 segmentation 성능에 직접 연결되므로 better ViT design, hybrid CNN/transformer/MLP, self-supervised learning이 핵심 설계 축이 된다. 둘째는 decoder의 cross-attention 설계다. image segmentation에서는 더 나은 cross-attention operator와 decoder 구조를, video segmentation에서는 spatial-temporal attention과 frame association 구조를 고민한다. 셋째는 object query 최적화다. positional prior를 넣어 localization을 쉽게 하거나, denoising loss와 one-to-many supervision으로 학습을 안정화한다. 넷째는 query를 association 도구로 사용하는 방향이다. video segmentation에서 tracking query나 association embedding으로 활용하거나, multi-task setting에서 task 간 feature linking에 사용한다. 다섯째는 conditional query fusion이다. language-conditioned query를 만들어 referring segmentation을 수행하거나, support image와 query image의 관계를 모델링해 few-shot segmentation을 수행한다.

video segmentation으로 확장될 때는 query의 의미가 “프레임별 object”가 아니라 “시간축을 따라 유지되는 tracked entity”로 바뀐다. VisTR는 여러 프레임을 flatten한 공간-시간 feature 위에서 query가 곧 object tube를 예측하게 만들었고, TubeFormer나 Video K-Net은 query를 통해 VSS, VIS, VPS를 통합적으로 다루려 했다. 이때 핵심 문제는 temporal consistency와 association이며, 논문은 이 부분을 spatial-temporal cross-attention design과 query-based association design으로 설명한다.

또한 survey는 세부 subfield도 같은 시각으로 재해석한다. 예를 들어 open-vocabulary segmentation은 결국 class-agnostic mask proposal과 language embedding matching의 결합으로 볼 수 있고, foundation model tuning은 pretrained vision-language model 또는 large vision model에서 knowledge를 끌어와 segmentation head와 연결하는 문제로 정리된다. domain adaptation, weakly supervised learning, unsupervised segmentation 역시 transformer의 representation과 query 구조를 어떻게 활용하느냐로 설명된다.

## 4. 실험 및 결과

이 논문은 survey이므로 하나의 모델에 대한 단일 실험이 아니라, 다양한 대표 모델의 benchmark를 표 형태로 비교하고 일부는 동일 조건에서 re-benchmarking까지 수행한다. 따라서 실험 파트의 핵심은 “어떤 task에서 어떤 모델 계열이 강한가”와 “decoder 설계가 실제로 얼마나 차이를 만드는가”를 정리하는 데 있다.

먼저 데이터셋 측면에서 이 논문은 transformer-based segmentation 연구에서 자주 사용되는 benchmark들을 체계적으로 정리한다. 이미지 segmentation에서는 Pascal VOC, Pascal Context, COCO, ADE20K, Cityscapes, Mapillary를 다루고, referring segmentation에서는 RefCOCO와 gRefCOCO를, video segmentation에서는 VSPW, Youtube-VIS-2019, OVIS, VIP-Seg, Cityscapes-VPS, KITTI-STEP, DAVIS-2017, Youtube-VOS, MOSE를 소개한다. point cloud segmentation까지 포함해 task별 대표 benchmark를 넓게 커버한다. 이 자체가 survey로서 유용한 기여다.

평가지표도 task별로 구분된다. semantic segmentation과 video semantic segmentation은 mIoU를 사용하고, instance segmentation과 video instance segmentation은 mAP를 사용한다. panoptic segmentation은 PQ, video panoptic segmentation은 VPQ와 STQ를 사용한다. 논문은 특히 VPS에서는 VPQ가 temporal window 위의 panoptic quality를 측정하고, STQ는 segmentation quality와 association quality를 분리해 보는 장점이 있다고 설명한다.

이미지 semantic segmentation 결과를 보면, Table V 기준으로 Cityscapes와 ADE20K에서는 Mask2Former와 OneFormer 계열이 매우 강하다. 예를 들어 ADE20K에서 OneFormer는 58.8 mIoU, Mask2Former는 57.3 mIoU를 기록한다. 반면 COCO-Stuff와 Pascal-Context에서는 SegNext가 각각 47.2, 60.9 mIoU로 좋은 결과를 보인다. 이 결과는 universal segmentation decoder가 강력하다는 점과 함께, backbone과 training recipe 차이도 성능에 크게 작용함을 시사한다.

COCO instance segmentation 결과(Table VI)에서는 Mask DINO가 가장 강한 성능을 보인다. ResNet50 backbone에서 46.3 AP, Swin-L backbone에서 52.3 AP를 기록하여 Mask2Former보다 앞선다. 특히 large object에서의 성능도 높게 보고된다. 이는 query optimization과 detection data의 joint utilization이 segmentation 성능을 밀어올릴 수 있음을 보여준다.

panoptic segmentation(Table VII)에서는 K-Max DeepLab, Mask DINO, OneFormer, CLUSTSEG 등이 경쟁한다. COCO에서는 Mask DINO와 K-Max DeepLab이 53.0 이상 PQ를 기록하고, Swin-L 또는 ConvNeXt-L backbone을 쓴 강한 모델들은 58 안팎까지 도달한다. ADE20K에서는 OneFormer가 51.4 PQ, Cityscapes에서는 K-Max DeepLab이 68.4 PQ로 좋은 결과를 보인다. 전반적으로 pure mask-based query segmentation이 panoptic task에서 매우 강력하다는 점이 확인된다.

video semantic segmentation(Table VIII)에서는 TubeFormer가 VSPW에서 63.2 mIoU를 기록해 다른 방법보다 크게 앞선다. 이는 단순한 frame-level segmentation이 아니라 tube-level modeling과 temporal token exchange가 실제로 큰 효과를 가질 수 있음을 보여준다.

video instance segmentation(Table XII)에서는 CTVIS가 가장 강한 결과를 보여준다. ResNet50 backbone 기준 YT-VIS-2019에서 55.1 AP, YT-VIS-2021에서 50.1 AP, Swin-L backbone 기준으로는 각각 65.6, 61.2 AP를 기록한다. OVIS처럼 occlusion이 심한 데이터셋에서는 GenVIS와 CTVIS가 강한 성능을 보인다. 이 결과는 단순히 image segmentation 모델을 video에 확장하는 것만으로는 충분하지 않고, temporal consistency training과 association 메커니즘이 중요함을 의미한다.

video panoptic segmentation(Table XIII)에서는 dataset별로 강자가 다르다. Cityscapes-VPS에서는 SLOT-VPS가 63.7 VPQ로 좋고, VIP-Seg에서는 TubeLink가 49.4 STQ, KITTI-STEP에서는 Video K-Net이 73.0 STQ를 기록한다. 이는 video panoptic segmentation이 여전히 매우 어려운 문제이며, dataset 특성에 따라 유리한 설계가 달라질 수 있음을 보여준다.

이 논문의 실험 파트에서 특히 의미 있는 부분은 re-benchmarking이다. 저자들은 semantic segmentation과 panoptic segmentation에서 backbone과 neck을 동일하게 맞춘 상태로 decoder 설계 차이를 비교한다. 이 설정은 매우 중요하다. 왜냐하면 많은 segmentation 논문은 backbone, augmentation, training schedule까지 모두 달라서 “정말 decoder가 좋아서 성능이 오른 것인지” 분리하기 어렵기 때문이다.

semantic segmentation re-benchmark(Table IX)에서는 같은 조건에서 SegFormer+가 COCO-Stuff와 Cityscapes에서 가장 좋고, ADE20K에서는 Mask2Former가 가장 좋다. instance segmentation re-benchmark(Table X)에서는 Mask2Former가 43.1 mAP로 가장 좋다. panoptic segmentation re-benchmark(Table XI)에서도 Mask2Former가 COCO 52.1, Cityscapes 62.3, ADE20K 39.2 PQ로 가장 좋은 결과를 보인다. 즉, 동일한 encoder와 neck 하에서는 Mask2Former의 decoder 설계가 상당히 강력하다는 결론을 도출할 수 있다.

또한 ablation(Table XIV)에서는 K-Net에 large-scale jittering augmentation과 deformable FPN을 추가할수록 PQ가 47.1에서 49.2까지 증가한다. 이는 논문이 단순히 모델 구조만이 아니라 training setup과 feature pyramid 설계도 성능에 큰 영향을 준다는 점을 인정하고 있음을 보여준다.

구현 세부사항도 제시된다. semantic segmentation re-benchmark는 MMSegmentation codebase를 사용하고, panoptic 및 instance segmentation은 MMDetection을 사용한다. optimizer는 AdamW이며, 데이터셋별 crop size와 iteration 설정이 구체적으로 제시된다. 이런 정보는 survey 논문치고는 실험 재현 가능성 측면에서 비교적 성실한 편이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 transformer-based segmentation 분야를 단순한 논문 목록이 아니라 공통 메커니즘 중심으로 정리했다는 점이다. 특히 DETR-like meta-architecture를 중심에 놓고 backbone, decoder, query, association, conditional generation이라는 축으로 문헌을 재구성한 것은 독자가 분야를 빠르게 파악하는 데 매우 유용하다. 서로 다른 task에 속한 방법들이 사실은 같은 설계 사상의 변형임을 드러내 주기 때문이다.

둘째 강점은 범위가 매우 넓다는 점이다. 이 논문은 semantic, instance, panoptic뿐 아니라 video, point cloud, referring, domain adaptation, open-vocabulary, weakly supervised, medical segmentation까지 포함한다. 따라서 transformer segmentation의 “전체 지형도”를 이해하기에 적합하다. 특히 visual segmentation을 image/video/3D/multimodal까지 확장된 개념으로 보는 시각이 돋보인다.

셋째는 단순 survey를 넘어서 re-benchmarking을 수행했다는 점이다. 이는 다양한 논문을 같은 조건에서 비교해 decoder의 실제 효과를 더 공정하게 보려는 시도이며, survey로서의 실용성을 높인다. 특히 Mask2Former가 동일 설정에서 강력한 baseline임을 보여준 부분은 이후 연구자가 baseline을 고를 때 도움을 준다.

넷째는 미래 방향 제시가 비교적 구체적이라는 점이다. 단순히 “더 좋은 모델이 필요하다” 수준이 아니라, unified image/video segmentation, joint multimodal learning, lifelong learning, long video segmentation, generative segmentation, visual reasoning과 segmentation의 결합처럼 구조적 과제를 제시한다. 이는 survey의 결론이 향후 연구 agenda 설정으로 이어질 수 있게 만든다.

하지만 한계도 분명하다. 첫째, 이 논문은 survey이기 때문에 각 개별 방법의 수학적 세부사항이나 아키텍처 차이를 매우 깊게 파고들지는 않는다. 예를 들어 Mask2Former, K-Net, kMaX-DeepLab 같은 핵심 모델들의 decoder 내부 차이를 엄밀히 분석하기보다는 큰 흐름 위주로 설명한다. 따라서 이 분야에 새로 들어오는 독자에게는 매우 좋지만, 특정 모델을 구현하거나 재현하려는 독자에게는 원논문을 다시 읽어야 한다.

둘째, 제시된 meta-architecture가 모든 transformer segmentation 방법을 완전히 포괄하지는 못한다는 점을 저자 스스로 인정한다. 예를 들어 SegFormer나 SETR처럼 dense prediction head 중심의 semantic segmentation은 query-based decoder 중심 틀에 완전히 들어맞지 않는다. 저자들은 이를 “기본형 meta-architecture”로 해석하려 하지만, 이 해석은 다소 넓은 추상화라서 설명력이 높아지는 대신 구체성은 줄어든다.

셋째, benchmark 정리는 유용하지만, 표에 포함된 결과들이 서로 다른 backbone, training data, extra data, augmentation, schedule 위에서 보고된 값이 많아서 직접 비교에 조심이 필요하다. 저자들이 re-benchmarking으로 일부 이를 보완했지만, 전체 표를 볼 때는 여전히 apples-to-apples 비교가 아닌 경우가 적지 않다. 특히 open-vocabulary segmentation 표처럼 extra data 사용 여부가 크게 다르기 때문에 숫자만으로 우열을 판단하기 어렵다.

넷째, 이 논문은 분야의 흐름을 “query 중심”으로 많이 설명한다. 이는 DETR 이후 흐름을 잡는 데는 좋지만, 동시에 non-query-based dense segmentation 계열이나 generative segmentation 계열의 잠재력을 다소 부차적으로 다룰 위험도 있다. 실제로 미래 방향에서 generative segmentation을 언급하긴 하지만, 본문 전체의 설명 무게중심은 분명 query-based transformer에 있다.

다섯째, 발표연도나 arXiv 링크 같은 메타데이터가 제공된 본문 추출 텍스트 안에는 명확히 나타나지 않는다. 따라서 이 보고서 역시 그 부분을 확정적으로 적을 수 없다. 이는 보고서의 한계라기보다 입력 텍스트의 한계이지만, 논문 분석 시 메타정보를 별도 확인하지 않는 한 정확한 서지 정보를 복원하기 어렵다는 점을 보여준다.

종합하면, 이 논문은 “새 알고리즘 제안”이 아니라 “분야 구조화”에 강점이 있고, 그 목적에는 매우 충실하다. 반면 개별 모델에 대한 세부 구현 이해나 엄밀한 공정 비교를 위해서는 반드시 원 논문과 코드를 함께 봐야 한다.

## 6. 결론

이 논문은 transformer-based visual segmentation 분야를 체계적으로 정리한 매우 포괄적인 survey다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, segmentation task를 image, video, point cloud, multimodal setting까지 포함하는 넓은 관점에서 정리했다. 둘째, DETR-like meta-architecture를 중심으로 transformer segmentation 방법들을 backbone, decoder, query, association, conditional query 설계라는 공통 축 위에 올려놓았다. 셋째, 대표 모델들의 benchmark와 re-benchmarking을 통해 어떤 설계가 실제로 강한지 비교 가능한 형태로 제공했다.

이 연구가 중요한 이유는 transformer segmentation이 이미 여러 task의 주류 설계가 되었고, 앞으로는 foundation model, open-vocabulary learning, multimodal reasoning, lifelong learning과 강하게 결합될 가능성이 크기 때문이다. 특히 query 기반 설계는 image와 video, detection과 segmentation, vision과 language를 하나의 표현 체계로 잇는 공통 인터페이스로 작동하고 있다. 이런 점에서 이 survey는 단순한 정리 문서를 넘어, 앞으로 어떤 연구 방향이 자연스럽게 확장될지를 보여주는 로드맵 역할을 한다.

실제 적용 측면에서도 의미가 크다. 자율주행, 로봇 내비게이션, 의료영상 분석, video editing, open-world perception 같은 환경에서는 단일 task에 특화된 segmentation보다 다양한 입력과 조건, 긴 시간축, 미지 클래스까지 다룰 수 있는 통합적 모델이 필요하다. 이 논문은 바로 그런 방향으로 분야가 이동하고 있음을 잘 보여준다.

다만 이 논문에서 제시한 미래 과제들, 예를 들어 long video segmentation, open-world lifelong segmentation, generative segmentation, visual reasoning 결합 등은 아직 해결되지 않은 문제들이다. 따라서 이 survey의 진짜 가치는 “무엇이 잘 되었는가”만이 아니라 “무엇이 아직 어려운가”를 분명히 해 준다는 데 있다. 그런 의미에서 이 논문은 transformer segmentation 분야를 처음 공부하는 연구자뿐 아니라, 새로운 연구 주제를 찾는 연구자에게도 매우 유용한 참고 문헌이다.
