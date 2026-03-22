# Pointly-Supervised Instance Segmentation

이 논문은 instance segmentation에서 가장 큰 실용적 병목인 **mask annotation 비용**을 줄이면서도, fully-supervised 성능에 매우 가깝게 가는 약지도(weak supervision) 학습이 가능한지를 묻습니다. 저자들은 bounding box에 더해, 그 박스 내부에서 **무작위로 샘플링한 점들에 대해 object/background 이진 라벨만 부여하는 매우 단순한 annotation scheme**을 제안합니다. 핵심은 이 점 supervision이 기존 Mask R-CNN, PointRend, CondInst 같은 모델에 **구조 변경 없이 바로 적용 가능**하다는 점입니다. 논문은 COCO, PASCAL VOC, Cityscapes, LVIS에서 객체당 10개 점만으로도 Mask R-CNN이 fully-supervised 성능의 **94%–98%**에 도달하고, annotation 시간은 polygon mask 대비 약 **5배 빠르다**고 보고합니다. 또한 point supervision에 더 잘 맞는 새로운 mask head인 **Implicit PointRend**도 함께 제안합니다.  

## 1. Paper Overview

이 논문이 다루는 핵심 문제는 “instance segmentation을 위해 정말 full mask annotation이 꼭 필요한가?”입니다. 기존에는 object mask annotation이 매우 비싸고 느렸습니다. 논문은 COCO 기준 polygon-based object mask 하나를 만드는 데 평균 **79.2초**가 걸리는 반면, bounding box는 약 **7초**면 가능하다고 설명합니다. 저자들은 이 차이가 instance segmentation의 확산을 막는 가장 큰 이유 중 하나라고 봅니다.

이 문제가 중요한 이유는, 실제 응용에서는 새로운 카테고리나 새로운 장면 유형에 대해 대량의 정밀 mask를 다시 수집하기가 어렵기 때문입니다. 기존 약지도 방법, 특히 box supervision 기반 방법은 많이 발전했지만 COCO 같은 대규모 데이터셋에서는 여전히 fully-supervised 성능과 큰 격차가 있었습니다. 논문은 BoxInst가 COCO에서 fully-supervised counterpart의 약 **85%** 수준이라고 언급하며, box보다 약간 더 풍부하지만 훨씬 저렴한 annotation form이 더 좋은 타협점을 제공할 수 있다고 주장합니다.

## 2. Core Idea

이 논문의 핵심 직관은 놀랄 만큼 단순합니다.

> **full mask 전체를 그리지 말고, bounding box 안의 랜덤 포인트 몇 개만 object/background로 분류하게 하자.**

즉 각 객체에 대해 bounding box를 먼저 얻고, 그 안에서 $N$개의 랜덤 point를 샘플링한 뒤 annotator는 각 점이 object인지 background인지만 표시합니다. 저자들은 이 annotation을 $\mathcal{P}_N$이라 부릅니다. 중요한 것은 점 위치를 사람이 직접 클릭하게 하지 않고 **무작위 샘플링**한다는 점입니다. 이 방식은 사람의 클릭 편향을 줄이고, 기존 full-mask 데이터셋으로도 쉽게 simulation할 수 있어 대규모 실험과 ablation이 가능합니다.

이 아이디어의 진짜 강점은 annotation form 그 자체보다, 이것이 **기존 instance segmentation 모델의 학습 파이프라인과 잘 맞는다**는 데 있습니다. Mask R-CNN이나 CondInst는 본질적으로 regular grid 위에서 mask logits를 예측하므로, ground-truth point 위치에서 예측값을 bilinear interpolation으로 읽어오면 그대로 point-level cross-entropy loss를 줄 수 있습니다. 다시 말해, 이 논문은 새로운 복잡한 latent constraint나 pseudo-mask generation보다, **기존 per-pixel mask supervision을 point supervision으로 자연스럽게 축소**한 것이 핵심입니다.

또 하나의 novelty는 **Implicit PointRend**입니다. 저자들은 standard PointRend가 full mask supervision에서는 Mask R-CNN보다 좋지만, point supervision에서는 오히려 장점이 줄어든다는 관찰에서 출발합니다. 그 원인을 PointRend의 저해상도 coarse mask head와 이중 loss 구조에서 찾고, 객체마다 point-wise prediction function의 파라미터를 직접 생성하는 더 단순한 implicit mask representation을 제안합니다. 이로써 point supervision에 더 적합한 구조를 만듭니다.  

## 3. Detailed Method Explanation

### 3.1 Annotation format: $\mathcal{P}_N$

논문이 제안하는 annotation format은 bounding box와 내부 랜덤 포인트 라벨들의 집합입니다. 각 객체의 bounding box 안에서 $N$개의 random point location을 생성하고, annotator는 각 점을 object 또는 background로 이진 분류합니다. 저자들이 point click이 아니라 random point classification을 선택한 이유는 크게 두 가지입니다.

첫째, 사람 클릭은 상관된 위치에 몰리기 쉽습니다. 둘째, 랜덤 포인트는 기존 mask annotation으로부터 자동 시뮬레이션할 수 있어 실험 재현과 대규모 비교가 쉽습니다. 논문은 semantic segmentation의 선행연구도 random point가 human click보다 유리할 수 있음을 보여줬다고 인용합니다.

### 3.2 Annotation time와 품질

논문은 실제 annotation tool을 만들어 COCO와 LVIS의 100개 객체를 대상으로 시간과 품질을 측정했습니다. 결과적으로 **점 하나 분류에 평균 0.9초**, bounding box 7초를 합치면 객체당 10개 점 supervision인 $\mathcal{P}_{10}$의 총 시간은

$$
7 + 10 \cdot 0.9 = 16 \text{ seconds}
$$

입니다. 저자들은 이를 polygon mask annotation의 79.2초와 비교해 약 **5배 빠르다**고 정리합니다. 또한 수집된 point labels는 COCO GT와 약 **90%**, 더 정밀한 LVIS GT와는 약 **95%** 일치했다고 보고합니다. 오차는 대부분 object boundary 근처나 polygon GT 자체가 부정확한 곳에서 발생했습니다.

### 3.3 Training with points

가장 중요한 방법론적 부분은 point supervision을 기존 모델에 넣는 방식입니다. fully-supervised setting에서는 예측 mask grid와 동일한 해상도의 GT mask grid를 만들어 loss를 계산합니다. 반면 point supervision에서는 GT가 sparse point들만 주어지므로, 예측된 regular grid mask에서 **GT point 위치의 값을 bilinear interpolation**으로 읽어낸 뒤, 그 점들에 대해서만 cross-entropy loss를 적용합니다. gradient는 interpolation을 통해 원래 mask prediction에 역전파됩니다.

이 방식은 상당히 우아합니다. 모델 구조를 바꾸지 않고도 sparse supervision을 줄 수 있기 때문입니다. Region-based model에서는 GT point가 predicted box 밖에 있을 경우 그 점을 무시하고, image-level mask model에서는 필요하다면 box 밖 배경점도 추가할 수 있지만, 본 논문은 기본 설정으로 각 객체당 $N$개의 annotated point만 사용합니다.

### 3.4 Point-based data augmentation

저자들은 point supervision에서는 supervision 자체가 sparse하므로, 긴 training schedule이나 큰 backbone에서 overfitting이 생길 수 있다고 봅니다. 이를 완화하기 위해 매우 단순한 augmentation을 제안합니다. 매 iteration마다 사용 가능한 point 전부를 쓰지 않고, 절반만 무작위로 subsample합니다. 예를 들어 $\mathcal{P}_{10}$이면 매 iteration에 5개 점만 씁니다. 직관적으로는 sparse label set의 조합 다양성을 늘리는 효과입니다. 논문은 특히 high-capacity ResNeXt 계열에서 이 방법이 더 도움이 된다고 보고합니다.  

### 3.5 Why PointRend struggles and Implicit PointRend

논문은 PointRend가 이름만 보면 point supervision에 잘 맞을 것 같지만 실제로는 그렇지 않다고 분석합니다. 그 이유로 standard PointRend의 coarse mask head가 **낮은 7×7 해상도**의 region-level representation을 먼저 만들기 때문에, sparse point supervision만으로 이를 정확히 학습하기가 어렵다고 봅니다. 저자들은 이 문제를 해결하기 위해 coarse mask head를 없애고, 각 객체마다 최종 point-wise mask prediction function의 파라미터를 직접 생성하는 **Implicit PointRend**를 제안합니다.  

Implicit PointRend의 장점은 두 가지입니다.

첫째, importance point sampling이 필요 없습니다.
둘째, coarse mask loss와 point refinement loss의 이중 구조 대신 **하나의 point-level mask loss**만 사용합니다.

즉 point supervision의 성격에 더 직접적으로 맞춘 implicit function learning 구조라고 볼 수 있습니다. 저자들은 이 모델이 point supervision에서 standard PointRend보다 명확히 더 잘 작동한다고 보고합니다.  

## 4. Experiments and Findings

### 4.1 데이터셋과 실험 설정

논문은 COCO를 중심으로, PASCAL VOC, Cityscapes, LVISv1.0까지 총 4개 데이터셋에서 point supervision을 평가합니다. COCO는 118k train / 5k val의 80개 클래스, VOC는 약 10k 이미지의 20개 클래스, Cityscapes는 2975/500의 고해상도 자율주행 장면 데이터, LVIS는 COCO 이미지를 공유하지만 1000개 이상 카테고리를 federated 방식으로 annotation한 장기꼬리 데이터셋입니다. 즉 이 방법이 단일 benchmark 전용이 아니라, 일반 객체, 자율주행 장면, long-tail category 분포에 걸쳐 적용 가능함을 보여주려는 구성입니다.

### 4.2 Number of points

COCO ablation에서 가장 중요한 관찰은 **point 수가 늘수록 성능은 빠르게 증가하지만, 수십 개 이상에서는 diminishing returns가 나타난다**는 점입니다. 특히 Mask R-CNN + ResNet-50-FPN 기준으로 $\mathcal{P}_{10}$만으로 **36.1 AP**를 달성합니다. 이는 full mask supervision의 **37.2 AP** 대비 약 **97%** 수준입니다. 또 20 points는 10 points보다 약 **0.3 AP**만 더 좋지만 annotation 시간은 2배 가까이 듭니다. 그래서 저자들은 $\mathcal{P}_{10}$을 실용적인 sweet spot으로 채택합니다.

### 4.3 Main results across datasets

논문이 가장 강하게 내세우는 결과는 Mask R-CNN이 $\mathcal{P}_{10}$만으로도 여러 데이터셋에서 full supervision 성능의 94%–98%를 달성한다는 점입니다. 구체적으로 본문 표에서는:

* COCO val2017: full mask **37.2 AP**, $\mathcal{P}_{10}$ **36.1 AP** → **97%**
* PASCAL VOC val: full mask **66.3 AP50**, $\mathcal{P}_{10}$ **64.2 AP50** → **97%**
* Cityscapes val: full mask **32.7 AP**, $\mathcal{P}_{10}$ **30.7 AP** → **94%**

로 보고됩니다. 논문 abstract는 LVIS도 포함해 전체 범위를 94%–98%라고 요약합니다.  

이 결과의 의미는 꽤 큽니다. weak supervision인데도 기존 full-mask용 모델과 거의 같은 구조와 학습 파이프라인으로 이 정도 수준까지 올라갔기 때문입니다. 즉, “instance segmentation에는 full masks가 절대적으로 필요하다”는 통념을 상당히 약화시키는 결과입니다.

### 4.4 Comparison with other weak supervision schemes

논문은 동일한 annotation budget 아래에서 point-based scheme이 다른 supervision form보다 더 좋은 trade-off를 보인다고 주장합니다. 특히 COCO에서 BoxInst 같은 box-supervised strong baseline과 비교할 때, point supervision은 조금 더 많은 annotation effort를 쓰지만 성능은 fully-supervised에 훨씬 더 가까워집니다. 또 DEXTR 기반 pseudo-mask 방식과 비교하는 baseline도 두는데, 저자들은 직접적인 point supervision이 더 간단하고 강한 기준선이라고 봅니다. Figure 5 설명에서도 같은 annotation budget 하에서 point-based scheme이 full-mask subset 학습이나 box supervision보다 더 낫다고 정리합니다.  

### 4.5 Self-training and transfer learning

저자들은 point/full mask 간 격차를 더 줄이기 위해 self-training도 실험합니다. 또한 downstream transfer setting에서 **point-based pre-training이 mask-based pre-training과 거의 동등하다**는 점도 강조합니다. 이는 point supervision이 단지 저렴한 대체 수단이 아니라, representation learning 관점에서도 꽤 강력할 수 있음을 시사합니다. 다만 이 대화에서 확보된 텍스트는 self-training과 transfer의 전체 수치표를 완전하게 보여주지는 않으므로, 세부 수치 대신 논문의 주장을 보수적으로 요약하는 것이 적절합니다.

### 4.6 Implicit PointRend results

논문은 standard PointRend와 Implicit PointRend를 full mask 및 point supervision 모두에서 비교합니다. 요지는 다음과 같습니다.

* full mask supervision에서는 Implicit PointRend가 PointRend와 대체로 비슷한 수준
* point supervision에서는 Implicit PointRend가 standard PointRend보다 더 우수
* 전체적으로 point-supervised Implicit PointRend도 full-mask counterpart의 약 **96%** 수준에 도달

저자들은 더 나아가, Implicit PointRend를 point supervision으로 학습해도 fully-supervised Mask R-CNN 수준에 도달할 수 있다고 주장합니다. 이는 annotation budget이 제한된 실제 환경에서 특히 강한 메시지입니다.

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **문제가 아니라 annotation protocol을 다시 설계했다**는 점입니다. 복잡한 pseudo-label 생성이나 handcrafted prior 없이, bounding box와 몇 개의 point만으로 기존 full-mask 모델을 거의 그대로 살릴 수 있다는 점은 매우 실용적입니다.

둘째, **단순성**이 강력한 무기입니다. point 위치는 랜덤 샘플링, annotator는 이진 분류만 수행, 학습은 bilinear interpolation 후 point loss. 전체 파이프라인이 지나치게 복잡하지 않아서 확장성과 재현성이 높습니다.

셋째, empirical evidence가 강합니다. COCO뿐 아니라 VOC, Cityscapes, LVIS에 걸쳐 일관된 결과를 제시하며, annotation time 측정까지 직접 했습니다. 약지도 segmentation 논문 중 상당수가 소규모 데이터셋이나 복잡한 multi-stage pipeline에 의존하는 점을 생각하면, 이 논문은 매우 설득력 있는 baseline 성격을 갖습니다.  

### 한계

첫째, point supervision은 싸지만 여전히 **bounding box annotation을 전제로** 합니다. 즉 pure point-only supervision은 아닙니다. 이 점에서 annotation 비용을 완전히 없애는 방향과는 다릅니다.

둘째, sparse supervision의 한계는 분명합니다. 저자들도 긴 schedule이나 큰 backbone에서 overfitting과 supervision 부족 문제가 커진다고 보고하며, 이를 point subsampling augmentation으로 완화합니다. 즉 point 수가 적을수록 데이터 효율은 좋지만, 모델 capacity가 커질수록 구조적 한계가 드러납니다.

셋째, PointRend 사례가 보여주듯, **모든 full-supervised 모델이 point supervision에 똑같이 잘 맞는 것은 아닙니다**. 결국 supervision form이 바뀌면 그에 맞는 model head 설계가 필요할 수 있으며, Implicit PointRend는 그 방향의 한 예입니다.  

### 비판적 해석

제 해석으로 이 논문의 진짜 공헌은 “instance segmentation에서 약지도의 ceiling이 생각보다 훨씬 높다”는 것을 보여준 데 있습니다. Box-supervised 접근들은 보통 projection loss, pairwise affinity, pseudo-mask 생성 등 복잡한 기법을 동원했는데, 이 논문은 오히려 **약간 더 풍부한 annotation을 매우 단순하게 추가**하는 편이 훨씬 효과적일 수 있음을 보여줍니다.

또한 이 논문은 weak supervision을 단순히 cheaper label의 문제가 아니라, **모델과 supervision granularity의 정합성** 문제로 바라보게 만듭니다. Implicit PointRend가 추가된 이유도 바로 이것입니다. 따라서 이 논문은 annotation design, loss design, architecture design이 함께 가야 한다는 점을 잘 보여주는 사례라고 볼 수 있습니다.  

## 6. Conclusion

이 논문은 instance segmentation을 위해 bounding box에 더해 **랜덤 포인트 몇 개만 object/background로 라벨링하는 point-based annotation scheme**을 제안하고, 이것만으로도 기존 full-mask용 모델을 구조 변경 없이 거의 fully-supervised 수준까지 학습시킬 수 있음을 보였습니다. 객체당 **10개 점**이라는 매우 작은 supervision으로 COCO, VOC, Cityscapes, LVIS에서 full mask 성능의 **94%–98%** 수준에 도달했고, annotation 시간은 polygon mask 대비 약 **5배 절감**됩니다.  

추가로 제안된 **Implicit PointRend**는 point supervision에 더 잘 맞는 간결한 implicit mask representation을 제공하며, sparse supervision에 맞는 architecture co-design의 필요성을 보여줍니다. 실무적으로는 “full mask를 다 만들 여유가 없는 새 데이터셋”에서 매우 가치가 크고, 연구적으로는 weakly-supervised instance segmentation의 강력한 baseline이자 기준점으로 남을 만한 논문입니다.  
