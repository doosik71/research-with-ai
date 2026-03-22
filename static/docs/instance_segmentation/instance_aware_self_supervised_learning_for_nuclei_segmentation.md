# Instance-aware Self-supervised Learning for Nuclei Segmentation

이 논문은 병리 영상(histopathological images)에서의 **nuclei instance segmentation** 문제를 다룬다. 저자들은 핵(nucleus)의 형태 변화가 크고, instance-level annotation 비용이 매우 높아 supervised segmentation 모델의 성능이 데이터 부족에 의해 제한된다는 점에 주목한다. 이를 해결하기 위해, 라벨 없이 얻을 수 있는 원시 영상으로부터 **핵의 크기(size)와 개수(quantity)에 대한 prior knowledge**를 스스로 학습하게 하는 **instance-aware self-supervised learning** 프레임워크를 제안한다. 구체적으로는 **scale-wise triplet learning**과 **count ranking**이라는 두 개의 proxy task를 설계해 backbone이 instance-aware representation을 먼저 학습하도록 만든 뒤, nuclei segmentation에 fine-tuning한다. 논문은 MoNuSeg에서 self-supervised ResUNet-101이 **AJI 70.63%**를 달성했다고 보고하며, 이를 당시 새로운 state of the art로 제시한다.  

## 1. Paper Overview

이 논문의 연구 문제는 단순한 semantic segmentation이 아니라, **서로 인접하거나 겹치는 개별 nuclei를 분리하여 instance 단위로 분할하는 것**이다. 병리 영상에서 nuclei instance segmentation은 종양 진단과 치료 관련 분석에서 중요하며, 세포 밀도나 위치뿐 아니라 morphology feature까지 제공할 수 있다. 하지만 핵의 경계는 복잡하고, 핵마다 크기와 모양이 크게 다르며, 수작업 annotation은 병리 전문가가 직접 contour를 그려야 하므로 매우 비싸다. 이런 이유로 충분한 annotated data 확보가 어렵고, supervised deep learning의 잠재력이 충분히 발휘되지 못한다.

저자들은 self-supervised learning이 의료영상 분야 여러 semantic segmentation 문제에서는 효과를 보였지만, **instance segmentation은 semantic segmentation과 질적으로 다른 문제**라고 본다. 픽셀이 어떤 class인지뿐 아니라, 같은 class 안에서도 “어느 개체(instance)에 속하는가”를 구분해야 하기 때문이다. 따라서 단순한 representation pretraining이 아니라, **instance를 의식하는 self-supervised task**가 필요하다는 것이 논문의 출발점이다.

## 2. Core Idea

논문의 핵심 아이디어는 매우 명확하다.
**핵 instance segmentation에 필요한 중요한 사전지식은 ‘핵의 크기’와 ‘핵의 개수’이며, 이 두 가지를 라벨 없이 self-supervised proxy task로 학습시키면 segmentation backbone이 더 좋은 instance-aware feature를 얻을 수 있다.**  

이를 위해 저자들은 하나의 histopathology image로부터 anchor, positive, negative patch를 만들고, 이들 사이의 관계를 이용해 두 가지 학습 신호를 정의한다.

첫째, **scale-wise triplet learning**은 서로 비슷한 핵 크기를 가진 patch는 embedding space에서 가깝게, 더 큰 핵을 포함하도록 조작된 patch는 멀어지게 만든다.
둘째, **count ranking**은 positive sample이 negative sample보다 더 많은 nuclei를 포함하도록 생성된다는 조작 규칙을 이용해, 모델이 nuclei quantity 차이를 반영하도록 만든다.

즉, 이 논문은 일반적인 “pretext task로 representation을 배우자” 수준을 넘어서, **nuclei instance segmentation에서 정말 필요한 구조적 prior를 proxy design에 직접 반영했다**는 점이 핵심이다.

## 3. Detailed Method Explanation

### 3.1 전체 흐름

전체 파이프라인은 2단계다.

1. 대량의 unlabeled histopathological image로 self-supervised pretraining 수행
2. pre-trained network를 nuclei instance segmentation task에 fine-tuning 수행

self-supervised pretraining 단계에서는 하나의 이미지로부터 triplet samples를 만들고, shared-weight encoder 세 개가 anchor, positive, negative를 각각 latent feature로 매핑한다. 이후 두 loss, 즉 **scale-wise triplet loss**와 **count ranking loss**로 encoder를 학습한다. 이때 encoder가 nuclei size와 quantity를 반영하는 표현을 배우도록 유도된다.  

### 3.2 Image Manipulation: Triplet Sample 생성 방식

저자들은 self-supervised signal을 만들기 위해 원본 병리 영상에서 규칙적으로 patch를 생성한다. 논문 설명에 따르면, MoNuSeg의 $1000 \times 1000$ 영상에서 먼저 $768 \times 768$ patch를 **anchor**로 잡는다. 그다음 인접한 같은 크기 patch를 **positive**로 사용한다. 이 positive는 anchor와 비슷한 nuclei size distribution을 가질 가능성이 높다. 마지막으로 positive 내부에서 더 작은 sub-patch를 랜덤 crop한 뒤 다시 $768 \times 768$로 resize하여 **negative**를 만든다. 이 negative는 원래 더 작은 영역을 확대했기 때문에, 결과적으로 더 큰 nuclei를 포함한 것처럼 보이게 된다. 저자들은 negative crop 크기를 ${512 \times 512, 256 \times 256, 128 \times 128, 64 \times 64}$ 중에서 랜덤 선택해 다양성을 확보한다.

이 조작은 매우 단순하지만 핵심을 잘 찌른다.
anchor와 positive는 비슷한 “평균 nuclei size”를 공유하고, negative는 더 큰 nuclei size를 갖도록 유도된다. 동시에 positive는 negative보다 더 많은 nuclei를 포함하는 경향이 생긴다. 따라서 한 번의 image manipulation으로 **size**와 **count**라는 두 종류의 supervision signal을 동시에 얻는다.

### 3.3 Proxy Task 1: Scale-wise Triplet Learning

첫 번째 proxy task는 **scale-wise triplet learning**이다. shared encoder가 anchor, positive, negative를 각각 feature vector $z_a$, $z_p$, $z_n$으로 인코딩하며, 논문에서는 이 feature가 128-dimensional이라고 설명한다.  

이후 triplet loss는 다음과 같이 정의된다.

$$
\mathcal{L}_{ST}(z_a, z_p, z_n)
===============================

\sum \max\left(0,\ d(z_a, z_p) - d(z_a, z_n) + m_1\right)
$$

여기서 $d(\cdot)$는 squared $L_2$ distance이고, margin $m_1$은 1.0으로 설정된다. 이 loss의 의미는 전형적인 triplet learning과 같다. 즉, anchor와 positive는 feature space에서 더 가깝게, anchor와 negative는 더 멀어지게 만든다. 하지만 여기서 class label 대신 사용되는 것은 semantic class가 아니라 **nuclei scale relationship**이다. 같은 scale로 crop된 샘플은 같은 “class”처럼, 다른 scale에서 유도된 샘플은 다른 “class”처럼 취급된다.

이 설계의 중요한 점은, 모델이 단순 texture나 color statistics가 아니라 **핵 크기와 연결된 형태적 차이**를 embedding에 반영하도록 강제한다는 것이다.

### 3.4 Proxy Task 2: Count Ranking

두 번째 proxy task는 **count ranking**이다. 논문은 positive sample이 negative보다 항상 더 많은 nuclei를 포함하도록 만들어졌다는 observation을 이용한다. 그래서 positive와 negative 간 순서 관계를 학습시키는 pair-wise ranking objective를 둔다. 즉, 모델은 단순히 “크기 차이”뿐 아니라 “핵 개수 차이”도 latent feature에 반영해야 한다.  

본문 snippet에서는 ranking loss의 완전한 수식 전체가 모두 보이지는 않지만, 논문의 설명은 분명하다. count ranking은 nuclei quantity 차이를 반영하는 self-supervised signal이고, scale-wise triplet learning과 함께 사용되어 **instance-awareness를 강화**한다. 즉, triplet loss가 크기 관련 구조를 embedding에 심는 역할이라면, ranking loss는 동일 patch family 내에서 **얼마나 많은 nuclei가 있는지에 대한 정렬 정보**를 주는 역할을 한다.

### 3.5 두 Proxy Task의 결합 의미

논문이 설득력 있는 이유는 두 task가 서로 보완적이기 때문이다.

* **Triplet learning**은 nuclei의 평균 크기와 scale difference를 반영한다.
* **Count ranking**은 nuclei quantity difference를 반영한다.  

nuclei instance segmentation에서 어려운 점은 경계 구분만이 아니라, crowded region에서 개체 수와 개별 object scale을 동시에 잘 파악해야 한다는 데 있다. 이 논문은 바로 그 점을 self-supervised task 설계에 반영했다. 결과적으로 backbone은 segmentation 전에 이미 nuclei instance에 대한 형태적 힌트를 내재화한 feature space를 얻게 된다.

### 3.6 Downstream Segmentation

논문은 self-supervised pretraining 후 이를 nuclei segmentation 모델의 initialization으로 사용한다. 결과 section에서 특히 **ResUNet-101**을 중심으로 성능 향상을 보고한다. 여기서 중요한 포인트는 proxy task 자체가 segmentation decoder를 직접 수행하는 것이 아니라, **encoder representation을 더 좋게 만드는 pretraining framework**라는 점이다. 즉, 방법론의 핵심 기여는 새로운 segmentation head가 아니라, instance segmentation에 특화된 **self-supervised pretraining strategy**다.  

## 4. Experiments and Findings

### 4.1 Dataset and Metric

주요 실험은 **MoNuSeg** dataset에서 수행된다. 논문 설명에 따르면 각 이미지는 $1000 \times 1000$ 픽셀이며, public training set을 80:20 비율로 train/validation으로 나눈다. 평가 지표는 **AJI (Aggregated Jaccard Index)**로, nuclei segmentation의 object-level matching을 반영하는 metric이라서 nuclei instance segmentation에 더 적합하다고 설명한다.

추가 검증으로는 **CPM (Computational Precision Medicine)** dataset도 사용한다. Table 4 caption에 따르면 5-fold cross validation을 수행했고, AJI 외에 CPM competition에서 쓰이는 Dice score도 함께 평가했다.

### 4.2 Main Results on MoNuSeg

논문의 가장 핵심 결과는 self-supervised pretraining이 nuclei instance segmentation 성능을 유의미하게 끌어올린다는 점이다. MoNuSeg test set에서 self-supervised ResUNet-101은 **AJI 70.63%**를 달성했다. 논문은 이것이 당시 SOTA라고 주장한다. 또한 challenge leaderboard 상위 팀들과 비교했을 때도 경쟁력 있는 결과라고 제시한다.  

특히 저자들은 이 성능 향상이 단순 미세한 개선이 아니라, self-supervised pretraining이 segmentation accuracy를 “remarkably boost”했다고 해석한다. 즉, nuclei annotation 부족 문제에서 representation pretraining이 실제로 큰 도움이 된다는 것을 보여준다.

### 4.3 Ablation Study

Ablation study는 논문의 중요한 근거다. Table 3 snippet에 따르면:

* baseline AJI: **65.29**
* triplet 계열 또는 일부 component 추가 시: **69.64**, **70.09**
* full SSL (Ours): **70.63**

이 결과는 두 가지를 보여준다.

첫째, self-supervised pretraining 자체가 큰 이득을 준다.
둘째, 두 proxy task를 함께 사용할 때 가장 좋다. 즉, nuclei size만 배우는 것도 유익하고, nuclei quantity까지 결합하면 추가 향상이 발생한다는 뜻이다.

논문 본문도 각 component가 accuracy improvement를 낳는지 확인하기 위해 ablation을 수행했다고 명시한다.

### 4.4 Validation on CPM

추가 dataset인 CPM에서는 self-supervised ResUNet-101이 **average Dice 86.36%**를 달성했다고 보고한다. 이는 CPM 2018 competition winner의 87.00%와 comparable하다고 논문은 해석한다. 또한 표 snippet에서는 fold별 결과와 평균 성능이 제시되며, AJI와 Dice 모두 양호한 generalization을 보인다.  

이 결과는 제안 방법이 MoNuSeg에만 과적합된 것이 아니라, 다른 nuclei segmentation dataset에도 어느 정도 일반화된다는 근거로 볼 수 있다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 **proxy task 설계가 문제 구조와 직접 맞닿아 있다**는 점이다. 많은 self-supervised learning 논문이 rotation prediction, jigsaw, contrastive task처럼 비교적 일반적인 pretext task를 쓰는 반면, 이 논문은 nuclei segmentation에 실제로 중요한 **size**와 **count**를 supervision signal로 삼았다. 이것은 task-specific self-supervision의 좋은 예다.

또한 image manipulation이 단순하고 직관적이다. 별도의 추가 annotation 없이도 하나의 원본 이미지에서 training signal을 만들어낼 수 있고, triplet과 ranking이라는 안정적인 학습 프레임워크 위에 올려져 있어 구현 관점에서도 비교적 명쾌하다.

실험 측면에서는 MoNuSeg SOTA claim, ablation, CPM validation까지 제시해, “정말 두 sub-task가 필요한가”와 “다른 dataset에도 통하나”를 함께 검증했다는 점이 좋다.

### Limitations

한계도 분명하다. 우선 이 방법은 nuclei segmentation이라는 매우 특수한 문제에 맞춰져 있다. negative sample을 “smaller crop 후 resize”로 만드는 방식은 nuclei size prior를 잘 반영하지만, 다른 종류의 instance segmentation에 그대로 일반화되기는 쉽지 않을 수 있다. 즉, 이 접근의 강점은 task-specific함이지만 동시에 범용성의 한계이기도 하다. 이는 논문 본문 설명으로부터 합리적으로 도출되는 해석이다.

또한 self-supervised task가 실제 nuclei morphology를 완전히 반영한다고 보장할 수는 없다. 예를 들어 crop-resize 조작이 nuclei size difference뿐 아니라 texture distortion이나 scale artifact도 함께 만들 수 있다. 모델이 일부는 이런 artifact를 학습했을 가능성도 있다. 논문 snippet만으로는 이에 대한 정밀한 분석은 충분히 보이지 않는다.

마지막으로, 이 논문은 instance-aware representation을 학습하는 데 초점을 맞추고 있으나, crowded nuclei의 boundary separation이 왜 정확히 개선되는지에 대한 feature-level 해석은 제한적이다. 즉, 성능 향상은 명확하지만 “무엇을 얼마나 배웠는가”에 대한 representation analysis는 상대적으로 적다.

### Interpretation

그럼에도 이 논문은 중요한 메시지를 던진다.
**의료영상 self-supervised learning은 generic proxy보다 domain-specific proxy가 더 강력할 수 있다.**
특히 nuclei segmentation처럼 annotation이 비싸고 구조적 prior가 강한 문제에서는, 데이터 증강 기반 contrastive learning보다 이런 구조화된 pretext task가 더 직접적으로 유효할 수 있다.

또한 “instance segmentation에 대한 self-supervised learning에 초점을 둔 첫 작업”이라고 저자들이 주장하는 점도 의미가 있다. 당시 문맥에서 이 논문은 semantic segmentation 중심이었던 medical SSL을 instance-aware 방향으로 확장한 선구적 시도로 볼 수 있다.

## 6. Conclusion

이 논문은 nuclei instance segmentation의 annotation bottleneck 문제를 해결하기 위해, **instance-aware self-supervised learning** 프레임워크를 제안했다. 핵심은 **scale-wise triplet learning**과 **count ranking** 두 sub-task를 통해 nuclei의 크기와 개수에 대한 prior를 라벨 없이 학습시키는 것이다. 이 pretraining은 downstream segmentation 성능을 크게 높였고, self-supervised ResUNet-101은 MoNuSeg에서 **AJI 70.63%**를 기록했다.  

실무적 관점에서 이 연구는 병리 영상처럼 annotation cost가 큰 분야에서 특히 의미가 있다. 미래 연구에서는 이 아이디어를 더 일반적인 medical instance segmentation, 또는 contrastive learning/transformer 기반 framework와 결합하는 방향으로 확장할 수 있을 것이다. 논문 결론에서도 저자들은 제안한 proxy가 nuclei size와 quantity knowledge를 암묵적으로 학습하게 하며, 이를 통해 segmentation accuracy를 향상시킨다고 정리한다.
