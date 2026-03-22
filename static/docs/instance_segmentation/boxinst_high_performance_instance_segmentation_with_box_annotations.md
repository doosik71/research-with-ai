# BoxInst: High-Performance Instance Segmentation with Box Annotations

## 1. Paper Overview

이 논문은 **mask annotation 없이 bounding box annotation만으로 high-quality instance segmentation을 학습할 수 있는가**라는 문제를 다룹니다. 저자들은 기존의 box-supervised 또는 weakly supervised segmentation 방법들이 대체로 pseudo mask 생성, iterative refinement, proposal generation 같은 복잡한 파이프라인에 의존했고, 특히 COCO 같은 대규모 벤치마크에서는 성능이 충분히 강하지 못했다는 점을 문제로 제기합니다. 이에 대해 BoxInst는 네트워크 구조 자체를 복잡하게 바꾸지 않고, **mask supervision loss만 새롭게 설계**함으로써 box-only supervision으로도 강한 instance mask를 학습할 수 있음을 보입니다. 논문은 COCO에서 기존 reported best인 21.1% mask AP를 31.6%로 크게 끌어올렸고, ResNet-101과 3x schedule에서는 test-dev 기준 33.2% mask AP를 달성했다고 보고합니다.  

이 문제가 중요한 이유는 분명합니다. instance segmentation은 object detection보다 훨씬 더 정밀한 위치 정보를 제공하지만, pixel-wise mask annotation은 box annotation보다 훨씬 비싸고 시간이 많이 듭니다. 논문은 이 annotation bottleneck 때문에 실제로는 box detection이 더 널리 쓰인다는 점을 강조하며, box만으로 mask 수준의 supervision을 대체할 수 있다면 annotation cost를 크게 낮추면서도 더 정밀한 vision system을 구축할 수 있다고 봅니다.

## 2. Core Idea

BoxInst의 핵심 아이디어는 매우 선명합니다.

**“mask annotation이 없더라도, box annotation만으로 mask를 간접적으로 제약하는 loss를 설계하면 충분히 좋은 mask를 학습할 수 있다.”**

이를 위해 저자들은 CondInst 기반 instance segmentation 모델은 그대로 유지하고, 기존 pixel-wise mask loss를 다음 두 항으로 바꿉니다.

1. **Projection loss**
2. **Pairwise affinity loss**

Projection loss는 예측한 mask의 x축, y축 projection이 ground-truth bounding box의 projection과 일치하도록 강제합니다. 직관적으로 말하면, **예측 mask를 감싸는 가장 타이트한 박스가 GT box와 같아지도록 만드는 제약**입니다. 하지만 이 제약만으로는 같은 박스를 만드는 mask가 여러 개 가능하므로 충분하지 않습니다. 그래서 두 번째로, pairwise affinity loss를 도입해 **가까운 픽셀들 사이의 label consistency**를 학습시킵니다. 이때 핵심 prior는 “서로 가깝고 색이 비슷한 픽셀은 같은 레이블일 가능성이 높다”는 것입니다. 저자들은 이 prior를 이용해 noisy supervision을 줄이고, box만으로도 mask 경계를 꽤 정확하게 복원합니다.

이 논문의 진짜 참신성은 “box supervision을 위해 별도 pseudo-mask 생성기를 얹는 것”이 아니라, **loss 자체를 재설계해서 mask supervision을 우회**했다는 데 있습니다. 따라서 inference는 CondInst와 동일하고, 학습 시 supervision 방식만 달라집니다. 이 점이 단순하면서도 강력한 설계입니다.

## 3. Detailed Method Explanation

### 3.1 기반 모델: CondInst

BoxInst는 **CondInst**를 기반으로 합니다. CondInst는 RoI-free fully convolutional instance segmentation 프레임워크로, 각 instance마다 동적으로 생성되는 filter를 통해 mask head를 instance-aware하게 바꿉니다. 이 구조는 Mask R-CNN처럼 RoI 내부에서만 mask를 예측하는 것이 아니라, **full-image mask prediction**을 수행할 수 있고, 논문은 이 점이 box-supervised setting에서 특히 중요하다고 말합니다. mask head는 class-agnostic하며, class는 detector branch가 예측합니다.

BoxInst의 중요한 특징은 **CondInst의 네트워크 구조를 전혀 바꾸지 않는다**는 점입니다. 즉, model architecture는 그대로 두고 supervision signal만 바꿉니다. 결과적으로 inference 과정은 CondInst와 동일하며, 추가 구조 복잡도 없이 weak supervision을 달성합니다.

### 3.2 Projection Loss

첫 번째 loss는 **projection loss**입니다.
논문은 ground-truth box 안을 1, 밖을 0으로 두는 binary mask $\mathbf{b} \in {0,1}^{H \times W}$를 정의하고, 이 box mask의 x/y축 projection을 GT의 1차원 label로 사용합니다. 예측 mask도 동일하게 각 축으로 projection한 뒤, 이 둘이 같아지도록 학습합니다. 논문은 이를 다음처럼 설명합니다.

* $\mathrm{Proj}_x(\mathbf{b}) = \mathbf{l}_x$
* $\mathrm{Proj}_y(\mathbf{b}) = \mathbf{l}_y$

그리고 projection 연산은 축 방향 max operation으로 구현됩니다. 즉,

$$
\mathrm{Proj}_x(\mathbf{b}) = \max_y(\mathbf{b}) = \mathbf{l}_x
$$

$$
\mathrm{Proj}_y(\mathbf{b}) = \max_x(\mathbf{b}) = \mathbf{l}_y
$$

라는 형태입니다.

이 loss의 의미는 직관적으로 매우 좋습니다.
예측 mask가 GT box 내부의 어느 정도 영역을 채우더라도, 최소한 그 mask의 외곽이 GT box와 정렬되도록 강제합니다. 다시 말해, “예측한 foreground가 GT box의 범위를 벗어나거나, GT box보다 과도하게 작게 수축되지 않도록” 제약합니다.

하지만 projection loss만으로는 부족합니다. 예를 들어 box 내부를 거의 다 채운 mask도, 물체의 얇은 부분만 남긴 mask도 projection만 보면 동일할 수 있습니다. 논문이 명시적으로 지적하듯, **여러 mask가 동일한 box projection을 가질 수 있기 때문**입니다. 따라서 추가 제약이 필요합니다.

### 3.3 Pairwise Affinity Loss

이 부족함을 보완하는 것이 **pairwise affinity loss**입니다.
핵심 아이디어는 **가까운 픽셀 쌍이 같은 레이블을 가질 가능성**을 supervision으로 이용하는 것입니다. 특히 저자들은 “가깝고 색이 유사한 proximal pixel pair는 같은 category label일 가능성이 높다”는 prior를 활용합니다. 단순히 모든 인접 픽셀 쌍에 동일 레이블을 강제하면 supervision noise가 커지므로, **색 유사도 threshold를 이용해 confident pair만 loss 계산에 포함**합니다. 그림 설명에 따르면 각 픽셀은 dilation rate 2 기준 8개 이웃과 edge를 구성하고, 그중 신뢰 가능한 pair만 사용합니다.

이 설계는 BBTP와의 차이를 잘 보여줍니다. BBTP도 pairwise term을 사용하지만, 논문은 BBTP가 단순히 spatially adjacent pixel pair를 같은 label로 묶는 oversimplified assumption을 써서 noise가 심했다고 비판합니다. 반면 BoxInst는 **color prior를 이용해 supervision noise를 크게 줄였다**고 주장합니다. 그리고 Table 1a에서 noisy supervision이 실제로 accuracy를 해칠 수 있음을 보였다고 설명합니다.

### 3.4 왜 이 두 손실의 조합이 중요한가

두 loss는 서로 보완적입니다.

* **Projection loss**는 global shape constraint를 제공합니다.
  최소한 mask의 외곽 범위가 GT box와 맞도록 만듭니다.
* **Pairwise affinity loss**는 local smoothness와 boundary consistency를 제공합니다.
  비슷한 색의 가까운 픽셀들이 같은 label이 되도록 유도해 세밀한 경계를 만듭니다.

Projection만 있으면 내부 구조가 모호하고, pairwise만 있으면 global extent가 모호해질 수 있습니다. BoxInst는 이 둘을 합쳐 **global box consistency + local appearance consistency**를 동시에 만족시키는 weak supervision을 구성합니다. 이것이 논문의 핵심 method insight입니다.

### 3.5 구조적 장점

이 방법의 구조적 장점은 다음과 같습니다.

첫째, **network modification이 없다**는 점입니다. CondInst의 architecture를 그대로 쓰므로, inference 비용 증가나 구현 복잡도가 적습니다. 논문은 “inference process of BoxInst is exactly the same as CondInst”라고 명확히 말합니다.

둘째, **proposal generation이나 iterative training이 필요 없다**는 점입니다. 기존 BoxSup, Box2Seg, SDI 같은 방법들은 MCG, GrabCut, iterative refinement 등에 의존했는데, BoxInst는 이를 제거합니다. 따라서 training pipeline이 단순합니다.

셋째, **RoI-free full-image mask prediction**이 box-supervised setting에 잘 맞습니다. CondInst가 full-image instance mask를 직접 만들 수 있기 때문에, projection/pairwise 형태의 supervision을 자연스럽게 적용할 수 있습니다.

## 4. Experiments and Findings

### 4.1 COCO에서의 핵심 결과

논문은 COCO에서 매우 강한 결과를 제시합니다.

* 기존 reported best weakly/box-supervised mask AP: **21.1%**
* BoxInst: **31.6%**
* ResNet-101 + 3x schedule, test-dev: **33.2% mask AP**
* fully supervised counterpart는 같은 설정에서 **39.1% mask AP**

즉, 완전 supervision과의 격차는 여전히 남아 있지만, **box-only supervision으로도 gap을 상당히 좁혔다**는 것이 논문의 가장 중요한 실험 결론입니다.

더 흥미로운 점은, BoxInst가 같은 backbone을 쓰는 일부 fully supervised one-stage 방법보다도 높은 성능을 보였다는 점입니다. 논문은 R-101, 3x setting에서 YOLACT 31.2% AP, PolarMask 32.1% AP를 언급하며, BoxInst의 33.2%가 이들을 넘어선다고 주장합니다. 즉, **약한 supervision인데도 일부 완전 supervision 방법보다 낫다**는 것입니다.

### 4.2 기존 weakly supervised 방법과 비교

논문은 BBTP 대비 absolute **10% AP** 향상을 강조합니다. 이 성능 차이는 단순 백본 차이보다도, supervision noise를 어떻게 처리하는지에서 나온다고 해석하는 것이 적절합니다. BBTP는 pairwise prior를 너무 거칠게 사용했지만, BoxInst는 color similarity를 이용해 더 신뢰도 높은 pixel pair만 loss에 반영합니다. 저자들이 반복해서 강조하는 것도 바로 이 점입니다.

### 4.3 Semi-supervised 확장

논문은 Box-only supervision에 일부 mask annotation을 섞는 **semi-supervised instance segmentation**도 다룹니다. 제공된 본문 조각에 따르면, partial mask annotation을 함께 쓰면 성능이 더 올라가며, unseen classes에서도 29.6%에서 30.9%로 개선되었다고 보고합니다. 이는 BoxInst가 단지 완전한 weak supervision 전용 기법이 아니라, **불완전 annotation 환경 전반에서 유용한 loss design**이라는 점을 보여줍니다.

### 4.4 Character segmentation 확장

논문은 BoxInst의 일반성을 보이기 위해 **ICDAR 2019 ReCTS**에서 character box annotation만으로 character mask를 생성하는 실험도 수행합니다.
데이터셋 규모는 20K training images, 5K testing images이며, test set에는 mask GT가 없어 AP는 보고하지 못하고 qualitative result만 제시합니다. 하지만 저자들은 BoxInst가 high-quality character mask를 만들어 arbitrary-shape text detection/recognition에 도움이 될 수 있다고 주장합니다. 이는 BoxInst가 단순히 COCO instance segmentation 전용이 아니라, **box-supervised dense labeling 문제 전반으로 확장 가능**함을 시사합니다.  

## 5. Strengths, Limitations, and Interpretation

### Strengths

가장 큰 강점은 **단순성**입니다.
네트워크를 새로 설계하지 않고 loss만 바꿔서 강한 성능을 얻었다는 점은 매우 설득력이 있습니다. 구현과 재현 측면에서도 장점이 큽니다.

두 번째 강점은 **성능 대비 가정의 경제성**입니다.
BoxInst는 mask GT 없이도 COCO에서 31.6~33.2 AP를 달성해, weak supervision이 생각보다 훨씬 강력할 수 있음을 보여줍니다. 이는 annotation-efficient learning 관점에서 의미가 큽니다.

세 번째 강점은 **prior의 사용 방식이 적절하다**는 점입니다.
단순 adjacency prior가 아니라 color similarity를 활용해 confident pixel pair만 쓰는 설계는 매우 실용적이고, 노이즈를 줄이는 방향으로 잘 정리되어 있습니다.

### Limitations

첫째, projection loss의 본질적 한계는 논문도 인정합니다.
projection만으로는 동일한 box에 대응하는 mask가 여러 개 존재할 수 있습니다. 즉, 전역 제약만으로는 정답 mask를 유일하게 결정할 수 없습니다. 그래서 pairwise term이 필수입니다. 이는 곧, BoxInst의 성공이 상당 부분 **appearance prior의 품질**에 의존한다는 뜻이기도 합니다.

둘째, pairwise supervision은 어디까지나 heuristic prior입니다.
“가깝고 색이 비슷하면 같은 레이블일 가능성이 높다”는 가정은 일반적으로 맞지만, 반사, 그림자, texture repetition, camouflage 같은 상황에서는 깨질 수 있습니다. 논문은 thresholding으로 noise를 줄이지만, 이 prior가 본질적으로 완전한 supervision을 대체하는 것은 아닙니다.

셋째, fully supervised 성능과의 차이는 여전히 존재합니다.
R-101, 3x setting에서 33.2 AP 대 39.1 AP라는 격차는 작지 않습니다. 따라서 BoxInst는 “mask annotation이 전혀 필요 없다”기보다, **annotation cost를 크게 줄이면서도 꽤 강한 baseline을 제공한다**고 보는 편이 정확합니다.

### Interpretation

비판적으로 해석하면, BoxInst의 가장 중요한 공헌은 새로운 segmentation architecture라기보다 **weak supervision을 위한 loss engineering의 힘**을 보여준 데 있습니다.
즉, 모델 자체보다도 “어떤 indirect constraint가 mask 학습에 충분한가”라는 질문에 좋은 답을 제시한 논문입니다.

또한 이 논문은 weak supervision이 pseudo-label generator나 iterative pipeline에 반드시 의존할 필요가 없고, 잘 설계된 differentiable loss만으로도 상당한 수준까지 갈 수 있음을 보여줍니다. 이 관점은 이후 box-, point-, scribble-supervised segmentation 연구를 이해하는 데도 중요한 기준점이 됩니다.

## 6. Conclusion

BoxInst는 **box annotation만으로 high-performance instance segmentation을 달성하기 위해, 기존 pixel-wise mask loss를 projection loss와 pairwise affinity loss로 대체한 방법**입니다. 핵심은 CondInst 같은 강한 fully convolutional instance segmentation framework 위에서, GT mask 없이도 학습 가능한 간접 supervision을 설계했다는 데 있습니다. Projection loss는 예측 mask의 외곽이 GT box와 일치하도록 만들고, pairwise affinity loss는 proximal/color-similar pixels의 local consistency를 통해 mask 경계를 정교화합니다. 그 결과 COCO에서 weak supervision의 성능을 크게 끌어올렸고, 일부 fully supervised one-stage 방법보다도 강한 결과를 보였습니다.

실무적으로는 **mask annotation 비용이 너무 큰 환경**에서 특히 유의미하며, 연구적으로는 **annotation-efficient dense prediction**과 **indirect supervision via loss design**의 좋은 사례로 볼 수 있습니다.
