# Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth

이 논문은 proposal-free instance segmentation의 약점을 정면으로 겨냥합니다. 기존 proposal-based 방법, 특히 Mask R-CNN 계열은 정확하지만 느리고 mask 해상도가 낮으며, 반대로 dense prediction 기반 proposal-free 방법은 고해상도 mask와 빠른 실행 속도를 제공하지만 정확도가 부족했습니다. 이 논문은 이 간극을 메우기 위해, 각 픽셀이 물체 중심을 가리키는 **spatial embedding**을 예측하되, 모든 물체에 동일한 clustering 기준을 두지 않고 **instance별 clustering bandwidth(= sigma, margin)**를 함께 학습하는 새로운 clustering loss를 제안합니다. 핵심은 픽셀을 정확히 중심점 하나에 맞추도록 강제하는 대신, 각 물체에 대해 IoU를 최대화하는 “최적의 attraction region” 안으로 모이도록 학습시키는 것입니다. 논문은 Cityscapes에서 이 방법이 **27.6 AP**, **11 fps**를 달성해, Mask R-CNN의 **26.2 AP**를 넘으면서도 실시간 수준 속도를 제공한다고 주장합니다.

## 1. Paper Overview

이 논문이 해결하려는 문제는 **고정밀 instance segmentation을 실시간에 가깝게 수행하는 것**입니다. instance segmentation은 단순 detection과 달리 픽셀 단위 mask가 필요하므로 연산량이 크고, 특히 자율주행처럼 2MP급 해상도에서 빠른 추론이 필요한 환경에서는 기존 정확도 중심 방법이 실용적이지 않습니다. 논문은 “proposal-free가 원래 더 빠를 수 있는데, 왜 proposal-based보다 정확도가 낮은가?”라는 문제의식에서 출발합니다. 저자들은 그 원인이 후처리 clustering이 학습 목표와 분리되어 있어, 네트워크가 실제 instance mask 품질을 직접 최적화하지 못하기 때문이라고 봅니다.  

기존 중심 회귀(regression-to-centroid) 방식은 각 픽셀이 자기 instance 중심으로 정확히 이동하도록 학습하지만, 큰 물체의 가장자리 픽셀까지 동일 강도로 중심점에 맞추게 만드는 것은 비효율적입니다. 큰 물체일수록 가장자리 픽셀은 중심에서 멀기 때문에 학습이 कठोर해지고, inference 시에도 중심을 먼저 찾고 다시 clustering해야 해서 최종 instance quality가 loss에 직접 반영되지 않습니다. 이 논문은 이 단절을 없애기 위해, **clustering 자체를 differentiable한 형태로 loss 안에 집어넣는 방향**으로 설계를 바꿉니다.  

## 2. Core Idea

핵심 아이디어는 두 가지입니다.

첫째, 픽셀마다 예측하는 것은 단순한 offset vector만이 아닙니다. 각 픽셀은 자기 물체 중심 쪽으로 향하는 **offset**과 더불어, 그 물체에 적절한 clustering 범위를 나타내는 **sigma**도 함께 예측합니다. sigma가 크면 더 넓은 margin이 허용되고, sigma가 작으면 더 타이트하게 모이게 됩니다. 이렇게 하면 큰 물체는 넓은 attraction region을, 작은 물체는 좁은 region을 갖게 되어, 객체 크기에 따라 loss가 자동으로 조정됩니다.  

둘째, 이 sigma와 embedding을 이용해 각 instance에 대해 **Gaussian foreground/background probability map**을 만들고, 이를 Lovasz-hinge loss로 학습합니다. 즉, 직접 offset 오차를 줄이는 것이 아니라, “이 instance의 mask IoU가 커지도록” embedding과 sigma를 공동 최적화합니다. 논문이 강조하는 novelty는 바로 여기입니다. sigma와 offset은 직접 supervision을 받지 않고, Gaussian과 Lovasz-hinge를 통해 **instance mask 품질을 최대화하는 방향으로 간접 학습**됩니다.

이 방식의 직관은 Figure 1 설명에 잘 드러납니다. 픽셀들은 물체 중심의 한 점으로 정확히 수렴할 필요가 없고, **IoU를 가장 잘 만드는 적절한 중심 주변 영역** 안으로 모이면 충분합니다. 큰 물체일수록 이 영역이 넓어질 수 있으므로, edge pixel에 대한 loss가 완화됩니다. 이는 단순 regression 대비 훨씬 자연스러운 목표입니다.

## 3. Detailed Method Explanation

### 3.1 기본 formulation: spatial embedding

입력 이미지의 각 픽셀 좌표를 $x_i$라 하고, 네트워크가 예측한 offset을 $o_i$라 하면 spatial embedding은 다음처럼 정의됩니다.

$$
e_i = x_i + o_i
$$

기존 방식은 픽셀이 자기 instance centroid $C_k$를 정확히 가리키도록 학습합니다. centroid는

$$
C_k = \frac{1}{N}\sum_{x \in S_k} x
$$

로 정의되고, 표준 regression loss는

$$
\mathcal{L}_{regr} = \sum\_{i=1}^{n} |o_i - \hat{o}_i|
$$

이며 여기서

$$
\hat{o}_i = C_k - x_i
$$

입니다. 이 방식은 결국 “embedding이 centroid로 가야 한다”는 점별 제약만 줄 뿐, 실제 inference에서 필요한 centroid localization과 clustering은 loss 밖에 남겨 둡니다. 논문은 이것이 end-to-end instance optimization을 방해한다고 지적합니다.

### 3.2 Learnable margin

이 문제를 해결하기 위해 저자들은 고정 margin을 둔 hinge loss 관점으로 넘어갑니다. 직관적으로는 “embedding이 centroid에 정확히 일치할 필요는 없고, 어떤 반경 $\delta$ 안에만 들어오면 된다”는 생각입니다. 하지만 고정된 $\delta$는 작은 객체와 큰 객체를 동시에 다루기 어렵습니다. 그래서 논문은 각 instance마다 다른 margin이 필요하다고 보고, 이를 sigma로 학습하게 합니다.

구체적으로, instance $k$에 대해 embedding과 center of attraction 사이 거리로 Gaussian probability map $\phi_k$를 만들고, 이 probability map이 해당 instance의 foreground/background mask를 잘 구분하도록 학습합니다. 이때 sigma는 instance 내부 픽셀들의 sigma 예측을 평균해 얻습니다. 중요한 점은 sigma가 단순 보조값이 아니라, **test time에도 실제 clustering margin으로 사용된다**는 것입니다. 논문은 이것이 aleatoric uncertainty를 학습하되 test-time에 직접 쓰지 않는 일부 prior work와의 차이라고 설명합니다.  

### 3.3 Loss 설계와 Lovasz-hinge

이 논문의 가장 중요한 기술적 선택은 cross-entropy 대신 **Lovasz-hinge loss**를 사용하는 것입니다. 저자들은 Gaussian이 출력하는 instance별 foreground/background probability map에 대해 binary classification loss를 적용하되, class imbalance에 덜 민감하고 Jaccard/IoU와 직접 맞닿는 surrogate인 Lovasz-hinge를 택합니다. 그래서 최적화 목표가 본질적으로 **instance mask의 IoU 최대화**가 됩니다.

이 결과, sigma와 offset에 대한 직접 정답이 없어도 괜찮습니다. 네트워크는 backpropagation을 통해 “어떤 offset과 어떤 sigma를 내야 Gaussian mask가 GT mask와 IoU가 최대가 되는가”를 스스로 학습합니다. 이 점이 기존 direct regression보다 훨씬 목적함수-일치적(task-aligned)입니다.

### 3.4 Seed map과 center localization

centroid 기반 방법의 또 다른 문제는 “중심을 어떻게 찾을 것인가”입니다. 이 논문은 이를 위해 **semantic class별 seed map**을 예측합니다. Figure 2 설명에 따르면, seed map의 값이 높을수록 그 픽셀의 offset이 실제 object center를 정확히 가리킨다는 뜻입니다. 경계 쪽 픽셀은 어느 중심을 가리켜야 할지 불확실하므로 값이 낮고, 중심 근처 픽셀은 값이 높습니다. inference 시 cluster center는 이 seed map으로부터 뽑힙니다.

즉 전체 파이프라인은 다음처럼 이해할 수 있습니다.

1. 네트워크가 pixel-wise offset, sigma, seed map을 예측한다.
2. offset과 좌표를 합쳐 spatial embedding을 만든다.
3. seed가 높은 위치를 중심 후보로 사용한다.
4. 각 중심에 대해 sigma가 정하는 Gaussian margin으로 픽셀을 clustering한다.
5. 이로부터 instance mask를 복원한다.

### 3.5 Learnable center of attraction

저자들은 center를 꼭 기하학적 centroid로 둘 필요도 없다고 봅니다. **Center of Attraction (CoA)**를 centroid로 고정할 수도 있지만, 더 일반적으로는 동일 instance의 spatial embeddings 평균으로부터 얻는 **learnable center**를 사용할 수 있습니다. 논문은 이 learnable CoA가 fixed centroid보다 더 좋은 AP를 준다고 보고합니다. 해석하면, 네트워크가 실제 clustering에 유리한 중심 위치를 스스로 선택할 수 있다는 뜻입니다.

### 3.6 Circular vs. elliptical margin

sigma는 scalar일 수도 있고 2D일 수도 있습니다. scalar sigma는 원형(circular) margin을 만들고, 2D sigma는 축별로 다른 폭을 갖는 타원형(elliptical) margin을 만듭니다. 논문은 pedestrian처럼 세로로 길쭉한 물체에서는 원형 margin이 비효율적일 수 있으므로, elliptical margin이 더 적합하다고 설명합니다. 실제로 2-dimensional sigma가 scalar sigma보다 더 좋다고 보고합니다.  

## 4. Experiments and Findings

### 4.1 Dataset과 세팅

주요 평가는 **Cityscapes**에서 수행됩니다. 이 데이터셋은 2048×1024 해상도의 도시 장면 이미지 5,000장을 fine annotation과 함께 제공하며, instance segmentation은 8개 semantic class에 대해 region-level AP로 평가됩니다. 이 고해상도와 복잡한 장면 구조 때문에, 정확도뿐 아니라 속도까지 동시에 잡기 매우 어렵습니다.

### 4.2 주요 정량 결과

논문은 Cityscapes test set에서 자사 방법이 **27.6 AP**, **50.9 AP50**, **11 fps**를 기록했다고 제시합니다. 같은 맥락에서 Mask R-CNN은 약 **26.2 AP**, **1 fps** 수준으로 비교되며, Box2Pix는 **13.1 AP**, **10.9 fps**, discriminative loss 계열은 **17.5 AP**, **5 fps** 정도로 언급됩니다. 즉, 이 논문은 “실시간에 가까운 속도”와 “proposal-based 수준의 정확도”를 동시에 달성했다는 점을 핵심 성과로 내세웁니다.  

추론 시간 분석도 중요합니다. 논문은 2MP 이미지에서 **forward pass 65ms**, **clustering 26ms**라고 설명합니다. 이 수치는 제안한 post-processing이 너무 비싸지 않다는 점, 즉 proposal-free 접근의 장점을 유지하고 있음을 보여 줍니다.

### 4.3 Ablation: learnable sigma의 중요성

가장 인상적인 ablation은 **fixed sigma vs. learnable sigma** 비교입니다. 논문은 고정 sigma를 사용하면 약 **28 AP**, learnable sigma를 사용하면 **38.7 AP**까지 올라간다고 보고합니다. 이 차이는 매우 커서, 이 논문의 성능 향상이 단순히 offset learning 때문이 아니라 **instance-specific margin learning** 자체에서 온다는 점을 강하게 시사합니다.

또한 저자들은 sigma와 객체 크기 사이에 양의 상관관계가 있음을 Figure 4로 보여 줍니다. 즉, 큰 물체일수록 큰 sigma가, 작은 물체일수록 작은 sigma가 학습됩니다. 이는 제안 방식의 직관이 실제로 네트워크 내부에서 실현되고 있음을 보여 주는 좋은 증거입니다.  

### 4.4 Learnable CoA와 elliptical sigma

Fixed centroid보다 learnable CoA가 더 높은 AP를 보였고, scalar sigma보다 2D sigma(elliptical margin)가 더 좋은 성능을 냈습니다. 이 결과는 두 가지를 말해 줍니다. 첫째, “중심”도 고정 geometric notion이 아니라 task-optimal 위치로 학습하는 것이 유리합니다. 둘째, 물체는 원형이 아니므로 clustering region도 비등방적이어야 더 잘 맞습니다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **loss와 inference가 잘 정렬되어 있다는 점**입니다. 기존 proposal-free embedding 방법은 학습 시 feature를 끌어당기고 밀어내는 surrogate loss를 쓰고, test time에는 별도 clustering을 수행하는 경우가 많았습니다. 이 논문은 Gaussian mask + Lovasz-hinge를 통해 최종 instance IoU에 더 직접적으로 연결된 목적함수를 제시합니다.

또 다른 강점은 **속도-정확도 균형**입니다. 단순히 빠르기만 한 것이 아니라, 당시 Cityscapes에서 proposal-free 계열 중 드물게 Mask R-CNN을 정확도 면에서도 경쟁 또는 상회하는 수준으로 끌어올렸습니다. 이는 실시간 자율주행 응용에 매우 중요한 포인트입니다.  

마지막으로, 설계가 비교적 간결합니다. offset, sigma, seed라는 해석 가능한 출력들을 사용하므로, 모델이 무엇을 배우는지 직관적으로 이해하기 쉽습니다. sigma가 객체 크기와 양의 상관관계를 띠는 실험은 그 해석 가능성을 더 강화합니다.

### 한계

한계도 분명합니다. 첫째, 이 방법은 여전히 **center-pointing formulation**에 기반합니다. 매우 복잡한 형태의 객체, 겹침이 심한 장면, 중심이 불명확한 객체에서는 “한 중심을 향해 픽셀이 모인다”는 가정이 충분치 않을 수 있습니다. 논문은 이를 직접 길게 비판하진 않지만, 방법론 자체가 갖는 구조적 제약입니다.

둘째, seed map 기반 center extraction과 clustering은 proposal-free지만 여전히 일정한 후처리를 필요로 합니다. 논문이 이를 loss와 더 잘 연결한 것은 맞지만, 완전히 mask를 직접 출력하는 end-to-end dense mask prediction과는 다릅니다. 특히 crowded scenes에서 center candidate selection이 불안정하면 성능이 흔들릴 여지가 있습니다. 이 부분은 본문에서 정성적으로는 암시되지만, 충분한 failure analysis는 많지 않습니다.

셋째, 주요 실험이 Cityscapes 중심이라 범용성 검증은 제한적입니다. 이 방법이 COCO처럼 클래스와 객체 형태가 훨씬 다양한 데이터셋에서 동일한 장점을 유지하는지는 본 논문만으로는 판단하기 어렵습니다. 이는 논문의 실험 범위상 자연스러운 제한입니다.

### 해석

비평적으로 보면, 이 논문은 proposal-free instance segmentation이 왜 proposal-based보다 약했는지를 단순 backbone 문제가 아니라 **목적함수 설계 문제**로 본 점이 인상적입니다. “embedding을 잘 만들라”가 아니라 “instance IoU를 잘 만들라”로 초점을 옮겼고, 그 매개체로 learnable sigma를 도입한 것이 핵심입니다. 이후 panoptic/center-based segmentation 계열 연구를 볼 때도, 이 논문은 **center representation + uncertainty/margin learning**의 중요한 선구 사례로 읽을 수 있습니다.  

## 6. Conclusion

이 논문은 proposal-free instance segmentation에 대해, 픽셀 spatial embedding과 instance-specific clustering bandwidth를 **공동 최적화**하는 새로운 loss를 제안합니다. 핵심은 offset 회귀를 직접 감독하는 대신, Gaussian mask와 Lovasz-hinge를 이용해 **instance mask IoU를 직접 최적화**하는 것입니다. 또한 seed map으로 중심을 찾고, learnable sigma로 물체마다 다른 clustering region을 허용함으로써, 큰 물체와 작은 물체를 동시에 잘 다루게 합니다. Cityscapes에서 이 방법은 **27.6 AP**, **11 fps**를 달성하며, 당시 기준으로 “실시간 proposal-free + 높은 정확도”라는 강한 조합을 보여 줍니다.  

실무적으로는 자율주행처럼 고해상도 입력에서 latency가 중요한 환경에 의미가 크고, 연구적으로는 clustering-based instance segmentation에서 **learnable margin**과 **IoU-aligned training**의 중요성을 분명히 보여 준 논문입니다. 후속 center-based, embedding-based, panoptic segmentation 연구를 이해할 때도 상당히 중요한 연결고리 역할을 합니다.
