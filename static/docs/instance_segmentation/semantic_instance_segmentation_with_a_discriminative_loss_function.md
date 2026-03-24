# Semantic Instance Segmentation with a Discriminative Loss Function

* **저자**: Bert De Brabandere, Davy Neven, Luc Van Gool
* **발표연도**: 2017
* **arXiv**: <https://arxiv.org/abs/1708.02551>

## 1. 논문 개요

이 논문은 semantic segmentation과 instance segmentation 사이의 간극을 매우 단순한 방식으로 메우려는 시도다. 저자들은 각 픽셀을 클래스 확률로 분류하는 대신, 각 픽셀을 feature space의 임베딩 벡터로 보낸 뒤, 같은 instance에 속한 픽셀은 서로 가깝게, 다른 instance의 픽셀은 충분히 멀어지도록 학습시키는 discriminative loss를 제안한다. 이렇게 학습된 출력은 복잡한 detection head, proposal generator, recurrent decoder 없이도 간단한 clustering 후처리만으로 instance mask로 변환될 수 있다는 것이 논문의 핵심 주장이다. 논문은 이 접근이 variable number of instances와 permutation invariance라는 instance segmentation의 본질적 난점을 자연스럽게 다룬다고 설명한다.

연구 문제가 중요한 이유는, 일반 semantic segmentation은 같은 클래스의 서로 다른 객체를 하나의 영역으로 합쳐 버리지만, 실제 응용에서는 객체별 분리가 필요하기 때문이다. 자율주행에서는 여러 대의 차량과 보행자를 각각 분리해야 하고, 식물 phenotyping에서는 잎 하나하나를 세어야 하며, 산업·의료 영상에서는 서로 가려진 개체까지 분리할 필요가 있다. 저자들은 기존 softmax 기반 segmentation이 이 문제에 잘 맞지 않는다고 본다. 한 이미지 안의 instance 수가 고정되어 있지 않고, instance label 자체가 순열 불변이기 때문에 “1번 객체, 2번 객체” 식의 고정 클래스화가 부자연스럽기 때문이다.

이 논문의 기여는 새로운 거대한 네트워크 구조를 제안했다기보다, semantic segmentation용 off-the-shelf backbone을 거의 그대로 재사용하면서 loss function만 바꿔 instance segmentation 문제를 풀 수 있음을 보인 데 있다. 즉, 구조적 복잡성을 늘리지 않고 목적 함수를 바꿔 문제를 재정의한 논문이라고 보는 것이 적절하다.

## 2. 핵심 아이디어

중심 직관은 매우 명확하다. 픽셀마다 $n$차원 embedding을 출력하게 하고, 같은 object instance에 속한 픽셀 embedding은 하나의 cluster를 이루게 만들며, 서로 다른 instance의 cluster center들은 넉넉한 margin을 두고 떨어지게 만든다. 그러면 추론 시에는 복잡한 proposal selection이나 sequence decoding 없이, embedding space에서 cluster를 찾는 것만으로 instance를 복원할 수 있다. 저자들은 이를 “pixel-level metric learning”에 가까운 관점으로 설명한다. 기존 siamese network나 triplet loss가 이미지 간 거리를 학습했다면, 이 논문은 그것을 이미지 내부의 픽셀들 사이 관계로 옮겨왔다고 볼 수 있다.

기존 접근과의 차별점은 세 가지로 정리할 수 있다. 첫째, proposal-based method처럼 bounding box나 object proposal에 의존하지 않는다. 둘째, recurrent instance segmentation처럼 instance를 순차적으로 하나씩 생성하지 않는다. 셋째, center direction, depth ordering, bounding box coordinate prediction처럼 출력 표현을 사람이 강하게 설계한 ad-hoc representation으로 제한하지 않는다. 네트워크는 그저 embedding space를 잘 조직하면 되고, 그 조직 원리는 loss가 정한다. 이 점에서 논문은 “표현 자체를 직접 규정하지 않고, 구분 가능한 구조만 요구하는 loss 중심 설계”라는 점이 특징이다.

또 하나 중요한 아이디어는 inter-cluster 분리를 모든 픽셀 쌍에 대해 계산하지 않고, 각 cluster의 평균 벡터인 cluster center들 사이에서만 계산했다는 점이다. instance segmentation에서는 픽셀 수는 매우 많지만 instance 수는 상대적으로 적으므로, 서로 다른 label의 모든 픽셀쌍을 직접 밀어내는 방식보다 훨씬 계산 효율적이다. 즉, “픽셀을 중심으로 모으고, 중심끼리만 밀어낸다”는 구조가 계산량과 최적화 단순성 모두를 잡는 설계다.

## 3. 상세 방법 설명

전체 파이프라인은 다음과 같다. 입력 이미지를 semantic segmentation용 backbone network에 통과시키고, 마지막 출력은 픽셀별 class logit 대신 픽셀별 embedding vector로 만든다. 학습 중에는 ground-truth instance mask를 이용해 어떤 픽셀들이 같은 cluster에 속해야 하는지 알 수 있으므로, 이 정보를 바탕으로 discriminative loss를 계산한다. 추론 시에는 네트워크가 만든 embedding map을 clustering하여 개별 instance mask를 얻는다. Cityscapes처럼 multi-class instance segmentation에서는 semantic class mask를 먼저 얻고, 클래스별로 instance clustering을 독립적으로 수행한다. 즉, 같은 클래스 내부의 instance들만 embedding space에서 서로 분리하도록 학습·추론한다.

### 3.1 Discriminative loss의 구조

논문은 손실 함수를 세 항으로 구성한다. 첫째는 variance term $L_{var}$이고, 같은 instance 내부의 픽셀 embedding을 그 instance의 평균 embedding인 cluster center $\mu_c$ 쪽으로 끌어당긴다. 둘째는 distance term $L_{dist}$이고, 서로 다른 instance의 cluster center들이 충분히 멀어지도록 민다. 셋째는 regularization term $L_{reg}$이고, 전체 cluster center가 원점에서 너무 멀리 떠나지 않도록 activation을 완만하게 제한한다. 최종 손실은 세 항의 가중합이다.

논문에 제시된 식은 다음과 같다.

$$
\begin{align}
L_{var} &= \frac{1}{C} \sum_{c=1}^{C} \frac{1}{N_c} \sum_{i=1}^{N_c} \left[ \lVert \mu_c - x_i \rVert - \delta_v \right]_+^2 \\
L_{dist} &= \frac{1}{C(C-1)} \sum_{c_A=1}^{C} \sum_{c_B=1,, c_A \neq c_B}^{C} \left[ 2\delta_d - \lVert \mu_{c_A} - \mu_{c_B} \rVert \right]_+^2 \\
L_{reg} &= \frac{1}{C} \sum_{c=1}^{C} \lVert \mu_c \rVert \\
L &= \alpha L_{var} + \beta L_{dist} + \gamma L_{reg}
\end{align}
$$

여기서 $C$는 이미지 내 instance 수, $N_c$는 instance $c$의 픽셀 수, $x_i$는 픽셀 embedding, $\mu_c$는 해당 instance의 평균 embedding이다. $\lVert \cdot \rVert$는 $L_1$ 또는 $L_2$ distance이며, $[x]_+ = \max(0, x)$는 hinge 연산이다. 실험에서는 $\alpha = 1$, $\beta = 1$, $\gamma = 0.001$을 사용한다.

이 식을 쉬운 말로 풀면 다음과 같다. 같은 instance의 픽셀이 자기 중심에서 $\delta_v$ 이내에 들어오면 더 이상 벌점을 주지 않는다. 즉, 무조건 한 점에 collapse시킬 필요는 없다. 그래서 cluster 내부가 작은 manifold처럼 퍼져 있어도 된다. 반대로 두 cluster center의 거리가 $2\delta_d$보다 충분히 크면 더 이상 추가로 밀어내지 않는다. 이것도 필요 이상으로 feature space를 왜곡하지 않기 위한 장치다. 논문은 이런 hinge 구조가 네트워크에 더 큰 representational freedom을 제공한다고 주장한다. 즉, “같은 것은 적당히 모으고, 다른 것은 충분히만 떼어놓는다”는 large-margin clustering 철학이다.

### 3.2 Softmax loss와의 비교

저자들이 특히 강조하는 비교는 softmax cross-entropy와의 차이다. softmax 분류는 각 픽셀을 미리 정해진 클래스 축 중 하나로 보내는 구조라서, 출력 차원이 클래스 수와 직접 연결된다. 그러나 instance segmentation에서는 이미지마다 instance 수가 달라지고, label identity는 permutation-invariant하다. 반면 이 논문의 embedding loss는 출력 차원이 instance 수와 무관하다. 다시 말해, “몇 개의 instance가 나올지 미리 정하지 않아도 되는 feature space”를 만든다. 이것이 instance segmentation에 더 자연스럽다는 것이 저자들의 논지다.

### 3.3 추론 및 후처리

이 논문의 후처리는 loss 설계와 직접 연결되어 있다. 학습이 이상적으로 이루어져 $L_{var}$와 $L_{dist}$가 충분히 작다면, 모든 픽셀은 자기 cluster center에서 $\delta_v$ 안에 있고, 서로 다른 cluster center들은 최소 $2\delta_d$ 이상 떨어져 있다. 따라서 $\delta_d > \delta_v$이면, 각 embedding은 자기 중심에 더 가깝고 타 instance 중심에는 더 멀다. 이 조건을 이용해 반경 $b = \delta_v$인 hypersphere thresholding으로 같은 instance 픽셀을 선택할 수 있다. 논문의 식은 다음과 같다.

$$
x_i \in C \iff \lVert x_i - x_c \rVert < b
$$

여기서 $x_c$는 기준이 되는 중심 embedding이다.

하지만 instance segmentation에서는 semantic class처럼 고정된 class center를 데이터셋 전체에서 저장해 둘 수 없다. 이미지마다 instance가 달라지고 label 순서도 의미가 없기 때문이다. 그래서 저자들은 unlabeled pixel 하나를 고르고, 그 픽셀 embedding 근처를 thresholding해서 하나의 instance를 찾고, 찾은 픽셀들을 같은 label로 묶은 다음, 아직 할당되지 않은 픽셀에 대해 같은 작업을 반복하는 절차를 제안한다. 즉, seed-based clustering이다.

현실에서는 test loss가 0이 아니므로 outlier 때문에 한 instance가 둘로 쪼개지는 문제가 생길 수 있다. 이를 줄이기 위해 논문은 mean-shift의 빠른 변형을 사용한다. 처음에 랜덤한 unlabeled pixel로 thresholding하고, 선택된 embedding들의 평균을 다시 계산한 뒤, 그 평균 주변으로 다시 thresholding한다. 이를 mean convergence까지 반복하면 density가 높은 진짜 cluster center로 이동할 가능성이 커진다. 즉, 단순 thresholding에 mean update를 추가해 outlier robustness를 높인 것이다.

### 3.4 장단점에 대한 방법론적 해석

저자들은 이 방법이 detect-and-segment 계열보다 복잡한 occlusion에 강하다고 주장한다. bounding box만으로는 객체 mask를 모호성 없이 복원하기 어려운 경우가 있기 때문이다. 예를 들어 막대 두 개가 X자 형태로 겹친 경우, 박스는 크게 중첩되며 박스 안에서 어느 픽셀이 어느 객체에 속하는지 판단하기 어렵다. 반면 이 논문은 이미지를 holistic하게 보고 픽셀 embedding을 학습하므로, 박스 prior 없이도 이런 상황을 더 자연스럽게 다룰 수 있다고 본다. 반대로, 장면 구성이 매우 다양하고 같은 객체가 예기치 않게 여러 개 등장하는 VOC/COCO 스타일 데이터셋에는 sliding-window detection 계열이 더 적합할 수 있다고 스스로 인정한다. 이 self-critique는 논문의 균형감을 높여 준다.

## 4. 실험 및 결과

논문은 두 개의 대표 벤치마크에서 실험한다. 하나는 CVPPP leaf segmentation이고, 다른 하나는 Cityscapes instance-level semantic labeling task다. CVPPP는 식물 잎을 개별적으로 분할하는 문제이고, Cityscapes는 도시 장면에서 차량, 보행자 등 객체 instance를 분리하는 문제다. 두 데이터셋 모두 이미지당 instance 수의 중앙값이 15개 이상이라, variable-instance setting에서 제안 손실의 장점을 보기 좋다.

### 4.1 데이터셋과 설정

CVPPP A1 subset은 128장의 학습 이미지와 33장의 테스트 이미지를 사용한다. 평가지표는 instance segmentation 정확도를 보는 Symmetric Best Dice (SBD)와, 잎 개수 예측 오차를 보는 $|DiC|$다. Leaf segmentation 실험에서는 데이터가 작기 때문에 좌우 반전, 회전, scale deformation을 포함한 online augmentation을 사용한다. 입력 이미지는 $512 \times 512$로 리사이즈하고, 추가로 x, y coordinate map을 채널로 붙인다. margin은 $\delta_v = 0.5$, $\delta_d = 1.5$, embedding 차원은 16으로 설정했다. 이 벤치마크는 foreground mask가 테스트셋에 제공되므로 instance separation 자체에 더 집중할 수 있다.

Cityscapes는 fine annotation 기준으로 2975개 train, 500개 validation, 1525개 test 이미지로 구성된다. 저자들은 validation으로 하이퍼파라미터를 조정하고 train set으로 최종 모델을 학습한다. 입력 해상도는 $768 \times 384$이며, 데이터 양과 다양성이 충분해 별도의 augmentation은 쓰지 않았다고 설명한다. margin은 동일하게 $\delta_v = 0.5$, $\delta_d = 1.5$이고 embedding 차원은 8이다. semantic class 구분을 위해 pretrained ResNet-38 semantic segmentation network를 별도로 사용하고, instance embedding loss는 각 semantic class에 대해 독립적으로 적용한다. 즉, pedestrian과 car는 같은 feature space 위치를 차지해도 상관없고, 같은 클래스 내부 instance만 분리하면 된다.

모든 실험은 off-the-shelf semantic segmentation backbone으로 ResNet-38을 사용하고, Cityscapes semantic segmentation으로 사전학습된 모델에서 fine-tuning한다. 최적화는 Adam, learning rate는 $10^{-4}$이며, NVidia Titan X GPU에서 학습했다고 명시한다. 여기서도 논문의 메시지는 일관된다. “새 구조가 아니라, 기존 구조에 새 loss를 꽂는다”는 점이다.

### 4.2 CVPPP 결과

CVPPP 테스트셋에서 제안 방법은 SBD 84.2, $|DiC|$ 1.0을 기록했다. 표에 따르면 End-to-end recurrent attention 기반 방법이 SBD 84.9, $|DiC|$ 0.8로 약간 높지만, 제안 방법도 사실상 동급 성능이다. 저자들은 자신들의 방법이 recurrent method보다 개념적으로 더 단순하면서도 비슷한 성능을 낸다고 해석한다. 특히 non-deep learning 방법들과 recurrent instance segmentation인 RIS+CRF보다 우수하거나 경쟁력 있는 결과를 보였다고 강조한다.

정성적 결과에서는 작은 잎이나 잎자루 부근에서만 소폭 오류가 나타나고, 대부분의 잎을 안정적으로 분리했다고 서술한다. 즉, 동일한 형태가 반복적으로 나타나는 장면에서는 embedding clustering 방식이 상당히 잘 맞는다는 점을 보여준다.

### 4.3 Cityscapes 결과

Cityscapes 테스트셋에서 제안 방법은 AP 17.5, AP0.5 35.9, AP100m 27.8, AP50m 31.0을 기록했다. 표 기준으로 Mask R-CNN 26.2 AP보다는 낮지만, Boundary-aware 17.4, DWT 19.4, Pixelwise DIN 20.0과 비슷한 경쟁권에 있으며, 일부 기존 방법보다 높다. 저자들은 “공개된 방법들 중 거의 최상위권”이며, 특히 multi-task network cascades 계열 SAIS와 사실상 비슷한 수준이라고 해석한다. 다만 표 숫자 자체만 보면 당시 최고 성능은 아니며, proposal-free하고 구조가 단순하다는 점을 감안한 경쟁력으로 이해하는 것이 정확하다.

정성적으로는 차량과 보행자가 많은 복잡한 거리 장면에서도 개별 instance를 자주 올바르게 구분한다. 그러나 대표적인 실패 사례는 두 가지다. 하나는 실제로 서로 다른 객체를 하나로 합쳐 버리는 incorrect merging이고, 다른 하나는 semantic segmentation 오류가 downstream instance segmentation까지 전파되는 경우다. 예를 들어 비어 있는 자전거 보관대를 bicycle로 잘못 semantic segmentation하면, instance embedding은 존재하지 않는 자전거들을 억지로 쪼개려 한다. 이 분석은 논문이 단순히 instance embedding 자체만이 아니라 semantic front-end 품질에도 강하게 의존한다는 사실을 잘 보여준다.

### 4.4 구성 요소 분석

논문에서 가장 유익한 실험 중 하나는 semantic segmentation과 clustering의 영향을 분리해서 본 ablation이다. Cityscapes validation에서 ResNet-38 semantic mask와 mean-shift clustering을 사용한 기본 설정은 AP 21.4, AP0.5 40.2다. 같은 semantic mask를 유지하고 clustering만 ground-truth center threshold로 바꾸면 AP가 22.9, AP0.5가 44.1로 올라간다. 즉, clustering 개선만으로도 일정한 이득이 있다.

더 큰 차이는 semantic segmentation 품질에서 나온다. semantic mask를 ground truth로 바꾸고 mean-shift clustering을 쓰면 AP 37.5, AP0.5 58.5가 되고, semantic mask도 ground truth, clustering도 ground-truth center threshold를 쓰면 AP 47.8, AP0.5 77.8까지 올라간다. 이 결과는 두 가지를 말해 준다. 첫째, 실제 병목은 semantic segmentation 품질이다. 둘째, embedding 자체도 완벽하지 않아 clustering 단계에도 여전히 개선 여지가 있다. 저자들도 특히 작은 instance에서 mean-shift clustering의 오류가 더 두드러진다고 해석한다. 즉, 이 논문의 core idea는 타당하지만, 최종 성능은 front-end semantic mask와 back-end clustering 모두의 영향을 크게 받는다.

### 4.5 Speed-accuracy trade-off

논문은 car class만 대상으로 ENet, SegNet, Dilation, ResNet-38 등 여러 semantic segmentation backbone을 서로 다른 해상도에서 비교한다. 핵심 메시지는 명확하다. ResNet-38이 정확도는 가장 좋지만 메모리 사용량이 크고 속도가 느리다. ENet은 정확도는 약간 낮지만 훨씬 빠르다. 또한 해상도를 $768 \times 384$ 이상으로 올려도 큰 정확도 향상은 없다고 보고한다. 후처리 오버헤드는 무시할 만하다고 적고 있어, 실제 비용 대부분은 backbone forward pass에서 발생한다고 볼 수 있다. 이 실험은 제안 손실이 특정 backbone에 묶이지 않고 다양한 segmentation network 위에 얹힐 수 있음을 보여주는 근거이기도 하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정식화의 우아함이다. instance segmentation을 “픽셀별 label classification”이 아니라 “픽셀 embedding clustering”으로 바꿈으로써, instance 수 가변성과 permutation invariance를 자연스럽게 처리한다. proposal, bounding box, recurrent decoding, Hungarian matching 같은 별도 장치 없이도 작동한다는 점은 개념적 단순성과 구현 용이성 측면에서 매우 큰 장점이다. 또, bounding box 기반 방식이 취약한 복잡한 occlusion 상황을 더 잘 다룰 수 있다는 점을 논리적으로 설득력 있게 제시한다. 산업 영상, 세포·염색체 분할, overlapping object sorting 같은 분야에 특히 잘 맞을 가능성이 크다.

또 다른 강점은 loss와 inference가 정합적이라는 점이다. 많은 방법들이 학습 목표와 추론 절차 사이에 간극이 있는데, 이 논문은 학습 시 margin 구조로 만들어진 embedding geometry를 추론 시 thresholding/mean-shift clustering이 직접 이용한다. 즉, 학습과 추론의 설계 철학이 맞물려 있다. 이런 일관성은 실제로 논문이 다양한 backbone에 쉽게 이식될 수 있었던 이유이기도 하다.

하지만 한계도 분명하다. 첫째, Cityscapes 결과와 ablation이 보여주듯 semantic segmentation quality에 크게 의존한다. multi-class setting에서는 semantic mask가 잘못되면 instance 분리는 그 위에서 아무리 잘해도 한계가 있다. 둘째, clustering이 완벽하지 않다. validation loss가 0이 아니므로 이론적 margin 조건이 완전히 충족되지 않고, 특히 작은 객체에서 splitting/merging 오류가 발생한다. 셋째, 데이터 분포가 일정하고 장면 구성이 비교적 규칙적인 경우에는 잘 맞지만, VOC나 MS COCO처럼 객체 배치가 매우 다양하고 등장 조합이 예측 불가능한 데이터에서는 detection 기반 접근보다 불리할 수 있다고 저자 스스로 인정한다. 즉, 이 방법은 모든 instance segmentation 문제의 보편 해법이라기보다, 특정 구조적 조건에서 특히 매력적인 proposal-free 해법이다.

비판적으로 보면, 논문은 “간단한 후처리”를 강조하지만 사실 실제 성능은 semantic segmentation과 clustering의 정교함에 적지 않게 좌우된다. 따라서 논문의 실질적 기여는 end-to-end complete system이라기보다, instance separation을 위한 매우 강력한 embedding loss design에 있다고 보는 편이 정확하다. 또한 Cityscapes에서는 semantic segmentation을 별도 pretrained network에 의존하므로, instance와 semantic segmentation의 완전한 통합 학습은 아직 미완 상태다. 논문 마지막의 future work가 바로 이 지점을 향하고 있다는 점도 주목할 만하다.

## 6. 결론

이 논문은 instance segmentation을 위해 discriminative embedding loss를 제안하고, 이를 통해 복잡한 proposal system 없이도 픽셀 임베딩을 cluster하여 instance를 복원할 수 있음을 보였다. 핵심 기여는 같은 instance는 모으고 다른 instance는 떼어놓는 간단한 margin-based objective를 픽셀 수준에 적용해, semantic segmentation backbone을 거의 그대로 활용하면서 instance-aware representation을 학습하게 만든 데 있다. 실험적으로는 CVPPP leaf segmentation에서 state-of-the-art에 근접한 성능을, Cityscapes에서는 구조 단순성 대비 경쟁력 있는 성능을 달성했다.

장기적으로 이 연구는 proposal-free instance segmentation 계열의 중요한 출발점 중 하나로 볼 수 있다. 특히 embedding-based grouping이라는 관점은 이후 panoptic/instance segmentation, medical image instance separation, metric-learning 기반 dense prediction 연구에 매우 큰 영향을 줄 수 있는 아이디어다. 논문 자체도 semantic segmentation과 instance segmentation의 joint training을 미래 과제로 제시하는데, 실제로 이후 연구 흐름을 생각해 보면 이 방향은 매우 타당하다. 정리하면, 이 논문은 “복잡한 구조 대신 잘 설계된 loss로 instance structure를 학습한다”는 강한 메시지를 남긴, 개념적으로도 실용적으로도 의미 있는 작업이다.
