# Bridging Category-level and Instance-level Semantic Image Segmentation

- **저자**: Zifeng Wu, Chunhua Shen, Anton van den Hengel
- **발표연도**: 2016
- **arXiv**: https://arxiv.org/abs/1605.06885

## 1. 논문 개요

이 논문은 **semantic category-level segmentation**과 **instance-level segmentation** 사이의 간극을 메우는 방법을 제안한다. 기존의 강력한 semantic segmentation 모델을 먼저 사용하고, 그 결과 위에서 각 픽셀이 어떤 객체 인스턴스에 속하는지를 복원하는 방식이다. 즉, 보통의 **detect-then-segment** 파이프라인처럼 먼저 bounding box를 검출한 뒤 그 안에서 mask를 자르는 것이 아니라, 먼저 픽셀 단위 category segmentation을 수행한 다음, 각 foreground pixel이 속한 인스턴스의 bounding box를 예측해 인스턴스를 찾아낸다.

연구 문제는 분명하다. semantic segmentation은 픽셀마다 category를 잘 맞히지만, 같은 category 안에 여러 개의 객체가 있을 때 그것들을 서로 다른 instance로 구분하지 못한다. 반대로 당시의 state-of-the-art instance segmentation 방법들은 대부분 object detector에 의존하고 있었고, 이는 별도의 bounding-box proposal 또는 detector 품질에 성능이 크게 좌우되는 구조였다. 저자들은 “instance segmentation은 semantic segmentation에서 한 단계만 더 나가면 되는 문제인데, 왜 기존의 강력한 semantic segmentation 결과를 더 직접적으로 활용하지 않는가?”라는 관점에서 출발한다.

이 문제는 중요하다. 실제 장면 이해에서는 단순히 “여기에 sheep category가 있다”는 것만으로는 충분하지 않고, “양이 몇 마리인지, 각각 어디에 있는지, 각 픽셀이 어떤 양에 속하는지”가 필요하다. 자율주행, 로보틱스, 이미지 편집, 장면 구조 이해 같은 응용에서는 category-level 이해와 instance-level 이해가 모두 요구되기 때문이다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 단순하다. **semantic score map을 Hough-like map으로 바꾸면 instance를 local maxima 탐색 문제로 다룰 수 있다**는 것이다. 구체적으로 각 foreground pixel에 대해, 그 픽셀이 속한 객체 인스턴스의 bounding box 중심 위치와 크기(height, width)를 회귀한다. 그러면 같은 인스턴스에 속한 픽셀들은 비슷한 bounding box를 가리키게 되고, 이 정보들을 semantic score와 결합하면 특정 인스턴스 중심에 점수가 모이는 transformed map이 만들어진다. 이후 이 map에서 local maxima를 찾으면 객체 인스턴스 후보를 얻을 수 있다.

기존 detect-then-segment 방식과의 차별점은 파이프라인의 방향 자체가 다르다는 점이다. 기존 방식은 먼저 object detection이 주가 되고 segmentation은 그 뒤를 따른다. 반면 이 논문은 **semantic segmentation이 주축**이고, instance 분리는 그 위에서 수행되는 후속 단계다. 저자들은 이 접근이 단순하면서도 generic하다고 주장한다. PFN(Proposal-Free Network)처럼 instance 개수까지 예측한 뒤 clustering하는 복잡한 구조도 아니고, 특정 장면에 강하게 의존하는 template matching이나 depth cue도 필요하지 않다.

또 하나의 핵심 기여는 **online bootstrapping of hard pixels**이다. 픽셀 단위 예측 문제에서는 쉬운 픽셀이 너무 많아 학습이 비효율적이 되기 쉽다. 특히 semantic segmentation에서는 큰 영역 중앙의 픽셀들이 매우 쉽고, 클래스 불균형도 심하다. 저자들은 학습 중 현재 모델이 이미 잘 맞히는 쉬운 픽셀은 버리고, 어려운 픽셀만 선택적으로 사용해 학습하는 전략이 semantic segmentation과 instance localization 모두에 결정적으로 중요하다고 보였다.

## 3. 상세 방법 설명

전체 시스템은 두 네트워크로 구성된다. 하나는 **semantic segmentation network**, 다른 하나는 **localization network**이다. 두 네트워크는 별도로 학습된다.

테스트 시 파이프라인은 다음 흐름으로 진행된다.

먼저 semantic segmentation network가 category-wise score map을 생성한다. 예를 들어 sheep category에 대해서는 각 픽셀이 sheep일 확률 점수를 갖는 맵이 생긴다. 다음으로 localization network가 각 foreground pixel에 대해 해당 픽셀이 속한 인스턴스의 bounding box를 예측한다. 예측 대상은 총 4개이며, 현재 픽셀에서 인스턴스 bounding box 중심까지의 horizontal offset과 vertical offset, 그리고 그 인스턴스의 height와 width이다.

그 다음 이 bounding-box regression 결과를 semantic score map에 적용해 **transform map**을 만든다. 직관적으로 말하면, 원래 이미지 공간에서 떨어져 있던 동일 인스턴스의 픽셀들이 예측된 bounding box 중심 쪽으로 “투표”하도록 바꾸는 과정이다. 이렇게 되면 같은 인스턴스에 속한 픽셀들의 신호가 특정 위치에 모이게 된다. 논문은 이를 generalized Hough transform과 연결지어 해석한다.

이후 transformed map 위에서 **non-maximum suppression (NMS)**를 수행해 local maxima를 찾고, 이를 인스턴스 hypothesis로 삼는다. 각 maximum에 대해 그 주변에서 suppression된 픽셀들을 다시 추적(trace back)하면 해당 인스턴스의 mask를 복원할 수 있다. 마지막으로 여러 instance hypothesis 사이의 중복을 제거하기 위해 region-based NMS를 적용해 최종 instance segmentation 결과를 만든다.

성능 향상을 위해 저자들은 각 category에 대해 top-$n$ semantic mask를 사용한다. 즉, 각 픽셀에 대해 해당 category가 상위 $n$개 예측 안에 들어가면 그 픽셀을 그 category의 후보로 포함한다. 이는 semantic segmentation의 실수 때문에 instance recall이 낮아지는 것을 완화하기 위한 장치로 이해할 수 있다.

학습 목표는 두 네트워크에서 다르다. semantic segmentation network는 고전적인 pixel-wise logistic regression loss를 사용한다. localization network는 Fast R-CNN에서 쓰이던 **smoothed $L_1$ loss**를 사용해 bounding box regression을 학습한다. 또한 큰 객체와 작은 객체의 기여가 불균형해지는 것을 막기 위해, 인스턴스 크기(height, width)에 따라 픽셀 loss를 re-weight한다. 다만 하나의 인스턴스 내부에서 중심 픽셀과 주변 픽셀의 중요도는 동일하다고 보고, 같은 인스턴스에 속한 픽셀에는 같은 weight를 준다.

semantic segmentation용 online bootstrapping은 식 (1)로 정의된다. 논문의 표기대로, $N$개의 픽셀과 $K$개의 category가 있을 때 픽셀 $a_i$의 정답 라벨을 $y_i$, category $c_j$에 대한 예측 확률을 $p_{ij}$라 하면 loss는 다음과 같다.

$$
\ell = - \frac{1}{\sum_i^N \sum_j^K \mathbf{1}\{y_i=j \text{ and } p_{ij}<t\}}
\left(
\sum_i^N \sum_j^K \mathbf{1}\{y_i=j \text{ and } p_{ij}<t\}\log p_{ij}
\right)
$$

여기서 핵심은 $\mathbf{1}\{y_i=j \text{ and } p_{ij}<t\}$ 조건이다. 즉, 정답 클래스에 대한 예측 확률이 threshold $t$보다 낮은, 다시 말해 현재 모델이 아직 잘 못 맞히는 픽셀만 loss에 포함한다. 이미 쉽게 맞히는 픽셀은 버린다. 다만 mini-batch마다 남는 픽셀이 너무 적어 gradient가 불안정해지는 것은 피해야 하므로, 논문은 threshold $t$를 동적으로 조절한다. 현재 배치가 너무 쉽게 풀리면 $t$를 높여 더 많은 hard pixel을 포함시키고, 반대로 모델이 아직 잘 못하면 $t$를 낮춘다.

localization network에서의 bootstrapping은 약간 다르다. 여기서는 각 픽셀의 regression loss 자체를 기준으로 easy/hard를 나누지 않고, **예측 box와 ground-truth box 사이의 IoU**를 기준으로 hard example을 정한다. 저자들의 논리는 분명하다. 실제로 중요한 것은 offset이나 width/height를 수치적으로 얼마나 정확히 회귀했는가가 아니라, 이후 NMS 단계에서 올바른 인스턴스를 분리할 만큼 box가 적절한가이다. NMS가 IoU에 기반하므로, regression target 각각의 오차보다 IoU를 직접 기준으로 삼는 것이 더 자연스럽다는 주장이다.

네트워크 구조 측면에서 보면, 저자들은 ResNet을 fully convolutional하게 변형한 **Fully Convolutional Residual Network (FCRN)**를 사용한다. 분류용 ResNet의 마지막 linear classifier를 convolution layer로 바꾸고, spatial prediction을 위해 7x7 pooling layer를 제거한다. 이 pooling은 receptive field를 늘려 문맥 정보를 제공하지만, 픽셀 간 특징 차이를 희석시켜 경계 예측에 해가 될 수 있기 때문이다.

또한 feature map resolution을 높이기 위해 down-sampling 일부를 제거하고, 그에 맞춰 이후 convolution에 dilation을 적용한다. 이는 DeepLab의 hole algorithm, 즉 atrous convolution과 같은 생각이다. 저자들은 feature map의 해상도와 classifier의 field-of-view(FoV)를 함께 조절하며 최적 구성을 탐색했다. 중요한 점은 문맥 정보를 충분히 보기 위해 FoV는 커야 하지만, feature map 해상도도 높아야 픽셀 단위 구분이 가능하다는 점이다. 따라서 저자들은 작은 kernel에 큰 dilation을 부여해 큰 FoV를 구현했다.

추가로, 논문은 본문 초반 기여 요약에서 residual block 일부에 dropout regularization을 넣고, top classifier를 multi-layer non-linear classifier로 바꾸며, multi-view testing도 사용했다고 언급한다. 그러나 제공된 발췌문에서는 이 변경들의 구체적인 구조나 수식, 삽입 위치, 세부 설정은 자세히 설명되지 않는다. 따라서 이 부분의 상세 구현을 논문 본문에서 지금 주어진 텍스트만으로 완전히 재구성할 수는 없다.

## 4. 실험 및 결과

논문은 semantic segmentation과 instance segmentation을 모두 평가한다. semantic segmentation은 PASCAL VOC 2012, Cityscapes, PASCAL-Context에서 평가했고, instance segmentation은 PASCAL VOC 2012에서 평가했다.

PASCAL VOC 2012 semantic segmentation에서는 augmented train set 10,582장을 사용했다. 평가지표는 pixel accuracy, mean pixel accuracy, mean IoU이다. 실험 결과, ResNet depth를 50에서 101로 늘리면 성능이 크게 향상되지만 152로 늘렸을 때는 추가 이득이 없었다고 보고한다. 저자들은 이를 overfitting 가능성으로 해석한다. 또 feature map resolution을 $1/16$에서 $1/8$로 높이고, classifier FoV를 224보다 크게 잡는 것이 유리했다. PASCAL VOC에서 가장 좋은 vanilla FCRN 설정은 **101-layer, resolution $1/8$, kernel 5, dilation 12, FoV 392**였고, validation mean IoU는 **73.41%**였다.

여기에 bootstrapping을 적용하면 PASCAL VOC validation mean IoU가 **74.80%**까지 오른다. test set에서는 category-wise 비교 결과, 제안 방법이 기존 강한 방법들인 FCN-8s, DeepLab, CRFasRNN, DPN, UoAContext를 넘어 **mean IoU 79.1%**를 달성했다. 이는 당시 기록이라고 주장한다. category별로는 20개 중 18개 category에서 최고 성능을 기록했다고 한다. 다만 bicycle, chair처럼 다양성과 가림이 심하고 annotation도 까다로운 클래스는 여전히 어렵다고 분석한다.

Cityscapes에서도 비슷한 경향이 나타난다. 더 깊은 네트워크, 더 높은 resolution, 더 큰 FoV가 대체로 유리했다. 가장 좋은 vanilla FCRN은 mean IoU **71.51%**, bootstrapping 적용 후에는 **74.6%**를 달성했다. 논문은 이전 최고 성능 68.6%보다 상당히 높다고 주장한다. 클래스별로는 traffic light, train 등 드문 클래스에서 bootstrapping의 효과가 컸다.

PASCAL-Context에서는 배경 포함 60개 class에 대해 평가했고, 제안 방법은 **pixel accuracy 72.9%, mean accuracy 54.8%, mean IoU 44.5%**를 기록했다. 이는 FCN-8s, BoxSup, UoA-Context보다 높은 결과다.

instance segmentation 평가는 PASCAL VOC 2012 val에서 수행했다. SBD annotation을 사용했고, 평가지표는 $mAP_r^{0.5}$, $mAP_r^{0.7}$, 그리고 $mAP_r^{vol}$이다. 비교 대상은 SDS, Hypercolumn, MNC 등이다. 결과를 보면, 단순 50-layer 모델은 $mAP_r^{0.5}=57.2\%$, $mAP_r^{0.7}=40.5\%$였다. 여기에 bootstrapping과 weighted loss를 추가하고 101-layer로 확장하면 **$mAP_r^{0.5}=60.9\%$, $mAP_r^{0.7}=44.6\%$, $mAP_r^{vol}=55.5\%$**가 된다. COCO pretraining까지 더하면 **$mAP_r^{0.5}=61.5\%$, $mAP_r^{0.7}=46.6\%$, $mAP_r^{vol}=56.4\%$**를 달성했다.

특히 $mAP_r^{0.7}$에서 이전 최고인 MNC의 **41.5%**를 **46.6%**로 끌어올린 점을 중요한 개선으로 제시한다. overlap threshold가 높은 $0.7$에서 성능이 크게 오른 것은 mask와 localization 품질이 더 정교해졌다는 의미로 볼 수 있다. 저자들은 추가 pilot experiment도 수행했는데, semantic score map을 ground-truth semantic mask로 대체하고 localization network는 그대로 사용했더니 성능이 **$mAP_r^{0.5}=73.0\%$, $mAP_r^{0.7}=60.6\%$**까지 상승했다. 이는 제안된 instance pipeline의 상한이 semantic segmentation 품질에 크게 좌우되며, semantic segmentation이 더 좋아지면 instance segmentation도 크게 개선될 수 있음을 보여주는 결과다.

bootstrapping의 중요성도 정량적으로 검증했다. PASCAL VOC에서는 hard pixel을 512개 유지하는 설정이 가장 좋았고, Cityscapes에서도 512개가 최적이었다. 예를 들어 Cityscapes에서 152-layer FCRN은 bootstrapping 없이 mean IoU **71.51%**였지만, 512 hard pixels 설정에서는 **74.64%**가 되었다. 이는 무려 3.13%p 향상으로, 단순한 trick 이상으로 핵심 성분임을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **instance segmentation을 semantic segmentation 위에서 직접 구성한 단순하고 명확한 설계**에 있다. 당시 주류였던 detect-then-segment와 다른 경로를 제시하면서도, 결과는 동등하거나 더 좋았다. 특히 semantic score를 Hough-like transform map으로 바꾸고 local maxima를 instance로 해석하는 관점은 직관적이며, 복잡한 proposal machinery 없이도 충분히 경쟁력 있다는 점을 보였다.

두 번째 강점은 **semantic segmentation 성능 자체를 매우 강하게 끌어올렸다는 점**이다. 이 논문의 instance segmentation 성능은 좋은 semantic segmentation에 기반한다. 따라서 단순히 후처리 파이프라인만 제시한 것이 아니라, 그 기반이 되는 FCRN 구조, resolution/FoV 설계, online bootstrapping까지 함께 최적화했다는 점이 설득력을 높인다.

세 번째 강점은 **hard pixel bootstrapping의 효과를 명확히 보였다**는 점이다. 픽셀 단위 예측에서 데이터 불균형과 쉬운 예제 과다 문제는 매우 일반적이다. 이 논문은 이를 직접 겨냥했고, 드문 클래스에서 특히 향상이 크다는 점까지 표를 통해 보여준다. semantic segmentation과 localization 모두에서 동일한 철학을 적용했다는 것도 장점이다.

반면 한계도 분명하다. 먼저, 이 방식은 **semantic segmentation 품질에 크게 의존**한다. 저자들 스스로 ground-truth semantic mask를 사용한 실험에서 성능이 크게 뛴다고 보여주었는데, 이는 동시에 semantic segmentation이 틀리면 downstream instance segmentation도 구조적으로 제한된다는 뜻이다. 즉, category segmentation 오류가 발생하면 이후 단계에서 회복하기 어렵다.

또한 이 방법은 결국 각 픽셀이 bounding box를 회귀해 같은 인스턴스로 모이게 만드는 구조이므로, **겹침이 심하거나 인접한 동일 클래스 인스턴스가 매우 복잡하게 얽힌 장면**에서는 box 기반 표현만으로 충분한 분리가 어려울 수 있다. 논문은 더 정교한 clustering이 성능을 더 올릴 수 있다고 언급하지만, 실제로는 단순 off-the-shelf NMS를 사용했다. 따라서 local maxima 탐색과 cluster 복원 단계가 병목일 가능성이 있다.

추가로, 논문 초반의 기여 목록에서는 dropout, non-linear classifier, multi-view testing 등의 개선을 언급하지만, 제공된 본문 발췌만으로는 이 요소들의 세부 구현과 각 기여도가 충분히 설명되지 않는다. 따라서 이 부분은 논문 전체를 보지 않고는 정확한 재현성이 다소 떨어진다. 마찬가지로 instance segmentation에서 사용한 top-$n$ 설정값이나 일부 세부 하이퍼파라미터도 발췌문만으로는 명확하지 않다.

비판적으로 보면, 제안 방식은 “proposal-free”라는 점에서 깔끔하지만, 실제로는 bounding box regression과 NMS를 여전히 사용한다. 즉, detector pipeline을 직접 쓰지 않을 뿐, object localization의 box 표현 자체에서 완전히 자유로운 것은 아니다. 그럼에도 불구하고 이 논문의 핵심 가치는 detect-first 사고방식이 유일한 길이 아님을 실험적으로 보여준 데 있다.

## 6. 결론

이 논문은 semantic segmentation과 instance segmentation을 하나의 연속된 문제로 보고, **semantic score map 위에서 instance를 복원하는 새로운 파이프라인**을 제안했다. 핵심 기여는 세 가지로 정리할 수 있다. 첫째, semantic segmentation 기반의 간결한 instance segmentation 방법을 제시했다. 둘째, hard pixel 중심의 online bootstrapping을 도입해 semantic segmentation과 localization 모두를 개선했다. 셋째, FCRN 구조를 체계적으로 최적화해 당시 기준 매우 강한 semantic segmentation 성능을 달성했다.

실제 적용 측면에서 이 연구는 매우 중요하다. 이 논문은 이후 더 발전한 semantic segmentation backbone이 등장할수록 instance segmentation도 함께 좋아질 수 있다는 가능성을 명확히 보여준다. 즉, instance segmentation을 반드시 detector 중심으로 풀 필요는 없으며, 강한 dense prediction 모델 위에서 instance를 복원하는 방향도 유효하다는 메시지를 남겼다. 향후 연구 관점에서는 더 정교한 grouping 방식, 더 나은 mask 복원 방식, 그리고 semantic segmentation 오류에 덜 민감한 통합 학습 구조로 발전할 여지가 크다.
