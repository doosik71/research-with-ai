# Context Encoding for Semantic Segmentation

- **저자**: Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, Amit Agrawal
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1803.08904

## 1. 논문 개요

이 논문은 semantic segmentation에서 **장면 전체의 문맥(global contextual information)** 을 명시적으로 활용하면 픽셀별 분류를 더 정확하게 만들 수 있다는 문제의식에서 출발한다. 기존의 FCN 계열 방법들은 dilated/atrous convolution, multi-scale feature, boundary refinement 등을 통해 해상도와 receptive field를 개선해 왔지만, 저자들은 이것만으로는 “장면이 어떤 상황인지”라는 전역적 의미를 충분히 반영하지 못한다고 본다. 그 결과, 서로 시각적으로 비슷하지만 문맥상 잘 등장하지 않는 클래스들 사이에서 오분류가 발생한다. 예를 들어 실내 장면에서 차량 관련 feature를 덜 강조하고, 침실 장면에서 bed, chair 같은 범주의 가능성을 더 높게 보는 식의 전역 문맥 활용이 필요하다는 것이다.

논문의 목표는 이런 전역 문맥을 CNN 내부에서 가볍고 명시적으로 모델링하는 **Context Encoding Module**을 제안하고, 이를 이용한 **Context Encoding Network (EncNet)** 를 설계하는 것이다. 저자들은 이 모듈이 기존 FCN 파이프라인에 거의 추가 비용 없이 결합될 수 있으며, PASCAL-Context, PASCAL VOC 2012, ADE20K에서 강한 성능 향상을 보인다고 보고한다. 추가로 이 모듈이 segmentation뿐 아니라 CIFAR-10 분류에서도 얕은 네트워크의 표현력을 강화할 수 있음을 실험한다.

이 문제가 중요한 이유는 semantic segmentation이 단순히 지역적인 텍스처나 경계만 보는 문제가 아니라, **장면 수준의 의미 구조를 이해하는 문제**이기 때문이다. 픽셀 단위 예측은 매우 세밀하지만, 실제로는 “이 장면이 무엇인지”가 어떤 클래스가 등장할 수 있는지를 강하게 제한한다. 따라서 전역 문맥을 잘 쓰는 모델은 특히 소형 객체나 혼동되기 쉬운 클래스에서 더 안정적인 예측을 할 가능성이 높다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 전역 문맥을 단순히 receptive field를 넓히는 방식으로 우회적으로 얻는 것이 아니라, **feature statistics를 직접 인코딩해 semantic context를 추출하고**, 그 결과를 다시 feature map channel의 중요도를 조절하는 데 사용하자는 것이다. 저자들은 이 과정을 classic computer vision의 encoding 계열 표현, 예를 들어 Bag-of-Words, VLAD, Fisher Vector처럼 전체 feature 분포를 요약하는 관점과 연결한다.

이를 위해 논문은 기존의 **Encoding Layer**를 확장해 semantic segmentation에 맞게 사용한다. 이 레이어는 feature map 전체를 하나의 집합으로 보고, 학습 가능한 codebook과 residual encoding을 통해 장면의 전역적 의미를 요약한다. 저자들은 이 출력을 **encoded semantics**라고 부른다. 그리고 이 encoded semantics를 기반으로 각 channel별 scaling factor를 예측해, 현재 장면 문맥에 맞는 feature map은 강조하고 맞지 않는 feature map은 약화한다.

기존 접근과의 차별점은 두 가지다. 첫째, pyramid pooling이나 atrous spatial pyramid처럼 receptive field를 늘리는 것과 달리, 이 논문은 **전역 문맥을 명시적으로 encoding**한다. 둘째, segmentation 학습에서 흔히 쓰는 per-pixel loss만으로는 장면 전체의 의미를 배우기 어렵다고 보고, 이미지 안에 어떤 클래스가 존재하는지를 맞히는 **Semantic Encoding Loss (SE-loss)** 를 추가해 전역 문맥 학습을 직접적으로 유도한다. 저자들에 따르면 이 손실은 큰 객체와 작은 객체를 동일한 비중으로 다루기 때문에, 특히 작은 객체 성능 개선에 도움이 된다.

## 3. 상세 방법 설명

전체 구조는 pre-trained ResNet 기반의 dilated FCN 위에 Context Encoding Module을 올린 형태이다. 논문에서는 stage 3와 stage 4에 dilation strategy를 적용해 출력 stride를 줄이고, 마지막 prediction 직전에 Context Encoding Module을 배치한다. 또한 주 segmentation branch 외에 별도의 branch를 두어 SE-loss를 학습한다. 더 나아가 stage 3 위에도 추가 Context Encoding Module을 얹어 SE-loss를 거는 보조 정규화도 사용한다. 이는 PSPNet의 auxiliary loss와 비슷한 역할을 하지만 계산량이 훨씬 작다고 설명한다.

Encoding Layer의 입력은 크기 $C \times H \times W$의 feature map이며, 이를 $N=H \times W$개의 $C$차원 feature 집합 $X=\{x_1,\dots,x_N\}$로 본다. 이 레이어는 $K$개의 codeword로 이루어진 codebook $D=\{d_1,\dots,d_K\}$와 smoothing factor 집합 $S=\{s_1,\dots,s_K\}$를 학습한다. 각 입력 feature $x_i$와 codeword $d_k$ 사이의 residual은 다음과 같다.

$$
r_{ik} = x_i - d_k
$$

그리고 soft-assignment를 사용해 residual encoder를 계산한다.

$$
e_k = \sum_{i=1}^{N} e_{ik}
$$

$$
e_{ik} =
\frac{\exp(-s_k \|r_{ik}\|^2)}
{\sum_{j=1}^{K}\exp(-s_j \|r_{ij}\|^2)} r_{ik}
$$

즉, 각 위치의 feature는 여러 codeword에 부드럽게 할당되고, 그 residual들이 모여 장면의 전역 feature statistics를 표현한다. 논문은 codeword별 encoder를 단순 concatenation하지 않고, Batch Normalization과 ReLU를 거친 뒤 합산해 하나의 표현으로 집계한다. 이를 통해 encoder들 사이에 순서를 강제하지 않고 차원도 줄인다.

이렇게 얻은 encoded semantics $e$는 두 방향으로 사용된다. 첫 번째는 **featuremap attention**이다. fully connected layer와 sigmoid를 통해 channel scaling factor $\gamma$를 계산한다.

$$
\gamma = \delta(W e)
$$

여기서 $W$는 가중치, $\delta$는 sigmoid이다. 그런 다음 입력 feature map $X$에 channel-wise multiplication을 적용해 출력 $Y$를 만든다.

$$
Y = X \otimes \gamma
$$

이 연산의 의미는 단순하다. 현재 장면 문맥과 잘 맞는 channel은 크게 유지하고, 덜 관련된 channel은 줄이는 것이다. 예를 들어 하늘 장면에서는 airplane과 관련된 표현을 더 강조하고, vehicle과 관련된 표현은 약화할 수 있다.

두 번째는 **Semantic Encoding Loss**이다. 논문은 standard segmentation training의 per-pixel cross-entropy만으로는 네트워크가 전역 문맥을 강하게 학습하지 못한다고 본다. 그래서 Encoding Layer 위에 또 다른 fully connected layer와 sigmoid를 두고, 이미지 안에 각 클래스가 존재하는지 여부를 multi-label prediction으로 맞히게 한다. 이때 손실은 binary cross entropy이다. SE-loss의 정답은 별도 annotation이 필요한 것이 아니라, ground-truth segmentation mask에서 등장한 클래스 집합을 추출해 자동으로 만든다. 즉, 픽셀 위치를 맞히는 것이 아니라 “이 이미지에 클래스가 있느냐”를 맞히게 함으로써 장면 전체 의미를 학습하게 한다.

최종적으로 EncNet은 기존 dilated FCN의 dense prediction 능력을 유지하면서, 위와 같은 context encoding과 channel reweighting을 통해 전역 문맥을 반영한다. 저자들은 이 모듈이 **미분 가능하고**, 추가 supervision 없이 기존 FCN 파이프라인에 바로 삽입 가능하며, 계산량도 매우 작다고 주장한다.

부록에서는 synchronized cross-GPU Batch Normalization 구현도 설명한다. 여러 GPU에 걸친 평균과 분산을 계산할 때 두 번 동기화하는 대신, 분산을 다음 식으로 계산해 한 번의 all-reduce로 처리한다.

$$
\sigma^2
=
\frac{\sum_{i=1}^{N}(x_i-\mu)^2}{N}
=
\frac{\sum_{i=1}^{N} x_i^2}{N}
-
\frac{\left(\sum_{i=1}^{N} x_i\right)^2}{N^2}
$$

여기서 평균은

$$
\mu = \frac{\sum_{i=1}^{N} x_i}{N}
$$

이다. 이후 각 샘플에 대해 BatchNorm 출력은

$$
y_i = \gamma \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta
$$

로 계산된다. 이는 큰 crop size와 작은 per-GPU batch size 문제를 완화하기 위한 구현적 기여다.

## 4. 실험 및 결과

실험은 semantic segmentation에 대해 PASCAL-Context, PASCAL VOC 2012, ADE20K를 사용했고, 추가로 image classification에 대해 CIFAR-10도 평가했다. 기본 backbone은 주로 ResNet-50과 ResNet-101이며, stage 3와 4에 dilation을 적용했다. 출력은 bilinear interpolation으로 8배 upsampling하여 loss를 계산했다. 학습 스케줄은 polynomial decay 형태의 learning rate를 사용했고, 데이터 증강은 random flip, scale 0.5에서 2, rotation -10도에서 10도, fixed-size crop을 포함한다. 평가에서는 multi-scale inference도 사용했다. 학습 batch size는 16이며, Encoding Layer의 codeword 수는 기본적으로 $K=32$를 사용했다.

PASCAL-Context에서는 59개 객체 클래스와 background를 포함한 60개 클래스를 다룬다. ablation 결과가 이 논문의 핵심을 잘 보여준다. baseline FCN은 73.4% pixAcc와 41.0% mIoU를 기록했는데, Context Encoding Module만 추가해도 78.1% / 47.6%로 크게 상승했다. 여기에 SE-loss를 더하면 79.4% / 49.2%가 되었고, ResNet-101 backbone을 쓰면 80.4% / 51.7%가 된다. multi-size evaluation까지 적용한 최종 모델은 81.2% pixAcc, 52.6% mIoU를 기록했다. 비교 표에서는 background를 포함한 mIoU 기준으로 EncNet이 51.7%를 기록해, DeepLab-v2의 45.7%, RefineNet의 47.3%를 넘어섰다.

ablation에서 SE-loss의 가중치 $\alpha$는 $\{0.0, 0.1, 0.2, 0.4, 0.8\}$ 를 비교했고, 저자들은 $\alpha = 0.2$가 가장 좋은 결과를 보였다고 보고한다. 또한 codeword 수 $K$를 바꾸어 본 결과, $K=32$ 근처에서 성능 향상이 포화되는 경향이 있었다. 논문은 $K=0$을 global average pooling에 해당하는 경우로 두고 비교했는데, 단순 평균 풀링보다 Encoding Layer가 더 효과적임을 시사한다.

PASCAL VOC 2012에서는 augmented annotation set을 사용했고, COCO pre-training 유무를 나누어 비교했다. COCO pre-training 없이 EncNet은 **82.9% mIoU**를 달성해 기존 방법들보다 높은 성능을 보였다. COCO pre-training을 적용하면 **85.9% mIoU**로 더 향상되며, 표 기준으로 PSPNet의 85.4%, DeepLabv3의 85.7%를 소폭 넘는다. 클래스별 결과에서도 bird, boat, bottle, bus, cat, dog, sheep 등 여러 범주에서 강한 성능을 보인다. 저자들은 EncNet이 PSPNet이나 DeepLabv3보다 계산 복잡도가 더 낮다고 언급하지만, 본문에 구체적 FLOPs 비교 수치는 제시되지 않았다.

ADE20K validation set에서는 EncNet-50이 79.73% pixAcc와 41.11% mIoU, EncNet-101이 81.69% pixAcc와 44.65% mIoU를 기록했다. 이는 baseline FCN Res50의 34.38% mIoU를 크게 넘는 결과다. 또한 EncNet-101은 더 깊은 backbone인 PSPNet-269의 44.94% mIoU와 비슷한 수준에 도달했다. test set에서는 EncNet-101 single model이 **0.5567**의 최종 점수를 기록해, 논문이 인용한 COCO-Place Challenge 2017 우승 팀 0.5547과 PSPNet-269 single model 0.5538을 모두 넘어섰다.

CIFAR-10 실험은 segmentation이 아니라 얕은 네트워크 분류 성능 개선을 보기 위한 보조 실험이다. 14-layer ResNet baseline은 4.93% error, SE-ResNet baseline은 4.65% error를 기록했다. 여기에 Context Encoding Module을 넣은 EncNet 16k64d는 3.96%, 더 큰 EncNet 32k128d는 3.45% error를 달성했다. 저자들은 이 결과를 통해 Context Encoding Module이 네트워크 초반 단계의 feature representation 자체를 강화할 수 있다고 해석한다. 다만 이 분류 실험은 semantic segmentation의 주장을 직접 검증하는 본 실험이라기보다, 제안 모듈의 일반성을 보여 주는 보조 증거에 가깝다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 전역 문맥을 단순히 “넓은 receptive field”로 간접적으로 처리하지 않고, **명시적인 context encoding과 channel reweighting으로 구조화했다는 점**이다. 문제 정의와 방법 설계가 잘 연결되어 있고, classic encoding 아이디어를 deep network 안으로 가져와 semantic segmentation에 맞게 재구성했다는 점도 설득력이 있다. 또한 제안 모듈이 기존 FCN 계열 네트워크에 가볍게 붙을 수 있고, ablation에서 baseline 대비 뚜렷한 성능 향상을 보였다는 점은 실용적으로 의미가 있다. 특히 PASCAL-Context에서 모듈 추가만으로 mIoU가 크게 오르는 결과는, 전역 문맥이 실제로 segmentation 성능에 기여함을 뒷받침한다.

SE-loss 역시 단순하지만 좋은 설계다. 픽셀별 손실만으로는 이미지 수준의 의미를 충분히 강제하기 어렵다는 문제를 multi-label 존재 예측으로 보완한다는 발상은 자연스럽다. 더구나 추가 annotation이 필요 없고 ground-truth mask에서 바로 생성 가능하다는 점에서 구현 부담이 작다. 저자들이 언급하듯, 큰 객체와 작은 객체를 동등하게 다루는 특성은 소형 객체의 존재 신호를 보존하는 데 도움을 줄 수 있다.

한편 한계도 분명하다. 첫째, 논문은 Context Encoding Module이 “marginal extra computation”만 든다고 반복해서 말하지만, 본문에 정교한 연산량, latency, memory 비교표는 없다. 따라서 실제 배포 환경에서 PSPNet, DeepLab 계열과 비교해 얼마나 효율적인지는 이 논문 텍스트만으로는 정량적으로 판단하기 어렵다. 둘째, 성능 향상이 왜 발생하는지에 대한 해석은 주로 정성적 사례와 ablation에 의존하며, channel scaling이 실제로 어떤 semantic concept를 얼마나 안정적으로 반영하는지에 대한 깊은 분석은 부족하다. 셋째, ADE20K test set 점수 비교는 강력하지만, challenge setting과 evaluation protocol 세부 조건이 본문에 자세히 서술되지는 않아 완전한 apples-to-apples 비교인지까지는 이 텍스트만으로 단정하기 어렵다.

또 하나의 비판적 해석은, 이 방법이 전역 문맥을 잘 쓰는 것은 맞지만, 동시에 **orderless feature statistics**를 이용하기 때문에 공간적 관계를 직접 모델링하는 방식은 아니라는 점이다. 즉, “무엇이 있는가”는 잘 요약하지만 “어디에 어떻게 배치되어 있는가”를 직접 다루는 메커니즘은 아니다. semantic segmentation에서는 두 정보가 모두 중요하므로, 이 모듈은 spatial reasoning을 대체하기보다는 보완하는 역할로 보는 것이 적절하다. 실제로 논문도 이를 기존 FCN 위에 붙이는 보완 모듈로 제시한다.

마지막으로 CIFAR-10 결과는 인상적이지만, 실험 설정에 dropout/shakeout 유사 regularization을 scaling factor $s_k$에 적용하는 등 추가적 학습 기법이 포함되어 있다. 따라서 분류 성능 향상이 오직 context encoding 자체 때문인지, 혹은 학습 레시피의 영향도 큰지에 대해서는 더 엄밀한 분리가 필요하다. 논문은 이 부분을 충분히 세분화해 분석하지는 않는다.

## 6. 결론

이 논문은 semantic segmentation에서 전역 문맥을 명시적으로 다루기 위해 **Context Encoding Module**과 **SE-loss**를 제안하고, 이를 기반으로 한 **EncNet**을 설계했다. 핵심 기여는 Encoding Layer를 이용해 장면의 전역 semantic context를 요약하고, 그 정보를 channel-wise scaling으로 다시 feature map에 반영함으로써 분할 문제를 더 쉽게 만드는 데 있다. 또한 이미지 수준 클래스 존재 예측을 통한 SE-loss로 전역 문맥 학습을 정규화했다.

실험적으로 EncNet은 PASCAL-Context, PASCAL VOC 2012, ADE20K에서 강한 성능을 보였고, 특히 적은 추가 비용으로 baseline FCN을 크게 개선했다. 논문 텍스트 기준으로 볼 때, 이 연구는 semantic segmentation에서 “전역 문맥을 어떻게 효율적으로 주입할 것인가”라는 질문에 간결하고 실용적인 답을 제시한 작업이다. 이후 연구나 실제 시스템에서도, backbone 전체를 바꾸지 않고 context-aware representation을 추가하는 방식의 설계에 중요한 영향을 줄 수 있는 아이디어로 읽힌다.
