# Learning to Refine Object Segments

- **저자**: Pedro O. Pinheiro, Tsung-Yi Lin, Ronan Collobert, Piotr Dollár
- **발표연도**: 2016
- **arXiv**: https://arxiv.org/abs/1603.08695

## 1. 논문 개요

이 논문은 object instance segmentation, 그중에서도 특히 **object proposal generation을 위한 고품질 segmentation mask 생성** 문제를 다룬다. 저자들은 기존의 feedforward CNN이 물체의 존재와 범주를 이해하는 데는 강하지만, pooling을 거치면서 공간 해상도가 낮아져 **물체 경계를 정밀하게 맞추는 능력은 부족하다**는 점을 핵심 문제로 제시한다.

이 문제는 object segmentation의 본질과 직접 연결된다. 물체를 정확히 분할하려면 한편으로는 “이 영역이 하나의 물체인가”를 판단할 수 있는 object-level semantic information이 필요하고, 다른 한편으로는 경계, 질감, 위치 같은 low-level spatial detail도 필요하다. 그런데 일반적인 CNN은 아래층에서 spatial detail을 잘 보존하지만 의미 정보가 약하고, 위층에서는 의미 정보가 강해지는 대신 해상도가 떨어진다. 따라서 상위 feature만으로 mask를 예측하면 대략적인 모양은 맞출 수 있어도 경계가 거칠어지는 문제가 생긴다.

저자들은 이 문제를 해결하기 위해 기존 DeepMask를 출발점으로 삼아, coarse한 mask representation을 먼저 만들고 이후 이를 점점 더 세밀하게 복원하는 **top-down refinement architecture**를 제안한다. 이 방법은 최종적으로 SharpMask라는 이름으로 제시되며, COCO에서 DeepMask 대비 segmentation proposal average recall을 약 10~20% 개선하고, 동시에 네트워크 구조 최적화를 통해 속도도 크게 줄였다고 보고한다. 논문의 중요성은 단순히 proposal quality를 높였다는 데만 있지 않고, **object-level reasoning과 pixel-level refinement를 어떻게 효율적으로 결합할 것인가**라는 구조적 문제에 대한 하나의 설계 원리를 제시했다는 데 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 명확하다. **좋은 object mask는 처음부터 고해상도로 직접 예측하는 것이 아니라, 먼저 상위 계층에서 coarse하지만 의미적으로 올바른 표현을 만든 뒤, 이를 하위 계층 feature를 이용해 점진적으로 refine해야 한다**는 것이다.

저자들은 기존 skip connection 기반 접근과 자신들의 방법을 분명히 구분한다. skip architecture는 여러 층에서 독립적으로 prediction을 만들고 이를 upsample 및 average하는 방식으로 이해할 수 있는데, semantic segmentation에서는 이런 방식이 잘 작동할 수 있다. 그러나 object instance segmentation에서는 단순히 각 픽셀을 지역적으로 판단하는 것만으로는 부족하다. 예를 들어 양 털 같은 local pattern은 “sheep class”임을 말해줄 수는 있어도, 그것이 같은 개체에 속하는지 다른 개체에 속하는지는 local receptive field만으로 결정하기 어렵다. 즉, object instance segmentation에서는 **전역적 물체 수준의 정보가 먼저 필요하고, 세밀한 경계 복원은 그 다음**이어야 한다는 것이 저자들의 주장이다.

이를 위해 논문은 skip connection처럼 각 층에서 독립된 출력을 만드는 대신, 상위 층에서 만든 **mask encoding**을 하향식으로 전달하면서 각 단계마다 대응되는 저수준 feature와 결합하는 refinement module을 도입한다. 이 모듈은 pooling으로 줄어든 해상도를 되돌리는 역할을 하며, 한 단계마다 spatial resolution을 2배로 늘린다. 따라서 전체 구조는 bottom-up path에서 semantic representation을 만들고, top-down path에서 그것을 spatially sharpen하는 형태가 된다.

기존 접근과의 차별점은 세 가지로 요약할 수 있다. 첫째, 각 층의 독립 예측을 평균하는 대신 하나의 coarse mask representation을 중심으로 refinement를 수행한다. 둘째, deconvolution network처럼 단순히 pooling switch를 전달하는 것이 아니라 실제 feature value를 전달하고 변환한다. 셋째, proposal generation처럼 이미지당 수백 개의 후보를 처리해야 하는 상황을 고려해 **정확도뿐 아니라 계산 효율성까지 함께 설계**했다는 점이 중요하다.

## 3. 상세 방법 설명

### 전체 파이프라인

SharpMask는 크게 두 부분으로 구성된다. 첫 번째는 기존 CNN과 유사한 **bottom-up feedforward pathway**이고, 두 번째는 새롭게 제안된 **top-down refinement pathway**이다.

bottom-up 경로는 입력 patch로부터 낮은 해상도의 의미적 표현을 만든다. DeepMask에서는 이 경로가 최종적으로 coarse mask를 직접 출력했지만, SharpMask에서는 refinement를 위해 먼저 **mask encoding** $M_1$을 만든다. 이 $M_1$은 단순한 1채널 binary mask가 아니라 여러 채널을 가진 feature map이며, 저자들은 이런 다채널 표현이 좋은 성능에 중요하다고 말한다.

그 다음 top-down 경로에서는 refinement module $R_i$를 여러 개 쌓아, $M_1$을 점차 더 높은 해상도의 표현으로 바꾼다. 각 module은 현재 단계의 coarse mask encoding $M_i$와 bottom-up 경로의 대응되는 feature $F_i$를 입력으로 받아 더 세밀한 표현 $M_{i+1}$을 만든다. 논문에서 이 관계는 다음처럼 정리된다.

$$
M_{i+1} = R_i(M_i, F_i)
$$

여기서 각 $R_i$는 하나의 pooling stage를 “되돌리는” 역할을 하며, 결과적으로 refinement를 여러 번 반복하면 최종 출력 해상도가 입력 patch와 같아진다.

### refinement module의 구조와 역할

각 refinement module의 핵심 목적은 상위 단계의 object-level mask representation과 하위 단계의 spatially rich feature를 적절히 결합하는 것이다. 다만 $F_i$를 그대로 $M_i$와 concatenate하면 두 가지 문제가 있다. 첫째, $F_i$의 channel 수 $k_i^f$가 매우 클 수 있어 계산량이 커진다. 둘째, 보통 $k_i^f \gg k_i^m$이므로, 단순 연결 시 refinement의 중심이 되어야 할 $M_i$의 정보가 묻힐 수 있다.

그래서 저자들은 먼저 $F_i$를 작은 channel 수로 줄여 **skip feature** $S_i$를 만든다. 이 과정은 3×3 convolution과 ReLU로 이루어진다. 즉, refinement는 다음과 같은 단계로 이해할 수 있다.

1. bottom-up feature $F_i$를 3×3 convolution으로 변환해 compact한 skip feature $S_i$를 만든다.
2. 현재 mask encoding $M_i$와 $S_i$를 concatenate한다.
3. 다시 3×3 convolution과 ReLU를 적용해 둘을 통합한 새로운 representation을 만든다.
4. bilinear upsampling으로 해상도를 2배 늘려 $M_{i+1}$을 얻는다.

논문은 이 구조가 convolution, ReLU, concatenation, bilinear upsampling만 사용하므로 fully backpropable이고 효율적이라고 강조한다. 또한 하위 단계로 갈수록 spatial resolution은 커지므로 channel 수는 오히려 줄이는 방향이 적절하다고 설명한다. 이것은 일반적인 CNN의 설계, 즉 깊어질수록 해상도는 줄고 channel 수는 늘어나는 패턴과 반대이다.

### mask encoding의 의미

이 논문에서 중요한 개념은 $M_i$가 단순한 binary mask가 아니라는 점이다. 저자들은 $k_1^m > 1$인 multi-channel mask encoding이 성능에 핵심적이라고 말한다. 이는 coarse stage에서 이미 단순한 foreground/background 예측을 넘는 richer representation이 형성되어야, 이후 refinement 단계가 이를 기반으로 경계를 세밀하게 복원할 수 있음을 뜻한다.

즉, top-down path는 “이미 거의 완성된 mask를 업샘플링하는 과정”이라기보다, **의미적 object representation을 점차 고해상도 공간으로 풀어내는 과정**에 가깝다.

### 학습 절차

SharpMask는 DeepMask와 **동일한 data definition과 loss function**을 사용한다고 명시한다. 각 학습 샘플은 입력 patch, 해당 patch가 적절한 크기의 centered object를 포함하는지에 대한 label, 그리고 positive sample인 경우의 binary object mask로 구성된다. trunk는 ImageNet pretraining으로 초기화하고, 나머지 층은 uniform random initialization을 사용한다.

학습은 두 단계로 이루어진다.

첫째 단계에서는 원래 DeepMask처럼 coarse한 pixel-wise mask와 objectness score를 jointly 학습한다. 이 부분은 기존 DeepMask 학습과 동일하다.

둘째 단계에서는 feedforward path를 고정한 뒤, 최종 mask prediction layer를 제거하고 대신 mask encoding $M_1$을 생성하는 linear layer로 바꾼다. 이후 refinement module들을 붙이고 학습한다. 이때 오차는 refinement module 내부의 horizontal 및 vertical convolution에만 backpropagation된다.

저자들이 이런 2-stage training을 택한 이유는 세 가지다. 첫째, 더 빨리 수렴했다. 둘째, 같은 네트워크에서 forward branch만 사용하면 coarse mask를, refinement까지 사용하면 sharp mask를 생성할 수 있다. 셋째, forward branch가 충분히 수렴한 뒤 전체 네트워크를 end-to-end로 미세조정해도 추가 이득이 거의 없었다.

다만 논문은 정확한 loss 식을 수식으로 다시 적지는 않는다. 따라서 SharpMask의 refinement stage가 어떤 세부 목적함수 형태를 가지는지는, 제공된 본문 기준으로는 **DeepMask와 동일하다고만 명시되어 있고 별도의 새로운 loss 방정식은 제시되지 않았다**고 보는 것이 정확하다.

### 추론 절차

full-image inference에서는 DeepMask와 마찬가지로 convolution을 통해 인접 window 간 계산을 공유한다. 특히 $S_i$도 공유 가능한 feature이므로 효율적이다. 하지만 refinement module은 위치별로 서로 다른 $M_1$을 입력받기 때문에 이 단계는 완전히 공유할 수 없다.

이를 해결하기 위해 저자들은 모든 위치를 refinement하지 않고, 먼저 objectness score가 높은 상위 $N$개 proposal window만 선택한 뒤 batch 형태로 refinement를 수행한다. 즉, coarse stage는 dense하게 돌리고, expensive한 refinement stage는 promising한 후보에만 적용한다. 이 설계가 SharpMask가 proposal generation setting에서 실용적인 속도를 유지할 수 있는 핵심 이유다.

### feedforward architecture 최적화

논문의 초점은 refinement module이지만, 저자들은 feedforward backbone도 함께 최적화한다. trunk 측면에서는 입력 크기 $W$, pooling layer 수 $P$, stride density $S$, depth $D$, 최종 feature dimension $F$를 분석한다.

핵심 관찰은 다음과 같다. 작은 입력 크기 $W$는 속도를 크게 높이지만 작은 물체 처리에 불리하고, pooling이 많을수록 계산은 빨라지지만 feature resolution이 줄어든다. stride density는 mask prediction에 중요하며, stride가 커지면 위치 정밀도가 떨어진다. 또한 저자들은 VGG-A 대신 ResNet-50 기반 trunk를 사용해 더 깊으면서도 효율적인 feature extraction을 얻는다. 마지막으로 1×1 convolution으로 feature dimension을 줄여 속도를 개선한다.

head 구조도 단순화한다. 원래 DeepMask는 mask branch와 score branch가 비교적 무겁게 분리되어 있었는데, 논문은 A, B, C 세 가지 head variant를 제안해 점점 더 많은 계산을 공유하게 만든다. 최종적으로 head C는 **하나의 compact한 512-dimensional vector에서 mask와 score를 함께 예측**하도록 구성되며, 저자들은 단순성과 속도 면에서 이를 채택한다.

이렇게 trunk와 head를 최적화한 feedforward baseline을 저자들은 **DeepMask-ours**라고 부르고, SharpMask는 이를 기반으로 refinement를 추가한 모델이다.

## 4. 실험 및 결과

### 실험 설정

실험은 COCO dataset의 train set 약 80k 이미지와 500k instance annotation으로 학습하고, 대부분의 평가는 validation set 앞 5k 이미지에서 수행한다. mask accuracy는 IoU로 측정하며, proposal 평가에는 주로 COCO에서 널리 쓰이는 **Average Recall (AR)**을 사용한다. IoU 0.5부터 0.95까지의 recall을 proposal 개수 기준으로 요약하며, AR@10, AR@100, AR@1000과 proposal 수 전반에 대한 평균인 AUC를 보고한다. 또한 object 크기에 따라 small, medium, large로 나누어 결과를 제시한다.

refinement stage의 학습은 learning rate $10^{-3}$을 사용하며, Nvidia Tesla K40m에서 약 2일이 걸린다고 한다. 또한 patch-based training과 convolutional inference 사이의 차이를 줄이기 위해, deeper model에서는 patch 주변에 32 pixel context를 추가하고 zero-padding 대신 reflective padding을 사용했다고 밝힌다. 최종 continuous mask는 threshold 0.2로 binarize한다.

### feedforward architecture 최적화 결과

Table 1에서 저자들은 다양한 trunk 설정의 “upper bound AR”을 비교한다. 원래 DeepMask는 AR 36.6, small 18.2, medium 48.7, large 50.6이며 mask inference 시간이 1.32초였다. 입력 크기를 224에서 160으로 줄인 VGG 기반 W160-P4-D8-VGG는 속도가 0.58초로 크게 빨라졌지만 AR은 35.5로 약간만 감소했다. 이후 ResNet trunk를 사용한 W160-P4-D39는 AR 37.0으로 성능을 회복했고, feature dimension을 128로 줄인 W160-P4-D39-F128도 AR 36.9로 거의 손실 없이 0.45초까지 빨라졌다.

반면 입력 크기를 112로 더 줄인 경우에는 일부 설정에서 정확도가 크게 떨어졌다. 저자들은 최종적으로 **W160-P4-D39-F128**이 속도와 정확도의 균형이 가장 좋다고 결론내린다.

head 비교인 Table 2에서는 원래 DeepMask total time 1.59초 대비, head A는 0.51초, head B는 0.50초, head C는 0.46초로 크게 빨라진다. 성능은 세 head가 비슷하며, AR@100 기준 head C가 25.8로 가장 좋거나 동등 수준이다. 따라서 저자들은 head C를 채택한다.

이 최적화를 종합한 **DeepMask-ours**는 원래 DeepMask보다 3배 이상 빠른 0.46초/image를 달성하며, parameter 수도 약 75M에서 17M으로 줄었다.

### SharpMask refinement 분석

Section 5.2에서는 refinement module의 channel capacity를 조절하는 두 가지 schedule을 비교한다. 하나는 모든 단계에서 $k_i^m = k_i^s = k$로 두는 방식이고, 다른 하나는 $k_i^m = k_i^s = \frac{k}{2^{i-1}}$ 형태로 단계가 내려갈수록 channel 수를 줄이는 방식이다. 실험 결과, capacity를 늘릴수록 성능은 꾸준히 좋아졌고 overfitting 징후는 없었다고 한다. 최종적으로 두 번째 schedule에서 $k=32$인 설정이 속도와 성능의 균형이 가장 좋아 최종 모델로 선택되었다.

또한 $F_i$에서 $S_i$를 만들 때 한 번의 3×3 convolution만 사용하는 경우, 특히 $k_i^s \ll k_i^f$일 때 학습이 어려웠다고 보고한다. 그래서 실제 실험에서는 두 번의 3×3 convolution을 사용해 먼저 64채널로 줄인 후 다시 $k_i^s$로 줄였다.

중요한 ablation도 수행했다. 첫째, downward convolution을 제거하고 $k_i^m = k_i^s = 1$로 두어 각 층 출력을 평균하는 구조를 만들었는데, 이는 전형적인 skip architecture와 유사하다. 둘째, horizontal convolution을 제거한 구조를 시험했는데, 이는 deconv network와 유사하다. 두 경우 모두 baseline feedforward network보다 의미 있는 향상이 없었다. 즉, 저자들은 **horizontal connection과 vertical connection이 모두 필요하다**고 결론낸다. 이는 SharpMask가 단순한 skip/deconv 변형이 아니라는 점을 실험적으로 뒷받침한다.

### 기존 방법과의 비교

Table 3은 COCO validation set에서 box proposals와 segmentation proposals를 비교한 핵심 결과다. segmentation proposal 기준으로 보면:

- DeepMask는 AR@10 12.6, AR@100 24.5, AR@1000 33.1, 전체 AUC 18.3
- DeepMask-ours는 AR@10 14.4, AR@100 25.8, AR@1000 33.1, AUC 19.4
- SharpMask는 AR@10 15.6, AR@100 27.6, AR@1000 35.5, AUC 20.9
- SharpMaskZoom은 AR@100 30.3, AR@1000 39.2, AUC 22.4
- SharpMaskZoom2는 AR@100 30.7, AR@1000 40.8, AUC 22.5

즉, 단순히 feedforward architecture를 최적화한 DeepMask-ours만으로도 DeepMask보다 향상이 있고, 여기에 top-down refinement를 추가한 SharpMask는 다시 한 번 분명한 성능 향상을 보인다. 특히 segmentation proposals에서 improvement가 더 크다는 점이 중요하다. 저자들도 sharpening된 mask가 box를 둘러싼 tight bounding box는 크게 바꾸지 않을 수 있으므로, segmentation에서 improvement가 더 크게 나타나는 것은 자연스럽다고 해석한다.

속도 측면에서도 SharpMask는 image당 0.76초로, 원래 DeepMask의 1.59초보다 2배 이상 빠르다. refinement module이 추가 파라미터 3M 미만만 요구한다는 점도 효율성을 보여준다.

### object detection에 미치는 영향

저자들은 SharpMask proposal을 Fast R-CNN pipeline에 연결해 object detection 성능도 평가한다. Figure 5(c)에서는 MPN classifier를 붙였을 때 SharpMask proposal이 Selective Search보다 약 5 AP 높은 약 28 AP를 달성하며, proposal 수는 약 500개 정도면 성능이 포화된다고 보고한다.

Table 4 상단에서는 VGG classifier를 사용한 단순 baseline 비교에서:

- Selective Search + VGG: AP 19.3
- RPN + VGG: AP 21.9
- SharpMask + VGG: AP 25.2

로 나타난다. 즉, proposal 품질 향상이 detection 성능으로도 연결된다는 것을 보여준다.

또한 2015 COCO challenge 관련 결과도 제시되는데, SharpMask+MPN ensemble은 segmentation AP 25.1, box AP 33.5를 달성해 당시 challenge에서 2위를 기록했다고 한다. 다만 이 결과는 ensembling 및 challenge setting이 포함되어 있고, ResNet 기반 최신 설정으로는 다시 돌리지 않았다고 명시한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 object instance segmentation의 구조적 요구를 명확히 짚고, 그에 맞는 네트워크 설계를 제안했다는 점이다. 저자들은 왜 단순 skip connection이 object segmentation에 충분하지 않은지 직관적으로 설명하고, 그 대안으로 coarse-to-fine top-down refinement를 제안했다. 그리고 이 아이디어가 단지 개념적 제안에 머물지 않고, 실제로 COCO proposal benchmark에서 일관된 성능 향상으로 이어진다는 점을 실험으로 입증했다.

또 다른 강점은 **효율성을 설계의 중심에 두었다는 점**이다. proposal generation은 이미지마다 많은 window를 다뤄야 하므로, semantic segmentation처럼 전체 이미지를 한 번 처리하는 구조를 그대로 쓰기 어렵다. 저자들은 refinement stage를 top-N proposal에만 적용하고, shared feature와 patch-dependent computation을 구분하는 방식으로 이 문제를 풀었다. refinement module의 계산 구성도 비교적 단순해 실제 적용 가능성이 높다.

논문은 architecture analysis도 비교적 충실하다. trunk, head, refinement capacity를 따로 분석하며, 단순히 최종 최고 성능만 제시하는 것이 아니라 어떤 요소가 속도와 정확도에 영향을 주는지 체계적으로 보여준다. 특히 skip-like 구조와 deconv-like 구조가 왜 충분하지 않은지를 ablation으로 확인한 부분은 설득력이 있다.

한편 한계도 분명하다. 첫째, 이 논문은 refinement가 다른 pixel-labeling task에도 적용 가능할 수 있다고 말하지만, 실제 실험은 거의 전부 object proposal generation에 집중되어 있다. 따라서 제안 방식의 일반성이 어느 정도인지, semantic segmentation이나 다른 dense prediction 문제에서 동일하게 유효한지는 이 논문만으로는 판단할 수 없다.

둘째, 학습 목표 함수는 DeepMask와 동일하다고만 설명되며, refinement module이 coarse representation을 어떤 방식으로 가장 잘 활용하는지에 대한 더 세밀한 이론적 분석은 제공되지 않는다. 다시 말해, 왜 multi-channel mask encoding이 효과적인지, refinement의 각 단계가 어떤 정보를 복원하는지에 대한 표현학적 해석은 제한적이다.

셋째, refinement는 top-N proposal에 대해서만 수행되므로, 전체 pipeline은 여전히 objectness ranking 품질에 의존한다. 상위 proposal selection이 부정확하면 refinement 자체가 좋은 후보에 적용되지 않을 수 있다. 논문은 이 문제를 직접적으로 다루지는 않는다.

넷째, small object에 대해서는 추가 image scale을 사용한 SharpMaskZoom이 상당한 개선을 보이는데, 이는 기본 SharpMask만으로는 작은 물체에 대한 한계가 남아 있음을 시사한다. 즉, refinement 자체만으로 scale 문제를 완전히 해결하지는 못한다.

비판적으로 보면, 이 논문의 핵심 기여는 “정확한 경계 복원”이라기보다 **proposal generation setting에서 쓸 수 있을 만큼 효율적인 boundary refinement 설계**에 있다. 따라서 pixel-perfect segmentation 자체를 최종 목표로 하는 방법과 비교하면 목적이 약간 다르다. 그럼에도 당시 맥락에서는 detection pipeline과 연결 가능한 practical segmentation proposal network를 제시했다는 점에서 충분히 중요한 기여로 볼 수 있다.

## 6. 결론

이 논문은 object instance segmentation을 위해 feedforward CNN에 **top-down refinement module**을 추가하는 SharpMask를 제안했다. 핵심은 상위 층에서 coarse한 object-aware mask encoding을 만든 뒤, 하위 층의 spatial feature를 단계적으로 결합해 해상도를 복원하고 mask 경계를 날카롭게 만드는 것이다. 이 설계는 단순 skip connection이나 deconvolution보다 object segmentation 문제에 더 적합하다고 주장되며, COCO 실험에서 DeepMask 대비 segmentation proposal 성능을 뚜렷하게 개선했다.

또한 저자들은 refinement 아이디어뿐 아니라 trunk와 head 구조까지 함께 최적화해 속도와 파라미터 효율도 개선했다. 결과적으로 SharpMask는 당시 object proposal generation에서 정확도와 속도 모두에서 강력한 결과를 보였고, downstream object detection 성능도 향상시켰다.

향후 관점에서 보면 이 연구는 **coarse semantic reasoning과 fine spatial refinement의 분리 및 재결합**이라는 설계 원리를 분명히 보여준다. 이 아이디어는 이후의 instance segmentation, encoder-decoder, feature pyramid, coarse-to-fine refinement 계열 연구들과도 자연스럽게 연결된다. 따라서 이 논문은 단순히 DeepMask의 성능 개선판이라기보다, dense prediction에서 top-down refinement가 왜 중요한지를 잘 보여준 연구로 평가할 수 있다.
