# Reversible Recursive Instance-level Object Segmentation

* **저자**: Xiaodan Liang, Yunchao Wei, Xiaohui Shen, Zequn Jie, Jiashi Feng, Liang Lin, Shuicheng Yan
* **발표연도**: 2015
* **arXiv**: [https://arxiv.org/abs/1511.04517](https://arxiv.org/abs/1511.04517)

## 1. 논문 개요

이 논문은 **instance-level object segmentation** 문제를 다룬다. 이 문제는 단순히 물체가 어디 있는지를 bounding box로 찾는 object detection이나, 각 픽셀의 semantic category만 구분하는 semantic segmentation보다 더 어렵다. 같은 클래스에 속한 여러 개체를 각각 분리해서, 각 개체마다 정확한 픽셀 단위 마스크를 예측해야 하기 때문이다. 예를 들어 이미지 안에 사람이 여러 명 있을 때, “person”이라는 클래스만 맞추는 것으로는 충분하지 않고, 각 사람을 서로 다른 인스턴스로 분리해 각자의 마스크를 예측해야 한다.

논문이 제기하는 핵심 문제는 두 가지다. 첫째, proposal-based instance segmentation의 성능이 **초기 object proposal의 품질에 크게 의존**한다는 점이다. 기존 방법들은 대체로 proposal을 먼저 만들고, 그 proposal 위에서 분류와 segmentation을 한 번 수행하는 구조였기 때문에, proposal이 조금만 어긋나도 최종 segmentation 품질이 크게 떨어졌다. 둘째, 하나의 proposal 안에 여러 객체가 겹쳐 들어 있는 경우, 특히 같은 클래스의 객체가 중첩된 경우에는 어떤 객체가 “주된(dominant)” 인스턴스인지 구분하기 어렵다는 문제가 있다.

이 논문은 이 문제를 해결하기 위해 **R2-IOS (Reversible Recursive Instance-level Object Segmentation)** 라는 프레임워크를 제안한다. 핵심은 instance segmentation과 proposal refinement를 분리된 후처리 단계로 보지 않고, **서로 도움을 주는 두 개의 sub-network를 recursive하게 반복 갱신**하도록 설계한 점이다. segmentation 결과는 proposal refinement를 더 정확하게 만들고, refinement된 proposal은 다시 segmentation을 더 잘하게 만든다. 이 상호작용을 여러 번 반복하여 점진적으로 성능을 개선한다.

이 연구의 중요성은 당시 proposal-based detection/segmentation 계열 방법이 주류이던 시점에서, **bounding box refinement와 instance mask prediction을 통합적이고 반복적인 구조로 결합했다는 점**에 있다. 또한 모든 proposal에 동일한 refinement iteration 수를 쓰지 않고, proposal마다 가장 적절한 iteration 수를 선택하는 **reversible gate** 개념을 도입한 점도 중요한 기여다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 크게 세 가지로 요약된다.

첫 번째는 **recursive joint refinement**이다. 논문은 instance-level segmentation과 bounding box refinement가 서로 보완적이라고 본다. segmentation은 물체의 경계와 픽셀 수준 구조를 제공하므로 bounding box localization을 더 잘하게 도와줄 수 있다. 반대로 refinement된 bounding box는 proposal 내부에 더 적절한 물체 영역을 담게 되므로 segmentation mask 예측이 더 쉬워진다. 따라서 한 번만 예측하고 끝내는 대신, 이 둘을 반복적으로 업데이트하는 것이 더 합리적이라는 직관을 취한다.

두 번째는 **reversible refinement**이다. 모든 proposal이 동일한 수의 refinement를 필요로 하지는 않는다. 어떤 proposal은 초기 위치가 이미 좋기 때문에 한두 번 refinement만으로 충분하고, 어떤 proposal은 주변 객체와 많이 겹쳐 더 많은 refinement가 필요할 수 있다. 논문은 이를 반영하기 위해 각 iteration마다 category confidence를 보고, 가장 높은 confidence를 얻은 iteration을 해당 proposal의 최적 refinement 단계로 선택한다. 즉, “고정된 iteration 수” 대신 **proposal별 adaptive stopping**을 수행한다.

세 번째는 **instance-aware denoising autoencoder**이다. 하나의 proposal 안에 여러 개의 객체가 들어 있을 수 있는데, 이때 단순한 convolutional prediction은 모든 비슷한 객체를 foreground로 표시하는 경향이 있다. 논문은 local convolutional feature만으로는 dominant instance를 분리하기 어렵다고 보고, proposal 전체를 압축한 hidden representation을 통해 global context를 활용하는 denoising autoencoder를 도입한다. 이 모듈은 proposal 전체를 보고 어떤 객체가 중심 객체인지 판단하여, 다른 겹친 객체의 noisy mask를 줄이고 dominant object mask를 복원한다.

기존 접근법과의 차별점은 분명하다. 당시 대표적인 방법들은 proposal 생성 후 단발성 예측을 수행하거나, refinement가 있더라도 segmentation 정보를 refinement에 직접적으로 반영하지 못했다. 반면 이 논문은 **segmentation-aware features**, **recursive training/testing**, **proposal별 adaptive iteration**, **dominant instance 분리를 위한 global autoencoder**를 하나의 통합 프레임워크로 제시한다.

## 3. 상세 방법 설명

### 전체 구조

R2-IOS는 크게 두 개의 sub-network로 이루어진다.

하나는 **instance-level segmentation sub-network**이고, 다른 하나는 **reversible proposal refinement sub-network**이다. 두 네트워크는 같은 이미지 feature를 바탕으로 각 proposal을 처리하지만, 각자의 출력이 서로에게 입력으로 다시 반영된다.

입력은 전체 이미지와 초기 object proposals이다. 논문에서는 selective search를 사용해 이미지당 약 2,000개의 proposal을 생성한다. 이미지 전체는 VGG-16 backbone을 통과해 convolutional feature maps를 만든다. 이후 각 proposal에 대해 ROI pooling을 수행하여 proposal별 고정 크기 feature를 얻고, 이를 각 sub-network에서 사용한다.

처리 흐름은 다음과 같다.

먼저 segmentation sub-network가 proposal 내부의 dominant object에 대한 foreground mask를 예측한다. 동시에 proposal refinement sub-network는 각 proposal의 class confidence와 bounding box regression offset을 예측한다. refinement된 bounding box는 다음 iteration에서 다시 입력 proposal이 되며, segmentation 결과도 refinement network의 feature에 반영된다. 이렇게 여러 iteration을 반복한 뒤, proposal마다 가장 적절한 iteration을 reversible gate가 선택한다.

### 3.1 Instance-level Segmentation Sub-network

이 sub-network는 VGG-16을 기반으로 한다. 다만 local detail 보존을 위해 원래 VGG-16의 마지막 두 max pooling layer를 제거한다. 또한 classification용 fully connected layer 대신 fully convolutional layer를 사용하여 spatial feature map을 유지한다. 각 proposal에 대해서는 ROI pooling을 통해 $40 \times 40$ 크기의 feature map을 추출한다.

그 위에 $1 \times 1$ convolution을 적용하여 foreground/background confidence map $\mathbf{C}$를 만든다. 이 map은 proposal 내부 각 위치가 foreground인지 background인지에 대한 로컬 예측이라고 볼 수 있다.

문제는 이 로컬 예측만으로는 proposal 안에 여러 개의 겹친 객체가 있을 때 **dominant instance만 분리**하기 어렵다는 점이다. 이를 해결하기 위해 논문은 **instance-aware denoising autoencoder**를 추가한다.

#### Instance-aware denoising autoencoder

confidence map $\mathbf{C}$를 벡터화하여 긴 벡터 $\tilde{\mathbf{C}}$로 만든다. 논문에 따르면 그 차원은 $40 \times 40 \times 2$에 해당한다. 이 벡터는 encoder를 통해 hidden representation $\mathbf{h}$로 압축된다.

$$
\mathbf{h} = \Phi(\tilde{\mathbf{C}})
$$

그 다음 decoder가 hidden representation으로부터 복원 벡터 $\mathbf{v}$를 생성한다.

$$
\mathbf{v} = \Phi'(\mathbf{h})
$$

여기서 $\mathbf{h}$는 proposal 전체에 대한 global 정보를 담는 compact representation이다. 저자들의 의도는 단순히 각 위치를 독립적으로 판단하는 것이 아니라, proposal 전체의 구조를 보고 “이 proposal에서 주 객체가 무엇인가”를 결정하도록 만드는 것이다. decoder는 그 global 정보를 바탕으로 noisy한 다른 객체들의 응답을 줄이고, dominant instance의 foreground mask를 복원한다.

구현 측면에서는 encoder와 decoder를 각각 fully connected layer와 ReLU로 근사한다. encoder의 출력 차원은 512, decoder의 출력 차원은 3200으로 설정했다. 마지막 출력은 다시 confidence map과 같은 spatial shape로 reshape되어 mask 예측에 사용된다.

이 segmentation branch의 학습에는 **pixel-wise cross-entropy loss**를 사용한다. 즉, 각 픽셀이 dominant object foreground인지 아닌지를 supervised하게 학습한다.

이 모듈의 핵심 의미는, 단순한 convolutional local predictor를 넘어 **proposal 단위의 전역적 판단(global inference)** 을 가능하게 만든다는 점이다.

### 3.2 Reversible Proposal Refinement Sub-network

이 sub-network 역시 VGG-16 기반이다. 각 proposal에 대해 ROI pooling으로 $7 \times 7$ feature map을 추출하고, 이를 두 개의 fully connected layer에 통과시킨다. 출력은 두 가지다.

첫째는 $K+1$개 클래스에 대한 confidence $p_t$이다. 여기서 $K$는 object class 수이고, 추가 1개는 background class이다.
둘째는 각 object class에 대응하는 bounding box regression offset $\mathbf{o}_{t,k}$이다.

#### Segmentation-aware features

이 논문의 중요한 설계는 refinement network가 segmentation branch의 출력을 활용한다는 점이다. 구체적으로, segmentation sub-network가 생성한 mask prediction $\mathbf{v}$를 refinement network의 마지막 fully connected feature와 **concatenate**하여 **segmentation-aware feature representation**을 만든다.

이 feature는 물체의 픽셀 수준 경계와 형상을 반영하므로, 단순한 appearance feature보다 더 정확한 localization과 classification에 도움이 된다. 즉, refinement network는 “이 proposal 안에 어떤 객체가 있는가”뿐 아니라, “그 객체가 proposal 내부 어디를 점유하는가”라는 segmentation 정보를 함께 사용하게 된다.

이 branch는 classification에 대해 softmax loss를, localization에 대해 smooth $L_1$ loss를 사용한다.

### 3.3 Reversible Gate

proposal마다 적절한 refinement iteration 수가 다르다는 문제를 해결하기 위해 논문은 reversible gate를 도입한다.

$t$번째 iteration에서 각 proposal은 category confidence $p_t$를 출력한다. 모든 iteration을 수행한 뒤, 가장 높은 category-level confidence를 보인 iteration을 최적 iteration $t'$로 선택한다. 이때 해당 iteration의 gate만 활성화된다.

즉, 최종 출력은 마지막 iteration 결과가 아니라, **confidence가 가장 높았던 iteration의 결과**이다. proposal 위치, class label, segmentation mask 모두 그 $t'$번째 결과를 최종값으로 사용한다.

학습 시에도 같은 원리를 적용한다. 모든 iteration의 loss를 다 사용하는 것이 아니라, 각 proposal에 대해 $1$번째부터 $t'$번째까지의 loss만 사용하고 그 이후 loss는 버린다. 따라서 네트워크는 각 proposal에 대해 “필요한 만큼만 refinement” 하도록 학습된다.

이 아이디어는 recurrent refinement를 무조건 많이 하는 것이 항상 좋은 것은 아니라는 관찰에 기반한다. 너무 많이 refinement하면 오히려 proposal이 drift할 수도 있다. 이 논문은 confidence를 stopping criterion으로 사용하여 이를 완화하려 했다.

### 3.4 수식과 학습 목표

논문은 proposal의 초기 위치를

$$
\mathbf{l}_0 = (l^x, l^y, l^w, l^h)
$$

로 둔다. 이는 bounding box의 중심 좌표와 너비, 높이이다. ground truth box는 $\tilde{\mathbf{l}}$로 표기한다.

$t$번째 iteration에서 refinement network는 class별 bounding box offset $\mathbf{o}*{t,k}$와 class confidence $p_t$를 예측한다. ground truth offset은 현재 proposal 위치 $\mathbf{l}*{t-1}$와 ground truth box $\tilde{\mathbf{l}}$ 사이의 변환으로 계산된다.

$$
\tilde{\mathbf{o}}*t = f^l(\mathbf{l}*{t-1}, \tilde{\mathbf{l}})
$$

이때 $f^l(\cdot)$는 Fast R-CNN류 box regression에서 쓰는, 중심점 이동과 로그 스케일 width/height shift를 포함하는 변환이다. refined box는 예측 offset을 inverse transform에 넣어 계산한다.

$$
\mathbf{l}*t = {f^l}^{-1}(\mathbf{l}*{t-1}, \mathbf{o}_{t,g})
$$

여기서 $g$는 ground truth class이다.

각 iteration의 multi-task loss는 다음과 같다.

$$
J_t = J_{\text{cls}}(p_t, g) + \mathbf{1}[g \ge 1] J_{\text{loc}}(\mathbf{o}_{t,g}, \tilde{\mathbf{o}}*t) + \mathbf{1}[g \ge 1] J*{\text{seg}}(\mathbf{v}_t, \tilde{\mathbf{v}}_t)
$$

각 항의 의미는 다음과 같다.

첫째,
$$
J_{\text{cls}} = -\log p_{t,g}
$$
는 proposal의 클래스 분류 loss이다.

둘째, $J_{\text{loc}}$는 smooth $L_1$ loss이며, bounding box regression 오차를 줄인다.

셋째, $J_{\text{seg}}$는 pixel-wise cross-entropy loss로 dominant instance의 foreground mask를 학습한다.

$\mathbf{1}[g \ge 1]$는 foreground proposal에만 localization loss와 segmentation loss를 적용하겠다는 뜻이다. background proposal에서는 이 두 loss를 0으로 둔다. 이는 당연한 설계인데, background에는 맞출 객체 box도, dominant instance mask도 없기 때문이다.

reversible gate가 선택한 최적 iteration을 $t'$라고 할 때, 전체 proposal loss는

$$
J = \sum_{t \le t'} J_t
$$

가 된다. 즉, proposal마다 서로 다른 길이의 recurrent supervision을 받는다.

### 3.5 학습과 추론 절차

학습은 두 단계로 이루어진다.

먼저 reversible gate 없이, 모든 proposal에 대해 최대 iteration 수 $T$까지 수행한 결과를 사용하여 네트워크를 pre-train한다. 논문에서 $T=4$로 설정했다. 그 다음 이 pre-trained model을 초기값으로 하여 reversible gate를 켠 상태에서 fine-tuning한다. 저자들은 training 초기에 confidence prediction이 불안정하기 때문에, 바로 reversible gate를 사용하면 최적 iteration 선택이 신뢰하기 어렵다고 본다. 그래서 먼저 gate 없이 학습한 뒤 gate를 사용하는 2-stage training을 채택했다.

테스트 시에는 모든 proposal을 $T$번 refinement하고, 각 iteration에서의 predicted confidence를 비교하여 최적 iteration $t'$를 선택한다. 최종 class, refined box, mask는 모두 $t'$번째 결과를 사용한다. 그리고 모든 proposal의 결과를 결합해 최종 instance segmentation을 생성한다.

## 4. 실험 및 결과

### 데이터셋과 평가 설정

논문은 **PASCAL VOC 2012 validation segmentation benchmark**에서 성능을 평가한다. 비교 설정은 두 가지로 나뉜다.

하나는 SBD annotation 기준 평가이고, 다른 하나는 VOC 2012 segmentation validation annotation 기준 평가이다. 논문은 VOC 2012의 annotation이 instance boundary를 더 정교하게 반영한다고 설명한다. 예를 들어 bicycle 같은 클래스에서 VOC 2012는 골격 구조까지 세심하게 라벨링하지만, SBD는 더 거친 영역을 주는 경우가 있다. 따라서 특히 [15], [10], [17]과의 비교에서는 VOC 2012 validation annotation 기준 결과를 많이 사용한다.

평가 지표는 **$AP^r$**이다. 이는 instance segmentation에서 예측 mask와 ground-truth mask 사이 IoU 기준을 적용하여 average precision을 계산하는 지표다. 논문은 IoU 0.5, 0.6, 0.7에서 결과를 제시한다. IoU threshold가 높아질수록 단순히 물체를 찾는 것뿐 아니라 **정확한 mask 경계**를 요구하므로 더 어려운 평가가 된다.

### 구현 세부사항

backbone은 VGG-16이며, Caffe 기반 Fast R-CNN 구현을 바탕으로 fine-tuning한다. 학습 시 mini-batch마다 proposal 64개를 사용하며, 그중 25%는 IoU 0.5 이상인 foreground, 나머지는 background다. 데이터 증강으로 horizontal flipping을 사용한다.

최대 refinement iteration 수는 $T=4$이다. 논문은 4회보다 더 많은 iteration을 써도 눈에 띄는 성능 향상이 없었다고 보고한다. 학습은 reversible gate 없이 120k iteration, 이후 gate를 사용해 100k iteration fine-tuning한다. 테스트 속도는 proposal 생성 시간을 제외하고 이미지당 약 1초라고 보고한다.

### 주요 성능 비교

#### SBD annotation 기준 결과

Table 1에서 mean $AP^r$ at IoU 0.5는 다음과 같다.

* SDS: 49.7
* HC: 60.0
* R2-IOS: 68.8

즉, R2-IOS는 SDS 대비 19.1%p, HC 대비 8.8%p 높다.
IoU 0.7에서도 HC의 40.4에 비해 R2-IOS가 47.5를 기록해 7.1%p 향상되었다.

이 결과는 단순히 더 많은 proposal을 쓰거나 detection 성능만 높인 것이 아니라, 실제로 **정확한 instance mask 예측**이 향상되었음을 시사한다.

#### VOC 2012 annotation 기준 결과

Table 2에서 IoU 0.5 기준 mean $AP^r$는 다음과 같다.

* SDS: 43.8
* Chen et al. [17]: 46.3
* PFN: 58.7
* R2-IOS: 66.7

R2-IOS는 당시 강한 baseline인 PFN보다도 8.0%p 높다. 특히 proposal-free 접근인 PFN보다 높은 성능을 기록했다는 점이 흥미롭다. 이는 proposal-based 방법이라도 proposal refinement와 segmentation을 충분히 결합하면 매우 강력해질 수 있음을 보여준다.

Table 3에서 높은 IoU 기준에서도 개선이 유지된다.

IoU 0.6 평균 성능:

* PFN: 51.3
* R2-IOS: 58.1

IoU 0.7 평균 성능:

* PFN: 42.5
* R2-IOS: 46.2

IoU threshold가 높아질수록 segmentation mask 경계 정확도가 중요해지는데, 이 영역에서도 R2-IOS가 우세하다. 이는 논문의 recursive refinement와 autoencoder가 실제로 더 정교한 mask를 만들어낸다는 주장과 잘 맞는다.

### 정성적 결과 해석

논문은 Figure 3, Figure 4, Figure 5를 통해 qualitative result도 제시한다. SDS와 비교했을 때, R2-IOS는 다양한 스케일, occlusion, background clutter가 있는 상황에서 더 안정적으로 객체를 찾고 분리한다. 특히 object proposal이 다소 부정확한 경우에도 refinement를 통해 box를 개선하고, segmentation이 다시 좋아지는 선순환을 보인다.

또한 autoencoder가 없는 모델은 같은 proposal 안의 여러 객체를 함께 foreground로 예측하는 경향이 있었고, autoencoder가 있는 모델은 dominant object만 더 잘 분리했다. 이는 global context를 압축한 hidden representation이 실제로 instance disambiguation에 기여했음을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **instance segmentation과 proposal refinement를 상호보완적인 반복 구조로 통합했다는 점**이다. 이전 방법들이 보통 detection과 segmentation을 직렬적 단계로 처리했다면, 이 논문은 두 작업을 recursive하게 연결해 성능을 함께 끌어올린다. 이는 개념적으로도 설득력이 있고, 실험적으로도 충분한 개선 폭으로 입증되었다.

두 번째 강점은 **reversible gate를 통한 proposal별 adaptive iteration 선택**이다. recurrent refinement를 무조건 동일 횟수로 적용하지 않고, proposal마다 다른 최적점을 택한다는 발상은 단순하지만 효과적이다. 실험에서도 recursive_4 대비 최종 R2-IOS가 추가 향상을 보인다.

세 번째 강점은 **instance-aware denoising autoencoder**다. 이 모듈은 논문에서 가장 큰 성능 향상을 가져오는 요소 중 하나로 보인다. ablation에 따르면 autoencoder 제거 시 평균 성능이 66.7에서 54.2로 크게 떨어진다. 이는 dominant instance 분리를 위해 global reasoning이 중요하다는 것을 강하게 뒷받침한다.

네 번째 강점은 **높은 IoU threshold에서의 성능 개선**이다. IoU 0.7에서도 baseline 대비 성능 향상이 유지된다는 것은, 단순히 box recall만 좋아진 것이 아니라 실제 mask boundary 품질이 향상되었다는 근거다.

하지만 한계도 분명하다.

우선 이 방법은 **object proposal에 여전히 의존**한다. selective search로 이미지당 약 2,000개의 proposal을 생성하는 구조이므로, end-to-end one-stage 구조에 비해 계산량과 시스템 복잡성이 크다. 논문도 proposal 생성 시간은 테스트 속도에서 제외하고 있다. 따라서 전체 시스템 효율을 논할 때는 주의가 필요하다.

또한 recursive refinement가 효과적이라고 해도, 이는 여전히 **proposal 단위 처리 방식**이다. proposal 수가 많아질수록 계산 비용이 증가하고, proposal quality가 너무 낮은 경우에는 refinement만으로 복구가 어려울 가능성도 있다. 논문은 이런 failure case를 체계적으로 분석하지는 않는다.

reversible gate는 category confidence를 기준으로 최적 iteration을 선택하는데, 이 기준이 항상 segmentation 품질과 완벽히 일치한다고 보장되지는 않는다. 논문은 실험적으로 confidence가 좋은 indicator라고 말하지만, 이는 경험적 선택이며 이론적으로 정당화되지는 않는다.

또 하나의 한계는 dominant instance 정의가 **proposal과 가장 많이 겹치는 객체**라는 점이다. 이는 proposal 중심의 학습 목표로서는 자연스럽지만, proposal 내부 구조가 매우 복잡하거나 큰 객체와 작은 객체가 특이하게 겹치는 경우에는 이 정의가 segmentation의 궁극적 목적과 어긋날 수 있다.

비판적으로 보면, 이 논문은 당시 기준으로는 매우 강력한 통합 설계이지만, 구조가 다소 복잡하고 모듈 간 상호작용이 많다. backbone feature, segmentation feature, autoencoder, bounding box regression, reversible gate를 모두 결합하고 있어 구현 난이도가 높은 편이다. 그럼에도 각 구성요소에 대한 ablation이 제시되어 있어, 논문이 제안한 핵심 설계가 실제 성능 개선에 기여했다는 점은 비교적 설득력 있게 보여준다.

## 6. 결론

이 논문은 instance-level object segmentation을 위해 **Reversible Recursive Instance-level Object Segmentation (R2-IOS)** 를 제안했다. 핵심 기여는 다음과 같이 정리할 수 있다.

첫째, instance segmentation과 proposal refinement를 하나의 프레임워크 안에서 recursive하게 결합했다.
둘째, reversible gate를 통해 proposal마다 최적 refinement iteration 수를 adaptively 선택했다.
셋째, instance-aware denoising autoencoder를 통해 proposal 내부의 여러 중첩 객체 중 dominant instance를 더 잘 분리했다.
넷째, segmentation-aware feature representation을 도입해 segmentation branch와 refinement branch가 서로를 강화하도록 만들었다.

실험 결과는 이 설계가 단순한 아이디어 수준이 아니라 실제로 강력한 성능 향상으로 이어짐을 보여준다. 특히 mean $AP^r$ 66.7% at IoU 0.5는 당시 state-of-the-art를 크게 넘어서는 결과였고, 높은 IoU 기준에서도 개선이 유지되었다.

실제 적용 관점에서 보면, 이 논문은 이후의 instance segmentation 연구에서 중요한 전환점을 보여준다. 즉, detection과 segmentation을 따로 처리하기보다, **반복적 상호보완 구조로 함께 최적화하는 방향**이 유망하다는 점을 보여주었다. 또한 proposal refinement의 stopping을 adaptive하게 만드는 아이디어는 다른 recurrent vision model에도 확장 가능하다는 점에서 의미가 있다.

향후 연구 방향으로 저자들은 LSTM을 이용해 주변 객체와 장면으로부터 더 긴 spatial contextual dependency를 활용하겠다고 언급한다. 논문 본문만 기준으로 보면, 이는 아직 제안 수준이며 실제로 구현되지는 않았다. 다만 본 연구가 제시한 recursive refinement와 global context 활용이라는 관점은 이후 더 발전된 instance segmentation 모델들로 이어질 수 있는 기반을 제공했다고 평가할 수 있다.
