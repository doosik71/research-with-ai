# Attention to Scale: Scale-aware Semantic Image Segmentation

- **저자**: Liang-Chieh Chen, Yi Yang, Jiang Wang, Wei Xu, Alan L. Yuille
- **발표연도**: 2016
- **arXiv**: https://arxiv.org/abs/1511.03339

## 1. 논문 개요

이 논문은 semantic image segmentation에서 매우 중요한 문제인 **multi-scale feature fusion**을 더 잘 수행하는 방법을 다룬다. semantic segmentation은 이미지의 모든 픽셀에 클래스를 할당해야 하므로, 작은 물체와 큰 물체를 동시에 잘 처리해야 한다. 그런데 하나의 고정된 receptive field나 단일 입력 해상도만으로는 다양한 크기의 객체를 안정적으로 분할하기 어렵다. 이 때문에 기존 FCN 계열 방법들은 여러 스케일의 특징을 결합하는 전략을 사용해 왔다.

논문의 핵심 문제의식은 다음과 같다. 여러 스케일의 입력 이미지를 shared network에 넣고 나온 결과를 합칠 때, 기존 방식은 보통 average-pooling이나 max-pooling을 사용한다. 그러나 이런 방식은 각 픽셀마다 어떤 스케일 정보가 더 중요한지를 세밀하게 반영하지 못한다. 예를 들어 작은 사람과 큰 사람 객체가 동시에 있는 이미지에서, 모든 위치에 동일한 방식으로 스케일을 합치는 것은 비효율적일 수 있다.

저자들은 이 문제를 해결하기 위해 **scale dimension에 대한 attention mechanism**을 도입한다. 즉, 각 픽셀 위치마다 각 스케일이 얼마나 중요한지를 soft weight로 학습하고, 그 가중합으로 최종 score map을 만든다. 논문은 이 방법이 단순 pooling보다 더 좋은 성능을 보일 뿐 아니라, 실제로 어떤 위치에서 어떤 스케일을 더 참고했는지 시각화할 수 있다는 점도 강조한다. 이는 단순 성능 향상뿐 아니라 모델 해석 가능성 측면에서도 의미가 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 명확하다. **“픽셀마다 적절한 스케일은 다를 수 있으니, 스케일별 특징을 고정 규칙으로 합치지 말고 attention으로 적응적으로 섞자”**는 것이다.

기존 share-net 계열 접근은 여러 해상도의 입력을 같은 네트워크에 통과시킨 뒤 결과를 합친다. 이때 average-pooling은 모든 스케일을 동일하게 취급하고, max-pooling은 일부 강한 반응만 선택한다. 하지만 segmentation에서는 물체 크기와 주변 문맥이 위치마다 달라서, 어떤 픽셀은 fine scale 정보가 중요하고 어떤 픽셀은 coarse scale 정보가 더 중요하다. 저자들은 이 선택을 사람이 설계한 규칙이 아니라 학습 가능한 attention module이 하도록 만든다.

논문이 기존 접근과 구별되는 지점은 두 가지다. 첫째, attention을 spatial dimension이나 temporal dimension이 아니라 **scale dimension**에 적용했다. 둘째, 단순히 scale attention만 넣은 것이 아니라, **각 스케일의 출력에 extra supervision을 추가**해 각 스케일에서 생성되는 score map 자체를 더 discriminative하게 만들었다. 저자들은 이 extra supervision이 multi-scale feature merging에서 사실상 필수적이라고 실험으로 보여준다.

즉 이 논문의 기여는 단순히 “attention을 넣었다”가 아니라, **share-net 기반 multi-scale segmentation을 end-to-end로 학습하면서, scale attention과 scale-wise supervision을 함께 결합해 효과적인 fusion 구조를 제안했다**는 데 있다.

## 3. 상세 방법 설명

전체 시스템은 DeepLab-LargeFOV를 backbone으로 사용한다. 입력 이미지를 여러 스케일 $s \in \{1, \dots, S\}$로 리사이즈한 뒤, 각 이미지를 동일한 FCN에 통과시킨다. 여기서 모든 스케일은 **가중치를 공유하는 shared deep network**를 사용한다. 각 스케일에서 클래스별 score map이 생성되며, 이를 동일 해상도로 bilinear interpolation하여 맞춘다.

각 스케일 $s$에서 위치 $i$, 클래스 $c$에 대한 score를 $f^s_{i,c}$라고 두면, 최종 결합 score $g_{i,c}$는 다음과 같이 정의된다.

$$
g_{i,c} = \sum_{s=1}^{S} w_i^s \cdot f_{i,c}^s
$$

여기서 $w_i^s$는 위치 $i$에서 스케일 $s$가 얼마나 중요한지를 나타내는 attention weight이다. 이 weight는 attention model이 생성한 값 $h_i^s$를 softmax에 통과시켜 계산된다.

$$
w_i^s = \frac{\exp(h_i^s)}{\sum_{t=1}^{S} \exp(h_i^t)}
$$

중요한 점은 $w_i^s$가 클래스 채널 $c$마다 다른 것이 아니라, **같은 위치와 같은 스케일에 대해 모든 클래스 채널에 공통으로 적용된다**는 것이다. 즉 attention은 “이 위치에서 어느 스케일을 더 볼 것인가”를 결정하고, 그 결정이 그 위치의 클래스 score 전체에 반영된다.

attention model 자체도 FCN으로 구현된다. 입력으로는 VGG-16의 convolutionalized `fc7` feature를 사용하며, 구조는 두 층이다. 첫 번째 층은 $3 \times 3$ 커널의 512개 필터를 가지며, 두 번째 층은 $1 \times 1$ 커널의 $S$개 필터를 가진다. 최종적으로 스케일 수만큼의 dense attention score map $h_i^s$를 출력한다. 저자들은 `fc6`, `fc7`, `fc8` 중 어떤 feature를 attention 입력으로 쓰는지도 실험했는데, `fc7`과 `fc6`는 비슷했고 `fc8`은 약간 더 나빴다고 보고한다.

이 구조는 average-pooling과 max-pooling을 일반화한 것으로 볼 수 있다. average-pooling은 모든 $w_i^s = 1/S$인 특수한 경우이고, max-pooling은 합 대신 최대값을 취하는 극단적 선택에 해당한다. 그러나 attention은 hard selection이 아니라 differentiable한 soft weighting이므로, 전체 네트워크를 standard backpropagation으로 end-to-end 학습할 수 있다.

학습은 pixel-wise cross-entropy loss를 사용한다. 최종 merged output에 softmax를 적용하고, 모든 위치에 대해 평균 cross-entropy를 최소화한다. 여기에 더해 논문은 **extra supervision**을 각 스케일의 FCN 출력에도 부여한다. 즉 손실 함수는 최종 출력용 1개와 각 스케일 출력용 $S$개를 합쳐 총 $1+S$개의 cross-entropy loss로 구성되며, 각 항의 가중치는 동일하다. 저자들의 설명에 따르면, 여러 스케일의 결과를 합치기 전에 각 스케일 score map이 이미 충분히 discriminative해야 fusion의 효과가 좋다. 실제 실험에서도 extra supervision 유무가 성능 차이에 매우 크게 작용한다.

추론 시에는 여러 스케일 입력을 동시에 통과시키고, attention이 만든 weight map으로 score map들을 가중합한 뒤 최종 segmentation 결과를 얻는다. 논문은 이때의 attention map을 직접 시각화해, 작은 객체에는 fine scale, 큰 객체나 background context에는 coarse scale이 더 높은 weight를 받는 경향을 보여준다.

## 4. 실험 및 결과

실험은 PASCAL-Person-Part, PASCAL VOC 2012, 그리고 MS-COCO 2014 subset의 세 데이터셋에서 수행되었다. 공통 backbone은 DeepLab-LargeFOV이며, 성능 평가는 class 평균 pixel intersection-over-union, 즉 mIOU를 사용한다. 학습은 SGD로 수행했고, mini-batch size는 30, 초기 learning rate는 0.001이며 마지막 classifier layer는 0.01을 사용했다. 2000 iteration 이후 learning rate를 0.1배로 줄였고, momentum은 0.9, weight decay는 0.0005였다. 보고된 환경에서는 NVIDIA Tesla K40 GPU에서 fine-tuning에 약 21시간이 걸렸고, PASCAL 이미지 한 장당 평균 추론 시간은 350ms였다.

PASCAL-Person-Part에서는 사람의 part segmentation을 평가한다. 클래스는 Head, Torso, Upper/Lower Arms, Upper/Lower Legs, Background의 7개이다. baseline인 단일 스케일 DeepLab-LargeFOV는 validation에서 51.91%를 기록했다. 여기에 scale fusion을 적용하면 성능이 상승했고, attention과 extra supervision을 함께 사용한 3-scale 설정 $\{1, 0.75, 0.5\}$이 **56.39%**로 가장 좋았다. 이는 DeepLab-MSc-LargeFOV의 53.72%보다 2.67% 높다. part별 성능을 보면 Head가 81.47%로 가장 높고, Lower Legs가 35.62%로 가장 낮다. 이는 작은 파트나 복잡한 경계가 여전히 어렵다는 점을 보여준다.

이 데이터셋에서 중요한 관찰은 세 가지다. 첫째, 단일 스케일보다 multi-scale이 확실히 유리하다. 둘째, attention은 average-pooling과 max-pooling보다 일관되게 낫다. 셋째, **extra supervision이 없으면 특히 3-scale fusion에서 성능이 충분히 나오지 않는다**. 저자들은 이 결과를 통해 multi-scale merging에서는 각 스케일의 출력 자체를 잘 학습시키는 것이 필수라고 주장한다.

PASCAL VOC 2012에서는 두 설정을 나눠 실험했다. 하나는 ImageNet만으로 pretrained한 경우이고, 다른 하나는 MS-COCO까지 pretraining한 더 강한 baseline이다. ImageNet pretrained setting에서 baseline DeepLab-LargeFOV는 validation mIOU 62.28%였고, attention + extra supervision + 3 scales 설정은 **69.08%**를 기록했다. 이는 baseline 대비 6.8%p 향상이며, DeepLab-MSc-LargeFOV의 64.39%보다 4.69% 높다. test set에서는 DeepLab-LargeFOV-Attention이 **71.5%**를 기록해 average-pooling 70.5%, max-pooling 70.6%보다 좋았다. 논문은 ParseNet 69.8%, TTI zoomout v2 69.6%보다도 높다고 비교한다.

MS-COCO pretrained setting에서는 baseline이 이미 강해 validation 67.58%를 기록한다. 그럼에도 attention + extra supervision + 3 scales 조합은 **71.42%**로 올라가 baseline 대비 3.84%p 향상된다. test set에서는 fully connected CRF post-processing을 사용한 변형이 **75.1%**를 기록했고, 데이터 augmentation으로 random scaling을 추가하면 **75.7%**까지 올라간다. 다만 Adelaide 77.8%, DPN 77.5% 등 당시 최고 성능 방법보다는 낮다. 저자들은 이 차이가 joint CRF training 같은 추가 기법의 유무와 관련 있다고 해석하며, 자신의 방법이 그런 기법들과 상호보완적일 수 있다고 본다.

MS-COCO subset 실험은 데이터가 더 어렵다는 점을 보여준다. baseline DeepLab-LargeFOV는 전체 mIOU가 31.22%로 낮다. 그러나 attention과 extra supervision을 쓰면 3-scale에서 **35.78%**까지 올라간다. 절대값은 낮지만 baseline 대비 4.6%p 개선이다. 또한 person 클래스만 따로 보면 baseline 68.76%에서 attention + extra supervision + 3 scales가 **72.72%**까지 올라, 스케일 변화가 큰 주요 클래스에서는 개선이 더 두드러진다. 저자들은 작은 객체 클래스의 예측 정확도가 매우 낮고 class imbalance가 심해 전체 mIOU 향상이 제한되었다고 해석한다.

정성적 결과도 논문의 중요한 근거다. attention map 시각화를 보면 scale-1 attention은 작은 객체나 작은 part에 집중하고, scale-0.75는 중간 크기 물체에, scale-0.5는 큰 객체와 배경 문맥에 높은 가중치를 주는 경향이 나타난다. 이는 저자들이 제안한 scale-aware attention이 실제로 의미 있는 방식으로 동작한다는 정성적 증거다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정과 해결 방식이 매우 자연스럽고 설득력 있다는 점이다. semantic segmentation에서 multi-scale fusion이 중요하다는 사실은 이미 알려져 있었지만, 이 논문은 “모든 위치에서 같은 방식으로 fusion하면 안 된다”는 점을 attention으로 정교하게 풀어냈다. 또한 attention map을 통해 모델이 어떤 위치에서 어떤 스케일을 활용하는지 보여주므로, 단순 성능 개선을 넘어 해석 가능성을 제공한다.

두 번째 강점은 **extra supervision의 중요성을 실험적으로 명확히 보여준 것**이다. 많은 논문이 구조를 제안하는 데 그치지만, 이 논문은 왜 3-scale fusion이 때때로 잘 안 되는지, 그리고 각 스케일 출력을 더 discriminative하게 만들어야 한다는 점을 구체적으로 분석한다. 즉, 모델 설계뿐 아니라 학습 설계에서도 실질적인 통찰을 제공한다.

세 번째 강점은 제안 방법이 완전히 새로운 backbone을 요구하지 않고, 기존 강력한 segmentation 모델인 DeepLab-LargeFOV 위에 비교적 깔끔하게 얹을 수 있다는 점이다. 따라서 확장성과 적용 가능성이 높다.

반면 한계도 분명하다. 첫째, 성능 개선은 일관되지만 폭발적이지는 않다. 특히 더 강한 baseline이나 더 어려운 데이터셋에서는 attention이 average-pooling보다 큰 차이를 못 내는 경우도 있다. 예를 들어 MS-COCO subset에서는 average-pooling과 attention의 전체 mIOU 차이가 매우 작다. 이는 scale attention만으로 해결되지 않는 어려움, 예를 들어 class imbalance나 아주 작은 객체의 분할 문제가 남아 있음을 의미한다.

둘째, 이 방법은 여러 스케일의 입력을 동시에 처리해야 하므로 학습과 추론 비용이 증가한다. 논문에서도 총 학습 시간이 vanilla DeepLab-LargeFOV의 약 두 배라고 밝힌다. 따라서 정확도 향상과 계산 비용 사이의 trade-off가 존재한다.

셋째, attention이 scale dimension에서 잘 작동한다는 것은 보여주지만, 왜 특정 데이터셋에서 2-scale이 3-scale보다 낫거나, 왜 어떤 경우 max-pooling이 비교적 robust한지에 대한 이론적 설명은 제한적이다. 논문은 가능한 해석을 제시하지만, 엄밀한 분석까지는 하지 않는다.

넷째, failure case 분석은 비교적 간단하다. 사람의 극단적인 pose나 clothing confusion이 문제라고 설명하지만, 이런 실패를 구조적으로 어떻게 해결할지에 대한 방법은 제시하지 않는다. 따라서 이 논문은 강한 실용적 개선을 보여주지만, 어려운 장면에서의 근본 문제를 완전히 해결한 것은 아니다.

## 6. 결론

이 논문은 semantic segmentation에서 multi-scale input을 효과적으로 활용하기 위해 **scale-aware attention mechanism**을 제안했다. 핵심은 각 픽셀 위치에서 각 스케일의 중요도를 softmax attention으로 계산하고, 그 가중합으로 최종 score map을 만드는 것이다. 여기에 각 스케일 출력에 대한 extra supervision을 추가해, 단순한 feature fusion이 아니라 더 discriminative한 scale-wise prediction을 학습하도록 만들었다.

실험 결과는 세 데이터셋 전반에서 일관되다. multi-scale input은 single-scale보다 좋고, attention 기반 fusion은 average/max-pooling보다 대체로 우수하며, extra supervision은 성능 향상에 매우 중요하다. 특히 attention map 시각화를 통해 작은 객체는 fine scale, 큰 객체와 배경 문맥은 coarse scale이 더 중요하다는 점을 확인한 것은 이 논문의 중요한 해석적 기여다.

실제 적용 측면에서 이 연구는 이후 semantic segmentation 모델들이 단순 multi-scale fusion을 넘어, **위치별로 어떤 해상도 정보를 활용할지 학습하는 방향**으로 발전하는 데 의미 있는 출발점이 된다. 또한 attention을 scale 축으로 확장했다는 점에서, 이후 feature pyramid, adaptive fusion, dynamic inference 같은 연구 흐름과도 연결되는 중요한 아이디어를 제공한다.
