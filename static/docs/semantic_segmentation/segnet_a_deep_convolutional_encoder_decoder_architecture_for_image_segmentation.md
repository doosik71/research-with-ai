# SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation

- **저자**: Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla
- **발표연도**: 2016
- **arXiv**: https://arxiv.org/abs/1511.00561

## 1. 논문 개요

이 논문은 semantic segmentation, 즉 이미지의 모든 픽셀에 대해 클래스 레이블을 예측하는 문제를 다룬다. 저자들은 특히 road scene understanding과 indoor scene understanding처럼 장면 전체를 조밀하게 이해해야 하는 응용을 염두에 두고, 정확도뿐 아니라 추론 시 메모리 사용량과 계산 효율까지 고려한 네트워크를 설계한다. 그 결과로 제안된 모델이 **SegNet**이다.

논문이 해결하려는 핵심 문제는, classification용 CNN이 반복적인 max-pooling과 subsampling을 거치면서 feature map의 해상도를 크게 잃어버리기 때문에, segmentation에 필요한 정밀한 경계 정보(boundary detail)가 손상된다는 점이다. segmentation에서는 단순히 “무엇이 있는가”만 아는 것이 아니라 “어디까지가 그 물체인가”를 픽셀 단위로 맞춰야 하므로, 낮은 해상도의 표현을 다시 입력 해상도 수준으로 복원하는 decoder가 매우 중요하다.

이 문제가 중요한 이유는 실제 장면 이해에서는 큰 영역을 매끄럽게 분할하는 능력과, 자동차·보행자·기둥·표지판처럼 작은 물체의 경계를 세밀하게 구분하는 능력이 동시에 필요하기 때문이다. 자율주행, 로보틱스, AR 같은 실제 응용에서는 정확도만 높고 메모리나 속도가 비현실적인 모델보다, 충분히 정확하면서도 실시간에 가까운 효율을 갖는 모델이 더 실용적이다. 이 논문은 바로 그 지점을 겨냥한다.

## 2. 핵심 아이디어

SegNet의 중심 아이디어는 **encoder에서 max-pooling할 때 발생한 위치 정보(pooling indices)를 저장해 두었다가, decoder에서 이를 사용해 비선형 upsampling을 수행하는 것**이다. 즉, decoder가 upsampling 자체를 학습하는 대신, encoder가 이미 선택한 “최대 활성 위치”를 그대로 복원에 활용한다. 이렇게 하면 decoder가 full encoder feature map을 전부 저장해 둘 필요가 없고, 작은 저장 비용으로 경계 정보를 어느 정도 유지할 수 있다.

이 설계의 직관은 명확하다. max-pooling은 값만 남기고 위치를 잃어버리기 쉬운데, SegNet은 값의 위치를 기억해 두었다가 복원 단계에서 다시 그 자리에 feature를 배치한다. 그 결과 upsampled feature map은 처음에는 sparse하지만, 이후 trainable convolution을 통해 dense한 feature map으로 바뀌고, 최종적으로 픽셀별 분류가 가능해진다.

기존 접근과의 차별점은 크게 세 가지다. 첫째, FCN처럼 encoder feature map 전체를 decoder에 전달하는 방식보다 추론 메모리 비용이 훨씬 작다. 둘째, DeconvNet처럼 매우 큰 fully connected 계층을 유지하지 않기 때문에 파라미터 수가 크게 줄어들고 end-to-end 학습이 쉬워진다. 셋째, 논문은 단순히 새 구조를 제안하는 데 그치지 않고, 다양한 decoder 변형들과 비교하여 **memory versus accuracy trade-off**를 체계적으로 분석한다. 이 점이 논문의 중요한 기여다.

## 3. 상세 방법 설명

SegNet은 크게 **encoder network**, **decoder network**, 그리고 마지막의 **pixel-wise softmax classifier**로 구성된다. encoder는 VGG16의 앞쪽 13개 convolution layer와 동일한 topology를 가진다. 다만 classification에 쓰이는 fully connected layer들은 제거한다. 이 결정은 두 가지 효과를 낸다. 하나는 파라미터 수를 크게 줄여 학습을 쉽게 만드는 것이고, 다른 하나는 너무 깊은 압축으로 인해 공간 해상도가 지나치게 사라지는 것을 막는 것이다. 논문에 따르면 encoder 파라미터 수는 134M에서 14.7M 수준으로 줄어든다.

각 encoder 블록은 convolution, batch normalization, ReLU, 그리고 $2 \times 2$ max-pooling(stride 2)으로 이루어진다. max-pooling은 작은 위치 변화에 대한 불변성을 제공하고 receptive field를 넓혀 주지만, 동시에 feature map의 해상도를 절반으로 줄인다. segmentation에서는 이것이 경계 손실로 이어질 수 있다. 그래서 SegNet은 pooling 시점마다 각 pooling window 안에서 최대값이 선택된 위치, 즉 **pooling index**를 저장한다.

decoder는 encoder와 1:1로 대응되는 계층 구조를 가진다. 각 decoder는 대응 encoder에서 전달받은 pooling index를 사용하여 입력 feature map을 upsample한다. 이 단계는 값을 학습으로 퍼뜨리는 것이 아니라, 저장된 위치에 다시 feature를 놓는 방식이므로 upsampled map이 sparse하다. 이후 decoder는 trainable convolution filter bank를 적용하여 sparse map을 dense한 feature map으로 변환한다. 다시 말해 SegNet에서 학습되는 것은 “어디에 복원할지”가 아니라, “복원된 sparse 표현을 어떻게 해석하고 정제할지”이다.

논문 본문 발췌에는 식 번호가 직접 제시되어 있지 않지만, 구조를 수식으로 요약하면 다음과 같이 이해할 수 있다. encoder의 어떤 층에서 입력 feature를 $x$라 하면, convolution, batch normalization, ReLU를 거쳐
$z = \mathrm{ReLU}(\mathrm{BN}(W * x))$
를 얻고, 여기에 max-pooling을 적용해 pooled output과 pooling index를 생성한다. decoder에서는 pooling index를 이용해
$u = \mathrm{Unpool}(z, \mathrm{index})$
를 만들고, 그 뒤 convolution으로
$y = W_d * u$
를 계산해 dense feature map을 얻는다. 마지막으로 각 픽셀 위치 $i$에 대해 softmax를 적용하여 클래스 확률 $p_i(k)$를 얻는다.

학습 목표는 픽셀 단위 cross-entropy loss이다. 논문은 mini-batch 안의 모든 픽셀에 대해 손실을 합산한다고 설명한다. 표준적인 형태로 쓰면
$$
\mathcal{L} = - \sum_i \sum_{k=1}^{K} w_{y_i}\,\mathbf{1}[y_i = k]\log p_i(k)
$$
처럼 볼 수 있다. 여기서 $y_i$는 픽셀 $i$의 정답 클래스, $p_i(k)$는 클래스 $k$의 예측 확률, $w_{y_i}$는 class balancing을 위한 가중치다. 논문은 특히 클래스 불균형이 심한 경우 **median frequency balancing**을 사용한다. 이는 각 클래스 빈도의 중앙값을 해당 클래스 빈도로 나눈 값을 가중치로 주는 방식이다. 따라서 road, sky, building처럼 자주 나오는 클래스는 작은 가중치를 받고, sign, pole, pedestrian처럼 드문 클래스는 더 큰 가중치를 받는다.

논문은 full SegNet 외에도 **SegNet-Basic**이라는 소형 버전을 만들어 decoder 설계를 분석한다. 이 버전은 4 encoder와 4 decoder로 구성되며, kernel size를 모두 $7 \times 7$로 설정한다. 저자들은 이를 FCN 방식의 decoder, bilinear interpolation, encoder feature addition 등 여러 변형과 비교하면서 어떤 요소가 성능과 메모리에 영향을 주는지 정량적으로 살핀다. 이 분석을 통해 논문은 단순히 “SegNet이 좋다”고 주장하는 것이 아니라, 어떤 자원을 쓸 수 있느냐에 따라 어떤 decoder 설계가 적절한지 설명한다.

## 4. 실험 및 결과

논문은 크게 두 가지 벤치마크를 사용한다. 첫 번째는 **CamVid road scene segmentation**이고, 두 번째는 **SUN RGB-D indoor scene segmentation**이다. CamVid는 11개 클래스를 다루며, 기본 실험에서는 367장 train, 233장 test 이미지가 사용된다. 추가 실험에서는 CamVid, KITTI, LabelMe 등 여러 데이터를 합쳐 약 3.5K 이미지 규모로 확장해 학습한다. SUN RGB-D는 훨씬 어려운 indoor benchmark로, 37개 클래스를 가지며 5285장 train, 5050장 test 이미지가 사용된다. 논문은 SUN RGB-D에서는 RGB만 사용하고 depth는 사용하지 않는다.

평가 지표는 네 가지다. **global accuracy (G)**는 전체 픽셀 중 맞춘 비율이고, **class average accuracy (C)**는 클래스별 정확도의 평균이다. **mIoU**는 false positive까지 엄격히 반영하는 대표적인 segmentation 지표다. 여기에 더해 논문은 **boundary F1-measure (BF)**를 사용한다. 이는 클래스 경계가 실제 경계와 얼마나 잘 맞는지 보는 지표로, 단순 면적 기반 지표인 mIoU가 놓칠 수 있는 contour quality를 보완하기 위해 도입되었다. 이 선택은 SegNet의 설계 목적이 boundary preservation과 직접 연결되기 때문에 매우 적절하다.

decoder variant 분석(Table 1)에서는 몇 가지 중요한 결론이 나온다. bilinear interpolation처럼 upsampling을 학습하지 않는 방식은 가장 성능이 낮다. 반면 SegNet-Basic과 FCN-Basic은 전반적으로 비슷한 정확도를 보인다. 하지만 FCN-Basic은 encoder feature map을 저장해야 해서 추론 메모리 비용이 크고, SegNet-Basic은 pooling indices만 저장하면 되므로 훨씬 효율적이다. 예시로 논문은 FCN-Basic의 첫 층 encoder feature map 저장이 약 11MB, 차원 축소를 해도 약 1.9MB인 반면, SegNet의 pooling indices 저장은 약 0.17MB 수준이라고 설명한다. 또한 BF 점수는 encoder 쪽 정보가 경계 복원에 중요함을 보여 준다. full encoder feature를 쓰는 큰 모델들이 가장 성능이 좋지만, 그만큼 메모리와 시간 비용도 커진다.

CamVid에서의 본격 벤치마크(Table 2, Table 3)를 보면, SegNet은 전통적인 hand-crafted feature + CRF 계열 방법들과 비교해 매우 경쟁력 있는 성능을 보인다. 특히 3.5K 데이터셋으로 학습한 SegNet은 class average 71.20, global accuracy 90.40, mIoU 60.10, BF 46.84를 기록한다. 저자들은 특히 작은 클래스와 얇은 구조물에서의 개선을 강조한다. 고전적 CRF 기반 방법과 비교하면 11개 클래스 중 8개에서 더 정확하다고 보고한다. 이는 deep feature learning 자체가 이미 강한 unary와 구조적 정보를 제공해, 별도 CRF 후처리의 필요성을 줄일 수 있음을 시사한다.

동일한 조건에서 DeepLab-LargeFOV, FCN, DeconvNet과 비교한 결과도 흥미롭다. CamVid에서는 SegNet과 DeconvNet이 가장 높은 전반적 성능을 보이지만, DeconvNet은 훨씬 큰 계산 비용이 필요하다. DeepLab-LargeFOV는 가장 작고 빠른 모델이지만 boundary accuracy가 낮고 작은 객체를 놓치는 경향이 있다. FCN은 bilinear upsampling보다 learned deconvolution이 확실히 낫지만, overall로는 SegNet보다 뒤처진다. 특히 BF 점수에서 SegNet과 DeconvNet이 강하다. 즉, SegNet은 “크고 무거운 모델만이 높은 경계 정확도를 얻는다”는 인상을 일부 깨고, 더 작은 메모리로도 경쟁력 있는 성능을 낼 수 있음을 보여 준다.

SUN RGB-D 결과(Table 4, Table 5)는 논문의 균형 잡힌 면을 보여 준다. 이 데이터셋에서는 모든 모델이 훨씬 낮은 성능을 보인다. SegNet은 G, C, BF에서는 가장 좋거나 가장 경쟁력 있지만, mIoU는 DeepLab-LargeFOV가 약간 더 높다. 예를 들어 SegNet은 최고 시점에서 G 72.63, C 44.76, mIoU 31.84, BF 12.66을 기록했고, DeepLab-LargeFOV는 mIoU 32.08을 기록한다. 하지만 실내 장면은 클래스 수가 많고 작은 물체가 많으며 spatial arrangement도 매우 다양하기 때문에, 저자들은 이 문제 자체가 outdoor scene보다 훨씬 어렵다고 해석한다. 클래스별 정확도를 보면 wall, floor, chair, bed 같은 큰 클래스는 상대적으로 잘 맞지만, floor mat, shower curtain, bag처럼 작은 클래스는 매우 낮다. 이는 데이터 불균형과 복잡한 장면 구조가 여전히 큰 도전임을 보여 준다.

계산 효율(Table 6)도 논문의 핵심 메시지 중 하나다. SegNet의 forward pass는 DeepLab보다 느리지만, inference memory는 가장 효율적이다. 논문 표에 따르면 inference memory는 SegNet 1052MB, DeepLab-LargeFOV 1993MB, FCN 1806MB, DeconvNet 1872MB이다. 모델 크기 역시 SegNet 117MB로 FCN 539MB, DeconvNet 877MB보다 훨씬 작다. 즉, SegNet은 절대적으로 가장 빠른 모델은 아니지만, **정확도 대비 메모리 효율**이라는 관점에서 매우 실용적인 선택지로 제시된다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 설계 선택이 매우 실용적이라는 점이다. SegNet은 단순히 새로운 네트워크 블록을 제안하는 것이 아니라, 왜 segmentation에서 decoder가 중요한지, 왜 boundary 정보가 중요하며, 왜 추론 메모리가 실제 응용에서 병목이 되는지를 분명히 짚는다. 그리고 pooling indices라는 매우 간단한 정보를 활용해 이 문제를 영리하게 완화한다. 구조도 비교적 단순하고, fully connected layer를 제거해 end-to-end SGD 학습이 가능하도록 만든 점도 강점이다.

또 다른 강점은 **분석의 질**이다. 논문은 SegNet을 제안한 뒤 곧바로 여러 decoder variant를 비교하면서 어떤 요소가 실제로 중요한지를 보여 준다. 단순히 최고 성능 수치만 제시하는 논문과 달리, 이 논문은 “encoder feature map을 전부 저장하면 가장 좋지만 메모리 비용이 크다”, “압축된 정보만 써도 적절한 decoder가 있으면 꽤 잘 된다”, “큰 decoder는 성능을 올리지만 느리다” 같은 설계 원칙을 도출한다. 이는 후속 연구자에게도 가치가 크다.

실험 설계에서도 장점이 있다. 저자들은 다양한 경쟁 모델을 같은 optimizer, 비슷한 학습 조건 아래에서 비교하려고 시도한다. 물론 완전히 동일한 최적 조건을 보장할 수는 없지만, 적어도 training recipe 차이 때문에 성능 차이가 과장되는 것을 줄이려는 의도가 분명하다. 또한 BF metric을 함께 사용해 region metric만으로는 놓치기 쉬운 contour quality를 측정한 점도 설득력이 있다.

한계도 분명하다. 첫째, SUN RGB-D 같은 복잡한 indoor scene에서는 성능이 여전히 낮다. 즉, SegNet의 효율성은 분명하지만, 매우 어려운 장면 이해 문제를 근본적으로 해결했다고 보기는 어렵다. 둘째, 논문 자체도 인정하듯 controlled benchmark라 해도 optimizer와 모델의 효과를 완전히 분리할 수는 없다. 따라서 어떤 구조가 더 좋다는 결론은 특정 학습 설정의 영향도 일부 포함할 수 있다. 셋째, CRF나 region proposal 같은 외부 모듈을 사용하지 않는 순수 feed-forward 구조의 장점을 강조하지만, 반대로 말하면 최고 정확도 자체는 후처리나 추가 모듈을 붙인 방법에 의해 더 올라갈 여지도 남는다.

또한 논문은 depth modality를 사용하지 않은 SUN RGB-D 결과를 제시한다. 이는 RGB-only 성능을 보는 데는 의미가 있지만, RGB-D segmentation이라는 문제 전체 관점에서는 한계이기도 하다. 저자들도 이 점을 인정하며, depth를 쓰려면 별도의 architectural redesign이 필요하다고 말한다. 따라서 이 논문만으로 RGB-D 장면 이해 전반의 결론을 내리기는 어렵다.

비판적으로 보면, SegNet의 핵심 아이디어는 매우 우아하지만, “pooling index만으로 충분한가?”라는 질문은 남는다. 실제로 논문 결과도 full encoder feature를 저장하는 방식이 가장 높은 성능을 보인다고 인정한다. 즉, SegNet은 최상의 정확도를 위한 구조라기보다, 정확도와 자원 사용량 사이의 좋은 타협점에 가깝다. 이 점은 약점이라기보다 정확한 위치 규정에 가깝다.

## 6. 결론

이 논문은 semantic segmentation을 위해 **encoder-decoder 구조**를 명확히 정립하고, decoder에서 **pooling indices 기반 upsampling**을 사용하는 SegNet을 제안한다. 이를 통해 full encoder feature map을 저장하지 않고도 경계 정보를 어느 정도 보존하면서, 메모리 효율적인 추론을 가능하게 한다. 또한 VGG16의 fully connected layer를 제거함으로써 모델을 작게 만들고 end-to-end 학습 가능성을 높였다.

논문의 주요 기여는 단순한 아키텍처 제안에 그치지 않는다. decoder 설계에 따른 memory, speed, accuracy trade-off를 체계적으로 비교했고, road scene와 indoor scene이라는 실제적으로 중요한 문제에서 SegNet의 유효성을 검증했다. 특히 road scene segmentation에서는 강한 성능과 실용적인 효율을 동시에 보여 주었다.

향후 연구 측면에서도 이 논문은 의미가 크다. 이후 등장한 많은 encoder-decoder segmentation 모델들은 skip connection, unpooling, learned upsampling, boundary-aware decoding 같은 주제를 계속 발전시켰는데, SegNet은 그 흐름에서 중요한 기준점 역할을 한다. 실제 적용 측면에서는 자율주행, 로보틱스, AR처럼 메모리와 속도가 중요한 환경에서 여전히 참고할 가치가 있다. 요약하면, SegNet은 “최고 성능만 추구한 모델”이라기보다, **실용적이고 해석 가능한 segmentation architecture**를 제시한 영향력 있는 논문이다.
