# Learning Deconvolution Network for Semantic Segmentation

- **저자**: Hyeonwoo Noh, Seunghoon Hong, Bohyung Han
- **발표연도**: 2015
- **arXiv**: https://arxiv.org/abs/1505.04366

## 1. 논문 개요

이 논문은 semantic segmentation을 위해 **deconvolution network를 직접 학습**하는 방법을 제안한다. 당시 대표적 접근이던 Fully Convolutional Network (FCN) 계열은 입력 전체 이미지를 한 번에 처리한다는 장점이 있었지만, receptive field 크기가 사실상 고정되어 있어 큰 물체와 작은 물체를 동시에 잘 다루기 어렵고, coarse한 feature map을 bilinear interpolation으로 단순히 키우는 방식 때문에 경계와 세부 구조가 쉽게 뭉개진다는 문제가 있었다.

논문은 이 문제를 두 축에서 해결하려고 한다. 첫째, convolutional feature를 다시 촘촘한 pixel-level prediction으로 복원하는 **깊은 deconvolution network**를 둔다. 둘째, 이미지 전체를 한 번에 segmentation하지 않고, **object proposal 단위로 instance-wise segmentation**을 수행한 뒤 이를 합쳐 최종 semantic map을 만든다. 이 설계는 물체 scale 변화에 자연스럽게 대응하고, 세밀한 shape 복원을 가능하게 하려는 목적을 가진다.

문제의 중요성은 분명하다. semantic segmentation은 단순 분류와 달리 모든 픽셀에 레이블을 붙여야 하므로, “무엇이 있는가”뿐 아니라 “정확히 어디까지가 그 물체인가”가 중요하다. 따라서 coarse한 분류 특징만으로는 충분하지 않으며, 위치 정보와 shape 정보를 얼마나 잘 복원하느냐가 성능의 핵심이 된다. 이 논문은 바로 그 지점을 deconvolution과 unpooling을 통해 정면으로 다룬다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **convolution network가 압축한 표현을 deconvolution network가 coarse-to-fine 방식으로 다시 복원하여 segmentation mask를 만든다**는 것이다. 저자들은 convolution network를 단순한 feature extractor로 보고, 그 뒤에 붙는 deconvolution network를 object shape generator처럼 해석한다. 즉, 앞단은 “무엇인지”를 인코딩하고, 뒷단은 “어떤 형태로 어디에 있는지”를 복원한다.

기존 FCN과의 가장 큰 차이는 deconvolution을 다루는 방식에 있다. FCN에서는 coarse score map을 upsampling하는 과정이 비교적 단순하며, 논문 저자 표현대로 “real deconvolution”을 충분히 하지 못한다. 반면 이 논문은 여러 단계의 **unpooling + deconvolution + ReLU**를 쌓아, 낮은 해상도의 추상 특징에서 높은 해상도의 세밀한 구조를 점진적으로 드러낸다.

또 하나의 차별점은 예측 단위다. FCN은 image-level prediction 중심이고, 이 논문은 **proposal-wise prediction** 중심이다. 각 proposal은 잠재적 object instance를 포함하는 sub-image이며, 네트워크는 여기에 대해 dense segmentation을 수행한다. 이후 여러 proposal의 결과를 원래 이미지 좌표계로 되돌려 결합한다. 저자들은 이 방식이 고정 receptive field의 한계를 줄이고, 다양한 크기의 물체를 다루는 데 유리하다고 주장한다.

마지막으로, 저자들은 자신들의 방법과 FCN이 **상보적(complementary)** 이라고 본다. DeconvNet은 fine detail과 multi-scale object handling에 강하고, FCN은 전체적 형태와 문맥을 잡는 데 강하므로, 두 방법을 ensemble하면 더 좋아진다는 것이 핵심 관찰이다.

## 3. 상세 방법 설명

전체 시스템은 크게 두 부분으로 구성된다. 앞부분은 VGG-16 기반의 convolution network이고, 뒷부분은 이를 거의 대칭적으로 뒤집은 형태의 deconvolution network이다. 입력은 $224 \times 224 \times 3$ 이미지이고, 출력은 같은 해상도의 $224 \times 224 \times 21$ class score map이다. 여기서 21은 PASCAL VOC 2012의 background 포함 class 수다.

convolution 부분은 VGG-16의 마지막 classification layer를 제거한 구조를 사용한다. 총 13개의 convolution layer와 pooling, 그리고 끝부분의 두 fully connected layer인 `fc6`, `fc7`이 포함된다. 공간 해상도는 pooling을 거치면서 점점 줄어들고, 최종적으로 `fc7`에서는 $1 \times 1 \times 4096$ 표현을 얻는다. 이 부분은 물체의 class-specific feature representation을 추출하는 역할을 한다.

그 다음 deconvolution network는 `fc7` 이후부터 시작된다. `deconv-fc6`를 거쳐 다시 $7 \times 7 \times 512$로 되돌린 뒤, `unpool5`, `deconv5-*`, `unpool4`, `deconv4-*` 식으로 점차 해상도를 높여 간다. 최종적으로 `deconv1-*`를 지나 $224 \times 224$ 해상도의 dense prediction map을 만들고, 마지막 `output` layer가 각 픽셀의 class score를 낸다. 이 구조는 convolution에서 잃어버린 spatial detail을 복원하도록 설계되었다.

이 방법의 핵심 연산은 **unpooling**과 **deconvolution**이다.

Unpooling은 max-pooling의 역연산에 해당한다. 일반적인 pooling은 receptive field 안에서 가장 강한 activation 하나만 남기므로 정확한 위치 정보가 사라진다. 그런데 segmentation에서는 이런 위치 정보가 중요하다. 저자들은 pooling 시 선택된 최대값의 위치를 switch variable로 저장해 두고, unpooling 단계에서 그 activation을 원래 위치에 다시 배치한다. 이렇게 하면 activation map 크기는 커지지만 값은 듬성듬성한 sparse map이 된다. 중요한 점은 이것이 **example-specific structure**, 즉 해당 입력 예제의 구체적 형태를 복원하는 데 유리하다는 것이다.

Deconvolution은 이 sparse map을 다시 **dense activation map**으로 바꾸는 역할을 한다. 논문은 이를 convolution과 반대 방향의 연결로 설명한다. convolution이 여러 입력 activation을 모아 하나의 출력을 만든다면, deconvolution은 하나의 입력 activation이 여러 출력에 영향을 주는 식으로 동작한다. 그리고 이 연산에 사용되는 filter는 학습된다. 저자들은 이 filter가 object shape를 복원하는 basis처럼 작동한다고 해석한다. 낮은 deconvolution layer는 대략적인 object shape와 위치를, 높은 deconvolution layer는 더 class-specific한 fine detail을 복원하는 경향이 있다고 설명한다.

논문은 이 과정을 시각화하여, 네트워크를 위로 진행할수록 거친 object outline에서 시작해 더 정교한 경계와 구조가 드러난다고 주장한다. 또한 background 쪽 noisy activation은 점차 억제되고, target class와 관련된 activation은 강화된다고 설명한다. 요약하면, **unpooling은 입력 인스턴스에 특화된 구조 복원**, **deconvolution은 클래스 특화 shape 복원**을 담당하며, 둘의 결합이 정밀 segmentation의 핵심이라는 것이다.

학습 측면에서, 이 네트워크는 매우 깊고 파라미터 수도 약 252M으로 크기 때문에 최적화가 어렵다. 저자들은 이를 해결하기 위해 **batch normalization**과 **two-stage training**을 도입한다. Batch normalization은 각 convolution 및 deconvolution layer 출력 뒤에 붙으며, internal covariate shift를 줄여 매우 깊은 네트워크 학습을 돕는다. 논문은 BN이 없으면 poor local optimum으로 빠진다고 명시한다.

Two-stage training은 학습 난이도를 점진적으로 올리는 전략이다. 1단계에서는 ground-truth annotation으로부터 object를 중심에 놓은 crop을 만들어 학습한다. 즉, object의 위치와 scale variation을 제한한 쉬운 예제로 먼저 학습한다. 이때 중심 object만 정답 class로 두고 나머지 픽셀은 background로 둔다. 2단계에서는 object proposal을 사용해 더 어려운 예제로 fine-tuning한다. ground-truth segmentation과 충분히 겹치는 proposal을 선택해 학습하며, 이 단계에서는 proposal misalignment나 scale variation을 더 많이 포함하므로 test-time 상황에 더 가깝다. 이 방식은 search space를 줄여 초기 학습을 안정화하고, 이후 실제 환경에 대한 robustness를 키우는 역할을 한다.

추론 과정은 proposal 기반이다. 테스트 이미지에서 Edge Boxes로 약 2000개의 object proposal을 생성하고, objectness score 상위 50개만 사용한다. 각 proposal마다 네트워크를 적용하여 class score map $g_i \in \mathbb{R}^{W \times H \times C}$를 얻고, 이를 원본 이미지 공간으로 되돌린 map을 $G_i$라고 둔다. 그런 다음 전체 이미지의 픽셀별 class score map $P(x,y,c)$를 proposal 결과들로부터 집계한다. 논문은 두 가지 집계식을 제시한다.

픽셀별 최대값 집계는 다음과 같다.

$$
P(x, y, c) = \max_i G_i(x, y, c)
$$

픽셀별 합 집계는 다음과 같다.

$$
P(x, y, c) = \sum_i G_i(x, y, c)
$$

실험에서는 최대값 집계를 사용했다고 적혀 있다. 이후 softmax를 적용해 class conditional probability map을 만들고, 마지막으로 fully-connected CRF를 후처리로 적용한다. 이때 CRF의 unary potential은 이 probability map에서 온다.

FCN과의 ensemble은 더 단순하다. DeconvNet과 FCN이 각각 만든 class conditional probability map의 평균을 구한 뒤, 다시 CRF를 적용해 최종 결과를 얻는다. 즉, 구조적으로 복잡한 joint model을 만드는 것이 아니라, 두 방법의 출력을 score level에서 평균하는 매우 단순한 ensemble이다.

## 4. 실험 및 결과

실험은 PASCAL VOC 2012 segmentation benchmark에서 수행되었다. 학습에는 Hariharan 등 [8]의 augmented annotation을 사용했고, train과 validation 이미지를 모두 학습에 사용했다. 테스트는 VOC 2012 test set에서 평가했다. 논문은 외부 데이터를 사용하지 않았음을 강조한다. 이는 일부 비교 방법들이 추가 데이터로 성능을 높였기 때문이다.

평가 지표는 `comp6` protocol의 Intersection over Union (IoU)이다. 클래스는 총 20개 object class와 background다. 구현은 Caffe 기반이며, optimizer는 SGD with momentum이다. 초기 learning rate는 0.01, momentum은 0.9, weight decay는 0.0005이다. convolution 부분은 ImageNet으로 pretrain된 VGG-16 가중치로 초기화하고, deconvolution 부분은 zero-mean Gaussian으로 초기화한다. Dropout은 batch normalization 때문에 제거했다. 1단계 학습은 약 20K iteration, 2단계는 약 40K iteration에서 수렴했고, batch size는 64였다. 단일 GTX Titan X 12GB GPU에서 총 6일이 걸렸다고 보고한다.

학습 데이터 규모도 제시되어 있다. 1단계는 약 0.2M examples, 2단계는 약 2.7M examples이다. 입력은 먼저 $250 \times 250$로 맞춘 후 무작위로 $224 \times 224$ crop을 하고, horizontal flip을 적용한다. 클래스 불균형 완화를 위해 적은 클래스에는 redundant example을 추가했다고 한다.

주요 정량 결과를 보면, 제안 방법 `DeconvNet`은 test set에서 mean IoU 69.6%를 기록했다. 여기에 CRF를 붙인 `DeconvNet+CRF`는 70.5%가 되었다. FCN-8s는 62.2%였으므로, 단독 비교에서도 제안 방법이 더 높다. 이후 FCN-8s와 ensemble한 `EDeconvNet`은 71.7%, 여기에 CRF를 더한 `EDeconvNet+CRF`는 72.5%를 달성했다. 논문은 이것이 **외부 데이터 없이 학습한 방법 중 최고 성능**이라고 주장한다.

결과를 조금 더 해석하면 다음과 같다. 첫째, 제안 방법 단독으로도 FCN-8s보다 약 7.4%p 높은 mean IoU를 보인다. 이는 coarse upsampling보다 deep deconvolution이 실제 segmentation 품질에 유의미하다는 논문의 핵심 주장을 뒷받침한다. 둘째, CRF는 약 1%p 내외의 개선을 주지만, 절대적인 게임 체인저는 아니다. 셋째, 가장 인상적인 부분은 ensemble이다. FCN-8s 자체는 DeconvNet보다 낮은 성능이지만, 둘을 합치면 DeconvNet 단독보다 3.1%p, FCN-8s보다 10.3%p 개선된다. 이는 두 방법이 정말로 다른 종류의 오류를 낸다는 논문의 해석과 맞아떨어진다.

클래스별 수치를 보면, 제안 방법은 `aeroplane`, `bus`, `table`, `train`, `sofa` 등 여러 클래스에서 경쟁력 있는 결과를 보인다. 반면 `bike`, `chair`, `dog` 등 일부 세부 구조가 복잡하거나 proposal 품질에 민감할 수 있는 클래스에서는 절대 최고는 아니다. 다만 논문의 핵심 메시지는 개별 클래스 최고치보다도, fine detail 복원과 multi-scale handling의 전체적 효과에 있다.

정성적 결과 설명도 중요하다. 저자들은 DeconvNet이 FCN보다 세밀한 segmentation을 만들고, 작은 물체나 큰 물체를 더 잘 처리한다고 주장한다. 반대로 proposal이 잘못 정렬되거나 background clutter가 심한 경우에는 noisy prediction이 생기기도 한다. 이 점은 instance-wise approach의 약점이기도 하다. Figure 7 설명에 따르면, 어떤 경우에는 FCN이 더 나은 결과를 내고, 어떤 경우에는 DeconvNet이 더 낫다. 그리고 ensemble은 양쪽의 장점을 모아 오류를 줄인다.

Figure 6의 실험은 proposal aggregation의 의미를 보여준다. 큰 proposal부터 순차적으로 합치면 초기에는 전체적인 object structure가 먼저 드러나고, 더 작은 proposal들이 추가될수록 finer detail이 채워진다. 이는 instance-wise, multi-scale prediction이 실제로 coarse-to-fine refinement처럼 작동한다는 간접 증거로 볼 수 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 semantic segmentation에서 **deconvolution network를 학습 가능한 핵심 모듈로 전면에 내세웠다**는 점이다. 단순 upsampling이 아니라, unpooling과 learned deconvolution을 결합해 위치 정보와 shape 정보를 복원하려는 관점이 명확하다. 이는 이후 encoder-decoder 계열 segmentation 모델들을 이해하는 데도 중요한 역사적 위치를 가진다.

또 다른 강점은 **instance-wise prediction의 실용적 장점**을 설득력 있게 제시했다는 것이다. FCN이 고정 receptive field 때문에 큰 물체를 조각내거나 작은 물체를 배경으로 놓치는 문제를 예시와 함께 지적하고, proposal 기반 접근이 이를 완화한다고 설명한다. 실제 정량 결과도 이 주장을 상당 부분 뒷받침한다.

학습 전략도 강점이다. 네트워크가 크고 데이터가 제한적이라는 현실적 문제를 그냥 두지 않고, batch normalization과 two-stage training으로 해결했다. 특히 쉬운 예제에서 시작해 어려운 proposal 예제로 fine-tuning하는 방식은 단순하지만 매우 실용적이다. 논문이 단지 구조 제안에 그치지 않고, “어떻게 실제로 학습시킬 것인가”까지 구체적으로 다룬다는 점이 좋다.

반면 한계도 분명하다. 첫째, 이 방법은 **object proposal 품질에 의존**한다. 논문도 misaligned proposal이나 background proposal 때문에 noisy prediction이 생긴다고 인정한다. 즉, segmentation 성능이 backbone과 decoder뿐 아니라 proposal stage에도 묶여 있다. 이는 end-to-end simplicity나 speed 측면에서 불리할 수 있다.

둘째, 추론 비용이 크다. 테스트 시 이미지당 약 2000 proposal을 만들고 상위 50개를 사용한다. 각 proposal마다 네트워크를 돌린 뒤 다시 aggregate해야 하므로, 이미지 한 번 forward로 끝나는 FCN류보다 계산량이 무거울 가능성이 크다. 논문 발췌문에는 정확한 test-time latency는 명시되지 않았으므로, 얼마나 느린지는 여기서 단정할 수 없다. 그러나 구조상 더 비쌀 가능성은 매우 높다.

셋째, 모델 규모가 상당하다. 파라미터 수가 약 252M이고, 학습에도 단일 GPU 기준 6일이 걸린다. 이는 당대 기준에서도 적지 않은 부담이다. 실제 deployment를 염두에 두면 무겁고, proposal 기반 추론까지 더해지면 시스템 복잡도가 높다.

넷째, 성능 향상의 중요한 일부가 **FCN과의 ensemble**에서 나온다. DeconvNet 단독도 강하지만 최종 최고 성능 72.5%는 FCN-8s와 결합한 결과다. 따라서 논문의 최종 최고 성능을 해석할 때는 “제안 디코더 자체의 힘”과 “상보적인 모델 결합 효과”를 구분해서 볼 필요가 있다.

비판적으로 보면, 논문은 deconvolution network의 장점을 잘 보여주지만, 그 효과 중 얼마가 decoder 구조 자체에서 오는지, 얼마가 proposal-based instance decomposition에서 오는지 완전히 분리해 보여주지는 않는다. 또한 aggregate 방식은 max 또는 sum처럼 단순한데, proposal confidence나 overlap quality를 더 정교하게 활용하는 설계가 가능했을 것이다. 다만 이것은 후속 연구 방향에 가깝고, 논문이 제안한 기본 프레임워크 자체는 충분히 설득력 있다.

## 6. 결론

이 논문은 semantic segmentation을 위해 **VGG 기반 convolution encoder와 대칭적 deconvolution decoder를 결합한 깊은 DeconvNet**을 제안했고, 여기에 **instance-wise proposal prediction과 aggregation**을 결합했다. 핵심 기여는 단순 upsampling이 아니라 학습 가능한 unpooling/deconvolution stack으로 세밀한 mask를 복원했다는 점, 그리고 proposal 기반 예측으로 multi-scale object handling 문제를 완화했다는 점이다.

실험적으로도 PASCAL VOC 2012에서 강한 결과를 보였고, 특히 FCN과의 ensemble을 통해 외부 데이터 없이 72.5% mean IoU를 달성했다. 이는 제안 방식이 FCN과 경쟁만 하는 것이 아니라, 서로 다른 오류 특성을 가지는 보완적 방법임을 보여준다.

실제 적용 관점에서는 정밀한 object boundary와 다양한 scale 처리 능력이 중요한 작업에 의미가 크다. 향후 연구 관점에서는 encoder-decoder segmentation architecture, learned upsampling, skip connection, proposal-free dense prediction 같은 흐름으로 이어지는 중요한 연결 고리로 볼 수 있다. 즉, 이 논문은 단지 한 가지 모델 제안에 그치지 않고, 이후 semantic segmentation이 더 정교한 decoder 구조를 적극적으로 채택하게 되는 흐름에 기여한 작업으로 이해할 수 있다.
