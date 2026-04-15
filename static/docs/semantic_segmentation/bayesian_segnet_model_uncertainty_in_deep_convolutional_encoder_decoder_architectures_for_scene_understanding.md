# Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding

- **저자**: Alex Kendall, Vijay Badrinarayanan, Roberto Cipolla
- **발표연도**: 2016
- **arXiv**: https://arxiv.org/abs/1511.02680

## 1. 논문 개요

이 논문은 semantic segmentation, 즉 이미지의 모든 픽셀에 대해 클래스 라벨을 예측하는 문제를 다룬다. 저자들은 단순히 픽셀별 예측 결과만 내는 것이 아니라, 각 예측이 얼마나 불확실한지까지 함께 출력할 수 있는 확률적 segmentation 모델인 **Bayesian SegNet**을 제안한다. 핵심 목표는 encoder-decoder 기반의 segmentation network를 Bayesian neural network처럼 동작하게 만들어, 픽셀별 semantic label과 함께 model uncertainty를 추정하는 것이다.

이 연구 문제는 실제 응용에서 매우 중요하다. 예를 들어 자율주행 시스템이 어떤 물체를 pedestrian으로 분류했더라도, 그 예측이 cyclist나 sign과 헷갈릴 가능성이 높은지 알 수 있어야 안전한 의사결정을 할 수 있다. 논문은 이러한 uncertainty가 active learning, semi-supervised learning, label propagation 같은 다른 응용에도 직접적으로 유용하다고 설명한다. 기존의 deep semantic segmentation 방법들은 성능은 높았지만, 예측의 신뢰도를 정량적으로 제공하지 못했다는 점이 이 논문의 출발점이다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 실용적이다. 학습 때 regularization 용도로 쓰이던 **dropout**을 test time에도 유지하고, 같은 입력 이미지에 대해 여러 번 stochastic forward pass를 수행하여 **Monte Carlo dropout sampling**으로 posterior predictive distribution을 근사한다. 이렇게 얻은 여러 softmax 출력 샘플의 평균을 segmentation prediction으로 사용하고, 분산을 model uncertainty로 해석한다.

이 접근의 중요한 차별점은, 기존 segmentation architecture를 완전히 새로 설계하지 않고도 확률적 추론을 가능하게 했다는 점이다. 저자들은 SegNet뿐 아니라 FCN, Dilation Network에도 같은 방식이 적용되어 성능 향상이 나타난다고 보고한다. 또한 softmax 출력값 자체를 uncertainty로 해석하는 것은 부적절하다고 분명히 구분한다. softmax는 클래스 간 상대적 확률만 보여줄 뿐이고, 모델 자체가 얼마나 확신하는지에 대한 절대적 uncertainty는 제공하지 못한다는 것이 논문의 주장이다.

## 3. 상세 방법 설명

기본 구조는 SegNet이다. SegNet은 convolutional encoder-decoder architecture로, encoder는 convolution, batch normalization, ReLU, max-pooling으로 구성되고, decoder는 encoder에서 저장한 max-pooling index를 이용해 upsampling을 수행한다. 이 구조는 segmentation 경계 정보를 잘 보존하면서도 파라미터 수를 줄이는 장점이 있다. 원 논문에서는 큰 모델인 SegNet과 더 작은 분석용 모델인 SegNet-Basic을 모두 사용한다.

Bayesian SegNet은 이 구조에 dropout을 삽입하여 Bayesian approximation을 수행한다. 논문이 관심을 두는 것은 학습 데이터 $X$와 라벨 $Y$가 주어졌을 때 convolutional weight $W$의 posterior distribution이다.

$$
p(W \mid X, Y)
$$

하지만 이 posterior는 직접 계산하기 어렵기 때문에 variational inference를 사용한다. 즉, 실제 posterior 대신 근사 분포 $q(W)$를 두고, 이 분포가 실제 posterior와 가깝도록 KL divergence를 최소화한다.

$$
KL(q(W) \,\|\, p(W \mid X, Y))
$$

각 convolutional layer $i$의 weight는 다음과 같이 Bernoulli random variable을 이용해 표현된다.

$$
b_{i,j} \sim \text{Bernoulli}(p_i) \quad \text{for } j = 1, \dots, K_i
$$

$$
W_i = M_i \operatorname{diag}(b_i)
$$

여기서 $b_i$는 dropout mask 역할을 하는 Bernoulli 변수 벡터이고, $M_i$는 variational parameter이다. 저자들은 dropout probability를 학습하지 않고, 일반적으로 널리 쓰이는 고정값 $p_i = 0.5$를 사용했다.

학습 측면에서는 cross entropy loss를 최소화하는 것이 variational inference의 관점에서 KL divergence 최소화와 연결된다고 설명한다. 따라서 학습 자체는 기존의 stochastic gradient descent 기반 end-to-end training과 유사하게 수행된다. 실제 구현은 Caffe로 되어 있으며, learning rate는 0.001, weight decay는 0.0005를 사용했고, training loss가 더 이상 줄지 않을 때까지 학습했다. 또한 모든 convolution 뒤에 batch normalization을 사용했고, test time의 batch norm 통계는 training set에서 계산한 값을 사용했다.

추론 과정이 이 논문의 핵심이다. test time에도 dropout을 켠 상태로 동일한 입력에 대해 여러 번 forward pass를 수행한다. 이렇게 얻은 softmax sample들의 평균을 최종 segmentation prediction으로 사용한다. 즉, 픽셀별 클래스 확률의 평균이 최종 예측에 대응한다. uncertainty는 각 클래스에 대한 softmax sample variance로 계산하고, 전체 uncertainty map은 클래스별 variance를 평균내어 만든다. 저자들은 variation ratio도 실험했지만, variance 기반 uncertainty가 덜 이진적이고 더 유용했다고 보고한다.

논문은 dropout을 어디에 넣어야 가장 좋은지도 체계적으로 비교한다. 모든 encoder/decoder 층에 dropout을 넣는 fully Bayesian 형태는 regularization이 너무 강해서 학습 적합도가 낮아졌고, 성능도 나빠졌다. 가장 좋은 결과는 네트워크의 깊은 절반, 즉 **central encoder-decoder 부분에만 dropout을 넣는 방식**이었다. 작은 SegNet-Basic에서는 central four encoder-decoder units, 전체 26-layer SegNet에서는 central six encoders and decoders에 dropout을 적용했다. 저자들은 낮은 층은 edge나 corner 같은 기초 특징을 추출하므로 deterministic weight로도 충분하고, 더 깊은 층에서 형성되는 shape와 contextual relationship 같은 high-level feature가 Bayesian weight의 이점을 더 잘 받는다고 해석한다.

또 하나 중요한 분석은 **weight averaging**과 **Monte Carlo sampling**의 비교이다. 일반 dropout의 표준 test 방식은 dropout을 끄고 weight를 스케일하는 weight averaging인데, 이 논문은 segmentation에서는 Monte Carlo sampling이 더 낫다고 보인다. CamVid에서 약 6개 이상의 sample부터 Monte Carlo 방식이 weight averaging보다 더 높은 global accuracy를 보였고, 약 40 samples 정도 이후에는 성능 향상이 거의 포화되었다. 즉, sampling은 inference cost를 더 요구하지만 더 좋은 성능과 uncertainty estimation을 동시에 제공한다.

## 4. 실험 및 결과

논문은 CamVid, SUN RGB-D, Pascal VOC 2012에서 Bayesian SegNet을 평가했다. CamVid는 367개 training image와 233개 test image로 이루어진 도로 장면 데이터셋이며, 11개 클래스를 분할한다. 입력 이미지는 $360 \times 480$으로 resize했다. SUN RGB-D는 5285개 training, 5050개 test image로 구성된 실내 장면 데이터셋이며 37개 클래스를 segmentation하는 매우 어려운 benchmark다. 여기서는 입력을 $224 \times 224$로 resize했고 RGB만 사용했다. Pascal VOC 2012는 20개 salient object class와 background를 포함하는 object segmentation benchmark이며, 12031개 training image와 1456개 test image를 사용했고 역시 $224 \times 224$로 resize했다.

성능 지표로는 global accuracy, class average accuracy, mean intersection over union이 사용되었다. CamVid 결과에서 Bayesian SegNet은 매우 강한 성능 향상을 보였다. 기존 SegNet의 수치는 global accuracy 88.6, class average 65.9, mean IoU 50.2였고, Bayesian SegNet은 global accuracy 86.9로 다소 낮지만 class average 76.3, mean IoU 63.1로 크게 향상되었다. 특히 sign-symbol, pedestrian, bicyclist 같은 작고 어려운 클래스에서 개선 폭이 크다고 논문은 강조한다. SegNet-Basic에서도 Bayesian 버전은 mean IoU가 46.3에서 55.8로 상승했다.

SUN RGB-D에서도 Bayesian SegNet은 RGB만 사용했음에도 강한 성능을 냈다. SegNet은 global accuracy 70.3, class average 35.6, mean IoU 22.1이었고, Bayesian SegNet은 각각 71.2, 45.9, 30.7을 기록했다. 논문은 depth modality를 사용한 일부 기존 방법보다도 outperform한다고 서술한다. NYUv2 subset에서도 Bayesian SegNet은 RGB 기반 방법 중 최고 성능으로 보고되었다. Pascal VOC 2012 test set에서는 SegNet이 IoU 59.1, Bayesian SegNet이 60.5를 기록했고, FCN-8은 62.2에서 65.4, Dilation Network는 71.3에서 73.1로 올라갔다. 이 결과는 Bayesian 접근이 SegNet에만 특화된 것이 아니라 여러 architecture에 일반적으로 적용 가능함을 보여준다.

논문은 uncertainty 자체의 질도 실험적으로 검증한다. 정성적으로는 uncertainty가 주로 object boundary, 가려진 물체, 먼 거리의 물체, 서로 시각적으로 비슷한 클래스 사이에서 높게 나타났다. 예를 들어 CamVid에서는 cyclist와 pedestrian, SUN에서는 chair와 sofa 또는 bench와 table, Pascal에서는 cat과 dog, train과 bus 같은 경우가 언급된다. 정량적으로는 class accuracy와 mean uncertainty 사이에 강한 역상관이 있었고, class frequency와 uncertainty 사이에도 역상관이 있었다. 즉, 자주 등장하고 쉬운 클래스일수록 모델이 더 confident했고, 희귀하고 어려운 클래스일수록 uncertainty가 높았다.

또한 confidence percentile별 pixel classification accuracy 분석도 제시된다. 예를 들어 CamVid에서 가장 confident한 상위 10% 픽셀은 99.7% accuracy를 보였고, SUN RGB-D에서도 97.6%였다. 전체 픽셀을 다 포함한 경우에는 CamVid 86.7%, SUN RGB-D 75.4%로 내려간다. 이는 uncertainty estimate가 실제 예측 정확도와 잘 대응함을 보여주는 근거로 제시된다.

실시간성에 대해서도 언급이 있다. SegNet은 Titan X GPU에서 frame당 35ms, Bayesian SegNet은 10 Monte Carlo samples 사용 시 frame당 90ms로 동작한다고 보고한다. sampling 때문에 느려지긴 하지만, practical application에서 병렬화로 부담을 줄일 수 있다고 설명한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 uncertainty estimation을 semantic segmentation에 실용적으로 도입했다는 점이다. 복잡한 Bayesian deep learning 기법을 새로 설계한 것이 아니라, 기존 dropout을 활용해 test time posterior sampling을 가능하게 만들었고, 추가 파라미터 없이 성능 향상과 uncertainty estimation을 동시에 얻었다. 또한 SegNet뿐 아니라 FCN, Dilation Network에도 일관된 2-3% 개선을 보여 범용성도 설득력 있게 제시했다. 특히 CamVid 같은 작은 데이터셋에서 효과가 더 크다는 관찰도 의미가 있다. 데이터가 적을수록 모델 불확실성을 다루는 것이 유리하다는 점을 실험으로 보여주기 때문이다.

또 다른 강점은 uncertainty를 단순 시각화에 그치지 않고, accuracy와 frequency와의 관계, confidence percentile별 정확도 등으로 정량적으로 검증했다는 점이다. 이 때문에 uncertainty map이 단순히 보기 좋은 보조 출력이 아니라 실제 의사결정에 쓸 수 있는 신호라는 주장이 어느 정도 뒷받침된다.

한계도 분명하다. 첫째, Bayesian approximation 자체가 dropout 기반의 근사이므로 진정한 posterior inference는 아니다. 논문도 이 점을 직접 길게 비판하지는 않지만, 방법의 본질상 approximation quality에는 한계가 있다. 둘째, Monte Carlo sampling은 inference time cost를 추가로 요구한다. 논문은 병렬 처리로 완화할 수 있다고 하지만, sample 수가 늘어날수록 계산량이 증가하는 점은 실제 배포에서 부담이 될 수 있다. 셋째, uncertainty가 주로 model uncertainty를 반영하는지, 아니면 데이터 noise나 annotation ambiguity까지 얼마나 분리해 설명하는지는 이 논문 범위에서 명확히 다뤄지지 않았다. 넷째, depth modality나 CRF, proposal, multi-stage training 같은 보조 기법은 일부러 배제했기 때문에, 절대 최고 성능을 목표로 한 시스템과 직접 비교할 때는 공정성보다는 “core segmentation engine” 비교에 가깝다. 이는 장점이기도 하지만, 실제 응용의 최종 시스템과는 차이가 있다.

비판적으로 보면, 논문의 핵심 메시지는 “uncertainty를 제공하면서 accuracy도 오르는 실용적 방법”인데, 그 원인이 Bayesian modelling 자체인지, 혹은 test-time ensemble 효과에 가까운 것인지에 대한 더 깊은 분해는 충분히 제시되지 않았다. 다만 논문이 의도한 범위에서는, 그 효과가 실제 segmentation benchmark에서 반복적으로 관찰된다는 점이 더 중요하게 다뤄진다.

## 6. 결론

이 논문은 deep convolutional encoder-decoder segmentation architecture를 Bayesian neural network처럼 사용하여, 픽셀별 semantic segmentation과 model uncertainty를 함께 출력하는 **Bayesian SegNet**을 제안했다. 방법의 핵심은 test time dropout을 이용한 Monte Carlo sampling이며, softmax sample 평균으로 예측을 만들고 sample variance로 uncertainty를 계산한다. 이 방식은 추가 파라미터 없이 구현 가능하고, SegNet뿐 아니라 FCN과 Dilation Network에도 적용되어 성능 향상을 보였다.

실험적으로 Bayesian SegNet은 CamVid와 SUN RGB-D에서 매우 강한 결과를 보였고, uncertainty가 object boundary, 시각적으로 어려운 물체, 희귀 클래스에서 높아지는 경향도 확인되었다. 따라서 이 연구는 semantic segmentation의 정확도 개선뿐 아니라, 실제 의사결정 시스템에서 필요한 신뢰도 추정의 관점에서도 중요한 의미를 가진다. 향후 연구에서는 논문이 언급하듯 video 정보 활용, 더 정교한 uncertainty modelling, 실제 시스템에서의 안전한 decision-making 연계로 확장될 가능성이 크다.
