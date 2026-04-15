# Learning to Adapt Structured Output Space for Semantic Segmentation

- **저자**: Yi-Hsuan Tsai, Wei-Chih Hung, Samuel Schulter, Kihyuk Sohn, Ming-Hsuan Yang, Manmohan Chandraker
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1802.10349

## 1. 논문 개요

이 논문은 semantic segmentation에서의 unsupervised domain adaptation 문제를 다룬다. 구체적으로는, 라벨이 있는 source domain 데이터로 학습한 segmentation 모델을 라벨이 없는 target domain에서도 잘 동작하게 만드는 것이 목표다. 저자들은 특히 synthetic-to-real 적응, 그리고 도시가 바뀌는 cross-city 적응처럼 입력 영상의 외형은 크게 달라도, segmentation 결과의 공간적 구조는 상당히 비슷하다는 점에 주목한다.

문제의 핵심은 일반적인 supervised semantic segmentation이 pixel-level annotation에 크게 의존한다는 데 있다. 하지만 실제로 새로운 도시, 날씨, 조명, 카메라 환경마다 모든 픽셀을 다시 라벨링하는 것은 비용이 매우 크다. 따라서 source에서 학습한 모델을 target으로 옮기는 domain adaptation이 중요하다. 저자들은 기존 feature-level adaptation이 semantic segmentation에서는 다루어야 할 feature가 매우 고차원이고 복잡해서 충분히 안정적이지 않을 수 있다고 보고, 대신 segmentation output space 자체를 맞추는 방향을 제안한다.

이 문제는 자율주행처럼 scene layout이 중요한 응용에서 특히 중요하다. 도로 장면은 도시나 렌더링 스타일이 달라져도 road는 아래쪽에, sky는 위쪽에, car와 person은 특정 맥락에서 나타나는 경우가 많다. 논문은 바로 이런 structured output의 공통성을 adaptation의 핵심 신호로 삼는다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 semantic segmentation의 출력이 단순한 클래스 확률 맵이 아니라, scene layout과 local context를 담은 structured output이라는 점이다. 입력 이미지의 appearance는 source와 target 사이에서 크게 달라질 수 있지만, 잘 예측된 segmentation map은 두 도메인에서 유사한 구조를 가져야 한다. 예를 들어 road, sidewalk, building, sky의 상대적 위치와 경계 패턴은 도메인이 바뀌어도 어느 정도 유지된다.

기존 접근은 주로 feature space를 adversarial하게 정렬한다. 그러나 segmentation용 feature는 appearance, shape, context를 모두 담아야 하므로 분포가 복잡하고, discriminator가 source와 target을 쉽게 구분해버리면 오히려 학습이 불안정해질 수 있다. 반면 output space는 클래스 수 $C$에 대응하는 비교적 저차원 확률 맵이므로, 도메인 정렬의 대상이 더 직접적이고 의미론적으로 해석 가능하다.

또 하나의 차별점은 multi-level adversarial learning이다. 저자들은 최종 output에만 adaptation을 걸면 너무 상위 레벨에만 신호가 집중되어 하위 feature가 충분히 적응되지 않을 수 있다고 본다. 그래서 중간 feature level에서도 auxiliary segmentation output을 만들고, 여기에 별도의 discriminator를 붙여 여러 단계에서 output-space adaptation을 수행한다.

## 3. 상세 방법 설명

전체 구조는 크게 segmentation network $G$와 discriminator $D_i$로 이루어진다. source 이미지 $I_s$는 정답 라벨이 있으므로 일반적인 segmentation loss로 학습한다. target 이미지 $I_t$는 라벨이 없기 때문에, $G(I_t)$가 만든 softmax prediction을 discriminator에 넣어 source에서 나온 prediction처럼 보이도록 adversarial하게 학습한다. 이 구조의 직관은 “target 이미지에 대한 segmentation 결과가 source 이미지의 segmentation 결과 분포와 구분되지 않게 만들자”는 것이다.

논문 전체 목적 함수는 single-level 기준으로 다음과 같다.

$$
L(I_s, I_t) = L_{\text{seg}}(I_s) + \lambda_{\text{adv}} L_{\text{adv}}(I_t)
$$

여기서 $L_{\text{seg}}$는 source 정답을 이용한 supervised segmentation loss이고, $L_{\text{adv}}$는 target prediction이 source prediction처럼 보이도록 만드는 adversarial loss다. $\lambda_{\text{adv}}$는 두 손실의 균형을 잡는 가중치다.

먼저 segmentation output은 다음처럼 정의된다.

$$
P = G(I) \in \mathbb{R}^{H \times W \times C}
$$

즉, 입력 이미지에 대해 각 픽셀마다 $C$개 클래스 확률을 가지는 softmax map을 만든다.

discriminator는 이 $P$를 입력받아, 각 위치 $(h,w)$에서 source인지 target인지 판단한다. discriminator의 학습 손실은 다음과 같다.

$$
L_d(P) = - \sum_{h,w} (1-z)\log D(P)^{(h,w,0)} + z \log D(P)^{(h,w,1)}
$$

논문 표기대로 $z=0$이면 target, $z=1$이면 source다. 즉 discriminator는 source output과 target output을 구분하도록 학습된다.

반면 segmentation network는 source에서는 일반 cross-entropy loss를 최소화한다.

$$
L_{\text{seg}}(I_s) = - \sum_{h,w} \sum_{c \in C} Y_s^{(h,w,c)} \log P_s^{(h,w,c)}
$$

여기서 $Y_s$는 source ground truth, $P_s = G(I_s)$는 source prediction이다. 이 식은 semantic segmentation에서 가장 표준적인 pixel-wise cross-entropy이다.

target에서는 정답이 없으므로, discriminator를 속이는 방향으로 adversarial loss를 건다.

$$
L_{\text{adv}}(I_t) = - \sum_{h,w} \log D(P_t)^{(h,w,1)}
$$

여기서 $P_t = G(I_t)$이다. 이 식의 의미는 target prediction이 discriminator에게 source처럼 분류되도록 만드는 것이다. 결과적으로 $G$는 target에서도 source와 비슷한 구조의 segmentation output을 내도록 학습된다.

논문의 multi-level 버전에서는 하나의 output만 쓰지 않고, 여러 feature level에서 segmentation output을 예측한 뒤 각 output마다 adversarial loss를 적용한다. 최종 목적 함수는 다음과 같다.

$$
L(I_s, I_t) = \sum_i \lambda^{i}_{\text{seg}} L^i_{\text{seg}}(I_s) + \sum_i \lambda^{i}_{\text{adv}} L^i_{\text{adv}}(I_t)
$$

그리고 전체 최적화는 다음 min-max 형태로 정리된다.

$$
\max_D \min_G L(I_s, I_t)
$$

즉 discriminator는 source와 target output을 잘 구분하도록 최대화하고, segmentation network는 source에서는 정확한 segmentation을 하면서 target output은 source처럼 보이게 만들어 discriminator를 속이도록 최소화된다.

네트워크 구조를 보면, segmentation backbone은 DeepLab-v2 with ResNet-101이다. ImageNet pretrained ResNet-101을 사용하고, 마지막 분류층을 제거한 뒤 conv4와 conv5의 stride를 조정하고 dilated convolution을 적용해 출력 feature map의 해상도를 입력의 $1/8$ 수준으로 유지한다. 최종 분류기로는 ASPP(Atrous Spatial Pyramid Pooling)를 쓴다. 마지막에는 upsampling과 softmax를 통해 입력 크기와 같은 segmentation map을 만든다. 이 baseline만으로도 Cityscapes validation에서 mean IoU 65.1%를 얻었다고 보고한다.

discriminator는 spatial information을 유지하기 위해 fully convolutional 구조로 설계된다. 5개의 convolution layer를 쓰며 kernel size는 $4 \times 4$, stride는 2이고 채널 수는 순서대로 $\{64,128,256,512,1\}$이다. 마지막 층을 제외하고 leaky ReLU를 사용하며, 작은 배치에서 segmentation network와 joint training하기 때문에 batch normalization은 쓰지 않는다.

학습 절차는 one-stage joint training이다. 한 배치마다 먼저 source 이미지를 넣어 segmentation loss를 계산하고 source prediction $P_s$를 얻는다. 다음으로 target 이미지를 넣어 $P_t$를 얻고, $P_s$와 $P_t$를 discriminator에 넣어 discriminator loss를 계산한다. 동시에 $P_t$에 대해 adversarial loss를 계산해 이 gradient를 segmentation network로 역전파한다. multi-level에서는 이 과정을 각 adaptation module마다 반복한다.

최적화는 segmentation network에 대해 SGD with Nesterov를 사용하고, momentum은 0.9, weight decay는 $5 \times 10^{-4}$, 초기 learning rate는 $2.5 \times 10^{-4}$이다. discriminator는 Adam을 사용하며 learning rate는 $10^{-4}$이고, 모멘텀 파라미터는 0.9와 0.99다. learning rate decay는 둘 다 polynomial decay를 사용한다.

부록에서는 vanilla GAN objective 대신 least-squares GAN(LS-GAN) loss도 실험한다. discriminator loss와 adversarial loss는 각각 다음처럼 바뀐다.

$$
L_d^{LS}(P) = \sum_{h,w} z \left(D(P)^{(h,w,1)} - 1\right)^2 + (1-z)\left(D(P)^{(h,w,0)}\right)^2
$$

$$
L_{\text{adv}}^{LS}(I_t) = \sum_{h,w} \left(D(P_t)^{(h,w,1)} - 1\right)^2
$$

저자들은 이 loss가 더 안정적이고 더 높은 mean IoU를 주는 경향이 있음을 추가 실험으로 보인다.

## 4. 실험 및 결과

실험은 크게 세 가지 설정에서 수행된다. 첫째는 GTA5 $\rightarrow$ Cityscapes, 둘째는 SYNTHIA $\rightarrow$ Cityscapes, 셋째는 Cityscapes $\rightarrow$ Cross-City이다. 모든 실험에서 평가지표는 IoU이며, 주로 mean IoU를 보고한다.

GTA5 $\rightarrow$ Cityscapes 설정에서 source는 GTA5 전체 24,966장, target adaptation용은 Cityscapes training set 2,975장, 평가는 Cityscapes validation 500장이다. 먼저 기존 연구들과 공정 비교를 위해 VGG-16 기반 모델에서 single-level output-space adaptation을 평가했다. 그 결과 저자 방법은 mIoU 35.0%를 기록해 FCNs in the Wild의 27.1%, CDA의 28.9%, CyCADA feature의 29.2%, CyCADA pixel의 34.8%보다 높았다. 즉 VGG 기반 비교에서도 output-space adaptation이 경쟁력이 있음을 보였다.

그다음 더 강한 baseline인 ResNet-101 기반으로 ablation을 수행했다. source만으로 학습한 baseline은 36.6%, feature adaptation은 39.3%, 제안한 single-level output-space adaptation은 41.4%, multi-level adaptation은 42.4%였다. 즉 같은 backbone에서 feature adaptation보다 output-space adaptation이 낫고, multi-level이 추가 이득을 준다. 클래스별로는 road, sidewalk, sign, terrain, person, car 등 다양한 항목에서 개선이 나타난다. 다만 pole, traffic sign 같은 작은 물체는 적응이 어렵고 배경에 섞이기 쉽다고 저자들이 직접 언급한다.

또한 oracle과의 성능 차이도 제시한다. GTA5 $\rightarrow$ Cityscapes에서 ResNet-101 oracle은 65.1%인데, 제안 multi-level adaptation은 42.4%로 gap이 -22.7이다. 비교 대상들보다 이 gap이 가장 작다. 이 결과는 “적응 성능 자체”뿐 아니라 “완전 감독 학습에 얼마나 가까워졌는가”라는 측면에서도 의미가 있다.

파라미터 민감도 분석도 중요하다. $\lambda_{\text{adv}}$를 바꾸었을 때 feature adaptation은 0.001에서 39.3%이지만 0.004에서는 32.8%까지 떨어져 민감하다. 반면 output-space adaptation은 0.0005, 0.001, 0.002, 0.004에서 각각 40.2, 41.4, 40.4, 40.1로 비교적 안정적이다. 저자들은 이를 output space가 더 저차원이고 discriminator 입장에서 덜 쉬운 문제라 adversarial training이 덜 불안정하기 때문으로 해석한다.

SYNTHIA $\rightarrow$ Cityscapes에서는 source로 SYNTHIA-RAND-CITYSCAPES 9,400장을 사용하고, Cityscapes validation에서 13개 클래스 기준으로 평가한다. VGG 기반 fair comparison에서 제안 single-level 방법은 37.6% mIoU를 달성하여 FCNs in the Wild 22.9%, CDA 34.8%, Cross-City 35.7%보다 높다. ResNet 기반에서는 baseline 38.6%, feature adaptation 40.8%, single-level 45.9%, multi-level 46.7%로 개선된다. oracle gap 역시 가장 작아졌고, 제안 방법만이 30% 미만 gap에 가까운 수준까지 줄였다고 저자들은 강조한다.

Cross-City 실험에서는 Cityscapes를 source로 두고 Rome, Rio, Tokyo, Taipei를 각각 target으로 삼는다. 각 도시마다 라벨 없는 3,200장으로 adaptation을 수행하고, 라벨 있는 100장으로 평가한다. 도메인 차이가 synthetic-to-real보다 작으므로 adversarial loss weight를 더 작게, 즉 $\lambda^i_{\text{adv}} = 0.0005$로 둔다. 결과는 모든 도시에서 output-space adaptation이 baseline과 feature adaptation보다 일관되게 좋다. 예를 들어 Rome에서는 baseline 50.9%, feature 51.4%, output-space 53.8%, Rio에서는 48.2%, 49.7%, 51.6%, Tokyo에서는 47.7%, 48.0%, 49.9%, Taipei에서는 46.5%, 46.6%, 49.1%다. 절대 개선 폭은 크지 않지만, 여러 도시에 걸쳐 일관된 상승을 보인다는 점이 중요하다.

부록 실험도 흥미롭다. LS-GAN objective를 사용하면 GTA5 $\rightarrow$ Cityscapes에서 vanilla GAN 41.4%보다 높은 44.1%를 얻고, SYNTHIA $\rightarrow$ Cityscapes에서도 45.9%보다 높은 47.6%를 얻는다. 또한 Synscapes $\rightarrow$ Cityscapes에서는 adaptation 없이도 45.3%로 이미 높지만, vanilla GAN 52.7%, LS-GAN 53.1%로 더 오른다. 이는 source가 더 photorealistic할수록 baseline도 강해지지만, output-space adaptation은 여전히 추가 이득을 줄 수 있음을 보여준다.

종합하면, 실험은 세 가지 주장을 뒷받침한다. 첫째, semantic segmentation에서는 feature adaptation보다 output-space adaptation이 더 효과적일 수 있다. 둘째, single-level보다 multi-level adaptation이 더 낫다. 셋째, 제안 방법은 synthetic-to-real뿐 아니라 cross-city 같은 상대적으로 작은 도메인 차이에도 일관되게 유효하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 semantic segmentation의 출력 구조 자체를 adaptation의 핵심 대상으로 삼았다는 점이다. 이는 단순히 “새로운 loss를 추가했다” 수준이 아니라, segmentation 문제의 본질적 특성인 spatial layout과 local context를 잘 짚은 설계다. feature space가 아니라 output space를 정렬한다는 발상은 문제 설정에 더 직접적으로 맞닿아 있고, 실험에서도 실제로 더 높은 성능과 더 안정적인 하이퍼파라미터 거동으로 이어졌다.

또 다른 강점은 방법이 단순하고 end-to-end라는 점이다. target domain의 별도 통계나 사전 지식 없이 discriminator만 붙여 학습할 수 있고, 테스트 시에는 discriminator를 버리고 segmentation network만 사용하므로 추가 추론 비용이 없다. 실용성 측면에서 설계가 깔끔하다.

multi-level adaptation도 설득력이 있다. 최종 출력 하나만 맞추면 중간 feature는 충분히 적응되지 않을 수 있다는 문제의식은 타당하며, auxiliary output을 통한 추가 adversarial supervision이 실제로 성능 향상으로 연결되었다. 특히 강한 baseline 위에서 feature adaptation, single-level, multi-level을 직접 비교한 점이 좋다.

반면 한계도 분명하다. 첫째, output structure가 source와 target에서 유사하다는 가정이 성립해야 한다. 도로 장면처럼 layout이 비교적 안정적인 경우에는 잘 맞지만, scene composition 자체가 크게 달라지는 domain shift에는 효과가 약해질 수 있다. 논문은 이 가정을 도시 장면 데이터셋에서 잘 활용했지만, 더 일반적인 segmentation 문제로 바로 확장된다는 증거는 본문에 충분히 제시되지 않는다.

둘째, 작은 객체 적응은 여전히 어렵다. 저자들도 pole, traffic sign 같은 작은 물체는 배경과 섞이기 쉽다고 인정한다. 이는 output space를 정렬해도 희소하고 경계가 얇은 클래스는 여전히 불리하다는 뜻이다.

셋째, adversarial training 자체의 불안정성은 완전히 사라지지 않는다. 본문에서는 vanilla GAN을 기본으로 쓰고, 부록에서 LS-GAN이 더 좋다고 보여준다. 이는 기본 objective가 최선은 아니며, 손실 설계에 따라 성능이 꽤 달라질 수 있음을 의미한다.

넷째, 논문은 왜 특정 중간 레벨로 conv4를 선택했고 왜 두 단계가 최적인지에 대해 직관은 제공하지만, 보다 체계적인 구조 탐색 결과를 본문에서 충분히 제공하지는 않는다. “효율과 정확성의 균형 때문에 두 레벨을 썼다”는 설명은 있으나, 더 많은 레벨이 왜 불필요한지 또는 언제 필요한지까지는 명확하지 않다.

비판적으로 보면, 이 논문은 output distribution alignment의 효과를 잘 보였지만, 그것이 정확히 어떤 구조적 통계를 맞추는지까지 해석적으로 분석하지는 않았다. 예를 들어 class co-occurrence, boundary regularity, spatial priors 중 무엇이 가장 중요한지에 대한 분해 분석은 없다. 따라서 “왜 잘 되는가”에 대한 설명은 직관적으로는 강하지만, 분석적으로는 다소 제한적이다. 그럼에도 당시 맥락에서는 매우 영향력 있는 문제 재정의였다고 볼 수 있다.

## 6. 결론

이 논문은 semantic segmentation domain adaptation에서 feature space 대신 output space를 adversarial하게 정렬하는 방법을 제안했다. 핵심 기여는 segmentation 결과가 structured output이라는 점을 적극 활용했다는 데 있다. source와 target의 appearance는 달라도, 의미론적 배치와 국소 문맥은 유사하다는 관찰을 실제 학습 objective로 연결했다.

또한 multi-level adversarial learning을 통해 상위 출력뿐 아니라 중간 레벨에서도 적응 신호를 주어 성능을 더 끌어올렸다. GTA5, SYNTHIA, Cross-City 실험 전반에서 baseline과 기존 방법보다 우수하거나 경쟁력 있는 결과를 보였고, oracle과의 gap도 줄였다. 부록의 LS-GAN 실험까지 보면 이 프레임워크는 손실 함수나 source dataset 변화에도 유연하게 확장 가능함을 시사한다.

실제 적용 측면에서는, 라벨 없는 target 도시나 synthetic-to-real 이전 같은 상황에서 annotation 비용을 줄이면서 segmentation 성능을 높일 수 있다는 점에서 의미가 크다. 향후 연구 측면에서는 이 아이디어가 이후 self-training, pseudo-labeling, image translation, entropy minimization 같은 기법들과 결합될 여지가 크며, 실제로 그런 방향의 후속 연구들이 자연스럽게 이어질 수 있는 출발점을 제공한 논문이라고 평가할 수 있다.
