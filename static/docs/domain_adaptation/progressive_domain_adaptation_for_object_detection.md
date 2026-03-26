# Progressive Domain Adaptation for Object Detection

* **저자**: Han-Kai Hsu, Chun-Han Yao, Yi-Hsuan Tsai, Wei-Chih Hung, Hung-Yu Tseng, Maneesh Singh, Ming-Hsuan Yang
* **발표연도**: 2019
* **arXiv**: [https://arxiv.org/abs/1910.11319](https://arxiv.org/abs/1910.11319)

## 1. 논문 개요

이 논문은 object detection에서의 unsupervised domain adaptation 문제를 다룬다. 기본 설정은 bounding box annotation이 있는 source domain에서 detector를 학습하고, annotation이 없는 target domain에서도 잘 동작하도록 만드는 것이다. 그러나 source와 target의 시각적 분포 차이, 즉 domain gap이 크면 기존의 직접적인 adaptation은 학습이 불안정해지고 성능도 제한된다. 저자들은 이 문제를 한 번에 풀지 않고, source와 target 사이에 intermediate synthetic domain을 삽입하여 더 쉬운 두 개의 적응 문제로 나누어 푸는 progressive adaptation을 제안한다.

이 연구가 중요한 이유는 object detection 모델이 실제 환경 변화에 매우 민감하기 때문이다. 카메라가 달라지거나, 날씨가 달라지거나, 데이터 수집 환경이 바뀌면 supervised detector의 일반화 성능이 크게 떨어질 수 있다. 하지만 새로운 도메인마다 bounding box annotation을 새로 수집하는 것은 비용이 매우 크다. 따라서 annotation 없는 target domain으로 detector를 이전하는 방법은 자율주행, 감시, 로보틱스 등에서 실제적인 가치가 크다.

## 2. 핵심 아이디어

이 논문의 핵심 직관은 source와 target을 직접 맞추는 대신, 두 분포 사이에 intermediate domain을 두고 적응을 단계적으로 수행하면 더 안정적이고 효과적이라는 것이다. 저자들은 source 이미지를 CycleGAN으로 target처럼 보이도록 변환하여 synthetic target image 집합을 만들고, 이를 별도의 도메인 $\mathbb{F}$로 정의한다. 이렇게 하면 원래의 어려운 적응 문제 $\mathbb{S} \rightarrow \mathbb{T}$를, 상대적으로 쉬운 두 문제인 $\mathbb{S} \rightarrow \mathbb{F}$와 $\mathbb{F} \rightarrow \mathbb{T}$로 분해할 수 있다.

이 접근의 차별점은 synthetic image를 단순한 데이터 증강처럼 취급하지 않는다는 점이다. 기존에도 image translation으로 생성한 이미지를 target-style training data처럼 활용하는 접근은 있었지만, 이 논문은 synthetic image 집합을 “중간 다리 역할의 독립 도메인”으로 명시적으로 정의하고 그 위에서 두 단계의 adversarial feature alignment를 수행한다. 또한 모든 synthetic image의 품질이 같지 않다는 점을 반영하여, target 분포에 더 가까운 synthetic image일수록 더 큰 가중치를 주는 weighted task loss를 도입한다. 즉, intermediate domain을 사용한다는 점과 그 안의 샘플 품질 차이를 학습 목적에 반영한다는 점이 핵심이다.

## 3. 상세 방법 설명

전체 파이프라인은 세 단계로 이해할 수 있다. 먼저 source image를 target 스타일로 변환해 intermediate domain $\mathbb{F}$를 만든다. 다음으로 labeled source domain $\mathbb{S}$와 unlabeled synthetic domain $\mathbb{F}$ 사이에서 1단계 adaptation을 수행한다. 마지막으로 synthetic domain $\mathbb{F}$를 supervision 가능한 쪽으로 간주하고, unlabeled target domain $\mathbb{T}$와의 2단계 adaptation을 수행한다. 이렇게 하면 첫 단계에서는 주로 appearance 차이를 줄이고, 두 번째 단계에서는 target에 더 가까운 분포에서 object와 context 차이를 맞추도록 유도할 수 있다.

기본 detection framework는 Faster R-CNN이다. Backbone encoder $E$가 이미지 $\mathbf{I}$로부터 feature map $E(\mathbf{I})$를 추출하고, 이후 RPN과 ROI classifier가 이어진다. Detection loss는 다음과 같다.

$$
\mathcal{L}_{det}(E(\mathbf{I})) = \mathcal{L}_{rpn} + \mathcal{L}_{cls} + \mathcal{L}_{reg}.
$$

여기서 $\mathcal{L}_{rpn}$은 proposal 관련 손실, $\mathcal{L}_{cls}$는 분류 손실, $\mathcal{L}_{reg}$는 bounding box 회귀 손실이다. 즉 detector 자체는 표준 Faster R-CNN의 목적함수를 따른다.

도메인 정렬을 위해 저자들은 encoder 뒤에 domain discriminator $D$를 추가한다. 이 discriminator는 feature가 source에서 왔는지 target에서 왔는지를 픽셀 위치 단위 확률 맵으로 예측한다. 출력은 $\mathbf{P} = D(E(\mathbf{I})) \in \mathbb{R}^{H \times W}$이고, source 이미지의 domain label은 $d=0$, target 이미지는 $d=1$이다. Discriminator loss는 binary cross-entropy로 정의된다.

$$
\mathcal{L}_{disc}(E(\mathbf{I})) = - \sum_{h,w} \Big( d \log \mathbf{P}^{(h,w)} - (1-d) \log(1-\mathbf{P}^{(h,w)}) \Big).
$$

이 손실은 discriminator가 domain을 잘 구분하도록 학습시키는 역할을 한다. 반대로 encoder는 discriminator를 속이는 방향으로 학습되어야 하므로, 저자들은 Gradient Reversal Layer, 즉 GRL을 사용한다. GRL은 forward pass에서는 아무 변화가 없지만 backward pass에서 gradient의 부호를 반전시킨다. 그 결과 encoder는 discriminator loss를 최대화하는 방향, 즉 domain 구분이 어렵도록 feature를 만드는 방향으로 업데이트된다. 이 adversarial learning을 통해 domain-invariant feature를 학습하게 된다.

직접적인 source-to-target adaptation의 전체 목적함수는 다음과 같다.

$$
\min_E \max_D \mathcal{L}(\mathbf{I}_{\mathbb{S}}, \mathbf{I}_{\mathbb{T}}) = \mathcal{L}_{det}(\mathbf{I}_{\mathbb{S}}) + \lambda_{disc} \Big[ \mathcal{L}_{disc}(E(\mathbf{I}_{\mathbb{S}})) + \mathcal{L}_{disc}(E(\mathbf{I}_{\mathbb{T}})) \Big].
$$

여기서 $\lambda_{disc}$는 detection loss와 discriminator loss의 균형을 조절하는 계수다.

논문의 핵심은 이 목적을 곧바로 $\mathbb{S} \rightarrow \mathbb{T}$에 적용하지 않고, progressive adaptation으로 나누는 것이다. Intermediate domain $\mathbb{F}$는 CycleGAN으로 source image를 target 스타일로 변환해 만든 synthetic target image 집합이다. 논문은 $\mathbb{S}$와 $\mathbb{F}$는 장면 내용은 같고 appearance만 다르며, $\mathbb{F}$와 $\mathbb{T}$는 픽셀 수준 분포는 유사하지만 세부 디테일이 다르다고 설명한다. 따라서 $\mathbb{F}$는 source와 target 사이의 중간 분포 역할을 한다.

1단계 adaptation에서는 $\mathbb{S}$를 labeled domain으로, $\mathbb{F}$를 unlabeled domain으로 놓고 정렬한다. 이 단계에서는 source 내용이 유지된 상태에서 appearance 변환만 일어났으므로, 모델은 주로 저수준 appearance discrepancy를 줄이는 데 집중할 수 있다. 2단계 adaptation에서는 이제 $\mathbb{F}$를 supervision 가능한 도메인처럼 사용하면서 $\mathbb{T}$와 정렬한다. 이 단계에서는 1단계에서 얻은 appearance-invariant representation을 바탕으로 target에 대한 더 높은 수준의 정렬을 수행하려는 의도가 있다.

하지만 synthetic image의 품질은 균일하지 않다. 어떤 이미지는 target처럼 자연스럽게 번역되지만, 어떤 이미지는 객체 디테일이 사라지거나 artifact가 생긴다. 논문은 이런 저품질 synthetic sample이 detector를 혼란스럽게 하고 잘못된 alignment를 유도할 수 있다고 본다. 이를 해결하기 위해, 2단계에서 synthetic image마다 다른 importance weight를 detection loss에 곱한다.

가중치는 CycleGAN의 discriminator $D_{cycle}$ 출력으로부터 얻는다. 논문은 최적 discriminator가 다음과 같은 형태를 가진다고 설명한다.

$$
D^*_{cycle}(\mathbf{I}) = \frac{p_{\mathbb{T}}(\mathbf{I})} {p_{\mathbb{S}}(\mathbf{I}) + p_{\mathbb{T}}(\mathbf{I})}.
$$

이 식은 synthetic image $\mathbf{I}$가 target 분포 $p_{\mathbb{T}}$에 더 가까울수록 discriminator score가 커진다는 뜻이다. 즉 score가 높을수록 더 target-like한 synthetic sample로 볼 수 있고, 이런 샘플에 더 큰 supervision weight를 주는 것이 자연스럽다. 실제 importance weight는 다음처럼 정의된다.

$$
w(\mathbf{I}) =
\begin{cases}
D_{cycle}(\mathbf{I}), & \text{if } \mathbf{I} \in \mathbb{F} \\
1, & \text{otherwise.}
\end{cases}
$$

즉 synthetic domain에 속한 샘플만 discriminator 기반 가중치를 적용하고, 그 외 샘플은 1의 고정 가중치를 쓴다. 이때 2단계 최종 목적함수는 다음과 같이 재정의된다.

$$
\min_E \max_D \mathcal{L}(\mathbf{I}_{\mathbb{F}}, \mathbf{I}_{\mathbb{T}}) = w(\mathbf{I}_{\mathbb{F}}) \mathcal{L}_{det}(\mathbf{I}_{\mathbb{F}}) + \lambda_{disc} \Big[ \mathcal{L}_{disc}(E(\mathbf{I}_{\mathbb{F}})) + \mathcal{L}_{disc}(E(\mathbf{I}_{\mathbb{T}})) \Big].
$$

이 식의 의미는 간단하다. Target에 더 가까운 synthetic image는 더 신뢰하고 강한 detector supervision을 주며, target과 거리가 먼 synthetic outlier는 학습에 덜 반영한다는 것이다.

구현 세부사항도 비교적 명확하다. Detection backbone은 VGG16 기반 Faster R-CNN이고, discriminator는 3×3 convolution 4개 층으로 구성된다. 앞의 3개 층은 64채널과 leaky ReLU를 사용하고, 마지막 층은 1채널 출력으로 domain classification을 수행한다. 학습은 SGD를 사용하며 learning rate 0.001, weight decay 0.0005, momentum 0.9, $\lambda_{disc}=0.1$, batch size 1을 사용한다. Synthetic domain은 CycleGAN으로 생성한다.

다만 논문이 세부 구현을 모두 명확히 밝히지는 않는다. 예를 들어 1단계 학습 후 2단계 학습으로 어떻게 정확히 이어 붙이는지, 그리고 $D_{cycle}$의 어떤 출력을 어떤 방식으로 이미지 단위 점수로 집계하는지는 본문에서 세밀하게 설명되지 않는다. 따라서 이 부분은 논문에 명시된 범위 이상으로 단정하면 안 된다.

## 4. 실험 및 결과

저자들은 세 가지 실제적 시나리오에서 방법을 평가한다. 첫째는 cross-camera adaptation, 둘째는 weather adaptation, 셋째는 large-scale dataset adaptation이다. 각 실험에서는 adaptation이 전혀 없는 Faster R-CNN baseline, target label을 모두 활용한 oracle, 그리고 제안 방법 및 변형들을 비교한다.

첫 번째 실험은 KITTI를 source, Cityscapes를 target으로 둔 cross-camera adaptation이다. 이 실험에서는 두 도메인에 공통으로 존재하는 **car** 클래스의 AP를 평가한다. 결과는 다음과 같다. Adaptation이 없는 Faster R-CNN은 AP 28.8, 기존 방법인 *FRCNN in the wild*는 38.5, 저자들의 “Ours (w/o synthetic)”는 38.2다. Synthetic image를 단순 증강처럼 넣은 “Ours (synthetic augment)”는 40.6으로 오른다. 제안한 “Ours (progressive)”는 43.9를 달성한다. Oracle은 55.8이다.

이 결과는 두 가지를 보여준다. 첫째, synthetic domain을 도입하는 것 자체가 도움이 된다. 둘째, synthetic image를 단순 증강으로만 쓰는 것보다 progressive adaptation 구조로 쓰는 편이 더 낫다. 40.6에서 43.9로의 추가 향상은 intermediate domain을 명시적 적응 단계로 다루는 설계가 실질적인 효과를 냈음을 보여준다.

같은 설정에서 weighted task loss 분석도 제시된다. KITTI to Cityscapes 실험에서 synthetic image에 고정 가중치 0.8, 0.9, 1.0, 1.1, 1.2를 주었을 때 AP는 각각 39.8, 42.8, 42.2, 41.1, 42.6이다. 반면 저자들의 discriminator 기반 importance weighting은 43.9를 달성한다. 즉 샘플마다 target 근접도를 반영한 동적 weighting이 단순한 고정 상수보다 더 효과적이다.

두 번째 실험은 Cityscapes에서 Foggy Cityscapes로의 weather adaptation이다. Foggy Cityscapes는 Cityscapes 이미지에 fog를 합성한 데이터이므로, 장면 구조는 거의 같고 날씨 표현만 다르다. 이 실험에서는 8개 클래스에 대한 mAP를 사용한다. Baseline Faster R-CNN은 mAP 19.6이고, *FRCNN in the wild*는 27.6, *Diversify & Match*는 34.6, *Strong-Weak Align*은 34.3, *Selective Align*은 33.8이다. 저자들의 “Ours (w/o synthetic)”는 26.9지만, “Ours (synthetic augment)”는 36.1, “Ours (progressive)”는 36.9를 기록해 비교 방법들을 모두 앞선다. Oracle은 39.2다.

이 실험은 논문의 설계를 잘 뒷받침한다. Source와 target이 본질적으로 같은 장면에 다른 weather effect만 입은 관계이므로, image translation으로 만든 synthetic domain이 target에 매우 가까워지기 쉽다. 그래서 synthetic domain을 활용하는 효과가 특히 크게 나타난다. 실제로 baseline 26.9에서 progressive 36.9로 약 10포인트 개선이 발생했다. 클래스별로도 person, rider, car, bus 등에서 전반적으로 향상된다.

세 번째 실험은 Cityscapes에서 BDD100k daytime subset으로의 adaptation이다. 이는 작은 source dataset에서 더 크고 훨씬 다양한 target dataset으로 일반화해야 하는 어려운 설정이다. 평가 클래스는 10개이고 지표는 mAP다. Baseline Faster R-CNN은 20.8, “Ours (w/o synthetic)”는 21.2, “Ours (synthetic augment)”는 23.7, “Ours (progressive)”는 24.3이다. Oracle은 43.3이다.

이 결과는 도메인 차이가 큰 경우의 한계도 함께 보여준다. Direct adaptation만으로는 0.4포인트 개선에 그쳐 거의 효과가 없지만, synthetic domain을 도입하면 23.7, progressive adaptation까지 적용하면 24.3으로 올라간다. 즉 intermediate feature space가 어려운 적응 문제에도 분명한 도움을 주지만, oracle과의 차이가 매우 크므로 이 문제가 여전히 본질적으로 어렵다는 점도 드러난다.

전체적으로 실험은 세 가지 메시지를 준다. 첫째, synthetic domain은 단순 데이터 증강보다 “중간 적응 단계”로 활용할 때 더 효과적이다. 둘째, synthetic sample 품질 차이를 weighting으로 반영하는 것이 실제로 성능 향상에 기여한다. 셋째, 이 접근은 카메라 차이, 날씨 변화, 데이터 다양성 증가 같은 서로 다른 domain-shift 상황에서 일관된 이득을 보인다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 domain adaptation의 난도를 구조적으로 낮추는 문제 분해 방식이 매우 직관적이면서도 실험적으로 설득력 있다는 점이다. Source와 target 사이에 synthetic intermediate domain을 두고 두 단계로 적응한다는 발상은 단순하지만, 왜 쉬워지는지에 대한 설명이 분명하다. $\mathbb{S}$와 $\mathbb{F}$는 내용은 같고 appearance만 다르며, $\mathbb{F}$와 $\mathbb{T}$는 픽셀 분포는 비슷하지만 디테일이 다르다는 해석은 실제 자율주행 데이터셋 상황과 잘 맞는다.

두 번째 강점은 synthetic data를 무조건 신뢰하지 않는다는 점이다. 많은 방법이 translated image를 추가 학습 데이터처럼 사용하지만, 이 논문은 translation quality가 고르지 않다는 현실적인 문제를 명확히 짚는다. 그리고 CycleGAN discriminator score를 사용해 synthetic sample의 중요도를 다르게 두는 weighted supervision을 제안한다. 구현 난도가 아주 높지 않으면서도 fixed weight보다 더 나은 성능을 보여 실용적 의미가 크다.

세 번째 강점은 평가 범위가 넓다는 점이다. 카메라 차이, 날씨 변화, 대규모 데이터셋 적응이라는 서로 다른 형태의 domain shift를 다루기 때문에, 방법이 특정 데이터셋 조합에만 통하는 것처럼 보이지 않는다. 특히 weather adaptation에서 강한 baseline들을 넘는 결과를 보여 방법의 강점을 분명히 드러낸다.

반면 한계도 있다. 첫째, 전체 방법이 image translation 품질에 크게 의존한다. Intermediate domain의 핵심이 CycleGAN으로 생성한 synthetic image인데, translation이 불안정하거나 객체 구조를 잘 보존하지 못하면 2단계 supervision 자체가 오염될 수 있다. 저자들은 weighted loss로 이를 완화하지만, translation 품질이 매우 낮은 상황에서 얼마나 견고한지는 본문만으로 충분히 알기 어렵다.

둘째, 파이프라인 복잡도가 커진다. 기본 detector 외에 CycleGAN 학습이 필요하고, adaptation도 두 단계로 나누어 수행해야 한다. 따라서 직접적인 adversarial alignment보다 학습 비용과 구현 부담이 커진다. 논문은 성능 향상을 보여주지만, 이 추가 복잡도 대비 이득이 어떤 상황에서 특히 큰지에 대한 정교한 분석은 제한적이다.

셋째, feature alignment 자체가 class-aware alignment를 명시적으로 보장하지는 않는다. 논문은 주로 global feature distribution alignment에 기반하며, instance-level 또는 class-conditional alignment를 별도로 설계하지 않았다. 따라서 도메인 차이가 매우 크고 클래스별 appearance 변화가 복잡한 경우에는 한계가 있을 수 있다. 실제로 Cityscapes to BDD100k 실험에서 oracle과 큰 격차가 남는 점이 이를 간접적으로 보여준다.

넷째, $D_{cycle}$의 점수를 target 근접도의 대리 지표로 쓰는 해석은 실용적으로는 타당해 보이지만, 그것이 sample quality를 얼마나 정확히 반영하는지에 대한 이론적 검증은 충분히 깊게 제시되지 않는다. 즉 이 weighting 전략은 효과적이지만, 해석의 엄밀성은 제한적이다.

비판적으로 보면, 이 논문은 adaptation의 이론적 본질을 새롭게 푼다기보다, “큰 domain gap을 작게 나누어 푼다”는 관점을 object detection에 잘 적용한 강한 engineering contribution에 가깝다. 그럼에도 실제 detection adaptation의 학습 안정성과 성능 향상을 함께 확보했다는 점에서 충분히 의미 있는 연구다.

## 6. 결론

이 논문은 object detection을 위한 unsupervised domain adaptation에서, source와 target 사이의 큰 domain gap을 intermediate synthetic domain으로 완충하여 두 단계로 적응하는 progressive adaptation framework를 제안했다. 또한 synthetic image 품질이 균일하지 않다는 문제를 고려해, target 분포에 가까운 샘플에 더 높은 비중을 주는 weighted task loss를 도입했다. 결과적으로 제안법은 KITTI to Cityscapes, Cityscapes to Foggy Cityscapes, Cityscapes to BDD100k daytime 같은 다양한 시나리오에서 기존 방법 대비 일관된 개선을 보였다.

이 연구의 중요한 의미는 domain adaptation을 한 번에 직접 해결해야 하는 정렬 문제로만 보지 않고, intermediate domain을 통해 점진적으로 해결할 수 있다는 관점을 object detection에 효과적으로 적용했다는 데 있다. 실제 응용 측면에서는 새로운 카메라, 새로운 날씨, 더 큰 배포 환경으로 detector를 이전해야 하는 자율주행 계열 문제에서 특히 유용하다. 향후에는 더 강력한 image translation, class-aware alignment, instance-level weighting, 또는 end-to-end joint optimization과 결합하면서 이 아이디어를 확장할 수 있을 것이다.
