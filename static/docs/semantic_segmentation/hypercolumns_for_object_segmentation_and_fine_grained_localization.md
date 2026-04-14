# Hypercolumns for Object Segmentation and Fine-grained Localization

- **저자**: Bharath Hariharan, Pablo Arbelaez, Ross Girshick, Jitendra Malik
- **발표연도**: 2015
- **arXiv**: https://arxiv.org/abs/1411.5752

## 1. 논문 개요

이 논문은 CNN의 마지막 레이어 feature만 사용하는 기존 관행이 정밀한 위치 추정에는 부적절하다는 문제를 다룬다. 분류나 detection에서는 top layer가 semantic information을 잘 담지만, pooling과 subsampling 때문에 spatial resolution이 거칠어져서 object boundary, keypoint, part 같은 세밀한 위치 정보를 잃기 쉽다. 반대로 낮은 레이어는 위치는 정밀하지만 semantic discrimination이 약하다.

저자들은 이 둘을 결합하기 위해 특정 pixel 위치에서 그 위를 통과하는 여러 CNN 레이어의 activation들을 하나로 모은 **hypercolumn** 표현을 제안한다. 핵심 주장은, fine-grained localization에 필요한 정보가 CNN의 특정 한 레이어가 아니라 여러 수준의 feature에 분산되어 있으므로 이를 함께 써야 한다는 것이다.

이 문제는 중요한데, object segmentation, keypoint localization, part labeling은 모두 detection 이후의 정밀한 이해를 요구하는 대표적 vision task이기 때문이다. 논문은 이들을 별개의 문제로 보지 않고, detection box 안에서 각 위치에 확률을 예측하는 **pixel-wise classification** 문제로 통일해 다룬다.

## 2. 핵심 아이디어

중심 아이디어는 간단하다. 어떤 위치를 정확히 이해하려면, 그 위치의 저수준 edge/texture 정보와 고수준 semantic 정보를 동시에 봐야 한다는 것이다. 이를 위해 입력 위치 $i$에서 여러 레이어의 feature map 값을 모은 벡터를 hypercolumn으로 정의한다. 이 표현은 낮은 레이어의 정밀한 localization과 높은 레이어의 semantic abstraction을 동시에 활용하려는 설계다.

또 하나의 중요한 아이디어는 **location-specific classifier**이다. 같은 feature라도 bounding box 내부의 어디에 나타나느냐에 따라 의미가 달라질 수 있다. 예를 들어 사람 bounding box의 위쪽에 있는 nose-like pattern은 head에 가까울 수 있지만, 아래쪽에서는 그렇지 않다. 저자들은 이 효과가 단순한 위치 bias로는 충분히 표현되지 않는다고 보고, 공간 위치마다 다른 classifier가 필요하다고 주장한다.

하지만 pixel마다 classifier를 하나씩 두면 파라미터 수가 너무 커지고 학습 데이터가 부족해진다. 그래서 $K \times K$ 크기의 coarse grid classifier를 학습하고, 실제 pixel 위치의 classifier는 이 grid 위 classifier들을 interpolation해서 얻는다. 이로써 위치 의존성을 유지하면서도 parameter sharing을 확보한다.

기존 접근과의 차별점은 다음과 같다. 이 논문은 단순히 multi-scale image input을 결합하거나 같은 수준의 CNN feature만 여러 해상도에서 합치는 것이 아니라, **CNN의 서로 다른 depth에 있는 feature를 같은 위치 기준으로 정렬하여 결합**한다. 또한 이를 실제 segmentation, keypoint, part labeling에 공통적으로 적용 가능한 형태로 정식화했다.

## 3. 상세 방법 설명

전체 파이프라인은 detection 결과를 입력으로 받아, detection box를 조금 확장한 영역에서 $50 \times 50$ heatmap을 예측하는 구조다. 이 heatmap은 task마다 의미가 다르다. object segmentation에서는 해당 위치가 object 내부일 확률, part labeling에서는 각 part에 속할 확률, keypoint prediction에서는 특정 keypoint가 그 위치에 있을 확률을 뜻한다.

먼저 detection bounding box를 crop하여 고정 크기로 resize한 뒤 CNN에 통과시킨다. 각 intermediate output은 feature map인데, 레이어마다 spatial resolution이 다르므로 바로 같은 위치의 feature를 읽을 수 없다. 저자들은 각 feature map을 bilinear interpolation으로 목표 해상도에 맞게 upsample한다. 논문은 upsampled feature를 다음과 같이 쓴다.

$$
f_i = \sum_k \alpha_{ik} F_k
$$

여기서 $F_k$는 원래 feature map의 위치 $k$의 값이고, $f_i$는 출력 해상도에서 위치 $i$의 feature이다. $\alpha_{ik}$는 bilinear interpolation 계수다. 이렇게 얻은 여러 레이어의 feature를 concat한 것이 hypercolumn이다. 예시로, `pool2` 256채널, `conv4` 384채널, `fc7` 4096채널을 쓰면 총 4736차원 feature가 된다.

이제 각 위치를 분류해야 하는데, 단일 classifier 하나로는 위치에 따른 의미 차이를 반영하기 어렵다. 그래서 저자들은 $K \times K$ grid의 classifier $g_k(\cdot)$를 두고, 실제 pixel 위치 $i$에서의 classifier $h_i(\cdot)$를 이들의 선형 결합으로 정의한다.

$$
h_i(\cdot) = \sum_k \alpha_{ik} g_k(\cdot)
$$

그리고 위치 $i$의 hypercolumn feature를 $f_i$라고 하면 최종 확률은

$$
p_i = \sum_k \alpha_{ik} g_k(f_i) = \sum_k \alpha_{ik} p_{ik}
$$

가 된다. 즉, 각 pixel에 대해 모든 grid classifier의 출력을 계산한 뒤, 위치에 따른 interpolation 계수로 섞어서 최종 예측을 만든다.

학습은 이상적으로는 이 interpolation까지 포함해 end-to-end로 해야 하지만, 처음 제시한 간단한 방법에서는 train time에 interpolation을 무시한다. 대신 각 training bounding box를 $K \times K$ cell로 나누고, $k$번째 classifier는 모든 training example의 $k$번째 cell에 해당하는 pixel만 이용해 logistic regression으로 학습한다. 이는 test-time objective를 직접 최적화하지는 않지만 구현이 단순하고 과적합을 줄이는 실용적 타협이다.

효율성도 중요한 문제다. feature map들을 모두 $50 \times 50$으로 upsample한 뒤 고차원 classifier를 적용하면 계산량이 크다. 저자들은 선형성 때문에 순서를 바꿀 수 있다는 점을 이용한다. classifier weight를 feature map별 block으로 나누면, 어떤 위치의 score는

$$
w^T f_i = \sum_j w^{(j)T} f_i^{(j)}
$$

처럼 쓸 수 있다. 또 upsampling이 선형이므로,

$$
f_i^{(j)} = \sum_k \alpha_{ik}^{(j)} F_k^{(j)}
$$

$$
w^{(j)T} f_i^{(j)} = \sum_k \alpha_{ik}^{(j)} w^{(j)T} F_k^{(j)}
$$

가 성립한다. 즉, 먼저 각 feature map 위에서 classifier를 적용한 뒤, 그 score map만 upsample해 합치면 된다. 저자들은 이를 `1 x 1` convolution으로 구현한다. 더 나아가 `n x n` convolution을 사용하면 단일 unit뿐 아니라 주변 activation 패턴까지 볼 수 있어, 특히 낮은 레이어에서 더 유용할 수 있다고 설명한다.

논문은 이 전체 과정을 하나의 neural network로 다시 표현한다. 각 선택된 feature map 위에 $K^2$ channel을 가진 convolution layer를 붙이고, 이를 upsample해 합한 뒤 sigmoid를 통과시켜 heatmap을 만든다. 그리고 grid classifier interpolation을 적용해 최종 출력으로 만든다. 이 모든 단계는 differentiable하므로 end-to-end fine-tuning이 가능하다. 학습 loss는 $50 \times 50$ 모든 위치에 대한 logistic loss의 합이다. 논문은 sigmoid, linear combination, log-likelihood를 하나의 composite function처럼 다루는 것이 더 안정적이었다고 적고 있다.

라벨 생성 방식은 task별로 다르다. SDS에서는 instance 내부 pixel을 positive, 외부를 negative로 둔다. part labeling에서는 특정 part 내부만 positive다. keypoint prediction에서는 정답 keypoint 주변 일정 반경 안을 positive, 그 밖을 negative로 둔다. 반경은 bounding box diagonal의 10%를 사용했다.

## 4. 실험 및 결과

### SDS: System 1

첫 번째 실험은 기존 SDS pipeline [22]의 refinement 단계를 hypercolumn 기반 refinement로 바꾼 것이다. VOC2012 Train으로 학습하고 VOC2012 Val에서 평가한다. baseline [22]는 region proposal을 CNN으로 scoring하고 refinement를 top-layer feature로 수행한다.

저자들의 `Hyp` 시스템은 bounding box regression과 fine-tuning 없이도 mean AP$r$@0.5를 49.7에서 51.2로, mean AP$r$@0.7을 25.3에서 31.6으로 높였다. 특히 overlap threshold 0.7에서 큰 향상이 나타났는데, 이는 coarse detection 개선이 아니라 **정밀한 mask localization**이 좋아졌다는 뜻이다. bounding box regression과 fine-tuning을 더한 `Hyp+FT+bbox-reg`는 mean AP$r$@0.5에서 52.8, @0.7에서 33.7을 기록해 [22]보다 각각 3.1점, 8.4점 높았다.

ablation도 중요하다. `Only fc7`는 [22]와 유사한 top-layer-only baseline인데 성능이 확실히 낮다. `fc7+pool2`, `fc7+conv4`, `pool2+conv4` 같은 실험은 어느 한 수준의 feature를 빼도 성능이 떨어진다는 점을 보여 준다. grid resolution 실험에서는 `1 x 1` classifier, 즉 사실상 위치 비의존 classifier로 가면 성능이 하락했고, `5 x 5` grid만 되어도 거의 full performance를 회복했다. 이 결과는 hypercolumn 자체뿐 아니라 위치별 classifier interpolation도 중요한 기여를 한다는 뜻이다.

semantic segmentation으로 변환한 결과도 보고한다. [9]의 pasting scheme을 이용했을 때 VOC2012 Segmentation Test에서 mean IU 54.6을 얻었고, [22]의 51.6보다 3점 높았다.

### SDS: System 2

두 번째 시스템은 계산 효율을 높인 새로운 pipeline이다. region proposal 전체를 처리하지 않고, R-CNN detection 결과에서 시작해 일부 인접 box만 추가한 뒤 각 box를 hypercolumn으로 segmentation하고, segmentation된 region을 다시 scoring한다. 전체 proposal보다 후보 수가 훨씬 적어 효율적이다.

T-Net 기반으로 bounding box만 이용해 segmentation을 예측했을 때, `Hyp`는 mean AP$r$@0.5 49.1, @0.7 29.1을 기록했다. region feature까지 쓰는 경우보다는 조금 낮지만, [22]와 비교하면 @0.7에서 여전히 약 4점 우수했다. `Only fc7` baseline은 @0.5 44.0, @0.7 16.3으로 크게 낮아, top layer만으로는 sharp segmentation이 어렵다는 점이 분명하게 드러난다.

더 큰 O-Net architecture를 쓰면 성능이 크게 상승한다. O-Net 기반 `Hyp`는 mean AP$r$@0.5 56.5, @0.7 37.0을 기록했고, 이는 T-Net보다 각각 7.5점, 8점 높다. 여기에 segmentation 기반 rescoring까지 포함한 최종 pipeline은 mean AP$r$@0.5 60.0, @0.7 40.4를 달성했다. 논문 기준으로 이는 SDS에서 state-of-the-art라고 주장한다. 또 semantic segmentation benchmark에서는 mean IU 62.6을 얻었다.

### Keypoint Prediction

사람 category의 keypoint localization은 VOC2009 val 후반부 person subset에서 APK metric으로 평가했다. detection과 keypoint localization이 동시에 요구되는 setting이다. keypoint마다 별도 heatmap을 예측하고, 가장 높은 응답 위치를 keypoint prediction으로 삼는다. 최종 keypoint score는 detection score와 predicted heatmap score를 곱해 만든다. 이는 keypoint가 안 보이는 경우를 false positive로 세는 APK 평가 방식에 맞춘 설계다.

결과를 보면, 기존 [20]의 평균 APK 15.2에 대해 hypercolumn 기반 `Hyp`는 17.0, fine-tuning한 `Hyp+FT`는 18.5를 기록했다. 즉, 논문이 강조하는 3.3점 향상은 `Hyp+FT`와 [20]의 차이다. 반대로 `Only fc7` baseline은 평균 10.6으로 매우 낮고, 심지어 HOG 기반 [21]보다도 못한 경우가 있다. 이는 keypoint처럼 아주 정밀한 위치 정보가 필요한 문제에서는 hypercolumn의 이점이 특히 크다는 것을 보여 준다.

### Part Labeling

part labeling은 person, horse, cow, sheep, cat, dog, bird에 대해 평가했다. part annotation은 [10]을 사용했고, person은 head, torso, arms, legs, 네 발 동물은 head, torso, legs, tail, bird는 head, torso, legs, wings, tail로 묶었다. 각 part마다 classifier를 따로 학습하고, test time에는 figure-ground mask 내부 각 pixel에 대해 최고 score part를 할당한다.

평가 지표는 AP$r^{part}$이며, intersection-over-union 계산 시 part label까지 맞아야 intersection으로 센다. `Only fc7` 대비 `Hyp`는 거의 모든 category에서 향상되었고, 평균적으로 6.6점 개선되었다고 논문은 말한다. 예를 들어 person은 21.9에서 28.5, horse는 16.6에서 27.8, cat은 19.2에서 30.3으로 크게 증가했다. 단, bird는 15.4에서 14.2로 오히려 약간 낮았다. 따라서 “거의 전 범주에서 향상”이라고 보는 것이 정확하며, 모든 범주에서 일관된 향상이라고 말하면 과장이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 매우 단순한 개념인 hypercolumn을 통해 서로 다른 세 가지 fine-grained localization task를 하나의 공통 framework로 설명하고, 실제로 모두에서 큰 성능 향상을 보였다는 점이다. 특히 SDS에서 overlap threshold 0.7에서의 큰 개선은 boundary와 shape 같은 정밀한 localization 품질이 좋아졌음을 잘 보여 준다. 또한 top-layer-only baseline과의 비교가 반복적으로 제시되어, 성능 향상이 단순히 더 큰 모델이나 더 좋은 detector 때문이 아니라 multi-layer representation 자체에서 온다는 점이 설득력 있게 드러난다.

또 다른 장점은 방법이 해석 가능하다는 점이다. 낮은 레이어는 localization, 높은 레이어는 semantics를 담당한다는 직관이 명확하고, location-specific classifier interpolation 역시 왜 필요한지 논문에서 충분히 설명한다. 연산 효율을 위해 classifier 적용과 upsampling의 순서를 바꾸는 부분도 수식으로 명료하게 정리되어 있다.

한계도 있다. 먼저, 논문의 기본 학습 절차는 train time에는 interpolation을 무시하고 각 grid classifier를 독립적으로 logistic regression으로 학습하는 heuristic에 의존한다. 저자들은 이후 neural network 형태로 joint training도 가능하다고 말하지만, 핵심 결과 중 일부는 이 근사 학습 방식에 기반한다. 즉, 제안한 test-time model과 train-time objective가 완전히 일치하지 않는다.

또한 이 방법은 detection box가 이미 주어졌다는 설정에 크게 의존한다. 논문도 unconstrained image 전체를 직접 dense prediction하는 것이 아니라, detection system이 제공한 bounding box를 시작점으로 사용한다. 따라서 localization 성능의 일부는 upstream detector 품질에 제한받는다.

task별로도 제한이 있다. part labeling은 대부분 향상되지만 bird category에서는 개선이 없다. 이는 hypercolumn이 항상 모든 세부 category 구조에 동일하게 잘 작동하는 것은 아니라는 점을 시사한다. 또한 계산 효율 개선을 제시하긴 했지만, 당시 기준으로도 여러 feature map, upsampling, per-box prediction을 포함하므로 완전히 가벼운 구조는 아니다.

비판적으로 보면, 이 논문은 “왜 특정 레이어 조합이 좋은가”, “왜 특정 neighborhood 크기나 grid 크기가 적절한가”를 완전히 일반적인 원리로 설명하지는 않는다. 상당 부분은 경험적 설계다. 그럼에도 불구하고 실험적 근거는 충분히 강하다.

## 6. 결론

이 논문은 CNN의 여러 depth에서 얻은 feature를 위치별로 결합한 hypercolumn representation이 fine-grained localization에 매우 효과적임을 보여 준다. 구체적으로 SDS, keypoint prediction, part labeling이라는 세 가지 task에서 일관된 성능 향상을 달성했고, 특히 top-layer-only feature의 한계를 명확히 드러냈다.

핵심 기여는 세 가지로 요약할 수 있다. 첫째, multi-layer pixel descriptor인 hypercolumn 개념을 제안했다. 둘째, 위치 의존적 예측을 위해 grid classifier interpolation을 도입했다. 셋째, 이를 segmentation과 part localization 같은 정밀한 vision task에 실질적으로 적용해 강한 성능을 보였다.

이 연구는 이후 semantic segmentation, instance segmentation, feature pyramid 계열 접근, multi-scale fusion 같은 흐름에 중요한 연결고리로 볼 수 있다. 논문 자체는 future work로 attribute나 action classification 같은 다른 세밀한 인식 문제를 언급하는데, 실제로 이 아이디어는 이후 dense prediction 전반에 널리 영향을 주었다.
