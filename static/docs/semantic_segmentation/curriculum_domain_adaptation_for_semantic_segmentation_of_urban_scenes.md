# Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes

- **저자**: Yang Zhang, Philip David, Boqing Gong
- **발표연도**: 2017
- **arXiv**: https://arxiv.org/abs/1707.09465

## 1. 논문 개요

이 논문은 synthetic urban scene 데이터로 학습한 semantic segmentation 모델을 real urban scene에 더 잘 일반화시키기 위한 unsupervised domain adaptation 문제를 다룬다. 구체적으로는 SYNTHIA 같은 가상 데이터셋에는 pixel-wise annotation이 풍부하지만, 실제 주행 장면 데이터인 Cityscapes에는 라벨이 없거나 매우 비싸다는 상황에서, source domain의 라벨을 이용해 target domain 성능을 높이는 것이 목표다.

저자들은 semantic segmentation이 단순 분류와 달리 매우 강한 구조를 가진 structured prediction 문제라는 점에 주목한다. 기존 domain adaptation은 보통 source와 target이 어떤 공통 feature space에서 비슷한 분포를 가지도록 만들고, 그 공간에서 같은 예측 함수 $P(Y \mid Z)$가 작동한다고 가정한다. 하지만 저자들은 semantic segmentation에서는 이 가정이 약하다고 본다. 픽셀 간 상관관계가 강하고, 출력 공간이 매우 크며, 중간 feature를 무작정 정렬하면 segmentation에 필요한 구조적 단서까지 약화될 수 있기 때문이다.

이 문제의 중요성은 명확하다. 실제 도심 장면 segmentation은 autonomous driving의 핵심 구성 요소 중 하나지만, 고품질 픽셀 단위 주석을 대규모로 수집하는 비용이 매우 크다. 논문에서도 Cityscapes 한 장을 정밀 주석하고 품질 검수하는 데 1.5시간 이상이 걸린다고 언급한다. 따라서 합성 데이터를 활용하면서도 real-world 성능을 확보하는 방법은 실용적 가치가 크다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 curriculum domain adaptation이다. 즉, 처음부터 가장 어려운 목표인 pixel-wise segmentation을 target domain에서 직접 맞추려 하지 않고, 먼저 더 쉬운 target-domain 추론 과제를 푼 뒤, 그 결과를 이용해 segmentation network를 regularize한다.

저자들이 선택한 쉬운 과제는 두 가지다. 첫째는 이미지 전체 수준의 global label distribution을 추정하는 것이다. 이는 각 target 이미지에서 road, building, sky, car 등이 각각 몇 퍼센트의 픽셀을 차지하는지를 뜻한다. 둘째는 landmark superpixel에 대한 local label distribution을 추정하는 것이다. 이는 신뢰도가 높은 일부 superpixel에 대해 어떤 클래스가 주된 라벨인지 추정하는 과정이다.

이 설계의 직관은 urban scene의 구조적 규칙성에 있다. 예를 들어 하늘은 대개 위쪽에 있고, road는 아래쪽을 넓게 차지하며, building은 road 옆에 위치하는 경향이 있다. 이런 전역적 비율과 국소적 배치는 domain gap이 있어도 픽셀 단위 정확한 분할보다 상대적으로 추정하기 쉽다. 저자들은 이 정보를 이용하면 segmentation network가 target domain에서 적어도 “비정상적으로 불균형한” 예측을 하지 않도록 유도할 수 있다고 본다.

기존 접근과의 차별점은, domain-invariant feature를 직접 학습하는 대신 target prediction이 만족해야 할 “필요한 속성”을 먼저 학습하고, 그 속성을 출력단 regularization으로 사용한다는 점이다. 즉, feature alignment보다는 output constraint와 posterior regularization에 가까운 접근이다.

## 3. 상세 방법 설명

전체 방법은 두 단계의 성격을 갖는다. 먼저 target domain의 쉬운 속성들을 추정한다. 그 다음 source domain의 정답 라벨로 segmentation network를 학습하면서, 동시에 target domain 예측이 앞서 추정한 속성과 일치하도록 만든다.

논문은 target 이미지 $I_t \in \mathbb{R}^{W \times H}$의 픽셀 라벨을 $Y_t \in \mathbb{R}^{W \times H \times C}$로 두고, one-hot encoding을 사용한다. segmentation network의 출력은 픽셀별 softmax 예측 $\hat{Y}_t(i,j,c) \in [0,1]$이다.

각 target 속성은 클래스 분포 $p_t \in \Delta$ 형태로 표현된다. 여기서 $p_t(c)$는 이미지 전체 또는 특정 superpixel에서 클래스 $c$가 차지하는 비율이다. 이미지 전체 수준의 label distribution은 다음과 같이 정의된다.

$$
p_t(c) = \frac{1}{WH} \sum_{i=1}^{W}\sum_{j=1}^{H} Y_t(i,j,c), \quad \forall c
$$

즉, 전체 픽셀 중 클래스 $c$에 속한 픽셀의 비율이다. 네트워크 예측 $\hat{Y}_t$로부터도 같은 방식으로 예측 분포 $\hat{p}_t$를 계산할 수 있다.

저자들의 학습 목표는 target domain에서 네트워크가 만들어내는 분포 $\hat{p}_t$가 추정된 target 속성 $p_t$를 따르도록 하는 것이다. 이를 위해 cross entropy를 사용한다.

$$
\mathcal{C}(p_t, \hat{p}_t) = H(p_t) + KL(p_t, \hat{p}_t)
$$

여기서 $H(p_t)$는 entropy이고, $KL(p_t, \hat{p}_t)$는 KL divergence다. 실제 최적화 목적함수는 다음과 같다.

$$
\min \ \gamma \frac{1}{|S|} \sum_{s \in S} L(Y_s, \hat{Y}_s)
+ (1-\gamma)\frac{1}{|T|} \sum_{t \in T} \sum_k \mathcal{C}(p_t^k, \hat{p}_t^k)
$$

첫 번째 항은 source domain의 pixel-wise cross-entropy loss $L$이다. 이 항이 segmentation network의 기본적인 픽셀 분류 능력을 학습시킨다. 두 번째 항은 target domain의 unlabeled image에 대해, 여러 종류의 속성 $k$에 대한 분포 일치를 강제한다. 여기서 $k$는 예를 들어 global image distribution, landmark superpixel distribution 같은 서로 다른 target property를 뜻한다. $\gamma$는 두 항의 균형을 조절하는 하이퍼파라미터다.

중요한 점은 target domain의 진짜 라벨이 없기 때문에 $p_t^k$를 직접 계산할 수 없다는 것이다. 그래서 저자들은 먼저 이를 추정하는 별도의 쉬운 과제를 푼다.

### Global label distribution 추정

저자들은 target 이미지의 global label distribution을 추정하기 위해 Inception-ResNet-v2 feature를 사용해 몇 가지 방법을 비교한다.

첫째는 multinomial logistic regression이다. 보통 logistic regression은 classification에 쓰이지만, 출력이 클래스 분포이므로 그대로 distribution prediction에 사용할 수 있다. source 이미지에 대해 실제 label distribution $p_s$를 정답으로 두고 학습한 뒤, target 이미지의 distribution을 예측한다.

둘째는 nearest neighbors 방식이다. target 이미지와 가장 가까운 source 이미지들을 찾고, 그들의 label distribution 평균을 target 이미지의 분포로 사용한다. 거리 계산은 $\ell_2$ distance를 사용한다.

비교용으로 source 전체 평균 분포를 모든 target 이미지에 공통으로 쓰는 방법과 uniform distribution도 실험한다.

### Landmark superpixel distribution 추정

이미지 전체 분포만으로는 어느 위치에서 어떤 클래스가 나와야 하는지 공간적 제약이 약하다. 이를 보완하기 위해 superpixel 기반 local property를 사용한다.

각 이미지를 linear spectral clustering으로 100개의 superpixel로 나눈다. source domain에서는 각 superpixel에 dominant label 하나를 부여할 수 있으므로, 이를 이용해 multi-class linear SVM을 학습한다. target superpixel에 대해 SVM은 클래스 라벨과 confidence를 출력한다.

모든 superpixel을 다 regularization에 쓰면 오히려 잘못된 제약이 너무 강해질 수 있으므로, 저자들은 confidence가 높은 상위 60% superpixel만 선택해 landmark superpixel로 사용한다. 이들의 예측 클래스는 one-hot vector로 표현되어 local label distribution 역할을 한다.

superpixel 표현은 시각 정보와 문맥 정보를 함께 담는다. FCN-8s를 PASCAL CONTEXT에 사전학습한 뒤, 각 픽셀에 대해 59개 클래스 detection score를 얻고, 이를 superpixel 내부에서 평균 낸다. 그리고 해당 superpixel 자체의 59차원 벡터뿐 아니라 좌우, 위아래 인접 superpixel의 벡터를 연결하여 최종 특징으로 사용한다. 즉, 단일 영역뿐 아니라 주변 구조까지 반영한다.

### 학습 및 추론 절차

실험에서는 FCN-8s를 segmentation network로 사용하고, convolution layer는 VGG-19로 초기화한다. optimizer는 AdaDelta이며, mini-batch는 source 5장과 target 5장으로 구성한다. adaptation이 없는 baseline은 source 이미지 15장까지 크게 사용한다.

이 방법의 장점은 intermediate feature layer를 수정하지 않고 output loss만 바꾸므로, 다른 segmentation architecture에도 쉽게 적용 가능하다는 점이다. 논문은 이것을 기존 adversarial feature alignment류 방법과 대비되는 실용적 장점으로 제시한다.

## 4. 실험 및 결과

실험은 source domain으로 SYNTHIA, target domain으로 Cityscapes를 사용한다. Cityscapes validation set을 test set으로 사용했고, training set에서 500장을 validation 용도로 따로 분리했다. 두 데이터셋 사이의 공통 클래스는 16개를 수작업으로 정했다. sky, building, road, sidewalk, fence, vegetation, pole, car, traffic sign, person, bicycle, motorcycle, traffic light, bus, wall, rider가 해당된다.

평가 지표는 PASCAL VOC 스타일의 intersection-over-union, 즉

$$
IoU = \frac{TP}{TP + FP + FN}
$$

이다. 여기서 $TP$, $FP$, $FN$은 전체 테스트셋 기준 true positive, false positive, false negative 픽셀 수다.

논문은 먼저 global label distribution 추정 정확도를 비교한다. $\chi^2$ distance 기준으로 성능을 측정했으며, 결과는 다음과 같다.

- Uniform: 1.13
- NoAdapt baseline network prediction: 0.65
- Source mean: 0.44
- Nearest neighbors: 0.33
- Logistic regression: 0.27

즉, target 이미지의 전역 클래스 비율은 단순 baseline segmentation 출력보다 logistic regression으로 더 정확하게 예측할 수 있었다. 이 결과를 바탕으로 이후 실험에서는 logistic regression 출력을 사용한다.

최종 semantic segmentation 결과에서 주요 비교 대상은 다음과 같다.

- `NoAdapt`: SYNTHIA로만 학습한 FCN-8s
- `FCN Wld`: Hoffman et al.의 FCNs in the Wild
- `SP`: superpixel classifier 결과
- `SP Lndmk`: landmark superpixel만 쓴 경우
- `Ours (I)`: global image distribution만 사용
- `Ours (SP)`: landmark superpixel distribution만 사용
- `Ours (I+SP)`: 두 속성을 모두 사용

SYNTHIA에서 Cityscapes로 adaptation하는 주요 mean IoU는 다음과 같다.

- `NoAdapt [27]`: 17.4
- `FCN Wld [27]`: 20.2
- `NoAdapt`(저자 재현): 22.0
- `Ours (I)`: 25.5
- `SP Lndmk`: 23.0
- `SP`: 25.6
- `Ours (SP)`: 28.1
- `Ours (I+SP)`: 29.0

핵심 결과는 `Ours (I+SP)`가 가장 높은 29.0% mean IoU를 기록했다는 점이다. 저자들이 재현한 baseline 22.0 대비 7.0%p 향상이며, 기존 비교 대상인 FCNs in the Wild의 20.2보다도 높다.

세부적으로 보면 global distribution만 쓴 `Ours (I)`는 road와 sidewalk처럼 전역 비율이 크게 왜곡되는 문제를 보정하는 데 효과가 있었다. 반면 superpixel 기반 방법은 sky, road, building처럼 넓은 영역을 차지하는 클래스에서 특히 강했다. 논문은 작은 객체인 fence, traffic light, traffic sign 등은 superpixel 기반 방법이 잘 놓친다고 명시한다. 따라서 image-level constraint와 superpixel-level constraint는 상보적이며, 둘을 합쳤을 때 가장 좋은 결과가 나왔다.

또한 superpixel classifier 자체의 정확도도 분석했다. target domain 전체 superpixel에 대한 분류 정확도는 71%였고, 상위 60% landmark superpixel만 선택하면 정확도가 88% 이상이었다. 이것이 왜 landmark selection이 중요한지 뒷받침한다.

논문은 submission 이후 GTA to Cityscapes 실험도 추가했다. 여기서는 GTA가 SYNTHIA보다 더 photorealistic하므로 전체 성능이 더 높다. 주요 mean IoU는 다음과 같다.

- `NoAdapt [27]`: 21.1
- `FCN Wld [27]`: 27.1
- `NoAdapt`: 22.3
- `Ours (I)`: 23.1
- `Ours (SP)`: 27.8
- `SP`: 26.8
- `Ours (I+SP)`: 28.9

여기서도 `Ours (I+SP)`가 최고 성능이다. 즉, 논문의 접근은 SYNTHIA뿐 아니라 GTA 같은 다른 synthetic dataset에서도 일관되게 유효하다는 점을 보여준다.

추가로 저자들은 domain-invariant feature learning도 시도했다고 밝힌다. FCN-8s의 출력 직전 layer에 maximum mean discrepancy를 걸거나, gradient reversal 기반 domain classifier를 사용하는 실험을 했지만, baseline 대비 눈에 띄는 향상을 얻지 못했다고 한다. 수치는 본문에 제시되지 않았고, “noticeable gain이 없었다”고만 보고된다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 semantic segmentation을 structured prediction 문제로 보고, classification용 domain adaptation 가정을 그대로 가져오는 것이 적절하지 않을 수 있음을 분명히 짚었다는 점이다. 단순 feature alignment 대신 target output이 만족해야 할 구조적 속성을 먼저 추정하고 그것으로 regularize하는 발상은 개념적으로 설득력이 있다.

또 다른 강점은 방법이 단순하고 구현 친화적이라는 점이다. intermediate feature layer를 복잡하게 바꾸지 않고 loss term만 추가하면 되므로 기존 segmentation network에 쉽게 붙일 수 있다. 또한 image-level global cue와 local superpixel cue를 결합해 “어떻게”와 “어디를” 수정할지를 동시에 제공하려는 설계도 합리적이다.

실험적으로도 의미 있는 개선을 보여준다. 저자 재현 baseline 대비 SYNTHIA to Cityscapes에서 22.0에서 29.0으로 향상되었고, 당시 유사 문제를 다룬 거의 유일한 선행연구보다도 좋은 성능을 보였다. 특히 landmark superpixel selection을 통해 noisy pseudo constraint를 그대로 쓰지 않고 confidence filtering을 적용한 부분은 실용적이다.

한계도 분명하다. 첫째, 이 방법이 사용하는 target property는 상대적으로 거친 제약이다. global label distribution은 클래스 비율만 알려줄 뿐 정확한 경계나 세밀한 위치 정보를 주지 못한다. 그래서 작은 객체나 가느다란 구조에 대한 성능 향상은 제한적이다. 실제로 논문도 superpixel 기반 방법이 traffic sign, traffic light, fence 같은 작은 객체를 잘 놓친다고 보고한다.

둘째, superpixel 분류의 품질에 상당히 의존한다. 저자들은 상위 60% landmark만 쓰면 정확도가 88% 이상이라고 보고하지만, 이 선택 비율 60% 자체는 경험적으로 정한 것으로 보이며, 왜 최적인지에 대한 이론적 설명은 없다. 또한 전체 superpixel을 모두 regularize하면 성능 향상이 거의 없었다고 하는데, 이는 constraint noise에 민감함을 의미한다.

셋째, target property 추정기 자체가 source supervision에 의존하기 때문에 domain gap이 더 큰 상황에서 얼마나 견고한지는 본문만으로는 충분히 알기 어렵다. 예를 들어 logistic regression으로 global distribution이 잘 맞는 이유가 urban scene의 공통 구조 덕분인지, 사용한 feature extractor가 충분히 강력해서인지 정교한 분해는 제공되지 않는다.

넷째, 논문은 일부 비교에서 실험 설정 차이를 완전히 제거하지 못했다. 예를 들어 FCNs in the Wild와의 baseline 수치가 꽤 차이나는데, 저자들도 implementation 또는 setup의 미묘한 차이 때문일 수 있다고만 언급한다. 따라서 절대적인 비교에는 약간의 주의가 필요하다.

마지막으로, 이 논문은 target label distribution이 “필요한 속성”이라는 점은 잘 보여주지만, 그것이 semantic segmentation adaptation의 본질적 해법인지까지는 아직 이르다. 즉, 충분조건이 아니라 어디까지나 부분적인 제약이며, 이후 연구에서는 더 풍부한 구조적 prior가 필요하다는 점을 저자들도 인정한다.

## 6. 결론

이 논문은 semantic segmentation domain adaptation에서, 어려운 픽셀 단위 적응을 바로 수행하기보다 먼저 쉬운 target-domain 속성을 학습하고 그것으로 segmentation network를 규제하는 curriculum domain adaptation 프레임워크를 제안했다. 핵심 기여는 target 이미지의 global label distribution과 landmark superpixel의 local label distribution을 추정하고, 이를 source supervised segmentation loss와 함께 학습 목적함수에 통합한 점이다.

실험적으로는 SYNTHIA to Cityscapes와 GTA to Cityscapes 모두에서 baseline 및 기존 방법보다 더 나은 mean IoU를 달성했다. 특히 전역 제약과 국소 제약이 상보적이라는 점을 실험으로 설득력 있게 보였다.

이 연구의 중요성은 synthetic-to-real adaptation을 단순한 feature alignment 문제로 보지 않고, 구조적 출력의 속성을 활용하는 방향을 제시했다는 데 있다. 실제 autonomous driving 같은 응용에서 synthetic label을 적극 활용해야 한다는 점을 고려하면, 이 논문은 이후의 output-space adaptation, self-training, structured regularization 계열 연구로 이어지는 중요한 아이디어적 연결고리로 볼 수 있다. 다만 작은 객체 처리와 제약의 표현력 한계는 남아 있으며, 더 정교한 target property 설계가 향후 연구 과제로 자연스럽게 이어진다.
