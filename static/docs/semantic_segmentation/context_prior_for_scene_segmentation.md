# Context Prior for Scene Segmentation

- **저자**: Changqian Yu, Jingbo Wang, Changxin Gao, Gang Yu, Chunhua Shen, Nong Sang
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2004.01547

## 1. 논문 개요

이 논문은 semantic segmentation에서 문맥 정보(contextual dependencies)를 더 정확하게 활용하기 위한 방법을 제안한다. 기존 연구들은 픽셀 주변의 넓은 문맥을 모으는 데 집중했지만, 서로 다른 종류의 문맥 관계를 명확히 구분하지 않는 경우가 많았다. 저자들은 이것이 오히려 장면 이해를 흐릴 수 있다고 본다. 예를 들어 같은 class에 속한 픽셀들 사이의 관계와, 서로 다른 class 사이의 차이를 나타내는 관계는 역할이 다르다. 그런데 이를 섞어서 집계하면 잘못된 정보가 feature에 들어갈 수 있다.

논문이 다루는 핵심 문제는 바로 이 지점이다. 장면 분할에서는 단순히 멀리 떨어진 픽셀 정보를 많이 모으는 것만으로는 충분하지 않으며, 어떤 픽셀이 현재 픽셀과 같은 semantic category에 속하는지, 어떤 픽셀이 다른 category에 속하는지를 구분해서 다뤄야 한다는 것이다. 저자들은 이를 intra-class context와 inter-class context로 나누어 명시적으로 모델링한다.

이 문제는 semantic segmentation의 본질과 직접 연결된다. segmentation은 픽셀 단위로 class를 예측해야 하므로, appearance가 유사하지만 class는 다른 영역을 구분해야 하고, 반대로 appearance가 조금 달라도 같은 class인 픽셀을 함께 묶어야 한다. 이런 이유로 문맥 관계를 더 구조적으로 표현하는 것은 segmentation 성능 향상에 중요하다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 각 픽셀에 대해 “어떤 다른 픽셀들이 같은 class인가”를 나타내는 **Context Prior**를 학습시키는 것이다. 이 prior는 단순한 attention map이 아니라, ground truth로부터 만든 이상적인 관계 지도(Ideal Affinity Map)를 직접 supervision으로 사용해 학습된다. 즉, attention을 간접적으로 배우는 것이 아니라, 같은 class끼리는 높은 응답을, 다른 class끼리는 낮은 응답을 갖도록 명시적으로 유도한다.

이 설계의 중요한 점은 두 종류의 문맥을 분리해서 다룬다는 것이다. 학습된 prior map $P$는 같은 class 픽셀을 강조하여 intra-class context를 모으고, 반대로 $1 - P$는 다른 class 픽셀을 강조하여 inter-class context를 모은다. 저자들은 이 두 정보가 서로 보완적이라고 본다. intra-class context는 같은 물체나 영역 내부의 일관성을 높여 주고, inter-class context는 경계와 class 차이를 더 분명하게 만드는 역할을 한다.

기존 접근법과의 차별점은 다음과 같다. pyramid pooling이나 ASPP 같은 방식은 넓은 범위의 정보를 모으지만, 그 관계를 semantic class 기준으로 구분하지 않는다. self-attention 계열 방법도 장거리 관계를 선택적으로 모을 수는 있으나, 어떤 관계를 선택해야 하는지에 대한 명시적 규제가 약하다. 반면 이 논문은 ground truth에서 직접 구성한 affinity supervision을 통해 관계의 의미를 분명하게 강제한다.

## 3. 상세 방법 설명

전체 구조는 backbone CNN과 **Context Prior Layer**로 구성된 **CPNet**이다. backbone은 dilated ResNet 같은 일반적인 segmentation backbone이며, backbone feature 위에 Context Prior Layer를 얹어 문맥을 정제한다. 이 레이어 안에는 크게 세 부분이 있다. 첫째, spatial information을 모으는 **Aggregation Module**, 둘째, pixel 간 관계를 예측하는 **Context Prior Map**, 셋째, 이 prior map을 직접 학습시키는 **Affinity Loss**이다.

먼저 입력 이미지 $I$를 backbone에 넣어 feature map $X$를 얻는다. 이 feature는 공간 해상도가 줄어든 상태이며, 각 위치는 하나의 픽셀이라기보다 downsampled spatial position에 대응한다. 이후 Aggregation Module이 주변 공간 정보를 효율적으로 섞어 새로운 feature $\tilde{X}$를 만든다. 저자들은 이 단계가 prior를 추론하는 데 필요하다고 본다. 왜냐하면 픽셀 간 semantic 관계를 판단하려면 단일 위치의 isolated feature만으로는 부족하고, 어느 정도 local spatial context가 있어야 하기 때문이다.

Aggregation Module은 큰 receptive field를 확보하기 위해 fully separable convolution을 사용한다. 구체적으로 $k \times k$ convolution을 바로 쓰지 않고, $k \times 1$과 $1 \times k$의 비대칭 convolution으로 나누고, 여기에 depthwise convolution을 적용한다. 논문은 이를 spatial과 depth 양쪽에서 분리한 fully separable convolution으로 설명한다. 이 방식은 표준 큰 커널 convolution보다 계산량을 줄이면서 비슷한 receptive field를 제공한다.

그 다음, $\tilde{X}$로부터 Context Prior Map $P$를 예측한다. 논문 설명에 따르면 $1 \times 1$ convolution, BN, sigmoid를 거쳐 $H \times W \times N$ 크기의 prior map을 만든다. 여기서 $N = H \times W$이며, reshape하면 사실상 각 위치가 다른 모든 위치와 가지는 관계를 나타내는 $N \times N$ 형태의 맵으로 이해할 수 있다. 각 행은 하나의 기준 픽셀에 대해, 다른 모든 픽셀이 같은 class인지 아닌지를 예측하는 역할을 한다.

이 prior를 학습시키기 위해 먼저 **Ideal Affinity Map** $A$를 ground truth에서 만든다. ground truth label map $L$을 feature 해상도에 맞게 downsample한 뒤 $\tilde{L}$을 얻고, 이를 one-hot encoding하여 $\hat{L}$로 만든다. 그 후 다음 식으로 이상적인 affinity map을 구성한다.

$$
A = \hat{L}\hat{L}^{\top}
$$

이 식의 의미는 단순하다. 두 위치의 one-hot label vector가 같으면 내적 결과가 1이 되고, 다르면 0이 된다. 따라서 $A_{ij}=1$이면 두 픽셀이 같은 class이고, $A_{ij}=0$이면 다른 class이다.

가장 기본적인 supervision은 binary cross entropy 형태의 unary loss이다.

$$
L_u = - \frac{1}{N^2} \sum_{n=1}^{N^2} \left( a_n \log p_n + (1-a_n)\log(1-p_n) \right)
$$

여기서 $p_n$은 prior map의 예측값, $a_n$은 ideal affinity map의 정답값이다. 이 항은 각 픽셀 쌍을 독립적인 binary classification 문제처럼 다룬다.

하지만 저자들은 이것만으로는 부족하다고 본다. 각 행은 하나의 기준 픽셀에 대한 전체 관계 분포를 나타내므로, 같은 class 전체를 얼마나 잘 찾았는지와 다른 class 전체를 얼마나 잘 배제했는지를 함께 반영해야 한다고 주장한다. 이를 위해 global term을 추가한다. 논문은 각 행 $j$에 대해 precision, recall, specificity에 대응하는 세 항을 정의한다.

$$
T_{pj} = \log \frac{\sum_{i=1}^{N} a_{ij} p_{ij}}{\sum_{i=1}^{N} p_{ij}}
$$

$$
T_{rj} = \log \frac{\sum_{i=1}^{N} a_{ij} p_{ij}}{\sum_{i=1}^{N} a_{ij}}
$$

$$
T_{sj} = \log \frac{\sum_{i=1}^{N} (1-a_{ij})(1-p_{ij})}{\sum_{i=1}^{N} (1-a_{ij})}
$$

이를 이용해 global loss는 다음과 같다.

$$
L_g = - \frac{1}{N} \sum_{j=1}^{N} (T_{pj} + T_{rj} + T_{sj})
$$

즉, 한 기준 픽셀에 대해 같은 class 픽셀을 얼마나 정확히 찾았는지, 놓치지 않았는지, 다른 class 픽셀을 얼마나 잘 제외했는지를 동시에 반영한다.

최종 **Affinity Loss**는 unary term과 global term의 합이다.

$$
L_p = \lambda_u L_u + \lambda_g L_g
$$

논문에서는 $\lambda_u = 1$, $\lambda_g = 1$로 설정한다.

학습된 prior map은 실제 feature aggregation에도 사용된다. $\tilde{X}$를 $N \times C_1$로 reshape한 뒤, prior map $P$를 곱해 intra-class context를 얻는다. 반대로 $1-P$를 곱해 inter-class context를 얻는다. 논문 표기에는 두 결과를 모두 $Y$로 적어 약간 혼동이 있으나, 문맥상 하나는 intra-context, 다른 하나는 inter-context를 의미한다. 마지막으로 원래 feature $X$와 이 두 context feature를 concatenate하여 최종 표현 $F$를 만든다.

$$
F = \text{Concat}(X, Y, \bar{Y})
$$

여기서 $Y$는 intra-class context, $\bar{Y}$는 inter-class context에 해당한다고 이해하는 것이 자연스럽다. 이 최종 표현으로 per-pixel segmentation prediction을 수행한다.

네트워크 전체의 최종 loss는 main segmentation loss, auxiliary loss, affinity loss의 합이다.

$$
L = \lambda_s L_s + \lambda_a L_a + \lambda_p L_p
$$

논문에서는 $\lambda_s = 1$, $\lambda_a = 0.4$, $\lambda_p = 1$로 설정했다. $L_s$와 $L_a$는 cross entropy 기반 segmentation loss이고, Cityscapes에서는 class imbalance를 완화하기 위해 bootstrapped cross entropy를 사용한다.

## 4. 실험 및 결과

실험은 ADE20K, PASCAL-Context, Cityscapes의 세 데이터셋에서 수행되었다. backbone은 주로 ResNet-50 또는 ResNet-101이며 dilation strategy를 사용했다. 입력 augmentation으로 mean subtraction, random horizontal flip, random scale을 사용했고, crop 크기는 데이터셋별로 다르다. 최적화는 SGD with momentum 0.9, weight decay $10^{-4}$, batch size 16을 사용했다. learning rate는 poly schedule을 따르며, ADE20K에서는 $2 \times 10^{-2}$, PASCAL-Context와 Cityscapes에서는 $1 \times 10^{-2}$를 사용했다.

평가 지표는 pixel accuracy와 mIoU이다. 추론 시에는 multi-scale and flip testing을 사용해 추가 성능 향상을 얻는다. ADE20K와 PASCAL-Context는 $\{0.5, 0.75, 1.0, 1.5, 1.75\}$ 배율, Cityscapes는 $\{0.5, 0.75, 1, 1.5\}$ 배율을 사용했다.

ADE20K에서의 ablation은 이 논문의 핵심을 가장 잘 보여 준다. 단순 ResNet-50 dilation model은 34.38 mIoU, auxiliary loss를 넣은 baseline은 36.24 mIoU였다. 여기에 ASPP를 붙이면 40.39, PSP를 붙이면 41.49, NonLocal은 40.96, PSA는 41.92가 된다. 반면 저자들의 ContextPriorLayer는 43.92 mIoU를 달성했고, multi-scale test에서는 44.46까지 올랐다. ResNet-101 backbone에서는 45.39, multi-scale test 포함 시 46.27 mIoU를 기록했다.

특히 intra/inter prior를 분리해서 본 실험이 중요하다. IntraPrior branch만 써도 BCE 기준 42.34 mIoU, InterPrior branch도 41.88 mIoU를 얻었다. 여기에 Affinity Loss를 붙이면 각각 42.74, 42.43으로 올라간다. 두 branch를 함께 쓰는 ContextPriorLayer는 43.92를 기록한다. 이는 intra-class context와 inter-class context가 서로 보완적이라는 저자들의 주장을 뒷받침한다.

Aggregation Module의 커널 크기에 대한 분석도 제시된다. Context Prior 없이 커널 크기만 키웠을 때는 성능 변화가 크지 않았다. 그러나 Context Prior를 사용할 때는 적절한 local spatial information이 있을수록 성능 향상이 커졌고, 커널 크기 11에서 43.92 mIoU로 최고치를 기록했다. 이후 더 큰 커널에서는 성능이 다시 떨어졌다. 저자들은 이를 통해 Context Prior가 관계를 추론하려면 “적절한” 공간 정보가 필요하다고 해석한다.

또한 Context Prior의 일반화 가능성을 보기 위해 PPM, ASPP 위에도 Context Prior를 얹어 보았다. 그 결과 PPM은 41.49에서 42.55로, ASPP는 40.39에서 42.69로 향상되었다. 즉, 제안한 prior 자체가 특정 aggregation module에만 묶이지 않고, 다른 문맥 집계 구조와도 결합될 수 있음을 보여 준다.

정량 비교 결과도 강하다. ADE20K validation에서 CPNet101은 46.27 mIoU를 기록해 PSPNet, PSANet, EncNet, CFNet, ANL 등 당시 강한 baseline보다 높은 수치를 보였다. CPNet50조차 44.46 mIoU로, 일부 더 깊은 backbone을 쓴 기존 방법보다 경쟁력 있는 결과를 냈다.

PASCAL-Context에서는 ResNet-101 기반 CPNet101이 53.9 mIoU를 기록했다. 표에서 EncNet 51.7, DANet 52.6, ANL 52.8보다 높은 값이다. 논문은 EncNet보다 1.0 point 이상 높다고 명시한다.

Cityscapes test set에서는 fine annotation만 사용한 조건에서 81.3 mIoU를 달성했다. 이는 DenseASPP의 80.6보다 0.9 point 높고, ANL과 동률이다. 논문은 coarse annotation 없이 fine dataset만으로 이 결과를 냈다는 점을 강조한다.

정성 결과에서도 CPNet은 혼동되기 쉬운 영역을 더 안정적으로 분할한다. 예시로 그림 1과 그림 5에서 sand와 sea, table과 bed처럼 외형이 유사한 경우 기존 pyramid/attention 기반 방법은 잘못된 문맥을 섞어 오분류하는 반면, CPNet은 class 관계를 구분하여 더 정확한 예측을 만든다고 설명한다. 또 prior map visualization에서는 Affinity Loss 없이 얻은 attention map도 대략적인 경향은 있으나, Affinity Loss를 넣었을 때 ground-truth affinity 구조에 더 가까운 prior map이 형성된다고 보인다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문맥을 많이 모으는 것과 문맥을 “올바르게 구분하는 것”을 분리해서 본 점이다. semantic segmentation에서는 넓은 문맥을 보는 것이 중요하다는 점은 이미 알려져 있었지만, 이 논문은 같은 class 관계와 다른 class 관계를 명시적으로 분리해 supervision하는 방향을 제시했다. 이는 단순한 architectural tweak보다는 문맥 관계의 의미 자체를 다시 정의한 접근으로 볼 수 있다.

두 번째 강점은 supervision 설계가 명확하다는 점이다. Ideal Affinity Map은 ground truth label에서 직접 구성되므로, prior map이 무엇을 배워야 하는지가 분명하다. attention 기반 방법들이 종종 “무엇을 보고 있는지”는 보이지만 왜 그런 관계가 형성되었는지는 अस्पष्ट한 반면, 이 방법은 prior의 의미가 명시적이다. 또한 unary BCE에 더해 row-level precision, recall, specificity를 반영한 global term을 설계한 점은 픽셀쌍 예측을 구조적으로 다룬다는 점에서 설득력이 있다.

세 번째 강점은 실험 설계가 비교적 충실하다는 점이다. 단순 성능 비교뿐 아니라 IntraPrior, InterPrior, Affinity Loss, Aggregation Module 크기, 다른 aggregation module과의 결합 등 여러 ablation을 제공해 제안 요소의 기여를 분해해서 보여 준다. 이는 논문의 주장을 검증하는 데 도움이 된다.

반면 한계도 분명하다. 첫째, prior map의 크기는 본질적으로 pixel-to-pixel 관계를 다루는 $N \times N$ 구조이므로 계산량과 메모리 비용 문제가 잠재적으로 크다. 논문은 fully separable convolution으로 일부 비용을 줄였지만, prior map 자체의 quadratic nature가 완전히 해소되지는 않는다. 다만 본문에서 정확한 메모리/연산 복잡도 비교를 자세히 제시하지는 않았다.

둘째, 이 방법은 학습 시 ground truth로부터 이상적인 affinity를 구성하는 강한 supervision에 의존한다. 따라서 label quality가 떨어지거나 boundary noise가 큰 데이터셋에서 어떤 영향을 받는지는 본문에서 상세히 논의되지 않았다. 또한 downsampled label map에서 affinity를 만드는 만큼, fine-grained boundary 정보가 얼마나 유지되는지도 더 분석할 여지가 있다.

셋째, intra-class와 inter-class를 binary하게 구분하는 설계는 명확하다는 장점이 있지만, class 간 관계의 세밀한 종류까지는 표현하지 않는다. 예를 들어 “chair와 table은 자주 함께 나타난다” 같은 higher-order semantic relation은 이 논문에서 직접 모델링하지 않는다. 즉, 관계를 같은 class / 다른 class의 2분법으로 요약한 셈이다.

비판적으로 보면, 이 논문은 attention을 “명시적 prior supervision이 없는 관계 모델링”으로 보고, 여기에 affinity supervision을 추가해 구조를 부여한 것으로 이해할 수 있다. 이런 방향은 타당하지만, 결국 prior map 역시 feature에서 예측되는 dense relation map이라는 점에서 self-attention과 완전히 다른 패러다임이라기보다는, supervised relation learning에 가까운 확장이라고 보는 것이 적절하다.

## 6. 결론

이 논문은 scene segmentation에서 문맥 정보를 단순히 넓게 모으는 데서 한 걸음 더 나아가, **같은 class 문맥과 다른 class 문맥을 명시적으로 구분해 활용하는 Context Prior**를 제안했다. 이를 위해 ground truth로부터 Ideal Affinity Map을 만들고, 이를 supervision으로 사용하는 **Affinity Loss**를 설계했다. 또한 이 prior를 실제 feature aggregation에 연결하는 **Context Prior Layer**와 이를 포함한 **CPNet**을 제시했다.

실험적으로 CPNet은 ADE20K, PASCAL-Context, Cityscapes에서 강한 성능을 보였고, 여러 ablation을 통해 IntraPrior, InterPrior, Affinity Loss가 각각 유효함을 보였다. 특히 문맥 관계를 semantic class 기준으로 구분해서 다루는 것이 segmentation 품질 향상에 실질적으로 도움이 된다는 점을 설득력 있게 보여 준다.

향후 연구 관점에서 보면, 이 논문은 segmentation에서 relation modeling을 supervision과 결합하는 방향의 대표적인 예로 볼 수 있다. 실제 응용에서는 자율주행이나 장면 이해처럼 class 혼동이 치명적인 환경에서 유용할 가능성이 크다. 동시에 더 큰 해상도, 더 효율적인 relation approximation, 혹은 binary affinity를 넘어선 richer semantic relation modeling으로 확장될 여지도 크다.
