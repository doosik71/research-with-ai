# Analyzing Overfitting under Class Imbalance in Neural Networks for Image Segmentation

- **저자**: Zeju Li, Konstantinos Kamnitsas, Ben Glocker
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2102.10365

## 1. 논문 개요

이 논문은 class imbalance가 매우 심한 image segmentation 문제에서 neural network가 어떤 방식으로 overfitting하는지를 분석하고, 그 분석에 근거해 새로운 asymmetric objective function들을 제안하는 연구이다. 저자들이 특히 주목한 상황은 medical image segmentation처럼 foreground가 매우 작고 background가 압도적으로 많은 경우이다. 이런 환경에서는 단순히 “소수 클래스가 어렵다”는 수준을 넘어, 모델이 under-represented class의 training sample에는 지나치게 잘 맞지만 test sample에는 잘 일반화하지 못하는 현상이 발생한다.

논문의 핵심 문제의식은 기존 class imbalance 대응법이 왜 기대만큼 잘 작동하지 않는가에 있다. 저자들은 단순히 성능 지표만 보는 대신, 마지막 classification layer의 activation인 logit 분포를 train/test에서 비교해 네트워크의 내부 거동을 직접 관찰한다. 그 결과, 데이터가 적고 imbalance가 강할수록 foreground 클래스의 test-time logit이 decision boundary 쪽으로 이동하거나 심지어 경계를 넘어가는 현상을 반복적으로 확인한다. 반면 background 클래스는 상대적으로 안정적이다. 이 비대칭적 shift는 small structure를 체계적으로 under-segment하게 만들며, 결과적으로 sensitivity 저하로 이어진다.

이 문제는 중요하다. segmentation에서는 foreground, 특히 작은 병변이나 작은 장기 구조를 놓치는 것이 실제 응용에서 치명적일 수 있기 때문이다. 논문은 단지 “imbalance가 문제다”를 반복하는 것이 아니라, 왜 false negative가 늘어나는지에 대한 기전을 logit distribution shift라는 관점에서 설명하고, 그 기전에 맞춘 asymmetric regularization을 제안했다는 점에서 의미가 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 명확하다. class imbalance 하에서 overfitting은 모든 클래스에 대칭적으로 나타나지 않고, under-represented class의 unseen sample에 대해서만 주로 불리한 방향으로 작동한다는 것이다. 저자들은 이 현상을 foreground logit의 biased shift로 해석한다. training 중에는 소수 클래스 샘플 수가 적기 때문에 CNN이 그 샘플들을 사실상 외워버리는 식으로 매우 case-specific한 filter를 만들 수 있다. 그러면 training foreground는 decision boundary에서 멀리 떨어진 위치로 강하게 매핑되지만, test foreground는 이 필터들과 잘 맞지 않아 activation magnitude가 줄어들고 boundary 쪽으로 밀리게 된다.

이 관찰에서 바로 설계 원리가 나온다. 만약 under-represented class의 logit이 test time에 경계 쪽으로 이동한다면, training 단계에서 이 클래스가 더 넓은 안전 여유를 갖도록 만들면 된다. 즉, foreground 쪽 logit을 의도적으로 decision boundary에서 더 멀리 떨어뜨리거나, foreground의 representation space를 더 넓히거나, foreground 중심으로 regularization을 더 강하게 주는 방향이 유리하다는 것이다.

기존 접근과의 차별점은 “대칭적 regularization”을 버렸다는 데 있다. large margin loss, focal loss, adversarial training, mixup, augmentation은 원래 대체로 모든 클래스에 비슷하게 적용된다. 하지만 저자들은 이 논문에서 그런 symmetric 처리 방식이 imbalance 환경에서는 적절하지 않다고 주장한다. background는 이미 안정적이므로 추가 보정이 크게 필요하지 않고, 오히려 foreground에만 편향된 asymmetric bias를 주어야 한다는 것이다. 이 점이 논문의 가장 중요한 차별점이다.

## 3. 상세 방법 설명

전체 방향은 기존 loss function과 regularization 기법을 새로 발명하기보다는, 잘 알려진 방법들을 “rare class에만 유리하게” 비대칭 수정하는 것이다. 공통 목표는 하나다. under-represented class의 logit activation이 decision boundary를 넘어 background 쪽으로 밀려나는 일을 막는 것이다.

논문은 semantic segmentation의 한 sample을 $(x_i, y_i)$로 두고, $y_i$를 one-hot label로 표현한다. 클래스 수를 $c$라 하면, 기본 cross-entropy loss는 다음처럼 쓴다.

$$
L_{CE}(x_i, y_i) = - \sum_{j=1}^{c} y_{ij}\log(p_{ij})
$$

여기서 $p_{ij}$는 softmax로 계산한 클래스 확률이며,

$$
p_{ij} = \frac{e^{z_{ij}}}{\sum_{j=1}^{c} e^{z_{ij}}}
$$

이다. $z_{ij}$는 마지막 layer의 logit이다.

논문은 DSC loss도 함께 다룬다. segmentation에서 자주 쓰이는 Dice 계열 손실을 sample-wise smooth form으로 사용하며, 이는 false positive와 false negative를 직접 반영하는 구조를 갖는다. 저자들은 cross-entropy와 DSC 양쪽에 대해 같은 asymmetric 아이디어를 적용한다.

### Asymmetric large margin loss

기존 large margin loss는 모든 클래스의 logit에 margin $m$을 대칭적으로 적용한다. 그러나 저자들의 해석에 따르면 이런 symmetric margin은 클래스 간 거리를 벌리더라도 decision boundary를 여전히 중앙에 두기 때문에, test foreground가 boundary 쪽으로 shift되는 현상을 근본적으로 막지 못한다.

그래서 rare class indicator $r_j \in \{0,1\}$를 두고, rare class에만 margin을 적용한다. 수정된 softmax는 다음과 같다.

$$
\hat{q}_{ij} = \frac{e^{z_{ij} - y_{ij}r_jm}}{\sum_{j=1}^{c} e^{z_{ij} - y_{ij}r_jm}}
$$

그리고 이를 이용해 asymmetric CE large margin loss를 정의한다.

$$
\hat{L}^{M}_{CE}(x_i, y_i) = - \sum_{j=1}^{c} y_{ij}\log(\hat{q}_{ij})
$$

직관적으로는 foreground에만 추가 margin을 주어, foreground sample이 학습 시 더 boundary에서 멀어지도록 유도하는 것이다. 논문에서는 binary segmentation에서는 foreground에 대해 $r_j=1$, background에 대해 $r_j=0$으로 둔다. 또한 더 일반적인 문제에서는 $r_j \in [0,1]$처럼 rarity를 연속값으로 둘 수도 있다고 언급한다.

### Asymmetric focal loss

기존 focal loss는 잘 분류된 sample의 loss weight를 줄이고, boundary 근처의 어려운 sample에 더 집중한다. CE 형태는 다음과 같다.

$$
L^{focal}_{CE}(x_i, y_i) = - \sum_{j=1}^{c}(1-p_{ij})^\gamma y_{ij}\log(p_{ij})
$$

하지만 논문은 imbalance 환경에서 foreground training sample이 꼭 “어려운 샘플”이 아니라고 본다. 오히려 소수 클래스는 training에서 이미 overfit되어 쉽게 맞혀질 수 있다. 이런 상황에서 focal term은 foreground의 weight를 줄여버리고, 결국 foreground logit을 boundary 근처에 머물게 만들어 test 시 false negative를 늘릴 수 있다.

그래서 저자들은 rare class에서는 attenuation을 제거한다. 즉 foreground는 기본 CE처럼 유지하고, background 같은 non-rare class에만 focal attenuation을 남긴다.

$$
\hat{L}^{focal}_{CE}(x_i, y_i) =
\sum_{j=1}^{c}
\left(
-r_j y_{ij}\log(p_{ij})
-
(1-r_j)(1-p_{ij})^\gamma y_{ij}\log(p_{ij})
\right)
$$

이 설계는 foreground를 decision boundary에서 더 멀리 유지하려는 목적과 정확히 맞물린다. 논문은 DSC 기반 focal variant도 새로 제안하는데, false negative 관련 항에 attenuation을 넣는 방식으로 loss 크기를 지나치게 왜곡하지 않으면서 focal-like behavior를 내도록 설계했다. 저자들은 기존 logarithmic DSC loss나 focal Tversky보다 자신들의 focal DSC가 다른 loss와 결합하기 쉬운 magnitude를 유지한다고 설명한다.

### Asymmetric adversarial training

기존 adversarial training은 입력 $x_i$에 대해 loss를 가장 크게 증가시키는 perturbation 방향 $d^{adv}$를 찾아, 원본 sample과 adversarial sample을 함께 학습한다.

$$
L_{adv}(x_i,y_i)=L(x_i,y_i)+L\left(x_i + l\cdot \frac{d^{adv}}{\|d^{adv}\|_2}, y_i\right)
$$

여기서

$$
d^{adv} = \arg\max_{d;\|d\|<\epsilon} L(x_i+d, y_i)
$$

이다.

문제는 symmetric adversarial training 역시 decision boundary를 대칭적으로 다루기 때문에 rare foreground에 필요한 추가 여유를 만들지 못할 수 있다는 점이다. 이에 따라 저자들은 rare class를 더 많이 고려하는 방향으로 adversarial sample을 생성한다. 식으로는 rare class indicator $r$를 label에 곱해, rare class와 관련된 sample에서 더 중점적으로 adversarial perturbation을 유도한다.

$$
\hat{d}^{adv} = \arg\max_{d;\|d\|<\epsilon} L(x_i+d, y_i \cdot r)\ \ \text{if } y_i \cdot r > 0
$$

핵심은 rare class 주변 feature space를 더 robust하게 만들어 test-time shift를 덜 일으키게 하는 것이다.

### Asymmetric mixup

기존 mixup은 두 sample $(x_i,y_i)$, $(x_k,y_k)$를 선형 결합해 soft label을 가진 중간 sample을 만든다.

$$
\tilde{x}_i = \lambda x_i + (1-\lambda)x_k,\quad
\tilde{y}_i = \lambda y_i + (1-\lambda)y_k
$$

논문은 이런 symmetric mixup이 decision boundary를 클래스 중간에 두도록 유도하기 때문에, imbalance 환경에서 foreground에 특별히 유리하지 않다고 본다. 그래서 저자들은 soft label 대신 hard label을 사용하고, foreground와 충분히 가까운 mixed sample은 foreground로 간주하는 asymmetric mixup을 제안한다. 구체적으로는 mixing coefficient $\lambda$와 margin $m$을 이용해, foreground 쪽 비중이 충분히 크면 mixed sample을 foreground label로 둔다. 반대로 애매한 sample은 학습에서 제외한다.

직관적으로 보면, foreground 주변 latent space를 더 넓게 채워 넣어 foreground decision region을 확장하는 방식이다. 논문은 BRATS에서는 이 방법이 매우 강하게 작동했지만, intensity overlap이 큰 단일 채널 데이터셋인 ATLAS나 KiTS에서는 효과가 상대적으로 제한적이었다고 해석한다.

### Asymmetric augmentation

표준 data augmentation은 모든 클래스에 비슷한 확률과 세기로 변환을 적용한다. 그러나 저자들은 background는 이미 충분히 많으므로, background augmentation을 똑같이 많이 할 필요가 없다고 본다. 따라서 rare foreground에는 기존 augmentation $A(x_i)$를 적용하고, background에는 더 약한 augmentation $A_{small}(x_i)$만 적용한다.

$$
\hat{\tilde{x}}_i =
\begin{cases}
A(x_i) & \text{if } y_i \cdot r = 1 \\
A_{small}(x_i) & \text{otherwise}
\end{cases}
$$

이 방법은 구현이 단순하지만, foreground 쪽 변이를 더 많이 늘려 일반화를 도와준다.

### 결합 방법

논문은 위 asymmetric 기법들이 서로 보완적이라고 보고, adversarial sample, mixup sample, asymmetric augmentation으로 확장한 데이터에 asymmetric large margin과 asymmetric focal을 결합한 combined objective도 제안한다. CE 기반 결합 loss는 다음과 같이 정리된다.

$$
\hat{L}^{combine}_{CE}(x_i, y_i)=
\sum_{j=1}^{c}
\left(
-r_j y_{ij}\log(\hat{q}_{ij})
-
(1-r_j)(1-\hat{q}_{ij})^\gamma y_{ij}\log(\hat{q}_{ij})
\right)
$$

즉, 여러 방식으로 foreground 쪽에 bias를 걸어 under-represented class의 logit distribution을 더 넓고 안정적으로 유지하려는 전략이다.

### 보충 분석: large margin과 focal loss의 gradient 해석

Supplementary에서 저자들은 focal loss와 large margin loss를 sample re-weighting 관점에서도 분석한다. 일반 CE의 logit gradient는

$$
\frac{\partial L_{CE}(x_i,y_i)}{\partial z_i}=p_i-y_i
$$

이다. sample weight $w_i$를 곱하면 gradient는 $w_i(p_i-y_i)$가 된다. 저자들은 focal loss도 결국 sample difficulty에 따라 gradient에 scalar weight를 곱하는 구조라고 보며, easy sample의 weight를 줄인다고 설명한다. 반대로 large margin loss는 gradient를 키우는 효과가 있고, 특히 잘 맞혀진 sample에 더 강하게 작동해 logit을 더 boundary 밖으로 밀어낸다고 해석한다. 이 분석은 왜 symmetric large margin이 overfitting된 foreground를 더 밀어버릴 수 있는지, 또 왜 asymmetric modification이 필요한지를 이론적으로 뒷받침한다.

## 4. 실험 및 결과

실험은 네 가지 medical segmentation task에서 수행되었다. BRATS2017의 brain tumor core segmentation, ATLAS의 brain stroke lesion segmentation, abdominal organ segmentation, KiTS19의 kidney 및 kidney tumor segmentation이다. 아키텍처는 DeepMedic과 3D U-Net을 사용했다. 중요한 점은 이 현상이 특정 데이터셋이나 특정 모델에만 국한되지 않는지를 보기 위해 데이터셋과 모델을 일부러 다양하게 구성했다는 것이다.

클래스 불균형은 매우 심하다. 예를 들어 평균 imbalance ratio는 BRATS에서 약 $712.8:1$, ATLAS에서 약 $1768.7:1$, KiTS-Tumor에서 약 $6736.6:1$, gallbladder에서도 약 $4887.5:1$ 수준이다. 이런 수치는 foreground가 극도로 희귀한 상황임을 보여준다.

### 관찰 실험: 데이터 양 변화에 따른 성능 변화

저자들은 training data 양을 줄여 가며 모델이 더 쉽게 overfit하도록 만들었다. 그 결과, train set에서는 accuracy가 높아지지만 test set에서는 DSC가 감소했다. 더 중요한 것은 이 성능 저하의 형태이다. precision은 비교적 안정적인 반면 sensitivity가 크게 떨어진다. specificity는 foreground에 대해 거의 항상 $>0.999$로 유지되었다고 보고한다. 즉, 모델은 background는 계속 잘 맞히지만 foreground를 더 많이 놓치게 된다. 이는 under-segmentation이 주된 실패 양상임을 뜻한다.

### Logit distribution 분석

BRATS, KiTS, ATLAS에서 train/test의 logit distribution을 비교한 결과, foreground sample의 test-time logit이 train-time보다 decision boundary 쪽으로 의미 있게 이동했다. 데이터가 적을수록 shift가 더 컸다. 또 multi-class인 KiTS에서는 더 희귀한 kidney tumor가 kidney보다 더 크게 shift했다. 이 결과는 rare class일수록 generalization gap이 logit 수준에서 더 크게 드러난다는 주장을 뒷받침한다.

### 정량 결과: BRATS

BRATS brain tumor core segmentation에서, baseline CE는 training data 5%일 때 DSC 50.4, sensitivity 41.0이었다. 여기에 asymmetric 기법을 적용하면 대부분 성능이 개선되었다. 예를 들어 asymmetric large margin은 DSC 56.8, asymmetric focal loss는 58.8, asymmetric adversarial training은 58.5, asymmetric mixup은 59.8을 달성했다. 가장 좋은 것은 asymmetric combination으로, DSC 63.4, sensitivity 63.1을 기록했다. 10% training data에서도 asymmetric combination은 DSC 72.4, sensitivity 72.9로 baseline CE의 DSC 62.5, sensitivity 56.0보다 크게 향상되었다.

흥미로운 점은 symmetric counterpart가 항상 좋지 않았다는 것이다. 예를 들어 original large margin loss는 5% data에서 DSC 44.5로 baseline보다도 낮았고, symmetric combination도 개선이 거의 없거나 악화되는 경우가 있었다. 이 결과는 “regularization을 더 넣으면 된다”가 아니라 “foreground에 비대칭적으로 넣어야 한다”는 논문 주장을 강하게 지지한다.

### 정량 결과: ATLAS

ATLAS stroke lesion segmentation에서는 baseline with augmentation이 30% training에서 DSC 22.2, sensitivity 18.3이었다. asymmetric combination은 이를 DSC 31.1, sensitivity 27.9까지 끌어올렸다. 50% training에서도 baseline 45.2 대비 asymmetric combination 52.2로 개선되었다. 100% training에서도 baseline 54.5보다 asymmetric combination 58.5가 높았다.

이 데이터셋에서도 성능 향상의 핵심은 주로 sensitivity 증가이다. 예를 들어 asymmetric focal loss는 100% training에서 sensitivity를 63.2까지 높였지만 precision은 55.6으로 다소 떨어졌다. 즉, foreground를 더 많이 잡는 대신 false positive가 늘 수 있다. 저자들은 이런 trade-off는 connected component 기반 post-processing으로 상당 부분 완화할 수 있다고 설명한다.

### 정량 결과: KiTS

KiTS에서는 kidney와 kidney tumor를 동시에 세그멘트하지만, 논문의 관심은 특히 더 희귀한 kidney tumor에 있다. kidney 자체는 비교적 쉬운 클래스라 baseline도 매우 높다. 예를 들어 kidney는 100% training에서 DSC가 대체로 96 이상이다. 반면 kidney tumor는 훨씬 어려워 baseline with augmentation이 10% training에서 DSC 54.6, sensitivity 46.0이다.

여기서 asymmetric 방법들은 kidney tumor에 대해 일관된 개선을 보였다. asymmetric focal loss는 10% training에서 DSC 57.9, asymmetric large margin은 55.5, asymmetric adversarial training은 55.2, asymmetric mixup은 56.8이었다. 가장 좋은 것은 again asymmetric combination으로 10% training에서 DSC 59.2, sensitivity 52.2를 기록했다. 50% training에서는 asymmetric combination이 DSC 79.4, sensitivity 77.0으로 baseline 76.0, 72.8보다 좋았고, 100% training에서도 82.7, 82.1로 baseline 79.2, 77.0보다 높았다.

반면 kidney class에서는 asymmetric combination이 오히려 최선은 아니었다. 이는 논문의 목적이 전체 클래스에 대해 균일한 개선을 만드는 것이 아니라, under-represented class의 sensitivity를 구제하는 데 있다는 점을 보여준다.

### Multi-class abdomen 실험

Supplementary에서는 abdominal organ segmentation에서도 asymmetric focal loss를 시험했다. 이 실험은 multi-class setting에서도 제안이 적용 가능함을 보여준다. 저자들은 class 4, 5, 8, 9, 10, 11, 12, 13을 rare class로 정의했다. 결과적으로 asymmetric focal loss는 rare class 평균 DSC를 48.3에서 53.2로 약 4.9%p 향상시켰다. 예를 들어 gallbladder는 DSC 34.6에서 53.7로, vena cava는 65.1에서 73.9로, vein은 42.0에서 43.3으로 개선되었다. 다만 esophagus는 오히려 나빠졌는데, 저자들은 지나치게 작은 구조라 post-processing이 진짜 positive를 지워버리는 문제가 있다고 해석한다.

### 후처리의 역할

논문은 post-processing 유무에 따라 결과를 모두 제시한다. asymmetric 방법들은 일반적으로 sensitivity와 DSC를 높이지만, false positive가 멀리 떨어져 생기면 Hausdorff distance 같은 distance metric이 나빠질 수 있다. 그래서 저자들은 connected component 기반으로 가장 큰 영역만 남기는 단순 후처리를 사용했고, 그 결과 HD도 대체로 개선되거나 비슷한 수준으로 맞출 수 있었다고 주장한다. 즉, 제안 방법은 “더 민감하게 잡는 대신 약간 noisy해질 수 있다”는 특성이 있으며, 이는 practical pipeline에서 보완 가능하다는 입장이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제를 정성적 직감이 아니라 logit distribution 분석으로 구체화했다는 점이다. 많은 연구가 class imbalance에서 성능이 떨어진다고 말하지만, 이 논문은 왜 특히 sensitivity가 망가지는지, 왜 foreground만 일방적으로 손해를 보는지를 네트워크 activation 수준에서 설명한다. 이 설명은 BRATS, ATLAS, KiTS, abdomen이라는 서로 다른 task와 DeepMedic, 3D U-Net이라는 서로 다른 architecture에서 반복적으로 관찰되었다.

또 다른 강점은 제안이 실용적이라는 점이다. 완전히 새로운 복잡한 알고리즘을 제시하기보다, large margin, focal loss, adversarial training, mixup, augmentation이라는 기존 도구를 약간 수정해 바로 적용 가능하게 만들었다. 이런 형태는 재현성과 확장성 면에서 장점이 있다. Supplementary에서 hyper-parameter 표와 sensitivity analysis, 코드 저장소까지 제공한 점도 실용성을 높인다.

실험 결과도 비교적 설득력 있다. 단순 baseline뿐 아니라 re-weighting, F-score 계열 loss, symmetric version과의 비교를 충분히 수행했고, asymmetric combination이 대부분의 어려운 setting에서 최고 또는 최상위 성능을 보였다. 특히 under-represented class의 sensitivity를 일관되게 끌어올린다는 점이 논문 주장과 정합적이다.

다만 한계도 분명하다. 첫째, 분석과 실험의 대부분이 medical image segmentation에 집중되어 있다. 논문은 image classification 같은 다른 불균형 문제에도 비슷한 logit shift가 있을 수 있다고 말하지만, 실제 검증은 하지 않았다. 따라서 일반 비전 과제로의 외삽은 조심해야 한다.

둘째, 제안의 핵심은 rare class를 아는 것이다. binary segmentation에서는 foreground를 rare class로 정의하기 쉽지만, multi-class 일반 문제에서는 어떤 클래스를 rare로 둘지, $r_j$를 이진으로 둘지 연속값으로 둘지 설계 선택이 필요하다. 논문은 가능성을 언급하지만 체계적 규칙을 제시하지는 않는다.

셋째, 일부 방법은 데이터 특성에 민감하다. asymmetric mixup은 BRATS에서는 강력했지만, intensity overlap이 큰 ATLAS와 KiTS에서는 제한적이었다. 즉 제안 전체가 항상 균일하게 잘 작동하는 것은 아니며, 데이터 modality와 클래스 간 intensity 구조가 중요하다.

넷째, post-processing 의존성이 있다. 논문 스스로도 without post-processing에서는 HD가 나빠질 수 있다고 인정한다. 실제 응용에서 false positive cost가 높은 문제라면 sensitivity 향상만으로 충분하지 않을 수 있다.

다섯째, hyper-parameter tuning 부담이 있다. Supplementary sensitivity analysis를 보면 대부분의 방법이 어느 정도 robust하긴 하지만, large margin과 mixup은 특정 값에서 오히려 baseline보다 나빠질 수 있다. 저자들도 새 응용에는 asymmetric focal loss부터 시도하라고 권고할 정도로, 방법별 tuning 난이도 차이가 존재한다.

비판적으로 보면, 논문의 핵심 통찰인 “foreground logit shift”는 강한 경험적 관찰로 제시되지만, 왜 이런 shift가 수학적으로 inevitable한지에 대한 이론적 정식화는 제한적이다. gradient re-weighting 분석은 일부 직관을 주지만, generalization bound 수준의 엄밀한 설명까지는 아니다. 그럼에도 empirical consistency가 높아, 설계 원리로서의 가치는 충분하다.

## 6. 결론

이 논문은 class imbalance가 심한 segmentation에서 overfitting이 단순한 일반화 저하가 아니라, under-represented class의 test-time logit이 decision boundary 쪽으로 이동하는 비대칭 현상으로 나타난다고 주장한다. 그 결과 모델은 background는 여전히 잘 맞히면서 foreground를 놓치는 방향으로 실패하고, 이는 낮은 sensitivity와 under-segmentation으로 드러난다.

이 관찰을 바탕으로 저자들은 asymmetric large margin loss, asymmetric focal loss, asymmetric adversarial training, asymmetric mixup, asymmetric augmentation을 제안했다. 공통된 철학은 rare class에만 추가 bias를 주어 foreground가 boundary를 넘지 않도록 더 넓은 margin과 더 강한 regularization을 확보하는 것이다. 실험에서는 여러 데이터셋과 아키텍처에서 이 방법들이 기존 symmetric 방법이나 일반적인 imbalance 대응법보다 더 나은 DSC와 sensitivity를 제공했다.

실제 적용 측면에서 이 연구의 의미는 크다. 작은 병변, 작은 종양, 작은 장기처럼 놓치면 안 되는 구조를 다루는 medical segmentation에서 sensitivity 개선은 매우 중요하다. 더 넓게 보면, 이 논문은 imbalance 문제를 단순한 sample count 문제가 아니라 representation과 decision boundary의 비대칭적 불안정성 문제로 보도록 관점을 확장한다. 향후 연구에서는 저자들이 제안하듯 intermediate activation 분석, domain shift, self-supervised learning 같은 더 어려운 setting에서도 이런 logit-based inspection이 유용한 진단 도구가 될 가능성이 있다.
