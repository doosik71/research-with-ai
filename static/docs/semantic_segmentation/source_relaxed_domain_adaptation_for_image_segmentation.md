# Source-Relaxed Domain Adaptation for Image Segmentation

- **저자**: Mathilde Bateson, Hoel Kervadec, Jose Dolz, Hervé Lombaert, Ismail Ben Ayed
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2005.03697

## 1. 논문 개요

이 논문은 semantic segmentation에서의 **unsupervised domain adaptation (DA)** 문제를 다룬다. 구체적으로는, 라벨이 있는 source domain에서 학습한 segmentation network를 라벨이 없거나 매우 제한적인 supervision만 있는 target domain에 적응시키는 것이 목표다. 기존 DA 방법들은 대개 adaptation 단계에서 source image와 target image를 동시에 접근할 수 있다고 가정하지만, 이 논문은 그 가정을 완화한다. 즉, **adaptation 단계에서 source data 자체가 없어도 작동하는 방법**을 제안한다.

이 문제가 중요한 이유는 특히 medical imaging에서 분명하다. 서로 다른 병원, 장비, 촬영 프로토콜, modality에서 수집된 영상은 분포가 다르며, 이로 인해 source에서 잘 작동하던 모델이 target에서 성능이 크게 떨어진다. 그런데 실제 임상 환경에서는 개인정보 이슈, 데이터 손실, 기관 간 공유 제약 때문에 adaptation 시점에 source image를 다시 불러올 수 없는 경우가 흔하다. 따라서 “source-free” 또는 본 논문의 표현대로 **source-relaxed** adaptation은 실용적 가치가 매우 크다.

논문은 이런 현실적 제약 아래에서, source에서 미리 학습된 모델 파라미터만 초기화에 사용하고, adaptation 단계에서는 target 데이터만으로 학습을 진행한다. 이때 단순 entropy minimization만 사용하면 모델이 모든 픽셀을 하나의 클래스에 몰아넣는 trivial solution으로 붕괴할 수 있으므로, 이를 막기 위해 **class-ratio prior**를 추가한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 간단하다. target domain에서 모델의 예측이 불확실하면 entropy가 높아지므로, **target prediction의 entropy를 직접 줄이면 target에 더 잘 맞는 결정 경계를 유도할 수 있다**는 것이다. 실제로 논문은 source에서 학습된 모델이 source와 유사한 샘플에는 confidence가 높지만, modality가 다른 target에서는 uncertainty가 높아진다고 설명한다.

하지만 entropy만 최소화하면 위험하다. 모델은 가장 쉬운 방식으로 entropy를 줄이기 위해 모든 위치를 하나의 dominant class로 예측할 수 있기 때문이다. 기존 entropy-based DA 방법들은 이를 방지하기 위해 adaptation 중에도 source supervised cross-entropy를 함께 사용한다. 문제는 그렇게 하면 adaptation 단계에서 source image와 label이 모두 필요하다는 점이다.

이 논문은 여기서 차별화된다. source supervised loss 대신, **도메인에 비교적 불변적이라고 볼 수 있는 segmentation region의 prior**를 사용한다. 논문이 실제로 선택한 prior는 각 이미지에서 관심 구조가 차지하는 비율, 즉 **class-ratio prior**다. 이 prior를 source 데이터에서 학습한 auxiliary regression network로 추정하고, 현재 segmentation network가 target image에 대해 출력한 class ratio와 맞추도록 **KL divergence**를 추가 손실로 넣는다. 요약하면, 이 방법은 다음 두 가지를 동시에 강제한다.

첫째, 픽셀 단위 예측은 confident해야 한다.  
둘째, 이미지 전체 수준에서 예측된 영역 비율은 해부학적으로 타당한 class ratio를 따라야 한다.

이 조합이 source data 없이도 entropy minimization의 붕괴를 막는 핵심 장치다.

## 3. 상세 방법 설명

논문은 source domain의 이미지 집합 $\{I_s\}_{s=1}^S$와 target domain의 이미지 집합 $\{I_t\}_{t=1}^T$를 정의한다. 각 source 이미지에는 픽셀 또는 voxel 단위의 정답 segmentation이 있으며, segmentation network는 softmax 확률 벡터를 출력한다.

먼저 source-only pretraining 단계에서는 일반적인 supervised segmentation loss를 사용한다. 논문에서 정의한 source loss는 다음과 같다.

$$
L_s(\theta, \Omega_s)=\frac{1}{|\Omega_s|}\sum_{s=1}^{S}\ell\big(y_s(i),p_s(i,\theta)\big)
$$

여기서 $\ell$은 표준 cross-entropy이고,

$$
\ell(y_s(i),p_s(i,\theta))=-\sum_k y_s^k(i)\log p_s^k(i,\theta)
$$

이다. 즉, 이 단계는 특별할 것이 없는 일반 segmentation training이다. 중요한 점은 adaptation 단계에서 이 supervised loss를 더 이상 사용하지 않는다는 것이다.

adaptation 단계에서는 target image에 대해 모델이 내는 픽셀별 softmax 출력 $p_t(i,\theta)$를 가지고 entropy loss를 계산한다.

$$
\ell_{ent}(p_t(i,\theta))=-\sum_k p_t^k(i,\theta)\log p_t^k(i,\theta)
$$

이 loss는 각 픽셀 예측의 불확실성을 줄인다. softmax가 한 클래스에 가까울수록 entropy가 낮아지므로, 모델은 target에서 더 confident한 예측을 하도록 유도된다.

하지만 앞서 말했듯 이것만 최소화하면 trivial solution이 가능하다. 그래서 논문은 class-ratio prior를 도입한다. target 이미지 $I_t$에서 클래스 $k$의 실제 비율은 원래 다음처럼 정의될 수 있다.

$$
\tau_{GT}(t,k)=\frac{1}{|\Omega_t|}\sum_{i\in \Omega_t} y_t^k(i)
$$

하지만 target에는 정답 마스크가 없으므로 이것을 직접 계산할 수 없다. 대신 논문은 source 데이터에서 auxiliary regression network $R$을 학습해 target의 class ratio를 추정한다. 이 auxiliary network는 source image를 입력받아 해당 이미지의 ground-truth class ratio를 회귀하도록 학습된다. 학습 목표는 squared $L_2$ loss이다.

$$
\min_{\tilde{\theta}} \sum_{s=1}^{S} \Big(R(I_s,\tilde{\theta})-\tau_{GT}(s,k)\Big)^2
$$

이렇게 학습한 $R$을 target image에 적용하면, 각 target 이미지에 대해 추정 class ratio $\tau_e(t,k)$를 얻는다.

한편 segmentation network 자체의 출력으로부터도 class ratio를 근사할 수 있다.

$$
\hat{\tau}(t,k,\theta)=\frac{1}{|\Omega_t|}\sum_{i\in \Omega_t} p_t^k(i,\theta)
$$

이 값은 target 이미지 전체에서 클래스 $k$가 차지하는 평균 확률로 볼 수 있으며, soft segmentation 기준의 class ratio 추정치다.

논문은 $\tau_e(t,k)$와 $\hat{\tau}(t,k,\theta)$를 일치시키기 위해 KL divergence를 사용한다. 최종 adaptation loss는 다음과 같다.

$$
\min_{\theta}\sum_t \sum_{i\in \Omega_t}\ell_{ent}(p_t(i,\theta))+\lambda \, KL\big(\tau_e(t,k),\hat{\tau}(t,k,\theta)\big)
$$

여기서 첫 항은 픽셀 수준에서 confidence를 높이는 역할을 하고, 두 번째 항은 이미지 수준에서 class proportion이 prior와 맞도록 제한한다. $\lambda$는 두 손실의 균형을 조절하는 하이퍼파라미터다.

이 방법의 전체 파이프라인은 다음과 같이 이해할 수 있다. 먼저 source 데이터로 segmentation model을 pretrain한다. 동시에 또는 별도로 source 데이터의 정답 마스크를 이용해 class ratio predictor를 학습한다. 그 다음 adaptation 단계에서는 source image나 source label 없이, target image만 입력하여 entropy loss와 ratio-prior KL loss를 이용해 segmentation network를 업데이트한다. 논문은 이 구조가 adversarial DA처럼 discriminator 등 여러 네트워크를 필요로 하지 않고, **실제 adaptation 자체는 하나의 segmentation network와 간단한 prior term으로 수행된다**는 점을 장점으로 강조한다.

또한 논문은 target 이미지에 구조가 존재하지 않는 경우에 대해 weak supervision도 사용했다. 즉, region of interest가 없는 target 이미지에는 $\tau_e(t,k)=(1,0)$으로 설정해 image-level tag 정보를 반영했다. 이 부분은 완전 무감독이 아니라 약한 supervision이 일부 추가된 설정이라는 점에서 실험 해석 시 중요하다.

## 4. 실험 및 결과

실험은 MICCAI 2018 IVDM3Seg Challenge의 공개 lower-spine multi-modal MRI 데이터셋에서 수행되었다. 총 16개의 수동 주석 3D multi-modal MRI scan이 사용되었고, 13개는 training, 3개는 validation에 사용되었다. 실험은 두 방향의 cross-modality adaptation으로 구성된다. 하나는 **Water (Wat) → In-Phase (IP)**, 다른 하나는 **IP → Wat**이다. segmentation은 binary segmentation ($K=2$)이며, 3D scan을 2D slice 단위로 학습했다. 전처리는 transverse plane rotation 정도만 수행했고, 그 외 특별한 preprocessing은 없었다고 명시한다.

비교 대상은 다음과 같다.  
`NoAdaptation`은 source only training으로, adaptation이 없는 하한선이다.  
`Oracle`은 target 라벨로 직접 학습한 상한선이다.  
`Adversarial`은 [17]의 structured output space adversarial adaptation이다.  
`AdaSource`는 [22] 기반의 방법으로, class-ratio prior를 사용하지만 adaptation 중 source supervised loss도 같이 써서 source data가 필요하다.  
제안 방법은 `AdaEnt`이다.

모든 segmentation 방법에서 backbone은 ENet을 사용했고, Adam optimizer, batch size 12, 100 epochs, 초기 learning rate $1\times 10^{-3}$로 학습했다. adaptation 모델은 모두 source pretraining 100 epoch로 초기화했다. 제안식의 $\lambda$는 경험적으로 $1\times 10^{-2}$로 설정했다. class-ratio prior 추정용 auxiliary network는 ResNeXt101 regression network이며, SGD, learning rate $5\times 10^{-6}$, momentum $0.9$를 사용했다.

평가 지표는 Dice similarity coefficient (DSC)와 Hausdorff distance (HD)다. 결과는 target domain 기준으로 보고된다.

Wat → IP에서 `NoAdaptation`은 DSC 46.7%, HD 2.45로 매우 낮은 성능을 보였다. 이는 modality shift가 심할 때 source-only 모델이 target에서 잘 동작하지 않음을 보여준다. 같은 방향에서 `Adversarial`은 DSC 65.3%, `AdaSource`는 67.0%, 제안한 `AdaEnt`도 67.0%를 달성했다. HD는 각각 1.67, 1.34, 1.33으로 개선되었다. 상한선인 `Oracle`은 DSC 82.3%, HD 1.09였다.

IP → Wat에서는 `NoAdaptation`이 DSC 63.7%, HD 1.44였고, `Adversarial`은 DSC 77.3%, `AdaSource`는 78.3%, `AdaEnt`는 77.8%였다. `Oracle`은 DSC 89.0%, HD 0.90이었다. 여기서도 제안법은 source data를 adaptation 때 전혀 사용하지 않으면서 source를 쓰는 `AdaSource`와 거의 동등한 성능을 보였다.

논문이 특히 강조하는 점은 두 가지다. 첫째, **source data를 더 많이 쓴다고 해서 반드시 adaptation 성능이 더 좋은 것은 아니었다**는 것이다. 실제로 `AdaEnt`는 Wat → IP에서 `AdaSource`와 동일한 DSC를 기록했고, HD는 오히려 약간 더 낮았다. 둘째, `AdaEnt`는 단순히 정확도만 비슷한 것이 아니라, prediction entropy 측면에서 더 자신감 있는 출력을 보였다. 정성적 결과에서 `AdaSource`는 구조 내부나 주변에도 높은 entropy가 남아 있는데, `AdaEnt`는 주로 경계에서만 entropy가 높고 내부는 확신 있는 예측을 보인다. 논문은 심지어 `Oracle`보다도 `AdaEnt`의 entropy map이 더 낮다고 보고하는데, 이는 `AdaEnt`만이 직접 entropy minimization으로 학습되었기 때문이라고 해석한다.

또 하나 중요한 관찰은 adaptation 난이도가 비대칭적이라는 점이다. Wat → IP가 IP → Wat보다 더 어렵고, Oracle 성능도 IP target에서 더 낮다. 논문은 Water modality가 contrast가 더 높아 segmentation 자체가 더 쉬운 조건이라고 설명한다. 따라서 단순 평균 성능만이 아니라 adaptation 방향에 따른 데이터 난이도 차이도 실험 결과 해석에 반영해야 한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정 자체가 현실적이라는 점이다. 많은 DA 논문은 source와 target을 동시에 쥐고 있는 편리한 실험 환경을 전제로 하지만, 이 논문은 실제 medical imaging 배포 환경에서 더 자주 마주치는 제약을 정면으로 다룬다. source data 없이 adaptation하는 설정을 segmentation에 대해 본격적으로 제시했다는 점이 핵심 기여다.

방법도 비교적 단순하고 명확하다. adversarial training처럼 discriminator, domain confusion, generator 등 복잡한 구성 없이, **entropy minimization + prior matching**이라는 해석 가능한 조합으로 설계되어 있다. 손실 함수의 역할이 분명하고, 왜 trivial solution이 생기며 그것을 class-ratio prior가 어떻게 막는지 논리 구조도 납득 가능하다. 또한 source data를 adaptation에 쓰지 않으면서도 source-access 방법과 동급 성능을 보인다는 실험 결과는 실용적으로 매우 설득력이 있다.

또 다른 강점은 정성적 분석이 loss의 성격과 잘 맞아떨어진다는 점이다. 단순히 DSC만 비교하는 것이 아니라 entropy map까지 제시하여, 제안법이 정말로 prediction confidence를 높였음을 시각적으로 보여준다. 이는 이 방법이 “왜” 작동하는지 설명하는 데 도움이 된다.

반면 한계도 분명하다. 첫째, prior의 품질에 상당히 의존한다. 이 논문에서는 class-ratio prior를 auxiliary regression network로 추정하는데, target에서 이 추정이 부정확하면 KL term이 잘못된 방향으로 모델을 끌고 갈 수 있다. 논문도 future work로 modality 간 class-label ratio prior shift, 예를 들어 field of view 차이로 인해 class ratio 자체가 달라지는 경우를 언급한다. 즉, prior가 domain-invariant하다는 가정이 깨지면 성능이 떨어질 수 있다.

둘째, 실험 범위가 제한적이다. lower-spine MRI의 binary segmentation, 그리고 16개 scan이라는 비교적 작은 데이터셋에서 검증되었다. 논문은 “우리 프레임워크는 다양한 segmentation 문제에 사용 가능하다”고 주장하지만, 제공된 본문만 기준으로 보면 다기관 대규모 데이터나 multi-class 복잡한 anatomy에서의 일반화는 아직 직접 입증되지 않았다.

셋째, 완전한 무감독이라고 보기는 어렵다. target에 대해 image-level tag 정보를 사용해 “구조가 없는 이미지”를 구분하고 $\tau_e(t,k)=(1,0)$으로 설정했기 때문이다. 논문은 이 weak supervision을 기존 비교 방법에도 동일하게 주어 공정성을 맞췄지만, 결과 해석에서는 이 정보가 실제로 어느 정도 기여했는지 분리해 보기 어렵다. 본문에는 weak supervision을 제거한 ablation이 제시되지 않는다.

넷째, 비교 대상 구성에서 entropy-based 최신 방법들과의 직접 비교는 제한적이다. 논문은 [19], [20]과의 개념적 차이는 설명하지만, 표의 정량 비교는 `Adversarial`, `AdaSource`, `NoAdaptation`, `Oracle`에 집중되어 있다. 따라서 “source-free entropy minimization”이 다른 비적대적 DA 기법 대비 얼마나 우수한지에 대한 폭넓은 비교는 본문만으로는 충분히 판단하기 어렵다.

## 6. 결론

이 논문은 semantic segmentation을 위한 domain adaptation에서 **adaptation 단계에 source data가 없어도 되는 간단한 프레임워크**를 제안한다. 핵심은 target prediction의 entropy를 최소화하되, trivial collapse를 막기 위해 auxiliary network가 추정한 class-ratio prior와 segmentation output의 class ratio를 KL divergence로 맞추는 것이다.

실험에서는 spine MRI cross-modality adaptation에서 제안법 `AdaEnt`가 source data를 함께 쓰는 `AdaSource`와 거의 같은 성능을 내고, adversarial 방법보다도 우수하거나 비슷한 결과를 보였다. 특히 prediction confidence 측면에서 강한 장점을 보였다. 이는 domain shift가 아주 극단적이지 않은 경우, adaptation에 반드시 source image 접근이 필요한 것은 아닐 수 있음을 시사한다.

실제 적용 측면에서 이 연구는 병원 간 데이터 공유가 어려운 medical imaging 환경에 특히 중요하다. 또한 아이디어 자체가 architecture-agnostic하므로 다른 segmentation backbone에도 쉽게 결합될 가능성이 있다. 향후에는 class-ratio prior가 domain 간에 실제로 얼마나 안정적인지, multi-class 구조나 더 큰 domain gap에서도 유지되는지, 그리고 weak supervision 없이도 같은 효과를 내는지 검증하는 방향이 중요해 보인다.
