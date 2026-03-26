# Domain-Specific Batch Normalization for Unsupervised Domain Adaptation

* **저자**: Woong-Gi Chang, Tackgeun You, Seonguk Seo, Suha Kwak, Bohyung Han
* **발표연도**: 2019
* **arXiv**: [https://arxiv.org/abs/1906.03950](https://arxiv.org/abs/1906.03950)

## 1. 논문 개요

이 논문은 **Unsupervised Domain Adaptation (UDA)** 문제를 다룬다. UDA는 라벨이 충분한 source domain의 지식을, 라벨이 없는 target domain으로 옮겨 target 성능을 높이려는 학습 문제다. 핵심 어려움은 두 도메인 사이의 **domain shift**이다. 예를 들어 source와 target의 이미지 스타일, 조명, 배경, 텍스처가 다르면, source에서 잘 학습된 분류기가 target에서는 쉽게 성능이 무너진다. 논문은 기존 딥러닝 기반 UDA가 대체로 **전체 네트워크를 source와 target이 거의 모두 공유**한다는 점을 문제로 본다. 저자들은 두 도메인 사이에 공통 정보도 있지만, 분명히 도메인별 고유 특성도 존재하므로 둘을 하나의 정규화 통계로 처리하는 것은 비효율적이라고 주장한다.

이를 해결하기 위해 논문은 **Domain-Specific Batch Normalization (DSBN)** 을 제안한다. 아이디어는 단순하지만 강력하다. 네트워크 대부분의 파라미터는 공유하되, **Batch Normalization(BN) 층만 도메인별로 분리**한다. 즉 source용 BN과 target용 BN을 따로 두고, 입력 샘플은 자기 도메인에 맞는 BN branch만 거친다. 저자들은 이 설계가 도메인 고유 통계는 BN이 담당하게 하고, 그 밖의 나머지 파라미터는 보다 잘 공유된 domain-invariant representation을 학습하게 만든다고 본다. 이 DSBN을 기존 UDA 모델에 꽂아 넣고, 이어서 pseudo-label 기반 self-training을 하는 **2단계 학습 프레임워크**를 제안한다. 논문은 이 방법이 VisDA-C, Office-31, Office-Home 등 표준 벤치마크에서 성능을 끌어올리고, multi-source adaptation으로도 자연스럽게 확장된다고 보고한다.

이 연구의 중요성은 두 가지다. 첫째, 기존 UDA 연구가 주로 alignment loss나 adversarial objective 설계에 집중한 반면, 이 논문은 **정규화 계층의 도메인별 분리**라는 매우 실용적인 관점에서 문제를 재구성한다. 둘째, DSBN은 새롭고 거대한 네트워크를 설계하지 않고도, **BN이 들어간 기존 모델에 쉽게 붙일 수 있는 모듈형 기법**이라는 점에서 적용성이 높다. 즉 이 논문은 복잡한 새 loss를 하나 더 제안했다기보다, “무엇을 공유하고 무엇을 분리할 것인가”에 대해 간결하면서도 효과적인 해답을 제시한 것으로 볼 수 있다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 다음과 같다. UDA에서는 source와 target이 완전히 같은 분포가 아니므로, 모든 층에서 같은 통계와 같은 affine transformation을 쓰는 것은 오히려 표현 학습을 방해할 수 있다. 특히 BN은 미니배치의 평균과 분산을 이용해 feature를 정규화하고, 이어서 학습 가능한 scale $\gamma$와 shift $\beta$를 적용한다. 그런데 source와 target의 활성값 분포가 다르면, 하나의 BN 통계로 둘을 모두 정규화하는 것은 도메인 차이를 억지로 한데 섞는 꼴이 된다. 논문은 이 지점이 domain adaptation에서 충분히 활용되지 않았다고 본다.

그래서 저자들은 **도메인별 BN만 분리하고 나머지는 공유**하는 구조를 택한다. 이 선택은 매우 절제되어 있다. 완전히 별도 네트워크를 두는 것도 아니고, 반대로 모든 것을 공유하는 것도 아니다. 저자들의 관점에서는 BN의 running mean, running variance, affine parameter가 도메인 고유 성질을 담기에 적절한 저장소 역할을 한다. 그러면 convolution, classifier, feature extractor의 나머지 파라미터는 두 도메인에 공통인 정보를 배우는 데 집중할 수 있다. 이 때문에 DSBN은 “도메인 특화 정보는 BN에, 도메인 불변 정보는 공유 trunk에”라는 분업 구조를 만든다.

두 번째 핵심 아이디어는 **2-stage 학습**이다. 첫 단계에서는 기존 UDA 기법(MSTN, CPUA 등)에 DSBN을 삽입해 target에 대한 초기 pseudo-label을 만든다. 두 번째 단계에서는 source의 진짜 라벨과 target의 pseudo-label을 모두 사용해, source와 target 각각의 분류기를 DSBN과 함께 학습한다. 여기서 중요한 점은 target pseudo-label을 고정하지 않고, 학습 중간에 점진적으로 갱신한다는 것이다. 초반에는 1단계 모델의 예측을 더 신뢰하고, 후반으로 갈수록 2단계 모델의 예측 비중을 키운다. 즉 논문은 DSBN만 단독으로 제안하는 것이 아니라, **DSBN이 pseudo-label self-training을 더 안정적으로 만들어 준다**는 맥락까지 함께 제시한다.

기존 접근과의 차별점은 명확하다. DANN, JAN, CDAN, MSTN, CPUA 같은 다수의 방법은 주로 alignment loss, adversarial confusion, semantic matching, class uncertainty alignment 등에 집중하지만, 이 논문은 **정규화 메커니즘 자체를 도메인 적응의 핵심 구성요소로 본다**. 또한 AdaBN처럼 target 통계로 BN을 재추정하는 방식과 달리, DSBN은 단순한 사후처리가 아니라 **훈련 단계 전체에서 source/target별 BN branch를 유지**한다는 점에서 더 적극적이다. 논문은 이를 통해 더 안정적인 pseudo-label과 더 나은 domain-invariant representation을 얻는다고 주장한다.

## 3. 상세 방법 설명

### 3.1 문제 설정과 전체 파이프라인

입력은 라벨이 있는 source dataset $\mathcal{X}_S$와 라벨이 없는 target dataset $\mathcal{X}_T$이다. 목표는 source supervision을 활용해 target의 분류 정확도를 높이는 것이다. 논문은 이 목표를 위해 두 단계 파이프라인을 제안한다.

첫 번째 단계에서는 MSTN 또는 CPUA 같은 기존 UDA 모델에 DSBN을 삽입해 학습한다. 이 모델은 source와 target을 정렬하면서, target 샘플에 대한 초기 pseudo-label을 만든다. 이 초기 pseudo-labeler를 논문에서는 $F_T^1$로 표기한다.

두 번째 단계에서는 DSBN이 들어간 분류 네트워크를 다시 학습한다. 이때 source 데이터는 정답 라벨로, target 데이터는 pseudo-label로 supervised classification을 수행한다. 최종적으로 source용 모델 $F_S^2$와 target용 모델 $F_T^2$를 얻는다. 이 단계는 사실상 multi-task classification처럼 동작하며, target pseudo-label을 점진적으로 개선하는 self-training 절차를 포함한다.

### 3.2 Batch Normalization과 DSBN의 차이

일반 BN은 채널별 activation $\mathbf{x}\in\mathbb{R}^{H\times W\times N}$에 대해 미니배치 평균 $\mu$와 분산 $\sigma^2$를 계산하고, 이를 이용해 정규화한 뒤 affine transform을 적용한다.

$$
\text{BN}(\mathbf{x}[i,j,n];\gamma,\beta)=\gamma \cdot \hat{\mathbf{x}}[i,j,n]+\beta
$$

여기서 정규화된 activation은

$$
\hat{\mathbf{x}}[i,j,n]=\frac{\mathbf{x}[i,j,n]-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

이며, 평균과 분산은 미니배치 전체에서 계산된다.

$$
\begin{aligned}
\mu &=\frac{\sum_n \sum_{i,j}\mathbf{x}[i,j,n]}{N\cdot H\cdot W} \\
\sigma^2 &=\frac{\sum_n \sum_{i,j}(\mathbf{x}[i,j,n]-\mu)^2}{N\cdot H\cdot W}
\end{aligned}
$$

훈련 중에는 running mean과 running variance를 exponential moving average로 누적하고, 테스트 시에는 이 누적 통계를 사용한다. 문제는 source와 target을 섞어 하나의 BN 통계를 쓰면, domain shift가 큰 경우 이 통계가 양쪽을 모두 잘 대표하지 못할 수 있다는 점이다.

DSBN은 이 BN을 도메인별로 분리한 것이다. 도메인 $d\in{S,T}$에 대해 별도의 affine parameter $\gamma_d,\beta_d$와 별도의 평균 $\mu_d$, 분산 $\sigma_d^2$를 둔다. 그러면 DSBN은 다음과 같이 표현된다.

$$
\begin{aligned}
\text{DSBN}_d(\mathbf{x}_d[i,j,n];\gamma_d,\beta_d) &= \gamma_d\cdot \hat{\mathbf{x}}_d[i,j,n]+\beta_d \\
\hat{\mathbf{x}}_d[i,j,n] &= \frac{\mathbf{x}_d[i,j,n]-\mu_d}{\sqrt{\sigma_d^2+\epsilon}} \\
\mu_d &= \frac{\sum_n \sum_{i,j}\mathbf{x}_d[i,j,n]}{N\cdot H\cdot W} \\
\sigma_d^2 &= \frac{\sum_n \sum_{i,j}(\mathbf{x}_d[i,j,n]-\mu_d)^2}{N\cdot H\cdot W}
\end{aligned}
$$

즉 source 샘플은 source BN branch만, target 샘플은 target BN branch만 통과한다. running mean과 running variance 역시 도메인별로 따로 누적한다. 이 구조의 효과는, 네트워크가 도메인 고유 통계를 BN에 저장하면서도 나머지 공유 파라미터로 공통 표현을 배울 수 있게 한다는 데 있다. 논문은 기존 분류 네트워크 $F(\cdot)$의 모든 BN을 DSBN으로 바꾼 네트워크를 $F_d(\cdot)$로 표기한다.

### 3.3 Stage 1: 초기 pseudo-label 생성

1단계는 기존 UDA 모델에 DSBN을 결합해 초기 pseudo-labeler를 학습하는 과정이다. 논문은 구체적으로 MSTN과 CPUA 두 방법을 backbone으로 사용했다.

MSTN의 전체 loss는 다음과 같다.

$$
\mathcal{L}=\mathcal{L}_{\text{cls}}(\mathcal{X}_S)+\lambda \mathcal{L}_{\text{da}}(\mathcal{X}_S,\mathcal{X}_T)+\lambda \mathcal{L}_{\text{sm}}(\mathcal{X}_S,\mathcal{X}_T)
$$

여기서 $\mathcal{L}_{\text{cls}}$는 source classification loss, $\mathcal{L}_{\text{da}}$는 domain adversarial loss, $\mathcal{L}_{\text{sm}}$은 source와 target의 같은 클래스 centroid를 맞추는 semantic matching loss이다. MSTN은 target pseudo-label을 이용해 semantic matching을 수행하므로, 초기 label quality가 중요하다. 논문은 DSBN이 이 부분을 개선해 준다고 본다.

CPUA는 class probability alignment에 초점을 맞춘 간단한 방법이다. source와 pseudo-labeled target의 클래스 prior를 이용해 class weight를 정의한다. source의 클래스 비율을 $p_S(c)$, pseudo-labeled target의 비율을 $\widetilde{p}_T(c)$라 하면 가중치는

$$
w_S(x,y)=\frac{\max_{y'}p_S(y')}{p_S(y)}
$$

$$
w_T(x)=\frac{\max_{y'}\widetilde{p}_T(y')}{\widetilde{p}_T(\widetilde{y}(x))}
$$

로 주어진다. 전체 loss는

$$
\mathcal{L}=\mathcal{L}_{\text{cls}}(\mathcal{X}_S)+\lambda \mathcal{L}_{\text{da}}(\mathcal{X}_S,\mathcal{X}_T)
$$

이며, 여기서 classification과 domain adversarial loss에 각 도메인별 class weight를 적용한다. 논문은 CPUA에 DSBN을 붙여 1단계 pseudo-labeler를 학습한다.

### 3.4 Stage 2: pseudo-label 기반 self-training

2단계는 훨씬 직관적이다. source는 정답 라벨, target은 pseudo-label을 이용해 classification loss만으로 학습한다. 전체 loss는

$$
\mathcal{L}=\mathcal{L}_{\text{cls}}(\mathcal{X}_S)+\mathcal{L}_{\text{cls}}^{\text{pseudo}}(\mathcal{X}_T)
$$

이다. 각 항은

$$
\mathcal{L}_{\text{cls}}(\mathcal{X}_S)=\sum_{(x,y)\in\mathcal{X}_S}\ell(F_S^2(x),y)
$$

$$
\mathcal{L}_{\text{cls}}^{\text{pseudo}}(\mathcal{X}_T)=\sum_{x\in\mathcal{X}_T}\ell(F_T^2(x),y')
$$

로 정의된다. 즉 2단계는 adversarial alignment나 semantic matching이 중심이 아니라, 더 좋은 pseudo-label을 바탕으로 **도메인별 분류 모델을 명시적으로 supervised 학습**하는 단계다. DSBN이 있으므로 source와 target은 같은 backbone을 공유하면서도, BN 수준에서 도메인 특화 적응을 유지할 수 있다.

핵심은 pseudo-label 생성 방식이다. target 샘플 $x$의 pseudo-label $y'$는 1단계 모델 $F_T^1$과 2단계 현재 모델 $F_T^2$의 예측을 혼합해 만든다.

$$
y'=\operatorname*{\arg\max}_{c\in C}\left\{(1-\lambda)F_T^1(x)[c]+\lambda F_T^2(x)[c]\right\}
$$

훈련 초반에는 $\lambda$가 작아서 $F_T^1$의 예측을 많이 믿고, 훈련이 진행될수록 $\lambda$가 커져 $F_T^2$의 예측을 더 많이 반영한다. 논문은 $\lambda$를

$$
\lambda=\frac{2}{1+\exp(-\gamma\cdot p)}-1,\quad \gamma=10
$$

으로 증가시킨다. 여기서 $p$는 0에서 1로 선형 증가하는 training progress다. 이 설계는 noisy pseudo-label의 악영향을 완화하기 위한 것이다. 초반의 불안정한 $F_T^2$ 예측이 전체 pseudo-label을 망치지 않도록 하고, 후반에는 더 강해진 $F_T^2$가 점차 자기 자신을 개선하도록 만든다. 논문은 이 2단계를 반복 수행하면 성능이 계속 올라간다고 보고한다.

### 3.5 Multi-source adaptation 확장

DSBN의 또 다른 장점은 multi-source 확장이 매우 자연스럽다는 점이다. source domain이 여러 개면 BN branch 수도 그 수만큼 늘리면 된다. 논문은 source domain 집합을 $\mathcal{D}_S={\mathcal{X}_{S_1},\mathcal{X}_{S_2},\dots}$로 두고, 전체 loss를 각 source의 classification loss와 alignment loss 평균으로 정의한다.

$$
\mathcal{L}=\frac{1}{|\mathcal{D}_S|}\sum_i^{|\mathcal{D}_S|}\Big(\mathcal{L}_{\text{cls}}(\mathcal{X}_{S_i})+\mathcal{L}_{\text{align}}(\mathcal{X}_{S_i},\mathcal{X}_T)\Big)
$$

핵심은 새 source마다 별도 BN branch를 추가할 수 있다는 점이다. 따라서 여러 source를 그냥 하나로 합치는 merged setting뿐 아니라, 각 source를 따로 취급하는 separate setting도 쉽게 구현된다. 논문은 특히 어려운 target task에서 separate DSBN이 merged보다 더 좋은 경우가 있다고 보고한다.

## 4. 실험 및 결과

### 4.1 데이터셋과 설정

논문은 세 가지 대표 벤치마크를 사용한다. **VisDA-C**는 synthetic-to-real 적응 문제로, 12개 클래스에 대해 152,409장의 synthetic 이미지와 55,400장의 real 이미지를 포함한다. **Office-31**은 Amazon(A), Webcam(W), DSLR(D)의 3개 도메인, 31개 클래스의 고전적인 domain adaptation 벤치마크다. **Office-Home**은 Art, Clipart, Product, Real-World의 4개 도메인과 65개 클래스의 더 어려운 벤치마크다. 평가 프로토콜은 fully transductive protocol을 따른다. backbone은 VisDA-C에서 ResNet-101, Office-31과 Office-Home에서는 ResNet-50을 사용하며, 모두 ImageNet pretrained model이다. 배치 크기는 40, optimizer는 Adam, stage 1과 stage 2의 초기 learning rate는 각각 $1.0\times10^{-4}$와 $5.0\times10^{-5}$, 최대 iteration은 50,000이다. 논문은 BN과 DSBN의 차이를 공정하게 비교하기 위해 도메인별 mini-batch를 따로 구성해 forward한다고 설명한다.

### 4.2 VisDA-C 결과

VisDA-C에서 결과는 매우 인상적이다. MSTN reproduced baseline은 평균 65.0%, CPUA reproduced baseline은 66.6%인데, 여기에 DSBN을 적용한 **Stage 1 only**는 각각 72.3%, 71.9%로 오른다. 즉 DSBN만 삽입해도 이미 큰 폭의 개선이 있다. 여기에 **Stage 1 and 2** 전체 프레임워크를 적용하면 MSTN 기반은 80.2%, CPUA 기반은 76.2%까지 상승한다. 특히 MSTN+DSBN의 80.2%는 표에 제시된 다른 기존 방법들(DAN, DANN, MCD, ADR 등)보다 높다. 논문은 knife, person, skate, truck 같은 어려운 클래스에서도 개선이 크다고 강조한다. 실제로 MSTN 기반에서 knife는 16.6에서 75.1, skate는 40.4에서 68.9, truck은 18.5에서 45.5로 크게 오른다. 다만 CPUA 기반에서는 knife가 18.7에서 Stage 1 only 37.9로 오르다가 Stage 1 and 2에서 20.6으로 다시 내려가는 등, 모든 클래스가 균등하게 좋아지는 것은 아니다. 평균적으로는 확실히 개선되지만, class별 변동성은 존재한다는 점을 읽을 수 있다.

### 4.3 Office-31 결과

Office-31에서도 DSBN은 일관된 이득을 보인다. MSTN reproduced는 평균 86.5%, CPUA reproduced는 86.4%인데, DSBN을 넣은 Stage 1 and 2는 둘 다 88.3%를 달성한다. 이는 표에 포함된 CDAN-M 87.7, iCAN 87.2 등을 넘어서는 수치다. 개별 task를 보면 A→W, W→A, A→D, D→A 등 대부분에서 baseline보다 낫다. 예를 들어 CPUA 기반에서는 A→W가 90.1에서 93.3으로, A→D가 86.8에서 90.8로 향상된다. 이 결과는 DSBN이 대형 synthetic-to-real 문제뿐 아니라, 상대적으로 작은 규모의 office benchmark에도 잘 작동함을 보여준다.

### 4.4 Multi-source 결과

Office-31 multi-source 실험에서는 single, merged, separate 세 가지 설정을 비교한다. 흥미로운 점은 쉬운 target task에서는 BN과 DSBN 차이가 작을 수 있지만, 어려운 task에서는 DSBN의 이점이 커진다는 것이다. 예를 들어 target이 A인 어려운 설정에서 merged BN은 71.3, merged DSBN은 73.2이며, separate BN은 69.9인데 separate DSBN은 75.6으로 크게 오른다. 이는 source들을 단순히 섞는 것보다, **각 source를 별도 도메인으로 다루며 BN branch도 따로 유지하는 방식**이 효과적일 수 있음을 시사한다. Office-Home에서도 merged BN 81.2 대비 merged DSBN 82.3, separate BN 81.4 대비 separate DSBN 83.0으로 개선이 보고된다. 이 결과는 DSBN의 구조적 장점이 single-source에만 한정되지 않음을 보여준다.

### 4.5 Ablation과 iterative learning

Ablation study는 이 논문의 중요한 설득 포인트다. VisDA-C에서 Stage 1과 Stage 2 각각에 BN 또는 DSBN을 넣는 조합을 비교한 결과, **Stage 2에서 DSBN이 특히 중요**하다는 결론이 드러난다. 예를 들어 MSTN baseline에서 BN→BN 조합은 Stage 2 후 평균 63.4로 오히려 Stage 1 대비 $-1.6$ 하락한다. 반면 BN→DSBN은 73.2로 $+7.2$, DSBN→DSBN은 80.2로 $+7.9$ 상승한다. CPUA에서도 BN→BN은 $-2.8$, BN→DSBN은 $+4.7$, DSBN→DSBN은 $+4.3$이다. 즉 단순한 self-training 자체보다, **도메인별 BN을 유지한 상태의 self-training**이 pseudo-label 활용을 성공적으로 만든다는 해석이 가능하다.

또한 iterative learning 실험에서 MSTN 기반 VisDA-C의 정확도는 Stage 1의 72.3에서 Stage 2 Iter 1: 80.2, Iter 2: 81.4, Iter 3: 82.2, Iter 4: 82.7로 계속 오른다. 즉 2단계 self-training을 한 번으로 끝내지 않고 반복 수행하면 추가 개선이 가능하다. 논문은 이것이 더 강해진 $F_T^2$가 다음 반복의 더 좋은 pseudo-labeler가 되기 때문이라고 해석한다. feature visualization에서도 DSBN 쪽이 BN보다 source와 target의 같은 클래스 샘플들이 더 잘 정렬된다고 주장한다. 다만 이 부분은 정성적 결과이므로, 시각화가 정량 성능을 완전히 대체하는 근거는 아니다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **아이디어의 단순성과 범용성**이다. DSBN은 기존 UDA 모델 전체를 다시 설계하지 않고, BN 레이어만 도메인별로 나누는 방식이다. 따라서 BN이 있는 CNN 계열 모델이라면 적용이 쉽고, 실제로 MSTN과 CPUA 두 서로 다른 baseline에 붙여 꾸준한 개선을 보였다. 즉 특정 loss 설계에 과도하게 의존하는 기법이 아니라, 구조적으로 넓은 적용 가능성을 가진다. 또 단순한 Stage 1 성능 향상뿐 아니라, Stage 2 self-training의 성공 여부까지 좌우한다는 점에서 실질적 영향력이 크다. Ablation 결과는 “DSBN이 있어야 pseudo-label refinement가 잘 작동한다”는 주장을 꽤 설득력 있게 뒷받침한다. multi-source 확장 역시 branch 수만 늘리면 된다는 점에서 깔끔하다.

또 다른 강점은 **무엇을 공유하고 무엇을 분리할지에 대한 설계 철학이 명확**하다는 점이다. 많은 domain adaptation 연구가 alignment objective를 계속 복잡하게 만들었다면, 이 논문은 BN이 실제로 distribution shift를 직접 다루는 계층이라는 사실에 주목했다. 도메인별 running statistics와 affine parameter만 분리해도 큰 이득이 있다는 결과는, representation learning에서 normalization이 얼마나 중요한지 다시 보여준다. 특히 VisDA-C처럼 shift가 큰 문제에서 이득이 큰 것은 이 설계가 실제 문제 구조와 잘 맞아떨어진다는 증거로 보인다.

한계도 분명하다. 첫째, 이 방법은 **BN이 있는 네트워크를 전제로 한다**. 논문 시점에서는 CNN+BN이 자연스럽지만, normalization이 다르거나 BN이 핵심이 아닌 구조에는 그대로 적용하기 어렵다. 둘째, 1단계 pseudo-labeler 품질에 어느 정도 의존한다. 논문은 DSBN이 pseudo-label을 더 신뢰성 있게 만든다고 주장하지만, 초기 pseudo-label이 지나치게 나쁘면 2단계도 흔들릴 수 있다. 셋째, DSBN이 도메인 차이를 BN 통계와 affine parameter에 주로 맡긴다는 가정이 항상 충분한지는 남는다. 어떤 domain shift는 low-level statistics뿐 아니라 더 깊은 semantic discrepancy에서 발생할 수 있으므로, BN 분리만으로 완전히 해결되지는 않을 수 있다.

또한 결과를 자세히 보면 모든 클래스가 항상 좋아지는 것은 아니다. 예를 들어 CPUA 기반 VisDA-C에서 knife 클래스는 Stage 1에서 개선되지만 Stage 2에서는 다시 내려간다. 이는 self-training이 일부 클래스에서 여전히 noisy label의 영향을 받을 수 있음을 보여준다. 그리고 논문은 성능 향상을 충분히 보여주지만, **왜 BN 파라미터만으로 domain-specific information이 충분히 표현되는지에 대한 이론적 분석은 제한적**이다. 즉 실험적 유효성은 강하지만, 표현 분해가 실제로 어떤 층에서 어떻게 이루어지는지에 대한 deeper analysis는 더 있었으면 좋았을 것이다. 마지막으로 논문은 두 baseline(MSTN, CPUA)에 대해서는 폭넓게 보이지만, 더 다양한 architecture나 non-BN 기반 설정에 대한 검증은 제공하지 않는다. 이 점은 후속 연구 과제로 남는다.

## 6. 결론

이 논문은 UDA에서 source와 target이 모든 파라미터를 공유해야 한다는 관행에 의문을 제기하고, **Batch Normalization만 도메인별로 분리하는 DSBN**이라는 간결한 대안을 제시했다. DSBN은 source와 target이 별도의 BN 통계와 affine parameter를 가지도록 하여 도메인 고유 정보는 BN branch에 담고, 그 외의 대부분 파라미터는 공유하여 domain-invariant representation을 학습하게 만든다. 여기에 pseudo-label 생성과 self-training을 결합한 2단계 학습 전략을 더함으로써, VisDA-C와 Office-31 등에서 강력한 성능 향상을 달성했다. 특히 ablation은 DSBN이 단순한 보조 요소가 아니라, self-training을 성공시키는 핵심 장치임을 보여준다.

실제 적용 측면에서 이 연구의 의미는 크다. 새로운 거대 모델이나 복잡한 training objective를 도입하지 않고도, 기존 BN 기반 domain adaptation 모델의 성능을 상당히 높일 수 있기 때문이다. 또한 multi-source domain adaptation으로의 확장이 자연스러워, 서로 다른 여러 source를 다뤄야 하는 현실적 상황에도 유용하다. 후속 연구 관점에서는 DSBN의 아이디어를 다른 normalization 구조, 더 현대적인 backbone, 혹은 semi-supervised / test-time adaptation setting으로 확장해 볼 여지가 크다. 정리하면, 이 논문은 UDA에서 “도메인 특화와 도메인 공유의 경계”를 BN 수준에서 정교하게 설계하는 것이 매우 중요하다는 점을 명확하게 보여준 작업이다. 분석은 사용자가 제공한 논문 추출 텍스트를 바탕으로 작성했다.
