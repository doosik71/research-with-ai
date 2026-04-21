# Learning Generative Models across Incomparable Spaces

- **저자**: Charlotte Bunne, David Alvarez-Melis, Andreas Krause, Stefanie Jegelka
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1905.05461

## 1. 논문 개요

이 논문은 서로 직접 비교할 수 없는 공간(incomparable spaces) 사이에서 generative model을 학습하는 문제를 다룬다. 기존 GAN이나 Wasserstein 기반 생성 모델은 보통 데이터 분포와 생성 분포가 같은 공간, 혹은 적어도 서로 비교 가능한 공간 위에 정의되어 있어야 한다. 예를 들어 둘 다 같은 차원의 벡터 공간에 있어야 하고, 데이터 샘플과 생성 샘플 사이의 거리나 판별 함수를 직접 계산할 수 있어야 한다. 하지만 실제로는 이런 가정이 맞지 않는 경우가 많다. 어떤 경우에는 데이터의 “절대 위치”보다 샘플들 사이의 관계나 구조만 보존하면 충분하고, 또 어떤 경우에는 원본 데이터가 아예 그래프 형태의 관계 정보로만 주어진다.

논문은 이런 상황을 위해 Gromov-Wasserstein(GW) distance를 생성 모델의 학습 목표로 도입한다. GW distance는 서로 다른 두 공간의 점들을 직접 비교하지 않고, 각 공간 내부의 pairwise distance 구조를 비교한다. 따라서 차원이 다르거나 데이터 타입이 달라도, 샘플들 사이의 구조적 관계만 유지되면 학습이 가능하다. 저자들은 이를 바탕으로 GW GAN이라는 새로운 생성 모델을 제안하고, 동일 공간에서의 일반 생성 문제뿐 아니라 차원 축소, manifold learning, graph-to-Euclidean 생성, 그리고 style control까지 하나의 틀 안에서 다룰 수 있음을 보인다.

이 문제가 중요한 이유는, 생성 모델의 목적이 항상 “원본 데이터와 완전히 똑같은 샘플 생성”은 아니기 때문이다. 어떤 응용에서는 클러스터 구조, manifold 구조, neighborhood 관계처럼 relational structure만 유지하면 충분하며, 대신 스타일, 두께, 회전, 차원 수 같은 표면적 특성은 바꾸고 싶을 수 있다. 이 논문은 바로 그 유연성을 수학적으로 정당화된 거리와 학습 절차로 구현하려는 시도라는 점에서 의미가 크다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 생성 분포와 목표 분포를 “절대 좌표”로 맞추지 않고, 각 공간 내부의 거리 관계를 맞추는 것이다. 기존 GAN류는 생성 샘플과 실제 샘플을 같은 의미의 공간 위에서 비교하려고 한다. 반면 이 논문은 데이터 공간 $X$와 생성 공간 $Y$가 달라도 괜찮다고 본다. 중요한 것은 $X$ 안에서 샘플들끼리 어떤 상대적 구조를 이루는지, 그리고 $Y$ 안에서 생성된 샘플들이 그와 유사한 구조를 이루는지다.

이 구조 비교를 위해 사용되는 것이 GW distance다. 보통 Wasserstein distance는 한 공간에서 다른 공간으로 질량을 옮기는 비용을 계산하려면 두 공간 사이의 pointwise cost가 필요하다. 그런데 GW distance는 각 공간 내부의 거리 행렬만으로 작동하므로, 두 공간이 서로 직접 비교 불가능해도 된다. 이 점이 기존 OT-based GAN과 가장 큰 차이이다.

또 하나의 핵심은 intra-space metric 자체를 adversarially 학습한다는 점이다. 단순히 원래 공간에서의 거리만 쓰면 고차원 문제에서 구분력이 떨어질 수 있기 때문에, 저자들은 adversary $f_\omega$를 통해 각 공간의 샘플을 feature space로 보낸 뒤 그곳에서 Euclidean distance를 계산한다. 즉, discriminator가 “진짜/가짜를 직접 분류”하는 대신, GW loss가 더 잘 작동하도록 구조를 드러내는 metric representation을 학습하는 셈이다.

마지막으로, GW loss는 구조만 고정하고 나머지는 자유롭게 두기 때문에 생성 결과의 스타일이나 전역적 성질을 별도 제약으로 조절할 수 있다. 저자들은 이를 style adversary로 일반화하여, 예를 들어 MNIST 숫자의 구조는 유지하면서 글씨 두께만 굵게 만드는 식의 제어가 가능함을 보인다. 이는 기존 GAN의 style transfer와는 다르게, 구조와 스타일을 목적 함수 수준에서 분리해서 다룬다는 점에서 차별적이다.

## 3. 상세 방법 설명

전체 설정은 다음과 같다. 데이터셋 $\{x_1,\dots,x_n\}$는 분포 $p \in \mathcal{P}(X)$에서 나왔고, 생성기 $g_\theta: Z \to Y$는 noise $z$를 받아 생성 공간 $Y$의 샘플 $y$를 만든다. 여기서 중요한 점은 $X$와 $Y$가 같을 필요가 없다는 것이다.

### Gromov-Wasserstein discrepancy

논문은 먼저 두 분포의 샘플들에 대해 각 공간 내부 거리 행렬을 만든다. 데이터 공간 쪽은 $(D, p)$, 생성 공간 쪽은 $(\bar D, q)$로 두고, GW discrepancy를 다음처럼 정의한다.

$$
\mathrm{GW}(D,\bar D,p,q)
:=
\min_{T \in U_{p,q}}
\sum_{i j k l}
L(D_{ik}, \bar D_{jl}) T_{ij} T_{kl}
$$

여기서 $T$는 두 분포 사이의 coupling이고, $U_{p,q}$는 주변분포가 각각 $p,q$가 되도록 하는 transport plan들의 집합이다. 손실 함수는 논문에서

$$
L(a,b)=\frac{1}{2}|a-b|^2
$$

를 사용한다. 이 식의 의미는 단순하다. 데이터 공간에서 두 점 $x_i,x_k$ 사이 거리가 크면, 생성 공간에서 이에 대응되는 두 점 $y_j,y_l$ 사이 거리도 비슷해야 한다는 것이다. 즉 점 자체를 맞추는 것이 아니라 “거리들의 관계”를 맞춘다.

### GW GAN objective

기본적으로는 mini-batch의 실제 샘플 $X$와 생성 샘플 $Y$에 대해 각 공간의 pairwise distance matrix를 만들고 GW를 최소화하도록 generator를 학습할 수 있다. 그러나 고차원에서는 단순한 원공간 거리만으로는 충분하지 않을 수 있으므로, 논문은 adversary $f_\omega$를 도입한다. 이 함수는 입력을 feature space로 보낸 뒤 그 안에서 거리를 계산한다.

$$
D^\omega_{ij} := \|f_\omega(x_i)-f_\omega(x_j)\|_2
$$

이를 데이터 쪽과 생성 쪽에 각각 적용하면, 최종 minimax objective는 다음과 같다.

$$
\min_\theta \max_{\omega=(\check\omega,\hat\omega)}
\mathrm{GW}(D^{\check\omega}, D^{\hat\omega}, p, q)
$$

여기서 generator는 GW discrepancy를 줄이도록 학습되고, adversary는 두 분포의 구조 차이를 더 잘 드러내는 metric representation을 학습한다. 표준 GAN처럼 real/fake probability를 출력하는 discriminator가 아니라, “거리 구조를 잘 분리하는 feature embedding”을 학습하는 adversary라는 점이 중요하다.

### 학습 절차

논문은 generator와 adversary를 alternating scheme으로 번갈아 학습한다. generator를 adversary보다 더 자주 업데이트하는데, 이는 adversarially learned distance function이 degenerate해지는 것을 막기 위함이라고 설명한다. 알고리즘 1에 따르면 매 반복마다 다음을 수행한다.

1. 실제 데이터 mini-batch $X$와 noise로부터 생성한 $Y=g_\theta(Z)$를 샘플링한다.
2. adversary를 통해 각 mini-batch 내 pairwise distance matrix를 만든다.
3. entropy-regularized, normalized GW loss를 계산한다.
4. adversary 업데이트 시에는 regularization을 포함한 목적을 최대화한다.
5. generator 업데이트 시에는 GW loss를 최소화한다.

저자들은 Appendix A에서 adversary를 고정한 뒤에도 generator가 발산하지 않고 계속 개선된다고 보고한다. 이는 GW objective가 비정상적으로 흔들리는 GAN objective보다 상대적으로 일관된 목적을 준다는 논지와 연결된다.

### Generator 제약과 style adversary

GW loss는 relational structure는 맞추지만, translation, orientation, style 같은 전역적 혹은 표면적 특성은 정하지 않는다. 저자들은 이 자유도를 적극 활용한다.

예를 들어 생성 분포를 원점 주변에 놓고 싶으면 생성 샘플의 norm에 패널티를 줄 수 있다. 이미지 생성에서는 자연스러운 이미지를 만들기 위해 total variation regularization을 추가했다고 밝힌다. 더 일반적으로는 style adversary $c$를 넣어서 스타일 특성을 강제할 수 있다. 이때 목적 함수는 다음과 같이 된다.

$$
\min_\theta \max_{\omega=(\check\omega,\hat\omega)}
\mathrm{GW}(D^{\check\omega}, D^{\hat\omega}, p, q)
-
\lambda \, c(g_\theta(z))
$$

여기서 generator는 구조적 내용은 GW loss를 통해 배우고, 스타일은 $c$를 통해 배우게 된다. 논문에서는 EMNIST 기반 binary classifier를 style adversary로 써서 MNIST 숫자의 두께를 굵게 만드는 실험을 수행했다.

### Adversary regularization

논문에서 중요한 기술적 기여 중 하나는 adversary regularization이다. adversary가 GW objective를 최대화할 때, 단순히 공간을 임의로 늘이거나 왜곡해서 pairwise distance를 부풀릴 수 있다. 이를 막지 않으면 generator가 의미 있는 구조를 배우기 어려워진다.

저자들은 네트워크 전체가 approximately orthogonal operator처럼 동작하도록 regularization을 건다. 기존 방식들은 각 레이어 weight에 대해 orthogonality를 강제하지만, 논문은 네트워크 전체의 input-output behavior를 제약해야 한다고 주장한다. 이를 위해 Orthogonal Procrustes problem 기반 regularizer를 제안한다.

$$
R_\beta(f_\omega(X), X)
:=
\beta \|f_\omega(X) - X P^{*\top}\|_F^2
$$

여기서 $P^*$는 $X$를 $f_\omega(X)$에 가장 가깝게 맞추는 orthogonal matrix이다.

$$
P^* = \arg\min_{P \in O(s)} \|f_\omega(X) - X P^\top\|_F
$$

만약 feature dimension $s$가 입력 차원과 같다면, $f_\omega(X)^\top X$의 SVD로부터 닫힌형 해를 구할 수 있다고 설명한다. 이 regularization의 목적은 adversary가 구조를 드러내는 유용한 feature map은 학습하되, arbitrary stretching으로 GW 값을 인위적으로 키우지 못하게 하는 것이다.

### GW를 loss function으로 쓰기 위한 수정

저자들은 GW를 딥러닝 학습 objective로 쓰기 위해 세 가지 수정을 한다.

첫째, entropy regularization이다. 원래 GW 문제는 quadratic programming 형태라 계산량이 매우 크다. 따라서 다음과 같이 entropy term을 넣는다.

$$
\mathrm{GW}_\varepsilon(D,\bar D,p,q)
=
\min_{T \in U_{p,q}}
E_{D,\bar D}(T) - \varepsilon H(T)
$$

여기서 $H(T)$는 coupling의 entropy이다. 이는 Sinkhorn 기반 최적화를 가능하게 하고, differentiability와 계산 효율을 높인다.

둘째, normalization이다. entropy regularization을 넣으면 같은 공간을 비교해도 값이 0이 아니어서 bias가 생긴다. 이를 줄이기 위해 논문은 normalized GW discrepancy를 다음처럼 정의한다.

$$
\overline{\mathrm{GW}}_\varepsilon(D,\bar D,p,q)
=
2\,\mathrm{GW}_\varepsilon(D,\bar D,p,q)
-
\mathrm{GW}_\varepsilon(D,D,p,p)
-
\mathrm{GW}_\varepsilon(\bar D,\bar D,q,q)
$$

본문 식 번호는 이것도 $\mathrm{GW}_\varepsilon$로 표기되어 있으나, 의미상 정규화된 버전이다.

셋째, numerical stability 개선이다. 저자들은 stabilized Sinkhorn algorithm과 log-domain 업데이트를 사용해 $\varepsilon \to 0$에 가까운 경우의 수치 불안정을 줄인다. 또한 Sinkhorn 반복에는 정규화된 거리 행렬을 쓰되, 최종 loss 계산은 원래 스케일의 거리로 수행하여 수치 안정성과 스케일 정보 보존을 동시에 노린다.

## 4. 실험 및 결과

논문은 GW GAN의 동작 범위를 보여주기 위해 여러 실험을 수행한다.

### 동일 공간에서의 학습

먼저 sanity check로 일반 GAN과 같은 setting, 즉 동일 공간에서의 생성 문제를 다룬다. 2D Gaussian mixture를 대상으로 했을 때, GW GAN은 여러 mode를 안정적으로 복원한다. 저자들은 $\ell_1$ regularization을 추가하면 생성 분포를 원점 주변에 배치할 수 있음을 Figure 2a와 Appendix B에서 보인다. 중요한 관찰은 클러스터 구조는 유지되지만 회전이나 방향은 달라질 수 있다는 점인데, 이는 GW가 절대 좌표보다 관계 구조에 민감하기 때문이다.

또한 MNIST, Fashion-MNIST, gray-scale CIFAR-10에 대해 convolutional generator/adversary를 사용한 이미지 생성도 수행한다. 본문 Figure 2b, 2c, 2d에 따르면 학습이 진행되면서 점차 의미 있는 이미지가 형성된다. 이때 total variation regularization이 이미지 품질 개선에 중요했고, adversary는 orthogonal Procrustes regularization으로 제약되었다. 논문은 Appendix C에서 특히 고차원 이미지에서는 adversary가 없으면 성능이 크게 떨어진다고 보여준다.

Appendix E에서는 OT GAN(Salimans et al., 2018)과 비교도 수행한다. 4개 Gaussian mixture에서는 유사한 수준의 복원이 가능했지만, 5개 mode와 중심 분포가 있는 더 어려운 경우에서는 GW GAN이 더 강건하다고 보고한다.

### 서로 다른 차원 사이의 학습

다음으로 저자들은 2D와 3D Gaussian mixture 사이를 오가는 생성 실험을 수행한다. 여기서는 adversary 없이, 각 공간에서 Euclidean distance를 직접 사용한다. 결과적으로 생성 분포는 목표 분포와 차원은 다르지만, mode 간의 상대적 거리와 전체 구조를 잘 복원했다. 이는 GW GAN이 단순한 density matching이 아니라 relational geometry matching을 수행한다는 주장을 잘 보여준다.

이 실험은 차원 축소나 차원 확장 문제를 생성 모델 관점에서 다룰 수 있음을 시사한다. 다만 논문이 이 실험에서 정량 지표를 표로 제시하지는 않았고, 본문에서는 주로 시각적 결과로 구조 보존을 보여준다.

### 데이터 modality와 manifold가 다른 경우

저자들은 3D S-curve manifold를 2D 공간으로 생성하는 실험도 수행한다. 이때 데이터 공간의 거리로 단순 Euclidean distance가 아니라, $k$-nearest neighbor graph 위 shortest path를 Floyd-Warshall 알고리즘으로 계산해 사용한다. 이 설정은 manifold 상의 geodesic 구조를 반영하기 위한 것이다. Figure 4a에서 생성된 2D 분포는 S-curve의 manifold structure를 복원한 것으로 제시된다.

더 나아가, 입력 데이터가 아예 weighted graph로만 주어진 경우도 다룬다. 절대 좌표 없이 그래프 shortest path만 있는 상황인데, 생성 공간은 2D Euclidean space로 설정한다. Figure 4b에 따르면 GW GAN은 그래프의 neighborhood structure를 대략적으로 보존하는 2D 분포를 학습한다. 이는 기존 GAN류로는 거의 직접 다루기 어려운 설정이다.

### 스타일 제어 실험

논문은 structure와 style의 분리를 보여주기 위해 MNIST 숫자의 “굵기”를 조절하는 실험을 제시한다. EMNIST 문자 데이터에 thin/bold label을 부여해 binary classifier를 학습하고, 이를 style adversary로 사용한다. 학습 초기에는 GW loss만으로 구조를 학습하고, 이후 style adversary를 켜서 generator가 더 굵은 숫자를 출력하도록 유도한다.

Figure 5에 따르면 숫자의 정체성이나 구조는 유지되면서 선 두께가 증가한다. 저자들의 해석은, GW loss가 구조적 content를 유지하게 하고 style adversary가 표면적 특성을 조절한다는 것이다. 이는 논문의 핵심 주장인 “관계 구조와 스타일 특성의 분리”를 보여주는 대표적 실험이다.

### 계산 비용

Appendix F의 Table 1은 MNIST 생성에서 epoch당 평균 학습 시간을 비교한다. Wasserstein GAN with gradient penalty는 약 17.57초/epoch이고, Sinkhorn GAN은 설정에 따라 약 145~154초/epoch, GW GAN은 약 156.62초/epoch이다. 즉, GW GAN은 WGAN-GP보다 훨씬 무겁고 Sinkhorn GAN과 비슷한 수준의 비용을 가진다. 이 점은 확실한 실용적 trade-off다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 생성 모델의 적용 범위를 명확히 확장했다는 점이다. 기존 GAN이나 OT 기반 생성 모델이 사실상 같은 공간 또는 비교 가능한 공간을 전제로 했다면, 이 논문은 GW distance를 통해 “비교 불가능한 공간”으로 문제를 일반화했다. 이 generalization은 단순한 이론적 주장에 그치지 않고, 서로 다른 차원, 그래프 대 유클리드 공간, manifold learning, style shaping 같은 다양한 실험으로 구체화되었다.

또 다른 강점은 GW를 실제 학습 loss로 쓰기 위한 공학적 보완이 잘 정리되어 있다는 점이다. entropy regularization, bias 보정을 위한 normalization, stabilized Sinkhorn, log-domain 계산, 그리고 adversary regularization은 모두 실전 학습을 가능하게 만든 핵심 요소들이다. 특히 orthogonal Procrustes 기반 regularization은 이 논문 안에서도 중요한 독립 기여로 보이며, 단순히 레이어별 orthogonality를 거는 기존 방법보다 네트워크 전체 동작을 제약한다는 점에서 설계 의도가 분명하다.

또한 논문은 GW의 불변성을 “문제”가 아니라 “기회”로 해석한다. 절대 위치나 스타일을 일부러 고정하지 않기 때문에, 원하는 속성을 별도 penalty나 style adversary로 조정할 수 있다는 관점은 생성 모델 설계 측면에서 상당히 유연하다.

반면 한계도 분명하다. 첫째, 실험 결과의 상당 부분이 시각적 정성 평가 중심이다. 구조를 보존했다는 주장은 설득력이 있지만, manifold recovery나 graph structure preservation에 대해 더 정량적인 평가 지표가 있었으면 더 강했을 것이다. 제공된 본문에는 그런 정량 비교가 충분히 상세히 나오지 않는다.

둘째, 계산 비용이 크다. GW 자체가 quadratic 구조를 가지기 때문에 entropy regularization과 Sinkhorn을 써도 여전히 무겁다. Table 1에서도 GW GAN은 WGAN-GP보다 훨씬 느리다. 따라서 대규모 고해상도 생성 문제에 바로 적용하기에는 비용 문제가 있을 수 있다.

셋째, adversary regularization과 추가 제약이 성능의 핵심인데, 이만큼 유연성이 큰 만큼 하이퍼파라미터와 설계 선택의 영향도 클 가능성이 있다. 논문은 $\beta$, $\lambda$, total variation, style adversary 전환 시점 등을 사용하지만, 어떤 설정이 얼마나 민감한지는 본문 기준으로 완전히 체계적으로 분석되지는 않았다.

넷째, 구조 보존과 스타일 변형 사이의 trade-off가 어디까지 가능한지, 그리고 어떤 종류의 스타일 특성이 truly disentangled될 수 있는지는 아직 초기 단계로 보인다. 논문도 digit thickness control 정도의 간단한 예시를 제시했을 뿐, 더 복잡한 자연 이미지 스타일 변경까지 일반화된다고 강하게 주장하지는 않는다.

비판적으로 보면, 이 논문은 “생성 모델”이라기보다 “구조 보존형 분포 학습 프레임워크”에 더 가깝다. 즉, 샘플 품질 자체보다 relational structure transfer에 더 초점이 있다. 이는 강점이기도 하지만, 고품질 이미지 합성 같은 전통적인 GAN 벤치마크와 직접 경쟁하는 방향과는 목적이 다르다. 따라서 이 논문의 가치는 이미지 realism보다는 cross-space structure learning에 있다.

## 6. 결론

이 논문은 Gromov-Wasserstein distance를 기반으로 서로 다른 혹은 직접 비교할 수 없는 공간 사이에서 분포를 학습하는 새로운 생성 모델 GW GAN을 제안했다. 핵심은 샘플 자체의 절대값을 맞추는 대신, 각 공간 내부의 pairwise relation을 맞추는 것이다. 이를 통해 기존 생성 모델이 다루기 어려웠던 차원 불일치, 그래프 기반 입력, manifold learning, 스타일 제어 같은 문제를 하나의 원리로 처리할 수 있음을 보였다.

기술적으로는 adversarial metric learning, orthogonal Procrustes regularization, entropy-regularized and normalized GW objective, stabilized Sinkhorn computation을 결합해 GW를 실제 end-to-end 학습에 사용할 수 있게 만들었다는 점이 중요하다. 실험은 이 방법이 동일 공간 문제에서도 작동하고, 더 나아가 비교 불가능한 공간 간 구조 학습이라는 본래 목표를 실제로 수행할 수 있음을 보여준다.

향후 연구 관점에서 이 논문은 의미가 크다. 생성 모델이 반드시 “같은 표현 공간에서의 복제”일 필요는 없고, 구조를 유지한 채 표현이나 스타일을 바꾸는 방향으로 확장될 수 있음을 보여주기 때문이다. 실제 적용 측면에서는 graph embedding, representation alignment, manifold-aware generation, style-constrained generation 같은 문제에 연결될 가능성이 있다. 다만 계산 비용과 정량 평가의 부족은 후속 연구에서 보완될 필요가 있다.
