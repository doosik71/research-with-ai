# Semi-supervised Medical Image Segmentation through Dual-task Consistency

- **저자**: Xiangde Luo, Jieneng Chen, Tao Song, Yinan Chen, Guotai Wang, Shaoting Zhang
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2009.04448

## 1. 논문 개요

이 논문은 의료영상 분할에서 매우 적은 수의 라벨 데이터와 많은 비라벨 데이터를 함께 사용할 수 있는 semi-supervised learning(SSL) 방법을 제안한다. 핵심 목표는 비라벨 데이터를 더 효과적으로 활용하면서도, 기존 방법보다 단순하고 계산 비용이 낮은 방식으로 segmentation 성능을 높이는 것이다.

문제가 되는 배경은 분명하다. 의료영상, 특히 3D MRI나 CT의 정확한 segmentation annotation은 숙련된 전문가의 시간이 많이 들기 때문에 대규모 fully labeled dataset을 구축하기가 어렵다. 따라서 적은 수의 정답 마스크와 많은 비라벨 영상을 함께 사용하는 semi-supervised segmentation이 실제 임상 환경에서 매우 중요하다.

기존 SSL 기반 의료영상 분할 방법의 다수는 입력 데이터에 perturbation을 주거나, teacher-student 구조 또는 adversarial regularization을 사용해 “같은 데이터의 예측이 서로 비슷해야 한다”는 data-level consistency를 강제한다. 이 논문은 여기서 한 걸음 나아가, 같은 입력에서 서로 다른 task branch가 만든 예측끼리도 일관성을 가져야 한다는 task-level consistency를 직접 구성한다. 저자들은 이것이 unlabeled data를 활용하는 새로운 regularization 방식이 될 수 있다고 주장한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 segmentation을 하나의 표현으로만 보지 않고, 서로 다른 두 표현으로 동시에 다루는 것이다. 하나는 일반적인 pixel-wise classification map이고, 다른 하나는 물체의 경계와 형상 정보를 반영하는 level set function(LSF) representation이다. 즉, 같은 segmentation target을 두 개의 서로 다른 task로 예측하게 만든다.

이 설계의 직관은 다음과 같다. pixel-wise segmentation branch는 voxel 단위 분류에 강하고, level set branch는 보다 전역적인 shape 및 geometry 정보를 담을 수 있다. 두 branch는 같은 대상을 보고 있지만 서로 다른 관점으로 예측하므로, 자연스럽게 prediction perturbation이 생긴다. 저자들은 이 차이를 오히려 regularization 자원으로 사용한다. 두 branch의 출력이 적절한 공통 공간으로 매핑될 수 있다면, labeled data뿐 아니라 unlabeled data에서도 두 출력이 서로 일치하도록 학습시킬 수 있다는 것이다.

기존 접근과의 차별점은 “입력 perturbation 기반 consistency”가 아니라 “task 간 consistency”를 직접 학습에 넣는다는 데 있다. 논문은 이를 dual-task consistency(DTC)라고 부른다. 또한 다른 consistency 기반 SSL과 달리 여러 번의 forward pass나 teacher network를 필요로 하지 않아, 구조가 비교적 단순하고 계산 비용도 낮다.

## 3. 상세 방법 설명

전체 프레임워크는 shared encoder-decoder backbone 위에 두 개의 head를 얹은 dual-task network로 구성된다. backbone은 VNet이며, 첫 번째 head는 segmentation probability map을 예측하는 classification branch이고, 두 번째 head는 level set function을 회귀하는 regression branch이다. 입력은 3D 의료영상이고, 출력은 동시에 segmentation map과 LSF map이다.

이 논문에서 segmentation mask를 LSF로 바꾸는 변환 $T(x)$는 다음 의미를 가진다. 어떤 voxel이 object 내부에 있으면 boundary까지의 거리를 음수로, 경계 위면 0으로, object 외부면 양수로 둔다. 수식은 다음과 같다.

$$
T(x)=
\begin{cases}
-\inf_{y \in \partial S} \|x-y\|_2, & x \in S_{in} \\
0, & x \in \partial S \\
+\inf_{y \in \partial S} \|x-y\|_2, & x \in S_{out}
\end{cases}
$$

여기서 $\partial S$는 zero level set이자 segmentation contour이고, $S_{in}$과 $S_{out}$은 각각 대상 내부와 외부 영역이다. 즉, 이 표현은 단순 binary mask보다 더 풍부한 geometry 정보를 담는다.

문제는 LSF branch의 출력을 segmentation 공간으로 다시 옮겨야 task consistency를 계산할 수 있다는 점이다. 원래의 정확한 inverse transform은 미분 가능하지 않아 학습에 바로 쓰기 어렵다. 그래서 저자들은 smooth Heaviside approximation을 사용한다. 구체적으로 다음과 같은 변환층 $T^{-1}(z)$를 둔다.

$$
T^{-1}(z)=\frac{1}{1+e^{-kz}}=\sigma(kz)
$$

여기서 $z$는 어떤 voxel의 LSF 값이고, $k$는 매우 큰 상수다. 이 함수는 사실상 scaled sigmoid이므로, LSF 내부/외부를 1과 0에 가깝게 바꾸는 differentiable approximation 역할을 한다. 논문에서는 $k=1500$을 사용했다. 도함수는 다음과 같이 주어지며, 따라서 backpropagation이 가능하다.

$$
\frac{\partial T^{-1}}{dz}
=
\left(\frac{1}{1+e^{-kz}}\right)'
=
k \cdot \frac{1}{1+e^{-kz}} \cdot \left(1-\frac{1}{1+e^{-kz}}\right)
$$

이제 두 branch의 출력을 같은 공간에서 비교할 수 있다. segmentation branch의 출력 $f_1(x_i)$와, LSF branch 출력 $f_2(x_i)$를 변환층에 통과시킨 $T^{-1}(f_2(x_i))$ 사이의 차이를 dual-task consistency loss로 둔다.

$$
L_{DTC}(x)
=
\sum_{x_i \in D}
\left\|
f_1(x_i)-T^{-1}(f_2(x_i))
\right\|^2
=
\sum_{x_i \in D}
\left\|
f_1(x_i)-\sigma(k f_2(x_i))
\right\|^2
$$

이 손실은 labeled data와 unlabeled data 모두에 적용된다. 이것이 논문의 핵심이다. 정답이 없는 unlabeled image라도, 두 task의 예측이 서로 일관되도록 강제함으로써 학습 신호를 얻는다.

labeled data에 대해서는 두 개의 supervised loss도 함께 사용한다. segmentation branch에는 Dice loss를 쓴다.

$$
L_{Seg}(x,y)=\sum_{x_i,y_i \in D_l} L_{Dice}(x_i,y_i)
$$

논문은 이를 각 3D volume 내부 voxel 합으로 계산되는 일반적인 Dice loss 형태로 적고 있다. 직관적으로는 예측 segmentation과 ground truth mask의 overlap을 최대화하는 목적이다.

LSF branch에는 예측된 level set map과 ground truth segmentation mask를 $T(y)$로 변환한 결과 사이의 $L_2$ loss를 사용한다.

$$
L_{LSF}(x,y)=\sum_{x_i,y_i \in D_l} \|f_2(x_i)-T(y_i)\|^2
$$

최종 loss는 다음과 같다.

$$
L_{total}=L_{Seg}+L_{LSF}+\lambda_d L_{DTC}
$$

여기서 $L_{Seg}$와 $L_{LSF}$는 labeled data에만 사용되고, $L_{DTC}$는 labeled와 unlabeled 양쪽 모두에 사용된다. 따라서 labeled data는 직접 supervision을 제공하고, unlabeled data는 task consistency를 통해 간접 supervision을 제공한다.

$\lambda_d$는 training step $t$에 따라 점진적으로 증가하는 Gaussian warm-up 함수로 설정된다.

$$
\lambda_d(t)=e^{-5\left(1-\frac{t}{t_{max}}\right)^2}
$$

초기 학습에서 consistency loss가 너무 강하게 작용하지 않도록 하고, 학습이 진행될수록 더 큰 비중을 주려는 설계다.

알고리즘 흐름은 비교적 단순하다. 각 iteration에서 labeled batch와 unlabeled batch를 함께 샘플링하고, labeled batch에 대해서는 ground truth LSF를 미리 생성한 뒤 두 branch 출력을 계산한다. 이후 LSF output을 differentiable transform layer로 segmentation probability map으로 바꾸고, supervised loss 둘과 DTC loss를 합쳐 end-to-end로 최적화한다. 추론 시에는 pixel-wise classification branch의 출력만 최종 segmentation 결과로 사용한다.

## 4. 실험 및 결과

실험은 두 개의 3D 의료영상 segmentation dataset에서 수행되었다. 첫 번째는 Left Atrium MRI dataset으로 총 100개의 3D gadolinium-enhanced MR image가 있으며, 80개를 학습, 20개를 검증에 사용했다. 두 번째는 Pancreas CT dataset으로 총 82개 복부 CT 이미지가 있고, 62개를 학습, 20개를 테스트에 사용했다. 논문은 대표적인 semi-supervised 설정으로 전체 학습 데이터 중 20%를 labeled, 80%를 unlabeled로 두었다.

구현은 PyTorch 기반이며, backbone은 모든 비교에서 VNet을 사용했다. 6000 iteration 동안 SGD로 학습했고, 초기 learning rate는 0.01이며 2500 iteration마다 0.1배 감소시켰다. batch size는 4이고, 그 안에 labeled image 2개와 unlabeled image 2개를 포함했다. 입력 크기는 Left Atrium에서는 $112 \times 112 \times 80$, Pancreas에서는 $96 \times 96 \times 96$의 sub-volume이다. 평가 지표는 Dice, Jaccard, ASD, 95HD를 사용했다.

먼저 ablation study에서 dual-task 자체와 consistency regularization의 효과를 따로 봤다. Pancreas CT에서 labeled 12장만 사용한 fully supervised setting에서, 단일 segmentation branch만 쓰는 경우 Dice는 70.63%였다. LSF branch만 쓰면 71.78%로 약간 개선되었고, 두 branch를 같이 supervision하는 Seg + LSF는 73.08%였다. 여기에 DTC를 추가한 Seg + LSF + DTC는 Dice 74.84%, Jaccard 60.78%, ASD 2.17, 95HD 9.34로 가장 좋았다. 즉, 단순히 두 task를 같이 학습하는 것보다, 두 task의 출력을 일관되게 맞추는 regularization이 추가 이득을 준다는 결과다.

같은 실험을 labeled 62장 전체로 했을 때도 유사한 경향이 유지되었다. Seg는 Dice 81.78%, LSF는 82.25%, Seg + LSF는 82.46%, Seg + LSF + DTC는 82.80%였다. 절대 향상 폭은 적지만, labeled data가 충분할 때도 DTC가 일관된 개선을 준다는 점을 보여준다.

semi-supervised 효율성 분석에서는 Pancreas CT에서 labeled 비율을 바꿔가며 fully supervised VNet, fully supervised dual-task VNet, 제안한 semi-supervised DTC 방법을 비교했다. 논문은 figure를 통해, labeled 데이터가 적을수록 semi-supervised DTC가 supervised baseline보다 더 큰 이득을 보인다고 설명한다. 반대로 labeled 수가 많아질수록 그 격차는 줄어드는데, 이는 semi-supervised learning에서 일반적으로 기대되는 현상과 일치한다.

기존 semi-supervised 방법들과의 비교도 중요하다. Pancreas CT에서 labeled 12장, unlabeled 50장 조건에서, fully supervised VNet은 Dice 70.63%였다. MT는 75.85%, DAN은 76.74%, Entropy Minimization은 75.31%, UA-MT는 77.26%, CCT는 76.58%, SASSNet은 77.66%를 기록했다. 제안 방법은 Dice 78.27%, Jaccard 64.75%, ASD 2.25, 95HD 8.36으로 모든 지표에서 가장 좋은 값을 기록했다. 특히 ASD와 95HD에서 개선이 눈에 띈다. 이는 단순 overlap뿐 아니라 표면 거리 관점에서도 경계 품질이 더 좋았음을 의미한다.

Left Atrium MRI에서도 유사한 결과가 나왔다. labeled 16장, unlabeled 64장 조건에서 fully supervised VNet은 Dice 86.03%였고, MT 88.23%, DAN 87.52%, Entropy Minimization 88.45%, UA-MT 88.88%, CCT 88.83%, SASSNet 89.27%였다. 제안 방법은 Dice 89.42%, Jaccard 80.98%, ASD 2.10, 95HD 7.32로 최고 성능을 달성했다. 여기서도 특히 표면 관련 지표 개선이 강조된다.

계산 비용 측면도 논문의 중요한 주장이다. Pancreas CT에서 제안 방법은 parameter 수 9.44M, training time 2.5시간으로 reported되었다. 이는 MT 2.9시간, DAN 3.3시간, UA-MT 3.9시간, CCT 4.1시간, SASSNet 3.9시간보다 짧다. Left Atrium MRI에서도 제안 방법은 2.2시간으로 다른 복잡한 SSL 방법들보다 빠르다. 저자들은 그 이유를, teacher-student나 multi-decoder처럼 여러 번 추론하지 않고 단일 네트워크 한 번의 forward로 consistency를 계산할 수 있기 때문이라고 설명한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 semi-supervised consistency regularization을 data perturbation이 아니라 task discrepancy라는 관점에서 새롭게 구성했다는 점이다. 같은 입력에 대해 segmentation map과 level set map이 서로 일치해야 한다는 아이디어는 직관적이면서도, geometry-aware representation을 자연스럽게 활용한다. 단순한 segmentation branch만 사용하는 모델보다 shape와 contour 정보를 더 잘 반영할 수 있다는 점도 장점이다.

둘째, 방법이 구조적으로 비교적 단순하다. teacher network, adversarial discriminator, iterative pseudo-label refinement, multi-view co-training 같은 복잡한 machinery 없이도 competitive한 성능을 낸다. 논문에 제시된 실험에서는 parameter 수와 training time 모두 좋은 trade-off를 보였다.

셋째, 완전 지도학습 설정에서도 DTC가 도움이 된다는 점이 흥미롭다. 이는 제안한 regularization이 unlabeled data 활용에만 한정되지 않고, task representation 자체의 상호보완성을 학습에 반영하는 일반적인 inductive bias로 기능할 수 있음을 보여준다.

다만 한계도 분명하다. 먼저 논문은 주로 single-class segmentation setting에 초점을 맞춘다. 저자들은 multi-class로의 확장이 straightforward하다고 말하지만, 본문에서 구체적인 multi-class formulation이나 실험은 제공하지 않는다. 따라서 실제 다중 장기 분할에서 동일한 정도의 이득이 유지되는지는 논문만으로 확언할 수 없다.

또한 level set representation과 sigmoid 기반 inverse approximation이 모든 종류의 의료영상 구조에 동일하게 적합한지는 추가 검증이 필요하다. 특히 객체의 위상 구조가 매우 복잡하거나, 경계가 희미하고 모호한 경우에 LSF branch가 얼마나 안정적으로 geometry prior를 제공하는지는 논문에서 깊게 다루지 않는다.

더불어 논문은 consistency를 두 branch 사이의 $L_2$ 차이로 두었는데, 왜 이 선택이 최적인지에 대한 비교는 없다. 예를 들어 KL divergence, boundary-aware consistency, uncertainty-weighted consistency 같은 대안과의 직접 비교는 제시되지 않았다. 따라서 성능 향상의 원인이 “task-level consistency”라는 큰 개념 때문인지, 아니면 특정 loss 설계와 level set representation의 조합 때문인지는 더 분리해서 볼 필요가 있다.

마지막으로, 성능 향상은 분명하지만 일부 setting에서는 절대 향상 폭이 아주 크지는 않다. 예를 들어 labeled data가 충분한 경우 Dice 개선은 제한적이다. 따라서 이 방법의 가장 큰 강점은 “극저라벨 상황에서의 unlabeled data 활용”에 있고, fully supervised regime에서 혁신적인 차이를 만드는 방식으로 이해하면 과장일 수 있다.

## 6. 결론

이 논문은 의료영상 semi-supervised segmentation을 위해 dual-task consistency라는 새로운 학습 원리를 제안한다. 같은 segmentation target을 pixel-wise classification과 level set regression이라는 두 task로 동시에 예측하게 하고, differentiable transform layer를 통해 두 출력을 같은 공간으로 맞춘 뒤 consistency loss를 가함으로써 unlabeled data까지 학습에 활용한다.

핵심 기여는 세 가지로 정리할 수 있다. 첫째, 기존의 data-level consistency와 다른 task-level consistency 기반 SSL 프레임워크를 제시했다. 둘째, level set representation을 활용해 shape와 geometry 정보를 segmentation 학습에 통합했다. 셋째, 두 개의 실제 3D 의료영상 데이터셋에서 state-of-the-art 수준의 성능과 좋은 계산 효율을 보였다.

실제 적용 측면에서 이 연구는 라벨링 비용이 큰 의료영상 환경에서 의미가 크다. 특히 적은 양의 fully annotated volume과 많은 unlabeled volume이 존재하는 상황에서, 비교적 단순한 구조로 정확도와 계산 효율을 함께 확보할 수 있다는 점이 실용적이다. 향후에는 multi-class segmentation, edge extraction이나 keypoint estimation 같은 추가 task와의 결합, 그리고 여러 task의 출력을 더 잘 융합하는 방식으로 확장될 가능성이 있다.
