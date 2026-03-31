# Beyond Sharing Weights for Deep Domain Adaptation

* **저자**: Artem Rozantsev, Mathieu Salzmann, Pascal Fua
* **발표연도**: 2016
* **arXiv**: [https://arxiv.org/abs/1603.06432](https://arxiv.org/abs/1603.06432)

## 1. 논문 개요

이 논문은 Deep Domain Adaptation에서 널리 사용되던 “source와 target이 같은 feature를 공유해야 한다”는 가정을 정면으로 비판한다. 기존의 많은 deep domain adaptation 방법은 source domain과 target domain에 대해 동일한 네트워크와 동일한 가중치를 사용하면서, 가능한 한 domain-invariant feature를 학습하려고 했다. 저자들은 이러한 전략이 domain shift를 줄이는 데는 도움이 될 수 있지만, 동시에 분류나 회귀에 필요한 discriminative power를 약화시킬 수 있다고 주장한다.

이 논문의 핵심 문제의식은 다음과 같다. source domain에서는 라벨이 풍부하지만, target domain에서는 라벨이 거의 없거나 아예 없을 수 있다. 그럼에도 불구하고 두 도메인은 완전히 같지 않다. 예를 들어 synthetic image와 real image는 같은 객체를 담고 있어도 texture, lighting, background statistics가 다르고, Amazon 제품 이미지와 Webcam 이미지도 시각적 특성이 다르다. 이런 차이를 무시하고 하나의 공유 표현만 강요하면, target에 맞는 적응력이 떨어질 수 있다.

그래서 저자들은 두 도메인을 각각 처리하는 two-stream architecture를 제안한다. source용 stream과 target용 stream을 따로 두되, 이 둘을 완전히 독립시키지는 않는다. 일부 층은 가중치를 공유할 수 있고, 일부 층은 공유하지 않되, 대응되는 가중치들이 너무 멀어지지 않도록 regularization을 건다. 더 나아가 최종 representation 수준에서는 MMD를 통해 source와 target의 feature distribution을 가깝게 유지한다. 즉, 이 논문은 “완전한 공유”와 “완전한 분리”의 중간 지점을 정교하게 모델링하려는 시도라고 볼 수 있다.

이 문제는 실제적으로도 중요하다. 딥러닝은 대량의 annotated data를 필요로 하지만, 실제 환경에서 target domain의 라벨을 충분히 모으는 것은 비용이 크다. 반면 synthetic data는 비교적 쉽게 대량 생성할 수 있다. 따라서 synthetic-to-real adaptation이 잘 된다면 detection, pose estimation 같은 실제 비전 문제에서 큰 효과를 기대할 수 있다. 이 논문은 바로 그런 응용 가능성을 실험으로 뒷받침한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 “도메인 차이를 없애려 하지 말고, 그 차이를 모델 안에 직접 반영하자”는 것이다. 기존 방법들이 source와 target에서 같은 가중치를 사용하며 domain-invariant representation을 추구했다면, 이 논문은 source와 target에 서로 다른 가중치를 허용한다. 다만 두 네트워크가 완전히 따로 놀면 같은 문제를 푼다는 구조적 공통성이 사라지고, target 쪽은 소량 데이터로 인해 쉽게 overfitting될 수 있다. 따라서 저자들은 대응되는 층의 가중치가 서로 관련되되 완전히 같지는 않도록 만드는 제약을 둔다.

이 설계의 중요한 포인트는 “차이는 허용하지만, 무질서는 허용하지 않는다”는 점이다. 저자들은 source와 target의 가중치가 정확히 같아야 한다고 강요하지 않고, 한쪽 가중치가 다른 쪽 가중치의 선형 변환으로 표현될 수 있으면 좋다고 본다. 이는 예를 들어 low-level appearance가 달라지는 상황, 즉 밝기 변화나 contrast 차이처럼 단순한 통계적 shift를 설명하는 데 자연스럽다.

또 하나의 핵심은, 어느 층에서 가중치를 공유하지 않을지를 수작업 직관에만 의존하지 않고 MMD 기반 기준으로 선택하려 했다는 점이다. 저자들은 여러 sharing/non-sharing configuration을 시도한 뒤, source와 target의 최종 representation 사이의 $MMD^2$가 가장 낮은 구조를 좋은 구조로 본다. 이는 validation label이 없는 상황에서도 어느 층을 domain-specific하게 둘지 결정하는 실용적 기준으로 제시된다.

기존 접근과의 차별점은 명확하다. DDC나 GRL 같은 대표적 방법은 서로 다른 도메인에서 같은 feature space를 학습하도록 압박한다. 반면 이 논문은 feature의 최종 분포는 어느 정도 가깝게 두면서도, 네트워크 내부에서는 도메인별 specialization을 허용한다. 즉, 완전한 invariant learning 대신 “related but different”라는 구조를 채택한 점이 가장 큰 차별점이다.

## 3. 상세 방법 설명

논문의 전체 목적함수는 source supervised loss, target supervised loss, weight regularizer, 그리고 unsupervised MMD regularizer의 합으로 구성된다. 논문은 source 데이터 집합을 $\mathbf{X}^s={\mathbf{x}_i^s}_{i=1}^{N^s}$, target 데이터 집합을 $\mathbf{X}^t={\mathbf{x}_i^t}_{i=1}^{N^t}$로 두고, 각 도메인의 라벨을 $Y^s$, $Y^t$로 표기한다. target이 unsupervised일 수도 있으므로, target에서 실제 라벨이 있는 샘플 개수를 $N_l^t$로 두며 unsupervised setting에서는 $N_l^t=0$이다.

전체 loss는 다음과 같다.

$$
L(\theta^s,\theta^t \mid \mathbf{X}^s, Y^s, \mathbf{X}^t, Y^t) = L_s + L_t + L_w + L_{MMD}
$$

여기서 $\theta_j^s$와 $\theta_j^t$는 각각 source stream과 target stream의 $j$번째 층 파라미터이다.

먼저 source stream의 supervised loss는 다음과 같다.

$$
L_s = \frac{1}{N^s}\sum_{i=1}^{N^s} c(\theta^s \mid \mathbf{x}_i^s, y_i^s)
$$

여기서 $c(\cdot)$는 일반적인 classification loss이며, 논문에서는 logistic loss나 hinge loss를 예로 든다. 즉, source domain에서 주어진 정답에 맞게 source stream이 학습되도록 만드는 항이다.

target domain에 라벨이 일부라도 있다면 target supervised loss도 함께 사용한다.

$$
L_t = \frac{1}{N_l^t}\sum_{i=1}^{N_l^t} c(\theta^t \mid \mathbf{x}_i^t, y_i^t)
$$

이 항은 supervised domain adaptation에서 중요하고, unsupervised setting에서는 $N_l^t=0$이므로 사실상 빠진다.

그다음이 이 논문의 가장 중요한 구성 요소인 weight regularizer이다.

$$
L_w = \lambda_w \sum_{j \in \Omega} r_w(\theta_j^s,\theta_j^t)
$$

여기서 $\Omega$는 가중치를 공유하지 않기로 한 층들의 집합이다. 즉, 모든 층에 대해 별도 가중치를 허용하는 것이 아니라, 특정 층만 non-shared로 두고 그 층들에 대해 regularization을 건다.

가장 단순한 생각은 $\theta_j^s$와 $\theta_j^t$의 차이를 직접 줄이는 것이다. 하지만 저자들은 이것이 domain shift를 제대로 모델링하지 못한다고 본다. 예를 들어 source와 target의 차이가 단순히 평균값이나 dynamic range의 차이라면, 두 가중치가 정확히 같을 필요는 없고 간단한 선형 변환 관계만 유지해도 충분할 수 있다. 그래서 다음과 같은 선형 변환 기반 regularizer를 제안한다.

L2 버전은 다음과 같다.

$$
r_w(\theta_j^s,\theta_j^t) = \left\lVert a_j \theta_j^s + b_j - \theta_j^t \right\rVert_2^2
$$

여기서 $a_j$와 $b_j$는 각 층마다 학습되는 scalar 파라미터이다. 이는 source 가중치를 단순한 affine transform으로 조정했을 때 target 가중치와 가까워지도록 만든다.

논문은 여기에 더해 exponential 형태도 제안한다.

$$
r_w(\theta_j^s,\theta_j^t) = \exp\left(\left\lVert a_j \theta_j^s + b_j - \theta_j^t \right\rVert^2\right)-1
$$

실험적으로는 이 exponential regularizer가 더 좋은 결과를 냈다고 보고한다. 저자들은 quadratic이나 piecewise linear 같은 더 복잡한 변환도 시도했지만 성능 향상은 없었다고 명시한다. 이 부분은 중요한데, 즉 논문의 성능 향상은 복잡한 weight mapping 때문이 아니라 “non-shared but softly related”라는 구조적 아이디어 자체에서 나온다는 해석이 가능하다.

다음은 unsupervised regularizer인 MMD 항이다.

$$
L_{MMD} = \lambda_u , r_u(\theta^s,\theta^t \mid \mathbf{X}^s,\mathbf{X}^t)
$$

이 항의 목적은 source와 target의 최종 feature representation 분포를 가깝게 만드는 것이다. 논문은 마지막 classifier 직전 feature를 각각 $\mathbf{f}_i^s$와 $\mathbf{f}_j^t$로 두고, 그 분포 간 거리를 Maximum Mean Discrepancy로 측정한다.

MMD의 정의는 다음과 같이 제시된다.

$$
\text{MMD}^2({\mathbf{f}_i^s}, {\mathbf{f}_j^t}) = \left| \sum_{i=1}^{N^s}\frac{\phi(\mathbf{f}_i^s)}{N^s} - \sum_{j=1}^{N^t}\frac{\phi(\mathbf{f}_j^t)}{N^t} \right|^2
$$

여기서 $\phi(\cdot)$는 RKHS로의 mapping이다. 실제 계산에서는 kernel trick을 이용해 다음 형태로 쓴다.

$$
r_u(\theta^s,\theta^t \mid \mathbf{X}^s,\mathbf{X}^t) = \sum_{i,i'} \frac{k(\mathbf{f}_i^s,\mathbf{f}_{i'}^s)}{(N^s)^2} - 2\sum_{i,j} \frac{k(\mathbf{f}_i^s,\mathbf{f}_j^t)}{N^s N^t} + \sum_{j,j'} \frac{k(\mathbf{f}_j^t,\mathbf{f}_{j'}^t)}{(N^t)^2}
$$

커널 함수는 RBF kernel

$$
k(u,v)=\exp(-|u-v|^2 / \sigma)
$$

를 사용하며, 논문에서는 $\sigma=1$로 고정했다. 저자들은 이 choice에 크게 민감하지 않았다고 말한다.

이 방법의 구조를 직관적으로 요약하면 다음과 같다. 네트워크 초반이나 중간의 일부 층에서는 source와 target이 서로 다른 표현을 갖도록 허용한다. 대신 그 차이는 학습 가능한 선형 관계 안에서만 허용한다. 그리고 마지막 representation 수준에서는 MMD로 두 도메인의 feature distribution을 정렬한다. 따라서 이 방법은 내부적으로는 domain-specific specialization을 허용하면서, 출력 representation 수준에서는 alignment를 유지하는 절충 구조이다.

학습 절차도 비교적 명확하다. 먼저 source stream을 source data만으로 pre-train한다. 그다음 target stream의 가중치를 source stream의 pre-trained weight로 초기화한다. 동시에 $a_j=1$, $b_j=0$으로 두어 처음에는 identity transformation에서 시작한다. 이후 두 stream 전체를 함께 optimization하며, 목적함수의 모든 항을 joint training한다. 최적화 알고리즘은 AdaDelta이고, 실제 계산은 mini-batch 단위로 수행된다.

또한 논문은 각 task에 따라 네트워크 구조를 다르게 사용한다. Office에서는 AlexNet, MNIST-USPS에서는 LeNet 계열 표준 CNN, UAV detection에서는 3개의 convolution/max-pooling 층과 2개의 fully-connected 층으로 이루어진 CNN을 사용한다. 즉, 제안 방식은 특정 backbone에 묶이지 않고 비교적 일반적인 adaptation framework로 제시된다.

## 4. 실험 및 결과

논문은 classification, detection, regression을 모두 포함하는 다양한 실험을 수행한다. 이는 제안 방법이 특정 벤치마크에 한정된 트릭이 아니라 비교적 일반적인 domain adaptation 원리라는 점을 보여주기 위한 구성으로 보인다.

### 4.1 Leveraging Synthetic Data for Drone Detection

가장 먼저 UAV detection 문제를 다룬다. 이 문제는 실제 드론 이미지 수집이 어렵고, 다양한 pose와 background를 모두 담기 어렵기 때문에 synthetic data 활용이 특히 자연스러운 분야이다. 논문은 synthetic 이미지를 source domain, real 이미지를 target domain으로 둔다.

UAV-200에는 두 가지 버전이 있다. UAV-200 (small)은 상대적으로 class imbalance가 덜 심해서 accuracy를 쓴다. UAV-200 (full)은 실제 detection에 가까운 불균형 데이터라 precision-recall과 Average Precision(AP)를 사용한다.

이 실험은 supervised domain adaptation setting이다. 즉 source와 target 모두 라벨이 존재한다.

네트워크는 각 stream마다 3개의 convolution/max-pooling 층과 2개의 fully-connected 층을 갖는 CNN이고, classification layer는 hinge loss를 사용한다. 저자들은 어느 층을 non-shared로 둘지 모든 가능한 조합을 실험한 뒤, 각 configuration에서 source/target 출력 간 $MMD^2$를 계산했다. 드론 데이터에서는 첫 세 개 층을 non-shared로 두고 나머지를 shared로 둘 때, 그리고 weight regularizer를 exponential 형태로 쓸 때 가장 낮은 MMD를 얻었다. 이는 synthetic와 real 간 차이가 low-level appearance에서 크기 때문이라고 해석한다.

흥미로운 점은 MMD 기반 architecture selection이 validation accuracy와도 대체로 잘 맞았다는 것이다. validation에서 최고 성능 구조와 MMD 최소 구조가 완전히 동일하지는 않았지만, MMD가 가장 낮은 구조가 두 번째로 좋은 성능을 기록했다. 저자들은 이를 validation data가 없을 때의 실용적 선택 기준으로 제시한다.

UAV-200 (small)에서 accuracy 비교 결과는 다음과 같다. ITML 0.60, ARC-t asymmetric 0.55, ARC-t symmetric 0.60, HFA 0.75, DDC 0.89, 제안 방법 0.92이다. 특히 DDC 대비 개선이 중요하다. 왜냐하면 DDC 역시 MMD를 사용해 source와 target feature를 정렬하지만, single-stream shared-weight 구조를 사용하기 때문이다. 따라서 이 비교는 “MMD 자체”가 아니라 “non-shared weights 허용”이 추가 성능 향상에 기여했다는 논문의 핵심 주장을 직접 뒷받침한다.

UAV-200 (full)에서는 AP 기준으로 비교한다. synthetic only CNN은 0.314, real only CNN은 0.575, source pretrain 후 target fine-tuning은 0.612, source 고정 + $L_t+L_w$는 0.655, real+synthetic을 함께 써서 $L_s+L_t$만 사용하는 방식은 0.569, DDC는 0.664였다. 제안 방법은 $L_s+L_t+L_w$일 때 0.673, $L_s+L_t+L_{MMD}$일 때 0.711, 전체 loss $L_s+L_t+L_w+L_{MMD}$일 때 0.732를 달성한다.

이 결과는 몇 가지 점에서 의미가 있다. 첫째, synthetic와 real을 그냥 합쳐서 single-stream CNN을 학습하는 것은 효과가 크지 않다. 둘째, MMD만 써도 성능이 올라가지만, weight regularizer를 함께 넣으면 더 좋아진다. 셋째, target stream을 독립적으로 두되 적절한 제약으로 source와 연결하는 것이 overfitting을 막고 adaptation을 안정화한다는 논문의 설명과 잘 맞는다. 저자들은 실제로 $L_w$ 없이 MMD만 쓰는 경우보다 full loss가 더 좋은 이유를 target stream overfitting 완화로 해석한다.

샘플 수 변화 실험도 실용적이다. real positive sample 수를 200에서 5000까지 늘리는 실험에서, 제안 방법은 항상 real-only CNN보다 우수했고 DDC보다도 대체로 더 좋았다. 특히 200개의 real sample만으로도, real-only single-stream CNN을 2500개의 real sample로 학습한 것과 비슷하거나 약간 더 좋은 성능을 보였다고 한다. 이는 실제로 5~10% 수준의 라벨만 모아도 synthetic data를 잘 활용하면 충분한 성능을 얻을 수 있음을 시사한다.

반대로 synthetic sample 수를 늘리는 실험에서는, 제안 방법의 AP가 synthetic 데이터 증가에 따라 꾸준히 상승했다. DDC도 향상되지만 대부분 구간에서 제안 방법이 더 좋았다. synthetic 데이터가 전혀 없을 때는 두 방법 모두 결국 real-only CNN으로 수렴하므로 차이가 사라진다. 이 역시 제안 방식의 강점이 “synthetic-to-real gap을 구조적으로 처리하는 능력”에 있음을 보여준다.

### 4.2 Unsupervised Domain Adaptation on Office

두 번째 주요 실험은 Office benchmark이다. Amazon, DSLR, Webcam 세 도메인으로 구성되며 31개 object category를 가진 전형적인 unsupervised domain adaptation 데이터셋이다. 논문은 fully-transductive protocol을 사용하여, source의 모든 정보는 쓰되 target에는 라벨을 전혀 사용하지 않는다.

이 실험에서는 AlexNet을 각 stream에 사용하고 ImageNet pretraining 후 fine-tuning한다. 구조 선택은 드론 실험과 마찬가지로 MMD 기반 기준을 사용했다. Amazon $\rightarrow$ Webcam 설정에서 실험한 결과, 마지막 두 fully-connected 층을 non-shared로 두는 것이 가장 낮은 $MMD^2$를 보였고, 이를 전체 Office 실험에 사용했다. 저자들은 이는 Office 도메인 간 차이가 단순 low-level appearance가 아니라 더 high-level semantic variation에 가깝기 때문이라고 해석한다.

비교 결과는 다음과 같다. A $\rightarrow$ W에서는 GFK 0.214, DLID 0.519, DDC 0.605, DAN 0.645, DRCN 0.687, GRL 0.730, Ours(+DDC) 0.630, Ours(+GRL) 0.760이다. 평균 성능은 GRL이 0.895인데, Ours(+GRL)은 0.908이다. 즉, 동일한 domain confusion 계열 손실을 쓰더라도 shared-weight 대신 two-stream non-shared 구조를 쓰면 성능이 더 오른다.

이 실험은 논문 주장에 매우 중요하다. 만약 성능 향상이 단순히 MMD를 잘 쓴 것 때문이라면 GRL 기반 설정에서는 이득이 없어야 한다. 하지만 실제로는 GRL loss를 쓰더라도 non-shared stream이 shared stream보다 좋다. 따라서 논문이 제기한 “feature invariance만 강요하는 것은 손해일 수 있다”는 주장이 데이터로 뒷받침된다.

### 4.3 Domain Adaptation on MNIST-USPS

MNIST와 USPS는 서로 다른 숫자 이미지 데이터셋이며, unsupervised adaptation 실험이 수행된다. MNIST 이미지는 USPS 크기에 맞게 리스케일하고 pixel intensity에 $L_2$ normalization을 적용했다. 네트워크는 LeNet 계열 표준 CNN을 사용한다.

MMD 기반 구조 선택 결과, 이 실험에서는 모든 층을 non-shared로 두는 것이 최적이었다고 한다. 이는 두 도메인 간 저수준 차이뿐 아니라 전반적 표현 차이가 비교적 커서 모든 단계에서 domain-specific adaptation이 이득이라는 뜻으로 해석할 수 있다. 다만 이 해석은 논문이 직접 장황하게 설명하지는 않고, 결과로만 제시한다.

성능은 M$\rightarrow$U에서 DDC 0.478 대비 제안 방법 0.607, U$\rightarrow$M에서 DDC 0.631 대비 제안 방법 0.673, 평균 0.554 대비 0.640이다. 비딥러닝 기반 기법들인 PCA, SA, GFK, TCA, SSTCA, TSL, JCSL보다도 모두 높다. 이 결과는 제안 방식이 복잡한 natural image뿐 아니라 비교적 단순한 digit classification domain shift에서도 효과가 있음을 보여준다.

### 4.4 Supervised Facial Pose Estimation

마지막으로 논문은 classification과 detection을 넘어 regression에도 이 아이디어가 유효함을 보이기 위해 facial pose estimation을 다룬다. 입력은 $50 \times 50$ patch이고, 출력은 5개 facial landmark의 2D 좌표를 포함한 10차원 벡터이다. source는 synthetic face image, target은 real face image이며, 각각 약 10k 이미지가 있다. 학습에는 synthetic 전체와 real 100장만 쓰고 나머지는 테스트에 사용한다.

평가지표는 PCP-score이다. 각 landmark가 ground truth로부터 2픽셀 이내에 있으면 correct로 친다. 결과는 오른쪽 눈 64.2 $\rightarrow$ 68.0 $\rightarrow$ 71.8, 왼쪽 눈 39.3 $\rightarrow$ 56.2 $\rightarrow$ 60.3, 코 56.3 $\rightarrow$ 64.1 $\rightarrow$ 64.5, 오른쪽 입꼬리 47.8 $\rightarrow$ 57.6 $\rightarrow$ 59.8, 왼쪽 입꼬리 42.3 $\rightarrow$ 55.5 $\rightarrow$ 57.7, 평균 50.0 $\rightarrow$ 60.3 $\rightarrow$ 62.8로, synthetic only보다 DDC가 좋고 DDC보다 제안 방법이 다시 좋다.

이 결과는 이 논문의 아이디어가 classifier 전용 트릭이 아니라 “representation adaptation”의 일반 구조임을 보여준다. 즉, target에 맞는 feature extractor를 별도로 두되 source와 유연하게 연결하는 전략이 regression에도 적용 가능하다는 것이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 domain adaptation에서 너무 당연하게 받아들여지던 “weight sharing” 가정을 실험적으로 깨뜨렸다는 점이다. 단순히 새로운 loss 하나를 추가한 것이 아니라, 문제를 보는 관점을 바꿨다. 즉, 좋은 adaptation이 반드시 domain-invariant representation을 의미하지는 않으며, 오히려 도메인 특화 표현을 일정 부분 허용해야 더 discriminative할 수 있다는 주장을 명확한 구조와 실험으로 제시했다.

또 다른 강점은 제안 방식이 매우 다양한 문제에서 일관되게 작동했다는 점이다. UAV detection, Office object recognition, digit classification, facial landmark regression까지 classification, detection, regression을 아우른다. 특히 synthetic-to-real adaptation과 같이 실제 응용에서 매우 중요한 설정에서 성능 향상을 보인 점은 실용성이 높다.

설계도 비교적 단순하다. 대응 층의 가중치 차이를 affine transform 수준으로 모델링하고, 마지막 representation에는 MMD를 거는 구조이므로 기존 CNN backbone에 비교적 자연스럽게 얹을 수 있다. 또한 어느 층을 분리할지 MMD 기반으로 선택하는 기준을 제시해, validation label이 없을 때도 구조 선택의 근거를 제공한다는 점이 실용적이다.

하지만 한계도 분명하다. 첫째, 어떤 층을 shared/non-shared로 둘지 여전히 여러 configuration을 실험해 봐야 한다. 논문은 MMD 기반 기준을 제안하지만, 결국 후보 구조들을 여러 개 학습해야 한다는 뜻이므로 계산 비용이 크다. 자동으로 구조를 학습하는 방식은 아니다.

둘째, weight relation을 선형 변환 $a_j \theta_j^s + b_j$로 제한한 것은 단순하고 안정적이지만, 실제 domain shift를 충분히 표현하지 못할 수 있다. 저자들도 더 복잡한 변환을 시도했으나 개선이 없었다고 보고하지만, 이것이 일반적으로 항상 불필요하다는 뜻은 아니다. 다만 논문 범위 내에서는 복잡한 변환의 이점이 입증되지는 않았다.

셋째, 논문은 최종 feature는 domain-invariant해야 한다고 하면서도, 중간층에서는 domain-specificity를 허용한다. 이 철학은 설득력 있지만, 어느 정도의 invariance와 어느 정도의 specialization이 최적인지에 대한 이론적 설명은 충분히 제공하지 않는다. 다시 말해, 왜 어떤 데이터셋에서는 초반층을 분리하고 어떤 데이터셋에서는 후반층을 분리해야 하는지에 대한 일반 원리는 경험적으로 제시될 뿐, 엄밀하게 정식화되지는 않았다.

넷째, Office 실험에서 Ours(+DDC)는 DDC와 거의 비슷하거나 일부 설정에서는 큰 차이가 없다. 즉, 제안 방법이 항상 극적으로 향상되는 것은 아니며, 어떤 base adaptation loss와 결합하느냐에 따라 이득의 크기가 달라진다. 또 논문이 비교한 방법들은 2015~2016년 무렵의 기법들이어서, 이후 등장한 adversarial adaptation, self-training, contrastive alignment, transformer 기반 방법들과의 비교는 당연히 포함되어 있지 않다.

비판적으로 보면, 이 논문은 “domain invariance는 해롭다”라고 다소 강하게 말하지만, 실제로는 invariance 자체를 버리는 것이 아니라 마지막 representation에서는 여전히 MMD로 정렬한다. 따라서 이 논문의 더 정확한 메시지는 “모든 층에서 무조건 invariance를 강요하는 것은 해로울 수 있다”에 가깝다. 즉, invariant learning을 전면 부정한다기보다, domain-specific parameterization과 invariant representation의 균형을 재설계한 논문으로 읽는 것이 더 정확하다.

## 6. 결론

이 논문은 Deep Domain Adaptation에서 source와 target이 반드시 같은 가중치를 공유해야 한다는 통념을 재검토하고, 두 도메인을 위한 two-stream network를 제안했다. 핵심은 일부 층의 가중치를 도메인별로 분리하되, 대응되는 가중치가 선형 변환 관계를 유지하도록 regularize하고, 마지막 feature representation은 MMD로 정렬하는 것이다.

주요 기여는 세 가지로 요약할 수 있다. 첫째, domain shift를 직접 모델링하는 two-stream non-shared architecture를 제안했다. 둘째, 가중치 차이를 단순한 자유도 증가가 아니라 학습 가능한 선형 관계로 제약하는 weight regularizer를 도입했다. 셋째, 어느 층을 공유하고 어느 층을 분리할지 MMD 기반 기준으로 선택하는 실용적 전략을 제시했다.

실험적으로도 이 방법은 UAV detection, Office, MNIST-USPS, facial pose estimation에서 shared-weight 기반 방법들보다 일관되게 우수한 성능을 보였다. 특히 synthetic-to-real adaptation에서 적은 수의 real label만으로도 좋은 성능을 확보할 수 있다는 결과는 실제 응용 가치가 크다.

향후 연구 측면에서 이 논문은 중요한 방향을 제시한다. domain adaptation에서 “무조건 같은 feature”를 강요하기보다, 어떤 층에서 어떤 형태의 domain-specificity를 허용할 것인가를 구조적으로 설계하는 것이 중요하다는 점을 보여주기 때문이다. 이후의 연구에서는 더 정교한 parameter transformation, 자동화된 layer-sharing selection, stronger distribution matching 기법과 결합하는 방향으로 확장될 수 있다. 그런 의미에서 이 논문은 deep domain adaptation이 invariance 중심 사고에서 보다 유연한 domain-aware modeling으로 이동하는 데 중요한 역할을 한 연구라고 평가할 수 있다.
