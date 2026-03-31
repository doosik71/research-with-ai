{"title": "Unsupervised Domain Adaptation with Residual Transfer Networks", "author": "Mingsheng Long, Han Zhu, Jianmin Wang, and Michael I. Jordan", "year": 2016, "url": "[https://arxiv.org/abs/1602.04433](https://arxiv.org/abs/1602.04433)", "summary": "unsupervised_domain_adaptation_with_residual_transfer_networks.md", "slide": ""}

# Unsupervised Domain Adaptation with Residual Transfer Networks

* **저자**: Mingsheng Long, Han Zhu, Jianmin Wang, and Michael I. Jordan
* **발표연도**: 2016
* **arXiv**: [https://arxiv.org/abs/1602.04433](https://arxiv.org/abs/1602.04433)

## 1. 논문 개요

이 논문은 **unsupervised domain adaptation** 문제를 다룬다. 즉, 라벨이 있는 source domain 데이터와 라벨이 없는 target domain 데이터를 함께 사용하여, target domain에서도 잘 동작하는 분류기를 학습하는 것이 목표다. 기존의 많은 deep domain adaptation 방법은 “좋은 feature를 만들면 source classifier를 그대로 target에도 써도 된다”는 가정을 둔다. 그러나 저자들은 이 가정이 실제로는 지나치게 강하다고 본다. 같은 클래스 집합을 다루더라도 source와 target의 데이터 분포가 다르면, feature 분포뿐 아니라 **classifier 자체도 달라질 수 있다**는 것이다.

이 논문의 핵심 문제의식은 다음과 같다. 기존 방법은 주로 $p(\mathbf{x}) \neq q(\mathbf{x})$ 같은 **feature distribution shift**를 줄이는 데 집중했지만, 실제로는 $f_s(\mathbf{x}) \neq f_t(\mathbf{x})$인 **classifier mismatch**도 존재할 수 있다. 이 경우 source에서 학습한 분류기를 target에 그대로 옮기면 성능이 제한될 수 있다. 따라서 저자들은 **transferable feature**와 **adaptive classifier**를 동시에 학습하는 하나의 end-to-end 구조를 제안한다.

이 문제가 중요한 이유는, 많은 실제 응용에서 target domain 라벨을 수집하기 어렵기 때문이다. 예를 들어 웹 이미지와 카메라 이미지, 서로 다른 장비로 촬영한 영상, 혹은 서로 다른 수집 환경에서 얻은 데이터는 같은 클래스라도 분포가 달라진다. 이런 상황에서 feature만 맞추는 것으로 충분하지 않다면, classifier까지 조정할 수 있는 방법이 필요하다. 이 논문은 바로 그 지점을 겨냥해, residual learning의 아이디어를 domain adaptation에 연결한다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 **source classifier와 target classifier가 완전히 동일하지는 않지만, 큰 차이가 아니라 작은 residual perturbation으로 연결될 수 있다**는 것이다. 즉, target classifier를 새로 완전히 학습하는 대신, source classifier와의 차이를 residual function으로 모델링하면 더 안전하고 현실적인 adaptation이 가능하다고 본다.

저자들은 ResNet의 residual learning에서 착안해, source classifier와 target classifier 사이를 다음과 같이 연결한다.

$$
f_S(\mathbf{x}) = f_T(\mathbf{x}) + \Delta f(\mathbf{x})
$$

여기서 $f_T(\mathbf{x})$는 target classifier의 pre-softmax 출력이고, $\Delta f(\mathbf{x})$는 source와 target classifier의 차이를 나타내는 residual function이다. 이 식의 의미는 “source classifier는 target classifier에 작은 보정값을 더한 형태”라는 것이다. 이 구조를 사용하면 classifier mismatch를 명시적으로 모델링할 수 있다.

동시에 feature adaptation도 수행한다. 그런데 단일 층 feature만 맞추는 대신, 여러 task-specific layer의 feature를 **tensor product**로 결합한 뒤, 그 fused representation에 대해 MMD를 최소화한다. 즉, “여러 층에서 나온 정보를 따로따로 정렬하는 대신, 서로의 상호작용까지 보존한 fused feature 공간에서 source와 target 분포를 맞춘다”는 발상이다.

또 하나의 중요한 아이디어는 **entropy minimization**이다. target에는 라벨이 없으므로 classifier adaptation이 쉽게 불안정해질 수 있다. 이를 보완하기 위해 저자들은 target 예측의 엔트로피를 낮추는 항을 추가해, decision boundary가 target의 low-density region을 지나가도록 유도한다. 이것은 target classifier가 target 데이터 구조에 더 잘 맞도록 만드는 역할을 한다.

정리하면, 이 논문의 차별점은 다음과 같다. 기존 deep adaptation이 주로 feature adaptation에 머물렀다면, RTN은 **feature adaptation + classifier adaptation**을 하나의 딥러닝 프레임워크로 결합한다. 그리고 classifier adaptation을 residual block으로 구현함으로써, target 라벨이 없는 상황에서도 source와 target classifier의 관계를 학습 가능하게 만든다.

## 3. 상세 방법 설명

### 전체 구조

논문은 AlexNet 기반 CNN을 확장하여 **Residual Transfer Network (RTN)** 를 구성한다. 전체적으로는 세 가지 축이 동시에 작동한다.

첫째, source 라벨 데이터에 대한 supervised learning이다. 이는 source classifier가 최소한 source domain에서는 잘 동작하도록 하는 기본 축이다.

둘째, feature adaptation이다. 마지막 feature layer 위에 bottleneck layer $fcb$를 추가하고, classifier layer $fcc$와 함께 여러 층의 feature를 tensor fusion한 뒤 MMD 기반 분포 정렬을 수행한다.

셋째, classifier adaptation이다. target classifier $f_T$와 source classifier $f_S$ 사이에 residual block을 넣어, 두 분류기 사이의 차이를 residual function $\Delta f$로 학습한다. 동시에 target classifier 예측 엔트로피를 줄여 target 구조에 맞는 decision boundary를 형성하도록 한다.

즉, RTN은 단순히 “좋은 shared feature를 만들고 source classifier를 재사용하는 구조”가 아니라, **feature와 classifier를 함께 옮기는 구조**다.

### 기본 supervised 학습

source domain $\mathcal{D}_s = {(\mathbf{x}_i^s, y_i^s)}_{i=1}^{n_s}$, target domain $\mathcal{D}_t = {\mathbf{x}_j^t}_{j=1}^{n_t}$가 주어진다. target에는 라벨이 없다.

source classifier에 대해서는 일반적인 cross-entropy loss를 최소화한다.

$$
\min_{f_s} \frac{1}{n_s} \sum_{i=1}^{n_s} L(f_s(\mathbf{x}_i^s), y_i^s)
$$

여기서 $L(\cdot,\cdot)$는 cross-entropy loss다. 이 항은 source supervision을 담당한다.

저자들은 convolutional layer는 비교적 generic feature를 학습하므로, 이 부분은 완전히 새로 적응시키기보다 fine-tuning하는 방식을 선택한다. 반면 상위 fully connected layer는 domain-specific해지기 쉬우므로 adaptation의 주요 대상이 된다.

### Feature adaptation: tensor MMD

저자들은 여러 층의 feature를 함께 적응시키기 위해 layer set $\mathcal{L} = {fcb, fcc}$를 사용한다. 각 샘플에 대해 여러 층의 출력을 tensor product로 결합한다.

$$
\mathbf{z}_i^s \triangleq \bigotimes_{\ell \in \mathcal{L}} \mathbf{x}_i^{s\ell}, \qquad \mathbf{z}_j^t \triangleq \bigotimes_{\ell \in \mathcal{L}} \mathbf{x}_j^{t\ell}
$$

이 식은 여러 층의 feature를 loss 없이 결합해 하나의 fused representation으로 만드는 역할을 한다. DAN처럼 층마다 독립적인 MMD penalty를 따로 두는 대신, RTN은 fused feature 하나에 대해 MMD를 적용한다.

그 다음 source와 target의 fused feature 분포 차이를 MMD로 측정한다.

$$
D_{\mathcal{L}}(\mathcal{D}_s,\mathcal{D}_t) = \sum_{i=1}^{n_s}\sum_{j=1}^{n_s}\frac{k(\mathbf{z}_i^s,\mathbf{z}_j^s)}{n_s^2} + \sum_{i=1}^{n_t}\sum_{j=1}^{n_t}\frac{k(\mathbf{z}_i^t,\mathbf{z}_j^t)}{n_t^2} = 2\sum_{i=1}^{n_s}\sum_{j=1}^{n_t}\frac{k(\mathbf{z}_i^s,\mathbf{z}_j^t)}{n_s n_t}
$$

여기서 커널은 Gaussian kernel이다.

$$
k(\mathbf{z},\mathbf{z}') = \exp\left( -\frac{|\mathrm{vec}(\mathbf{z})-\mathrm{vec}(\mathbf{z}')|^2}{b} \right)
$$

즉, tensor product로 결합한 multilayer feature를 RKHS에 올린 뒤, source와 target 평균 임베딩 차이를 줄이는 방식이다.

이 설계의 장점은 논문 기준으로 두 가지다. 하나는 여러 층의 상호작용을 더 풍부하게 반영할 수 있다는 점이고, 다른 하나는 DAN처럼 층 수만큼 여러 MMD 하이퍼파라미터를 고를 필요 없이 **하나의 penalty로 adaptation을 제어할 수 있다**는 점이다.

다만 tensor feature는 차원이 커질 수 있기 때문에, 논문은 bilinear pooling을 이용해 fusion feature 차원을 줄였다고 설명한다. 구체적인 bilinear pooling 설계 상세는 본문에 길게 제시되지 않았다.

### Classifier adaptation: residual transfer block

이 논문의 가장 중요한 부분이다. 기존 방법은 adapted feature 위에서 source classifier와 target classifier를 동일하다고 가정하는 경우가 많다. 그러나 저자들은 이 가정을 완화해, 두 분류기가 residual로 연결된다고 본다.

기본 관계식은 다음과 같다.

$$
f_S(\mathbf{x}) = f_T(\mathbf{x}) + \Delta f(\mathbf{x})
$$

여기서 중요한 점은 $f_S, f_T$가 **softmax 이전의 logit**이라는 것이다. 최종 분류 확률은 각각
$$
f_s(\mathbf{x}) \triangleq \sigma(f_S(\mathbf{x})), \qquad f_t(\mathbf{x}) \triangleq \sigma(f_T(\mathbf{x}))
$$
로 정의된다.

논문은 residual block을 source와 target classifier 사이에 삽입한다. $f_T(\mathbf{x})$를 shortcut 경로의 입력처럼 보고, residual layers $fc1$–$fc2$가 $\Delta f(\mathbf{x})$를 학습하며, element-wise sum 후의 출력이 $f_S(\mathbf{x})$가 된다. residual layers는 $c \times c$ fully connected 구조이며, 여기서 $c$는 클래스 수다.

이때 왜 $f_S$를 residual block의 출력으로 두는지가 중요하다. source에는 라벨이 있으므로 $f_S$는 source supervision으로 직접 학습할 수 있다. 반면 만약 $f_T$를 residual block 출력으로 두면, target에는 라벨이 없어서 표준 back-propagation만으로 학습이 어렵다. 따라서 source-supervised 경로를 유지하면서도, residual 연결을 통해 $f_T$가 간접적으로 조정되도록 설계한 것이다.

이 구조의 직관은 다음과 같다. 완전히 새로운 target classifier를 무에서 학습하는 것보다, source classifier와 가까운 지점에서 작은 차이만 배우는 것이 더 쉽다. 논문도 residual learning의 원리를 빌려, 일반적으로

$$
|\Delta f(\mathbf{x})| \ll |f_T(\mathbf{x})| \approx |f_S(\mathbf{x})|
$$

형태가 기대된다고 설명한다. 즉 residual은 classifier 간의 작은 차이를 표현한다.

### Entropy minimization

그러나 residual 구조만으로는 target classifier가 target data 구조를 충분히 반영한다고 보장할 수 없다. residual이 너무 작아져 사실상 $f_S \approx f_T$가 되어버릴 위험도 있다. 이를 막기 위해 저자들은 target unlabeled data에 대해 entropy minimization을 추가한다.

$$
\min_{f_t} \frac{1}{n_t} \sum_{i=1}^{n_t} H(f_t(\mathbf{x}_i^t))
$$

여기서 엔트로피는
$$
H(f_t(\mathbf{x}_i^t)) = -\sum_{j=1}^{c} f_j^t(\mathbf{x}_i^t)\log f_j^t(\mathbf{x}_i^t)
$$
로 정의된다.

이 항은 target 예측을 더 confident하게 만들고, decision boundary가 데이터가 드문 low-density region을 통과하도록 유도한다. 쉽게 말하면, target 데이터 군집 구조에 맞춰 분류 경계를 정리하는 역할을 한다. 저자들은 residual module과 entropy penalty가 함께 사용되어야 의미가 있다고 강조한다. entropy가 없으면 residual이 쓸모없는 zero mapping으로 수렴해 source와 target classifier가 거의 같아질 수 있다고 본다.

### 최종 목적함수

RTN의 최종 학습 목적은 supervised source loss, target entropy penalty, 그리고 tensor MMD penalty를 함께 최소화하는 것이다.

$$
\min_{f_S = f_T + \Delta f} \frac{1}{n_s}\sum_{i=1}^{n_s} L(f_s(\mathbf{x}_i^s), y_i^s) + \frac{\gamma}{n_t}\sum_{i=1}^{n_t} H(f_t(\mathbf{x}_i^t)) + \lambda D_{\mathcal{L}}(\mathcal{D}_s,\mathcal{D}_t)
$$

여기서 $\lambda$는 feature adaptation 강도, $\gamma$는 entropy minimization 강도를 조절하는 하이퍼파라미터다.

이 목적함수는 의미가 분명하다. 첫 번째 항은 source에서 분류 성능을 유지하게 하고, 두 번째 항은 target에서 classifier를 구조적으로 정리하게 하고, 세 번째 항은 source와 target feature representation을 가깝게 만든다. 결국 RTN은 **source에서 배울 수 있는 것과 target에서 간접적으로 유도할 수 있는 것**을 함께 묶은 구조라고 볼 수 있다.

### 학습 절차

논문은 ImageNet으로 사전학습된 AlexNet을 출발점으로 사용한다. 모든 feature layer를 fine-tuning하고, 새로 추가한 bottleneck layer $fcb$, classifier layer $fcc$, residual layer $fc1$–$fc2$는 scratch에서 학습한다. 새 층들은 초기화 상태이므로 다른 층보다 learning rate를 10배 크게 설정한다.

최적화는 momentum 0.9를 갖는 mini-batch SGD로 수행한다. learning rate는 RevGrad에서 사용한 annealing schedule을 따른다.

$$
\eta_p = \frac{\eta_0}{(1+\alpha p)^\beta}
$$

여기서 $p$는 0에서 1까지 선형 증가하는 training progress이고, 논문에서 사용한 값은 $\eta_0 = 0.01$, $\alpha = 10$, $\beta = 0.75$다.

MMD penalty 최적화는 선형 시간 학습이 가능하도록 [5]의 기법을 따른다고만 설명되어 있으며, 본문 발췌 텍스트 안에는 그 세부 구현이 자세히 재현되어 있지는 않다.

## 4. 실험 및 결과

### 데이터셋과 설정

논문은 두 개의 표준 벤치마크에서 RTN을 평가한다.

첫 번째는 **Office-31**이다. 총 4,110장, 31개 클래스이며, 세 개 도메인 Amazon (A), Webcam (W), DSLR (D)로 구성된다. 평가 태스크는 총 6개 전이 태스크인 $A \rightarrow W$, $D \rightarrow W$, $W \rightarrow D$, $A \rightarrow D$, $D \rightarrow A$, $W \rightarrow A$다.

두 번째는 **Office-Caltech**다. Office-31과 Caltech-256의 공통 10개 클래스를 사용하며, 도메인 A, W, D, C 사이의 총 12개 전이 태스크를 구성한다. Office-31은 클래스 수가 많아 더 어렵고, Office-Caltech은 태스크 수가 많아 보다 넓게 편향을 볼 수 있다는 설명이다.

얕은 방법에는 DeCAF7 feature를 사용하고, deep adaptation 방법에는 원본 이미지를 사용했다.

### 비교 대상

비교 대상은 전통적 shallow transfer 방법인 TCA, GFK와, deep learning / deep adaptation 방법인 AlexNet, DDC, DAN, RevGrad다. 또한 제안법의 구성요소 기여를 보기 위해 다음 ablation도 수행한다.

RTN (mmd)는 tensor MMD만 추가한 버전이다. RTN (mmd+ent)는 여기에 entropy penalty를 추가한 버전이다. RTN (mmd+ent+res)는 최종 full model로 residual module까지 포함한다.

이 ablation 구성은 논문의 핵심 주장, 즉 “feature adaptation만으로는 부족하고 entropy와 residual classifier adaptation이 추가되어야 한다”를 검증하기 위한 설계다.

### 하이퍼파라미터와 학습 설정

MMD 기반 방법들은 Gaussian kernel을 쓰며 bandwidth $b$는 median heuristic으로 정한다. target 라벨이 없으므로 모델 선택이 어렵기 때문에, 저자들은 source labeled data에 대해 cross-validation을 하고, 추가로 $A \rightarrow W$ 태스크에서 target domain W의 각 클래스당 1개 라벨 샘플을 validation 용도로 사용해 파라미터를 고른 뒤, 다른 태스크에는 고정한다. 이는 엄밀한 의미에서 완전히 label-free model selection은 아니므로, 해석 시 약간 주의할 필요가 있다.

RTN은 모든 태스크에서 $\lambda = 0.3$, $\gamma = 0.3$을 사용했다고 보고한다.

### Office-31 결과

Office-31에서 평균 정확도는 다음과 같다.

TCA는 65.8, GFK는 66.7, AlexNet은 68.8, DDC는 69.3, DAN은 71.7, RTN (mmd)는 72.1, RTN (mmd+ent)는 72.9, RTN (mmd+ent+res)는 **73.7**이다.

즉, full RTN은 DAN보다 평균적으로 2.0포인트 높고, AlexNet보다 4.9포인트 높다. 특히 어려운 전이 태스크에서 개선이 크다. 예를 들어 $A \rightarrow W$에서 DAN은 68.5, RTN full은 **73.3**으로 4.8포인트 향상되었다. $A \rightarrow D$에서도 DAN 66.8 대비 RTN full 71.0으로 4.2포인트 향상되었다.

쉬운 태스크인 $D \rightarrow W$, $W \rightarrow D$에서는 이미 거의 포화 성능이기 때문에 개선 폭은 작지만, 여전히 RTN이 최고 또는 동급 최고 성능을 보인다. 이는 RTN이 특히 **domain gap이 큰 hard transfer task에서 강점**이 있음을 시사한다.

### Office-Caltech 결과

Office-Caltech에서도 full RTN이 가장 높은 평균 정확도인 **93.4**를 기록한다. DAN은 90.1이므로 평균 3.3포인트 향상이다.

몇몇 태스크를 보면 $A \rightarrow W$에서 DAN 91.8, RTN full 95.2이고, $A \rightarrow D$에서 DAN 91.7, RTN full 95.5다. $C \rightarrow W$에서는 DAN 90.6, RTN full 96.9로 큰 폭의 향상이 관찰된다. 즉 다양한 도메인 조합에서도 residual classifier adaptation이 유효하다는 결과다.

### Ablation 해석

논문이 제시한 ablation은 매우 중요하다. RTN (mmd)는 DAN보다 약간 낫다. 이것은 tensor MMD 설계가 multilayer adaptation에 효과적이며, 동시에 파라미터 선택을 단순화한다는 저자 주장과 맞아떨어진다.

RTN (mmd+ent)는 RTN (mmd)보다 확실히 더 좋다. 예를 들어 Office-31 평균은 72.1에서 72.9로 상승하고, Office-Caltech 평균은 90.0에서 92.6으로 크게 오른다. 이는 target unlabeled data의 cluster structure를 반영하는 entropy minimization이 classifier adaptation에 실질적으로 기여한다는 뜻이다.

마지막으로 RTN (mmd+ent+res)가 최고 성능을 보인다. 이는 residual classifier transfer가 실제로 의미 있는 추가 이득을 준다는 직접적 증거다. 저자들은 이 결과를 근거로, “source classifier와 target classifier를 동일하다고 두는 기존 가정은 실제로는 충분하지 않다”고 주장한다.

### 정성적 분석과 추가 논의

논문은 $A \rightarrow W$ 태스크에서 DAN과 RTN의 prediction embedding을 t-SNE로 시각화한다. DAN에서는 target category들이 충분히 잘 분리되지 않는 반면, RTN에서는 target classifier가 더 큰 class-to-class distance를 형성한다고 설명한다. 이는 target classifier가 target 구조에 더 잘 맞춰졌다는 정성적 근거다. 다만 발췌 텍스트에는 그림 자체의 수치나 개별 점 분포가 자세히 텍스트로 기술되지는 않아, 여기서는 저자 설명 수준만 해석할 수 있다.

또한 layer response 분석에서는 residual function $\Delta f(\mathbf{x})$의 응답 크기가 shortcut function인 $f_T(\mathbf{x})$보다 대체로 훨씬 작다고 보고한다. 이는 residual이 실제로 “작은 classifier gap”을 학습하고 있다는 논문의 가정을 뒷받침한다.

classifier shift 실험에서는 source와 target에 각각 라벨을 주고 별도로 학습한 classifier weight를 비교하여, 실제로 source classifier와 target classifier가 상당히 다르다는 점을 그림으로 보인다. 이것은 RTN의 문제 설정 자체를 정당화하는 근거다. 다만 이 실험은 target labeled data가 있는 분석용 진단 실험이지, 실제 unsupervised adaptation 학습 절차는 아니다.

마지막으로 entropy coefficient $\gamma$에 대한 민감도 실험에서는 성능이 bell-shaped curve를 보이며, 너무 작거나 너무 크면 성능이 떨어진다. 이는 feature alignment와 classifier confidence regularization 사이의 균형이 중요함을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **domain adaptation의 병목을 feature mismatch 하나로만 보지 않고 classifier mismatch까지 명시적으로 다뤘다**는 점이다. 이는 기존 deep domain adaptation 문헌의 강한 shared-classifier 가정을 완화하는 시도이며, 문제 정의 차원에서 분명한 가치가 있다.

둘째, residual learning을 classifier adaptation에 연결한 설계가 자연스럽고 구현 친화적이다. ResNet의 성공 원리를 “깊은 feature mapping”이 아니라 “source-target classifier bridge”에 재해석해 적용했다는 점이 인상적이다. residual block이 standard back-propagation 안에서 동작하므로 기존 딥러닝 프레임워크에 쉽게 붙일 수 있다는 실용성도 있다.

셋째, feature adaptation과 classifier adaptation이 서로 보완되도록 설계되었다. tensor MMD는 representation alignment를 담당하고, entropy minimization은 target unlabeled structure를 활용하며, residual block은 classifier gap을 메운다. ablation 결과도 이 세 모듈이 누적적으로 성능을 끌어올린다는 점을 잘 보여준다.

넷째, hard transfer task에서 성능 향상이 뚜렷하다. 이는 제안한 classifier adaptation이 실제 domain gap이 큰 상황에서 유효하다는 점을 시사한다.

반면 한계도 분명하다. 첫째, residual relation $f_S(\mathbf{x}) = f_T(\mathbf{x}) + \Delta f(\mathbf{x})$라는 가정은 합리적이지만, 모든 domain shift에서 성립한다고 보장할 수는 없다. source와 target classifier 차이가 “작은 perturbation” 이상으로 복잡한 경우에는 이 구조가 충분하지 않을 수 있다.

둘째, entropy minimization은 잘 알려진 semi-supervised / domain adaptation 기법이지만, cluster assumption이 약한 경우에는 오히려 잘못된 confident prediction을 강화할 위험이 있다. 논문은 성능 향상을 보였지만, target 데이터가 class-overlap이 큰 경우에도 안정적인지는 발췌 텍스트만으로는 알 수 없다.

셋째, model selection 과정이 완전히 엄격한 unsupervised 조건이라고 보기는 어렵다. 논문은 $A \rightarrow W$ validation을 위해 target domain에서 클래스당 1개 라벨을 사용했다고 기술한다. 이 방식은 실험 비교를 위해 당시 관행에 따랐다고 볼 수 있지만, 엄밀한 “무라벨 target” 설정과는 약간 긴장이 있다.

넷째, tensor MMD가 multilayer interaction을 잘 반영하는 장점은 있지만, tensor product와 bilinear pooling은 계산량과 표현 차원 측면에서 부담이 생길 수 있다. 논문은 선형 시간 최적화와 차원 축소를 언급하지만, 실제 계산 비용이나 메모리 이슈를 상세하게 비교하지는 않는다.

다섯째, 실험은 주로 Office 계열의 비교적 작은 벤치마크에 집중되어 있다. 당시 기준으로는 표준 벤치마크였지만, 더 큰 규모나 더 복잡한 semantic shift 환경까지 일반화된다고 단정하기는 어렵다. 이는 후속 연구가 검증해야 할 부분이다.

비판적으로 보면, 이 논문은 “classifier shift가 존재한다”는 중요한 메시지를 잘 제시했지만, 그 shift를 residual block으로 모델링하는 것이 유일하거나 최선의 방식이라는 점까지는 증명하지 않는다. 그럼에도 당시 deep adaptation 흐름에서 feature-level alignment를 넘어 classifier-level adaptation으로 시야를 넓혔다는 점에서 충분히 의미 있는 기여다.

## 6. 결론

이 논문은 unsupervised domain adaptation에서 **transferable feature와 adaptive classifier를 동시에 학습해야 한다**는 관점을 제시하고, 이를 실현하는 **Residual Transfer Network (RTN)** 를 제안했다. RTN은 세 가지 요소로 구성된다. source labeled data에 대한 supervised learning, tensor MMD 기반 feature adaptation, residual block과 entropy minimization을 통한 classifier adaptation이 그것이다.

핵심 기여는 두 가지로 요약할 수 있다. 하나는 multilayer feature를 tensor fusion한 뒤 MMD로 정렬하는 방식으로 feature adaptation을 수행한 점이다. 다른 하나는 source classifier와 target classifier의 차이를 residual function으로 모델링해, target 라벨 없이도 classifier mismatch를 부분적으로 교정하려 했다는 점이다. 실험 결과는 이 두 방향이 모두 유효하며, 특히 entropy minimization과 residual transfer를 함께 넣었을 때 가장 좋은 성능이 나온다는 점을 보여준다.

실제 적용 측면에서 이 연구는, 단순히 domain-invariant representation만 찾는 것으로 충분하지 않은 상황에서 classifier-level adaptation의 필요성을 제기했다는 점에서 중요하다. 이후의 domain adaptation 연구들이 conditional shift, decision boundary adaptation, classifier discrepancy 등에 더 관심을 갖게 되는 흐름과도 연결된다. 따라서 이 논문은 2016년 시점의 deep domain adaptation 문헌에서, **feature alignment 중심 접근을 classifier adaptation까지 확장한 중요한 과도기적 작업**으로 평가할 수 있다.
