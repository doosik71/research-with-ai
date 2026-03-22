# Weakly Supervised Instance Segmentation by Learning Annotation Consistent Instances

이 논문은 weakly supervised instance segmentation에서 흔히 쓰이던 “pseudo label 한 번 만들고, 그것을 정답처럼 두고 supervised instance segmentation model을 학습하는” 2단계 관행을 비판하고, 그 과정의 **불확실성(uncertainty)** 자체를 모델링해야 한다고 주장한다. 이를 위해 저자들은 annotation과 일관된 pseudo label을 생성하는 **conditional distribution** 과, annotation 없이 instance mask를 예측하는 **prediction distribution** 을 명시적으로 분리해 정의한다. 그리고 두 분포 사이의 차이를 줄이는 **joint probabilistic objective** 로 함께 학습한다. 특히 conditional distribution에는 semantic class-aware unary, boundary-aware pairwise, annotation-aware higher-order term을 넣어 더 정확한 pseudo instance를 샘플링하고, prediction side는 Mask R-CNN을 기반으로 한 annotation-agnostic 분포로 본다. 논문은 Pascal VOC 2012에서 image-level supervision과 bounding-box supervision 모두에서 당시 state-of-the-art를 보고한다.

## 1. Paper Overview

이 논문의 문제는 **instance-level pixel annotation 없이도** instance segmentation을 얼마나 잘 학습할 수 있는가이다. 완전 지도 학습 기반 instance segmentation은 강력하지만, per-instance mask annotation 비용이 매우 높다. 그래서 기존 연구들은 image-level label이나 bounding box 같은 값싼 weak annotation을 이용해 pseudo labels를 생성하고, 이를 기반으로 Mask R-CNN류 모델을 학습해 왔다.  

하지만 저자들이 보기에 기존 접근은 두 가지 핵심 약점이 있다.
첫째, weak annotation에서 pseudo label은 본질적으로 애매한데도, 기존 방식은 보통 **하나의 pseudo label만 정답처럼 고정**한다.
둘째, pseudo label generator와 segmentation network를 연결하는 **일관된 학습 목적함수**가 부족하다. 일부는 단순 2-step, 일부는 iterative refinement를 쓰지만, 왜 그 절차가 정당한지 이론적으로 깔끔하지 않다. 논문은 이 문제를 “두 분포를 따로 세우고, 그 사이의 dissimilarity를 줄이는 방식”으로 재정의한다.  

## 2. Core Idea

논문의 중심 아이디어는 다음과 같다.

> **weak annotation 아래에서 정답 instance mask는 하나가 아니라 여러 개일 수 있으므로, pseudo label을 단일 mask가 아니라 조건부 분포로 모델링하고, 이를 annotation-agnostic prediction distribution과 정합시키자.**  

즉, 이 논문은 weakly supervised instance segmentation을 다음 두 컴포넌트의 결합으로 본다.

1. **Conditional distribution**
   weak annotation이 주어졌을 때, 그와 일관적인 instance label들을 생성하는 분포

2. **Prediction distribution**
   테스트 시 실제로 써야 하는 instance segmentation model의 출력 분포

이때 novelty는 단순히 두 분포를 나눈 것에 그치지 않는다. conditional distribution은 semantic unary, boundary pairwise, annotation-consistent higher-order term을 결합해 더 나은 pseudo label을 샘플링하고, prediction distribution은 Mask R-CNN 기반으로 구현한다. 그리고 최종적으로는 이 둘 사이의 **dissimilarity coefficient** 를 줄이는 joint objective를 사용한다. 기존 pseudo-label-then-train 접근보다 훨씬 principled한 구성이다.  

## 3. Detailed Method Explanation

### 3.1 기본 설정과 표기

입력 이미지를 $\mathbf{x}$, weak annotation을 $\mathbf{a}$, segmentation proposal set을 $\mathcal{R}={r_1,\dots,r_P}$ 로 둔다. 논문은 class-agnostic object proposal로 **MCG** 를 사용하고, 각 proposal이 background 포함 $C+1$ 개 클래스 중 어디에 속하는지 label vector $\mathbf{y}$ 를 정의한다. 설명은 image-level annotation 기준으로 시작하지만, 저자들은 bounding box annotation으로도 쉽게 확장된다고 명시한다. 실제 실험도 둘 다 수행한다.

### 3.2 Conditional Distribution

논문의 핵심 첫 축은 conditional distribution이다. 이는 weak annotation이 주어진 상태에서 pseudo instance labels를 생성하는 분포다.

$$
\Pr_c(\mathbf{y}\mid \mathbf{x}, \mathbf{a}; \boldsymbol{\theta}_c)
$$

이 분포의 목적은 **annotation-consistent pseudo labels** 를 만들되, weak supervision의 애매함 때문에 가능한 여러 해를 모두 고려하는 것이다. 예를 들어 image-level label만 있으면 “person이 있다”는 정보는 있지만, 어떤 proposal이 진짜 person instance인지, 어디까지 foreground인지 여러 해석이 가능하다. 논문은 이 불확실성을 분포로 다룬다.

저자들에 따르면 conditional distribution은 세 항으로 구성된다.

* **semantic class-aware unary term**
  각 segmentation proposal의 class score를 주는 항
* **boundary-aware pairwise smoothness term**
  객체 전체를 덮도록 유도하는 항
* **annotation-aware higher-order term**
  weak annotation과 전역적으로 일관되도록 만드는 항  

higher-order term의 의미가 중요하다. image-level annotation에서는 “각 class label에 대해 적어도 하나의 corresponding proposal이 존재해야 한다”는 제약이고, bounding box supervision에서는 “각 box에 대해 충분히 겹치는 proposal이 있어야 한다”는 제약이 된다. 즉, 이 항이 weak annotation consistency를 전역적으로 강제한다.

### 3.3 Conditional Distribution을 직접 모델링하지 않는 이유

이 조건부 분포는 구조적으로 **non-factorizable** 이다. 특히 annotation consistency 같은 전역 제약은 proposal별로 독립 분해가 되지 않는다. 그래서 정규화 상수(partition function)를 정확히 계산하며 직접 모델링하는 것은 너무 비싸다. 이 문제를 해결하기 위해 논문은 **Discrete Disco Nets** 를 사용해 representative sample을 뽑는 방식으로 conditional distribution을 구현한다. 논문에서는 이것을 **conditional network** 라고 부른다.

구현 측면에서 conditional network는 **modified U-Net** 구조를 사용하며, noise sample을 추가 채널로 넣어 같은 입력 이미지에 대해서도 다양한 plausible pseudo labels를 샘플링할 수 있게 한다. appendix 구현 설명에 따르면 image-level supervision 실험에서는 ResNet-50, bounding-box supervision 실험에서는 ResNet-101 기반 설정도 검토한다. 또 conditional network 쪽에는 $K=10$ 개 샘플 설정을 사용했다고 밝힌다.  

### 3.4 Prediction Distribution

두 번째 축은 prediction distribution이다. 이는 test time에 실제로 사용하는 instance segmentation model의 출력 분포다. 논문은 이것을 **annotation-agnostic prediction distribution** 으로 본다. 직관은 간단하다. conditional distribution은 weak annotation을 추가로 보므로 더 나은 pseudo label을 생성할 수 있어야 하고, prediction distribution은 그 구조를 학습해 annotation 없이도 비슷한 출력을 내야 한다는 것이다.

구현은 **standard Mask R-CNN** 이다. appendix에 따르면 image-level annotation 실험에서는 ImageNet pretrained **ResNet-50**, bounding-box annotation 실험에서는 pretrained **ResNet-101** 을 prediction network backbone으로 사용했다. 즉, 이 논문의 novelty는 prediction model을 새로 설계한 것이 아니라, **weak supervision을 어떻게 probabilistically 연결하느냐** 에 있다.

### 3.5 Joint Learning Objective

논문의 가장 중요한 수식적 기여는 두 분포를 함께 묶는 학습 목적이다. 저자들은 task-specific loss $\Delta$ 를 이용해 두 분포 사이의 diversity / self-diversity를 정의하고, 이를 바탕으로 **dissimilarity coefficient** 를 최소화한다. 최종 objective는 다음처럼 주어진다.

$$
\theta_p^{\*}, \theta_c^{\*} = \operatorname*{arg min}_{\theta_p,\theta_c}
DISC*{\Delta}\left(\Pr_p(\theta_p), \text{Pr}_c(\theta_c)\right)
$$

즉, conditional distribution이 샘플링하는 annotation-consistent instance와 prediction distribution이 내는 instance mask가 task-specific loss 기준으로 가까워지도록 양쪽을 동시에 조정한다.

appendix snippet에 따르면 diversity 항은 예를 들어 prediction과 conditional 사이에 대해 다음 꼴로 정의된다.

$$
DIV_{\Delta}(\text{Pr}_p,\text{Pr}_c) =
\frac{1}{K}\sum_{k=1}^{K}\sum_{\mathbf{y}_p^{(i)}} \text{Pr}_p(\mathbf{y}_p^{(i)};\mathbf{\theta}_p),\Delta(\mathbf{y}_p^{(i)}, \mathbf{y}_c^k)
$$

그리고 conditional network의 self-diversity는

$$
DIV_{\Delta}(\text{Pr}_c,\text{Pr}_c) =
\frac{1}{K(K-1)}
\sum_{\substack{k,k'=1\\\\k'\neq k}}^K
\Delta(\mathbf{y}_c^k,\mathbf{y}_c^{k'})
$$

처럼 정의된다. 이 self-diversity 항은 conditional network가 한 이미지에 대해 다양한 plausible pseudo labels를 내도록 유도한다.  

### 3.6 Optimization

논문은 conditional network 쪽 objective가 argmax inference를 포함해 직접적으로는 non-differentiable하다고 설명한다. 그래서 appendix에서 unbiased approximate gradient를 유도하고, conditional network는 gradient descent 기반 알고리즘으로 최적화한다. snippet에서는 conditional network training algorithm과, 최종 update가 dissimilarity coefficient의 gradient를 따라 수행된다고 나온다.  

여기서 중요한 건 “학습 절차가 iterative하다”는 사실보다, 그 iterative update가 단순 heuristic이 아니라 **joint probabilistic objective를 근사적으로 최적화하는 과정**으로 제시된다는 점이다. 이것이 기존 iterative pseudo-label refinement와의 가장 큰 차이다.

## 4. Experiments and Findings

### 4.1 Dataset and Settings

실험은 **Pascal VOC 2012** 에서 수행되며, weak supervision으로 **image-level annotations** 와 **bounding-box annotations** 를 모두 다룬다. prediction network는 standard Mask R-CNN, conditional network는 modified U-Net 또는 대안적 ResNet-based conditional network를 사용한다. 구현 설명상 image-level에서는 ResNet-50, box supervision에서는 ResNet-101을 사용한다.  

### 4.2 Main Results

논문이 report하는 핵심 수치는 다음과 같다.

* **image-level annotations**

  * $50.9%\ \mathrm{mAP}^{r}_{0.5}$
  * $28.5%\ \mathrm{mAP}^{r}_{0.75}$

* **bounding-box annotations**

  * $32.1%\ \mathrm{mAP}^{r}_{0.75}$

그리고 abstract에서는 best baseline 대비 각각 **4.2% mAP$^r_{0.5}$**, **4.8% mAP$^r_{0.75}$** 향상을 강조한다. introduction에서는 bounding-box setting에서 SOTA 대비 **10% 이상** 향상이라고도 요약한다.

즉, 이 논문은 단지 “약간 좋아졌다”가 아니라, weakly supervised instance segmentation에서 **uncertainty-aware pseudo labeling + joint objective** 가 성능 차이로 이어진다는 것을 보여준다.

### 4.3 Comparison with Prior Methods

저자들은 기존 weakly supervised instance segmentation 방법들이 대개 pseudo label 생성 후 supervised model을 따로 학습하는 구조였다고 정리한다. Ahn et al.은 displacement field와 pixel affinity로 pseudo label을 만들고 Mask R-CNN을 학습하며, Laradji et al.은 MCG proposal에서 pseudo segmentation을 iterative sampling하고 supervised Mask R-CNN을 학습한다. 논문은 이들과 달리, pseudo label uncertainty를 명시적으로 모델링하고 두 분포를 하나의 objective 아래 묶는 점을 차별점으로 내세운다.

### 4.4 Ablation: Unary, Pairwise, Higher-order Terms

논문은 conditional distribution 안의 세 항이 각각 얼마나 중요한지 ablation도 수행한다. 결과 해석은 꽤 분명하다.

* **Unary only** 는 성능이 낮다.
  가장 discriminative region만 잡는 편향이 강하기 때문이다.
* **Unary + Pairwise** 는 성능이 크게 오른다.
  pairwise가 객체 전체를 더 잘 덮게 만들어 특히 높은 IoU threshold에서 이득이 크다.
* **Unary + Pairwise + Higher-order** 가 가장 좋다.
  higher-order가 annotation-consistent sample을 강제해 정확도를 더 높인다.

이는 논문의 직관과 정확히 맞는다. class-aware unary만으로는 CAM류 방법처럼 물체 일부만 잡기 쉽고, boundary-aware pairwise가 object extent를 넓히며, annotation-aware higher-order가 weak label consistency를 완성한다는 것이다.

### 4.5 Ablation: Diversity Terms

또 다른 중요한 ablation은 diversity/self-diversity 항이다. 논문은 두 diversity coefficient 모두 중요하며, 특히 **conditional network의 self-diversity** 가 상대적으로 더 중요하다고 말한다. self-diversity가 다양한 pseudo outputs를 생성하게 해 difficult case 대응과 overfitting 방지에 도움이 된다는 해석이다.

이 결과는 논문의 핵심 주장, 즉 “pseudo label uncertainty를 분포적으로 다뤄야 한다”는 메시지를 직접 지지한다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 weakly supervised instance segmentation의 병목을 정확히 짚었다는 점이다. 기존 방법은 pseudo label을 사실상 deterministic target처럼 취급했지만, 이 논문은 **pseudo label generation의 불확실성 자체가 문제의 본질**이라고 본다. 그리고 이를 conditional distribution으로 명시적으로 모델링한다.  

또 다른 강점은, pseudo label generator와 prediction model을 **하나의 coherent objective** 로 연결했다는 점이다. 단순 iterative training이 아니라 dissimilarity coefficient 최소화라는 해석을 주기 때문에, 방법이 더 분석 가능하고 다른 weak label setting으로도 확장 가능하다. 실제로 논문은 image-level과 bounding-box supervision 모두에 적용한다.  

마지막으로, pairwise와 higher-order term의 설계가 weak supervision 문제 구조와 잘 맞는다. unary만으로는 물체 일부만 잡는다는 CAM 계열의 고질적 약점을 pairwise와 higher-order가 보완한다.  

### Limitations

한계도 있다. 우선 이 방법은 **segmentation proposals(MCG)**에 의존한다. 즉, proposal quality가 pseudo label quality의 상한을 결정할 가능성이 높다. proposal set 자체에 적절한 instance가 없으면 conditional distribution이 아무리 좋아도 좋은 sample을 만들기 어렵다. 이 부분은 논문의 표기와 method 설명에서 자연스럽게 드러난다.

또한 conditional distribution은 전역 제약을 포함한 non-factorizable 구조라 직접 모델링이 어렵고, 결국 Discrete Disco Nets 기반 샘플링과 근사 gradient 최적화에 의존한다. 이는 elegant하지만, optimization complexity와 구현 복잡성 측면에서는 단순 2-step보다 부담이 있다.  

### Interpretation

비판적으로 보면, 이 논문은 “더 강한 segmentation backbone”을 제안한 것이 아니라 **weak supervision learning principle** 을 제안한 논문에 가깝다. 하지만 바로 그 점이 중요하다. weak supervision에서 중요한 것은 backbone 자체보다도, 약한 라벨과 latent mask 사이의 관계를 어떻게 모델링하느냐이기 때문이다.

이 논문의 진짜 메시지는 다음과 같다.

**weakly supervised instance segmentation에서는 pseudo label을 하나의 정답처럼 고정하기보다, annotation-consistent한 여러 가능한 해를 분포로 다루고, prediction model이 그 분포와 가까워지도록 학습해야 한다.**

## 6. Conclusion

이 논문은 weakly supervised instance segmentation을 위해 **conditional distribution** 과 **prediction distribution** 을 분리해 정의하고, 둘 사이의 **dissimilarity coefficient** 를 줄이는 joint probabilistic objective를 제안한다. conditional distribution은 semantic unary, boundary-aware pairwise, annotation-aware higher-order term을 통해 annotation-consistent pseudo labels를 샘플링하고, prediction distribution은 Mask R-CNN 기반 annotation-agnostic model로 구현된다.  

실험 결과, Pascal VOC 2012에서 image-level supervision으로 **50.9% mAP$^r_{0.5}$**, **28.5% mAP$^r_{0.75}$**, bounding-box supervision으로 **32.1% mAP$^r_{0.75}$** 를 기록하며 당시 SOTA를 달성했다. 또한 ablation은 pairwise, higher-order, self-diversity가 모두 중요함을 보여준다.

이 논문은 weak supervision에서 “pseudo label 생성 → supervised 학습”이라는 단순 파이프라인을 넘어, **uncertainty-aware distribution matching** 이라는 더 일반적이고 강력한 시각을 제시했다는 점에서 의미가 크다. 후속 연구 관점에서는 proposal-free formulation, end-to-end differentiable latent mask modeling, 또는 modern instance foundation model과의 결합이 자연스러운 확장 방향이다.
