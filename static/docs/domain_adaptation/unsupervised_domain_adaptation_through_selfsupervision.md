# Unsupervised Domain Adaptation through Self-Supervision

* **저자**: Yu Sun, Eric Tzeng, Trevor Darrell, Alexei A. Efros
* **발표연도**: 2019
* **arXiv**: [https://arxiv.org/abs/1909.11825](https://arxiv.org/abs/1909.11825)

## 1. 논문 개요

이 논문은 **unsupervised domain adaptation (UDA)** 문제를 다룬다. 즉, source domain에는 라벨이 있지만 target domain에는 라벨이 없고, 목표는 source에서 학습한 모델이 target에서도 잘 동작하도록 만드는 것이다. 컴퓨터 비전에서는 조명, 배경, 촬영 장치, 데이터 수집 환경 차이 때문에 동일한 클래스라도 source와 target의 분포가 달라지는 경우가 흔하다. 이런 분포 차이 때문에 source에서 높은 정확도를 보인 분류기가 target에서는 크게 성능이 떨어진다.

기존 UDA 연구의 핵심 철학은 대체로 같다. source와 target이 공유된 feature space에서 비슷한 표현을 갖도록 정렬(alignment)시키고, 동시에 source에서의 class discrimination을 유지하면, source에서 학습한 classifier가 target에도 일반화될 수 있다는 생각이다. 문제는 이 정렬을 구현하는 방법이다. 많은 기존 방법은 MMD나 domain discriminator 같은 discrepancy measure를 최소화하는 방식, 특히 adversarial learning에 의존해 왔다. 하지만 논문은 이런 minimax 최적화가 불안정하고 튜닝이 어렵다는 점을 강하게 비판한다.

저자들의 제안은 매우 단순하다. **source와 target 모두에서 같은 self-supervised task를 동시에 학습시키면, 그 task에 필요한 방향으로 두 도메인의 표현이 자연스럽게 가까워질 수 있다**는 것이다. target에는 진짜 class label이 없으므로 supervised main task를 직접 적용할 수는 없지만, self-supervised auxiliary task는 데이터 자체로부터 label을 만들 수 있다. 따라서 source에서는 원래의 supervised classification loss를, source와 target 모두에서는 self-supervised loss를 같이 학습하여 shared encoder가 두 도메인에서 공통된 표현을 형성하도록 만든다.

이 문제는 중요한 이유가 분명하다. 실제 응용에서는 target domain에 라벨이 없는 경우가 매우 많다. 예를 들어 시뮬레이션에서 학습한 모델을 실제 도로 환경에 배포하거나, 한 종류의 이미지 데이터셋에서 학습한 모델을 다른 카메라나 다른 스타일의 이미지로 옮기는 상황은 흔하다. 이 논문은 그러한 상황에서 **복잡한 adversarial alignment 없이도 self-supervision만으로 domain adaptation을 달성할 수 있다**는 점을 보여준다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 다음과 같다. **도메인 정렬을 직접 강제하지 말고, 두 도메인 모두에서 풀 수 있어야 하는 self-supervised task를 shared representation 위에 얹어 함께 학습시키면, 그 task를 잘 수행하기 위해 필요한 구조적 정보가 양쪽 도메인에서 공통적으로 학습되면서 표현이 정렬된다**는 것이다.

논문 Figure 1의 설명을 빌리면, source classifier만 학습했을 때는 source와 target의 feature cluster가 멀리 떨어져 있어서 source classifier가 target에 일반화되지 않는다. 하지만 회전 예측 같은 self-supervised task 하나를 양쪽 도메인에 함께 적용하면, 그 task와 관련된 방향으로는 두 도메인이 가까워진다. 여기에 여러 self-supervised task를 더하면 여러 방향에서 alignment가 일어나고, 결과적으로 source classifier가 target에도 적용 가능해진다.

기존 접근과의 차별점은 크게 세 가지다.

첫째, **adversarial discrepancy minimization을 쓰지 않는다.** 따라서 domain discriminator와 feature extractor 사이의 minimax game이 없고, 불안정한 학습이나 급격한 발산 문제를 피할 수 있다.

둘째, **self-supervision을 pre-training이 아니라 joint training으로 사용한다.** 많은 self-supervised learning 연구는 unlabeled data로 pre-training한 후 downstream task에 fine-tuning한다. 하지만 이 논문은 source supervised task와 auxiliary self-supervised task를 동시에 학습한다. 저자들은 target에서 self-supervised pre-training 후 source fine-tuning을 해도 거의 이득이 없다고 말한다.

셋째, **어떤 self-supervised task를 써야 adaptation에 적합한지에 대한 설계 원칙을 제시한다.** 핵심은 self-supervised label이 도메인 간의 “의미 없는 차이”에 의존하지 않아야 한다는 점이다. 예를 들어 밝기, 색감, 저수준 appearance 차이를 복원해야 하는 colorization이나 denoising autoencoding은 오히려 도메인 차이를 더 잘 보존하게 될 수 있으므로 부적절하다고 본다. 반면 rotation, flip, patch location prediction처럼 이미지의 구조적 성질을 묻는 분류형 과제는 더 적합하다고 주장한다.

## 3. 상세 방법 설명

전체 구조는 shared encoder $\phi$와 여러 개의 task-specific head $h_k$로 이루어진 **multi-task learning** 형태이다. $h_0$는 source label을 예측하는 주된 supervised head이고, $h_1, \dots, h_K$는 self-supervised task를 위한 head들이다. 중요한 점은 모든 head가 동일한 encoder $\phi$ 위에 얹힌다는 것이다. 따라서 각 task를 잘 풀기 위해 필요한 정보가 encoder 내부에 공통 표현으로 축적된다.

논문에서 $\phi$는 deep convolutional neural network이며, 각 head는 가능한 한 저용량(low-capacity)으로 둔다. 구체적으로 각 $h_k$는 보통 선형층 하나이며, task가 multiclass면 softmax, binary면 sigmoid를 뒤에 둔다. 저자들은 head를 작게 만들어야 task-specific shortcut보다 shared feature 학습이 유도된다고 본다. 즉, alignment의 핵심은 복잡한 head가 아니라 encoder가 학습하는 공통 표현에 있다.

### self-supervised task 설계

저자들은 세 가지 task를 선택했다.

**Rotation Prediction**은 이미지를 $0^\circ, 90^\circ, 180^\circ, 270^\circ$로 회전시킨 뒤 회전 각도를 맞히는 4-way classification이다.

**Flip Prediction**은 입력 이미지를 수직(vertical) 방향으로 뒤집을지 말지를 랜덤하게 정하고, 그 여부를 예측하는 binary classification이다. 일반적인 horizontal flip은 자연 영상에서 불변성이 desirable하므로 사용하지 않았다고 명시한다.

**Patch Location Prediction**은 이미지에서 패치를 잘라낸 뒤, 그 패치가 원래 어디에서 왔는지를 예측하는 task이다. 작은 이미지에서는 네 개 quadrant 중 하나를 맞히는 4-class classification이고, 큰 이미지 특히 segmentation에서는 연속 좌표를 회귀(regression)하는 2D regression으로 바뀐다.

이 과제들이 적합한 이유는 도메인 차이와 직교하는 정보를 요구하기 때문이다. 예컨대 source와 target이 단지 밝기 스케일만 다른 경우라도, 회전/flip/location 예측은 여전히 가능하다. 반면 colorization이나 denoising처럼 픽셀 재구성을 요구하는 task는 밝기나 appearance 차이에 민감하므로, adaptation에 필요한 불변 표현이 아니라 도메인 차이 자체를 학습할 위험이 있다.

### 손실 함수와 최적화 목표

source의 labeled dataset을 $S={(x_i, y_i)}_{i=1}^m$, target의 unlabeled dataset을 $T={x_i}_{i=1}^n$라고 두자.

주요 supervised task의 loss는 source에만 적용된다.

$$
\mathcal{L}_{0}(S;\phi,h_{0})=\sum_{(x,y)\in S} L_{0}(h_{0}(\phi(x)), y)
$$

여기서 $L_0$는 보통 분류용 cross-entropy loss로 이해하면 된다. 이 항은 source에서 class-discriminative한 표현을 유지하는 역할을 한다.

각 self-supervised task $k$는 입력 변환 $f_k$와 그에 따라 자동 생성되는 pseudo-label $\tilde{y}$를 갖는다. 이를 통해 source와 target 각각에서 self-supervised 데이터셋 $F_k(S)$와 $F_k(T)$를 만든다. 그러면 task $k$의 loss는 다음과 같다.

$$
\begin{aligned}
\mathcal{L}_{k}(S,T;\phi,h_{k}) = & \sum_{(f_k(x),\tilde{y})\in F_k(S)} L_k(h_k(\phi(f_k(x))), \tilde{y}) \\
& + \sum_{(f_k(x),\tilde{y})\in F_k(T)} L_k(h_k(\phi(f_k(x))), \tilde{y})
\end{aligned}
$$

이 식에서 가장 중요한 점은 **self-supervised loss가 source와 target 모두에 대해 동시에 계산된다는 것**이다. 저자들은 이것이 alignment에 결정적이라고 반복해서 강조한다. 만약 self-supervision을 target에만 적용하면, 그것은 semi-supervised learning식의 “추가 데이터 활용”에 가까워지고 alignment 효과는 약해진다.

최종 목적함수는 단순한 합이다.

$$
\min_{\phi, h_k} ;
\mathcal{L}_{0}(S;\phi,h_0) + \sum_{k=1}^{K}\mathcal{L}_{k}(S,T;\phi,h_k)
$$

즉, adversarial objective도 없고 minimax도 없다. 모든 항이 같은 방향, 즉 좋은 표현과 자연스러운 alignment를 향해 공동으로 작동한다. 저자들은 loss trade-off hyper-parameter $\lambda_k$도 실험해 보았지만 필요하지 않았다고 말한다. target validation label이 없는 환경에서 hyper-parameter 수를 줄이는 것은 실용적으로도 중요하다.

### 학습 절차

실제 학습은 SGD로 수행된다. 구현상 두 가지 방식이 가능하다. 모든 loss를 한 번에 더해 joint step을 할 수도 있고, 메모리 효율을 위해 self-supervised task별로 순차적으로 gradient step을 할 수도 있다. 저자들은 후자를 사용했다고 설명한다. 즉, 각 self-supervised task마다 source/target 혼합 배치를 샘플링해 loss를 계산하고 encoder와 해당 head를 업데이트한 뒤, 마지막에 source labeled batch로 supervised head를 업데이트한다. 이 순차적 구현은 메모리를 아끼며, test accuracy 차이는 보통 1% 미만이라고 한다.

또 하나의 중요한 구현 디테일은 **balanced batches**이다. source와 target의 데이터 수가 다를 수 있으므로, self-supervised batch를 만들 때는 절반은 source, 절반은 target에서 뽑아 두 도메인 중 하나가 지나치게 loss를 지배하지 않도록 한다. 이는 encoder가 특정 도메인에 치우친 표현을 학습하는 것을 막기 위한 장치다.

### 테스트 단계

테스트 시에는 self-supervised head들을 모두 버리고, 최종 예측은 오직 $h_0(\phi(x))$만 사용한다. 즉, self-supervision은 학습 중 alignment를 유도하는 보조 장치이며, 추론 시 추가 비용을 거의 남기지 않는다.

### 조기 종료와 하이퍼파라미터 선택 휴리스틱

UDA에서는 target validation label이 없기 때문에 early stopping과 model selection이 어렵다. 논문은 이를 별도의 기술적 공헌으로 강하게 주장하지는 않지만, 실용적인 heuristic을 제안한다.

먼저 source validation set과 target validation set의 feature mean distance를 계산한다.

$$
D(S',T';\phi)=\left|\frac{1}{m}\sum_{x\in S'}\phi(x) - \frac{1}{n}\sum_{x\in T'}\phi(x)\right|_2
$$

이는 사실상 linear kernel에서의 MMD와 같은 형태다. 저자들의 논리는 이렇다. 자신들의 방법은 이 discrepancy를 **직접 최적화하지 않기 때문에**, 오히려 그것을 모델 선택 지표로 활용할 수 있다는 것이다. 학습 도중 explicit objective가 아니므로 Goodhart’s law 식의 과적합 가능성이 상대적으로 줄어든다는 설명이다.

구체적으로, epoch별로 측정한 source validation error 벡터를 $\mathbf{w}=(w_1,\dots,w_T)$, mean distance 벡터를 $\mathbf{v}=(v_1,\dots,v_T)$라고 두고, 정규화된 합

$$
\mathbf{u}=\mathbf{v}/\min(\mathbf{v}) + \mathbf{w}/\min(\mathbf{w})
$$

를 만든다. 그리고 early stopping epoch는

$$
\operatorname*{argmin}_{t\in{1,\dots,T}} \mathbf{u}_t
$$

로 정한다. 직관적으로는 “source에서의 discrimination은 유지하면서 source-target representation mean distance는 줄어드는 시점”을 선택하는 것이다. 이는 엄밀한 이론 보장보다는 practical rule-of-thumb으로 제시된다.

## 4. 실험 및 결과

### 객체 인식(object recognition) 7개 벤치마크

논문은 여섯 개 데이터셋으로부터 일곱 개의 표준 UDA benchmark를 구성했다. 사용된 데이터셋은 MNIST, MNIST-M, SVHN, USPS, CIFAR-10, STL-10이다. 이 중 digits 계열과 natural scene 계열이 섞여 있다.

실험 비교 대상은 DANN, DRCN, DSN, kNN-Ad, PixelDA, ATT, $\Pi$-Model, ADDA, CyCADA, VADA, DIRT-T 등 당시 대표적인 UDA 방법들이다. 평가 지표는 target test accuracy(%)이다.

이 논문의 결과는 Table 2에 정리되어 있으며, self-supervised task 조합에 따라 결과가 달라진다. 표기에서 R은 rotation, L은 location, F는 flip을 뜻한다.

핵심 결과는 다음과 같다.

MNIST $\rightarrow$ MNIST-M에서는 R을 사용한 방법이 **98.9%**로 DIRT-T의 98.9%와 동률 수준이며, 매우 강한 성능을 보인다. source only가 44.9%였다는 점을 생각하면 adaptation 효과가 매우 크다.

USPS $\rightarrow$ MNIST에서는 R이 **90.2%**로 ADDA의 90.1%, source only의 81.4%를 넘어선다.

CIFAR-10 $\rightarrow$ STL-10에서는 R+L+F 조합이 **82.1%**를 기록해 VADA의 80.0%, source only의 75.6%보다 높다.

STL-10 $\rightarrow$ CIFAR-10에서는 R+L+F가 **74.0%**로 DIRT-T의 75.3%에 아주 근접하면서도 여러 baseline을 능가하거나 경쟁력 있는 수준을 보인다. 저자들은 특히 이 경우 base model이 경쟁 기법보다 더 약한데도 adaptation으로 큰 향상을 얻었다는 점을 강조한다.

저자들은 전체적으로 **7개 벤치마크 중 4개에서 state-of-the-art**를 달성했다고 주장한다. 특히 natural scene object recognition에서는 세 가지 self-supervised task를 함께 쓰는 것이 강력했고, source only 대비 약 9%의 추가 향상 위에 또 다른 9% 정도가 더해졌다고 해석한다.

반면 실패 사례도 분명히 보고한다. SVHN가 포함된 벤치마크에서는 성능이 좋지 않다. 예를 들어 MNIST $\rightarrow$ SVHN에서 R은 61.3%로 DIRT-T(IN)의 76.5%보다 낮고, SVHN $\rightarrow$ MNIST에서도 85.8%로 최고 수준인 99.4%와 큰 차이가 난다. 저자들의 설명에 따르면 SVHN는 house number 이미지라 주변부에 인접 숫자 조각이 많이 포함되는데, rotation task는 중심 숫자보다 주변부 단서를 보고 쉽게 맞힐 수 있는 shortcut을 제공한다. 즉, self-supervised task가 실제 semantic recognition과 맞지 않아 trivial solution으로 흐른 것이다. 이 실패 분석은 이 논문의 중요한 메시지 중 하나다. **아무 self-supervised task나 쓰면 되는 것이 아니라, application에 맞는 task 설계가 중요하다**는 것이다.

### 세그멘테이션 실험: GTA5 $\rightarrow$ Cityscapes

객체 인식 외에도 논문은 semantic segmentation adaptation을 평가한다. source는 GTA5의 synthetic driving scenes, target은 Cityscapes의 real dash-cam scenes다. 공통 19개 클래스에 대해 pixel-wise segmentation을 수행하며, 지표는 class별 IoU 및 평균 mIoU다.

Table 3에 따르면 source only의 mIoU는 **25.3**이고, 저자들의 방법은 **28.9**로 유의미한 향상을 보인다. 절대적으로 아주 높은 수치는 아니지만, self-supervised task가 원래 분류 중심으로 설계되었음을 감안하면 의미 있는 개선이다. 흥미로운 점은 CyCADA와의 결합 실험이다. CyCADA 단독은 **39.5** mIoU이고, 여기에 저자들의 방법을 더한 Ours + CyCADA는 **41.2** mIoU로 더 높다. 이는 pixel-level adaptation 후에도 representation-level self-supervised alignment가 추가 이득을 줄 수 있음을 보여준다.

클래스별로 보면 road, sidewalk, building, sky, car 같은 주요 클래스에서 source only 대비 개선이 보인다. 예를 들어 road는 28.8에서 69.9, building은 39.6에서 69.7, sky는 45.8에서 58.4로 상승한다. 반면 일부 클래스는 감소하거나 거의 개선이 없기도 하다. 예컨대 rider, bus, motorbike, bicycle 등은 상대적으로 약하거나 불안정하다. 이는 self-supervised task가 segmentation의 highly local한 구조와 완전히 맞아떨어지지 않기 때문으로 볼 수 있다. 저자들도 이 부분을 인정하며, segmentation에 더 적합한 self-supervised task 개발이 앞으로 필요하다고 말한다.

### 구현 세부사항

객체 인식에서는 26-layer pre-activation ResNet을 사용했고, SGD with momentum 0.9, weight decay $5\times 10^{-4}$, batch size 128, 초기 learning rate 0.1을 사용했다. 공정 비교를 위해 data augmentation은 사용하지 않았다고 명시한다.

세그멘테이션에서는 DeepLab-v3를 ImageNet pre-training으로 초기화하고, self-supervised head 앞에 global average pooling과 linear layer를 두었다. location prediction은 연속 좌표 회귀로 바꾸고, $400\times400$ patch를 랜덤 crop하여 square loss로 좌표를 회귀했다. 학습은 learning rate 0.007, 15,000 iterations, batch size 48로 진행했다.

또한 Figure 3에서 저자들은 학습 곡선이 source task, target task, self-supervised task 모두에서 부드럽게 수렴하고, feature centroid distance도 함께 줄어드는 모습을 보여준다. 이 점은 adversarial learning에서 자주 보이는 불안정성과 대비되는 실용적 장점으로 제시된다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 설정과 방법 설계가 매우 잘 맞아떨어진다**는 점이다. UDA의 핵심이 alignment라는 점을 분명히 하고, 그 alignment를 adversarial discrepancy minimization이 아니라 self-supervised joint learning으로 달성할 수 있다는 새로운 관점을 제시한다. 이 아이디어는 복잡한 수학적 장치보다도 학습 구조 자체를 바꾸는 방향이라 직관적이고 구현도 쉽다.

두 번째 강점은 **안정성과 단순성**이다. 최종 목적함수는 여러 loss의 단순 합이며, 모든 항이 같은 방향으로 encoder를 업데이트한다. 따라서 minimax dynamics가 없고, 논문도 smooth convergence를 실험적으로 보여준다. 실제 현업 관점에서도 구현 난이도와 디버깅 부담이 비교적 낮을 가능성이 크다.

세 번째 강점은 **task design에 대한 통찰**이다. self-supervised learning을 adaptation에 가져오면서도, 어떤 task가 domain-invariant representation 학습에 유리한지 명시적으로 논의한다. reconstruction task가 왜 부적절할 수 있는지, classification-style structural task가 왜 더 적합한지 설득력 있게 설명한다.

네 번째 강점은 **조합 가능성**이다. CyCADA 같은 pixel-level adaptation과 결합했을 때 추가 향상이 나온다는 결과는 이 방법이 단독 기법일 뿐 아니라 다른 adaptation pipeline의 모듈로도 활용 가능함을 시사한다.

반면 한계도 분명하다.

가장 중요한 한계는 **성능이 self-supervised task의 적합성에 크게 의존한다**는 점이다. SVHN 실패 사례는 이 방법의 취약점을 잘 보여준다. auxiliary task가 진짜 semantic factor 대신 주변부 artifact나 trivial cue를 학습하게 되면 오히려 adaptation에 해가 될 수 있다. 따라서 이 방법은 보편 만능 해법이라기보다, 도메인 지식에 기반해 task를 신중히 설계해야 하는 프레임워크에 가깝다.

둘째, **이론적 보장은 제한적**이다. 왜 self-supervised joint training이 실제로 alignment를 보장하는지에 대한 강한 이론은 제공되지 않는다. 저자들도 implicit regularization이나 SGD의 성질을 언급하지만, 이는 주로 경험적 정당화다. 즉, 모델이 도메인별로 다른 decision boundary를 내부적으로 따로 학습하지 않을 것이라는 점은 엄밀히 증명되지 않는다.

셋째, **early stopping heuristic은 실용적이지만 원리적으로는 약하다**. mean distance와 source validation error를 조합한 선택 규칙은 합리적이지만, 통계적 최적성을 보장하지 않는다. 또한 unlabeled target validation set은 필요하므로 완전히 정보가 없는 설정은 아니다.

넷째, 세그멘테이션 결과는 의미 있는 개선이지만 **state-of-the-art를 압도하는 수준은 아니다**. 저자들도 자신들의 self-supervised task가 분류를 위해 설계되었다고 인정하며, segmentation에서는 local structure에 더 적합한 auxiliary task가 필요함을 시사한다.

비판적으로 보면, 이 논문의 핵심 공헌은 “새로운 self-supervised task” 자체라기보다는 **UDA를 self-supervised multi-task joint training의 관점으로 재해석한 것**에 있다. 따라서 이 논문의 가치 평가는 task engineering 성과와 학습 프레임워크의 아이디어를 구분해서 보는 것이 좋다. 제공된 텍스트 기준으로는, 이 논문은 후자에서 특히 강하다.

## 6. 결론

이 논문은 unsupervised domain adaptation을 위해 adversarial discrepancy minimization에 의존하지 않고, **source의 supervised task와 source+target의 self-supervised task를 함께 학습하는 단순한 multi-task framework**를 제안한다. 핵심은 self-supervised task가 두 도메인 모두에서 공통 구조를 학습하도록 만들어 shared encoder의 표현을 자연스럽게 정렬시키는 것이다.

주요 기여는 세 가지로 요약할 수 있다. 첫째, self-supervision과 UDA 사이의 연결고리를 분명히 제시했다. 둘째, adaptation에 적합한 self-supervised task 설계 원칙을 설명했다. 셋째, 단순하고 안정적인 학습 절차로 여러 표준 벤치마크에서 매우 강한 성능을 보였고, 특히 7개 객체 인식 벤치마크 중 4개에서 state-of-the-art를 달성했다.

실제 적용 측면에서도 의미가 크다. adversarial adaptation이 지나치게 복잡하거나 불안정한 상황에서, 이 방법은 비교적 구현이 쉽고 다른 방법과 결합도 가능하다. 향후 연구에서는 segmentation이나 detection처럼 더 구조적인 문제에 맞는 self-supervised task를 설계하거나, 작은 target sample size 환경에서의 장점을 체계적으로 검증하는 방향이 유망해 보인다. 논문 자체도 이 지점을 미래 연구 방향으로 제시한다.

전반적으로 이 논문은 “어떻게 discrepancy를 직접 줄일 것인가”에서 “어떤 공통 과제를 함께 풀게 하면 표현이 자연스럽게 정렬되는가”로 시각을 전환한 작업이다. 제공된 텍스트만 기준으로 보았을 때, 이 전환은 개념적으로 명확하고 실험적으로도 상당히 설득력 있다.
