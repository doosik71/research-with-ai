# A Closer Look at Rehearsal-Free Continual Learning

* **저자**: James Seale Smith, Junjiao Tian, Shaunak Halbe, Yen-Chang Hsu, Zsolt Kira
* **발표연도**: 2022
* **arXiv**: [https://arxiv.org/abs/2203.17269](https://arxiv.org/abs/2203.17269)

## 1. 논문 개요

이 논문은 **rehearsal-free continual learning**을 다시 면밀히 들여다보는 연구이다. 문제 설정은 class-incremental continual learning으로, 시간이 지나며 새로운 클래스들이 순차적으로 들어오지만 과거 데이터를 저장하지 못하는 상황을 다룬다. 이때 모델은 새 클래스를 배워야 할 뿐 아니라, 이미 배운 과거 클래스에 대한 성능도 유지해야 한다. 그러나 새 데이터만으로 학습하면 기존 지식을 덮어써 버리는 **catastrophic forgetting**이 발생한다.

기존의 강한 성능을 보이는 class-incremental continual learning 방법들은 대체로 과거 샘플 일부를 저장해 섞어 학습하는 **rehearsal**에 의존한다. 하지만 이 방식은 메모리 비용이 크고, 개인정보나 민감 데이터의 장기 저장이 불가능한 실제 응용에서는 적용이 어렵다. 저자들은 바로 이 지점에 주목하여, **과거 데이터를 전혀 저장하지 않으면서도 forgetting을 줄일 수 있는 단순하고 고전적인 regularization 기법들을 다시 평가**한다.

이 논문의 핵심 문제는 다음과 같이 정리할 수 있다. rehearsal 없이 class-incremental continual learning을 수행할 때, 과거 지식을 보존하는 데 더 효과적인 제약은 무엇인가? 모델 파라미터를 직접 묶는 **parameter regularization**이 더 나은가, 아니면 과거 모델의 출력이나 feature를 유지하게 하는 **knowledge distillation / feature regularization**이 더 나은가? 그리고 **pre-training**이 주어졌을 때 그 결론이 달라지는가? 저자들은 이 질문에 대해 체계적인 ablation과 representation 분석을 통해 답한다.

이 문제의 중요성은 분명하다. 실제 서비스 환경에서는 데이터가 지속적으로 들어오지만, 이전 데이터를 무기한 보관하기 어렵다. 따라서 rehearsal-free 성능을 높이는 것은 단순한 학술적 관심이 아니라, privacy와 memory constraints가 있는 현실적 응용을 위한 핵심 과제다. 이 논문은 완전히 새로운 알고리즘을 제안한다기보다, 널리 알려진 기법들의 역할을 다시 분해해 보고, 어떤 조건에서 무엇이 실제로 잘 작동하는지를 정리한다는 점에서 가치가 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 **rehearsal-free continual learning에서 forgetting은 단일 원인으로 설명되지 않으며, feature encoder와 classifier head에서 일어나는 문제가 다르다**는 관찰에 있다. 저자들은 기존 continual learning 문헌에서 자주 함께 언급되지만 역할이 섞여 있던 여러 regularization 기법을 분해해서 본다. 그 결과, 단순히 “어떤 방법이 더 좋다”가 아니라 **어떤 부분을 보호하는 데 어떤 regularization이 더 적합한가**를 밝히려 한다.

구체적으로, 저자들은 세 가지 지식 이전 방식을 비교한다. 첫째는 파라미터가 과거 체크포인트에서 너무 멀어지지 않도록 하는 **parameter regularization**이다. 여기에는 Fisher 정보로 가중한 EWC와, 단순한 거리 제약인 L2가 포함된다. 둘째는 과거 모델의 예측을 유지하도록 하는 **prediction distillation (PredKD)**이다. 셋째는 중간 표현을 유지하도록 하는 **feature distillation (FeatKD)**이다. 이 셋은 모두 과거 모델 $\theta_{n-1}$를 현재 모델 $\theta_n$의 teacher 또는 anchor로 사용한다는 공통점이 있다.

논문이 제시하는 중요한 통찰은 다음과 같다.

첫째, **prediction distillation은 classifier 쪽의 지식을 유지하는 데 특히 중요**하다. rehearsal-free setting에서는 과거 클래스 예시가 없기 때문에, 새 데이터만 softmax classification으로 학습하면 과거 클래스 logits가 구조적으로 불리해지고 최근 task 쪽으로 bias가 생긴다. 저자들은 이를 완화하기 위해 softmax CE 대신 **sigmoid binary cross-entropy (BCE)**를 사용하고, 여기에 PredKD를 결합하면 rehearsal-free에서도 강한 baseline이 된다고 주장한다.

둘째, **parameter regularization과 feature distillation은 주로 feature drift를 줄이는 역할**을 한다. 즉, encoder 내부 표현이 과거 task에서 배운 상태에서 크게 벗어나지 않도록 잡아준다. 하지만 이것만으로는 classifier의 bias를 충분히 제어하지 못하므로, prediction-space regularization 없이 단독으로 쓰면 class-incremental setting에서 성능이 매우 낮다.

셋째, **pre-training이 있으면 결론이 바뀐다**. 랜덤 초기화에서는 parameter regularization이 forgetting은 줄여도 plasticity를 떨어뜨려 최종 정확도가 낮아질 수 있다. 하지만 ImageNet pre-training이 있으면, 모델이 이미 유용한 feature를 갖고 시작하므로 낮은 plasticity가 큰 문제가 아니게 되고, 오히려 **L2 parameter regularization이 가장 좋은 선택**이 된다. 이 결과는 “복잡한 EWC가 단순한 L2보다 항상 낫다”는 일반적 직관에 반하는 흥미로운 발견이다.

넷째, 이 관찰은 CNN 실험에 그치지 않는다. 저자들은 ViT에서 self-attention의 QKV projection만 미세조정하고 여기에 L2 regularization을 거는 단순한 방식이, 최근의 prompting 기반 continual learning 방법들보다 더 좋은 성능을 낼 수 있음을 보인다. 즉, 이 논문은 **복잡한 prompt 설계보다 적절한 fine-tuning regularization이 더 강력할 수 있다**는 메시지를 준다.

## 3. 상세 방법 설명

이 논문은 새로운 아키텍처를 도입하기보다, 기존 continual learning 목적함수들을 rehearsal-free 관점에서 체계적으로 비교·결합한다. 기본 설정은 task가 순차적으로 들어오는 class-incremental learning이다. 전체 클래스는 $c_1, c_2, \dots, c_M$로 구성되며, 각 task $\mathcal{T}_n$는 이전 task와 겹치지 않는 클래스 집합을 가진다. 각 클래스는 오직 한 번만 등장한다. task $n$까지 학습한 모델을 $\theta_n$이라 두고, task 전환 시점마다 직전 모델 $\theta_{n-1}$를 복사해 **frozen checkpoint model**로 둔다. 이후 task $n$ 학습에서는 $\theta_{n-1}$의 지식을 $\theta_n$으로 전달하는 것이 핵심이 된다.

### 3.1 전체 파이프라인의 관점

task $n$에서 현재 모델은 새 task 데이터만 본다. 과거 데이터는 저장하지 않는다. 따라서 과거 지식을 유지하려면, 이전 모델 $\theta_{n-1}$ 자체를 일종의 reference로 활용해야 한다. 논문은 Figure 1에서 세 가지 지식 이전 경로를 제시한다.

하나는 **parameter space regularization**이다. 현재 모델의 파라미터가 이전 체크포인트 파라미터에서 너무 멀어지지 않게 하는 방식이다.
둘은 **prediction distillation**이다. 현재 입력 $x$에 대해 이전 모델이 내놓던 class prediction을 현재 모델이 유지하게 한다.
셋은 **feature distillation**이다. prediction 대신 중간 layer의 feature representation을 유지하게 한다.

중요한 점은, distillation에 사용되는 입력 $x$가 **현재 task의 새 데이터만으로도 가능하다**는 것이다. 즉, 과거 샘플을 다시 보여주지 않아도 현재 task 데이터에 대해 teacher와 student 출력을 맞추게 만들 수 있으므로 rehearsal-free라고 부를 수 있다.

### 3.2 Parameter Space Regularization: EWC와 L2

EWC는 오래된 continual learning 방법 중 하나로, 파라미터 변화에 패널티를 부과한다. 논문에서 제시된 손실은 다음과 같다.

$$
\mathcal{L}_{ewc}=\sum*{j=1}^{N_{params}}F_{n-1}^{jj}\left(\theta_n^j-\theta_{n-1}^j\right)^2
$$

여기서 $F_{n-1}^{jj}$는 이전 task에서 계산한 Fisher Information Matrix의 대각 원소이다. 직관적으로는, 이전 task 성능에 더 중요하다고 판단된 파라미터일수록 더 강하게 보존하려는 것이다.

이 식에서 $F$가 항등행렬이라면 손실은 단순한 L2 regularization이 된다. 즉,

$$
\mathcal{L}_{L2} \propto \sum_j \left(\theta_n^j-\theta*{n-1}^j\right)^2
$$

와 같은 형태다. EWC는 “중요한 파라미터를 더 강하게 지키는” 정교한 방법이고, L2는 “모든 파라미터를 균등하게 과거 상태 근처에 묶는” 단순한 방법이라고 볼 수 있다.

저자들이 특히 강조하는 포인트는, **L2는 task 1에서도 적용 가능하다**는 점이다. EWC는 이전 task 데이터로 Fisher를 계산해야 하므로 task 1 시작 시점에는 적용이 어렵다. 반면 pre-trained initialization이 있는 경우 L2는 처음부터 pre-trained 상태 근처에 머물도록 만들 수 있다. 이 점이 뒤의 ViT 실험과 pre-training 실험에서 중요한 차이를 만든다.

### 3.3 Prediction Distillation

Prediction distillation은 Learning without Forgetting (LwF) 계열의 접근이다. 입력 $x$에 대해 이전 모델과 현재 모델이 과거 클래스에 대해 유사한 예측을 하도록 만든다. 논문은 다음과 같이 정의한다.

$$
\mathcal{L}_{PredKD}=CE\left(p*{\theta_{n,1:n-1}}(x),, p_{\theta_{n-1,1:n-1}}(x)\right)
$$

여기서 $p_\theta(y \mid x)$는 모델 $\theta$가 입력 $x$에 대해 내는 class distribution이다. teacher는 이전 모델 $\theta_{n-1}$이고, student는 현재 모델 $\theta_n$이다. 중요한 것은 current task 데이터 $x$만으로도 teacher prediction을 계산할 수 있다는 점이다.

이 방법은 특히 **classifier head와 prediction space의 drift를 제어**하는 데 강하다. 과거 데이터가 없을 때는 현재 데이터에 대해 기존 클래스 확률이 급격히 무너질 수 있는데, PredKD는 적어도 teacher가 과거 클래스에 대해 갖고 있던 응답 패턴을 유지하도록 돕는다. 논문의 결과에서 PredKD는 rehearsal-free setting의 핵심 baseline 역할을 한다.

### 3.4 Feature Distillation

Feature distillation은 intermediate representation을 직접 맞춘다. 논문은 다음 손실을 사용한다.

$$
\mathcal{L}_{FeatKD}= \left|\theta_n^l(x)-\theta*{n-1}^l(x)\right|_2^2
$$

여기서 $\theta_n^l(x)$는 layer $l$에서의 feature를 뜻한다. prediction과 달리 intermediate feature에는 class probability처럼 명시적 의미가 없으므로, cross-entropy가 아니라 L2 squared error를 사용한다.

직관적으로 FeatKD는 representation이 과거 모델에서 멀어지지 않도록 하여 **feature drift를 억제**한다. 다만 논문은 이 방법이 의외로 최종 정확도 향상에 거의 도움을 주지 못하며, 오히려 plasticity를 떨어뜨려 성능을 낮출 수 있음을 보인다.

### 3.5 Task Bias와 BCE 분류기

이 논문의 실질적으로 매우 중요한 설계는 **softmax CE 대신 sigmoid BCE를 사용하는 것**이다. 저자들은 rehearsal-free class-incremental setting에서 softmax classification이 최근 task 쪽으로 심한 bias를 만든다고 본다. 새 task 데이터만으로 softmax loss를 최소화하면, 현재 존재하지 않는 과거 클래스 logits는 자연스럽게 억눌리게 된다. 이는 classifier head 관점에서 과거 클래스에 불리한 구조를 만든다.

이를 완화하기 위해 저자들은 LWF.MC에서 차용한 방식처럼 **multi-class sigmoid BCE**를 사용한다. 논문은 이 설계가 PredKD와 결합될 때 EWC, L2, FeatKD 같은 방법들이 “실패한다”는 기존 인식을 상당 부분 뒤집는다고 주장한다. 즉, 이 논문이 보여주는 중요한 메시지 중 하나는, **regularization 자체보다도 classifier formulation이 rehearsal-free setting에서 매우 중요하다**는 점이다.

### 3.6 평가 지표

논문은 최종 정확도와 forgetting을 함께 본다. 우선 task $n$에 대한 local task accuracy는 다음과 같이 정의된다.

$$
A_{i,n}=\frac{1}{|\mathcal{D}_{n}^{test}|}\sum*{(x,y)\in\mathcal{D}_{n}^{test}} \mathbf{1}(\hat{y}(x,\theta*{i,n})=y \mid \hat{y}\in\mathcal{T}_n)
$$

이는 task label이 주어졌다고 가정하는 task-incremental 관점의 정확도다. 반면 global accuracy는 task label 없이 지금까지 본 모든 클래스 중에서 맞혀야 하는 class-incremental 관점이다. 논문에서는 최종 시점의 global accuracy $A_{N,1:N}$를 핵심 성능 지표로 사용하고, 표에서는 이를 $A_{1:N}$로 줄여 적는다.

global forgetting은 다음과 같이 정의된다.

$$
F_N^G=\frac{1}{N-1}\sum_{i=2}^{N}\sum_{n=1}^{i-1}\frac{|\mathcal{T}_n|}{|\mathcal{T}_{1:i}|}(R_{n,n}-R_{i,n})
$$

여기서

$$
R_{i,n}=\frac{1}{|\mathcal{D}_{n}^{test}|}\sum*{(x,y)\in\mathcal{D}_{n}^{test}} \mathbf{1}(\hat{y}(x,\theta*{i,1:n})=y)
$$

이다. 이는 task label이 없는 진짜 class-incremental 조건에서, 과거 task 성능이 얼마나 감소했는지를 측정한다.

local forgetting은 다음과 같다.

$$
F_N^L=\frac{1}{N-1}\sum_{n=1}^{N-1}(A_{N,n}-A_{n,n})
$$

이 지표는 task label이 주어진 조건에서 representation 자체가 얼마나 무너졌는지를 보는 데 가깝다. 저자들은 global forgetting과 local forgetting을 모두 봄으로써, **classifier bias 문제와 feature drift 문제를 분리해서 해석**하려 한다.

### 3.7 CKA 분석의 역할

논문은 단순 accuracy 비교를 넘어, hidden representation이 task가 진행됨에 따라 어떻게 변하는지 보기 위해 **Centered Kernel Alignment (CKA)** 분석을 수행한다. task 1 데이터에 대해 $\theta_1$과 $\theta_n$의 각 layer representation 유사도를 측정하고, 이를 통해 어느 layer에서 drift가 크게 일어나는지 본다. 결과적으로 forgetting이 주로 **후반부 layer, 특히 penultimate layer와 linear head 근처에서 크게 발생**하며, parameter regularization은 encoder representation 안정화에는 도움을 주지만 plasticity를 희생시킬 수 있다는 해석으로 이어진다.

## 4. 실험 및 결과

### 4.1 기본 설정

가장 먼저 CIFAR-100 10-task 실험을 수행한다. CIFAR-100은 100개 클래스를 가진 $32 \times 32 \times 3$ 이미지 데이터셋이며, 이를 task당 10개 클래스씩 10개 task로 나눈다. backbone은 18-layer ResNet이다. 학습은 250 epoch 동안 Adam optimizer로 수행하고, learning rate는 $1e{-3}$에서 시작해 100, 150, 200 epoch 뒤에 10배씩 줄인다. weight decay는 0.0002, batch size는 128이다.

저자들은 continual learning 원칙상 전체 task를 본 뒤 hyperparameter를 튜닝하는 것은 부적절할 수 있다고 지적한다. 그래서 전체 task sequence가 아니라 각 데이터셋의 작은 task sequence에 대해 loss weight를 sweep해서 정했다. 이는 실험 설계의 공정성을 의식한 부분이다.

비교 대상은 크게 다음과 같다.
첫째, classification loss만 사용하는 **Naive**.
둘째, 모든 task의 데이터를 한꺼번에 학습하는 **Upper-Bound**.
셋째, regularization 계열인 **PredKD, FeatKD, EWC, L2**, 그리고 그 조합들이다.

### 4.2 Rehearsal-free setting에서의 핵심 결과

Table 1과 Table 2는 rehearsal-free CIFAR-100 10-task 결과를 보여준다. 가장 중요한 메시지는 **PredKD와 BCE가 rehearsal-free continual learning의 핵심 기반**이라는 점이다.

Table 1에서 EWC나 FeatKD를 단독으로 쓰면 성능이 매우 낮다. 예를 들어 EWC(BCE)는 최종 정확도 $A_{1:N}=7.7$이고, EWC(Soft)도 $7.3$에 불과하다. FeatKD 역시 BCE나 Soft 모두 약 8% 수준이다. 이는 class-incremental global accuracy 기준으로 사실상 실패에 가깝다.

하지만 여기에 **PredKD를 결합**하면 상황이 크게 달라진다.
PredKD + EWC(BCE)는 $A_{1:N}=22.7$까지 올라가고,
PredKD + FeatKD(BCE)는 $19.1$을 기록한다.
즉, feature drift를 줄이는 regularization만으로는 부족하고, **prediction-level knowledge transfer가 반드시 같이 필요**하다는 것이 드러난다.

더 나아가 Table 2를 보면, 단독 PredKD가 오히려 가장 높은 최종 정확도 $25.2$를 기록한다. 반면 PredKD + EWC는 최종 정확도는 조금 낮은 $22.7$이지만 global forgetting $F_N^G=-0.7$로 가장 낮다. PredKD + L2도 $A_{1:N}=21.6$, $F_N^G=1.6$으로 낮은 forgetting을 보인다. 이 결과는 저자들의 해석과 일치한다. 즉, **parameter regularization은 forgetting을 줄이지만, plasticity를 억제해 새로운 task 학습 성능을 떨어뜨릴 수 있다**.

FeatKD는 예상보다 실망스럽다. PredKD + FeatKD는 PredKD 단독보다 최종 정확도가 낮고, forgetting 측면에서도 뚜렷한 개선이 없다. 저자들은 이를 바탕으로, feature alignment가 직관적으로는 좋아 보이지만 실제 class-incremental rehearsal-free 조건에서는 classifier bias 문제를 충분히 해결하지 못한다고 본다.

### 4.3 CKA 분석으로 본 forgetting의 위치

Figure 2의 CKA 분석은 매우 중요한 보조 근거다. task 1 데이터에 대해, 이후 task를 거치면서 각 layer representation이 얼마나 유지되는지를 비교한다. 저자들은 linear output, penultimate layer, 그리고 그 이전의 여러 late layer를 추적한다.

이 그림의 해석은 다음과 같다. PredKD는 최종 정확도가 가장 높지만, representation similarity를 전 layer에서 가장 강하게 유지하는 것은 아니다. 반대로 PredKD + EWC와 PredKD + L2는 task 1 표현을 더 잘 보존해 forgetting 지표는 낮지만, 그만큼 새 task 학습을 위한 parameter 이동이 억제되어 최종 정확도는 낮다. 즉, continual learning에서는 **representation retention과 final accuracy 사이에 trade-off**가 존재한다.

또한 forgetting은 네트워크 초반부보다 **뒷부분 layer에서 더 강하게 나타난다**. 이는 feature extractor 전체를 균일하게 보호하는 것보다, 후반부 표현과 classifier에 대한 적절한 제어가 중요함을 시사한다.

### 4.4 Pre-training을 사용할 때의 변화

Table 3은 ImageNet1k pre-training으로 초기화한 뒤 같은 CIFAR-100 continual learning을 수행한 결과다. 여기서 논문의 가장 흥미로운 결과 중 하나가 나온다. **성능 순위가 뒤집힌다.**

랜덤 초기화에서는 PredKD가 최종 정확도 기준으로 가장 좋았지만, pre-training이 있으면
PredKD는 $26.6$,
PredKD + EWC는 $31.1$,
PredKD + L2는 $35.6$으로 올라간다.
특히 **PredKD + L2가 가장 높은 성능**을 낸다.

저자들의 해석은 설득력 있다. pre-training이 없는 경우 parameter regularization은 파라미터 이동을 막아 새 task 학습을 어렵게 만든다. 하지만 pre-training이 있으면 모델은 이미 좋은 feature space에서 시작하므로, 큰 폭의 plasticity가 꼭 필요하지 않다. 따라서 과거 지식과 pre-trained state 근처를 유지하도록 하는 parameter regularization의 장점이 살아나고, 단점은 줄어든다.

흥미롭게도 forgetting 지표 자체는 pre-training으로 크게 달라지지 않는다. Figure 3의 CKA를 보면 representation similarity 패턴은 Figure 2와 극적으로 다르지 않다. 대신 Figure 4는 **새 task 학습 능력**이 pre-training 덕분에 크게 향상되었음을 보여준다. 즉, pre-training은 forgetting을 직접 줄인다기보다, **낮은 plasticity 환경에서도 새 task를 잘 배우게 해 주기 때문에 parameter regularization의 약점을 상쇄**한다는 것이다.

### 4.5 문헌 맥락 속 ResNet 결과

Table 4는 CIFAR-100 10-task에서 다양한 rehearsal 및 pre-training 조건을 비교한다. 이 표에서 중요한 사실은, **pre-training을 결합한 rehearsal-free 방법이 일부 rehearsal 기반 방법보다도 더 좋은 결과를 낸다**는 점이다.

예를 들어 ResNet에서 PredKD + L2 with ImageNet pre-training은 최종 정확도 $34.4$를 기록한다. 이는 synthetic replay 기반 ABD의 $33.7$보다 높고, 2000-image coreset rehearsal $24.0$, LwF $27.4$보다도 높다. 이는 단순히 “rehearsal-free라서 불리하다”는 통념을 약화시킨다. 적절한 pre-training과 regularization 설계를 통해, 적어도 이 benchmark에서는 stored/synthetic rehearsal 없이도 매우 경쟁력 있는 결과를 얻을 수 있다는 뜻이다.

### 4.6 ViT와 ImageNet-R에서의 결과

논문의 마지막 실험은 ViT 기반 continual learning이다. 데이터셋은 ImageNet-R이며, 200개 클래스를 10개 task로 나누고 task당 20개 클래스를 둔다. backbone은 ImageNet-1K pre-trained ViT-B/16이다. 최근 prompting 계열 논문들과 공정하게 비교하기 위해, 저자들은 backbone 대부분을 freeze하고 **self-attention block의 QKV projection matrices만 fine-tune**한다. 이는 prompting 방법들이 주로 영향을 주는 부분과 비슷한 범위를 건드리게 하려는 설계다.

비교 대상에는 LwF.MC, L2P, L2P++, DualPrompt, CODA-Prompt가 포함된다. 결과(Table 5)는 상당히 강력하다.
L2는 $A_{1:N}=76.06 \pm 0.65$로,
CODA-P의 $75.45 \pm 0.56$,
CODA-P(small)의 $73.93 \pm 0.49$,
DualPrompt의 $71.32 \pm 0.62$를 모두 넘어선다.
forgetting 지표는 EWC가 $1.55 \pm 0.25$로 가장 낮지만, 최종 정확도는 L2가 가장 높다.

이 결과는 논문의 핵심 주장과 정확히 연결된다. pre-trained transformer에서는 **복잡한 prompting 기법보다도, 적절한 범위의 fine-tuning과 단순한 L2 regularization이 더 효과적일 수 있다**. 특히 저자들은 L2가 task 1부터 regularization을 적용할 수 있다는 점, 즉 초기 pre-trained state를 task 전반에 걸쳐 anchor로 삼을 수 있다는 점을 장점으로 해석한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은, 새로운 복잡한 방법을 제안하기보다 **기존 기법들을 정확히 분해하고 그 역할을 다시 해석했다는 점**이다. continual learning 문헌에서는 종종 EWC, distillation, replay 같은 요소들이 함께 섞여 비교되는데, 이 논문은 rehearsal-free setting에서 parameter regularization, prediction distillation, feature distillation, classifier formulation(BCE vs softmax)을 분리해 보여 준다. 그 결과 “무엇이 실제로 중요한가”가 상당히 선명해진다.

둘째 강점은 **분석의 수준**이다. 단순히 정확도만 비교하지 않고, global/local forgetting을 함께 보고, CKA 분석으로 layer별 representation drift까지 관찰한다. 이를 통해 저자들은 feature drift와 classifier bias를 구분하고, 각각에 대해 어떤 기법이 더 관련 있는지 설명한다. 이 점은 단순 benchmark 논문보다 해석력이 높다.

셋째, **pre-training의 역할을 명확히 재정의**했다는 점이 중요하다. 이 논문은 pre-training이 forgetting 자체를 크게 줄인다기보다, parameter regularization이 갖는 plasticity 부족 문제를 상쇄해 준다고 본다. 이 해석은 실험 결과와 잘 맞고, 향후 continual learning 설계에서 매우 실용적인 시사점을 준다.

넷째, ViT 실험을 통해 **prompting 기반 최신 방법들과 직접 경쟁 가능한 단순 baseline**을 제시했다는 점도 강점이다. 이는 분야 전체에 대해 “더 복잡한 방법이 항상 더 낫지 않다”는 건강한 메시지를 준다.

반면 한계도 있다. 가장 먼저, 이 논문은 **완전히 새로운 알고리즘 자체를 제안하는 논문은 아니다**. 기여는 주로 재평가와 조합, 그리고 해석에 있다. 따라서 method novelty 측면에서는 상대적으로 약하게 느껴질 수 있다.

둘째, 논문의 결론은 주로 **CIFAR-100 10-task**와 **ImageNet-R ViT setting**에 기반한다. 물론 이 두 실험은 의미가 있지만, rehearsal-free continual learning의 전체 스펙트럼을 대표한다고 말하기에는 제한적이다. 예를 들어 task granularity, task 수, class imbalance, domain shift 정도가 더 다양할 때도 같은 결론이 유지되는지는 이 텍스트만으로는 확인할 수 없다.

셋째, 저자들은 parameter regularization이 feature drift를 줄이고 classifier bias는 PredKD/BCE가 해결한다고 설명하지만, **이 둘의 상호작용을 이론적으로 엄밀하게 증명하지는 않는다**. 주장은 실험적으로 설득력 있지만, 왜 FeatKD가 예상보다 잘 안 되는지에 대한 더 깊은 이론 분석은 부족하다.

넷째, pre-training이 중요한 변수로 등장하면서, 결과가 사실상 **강한 pre-trained representation에 상당히 의존**하는 것처럼 보이기도 한다. 이는 실제 응용에서 강력한 auxiliary pre-training이 항상 사용 가능한 것은 아니라는 점에서 한계가 될 수 있다.

다섯째, 논문은 privacy와 data storage 문제를 rehearsal-free 연구의 동기로 제시하지만, **model inversion이나 synthetic replay의 법적·윤리적 위험성에 대한 분석은 정성적 수준**이다. 실제 privacy leakage 비교를 실험적으로 보여 주지는 않는다.

그럼에도 비판적으로 보았을 때, 이 논문은 “parameter regularization은 rehearsal-free class-incremental learning에서 소용없다”는 널리 퍼진 인식을 정정하고, **setting에 따라 method ranking이 뒤바뀔 수 있다**는 중요한 교훈을 준다는 점에서 충분히 의미 있다.

## 6. 결론

이 논문은 rehearsal-free continual learning에서 흔히 사용되는 regularization 전략들을 다시 분석하여, 어떤 조건에서 무엇이 효과적인지 정리한 연구다. 핵심 결론은 명확하다. **랜덤 초기화 상태에서는 PredKD와 BCE가 가장 중요한 기반**이며, parameter regularization은 forgetting은 줄이지만 plasticity를 희생할 수 있다. 반면 **pre-training이 존재하면 단순한 L2 parameter regularization이 매우 강력해지며, EWC나 feature distillation보다도 더 좋은 성능**을 낸다.

또한 이 논문은 continual learning에서 forgetting을 단순히 하나의 숫자로만 볼 것이 아니라, **feature drift와 classifier bias를 분리해서 봐야 한다**는 관점을 제시한다. 이 관점은 향후 method 설계에 유용하다. 예를 들어 encoder 보존과 classifier calibration을 서로 다른 수단으로 다루는 방향의 연구로 이어질 수 있다.

실제 적용 측면에서도 의미가 크다. 데이터 저장이 불가능한 privacy-sensitive 환경에서, 복잡한 replay나 generative rehearsal 없이도 pre-trained model과 적절한 regularization만으로 강한 성능을 얻을 수 있다는 점은 매우 실용적이다. 더 나아가 ViT에서 단순한 L2 fine-tuning이 최신 prompting 방법을 능가한 결과는, continual learning 연구에서 **복잡성보다 문제 설정에 맞는 inductive bias와 optimization design이 더 중요할 수 있음**을 보여 준다.

종합하면, 이 논문은 rehearsal-free continual learning에서의 강한 baseline과 해석 프레임을 제공한다. 완전히 새로운 학습 원리를 제시한 논문이라기보다, 기존 방법들의 역할과 한계를 정리하고, 특히 pre-training 시대에 어떤 regularization이 실제로 유효한지를 분명히 한 논문으로 평가할 수 있다.
