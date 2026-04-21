# A Comprehensive Overview and Survey of Recent Advances in Meta-Learning

- **저자**: Huimin Peng
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2004.11149

## 1. 논문 개요

이 논문은 meta-learning, 즉 learning-to-learn 분야의 전반적 흐름을 정리하는 survey paper이다. 저자는 meta-learning을 “이미 학습된 경험을 바탕으로, 새로운 unseen task에 빠르고 정확하게 적응하는 방법론”으로 설명한다. 특히 deep learning이 주로 in-sample prediction, 즉 학습 분포 내부의 예측 정확도에 집중하는 반면, meta-learning은 out-of-sample prediction과 unseen task adaptation에 더 직접적으로 초점을 둔다고 본다.

논문이 다루는 핵심 연구 문제는 다음과 같다. 첫째, 매우 적은 데이터만 주어진 few-shot setting에서 어떻게 높은 차원의 입력을 다룰 것인가. 둘째, 기존에 본 적 없는 task로 모델을 어떻게 일반화할 것인가. 셋째, 단순히 pretrained model을 미세조정하는 transfer learning을 넘어, 서로 꽤 다른 task들 사이의 공통 구조를 어떻게 포착할 것인가. 저자는 이 문제들이 computer vision뿐 아니라 natural language processing, robotics, reinforcement learning, imitation learning, unsupervised learning 등으로 넓게 이어진다고 본다.

이 문제가 중요한 이유는 분명하다. 실제 환경에서는 충분한 라벨 데이터가 없는 경우가 많고, 환경이 계속 변하며, 매번 모델을 처음부터 다시 학습시키는 비용이 크다. 논문은 meta-learning이 이런 한계를 줄이면서, 인간처럼 소수의 예시로 새로운 개념을 빠르게 배우는 방향의 AI에 가까워지게 해 준다고 주장한다. 특히 few-shot image classification이 최근 메타러닝 연구의 중심 응용 분야로 강조된다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 meta-learning을 하나의 단일 기술이 아니라, “task 간 공통 구조를 학습해 빠른 적응을 가능하게 하는 일반 프레임워크”로 보는 것이다. 저자는 이를 네 가지 큰 범주로 나누어 정리한다. black-box meta-learning, metric-based meta-learning, layered meta-learning, Bayesian meta-learning이 그것이다. 또한 응용 측면에서는 meta-reinforcement learning, meta-imitation learning, online meta-learning, unsupervised meta-learning으로 확장해서 설명한다.

논문이 반복해서 강조하는 개념은 task similarity이다. 방법은 달라도 meta-learning은 결국 “서로 다른 task 사이에 무엇이 공유되는가”를 모델링해야 한다는 것이다. 어떤 방법은 similarity를 명시적으로 거리 함수나 prototype으로 표현하고, 어떤 방법은 meta-learner의 파라미터 내부에 암묵적으로 집어넣는다. 저자는 이 similarity modeling이 메타러닝의 사실상 필수 요소라고 본다.

기존 접근과의 차별점은 이 survey가 meta-learning을 단순 few-shot classification 기법 모음으로 보지 않고, 보다 넓은 관점에서 self-improvement, optimizer learning, Bayesian uncertainty estimation, robotics adaptation, AutoML, coevolution까지 연결한다는 점이다. 다만 이 논문은 새로운 알고리즘을 제안하는 논문이 아니라, 여러 흐름을 묶어 최근 발전 방향을 정리하는 논문이다. 따라서 “무엇이 가장 좋은 방법인가”를 엄밀히 결론내기보다는, 어떤 설계 축들이 존재하는지 보여주는 데 더 가깝다.

## 3. 상세 방법 설명

논문은 먼저 meta-learning의 기본 학습 형식을 task 단위의 episodic training으로 설명한다. 전체 데이터는 `meta-train`, `meta-val`, `meta-test`로 나뉘며, 각 task 내부에는 `D^{tr}`, `D^{val}`, `D^{test}`가 있다. few-shot classification에서는 보통 support set의 소량 라벨 데이터로 task-specific parameter를 적응시키고, validation loss를 이용해 meta-parameter를 업데이트한다. 논문은 task를 다음과 같이 정의한다.

$$
T = \{p(x), p(y|x), L\}
$$

여기서 $L$은 loss function이고, $p(x)$와 $p(y|x)$는 입력과 라벨의 생성 분포이다. 또 task는 어떤 task distribution에서 샘플링된다고 본다.

black-box meta-learning에서는 neural network 자체를 사용해 optimizer, adaptation rule, 혹은 pretrained model의 일부를 조정한다. 예를 들어 [93]의 activation-to-parameter는 feature extractor의 activation 평균 $\bar a_y$를 이용해 마지막 분류층의 파라미터 $w_y$를 예측하는 사상 $\phi$를 학습한다. 이때 분류 확률은 다음과 같이 표현된다.

$$
P(y|x) = \frac{\exp\{E_S[\phi(s_y)a(x)]\}}{\sum_{k\in C}\exp\{E_S[\phi(s_k)a(x)]\}}
$$

직관적으로는 “새로운 클래스에 대해 마지막 분류기 가중치를 직접 생성하자”는 접근이다. 또 AdaResNet/AdaCNN류 방법은 neuron activation이나 weight에 task-specific shift를 넣고, 외부 memory에서 유사 task 정보를 attention으로 읽어와 adaptation을 가속한다.

metric-based meta-learning에서는 embedding function과 similarity metric이 핵심이다. Relation Network는 feature extractor $f_\phi$와 relation function $g_\theta$를 두고, query와 support의 유사도를

$$
r_{i,j} = g_\theta[f_\phi(x_i), f_\phi(x_j^*)]
$$

로 계산한다. 학습은 같은 클래스면 1, 다른 클래스면 0에 가깝게 만드는 형태의 제곱 오차로 진행된다. Prototypical Network는 더 단순하다. 클래스별 prototype을

$$
c_k = \frac{1}{|S_k|}\sum_{(x_i,y_i)\in S_k} f_\phi(x_i)
$$

로 두고, query가 클래스 $k$에 속할 확률을

$$
p_\phi(y=k|x) = \frac{\exp(-g[f_\phi(x), c_k])}{\sum_{k'} \exp(-g[f_\phi(x), c_{k'}])}
$$

로 계산한다. 즉 “같은 클래스의 embedding 중심에 얼마나 가까운가”로 분류하는 방식이다. 논문은 TADAM, DAPNA, TRAML 같은 후속 방법들이 이 구조에서 task-dependent component를 늘리거나 loss를 더 잘 설계해 성능을 높였다고 정리한다.

layered meta-learning은 base learner와 meta-learner를 분리하는 관점이다. 가장 대표적인 MAML에서 inner loop는 각 task에 대해 task-specific parameter를 갱신하고, outer loop는 그 갱신이 잘 되도록 초기값인 meta-parameter를 학습한다. MAML의 inner update는

$$
\phi_i = \theta - \alpha \nabla_\theta L_{T_i}(h_\theta)
$$

이고, outer update는

$$
\theta \leftarrow \theta - \beta \nabla_\theta \sum_{T_i\sim p(T)} L_{T_i}(h_{\phi_i})
$$

이다. 쉽게 말해, $\theta$는 “새 task에 들어갔을 때 몇 번의 gradient step만으로 잘 적응할 수 있는 좋은 시작점”이다. Meta-SGD는 여기에 inner-loop step size $\alpha$까지 함께 학습한다. Reptile은 higher-order derivative를 피하기 위해 더 단순한 1차 근사 업데이트를 사용한다. Meta-LSTM은 LSTM cell state update와 gradient-based optimization의 대응 관계를 이용해, LSTM을 meta-learner로 활용한다.

이 범주 안에서 흥미로운 방향은 base learner를 꼭 deep network로 둘 필요가 없다는 점이다. R2-D2, LR-D2, MetaOptNet은 ridge regression, logistic regression, SVM 같은 비교적 단순하고 닫힌형식 또는 convex optimization이 가능한 학습기를 base learner로 사용한다. 논문의 관점에서는 이것이 overfitting을 줄이고 few-shot setting에 더 적합할 수 있다. TPN은 graph-based transductive label propagation을 base learner처럼 사용해, support와 query를 함께 그래프로 묶고

$$
F^{t+1} = \theta W F^t + (1-\theta)Y
$$

로 label propagation을 수행한 뒤, 수렴점

$$
F^* = (I-\theta W)^{-1}Y
$$

를 바로 계산한다. LEO는 latent embedding 공간에서 parameter optimization을 수행한 뒤 decoder를 통해 classifier parameter를 복원한다.

Bayesian meta-learning은 task-specific parameter와 meta-parameter를 모두 random variable로 다룬다. 핵심 목적은 few-shot 상황에서 큰 uncertainty를 함께 모델링하는 것이다. LLAMA는 MAML을 hierarchical Bayesian inference로 다시 해석하고 Laplace approximation을 사용한다. BMAML은 MAML의 SGD를 SVGD로 바꾸어 posterior particle을 직접 업데이트한다. PLATIPUS는 variational inference로 approximate posterior를 구하고, VERSA는 amortized variational inference와 Bayesian decision theory를 이용해 예측 분포를 직접 모델링한다. 이 범주의 장점은 단일 point estimate가 아니라 uncertainty-aware prediction이 가능하다는 점이다. 반면 계산이 더 복잡하고, LLAMA처럼 Gaussian/Laplace approximation이 잘 맞는 분포 가정에 기대는 경우도 있다.

## 4. 실험 및 결과

이 논문은 survey이므로 자체 실험을 새로 수행하기보다, 기존 논문들의 대표 성능을 표 형태로 모아 비교한다. 가장 자주 등장하는 benchmark는 few-shot image classification이며, 데이터셋으로는 Omniglot, ImageNet, miniImageNet, tieredImageNet, CIFAR-FS, FC100, CUB-200, CelebA 등이 소개된다. 그중에서도 저자는 miniImageNet과 tieredImageNet을 대표적인 few-shot classification benchmark로 본다.

평가 설정은 주로 `N-way K-shot`이다. 예를 들어 5-way 5-shot은 5개의 클래스가 있고 클래스당 5개의 support example이 있는 문제다. 지표는 대부분 classification accuracy이며, 표에는 평균 정확도와 confidence interval이 함께 제시된다.

black-box 계열에서는 5-way 5-shot miniImageNet에서 activation-to-parameter가 약 67.87%, wide residual network를 사용한 버전이 73.74%, AdaResNet이 71.94% 수준으로 요약된다. metric-based 계열에서는 Matching Net 60.0%, Relation Net 65.32%, Prototypical Net 68.20%, TADAM 76.7%, Prototypical Net+TRAML 77.94%, AM3+TRAML 79.54%, DAPNA 84.07%가 제시된다. layered meta-learning 계열에서는 MAML 63.11%, Meta-SGD 64.03%, R2-D2 68.4%, MetaOptNet-RidgeReg 77.88%, MetaOptNet-SVM 78.63%, MetaOptNet-SVM-trainval 80.00%, LEO 77.59%가 보고된다. Bayesian 계열은 5-way 1-shot miniImageNet에서 비교되며, LLAMA 49.40%, BMAML 53.8%, PLATIPUS 50.13%, VERSA 53.40% 정도가 정리된다.

논문이 직접 강조하는 해석은 다음과 같다. 첫째, 단순 backbone 비교가 아니라 feature extractor, loss function, transductive setting, tuning 방식이 다르기 때문에 표의 수치 비교는 엄밀하지 않다. 저자도 여러 번 “rough and not exact”라고 밝힌다. 둘째, task-dependent component를 더 잘 설계하거나, 더 좋은 feature extractor와 discriminative loss를 쓰는 방법들이 대체로 성능이 좋다. 셋째, statistical learner와 deep meta-learner를 결합한 MetaOptNet 류나, prototype 기반 구조를 정교화한 TADAM/DAPNA 류가 당시 few-shot image classification에서 강한 결과를 보였다고 정리한다.

응용 실험에 대해서는 meta-RL, meta-imitation learning, online meta-learning, unsupervised meta-learning 사례를 소개한다. 예를 들어 PEARL은 off-policy actor-critic과 latent context inference를 결합한 meta-RL로 설명되고, one-shot imitation 및 Watch-Try-Learn은 적은 demonstration으로 policy를 적응시키는 예시로 다뤄진다. 다만 survey의 특성상 각 응용 논문을 동일한 기준으로 세밀 비교하지는 않는다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 meta-learning을 매우 넓은 지형도로 정리한다는 점이다. few-shot image classification에만 머무르지 않고 optimizer learning, AutoML, memory-based method, Bayesian inference, meta-RL, imitation learning, online learning, unsupervised learning, coevolution까지 연결한다. 따라서 특정 알고리즘 하나를 이해하기보다, 분야 전체의 축을 잡는 데 유용하다. 또한 MAML, ProtoNet, TPN, LEO, BMAML, VERSA 등 핵심 방법들의 구조와 수식을 비교적 많이 포함하고 있어, 입문자에게 개념 지도를 제공한다.

또 다른 강점은 task similarity를 메타러닝의 공통 핵심으로 반복해서 부각한다는 점이다. metric-based 방법처럼 similarity를 직접 거리로 정의하든, MAML처럼 초기화와 업데이트 규칙 안에 암묵적으로 담든, 결국 unseen task adaptation의 성패는 task 간 공유 구조를 얼마나 잘 모델링하느냐에 달려 있다는 통찰을 제공한다. 또한 statistical model과 deep model의 결합 가능성을 강조한 부분도 실용적이다.

반면 한계도 분명하다. 우선 이 논문은 survey이므로 새로운 실험적 근거를 제시하지 않는다. 또한 여러 표를 통해 정확도를 비교하지만, backbone, training protocol, transductive 사용 여부, loss 설계 등이 통일되지 않았다고 스스로 인정한다. 따라서 표의 성능 순위를 그대로 방법론의 우열로 받아들이면 곤란하다. 둘째, 분류 체계가 엄밀하지 않다. 예를 들어 memory-based, metric-based, optimization-based, Bayesian method 사이의 경계가 실제로 많이 겹치며, 저자도 이를 인정한다. 셋째, 논문 전반의 범위가 매우 넓어 각 하위 주제에 대한 깊이는 일정하지 않다. 일부 방법은 수식과 절차가 비교적 자세하지만, 어떤 응용 분야는 개괄 수준에 머문다.

비판적으로 보면, 이 논문은 meta-learning의 잠재력과 범용성을 강하게 강조하는 편이다. 그러나 어떤 조건에서 meta-learning이 실제로 transfer learning보다 유리한지, task distribution misspecification이 있을 때 얼마나 취약한지, 계산 비용 대비 이득이 어느 정도인지에 대한 엄밀한 비교는 충분하지 않다. 또한 “strong AI”나 “high autonomy” 같은 큰 비전이 자주 언급되지만, 그것이 현재 few-shot benchmark 성능과 어떻게 연결되는지는 논문 내부에서 체계적으로 검증되지는 않는다.

## 6. 결론

이 논문은 meta-learning을 “few-shot 데이터에서 unseen task에 빠르게 적응하기 위한 일반화 프레임워크”로 정리하며, 그 주요 흐름을 black-box, metric-based, layered, Bayesian의 네 축으로 묶고, robotics와 sequential decision making까지 응용을 확장해 설명한다. 핵심 기여는 새로운 알고리즘 제안이 아니라, 최근 메타러닝 연구를 비교적 포괄적으로 구조화하고, 공통 개념인 task similarity, base learner와 meta-learner의 역할 분리, uncertainty modeling, 그리고 응용 확장성을 한 자리에서 보여준 데 있다.

실제 적용 측면에서는 few-shot vision, robotics, online adaptation, imitation learning처럼 데이터가 적고 빠른 적응이 필요한 문제에서 메타러닝이 중요한 역할을 할 가능성이 크다. 향후 연구 방향으로는 out-of-distribution generalization 강화, 더 정교한 similarity modeling, Bayesian uncertainty estimation의 실용화, statistical learning과의 결합, 그리고 복잡한 환경에서의 coevolutionary meta-learning이 제시된다. 다만 이 논문이 survey라는 점을 감안하면, 제시된 결론은 “정리와 전망”에 가깝고, 특정 방법의 우위를 확정하는 실증 논문으로 읽어서는 안 된다.
