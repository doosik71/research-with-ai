# DEAL: Deep Evidential Active Learning for Image Classification

## 1. Paper Overview

이 논문은 image classification에서 **deep active learning**의 효율을 높이기 위해 **DEAL (Deep Evidential Active Learning)**을 제안한다. 문제의식은 분명하다. CNN은 높은 성능을 내지만 대량의 labeled data를 필요로 하고, 의료영상이나 산업 영상처럼 전문가 라벨링이 비싼 환경에서는 annotation cost가 병목이 된다. 저자들은 이러한 상황에서, unlabeled pool 중 어떤 샘플을 먼저 라벨링해야 학습 효율이 가장 좋은지를 결정하는 active learning 전략이 중요하다고 본다. 기존 CNN 기반 AL 방법들은 uncertainty를 softmax, MC-Dropout, ensemble, loss prediction 등으로 추정하지만, 성능이 일관되지 않거나 계산비용이 큰 문제가 있었다. DEAL은 softmax 대신 **Dirichlet density의 파라미터를 직접 출력하는 evidential formulation**을 사용해, 더 좋은 uncertainty를 단일 forward pass로 얻고 이를 acquisition에 활용한다. 논문은 MNIST, CIFAR-10, 그리고 pediatric chest X-ray pneumonia detection 실험에서 DEAL이 기존 방법을 일관되게 앞선다고 주장한다.  

이 문제가 중요한 이유는 active learning의 목표가 단순 정확도 향상이 아니라, **같은 labeling budget으로 더 높은 성능을 얻는 것**이기 때문이다. 논문은 특히 의료 영상처럼 라벨링에 전문가 시간이 많이 드는 분야에서 DEAL의 실효성을 강조한다. 초록과 서론에서 저자들은 pneumonia chest radiograph 사례에서 90% 정확도 달성에 필요한 라벨 수를 random acquisition 대비 **34.52% 줄일 수 있었다**고 요약한다.  

## 2. Core Idea

핵심 아이디어는 기존 softmax classifier가 내놓는 class probability point estimate만으로는 uncertainty를 제대로 표현하기 어렵다는 점에서 출발한다. softmax 값이 높더라도 모델이 실제로는 불확실할 수 있다는 문제를 저자들은 반복해서 지적한다. MC-Dropout이나 deep ensemble은 이 문제를 완화할 수 있지만, acquisition마다 여러 번 forward pass를 하거나 여러 모델을 유지해야 하므로 계산비용이 커진다. DEAL은 이 대신 **CNN의 출력 자체를 evidential output으로 바꾸어 Dirichlet posterior를 직접 학습**한다. 이렇게 하면 예측 클래스별 belief mass와 전체 uncertainty mass를 동시에 얻을 수 있고, 그 uncertainty를 기반으로 informative sample을 고를 수 있다.

즉, DEAL의 novelty는 다음 두 점에 있다. 첫째, **evidential deep learning**을 active learning acquisition에 접목했다는 점이다. 둘째, uncertainty quality와 computational efficiency를 함께 잡으려 했다는 점이다. 논문은 DEAL이 high-quality uncertainty estimate를 제공하면서도 **각 데이터 포인트당 한 번의 forward pass만 필요**하다고 강조한다. 이것이 MC-Dropout, ensemble, learning-loss 계열과의 실용적 차별점이다.

## 3. Detailed Method Explanation

### 3.1 기존 softmax uncertainty의 한계

논문은 image classification용 CNN active learning에서 uncertainty-based selection이 일반적이지만, softmax probability는 본질적으로 point estimate이기 때문에 불확실성을 충분히 반영하지 못한다고 설명한다. 높은 softmax confidence가 곧 낮은 epistemic uncertainty를 의미하지는 않는다. 그래서 기존 연구들은 MC-Dropout, ensemble, loss prediction, BADGE, BatchBALD 같은 기법으로 uncertainty 또는 diversity를 보완해 왔다. 하지만 이들은 계산량 증가, 구현 복잡성, batch redundancy 문제를 동반한다.

### 3.2 Theory of Evidence와 Dirichlet modeling

DEAL의 이론적 기반은 Sensoy et al.의 evidential deep learning이며, 이는 Dempster-Shafer Theory of Evidence를 subjective logic과 Dirichlet distribution으로 정식화한 것이다. 논문은 $K$개의 class에 대해 각 class의 belief mass $b_k$와 전체 uncertainty mass $u$를 정의하고, 이들이 다음 관계를 만족한다고 둔다.

$$
u + \sum_{k=1}^{K} b_k = 1
$$

또한 각 class에 대한 non-negative evidence를 $e_k \ge 0$라 두고, Dirichlet strength를

$$
S = \sum_{i=1}^{K}(e_i + 1)
$$

로 정의한다. 그러면 belief mass와 uncertainty는

$$
b_k = \frac{e_k}{S}, \qquad u = \frac{K}{S}
$$

로 주어진다. 그리고 Dirichlet parameter는

$$
\alpha_k = e_k + 1
$$

이 된다. 직관적으로 보면 evidence가 크면 특정 class에 대한 belief가 커지고, 전체 strength $S$가 커질수록 uncertainty mass $u$는 줄어든다. 즉, DEAL은 단순히 “가장 높은 class probability”가 아니라, **예측을 뒷받침하는 evidence의 총량과 분포**를 함께 본다.  

### 3.3 네트워크 출력과 loss

저자들은 softmax를 제거하고, 그 대신 Softsign 같은 nonlinear activation의 출력을 evidence vector로 사용한다고 설명한다. 이 evidence로 Dirichlet posterior를 형성하고, 학습 loss는 단순 classification loss뿐 아니라 **KL divergence regularization**을 포함한다. 논문 설명에 따르면 이 regularization은 predictive distribution을 정규화해 불필요하게 과도한 evidence를 억제하는 역할을 한다. 여기서 중요한 점은 DEAL이 Bayesian posterior sampling을 직접 하는 것이 아니라, **deterministic neural network가 Dirichlet hyperparameter를 직접 예측**한다는 것이다.

### 3.4 Acquisition 방식

DEAL은 uncertainty-based AL에 속한다. 즉, 각 AL round에서 unlabeled pool 전체에 대해 uncertainty estimate를 계산하고, uncertainty가 높은 샘플을 다음 batch로 query한다. 저자들은 DEAL을 minimal margin, entropy, least confidence 등 softmax 기반 uncertainty와 비교하고, MC-Dropout, Deep Ensemble, Learning Loss, core-set, BADGE 등과도 비교한다. 논문의 강조점은 DEAL이 **uncertainty quality는 높지만 acquisition time은 낮다**는 것이다. 특히 MC-Dropout과 ensemble은 uncertainty 추정 품질은 좋을 수 있으나 acquisition step이 반복될수록 더 비싸진다. 반면 DEAL은 단일 네트워크와 단일 pass로 uncertainty를 얻는다.

### 3.5 이 방법이 왜 유리한가

논문의 논리는 비교적 명확하다. softmax uncertainty는 calibration 문제가 있고, Bayesian 근사나 ensemble은 비용이 높다. DEAL은 Dirichlet posterior를 직접 학습하여 class belief와 global uncertainty를 분리해 표현함으로써 더 informative한 sample selection이 가능하다고 주장한다. 이 방식은 extra loss-prediction module도 필요 없고, 여러 stochastic forward pass도 필요 없다. 따라서 **정확도 개선과 계산 효율 개선을 동시에 노리는 uncertainty-based AL**로 이해할 수 있다.

## 4. Experiments and Findings

### 4.1 실험 설정

논문은 공개 데이터셋인 **MNIST**와 **CIFAR-10**을 사용하고, real-world case로 **pediatric pneumonia chest X-ray** 데이터를 추가한다. MNIST는 58,000/2,000/10,000, CIFAR-10은 48,000/2,000/10,000의 train/validation/test split을 사용한다. 비교 모델로는 LeNet과 ResNet을 사용하고, DEAL을 포함한 여러 state-of-the-art AL 방법과 benchmark한다. Figure 1은 MNIST와 CIFAR-10에서 labeled training data 비율에 따른 test accuracy를 보여준다.

### 4.2 MNIST와 CIFAR-10 결과

논문은 DEAL이 두 네트워크와 두 데이터셋 모두에서 **모든 acquisition round에 걸쳐 일관되게 우수**했다고 보고한다. 구체적으로 5회 반복 실험과 모든 AL round 평균 기준으로, MNIST에서는 DEAL이 두 번째로 좋은 방법 대비 LeNet에서 **1.01%**, ResNet에서 **1.06%** 높았다. CIFAR-10에서는 LeNet 기준 다른 방법들이 random sampling보다도 낫지 않은 경우가 있었지만, DEAL은 random baseline보다 평균 **1.51%** 높았다. ResNet에서는 Deep Ensemble이 두 번째로 좋았고, DEAL은 그보다 평균 **0.51%** 개선되었다. 저자들은 paired t-test를 수행했고, 네트워크와 데이터셋 전반에서 **0.01 수준의 통계적 유의성**을 보였다고 말한다.  

### 4.3 라벨 절감 효과

실무적으로 더 중요한 결과는 “목표 성능에 도달하기 위해 몇 장을 라벨링해야 하는가”다. 논문은 MNIST에서 95%, CIFAR-10에서 87% test accuracy를 달성하기 위해 필요한 이미지 수를 비교했고, DEAL을 쓰면 random sampling 대비 **MNIST 280장**, **CIFAR-10 6,800장**을 덜 라벨링해도 된다고 보고한다. 이는 각각 **34.15%**, **24.29%**의 annotation effort 절감이다. 또한 두 번째로 좋은 softmax minimal margin 대비로도 MNIST는 80장, CIFAR-10은 1,300장 적게 필요했다고 한다. active learning 논문으로서 이 결과는 매우 중요하다. 왜냐하면 단순 curve superiority가 아니라 **실제 budget 관점에서 의미 있는 절감**을 보여주기 때문이다.

### 4.4 Acquisition time

논문은 정확도뿐 아니라 acquisition runtime도 분석한다. 제공된 결과 조각에 따르면, MC-Dropout과 Deep Ensemble은 uncertainty 추정에 시간이 많이 들고, CIFAR-10에서는 차이가 더 커진다. 예를 들어 일부 benchmark 대비 DEAL은 acquisition 시간이 훨씬 짧으며, core-set도 CIFAR-10에서 DEAL보다 평균 **931.28초** 느렸다고 보고한다. 저자들의 결론은 DEAL이 목표 성능을 더 적은 라벨로 달성할 뿐 아니라, **acquisition time 측면에서도 유리**하다는 것이다.

### 4.5 Real-world pneumonia 데이터셋

의료 사용 사례에서는 1,500장을 train, 200장을 validation, 1,400장을 test로 배정하고, ResNet을 사용해 실험한다. 각 AL round마다 모델을 scratch에서 100 epoch 학습하고, batch size 8, learning rate 0.0005, Adam optimizer를 사용한다. 초기에는 64장의 랜덤 이미지로 시작하고, 각 round마다 64장을 추가 query하며, 총 704장까지 acquisition을 진행한다.

이 실험에서도 DEAL은 모든 benchmark를 일관되게 앞섰다. 평균적으로 DEAL은 두 번째 방법인 softmax minimal margin보다 **1.76%**, random baseline보다 **2.86%** 높은 정확도를 보였다. 특히 **90% 정확도**를 달성하기 위해 필요한 라벨 수를 기준으로, 두 번째 방법보다 **64장 적게**, random sampling보다 **243장 적게** 라벨링하면 되었다. 이는 각각 **12.19%**, **34.52%** 절감이다. paired t-test 역시 0.01 수준에서 유의했다. 다만 저자들은 이 데이터셋에서는 MNIST/CIFAR-10보다 run 간 표준편차가 크며, 그 이유로 학습에 사용된 이미지 수가 작기 때문일 수 있다고 해석한다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **불확실성 표현 방식의 개선**을 active learning에 직접 연결했다는 점이다. softmax confidence의 한계를 정확히 짚고, evidential deep learning을 통해 Dirichlet 기반 uncertainty를 직접 학습하는 구조를 제시했다. 또한 이 접근이 MC-Dropout이나 deep ensemble처럼 비싼 반복 추론 없이도 가능하다는 점에서 practical value가 있다.

또 하나의 강점은 실험 결과가 단지 한 데이터셋의 우연으로 보이지 않는다는 점이다. MNIST, CIFAR-10, 그리고 의료 X-ray 사례까지 모두에서 일관된 개선을 보였고, 정확도뿐 아니라 라벨 절감과 acquisition time까지 함께 비교했다. active learning 논문에서 이런 세 축을 동시에 보는 것은 설득력이 높다.  

### 한계

논문이 스스로 인정하는 가장 중요한 한계는, DEAL이 **오직 uncertainty-based selection**에 의존한다는 점이다. batch acquisition에서 uncertainty만 기준으로 샘플을 고르면 서로 비슷한 샘플이 한 번에 많이 뽑혀 redundancy가 생길 수 있다. 이는 suboptimal batch selection으로 이어질 수 있다. 저자들은 conclusion에서 future work로 **diversity criterion의 통합**을 제안한다. 이 부분은 DEAL의 가장 분명한 구조적 한계다.

또 다른 한계는 method의 핵심이 evidential output calibration에 크게 의존한다는 점이다. 논문은 이를 실험적으로 뒷받침하지만, “왜 이 uncertainty가 모든 상황에서 더 잘 작동하는가”에 대한 보다 깊은 이론적 분석은 제한적이다. 또한 batch diversity를 직접 고려하는 BADGE, BatchBALD류와 비교해도 DEAL이 우수하다고 보고하지만, 장기적으로는 uncertainty와 diversity를 결합한 변형이 더 강할 가능성이 있다. 이는 논문 내용에 기반한 합리적 해석이다.  

### 해석

비판적으로 보면, DEAL은 Bayesian AL을 완전히 대체하는 접근이라기보다, **CNN에 더 적합하고 더 가벼운 uncertainty surrogate**를 제안한 논문으로 보는 것이 적절하다. 논문의 진짜 가치는 “uncertainty를 더 잘 표현하면 active learning도 좋아진다”를 evidential modeling으로 실증했다는 데 있다. 특히 의료 영상처럼 annotation cost가 실제로 큰 환경에서 성능 개선을 보였다는 점이 중요하다. 다만 batch redundancy를 해결하지 못한 점을 감안하면, 후속 연구는 DEAL의 evidential uncertainty 위에 diversity-aware batch selection을 얹는 방향으로 이어질 가능성이 크다.

## 6. Conclusion

이 논문은 CNN 기반 image classification을 위한 새로운 uncertainty-based active learning 방법인 **DEAL**을 제안한다. 핵심은 softmax output 대신 **Dirichlet density의 파라미터를 직접 출력하는 evidential network**를 사용해, 더 질 좋은 uncertainty estimate를 single forward pass로 얻는 것이다. 이를 통해 DEAL은 MNIST, CIFAR-10, pediatric pneumonia chest X-ray에서 기존 state-of-the-art 방법보다 더 좋은 정확도와 더 낮은 labeling cost를 보였고, acquisition time 측면에서도 경쟁력이 있음을 보여준다.  

실무적으로는, 라벨링이 비싼 image classification 문제에서 **쉽게 구현 가능하면서 계산 부담이 낮은 AL 방법**으로 의미가 있다. 연구적으로는, evidential deep learning을 active learning과 연결해 uncertainty estimation의 품질이 sample acquisition에 미치는 영향을 잘 보여준 사례다. 다만 논문이 직접 언급하듯, uncertainty만으로 batch를 고르면 redundancy 문제가 남기 때문에, 향후 diversity-aware extension이 중요한 후속 과제가 된다.
