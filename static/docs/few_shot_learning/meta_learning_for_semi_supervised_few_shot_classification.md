# Meta-Learning for Semi-Supervised Few-Shot Classification

- **저자**: Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell, Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle, Richard S. Zemel
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1803.00676

## 1. 논문 개요

이 논문은 few-shot classification을 더 현실적인 방향으로 확장하려는 연구이다. 기존 few-shot learning은 각 episode 안에서 아주 적은 수의 labeled support example만 주어지고, 그로부터 새로운 class를 구분하는 문제를 다뤘다. 그러나 실제 환경에서는 소량의 labeled data 외에도 label이 없는 unlabeled example이 함께 사용가능한 경우가 많다. 저자들은 바로 이 상황을 semi-supervised few-shot classification으로 정의하고, meta-learning 기반으로 이를 학습하는 방법을 제안한다.

핵심 연구 문제는 다음과 같다. 새로운 class들에 대해 class당 몇 개의 labeled sample만 있는 상황에서, 추가로 주어진 unlabeled sample을 어떻게 활용하면 query set 분류 성능을 높일 수 있는가? 더 나아가, unlabeled set 안에 support class와 무관한 distractor class의 샘플이 섞여 있어도 robust하게 동작할 수 있는가? 논문은 이 문제를 정식으로 설정하고, 기존 Prototypical Networks를 확장해 unlabeled data를 prototype refinement에 활용하는 방법들을 제시한다.

이 문제가 중요한 이유는, 일반적인 semi-supervised learning은 train/test에서 class가 동일하다는 가정을 두는 경우가 많지만, few-shot learning은 training에서 보지 못한 novel class로 generalization해야 하기 때문이다. 즉, 단순히 unlabeled data를 쓰는 것이 아니라, “새로운 class에 대한 반지도 few-shot adaptation”을 잘 수행할 수 있어야 한다. 논문은 이 점을 few-shot transfer와 semi-supervised learning의 결합 문제로 본다.

## 2. 핵심 아이디어

중심 아이디어는 Prototypical Networks의 class prototype을 unlabeled example을 이용해 refinement하는 것이다. 기본 ProtoNet은 labeled support example의 embedding 평균으로 각 class prototype을 만든다. 이 논문은 unlabeled example이 현재 episode의 class structure를 더 잘 드러내도록 도와줄 수 있다고 보고, initial prototype을 만든 뒤 unlabeled embedding을 soft assignment 방식으로 각 class에 부분적으로 배정하여 prototype을 다시 계산한다.

직관적으로 보면, labeled sample이 매우 적을 때 초기 prototype은 noisy할 수 있다. 그런데 unlabeled sample이 실제로 해당 class 주변에 많이 분포한다면, 그것들을 적절히 반영하면 prototype이 더 정확한 class center에 가까워질 수 있다. 이는 semi-supervised clustering이나 self-training과 비슷한 성격을 가진다. 다만 이 논문은 그 refinement 규칙 자체를 episodic meta-training 안에 넣어, 모델이 unlabeled data를 “어떻게 활용해야 하는지”까지 함께 학습하게 만든다.

기존 접근과의 차별점은 두 가지다. 첫째, few-shot meta-learning episode 내부에 unlabeled set을 명시적으로 도입했다. 둘째, 모든 unlabeled sample이 support class에 속한다고 가정하지 않고, distractor가 포함된 더 어려운 setting까지 다룬다. 이를 위해 단순 soft $k$-means뿐 아니라 distractor cluster를 추가하거나, masking mechanism으로 outlier contribution을 줄이는 변형을 제안한다.

## 3. 상세 방법 설명

전체 설정에서 하나의 episode는 labeled support set $S$, unlabeled set $R$, query set $Q$로 구성된다. support set은 기존 few-shot learning과 동일하게 class당 $K$개의 labeled example을 가지며, unlabeled set $R=\{\tilde{x}_1,\dots,\tilde{x}_M\}$은 label 없이 주어진다. 모델은 $S$와 $R$을 사용해 class representation을 만들고, query set의 label을 예측하도록 학습된다.

기본이 되는 Prototypical Networks에서는 embedding network $h(x)$를 학습하고, class $c$의 prototype을 support example 평균으로 계산한다.

$$
p_c = \frac{\sum_i h(x_i) z_{i,c}}{\sum_i z_{i,c}}, \quad z_{i,c} = \mathbf{1}[y_i = c]
$$

그 다음 query sample $x^*$는 embedding 공간에서 prototype과의 squared Euclidean distance를 이용해 분류된다.

$$
p(c \mid x^*, \{p_c\}) =
\frac{\exp(-\|h(x^*) - p_c\|_2^2)}
{\sum_{c'} \exp(-\|h(x^*) - p_{c'}\|_2^2)}
$$

학습 loss는 query set 전체에 대한 평균 negative log-likelihood이다.

$$
-\frac{1}{T}\sum_i \log p(y_i^* \mid x_i^*, \{p_c\})
$$

이 논문의 핵심은 기존 prototype $p_c$를 refined prototype $\tilde{p}_c$로 바꾸는 부분이다.

첫 번째 방법은 **Soft $k$-Means ProtoNet**이다. 여기서는 unlabeled example $\tilde{x}_j$가 각 class prototype에 얼마나 가까운지에 따라 soft assignment $\tilde{z}_{j,c}$를 부여한다.

$$
\tilde{z}_{j,c} =
\frac{\exp(-\|h(\tilde{x}_j)-p_c\|_2^2)}
{\sum_{c'} \exp(-\|h(\tilde{x}_j)-p_{c'}\|_2^2)}
$$

이 soft assignment를 사용해 labeled sample과 unlabeled sample을 함께 평균내어 refined prototype을 만든다.

$$
\tilde{p}_c =
\frac{\sum_i h(x_i) z_{i,c} + \sum_j h(\tilde{x}_j)\tilde{z}_{j,c}}
{\sum_i z_{i,c} + \sum_j \tilde{z}_{j,c}}
$$

즉, unlabeled example도 각 class에 “부분적으로 속한다”고 보고 prototype 계산에 기여하게 한다. 저자들은 일반적인 $k$-means처럼 refinement를 여러 번 반복할 수도 있지만, 실험상 한 번의 refinement step으로 충분했다고 보고한다.

두 번째 방법은 **Soft $k$-Means with a Distractor Cluster**이다. 앞선 방법은 모든 unlabeled sample이 현재 $N$개 class 중 하나에 속한다고 암묵적으로 가정한다. 하지만 현실적으로는 무관한 distractor가 들어올 수 있고, 이 경우 soft assignment가 잘못된 class prototype을 끌어당길 수 있다. 이를 막기 위해 논문은 추가적인 distractor cluster를 하나 더 둔다. class $1$부터 $N$까지는 기존 prototype을 사용하고, distractor cluster $N+1$의 prototype은 단순화를 위해 원점 $0$으로 둔다.

또한 cluster별 length-scale $r_c$를 도입하여 assignment를 다음처럼 계산한다.

$$
\tilde{z}_{j,c} =
\frac{
\exp\left(-\frac{1}{r_c^2}\|\tilde{x}_j-p_c\|_2^2 - A(r_c)\right)
}{
\sum_{c'} \exp\left(-\frac{1}{r_{c'}^2}\|\tilde{x}_j-p_{c'}\|_2^2 - A(r_{c'})\right)
}
$$

여기서

$$
A(r)=\frac{1}{2}\log(2\pi)+\log(r)
$$

이다. 실험에서는 실제 class의 $r_1,\dots,r_N$은 1로 고정하고, distractor cluster의 scale만 학습한다. 이 방식은 distractor를 하나의 넓은 분산을 가진 “잡음 cluster”로 흡수하려는 발상이다.

세 번째 방법은 **Masked Soft $k$-Means**이다. 저자들은 단일 distractor cluster가 지나치게 단순하다고 본다. 실제 distractor는 여러 class에서 올 수 있으므로 하나의 cluster로 표현하기 어렵다. 그래서 대신 “어떤 unlabeled sample이 특정 prototype 주변의 유효 영역 안에 있으면 반영하고, 멀리 있으면 거의 무시하는” soft mask를 도입한다.

먼저 unlabeled example과 prototype 사이의 distance를 계산하고, 각 prototype별 평균 distance로 정규화한다.

$$
\tilde{d}_{j,c} = \frac{d_{j,c}}{\frac{1}{M}\sum_j d_{j,c}}, \quad
d_{j,c} = \|h(\tilde{x}_j)-p_c\|_2^2
$$

그 다음 각 prototype마다 작은 MLP가 threshold $\beta_c$와 slope $\gamma_c$를 예측한다. 입력은 normalized distance들의 통계량이다.

$$
[\beta_c,\gamma_c] = \text{MLP}([
\min_j(\tilde{d}_{j,c}),
\max_j(\tilde{d}_{j,c}),
\mathrm{var}_j(\tilde{d}_{j,c}),
\mathrm{skew}_j(\tilde{d}_{j,c}),
\mathrm{kurt}_j(\tilde{d}_{j,c})
])
$$

이 threshold와 slope를 사용해 mask를 만든다.

$$
m_{j,c} = \sigma(-\gamma_c(\tilde{d}_{j,c}-\beta_c))
$$

최종 refined prototype은 다음처럼 계산된다.

$$
\tilde{p}_c =
\frac{\sum_i h(x_i) z_{i,c} + \sum_j h(\tilde{x}_j)\tilde{z}_{j,c} m_{j,c}}
{\sum_i z_{i,c} + \sum_j \tilde{z}_{j,c} m_{j,c}}
$$

즉, 어떤 unlabeled sample이 prototype에 충분히 가깝다고 판단되면 $m_{j,c}$가 1에 가까워져 prototype refinement에 반영되고, 그렇지 않으면 거의 0이 되어 무시된다. 이 구조는 distractor가 여러 class에 걸쳐 복잡하게 섞여 있어도 대응할 수 있다. 논문은 이 과정이 end-to-end differentiable하다고 설명하며, 단 통계량 계산 쪽으로는 gradient를 막아 수치적 불안정을 피했다고 밝힌다.

## 4. 실험 및 결과

논문은 Omniglot, mini ImageNet, 그리고 새로 제안한 tiered ImageNet에서 실험했다. Omniglot은 handwritten character 데이터셋이며, 회전을 포함해 총 6,492 class로 구성된다. mini ImageNet은 100개 class의 축소 ImageNet 변형이며, tiered ImageNet은 저자들이 새로 제안한 데이터셋으로 608 class와 34개의 상위 category를 가진다. 특히 tiered ImageNet은 train/validation/test를 상위 semantic category 단위로 분리하여, train과 test class가 너무 비슷해지는 문제를 줄이려 했다.

semi-supervised setting을 만들기 위해 각 class의 이미지를 labeled split과 unlabeled split으로 나눴다. Omniglot과 tiered ImageNet은 class별 10%만 labeled로 두고 90%를 unlabeled로 사용했다. mini ImageNet은 10%로는 성능이 너무 낮아 40% labeled, 60% unlabeled split을 사용했다. 이 때문에 저자들은 자신들의 supervised baseline이 기존 few-shot 논문들보다 적은 label 정보를 사용한다고 명시한다.

episode 생성은 다음과 같다. 먼저 $N$개의 target class를 뽑고, 각 class에서 $K$개의 labeled sample로 support set을 만든다. 그리고 각 class당 $M$개의 unlabeled sample을 unlabeled set에 넣는다. distractor setting에서는 추가로 $H$개의 다른 class를 뽑고, 각 distractor class당 $M$개의 unlabeled sample을 더 넣는다. 논문에서는 주로 $H=N=5$를 사용했고, training 시 $M=5$, test 시 $M=20$을 주로 사용해 더 큰 unlabeled set으로 generalization 가능한지도 확인했다.

비교 대상은 크게 두 가지 baseline이다. 첫째는 labeled support만 사용하는 일반 **Supervised ProtoNet**이다. 둘째는 supervised ProtoNet으로 embedding을 학습한 뒤, test time에만 Soft $k$-Means refinement를 적용하는 **Semi-Supervised Inference**이다. 이에 비해 제안 방법들은 training과 test 모두에서 refinement를 사용하므로, unlabeled data 활용에 맞는 embedding 자체를 meta-learn한다는 차이가 있다.

Omniglot 1-shot 결과를 보면, supervised는 94.62%이고 distractor가 없는 setting에서 Semi-Supervised Inference는 97.45%, Soft $k$-Means는 97.25%, Soft $k$-Means+Cluster는 97.68%, Masked Soft $k$-Means는 97.52%를 기록했다. distractor가 있을 때는 supervised 94.62%, Semi-Supervised Inference 95.08%, Soft $k$-Means 95.01%에 비해, Soft $k$-Means+Cluster 97.17%, Masked Soft $k$-Means 97.30%로 크게 개선되었다. 즉 distractor 대응 장치가 특히 중요함을 보여준다.

mini ImageNet에서는 1-shot, 5-shot 모두에서 제안법들이 baseline보다 전반적으로 좋았다. 예를 들어 1-shot non-distractor에서 supervised는 43.61%, Soft $k$-Means는 50.09%, Masked Soft $k$-Means는 50.41%였다. 5-shot non-distractor에서는 supervised 59.08%, Soft $k$-Means 64.59%, Masked Soft $k$-Means 64.39%였다. distractor가 있는 경우에도 Masked Soft $k$-Means는 1-shot 49.04%, 5-shot 62.96%로 supervised 및 test-time-only refinement보다 높았다.

tiered ImageNet에서도 같은 경향이 나타났다. 1-shot non-distractor에서는 supervised 46.52%, Semi-Supervised Inference 50.74%, Soft $k$-Means 51.52%, Soft $k$-Means+Cluster 51.85%, Masked Soft $k$-Means 52.39%였다. 5-shot non-distractor에서는 supervised 66.15%, Soft $k$-Means 70.25%, Masked Soft $k$-Means 69.88%였다. distractor가 있는 setting에서도 Masked Soft $k$-Means가 1-shot 51.38%, 5-shot 69.08%로 가장 robust한 축에 속했다.

논문이 강조하는 실험적 메시지는 세 가지다. 첫째, unlabeled example은 실제로 few-shot 분류 성능을 개선할 수 있다. 둘째, refinement를 test time에만 하는 것보다, 그런 refinement를 염두에 두고 meta-training까지 수행한 모델이 더 좋다. 셋째, distractor가 존재할 때는 단순 Soft $k$-Means보다 masking 기반 접근이 더 안정적이다.

또한 Figure 4와 부록 Figure 6에서 test 시 class당 unlabeled item 수 $M$을 늘릴수록 정확도가 계속 상승하는 경향을 보였다. 특히 모델들은 training 때 $M=5$로 배웠는데도, test에서 $M=25$까지 늘어났을 때 성능이 개선되었다. 이는 모델이 단순히 고정된 episode 크기에 과적합한 것이 아니라, unlabeled sample이 많아질수록 더 나은 refinement를 할 수 있음을 시사한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의 자체가 명확하고 중요하다는 점이다. 기존 few-shot learning과 semi-supervised learning 사이의 빈 공간을 잘 짚었고, “새로운 class에 대한 unlabeled data 활용”이라는 현실적인 시나리오를 episodic meta-learning 틀 안에 자연스럽게 넣었다. 또한 distractor가 있는 경우까지 함께 다뤄, 단순한 이상적 설정에 머물지 않았다는 점도 강점이다.

방법론 측면에서는 Prototypical Networks를 크게 복잡하게 만들지 않으면서 확장했다는 점이 좋다. refinement 규칙이 prototype 계산에 직접 들어가므로 직관적이고, Soft $k$-Means, distractor cluster, masking이라는 세 단계의 변형도 비교적 이해하기 쉽다. 특히 Masked Soft $k$-Means는 “유효한 unlabeled sample만 softly 반영한다”는 설계가 매우 설득력 있고, 실험적으로도 distractor 환경에서 가장 안정적인 결과를 보인다.

실험도 비교적 충실하다. Omniglot, mini ImageNet뿐 아니라 새 데이터셋인 tiered ImageNet을 제안하여 더 현실적인 class split을 제공했다. 특히 tiered ImageNet의 hierarchical split은 train/test class가 semantic하게 지나치게 가까워지는 문제를 줄이려는 시도라는 점에서 의의가 있다. 또한 unlabeled item 수를 변화시키며 분석한 부분도 실용적인 insight를 준다.

한계도 분명하다. 먼저, 방법의 핵심이 prototype refinement이기 때문에, class가 embedding 공간에서 평균 중심으로 잘 요약된다는 ProtoNet의 가정에 여전히 의존한다. class 분포가 multi-modal하거나 복잡하면 평균 prototype 기반 refinement는 충분하지 않을 수 있다. 논문은 이 문제를 직접 다루지 않는다.

또한 distractor cluster 버전은 distractor 전체를 하나의 cluster로 표현하고, 그 중심을 원점에 둔다는 강한 단순화를 사용한다. 저자들도 이것이 지나치게 단순하다고 인정하며, 그래서 masking 모델을 제안한다. 즉 이 버전은 개념적 중간 단계로는 유용하지만, 일반적인 distractor 상황을 잘 모델링한다고 보기 어렵다.

Masked Soft $k$-Means 역시 한계가 있다. mask threshold를 distance 통계량으로부터 MLP가 추정하는데, 왜 그 통계량 조합이 충분한지에 대한 깊은 이론적 설명은 없다. 또한 MLP hyperparameter는 엄밀히 튜닝하지 않았다고 부록에서 밝힌다. 따라서 성능이 더 좋아질 여지는 있지만, 동시에 현재 결과가 설계 선택에 얼마나 민감한지도 완전히 드러나지 않는다.

실험 해석에서도 주의할 점이 있다. 논문은 labeled/unlabeled split을 새로 만들었기 때문에, 이전 few-shot benchmark 수치와 직접 비교해서는 안 된다고 분명히 말한다. 따라서 이 논문의 성과는 “같은 semi-supervised split 안에서의 비교”로 이해해야 한다. 또 query set과 unlabeled set을 분리해 transductive memorization을 피하려 했는데, 이것은 타당한 선택이지만 실제 응용에서는 query 자체가 unlabeled pool이 되는 경우도 있으므로, 그런 setting에서의 동작은 별도 검토가 필요하다.

비판적으로 보면, 제안 방법은 사실상 “meta-trained self-labeling/prototype adjustment”로 해석할 수 있고, 구조적으로 매우 강력한 generative or graph-based semi-supervised inference를 도입한 것은 아니다. 그러나 이 논문의 공헌은 복잡한 모델 제안보다, semi-supervised few-shot learning 문제를 명확히 정의하고 strong baseline 위에서 잘 작동하는 간결한 확장을 보여준 데 있다.

## 6. 결론

이 논문은 few-shot learning episode에 unlabeled set을 추가한 semi-supervised few-shot classification 패러다임을 제안하고, 이를 위해 Prototypical Networks를 확장한 세 가지 방법을 제시했다. 핵심은 labeled support로 만든 초기 prototype을 unlabeled example을 통해 refinement하는 것이며, distractor가 없는 경우에는 모든 변형이 대체로 효과적이고, distractor가 있는 경우에는 Masked Soft $k$-Means가 가장 robust한 성능을 보였다.

또한 저자들은 더 현실적인 class split을 위한 tiered ImageNet을 제안해 데이터셋 측면에서도 기여했다. 전체적으로 이 연구는 “적은 label + 많은 unlabeled sample + 새로운 class”라는 실제 문제를 다루기 위한 중요한 첫걸음으로 볼 수 있다. 향후에는 fast weights, episode-dependent embedding, 더 정교한 distractor modeling, 혹은 hierarchy 활용 방식과 결합되어 실제 응용과 후속 연구에 의미 있는 기반이 될 가능성이 크다.
