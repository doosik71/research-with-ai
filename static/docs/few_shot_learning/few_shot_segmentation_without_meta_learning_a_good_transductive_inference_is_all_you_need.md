# Few-Shot Segmentation Without Meta-Learning: A Good Transductive Inference Is All You Need?

- **저자**: Malik Boudiaf, Hoel Kervadec, Ziko Imtiaz Masud, Pablo Piantanida, Ismail Ben Ayed, Jose Dolz
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2012.06166

## 1. 논문 개요

이 논문은 few-shot segmentation에서 성능을 좌우하는 핵심이 꼭 meta-learning일 필요는 없으며, 오히려 **test-time inference를 어떻게 하느냐**가 매우 중요하다고 주장한다. 기존 few-shot segmentation 연구들은 주로 episodic training, prototype design, support-query interaction module 같은 학습 구조에 집중해 왔다. 그러나 저자들은 이런 방향이 실제 환경에서는 몇 가지 약점을 가진다고 본다. 예를 들어, 학습 시 가정한 shot 수와 테스트 시 shot 수가 다를 수 있고, base classes와 novel classes가 같은 데이터셋 분포를 따른다는 가정도 현실적이지 않을 수 있다.

논문이 다루는 문제는 다음과 같다. base classes로 feature extractor를 학습한 뒤, novel class에 대해 소수의 support image와 하나의 unlabeled query image가 주어졌을 때 query를 정확히 segment해야 한다. 여기서 중요한 질문은 “복잡한 meta-learning 없이도 좋은 few-shot segmentation이 가능한가?”이다. 저자들의 답은 긍정적이다. 이들은 **RePRI (Region Proportion Regularized Inference)** 라는 transductive inference 방법을 제안하고, support pixel뿐 아니라 query image의 unlabeled pixel 분포까지 활용하면 기존 SOTA를 넘을 수 있음을 보인다.

이 문제가 중요한 이유는 few-shot segmentation이 의료영상, 로보틱스, 드문 객체 인식처럼 annotation이 매우 비싼 환경에서 유용하기 때문이다. 또한 실제 응용에서는 test task의 구조나 데이터 도메인이 학습 시점과 다를 수 있으므로, 학습 편향이 적고 test-time 적응력이 좋은 방법이 필요하다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 단순하다. **좋은 embedding만 있으면, query image 자체의 통계를 활용하는 transductive inference만으로도 강한 few-shot segmentation 성능을 얻을 수 있다**는 것이다. 이는 few-shot classification 분야에서 제기된 “good embedding is all you need”와 유사한 문제의식을 segmentation으로 확장한 셈이다.

기존 접근법은 대체로 support에서 class prototype을 만들고, 이를 query와 비교하는 구조를 meta-learning 방식으로 학습한다. 반면 이 논문은 base training 단계에서 복잡한 episodic training을 버리고, 표준적인 cross-entropy supervision으로 feature extractor를 학습한다. 그리고 실제 few-shot task를 받을 때마다 support와 query의 feature 위에 **간단한 선형 classifier**를 얹고, 이를 task별로 직접 최적화한다.

여기서 중요한 차별점은 query image를 완전히 수동적으로 처리하지 않는다는 점이다. 저자들은 query의 unlabeled pixel에 대해 posterior entropy를 줄이고, 예측된 foreground 비율이 어떤 전역적인 비율 파라미터와 맞도록 regularization을 건다. 이때 단순 entropy minimization만으로는 segmentation에서 trivial solution이 생기기 쉽다고 보고, **foreground/background region proportion에 대한 KL regularization**을 함께 넣는다. 저자들은 이 항이 실제로 핵심적인 regularizer라고 주장하며, 실험도 그 주장을 뒷받침한다.

즉, 이 논문이 제안하는 관점은 “더 정교한 학습 구조”보다 “더 잘 설계된 test-time adaptation”에 무게를 둔다. 이 점이 본 논문의 가장 중요한 메시지다.

## 3. 상세 방법 설명

전체 파이프라인은 두 단계로 구성된다. 먼저 base dataset $D_{\text{base}}$의 클래스들로 feature extractor $f_\phi$를 학습한다. 이때 사용하는 학습 방식은 특별하지 않다. episodic training 없이, 전체 base dataset에 대해 일반적인 semantic segmentation용 cross-entropy로 supervised training을 수행한다. 이후 novel class few-shot task가 주어지면, support image들의 annotation과 unlabeled query image를 사용해 task-specific classifier $\theta$를 최적화한다.

few-shot task는 $K$개의 fully annotated support image와 1개의 unlabeled query image로 구성된다. support와 query를 feature extractor에 통과시켜 feature map $z_k$, $z_Q$를 얻는다. 여기서 목표는 feature space에서 foreground와 background를 구분하는 classifier $\theta$를 학습하는 것이다.

논문의 핵심 objective는 다음과 같다.

$$
\min_{\theta} \; CE + \lambda_H H + \lambda_{KL} D_{KL}
$$

여기서 각 항의 역할은 다음과 같다.

첫 번째 항 $CE$는 support image의 labeled pixel에 대한 cross-entropy다. support의 downsampled label $\tilde y_k$와 classifier prediction $p_k$ 사이의 supervised loss이다.

$$
CE = - \frac{1}{K|\Psi|} \sum_{k=1}^{K} \sum_{j \in \Psi} \tilde y_k(j)^\top \log(p_k(j))
$$

이 항만 최소화하면 support에는 잘 맞지만 query로 일반화되지 않는 문제가 생긴다. 특히 1-shot에서는 support 하나에 과적합되어 query에서 foreground 일부만 활성화되는 경향이 있다. 논문 Figure 1의 qualitative result도 이를 보여준다.

두 번째 항 $H$는 query image pixel에 대한 Shannon entropy이다.

$$
H = - \frac{1}{|\Psi|} \sum_{j \in \Psi} p_Q(j)^\top \log(p_Q(j))
$$

이 항은 query 예측을 더 confident하게 만들고, decision boundary가 feature density가 낮은 영역에 놓이도록 유도한다. 이는 semi-supervised learning의 entropy minimization과 유사하다. 다만 segmentation에서는 이 항만 추가해도 여전히 degenerate solution이 발생할 수 있고, 경우에 따라 더 악화될 수도 있다고 저자들은 설명한다.

세 번째 항이 본 논문의 핵심인 $D_{KL}$이다.

$$
D_{KL} = \hat p_Q^\top \log\left(\frac{\hat p_Q}{\pi}\right),
\qquad
\hat p_Q = \frac{1}{|\Psi|} \sum_{j \in \Psi} p_Q(j)
$$

여기서 $\hat p_Q$는 query 전체에 대해 평균 낸 background/foreground posterior이고, $\pi \in [0,1]^2$는 query image의 foreground/background 비율에 대한 목표 분포이다. 즉 이 항은 query 전체에서 예측된 foreground 크기가 어떤 전역 비율과 맞도록 강제한다. 저자들은 이 항이 support 과적합과 trivial solution을 막는 강력한 regularizer라고 본다.

분류기 자체는 매우 단순한 선형 classifier다. 매 iteration $t$에서 classifier parameter는 $\theta^{(t)} = \{w^{(t)}, b^{(t)}\}$이고, $w^{(t)} \in \mathbb{R}^C$는 foreground prototype, $b^{(t)} \in \mathbb{R}$는 bias이다. query와 support 모두 같은 classifier를 사용한다. 각 pixel의 foreground score는 cosine similarity 기반으로 계산된다.

$$
p_\bullet^{(t)}(j) :=
\begin{pmatrix}
1 - s_\bullet^{(t)}(j) \\
s_\bullet^{(t)}(j)
\end{pmatrix}
$$

그리고

$$
s_\bullet^{(t)}(j) = \text{sigmoid}\left(\tau \left[\cos(z_\bullet(j), w^{(t)}) - b^{(t)}\right]\right)
$$

이다. 여기서 $\tau$는 temperature hyper-parameter다. 결국 query segmentation은 feature와 prototype의 cosine similarity에 기반한 binary classification 문제로 바뀐다.

초기화도 중요하다. 초기 prototype $w^{(0)}$는 support foreground feature의 평균으로 설정한다. 즉 support의 annotated foreground pixel만 모아 평균한 벡터다. 초기 bias $b^{(0)}$는 query image에서 foreground soft prediction의 평균으로 설정한다. 그 뒤 gradient descent로 $w$와 $b$를 50 iteration 동안 업데이트한다.

$\pi$의 설정은 조금 더 흥미롭다. query의 실제 foreground 비율을 모르는 것이 일반적이므로, 저자들은 $\pi$를 query prediction의 marginal distribution으로부터 jointly estimate한다. 식만 보면 $\pi$에 대해 최소화할 때 $\pi^{(t)} = \hat p_Q^{(t)}$가 된다. 그러나 매 step마다 계속 바꾸는 대신, 실험적으로는 한 번만 업데이트해도 충분했다고 한다. 그래서 다음처럼 쓴다.

$$
\pi^{(t)} =
\begin{cases}
\hat p_Q^{(0)} & 0 \le t \le t_\pi \\
\hat p_Q^{(t_\pi)} & t > t_\pi
\end{cases}
$$

즉 초기에 한 번 추정한 비율을 쓰다가, 일정 step $t_\pi$ 이후 더 refined된 prediction으로 한 번 교체한다. 논문에서는 $t_\pi = 10$을 사용했다. 직관적으로는, 초기 query prediction은 blurry하지만 entropy minimization과 support supervision을 거치며 조금씩 foreground shape가 정리되고, 그 결과 나온 $\hat p_Q^{(t_\pi)}$가 더 나은 region proportion estimate가 된다는 것이다.

또한 upper bound로서 **oracle case**도 다룬다. 이 경우 query ground truth를 통해 정확한 foreground/background 비율 $\pi^\*$를 알고 있다고 가정한다.

$$
\pi^\* = \frac{1}{|\Psi|} \sum_{j \in \Psi} \tilde y_Q(j)
$$

이 oracle 실험은 “정확한 region proportion 정보가 있다면 성능이 어디까지 올라갈 수 있는가”를 보여주는 용도이며, 실제로 매우 큰 성능 향상이 나타난다.

## 4. 실험 및 결과

실험은 PASCAL-5$^i$와 COCO-20$^i$에서 수행되었다. PASCAL-5$^i$는 PASCAL VOC 2012 기반으로 20개 클래스를 4개 fold로 나누고, fold마다 15개 base class와 5개 novel class를 사용한다. COCO-20$^i$는 MS-COCO 기반으로 더 많은 클래스와 더 복잡한 장면을 포함하는 더 어려운 benchmark다. 역시 4-fold split을 사용한다.

backbone은 PSPNet 기반이며, ResNet-50과 ResNet-101을 사용했다. base training은 standard cross-entropy로 수행한다. PASCAL-5$^i$는 100 epoch, COCO-20$^i$는 20 epoch 학습했고, optimizer는 SGD, learning rate는 $2.5 \times 10^{-3}$, cosine decay를 적용했다. label smoothing $\epsilon = 0.1$을 사용했고, augmentation은 random mirror flip만 사용했다. multi-scale이나 deep supervision은 사용하지 않았다.

inference 시에는 모든 이미지를 $417 \times 417$로 resize하고, penultimate layer feature 위에서 classifier $\theta$를 task별로 학습한다. ResNet-50의 경우 feature map 크기는 $53 \times 53 \times 512$다. classifier 학습은 SGD with learning rate 0.025, 총 50 iterations 수행한다. $\lambda_H$와 $\lambda_{KL}$는 기본적으로 $1/K$로 두고, $t \ge t_\pi$ 이후에는 $\lambda_{KL}$을 1만큼 증가시킨다. 평가지표는 classwise IoU를 평균한 mIoU이며, 각 fold에서 5 runs of 1000 tasks를 수행해 평균을 보고했다.

표준 benchmark 결과를 보면, 1-shot에서는 RePRI가 기존 최상위권 방법들과 경쟁력 있는 수준이다. 예를 들어 PASCAL-5$^i$에서 ResNet-50 기준 RePRI는 mean mIoU 59.1이고, PFENet은 60.8이다. 따라서 1-shot에서는 최고 성능은 아니지만 거의 근접하다. 그러나 5-shot에서는 RePRI가 훨씬 강하다. 같은 설정에서 RePRI는 66.8, PFENet은 61.9로 약 5%p 차이로 앞선다. ResNet-101에서도 RePRI는 65.6으로 PFENet 61.4보다 높다.

COCO-20$^i$에서도 비슷한 경향이 나온다. 1-shot에서는 RePRI 34.0, PFENet 35.8로 약간 낮다. 그러나 5-shot에서는 RePRI 42.1, PFENet 39.0으로 우세하다. 즉 이 방법은 support 수가 늘어날수록 더 잘 작동한다. 논문은 이것을 “meta-learning 기반 방법은 shot 수가 달라질 때 saturation이 오지만, RePRI는 support 정보를 더 효과적으로 활용한다”는 근거로 해석한다.

10-shot 실험은 이 논문의 주장에 특히 중요하다. 기존 메타러닝 방법들은 train-time shot과 test-time shot이 다르면 성능이 쉽게 포화되거나 잘 늘지 않는다. Table 3에 따르면 PASCAL-5$^i$에서 RPMM은 1-shot 56.3, 5-shot 57.3, 10-shot 57.6으로 거의 늘지 않는다. PFENet도 60.8, 61.9, 62.1로 증가폭이 작다. 반면 RePRI는 59.1, 66.8, 68.2로 큰 폭으로 향상된다. COCO-20$^i$에서도 RePRI는 34.0, 42.1, 44.4로 증가하며, 기존법보다 support 증가의 이점을 더 잘 활용한다.

oracle 결과는 더 인상적이다. PASCAL-5$^i$에서 Oracle-RePRI는 1-shot 73.3, 5-shot 77.9, 10-shot 78.6이고, COCO-20$^i$에서도 45.1, 55.5, 58.7까지 올라간다. 이는 단순한 linear classifier 위에서도, query foreground proportion만 정확히 알 수 있다면 성능이 크게 좋아질 수 있음을 의미한다. 저자들은 이것을 통해 “좋은 feature와 적절한 inference regularization만으로도 강력한 few-shot segmentation이 가능하다”고 주장한다.

논문은 더 현실적인 **domain shift setting**도 제안한다. 학습은 COCO-20$^i$ folds에서 하고, 테스트는 PASCAL-VOC novel classes에서 하는 cross-domain setting이다. 이는 class shift뿐 아니라 image distribution shift도 포함한다. 이 실험에서 RePRI는 1-shot 63.2, 5-shot 67.7로 PFENet의 61.1, 63.4보다 높다. 특히 5-shot에서 4%p 이상 개선된다. 저자들은 이를 근거로 meta-learning 방식이 도메인 변화에 약할 수 있으며, RePRI가 더 현실적인 few-shot segmentation 조건에서도 강하다고 본다.

ablation study도 핵심 주장을 잘 뒷받침한다. PASCAL-5$^i$ 1-shot에서 $CE$만 사용하면 mean mIoU가 38.5에 불과하다. $CE + H$를 쓰면 48.0으로 좋아지지만, 최종 full loss인 $CE + H + D_{KL}$은 59.1까지 오른다. 5-shot에서도 58.0, 58.4, 66.8 순이다. 즉 query entropy만으로는 부족하고, region proportion regularizer가 결정적이다.

또한 $\pi$ 추정 오차에 대한 분석도 있다. 논문은 foreground proportion relative error를

$$
\delta^{(t)} = \frac{\pi^{(t)}_1}{\pi^\*_1} - 1
$$

로 정의한다. 초기 예측은 대체로 foreground 크기를 과대추정하지만, 10 step 정도 최적화한 후에는 더 정확해진다. 흥미로운 점은 oracle 수준의 정확한 비율이 없어도 성능이 크게 좋아질 수 있다는 것이다. Figure 3b에 따르면 foreground size estimate가 oracle 대비 대략 $-10\%$에서 $+30\%$ 범위의 오차만 가져도 70% 이상의 mIoU를 달성할 수 있다. 이는 향후 size estimation을 조금만 더 잘해도 큰 성능 향상이 가능하다는 강한 신호이다.

계산 비용도 비교했다. inference 시 task마다 최적화를 수행하므로 기존의 one-forward-pass 방식보다 느리다. PASCAL-5$^i$ 기준 1-shot에서 RePRI는 12.8 FPS, PFENet은 15.9 FPS, RPMM은 18.2 FPS다. 5-shot에서는 각각 4.4, 5.1, 9.4 FPS다. 느려지긴 하지만, 논문은 이 정도 속도 저하는 실용적으로 수용 가능한 수준이라고 본다. 최적화 대상이 $512$차원 prototype 벡터와 scalar bias뿐이므로, 완전히 무거운 test-time adaptation은 아니라는 점을 강조한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 few-shot segmentation에서 너무 당연하게 받아들여지던 meta-learning 중심 관점을 정면으로 다시 묻는다는 점이다. 저자들은 단순한 supervised base training과 task-specific transductive inference만으로도 강한 성능을 얻을 수 있음을 실험적으로 설득력 있게 보였다. 특히 5-shot과 10-shot에서 성능 향상이 크고, domain shift 상황에서도 강하다는 점은 실질적인 의미가 크다. 즉 이 방법은 “benchmark에 맞춘 meta-learning”보다 “테스트 시 정보 활용”이 더 본질적일 수 있다는 중요한 메시지를 준다.

또 다른 강점은 방법이 모듈형이라는 점이다. RePRI는 특정 architecture나 학습 방식에 강하게 묶여 있지 않고, 임의의 trained feature extractor 위에 얹을 수 있다. 따라서 기존 few-shot segmentation 모델의 embedding 위에도 적용 가능할 여지가 있다. 또한 loss 구성요소별 ablation, domain shift, 10-shot, oracle, misestimation study까지 실험 설계가 비교적 탄탄하다.

반면 한계도 분명하다. 첫째, query proportion $\pi$가 중요한데, 실제 환경에서는 이를 정확히 알 수 없다. 논문은 self-estimated $\pi$만으로도 성능이 좋다고 보였지만, 여전히 이 항의 효과는 적절한 비율 추정에 크게 의존한다. oracle 결과가 지나치게 좋은 만큼, 반대로 말하면 현재 방법의 병목이 정확한 size prior 부족이라는 뜻이기도 하다.

둘째, inference-time optimization이 필요하다. 최적화 대상이 작아서 상대적으로 가볍긴 하지만, 그래도 forward-only 방식보다 느리다. 대규모 실시간 시스템에서는 부담이 될 수 있다.

셋째, 논문은 1-way few-shot segmentation에 초점을 둔다. 다중 novel class가 동시에 존재하는 더 복잡한 설정에서 같은 방식이 얼마나 잘 확장되는지는 이 텍스트만으로는 확인할 수 없다. 또한 $\pi$를 왜 한 번만 업데이트하는 것이 충분한지, 혹은 더 정교한 update scheme이 어떤 영향을 미치는지는 제한적으로만 다뤄진다.

넷째, 본문에서 주장하는 “meta-learning보다 inference가 더 중요하다”는 메시지는 상당히 강하지만, 이것이 모든 setting에서 성립한다고 일반화하기는 어렵다. 실제로 1-shot에서는 일부 benchmark에서 PFENet이 더 높은 수치를 보인다. 따라서 이 논문은 meta-learning이 불필요하다고 완전히 증명한다기보다, **shot 수가 늘거나 domain shift가 있을 때 meta-learning의 이점이 약해질 수 있으며, transductive inference가 강력한 대안이 된다**는 정도로 해석하는 것이 더 정확하다.

## 6. 결론

이 논문은 few-shot segmentation에서 복잡한 meta-learning 없이도 강한 성능을 낼 수 있음을 보인 작업이다. 핵심 기여는 support supervision, query entropy minimization, 그리고 foreground proportion KL regularization을 결합한 **RePRI transductive inference**를 제안한 점이다. 이 방법은 1-shot에서는 경쟁력 있는 수준, 5-shot과 10-shot에서는 기존 SOTA를 크게 넘는 결과를 보였고, domain shift 설정에서도 우수했다.

더 중요한 의미는 연구 방향의 재설정에 있다. 저자들은 few-shot segmentation의 핵심을 training episode 설계가 아니라 test-time adaptation과 query-aware regularization에서 찾는다. 또한 oracle 실험은 query object size와 같은 전역 prior가 매우 강력한 signal임을 보여주며, 향후 연구에서 better proportion estimation, stronger test-time optimization, semi-supervised constraint design 같은 방향이 매우 유망함을 시사한다.

요약하면, 이 논문은 “few-shot segmentation에는 반드시 meta-learning이 필요하다”는 통념을 약화시키고, **좋은 feature extractor와 잘 설계된 transductive inference만으로도 매우 강한 baseline을 만들 수 있다**는 점을 분명하게 보여준다.
