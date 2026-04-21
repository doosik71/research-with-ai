# XtarNet: Learning to Extract Task-Adaptive Representation for Incremental Few-Shot Learning

- **저자**: Sung Whan Yoon, Do-Yeon Kim, Jun Seo, Jaekyun Moon
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2003.08561

## 1. 논문 개요

이 논문은 incremental few-shot learning 문제를 다룬다. 이는 이미 학습한 base classes를 잊지 않으면서, 매우 적은 수의 labeled examples만으로 novel classes를 빠르게 학습해야 하는 문제다. 일반적인 few-shot learning은 novel categories를 잘 분류하는 데 초점을 두는 경우가 많지만, incremental few-shot learning에서는 base categories와 novel categories를 함께 다뤄야 하므로 문제가 훨씬 더 어렵다. 특히 novel classes를 학습하는 과정에서 base classes에 대한 성능이 떨어지는 catastrophic forgetting이 큰 장애물이다.

저자들은 기존 방법들이 주로 novel classifier weight를 어떻게 만들 것인가에 집중해 왔다고 지적한다. 반면 이 논문은 classifier만 조정하는 것으로는 충분하지 않으며, 아예 주어진 task에 맞는 representation 자체를 새로 구성해야 한다고 본다. 이를 위해 제안된 방법이 XtarNet이며, 핵심 개념은 task-adaptive representation(TAR)이다. TAR은 pretrained backbone이 주는 기존 feature와, meta-trained module이 새롭게 추출하는 novel feature를 task별로 혼합하여 만든 표현이다.

이 문제가 중요한 이유는 실제 응용에서 새로운 클래스를 적은 데이터만으로 추가해야 하는 경우가 매우 많기 때문이다. 예를 들어 제품 분류, 의료 영상, 로봇 인식처럼 새 범주가 자주 등장하지만 대규모 재학습은 어려운 환경에서는, 기존 지식을 유지하면서 적은 샘플로 새 개념을 학습하는 능력이 매우 중요하다. 이 논문은 그 요구에 대해 representation 수준의 적응이라는 방향을 제시한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 간단히 말해, pretrained backbone이 주는 base feature만으로는 novel task를 충분히 설명하기 어렵기 때문에, 새로운 task에 맞는 novel feature를 추가로 추출하고 이를 base feature와 섞어서 task-adaptive representation을 만들자는 것이다. 저자들은 이 TAR이 base와 novel categories를 동시에 잘 분류하는 데 필요한 정보를 담는다고 주장한다.

구성적으로 보면 XtarNet은 backbone 위에 세 개의 meta-learned module을 올린다. 첫째, MetaCNN은 backbone의 intermediate feature를 입력으로 받아 backbone과는 다른 high-level novel feature를 뽑는다. 둘째, MergeNet은 support set 전체를 보고 현재 task를 요약한 뒤, base feature와 novel feature를 어떤 비율로 섞을지 결정하는 mixture weights를 만든다. 셋째, TconNet은 이렇게 얻은 TAR을 활용해 base classifier와 novel classifier를 task에 맞게 조정한다.

기존 incremental few-shot learning 방법들과의 가장 큰 차별점은 representation을 고정된 것으로 두지 않는다는 점이다. 예를 들어 Imprint, LwoF, AAN 같은 기존 방법은 대체로 backbone이 만든 feature 공간은 유지한 채 novel classifier만 더 잘 만들려고 한다. 반면 XtarNet은 “현재 task에 맞는 feature space를 다시 구성해야 한다”는 관점을 취한다. 즉, 분류기 생성이 아니라 representation extraction 자체를 meta-learning의 대상으로 삼는다.

또 하나 중요한 점은 TAR이 XtarNet 단독 구조에만 쓰이는 것이 아니라, Imprint, LwoF, AAN 같은 기존 방법 위에 plug-in처럼 붙어서도 성능을 향상시킨다는 것이다. 즉 이 논문은 완전히 새로운 classifier 생성법만 제안하는 것이 아니라, 더 일반적인 표현 학습 아이디어를 제안했다고 볼 수 있다.

## 3. 상세 방법 설명

전체 파이프라인은 세 단계로 이해할 수 있다. 먼저 base classes로 backbone과 base classifier를 일반적인 supervised learning으로 pretraining한다. 다음으로 meta-training 단계에서는 episode 단위 학습을 수행한다. 각 episode마다 적은 수의 labeled support samples로 novel classes가 주어지고, query set에는 base와 novel 양쪽 클래스의 샘플이 함께 포함된다. 마지막 test 단계에서는 backbone과 meta-learned modules를 고정한 뒤, 보지 못한 novel classes에 대해 support set만으로 빠르게 적응한다.

입력 이미지 $x$에 대해 pretrained backbone $f_\theta(x) \in \mathbb{R}^D$는 base feature를 출력한다. 그리고 backbone의 intermediate output $a_\theta(x)$를 입력으로 받는 MetaCNN $g(a_\theta(x)) \in \mathbb{R}^D$는 novel feature를 출력한다. 논문은 이 novel feature가 backbone alone으로는 얻기 어려운 task-relevant 정보를 제공한다고 본다.

MergeNet은 support set 전체에서 task 정보를 요약해 mixture weights를 만든다. 각 support sample에 대해 base feature와 novel feature를 concatenate한 뒤, support 전체 평균을 내어 task representation $c$를 만든다.

$$
c = \frac{1}{|S|}\sum_{x \in S}[f_\theta(x), g(a_\theta(x))]
$$

이 $c$를 두 개의 작은 fully connected network인 $r_{\text{pre}}$와 $r_{\text{meta}}$에 넣어 각각 $\omega_{\text{pre}}$와 $\omega_{\text{meta}}$를 얻는다.

$$
\omega_{\text{pre}} = r_{\text{pre}}(c), \quad \omega_{\text{meta}} = r_{\text{meta}}(c)
$$

이 두 벡터는 element-wise gate처럼 작동하여 base feature와 novel feature를 섞는다. 따라서 입력 $x$의 combined feature, 즉 TAR의 기본 형태는

$$
\omega_{\text{pre}} \odot f_\theta(x) + \omega_{\text{meta}} \odot g(a_\theta(x))
$$

가 된다. 여기서 $\odot$는 element-wise product이다. novel class $k$에 대한 support prototype은 이 combined feature의 class-wise average로 계산된다.

$$
c_k^* = \frac{1}{|S_k|}\sum_{x \in S_k}\left[\omega_{\text{pre}} \odot f_\theta(x) + \omega_{\text{meta}} \odot g(a_\theta(x))\right]
$$

이 $c_k^*$가 TAR 기반 novel class summary라고 이해하면 된다.

이제 TconNet은 classifier를 task에 맞게 조정한다. 우선 모든 novel class prototype의 평균

$$
c^* = \frac{1}{N}\sum_{k=N_b+1}^{N_b+N} c_k^*
$$

를 만든 뒤, 이를 입력으로 두 개의 network $h_\gamma$, $h_\beta$가 scaling vector와 bias vector를 생성한다. pretrained base classifier weight $w_i$는 다음처럼 조정된다.

$$
w_i^* = (1 + h_\gamma(c^*)) \odot w_i + h_\beta(c^*)
$$

이 식은 FiLM류의 affine conditioning과 유사하다. 직관적으로는 “현재 novel task를 고려했을 때 base classifier를 얼마나 키우고 이동시킬지”를 정하는 것이다. 이 과정 덕분에 base classifier가 단순 고정되지 않고 현재 episode에 맞게 적응한다.

novel classifier의 초기화는 세 방식이 가능하다고 설명한다. 가장 단순하게는 Imprint처럼 $c_k^*$를 novel class weight로 그대로 둘 수 있다. 또는 LwoF의 attention-based generator를 사용할 수 있다. 논문에서 XtarNet 본체의 최고 성능 버전은 TapNet을 사용한다. TapNet은 meta-learned reference vectors $\{\phi_n\}$와 class prototypes $\{c_n\}$가 특정 projection space $M$에서 잘 정렬되도록 만든다. 논문은 오차

$$
\epsilon_n = \frac{\phi_n}{\|\phi_n\|} - \frac{c_n}{\|c_n\|}
$$

를 정의하고, 이 오차가 projection space $M$에서 0이 되도록, 즉 $\epsilon_n M = 0$이 되도록 $M$을 SVD 기반 null-space로 구한다고 설명한다.

novel classifier는 여기서 끝나지 않고 base classifier와의 간섭을 줄이기 위해 한 번 더 조정된다. novel class $k$에 대해 combined prototype $c_k^*$와 각 conditioned base weight $w_i^*$의 inner product similarity를 계산하여 correlation vector $\sigma_k$를 만든다.

$$
\sigma_i^k = c_k^* \cdot w_i^*
$$

이 $\sigma_k$를 또 다른 network $h_\lambda$에 넣어 $\lambda_k$를 만들고, 이를 이용해 novel classifier가 base classifier 방향과 너무 겹치지 않도록 bias correction을 가한다.

$$
w_i^* = w_i - \sum_{j=1}^{N_b}\frac{\exp(\lambda_i^j)}{\sum_l \exp(\lambda_i^l)} w_j^*
$$

논문 표기상 인덱스가 약간 복잡하지만, 핵심은 novel weight가 conditioned base weights의 가중합만큼 밀려나도록 만들어 base-novel interference를 줄인다는 것이다.

최종 분류는 TapNet 스타일의 projection space $M$ 안에서 수행된다. query $x$에 대해 combined feature를 만든 뒤, projected classifier weight와의 Euclidean distance를 계산한다.

$$
D_i(x) = d\left(w_i^* M,\; [\omega_{\text{pre}} \odot f_\theta(x) + \omega_{\text{meta}} \odot g(a_\theta(x))] M\right)
$$

그 다음 softmax 형태로 posterior probability를 계산한다.

$$
p(y=i \mid x) = \frac{\exp(-D_i(x))}{\sum_l \exp(-D_l(x))}
$$

그리고 query set 전체에 대한 cross entropy loss로 MetaCNN, MergeNet, TconNet 및 episode별 novel weights를 학습한다. 중요한 점은 backbone은 pretraining 후 고정되고, 추가 모듈들만 episodic meta-training을 통해 inductive bias를 학습한다는 것이다.

구현 측면에서 miniImageNet에서는 ResNet12 backbone과 single residual block MetaCNN을 사용했고, tieredImageNet에서는 ResNet18 backbone과 two-block MetaCNN을 사용했다. MergeNet은 두 개의 4-layer fully connected network, TconNet은 base conditioning용 3-layer FC 두 개와 novel conditioning용 3-layer FC 하나로 구성된다. 논문이 제공한 supplemental material에 따르면, miniImageNet에서는 concat feature 길이가 512, combined feature 길이가 256이며, tieredImageNet에서는 이 크기들이 두 배로 증가한다.

## 4. 실험 및 결과

실험은 miniImageNet과 tieredImageNet의 incremental few-shot classification 설정에서 수행되었다. miniImageNet은 64 base classes 위에 5-way novel task를 얹는 64+5-way, tieredImageNet은 200 base classes 위에 5-way novel task를 얹는 200+5-way 설정이다. 각 episode에서는 novel classes 5개가 선택되고, 각 클래스당 1-shot 또는 5-shot support examples가 주어진다. query set은 base와 novel에서 같은 수만큼 샘플을 뽑아 합친다.

평가 지표는 단순 accuracy 하나만이 아니다. 논문은 base와 novel을 함께 분류하는 joint accuracy를 기본 지표로 사용하고, joint setting에서의 성능 저하를 보기 위해 $\Delta$도 함께 제시한다. $\Delta$는 base-only accuracy와 novel-only accuracy 대비 joint classification 시 생기는 gap의 평균이다. 따라서 $\Delta$가 0에 가까울수록 base와 novel의 균형이 잘 맞는다고 볼 수 있고, 음수 폭이 작을수록 좋다.

miniImageNet 64+5-way 결과를 보면, 1-shot에서는 XtarNet이 55.28 $\pm$ 0.33%, $\Delta$=-13.13%$를 기록했다. accuracy 기준으로는 Attention Attractor Networks(AAN)의 54.95%보다 약간 높지만, $\Delta$는 AAN의 -11.84%보다 다소 나쁘다. 즉 1-shot miniImageNet에서는 absolute accuracy는 가장 높지만, base-novel 균형성에서는 최고는 아니다. 반면 5-shot에서는 XtarNet이 66.86 $\pm$ 0.31%, $\Delta$=-10.34%로 accuracy와 균형성 모두에서 가장 좋은 결과를 보였다.

tieredImageNet 200+5-way에서는 개선 폭이 훨씬 크다. 1-shot에서 XtarNet은 61.37 $\pm$ 0.36%, $\Delta$=-1.85%를 기록해 AAN의 56.11%, -6.11%를 크게 앞섰다. 5-shot에서도 69.58 $\pm$ 0.32%, -1.79%로 AAN의 65.52%, -4.48%보다 뚜렷하게 좋다. 특히 $\Delta$가 매우 작아져 base와 novel을 함께 분류할 때 균형이 크게 향상되었음을 보여 준다.

논문의 또 다른 중요한 실험은 TAR을 기존 방법에 붙였을 때의 효과다. Imprint 단독은 tieredImageNet 5-shot에서 53.87%에 불과했지만, TAR을 붙인 Proposed method with Imprint는 65.56%까지 상승한다. 이는 11%p 이상 향상된 매우 큰 개선이다. LwoF와 결합한 경우도 일관된 향상이 나타난다. Supplementary에서는 AAN과 결합한 결과도 제시하며, 5-shot tieredImageNet에서 68.06%, $\Delta$=-1.72%를 달성했다고 보고한다. 즉 TAR은 특정 classifier 설계에 종속되지 않는 일반적 개선 요소로 제시된다.

Ablation study는 TAR의 각 구성요소가 실제로 기여하는지 보여 준다. tieredImageNet 5-shot, Imprint 기반 실험에서 baseline Imprint는 53.87%였다. 여기에 MetaCNN만 추가하면 63.15%로 급상승한다. 다시 MergeNet까지 추가하면 64.62%, 마지막으로 TconNet까지 모두 넣으면 65.56%가 된다. 즉 가장 큰 기여는 novel feature를 추출하는 MetaCNN에서 나오고, MergeNet과 TconNet이 그 위에 추가 개선을 제공한다는 해석이 가능하다.

정성 분석에서도 TAR의 의미를 강조한다. t-SNE 분석에 따르면 backbone feature만 사용하면 base queries는 잘 분리되지만 novel queries는 서로 뭉쳐 구분이 어렵다. 반면 MetaCNN feature는 novel class separation을 개선하고, combined TAR feature는 base clustering을 유지하면서 novel clustering도 개선한다. 이는 TAR이 단순 average feature가 아니라 base와 novel 모두에 유리한 feature geometry를 만든다는 논문의 주장을 시각적으로 뒷받침한다.

정량 분석으로는 clustering quality를 위해 SSE와 normalized SSE를 제시했다. tieredImageNet 5-shot에서 Imprint 대비 TAR-combined method는 base SSE를 10.65에서 4.94로, novel SSE를 3.96에서 2.38로 낮췄다. reduction ratio는 base -53.6%, novel -39.9%였다. normalized SSE 기준으로는 novel 쪽 개선이 특히 두드러져 -28.3% 감소했다. 이는 단순히 클래스 내부가 더 조밀해졌을 뿐 아니라 인접 클래스와의 간섭까지 고려했을 때 novel 분리에 더 강한 효과가 있다는 뜻이다.

Entropy 분석에서도 TAR의 효과가 크다. tieredImageNet에서 cross entropy $E$와 Shannon entropy $H$를 비교했을 때, Imprint 대비 Combined method는 base와 novel 모두에서 훨씬 더 낮은 값을 보였다. 예를 들어 novel query의 cross entropy는 4.97에서 0.92로 줄었고, Shannon entropy도 5.31에서 1.37로 감소했다. 이는 TAR이 더 정확하고 더 confident한 예측을 가능하게 한다는 의미다.

Supplementary에서는 시간 복잡도 측면의 장점도 언급한다. AAN은 novel classifier를 얻기 위해 inner-loop optimization이 여러 번 필요하고, training에서는 recurrent back-propagation까지 들어가므로 느리다. 반면 XtarNet은 recursive optimization 없이 forward-style meta-modules로 task adaptation을 수행한다. 논문은 이 점을 실험 수치로 직접 계량하지는 않았지만, 구조적으로 latency advantage가 있다고 주장한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정에 맞는 관점 전환이다. 기존 incremental few-shot learning이 classifier generation 중심이었다면, 이 논문은 representation 자체가 task-adaptive해야 한다고 주장하고 실제 성능 향상으로 이를 뒷받침했다. 특히 tieredImageNet에서 accuracy와 $\Delta$ 모두 크게 개선된 것은 단순히 novel class를 더 잘 맞추는 수준이 아니라 base-novel trade-off를 실제로 완화했음을 보여 준다.

둘째 강점은 modularity다. TAR은 XtarNet 단독 구조뿐 아니라 Imprint, LwoF, AAN과 결합해서도 성능을 올린다. 이는 제안이 특정 classifier trick이 아니라 보다 일반적인 feature adaptation 메커니즘임을 시사한다. 연구 관점에서는 확장성이 좋고, 실용 관점에서는 기존 시스템에 붙이기 쉬운 장점이 있다.

셋째, 분석이 비교적 충실하다. 단순 accuracy table만 제시하지 않고, ablation, t-SNE, SSE, entropy 분석으로 TAR이 왜 도움이 되는지 설명하려고 시도한다. 특히 MetaCNN이 novel clustering을 개선하고 combined feature가 base 성능을 유지한다는 해석은 논문의 핵심 가설과 잘 연결된다.

하지만 한계도 분명하다. 첫째, 방법이 꽤 복잡하다. backbone 위에 MetaCNN, MergeNet, TconNet, 그리고 TapNet projection까지 얹기 때문에 구성 요소가 많고, 실제 구현 난이도와 튜닝 부담이 크다. 논문은 state-of-the-art 성능을 보이지만, 그만큼 단순성과 해석 가능성은 떨어진다.

둘째, 최고 성능 버전이 TapNet projection에 의존한다는 점도 중요하다. 즉 XtarNet의 핵심 아이디어는 TAR이지만, 최종 성능은 TapNet 기반 novel classifier 및 projection space와 결합되었을 때 가장 좋다. 따라서 “TAR 자체의 순수 기여”와 “TapNet과의 시너지”가 어느 정도 분리되는지는 완전히 명확하지 않다. 물론 ablation이 일부 답을 주지만, strongest setting이 여러 아이디어의 조합이라는 점은 감안해야 한다.

셋째, 실험 도메인이 image classification의 표준 benchmark 두 개로 한정되어 있다. incremental few-shot learning 자체가 niche한 설정이므로 이는 자연스러운 선택이지만, 다른 데이터 분포, class imbalance, domain shift, 장기적 multi-session incremental setting 등으로 일반화되는지는 이 논문만으로는 판단하기 어렵다. 논문은 episode 단위 설정에서의 성능을 보였지만, 더 현실적인 연속적 class arrival 시나리오를 다루지는 않는다.

넷째, mixture weight의 의미에 대한 해석은 어느 정도 제시되지만, 왜 특정 task에서 어떤 feature dimension이 강조되는지에 대한 보다 깊은 분석은 부족하다. Supplementary에서 PCA와 magnitude analysis를 제공하긴 하지만, 이는 관찰적 분석에 가깝다. TAR이 정확히 어떤 semantic variation을 포착하는지까지는 논문에서 명확히 설명하지 않는다.

또 하나 비판적으로 볼 부분은 baseline 재현 방식이다. AAN 결과는 논문 reported numbers를 사용했고, 다른 방법은 재현했다고 적혀 있다. 이런 비교는 일반적으로 허용되지만, 동일한 backbone 세팅과 training budget에서 완전 공정 비교가 이루어졌는지는 본문만으로는 완전히 검증하기 어렵다. 다만 이 점은 제공된 텍스트 범위에서 추가로 판단할 수 없다.

## 6. 결론

이 논문은 incremental few-shot learning에서 핵심 병목이 classifier 생성만이 아니라 representation adaptation 자체에 있다고 보고, 이를 해결하기 위해 task-adaptive representation(TAR) 개념을 제안했다. XtarNet은 pretrained backbone의 base feature와 meta-trained MetaCNN의 novel feature를 MergeNet으로 task별 혼합하고, TconNet으로 base 및 novel classifier를 조정한다. 그 결과 base knowledge를 유지하면서도 novel classes를 적은 샘플로 빠르게 학습할 수 있도록 한다.

실험적으로는 miniImageNet과 tieredImageNet에서 강한 성능을 보였고, 특히 tieredImageNet에서는 accuracy와 base-novel balance를 모두 크게 향상시켰다. 또한 TAR을 기존 incremental few-shot learning 방법들에 결합해도 성능이 좋아져, 이 아이디어가 비교적 일반적인 가치가 있음을 보여 주었다.

실제 적용 측면에서는, 이미 학습된 시각 인식 시스템에 소수 샘플 기반 새 클래스를 계속 추가해야 하는 환경에서 유용할 가능성이 크다. 향후 연구 측면에서는 TAR 개념을 더 단순한 구조로 구현하거나, long-horizon continual setting, domain shift, multimodal setting으로 확장하는 방향이 자연스럽다. 요약하면 이 논문은 incremental few-shot learning에서 “분류기만 바꾸는 것”을 넘어 “task에 맞는 표현을 다시 만든다”는 중요한 관점을 제시한 작업이다.
