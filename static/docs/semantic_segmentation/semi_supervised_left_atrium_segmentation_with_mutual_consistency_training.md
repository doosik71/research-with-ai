# Semi-supervised Left Atrium Segmentation with Mutual Consistency Training

- **저자**: Yicheng Wu, Minfeng Xu, Zongyuan Ge, Jianfei Cai, Lei Zhang
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2103.02911

## 1. 논문 개요

이 논문은 3D MR 영상에서 left atrium(LA)을 분할하는 문제를 반지도학습(semi-supervised learning)으로 다룬다. 의료 영상 분할에서는 정확한 voxel-level annotation을 대량으로 확보하는 비용이 매우 크기 때문에, 적은 수의 라벨 데이터와 많은 비라벨 데이터를 함께 활용하는 방법이 중요하다. 저자들은 기존 반지도 분할 방법들이 비라벨 데이터의 어려운 영역, 예를 들어 작은 가지 구조(small branches)나 경계가 흐린 부분(blurred edges)을 학습 중 충분히 중시하지 않는다고 지적한다.

논문의 핵심 문제의식은 다음과 같다. 모델이 적은 라벨 데이터로 학습될 때 불확실한 예측은 주로 어려운 영역에 집중되며, 반대로 더 많은 라벨로 학습된 모델은 이런 불확실성이 줄어든다. 저자들은 이 관찰을 바탕으로, 비라벨 데이터의 어려운 영역에서 나타나는 예측 불확실성 자체가 모델 일반화를 높이는 데 유용한 추가 감독 신호가 될 수 있다고 본다. 즉, 기존처럼 불확실한 영역을 피해야 할 노이즈로만 보지 않고, 오히려 가장 가치 있는 학습 신호로 활용하자는 것이 이 논문의 출발점이다.

이 문제는 실제 의료 영상 분석에서 중요하다. left atrium segmentation은 이후 정량 분석과 computer-aided diagnosis(CAD)의 기반 단계가 되므로, 작은 구조 누락이나 경계 오류가 downstream 분석에 직접 영향을 줄 수 있다. 따라서 적은 라벨 데이터 환경에서도 어려운 영역을 잘 복원하는 반지도 분할 기법은 실용적 가치가 크다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 두 개의 서로 약간 다른 decoder가 같은 입력에 대해 낸 예측 차이를 이용해 model-based epistemic uncertainty를 근사하고, 이 차이를 다시 학습 신호로 바꾸어 두 decoder가 서로 일관되고 낮은 엔트로피의 예측을 내도록 만드는 것이다. 이를 위해 저자들은 Mutual Consistency Network(MC-Net)를 제안한다.

기존 teacher-student 계열 방법이나 Monte Carlo dropout 기반 uncertainty estimation과 비교하면, 이 논문은 uncertainty를 얻기 위해 반복적인 stochastic inference를 수행하지 않는다. 대신 하나의 encoder와 두 개의 약간 다른 decoder를 둠으로써, 같은 backbone 내부에서 구조적 다양성을 만들고 두 출력의 discrepancy를 uncertainty의 대용값으로 사용한다. 이는 계산량을 줄이면서도 uncertainty 정보를 훈련 과정에 넣을 수 있게 한다.

또 하나의 핵심은 cycled pseudo label(CPL) 설계다. 각 decoder의 확률 출력을 sharpening function으로 부드러운 pseudo label로 바꾼 뒤, decoder A가 만든 soft pseudo label로 decoder B를 감독하고, decoder B가 만든 soft pseudo label로 decoder A를 감독한다. 이 상호 감독(mutual consistency)은 단순히 두 출력이 같아지게만 하는 것이 아니라, 예측 분포의 엔트로피를 줄이는 방향으로 작동한다. 저자들의 주장에 따르면, 이런 구조는 비라벨 데이터의 어려운 영역에서 더 일반화된 feature를 학습하게 만든다.

## 3. 상세 방법 설명

전체 구조는 하나의 encoder $\Theta_e$와 두 개의 decoder $\Theta_{dA}$, $\Theta_{dB}$로 이루어진다. backbone은 V-Net이며, 차이는 decoder 쪽에 있다. 첫 번째 decoder $\Theta_{dA}$는 원래 V-Net처럼 transposed convolution을 사용해 up-sampling을 수행한다. 두 번째 decoder $\Theta_{dB}$는 tri-linear interpolation을 사용하여 feature map을 확장한다. 저자들은 이처럼 decoder 구조를 약간 다르게 두는 것이 segmentation sub-model의 diversity를 높여 overfitting을 줄이고 성능 향상에 기여한다고 설명한다.

입력 3D 영상 $X \in \mathbb{R}^{H \times W \times D}$에 대해 encoder가 공통 feature $F_e$를 만들고, 각 decoder가 이를 받아 각각 $F_A$, $F_B$를 생성한다. 논문에서는 이를 다음처럼 적는다.

$$
F_e = f_{\Theta_e}(X), \quad F_A = f_{\Theta_{dA}}(F_e), \quad F_B = f_{\Theta_{dB}}(F_e)
$$

이후 각 feature는 Sigmoid activation을 거쳐 확률 출력 $P_A$, $P_B$가 된다. 저자들은 이 두 출력의 차이가 epistemic uncertainty를 근사한다고 본다.

논문은 먼저 기존 uncertainty estimation의 예로 Monte Carlo dropout을 설명한다. 하나의 입력에 대해 dropout이 적용된 여러 sub-model $\theta_n$을 샘플링하여 $N$번 forward pass를 수행하고, 각 클래스별 평균 예측 $\mu_c$를 구한 뒤 voxel-wise entropy를 uncertainty로 계산한다.

$$
\mu_c = \frac{1}{N}\sum_n P_n^c, \quad
u = - \sum_C \mu_c \log \mu_c
$$

여기서 $u$는 각 voxel 위치의 entropy이며, 값이 클수록 예측이 불확실하다는 뜻이다. 하지만 이런 방식은 iteration마다 여러 번 추론해야 하므로 계산 비용이 크다. MC-Net은 이를 두 decoder의 예측 불일치로 대체해 더 효율적으로 uncertainty를 활용한다.

다음 단계는 pseudo label 생성이다. 각 확률 출력은 sharpening function을 통해 soft pseudo label로 변환된다.

$$
sPL = \frac{P^{1/T}}{P^{1/T} + (1-P)^{1/T}}
$$

여기서 $T$는 temperature 상수다. 이 함수는 확률 분포를 더 날카롭게(sharpen) 만들어 low-entropy target을 제공한다. 저자들은 hard threshold로 만드는 pseudo label보다 soft pseudo label이 잘못된 라벨의 영향을 줄일 수 있다고 설명한다.

이제 mutual consistency 학습이 이루어진다. $sPL_A$는 decoder B의 출력 $P_B$를 감독하고, $sPL_B$는 decoder A의 출력 $P_A$를 감독한다. 즉, 두 decoder가 서로를 가르치는 구조다. 최종 손실은 supervised segmentation loss와 unsupervised consistency loss의 합이다.

$$
loss =
\underbrace{Dice(P_A, Y) + Dice(P_B, Y)}_{L_{seg}}
+ \lambda \times
\underbrace{\left(L_2(P_A, sPL_B) + L_2(P_B, sPL_A)\right)}_{L_c}
$$

여기서 $Y$는 ground truth이고, $L_{seg}$는 라벨 데이터에 대해서만 계산된다. 반면 $L_c$는 모든 훈련 데이터, 즉 라벨과 비라벨 데이터 모두에 적용된다. $L_2$는 Mean Squared Error(MSE) loss이다. $\lambda$는 두 항의 균형을 맞추는 가중치이며, 논문에서는 time-dependent Gaussian warming-up function으로 설정했다고 밝힌다.

이 설계의 의도는 분명하다. 두 decoder의 예측이 서로 다르면 그것이 곧 불확실성의 신호이고, cycled pseudo label을 사용해 서로를 일치시키도록 강제하면 점차 low-entropy, low-discrepancy 예측이 형성된다. 저자들은 이 과정이 특히 비라벨 데이터의 어려운 영역에서 일반화 가능한 특징을 학습하게 만든다고 주장한다.

훈련 절차를 정리하면 다음과 같다. 먼저 labeled patch와 unlabeled patch를 함께 미니배치에 넣는다. 각 입력은 공통 encoder와 두 decoder를 통과한다. labeled 데이터는 Dice loss로 직접 감독되고, 모든 데이터는 두 출력에서 생성된 soft pseudo label을 이용해 consistency loss로 추가 감독된다. 추론 시에는 $P_A$와 $P_B$의 평균을 최종 출력으로 사용한다.

## 4. 실험 및 결과

실험은 2018 Atrial Segmentation Challenge의 LA database에서 수행되었다. 데이터는 총 100개의 gadolinium-enhanced MR scan으로 구성되며, 80개는 training, 20개는 validation에 사용된다. 해상도는 isotropic $0.625 \times 0.625 \times 0.625$ mm이다. 비교의 공정성을 위해 기존 연구들과 동일하게 validation set 성능을 보고했다.

전처리로는 target 주변을 margin을 두고 crop한 뒤 zero mean, unit variance 정규화를 수행했다. 학습 시에는 크기 $112 \times 112 \times 80$의 3D patch를 랜덤 crop했고, 2D rotation과 flip을 적용했다. batch size는 4이며, 각 batch는 labeled patch 2개와 unlabeled patch 2개로 구성된다. sharpening temperature $T$는 0.1로 설정했고, optimizer는 SGD를 사용했다. 총 6,000 iteration 동안 학습했으며 초기 learning rate는 0.01이고 2,500 iteration마다 10%씩 감소시켰다. 테스트 시에는 sliding window 크기 $112 \times 112 \times 80$, stride $(18,18,4)$를 사용해 patch 예측을 합성했다.

평가 지표는 Dice, Jaccard, 95% Hausdorff Distance(95HD), average surface distance(ASD)이다. 반지도 설정은 두 가지다. 첫째는 10% labeled + 90% unlabeled, 둘째는 20% labeled + 80% unlabeled이다.

정량 결과에서 MC-Net은 모든 비교 방법을 앞섰다. 10% labeled 설정에서 MC-Net은 Dice 87.71%, Jaccard 78.31%, 95HD 9.36 voxel, ASD 2.18 voxel을 기록했다. 이는 UA-MT의 Dice 84.25%, SASSNet의 87.32%, DTC의 86.57%보다 높다. 특히 95HD와 ASD가 낮다는 것은 경계 품질과 surface accuracy 면에서 개선이 있음을 뜻한다.

20% labeled 설정에서도 MC-Net은 Dice 90.34%, Jaccard 82.48%, 95HD 6.00 voxel, ASD 1.77 voxel을 기록해 여섯 개의 기존 SOTA 방법을 모두 능가했다. fully supervised V-Net이 모든 라벨 데이터를 사용했을 때 Dice 91.14%였으므로, MC-Net은 단지 20% 라벨만으로도 upper bound에 매우 근접한 성능을 냈다. 저자들은 이를 비라벨 데이터를 효과적으로 활용한 결과로 해석한다.

정성 결과에서도 MC-Net은 더 완전한 left atrium 구조를 복원하고, 기존 방법들보다 challenging area에서 누락이 적으며 isolated region도 많이 제거했다고 보고한다. 논문은 morphology 기반 post-processing 없이 이런 결과를 얻었다고 명시한다.

Ablation study는 각 구성 요소의 기여를 보여준다. 먼저 두 decoder가 동일한 구조인 V2-Net보다, 서로 약간 다른 구조를 가진 V2d-Net이 더 좋은 성능을 보였다. 이는 decoder diversity가 실제로 도움이 된다는 증거로 제시된다. 다음으로 soft pseudo label만 추가한 설정(+sPL)도 대체로 성능이 좋아졌고, cycled pseudo label(+CPL)을 사용한 경우가 전반적으로 가장 좋았다. 예를 들어 10% labeled 설정에서 V2d-Net+CPL이 Dice 87.71%로 가장 높았고, 20% labeled에서도 V2d-Net+CPL이 Dice 90.34%를 기록했다.

Supplementary material에서는 두 decoder 각각의 출력 성능과 전체 MC-Net의 계산 복잡도도 제시된다. 흥미롭게도 decoder $\Theta_{dA}$ 또는 $\Theta_{dB}$ 단독 출력도 strong baseline보다 매우 높은 성능을 보인다. 다만 본문 핵심 주장은 두 decoder의 평균과 mutual consistency 학습이 최종적으로 가장 안정적인 결과를 제공한다는 점이다. 또 supplementary에는 Pancreas-CT 데이터셋 결과도 포함되어 있으나, 본문에서 중심적으로 다루는 주 실험은 LA database이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 반지도 학습에서 불확실한 영역을 단순한 노이즈가 아니라 중요한 학습 자원으로 재해석했다는 점이다. 많은 방법이 consistency 자체를 일반적으로 강제하는 데 그치는 반면, 이 논문은 어려운 영역에서의 prediction discrepancy를 활용해 왜 consistency가 필요한지 더 구체적인 동기를 제시한다.

둘째, 방법이 비교적 단순하면서도 효과적이다. backbone을 크게 바꾸지 않고 V-Net에 decoder 하나를 더 추가하고, cycled pseudo label을 통해 mutual consistency를 거는 방식이라 구현 복잡도가 과도하지 않다. 특히 Monte Carlo dropout처럼 여러 번의 stochastic forward pass 없이 uncertainty를 근사하려는 설계는 실용적이다.

셋째, 실험적으로도 설득력이 있다. LA benchmark에서 여러 최근 SOTA 방법을 일관되게 능가했고, 20% labeled 데이터만으로 fully supervised upper bound에 근접한 성능을 보였다. 또한 ablation study를 통해 decoder diversity, soft pseudo label, CPL 각각의 기여를 분리해서 보여준다.

반면 한계도 있다. 먼저 논문이 주장하는 “어려운 비라벨 영역이 더 중요하다”는 가설은 직관적이고 실험적으로 뒷받침되지만, 실제로 어떤 voxel이나 region이 얼마나 더 학습에 기여했는지를 정량적으로 직접 분석한 것은 아니다. 즉, uncertainty reduction과 generalized feature learning 사이의 인과를 강하게 증명했다기보다 성능 향상과 시각적 예시로 뒷받침한 수준이다.

또한 방법의 유효성은 주로 left atrium segmentation에 대해 본문에서 입증된다. supplementary에 Pancreas-CT 결과가 포함되어 있기는 하지만, 본문 수준의 상세 분석과 비교는 LA 중심이다. 따라서 다른 장기나 다른 imaging modality에서도 동일한 경향이 얼마나 강하게 재현되는지는 본문만으로는 제한적이다.

추가로, 두 decoder의 예측 차이를 epistemic uncertainty의 근사로 사용하는 설계는 실용적이지만, 이것이 Monte Carlo dropout이나 Bayesian uncertainty와 이론적으로 얼마나 정밀하게 대응하는지는 엄밀히 증명하지 않는다. 논문은 근사적 대안으로 제안하고 있으며, 이 부분은 실용적 설계 선택이지 이론적 보장이라고 보기는 어렵다.

마지막으로, 본문에서 계산 비용에 대한 절대적 절감 수치를 중심적으로 비교하지는 않는다. supplementary에 parameter와 MACs가 제시되어 있어 MC-Net이 V-Net보다 더 무거워졌다는 점은 확인할 수 있다. 즉, stochastic inference 반복은 없지만, decoder를 하나 더 둔 만큼 단일 모델 기준 복잡도는 증가한다.

## 6. 결론

이 논문은 3D MR 영상의 semi-supervised left atrium segmentation을 위해 MC-Net이라는 mutual consistency 기반 구조를 제안했다. 핵심 기여는 하나의 encoder와 두 개의 약간 다른 decoder를 사용해 prediction discrepancy를 uncertainty 신호로 활용하고, 이를 cycled pseudo label로 변환하여 두 decoder가 서로 일관되고 low-entropy한 예측을 내도록 학습시킨 점이다.

결과적으로 MC-Net은 적은 라벨 데이터 환경에서도 비라벨 데이터의 어려운 영역을 효과적으로 활용해 기존 반지도 방법들을 능가했고, LA benchmark에서 새로운 state-of-the-art 성능을 달성했다고 저자들은 주장한다. 실제 적용 측면에서는 의료 영상 annotation 비용을 줄이면서도 정밀한 분할을 유지할 수 있다는 점에서 의미가 크다. 향후 연구 측면에서는 uncertainty-aware semi-supervised segmentation, multi-decoder consistency learning, 그리고 shape prior나 구조 제약 모델과의 결합 가능성에서 확장 여지가 있다.
