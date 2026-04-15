# A Closer Look at Self-training for Zero-Label Semantic Segmentation

- **저자**: Giuseppe Pastore, Fabio Cermelli, Yongqin Xian, Massimiliano Mancini, Zeynep Akata, Barbara Caputo
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2104.11692

## 1. 논문 개요

이 논문은 **Generalized Zero-Label Semantic Segmentation (GZLSS)** 문제를 다룬다. 이 설정에서는 학습 시 직접 라벨이 주어진 클래스만이 아니라, 학습 중에는 라벨이 없었던 **unseen classes**까지 테스트 시점에 함께 분할해야 한다. 일반적인 zero-label semantic segmentation은 unseen class만 평가하는 경우가 많지만, 실제 환경에서는 seen과 unseen이 동시에 등장하므로 generalized setting이 더 현실적이다.

문제의 핵심은 학습 데이터 안에 unseen class에 해당하는 픽셀이 실제로 포함되어 있을 수 있지만, 이 픽셀들은 라벨이 없어 손실 계산에서 무시된다는 점이다. 기존 방법들은 visual-semantic embedding이나 feature generation으로 seen에서 unseen으로 지식을 옮기려 했지만, unseen에 대한 직접적인 학습 신호가 없어서 seen class 쪽으로 편향되기 쉽다. 특히 GZLSS에서는 이 편향이 더 심해져 unseen 성능이 크게 떨어진다.

저자들은 이 점을 정면으로 이용한다. 즉, 학습 이미지 안에 이미 존재하는 unseen 픽셀을 버리지 말고, 모델이 스스로 만든 **pseudo-label**로 다시 감독 신호로 사용하자는 것이다. 다만 pseudo-label은 노이즈가 많을 수 있으므로, 서로 다른 augmentation에서 일관되게 예측된 픽셀만 남기는 **consistency constraint**를 도입한다. 이를 반복적으로 수행해 pseudo-label 품질과 모델 성능을 함께 높이는 것이 논문의 목표다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 간단하다. 학습 데이터에 unseen class 픽셀이 숨어 있는데도 기존 방법들이 이를 활용하지 못했으므로, 모델이 스스로 그 픽셀에 라벨을 붙여 다시 학습에 사용하는 것이다. 저자들은 이를 단순한 self-training으로 끝내지 않고, **같은 이미지의 여러 augmentation에서 동일하게 예측된 pseudo-label만 채택**하는 방식으로 노이즈를 줄인다.

이 설계의 직관은 명확하다. 어떤 픽셀이 정말 unseen class에 속한다면, 이미지를 좌우 반전하거나 스케일을 바꾸더라도 원래 좌표계로 되돌렸을 때 같은 클래스가 예측될 가능성이 높다. 반대로 augmentation마다 라벨이 흔들리면 그 예측은 신뢰하기 어렵다. 따라서 여러 augmentation에서 합의된 픽셀만 pseudo-label로 사용하는 것이 더 안전하다.

기존 접근과의 차이도 분명하다. SPNet은 unlabeled unseen pixels를 아예 무시한다. ZS3와 CaGNet은 unseen class feature를 생성하는 generative 접근을 사용하지만, 저자들은 생성된 feature가 실제 분포를 본 적이 없어 domain shift 문제가 생길 수 있다고 지적한다. 반면 STRICT는 합성 feature를 만들지 않고, **실제 학습 이미지 내부의 unlabeled pixels 자체를 활용**한다. 또한 ZS5나 CaGNet+ST처럼 confidence 상위 $p\%$를 고르는 민감한 hyperparameter에 크게 의존하지 않는다는 점도 차별점이다.

## 3. 상세 방법 설명

기반 모델은 SPNet이다. SPNet은 입력 이미지 $x$를 semantic embedding space로 사상하는 CNN 모듈 $\phi$를 사용한다. 출력은 각 픽셀마다 $D$차원 임베딩이며, 이를 클래스 이름의 word embedding과 내적해 각 클래스 점수를 계산한다. seen class에 대한 픽셀별 posterior는 다음과 같다.

$$
P(\hat{y}_{nm}=c \mid x; W_s)=
\frac{\exp(w_c^T \phi(x)_{nm})}
{\sum_{c' \in S}\exp(w_{c'}^T \phi(x)_{nm})}
$$

여기서 $w_c$는 클래스 $c$의 word embedding이고, $\phi(x)_{nm}$은 픽셀 $(n,m)$의 visual embedding이다. 즉, 픽셀 특징과 클래스 의미 벡터의 호환성을 이용해 분류한다.

학습 데이터는 $T=\{(x,y)\}$이며, 각 픽셀 라벨 $y_{nm}$은 seen class 집합 $S$에 속하거나, unseen이지만 라벨이 없음을 뜻하는 $0$이다. 기본 학습 손실은 cross-entropy인데, unlabeled pixel은 무시된다.

$$
L_{CE}=
\sum_{n,m}
- \mathbf{1}[y_{nm}\neq 0]\log P(\hat{y}_{nm}=y_{nm}\mid x)
$$

즉, unseen 픽셀이 이미지 안에 있어도 $y_{nm}=0$이면 손실에 포함되지 않는다. 테스트 시에는 seen과 unseen의 word embedding을 모두 넣고, 전체 클래스 집합 $S \cup U$ 중 가장 높은 확률의 클래스를 픽셀별로 예측한다.

$$
\arg\max_{c \in S \cup U} P(\hat{y}_{nm}=c \mid x; [W_s, W_u])
$$

논문의 전체 파이프라인은 반복형 self-training이다. 먼저 seen 라벨만으로 SPNet을 학습한다. 그다음 이 모델 $P_{t-1}$을 이용해, 원래는 $y_{nm}=0$이었던 unlabeled pixels에 unseen class pseudo-label을 붙인다. 이후 원래의 seen 라벨 $y$와 새로 생성한 unseen pseudo-label $\bar{y}$를 함께 사용해 모델을 다시 fine-tuning한다. 이 과정을 여러 번 반복한다.

재학습 손실은 다음과 같다.

$$
L = L_{CE}(x,y) + \lambda L_{CE}(x,\bar{y})
$$

첫 번째 항은 원래 인간이 라벨링한 seen 픽셀에 대한 손실이고, 두 번째 항은 pseudo-labeled unseen 픽셀에 대한 손실이다. $\lambda$는 pseudo-label 손실의 가중치다. 논문은 이 구조를 통해 학습 데이터가 seen class에만 치우친 문제를 완화한다고 본다.

이 논문의 핵심 기술은 pseudo-label 생성 방식이다. 단일 예측으로 hard pseudo-label을 만들면 노이즈가 많으므로, 이미지 $x$에 대해 $K$개의 augmentation $A_k(x)$를 만든 뒤 각 augmented image에서 unseen class만 대상으로 hard label을 뽑는다.

$$
\bar{y}_{nm}^k=
\arg\max_{c \in U}
P(\hat{y}_{nm}=c \mid A_k(x); W_u)
$$

이렇게 얻은 각 augmentation별 pseudo-label mask를 원래 좌표계로 되돌린 뒤, 교집합을 취해 최종 pseudo-label을 만든다.

$$
\bar{y} = A_1^{-1}(\bar{y}^1) \cap \cdots \cap A_K^{-1}(\bar{y}^K)
$$

이 식의 의미는, 여러 augmentation에서 일관되게 같은 unseen 클래스로 예측된 픽셀만 남긴다는 것이다. 논문에서는 주로 horizontal mirroring과 scaling을 사용하며, 특히 upscaling이 효과적이었다고 보고한다.

## 4. 실험 및 결과

실험은 PascalVOC12와 COCO-stuff에서 수행된다. PascalVOC12는 20개 foreground object와 background를 포함하는 object segmentation benchmark이고, COCO-stuff는 object와 stuff class가 함께 있는 더 큰 장면 분할 데이터셋이다. 논문은 기존 연구와 같은 split과 validation 절차를 따른다고 명시한다. 성능 평가는 seen class mIoU, unseen class mIoU, 그리고 둘의 harmonic mean인 HM으로 측정한다.

비교 대상은 SPNet, ZS3, CaGNet, 그리고 self-training 변형인 ZS5, CaGNet+ST다. 또한 consistency 없이 hard pseudo-label만 쓰는 SPNet+ST도 별도 baseline으로 둔다. backbone은 ImageNet pretrained ResNet-101 기반 DeepLabV2이며, optimizer는 SGD, momentum은 0.9, weight decay는 $5 \cdot 10^{-4}$, 초기 learning rate는 $2.5 \cdot 10^{-4}$다.

주요 결과는 매우 분명하다. PascalVOC12에서 STRICT는 seen 82.7, unseen 35.6, HM 49.8을 기록했다. 이는 CaGNet+ST의 HM 43.7보다 높고, SPNet+ST의 HM 38.8보다도 크게 높다. COCO-stuff에서는 STRICT가 seen 35.3, unseen 30.3, HM 32.6을 기록했는데, 특히 unseen mIoU와 HM에서 기존 방법들을 큰 폭으로 앞선다. 논문은 이를 근거로, 복잡한 generative 전략보다도 실제 unlabeled unseen pixels를 잘 활용하는 self-training이 더 효과적일 수 있다고 해석한다.

흥미로운 점은 self-training 자체가 전반적으로 모든 방법에 도움이 되었다는 것이다. 예를 들어 PascalVOC12에서 SPNet은 HM 21.8인데 SPNet+ST는 38.8로 크게 상승한다. 즉, consistency constraint가 없더라도 pseudo-label 활용만으로도 이득이 있지만, STRICT는 여기서 한 단계 더 나아가 pseudo-label 품질을 안정적으로 개선한다.

저자들은 background를 seen class에 포함하는 더 어려운 설정도 평가한다. 이 경우 unseen class 픽셀이 background로 잘못 흡수되기 쉬워 전체 성능이 크게 떨어진다. PascalVOC12에서 STRICT는 seen 74.7, unseen 14.3, HM 24.0을 기록해 여전히 가장 좋았지만, background를 무시한 표준 설정보다 성능 저하가 크다. 논문은 이것이 background semantic shift 문제를 시사한다고 해석한다.

ablation도 설득력 있다. transformation 종류를 비교한 결과, mirroring만 쓰는 것보다 scaling이 대체로 낫고, 특히 upscaling이 가장 좋았다. mirroring과 upscaling을 함께 쓸 때 PascalVOC12에서 unseen 32.9, HM 47.0으로 가장 좋은 결과를 얻었다. 또한 iterative self-training 횟수를 늘릴수록 성능이 대체로 상승하다가 약 6회 이후 포화되거나 약간 감소하는데, 이는 consistency constraint가 노이즈를 줄이기는 하지만 완전히 제거하지는 못하기 때문이라고 설명한다.

정성적 결과에서도 논문의 주장을 뒷받침한다. STRICT는 SPNet+ST보다 잘못된 unseen pseudo-label을 더 많이 제거하고, 공간적으로 더 일관된 mask를 만든다. 또한 ZS5와 비교했을 때 seen class 편향이 줄어, unseen object를 seen class로 오분류하는 현상이 적다고 보고한다. 다만 작은 영역만 차지하는 클래스, 예를 들어 plant처럼 co-occurring pixel 수가 적은 경우에는 여전히 어려움이 남는다고 언급한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정에 매우 직접적으로 대응한다는 점이다. unseen class 픽셀이 학습 이미지 안에 실제로 존재한다는 사실을 적극적으로 활용했고, 이를 위해 복잡한 생성 모델 대신 간단한 iterative self-training과 consistency filtering만 사용했다. 설계가 단순한데도 기존 state-of-the-art를 이겼다는 점은 실용적으로도 중요하다.

또 다른 강점은 pseudo-label 선택 기준이 비교적 견고하다는 것이다. 기존 self-training 계열은 confidence threshold나 상위 $p\%$ 선택 같은 민감한 hyperparameter에 크게 좌우될 수 있는데, STRICT는 augmentation 간 일관성이라는 더 구조적인 기준을 사용한다. 또한 background 포함 여부, transformation 종류, iteration 횟수 등 다양한 조건에서 분석을 수행해 방법의 동작 특성을 비교적 명확히 보여준다.

한계도 분명하다. 첫째, pseudo-label이 근본적으로 모델 예측에서 나오기 때문에 초기 모델이 심하게 편향되면 잘못된 pseudo-label이 반복적으로 강화될 위험이 있다. 논문도 iteration이 지나치게 늘어나면 성능이 포화되거나 약간 감소한다고 보고한다. 둘째, background를 포함한 설정에서는 성능이 크게 떨어진다. 이는 object segmentation에서 unseen과 background를 구분하는 문제가 여전히 어렵다는 뜻이다. 셋째, 적게 등장하는 unseen class는 충분한 pseudo-label을 얻기 어려워 성능이 낮을 수 있다. 논문은 이 문제를 qualitative analysis에서 직접 언급하지만, 이를 해결하는 구체적 메커니즘은 제안하지 않는다.

비판적으로 보면, 논문의 핵심은 매우 설득력 있지만, pseudo-label의 품질을 정량적으로 얼마나 개선했는지에 대한 더 직접적인 분석이 있었으면 더 강했을 것이다. 예를 들어 consistency 적용 전후 pseudo-label precision이나 class별 pseudo-label coverage 같은 통계가 제시되면, 왜 성능이 좋아졌는지 더 명료하게 설명할 수 있었을 것이다. 또한 $\lambda$의 값 자체는 본문 발췌 기준으로 명시적으로 보이지 않는다. 따라서 손실 가중치의 민감도는 이 텍스트만으로는 평가할 수 없다.

## 6. 결론

이 논문은 GZLSS에서 기존 방법들이 놓치고 있던 **학습 이미지 내부의 unlabeled unseen pixels**를 활용해 성능을 끌어올리는 간단하고 효과적인 방법을 제시한다. 핵심 기여는 augmentation 간 예측 일관성을 이용해 noisy pseudo-label을 걸러내는 **STRICT self-training framework**와, 이를 반복적으로 적용해 pseudo-label generator와 segmentation model을 함께 개선한 점이다.

실험적으로 STRICT는 PascalVOC12와 COCO-stuff에서 기존 방법들을 능가했고, 특히 unseen class 성능과 harmonic mean에서 큰 개선을 보였다. 이는 실제 적용 측면에서도 의미가 있다. 추가 pixel annotation 없이 이미 존재하는 학습 이미지로부터 unseen supervision을 끌어낼 수 있기 때문이다. 향후 연구에서는 background modeling, class imbalance, 드물게 등장하는 unseen class에 대한 pseudo-label regularization 같은 요소를 결합하면 더 강한 generalized zero-label segmentation 시스템으로 발전할 가능성이 크다.
