# Concurrent Segmentation and Localization for Tracking of Surgical Instruments

* **저자**: Iro Laina, Nicola Rieke, Christian Rupprecht, Josué Page Vizcaíno, Abouzar Eslami, Federico Tombari, Nassir Navab
* **발표연도**: 2017
* **arXiv**: <https://arxiv.org/abs/1703.10701v2>

## 1. 논문 개요

이 논문은 수술 장면에서 **수술 도구(surgical instrument)를 실시간에 가깝게 추적**하기 위한 방법을 제안한다. 더 구체적으로는, 영상에서 도구의 각 부분이 어디에 있는지 찾는 **landmark localization**과, 도구의 픽셀 영역을 구분하는 **semantic segmentation**을 하나의 네트워크 안에서 동시에 수행하는 문제를 다룬다. 논문의 핵심 대상 응용은 **Minimally Invasive Surgery (MIS)**와 **Retinal Microsurgery (RM)**이며, 특히 RM에서는 도구 끝점의 위치를 정확히 알아야 망막과의 거리나 접근 상태를 정밀하게 추정할 수 있다.

논문이 겨냥하는 연구 문제는 기존의 marker-free vision 기반 도구 추적이 실제 수술 환경에서 매우 어렵다는 점이다. 저자들은 기존 방법들이 강한 조명 변화, **specular reflections**, motion blur, 그림자, 그리고 도구의 복잡한 형상 변화에 취약하다고 지적한다. 기존에는 Haar wavelets, gradient, color 같은 handcrafted feature를 쓰거나, segmentation을 먼저 하고 그 결과로 localization을 보정하는 식의 2단계 파이프라인이 주로 사용되었다. 그러나 이런 방식은 일반화 성능이 제한되거나, 초기화가 필요하거나, landmark localization과 segmentation의 상호작용을 충분히 활용하지 못한다.

이 문제의 중요성은 분명하다. 도구 segmentation은 수술 화면 위에 유용한 정보를 겹쳐 표시할 때 의사의 시야를 가리지 않도록 도와줄 수 있고, 도구 움직임은 수술 workflow 분석에 활용될 수 있다. 또한 RM에서는 도구 tip 위치가 망막과 얼마나 가까운지를 추정하는 핵심 단서가 된다. 따라서 robust한 도구 추적은 단순한 computer vision 문제가 아니라, 실제 computer-assisted intervention의 기반 기술이다.

이 논문의 직접적인 목표는 **도구 landmark의 2D pose estimation을 heatmap regression으로 재정의**하고, 이를 segmentation과 함께 하나의 CNN으로 end-to-end 학습해, 더 정확하고 더 실용적인 추적 시스템을 만드는 것이다.

![그림 1: 제안하는 방법(CSL)의 전체 구조를 나타낸다. CNN 기반 접근을 통해 의미론적 분할과 랜드마크 위치 추정을 동시에 수행한다. 위치 추정을 히트맵 회귀 문제로 정의함으로써 두 작업이 가중치를 공유하는 end-to-end 학습이 가능하며, 랜드마크의 2D 좌표를 직접 회귀하는 방식보다 더 높은 정확도를 달성한다.
](https://ar5iv.labs.arxiv.org/html/1703.10701/assets/images/teaser.png)

## 2. 핵심 아이디어

논문의 중심 아이디어는 **segmentation과 localization이 단순히 연속된 두 단계가 아니라, 서로 강하게 의존하는(interdependent) 문제**라는 관찰에 있다. 도구의 정확한 landmark를 찾으려면 우선 도구가 어디에 있는지 알아야 하고, 반대로 도구의 tip이나 joint 위치를 알면 어떤 픽셀이 도구에 속하는지도 더 잘 판단할 수 있다. 저자들은 이 관계를 단순한 전처리-후처리 관계가 아니라, **공동으로 학습해야 할 하나의 구조적 문제**로 본다.

이를 위해 저자들은 landmark localization을 좌표값 자체인 $(x,y)$를 직접 예측하는 방식에서 벗어나, **각 landmark마다 하나의 heatmap을 예측하는 문제**로 바꾼다. 이 heatmap의 각 픽셀은 “이 위치가 실제 landmark에 얼마나 가까운가”를 나타내는 confidence를 의미한다. 이렇게 바꾸면 localization의 출력도 segmentation처럼 **공간적인 dense map**이 되므로, 두 작업을 거의 같은 형태의 출력으로 다룰 수 있다.

이 설계의 장점은 세 가지다. 첫째, direct coordinate regression은 정답이 오직 하나의 좌표라고 가정하는데, 실제 수동 annotation은 몇 픽셀 정도 흔들릴 수 있다. heatmap 방식은 정답 근처를 넓게 높은 값으로 주기 때문에 이런 annotation ambiguity를 더 자연스럽게 처리한다. 둘째, 좌표 벡터를 바로 회귀하는 방식은 중간에 spatial structure를 많이 잃기 쉽지만, heatmap은 위치 정보를 끝까지 보존한다. 셋째, segmentation과 localization이 같은 해상도의 출력이 되므로, 두 작업이 encoder뿐 아니라 decoder 단계까지도 더 깊게 feature를 공유할 수 있다.

기존 접근과의 차별점은 분명하다. 기존의 2단계 접근은 segmentation을 먼저 구한 뒤 connected component나 post-processing으로 위치를 추정하거나, localization 결과를 segmentation으로 보정했다. 반면 이 논문은 두 작업을 하나의 네트워크 안에서 동시에 학습하고, 최종적으로는 segmentation logits를 localization branch에 직접 연결해 **segmentation 정보가 landmark heatmap을 유도하도록 설계**한다. 즉, 단순히 “같이 학습한다” 수준이 아니라, 두 작업을 구조적으로 강하게 결합한다는 점이 차별점이다.

## 3. 상세 방법 설명

![그림 2: 모델링 전략. 제안된 CSL 아키텍처와 두 가지 기준 모델(SL, L).](https://ar5iv.labs.arxiv.org/html/1703.10701/assets/images/models.png)

### 문제 설정

논문은 하나의 학습 샘플을 $(X, S, y)$로 둔다. 여기서 $X \in \mathbb{R}^{w \times h \times 3}$는 입력 RGB 이미지, $y \in \mathbb{R}^{(n \times 2)}$는 $n$개의 landmark에 대한 2D 좌표, $S \in \mathbb{R}^{\frac{w}{2} \times \frac{h}{2} \times c}$는 $c$개의 semantic class에 대한 segmentation label map으로 **입력 해상도의 절반 크기**로 예측되도록 설계되어 있다.

논문은 세 가지 모델을 비교한다. 첫 번째는 localization만 하는 **L**, 두 번째는 segmentation과 localization을 함께 하지만 좌표는 직접 회귀하는 **SL**, 세 번째이자 최종 제안 모델은 heatmap 기반의 **CSL (Concurrent Segmentation and Localization)**이다.

### Encoder

세 모델 모두 공통 encoder로 **ResNet-50**을 사용한다. 입력 이미지는 $480 \times 480$으로 처리되며, 이때 ResNet 마지막 convolution layer의 feature map은 $15 \times 15$ 해상도가 된다. 원래 ResNet의 마지막 pooling layer와 loss layer는 제거한다.

저자들이 더 깊은 ResNet 대신 50-layer 버전을 쓴 이유는, 이 문제가 실제 수술 장면을 겨냥하기 때문에 **계산 시간도 중요**하기 때문이다. 즉, 성능뿐 아니라 near real-time 동작을 고려한 절충이 들어가 있다.

### 모델 1: Localization (L)

가장 단순한 baseline인 L은 landmark의 실제 2D 좌표를 직접 회귀한다. 출력은 $2n$차원의 벡터이며, 각 landmark에 대해 $x, y$ 좌표를 낸다. 이를 위해 encoder 뒤에 stride를 가진 residual block을 하나 더 붙여 feature map을 $8 \times 8 \times 2048$로 줄인 뒤, $8 \times 8$ average pooling과 fully connected layer를 사용해 최종 좌표를 예측한다.

학습 손실은 표준적인 $L_2$ 회귀 손실이다.

$$
l_L(\tilde y, y)=|\tilde y-y|_2^2
$$

이 방식은 구현은 간단하지만, 논문이 보기에는 중요한 한계가 있다. 좌표 하나만 정답으로 두면 annotation noise를 반영하지 못하고, average pooling과 fully connected layer를 거치면서 **공간적 정밀도(spatial precision)**가 약해진다.

### 모델 2: Segmentation and Localization (SL)

SL은 하나의 네트워크 안에서 **좌표 회귀와 segmentation을 동시에 학습**한다. encoder는 공유하고, 이후에 localization branch와 segmentation branch로 갈라진다. localization branch는 L과 동일하게 좌표를 직접 회귀하고, segmentation branch는 residual up-sampling layer를 사용해 각 픽셀의 class score를 예측한다.

실시간 제약 때문에 segmentation 출력은 입력보다 절반 해상도로 만들고, 최종적으로 bilinear upsampling을 사용한다. 이 모델의 의미는, 비록 localization과 segmentation의 출력 형태가 다르더라도, 적어도 encoder 수준에서는 feature를 공유하게 해서 두 작업이 서로 도움을 주게 하자는 데 있다.

전체 손실은 다음처럼 두 작업 손실의 합이다.

$$
l_{SL}(\tilde y, y, \tilde S, S)=\lambda_L , l_L(\tilde y, y)+l_S(\tilde S, S)
$$

segmentation 손실은 픽셀 단위 softmax-log loss이다.

$$
l_S(\tilde S, S)=
-\frac{1}{wh}
\sum_{u=1}^{w}
\sum_{v=1}^{h}
\sum_{j=1}^{c}
S(u,v,j)\log
\left(
\frac{e^{\tilde S(u,v,j)}}
{\sum_{k=1}^{c} e^{\tilde S(u,v,k)}}
\right)
$$

쉬운 말로 설명하면, 각 픽셀에서 정답 class의 확률이 높아지도록 학습하는 전형적인 segmentation loss이다.

### 모델 3: Concurrent Segmentation and Localization (CSL)

논문의 최종 모델인 CSL은 landmark를 좌표가 아니라 **heatmap**으로 예측한다. 각 landmark $i$에 대해 정답 좌표 $y_i$ 주변에 Gaussian을 놓아 정답 heatmap을 만든다. 이 heatmap은 “정확히 이 점만 정답”이 아니라, **정답 근처일수록 높은 점수**를 주는 target이다.

정답 heatmap의 개념은 다음과 같이 이해할 수 있다.

$$
G_i(u,v)=
\frac{1}{2\pi \sigma^2}
\exp\left(
-\frac{|y_i-(u,v)^\top|_2^2}{2\sigma^2}
\right)
$$

여기서 $\sigma$는 Gaussian의 퍼짐 정도를 정한다. $\sigma$가 크면 landmark 주변 넓은 영역에 높은 점수를 주고, 작으면 정답 주변만 날카롭게 강조한다.

CSL의 중요한 구조적 특징은 다음과 같다.

첫째, **heatmap과 segmentation map의 해상도가 같기 때문에** decoder 단계에서도 두 작업이 많은 표현을 공유할 수 있다. 이것이 SL과 가장 큰 차이다. SL은 localization이 좌표 벡터라서 decoder를 본질적으로 분리해야 하지만, CSL은 localization도 dense map이므로 더 오래 함께 갈 수 있다.

둘째, 저자들은 **long-range skip connection**을 추가한다. 이는 encoder의 낮은 단계 feature를 decoder에 더해 주는 구조로, 초기 layer의 고해상도 정보를 후반부까지 전달해 localization과 segmentation 정밀도를 높인다.

셋째, 두 작업을 마지막에만 분리하고, **softmax 이전의 segmentation score(logits)**를 최종 feature map에 concatenation하여 landmark heatmap 예측을 돕는다. 이는 “도구가 어디에 있는지”에 대한 segmentation branch의 지식을 localization branch가 직접 쓰게 만드는 장치다.

추출된 수식 (2)는 일부 표기가 깨져 있어 정규화 계수나 landmark 평균 처리 방식이 완전히 선명하지 않다. 하지만 논문의 의미는 분명하다. 즉, segmentation loss에 더해, 각 landmark heatmap에 대해 **정답 Gaussian heatmap과 예측 heatmap 사이의 pixel-wise squared error**를 최소화하는 것이다. 의미를 보존하면 다음과 같이 쓸 수 있다.

$$
l_{CSL}=
l_S(\tilde S,S)
+
\lambda_H
\sum_{i=1}^{n}
\sum_{u,v}
\left(
G_i(u,v)-\tilde H_i(u,v)
\right)^2
$$

여기서 $\tilde H_i$는 예측된 $i$번째 landmark heatmap이다. 이 손실의 직관은 간단하다. 모델이 정답 좌표 근처에서 높은 응답을 만들수록 손실이 작아진다. direct coordinate regression보다 훨씬 덜 딱딱하고, 정답 부근의 오차를 더 자연스럽게 처리한다.

추론 시에는 각 predicted heatmap에서 **최대 confidence를 가지는 위치**를 landmark 좌표로 사용한다. 논문은 또한 heatmap이 뾰족하지 않고 넓게 퍼져 있으면 **misdetection의 신호**가 될 수 있다고 말한다. 즉, 출력 heatmap의 분산이 불확실성의 힌트로 쓰일 수 있다는 뜻이다.

### 학습 절차와 구현 세부사항

encoder는 ImageNet으로 pretrained된 ResNet-50으로 초기화하고, 새로 추가한 layer는 평균 0, 분산 0.01인 정규분포로 초기화한다. 모든 이미지는 먼저 $640 \times 480$으로 resize한 뒤, 학습 때는 회전 $[-5^\circ,5^\circ]$, scale $[1,1.2]$, random crop $480 \times 480$, gamma correction $\gamma \in [0.9,1.1]$, color factor, 그리고 **specular reflections augmentation**을 적용한다. 이 augmentation 설계는 논문이 강조한 실제 수술 영상의 방해 요소를 직접 겨냥한다는 점에서 의미가 있다.

Localization heatmap의 Gaussian 폭은 RM에서 $\sigma=5$, EndoVis에서 $\sigma=7$로 둔다. 저자들은 EndoVis의 도구가 더 커서 더 넓은 spread를 쓴다고 설명한다. 모든 CNN은 SGD로 학습하며, learning rate는 $10^{-7}$, momentum은 $0.9$, 그리고 $\lambda_L, \lambda_H$는 모두 경험적으로 1로 설정했다. 추론 시간은 NVIDIA GeForce GTX TITAN X에서 프레임당 56ms로 보고되며, 이는 대략 초당 18프레임 수준으로 **near real-time**에 해당한다.

## 4. 실험 및 결과

### 데이터셋과 평가 설정

논문은 두 개의 대표적인 benchmark에서 성능을 검증한다.

첫 번째는 **Retinal Microsurgery dataset**이다. 이 데이터셋은 18개의 in-vivo sequence로 구성되며, 각 sequence는 200 프레임, 해상도는 1,920 &times; 1,080이다. 데이터는 네 가지 instrument-dependent subset으로 나뉜다. annotation된 landmark는 $n$=3개이며, 이는 좌우 tip과 center joint에 해당한다. semantic class는 도구와 배경의 $c$=2개다.

두 번째는 **MICCAI EndoVis Challenge 2015** 데이터셋이다. 학습 데이터는 네 개의 ex-vivo 45초 sequence이고, 테스트는 같은 sequence의 나머지 15초 구간과 두 개의 새로운 60초 비디오로 구성된다. 가이드라인상, 추가 15초 sequence를 테스트할 때는 해당 surgery를 학습에서 제외해야 하며, 긴 테스트 sequence 중 하나는 **이전에 보지 못한 도구 유형**을 포함한다. 해상도는 720 &times; 576이고, 한 장면에 하나 또는 두 개의 수술 도구가 등장한다. 기본 설정에서는 도구당 $n$=1개의 joint를 추적하고, semantic class는 manipulator, shaft, background의 $c$=3개다.

### 모델링 전략 비교: L, SL, CSL

논문은 먼저 RM 데이터셋에서 9개 sequence로 학습하고 나머지 9개로 테스트하여, 모델링 방식 자체의 차이를 비교한다. 결과는 명확하다. **좌표를 직접 회귀하는 L이 가장 낮은 localization 성능**을 보이고, segmentation을 함께 학습하는 SL은 이보다 좋아진다. 그리고 제안한 **CSL이 가장 높은 정확도**를 기록한다.

Figure 3에 대한 서술에 따르면, acceptance pixel threshold를 20픽셀로 둘 때 CSL은 **좌우 tool tip에서 90% 이상**, **center joint에서 79%**의 정확도를 보인다. 여기서 Figure 3의 “Threshold Score”의 정확한 수식 정의는 제공된 발췌문에는 없지만, 문맥상 허용 픽셀 오차 threshold 안에 landmark가 들어왔는지를 기준으로 한 정확도 곡선으로 이해할 수 있다.

이 비교는 논문의 핵심 주장을 잘 뒷받침한다. 즉, segmentation을 함께 학습하는 것 자체가 localization에 도움이 되며, 더 나아가 localization을 heatmap regression으로 바꾸면 그 도움이 더 크게 나타난다는 것이다. 저자들은 이는 **contextual information을 더 잘 활용하기 때문**이라고 해석한다.

Segmentation 측면에서도 CSL이 가장 좋다. DICE score는 **75.4%**이며, skip connection이 없는 CSL은 **74.4%**, SL은 **73.7%**, U-Net은 **72.5%**다. 이 결과는 heatmap formulation뿐 아니라, long-range skip connection과 강한 feature sharing이 segmentation에도 이득을 준다는 점을 보여 준다.

### Retinal Microsurgery 결과

RM에서는 두 종류의 평가가 수행된다.

첫 번째는 **Half Split Experiment**로, 18개 sequence 각각의 첫 절반으로 학습하고 나머지 절반으로 평가한다. 논문은 이 설정에서 CSL이 기존 state-of-the-art인 **FPBC [23]**, **POSE [3]**, **Online Adaption [9]**보다 뚜렷하게 좋은 결과를 내며, $ \alpha = 0.15 $일 때 **KBB score 평균 84% 이상**을 달성했다고 보고한다. 다만 KBB score의 정확한 정의식은 제공된 발췌문에는 포함되어 있지 않다.

두 번째는 **Cross Validation Experiment**로, 네 개의 instrument-dependent subset 중 세 개로 학습하고 나머지 하나로 테스트하는 leave-one-out 방식이다. 이 실험의 목적은 단순히 새로운 sequence에 대한 일반화만이 아니라, **보지 못한 도구 geometry**에 대한 일반화 능력을 보는 것이다. 논문은 이 설정에서도 state-of-the-art 성능을 달성했다고 말한다. 다만 발췌문에는 이 실험의 정확한 수치가 제시되어 있지 않다.

이 결과의 의미는 크다. RM은 고해상도이면서 아주 섬세한 조작을 요구하는 환경이므로, landmark localization의 작은 오차도 중요하다. CSL이 여기서 direct coordinate regression이나 기존 handcrafted/online adaptation 기반 방법을 이긴다는 것은, 제안된 heatmap 기반 joint learning이 단순한 구조적 미학이 아니라 실제 정밀도 개선으로 이어진다는 뜻이다.

### EndoVis Challenge 결과

EndoVis에서는 가이드라인대로 **leave-one-surgery-out** 방식으로 평가한다. 모든 실험에서 네트워크는 multi-class segmentation 목표로 학습되며, binary 결과를 계산할 때만 Shaft와 Grasper를 하나의 instrument class로 합친다.

Table 1의 평균 성능은 다음과 같다. Binary segmentation 기준으로 **Balanced Accuracy 92.6%, Recall 86.2%, Specificity 99.0%, DICE 88.9%**를 기록한다. 세부 class 기준으로는 Shaft의 DICE가 **87.7%**, Grasper의 DICE가 **77.7%**다. 즉, 상대적으로 길고 뚜렷한 shaft보다, 더 작고 복잡한 grasper 부위가 어려운 문제라는 점이 수치로 드러난다.

Localization error는 표에서 평균 **24.8/51.6 pixels**로 보고되어 있다. 다만 일부 sequence와 평균 행에서 error가 슬래시로 두 값으로 제시되는데, 발췌문만으로는 이 두 값의 정확한 구분 기준이 완전히 설명되지 않는다. 단, 개별 sequence 결과를 보면 sequence 2와 3은 각각 **9.7 pixels**, **10.9 pixels**로 매우 낮은 오차를 보이는 반면, sequence 5와 6은 **38.4/60.0**, **36.4/63.9 pixels**로 훨씬 어렵다. 논문은 그 이유로 **이전에 보지 못한 instrument와 viewpoint**를 들고, challenge 관리자들이 sequence 5와 6의 tracking ground truth 자체도 다른 sequence보다 덜 정확할 수 있다고 언급했다고 전한다.

기존 방법과 비교하면, CSL은 FCN [14], FCN+OF [14], DRL [15]보다 전반적으로 더 강한 성능을 보인다. 표에서 prior method들은 세부 class별 DICE나 localization error를 모두 보고하지는 않지만, 적어도 보고된 binary-level 수치만 놓고 봐도 CSL은 매우 경쟁력 있거나 그 이상이다. 특히 CSL은 단순 segmentation만이 아니라 **joint localization까지 동시에 수행**한다는 점에서, 결과의 실질적 가치가 더 크다.

![그림 5(좌): EndoVis의 정성적 결과. 본 방법에서는 히트맵의 개수를 조절함으로써 의미론적 분할 클래스 수와 추적할 관절 수를 쉽게 변경할 수 있다. 하나의 기구만 존재하는 경우로 학습 데이터셋에 포함되지 않았음에도 불구하고 잘 처리된 것을 확인할 수 있다.](https://ar5iv.labs.arxiv.org/html/1703.10701/assets/images/endovis3.png)

![그림 5(우): EndoVis의 정성적 결과. 두 개의 기구가 존재하며 서로 다른 도구로 구분되는 경우를 나타낸다.](https://ar5iv.labs.arxiv.org/html/1703.10701/assets/images/endovis2.png)

정성적 결과도 흥미롭다. 논문은 class 수를 $c$=5로, joint 수를 $n$=2로 바꾸면 **left shaft, left grasper, right shaft, right grasper, background**를 구분하면서, 두 개의 도구를 따로 인식하고 각 도구의 joint까지 추적할 수 있음을 보인다. Figure 5의 설명에 따르면, 학습 때 보지 못한 왼쪽 도구까지 구분해 낸다. 이는 이 방법이 단순히 “하나의 도구를 segmentation하는 네트워크”가 아니라, **출력 heatmap 수와 semantic class 수만 바꾸면 더 복잡한 다중 도구 상황으로 확장될 수 있는 유연한 프레임워크**라는 점을 보여 준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의 자체를 바꾼 데 있다. 저자들은 localization을 좌표 회귀로 두지 않고 heatmap regression으로 바꾸면서, segmentation과 localization을 동일한 공간 표현으로 정렬했다. 이로 인해 두 작업이 단순히 encoder만 공유하는 것이 아니라, decoder와 최종 예측 단계까지 더 깊게 결합될 수 있게 되었다. 이 설계는 논문이 보고한 성능 향상과 잘 맞아떨어진다.

또 하나의 강점은 **annotation ambiguity에 대한 처리**다. 실제 landmark annotation은 몇 픽셀 단위로 흔들릴 수 있는데, direct coordinate regression은 이런 상황에서 지나치게 경직된 target을 사용한다. 반면 Gaussian heatmap은 정답 주변을 연속적인 confidence 분포로 표현하므로, 학습이 더 안정적이고 실제 영상의 모호성을 더 잘 반영한다. 이는 의료 영상처럼 정답의 경계나 점 위치가 완전히 절대적이지 않은 문제에서 특히 적절한 설계다.

실용성도 강점이다. 제안 방식은 **초기화가 필요 없고**, **post-processing이나 temporal regularization이 없어도** 동작하며, 프레임당 56ms로 near real-time 속도를 달성한다. 또한 joint 수와 semantic class 수를 바꾸면 단일 도구뿐 아니라 다중 도구 상황에도 대응할 수 있어 구조적으로 유연하다.

반면 한계도 분명하다. 첫째, 결과를 보면 **center joint**와 **grasper**처럼 더 작거나 더 복잡한 부분은 여전히 어렵다. 예를 들어 RM에서 tip은 20픽셀 threshold 기준 90% 이상인데 center joint는 79%이고, EndoVis에서도 Shaft DICE 87.7%에 비해 Grasper DICE는 77.7%로 낮다. 즉, 제안 방법이 강력하더라도 fine articulated part에 대해서는 문제 난도가 여전히 높다.

둘째, 보지 못한 도구나 viewpoint에 대해서는 성능 저하가 발생한다. sequence 5와 6의 localization error가 크게 증가한 점은, 논문이 generalization 능력을 보여 주긴 했지만 **domain shift나 unseen tool geometry가 여전히 어려운 문제**임을 시사한다. 논문은 이를 ground truth 품질 문제와 함께 설명하지만, 사용자 입장에서는 실제 새로운 수술 장비나 새로운 촬영 조건으로 넘어갈 때 안정성이 얼마나 유지되는지가 여전히 중요한 질문으로 남는다.

셋째, 이 논문이 다루는 것은 **2D pose estimation**이다. 즉, landmark의 영상 평면 내 위치를 추정하는 것이지, 3D pose 전체를 복원하는 것은 아니다. 수술 로봇이나 고정밀 navigation으로 확장하려면 이후 연구에서 depth 또는 3D geometry와의 결합이 필요할 가능성이 크다. 또한 논문은 temporal tracker의 초기화 문제를 피하는 장점을 갖지만, 반대로 **시간 정보 자체를 적극 활용하는 모델**과의 비교는 중심적으로 다루지 않는다. 따라서 occlusion이 길게 이어지거나 매우 빠른 motion이 있는 경우 temporal modeling이 추가되면 더 좋아질 가능성은 열려 있다.

넷째, 학습 데이터 요구량 측면에서도 완전히 가볍다고 보기는 어렵다. 논문은 limited data에서도 성공했다고 주장하고 실제로 ImageNet pretraining과 augmentation으로 이를 보완하지만, 여전히 학습에는 **landmark annotation과 semantic segmentation annotation**이 모두 필요하다. 의료 데이터에서는 이런 라벨링 비용이 적지 않다.

마지막으로, 제공된 발췌문 기준으로는 일부 세부 정보가 완전하지 않다. 예를 들어 수식 (2)의 표기는 일부 깨져 있고, KBB score나 Threshold Score의 정확한 정의도 발췌문에는 없다. 또한 RM cross-validation의 세부 수치도 서술만 있고 표나 숫자가 주어지지 않는다. 따라서 이 보고서에서 해당 부분은 논문이 실제로 명시한 범위 안에서만 해석해야 한다.

## 6. 결론

이 논문은 수술 도구 추적 문제에서 **landmark localization을 heatmap regression으로 재정의**하고, 이를 **semantic segmentation과 동시에 예측하는 end-to-end CNN 구조**를 제안했다. 핵심 기여는 단순히 두 작업을 묶었다는 데 있지 않고, heatmap과 segmentation map의 **공간적 동형성**을 이용해 두 작업을 강하게 결합한 데 있다. 여기에 long-range skip connection과 segmentation logits를 localization에 연결하는 설계를 더해, direct coordinate regression보다 더 정확하고 더 robust한 추적을 달성했다.

실험적으로도 제안법은 RM benchmark와 EndoVis Challenge에서 강한 성능을 보였고, 특히 기존 state-of-the-art를 넘어서는 결과를 제시했다. 또한 모델이 joint 수와 class 수 변화에 유연하다는 점, unseen tool이나 multi-instrument setting에도 어느 정도 대응한다는 점은 실제 확장 가능성을 높인다.

실제 적용 측면에서 이 연구는 surgical navigation, augmented overlay, workflow analysis, 그리고 정밀한 도구 모니터링 시스템의 핵심 구성 요소가 될 잠재력이 있다. 향후 연구에서는 이 프레임워크를 3D pose estimation, temporal modeling, uncertainty estimation, 더 다양한 수술 도메인으로 확장하는 방향이 자연스러워 보인다. 제공된 텍스트만 기준으로 보더라도, 이 논문은 **“수술 도구 추적을 segmentation과 pose estimation의 공동 dense prediction 문제로 바꿔 놓았다”**는 점에서 의미 있는 전환점을 만든 작업이라고 평가할 수 있다.
