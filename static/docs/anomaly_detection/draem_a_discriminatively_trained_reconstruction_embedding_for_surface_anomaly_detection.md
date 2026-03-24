# DRAEM - A discriminatively trained reconstruction embedding for surface anomaly detection

- **저자**: Vitjan Zavrtanik, Matej Kristan, Danijel Skočaj
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2108.07610

## 1. 논문 개요

이 논문은 표면 이상 탐지(surface anomaly detection)를 위한 새로운 방법인 DRÆM을 제안한다. 문제 설정은 정상 이미지들만 학습에 사용할 수 있고, 테스트 시에는 이미지의 일부 픽셀만 이상을 포함할 수 있는 매우 어려운 산업 비전 상황이다. 이런 환경에서는 이상이 이미지 전체를 크게 바꾸지 않고, 정상 텍스처와 매우 비슷한 미세 결함으로 나타나기 때문에 단순한 reconstruction error만으로는 충분하지 않다.

기존의 많은 비지도 이상 탐지 방법은 정상 이미지만으로 autoencoder나 GAN 같은 generative model을 학습시키고, 입력과 복원 결과의 차이를 이용해 이상을 찾는다. 하지만 이런 접근은 두 가지 구조적 한계를 가진다. 첫째, 이상 영역도 의외로 잘 복원되는 경우가 많아 미세한 결함을 놓칠 수 있다. 둘째, 복원 결과와 입력 결과를 비교하는 후처리나 hand-crafted similarity measure에 크게 의존한다. 논문은 바로 이 지점에서 문제를 다시 정의한다. 표면 이상 탐지를 “복원 문제”로만 보지 않고, “복원 결과와 원본의 관계를 바탕으로 이상 여부를 구분하는 discriminative problem”으로 본다.

핵심적으로 DRÆM은 reconstruction sub-network와 discriminative sub-network를 결합한다. 전자는 이상이 포함된 입력을 정상처럼 복원하려 하고, 후자는 원본 이미지와 복원 이미지를 함께 보고 픽셀 단위 anomaly map을 출력한다. 이 구조의 중요한 점은 실제 이상 데이터 없이도 학습이 가능하다는 것이다. 논문은 anomaly-free image 위에 synthetic anomaly를 입히는 간단한 시뮬레이션 전략을 사용하여 학습용 양성 예제를 만든다.

이 연구가 중요한 이유는 산업 검사에서 결함 샘플이 희귀하고 라벨링 비용이 높기 때문이다. 특히 실제 생산 환경에서는 결함 종류가 다양하고 예측 불가능하므로, 실제 결함을 충분히 수집해서 supervised segmentation을 학습시키기 어렵다. DRÆM은 이런 제약 아래에서도 높은 탐지와 localization 성능을 달성하며, 특히 MVTec AD에서 기존 비지도 방법을 큰 폭으로 앞서고, DAGM에서는 supervised 방법에 가까운 image-level classification 성능과 더 나은 localization 품질을 보였다는 점에서 실용적 가치가 높다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 “이상 자체의 appearance를 직접 학습하려고 하지 말고, 정상으로 복원된 결과와 원본 이미지 사이의 차이를 discriminative하게 해석하자”는 것이다. 즉, synthetic anomaly를 그대로 외우는 segmentation network를 만드는 것이 아니라, 복원된 정상 표현과 실제 입력 사이의 불일치를 이용해 anomaly decision boundary를 학습한다.

논문은 기존 접근의 실패 원인을 두 방향에서 설명한다. 하나는 pure reconstructive approach의 과잉 일반화(over-generalization)이다. autoencoder류 모델은 이상도 그럴듯하게 복원해버릴 수 있어, reconstruction error가 충분히 커지지 않는다. 다른 하나는 pure discriminative approach의 과적합(overfitting)이다. synthetic anomaly만으로 segmentation network를 직접 학습하면, 학습에 사용한 인공 패턴에는 잘 맞지만 실제 결함에는 잘 일반화되지 않는다.

DRÆM은 이 두 극단을 절충하는 것이 아니라, 구조적으로 결합한다. reconstructive sub-network가 입력을 정상 분포 쪽으로 끌어당겨 anomaly-free reconstruction을 만들고, discriminative sub-network는 원본과 복원본을 함께 입력받아 anomaly map을 예측한다. 이렇게 하면 discriminative network는 단순히 “이상 텍스처의 외형”을 외우는 대신, “주어진 지역 패턴이 정상 복원과 얼마나 어긋나는가”를 판단하는 local-appearance-conditioned distance function을 학습하게 된다.

또 하나의 중요한 아이디어는 synthetic anomaly가 실제 결함과 닮아야 할 필요가 없다는 주장이다. 논문은 target-domain anomaly를 정교하게 모사하려 하지 않는다. 대신 just-out-of-distribution한 패턴만 만들면 충분하다고 본다. 이는 decision boundary를 정상 분포 주변에서 타이트하게 형성하는 데 목적이 있기 때문이다. 이 관점은 anomaly simulation을 훨씬 단순하게 만들며, texture image 몇 장만으로도 좋은 성능이 가능하다는 실험 결과와 연결된다.

기존 방법과의 차별점은 세 가지로 요약할 수 있다. 첫째, anomaly-free reconstruction과 anomaly segmentation을 end-to-end로 함께 학습한다. 둘째, 원본과 복원본을 joint embedding으로 사용하여 hand-crafted post-processing 없이 직접 anomaly map을 얻는다. 셋째, synthetic anomaly의 realism에 덜 의존하면서도 실제 이상에 대한 일반화를 확보한다.

## 3. 상세 방법 설명

전체 파이프라인은 크게 세 단계로 이해할 수 있다. 먼저 anomaly-free image $I$에 synthetic anomaly를 삽입해 오염 이미지 $I_a$와 정답 마스크 $M_a$를 만든다. 그다음 reconstructive sub-network가 $I_a$를 입력받아 anomaly-free reconstruction $I_r$를 출력한다. 마지막으로 discriminative sub-network가 원본 입력과 복원 결과를 channel-wise concatenation한 입력을 받아 픽셀 단위 anomaly score map $M_o$를 생성한다.

### 3.1 Reconstructive sub-network

재구성 네트워크는 encoder-decoder 구조이며, 역할은 입력 이미지의 local pattern을 정상 분포에 더 가까운 패턴으로 바꾸는 것이다. 중요한 점은 이 네트워크가 단순히 입력을 복제하는 것이 아니라, synthetic anomaly가 있는 입력 $I_a$를 받아 원래의 anomaly-free image $I$를 복원하도록 학습된다는 점이다. 따라서 네트워크는 암묵적으로 이상 영역을 감지하고, 그 위치를 정상적인 내용으로 inpainting하듯 복원해야 한다.

복원 손실은 $l_2$와 patch-based SSIM 손실을 함께 사용한다. 논문은 단순 $l_2$만 사용하면 인접 픽셀 간 구조적 상관관계를 충분히 반영하지 못한다고 본다. 그래서 SSIM을 추가하여 구조 보존을 유도한다. SSIM 손실은 다음과 같이 정의된다.

$$
L_{SSIM}(I, I_r)=\frac{1}{N_p}\sum_{i=1}^{H}\sum_{j=1}^{W}\left(1-SSIM(I, I_r)(i,j)\right)
$$

여기서 $H, W$는 이미지의 높이와 너비이고, $N_p$는 전체 픽셀 수이며, $SSIM(I, I_r)(i,j)$는 좌표 $(i,j)$를 중심으로 하는 patch 간 structural similarity이다.

최종 reconstruction loss는 다음과 같다.

$$
L_{rec}(I, I_r)=\lambda L_{SSIM}(I, I_r)+l_2(I, I_r)
$$

즉, 픽셀 단위 차이와 구조적 유사도를 동시에 고려한다. 논문은 추가로, reconstructive network가 downstream discriminative network로부터도 간접적인 학습 신호를 받는다고 설명한다. 이것은 end-to-end 학습의 장점으로, reconstruction 자체가 anomaly localization에 유리한 방향으로 조정된다는 뜻이다.

### 3.2 Discriminative sub-network

이 서브네트워크는 U-Net 유사 구조를 사용한다. 입력은 reconstruction 결과 $I_r$와 원본 이미지 $I$를 채널 방향으로 이어 붙인 $I_c$이다. 즉,

$$
I_c = [I_r; I]
$$

와 같이 이해할 수 있다. 여기서 핵심 직관은, 정상 영역에서는 원본과 복원본이 거의 비슷하지만, 이상 영역에서는 복원 네트워크가 그 부분을 정상처럼 바꾸므로 두 표현 간 차이가 커진다는 점이다.

기존 reconstruction 기반 방법은 이런 차이를 SSIM 같은 hand-crafted similarity로 계산한다. 하지만 논문은 표면 이상 탐지에 적합한 similarity measure를 사람이 설계하기 어렵다고 본다. 대신 segmentation network가 원본과 복원본의 joint appearance로부터 적절한 distance function을 자동으로 학습하게 한다.

출력은 입력과 같은 크기의 anomaly score map $M_o$이다. 학습에는 Focal Loss를 사용한다. 이는 픽셀 단위 segmentation에서 hard example에 더 집중하도록 만들어, 어려운 이상 영역을 좀 더 잘 분리하게 해준다. 논문 본문에는 Focal Loss의 구체적 전개식은 쓰지 않았지만, 목적은 class imbalance와 hard pixel 대응이다.

전체 학습 손실은 reconstruction loss와 segmentation loss의 합이다.

$$
L(I, I_r, M_a, M)=L_{rec}(I, I_r)+L_{seg}(M_a, M)
$$

여기서 $M_a$는 synthetic anomaly의 정답 마스크, $M$은 네트워크가 예측한 anomaly mask이다. 이 식은 DRÆM이 복원과 구분을 동시에 학습한다는 사실을 잘 보여준다.

### 3.3 Simulated anomaly generation

이 논문에서 매우 중요한 부분이다. DRÆM은 실제 anomaly sample 없이 학습되기 때문에 synthetic anomaly 생성기가 사실상 양성 예제 생성기 역할을 한다. 그런데 논문은 이 생성기가 실제 결함을 사실적으로 재현해야 한다고 보지 않는다. 목표는 단지 정상 분포에서 약간 벗어난 just-out-of-distribution 패턴을 만드는 것이다.

과정은 다음과 같다. 먼저 Perlin noise generator로 다양한 shape를 갖는 noise image $P$를 만든다. 여기에 무작위 threshold를 적용해 binary anomaly mask $M_a$를 생성한다. 이 마스크는 anomaly의 위치와 형태를 정의한다. 이후 anomaly texture source image $A$를 외부 texture dataset에서 샘플링한다. 논문 실험에서는 DTD를 기본적으로 사용한다.

이 texture image에는 RandAugment 스타일의 무작위 augmentation 3개를 적용한다. 후보 연산은 posterize, sharpness, solarize, equalize, brightness change, color change, auto-contrast이다. 그 뒤 anomaly map으로 texture를 마스킹하고, 원본 이미지와 blending하여 synthetic anomaly image $I_a$를 만든다.

식은 다음과 같다.

$$
I_a=\bar{M}_a \odot I + (1-\beta)(M_a \odot I)+\beta(M_a \odot A)
$$

여기서 $\bar{M}_a$는 $M_a$의 inverse mask이고, $\odot$는 element-wise multiplication이다. $\beta$는 blending opacity로, $[0.1, 1.0]$ 구간에서 균일 샘플링된다.

이 식을 쉽게 풀어 말하면, 정상 영역은 그대로 유지하고, anomaly 영역에서는 원본 이미지 일부와 anomaly texture를 섞는다. $\beta$가 크면 anomaly texture가 강하게 드러나고, 작으면 원본에 더 가까운 미세 결함이 된다. 논문은 바로 이 opacity randomization이 decision boundary를 조이는 데 중요하다고 본다. 너무 강한 anomaly만 학습하면 쉬운 경우만 맞추게 되고, 약한 perturbation까지 포함해야 실제 subtle anomaly에 잘 대응할 수 있기 때문이다.

최종적으로 학습 샘플은 $(I, I_a, M_a)$ 삼중항으로 구성된다. 즉, 정상 원본, synthetic anomaly가 삽입된 입력, 그리고 픽셀 수준 정답 마스크를 자동으로 얻는다.

### 3.4 Surface anomaly localization and image-level detection

DRÆM의 직접 출력은 pixel-level anomaly map $M_o$이다. 이것은 localization에는 그대로 사용될 수 있다. 하지만 image-level anomaly detection도 필요하므로, 논문은 이 anomaly map을 기반으로 이미지 단위 점수 $\eta$를 계산한다.

우선 $M_o$에 mean filter를 적용해 local response를 smoothing한다. 그리고 그 최대값을 이미지 단위 anomaly score로 쓴다.

$$
\eta = \max(M_o * f_{s_f \times s_f})
$$

여기서 $f_{s_f \times s_f}$는 크기 $s_f \times s_f$의 평균 필터이고, $*$는 convolution이다.

이 식의 의미는 단순하다. 이미지 어딘가에 충분히 강한 이상 반응이 있으면 그 이미지를 anomaly로 판단한다. 논문은 preliminary study에서 별도의 classification network도 시도했지만, 이 직접 score estimation보다 개선을 보지 못했다고 보고한다. 따라서 구조를 단순하게 유지한다.

## 4. 실험 및 결과

### 4.1 MVTec AD에서의 비지도 방법 비교

논문은 MVTec anomaly detection dataset을 기본 벤치마크로 사용한다. 이 데이터셋은 15개 object class를 포함하며, 다양한 실제 결함 유형을 제공하므로 비지도 surface anomaly detection 평가의 표준 벤치마크로 널리 사용된다. 평가는 image-level AUROC로 detection 성능을, pixel-level AUROC와 pixel-wise average precision(AP)으로 localization 성능을 본다.

여기서 AP를 별도로 강조한 점이 중요하다. 논문은 surface anomaly localization에서는 anomaly pixel이 매우 적기 때문에, AUROC만으로는 localization 품질을 충분히 반영하지 못한다고 지적한다. 실제로 false positive rate는 정상 픽셀이 너무 많아서 낮게 유지될 수 있으므로, precision을 반영하는 AP가 더 적절하다는 주장이다. 이 평가지표 선택은 설득력이 있다.

학습 설정은 MVTec에서 700 epochs, learning rate $10^{-4}$, 400과 600 epoch에서 0.1배 감소이다. anomaly-free training image에는 $(-45,45)$도 범위의 rotation augmentation을 적용해 overfitting을 줄였다. anomaly source dataset으로는 DTD를 사용했다.

#### Image-level anomaly detection

Table 1에 따르면 DRÆM의 평균 image-level AUROC는 98.0이다. 비교 대상들의 평균은 대략 78.2, 87.3, 87.7, 91.7, 94.4, 95.5 수준이며, 논문은 이전 최고 성능 대비 2.5 percentage point 향상을 주장한다. 실제 표를 보면 가장 강한 baseline이 평균 95.5이고, DRÆM은 98.0이므로 이 주장은 수치적으로 일치한다.

클래스별로 보면 capsule 98.5, grid 99.9, leather 100, pill 98.9, tile 99.6, zipper 100, hazelnut 100, toothbrush 100 등 매우 높은 값을 보인다. 다만 transistor는 93.1, cable은 91.8, screw는 93.9로 상대적으로 낮다. 논문은 특히 “부품 일부가 사라지는 결함”이 어렵다고 설명한다. 예를 들어 transistor lead가 잘려나간 경우, 잘려나간 자리에 흔한 배경이 보이면 모델이 그 부분을 이상으로 강하게 인식하지 못할 수 있다. 이 해석은 reconstruction 기반 방법의 특성과도 잘 맞는다. “무언가 이상한 것이 추가된 경우”보다 “있어야 할 것이 사라진 경우”가 더 어렵다는 뜻이다.

#### Pixel-level anomaly localization

Table 2의 평균 성능은 DRÆM이 97.3 / 68.4 (AUROC / AP)이다. 비교 대상은 US가 93.9 / 45.5, RIAD가 94.2 / 48.2, PaDim이 97.4 / 55.0이다. 즉, pixel AUROC는 PaDim과 거의 비슷하거나 약간 낮지만, AP는 13.4 point 이상 크게 높다. 논문이 localization accuracy에서 큰 향상을 주장하는 이유가 바로 여기에 있다.

이 차이는 중요한 의미를 가진다. DRÆM은 anomaly map의 thresholding 시 precision이 더 좋고, 불필요한 false positive가 적으며, 실제 결함 영역을 더 정밀하게 잡는다는 뜻이다. 클래스별로 보면 tile 92.3 AP, zipper 81.5 AP, hazelnut 92.9 AP, metal nut 96.3 AP, wood 77.7 AP 등 상당히 높은 localization 품질을 보인다. 반면 pill은 48.5 AP로 높지 않은데, 논문은 그 이유 중 하나로 ground-truth annotation ambiguity를 든다. 실제로 노란 점만 이상인데 GT가 알약 전체 표면을 이상으로 표기한 사례를 설명하며, 이런 경우 DRÆM은 오히려 실제 이상 위치를 더 정확히 짚지만 평가 점수는 손해를 볼 수 있다.

이 논문의 실험 해석에서 특히 인상적인 부분은 “GT가 항상 완벽하지 않다”는 점을 명시한 것이다. 이는 anomaly localization 연구에서 자주 간과되는 문제인데, coarse하거나 애매한 라벨은 AP를 왜곡할 수 있다.

### 4.2 Ablation study

논문은 ablation을 세 축으로 수행한다. 첫째는 architecture, 둘째는 anomaly appearance pattern, 셋째는 low perturbation example generation이다.

#### Architecture ablation

Discriminative network만 쓰는 Disc. 실험은 detection 93.9, localization 92.7 / 62.5이다. 전체 DRÆM의 98.0, 97.3 / 68.4보다 명확히 낮다. 이는 synthetic anomaly에 대한 direct segmentation 학습이 실제 anomaly로 일반화되지 못하고 overfitting된다는 논문의 가설을 지지한다.

Reconstructive network만 사용하는 Recon.-AE는 83.9, 89.7 / 47.5로 훨씬 낮다. 즉, reconstruction alone으로는 한계가 크다. 다만 기존 AE-SSIM보다 낫다는 점에서 synthetic anomaly를 이용한 reconstruction training 자체는 도움이 된다고 볼 수 있다. 여기에 MSGMS similarity를 쓴 Recon.-AEMSGMS는 90.7, 93.4 / 50.9로 개선되지만, 여전히 전체 DRÆM보다 크게 뒤처진다. 이것은 hand-crafted similarity measure를 아무리 바꿔도 discriminative joint modeling을 완전히 대체하기 어렵다는 근거가 된다.

또한 최근 supervised defect detection backbone인 Božič et al. [6]을 synthetic anomaly로 재학습한 결과도 92.8, 93.9 / 60.7로 DRÆM보다 낮다. 논문은 이를 통해 anomaly 또는 normality appearance 자체를 배우는 것보다 “normal reconstruction 대비 deviation extent”를 배우는 것이 더 중요하다고 주장한다.

#### Anomaly source와 shape ablation

ImageNet을 anomaly texture source로 사용한 DRÆMImageNet은 97.9, 97.0 / 67.9로 DTD 기반 기본 DRÆM과 거의 동일하다. 즉, anomaly texture source dataset의 정체는 크게 중요하지 않다.

더 극단적으로 homogeneous color region만 사용한 DRÆMcolor조차 96.2, 92.6 / 56.5를 달성한다. 성능은 떨어지지만 여전히 강하다. 이것은 synthetic anomaly가 실제 결함과 닮아야 한다는 통념을 반박하는 실험이다.

Perlin noise 대신 사각형 마스크를 쓴 DRÆMrect도 96.9, 96.8 / 65.1로 비교적 작은 성능 감소만 보인다. 즉, anomaly shape의 realism도 절대적 요소가 아니다. 핵심은 정상 분포 주변에서 구분 경계를 학습할 수 있을 정도의 perturbation diversity다.

#### Low perturbation examples

DRÆMno_aug는 97.4, 94.5 / 64.3이고, image augmentation만 쓴 DRÆMimg_aug는 97.4, 95.0 / 64.5이다. opacity randomization만 쓴 DRÆMβ는 97.9, 97.1 / 68.4로 거의 full DRÆM과 같고, full DRÆM은 98.0, 97.3 / 68.4이다.

이 결과는 매우 의미가 크다. 논문은 특히 opacity randomization이 localization 향상에 핵심적이라고 해석한다. 즉, 합성 이상을 강하게만 주는 것이 아니라, 약한 perturbation까지 섞어 decision boundary를 정상 분포 가까이에 학습시키는 것이 중요하다.

### 4.3 DAGM에서의 supervised 방법과 비교

DAGM은 10개 textured object class를 포함하며, anomaly가 배경과 매우 비슷해서 비지도 방법에 특히 어려운 데이터셋이라고 설명된다. 여기서는 anomaly classification이 기본 과제이며, training labels는 supervised methods에만 제공된다.

DRÆM은 여전히 anomaly-free sample만으로 학습한다. 그럼에도 Table 4에서 AUROC 99.0, TPR 96.5, TNR 99.4, CA 98.5를 기록한다. 이는 RIAD 78.6, US 72.5, MAD 82.4, PaDim 95.0보다 훨씬 높다. 약지도 CADN의 89.1보다도 높다. fully supervised 최고 성능인 Božič et al.의 100에는 못 미치지만 상당히 근접하다.

더 흥미로운 것은 localization 해석이다. DAGM의 training annotation은 defect를 대략 둘러싼 ellipse 수준의 coarse label이므로, supervised methods는 그 부정확한 annotation 형태를 그대로 따라가는 경향이 있다고 논문은 지적한다. 반면 DRÆM은 GT를 쓰지 않으므로 오히려 더 날카롭고 정확한 anomaly map을 낸다. 이 주장은 정성적 예시 Figure 11과 연결된다. 즉, 약하거나 거친 라벨이 있을 때는 supervised learning이 localization 품질 면에서 반드시 유리하지 않을 수 있다는 중요한 관찰이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 reconstruction과 discrimination을 매우 자연스럽게 연결했다는 점이다. 단순히 두 네트워크를 붙인 것이 아니라, reconstruction이 “정상 기준점”을 제공하고, discrimination이 그 기준 대비 deviation을 학습하도록 설계했다. 그래서 pure reconstruction의 과잉 일반화와 pure discrimination의 synthetic overfitting이라는 두 문제를 동시에 완화한다.

두 번째 강점은 synthetic anomaly 설계 철학이 단순하면서도 실용적이라는 점이다. 많은 연구가 “어떻게 더 현실적인 fake defect를 만들 것인가”에 매달리지만, 이 논문은 just-out-of-distribution이면 충분하다고 주장하고 실험으로 뒷받침한다. DTD와 ImageNet이 비슷하고, 심지어 단색 color anomaly도 강한 결과를 내는 점은 이 주장을 상당히 설득력 있게 만든다.

세 번째 강점은 localization metric으로 AP를 강조하고, annotation ambiguity까지 논의했다는 점이다. 이는 단순히 숫자가 높다는 보고를 넘어, evaluation 자체를 비판적으로 다뤘다는 뜻이다. anomaly segmentation 분야에서는 실제로 매우 중요한 관점이다.

네 번째 강점은 실험 결과의 폭이다. MVTec에서 비지도 SOTA를 크게 넘고, DAGM에서는 supervised에 근접한 분류 성능과 더 나은 localization을 보인다. 특히 AP 개선 폭이 크기 때문에 실제 픽셀 수준 defect map이 중요한 응용에서 의미가 크다.

반면 한계도 분명하다. 첫째, 논문은 synthetic anomaly가 실제와 유사할 필요가 없다고 주장하지만, 모든 도메인에서 이 주장이 유지된다고 단정할 수는 없다. 실험은 주로 industrial surface defect 데이터셋에 국한되어 있으며, 자연 이미지나 구조적으로 복잡한 semantic anomaly에는 그대로 확장될지 명확하지 않다.

둘째, reconstruction이 본질적으로 어려운 경우, 예를 들어 정상 패턴 자체가 매우 다양한 경우에는 reconstructive sub-network가 안정적으로 정상 기준을 만들지 못할 가능성이 있다. 논문은 one-class method의 unimodal assumption 문제를 비판하지만, reconstruction module 역시 데이터 다양성이 매우 큰 환경에서는 도전이 있을 수 있다. 다만 본문에서 이를 직접 실험으로 다루지는 않았다.

셋째, transistor의 broken lead 사례처럼 “없어져야 할 것이 아니라 있어야 할 것이 사라진 anomaly”는 여전히 어렵다. 이는 DRÆM의 한계라기보다는 anomaly detection 전반의 난점이지만, 논문 스스로도 일부 클래스에서 성능 저하 원인으로 지적했다.

넷째, 방법의 실제 구현 세부사항 중 일부는 본문에 충분히 상세하지 않다. 예를 들어 reconstructive/discriminative sub-network의 구체적 layer configuration, filter 수, training batch 구성, mean filter 크기 $s_f$ 등은 제공된 본문만으로는 모두 알 수 없다. 따라서 재현성 측면의 완전한 판단은 supplementary material이나 code를 함께 봐야 한다. 이 보고서는 제공된 텍스트만 바탕으로 작성되었으므로, 명시되지 않은 구현 세부는 추정하지 않았다.

비판적으로 보면, DRÆM은 “reconstruction difference를 learned similarity로 해석한다”는 점에서 매우 설득력 있지만, 결국 성능은 reconstruction 품질에 상당 부분 의존한다. 즉, discriminative head가 강하더라도 reconstruction이 이상 영역을 충분히 정상화하지 못하면 joint representation의 separability가 떨어질 수 있다. 따라서 이 방식의 성공은 reconstructive module의 inductive bias에 어느 정도 기대고 있다고 볼 수 있다.

## 6. 결론

이 논문은 surface anomaly detection을 위한 매우 강력한 비지도 프레임워크를 제시한다. 핵심 기여는 anomaly-free reconstruction과 discriminative segmentation을 결합하여, 원본과 복원본의 joint representation 위에서 anomaly decision boundary를 end-to-end로 학습한 것이다. 이로써 hand-crafted post-processing 없이 직접 anomaly localization이 가능해졌고, synthetic anomaly만으로도 실제 결함에 잘 일반화하는 모델을 만들 수 있었다.

정량적으로는 MVTec에서 기존 비지도 방법을 image-level detection에서 평균 2.5 AUROC point, localization에서는 13점 이상 AP로 능가했다. DAGM에서는 supervised 방법에 근접한 classification 성능과 더 나은 localization 품질을 보였다. 이런 결과는 실제 defect sample이 부족한 산업 환경에서 매우 큰 의미를 가진다.

더 넓게 보면, 이 연구는 anomaly detection에서 “정상성 자체를 모델링할 것인가, 이상을 직접 분류할 것인가”라는 이분법을 넘어서는 방향을 제시한다. 즉, 정상 복원을 기준점으로 두고 그로부터의 deviation을 discriminative하게 학습하는 방식이다. 이 아이디어는 이후 industrial inspection뿐 아니라 medical anomaly localization, video anomaly segmentation, representation learning 기반 OOD detection 등으로 확장될 가능성이 있다.

결론적으로 DRÆM은 단순히 성능이 좋은 모델을 제안한 것을 넘어, 실제 anomaly sample 없이도 강한 pixel-level localization이 가능하다는 설계 원리를 명확히 보여준 논문이다. 특히 “realistic simulation이 꼭 필요하지 않다”는 메시지는 향후 anomaly detection 연구에서 데이터 생성과 학습 목표를 재해석하는 데 중요한 영향을 줄 수 있다.
