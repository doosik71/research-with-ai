# Boosting Deep Learning Risk Prediction with Generative Adversarial Networks for Electronic Health Records

이 논문의 핵심 주장은 **제한된 라벨을 가진 EHR 시계열 데이터에서, GAN 기반 생성 모델 ehrGAN으로 plausible한 synthetic labeled EHR를 만들고 이를 semi-supervised data augmentation에 활용하면 onset/risk prediction 성능을 높일 수 있다**는 것이다.  

## 1. Paper Overview

이 논문은 Electronic Health Records(EHR) 기반 위험 예측(risk prediction)에서 흔히 발생하는 **라벨 부족 문제**를 해결하기 위한 딥러닝 프레임워크를 제안한다. 의료 데이터는 시계열적이고, 불규칙하며, 노이즈가 많고, 개인정보 제약 때문에 대규모 고품질 라벨 데이터를 확보하기 어렵다. 저자들은 이러한 한계 때문에 기존 supervised deep learning이 충분한 성능을 발휘하지 못한다고 본다.  

논문의 문제 설정은 단순히 “더 좋은 분류기를 만들자”가 아니라, **EHR와 같은 고차원 temporal structured data에서 생성 모델을 이용해 학습 데이터를 보강하고, 그 결과 실제 clinical prediction 성능을 얼마나 향상시킬 수 있는가**에 있다. 이를 위해 저자들은 EHR 특성에 맞게 수정한 GAN인 ehrGAN과 CNN 기반 prediction model을 결합한 semi-supervised learning 프레임워크를 제안한다.  

이 문제가 중요한 이유는 의료 영역에서 rare condition, 정확한 진단 라벨, 장기 추적 기록을 충분히 확보하기 어렵기 때문이다. 따라서 실제로 쓸 수 있는 의료 AI는 “대량의 완전 라벨 데이터”가 아니라 “적은 라벨 + 많은 구조적 제약” 환경에서도 잘 작동해야 한다. 이 논문은 바로 그 조건을 겨냥한다.

## 2. Core Idea

핵심 아이디어는 다음 두 단계로 요약된다.

첫째, EHR 시계열을 직접 생성할 수 있도록 **EHR 전용 GAN(ehrGAN)**을 설계한다. 이 생성기는 단순한 noise-to-sample 방식보다, 실제 환자 기록 주변의 plausible한 variation을 생성하는 방향으로 설계된다. 논문은 생성 분포를 실제 데이터 manifold 주변의 **transition distribution**처럼 다루며, 이를 통해 완전히 엉뚱한 synthetic data가 아니라 실제 샘플과 유사하면서도 새로운 training example을 만든다.  

둘째, 이렇게 생성한 synthetic EHR를 **semi-supervised data augmentation**으로 사용한다. 즉, 생성기가 만들어낸 샘플을 원래의 labeled training set에 보강하여 classifier를 학습시키고, 그 결과 onset prediction의 generalization을 높인다. 논문은 이 방식을 SSL-GAN이라고 부른다.

이 논문의 novelty는 “GAN을 의료 데이터에 썼다” 수준이 아니다. 더 정확히는 다음 두 점이 새롭다.

1. **Sequential / temporal EHR 데이터에 맞춘 GAN 설계**
2. **생성 데이터를 단순 시각화가 아니라 실제 risk prediction boost에 연결**

기존 semi-supervised GAN 연구는 주로 vision/NLP 중심이었고, structured quantitative EHR의 time-series setting에는 직접 적용하기 어렵다고 논문은 지적한다.

## 3. Detailed Method Explanation

### 3.1 전체 파이프라인

전체 구조는 세 요소로 구성된다.

* 기본 예측기: CNN 기반 onset prediction model
* 생성기: ehrGAN
* 학습 방식: generated sample을 활용하는 semi-supervised augmentation (SSL-GAN)

즉, 먼저 baseline predictor로 강한 CNN 분류기를 두고, 그 위에 ehrGAN이 생성한 synthetic EHR 데이터를 추가해 성능을 높이는 구조다.  

### 3.2 Basic Deep Prediction Model

기본 예측기는 1D CNN이다. 환자 $p$의 EHR는 시간 순으로 정렬된 event embedding matrix로 표현된다.

$$
\mathbf{x}^{p} \in \mathbb{R}^{T_p \times M}
$$

여기서 $T_p$는 환자의 medical event 개수, $M$은 embedding 차원이다. 각 row는 시간 순서대로 정렬된 medical event embedding이며, embedding 자체는 동일 EHR corpus에 대해 Word2Vec으로 학습한다.

이 모델은 embedding 차원이 아니라 **temporal dimension에 대해서만 convolution**을 적용한다. 또한 서로 다른 길이의 filter를 조합해 다양한 temporal dependency를 포착하고, 이후 **max-over-time pooling**으로 가변 길이 입력을 고정 길이 representation으로 바꾼 뒤, fully connected softmax로 예측 확률을 출력한다. 저자들은 이 CNN이 여러 baseline 중 가장 경쟁력 있는 기본 모델이라고 설명한다.

이 선택은 중요하다. 많은 EHR 연구가 RNN/LSTM을 떠올리게 하지만, 이 논문에서는 temporal locality를 잡는 CNN이 baseline으로 더 강하게 작동했다고 보고한다. 실험 부분에서도 LSTM/GRU보다 CNN이 낫다고 명시된다.

### 3.3 원래 GAN objective

논문은 표준 GAN 목적함수에서 출발한다.

$$
\min_G \max_D
\mathbb{E}_{\mathbf{x}\sim p*{data}(\mathbf{x})}[\log D(\mathbf{x})]
+
\mathbb{E}_{\mathbf{z}\sim p*{\mathbf{z}}(\mathbf{z})}
[\log(1-D(G(\mathbf{z})))]
$$

여기서 $D(\mathbf{x})$는 샘플이 real일 확률을 내는 discriminator이고, $G(\mathbf{z})$는 noise $\mathbf{z}$를 입력받아 synthetic sample을 생성하는 generator다. Nash equilibrium에 도달하면 생성 분포 $p_g(\mathbf{x})$가 실제 데이터 분포 $p_{data}(\mathbf{x})$를 복원하게 된다.

하지만 논문은 EHR에 이 표준 GAN을 그대로 쓰는 것이 부적절하다고 본다. 이유는 EHR가 이미지처럼 연속적이고 정렬된 grid 데이터가 아니며, temporal irregularity와 structured sparsity를 지니기 때문이다. 따라서 generator/discriminator 설계와 학습 안정화가 모두 수정되어야 한다.  

### 3.4 ehrGAN의 설계 철학

논문에 따르면 discriminator와 generator는 각각 **1D CNN / 1D deconvolutional neural network** 계열로 설계된다. 즉, 기본 예측기와 마찬가지로 temporal structure를 직접 반영하는 방향이다.

특히 생성기는 실제 샘플 $\mathbf{x}$ 주변에서 synthetic sample $\tilde{\mathbf{x}}$를 만드는 조건부 transition 관점으로 설명된다. 논문은 이 분포 $p(\tilde{\mathbf{x}}|\mathbf{x})$가 training example 주변 data manifold의 풍부한 구조를 담고 있으며, 이것이 classifier에 유용한 추가 훈련 데이터를 제공한다고 해석한다.

즉, 이 논문에서 generator는 완전히 무작위 의료기록을 만드는 기계라기보다, **실제 환자 기록에 기반한 근접하지만 새로운 변형 샘플**을 만드는 역할에 가깝다. 이것이 일반 GAN보다 의료 데이터 증강에 더 적합하다는 주장이다.

### 3.5 Generator objective의 의미

논문이 제시하는 generator objective의 핵심은 adversarial loss와 reconstruction-like regularization을 함께 쓰는 것이다. 스니펫에 따르면 생성기는 대략 다음 형태의 목적을 최소화한다.

$$
\mathbb{E}_{\mathbf{x}\sim p*{data}(\mathbf{x})}
\left[
\rho \cdot
\mathbb{E}_{\tilde{\mathbf{x}}\sim p_g(\tilde{\mathbf{x}}|\mathbf{x})}
[-\log D(\tilde{\mathbf{x}})]
+
(1-\rho)\cdot |\bar{\mathbf{x}}-\mathbf{x}|_2^2
\right]
$$

여기서 해석은 다음과 같다.

* 첫 항: synthetic sample이 discriminator를 속이도록 함
* 둘째 항: 생성 샘플이 원본으로부터 너무 멀어지지 않도록 regularize
* $\rho$: realism과 fidelity 사이의 trade-off를 조절하는 하이퍼파라미터

즉, $\rho$가 크면 more adversarial, 작으면 more conservative reconstruction 쪽으로 기운다. 논문 후반의 파라미터 실험은 이 값이 너무 크면 오히려 label consistency가 무너질 수 있음을 보여준다. 특히 $\rho = 1$이면 conditioning sample과 같은 label을 유지한다는 보장이 없어져 성능이 크게 망가진다고 보고한다.  

이 점이 매우 중요하다. 의료 데이터 증강은 “진짜처럼 보이는 샘플”만으로 충분하지 않다. **예측 task에 대해 label-preserving augmentation**이어야 한다. 저자들은 이 문제를 adversarial term과 reconstruction term 사이의 균형으로 다루고 있다.

### 3.6 Training Techniques

논문은 ehrGAN을 SGD 기반으로 generator와 discriminator를 번갈아 최적화한다고 말한다. 또한 표준 GAN처럼 불안정하므로 몇 가지 stabilization trick을 사용한다. 제공된 스니펫에 따르면 discriminator와 generator training 순서를 바꾸고, discriminator를 여러 번 업데이트하는 설정($k=5$가 언급됨)을 사용한다. 이는 GAN training instability와 hyperparameter sensitivity를 줄이기 위한 조치다.  

### 3.7 Semi-supervised Augmentation: SSL-GAN

최종적으로 저자들은 생성된 샘플을 classifier 학습에 넣는 semi-supervised augmentation 방식을 제안한다. 논문은 이를 SSL-GAN이라 부른다. 핵심은 learned ehrGAN이 생성한 synthetic sample들이 단순 노이즈가 아니라, 실제 training sample 주변의 plausible variation을 제공하여 classifier의 decision boundary를 더 robust하게 만든다는 것이다.

정리하면, 이 논문의 방법론은 다음과 같은 흐름을 가진다.

1. EHR event sequence를 embedding sequence로 변환
2. CNN predictor로 baseline risk prediction 수행
3. ehrGAN으로 realistic synthetic EHR 생성
4. synthetic data를 labeled training set에 augment
5. augmented dataset으로 predictor를 다시 학습하여 성능 향상

## 4. Experiments and Findings

### 4.1 데이터와 과제

실험은 두 개의 실제 임상 데이터셋에서 수행된다.

* Heart Failure cohort
* Diabetes cohort

논문은 이들이 real-world longitudinal EHR database에서 추출된 데이터라고 설명하며, 전체 원천 데이터 규모로 218,680명 수준이 언급된다. 실험 목적은 두 가지다.

1. GAN이 실제 EHR와 유사한 샘플을 생성하는지
2. 생성 데이터를 이용해 onset prediction이 향상되는지

### 4.2 Baselines

논문은 기본 비교군으로 여러 deep model을 둔다. 중요한 메시지는 **CNN baseline이 다른 deep baselines보다 강했다**는 점이다. 스니펫에 따르면 GRU와 LSTM도 잘 작동하지만 CNN을 넘지는 못했다. 따라서 이후 boosting 실험은 강한 baseline 위에서 수행된다고 볼 수 있다.

또한 boosted model 비교에서는 최소한 다음 범주들이 등장한다.

* CNN-BASIC: augmentation 없이 원래 training subset만으로 학습한 CNN
* SSL-GAN: learned ehrGAN 기반 augmentation 사용
* 기타 semi-supervised / augmentation 계열 비교군

### 4.3 생성 데이터의 품질 분석

논문은 generated data가 단순히 분류 성능만 높였는지, 아니면 실제 분포를 잘 모사하는지도 분석한다.

첫째, **sequence length distribution**이 원본 데이터와 유사하다. Figure 2에서 generated dataset의 길이 분포가 original dataset과 비슷하다고 보고한다. 이는 generator가 EHR sequence의 기본 구조적 통계를 어느 정도 학습했다는 뜻이다.  

둘째, **top-100 frequent feature의 빈도 분포**도 원본과 유사하다. Figure 3 분석에서 generated data가 상위 빈도 feature들의 frequency pattern을 유지한다고 말한다. 이는 generator가 단순히 길이만 맞춘 것이 아니라 주요 medical code / feature distribution도 상당 부분 재현했음을 시사한다.  

셋째, 논문은 comorbidity/co-occurrence도 중요하다고 강조한다. 제공된 스니펫은 여기서 끊기지만, 문맥상 저자들은 단일 feature 빈도뿐 아니라 환자 기록 내 feature 공존 구조도 생성 데이터가 어느 정도 반영하는지를 보려 한 것으로 읽힌다. 다만 이 부분의 수치적 세부는 제공된 내용만으로는 충분히 확인되지 않는다. 이 점은 논문 텍스트 일부가 잘린 한계 때문에 명시적으로 남겨둔다.

### 4.4 Boosted model의 분류 성능

논문의 핵심 실험은 **라벨 데이터 양이 제한된 상황에서 augmentation이 얼마나 도움이 되는가**이다. 저자들은 labeled fraction을 달리하며 성능을 비교한다. 예를 들어 HF50은 Heart Failure training set의 50%만 사용한 경우, Dia67은 Diabetes training set의 $2/3$만 사용한 경우를 의미한다.

결론적으로 저자들은 generated data를 사용한 제안 방식이 여러 baseline 대비 **significant improvements**를 보였다고 주장한다. 초록과 결론에서도 classification tasks에서 state-of-the-art baselines 대비 유의미한 향상이 있었다고 반복한다.  

제공된 스니펫에서 테이블의 일부 수치(예: 0.9330, 0.9563 등)는 보이지만, 전체 테이블 맥락이 잘려 있어 특정 설정에서 어떤 모델이 얼마만큼 이겼는지를 완전하게 재구성하기는 어렵다. 따라서 여기서는 정성적으로만 정리한다. 다만 논문의 주장은 명확하다.

* 생성 데이터는 단순히 real-like하다
* 그리고 실제 prediction AUROC 등 분류 지표를 개선한다
* 특히 labeled data가 줄어드는 low-resource setting에서 그 이점이 더 중요하다

### 4.5 파라미터 $\rho$의 해석

Figure 5 설명에 따르면 $\rho$ 값에 따라 AUROC가 달라진다. 특히 $\rho = 1$은 학습을 완전히 망친다고 언급되며, 이유는 conditioning된 샘플과 같은 label을 유지하는 제약이 사라지기 때문이라고 해석한다. 이 결과는 이 논문의 증강 방식이 단지 “더 다양한 샘플”이 아니라, **적절한 정도로 원본 레이블 의미를 유지하는 샘플**을 만들어야 함을 보여준다.

## 5. Strengths, Limitations, and Interpretation

### 5.1 Strengths

이 논문의 가장 큰 강점은 문제 정의와 방법이 잘 맞물린다는 점이다. 의료 EHR에서는 대규모 정답 라벨 확보가 어렵고, 따라서 data augmentation이 매우 자연스러운 해법이다. 저자들은 이를 generic GAN이 아니라 **EHR 전용 temporal generator**와 결합했다는 점에서 설계 타당성이 높다.

또 다른 강점은 생성 모델 평가를 “그럴듯해 보인다” 수준에서 멈추지 않고, **길이 분포, feature frequency, classification boost**까지 연결해 봤다는 점이다. 즉, 생성 품질과 downstream utility를 함께 점검했다.  

또한 baseline이 약하지 않다. CNN이 이미 강력한 baseline인데도 그 위에 성능 향상을 보였다는 것은 제안 방식의 실질적 가치를 높인다.  

### 5.2 Limitations

한계도 분명하다.

첫째, 논문의 generated EHR 평가는 여전히 **통계적 유사성 중심**이다. 길이 분포, 상위 feature 빈도, co-occurrence 분석은 중요하지만, 실제 임상적으로 의미 있는 conditional pattern이나 causality까지 보장하지는 않는다. 저자들 스스로 future work에서 domain expert와 함께 더 포괄적인 clinical pattern 분석이 필요하다고 말한다.

둘째, label preservation이 하이퍼파라미터 $\rho$에 민감하다. 이는 생성 데이터가 항상 안전한 augmentation이 아니라는 뜻이다. 의료 영역에서는 잘못된 synthetic sample이 오히려 harmful bias를 강화할 수 있으므로, 이 부분은 실사용 관점에서 큰 리스크다.

셋째, 논문은 2017년 시점의 GAN 안정화 기법을 기반으로 하므로, 오늘날 기준으로 보면 mode collapse, training instability, evaluation insufficiency 등의 오래된 GAN 문제를 충분히 해소했다고 보긴 어렵다. 물론 이는 논문 당시의 기술적 맥락을 감안해야 한다.

넷째, 실험이 두 clinical task에 한정되어 있다. Heart Failure와 Diabetes onset prediction에서 유효하다고 해서 모든 EHR task로 바로 일반화되지는 않는다. 저자도 readmission prediction, representation learning 등 다른 task로의 확장을 future work로 남겨둔다.

### 5.3 해석

비판적으로 보면, 이 논문의 진짜 공헌은 “GAN이 의료에도 된다”가 아니라, **생성 모델을 representation learning이나 privacy surrogate가 아니라 직접적인 low-label prediction booster로 쓴 초기 시도**라는 데 있다. 특히 이미지 분야 augmentation 직관을 구조적 시계열 EHR로 옮기려 했다는 점에서 의미가 있다.

반면 현대 관점에서는 diffusion model, sequence transformer, masked pretraining, contrastive learning 같은 더 강력한 self/semi-supervised 대안과 비교가 없다는 한계가 있다. 하지만 이건 사후적 관점이며, 논문이 나온 시기를 고려하면 충분히 선구적이다.

## 6. Conclusion

이 논문은 제한된 라벨을 가진 temporal EHR 데이터에서 위험 예측 성능을 높이기 위해, **수정된 GAN인 ehrGAN과 CNN predictor를 결합한 semi-supervised data augmentation framework**를 제안했다. 핵심은 실제 환자 기록과 유사한 synthetic EHR를 생성하고, 이를 classifier 학습에 활용하여 generalization을 높이는 것이다.  

실험 결과에 따르면 제안 모델은 생성 데이터의 구조적 plausibility를 보여주었고, Heart Failure 및 Diabetes onset prediction에서 여러 baseline 대비 성능 향상을 달성했다. 따라서 이 연구는 EHR 생성 모델을 단순 샘플링이 아니라 **실질적 임상 예측 향상 도구**로 연결했다는 점에서 의미가 크다.  

실무적으로는 “라벨이 적은 의료 시계열 데이터에서도 augmentation으로 prediction을 강화할 수 있다”는 메시지가 중요하다. 연구적으로는 이후의 healthcare generative modeling, semi-supervised EHR learning, synthetic patient record generation 연구의 초기 기반 중 하나로 읽을 수 있다.
