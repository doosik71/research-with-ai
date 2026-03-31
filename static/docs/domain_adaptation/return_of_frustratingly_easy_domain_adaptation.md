# Return of Frustratingly Easy Domain Adaptation

* **저자**: Baochen Sun, Jiashi Feng, Kate Saenko
* **발표연도**: 2015
* **arXiv**: [https://arxiv.org/abs/1511.05547](https://arxiv.org/abs/1511.05547)

## 1. 논문 개요

이 논문은 **unsupervised domain adaptation** 문제를 매우 단순한 방식으로 해결하려는 연구이다. 문제의 출발점은, 소스 도메인(source domain)에서 학습한 분류기가 타깃 도메인(target domain)에서는 잘 작동하지 않는다는 점이다. 예를 들어 한 데이터셋의 이미지로 학습한 객체 인식기가 다른 촬영 환경이나 다른 수집 방식의 이미지에서는 성능이 크게 떨어질 수 있다. 텍스트 감성 분류도 마찬가지로, 어떤 상품 카테고리의 리뷰로 학습한 모델이 다른 카테고리 리뷰에는 잘 일반화되지 않을 수 있다. 이런 현상을 논문은 **domain shift**라고 다룬다.

기존의 많은 domain adaptation 방법은 타깃 도메인에 일부 라벨이 있다고 가정하는 **supervised adaptation**에 초점을 두었다. 그러나 현실에서는 타깃 도메인 라벨이 없는 경우가 훨씬 흔하다. 따라서 이 논문은 **타깃 라벨이 전혀 없는 상황**에서 소스와 타깃의 분포 차이를 줄이는 방법을 제안한다.

핵심 제안은 **CORAL (CORrelation ALignment)** 이다. 이 방법은 소스와 타깃 특징 벡터의 **2차 통계(second-order statistics)**, 즉 **covariance**를 맞추는 방식으로 도메인 차이를 줄인다. 논문은 이 방법이 매우 단순하고 계산량도 적으면서, 객체 인식과 감성 분석의 여러 표준 벤치마크에서 강력한 기존 기법들을 이긴다고 주장한다. 특히 deep feature에 적용했을 때 매우 좋은 성능을 보였다는 점을 강조한다.

이 연구의 중요성은 두 가지다. 첫째, 매우 복잡한 적응 네트워크나 추가 손실 없이도 strong baseline을 만들 수 있음을 보여준다. 둘째, deep learning 시대에도 간단한 통계 정렬만으로 큰 성능 향상을 얻을 수 있음을 실험으로 보여준다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 다음 한 문장으로 요약할 수 있다. **소스 특징을 whitening한 뒤, 타깃 covariance로 다시 coloring하면 소스 분포를 타깃 분포에 가깝게 만들 수 있다.**

논문은 특징을 각 차원별로 평균 0, 표준편차 1로 정규화하더라도, 도메인 차이는 여전히 남는다고 지적한다. 그 이유는 각 차원의 스케일만 맞춰도 **특징들 사이의 상관관계(correlation)** 는 맞춰지지 않기 때문이다. 즉, 두 도메인이 평균과 분산은 비슷해도 covariance 구조가 다르면 분포는 여전히 다르다. CORAL은 바로 이 covariance 차이를 줄이는 데 집중한다.

기존 manifold 기반 방법들, 예를 들어 GFK, SA, TCA 등은 source와 target을 어떤 저차원 subspace에 투영하거나 subspace를 맞추는 방식이다. 논문은 이런 접근이 복잡하고 하이퍼파라미터 선택이 필요하며, 주로 eigenvector 정렬에는 초점을 맞추지만 eigenvalue 차이까지 충분히 다루지 못한다고 비판한다. 반면 CORAL은 covariance 전체를 정렬하므로, eigenvector뿐 아니라 eigenvalue까지 함께 반영한다는 점을 차별점으로 제시한다.

또한 MMD 계열 방법과 비교하면, CORAL은 일종의 moment matching으로 볼 수 있지만, 보다 직접적으로 **2차 모멘트**를 맞추며, 무엇보다 **닫힌형(closed-form) 해석**과 단순한 구현을 제공한다. 게다가 많은 MMD 기반 방법이 source와 target에 같은 변환을 적용하는 symmetric transformation을 쓰는 반면, CORAL은 source를 target 쪽으로 보내는 **asymmetric transformation**을 사용한다. 논문은 이런 비대칭 변환이 더 유연하고 실제 적응에 더 유리하다고 설명한다.

## 3. 상세 방법 설명

### 전체 파이프라인

CORAL의 전체 절차는 매우 간단하다.

먼저 소스 도메인 특징 집합을 $D_S = {\vec{x}_i}$, 타깃 도메인 특징 집합을 $D_T = {\vec{u}_i}$라고 하자. 각 특징은 $D$차원 벡터이다. 이때 소스와 타깃의 평균과 covariance를 각각 $\mu_s, \mu_t$ 및 $C_S, C_T$라고 두면, 정규화 이후 평균은 대체로 0에 가깝게 맞출 수 있지만 covariance는 다를 수 있다.

CORAL은 소스 특징에 선형변환 $A$를 적용하여, 변환된 소스 covariance가 타깃 covariance와 최대한 가까워지도록 한다. 그 다음 이 변환된 소스 특징으로 평범하게 supervised classifier를 학습하고, 이를 타깃 데이터에 적용한다. 즉, 적응은 특징 공간에서 끝나고, 분류기 자체는 일반적인 SVM 같은 모델을 그대로 사용할 수 있다.

### 최적화 목표

논문이 세운 목표는 변환 후 소스 covariance와 타깃 covariance의 Frobenius norm 차이를 최소화하는 것이다.

$$ \min_A | C_{\hat{S}} - C_T |_F^2 $$

여기서 $C_{\hat{S}}$는 변환된 소스 특징의 covariance이다. 선형변환 $A$를 적용하면,

$$ C_{\hat{S}} = A^\top C_S A $$

이므로 목적함수는 다음과 같다.

$$ \min_A | A^\top C_S A - C_T |_F^2 $$

이 식의 의미는 명확하다. 소스 특징에 적절한 선형변환을 가해서, 그 결과의 2차 통계 구조가 타깃과 비슷해지도록 하겠다는 것이다.

### 해석적 해와 그 의미

논문은 일반적인 경우 covariance가 low-rank일 수 있다고 본다. 실제 데이터는 종종 저차원 manifold 위에 놓여 있으므로, covariance가 full rank가 아닐 수 있다. 이 점을 고려해 정리와 정리를 통해 최적해를 유도한다.

최종적으로 제시된 최적 선형변환은 다음 구조를 가진다.

$$ A^* = (U_S {\Sigma_S^+}^{\frac{1}{2}} U_S^\top) (U_{T[1:r]} {\Sigma_{T[1:r]}}^{\frac{1}{2}} U_{T[1:r]}^\top) $$

여기서 $C_S = U_S \Sigma_S U_S^\top$, $C_T = U_T \Sigma_T U_T^\top$는 covariance의 SVD 또는 고유값 분해 형태이고, $\Sigma_S^+$는 Moore-Penrose pseudoinverse이다. 또 $r = \min(r_{C_S}, r_{C_T})$이다.

이 식을 직관적으로 보면 두 단계로 나뉜다.

첫 번째 부분인
$$ U_S {\Sigma_S^+}^{\frac{1}{2}} U_S^\top $$
는 소스 데이터를 **whitening**한다. 즉, 소스 데이터의 상관관계를 제거해 decorrelated한 공간으로 보낸다.

두 번째 부분인
$$ U_{T[1:r]} {\Sigma_{T[1:r]}}^{\frac{1}{2}} U_{T[1:r]}^\top $$
는 그 whitening된 소스 데이터를 타깃 covariance 구조에 맞게 다시 **re-coloring**한다.

그래서 전체적으로는 “source를 하얗게 만든 뒤, target의 색을 입힌다”는 설명이 가능하다. 논문 제목이 쉽고 단순한 적응을 강조하는 이유가 여기에 있다.

### 실제 구현 방식

이론적으로는 위와 같은 해석적 해를 쓸 수 있지만, 논문은 실제 구현에서는 더 간단하고 안정적인 **classical whitening/coloring**을 추천한다. 이 방식에서는 covariance에 작은 정규화 항 $\lambda$를 더해 수치적 불안정을 줄인다. 논문에서는 $\lambda = 1$을 사용했다.

알고리즘은 사실상 다음 네 단계이다.

먼저 소스와 타깃 covariance를 구한다.

$$ C_S = cov(D_S) + I $$

$$ C_T = cov(D_T) + I $$

여기서 $I$는 identity matrix이다. 본문 알고리즘에는 $+I$가 들어가며, 설명 부분에서는 일반적으로 작은 regularization $\lambda$를 diagonal에 더하는 방식으로 이해하면 된다.

그 다음 소스를 whitening한다.

$$ D_S \leftarrow D_S C_S^{-\frac{1}{2}} $$

마지막으로 타깃 covariance로 recoloring한다.

$$ D_S^* \leftarrow D_S C_T^{\frac{1}{2}} $$

즉 최종 적응된 소스 특징은

$$ D_S^* = D_S C_S^{-\frac{1}{2}} C_T^{\frac{1}{2}} $$

형태로 이해할 수 있다.

### 왜 source만 target 쪽으로 변환하는가

논문은 source와 target 둘 다 whitening하는 방법은 실패한다고 설명한다. 이유는 두 도메인이 서로 다른 subspace 위에 있을 가능성이 높기 때문이다. 둘 다 decorrelate만 해버리면 서로 맞춰지는 것이 아니라, 오히려 구조 정보가 사라질 수 있다.

또한 타깃을 소스 쪽으로 보내는 대신 source를 target 쪽으로 보내는 쪽이 더 낫다고 주장한다. 논문은 그 이유를 완전히 이론적으로 증명하지는 않지만, 직관적으로는 **source label 정보와 target의 unlabeled structure를 함께 활용하는 방향**이기 때문에 더 좋을 수 있다고 설명한다.

### 분류기 학습과 추론

CORAL은 특징만 바꾸므로 어떤 base classifier와도 결합 가능하다. 논문 실험에서는 linear SVM을 사용한다. 적응된 소스 특징 위에서 분류기를 학습한 뒤, 타깃 특징에는 별도 적응 없이 그대로 적용한다.

선형 분류기 $f_{\vec{w}}(I)=\vec{w}^T \phi(I)$를 사용할 때는 특징을 직접 바꾸는 대신 분류기 파라미터 $\vec{w}$ 쪽에 동등한 변환을 적용할 수도 있다고 언급한다. 이는 타깃 샘플 수나 차원이 매우 큰데 분류기 수는 적을 때 효율적일 수 있다. 다만 논문 본문에는 이 부분의 구체적인 수식 전개까지는 자세히 나오지 않는다.

### Deep neural network와의 관계

논문은 CORAL을 deep network의 hidden layer feature에도 적용할 수 있다고 설명한다. 예를 들어 AlexNet의 fc6, fc7 activation을 feature로 보고, 이 위에 CORAL을 적용할 수 있다. 또한 원칙적으로는 여러 층 뒤에 CORAL 변환층을 넣는 **multilayer CORAL**도 가능하다고 말한다. 하지만 본 논문의 실험은 한 번에 하나의 hidden layer feature에 적용하는 수준이다. 따라서 deep architecture 내부에 end-to-end로 통합한 구체적인 학습 절차까지를 제시한 논문은 아니다.

## 4. 실험 및 결과

논문은 객체 인식과 감성 분석 두 작업에서 CORAL을 평가한다. 공통적으로 **타깃 도메인 라벨은 사용하지 않는다**. 기본 분류기는 linear SVM이며, SVM의 $C$ 값은 source domain에서 cross-validation으로 선택한다. CORAL 자체에는 실질적으로 추가 하이퍼파라미터가 거의 없고, whitening/coloring 정규화용 $\lambda$ 정도만 있다.

### 4.1 객체 인식: Office-Caltech10, shallow SURF feature

첫 번째 실험은 Office-Caltech10 데이터셋에서 800-bin bag-of-words SURF feature를 사용한다. 네 도메인 Amazon, Caltech, DSLR, Webcam 사이의 12개 domain shift를 평가한다. 각 설정마다 20회 랜덤 trial 평균 정확도를 보고한다. 학습은 소스 도메인에서 제한된 수의 라벨 샘플만 사용하고, 타깃은 모두 unlabeled test로 쓴다.

Table 1의 평균 정확도는 다음과 같다.

* NA: 37.8
* SVMA: 41.0
* DAM: 40.2
* GFK: 43.4
* TCA: 45.7
* SA: 45.9
* **CORAL: 46.7**

즉 CORAL이 전체 평균에서 가장 높다. 특히 일부 shift에서는 개선 폭이 크다. 논문이 직접 강조한 예시는 $D \rightarrow W$에서 **56.4에서 85.9로** 크게 상승한 경우다. 다만 모든 개별 shift에서 무조건 최고는 아니고, 예를 들어 $D \rightarrow A$에서는 SA의 42.0이 CORAL의 38.1보다 높다. 따라서 정확한 해석은 “전반적인 평균 성능에서 strongest”이지 “모든 경우에서 절대 최고”는 아니다.

### 4.2 객체 인식: Office, deep feature

두 번째 실험은 표준 Office dataset에서 deep feature를 사용한다. 여기서는 AlexNet의 fc6, fc7과 source-only fine-tuning 후의 FT6, FT7 feature를 사용한다. 비교 대상에는 SA, GFK, TCA뿐 아니라 당시의 deep adaptation 방법들인 DLID, DANN, DDC, DAN, ReverseGrad 등이 포함된다.

Table 2의 평균 결과를 보면:

* NA-fc6: 62.2
* NA-fc7: 64.4
* NA-FT6: 62.0
* NA-FT7: 65.5
* CORAL-fc6: 64.0
* CORAL-fc7: 66.9
* CORAL-FT6: 68.5
* **CORAL-FT7: 69.4**

즉 CORAL은 pre-trained deep feature에도 효과가 있고, fine-tuned source feature와 결합했을 때 더 큰 향상을 보인다. 흥미로운 점은 source-only fine-tuning 자체는 target 성능을 꼭 올리지 않는데, 그 위에 CORAL을 얹으면 성능이 크게 좋아진다는 것이다. 논문은 이를 “pre-trained network는 underfitting일 수 있고, fine-tuned network는 overfitting인데, CORAL이 분포를 맞춰 주면서 overfitting 문제를 완화할 수 있다”는 방향으로 해석한다. 다만 이것은 저자들의 가능한 설명이지, 엄밀히 증명된 사실은 아니다.

또한 DAN과 ReverseGrad가 보고한 일부 shift와 비교했을 때, CORAL은 3개 중 2개 shift에서 더 좋은 성능을 냈다고 서술한다. 논문이 강조하는 포인트는 복잡한 adaptation loss나 network retraining 없이도 competitive하거나 더 나은 결과를 얻는다는 점이다.

### 4.3 더 큰 규모 평가: full training protocol

논문은 source training data를 더 많이 썼을 때도 평가한다. 여기서는 subsampling 대신 모든 source 데이터를 사용한다.

#### Office-Caltech10 with SURF

Table 3 평균 정확도는 다음과 같다.

* NA: 41.1
* SA: 41.5
* GFK: 46.4
* TCA: 42.8
* **CORAL: 48.8**

여기서도 CORAL이 가장 좋다.

#### Cross-Dataset Testbed with DECAF-fc7

Table 4 평균 정확도는 다음과 같다.

* NA: 38.5
* SA: 25.8
* GFK: 31.3
* TCA: 26.6
* **CORAL: 40.2**

이 실험에서는 클래스 수가 40개로 늘어난 상황에서도 CORAL이 가장 높은 평균 성능을 낸다. 또한 deep feature에서 CORAL의 우위가 shallow feature보다 더 크게 나타난다고 논문은 해석한다.

논문은 Table 1과 Table 3을 비교해, source 데이터가 많아질수록 NA와 adaptation 방법 간의 차이가 상대적으로 줄어든다고 말한다. 이는 source 학습 데이터가 많을수록 classifier 자체가 더 강해져 domain generalization이 어느 정도 개선되기 때문일 수 있다고 추정한다.

### 4.4 감성 분석: Amazon review

논문은 Amazon review dataset에서도 실험한다. 도메인은 Kitchen, DVD, Books, Electronics 네 개이고, 각 도메인마다 positive 1000개, negative 1000개 리뷰가 있다. 특징은 상위 400개 단어로 줄인 bag-of-words이다. 20 random split 평균을 보고한다.

Table 5 평균 정확도는 다음과 같다.

* NA: 76.7
* TCA: 63.0
* SA: 77.0
* GFS: 69.6
* GFK: 71.7
* SCL: 76.7
* KMM: 77.8
* **CORAL: 78.0**

여기서 CORAL의 평균 성능이 가장 좋다. 다만 이미지 실험보다 개선 폭은 작다. 흥미로운 점은 TCA, GFS, GFK 같은 일부 최신 방법이 오히려 NA보다 못한 경우가 있다는 점이다. 논문은 이 감성 분석 문제 자체가 어렵고, sparse bag-of-words 특성상 상관구조가 이미지보다 약하기 때문일 수 있다고 해석한다.

### 결과 해석

논문 전체 결과의 핵심 메시지는, **CORAL의 이점이 deep feature에서 특히 크게 나타난다**는 점이다. 논문은 deep feature covariance의 largest singular value가 shallow SURF보다 훨씬 크다는 예를 들며, deep feature가 더 강한 correlation 구조를 갖기 때문일 수 있다고 설명한다. 또한 텍스트 bag-of-words는 매우 sparse하고 correlation이 약해 개선 폭이 상대적으로 작다고 본다.

즉 이 논문은 “CORAL이 모든 표현에서 동일하게 잘 먹힌다”보다, **특히 correlation이 강한 feature일수록 더 큰 효과를 낼 가능성**을 시사한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **극단적으로 단순한 방법인데도 강력한 성능을 낸다**는 점이다. 실제 구현은 covariance 계산과 행렬 제곱근/역제곱근만으로 끝나며, 논문 표현대로 거의 네 줄짜리 MATLAB 코드로 구현 가능하다. 많은 복잡한 deep adaptation 기법과 달리 구조 변경, adversarial loss, 여러 손실 간 가중치 조정 같은 설계 부담이 없다.

둘째, **모델 비종속적(model-agnostic)** 이다. CORAL은 특징 변환 방식이므로 SVM뿐 아니라 다른 분류기에도 적용 가능하다. 또한 deep network의 특정 hidden layer feature에도 쉽게 적용할 수 있어 활용 범위가 넓다.

셋째, **재현성이 높다**. 논문이 제시한 방법은 하이퍼파라미터 의존성이 작고, 실험 설정도 상대적으로 단순하다. 복잡한 adaptation network에 비해 결과 재현이 쉽다는 점은 실제 연구와 응용에서 큰 장점이다.

넷째, **비대칭 적응(asymmetric adaptation)** 이라는 직관이 실용적으로 잘 작동한다. source를 target covariance에 맞추는 것은, 단지 공통 공간을 찾는 것보다 실제 적용 상황에 더 직접적이다.

하지만 한계도 분명하다.

첫째, CORAL은 기본적으로 **2차 통계만 정렬**한다. 즉 평균과 covariance 수준의 차이를 다루지만, 더 복잡한 고차 통계나 class-conditional mismatch까지 직접 다루지는 않는다. source와 target 간 차이가 단순한 선형 covariance mismatch를 넘는 경우에는 한계가 있을 수 있다.

둘째, **선형변환 가정**에 기반한다. 복잡한 비선형 도메인 차이를 직접 모델링하지 않기 때문에, 적응이 필요한 구조가 매우 비선형적이면 충분하지 않을 수 있다. 논문은 deep feature 위에서 적용함으로써 어느 정도 이 문제를 우회하지만, CORAL 자체는 본질적으로 선형 정렬이다.

셋째, 논문은 deep architecture 내부에 multilayer CORAL을 넣을 수 있다고 언급하지만, **실제로 end-to-end 학습된 CORAL layer 체계**를 실험으로 충분히 검증하지는 않는다. 따라서 “깊은 네트워크 내부의 일반적 해결책”까지 확장해서 읽으면 과한 해석이 된다.

넷째, 실험은 강력하지만, 일부 비교는 각 논문이 보고한 숫자나 공개 코드 기반이며, 모든 방법이 정확히 동일한 조건에서 재현된 것은 아니다. 특히 deep adaptation 계열은 일부 shift만 보고된 경우가 있어, 완전히 동일한 범위의 공정 비교라고 단정할 수는 없다.

다섯째, 논문은 source를 target 쪽으로 보내는 것이 target을 source로 보내는 것보다 낫다고 말하지만, 그 이유는 주로 경험적 설명에 가깝다. 왜 이러한 비대칭성이 항상 좋은지에 대한 강한 이론은 이 논문 안에 충분히 제시되어 있지 않다.

비판적으로 보면, 이 논문은 매우 간단한 baseline이 얼마나 강력할 수 있는지를 보여준다는 점에서 큰 가치가 있다. 동시에, 그 강점이 곧 한계이기도 하다. CORAL은 복잡한 상황을 모두 해결하는 보편 이론이 아니라, **“domain shift의 상당 부분이 covariance mismatch로 설명되는 경우”에 매우 효과적인 실용 기법**으로 보는 것이 가장 정확하다.

## 6. 결론

이 논문은 unsupervised domain adaptation을 위해 **CORAL**이라는 매우 단순한 방법을 제안했다. 핵심은 소스 특징을 whitening한 뒤 타깃 covariance로 recoloring하여, 두 도메인의 **2차 통계**를 맞추는 것이다. 목적함수는 transformed source covariance와 target covariance의 Frobenius norm 차이를 최소화하는 형태이며, 실제 구현은 매우 간단하다.

실험적으로 CORAL은 객체 인식과 감성 분석의 여러 표준 벤치마크에서 no adaptation baseline과 여러 기존 기법을 대체로 능가했다. 특히 deep feature에서 성능 향상이 더 크게 나타났고, 이 점은 deep representation이 강한 correlation 구조를 가지기 때문일 가능성이 있다고 논문은 해석한다.

이 연구의 가장 큰 기여는 복잡한 적응 네트워크나 다중 손실 없이도, 단순한 covariance alignment만으로 강력한 domain adaptation 성능을 낼 수 있음을 설득력 있게 보여준 데 있다. 실제 응용 측면에서는 feature extraction 뒤 가볍게 붙일 수 있는 강력한 후처리 기법으로 유용하다. 향후 연구 측면에서는 deep text feature, multilayer integration, 그리고 보다 복잡한 통계 정렬 방식으로 확장될 여지를 남긴다.

종합하면, 이 논문은 “단순한 방법이 반드시 약한 방법은 아니다”라는 점을 domain adaptation 분야에서 매우 인상적으로 보여준 대표적인 연구라고 평가할 수 있다.
