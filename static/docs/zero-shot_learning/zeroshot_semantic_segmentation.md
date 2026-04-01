# Zero-Shot Semantic Segmentation

Maxime Bucher, Tuan-Hung Vu, Matthieu Cord, Patrick Pérez (2019)

## 🧩 Problem to Solve

Semantic segmentation 모델은 대규모 객체 클래스로 확장하는 데 한계가 있다. 이는 각 새로운 클래스에 대한 광범위한 픽셀 단위 주석이 필요하기 때문이다. 특히, 학습 데이터에서 전혀 접해보지 못한(never-seen) 카테고리의 객체에 대해서는 픽셀 단위 분류를 수행할 수 없다는 문제가 존재한다. 기존 지도 학습 기반 모델은 'unseen' 객체를 'seen' 클래스의 혼합이나 배경으로 잘못 분류하는 경향이 있다.

이 논문은 학습 시점에 어떠한 훈련 예제도 없이, 즉 'zero training examples'로 이전에 보지 못한 객체 카테고리에 대한 픽셀 단위 분류기를 학습하는 **zero-shot semantic segmentation**이라는 새로운 태스크를 소개하고 해결하는 것을 목표로 한다. 문제의 중요성은 실제 응용 환경에서 학습 데이터에 없는 새로운 객체를 마주칠 가능성이 높다는 점에 있다. 논문의 궁극적인 목표는 기존 인식 아키텍처를 재설계하여, 'unseen' 카테고리를 쉽게 수용할 수 있도록 하는 것이며, 이를 위해 'unseen' 레이블만으로 학습이 가능하게 만드는 것이다. 또한, 테스트 시점에 'seen' 및 'unseen' 카테고리 모두를 처리해야 하는 '일반화된(generalized)' zero-shot 분류 시나리오를 해결하고자 한다.

## ✨ Key Contributions

* **새로운 태스크 정의 및 벤치마크**: 이전에 보지 못한 (unseen) 객체 카테고리에 대한 픽셀 단위 분류기 학습인 zero-shot semantic segmentation (ZS3)이라는 새로운 태스크를 도입한다. Pascal-VOC 및 Pascal-Context 데이터셋에 대한 zero-shot 벤치마크를 제안하고 경쟁력 있는 기준선(baselines)을 설정한다.
* **ZS3Net 아키텍처 제안**: 딥 비주얼 세그멘테이션 모델(DeepLabv3+)과 의미론적 단어 임베딩으로부터 시각적 표현을 생성하는 생성 모델을 결합한 새로운 아키텍처인 ZS3Net을 제안한다. ZS3Net은 테스트 시점에 'seen' 및 'unseen' 카테고리 모두를 처리하는 '일반화된' zero-shot 분류 문제를 해결한다.
  * **합성 특징 생성**: 'seen' 클래스의 word2vec 임베딩에 조건을 부여하여, DeepLabv3+에서 추출된 'seen' 클래스의 실제 시각적 특징과 일치하는 합성 특징을 생성하는 GMMN(Generative Moment Matching Network) 기반의 생성 모델을 훈련한다. 이 생성 모델은 'unseen' 클래스를 위한 픽셀 수준 합성 특징을 생성하는 데 활용된다.
  * **분류기 재훈련**: 생성된 'unseen' 클래스의 합성 특징과 'seen' 클래스의 실제 특징을 결합하여, 세그멘테이션 네트워크의 최종 분류 레이어를 재훈련한다. 이 과정을 통해 모델은 'seen' 및 'unseen' 클래스 모두를 분류할 수 있는 능력을 얻는다.
* **자가 학습 (Self-training) 도입 (ZS5Net)**: 'unseen' 클래스에 대한 레이블 없는 픽셀 데이터가 훈련 시점에 사용 가능한 완화된 zero-shot 설정에서 모델 성능을 추가적으로 향상시키기 위한 자가 학습(self-training) 단계를 제안한다. 이는 ZS3Net의 예측을 기반으로 자동화된 pseudo-label을 생성하여 활용한다. 이 모델을 ZS5Net (ZS3Net with Self-Supervision)이라고 명명한다.
* **그래프-맥락 인코딩 (Graph-context encoding) 확장**: 특히 복잡한 장면에서 객체 간의 공간적 관계와 같은 맥락 정보가 중요함을 인지하고, 클래스별 세그멘테이션 맵에서 얻은 공간적 우선순위(spatial context priors)를 완전히 활용하기 위해 그래프-맥락 인코딩을 도입한다. 이는 생성 파이프라인에 그래프 컨볼루션 레이어를 사용하여 구현된다.

## 📎 Related Works

* **Semantic Segmentation**: 컨볼루션 신경망(CNN)의 발전과 함께 크게 진보한 분야로, 초기에는 region proposal 분류 방식 [17]에서 시작하여, FCN [28]에 의해 end-to-end 방식으로 전환되었다. U-Net [37], SegNet [3], DeepLab 시리즈 [10,11], PSPNet [50]과 같은 최신 SOTA 모델들은 FCN 기반이며, atrous/dilated convolution [10,46]이나 pyramid context pooling [50] 등을 활용하여 CNN 특징에 맥락 정보를 보강하는 전략을 사용한다.
  * **본 논문과의 차별점**: 기존의 대부분의 semantic segmentation 접근 방식은 지도 학습(supervised) 방식에 의존하며, 학습 데이터에 포함되지 않은 새로운 클래스를 인식하는 데 근본적인 한계가 있다. 본 논문은 학습 데이터에 특정 클래스가 완전히 누락된(missing altogether) 상황, 즉 zero-shot learning 문제를 해결한다.
* **Weakly-Supervised Segmentation**: 이미지 레벨 [33,34] 또는 바운딩 박스 레벨 [13] 주석을 활용하는 약한 지도 학습 방식의 세그멘테이션 연구도 활발하다.
  * **본 논문과의 차별점**: 본 논문은 주석의 강도(weakly-supervised)가 아닌, 클래스의 존재 유무(zero-shot)에 초점을 맞춘다.
* **Zero-Shot Learning (ZSL) for Image Classification**: 최근 몇 년간 활발히 연구되어 온 분야로, 이미지 데이터와 클래스 설명을 공통 임베딩 공간으로 매핑하여 의미론적 유사성을 공간적 근접성으로 변환하는 접근 방식 [1,2,6,8,16,24,32,36,39,42,43,48]이 주를 이룬다. 투영(projection) 방식으로는 단순 선형 [1,2,6,16,24,36], 비선형 다중 모달 [39,43], 하이브리드 [8,32,42,48] 등이 있다. 최근에는 'seen' 클래스로부터 'unseen' 클래스의 합성 인스턴스를 생성하는 방식 [7,25,45]도 제안되었다.
  * **본 논문과의 차별점**: ZSL이 이미지 분류 외의 다른 태스크로 확장된 사례는 객체 탐지(zero-shot object detection) [4,14,35,51]가 있으나, 이 논문이 Zero-Shot Semantic Segmentation에 대한 최초의 접근 방식임을 주장한다. 본 연구는 zero-shot learning의 개념을 픽셀 단위의 dense prediction 태스크인 semantic segmentation으로 확장한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

ZS3Net의 방법론은 세 단계로 구성되며, DeepLabv3+ 세그멘테이션 모델을 기반으로 'seen' 및 'unseen' 클래스 모두에 대한 픽셀 단위 분류를 수행한다. 여기에 자가 학습(Self-training)과 그래프-맥락 인코딩(Graph-context encoding)이 추가 확장 기법으로 제시된다.

### 1. 전체 파이프라인 및 시스템 구조 (ZS3Net)

ZS3Net은 크게 훈련된 semantic segmentation 모델에서 특징을 추출하고, 이 특징과 클래스 임베딩을 활용하여 'unseen' 클래스의 특징을 생성한 다음, 이 합성 특징으로 분류 레이어를 재훈련하는 과정을 따른다.

#### 1.1. 픽셀 단위 데이터 정의 및 수집 (Step 0)

* **기반 모델**: DeepLabv3+ semantic segmentation 모델 [11]을 'seen' 클래스의 주석 데이터로 완전 지도 학습(full-supervision) 방식으로 사전 훈련한다. 이 모델은 표준 교차 엔트로피 손실로 훈련된다.
* **특징 추출**: 사전 훈련된 DeepLabv3+에서 마지막 분류 레이어를 제거하고, 나머지 네트워크를 특징 추출기("backbone network")로 활용한다. 이때 추출되는 특징 $x$는 DeepLabv3+의 $1 \times 1$ 컨볼루션 분류 레이어가 입력으로 받는 특징이다.
* **클래스 임베딩**: 각 클래스 $c \in C$는 word2vec [30] 모델을 통해 $d_a$ 차원의 벡터 표현 $a[c] \in \mathbb{R}^{d_a}$으로 매핑된다. 이 임베딩은 위키피디아 말뭉치로 학습되었으며, 단어 간의 의미론적 관계를 기하학적으로 포착한다.
* **훈련 데이터 구성**: 'seen' 클래스에 대한 훈련 세트 $D_s = \{(x^s_i, y^s_i, a^s_i)\}$는 $d_x$ 차원의 특징 맵 $x^s_i \in \mathbb{R}^{M \times N \times d_x}$, 해당 ground-truth 세그멘테이션 맵 $y^s_i \in S^{M \times N}$, 그리고 각 픽셀에 해당 클래스의 semantic embedding을 할당한 클래스 임베딩 맵 $a^s_i \in \mathbb{R}^{M \times N \times d_a}$으로 구성된다. 여기서 $M \times N$은 인코딩된 특징 맵의 해상도이다.
* **Unseen 클래스**: $K=|U|$개의 'unseen' 클래스에 대해서는 훈련 데이터가 없으며, 오직 해당 클래스의 카테고리 임베딩 $a[c]$ (c $\in U$)만 사용 가능하다.

#### 1.2. 생성 모델 훈련 (Step 1)

* **목표**: 클래스의 이미지를 직접 접하지 않고도 클래스 임베딩 벡터에 조건을 부여하여 이미지 특징을 생성하는 능력이다.
* **모델**: [7]을 따라 "Generative Moment Matching Network (GMMN)" [27]을 특징 생성기 $G$로 채택한다. GMMN은 목표 데이터 분포와 생성된 데이터 분포를 비교하는 미분 가능한 기준을 사용하는 파라메트릭 무작위 생성 프로세스이다.
* **생성 과정**: 무작위 샘플 $z$ (고정된 다변수 가우시안 분포에서 추출)와 의미론적 설명 $a$를 입력으로 받아, 새로운 픽셀 특징 $\hat{x} = G(a, z; w) \in \mathbb{R}^{d_x}$를 생성한다. 여기서 $w$는 생성기의 학습 가능한 파라미터이다.
* **훈련**: 생성기 $G$는 'seen' 클래스에서 추출된 실제 특징의 감독 하에 훈련된다. 학습 목표는 각 의미론적 설명 $a$에 대해, $D_s$에서 추출된 실제 특징 집합 $X(a)$와 생성기 $G$로 샘플링된 합성 특징 집합 $\hat{X}(a;w)$ 간의 **최대 평균 불일치(Maximum Mean Discrepancy, MMD)**를 최소화하는 것이다.
    $$ L_{GMMN}(a) = \sum_{x,x' \in X(a)} k(x,x') + \sum_{\hat{x},\hat{x}' \in \hat{X}(a;w)} k(\hat{x},\hat{x}') - 2 \sum_{x \in X(a)} \sum_{\hat{x} \in \hat{X}(a,w)} k(x,\hat{x}) $$
    여기서 $k(x,x')$는 Gaussian 커널이며, $k(x,x') = \exp(-\frac{1}{2\sigma^2} \|x-x'\|^2)$으로 정의된다. 파라미터 $w$는 확률적 경사 하강법(SGD) [5]으로 최적화된다.

#### 1.3. 분류 모델 미세 조정 (Step 2)

* **합성 데이터셋 구성**: GMMN이 훈련된 후에는 'unseen' 클래스를 포함한 모든 클래스에 대해 픽셀 단위 특징을 무작위로 샘플링하여 합성 'unseen' 훈련 세트 $\hat{D}_u = \{(\hat{x}^u_j, y^u_j, a^u_j)\}$를 구성할 수 있다.
* **분류기 재훈련**: 'unseen' 클래스의 합성 특징 $\hat{D}_u$와 'seen' 클래스의 실제 특징 $D_s$를 결합하여 DeepLabv3+의 분류 레이어 $f$ (1x1 컨볼루션 레이어로 구성)를 미세 조정한다.
* **추론**: 재훈련된 픽셀 단위 분류기 $\hat{y} = f(x; \hat{D}_u, D_s)$는 이제 'seen' 및 'unseen' 클래스 모두의 카테고리를 처리할 수 있게 되며, 이를 통해 이미지 내의 객체를 semantic segmentation할 수 있다.

### 2. 자가 학습 (Self-training, ZS5Net)

* **목표**: 'unseen' 클래스의 레이블 없는 이미지를 훈련 시점에 사용할 수 있는 완화된(relaxed) zero-shot 설정에서 모델의 성능을 향상시킨다.
* **절차**:
    1. ZS3Net(단계 2 이후 훈련된 모델)을 사용하여 레이블 없는 추가 이미지들을 자동으로 "주석" 처리한다.
    2. 생성된 pseudo-label 중에서 가장 신뢰도가 높은 상위 $p\%$의 픽셀들만 'unseen' 클래스를 위한 새로운 훈련 특징으로 사용한다.
    3. 이 추가된 데이터와 함께 semantic segmentation 네트워크를 다시 훈련한다.
* **명칭**: 이 모델은 ZS3Net with Self-Supervision의 약자인 ZS5Net으로 불린다. 이는 순수하게 'unseen' 데이터가 없는 transductive ZSL과는 다르다.

### 3. 그래프-맥락 인코딩 (Graph-context encoding)

* **목표**: 특히 객체가 많은 복잡한 장면에서, 객체들의 공간적 배열과 같은 맥락 정보가 인식 성능 향상에 기여할 수 있다.
* **방법**:
    1. 세그멘테이션 마스크를 의미론적으로 연결된 구성 요소들의 인접 그래프 $G=(V,E)$로 표현한다. 각 노드 $v \in V$는 단일 클래스 레이블의 연결된 그룹에 해당하며, 경계를 공유하는 두 그룹은 이웃으로 간주된다.
    2. 생성기 $G$를 그래프 컨볼루션 레이어 [23]를 사용하여 이 그래프 $G$를 추가 입력으로 받도록 재설계한다.
    3. 각 입력 노드 $v$는 해당 의미론적 임베딩 $a_v$와 무작위 가우시안 샘플 $z_v$를 연결(concatenate)하여 표현된다.
    4. 수정된 생성기는 입력 그래프와 동일한 구조를 가지지만, 각 출력 노드에는 생성된 시각적 특징이 첨부된 그래프를 출력한다.
* **적용**: ZS3Net 모델에 이 그래프 맥락을 추가한 모델은 'ZS3Net + GC'로 표기된다.

### 구현 세부 사항

* **백본**: ResNet-101 [20] 기반의 DeepLabv3+ [11] 프레임워크를 사용한다.
* **옵티마이저**: SGD [5]를 사용하며, 다항식 학습률 감쇠(base learning rate $7e^{-3}$, weight decay $5e^{-4}$, momentum $0.9$)를 적용한다.
* **GMMN**: 하나의 hidden layer를 가진 Multi-Layer Perceptron (MLP)으로 구성되며, Leaky-ReLU [29]와 Dropout [41]을 사용한다. hidden 뉴런 수는 256개, 커널 대역폭은 $\{2, 5, 10, 20, 40, 60\}$으로 설정된다. 입력 가우시안 노이즈는 300차원 word2vec 임베딩과 동일한 차원이다. Adam 옵티마이저 [21]와 학습률 $2e^{-4}$로 훈련된다.
* **그래프 컨텍스트**: GMMN의 선형 레이어를 그래프 컨볼루션 레이어 [23]로 대체한다.

## 📊 Results

### 1. 데이터셋, 작업, 기준선, 지표

* **데이터셋**:
  * **Pascal-VOC 2012 [15]**: 20개 객체 클래스, 1,464개 훈련 이미지. 추가 semantic boundary 주석 [19] 사용.
  * **Pascal-Context [31]**: 59개 객체/사물 클래스, 4,998개 훈련 및 5,105개 검증 이미지 (Pascal-VOC 2010 기반).
* **작업**: Zero-Shot Semantic Segmentation (ZS3).
* **Zero-shot 설정**: 2, 4, 6, 8, 10개의 'unseen' 클래스를 포함하는 다양한 무작위 분할을 구성한다. 'unseen' 세트는 점진적으로 확장된다 (예: 4-unseen은 2-unseen을 포함).
* **기준선**: [16]의 ZSL 분류 접근 방식을 변형하여 DeepLabv3+의 마지막 분류 레이어를 300채널 word2vec 임베딩 맵으로 변환하는 투영 레이어로 대체한다. 모델은 출력 임베딩과 목표 임베딩 간의 코사인 유사도를 최대화하도록 훈련된다. ALE [1]도 K=2에 대해 실험하여 유사한 성능을 보였다.
* **평가 지표**:
  * **Pixel Accuracy (PA)**: 픽셀 정확도.
  * **Mean Accuracy (MA)**: 클래스별 평균 정확도.
  * **Mean Intersection-over-Union (mIoU)**: 클래스별 평균 IoU, semantic segmentation의 표준 지표.
  * **Harmonic mean (hIoU)**: 'seen' 및 'unseen' mIoU의 조화 평균. 'seen' 클래스에 대한 편향을 완화하고, 두 카테고리 모두에서 균형 잡힌 성능을 나타내는 지표로 사용된다.
* **평가 프로토콜**: "일반화된 ZSL 평가(generalized ZSL evaluation)" [9]를 사용한다. 이는 'unseen' 세트뿐만 아니라 'seen' 클래스를 포함한 모든 클래스에 대해 공동으로 평가하는 더 도전적인 방식이다.

### 2. 주요 정량적 결과

#### Pascal-VOC (Table 1)

* **ZS3Net vs. Baseline**:
  * 기준선은 'seen' 클래스에서 높은 mIoU를 보이지만, 'unseen' 클래스에서는 매우 낮은 mIoU(예: 2-split에서 3.2%)를 기록하여 'seen' 클래스에 대한 강한 편향을 보인다.
  * ZS3Net은 'unseen' 클래스에서 기준선 대비 PA, MA, mIoU에서 상당한 성능 향상을 달성한다 (예: 2-split에서 'unseen' mIoU +32.2%p).
  * 'seen' 클래스에서는 ZS3Net이 기준선과 비슷하거나 약간 더 나은 성능을 보인다.
  * 전체적으로 ZS3Net은 모든 클래스에 걸쳐 기준선 대비 우수한 성능을 나타내며, 특히 hIoU에서 큰 폭의 개선을 보인다.
* **Generalized vs. Vanilla ZSL Evaluation (Table 2)**:
  * 'unseen' 10-split에 대해 기준선은 vanilla ZSL 평가에서 41.7% mIoU를 달성하지만, generalized 평가에서는 1.9% mIoU로 크게 하락한다.
  * ZS3Net은 generalized 평가에서 18.1% mIoU를 기록하며, 'seen' 및 'unseen' 클래스의 실제 및 합성 특징을 활용하여 'seen' 클래스에 대한 편향을 효과적으로 줄였음을 보여준다.

#### Pascal-Context (Table 3)

* Pascal-VOC보다 더 복잡한 데이터셋임에도 불구하고, ZS3Net은 기준선 대비 모든 평가 지표에서 상당한 성능 향상을 보인다.
* **ZS3Net + GC (Graph-context encoding)**:
  * 그래프-맥락 인코딩을 추가한 'ZS3Net + GC' 모델은 ZS3Net 단독 모델 대비 꾸준한 성능 향상을 보인다 (예: 2-split에서 'unseen' mIoU가 21.6%에서 30.0%로 증가). 이는 복잡한 장면에서 맥락 정보 활용의 중요성을 뒷받침한다.

#### Zero-shot segmentation with self-training (ZS5Net) (Table 4)

* 자가 학습 단계를 추가한 ZS5Net은 ZS3Net 대비 'seen', 'unseen' 및 전체 클래스 모두에서 성능을 크게 향상시킨다.
* 특히, 2-unseen split에서는 Pascal-VOC 및 Pascal-Context 모두에서 모든 지표에서 전체 성능이 supervised 성능과 매우 근접한 수준을 달성한다.
* **파라미터 p의 영향 (Figure 5)**: 자가 학습에서 높은 점수를 받은 'unseen' 픽셀의 비율 $p$는 성능에 영향을 미치며, Pascal-VOC에서는 $p=25\%$, Pascal-Context에서는 $p=75\%$로 고정되었다.

### 3. 정성적 결과 (Figure 1, 4, 6, 7, 8, 9, 10)

* 'w/o ZSL' 모델(seen 클래스로만 훈련)은 'unseen' 객체를 배경이나 'seen' 클래스로 잘못 분류하는 경향을 보인다. 예를 들어, 'cat'을 'dog'으로, 'plane'을 'boat'이나 'background'로 인식한다.
* 제안된 ZS3Net은 이러한 'unseen' 객체('motorbike', 'cat', 'plane', 'cow', 'boat')를 올바르게 인식하고 분할하는 능력을 시각적으로 보여준다.
* 자가 학습을 추가한 ZS5Net은 ZS3Net이 'seen' 클래스로 잘못 분류한 픽셀('motorbike', 'sofa')을 더 명확하게 'unseen' 클래스로 구분하는 데 도움을 준다.

## 🧠 Insights & Discussion

### 1. 논문에서 뒷받침되는 강점

* **선구적인 연구**: Semantic segmentation 분야에 zero-shot learning 개념을 성공적으로 도입하고, 'unseen' 클래스에 대한 픽셀 단위 분류를 가능하게 하는 효과적인 아키텍처 ZS3Net을 제시함으로써 해당 분야의 새로운 방향을 제시한다.
* **효과적인 합성 특징 생성**: word2vec 임베딩을 조건으로 GMMN 기반 생성 모델을 사용하여 'unseen' 클래스의 시각적 특징을 성공적으로 합성한다. 이 방법은 'unseen' 클래스에 대한 훈련 데이터가 없는 상황에서 'generalized zero-shot' 세팅에서 분류 성능을 크게 향상시킨다.
* **실용적 평가 프로토콜**: 'seen' 클래스와 'unseen' 클래스 모두를 평가하는 '일반화된 ZSL 평가' 프로토콜을 채택하여, 모델이 'seen' 클래스에 대해 가질 수 있는 편향을 명확히 분석하고, ZS3Net이 이 편향을 효과적으로 완화함을 입증한다. 이는 실제 환경에서의 적용 가능성을 높이는 중요한 요소이다.
* **성능 향상 기법의 유효성**: 자가 학습(ZS5Net)은 레이블 없는 'unseen' 데이터가 있는 경우 성능을 대폭 개선하며, 그래프-맥락 인코딩(ZS3Net + GC)은 복잡한 장면에서 공간적 맥락 활용을 통해 추가적인 성능 이득을 제공한다. 이러한 확장 기법들은 ZS3Net의 강점을 더욱 강화한다.
* **프레임워크의 유연성**: 생성 모델로 GMMN 외에 GAN [45]도 실험하여 유사한 성능을 보였으며, GMMN이 더 나은 안정성을 제공한다고 언급함으로써 제안된 프레임워크가 특정 생성 모델에 종속되지 않는 유연성을 가지고 있음을 시사한다.

### 2. 한계, 가정 또는 미해결 질문

* **Word Embedding의 한계**: 클래스 간의 의미론적 관계를 word2vec 임베딩이 잘 포착한다고 가정하지만, 특정 시각적 속성이 의미론적 임베딩 공간에 완전히 반영되지 않거나, 추상적인 개념에 대한 임베딩의 표현력이 제한될 수 있다. 이는 'unseen' 클래스의 시각적 특징을 정확히 생성하는 데 방해가 될 수 있다.
* **생성된 특징의 품질**: 생성 모델이 실제 'unseen' 클래스 특징 분포를 얼마나 정확하게 모사하는지에 대한 심층적인 분석은 부족하다. 생성된 특징이 현실적이지 않을 경우, 분류기 훈련에 부정적인 영향을 미칠 수 있다.
* **ZS5Net의 현실성(Relaxed Setting)**: 자가 학습을 위한 ZS5Net은 'unseen' 클래스의 "레이블 없는 픽셀이 학습 시점에 사용 가능하다"는 가정을 한다. 이는 순수한 zero-shot learning (unseen 데이터가 전혀 없는 상황)과는 다른 '완화된 설정'이며, 모든 실제 시나리오에서 이러한 레이블 없는 데이터를 쉽게 얻을 수 있을지 불분명하다.
* **Graph-Context Encoding의 추론 단계**: 그래프-맥락 인코딩이 생성 모델 훈련 시 'true segmentation masks'를 활용한다고 언급되어 있다. 'unseen' 클래스를 포함하는 마스크는 사용되지 않는다고 하지만, 실제 추론 시에 이러한 마스크 정보를 어떻게 얻을지에 대한 명확한 설명이 부족하다. 만약 추론 시에도 ground-truth 마스크에 의존한다면, 실제 적용 가능성에 한계가 있을 수 있다.
* **하이퍼파라미터 튜닝의 투명성**: GMMN의 커널 대역폭이나 자가 학습의 $p$ 값과 같은 하이퍼파라미터가 "zero-shot cross-validation procedure"를 통해 선택되었다고 명시되어 있으나, 이 절차의 구체적인 내용이나 하이퍼파라미터의 민감도 분석은 제시되지 않았다.
* **스케일링에 따른 성능 저하**: 'unseen' 클래스 수가 증가함에 따라 (예: 2개에서 10개로) ZS3Net의 'unseen' mIoU 및 hIoU가 감소하는 경향을 보인다. 이는 'unseen' 클래스 수가 매우 많아질 경우 zero-shot 성능을 유지하는 것이 여전히 도전적인 과제임을 시사한다.

### 3. 논문에 근거한 간략한 비판적 해석 및 논의사항

이 논문은 zero-shot learning을 semantic segmentation이라는 도전적인 비전 태스크에 성공적으로 적용한 선구적인 연구이다. 특히, 'unseen' 클래스를 위한 합성 특징 생성이라는 핵심 아이디어는 독창적이며 효과적임을 입증한다. ZS3Net은 'generalized zero-shot' 평가 프로토콜을 통해 현실적인 성능을 보여주며, 'seen' 클래스에 대한 편향을 효과적으로 완화함으로써 기존 ZSL 연구의 한계를 극복하는 데 중요한 기여를 한다. 자가 학습과 그래프-맥락 인코딩은 모델의 성능을 추가적으로 개선하는 스마트한 확장으로 평가할 수 있다.

그러나 ZS5Net의 '완화된 설정'은 순수한 zero-shot 시나리오의 엄격함에서 벗어나므로, 레이블 없는 'unseen' 데이터가 전혀 없는 상황에서의 성능 개선 방안에 대한 추가 연구가 필요할 수 있다. 또한, 그래프-맥락 인코딩의 추론 과정에서의 현실적인 적용 방안과, word embedding의 한계를 극복하기 위한 더 풍부한 의미론적 표현 방식 탐색은 향후 연구에서 다룰 수 있는 중요한 방향이다. 전반적으로, 이 연구는 zero-shot semantic segmentation 분야의 초석을 다지고, 미래 연구를 위한 강력한 기준선을 제시했다는 점에서 컴퓨터 비전 분야에 상당한 영향을 미칠 것으로 예상된다.

## 📌 TL;DR

이 논문은 학습 과정에서 전혀 보지 못한(unseen) 객체 카테고리를 픽셀 단위로 분류하는 **Zero-Shot Semantic Segmentation (ZS3)**이라는 새로운 컴퓨터 비전 태스크를 제시한다. 핵심 아이디어는 기존 딥 세그멘테이션 모델(DeepLabv3+)과 **생성 모델(GMMN)**을 결합하는 **ZS3Net** 아키텍처이다. ZS3Net은 'seen' 클래스의 word2vec 임베딩에 기반하여 'unseen' 클래스의 시각적 특징을 합성하고, 이 합성 특징과 'seen' 클래스의 실제 특징을 함께 사용하여 최종 분류 레이어를 재훈련한다. 이를 통해 모델은 'seen' 및 'unseen' 클래스 모두를 인식하고 분할할 수 있게 된다.

또한, 논문은 'unseen' 클래스의 레이블 없는 픽셀이 사용 가능한 경우 성능을 더욱 향상시키는 **자가 학습(Self-training)** 단계(ZS5Net)와, 복잡한 장면에서 공간적 맥락 정보를 활용하기 위한 **그래프-맥락 인코딩(Graph-context encoding)** 기법(ZS3Net + GC)을 추가적으로 제안한다. Pascal-VOC 및 Pascal-Context 데이터셋에 대한 실험 결과, ZS3Net은 기존 zero-shot learning 기준선 대비 'unseen' 클래스에서 뛰어난 성능 향상을 보였으며, 자가 학습과 그래프-맥락 인코딩은 이 성능을 더욱 강화했다. 이는 'seen' 클래스에 대한 편향을 줄이고 '일반화된 zero-shot' 설정에서 강건한 성능을 달성했음을 의미한다.

이 연구는 방대한 수동 주석 없이 semantic segmentation 모델을 수많은 클래스로 확장할 수 있는 가능성을 열어주며, 현실 세계에서 발생하는 새로운 객체에 대한 인식 및 분할 문제를 해결하는 데 중요한 역할을 할 잠재력을 가진다.
