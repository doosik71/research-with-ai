# Application of belief functions to medical image segmentation: A review

## 논문 메타데이터

- **제목**: Application of belief functions to medical image segmentation: A review
- **저자**: Ling Huang, Su Ruan, Thierry Denoeux
- **출판 연도**: 2023
- **저널**: Information Fusion, Volume 91, March 2023, Pages 737-756
- **arXiv ID**: 2205.01733
- **DOI**: 10.1016/j.inffus.2022.11.008
- **arXiv URL**: https://arxiv.org/abs/2205.01733
- **PDF URL**: https://arxiv.org/pdf/2205.01733v4

## 연구 배경 및 문제 정의

이 논문은 의료영상 분할을 단지 정확도 경쟁의 문제로 보지 않는다. 저자들은 의료영상 분할이 본질적으로 불완전한 입력, 애매한 경계, noisy ground truth, 상충하는 다중 정보원을 다루는 문제라고 본다. 이런 환경에서는 segmentation mask 자체뿐 아니라 그 결과가 얼마나 신뢰할 만한지도 중요하다.

논문의 핵심 문제의식은 다음과 같다.

- 의료영상은 modality가 다양하고 품질이 일정하지 않다.
- voxel/pixel 단위 annotation은 불완전하거나 전문가 간 불일치가 존재한다.
- 기존 확률 기반 모델은 불확실성을 하나의 확률분포로 압축해 ignorance와 randomness를 충분히 분리하지 못한다.
- 실제 임상에서는 여러 영상 modality, 여러 특징, 여러 분류기, 여러 전문가 의견을 함께 종합해 판단한다.

이 논문은 이런 문제를 다루기 위한 형식주의로 belief function theory(BFT), 즉 Dempster-Shafer evidence theory에 주목한다. 저자들의 주장은 명확하다. BFT는 단순 probability보다 불확실성과 무지를 더 직접적으로 표현할 수 있고, 서로 충돌하는 다중 증거를 결합하는 데 유용하다.

## 논문의 핵심 기여

이 논문의 기여는 새 segmentation model 제안이 아니라, BFT 기반 의료영상 분할 연구를 체계적으로 재구성한 데 있다.

1. 의료영상 분할의 불확실성 문제를 `belief, plausibility, ignorance, conflict`의 언어로 다시 정리했다.
2. mass function을 생성하는 BBA(Basic Belief Assignment) 방법을 `supervised`와 `unsupervised`로 구분해 설명했다.
3. 실제 segmentation 방법을 `fusion이 어느 단계에서 일어나는가`를 기준으로 분류했다.
4. single classifier/clusterer와 multiple classifier/clusterer, single-modal과 multimodal input을 조합한 taxonomy를 제시했다.
5. BFT와 deep learning을 결합해 더 정확하고 더 신뢰할 수 있는 segmentation으로 나아가야 한다는 연구 방향을 제안했다.

## 핵심 이론 틀

## 1. 왜 belief function theory인가

논문은 BFT를 Bayesian probability의 대체재라기보다 확장된 불확실성 표현 도구로 다룬다. 핵심 차이는 probability가 각 클래스에 분포를 강제로 할당하는 반면, BFT는 `어느 클래스인지 모른다`는 무지 자체를 질량으로 표현할 수 있다는 점이다.

예를 들어 어떤 voxel이 edema인지 necrosis인지 확신이 없을 때, 확률 모델은 둘 중 하나로 비율을 나눠야 하지만 BFT는 그 불확실성을 집합 수준 가설에 직접 배정할 수 있다. 이 특징이 의료영상처럼 ambiguous boundary와 noisy label이 흔한 환경에서 중요하다고 본다.

## 2. BFT의 기본 구성 요소

논문은 BFT 기초를 간단히 정리한 뒤 segmentation 응용으로 연결한다.

- **Mass function**: 각 가설 또는 가설 집합에 믿음의 질량을 부여
- **Belief / Plausibility**: 어떤 가설에 대한 보수적 신뢰와 가능한 지지 범위 표현
- **Dempster’s rule**: 서로 다른 증거원을 결합
- **Discounting**: 신뢰도가 낮은 정보원에 가중치 감소
- **Decision-making**: 최종적으로 segmentation label을 선택하는 단계

이 구조의 장점은 의료영상 분할 파이프라인을 `특징 추출 -> mass assignment -> evidence fusion -> decision`으로 분해해 생각하게 만든다는 데 있다.

## BBA 방법론 요약

논문은 mass function을 어떻게 만들 것인가를 먼저 정리한다. 이는 BFT 기반 segmentation의 출발점이다.

## 1. Supervised BBA methods

supervised BBA는 라벨이 있는 데이터를 바탕으로 mass function을 구성하는 방법이다.

### 1.1 Likelihood-based methods

저자들은 Shafer’s model과 Appriou’s model을 대표 예시로 든다. 이 계열은 클래스 조건부 likelihood를 이용해 각 가설에 대한 믿음을 정의한다. Appriou 모델은 여기에 reliability factor를 포함해 정보원의 신뢰도를 반영할 수 있다는 점이 특징이다.

이 방법은 수학적으로 해석이 명확하지만, 실제 고차원 딥 특징과 직접 결합하기에는 상대적으로 덜 유연하다는 인상을 준다.

### 1.2 Distance-based methods

evidential KNN, evidential neural network classifier, RBF network 같은 방법이 이 범주에 속한다. 논문은 이 계열이 딥러닝 기반 분할과 결합하기 더 쉽다고 본다. 실제로 deep feature를 뽑은 뒤 그것을 evidential classifier에 넣어 확률 대신 mass function으로 바꾸는 방식이 가능하기 때문이다.

이 점은 실무적으로 중요하다. BFT를 현대 segmentation에 붙이려면 결국 deep feature와 결합돼야 하는데, 저자들은 그 접점이 distance-based evidential classifier 쪽에 더 크다고 본다.

## 2. Unsupervised BBA methods

라벨이 부족한 환경에서는 unsupervised BBA가 중요하다. 논문은 다음 계열을 정리한다.

- FCM 기반 모델
- Evidential C-means(ECM) 같은 credal partition 기반 모델
- Gaussian distribution 기반 모델
- BFOD(binary frame of discernment) 기반 변환

특히 ECM은 fuzzy partition보다 더 풍부한 credal partition을 허용해, 한 샘플을 단일 클러스터가 아니라 클러스터 집합에 대한 질량으로 표현할 수 있다. 이는 영상 경계가 애매한 문제에 잘 맞는다.

논문은 PET/CT 종양 분할에서 ECM 계열이 자주 쓰인다고 지적한다. 다만 이들 방법도 여전히 특징 추출 품질에 크게 의존한다는 점을 한계로 본다.

## 응용 taxonomy: 어디서 fusion이 일어나는가

이 논문의 가장 좋은 부분은 BFT segmentation 연구를 fusion step 기준으로 분류한 것이다. 저자들은 BFT의 핵심을 "어떤 증거를 어느 단계에서 결합하는가"로 본다.

## 1. Single classifier or clusterer

### 1.1 Single-modal evidence fusion

하나의 modality와 하나의 classifier/clusterer를 쓰되, feature-level evidence를 결합하는 구조다. 논문은 cardiac MR, brain tissue MRI, lung CT, spinal canal CT, retinal/color biomedical image, PET/CT lymphoma 등의 사례를 정리한다.

전형적 파이프라인은 다음과 같다.

1. feature extraction
2. BBA를 통한 mass calculation
3. Dempster’s rule 기반 feature-level fusion
4. decision-making

이 범주는 가장 단순한 구조지만, probability 대신 belief mass를 사용함으로써 uncertainty-aware segmentation을 수행하는 기본형으로 제시된다.

### 1.2 Multimodal evidence fusion

PET/CT, multi-sequence MRI처럼 복수 modality를 쓰는 경우다. 이때 BFT는 서로 다른 modality가 제공하는 상보 정보와 충돌 정보를 동시에 다룰 수 있다는 점에서 특히 강점을 가진다.

논문은 multimodal fusion이 BFT segmentation의 가장 인기 있는 응용 축이라고 본다. PET와 CT가 서로 다른 종류의 종양 정보를 제공하는 경우, 단순 concatenation보다 modality-level evidential fusion이 더 설득력 있는 해석을 제공할 수 있기 때문이다.

## 2. Several classifiers or clusterers

저자들은 실제 임상에서 여러 전문가의 의견을 종합하듯, 여러 classifier를 결합하는 구조도 중요하다고 본다.

### 2.1 Single-modal with several classifiers

여기서는 하나의 영상 modality에 대해 여러 classifier 또는 clusterer를 돌리고, 내부적으로 feature-level fusion을 수행한 뒤 classifier-level fusion을 다시 수행한다. 목적은 개별 모델의 편향이나 오판을 완화하는 것이다.

논문은 brain tumor, brain tissue, breast cancer, lymphoma 같은 예시를 제시하며, 여러 evidential classifier를 합치는 방식이 segmentation의 신뢰성과 성능을 함께 높일 수 있다고 본다.

### 2.2 Multimodal with several classifiers

가장 복잡한 구조다. feature-level, classifier-level, modality-level fusion이 모두 등장한다. 저자들이 제시한 프레임에 따르면 이 경우 segmentation 파이프라인은 다섯 단계 이상으로 분해된다.

- mass calculation
- feature-level fusion
- classifier-level fusion
- modality-level fusion
- decision-making

이 구조는 계산적으로 복잡하지만, 여러 입력원과 여러 모델을 모두 불확실성까지 포함해 통합할 수 있다는 점에서 BFT의 철학을 가장 잘 보여준다.

## 대표 사례와 정량적 시사점

이 논문은 자체 benchmark 실험을 하지 않는다. 대신 다양한 사례를 통해 BFT가 어떤 상황에서 강점을 보이는지를 보여준다.

몇 가지 시사점은 다음과 같다.

- cardiac MR segmentation에서 위치와 픽셀 정보를 각각 mass로 만든 뒤 결합하는 접근이 성공적으로 쓰였다.
- CT 기반 lung/spinal canal segmentation에서는 credal partition이 구조 연결 오류를 줄이는 데 도움을 줬다.
- color biomedical image segmentation에서는 Appriou 모델 기반 active contour 결합이 F-score 개선을 보였다.
- PET/CT tumor segmentation에서는 ECM 기반 evidential fusion이 single-modal 대비 Dice 개선을 보였다.
- deep evidential classifier를 PET/CT lymphoma segmentation에 결합한 최근 연구는 accuracy뿐 아니라 reliability 측면의 가능성을 보여줬다.

이 논문이 주는 핵심 메시지는 "BFT가 항상 더 높은 Dice를 낸다"가 아니다. 오히려 `불확실성과 충돌을 명시적으로 모델링할 때 특히 가치가 커진다`는 점이다.

## 일반 segmentation survey와의 차이

이 논문은 U-Net 변형, attention, transformer, decoder 설계 비교에 중심을 두는 일반 survey와 시선이 다르다.

- 일반 survey는 주로 feature representation과 architecture를 본다.
- 이 논문은 uncertainty representation과 evidence fusion을 본다.
- 일반 survey가 `정확도 향상`을 주요 평가축으로 두는 반면, 이 논문은 `신뢰성`과 `해석 가능성`까지 함께 본다.

따라서 이 논문은 segmentation model을 직접 고르는 용도보다는, `의료영상 분할에서 불확실성을 어떻게 모델링할 것인가`를 고민할 때 더 유용하다.

## 한계와 저자들이 제시한 미래 방향

논문 후반부의 한계 분석은 상당히 현실적이다.

### 1. 여전히 low-level feature 중심 연구가 많다

저자들은 기존 BFT segmentation 연구 상당수가 hand-crafted feature나 전통적 clustering/classifier 수준에 머물러 있다고 지적한다. 즉, BFT의 이론적 장점이 최신 deep segmentation backbone과 충분히 결합되지 않았다는 것이다.

### 2. Deep learning과의 결합이 초기 단계다

논문은 Dempster’s rule을 여러 CNN 출력 결합에 적용하거나, evidential neural classifier를 deep feature 위에 얹는 초기 사례를 언급한다. 하지만 이것이 아직 본격적인 표준 패러다임으로 정착한 것은 아니라고 본다.

### 3. 신뢰성 평가가 더 필요하다

정확도 외에 calibration, confidence, out-of-distribution robustness 같은 평가가 필요하지만, 관련 비교 체계는 아직 부족하다.

### 4. Unsupervised evidential deep segmentation은 거의 비어 있다

논문은 ECM 같은 unsupervised BBA를 deep segmentation과 end-to-end로 결합한 연구가 거의 없다고 지적한다. 이는 라벨 부족이 심한 의료영상에서 중요한 미개척 영역으로 남는다.

## 실무적 관점의 해설

### 1. 이 논문은 segmentation보다는 uncertainty survey에 가깝다

제목은 segmentation review지만, 실제 핵심은 `belief function theory를 이용한 uncertainty-aware segmentation`이다. 따라서 최신 segmentation backbone 비교를 기대하면 다소 방향이 다르게 느껴질 수 있다.

### 2. BFT는 확률모델의 경쟁자라기보다 보완재로 읽는 편이 맞다

오늘 관점에서 보면 BFT가 Bayesian deep learning이나 ensemble uncertainty를 대체한다기보다, 무지와 충돌을 더 잘 드러내는 보완적 표현 체계로 읽는 것이 타당하다.

### 3. multimodal medical imaging에서 특히 설득력이 크다

PET/CT, multi-sequence MRI처럼 서로 다른 modality가 서로 다른 오류 양상을 가질 때, BFT의 evidential fusion 프레임은 단순 feature concatenation보다 더 구조적인 해석을 제공한다.

### 4. 최신 foundation model 흐름과는 거리가 있다

이 논문은 2023년 출판이지만, 주된 문제의식은 foundation model이나 promptable segmentation이 아니라 reliability와 uncertainty다. 따라서 오늘 연구 맥락에서는 `trustworthy medical segmentation`의 기반 문헌으로 읽는 편이 적절하다.

## 후속 연구와의 연결

이 논문 이후 자연스럽게 이어질 연구 방향은 다음과 같다.

- U-Net, Swin UNETR, nnU-Net 같은 강력한 backbone 위에 evidential output layer 결합
- ensemble과 BFT를 결합한 uncertainty-aware multimodal segmentation
- weakly supervised / semi-supervised setting에서 evidential clustering 활용
- calibration, abstention, human-in-the-loop correction과 belief mass 연결
- 임상 워크플로에서 voxel uncertainty map을 correction priority로 활용

즉, 이 논문은 segmentation architecture 진화선의 중심에 있지는 않지만, 의료영상 AI가 임상 신뢰성을 어떻게 다룰 것인가라는 더 근본적인 질문에 연결된다.

## 종합 평가

`Application of belief functions to medical image segmentation: A review`는 의료영상 분할 문헌 중에서도 매우 특수한 위치를 차지한다. 이 논문은 더 좋은 backbone이나 더 높은 Dice를 찾는 대신, 분할 결과를 얼마나 신뢰할 수 있는지, 다중 증거와 충돌을 어떻게 다룰지에 초점을 맞춘다.

강점은 분명하다. BFT 기초부터 BBA 생성 방식, fusion 단계별 taxonomy, multimodal 및 multi-classifier 응용까지 일관된 틀로 정리한다. 반면 한계도 분명하다. 최신 딥러닝 분할 구조와의 통합은 아직 초기 단계이며, 다수 사례가 전통적 특징 기반 접근에 머무른다. 그럼에도 의료영상 segmentation을 `정확도`에서 `신뢰성`으로 확장해 읽고 싶다면, 이 논문은 매우 유용한 기준점이다.
