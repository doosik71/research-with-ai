# Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer

## 논문 메타데이터

- **제목**: Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer
- **저자**: Xiangde Luo, Minhao Hu, Tao Song, Guotai Wang, Shaoting Zhang
- **출판 연도**: 2022
- **학회**: The 5th International Conference on Medical Imaging with Deep Learning (MIDL 2022)
- **시리즈**: Proceedings of Machine Learning Research, Volume 172
- **페이지**: 820-833
- **arXiv ID**: 2112.04894
- **PMLR URL**: https://proceedings.mlr.press/v172/luo22b.html
- **PDF URL**: https://proceedings.mlr.press/v172/luo22b/luo22b.pdf
- **코드**: https://github.com/HiLab-git/SSL4MIS

## 연구 배경 및 문제 정의

이 논문은 semi-supervised medical image segmentation에서 CNN과 Transformer를 함께 쓰되, 복잡한 consistency regularization 대신 서로의 pseudo label을 직접 가르치는 `cross teaching` 전략을 제안한다. 저자들의 문제의식은 분명하다. 의료영상 분할은 픽셀 또는 복셀 단위 주석이 매우 비싸기 때문에, 소량의 라벨과 대량의 비라벨 데이터를 함께 쓰는 semi-supervised learning이 중요하다.

기존 semi-supervised segmentation은 주로 CNN backbone 위에서 발전해 왔다. 하지만 저자들은 두 가지 구조적 한계를 본다.

- CNN은 local convolution 기반이라 전역 문맥과 장거리 관계 모델링이 약하다.
- Transformer는 전역 표현력이 강하지만 데이터가 더 많이 필요하고, 적은 라벨 환경에서 직접 semi-supervised 학습하기 어렵다.

따라서 이 논문의 핵심 질문은 "CNN과 Transformer의 서로 다른 학습 패러다임을 semi-supervised training에서 상호 보완적으로 활용할 수 있는가"다.

## 논문의 핵심 기여

이 논문의 기여는 방법의 단순성과 구조적 통찰에 있다.

1. CNN과 Transformer 사이의 `cross teaching`이라는 매우 단순한 semi-supervised 학습 프레임워크를 제안했다.
2. classical deep co-training의 explicit consistency regularization을 직접적인 pseudo-label supervision으로 단순화했다.
3. 서로 다른 learning paradigm인 CNN과 Transformer를 조합하는 것이 같은 계열 네트워크끼리 교차 학습하는 것보다 낫다는 점을 보였다.
4. ACDC benchmark에서 8개의 기존 semi-supervised 방법보다 더 좋은 성능을 보고했다.
5. Transformer branch를 최종 추론에 쓰지 않더라도, CNN branch 학습을 보조하는 complementary teacher로 충분히 가치가 있음을 보였다.

## 방법론 구조 요약

## 1. 전체 프레임워크

프레임워크는 매우 단순하다. 하나의 입력 이미지를 두 개의 분할 네트워크에 동시에 넣는다.

- **CNN branch**: U-Net
- **Transformer branch**: Swin-UNet

라벨이 있는 데이터에 대해서는 두 네트워크 모두 ground truth로 supervised loss를 받는다. 라벨이 없는 데이터에 대해서는 한 네트워크의 예측이 다른 네트워크의 pseudo label이 된다.

즉, 학습 구조는 다음과 같다.

- labeled image: 각자 정답으로 학습
- unlabeled image: CNN prediction이 Transformer를 가르치고, Transformer prediction이 CNN을 가르침

## 2. Cross teaching의 핵심

논문이 제안하는 cross teaching은 unlabeled sample에 대해 각 네트워크의 예측을 argmax로 pseudo label로 만든 뒤, 반대편 네트워크의 loss를 계산하는 방식이다.

수식 수준에서 핵심은 다음과 같다.

- CNN prediction `p_i^c`
- Transformer prediction `p_i^t`
- CNN용 pseudo label은 `argmax(p_i^t)`
- Transformer용 pseudo label은 `argmax(p_i^c)`

이후 unlabeled loss는 두 Dice loss의 합으로 구성된다.

저자들이 강조하는 점은 이것이 explicit consistency regularization과 다르다는 것이다. consistency regularization은 두 예측이 비슷해지도록 직접 제약하지만, cross teaching은 pseudo label을 통해 간접적으로 더 안정적인 정답 비슷한 신호를 만든다고 본다.

## 3. 왜 CNN과 Transformer의 조합인가

논문의 핵심 논리는 네트워크 구조의 `차이`가 오히려 도움이 된다는 것이다.

- CNN은 local information에 강하다.
- Transformer는 long-range relation modeling에 강하다.
- 두 네트워크의 오류 패턴이 다를 가능성이 높다.

이 차이 때문에 같은 구조의 두 CNN을 쓰는 것보다, CNN과 Transformer를 함께 두는 것이 unlabeled pseudo supervision에서 더 상보적인 효과를 낼 수 있다는 것이 저자들의 주장이다.

## 4. 손실 함수

전체 목적함수는 supervised loss와 cross-teaching loss의 합이다.

- **Supervised loss**: cross-entropy + Dice
- **Unsupervised loss**: bidirectional Dice-based cross teaching

그리고 unsupervised loss의 가중치는 Gaussian warm-up 방식으로 점진적으로 증가시킨다.

이 설계는 기존 semi-supervised segmentation과 유사하지만, 핵심 차이는 unlabeled supervision source가 EMA teacher나 consistency target이 아니라 `상대 branch의 hard pseudo label`이라는 점이다.

## 실험 설정

논문은 ACDC public benchmark를 사용한다. 이 데이터셋은 cardiac cine-MR short-axis 영상으로, 세 구조를 분할한다.

- RV
- Myo
- LV

세부 설정은 다음과 같다.

- 총 100명 환자, 200개 annotated cine-MR images
- 70명 환자 데이터를 train, 30명 환자 데이터를 validation
- 입력은 256x256으로 resize
- 2D slice-wise segmentation 수행
- 평가 지표는 3D Dice coefficient와 HD95

네트워크는 U-Net과 Swin-UNet을 사용했고, 배치 크기 16, SGD optimizer, poly learning rate를 사용했다.

## 주요 결과 해석

## 1. Transformer를 기존 semi-supervised 기법에 바로 넣는 것은 잘 안 됐다

ablation에서 저자들은 Mean Teacher, DAN, DCT, Entropy Minimization 같은 기존 semi-supervised 방법의 backbone을 U-Net 대신 Swin-UNet으로 바꿔본다. 결과는 좋지 않았다. 7개 labeled case 기준으로 Swin-UNet 기반 방법들의 mean DSC는 약 `0.511~0.529` 수준에 머문다.

논문이 내리는 결론은 분명하다. Transformer는 data-hungry하므로, 기존 CNN 중심 semi-supervised 전략을 그대로 옮기는 것만으로는 충분하지 않다.

## 2. CNN-Transformer cross teaching이 가장 효과적이었다

7 labeled cases 기준 ablation에서 다음 결과가 나온다.

- **CNN & CNN (CT)**: mean DSC `0.833`, HD95 `11.0`
- **Trans & Trans (CT)**: mean DSC `0.813`, HD95 `10.4`
- **CNN & Trans (CR)**: mean DSC `0.820`, HD95 `15.1`
- **CNN & Trans (CT, 제안법)**: mean DSC `0.864`, HD95 `8.60`

이 결과는 두 가지를 말해 준다.

- consistency regularization보다 cross teaching이 낫다.
- 같은 계열끼리보다 CNN과 Transformer의 조합이 더 잘 작동한다.

즉, 성능 향상의 핵심은 Transformer를 추가한 것 자체보다도 `서로 다른 학습 패러다임의 상호 보완`에 있다.

## 3. 기존 8개 semi-supervised 방법보다 우수

ACDC에서 3 labeled cases와 7 labeled cases 모두 제안법이 가장 좋은 결과를 냈다.

### 3 labeled cases

- **Ours**: mean DSC `0.656`, HD95 `16.2`
- **가장 강한 기존 비교군(CPS)**: mean DSC `0.603`, HD95 `25.5`

라벨이 매우 적은 극저라벨 환경에서 특히 개선 폭이 크다.

### 7 labeled cases

- **Ours**: mean DSC `0.864`, HD95 `8.60`
- **CPS**: mean DSC `0.833`, HD95 `11.0`
- **Fully supervised with all labels**: mean DSC `0.911`, HD95 `3.60`

논문 본문은 7 labeled setting에서 제안법이 두 번째 방법인 CPS 대비 DSC 약 `3.8%`, HD95 약 `3.6 mm` 개선이라고 해석한다.

## 4. 최종 추론은 CNN branch만 사용

흥미로운 점은 최종 비교에서 Transformer branch를 inference에 쓰지 않는다는 것이다. 논문은 공정 비교를 위해 최종 예측은 학습된 U-Net branch만 사용한다.

이는 중요한 메시지를 준다.

- Transformer는 반드시 deployment model일 필요가 없다.
- training-time complementary teacher로만 써도 충분히 가치가 있다.

저자들은 Transformer branch와 ensemble이 더 좋은 결과를 낼 수도 있지만 계산 비용이 더 크다고 설명한다. Swin-UNet은 약 `27.12M` 파라미터, U-Net은 약 `1.81M` 파라미터로 차이가 크다.

## 논문의 핵심 메시지

### 1. 복잡한 consistency보다 단순한 pseudo supervision이 더 나을 수 있다

이 논문은 semi-supervised segmentation에서 explicit consistency regularization이 꼭 최선은 아니라는 점을 보여준다. 서로 다른 모델이 직접 pseudo label을 주고받는 방식이 더 효과적일 수 있다는 것이다.

### 2. 구조가 다른 네트워크를 함께 학습시키는 것이 중요하다

같은 구조의 두 CNN보다 CNN과 Transformer 조합이 더 좋았다는 결과는, semi-supervised learning에서 diversity가 중요하다는 사실을 다시 보여준다.

### 3. Transformer를 inference보다 training에 쓰는 전략도 가능하다

이 논문은 Transformer를 최종 배포 모델로 삼지 않더라도, CNN 학습을 강화하는 training-time teacher로 활용할 수 있음을 보여준다. 실무적으로 꽤 유용한 관점이다.

## 한계와 저자들이 시사하는 미래 방향

논문 말미와 결과 구성을 보면 한계도 분명하다.

### 1. 단일 데이터셋 검증

실험은 ACDC 하나에 집중돼 있다. 따라서 다른 modality나 다른 구조의 의료영상에서도 같은 효과가 유지되는지는 추가 검증이 필요하다.

### 2. 2D 설정 중심

ACDC를 slice-wise 2D segmentation으로 처리한다. 3D volumetric semi-supervised segmentation에서 cross teaching이 어떻게 작동하는지는 별도 문제다.

### 3. hard pseudo label의 오류 전파 가능성

argmax pseudo label을 직접 쓰기 때문에, 잘못된 예측이 상대 branch로 전파될 위험이 있다. 다만 논문은 서로 다른 구조의 상보성이 이를 완화한다고 본다.

### 4. 계산 비용 증가

최종 추론은 CNN만 써도 되지만, 학습 시에는 CNN과 Transformer를 동시에 돌려야 하므로 training cost는 증가한다.

## 실무적 관점의 해설

### 1. 이 논문은 아이디어가 매우 단순하다

많은 semi-supervised segmentation 논문이 teacher-student, perturbation, uncertainty weighting, consistency branches를 복잡하게 쌓는 반면, 이 논문은 pseudo label을 서로 교차로 주는 것만으로 큰 개선을 얻었다고 주장한다. 그 단순함이 장점이다.

### 2. CPS의 구조 다양화 버전으로 읽을 수 있다

핵심 메커니즘은 CPS와 유사하지만, 두 네트워크를 같은 구조가 아니라 CNN과 Transformer로 바꾼 점이 차별화다. 따라서 이 논문은 `cross pseudo supervision + architectural diversity`로 이해하면 가장 정확하다.

### 3. Semi-supervised medical segmentation에 Transformer를 처음 본격적으로 넣은 초기 사례다

저자들도 이 점을 강조한다. 후속 CNN-ViT co-training 계열 논문들이 이 작업을 자주 출발점으로 삼는 이유다.

### 4. 지금 기준에서는 더 큰 foundation model 흐름 이전의 논문이다

이 논문은 2022년 논문이라, SAM, MedSAM, promptable segmentation, medical foundation model 전개 이전 단계에 있다. 따라서 지금 읽을 때는 `CNN-Transformer 협업형 SSL의 초기 대표작`으로 보는 것이 맞다.

## 후속 연구와의 연결

이 논문 이후 자연스럽게 이어진 방향은 다음과 같다.

- CNN-ViT co-training 고도화
- cross teaching에 uncertainty, confidence filtering, feature perturbation 추가
- MobileNet-MobileViT 같은 경량 구조로 확장
- semi-supervised segmentation에서 foundation model teacher 활용으로 확장

즉, 이 논문은 후속 many-branch consistency model보다 훨씬 단순하지만, 의료영상 semi-supervised segmentation에서 CNN과 Transformer를 함께 쓰는 기본 아이디어를 정립한 작업으로 볼 수 있다.

## 종합 평가

`Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer`는 아이디어가 단순하지만 영향력이 큰 논문이다. CNN과 Transformer가 서로의 pseudo label을 직접 가르치게 함으로써, 적은 라벨 환경에서 두 구조의 상보성을 실제 성능 향상으로 연결했다.

강점은 단순성, 구현 용이성, 그리고 ACDC에서의 분명한 개선이다. 한계는 단일 데이터셋, 2D 설정, 추가 학습 비용이다. 그럼에도 semi-supervised medical segmentation에서 `architectural diversity matters`라는 메시지를 가장 직관적으로 보여준 초기 대표작으로 평가할 수 있다.
