# Amodal Instance Segmentation

- **저자**: Ke Li, Jitendra Malik
- **발표연도**: 2016
- **arXiv**: https://arxiv.org/abs/1604.08202

## 1. 논문 개요

### 논문의 목표

본 논문의 목표는 **amodal instance segmentation**이라는 새로운 컴퓨터 비전 작업을 정의하고, 이에 대한 최초의 알고리즘적 방법을 제안하는 것이다. Amodal instance segmentation이란 객체의 가시적인(visible) 부분뿐만 아니라 가려진(occluded) 부분까지 포함하는 전체 영역을 예측하는 작업을 말한다.

### 연구 문제

기존의 instance segmentation이 객체의 가시적인 부분만을 분할(modal segmentation)하는 반면, 본 논문은 객체가 완전히 보였을 때의 형태, 즉 가려진 부분까지 포함한 전체 형태를 예측하는 것을 목표로 한다. 이 문제는 인간이 자연스럽게 수행하는 **amodal completion** 능력을 컴퓨터 비전 시스템에 구현하는 것에 해당한다.

### 문제의 중요성

Amodal segmentation 시스템은 정교한 **occlusion reasoning(폐색 추론)**을 가능하게 한다. 예를 들어, amodal segmentation mask를 얻으면modal mask와 비교하여 가려진 부분의 존재, 범위, 경계, 영역을 유추할 수 있고, 가려진 객체와 가리고 있는 객체 간의 상대적 깊이 순서도 판별할 수 있다. 이러한 정보는 물체의 실제 크기 추정 등 다양한 하류 응용에 활용될 수 있다.

또한 공개된 amodal segmentation 주석 데이터가 존재하지 않는다는 심각한 데이터 부족 문제를 해결해야 하는 과제가 있다.

## 2. 핵심 아이디어

### 중심적인 직관

본 논문의 핵심洞見은 **occlusion을 제거하는 것(undoing occlusion) 대신, occlusion을 추가하는 것(adding occlusion)이 훨씬 쉽다**는 점이다. Modal mask에서 amodal mask를 계산하는 것은 불가능하지만, 그 반대의 방향은 매우 쉽다. 따라서 기존 modal instance segmentation 주석 데이터를 활용하여 합성 occlusion을 만들어내고, 원본 mask를 그대로 보존하면 이것이 합성 이미지의 amodal mask가 된다.

### 기존 접근 방식과의 차별점

이전까지 amodal 관련 연구는 **amodal bounding box 예측**[Kar et al., 2015]이나 깊이 정보를 활용한 평면 보완[gupta et al., 2013] 등으로 제한적이었다. 일반적인 amodal segmentation 방법론은 존재하지 않았으며, 공개 데이터셋도 없었다. 본 논문은 이러한 한계를 극복하여 **최초의 일반적인 amodal instance segmentation 방법**을 제시한다.

### 합성 데이터 생성 전략

학습 데이터를 생성하기 위해, 먼저前景 객체 인스턴스(메인 객체)가 포함된 무작위 이미지 패치를 생성한다. 그런 다음 다른 이미지에서 무작위로 선택한 객체 인스턴스를 추출하여,它们的 modal segmentation mask를 alpha matte로 사용하여 합성 패치 위에 오버레이한다. 이 과정에서 원본 modal mask는 합성 occlusion의 영향을 받지 않으므로, 이는 합성 이미지에 대한 올바른 amodal mask가 된다.

## 3. 상세 방법 설명

### 전체 파이프라인 개요

제안된 방법의 파이프라인은 다음과 같은 단계로 구성된다:

1. **입력**: Modal bounding box(가시적 객체 경계 상자)와 객체 카테고리
2. **Modal segmentation**: IIS(Iterative Instance Segmentation)를 사용하여 modal segmentation heatmap 예측
3. **Iterative Bounding Box Expansion**: 단계적으로 amodal bounding box를 확장
4. **Amodal mask 예측**: 확장된 bounding box 내 패치에서 amodal segmentation mask 예측

### 합성 데이터 생성 절차

학습 데이터 생성의 구체적인 절차는 다음과 같다:

1. 이미지를 균일하게 샘플링하고, 해당 이미지의 객체 인스턴스 중 하나를 메인 객체로 선택한다.
2. 메인 객체의 bounding box와 각 차원에서 최소 70%重叠하는 무작위 bounding box를 샘플링한다. 샘플링된 박스의 크기는 해당 차원에서 메인 객체 bounding box 길이의 70%에서 200% 사이로 무작위로 선택한다.
3. 패치 내부에 오버레이할 객체 수를 0에서 2 사이의 정수로 무작위 선택한다. 오버레이할 객체를 선택하기 위해 이미지를 샘플링하고, 해당 이미지에서 무작위 객체 인스턴스를 샘플링한다.
4. 각 연산 후 메인 객체의 가시적인 부분 비율이 30% 이하로 떨어지면 해당 연산을 취소하고 다시 시도한다.
5. 메인 객체의 가시적인 부분을 포함하는 bounding box를 찾고, 이를 기준 bounding box와 각 차원에서 최소 75%重叠하면서 크기가 최대 10%까지 다른 무작위 박스를 샘플링한다.

### Target Segmentation Mask 생성

각 합성 패치에 대해, 원본 modal segmentation mask의 해당 부분을 가져와서 다음과 같이 레이블을 할당한다:

- 객체에 속하는 픽셀: **positive**
- 배경에 속하는 픽셀: **negative**
- 다른 객체에 속하는 픽셀: **unknown**

이 레이블 할당 방식은 modal mask로부터 amodal mask에 대해 알고 있는 정보를 반영한다. 즉, 객체의 가시적인 부분은 전체 객체의 일부임은 확실하지만, 다른 객체에 의해 가려졌을 수 있으므로 해당 픽셀은 unknown으로 처리한다.

### Iterative Bounding Box Expansion

추론 시점에서 제안된 알고리즘은 **Iterative Bounding Box Expansion**라는 새로운 전략을 사용하여 amodal segmentation mask와 bounding box를 반복적으로 예측한다.

초기 상태에서 amodal bounding box는 modal bounding box와 동일하게 설정된다. 각 반복에서:

1. Amodal bounding box 내 패치를 convolutional neural network에 입력한다.
2. 네트워크는 확장된 amodal bounding box(원본 영역을 넘어서는 주변 영역 포함)에 대한 amodal segmentation heatmap을 예측한다.
3. 원본 bounding box의 위, 아래, 왼쪽, 오른쪽에 해당하는 영역에서 평균 heat intensity를 계산한다.
4. 특정 방향의 평균 heat intensity가 임계값(threshold, 실험에서는 0.1) 이상이면 해당 방향으로 bounding box를 확장하고, 이를 다음 반복에서 사용한다.
5. 모든 방향의 평균 heat intensity가 임계값 미만이 될 때까지 이 과정을 반복한다.

최종 amodal segmentation mask는 해당 heatmap에서 intensity가 0.7 이상인 모든 픽셀을 채워서 얻는다. Modal segmentation mask는 modal heatmap을 0.8에서 thresholding하여 얻는다.

### 모델 아키텍처

사용되는 convolutional neural network는 IIS에서 사용되는 architecture를 기반으로 하며, Hariharan et al.이 도입한 **hypercolumn architecture**를 따른다. 이 architecture는 미세한 스케일의 low-level 이미지 feature와 거친 스케일의 high-level 이미지 feature를 모두 활용하도록 설계되었다.

사용되는 구체적인 architecture 버전은 VGG 16-layer net을 기반으로 하며, IIS architecture의 변형을 사용한다. 이 architecture는 초기 heatmap 가설을 입력으로 받기 위해 추가적인 category-dependent 채널을 통해 modal segmentation heatmap도 입력으로 받는다.

### 손실 함수 및 학습 절차

모델은 알려진 ground truth 레이블을 가진 모든 픽셀에 대해 pixel-wise negative log likelihood의 합을 손실 함수로 사용한다. 각 패치의 업샘플링 비율에 반비례하는 인스턴스별 가중치를 적용한다.

학습은 다음과 같은 하이퍼파라미터로 진행된다:

- Learning rate: $10^{-5}$
- Weight decay: $10^{-3}$
- Momentum: 0.9
- Batch size: 32
- 총 iteration 수: 50,000

IIS 모델의 가중치에서 시작하여 end-to-end로 stochastic gradient descent로 학습한다.

## 4. 실험 및 결과

### 데이터셋 및 평가 설정

Amodal instance segmentation에 대한 공개 주석 데이터가 없기 때문에, 저자들은 PASCAL VOC 2012 validation set에서 무작위로 선택한 100개의 가려진 객체에 amodal segmentation mask를 직접 주석 달았다. 또한 PASCAL VOC 2012 val set의 occlusion presence 주석을 사용한 간접 평가도 수행했다.

### 정성적 결과

제안된 방법은 PASCAL VOC 2012 val set의 객체에 대해 amodal segmentation mask 예측을 생성했다. 실험 결과, 합성 occlusion으로 학습했음에도 불구하고 실제 occlusion에서 가려진 부분에 대해 그럴듯한 가설을 도출할 수 있음을 보여주었다.

Oclusion을 두 가지 유형으로 분류하여 분석했다:

- **내부 가림(interior occlusion)**: 가리는 객체가 주로 가려진 객체 내부에 포함되는 경우
- **외부 가림(exterior occlusion)**: 가리는 객체의 상당 부분이 가려진 객체 외부에 위치하는 경우

내부 가림의 경우 일반적으로 단일 올바른 해결책이 있고 비교적 쉬우며, 외부 가림의 경우 여러 가지 Equally valid한 확장이 가능하여 더 어렵고 모호하다.

또한 가려지지 않은 객체에 대해서는 amodal 예측이 modal 예측과 유사하거나 더 정확함을 확인했다. 이는 occlusion에 견딜 있도록 학습함으로써, low-level 이미지 패턴의 변화에도 견딜 수 있게 되었기 때문으로 해석된다.

### 간접 평가: 폐색 존재 예측

Modal mask와 amodal mask의 영역 비율을 다음과 같이 계산하여 occlusion 존재 여부를 예측했다:

$$\frac{\text{area}(\text{modal mask} \cap \text{amodal mask})}{\text{area}(\text{amodal mask})}$$

이 비율을 사용하여 occlusion absence를 예측하는 분류기의 precision-recall curve를 분석한 결과, **average precision이 77.17%**로 나타났다. 이는 가려지지 않은 객체의 분포가 높은 area ratio로 치우쳐 있는 반면, 가려진 객체의 분포는 약 0.75 부근에서 피크를 보이는 것으로 나타났다.

### 직접 평가: 분할 성능

100개 주석 객체에 대한 정량 평가 결과는 다음과 같다:

**IoU 기반 분할 성능:**

| 방법      | 50% 정확도 | 70% 정확도 | AUC      |
| --------- | ---------- | ---------- | -------- |
| IIS       | 68.0       | 37.0       | 57.5     |
| 제안 방법 | **80.0**   | **48.0**   | **64.3** |

제안된 방법은 대부분의 객체에서 IIS보다 훨씬 정확한 mask를 생성했으며, 전체적으로 73%의 객체에서 IIS보다 나은 결과를 보였다. 나머지 27%의 객체에서도 성능 저하가 5% 미만이었다.

**Combined Detection and Segmentation (Faster R-CNN 기반):**

| 방법                     | mAP^r at 50% IoU | mAP^r at 70% IoU |
| ------------------------ | ---------------- | ---------------- |
| Faster R-CNN + IIS       | 34.1             | 14.0             |
| Faster R-CNN + 제안 방법 | **45.2**         | **22.6**         |

### 제거 연구(Ablation Study)

제거 연구 결과를 통해 modal segmentation 예측을 입력으로 제공하고, 다양한 occlusion 구성을 동적으로 생성하는 것이 모두 성능 향상에 중요함을 확인했다.

| 변형                         | mAP^r at 50% IoU | mAP^r at 70% IoU |
| ---------------------------- | ---------------- | ---------------- |
| Modal segmentation 예측 없음 | 35.2             | 18.4             |
| 동적 샘플 생성 없음          | 39.8             | 22.7             |
| 둘 다 적용                   | **45.2**         | **22.6**         |

## 5. 강점, 한계

### 강점

**첫 번째 획기적 기여**: 본 논문은 일반적인 amodal instance segmentation을 위한 최초의 알고리즘적 방법론을 제시했다. 이는 컴퓨터 비전 분야의 새로운 연구 방향을 열었다고 평가할 수 있다.

**학습 데이터 의존성 회피**: 기존 modal instance segmentation 주석만으로 amodal 모델을 학습할 수 있다는洞見은 실제 응용에 매우 실용적이다. 공개된 amodal 데이터가 없다는 근본적인 문제를 합성 데이터를 통해 우회했다.

**실제 occlusion 적용 가능**: 합성 occlusion으로 학습했음에도 실제 이미지에서 발생한 실제 occlusion에 대한 amodal mask를 성공적으로 예측했다. 이는 모델의 일반화 능력을 보여준다.

**정량적 개선**: IIS 대비 분할 정확도에서 일관된 개선을 보였으며, 특히 IoU 50%에서 12포인트, IoU 70%에서 11포인트의 mAP^r 향상을 달성했다.

### 한계 및 가정

**ambiguity 문제**: Amodal segmentation은 본질적으로 불명확한 문제이다. 예를 들어, 사람의 하반신이 가려졌을 때 해당 사람이 앉아 있는지 서 있는지에 따라 여러 Equally valid한 가설이 존재할 수 있다. 저자들은 이에 대해 인간이 highly consistent하게 가려진 부분을 예측하지만, 단일 정답이 없음을 인정한다.

**학습 데이터의 비현실성**: 합성 occlusion은 완전히 현실적으로 보이지 않으며, 이것이 모델 성능에 미치는 영향에 대해서는 추가 연구가 필요하다.

**occlusion 유형 제한**: 학습 시 생성하는 합성 occlusion은 실제 세계의 occlusion 다양성을 완전히 반영하지 못할 수 있다.

**정확도 한계**: IoU 70%에서의 정확도가 48%로, 실제 응용에는 추가적인 개선이 필요할 수 있다.

**중간 정도 가림만 지원**: 학습 과정에서 최소 30%의 객체가 가시적이어야 한다는 제약으로 인해, 매우 심한 occlusion을 가진 객체에 대한 성능은 보장되지 않는다.

## 6. 결론

### 주요 기여 사항

본 논문은 **amodal instance segmentation**이라는 새로운 컴퓨터 비전 작업을 정의하고, 이를 위한 최초의 알고리즘적 방법론을 제안했다. 핵심 기여 사항은 다음과 같다:

1. **합성 amodal 데이터 생성 전략**: Modal instance segmentation 주석만으로 amodal 학습 데이터를 생성하는 새로운 방법을 제시했다. Occlusion을 추가한 합성 이미지의 원본 mask가 올바른 amodal mask가 된다는洞見을 활용했다.

2. **Iterative Bounding Box Expansion**: Amodal segmentation heatmap에서 amodal bounding box를 반복적으로 유추하는 새로운 전략을 개발했다.

3. **최초의 amodal 분할 방법**: 공개된 amodal 데이터가 없음에도 불구하고, 합성 데이터로 학습한 모델이 실제 이미지의 가려진 객체에 대한 그럴듯한 amodal mask를 예측할 수 있음을 실증했다.

### 실제 적용 및 향후 연구 가능성

제안된 amodal instance segmentation 방법은 다음과 같은 실제 응용에 활용될 수 있다:

- **로봇 비전**: 물체의 전체 형태를 인식하여 파악(grasp) 계획에 활용
- **자율주행**: 가려진 보행자나 장애물의 전체 범위 추정
- **증강현실**: 물체의 가려진 부분을 가상으로 복원하여 시각적 일관성 유지
- **의료 영상**: 가려진 장기나 병변의 전체 범위 추정

향후 연구 방향으로는:

- 더 다양한 occlusion 패턴을 포함한 대규모 amodal 데이터셋 구축
- 3D amodal segmentation으로의 확장
- 다른 모달리티(깊이, 깊이 등)를 결합한 방법 연구
- 실제 환경에서 직접적인 occlusion reasoning으로의 통합
