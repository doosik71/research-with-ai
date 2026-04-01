# Deep Residual Learning for Instrument Segmentation in Robotic Surgery

Daniil Pakhomov, Vittal Premachandran, Max Allan, Mahdi Azizian and Nassir Navab

## 🧩 Problem to Solve

로봇 보조 최소 침습 수술(RMIS)에서 수술 도구의 자동 분할(segmentation)은 컴퓨터 보조 시스템(감지, 추적, 자세 추정)에 필수적인 핵심 작업입니다. 이 작업은 그림자 및 반사광과 같은 복잡한 조명, 연기 및 혈액과 같은 시각적 가림, 복잡한 배경 질감으로 인해 매우 어렵습니다. 기존 연구는 대부분 도구와 배경을 이진(binary)으로 분할하는 데 초점을 맞췄으며, 도구의 여러 부분을 구분하는 다중 클래스(multi-class) 분할은 제한적이었습니다. 또한, 기존의 완전 합성곱 네트워크(FCN)는 출력 해상도가 감소하는 문제가 있었습니다.

## ✨ Key Contributions

- 딥 잔차 학습(Deep Residual Learning) 및 팽창 합성곱(Dilated Convolutions)과 같은 최신 기술을 활용하여 이진 분할 성능을 향상했습니다.
- 기존 이진 분할 접근 방식을 확장하여 배경 외에 도구의 샤프트(shaft)와 조작기(manipulator)와 같은 다른 부분들을 분할하는 다중 클래스 분할(multi-class segmentation)을 수행했습니다.
- MICCAI Endoscopic Vision Challenge Robotic Instruments 데이터셋에서 이진 분할 작업에 대해 이전 최신 기술 대비 밸런스 정확도(balanced accuracy)에서 4% 개선을 달성하며 최신 성능을 경신했습니다.
- 해당 데이터셋에서 다중 클래스 로봇 도구 분할 결과를 최초로 제시하여 향후 연구를 위한 견고한 기준선을 제공했습니다.
- 실험에 사용된 소스 코드를 공개했습니다.

## 📎 Related Works

- **초기 방법:** 도구의 외형을 변경하여 분할 문제를 단순화하려 했으나, 임상 적용에 어려움이 있었습니다 [15].
- **자연스러운 외형 기반 기계 학습:** 색상 특징에 대해 훈련된 Random Forests [4], 최대 우도 가우시안 혼합 모델(maximum likelihood Gaussian Mixture Models) [12], 나이브 베이즈 분류기(Naive Bayesian classifiers) [13] 등이 사용되었습니다.
- **심층 학습 기반:** Fully Convolutional Networks (FCNs), 특히 FCN-8s 모델 [10]이 로봇 도구의 이진 분할에 최신 성능을 보였습니다 [7].
- **FCN의 해상도 문제 해결:**
  - FCN-8s 모델 [10]의 deconvolutional layer와 "skip architecture"를 사용하는 방법.
  - 일부 특징 맵의 다운샘플링을 피하고(스트라이드를 1로 설정) 팽창 합성곱을 사용하는 방법 [5, 16]. 본 연구는 이 두 번째 방법을 채택했습니다.
- **잔차 네트워크(ResNets):** He et al. [8]은 잔차 유닛을 사용하여 깊은 CNN 모델을 구성하고 이미지 분류 및 분할 작업에서 최첨단 정확도를 달성했습니다.

## 🛠️ Methodology

1. **기본 아키텍처:** 최신 이미지 분류 CNN인 ResNet-101 [8]을 기반으로 합니다.
2. **FCN으로의 변환:**
    - ResNet-101의 최종 평균 풀링(average pooling) 계층을 제거하고 완전 연결(fully connected) 계층을 $1 \times 1$ 합성곱 계층으로 대체하여 FCN으로 변환합니다 [5, 10].
    - 이로써 네트워크는 임의 크기의 입력을 받아 공간 해상도가 감소된 예측 출력을 생성하게 됩니다 (원래 $32 \times$ 다운샘플링).
3. **고해상도 특징 맵 복원:**
    - **다운샘플링 비율 감소:** ResNet-101에서 다운샘플링을 담당하는 마지막 두 합성곱 계층의 스트라이드(stride)를 2에서 1로 변경합니다. 이는 출력 특징 맵의 공간 해상도 감소를 $32 \times$에서 $8 \times$로 줄입니다.
    - **팽창 합성곱(Dilated Convolutions) 적용:** 스트라이드를 변경한 후속 합성곱 계층에 팽창 합성곱을 적용합니다. 팽창 합성곱은 필터의 가중치를 희소하게 적용하여 다운샘플링 없이 수용장(receptive field)을 확장하면서 미리 훈련된 가중치를 재사용할 수 있도록 합니다.
        - 수식: $y[i] = \sum_{k=1}^{K} x[i+rk]w[k]$ ($r$은 팽창 계수)
    - **이중 선형 보간(Bilinear Interpolation):** $8 \times$ 다운샘플링된 출력 맵에 이중 선형 보간을 적용하여 입력 이미지와 동일한 공간 해상도를 가진 최종 분할 맵을 얻습니다.
4. **학습:**
    - **손실 함수:** 정규화된 픽셀 단위 교차 엔트로피 손실(normalized pixel-wise cross-entropy loss) [10]을 최소화합니다.
    - **최적화 알고리즘:** Adam 최적화 알고리즘 [9]을 사용합니다.
    - **학습률:** $10^{-4}$로 설정합니다.
5. **클래스 정의:**
    - **이진 분할:** 도구 대 배경 ($C=2$).
    - **다중 클래스 분할:** 도구의 샤프트(shaft), 도구의 조작기(manipulator), 배경 ($C=3$).

## 📊 Results

- **데이터셋:** MICCAI Endoscopic Vision Challenge’s Robotic Instruments 데이터셋 [1]을 사용하여 평가했습니다.
- **이진 분할:**
  - 이전 최신 기술인 FCN-8s [7]와 비교하여 밸런스 정확도에서 4% 향상을 달성했습니다.
  - 본 연구: 민감도(Sensitivity) 85.7%, 특이도(Specificity) 98.8%, 밸런스 정확도(Balanced Accuracy) 92.3%.
  - FCN-8s [7]: 민감도 87.8%, 특이도 88.7%, 밸런스 정확도 88.3%.
- **다중 클래스 분할:**
  - 해당 데이터셋에서 다중 클래스 분할 결과는 본 연구가 처음입니다.
  - IoU(Intersection Over Union) 지표를 사용했습니다.
  - 조작기(C1), 샤프트(C2), 배경(C3)에 대한 IoU 점수를 6개 비디오에 걸쳐 보고했습니다.
  - 전체 평균 IoU는 비디오에 따라 72.2에서 83.7% 사이로 나타났으며, 배경(C3)에 대한 IoU가 가장 높고 조작기(C1)와 샤프트(C2)가 그 뒤를 이었습니다.

## 🧠 Insights & Discussion

- 본 연구는 로봇 수술 도구 분할의 중요성을 강조하며, 이는 렌더링된 오버레이가 도구를 가리는 것을 방지하거나 도구의 자세를 추정하는 데 활용될 수 있습니다 [2].
- 최신 딥 네트워크(ResNet-101)와 팽창 합성곱의 결합이 이진 도구 분할에서 이전 최신 기술 대비 4% 개선을 가져왔음을 입증했습니다.
- 이진 분할 작업을 다중 클래스 분할 작업으로 확장하고 MICCAI Endoscopic Vision Challenge’s Robotic Instruments 데이터셋에서 이를 처음으로 수행하여, 향후 다중 클래스 분할 연구를 위한 견고한 기준선을 제공했습니다.
- 딥 잔차 네트워크의 사용이 이 작업에 효과적임을 보여주었습니다. 특히, 다중 클래스 분할의 경우 조작기와 같은 복잡한 부분의 분할 정확도 향상이 추가 연구의 방향이 될 수 있습니다.

## 📌 TL;DR

로봇 수술에서 수술 도구의 정확한 분할(이진 및 다중 클래스)은 컴퓨터 지원에 필수적이지만 어려운 문제입니다. 이 논문은 ResNet-101을 FCN으로 개조하고, 다운샘플링 비율을 줄이며 팽창 합성곱을 사용하여 고해상도의 픽셀 단위 예측을 생성하는 방법을 제안합니다. 이 방법은 이진 분할에서 이전 최신 기술보다 밸런스 정확도를 4% 향상했으며, MICCAI 데이터셋에서 도구 부품(샤프트, 조작기)을 구분하는 다중 클래스 분할 결과를 최초로 제공하여 딥 잔차 학습의 효과를 입증했습니다.
