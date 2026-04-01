# RAUNet: Residual Attention U-Net for Semantic Segmentation of Cataract Surgical Instruments

Zhen-Liang Ni, Gui-Bin Bian, Xiao-Hu Zhou, Zeng-Guang Hou, Xiao-Liang Xie, Chen Wang, Yan-Jie Zhou, Rui-Qi Li, and Zhen Li

## 🧩 Problem to Solve

로봇 보조 수술에서 수술 도구의 시맨틱 분할은 매우 중요합니다. 그러나 백내장 수술 도구의 정확한 분할은 다음과 같은 문제로 인해 여전히 어려운 과제입니다:

- **강한 반사 (Specular reflection):** 백내장 수술은 강한 조명 조건에서 이루어져 도구 표면에 심한 반사가 발생하며, 이는 도구의 시각적 특성을 변화시킵니다.
- **클래스 불균형 (Class imbalance):** 백내장 수술 도구는 미세 조작을 위해 작고, 이미지에서 작은 영역만 차지하는 경우가 많아 배경 픽셀이 전경 픽셀보다 훨씬 많습니다. 이로 인해 도구가 배경으로 오분류될 가능성이 높습니다.
- **가려짐 (Occlusion):** 안구 조직과 카메라의 제한된 시야로 인해 수술 도구의 일부가 가려질 수 있습니다.
대부분의 기존 연구는 내시경 수술에 초점을 맞추고 있으며, 백내장 수술에 대한 연구는 부족합니다.

## ✨ Key Contributions

- **증강된 어텐션 모듈 (Augmented Attention Module, AAM) 설계:** 다단계 특징을 효율적으로 융합하고 특징 표현력을 향상시켜 specular reflection 문제를 해결하는 데 기여합니다. 이 모듈은 파라미터 수가 적어 메모리 사용량이 효율적이며, 다른 네트워크에 유연하게 적용할 수 있습니다.
- **하이브리드 손실 함수 (Hybrid Loss Function) 도입:** Cross Entropy와 Logarithm of Dice Loss를 병합한 CEL-Dice 손실 함수를 제안하여 클래스 불균형 문제를 효과적으로 해결합니다. 이는 두 손실 함수의 장점을 모두 활용합니다.
- **Cata7 데이터셋 구축:** 백내장 수술 도구의 시맨틱 분할을 위한 최초의 데이터셋인 Cata7을 구축하여 제안하는 네트워크를 평가했습니다.

## 📎 Related Works

- **수술 도구 분할 연구:**
  - FCN (Fully Convolutional Networks) 및 광학 흐름(optic flow) 기반 네트워크 [3].
  - RASNet: 어텐션 모듈을 사용하여 목표 영역을 강조하고 특징 표현력을 개선 [4].
  - U-형 네트워크로 분할과 자세 추정을 동시에 수행 [5].
  - 순환 신경망(recurrent network)과 컨볼루션 네트워크(convolutional network)를 결합한 방법 [6].
- **의료 영상 분할 연구:**
  - U-Net: 바이오메디컬 영상 분할을 위한 인코더-디코더 구조 [8].
  - TernausNet: ImageNet으로 사전 학습된 VGG11 인코더를 사용하는 U-Net 변형 [10].
  - LinkNet: 효율적인 시맨틱 분할을 위해 인코더 표현을 활용 [12].
- **어텐션 메커니즘:**
  - GAU (Pyramid Attention Network) [11]와 같은 어텐션 모듈이 제안되었으며, 본 논문의 AAM과 성능 비교에 사용되었습니다.

## 🛠️ Methodology

RAUNet (Residual Attention U-Net)은 인코더-디코더 아키텍처를 기반으로 하여 고해상도 마스크를 생성합니다.

- **네트워크 아키텍처:**
  - **인코더 (Encoder):** ImageNet으로 사전 학습된 ResNet34를 사용하여 깊은 시맨틱 특징을 추출하고, 모델 크기 및 추론 속도를 최적화합니다.
  - **디코더 (Decoder):** 증강된 어텐션 모듈 (Augmented Attention Module, AAM)과 전치 컨볼루션(transposed convolution)으로 구성됩니다.
    - **AAM:** 고수준 및 저수준 특징 맵을 융합하여 특징 표현력을 개선합니다.
      - 글로벌 평균 풀링(Global Average Pooling, GAP)을 통해 고수준 특징 맵의 시맨틱 정보와 저수준 특징 맵의 글로벌 컨텍스트를 추출합니다.
      - 추출된 정보를 1$\times$1 컨볼루션, 배치 정규화, ReLU 및 Softmax 함수를 거쳐 어텐션 벡터를 생성합니다.
      - 생성된 어텐션 벡터를 저수준 특징 맵에 곱하여 'attentive feature map'을 만들고, 이를 고수준 특징 맵과 더하여 보정합니다.
      - 이 과정은 기존 스킵 연결(skip connection)의 문제점(저수준 특징의 불필요한 배경 정보)을 해결하고, 파라미터 수를 적게 유지하면서 효율적인 특징 융합을 가능하게 합니다.
    - **전치 컨볼루션 (Transposed Convolution):** 업샘플링을 수행하여 분할 마스크의 가장자리(edge)를 정교하게 복구합니다.

- **손실 함수 (CEL-Dice Loss):**
  - 클래스 불균형 문제를 해결하기 위해 Cross Entropy ($H$)와 Dice Loss ($D$)를 결합한 하이브리드 손실 함수 $L = (1-\alpha)H - \alpha \log(D)$를 사용합니다.
  - Cross Entropy는 픽셀 단위 분류에 주로 사용되는 손실 함수입니다.
  - Dice Loss는 예측과 실제 값 사이의 유사성을 측정하며 클래스 불균형에 덜 민감합니다.
  - $\log(D)$를 사용하면 $D$ 값이 작을 때 (즉, 예측 정확도가 낮을 때) 손실 값이 크게 증가하여 잘못된 예측에 대한 페널티를 강화하고 손실의 민감도를 높입니다.
  - $\alpha$는 두 손실의 균형을 조절하는 가중치로, 실험을 통해 0.2로 설정되었습니다.

- **데이터셋:**
  - 백내장 수술 영상으로 구성된 새로운 Cata7 데이터셋을 구축했습니다.
  - 총 2,500개의 이미지와 10가지 백내장 수술 도구 클래스로 구성됩니다. 각 이미지는 정밀한 가장자리와 도구 유형으로 라벨링되었습니다.
  - 훈련 세트(1,800장)와 테스트 세트(700장)로 분할됩니다.

- **훈련 세부 사항:**
  - 훈련 이미지는 $960 \times 544$ 픽셀로 크기가 조정됩니다.
  - 옵티마이저는 Adam을 사용하고 배치 크기는 8입니다.
  - 초기 학습률은 $4 \times 10^{-5}$이며, 30회 반복마다 0.8배 감소합니다.
  - 데이터 증강(무작위 회전, 이동, 뒤집기)을 통해 과적합을 방지합니다.
  - 배치 정규화는 각 컨볼루션 이후에 적용됩니다.

## 📊 Results

- **AAM 성능 평가:**
  - AAM을 적용하지 않은 기본 네트워크(BaseNet)는 Mean Dice 95.12%, Mean IOU 91.31%를 달성했습니다.
  - AAM을 적용한 RAUNet은 Mean Dice 97.71%, Mean IOU 95.62%를 달성하여, Mean Dice 2.59%p, Mean IOU 4.31%p의 성능 향상을 보였습니다.
  - AAM은 GAU [11]와 비교했을 때 더 높은 성능을 보이면서도 파라미터 증가는 0.26M(기본 네트워크의 1.19%)으로 매우 적었습니다. 이는 GAU의 파라미터 증가량(0.60M)보다 훨씬 적습니다.
  - 시각화 결과, AAM은 기본 네트워크의 분류 오류 및 불완전한 분할 문제를 해결하여 Ground Truth와 동일한 수준의 마스크를 생성했습니다.

- **최첨단(State-of-the-Art) 방법과의 비교:**
  - Cata7 데이터셋에서 RAUNet은 Mean Dice 97.71%, Mean IOU 95.62%를 달성하여 U-Net [8], TernausNet [10], LinkNet [12] 등 다른 방법들을 능가하는 최첨단 성능을 보였습니다.
  - 특히, 샘플 수가 적거나 얇은 도구(예: primary incision knife, lens hook)에 대해서도 각각 100%와 90.23%의 높은 픽셀 정확도를 달성하여 클래스 불균형 문제 해결 능력을 입증했습니다.

- **CEL-Dice Loss 성능 검증:**
  - CEL-Dice는 Cross Entropy 또는 Dice Loss 단독 사용보다 Mean Dice와 Mean IOU 모두에서 유의미하게 더 높은 분할 정확도를 보여주었습니다. 이는 클래스 불균형 문제를 해결하는 데 효과적임을 입증합니다.

## 🧠 Insights & Discussion

RAUNet은 백내장 수술 도구의 시맨틱 분할이라는 어려운 과제에 대한 효과적인 해결책을 제시합니다. 제안된 AAM은 강한 반사로 인한 특징 추출의 어려움을 극복하고, 적은 파라미터로도 효율적인 특징 융합과 표현력 향상을 가능하게 합니다. 또한, CEL-Dice 손실 함수는 수술 도구의 작은 크기로 인한 심각한 클래스 불균형 문제를 성공적으로 해결하여 네트워크가 배경에 치우치지 않고 도구를 정확하게 식별하도록 돕습니다. Cata7 데이터셋은 백내장 수술 도구 분할 연구를 위한 중요한 기반을 마련했으며, RAUNet이 이 데이터셋에서 보여준 최첨단 성능은 로봇 보조 수술, 수술 기술 평가, 워크플로우 최적화 등 다양한 임상 응용 분야에서 큰 잠재력을 가짐을 시사합니다.

## 📌 TL;DR

백내장 수술 도구의 시맨틱 분할은 강한 반사 및 클래스 불균형으로 인해 도전적입니다. 본 논문은 이러한 문제를 해결하기 위해 Residual Attention U-Net (RAUNet)을 제안합니다. RAUNet은 멀티레벨 특징을 효과적으로 융합하고 특징 표현력을 향상시키는 새로운 Augmented Attention Module (AAM)과 Cross Entropy 및 Log Dice 손실을 결합한 하이브리드 CEL-Dice 손실 함수를 포함합니다. 또한, 백내장 수술 도구 분할을 위한 최초의 데이터셋인 Cata7을 구축했습니다. RAUNet은 Cata7 데이터셋에서 Mean Dice 97.71%, Mean IOU 95.62%의 최첨단 성능을 달성하여 AAM과 CEL-Dice Loss의 효과를 입증했습니다.
