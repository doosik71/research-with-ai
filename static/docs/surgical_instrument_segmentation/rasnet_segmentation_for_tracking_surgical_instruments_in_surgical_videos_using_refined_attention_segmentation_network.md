# RASNet: Segmentation for Tracking Surgical Instruments in Surgical Videos Using Refined Attention Segmentation Network

Zhen-Liang Ni, Gui-Bin Bian, Xiao-Liang Xie, Zeng-Guang Hou, Xiao-Hu Zhou and Yan-Jie Zhou

## 🧩 Problem to Solve

로봇 보조 수술에서 수술 도구의 정확한 추적을 위해서는 정밀한 공간 정보를 캡처하는 것이 필수적입니다. 기존의 많은 수술 도구 추적 및 감지 방법들은 바운딩 박스(bounding box)만을 제공하여 수술 도구의 정확한 경계를 파악하는 데 한계가 있었습니다. 수술 도구 분할(segmentation)은 다음의 과제에 직면합니다: 좁은 시야, 도구 자세 변화로 인한 다양한 기하학적 형태, 그리고 전경 픽셀이 배경 픽셀보다 훨씬 적은 심각한 클래스 불균형(class imbalance) 문제입니다.

## ✨ Key Contributions

* 수술 도구를 동시에 분할하고 그 유형을 식별하기 위한 새로운 네트워크인 Refined Attention Segmentation Network (RASNet)을 제안합니다.
* 네트워크가 핵심 영역에 집중하도록 돕는 어텐션 모듈(Attention Fusion Module, AFM)을 도입하여 분할 정확도를 향상시킵니다.
* 클래스 불균형 문제를 해결하고 작은 객체에 대한 분할 성능을 개선하기 위해 교차 엔트로피 손실(cross entropy loss)과 Jaccard 지수(Jaccard index)의 로그값을 가중합한 손실 함수를 사용합니다.
* ImageNet으로 사전 훈련된 ResNet-50을 인코더로 활용하는 전이 학습(transfer learning) 전략을 채택하여 분할 정확도를 크게 향상시킵니다.
* MICCAI EndoVis Challenge 2017 데이터셋에서 평균 Dice 94.65%, 평균 IOU 90.33%로 최첨단 성능(state-of-the-art)을 달성합니다.

## 📎 Related Works

* **수술 도구 추적 및 감지**: Bareum et al. [4]은 YOLO를 수정하여 실시간 추적을 수행했으며, Amy et al. [5]은 Faster R-CNN을 사용하여 분류 및 바운딩 박스 회귀를 수행했습니다. Duygu et al. [1]은 RPN(Region Proposal Network)과 투 스트림(two-stream) CNN을 병합하여 수술 도구를 감지했습니다.
* **정밀한 경계 획득 (분할)**: Iro et al. [6]은 CSL 모델이라는 새로운 CNN을 제안하여 수술 도구의 분할과 자세 추정을 동시에 수행했으며, 이는 본 논문의 영감이 되었습니다.
* **의료 영상 분할 네트워크**: U-Net [10]은 저수준 특징의 세부 정보와 고수준 특징의 의미 정보를 효과적으로 활용하는 U-자형 네트워크이며, TernausNet [11]은 ImageNet으로 사전 훈련된 VGG11 인코더를 사용하는 U-Net의 변형입니다.

## 🛠️ Methodology

RASNet은 U-자형 네트워크 구조를 가지며, 심층 의미 특징을 캡처하는 수축 경로(contracting path)와 정밀한 위치 파악을 위한 확장 경로(expanding path)로 구성됩니다.

* **네트워크 아키텍처**:
  * **인코더**: ImageNet으로 사전 훈련된 ResNet-50 [12]을 사용합니다. 7x7 컨볼루션 레이어(스트라이드 2)와 3x3 최대 풀링 레이어(스트라이드 2)로 시작하며, 이후 4개의 잔차 블록(residual block)으로 구성된 인코더 블록들을 포함합니다.
  * **디코더**: 역컨볼루션(deconvolution)을 사용하여 특징 맵을 업샘플링하고, 스키프 연결(skip connections)을 통해 고수준 및 저수준 특징 맵을 병합합니다.
* **어텐션 퓨전 모듈 (Attention Fusion Module, AFM)**:
  * U-Net의 단순한 특징 연결 방식과 달리, AFM은 고수준 특징의 전역 문맥(global context)을 활용하여 저수준 특징이 정밀한 위치 정보를 선택하도록 안내합니다.
  * 고수준 특징에서 전역 평균 풀링을 통해 전역 문맥 벡터를 추출하고, 1x1 컨볼루션과 배치 정규화(batch normalization)로 가중치를 정규화합니다.
  * Softmax 함수를 사용하여 가중치 합이 1이 되도록 만든 후, 이 가중치 벡터를 저수준 특징에 곱하고 이를 고수준 특징에 더하여 다단계 특징을 효율적으로 융합합니다.
* **디코더 블록**: 1x1 컨볼루션으로 특징 맵의 차원을 줄여 계산 복잡성을 낮추고, 4x4 역컨볼루션(스트라이드 2)으로 업샘플링하며, 마지막으로 1x1 컨볼루션으로 특징 맵의 차원을 조정합니다.
* **손실 함수 (Loss Function)**:
  * 클래스 불균형 문제를 해결하고 작은 객체 분할 성능을 개선하기 위해 교차 엔트로피 손실($H$)과 Jaccard 지수($J$)의 로그값의 가중합($L$)을 사용합니다.
  * $$H = -\frac{1}{w \times h} \sum_{k=1}^{c} \sum_{i=1}^{w} \sum_{j=1}^{h} y_{{ijk}} \log\left(\frac{e^{\hat{y}_{{ijk}}}}{\sum_{k=1}^{c} e^{\hat{y}_{{ijk}}}}\right)$$
  * $$J = \frac{TP}{TP+FP+FN}$$
  * $$L = H - \alpha \log(J)$$
  * 여기서 $w, h$는 예측 이미지의 너비와 높이, $c$는 클래스 수, $y_{{ijk}}$는 픽셀의 실제값, $\hat{y}_{{ijk}}$는 픽셀의 예측값, $TP, FP, FN$은 각각 True Positives, False Positives, False Negatives를 나타내며, $\alpha$는 가중치입니다.

## 📊 Results

* **데이터셋**: MICCAI EndoVis Challenge 2017 훈련 세트(1800 프레임) 중 1400 프레임을 훈련용으로, 400 프레임을 테스트용으로 사용했습니다. 이미지는 320x256 픽셀로 크기가 조정되었습니다.
* **평가 지표**: IOU(Intersection Over Union)와 Dice ratio를 사용했습니다.
* **사전 훈련의 효과**: 무작위 초기화된 RASNet은 평균 Dice 78.11%, 평균 IOU 70.72%를 달성한 반면, ImageNet으로 사전 훈련된 RASNet은 평균 Dice 94.65%, 평균 IOU 90.33%를 달성하여 사전 훈련이 분할 정확도를 크게 향상시킴을 입증했습니다.
* **AFM의 효과**: AFM이 없는 RASNet은 평균 Dice 89.31%, 평균 IOU 82.75%를 기록했지만, AFM을 사용한 RASNet은 94.65% Dice와 90.33% IOU를 달성하여 AFM이 네트워크 성능을 Dice 5.34%, IOU 7.58% 향상시켰습니다.
* **다른 네트워크와의 비교**: RASNet은 U-Net(70.04% Dice, 56.76% IOU) 및 TernausNet(88.04% Dice, 80.34% IOU)보다 훨씬 우수한 최첨단 성능을 달성했습니다.
* **클래스별 성능**: Curved Scissors (99.01% Dice)가 가장 높은 성능을 보였고, Prograsp Forceps (84.59% Dice)와 Grasping Retractor (88.05% Dice)는 상대적으로 낮은 성능을 기록했습니다.

## 🧠 Insights & Discussion

* **RASNet의 우수한 성능 요인**:
    1. ImageNet으로 사전 훈련된 ResNet-50을 인코더로 사용했습니다.
    2. Attention Fusion Module (AFM)이 고수준 특징의 전역 문맥을 활용하여 저수준 특징의 정밀한 위치 정보를 안내했습니다.
    3. 클래스 불균형 문제를 해결하기 위해 교차 엔트로피 손실과 Jaccard 지수 로그의 조합을 손실 함수로 사용했습니다.
* **성능 제한 및 분석**: 특정 수술 도구(예: Prograsp Forceps)는 다른 도구(예: Bipolar Forceps)와 유사하여 오분류되는 경향이 있었습니다. 또한, Grasping Retractor와 같이 데이터셋 내 샘플 수가 적은 도구는 네트워크의 과소적합(underfitting)으로 인해 성능이 저하되었습니다.
* **향후 연구**: 미래에는 더욱 효과적인 어텐션 모듈을 설계하여 네트워크 성능을 지속적으로 개선할 계획입니다.

## 📌 TL;DR

본 논문은 로봇 보조 수술 영상에서 수술 도구를 정밀하게 분할하고 분류하기 위한 Refined Attention Segmentation Network (RASNet)을 제안합니다. RASNet은 ImageNet으로 사전 훈련된 ResNet-50을 인코더로 사용하며, 고수준 특징의 전역 문맥을 활용하여 저수준 특징의 위치 정보를 안내하는 Attention Fusion Module (AFM)을 통합합니다. 또한, 클래스 불균형 문제를 완화하고 분할 성능을 높이기 위해 교차 엔트로피 손실과 Jaccard 지수의 로그값을 결합한 손실 함수를 사용합니다. MICCAI EndoVis Challenge 2017 데이터셋에서 RASNet은 AFM의 효과와 사전 훈련의 중요성을 입증하며, 기존 U-Net 및 TernausNet을 크게 능가하는 최첨단 분할 성능(평균 Dice 94.65%)을 달성했습니다.
