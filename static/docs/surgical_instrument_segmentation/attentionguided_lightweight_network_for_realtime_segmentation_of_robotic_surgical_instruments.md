# Attention-Guided Lightweight Network for Real-Time Segmentation of Robotic Surgical Instruments

Zhen-Liang Ni, Gui-Bin Bian, Zeng-Guang Hou, Xiao-Hu Zhou, Xiao-Liang Xie, and Zhen Li

## 🧩 Problem to Solve

로봇 보조 수술에서 수술 도구의 실시간 분할(segmentation)은 매우 중요하지만, 딥러닝 모델의 높은 계산 비용과 느린 추론 속도로 인해 실시간 적용이 어렵습니다. 또한, 수술 중 발생하는 심한 정반사(specular reflection), 그림자, 도구의 부분적인 노출, 클래스 불균형(class imbalance)과 같은 문제들이 도구의 정확한 위치 파악 및 분할을 더욱 어렵게 만듭니다.

## ✨ Key Contributions

* **실시간 수술 도구 분할을 위한 경량 네트워크 제안:** 어텐션 기반 경량 네트워크(Attention-Guided Lightweight Network, LWANet)를 제안했습니다. 이 네트워크는 작은 모델 크기와 낮은 계산 비용(960x544 입력에 대해 3.39 GFLOPs)으로 39fps의 빠른 추론 속도를 달성하여 실시간 로봇 수술 제어에 적용 가능합니다.
* **어텐션 퓨전 블록(Attention Fusion Block, AFB) 설계:** 채널 간의 의미론적 종속성(semantic dependencies)을 모델링하고 타겟 영역을 강조하기 위해 AFB를 설계했습니다. 이는 정반사 및 그림자 문제 해결에 기여하며 분할 정확도를 향상시킵니다.
* **최첨단 성능 달성:** Cata7 데이터셋에서 94.10%의 평균 IOU(Intersection-Over-Union)로 최첨단 성능을 달성했으며, EndoVis 2017 데이터셋에서는 평균 IOU를 4.10% 증가시켜 새로운 기록을 세웠습니다.

## 📎 Related Works

* **수술 도구의 의미론적 분할:**
  * **Hybrid CNN-RNN [5]:** FCN에 순환 신경망(RNN)을 도입하여 전역 컨텍스트(global contexts)를 포착.
  * **RASNet [6]:** 어텐션 메커니즘을 사용하여 타겟 영역을 강조하고 특징 표현을 개선.
  * **Qin et al. [7]:** CNN 예측과 운동학적 자세 정보(kinematic pose information)를 융합하여 분할 정확도 향상.
  * **Luis et al. [11]:** FCN 및 광학 흐름(optic flow) 기반 네트워크로 폐색(occlusion) 및 변형 문제 해결.
  * **Pakhomov et al. [12]:** Dilated convolution을 사용한 잔차 네트워크(residual network).
  * **한계:** 대부분의 연구는 분할 정확도 향상에 초점을 맞췄고, 실시간 분할은 고려하지 않음.
* **경량 네트워크:**
  * **Light-Weight RefineNet [13]:** RefineNet [14]의 디코더를 수정하여 파라미터 및 FLOPs 감소.
  * **MobileNet [9], MobileNetV2 [8]:** Depthwise separable convolution 및 inverted residual structure를 통해 모델 크기와 계산 비용을 줄이고 성능을 향상.
  * **ShuffleNet [16], SqueezeNet [18], Xception [19] 등:** 다른 경량 네트워크들.
* **어텐션 모듈:**
  * **Squeeze-and-excitation block [22]:** 전역 컨텍스트를 벡터로 압축하여 채널 간 의미론적 종속성 모델링.
  * **Non-local block [23]:** 전역 컨텍스트를 추출하여 수용장(receptive field) 확장.
  * **Dual Attention Network [21]:** 채널 어텐션 모듈과 위치 어텐션 모듈로 구성.

## 🛠️ Methodology

LWANet은 인코더-디코더 아키텍처를 채택하여 고해상도 마스크를 생성하고 상세한 위치 정보를 제공합니다.

* **인코더 (Encoder):**
  * 경량 네트워크인 **MobileNetV2 [8]**를 사용하여 의미론적 특징을 추출합니다. MobileNetV2는 빠른 추론 속도와 강력한 특징 추출 능력을 갖추고 있습니다.
  * 의미론적 분할 작업에 적합하지 않은 마지막 두 레이어(평균 풀링 레이어, 완전 연결 레이어)는 제거됩니다. 인코더의 출력 스케일은 원본 이미지의 $1/32$입니다.
* **경량 어텐션 디코더 (Lightweight Attention Decoder):**
  * 위치 디테일을 복구하기 위해 설계되었으며, 낮은 계산 비용으로 실시간 분할을 가능하게 합니다. LWANet의 최종 출력 스케일은 원본 이미지의 $1/4$입니다.
  * **Depthwise Separable Convolution (DSC) [9]:** 디코더의 기본 단위로 사용됩니다. 표준 컨볼루션을 Depthwise Convolution과 Pointwise Convolution으로 분해하여 계산 비용과 모델 크기를 크게 줄입니다 (커널 크기가 $3 \times 3$일 때 약 9배 감소).
        $$
        \frac{\text{k} \times \text{k} \times \text{d}_1 \times \text{m} \times \text{n} + \text{d}_1 \times \text{d}_2 \times \text{m} \times \text{n}}{\text{k} \times \text{k} \times \text{d}_1 \times \text{d}_2 \times \text{m} \times \text{n}} = \frac{1}{\text{d}_2} + \frac{1}{\text{k}^2}
        $$
  * **어텐션 퓨전 블록 (Attention Fusion Block, AFB):** 고수준 특징 맵과 저수준 특징 맵을 융합합니다.
    * **전역 평균 풀링(Global Average Pooling, GAP)**을 사용하여 전역 컨텍스트를 포착하고 채널 간 의미론적 종속성을 인코딩합니다.
    * **Squeeze-and-excitation 메커니즘 [22]**을 도입하여 채널 어텐션 메커니즘을 구현합니다. 이는 저수준 특징 맵에서 타겟 위치의 디테일을 강조하고, 고수준 특징 맵에서 전역 컨텍스트와 의미론적 정보를 캡처합니다.
    * 어텐션 벡터는 $1 \times 1$ 컨볼루션, ReLU ($\delta_1$), Sigmoid ($\delta_2$) 함수를 통해 생성됩니다:
        $$
        A_c = \delta_2[W_{\beta} \cdot \delta_1[W_{\alpha} \cdot g(x) + b_{\alpha}] + b_{\beta}]
        $$
        여기서 $g(x)$는 전역 평균 풀링을 나타냅니다.
    * 최종적으로 두 개의 어텐션 특징 맵은 덧셈(addition)으로 병합됩니다 (연결(concatenation)보다 파라미터가 적어 계산 비용 절감).
  * **전치 컨볼루션 (Transposed Convolution):** 업샘플링을 수행하여 위치 디테일을 복구하고 흐려진 경계 문제를 해결합니다. 다양한 객체에 적합한 가중치를 학습하여 정교한 엣지 정보를 보존합니다.
* **전이 학습 (Transfer Learning):**
  * 인코더 MobileNetV2는 ImageNet으로 사전 학습되어 객체의 경계, 색상, 질감과 같은 저수준 특징을 학습합니다.
  * 이후 수술 도구 데이터셋으로 네트워크를 미세 조정(fine-tune)하여 고수준 의미론적 특징을 캡처하고 네트워크 성능을 향상하며 수렴을 가속화합니다.
* **손실 함수 (Loss Function):**
  * 수술 도구 분할의 심각한 클래스 불균형 문제를 해결하기 위해 **Focal Loss [24]**를 사용합니다.
  * 쉬운 샘플의 가중치를 줄이고, 훈련 중 어려운 샘플에 모델이 더 집중하도록 합니다.
    $$
    FL(p_t) = -(1-p_t)^{\gamma} \log(p_t)
    $$
    여기서 $\gamma$는 샘플의 가중치를 조절하는 데 사용되며, 본 논문에서는 $6$으로 설정됩니다.

## 📊 Results

* **Cata7 데이터셋 (960x544 입력):**
  * **성능:** 96.91% mean Dice, 94.10% mean IOU를 달성하여 최첨단 성능을 기록했습니다. MobileV2-RefineNet [15]보다 mean Dice 0.58%, mean IOU 1.03% 향상되었습니다.
  * **모델 크기:** 2.06M 파라미터로 MobileV2-RefineNet보다 약 31.56% 작습니다.
  * **추론 속도:** 39.49 fps (이미지당 26ms 이내)로 실시간 수술 영상(30 fps)보다 빠릅니다. MobileV2-RefineNet [15] (25 fps)보다 약 14 fps 빠릅니다.
  * **계산 비용:** 3.39 GFLOPs로 MobileV2-RefineNet [15] (16.62 GFLOPs)의 4.9배 낮습니다. 디코더의 FLOPs는 전체의 8.26%에 불과합니다.
  * **AFB 효과:** AFB 사용 시 mean Dice는 1.11%, mean IOU는 1.92% 증가하여 분할 정확도 향상에 기여함을 입증했습니다.
  * **전이 학습 효과:** 전이 학습을 통해 mean Dice는 5.27%, mean IOU는 7.90% 증가했습니다.
* **EndoVis 2017 데이터셋 (640x544 입력):**
  * **성능:** 58.30% mean IOU를 달성하여 새로운 기록을 세웠고, TernausNet [26]보다 4.10% mean IOU 향상되었습니다.
  * **추론 속도:** 640x512 입력 시 약 42 fps를 달성하여 수술 영상 프레임 속도보다 훨씬 빠릅니다. 입력 크기가 작아질수록 추론 속도는 증가하고 계산 비용은 감소합니다.
* **시각화 결과:** LWANet은 정반사 및 그림자와 같은 어려운 조건에서도 다른 방법에 비해 훨씬 정확한 분할 마스크를 생성했습니다.

## 🧠 Insights & Discussion

* **실용적 가치:** LWANet은 수술 도구의 실시간 분할을 위한 강력하고 효율적인 솔루션을 제공하여 로봇 보조 수술 및 컴퓨터 지원 수술의 안전성을 높이고 의사의 부담을 줄이는 데 기여합니다.
* **경량 설계의 효율성:** MobileNetV2 인코더와 Depthwise Separable Convolution, Attention Fusion Block으로 구성된 경량 디코더의 조합은 정확도를 유지하면서도 계산 비용과 모델 크기를 획기적으로 줄였습니다. 이는 엣지 디바이스(edge devices)에서의 실시간 배포 가능성을 높입니다.
* **어텐션 메커니즘의 중요성:** Attention Fusion Block은 전역 컨텍스트와 채널별 의미론적 종속성을 효과적으로 모델링하여, 정반사나 그림자와 같은 도전적인 환경에서 타겟 영역에 네트워크가 집중하게 함으로써 분할 정확도를 크게 향상시킵니다.
* **전치 컨볼루션의 역할:** 업샘플링 과정에서 섬세한 엣지 정보를 보존하는 데 중요한 역할을 하여 정밀한 도구 위치 파악에 필수적입니다.
* **전이 학습 및 Focal Loss의 효과:** ImageNet을 통한 사전 학습은 일반적인 저수준 특징 학습에 효과적이었고, Focal Loss는 의료 영상 분할에서 흔히 발생하는 클래스 불균형 문제를 성공적으로 해결했습니다.

## 📌 TL;DR

**문제:** 로봇 보조 수술에서 수술 도구의 실시간 분할은 필수적이지만, 기존 딥러닝 모델은 높은 계산 비용과 정반사/그림자 같은 문제로 실시간 적용이 어려웠습니다.
**방법:** 본 논문은 경량 인코더(MobileNetV2)와 커스텀 경량 어텐션 디코더로 구성된 LWANet을 제안합니다. 디코더는 Depthwise Separable Convolution, 채널별 의미론적 어텐션을 위한 Attention Fusion Block, 그리고 정교한 업샘플링을 위한 Transposed Convolution을 포함합니다. 전이 학습과 Focal Loss도 활용됩니다.
**결과:** LWANet은 최첨단 정확도(Cata7에서 94.10% mean IOU, EndoVis 2017에서 58.30% mean IOU)를 달성함과 동시에, 3.39 GFLOPs의 낮은 계산 비용과 약 39fps의 실시간 추론 속도를 보여 로봇 보조 수술에 실질적인 적용 가능성을 제시합니다.
