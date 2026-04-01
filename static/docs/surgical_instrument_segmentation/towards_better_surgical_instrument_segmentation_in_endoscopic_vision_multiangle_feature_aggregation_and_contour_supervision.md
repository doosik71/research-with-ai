# Towards Better Surgical Instrument Segmentation in Endoscopic Vision: Multi-Angle Feature Aggregation and Contour Supervision

Fangbo Qin, Shan Lin, Yangming Li, Randall A. Bly, Kris S. Moe, Blake Hannaford*, Fellow, IEEE

## 🧩 Problem to Solve

로봇 보조 수술의 내시경 영상에서 수술 도구를 정확하고 실시간으로 분할(segmentation)하는 것은 매우 중요합니다. 하지만 잦은 도구-조직 접촉, 관찰 시점의 지속적인 변화, 심층 신경망(DNN) 모델의 제한적인 회전 불변성(rotation-invariance) 성능, 그리고 도구 경계면 근처에서 발생하는 불완전하고 부정확한 분할 마스크 등으로 인해 많은 어려움이 있습니다. 본 연구는 이러한 문제들을 해결하여 기존 DNN 분할 모델의 정확도를 향상시키는 일반적인 임베딩 가능(embeddable) 접근 방식을 제안합니다.

## ✨ Key Contributions

* **다각도 특징 집합 (Multi-Angle Feature Aggregation, MAFA) 방법론 제안**: 모델 파라미터 수를 늘리지 않고도 DNN 분할 모델에 유연하게 통합될 수 있습니다. 여러 각도로 이미지를 능동적으로 회전시켜 시각적 단서를 풍부하게 만들고, 도구 방향 변화에 대한 예측의 강건성(robustness)과 정확도를 향상시킵니다.
* **윤곽선 감독 (Contour Supervision) 활용**: 학습 단계에서 보조 학습 작업으로 윤곽선 감독을 사용하여 모델이 도구 경계와 밀접하게 관련된 특징을 학습하도록 유도함으로써, 분할 마스크의 윤곽선 모양을 더욱 정밀하게 만듭니다.
* **새로운 Sinus-Surgery 데이터셋 공개**: 외과의의 실제 수술에서 수집된 비강(Sinus) 수술 데이터셋을 공개했습니다. 이 데이터셋은 숙련된 끝단 움직임(dexterous tip motion), 좁은 수술 공간, 근접한 렌즈-객체 거리 등의 특징을 가집니다.

## 📎 Related Works

* **전통적인 수공예 특징 기반 방법**: DNN의 특징 학습 능력과 깊은 계층 구조 덕분에 최근 DNN 모델에 의해 성능이 뛰어넘어졌습니다.
* **DNN 기반 분할 모델**:
  * **ToolNet**: 가벼운 구조로 FCN-8s보다 우수한 실시간 도구 분할 모델.
  * **U-Net**: 생체 의학 영상 분할을 위한 고전적인 대칭 구조 및 스킵 연결(skip connection) 모델.
  * **LinkNet & TernausNet**: Shvets 등은 LinkNet-34 및 TernausNet-16을 의료 도구 분할에 적용했습니다.
  * **Multi-resolution Feature Fusion (MFF)**: Islam 등은 PSPNet 및 ICNet보다 우수한 실시간 다중 해상도 특징 융합 모델을 제안했습니다.
  * **LWANet**: Ni 등은 경량 인코더와 채널-어텐션 디코더를 기반으로 대규모 수술 이미지에서 실시간 분할을 달성했습니다.
  * **광학 흐름(Optical Flow) 기반 방법**: 프레임 간 움직임 정보를 활용하지만, 내시경 영상의 동적인 조명 및 배경 때문에 어려움이 있습니다.

## 🛠️ Methodology

본 연구는 MAFA와 윤곽선 감독을 제안하며, 이들은 DeepLabv3+ 및 TernausNet-16과 같은 기존 심층 분할 모델에 통합될 수 있습니다.

1. **회전-정렬 하의 의미론적 일관성 (Semantic Consistency under Rotation-and-Alignment)**:
    * 이미지 $I$를 각도 $\phi$로 회전하면 $I_{\phi} = Rot_{\phi}(I)$가 됩니다.
    * 회전된 이미지에서 특징 $H_{\phi} = F(I_{\phi})$를 추출한 후, 동일한 각도 $\phi$로 역방향 정렬(alignment)하면 $H_{\phi}^{A} = Rot_{-\phi}(H_{\phi})$를 얻습니다.
    * 원래 이미지에서 추출한 특징 $H = F(I)$와 $H_{\phi}^{A}$는 값은 다르지만 의미론적으로 일관성이 있다고 가정합니다.

2. **다각도 특징 집합 (Multi-Angle Feature Aggregation, MAFA)**:
    * 하나의 입력 이미지 $I$와 여러 회전 각도 $\{\phi_k\}$ ($k=1,2,\dots,N_A$)에 대해 회전-정렬 작업을 통해 증강된 특징들을 생성합니다. 각도들은 $[0^\circ, 360^\circ]$ 범위에 걸쳐 균일한 간격으로 분할됩니다.
    * 공유 인코더(shared encoder)는 $N_A$개의 회전된 이미지를 병렬로 처리하여 특징 맵 $H_{\phi_k}$를 생성합니다.
    * 이 특징 맵들은 다시 원본 각도로 정렬되어 다각도 특징 맵 $H_{\phi_k}^{A}$를 만듭니다.
    * MAFA 블록은 이 다수의 특징 맵들을 평균화하여 하나의 집합된 특징 맵 $H_{MA} = \frac{1}{N_A} \sum_{k=1}^{N_A} H_{\phi_k}^{A}$를 생성합니다.
    * 이 집합된 특징 맵은 디코더(decoder)로 전달됩니다.

3. **윤곽선 감독 및 손실 함수 (Contour Supervision and Loss Function)**:
    * 모델의 최종 출력은 분할 맵 $\Sigma$와 윤곽선 맵 $C$입니다. 두 맵 모두 2개 채널(전경/배경)을 가집니다.
    * 윤곽선 맵의 Ground Truth $G_{C}'$는 분할 맵의 Ground Truth $G_{S}'$에서 전경 영역의 외부 윤곽선을 찾아 생성하며, 윤곽선 너비는 3픽셀로 설정됩니다.
    * **윤곽선 손실**: Dice 손실을 사용하여 윤곽선 예측을 감독합니다 (수학적으로는 윤곽선 확률 채널에 대해서만 합산).
        $$L_C = -2 \frac{\sum_{i,j,k=2} C_{i,j,k} G'_{C,i,j,k}}{\sum_{i,j,k=2} C^2_{i,j,k} + \sum_{i,j,k=2} G_{C,i,j,k}'^2 + \tau}$$
    * **분할 손실**: 표준 교차 엔트로피(cross-entropy) 손실을 사용합니다.
        $$L_S = - \frac{1}{N} \sum_{i,j,k} G'_{S,i,j,k} \log(\Sigma_{i,j,k})$$
    * **총 손실**: 두 손실의 가중치 합으로 정의됩니다.
        $$L = L_S + L_C$$

## 📊 Results

* **Sinus-Surgery 데이터셋 (C 및 L)**:
  * **MAFA의 효과**: MAFA는 mIOU(mean Intersection Over Union)를 Sinus-Surgery-C에서 평균 4.8%, Sinus-Surgery-L에서 7.6% 향상시켰습니다. 또한, MAFA를 사용하지 않았을 때 $|mIOU - mRM_{IOU}|$가 약 10%였던 반면, MAFA 사용 시 2.0% 미만으로 감소하고 mRSD$_{IOU}$가 줄어들어 회전 불변성이 크게 개선되었음을 입증했습니다.
  * **윤곽선 감독의 효과**: 윤곽선 감독은 mIOU를 Sinus-Surgery-C에서 0.5%, Sinus-Surgery-L에서 3.3% 추가적으로 향상시켰습니다. 특히, mIOU$_{NB}$ (경계면 근처 IOU)는 Sinus-Surgery-C에서 1.1%, Sinus-Surgery-L에서 3.4% 향상되어 경계면 정확도가 개선되었음을 보여주었습니다.
  * MAFA와 경량 모델(예: MobileNet)의 조합은 더 많은 레이어와 특징 채널을 가진 무거운 모델보다 더 나은 정확도를 보였고, 처리 속도는 비슷하거나 더 빨랐습니다.

* **EndoVis2017 데이터셋**:
  * DeepLabV3+ (MobileNet)에 MAFA ($N_A=4$)를 적용한 결과, mDSC(mean Dice Similarity Coefficient)가 **93.1%**로, MFF (91.6%), LinkNet (90.6%), ICNet (88.2%), U-Net (87.8%), PSPNet (83.1%) 등 다른 기존 분할 모델들을 뛰어넘는 최고 성능을 달성했습니다.
  * MAFA는 mDSC를 2.7%, mSpec.를 0.3%, mSens.를 2.0% 향상시켰습니다.
  * MAFA는 회전 분산(mRSD$_{DSC}$)을 3.7%에서 2.3%로 줄여 회전 불변성을 향상시켰습니다.
  * 윤곽선 감독은 EndoVis2017 데이터셋에서 이진 분할 성능에 큰 개선을 제공하지 않았는데, 이는 이 데이터셋의 도구-배경 대비가 Sinus-Surgery 데이터셋보다 강하여 윤곽선 인식이 보조 감독 없이도 쉽게 달성될 수 있었기 때문으로 추정됩니다.

## 🧠 Insights & Discussion

* **MAFA의 의미**: MAFA는 입력 이미지를 능동적으로 회전시켜 더 풍부한 시각적 단서를 제공함으로써 도구 방향 변화에 대한 분할 모델의 강건성을 향상시킵니다. 또한, 모델 파라미터 수를 늘리지 않고도 정확도를 개선할 수 있다는 점에서 깊고 넓은 백본(backbone)을 사용하는 것보다 효율적일 수 있습니다.
* **윤곽선 감독의 의미**: 윤곽선 예측 계층은 분할 계층의 바이패스(bypass) 역할을 하며, 동일한 입력 특징을 공유합니다. 따라서 윤곽선 감독 손실이 분할 계층을 통해 직접 역전파되지 않더라도, 모델이 윤곽선 인식 특징(contour-aware features)을 학습하도록 간접적으로 유도하여 분할 정확도, 특히 경계면 근처의 정확도를 향상시킵니다.
* **한계 및 향후 연구**: 윤곽선 감독의 효과는 데이터셋의 특성(예: 대비 강도)에 따라 달라질 수 있습니다. 향후 연구에서는 특징 집합 관점을 넘어 회전 불변성 동작을 더 탐구하고, 다중 프레임 간의 시간적 정보(temporal information)를 활용하여 더욱 강건한 분할을 달성하는 것이 매력적입니다.

## 📌 TL;DR

**문제**: 내시경 영상에서 수술 도구를 정확하고 실시간으로 분할하는 것은 도구의 잦은 회전 및 경계면의 부정확성으로 인해 어렵습니다.
**방법**: 본 연구는 두 가지 임베딩 가능한 방법을 제안합니다. 첫째, **다각도 특징 집합(MAFA)**은 입력 이미지를 여러 각도로 회전시키고, 특징을 추출한 뒤 다시 정렬하여 평균화함으로써 회전 불변성과 정확도를 향상시킵니다. 둘째, **윤곽선 감독(Contour Supervision)**은 Dice 손실을 사용하여 보조 작업으로 윤곽선 예측을 추가하여 경계면 분할의 정밀도를 높입니다. 이 방법들은 기존 모델의 파라미터 수를 늘리지 않습니다.
**결과**: 새로 수집된 Sinus-Surgery 데이터셋과 공개 EndoVis2017 데이터셋을 통한 실험에서 MAFA는 분할 정확도(mIOU +4.8% ~ +7.6%)와 회전 불변성을 크게 향상시켰습니다. 윤곽선 감독은 경계면 분할 정확도(mIOU$_{NB}$ +1.1% ~ +3.4%)를 더욱 높였습니다. 특히 MAFA는 경량 모델이 EndoVis2017 데이터셋에서 기존의 다른 최신 방법들보다 뛰어난 성능을 발휘하도록 했습니다.
