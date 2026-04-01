# EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything

Yunyang Xiong, Bala Varadarajan, Lemeng Wu, Xiaoyu Xiang, Fanyi Xiao, Chenchen Zhu, Xiaoliang Dai, Dilin Wang, Fei Sun, Forrest Iandola, Raghuraman Krishnamoorthi, Vikas Chandra

## 🧩 Problem to Solve

Segment Anything Model (SAM)은 제로샷 전이 학습과 다재다능함에서 인상적인 성능을 보이지만, 초대형 트랜스포머 모델(특히 `ViT-H` 이미지 인코더는 6억 3,200만 개의 파라미터를 가짐)에 기반하고 `SA-1B`와 같은 방대한 데이터셋으로 훈련되어 계산 비용과 메모리 요구량이 매우 높습니다. 이로 인해 SAM은 실시간 애플리케이션 및 광범위한 실제 배포에 한계가 있었습니다. 기존의 경량화된 SAM 모델(`FastSAM`, `MobileSAM` 등)은 복잡성을 줄였지만, 상당한 성능 저하를 감수해야 하는 문제가 있었습니다.

## ✨ Key Contributions

* SAM의 `ViT-H` 이미지 인코더의 특징을 재구성하도록 모델을 훈련시키는 `SAMI (SAM-leveraged masked image pretraining)`라는 새로운 마스크 이미지 사전 훈련 프레임워크를 제안했습니다. 이를 통해 이미지 마스크 사전 훈련 방법의 성능을 크게 향상시켰습니다.
* `SAMI`로 사전 훈련된 백본이 이미지 분류, 객체 탐지, 인스턴스 분할, 의미론적 분할 등 다양한 다운스트림 작업에서 뛰어난 일반화 성능을 보임을 입증했습니다.
* 최첨단 품질-효율성 절충점을 제공하는 경량 SAM 모델인 `EfficientSAMs`를 제시했습니다. 이는 실제 배포를 위한 SAM의 보완적인 버전으로 활용될 수 있습니다.

## 📎 Related Works

* **Segment Anything Model (SAM) [31]:** 프롬프트 기반의 객체 분할을 가능하게 하는 이정표적인 비전 기초 모델입니다. SAM의 높은 연산 비용을 줄이기 위한 노력으로 `FastSAM [68, 71]` (CNN 기반) 및 `MobileSAM [71]` (디커플링 증류) 등이 있습니다.
* **Vision Transformers (ViTs) [19]:** 이미지 애플리케이션에서 인상적인 성능을 달성했으며, `ViT-Small/DeiT-Small [53]` 등 효율적인 `ViT` 아키텍처를 설계하는 연구가 활발합니다.
* **Knowledge Distillation (KD) [27]:** 대규모 교사 모델의 지식을 소규모 학생 모델로 이전하여 성능을 향상시키는 기법입니다. 중간 특징을 활용하는 `FitNet [47]`, 자기 지도 학습을 사용하는 `SSTA [60]`, 디커플링된 `KD [70]` 등이 있습니다.
* **Masked Image Pretraining (MIM):** 마스크된 이미지 패치를 재구성하여 의미 있는 표현을 학습하는 자기 지도 사전 훈련 방법입니다. `BEiT [3]`, `MaskFeat [59]`, `SimMIM [63]` 등 다양한 연구가 있으며, 특히 `MAE [26]`를 기반으로 하여 SAM 특징을 재구성하는 방식을 탐구합니다.

## 🛠️ Methodology

1. **SAMI 사전 훈련 (SAM-Leveraged Masked Image Pretraining)**:
    * **목표**: `SAM ViT-H` 이미지 인코더의 특징을 재구성하도록 경량 `ViT` 이미지 인코더(`ViT-Tiny`, `ViT-Small`, `ViT-Base`)를 사전 훈련합니다.
    * **MAE 프레임워크 활용**: `Masked Autoencoders (MAE)` 프레임워크를 기반으로 합니다. 입력 이미지는 마스크되지 않은 토큰과 마스크된 토큰으로 나뉩니다.
    * **교사-학생 학습**:
        * `SAM ViT-H` (교사 모델)는 원본 이미지로부터 대상 특징 임베딩을 생성합니다.
        * 경량 인코더 (학생 모델)는 마스크되지 않은 토큰만 입력으로 받아 잠재 특징 표현을 추출합니다.
        * 크로스-어텐션 디코더는 인코더의 출력 특징을 앵커로 사용하여 마스크된 토큰의 특징 표현을 재구성합니다.
        * 재구성된 특징은 간단한 선형 투영 헤드를 통해 `SAM` 특징과 차원 불일치를 해결하고 정렬됩니다.
    * **재구성 손실**: `SAM` 이미지 인코더 출력 ($f_{sam}(x)$)과 `MAE`의 선형 투영 헤드 출력 ($f_h(x)$) 사이의 `L2` 노름(`ℓ_2` norm)을 최소화하는 방식으로 훈련됩니다.
        $$ L_{W_e,W_d,W_\theta} = \frac{1}{N} \cdot \sum_{j=1}^{N} ||f_{sam}(x) - f_h(x)||^2 $$
        여기서 $f_h(x) = h_\theta(\phi(g_e(\{x_i\}_{i \in U}) \oplus g_d(\{x_i\}_{i \in M})))$ 이며, $g_e$는 경량 인코더, $g_d$는 디코더, $h_\theta$는 선형 투영 헤드, $\phi$는 특징 재정렬 연산자, $\oplus$는 병합 연산자, $\{x_i\}_{i \in U}$는 마스크되지 않은 토큰, $\{x_i\}_{i \in M}$은 마스크된 토큰을 나타냅니다.
    * 마스크 비율은 75%로 설정되었습니다.
2. **EfficientSAM 미세 조정**:
    * `SAMI` 사전 훈련이 완료되면, 디코더는 폐기됩니다.
    * `SAMI`로 사전 훈련된 경량 인코더를 `SAM` 프레임워크의 이미지 인코더로 사용하고, `SAM`의 기본 마스크 디코더와 결합합니다.
    * 이렇게 구성된 `EfficientSAM` 모델은 `SA-1B` 데이터셋에 대해 'Segment Anything' 작업을 위해 미세 조정됩니다.

## 📊 Results

* **이미지 분류 (ImageNet-1K)**: `SAMI`는 `MAE`, `DMAE`, `iBOT`, `CAE`, `BEiT` 등 다른 마스크 이미지 사전 훈련 방법과 `DeiT`, `SSTA`와 같은 증류 방법보다 지속적으로 우수한 성능을 보였습니다. 특히, `SAMI-B`는 84.8%의 Top-1 정확도를 달성하여 `MAE-B`보다 1.2% 높았으며, 경량 모델(`SAMI-Ti`, `SAMI-S`)에서도 상당한 성능 향상을 보였습니다.
* **객체 탐지 및 인스턴스 분할 (COCO)**: `SAMI`로 사전 훈련된 `ViT` 백본은 `ViTDet` 프레임워크에서 다른 기준 모델보다 우수한 성능을 보였습니다. `SAMI-B`는 `MAE-B` 대비 0.9 `AP_{bbox}` 및 0.6 `AP_{mask}` 향상을, `SAMI-S`는 `DeiT-S` 대비 2.6 `AP_{bbox}` 및 2.3 `AP_{mask}`의 상당한 향상을 달성했습니다.
* **의미론적 분할 (ADE20K)**: `Mask2former` 프레임워크에서 `SAMI`로 사전 훈련된 백본은 `MAE` 사전 훈련 백본 대비 `mIoU`에서 큰 향상(예: `SAMI-B`는 2.5 `mIoU` 증가)을 보였습니다.
* **Segment Anything 작업 (제로샷 인스턴스 분할)**:
  * `EfficientSAMs`는 `FastSAM` 및 `MobileSAM`과 같은 다른 경량 `SAM` 모델을 크게 능가했습니다. 예를 들어, `EfficientSAM-S`는 `FastSAM`보다 `COCO`에서 6.5 `AP`, `LVIS`에서 7.8 `AP` 이상 높은 성능을 보였습니다.
  * `EfficientSAM-Ti`는 `FastSAM`보다 `COCO`에서 4.1 `AP`, `LVIS`에서 5.3 `AP` 더 높았습니다.
  * `EfficientSAM-S`는 `SAM` (6.36억 파라미터) 대비 약 20배 적은 파라미터 수(2,500만 파라미터)와 약 20배 빠른 추론 속도를 가지면서도 `SAM`과의 성능 격차를 약 2 `AP`로 크게 줄였습니다.
* **정성적 평가**: `EfficientSAMs`는 포인트 및 박스 프롬프트 입력, 그리고 전체 객체 분할에서 `SAM`에 필적하는 분할 마스크를 생성하는 능력을 보여주었습니다.

## 🧠 Insights & Discussion

* **SAM 특징 활용의 효과**: `SAM`의 강력한 `ViT-H` 인코더의 잠재 특징을 재구성하는 것은 경량 `ViT` 모델이 다양한 작업에 잘 일반화되는 풍부한 시각적 표현을 학습하는 데 매우 효과적인 지도 신호를 제공합니다. 이는 단순히 픽셀 값을 재구성하는 것보다 우수합니다.
* **광범위한 일반화**: `SAMI`로 사전 훈련된 백본은 이미지 분류, 객체 탐지, 분할 등 다양한 다운스트림 작업에서 일관된 성능 향상을 보여, 학습된 표현의 넓은 적용 가능성을 입증했습니다.
* **품질-효율성 절충**: `EfficientSAMs`는 `SAM`의 고성능과 실제 배포에 필요한 효율성 사이의 격차를 성공적으로 줄였습니다. 이는 제한된 컴퓨팅 자원을 가진 환경에서 `SAM`을 실용적으로 사용할 수 있는 대안을 제공합니다.
* **어블레이션 연구**:
  * **재구성 손실**: `MSE`(`Mean Squared Error`) 손실이 코사인 유사도 손실보다 좋은 성능을 보여, `SAM` 특징의 직접적인 재구성이 더 효과적임을 시사합니다.
  * **크로스-어텐션 디코더**: 인코더의 출력을 앵커 토큰으로 활용하여 마스크된 토큰만 재구성하는 크로스-어텐션 방식이 `MAE`처럼 모든 토큰을 디코더로 처리하는 것보다 효과적이었습니다.
  * **마스크 비율**: `MAE`와 일관되게 높은 마스크 비율(75%)이 좋은 결과를 도출했습니다.
  * **재구성 목표**: `CLIP`과 같은 다른 강력한 인코더의 특징을 재구성하는 것도 효과적이며, 이는 강력한 교사 모델의 지도가 `MIM`의 성능 향상에 핵심적임을 시사합니다.
* **한계 및 향후 연구**: 모델이 때때로 노이즈가 있는 분할을 생성할 수 있다는 한계가 있습니다. 현저한 객체 분할과 같은 'Segment Anything'을 넘어서는 잠재적 응용 가능성도 제시되었습니다.

## 📌 TL;DR

SAM은 강력하지만 높은 연산 비용과 파라미터 크기로 인해 실제 배포에 제약이 있었습니다. 이 논문은 SAM의 `ViT-H` 이미지 인코더의 특징을 재구성하도록 경량 `ViT` 인코더를 사전 훈련하는 `SAMI`라는 새로운 마스크 이미지 사전 훈련 방법을 제안합니다. 이렇게 사전 훈련된 경량 인코더와 `SAM`의 디코더를 결합한 `EfficientSAM`은 `SAM`에 근접한 분할 성능을 유지하면서도 `~20배` 더 빠르고 `~20배` 더 작아 `SAM`의 실용적인 배포 가능성을 크게 향상시킵니다.
