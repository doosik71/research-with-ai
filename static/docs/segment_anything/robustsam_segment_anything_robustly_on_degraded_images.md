# RobustSAM: Segment Anything Robustly on Degraded Images

Wei-Ting Chen, Yu-Jiet Vong, Sy-Yen Kuo, Sizhuo Ma, Jian Wang

## 🧩 Problem to Solve

Segment Anything Model (SAM)은 뛰어난 zero-shot 분할 능력과 유연한 프롬프트 시스템을 갖추고 있지만, 이미지 품질이 저하될 경우 성능이 현저히 감소하는 한계가 있습니다. 기존의 이미지 복원 기술을 SAM의 전처리 단계로 활용하는 것은 분할 성능 향상을 보장하지 않으며, SAM 모델을 직접 파인튜닝하는 방식은 모델의 zero-shot 일반화 능력을 손상시키거나 '치명적인 망각(catastrophic forgetting)'을 유발할 수 있습니다.

## ✨ Key Contributions

* **RobustSAM 제안**: 다양한 이미지 품질 저하에 강인한 zero-shot 분할 모델인 RobustSAM을 제안합니다. 이 모델은 다운스트림 애플리케이션의 성능을 크게 향상시키는 것으로 입증되었습니다.
* **Robust-Seg 데이터셋 구축**: 68만 8천 개의 이미지-마스크 쌍으로 구성된 대규모 데이터셋인 Robust-Seg를 구축했습니다. 이 데이터셋은 다양한 유형의 합성 품질 저하를 포함하며, 품질 저하 이미지 분할 모델의 새로운 벤치마크를 제공합니다.
* 최소한의 추가 파라미터와 계산 요구 사항으로 SAM의 사전 학습된 지식을 효과적으로 활용하여 효율적인 학습을 가능하게 합니다.
* SAM 기반 다운스트림 작업(예: 단일 이미지 디헤이징 및 디블러링)의 성능을 효과적으로 개선합니다.

## 📎 Related Works

* **Segment Anything Model (SAM)**: SAM [32]은 탁월한 zero-shot 전이 학습 능력으로 다양한 컴퓨터 비전 하위 도메인에서 활용되지만, 이미지 품질 저하 [26, 62, 65, 75] 시 성능이 저하되는 문제점이 지적되었습니다.
* **Robust Segmentation**: 자율 주행 및 감시 분석 분야에서 CNN 기반 분할 성능이 품질 저하 이미지에서 감소하는 문제를 다루는 연구들 (예: QualNet [31], URIE [68], FIFO [38])이 있습니다. 그러나 이들은 주로 단일 유형의 품질 저하에 초점을 맞추거나 SAM의 zero-shot 이점을 희석시킬 수 있습니다.
* **Image Restoration**: SRCNN [14]과 같이 단일 유형의 품질 저하(예: 슈퍼 레졸루션, 디노이징, 디헤이징)를 목표로 하는 방법과 MPRNet [91], HINet [7], IPT [4], AirNet [40]과 같이 다중 품질 저하를 다루는 방법들이 있습니다. 이들은 주로 인간의 시각적 품질 향상에 중점을 둡니다.

## 🛠️ Methodology

RobustSAM은 SAM의 zero-shot 학습 능력을 보존하면서 이미지 품질 저하 문제를 해결합니다.

* **핵심 모듈**:
  * **Anti-Degradation Output Token Generation (AOTG) 모듈**: Robust Output Token (ROT)을 정제하여 품질 저하 관련 정보를 제거합니다. Instance Normalization (IN)과 MLP 레이어를 활용하여 경량화 및 효율성을 확보합니다.
  * **Anti-Degradation Mask Feature Generation (AMFG) 모듈**: 이미지 인코더에서 추출된 마스크 특징($F_{MFD}$) 및 보조 특징($F_{CFD}$)을 정제하여 품질 저하 관련 정보를 제거합니다.
    * **Instance Normalization (IN)** 및 **Batch Normalization (BN)**을 병렬로 적용하여 품질 저하로 인한 변화를 표준화하고 세부 정보를 보존합니다.
    * 어텐션 메커니즘을 통해 IN 및 BN 특징을 융합합니다.
    * **Fourier Degradation Suppression 모듈**: 특징을 주파수 도메인으로 변환하여 진폭 컴포넌트에서 스타일 정보를 포착하고, 1x1 컨볼루션을 통해 품질 저하 요소를 제거하며, 위상 컴포넌트는 구조적 무결성 유지를 위해 보존합니다. 이후 역 푸리에 변환으로 공간 도메인으로 복원합니다.
* **학습 과정**:
    1. 원본 선명한 이미지에 15가지 유형의 합성 품질 저하(예: 블러, 노이즈, 저조도, 악천후 등)를 적용하여 손상된 이미지를 생성합니다.
    2. 손상된 이미지는 RobustSAM을 통해 처리됩니다. 이 과정에서 파인튜닝된 Robust Output Token (ROT)과 함께 Image Encoder의 특징이 활용됩니다.
    3. AOTG는 $T_{RO}$를 $\hat{T}_{RO}$로, AMFG는 $F_{MFD}$ 및 $F_{CFD}$를 $\hat{F}_{MFD}$ 및 $\hat{F}_{CFD}$로 정제합니다.
    4. 원본 선명한 이미지는 SAM을 통해 처리되어 선명한 특징($F_{CFC}$, $F_{MFC}$, $T_{OC}$)을 추출합니다.
    5. **일관성 손실 (Consistency Losses)**: 정제된 손상 이미지 특징과 선명 이미지 특징 간의 일관성을 강화하는 손실 함수를 적용합니다.
        * **Mask Feature Consistency Loss ($L_{MFC}$)**: $$L_{MFC} = ||\hat{F}_{CFD} - F_{CFC}||^2 + ||\hat{F}_{MFD} - F_{MFC}||^2$$
        * **Token Consistency Loss ($L_{TC}$)**: $$L_{TC} = ||\hat{T}_{RO} - T_{OC}||^2$$
    6. **세그멘테이션 손실 ($L_{Seg}$)**: 예측된 마스크와 Ground Truth 마스크 간의 손실을 계산합니다 (Dice Loss + Focal Loss).
    7. **총 손실 ($L_{Overall}$)**: $$L_{Overall} = L_{MFC} + \lambda_1 L_{TC} + \lambda_2 L_{Seg}$$
    8. RobustSAM 학습 시, SAM의 사전 학습된 파라미터는 고정하고 제안된 모듈만 최적화합니다.
* **추론 (Inference)**: 학습된 RobustSAM (AOTG, AMFG, ROT 포함)만 사용하여 입력 이미지로부터 직접 분할 마스크를 생성합니다.
* **Robust-Seg 데이터셋**: LVIS, ThinObject-5k, MSRA10K 등 7개 기존 데이터셋의 43,000개 이미지에 15가지 합성 품질 저하를 적용하여 총 688,000개의 이미지-마스크 쌍을 구축했습니다.

## 📊 Results

* **성능 비교**: RobustSAM은 MSRA10K, LVIS (학습 데이터셋), NDD20, STREETS, FSS-1000, COCO (zero-shot 합성 품질 저하 데이터셋), BDD-100k, LIS (zero-shot 실세계 품질 저하 데이터셋) 등 다양한 데이터셋에서 기존 SAM, HQ-SAM, AirNet+SAM, URIE+SAM을 상회하는 우수한 성능을 보였습니다. 특히 IoU 및 Pixel Accuracy (PA)와 같은 주요 지표에서 모든 시나리오에서 가장 좋은 결과를 달성했습니다 (Table 2, 3, 4, 5, 6).
* **정성적 분석**: 그림 4에서 RobustSAM은 손상된 이미지에서 더 정확한 경계와 온전한 구조를 가진 분할 마스크를 생성하는 반면, SAM은 오류와 파편화된 결과를 보였습니다.
* **다운스트림 작업 개선**: RobustSAM을 사전 정보(prior)로 활용했을 때, SAM 기반 단일 이미지 디헤이징 [28] 및 디블러링 [44] 작업의 PSNR 및 SSIM이 크게 향상되었습니다 (Table 8, Figure 5).
* **효율성**: RobustSAM은 SAM에 비해 추가 파라미터가 적고 (403MB vs 1250MB), 8개의 A100 GPU로 약 30시간 만에 학습이 완료되어 높은 효율성을 입증했습니다 (Table 1).
* **Ablation Study**: 제안된 AMFG, Fourier Degradation Suppression, AOTG, Robust Output Token (ROT) 등 각 모듈이 RobustSAM의 전체 성능 향상에 긍정적인 영향을 미침을 확인했습니다 (Table 7). SAM 전체 또는 디코더를 파인튜닝하는 것은 zero-shot 성능을 크게 저하시켰습니다.
* **다양한 ViT 백본에서의 일관된 우수성**: ViT-B, ViT-L, ViT-H 등 다양한 Vision Transformer 백본에 걸쳐 SAM 대비 일관되게 우수한 성능을 보였습니다 (Supplemental Table S.2).

## 🧠 Insights & Discussion

* **영향**: RobustSAM은 SAM의 주요 한계점이었던 이미지 품질 저하에 대한 취약성을 효과적으로 해결하여 로봇 공학, AR/VR, 콘텐츠 제작 등 다양한 실제 응용 분야에서 SAM의 활용성을 크게 확장합니다.
* **Robust-Seg 데이터셋의 중요성**: 광범위한 합성 품질 저하를 포함하는 대규모 Robust-Seg 데이터셋은 품질 저하 이미지 분할 연구를 위한 새로운 벤치마크를 제공하며, 향후 연구의 귀중한 자원이 될 것입니다.
* **Zero-shot 일반화 유지**: SAM의 핵심 강점인 zero-shot 일반화 능력을 손실 없이 품질 저하에 대한 견고성을 확보했다는 점은 모델의 실용성을 높입니다.
* **다운스트림 개선**: SAM 기반 복원 작업에 RobustSAM을 prior로 활용하여 실제 적용 가능성을 높였으며, 이는 SAM 마스크의 품질이 이러한 작업에 얼마나 중요한지를 보여줍니다.

## 📌 TL;DR

RobustSAM은 품질 저하 이미지에서의 Segment Anything Model (SAM)의 성능 문제를 해결하기 위해 제안된 zero-shot 분할 모델입니다. Anti-Degradation Output Token Generation (AOTG) 및 Anti-Degradation Mask Feature Generation (AMFG) 모듈을 통해 품질 저하에 불변하는 특징을 추출하고, 원본 SAM 특징과의 일관성 손실을 포함한 복합 손실 함수로 학습됩니다. Robust-Seg 데이터셋을 구축하여 학습 및 평가한 결과, RobustSAM은 합성 및 실세계 품질 저하 환경에서 기존 SAM 및 다른 방법들보다 뛰어난 zero-shot 분할 성능을 보였으며, SAM 기반 다운스트림 작업의 효율성도 향상시켰습니다.
