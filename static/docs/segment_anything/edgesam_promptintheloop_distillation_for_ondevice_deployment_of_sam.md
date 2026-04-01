# EdgeSAM: Prompt-In-the-Loop Distillation for On-Device Deployment of SAM

Chong Zhou, Xiangtai Li, Chen Change Loy, Bo Dai

## 🧩 Problem to Solve

Segment Anything Model (SAM)은 이미지 분할에 강력한 성능을 보여주지만, 거대한 파라미터 수(641M)와 높은 연산량(2735 GFLOPs)으로 인해 스마트폰과 같은 엣지 디바이스에서 실시간 구동이 불가능합니다. NVIDIA 2080 Ti에서도 초당 4장의 이미지밖에 처리하지 못합니다. MobileSAM이나 EfficientSAM과 같은 기존 경량화 시도는 속도를 높였지만, 상당한 성능 저하가 발생했으며 엣지 디바이스에서는 여전히 실시간 처리가 어렵습니다(iPhone 14에서 MobileSAM은 5 FPS). 특히, 단순한 인코더 증류 방식만으로는 SAM의 모든 지식을 포착하기 어렵고, SA-1B 데이터셋으로 학습된 SAM은 단일 포인트와 같은 모호한 프롬프트에 대해 원하는 수준의 세분성(granularity)으로 마스크를 생성하지 못하는 문제가 있습니다.

## ✨ Key Contributions

* **EdgeSAM**을 제안하여, 원본 SAM의 성능을 최소한으로 희생하면서 엣지 디바이스에서 효율적으로 실행되도록 최적화했습니다.
* **프롬프트-인-더-루프 지식 증류(Prompt-In-the-Loop Knowledge Distillation)** 방식을 도입했습니다. 이는 증류 과정에 프롬프트 인코더와 마스크 디코더를 포함시키고, 잘못 세그먼트된 영역에서 박스 및 포인트 프롬프트를 반복적으로 샘플링하여 사용자 입력과 마스크 생성 간의 복잡한 역학을 정확하게 포착하도록 돕습니다.
* 순수 **CNN 기반 아키텍처**가 엣지 디바이스 배포에 더 적합함을 입증했습니다. 특히 Apple Neural Engine(ANE)과 같은 온-디바이스 AI 가속기가 CNN에 최적화되어 있음을 강조합니다.
* 포인트 프롬프트 증류에서 발생하는 데이터셋 편향 문제를 완화하기 위해 인코더 내에 **경량 모듈**을 통합하여 테스트 셋의 세분성 우선순위(granularity priors)를 명시적으로 학습하도록 했습니다.
* 결과적으로 원본 SAM 대비 **37배 빠른 속도**를 달성했으며, 엣지 디바이스에서 MobileSAM/EfficientSAM보다 **7배 이상 빨라졌습니다**. COCO와 LVIS에서 mIoU를 각각 2.3/1.5 및 3.1/1.6 향상시켰습니다.
* iPhone 14에서 **30 FPS 이상으로 실행될 수 있는 최초의 SAM 변형**입니다.

## 📎 Related Works

* **Efficient Model Design**: 경량 CNN (MobileNet, ShuffleNet, GhostNet), 트랜스포머 (MobileViT, EdgeViTs), 및 하이브리드 네트워크 (Mobile-Former, Next-ViT) 설계 연구와 관련이 있으며, EdgeSAM도 효율적인 이미지 인코더를 채택합니다.
* **Knowledge Distillation in Detection and Segmentation**:
  * 분류 작업에 집중했던 기존 지식 증류(Hinton et al.).
  * 밀집 예측(dense prediction) 작업(semantic segmentation, object detection)에 적용된 연구들 (Structured KD [32], Channel-wise KD [44]).
  * 쿼리 기반 디텍터(query-based detectors)를 위한 특화된 지식 증류 손실 (DETRDistill, D3ETR, Teach-DETR).
  * **MobileSAM [61]**: SAM 인코더와 경량 백본 간의 픽셀 단위 특징 증류를 구현했으나, 프롬프트 인코더와 마스크 디코더를 다루지 않아 성능 차이가 컸습니다.
  * **FastSAM [68]**: SA-1B 데이터셋으로 YOLACT 기반 모델을 학습했지만, SAM 원리와는 거리가 있었습니다.
  * **EfficientSAM [57]**: 동시 연구로, 마스킹된 이미지 사전 학습을 통해 성능-속도 트레이드오프를 달성했으나, 학습 비용이 크고 MobileSAM보다 빠르지 않습니다.
* **Efficient Segmentation Models**: 특정 도메인 내의 폐쇄형 세그멘테이션(close-set segmentation)에 중점을 두었으며, 일부는 온-디바이스 구현을 목표로 했으나, 온-디바이스 인터랙티브 세그멘테이션은 아직 미개척 분야입니다.

## 🛠️ Methodology

EdgeSAM은 SAM의 인코더-디코더 아키텍처를 유지하며, 주로 세 가지 단계로 학습됩니다.

1. **인코더-온리 지식 증류 (Encoder-Only Knowledge Distillation)**:
    * SAM의 ViT 기반 이미지 인코더를 효율적인 CNN 기반 네트워크($S_{enc}$)로 증류합니다.
    * 입력 이미지 $I$에 대해 SAM 인코더($T_{enc}$)와 EdgeSAM 인코더($S_{enc}$)의 출력 특징 간의 평균 제곱 오차(MSE) 손실 $L_p = \text{MSE}(T_{enc}(I), S_{enc}(I))$를 사용합니다.
    * MobileSAM과 달리 다운샘플링 레이어를 유지하고, 채널 정렬을 위해 경량 FPN을 사용하여 특징 해상도를 맞춥니다.
    * 순수 CNN 기반의 RepViT-M1 백본이 엣지 디바이스에 가장 적합함을 발견하여 채택했습니다.

2. **프롬프트-인-더-루프 지식 증류 (Prompt-In-the-Loop Knowledge Distillation)**:
    * 원본 SAM의 경량 마스크 디코더 아키텍처를 그대로 사용하고 사전 학습된 가중치를 상속받습니다.
    * 학생 모델($S_{dec}$)을 교사 모델($T_{dec}$)의 마스크 출력으로 감독하며, Dice loss와 BCE loss의 조합인 $L_{mask}$를 마스크 손실로 사용합니다. 디코더 손실은 $L_d = L_{mask}(\phi(T_{dec}(f_t,p,m,c)), S_{dec}(f_s,p,m,c))$로 정의됩니다. 여기서 $f_t, f_s$는 교사와 학생 인코더의 특징, $p, m, c$는 프롬프트, 마스크 토큰, IoU 토큰입니다.
    * **동적 프롬프트 샘플링 전략**을 도입합니다. 초기 프롬프트(박스 또는 포인트)로 시작하여 교사와 학생 모델의 마스크 예측이 불일치하는 영역(오탐 또는 미탐)을 식별합니다. 이 불일치 영역에서 새로운 긍정/부정 포인트 프롬프트를 반복적으로 샘플링하고 기존 프롬프트와 결합하여 다음 디코딩 반복에 사용합니다. 이는 학생 모델이 부정확한 영역에 집중하고 교사 모델이 더 높은 품질의 마스크를 생성하도록 유도합니다.

3. **세분성 우선순위 모듈 (Granularity Priors Module) (선택 사항)**:
    * 모호한 포인트 프롬프트에 대한 SAM의 세분성 결정 문제를 해결합니다.
    * 프리징된 이미지 인코더 위에 경량 Region Proposal Network (RPN)를 구축합니다.
    * RPN은 COCO와 같은 특정 데이터셋에 대해 학습되어 해당 데이터셋의 세분성 우선순위를 포착합니다.
    * 추론 시, 포인트 프롬프트에 가장 가까운 K개의 제안 박스를 신뢰도 점수에 따라 가중치를 부여하여 병합합니다. 이 병합된 박스는 포인트 입력과 함께 마스크 디코더의 프롬프트로 사용됩니다.

**학습 파이프라인**: 3단계로 진행됩니다.

* 1단계: 1% SA-1B 이미지로 인코더-온리 지식 증류.
* 2단계: 동일한 1% SA-1B 이미지로 프롬프트-인-더-루프 증류 (인코더는 1단계 가중치 로드, 디코더는 SAM 가중치 상속).
* 3단계 (선택): 경량 RPN을 COCO와 같은 데이터셋으로 추가 학습 (다른 모듈은 프리징).

## 📊 Results

* **효율성**:
  * NVIDIA 2080 Ti에서 원본 SAM 대비 **37배 빠른** 164.3 FPS 달성.
  * iPhone 14에서 MobileSAM 및 EfficientSAM 대비 **7배 이상 빠른** 38.7 FPS 달성 (iPhone 14에서 30 FPS 이상 실행되는 최초의 SAM 변형).
  * 낮은 GFLOPs (9.6 GFLOPs)와 파라미터 수 (22.1M).
* **정확도 (mIoU)**:
  * **GT 박스 프롬프트**: SA-1K, COCO, LVIS 데이터셋 전반에 걸쳐 MobileSAM 및 EfficientSAM을 꾸준히 능가합니다. COCO 데이터셋에서 1-2개의 추가 정제 포인트 사용 시 원본 SAM을 능가하는 경우도 있습니다.
  * **중심점 프롬프트**: MobileSAM 및 EfficientSAM보다 대부분의 경우에서 우수합니다. RPN 적용 시 COCO의 단일 포인트 mIoU는 48.0에서 54.3으로 향상됩니다.
  * **외부 객체 탐지기 프롬프트**: MobileSAM을 상당한 차이로 능가하며, EfficientSAM과도 비등한 성능을 보입니다. 원본 SAM보다는 여전히 성능 차이가 있습니다.
* **어블레이션 스터디**:
  * **프롬프트-KD 효과**: 인코더-온리 KD 대비 성능을 크게 향상시킵니다.
  * **RPN 트레이드오프**: 합리적인 계산 비용으로 단일 포인트 성능을 효과적으로 개선합니다.
  * **백본 선택**: 순수 CNN 기반 RepViT-M1이 FPN과의 조합으로 최상의 속도-성능 균형을 제공하며, 특히 엣지 디바이스 가속기에서 강점을 보입니다.
  * **프롬프트-KD 일반화**: 제안된 KD 방식은 다양한 백본(TinyViT, EfficientViT, RepViT)에 걸쳐 성능을 일관되고 유의미하게 향상시킵니다.
  * **학습 데이터**: 더 많은 학습 데이터를 사용하면 EdgeSAM의 성능이 더욱 향상되어 EfficientSAM을 능가할 수 있음을 보였습니다.
  * 디코더 프리징보다는 인코더와 디코더 모두 미세 조정하는 것이 더 나은 일반화 성능을 보여주며, 프롬프트 샘플링 루프는 1회가 적절하고, 2단계 순차 학습이 공동 학습보다 더 나은 결과를 보입니다.

## 🧠 Insights & Discussion

* 이 연구의 핵심 통찰은 SAM의 기능을 경량 모델로 증류하는 데 있어 **프롬프트 인식 지식 증류(prompt-aware knowledge distillation)**가 단순한 인코더 특징 증류보다 훨씬 중요합니다. 프롬프트 인코더와 마스크 디코더를 증류 루프에 직접 포함하고 오류 영역에서 동적으로 프롬프트를 샘플링하는 방식은 SAM의 복잡한 대화형 분할 지식을 효과적으로 전이시킵니다.
* 현재 **엣지 디바이스 AI 가속기**는 CNN에 최적화되어 있으므로, RepViT-M1과 같은 CNN 기반 백본이 대규모 ViT 모델의 강력한 성능에도 불구하고 실시간 온-디바이스 배포에 더 적합합니다.
* SA-1B에서 다른 데이터셋으로 SAM을 증류할 때, 모호한 프롬프트(예: 단일 포인트)에 대한 **세분성 모호성**이 중요한 과제입니다. 경량 RPN 모듈을 통해 이 간극을 효과적으로 메울 수 있습니다.
* **한계점**: EdgeSAM은 매우 효율적이지만, 외부 객체 탐지기 프롬프트 사용 시 원본 SAM 대비 여전히 성능 차이가 있습니다. 이는 모델 용량의 내재적 한계 또는 GT 박스로만 학습된 결과일 수 있습니다.
* **향후 연구**: 양자화(quantization), 모델 가지치기(pruning), 온-디바이스 최적화, 혼합 정밀도 추론(mixed-precision inference), 데이터/프롬프트 증강 등을 통해 추가 성능 향상이 기대됩니다. EdgeSAM은 온-디바이스 비디오 편집 및 비디오 인스턴스 세그멘테이션과 같은 실시간 응용 분야의 가능성을 열어줍니다.

## 📌 TL;DR

* **문제**: SAM은 강력하지만 엣지 디바이스에서 너무 느립니다. 기존 경량화 모델은 성능 저하가 크고 엣지에서 여전히 느리며, 단순 인코더 증류는 SAM의 지식을 완전히 포착하지 못합니다.
* **제안 방법**: **EdgeSAM**은 SAM을 경량 CNN 기반으로 증류합니다. 핵심은 인코더/디코더를 모두 포함하고 잘못된 영역에서 프롬프트를 반복 샘플링하는 **프롬프트-인-더-루프 지식 증류**입니다. 또한, 경량 RPN 모듈을 추가하여 포인트 프롬프트의 세분성 문제를 해결합니다.
* **주요 결과**: EdgeSAM은 원본 SAM보다 **37배 빠르고**, iPhone 14에서 MobileSAM/EfficientSAM보다 **7배 이상 빠르며** (iPhone 14에서 30 FPS 이상 실행되는 최초의 SAM 변형), 동시에 높은 mIoU 성능을 유지합니다.
