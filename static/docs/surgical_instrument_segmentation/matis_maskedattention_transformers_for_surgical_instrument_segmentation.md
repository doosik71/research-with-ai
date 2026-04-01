# MATIS: MASKED-ATTENTION TRANSFORMERS FOR SURGICAL INSTRUMENT SEGMENTATION

Nicolás Ayobi, Alejandra Pérez-Rondón, Santiago Rodríguez, Pablo Arbeláez

## 🧩 Problem to Solve

로봇 보조 수술에서 수술 도구 분할(surgical instrument segmentation)은 수술 장면 이해 및 컴퓨터 보조 시스템 개발에 필수적입니다. 그러나 이 분야에는 두 가지 주요 도전 과제가 있습니다. 첫째, 여러 도구 유형 간의 높은 유사성과 데이터셋의 클래스 불균형으로 인해 수술 도구를 정확하게 식별하고 분할하는 것이 어렵습니다. 둘째, 동영상 내에서 프레임 간의 일관된 인식을 가능하게 하고 수술 절차에 대한 전반적인 이해를 향상시키기 위해 시간적 정보를 효과적으로 통합해야 합니다. 기존 CNN 기반 방법들은 종종 픽셀 단위 분류 방식을 사용하여 공간적 일관성이 부족하고 다중 인스턴스 문제를 간과합니다. 최신 트랜스포머 기반 방법들조차 장기적인 비디오 추론이나 인스턴스를 구분하지 못하는 픽셀 분류 방식을 사용하는 경우가 있었습니다.

## ✨ Key Contributions

* **완전한 트랜스포머 기반 마스크 분류 아키텍처 제안**: 마스크드 어텐션(masked attention) 및 변형 어텐션(deformable attention)을 활용하여 인스턴스 기반 도구 분할을 수행하는 MATIS를 제안했습니다.
* **비디오 트랜스포머 활용**: 장기적인 시간 정보를 완전히 활용하기 위해 비디오 트랜스포머를 도구 분할에 통합했습니다.
* **새로운 최첨단 성능 달성**: Endovis 2017 및 Endovis 2018 데이터셋에서 기존의 모든 도구 유형 분할 방법보다 뛰어난 성능을 달성하여 새로운 최첨단(State-of-the-Art, SOTA)을 수립했습니다.

## 📎 Related Works

* **CNN 기반 방법**: 초기 FCN [5, 6] 및 Endovis 챌린지 [7, 8] 이후 FCN 기반 모델 [7, 8, 9, 10]이 도구 유형 분할에 사용되었습니다. 광학/모션 흐름 [11, 12], 스테레오 정보 [13], saliency 맵 [14] 등의 추가 사전 정보가 활용되거나, 약한 지도 학습 [11, 15], 도메인 적응 [16], 자세 추정 [3], 운동학 데이터 [17, 18], 이미지 생성 [19, 17] 등으로 작업이 확장되었습니다. ISINet [20]은 Mask-RCNN [21]을 사용한 인스턴스 기반 접근 방식을 처음 제시했습니다.
* **트랜스포머 기반 방법**: Vision Transformer (ViT) [23, 24] 이후, [27]은 CNN과 Swin Transformer [23]를 Mask-RCNN의 백본으로 통합했습니다. DETR [25] 및 MaskFormer [26]와 같은 아키텍처는 학습 가능한 객체 쿼리(object queries)를 사용하여 영역 예측(set prediction)을 수행했습니다. TraSeTr [28]은 MaskFormer를 변형하여 도구 분할에서 SOTA를 달성했지만, 장기적인 비디오 추론에는 한계가 있었습니다. Deformable DETR [29] 및 Mask2Former [30]는 변형 어텐션 및 마스크드 어텐션 메커니즘의 잠재력을 보여주었습니다.
* **비디오 트랜스포머**: 비디오 트랜스포머 [31, 32]는 SOTA를 달성했지만, 수술 도구 분할에는 충분히 활용되지 못했습니다. 대부분 광학 흐름 [12, 11, 20]을 사용했습니다. STSwinCL [33]은 Swin Transformer를 비디오 분석에 적용했지만 픽셀 분류 방식이었습니다. TAPIR [34]는 Deformable DETR을 도구 감지에 통합하여 다단계 수술 워크플로우 분석을 위한 완전한 트랜스포머 기반 모델을 제시했으며, MATIS는 TAPIR의 비디오 분석 방법론을 따릅니다.

## 🛠️ Methodology

MATIS는 두 단계의 마스크 분류 접근 방식을 사용합니다.

1. **마스크드 어텐션 베이스라인 (Masked Attention Baseline)**:
    * Mask2Former [30]를 베이스라인으로 사용하며, Swin Transformer [23] 백본을 포함합니다.
    * Mask2Former의 다중 스케일 변형 어텐션 픽셀 디코더($$deformable \ attention \ pixel \ decoder$$)와 마스크드 어텐션 메커니즘을 활용합니다.
    * $N$개의 학습 가능한 쿼리(learnable queries)를 사용하여 $N$개의 클래스 확률-이진 마스크($$class \ probability-binary \ mask$$) 쌍으로 구성된 고정 크기의 집합 $z$를 예측합니다. (MATIS는 $N=100$을 사용합니다.)
    * **추론 과정**: 각 클래스에 대해 가능한 인스턴스 수(사전 지식)에 따라 상위 $k$개의 점수를 가진 영역을 선택합니다 ($$k=2$$는 다중 인스턴스, $$k=1$$은 단일 인스턴스 도구). 클래스 빈도 불균형으로 인한 클래스 간 점수 편차를 해결하기 위해 클래스별 점수 임계값도 적용합니다.

2. **시간 일관성 모듈 (Temporal Consistency Module)**:
    * TAPIR [34]의 비디오 분석 방법론을 따릅니다.
    * 비디오 분석을 위해 Multi-Scale Vision Transformer (MViT) [32]를 백본으로 사용합니다.
    * 키프레임($$keyframe$$)을 중심으로 한 시간 윈도우($$time \ window$$)를 사용하여 중간 프레임의 복잡한 시간적 컨텍스트를 인코딩하는 전역 시공간 특징($$spatio-temporal \ features$$)을 계산합니다.
    * Mask2Former의 세그먼트별 임베딩($$per-segment \ embeddings$$)을 사용하여 TAPIR의 박스 분류 헤드($$box \ classification \ head$$)를 영역 분류 헤드($$region \ classification \ head$$)로 변형합니다.
    * 시간 특징을 시간적으로 풀링(pooling)한 후 다층 퍼셉트론(MLP)을 적용하고, 이를 세그먼트별 임베딩의 선형 변환과 연결하여 각 영역을 선형적으로 분류합니다.
    * **추가 지도 학습**: 풀링된 시간 특징에 또 다른 MLP를 사용하여 중간 프레임에 각 도구가 존재하는지 예측하는 다중 대상 인식 작업을 수행합니다. 이에는 기존 마스크 분류 손실($$mask \ classification \ loss$$)에 이진 교차 엔트로피 손실($$binary \ cross-entropy \ loss$$)이 추가됩니다.

* **구현 세부 사항**: 베이스라인은 MS-COCO [35]로 사전 학습된 Mask2Former를 사용하고, 시간 일관성 모듈은 Kinetics 400 [36]으로 사전 학습된 TAPIR를 사용합니다.

## 📊 Results

* **데이터셋 및 평가 지표**: Endovis 2017 [7] (4-fold 교차 검증) 및 Endovis 2018 [8] (사전 정의된 분할) 데이터셋에서 평가되었으며, Mean Intersection over Union (mIoU), Intersection over Union (IoU), Mean Class Intersection over Union (mcIoU)을 사용합니다.
* **MATIS Frame (베이스라인만)**:
  * Endovis 2017 및 Endovis 2018에서 모든 세 가지 전체 분할 지표에서 이전의 모든 SOTA 방법을 능가합니다 (표 1).
  * 예: Endovis 2017 mIoU 68.79% (TraSeTr 60.40% 대비), Endovis 2018 mIoU 82.37% (TraSeTr 76.20% 대비).
  * 트랜스포머 기반 및 마스크 분류 접근 방식이 CNN 기반 픽셀 분류 방법보다 우수함을 입증합니다.
  * 다중 스케일 변형 어텐션 및 마스크드 어텐션이 도구의 시각적 특징에 대한 더 지역화되고 유연한 이해를 제공하여 더 나은 분할을 가능하게 합니다.
* **상한선(Upper Bounds)**: 높은 Inferred Upper Bound (Endovis 2017에서 83.44% mIoU)와 Total Upper Bound (Endovis 2017에서 90.75% mIoU)를 달성했습니다 (표 2). 이는 마스크 생성의 픽셀 정확도가 매우 높으며, 주요 오류 원인이 마스크 생성보다는 영역 분류에 있음을 시사합니다.
* **MATIS Full (시간 일관성 모듈 포함)**:
  * 모든 전체 평가 지표를 크게 향상시킵니다 (표 1).
  * 예: Endovis 2017 mIoU 71.36% (MATIS Frame 68.79% 대비), Endovis 2018 mIoU 84.26% (MATIS Frame 82.37% 대비).
  * 비디오 처리와 베이스라인의 지역화된 세그먼트 임베딩($$localized \ segment \ embeddings$$)을 결합하여 낮은 클래스 간 변동성($$interclass \ variability$$)으로 인한 여러 오분류를 수정합니다.
* **어블레이션 연구**: 클래스별 영역 선택 및 임계값을 결합한 추론 전략이 최적임을 확인했습니다. 시간 일관성 모듈의 Time MLP ($$pooled \ time \ features$$의 선형 투영)와 Presence Supervision (도구 존재에 대한 다중 레이블 분류) 모두 성능을 향상시키며, 이들을 동시에 적용했을 때 가장 좋은 결과를 얻었습니다 (표 4). 장기적인 시간 정보를 활용하는 것이 유리하지만, 너무 작거나 너무 큰 윈도우 크기는 성능 저하를 가져옵니다.

## 🧠 Insights & Discussion

MATIS는 수술 도구 분할에서 새로운 SOTA를 수립하여 로봇 보조 수술 장면의 더 섬세하고 일관된 이해를 가능하게 합니다. 이 연구는 마스크 분류를 사용하는 트랜스포머 기반 아키텍처가 CNN 기반 픽셀 분류 방법보다 우월함을 보여줍니다. 다중 스케일 변형 어텐션과 마스크드 어텐션이 도구의 시각적 특징을 더 정확하게 포착하는 데 효과적임을 입증했습니다. 또한, 비디오 트랜스포머를 통한 장기적인 비디오 추론이 클래스 간 변동성으로 인한 오분류를 개선하고 시간적 일관성을 높이는 데 결정적인 역할을 함을 강조합니다. 높은 상한선 값은 마스크 생성의 픽셀 정확도가 매우 뛰어나며, 모델 개선의 주요 초점이 영역 분류의 정확도를 높이는 데 있음을 시사합니다. 하지만, 겹치거나 불연속적인 인스턴스 분할에 어려움을 겪고, 시간 일관성 모듈을 포함한 후에도 일부 클래스에서 장기 비디오 추론으로 인한 분류 노이즈로 인해 성능이 감소하는 한계점이 관찰되었습니다.

## 📌 TL;DR

수술 도구 분할의 정확성 및 시간적 일관성 부족 문제를 해결하기 위해, MATIS는 두 단계의 완전 트랜스포머 기반 방법을 제안합니다. 첫 번째 단계는 Mask2Former와 Swin Transformer 백본을 활용한 마스크드 어텐션 베이스라인으로, 변형 및 마스크드 어텐션을 사용하여 인스턴스 마스크를 생성합니다. 두 번째 단계는 TAPIR에서 영감을 받은 MViT 기반 시간 일관성 모듈로, 장기적인 비디오 정보를 활용하여 마스크 분류의 시간적 일관성을 향상시킵니다. 이 방법은 Endovis 2017 및 2018 벤치마크에서 새로운 SOTA를 달성했으며, 베이스라인의 우수한 성능과 시간 일관성 모듈이 분류 정확도를 더욱 높임을 입증했습니다. 마스크 생성의 픽셀 정확도는 매우 높았으나, 분류 정확도 향상이 주요 과제로 남아있습니다.
