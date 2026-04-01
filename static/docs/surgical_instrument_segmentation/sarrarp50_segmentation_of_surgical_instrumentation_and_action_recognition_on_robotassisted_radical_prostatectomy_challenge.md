# SAR-RARP50: Segmentation of surgical instrumentation and Action Recognition on Robot-Assisted Radical Prostatectomy Challenge

Dimitrios Psychogyios, Emanuele Colleoni, Beatrice Van Amsterdam, Chih-Yang Li, Shu-Yu Huang, Yuchong Li, Fucang Jia, Baosheng Zou, Guotai Wang, Yang Liu, Maxence Boels, Jiayu Huo, Rachel Sparks, Prokar Dasgupta, Alejandro Granados, Sébastien Ourselin, Mengya Xu, An Wang, Yanan Wu, Long Bai, Hongliang Ren, Atsushi Yamada, Yuriko Harai, Yuto Ishikawa, Kazuyuki Hayashi, Jente Simoens, Pieter DeBacker, Francesco Cisternino, Gabriele Furnari, Alex Mottrie, Federica Ferraguti, Satoshi Kondo, Satoshi Kasai, Kousuke Hirasawa, Soohee Kim, Seung Hyun Lee, Kyu Eun Lee, Hyoun-Joong Kong, Kui Fu, Chao Li, Shan An, Stefanie Krell, Sebastian Bodenstedt, Nicolas Ayobi, Alejandra Perez, Santiago Rodriguez, Juanita Puentes, Pablo Arbelaez, Omid Mohareri, Danail Stoyanov

## 🧩 Problem to Solve

수술 도구 분할(surgical tool segmentation) 및 동작 인식(action recognition)은 수술 기술 평가부터 의사 결정 지원 시스템에 이르기까지 컴퓨터 보조 개입(computer-assisted intervention) 애플리케이션의 핵심 구성 요소입니다. 현재 학습 기반 접근 방식은 고전적인 방법을 능가하지만, 대규모의 주석 처리된 데이터셋에 의존합니다. 또한, 동작 인식 및 도구 분할 알고리즘은 잠재적인 교차 작업 관계를 활용하지 않고 서로 독립적으로 훈련되고 예측을 수행하는 경우가 많습니다. 실제 수술 시나리오의 복잡성을 포착하지 못하는 소규모 데이터셋과 훈련 데이터와 실제 수술 비디오 간의 도메인 간극(domain gap)이 주요한 문제입니다.

## ✨ Key Contributions

* **SAR-RARP50 데이터셋 공개**: 로봇 보조 근치적 전립선 절제술(RARP)의 봉합 비디오 세그먼트 50개를 포함하는 최초의 다중 모드(multimodal), 공개, 생체 내(in-vivo) 수술 동작 인식 및 기구 의미 분할(semantic instrumentation segmentation) 데이터셋을 공개했습니다.
* **단일 작업 및 다중 작업 학습 탐구 촉진**: 제공된 데이터셋을 활용하여 수술 영역에서 강력하고 정확한 단일 작업(single-task) 동작 인식 및 도구 분할 접근 방식을 개발하도록 연구를 장려했습니다.
* **다중 작업 학습의 잠재력 평가**: 다중 작업 학습(multitask learning) 접근 방식의 잠재력을 탐색하고 단일 작업 방식에 대한 비교 우위를 파악했습니다.

## 📎 Related Works

* **수술 동작 인식**: 복잡하고 가변적인 수술 활동으로 인해 어려운 문제로, 주로 계층적 시간(Lea et al., 2016) 또는 그래프(Kadkhodamohammadi et al., 2022) 컨볼루션, 순환 모듈(Jin et al., 2017), 어텐션 메커니즘(Gao et al., 2021; Nwoye et al., 2022)을 사용하는 딥 뉴럴 네트워크 기반 방법이 발전해왔습니다. 수술 도구 궤적 추정(Qin et al., 2020) 또는 수술 기술 평가(Wang et al., 2021b)와 같은 보조 작업을 사용한 다중 작업 학습도 성능 향상 가능성을 보여주었습니다.
* **수술 기구 분할**: 오랫동안 U-Net 기반 아키텍처(Ronneberger et al., 2015)와 사전 훈련된 ResNet 백본을 활용하는 FCNN(Fully Convolutional Neural Networks)이 지배적이었습니다. 최근에는 트랜스포머(Vaswani et al., 2017; Zhao et al., 2022; Shamshad et al., 2023)가 FCNN을 능가하는 결과를 보이며 새로운 기준을 제시했습니다. 시간 정보를 추가하거나(Jin et al., 2019; Kanakatte et al., 2020) 생성 및 적대적으로 훈련된 모델(Colleoni et al., 2022; Kalia et al., 2021; Sahu et al., 2021)도 연구되었습니다.

## 🛠️ Methodology

SAR-RARP50 챌린지는 세 가지 주요 작업을 제안합니다.

1. **작업 1: 동작 인식 (Action Recognition)**
    * **목표**: 실제 수술 시연을 미세한 시간 세그먼트로 분해하고 미리 정의된 동작 클래스로 분류합니다. 복잡한 실제 수술 비디오에 대한 정확한 솔루션을 찾는 것이 목표입니다.
    * **평가 지표**:
        * **프레임별 정확도(Frame-wise accuracy, FWA)**:
            $$FWA_{i} = \frac{\text{#correctly classified frames}}{\text{# frames in the video i}}$$
        * **세그먼트별 F1@K (Segmental F1@K)**: $K=10$으로 설정되며, 예측된 세그먼트와 실제 세그먼트 간의 IoU 오버랩이 임계값($T=K/100$) 이상일 때 TP로 간주하여 F1 점수를 계산합니다.
            $$segmentalF1@K = \frac{2 \times (\text{precision} \times \text{recall})}{(\text{precision} + \text{recall})}$$
        * 최종 점수: $Score_{ar} = \sqrt{FWA_{avg} \ast F1@10_{avg}}$

2. **작업 2: 수술 기구 의미 분할 (Surgical Instrumentation Semantic Segmentation)**
    * **목표**: RGB 이미지에서 픽셀 수준으로 수술 도구(바늘, 봉합사, 클립, 흡입 도구, 바늘 홀더, 카테터 등)의 의미 레이블을 할당하여 이미지 마스크를 생성합니다. 까다로운 조명 조건, 카메라 초점, 혈액 가림이 있는 데이터에서 모델의 성능을 조사하는 것이 목표입니다.
    * **주석 프로토콜**: 9가지 의미 클래스를 포함하며, 픽셀이 하나의 클래스에만 속하도록 정의되고, 불완전하게 보이는 도구 부분도 일관된 맥락을 유지하도록 규칙이 적용됩니다.
    * **평가 지표**:
        * **평균 IoU (mean Intersection over Union, mIoU)**: 픽셀 수준에서 예측 마스크와 실제 마스크 간의 오버랩을 측정합니다.
            $$IoU_{ijk} = \frac{GT_{ijk} \cap Prediction_{ijk}}{GT_{ijk} \cup Prediction_{ijk}}$$
            $$mIoU = \frac{1}{I} \sum_{i=1}^{I} \left( \frac{1}{J} \sum_{j=1}^{J} \left( \frac{1}{K} \sum_{k=1}^{K} IoU_{ijk} \right) \right)$$
        * **평균 정규화 표면 주사위(mean Normalized Surface Dice, mNSD)**: 예측된 경계 픽셀과 실제 마스크의 가장 가까운 경계 픽셀 사이의 거리가 주어진 임계값($\tau$)보다 짧은 경우를 계산하여 예측 품질을 가중치 있게 평가합니다.
            $$NSD_{ijk} = \frac{\parallel\delta_{Pred_{ijk}}\parallel + \parallel\delta_{Target_{ijk}}\parallel}{\parallel\Delta_{Target_{ijk}}\parallel + \parallel\Delta_{Target_{ijk}}\parallel}$$
            $$mNSD = \frac{1}{I} \sum_{i=1}^{I} \left( \frac{1}{J} \sum_{j=1}^{J} \left( \frac{1}{K} \sum_{k=1}^{K} NSD_{ijk} \right) \right)$$
        * 최종 점수: $Score_{s} = \sqrt{mIoU \ast mNSD}$

3. **작업 3: 다중 작업 (Multitask)**
    * **목표**: 수술 비디오만을 입력으로 사용하여 수술 동작 레이블과 수술 기구 분할 마스크를 동시에 예측합니다. 단일 작업 방식 대비 다중 모드 정보 통합의 이점을 탐구하고, 네트워크 구성 요소를 공유하여 더 빠른 추론을 가능하게 하는 것이 목표입니다.
    * **평가 지표**: $Score_{mt} = \sqrt{Score_{ar} \ast Score_{s}}$

## 📊 Results

* **동작 인식**:
  * **최고 성능**: SummerLab-AI 팀이 최종 점수 0.828로 선두를 차지했습니다. Uniandes (0.804), CAMI-SIAT (0.788)이 뒤를 이었습니다.
  * **접근 방식**: 대부분의 상위 팀은 특징 추출기(feature extractor)와 장거리 시간 모델(long-range temporal model)을 결합한 2단계 접근 방식을 사용했으며, 이는 긴 시간 시퀀스를 한 번에 처리하는 데 효과적이었습니다. 어텐션(attention) 기반 모델이 탁월한 성능을 보였습니다.
  * **사전 훈련**: Kinetics400과 같은 대규모 비디오 동작 데이터셋으로 사전 훈련된 접근 방식이 그렇지 않은 경우보다 훨씬 우수했습니다.
  * **사후 처리**: Uniandes 팀은 고전적인 윈도우 기반 필터링(window-based filtering)이 가장 효과적임을 발견했습니다.
  * **영향 요인**: 숙련된 외과 의사의 수술 비디오는 높은 인식 점수를 얻은 반면, 초급 레지던트의 비디오(더 많은 출혈, 카메라 움직임, 불규칙한 동작 시퀀스)는 낮은 점수를 기록했습니다. 이는 모델 강건성(robustness) 향상을 위해 다양한 난이도의 수술 비디오 통합의 중요성을 시사합니다.

* **수술 기구 의미 분할**:
  * **최고 성능**: Uniandes 팀이 최종 점수 0.847로 1위를 차지했으며, HiLab-2022 (0.840), SummerLab-AI (0.839)가 근소한 차이로 뒤를 이었습니다.
  * **접근 방식**: 상위 3개 팀 모두 어텐션(attention) 기반 모델을 사용했습니다.
  * **테스트 시간 증강(TTA)**: 절반의 제출물에서 TTA가 예측 성능을 효과적으로 향상시키는 것으로 나타났습니다. (예: AIA-Noobs의 CNN 기반 아키텍처가 TTA를 활용하여 일부 트랜스포머 기반 모델을 능가함).
  * **사전 훈련**: ImageNet으로 사전 훈련된 네트워크를 활용했으며, 특히 Uniandes는 수술 도구 데이터셋으로 사전 훈련하여 상위권에 랭크되었습니다.
  * **정성적 평가**: 대부분의 상위 8개 모델은 혈액으로 가려진 시나리오에서도 정확한 마스크를 생성했지만, 클립 홀더와 같은 작고 빈도가 낮은 도구의 클래스 분류에는 어려움을 겪는 경우가 있었습니다. 일부 방법에서 클램프를 바늘로 잘못 분류하는 편향도 발견되었습니다.

* **다중 작업**:
  * **최고 성능**: Uniandes 팀이 최종 점수 0.824로 1위를 차지했으며, AIA-Noobs (0.706), SummerLab-AI (0.625)가 뒤를 이었습니다.
  * **시간 정보**: 상위 3개 팀은 동작 인식 예측에 시간 정보를 사용한 반면, Team-SK는 단일 프레임 예측에 의존하여 F1@10 점수에서 상당한 성능 차이를 보였습니다.
  * **샘플링 속도**: Uniandes와 AIA-Noobs는 동작 인식 샘플을 10Hz로 활용하여 다른 두 팀보다 우수한 동작 인식 점수를 얻었습니다.
  * **단일 작업 vs. 다중 작업**: Uniandes의 다중 작업 접근 방식은 단일 작업 제출물에 비해 분할 점수에서 약간 높았지만, 동작 인식에서는 약간 낮았습니다. 전반적으로 단일 작업과 다중 작업 접근 방식의 성능은 거의 동일했습니다.
  * **결론**: 다중 작업 학습 시나리오에서 SAR-RARP50의 다중 모드 특성을 활용하여 모델 정확도를 향상시킬 수 있는지 여부는 제출된 솔루션을 기반으로 결론 내리기 어려웠습니다.

## 🧠 Insights & Discussion

* **어텐션 메커니즘의 효과**: 동작 인식 및 기구 분할 모두에서 어텐션 기반 모델, 특히 트랜스포머 아키텍처가 뛰어난 성능을 보였습니다.
* **데이터 다양성의 중요성**: 덜 숙련된 외과 의사나 도전적인 수술 비디오를 포함하는 것이 모델의 강건성을 향상시키는 데 중요합니다. 실제 수술 비디오와 수술 훈련 데이터의 통합이 성능 향상에 긍정적으로 기여할 수 있습니다.
* **테스트 시간 증강(TTA)의 유효성**: TTA는 분할 예측 성능을 효과적으로 향상시키는 것으로 입증되었습니다.
* **다중 작업 학습의 과제**: 다중 모드 데이터를 사용한 다중 작업 학습은 여전히 ​​어려움이 많으며, 본 챌린지에서는 단일 작업 모델에 비해 다중 작업 모델이 명확한 이점을 제공한다는 결론을 내리지 못했습니다. 이는 팀들이 단일 작업 아키텍처를 다중 작업으로 확장하지 않거나, 공동 최적화 방식에서 데이터셋의 다중 모드 특성을 완전히 활용하지 못했기 때문일 수 있습니다.
* **작은 객체 및 과소 대표 클래스 문제**: 작은 객체나 데이터셋에서 과소 대표된 클래스의 분할은 여전히 어려움이 있으며, 일부 모델에서 편향된 예측(예: 클램프를 바늘로 오분류)이 나타났습니다.
* **향후 연구 방향**: 시스템의 강건성을 높이기 위해 기존 데이터셋과 추가 양식(modalities)을 활용하여 이전에 관찰되지 않은 유형의 작업에서도 예측할 수 있는 모델을 탐색해야 합니다.

## 📌 TL;DR

본 논문은 EndoVis 2022 SAR-RARP50 챌린지에서 **로봇 보조 근치적 전립선 절제술(RARP) 비디오**에 대한 **수술 도구 분할** 및 **동작 인식**을 다루는 연구들을 요약합니다. 챌린지는 **SAR-RARP50이라는 최초의 다중 모드, 공개, 생체 내 데이터셋**을 공개하고, 이 데이터셋을 활용하여 단일 작업 및 다중 작업 학습 방식의 성능을 평가했습니다. 상위권 팀들은 주로 **어텐션(attention) 기반 모델** (예: 트랜스포머)과 **사전 훈련**을 활용했으며, **테스트 시간 증강(TTA)**이 분할 성능을 향상시키는 데 효과적이었습니다. 그러나 **다중 작업 학습은 단일 작업 방식에 비해 명확한 이점을 보이지 않았으며**, 덜 숙련된 외과 의사의 비디오에서 모델 성능이 저하되는 등 **데이터 다양성과 모델의 강건성**에 대한 중요성을 강조했습니다.
