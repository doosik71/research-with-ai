# Rethinking Surgical Instrument Segmentation: A Background Image Can Be All You Need

An Wang, Mobarakol Islam, Mengya Xu, and Hongliang Ren

## 🧩 Problem to Solve

의료 영상 분야, 특히 로봇 수술에서의 수술 도구 분할(surgical instrument segmentation)은 데이터 수집 및 주석(annotation)의 높은 비용, 데이터 희소성, 클래스 불균형 등으로 인해 딥러닝 모델의 정확성과 배포에 큰 제약을 겪고 있습니다. 새로운 수술 절차나 도구에 대한 데이터셋이 없는 경우, 이 문제는 더욱 심각해집니다. 기존의 합성 데이터 생성 방법론도 여전히 많은 수동적인 노력과 원본 이미지 수집을 필요로 하는 한계가 있었습니다.

## ✨ Key Contributions

* 최소한의 데이터 수집 노력과 수동 주석 없이, 수술 도구 분할을 위한 고품질 합성 데이터셋을 생성하는 데이터 효율적인 프레임워크를 제안했습니다.
* 전경 및 배경 원본 이미지에 다양한 증강(augmentation) 및 블렌딩(blending) 조합을 적용하고, 학습 시 연쇄 증강 혼합(chained augmentation mixing)을 도입하여 데이터 다양성을 높이고 도구 클래스 분포의 균형을 맞췄습니다.
* 단일 수술 배경 이미지만을 사용하여 생성된 합성 데이터로 훈련함으로써, 실제 데이터셋(EndoVis-2018 및 EndoVis-2017)에서 수용 가능한 수술 도구 분할 성능을 달성했으며, 심지어 새로운(novel) 도구에 대해서도 예측이 가능함을 입증했습니다.

## 📎 Related Works

* **합성 데이터 활용**: Tremblay et al.의 도메인 무작위화(domain randomization), Gabriel et al.의 GAN(Generative Adversarial Networks) 앙상블, Kishore et al.의 모방 훈련(imitation training) 등이 컴퓨터 비전 분야에서 합성 데이터를 활용하는 연구로 언급됩니다.
* **의료 분야 합성 데이터**: GAN 기반 데이터 합성(Cao et al., Han et al., Shin et al.)과 이미지 블렌딩/합성(Garcia-Peraza-Herrera et al.의 mix-blend) 연구가 있었습니다. 특히 Garcia-Peraza-Herrera et al.의 연구는 블렌딩을 사용했으나 여전히 수천 장의 전경 및 배경 이미지를 수동으로 수집해야 하는 한계가 있었습니다.
* **데이터 증강**: Goodfellow et al.에 따르면 데이터 증강은 모델의 일반화 오류를 낮추는 정규화(regularisation) 기법으로 사용될 수 있으며, 클래스 불균형 문제 완화에도 기여합니다.
* **학습 시 증강**: Hendrycks et al.의 AugMix은 학습 시 다양한 연쇄 증강을 혼합하여 모델의 강건성(robustness)과 불확실성(uncertainty)을 개선하는 방법을 제안했습니다.

## 🛠️ Methodology

본 연구는 단일 배경 이미지와 소수의 전경 도구 이미지를 활용하여 다양한 합성 수술 장면을 생성합니다.

1. **배경 조직 이미지 처리**:
    * 오픈 소스 EndoVis-2018 데이터셋에서 도구의 출현이 최소화된 **단일 배경 조직 이미지**를 수집합니다.
    * `imgaug` 라이브러리를 사용하여 `LinearContrast`, `FrequencyNoiseAlpha`, `AddToHueAndSaturation`, `Multiply`, `PerspectiveTransform`, `Cutout`, `Affine`, `Flip`, `Sharpen`, `Emboss`, `SimplexNoiseAlpha`, `AdditiveGaussianNoise`, `CoarseDropout`, `GaussianBlur`, `MedianBlur` 등 **다양한 증강 기법**을 적용하여 $p$개의 배경 이미지 변형(pool $X^p_b$)을 생성합니다.

2. **전경 도구 이미지 처리**:
    * EndoVis-2018 데이터셋에서 각 도구 유형별로 2~3개의 이미지만을 **원본 전경 이미지**로 사용합니다. 도구 마스크와 함께 추출하여 배경을 투명하게 만듭니다.
    * 도구의 자세(예: 가위의 개폐 상태)를 커버하기 위해 사전 지식을 바탕으로 원본 이미지를 신중하게 선택합니다.
    * 배경 이미지와 유사하게 다양한 증강 기법을 적용하여 $q$개의 전경 이미지 변형(pool $X^q_f$)을 생성하며, 이 과정에서 마스크도 함께 증강하여 분할 정확도를 유지합니다.

3. **이미지 블렌딩**:
    * 생성된 배경 이미지 풀 $X^p_b$과 전경 이미지 풀 $X^q_f$에서 이미지를 무작위로 선택하여 블렌딩합니다.
    * 전경 이미지는 배경 이미지 위에 붙여넣어지며, 겹치는 부분의 픽셀 값은 도구에서 가져옵니다.
    * 실제 수술 장면을 고려하여 **때때로 두 개의 도구 이미지를 배경에 붙여넣어** 모델이 이미지 내 도구의 픽셀 점유율을 더 잘 추정하도록 돕습니다.
    * 이를 통해 총 $t$개의 합성 이미지 풀 $X^t_s = \{\Theta(x^i_f, x^j_b)\}$를 생성합니다.

4. **학습 중 연쇄 증강 혼합 (In-training chained augmentation mixing)**:
    * AugMix에서 영감을 받아 학습 시간에 `autocontrast`, `equalize`, `posterize`, `solarize` 등으로 구성된 AugMix-Soft와 여기에 `color`, `contrast`, `brightness`, `sharpness`를 추가한 AugMix-Hard 두 가지 증강 컬렉션을 사용합니다.
    * 합성 훈련 샘플은 다음과 같이 표현됩니다:
        $$x_{AM}^s = m \cdot \Theta(x_f, x_b) + (1-m) \cdot \sum_{i=1}^N (w_i \cdot H_i(\Theta(x_f, x_b)))$$
        여기서 $m$은 Beta 분포에서 샘플링된 무작위 볼록 계수이고, $w_i$는 Dirichlet 분포에서 샘플링된 무작위 볼록 계수로 증강 체인의 혼합 가중치를 제어합니다. $H_i$는 $i$번째 증강 체인에 통합된 증강 연산을 나타냅니다.

5. **모델 및 평가**:
    * 분할 모델 백본으로는 UNet [17]을 사용합니다.
    * 평가 지표로는 Dice Similarity Coefficient (DSC)를 사용합니다.

## 📊 Results

* **합성 데이터셋 구축**: 단일 배경 이미지를 기반으로 Synthetic-A (4000장, 1개 도구), Synthetic-B (6000장, 2개 도구 이미지 추가), Synthetic-C (8000장, 3개 원본 도구 이미지, 80% 1개 도구, 20% 2개 도구) 데이터셋을 구축했습니다.
* **성능 평가**:
  * **EndoVis-2018 (타겟 도메인)**: Synthetic-C 데이터셋과 AugMix-Hard를 사용했을 때 73.51%의 DSC를 달성했습니다. 이는 실제 EndoVis-2018 데이터셋으로 훈련했을 때의 최고 성능(83.15% with AugMix-Soft)보다는 낮지만, 데이터 수집 및 주석 비용을 고려할 때 매우 고무적인 결과입니다.
  * **EndoVis-2017 (미등록 도메인)**: EndoVis-2017 데이터셋에 대한 일반화 능력도 평가했으며, Synthetic-C와 AugMix-Soft로 훈련했을 때 75.69%의 DSC를 달성했습니다. 이는 새로운 도메인과 미등록 도구(예: Vessel Sealer)에 대한 모델의 좋은 일반화 능력을 보여줍니다.
* **증강 혼합의 효과**: 제안된 학습 시 증강 혼합(AugMix-Soft/Hard) 전략은 `ColorJitter`와 같은 다른 증강 기법에 비해 EndoVis-2018에서 5.33%, EndoVis-2017에서 4.29%의 DSC 성능 향상을 보이며 그 효율성을 입증했습니다.
* **클래스 증분 문제 해결**: EndoVis-2018 훈련 데이터에 새로운 도구(Vessel Sealer)에 대한 합성 이미지 2000장을 추가했을 때, EndoVis-2017 데이터셋에서 전체 성능이 크게 향상되어(예: AugMix-Soft 사용 시 84.06%에서 85.72%로) 클래스 증분 문제에 효과적으로 대처할 수 있음을 보여주었습니다.
* **합성-실제 공동 훈련**: Synthetic-C 데이터셋에 EndoVis-2018의 실제 데이터 중 10% 또는 20%의 소량만 추가했을 때, 분할 성능이 크게 향상되어 전체 실제 데이터로 훈련한 모델과 유사한 수준(예: Synthetic-C + 20% Endo18은 EndoVis-2018에서 82.45% DSC를 달성, 전체 Endo18 훈련 시 82.91%)에 도달했습니다.

## 🧠 Insights & Discussion

본 연구는 수술 도구 분할 작업에서 데이터 희소성 및 높은 주석 비용 문제를 해결하기 위한 비용 효율적인 데이터 중심 프레임워크의 가능성을 제시합니다. 단일 배경 이미지와 소수의 전경 도구 이미지만으로 생성된 합성 데이터셋이 실제 환경에서 수용 가능한 성능을 달성할 수 있음을 입증한 것은 매우 중요한 통찰입니다.

이러한 결과는 데이터 부족 외에도 클래스 불균형, 도메인 적응, 증분 학습 등 딥러닝의 여러 한계를 극복하기 위한 데이터 중심 방법론의 중요성을 강조합니다. 특히, 소량의 실제 데이터를 합성 데이터와 함께 사용할 때 성능이 크게 향상된다는 점은 실제 환경에서의 효과적인 훈련 전략을 제안합니다.

향후 연구에서는 더 복잡한 도구별(instrument-wise) 분할로 확장하고, 소작 연기(cautery smoke)나 도구 그림자(instrument shadow)와 같은 실제 수술 장면의 더 많은 사전 지식을 통합하여 합성 데이터셋의 품질을 더욱 향상시킬 수 있을 것입니다.

## 📌 TL;DR

수술 도구 분할의 데이터 부족 및 고비용 주석 문제에 대응하여, 본 논문은 단일 수술 배경 이미지와 소수의 오픈 소스 도구 이미지만을 활용하여 합성 데이터셋을 생성하는 비용 효율적인 데이터 중심 프레임워크를 제안합니다. 광범위한 증강과 블렌딩 기법, 그리고 학습 중 연쇄 증강 혼합(AugMix-inspired)을 통해 데이터 다양성을 극대화합니다. 실험 결과, 이 방법은 실제 데이터셋에서 수용 가능한 분할 성능을 달성했으며, 미등록 도메인과 새로운 도구에 대한 뛰어난 일반화 능력을 보였습니다. 또한, 소량의 실제 데이터를 추가하면 성능이 크게 향상될 수 있음을 입증하여, 최소한의 노력으로 효율적인 딥러닝 모델 학습이 가능함을 보여줍니다.
