# Learning from Few Samples: A Survey

Nihar Bendre, Hugo Terashima Marín, Peyman Najafirad

## 🧩 Problem to Solve

심층 신경망은 방대한 양의 데이터에서는 뛰어난 성능을 보이지만, 새로운 범주나 작업에 대해 제한된 수의 샘플만으로 학습하고 일반화하는 데 어려움을 겪습니다. 이 논문은 이러한 "소수 샘플 학습(few-shot learning)"의 도전 과제를 해결하는 데 초점을 맞춥니다. 인간은 적은 정보만으로도 새로운 개념을 빠르게 학습할 수 있지만, 기계는 대규모의 균형 잡힌 레이블 데이터셋에 크게 의존하는 경향이 있으며, 이러한 데이터셋을 구축하는 것은 시간, 노력, 비용 면에서 비현실적입니다.

## ✨ Key Contributions

* 2016년부터 2020년까지 발표된 소수 샘플 메타 학습(few-shot meta-learning) 분야의 주요 연구들을 분석하고, 기술들을 네 가지 범주(데이터 증강 기반, 임베딩 기반, 최적화 기반, 의미론 기반)로 분류하는 분류 체계를 제시합니다.
* 각 범주에 속하는 대표적인 기술들을 요약하고 그 접근 방식을 설명합니다.
* 널리 사용되는 벤치마크 데이터셋인 Omniglot과 MiniImageNet에서 이러한 기술들의 성능을 비교합니다.
* 현재 기술의 한계점과 인간의 성능을 능가하기 위한 향후 연구 방향에 대해 논의합니다.

## 📎 Related Works

* **메타 학습 초기 연구:** Yoshua Bengio와 Samy Bengio [67]가 초기 메타 학습 연구를 수행했습니다.
* **소수 샘플 학습 초기 연구:** Fei-Fei Li [68]가 소수 샘플 학습의 초기 작업을 진행했습니다.
* **측정 학습 (Metric Learning):** 임베딩 공간에서 유사성을 학습하는 오래된 기술로, 같은 범주의 샘플은 가깝게, 다른 범주의 샘플은 멀리 배치합니다.
* **전이 학습 (Transfer Learning):** 기존 작업에서 학습된 지식을 관련성 있는 새로운 작업에 전이하는 방식으로, 소수 샘플 학습과 유사하나 메타 학습만큼 새로운 작업에 빠르게 적응하지 못하는 경우가 많습니다.
* **자기 지도 학습 (Self-Supervised Learning):** 레이블 없는 대량의 데이터에서 사전 텍스트(pre-text) 작업을 통해 사전 지식을 습득하고, 이를 주요 하위 작업에 미세 조정하는 방식으로, 메타 학습과 유사하게 사전 지식 활용을 목표로 합니다.

## 🛠️ Methodology

이 설문 논문은 컴퓨터 비전 영역 내 소수 샘플 메타 학습 기술들을 체계적으로 분석합니다.

* **분류 체계 수립:** 기존 기술들을 다음 네 가지 주요 범주로 분류합니다.
  * **데이터 증강 기반 (Data Augmentation Based):** 제한된 샘플을 증강하여 더 많은 다양성을 가진 훈련 데이터를 생성합니다. (예: LaSO [71], Shrinking and Hallucinating Features [69], Saliency-guided Hallucination [72], Imaginary Data [70], Maximum-Entropy Patch Sampler [73], Image Deformation Meta-Networks [74])
  * **임베딩 기반 (Embedding Based) / 측정 학습 기반 (Metric Learning Based):** 데이터를 저차원 임베딩 공간으로 변환한 후, 특정 거리 함수를 사용하여 클러스터링하고 비교합니다. (예: Relation Network [78], Prototypical Network [77], Localization in Realistic Settings [83], Semi-Supervised Classification [79], Transferable Prototypical Networks [82], Matching Network [75], Task Dependent Adaptive Metric for Improved Few-Shot Learning (TADAM) [88], Representative-based Metric Learning [81], Task-Aware Feature Embedding Network (TAFE-Net) [80])
  * **최적화 기반 (Optimization Based):** 메타 최적화기(meta-optimizer)를 사용하여 모델이 학습 과정 자체를 학습하고 새로운 작업에 더 잘 일반화되도록 합니다. (예: LSTM-based Meta Learner [76], Memory Augmented Neural Network (MANN) [84], Model Agnostic Meta Learning (MAML) [85], Task-Agnostic Meta-Learning (TAML) [90], Meta-SGD [86], Deep Meta-Learning (DEML) [87], $\Delta$-encoder [89])
  * **의미론 기반 (Semantic Based):** 샘플과 함께 의미론적 정보를 활용하여 새로운 범주를 학습하고 일반화합니다. (예: Multiple Semantics [92], Aligned Variational Autoencoders (CADA-VAE) [93], Knowledge Transfer With Class Hierarchy [94])
* **벤치마크 데이터셋 비교:** Omniglot 및 MiniImageNet 데이터셋에서 각 범주의 대표적인 기술들의 성능을 정량적으로 비교합니다.

## 📊 Results

* **Omniglot 데이터셋:** 대부분의 모델이 1-shot 및 5-shot 설정에서 높은 정확도(90% 후반)를 달성했습니다. 특히 최적화 기반 기술인 Meta-SGD [86]는 99.53%의 1-shot 정확도로 가장 우수한 성능을 보였습니다.
* **MiniImageNet 데이터셋:** Omniglot 데이터셋에 비해 전반적인 성능이 크게 하락했습니다. 이는 이미지의 복잡성과 정보의 풍부함이 더 높기 때문입니다.
  * 데이터 증강, 임베딩, 최적화 기반 접근 방식은 1-shot 정확도 약 55% 수준을 보였으며, 5-shot에서는 약 70% 수준으로 상승했습니다.
  * **의미론 기반 (Semantics-based) 기술**인 Multiple Semantics [92]가 1-shot에서 67.3%로 당시 가장 높은 정확도를 달성하며 최첨단(state-of-the-art) 성능을 보여주었습니다.
* **인간 성능과의 비교:** 4-5세 아동의 성능(약 70%)과 비교했을 때, 현재 최첨단 모델은 MiniImageNet에서 여전히 부족하며, 성인 인간의 성능(99%)과는 상당한 격차가 존재합니다.

## 🧠 Insights & Discussion

* **현재 모델의 한계:** MiniImageNet과 같은 복잡한 데이터셋에서 현재의 소수 샘플 메타 학습 모델들은 여전히 인간의 학습 능력에 크게 미치지 못합니다. 특히 소수 샘플에 대한 일반화 능력이 부족합니다.
* **의미론 기반 접근의 잠재력:** 의미론적 정보(예: 자연어 설명, 속성)를 데이터와 함께 사용하는 접근 방식이 복잡한 이미지 분류에서 가장 좋은 성능을 보였으나, 단독으로는 인간 수준의 성능에 도달하기에 부족합니다.
* **향후 연구 방향:**
  * **하이브리드 모델 개발:** 데이터 증강, 임베딩, 의미론 기반 기술의 장점을 결합하여 성능을 향상시키는 하이브리드 모델이 필요합니다.
  * **고급 모델 아키텍처 통합:** 어텐션 메커니즘, 셀프-어텐션 메커니즘, 트랜스포머 [1] 또는 Variational Autoencoder (VAE)의 변형(예: $\beta$-VAE [145], VQ-VAE [146], VQ-VAE-2 [147], TD-VAE [148])과 같은 고급 모델을 통합하여 제한된 샘플에 대한 저차원 의미론적 정보를 더 잘 생성하고 일반화 능력을 향상시켜야 합니다.
  * **개선된 거리 함수:** 저차원 임베딩을 견고하게 분류하거나 클러스터링할 수 있는 개선된 거리 함수(예: 유클리드 또는 코사인 거리)의 개발이 중요합니다.

## 📌 TL;DR

이 논문은 컴퓨터 비전 분야의 **소수 샘플 메타 학습(Few-Shot Meta-Learning)** 기술들을 종합적으로 **조사**합니다. 심층 신경망이 제한된 샘플만으로 새로운 범주에 일반화하는 **문제**를 해결하기 위해, 연구들은 **데이터 증강, 임베딩, 최적화, 의미론 기반**의 네 가지 접근 방식으로 분류됩니다. **결과**는 Omniglot과 같은 단순한 데이터셋에서는 높은 성능을 보였지만, MiniImageNet과 같은 복잡한 데이터셋에서는 성능이 크게 하락하며 특히 인간의 학습 능력에 비하면 아직 큰 격차가 있음을 보여줍니다. **향후 연구**는 이러한 접근 방식들을 결합한 하이브리드 모델과 고급 아키텍처를 통해 모델의 일반화 능력을 향상시키는 방향으로 나아가야 함을 제안합니다.
