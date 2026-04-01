# Learning Efficient Representations for Keyword Spotting with Triplet Loss

Roman Vygon, Nikolay Mikhaylovskiy

## 🧩 Problem to Solve

트리플렛 손실(triplet loss) 기반의 메트릭 임베딩은 컴퓨터 비전 분야(예: 사람 재식별)에서 널리 사용되는 반면, 음성 인식 분야의 분류 문제에는 거의 적용되지 않습니다. 이 연구는 키워드 스포팅(KWS)을 위한 소형 컨볼루션 네트워크의 분류 정확도를 향상시키기 위해 메트릭 학습 기술을 효과적으로 적용하는 것이 주요 문제입니다.

## ✨ Key Contributions

- 트리플렛 손실 기반 메트릭 임베딩과 kNN 분류기를 결합하여 CNN 기반 오디오 분류 모델의 정확도를 **26%에서 38%까지** 크게 향상시켰습니다.
- Google Speech Commands 데이터셋에서 새로운 SOTA(State-of-the-Art) 결과를 달성했습니다 (V1 10+2 클래스 98.55%, V2 10+2 클래스 98.37%, V2 35 클래스 97.0% 정확도).
- 불균형 데이터셋에서 F1 측정값을 개선하기 위해 음성학적 유사성(phonetic similarity) 기반의 새로운 배치 샘플링(batch sampling) 방식을 제안했습니다.

## 📎 Related Works

- **기존 KWS 아키텍처:** HMM, 규칙 기반 시스템 등 다양한 머신러닝 아키텍처가 제안되었습니다. 신경망 시대에는 CNN (Warden, Sainath and Parada), RNN 기반 모델(de Andrade et al.의 Attention-RNN, Rybakov et al.의 Multihead Attention RNN), 경량 CNN (Majumdar and Ginsburg의 MatchboxNet, Wei et al.의 EdgeCRNN) 등이 사용됩니다.
- **음성 분야에서의 트리플렛 손실:** 음성 감정 인식(Huang J. et al. 등), 화자 분할(Bredin, Song et al.), 화자 검증(Zhang and Koshida, Li et al.), 데이터 증강(Turpault et al.) 등 특정 음성 관련 작업에 사용되었습니다.
- **본 연구와 유사한 작업:** Sacchi et al., Shor et al., Yuan et al., Huh et al., Huang et al. 등의 연구가 있으나, 본 연구는 폐쇄형 어휘 KWS에 초점을 맞추고 기존 소형 아키텍처에 트리플렛 손실과 kNN을 결합한 분류에 중점을 둡니다.

## 🛠️ Methodology

- **모델 아키텍처:** 주로 ResNet 기반 모델(res8, res15)을 사용했으며, 이 모델의 인코더 부분이 트리플렛 손실 기반 임베딩을 생성합니다. RNN 기반 아키텍처는 트리플렛 손실에서 성능이 좋지 않아 제외되었습니다.
- **입력 전처리:** 25ms 윈도우 크기와 10ms 프레임 이동으로 64차원(LibriWords) 또는 80차원(Google Speech Commands) 멜 스펙트로그램을 구성합니다.
- **트리플렛 손실:** 표준 트리플렛 손실 함수인 $l(p_i, p_i^+, p_i^-) = \max(0, g + D(f(p_i), f(p_i^+)) - D(f(p_i), f(p_i^-)))$를 유클리드 거리 $D(f(P), f(Q)) = ||f(P) - f(Q)||_2^2$와 함께 사용합니다. 여기서 $p_i$는 앵커, $p_i^+$는 긍정, $p_i^-$는 부정 샘플입니다.
- **배치 샘플링 (트리플렛 마이닝):**
  - **Uniform:** 모든 클래스에서 동일한 수의 객체를 무작위로 샘플링합니다.
  - **Proportional:** 데이터셋의 단어 분포에 비례하여 클래스를 무작위로 샘플링합니다.
  - **Phonetic (새로운 접근 방식):** SoundEx, Caverphone, Metaphone, NYSIIS 알고리즘을 사용하여 음성학적 유사성 행렬을 계산합니다. Metaphone이 개별 알고리즘 중 가장 좋은 성능을 보였으며, 네 가지 알고리즘의 가중 평균을 사용했습니다: $D_{\text{Phonetic}} = D_{\text{Soundex}} \cdot 0.2 + D_{\text{Caverphone}} \cdot 0.2 + D_{\text{Metaphone}} \cdot 0.5 + D_{\text{NYSIIS}} \cdot 0.1$.
- **트리플렛 선택:** [23]을 기반으로 손실이 0이 아닌 부정 샘플을 무작위로 선택하는 온라인 배치 트리플렛 마이닝 방식을 사용합니다.
- **분류기:** 학습된 임베딩에 대해 K-최근접 이웃(kNN)을 사용하여 분류합니다. kNN을 완전 연결 네트워크로 대체했을 때 성능이 현저히 떨어져, kNN의 중요성을 확인했습니다.
- **최적화 및 훈련:** Novograd 최적화 도구를 사용하여 초기 학습률 0.001에서 코사인 어닐링(cosine annealing)으로 $1e-4$까지 감소시켰습니다.
- **데이터 증강:** 샘플 쉬프팅(-100ms; +100ms), SpecAugment, Google Speech Commands 데이터셋의 배경 노이즈 추가를 사용했습니다.

## 📊 Results

- **크게 향상된 성능:** 트리플렛 손실 + kNN 모델은 교차 엔트로피(crossentropy) 기반 기준 모델보다 정확도에서 **25% ~ 38%**, F1 측정값에서 **16% ~ 57%** 더 나은 성능을 보였습니다. 클래스 수가 많을수록 향상 폭이 더 컸습니다.
- **SOTA 달성:**
  - Google Speech Commands V2 35-클래스: 97.0% 정확도 (이전 SOTA 대비 50% 이상 향상).
  - Google Speech Commands V2 10+2-클래스: 98.37% 정확도 (약 16% 향상).
  - Google Speech Commands V1 10+2-클래스: 98.55% 정확도 (약 34% 향상).
- **샘플링 전략의 효과:** Proportional 샘플링은 정확도를, Phonetic 샘플링은 F1 측정값을 향상시키는 데 기여했습니다. Uniform 샘플링은 적절한 클래스 커버리지에 필수적이었습니다.
- **kNN의 결정적인 역할:** kNN을 완전 연결 네트워크로 대체했을 때 정확도가 98.37%에서 약 90%로 크게 떨어져, 트리플렛 손실 임베딩이 선형적으로 분리되지 않으며 kNN이 고품질 디코딩에 매우 중요함을 입증했습니다.

## 🧠 Insights & Discussion

- 이 연구는 키워드 스포팅 작업에서 메트릭 학습(트리플렛 손실)과 비모수 분류기(kNN)의 결합이 매우 효과적임을 보여줍니다. 이는 학습된 임베딩이 유사성 측정에는 적합하지만, 반드시 직접적인 선형 분류에 적합하지는 않음을 시사합니다.
- 트리플렛 손실에 의해 생성된 임베딩이 선형적으로 분리되지 않으므로, 고품질 디코딩을 위해 kNN의 역할이 매우 중요합니다.
- 음성학적 유사성 기반 배치 샘플링은 특히 불균형 데이터셋이나 음성학적으로 유사한 단어(예: "at"-"ate")를 구별하는 데 있어 F1 점수를 개선하는 데 결정적인 역할을 합니다.
- 이 접근 방식은 소형 아키텍처에 잘 작동하므로 온디바이스(on-device) KWS 애플리케이션에 적합합니다.
- 향후 연구로는 음성학적 유사성 알고리즘의 최적화된 사용법과 kNN 양자화(quantization) 시 정확도와의 균형점 탐색이 필요합니다.

## 📌 TL;DR

본 논문은 트리플렛 손실 기반 메트릭 임베딩과 kNN 분류기를 CNN 모델에 결합하여 키워드 스포팅 정확도를 크게 향상시켰습니다. 특히 음성학적 유사성 기반의 새로운 배치 샘플링 방법을 제안하여 Google Speech Commands 데이터셋에서 새로운 최고 성능을 달성했으며, LibriWords 데이터셋에서는 교차 엔트로피 기준 모델 대비 25-38%의 정확도와 16-57%의 F1 점수 향상을 이루었습니다. 이는 소형 KWS에서 비선형적으로 분리 가능한 음성 임베딩에 대한 메트릭 학습 및 kNN의 효과를 강조합니다.
