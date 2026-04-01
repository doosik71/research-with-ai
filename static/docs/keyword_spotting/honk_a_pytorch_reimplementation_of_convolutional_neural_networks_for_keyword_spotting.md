# Honk: A PyTorch Reimplementation of Convolutional Neural Networks for Keyword Spotting

Raphael Tang and Jimmy Lin

## 🧩 Problem to Solve

이 논문은 "Hey Siri"와 같은 음성 기반 인터페이스의 "명령 트리거(command trigger)"를 인식하는 데 사용되는 키워드 스포팅(keyword spotting)을 위한 컨볼루션 신경망(CNN) 모델을 다룹니다. 특히, 장치에서 직접(on-device) 키워드를 감지하여 사용자의 발화를 클라우드로 전송할지 여부를 결정하는 것이 중요하며, 이는 프라이버시 문제와 계산 효율성(저전력 설정)을 해결하는 데 필수적입니다. 기존에 TensorFlow에 구현되어 있던 모델의 PyTorch 재구현을 통해 접근성과 확장성을 높이는 것이 목표입니다.

## ✨ Key Contributions

- **PyTorch 재구현:** Google TensorFlow의 키워드 스포팅 CNN 모델을 PyTorch로 충실히 재구현한 Honk를 공개했습니다.
- **성능 비교:** 재구현된 Honk 모델이 TensorFlow 참조 구현과 유사한 정확도를 달성함을 입증했습니다.
- **오픈 소스 코드베이스:** 연구자들이 키워드 스포팅 작업에서 활용할 수 있는 오픈 소스 코드베이스와 추가 유틸리티(사용자 지정 음성 명령 기록, 학습/테스트 도구, RESTful 서비스, 데스크톱 애플리케이션 등)를 제공했습니다.
- **모멘텀 학습의 효과:** 확률적 경사 하강법(SGD)에 모멘텀을 적용했을 때 모델 정확도가 향상됨을 확인했습니다.

## 📎 Related Works

- **Sainath and Parada (2015) [1]:** 키워드 스포팅을 위한 간단한 컨볼루션 신경망 모델을 제안했습니다. 본 논문의 모델 아키텍처는 이들의 작업을 기반으로 합니다.
- **Google Speech Commands Dataset (2017) [2]:** Google에서 공개한 대규모 음성 명령 데이터셋으로, 키워드 스포팅 작업의 공공 벤치마크 역할을 합니다.
- **TensorFlow 참조 구현:** Google의 Speech Commands Dataset과 함께 제공된 키워드 스포팅 CNN 모델의 공식 TensorFlow 구현입니다.

## 🛠️ Methodology

Honk는 입력 전처리기(input preprocessor)와 CNN 모델 자체의 두 가지 구성 요소로 구현되었습니다.

1. **입력 전처리:**

   - **데이터 증강:** 배경 소음(백색 소음, 핑크 소음, 인공 소음)을 섞고, 무작위 시간 이동($Y \sim \text{Uniform}[-100, 100]$ 밀리초)을 적용합니다.
   - **특징 추출:** 20Hz/4kHz 대역통과 필터(band-pass filter)를 적용한 후, 30밀리초 윈도우 크기와 10밀리초 프레임 이동을 사용하여 40차원 Mel-Frequency Cepstrum Coefficient (MFCC) 프레임을 생성합니다.
   - **입력 스태킹:** 1초 샘플 내의 모든 30밀리초 윈도우를 스태킹하여 CNN 모델의 입력으로 사용합니다.

2. **모델 아키텍처:**

   - **기본 구조:** 하나 이상의 컨볼루션 레이어(Convolutional Layer)와 이어서 완전 연결 은닉 레이어(Fully-Connected Hidden Layers)로 구성되며, 마지막에 소프트맥스 출력(Softmax Output)이 옵니다.
   - **활성화 함수:** Rectified Linear Unit (ReLU)을 각 비선형 레이어에 사용합니다.
   - **Full Model (`cnn-trad-pool2`):**
     - Sainath와 Parada의 `cnn-trad-fpool3` 모델에서 파생된 TensorFlow의 모델을 재구현했습니다.
     - 두 개의 컨볼루션 레이어를 사용하고, `p=2`, `q=2`로 맥스 풀링(max-pooling)을 수행합니다. 원본 논문의 은닉/선형 레이어는 제외되었습니다.
     - 이 아키텍처는 약 $9.88 \times 10^7$의 곱셈 연산을 요구합니다.
   - **Compact Model (`cnn-one-fstride4` TensorFlow variant):**
     - 저전력 장치를 위한 경량화 모델로, 파라미터와 곱셈 연산 수를 줄입니다.
     - 하나의 컨볼루션 레이어만 사용하며, TensorFlow 변형은 주파수나 시간에서 스트라이딩(striding)을 증가시키지 않습니다.
     - 이 아키텍처는 약 $5.76 \times 10^6$의 곱셈 연산을 요구하여 Full Model보다 훨씬 적습니다.

3. **학습 설정:**
   - **데이터 분할:** Speech Commands Dataset을 학습(80%), 검증(10%), 테스트(10%) 세트로 분할했습니다.
   - **초기화:** 모든 편향(bias)은 0으로, 가중치(weights)는 $\mu=0$, $\sigma=0.01$인 절단 정규 분포(truncated normal distribution)에서 샘플링하여 초기화했습니다.
   - **최적화:** 미니 배치(mini-batch) 크기 100을 사용한 확률적 경사 하강법(SGD)을 사용했습니다. Full 모델에는 학습률 0.001, Compact 모델에는 0.01(모멘텀 적용 시 0.001로 조정)을 적용했습니다.
   - **에포크:** Full 모델은 약 30 에포크, Compact 모델은 약 55 에포크 후에 수렴했습니다.
   - **모멘텀:** 모멘텀 0.9를 사용하여 학습했을 때 성능 향상을 확인했습니다.

## 📊 Results

| Model         | TensorFlow (TF)    | PyTorch (PT)       | PT with momentum            |
| :------------ | :----------------- | :----------------- | :-------------------------- |
| Full Model    | $87.8\% \pm 0.435$ | $87.5\% \pm 0.340$ | $\textbf{90.2\%} \pm 0.515$ |
| Compact Model | $77.4\% \pm 0.839$ | $77.9\% \pm 0.715$ | $\textbf{78.4\%} \pm 0.631$ |

- PyTorch로 재구현된 Honk 모델은 TensorFlow 참조 구현과 **유사한 정확도**를 보였으며, 95% 신뢰 구간이 겹쳤습니다. 이는 Honk가 TensorFlow 모델을 충실히 재현했음을 시사합니다.
- **모멘텀(momentum)을 적용한 SGD 학습**은 특히 Full Model에서 **더 높은 정확도**를 달성했습니다 (Full Model: $87.5\% \to 90.2\%$, Compact Model: $77.9\% \to 78.4\%$).

## 🧠 Insights & Discussion

- 이 연구는 PyTorch를 사용하여 기존 TensorFlow 모델을 성공적으로 재구현하고, 그 성능이 원본과 동등함을 보여줌으로써, PyTorch 생태계 내에서 키워드 스포팅 연구의 **새로운 출발점**을 제공했습니다.
- PyTorch의 **높은 가독성**은 모델 사양을 이해하고 수정하는 데 이점을 제공하며, 이는 향후 연구 그룹의 프로젝트와 일관성을 유지하는 데 도움이 됩니다.
- **오픈 소스 코드베이스**와 추가 유틸리티(녹음, 학습, RESTful 서비스, 데스크톱 앱)는 다른 연구자들이 이 분야에서 실험을 복제하고 확장하는 데 용이한 플랫폼을 제공합니다.
- **모멘텀을 이용한 학습**이 정확도를 향상시킬 수 있음을 보여주었으며, 이는 모델 훈련 전략에 대한 추가적인 통찰을 제공합니다.
- **향후 연구 방향**으로는 제한된 컴퓨팅 파워를 가진 장치에의 배포, 입력 데이터 전처리에 대한 다양한 기술 탐구, 명령 트리거를 쉽게 추가할 수 있는 프레임워크 개발 등이 제안되었습니다.

## 📌 TL;DR

이 논문은 Google의 키워드 스포팅 CNN 모델을 PyTorch로 재구현한 오픈 소스 프로젝트 **Honk**를 소개합니다. Google Speech Commands Dataset에 대한 평가를 통해 Honk가 기존 TensorFlow 구현과 **유사한 정확도를 달성**함을 입증했습니다. 특히, 모멘텀을 사용한 학습이 모델 성능을 향상시킬 수 있음을 보여주며, 키워드 스포팅 작업을 위한 **접근성 높고 확장 가능한 PyTorch 기반 연구 플랫폼**을 제공합니다.
