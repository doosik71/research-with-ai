# SMALL-FOOTPRINT KEYWORD SPOTTING WITH GRAPH CONVOLUTIONAL NETWORK

Xi Chen, Shouyi Yin, Dandan Song, Peng Ouyang, Leibo Liu, Shaojun Wei

## 🧩 Problem to Solve

자원 제약적인(resource-constrained) 장치에서 높은 정확도를 가진 키워드 스포팅(KWS)을 달성하는 것은 여전히 도전 과제입니다. 기존의 심층 신경망, 특히 CNN 기반 방법은 특징 맵의 장거리 의존성(long-range dependencies) 및 전역 컨텍스트(global context)를 효과적으로 포착하지 못하는 한계가 있습니다. RNN 기반 접근 방식은 더 긴 시간적 컨텍스트를 모델링할 수 있지만, 연속적인 입력 스트림에 직면했을 때 상태 포화(state saturation) 문제와 계산 비용 및 지연 시간 증가를 초래할 수 있습니다.

## ✨ Key Contributions

- 병목 구조(bottleneck structure)를 활용하여 소형(small-footprint) KWS를 위한 작고 효율적인 합성곱 신경망(CENet)을 제안했습니다.
- 그래프 합성곱 신경망(GCN)을 도입하여 특징 맵 내의 장거리 의존성을 포착하고 컨텍스트 특징 증강(contextual feature augmentation)을 달성했습니다.
- Google Speech Command Dataset에서 기존 최신 기술(state-of-the-art)을 훨씬 능가하는 성능을 더 낮은 계산 비용으로 달성했습니다.

## 📎 Related Works

- **기존 KWS 방법:** 키워드/필터 은닉 마르코프 모델(HMM) [2, 3, 4, 5, 6], 대용량 어휘 연속 음성 인식기(LVCSR) [7, 8, 9, 10].
- **DNN 기반 KWS:** 딥 KWS [11] (프레임 레벨 후방 확률), CNN [12, 13] (제한된 메모리/연산 자원), ResNet [14] (모델 크기 및 성능 절충), RNN [15, 16, 17, 18, 19] (긴 시간 컨텍스트 모델링).
- **그래프 합성곱 네트워크 (GCN):** Kipf 및 Welling [20]의 준지도 학습 GCN.
- **잔여 학습(Residual Learning):** He 등 [21]의 심층 잔여 학습 (ResNet).
- **비국소적(Non-local) 관계 모델링:** Wang 등 [23]의 비국소적 신경망, Fu 등 [25]의 듀얼 어텐션 네트워크.

## 🛠️ Methodology

본 연구에서는 잔여 연결(residual connection)과 병목 구조(bottleneck structure)를 기반으로 하는 작고 효율적인 CENet(Compact and Efficient Network)과 장거리 의존성을 인코딩하는 GCN(Graph Convolutional Network)을 결합한 CENet-GCN 모델을 제안합니다.

1. **KWS 작업 및 전처리:**

   - 오디오 스트림에서 사전 정의된 키워드를 감지하는 분류 문제입니다.
   - **전처리 단계:**
     - 20Hz/4kHz 대역통과 필터를 사용하여 노이즈를 감소시킵니다.
     - 30ms 윈도우 크기와 10ms 프레임 이동을 사용하여 MFCC(Mel-Frequency Cepstrum Coefficient) 특징을 추출합니다.
     - 입력 MFCC 특징은 $I \in \mathbb{R}^{t \times f}$로, 여기서 $t=101$은 프레임 수, $f=40$은 MFCC 특징의 차원입니다.
     - KWS는 신경망으로 구현된 매핑 함수 $F$를 통해 입력 특징 공간을 레이블 공간으로 매핑하는 $y = F(I, \Theta)$로 정식화됩니다.

2. **작고 효율적인 신경망 (CENet):**

   - ResNet [21]의 아이디어를 바탕으로 리소스가 제한된 환경을 위한 KWS 모델을 설계합니다.
   - **구성 블록:**
     - **Initial block:** $3 \times 3$ 편향 없는(bias-free) 합성곱 층, 배치 정규화(batch normalization), ReLU 활성화 함수, $2 \times 2$ 평균 풀링(average pooling) 층으로 구성됩니다.
     - **Bottleneck block:** 모델 복잡성을 줄이기 위해 $1 \times 1$, $3 \times 3$, $1 \times 1$ 합성곱 층으로 구성된 3개의 층 스택을 사용합니다. $1 \times 1$ 층은 차원 감소 및 복원 역할을 하며, $3 \times 3$ 층을 더 작은 입/출력 차원을 가진 병목으로 만듭니다.
     - **Connection block:** 각 단계(stage)의 끝에 사용되는 특수 병목 블록으로, 스트라이드 2(stride of 2)를 가진 합성곱 층을 통해 차원을 증가시키고 특징 맵의 크기를 줄입니다.
   - CENet은 하나의 initial block과 세 개의 단계(stage)로 구성된 다단계(multi-stage) 구조를 가집니다. 각 단계는 여러 bottleneck block과 하나의 connection block을 포함합니다. 네트워크는 전역 평균 풀링(global average pooling) 층과 softmax 함수를 가진 $l$방향 완전 연결 층으로 끝납니다.
   - 모델 크기를 제어하기 위해 채널 수를 작게 유지하며, CENet-6 (16.2K 파라미터), CENet-24 (44.3K 파라미터), CENet-40 (61K 파라미터)의 세 가지 변형 모델을 제안합니다.

3. **GCN을 이용한 컨텍스트 특징 증강:**
   - 합성곱 특징 표현에서 장거리 의존성을 포착하기 위해 GCN [20]을 도입하여 비국소적 관계(non-local relations)를 모델링합니다.
   - 글로벌 컨텍스트 특징 추정은 GCN 형태로 정식화됩니다. $X = [x_1, \cdots, x_N]^{\top}$를 합성곱 특징 집합이라고 할 때, 여기서 $x_i \in \mathbb{R}^c$는 $c$채널 특징 벡터이고 $i$는 특징 벡터의 공간 위치를 나타냅니다. KWS에서는 $N=w \times h$ 크기의 2D 특징 맵을 가집니다.
   - 노드 $v_i \in V$와 엣지 $(v_i, v_j) \in E$를 가진 완전 연결 그래프 $G=(V, E)$가 구축됩니다.
   - 비국소적 관계는 다음처럼 정의됩니다:
     $\tilde{x}_i = \sigma\left(\frac{1}{Z_i(X)}\sum_{j=1}^{N}g(x_i,x_j)W^{\top}x_j\right)$
     여기서 $\tilde{x}_i$는 노드 $i$에서 업데이트된 특징 표현이고, $\sigma$는 활성화 함수(예: ReLU)이며, $g$는 쌍별 관계를 인코딩하는 거리 측정 함수이고, $Z_i(X)$는 정규화 인자이며, $W \in \mathbb{R}^{c \times c}$는 노드 $i$로부터 메시지를 인코딩하는 선형 매핑을 정의하는 가중치 행렬입니다.
   - 이를 행렬 형태로 표현하면 $\tilde{X}=\sigma(A(X)XW)$가 됩니다. 여기서 $A(X) \in \mathbb{R}^{N \times N}$는 그래프의 유사성 행렬로 $A_{i,j} = \frac{1}{Z_i(X)}g(x_i, x_j)$입니다.
   - 본 연구에서는 쌍별 유사도를 측정하기 위해 softmax 함수를 사용한 임베디드 가우시안(Embedded Gaussian)을 채택합니다. 따라서 $\tilde{X}$는 다음처럼 재작성됩니다:
     $\tilde{X}=\sigma(\text{softmax}(X^{\top}W^{\top}_{\theta}W_{\phi}X)XW)$
     여기서 $\theta(x_i) = W_{\theta}x_i$와 $\phi(x_j) = W_{\phi}x_j$는 두 개의 임베딩입니다.
   - 증강된 특징 $X_a$는 컨텍스트 특징 $\tilde{X}$와 원래 합성곱 특징 $X$의 가중치 합산으로 계산됩니다: $X_a = \gamma \tilde{X} + X$, 여기서 $\gamma$는 학습 가능한 스케일링 파라미터입니다.
   - GCN 모듈은 CENet의 각 단계(stage) 끝에 삽입되어 다른 수준에서 장거리 의존성을 인코딩합니다. 모델 복잡성을 낮게 유지하기 위해 단일 층 GCN을 사용합니다.

## 📊 Results

- **데이터셋:** Google Speech Command Dataset [22] (12개 명령, 65,000개의 1초 음성).
- **평가 지표:** 클래스 정확도, 파라미터 수, 곱셈(연산량).
- **CENet 기본 모델 성능:**
  - CENet-6 (16.2K 파라미터, 1.95M 곱셈): 93.9% 정확도. 기존의 더 작은 모델(one-stride1 [12], res8-narrow [14])을 능가합니다.
  - CENet-24 (44.3K 파라미터, 8.51M 곱셈): 95.6% 정확도. res15 [14] (238K 파라미터, 894M 곱셈)와 필적하는 성능을 5배 적은 파라미터와 100배 적은 연산량으로 달성합니다.
  - CENet-40 (61K 파라미터, 16.18M 곱셈): 96.4% 정확도.
- **GCN 모듈 결합 CENet-GCN 성능:** (괄호 안은 fbank 특징을 사용한 결과)
  - CENet-GCN-6 (27.6K 파라미터, 2.55M 곱셈): 95.2% (95.7%) 정확도. CENet-6 대비 1.3%p 향상.
  - CENet-GCN-24 (55.6K 파라미터, 9.11M 곱셈): 96.5% (96.5%) 정확도. CENet-24 대비 0.9%p 향상.
  - CENet-GCN-40 (72.3K 파라미터, 16.78M 곱셈): 96.8% (97.0%) 정확도. CENet-40 대비 0.4%p 향상.
  - GCN의 성능 향상은 얕은 네트워크에서 더 크게 나타나고, 깊은 네트워크에서는 감소합니다.
- **GCN 삽입 단계 분석:** CENet-6 백본을 기준으로 GCN 모듈을 Stage-2에 추가했을 때 1.1%p 향상, 모든 단계(1, 2, 3)에 추가했을 때 1.3%p 향상 (95.2% 정확도)를 달성했습니다.
- **특징 맵 시각화:** GCN 모듈이 가장 식별적인 영역을 강조하고 음성/비음성 영역 간의 간격을 확대하는 데 도움이 됨을 보여줍니다.
- **ROC 곡선 분석:** CENet-GCN 모델이 CENet 모델보다 더 낮은 AUC(Area Under the Curve)를 가지며, 이는 GCN 모듈의 효과를 입증합니다.

## 🧠 Insights & Discussion

- CENet의 병목 구조와 잔여 연결은 적은 파라미터와 계산량으로도 높은 성능을 달성할 수 있어, 리소스가 제한된 KWS 환경에 매우 효율적임을 입증했습니다.
- GCN은 합성곱 특징 맵에서 장거리 의존성과 전역 컨텍스트를 효과적으로 포착하여 특징 학습을 향상시키고, 모든 CENet 변형 모델에서 일관된 정확도 향상을 가져왔습니다.
- 특히, GCN 모듈은 파라미터 수가 적은 소형 네트워크(예: CENet-6)에서 더 큰 상대적 성능 이득을 보여, 모델 크기 제약이 큰 시나리오에서 특히 유용합니다.
- 다양한 단계에 GCN 모듈을 통합함으로써 다단계(multi-level) 컨텍스트 정보를 활용하여 추가적인 성능 향상을 이끌어낼 수 있었습니다.
- 특징 맵 시각화를 통해 GCN이 음성 신호의 중요한 부분을 더 잘 구분하도록 돕는다는 것을 확인했으며, 이는 GCN이 특징의 식별 능력을 향상시켰음을 시사합니다.

## 📌 TL;DR

**문제:** 자원 제약적인 장치에서 높은 정확도를 가진 키워드 스포팅(KWS)을 달성하는 동시에, 기존 CNN의 단거리 컨텍스트 포착 한계와 RNN의 높은 계산 비용 및 지연 시간 문제를 해결하는 것.
**제안 방법:** 병목 구조를 가진 효율적인 합성곱 신경망(CENet)과 특징 맵의 장거리 의존성 및 컨텍스트를 인코딩하는 그래프 합성곱 신경망(GCN)을 결합한 CENet-GCN 모델을 제안.
**핵심 결과:** Google Speech Command Dataset에서 기존 최신 모델보다 훨씬 적은 파라미터와 낮은 연산량으로 뛰어난 정확도를 달성. GCN 모듈은 특히 소형 모델에서 큰 성능 향상을 가져왔으며, 특징 맵의 식별 능력을 효과적으로 증강시킴.
