# Online Hybrid CTC/attention End-to-End Automatic Speech Recognition Architecture

Haoran Miao, Gaofeng Cheng, Pengyuan Zhang, Yonghong Yan (2023)

## 🧩 Problem to Solve

자동 음성 인식(ASR) 분야에서 End-to-End 아키텍처는 사전 학습된 정렬(alignments) 없이 음성을 텍스트로 변환하는 데 상당한 발전을 이루었다. 특히, CTC(Connectionist Temporal Classification)와 어텐션(attention) 메커니즘의 장점을 모두 활용하는 하이브리드 CTC/어텐션 기반 ASR 시스템은 기존의 심층 신경망(DNN)/은닉 마르코프 모델(HMM) 기반 ASR 시스템에 필적하는 성능을 보인다.

그러나 이러한 하이브리드 CTC/어텐션 시스템을 온라인 음성 인식(online speech recognition) 환경에 배포하는 것은 여전히 해결되지 않은 문제이다. 오프라인(offline) 환경을 위해 설계된 기존 하이브리드 CTC/어텐션 ASR 아키텍처에는 온라인 환경에서의 실시간 처리(real-time processing)를 방해하는 몇 가지 핵심 구성 요소들이 존재한다. 이 논문에서 제시하는 주요 난제는 다음과 같다.

* **어텐션 메커니즘(Attention mechanism)**: 기존의 어텐션 메커니즘(예: Global Attention)은 전체 입력 표현(input representations)에 대해 어텐션을 수행하므로, 전체 발화가 들어와야만 처리가 가능하다. 이는 온라인 스트리밍 환경에서 높은 지연시간(latency)을 유발한다.
* **CTC Prefix Score**: CTC Prefix Score는 주어진 접두사(prefix)를 공유하는 모든 레이블 시퀀스의 누적 확률로 정의되는데, 이를 계산하기 위해서는 완전한 발화 정보가 필요하다.
* **CTC와 어텐션 간의 비동기화된 예측(Unsynchronized predictions)**: 어텐션 기반 ASR 모델은 레이블 동기식(label-synchronous) 디코딩을 수행하는 반면, CTC 기반 ASR 모델은 프레임 동기식(frame-synchronous)이다. 이 둘의 예측이 엄격하게 동기화되지 않아 온라인 공동 디코딩(online joint decoding)이 어렵다.
* **양방향 인코더(Bidirectional encoder)**: BLSTM(Bidirectional Long Short-Term Memory) 네트워크는 입력 음성의 장기 의존성(long-term dependency)을 활용하지만, 양방향 특성으로 인해 미래 정보를 요구하므로 온라인 배포를 어렵게 만든다.

이러한 문제들을 해결함으로써, 논문의 목표는 기존 하이브리드 CTC/어텐션 ASR 아키텍처의 모든 오프라인 구성 요소를 상응하는 스트리밍(streaming) 구성 요소로 대체하여, CTC/어텐션 End-to-End ASR 아키텍처를 위한 완전한 온라인 솔루션을 제공하는 것이다. 이는 인간-컴퓨터 상호작용 서비스에서 실시간 요인(Real Time Factor, RTF)을 개선하고 지연 시간을 줄여 사용자 경험을 향상하는 데 중요한 기여를 한다.

## ✨ Key Contributions

이 논문의 중심적인 직관은 기존 오프라인 하이브리드 CTC/어텐션 ASR 아키텍처의 각 핵심 구성 요소를 온라인 스트리밍 환경에 적합하도록 재설계하는 것이다. 이를 통해 전체 ASR 시스템이 낮은 지연시간(low-latency)으로 실시간 동작할 수 있도록 한다. 주요 설계 아이디어 및 기여는 다음과 같다.

* **Stable Monotonic Chunk-wise Attention (sMoChA) 제안**: 기존 단조 청크 단위 어텐션(Monotonic Chunk-wise Attention, MoChA) 및 하드 단조 어텐션(Hard Monotonic Attention, HMA)의 수렴 문제(vanishing attention weights problem)를 해결하고, 학습-디코딩 불일치 문제(training-and-decoding mismatch problem)를 완화하여 안정적인 스트리밍 어텐션 메커니즘을 제공한다.
* **Monotonic Truncated Attention (MTA) 제안**: sMoChA를 더욱 단순화하고, 수신 필드(receptive field)를 확장하며, 학습-디코딩 불일치 문제를 효과적으로 해결한다. MTA는 제한된 청크 순서(chunk order)를 가진 sMoChA보다 더 넓은 수신 필드와 더 나은 모델링 능력을 가지며, 어텐션 가중치 계산 방식을 단순화한다.
* **Truncated CTC (T-CTC) Prefix Score 제안**: 전체 발화 정보가 필요한 기존 CTC Prefix Score 계산을 스트리밍 환경에 맞춰 최적화한다. CTC 출력 정보를 활용하여 입력 표현을 분할하고, 분할된 입력 표현에 대해서만 T-CTC Prefix Score를 계산하여 디코딩 속도를 가속화한다.
* **Dynamic Waiting Joint Decoding (DWJD) 알고리즘 설계**: CTC와 어텐션 간의 비동기화된 예측 문제를 해결하기 위해, 온라인 방식으로 CTC와 어텐션의 예측 점수를 동적으로 수집하고 조합하여 가설(hypotheses)을 생성하고 가지치기(pruning)한다.
* **Latency-Controlled Bidirectional Long Short-Term Memory (LC-BLSTM) 활용**: 기존의 양방향 인코더 네트워크를 스트리밍 환경에 맞게 대체하여, 제한된 미래 정보만을 사용하여 양방향 LSTM의 장점을 유지하면서 지연시간을 제어한다.

이러한 구성 요소들을 결합하여, 논문은 하이브리드 CTC/어텐션 End-to-End ASR 아키텍처를 위한 최초의 완전한 온라인 솔루션을 제공하며, 이는 기존 오프라인 모델에 필적하는 성능을 유지하면서 실시간 디코딩을 가능하게 한다.

## 📎 Related Works

논문은 End-to-End ASR 모델, 특히 CTC 기반과 어텐션 기반 아키텍처를 소개하고, 이 둘을 결합한 하이브리드 CTC/어텐션 아키텍처의 중요성을 강조한다. 또한, 온라인 ASR을 위한 기존 스트리밍 어텐션 방법론들을 제시하고 그 한계를 설명한다.

**기존 End-to-End ASR 아키텍처:**

* **DNN/HMM 하이브리드 시스템**: 음향 모델(AM), 발음 모델(PM), 언어 모델(LM) 등 여러 모듈로 구성되며 수동으로 설계되고 개별적으로 최적화된다. 언어학적 정보와 복잡한 디코더가 필요하여 새로운 언어에 대한 개발이 어렵다는 한계가 있다.
* **End-to-End ASR 모델**: DNN/HMM 시스템을 단일 심층 신경망 아키텍처로 단순화한다. 렉시콘(lexicon)이 필요 없으며 음소(graphemes) 또는 단어(words)를 직접 예측하여 디코딩 절차가 훨씬 간단하다.
  * **CTC 기반 ASR 아키텍처**: 음성 시퀀스에 대한 레이블 시퀀스의 조건부 확률을 직접 모델링하여 강제 정렬(forced alignment) 없이도 시퀀스 레이블링을 수행한다.
  * **어텐션 기반 ASR 아키텍처**: 인코더-디코더(encoder-decoder) 모델로, 디코더가 각 출력 레이블을 생성할 때 인코더 출력 중 관련 있는 부분에 동적으로 초점을 맞춘다.
  * **하이브리드 CTC/어텐션 아키텍처**: 어텐션 기반 인코더-디코더 네트워크에 CTC 목적 함수를 보조 작업(auxiliary task) 및 정규화(regularization)로 활용한다. 디코딩 시에는 빔 탐색(beam search) 알고리즘에서 디코더 점수(decoder scores)와 CTC 점수(CTC scores)를 결합하는 공동 디코딩(joint decoding) 방식을 사용한다. 특히 CTC 브랜치는 CTC Prefix Score를 사용하여 정렬이 좋지 않은 가설을 효율적으로 제외한다. CTC와 어텐션 메커니즘의 장점을 결합하여 단순 어텐션 기반 및 CTC 기반 모델보다 우수한 성능을 보인다.

**기존 스트리밍 어텐션 방법론 및 한계:**

온라인 ASR을 위한 스트리밍 어텐션 메커니즘을 구현하기 위해 Neural Transducer (NT), Hard Monotonic Attention (HMA), Monotonic Chunk-wise Attention (MoChA), Triggered Attention (TA) 등 여러 노력이 있었다.

* **Neural Transducer (NT) [22], [23]**: 고정된 수의 입력 프레임을 처리하고 다음 입력 청크가 도착하기 전에 가변적인 수의 레이블을 생성하는 제한된 시퀀스 스트리밍 어텐션 기반 모델이다. 그러나 NT는 학습 중에 청크 단위의 대략적인 정렬(coarse alignments)을 필요로 한다는 한계가 있다.
* **Hard Monotonic Attention (HMA) [24]**: 정렬이 필요 없는 스트리밍 어텐션 모델로, 어텐션이 단조로운(monotonic) 좌-우(left-to-right) 방식으로 고정된 수의 입력 표현을 자동으로 선택한다.
  * **한계**: (1) $E[z]$ 값이 지수적으로 감소하는 **어텐션 가중치 소멸 문제(vanishing attention weights problem)**가 발생하여 장거리 입력 표현에 대한 어텐션이 어렵다. (2) 학습 단계에서는 $E[z_{i,j}]$를 모든 입력 표현에 대해 계산하는 반면, 디코딩 단계에서는 이산 변수 $z_{i,j}$를 지시 함수(indicator function)로 대체하므로 **학습-디코딩 불일치(training-and-decoding mismatch)** 문제가 발생한다.
* **Monotonic Chunk-wise Attention (MoChA) [25]**: HMA를 확장하여 각 출력 레이블을 연속적인 입력 표현 청크(consecutive input representations chunk)에 정렬한다.
  * **한계**: HMA와 유사하게 $E[z]$의 **소멸 문제**를 겪으며, [27]에서는 수렴 문제, [28]에서는 긴 발화에 대한 확장성 문제를 발견했다. 또한, HMA와 마찬가지로 **학습-디코딩 불일치 문제**가 성능 저하를 야기한다는 연구 [36]가 있었다.
* **Triggered Attention (TA) [26]**: CTC 기반 네트워크를 활용하여 발화를 동적으로 분할하고, 디코더가 레이블을 생성함에 따라 증분 입력 표현(incremental input representations)에 대해 어텐션을 수행한다.

**기존 접근 방식과의 차별점:**

본 논문은 기존 CTC/어텐션 아키텍처의 온라인 배포를 방해하는 네 가지 주요 난제(글로벌 어텐션, CTC Prefix Score, 비동기화된 예측, 양방향 인코더)에 대한 포괄적인 해결책을 제시한다. 특히, 기존 스트리밍 어텐션 방식(HMA, MoChA)의 단점(어텐션 가중치 소멸, 학습-디코딩 불일치)을 개선한 **sMoChA**와 **MTA**를 제안하고, CTC Prefix Score를 스트리밍에 맞춘 **T-CTC Prefix Score**를 도입하며, 비동기화된 예측 문제를 해결하는 **DWJD** 알고리즘을 설계한다. 또한, 양방향 인코더를 **LC-BLSTM**으로 대체한다. 이는 기존의 부분적인 온라인 솔루션들과 달리, 하이브리드 CTC/어텐션 End-to-End ASR 아키텍처를 위한 **최초의 완전한 온라인 솔루션**이라는 점에서 차별화된다.

## 🛠️ Methodology

본 논문은 온라인 하이브리드 CTC/어텐션 End-to-End ASR 아키텍처를 개발하기 위해 여러 혁신적인 알고리즘을 제안한다. 전체 시스템은 스트리밍 인코더, 스트리밍 어텐션 기반 디코더, CTC 브랜치, 그리고 동적 대기 공동 디코딩(DWJD) 알고리즘으로 구성된다.

### 전체 파이프라인 또는 시스템 구조

제안된 온라인 하이브리드 CTC/어텐션 End-to-End ASR 아키텍처는 Fig. 1에 나타나 있다. 이 아키텍처는 기존 오프라인 CTC/어텐션 모델의 각 구성 요소를 온라인 스트리밍에 적합하도록 대체한다.

1. **스트리밍 인코더**: 입력 오디오 프레임 $X$를 받아 입력 표현 벡터 $H = (h_1, \ldots, h_T)$로 변환한다. 기존의 BLSTM 대신 LC-BLSTM (Latency-Controlled BLSTM) 또는 Uni-LSTM을 사용하여 낮은 지연시간을 유지한다.
2. **스트리밍 어텐션 메커니즘**: 인코더의 출력인 $H$와 디코더의 이전 상태 $q_{i-1}$ 및 이전 레이블 $y_{i-1}$을 기반으로 현재 레이블 $y_i$를 예측하기 위한 레이블 단위 표현 벡터 $r_i$를 계산한다. 본 논문에서는 Stable Monotonic Chunk-wise Attention (sMoChA)과 Monotonic Truncated Attention (MTA)을 제안하여 글로벌 어텐션을 대체한다.
3. **CTC 브랜치**: 인코더와 네트워크를 공유하며, 추가적인 분류 레이어(classification layer)를 통해 레이블 또는 `⟨b⟩`(blank) 레이블을 예측한다. CTC Prefix Score 계산을 스트리밍화하기 위해 Truncated CTC (T-CTC) Prefix Score를 사용한다.
4. **Dynamic Waiting Joint Decoding (DWJD) 알고리즘**: 어텐션 기반 디코더($S_{att}$)와 CTC 브랜치($S_{tctc}$)의 예측 점수를 동적으로 수집하여 온라인 방식으로 가설을 결합하고 가지치기(pruning)한다. 외부 RNN 언어 모델($S_{lm}$)도 활용된다.

### 각 주요 구성 요소 및 역할

#### 1. Stable Monotonic Chunk-wise Attention (sMoChA)

기존 HMA 및 MoChA의 어텐션 가중치 소멸(vanishing attention weights) 문제와 학습-디코딩 불일치 문제를 해결하기 위해 고안되었다.

**1.1. 어텐션 가중치 소멸 문제 해결:**
sMoChA는 MoChA와 달리 인접한 $E[z_{i-1,k}]$에 의존하지 않고, 각 레이블 출력 시점에 현재 입력 표현 $h_j$까지의 확률만 고려하여 $E[z_{i,j}]$를 계산한다.

$$
E[z_{i,j}] = p_{i,j} \prod_{k=1}^{j-1} (1-p_{i,k}) \quad (23)
$$

이는 재귀적으로 다음과 같이 표현될 수 있다.

$$
E[z_{i,j}] = \frac{p_{i,j}}{p_{i,j-1}} \cdot (1-p_{i,j-1}) E[z_{i,j-1}] \quad (24)
$$

여기서 $p_{i,j}$는 시그모이드(Sigmoid) 함수 $\sigma(\cdot)$를 통해 계산되는 선택 확률이며, $p_{i,j} = \sigma(e_{i,j})$이다. $e_{i,j}$는 이전 디코더 상태 $q_{i-1}$과 현재 인코더 출력 $h_j$의 유사도를 나타내는 에너지 함수 $Energy(q_{i-1}, h_j)$에 의해 결정된다. HMA와 MoChA에서와 같이, Energy 함수는 학습 가능한 스케일 및 오프셋 파라미터 $g, r$을 포함하여, $e_{i,j} = g \frac{v^\top}{||v||} \tanh(W_1 q_{i-1} + W_2 h_j + b) + r$로 정의된다. 여기서 바이어스 $r$을 음수 값(예: $r=-4$)으로 초기화하여 $(1-p_{i,j})$의 평균이 1에 가깝게 유지되도록 하여 $E[z_{i,j}]$가 지수적으로 0으로 소멸되는 것을 방지한다.

**1.2. 학습-디코딩 불일치 문제 완화:**
디코딩 단계에서 sMoChA는 단일 청크 내에서만 정렬을 수행하지만, 학습 단계에서는 모든 입력 표현에 대해 $r_i$가 계산되어 불일치가 발생한다. 이를 완화하기 위해 "higher order decoding chunks mode"를 제안한다. 디코딩 단계에서 $n$개의 연속적인 청크를 고려하여 레이블 단위 표현 벡터 $r_i$를 계산한다.

$$
E[\hat{z}_{i,k}] = \frac{E[z_{i,k}]}{\sum_{l=t_i-n+1}^{t_i} E[z_{i,l}]} \quad (25)
$$

$$
\alpha_{i,j} = \sum_{k=j}^{j+w-1} E[\hat{z}_{i,k}] \frac{\exp(u_{i,j})}{\sum_{l=k-w+1}^{k} \exp(u_{i,l})} \quad (26)
$$

$$
r_i = \sum_{j=t_i-w+n}^{t_i} \alpha_{i,j} h_j \quad (27)
$$

여기서 $t_i$는 선택된 끝점(end-point), $n$은 청크 순서(chunk order, 연속적인 디코딩 청크의 수), $w$는 청크 너비(chunk width)를 나타낸다. $E[\hat{z}_{i,k}]$는 연속적인 청크에 걸쳐 정규화된 $E[z_{i,k}]$ 값이다. $u_{i,j}$는 (18)에서 계산된 pre-softmax 활성화 값이며, $\alpha_{i,j}$는 청크 내 어텐션 가중치이다. $n$이 증가함에 따라 학습 및 디코딩 단계 간의 불일치가 줄어든다. $n=1$일 때 MoChA와 동일하게 $O(T+wL)$의 디코딩 계산 복잡도를 가지며, $n>1$일 경우 $O(TL+wL+nL)$로 증가한다. $n \to \infty$일 경우, 모든 과거 입력 표현 벡터를 포함하게 된다.

#### 2. Monotonic Truncated Attention (MTA)

sMoChA의 지연시간-계산 비용 트레이드오프(latency-computation cost tradeoff)를 개선하고 학습-추론 불일치 문제를 더 잘 해결하기 위해 MTA를 제안한다. MTA는 잘린 과거 입력 표현(truncated historical input representations)에 대해 어텐션을 수행한다.

**2.1. 디코딩 단계:**
이전 끝점 $t_{i-1}$에서 검색을 시작하여 새로운 끝점 $t_i$를 선택하고, 시퀀스 입력 표현을 자른다. 그리고 잘린 시퀀스 입력 표현에 대해 정렬을 수행한다.

$$
e_{i,j} = Energy(q_{i-1}, h_j) \quad (28)
$$

$$
p_{i,j} = Sigmoid(e_{i,j}) \quad (29)
$$

$$
z_{i,j} = I(p_{i,j} > 0.5 \land j \ge t_{i-1}) \quad (30)
$$

$$
\alpha_{i,j} = p_{i,j} \prod_{k=1}^{j-1} (1-p_{i,k}) \quad (31)
$$

$p_{i,j}$는 절단 확률(truncation probability)이며, $z_{i,j}$는 이산 절단 결정(discrete truncate or do not truncate decision)이다. $p_{i,j} > 0.5$이고 $j \ge t_{i-1}$일 때 어텐션 가중치 계산을 중단하고 현재 끝점 $t_i$를 $j$로 설정한다. 이로써 끝점은 $t_i \ge t_{i-1}$을 만족하며 일관되게 전진한다. 레이블 단위 표현 벡터 $r_i$는 다음과 같이 계산된다.

$$
r_i = \sum_{j=1}^{t_i} \alpha_{i,j} h_j \quad (32)
$$

MTA는 모든 과거 입력 표현 벡터를 포함하므로, HMA, MoChA 및 제한된 청크 순서를 가진 sMoChA보다 더 넓은 수신 필드와 더 나은 모델링 능력을 가진다. 또한, 청크 에너지 함수(Chunk Energy function)를 사용하여 어텐션 가중치를 재계산하는 대신 절단 확률을 직접 사용하여 어텐션 가중치 계산 방법을 단순화한다. 디코딩 계산 복잡도는 $O(TL)$이다.

**2.2. 학습 단계:**
학습 단계에서는 (30)의 지시 함수를 사용하지 않고, 레이블 단위 표현 벡터 $r_i$를 다음과 같이 계산한다.

$$
r_i = \sum_{j=1}^{T} \alpha_{i,j} h_j \quad (33)
$$

Energy 함수로는 sMoChA와 동일하게 (17)을 사용하고, 바이어스 $r$을 음수 값(예: $r=-4$)으로 초기화하여 어텐션 가중치 소멸을 방지한다. (31)과 (33)에 따르면, 학습 단계에서는 모든 입력 표현 벡터에 대해 정렬을 수행하지만, 끝점에서의 $p_{i,j}$가 급격하게 선택되므로 끝점 이후의 어텐션 가중치 $\alpha_{i,j}$는 0에 가깝다. 따라서 디코딩 단계에서 미래 정보를 사용할 수 없을 때 인식 정확도에 미치는 영향이 거의 없으므로, 학습-디코딩 불일치 문제를 완화한다.

#### 3. Truncated CTC (T-CTC) Prefix Score

CTC Prefix Score는 전체 발화에 대해 계산되므로 온라인 시나리오에 부적합하다. CTC 기반 모델의 피크 후방 확률(peaky posterior properties)을 고려할 때, $p(y_n|h_j)$는 CTC 기반 모델이 $y_n$을 처음 예측하는 시점을 제외하고는 거의 0이다. 따라서, $j > t_n$일 때 $p_{ctc}(l|H_{1:j}) \approx 0$이 되는 끝점 $t_n$을 찾을 수 있다. 이 속성을 기반으로, $l$의 CTC Prefix Score를 추정하기 위해 $t_n$개의 입력 표현 벡터만 사용한다. 제안된 T-CTC Prefix Score는 다음과 같이 계산된다.

$$
S_{tctc} = \log \sum_{j=1}^{t_n} p_{ctc}(l|H_{1:j}) \quad (36)
$$

**T-CTC Prefix Score 알고리즘 (Algorithm 1):**
이 알고리즘은 CTC 브랜치가 끝점 $t_n$을 결정하고 $H_{1:t_n}$에 대한 $l$의 T-CTC Prefix Score를 계산하는 방법을 설명한다.

* **초기화 및 재귀**: `⟨sos⟩` (start-of-sequence) 및 `⟨eos⟩` (end-of-sequence) 레이블을 포함한 부분 가설 $l=(y_1, \ldots, y_n)$에 대해 전방 확률(forward probabilities) $\gamma_j^n(l)$ (비-블랭크(non-blank)로 끝나는 경우)과 $\gamma_j^b(l)$ (블랭크로 끝나는 경우)를 계산한다.
* **끝점 결정**: CTC Prefix Score 누적 확률 $\Psi$가 갱신되는 과정에서 $j > t_{n-1}$ (이전 끝점 이후) 이고 $\Phi \cdot p(y_n|h_j) < \theta$ (특정 레이블 $y_n$의 확률 기여가 임계값 $\theta$보다 작을 때)이면 재귀 단계를 종료하고 현재 $j$를 끝점 $t_n$으로 설정한다. $\theta$는 작은 값(예: $10^{-8}$)으로 설정된다.
* **온라인 처리**: 알고리즘은 가설이 확장됨에 따라 끝점이 전진하도록 하여 CTC 기반 모델의 단조 정렬(monotonic alignments)과 일관성을 유지한다. VAD(Voice Activity Detection)에 의해 결정되는 최대 입력 표현 벡터 수 $T_{max}$를 초과하지 않는다.
* **계산 비용 감소**: 부분 가설의 끝점 $t_n$에 대해 T-CTC Prefix Score 알고리즘은 계산 비용을 $t_n/T_{max}$로 줄인다. 빔 탐색(beam search) 중 대부분의 부분 가설이 가지치기되기 때문에 공동 CTC/어텐션 디코딩 프로세스를 가속화한다.

#### 4. Dynamic Waiting Joint Decoding (DWJD) 알고리즘

스트리밍 어텐션 모델과 T-CTC Prefix Score 알고리즘이 제시되었지만, 인코더, 어텐션 기반 디코더, CTC 브랜치가 온라인 디코딩 단계에서 어떻게 협력하여 가설을 생성하고 가지치기할지는 여전히 문제이다. 어텐션 기반 디코더와 CTC 브랜치는 동기화된 예측을 생성하지 않기 때문에, 동일한 가설의 점수를 다른 시점에서 계산해야 한다 (Fig. 4 참고). DWJD 알고리즘은 온라인 설정에서 CTC/어텐션의 비동기화 예측 문제를 해결하기 위해, 어텐션 기반 디코더와 CTC 브랜치의 예측 점수를 동적으로 수집한다.

**DWJD 알고리즘 (Algorithm 2):**

* **스트리밍 어텐션 처리**: 스트리밍 어텐션 모델은 먼저 끝점 $t_{att}$를 검색하고, $t_{att}$개의 입력 표현 벡터를 기반으로 레이블 단위 표현 벡터 $r_i$를 계산한다. 이 과정에서 $j > t_{enc}$ (현재 인코더 출력보다 더 많은 프레임이 필요할 때)이면 인코더가 추가 입력을 생성할 때까지 디코딩을 일시 중단한다 (라인 4-20).
* **디코더 예측**: 어텐션 끝점 $t_{att}$를 기반으로 디코더는 새로운 레이블 $y_i$와 해당 디코더 점수 $S_{att}$를 예측한다 (라인 21-22).
* **T-CTC Prefix Score 계산**: CTC 브랜치는 끝점 $t_{ctc}$를 검색하고, $t_{ctc}$개의 입력 표현 벡터를 조건으로 새로운 가설의 T-CTC Prefix Score $S_{tctc}$를 계산한다 (라인 23-34). 여기서도 인코더 출력이 부족하면 대기한다.
* **가설 가지치기**: $S_{att}$와 $S_{tctc}$의 조합(외부 언어 모델 $S_{lm}$과 함께)으로 가설을 가지치기한다.
* **동적 대기**: 어텐션 기반 디코더와 CTC 브랜치가 다른 시점에서 끝점을 예측하거나, 다른 가설이 다른 끝점에 해당할 때, 뒤쪽 끝점에 해당하는 가설은 인코더가 충분한 출력을 제공할 때까지 일시 중단된다.

**끝점 감지 기준(End Detection Criteria):**
온라인 공동 디코딩에서 빔 탐색을 적절하게 종료하는 것이 중요하다. T-CTC Prefix Score는 빔 탐색 중 조기 종료 가설(premature ending hypotheses)을 제외하지 못할 수 있다. 또한, 짧은 가설의 점수가 더 높은 경향이 있어 조기 종료 문제가 발생할 수 있다. 이를 해결하기 위해 다음과 같은 새로운 끝점 감지 기준을 설계한다.

$$
\sum_{m=1}^{M} I \left[ \max_{l \in \Omega: |l|=n} S(l) - \max_{l' \in \Omega: |l'|=n-m} S(l') < D_{end} \right] = M \quad (37)
$$

현재 완전한 가설의 길이가 $n$일 때, 이 방정식은 현재 완전한 가설의 최고 점수가 $M$단계 이전의 더 짧은 완전한 가설의 최고 점수보다 $D_{end}$ (예: -10) 이상 현저히 낮을 경우 참이 된다. 이는 더 높은 점수를 가진 더 긴 가설을 찾을 가능성이 거의 없음을 의미한다. 본 논문에서는 $M=3$으로 설정한다. 또한, 최종 인식된 가설에 대해서는 수집된 종료 가설의 T-CTC Prefix Score를 CTC Prefix Score로 대체하여 조기 종료 가설을 제거할 것을 제안한다.

#### 5. 저지연 인코더 (Low-latency Encoder)

* **VGGNet-style CNNs**: 인코더 프런트엔드로 VGGNet 스타일 CNN 블록을 사용한다. 각 CNN 블록은 두 개의 CNN 레이어와 하나의 Max-pooling 레이어로 구성된다.
* **LC-BLSTM (Latency-Controlled BLSTM)**: 온라인 시스템을 위해 BLSTM을 대체한다. LC-BLSTM은 세그먼트화된 프레임 청크(segmented frame chunks)에서 작동한다. 각 청크는 $N_c$개의 현재 프레임과 $N_r$개의 미래 프레임을 포함하며, LC-BLSTM은 각 시점에서 $N_c$ 프레임씩 이동한다. LC-BLSTM의 지연시간은 $N_r$로 제한된다. Uni-LSTM도 저지연 인코더로 비교된다.

## 📊 Results

본 논문은 LibriSpeech 영어 데이터셋(약 960시간)과 HKUST 만다린(Mandarin) 대화형 전화 데이터셋(약 200시간)을 사용하여 제안된 온라인 방법을 평가하였다. HKUST 데이터셋에는 속도 섭동(speed perturbation)이 적용되었다. 성능 지표로는 LibriSpeech에 대해 WER(Word Error Rate)을, HKUST에 대해 CER(Character Error Rate)을 사용하였다. 모든 ASR 모델은 ESPNet 툴킷을 기반으로 구축되었다.

**실험 설정:**

* **입력**: 83차원 특징(80차원 필터 뱅크, 피치, 델타-피치, NC-CFs)을 사용하며, 25ms 윈도우와 10ms 시프트로 계산되었다.
* **출력**: LibriSpeech의 경우 영어 문자 및 PASM(Pronunciation-Assisted Sub-word Modeling) 단위(200개), HKUST의 경우 3655개 출력 세트(3623개 중국어 문자, 26개 영어 문자, 6개 비언어 심볼)를 사용하였다.
* **인코더/디코더**: 인코더 프런트엔드로 VGGNet 스타일 CNN 블록을 사용하고, 백엔드로는 오프라인 시스템에 Multi-layer BLSTM, 온라인 시스템에 Multi-layer Uni-LSTM 또는 LC-BLSTM을 사용하였다. 디코더는 Uni-LSTM을 사용하였다.
* **외부 언어 모델**: 1/2-레이어 Uni-LSTM을 사용하며, CTC/어텐션 ASR 모델과 별도로 학습되었다. LibriSpeech의 경우 multi-level language model decoding이 적용되었다.
* **훈련/디코딩 파라미터**: Table I에 상세히 기술되어 있다 (예: CTC Weight for Training $\lambda=0.5$, Batch Size=20, Epoch=10~15 등).

### 1. 스트리밍 어텐션 결과

* **어텐션 가중치 비교 (Fig. 5)**:
  * HMA 및 MoChA는 LibriSpeech의 긴 발화(long utterances)에 대해 단조 정렬을 학습하지 못하고 어텐션 가중치 소멸 문제를 겪었다 (Fig. 5(a)-(d)). MoChA는 짧은 발화에 대해서는 단조 정렬을 학습할 수 있었다 (Fig. 5(g)-(i)).
  * 제안된 sMoChA와 MTA는 긴 발화에서도 안정적으로 거의 단조로운 정렬을 학습했으며, 어텐션 가중치 소멸 문제를 성공적으로 해결했다 (Fig. 5(e)-(f)).

* **LibriSpeech 성능 (Table II)**:
  * PASM 기반 모델이 문자 기반 모델보다 우수했고, VBL(VGG-BLSTM-Large) 모델이 VBS(VGG-BLSTM-Small) 모델보다 우수하여 발음 정보와 큰 모델이 성능에 기여함을 보였다.
  * **sMoChA**: HMA 및 MoChA보다 우수한 성능을 보였다. 청크 순서($n$)가 증가할수록 sMoChA의 디코딩 정확도가 일관되게 향상되었으며, VBS 모델에서 개선 폭이 더 컸다. 이는 작은 모델이 학습-디코딩 불일치에 더 취약했기 때문으로 해석된다.
  * **MTA**: 청크 너비나 청크 순서 개념 없이, 스트리밍 어텐션 방법 중 거의 최고의 성능을 달성했으며, 오프라인 LoAA(Location-aware attention) 기반 모델과 비교할 만한 성능을 보였다.

* **HKUST 성능 (Table III)**:
  * LibriSpeech와 유사하게 sMoChA가 HMA 및 MoChA보다 우수했다.
  * 청크 순서가 높을수록 디코딩 성능이 향상되었다.
  * MTA가 제안된 스트리밍 어텐션 방법 중 가장 우수한 성능을 보였으며, LoAA와 동일한 수준의 성능을 달성했다.

결론적으로, 제안된 스트리밍 MTA 방법은 ASR 정확도 저하를 거의 유발하지 않으며, 다양한 언어(영어, 중국어)에 걸쳐 견고함을 입증했다.

### 2. T-CTC Prefix Score 결과

* **T-CTC와 CTC Prefix Score 비교 (Table IV)**:
  * MTA를 사용하여 어텐션을 스트리밍하고, T-CTC를 사용하여 CTC Prefix Score 계산을 스트리밍하며, DWJD 알고리즘을 사용하여 공동 CTC/어텐션 디코딩을 스트리밍했을 때, ASR 정확도에 매우 미미한 저하만을 유발했다. 이는 T-CTC Prefix Score가 오프라인 CTC Prefix Score를 잘 근사함을 보여준다.

### 3. 저지연 인코더 결과

* **다양한 인코더 유형별 온라인 End-to-End 모델 비교 (Table V)**:
  * 제안된 온라인 CTC/어텐션 기반 모델은 단순 Uni-LSTM CTC End-to-End 모델보다 일관된 정확도 향상을 달성했다.
  * LC-BLSTM 기반 인코더가 Uni-LSTM 기반 인코더보다 우수했다. 이는 LC-BLSTM이 제한된 미래 정보를 활용하여 성능을 개선할 수 있기 때문이다.
  * LC-BLSTM 인코더에서 $N_r$ (미래 프레임 수)를 증가시킬 때 WER이 크게 개선되었지만, $N_c$ (현재 프레임 수)를 증가시킬 때는 미미한 개선만을 보였다. 이는 LC-BLSTM이 청크를 넘어선 과거 정보를 활용할 수 있지만, 미래 컨텍스트는 $N_r$에 의해 제한되기 때문이다.
  * 320ms의 지연시간을 가진 제안된 온라인 하이브리드 CTC/어텐션 모델은 LibriSpeech에서 4.2% / 13.3% WER, HKUST에서 29.4% / 27.8% CER을 달성하여, 오프라인 기준선 대비 허용 가능한 성능 저하를 보였다.
  * 이 온라인 모델은 제한된 미래 컨텍스트만 필요하므로, 지연시간을 발화 수준에서 프레임 수준으로 크게 줄여 인간-컴퓨터 상호작용의 사용자 경험을 향상시킨다.

### 4. 디코딩 속도 결과

* **RTF (Real Time Factor) vs. WER/CER (Fig. 6 & 7)**:
  * MTA와 T-CTC Prefix Score는 특히 빔 사이즈가 5보다 클 때 디코딩 속도를 가속화했다. 이는 이들이 더 적은 계산 비용을 가지기 때문이다.
  * 온라인 시스템(빨간색 실선)은 오프라인 기준선(검은색 실선)보다 일관되게 빨랐다. 모델 크기가 거의 동일함에도 불구하고 온라인 시스템은 약 1.5배 빨랐다. 이는 온라인 시스템이 사용자가 말하기 시작할 때부터 음성을 처리할 수 있어 디코딩 지연시간을 줄이고 CPU 활용도를 높이기 때문이다.

## 🧠 Insights & Discussion

이 논문은 하이브리드 CTC/어텐션 End-to-End ASR 아키텍처를 온라인 환경에 성공적으로 적용하기 위한 포괄적인 프레임워크를 제시한다.

**논문에서 뒷받침되는 강점:**

* **최초의 완전한 온라인 솔루션**: 기존 연구들이 스트리밍 어텐션과 같은 개별 구성 요소에 집중했던 것과 달리, 이 논문은 기존 CTC/어텐션 모델의 모든 오프라인 구성 요소(글로벌 어텐션, CTC Prefix Score 계산, 비동기화 예측, 양방향 인코더)를 온라인에 적합한 스트리밍 구성 요소로 대체하여, CTC/어텐션 End-to-End ASR을 위한 최초의 "full-scale" 온라인 솔루션을 제공한다.
* **오프라인에 필적하는 성능**: 제안된 온라인 시스템은 오프라인 CTC/어텐션 기준선과 비교했을 때 매우 적은 성능 저하로 비슷한 수준의 ASR 정확도를 달성한다. 특히 MTA, T-CTC Prefix Score, DWJD 알고리즘은 인식 정확도를 거의 저하시키지 않았다.
* **현저한 지연시간 감소 및 디코딩 속도 향상**: 기존 오프라인 시스템이 전체 발화를 필요로 하는 반면, 제안된 온라인 시스템은 제한된 미래 컨텍스트만을 요구하여 지연시간을 발화 수준에서 프레임 수준으로 크게 줄인다. 또한, 사용자가 말하기 시작할 때부터 음성을 처리할 수 있어 CPU 활용도를 높이고 디코딩 속도를 약 1.5배 가속화한다.
* **다국어 및 다양한 모델 크기에 대한 견고성**: 제안된 MTA 방법은 LibriSpeech(영어) 및 HKUST(만다린) 두 가지 언어 및 다양한 모델 크기(Large, Small)에서 일관된 성능을 보여, 다양한 환경에 대한 방법론의 견고함을 입증한다.

**한계, 가정 또는 미해결 질문:**

* **저지연 인코더에 의한 성능 저하**: 논문에서 MTA, T-CTC Prefix Score, DWJD 알고리즘은 인식 정확도에 거의 영향을 미치지 않았다고 명시하고 있다. 그러나 Uni-LSTM 및 LC-BLSTM과 같은 저지연 인코더의 적용이 인식 정확도를 하락시키는 주된 원인이었다고 언급된다. 이는 미래 정보를 제한적으로만 활용할 수밖에 없는 온라인 환경의 근본적인 한계로 보인다.
* **RNN 기반 아키텍처의 병렬화 문제**: 현재 Recurrent Neural Network (RNN) 기반 아키텍처는 학습 및 디코딩 프로세스의 병렬화가 어렵다는 한계가 있다. 이로 인해 잠재적인 성능 향상 및 효율성 개선이 제한될 수 있다.
* **DWJD의 복잡성**: DWJD 알고리즘은 CTC와 어텐션 브랜치 간의 비동기화를 관리하기 위해 인코더 출력의 동적 대기(dynamic waiting)를 도입한다. 이는 복잡한 로직을 수반하며, 최적의 동기화 및 대기 전략에 대한 추가적인 연구가 필요할 수 있다.
* **끝점 감지 기준의 튜닝**: 논문은 조기 종료 가설 문제를 해결하기 위해 새로운 끝점 감지 기준(Equation 37)을 제안하며 $M=3, D_{end}=-10$과 같은 특정 하이퍼파라미터를 사용한다. 이러한 값들의 일반성(generality)과 최적화에 대한 더 깊은 분석이 필요할 수 있다.

**논문에 근거한 간략한 비판적 해석 및 논의사항:**

이 연구는 End-to-End ASR 시스템을 실시간 애플리케이션에 적용하는 데 있어 중요한 이정표를 제시한다. 특히, 기존 CTC/어텐션 모델의 핵심 병목 현상들을 체계적으로 식별하고 효과적인 스트리밍 대안들을 제시함으로써 실제 배포 가능성을 크게 높였다. MTA와 T-CTC Prefix Score, DWJD 알고리즘은 성능 저하 없이 오프라인 모델의 이점을 온라인으로 가져오는 데 성공적인 접근 방식임을 보여준다.

그러나 저지연 인코더로 인한 정확도 손실은 여전히 중요한 과제로 남아있다. 현재 연구는 LC-BLSTM을 사용하여 제한된 미래 컨텍스트를 활용하는 방식을 채택했지만, 이는 본질적으로 인코더가 전체 발화에서 얻을 수 있는 장기적인 컨텍스트를 포기하는 것을 의미한다. 이를 해결하기 위해 논문에서 제시된 교사-학생 학습(teacher-student learning) 접근 방식은 유망한 방향이 될 수 있으며, 오프라인 "교사" 모델의 풍부한 지식을 온라인 "학생" 모델로 전이하여 성능을 유지하면서 지연시간을 줄일 수 있을 것이다. 또한, RNN의 병렬화 한계를 극복하기 위해 Transformer와 같은 병렬화 친화적인 아키텍처를 온라인 설정에 맞게 개조하는 연구는 향후 중요한 방향이 될 것으로 보인다.

## 📌 TL;DR

이 논문은 하이브리드 CTC/어텐션 End-to-End ASR 아키텍처를 온라인 환경에 적합하도록 전면적으로 재설계한 최초의 완전한 온라인 솔루션을 제시한다. 주요 기여 사항은 다음과 같다: (1) 기존 단조 어텐션의 소멸 및 불일치 문제를 해결한 **Stable Monotonic Chunk-wise Attention (sMoChA)**과 이를 더욱 단순화하고 성능을 개선한 **Monotonic Truncated Attention (MTA)**, (2) 전체 발화가 필요한 CTC Prefix Score를 스트리밍화한 **Truncated CTC (T-CTC) Prefix Score**, (3) CTC와 어텐션의 비동기화 예측을 동적으로 관리하는 **Dynamic Waiting Joint Decoding (DWJD)** 알고리즘, (4) 기존 양방향 인코더를 저지연 **Latency-Controlled BLSTM (LC-BLSTM)**으로 대체한 것이다.

실험 결과, 제안된 온라인 시스템은 오프라인 CTC/어텐션 기준선에 필적하는 ASR 정확도를 유지하면서도, 지연시간을 발화 수준에서 프레임 수준으로 현저히 감소시키고 디코딩 속도를 약 1.5배 가속화함을 입증하였다. 특히 MTA, T-CTC, DWJD는 인식 정확도 저하 없이 효율적인 온라인 처리를 가능하게 했다. 이 연구는 인간-컴퓨터 상호작용 서비스와 같이 실시간 ASR이 필수적인 분야에서 사용자 경험을 크게 향상시키고, 향후 저지연 ASR 시스템의 개발 및 병렬화된 신경망 아키텍처 탐색, 교사-학생 학습을 통한 성능 유지 연구에 중요한 기반을 제공할 가능성이 높다.
