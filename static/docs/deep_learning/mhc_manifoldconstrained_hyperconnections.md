# mHC: Manifold-Constrained Hyper-Connections

Zhenda Xie*, Yixuan Wei*, Huanqi Cao\*, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, Wenfeng Liang

## 🧩 Problem to Solve

지난 10년간 심층 신경망 아키텍처의 핵심 요소인 잔차 연결(residual connection)은 항등 매핑(identity mapping) 속성을 통해 안정성과 효율성을 제공해 왔습니다. 최근 Hyper-Connections(HC)는 잔차 스트림의 폭을 확장하고 연결 패턴을 다양화하여 상당한 성능 향상을 이끌어냈습니다. 그러나 이러한 다양화는 잔차 연결 본연의 항등 매핑 속성을 훼손하여 다음과 같은 문제를 야기합니다:

- **훈련 불안정성**: 제약 없는 HC는 신호의 무한 증폭 또는 소실을 초래하여 대규모 훈련 시 심각한 불안정성과 제한된 확장성을 발생시킵니다.
- **메모리 접근 오버헤드**: 확장된 잔차 스트림으로 인해 상당한 메모리 접근(I/O) 오버헤드가 발생하여 하드웨어 효율성이 저하됩니다.

## ✨ Key Contributions

- **다양체 제약 잔차 연결(Manifold-Constrained Hyper-Connections, mHC) 프레임워크 제안**: HC의 잔차 연결 공간을 특정 다양체(manifold)에 투영하여 항등 매핑 속성을 복원하고 효율성을 보장하는 일반적인 프레임워크를 제시합니다.
- **이중 확률 행렬(Doubly Stochastic Matrix) 제약**: 잔차 매핑 $H^{res}_l$을 이중 확률 행렬 다양체(비르코프 다면체, Birkhoff polytope)에 제약합니다. 이는 신호의 노름(norm)을 보존하고, 행렬 곱셈에 대해 닫혀있어 깊은 네트워크 전체에서 안정적인 신호 전파를 보장합니다.
- **입력/출력 매핑의 비음수성 제약**: 입력 매핑 $H^{pre}_l$과 출력 매핑 $H^{post}_l$에 Sigmoid 함수를 적용하여 비음수성을 확보하고 신호 소실을 방지합니다.
- **효율적인 인프라 최적화**: 커널 융합(kernel fusion), 선택적 재계산(selective recomputing), DualPipe 스케줄에서 통신 오버랩(overlapping communication)을 통해 대규모 훈련의 효율성을 극대화합니다.
- **우수한 성능 및 확장성 입증**: mHC는 HC의 성능 이점을 유지하면서 대규모 훈련에서 뛰어난 안정성과 확장성을 제공하며, 베이스라인 및 HC 대비 다운스트림 작업에서 성능 향상을 달성합니다.
- **낮은 오버헤드**: 확장률 $n=4$일 때, 대규모 훈련에서 단 6.7%의 미미한 추가 시간 오버헤드만을 발생시킵니다.

## 📎 Related Works

- **미세 설계 (Micro Design)**: 컨볼루션(Convolution)의 발전(Depthwise Separable, Grouped), 트랜스포머(Transformer)의 어텐션(Attention) 및 FFN(Feed-Forward Network), 효율적인 어텐션 변형(MQA, GQA, MLA), 희소 컴퓨팅 패러다임(MoE).
- **거시 설계 (Macro Design)**: ResNet 이후 DenseNet, FractalNet, DLA(Deep Layer Aggregation) 등 네트워크의 전역 토폴로지 및 블록 간 연결 구조를 다루는 연구들.
- **잔차 스트림 폭 확장**: Highway Transformer, Hyper-Connections (HC), Residual Matrix Transformer (RMT), MUDDFormer 등은 잔차 스트림의 폭을 확장하여 모델의 표현력을 높이려 했지만, 항등 매핑 속성을 훼손하고 메모리 오버헤드를 발생시키는 한계가 있었습니다. 본 연구는 이러한 HC의 한계를 보완합니다.

## 🛠️ Methodology

mHC는 HC의 잔차 매핑 $H^{res}_l$을 이중 확률 행렬 다양체에 제약하는 것을 핵심으로 합니다.

1. **잔차 매핑 $H^{res}_l$ 제약**:
   - $l$-번째 레이어의 잔차 매핑 $H^{res}_l$을 다음 조건을 만족하는 이중 확률 행렬로 제한합니다:
     $$ P*{M*{res}}(H^{res}_l) := \{ H^{res}\_l \in \mathbb{R}^{n \times n} \mid H^{res}\_l \mathbf{1}_{n} = \mathbf{1}_{n}, \mathbf{1}^{\top}_{n} H^{res}_l = \mathbf{1}^{\top}_{n}, H^{res}\_l \geq 0 \} $$
   - 여기서 $\mathbf{1}_{n}$은 모든 원소가 1인 $n$-차원 벡터입니다.
   - 이 제약은 노름 보존, 합성적 폐쇄성, 순열 행렬의 볼록 조합이라는 기하학적 해석을 제공하여 신호 전파의 안정성과 효율적인 특징 융합을 보장합니다.
2. **$H^{pre}_l$ 및 $H^{post}_l$ 제약**:
   - 입력 매핑 $H^{pre}_l$과 출력 매핑 $H^{post}_l$에 비음수성 제약을 적용합니다:
     $$ H^{pre}\_l = \sigma(\tilde{H}^{pre}\_l) $$
    $$ H^{post}\_l = 2\sigma(\tilde{H}^{post}\_l) $$
   - 여기서 $\sigma(\cdot)$는 Sigmoid 함수입니다.
3. **매개변수화 및 다양체 투영**:
   - 입력 은닉 행렬 $x_l \in \mathbb{R}^{n \times C}$를 $\vec{x}_l = \text{vec}(x_l) \in \mathbb{R}^{1 \times nC}$로 평탄화합니다.
   - $\text{RMSNorm}(\vec{x}_l)$을 사용하여 동적 매핑 $\tilde{H}^{pre}_l, \tilde{H}^{post}_l, \tilde{H}^{res}_l$을 계산하고, 학습 가능한 바이어스를 추가합니다.
   - 최종 $H^{res}_l$은 다음을 통해 얻습니다:
     $$ H^{res}\_l = \text{Sinkhorn-Knopp}(\tilde{H}^{res}\_l) $$
   - `Sinkhorn-Knopp` 연산은 $\text{exp}(\cdot)$를 통해 원소를 양수로 만든 후, 행과 열의 합이 1이 되도록 반복적인 정규화 과정을 수행합니다 (실험에서는 20회 반복).
4. **효율적인 인프라 설계**:
   - **커널 융합**: `RMSNorm` 연산 재정렬, 혼합 정밀도 전략 활용, 여러 연산을 통합 컴퓨트 커널로 융합하여 메모리 대역폭 병목 현상을 줄입니다 (TileLang 활용).
   - **재계산**: 순방향 후 mHC 커널의 중간 활성화 값을 폐기하고, 역방향에서 필요할 때 재계산하여 GPU 메모리 사용량을 절감합니다. 재계산 블록 크기 $L^{*}_{r} \approx \sqrt{nL / (n+2)}$는 파이프라인 스테이지 경계와 동기화됩니다.
   - **DualPipe에서 통신 오버랩**: `DualPipe` 스케줄을 확장하여 파이프라인 스테이지 경계에서 통신과 계산을 효과적으로 오버랩합니다. `FFN` 레이어의 $F_{post,res}$ 커널은 전용 고우선순위 컴퓨트 스트림에서 실행됩니다.

## 📊 Results

- **훈련 안정성**: 27B 모델 훈련 시, mHC는 HC에서 나타나는 훈련 불안정성을 완화하고, 베이스라인 대비 0.021의 최종 손실 감소를 달성합니다. 기울기 노름(gradient norm) 또한 HC에 비해 현저히 안정적인 프로파일을 보입니다.
- **다운스트림 성능**: 다양한 벤치마크(BBH, DROP, GSM8K, HellaSwag, MATH, MMLU, PIQA, TriviaQA)에서 mHC는 베이스라인을 일관되게 능가하며, HC보다도 우수한 성능을 보입니다. 특히 추론 능력 관련 벤치마크(BBH 2.1%, DROP 2.3%)에서 큰 개선을 이룹니다.
- **확장 실험**:
  - **컴퓨트 확장 (3B, 9B, 27B)**: mHC의 성능 이점은 높은 컴퓨트 예산에서도 견고하게 유지됩니다.
  - **토큰 확장 (3B, 1조 토큰)**: 훈련 토큰 수의 증가에 비례하여 mHC의 효과가 일관되게 유지됩니다.
- **오버헤드**: 확장률 $n=4$인 대규모 훈련에서 6.7%의 미미한 추가 훈련 시간 오버헤드만을 발생시킵니다.
- **전파 안정성 분석**: HC의 복합 매핑 Amax Gain Magnitude가 최대 3000에 달하는 반면, mHC는 이를 최대 1.6 수준으로 3자리수 이상 감소시켜 신호 전파 안정성을 크게 향상시킵니다.

## 🧠 Insights & Discussion

mHC는 Hyper-Connections(HC)의 성능 잠재력을 유지하면서, 잔차 연결의 항등 매핑 속성 상실로 인한 훈련 불안정성과 확장성 문제를 성공적으로 해결했습니다. 이중 확률 행렬 다양체에 대한 잔차 매핑의 제약은 신호 전파를 특징의 볼록 조합으로 변환하여, 신호의 폭발이나 소실 없이 안정적인 대규모 학습을 가능하게 합니다. 동시에 효율적인 인프라 수준의 최적화를 통해 이러한 안정성 및 성능 향상을 미미한 계산 오버헤드로 달성했습니다.

**한계 및 미래 방향**:

- 현재 연구는 이중 확률 행렬을 활용했지만, 향후 특정 학습 목표에 맞는 다양한 다양체 제약 조건을 탐색하여 모델의 유연성(plasticity)과 안정성(stability) 간의 최적의 균형을 찾는 연구가 가능합니다.
- mHC는 거시-아키텍처 설계에 대한 연구 관심을 재점화하고, 토폴로지 구조가 최적화 및 표현 학습에 미치는 영향을 심층적으로 이해함으로써 차세대 기반 모델(foundational models)의 발전에 새로운 길을 제시할 것으로 기대됩니다.

## 📌 TL;DR

HC(Hyper-Connections)는 잔차 스트림 폭 확장으로 성능을 높였지만, 불안정한 훈련과 확장성 문제가 있었습니다. 본 논문은 **mHC(Manifold-Constrained Hyper-Connections)**를 제안하여, HC의 잔차 매핑을 **이중 확률 행렬 다양체**에 제약함으로써 항등 매핑 속성을 복원합니다. 이는 **Sinkhorn-Knopp 알고리즘**과 효율적인 **인프라 최적화** (커널 융합, 재계산, 통신 오버랩)를 통해 구현됩니다. 결과적으로 mHC는 HC의 성능 이점을 유지하면서 **대규모 훈련의 안정성과 확장성을 크게 향상**시켰고, 미미한 오버헤드($n=4$일 때 6.7%)로 다운스트림 작업에서 우수한 성능을 달성했습니다.
