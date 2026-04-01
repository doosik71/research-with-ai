# Neural Architecture Search For Keyword Spotting

Tong Mo, Yakun Yu, Mohammad Salameh, Di Niu, Shangling Jui

## 🧩 Problem to Solve

스마트 기기에서 음성으로 제어하는 키워드 스포팅(Keyword Spotting, KWS) 시스템은 사용자 프라이버시 보호 및 데이터 전송 중 데이터 유출 방지를 위해 온디바이스(on-device) 배포가 중요합니다. 하지만 이러한 장치는 리소스 제약이 심하여 작은 메모리 공간을 유지하면서도 높은 정확도를 달성해야 하는 문제가 있습니다. 기존의 딥러닝 기반 KWS 모델은 수동으로 설계된 CNN 아키텍처에 의존해 왔으며, 이는 최적의 아키텍처를 찾는 데 많은 시간과 노력이 소요됩니다.

## ✨ Key Contributions

- **KWS에 차등 가능한 신경망 아키텍처 탐색(DARTS) 적용:** 효율적인 그래디언트 기반 NAS(Neural Architecture Search) 기법인 DARTS를 KWS 작업에 성공적으로 적용했습니다.
- **새로운 최첨단 성능 달성:** Google Speech Commands Dataset (12-class 분류 설정)에서 97% 이상의 정확도를 달성하여 기존 KWS 모델들의 성능을 뛰어넘었습니다.
- **효율적인 아키텍처 탐색:** 비교적 적은 GPU 시간(NAS1: 0.58 GPU day, NAS2: 0.29 GPU day)으로 고성능 KWS 아키텍처를 찾아냈습니다.
- **다양한 연산자 세트에 대한 강건성 입증:** 분리 가능한 컨볼루션(separable convolutions)을 포함하는 NAS1과 일반 컨볼루션(regular convolutions)을 포함하는 NAS2 두 가지 연산자 탐색 공간에서 모두 96% 이상의 높은 정확도를 달성하며 방법론의 강건성을 보여주었습니다.

## 📎 Related Works

- **기존 KWS CNN 모델:**
  - **ResNet 변형:** Tang et al. [1]은 ResNet 변형을 사용하여 95.8% 정확도 달성. Choi et al. [4]는 TC-ResNet을 통해 96.6%로 개선.
  - **Sinc-convolution:** Mittermaier et al. [5]은 Sinc-convolution을 사용하여 모델 크기를 줄이면서 정확도 유지.
  - **CENet-GCN:** Chen et al. [9]은 그래프 컨볼루션 네트워크(GCN)를 삽입하여 96.8% 정확도 달성.
  - **MatchboxNet:** Majumdar et al. [10]은 1D time-channel separable convolutions를 사용하여 30-class 분류에서 97.48% 달성(다른 설정).
- **신경망 아키텍처 탐색(NAS):**
  - **강화 학습 기반 NAS:** Zoph et al. [11, 12]은 강화 학습을 사용하여 초기 NAS 방법을 제안했으나 계산 비용이 매우 높음.
  - **진화 알고리즘 기반 NAS:** AmoebaNet [13] 등.
  - **차등 가능한 아키텍처 탐색 (DARTS):** Liu et al. [15]이 제안한 방법으로, 탐색 공간을 연속적으로 만들어 그래디언트 기반 최적화를 통해 효율적인 아키텍처 탐색을 가능하게 함.
- **KWS를 위한 NAS 시도:** Veniat et al. [16] (86.5% 정확도), Anderson et al. [17] (95.11% 정확도) 등 일부 연구가 있었으나 최첨단 성능에는 미치지 못함.

## 🛠️ Methodology

1. **KWS 작업 설정:** 오디오 샘플에서 추출된 MFCC 특징 매트릭스를 기반으로 12개 클래스(10개 키워드, 알 수 없는 클래스, 침묵 클래스) 분류 문제를 해결합니다.
2. **네트워크 구조:** CNN은 초기 3x3 컨볼루션을 수행하는 헤드 레이어, `L`개의 셀(Cell) 스택, 그리고 분류를 수행하는 스템(stem)으로 구성됩니다.
3. **셀(Cell) 아키텍처 탐색:**
   - 전체 네트워크 대신 `정규 셀(normal cell)`과 `축소 셀(reduction cell)` 두 가지 유형의 셀 아키텍처를 탐색합니다.
   - `정규 셀`은 입력과 동일한 출력 크기를 유지하고, `축소 셀`은 채널 수를 두 배로 늘리고 높이와 너비를 절반으로 줄입니다.
   - 각 셀은 노드와 방향성 에지로 구성된 DAG(Directed Acyclic Graph)로 표현됩니다. 노드는 잠재 표현이고, 에지는 사전 정의된 연산자 집합 `O`의 혼합 연산을 나타냅니다.
   - 에지 $(i, j)$의 연산 $f_{i,j}(x_i)$는 다음과 같이 모든 가능한 연산자 $o(\cdot) \in O$의 가중치 합으로 정의됩니다:
     $$f_{i,j}(x_i) = \sum_{o \in O} \alpha_{(i,j),o} o(x_i)$$
     여기서 $\alpha_{(i,j),o}$는 학습 가능한 아키텍처 가중치입니다.
4. **DARTS 알고리즘 활용:**
   - DARTS [15]는 아키텍처 가중치 $\alpha$와 모델 가중치 $w$를 동시에 학습하는 이중 레벨 최적화 문제($\min_{\alpha} L_{val}(w^*(\alpha), \alpha)$ subject to $w^*(\alpha) = \arg \min_w L_{train}(w, \alpha)$)를 해결합니다.
   - 탐색 종료 시, 각 에지에서 가장 높은 $\alpha$ 가중치를 가진 연산자를 최종적으로 선택하여 셀 아키텍처를 구성합니다.
5. **후처리 및 평가:**
   - 탐색된 최적의 셀 아키텍처를 기반으로 네트워크의 깊이(셀 수)와 폭(초기 채널 수)을 조절하여 스케일업합니다 (예: 깊이 6 또는 12).
   - 스케일업된 네트워크는 처음부터 다시 훈련되어 평가됩니다.
6. **탐색 공간 (Candidate Operations):**
   - **NAS1:** separable convolution, dilated convolution, pooling (max/avg), skip connection, zero.
   - **NAS2:** regular convolution, dilated convolution, pooling (max/avg), skip connection, zero. (NAS2는 ResNet에서 사용되는 연산자와 유사한 공간을 탐색).
7. **데이터 전처리:** Honk [26]의 절차를 따르며, 배경 노이즈 추가, 시간 이동(time shift), 20Hz/4kHz 필터 적용 후, 40개의 MFCC(Mel-Frequency Cepstral Coefficients) 특징을 추출합니다.

## 📊 Results

- **최첨단 정확도 달성:** NAS1로 탐색된 12셀, 16 초기 채널 모델이 12-class 분류 설정에서 **97.06%**의 새로운 최첨단 정확도를 달성했습니다. 모델 크기는 281K 매개변수로 경쟁력 있었습니다.
- **베이스라인 모델 대비 우수성:**
  - Res15 [1] (95.8%, 239K 파라미터)
  - TC-ResNet14-1.5 [4] (96.6%, 305K 파라미터)
  - SincConv+DSConv [5] (96.6%, 122K 파라미터)
  - CENet-GCN-40 [9] (96.8%, 72.3K 파라미터)
  - NAS1의 6셀, 16 초기 채널 모델은 96.74% 정확도로 Res15, TC-ResNet, SincConv를 능가했습니다.
- **탐색 효율성:** NAS1은 0.58 GPU day, NAS2는 0.29 GPU day의 낮은 탐색 비용을 기록했습니다.
- **NAS2의 효과:** NAS2 모델들은 Res15와 유사한 연산자 공간을 사용했음에도 불구하고, Res15보다 0.94%p 높은 정확도(96.74%)를 달성하며 모델 크기는 24% 더 작았습니다. 이는 동일한 연산자 공간 내에서도 NAS를 통한 아키텍처 탐색의 이점을 보여줍니다.
- **깊이 및 폭의 영향:** 깊이(셀 수) 또는 폭(초기 채널 수)을 증가시키면 모델 성능이 향상되지만, 모델 크기도 증가하는 트레이드오프 관계를 보였습니다.
- **NAS1 vs. NAS2:** 분리 가능한 컨볼루션을 사용하는 NAS1 모델은 NAS2 모델보다 적은 수의 매개변수를 가지는 경향이 있었습니다.

## 🧠 Insights & Discussion

- **NAS의 잠재력:** 이 연구는 키워드 스포팅 작업에 신경망 아키텍처 탐색(NAS)을 적용하는 것의 엄청난 잠재력을 입증했습니다. 수동 설계에 의존하는 대신, NAS는 최적의 KWS 아키텍처를 자동으로 효율적으로 찾아낼 수 있습니다.
- **SOTA 달성:** 제안된 방법은 Google Speech Commands Dataset에서 최첨단 정확도를 달성하며, 경쟁력 있는 모델 크기를 유지했습니다. 이는 자원 제약이 있는 온디바이스 KWS 시스템에 특히 중요합니다.
- **방법론의 강건성:** NAS1(분리 가능한 컨볼루션)과 NAS2(일반 컨볼루션)라는 두 가지 다른 연산자 세트를 사용했음에도 불구하고 모두 높은 성능을 보여, 제안된 NAS 방법론이 다양한 연산자 공간에 걸쳐 강건함을 시사합니다.
- **아키텍처 스케일링:** 셀의 깊이와 초기 채널 수를 조절하여 성능과 모델 크기 간의 균형을 찾을 수 있음을 보여주었습니다. 이는 실제 배포 환경에서 요구사항에 맞춰 모델을 최적화할 수 있는 유연성을 제공합니다.
- **향후 연구 방향:** KWS 친화적인 다른 유형의 신경망(예: RNN)이나 특정 연산자를 탐색 공간에 포함시키는 것이 향후 연구의 유망한 방향이 될 수 있습니다.

## 📌 TL;DR

본 논문은 리소스 제약이 있는 온디바이스 키워드 스포팅(KWS) 시스템의 정확도와 메모리 효율성 문제를 해결하기 위해 **차등 가능한 아키텍처 탐색(DARTS) 기법**을 활용했습니다. 연구는 특정 컨볼루션 셀 아키텍처(정규/축소 셀)를 탐색한 후, 이를 스케일업하여 전체 네트워크를 구성합니다. Google Speech Commands Dataset에서 12-class 분류 설정 시 **97.06%의 최첨단 정확도**를 달성했으며, 이는 기존 수동 설계 모델들을 능가하는 결과입니다. 이 방법은 적은 탐색 비용으로 고성능 KWS 모델을 자동으로 생성할 수 있음을 보여주며, KWS 분야에서 NAS의 큰 잠재력을 입증했습니다.
