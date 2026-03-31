# G-CASCADE: Efficient Cascaded Graph Convolutional Decoding for 2D Medical Image Segmentation

Md Mostafijur Rahman, Radu Marculescu (2023)

## 🧩 Problem to Solve

자동 의료 영상 분할(automatic medical image segmentation)은 컴퓨터 지원 진단(computer-aided diagnosis) 분야에서 중요한 응용 분야이며, 다양한 질병의 진단, 치료 계획 수립, 치료 후 평가에 핵심적인 역할을 수행한다. 이 작업은 픽셀을 분류하여 병변, 종양 또는 장기를 식별하고 분할 맵을 생성하는 것을 목표로 한다.

기존 의료 영상 분할 방법론은 다음과 같은 한계를 가진다:

* **합성곱 신경망(CNN) 기반 U-shaped 네트워크 (예: UNet, UNet++, UNet 3+, DC-UNet)**: 이러한 네트워크는 합리적인 성능을 보이고 고해상도 분할 맵을 생성하지만, 주로 지역 정보(local information) 처리에 중점을 두어 **장거리 의존성(long-range dependencies)을 효과적으로 포착하는 데 어려움**을 겪는다. 어텐션 모듈(attention modules)을 통합한 방법들도 이 문제를 완전히 해결하지 못한다.
* **Vision Transformers (ViT) 및 계층적 Vision Transformers (예: Swin Transformer, PVT, MaxViT)**: 자기 어텐션(Self-Attention, SA) 메커니즘을 통해 픽셀 간의 장거리 의존성을 포착하는 데 뛰어난 능력을 보여 의료 영상 분할에서 유망한 성능을 달성한다. 그러나 트랜스포머의 자기 어텐션 모듈은 픽셀 간의 **지역 공간 관계(local spatial relationships)를 학습하는 데 제한적인 능력**을 가진다. 이를 보완하기 위해 일부 방법들은 디코더에 지역 합성곱 어텐션 모듈(local convolutional attention modules)을 통합하지만, 합성곱 연산의 본질적인 지역성(locality)으로 인해 여전히 **장거리 상관관계(long-range correlations)를 효과적으로 포착하는 데 어려움**이 존재한다.
* **기존 디코더의 비효율성 (예: CASCADE [28])**: 캐스케이드 디코더는 채널 어텐션과 공간 어텐션 모듈을 사용하여 특징을 정제하지만, 디코딩 과정에서 합성곱 연산만을 사용하므로 장거리 어텐션 부족을 초래할 수 있고, 각 단계에서 여러 개의 3x3 합성곱을 사용하여 높은 계산 비효율성을 가진다.

본 논문의 목표는 이러한 한계들을 극복하기 위해 그래프 합성곱(graph convolutions)을 활용하는 새로운 디코더, 즉 G-CASCADE를 제안하는 것이다. G-CASCADE는 그래프 합성곱 블록의 전역 수용장(global receptive field)을 통해 장거리 어텐션을 보존하고, 공간 어텐션 메커니즘을 통해 지역 어텐션을 통합하여 특징 맵을 효율적으로 강화하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 주요 기여 사항은 다음과 같다.

* **새로운 그래프 합성곱 디코더(G-CASCADE) 제안**: 2D 의료 영상 분할을 위한 새로운 그래프 기반 캐스케이드 합성곱 어텐션 디코더인 G-CASCADE를 소개한다. 이 디코더는 Vision Transformer의 다단계 특징을 입력받아 다중 스케일 및 다중 해상도 공간 표현을 학습한다. 이 연구는 시맨틱 분할을 위한 그래프 합성곱 네트워크(Graph Convolutional Network, GCN) 기반 디코더를 제안한 최초의 시도이다.
* **효율적인 그래프 합성곱 어텐션 블록(GCAM) 도입**: Vision Transformer의 장거리 어텐션을 보존하고 관련 없는 영역을 억제하여 특징을 강조하는 새로운 그래프 합성곱 어텐션 모듈을 도입한다. 이 모듈은 그래프 합성곱을 사용하여 디코더의 효율성을 높인다.
* **효율적인 업-합성곱 블록(UCB) 설계**: 성능 저하 없이 계산 효율을 높이는 효율적인 업-합성곱 블록을 설계한다. 이 UCB는 $3 \times 3$ 합성곱을 깊이별 합성곱(depth-wise convolution)으로 대체하여 경량화를 달성한다.
* **향상된 성능 및 계산 효율성 입증**: G-CASCADE가 어떠한 계층적 Vision Encoder(예: PVT, MERIT)와도 함께 사용될 때 2D 의료 영상 분할 성능을 크게 향상시킴을 경험적으로 증명한다. 여러 SOTA 방법론과 비교했을 때, G-CASCADE는 ACDC, Synapse Multi-organ, ISIC2018 피부 병변, Polyp, Retinal vessels 분할 벤치마크에서 현저히 낮은 계산 비용으로 더 나은 결과를 제공한다. 특히, SOTA CASCADE 디코더보다 80.8% 적은 파라미터와 82.3% 적은 FLOPs로 더 나은 DICE 점수를 달성한다.

## 📎 Related Works

본 논문은 관련 연구를 크게 세 가지 범주로 나누어 설명하고, 기존 방법론의 한계점과 G-CASCADE의 차별점을 제시한다.

### Vision Transformers

Dosovitskiy 등 [10]이 자기 어텐션(self-attention)을 통해 픽셀 간의 장거리 관계를 학습하는 Vision Transformer(ViT)를 개척한 이래로, 다양한 개선 연구가 진행되었다. 여기에는 합성곱 신경망(CNN)과의 통합 [40, 34], 새로운 자기 어텐션(SA) 블록 도입 [23, 34], 그리고 Swin Transformer [23], SegFormer [44], Pyramid Vision Transformer (PVT) [39], PVTv2 [40], MaxViT [34]와 같은 새로운 아키텍처 설계가 포함된다.

* **한계**: Vision Transformer는 인상적인 성능을 보이지만, 픽셀 간의 **지역(local) 공간 정보 처리 능력에 한계**를 가진다. G-CASCADE는 그래프 합성곱을 통해 장거리 어텐션을 보존하고, 공간 어텐션 메커니즘을 통해 지역 어텐션을 통합함으로써 이러한 한계를 극복하는 것을 목표로 한다.

### Vision Graph Convolutional Networks (GCNs)

그래프 합성곱 네트워크(GCNs)는 주로 컴퓨터 비전 분야에서 포인트 클라우드 분류 [20, 21], 장면 그래프 생성 [45], 액션 인식 [47] 등에 초점을 맞춰 개발되었다. Vision GNN (ViG) [13]는 이미지 데이터를 직접 처리하는 최초의 그래프 합성곱 백본 네트워크를 소개했다. ViG는 이미지를 패치로 나누고 K-최근접 이웃(KNN) 알고리즘을 사용하여 다양한 패치를 연결함으로써 Vision Transformer와 유사하게 장거리 의존성을 처리할 수 있다.

* **특징 및 차별점**: ViG에서 사용되는 그래프 합성곱 블록은 1x1 합성곱 연산을 전후로 사용하여 Vision Transformer나 3x3 합성곱 기반 CNN 블록보다 훨씬 빠르다. 본 논문에서는 이러한 그래프 합성곱 블록을 사용하여 특징 맵을 디코딩함으로써 계산 효율성을 높이고 장거리 정보를 보존하고자 한다.

### Medical Image Segmentation

의료 영상 분할은 의료 영상(예: 내시경, MRI, CT)에서 픽셀을 병변, 종양 또는 장기로 분류하는 작업이다 [4]. 이 작업에는 U-shaped 아키텍처 (UNet [30], UNet++ [49], UNet 3+ [15], DC-UNet [26])가 정교한 인코더-디코더 구조와 스킵 연결(skip connections) 덕분에 일반적으로 활용되어 왔다.
최근에는 TransUNet [4], Swin-Unet [2], MERIT [29] 등 트랜스포머가 의료 영상 분할 분야에서 인기를 얻었으며, CNN과 트랜스포머를 결합하여 지역 및 전역 픽셀 관계를 모두 포착하려는 하이브리드 아키텍처가 제안되기도 했다.
어텐션 메커니즘 또한 CNN [27, 11] 및 트랜스포머 기반 아키텍처 [9]와 결합되어 의료 영상 분할에 활용되었다 (예: PraNet [11], PolypPVT [9]).

* **CASCADE [28]의 한계와 차별점**: CASCADE [28]는 채널 어텐션 [14]과 공간 어텐션 [5] 모듈을 활용하는 캐스케이드 디코더를 제안하여 특징 정제를 수행한다. CASCADE는 트랜스포머 인코더의 네 단계에서 특징을 추출하고 캐스케이드 정제(cascaded refinement)를 사용하여 고해상도 분할 맵을 생성한다. 트랜스포머의 전역 정보와 지역 정보를 통합하여 뛰어난 성능을 보이지만, 두 가지 주요 한계가 있다: (i) 디코딩 중 합성곱 연산만 사용하여 **장거리 어텐션 부족**을 초래할 수 있고, (ii) 각 디코더 단계에서 세 개의 3x3 합성곱을 사용함으로써 **높은 계산 비효율성**을 가진다. G-CASCADE는 이러한 한계를 극복하기 위해 그래프 합성곱을 사용한다.

## 🛠️ Methodology

### 전체 파이프라인 또는 시스템 구조

G-CASCADE는 계층적 트랜스포머 인코더(hierarchical transformer encoders)가 생성하는 다단계 특징 맵(multi-stage feature maps)을 효율적인 그래프 합성곱 블록을 통해 점진적으로 정제하는 새로운 디코더이다. 인코더는 자기 어텐션(self-attention) 메커니즘을 사용하여 장거리 의존성(long-range dependencies)을 포착하고, 디코더는 그래프 합성곱 블록의 전역 수용장(global receptive fields) 덕분에 장거리 정보를 보존하면서 특징 맵을 정제한다.

전체 네트워크 아키텍처는 주로 (a) PVTv2-b2 인코더 백본과 (b) G-CASCADE 디코더로 구성된다(제공된 Figure 1 참조). 인코더의 네 단계(X1, X2, X3, X4)에서 출력된 특징들은 G-CASCADE 디코더로 전달된다. G-CASCADE 디코더는 효율적인 업-합성곱 블록(UCB), 그래프 합성곱 어텐션 모듈(GCAM), 그리고 분할 헤드(SegHead)로 구성되어 특징을 업샘플링, 강화, 그리고 최종 분할 출력으로 변환한다.

### 각 주요 구성 요소 및 역할

1. **G-CASCADE 디코더 (Cascaded Graph Convolutional Decoder)**:
    * 기존 트랜스포머 기반 모델이 갖는 지역 문맥 정보 처리 능력의 한계와, 이를 보완하기 위해 사용된 2D 합성곱 블록의 계산 비용 및 장거리 어텐션 부족 문제를 해결하기 위해 설계되었다.
    * `Figure 1(b)`에서 볼 수 있듯이, G-CASCADE는 특징을 업샘플링하는 효율적인 **업-합성곱 블록(UCB)**, 특징 맵을 강건하게 강화하는 **그래프 합성곱 어텐션 모듈(GCAM)**, 그리고 분할 출력을 생성하는 **분할 헤드(SegHead)**로 구성된다.
    * 인코더에서 나오는 네 단계의 피라미드 특징에 대해 네 개의 GCAM이 사용된다.
    * 다중 스케일 특징을 통합하기 위해, 이전 디코더 블록에서 업샘플링된 특징과 스킵 연결(skip connections)의 특징을 먼저 통합(예: 덧셈 또는 연결)한다. 통합된 특징은 GCAM에 의해 처리되어 시맨틱 정보가 강화된다.
    * 각 GCAM의 출력은 예측 헤드로 전달되고, 최종적으로 네 개의 다른 예측 맵을 통합하여 최종 분할 출력을 생성한다.

2. **그래프 합성곱 어텐션 모듈 (Graph Convolutional Attention Module, GCAM)**:
    * 특징 맵을 정제하는 핵심 모듈이다.
    * 장거리 어텐션을 보존하며 특징을 정제하는 **그래프 합성곱 블록(GCB)**과 지역 문맥 정보를 포착하는 **공간 어텐션(Spatial Attention, SPA)** 블록으로 구성된다.
    * 수식은 다음과 같다:
        $$ \text{GCAM}(x) = \text{SPA}(\text{GCB}(x)) $$
        여기서 $x$는 입력 텐서이다. 그래프 합성곱의 사용으로 기존 합성곱 기반 어텐션 모듈보다 훨씬 효율적이다.

3. **그래프 합성곱 블록 (Graph Convolution Block, GCB)**:
    * G-CASCADE의 캐스케이드 확장 경로를 통해 생성된 특징을 강화하는 데 사용된다.
    * Vision GNN [13]의 Grapher 설계를 따른다.
    * 하나의 그래프 합성곱 레이어 $\text{GConv}(\cdot)$와 두 개의 $1 \times 1$ 합성곱 레이어 $\text{C}(\cdot)$로 구성되며, 각 $\text{C}(\cdot)$ 뒤에는 배치 정규화(Batch Normalization, BN)와 ReLU 활성화(ReLU Activation, R)가 순차적으로 적용된다.
    * $\text{GCB}(\cdot)$는 다음 수식으로 표현된다:
        $$ \text{GCB}(x) = \text{R}(\text{BN}(\text{C}(\text{GConv}(\text{R}(\text{BN}(\text{C}(x))))))) $$
    * 여기서 $\text{GConv}(\cdot)$는 다음과 같이 정의된다:
        $$ \text{GConv}(x) = \text{GELU}(\text{BN}(\text{DynConv}(x))) $$
        * $\text{DynConv}(\cdot)$는 밀집 확장 K-최근접 이웃(dense dilated K-nearest neighbour, KNN) 그래프 내에서 수행되는 그래프 합성곱(예: Max-Relative, Edge, GraphSAGE, GIN)이다.
        * $\text{BN}(\cdot)$은 배치 정규화, $\text{GELU}(\cdot)$는 GELU 활성화 함수이다.
        * 실험에서는 $K=11$ 이웃을 가진 KNN과 Max-Relative (MR) 그래프 합성곱을 사용한다. 그래프 구성에 상대 위치 벡터를 사용하고, 그래프 합성곱 블록의 여러 단계에서 [1, 1, 4, 2]의 차원 감소율(reduction ratios)을 적용한다.

4. **공간 어텐션 (Spatial Attention, SPA)**:
    * 특징 맵에서 중요한 영역에 집중하고 해당 특징을 강화하는 역할을 한다.
    * $\text{SPA}(x)$는 다음 수식으로 표현된다:
        $$ \text{SPA}(x) = \text{Sigmoid}(\text{Conv}([\text{C}_{\text{max}}(x), \text{C}_{\text{avg}}(x)])) \circledast x $$
        여기서 $\text{Sigmoid}(\cdot)$는 시그모이드 활성화 함수, $\text{C}_{\text{max}}(\cdot)$와 $\text{C}_{\text{avg}}(\cdot)$는 채널 차원을 따라 얻은 최대값과 평균값이다. $\text{Conv}(\cdot)$는 지역 문맥 정보를 강화하기 위한 패딩 3을 가진 $7 \times 7$ 합성곱 레이어이다. $\circledast$는 아다마르 곱(Hadamard product)이다.

5. **업-합성곱 블록 (Up-convolution block, UCB)**:
    * 현재 레이어의 특징을 다음 스킵 연결의 차원에 맞게 점진적으로 업샘플링한다.
    * 각 UCB 레이어는 스케일 인자 2의 업샘플링 $\text{Up}(\cdot)$, 입력 채널 수와 동일한 그룹을 가진 $3 \times 3$ 깊이별 합성곱(Depth-Wise Convolution, DWC), 배치 정규화 $\text{BN}(\cdot)$, ReLU 활성화 $\text{ReLU}(\cdot)$, 그리고 $1 \times 1$ 합성곱 $\text{Conv}(\cdot)$으로 구성된다.
    * $\text{UCB}(\cdot)$는 다음 수식으로 표현된다:
        $$ \text{UCB}(x) = \text{Conv}(\text{ReLU}(\text{BN}(\text{DWC}(\text{Up}(x))))) $$
    * 업샘플링 후 $3 \times 3$ 합성곱을 깊이별 합성곱으로 대체하여 블록의 경량화(light-weight)를 달성한다.

6. **분할 헤드 (Segmentation Head, SegHead)**:
    * 디코더의 네 단계에서 정제된 특징 맵을 입력으로 받아 네 개의 출력 분할 맵을 예측한다.
    * 각 SegHead 레이어는 $1 \times 1$ 합성곱 $\text{Conv}_{1 \times 1}(\cdot)$으로 구성된다. 이 합성곱은 단계 $i$의 특징 맵 채널 수 $N_i$를 가진 특징 맵을 입력으로 받아 다중 클래스 분할의 경우 대상 클래스 수와 동일한 채널, 이진 분할의 경우 1채널의 출력을 생성한다.
    * $\text{SegHead}(\cdot)$는 다음 수식으로 표현된다:
        $$ \text{SegHead}(x) = \text{Conv}_{1 \times 1}(x) $$

### 훈련 목표, 손실 함수, 추론 절차 또는 알고리즘 흐름

* **전체 아키텍처 통합**: 제안하는 G-CASCADE 디코더는 효과적인 일반화 및 다중 스케일 특징 처리 능력을 보장하기 위해 PVTv2 [40] 및 MERIT [29]와 같은 두 가지 다른 계층적 백본 인코더 네트워크와 통합된다.
  * **PVT-GCASCADE**: PVTv2-b2 인코더의 네 레이어에서 특징(X1, X2, X3, X4)을 추출하여 G-CASCADE 디코더에 입력한다. 디코더는 이 특징들을 처리하여 인코더 네트워크의 네 단계에 해당하는 네 개의 예측 맵을 생성한다.
  * **MERIT-GCASCADE**: MERIT 네트워크의 아키텍처 설계를 채택하되, MERIT의 디코더를 제안된 디코더로 대체한다. 첫 번째 인코더의 네 단계에서 계층적 특징 맵을 추출하여 해당 디코더에 전달한다. 최종 디코더 단계의 피드백을 입력 이미지에 통합하고, 다른 윈도우 크기를 가진 두 번째 인코더에 전달한다. 두 번째 인코더의 네 단계에서 특징 맵을 추출하여 두 번째 디코더에 전달하고, MERIT [29]와 같이 캐스케이드 스킵 연결을 보낸다. 두 번째 디코더의 네 단계에서 네 개의 출력 분할 맵을 얻고, 최종적으로 두 디코더의 네 단계별 분할 맵을 통합하여 네 개의 출력 분할 맵을 생성한다.

* **다단계 출력 및 손실 통합**:
  * G-CASCADE 디코더의 네 예측 헤드에서 네 개의 출력 분할 맵 $p_1, p_2, p_3, p_4$를 얻는다.
  * **출력 분할 맵 통합**: 최종 분할 출력은 가산 통합(additive aggregation)을 사용하여 계산된다.
        $$ \text{segoutput} = \alpha p_1 + \beta p_2 + \gamma p_3 + \zeta p_4 $$
        여기서 $\alpha, \beta, \gamma, \zeta$는 각 예측 헤드의 가중치이며, 모든 실험에서 1.0으로 설정된다. 이진 분할의 경우 시그모이드(Sigmoid) 활성화를, 다중 클래스 분할의 경우 소프트맥스(Softmax) 활성화를 적용하여 최종 예측 출력을 얻는다.
  * **손실 통합**: MERIT [29]를 따라 조합적 손실 통합 전략인 MUTATION을 모든 실험에서 사용한다. 이는 $n$개의 헤드에서 합성된 $2^n-1$개의 조합 예측에 대한 손실을 각각 계산한 다음, 이들을 합산하여 훈련 중에 최적화한다.

* **훈련 절차**:
  * Pytorch 1.11.0으로 구현하며, NVIDIA RTX A6000 GPU (48GB)에서 실험을 수행한다.
  * PVTv2-b2 및 Small Cascaded MERIT을 대표 네트워크로 사용하고, 두 백본 네트워크 모두 ImageNet으로 사전 훈련된 가중치를 사용한다.
  * AdamW 옵티마이저 [24]를 사용하며, 학습률과 가중치 감쇠는 모두 0.0001로 설정한다.
  * **GCB**: KNN의 경우 $K=11$ 이웃, Max-Relative (MR) 그래프 합성곱을 사용한다. MR 그래프 합성곱 후 배치 정규화를 적용한다. ViG [13]에 따라 그래프 구성에 상대 위치 벡터를 사용하고, 그래프 합성곱 블록의 다른 단계에서 차원 감소율은 [1, 1, 4, 2]로 설정한다.
  * **데이터셋별 훈련 설정**:
    * **Synapse Multi-organ**: 배치 크기 6, 최대 300 에포크. 입력 해상도는 PVT-GCASCADE의 경우 $224 \times 224$, MERIT-GCASCADE의 경우 $(256 \times 256, 224 \times 224)$. 무작위 회전 및 뒤집기(flipping)로 데이터 증강. 결합된 가중치 교차 엔트로피(0.3) 및 DICE(0.7) 손실 함수를 사용한다.
    * **ACDC**: 배치 크기 12, 최대 150 에포크. 입력 해상도는 PVT-GCASCADE의 경우 $224 \times 224$, MERIT-GCASCADE의 경우 $(256 \times 256, 224 \times 224)$. 무작위 뒤집기 및 회전으로 데이터 증강. 결합된 가중치 교차 엔트로피(0.3) 및 DICE(0.7) 손실 함수를 사용한다.
    * **ISIC2018**: 이미지를 $384 \times 384$ 해상도로 리사이즈. 200 에포크, 배치 크기 4, 기울기 클립(gradient clip) 0.5. 결합된 가중치 BCE(Binary Cross Entropy) 및 가중치 IoU 손실 함수를 최적화한다.
    * **Polyp**: 이미지를 $352 \times 352$ 해상도로 리사이즈. CASCADE [28]와 동일하게 다중 스케일 $\{0.75, 1.0, 1.25\}$ 훈련 전략과 기울기 클립 0.5를 사용한다. 배치 크기 4, 최대 200 에포크. 결합된 가중치 BCE 및 가중치 IoU 손실 함수를 최적화한다.
    * **Retinal vessels**: 훈련 세트 확장을 위해 수평 뒤집기, 수직 뒤집기, 수평-수직 뒤집기, 무작위 회전, 무작위 색상, 무작위 가우시안 블러를 적용한다. DRIVE 데이터셋은 PVT에 $768 \times 768$, MERIT에 $(768 \times 768, 672 \times 672)$ 해상도. CHASEDB1은 PVT에 $960 \times 960$, MERIT에 $(768 \times 768, 672 \times 672)$ 해상도. 추론 시 출력 분할 맵은 원본 해상도로 리사이즈한다. 훈련 중 0.5 확률로 무작위 뒤집기 및 회전 증강. AdamW 옵티마이저, 학습률 및 가중치 감쇠 1e-4. 결합된 가중치 BCE 및 가중치 mIoU 손실 함수. MUTATION으로 다단계 손실 통합. DRIVE에 배치 크기 4, CHASEDB1에 배치 크기 2로 200 에포크 훈련.

## 📊 Results

### 데이터셋, 작업, 기준선, 지표

* **데이터셋**:
  * **Synapse Multi-organ**: 30개의 복부 CT 스캔 (총 3779개 축 단면), 8개 복부 장기 (대동맥, 담낭(GB), 좌신장(KL), 우신장(KR), 간, 췌장(PC), 비장(SP), 위(SM)) 분할.
  * **ACDC**: 100개의 심장 MRI 스캔, 3개 심장 장기 (우심실(RV), 심근(Myo), 좌심실(LV)) 분할.
  * **ISIC2018**: 2596개의 피부 병변 이미지 분할.
  * **Polyp datasets**: Kvasir, CVC-ClinicDB (훈련), EndoScene, ColonDB (테스트) 폴립 분할.
  * **Retinal vessels segmentation datasets**: DRIVE, CHASEDB1 망막 혈관 분할.
* **작업**: 2D 의료 영상 시맨틱 분할.
* **기준선 (Baselines)**: UNet [30], AttnUNet [27], R50+UNet [4], R50+AttnUNet [4], SSFormerPVT [38], PolypPVT [9], TransUNet [4], SwinUNet [2], MT-UNet [37], MISSFormer [16], PVT-CASCADE [28], TransCASCADE [28], Cascaded MERIT [29] 등 다양한 CNN 및 트랜스포머 기반 SOTA 방법들.
* **지표 (Metrics)**:
  * **DICE** (Dice Similarity Coefficient, ↑): $ \text{DSC}(Y, \hat{Y}) = \frac{2 \times |Y \cap \hat{Y}|}{|Y| + |\hat{Y}|} \times 100 $
  * **mIoU** (Mean Intersection over Union, ↑): $ \text{IoU}(Y, \hat{Y}) = \frac{|Y \cap \hat{Y}|}{|Y \cup \hat{Y}|} \times 100 $
  * **95% Hausdorff Distance (HD95)** (↓): $ D_H(Y, \hat{Y}) = \max\{\max_{y \in Y} \min_{\hat{y} \in \hat{Y}} d(y,\hat{y}), \max_{\hat{y} \in \hat{Y}} \min_{y \in Y} d(y,\hat{y})\} $
  * **Accuracy (Acc)**, **Sensitivity (Sen)**, **Specificity (Sp)** (↑)
  * $Y$는 실제(ground truth) 분할 맵, $\hat{Y}$는 예측 분할 맵이다.
  * 모든 G-CASCADE 결과는 5회 실행의 평균으로 보고된다.

### 주요 정량적 또는 정성적 결과

#### 1. Synapse Multi-organ 데이터셋 (Table 1)

* **MERIT-GCASCADE**는 평균 DICE 점수 84.54%로, 모든 SOTA CNN 및 트랜스포머 기반 2D 의료 영상 분할 방법론을 크게 능가한다.
* **PVT-GCASCADE**는 PVT-CASCADE보다 2.22%, **MERIT-GCASCADE**는 Cascaded MERIT보다 0.22% DICE 점수가 향상되었으며, 계산 비용은 훨씬 낮다.
* HD95 거리 측면에서도 PVT-GCASCADE는 4.4, MERIT-GCASCADE는 3.89만큼 기준선 모델 대비 개선을 보였다. MERIT-GCASCADE는 가장 낮은 HD95 (10.38)를 기록하여, SOTA인 Cascaded MERIT (14.27)보다 3.89 낮다. 이는 G-CASCADE 디코더가 장기 경계를 더 정확하게 식별하는 능력이 있음을 나타낸다.
* 개별 장기 분할 DICE 점수에서 MERIT-GCASCADE는 8개 장기 중 5개에서 SOTA 방법론을 능가한다. 이는 그래프 합성곱과 트랜스포머 인코더의 결합이 성능 향상에 기여했음을 시사한다.

#### 2. ACDC 데이터셋 (Table 2)

* **MERIT-GCASCADE**는 가장 높은 평균 DICE 점수 92.23%를 달성하여 Cascaded MERIT 대비 약 0.38% 향상되었다. 이는 디코더의 계산 비용이 현저히 낮음에도 불구하고 달성된 결과이다.
* **PVT-GCASCADE** 또한 91.95%의 DICE 점수를 기록하여 다른 모든 방법론보다 우수한 성능을 보인다.
* 두 제안 모델 모두 세 가지 심장 장기 분할(RV, Myo, LV) 모두에서 더 나은 DICE 점수를 달성했다.

#### 3. ISIC2018 데이터셋 (Table 7)

* CT 및 MRI 이미지와는 다른 피부 병변 데이터셋인 ISIC2018에서도 **PVT-GCASCADE**는 가장 높은 평균 DICE (91.51%) 및 mIoU (86.53%) 점수를 달성했다.
* PVT-GCASCADE는 PVT-CASCADE 대비 DICE에서 0.4%, mIoU에서 0.6% 향상된 성능을 보였다.

#### 4. Polyp 데이터셋 (Table 8)

* G-CASCADE는 CVC-ClinicDB, Kvasir, ColonDB, EndoScene 등 모든 폴립 분할 데이터셋에서 DICE 및 mIoU 점수 모두에서 다른 모든 방법론을 크게 능가한다.
* 특히, 미지의 데이터셋인 ColonDB에서 CNN 기반 최고 모델인 UACANet 대비 DICE 점수 9.8%라는 큰 폭의 향상을 보여준다. 이는 G-CASCADE 디코더가 트랜스포머 백본 네트워크와 그래프 기반 합성곱 어텐션을 활용하여 트랜스포머, GCN, CNN, 지역 어텐션의 장점을 모두 상속받아 미지의 데이터셋에 대해 높은 일반화 가능성(generalizability)을 갖는다는 것을 입증한다.

#### 5. Retinal vessels segmentation 데이터셋 (Table 9, 10)

* **DRIVE 데이터셋 (Table 9)**: MERIT-GCASCADE는 97.07% Acc, 82.81% Sen, 98.44% Sp, 82.90% DICE, 70.81% IoU를 달성하며 SOTA 접근 방식과 경쟁력 있는 성능을 보여준다. PVT-GCASCADE는 PVT-CASCADE 대비 DICE에서 0.37%, MERIT-GCASCADE는 MERIT-CASCADE 대비 DICE에서 0.69% 향상되었다.
* **CHASEDB1 데이터셋 (Table 10)**: MERIT-GCASCADE는 97.76% Acc, 84.93% Sen, 98.62% Sp, 82.67% DICE, 70.50% IoU를 기록한다. PVT-GCASCADE는 PVT-CASCADE 대비 DICE에서 1.01%, MERIT-GCASCADE는 MERIT-CASCADE 대비 DICE에서 0.99% 향상되었다.
* FR-UNet [22]이 DRIVE 데이터셋에서 약간 더 나은 DICE 점수를 보였지만, CHASEDB1에서는 MERIT-GCASCADE보다 1.16% 낮은 DICE 점수를 기록했다. FR-UNet은 훈련 시 이미지를 48x48 픽셀 패치로 분할하는 반면, G-CASCADE는 전체 이미지를 사용하므로 훈련 샘플 수가 훨씬 적음에도 불구하고 경쟁력 있는 성능을 보여준다.

#### 정성적 결과 (Figure 2)

* Synapse Multi-organ 데이터셋에 대한 정성적 결과는 MERIT-GCASCADE가 최소한의 거짓 음성(false negative) 및 거짓 양성(false positive) 결과를 통해 장기를 일관되게 분할함을 보여준다. PVT-GCASCADE와 Cascaded MERIT은 비교 가능한 결과를 보이지만, PVT-GCASCADE가 첫 번째 샘플에서 거짓 양성을, Cascaded MERIT이 두 번째 샘플에서 더 큰 거짓 양성을 보이는 등 일부 불일치가 관찰된다. TransCASCADE와 PVT-CASCADE는 두 샘플 모두에서 더 큰 부정확한 분할 출력을 제공한다.

### 실험의 실제 결과 (Ablation Study)

#### 1. G-CASCADE의 다른 구성 요소의 효과 (Table 3)

* Synapse Multi-organ 데이터셋에서 G-CASCADE 디코더의 캐스케이드 구조, GCB, SPA 모듈 각각이 성능 향상에 기여함을 보여준다.
* 특히, SPA와 GCB 모듈을 함께 사용할 때 83.3%의 가장 좋은 DICE 점수를 얻는다.
* 이러한 구성 요소들을 모두 사용함으로써 DICE 점수는 0.342G FLOPs와 1.78M 추가 파라미터로 약 3.2% 향상된다.

#### 2. GCAM 내 GCB와 SPA의 배열 순서 효과 (Table 4)

* GCAM 내에서 그래프 합성곱 블록(GCB) 다음에 공간 어텐션(SPA) 블록이 오는 순서($\text{GCB} \to \text{SPA}$)가 SPA 다음에 GCB 블록이 오는 순서($\text{SPA} \to \text{GCB}$)보다 더 나은 성능(DICE 83.28% vs 82.93%)을 보인다. 따라서 G-CASCADE 디코더의 각 GCAM에서는 GCB 다음에 SPA 블록을 사용한다.

#### 3. 기준선 디코더(CASCADE)와의 비교 (Table 5)

* 수정된 UCB(Modified UCB)는 기존 CASCADE 디코더에 사용된 UCB(Original)보다 FLOPs 및 파라미터가 현저히 적으면서도 동일하거나 더 나은 성능을 제공한다.
* 제안된 G-CASCADE 디코더는 CASCADE 디코더보다 DICE 점수가 0.5% 더 높으면서도 파라미터는 80.8% 적고 FLOPs는 82.3% 적은 것으로 나타났다 (G-CASCADE: 0.342G FLOPs, 1.78M 파라미터 vs CASCADE: 1.93G FLOPs, 9.27M 파라미터).

#### 4. G-CASCADE 디코더의 다른 스킵-통합(skip-aggregations)의 효과 (Table 6)

* Upsampled 특징과 스킵 연결을 통합하는 방식에서 Concatenation 기반 통합은 Additive 통합보다 미미하게 더 나은 DICE 점수를 달성한다.
* 그러나 Concatenation 방식은 FLOPs와 파라미터가 현저히 높다. 이는 연결된 채널(원본 채널의 2배)에 GCAM을 사용함으로써 계산 복잡도가 증가하기 때문이다.
* 낮은 계산 복잡도를 고려하여 모든 실험에서 Additive 통합을 사용했다.

#### 5. GCAM의 다른 그래프 합성곱 비교 (Table 11)

* GCAM 블록 내에서 다양한 그래프 합성곱(GIN [46], EdgeConv [41], GraphSAGE [12], Max-Relative [21])을 비교한 결과, Max-Relative (MR) 그래프 합성곱이 0.342G FLOPs와 1.78M 파라미터로 가장 좋은 DICE 점수(83.28%)를 제공한다.
* GIN은 약간 더 낮은 FLOPs와 파라미터를 가지지만 가장 낮은 DICE 점수를 보였고, EdgeConv와 GraphSAGE는 MR 그래프 합성곱보다 DICE 점수가 낮고 계산 비용은 높다.

#### 6. 전체 계산 복잡도 비교 (Table 12)

* 전체 계산 복잡도는 인코더 백본의 파라미터 수와 FLOPs에 크게 의존한다.
* PVT-GCASCADE는 PVT-CASCADE보다 FLOPs (4.252G vs 5.843G)와 파라미터 (26.64M vs 34.13M)가 각각 1.588G와 7.49M 적다.
* MERIT-GCASCADE도 MERIT-CASCADE보다 FLOPs (26.143G vs 33.31G)와 파라미터 (132.93M vs 147.86M)가 적다.
* 이러한 FLOPs 및 파라미터 절감은 오직 G-CASCADE 디코더에서 비롯된다. 이는 제안된 디코더가 다른 계층적 인코더에 쉽게 연결될 수 있으며, 경량 인코더와 함께 사용될 경우 총 계산 비용을 더욱 줄일 수 있음을 시사한다.

#### 7. 입력 해상도의 영향 (Table 13)

* PVT-GCASCADE 네트워크의 입력 해상도(224x224, 256x256, 384x384)에 따른 분할 성능을 평가한 결과, 입력 해상도가 높아질수록 모든 세 가지 평가 지표(DICE, mIoU, HD95)에서 성능이 향상된다.
* $384 \times 384$ 해상도에서 가장 좋은 DICE (86.01%) 및 mIoU (78.10%)를 얻었다.

## 🧠 Insights & Discussion

### 논문에서 뒷받침되는 강점

* **효율적인 장거리 및 지역 특징 학습의 시너지**: G-CASCADE는 그래프 합성곱의 전역 수용장(global receptive field)을 통해 트랜스포머 인코더가 포착한 장거리 의존성 정보를 효과적으로 보존하고, 동시에 공간 어텐션 메커니즘을 통해 지역 문맥 정보를 강화한다. 이러한 하이브리드 접근 방식은 기존 트랜스포머 기반 모델의 지역 정보 처리 한계와 CNN 기반 모델의 장거리 상관관계 포착 어려움을 동시에 해결한다.
* **뛰어난 계산 효율성**: CASCADE 디코더와 비교하여, G-CASCADE는 80.8% 적은 파라미터와 82.3% 적은 FLOPs로 더 높은 분할 성능(0.5% DICE 향상)을 달성한다. 이는 고성능을 유지하면서도 계산 자원 요구량을 크게 줄여, 실제 시스템(예: 임베디드 장치)에 배포될 가능성을 높이는 중요한 강점이다.
* **높은 성능과 일반화 능력**: Synapse Multi-organ, ACDC, ISIC2018, Polyp, Retinal vessels 등 5가지 다양한 의료 영상 분할 벤치마크에서 SOTA 방법론들을 일관되게 능가하는 성능을 입증한다. 특히, Polyp 데이터셋의 미지의 이미지에 대한 뛰어난 일반화 능력은 다양한 임상 시나리오에서의 적용 가능성을 높인다.
* **모듈식 및 유연한 설계**: G-CASCADE 디코더는 PVTv2, MERIT과 같은 다양한 계층적 인코더 백본에 쉽게 통합될 수 있도록 설계되었다. 이러한 모듈성은 연구자들이 자신의 특정 작업에 최적화된 인코더와 G-CASCADE를 결합할 수 있는 유연성을 제공하며, 의료 영상 외 일반적인 시맨틱 분할 작업으로의 확장 가능성을 제시한다.
* **정교한 구성 요소 최적화**: 효율적인 업-합성곱 블록(UCB)의 설계(깊이별 합성곱 사용)와 그래프 합성곱 어텐션 모듈(GCAM) 내 GCB와 SPA의 최적 순서 탐색 등 디코더의 세부적인 구성 요소에 대한 면밀한 분석과 최적화 노력이 전반적인 성능 향상에 기여한다.

### 한계, 가정 또는 미해결 질문

* **스킵 연결 통합 방식의 트레이드오프**: 논문은 Additive 통합과 Concatenation 통합을 비교하며, Concatenation이 약간 더 나은 성능을 보이지만 훨씬 높은 계산 비용을 유발하기 때문에 Additive 통합을 선택했다고 명시한다. 이는 효율성을 위한 실용적인 결정이지만, 최고 성능을 추구하는 특정 시나리오에서는 Concatenation 방식의 잠재력을 완전히 탐색하지 않았다는 한계가 될 수 있다.
* **그래프 구성의 하이퍼파라미터 민감성**: GCB 내에서 그래프를 구성하기 위해 K-Nearest Neighbors(KNN) 알고리즘을 사용하며, $K=11$과 같은 하이퍼파라미터를 설정한다. 이 $K$ 값의 최적성이나 다른 데이터셋 또는 다양한 이미지 특성에서의 일반화 능력에 대한 추가적인 분석은 명확하게 제시되지 않았다. 그래프 구조 자체가 G-CASCADE의 성능에 큰 영향을 미칠 수 있으므로, 이에 대한 심층적인 연구가 필요할 수 있다.
* **백본 인코더 의존성**: G-CASCADE 디코더 자체는 효율적이지만, 전체 시스템의 계산 복잡도는 여전히 백본 인코더의 크기에 크게 의존한다. 예를 들어, MERIT 인코더를 사용할 경우 전체 FLOPs와 파라미터는 PVT 인코더를 사용하는 경우보다 훨씬 높다. 디코더의 효율성에도 불구하고, 전체 시스템의 경량화를 위해서는 여전히 인코더의 선택이 중요한 요소로 남는다.
* **의료 영상 외 시맨틱 분할에 대한 명시적 증명 부족**: G-CASCADE가 일반적인 시맨틱 분할 작업에도 쉽게 사용될 수 있다고 언급하지만, 이에 대한 구체적인 실험적 증거는 의료 영상 도메인 외에는 제시되지 않았다. 다른 일반적인 컴퓨터 비전 데이터셋(예: Cityscapes, ADE20K)에 대한 성능 평가가 이루어진다면 디코더의 범용성을 더욱 강력하게 입증할 수 있을 것이다.

### 논문에 근거한 간략한 비판적 해석 및 논의사항

G-CASCADE는 의료 영상 분할 분야에서 트랜스포머의 전역적 특징 학습 능력과 그래프 합성곱의 효율적인 구조적 특징 포착 능력을 성공적으로 융합한 의미 있는 연구이다. 기존 트랜스포머 및 CNN 기반 방법론이 가진 장거리/지역 특징 처리의 한계를 동시에 극복하고, 특히 CASCADE와 같은 선행 연구의 계산 비효율성을 효과적으로 해결하면서도 성능을 향상시켰다는 점에서 높은 평가를 받을 만하다. 이는 실제 임상 환경이나 컴퓨팅 자원이 제한된 환경에서 고성능 의료 영상 분할 모델을 개발하는 데 중요한 기여를 할 것으로 예상된다.

논문은 Max-Relative 그래프 합성곱의 효율성 입증, GCB와 SPA 블록의 최적 배치 순서 탐색, 그리고 깊이별 합성곱을 활용한 경량 UCB 설계 등 디코더의 세부적인 구성 요소에 대한 면밀한 분석과 최적화 노력을 통해 연구의 신뢰성을 높였다. 다만, 스킵 연결 통합 방식에서 효율성을 위해 Additive 방식을 선택한 것은 실용적이지만, Concatenation 방식의 미미한 성능 우위를 감안할 때, 향후 연구에서는 계산 효율성과의 균형점을 더욱 정교하게 탐색할 여지가 남아있다고 본다.

결론적으로, G-CASCADE는 의료 영상 분할의 정확도와 효율성을 동시에 개선하는 혁신적인 디코더 아키텍처를 제시하며, 트랜스포머와 그래프 신경망의 융합을 통한 새로운 연구 방향을 제시했다는 점에서 중요한 연구이다. 이 연구는 향후 다양한 시맨틱 분할 문제 해결에도 영감을 줄 것으로 기대된다.

## 📌 TL;DR

G-CASCADE는 2D 의료 영상 분할을 위해 제안된 효율적인 그래프 기반 캐스케이드 합성곱 어텐션 디코더이다. 이 연구는 계층적 트랜스포머 인코더에서 생성된 다단계 특징 맵을 효율적인 그래프 합성곱 블록을 통해 점진적으로 정제하는 것을 목표로 한다. 핵심 아이디어는 그래프 합성곱의 전역 수용장(global receptive field)을 활용하여 트랜스포머의 장거리 의존성 정보를 보존하는 동시에, 공간 어텐션 메커니즘을 통해 지역 문맥 정보를 효과적으로 포착함으로써 기존 방법론의 한계를 극복하는 것이다. G-CASCADE는 또한 $3 \times 3$ 합성곱을 깊이별 합성곱으로 대체한 경량 업-합성곱 블록(UCB)과 그래프 합성곱 블록(GCB) 및 공간 어텐션(SPA)으로 구성된 그래프 합성곱 어텐션 모듈(GCAM)을 도입하여 계산 효율성을 극대화한다. 실험 결과, G-CASCADE는 Synapse Multi-organ, ACDC, ISIC2018, Polyp, Retinal vessels 등 5가지 주요 의료 영상 분할 벤치마크에서 SOTA 방법론들을 능가하는 성능을 보였다. 특히, 기존 CASCADE 디코더보다 80.8% 적은 파라미터와 82.3% 적은 FLOPs로 더 높은 DICE 점수를 달성하며 뛰어난 효율성을 입증했다. 이 연구는 효율성과 고성능을 동시에 달성하는 새로운 디코더 아키텍처를 제시함으로써 의료 영상 분할 및 일반 시맨틱 분할 분야의 실제 적용 및 향후 연구에 중요한 기여를 할 것으로 기대된다.
