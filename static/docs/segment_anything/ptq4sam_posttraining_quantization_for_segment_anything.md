# PTQ4SAM: Segment Anything을 위한 Post-Training Quantization

Chengtao Lv, Hong Chen, Jinyang Guo, Yifu Ding, Xianglong Liu

## 🧩 Problem to Solve

Segment Anything Model (SAM)은 컴퓨터 비전 분야에서 인상적인 성능을 보여주지만, 대규모 모델인 탓에 막대한 메모리 및 컴퓨팅 비용이 발생하여 실제 배포에 어려움이 있습니다. 기존의 Post-Training Quantization (PTQ) 방법들을 SAM에 직접 적용하기 어려운 두 가지 주요 과제가 있습니다:

1. **이중 모드(Bimodal) 분포:** `post-Key-Linear` 활성화에서 이중 모드 분포가 관찰됩니다. 이는 두 개의 피크와 그 사이의 공백 구간으로 인해 전체 분포 범위가 크게 확장되어 양자화 성능에 부정적인 영향을 미칩니다.
2. **다양한 `post-Softmax` 분포:** SAM은 다양한 어텐션 메커니즘(self-attention, token-to-image, image-to-token cross-attention)을 포함하고 있으며, 이로 인해 `post-Softmax` 분포에서 상당한 변동성이 나타납니다. 기존 연구들은 이러한 차이를 충분히 다루지 않고 동일하게 처리하여 고유한 정보 손실을 초래합니다.

## ✨ Key Contributions

* Segment Anything Model을 위해 특별히 고안된 최초의 후방-훈련 양자화(Post-Training Quantization, PTQ) 솔루션인 **PTQ4SAM**을 제안합니다.
* 양자화에 어려운 이중 모드 분포를 관찰하고 그 특성을 분석했습니다. 이를 극복하기 위해 자동으로 이중 모드 분포를 감지하고 동등하게 정규 분포로 변환하는 **Bimodal Integration (BIG)** 전략을 제안했습니다.
* 다양한 `post-Softmax` 분포를 적절한 세분성(granularity)으로 정확하게 표현하는 **Adaptive Granularity Quantization (AGQ)**을 제안했습니다.
* 다양한 작업, 모델 변형 및 비트 폭에 대한 광범위한 실험을 통해 PTQ4SAM이 플러그 앤 플레이 방식이며 이전 최첨단 PTQ 방식을 크게 능가함을 입증했습니다.

## 📎 Related Works

* **Segment Anything (SAM):** Meta AI Research에서 제안한 일반적이고 프롬프트 가능한 모델로, 웹 규모 데이터셋(SA-1B)으로 사전 학습되어 이미지 분할, 객체 탐지, 추적 등 다양한 다운스트림 작업에서 뛰어난 제로샷 능력을 보입니다. HQ-SAM, SEEM, MedSAM 등 다양한 변형 모델이 연구되었으며, MobileSAM, FastSAM, TinySAM과 같은 효율적인 SAM 모델도 제안되었으나 여전히 상당한 자원 소모 문제를 안고 있습니다.
* **Post-Training Quantization (PTQ):** 널리 사용되는 모델 압축 기법으로, 크게 통계 기반 PTQ와 학습 기반 PTQ로 나뉩니다.
  * **통계 기반 PTQ:** 최적의 양자화 파라미터를 찾아 양자화 오류를 최소화합니다. CNNs (MinMax, OMSE)와 Vision Transformers (ViTs) (Log-Int-Softmax, scale reparameterization)를 위한 방법들이 개발되었으며, LLMs를 위한 가중치 및 활성화 양자화(GPTQ, SmoothQuant)도 연구되었습니다.
  * **학습 기반 PTQ:** 가중치와 양자화 파라미터 모두를 미세 조정합니다 (AdaRound, BRECQ, QDrop).
* 하지만 기존의 PTQ 방법들은 SAM의 고유한 아키텍처적 특성, 특히 `post-Key-Linear`의 이중 모드 분포와 다양한 `post-Softmax` 분포를 직접적으로 해결하지 못했습니다.

## 🛠️ Methodology

본 논문은 SAM 양자화를 위한 PTQ4SAM 프레임워크를 제안하며, 다음 두 가지 핵심 전략으로 구성됩니다.

1. **Bimodal Integration (BIG) 전략:**
    * **문제 식별 및 분석:** `post-Key-Linear` 활성화에서 이중 모드 분포를 발견했습니다. 이 분포는 per-tensor 관점에서 대칭적인 두 개의 피크(-8, 8)를 가지지만, per-channel 관점에서는 각 채널이 특정 피크(음수 또는 양수)에 집중되는 강한 비대칭성을 보였습니다. 이로 인해 양자화 오류가 5배 이상 증가했습니다.
    * **부호 인자($\gamma$) 기반 변환:** 이러한 이중 모드 분포를 해결하기 위해 채널별 부호 인자 $\gamma \in \mathbb{R}^{n}$를 도입합니다. 각 채널 $j$에 대해 $\gamma_j$는 다음 식을 통해 계산됩니다:
        $$
        \gamma_{j} = \begin{cases} +1, & \text{if mean}(K_{:,j}) \ge 0 \\ -1, & \text{otherwise} \end{cases}
        $$
        여기서 $K_{:,j}$는 $j$-th 채널의 `post-Key-Linear` 활성화입니다.
    * **오프라인 등가 변환:** 계산된 $\gamma$를 `query` 선형 계층과 `key` 선형 계층에 동시에 곱합니다. 이는 수학적으로 등가이며, $\gamma$는 오프라인으로 이전 선형 계층의 가중치 $W$와 편향 $b$에 흡수될 수 있습니다 ($W' = W \odot \gamma$, $b' = b \odot \gamma$). 이 과정을 통해 이중 모드 `key` 활성화는 정규 분포로 변환되고, `query` 활성화는 원래의 정규 분포를 유지합니다.
    * **이중 모드 분포 감지:** 모든 `post-Key-Linear` 활성화가 이중 모드 분포는 아니므로, Gaussian kernel density estimation(PDF)을 사용하여 피크를 감지하고, 피크 높이 및 피크 간 거리에 대한 제약 조건을 적용하여 정확하게 이중 모드 분포를 식별합니다.

2. **Adaptive Granularity Quantization (AGQ) 전략:**
    * **문제 식별 및 분석:** SAM의 다양한 어텐션 메커니즘(`self-attention`, `token-to-image`, `image-to-token`)은 `post-Softmax` 분포에서 큰 차이를 보입니다. 예를 들어, `token-to-image`는 초저(ultra-low) 값을 많이 가지는 반면, `image-to-token`과 `self-attention`은 더 높은 첨도(kurtosis)와 고(high) 값을 가집니다. 기존 로그 양자화는 이러한 다양성을 포착하기에 부족합니다.
    * **적응형 파라미터 $\tau$ 도입:** 기저(base)를 조절하는 적응형 파라미터 $\tau$를 가진 로그 양자화를 제안합니다. 양자화 및 역양자화 연산은 다음과 같습니다:
        $$
        a_q = \text{clamp}\left(\left\lfloor -\log_2 \frac{1}{\tau} \frac{a}{s_a} \right\rceil, 0, 2^k - 1\right) \\
        \hat{a} = s_a \cdot 2^{-\frac{a_q}{\tau}}
        $$
        여기서 $\tau \in \{2^0, 2^1, \dots, 2^n\}$는 하드웨어 효율성을 위해 2의 거듭제곱으로 선택됩니다.
    * **하드웨어 친화적 구현:** $\hat{a} \cdot \hat{v}$ 연산 시 비트 시프팅(bit-shifting)을 효율적으로 수행하기 위해 작은 룩업 테이블(LUT)을 활용합니다.
    * **최적 $\tau$ 선택 목표 함수:** `attention` 맵 $A$의 로컬 양자화 오류 대신, `attention` 맵 $A$와 `value` $V$의 행렬 곱셈 출력의 양자화 오류를 최소화하는 것을 목표 함수로 설정합니다:
        $$
        \arg \min_{\tau} E \left[ \|AV - \hat{A}_{\tau}V\|_{F}^2 \right]
        $$
        이를 통해 다양한 `post-Softmax` 시나리오와 비트 폭에서 저(low) 어텐션 점수와 고(high) 어텐션 점수 모두에 적합한 세분화 균형을 찾습니다.

## 📊 Results

PTQ4SAM은 다양한 비전 작업, 데이터셋 및 SAM 모델 변형(SAM-B, SAM-L, SAM-H)에 걸쳐 뛰어난 성능을 입증했습니다.

* **인스턴스 분할 (MS-COCO):**
  * 기존 통계 기반 및 학습 기반 PTQ 방법들(MinMax, OMSE, QDrop 등)을 큰 폭으로 능가합니다.
  * W6A6 (6비트 가중치/활성화) 설정에서 SAM-L 및 SAM-H에 대해 **무손실(lossless) 정확도**를 달성했으며, 이론적으로 **3.9배 가속화**와 **4.9배 저장 공간 절약** 효과를 얻었습니다.
  * 더 어려운 W4A4 (4비트) 설정에서도 AdaRound나 BRECQ가 비실용적일 때, PTQ4SAM은 QDrop을 5.1% (SAM-B, YOLOX)에서 6.3% (SAM-L, H-Deformable-DETR)까지 능가하며 사용 가능한 수준의 성능을 제공합니다.
* **의미론적 분할 (ADE20K):**
  * 양자화된 SAM 모델도 마스크 개선에 기여하며, PTQ4SAM-L은 6비트 설정에서 SAM-L 및 SAM-H의 풀프리시전 모델보다 더 나은 성능을 보이기도 했습니다. W4A4에서 SAM-L의 정확도를 QDrop보다 0.15% 더 높은 1.04% 향상시켰습니다.
* **회전 객체 탐지 (DOTA-v1.0):**
  * W6A6에서 풀프리시전 모델 대비 약 0.3%의 미미한 성능 저하를 보였습니다. W4A4에서는 AdaRound 및 BRECQ가 상당한 성능 저하를 겪는 반면, PTQ4SAM은 SAM-B에서 44% 이상, SAM-L에서 56% 이상의 정확도를 달성하며 QDrop을 각각 2.2%, 6.2% 능가했습니다.
* **저장 공간 절약 및 속도 향상:**
  * W6A6 SAM-B 모델은 FastSAM(1.98배)보다 더 나은 **2.96배의 이론적 가속**을 달성했습니다.
  * W4A4에서는 FLOPs를 70% 이상, 저장 공간을 85% 이상 감소시킵니다. 모델 규모가 커질수록 가속 비율과 메모리 절감 효과가 더욱 두드러집니다.
* **정성적 결과:** W4A4 저비트 양자화에서 PTQ4SAM은 다른 SOTA 방법들에 비해 더 완전하고 명확한 객체 경계를 가진 인스턴스 분할 결과를 시각적으로 보여주었습니다.

## 🧠 Insights & Discussion

* **함의:** PTQ4SAM은 SAM의 높은 컴퓨팅 및 메모리 요구 사항을 효과적으로 해결하여, 자원 제약이 있는 엣지 디바이스 등 실제 환경에서의 SAM 배포 가능성을 크게 높였습니다. 특히, 무손실에 가까운 정확도를 유지하면서 상당한 가속화와 저장 공간 절약을 달성했다는 점에서 그 실용적 가치가 매우 큽니다.
* **강점:** 이중 모드 분포와 다양한 `post-Softmax` 분포와 같은 SAM의 고유한 양자화 난제를 해결하기 위한 전문화된 전략(BIG, AGQ)의 중요성을 입증했습니다. 이러한 전략은 기존 PTQ 방식의 한계를 극복하고 SAM에 특화된 최적화를 제공합니다.
* **한계 및 향후 연구:** SAM에서 이중 모드 분포가 발생하는 근본적인 원인은 아직 명확하게 밝혀지지 않았습니다. 이는 향후 연구를 위한 흥미로운 방향이 될 수 있으며, SAM의 다른 아키텍처 부분에 대한 추가적인 양자화 최적화 연구 또한 가능합니다.

## 📌 TL;DR

SAM의 높은 연산 및 메모리 비용으로 인한 배포 문제를 해결하기 위해 PTQ4SAM 프레임워크를 제안한다. 이중 모드 분포를 정규 분포로 변환하는 **Bimodal Integration (BIG)**과 다양한 Softmax 분포에 적합한 세분화를 제공하는 **Adaptive Granularity Quantization (AGQ)**을 도입했다. 이 방법은 인스턴스, 의미론적, 객체 탐지 등 다양한 비전 작업에서 6비트 양자화 시 무손실에 가까운 정확도로 최대 3.9배 가속화 및 4.9배 스토리지 절감을 달성하며, 기존 SOTA PTQ 방식을 크게 능가한다.
