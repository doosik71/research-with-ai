# TinySAM: 효율적인 Segment Anything Model을 위한 한계 돌파

Han Shu, Wenshuo Li, Yehui Tang, Yiman Zhang, Yihao Chen, Houqiang Li, Yunhe Wang, Xinghao Chen

## 🧩 Problem to Solve

최근 Segment Anything Model (SAM)은 강력한 분할 능력을 보여 컴퓨터 비전 분야에서 큰 주목을 받고 있습니다. 그러나 SAM은 거대한 아키텍처와 막대한 연산 용량을 필요로 하여, 연산 제약이 있는 엣지 장치에서의 추가적인 응용을 어렵게 합니다. 기존의 효율적인 SAM 모델들은 부분적인 지식 증류로 인한 성능 저하나 프롬프트 기반 분할 기능 미흡 등의 한계를 가졌습니다.

## ✨ Key Contributions

* **하드 마이닝 전단계 지식 증류 (Hard Mining Full-Stage Knowledge Distillation):** 경량 학생 모델을 훈련하기 위해 하드 프롬프트 샘플링 및 하드 마스크 가중치 전략을 포함하는 전단계 지식 증류 방법을 제안했습니다.
* **프롬프트 기반 분할을 위한 사후 양자화 (Post-Training Quantization):** 프롬프트 기반 분할 작업에 사후 양자화 기법을 적용하여 모델의 연산 비용을 추가적으로 절감했습니다.
* **계층적 "Everything" 분할 전략 (Hierarchical Segmenting Everything Strategy):** "Everything" 추론 모드를 약 $2 \times$ 가속화하면서도 성능 저하를 거의 발생시키지 않는 계층적 전략을 제안했습니다.
* **TinySAM 개발:** 위의 모든 방법을 통합하여 원래 SAM 대비 연산량을 획기적으로 줄이면서도 강력한 제로샷 분할 성능을 유지하는 TinySAM을 달성했습니다 (예: SAM 대비 $100 \times$ 가속).

## 📎 Related Works

* **Segment Anything Model (SAM):** Kirillov et al. (2023)이 제안한 SAM은 이미지 인코더, 프롬프트 인코더, 마스크 디코더로 구성되며, SA-1B 데이터셋과 함께 임의의 객체에 대한 고품질 분할 및 제로샷 일반화 능력을 입증했습니다. 하지만 무거운 이미지 인코더로 인해 배포에 높은 연산 용량이 필요합니다.
* **지식 증류 (Knowledge Distillation):** Hinton et al. (2015)이 제안한 방법으로, 무거운 교사 네트워크의 출력을 사용하여 경량 학생 네트워크의 훈련을 지도합니다. MobileSAM (Zhang et al. 2023)은 이미지 임베딩에만 부분적인 증류를 사용했지만, 최종 마스크 예측에 대한 지도가 없어 성능 저하를 겪었습니다.
* **양자화 (Quantization):** 모델의 가중치나 활성화를 낮은 비트 폭으로 변환하여 저장 공간 및 연산 복잡도를 줄이는 방법입니다. QAT (Quantization-Aware Training)와 PTQ (Post-Training Quantization)가 있으며, PTQ는 적은 보정 데이터만으로 효율적인 양자화를 가능하게 합니다. 기존 연구는 주로 이미지 분류나 비전 트랜스포머에 집중되었으나, 프롬프트 기반 분할 작업에 대한 탐색은 드뭅니다.

## 🛠️ Methodology

* **TinySAM 개요:** 경량 TinyViT (Wu et al. 2022)를 이미지 인코더로 사용하며, 세 가지 핵심 모듈(하드 마이닝 전단계 지식 증류, 사후 양자화, 계층적 "Everything" 추론 모드)로 구성됩니다.
* **하드 마이닝 전단계 지식 증류:**
  * **다단계 증류 손실:** 이미지 인코더 출력 (이미지 임베딩), 마스크 디코더의 출력 토큰, 최종 마스크 예측 결과에 대한 증류 손실을 도입합니다.
    * $L_{\text{embedding}} = L(E^{\text{T}}_{\text{img}}(I), E^{\text{S}}_{\text{img}}(I))$
    * $L_{\text{token}} = L(T^{\text{T}}(E^{\text{T}}_{\text{img}}(I), q), T^{\text{S}}(E^{\text{S}}_{\text{img}}(I), q))$
    * $L_{\text{output}} = L(D^{\text{T}}_{\text{mask}}(E^{\text{T}}_{\text{img}}(I), q), D^{\text{S}}_{\text{mask}}(E^{\text{S}}_{\text{img}}(I), q))$
    * 총 증류 손실: $L_{\text{distill}} = \alpha \cdot L_{\text{embedding}} + \beta \cdot L_{\text{token}} + \gamma \cdot L_{\text{output}}$ (여기서 $L$은 $\ell_1$ 거리 함수를 사용)
    * 총 훈련 손실은 증류 손실 $L_{\text{distill}}$, 마스크 예측을 위한 $L_{\text{mask}}$ (focal loss와 dice loss 조합), IoU 예측을 위한 $L_{\text{ious}}$의 합으로 구성됩니다.
  * **하드 마스크 가중치 (Hard Mask Weighting):** 교사 및 학생 네트워크의 예측 마스크와 Ground Truth 마스크 간의 IoU를 기반으로 마스크 난이도 $H_i$를 계산하고, 어려운 마스크에 더 높은 가중치를 부여하여 증류 손실을 조절합니다.
    * $H_i = \text{sigmoid}(\frac{\text{IoU}(M^{\text{T}}_i, M^{\text{GT}}_i)}{\text{IoU}(M^{\text{S}}_i, M^{\text{GT}}_i) + \varepsilon} - 1)$
  * **하드 프롬프트 샘플링 (Hard Prompt Sampling):** 정답 마스크와 예측 마스크의 차이 집합($M_{\text{gt}} - M_i$)에서 반복적으로 프롬프트 포인트를 샘플링하여, 학생 네트워크가 예측하기 어려운 영역에 집중하도록 훈련을 유도합니다.
* **양자화:**
  * 균일 대칭 양자화 $x_q = \text{clip}(\text{round}(\frac{x}{s}), -2^{b-1}, 2^{b-1}-1)$를 사용합니다.
  * 마스크 및 IoU의 Kullback-Leibler (KL) divergence를 태스크 손실로 사용하여 스케일링 인자를 최적화합니다: $L=\text{KL}(\hat{y}_{\text{pred}}, y_{\text{pred}}) + \text{KL}(\hat{y}_{\text{iou}}, y_{\text{iou}})$.
  * Softmax 및 GELU 이후의 불균일한 활성화 분포에 대응하기 위해 특징을 두 그룹으로 분할하고 두 개의 스케일링 인자를 적용하여 양자화 오류를 줄입니다.
* **계층적 "Everything" 분할:**
  * **1단계 (희소 프롬프트):** 먼저 이미지 측면당 $1/4$ 포인트(총 $1/16$ 포인트)의 희소 그리드 프롬프트를 사용하여 마스크를 생성하고, 높은 신뢰도 영역을 식별합니다.
  * **2단계 (조밀 프롬프트):** 1단계에서 높은 신뢰도로 분할된 영역의 포인트는 무시하고, 나머지 미분할 영역에 대해서만 원본 설정과 동일한 밀도로 프롬프트 포인트를 샘플링하여 추가 분할합니다.
  * 두 단계의 결과를 병합 및 후처리하여 중복 계산을 줄이고 추론 시간을 약 $50\%$ 단축합니다.

## 📊 Results

* **제로샷 인스턴스 분할 (Zero-Shot Instance Segmentation):**
  * COCO 및 LVIS v1 데이터셋에서 기존의 효율적인 SAM 모델들(FastSAM, MobileSAM, EfficientSAM, SlimSAM) 및 SAM의 다른 변형들보다 우수한 성능을 달성했습니다.
  * TinySAM은 FastSAM 대비 $4\%$ AP 개선, $9.5\%$ MACs, $25\%$ 지연 시간을 보였습니다.
  * MobileSAM과 동일한 연산 비용에서 COCO에서 $1.3\%+$ AP, LVIS v1에서 $1.9\%+$ AP를 추가로 개선했습니다.
  * 8비트 양자화 버전인 Q-TinySAM은 SlimSAM보다 COCO에서 $0.1\%+$ AP, LVIS v1에서 $0.2\%+$ AP를 달성하면서 MACs는 $39\%$, 지연 시간은 $21.8\%$ 수준에 불과했습니다.
* **제로샷 포인트 유효 마스크 평가 (Zero-shot Points Valid Mask Evaluation):**
  * LVIS, DOORS, BBBC038v1, TimberSeg 데이터셋에서 MobileSAM보다 우수하거나 유사한 mIoU를 달성하며, SAM-B에 근접한 성능을 보였습니다.
* **"Everything" 모드 가속:**
  * 제안된 계층적 "Everything" 추론 전략은 COCO val2017에서 원래의 $32 \times 32$ 포인트 그리드 전략과 비슷한 결과를 얻으면서 추론 시간을 약 $50\%$ 단축했습니다.

## 🧠 Insights & Discussion

* **효율성 및 성능 균형:** TinySAM은 전단계 지식 증류, 사후 양자화, 계층적 "Everything" 분할 전략의 시너지 효과를 통해 SAM의 연산 비용을 획기적으로 줄이면서도 강력한 제로샷 분할 성능을 유지하는 데 성공했습니다. 이는 복잡한 대규모 모델의 실용적인 배포 가능성을 크게 높였습니다.
* **지식 증류의 효과:** 이미지 임베딩, 출력 토큰, 최종 마스크 예측 등 다단계에서 교사 네트워크의 지식을 증류하고, 특히 하드 마이닝 전략(하드 프롬프트 샘플링 및 하드 마스크 가중치)을 적용함으로써 경량 학생 모델의 학습 효율과 성능을 극대화할 수 있었습니다.
* **"Everything" 모드 최적화의 중요성:** 계층적 분할 전략은 대규모 객체를 먼저 처리하고 나머지 미분할 영역에 집중하여 불필요한 연산을 효과적으로 줄이는 혁신적인 방법을 제시합니다. 이는 전반적인 추론 시간 단축에 크게 기여합니다.
* **양자화의 실용성:** 8비트 양자화가 성능 저하를 최소화하면서 연산 효율성을 크게 높일 수 있음을 입증하여, 엣지 디바이스와 같은 리소스 제약 환경에서의 모델 배포를 위한 중요한 단계를 제공합니다.

## 📌 TL;DR

**문제:** SAM은 강력한 분할 모델이지만, 막대한 연산 비용으로 인해 엣지 디바이스에 배포하기 어렵습니다.
**해결책:** TinySAM은 1) 하드 프롬프트 샘플링 및 하드 마스크 가중치 기법을 포함한 **전단계 지식 증류**, 2) 프롬프트 기반 분할 태스크에 적합한 **사후 양자화**, 그리고 3) "Everything" 모드의 추론을 가속화하는 **계층적 분할 전략**의 세 가지 주요 기법을 제안합니다.
**핵심 발견:** TinySAM은 기존 SAM 대비 연산량을 $100 \times$ 이상 절감하면서도 다른 경량 SAM 모델들보다 뛰어난 제로샷 분할 성능을 달성합니다. 특히, "Everything" 모드에서는 거의 성능 저하 없이 추론 시간을 $2 \times$ 가속화합니다.
