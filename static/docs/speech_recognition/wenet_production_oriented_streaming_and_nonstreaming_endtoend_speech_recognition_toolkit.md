# WeNet: Production Oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit

Zhuoyuan Yao, Di Wu, Xiong Wang, Binbin Zhang, Fan Yu, Chao Yang, Zhendong Peng, Xiaoyu Chen, Lei Xie, Xin Lei

## 🧩 Problem to Solve

기존 End-to-End (E2E) 자동 음성 인식(ASR) 모델은 연구 분야에서 큰 진전을 보였지만, 실제 프로덕션 환경에 배포하는 데에는 여러 과제가 있습니다. 주요 문제점은 다음과 같습니다:

1. **스트리밍 문제**: 낮은 지연 시간으로 빠른 응답이 필요한 스트리밍 추론은 많은 시나리오에서 필수적이지만, LAS(Listen, Attend and Spell)나 Transformer와 같은 일부 E2E 모델은 스트리밍 방식으로 실행하기 어렵거나 상당한 정확도 손실이 발생합니다.
2. **스트리밍 및 비스트리밍 모드 통합**: 스트리밍 및 비스트리밍 시스템은 일반적으로 개별적으로 개발됩니다. 단일 모델로 두 모드를 통합하면 개발 노력, 훈련 비용 및 배포 비용을 줄일 수 있습니다.
3. **생산성 문제**: E2E 모델을 실제 프로덕션 애플리케이션으로 적용하기 위해서는 모델 아키텍처, 애플리케이션 및 런타임 플랫폼 측면에서 추론 워크플로우를 신중하게 설계해야 합니다. 특히 오토회귀 빔 서치 디코딩의 복잡성과 엣지 디바이스에서의 계산 및 메모리 비용이 중요하게 고려되어야 합니다.

이 논문은 이러한 연구와 배포 사이의 간극을 줄이고, E2E ASR 모델의 실제 배포를 위한 효율적인 방법을 제공하는 것을 목표로 합니다.

## ✨ Key Contributions

- **WeNet 툴킷 제안**: 스트리밍 및 비스트리밍 E2E 음성 인식을 단일 모델로 통합하는 새로운 U2(Unified Two-pass) 접근 방식을 구현한 오픈 소스 생산 지향적(production-oriented) E2E ASR 툴킷을 소개합니다.
- **연구-생산 간극 해소**: E2E ASR 모델의 연구와 배포 간의 간극을 좁히는 데 중점을 두며, TorchScript 및 LibTorch를 활용하여 연구 모델을 프로덕션 환경에 직접 안전하게 내보낼 수 있도록 합니다.
- **통합된 스트리밍/비스트리밍 솔루션(U2 프레임워크)**: CTC(Connectionist Temporal Classification)와 AED(Attention-based Encoder-Decoder)를 결합한 U2 프레임워크를 통해 정확하고 빠르며 통합된 E2E 모델을 제공합니다. 이는 산업 적용에 매우 유리합니다.
- **동적 청크 기반 훈련**: 스트리밍 및 비스트리밍 모델을 통합하기 위해 동적 청크(dynamic chunk) 훈련 기법을 도입하여 모델이 임의의 청크 크기로 예측하도록 학습시킵니다.
- **이식성 있는 런타임 지원**: 서버(x86) 및 임베디드(ARM 기반 Android)를 포함한 다양한 플랫폼에서 WeNet 훈련 모델을 호스팅하는 방법을 제공합니다.
- **경량화 및 PyTorch 기반**: E2E 음성 인식을 위해 특별히 설계되었으며, Kaldi와 같은 다른 툴킷에 대한 의존성 없이 PyTorch 및 그 생태계에만 의존하여 설치 및 사용이 간편합니다.
- **양자화(Quantization) 지원**: float32 모델과 int8 양자화 모델을 모두 지원하여 임베디드 장치에서 추론 속도를 크게 향상시킵니다.

## 📎 Related Works

- **E2E ASR 모델**:
  - CTC (Connectionist Temporal Classification) [1, 2]
  - RNN-T (Recurrent Neural Network Transducer) [3, 4, 5, 6]
  - AED (Attention-based Encoder-Decoder) (예: LAS [8], Transformer [15]) [7, 8, 9, 10, 11]
- **E2E ASR 툴킷**:
  - ESPnet [25]: E2E 음성 연구를 위한 인기 있는 오픈 소스 플랫폼으로, Chainer 및 PyTorch를 사용합니다. WeNet은 ESPnet의 많은 설계를 참조했습니다.
  - Kaldi [26]: 기존 하이브리드 ASR 프레임워크의 대표적인 툴킷입니다. WeNet은 Kaldi에 대한 의존성을 제거하여 경량화했습니다.
- **관련 기술**:
  - Transformer [15], Conformer [27]: 인코더 아키텍처로 사용됩니다.
  - SpecAugment [29]: 오버피팅 완화를 위한 데이터 증강 기법.
  - TorchScript, PyTorch JIT, LibTorch, TensorRT [23], OpenVINO, MNN [24], NCNN: 모델 배포 및 런타임 최적화를 위한 도구들.

## 🛠️ Methodology

WeNet의 핵심은 스트리밍 및 비스트리밍 E2E ASR을 단일 모델로 통합하는 U2(Unified Two-pass) 아키텍처와 프로덕션 중심의 시스템 설계입니다.

### 모델 아키텍처 (U2)

U2 모델은 다음 세 부분으로 구성됩니다 (그림 1 참조):

1. **Shared Encoder**: Transformer [15] 또는 Conformer [27] 레이어로 구성되며, 스트리밍 모드에서 낮은 지연 시간을 유지하기 위해 제한된 우측 컨텍스트(right context)만을 고려합니다.
2. **CTC Decoder**: Shared Encoder의 출력을 CTC 활성화로 변환하는 선형 레이어로 구성됩니다. 첫 번째 패스에서 스트리밍 모드로 실행됩니다.
3. **Attention Decoder**: 여러 Transformer 디코더 레이어로 구성됩니다. 두 번째 패스에서 CTC 가설을 재평가하여 더 정확한 결과를 제공합니다.

### 훈련

- **결합된 손실 함수**: CTC 손실($L_{CTC}(x,y)$)과 AED 손실($L_{AED}(x,y)$)을 가중치 $\lambda$를 사용하여 결합하여 훈련합니다:
  $$L_{combined}(x,y) = \lambda L_{CTC}(x,y) + (1-\lambda)L_{AED}(x,y)$$
  여기서 $x$는 음향 특징, $y$는 해당 레이블입니다.
- **동적 청크 훈련**: 비스트리밍 및 스트리밍 모델을 통합하기 위해 입력 오디오를 청크 크기 $C$로 분할합니다. 훈련 중 청크 크기를 1부터 현재 발화의 최대 길이까지 동적으로 변화시켜 모델이 임의의 청크 크기로 예측하도록 학습시킵니다. 이를 통해 실행 시 청크 크기 조절로 정확도와 지연 시간의 균형을 맞출 수 있습니다.
- **On-the-fly Feature Extraction**: Torchaudio를 기반으로 훈련 중 원본 PCM 데이터에서 즉석으로 특징을 추출하며, 시간, 주파수, 특징 레벨에서 데이터 증강을 수행하여 데이터 다양성을 높입니다.
- **분산 훈련**: PyTorch의 DistributedDataParallel (DDP)를 사용하여 멀티 GPU 훈련을 지원하여 훈련 속도를 높입니다.

### 디코딩

연구 단계에서는 네 가지 디코딩 모드를 지원하지만, 프로덕션용으로는 `attention_rescoring` 모드만 지원합니다:

- `attention`: AED 부분에 표준 오토회귀 빔 서치 적용.
- `ctc_greedysearch`: CTC 부분에 CTC 그리디 서치 적용 (가장 빠름).
- `ctc_prefixbeamsearch`: CTC 부분에 CTC 접두사 빔 서치 적용 (n-best 후보 생성).
- `attention_rescoring` (생산용 솔루션): 먼저 CTC 접두사 빔 서치를 통해 n-best 후보를 생성하고, 이후 AED 디코더로 이 후보들을 재평가합니다.

### 시스템 설계

- **PyTorch 기반 스택**: 모든 하위 스택은 PyTorch 및 그 생태계를 기반으로 합니다.
- **모델 개발 및 내보내기**: TorchScript를 사용하여 모델을 개발하고, `torch.jit`를 통해 모델을 프로덕션으로 직접 내보낼 수 있습니다.
- **런타임**: 내보낸 모델은 LibTorch 라이브러리를 통해 호스팅되며, float32 및 int8 양자화 모델을 모두 지원합니다. C++ API 라이브러리와 실행 가능한 데모가 x86 서버 및 Android ARM 플랫폼용으로 제공됩니다.

## 📊 Results

실험은 150시간 분량의 중국어 AISHELL-1 데이터셋에서 수행되었으며, 15,000시간 분량의 대규모 만다린 데이터셋에서도 추가 실험이 진행되었습니다.

### 통합 모델 평가 (AISHELL-1)

- **CER (Character Error Rate)**: 동적 청크 전략을 사용한 통합 모델(M2)은 비스트리밍 모델(M1)과 비교하여 전체 어텐션(full attention)의 경우 비슷한 성능을 보였고, 제한된 청크 크기(16/8/4) 스트리밍 경우에서도 유망한 결과를 나타냈습니다.
  - `attention_rescoring` 모드는 CTC 결과 대비 CER을 지속적으로 개선했으며, 가장 좋은 종합 성능을 보였습니다 (예: 비스트리밍 풀 어텐션에서 5.30%, 청크 16에서 5.52%).
  - `ctc_greedysearch` 및 `ctc_prefixbeamsearch`는 청크 크기가 줄어들수록 성능이 크게 저하되었습니다.
- **RTF (Real-Time Factor)**: `attention_rescoring` 모드는 `attention` 모드보다 더 빠르고 RTF가 낮습니다. 이는 `attention` 모드가 오토회귀적 절차인 반면 `attention_rescoring`은 그렇지 않기 때문입니다.

### 런타임 벤치마크 (양자화, RTF, 지연 시간)

- **양자화**: int8 양자화 모델은 float32 모델과 비교하여 CER에 거의 영향을 미치지 않았습니다 (0.01-0.04%p 차이).
- **RTF**:
  - 청크 크기가 작아질수록 RTF는 증가합니다 (더 많은 순방향 계산 반복).
  - 양자화는 온디바이스(Android)에서 약 2배의 속도 향상을 가져왔으며, 서버(x86)에서도 약간의 개선을 보였습니다.
    - 예: Android에서 청크 16의 경우 float32 0.251 → int8 0.114.
- **지연 시간 (Latency)**:
  - 모델 지연 시간($L_1$)은 청크 크기에 비례합니다 (청크가 작을수록 $L_1$ 감소).
  - 재평가 비용($L_2$)은 청크 크기에 관계없이 거의 동일했습니다.
  - 최종 지연 시간($L_3$)은 주로 재평가 비용($L_2$)에 의해 지배됩니다.

### 15,000시간 만다린 데이터셋 실험

- **Conformer 인코더 사용**: Conformer 인코더를 사용하여 대규모 데이터셋에 대한 모델의 능력을 시연했습니다.
- **성능**: U2 모델은 전반적으로 Conformer 베이스라인과 유사한 결과를 달성했으며, 특히 AISHELL-1 테스트셋에서는 풀 어텐션 추론 시 더 나은 결과를 보였습니다 (Conformer 3.96% vs. U2 3.70%). 이는 AISHELL-1의 긴 발화 길이가 U2 모델의 어텐션 디코더를 통한 재평가 능력으로부터 더 큰 이점을 얻기 때문으로 분석됩니다.
- **스트리밍 성능**: 청크 크기가 16일 때도 CER이 크게 나빠지지 않았습니다.

## 🧠 Insights & Discussion

- **연구-생산 통합의 중요성**: WeNet은 E2E ASR 연구 결과를 실제 프로덕션 환경에 효과적으로 적용하는 데 중점을 두어, 기존 툴킷들이 간과했던 중요한 문제를 해결합니다. TorchScript와 LibTorch의 활용은 이 통합을 가능하게 하는 핵심 요소입니다.
- **U2 프레임워크의 효율성**: CTC와 어텐션 디코더를 결합하고 동적 청크 훈련을 통해 단일 모델로 스트리밍 및 비스트리밍 시나리오 모두에서 균형 잡힌 성능과 효율성을 달성했습니다. 이는 산업 현장에서의 개발 및 배포 비용 절감에 기여합니다.
- **`attention_rescoring`의 강점**: 디코딩 모드 중 `attention_rescoring`이 정확도와 RTF 측면에서 가장 유리한 것으로 입증되어, 프로덕션 환경에서 최종 솔루션으로 채택되었습니다. 이는 CTC의 속도와 AED의 정확도를 결합하는 효과적인 전략입니다.
- **양자화의 실용성**: int8 양자화는 성능 손실을 최소화하면서 온디바이스 환경에서 상당한 추론 속도 향상을 제공하여, 엣지 디바이스 ASR 애플리케이션에 대한 WeNet의 잠재력을 높입니다.
- **지연 시간 최적화 과제**: 최종 지연 시간이 재평가 비용에 의해 지배된다는 점은 향후 재평가 단계의 최적화를 통해 사용자 경험을 더욱 개선할 수 있는 여지를 시사합니다.
- **확장성 및 유연성**: 15,000시간 규모의 대규모 데이터셋에서도 Conformer 인코더를 활용하여 우수한 성능을 보여주며, 다양한 도메인에 대한 모델의 확장성과 유연성을 입증했습니다.

## 📌 TL;DR

WeNet은 E2E 음성 인식 모델의 연구와 프로덕션 배포 간의 간극을 해소하기 위한 오픈 소스 툴킷입니다. 스트리밍 및 비스트리밍 ASR을 단일 모델로 통합하는 U2 투 패스 접근 방식을 제안하며, 이는 Transformer 또는 Conformer 인코더와 CTC/Attention 디코더의 결합된 아키텍처를 기반으로 합니다. 동적 청크 기반 훈련과 `attention_rescoring` 디코딩 전략을 통해 높은 정확도와 낮은 지연 시간을 달성하며, TorchScript 내보내기 및 LibTorch 런타임, 양자화 지원 등을 통해 실제 환경에 최적화된 경량 솔루션을 제공합니다. AISHELL-1 및 대규모 만다린 데이터셋 실험에서 WeNet은 효율적인 런타임 성능과 함께 경쟁력 있는 CER을 입증했습니다.
