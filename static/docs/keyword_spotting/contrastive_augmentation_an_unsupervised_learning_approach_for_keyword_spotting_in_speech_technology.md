# Contrastive Augmentation: An Unsupervised Learning Approach for Keyword Spotting in Speech Technology

Weinan Dai, Yifeng Jiang, Yuanjing Liu, Jinkun Chen, Xin Sun, and Jinglei Tao

## 🧩 Problem to Solve

음성 기술의 핵심 요소인 KWS(키워드 스포팅)는 훈련을 위한 방대한 양의 레이블링된 데이터, 특히 긍정 샘플을 확보하는 데 지속적인 어려움을 겪고 있습니다. 키워드가 변경될 때마다 새로운 대상 샘플을 수집해야 하는 노동 집약적인 과정은 이러한 문제를 더욱 심화시킵니다. 또한, 음성 데이터는 잡음이 많고 복잡하여 키워드와 관련된 핵심 구절만 중요한데, 기존 합성곱 방식은 모든 단어 창을 동등하게 처리하여 불필요하거나 중복된 정보를 포함할 수 있다는 문제가 있습니다.

## ✨ Key Contributions

- **비지도 대조 학습 및 증강 기법 도입:** 레이블링되지 않은 데이터셋으로 신경망을 훈련할 수 있게 하여, 제한된 레이블 데이터셋에서도 성능 향상을 가능하게 합니다.
- **압축된 합성곱 아키텍처 제안:** KWS 작업에서 잠재적인 중복 및 비정보적인 정보를 처리하고, 모델이 지역적 특징과 장기적 정보를 동시에 학습할 수 있도록 합니다.
- **음성 증강 기반 비지도 학습 방법 개발:** 속도나 볼륨 변화에도 불구하고 동일한 키워드를 포함하는 음성 발화는 유사한 고수준 특징 표현을 가져야 한다는 전제 하에, 병목 계층 특징과 오디오 재구성 정보 간의 유사성을 활용하여 보조 훈련을 수행합니다.
- **새로운 손실 함수 정의:** 원본 음성과 증강된 음성 간의 유사성을 평가하고, 미니 배치 내에서 샘플 간의 근접성을 평가하기 위한 비지도 손실($L_{\text{sim}}$, $L_x$, $L_{x}^{\text{aug}}$) 및 대조 손실($L_{\text{Dual}}$)을 통합적으로 사용합니다.
- Google Speech Commands V2 데이터셋에서 강력한 성능을 달성하여 기존 비지도 학습 방식(CPC, APC, MPC)을 능가합니다.

## 📎 Related Works

- **데이터 증강:** ASR(자동 음성 인식) 및 KWS에서 훈련 데이터셋을 풍부하게 하는 효과적인 기술로, 성도 길이 교란(vocal tract length perturbation), 속도 교란(speed-perturbation), 잡음 오디오 신호 도입 등 다양한 방법이 탐구되었습니다. 최근에는 SpecAugment, WavAugment와 같은 스펙트럼 도메인 증강 기법이 개발되었습니다. 본 논문은 속도 및 볼륨 교란을 활용합니다.
- **비지도 및 약한 지도 학습:** 레이블링된 데이터 수집의 어려움으로 인해 Noisy Student Training과 같은 준지도 학습 기법이 ASR 및 KWS에 적용되었으며, 순수 비지도 KWS 방법도 연구되었습니다.
- **KWS 벤치마크:** Google Speech Commands V2 데이터셋은 KWS 분야의 새로운 아이디어를 위한 널리 사용되는 벤치마크입니다. 다양한 아키텍처와 방법론(예: 합성곱 순환 신경망, MatchboxNet, EdgeCRNN, Triplet loss 기반 임베딩)이 이 데이터셋에서 실험되었습니다. 본 논문은 이 데이터셋에서 CPC, APC, MPC와 같은 다른 비지도 학습 방법과 비교 평가를 수행합니다.

## 🛠️ Methodology

본 논문은 KWS 작업을 시퀀스 분류 문제로 프레임화하며, 제안하는 CAB-KWS 모델은 다음의 주요 구성 요소로 이루어집니다.

1. **압축된 합성곱 계층 (Compressed Convolutional Layer):**
   - **프레임 합성곱 (Frame Convolution):** 입력 시퀀스 $X$에 합성곱을 적용하여 특징을 추출합니다. $i$-번째 필터와 $j$-번째 프레임에 대한 합성곱은 `$$x_{i,j} = \text{conv}(\{x_j, x_{j+1}, \ldots, x_{j+k_i-1}\}; W_i^x)$$` 로 표현됩니다.
   - **어텐션 기반 소프트 풀링 (Attention-based Soft-pooling):** `$$o_i^p = \sum_{q=j}^{j+g-1} \beta_{i,q} x_{i,q}$$` 를 통해 프레임 표현에서 중복 정보를 제거합니다. 이는 국부 기반 어텐션 점수 $\alpha_{i,j}$ (소프트맥스 함수 적용)를 사용하여 가중치를 부여한 합산 방식으로 이루어집니다.
   - **잔여 합성곱 블록 (Residual Convolution Block):** `$$r_{i,p} = \text{ResidualBlock}(\{o_{i,p}, \ldots, o_{i,p+a-1}\})$$`을 통해 압축된 특징 위에 잔여 블록을 추가하여 기울기 소실을 방지하고 훈련을 용이하게 합니다. 배치 정규화 대신 그룹 정규화를 사용합니다.
2. **ResLayer 블록 (ResLayer Block):**
   - **트랜스포머 블록 (Transformer Block):** 압축된 합성곱 계층의 출력 $R$을 받아 셀프-어텐션 메커니즘을 통해 시퀀스 내 장기 의존성을 포착합니다: `$$E_{\text{tran}} = \text{Self-Attention}^M(R)$$`.
   - **특징 선택 계층 (Feature Selecting Layer):** `$$E_{\text{feat}} = \text{Concat}(E_{\text{tran}}[T-r, T])$$` 를 통해 트랜스포머 블록의 출력에서 키워드 정보를 추출합니다 (마지막 $r$ 프레임을 연결).
   - **병목 계층 및 투영 계층 (Bottleneck and Projection Layers):** `$$E_{\text{bn}} = \text{FC}_{\text{bn}}(E_{\text{feat}})$$` , `$$\tilde{Y} = \text{FC}_{\text{proj}}(E_{\text{bn}})$$` 를 통해 숨겨진 상태를 예측된 분류 클래스 $\tilde{Y}$로 매핑합니다. 지도 학습 및 모델 미세 조정을 위해 교차 엔트로피 손실 $L_{\text{ce}} = \text{CE}(Y, \tilde{Y})$을 계산합니다.
3. **증강 방법 (Augmentation Method):**
   - **속도 증강:** `$$X_{\text{aug}} = A(\lambda_{\text{speed}} t)$$`
   - **볼륨 증강:** `$$X_{\text{aug}} = \lambda_{\text{volume}} A(t)$$`
   - 다양한 $\lambda_{\text{speed}}$ 및 $\lambda_{\text{volume}}$ 비율을 사용하여 원본 음성 $X$와 증강된 음성 $X_{\text{aug}}$ 쌍을 생성합니다. 동일 키워드는 속도나 볼륨 변화에도 유사한 고수준 특징 표현을 가져야 한다는 가정을 기반으로 합니다.
4. **비지도 대조 학습 손실 (Unsupervised Contrastive Learning Loss):**
   - **사전 훈련 (Pre-training):** 레이블링되지 않은 음성 데이터를 사용하여 병목 특징을 추출합니다.
     - `$$L_{\text{sim}} = \frac{1}{U_{\text{bn}}} \sum_{u=0}^{U_{\text{bn}}} |E_{\text{bn}}(u) - E_{\text{aug,bn}}(u)|^2$$` : 원본 음성과 증강된 음성의 병목 계층 출력 간의 MSE(평균 제곱 오차)를 최소화하여 유사성을 강화합니다.
     - **오디오 재구성:** 입력 Fbank 벡터 $X$의 시간 축을 따라 평균 벡터 $\bar{X}$를 계산하고, 병목 계층에서 이를 $\tilde{X}$로 재구성합니다. `$$L_x = \frac{1}{U_x} \sum_{u=0}^{U_x} |\bar{X}(u) - \tilde{X}(u)|^2$$` 와 `$$L_{x}^{\text{aug}} = \frac{1}{U_x} \sum_{u=0}^{U_x} |\bar{X}_{\text{aug}}(u) - \tilde{X}_{\text{aug}}(u)|^2$$` 를 통해 재구성 오류를 최소화합니다.
     - **대조 손실 ($L_{\text{Dual}}$):** 미니 배치 내에서 원본 음성과 증강된 음성을 긍정 쌍으로, 다른 샘플들을 부정 쌍으로 간주하여 특징 공간에서 유사한 샘플은 가깝게, 다른 샘플은 멀리 떨어뜨리도록 학습합니다.
     - **최종 비지도 손실:** `$$L_{\text{ul}} = \lambda_1 L_{\text{sim}} + \lambda_2 L_x + \lambda_3 L_{x}^{\text{aug}} + \lambda_4 L_{\text{Dual}}$$`
   - **미세 조정 (Fine-tuning):** 사전 훈련된 네트워크를 지도 KWS 데이터에 대해 미세 조정합니다. 이 단계에서는 모든 파라미터를 업데이트합니다.

## 📊 Results

- **데이터 증강 효과:** 속도 증강을 적용한 CAB-KWS 모델(Dev 87.3%, Eval 85.8%)은 Google의 Sainath and Parada 모델(Eval 84.7%)과 볼륨 증강이 없는 CAB-KWS 모델(Dev 86.4%, Eval 85.3%)보다 높은 분류 정확도를 달성했습니다. 이는 데이터 증강이 KWS 모델 성능 향상에 효과적임을 보여줍니다.
- **제안된 기법들의 시너지 효과 (Ablation Study):**
  - 속도 사전 훈련(sp-pre)이 볼륨 사전 훈련(vo-pre)보다 더 효과적이었습니다.
  - 볼륨 및 속도 사전 훈련(vo-sp-pre)을 결합했을 때 성능이 추가적으로 향상되었습니다.
  - 볼륨, 속도 증강, 그리고 대조 학습을 모두 포함한 완전한 CAB-KWS 모델(`vo-sp-pre-contras`)이 Speech Commands 데이터셋 사전 훈련 시 (Dev 88.1%, Eval 88.3%), Librispeech-100 사전 훈련 시 (Dev 88.4%, Eval 88.5%)로 가장 높은 분류 정확도를 달성했습니다. 이는 여러 사전 훈련 기법의 통합이 모델 성능을 극대화함을 보여줍니다.
- **다른 비지도 모델과의 비교:** CAB-KWS(full) 모델은 CPC, APC, MPC와 같은 다른 비지도 사전 훈련 모델들에 비해 지속적으로 우수한 분류 정확도를 보였습니다. 특히 Librispeech-100으로 사전 훈련했을 때 가장 좋은 성능(Dev 88.4%, Eval 88.5%)을 나타냈으며, 이는 경쟁 모델들 중 최고 성능인 CPC(Librispeech-100 사전 훈련 시 Dev 87.8%, Eval 87.4%)보다 높았습니다.
- **사전 훈련 단계의 영향:** 사전 훈련 단계 수가 많을수록(예: 30K 단계) 분류 정확도가 가장 높고 미세 조정 수렴 속도가 가장 빨랐습니다.

## 🧠 Insights & Discussion

본 논문은 KWS 과제에서 레이블링된 데이터의 부족 문제를 해결하는 강력한 비지도 학습 접근 방식을 제시합니다. 제안된 CAB-KWS는 음성 증강과 대조 학습을 결합하여, 속도나 볼륨 변화에도 강인한 특징 표현을 학습하고 미니 배치 내 대조 손실을 통해 훈련 효율성을 높였습니다. 이는 레이블링된 데이터 수집의 부담을 줄이고 키워드 변경에 대한 시스템의 유연성을 향상시킵니다. 기존의 비지도 KWS 방법론들을 능가하는 성능은 이 접근 방식의 유효성을 입증합니다. 향후 연구에서는 이 방법론을 다른 음성 작업에 적용하거나, 추가적인 증강 기법 및 아키텍처 탐색을 통해 성능을 더욱 개선할 수 있을 것입니다. 이는 음성 인식 시스템, 특히 음성 제어 시스템 및 대화형 에이전트의 신뢰성을 높이는 데 중요한 기여를 할 것입니다.

## 📌 TL;DR

KWS의 레이블 데이터 부족 문제를 해결하기 위해, 본 논문은 압축된 합성곱 아키텍처와 음성 증강 기반 비지도 대조 학습을 결합한 CAB-KWS를 제안합니다. 이 방법은 속도 및 볼륨 증강으로 원본/증강 음성 쌍을 생성하고, 병목 특징 유사성, 오디오 재구성, 미니 배치 내 대조 손실을 포함하는 통합 비지도 손실 함수($L_{\text{ul}}$)로 사전 훈련을 수행합니다. Google Speech Commands V2 데이터셋에서 실험 결과, CAB-KWS는 기존 비지도 학습 모델(CPC, APC, MPC) 및 단일 증강 방법 대비 우수한 성능을 달성하여, 레이블링되지 않은 데이터를 효율적으로 활용하며 KWS 시스템의 강인성을 크게 향상시켰습니다.
