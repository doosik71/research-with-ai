# ASI-Seg: Audio-Driven Surgical Instrument Segmentation with Surgeon Intention Understanding

* **저자**: Zhen Chen, Zongming Zhang, Wenwu Guo, Xingjian Luo, Long Bai, Jinlin Wu, Hongliang Ren, Hongbin Liu
* **발표연도**: 2024
* **arXiv**: <https://arxiv.org/abs/2407.19435v1>

## 1. 논문 개요

이 논문은 수술 장면에서 **외과의의 음성 명령(audio command)** 을 이해하고, 그 의도에 맞는 **특정 surgical instrument만 분할(segmentation)** 하는 프레임워크인 **ASI-Seg**를 제안한다. 기존 surgical instrument segmentation 연구들은 일반적으로 입력 영상 안에 존재하는 미리 정의된 모든 기구를 한꺼번에 분할하거나, category를 기준으로 전체를 찾는 방식이었다. 그러나 실제 수술에서는 모든 기구가 항상 중요한 것이 아니며, 수술 단계나 외과의의 현재 작업에 따라 **지금 당장 필요한 도구 하나 또는 일부만 집중적으로 인식하는 능력**이 중요하다.

논문이 문제 삼는 핵심은 바로 이 지점이다. 기존 방법은 “무엇이 영상 안에 있는가”를 중심으로 설계되었지만, 실제 임상에서는 “지금 외과의가 보고 싶어 하는 도구가 무엇인가”가 더 중요할 수 있다. 예를 들어 여러 도구가 동시에 보이는 장면에서 외과의가 특정 instrument만 필요로 한다면, 나머지 도구까지 모두 강조하는 방식은 오히려 인지 부담을 늘릴 수 있다. 논문은 이를 해결하기 위해 **surgeon intention understanding**을 segmentation에 직접 연결한다.

또 하나의 중요한 배경은 SAM(Segment Anything Model) 계열 방법이다. SAM은 prompt를 주면 특정 객체를 분할할 수 있지만, 실제 수술실에서 point나 bounding box 같은 **manual prompt**를 매번 사람이 입력하는 것은 비현실적이다. 따라서 이 논문은 “manual prompt 없이, 음성만으로, surgeon intention에 따라 필요한 instrument를 분할할 수 있는가?”라는 실용적이고 임상적인 문제를 다룬다.

정리하면, 이 논문의 목표는 단순히 segmentation accuracy를 높이는 것만이 아니라, **수술실의 workflow에 맞는 intention-aware segmentation 시스템**을 만드는 데 있다. 이는 surgical scene understanding을 더 실용적인 방향으로 확장하려는 시도라고 볼 수 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **음성에서 의도를 인식하고, 그 의도를 segmentation prompt로 변환하여, 원하는 instrument만 골라서 분할한다**는 것이다. 이를 위해 저자들은 크게 두 가지 설계 축을 제안한다.

첫째는 **Intention-Oriented Multimodal Fusion**이다. 외과의의 음성 명령만으로는 보통 instrument 이름 정도의 고수준 정보만 담기기 쉽다. 예를 들어 “bipolar forceps 보여줘” 같은 명령은 해당 도구의 이름은 알려 주지만, 시각적으로 어떤 모양과 특징을 가지는지까지는 설명하지 않는다. 그래서 논문은 음성에서 인식한 의도를 기반으로, 각 기구에 대한 **사전 준비된 textual description**을 함께 활용한다. 다시 말해, audio만 쓰는 것이 아니라 **audio + text + image**를 결합해 더 풍부한 instrument representation을 만든다.

둘째는 **Contrastive Learning Prompt Encoder**이다. 의도에 맞는 instrument를 잘 찾으려면, 단지 target feature를 잘 만드는 것만으로는 부족하고, 동시에 **irrelevant instrument와 명확히 구분**되어야 한다. 논문은 required instrument feature와 irrelevant instrument feature를 나누고, 둘 사이를 contrastive하게 벌려 주는 방식으로 prompt를 학습한다. 이 설계는 “원하는 것에만 집중하고, 헷갈리는 다른 기구들은 배제하는” 효과를 노린다.

기존 접근과의 차별점은 다음과 같이 정리할 수 있다. 기존 일반 segmentation 모델은 전체 instrument를 다 찾는 데 초점이 있었고, SAM 기반 의료 영상 방법은 수동 prompt에 크게 의존했다. 반면 ASI-Seg는 **수술자의 intention을 직접 입력 신호로 사용**하고, **manual prompt 없이도 target-specific segmentation**을 수행한다. 또한 text description bank를 통해 category name 이상의 세부 지식을 활용한다는 점도 중요한 차별점이다.

## 3. 상세 방법 설명

![그림 1: ASI-Seg의 전체 구조. ASI-Seg는 주로 의도 기반 다중모달 융합과 대조학습 프롬프트 인코더로 구성된다. 외과의의 분할 의도를 해석한 뒤, 다중모달 정보를 활용하여 의도 지향적 특징을 생성하고, 필요한 기구 특징과 불필요한 특징 간의 대조학습을 수행하여 원하는 기구를 분할하기 위한 프롬프트를 생성한다.](https://ar5iv.labs.arxiv.org/html/2407.19435/assets/x1.png)

### 전체 파이프라인

ASI-Seg의 입력은 수술 영상의 한 이미지와 외과의의 음성 명령이다. 시스템은 먼저 음성에서 “어떤 instrument를 분할하려는가”라는 intention을 예측한다. 그 다음, 각 instrument에 대한 textual description과 이미지 feature를 융합해 instrument-aware multimodal feature를 만든다. 이후 audio intention 결과를 이용해 이들 feature를 **required feature**와 **irrelevant feature**로 나눈다. 마지막으로 contrastive learning 기반 prompt encoder가 required instrument를 더 잘 구분하도록 prompt를 정제하고, 이를 SAM의 mask decoder에 넣어 원하는 instrument mask를 생성한다.

즉, 흐름은 다음과 같이 이해할 수 있다.

1. 음성 명령 처리
2. 음성으로부터 instrument intention 분류
3. text description과 learnable query 융합
4. image feature와의 cross-modal 결합
5. target / non-target feature 분리
6. contrastive prompt 생성
7. SAM mask decoder로 최종 segmentation

이 구조는 단순히 audio classifier를 붙인 segmentation 모델이 아니라, **의도 인식과 prompt 생성이 구조적으로 연결된 multimodal framework**라는 점이 핵심이다.

---

### 3.1 Audio Intention Recognition

외과의의 raw audio signal $a$에서 16kHz로 샘플링한 신호 $a'$를 사용하고, 이를 Mel spectrogram으로 변환한다.

$$
A_{mel} = \pi(a, a', C_s, W_s, s)
$$

여기서 $\pi$는 Mel spectrogram transformation이고, $C_s$는 channel size, $W_s$는 window size, $s$는 stride size이다.

이후 수치적 안정성을 위해 Mel spectrogram을 정규화한다.

$$
A_{norm} = 2 \times \frac{(A_{mel} - \mu)}{\max(A_{mel}) - \min(A_{mel})} - 1
$$

여기서 $\mu$는 training data에서의 Mel spectrogram 평균이다. 이 식의 목적은 입력 스케일을 $[-1, 1]$ 범위로 맞춰서 audio encoder가 보다 안정적으로 학습되도록 하는 것이다.

정규화된 spectrogram $A_{norm}$은 audio encoder $E_A$와 classifier $\phi$로 들어가서 최종적으로 surgeon intention class $\mathcal{C}$를 예측한다.

$$
\mathcal{C} = \phi(E_A(A_{norm}))
$$

즉, $\mathcal{C}$는 현재 외과의가 요구하는 instrument category라고 볼 수 있다. 이 분류 결과가 이후 전체 segmentation 파이프라인에서 “무엇을 foreground로 보고 무엇을 irrelevant로 볼지”를 결정하는 기준이 된다.

---

### 3.2 Text Fusion

논문은 audio command만으로는 부족하다고 본다. 예를 들어 instrument 이름만으로는 유사한 도구들 사이의 세밀한 시각적 차이를 충분히 반영하기 어렵기 때문이다. 그래서 각 instrument마다 사전에 준비한 설명문을 모아 둔 **Instrument Description Bank** ${B_k}_{k=1}^K$를 사용한다.

이 description bank는 text encoder $E_T$를 통해 feature로 변환된다.

$$
f_t = concat(E_T({B_k}_{k=1}^K))
$$

여기서 $f_t \in \mathbb{R}^{K \times d}$는 $K$개 instrument 각각에 대한 text feature를 의미한다.

이후 모델은 instrument category마다 하나씩 대응되는 $K$개의 learnable query $f_c$를 두고, 이를 text feature와 **mutual cross-attention**으로 결합한다. 논문은 다음과 같이 표현한다.

$$
q_t = softmax\left(\frac{Q_t K_c^T}{D}\right)V_c
$$

$$
q_c = softmax\left(\frac{Q_c K_t^T}{D}\right)V_t
$$

$$
q = MLP(concat(q_t, q_c))
$$

여기서 $Q_t, K_t, V_t$는 textual feature에서, $Q_c, K_c, V_c$는 learnable query에서 온 attention 요소들이다. 결과적으로 만들어지는 $q$는 **text-informed instrument query**이다.

쉽게 말하면, learnable query가 단순한 파라미터 벡터에 머무는 것이 아니라, 각 instrument의 textual description을 흡수하여 더 의미 있는 category representation으로 바뀌는 구조이다. 이 부분은 ablation 결과에서도 성능 향상에 큰 기여를 보인다.

---

### 3.3 Visual Fusion

이제 위에서 만든 instrument query $q$를 사용해 이미지에서 해당 instrument와 관련된 시각 정보를 끌어온다. 입력 이미지 $I \in \mathbb{R}^{H \times W \times 3}$는 image encoder $E_I$를 거쳐 image feature $f_i \in \mathbb{R}^{h \times w \times d}$가 된다.

$$
f_i = E_I(I)
$$

그 다음 instrument query $q_k$와 image feature $f_i$ 사이의 similarity를 계산하여 각 instrument별 Similarity Matrix $S_k$를 만든다. 논문 표기는 다음과 같다.

$$
{S_k \mid S_k = q_k \cdot f_i}_{k=1}^K
$$

이 similarity는 이미지 내 어떤 위치가 특정 instrument query와 잘 맞는지를 나타내는 값으로 이해할 수 있다. 이후 image feature와 similarity를 결합해 multimodal feature를 만든다.

$$
{f_{i-t}^k}_{k=1}^K = {f_i \cdot S_k + f_i}_{k=1}^K
$$

즉, 원래의 image feature에 instrument-query 기반 가중을 더해 **text-aware visual feature**를 생성하는 것이다. 이 결과는 각 instrument에 대해 하나씩 생성되며, 이미지 안에서 그 instrument와 관련성이 높은 영역을 더 강조한다.

---

### 3.4 Feature Assignment with Audio Intention

앞 단계까지 오면 instrument별 multimodal feature 집합 $F$가 있다. 여기에서 audio intention 결과 $\mathcal{C}$를 이용해 target과 non-target을 나눈다.

$$
F^+ = {f_{i-t}^{\mathcal{C}}}
$$

$$
F^- = {f_{i-t}^k, k \neq \mathcal{C}}_{k=1}^K
$$

여기서 $F^+$는 현재 외과의가 요구한 instrument의 feature이고, $F^-$는 나머지 모든 irrelevant instrument feature이다.

이 분리는 ASI-Seg의 매우 중요한 설계다. 일반적인 segmentation에서는 모든 class를 동시에 예측하거나, 여러 object를 함께 다루는 경우가 많다. 그러나 이 논문은 아예 구조적으로 target과 distractor를 분리해서 다룬다. 이는 surgeon intention을 직접 반영하는 방법이며, 나중의 contrastive prompt learning과도 자연스럽게 연결된다.

---

### 3.5 Contrastive Learning Prompt Encoder

#### Distinguishing Cross-Attention

요구된 instrument와 irrelevant instrument를 더 잘 구분하기 위해 논문은 mutual cross-focusing 형태의 attention을 사용한다. 먼저 required feature $F^+$와 irrelevant feature $F^-$ 사이의 attention을 계산한다.

$$
Attention(F^+, F^-) = softmax\left(\frac{Q_{F^+} K_{F^-}^T}{D}\right)V_{F^-}
$$

이 attention은 required feature가 irrelevant feature와 공유하거나 혼동될 수 있는 부분을 드러낸다고 볼 수 있다. 논문은 이를 “easily confounded regions”를 찾는 과정으로 설명한다.

그 다음 inverse residual 메커니즘을 적용한다.

$$
P^* = P - Attention(F^+, F^-)
$$

이 식의 의미는 required feature $P$에서 irrelevant feature와 비슷한 성분을 빼 버린다는 것이다. 결과적으로 $P^*$는 target instrument의 고유한 특성만 더 많이 남게 된다. 이 아이디어는 매우 직관적이다. segmentation 오류의 상당수는 배경 때문이 아니라 **유사한 다른 instrument와의 혼동**에서 나오므로, 그 혼동 성분을 제거하면 target-specific prompt가 더 좋아질 수 있다.

#### Contrastive Learning

논문은 여기서 한 단계 더 나아가 contrastive loss를 도입한다. required instrument feature를 positive로, irrelevant instrument feature를 negative로 두고, ground truth mask로 필터링된 image embedding feature $v$와의 관계를 학습한다.

$$
\mathcal{L}_{CL} =
-\frac{1}{K} \sum_{n=1}^{K}
\log
\frac{\exp(P(\mathcal{C}) \cdot v(\mathcal{C}) / \tau)}
{\sum_{n=1}^{K}\exp(P(\mathcal{C}) \cdot v(n) / \tau)}
$$

여기서 $\tau$는 temperature factor이다. 이 loss는 target class의 feature가 같은 class의 visual evidence와는 가까워지고, 다른 class들과는 멀어지게 만든다.

쉽게 설명하면, required instrument prompt가 “내가 찾는 기구답다”는 방향으로 수렴하도록 하고, 동시에 다른 instrument들과는 구별되도록 강제하는 손실 함수다. 논문은 이 contrastive loss가 learnable query를 동적으로 더 잘 업데이트하게 만든다고 설명한다.

---

### 3.6 Mask Decoder와 최종 학습 목표

최종 segmentation은 SAM의 mask decoder를 사용하여 수행된다. 논문은 required instrument feature를 **foreground prompt**, irrelevant instrument feature를 **background prompt**처럼 사용한다고 설명한다. 즉, 단지 target을 잘 설명하는 prompt만 넣는 것이 아니라, “무엇이 target이 아닌가”도 함께 mask decoder에 전달한다는 점이 특징이다.

전체 loss는 다음과 같다.

$$
\mathcal{L} = \mathcal{L}_{DICE} + \mathcal{L}_{CL}
$$

여기서 $\mathcal{L}_{DICE}$는 segmentation용 dice loss이고, $\mathcal{L}_{CL}$은 앞서 설명한 contrastive learning loss이다.

또한 학습 효율을 위해 큰 encoder들은 freeze한다. 즉, image encoder, audio encoder, text encoder는 고정하고, 상대적으로 가벼운 instrument classifier, mask decoder, intention-oriented multimodal fusion module, contrastive prompt encoder만 학습한다. 이는 SAM 계열의 대규모 backbone을 실용적으로 활용하기 위한 전략으로 이해할 수 있다.

## 4. 실험 및 결과

### 데이터셋과 평가 설정

논문은 **EndoVis2018**과 **EndoVis2017** 두 데이터셋에서 평가를 수행한다. EndoVis2017은 8개 비디오로 구성되며, 표준 프로토콜에 따라 4-fold cross validation을 사용한다. 영상 해상도는 $1280 \times 1024$이고, da Vinci Xi surgical system에서 획득되었다. EndoVis2018은 11개 training video와 4개 validation video로 구성되어 있다. 두 데이터셋 모두 7개의 고유한 surgical instrument category를 포함한다.

평가지표는 **Challenge IoU**, **IoU**, 그리고 **mean class IoU (mc IoU)** 이다. semantic segmentation에서는 주로 IoU와 Challenge IoU를 보고, intention-oriented segmentation에서는 category별 IoU를 평균한 mc IoU를 핵심 지표로 사용한다.

### 구현 세부사항

모델은 PyTorch로 구현되었고, 단일 NVIDIA A800 GPU에서 학습되었다. image encoder는 pre-trained ViT, text encoder는 CLIP의 text encoder, audio encoder는 pre-trained audio encoder를 사용한다. contrastive loss의 temperature는 $\tau = 0.07$이고, optimizer는 Adam, learning rate는 $0.0001$이다. batch size는 EndoVis2017에서 16, EndoVis2018에서 64로 설정되었다.

이 부분에서 주목할 점은 backbone 대부분을 freeze하고 pre-computed image embedding을 활용했다는 점이다. 따라서 제안법의 성능 향상이 단순히 더 큰 모델을 통째로 fine-tuning해서 나온 결과라고 보기는 어렵고, prompt 설계와 multimodal fusion의 효과가 반영되었다고 해석할 수 있다.

---

### 4.1 Semantic Segmentation 결과

EndoVis2018에서 ASI-Seg는 **IoU 82.37%**를 기록하여 가장 높은 성능을 보였다. 비교 대상 중 가장 강한 baseline인 SurgicalSAM은 **80.33%**였으므로, ASI-Seg는 **2.04%p** 앞선다. EndoVis2017에서는 ASI-Seg가 **71.64%**, SurgicalSAM이 **69.94%**로, **1.70%p** 개선을 보였다.

이는 흥미로운 결과다. ASI-Seg는 본래 intention-oriented segmentation을 목표로 설계된 방법인데, 모든 instrument를 분할하는 일반 semantic segmentation에서도 최고 성능을 보였다. 논문은 이를 required instrument와 irrelevant instrument를 더 잘 구분하도록 만든 구조 덕분이라고 해석한다. 즉, target-aware 설계가 오히려 일반적인 segmentation 표현력까지 끌어올렸다는 주장이다.

다만 표를 보면 EndoVis2017에서 S3Net의 IoU는 **71.99%**로 제시되어 있고, ASI-Seg는 **71.64%**이다. 그런데 본문은 ASI-Seg가 semantic segmentation에서 전반적으로 최고라고 서술한다. 따라서 EndoVis2017의 경우에는 표와 본문 사이에 약간의 긴장이 있다. 본문에서는 SurgicalSAM과의 비교를 강조하지만, 표만 기준으로 보면 S3Net과의 상대적 우열은 다시 확인이 필요하다. 이 부분은 논문 텍스트만으로는 명확히 정리되지 않는다.

---

### 4.2 Intention-Oriented Segmentation 결과

이 논문의 진짜 핵심 실험은 intention-oriented segmentation이다. 즉, 영상 속 모든 instrument가 아니라, 지정된 category만 목표로 했을 때 얼마나 잘 분할하는지를 평가한다.

EndoVis2018에서 ASI-Seg는 **mc IoU 64.18%**를 기록했다. SurgicalSAM은 **58.87%**이므로, **5.31%p** 개선이다. 이는 논문이 가장 강조하는 수치 중 하나다. category별 결과를 보면 ASI-Seg는 특히 **SI 90.43**, **UP 55.62**처럼 몇몇 class에서 큰 강점을 보인다. 반면 **CA 34.90**, **MCS 60.10** 등 일부 category에서는 SurgicalSAM보다 낮은 값도 있다. 즉, 모든 class에서 일관되게 이긴 것은 아니지만, 평균적으로는 분명한 우세를 보인다.

EndoVis2017에서는 표상 ASI-Seg가 **mc IoU 68.37%**를 기록했고, SurgicalSAM은 **67.03%**이다. 본문은 **68.17%**라고 적고 있어 수치 표기상 작은 불일치가 존재한다. 이 경우 표와 본문 중 어느 쪽이 최종값인지 텍스트만으로는 확정할 수 없다. 다만 어느 수치를 따르더라도 ASI-Seg가 가장 높은 mc IoU를 달성했다는 결론 자체는 유지된다.

category별로 보면 EndoVis2017에서 ASI-Seg는 **BF 73.92**, **LND 80.33**, **VS 75.44**, **MCS 89.78**에서 강력한 성능을 보인다. 그러나 **PF 47.61**, **GR 52.60**, **UP 58.90**에서는 SurgicalSAM보다 다소 낮다. 따라서 이 방법의 장점은 “모든 class를 압도한다”기보다, **전반적인 평균 성능을 높이면서 여러 class에서 더 균형 잡힌 target-aware segmentation을 달성한다**는 데 있다고 보는 편이 정확하다.

---

### 4.3 정성적 비교와 강건성

![그림 2: EndoVis2017 데이터셋에서 의도 지향 분할 결과의 정성적 비교.](https://ar5iv.labs.arxiv.org/html/2407.19435/assets/x2.png)

논문은 Fig. 2를 통해 정성적 비교도 제시한다. 원문 설명에 따르면, ASI-Seg는 manual annotation이나 수동 category assignment 없이도 segmentation intention을 올바르게 이해하고 가장 정확한 mask를 생성했다고 한다. 다만 제공된 텍스트에는 실제 figure 이미지가 없으므로, 시각적 결과의 세부 형태까지는 확인할 수 없다. 따라서 정성적 우수성에 대한 평가는 저자 설명 수준에서만 이해해야 한다.

Robustness study에서는 mispronunciation에 대한 내성을 조사한다. 예를 들어 surgeon이 “Bipolar Forceps”를 “Bipolyr Frocips”처럼 잘못 발음해도 ASI-Seg가 올바른 instrument category를 인식하고 정확한 segmentation을 수행한다고 보고한다. 이는 audio-driven interface의 실제 사용성 측면에서 중요한 장점이다. 텍스트 명령 기반 시스템은 오타에 취약할 수 있지만, 이 방법은 음성 인식과 intention understanding을 함께 다루므로 일정 수준의 발음 오류를 견딜 수 있다고 주장한다.

다만 이 robustness 평가는 정량 표보다 주로 정성적 설명과 figure 중심으로 제시되어 있다. mispronunciation 조건에서의 수치적 성능 저하 정도, 오류율, confusion matrix 등은 본문 텍스트에 없다. 따라서 “강건하다”는 결론은 가능하지만, 그 강건성의 정확한 범위를 수치로 판단하기는 어렵다.

---

### 4.4 Ablation Study

EndoVis2018에서 수행한 ablation 결과는 제안 모듈의 기여를 비교적 명확히 보여 준다.

기본 baseline은 **IoU 76.14, mc IoU 51.00**이다. 여기에 **Instrument Description Bank**를 추가하면 **IoU 80.17, mc IoU 59.42**가 되어, mc IoU 기준 **8.42%p** 향상된다. 이는 textual knowledge 주입이 category 구분에 매우 큰 도움을 준다는 강한 증거다.

반대로 contrastive learning만 추가한 경우는 **IoU 78.63, mc IoU 55.98**로, baseline 대비 개선되지만 description bank보다는 상승폭이 작다. 마지막으로 둘 다 사용하면 **IoU 82.37, mc IoU 64.18**로 최고 성능을 달성한다.

이 결과는 제안된 두 모듈이 서로 보완적임을 보여 준다. description bank는 class semantics를 풍부하게 해 주고, contrastive learning은 target과 non-target을 더 선명하게 분리한다. 즉, 하나는 “무엇인지 더 잘 알게 하는” 역할이고, 다른 하나는 “무엇이 아닌지 더 잘 구분하게 하는” 역할이라고 이해할 수 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 설정 자체가 실용적**이라는 점이다. 기존 surgical instrument segmentation은 대부분 전체 장면 이해에 초점을 맞췄지만, 실제 임상 workflow에서는 특정 시점에 특정 instrument만 정확히 강조하는 기능이 중요할 수 있다. ASI-Seg는 이 점을 surgeon intention이라는 관점에서 정면으로 다루며, audio-driven interface를 통해 수술실 환경에 더 가까운 사용 시나리오를 제시한다.

두 번째 강점은 **multimodal 설계의 논리성이 높다**는 점이다. audio만으로는 정보가 거칠고, image만으로는 intention을 알 수 없으며, text description은 category semantics를 보완한다. 논문은 이 세 모달리티를 단순 병렬 결합이 아니라, intention prediction, query enrichment, prompt refinement라는 구조적 흐름으로 엮었다. 특히 description bank와 contrastive learning의 조합은 ablation 결과로도 효과가 뒷받침된다.

세 번째 강점은 **SAM의 실질적 활용 방식**이다. 단순히 SAM을 fine-tune하는 것이 아니라, prompt generation 문제를 새롭게 설계해서 수술 환경에 맞게 적용했다. manual prompt를 요구하지 않는다는 점은 임상 도입 가능성 측면에서 중요한 장점이다.

그러나 한계도 분명하다. 첫째, 이 방법은 audio intention recognition이 전체 파이프라인의 시작점이므로, 음성 분류가 잘못되면 이후 segmentation도 연쇄적으로 틀릴 가능성이 크다. 논문은 mispronunciation robustness를 보여 주지만, 실제 수술실의 소음, 겹치는 화자, 다국어 발화, 문장형 명령 등 복잡한 음성 환경에서 얼마나 안정적인지는 텍스트만으로 알 수 없다.

둘째, **Instrument Description Bank**가 필요하다는 점은 실용성과 일반화 측면에서 약간의 제약이 될 수 있다. 각 instrument에 대해 사전에 textual description을 준비해야 하므로, 새로운 도구가 등장하거나 도메인이 바뀌면 description bank를 다시 구성해야 할 가능성이 있다. 또한 description의 품질이 성능에 영향을 미칠 수 있는데, 논문은 description 작성 방식이나 문장 설계 원칙을 자세히 설명하지 않는다.

셋째, intention-oriented segmentation의 평가 설정이 실제 사용 시나리오를 얼마나 충실히 반영하는지는 추가 검토가 필요하다. 논문은 category intention을 기준으로 평가하지만, 실제 음성 명령은 더 복잡한 문맥을 포함할 수 있다. 예를 들어 “왼쪽에 있는 forceps”나 “조금 전에 집었던 기구”처럼 공간적·시간적 지시가 들어가면 현재 구조만으로는 충분하지 않을 수 있다. 논문 텍스트에는 이러한 더 복잡한 intention handling은 명시되어 있지 않다.

넷째, 표와 본문 사이에 일부 수치 불일치가 보인다. EndoVis2017 semantic segmentation에서는 S3Net과의 관계가 본문 설명과 완전히 일치하지 않고, intention-oriented segmentation의 mc IoU도 표와 서술 값이 다르게 나타난다. 결과의 전체 방향성은 유지되지만, 보고서 관점에서는 이런 수치 일관성 문제를 지적할 필요가 있다.

비판적으로 보면, 이 논문은 **“intention-aware surgical segmentation”이라는 중요한 문제를 설득력 있게 제기하고, 꽤 효과적인 첫 해법을 제시했다**는 점에서 가치가 크다. 다만 현재 방법은 instrument category selection 중심의 intention understanding에 가깝고, 보다 풍부한 언어적 지시나 복잡한 수술실 환경을 다루는 단계까지는 아직 가지 않았다. 따라서 후속 연구에서는 더 자연스러운 spoken instruction understanding, temporal context 활용, open-vocabulary instrument extension 등이 중요해 보인다.

## 6. 결론

이 논문은 수술 장면에서 외과의의 음성 명령을 이해해 필요한 surgical instrument만 분할하는 **ASI-Seg**를 제안했다. 핵심 기여는 크게 세 가지로 요약할 수 있다. 첫째, surgeon intention을 segmentation 문제에 직접 연결하는 **audio-driven intention-oriented segmentation** 프레임워크를 제시했다. 둘째, text description bank와 image feature를 결합하는 **intention-oriented multimodal fusion**을 통해 target instrument representation을 강화했다. 셋째, required instrument와 irrelevant instrument를 분리하는 **contrastive learning prompt encoder**를 통해 SAM mask decoder에 더 적합한 prompt를 생성했다.

실험적으로도 ASI-Seg는 EndoVis2018과 EndoVis2017에서 semantic segmentation과 intention-oriented segmentation 모두 강한 성능을 보였고, 특히 intention-oriented setting에서 기존 SAM 기반 방법보다 더 큰 이점을 보였다. 또한 mispronunciation에 대한 robustness와 ablation 결과를 통해 설계 요소의 타당성도 어느 정도 입증했다.

실제 적용 측면에서 이 연구는 수술실에서 **필요한 정보만 선택적으로 강조해 외과의의 인지 부담을 줄이는 보조 시스템**으로 발전할 가능성이 있다. 향후에는 더 자연스러운 음성 명령, 다양한 수술 환경, unseen instrument, temporal reasoning까지 확장된다면, 단순 segmentation을 넘어 실제 surgical assistant system의 핵심 구성 요소가 될 수 있다.
