# Self-Promoted Supervision for Few-Shot Transformer

- **저자**: Bowen Dong, Pan Zhou, Shuicheng Yan, Wangmeng Zuo
- **발표연도**: 추출 텍스트에 명시되지 않음
- **arXiv**: 추출 텍스트에 명시되지 않음

## 1. 논문 개요

이 논문은 Vision Transformer(ViT)를 few-shot classification에 그대로 적용하면 CNN 기반 특징 추출기보다 성능이 크게 떨어진다는 문제를 다룬다. 저자들은 특히 Meta-Baseline 같은 동일한 few-shot learning framework 안에서 backbone만 CNN에서 ViT로 바꾸었을 때 성능 저하가 심하다는 점을 실험적으로 확인한다. 논문의 핵심 질문은 두 가지다. 첫째, ViT가 few-shot learning 환경에서도 잘 작동하는가. 둘째, 그렇지 않다면 왜 그런가, 그리고 어떻게 개선할 수 있는가.

이 문제는 실용적으로 중요하다. few-shot learning은 라벨이 극히 적은 환경에서 새로운 클래스를 인식해야 하는 문제이므로, 의료 데이터처럼 수집이 어렵거나 annotation 비용이 큰 분야에서 특히 중요하다. ViT는 최근 다양한 비전 과제에서 강력한 성능을 보였지만, 일반적으로 많은 데이터가 있어야 잘 학습된다는 약점이 있다. 따라서 제한된 데이터만 주어지는 few-shot regime에서 ViT의 약점을 보완할 수 있다면, ViT의 표현력과 transformer 구조의 장점을 더 넓은 실제 문제에 적용할 수 있다.

저자들은 성능 저하의 주요 원인으로 ViT의 inductive bias 부족을 지목한다. CNN은 지역성(locality)과 평행이동 등 구조적 bias가 강해 적은 데이터에서도 비교적 안정적으로 일반화하지만, ViT는 patch token들 사이의 dependency를 충분히 배우려면 더 많은 데이터가 필요하다. few-shot에서는 그 과정이 불안정해지고, 결과적으로 novel class에 대한 generalization이 약해진다는 것이 논문의 출발점이다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 ViT가 few-shot 환경에서 잘 안 되는 이유를 “token dependency learning의 질과 속도” 문제로 해석하고, 이를 직접 보조하는 dense local supervision을 제공하자는 것이다. 이를 위해 저자들은 **SUN (Self-promoted sUpervisioN)** 이라는 학습 프레임워크를 제안한다.

SUN의 직관은 단순하다. 이미지 전체에 대한 global label만 주면, ViT는 각 patch token이 무엇을 보고 있는지 충분히 빨리 배우기 어렵다. 그래서 이미지 전체 분류 supervision 외에, 각 patch token마다 location-specific supervision을 추가로 준다. 이 supervision은 외부 annotation이 아니라, 먼저 같은 few-shot base dataset에서 학습한 teacher ViT가 생성한 pseudo label이다. 즉 같은 구조의 ViT가 자기 자신을 돕는 방식이므로 “self-promoted”라고 부른다.

저자들은 이 local supervision이 두 가지 효과를 낸다고 주장한다. 첫째, 비슷한 patch token에는 비슷한 pseudo label이 부여되므로, 어떤 token들이 유사하거나 다른지를 ViT가 더 빨리 학습할 수 있다. 이는 self-attention이 올바른 dependency를 더 빨리 형성하도록 돕는다. 둘째, 각 patch 수준의 의미를 더 잘 모델링함으로써 object grounding과 recognition 능력이 좋아지고, 결과적으로 novel class에 대한 generalization이 개선된다.

기존 접근과의 차별점은, 단순히 CNN-like inductive bias를 ViT에 주입하는 것이 아니라 ViT 내부 token 학습 자체를 직접 밀어주는 supervision을 설계했다는 점이다. 논문은 CNN branch 추가, CNN distillation, local attention 등 여러 방식이 성능 향상에 도움을 주는 것을 보여주지만, SUN은 patch-level pseudo supervision을 통해 더 직접적이고 효과적으로 dependency learning을 촉진한다고 주장한다.

## 3. 상세 방법 설명

SUN은 전체적으로 두 단계로 구성된다. 첫 단계는 **meta-training**, 두 번째는 **meta-tuning**이다. 구조적으로는 Meta-Baseline과 유사한 틀을 가지지만, meta-training에서 location-specific supervision을 추가한다는 점이 핵심 차이이다.

### 3.1 Meta-Training 단계

meta-training의 목표는 base dataset $D_{base}$ 위에서 meta-learner $f$를 학습하는 것이다. 여기서 $f$는 ViT feature extractor이다. SUN은 이 단계에서 두 종류의 supervision을 사용한다.

첫 번째는 일반적인 global supervision이다. 이미지 전체의 ground-truth class label을 이용해 전체 이미지의 의미를 학습한다. 두 번째가 논문의 핵심인 local supervision인데, 각 patch token에 대해 patch-level pseudo label을 준다.

이를 위해 먼저 teacher 모델 $f_g$를 학습한다. 이 teacher는 같은 구조의 ViT backbone $f_0$와 classifier $g_0$로 이루어진다. teacher는 base dataset에서 supervised classification pretraining으로 학습된다. 이후 teacher가 각 patch token에 대해 class confidence를 예측하고, 이것을 pseudo label로 사용한다.

논문은 이미지 $x_i$의 token 출력을 다음처럼 둔다.

$$
z = f(x_i) = [z_{cls}, z_1, z_2, \cdots, z_K]
$$

여기서 $z_{cls}$는 class token이고, $z_1, \dots, z_K$는 patch token이다. teacher의 patch-level confidence는 다음과 같이 계산된다.

$$
\hat{s}_i = [\hat{s}_{i1}, \hat{s}_{i2}, \cdots, \hat{s}_{iK}] = f_g(x_i) = [g(z_1), g(z_2), \cdots, g(z_K)] \in \mathbb{R}^{c \times K}
$$

즉 각 patch $x_{ij}$에 대해 $\hat{s}_{ij}$는 그 patch가 어떤 클래스 의미를 담고 있는지에 대한 pseudo label 역할을 한다.

### 3.2 Background Patch Filtration (BGF)

teacher는 base class만 보고 학습했기 때문에, 실제로는 배경인 patch도 어떤 semantic class로 억지로 분류할 수 있다. 이는 local supervision의 품질을 떨어뜨린다. 이를 해결하기 위해 저자들은 **BGF (Background Patch Filtration)** 를 도입한다.

방법은 간단하다. patch별 confidence score 중 최대값을 뽑고, 배치 내 모든 patch를 confidence 기준으로 오름차순 정렬한 뒤, 가장 낮은 상위 $p\%$ patch를 background로 간주한다. 그리고 원래 $c$개 class 외에 추가적인 background class 하나를 만든다. background patch에는 마지막 차원만 1인 one-hot 성격의 label을 부여하고, foreground patch에는 기존 semantic score를 유지하되 background 차원은 0으로 둔다.

이렇게 하면 배경을 억지로 의미 클래스에 할당하는 오류를 줄일 수 있다. few-shot에서는 데이터가 적기 때문에 이런 노이즈가 특히 치명적일 수 있는데, BGF는 local pseudo supervision의 품질을 보정하는 장치로 작동한다.

### 3.3 전체 학습 손실

SUN의 meta-training loss는 global classification loss와 local classification loss의 합으로 구성된다.

$$
L_{SUN} = H(g_{global}(z_{global}), y_i) + \lambda \sum_{j=1}^{K} H(g_{local}(z_j), s_{ij})
$$

여기서 $z_{global}$은 모든 patch token의 global average pooling 결과이고, $H$는 cross-entropy loss이다. $g_{global}$은 이미지 전체 분류기, $g_{local}$은 patch 분류기이다. $s_{ij}$는 teacher가 생성한 location-specific supervision이다. 논문에서는 $\lambda = 0.5$로 고정한다.

이 식의 의미는 명확하다. 첫 항은 이미지 전체가 어떤 클래스인지 맞추게 하고, 둘째 항은 각 patch가 어떤 의미를 갖는지 맞추게 한다. 이 local term이 token dependency learning을 촉진하는 핵심 장치다.

### 3.4 Spatial-Consistent Augmentation (SCA)

local pseudo label을 teacher가 생성할 때 augmentation이 너무 강하면 patch correspondence가 깨질 수 있다. 반대로 augmentation이 약하면 일반화가 약해질 수 있다. 이를 동시에 해결하기 위해 **SCA (Spatial-Consistent Augmentation)** 를 제안한다.

SCA는 두 단계 augmentation으로 구성된다.

첫째, **spatial-only augmentation**은 random crop and resize, horizontal flip, rotation처럼 공간 구조를 바꾸는 변환만 적용해서 $\tilde{x}_i$를 만든다.  
둘째, **non-spatial augmentation**은 color jitter, blur, solarization, grayscale, random erasing 같은 비공간 변환을 추가해 $\bar{x}_i$를 만든다.

teacher는 상대적으로 약한 형태의 $\tilde{x}_i$를 입력받아 patch supervision $s_{ij}$를 생성하고, 실제 meta-learner는 더 다양성이 큰 $\bar{x}_i$를 입력받아 학습한다. 이렇게 하면 local supervision의 정확성은 유지하면서도 학습 데이터 다양성은 확보할 수 있다.

### 3.5 Meta-Tuning 단계

meta-tuning은 기존 few-shot learning 방법을 따라간다. 논문 본문에서는 Meta-Baseline 방식이 기본으로 설명된다. 새로운 task $\tau$의 support set $S$가 주어지면, 각 class $k$의 prototype을 다음과 같이 계산한다.

$$
w_k = \frac{\sum_{x \in S_k} GAP(f(x))}{|S_k|}
$$

여기서 $S_k$는 class $k$의 support sample 집합이다. query image $x$에 대해 class $k$의 confidence는 cosine similarity 기반으로 계산된다.

$$
p_k = \frac{\exp(\gamma \cdot \cos(GAP(f(x)), w_k))}
{\sum_{k'} \exp(\gamma \cdot \cos(GAP(f(x)), w_{k'}))}
$$

여기서 $\gamma$는 temperature parameter이다. 이후 query에 대한 cross-entropy loss $L_{few-shot} = H(p_x, y_x)$를 최소화하도록 fine-tuning한다. 테스트 시에도 같은 방식으로 prototype을 만들고 query를 분류한다.

즉 SUN의 본질은 meta-tuning 자체를 새로 만든 것이 아니라, meta-training에서 ViT representation을 few-shot 친화적으로 바꿔 놓는 데 있다. 논문은 Appendix에서 FEAT, DeepEMD 같은 다른 meta-tuning에도 SUN을 결합할 수 있다고 보여준다.

## 4. 실험 및 결과

실험은 few-shot classification의 대표 벤치마크인 miniImageNet, tieredImageNet, CIFAR-FS에서 수행되었다. 평가는 5-way 1-shot과 5-way 5-shot 설정으로 진행되며, 각 벤치마크마다 2,000개 task에 대한 평균 정확도와 95% confidence interval을 보고한다.

### 4.1 ViT는 왜 few-shot에서 약한가

Table 1은 Meta-Baseline 프레임워크에서 backbone만 바꿨을 때의 결과를 보여준다. miniImageNet 기준으로 ResNet-12는 meta-training 후 5-way 1-shot에서 60.00%, meta-tuning 후 64.53%를 기록하지만, ViT 계열은 크게 낮다. 예를 들어 NesT는 meta-training 후 49.23%, meta-tuning 후 54.57%이다. LeViT와 LV-ViT는 더 낮다.

흥미로운 점은 ViT가 base training/validation set에서는 잘 수렴하지만, novel class test set에서는 일반화가 약하다는 것이다. Fig. 2에 따르면 ViT는 학습 데이터와 base validation에서는 높은 성능을 보이지만, novel category의 test accuracy는 낮고 때로는 훈련이 진행되며 떨어지기까지 한다. 예를 들어 NesT는 초반 약 52%까지 갔다가 30 epoch 부근 이후 49.2% 수준으로 떨어진다. 반면 ResNet-12는 마지막 100 epoch 동안 약 60% 수준을 유지한다. 저자들은 이를 “base class에는 맞지만 novel class generalization이 약한 현상”으로 해석한다.

### 4.2 Inductive Bias 관련 분석

Table 2는 inductive bias가 few-shot ViT에 도움이 되는지 보여준다. CNN branch를 추가한 경우, CNN distillation을 적용한 경우, local attention이 있는 Swin/NesT와 없는 LeViT를 비교한 경우 모두 성능 향상이 나타난다. 예를 들어 miniImageNet 1-shot에서 vanilla NesT 기반 Meta-Baseline은 54.57%인데, CNN distill을 추가하면 55.79%, CNN branch를 결합하면 57.91%가 된다.

attention map 시각화에서도 CNN-distilled ViT가 vanilla ViT보다 semantic patch를 더 잘 포착한다. 논문은 이를 근거로 “few-shot에서 중요한 것은 단순한 빠른 수렴이 아니라 높은 품질의 token dependency 학습”이라고 주장한다. 실제로 CNN-distilled ViT는 training accuracy는 더 느리게 오르지만, attention quality는 더 좋게 나타난다.

### 4.3 SUN의 다양한 ViT backbone에 대한 효과

Table 3은 miniImageNet에서 SUN을 LV-ViT, Swin, Visformer, NesT에 적용한 결과를 보여준다. 모든 backbone에서 Meta-Baseline 대비 큰 폭의 향상이 있다.

LV-ViT는 43.08%에서 59.00%로,  
Swin은 54.63%에서 64.94%로,  
Visformer는 47.61%에서 67.80%로,  
NesT는 54.57%에서 66.54%로 향상된다.

특히 1-shot에서 개선폭이 매우 크다. 이는 데이터가 더 적은 설정일수록 local supervision의 효과가 더 강하게 나타남을 시사한다. 저자들은 Visformer가 CNN module을 포함하고 있어 가장 높은 성능을 냈다고 해석하지만, 본문의 주된 비교 backbone은 CNN module 없이 local modeling만 강조한 NesT이다.

### 4.4 다른 few-shot framework와 비교

Table 4는 동일하거나 유사한 ViT 조건에서 여러 few-shot framework와 SUN을 비교한다. miniImageNet 기준으로 SUN은 5-way 1-shot에서 66.54%, 5-shot에서 82.09%를 기록한다. 이는 BML의 59.35% / 76.00%, [8]+DrLoc의 57.85% / 74.03%, [8]+Semiformer의 57.91% / 73.31%보다 훨씬 높다.

저자들이 강조하는 포인트는, SUN이 단순히 ViT를 조금 개선한 정도가 아니라 “같은 ViT backbone을 썼을 때 기존 few-shot training framework들보다 훨씬 더 few-shot 친화적인 representation을 만든다”는 점이다.

### 4.5 CNN 기반 SoTA와 비교

Table 5는 CNN backbone 기반의 state-of-the-art few-shot classification 방법들과 SUN을 비교한다. 결과는 꽤 강하다.

NesT backbone의 SUN은  
miniImageNet에서 66.54% / 82.09%,  
tieredImageNet에서 72.93% / 86.70%,  
CIFAR-FS에서 78.17% / 88.98%를 기록한다.

Visformer backbone의 SUN은  
miniImageNet에서 67.80% / 83.25%,  
tieredImageNet에서 72.99% / 86.74%,  
CIFAR-FS에서 78.37% / 88.84%를 기록한다.

이 수치는 tieredImageNet과 CIFAR-FS에서 기존 SoTA를 넘어서는 결과로 제시된다. 특히 CIFAR-FS에서는 기존 방법들보다 1-shot에서 2.5% 이상 높다고 논문이 주장한다. miniImageNet에서는 Visformer 기반 SUN이 67.80%로 RE-Net 67.60%, TPMN 67.64%보다 소폭 높다. 즉, 논문의 주장대로 ViT 기반 few-shot 방법이 유사 크기의 CNN 기반 최신 기법과 경쟁하거나 일부 데이터셋에서는 앞서기 시작했다는 실험적 근거가 제시된다.

### 4.6 Ablation Study

Table 6은 SUN의 각 구성요소를 단계적으로 넣은 결과다. base는 49.23% / 66.57%이고, teacher ViT만 도입한 타입 (a)는 62.40% / 79.45%로 크게 오른다. 이는 teacher pretraining 자체가 이미 큰 효과를 준다는 뜻이다. 여기에 local supervision, SCA, BGF를 차례로 넣으면 최종 66.54% / 82.09%까지 올라간다.

즉, 성능 향상의 대부분은 teacher 기반 meta-training과 local supervision에서 오고, SCA와 BGF가 그 위에 추가적인 안정성과 일반화 향상을 준다고 해석할 수 있다.

Table 7에서는 teacher의 drop path rate를 비교하는데, $p_{dpr}=0.5$일 때 가장 좋다. 이는 few-shot처럼 overfitting 위험이 큰 환경에서 큰 drop path가 유리함을 시사한다.

Table 8은 Meta-Baseline, FEAT, DeepEMD와 결합한 SUN-M/F/D 결과를 보여준다. 예를 들어 SUN-D는 miniImageNet에서 69.56% / 85.38%로, DeepEMD 68.77% / 84.13%보다 높고 COSOC 69.28% / 85.16%보다도 약간 높다. 이는 SUN이 특정 meta-tuning 방식에 묶이지 않는다는 점을 보여준다.

Appendix의 추가 실험도 같은 메시지를 강화한다. 더 많은 meta-training epoch는 miniImageNet에서 성능 향상을 주고, global JSD를 loss에 추가하는 것은 오히려 약간 성능을 떨어뜨린다. 저자들은 JSD가 global prediction을 teacher에 지나치게 맞추게 해서 local supervision의 이점을 일부 약화시킨다고 해석한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 방법 설계가 명확하게 연결된다는 점이다. 저자들은 먼저 “ViT가 few-shot에서 왜 안 되는가”를 실험적으로 보여주고, 그 원인을 token dependency learning 문제로 분석한 뒤, 이를 직접 겨냥하는 local supervision 기반 방법을 제안한다. 단순히 성능을 올렸다고 주장하는 것이 아니라, attention map, 학습 곡선, inductive bias 비교 실험 등을 통해 왜 그런지 설명하려 한 점이 설득력 있다.

둘째, 방법이 구조적으로 비교적 단순하다. 외부 annotation을 추가로 요구하지 않고, 같은 ViT teacher가 patch-level pseudo supervision을 생성한다. 또한 Meta-Baseline, FEAT, DeepEMD와 결합 가능하다는 점에서 범용성이 있다. backbone도 LV-ViT, Swin, Visformer, NesT 등 여러 종류에 대해 효과를 보였다.

셋째, 실험 결과가 강하다. 단순한 baseline 개선이 아니라, 당시 CNN 기반 SoTA와 경쟁 가능한 수준까지 ViT few-shot 성능을 끌어올렸다는 점은 논문의 실질적 기여로 볼 수 있다.

반면 한계도 분명하다. 첫째, local supervision은 teacher pseudo label의 품질에 크게 의존한다. 논문이 BGF와 SCA로 이를 개선하려 했지만, pseudo label 자체가 잘못되면 오히려 잘못된 local bias를 강화할 위험이 있다. 이 문제는 논문도 간접적으로 인정하고 있으며, top-$k$ confidence만 유지하는 식으로 label noise를 줄이려 한다.

둘째, 방법의 계산 비용은 baseline보다 커진다. teacher를 먼저 따로 학습해야 하고, meta-training에서 global/local classifier와 patch-level supervision을 모두 사용한다. 논문은 정확도 개선을 주로 강조하지만, 학습 복잡도나 시간 비용을 본문에서 정량적으로 비교하지는 않는다.

셋째, 논문은 token dependency learning의 향상을 attention map과 성능 향상으로 설명하지만, 이것이 정확히 어떤 조건에서 얼마나 일반적으로 성립하는지에 대한 이론적 설명은 제한적이다. 즉, “왜 이 pseudo supervision이 self-attention 구조에서 본질적으로 잘 작동하는가”에 대한 더 깊은 해석은 남아 있다.

넷째, 발표연도와 arXiv 링크, 그리고 일부 비교 방법의 정확한 실험 세팅 차이 같은 메타 정보는 제공된 추출 텍스트만으로는 완전히 확인되지 않는다. 따라서 이 보고서는 제공된 원문 텍스트 범위 내에서만 분석했으며, 명시되지 않은 사항은 확정적으로 서술하지 않았다.

## 6. 결론

이 논문은 few-shot classification에서 ViT가 CNN보다 약한 이유를 inductive bias 부족과 느리고 저품질의 token dependency learning 문제로 진단하고, 이를 해결하기 위해 patch-level dense supervision을 사용하는 **SUN** 프레임워크를 제안한다. SUN은 teacher ViT가 생성한 location-specific pseudo label로 각 patch token을 지도하고, BGF와 SCA를 통해 그 supervision의 품질을 높인다. 그 결과 ViT backbone의 few-shot generalization이 크게 향상되며, 일부 데이터셋에서는 CNN 기반 state-of-the-art를 넘어서는 성능까지 달성한다.

실제로 이 연구의 의미는 단순히 한 방법의 성능 향상에 그치지 않는다. ViT가 “데이터가 많아야만 강한 모델”이라는 인식을 넘어서, 적은 데이터 환경에서도 적절한 supervision 설계를 통해 충분히 경쟁력 있을 수 있음을 보여준다. 향후에는 이 아이디어가 few-shot classification을 넘어 detection, segmentation, multimodal few-shot learning 같은 더 복잡한 low-data setting으로 확장될 가능성이 있다.
