# Learning Mask-aware CLIP Representations for Zero-Shot Segmentation

- **저자**: Siyu Jiao, Yunchao Wei, Yaowei Wang, Yao Zhao, Humphrey Shi
- **발표연도**: 2023
- **arXiv**: https://arxiv.org/abs/2310.00240

## 1. 논문 개요

이 논문은 zero-shot segmentation에서 널리 쓰이는 "mask proposal 생성 후 frozen CLIP으로 proposal 분류"라는 기존 패러다임의 약점을 분석하고, 이를 개선하기 위한 **MAFT (Mask-Aware Fine-tuning)** 를 제안한다. 저자들의 핵심 문제 제기는 간단하다. 기존 CLIP은 image-level supervision으로 학습되었기 때문에, 같은 이미지에서 나온 서로 다른 mask proposal들을 충분히 구분하지 못한다. 즉 proposal이 정확한 객체 영역을 담고 있든, 일부만 담고 있든, 혹은 배경까지 많이 포함하든, frozen CLIP은 비슷한 클래스를 높은 확률로 예측하는 경향이 있다.

이 문제는 zero-shot segmentation에서 매우 중요하다. 왜냐하면 이 계열 방법들은 최종적으로 proposal 단위의 분류 품질에 크게 의존하기 때문이다. proposal generator가 여러 mask를 만들어도, CLIP이 그것들을 구분하지 못하면 false positive가 많이 생긴다. 특히 unseen class에 대한 segmentation에서는 proposal 자체의 품질뿐 아니라 proposal별 semantic discrimination이 핵심인데, 기존 frozen CLIP은 이 점에서 한계를 보인다고 논문은 주장한다.

이에 따라 저자들은 CLIP이 서로 다른 mask proposal에 더 민감하게 반응하도록 미세조정하되, 동시에 CLIP의 본래 장점인 novel class에 대한 transferability는 유지하는 방향을 목표로 삼는다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 CLIP을 pixel-level segmentation 모델로 바꾸려는 것이 아니라, **region-level 또는 proposal-level 인식에 더 적합한 방향으로 최소한의 구조 변경과 손실 설계를 통해 fine-tuning** 하는 것이다. 저자들은 pixel 단위의 dense prediction으로 곧바로 CLIP을 적응시키는 것은 domain gap이 너무 커서 잘 안 될 수 있다고 보고, proposal classification이라는 더 가까운 중간 단계에 초점을 맞춘다.

이를 위해 두 가지를 결합한다. 첫째, **IP-CLIP Encoder (Image-Proposals CLIP Encoder)** 를 도입해 한 장의 이미지와 여러 개의 mask proposal을 동시에 처리할 수 있게 만든다. 둘째, **mask-aware loss** 를 사용해 proposal의 IoU 품질과 CLIP의 classification score가 정렬되도록 학습시킨다. 이 과정에서 좋은 proposal은 높은 점수, 나쁜 proposal은 낮은 점수를 받도록 유도한다. 셋째, **self-distillation loss** 를 사용해 fine-tuned 모델이 마스크 없이 전체 이미지를 볼 때는 원래 frozen CLIP의 출력을 따르도록 하여 transferability 손상을 줄인다.

기존 접근과의 차별점은, 단순히 CLIP을 freeze한 채 proposal을 자르고 분류하는 것이 아니라, CLIP 내부 attention이 mask proposal을 반영하도록 만들고, proposal 품질에 맞춰 점수를 조정하도록 학습시킨다는 점이다. 동시에 새로운 대규모 모듈이나 많은 추가 파라미터 없이 기존 frozen-CLIP 계열 방법에 plug-and-play로 들어갈 수 있게 설계했다는 점도 차별점이다.

## 3. 상세 방법 설명

기존 "frozen CLIP" 계열 방법은 두 단계로 동작한다. 먼저 Proposal Generator가 class-agnostic한 mask proposal $M \in \mathbb{R}^{N \times H \times W}$ 와 seen class에 대한 proposal classification score $A_p$ 를 생성한다. 그 다음 각 proposal과 원본 이미지를 합성한 sub-image를 CLIP image encoder에 넣고, text encoder가 만든 text embedding과 cosine similarity를 계산해 CLIP score $A_c$ 를 얻는다. 논문은 이 과정에서 CLIP이 proposal 차이를 충분히 반영하지 못한다고 본다.

CLIP의 기존 분류 점수는 다음과 같이 계산된다.

$$
A^i_c = \text{Softmax}\left(
\frac{\exp\left(\frac{1}{\tau}s_c(E^i_T, E_I)\right)}
{\sum_{i=0}^{C}\exp\left(\frac{1}{\tau}s_c(E^i_T, E_I)\right)}
\right)
$$

여기서 $\tau$ 는 temperature이고, $s_c(E^i_T, E_I)$ 는 text embedding과 image embedding 사이의 cosine similarity이다.

### IP-CLIP Encoder

저자들은 CLIP ViT image encoder를 수정해 **IP-CLIP Encoder** 를 만든다. 핵심은 Transformer의 중간 이후 layer에서, 각 proposal마다 별도의 classification query를 두고, 이 query가 해당 proposal mask 내부 영역과 자기 자신에게만 attention하도록 만드는 것이다.

Transformer layer $i$ 에서 feature를 다음처럼 둔다.

$$
F_i = [F^{cls}_i; F^{feat}_i] \in \mathbb{R}^{(1+hw)\times d}
$$

여기서 $F^{cls}_i$ 는 class token, $F^{feat}_i$ 는 flatten된 image feature이다. proposal이 $N$개 있으면 class token을 $N$번 복제해

$$
F^*_i = [F^{*cls}_i; F^{feat}_i] \in \mathbb{R}^{(N+hw)\times d}
$$

형태로 만든다. 즉 proposal마다 하나의 class-like token이 생긴다.

이후 proposal mask를 attention bias로 변환한다. attention bias $B$ 는 proposal query가 자신에게 해당하는 mask 영역과 자기 자신만 볼 수 있도록 만든다.

$$
B(i,j) =
\begin{cases}
0, & \hat{M}(i,j)=1 \\
-\infty, & \hat{M}(i,j)=0
\end{cases}
,\quad
\hat{M} = [I(N,N); \text{Flat}(M)]
$$

이 bias를 이용해 proposal query 쪽 attention은 다음처럼 계산된다.

$$
F^{(i+1)*}_{cls}
=
\text{Softmax}
\left(
\frac{\text{Que}(F^{*cls}_i)\text{Key}(F^*_i)^T}{\sqrt{d}} + B
\right)
\text{Val}(F^*_i)
$$

반면 image feature 쪽은 standard self-attention으로 유지된다.

$$
F^{i+1}_{feat}
=
\text{Softmax}
\left(
\frac{\text{Que}(F^{feat}_i)\text{Key}(F^{feat}_i)^T}{\sqrt{d}}
\right)
\text{Val}(F^{feat}_i)
$$

이 설계의 의미는 분명하다. proposal $n$ 에 대응하는 class embedding은 그 proposal mask가 가리키는 영역과 자기 자신만 사용해 정보를 모은다. 하지만 이미지 feature 자체는 전체 문맥을 계속 유지한다. 논문은 이 방식이 단순 crop 기반 sub-image 처리보다 context를 더 잘 보존하고, proposal마다 이미지를 반복 인코딩하지 않아 계산량도 크게 줄인다고 주장한다. 실제 ablation에서 CLIP image encoder 계산량이 1127.0 GFLOPs에서 53.4 GFLOPs로 크게 감소했다.

### Mask-aware Loss

IP-CLIP Encoder만 도입해도 proposal 처리 구조는 좋아지지만, 여전히 CLIP pretraining만으로는 proposal quality 차이를 충분히 반영하지 못할 수 있다. 그래서 저자들은 ground-truth와 proposal 간 IoU를 supervision으로 사용한다.

ground-truth에 존재하는 $k$개 클래스에 대해 binary mask를 만들고, 각 proposal과의 IoU를 계산해 $S_{IoU} \in \mathbb{R}^{K \times N}$ 를 얻는다. 다만 $A_c$ 의 최대값은 1에 가깝고, IoU 최대값은 보통 0.75~0.99라 scale mismatch가 있어서 min-max normalization을 적용한다.

$$
S^{norm}_{IoU} =
\frac{S_{IoU} - \min(S_{IoU})}
{\max(S_{IoU}) - \min(S_{IoU})}
$$

그 다음 $A_c$ 에서 ground-truth에 존재하는 클래스에 해당하는 부분 $A^{select}_c$ 를 뽑아, 이를 $S^{norm}_{IoU}$ 와 **SmoothL1 Loss** 로 맞춘다.

$$
L_{ma}(A^{select}_c, S^{norm}_{IoU}) = \text{SmoothL1}(A^{select}_c, S^{norm}_{IoU})
$$

SmoothL1은 다음과 같다.

$$
\text{SmoothL1}(x,y) =
\begin{cases}
0.5(x-y)^2, & |x-y| < 1 \\
|x-y| - 0.5, & \text{otherwise}
\end{cases}
$$

이 손실의 목적은 "proposal이 GT와 잘 맞을수록 해당 클래스 score도 높아져야 한다"는 것을 CLIP에 가르치는 것이다. 즉, 분류 결과를 단순 class identity만 맞추는 것이 아니라 proposal quality까지 반영하도록 만든다.

### Self-distillation Loss

fine-tuning 과정에서 seen class에 과적합되면 zero-shot transferability가 망가질 수 있다. 이를 막기 위해 frozen CLIP을 teacher, fine-tuned IP-CLIP을 student로 두는 self-distillation을 추가한다. 이때 mask 없이 전체 이미지만 입력했을 때 student의 출력 $A_S$ 가 teacher의 출력 $A_T$ 와 같아지도록 한다.

$$
L_{dis}(A_S, A_T) = \text{SmoothL1}(A_S, A_T)
$$

최종 손실은 다음과 같다.

$$
L = L_{ma} + \lambda L_{dis}
$$

논문에서는 $\lambda = 1$ 을 사용했다.

이 self-distillation의 의미는, proposal-aware한 구분력은 새로 학습하되, 전체 이미지에 대한 CLIP의 원래 semantic alignment는 유지하자는 것이다. 이것이 논문이 말하는 "mask-aware하면서도 transferability를 잃지 않는" 핵심 장치다.

### 학습 절차

학습은 2단계다. 1단계에서는 기존 ZegFormer, ZSSeg, FreeSeg 방식대로 proposal generator를 학습한다. 2단계에서는 proposal generator와 CLIP text encoder는 freeze하고, IP-CLIP Encoder만 MAFT로 미세조정한다. batch size는 16, 입력 크기는 $480 \times 480$, optimizer는 AdamW, learning rate와 weight decay는 각각 $10^{-5}$ 이다. iteration 수는 Pascal-VOC 100, COCO-Stuff 1000, ADE20K 5000으로 설정했다. 논문은 전체 fine-tuning이 1 epoch보다 적은 수준으로 매우 효율적이라고 설명한다.

## 4. 실험 및 결과

### 데이터셋과 평가 설정

논문은 zero-shot segmentation 설정에서 Pascal-VOC, COCO-Stuff, ADE20K를 사용한다. Pascal-VOC는 15 seen / 5 unseen, COCO-Stuff는 156 seen / 15 unseen, ADE20K는 572 seen / 275 unseen 분할을 따른다. 또한 open-vocabulary 설정에서는 COCO-Stuff로 학습하고 A-847, A-150, PC-459, PC-59, PAS-20에서 평가한다.

평가 지표는 seen class에 대한 mIoU인 $mIoU_s$, unseen class에 대한 $mIoU_u$, 그리고 둘의 조화평균인 $hIoU$ 이다.

### Zero-shot setting에서의 주요 결과

가장 중요한 결과는 MAFT가 여러 대표 방법에 일관되게 성능 향상을 준다는 점이다.

FreeSeg 기준으로 보면, unseen class mIoU가 다음과 같이 개선된다.

- COCO-Stuff: 42.2% $\rightarrow$ 50.4% (+8.2)
- Pascal-VOC: 78.6% $\rightarrow$ 81.8% (+3.2)
- ADE20K: 4.4% $\rightarrow$ 8.7% (+4.3)

이는 단순한 미세 개선이 아니라, 특히 COCO와 ADE20K에서 상당히 큰 향상이다. 논문은 이 결과를 통해 MAFT가 proposal classification 단계의 병목을 실제로 해결한다고 주장한다.

### Ensemble 제거 실험

기존 frozen-CLIP 계열 방법은 proposal generator의 seen-class score $A_p$ 와 CLIP의 score $A_c$ 를 ensemble한다. 그래서 원래 표에서는 seen class 성능이 proposal generator에 많이 의존한다. 저자들은 MAFT가 정말 CLIP classifier 자체를 개선하는지 보기 위해 ensemble을 제거한 실험도 수행한다.

이 실험에서 성능 향상은 더 극적이다. 예를 들어 FreeSeg는 COCO에서 $hIoU$ 가 25.3에서 44.4로, unseen mIoU는 29.3에서 49.7로 크게 상승한다. 이는 MAFT의 핵심 이득이 CLIP proposal classifier 자체에 있음을 잘 보여준다.

### Open-vocabulary setting 결과

open-vocabulary에서도 FreeSeg + MAFT는 다음처럼 개선된다.

- A-847: +3.0
- A-150: +11.2
- PC-459: +6.4
- PC-59: +19.1
- PAS-20: +4.4

논문은 OpenSeg와 OVSeg와도 비교하며, 추가 학습 데이터를 쓰는 OpenSeg보다도 여러 벤치마크에서 우수한 결과를 보였다고 주장한다. 이는 self-distillation을 통해 transferability를 유지했다는 논문의 주장을 어느 정도 뒷받침한다.

### Ablation Study

Ablation은 이 논문의 설계 타당성을 보여주는 중요한 부분이다.

첫째, 구성요소별 ablation에서 IP-CLIP만 추가해도 COCO에서 seen/unseen이 각각 +7.1, +6.9 향상된다. 이는 구조 변경만으로도 context 활용과 계산 효율의 이득이 있음을 뜻한다. 여기에 $L_{ma}$ 를 더하면 unseen mIoU가 47.1까지 오르고, 다시 $L_{dis}$ 를 추가하면 49.7까지 올라간다. 즉, 구조 변경과 손실 설계가 각각 의미 있는 기여를 한다.

둘째, $L_{ma}$ 의 형태를 비교하면 SmoothL1이 unseen class 성능에서 가장 균형 잡힌 결과를 낸다. KL loss는 seen class에는 좋지만 unseen class에서는 상대적으로 불리해 transferability를 해친다고 해석한다.

셋째, training iteration을 늘리면 seen class는 계속 오르지만 unseen class는 1k 이후 감소한다. 이는 과적합의 증거로 해석되며, 논문이 "짧은 fine-tuning"을 강조하는 이유다.

넷째, CLIP 내부 어떤 unit을 freeze할지 실험한 결과, 일부 unit, 특히 conv, cls, pos, mlp 등을 freeze하는 것이 unseen 성능 개선에 도움이 되었다. 이는 모든 파라미터를 다 건드리는 것보다 선택적 fine-tuning이 더 낫다는 신호다.

다섯째, mask attention을 시작하는 layer $L$ 에 대한 실험에서는 $L=8$ 이 가장 좋았다. 너무 이르게 mask attention을 넣으면 context를 충분히 얻지 못하고, 너무 늦게 넣으면 mask-aware 특성이 약해지는 것으로 해석된다.

### SAM 및 다른 Vision-Language Model로의 확장

논문은 Proposal Generator로 SAM-H를 사용한 실험도 제시한다. SAM + MAFT는 SAM 대비 zero-shot과 open-vocabulary 모두에서 성능이 향상되며, 일부 벤치마크에서는 FreeSeg + MAFT보다도 더 좋다. 특히 Pascal-VOC zero-shot에서 unseen mIoU가 86.7에서 88.6으로 올라가고, 이는 FreeSeg + MAFT보다도 높다. 저자들은 이를 SAM의 stronger proposal generalization 덕분으로 해석한다.

또한 CLIP-ViT-L과 CLIP-Res50에도 MAFT를 적용했다. ViT-L에서는 state-of-the-art 수준을 갱신했고, Res50 기반 CLIP에서도 큰 향상이 나타났다. 이는 MAFT가 특정 backbone에만 맞춘 기법이 아니라는 근거로 제시된다.

### 정성적 결과

정성적 분석에서 frozen CLIP은 foreground 일부만 포함한 proposal이나 배경이 많이 섞인 proposal에도 원래 객체 클래스를 계속 높게 주는 경향을 보인다. 반면 MAFT 적용 후에는 좋은 foreground proposal, 좋은 background proposal, noisy proposal을 더 잘 구분한다. 즉 "proposal quality-aware classification"이 실제 예측에서 나타난다는 것을 시각적으로 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의가 명확하다는 점이다. 저자들은 단순히 성능을 올렸다고 주장하는 것이 아니라, 기존 frozen CLIP 기반 zero-shot segmentation이 왜 실패하는지 구체적인 failure mode를 제시한다. "CLIP이 서로 다른 proposal에 둔감하다"는 관찰은 설득력이 있고, 이후 방법론도 정확히 그 문제를 겨냥한다.

또 다른 강점은 방법이 비교적 단순하고 plug-and-play라는 점이다. 대규모 새 모듈을 붙이기보다 CLIP image encoder 내부 attention 동작을 proposal-aware하게 바꾸고, IoU 기반 정렬 손실과 distillation을 추가하는 방식이라 기존 시스템에 붙이기 쉽다. 실제로 ZegFormer, ZSSeg, FreeSeg, SAM, ViT-L, Res50 등 다양한 설정에서 개선을 보였다는 점은 일반성을 뒷받침한다.

실험적으로도 강하다. 단순 zero-shot뿐 아니라 open-vocabulary setting, SAM, 다른 VLM backbone까지 검증했고, ensemble 제거 실험과 upper bound 분석도 포함했다. 특히 upper bound가 COCO에서 77.6 mIoU인데 현재 MAFT는 43.9 수준이라는 결과는 proposal 품질 자체보다 proposal classification이 여전히 더 큰 병목임을 보여준다.

한편 한계도 분명하다. 첫째, 성능은 여전히 pre-trained vision-language model의 본래 표현력에 의해 제한된다. 논문 마지막에서도 novel class classification 능력 자체는 여전히 한계가 있다고 인정한다. 둘째, mask-aware loss는 ground-truth mask와 proposal IoU에 의존하므로 학습 시 supervision이 필요하다. 따라서 완전히 annotation-free한 방향은 아니다. 셋째, self-training 실험은 unseen class 성능을 더 올릴 수 있었지만, 학습 중 unseen class 이미지가 있어야 한다는 제약이 있어 일반적인 open-vocabulary 상황에 바로 적용하기 어렵다고 논문이 직접 밝힌다.

비판적으로 보면, 논문은 proposal-level CLIP adaptation의 효과를 잘 보여주지만, 왜 SmoothL1 alignment가 semantic ranking 측면에서 가장 적절한지에 대한 이론적 설명은 제한적이다. 또한 attention bias 설계가 다양한 proposal noise 유형에서 어떤 식으로 작동하는지에 대한 더 세밀한 분석은 부족하다. 하지만 이는 후속 연구 여지를 남기는 정도이지, 논문의 핵심 기여를 훼손하는 수준은 아니다.

## 6. 결론

이 논문은 zero-shot segmentation에서 널리 사용되는 frozen CLIP 패러다임의 핵심 약점이 **mask proposal 간 구분력 부족** 에 있음을 지적하고, 이를 해결하기 위해 **MAFT** 를 제안했다. IP-CLIP Encoder는 여러 proposal을 동시에 다루면서도 context를 유지하도록 CLIP을 구조적으로 수정하고, mask-aware loss는 proposal 품질과 분류 점수를 정렬시켜 CLIP을 proposal-sensitive하게 만든다. self-distillation loss는 이 과정에서 CLIP의 zero-shot transferability를 보존하는 역할을 한다.

실험 결과는 이 방법이 단순한 아이디어 이상임을 보여준다. 다양한 zero-shot segmentation 방법에 plug-in 했을 때 unseen class 성능이 일관되게 향상되었고, open-vocabulary, SAM 기반 proposal, 더 강한 VLM backbone에서도 효과가 유지되었다. 따라서 이 연구는 "CLIP을 segmentation에 그대로 가져다 쓰는 것"에서 한 단계 나아가, **CLIP을 proposal-aware하게 적응시키는 방향** 이 중요하다는 점을 보여준다는 의미가 있다.

실제 적용 측면에서도 가치가 있다. proposal generator와 language-image alignment를 사용하는 기존 segmentation 시스템에 비교적 적은 수정으로 성능 개선을 줄 수 있기 때문이다. 향후 연구로는 더 강한 vision-language model과 결합하거나, 더 적은 supervision으로 mask-aware adaptation을 수행하는 방향, 또는 proposal-level을 넘어 dense prediction 수준까지 transferability를 유지하며 확장하는 방향이 자연스럽게 이어질 수 있다.
