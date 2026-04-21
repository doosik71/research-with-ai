# Image Segmentation in Foundation Model Era: A Survey

- **저자**: Tianfei Zhou, Wang Xia, Fei Zhang, Boyu Chang, Wenguan Wang, Ye Yuan, Ender Konukoglu, Daniel Cremers
- **발표연도**: 2024
- **arXiv**: https://arxiv.org/abs/2408.12957

## 1. 논문 개요

이 논문은 새로운 segmentation model 하나를 제안하는 연구가 아니라, foundation model(FM) 시대에 image segmentation이 어떻게 재편되고 있는지를 체계적으로 정리한 survey이다. 저자들은 기존 segmentation 연구가 semantic, instance, panoptic segmentation 같은 전통적 과제 중심으로 발전해 왔지만, CLIP, Stable Diffusion, DINO, SAM, LLM/MLLM 같은 foundation model이 등장하면서 문제 정의, 학습 방식, 모델 인터페이스, 그리고 가능한 능력 자체가 크게 바뀌었다고 본다.

논문의 핵심 문제의식은 다음과 같다. 기존 survey들은 대체로 2021년 이전의 deep learning 기반 segmentation을 다루거나, SAM 같은 특정 모델만 좁게 다루기 때문에, FM이 segmentation에 가져온 변화 전체를 설명하지 못한다. 특히 최근에는 단순히 segmentation 성능을 올리는 수준을 넘어, promptable segmentation, training-free segmentation, reasoning이 결합된 segmentation, diffusion model을 통한 synthetic data 생성 등 과거에는 없던 연구 흐름이 빠르게 생겨나고 있다. 이 논문은 바로 이 변화의 지형도를 제공하려는 목적을 가진다.

이 문제가 중요한 이유는 segmentation이 여전히 computer vision의 핵심 기반 문제이기 때문이다. 논문은 segmentation이 scene understanding, visual reasoning, affordance perception의 출발점이며, autonomous driving, medical image analysis, surveillance, image editing 등 실제 응용 전반에 연결된다고 강조한다. 따라서 FM이 segmentation을 어떻게 바꾸는지 이해하는 일은 단순한 모델 리뷰가 아니라, 앞으로 vision system 전체가 어떤 방향으로 진화할지를 이해하는 일과 연결된다.

## 2. 핵심 아이디어

이 survey의 가장 중요한 아이디어는 현대 image segmentation을 단순히 task 목록으로 정리하지 않고, foundation model이 segmentation에 기여하는 방식에 따라 다시 구조화했다는 점이다. 저자들은 크게 두 줄기의 segmentation을 구분한다. 하나는 prompt 없이 이미지를 분할하는 generic image segmentation(GIS)이고, 다른 하나는 prompt를 받아 특정 대상을 분할하는 promptable image segmentation(PIS)이다. 이 구분은 FM 시대의 segmentation을 설명하는 데 매우 적합하다. 왜냐하면 SAM, SEEM, SegGPT, LISA 같은 최근 모델들은 “무엇을 분할할지”를 prompt로 지정받는 인터페이스를 가지며, 이는 과거의 closed-set segmentation과 본질적으로 다르기 때문이다.

논문이 제시하는 중심 직관은, FM은 segmentation을 위해 직접 학습되지 않았더라도 segmentation knowledge를 내부적으로 갖고 있을 수 있다는 것이다. 예를 들어 CLIP은 원래 image-text contrastive learning을 위해 학습되었고, Stable Diffusion은 image generation을 위해 학습되었으며, DINO는 self-supervised representation learning을 위해 학습되었다. 그런데도 이들 모델 내부의 attention map, token affinity, latent representation을 잘 읽어내면 segmentation mask를 만들 수 있다. 저자들은 이것을 “segmentation knowledge emerges from FMs”라는 관점으로 정리한다.

기존 접근과의 차별점도 분명하다. 전통적인 segmentation은 대체로 특정 task를 위해 supervised training을 수행하는 전용 모델을 만드는 방향이었다. 반면 FM 시대에는 다음과 같은 차이가 있다. 첫째, segmentation generalist가 등장했다. 둘째, training-free segmentation이라는 새로운 패러다임이 생겼다. 셋째, LLM/MLLM이 segmentation에 reasoning과 world knowledge를 가져오기 시작했다. 넷째, diffusion model이 단지 생성기가 아니라 segmentation feature extractor이자 synthetic data engine으로 사용된다. 이 네 가지 흐름이 이 논문의 핵심 메시지다.

## 3. 상세 방법 설명

이 논문은 먼저 image segmentation을 하나의 통합 수식으로 정리한다. 저자들은 segmentation을 입력 공간 $X$에서 출력 공간 $Y$로 가는 함수 $f$를 학습하는 문제로 쓴다.

$$
f : X \to Y, \quad X = I \times P,\quad Y = M \times C
$$

여기서 $I$는 image, $P$는 prompt 집합, $M$은 segmentation mask, $C$는 semantic category vocabulary이다. 이 식의 장점은 segmentation task를 하나의 공통 틀에서 설명할 수 있다는 점이다. prompt가 없으면 generic image segmentation이고, prompt가 있으면 promptable image segmentation이 된다. 또한 출력이 class label 중심인지, instance 구분까지 포함하는지에 따라 semantic / instance / panoptic segmentation으로 세분화된다.

논문은 GIS와 PIS를 다음처럼 나눈다. GIS는 $P=\varnothing$인 경우이고, semantic segmentation은 픽셀마다 class를 예측하며, instance segmentation은 같은 class 안에서도 object instance를 구분하고, panoptic segmentation은 둘을 합친다. 또 training class와 test class가 같은 closed-vocabulary와, test에 unseen class가 포함되는 open-vocabulary를 구분한다. PIS는 prompt를 이용하는데, interactive segmentation은 click, scribble, box, polygon 같은 visual prompt를 쓰고, referring segmentation은 text phrase를 쓰며, few-shot segmentation은 support image-mask pair를 prompt처럼 사용한다.

논문은 학습 패러다임도 함께 정리한다. supervised, unsupervised, weakly supervised 외에 FM 시대에 새롭게 주목받는 training-free segmentation을 별도 범주로 둔다. 이는 pre-trained FM에서 segmentation을 직접 추출하며 추가 학습이 필요 없다는 뜻이다.

이후 foundation model 자체를 간단히 설명한다. CLIP은 image-text contrastive learning으로 학습되며 손실은 다음과 같이 제시된다.

$$
L_{i2t} = -\log \left[
\frac{\exp(\mathrm{sim}(x_k,t_k)/\tau)}
{\sum_{j=1}^{J}\exp(\mathrm{sim}(x_k,t_j)/\tau)}
\right]
$$

실제 학습은 image-to-text와 text-to-image를 함께 써서 $L_{\text{contrast}} = L_{i2t} + L_{t2i}$ 형태가 된다. 이 수식의 의미는 맞는 image-text 쌍의 임베딩은 가깝게, 틀린 쌍은 멀게 만드는 것이다. 그래서 CLIP은 풍부한 semantic concept를 가지지만, 원래는 image-level alignment 중심이라 spatial localization에는 약점이 있다.

Diffusion model은 forward noising과 reverse denoising으로 설명된다. DDPM의 핵심은 원본 데이터 $z_0$에 점점 noise를 추가해 $z_t$를 만들고,

$$
z_t \sim \mathcal{N}(\sqrt{\alpha_t} z_{t-1}, (1-\alpha_t)I)
$$

reverse 과정에서 네트워크 $\epsilon_\theta$가 추가된 noise를 예측하도록 학습하는 것이다.

$$
L_{DM} = \mathbb{E}_{z_0,\epsilon \sim \mathcal{N}(0,1),t}
\left[\|\epsilon - \epsilon_\theta(z_t(z_0,\epsilon), t; C)\|_2^2\right]
$$

LDM은 이를 latent space에서 수행한다. 논문은 이 구조 때문에 diffusion model 내부가 high-level semantic과 low-level structure를 모두 품을 수 있으며, attention map이 segmentation에 유용하다고 설명한다.

DINO/DINOv2는 self-supervised ViT인데, 마지막 attention layer의 class token이 object localization 정보를 가진다는 점이 중요하다. 논문은 class token과 patch token 사이의 affinity를

$$
\alpha_{\text{CLS}} = q_{\text{CLS}} \cdot k_I^\top
$$

로 쓰고, 이것을 binarize하면 segmentation mask를 얻을 수 있다고 설명한다. patch 간 affinity는

$$
A_I = k_I \cdot k_I^\top
$$

로 계산되며, 이것을 clustering이나 graph partition에 사용하면 unsupervised segmentation으로 이어진다.

이 survey가 특히 잘 정리하는 부분은 “어떤 FM이 어떤 방식으로 segmentation에 쓰이는가”이다.

CLIP 기반 방법에서는 크게 다섯 부류가 나온다. 첫째, training-free segmentation이다. MaskCLIP류 방법은 CLIP attention pooling을 수정해 spatially localized feature를 얻고, text encoder를 classifier로 써서 추가 학습 없이 segmentation을 한다. 둘째, CLIP fine-tuning이다. DenseCLIP, ZegCLIP, SAN 같은 방법은 전체 또는 일부 파라미터를 조정해 pixel-text alignment를 높인다. 셋째, CLIP을 zero-shot classifier로 쓰는 방법이다. class-agnostic mask proposal을 먼저 만들고 CLIP으로 class를 붙이거나, 직접 pixel classification을 수행한다. 넷째, text supervision만으로 semantic segmentation이 emergence되도록 하는 방법이다. GroupViT, SegCLIP 등이 여기에 속한다. 다섯째, knowledge distillation이다. CLIP의 semantic knowledge를 segmentation model로 옮긴다.

Diffusion model 기반 방법도 네 갈래로 정리된다. 첫째, training-free semantic segmentation이다. cross-attention map에서 class token 관련 map을 뽑고, 필요하면 self-attention으로 보정해 mask를 만든다. 둘째, diffusion feature를 dense representation으로 쓰는 방법이다. VPD, LD-ZNet 같은 모델이 SD의 UNet feature를 segmentation decoder에 공급한다. 셋째, segmentation 자체를 denoising diffusion process로 재정의하는 접근이다. 넷째, text-to-image diffusion model을 segmentation dataset synthesizer로 쓰는 흐름이다. synthetic image와 mask를 대량 생성해 downstream model을 학습시킨다.

DINO 기반 방법은 주로 unsupervised segmentation과 self-training에 강하다. DINO feature를 바로 clustering하거나, DINO에서 얻은 pseudo label로 별도 segmentation model을 학습한다. SAM 기반 방법은 semantic understanding이 약하지만 object boundary와 mask quality가 강하기 때문에 weakly supervised setting, promptable segmentation, multi-granularity interactive segmentation, medical interactive segmentation에서 널리 활용된다.

PIS 영역에서는 interactive segmentation, referring segmentation, few-shot segmentation이 다뤄진다. interactive segmentation에서는 SAM이 중심이다. HQ-SAM은 output token을 바꿔 mask quality를 높이고, GraCo는 granularity-controllable interaction을 제안한다. referring segmentation에서는 CLIP 기반 dense fusion, diffusion feature 활용, 그리고 LLM/MLLM을 통한 reasoning segmentation이 핵심이다. LISA는 입력 image와 query text를 보고 LLaVA가 생성한 특수한 `[seg]` token embedding을 SAM decoder로 보내 mask를 만드는 구조다. few-shot segmentation에서는 CLIP, DINO, SAM, diffusion 모두 support-query correlation을 만드는 데 사용되며, 특히 in-context segmentation은 task-specific finetuning 없이 support examples를 context로 넣어 query segmentation을 수행하는 새로운 흐름으로 소개된다.

## 4. 실험 및 결과

이 논문은 survey이므로 새로운 benchmark 실험이나 단일 모델의 자체 성능표를 제시하는 논문은 아니다. 따라서 일반적인 방법 논문처럼 하나의 dataset에서 어떤 baseline보다 몇 점 좋았는지를 체계적으로 비교하는 “실험 섹션”이 중심이 되지 않는다. 대신 300편이 넘는 기존 연구를 task와 foundation model 축으로 정리하면서, 각 흐름에서 어떤 방식이 어떤 문제를 해결하는지 설명한다. 이 점은 독자가 반드시 구분해서 읽어야 한다. 즉, 이 논문의 실험적 기여는 새로운 수치 보고가 아니라 기존 연구 결과를 구조화한 데 있다.

그럼에도 본문에는 몇 가지 대표적 실험적 관찰이 등장한다. 예를 들어 CLIP의 spatial invariance 문제를 간단히 수정한 MaskCLIP은 COCO-Stuff에서 mIoU를 11% 향상시켰다고 소개된다. 이는 “CLIP은 원래 localization에 약하다”는 통념과 달리, attention 구조를 약간 바꾸면 segmentation knowledge를 상당 부분 끌어낼 수 있음을 보여주는 사례로 제시된다.

Semantic segmentation 영역에서는 open-vocabulary segmentation이 FM 덕분에 급격히 발전한 것으로 정리된다. CLIP 기반 방법은 semantic recognition에 강하고, DINO나 SAM은 spatial grouping과 mask quality에 강하기 때문에, 최근 방법들은 하나의 FM만 쓰기보다 여러 FM을 조합하는 방향으로 간다고 설명한다. 예를 들어 training-free open-vocabulary segmentation에서는 DINO 또는 SD로 class-agnostic segment를 찾고, CLIP으로 label을 부여하는 조합이 반복적으로 등장한다.

Instance segmentation에서는 CLIP이 unseen category classification에, diffusion model은 synthetic data generation에, DINO는 label-free object discovery에 기여한다. Panoptic segmentation에서는 Mask2Former류 query-based mask classification과 CLIP zero-shot classifier 조합, 혹은 diffusion feature와 CLIP classifier의 조합이 핵심 흐름으로 묘사된다.

Promptable image segmentation 쪽에서는 quantitative benchmark보다 capability 자체가 중요하게 다뤄진다. 예를 들어 SAM은 zero-shot generality를 갖춘 promptable segmentor로 소개되고, medical image 분야에서는 잘 정의된 target에는 비교적 잘 작동하지만, weak boundary, low contrast, small object, irregular shape에서는 성능이 부족하다고 survey한다. 즉, “SAM은 만능”이 아니라 domain shift가 큰 분야에서는 fine-tuning이나 adapter가 필요하다는 점이 실험적 관찰로 정리된다.

Referring segmentation과 reasoning segmentation에서는 LLM/MLLM이 복잡한 문장 질의를 처리할 수 있게 해 준다는 점이 강조된다. 기존 referring segmentation이 “the front-runner”처럼 직접적인 지시문에 머물렀다면, LISA류 모델은 “who will win the race?”처럼 reasoning이 필요한 질의도 다룰 수 있다고 정리한다. 다만 이 부분 역시 capability 소개가 중심이며, survey 본문에서 통일된 정량 비교를 제공하지는 않는다.

종합하면, 이 논문의 결과 파트는 특정 수치 경쟁을 보여주기보다 다음의 메시지를 준다. 첫째, FM은 segmentation 성능뿐 아니라 문제 정의 자체를 넓혔다. 둘째, training-free, zero-shot, open-vocabulary, reasoning-aware segmentation이 실제 연구 흐름으로 자리 잡았다. 셋째, semantic understanding과 spatial understanding은 서로 다른 FM에서 강하게 나타나므로 조합형 시스템이 매우 중요하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 범위와 구조화 능력이다. 저자들은 CLIP, Diffusion Models, DINO/DINOv2, SAM, LLM/MLLM을 모두 segmentation 관점에서 연결하고, 이를 GIS와 PIS라는 큰 틀 안에서 다시 정리한다. 단순한 모델 나열이 아니라, “segmentation knowledge가 어디서 emergence되는가”, “어떤 FM이 semantic / spatial / reasoning 능력 중 무엇에 강한가”, “training-free와 fine-tuning 기반 연구가 어떻게 갈라지는가”를 중심으로 서술하기 때문에 독자가 연구 지형을 이해하기 쉽다.

또 다른 강점은 survey임에도 수학적 통합 관점을 제공한다는 점이다. $X = I \times P$, $Y = M \times C$라는 형식화는 각 task를 하나의 프레임에서 바라보게 해 준다. 이는 특히 promptable segmentation을 segmentation의 주변 주제가 아니라 중심 주제로 끌어올리는 데 유용하다. 또한 CLIP, diffusion, DINO에서 segmentation이 어떻게 나오는지를 attention, affinity, latent representation 수준에서 설명해 주어 단순 현상 나열을 넘어서려는 시도가 보인다.

미래 연구 방향을 비교적 구체적으로 제시한 점도 장점이다. 저자들은 explainability, in-context segmentation, MLLM hallucination, scalable data engine, diffusion-based data generation, efficient segmentation model을 주요 열린 문제로 제시한다. 이는 단순 “앞으로 더 연구가 필요하다” 수준이 아니라, 현재 FM 기반 segmentation이 실제로 어디에서 막히는지를 잘 반영한다.

반면 한계도 있다. 첫째, 이 논문은 survey이기 때문에 새로운 실험적 검증이나 공정한 통합 benchmark를 제공하지 않는다. 따라서 독자는 어떤 흐름이 실제로 가장 강한지, 어떤 설정에서 무엇이 실용적인지에 대해서는 원 논문들을 추가로 확인해야 한다. 둘째, 방법 수가 매우 많아 폭넓은 대신 각 개별 방법의 내부 차이를 깊게 파고들기보다는 범주화 수준에 머무는 부분이 있다. 셋째, 논문이 “foundation model이 segmentation을 바꾼다”는 큰 흐름을 설득력 있게 정리하지만, 서로 다른 방법들 사이의 비교 조건이 통일되어 있지 않기 때문에 직접적인 성능 우열을 이 survey만으로 판단하기는 어렵다.

비판적으로 보면, 저자들이 강조하는 “segmentation knowledge emergence”는 흥미롭지만, 그 현상이 왜 나타나는지에 대한 이론적 설명은 아직 충분하지 않다고 스스로 인정한다. 이는 survey의 약점이라기보다 분야 자체의 미해결 문제다. 또 LLM/MLLM 기반 reasoning segmentation은 인상적이지만, hallucination과 reliability 문제가 크다는 점이 함께 제시되므로, 실제 deployment 관점에서는 아직 초기 단계라고 보는 것이 타당하다.

## 6. 결론

이 논문은 foundation model 시대의 image segmentation을 매우 넓고 체계적으로 정리한 첫 종합 survey라는 점에서 가치가 크다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, image segmentation을 GIS와 PIS라는 통합 프레임으로 재구성했다. 둘째, CLIP, diffusion model, DINO, SAM, LLM/MLLM에서 segmentation knowledge가 어떻게 나오고 실제 방법론으로 연결되는지 정리했다. 셋째, 300편 이상을 아우르며 open-vocabulary, training-free, reasoning-aware, in-context segmentation 같은 새로운 연구 흐름을 한눈에 보이게 했다.

실제 적용 측면에서 이 연구는 앞으로 segmentation system이 단일 task 전용 모델에서 벗어나, promptable하고 open-world이며 reasoning-capable한 방향으로 갈 가능성이 높다는 점을 시사한다. 또한 semantic understanding, spatial precision, data generation, reasoning 능력이 하나의 모델에서 모두 완성되지 않았기 때문에, 서로 다른 FM의 장점을 조합하는 연구가 당분간 중요할 가능성이 크다. 향후 연구에서는 explainability, efficiency, hallucination 완화, scalable data engine, stronger in-context learning이 핵심 과제가 될 것으로 보인다.

이 논문은 새로운 알고리즘을 제안하지는 않지만, 현재 segmentation 연구가 어디까지 왔고 다음 질문이 무엇인지 분명하게 보여주는 지도로서 매우 유용하다.
