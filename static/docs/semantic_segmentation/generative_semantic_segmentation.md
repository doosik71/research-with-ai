# Generative Semantic Segmentation

- **저자**: Jiaqi Chen, Jiachen Lu, Xiatian Zhu, Li Zhang
- **발표연도**: 2023
- **arXiv**: https://arxiv.org/abs/2303.11316

## 1. 논문 개요

이 논문은 semantic segmentation을 기존의 pixel-wise discriminative classification 문제가 아니라, **입력 이미지에 조건부인 segmentation mask 생성 문제**로 다시 정의한다. 저자들은 이를 **Generative Semantic Segmentation (GSS)** 라고 부르며, segmentation mask 자체를 생성 모델의 출력으로 취급한다. 핵심은 각 픽셀의 클래스를 직접 분류하는 대신, segmentation mask를 설명하는 latent variable distribution을 도입하고, 입력 이미지로부터 그 latent distribution을 예측한 뒤 다시 mask를 생성하도록 만드는 것이다.

연구 문제는 분명하다. 기존 semantic segmentation은 FCN, DeepLab, SegFormer, MaskFormer류 방법을 포함해 거의 모두가 per-pixel classification에 기반한 discriminative learning을 사용한다. 이 방식은 효율적이지만, generative model이 가진 대규모 사전학습 표현을 직접 활용하기 어렵다. 반면 생성 모델은 이미지 전체 분포를 모델링하므로 더 task-agnostic한 틀을 제공하고, 충분히 잘 설계되면 cross-domain generalization에도 유리할 수 있다. 논문은 바로 이 지점을 파고든다.

이 문제가 중요한 이유는 두 가지다. 첫째, segmentation을 생성 문제로 바꾸면 DALL·E나 VQVAE 같은 대규모 generative pretraining의 표현을 가져와 사용할 가능성이 열린다. 둘째, 논문 실험에 따르면 이런 generative formulation은 단일 도메인에서는 경쟁력 있는 수준을 보이면서도, 더 어려운 cross-domain semantic segmentation에서는 기존 discriminative 계열보다 더 강한 일반화를 보인다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 segmentation mask를 직접 다루기보다, 먼저 이를 RGB 이미지처럼 보이는 특별한 표현인 **maskige** 로 바꾸는 데 있다. segmentation mask는 원래 $H \times W \times K$의 one-hot 또는 multi-channel 형태인데, 이를 각 클래스마다 특정 RGB color를 할당해 $H \times W \times 3$ 이미지처럼 표현한다. 이렇게 하면 이미지용으로 이미 잘 학습된 pretrained VQVAE를 거의 그대로 사용할 수 있다.

그 다음 구조는 variational inference 관점으로 정리된다. 저자들은 segmentation mask $c$가 주어졌을 때 latent token $z$의 posterior $q_\phi(z|c)$를 학습하고, 입력 이미지 $x$가 주어졌을 때 latent prior $p_\psi(z|x)$를 학습한다. 학습 시에는 이 둘이 가까워지도록 만들고, 추론 시에는 이미지로부터 예측한 $z$를 decoder에 넣어 maskige를 생성한 뒤 다시 segmentation mask로 변환한다.

기존 접근과의 차별점은 명확하다. 기존 방법은 결국 각 픽셀에 대한 분류 경계를 학습하는 데 초점이 있다. 반면 GSS는 **mask 전체를 생성 가능한 구조화된 출력**으로 본다. 또한 UViM 같은 기존 generative perception 모델과 비교해도, GSS는 maskige를 통해 대규모 pretrained image representation model을 직접 재사용할 수 있으므로, 첫 단계 학습 비용을 크게 줄이면서도 더 좋은 reconstruction 품질을 얻는다고 주장한다.

## 3. 상세 방법 설명

전체 방법은 두 단계로 구성된다.

첫째는 **latent posterior learning** 이다. 여기서는 segmentation mask로부터 latent token을 얻고 다시 mask를 복원할 수 있어야 한다. 둘째는 **latent prior learning** 이다. 여기서는 입력 이미지로부터, 첫 단계에서 얻은 latent token distribution을 예측할 수 있게 만든다. 추론 시에는 둘째 단계의 image encoder만으로 latent token을 예측하고, decoder를 통해 segmentation을 생성한다.

논문은 기존 semantic segmentation의 목표를 다음과 같이 둔다.

$$
\max_\pi \log p_\pi(c|x)
$$

여기서 $x$는 입력 이미지, $c$는 segmentation mask, $p_\pi$는 pixel classifier이다. 이는 전형적인 discriminative formulation이다.

GSS는 여기에 latent variable $z$를 도입하고 ELBO를 사용한다.

$$
\log p(c|x) \ge E_{q_\phi(z|c)} \left[\log \frac{p(z,c|x)}{q_\phi(z|c)}\right]
$$

이를 전개하면 다음이 된다.

$$
E_{q_\phi(z|c)}[\log p_\theta(c|z)] - D_{KL}(q_\phi(z|c), p_\psi(z|x))
$$

이 식은 매우 중요하다. 첫 번째 항 $E_{q_\phi(z|c)}[\log p_\theta(c|z)]$ 는 **reconstruction term** 이다. 즉 latent token $z$로부터 원래 segmentation mask를 잘 복원해야 한다는 뜻이다. 두 번째 항 $D_{KL}(q_\phi(z|c), p_\psi(z|x))$ 는 **prior matching term** 이다. 즉 mask로부터 얻은 latent posterior와 이미지로부터 예측한 latent prior가 가까워져야 한다는 뜻이다.

모델 구성은 세 부분이다.

$ p_\psi(z|x) $ 는 image encoder $I_\psi$ 이다. backbone으로 ResNet 또는 Swin Transformer를 쓰고, 그 위에 **Multi-Level Aggregation (MLA)** 를 얹는다. 부록 기준으로 Swin-Large backbone의 여러 scale feature를 $1/4$ 해상도로 맞춘 뒤 concat하고, MLP와 shifted-window Transformer 층으로 latent code를 만든다.

$ q_\phi(z|c) $ 는 segmentation mask를 latent token으로 바꾸는 posterior encoder이다. 실제 구현은 pretrained VQVAE encoder $E_\phi$ 와 변환 $X$ 로 구성된다.

$ p_\theta(c|z) $ 는 latent token으로부터 segmentation을 복원하는 decoder이다. 실제로는 pretrained VQVAE decoder $D_\theta$ 와 역변환 $X^{-1}$ 로 구성된다.

여기서 핵심 난점은 segmentation mask가 자연 이미지와 형식이 다르다는 점이다. 저자들은 이를 해결하기 위해 $X: \mathbb{R}^K \to \mathbb{R}^3$ 라는 변환을 둔다. 각 클래스 one-hot vector를 RGB color vector로 보내는 것이다. 이렇게 만들어진 $x(c)=X(c)$ 가 바로 maskige다. 그러면 pretrained image VQVAE가 maskige reconstruction에 사용될 수 있다.

1단계 학습의 원래 목적은 대략 다음과 같다.

$$
\min_{\theta,\phi} E_{q_\phi(z|c)} \|p_\theta(c|z)-c\|
$$

그런데 이를 직접 최적화하는 대신, mask를 maskige로 바꿔 image reconstruction 문제로 우회한다. 논문은 이를 변형해 pretrained VQVAE가 담당하는 부분과, 새로 학습해야 하는 부분을 분리한다. 결과적으로 실제 핵심 최적화는 다음처럼 정리된다.

$$
\min_{X^{-1}} E_{q_{\hat{\phi}}(\hat{z}|X(c))} \|X^{-1}(\hat{x}(c)) - c\|
$$

즉 pretrained decoder가 만들어낸 maskige $\hat{x}(c)$ 를 다시 segmentation mask로 잘 되돌리는 $X^{-1}$ 를 잘 설계하면 된다. 이 덕분에 거대한 VQVAE 전체를 다시 학습하지 않아도 된다.

논문은 $X$ 와 $X^{-1}$ 설계를 여러 방식으로 나눈다.

- **GSS-FF**: $X$ 와 $X^{-1}$ 모두 training-free인 선형 설계
- **GSS-FT**: $X$ 는 고정, $X^{-1}$ 는 학습
- **GSS-TF**: $X$ 는 학습, $X^{-1}$ 는 선형 least-squares
- **GSS-TT**: 둘 다 학습

선형 설계에서 $x(c)=c\beta$ 로 두고, $\beta \in \mathbb{R}^{K \times 3}$ 는 각 클래스를 어떤 RGB color로 보낼지를 정한다. 역변환은 pseudo-inverse 형태로

$$
\beta^\dagger = \beta^\top(\beta\beta^\top)^{-1}
$$

를 사용한다. 이 경우 $X^{-1}$ 는 별도 학습 없이 계산된다. 저자들은 클래스별 색이 최대한 멀어지도록 배치하는 **maximal distance assumption** 을 제안한다. 직관적으로는 class color들이 RGB 공간에서 서로 충분히 떨어져 있어야 복원이 쉬워진다는 뜻이다.

학습이 필요한 변형에서는 hard Gumbel-softmax를 사용해 이산 latent token에 대해 gradient를 흘린다. 특히 GSS-FT-W는 $X^{-1}$ 에 shifted-window Transformer block을 써서 가장 강한 복원 성능을 얻는다.

2단계인 latent prior learning에서는 image encoder $I_\psi$ 를 학습해 이미지로부터 latent token 분포를 예측한다. 목적은

$$
\min_\psi D_{KL}(q_\phi(z|c), p_\psi(z|x))
$$

이다. $z$ 가 이산 codebook index이므로, 실제로는 ground-truth posterior token 분포와 image encoder 출력 사이의 cross-entropy로 정렬한다.

이 단계에서 논문은 **unlabeled area auxiliary** 를 추가한다. 일부 dataset에서는 unlabeled pixel이 있는데, discriminative segmentation은 그 픽셀만 ignore하면 되지만, generative latent token 수준에서는 부분 ignore가 쉽지 않다. 그래서 auxiliary head $p_\xi(\bar{c}|z)$ 를 두어 unlabeled 영역에 pseudo label을 채우고, 보강된 mask $\tilde{c}$ 로 학습한다. 최종 목적은 논문 기준으로 다음처럼 적혀 있다.

$$
\min_\psi D_{KL}(q_\phi(z|\tilde{c}), p_\psi(z|x)) + p_\xi(\bar{c}|z)
$$

다만 마지막 보조항의 정확한 수식 형태는 본문에 매우 간략히 적혀 있어, 이것이 어떤 loss 형태로 더해지는지까지는 제공된 텍스트만으로는 명확하지 않다.

추론 과정은 단순하다. 이미지 $x$ 를 image encoder에 넣어 latent token $z$ 를 예측하고, 이를 VQVAE decoder에 넣어 predicted maskige $\hat{x}(c)$ 를 만든다. 마지막으로 $X^{-1}$ 를 적용해 segmentation mask $\hat{c}$ 를 얻는다.

## 4. 실험 및 결과

실험은 세 데이터셋 축에서 진행된다. 단일 도메인 benchmark로는 **Cityscapes** 와 **ADE20K**, cross-domain benchmark로는 **MSeg** 를 사용한다. Cityscapes는 19개 클래스의 urban scene segmentation, ADE20K는 150개 클래스의 scene parsing, MSeg는 여러 도메인을 통합한 대규모 multi-domain segmentation benchmark다.

평가 지표는 기본적으로 **mIoU** 와 **mAcc** 이다. 구현은 MMSegmentation 기반이며, latent prior learning은 Cityscapes 80k iteration, ADE20K와 MSeg는 160k iteration으로 수행된다.

가장 먼저 중요한 것은 **latent posterior learning ablation** 이다. Table 1에 따르면 ADE20K validation에서 reconstruction mIoU는 GSS-FF-R 62.83, GSS-FF 84.31, GSS-FT 86.10, GSS-TF 84.37, GSS-TT 36.11, GSS-FT-W 87.73이다. 여기서 두 점이 두드러진다. 첫째, random color initialization인 GSS-FF-R은 성능이 크게 떨어진다. 둘째, joint optimization인 GSS-TT는 오히려 붕괴에 가깝다. 반면 hand-crafted color design인 GSS-FF는 학습 비용 0으로도 강한 성능을 내고, GSS-FT-W는 더 비싸지만 가장 높은 reconstruction 정확도를 낸다. 이는 maskige 설계와 $X^{-1}$ 설계가 매우 중요하다는 뜻이다.

**VQVAE 설계 비교** 에서는 DALL·E-style pretrained VQVAE + maskige 조합이 가장 좋다. Table 2에서 Cityscapes/ADE20K reconstruction mIoU는 DALL·E-style이 각각 95.17/87.73으로 최고이며, UViM-style보다 정확도와 학습 비용 모두 유리하다. 논문은 이를 통해 segmentation mask를 직접 $K$-channel로 다루는 것보다, maskige를 통해 image-pretrained representation을 빌려오는 전략이 훨씬 실용적이라고 주장한다.

**latent prior learning ablation** 에서는 unlabeled area auxiliary와 MLA의 효과가 확인된다. ADE20K validation에서 baseline은 mIoU 40.64인데, unlabeled auxiliary를 넣으면 43.72로 약 3.1%p 오른다. 해상도 비율을 $1/8$ 에서 $1/4$ 로 높이면 43.98로 소폭 증가한다. 여기에 MLA까지 추가하면 46.29가 된다. 즉 second-stage 성능은 단순히 latent를 예측하는 것만으로 충분하지 않고, multi-scale image aggregation과 unlabeled region 처리가 꽤 중요하다.

**단일 도메인 semantic segmentation** 결과를 보면, Cityscapes에서는 GSS-FT-W가 ResNet-101 기준 78.46, Swin-Large 기준 80.05 mIoU를 기록한다. 이는 최상위 discriminative model에는 못 미치지만, MaskFormer나 SETR 등과 비교해 경쟁력 있는 수준이다. ADE20K에서는 GSS-FT-W가 Swin-Large 기준 48.54 mIoU로, UViM 재현 결과 43.71보다 훨씬 높고 SETR 48.28과 비슷하거나 약간 높다. 즉 generative segmentation으로서 성능이 충분히 현실적인 수준까지 올라왔음을 보여준다.

논문이 특히 강조하는 부분은 **cross-domain semantic segmentation** 이다. MSeg test에서 GSS-FT-W는 HRNet-W48 기준 harmonic mean 55.2, Swin-Large 기준 61.9를 기록한다. 이는 재현된 MSeg baseline의 54.9, 61.7보다 높고, 원 논문 표의 다른 경쟁 방법들보다도 우수하다. VOC, Pascal Context, CamVid, WildDash, KITTI, ScanNet 전반에서 균형 있게 좋은 성능을 내는 점이 중요하다. 저자들은 이것을 generative learning이 discriminative learning보다 더 domain-generic representation을 만들 가능성의 근거로 해석한다.

또 하나 흥미로운 결과는 **domain-generic maskige** 이다. MSeg에서 만든 maskige를 Cityscapes에 그대로 옮겨도 80.5에서 79.5로 약 1%p만 감소한다. 반면 image encoder를 공유하면 성능 하락이 더 크다. 이 결과는 maskige가 시각적 appearance와 독립적이어서 데이터셋 간 전이성이 높다는 저자들의 주장을 뒷받침한다.

정성적 결과에서도 GSS는 edge가 선명하고 작은 물체를 비교적 잘 복원한다. Figure 5에서 reconstruction 품질은 UViM이나 VQGAN보다 깔끔하며, Figure 6 이후 예시에서는 가는 pole, 먼 pedestrian, 실내 가구 경계 같은 세밀한 구조를 비교적 잘 분할하는 모습이 제시된다. 물론 이것은 논문이 선택한 사례 기반 설명이며, 전체 분포에 대한 정량적 보장은 표의 성능 수치가 담당한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 재정의 자체가 명확하고 일관되다** 는 점이다. semantic segmentation을 discriminative classification이 아니라 conditional generation으로 본다는 관점은 단순한 수사에 그치지 않고, ELBO, latent posterior/prior, maskige, two-stage optimization이라는 구체적 설계로 연결된다.

둘째 강점은 **pretrained generative model 재사용 전략이 실용적** 이라는 점이다. segmentation 전용 VQVAE를 처음부터 학습하지 않고, maskige를 도입해 DALL·E-style VQVAE를 그대로 활용한다. 이 덕분에 첫 단계 학습 비용을 크게 줄이고, UViM 대비 더 좋은 reconstruction과 더 낮은 비용을 동시에 달성했다는 실험 결과가 있다.

셋째 강점은 **cross-domain generalization** 이다. 단일 도메인 최고 성능은 아니지만, MSeg benchmark에서 SOTA 수준의 결과를 제시했다. 실제 적용에서는 도메인 이동이 매우 흔하므로, 이 특성은 의미가 크다.

반면 한계도 논문 안에 비교적 솔직하게 적혀 있다. 첫째, 최고 discriminative model 대비 절대 성능은 아직 낮다. 특히 object edge precision이 discriminative model보다 떨어질 수 있다고 저자들이 직접 인정한다. 둘째, generative model은 전체 데이터 분포를 학습해야 해서 더 많은 데이터가 필요하다. 저자들은 MSeg처럼 더 큰 데이터셋에서 성능이 상대적으로 더 좋은 점을 이 근거로 해석한다. 셋째, maskige는 결국 RGB 3차원 color space에 클래스를 배치하는 방식이라, 클래스 수가 많아질수록 color crowding 문제가 생긴다. 이 경우 $X^{-1}$ 가 유사 색을 혼동할 수 있고, 특히 경계 부근에서 오류를 유발할 수 있다.

비판적으로 보면, 이 논문은 “semantic segmentation을 truly generative하게 했다”기보다, **segmentation mask를 discrete latent image generation 틀에 잘 맞게 embedding한 방법** 에 가깝다. 이것은 장점이기도 하지만, 동시에 생성 모델의 일반성에 크게 기대는 구조다. 또한 unlabeled auxiliary의 정확한 학습식이나 pseudo label 품질이 최종 성능에 어떤 영향을 주는지는 제공된 본문만으로는 충분히 세부적으로 드러나지 않는다. 그리고 single-domain 성능이 최상위 discriminative method를 넘지 못한다는 점은, 이 접근이 당장 표준 segmentation 파이프라인을 대체한다기보다 특정 장점, 특히 domain generalization 측면에서 더 설득력이 있음을 뜻한다.

## 6. 결론

이 논문은 semantic segmentation을 image-conditioned mask generation으로 재정의하고, 이를 위해 **maskige**, **latent posterior/prior learning**, **pretrained VQVAE 재사용** 이라는 조합을 제안했다. 기술적으로는 segmentation mask를 RGB-like 표현으로 바꿔 generative image model의 표현 공간에 올려놓고, 이미지에서 해당 latent token을 예측하도록 학습하는 방식이다.

주요 기여는 세 가지로 요약할 수 있다. 첫째, semantic segmentation에 대한 generative formulation을 제시했다. 둘째, off-the-shelf generative model을 거의 그대로 활용할 수 있는 효율적 구현을 보였다. 셋째, 단일 도메인에서는 경쟁력 있는 결과를, cross-domain에서는 더 강한 일반화를 실험적으로 보였다.

실제 적용 관점에서는, 여러 도메인에 걸친 segmentation 시스템이나 대규모 사전학습 생성 모델의 표현을 활용하려는 환경에서 특히 의미가 있다. 향후 연구로는 논문이 제시하듯 instance-level segmentation, 더 높은 차원의 mask representation, 그리고 여러 vision task를 latent prior learning 관점에서 통합하는 unified model 방향이 자연스럽다. 전체적으로 이 논문은 segmentation 분야에서 generative modeling을 진지한 대안으로 끌어올린 시도라고 볼 수 있다.
