# Generative AI for Medical Imaging: extending the MONAI Framework

## 1. Paper Overview

이 논문은 의료영상 분야에서 빠르게 확산되고 있는 **generative AI** 모델들을 보다 쉽게 **학습, 재현, 평가, 배포**할 수 있도록, 기존 의료 AI 프레임워크인 **MONAI**를 확장한 **MONAI Generative Models** 플랫폼을 제안한다. 저자들은 diffusion model, autoregressive transformer, GAN, VAE 등 의료영상 생성에 쓰이는 모델들이 점점 강력해지고 있지만, 모델 구조와 학습 절차, 평가 지표, 구현 세부사항이 너무 복잡해 **재현성(reproducibility)** 과 **비교 가능성**이 떨어진다는 문제를 지적한다. 이에 따라 이 논문은 다양한 생성 모델과 응용을 공통 인터페이스와 모듈형 구성으로 묶어, 연구자와 개발자가 일관된 방식으로 실험할 수 있게 하는 오픈소스 플랫폼을 소개한다.  

이 문제가 중요한 이유는 의료영상이 본질적으로 **고차원 데이터**이고, 동시에 **개인정보 보호** 이슈 때문에 대규모 원본 데이터 공유가 어렵기 때문이다. 생성 모델은 synthetic data 공유, anomaly detection, image-to-image translation, denoising, MRI reconstruction 등에서 큰 잠재력을 가지지만, 실제 연구 현장에서는 구현 난이도와 실험 재현의 어려움이 진입장벽으로 작용한다. 이 논문은 바로 그 장벽을 낮추는 **engineering framework paper**라는 점에서 의미가 크다.

## 2. Core Idea

이 논문의 핵심 아이디어는 간단하다.

**의료영상용 generative model 연구를 개별 논문별 구현의 집합으로 두지 말고, 표준화된 공통 플랫폼 위에서 모듈화해 재사용 가능하게 만들자.**

즉, 논문의 novelty는 완전히 새로운 생성 모델을 하나 제안하는 데 있지 않다. 대신 다음을 하나의 프레임워크로 묶는다.

* diffusion model 계열
* latent diffusion model 계열
* autoregressive transformer 계열
* GAN/SPADE 계열
* generative model용 metric/loss/evaluation 도구
* 2D/3D 의료영상 전반에 대한 확장성

특히 저자들은 이 플랫폼이 단순한 코드 저장소가 아니라, **state-of-the-art 연구들을 표준화된 방식으로 재현**하고, **사전학습 모델(pretrained models)** 을 제공하며, **2D와 3D, CT/MRI/X-ray 등 다양한 modality** 로 일반화 가능하도록 설계되었다고 강조한다. 이 점에서 공헌은 “새 모델”보다는 **reproducible generative medical imaging ecosystem** 구축에 가깝다.

## 3. Detailed Method Explanation

### 3.1 전체 구조

논문이 제안하는 MONAI Generative Models는 기존 MONAI 위에 구축된 확장 패키지다. 목표는 의료영상용 생성 모델에 필요한 핵심 구성요소를 통합하는 것이다. 논문이 소개하는 기능은 크게 두 축으로 나뉜다.

1. **Models**
2. **Metrics and Losses**

즉, 단순히 네트워크만 제공하는 것이 아니라, 생성 모델 학습에 필요한 평가 지표와 loss까지 함께 넣어 실제 연구 워크플로우 전체를 지원한다.

### 3.2 Diffusion Model 지원

논문은 diffusion model을 의료영상에서 매우 효과적인 생성 모델로 소개한다. 기본 아이디어는 초기 Gaussian noise 이미지에서 시작해 점진적으로 denoising을 반복하며 샘플을 생성하는 것이다. 수식 수준에서 보면 시작점은 대체로

$x_T \sim \mathcal{N}(0, I)$

이며, 역과정에서 신경망이 각 timestep의 noise를 예측해 더 깨끗한 샘플 $x_{t-1}$ 을 만들어 간다. 논문은 이 denoising network를 $\theta(x_t, t)$ 형태로 설명한다.

이를 위해 프레임워크는 다음 핵심 구성요소를 제공한다.

* `DiffusionModelUNet`
  timestep conditioning이 residual connection으로 들어가며, spatial transformer layer를 통해 text나 numerical score 같은 외부 조건(conditioning)도 받을 수 있는 UNet 기반 구조다.
* `Scheduler`
  diffusion sampling에서 noise schedule을 관리하는 공통 인터페이스다.
* `DDIMScheduler`, `PNDMScheduler`
  서로 다른 샘플링 스케줄러 구현체다.
* noise profile
  `linear`, `scaled linear`, `cosine` 등 다양한 noise scheduling profile을 모듈 방식으로 확장 가능하게 구현했다.

이 설계의 장점은 diffusion 연구에서 자주 바뀌는 요소인 **UNet backbone, scheduler, conditioning 방식, noise schedule** 을 서로 독립적으로 조합할 수 있다는 점이다.

### 3.3 Latent Diffusion Model과 Compression Model

논문은 의료영상처럼 2D보다 큰 3D 고해상도 데이터에서 vanilla diffusion을 직접 돌리는 것이 비효율적이라는 점을 의식해 **Latent Diffusion Model(LDM)** 을 지원한다. LDM은 픽셀 공간이 아니라 compression model이 학습한 latent space에서 diffusion을 수행하므로, 훨씬 더 큰 2D/3D 데이터를 다룰 수 있다.

이를 위해 프레임워크는 두 종류의 compression model을 제공한다.

* `AutoencoderKL`
* `VQVAE`

즉, KL-regularized autoencoder와 VQ-VAE를 latent representation 학습에 사용할 수 있게 하고, 이를 diffusion backbone과 교체 가능하게 연결한다. 이 구조 덕분에 동일한 LDM 파이프라인을 유지하면서도 데이터 특성에 따라 latent encoder 방식을 바꿀 수 있다.

### 3.4 Diffusion 확장: Encoder와 ControlNet

논문은 diffusion backbone을 다양한 downstream generative task에 확장한다.

* `DiffusionModelEncoder`
  UNet 일부를 encoder처럼 사용해 입력 영상의 latent representation을 얻는 구성이다. 논문은 anomaly detection 계열 접근을 지원하기 위해 이를 포함했다고 설명한다.
* `ControlNet`
  diffusion UNet에 adapter를 연결해 특정 conditioning을 더 강하게 반영하도록 하는 구조다. 논문은 image translation task에서 강한 성능을 보인다고 설명한다.

즉, 이 프레임워크는 “샘플 생성”만이 아니라, **조건부 생성**, **translation**, **anomaly detection** 같은 파생 응용까지 염두에 두고 설계되었다.

### 3.5 Autoregressive Transformer 지원

논문은 생성 모델의 또 다른 축으로 **autoregressive transformer** 를 제공한다. 이 계열은 joint distribution을 조건부 확률의 곱으로 분해해 모델링한다.

$$
p(x)=\prod_{i=1}^{n} p(x_i \mid x_1, \ldots, x_{i-1})
$$

이 구조는 sequence 모델링에는 자연스럽지만, 이미지처럼 2D/3D 구조 데이터를 쓰려면 먼저 이를 1D sequence로 바꿔야 한다. 이를 위해 프레임워크는 `Ordering` 클래스를 제공하며, raster scan, S-curve, random ordering 같은 여러 ordering 방식을 지원한다. 또한 latent VQ-VAE의 discrete codebook index를 token처럼 써서 transformer 입력으로 삼을 수 있게 한다.

이 부분의 의미는 의료영상 생성이 diffusion 일변도가 아니라, **likelihood-based transformer generation** 도 같은 프레임워크에서 실험 가능하다는 데 있다.

### 3.6 GAN/SPADE와 Adversarial Component

논문은 GAN 계열 생성도 배제하지 않는다. 특히 **SPADE** 구성요소를 포함한다. SPADE는 semantic layout을 입력으로 받아 spatially adaptive normalization을 수행해, 생성기가 부위별 특성을 더 정교하게 반영하도록 만드는 구조다. 논문은 이 방식이 image-to-image translation에서 효과적이라고 설명한다.

또한 adversarial training을 위해 다음도 제공한다.

* `PatchDiscriminator`
* `MultiScalePatchDiscriminator`

이는 Pix2Pix, Pix2PixHD 계열의 patch-based 판별기 구조를 의료영상 생성 모델 학습에 붙일 수 있도록 만든 것이다. 논문은 이들이 SPADE뿐 아니라 VQ-GAN류 compression model 학습에도 활용될 수 있다고 설명한다.

### 3.7 Metrics와 Losses

생성 모델에서 중요한 것은 “좋아 보이는 샘플”이 아니라 **정량적 평가 가능성**이다. 이를 위해 논문은 다음 metric을 프레임워크에 넣는다.

* **FID**
  synthetic image distribution과 real image distribution의 차이를 보는 fidelity metric
* **MMD**
  분포 유사도를 보는 또 다른 metric
* **MS-SSIM**
  synthetic sample 사이의 유사도를 통해 diversity를 평가하는 metric

또한 학습용 loss로 다음을 제공한다.

* spectral loss
* patch-based adversarial loss
* perceptual loss

특히 perceptual loss에서 저자들은 일반 ImageNet backbone뿐 아니라 **RadImageNet** 과 **MedicalNet** 같은 의료 특화 사전학습 네트워크를 지원한다고 설명한다. 3D perceptual loss는 메모리 문제 때문에 2.5D 방식으로 계산하는 아이디어도 포함한다. 이 부분은 매우 실용적인 engineering contribution이다.

## 4. Experiments and Findings

### 4.1 실험의 목적

논문은 플랫폼의 유연성과 범용성을 보이기 위해 여러 실험을 수행했다고 밝힌다. 서론에서는 총 **다섯 개 실험**을 언급하며, out-of-distribution detection, image translation, super-resolution 등 다양한 응용을 다뤘다고 설명한다. 다만 현재 첨부 파일에서 전체 본문이 길게 잘려 있어, 후반 실험들의 세부 서술은 부분적으로만 확인된다. 따라서 아래 정리는 **본문에서 명확히 확인 가능한 실험 1과 서론/부록에서 확인 가능한 범위**를 중심으로 한다.  

### 4.2 Experiment I: 다양한 의료영상 유형에 대한 적응성

가장 자세히 확인되는 실험은 **Latent Diffusion Model의 modality/해부학적 부위 전반 적응성**을 평가한 실험이다. 저자들은 동일 프레임워크를 사용해 여러 데이터셋과 차원에서 LDM을 학습했다. 사용 데이터는 다음과 같다.

* **MIMIC-CXR**: 96,161장의 2D chest X-ray, $512 \times 512$
* **CSAW-M**: 9,523장의 2D mammography, $640 \times 512$
* **UK Biobank**: 41,162장의 3D T1 brain MRI, $160 \times 224 \times 160$
* **UK Biobank 2D slices**: 360,525장의 2D brain slice, $160 \times 224$
* **Retinal OCT**: 84,483장의 2D OCT, $512 \times 512$

평가는 다음 기준으로 이뤄진다.

* autoencoder reconstruction 품질: **MS-SSIM Recons.**
* diffusion sample fidelity: **FID**
* generated sample diversity: **MS-SSIM**

논문 표 1의 주요 수치는 다음과 같다.

* **MIMIC-CXR**: FID 8.8325, MS-SSIM 0.4276, Recons. 0.9834
* **CSAW-M**: FID 1.9061, MS-SSIM 0.5356, Recons. 0.9789
* **Retinal OCT**: FID 2.2501, MS-SSIM 0.3593, Recons. 0.8966
* **UK Biobank 2D**: FID 2.1986, MS-SSIM 0.5368, Recons. 0.9876
* **UK Biobank 3D**: FID 0.0051, MS-SSIM 0.9217, Recons. 0.9820

이 결과가 보여주는 바는, 프레임워크가 특정 modality에 고정된 것이 아니라 **X-ray, mammography, OCT, 2D/3D MRI** 를 모두 처리할 수 있고, LDM 구현이 다양한 환경에서 안정적으로 작동한다는 점이다. 특히 저자들은 Figure 1과 함께 synthetic sample이 전반적으로 고품질이라고 주장한다.

### 4.3 확인 가능한 추가 응용 방향

서론과 부록 일부로부터, 프레임워크가 다음 응용을 지원한다는 점은 확인된다.

* **out-of-distribution / anomaly detection**
* **image translation**
* **super-resolution**
* **ControlNet 기반 조건부 생성**
* **3D low-resolution MRI의 super-resolution**  

부록 스니펫에는 ControlNet 샘플, 3D UK Biobank 저해상도 LDM 샘플과 super-resolved 결과 그림이 언급된다. 다만 현재 확보된 본문 텍스트만으로는 후속 실험들의 데이터셋, baseline, 세부 수치까지는 완전하게 복원되지 않는다. 따라서 이 논문을 엄밀히 해석하면, **플랫폼의 적용 범위는 넓게 제시되지만, 지금 확인 가능한 정량 비교는 실험 1이 가장 핵심적**이다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 **새로운 생성 모델 하나를 제안하는 대신, 의료영상 생성 연구 전반을 위한 공통 인프라를 구축했다는 점**이다. 저자들은 의료 generative model 연구가 빠르게 확산되는 반면, 모델 복잡성, 구현 난이도, 평가 지표의 난립 때문에 재현성과 비교 가능성이 떨어진다고 진단하고, 이를 해결하기 위해 MONAI 위에 표준화된 generative framework를 올렸다. 이 문제 설정 자체가 매우 실용적이며, 실제 연구 생산성을 높이는 방향의 공헌이다.  

둘째, 프레임워크의 **모듈성과 범용성**이 강점이다. 논문은 diffusion, latent diffusion, autoregressive transformer, SPADE/GAN 계열을 한 패키지 안에서 지원하고, `Scheduler`, `DiffusionModelUNet`, `AutoencoderKL`, `VQVAE`, `Ordering`, `PatchDiscriminator`, `MultiScalePatchDiscriminator` 같은 구성요소를 조합 가능한 형태로 제공한다. 이는 의료영상 생성 연구를 “논문별 일회성 코드”가 아니라 “재사용 가능한 building blocks”로 전환한다는 점에서 가치가 크다.  

셋째, **2D와 3D를 함께 다룬다는 점**이 특히 중요하다. 의료영상은 자연영상과 달리 3D 볼륨 데이터가 많고, 이 때문에 많은 공개 구현이 사실상 2D에 머무르는 경우가 많다. 그런데 이 논문은 MONAI의 확장성을 활용해 3D까지 자연스럽게 포함하고, 다양한 modality와 anatomical area로 결과를 확장할 수 있음을 강조한다. 이는 의료영상 분야에서 매우 실질적인 장점이다.  

넷째, **metrics와 losses까지 함께 패키징**한 점도 강하다. 단순히 네트워크만 제공하는 것이 아니라 FID, MMD, MS-SSIM, spectral loss, adversarial loss, perceptual loss를 포함하고, 특히 RadImageNet·MedicalNet 같은 의료 특화 pretrained backbone과 3D에서의 2.5D perceptual loss 전략까지 지원한다. 이는 “생성 모델 학습 전체 파이프라인”을 다루는 프레임워크라는 점을 분명히 보여준다.

### Limitations

한계도 분명하다.

첫째, 이 논문은 기본적으로 **framework paper**이기 때문에, 특정 생성 모델의 알고리즘적 우월성을 엄밀히 입증하는 논문은 아니다. 즉, “이 프레임워크가 있으니 어떤 새 모델이 항상 더 좋다”를 보여주는 것이 아니라, 실험 예시를 통해 플랫폼의 유용성과 유연성을 보여주는 데 초점이 있다. 따라서 이 논문을 읽을 때는 SoTA model paper가 아니라 **infrastructure / standardization paper**로 해석해야 한다.  

둘째, 실험 범위는 넓지만, 현재 논문에서 가장 강하게 제시되는 정량 결과는 주로 **Experiment I의 adaptability demonstration**에 집중되어 있다. 논문은 out-of-distribution detection, image translation, super-resolution 등 다섯 실험을 언급하지만, 각 응용에 대해 모두 동일한 수준의 정량 비교와 baseline 분석을 깊게 제공하는 구조는 아니다. 따라서 프레임워크의 포괄성은 잘 드러나지만, 각 개별 태스크에서의 “최적 사용법”이나 “성능 한계”는 후속 연구가 더 보완해야 한다.  

셋째, framework가 강력할수록 반대로 **설계 공간이 넓어져 사용자 선택 부담이 커질 수 있다**. 예를 들어 scheduler, noise profile, compression model, perceptual backbone, discriminator 구성을 어떻게 조합하느냐에 따라 결과가 달라질 수 있다. 논문은 이를 유연성으로 제시하지만, 초보 사용자에게는 여전히 상당한 실험 설계 부담이 남는다. 이 점은 논문이 직접 한계라고 길게 쓰지는 않지만, 모듈형 시스템의 구조적 trade-off로 읽을 수 있다.  

넷째, 저자들 스스로도 future work로 **더 최근 모델의 통합과 MRI reconstruction 같은 추가 응용 지원 확대**를 언급한다. 이는 현재 프레임워크가 충분히 유용하지만, 의료 generative model 생태계의 빠른 발전 속도를 따라가기 위해 지속적 확장이 필요하다는 뜻이기도 하다.

### Brief Critical Interpretation

비판적으로 보면, 이 논문의 진짜 공헌은 “의료영상 generative AI를 위한 최고 성능 모델”이 아니라, **의료 생성 모델 연구를 공학적으로 정돈하는 표준 계층**을 만든 데 있다. 즉, 이 논문은 모델 자체보다도 **research workflow의 표준화**에 더 큰 의미가 있다.

이 관점은 매우 중요하다. 의료영상 분야에서는 데이터 접근 제한, 3D 처리 비용, 평가 방식의 불일치 때문에 “좋은 아이디어”만으로는 연구가 확장되기 어렵다. MONAI Generative Models는 이 병목을 줄이기 위해 공통 컴포넌트, 공통 메트릭, 공통 학습 도구를 제공한다. 따라서 이 논문은 algorithm paper라기보다, **medical generative AI의 재현성과 확장성을 끌어올리는 enabling paper**로 이해하는 것이 가장 정확하다.  

## 6. Conclusion

이 논문은 **MONAI Generative Models**라는 오픈소스 프레임워크를 제안해, 의료영상 생성 모델의 개발·학습·평가·배포를 더 쉽고 일관되게 수행할 수 있도록 했다. 핵심 기여는 diffusion, latent diffusion, autoregressive transformer, GAN/SPADE 등 다양한 생성 모델 계열을 MONAI 생태계 안에서 통합하고, 여기에 metrics, losses, pretrained support, 2D/3D 확장성을 함께 제공한 점이다.  

저자들의 결론도 분명하다. MONAI Generative Models는 image synthesis뿐 아니라 다양한 응용에서 연구자가 드는 시간과 노력을 줄여 주며, 앞으로는 더 최신 모델을 통합하고 MRI reconstruction 같은 추가 응용까지 지원 범위를 넓히겠다고 한다. 따라서 이 논문은 단순한 한 편의 모델 논문이 아니라, **의료영상 generative AI 연구를 위한 장기적 기반 인프라**를 제시한 작업으로 평가할 수 있다.
