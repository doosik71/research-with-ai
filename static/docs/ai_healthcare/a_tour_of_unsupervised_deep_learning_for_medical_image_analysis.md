# A Tour of Unsupervised Deep Learning for Medical Image Analysis

이 논문은 의료영상 분석에서 supervised learning이 가진 라벨 의존성, 수작업 주석 비용, supervision bias 문제를 넘어, **unsupervised deep learning이 어떤 방식으로 의료영상 표현 학습과 분석에 활용되어 왔는지 체계적으로 정리한 review paper**다. 초록과 서론에 따르면 저자들은 autoencoder 계열, Restricted Boltzmann Machine(RBM), Deep Belief Network(DBN), Deep Boltzmann Machine(DBM), Generative Adversarial Network(GAN)를 중심으로, 이들 모델이 의료영상 분류, 분할, 특징학습, 합성, 검색 등에 어떻게 쓰였는지를 폭넓게 정리한다. 또한 software tools, benchmark datasets, future opportunities와 challenges까지 한 번에 제공하는 “tour” 형식의 문헌 리뷰를 목표로 한다.

## 1. Paper Overview

논문의 문제의식은 분명하다. MRI, PET, CT, mammography, ultrasound, X-ray, digital pathology 같은 의료영상은 진단과 치료에서 핵심적이지만, 실제 임상에서는 여전히 인간 전문가의 판독에 크게 의존한다. 그러나 병변의 다양성, 판독 피로도, 데이터의 고차원성과 이질성 때문에 자동화된 computer-aided diagnosis(CAD)의 필요성이 커지고 있다. 저자들은 이런 상황에서 deep learning이 큰 가능성을 보였지만, supervised learning만으로는 충분하지 않다고 본다. 많은 상황에서 라벨이 없거나 부족하거나 편향되어 있기 때문이다.

그래서 이 논문은 “의료영상 분석에서 unsupervised deep learning은 어떤 역할을 할 수 있는가?”를 정면으로 다룬다. 1쪽과 2쪽의 설명을 따르면, unsupervised learning은 단지 라벨이 없는 상황의 대체제가 아니라, 데이터에서 직접 구조를 발견하고, 표현을 학습하고, clustering, compression, dimensionality reduction, denoising, super-resolution, decision support 등 다양한 하위 문제를 수행할 수 있는 기반 기술로 제시된다. 저자들은 특히 unsupervised learning이 supervised learning의 preprocessing 또는 representation learning 단계로도 유용할 수 있다고 본다.

이 문제가 중요한 이유는 의료영상 데이터의 실제 특성 때문이다. 의료영상은 대체로 고차원이며, acquisition protocol과 기관, 장비, 환자 상태에 따라 큰 이질성을 가진다. 또한 충분한 expert label을 확보하기 어렵다. 따라서 대규모 unlabeled data를 활용할 수 있는 unsupervised 방법은 의료영상 AI의 확장성과 실용성 측면에서 매우 중요하다. 이 논문은 바로 그 맥락에서 2018년 시점까지의 주요 비지도 딥러닝 계열을 일람한다.

## 2. Core Idea

이 논문의 핵심 아이디어는 의료영상 비지도 딥러닝 문헌을 개별 논문 나열이 아니라 **모델 family 중심의 taxonomy**로 재구성하는 데 있다. 8쪽 Figure 4에 따르면 저자들은 unsupervised deep learning 모델을 크게 다음 다섯 계열로 정리한다.

* Autoencoders and variants
* Restricted Boltzmann Machines
* Deep Belief Networks
* Deep Boltzmann Machines
* Generative Adversarial Networks

이 다섯 축이 논문의 중심 구조다.

또 다른 핵심 아이디어는 unsupervised learning을 단순 “라벨 없는 학습”으로 보지 않고, **의료영상 표현 학습의 기반 인프라**로 보는 것이다. 2쪽 “Why Unsupervised Learning?”에서는 supervised workflow가 수작업 라벨 비용, supervision bias, scalability 한계를 가진다고 지적한다. 반면 unsupervised learning은 데이터 자체로부터 구조를 학습하고, 나중의 분류나 진단 과제에 더 잘 일반화되는 representation을 제공할 수 있다고 설명한다. 즉 이 논문은 unsupervised learning을 standalone 분석 기법이면서 동시에 supervised CAD를 위한 representation pretraining 도구로도 위치시킨다.

엄밀히 말하면 이 논문의 novelty는 새로운 모델 제안이 아니다. 진짜 공헌은 다음 세 가지에 있다.

첫째, 의료영상 분석에서 비지도 딥러닝 모델군을 한 번에 정리했다는 점.
둘째, 각 모델의 수학적·구조적 개요와 의료 응용 사례를 연결했다는 점.
셋째, 소프트웨어 도구와 벤치마크 데이터셋, 그리고 향후 과제까지 함께 정리해 field overview를 제공했다는 점.

## 3. Detailed Method Explanation

이 논문은 review paper이므로 여기서의 “방법”은 저자들이 소개하는 각 모델 계열의 원리와 그 의료영상 적용 맥락을 설명하는 것이다.

### 3.1 비지도 학습의 기본 과제

3장에서는 unsupervised learning task를 먼저 세 가지 큰 범주로 정리한다.

* density estimation
* dimensionality reduction
* clustering

예를 들어 density estimation은 데이터의 내재 구조를 확률적 혹은 비확률적으로 추정하는 문제로 소개되고, dimensionality reduction은 고차원 영상 데이터에서 중요한 정보만 남기고 차원을 줄이는 과정으로 설명된다. 저자들은 X-ray, CT, MRI처럼 고차원 영상이 누적되는 상황에서 PCA, Kernel PCA 등이 왜 중요한지를 예시와 함께 설명한다. clustering은 unlabeled data를 유사한 그룹으로 묶는 가장 전형적인 비지도 과제로, hierarchical clustering과 partitional clustering(k-means)을 간단히 소개한다. 이 부분은 이후 deep unsupervised model이 어떤 문제를 해결하려는지 보여주는 배경 역할을 한다.

### 3.2 Autoencoders와 변형들

논문에서 가장 비중 있게 다뤄지는 계열은 autoencoder다. 8~12쪽은 basic autoencoder와 여러 변형을 연속적으로 소개한다.

기본 autoencoder는 입력을 latent representation으로 압축하는 encoder와, 다시 복원하는 decoder로 구성된다. 학습 목표는 입력과 복원 간 reconstruction error를 최소화하는 것이다. 논문은 encoder를 $h = f_\theta(x)$, decoder를 $g_\theta(h)$로 설명하고, 전체 목적은 reconstruction loss 최소화라고 정리한다. 이 구조는 의료영상에서 feature extraction, representation learning, pretraining에 널리 쓰였다고 설명한다.

그 위에 여러 변형이 추가된다.

**Stacked Autoencoder (SAE)**는 여러 autoencoder를 층층이 쌓아 deep representation을 학습한다. greedy layer-wise pretraining을 통해 더 높은 표현력을 얻는 구조로 설명된다. 의료영상에서는 Alzheimer’s disease 분류, hippocampus segmentation, lesion classification, organ detection, right ventricle segmentation 등에 적용된 사례가 Table 2에 정리되어 있다.

**Denoising Autoencoder (DAE)**는 입력을 일부러 오염시킨 뒤 원래 clean input을 복원하도록 학습한다. 이렇게 하면 trivial identity mapping을 피하고, 노이즈에 강한 표현을 얻을 수 있다. 논문은 corrupted input을 hidden representation으로 보낸 뒤 reconstruction loss를 clean input 기준으로 최소화한다고 설명한다. stacked denoising autoencoder(SDAE)는 이 아이디어를 deep network로 확장한 것이다.

**Sparse Autoencoder** 는 hidden unit가 대부분 비활성 상태를 유지하도록 sparsity constraint를 추가한 모델이다. 논문은 hidden neuron 평균 활성도를 낮게 유지하도록 Kullback-Leibler divergence penalty를 cost function에 넣는 방식을 설명한다. 이는 과적합을 줄이고 복잡도를 제어하는 regularization 역할도 한다. 수식적으로는 hidden activation 평균을 $\hat{\rho}_j$로 두고, 목표 sparsity $\rho$와의 KL divergence를 페널티로 더한다.

**Convolutional Autoencoder (CAE)**는 fully connected 구조 대신 convolution/deconvolution을 사용해 이미지의 지역 구조를 보존하면서 특징을 학습한다. 논문은 stacked AE의 layer-wise pretraining 부담을 줄이고, image local structure를 더 잘 반영할 수 있다는 점을 강조한다. 의료영상에서는 fMRI modeling, AD/MCI/HC classification, nucleus detection 등에 적용 사례가 제시된다.

**Variational Autoencoder (VAE)**는 generative model로 소개된다. 논문은 probabilistic encoder $q_\phi(z|x)$와 generative model $p_\theta(x,z)$를 사용해 latent variable model을 학습하며, variational lower bound와 SGVB/AEVB 같은 최적화 아이디어를 쓴다고 설명한다. 즉 VAE는 단순 압축기보다, latent variable distribution을 명시적으로 모델링하는 deep generative model이다.

**Contractive Autoencoder** 는 입력의 작은 변화에 representation이 과도하게 반응하지 않도록 Jacobian norm regularization을 추가하는 방식이다. 논문은 objective에 activation의 Jacobian에 대한 Frobenius norm penalty를 더한다고 설명한다. 저자들은 DAE가 reconstruction robustness를 강조하는 반면, contractive AE는 representation robustness를 더 직접적으로 장려한다고 설명한다.

요약하면 autoencoder 계열은 이 논문에서 의료영상 비지도학습의 중심 축으로 제시되며, 실제 응용 폭도 가장 넓게 정리되어 있다.

### 3.3 Restricted Boltzmann Machine (RBM)

14쪽에서 RBM은 Markov Random Field의 변형이자, visible layer와 hidden layer를 가진 양방향 bipartite graphical model로 설명된다. visible unit $x$와 hidden unit $h$ 사이에는 연결이 있지만 같은 층 내부에는 연결이 없는 “restricted” 구조이기 때문에 일반 Boltzmann machine보다 학습이 효율적이다. 논문은 RBM이 input space 위의 probability distribution을 학습해 새로운 data point를 생성할 수 있는 generative model이라고 정리한다.

의료영상 응용으로는 Alzheimer disease variation detection, multiple sclerosis lesion segmentation, mass detection in breast cancer, fMRI latent source mining, vertebrae localization, benign/malignant ultrasound classification, brain lesion segmentation 등이 Table 3에 정리돼 있다. 즉 RBM은 직접 classifier라기보다 feature learning, dimensionality reduction, generative representation learning 도구로 많이 활용된 것으로 제시된다.

### 3.4 Deep Belief Network (DBN)

15~16쪽에서 DBN은 여러 RBM을 층층이 쌓은 greedy layer-wise unsupervised model로 소개된다. visible layer $v$와 여러 hidden layer $h^1, h^2, \dots, h^L$로 구성되며, 상위 두 층은 undirected generative model, 하위 층은 directed generative model을 형성한다고 설명된다. 논문은 DBN의 joint distribution을 식 (14)로 제시하며, layer-wise training이 optimization과 generalization에 도움이 된다고 정리한다.

응용 사례로는 manifold learning of brain MRI, AD/MCI/HC classification, left ventricle segmentation, schizophrenia classification, lesion classification, cardiac arrhythmia classification, autism spectrum disorder classification 등이 나온다. 이들을 보면 DBN은 특히 MRI, ultrasound, ECG처럼 구조적 신호와 영상에서 pretraining과 hierarchical feature learning에 많이 활용되었다는 점을 알 수 있다.

### 3.5 Deep Boltzmann Machine (DBM)

17쪽에서 DBM은 multiple RBM을 계층적으로 쌓은 robust deep generative model로 설명된다. DBN과 달리 DBM은 상하층 정보가 모두 반영되는 **undirected generative model**이라는 점이 핵심 차이로 강조된다. 이 때문에 representation power가 더 높다고 논문은 주장한다. 또한 layer-wise greedy training을 위해 DBN과는 조금 다른 절차가 필요하다고 설명한다.

논문이 소개한 응용은 상대적으로 적다. heart motion tracking on cine MRI, medical image retrieval, 그리고 일부 DBN/RBM 관련 예시가 같이 정리된다. 이 점은 DBM이 2018년 시점까지 의료영상에서 autoencoder나 RBM/DBN만큼 폭넓게 쓰이지는 않았음을 시사한다.

### 3.6 Generative Adversarial Network (GAN)

17~18쪽에서 GAN은 가장 최근의 유망한 unsupervised generative architecture로 소개된다. generator $G$는 data distribution $p_g$를 학습하고, discriminator $D$는 샘플이 real data에서 왔는지 generator에서 왔는지를 구분한다. 논문은 다음의 전형적인 minimax objective를 제시한다.

$$
\min_G \max_D V(G,D)
====================

\mathbb{E}_{x \sim p*{data}}[\log D(x)]
+
\mathbb{E}_{z \sim p_z}[\log(1-D(G(z)))]
$$

이 구조는 두 네트워크가 적대적으로 경쟁하면서 generator가 점점 더 realistic sample을 생성하도록 학습된다고 설명된다.

의료영상 응용으로는 retinal image synthesis, chest X-ray synthesis, ROI segmentation을 위한 dual GAN-FCN, freehand ultrasound simulation, PET synthesis 등이 Table 6에 정리된다. 이를 보면 GAN은 classification 전처리나 feature learning보다는 **image synthesis, simulation, augmentation, segmentation support** 쪽에서 특히 주목받고 있었음을 알 수 있다.

## 4. Experiments and Findings

이 논문은 실험 논문이 아니라 review이므로, “실험 결과”는 저자들이 정리한 문헌 전반의 경향성과 적용 사례에서 나온다.

가장 먼저 드러나는 사실은 **autoencoder 계열이 가장 폭넓게 활용되고 있다**는 점이다. 12~13쪽의 Table 1과 Table 2는 SAE, SDAE, SSAE, CAE, DCAE 등이 AD/MCI 분류, nucleus detection, stain normalization, density classification, lesion classification, organ detection, segmentation 등에 폭넓게 적용되었음을 보여준다. 이 범위만 봐도 autoencoder는 단순 압축 도구가 아니라 feature extractor, pretraining engine, denoiser, segmentation representation learner로 다양하게 사용되고 있다.

둘째, RBM/DBN 계열은 2018년 시점 의료영상 비지도 딥러닝의 중요한 역사적 축으로 제시된다. 특히 MRI 기반 neuroimaging, ultrasound lesion analysis, ECG arrhythmia classification에서 representation learning이나 multimodal fusion, fine-tuning 기반의 성능 향상을 노린 사례가 많이 소개된다. 이는 당시 의료영상 딥러닝이 end-to-end supervised CNN 일변도라기보다, **unsupervised pretraining과 feature learning**의 비중이 꽤 컸음을 시사한다.

셋째, GAN은 비교적 최신 계열로서 의료영상 합성과 augmentation에 큰 잠재력을 가진 것으로 정리된다. retinal image synthesis, chest X-ray photorealistic generation, ultrasound simulation, PET synthesis 등이 대표 사례다. 즉 GAN은 기존 비지도 모델들이 주로 representation learning에 강점이 있었다면, **data generation과 modality simulation** 영역에서 새로운 가능성을 열었다는 식으로 위치 지워진다.

넷째, 논문은 tools와 datasets를 별도 섹션으로 정리한다. 19쪽 Table 7에는 deeplearning4j, torch7 unsup, DeepPy, SAENET, H2O, dbn, darch, pydbm, xRBM, DCGAN.torch, pix2pix 등 구현 도구가 정리되고, 20쪽 Table 8에는 ABIDE, ADNI, BCDR, DDSM, DRIVE, OASIS, TCIA, TCGA 등 주요 의료영상 데이터셋이 요약된다. 이는 이 논문이 단순 survey를 넘어서, 실제 연구자가 field에 진입할 때 필요한 구현·데이터 출발점을 제공하려 했음을 보여준다.

## 5. Strengths, Limitations, and Interpretation

이 논문의 가장 큰 강점은 **정리의 폭과 구조가 좋다**는 점이다. 비지도 딥러닝을 의료영상 맥락에서 AE, RBM, DBN, DBM, GAN으로 정리하고, 각 모델의 원리와 의료 응용을 나란히 제시한다. 특히 Table 2~6은 모델 계열별 응용 지형도를 빠르게 파악하는 데 유용하다. 또한 Table 7, 8에서 tools와 datasets까지 덧붙여 실무적 참고가치도 높다.

둘째, 저자들은 unsupervised learning의 장점을 과도하게 낭만화하지 않고, 21~23쪽 discussion에서 한계도 명확히 짚는다. 예를 들어 라벨이 없기 때문에 “알고리즘이 정말 유용한 것을 배웠는지 평가하기 어렵다”는 점, 데이터 종류에 따라 맞는 알고리즘과 하드웨어 선택이 어렵다는 점, 의료영상에서 아직 common choice가 아니라는 점, heterogeneous image data와 semantic segmentation, black-box acceptance, radiologist와의 관계 같은 실제적인 문제를 나열한다. 이 부분은 survey의 균형감을 높인다.

하지만 한계도 뚜렷하다.

첫째, 2018년 시점의 survey이기 때문에 이후의 self-supervised learning, contrastive learning, masked modeling, diffusion model, foundation model 계열은 당연히 포함되지 않는다. 오늘날 관점에서 보면 이 논문은 비지도 의료영상 학습의 “초중기 계보”를 정리한 문헌으로 읽는 것이 맞다.

둘째, 모델 원리에 대한 설명은 비교적 폭넓지만, **각 기법 간 공정한 empirical comparison**은 제공하지 않는다. 이는 review 논문의 구조적 한계지만, 독자는 “어떤 모델이 가장 좋다”는 결론을 이 논문에서 기대하면 안 된다.

셋째, unsupervised learning을 medical image analysis의 큰 대안으로 제시하지만, 실제 많은 예시는 결국 downstream supervised fine-tuning이나 feature extraction 보조 역할에 가깝다. 즉 논문이 제시한 vision은 넓지만, 당시 응용 현실은 아직 representation pretraining 단계가 많았다고 해석할 수 있다.

비판적으로 보면 이 논문의 진짜 메시지는 “비지도학습이 곧 의료영상의 미래다”보다는, **의료영상에서 라벨 없는 데이터의 활용은 피할 수 없고, 이를 위해 다양한 unsupervised deep model family가 이미 탐색되고 있다**는 선언에 가깝다. 이는 이후 self-supervised medical imaging이 커지는 흐름을 생각하면 꽤 선견지명이 있었다고 볼 수 있다.

## 6. Conclusion

이 논문은 의료영상 분석에서 unsupervised deep learning의 주요 계열과 응용을 폭넓게 정리한 review다. 핵심은 autoencoder와 그 변형들, RBM, DBN, DBM, GAN을 의료영상 분류, 분할, 특징학습, 합성, 검색 등의 문제와 연결해 소개했다는 점이다. 저자들은 unsupervised learning이 라벨 비용과 supervision bias 문제를 줄이고, 데이터에서 직접 구조를 학습하며, supervised learning의 전처리·표현학습 기반으로도 매우 유용하다고 본다. 동시에 평가의 어려움, 알고리즘 선택 문제, heterogeneous data, black-box 수용성, 의료현장 협업 같은 과제를 남긴다.

정리하면 이 논문은 2018년 시점 의료영상 비지도 딥러닝의 지형도를 보여주는 문헌이다. 오늘날 기준으로는 다소 고전적인 모델군 중심이지만, 의료영상 representation learning의 역사와 흐름을 이해하는 데는 여전히 의미가 있다. 특히 unsupervised learning을 단순 보조기술이 아니라 **의료영상 AI의 확장성과 일반화를 뒷받침하는 핵심 축**으로 본 시각이 이 논문의 가장 중요한 기여다.
