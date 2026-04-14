# U-Net: Convolutional Networks for Biomedical Image Segmentation

- **저자**: Olaf Ronneberger, Philipp Fischer, Thomas Brox
- **발표연도**: 2015
- **arXiv**: https://arxiv.org/abs/1505.04597

## 1. 논문 개요

이 논문은 biomedical image segmentation, 즉 영상의 각 픽셀에 대해 클래스 라벨을 부여해야 하는 문제를 다룬다. 일반적인 image classification과 달리 segmentation은 단순히 무엇이 있는지를 맞히는 것이 아니라, 어디에 있는지까지 정확히 찾아야 한다. 특히 생의학 영상에서는 세포 경계나 막 구조처럼 매우 정밀한 위치 정보가 중요하다.

저자들이 제기하는 핵심 문제는 두 가지이다. 첫째, biomedical domain에서는 ImageNet 같은 대규모 annotated dataset을 확보하기 어렵다. 둘째, 기존의 patch-based sliding-window 방식은 각 픽셀마다 주변 patch를 따로 넣어 예측하므로 계산이 매우 비효율적이고, 넓은 문맥(context)을 볼수록 localization precision이 떨어지는 구조적 한계가 있다.

이 논문의 목표는 적은 수의 annotated image만으로도 end-to-end 학습이 가능하고, 동시에 문맥 정보와 정밀한 localization을 모두 확보할 수 있는 fully convolutional segmentation network를 제안하는 것이다. 저자들은 이를 위해 contracting path와 symmetric expanding path를 결합한 U-shaped 구조를 설계하고, 강한 data augmentation과 weighted loss를 통해 생의학 segmentation에서 매우 높은 성능을 달성했다고 보고한다.

이 문제는 biomedical image analysis에서 매우 중요하다. 세포 분할, 세포 추적, 전자현미경 이미지에서의 구조 추출 등은 정량적 생물학 연구와 실험 자동화의 핵심 단계이기 때문이다. 따라서 적은 데이터로도 잘 학습되고, 정밀하며, 빠르게 동작하는 segmentation 모델은 실제 적용 가치가 매우 크다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 encoder-decoder 계열 구조를 매우 정교하게 segmentation 목적에 맞게 설계한 것이다. 왼쪽의 contracting path는 해상도를 줄이면서 더 추상적이고 넓은 문맥 정보를 추출하고, 오른쪽의 expanding path는 해상도를 복원하면서 픽셀 단위의 정밀한 출력을 만든다. 이 두 경로를 대응되는 해상도 수준에서 연결하여, downsampling 과정에서 얻은 high-resolution feature를 upsampled feature와 결합한다.

이 설계의 핵심 직관은 다음과 같다. 깊은 층의 feature는 문맥은 풍부하지만 공간 해상도가 낮고, 얕은 층의 feature는 위치 정보는 풍부하지만 의미적 문맥이 부족하다. U-Net은 이 둘을 결합해 “어디에 무엇이 있는가”를 동시에 잘 추정하도록 만든다. 저자들은 upsampling path에도 충분한 수의 채널을 유지하여 문맥 정보를 고해상도 층까지 전달할 수 있게 한 점을 중요하게 강조한다.

기존 sliding-window 방식과 비교하면 차별점이 분명하다. sliding-window 방식은 각 픽셀을 위한 patch를 별도로 평가하기 때문에 중복 계산이 많고 느리다. 또한 큰 patch는 더 많은 context를 보게 하지만 pooling이 많아져 localization이 약해지고, 작은 patch는 localization은 낫지만 context가 부족하다. U-Net은 fully convolutional 방식으로 전체 타일을 한 번에 처리하면서, skip connection을 통해 localization과 context를 동시에 확보한다.

또한 이 논문은 architecture만 제안한 것이 아니라, 적은 데이터 상황에 맞는 training strategy도 함께 제안한다. 특히 elastic deformation 기반 data augmentation과 touching cells를 분리하기 위한 weighted loss는 이 논문의 실질적 성공 요인으로 제시된다.

## 3. 상세 방법 설명

전체 네트워크는 contracting path와 expansive path로 구성된다. contracting path에서는 각 단계마다 두 번의 $3 \times 3$ unpadded convolution과 ReLU를 적용한 뒤, stride 2의 $2 \times 2$ max pooling으로 downsampling한다. 각 downsampling 단계가 끝날 때마다 feature channel 수는 두 배로 늘어난다. 이는 해상도는 줄이되 표현력은 높이기 위한 전형적인 설계이다.

expansive path에서는 먼저 feature map을 upsampling한 뒤, $2 \times 2$ convolution, 즉 논문에서 말하는 “up-convolution”을 적용하여 채널 수를 절반으로 줄인다. 그 다음 contracting path의 대응되는 층에서 온 feature map을 crop한 후 concatenate한다. crop이 필요한 이유는 모든 convolution을 padding 없이 valid convolution으로 수행하기 때문에, 경계 픽셀이 계속 줄어들기 때문이다. concatenate 이후 다시 두 번의 $3 \times 3$ convolution과 ReLU를 적용하여 feature를 정제한다.

마지막 출력층에서는 $1 \times 1$ convolution을 사용해 각 픽셀 위치의 64차원 feature vector를 원하는 클래스 수로 사상한다. 논문은 전체 네트워크가 총 23개의 convolutional layer를 가진다고 명시한다.

이 구조의 중요한 구현상 특징은 fully connected layer가 전혀 없고, valid convolution만 사용한다는 점이다. 이 때문에 출력 segmentation map은 입력보다 작지만, 출력에 포함된 각 픽셀은 완전한 receptive field context를 가진다. 저자들은 큰 이미지를 처리하기 위해 overlap-tile strategy를 사용한다. 즉, 큰 이미지를 타일로 나누어 처리하되 인접 타일이 겹치도록 하여 경계 문제를 줄이고, 이미지 바깥의 필요한 문맥은 mirroring으로 보완한다. 이 전략은 GPU 메모리 제약 아래에서도 arbitrarily large image에 적용할 수 있게 해 준다.

학습은 Caffe의 stochastic gradient descent 구현으로 수행된다. 출력은 최종 feature map에 대해 pixel-wise softmax를 적용하여 클래스 확률로 변환한다. 논문에서 softmax는 다음과 같이 정의된다.

$$
p_k(x)=\frac{\exp(a_k(x))}{\sum_{k'=1}^{K}\exp(a_{k'}(x))}
$$

여기서 $a_k(x)$는 픽셀 위치 $x$에서 클래스 채널 $k$의 activation이고, $K$는 클래스 수이다. $p_k(x)$는 해당 픽셀이 클래스 $k$일 확률에 해당하는 값으로 해석된다.

손실은 pixel-wise softmax와 cross entropy를 결합한 형태이며, 논문은 이를 다음과 같이 적고 있다.

$$
E=\sum_{x\in\Omega} w(x)\log\big(p_{l(x)}(x)\big)
$$

여기서 $l:\Omega\to\{1,\dots,K\}$는 각 픽셀의 정답 라벨이고, $w(x)$는 픽셀별 가중치이다. 논문 설명상 이 항은 정답 클래스 확률이 낮을수록 더 큰 패널티를 주는 cross entropy 역할을 한다. 중요한 점은 모든 픽셀이 동일한 중요도를 갖지 않도록 $w(x)$를 도입했다는 것이다.

가중치 맵은 두 목적을 가진다. 하나는 class imbalance를 보정하는 것이고, 다른 하나는 서로 붙어 있는 세포들 사이의 얇은 경계 픽셀을 더 강하게 학습하게 만드는 것이다. 이 가중치 맵은 다음과 같이 정의된다.

$$
w(x)=w_c(x)+w_0\cdot \exp\left(-\frac{(d_1(x)+d_2(x))^2}{2\sigma^2}\right)
$$

여기서 $w_c(x)$는 클래스 빈도 보정용 weight이고, $d_1(x)$는 가장 가까운 세포 경계까지의 거리, $d_2(x)$는 두 번째로 가까운 세포 경계까지의 거리이다. 두 경계에 동시에 가까운 위치는 보통 touching cells 사이의 좁은 separation border에 해당하므로, 이 영역에 큰 weight가 부여된다. 논문에서는 $w_0=10$, $\sigma \approx 5$ pixels로 설정했다.

이 weighted loss는 biomedical segmentation에서 매우 실용적이다. 단순 foreground/background 분류만 학습하면 인접 세포가 하나의 큰 blob으로 합쳐질 수 있는데, 이 논문은 경계 픽셀을 의도적으로 강조함으로써 instance separation에 가까운 효과를 얻는다. 다만 논문은 이것을 semantic segmentation 틀 안에서 수행하고 있으며, 별도의 instance segmentation head를 도입한 것은 아니다.

초기화도 명시되어 있다. 저자들은 깊은 네트워크에서 경로가 많기 때문에 적절한 weight initialization이 중요하다고 말한다. 각 feature map의 분산을 대략 unit variance 수준으로 유지하기 위해, 가중치를 표준편차 $\sqrt{2/N}$의 Gaussian distribution에서 샘플링한다. 여기서 $N$은 한 뉴런으로 들어오는 입력 수이다. 예를 들어 이전 층 채널 수가 64인 $3 \times 3$ convolution이라면 $N=9\cdot64=576$이다. 이는 He initialization과 같은 맥락의 설명이다.

학습 설정도 이 논문의 현실적인 설계 철학을 잘 보여준다. valid convolution 때문에 출력이 입력보다 작아지므로, GPU 메모리를 최대한 효율적으로 쓰기 위해 큰 batch 대신 큰 input tile을 선호한다. 따라서 batch size는 1로 두고, 대신 momentum을 0.99로 높게 설정하여 이전 샘플들의 영향을 충분히 누적하도록 했다.

data augmentation은 이 논문의 핵심 요소 중 하나다. 저자들은 적은 training sample 상황에서 shift, rotation, deformation, gray value variation에 대한 불변성을 학습시키기 위해 augmentation이 필수라고 본다. 특히 elastic deformation이 매우 중요하다고 강조한다. 구체적으로는 $3 \times 3$ coarse grid 위에 random displacement vector를 두고, 그 displacement를 표준편차 10 pixels의 Gaussian에서 샘플링한 뒤 bicubic interpolation으로 per-pixel displacement를 만든다. 또한 contracting path 끝에 drop-out layer를 넣어 추가적인 implicit augmentation 효과도 얻는다.

## 4. 실험 및 결과

논문은 세 가지 biomedical segmentation task에서 U-Net을 평가했다고 말하지만, 본문에서 구체적으로 수치가 제시된 것은 전자현미경 기반 neuronal structure segmentation과 광학 현미경 기반 cell segmentation 두 종류이다.

첫 번째 실험은 ISBI 2012에서 시작된 EM segmentation challenge 데이터셋이다. 데이터는 Drosophila first instar larva ventral nerve cord의 serial section transmission electron microscopy 영상 30장으로 구성되며, 각 이미지는 $512 \times 512$ 크기이고 fully annotated ground truth segmentation map이 제공된다. 클래스는 cells와 membranes이다. 테스트셋의 ground truth는 비공개이며, 예측한 membrane probability map을 제출하면 평가를 받을 수 있다. 평가지표는 10개의 threshold에 대해 계산한 warping error, Rand error, pixel error이다.

U-Net은 입력 데이터를 7개의 rotated version으로 평균한 설정에서, 추가적인 pre-processing이나 post-processing 없이 warping error 0.0003529, Rand error 0.0382를 달성했다. 이는 기존 sliding-window CNN인 IDSIA 결과인 warping error 0.000420, Rand error 0.0504보다 더 좋다. 표에 따르면 warping error 기준으로 U-Net이 1위이며, 인간 성능 수치와도 비교가 제시된다. 저자들은 Rand error 측면에서 더 좋은 알고리즘도 존재하지만, 그것들은 데이터셋 특화 post-processing을 사용했다고 설명한다. 즉, U-Net의 성과는 순수한 network output 수준에서 매우 강력하다는 점이 강조된다.

두 번째 실험은 ISBI cell tracking challenge의 광학 현미경 cell segmentation task이다. 첫 번째 데이터셋 “PhC-U373”는 phase contrast microscopy로 촬영한 Glioblastoma-astrocytoma U373 세포 영상이며, 35장의 partially annotated training image를 포함한다. 이 데이터셋에서 U-Net은 평균 IOU 92.03%를 기록했다. 논문 표에서 second-best 2015 알고리즘은 83%이므로, 약 9%p 이상 큰 차이로 앞선다.

세 번째로 소개되는 “DIC-HeLa” 데이터셋은 differential interference contrast microscopy로 촬영한 HeLa cell 영상이며, 20장의 partially annotated training image를 포함한다. 여기서 U-Net은 평균 IOU 77.56%를 기록했다. second-best 2015 알고리즘의 46%와 비교하면 훨씬 큰 격차를 보인다. 논문은 이 결과를 “large margin”이라고 표현하고 있으며, 실제 표 수치도 그 표현을 뒷받침한다.

실험 결과에서 특히 중요한 점은, 훈련 이미지 수가 매우 적음에도 불구하고 좋은 성능을 냈다는 것이다. EM 데이터셋은 30장, PhC-U373은 35장, DIC-HeLa는 20장 수준이다. 이는 이 논문이 architecture 자체뿐 아니라 augmentation, weighted loss, overlap-tile inference 같은 실전적 설계를 통해 low-data biomedical setting에 맞춘 방법이라는 점을 분명하게 보여준다.

또 하나의 실용적 결과는 속도이다. 초록에서는 $512 \times 512$ 이미지 segmentation에 최근 GPU에서 1초 미만이 걸린다고 했고, 결론에서는 NVidia Titan GPU 6GB에서 전체 training time이 약 10시간이라고 보고한다. 당시 기준으로는 정확도뿐 아니라 계산 효율 측면에서도 상당히 강한 결과로 볼 수 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 segmentation에 필요한 두 요구사항, 즉 넓은 context 활용과 정밀한 localization을 하나의 일관된 구조 안에서 효과적으로 결합했다는 점이다. skip connection을 통한 feature concatenation은 이후 많은 encoder-decoder segmentation 모델의 표준이 되었으며, 이 논문은 그 구조를 biomedical setting에서 매우 설득력 있게 입증했다.

또 다른 강점은 적은 데이터에서의 학습 전략이 구체적이고 현실적이라는 점이다. 단순히 “데이터가 적다”는 문제를 제기하는 데서 끝나지 않고, elastic deformation augmentation, high-momentum SGD, pixel-wise weighted loss, overlap-tile inference 등 실제로 동작하는 학습 및 추론 전략을 함께 제시했다. 특히 touching cells 분리를 위해 경계 부근 픽셀에 큰 loss weight를 두는 설계는 biomedical segmentation의 실제 어려움을 정확히 겨냥한 방법이다.

실험적으로도 강점이 분명하다. 논문은 서로 다른 modality인 EM, phase contrast microscopy, DIC microscopy에서 모두 강한 성능을 보였고, challenge benchmark에서 prior best 또는 second-best 대비 상당한 개선을 제시했다. 이는 특정 데이터셋에만 맞춘 방법이 아니라 구조적으로 일반성이 있음을 시사한다.

한계도 있다. 첫째, 논문은 주로 2D segmentation 설정을 다루며, 3D volumetric context를 직접 모델링하지 않는다. EM stacks라는 표현이 등장하지만, 본문에 제시된 네트워크와 실험 설명은 기본적으로 2D 이미지 단위 처리이다. 따라서 인접 슬라이스 간 정보를 어떻게 활용하는지는 이 논문 범위에서는 다뤄지지 않는다.

둘째, valid convolution을 사용하기 때문에 출력 크기가 입력보다 줄어들고, 이를 보완하기 위해 crop과 overlap-tile strategy가 필요하다. 이 방식은 당시에는 합리적이지만, 구현 복잡성과 경계 처리 부담을 수반한다. 이후 padding을 활용한 더 단순한 구현들이 널리 쓰이게 된 이유 중 하나이기도 하다.

셋째, 세포 분리를 위해 weighted loss를 사용하지만, 이것이 명시적 instance segmentation을 수행하는 것은 아니다. 즉, 서로 붙은 객체를 분리하는 데 도움은 주지만, 객체 단위 식별을 직접적으로 모델링한 구조는 아니다. 논문도 이에 대해 instance-level formulation을 제시하지는 않는다.

넷째, 논문은 매우 성공적인 결과를 보였지만, ablation study는 제한적이다. 예를 들어 elastic deformation이 얼마나 기여했는지, weighted loss가 어느 정도 성능 향상을 주는지, expansive path의 채널 수가 왜 중요한지 등을 체계적으로 분리하여 분석하지는 않는다. 따라서 어떤 요소가 성능 향상에 가장 크게 기여했는지는 본문만으로 정량적으로 분해하기 어렵다.

비판적으로 보면, 이 논문의 영향력은 매우 크지만, 본문 자체는 오늘날 기준으로는 비교적 짧고 실험 해설도 압축적이다. 그러나 그럼에도 불구하고 제안 구조와 핵심 설계 논리는 명확하고, 제시된 결과는 충분히 설득력 있다. 특히 “적은 데이터에서 잘 작동하는 segmentation network”라는 목표에 대해, 단순한 주장 수준이 아니라 benchmark 성능으로 입증했다는 점이 중요하다.

## 6. 결론

이 논문은 biomedical image segmentation을 위해 U-shaped fully convolutional architecture를 제안하고, contracting path와 expansive path를 결합한 구조를 통해 문맥 정보와 정밀한 localization을 동시에 확보했다. 또한 elastic deformation 기반 data augmentation, class imbalance 및 touching-cell separation을 고려한 weighted loss, overlap-tile inference 전략을 결합하여 적은 수의 annotated image만으로도 높은 성능을 달성했다.

주요 기여는 단순히 새로운 네트워크 모양을 제안한 데 있지 않다. 이 논문은 low-data biomedical segmentation이라는 실제 문제 조건에 맞게 architecture, loss, augmentation, inference strategy를 하나의 완성된 시스템으로 제시했다. 그리고 EM segmentation challenge와 ISBI cell tracking challenge에서 매우 강한 결과를 통해 그 유효성을 입증했다.

이 연구는 이후 semantic segmentation, medical imaging, 그리고 encoder-decoder 기반 dense prediction 전반에 매우 큰 영향을 미쳤다고 볼 수 있다. 실제 적용 측면에서는 세포 분할, 조직 분석, 현미경 영상 처리 자동화 등 다양한 biomedical workflow의 기반 기술이 될 수 있다. 향후 연구 측면에서도 3D U-Net, attention-based U-Net, residual U-Net 등 수많은 후속 확장의 출발점이 되는 구조적 아이디어를 제공했다.
