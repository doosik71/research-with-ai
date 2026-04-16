# DDANet: Dual Decoder Attention Network for Automatic Polyp Segmentation

- **저자**: Nikhil Kumar Tomar, Debesh Jha, Sharib Ali, Håvard D. Johansen, Dag Johansen, Michael A. Riegler, Pål Halvorsen
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2012.15245

## 1. 논문 개요

이 논문은 대장내시경(colonoscopy) 영상에서 polyp을 자동으로 분할하는 문제를 다룬다. 단순히 polyp의 존재 여부를 찾는 detection보다, 병변의 경계를 픽셀 단위로 정확히 delineation하는 segmentation은 수술 계획, 병변 크기 추정, 임상 판독 보조 등에서 더 직접적인 가치를 가진다. 저자들은 특히 임상 현장에서 사람이 직접 병변 경계를 표시하는 과정이 시간이 많이 들고, 놓침(miss)이나 주관적 오차가 생길 수 있다는 점을 문제의 출발점으로 둔다.

이 연구의 핵심 문제는 colonoscopy 영상이 매우 까다롭다는 데 있다. polyp은 크기, 색, 질감, 모양이 다양하고, 배경 점막과 시각적으로 유사한 경우가 많으며, 반사광이나 잡음도 존재한다. 이런 이유로 단순한 encoder-decoder 구조만으로는 일반화 성능이 충분하지 않을 수 있다. 논문은 EndoTect challenge의 polyp segmentation task를 배경으로, 공개 데이터셋으로 학습하고 별도의 unseen dataset에서 잘 동작하는 모델을 만드는 것을 목표로 한다.

문제의 중요성은 분명하다. colorectal cancer는 주요 암 원인 중 하나이며, 대장내시경은 조기 발견과 예방에서 핵심적이다. 논문은 adenoma miss-rate가 6%에서 27% 사이일 수 있다고 인용하며, 자동화된 segmentation이 임상 보조 도구(CADx)로서 실제적인 의미를 가질 수 있음을 강조한다. 따라서 이 논문은 정확도뿐 아니라 실시간 처리 가능성(FPS)까지 함께 고려한 시스템을 제안한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 하나의 encoder를 두 개의 decoder가 공유하는 dual decoder 구조에 있다. 첫 번째 decoder는 최종 segmentation mask를 예측하는 주 작업(main task)을 담당하고, 두 번째 decoder는 grayscale image reconstruction을 수행하는 auxiliary task를 담당한다. 저자들의 주장은 이 보조 reconstruction branch가 encoder의 feature를 더 강하게 만들고, 그 결과 segmentation 성능도 함께 향상된다는 것이다.

보다 구체적으로는, autoencoder branch가 중간 단계에서 attention map을 생성하고, 이 attention map을 segmentation branch의 feature에 곱해 주는 방식으로 semantic representation을 강화한다. 즉, reconstruction 자체가 최종 목표는 아니고, segmentation에 유용한 attention을 학습시키기 위한 보조 경로로 사용된다. 이 점이 일반적인 UNet, ResUNet++, DoubleUNet 같은 단일 decoder 기반 구조와의 가장 분명한 차별점이다.

또 하나의 중요한 설계 요소는 residual learning과 squeeze-and-excitation(SE) block의 결합이다. residual block은 깊은 네트워크에서 gradient 흐름을 안정화하고, SE block은 channel-wise attention을 통해 중요한 채널을 더 강조한다. 저자들은 이런 구성 위에 dual decoder를 얹어 feature refinement를 유도한다. 논문이 직접 주장하는 차별성은 “encoder-decoder 기반 segmentation network에 autoencoder branch를 추가하고, 이 branch에서 유도한 attention map을 segmentation decoder에 활용했다”는 점으로 요약할 수 있다.

## 3. 상세 방법 설명

전체 구조는 fully convolutional encoder-decoder 네트워크이며, single encoder와 parallel dual decoders로 이루어진다. encoder는 4개의 encoder block으로 구성되고, 각 decoder도 4개의 decoder block으로 구성된다. 입력은 RGB colonoscopy image이고, 출력은 두 종류다. 첫 번째 decoder는 polyp segmentation mask를 출력하고, 두 번째 decoder는 grayscale reconstruction image를 출력한다.

먼저 입력 RGB 영상은 encoder를 통과하면서 점점 downsampling되며 더 추상적인 feature representation으로 변환된다. encoder의 출력은 두 decoder로 동시에 전달된다. 각 decoder에서는 먼저 $4 \times 4$ transpose convolution을 사용해 spatial resolution을 두 배로 키운다. 이후 encoder의 대응 해상도 feature와 skip connection으로 concatenate한다. 이 skip connection은 원래 해상도 수준의 저수준 정보를 decoder에 전달해 경계 복원에 도움을 주고, 동시에 gradient 전달 경로도 추가해 학습 안정성을 높인다.

decoder 내부에서는 두 개의 residual block이 사용된다. residual block 자체는 두 개의 $3 \times 3$ convolution으로 구성되며, 각 convolution 뒤에 batch normalization과 ReLU activation이 붙는다. 입력과 출력 사이에는 identity shortcut이 연결되어 있어, 깊은 네트워크에서 vanishing gradient 또는 exploding gradient 문제를 줄이도록 설계되었다. 이 부분은 전형적인 ResNet 계열의 residual learning 아이디어를 따른다.

SE block은 channel-wise attention 역할을 한다. 일반적인 CNN은 모든 feature channel을 동등하게 취급하는 경향이 있는데, SE block은 global average pooling으로 각 채널을 요약한 뒤, 2-layer neural network를 통해 채널 importance를 다시 계산한다. 이렇게 얻은 벡터로 feature channel을 re-weighting하여 중요한 채널은 강조하고 덜 중요한 채널은 약화한다. 논문은 DDANet이 ResUNet++와 유사한 encoder-decoder 설계를 따르면서 residual learning과 SE를 함께 사용한다고 설명한다.

DDANet에서 가장 중요한 상호작용은 autoencoder branch와 segmentation branch 사이의 attention 전달이다. 두 번째 decoder, 즉 autoencoder branch의 두 번째 decoder block 출력은 $1 \times 1$ convolution과 sigmoid activation을 통과해 attention map을 만든다. 이 attention map은 첫 번째 decoder, 즉 segmentation branch의 해당 출력 feature와 element-wise multiplication된다. 이렇게 attention이 적용된 결과가 segmentation branch의 다음 decoder block 입력으로 사용된다. 따라서 reconstruction branch는 단순 병렬 보조 출력이 아니라, segmentation branch의 중간 표현을 직접 조절하는 역할을 한다.

최종 출력 단계에서는 각 decoder의 마지막 출력이 각각 $1 \times 1$ convolution과 sigmoid activation을 통과한다. segmentation decoder는 binary mask를 출력하고, autoencoder decoder는 grayscale reconstruction을 출력한다. sigmoid를 사용했다는 점을 보면 두 출력 모두 픽셀 단위 값 예측 문제로 다뤄졌음을 알 수 있다.

학습 목표도 두 갈래로 구성된다. segmentation mask 예측에는 binary cross-entropy(BCE)와 Dice loss의 조합을 사용한다. grayscale image reconstruction에는 BCE를 사용한다. 다만 논문 본문에는 두 손실을 어떻게 최종적으로 합치는지, 예를 들어
$$
\mathcal{L} = \mathcal{L}_{seg} + \lambda \mathcal{L}_{rec}
$$
같은 형태의 전체 loss 식이나 $\lambda$ 가중치 값이 명시되어 있지 않다. 따라서 “두 손실이 함께 사용되었다”는 사실은 분명하지만, 정확한 결합 방식은 논문 텍스트만으로는 알 수 없다. 이 점은 명시적으로 한계로 봐야 한다.

학습 설정은 비교적 단순하다. 입력 해상도는 $512 \times 512$이고, optimizer는 Adam, learning rate는 $1 \times 10^{-4}$, 학습 epoch는 200이다. 구현은 PyTorch 1.6이며, 학습 장비는 NVIDIA DGX-2와 Nvidia V100 Tensor Core GPU를 사용했다.

## 4. 실험 및 결과

학습 데이터는 Kvasir-SEG 데이터셋이다. 이 데이터셋은 1000개의 polyp image와 corresponding segmentation mask, bounding box를 포함한다. 저자들은 이 중 88%를 training set으로, 나머지 12%를 development-test-set으로 사용했다. 또한 challenge 주최 측이 제공한 별도의 unseen test dataset 200장에 대해 예측을 수행했다. 이 unseen dataset의 ground truth는 참가자에게 공개되지 않았기 때문에, test 결과는 challenge 평가 시스템을 통해 얻은 것으로 이해해야 한다.

평가 지표는 challenge의 공식 지표인 Dice Coefficient(DSC)가 중심이지만, 논문은 추가로 mIoU, recall, precision, FPS도 보고한다. 이는 segmentation 정확도뿐 아니라 false negative와 false positive 경향, 그리고 실시간성까지 함께 보려는 의도다.

Kvasir-SEG 내부 분할에서 DDANet은 DSC 0.8576, mIoU 0.7800, recall 0.8880, precision 0.8643, FPS 69.59를 기록했다. unseen challenge dataset에서는 DSC 0.7874, mIoU 0.7010, recall 0.7987, precision 0.8577, FPS 70.23을 달성했다. unseen 데이터에서 내부 평가보다 성능이 떨어지기는 하지만, precision과 FPS가 유지되고 Dice도 비교적 높은 수준이라는 점에서 저자들은 일반화 능력이 있다고 해석한다.

정성적 결과에서는 큰 polyp과 작은 polyp 모두를 어느 정도 잘 분할하는 모습을 보여 준다고 설명한다. 동시에 flat polyp 또는 sessile polyp처럼 경계가 두드러지지 않는 사례는 여전히 어렵다고 인정한다. 이는 실제 임상에서도 어려운 케이스이므로, 논문이 결과를 과장하지 않고 남은 문제를 드러냈다는 점은 긍정적이다.

저자들은 이전 작업인 ResUNet++ 및 관련 선행 결과와 DSC 수치를 비교하며 DDANet이 더 높다고 언급한다. 다만 동일한 train-test split이 아니기 때문에 직접 비교는 어렵다고 스스로 밝힌다. 이 점은 매우 중요하다. 숫자만 보면 개선처럼 보이지만, 실험 프로토콜이 다르면 엄밀한 성능 우위라고 단정할 수 없다. 따라서 이 논문의 정량 결과는 “좋은 성능을 보였다”는 수준으로는 해석할 수 있으나, “기존 방법보다 확실히 우월하다”는 강한 결론을 내리기에는 근거가 부족하다.

실시간성 측면에서는 약 70 FPS 수준이 보고되었다. 이는 inference speed가 실제 임상 보조 시스템에 중요하다는 논문의 문제의식과 잘 맞는다. 다만 FPS 측정 조건, 예를 들어 batch size, 정확한 추론 하드웨어 조건, 전처리 및 후처리 포함 여부는 본문 텍스트에서 자세히 설명되지 않았다. 따라서 절대적인 실시간성 비교에는 주의가 필요하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 architecture 아이디어가 분명하고 설득 가능하다는 점이다. segmentation만 수행하는 것이 아니라 grayscale reconstruction을 auxiliary task로 두고, 여기서 attention map을 생성해 segmentation decoder를 보조하도록 설계했다. 이는 multi-task learning과 attention을 비교적 단순한 구조 안에 결합한 형태로 볼 수 있으며, 논문이 제시한 qualitative, quantitative 결과와도 방향성이 일치한다.

두 번째 강점은 실제 적용 가능성을 의식한 평가다. 의료 영상 segmentation 논문 중에는 정확도만 강조하는 경우가 많지만, 이 논문은 FPS를 명시적으로 제시해 real-time CADx 가능성을 논한다. 또한 unseen dataset에서 결과를 보고함으로써 단순한 closed-set overfitting이 아니라 generalization을 어느 정도 점검하려고 했다.

세 번째 강점은 모델 구성 요소들의 역할이 비교적 직관적이라는 점이다. residual block은 학습 안정화, SE block은 채널 중요도 조절, autoencoder branch는 auxiliary supervision과 attention 제공이라는 식으로 각 모듈의 기능이 분리되어 있다. 구조적 설명이 명확해서 후속 연구자가 변형하거나 ablation study를 설계하기 쉬운 편이다.

반면 한계도 분명하다. 가장 먼저, 논문 텍스트에는 ablation study가 제시되지 않는다. 즉, dual decoder가 실제로 얼마나 기여했는지, attention map이 없으면 얼마나 성능이 떨어지는지, grayscale reconstruction이 다른 auxiliary task보다 나은지 같은 핵심 질문에 직접 답하지 못한다. 현재 결과만으로는 “이 설계가 효과적일 가능성”은 보이지만, 어떤 구성 요소가 실제 개선의 주원인인지는 분리해서 증명되지 않았다.

또 다른 한계는 손실 함수와 학습 설정의 세부 사항이 충분히 공개되지 않았다는 점이다. segmentation loss와 reconstruction loss를 어떻게 결합했는지, 두 branch의 gradient 균형이 어떻게 조절되는지, attention map이 어느 정확한 해상도 단계에서 어떤 텐서 차원으로 작용하는지 등 재현성에 중요한 정보가 일부 빠져 있다. 논문이 GitHub repository를 언급하지만, 본문만 읽고는 완전한 재현이 쉽지 않다.

실험 비교의 엄밀성도 다소 약하다. 논문은 이전 연구보다 좋은 DSC를 언급하지만 split이 달라 직접 비교가 어렵다고 인정한다. 그렇다면 같은 split, 같은 데이터, 같은 지표에서 UNet, ResUNet++, DoubleUNet 등을 재구현해 비교했어야 설계의 장점을 더 강하게 입증할 수 있었을 것이다. challenge unseen set 결과는 의미가 있지만, baseline 대비 향상 폭이 표 안에서 제시되지 않아 독자가 상대적 성능을 평가하기 어렵다.

마지막으로, 논문이 스스로 인정하듯 flat polyp 같은 어려운 사례는 여전히 미해결이다. 이는 임상적으로 중요한 실패 유형일 수 있다. reconstruction branch가 grayscale 복원에는 거의 완벽하게 보이더라도, 그것이 실제로 어려운 병변의 분할 정확도를 얼마나 높였는지는 별도 분석이 필요하다. 향후 super-resolution을 auxiliary task로 쓰겠다는 제안은 흥미롭지만, 현재 논문 안에서는 아직 아이디어 수준이다.

종합하면, 이 논문은 새로운 구조 제안과 실용적 성능 보고라는 측면에서는 가치가 있지만, 학술적으로 더 강한 설득력을 가지려면 더 엄밀한 비교 실험과 ablation이 필요하다.

## 6. 결론

이 논문은 automatic polyp segmentation을 위해 DDANet이라는 dual decoder attention network를 제안한다. 핵심 기여는 shared encoder 위에 segmentation decoder와 grayscale reconstruction decoder를 함께 두고, reconstruction branch에서 생성한 attention map으로 segmentation branch의 feature를 강화했다는 점이다. 그 결과 Kvasir-SEG와 unseen challenge dataset에서 비교적 높은 Dice, mIoU, recall, precision을 기록했고, 약 70 FPS의 빠른 추론 속도도 함께 제시했다.

실제 의미는 분명하다. 이 연구는 polyp segmentation에서 auxiliary reconstruction task가 단순 부가 출력이 아니라 feature refinement 도구로 작동할 수 있음을 보여 준다. 또한 의료 영상에서 정확도와 실시간성을 함께 고려한 설계라는 점에서 CADx 시스템 연구에 실용적인 방향을 제시한다. 다만 후속 연구에서는 ablation, 더 엄밀한 baseline 비교, 어려운 병변 유형에 대한 실패 분석, 손실 설계의 명확화가 필요하다. 그런 보완이 이루어진다면, 이 논문은 multi-task attention 기반 의료 영상 segmentation의 실용적 확장 사례로 더 큰 의미를 가질 수 있다.
