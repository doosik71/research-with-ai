# Robust Semantic Segmentation of Brain Tumor Regions from 3D MRIs

- **저자**: Andriy Myronenko, Ali Hatamizadeh
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2001.02040

## 1. 논문 개요

이 논문은 multimodal 3D MRI로부터 brain tumor region을 정확하고 안정적으로 분할하는 semantic segmentation 방법을 제안한다. 구체적으로는 BraTS 2019 challenge를 대상으로, 여러 MRI modality를 함께 입력으로 사용하여 tumor의 서로 다른 하위 영역을 자동으로 분할하는 것이 목표이다.

연구 문제는 의료 영상에서의 3D brain tumor segmentation이다. 뇌종양은 수술 계획, 치료 반응 추적, 질병 모니터링에 직접 연결되므로, 종양의 위치와 범위를 정밀하게 나누는 작업이 매우 중요하다. 특히 glioma는 저등급과 고등급으로 나뉘고, 고등급 glioma는 빠르게 성장하며 예후가 좋지 않기 때문에 정밀한 영상 기반 분석의 필요성이 크다.

BraTS 데이터는 T1, T1c, T2, FLAIR의 4개 MRI modality를 제공하며, 종양의 서로 다른 특성을 강조한다. 예를 들어 T1c는 contrast agent에 의해 enhancing tumor를 더 잘 보이게 한다. 논문은 이런 multimodal 정보를 함께 사용하면 종양의 전체 범위와 내부 구조를 더 잘 구분할 수 있다고 본다.

이 논문의 실질적인 목적은 새로운 복잡한 이론을 제시하기보다는, 3D encoder-decoder 구조, normalization 선택, loss 조합, multi-GPU 학습 등 3D medical segmentation의 여러 실전적 설계 요소를 정리하고 조합하여 성능을 높이는 데 있다. 즉, “무엇이 실제로 잘 동작하는가”를 BraTS 2019 기준으로 검증한 실용적 성격이 강한 논문이다.

## 2. 핵심 아이디어

핵심 아이디어는 비교적 표준적인 3D encoder-decoder CNN을 기반으로 하되, segmentation 정확도를 높이기 위해 loss를 단일 손실이 아니라 hybrid loss로 구성하는 것이다. 저자들은 soft Dice loss만으로는 부족할 수 있다고 보고, 여기에 focal loss와 active contour 기반 손실을 추가했다. 이 조합은 class imbalance, 경계 품질, region-level overlap을 동시에 다루려는 의도를 가진다.

또 하나의 중요한 아이디어는 지나치게 복잡한 구조를 도입하기보다, 해상도 보존과 학습 안정성에 더 집중한 점이다. encoder는 입력을 8배까지만 downsample하고, 그 이상 축소하지 않는다. 논문은 더 깊은 downsampling 대신 더 많은 spatial content를 보존하는 것이 3D brain tumor segmentation에 유리하다고 판단했다.

이전 해의 저자 본인 방법은 secondary task decoder를 둔 autoencoder regularization 구조였지만, 이번 논문에서는 그 보조 decoder를 사용하지 않고 architecture choice와 complementary loss function에 집중한다. 따라서 차별점은 auxiliary task보다는 segmentation 본체의 설계와 학습 objective를 단순하고 강하게 조합하는 데 있다.

또한 normalization에 대한 비교도 실용적 기여 중 하나다. 저자들은 Group Normalization, Instance Normalization, Batch Normalization을 실험했고, batch size가 크지 않은 3D MRI 환경에서는 BatchNorm이 불리하며 GroupNorm과 InstanceNorm이 유사하게 잘 작동했다고 보고한다. 최종적으로는 구현과 이해가 더 단순한 InstanceNorm을 기본 선택으로 삼는다.

## 3. 상세 방법 설명

전체 시스템은 4채널 3D MRI 입력을 받아 3개의 tumor subregion을 동시에 예측하는 3D encoder-decoder 네트워크다. 입력은 T1, T1c, T2, FLAIR를 channel-wise로 concat한 4채널 볼륨이며, 출력은 WT(whole tumor), TC(tumor core), ET(enhancing tumor)의 3개 channel이다. 각 채널은 sigmoid를 거쳐 확률 맵으로 예측된다. 논문은 이 3개 클래스가 nested subregion이라는 점을 명시한다. WT는 전체 종양, TC는 그 내부 core, ET는 그중 enhancing 부분이다.

Encoder는 ResNet block 기반이다. 각 block은 normalization, ReLU, convolution을 두 번 수행한 뒤 identity skip connection을 더하는 residual 구조다. 모든 convolution은 $3 \times 3 \times 3$이다. 초기 filter 수는 32이며, spatial size를 절반으로 줄일 때마다 feature channel 수는 2배가 된다. Downsampling은 pooling이 아니라 stride convolution으로 수행한다.

입력 crop 크기는 $4 \times 160 \times 192 \times 128$이다. Encoder를 지나면 최종 endpoint는 $256 \times 20 \times 24 \times 16$이 된다. 즉 spatial dimension은 입력보다 8배 작아진다. 저자들은 더 깊은 downsampling을 하지 않은 이유를 “더 많은 spatial content를 보존하기 위해서”라고 설명한다.

Decoder는 encoder와 대칭적이지만 더 단순하다. 각 spatial level마다 먼저 $1 \times 1 \times 1$ convolution으로 channel 수를 절반으로 줄이고, 그 다음 3D bilinear upsampling으로 spatial size를 2배 키운다. 이후 같은 해상도의 encoder feature를 더한다. 이는 U-Net류의 skip connection과 유사하지만, concat이 아니라 addition을 사용한다. 각 decoder level에는 하나의 residual block만 둔다. 마지막에는 $1 \times 1 \times 1$ convolution으로 3개 출력 채널을 만들고 sigmoid를 적용한다.

손실 함수는 다음과 같은 hybrid loss다.

$$
L = L_{dice} + L_{focal} + L_{acl}
$$

여기서 $L_{dice}$는 soft Dice loss이며, 예측 $p_{pred}$와 정답 $p_{true}$ 사이의 overlap을 직접 최적화한다.

$$
L_{dice} = 1 - \frac{2 \sum p_{true} p_{pred}}{\sum p_{true}^2 + \sum p_{pred}^2 + \epsilon}
$$

합은 voxel 단위로 계산된다. 출력이 3채널이므로 각 tumor subregion에 대한 dice loss를 더한다. Dice loss는 medical segmentation에서 class imbalance에 강하고 영역 겹침을 잘 반영한다는 장점이 있다.

$L_{acl}$은 supervised active contour loss의 3D 확장판이다. 이는 volumetric term과 length term으로 구성된다.

$$
L_{acl} = L_{vol} + L_{length}
$$

volumetric term은 foreground와 background에 대한 에너지 관점의 오차를 계산한다.

$$
L_{vol} = \left| \sum p_{pred}(c_1 - p_{true})^2 \right| + \left| \sum (1 - p_{pred})(c_2 - p_{true})^2 \right|
$$

여기서 $c_1$, $c_2$는 각각 foreground와 background의 energy를 의미한다. 논문은 이 값들의 구체적 설정 방식은 설명하지 않는다. 따라서 식의 구조는 제시되지만, 실제 구현 세부값은 본문만으로는 완전히 알 수 없다.

length term은 예측 segmentation의 경계 길이, 즉 3D에서는 표면 복잡도에 대응하는 항이다.

$$
L_{length} = \sum \sqrt{|(\nabla p_{pred,x})^2 + (\nabla p_{pred,y})^2 + (\nabla p_{pred,z})^2| + \epsilon}
$$

직관적으로 이 항은 지나치게 들쭉날쭉한 경계를 억제하고 더 매끈한 segmentation surface를 유도하려는 목적을 가진다.

$focal loss$는 다음과 같다.

$$
L_{focal} = - \frac{1}{N} \sum (1 - p_{pred})^\gamma p_{true} \log(p_{pred} + \epsilon)
$$

여기서 $N$은 전체 voxel 수이고, $\gamma = 2$로 설정한다. focal loss는 이미 쉽게 맞히는 voxel보다 어려운 voxel에 더 집중하도록 만들어, 어려운 종양 영역이나 불균형 클래스 상황에서 도움이 되도록 설계되었다.

최적화는 Adam optimizer를 사용하며, 초기 learning rate는 $\alpha_0 = 10^{-4}$이다. 학습률 스케줄은 다음과 같다.

$$
\alpha = \alpha_0 \left(1 - \frac{e}{N_e}\right)^{0.9}
$$

여기서 $e$는 현재 epoch, $N_e$는 총 epoch 수로 300이다. 각 epoch에서는 training image를 무작위 순서로 한 번씩 사용한다.

Regularization으로는 convolution kernel에 대해 weight $10^{-5}$의 L2 regularization을 사용하고, encoder의 초기 convolution 뒤에 rate 0.2의 spatial dropout을 둔다.

입력 전처리는 non-zero voxel만 기준으로 평균 0, 표준편차 1이 되도록 정규화한다. Data augmentation은 비교적 단순하다. 채널별로 무작위 intensity shift를 $-0.1$에서 $0.1$ 표준편차 범위로 적용하고, scale은 0.9에서 1.1 범위로 조정한다. 또 세 축 각각에 대해 확률 0.5로 random mirror flip을 적용한다. 논문에 따르면 histogram matching, affine transform, rotation, random filtering 같은 더 복잡한 augmentation은 추가적인 성능 향상을 주지 않았다.

## 4. 실험 및 결과

실험은 BraTS 2019 training set 335 cases로 학습하고, validation set 125 cases 및 testing set에서 평가하는 방식으로 진행되었다. 추가적인 in-house data는 사용하지 않았다고 명시한다. 각 case는 4개 MRI modality를 포함하며, 원본 볼륨 크기는 $240 \times 240 \times 155$이지만 학습 시에는 random crop $160 \times 192 \times 128$을 사용했다. 저자들은 이 crop이 대부분의 유효한 image content를 포함한다고 설명한다.

평가는 BraTS 서버에 segmentation mask를 업로드하여 수행했다. 지표는 per-class Dice, sensitivity, specificity, Hausdorff distance이다. 다만 본문 표에는 Dice와 Hausdorff 값이 주로 제시되어 있고, sensitivity와 specificity의 구체 수치는 이 발췌문에는 포함되어 있지 않다. 따라서 해당 지표를 사용했다는 사실은 명확하지만, 수치까지 보고하려면 원문 표 전체가 더 필요하다.

Validation set에서 single model(batch size 8)의 결과는 다음과 같다. Dice는 ET 0.800, WT 0.894, TC 0.834였다. Hausdorff distance는 ET 3.921 mm, WT 5.89 mm, TC 6.562 mm였다. 이는 whole tumor에서 가장 높은 Dice를 보이고, enhancing tumor에서 상대적으로 어려운 경향을 보인다는 전형적인 BraTS 특성과도 맞아 있다.

Testing set에서는 ensemble 결과가 보고되었다. Dice는 ET 0.826, WT 0.882, TC 0.837였고, Hausdorff distance는 ET 2.203 mm, WT 4.713 mm, TC 3.968 mm였다. 논문은 최종 testing 결과로 ET 0.826, WT 0.882, TC 0.837의 평균 Dice를 강조한다. 다만 테스트 표는 ensemble 결과이고, ensemble을 어떻게 구성했는지에 대한 상세 설명은 이 발췌문에는 없다. 따라서 testing 성능이 단일 모델인지 여러 모델 결합인지 구분해서 해석해야 한다.

정성적 결과로 Figure 1에서는 axial, sagittal, coronal slice 위에 예측과 정답 segmentation을 겹쳐 보여주며, 예측이 ground truth와 잘 일치한다고 설명한다. 다만 본문 발췌만으로는 어떤 실패 사례가 있었는지, 어느 subregion에서 오차가 컸는지까지는 알 수 없다.

학습 및 추론 시간도 보고된다. 단일 NVIDIA Tesla V100 32GB GPU에서 1 epoch는 약 10분이며, 300 epoch 전체 학습은 약 2일이 걸린다. NVIDIA DGX-1의 8개 V100 GPU를 사용하면 약 8시간으로 줄어든다. 단일 모델 inference time은 단일 V100 GPU에서 0.4초라고 한다. 이는 실용적 배치나 임상 보조 도구 관점에서 충분히 빠른 편으로 해석할 수 있다.

Discussion에서 저자들은 normalization과 batch size 관련 실험 결과도 정리한다. GroupNorm과 InstanceNorm은 비슷했고, BatchNorm은 항상 열세였다고 한다. 이는 최대 batch size가 16에 불과했던 3D 환경에서 BatchNorm의 통계 추정이 불안정했기 때문일 가능성을 제시한다. 또 multi-GPU에서 각 GPU당 batch 1로 data parallel training을 해도 single GPU batch 1과 성능이 비슷했고, 전체 batch 8이 훨씬 빠르기 때문에 이를 기본값으로 사용했다고 말한다.

또한 network depth를 더 늘리는 것은 성능 향상을 주지 않았지만, network width 즉 filter 수를 늘리는 것은 일관되게 성능을 향상시켰다고 보고한다. 이는 3D segmentation에서 지나친 downsampling이나 깊이 증가보다 충분한 channel capacity가 더 중요할 수 있음을 시사한다.

## 5. 강점, 한계

이 논문의 강점은 첫째, 구조가 비교적 단순하면서도 BraTS benchmark에서 강한 성능을 보인다는 점이다. 복잡한 cascade나 attention ensemble 없이도, 잘 설계된 3D encoder-decoder와 적절한 loss 조합만으로 높은 정확도를 달성했다는 점은 재현성과 실용성 측면에서 의미가 있다.

둘째, loss 설계가 목적에 맞게 조합되어 있다. Dice loss는 overlap, focal loss는 어려운 voxel과 class imbalance, active contour loss는 boundary regularization과 region smoothness를 담당한다. 이 조합은 brain tumor segmentation처럼 클래스 불균형이 심하고 경계가 복잡한 문제에 논리적으로 잘 맞는다.

셋째, 논문은 실전적 관찰을 제공한다. 예를 들어 BatchNorm보다 InstanceNorm/GroupNorm이 낫다거나, 더 복잡한 augmentation이 항상 도움이 되지 않는다거나, depth 증가보다 width 증가가 효과적이었다는 식의 결론은 비슷한 3D medical imaging 문제를 다루는 연구자에게 유용하다.

넷째, 계산 효율도 나쁘지 않다. 0.4초 inference와 multi-GPU 기반 빠른 학습 시간은 단순히 정확도만이 아니라 실제 시스템 구성 가능성까지 보여준다.

반면 한계도 분명하다. 우선 논문은 ablation study를 상세하게 제시하지 않는다. hybrid loss의 각 항이 얼마나 기여했는지, Dice만 썼을 때와 비교해 active contour나 focal term이 각각 어떤 개선을 만들었는지 발췌문에서는 확인할 수 없다. 따라서 제안된 각 설계 요소의 독립적인 효과를 정량적으로 판단하기는 어렵다.

둘째, testing 결과가 ensemble로 제시되는데 ensemble 구성 방식이 자세히 설명되지 않는다. 몇 개 모델을 어떤 방식으로 결합했는지, single model 대비 얼마나 개선되었는지, validation 성능과 어떤 관계가 있는지 이 텍스트만으로는 충분히 알 수 없다.

셋째, active contour loss의 계수나 $c_1$, $c_2$ 설정 방법 등 구현에 중요한 세부 정보가 본문 발췌에 없다. 식은 제시되었지만 실제 reproduction에는 추가 정보가 필요할 수 있다.

넷째, 일반화 범위에 대한 논의가 제한적이다. BraTS는 강력한 benchmark이지만, 이 모델이 다른 병원, 다른 acquisition protocol, 다른 tumor type으로 얼마나 잘 일반화되는지는 본문에서 별도로 분석하지 않는다. 데이터가 여러 기관에서 수집되었다는 점은 장점이지만, domain shift에 대한 체계적 분석은 없다.

다섯째, 논문은 robust라는 제목을 사용하지만, robustness를 명시적으로 정의하거나 adversarial noise, missing modality, scanner variation, annotation uncertainty 같은 조건에서 체계적으로 robustness를 평가하지는 않는다. 따라서 여기서의 robust는 실험적으로 잘 동작하는 강건한 성능이라는 의미에 가깝고, 엄밀한 robustness benchmark를 수행한 것은 아니다.

비판적으로 보면, 이 논문은 매우 강한 engineering paper다. 즉, 새로운 원리보다 “좋은 설계 조합”의 가치가 큰 논문이다. 이런 유형의 논문은 실제 현장에서 매우 중요하지만, 각 구성 요소의 기여를 더 세밀하게 해부했더라면 학술적 설득력이 더 높아졌을 것이다.

## 6. 결론

이 논문은 multimodal 3D MRI 기반 brain tumor segmentation을 위해, ResNet-style 3D encoder-decoder와 hybrid loss를 결합한 실용적이고 강력한 접근을 제시한다. 입력 해상도를 과도하게 줄이지 않고 spatial 정보를 보존하면서, Dice loss, focal loss, active contour loss를 함께 사용해 영역 겹침, class imbalance, 경계 품질을 동시에 개선하려 한 점이 핵심이다.

BraTS 2019에서 validation과 testing 모두 경쟁력 있는 성능을 보였고, 특히 WT, TC, ET의 세 하위 영역을 안정적으로 예측했다. 또한 normalization, batch size, augmentation, network width/depth에 대한 관찰은 이후 3D medical image segmentation 연구에도 직접적인 실무 지침이 될 수 있다.

실제 적용 측면에서는 빠른 inference와 단순한 구조 덕분에 임상 워크플로우 보조 시스템으로 연결될 가능성이 있다. 향후 연구에서는 loss 구성 요소의 정밀한 ablation, missing modality나 domain shift에 대한 robustness 검증, uncertainty estimation, 더 다양한 의료기관 데이터에 대한 일반화 평가가 뒤따르면 이 접근의 가치가 더 분명해질 것이다.
