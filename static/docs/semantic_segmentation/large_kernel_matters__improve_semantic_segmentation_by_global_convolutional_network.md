# Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network

- **저자**: Chao Peng, Xiangyu Zhang, Gang Yu, Guiming Luo, Jian Sun
- **발표연도**: 2017
- **arXiv**: https://arxiv.org/abs/1703.02719

## 1. 논문 개요

이 논문은 semantic segmentation에서 왜 large kernel이 중요한지 분석하고, 이를 실제 네트워크 구조로 구현한 **Global Convolutional Network (GCN)** 와 **Boundary Refinement (BR)** 블록을 제안한다. 저자들은 semantic segmentation을 단순히 픽셀별 분류로 보지 않고, 동시에 두 가지 요구를 만족해야 하는 문제로 정리한다. 하나는 각 픽셀이 어떤 semantic class에 속하는지 정확히 맞히는 **classification** 문제이고, 다른 하나는 그 예측이 물체 경계와 정확히 맞아야 하는 **localization** 문제이다.

논문의 핵심 문제의식은 이 두 요구가 서로 긴장 관계에 있다는 점이다. classification은 입력의 이동, 회전, 크기 변화에 대해 어느 정도 불변성을 갖는 표현을 원하지만, localization은 반대로 위치 변화에 민감해야 한다. 기존 segmentation 모델들은 주로 localization 쪽에 치우쳐 설계되어 왔고, 저자들은 이것이 classification 성능을 제한할 수 있다고 본다.

이 문제는 semantic segmentation의 본질과 직접 연결되기 때문에 중요하다. segmentation은 픽셀 단위의 정밀한 출력이 필요하지만, 동시에 그 픽셀이 속한 물체 전체 문맥을 이해해야 한다. 논문은 바로 이 균형을 large kernel 기반 설계로 풀어내려 한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 명확하다. **semantic segmentation에서도 classification 네트워크처럼 넓은 연결 범위가 필요하다**는 것이다. 일반적인 classification 네트워크는 fully-connected layer나 global pooling을 통해 feature map 전체와 classifier가 조밀하게 연결된다. 반면 기존 FCN 계열 segmentation 네트워크는 각 위치 주변의 로컬 feature에 주로 의존한다. 저자들은 이런 구조가 큰 물체나 다양한 변형을 가진 입력에 대해 분류 능력을 떨어뜨릴 수 있다고 주장한다.

이를 해결하기 위해 제안된 것이 GCN이다. GCN은 fully convolutional 구조를 유지하면서도, 각 위치의 classifier가 feature map의 넓은 영역과 상호작용하도록 만든다. 이상적으로는 feature map 전체를 보는 “global convolution”이 가장 좋지만, 큰 $k \times k$ convolution은 계산량과 파라미터 수가 너무 크다. 그래서 저자들은 이를 직접 쓰지 않고, **$1 \times k$ 와 $k \times 1$ convolution을 조합한 separable한 대형 커널 구조**로 근사한다.

기존 접근과의 차별점은 두 가지다. 첫째, 이 논문은 context를 추가로 집어넣거나 후처리로 경계를 다듬는 수준을 넘어서, segmentation의 구조적 모순 자체를 classification 대 localization이라는 관점에서 분석한다. 둘째, boundary refinement도 denseCRF 같은 외부 후처리가 아니라 네트워크 내부의 residual block으로 설계하여 end-to-end 학습이 가능하도록 했다.

## 3. 상세 방법 설명

전체 파이프라인은 pretrained ResNet을 backbone으로 사용하고, FCN 스타일의 multi-scale segmentation framework 위에 GCN과 BR을 얹는 구조다. 서로 다른 stage에서 나온 multi-scale feature map으로부터 semantic score map을 만들고, 낮은 해상도의 score map은 deconvolution으로 upsample한 뒤 높은 해상도의 score map과 더한다. 마지막 upsampling 후 최종 semantic score map을 얻는다.

GCN의 목적은 각 픽셀 분류기가 더 넓은 feature context를 볼 수 있게 하는 것이다. 저자들은 단순한 $k \times k$ convolution 대신 두 경로를 둔다. 한 경로는 $1 \times k$ 뒤에 $k \times 1$ convolution을 적용하고, 다른 경로는 $k \times 1$ 뒤에 $1 \times k$ convolution을 적용한다. 이 둘을 결합하면 큰 $k \times k$ receptive interaction을 만들 수 있다. 논문은 이 구조가 trivial한 대형 convolution보다 훨씬 적은 비용으로 넓은 연결을 제공한다고 설명한다. 파라미터와 연산량은 대략 $O(2k)$ 수준으로 증가하므로, 큰 kernel을 보다 실용적으로 쓸 수 있다.

저자들이 강조하는 개념 중 하나는 **valid receptive field (VRF)** 이다. 네트워크의 이론적 receptive field는 매우 크더라도, 실제로 예측에 강하게 기여하는 영역은 더 작을 수 있다. 기존 segmentation 모델에서는 물체가 커질 때 VRF가 물체 전체를 충분히 덮지 못해 classification이 약해질 수 있다고 본다. GCN은 이 VRF를 실질적으로 확장해 준다는 것이 저자들의 주장이다.

Boundary Refinement 블록은 coarse score map의 경계를 보정하기 위해 추가된다. 논문은 이를 residual 형태로 정의한다.

$$
\tilde{S} = S + R(S)
$$

여기서 $S$는 원래의 coarse score map이고, $R(S)$는 refinement를 담당하는 residual branch이다. 즉, BR은 처음부터 새로운 score map을 만들기보다, 기존 예측을 기준으로 경계 근처의 오차만 보정하는 방식이다. 이 설계는 optimization 측면에서도 자연스럽고, 경계 정렬 문제에 집중하도록 만든다.

논문은 또 pretrained classification backbone 자체에도 large kernel 아이디어를 적용해 **ResNet-GCN**을 제안한다. 원래 ResNet bottleneck의 앞 두 convolution을 GCN 스타일 블록으로 교체한다. 다만 이 부분에서는 Batch Normalization과 ReLU를 각 convolution 뒤에 붙여 원래 ResNet과의 일관성을 유지한다. 이 구조는 classification 성능은 거의 유지하면서 segmentation fine-tuning에서 더 나은 성능을 유도하는지 확인하기 위한 실험적 설계다.

학습 절차는 비교적 표준적이다. PASCAL VOC 2012 실험에서는 ResNet-152를 ImageNet pretrained model로 초기화하고, SGD를 사용하며 batch size는 1, momentum은 0.99, weight decay는 0.0005다. mean subtraction과 horizontal flip을 사용한다. 성능 평가는 mean IoU로 측정한다. 모든 실험은 Caffe에서 수행되었다고 명시되어 있다.

## 4. 실험 및 결과

실험은 크게 ablation, PASCAL VOC 2012, Cityscapes로 구성된다.

먼저 ablation에서 가장 중요한 질문은 정말 large kernel이 효과적인가이다. PASCAL VOC 2012 validation set에서 GCN의 kernel size $k$를 3부터 15까지 바꿔 가며 실험했는데, $k$가 커질수록 mean IoU가 꾸준히 상승한다. baseline인 $1 \times 1$ convolution은 69.0이고, GCN은 $k=3$에서 70.1, $k=7$에서 72.8, $k=15$에서 74.5를 기록한다. 즉, 거의 global convolution에 가까운 큰 kernel이 가장 좋은 성능을 보였다. 이 결과는 “large kernel matters”라는 제목을 직접 뒷받침한다.

다음으로 저자들은 성능 향상이 단순히 파라미터 증가 때문인지 검증한다. trivial한 $k \times k$ convolution과 GCN을 비교했을 때, trivial convolution은 더 많은 파라미터를 쓰지만 성능은 오히려 낮았다. 예를 들어 $k=9$에서 GCN은 73.4, trivial convolution은 68.8이다. 이는 large kernel 자체가 중요하더라도, 그것을 어떻게 구현하느냐가 성능과 학습 안정성에 결정적이라는 뜻이다.

또한 작은 convolution을 여러 층 쌓아 큰 effective kernel을 흉내 내는 방식과도 비교한다. nonlinearity 없이 쌓은 stack of small convolutions는 작은 kernel 구간에서는 어느 정도 경쟁력이 있으나, kernel size가 커질수록 성능이 떨어진다. 예를 들어 $k=11$에 해당하는 경우 GCN은 73.7, stack은 67.5이다. 채널 수를 줄여 파라미터를 맞추면 성능은 더 악화된다. 저자들은 이를 통해 GCN이 단순한 대체 구현이 아니라, 큰 kernel을 효율적으로 쓰는 더 적절한 구조라고 주장한다.

GCN과 BR의 역할 분담도 흥미롭다. 예측 맵을 boundary region과 internal region으로 나눠 분석했을 때, baseline 대비 GCN은 내부 영역 정확도를 93.9에서 95.0으로 크게 올리지만 경계 영역은 71.3에서 71.5로 소폭 개선된다. 반면 GCN에 BR을 추가하면 boundary accuracy가 73.4까지 오른다. overall IoU는 baseline 69.0, GCN 74.5, GCN+BR 74.7이다. 즉, GCN은 주로 물체 내부의 분류를 강화하고, BR은 경계 정렬을 보완한다는 논문의 논리가 실험으로 확인된다.

ResNet-GCN 실험도 중요한 보조 증거다. ResNet50과 ResNet50-GCN을 비교하면 ImageNet classification error는 7.7% 대 7.9%로 거의 비슷하거나 약간 나빠진다. 하지만 segmentation baseline 점수는 65.7에서 71.2로 크게 증가한다. GCN+BR까지 포함하면 72.3 대 72.5로 차이가 줄지만 여전히 GCN backbone이 소폭 우세하다. 즉, GCN은 일반 classification accuracy를 크게 높이는 구조라기보다, segmentation이라는 과제에 더 잘 맞는 inductive bias를 제공한다고 해석할 수 있다.

PASCAL VOC 2012에서는 COCO pretraining을 포함한 3-stage 학습을 사용한다. validation set에서 baseline은 Stage-3 기준 74.0, GCN은 78.7, GCN+BR은 80.3이다. 여기에 multi-scale testing을 추가하면 80.4, denseCRF까지 더하면 81.0이 된다. 최종적으로 test set에서는 **82.2% mean IoU**를 기록하여 당시 기존 최고 성능 80.2%를 넘는다.

Cityscapes에서는 이미지가 더 크기 때문에 학습 시 $800 \times 800$ crop을 사용하고, 마지막 feature map이 $25 \times 25$가 되므로 GCN의 kernel size를 25로 늘린다. coarse annotation과 fine annotation을 섞은 2-stage 학습을 수행한 결과, validation set에서 GCN+BR은 76.9, multi-scale은 77.2, CRF 포함 시 77.4를 얻는다. test set에서도 **76.9% mean IoU**를 기록해, 논문에 제시된 이전 최고 성능 71.8%를 크게 넘는다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의가 분명하다는 점이다. 많은 segmentation 논문이 context aggregation, upsampling, CRF refinement 같은 구성 요소 수준의 개선에 집중하는 반면, 이 논문은 semantic segmentation을 classification과 localization의 긴장 관계로 다시 해석한다. 그리고 그 해석이 단순한 직관에 머무르지 않고, GCN과 BR이라는 구조적 제안으로 이어진다.

두 번째 강점은 ablation이 비교적 설득력 있다는 점이다. 저자들은 단순히 “우리 방법이 baseline보다 좋다”에서 멈추지 않고, trivial large convolution, stack of small convolutions, boundary/internal region 분석, pretrained backbone 교체 실험까지 제시한다. 덕분에 GCN이 왜 작동하는지에 대한 정성적, 정량적 근거가 함께 제공된다.

세 번째 강점은 practical design이다. 진짜 global convolution은 너무 비싸지만, separable large kernel 조합으로 이를 현실적으로 구현했다. 즉, 이 논문은 이론적 주장과 엔지니어링 타협을 동시에 잘 맞춘 사례라고 볼 수 있다.

한계도 분명하다. 첫째, large kernel이 왜 그렇게 효과적인지에 대한 설명은 주로 VRF와 dense connection에 기반한 직관적 논의이며, 더 엄밀한 이론 분석은 부족하다. 저자들도 trivial large convolution이 왜 학습이 잘 안 되는지 “further study가 필요하다”고 적고 있어, 메커니즘은 완전히 해명되지 않았다.

둘째, BR 블록의 내부 구조와 세부 설정은 본문에서 개념적으로는 명확하지만, 제공된 텍스트만으로는 모든 구현 디테일을 완전히 재현할 정도로 상세하지는 않다. 따라서 재현 관점에서는 그림과 appendix, 코드가 추가로 필요할 가능성이 있다. 이는 논문 본문에 명시되지 않은 부분이므로 여기서 더 추측할 수는 없다.

셋째, 당시 기준으로는 강력한 결과지만, 이 논문은 여전히 CNN 기반 dense prediction 패러다임 안에 있다. 이후 등장한 self-attention이나 transformer 계열 방식과 비교하면, global dependency를 다루는 방식이 convolution receptive field 확장에 머문다. 물론 이것은 후대 관점의 해석이며, 논문 자체는 당시 문제 설정 안에서 충분히 강한 기여를 한다.

간단한 비판적 해석을 덧붙이면, 이 논문의 진짜 기여는 “큰 커널을 쓰자”라는 표면적 주장보다, **segmentation에서 local prediction만으로는 classification이 약해질 수 있다**는 구조적 통찰에 있다. large kernel은 그 통찰을 구현한 하나의 강력한 수단이다.

## 6. 결론

이 논문은 semantic segmentation이 단순한 고해상도 예측 문제가 아니라, **classification과 localization을 동시에 만족해야 하는 과제**라는 점을 전면에 내세운다. 이를 위해 fully convolutional 구조를 유지하면서도 넓은 문맥을 활용할 수 있는 Global Convolutional Network를 제안했고, 경계 보정을 위해 residual 형태의 Boundary Refinement를 추가했다.

실험적으로도 이 설계는 매우 강력했다. PASCAL VOC 2012에서 82.2%, Cityscapes에서 76.9%를 기록하며 당시 state-of-the-art를 달성했다. 특히 GCN이 물체 내부의 분류 정확도를 높이고, BR이 경계 정렬을 개선한다는 분석은 모델 각 부분의 역할을 이해하는 데 도움이 된다.

실제 적용 측면에서 이 연구는 dense prediction 문제에서 receptive field와 classifier connectivity를 다시 생각하게 만든다. 향후 연구에서도 단순히 해상도를 높이는 것뿐 아니라, 각 위치의 예측기가 얼마나 넓은 문맥과 연결되는지를 설계하는 것이 중요하다는 점을 보여 준다. 이런 의미에서 이 논문은 이후 semantic segmentation architecture 설계에 꽤 직접적인 영향을 줄 수 있는 작업이다.
