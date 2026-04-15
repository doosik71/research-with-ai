# Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

- **저자**: Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1802.02611

## 1. 논문 개요

이 논문은 semantic segmentation에서 널리 쓰이던 두 계열의 장점을 결합하는 것을 목표로 한다. 하나는 DeepLab 계열처럼 spatial pyramid pooling, 특히 ASPP(Atrous Spatial Pyramid Pooling)를 이용해 다양한 스케일의 문맥 정보를 잘 모으는 방식이고, 다른 하나는 U-Net이나 SegNet처럼 encoder-decoder 구조를 이용해 물체 경계를 더 정교하게 복원하는 방식이다. 저자들은 기존 DeepLabv3가 강한 문맥 표현 능력을 갖고 있지만 최종 예측을 bilinear upsampling에 크게 의존하기 때문에 object boundary가 상대적으로 거칠어질 수 있다는 점에 주목한다.

연구 문제는 명확하다. semantic segmentation에서는 픽셀 단위로 정확한 class를 예측해야 하므로, 넓은 문맥 정보와 세밀한 경계 정보가 동시에 필요하다. 그런데 문맥을 강하게 잡기 위해 깊은 backbone과 downsampling을 사용하면 경계가 흐려지고, 반대로 고해상도 특징을 끝까지 유지하면 계산량과 메모리 비용이 급격히 증가한다. 특히 atrous convolution으로 output stride를 8이나 4까지 줄여 촘촘한 feature를 얻는 것은 좋은 방향이지만, 현대 backbone에서는 계산 비용이 매우 커진다. 따라서 이 논문은 “문맥 정보는 유지하면서도 계산 효율을 크게 해치지 않고 경계를 더 잘 복원할 수 있는 구조”를 제안한다.

이 문제는 실제로 중요하다. semantic segmentation은 자율주행, 로봇 비전, 의료 영상, 장면 이해 등에서 핵심 과제이며, 특히 물체 경계가 불명확하면 downstream decision의 품질이 크게 떨어질 수 있다. 논문은 이 점을 해결하기 위해 DeepLabv3를 encoder로 사용하고, 가벼운 decoder를 덧붙인 DeepLabv3+를 제안한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 실용적이다. 강한 encoder는 유지하되, decoder는 복잡하게 만들지 않고 “정말 필요한 수준으로만” 추가한다. 즉, DeepLabv3가 이미 encoder 단계에서 ASPP를 통해 풍부한 multi-scale context를 잘 담고 있으므로, decoder는 이를 다시 크게 재구성하기보다 low-level feature와 결합해 경계만 정교하게 다듬는 역할을 한다.

기존 encoder-decoder 모델과 비교했을 때의 차별점은 두 가지다. 첫째, 이 구조는 DeepLabv3의 atrous convolution 특성을 그대로 활용하므로 encoder output resolution을 계산 예산에 따라 조절할 수 있다. 논문은 이것이 기존 encoder-decoder 방식에는 없던 유연성이라고 강조한다. 둘째, 단순한 bilinear upsampling 대신 low-level feature를 적절히 끌어와 합치는 decoder를 도입함으로써, 지나치게 무거운 decoder 없이도 경계 성능을 개선한다.

또 하나의 핵심은 atrous separable convolution이다. 저자들은 Xception backbone을 segmentation에 맞게 수정하고, depthwise separable convolution을 ASPP와 decoder에도 적용한다. 이렇게 하면 정확도를 유지하거나 높이면서도 연산량을 크게 줄일 수 있다. 즉, 이 논문은 단순히 성능만 높인 것이 아니라, “정확도와 속도의 균형”도 함께 개선했다.

## 3. 상세 방법 설명

전체 구조는 encoder-decoder 형태다. encoder는 DeepLabv3이고, decoder는 새롭게 추가된 간단한 refinement 모듈이다. encoder에서는 backbone network로부터 feature를 추출하고, ASPP를 통해 여러 dilation rate의 atrous convolution을 병렬 적용해 multi-scale context를 모은다. 여기에 image-level feature도 함께 사용한다. 이렇게 얻은 마지막 feature map이 decoder의 입력이 된다.

이 논문의 기반이 되는 연산은 atrous convolution이다. 논문은 2차원 신호에 대해 다음과 같이 정의한다.

$$
y[i] = \sum_k x[i + r \cdot k] w[k]
$$

여기서 $x$는 입력 feature map, $w$는 convolution filter, $y$는 출력 feature map, 그리고 $r$은 atrous rate이다. $r=1$이면 일반 convolution과 같고, $r>1$이면 filter가 입력을 더 띄엄띄엄 샘플링한다. 직관적으로 보면, 파라미터 수를 늘리지 않고도 receptive field를 넓히는 효과가 있다. 이 덕분에 downsampling을 과도하게 늘리지 않아도 더 넓은 문맥을 볼 수 있다.

논문은 output stride라는 개념도 중요하게 사용한다. 이것은 입력 이미지 해상도를 최종 feature 해상도로 나눈 비율이다. 예를 들어 image classification backbone은 보통 output stride가 32이지만, segmentation에서는 더 촘촘한 예측이 필요하므로 stride를 제거하고 atrous convolution을 넣어 output stride를 16이나 8로 맞춘다. 논문은 train/eval 시 output stride를 바꾸어 정확도와 연산량의 trade-off를 분석한다.

decoder는 구조가 단순하다. 먼저 DeepLabv3 encoder output을 bilinear upsampling으로 4배 키운다. 그다음 backbone의 low-level feature, 구체적으로 ResNet-101에서는 Conv2 단계의 feature를 가져와 같은 해상도로 맞춘 뒤 concatenate한다. 그런데 low-level feature는 채널 수가 많아 semantic-rich한 encoder feature를 압도할 수 있으므로, 먼저 $1 \times 1$ convolution으로 채널 수를 줄인다. 실험 결과 저자들은 low-level feature를 48채널로 줄이는 설정이 가장 적절하다고 보고했다. 이후 concatenated feature에 두 번의 $3 \times 3$ convolution(각 256 filters)을 적용해 경계와 세부 구조를 정제하고, 마지막으로 다시 4배 bilinear upsampling을 수행해 최종 segmentation map을 얻는다.

중요한 점은 저자들이 decoder를 의도적으로 간단하게 유지했다는 것이다. Conv2와 Conv3를 모두 이용해 U-Net처럼 더 깊은 decoder를 구성하는 실험도 했지만, 유의미한 개선이 없었다. 따라서 최종 설계는 “DeepLabv3 feature + channel-reduced Conv2 + 두 개의 $3 \times 3$ convolution”으로 정리된다.

이 논문에서 또 하나의 기술적 축은 depthwise separable convolution이다. 표준 convolution은 공간 방향과 채널 혼합을 동시에 수행하지만, depthwise separable convolution은 이를 두 단계로 분해한다. 먼저 depthwise convolution이 각 채널별로 독립적으로 spatial filtering을 수행하고, 이어서 pointwise convolution인 $1 \times 1$ convolution이 채널 간 정보를 섞는다. 논문은 이 depthwise 부분에 atrous convolution을 적용한 atrous separable convolution을 사용했다. 이 방식은 계산량을 크게 줄이면서 성능을 유지하는 데 기여한다.

backbone으로는 ResNet-101뿐 아니라 수정된 Aligned Xception도 사용한다. 저자들은 Xception을 segmentation에 맞게 바꾸기 위해 세 가지 변경을 가한다. 첫째, 더 깊은 구조를 사용한다. 둘째, max pooling을 stride가 있는 depthwise separable convolution으로 바꾸어 atrous convolution을 자연스럽게 적용할 수 있게 한다. 셋째, 각 $3 \times 3$ depthwise convolution 뒤에 batch normalization과 ReLU를 추가한다. 논문은 이 수정이 정확도와 속도 모두에 도움이 된다고 보고한다.

훈련은 end-to-end로 진행되며, 각 구성 요소를 따로 pretrain하는 piecewise pretraining은 하지 않는다. PASCAL VOC에서는 DeepLabv3와 같은 학습 정책을 사용한다. learning rate는 초기값 0.007, “poly” schedule, crop size는 $513 \times 513$, random scale augmentation을 사용하고, output stride가 16일 때 batch normalization도 fine-tune한다. decoder 안에도 batch normalization이 포함된다. 다만 segmentation loss의 정확한 수식 형태는 본문 발췌 부분에 직접 명시되어 있지 않으므로, 이 보고서에서는 추측하지 않는다. 문맥상 일반적인 per-pixel classification objective를 사용했을 가능성이 높지만, 논문 발췌 텍스트에 명시된 내용만 기준으로 하면 손실 함수의 구체식은 제시되지 않았다.

## 4. 실험 및 결과

논문은 주로 PASCAL VOC 2012와 Cityscapes에서 실험한다. PASCAL VOC 2012는 20개 foreground class와 1개 background class를 가지며, 원래 학습 데이터는 1,464장이지만 추가 annotation을 포함한 trainaug 10,582장을 사용한다. 평가 지표는 21개 클래스 평균 mIOU(mean Intersection-over-Union)이다. Cityscapes는 도시 장면 segmentation 벤치마크로, fine annotation 5,000장과 coarse annotation 약 20,000장을 포함한다.

먼저 decoder 설계에 대한 ablation이 자세하다. low-level feature 채널 수를 줄이는 $1 \times 1$ convolution에서 48채널이 가장 좋은 결과를 보였다. Table 1에 따르면 mIOU는 48채널일 때 78.21%로 가장 높고, 32채널도 비슷하지만 약간 낮다. decoder 내부 refinement convolution의 설계에서는 Conv2 feature를 쓰고, 두 번의 $[3 \times 3, 256]$ convolution을 적용하는 구성이 가장 좋았다. Table 2에서 이 구성은 78.85%를 기록해, 한 번만 쓰거나 세 번 쓰는 경우보다 좋았다. Conv3까지 함께 쓰는 더 복잡한 U-Net식 decoder는 기대만큼 개선되지 않았다.

ResNet-101 backbone 실험에서는 decoder 추가의 효과가 분명하다. 기본 DeepLabv3에서 train output stride 16, eval output stride 16일 때 mIOU는 77.21%였고, decoder를 추가하면 78.85%로 오른다. eval output stride 8일 때도 78.51%에서 79.35%로 향상된다. multi-scale inference와 left-right flip을 추가하면 성능은 더 오르지만 연산량이 크게 증가한다. 예를 들어 train/eval output stride 16, decoder 사용, multi-scale과 flip 사용 시 80.22%까지 올라가지만 Multiply-Adds는 1797.23B로 매우 크다. 논문은 bilinear upsampling만 쓰는 naive decoder보다 제안한 decoder가 경계 복원에 분명히 효과적이라고 해석한다.

coarser feature map도 실험했다. train output stride 32, 즉 학습 시 atrous convolution을 쓰지 않는 설정은 더 빠르지만 성능이 1%에서 1.5% 정도 낮다. decoder를 붙이면 약 2% 개선되지만, 최종적으로는 output stride 16이나 8을 쓰는 쪽이 더 낫다고 결론낸다.

Xception backbone에서는 성능이 더 좋아진다. 논문은 먼저 ImageNet-1K pretraining 성능을 제시하는데, reproduced ResNet-101이 Top-1 error 22.40%, Top-5 error 6.02%인 반면 modified Xception은 각각 20.19%, 5.17%로 더 우수했다. segmentation에서도 이 이점이 이어진다. decoder 없이도 Xception은 ResNet-101보다 약 2% 정도 좋은 성능을 보이고, decoder를 추가하면 eval output stride 16 기준으로 약 0.8% 추가 개선이 있었다.

또한 ASPP와 decoder에 depthwise separable convolution을 도입하면 연산량이 크게 줄어든다. 논문은 Multiply-Adds 기준으로 33%에서 41% 감소를 보고하면서도 mIOU는 거의 유지된다고 밝혔다. 예를 들어 Xception backbone에서 decoder, multi-scale, separable convolution을 쓰는 여러 설정이 비슷한 성능을 유지하면서 더 적은 계산량을 요구한다.

사전학습의 효과도 크다. MS-COCO pretraining은 모든 inference 전략에서 약 2% 정도의 추가 향상을 제공했고, JFT pretraining은 여기에 0.8%에서 1% 정도를 더 올려 주었다. 최종적으로 PASCAL VOC 2012 test set에서 DeepLabv3+는 Xception backbone 기준 87.8%, JFT pretraining을 포함하면 89.0% mIOU를 달성했다. 이는 Table 6에 제시된 당시 top-performing 모델들보다 높은 수치다.

객체 경계 개선 효과를 더 정량적으로 보기 위해 논문은 trimap 실험도 수행한다. object boundary 주변의 좁은 띠 영역에 대해서만 mIOU를 계산한 결과, decoder를 사용할 때 bilinear upsampling만 쓴 경우보다 명확히 개선되었다. 가장 좁은 trimap에서 ResNet-101은 4.8%, Xception은 5.4%의 향상이 있었다. 이것은 제안한 decoder가 특히 경계 주변 예측을 개선한다는 논문의 핵심 주장과 직접 연결된다.

Cityscapes에서도 비슷한 경향이 나타난다. Xception-65 backbone에서 DeepLabv3 baseline은 validation mIOU 77.33%였고, decoder를 추가하면 78.79%로 오른다. 흥미롭게도 Cityscapes에서는 image-level feature를 제거했을 때 79.14%로 오히려 성능이 더 좋아졌다. 이는 image-level feature가 PASCAL VOC에서는 유용하지만 Cityscapes에서는 항상 최적은 아니라는 뜻이다. 더 깊어진 Xception-71 backbone을 쓰면 validation에서 79.55%를 기록한다. 이후 coarse annotations까지 이용해 fine-tuning한 최종 DeepLabv3+는 Cityscapes test set에서 82.1% mIOU를 달성하며 새로운 state-of-the-art를 세웠다.

정성적 결과도 제시된다. 논문은 후처리 없이도 객체를 잘 분할한다고 주장한다. 다만 실패 사례로는 sofa와 chair처럼 시각적으로 유사한 클래스 구분, 심하게 가려진 객체, 드문 시점의 객체가 언급된다. 이러한 실패 모드는 모델이 문맥과 경계를 모두 강화했음에도 불구하고, class ambiguity와 데이터 희소성 문제는 여전히 남아 있음을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 설계가 단순하면서도 효과가 분명하다는 점이다. DeepLabv3라는 강력한 encoder 위에 무거운 decoder를 얹지 않고, 필요한 정도의 low-level feature fusion만 추가해 실제 성능 향상을 이끌어냈다. 특히 trimap 분석을 통해 “경계가 좋아졌다”는 주장을 정량적으로 뒷받침한 점이 설득력 있다. 또한 ResNet-101과 Xception 두 backbone 모두에서 일관된 개선을 보였고, PASCAL VOC와 Cityscapes 두 벤치마크에서도 강한 결과를 얻었다.

또 다른 강점은 정확도와 효율의 균형을 함께 다뤘다는 점이다. 논문은 단순히 mIOU만 보고하지 않고 Multiply-Adds를 함께 제시해 각 설정의 비용을 비교한다. atrous separable convolution을 ASPP와 decoder에 적용해 연산량을 33%에서 41% 줄인 결과는 실용적 가치가 높다. segmentation 모델은 고해상도 feature를 다루므로 비용 문제가 매우 중요하기 때문이다.

방법론적으로도 기여가 명확하다. output stride를 유연하게 조절할 수 있는 encoder-decoder 구조를 제안했고, Xception을 segmentation용으로 구체적으로 수정했으며, decoder 내부의 channel reduction과 convolution 구조에 대한 ablation도 제공했다. 이런 점은 후속 연구나 실전 구현에서 그대로 참고하기 좋다.

한편 한계도 있다. 첫째, decoder는 경계 개선에는 효과적이지만, 논문이 보여주듯 개선 폭은 상황에 따라 제한적이다. 예를 들어 Xception backbone에서 eval output stride 8처럼 이미 비교적 촘촘한 feature를 쓰는 경우 decoder 추가 이득이 더 작아진다. 즉, decoder의 가치가 모든 설정에서 극적으로 큰 것은 아니다.

둘째, 계산량 문제는 완전히 해결되지 않았다. separable convolution을 써도 multi-scale inference와 flip을 사용하면 Multiply-Adds가 매우 커진다. 최고 성능은 여전히 상당한 추론 비용 위에 세워져 있다. 따라서 실제 배포 환경에서는 최고 성능 설정을 그대로 쓰기 어렵다.

셋째, 논문은 image-level feature가 데이터셋에 따라 다르게 작동함을 보여주지만, 왜 그런 차이가 나는지에 대한 깊은 분석은 제한적이다. Cityscapes에서는 image-level feature를 제거하는 것이 더 좋았는데, 이것이 장면 특성 때문인지, 클래스 분포 때문인지, 아니면 학습 설정 때문인지는 본문 발췌 기준으로 명확히 설명되지 않는다.

넷째, 실패 사례가 암시하듯 class ambiguity, severe occlusion, rare viewpoints 같은 어려운 조건은 여전히 남아 있다. 즉, 구조 개선만으로 semantic confusion 자체가 해결되는 것은 아니다. 또한 손실 함수 설계나 class imbalance 대응 같은 측면은 이 논문의 중심이 아니며, 따라서 그러한 문제에 대한 직접적 해결책은 제공하지 않는다.

비판적으로 보면, 이 논문은 혁신적인 완전 신구조라기보다는 DeepLabv3의 강점을 유지하면서 decoder와 separable convolution을 정교하게 통합한 “강한 시스템 설계”에 가깝다. 그러나 바로 그 점 때문에 오히려 영향력이 컸다. 새롭고 복잡한 아이디어보다도, 실제 성능과 효율을 동시에 높이는 합리적 구조를 제안했다는 데 의미가 있다.

## 6. 결론

이 논문은 DeepLabv3를 encoder로 사용하고, low-level feature를 활용하는 간단한 decoder를 붙인 DeepLabv3+를 제안했다. 이를 통해 semantic segmentation에서 중요한 두 요소인 multi-scale contextual understanding과 object boundary refinement를 동시에 강화했다. 또한 modified Xception backbone과 atrous separable convolution을 도입해 속도와 정확도를 함께 개선했다.

실험적으로는 PASCAL VOC 2012 test set에서 최대 89.0%, Cityscapes test set에서 82.1% mIOU를 달성하며 당시 state-of-the-art를 세웠다. 특히 decoder가 object boundary 근처에서 실제로 유의미한 개선을 만든다는 점을 trimap 분석으로 보여준 것이 핵심이다.

실제 적용 측면에서 이 연구는 매우 중요하다. DeepLabv3+는 이후 semantic segmentation의 대표적인 강력 baseline이 되었고, 다양한 backbone과 조합되며 널리 사용되었다. 향후 연구에서도 이 논문의 메시지는 유효하다. 즉, segmentation에서는 더 깊고 복잡한 구조만이 답이 아니라, 문맥과 경계의 역할을 분리해 효율적으로 결합하는 설계가 강력한 성능을 만들 수 있다는 점이다.
