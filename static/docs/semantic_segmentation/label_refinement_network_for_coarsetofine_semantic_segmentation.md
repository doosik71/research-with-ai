# Label Refinement Network for Coarse-to-Fine Semantic Segmentation

- **저자**: Md Amirul Islam, Shujon Naha, Mrigank Rochan, Neil Bruce, Yang Wang
- **발표연도**: 2017
- **arXiv**: https://arxiv.org/abs/1703.00551

## 1. 논문 개요

이 논문은 semantic segmentation, 즉 이미지의 모든 픽셀에 대해 어떤 semantic class에 속하는지를 예측하는 문제를 다룬다. 저자들은 기존의 많은 CNN 기반 segmentation 방법이 최종 해상도에서 한 번에 결과를 내는 “single-shot” 방식에 가깝다고 보고, 이를 대체하는 coarse-to-fine 예측 구조를 제안한다. 핵심은 처음에는 거친 해상도에서 대략적인 label map을 만들고, 이후 더 높은 해상도의 feature와 결합해 점점 더 정밀한 segmentation map으로 다듬어 가는 것이다.

연구 문제의 본질은 분류에 유리한 고수준 특징과 픽셀 단위 정밀성이 서로 충돌한다는 데 있다. CNN encoder는 pooling과 subsampling을 거치면서 object-level 의미 정보는 잘 보존하지만, 경계나 얇은 구조물처럼 세밀한 위치 정보는 잃기 쉽다. semantic segmentation은 이런 정밀한 공간 정보가 매우 중요하므로, 단순히 encoder의 마지막 feature만으로는 충분하지 않다.

이 문제가 중요한 이유는 semantic segmentation이 자율주행, 장면 이해, 로보틱스, 실내 인식 등에서 핵심적인 기반 기술이기 때문이다. 특히 작은 물체, 얇은 구조물, 경계 영역처럼 어려운 부분에서 성능을 높이려면, coarse semantic understanding과 fine spatial detail을 동시에 다루는 설계가 필요하다. 이 논문은 바로 그 지점을 겨냥해 architecture 차원의 해법을 제시한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 segmentation 결과를 한 번에 완성하지 않고, 여러 해상도에서 점진적으로 정제한다는 것이다. 가장 낮은 해상도에서는 큰 구조와 대략적인 object layout만 맞추고, 이후 각 단계에서 encoder의 더 세밀한 convolutional feature를 받아 현재의 coarse label map을 refinement하는 방식이다. 즉, segmentation map 자체가 decoder를 따라 점차 정교해지는 intermediate representation이 된다.

기존 encoder-decoder 계열 방법들과의 차별점은 두 가지다. 첫째, decoder 내부 여러 단계에서 실제 segmentation label map을 명시적으로 예측한다. 논문은 각 단계의 출력 채널 수를 클래스 수 $C$와 동일하게 강제하여, 각 stage의 출력 $s_k(I)$를 단순 feature map이 아니라 “soft label map”으로 해석한다. 둘째, 마지막 출력에만 loss를 두는 대신, 여러 해상도에서 down-sampled ground truth를 사용해 deep supervision을 준다. 따라서 학습 신호가 decoder 말단뿐 아니라 중간 단계들에도 직접 전달된다.

저자들의 관점에서 이 구조의 의미는 단순한 skip connection 추가 이상이다. coarse prediction이 다음 refinement의 입력으로 직접 사용되므로, 네트워크는 “현재까지의 segmentation 가설”을 점진적으로 수정하는 형태로 동작한다. 이는 feature를 한 번에 복원하는 접근보다 segmentation 문제 자체를 더 직접적으로 모델링한 것이라 볼 수 있다.

## 3. 상세 방법 설명

전체 구조는 encoder-decoder framework를 따른다. encoder는 VGG16 기반 SegNet과 유사하며, 입력 이미지 $I \in \mathbb{R}^{h \times w \times d}$를 여러 convolution과 pooling을 거쳐 더 작은 spatial size의 feature map으로 변환한다. 논문에서는 encoder의 각 stage 출력 feature를 $f_k(I) \in \mathbb{R}^{h_k \times w_k \times d_k}$로 표기한다. spatial resolution은 깊어질수록 감소한다.

decoder의 핵심은 총 6개의 label map $s_1, s_2, \dots, s_6$를 생성한다는 점이다. 가장 첫 번째 coarse label map은 encoder 마지막 feature $f_5(I)$에서 바로 얻는다. 논문은 이를 다음과 같이 쓴다.

$$
s_1(I) = conv_{3 \times 3}(f_5(I))
$$

여기서 $s_1(I)$의 채널 수는 클래스 수 $C$와 동일하다. 따라서 각 위치에서 $C$개 클래스에 대한 score를 가지는 coarse segmentation map이 된다. ground truth segmentation을 $Y \in \mathbb{R}^{h \times w \times C}$라고 하고, 이를 $s_1(I)$의 해상도로 resize한 결과를 $R_1(Y)$라고 하면, 첫 번째 loss는 다음과 같다.

$$
\ell_1 = Loss(R_1(Y), softmax(s_1(I)))
$$

논문은 $Loss(\cdot)$로 cross-entropy loss를 사용한다고 명시한다. 추출 텍스트에는 ground truth를 “one-shot representation”이라 적고 있으나, 문맥상 일반적인 one-hot representation을 의미하는 것으로 읽힌다. 다만 제공된 텍스트 자체에는 그렇게 추출되어 있다.

이후 단계의 핵심은 refinement module이다. 단순히 이전 label map을 upsample만 하면 해상도만 커질 뿐 여전히 거친 결과가 나온다. 그래서 저자들은 encoder의 대응되는 convolutional feature를 skip connection으로 가져와 결합한다. 논문의 설명에 따르면 refinement module은 두 입력을 받는다.

첫 번째 입력은 이전 stage에서 온 coarse label map $R_f$이고, 두 번째 입력은 encoder 쪽 skip feature map $M_f$이다. 이 둘은 spatial size는 같지만 channel 수는 다르다. 따라서 먼저 $M_f$에 $3 \times 3$ convolution, batch normalization, ReLU를 적용해 채널 수를 맞춘 feature $m_f$를 만든다. 그런 다음 coarse label map과 이 skip feature를 결합한다. 그림 설명에는 concatenation으로 표현되지만, 텍스트 중 일부는 $(R+m)_f$처럼 덧셈 표기를 사용하고 있어 OCR 추출 과정에서 표기가 섞였을 가능성이 있다. 다만 핵심은 coarse label 정보와 finer encoder feature를 함께 사용한다는 점이다. 이후 다시 $3 \times 3$ convolution을 적용해 class-channel 형태의 prediction map을 만들고, 이를 bilinear upsampling으로 2배 키워 다음 stage로 넘긴다.

논문 본문은 이후 stage를 다음과 같이 요약한다.

$$
s_k(I) = conv_{3 \times 3}\big(concat(upsample(s_{k-1}(I)), f_{7-k}(I))\big)
$$

$$
\ell_k = Loss(R_k(Y), softmax(s_k(I))), \quad k = 2, \dots, 6
$$

추출 텍스트의 식 (2)에는 $upsample(s_k(I))$처럼 보이는 부분이 있으나, 문맥상 이전 stage 출력 $s_{k-1}(I)$를 upsample하는 의미로 이해하는 것이 자연스럽다. 실제 설명도 “previous segmentation map”을 upsample한다고 말한다. 중요한 점은 각 단계마다 더 큰 spatial resolution의 label map을 만들고, 이에 맞춰 resize한 ground truth $R_k(Y)$로 supervision을 준다는 것이다.

최종 학습 목표는 모든 단계의 loss 합이다.

$$
\sum_{k=1}^{6} \ell_k
$$

이 multi-loss 학습은 decoder의 각 단계가 독립적으로도 의미 있는 segmentation을 내도록 강제한다. 따라서 학습 초기에 gradient가 더 직접적으로 전달되고, 네트워크는 coarse semantic prediction과 fine detail recovery를 함께 배우게 된다.

직관적으로 보면, encoder의 마지막 feature는 물체가 “무엇인지”는 잘 담고 있지만 “정확히 어디까지인지”는 부족하다. 반면 얕은 encoder feature는 spatial detail은 풍부하지만 semantic abstraction은 약하다. LRN은 coarse label hypothesis를 유지한 채, 각 단계에서 더 세밀한 feature를 주입해 경계와 작은 구조를 복원한다. 논문 Fig. 3과 Fig. 5는 실제로 $s_1$에서 $s_6$으로 갈수록 사람 다리, 양의 다리, pole 같은 구조가 점점 복원되는 모습을 보여준다.

## 4. 실험 및 결과

논문은 PASCAL VOC 2012, CamVid, SUN RGB-D의 세 데이터셋에서 실험한다. 구현은 Caffe를 사용했고, GTX Titan X GPU에서 학습 및 테스트했다. 공통 학습 설정은 mini-batch size 10, learning rate 0.001, momentum 0.9, weight decay 0.0005, 약 80,000 iterations이며, 50,000 iteration 후 learning rate를 0.1배로 줄인다. encoder는 pre-trained VGG-16에서 fine-tuning했다.

CamVid와 SUN RGB-D에서는 클래스 불균형이 커서 class balancing을 사용한다. 각 클래스 loss weight는 training set에서 클래스 빈도의 median을 해당 클래스 빈도로 나눈 비율로 계산한다. 이는 도로, 차량 등 특정 클래스가 지나치게 많은 경우를 완화하기 위한 것이다. CamVid와 SUN RGB-D 이미지는 $360 \times 480$으로 resize해 사용했고, PASCAL VOC 2012는 training 시 $320 \times 320$ random crop을 사용했다. test 시에는 $512 \times 512$ segmentation map을 생성한 뒤 crop했다고 적고 있다.

비교 기준선으로는 SegNet, 그리고 SegNet에 deep supervision만 추가한 SegNet+DS를 둔다. 이 비교는 중요하다. 왜냐하면 LRN의 개선이 단지 “multi-loss를 붙였기 때문”인지, 아니면 refinement architecture 자체의 효과인지 구분할 수 있기 때문이다.

PASCAL VOC 2012에서는 20개 object class와 background를 포함한 표준 설정을 사용했다. train 10,582장, val 1,449장, test 1,456장 구성을 따랐고, 최종 평가는 test set에서 mean IoU로 보고한다. 결과는 FCN-8s가 62.2 mean IoU, SegNet이 59.1, SegNet+DS가 61.1, LRN이 64.2를 기록했다. 즉 LRN은 SegNet 대비 5.1 point, SegNet+DS 대비 3.1 point, FCN-8s 대비 2.0 point 높다. 클래스별로도 bird 79.7, person 77.5, motorbike 82.1 등 여러 항목에서 강한 수치를 보인다. 반면 chair 21.9, sofa 40.3처럼 어려운 클래스는 여전히 낮다. 논문은 qualitative result를 통해 LRN이 object의 세부 형상과 누락된 부분을 더 잘 복원한다고 주장한다.

CamVid에서는 32개 semantic class 중 11개의 큰 클래스 그룹을 사용했다. 표 2에 따르면 SegNet의 mean IoU는 50.2, SegNet+DS는 53.7, LRN은 61.7이다. improvement 폭이 매우 크다. 특히 sign 61.2, pedestrian 82.1, pole 45.4, bicyclist 69.7처럼 작고 얇은 객체에서 향상이 두드러진다. 이는 논문이 강조하는 coarse-to-fine refinement의 장점과 정확히 맞닿는다. 도로 장면 이해에서 pole, sign-symbol 같은 클래스는 픽셀 수가 적고 구조가 가늘어 segmentation이 어렵기 때문에, 이 결과는 구조적 설계의 효과를 잘 보여준다.

SUN RGB-D에서는 depth 정보를 사용하지 않고 RGB만으로 학습/평가한다. 5,285 training images와 5,050 test images를 사용하며, 총 37개 indoor scene class를 다룬다. 배경은 class로 취급하지 않고 학습과 평가에서 무시한다. instance-wise label을 class-wise label로 변환해 사용했다. 표 3에 따르면 SegNet의 mean IoU는 26.3, SegNet+DS는 31.2, LRN은 33.1이다. 절대 수치는 낮지만, indoor scene segmentation의 난이도를 고려하면 baseline 대비 일관된 개선으로 볼 수 있다. class average 역시 SegNet 35.6, SegNet+DS 49.2, LRN 46.8로 표기되어 있는데, 표의 정렬이 일부 OCR 오류를 포함하고 있어 개별 수치 해석에는 약간 주의가 필요하다. 그러나 mean IoU 기준으로는 LRN이 가장 높다는 논문의 결론은 분명하다.

논문은 추가로 stage-wise ablation analysis를 수행한다. 핵심 질문은 “중간 stage의 coarse label map을 바로 upsample해도 충분한가?”이다. 표 4에 따르면 그렇지 않다. PASCAL VOC 2012 validation에서 $s_1$은 58.4, $s_6$은 62.8이다. CamVid에서는 $s_1$ 50.9에서 $s_6$ 61.7까지 상승하고, SUN RGB-D에서는 $s_1$ 31.0에서 $s_6$ 33.1까지 오른다. 즉 decoder가 진행될수록 성능이 꾸준히 향상되며, coarse-to-fine refinement가 실제로 유효함을 보여준다. 논문은 이 결과를 통해 final prediction을 단순 upsampling이 아니라 다단계 refinement로 만드는 설계의 필요성을 뒷받침한다.

정리하면, 실험은 세 가지 메시지를 전달한다. 첫째, deep supervision만으로도 SegNet보다 개선된다. 둘째, 그 위에 refinement module을 추가한 LRN이 더 크게 개선된다. 셋째, 그 개선은 특히 작은 객체, 얇은 구조, 경계 복원 같은 세밀한 영역에서 강하게 나타난다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정과 architecture가 잘 맞물린다는 점이다. semantic segmentation에서 coarse semantic understanding과 fine spatial localization이 모두 필요하다는 점을 명확히 짚고, 이를 decoder 내부의 progressive label refinement로 직접 구현했다. 많은 방법이 skip feature를 쓰더라도 최종 출력은 마지막에만 내는 반면, 이 논문은 intermediate output을 실제 label map으로 취급하고 단계별 supervision을 부여했다. 이 설계는 해석 가능성도 높다. 각 stage 출력 $s_1$부터 $s_6$까지 시각화하면 refinement가 실제로 어떻게 진행되는지 관찰할 수 있다.

또 다른 강점은 ablation이 비교적 설득력 있게 구성되어 있다는 점이다. SegNet과의 비교만 했다면 개선 원인이 모호할 수 있었지만, SegNet+DS를 추가해 multi-loss 자체의 효과와 refinement architecture의 효과를 어느 정도 분리했다. 실험 결과를 보면 deep supervision만으로도 성능이 오르고, refinement module이 추가되면 추가 개선이 발생한다. 이는 제안 방법의 각 요소가 실제로 기여한다는 점을 잘 보여준다.

실험적으로도 CamVid처럼 작은 객체와 얇은 구조가 중요한 데이터셋에서 개선 폭이 크다는 점은 의미가 있다. 논문이 주장하는 장점이 가장 필요한 영역에서 실제로 성능 향상이 나타났기 때문이다. 또한 VOC, CamVid, SUN RGB-D처럼 성격이 다른 세 데이터셋에서 일관된 개선을 보였다는 점은 구조의 일반성을 어느 정도 지지한다.

한계도 분명하다. 첫째, 제안 구조는 기본적으로 VGG16 기반 encoder와 bilinear upsampling, skip connection을 조합한 2015~2017년대 encoder-decoder 계열의 연장선에 있다. 따라서 현대 기준에서는 feature fusion 방식이 비교적 단순하다. refinement module 내부에서 attention, learned upsampling, boundary-aware loss 같은 더 정교한 요소는 없다. 물론 이는 논문 시점의 맥락을 고려해야 하지만, 구조 자체가 매우 강력한 표현 학습을 하지는 않는다.

둘째, 수식과 설명 일부가 엄밀하지 않다. 제공된 텍스트 기준으로는 식 (2)의 인덱스 표기, refinement module에서 concatenation인지 summation인지가 완전히 일관되지 않는다. 실제 원 PDF에서는 더 명확할 가능성이 있지만, 최소한 추출 텍스트 수준에서는 표기 정합성이 약간 떨어진다. 이는 구현 재현성 측면에서 작은 약점이다.

셋째, 논문은 coarse-to-fine 구조의 효과를 잘 보여주지만, 계산량이나 메모리 비용에 대한 분석은 거의 제공하지 않는다. decoder 각 단계에서 intermediate prediction과 loss를 두기 때문에 학습 비용이 늘 수 있는데, 이에 대한 정량적 논의는 없다. 또한 당시 강력한 baseline인 CRF 결합형 방법이나 더 다양한 decoder 구조와의 폭넓은 비교도 제한적이다.

넷째, SUN RGB-D의 경우 depth 정보를 전혀 사용하지 않았다. 이는 RGB-only setting으로서 공정한 선택일 수 있지만, dataset의 강점을 충분히 활용하지는 않았다. 따라서 indoor scene understanding 전반의 최선 성능을 노렸다기보다, architecture의 일반성을 보여주는 데 초점을 맞췄다고 보는 편이 적절하다.

비판적으로 보면, LRN의 핵심 공헌은 segmentation을 “반복적 refinement problem”으로 재정의한 점에 있다. 다만 이 refinement가 엄밀한 recurrent modeling이나 probabilistic iterative optimization이라기보다는, skip-connected multi-stage decoder에 deep supervision을 부여한 구조적 변형으로 구현되어 있다. 즉 개념은 강하지만, 기계적 구현은 비교적 단순하다. 그럼에도 불구하고 실험 개선이 일관되기 때문에, 당시 semantic segmentation architecture 발전 맥락에서는 충분히 의미 있는 기여라고 평가할 수 있다.

## 6. 결론

이 논문은 semantic segmentation을 한 번에 예측하는 대신, coarse label map을 시작점으로 삼아 finer resolution에서 점진적으로 정제하는 Label Refinement Network를 제안한다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, segmentation 자체를 coarse-to-fine refinement 문제로 바라보는 관점을 제시했다. 둘째, 각 단계 출력이 실제 label map이 되도록 설계하고, 여러 해상도에서 loss를 주는 end-to-end architecture를 만들었다. 셋째, PASCAL VOC 2012, CamVid, SUN RGB-D에서 baseline보다 일관되게 더 나은 성능을 보였다.

실제 적용 관점에서 이 연구는 경계 복원, 작은 객체 분할, 고해상도 detail recovery가 중요한 segmentation 문제에 유용한 방향을 제시한다. 또한 semantic segmentation뿐 아니라 depth estimation, edge detection, saliency prediction 같은 다른 pixel-wise labeling 문제에도 비슷한 coarse-to-fine refinement와 deep supervision 아이디어를 확장할 여지가 있다. 후속 연구 관점에서도, intermediate prediction을 단순 feature가 아닌 의미 있는 structured output으로 두고 이를 반복적으로 수정하는 방식은 이후 segmentation decoder 설계에 중요한 통찰을 제공한다고 볼 수 있다.
