# Combining the Best of Convolutional Layers and Recurrent Layers: A Hybrid Network for Semantic Segmentation

- **저자**: Zhicheng Yan, Hao Zhang, Yangqing Jia, Thomas Breuel, Yizhou Yu
- **발표연도**: 2016
- **arXiv**: https://arxiv.org/abs/1603.04871

## 1. 논문 개요

이 논문은 semantic segmentation에서 널리 쓰이던 Fully Convolutional Network(FCN)의 한계를 보완하기 위해, convolutional layer와 recurrent layer를 결합한 hybrid architecture인 **H-ReNet**을 제안한다. 저자들의 문제의식은 명확하다. FCN은 convolution과 pooling을 여러 층 쌓아 receptive field를 점점 넓히지만, 먼 거리의 문맥 정보(long-range contextual dependence)를 직접적으로 모델링하지는 못한다. 즉, 이론적으로 receptive field가 커질 수는 있어도, 실제로 이미지의 서로 멀리 떨어진 영역 사이의 관계가 어떻게 전달되고 결합되는지는 불분명하다는 것이다.

이 문제는 semantic segmentation에서 특히 중요하다. 어떤 픽셀의 클래스는 그 주변 작은 패치만 보고는 판단하기 어려운 경우가 많고, 이미지 전체의 장면 구조나 멀리 떨어진 영역의 단서가 필요할 수 있다. 논문은 이런 전역 문맥(global context)을 보다 명시적으로 다루기 위해, 이미지 위를 수직 및 수평 방향으로 훑는 spatially recurrent layer, 즉 **ReNet layer**를 도입한다.

저자들은 먼저 ReNet layer만 여러 개 쌓은 **N-ReNet**을 만들어 Stanford Background에서 경쟁력 있는 성능을 보였고, 이후 pretrained FCN 위에 ReNet layer group을 얹은 **H-ReNet**을 제안하여 PASCAL VOC 2012에서 당시 강력한 방법들과 비교해 우수한 결과를 보고한다. 논문의 핵심 주장은, local feature extraction에는 convolution이 강하고, global context propagation에는 recurrent structure가 강하므로, 둘을 결합하면 semantic segmentation 표현력이 유의미하게 좋아진다는 것이다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 직관적이다. convolutional layer는 지역적인 패턴을 추출하는 데 강하지만, 이미지 전체를 가로지르는 문맥 전달은 간접적이다. 반대로 recurrent layer는 순차적 또는 구조적 정보를 따라 장거리 의존성을 직접 전파하는 데 적합하다. 저자들은 이 장점을 2차원 이미지에 맞게 활용하기 위해, 한 방향으로만 순차 처리하는 일반 RNN이 아니라 **이미지 격자를 수평/수직으로 스캔하는 1D RNN 두 개를 양방향으로 적용**하는 ReNet layer를 사용했다.

이 설계의 중요한 점은 두 가지다. 첫째, 수직 방향과 수평 방향의 recurrent sweep를 조합하면 각 위치의 feature가 사실상 **전체 이미지 영역을 참조하는 full-image receptive field**를 가지게 된다. 둘째, 그래프 모델을 별도로 붙여 후처리하는 방식과 달리, 이 recurrent layer는 네트워크 내부에 포함되어 **end-to-end training**이 가능하다.

기존 FCN 기반 segmentation 방법들은 boundary refinement를 위해 CRF 같은 graphical model을 붙이는 경우가 많았다. 논문은 그런 접근과 달리, 먼저 feature representation 자체를 개선하는 방향을 택한다. 즉, 픽셀 간 상호작용을 사후적으로 보정하는 대신, 중간 표현 단계에서 전역 문맥이 이미 반영된 representation을 만들도록 설계한 것이다. 저자들은 이 점을 H-ReNet의 차별점으로 내세운다.

## 3. 상세 방법 설명

논문이 제안한 기본 구성 요소는 **ReNet layer**다. 입력은 크기 $H \times W$의 이미지 또는 feature map $I$이며, 이를 $s \times t$ 크기의 patch들로 나누어 $h \times w$ 격자를 만든다. 여기서 $h = \lceil H/s \rceil$, $w = \lceil W/t \rceil$이다. 각 ReNet layer는 한 축을 따라 두 개의 1D RNN이 반대 방향으로 스캔한다. 예를 들어 vertical sweep에서는 위에서 아래로 가는 forward RNN과 아래에서 위로 가는 backward RNN이 동시에 동작한다.

논문은 RNN 구현으로 LSTM을 사용한다. 이유는 vanishing gradient 문제를 줄이고, forget gate, input gate, output gate, cell memory를 통해 정보를 선택적으로 잊고 저장하고 노출할 수 있기 때문이다. 각 위치 $(y, x)$에서의 업데이트는 다음과 같이 표현된다.

$$
(h^F_{y,x}, C^F_{y,x}) = LSTM^F(I_{y,x}, h^F_{y-1,x}, C^F_{y-1,x})
$$

$$
(h^B_{y,x}, C^B_{y,x}) = LSTM^B(I_{y,x}, h^B_{y+1,x}, C^B_{y+1,x})
$$

여기서 $F$와 $B$는 forward, backward를 뜻한다. 두 방향의 hidden state를 concat하면 각 위치마다 $2d$ 차원의 출력을 얻게 되고, 이 출력은 적어도 같은 column 전체의 문맥을 반영한다. 여기에 서로 직교하는 방향의 ReNet layer를 하나 더 쌓으면, 수직과 수평 방향 문맥이 모두 합쳐져 출력 feature는 전체 이미지 정보를 담게 된다. 논문은 이렇게 **서로 직교하는 두 ReNet layer를 하나의 recurrent layer group**이라고 부른다.

이 구성으로부터 두 가지 모델이 나온다.

첫째, **N-ReNet**은 recurrent layer group만 여러 층 쌓아 만든 순수 recurrent segmentation network다. 마지막에는 클래스 수에 맞추기 위한 $1 \times 1$ convolution layer를 두고, bilinear interpolation으로 원래 해상도로 upsampling하여 dense prediction을 만든다. Stanford Background 실험에서는 3개의 recurrent layer group을 사용했고, 첫 번째 group은 $2 \times 2$ patch를 스캔하여 공간 해상도를 4배 줄였고, 이후 group들은 $1 \times 1$ patch 단위로 스캔했다.

둘째, **H-ReNet**은 pretrained FCN의 상단에 recurrent layer group을 결합한 hybrid architecture다. 논문에서 사용한 baseline FCN은 VGG-16을 segmentation용으로 변형한 구조이며, `pool4`와 `pool5`의 stride를 줄이고 dilated convolution을 사용하여 출력 downsampling factor를 32에서 8로 줄였다. 그 위에 `conv6`, `conv7`, `conv8`을 쌓고 bilinear upsampling으로 최종 출력을 복원한다. H-ReNet은 이 baseline에서 `conv7`과 `conv8` 사이에 하나의 recurrent layer group `renet1`을 삽입한다. 즉, convolutional layers가 local feature를 만들고, ReNet layer가 전역 문맥을 전파한 뒤, 마지막 classifier가 segmentation map을 예측하는 구조다.

논문은 여기에 **multi-layer feature combination**도 추가로 실험했다. `pool4`, `pool5`, `conv7`의 feature map을 concat한 뒤 ReNet layer에 넣는 방식이다. 다만 서로 다른 층의 feature magnitude 차이 때문에 정규화가 필요한데, 저자들은 L2 normalization과 batch normalization을 비교했고, 실제로는 **batch normalization이 효과적**이었다고 보고한다.

학습은 pixelwise multinomial cross entropy를 사용하며, 클래스별 가중치는 동일하게 둔다. 즉, 각 위치의 정답 클래스를 기준으로 softmax loss를 계산한다. 논문은 ReNet layer가 완전히 미분 가능하므로 전체 네트워크를 standard backpropagation으로 end-to-end 학습할 수 있다고 설명한다. 테스트 시에는 입력 크기를 고정할 필요가 없고 원본 해상도의 이미지를 그대로 넣을 수 있으며, convolution과 ReNet LSTM layer 모두 variable-size 입력을 처리할 수 있다.

추론 후 boundary를 더 정교하게 만들기 위해 **DenseCRF**를 후처리로 붙인 버전도 실험했다. 여기서 unary potential은 네트워크가 예측한 negative log-probability이고, pairwise potential은 spatial kernel과 bilateral kernel의 조합이다. 중요한 점은 ReNet layer와 DenseCRF가 비슷한 일을 하는 것이 아니라, 서로 다른 수준의 상호작용을 다룬다는 것이다. 논문에 따르면 ReNet은 high-level feature를 바탕으로 지역 간 문맥 의존성을 모델링하고, DenseCRF는 위치와 intensity 같은 low-level 정보를 바탕으로 픽셀 쌍 상호작용을 모델링한다.

## 4. 실험 및 결과

실험은 두 갈래로 진행된다. N-ReNet은 **Stanford Background**에서, H-ReNet은 **PASCAL VOC 2012**에서 평가된다.

Stanford Background는 715장의 outdoor scene 이미지와 8개 라벨로 구성되어 있으며, 저자들은 5-fold cross validation 결과를 보고한다. 여기서 N-ReNet은 pixel accuracy 80.4%, class accuracy 71.8%, 추론 시간 0.07초(GPU)를 기록했다. 이는 nonparametric 계열이나 일부 CNN 기반 방법보다 정확도와 속도 모두에서 인상적인 결과다. 예를 들어 Deep 2D LSTM 대비 pixel/class accuracy가 각각 1.8%, 3.0% 높고, 실행 속도는 18배 이상 빠르다고 주장한다. 논문은 ReNet layer의 계산이 두 방향 sweep에서 병렬화 가능하기 때문에 GPU에서 효율적이라고 설명한다.

또한 N-ReNet의 중간 feature map을 시각화한 결과, 초반 층에서는 창문, 문 같은 세부 구조가 남아 있고, 깊은 층으로 갈수록 이런 세부가 부드럽게 정리되며 semantic region inference에 유리한 high-level feature로 바뀌는 경향을 관찰했다고 한다. 저자들은 이것을 CNN과 유사한 **hierarchical feature representation**의 증거로 해석한다.

PASCAL VOC 2012에서는 augmented dataset을 사용했으며, train 10,582장, val 1,449장, test 1,456장으로 기술되어 있다. 평가 지표는 mean Intersection over Union, 즉 mean IoU다. baseline FCN은 val set에서 63.4% mean IoU를 기록했다. 여기에 ReNet layer group 하나만 추가한 H-ReNet은 70.0%로 **무려 6.6%p 향상**되었다. 이는 구조 변화의 핵심이 recurrent layer group 하나뿐이라는 점을 고려하면 매우 큰 개선이다.

여기에 batch normalization을 추가하면 70.4%, `pool4`, `pool5`, `conv7` feature를 결합하는 multi-layer feature combination까지 적용하면 71.1%가 된다. baseline에 DenseCRF를 붙이면 67.5%, H-ReNet + multi-layer feature combination에 DenseCRF를 붙이면 72.6%까지 오른다. 즉, ReNet layer만으로도 큰 폭의 향상이 있고, 그 위에 normalization, multi-layer feature, CRF refinement가 추가로 보완 효과를 낸다.

세부 분석도 흥미롭다. 먼저 RNN unit을 LSTM 대신 IRNN으로 바꾼 경우 mean IoU가 67.2%로 떨어져, LSTM 기반 H-ReNet보다 3.2% 낮았다. 이는 gating과 memory mechanism이 실제로 중요했음을 보여준다. 또 feature source를 바꿔본 실험에서 `pool4`만 입력하면 59.2%, `pool5`만 쓰면 68.3%, `conv7`만 쓰면 70.4%였다. 즉, deeper feature가 더 유용했지만, 여기에 `pool5`, `pool4`를 추가로 concat하면 70.9%, 71.1%로 조금씩 향상되어 서로 보완적이라는 점도 확인했다.

training crop size에 대한 실험도 논문의 주장을 뒷받침한다. baseline FCN은 $(256,320)$, $(320,400)$, $(400,500)$ 사이에서 IoU가 거의 변하지 않았다. 반면 H-ReNet은 각각 67.3%, 69.3%, 71.1%로 커졌다. 저자들은 큰 crop을 사용할수록 ReNet layer가 더 긴 거리의 문맥을 전달하게 되어 representation이 좋아진다고 해석한다.

VOC12 validation set에서 타 방법과의 비교를 보면, H-ReNet은 DeepLab-LargeFOV의 62.3%보다 8.8% 높고, DeepParsing의 67.8%보다 3.3% 높다. Piecewise는 70.3%로 H-ReNet 71.1%보다 낮고, CRFasRNN은 69.6%다. 속도 측면에서도 baseline FCN이 0.21초, H-ReNet이 0.24초로, recurrent layer group 추가 비용은 0.03초에 불과하다. DenseCRF까지 붙이면 0.46초가 되어 정확도와 속도 사이의 trade-off가 생긴다.

VOC12 test set에서는 H-ReNet이 72.7%, H-ReNet + DenseCRF가 74.3%를 기록했다. 논문은 이것이 Piecewise, CRFasRNN, DeepParsing 등을 능가하거나 비슷한 수준이며, DenseCRF를 붙인 버전은 DeepParsing보다 0.2% 높아 새로운 state-of-the-art라고 주장한다. 20개 클래스 중 13개 클래스에서 최고 IoU를 기록했다고도 밝힌다. 다만 bike 클래스에서는 DeepParsing이 59.4%, H-ReNet + DenseCRF가 39.6%로 크게 뒤처졌는데, 논문은 이 차이를 DeepParsing이 고차 label relation을 더 잘 모델링하기 때문이라고 해석한다.

MS COCO 추가 데이터까지 사용한 경우 H-ReNet + DenseCRF는 76.8%를 기록했지만, DeepParsing의 77.5%보다는 낮았다. 저자들은 충분히 큰 데이터가 주어졌을 때 DeepParsing이 모델링하는 higher-order relation의 이점이 더 드러났을 가능성을 언급한다.

정성적 비교에서도 H-ReNet은 더 coherent한 region prediction과 더 완전한 object body segmentation을 보여준다고 보고된다. 특히 ambiguous region을 해석할 때 이미지의 넓은 영역을 함께 보는 능력이 도움이 된다고 논문은 주장한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 해결 방식이 잘 맞아떨어진다는 점이다. semantic segmentation에서 장거리 문맥이 중요하다는 문제를 제기하고, 이를 위해 이미지 전역을 직접 스캔하는 recurrent layer를 제안했으며, 실제로 baseline FCN 대비 큰 성능 향상을 수치로 보여준다. 특히 H-ReNet은 full-image receptive field, end-to-end training, 효율적 GPU 실행이라는 세 가지 특성을 동시에 강조하는데, 실험 결과도 이 주장을 상당 부분 뒷받침한다.

또 다른 강점은 ReNet layer를 완전히 새로운 독립 모델(N-ReNet)로도 시험하고, 기존 강력한 FCN에 결합한 hybrid(H-ReNet)로도 시험했다는 점이다. 덕분에 ReNet layer 자체의 유효성과, 기존 convolutional representation을 개선하는 모듈로서의 가치가 모두 드러난다. ablation study도 비교적 충실하다. batch normalization, multi-layer feature combination, IRNN 대 LSTM, crop size, DenseCRF 효과 등을 따로 떼어 검증해 각 요소의 기여를 살폈다.

다만 한계도 분명하다. 첫째, boundary localization은 좋아졌다고 하지만, 본질적으로 feature map 해상도가 낮아지는 구조 자체는 여전히 유지되기 때문에, 최종 boundary 품질은 DenseCRF 같은 후처리에 의존하는 부분이 있다. 논문도 H-ReNet만으로는 여전히 blurry boundaries가 생긴다고 인정한다. 둘째, bike 클래스 사례에서 보이듯이, 전역 문맥 전파만으로는 해결되지 않는 **higher-order label relation**이나 구조적 prior의 한계가 남아 있다.

셋째, ReNet layer가 왜 정확히 이런 수준의 향상을 주는지에 대한 메커니즘 분석은 제한적이다. 논문은 “explicit context propagation”이라는 설명을 제시하지만, 어떤 종류의 문맥 관계가 실제로 잘 학습되는지까지는 깊게 파고들지 않는다. 넷째, 실험 비교는 당시 기준으로 충분히 강력하지만, 일부 경쟁 방법은 superpixel이나 별도 graphical model을 사용하고, 일부는 추가 데이터 사용 여부가 다르므로 완전히 동일 조건 비교는 아니다. 물론 논문도 COCO 추가 실험을 따로 분리해 제시하고 있다.

비판적으로 보면, 이 논문은 FCN의 문맥 모델링 한계를 정확히 짚었고 실제로 성능을 끌어올렸지만, 이후 연구 흐름의 관점에서 보면 이는 “전역 문맥을 모델링하기 위한 하나의 설계”이지 최종 해답은 아니다. 특히 구조적 관계를 얼마나 풍부하게 담아낼 수 있는지, fine-grained boundary와 small structure를 얼마나 유지할 수 있는지는 여전히 미해결 문제로 남아 있다. 논문 자체도 trainable CRF-based post-processing과의 더 강한 결합을 future work로 남긴다.

## 6. 결론

이 논문은 semantic segmentation에서 convolutional representation과 recurrent context propagation을 결합한 **H-ReNet**을 제안하고, spatially recurrent layer인 **ReNet layer**가 전역 문맥을 명시적으로 전달하는 데 효과적임을 보였다. 순수 recurrent 구조인 N-ReNet도 경쟁력 있는 성능을 보였고, pretrained FCN 위에 ReNet layer group을 추가한 H-ReNet은 PASCAL VOC 2012에서 당시 최고 수준의 결과를 달성했다.

핵심 기여는 세 가지로 요약할 수 있다. 첫째, semantic segmentation에 spatially recurrent layer를 본격적으로 도입했다. 둘째, convolution이 local feature를, recurrence가 global context를 담당하는 hybrid design을 제시했다. 셋째, 이 구조가 정확도 향상뿐 아니라 실용적인 계산 효율도 유지할 수 있음을 실험으로 보였다.

실제 적용 관점에서 보면, 이 연구는 segmentation에서 단순히 receptive field를 넓히는 것과 전역 문맥을 **직접 전파하는 것**이 다를 수 있음을 보여준 사례다. 향후 연구에서는 이 아이디어가 더 정교한 structured prediction, high-order relation modeling, boundary refinement 모듈과 결합될 수 있으며, 논문도 바로 그 방향을 자연스러운 후속 과제로 제시하고 있다.
