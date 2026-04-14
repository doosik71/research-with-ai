# Feedforward semantic segmentation with zoom-out features

- **저자**: Mohammadreza Mostajabi, Payman Yadollahpour, Gregory Shakhnarovich
- **발표연도**: 2014
- **arXiv**: https://arxiv.org/abs/1412.0774

## 1. 논문 개요

이 논문은 semantic segmentation을 복잡한 structured prediction 문제로 다루는 기존 관행에서 벗어나, **superpixel 단위의 feedforward classification**으로 해결할 수 있는지 탐구한다. 저자들의 핵심 주장은, 각 superpixel을 아주 좁은 local 정보만으로 분류하면 부족하지만, 그 주변을 점점 넓혀 가며 얻은 multi-scale context를 함께 사용하면 별도의 CRF나 복잡한 inference 없이도 매우 강력한 segmentation 성능을 얻을 수 있다는 점이다.

연구 문제는 명확하다. semantic segmentation에서는 어떤 픽셀의 라벨이 그 픽셀 자체의 색이나 질감만으로 결정되지 않는다. 주변 물체, 더 넓은 장면 구조, 이미지 전체의 scene context가 모두 영향을 준다. 기존 방법들은 이런 상호작용을 모델링하기 위해 CRF, hierarchical model, structured SVM 같은 복잡한 구조를 사용했지만, 이런 접근은 보통 inference와 learning이 어렵고 계산 비용도 크다. 이 논문은 그 구조적 의존성을 명시적으로 모델링하지 않고도, 입력 표현 자체를 잘 설계하면 충분히 흡수할 수 있다고 본다.

문제의 중요성도 크다. 당시 image classification과 object detection은 deep convolutional network의 도입으로 크게 발전했지만, segmentation은 상대적으로 개선 폭이 작았다. 저자들은 그 이유 중 하나가 segmentation이 structured prediction으로 고정적으로 이해되어 왔기 때문이라고 본다. 따라서 이 논문은 segmentation에도 deep representation의 장점을 직접 가져올 수 있는 간단하고 실용적인 경로를 제시한다는 점에서 의미가 있다.

## 2. 핵심 아이디어

중심 아이디어는 각 superpixel을 볼 때 한 가지 크기의 patch만 보지 않고, **점점 더 넓은 네 개의 spatial scope**에서 feature를 추출해 하나의 feature vector로 합치는 것이다. 논문은 이를 “zoom-out”이라고 부른다. 네 수준은 superpixel 자체를 보는 `local`, 그 주변의 작은 이웃을 보는 `proximal`, 더 넓은 영역을 보는 `distant`, 그리고 이미지 전체를 보는 `global`이다.

이 설계의 직관은 다음과 같다. local feature는 경계 근처의 세부 appearance와 texture를 잘 잡아주고, proximal feature는 인접 영역과의 혼합된 통계를 담아 object boundary 주변의 단서를 준다. distant feature는 물체의 큰 부분이나 물체들 사이의 배치를 포착해 shape와 context를 이해하게 하며, global feature는 지금 장면이 실내인지 초원인지 같은 scene-level prior를 제공한다. 즉, segmentation의 구조적 성질을 explicit graphical model로 쓰지 않고, **표현 학습과 컨텍스트 결합**으로 우회하는 방식이다.

기존 접근과의 가장 큰 차이는 두 가지다. 첫째, 복잡한 pairwise 또는 higher-order potential을 가진 structured model을 두지 않는다. 둘째, region proposal 기반 파이프라인처럼 후보 영역 생성과 재순위를 여러 단계로 수행하지 않는다. 대신 이미지 over-segmentation 이후에는 각 superpixel을 독립적으로 feedforward network에 넣어 분류한다. 저자들은 이 독립 분류가 실제로는 독립적이지 않다고 본다. 왜냐하면 인접 superpixel들의 proximal/distant/global 영역이 많이 겹치므로, feature 수준에서 이미 “implicit smoothness”와 “soft global constraint”가 들어가기 때문이다.

또 하나 중요한 아이디어는 **asymmetric loss**이다. segmentation 데이터는 클래스 불균형이 심하다. 흔한 클래스와 드문 클래스가 매우 다르기 때문에, 일반적인 대칭적 log-loss를 그대로 쓰면 희귀 클래스 성능이 쉽게 희생된다. 저자들은 모든 데이터를 버리지 않고 그대로 사용하되, 각 샘플의 loss를 해당 클래스 빈도의 역수로 가중하는 방식을 사용한다. 이 선택이 성능에 큰 영향을 준다고 실험으로 보여준다.

## 3. 상세 방법 설명

전체 파이프라인은 비교적 단순하다. 먼저 입력 이미지를 약 500개의 SLIC superpixel로 분할한다. 각 superpixel $s$에 대해 네 수준의 zoom-out region을 정의하고, 각 수준에서 feature를 계산한 뒤 모두 이어 붙여 하나의 표현으로 만든다. 그 다음 이 벡터를 multilayer neural network에 넣어 superpixel의 클래스를 예측한다. 최종 pixel label은 각 픽셀이 속한 superpixel의 label로 정해진다.

논문에서 feature vector는 다음과 같이 정의된다.

$$
\phi_{\text{zoom-out}}(s, I)=
\begin{bmatrix}
\phi_{\text{loc}}(s, I) \\
\phi_{\text{prox}}(s, I) \\
\phi_{\text{dist}}(s, I) \\
\phi_{\text{glob}}(I)
\end{bmatrix}
$$

여기서 $\phi_{\text{loc}}$는 superpixel 자체의 특징, $\phi_{\text{prox}}$는 주변 작은 이웃 영역의 특징, $\phi_{\text{dist}}$는 더 큰 문맥 영역의 특징, $\phi_{\text{glob}}$는 이미지 전체 특징이다.

논문은 각 zoom level의 역할을 꽤 구체적으로 설명한다. `local`은 색, texture, 미세 패턴처럼 국소 evidence를 담는다. 이 수준에서는 인접 superpixel끼리 feature가 크게 다를 수 있다. `proximal`은 반경 2의 superpixel 이웃 집합으로 정의되며, 평균적으로 약 $100 \times 100$ 픽셀 크기다. 여기서는 object boundary를 가로지르는 주변 appearance 분포를 더 잘 볼 수 있다. `distant`는 3차 이웃까지 포함한 superpixel들의 bounding box로 정의되고 평균 크기는 약 $170 \times 170$ 픽셀이다. 이 수준은 물체의 큰 부분, 경우에 따라 물체 전체, 그리고 서로 다른 클래스의 공간적 배치를 보게 해준다. `global`은 이미지 전체이며, 모든 superpixel이 동일한 global feature를 공유한다.

local과 proximal에서는 hand-crafted feature를 사용한다. local feature는 color histogram, adaptive binning histogram, entropy, texton histogram, SIFT bag-of-words, 그리고 superpixel 위치 정보를 포함한다. 구체적으로 color 관련 특징은 L*a*b 채널별 histogram과 entropy를 포함하고, texture는 64-texton dictionary 기반 histogram과 entropy를 쓴다. SIFT는 여러 patch 크기와 채널에 대해 계산된 descriptor를 500 visual words dictionary에 할당해 histogram으로 만든다. 여기에 image center 기준 정규화 좌표를 더한다. 또한 별도로 작은 local convnet도 학습해 softmax 출력을 21차원 feature로 사용하고, foreground/background 이진 분류 network의 출력 2차원도 추가한다.

proximal level에서는 local과 같은 종류의 hand-crafted feature를 사용하며, 총 1818차원이라고 적고 있다. 반면 `distant`와 `global`에서는 ImageNet으로 사전학습된 deep convnet의 마지막 fully connected layer activation을 사용한다. 초기에는 CNN-S를 썼고, 이후에는 더 깊은 VGG-16을 사용했다. 각 distant region 또는 전체 이미지를 $224 \times 224$로 resize한 뒤 네트워크에 넣고, 마지막 FC layer의 4096차원 activation을 feature로 채택한다. 중요한 점은 이 convnet들을 VOC 데이터에 fine-tuning하지 않았다는 것이다.

분류기는 superpixel feature vector를 입력으로 받는 softmax 또는 multilayer neural network다. 논문은 linear softmax, 2-layer MLP, 3-layer MLP를 비교했고, 최종적으로 VGG-16 distant/global feature 위에 hidden unit 1024개를 둔 2-layer network가 가장 좋았다고 보고한다. hidden activation은 ReLU이며, 일부 deeper model에는 dropout도 시험했지만 최선은 아니었다.

학습 목표에서 핵심은 클래스 불균형 보정이다. 저자들은 일반 log-loss를 다음과 같이 수정한다.

$$
-\frac{1}{N}\sum_{i=1}^{N}\frac{1}{f_{y_i}}
\log \hat{p}(y_i \mid \phi(s_i, I_i))
$$

여기서 $f_c$는 클래스 $c$의 훈련 데이터 내 빈도이고, $\hat{p}(y_i \mid \phi(s_i, I_i))$는 모델이 정답 클래스에 부여한 확률이다. 즉, 드문 클래스일수록 loss weight가 커진다. 논문은 이것이 convexity를 바꾸지 않으며, backpropagation 코드에도 작은 수정만 필요하다고 설명한다.

학습 설정도 비교적 명확하다. 훈련 데이터는 VOC 원래 `train`과 추가 annotation 9,118장을 합쳐 사용했고, `val`은 validation 전용으로 사용했다. 전체 훈련 예시는 약 5백만 개 superpixel이다. learning rate는 `0.0001`, weight decay는 `0.001`이며, Caffe와 Tesla K40 GPU 한 대로 학습했다고 적는다.

한편 superpixel 단위로 라벨을 주는 것이 성능 상한을 크게 낮추지 않는지도 확인한다. 각 superpixel에 다수결로 ground truth class를 부여했을 때 `val`에서 94.4% 정확도가 나왔으므로, 당시 segmentation 방법들의 실제 성능보다 훨씬 높아, superpixelization 자체가 주요 병목은 아니라고 주장한다.

## 4. 실험 및 결과

주요 실험은 PASCAL VOC 2012 semantic segmentation benchmark에서 수행되었다. 클래스는 background를 포함해 21개다. 평가 지표는 각 클래스별 intersection-over-union인 IU를 구한 뒤 평균한 mean IU이다. test set은 1,456장이고, ground truth가 공개되지 않아 서버 제출로만 평가할 수 있다고 설명한다.

먼저 ablation study는 각 zoom-out level이 실제로 얼마나 중요한지 보여준다. linear softmax classifier 기준으로 local만 쓰면 mean accuracy가 14.6%, proximal만 쓰면 15.5%, local+proximal은 17.7%로 낮다. 하지만 distant가 들어가면 local+distant가 37.38%, global이 들어간 local+global은 41.8%, distant+global은 47.0%로 크게 올라간다. 즉, segmentation에서 넓은 문맥 정보가 절대적으로 중요하다는 점이 드러난다. 동시에 저자들은 local과 proximal이 없으면 경계 localization이 나빠진다고 말한다. 따라서 넓은 context만으로 충분한 것이 아니라, 정교한 경계와 scene context를 함께 써야 한다는 결론이 나온다.

가장 인상적인 결과는 asymmetric loss의 효과다. 전체 zoom-out feature를 다 사용하되 symmetric loss를 쓰면 20.4%에 그치고, asymmetric loss를 쓰면 52.4%까지 오른다. 이 수치는 논문에서 loss design이 단순 보조 요소가 아니라 사실상 성능을 결정하는 핵심 구성임을 보여준다.

다음으로 classifier 구조와 backbone convnet의 영향을 비교한다. CNN-S 기반일 때 linear 모델은 52.4%, 2-layer 256 hidden은 57.9%, 2-layer 512 hidden은 59.1%였다. 3-layer는 오히려 일반화가 덜 잘되었다. VGG-16으로 distant/global feature를 바꾸면 성능이 더 오른다. 2-layer 256 hidden은 62.3%, 512 hidden은 63.0%, 1024 hidden은 63.5%를 기록했다. 즉, 더 좋은 pre-trained representation과 적당한 깊이의 classifier가 핵심이다.

최종적으로 선택된 모델은 **VGG-16 기반 distant/global feature + 2-layer classifier + hidden unit 1024개**이며, VOC 2012 test에서 **64.4% mean IU**를 기록했다. 이는 논문 제출 시점 기준으로 이전 방법들보다 높다. 표에 따르면 FCN-8s는 62.2%, Hypercolumns는 59.2%, SDS는 51.6%, DivMBest+convnet은 52.2%였다. 이 논문은 VOC 2010과 2011 test에서도 각각 64.4%, 64.1%를 기록했다.

세부 클래스별로는 background 89.8, aeroplane 81.9, boat 78.2, car 80.5, chair 79.8, person 76.6 등 전반적으로 높았고, 특히 dog 74.0, cat 74.0, train 68.9 같은 클래스에서 큰 강점을 보였다고 저자들은 주장한다. 반면 bird 35.1, cow 22.4, potted plant 44.3, sofa 40.2 등 낮은 클래스도 있다. 논문은 20개 object category 중 15개에서 당시 최고 성능이라고 말한다.

정성적 결과에 대해서는, 많은 예시에서 객체 클래스와 장면 레이아웃을 잘 맞추고 과도하게 smooth한 CRF식 결과보다 더 세밀한 구조를 잡는다고 설명한다. 하지만 동시에 결과가 다소 **under-smoothed**해서 작은 “섬(island)” 형태의 잡음 클래스가 생기는 문제가 있다고 인정한다. 저자들은 고립된 작은 영역을 주변 label로 뒤집는 후처리 classifier를 시험해 `val`에서 약 0.5% 향상을 얻었지만, 이를 본 논문의 주 결과로 밀지는 않았다.

추가로 Stanford Background Dataset에서도 5-fold cross validation을 수행했다. 여기서는 local convnet 출력 차원과 classifier 크기만 축소해 같은 아키텍처를 적용했다. 결과는 pixel accuracy 82.1%, class accuracy 77.3%로, Multiscale convnet, Recurrent CNN, Pylon 등 비교 대상보다 높았다. VOC에서 직접 비교가 어려운 관련 연구들과의 비교 근거를 보강하기 위한 실험으로 해석할 수 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 구조적으로 복잡한 segmentation 문제를 surprisingly simple한 pipeline으로 풀어내면서도 매우 높은 성능을 냈다는 점이다. 특히 이 성능 향상이 단순한 engineering trick이 아니라, local부터 global까지 이어지는 representation 설계와 class imbalance를 반영한 loss 설계에서 나온다는 점을 실험으로 설득력 있게 보여준다. ablation 결과가 명확해서 각 구성 요소의 필요성이 잘 드러난다.

또 다른 강점은 deep convnet을 segmentation에 실질적으로 연결했다는 점이다. 논문 당시에는 classification과 detection에서는 convnet이 이미 강했지만 segmentation은 상대적으로 뒤처져 있었는데, 저자들은 segmentation을 structured inference가 아닌 feedforward classification으로 재구성함으로써 deep feature의 장점을 직접 가져왔다. 특히 global scene context를 segmentation에 자연스럽게 넣는 방식은 이후 문맥 기반 segmentation 연구와도 잘 연결된다.

실험 설계 측면에서도 장점이 있다. superpixel majority label의 상한 정확도 94.4%를 따로 계산해, superpixelization이 근본적 한계인지 아닌지를 먼저 점검했다. 또한 local/proximal/distant/global 각각의 기여, symmetric 대 asymmetric loss 차이, CNN-S 대 VGG-16 차이, classifier depth 차이를 모두 비교해 논문의 주장과 결과가 잘 맞물린다.

한계도 분명하다. 첫째, 이 방법은 여전히 superpixel preprocessing에 의존한다. superpixel 품질이 나쁘면 경계 정밀도가 손상될 수 있고, 본 방법은 end-to-end pixelwise learning이 아니다. 논문은 이 한계가 실험상 크지 않다고 말하지만, 원리적으로는 분명한 제약이다.

둘째, local과 proximal feature의 상당 부분이 hand-crafted이다. 색 histogram, texton, SIFT bag-of-words 등을 많이 사용하고 있어서, 전체 시스템이 완전히 learned representation 기반은 아니다. 저자들 스스로도 향후 local/proximal을 convnet feature로 바꾸겠다고 말한다. 따라서 이 논문은 완전한 deep segmentation framework라기보다, hand-crafted + deep feature의 과도기적 설계로 보는 편이 정확하다.

셋째, structured model을 쓰지 않는 대가로 출력이 under-smoothed해지는 문제가 있다. 논문이 보여주듯 작은 irrelevant region이 군데군데 생기며, 이는 명시적 pairwise consistency나 CRF-style refinement가 없기 때문으로 볼 수 있다. 저자들도 후처리나 unrolled inference 같은 보완 필요성을 인정한다.

넷째, distant/global convnet은 ImageNet pretrained network를 그대로 feature extractor로 쓰고 있으며 VOC에 fine-tuning하지 않았다. 이 선택은 방법의 일반성과 단순성을 보여주지만, 동시에 성능이 얼마나 더 오를 수 있었는지는 열어 둔다. 반대로 말하면 이 논문의 성과 일부는 강력한 외부 pretrained classifier 표현에 의존하고 있다.

비판적으로 보면, “structured prediction이 필요 없다”기보다는 “좋은 multiscale representation이 있으면 structured component의 필요성이 상당히 줄어든다”는 해석이 더 정확하다. 실제로 논문도 CRF를 완전히 부정하지 않고, 이후에는 prediction cleanup을 위한 structured inference가 다시 유용할 수 있다고 말한다. 따라서 이 논문의 진짜 기여는 structured modeling의 종말 선언이 아니라, segmentation에서 representation 설계의 힘을 분명히 보여준 데 있다.

## 6. 결론

이 논문은 semantic segmentation을 superpixel 분류 문제로 재해석하고, 각 superpixel에 대해 local, proximal, distant, global의 네 수준 문맥을 결합한 **zoom-out feature**를 설계함으로써, 별도의 복잡한 structured inference 없이도 당시 최고 수준의 성능을 달성했다. 여기에 class imbalance를 반영한 asymmetric loss와 multilayer classifier를 결합해 VOC 2012 test에서 64.4% mean IU를 기록했다.

핵심 기여는 세 가지로 요약할 수 있다. 첫째, segmentation의 구조적 의존성을 explicit CRF가 아니라 multiscale contextual feature 안에 암묵적으로 녹여낼 수 있음을 보였다. 둘째, deep convnet의 scene-level representation을 segmentation에 효과적으로 연결했다. 셋째, class imbalance 보정을 위한 단순한 weighted log-loss가 segmentation 정확도에 매우 큰 영향을 준다는 점을 실험으로 입증했다.

실제 적용 측면에서는, 이 방법이 복잡한 inference 없이도 강력한 성능을 보인다는 점에서 실용적 가치가 있다. 향후 연구 측면에서는 local/proximal도 fully learned feature로 대체하고, 전체 zoom-out 과정을 하나의 end-to-end network로 통합하며, 필요하다면 unrolled inference나 refinement module로 출력을 정리하는 방향으로 자연스럽게 확장될 수 있다. 즉, 이 논문은 segmentation에서 “구조”를 어떻게 다룰 것인가에 대한 관점을 representation 중심으로 이동시킨 중요한 전환점으로 볼 수 있다.
