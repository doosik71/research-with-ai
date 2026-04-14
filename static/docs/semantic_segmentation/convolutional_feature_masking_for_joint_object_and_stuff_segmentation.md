# Convolutional Feature Masking for Joint Object and Stuff Segmentation

- **저자**: Jifeng Dai, Kaiming He, Jian Sun
- **발표연도**: 2015
- **arXiv**: https://arxiv.org/abs/1412.1283

## 1. 논문 개요

이 논문은 semantic segmentation에서 shape 정보를 활용하는 방식을 다시 설계한 연구이다. 당시 강한 성능을 보이던 R-CNN 계열 방법들은 object proposal이나 segment proposal을 원본 이미지 위에서 직접 잘라내거나 마스킹한 뒤 CNN에 넣어 feature를 추출했다. 논문은 이 방식이 두 가지 문제를 갖는다고 지적한다. 첫째, 원본 이미지에 mask를 씌우면 인위적인 경계가 생기는데, 이런 경계는 ImageNet pretraining 단계에서 보지 못한 패턴이므로 feature 품질을 떨어뜨릴 수 있다. 둘째, 이미지 한 장마다 수천 개의 region에 대해 CNN을 반복 실행해야 하므로 매우 느리다.

논문의 핵심 문제는 다음과 같다. semantic segmentation에서 segment의 shape 정보를 유지하면서도, 원본 이미지에 직접 mask를 씌우지 않고, 그리고 훨씬 빠르게 feature를 추출할 수 있는가? 저자들은 이에 대해 convolutional feature map 상에서 직접 마스킹하는 방식인 Convolutional Feature Masking, 즉 CFM을 제안한다.

이 문제는 중요하다. semantic segmentation은 픽셀 단위 예측을 해야 하므로 object detection보다 더 세밀한 shape 정보가 필요하다. 그런데 shape 정보를 얻기 위해 매번 raw image를 잘라 쓰면 계산량이 폭증한다. 따라서 정확도와 속도를 동시에 개선하는 방식은 실용성과 학술적 가치가 모두 크다. 또한 이 논문은 object뿐 아니라 sky, grass, water 같은 stuff까지 같은 프레임워크에서 다루려 했다는 점에서도 의미가 있다.

## 2. 핵심 아이디어

중심 아이디어는 매우 명확하다. segment mask를 raw image에 적용하지 말고, 이미 한 번 계산된 convolutional feature map에 투영해서 그 위에서 masking하자는 것이다. 즉, CNN의 convolutional part는 전체 이미지에 대해 한 번만 수행하고, 각 proposal segment는 마지막 convolutional feature map 위의 binary mask로 바꿔 적용한다. 그러면 shape 정보는 유지하면서도, raw-image masking이 만드는 인공 경계 문제를 피할 수 있다.

기존 SDS 같은 방법과의 차별점은 shape-aware feature를 얻는 위치가 다르다는 점이다. 기존 방법은 입력 이미지나 잘라낸 이미지 패치 수준에서 shape를 반영했고, 이 논문은 feature hierarchy의 더 깊은 단계, 즉 convolutional activation 위에서 shape를 반영한다. 저자들은 이것이 더 효율적이며, pre-trained feature를 덜 훼손한다고 본다.

또 하나의 핵심 아이디어는 stuff를 단일 proposal로 보지 않는 것이다. object는 하나의 인스턴스가 비교적 명확한 경계를 갖지만, stuff는 넓고 불규칙하게 퍼져 있어서 하나의 box나 하나의 segment로 표현하기 어렵다. 따라서 저자들은 stuff를 여러 segment proposal의 compact combination으로 표현하고, 이를 위해 Segment Pursuit라는 샘플링 절차를 도입한다.

## 3. 상세 방법 설명

전체 파이프라인은 다음과 같다. 먼저 입력 이미지를 여러 scale로 resize한 뒤, 전체 이미지에 대해 convolutional layers를 한 번만 통과시켜 feature map을 만든다. 동시에 Selective Search나 MCG 같은 region proposal 알고리즘으로 약 2,000개의 proposal box와 그에 대응하는 segment를 얻는다. 이후 각 segment mask를 마지막 convolutional feature map 영역으로 투영하고, 이 mask를 이용해 해당 proposal의 segment feature를 만든다. 이 feature와 proposal box에서 얻은 regional feature를 함께 사용해 분류한다.

CFM layer의 동작은 논문의 가장 중요한 기술적 구성 요소이다. 각 convolutional activation은 입력 이미지의 receptive field에 대응하므로, 저자들은 먼저 각 activation을 그 receptive field 중심점으로 이미지 공간에 매핑한다. 그런 다음 원본 이미지의 binary segment mask의 각 픽셀을 가장 가까운 receptive field center에 할당한다. 이후 이 정보를 다시 feature map 위치로 되돌려, 각 feature map 위치가 segment 내부인지 아닌지 결정한다. 한 위치에 여러 픽셀이 대응될 수 있으므로, binary 값을 평균한 뒤 0.5 threshold를 적용해 최종 binary mask를 만든다. 마지막으로 이 binary mask를 feature map의 모든 channel에 곱한다. 결과적으로 segment 영역에 해당하는 activation만 남은 feature가 생성된다.

논문은 두 가지 network design을 제안한다.

Design A에서는 마지막 convolutional layer 뒤에서 두 갈래를 만든다. 하나는 SPP를 통해 bounding box 기반 regional feature를 추출하는 경로이고, 다른 하나는 CFM을 적용한 뒤 다시 SPP를 통해 segment feature를 추출하는 경로이다. 이후 두 경로의 fc output을 concatenate하여 classifier에 넣는다. 이 설계는 region과 segment 정보를 명시적으로 분리하지만, fc pathway가 두 개라 계산량이 더 크다.

Design B에서는 먼저 SPP를 수행하고, 그중 가장 세밀한 spatial level인 $6 \times 6$ feature map에 CFM을 적용한다. 이렇게 얻은 masked feature를 나머지 pyramid level과 concatenate하여 하나의 fc pathway만 사용한다. 이 방식은 계산 비용과 overfitting 위험을 줄인다. 실험에서는 Design A와 B의 성능이 비슷했고, 저자들은 이후 주로 Design B를 사용한다. 다만 VGG-16의 경우 저자 설명상 마지막 pooling layer를 사실상 단일-level SPP로 취급하므로 Design B를 적용하기 어렵고, Design A를 사용한다.

학습 절차는 2단계 fine-tuning과 SVM 학습으로 구성된다. 먼저 SPP-Net 방식으로 object detection용 network를 fine-tune한다. 그다음 CFM이 포함된 segmentation architecture로 다시 fine-tune한다. 이때 segment proposal이 ground-truth foreground segment와 IoU $[0.5, 1]$이면 positive, $[0.1, 0.3]$이면 negative로 둔다. 중요한 점은 여기서 IoU가 bounding box 기준이 아니라 segment area 기준이라는 것이다. fine-tuning 후에는 각 category별 linear SVM을 학습한다. SVM 단계에서는 positive sample로 ground-truth segment만 사용한다.

추론 시에는 각 proposal을 적절한 image scale에 할당하고, region feature와 segment feature를 추출한 뒤 SVM으로 점수를 계산한다. 최종 픽셀 예측은 SDS의 pasting scheme을 사용한다. 이 pasting scheme은 높은 점수의 proposal부터 순차적으로 선택하고, region refinement를 수행하고, 겹치는 proposal을 억제하면서 픽셀 label을 누적해 나간다. 논문에 따르면 region refinement는 PASCAL VOC 2012에서 약 1% 정도 정확도를 올린다.

stuff 처리를 위한 확장은 별도의 표현 설계가 핵심이다. 저자들은 stuff를 하나의 segment가 아니라 여러 segment proposal의 조합으로 본다. 이를 위해 purity score를 정의하는데, 이는 한 segment proposal과 그 bounding box 내부의 stuff portion 사이 IoU이다. purity score가 0.6보다 큰 proposal들을 candidate set으로 둔다.

그다음 Segment Pursuit를 수행한다. deterministic version에서는 후보 중 가장 큰 segment를 먼저 고르고, 그와 IoU 0.2 이상 겹치는 다른 후보를 제거한다. 남은 segment들의 면적이 초기 candidate set 평균 면적보다 작아질 때까지 이를 반복한다. 이렇게 하면 적은 수의 크고 informative한 segment들로 stuff를 compact하게 덮을 수 있다.

fine-tuning에는 deterministic 방식만으로는 sample 수가 부족하므로 stochastic Segment Pursuit를 쓴다. 각 단계에서 가장 큰 segment를 항상 고르는 대신, 면적에 비례한 확률로 후보를 샘플링한다. 이 과정에서 얻은 segment들을 stuff class의 positive로 사용한다. purity score가 0.3보다 작은 proposal은 negative로 사용한다. 각 epoch마다 각 이미지에서 새로운 stochastic compact combination을 만들고, 이들을 SGD에 공급한다. 논문은 SGD를 200k mini-batches에서 종료했다고 명시한다. 반면 SVM 학습에서는 deterministic Segment Pursuit로 얻은 단일 조합만 사용한다.

이 방법의 중요한 해석은, testing 절차 자체는 object-only 설정과 거의 같지만, training 시 sample distribution을 바꿔 classifier가 stuff를 더 compact한 proposal 조합으로 인식하도록 bias를 준다는 점이다.

## 4. 실험 및 결과

### PASCAL VOC 2012: Object Segmentation

첫 번째 실험은 20개 object category를 갖는 PASCAL VOC 2012 semantic segmentation benchmark이다. 평가는 region IoU로 수행된다. 학습에는 VOC 2012 training set과 추가 segmentation annotation을 함께 사용한다.

먼저 CFM 자체의 효과를 보기 위해 no-CFM baseline과 두 설계를 비교했다. ZF SPPnet과 Selective Search를 사용했을 때, no-CFM은 mean IoU 43.4, CFM Design A는 51.0, Design B는 50.9였다. 즉 CFM의 도입만으로 약 7.5포인트 이상의 큰 향상이 있었다. 이는 segment-aware feature가 실제로 중요한 정보를 준다는 뜻이다.

proposal method 비교에서는 Selective Search보다 MCG가 더 좋은 결과를 보였다. ZF SPPnet 기준으로 Selective Search는 50.9, MCG는 53.0이었다. VGG net에서는 Selective Search 56.3, MCG 60.9로 차이가 더 컸다. 이는 더 정확한 segment proposal이 CFM의 장점을 더 잘 끌어낸다는 해석이 가능하다.

backbone 비교에서는 VGG가 ZF보다 크게 우수했다. 같은 MCG를 쓸 때 ZF는 53.0, VGG는 60.9였다. 논문은 이 결과를 deeper model이 더 강한 표현력을 갖기 때문으로 설명한다.

multi-scale과 single-scale 비교에서는 성능 차이가 매우 작았다. MCG 기준으로 ZF는 5-scale 53.0, 1-scale 52.9였고, VGG는 60.9 대 60.5였다. 이는 single-scale만 써도 큰 정확도 손실 없이 훨씬 빠르게 동작할 수 있음을 뜻한다.

test set 결과를 보면, O2P는 47.8, SDS는 51.6, CFM(ZF+MCG)은 55.4, CFM(VGG+MCG)은 61.8이었다. 특히 CFM(ZF+MCG)이 SDS보다 3.8포인트 높고, CFM(VGG+MCG)은 61.8로 당시 매우 강력한 성능을 보였다.

속도 측면은 이 논문의 중요한 기여이다. feature extraction time per image를 보면 SDS(AlexNet)는 총 17.9초, CFM(ZF, 5 scales)는 0.38초, CFM(ZF, 1 scale)는 0.12초였다. 따라서 5-scale 기준 약 $47\times$, 1-scale 기준 약 $150\times$ 빠르다. VGG를 써도 1-scale에서 0.57초였다. 속도 향상의 핵심 이유는 convolutional feature map을 proposal마다 다시 계산하지 않고, 전체 이미지에 대해 한 번만 계산하기 때문이다.

논문은 동시대 FCN도 언급한다. FCN은 test set에서 62.2로 유사한 정확도를 내고 빠르지만, instance-wise 결과를 만들 수 없다고 저자들은 지적한다. 반면 CFM은 instance segmentation 평가에도 적용 가능하다.

### PASCAL VOC 2012: Simultaneous Detection and Segmentation

이 설정에서는 semantic category뿐 아니라 서로 다른 object instance도 구분해야 하며, 평가는 mean AP$r$로 수행된다. 논문은 validation set에서 결과를 보고한다.

SDS(AlexNet+MCG)는 49.7, CFM(ZF+SS)는 51.0, CFM(ZF+MCG)는 53.2, CFM(VGG+MCG)는 60.7이었다. 즉 CFM은 semantic segmentation뿐 아니라 instance-aware 평가에서도 SDS를 능가했다. 이는 feature-map 기반 masking이 단순히 픽셀 label accuracy만 높이는 것이 아니라, instance-level discrimination에도 도움이 된다는 점을 보여준다.

### PASCAL-CONTEXT: Joint Object and Stuff Segmentation

이 실험은 논문의 두 번째 큰 기여를 검증한다. PASCAL-CONTEXT는 59개 frequent category와 1개 background를 포함하는 60-way segmentation 문제이며, object와 stuff가 함께 등장한다. 평가는 60개 클래스 전체 mean IoU와, 논문 [18]에서 정의한 33개 easier categories 평균 IoU로 보고된다.

비교 대상은 SuperParsing과 O2P이다. 공정한 비교를 위해 region refinement는 사용하지 않는다.

전체 mean IoU에서 O2P는 18.1이고, 저자들의 no-CFM baseline은 20.7이었다. 즉 CNN feature만 써도 기존 비딥러닝 방법보다 강했다. CFM을 쓰되 Segment Pursuit를 쓰지 않으면 24.0으로 향상되었고, Segment Pursuit까지 쓰면 26.6이 되었다. 이는 masked convolutional feature 자체의 기여와, stuff sample selection 전략의 추가 기여가 모두 있음을 보여준다.

더 강한 설정인 VGG+MCG에서는 overall mean IoU가 34.4, easier 33 categories 평균은 49.5였다. 이는 표 전체에서 가장 높은 수치이다. 논문은 이를 통해 deeper backbone과 더 정확한 proposal, 그리고 Segment Pursuit가 결합될 때 성능이 크게 향상된다고 주장한다.

또한 저자들은 외부 데이터셋 MIT-Adobe FiveK에도 학습된 모델을 적용해 시각적으로 합리적인 결과를 얻었다고 제시한다. 다만 이 부분은 정량 평가가 아니라 qualitative example 중심이며, 수치적 일반화 성능은 논문에 명시되어 있지 않다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정이 명확하고, 제안 방식이 그 문제를 직접 해결한다는 점이다. raw-image masking의 인공 경계 문제와 proposal마다 CNN을 반복 실행하는 속도 문제를 동시에 겨냥했고, 실제로 정확도와 속도 모두 개선했다. 특히 feature extraction 시간에서의 대폭적인 감소는 매우 설득력이 있다.

또 다른 강점은 구조가 비교적 단순하면서도 기존 SPP-Net, proposal-based pipeline과 잘 결합된다는 점이다. 완전히 새로운 end-to-end 구조를 만드는 대신, 기존 CNN detection/segmentation 흐름을 feature-map 수준으로 재배치해 큰 이득을 얻었다. 이런 점에서 당시 기술 흐름 속에서 실용성이 높았다.

stuff 처리 방식도 흥미롭다. stuff를 object처럼 하나의 bounding box나 하나의 segment로 억지로 표현하지 않고, compact한 segment 조합으로 본 것은 개념적으로 타당하다. Segment Pursuit는 엄밀한 최적화 공식으로 제시되지는 않지만, 실제 데이터 특성에 맞춘 합리적인 heuristic으로 기능한다.

한계도 분명하다. 첫째, 전체 방법은 여전히 external region proposal에 크게 의존한다. Selective Search나 MCG의 품질이 성능에 직접 영향을 주며, 이는 완전한 end-to-end 학습 구조와는 거리가 있다. 둘째, 최종 예측에도 SVM, pasting, refinement 등 여러 단계가 들어가 pipeline이 다소 복합적이다. 현대적 관점에서는 통합성이 떨어진다고 볼 수 있다.

셋째, stuff 표현은 segment combination으로 설계했지만, 이것이 이론적으로 최적이라는 보장은 없다. Segment Pursuit는 heuristic이며, purity threshold 0.6, negative 0.3, inhibition IoU 0.2 같은 값들은 경험적 설정으로 보인다. 논문은 이 값들에 대한 민감도 분석을 제공하지 않는다. 따라서 방법의 안정성이나 일반성이 어느 정도인지는 논문만으로 완전히 판단하기 어렵다.

넷째, 논문은 수학적으로 복잡한 objective function을 제시하기보다 절차적 알고리즘 설명에 의존한다. 이는 구현 관점에서는 이해하기 쉽지만, 왜 이 방식이 최적인지 이론적으로 설명하는 데는 제한이 있다.

마지막으로, FCN과 비교할 때 저자들은 instance-wise 결과 생성 가능성을 장점으로 들지만, 반대로 proposal-free dense prediction의 단순성과 end-to-end 학습 이점은 CFM 쪽에서 가지기 어렵다. 즉 이 논문은 proposal-based paradigm 안에서 매우 강한 개선이지만, 이후 field가 proposal-free dense architectures로 이동했다는 점을 고려하면 역사적 위치를 그렇게 이해하는 것이 적절하다.

## 6. 결론

이 논문은 convolutional feature map 위에서 직접 segment mask를 적용하는 Convolutional Feature Masking을 제안하여, semantic segmentation에서 shape 정보를 효과적으로 활용하면서도 계산량을 크게 줄였다. 또한 stuff를 compact한 segment combination으로 표현하는 Segment Pursuit를 도입해 object와 stuff를 같은 프레임워크에서 함께 다룰 수 있게 했다.

정리하면, 주요 기여는 세 가지이다. 첫째, raw-image masking 대신 feature-map masking이라는 새로운 설계. 둘째, region feature와 segment feature를 결합하는 실용적인 network design. 셋째, joint object and stuff segmentation을 위한 sample generation 전략이다. 실험에서는 PASCAL VOC 2012와 PASCAL-CONTEXT에서 강력한 성능을 보였고, 특히 속도 개선이 매우 컸다.

이 연구는 proposal-based segmentation 시대에서 정확도와 효율의 균형을 끌어올린 중요한 작업이다. 실제 응용 측면에서는 빠른 instance-aware segmentation 시스템 설계에 의미가 있었고, 향후 연구 측면에서는 feature hierarchy의 어느 단계에서 shape 정보를 주입하는 것이 좋은지에 대한 중요한 관점을 제공했다.
