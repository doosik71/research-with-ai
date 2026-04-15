# Convolutional CRFs for Semantic Segmentation

- **저자**: Marvin T. T. Teichmann, Roberto Cipolla
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1805.04777

## 1. 논문 개요

이 논문은 semantic segmentation에서 오랫동안 강력한 후처리 도구로 사용되어 온 fully-connected Conditional Random Field(FullCRF)의 속도와 학습 난점을 해결하기 위해, **Convolutional CRF(ConvCRF)**라는 새로운 구조를 제안한다. 저자들의 문제의식은 명확하다. CNN은 픽셀별 예측을 잘 하지만, 예측들 사이의 상호작용이나 구조적 일관성을 직접 모델링하는 데는 약하다. 반면 CRF는 이런 구조적 reasoning에 강하지만, 기존 FullCRF는 inference와 training이 매우 느리고, 내부 pairwise parameter 및 feature를 학습하기 어렵다.

논문이 다루는 핵심 연구 문제는 다음과 같다. **CRF의 structured modeling 능력은 유지하면서도, 기존 FullCRF의 병목인 mean-field inference를 GPU 친화적으로 재구성할 수 있는가?** 저자들은 여기에 대해, 픽셀 간 상호작용이 무한 범위가 아니라 일정한 local neighborhood 안에서만 의미 있다고 보는 강한 conditional independence/locality 가정을 도입하면, message passing을 convolution 형태로 다시 쓸 수 있다고 주장한다.

이 문제는 실제로 중요하다. semantic segmentation에서는 경계 보정, 인접 픽셀 간 label consistency, 비슷한 색/질감을 갖는 영역의 구조적 정합성이 성능에 큰 영향을 준다. 하지만 CRF가 너무 느리면 실제 시스템이나 end-to-end 학습에 넣기 어렵다. 이 논문은 CRF를 다시 실용적인 도구로 만들려는 시도이며, 특히 “정확한 message passing”과 “GPU에서의 빠른 구현”을 동시에 노린다는 점에서 의미가 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 **FullCRF에 locality assumption을 추가해 pairwise interaction을 유한한 범위로 잘라내고, 그 결과 message passing을 convolution-like operation으로 바꾸는 것**이다. 기존 FullCRF는 사실상 모든 픽셀 쌍 $(i, j)$를 연결하는 fully connected graph를 사용한다. 이 때문에 message passing이 본질적으로 매우 비싸고, 실제 구현에서는 permutohedral lattice라는 근사 기법에 의존한다. 저자들은 이 근사 자체가 GPU 구현을 어렵게 하고, gradient 계산과 feature learning도 방해한다고 본다.

ConvCRF는 두 픽셀 사이의 Manhattan distance가 filter-size $k$보다 크면 조건부 독립이라고 가정한다. 즉, 거리가 너무 멀면 pairwise potential을 0으로 둔다. 이 가정은 분명히 FullCRF보다 더 강한 제약이지만, 저자들은 CNN 자체가 이미 local processing에 기반하고 있다는 점을 들어 이것이 실용적으로 타당하다고 본다.

기존 접근과의 차별점은 크게 세 가지다. 첫째, **permutohedral lattice approximation을 제거하고 exact message passing을 수행**한다. 둘째, 그 exact 연산을 convolution 유사 구조로 바꾸어 **GPU에서 매우 빠르게 계산**한다. 셋째, 그 결과 **CRF 내부 parameter뿐 아니라 Gaussian feature 관련 요소도 backpropagation으로 학습하기 쉬워진다**. 논문은 속도 향상뿐 아니라, approximation artifact가 줄어들어 결과 품질도 약간 좋아질 수 있음을 보여준다.

## 3. 상세 방법 설명

논문은 semantic segmentation을 CRF 에너지 최소화 문제로 놓는다. 입력 이미지 $I$의 각 픽셀 $i$에 대해 label random variable $X_i \in \{1, \dots, k\}$를 두고, 전체 label assignment $X$의 posterior를 Gibbs distribution으로 모델링한다.

확률 모델은 다음과 같다.

$$
P(X=\hat{x}\mid \tilde{I}=I)=\frac{1}{Z(I)}\exp(-E(\hat{x}\mid I))
$$

여기서 $Z(I)$는 partition function이고, 에너지 함수는 unary term과 pairwise term의 합이다.

$$
E(\hat{x}\mid I)=\sum_{i \le N}\psi_u(\hat{x}_i\mid I)+\sum_{i \ne j \le N}\psi_p(\hat{x}_i,\hat{x}_j\mid I)
$$

$\psi_u$는 unary potential로, 실제로는 CNN이 출력하는 픽셀별 class score를 의미한다. 논문에서는 ResNet101 기반 FCN이 unary를 만든다. $\psi_p$는 pairwise potential로, 픽셀 간 상호작용을 모델링한다. 예를 들어 색이 비슷하고 공간적으로 가까운 픽셀은 같은 label일 가능성이 높다는 prior를 줄 수 있다.

FullCRF에서 pairwise potential은 Gaussian kernel들의 weighted sum으로 정의된다.

$$
\psi_p(x_i, x_j \mid I) := \mu(x_i, x_j)\sum_{m=1}^{M} w^{(m)} k_G^{(m)}(f_i^I, f_j^I)
$$

여기서 $w^{(m)}$는 kernel weight이고, $f_i^I$는 픽셀 $i$의 feature vector이다. $\mu(x_i, x_j)$는 compatibility transformation이다. 가장 흔한 형태는 Potts model로, label이 다를 때만 패널티를 주는 방식이다. 즉, 대략적으로 “feature가 비슷한데 label이 다르면 비용이 크다”는 구조를 만든다.

기존 FullCRF는 보통 두 개의 Gaussian kernel을 쓴다. 하나는 appearance kernel이고, 다른 하나는 smoothness kernel이다.

$$
k(f_i^I, f_j^I) :=
w^{(1)} \exp\left(
-\frac{|p_i-p_j|^2}{2\theta_\alpha^2}
-\frac{|I_i-I_j|^2}{2\theta_\beta^2}
\right)
+
w^{(2)} \exp\left(
-\frac{|p_i-p_j|^2}{2\theta_\gamma^2}
\right)
$$

여기서 $p_i$는 spatial coordinate, $I_i$는 pixel color 값이다. 첫 번째 항은 위치와 색이 모두 비슷한 경우 강하게 연결하고, 두 번째 항은 단순한 spatial smoothness를 준다. $\theta_\alpha, \theta_\beta, \theta_\gamma$는 Gaussian scale parameter다.

문제는 mean-field inference의 message passing step이다. 논문에 제시된 알고리즘은 초기값을 unary에서 softmax로 만들고, 반복적으로 message passing, compatibility transform, unary 추가, normalization을 수행한다. 이 중 병목은 message passing이다. FullCRF에서는 모든 픽셀과의 상호작용이 있으므로 정확 계산이 사실상 quadratic complexity를 가지며, 이를 피하기 위해 permutohedral lattice를 사용했다.

ConvCRF의 핵심은 여기서 등장한다. 저자들은 두 픽셀 $i, j$ 사이의 Manhattan distance가 $k$보다 크면 상호작용이 없다고 가정한다. 그러면 pairwise potential은 local window 안에서만 계산하면 된다. 이때 Gaussian kernel을 feature map 위의 위치별 filter로 정의할 수 있다.

입력 $P$가 shape `[bs, c, h, w]`라고 하자. Gaussian kernel $g$는 여러 feature map $f_1, \dots, f_d$로 정의되며, kernel matrix는 다음과 같이 쓴다.

$$
k_g[b, dx, dy, x, y]
:=
\exp\left(
-\sum_{i=1}^{d}
\frac{
|f_i^{(d)}[b,x,y]-f_i^{(d)}[b,x-dx,y-dy]|^2
}{
2\dot{\theta}_i^2
}
\right)
$$

즉, 어떤 위치 $(x,y)$에서 주변 offset $(dx,dy)$에 대해 feature 차이에 기반한 Gaussian weight를 계산한다. 여러 Gaussian kernel이 있으면 이를 가중합하여 merged kernel $K$를 만든다.

그 다음 combined message passing 결과 $Q$는 다음처럼 계산된다.

$$
Q[b,c,x,y]
=
\sum_{dx,dy \le k}
K[b,dx,dy,x,y]\cdot P[b,c,x+dx,y+dy]
$$

이 식은 일반적인 2D convolution과 닮아 있다. 다만 차이는 보통 convolution filter는 위치에 독립적이지만, 여기서는 filter 값이 $(x,y)$ 위치에 따라 달라진다는 점이다. 논문은 이를 locally connected layer와 비슷하다고 설명한다. 또 filter는 class channel $c$에 대해서는 공유된다. 저자들은 일반 CNN 연산들만으로도 구현은 가능하지만, 데이터 재배열(im2col 유사 과정)에 시간이 너무 많이 들어 실제로는 low-level native implementation을 만들었다고 한다. profiling 결과 GPU 시간의 약 90%가 데이터 재배열에 쓰였고, native 구현으로 추가 10배 속도 향상을 얻었다고 보고한다.

구현 측면에서 baseline ConvCRF는 comparability를 위해 FullCRF와 같은 설계를 유지한다. normalization은 softmax, compatibility는 Potts model, Gaussian feature는 hand-crafted feature를 사용한다. 또한 pairwise kernel에 Gaussian blur를 적용해 effective filter size를 4배 키운다고 설명한다.

추가 실험에서는 두 가지 확장도 다룬다. 하나는 smoothness kernel의 입력 feature $p_i$를 고정된 좌표 대신 **learnable variable**로 바꾸는 것이다. 다른 하나는 CRFasRNN처럼 **$1\times1$ convolution을 compatibility transformation으로 사용하는 학습형 compatibility**이다. 이 둘이 ConvCRF의 학습 가능성을 보여주는 장치다.

## 4. 실험 및 결과

실험은 synthetic task와 실제 PASCAL VOC 2012 semantic segmentation task로 나뉜다.

먼저 데이터셋 설정을 보면, 실제 segmentation 실험은 PASCAL VOC 2012와 추가 annotation을 사용해 총 10,582장의 학습 이미지를 구성한다. 이 중 200장은 CRF 내부 parameter tuning용으로 따로 떼어 두고, 나머지 10,382장으로 unary CNN을 학습한다. 평가는 공식 validation set 1,464장에서 수행한다.

Unary network는 ResNet101 backbone 위에 간단한 FCN decoder를 붙인 구조다. ImageNet pretrained weight로 초기화하고, MS COCO 같은 대규모 segmentation dataset은 사용하지 않았다. 학습은 200 epochs, batch size 16, Adam optimizer, 초기 learning rate $5\times10^{-5}$로 진행한다. learning rate는 polynomial decay를 쓰며, weight decay는 $5\times10^{-4}$, dropout은 final convolution 위에 rate 0.5를 사용한다. augmentation은 horizontal flip, rotation, scale resize, color jitter 등을 포함한다. 이 unary 모델은 validation mIoU 71.23%, train mIoU 91.84%를 기록한다. 이 큰 gap에 대해 저자들은 segmentation 모델의 일반화 이슈와 CRF tuning의 필요성을 논의한다.

CRF inference는 모든 실험에서 mean-field 5 iteration을 사용한다. training 시에는 이 5 step을 unroll한다.

### Synthetic data 실험

첫 번째 실험은 ConvCRF의 구조적 장점을 보여주기 위한 synthetic denoising task다. PASCAL VOC ground truth를 8배 downsample했다가 noise를 넣고 다시 upsample하여, low-resolution prediction의 오류와 checkerboard artifact를 흉내 낸 unary를 만든다. 그 뒤 CRF가 이를 원래 label로 복원하도록 한다.

여기서 FullCRF와 ConvCRF는 같은 hand-crafted Gaussian feature와 같은 초기 parameter를 사용한다. 결과는 다음과 같다.

- Unary: mIoU 51.87%, Accuracy 86.60%, Speed 68 ms
- FullCRF: mIoU 84.37%, Accuracy 94.79%, Speed 647 ms
- ConvCRF filter size 5: mIoU 90.90%, Accuracy 97.13%, Speed 7 ms
- ConvCRF filter size 7: mIoU 92.98%, Accuracy 97.13%, Speed 13 ms
- ConvCRF filter size 11: mIoU 93.74%, Accuracy 98.97%, Speed 26 ms
- ConvCRF filter size 13: mIoU 93.89%, Accuracy 98.99%, Speed 34 ms

이 결과는 두 가지를 보여준다. 첫째, ConvCRF는 FullCRF보다 **훨씬 빠르다**. 특히 FullCRF 647 ms에 비해 ConvCRF는 수 ms~수십 ms 수준이다. 둘째, ConvCRF는 같은 parameter를 써도 **성능이 더 좋다**. 저자들은 이를 permutohedral lattice approximation error가 사라졌기 때문이라고 해석한다. 시각화에서도 FullCRF는 object boundary 주변에 approximation artifact가 보인다고 주장한다.

### Decoupled training 실험

두 번째는 실제 Pascal VOC에서 **2-stage decoupled training**을 적용한 실험이다. 먼저 unary CNN을 학습한 뒤, 그 출력을 고정하고 CRF 내부 parameter만 학습한다. 이 방식의 장점으로 저자들은 유연성, 해석 가능성, 빠른 프로토타이핑, vanishing gradient 완화 등을 든다.

비교 대상은 unary baseline, DeepLab 방식의 FullCRF, ConvCRF, 그리고 feature/compatibility를 학습하는 ConvCRF 변형들이다. 결과는 다음과 같다.

- Unary: mIoU 71.23%, Accuracy 91.84%
- DeepLab FullCRF: mIoU 72.02%, Accuracy 94.01%
- ConvCRF: mIoU 72.04%, Accuracy 93.99%
- ConvCRF + learnable Gaussian feature (`+T`): mIoU 72.07%, Accuracy 94.01%
- ConvCRF + learnable compatibility (`+C`): mIoU 72.30%, Accuracy 94.01%
- ConvCRF + both (`+CT`): mIoU 72.37%, Accuracy 94.03%

절대 수치 향상은 크지 않지만, unary 대비 약 1.1 point 정도 mIoU가 상승했고, ConvCRF가 FullCRF보다 소폭 우세하다. 특히 compatibility transformation과 Gaussian feature를 학습했을 때 성능이 가장 좋다. 이는 ConvCRF가 단지 빠른 근사 대체물이 아니라, **학습 가능성이 더 높은 structured module**이라는 논문의 주장을 뒷받침한다.

다만 train mIoU를 보면 일부 ConvCRF 변형이 validation 향상에 비해 train score도 꽤 올라간다. 이는 모델 capacity 증가와 held-out tuning protocol의 영향이 함께 있을 수 있음을 시사한다. 논문은 held-out 200장에 대해 CRF를 따로 fine-tuning하는 전략을 썼으며, 이는 validation 분포와 training 분포의 unary quality 차이를 반영하기 위한 선택이라고 설명한다.

### End-to-End learning 실험

세 번째는 end-to-end 학습이다. 여기서는 CNN과 CRF가 함께 최적화된다. 저자들은 joint training의 장점으로 CNN과 CRF의 co-adaptation을 든다. 하지만 mean-field 5 iteration을 통과해 gradient를 전달해야 하므로 vanishing gradient 문제가 있다.

이를 해결하기 위해 저자들은 CRFasRNN의 protocol을 그대로 따르지 않고 다음과 같은 변형을 사용한다.

1. unary-only training을 200 epoch까지 하지 않고 100 epoch에서 멈춘다.
2. 이후 CNN+CRF joint training을 시작한다.
3. auxiliary unary loss를 추가해 gradient 약화를 완화한다.
4. unary-only gradient update와 joint update를 번갈아 수행한다.
5. 각 epoch 끝에서 held-out 200장으로 CRF parameter를 fine-tune한다.

전체 학습은 4개의 GTX 1080 Ti GPU에서 약 30시간이 걸린다.

결과는 다음과 같다.

- Unary: mIoU 70.99%, Accuracy 93.76%
- ConvCRF: mIoU 72.18%, Accuracy 94.04%
- CRFasRNN: mIoU 69.6%, Accuracy 93.03%

저자들은 이 비교가 완전히 공정하지는 않다고 직접 인정한다. 특히 CRFasRNN은 batch size 1만 지원하고, 한 epoch가 약 5시간 걸리며, 매우 작은 고정 learning rate $10^{-13}$를 써야 했다고 한다. 따라서 수치 비교 자체보다는, **ConvCRF의 빠른 학습이 더 유연한 training protocol을 가능하게 하고, 그 결과 더 나은 모델을 만들 수 있다**는 점을 강조한다. 이 점은 논문의 중요한 실용적 메시지다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 해결 방식이 매우 일관적이라는 점이다. CRF가 느리고 배우기 어렵다는 기존 불만을, 단순한 locality assumption 하나로 정면 돌파했다. 그 결과 inference와 training 속도가 약 두 자릿수 배 이상 향상되고, message passing을 exact하게 수행할 수 있게 되었다는 점은 분명한 기여다. 특히 synthetic 실험에서 FullCRF보다 더 높은 정확도와 훨씬 빠른 속도를 동시에 보인 것은 설득력이 있다.

또 다른 강점은 ConvCRF를 단순한 engineering optimization으로만 제시하지 않았다는 점이다. 논문은 Gaussian feature 학습과 learnable compatibility transformation까지 실험하여, ConvCRF가 **학습 가능한 structured prediction module**로 확장될 수 있음을 보여준다. 실제 Pascal VOC에서 성능 향상은 modest하지만, end-to-end와 decoupled training 모두에서 일관되게 개선을 보인 점도 긍정적이다.

하지만 한계도 분명하다. 가장 핵심적인 것은 **conditional independence/locality assumption의 강함**이다. ConvCRF는 사실상 일정 거리 밖의 long-range interaction을 버린다. 이는 속도 향상의 직접 원인이지만, 동시에 FullCRF가 의도했던 “전역적으로 연결된 구조 모델”의 장점을 일부 포기한 것이다. 논문은 이 가정이 CNN의 local processing과 잘 맞는다고 주장하지만, 장거리 문맥 정보가 중요한 장면에서 어떤 손실이 생기는지는 정량적으로 깊게 분석하지 않았다.

또한 실제 Pascal VOC 실험에서의 성능 향상은 비교적 작다. unary 71.23%에서 72.37%로의 향상은 의미는 있지만, 구조를 크게 바꾼 것에 비해 절대 개선폭은 제한적이다. 이는 semantic segmentation에서 CRF 후처리의 한계, 혹은 stronger CNN backbone이 이미 많은 구조 정보를 내재적으로 학습하고 있기 때문일 수도 있다. 논문은 이 부분을 깊게 논의하지는 않는다.

비교의 공정성 문제도 있다. end-to-end 실험에서 CRFasRNN과의 비교가 완전히 fair하지 않다고 저자 스스로 인정한다. 실제로 training protocol, batch size, learning rate, 하드웨어 제약이 다르기 때문에 “ConvCRF가 더 좋은 모델”이라는 결론보다는 “ConvCRF가 훨씬 실용적인 학습 환경을 제공한다” 정도로 읽는 것이 더 정확하다.

마지막으로, 논문은 ConvCRF의 이론적 표현력을 엄밀히 분석하지 않는다. 예를 들어 locality truncation이 원래 FullCRF의 어떤 해를 얼마나 잘 근사하는지, 혹은 learned feature가 실제로 어떤 구조를 포착하는지에 대한 분석은 없다. 따라서 이 논문은 강한 empirical/system contribution으로 읽는 것이 적절하다.

## 6. 결론

이 논문은 semantic segmentation용 CRF를 더 빠르고 학습 가능하게 만들기 위해 **Convolutional CRF**를 제안했다. 핵심은 fully connected pairwise interaction을 local neighborhood로 제한하는 강한 가정을 도입하고, 그 결과 mean-field message passing을 convolution 형태로 재구성한 것이다. 이를 통해 permutohedral lattice approximation을 제거하고, GPU에서 exact message passing을 매우 빠르게 수행할 수 있게 되었다.

실험적으로는 synthetic task에서 FullCRF보다 더 높은 정확도와 훨씬 빠른 속도를 보였고, Pascal VOC에서도 decoupled 및 end-to-end 설정 모두에서 unary 또는 기존 CRF baseline 대비 소폭의 성능 향상을 보였다. 특히 learnable Gaussian feature와 compatibility transformation을 쉽게 통합할 수 있다는 점은 향후 structured prediction 연구에 실질적인 장점을 제공한다.

종합하면, 이 연구의 가장 중요한 의미는 “CRF를 다시 쓸 만하게 만들었다”는 데 있다. 절대적인 mIoU 개선폭보다도, CRF inference/training을 deep learning pipeline에 자연스럽게 넣을 수 있을 정도로 가볍고 유연하게 만든 점이 더 본질적인 기여다. 따라서 이 논문은 semantic segmentation 자체뿐 아니라, instance segmentation, landmark recognition 등 다른 structured prediction 문제로 확장될 가능성을 가진 실용적 기반 연구로 볼 수 있다.
