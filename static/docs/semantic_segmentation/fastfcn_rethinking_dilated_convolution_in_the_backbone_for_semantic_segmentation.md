# FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation

- **저자**: Huikai Wu, Junge Zhang, Kaiqi Huang, Kongming Liang, Yizhou Yu
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1903.11816

## 1. 논문 개요

이 논문은 semantic segmentation에서 널리 쓰이던 dilated convolution 기반 backbone의 계산량과 메모리 사용량이 지나치게 크다는 문제를 다룬다. 기존의 DeepLab 계열이나 EncNet, PSPNet 같은 강한 segmentation 모델들은 최종 feature map의 해상도를 높게 유지하기 위해 backbone의 마지막 downsampling을 제거하고 dilated convolution을 사용한다. 이 방식은 receptive field를 유지하면서도 더 촘촘한 spatial feature를 얻을 수 있어 성능상 이점이 있지만, 계산 비용이 매우 커진다.

논문의 핵심 목표는 dilated convolution을 직접 쓰지 않고도, 그것이 만들어내는 high-resolution feature map과 유사한 표현을 훨씬 효율적으로 얻는 것이다. 이를 위해 저자들은 **Joint Pyramid Upsampling (JPU)** 라는 새로운 모듈을 제안한다. 이 모듈은 원래의 FCN backbone처럼 output stride가 32인 저해상도 feature를 유지한 뒤, 중간 단계 feature들을 함께 이용해 output stride 8 수준의 고해상도 표현을 복원한다.

이 문제가 중요한 이유는 semantic segmentation이 정확도뿐 아니라 실제 적용 가능성도 요구하기 때문이다. 자율주행, 의료 영상, 로보틱스, 실시간 장면 이해 같은 환경에서는 segmentation의 품질과 함께 추론 속도와 메모리 효율이 모두 중요하다. 논문은 이 지점에서, “dilated convolution이 정말 backbone 내부에 꼭 있어야 하는가?”라는 질문을 던지고, 이를 다시 설계함으로써 속도와 정확도를 동시에 개선하려고 한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 backbone의 마지막 부분에서 dilated convolution으로 고해상도 feature를 직접 계산하는 대신, 그것을 **joint upsampling 문제**로 다시 해석하는 것이다. 즉, 저해상도 feature와 더 높은 해상도의 중간-level feature 사이의 관계를 학습하여, dilated backbone이 만들었을 high-resolution feature를 근사하자는 발상이다.

기존 DilatedFCN 계열은 backbone 자체를 수정해서 feature map 해상도를 높인다. 반면 FastFCN은 backbone은 원래 classification용 FCN처럼 유지하고, backbone 뒤에 별도의 JPU 모듈을 붙인다. 이로 인해 backbone 내부의 많은 residual block들이 훨씬 작은 spatial resolution에서 동작하게 되어 계산량이 크게 줄어든다.

또 하나의 차별점은 단순한 bilinear upsampling이나 FPN처럼 feature를 단순 융합하는 수준이 아니라, 여러 단계의 feature map을 함께 보고 multi-scale 관계를 학습한다는 점이다. 저자들은 JPU가 단순 업샘플링이 아니라, 저해상도 표현과 고해상도 guidance feature 사이의 구조적 관계를 추론하는 모듈이라고 본다. 이 때문에 JPU는 단순 interpolation보다 훨씬 낫고, 기존 dilated backbone을 쓰는 모델을 대체하면서도 성능 저하가 거의 없거나 오히려 개선된다고 주장한다.

## 3. 상세 방법 설명

전체 구조는 비교적 단순하다. backbone은 dilated convolution을 쓰는 DeepLab식 구조가 아니라, 원래 FCN처럼 Conv1부터 Conv5까지 단계적으로 다운샘플링하는 구조를 그대로 사용한다. 따라서 마지막 feature map은 output stride 32 해상도를 가진다. 이후 JPU가 Conv3, Conv4, Conv5를 입력으로 받아 더 고해상도의 feature를 생성하고, 그 위에 PSP, ASPP, Encoding 같은 context head를 얹어 최종 segmentation prediction을 만든다.

### DilatedFCN과 FastFCN의 차이

DilatedFCN에서는 마지막 두 stage의 downsampling을 제거하고, 그 대신 dilated convolution을 넣는다. 이렇게 하면 최종 feature map의 해상도가 높아진다. 하지만 spatial resolution이 큰 상태에서 많은 convolution을 수행해야 하므로 계산량과 메모리 사용량이 급증한다.

반면 FastFCN은 이 downsampling을 다시 복원한다. 즉, backbone은 원래 FCN처럼 작동하고, 고해상도 표현 복원은 backbone 밖의 JPU가 맡는다. 논문은 ResNet-101 기준으로, 이 방식이 많은 residual block에서 4배 혹은 16배 적은 계산/메모리를 요구한다고 설명한다.

### Joint Upsampling으로의 재정식화

논문은 dilated convolution과 stride convolution의 관계를 1D 예제로 설명한다.

joint upsampling의 일반 문제는 다음과 같이 정의된다. 저해상도 guidance image $x^l$ 와 저해상도 target image $y^l$ 가 있고, $y^l = f(x^l)$ 라고 하자. 여기서 계산이 더 가벼운 근사 함수 $\hat f$ 를 찾아, 고해상도 guidance image $x^h$ 에 적용하여 고해상도 target $y^h$ 를 얻는다.

본문의 정의는 다음과 같다.

$$
y^h = \hat f(x^h), \quad \text{where } \hat f(\cdot)=\arg\min_{h(\cdot)\in H}\|y^l-h(x^l)\|
$$

여기서 $H$ 는 가능한 함수들의 집합이고, $\|\cdot\|$ 는 거리 metric이다.

저자들은 DilatedFCN의 어떤 stage 출력과, stride convolution 기반 backbone의 대응 출력을 비교한다. 핵심은 dilated convolution을 거친 출력 $y^d$ 와 stride convolution을 거친 출력 $y^s$ 가 본질적으로 비슷한 convolution 함수 $C_r^n$ 를 공유하지만 입력 해상도가 다르다는 점이다. 논문은 이를 식 (2), (3)으로 전개한다.

DilatedFCN의 출력은 다음처럼 표현된다.

$$
y^d = x \to C_r \to C_d \to \cdots \to C_d
= \{y_m^0, y_m^1\} \to C_r^n \to M
$$

반면 제안 방식의 출력은

$$
y^s = x \to C_s \to C_r \to \cdots \to C_r
= y_m^0 \to C_r^n
$$

처럼 표현된다.

여기서 $C_r$, $C_d$, $C_s$ 는 각각 regular, dilated, stride convolution이고, $S$, $M$, $R$ 는 split, merge, reduce 연산이다. 논문의 핵심 관찰은, $y^s$ 와 $y^d$ 가 같은 convolution 함수에 서로 다른 입력이 들어간 결과처럼 보일 수 있다는 점이다. 따라서 dilated backbone의 고해상도 출력을 직접 계산하는 대신, 저해상도 출력과 중간 feature를 이용해 이를 복원하는 문제가 joint upsampling과 닮았다고 본다.

이를 식 (4)로 정리하면 다음과 같다.

$$
y = \{y_m^0, y_m^1\} \to \hat h \to M
$$

$$
\hat h = \arg\min_{h\in H}\|y^s-h(y_m^0)\|, \quad y_m=x \to C_r
$$

즉, 저자는 dilated convolution이 만들어냈을 고해상도 activation을 직접 계산하는 대신, 저해상도 feature와 고해상도 guidance 사이의 매핑 $\hat h$ 를 학습해 근사하려고 한다.

### JPU 구조

JPU는 세 개의 입력 feature map, 즉 Conv3, Conv4, Conv5를 사용한다. 각 feature map은 먼저 regular convolution block을 통과한다. 이 단계의 목적은 두 가지다.

첫째, 각 feature를 공통 embedding space로 보낸다.  
둘째, 채널 수를 줄여 이후 fusion 비용을 낮춘다.

그 다음 각 feature map을 같은 spatial resolution으로 upsample하고 concatenate하여 하나의 feature $y_c$ 를 만든다.

이후 핵심 연산이 들어간다. $y_c$ 에 대해 dilation rate가 서로 다른 네 개의 separable convolution을 병렬로 적용한다. dilation rate는 1, 2, 4, 8이다. 저자 설명에 따르면:

- dilation rate 1 convolution은 $y_m^0$ 와 나머지 $y_m$ 부분 사이의 관계를 본다.
- dilation rate 2, 4, 8 convolution은 $y_m^0$ 에서 $y^s$ 로 가는 매핑 $\hat h$ 를 다양한 scale에서 학습한다.

즉, JPU는 단순히 마지막 feature map만 보는 ASPP와 달리, **multi-level feature map들 사이의 관계 자체를 multi-scale manner로 학습**한다. 이 점이 논문에서 매우 강조되는 차별점이다.

마지막으로 또 하나의 regular convolution block을 사용해 병렬 dilated separable convolution들의 출력을 통합하고, 이를 최종 high-resolution representation으로 변환한다. 이후 이 출력은 Encoding, PSP, ASPP 같은 context head로 전달된다.

논문은 또한 JPU가 실제로는 두 개의 joint upsampling 문제를 동시에 푼다고 설명한다.

- Conv4를 Conv3의 guidance로 업샘플링하는 문제
- Conv5를 업샘플된 Conv4의 guidance로 다시 업샘플링하는 문제

즉, JPU는 stage 간 계층 구조를 활용한 joint upsampling 모듈이다.

### 학습 목표와 절차

훈련 손실은 pixel-wise cross-entropy이다. 추가적인 특별한 auxiliary loss나 distillation loss는 논문 본문에서 명시하지 않았다. Pascal Context에서는 초기 learning rate 0.001, poly schedule, batch size 16, crop size $480 \times 480$, SGD with momentum 0.9, weight decay $1e{-4}$ 를 사용했다고 적혀 있다. ADE20K에서는 120 epoch 동안 learning rate 0.01로 학습하고, 이후 train+val로 20 epoch 더 fine-tuning했다고 설명한다.

## 4. 실험 및 결과

### 데이터셋과 평가 설정

Pascal Context는 4,998장 train, 5,105장 val 이미지로 구성되며, 59개 object category와 background를 포함한 60 class를 사용한다. 평가 지표는 pixel accuracy와 mIoU이다.

ADE20K는 150개 class를 포함하는 scene parsing benchmark이며, 20K/2K/3K 이미지가 각각 train/val/test에 배정된다. 여기서도 pixel accuracy와 mIoU를 사용하고, test server에서는 final score로 $(\text{pixAcc} + \text{mIoU}) / 2$ 를 사용한다.

### Ablation Study

가장 중요한 ablation은 “dilated convolution을 그냥 제거하고 단순 업샘플링으로 대체하면 되는가?”에 대한 것이다. 결과는 그렇지 않다.

Pascal Context val에서 ResNet-50 backbone 기준:

- Encoding-8-None: pixAcc 78.39, mIoU 49.91
- Encoding-32-Bilinear: pixAcc 76.10, mIoU 46.47

즉, dilated backbone을 원래 FCN backbone으로 바꾸고 bilinear interpolation만 쓰면 성능이 크게 떨어진다. 이는 dilated convolution의 역할이 단순 해상도 보정 이상이라는 뜻이다.

다른 upsampling 방식과 비교하면:

- FPN: pixAcc 78.16, mIoU 49.59
- JPU: pixAcc 78.98, mIoU 51.05

JPU는 FPN보다도 높고, 원래 EncNet보다도 높다. 즉, 논문 주장대로 JPU는 단순 업샘플링보다 훨씬 강력하며, dilated backbone의 대체재로 기능한다.

또한 JPU는 특정 head에만 맞는 모듈이 아니다. ASPP와 PSP head에 붙였을 때도 원래 dilated backbone 버전보다 더 나은 결과를 보였다.

- ASPP: 49.19 → JPU 적용 시 50.07
- PSP: 50.58 → JPU 적용 시 50.89

이 결과는 JPU가 EncNet 전용이 아니라 여러 DilatedFCN 계열 모델에 일반적으로 적용 가능함을 보여준다.

### 속도 비교

Titan-Xp GPU에서 $512 \times 512$ 입력으로 FPS를 측정했다.

ResNet-50 + Encoding 기준:

- 원래 EncNet: 18.77 FPS
- Bilinear: 45.67 FPS
- FPN: 37.87 FPS
- JPU: 37.56 FPS

ResNet-101 + Encoding 기준:

- 원래 EncNet: 10.51 FPS
- JPU: 32.02 FPS

즉, ResNet-101에서는 약 3배 이상 빠르다. FPN과 속도는 비슷하지만 성능은 JPU가 더 좋다. 이 논문의 실용적 가치는 바로 여기에 있다. 단순히 속도만 올린 것이 아니라, 높은 정확도를 유지하거나 오히려 개선했다.

### Pascal Context 결과

state-of-the-art 비교에서:

- EncNet (ResNet-101): 51.7 mIoU
- DUpsampling (Xception-71): 52.5
- EncNet+JPU (ResNet-50): 51.2
- EncNet+JPU (ResNet-101): 53.1

ResNet-101 기반 FastFCN은 53.1 mIoU로 당시 최고 성능을 달성했다고 보고한다. 특히 ResNet-50 기반 버전도 매우 강력해서, 더 무거운 backbone을 쓰는 이전 방법들과 경쟁 가능했다.

다만 본문 각 표는 evaluation setup이 조금 다르다. Table 1은 59 classes without background, Table 3은 60 classes with background 및 multi-scale evaluation이다. 저자도 이 차이를 명시하고 있으므로, 표 간 수치를 직접 단순 비교하면 안 된다.

### ADE20K 결과

val set에서는:

- EncNet (ResNet-50): 41.11 mIoU
- Ours (ResNet-50): 42.75
- EncNet (ResNet-101): 44.65
- Ours (ResNet-101): 44.34

ResNet-50에서는 분명한 개선이 있다. ResNet-101에서는 val set 기준 EncNet보다 약간 낮다. 저자들은 이를 training crop size 차이로 설명한다. 자신들은 12GB GPU 제한 때문에 $480 \times 480$ crop을 사용했고, EncNet은 더 큰 메모리에서 $576 \times 576$ 로 학습했다고 적었다.

그런데 test set 제출 결과에서는:

- PSPNet (ResNet-269): 0.5538 final score
- EncNet (ResNet-101): 0.5567
- Ours (ResNet-101): 0.5584

즉, test server 기준으로는 오히려 EncNet과 PSPNet을 넘었다. val에서 약간 낮았던 모델이 test에서는 더 좋은 점수가 나온 것은 흥미롭지만, 논문은 그 이유를 깊게 분석하지는 않는다.

### 정성적 결과

논문은 시각화 결과를 통해 JPU가 bilinear나 FPN보다 경계와 가는 구조를 더 잘 복원한다고 주장한다. 예시에서는 새의 boundary, 가지, side shoot 같은 가늘고 세밀한 구조를 더 잘 분할했다고 설명한다. 이는 JPU가 multi-level feature들 사이의 구조 정보를 잘 결합했기 때문이라는 해석이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정 자체가 매우 설득력 있다는 점이다. semantic segmentation에서 dilated convolution은 오랫동안 성능 향상의 핵심 요소처럼 취급되었는데, 저자들은 그것이 backbone 내부의 필수 연산이라기보다 결국 “고해상도 feature 복원 문제”일 수 있다고 보고 다시 정식화했다. 이는 단순한 엔지니어링 트릭이 아니라, 구조적 재해석에 가깝다.

두 번째 강점은 실험이 명확하다는 점이다. bilinear, FPN, JPU를 같은 setting에서 비교했고, ASPP와 PSP로도 확장해 일반화 가능성을 보였다. 또한 성능뿐 아니라 FPS를 함께 보고하여 efficiency claim을 정량적으로 뒷받침했다.

세 번째 강점은 호환성이다. JPU는 특정 segmentation head에 강하게 종속되지 않고, 여러 DilatedFCN 계열 head 앞에 붙일 수 있는 모듈로 제시된다. 실제로 EncNet, ASPP, PSP에서 모두 개선이 보고되었다.

반면 한계도 있다. 첫째, 이 논문은 JPU가 왜 잘 작동하는지에 대해 직관과 구조적 해석은 제공하지만, 엄밀한 이론적 보장은 없다. joint upsampling 문제로의 재정식화는 흥미롭지만, 실제 CNN 근사 과정이 어느 정도까지 dilated backbone을 충실히 대체하는지는 경험적으로만 입증된다.

둘째, 속도 비교는 Titan-Xp와 특정 입력 크기에서 측정되었다. 실제 배포 환경이나 다른 GPU/하드웨어에서 동일한 비율의 이득이 유지되는지는 논문만으로는 확정할 수 없다.

셋째, ADE20K에서 ResNet-101 val 성능이 EncNet보다 낮았다는 점은, 이 방법이 항상 일관되게 우월하다고 말하기 어렵다는 신호이기도 하다. 저자들은 crop size 차이 때문이라고 설명하지만, 이는 완전히 분리된 통제 실험으로 검증된 것은 아니다.

넷째, JPU는 Conv3, Conv4, Conv5를 동시에 다루고 병렬 dilated separable convolution도 사용하므로, backbone 내부의 dilated convolution을 제거했다고 해서 전체 구조가 극단적으로 단순해지는 것은 아니다. 즉, 계산량은 줄지만 추가 모듈 설계 복잡성은 존재한다.

비판적으로 보면, 이 논문의 메시지는 “dilated convolution이 불필요하다”라기보다는 “dilated convolution이 backbone 내부에 있을 필요는 없고, multi-level feature 기반 joint upsampling으로 대체 가능하다”에 더 가깝다. 이 해석이 더 정확하다.

## 6. 결론

이 논문은 semantic segmentation에서 고해상도 feature를 얻기 위해 backbone 내부에서 dilated convolution을 유지해야 한다는 통념을 재검토한다. 저자들은 이를 joint upsampling 문제로 재구성하고, Conv3-5의 multi-level feature를 활용하는 **Joint Pyramid Upsampling (JPU)** 모듈을 제안했다. 그 결과, dilated backbone을 쓰는 기존 접근 대비 3배 이상 빠른 추론 속도와 더 낮은 메모리 사용량을 달성하면서도, Pascal Context와 ADE20K에서 매우 강하거나 당시 최고 수준의 성능을 보고했다.

이 연구의 의미는 단순히 하나의 빠른 segmentation 모델을 제안한 데 그치지 않는다. 더 중요한 점은, segmentation architecture에서 backbone 설계와 feature resolution 확보 방식을 다시 생각하게 만들었다는 것이다. 실제 응용 측면에서는 실시간성이나 메모리 제약이 중요한 환경에 유리하며, 향후 연구 측면에서는 backbone 내부 연산을 무겁게 유지하지 않고도 고해상도 semantic representation을 복원하는 다양한 구조 설계로 이어질 가능성이 크다.

논문 원문에 근거해 말할 수 있는 범위에서 정리하면, FastFCN의 핵심 기여는 **dilated convolution의 역할을 대체 가능한 형태로 분해하고, 이를 효율적인 learned upsampling 모듈로 구현했다는 점**이다. 이는 semantic segmentation에서 정확도와 효율을 함께 추구하는 매우 실용적인 방향의 연구라고 평가할 수 있다.
