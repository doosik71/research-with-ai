# MISSFormer: An Effective Medical Image Segmentation Transformer

## 논문 메타데이터

- **제목**: MISSFormer: An Effective Medical Image Segmentation Transformer
- **저자**: Xiaohong Huang, Zhifang Deng, Dandan Li, Xueguang Yuan
- **arXiv 공개 연도**: 2021
- **arXiv ID**: 2109.07162
- **arXiv URL**: <https://arxiv.org/abs/2109.07162>
- **PDF URL**: <https://arxiv.org/pdf/2109.07162v2>
- **코드**: <https://github.com/ZhifangDeng/MISSFormer>

## 연구 배경 및 문제 정의

이 논문은 의료영상 분할에서 CNN과 Transformer가 각각 가진 구조적 한계를
동시에 겨냥한다. 저자들은 U-Net 계열 CNN이 의료영상 분할에서 강력한
기준선이지만, convolution의 국소성 때문에 장거리 의존성
(long-range dependency)을 충분히 포착하지 못한다고 본다.
반대로 Transformer는 전역 상호작용 모델링에는 강하지만, 의료영상 분할에서
중요한 local context와 경계 세부 정보 복원에는 취약하다고 지적한다.

기존 연구들도 dilated convolution, pyramid pooling, 일부 self-attention
layer 삽입, CNN-Transformer 결합 등을 시도했지만 저자들은 두 가지
미해결 문제가 남아 있다고 본다.

- Transformer의 FFN 안에 convolution을 단순 삽입하는 방식은
  local context를 일부 보완하지만 feature distribution 정렬이 충분하지 않아
  표현력이 제한된다.
- 계층형 encoder가 만들어내는 multi-scale feature를 전역 문맥과 함께
  통합하는 메커니즘이 부족하다.

따라서 이 논문의 핵심 문제 설정은 다음과 같다. 의료영상 분할에서
Transformer의 전역 문맥 모델링 능력을 유지하면서도, local context와
multi-scale feature aggregation을 함께 강화하는 순수 Transformer 기반
U-shaped segmentation 모델을 설계할 수 있는가.

## 논문의 핵심 기여

저자들이 제시하는 핵심 기여는 네 가지다.

1. 의료영상 분할을 위한 position-free hierarchical U-shaped Transformer인
   MISSFormer를 제안했다.
2. 기존 Mix-FFN을 재설계한 `Enhanced Mix-FFN`과 이를 포함한
   `Enhanced Transformer Block`을 도입해 장거리 의존성과 local context를
   동시에 강화했다.
3. encoder의 multi-scale hierarchical feature를 통합하는
   `Enhanced Transformer Context Bridge`를 제안했다.
4. Synapse multi-organ CT와 ACDC cardiac MRI에서, ImageNet 사전학습 없이
   scratch 학습만으로도 기존 SOTA 대비 경쟁력 있거나 더 나은 성능을 보였다.

이 논문은 단순히 또 하나의 segmentation backbone을 제안하는 것이 아니라,
Transformer 기반 의료영상 분할에서 "전역 정보만 보는 모델"을 넘어서
"global-local feature discrimination"과
"multi-scale context aggregation"을 함께 설계해야 한다는 주장을
구체화한다.

## 방법론 요약

### 1. 전체 구조

MISSFormer는 U-shaped encoder-decoder 구조를 기반으로 한다. 전체 구성은
다음과 같다.

- **Encoder**: overlapped patch embedding과 patch merging을 이용한
  hierarchical transformer encoder
- **Bridge**: multi-scale encoder feature를 통합하는
  Enhanced Transformer Context Bridge
- **Decoder**: patch expanding 기반의 hierarchical decoder
- **Skip connection**: 대응되는 encoder-decoder stage를 연결

기본 골격은 SegFormer와 Swin-Unet 계열의 계층형 Transformer 구조를
떠올리게 하지만, 저자들은 핵심 차별점을 FFN 재설계와 bridge 설계에 둔다.

### 2. Efficient Self-Attention 기반 hierarchical encoder

각 stage의 Transformer block은 Efficient Self-Attention을 사용한다.
이는 PVT 계열의 spatial reduction self-attention과 유사한 방향으로,
고해상도 feature map에서 self-attention의 비용을 줄이기 위한 설계다.
결과적으로 고해상도 의료영상에서도 pure transformer 구조를 유지하면서
계산량을 통제하려는 의도가 분명하다.

또한 encoder는 overlapped patch embedding을 사용해 patch 경계에서 생기는
정보 손실을 줄이고, patch merging으로 계층적 표현을 점진적으로 만든다.
이는 의료영상처럼 경계와 구조적 연속성이 중요한 문제에 더 적합한 선택이다.

### 3. Enhanced Mix-FFN

이 논문의 가장 중요한 설계는 FFN 재설계다. 기존 SegFormer류 Mix-FFN은
FFN 내부에 depth-wise convolution을 넣어 local context를 보완하지만,
저자들은 이것만으로는 feature alignment와 discrimination이 부족하다고
본다.

Enhanced Mix-FFN의 핵심은 다음과 같다.

- first FC 이후 depth-wise `3x3` convolution 적용
- convolution 출력과 FC 출력을 skip connection으로 합산
- 합산 뒤 LayerNorm을 적용해 feature distribution 정렬
- 이후 GELU와 두 번째 FC를 거쳐 residual connection으로 원 입력을 다시 더함

논문이 제시한 식은 다음과 같다.

$$
y_1 = \mathrm{LN}(\mathrm{Conv}_{3 \times 3}(\mathrm{FC}(x_{in})) + \mathrm{FC}(x_{in}))
$$

$$
x_{out} = \mathrm{FC}(\mathrm{GELU}(y_1)) + x_{in}
$$

저자들의 해석은 단순하다. convolution을 FFN에 직접 삽입하면
local context는 늘지만 feature consistency가 흔들릴 수 있는데,
skip connection과 LayerNorm을 추가하면 local information 보완과
feature alignment를 동시에 달성할 수 있다는 것이다.

### 4. Enhanced Transformer Block

Enhanced Transformer Block은 LayerNorm, Efficient Self-Attention,
Enhanced Mix-FFN으로 구성된다. 즉, 전역 의존성은 attention으로,
local context와 feature discrimination은 재설계된 FFN으로 처리하는
구조다.

이 블록은 encoder, bridge, decoder 전체에 공통적으로 사용된다.
따라서 MISSFormer의 개선은 특정 stage만의 기법이 아니라 네트워크 전반의
building block 차원에서 일관되게 적용된다.

### 5. Enhanced Transformer Context Bridge

또 하나의 핵심은 bridge 모듈이다. 기존 segmentation transformer들이
주로 bottleneck 한 지점의 global information만 처리하는 데 비해,
MISSFormer는 encoder에서 나온 여러 stage의 multi-scale feature를
bridge에서 함께 다룬다.

저자들은 이 bridge가 다음 역할을 한다고 본다.

- 서로 다른 scale에서 얻은 feature를 함께 사용
- long-range dependency와 local context를 동시에 추출
- hard case에서 boundary와 구조적 일관성을 더 안정적으로 복원

실제로 ablation에서는 bridge에 어떤 FFN을 넣는지가 성능에 의미 있는
영향을 준다. 단순 MLP bridge보다 Mix-FFN bridge가 local detail을
보완하고, Enhanced Mix-FFN bridge가 가장 좋은 전체 성능을 보였다.

## 실험 설정

### 1. 데이터셋

논문은 두 가지 의료영상 분할 벤치마크를 사용한다.

#### 1.1 Synapse multi-organ segmentation

- abdominal CT 30건
- 총 3779개 axial slice
- 18 scans train, 12 scans test
- 8개 장기 대상: aorta, gallbladder, spleen, left kidney, right kidney,
  liver, pancreas, stomach
- 평가 지표: 평균 DSC, 평균 HD

#### 1.2 ACDC

- cardiac MRI 100건
- 70 train, 10 validation, 20 test
- 3개 구조: RV, Myo, LV
- 평가 지표: 평균 DSC

### 2. 학습 설정

논문은 모든 실험을 ImageNet 사전학습 없이 scratch에서 수행했다고
강조한다. 주요 설정은 다음과 같다.

- 입력 크기: `224 x 224`
- 최대 epoch: 400
- batch size: 24
- optimizer: SGD
- 초기 learning rate: 0.05
- learning rate schedule: poly
- momentum: 0.9
- weight decay: `1e-4`

비교 대상에는 TransUNet, Swin-Unet 등 ImageNet 사전학습 backbone을
사용하는 방법이 포함되어 있어, 저자들은 scratch training으로 이 정도
성능을 낸 점을 주요 장점으로 내세운다.

## 주요 결과 해석

### 1. Synapse에서의 성능

Synapse에서 MISSFormer는 평균 DSC `81.96`, 평균 HD `18.20`을 기록했다.
주요 비교 결과는 다음과 같다.

| 방법 | DSC | HD |
| --- | ---: | ---: |
| R50 U-Net | 74.68 | 36.87 |
| U-Net | 76.85 | 39.70 |
| R50 Att-UNet | 75.57 | 36.97 |
| Att-UNet | 77.77 | 36.02 |
| R50 ViT | 71.29 | 32.87 |
| TransUNet | 77.48 | 31.69 |
| Swin-Unet | 79.13 | 21.55 |
| MISSFormer S | 80.74 | 19.65 |
| **MISSFormer** | **81.96** | **18.20** |

이 결과는 몇 가지를 시사한다.

- pure Transformer 구조라도 설계가 적절하면 CNN hybrid 모델보다 더 나은
  성능을 낼 수 있다.
- 단순 hierarchical transformer만으로는 부족하고, Enhanced Mix-FFN과
  bridge가 실제 성능 차이를 만든다.
- Swin-Unet과 TransUNet이 사전학습 backbone에 의존하는 데 비해,
  MISSFormer는 scratch 학습으로 더 높은 평균 Dice를 달성했다.

장기별 점수에서도 MISSFormer는 대체로 상위권이다. 특히 pancreas,
stomach처럼 어려운 장기에서 기존 방법보다 개선 폭이 눈에 띈다.

### 2. ACDC에서의 성능

ACDC MRI에서도 MISSFormer는 평균 DSC `90.86`으로 최고 성능을 기록했다.

| 방법 | 평균 DSC | RV | Myo | LV |
| --- | ---: | ---: | ---: | ---: |
| R50 U-Net | 87.55 | 87.10 | 80.63 | 94.92 |
| R50 Att-UNet | 86.75 | 87.58 | 79.20 | 93.47 |
| R50 ViT | 87.57 | 86.07 | 81.88 | 94.75 |
| TransUNet | 89.71 | 88.86 | 84.53 | 95.73 |
| Swin-Unet | 90.00 | 88.55 | 85.62 | 95.83 |
| **MISSFormer** | **90.86** | **89.55** | **88.04** | 94.99 |

여기서 특히 중요한 부분은 Myo 점수다. MISSFormer는 myocardium에서
`88.04`를 기록해 Swin-Unet의 `85.62`보다 크게 높다. 이는 복잡한 경계와
얇은 구조를 가진 영역에서 제안한 global-local feature 설계가 실제로
유효함을 보여주는 지점이다. 반면 LV는 Swin-Unet보다 약간 낮아,
모든 구조에서 일관된 절대 우세라기보다는 어려운 구조에서의 강점이 더
뚜렷하다고 볼 수 있다.

### 3. 핵심 ablation

이 논문은 ablation 구성이 비교적 설득력 있다. 단순 baseline 대비
각 설계 요소가 성능에 어떻게 기여하는지 단계적으로 보여준다.

#### 3.1 U-shaped transformer 구조의 효과

SegFormer B1의 평균 DSC는 `75.24`, U-SegFormer는 `76.10`이다.
저자들은 이 차이를 skip connection을 통한 detail fusion 효과로
해석한다. 즉, 의료영상 분할에서는 segmentation용 decoder와
U-shaped 구조 자체가 여전히 중요하다는 뜻이다.

#### 3.2 Enhanced Mix-FFN의 효과

baseline인 U-SegFormer `76.10/26.97`에서,

- skip concatenation 추가: `78.14/28.77`
- skip summation 추가: `78.74/20.20`
- LayerNorm 정렬을 더한 Simple MISSFormer: `79.73/20.14`

까지 올라간다. 논문은 여기서 feature alignment와 distribution
consistency가 실제로 중요하다고 본다.

추가로 local information 보완 모듈 비교에서,

- U-mlpFormer: `75.88/27.22`
- U-SegFormer: `76.10/26.97`
- U-LocalViT: `76.92/23.62`
- Simple MISSFormer: `79.73/20.14`

을 보고한다. 즉, local context를 넣는 것만으로는 충분하지 않고,
어떻게 feature를 정렬하느냐가 핵심이라는 주장이다.

#### 3.3 Recursive skip과 MISSFormer S

Enhanced Mix-FFN을 일반화한 MISSFormer S에서는 recursive step을
늘릴수록 성능이 대체로 좋아졌다.

- step 1: DSC `79.73`
- step 2: DSC `79.91`
- step 3: DSC `80.74`

저자들은 이를 convolution이 FFN에 직접 삽입될 때 생기는 표현 불일치를
더 강하게 보정한 결과로 해석한다.

#### 3.4 Enhanced Transformer Context Bridge의 효과

MISSFormer S에 bridge를 추가한 full MISSFormer는 성능이 더 올라간다.

- MISSFormer S step 1: `79.73/20.14`
- MISSFormer step 1 with bridge: `81.96/18.20`

즉, bridge만으로 약 `2.26` DSC 개선이 있었다. 하지만 recursive step을
2, 3으로 늘리면 성능이 오히려 감소해, 저자들은 model capacity와
normalization 수 사이의 균형이 존재한다고 해석한다.

bridge depth와 scale 사용에 대한 결과도 의미가 있다.

- bridge depth 4가 depth 2, 6보다 가장 좋았다.
- multi-scale feature를 `4/3/2/1` 모두 사용할 때 가장 좋았고,
  일부 stage만 사용할수록 성능이 감소했다.

이는 의료영상 분할에서 고수준 semantic 정보뿐 아니라 여러 해상도의
문맥 결합이 중요하다는 점을 다시 확인해 준다.

## 한계 및 향후 연구 가능성

논문은 성능은 좋지만 몇 가지 한계를 가진다.

### 1. 데이터셋 규모와 일반화 검증의 한계

Synapse와 ACDC는 의료영상 분할에서 널리 쓰이는 벤치마크지만 규모가
크지 않다. 특히 3D clinical deployment나 더 다양한 modality로의
일반화는 충분히 검증되지 않았다.

### 2. 계산량과 구조 복잡도

MISSFormer는 pure transformer U-shaped 구조라 CNN 대비 구현과 연산 비용이
가볍지 않다. 논문은 scratch 학습의 강점을 강조하지만, 실제 임상 배치에서는
latency와 memory cost가 여전히 문제가 될 수 있다.

### 3. 2D 중심 평가

논문은 주로 2D slice 기반 segmentation 실험에 집중한다. 의료영상의 중요한
상당수 과제는 3D volumetric consistency가 핵심이므로, 이 구조가 3D
setting에서 얼마나 효율적으로 확장되는지는 별도 문제다.

### 4. 사전학습 이점과의 직접 비교 한계

MISSFormer가 scratch로 strong baseline을 넘는다는 점은 인상적이지만,
더 큰 규모의 self-supervised pretraining이나 foundation model
adaptation과 직접 비교한 것은 아니다. 현재 기준으로는 후속 foundation
model 계열과의 비교가 필요하다.

## 실무적 또는 연구적 인사이트

### 1. 의료영상 Transformer에서 핵심은 pure vs hybrid가 아니라

### global-local balance다

이 논문은 pure transformer도 충분히 경쟁력이 있을 수 있음을 보이지만,
그 전제는 local context 보완이 매우 정교해야 한다는 것이다.
따라서 의료영상 분할에서 중요한 질문은 "Transformer를 쓰느냐"보다
"전역 문맥과 국소 경계를 어떻게 동시에 다루느냐"에 가깝다.

### 2. FFN 설계가 segmentation 품질에 미치는 영향이 크다

많은 Transformer 논문은 attention에만 초점을 맞추지만, MISSFormer는 FFN이
실제 segmentation 품질을 크게 좌우할 수 있음을 보여준다. 특히 feature
alignment를 위한 LayerNorm 위치와 skip design은 후속 구조 설계에서도
참고할 가치가 있다.

### 3. Multi-scale bridge는 의료영상에서 특히 중요하다

장기 분할이나 심장 구조 분할처럼 크기와 형태가 다양한 객체가 섞인
문제에서는 단일 bottleneck feature보다 multi-scale feature integration이
더 유효하다. MISSFormer의 bridge는 이 점을 Transformer 방식으로 구현한
사례로 볼 수 있다.

### 4. scratch 학습 강점은 의료 도메인 실용성과 연결된다

의료영상에서는 자연영상 pretraining이 항상 잘 맞지 않는다. MISSFormer가
ImageNet 사전학습 없이도 강한 성능을 보였다는 점은, 도메인 격차가 큰
의료영상 문제에서 architecture 자체의 inductive bias가 여전히 중요하다는
사실을 보여준다.

## 종합 평가

`MISSFormer: An Effective Medical Image Segmentation Transformer`는
의료영상 분할용 Transformer가 단순히 self-attention을 도입하는 수준을
넘어, local context와 multi-scale feature aggregation까지 함께
설계해야 한다는 점을 분명하게 보여준 논문이다. 핵심은
`Enhanced Mix-FFN`과 `Enhanced Transformer Context Bridge`이며,
이 두 요소가 pure transformer U-shaped 구조를 실제로 강하게 만든다.

실험적으로는 Synapse와 ACDC에서 강한 결과를 보였고, 특히 scratch
training으로 기존 사전학습 모델을 넘어섰다는 점이 눈에 띈다.
다만 데이터셋 규모, 2D 중심 검증, 계산량 문제는 남아 있다.
그럼에도 이 논문은 의료영상 segmentation transformer 설계에서
`feature discrimination`, `global-local fusion`, `multi-scale bridge`가
왜 중요한지를 구조적으로 잘 보여주는 초기 대표작으로 평가할 수 있다.
