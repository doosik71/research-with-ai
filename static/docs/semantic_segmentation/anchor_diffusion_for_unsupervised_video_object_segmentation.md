# Anchor Diffusion for Unsupervised Video Object Segmentation

- **저자**: Zhao Yang, Qiang Wang, Luca Bertinetto, Weiming Hu, Song Bai, Philip H.S. Torr
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1910.10895

## 1. 논문 개요

이 논문은 **unsupervised video object segmentation (VOS)** 문제를 다룬다. 목표는 비디오의 각 프레임에서 별도의 테스트 시점 supervision 없이 foreground object를 픽셀 단위로 분할하는 것이다. 여기서 foreground는 보통 화면에서 충분히 크고, 움직이거나 시선을 끌 가능성이 높으며, 장면의 중심부에 있는 물체를 뜻한다.

저자들이 제기하는 핵심 문제는, 기존 unsupervised VOS 방법들이 주로 **optical flow** 또는 **RNN** 기반의 순차적 시간 모델링에 의존한다는 점이다. 이런 방식은 현재 프레임의 예측이 이전 프레임들의 누적 결과에 의존하게 만들어, 시간이 지날수록 오차가 누적되고 **drift**가 발생하기 쉽다. 논문은 바로 이 지점, 즉 “긴 시간 범위의 temporal dependency를 어떻게 더 안정적으로 다룰 것인가”를 중심 문제로 삼는다.

이 문제는 중요하다. 비디오 객체 분할은 자율주행, 로봇 조작, 감시, 비디오 편집 같은 응용에서 기본적인 역할을 한다. 특히 unsupervised setting에서는 테스트 시점의 초기 마스크조차 없기 때문에, 장기적인 일관성을 유지하면서도 foreground를 정확히 찾아내는 능력이 더욱 중요하다. 논문은 기존의 복잡한 temporal propagation보다 더 단순하면서도 효과적인 대안을 제안한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **첫 프레임(anchor frame)** 과 현재 프레임 사이에서 직접 **dense pixel correspondence**를 학습해, 중간 프레임을 거치지 않고도 장기적인 temporal dependency를 모델링하자는 것이다. 즉, 이전 프레임들을 순차적으로 따라가며 정보를 전달하는 대신, 고정된 기준 프레임 하나와 현재 프레임을 바로 연결한다.

이 설계의 직관은 분명하다. foreground object는 비디오 전체에서 어느 정도 지속적으로 나타나는 반면, background는 더 자주 바뀐다. 따라서 anchor frame과 현재 프레임 사이의 픽셀 유사도를 잘 학습하면, anchor에 공통적으로 존재하는 foreground 신호는 강화되고, 일시적이거나 변동이 큰 background는 약화될 수 있다.

기존 접근과의 차별점은 다음과 같다. 첫째, optical flow처럼 한 단계씩 이동 벡터를 누적하지 않으므로 장기 drift를 줄일 수 있다. 둘째, RNN처럼 긴 시퀀스 학습에서 발생하는 exploding/vanishing gradient 문제를 피한다. 셋째, seed selection, ranking, separate motion/objectness score calibration 같은 복잡한 절차 없이, **similarity learning, feature propagation, binary segmentation**을 하나의 end-to-end 네트워크로 통합한다. 저자들은 이 구조를 **Anchor Diffusion Network (AD-Net)** 라고 부른다.

## 3. 상세 방법 설명

전체 입력은 두 장의 이미지, 즉 **anchor frame** $I_0$ 와 분할 대상인 현재 프레임 $I_t$ 이다. 실제 구현에서는 계산 편의성과 벤치마크 조건 때문에 anchor로 항상 첫 프레임을 사용한다. 학습 시에는 첫 프레임과 임의의 다른 프레임을 짝으로 샘플링한다.

두 프레임은 공통 **feature encoder**인 DeepLabv3를 통해 각각 임베딩 $X_0 \in \mathbb{R}^{hw \times c}$, $X_t \in \mathbb{R}^{hw \times c}$ 로 변환된다. 여기서 $h, w$ 는 feature map의 공간 해상도이고, $c$ 는 채널 수이다. 각 위치의 $c$ 차원 벡터를 논문은 **pixel embedding**이라 부른다.

이후 네트워크는 세 갈래로 나뉜다.

첫 번째는 **skip connection**이다. 현재 프레임의 embedding $X_t$ 를 그대로 뒤쪽으로 전달한다. 이는 원래 appearance 정보를 보존하는 역할을 한다.

두 번째는 **intra-frame branch**이다. 이는 현재 프레임 내부에서 non-local operation을 적용해, 한 픽셀이 같은 프레임의 다른 픽셀들과 얼마나 관련 있는지를 반영한다. 쉽게 말해, 현재 프레임 안에서 전역적인 문맥 정보를 모으는 장치다. 저자들은 이 branch가 개별 프레임에서 segmentation accuracy를 높인다고 본다.

세 번째가 핵심인 **anchor-diffusion branch**이다. 여기서는 anchor frame과 현재 프레임의 모든 픽셀 쌍 사이 유사도를 계산해 **transition matrix** $P \in \mathbb{R}^{hw \times hw}$ 를 만든다. 이 행렬은 다음과 같이 정의된다.

$$
P = \text{softmax}\left(\frac{1}{z} X_0 X_t^T\right)
$$

여기서 $X_0 X_t^T$ 는 anchor의 각 픽셀 embedding과 현재 프레임의 각 픽셀 embedding 사이의 pairwise dot-product similarity이다. 스케일 계수는

$$
z = \sqrt{c}
$$

로 둔다. 이는 Transformer의 scaled dot-product attention과 같은 이유로, 채널 수가 클 때 dot product 값이 과도하게 커져 softmax가 포화되는 현상을 줄이기 위해서다.

이렇게 얻은 $P$ 를 사용해 현재 프레임 embedding을 다시 변환한다.

$$
\tilde{X}_t = P X_t
$$

이 식의 의미를 쉬운 말로 풀면, 현재 프레임의 각 위치 feature가 anchor와의 유사도에 따라 재가중되어 다시 표현된다는 뜻이다. 논문은 이 과정을 통해 foreground에 대응하는 feature는 강화되고, anchor와 잘 대응되지 않는 background는 약화된다고 설명한다. 실제 qualitative visualization에서도 foreground car에 해당하는 anchor pixel은 현재 프레임의 같은 물체 위치에 높은 similarity를 보이고, distractor object나 road 같은 background 관련 pixel은 배경 영역 쪽에 더 높은 similarity를 보인다.

세 branch의 출력은 채널 방향으로 concatenate한 뒤, $1 \times 1$ convolution으로 128차원으로 줄인다. 여기에는 LeakyReLU와 dropout 0.1이 사용된다. 마지막 분류기는 single-channel $1 \times 1$ convolution 뒤에 sigmoid를 붙여 각 픽셀의 foreground 확률을 출력한다. 학습 손실은 **binary cross-entropy loss**이다.

학습 설정도 비교적 명확하다. encoder는 ResNet101 backbone을 가진 DeepLabv3이고, backbone은 ImageNet pretrained weight로 초기화한다. 마지막 출력 채널 수는 128로 바꾼다. 최적화는 SGD, weight decay는 $0.0005$, 초기 learning rate는 $0.005$ 이다. learning rate는 poly schedule을 따른다.

$$
\text{lr} = 0.005 \left(1 - \frac{\text{iter}}{40000}\right)^{0.9}
$$

학습은 30,000 iteration, batch size 8로 진행한다. 입력은 foreground를 포함하는 랜덤 크롭과 회전 augmentation을 적용한다.

추론 시에는 anchor frame feature를 한 번만 계산하고 비디오 전체에서 재사용한다. 입력 스케일 $0.75$, $1.0$, $1.5$ 와 horizontal flip을 사용해 multi-scale inference를 수행하고, 모든 heatmap의 평균을 취한 뒤 0.5 threshold로 binary mask를 만든다.

추가로 DAVIS-2016처럼 실제로는 여러 객체가 보일 수 있지만 하나의 foreground만 정답인 경우를 위해, 저자들은 **instance pruning**이라는 후처리를 실험했다. 이는 pretrained object detector로 객체를 검출하고, box trajectory를 따라가며 작고 정적인 객체를 제거하는 방식이다. 다만 이 단계는 본체 네트워크의 핵심이 아니라 후처리이다.

## 4. 실험 및 결과

논문은 DAVIS-2016, FBMS, ViSal 세 데이터셋에서 실험한다. DAVIS-2016은 unsupervised VOS의 대표 벤치마크이며, training 30개, validation 20개 비디오로 구성되고 한 비디오당 단일 foreground를 annotation한다. FBMS는 더 복잡한 unsupervised VOS 벤치마크로, 여러 foreground object가 가능하다. ViSal은 본래 video saliency detection 데이터셋이지만, foreground/salient object의 정의가 유사해 함께 평가한다.

평가 지표는 DAVIS에서 region similarity인 $J$ 와 contour accuracy인 $F$ 를 사용한다. $J$ 는 IoU이고, $F$ 는 contour 기반 F-measure이다. FBMS에서는 주로 F-measure를, ViSal에서는 MAE와 F-measure를 사용한다.

ablation study가 이 논문의 주장을 잘 뒷받침한다. 단순히 DeepLabv3를 DAVIS에 fine-tune한 static segmentation baseline이 이미 $J = 75.41$, $F = 75.58$ 로 상당히 강하다. 이것은 기존 unsupervised VOS가 temporal dependency를 충분히 잘 활용하지 못하고 있음을 시사한다. 여기에 intra-frame branch만 추가하면 $J$ 가 76.17로 오른다. anchor를 단순 concat하면 76.84, anchor-diffusion을 쓰면 77.43, 최종 AD-Net(single scale)은 78.26까지 올라간다. 특히 baseline 대비 **anchor-diffusion만으로 $+2.02$ 포인트**, 최종 모델로는 **$+2.85$ 포인트**의 절대 향상이 나타난다. 이는 단순히 anchor frame을 넣는 것보다, anchor와 current 사이 correspondence를 학습하는 방식이 더 중요하다는 증거다.

state-of-the-art 비교에서도 결과가 강하다. DAVIS-2016 validation에서 AD-Net은 multi-scale 기준 $J = 79.4$, $F = 78.2$ 를 기록하고, instance pruning 적용 후 최종적으로 **$J = 81.7$, $F = 80.5$** 를 달성한다. 논문에 따르면 이는 당시 unsupervised 방법 중 1위이며, MotAdapt 대비 $J$ 에서 4.5포인트, $F$ 에서 3.1포인트 앞선다. 주목할 점은 일부 semi-supervised 방법은 첫 프레임 mask를 입력으로 받고도 AD-Net보다 못한 성능을 보였다는 것이다.

FBMS에서는 Table 2 기준 **F-measure 81.2** 를 기록해 경쟁력 있는 결과를 낸다. ViSal에서는 saliency용으로 별도 학습하지 않았음에도 **MAE 0.030, F-measure 90.4** 를 달성한다. Table 3에서 보면 DAVIS와 FBMS의 saliency metrics에서도 기존 saliency 방법들을 여러 경우 앞선다. 이는 AD-Net이 foreground object를 안정적으로 강조하는 방식이 saliency detection 관점에서도 유효함을 보여준다.

시간에 따른 안정성 분석도 인상적이다. 논문은 foreground embedding이 첫 프레임 embedding과 얼마나 멀어지는지를 cosine distance로 측정했는데, baseline보다 AD-Net의 embedding이 시간이 지나도 훨씬 더 안정적으로 유지된다. 또한 DAVIS detailed metric(Table 4)에서 $J$ decay 2.2, $F$ decay 0.6으로 비교적 낮아, 장기 시퀀스에서 drift가 적음을 정량적으로 보여준다.

속도 측면에서도 장점이 있다. optical flow나 online fine-tuning 없이, instance pruning을 제외하면 원본 DAVIS 해상도 $854 \times 480$ 에서 NVIDIA TITAN X 기준 약 4 fps로 동작한다고 한다. 즉, 성능뿐 아니라 구조 단순성과 추론 효율성도 함께 확보했다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 방법 설계가 잘 맞아떨어진다는 점이다. 저자들은 기존 방법의 약점이 “순차적 temporal propagation으로 인한 drift”라고 진단했고, 그에 대한 해법으로 anchor frame과의 직접 correspondence 학습을 제안했다. 이 아이디어는 구조적으로 단순하면서도 실험적으로 강하게 뒷받침된다. 특히 ablation study에서 baseline, intra-frame, anchor, anchor-diffusion을 단계적으로 비교해 각 구성 요소의 기여를 분명히 보여준다.

또 다른 강점은 **optical flow 없이도 높은 성능**을 달성했다는 점이다. 이는 synthetic-to-real domain gap, static foreground 처리 문제, synchronized motion ambiguity 같은 optical flow 기반 방법의 구조적 문제를 우회하게 해준다. 또한 seed selection이나 graph energy minimization 같은 복잡한 파이프라인 없이 end-to-end 학습이 가능하다는 점도 실용적이다.

장기적 일관성에 관한 분석도 설득력이 있다. 단순히 leaderboard 수치만 제시하지 않고, embedding distance 변화와 decay metric을 통해 시간이 지날수록 얼마나 안정적인지 보인 점이 좋다. 이는 논문의 핵심 주장인 long-term temporal dependency modeling을 직접적으로 지지한다.

다만 한계도 분명하다. 첫째, anchor를 항상 첫 프레임으로 두는 가정은 benchmark에서는 적절하지만, 실제 응용에서는 첫 프레임이 항상 foreground를 잘 담고 있다고 보장하기 어렵다. 논문도 사실상 이 설정을 benchmark 조건에 맞춰 사용한다. 둘째, foreground가 비디오 초반 이후 크게 변형되거나 오랫동안 가려지거나, 첫 프레임에 없던 중요한 객체가 뒤늦게 등장하는 경우에는 anchor 중심 구조가 약해질 가능성이 있다. 이 가능성은 논문에서 명시적으로 깊게 분석되지는 않았다.

셋째, instance pruning은 성능 향상에 기여하지만, pretrained object detector와 tracking heuristic에 의존하는 후처리이다. 따라서 최종 leaderboard 수치 일부는 순수한 segmentation network 자체의 성능이라기보다, detection 기반 후처리의 도움을 받은 결과라고 보는 것이 정확하다. 물론 논문은 pruning 전 성능도 함께 제시하므로 이 점을 숨기지는 않는다.

넷째, method의 직관은 foreground가 anchor와 지속적으로 대응된다는 전제에 크게 기대고 있다. background가 반복적으로 나타나거나 foreground와 background의 appearance가 매우 유사한 경우, correspondence가 얼마나 안정적으로 분리되는지는 제한적으로만 보여준다. 논문은 qualitative example에서 distractor를 잘 억제한다고 보이지만, 이런 실패 사례 분석은 충분히 많지 않다.

종합하면, 이 논문은 “복잡한 temporal modeling이 꼭 필요한가?”라는 중요한 질문을 던지고, 상당히 설득력 있게 “직접적인 anchor-based similarity propagation이 더 낫다”는 답을 제시한다. 다만 그 강점은 DAVIS류의 foreground 정의와 첫 프레임 anchor 가정 위에서 특히 잘 드러난다고 보는 것이 공정하다.

## 6. 결론

이 논문의 주요 기여는 **Anchor Diffusion Network (AD-Net)** 를 제안해, unsupervised VOS에서 long-term temporal dependency를 단순한 **anchor-to-current pixel correspondence** 방식으로 모델링했다는 점이다. 이 방법은 non-local operation을 이용해 anchor frame과 현재 프레임 사이의 dense similarity를 학습하고, 이를 통해 foreground는 강화하고 background는 억제한다. 결과적으로 optical flow나 RNN에 의존하지 않고도 높은 정확도와 시간적 일관성을 달성했다.

실험적으로는 DAVIS-2016에서 당시 unsupervised 방법 중 최고 성능을 기록했고, FBMS와 ViSal에서도 매우 경쟁력 있는 결과를 보였다. 특히 static image segmentation baseline보다 분명히 낫고, 일부 semi-supervised 방법과도 경쟁 가능하다는 점이 중요하다.

향후 연구 관점에서 이 논문은 두 가지 의미를 가진다. 첫째, 비디오 문제에서도 복잡한 순차 모델 대신 **attention/non-local correspondence 기반 장기 연결**이 더 효과적일 수 있다는 점을 보여준다. 둘째, VOS와 video saliency 사이의 경계가 생각보다 가깝다는 점을 실험적으로 시사한다. 실제 적용 면에서는 online fine-tuning 없이도 비교적 빠르게 동작하므로, 실시간성이나 계산 효율이 중요한 시스템에서도 의미 있는 출발점이 될 가능성이 크다.
