# CCNet: Criss-Cross Attention for Semantic Segmentation

- **저자**: Zilong Huang, Xinggang Wang, Yunchao Wei, Lichao Huang, Humphrey Shi, Wenyu Liu, Thomas S. Huang
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/1811.11721

## 1. 논문 개요

이 논문은 semantic segmentation에서 중요한 문제인 `문맥 정보(contextual information)`를 더 효율적으로 모으는 방법을 제안한다. 기존의 fully convolutional network(FCN) 기반 방법은 convolution의 지역적 receptive field에 크게 의존하기 때문에, 멀리 떨어진 픽셀 사이의 관계를 충분히 반영하지 못한다. semantic segmentation은 각 픽셀을 정확히 분류해야 하므로, 단순히 주변 이웃만 보는 정보로는 경계가 복잡한 객체나 넓게 퍼진 동일 물체를 안정적으로 구분하기 어렵다.

기존에는 atrous convolution, pyramid pooling, non-local attention 같은 방식으로 이 문제를 해결하려고 했다. 그러나 dilated convolution 계열은 실제로 모든 위치와의 dense한 관계를 모델링하지 못하고, pooling 기반 방법은 모든 픽셀에 동일한 방식으로 context를 적용하는 비적응적 한계가 있다. 반면 non-local network는 모든 픽셀 쌍의 관계를 직접 계산해 full-image dependency를 얻을 수 있지만, 계산량과 메모리 사용량이 $O(N^2)$로 커져 dense prediction에서는 매우 비싸다.

이 논문의 핵심 문제는 바로 이것이다. 즉, `full-image contextual information`은 필요하지만, non-local처럼 모든 픽셀 쌍을 직접 연결하는 방식은 너무 비효율적이다. 저자들은 이 문제를 해결하기 위해, 한 번에 모든 픽셀과 연결하지 않고 `sparse한 연결을 반복적으로 적용`하여 결국 전체 이미지 문맥을 얻는 구조를 제안한다. 그 결과 semantic segmentation뿐 아니라 human parsing, instance segmentation, video segmentation에서도 성능 향상을 보였다고 주장한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 `Criss-Cross Attention`이다. 각 픽셀은 한 번의 attention에서 이미지 전체를 보지 않고, 자기와 같은 행(row)과 열(column)에 있는 픽셀들만 본다. 즉, 십자 모양(criss-cross path)으로 연결된 위치들만 대상으로 attention을 계산한다. 이렇게 하면 한 픽셀이 직접 참조하는 위치 수는 전체 $N$개가 아니라 대략 $H + W - 1$개가 되므로 훨씬 가볍다.

하지만 한 번의 criss-cross attention만으로는 대각선 방향처럼 같은 행·열에 속하지 않는 픽셀과는 직접 정보를 주고받지 못한다. 이를 해결하기 위해 저자들은 같은 모듈을 `recurrent`하게 두 번 적용한다. 첫 번째 루프에서 어떤 픽셀은 자신의 행·열 문맥을 모으고, 두 번째 루프에서는 이미 문맥이 섞인 feature를 다시 criss-cross attention에 넣음으로써 간접적으로 이미지 전체와 연결된다. 논문은 이 구조를 `RCCA(Recurrent Criss-Cross Attention)`라고 부른다.

기존 non-local과의 차별점은 분명하다. non-local은 한 번에 dense attention map을 만들고 모든 픽셀 쌍 관계를 계산한다. 반면 CCNet은 sparse attention을 두 번 반복하여 비슷한 효과를 더 적은 자원으로 얻는다. 저자들은 이로 인해 시간/공간 복잡도를 $O(N^2)$에서 $O(N\sqrt{N})$ 수준으로 줄일 수 있다고 설명한다. 또한 attention 기반 context aggregation 자체는 유지하면서도, full-image dependency를 훨씬 메모리 친화적으로 구현했다는 점을 강하게 내세운다.

추가로 저자들은 `Category Consistent Loss (CCL)`를 도입한다. attention으로 문맥을 강하게 섞으면 feature가 과도하게 평균화(over-smoothing)될 수 있는데, 이를 막기 위해 같은 class 픽셀의 feature는 가깝게, 다른 class 픽셀의 feature는 멀어지도록 직접 학습시킨다. 즉, attention 구조와 feature-space regularization을 함께 사용한 것이 이 논문의 또 다른 설계 포인트다.

## 3. 상세 방법 설명

전체 네트워크는 backbone CNN 위에 RCCA 모듈을 붙이는 구조다. 입력 이미지를 ResNet-101 같은 backbone에 통과시켜 feature map $X$를 만든다. 저자들은 segmentation에서 해상도를 유지하기 위해 마지막 두 번의 downsampling을 제거하고 dilated convolution을 사용하여 output stride를 8로 맞춘다. 이후 channel을 줄인 feature $H$를 만들고, 여기에 criss-cross attention을 두 번 반복 적용하여 $H'$와 $H''$를 얻는다. 마지막으로 $H''$를 원래 local feature $X$와 concatenate한 뒤 몇 개의 convolution으로 fusion하고 segmentation head로 보낸다.

### Criss-Cross Attention

입력 feature map을 $H \in \mathbb{R}^{C \times W \times H}$라고 하자. 먼저 $1 \times 1$ convolution 세 개를 사용해 $Q$, $K$, $V$를 만든다. 여기서 $Q$와 $K$는 channel-reduced representation이고, $V$는 aggregation에 사용할 adapted feature다.

각 위치 $u$에 대해, $Q_u$는 query vector이고, $\Omega_u$는 $u$와 같은 행 또는 열에 있는 모든 key vector들의 집합이다. attention score는 다음과 같이 계산된다.

$$
d_{i,u} = Q_u \Omega_{i,u}^{\top}
$$

여기서 $d_{i,u}$는 위치 $u$와 그 criss-cross path 상의 $i$번째 위치 사이의 상관도다. 이렇게 얻은 score들에 softmax를 적용해 attention map $A$를 얻는다. 그 다음, 같은 행·열에 있는 $V$의 feature들을 attention weight로 가중합하여 출력 feature를 만든다.

$$
H'_u = \sum_{i=0}^{H+W-1} A_{i,u}\Phi_{i,u} + H_u
$$

여기서 $\Phi_u$는 $V$에서 $u$와 같은 행·열에 있는 feature들의 집합이다. 마지막의 $+H_u$는 residual connection 역할을 한다. 이 연산의 의미는 간단하다. 각 픽셀은 자기 주변의 작은 convolution window만 보는 대신, 자기 행과 열 전체를 따라 중요한 위치를 선택적으로 참조하게 된다.

### RCCA: Recurrent Criss-Cross Attention

한 번의 criss-cross attention은 row/column 방향 정보만 모으므로 full-image dependency가 완전하지 않다. 그래서 같은 모듈을 두 번 반복한다. 첫 번째 루프에서 한 위치는 자기와 직접 같은 행·열에 있는 픽셀들의 정보를 모은다. 두 번째 루프에서는 이미 한 번 context가 섞인 feature를 다시 입력받기 때문에, 원래 직접 연결되지 않았던 위치의 정보도 간접적으로 전달될 수 있다.

논문은 그림과 함께, 어떤 위치 $\theta=(\theta_x,\theta_y)$가 위치 $u=(u_x,u_y)$의 criss-cross path에 직접 없더라도, 첫 번째 루프에서 $(u_x,\theta_y)$ 또는 $(\theta_x,u_y)$ 같은 중간 위치로 정보가 이동하고, 두 번째 루프에서 그 정보가 다시 $u$로 전달된다고 설명한다. 즉, 두 번의 sparse propagation으로 dense dependency를 근사하는 방식이다. 논문에서는 기본 설정으로 $R=2$를 사용하며, 두 attention module은 parameter를 공유한다.

### Category Consistent Loss

저자들은 RCCA가 전역 문맥을 강하게 집계하는 대신 feature를 지나치게 smooth하게 만들 수 있다고 본다. 이를 보완하기 위해 segmentation loss 외에 category consistent loss를 추가한다. 목표는 같은 category에 속한 픽셀 feature는 같은 cluster에 모으고, 다른 category의 cluster는 서로 멀어지게 하는 것이다.

손실은 세 부분으로 구성된다.

첫째, 같은 class 내부에서 feature가 class mean 주변에 모이도록 하는 intra-class term:

$$
\ell_{var} = \frac{1}{|C|}\sum_{c \in C}\frac{1}{N_c}\sum_{i=1}^{N_c}\phi_{var}(h_i,\mu_c)
$$

둘째, 서로 다른 class의 평균 feature들이 너무 가까워지지 않도록 하는 inter-class separation term:

$$
\ell_{dis} = \frac{1}{|C|(|C|-1)}\sum_{c_a \in C}\sum_{c_b \in C, c_a \neq c_b}\phi_{dis}(\mu_{c_a}, \mu_{c_b})
$$

셋째, class mean이 무한정 커지지 않도록 하는 regularization term:

$$
\ell_{reg} = \frac{1}{|C|}\sum_{c \in C}\|\mu_c\|
$$

최종 손실은 다음과 같다.

$$
\ell = \ell_{seg} + \alpha \ell_{var} + \beta \ell_{dis} + \gamma \ell_{reg}
$$

논문에서 사용한 하이퍼파라미터는 $\delta_v=0.5$, $\delta_d=1.5$, $\alpha=\beta=1$, $\gamma=0.001$이다.

여기서 중요한 점은 $\ell_{var}$에 쓰이는 거리 함수 $\phi_{var}$를 단순 quadratic이 아니라 piece-wise 함수로 설계했다는 것이다.

$$
\phi_{var} =
\begin{cases}
\|\mu_c - h_i\| - \delta_d + (\delta_d - \delta_v)^2, & \|\mu_c - h_i\| > \delta_d \\
(\|\mu_c - h_i\| - \delta_v)^2, & \delta_v < \|\mu_c - h_i\| \le \delta_d \\
0, & \|\mu_c - h_i\| \le \delta_v
\end{cases}
$$

즉, 중심에 충분히 가까우면 벌점을 주지 않고, 중간 거리에서는 quadratic penalty를 주며, 너무 멀어지면 linear 형태로 완화한다. 논문은 이 설계가 학습 안정성을 높인다고 주장한다.

### 3D Criss-Cross Attention

논문은 2D attention을 video segmentation에 맞게 3D로 확장한다. 입력을 $H \in \mathbb{R}^{C \times T \times W \times H}$로 두고, 공간 차원뿐 아니라 시간 차원까지 같이 본다. 이제 한 위치 $u=(t,x,y)$는 같은 time-column-row 구조를 따라 연결된 위치들과 attention을 계산한다. attention 대상 수는 $T + H + W - 2$개가 된다.

score와 aggregation은 2D와 같은 방식으로 정의된다.

$$
d_{i,u} = Q_u \Omega_{i,u}^{\top}
$$

$$
H'_u = \sum_{i=0}^{T+H+W-2} A_{i,u}\Phi_{i,u} + H_u
$$

의미상으로는 spatial context와 temporal context를 동시에 모으는 구조다. CamVid 실험에서 이 3D 버전의 효과를 검증한다.

## 4. 실험 및 결과

논문은 Cityscapes, ADE20K, LIP, COCO, CamVid에서 실험했다. semantic segmentation 계열 평가는 주로 mIoU를 사용했고, COCO instance segmentation은 AP를 사용했다.

Cityscapes는 2,975/500/1,525장의 train/val/test split을 사용했고, ADE20K는 20k/2k/3k, LIP는 30k/10k/10k, COCO는 115k/5k/20k, CamVid는 별도 프로토콜에 따른 367/101/233 split을 사용했다. semantic segmentation backbone은 주로 ImageNet pre-trained ResNet-101이며, output stride 8 설정을 사용했다. Cityscapes에서는 random scaling 0.75에서 2.0, crop size $769 \times 769$를 사용했고, ADE20K는 short side를 여러 크기 중 하나로 resize했다. 학습은 SGD와 poly learning rate policy를 사용했다.

### Cityscapes

가장 핵심적인 결과 중 하나는 Cityscapes test에서 CCNet이 mIoU 81.9%를 기록했다는 점이다. 논문 표에 따르면 이는 PSPNet, PSANet, DenseASPP 등 당시 강한 baseline들보다 높다. val set에서도 ResNet-101 backbone 기반 CCNet은 80.5%를 기록한다.

Ablation에서 RCCA 반복 횟수의 효과가 분명하게 나타난다. baseline은 75.1%, $R=1$은 78.0%, $R=2$는 79.8%, $R=3$는 80.2%였다. 즉 첫 번째 attention 추가만으로도 큰 이득이 있고, 두 번째 루프가 dense context를 보완해 추가 상승을 만든다. 다만 $R=2$에서 $R=3$으로 갈 때 향상은 0.4%로 작고, FLOPs와 메모리는 계속 증가한다. 그래서 논문은 실용적인 기본값으로 $R=2$를 선택한다.

또 다른 중요한 비교는 non-local과의 자원 효율성이다. ResNet-50 기준으로 baseline 대비 non-local 1회는 108 GFLOPs, 메모리 1411MB 증가, mIoU 77.3%였고, non-local 2회는 216 GFLOPs, 2820MB, 78.7%였다. 반면 RCCA($R=2$)는 16.5 GFLOPs, 127MB, 78.5%였다. 즉 성능은 비슷하거나 약간 낮은 수준에서, 메모리는 약 11배 적고 계산량도 크게 줄어든다. 논문이 주장하는 “효율적인 full-image dependency”가 여기서 가장 설득력 있게 드러난다.

Context aggregation 방식 비교에서도 RCCA는 강하다. ResNet-101에서 baseline 75.1%, +GCN 78.1%, +PSP 78.5%, +ASPP 78.9%, +NL 79.1%, +RCCA($R=2$) 79.8%, +RCCA+CCL 80.5%였다. 즉 RCCA 자체가 non-local보다 좋았고, CCL까지 더하면 추가 향상이 있었다.

### Category Consistent Loss 효과

CCL은 Cityscapes val에서 ResNet-101과 ResNet-50 모두에 대해 약 0.7% 정도 mIoU 향상을 가져왔다. 또한 quadratic-only loss와 proposed piece-wise loss를 비교한 실험에서, piece-wise function은 10회 중 9회 학습 성공, quadratic은 10회 중 6회 성공이었다. 평균 mIoU도 piece-wise가 약간 높았다. 논문은 이를 근거로 piece-wise penalty가 학습 안정성을 높인다고 해석한다.

### ADE20K

ADE20K validation에서 CCNet은 45.76% mIoU를 기록했다. 표에 따르면 EncNet의 44.65%, PSANet의 43.77%, PSPNet의 43.29%보다 높다. 논문은 conference version보다도 0.5% 높아졌다고 설명한다. 이 결과는 scene parsing처럼 클래스 종류가 많고 장면 구성이 복잡한 데이터에서도 full-image context modeling이 유효하다는 근거로 제시된다.

### LIP

LIP human parsing에서는 CE2P 프레임워크에서 context embedding module을 RCCA로 교체하고 CCL을 추가해 평가했다. 결과는 pixel accuracy 88.01, mean accuracy 63.91, mIoU 55.47이었다. 이전 강한 baseline인 CE2P의 53.10 mIoU보다 2.37% 높다. 논문은 복잡한 자세에서도 더 정확한 parsing을 보여준다고 정성 결과와 함께 제시한다.

### COCO

COCO instance segmentation에서는 Mask R-CNN backbone 내부에 RCCA를 삽입했다. 이 실험에서는 공정한 비교를 위해 CCL은 사용하지 않았다. ResNet-50에서는 baseline이 box AP 38.2, mask AP 34.8이고, +NL은 39.0/35.5, +RCCA는 39.3/36.1이다. ResNet-101에서는 baseline 40.1/36.2, +NL 40.8/37.1, +RCCA 41.0/37.3이다. 즉 segmentation 외의 dense prediction 성격이 강한 instance segmentation에서도 RCCA가 일관된 이득을 보였다.

### CamVid

CamVid video segmentation에서는 3D-RCCA를 평가했다. CCNet3D($T=1$)은 77.9%, CCNet3D($T=5$)는 79.1% mIoU를 기록해 VideoGCRF 75.2%를 넘었다. 특히 입력 프레임 수를 1장에서 5장으로 늘렸을 때 1.2% 향상된 것은 3D criss-cross attention이 temporal context까지 잘 활용했음을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 `효율성과 성능의 균형`이다. non-local attention의 장점인 full-image dependency modeling을 유지하면서도, 직접적인 모든 쌍 연결 대신 sparse recurrent attention을 사용해 메모리와 계산량을 크게 줄였다. 단순히 이론적인 주장에 그치지 않고, Cityscapes에서 FLOPs와 GPU memory 수치를 함께 제시해 실용성을 뒷받침한 점이 좋다.

또한 방법이 구조적으로 단순하다. backbone feature 위에 RCCA를 삽입하는 방식이라 기존 FCN 계열이나 Mask R-CNN 같은 구조에 비교적 쉽게 붙일 수 있다. 논문에서도 semantic segmentation, human parsing, instance segmentation, video segmentation까지 폭넓게 적용해 generality를 보여준다. CCL 역시 over-smoothing 문제를 겨냥한 보완책으로 논리적 일관성이 있다.

실험 설계도 비교적 충실하다. loop 수, CCL 유무, loss 함수 형태, 다른 context aggregation 방법과의 비교, non-local과의 효율성 비교 등 ablation이 충분히 포함되어 있다. 특히 `+HV`, `+HV&VH`와 같은 변형까지 넣어 criss-cross 구조 자체의 효과를 검증하려 한 점은 설득력이 있다.

한편 한계도 분명하다. 첫째, full-image dependency를 “정확히” 직접 계산하는 것이 아니라 두 번의 sparse propagation으로 간접적으로 얻는 구조이므로, 정보 전달 품질은 recurrent propagation에 의존한다. 논문은 $R=2$면 충분하다고 보이지만, 왜 두 번이면 충분한지에 대한 이론적 보장은 강하지 않다. 설명은 직관적이고 그림 기반이지만, 더 일반적인 분석은 제한적이다.

둘째, CCL은 feature clustering을 강제하는 추가 손실이므로 하이퍼파라미터 $\delta_v$, $\delta_d$, $\alpha$, $\beta$, $\gamma$에 민감할 가능성이 있다. 논문은 특정 값들을 제시하지만, 이 값들의 민감도 분석은 제공하지 않는다. 또한 class imbalance나 noisy label 상황에서 이 loss가 어떻게 동작하는지는 명시되지 않았다.

셋째, 논문은 효율성을 non-local과 비교해 강하게 주장하지만, backbone 전체 기준 실시간성이나 latency 관점의 자세한 분석은 제한적이다. 예를 들어 실제 wall-clock inference speed, 다양한 해상도에서의 scaling, multi-head/self-attention 계열과의 더 넓은 비교는 포함되지 않았다. 따라서 “효율적”이라는 표현은 주로 module-level FLOPs와 메모리 관점에서 해석하는 것이 정확하다.

넷째, 3D extension은 흥미롭지만 CamVid 실험만으로 검증되며, 더 큰 video benchmark나 longer temporal window에서의 분석은 없다. 따라서 3D-RCCA의 확장성이 충분히 검증되었다고 보기는 어렵다.

종합하면, 이 논문은 매우 강한 engineering-oriented 아이디어를 제시하고 실험적으로 설득력 있게 보이지만, 이론적 분석이나 광범위한 조건 변화에 대한 robustness 분석은 상대적으로 제한적이다.

## 6. 결론

이 논문은 semantic segmentation을 비롯한 dense prediction 문제에서 중요한 full-image context modeling을 더 효율적으로 수행하기 위해 `Criss-Cross Attention`과 `Recurrent Criss-Cross Attention`을 제안했다. 각 픽셀이 같은 행과 열의 정보만 먼저 수집하고, 이를 반복 적용해 결국 전체 이미지 문맥을 반영하도록 만든 것이 핵심이다. 여기에 `Category Consistent Loss`를 더해 feature의 class-level discriminability를 강화했다.

주요 기여는 세 가지로 정리할 수 있다. 첫째, non-local 대비 훨씬 적은 계산량과 메모리로 full-image dependency를 근사하는 attention 구조를 설계했다. 둘째, over-smoothing을 줄이고 같은 클래스 feature를 더 응집시키는 CCL을 도입했다. 셋째, Cityscapes, ADE20K, LIP, COCO, CamVid에서 강력한 실험 결과를 통해 방법의 효과와 범용성을 보였다.

실제 적용 측면에서 이 연구는 high-resolution dense prediction에서 global context가 필요하지만 자원 제약이 큰 상황에 특히 중요하다. 이후 vision 분야에서 더 다양한 sparse attention, axial attention, factorized attention 계열 연구로 이어질 수 있는 방향성을 보여준다는 점에서도 의미가 있다. 다만 논문이 제시한 장점은 주로 당시 non-local 대비 상대적 우위에 기반하므로, 후속 연구에서는 더 다양한 attention 구조와의 비교, 이론적 분석, 더 큰 비디오 실험으로 확장할 필요가 있다.
