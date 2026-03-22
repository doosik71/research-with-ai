# Instance Segmentation and Tracking with Cosine Embeddings and Recurrent Hourglass Networks

* **저자**: Christian Payer, Darko Štern, Thomas Neff, Horst Bischof, Martin Urschler
* **발표연도**: 2018
* **arXiv**: [https://arxiv.org/abs/1806.02070](https://arxiv.org/abs/1806.02070)

## 1. 논문 개요

이 논문은 **instance segmentation**과 **tracking**을 하나의 일관된 프레임워크 안에서 다루는 방법을 제안한다. 일반적인 semantic segmentation이 픽셀마다 클래스만 예측하는 것과 달리, instance segmentation은 같은 클래스 안에서도 각각의 개체를 서로 다른 ID로 구분해야 한다. 예를 들어 세포 영상에서는 모든 세포를 단순히 “cell”로 분류하는 것만으로는 부족하고, 각 세포가 어느 픽셀 집합인지, 또 시간이 지나도 같은 세포가 어떤 ID를 유지하는지를 알아야 한다.

논문이 풀고자 하는 핵심 문제는 두 가지가 결합된 형태다. 첫째, 한 프레임 안에서 서로 다른 개체를 분리하는 **instance segmentation** 문제다. 둘째, 비디오나 시계열 영상에서 그 개체들이 시간에 따라 어떻게 이어지는지를 추적하는 **instance tracking** 문제다. 특히 biomedical imaging에서는 세포가 빽빽하게 존재하고, 형태가 비슷하며, 분열(mitosis)까지 일어나기 때문에 단순한 프레임별 segmentation만으로는 충분하지 않다.

저자들은 이 문제를 해결하기 위해 픽셀마다 instance를 나타내는 **embedding vector**를 예측하고, 이 embedding이 시간에 따라 일관되게 유지되도록 네트워크에 **temporal information**을 넣는다. 이를 위해 stacked hourglass network 내부에 **ConvGRU**를 삽입한 recurrent fully convolutional architecture를 제안한다. 또한 embedding 학습을 위해 기존의 Euclidean distance 중심 손실 대신 **cosine similarity** 기반의 새로운 embedding loss를 설계한다. 최종적으로는 프레임 간 embedding을 clustering하여 instance segmentation과 tracking을 동시에 얻는다.

이 문제가 중요한 이유는, 의료 영상과 생명과학 영상에서는 단순 분할을 넘어 개별 객체의 시간적 행동을 측정해야 하는 경우가 많기 때문이다. 세포 이동, 세포 분열, 심장 구조의 시간적 변화처럼 시간 축을 따라 개체 정체성을 유지하는 것이 분석의 핵심인 경우가 많다. 따라서 이 논문은 segmentation과 tracking을 분리된 후처리 단계가 아니라, embedding과 recurrent modeling을 결합한 하나의 구조로 다루려 했다는 점에서 의미가 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 각 픽셀을 클래스 확률이 아니라 **instance-aware embedding**으로 표현하고, 이 embedding을 시간 축을 따라 안정적으로 전파하도록 **recurrent stacked hourglass network**를 학습시키는 것이다. 즉, 같은 instance에 속한 픽셀들은 비슷한 embedding 방향을 갖게 하고, 서로 인접한 다른 instance의 픽셀들은 서로 다른 embedding 방향을 갖게 만든다. 이렇게 학습된 embedding 공간에서 clustering을 수행하면 개별 instance를 얻을 수 있다.

기존 접근과 비교했을 때 차별점은 크게 두 가지다.

첫째, 기존의 많은 instance segmentation 방법은 한 번에 한 instance씩 분할하거나, detection 뒤에 instance mask를 개별적으로 생성하는 구조를 취했다. 하지만 세포처럼 한 장면에 수십 개에서 수백 개의 객체가 동시에 존재하는 경우, 이런 방식은 비효율적일 수 있다. 반면 이 논문은 **모든 픽셀의 embedding을 한 번에 예측**하고, 그 후 clustering으로 각 instance를 분리한다.

둘째, 기존 embedding 기반 instance segmentation은 주로 이미지 한 장 내의 grouping에 초점을 두었지만, 이 논문은 여기에 **시간 정보**를 결합한다. 즉, 이전 프레임의 정보를 ConvGRU로 내부 상태에 담아 다음 프레임의 embedding을 예측함으로써, 단순히 프레임별 instance 분할이 아니라 **시간적으로 연결된 instance identity**를 얻고자 한다. 저자들은 특히 cell tracking처럼 시간 연속성이 중요한 문제에서 이것이 강한 장점이라고 본다.

또 하나의 중요한 아이디어는 손실 함수 설계다. 저자들은 **four color map theorem**에서 영감을 받아, 모든 다른 instance 쌍을 멀리 떨어뜨릴 필요는 없고 **서로 이웃한 instance들만 다르게 만들면 충분하다**고 본다. 이 완화는 매우 많은 객체가 존재하는 장면에서 embedding 학습을 더 단순하게 해 준다. 따라서 이 논문의 핵심은 “모든 instance를 서로 전역적으로 분리하는 것”이 아니라, “국소적으로 충돌하지 않는 embedding을 학습하고, 시간축으로 그 embedding을 안정화하는 것”이라고 요약할 수 있다.

## 3. 상세 방법 설명

전체 방법은 크게 세 단계로 이해할 수 있다. 먼저 네트워크가 입력 영상 또는 비디오 프레임 시퀀스로부터 픽셀별 embedding을 예측한다. 다음으로 학습 시 cosine embedding loss를 이용해 같은 instance 내부는 embedding이 유사해지고, 인접한 다른 instance와는 구별되도록 만든다. 마지막으로 추론 시 HDBSCAN clustering을 통해 embedding들을 instance 단위로 묶고, 연속 프레임 사이에서 이를 이어 붙여 tracking 결과를 만든다.

### 3.1 Recurrent Stacked Hourglass Network

저자들은 원래 human pose estimation에 쓰였던 **stacked hourglass network**를 수정하여 사용한다. hourglass 구조는 contracting path와 expanding path를 통해 넓은 문맥 정보와 세밀한 위치 정보를 모두 활용하는 구조다. 이 논문에서는 여기에 **ConvGRU**를 삽입하여 시간 정보를 처리한다.

구체적으로는 contracting path와 expanding path 사이에 3×3 필터, 64채널의 ConvGRU를 넣는다. 일반 convolution block 역시 3×3 convolution과 64 output을 사용한다. 또한 hourglass를 두 개 연속으로 쌓는다. 첫 번째 hourglass의 출력은 원래 입력 이미지와 concatenate되어 두 번째 hourglass의 입력으로 들어간다. 손실 함수는 두 hourglass의 출력 모두에 적용되지만, 최종 clustering에는 두 번째 hourglass의 출력만 사용한다.

이 설계의 의도는 다음과 같다. hourglass는 멀티스케일 특징을 잘 잡을 수 있고, ConvGRU는 시간에 따라 변하는 객체의 위치와 형태 정보를 hidden state에 축적할 수 있다. 따라서 단일 프레임만 보고 segmentation하는 것이 아니라, 이전 프레임들의 맥락을 반영한 embedding을 예측할 수 있게 된다. 저자들은 이 구조가 특히 비디오에서 instance identity를 유지하는 데 유리하다고 주장한다.

### 3.2 픽셀 임베딩 정의

각 픽셀 $p$에 대해 네트워크는 $d$차원 embedding vector $\vec{e}_p \in \mathbb{R}^d$를 예측한다. 이 embedding은 instance identity를 표현하는 잠재 표현이다. 이상적으로는 같은 instance에 속한 픽셀들의 embedding은 비슷해야 하고, 다른 instance에 속한 픽셀들의 embedding은 달라야 한다.

논문에서는 각 instance $i \in \mathbb{I}$에 대해 해당 instance 내부 픽셀 집합을 $\mathbb{S}^{(i)}$로 둔다. 배경도 독립적인 하나의 instance처럼 취급한다. 또한 모든 다른 instance를 다 고려하지 않고, 반경 $r_{\mathbb{N}}$ 이내에 있는 **neighboring instance들**만 고려해 $\mathbb{N}^{(i)}$를 정의한다. 이 점이 손실 설계의 핵심이다.

### 3.3 Cosine similarity와 손실 함수

두 embedding $\vec{e}_1, \vec{e}_2$의 유사도는 cosine similarity로 정의된다.

$$
\text{cos}(\vec{e}_1,\vec{e}_2)=\frac{\vec{e}_1\cdot\vec{e}_2}{|\vec{e}_1||\vec{e}_2|}
$$

이 값은 $-1$에서 $1$ 사이를 가지며, $1$이면 방향이 같고, $0$이면 직교하며, $-1$이면 정반대 방향이다. 저자들은 embedding의 절대 크기보다 **방향(direction)** 이 instance 구분에 더 중요하도록 만들고자 cosine similarity를 사용한다.

각 instance $i$의 평균 embedding은 다음과 같다.

$$
\vec{\bar{e}}^{(i)}=\frac{1}{|\mathbb{S}^{(i)}|}\sum_{p\in\mathbb{S}^{(i)}} \vec{e}_p
$$

이때 전체 손실은 다음과 같이 주어진다.

$$
L=\frac{1}{|\mathbb{I}|}\sum_{i\in\mathbb{I}}
\left(
1-\frac{1}{|\mathbb{S}^{(i)}|}\sum_{p\in\mathbb{S}^{(i)}} \text{cos}(\vec{\bar{e}}^{(i)},\vec{e}*p)
\right)
+
\left(
\frac{1}{|\mathbb{N}^{(i)}|}\sum*{p\in\mathbb{N}^{(i)}} \text{cos}(\vec{\bar{e}}^{(i)},\vec{e}_p)^2
\right)
$$

이 수식은 직관적으로 두 부분으로 나뉜다.

첫 번째 항은 같은 instance 내부 픽셀들이 그 instance의 평균 embedding과 같은 방향을 가지도록 만든다. 즉, $p \in \mathbb{S}^{(i)}$에 대해 $\text{cos}(\vec{\bar{e}}^{(i)},\vec{e}_p)\approx 1$이 되게 한다. 이것은 같은 instance 내부의 응집도(cohesion)를 높이는 역할이다.

두 번째 항은 이웃 instance의 픽셀들이 평균 embedding과 **직교(orthogonal)** 하도록 만든다. 즉, $p \in \mathbb{N}^{(i)}$에 대해 $\text{cos}(\vec{\bar{e}}^{(i)},\vec{e}_p)\approx 0$이 되게 유도한다. 제곱을 취했기 때문에 양의 상관이든 음의 상관이든 0에서 멀어지면 불이익을 받는다. 결과적으로 neighboring instance는 같은 방향도, 반대 방향도 아닌 거의 직교 방향으로 분리되는 셈이다.

이 손실 함수의 장점은 두 가지로 해석할 수 있다. 첫째, cosine similarity는 embedding norm을 사실상 정규화한 형태라 recurrent network와 결합했을 때 값의 폭주를 줄이는 데 유리하다. 논문에서도 저자들은 Euclidean distance 기반 embedding loss를 recurrent network와 결합할 때 문제가 있었다고 언급한다. 둘째, 모든 instance 쌍을 분리하지 않고 인접한 것만 분리하므로, 장면에 많은 객체가 있어도 최적화 부담이 줄어든다.

### 3.4 Clustering과 tracking 생성

네트워크가 예측한 embedding만으로는 최종 instance mask가 바로 나오지 않는다. embedding을 어떤 픽셀 그룹이 하나의 instance인지로 바꾸는 과정이 필요하다. 이를 위해 저자들은 **HDBSCAN**을 사용한다. HDBSCAN은 군집 수를 자동으로 추정할 수 있어서, 이미지마다 instance 개수가 달라도 적용 가능하다.

클러스터링에는 embedding 값만 쓰지 않고 **image coordinates**도 함께 사용한다. 이유는 이 논문의 손실이 멀리 떨어진 instance들이 같은 embedding을 가져도 허용하기 때문이다. 따라서 공간적으로 멀리 떨어진 두 객체가 embedding만 비슷하다고 하나의 군집으로 합쳐지지 않도록, 좌표를 스케일된 형태로 embedding에 붙여 clustering 입력으로 넣는다.

저자들은 모든 시퀀스를 한꺼번에 클러스터링하지 않고, **겹치는 연속 프레임 쌍**만 묶어서 clustering한다. 이렇게 하면 계산을 단순화하면서도 분열 이벤트를 잡을 수 있다. 각 프레임 쌍에서 얻은 instance 결과는, 겹치는 프레임에서 instance 간 IoU가 가장 큰 대응을 찾아 이어 붙인다. 마지막으로 원래 해상도로 upsampling하여 최종 segmented and tracked instances를 생성한다.

### 3.5 학습 절차와 구현 세부 사항

실험 섹션과 appendix에 따르면, 네트워크는 TensorFlow로 학습되며 데이터 증강은 SimpleITK를 이용해 on-the-fly로 수행된다. 기본 입력 크기는 $256 \times 256$이며, hourglass는 7 levels를 사용한다. recurrent network는 길이 10의 frame sequence로 학습된다.

학습 하이퍼파라미터는 다음과 같다. convolution weight는 He initialization을 사용하고, bias는 0으로 초기화한다. normalization layer나 dropout은 사용하지 않으며, $L_2$ weight regularization factor는 $0.00001$이다. recurrent network는 메모리 제약 때문에 mini-batch size 1, non-recurrent는 batch size 10을 사용한다. optimizer는 ADAM이며 총 40,000 iteration 동안 학습하고, learning rate는 처음 $0.0001$, 20,000 iteration 이후 $0.00001$로 낮춘다. 저자에 따르면 recurrent network 학습은 단일 NVIDIA Titan Xp 12GB에서 약 12시간, non-recurrent는 약 8시간이 걸렸다.

논문은 데이터셋별 intensity augmentation과 spatial deformation, elastic deformation, neighbor radius $r_{\mathbb{N}}$, clustering parameter $m_{\text{pts}}$, coordinate scaling factor $c$ 등의 세부 값도 appendix에서 제공한다. 이는 실제 재현성 측면에서 유용하지만, 본문만으로는 왜 특정 데이터셋에 특정 값이 선택되었는지에 대한 원리는 자세히 설명되지 않는다.

## 4. 실험 및 결과

이 논문은 하나의 방법이 세 가지 서로 다른 상황에서 작동함을 보이기 위해 세 종류의 실험을 수행한다. 첫 번째는 recurrent architecture가 temporal information을 실제로 활용하는지 보기 위한 **left ventricle semantic segmentation**이다. 두 번째는 temporal 정보 없이도 cosine embedding loss가 instance segmentation에 유효한지 보기 위한 **plant leaf instance segmentation**이다. 세 번째이자 핵심 실험은 **cell instance segmentation and tracking**이다.

### 4.1 Left Ventricle Segmentation

이 실험의 목적은 제안한 recurrent stacked hourglass network가 실제로 시간 정보를 활용할 수 있는지 검증하는 것이다. 데이터는 left ventricle segmentation challenge의 short-axis cardiac MR video이며, 평가 대상은 좌심실 myocardium과 blood cavity가 모두 보이는 세 개의 central slice다.

비교 대상은 recurrent와 non-recurrent 두 버전이다. non-recurrent baseline은 ConvGRU를 일반 convolution으로 바꾸어 네트워크 복잡도는 비슷하게 맞춘다. 학습 목표는 background, myocardium, blood cavity의 3-label semantic segmentation이며 손실은 softmax cross entropy loss를 사용한다. 데이터는 96명의 환자를 3-fold cross-validation으로 나눠 평가한다.

결과는 다음과 같다.

* myocardium IoU: non-recurrent $78.3 \pm 9.2$, recurrent $79.4 \pm 8.5$
* blood cavity IoU: non-recurrent $89.1 \pm 7.7$, recurrent $89.4 \pm 7.2$

수치 차이는 크지 않지만 recurrent 모델이 두 클래스 모두에서 더 높다. 이는 ConvGRU를 네트워크 깊은 내부에 넣어 temporal context를 반영하는 것이 semantic segmentation에도 이득이 있음을 보여준다. 다만 저자들도 인정하듯, 이 실험은 challenge의 원래 평가 프로토콜을 단순화했기 때문에 다른 논문들의 결과와 직접 비교하면 안 된다.

### 4.2 Leaf Instance Segmentation

이 실험은 cosine embedding loss 자체가 정적 이미지에서도 instance segmentation에 유효한지 보이기 위한 것이다. 데이터는 CVPPP challenge의 A1 dataset이며, 개별 plant leaf를 분할하는 문제다. 이때는 temporal 정보가 없으므로 recurrent가 아닌 non-recurrent 네트워크를 사용하고, embedding 차원은 32로 설정한다. clustering은 각 이미지별로 수행된다.

평가는 128장의 training image에 대해 3-fold cross-validation으로 이루어졌다. 지표는 **SBD (symmetric best Dice)** 와 **$|!|!|DiC|!|!|$ (absolute difference in count)** 이다.

주요 결과는 다음과 같다.

* Ours: SBD $84.5 \pm 5.5$, $|!|!|DiC|!|!| = 1.5 \pm 1.2$
* IS+RA [11]: SBD $84.9 \pm 4.8$, $|!|!|DiC|!|!| = 0.8 \pm 1.0$
* 그 외 RIS+CRF, MSU, Nottingham, Wageningen, IPK보다 Ours가 SBD에서 크게 앞선다.

즉, 제안 방법은 당시 선도 방법인 IS+RA와 거의 비슷한 성능을 보였고, 나머지 경쟁 방법들보다는 상당히 강했다. 특히 저자들은 자기 방법이 leading method보다 구조가 더 단순하다고 강조한다. 다만 여기서도 challenge 공식 test set 결과가 아니라 내부 cross-validation 결과라는 점은 분명히 봐야 한다.

### 4.3 Cell Instance Tracking

이 논문의 핵심 실험이다. 데이터는 ISBI cell tracking challenge의 여섯 개 데이터셋이다: DIC-HeLa, Fluo-MSC, Fluo-GOWT1, Fluo-HeLa, PhC-U373, Fluo-SIM+. 각 데이터셋은 두 개의 annotated training video와 두 개의 testing video를 갖고 있으며, 해상도는 $512 \times 512$부터 $1200 \times 1024$까지 다양하고 길이는 48~138 frames다.

여기서 중요한 데이터 처리 방식이 있다. tracking ground truth에서는 instance ID가 비디오 전체에서 일관되지만, segmentation ground truth는 그렇지 않기 때문에 저자들은 두 ground truth를 합쳐 각 프레임에서 일관된 instance ID를 만들었다. 또한 background embedding을 학습하기 위해 모든 세포가 segmentation된 프레임만 사용했다. challenge submission용 모델은 각 데이터셋의 두 annotated training video로 학습되며 embedding 차원은 16이다.

tracking metric을 만족하려면 분열 후 새로 생성된 세포의 **parent ID**도 추정해야 한다. 논문은 새 instance가 만들어졌을 때 이전 프레임들과의 IoU가 가장 높은 instance를 parent로 정한다. 이후 family tree가 평가 기준과 맞도록 후처리도 수행한다.

표 2의 결과를 보면, 제안 방법은 여섯 데이터셋 전반에서 매우 강한 tracking 성능을 보인다.

#### Overall Performance (OP)

* DIC-HeLa: Ours 2위, 0.828
* Fluo-MSC: Ours 2위, 0.676
* Fluo-GOWT1: Ours 2위, 0.914
* Fluo-HeLa: Ours 2위, 0.940
* PhC-U373: Ours 2위, 0.896
* Fluo-SIM+: Ours 2위, 0.878

표 해석상 각 데이터셋의 최고 성능은 다른 팀이 차지한 경우가 있지만, 논문 본문 설명에 따르면 overall 기준으로 Fluo-GOWT1에서는 2위, 여러 데이터셋에서 상위권을 기록했다.

#### Segmentation (SEG)

* DIC-HeLa: Ours 2위, 0.776
* Fluo-MSC: Ours 2위, 0.590
* Fluo-GOWT1: Ours 2위, 0.893
* Fluo-HeLa: Ours 2위, 0.893
* PhC-U373: Ours 2위, 0.832
* Fluo-SIM+: Ours 2위, 0.791

#### Tracking (TRA)

* DIC-HeLa: Ours 2위, 0.881
* Fluo-MSC: Ours 2위, 0.765
* Fluo-GOWT1: Ours 2위, 0.947
* Fluo-HeLa: Ours 2위, 0.987
* PhC-U373: Ours 2위, 0.981
* Fluo-SIM+: Ours 2위, 0.961

본문의 해설을 따르면, 이 방법은 tracking metric 기준으로 여섯 데이터셋 중 **두 개에서 1위, 두 개에서 2위**를 달성했다고 한다. 특히 세포가 밀집한 DIC-HeLa에서는 segmentation과 tracking 모두에서 최고 성능을 달성했다. 반면 Fluo-HeLa와 Fluo-SIM+처럼 세포가 매우 작고 다운샘플링의 영향을 크게 받는 데이터에서는 성능이 상대적으로 낮았다. 저자들은 입력 해상도를 $256 \times 256$에 맞추기 위해 이미지 축소가 필요했는데, 이때 세포가 지나치게 작아져 instance segmentation이 어려워졌다고 해석한다.

### 4.4 실험 결과의 의미

이 세 가지 실험은 각각 다른 질문에 답한다.

첫 번째 실험은 “ConvGRU를 넣은 recurrent hourglass가 temporal cue를 활용하는가?”를 검증한다. 두 번째 실험은 “cosine embedding loss가 정적 이미지 instance segmentation에도 효과적인가?”를 보여준다. 세 번째 실험은 “둘을 결합한 전체 시스템이 실제 biomedical tracking benchmark에서 경쟁력이 있는가?”를 입증한다.

즉, 저자들은 단순히 benchmark 결과만 제시한 것이 아니라, 각 구성 요소의 의미를 단계적으로 검증하려 했다. 이 점은 논문의 설득력을 높이는 요소다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **instance embedding**과 **temporal recurrence**를 자연스럽게 결합했다는 점이다. 당시 embedding 기반 instance segmentation은 존재했지만, 이를 시간 정보와 결합해 tracking까지 수행하는 방향은 드물었다. 저자들이 주장하듯, embedding이 프레임마다 안정적으로 유지되도록 recurrent structure를 넣은 것은 분명한 기여다.

또 다른 강점은 손실 함수 설계가 매우 직관적이면서도 실용적이라는 점이다. cosine similarity를 사용함으로써 embedding의 크기보다 방향을 학습하게 만들고, 모든 instance 쌍이 아니라 neighboring instance만 구분하도록 완화했다. 이는 densely packed object 환경에서 특히 유리하다. 실제로 DIC-HeLa처럼 세포가 조밀한 데이터셋에서 좋은 성능을 보인 것은 이 설계가 효과적이었음을 뒷받침한다.

세 번째 강점은 적용 범위의 넓이이다. 심장 MR video의 semantic segmentation, 식물 잎 instance segmentation, 현미경 영상의 cell tracking이라는 서로 다른 세 문제에 같은 기본 틀을 적용해 성능을 입증했다. 이는 방법이 특정 데이터셋에만 과도하게 맞춰진 것이 아니라 비교적 일반적인 구조라는 인상을 준다.

하지만 한계도 분명하다.

가장 뚜렷한 한계는 **end-to-end가 아니라는 점**이다. 최종 instance 생성이 HDBSCAN clustering에 의존하며, 이 clustering parameter는 데이터셋마다 따로 조정해야 한다. appendix를 보면 각 데이터셋마다 $m_{\text{pts}}$와 coordinate scaling factor $c$를 별도로 정했다. 즉, 네트워크 자체가 모든 것을 자동으로 해결하는 구조는 아니며, 후처리와 하이퍼파라미터 튜닝의 비중이 적지 않다.

또한 해상도 문제에 취약하다. 입력을 $256 \times 256$으로 맞추기 위해 다운샘플링하는 과정에서 작은 객체는 정보가 크게 손실될 수 있다. 저자들도 Fluo-HeLa와 Fluo-SIM+에서 이 문제가 성능 저하의 핵심 원인이라고 직접 언급한다. 이는 이 방법이 작은 객체가 많은 고해상도 영상에서는 추가적인 patch-wise 처리 없이 한계가 있음을 보여준다.

세 번째로, recurrent 정보를 네트워크 깊은 내부에 넣는 것이 왜 최적인지는 아직 충분히 검증되지 않았다. 저자들 스스로도 다른 위치에 recurrent layer를 두는 방법과의 비교는 앞으로 더 필요하다고 말한다. 따라서 ConvGRU 삽입 위치에 대한 설계가 절대적으로 최선이라고 보기는 어렵다.

또 하나의 실용적 한계는 학습과 추론 복잡도다. recurrent network는 batch size 1로 학습했고, 메모리와 계산 요구량이 크다고 설명한다. 따라서 더 큰 영상, 더 긴 시퀀스, 더 많은 객체가 있는 환경으로 확장할 때 효율성 문제가 나타날 수 있다.

비판적으로 보면, 논문은 구성 요소의 결합 아이디어는 강하지만, 전체 시스템이 비교적 많은 수작업 설계에 의존한다. neighbor radius $r_{\mathbb{N}}$, clustering parameter, coordinate scaling 등은 모두 문제별로 조정되어야 한다. 따라서 “깔끔하게 end-to-end로 학습되는 보편적 추적기”라기보다는, 잘 설계된 embedding 기반 segmentation-tracking pipeline에 가깝다. 그럼에도 불구하고 당시 biomedical tracking 문제에서 이 정도의 성능을 낸 것은 충분히 의미가 있다.

## 6. 결론

이 논문은 instance segmentation을 위한 embedding 표현과 recurrent neural network를 결합해, **instance segmentation과 tracking을 동시에 다루는 프레임워크**를 제안했다. 핵심 기여는 세 가지로 정리할 수 있다.

첫째, stacked hourglass network 내부에 ConvGRU를 삽입한 **recurrent fully convolutional architecture**를 제안했다. 이를 통해 시간 정보를 네트워크가 직접 활용하도록 만들었다.

둘째, **cosine embedding loss**를 제안했다. 같은 instance 내부에서는 embedding 방향이 유사하도록, neighboring instance와는 직교에 가깝도록 만들며, 전역적인 instance 분리 대신 국소적인 이웃 관계만 고려하도록 완화했다.

셋째, 실제 biomedical 영상 문제에서 이 방법의 유효성을 보였다. 좌심실 segmentation에서는 temporal modeling의 효과를, CVPPP leaf 데이터셋에서는 cosine embedding loss의 유효성을, ISBI cell tracking challenge에서는 segmentation과 tracking 전체 파이프라인의 경쟁력을 입증했다.

이 연구는 특히 biomedical imaging에서 개별 객체의 시간적 정체성을 추적해야 하는 문제에 중요한 의미가 있다. 세포 추적, 분열 분석, 장기 구조의 시간적 변화 분석 등 다양한 분야에 적용 가능성이 있다. 동시에 저자들이 결론에서 말하듯, clustering을 네트워크 내부로 흡수하여 **single end-to-end trainable model**로 발전시키는 방향은 이후 연구에서 매우 중요한 과제가 될 수 있다. 따라서 이 논문은 embedding-based instance representation과 temporal modeling을 연결하는 초창기이자 의미 있는 시도로 볼 수 있다.
