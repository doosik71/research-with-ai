# Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth

* **저자**: Davy Neven, Bert De Brabandere, Marc Proesmans, Luc Van Gool
* **발표연도**: 2019
* **arXiv**: <https://arxiv.org/abs/1906.11109>

## 1. 논문 개요

이 논문은 instance segmentation에서 정확도와 속도를 동시에 확보하려는 문제를 다룬다. 당시 대표적인 proposal-based 방법, 특히 Mask R-CNN 계열은 정확도는 높지만 연산이 느리고 마스크 해상도도 낮다는 한계가 있었다. 반대로 proposal-free 방법은 고해상도 마스크를 만들고 상대적으로 빠를 수 있지만, 정확도에서 proposal-based 방법을 따라가지 못하는 경우가 많았다. 이 논문은 이 둘의 장점을 결합하려는 시도로, 각 픽셀이 자신이 속한 객체의 중심 쪽으로 이동하도록 하는 spatial embedding을 학습하고, 동시에 객체별 clustering bandwidth를 학습하여 instance mask의 IoU를 직접 높이는 방향으로 설계되었다. 저자들의 핵심 주장은, 적절한 loss 설계와 빠른 dense prediction 네트워크를 결합하면 proposal-free 방식으로도 real-time 성능과 높은 정확도를 동시에 달성할 수 있다는 점이다.

연구 문제가 중요한 이유는 자율주행, 로보틱스, 고해상도 장면 이해처럼 픽셀 단위의 정확한 객체 분할이 필요하면서도 동시에 지연이 매우 낮아야 하는 응용이 많기 때문이다. 단순한 bounding box는 객체 경계를 충분히 표현하지 못하고, 반대로 정교한 mask를 생성하는 기존 고성능 방법들은 속도 문제가 있었다. 따라서 “빠르면서도 정확한 instance segmentation”은 단순한 engineering 개선이 아니라 실제 적용 가능성을 결정하는 핵심 과제였다. 이 논문은 바로 이 지점을 겨냥해, 후처리에 의존하던 clustering 문제를 학습 목표 안으로 끌어들였다는 점에서 의미가 있다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 모든 픽셀이 객체 중심 하나의 점에 정확히 꽂히도록 강하게 회귀시키는 것이 최선은 아니라는 데 있다. 객체가 커질수록 가장자리 픽셀은 중심에서 멀기 때문에, 모든 픽셀을 중심점 하나로 몰아붙이는 regression loss는 불필요하게 어려운 학습 문제를 만든다. 저자들은 대신 “객체 중심 주변의 적절한 영역” 안으로 픽셀이 들어오도록 학습시키는 것이 더 합리적이라고 본다. 그리고 그 영역의 크기를 고정하지 않고 객체마다 다르게 학습하게 만든다. 즉, 작은 객체에는 작은 margin, 큰 객체에는 큰 margin을 허용하여, 픽셀 임베딩과 clustering bandwidth를 함께 최적화한다.

기존 proposal-free embedding 계열 방법과의 차별점은, 단순히 같은 인스턴스의 feature를 가깝게 하고 다른 인스턴스의 feature를 멀게 만드는 일반적인 embedding loss를 쓰는 것이 아니라, clustering 과정 자체를 loss 안에 직접 반영한다는 점이다. 특히 Gaussian mapping과 Lovasz-hinge loss를 결합해 각 인스턴스의 foreground/background 확률 지도를 만들고, 이를 통해 instance mask의 intersection-over-union을 직접 최적화한다. 다시 말해 이 방법은 “좋은 embedding을 만들자”가 최종 목표가 아니라, “좋은 instance mask가 나오도록 embedding과 margin을 같이 학습하자”는 더 직접적인 목표를 가진다.

또 하나의 중요한 아이디어는 seed map이다. 추론 시에는 객체 중심을 알아야 clustering을 할 수 있는데, 이 논문은 각 픽셀이 중심에 얼마나 가까운 임베딩인지 나타내는 seediness score를 별도로 예측한다. seed score가 가장 높은 픽셀을 중심 후보로 사용하고, 그 위치의 sigma를 이용해 해당 인스턴스를 순차적으로 복원한다. 이 설계 덕분에 복잡한 density-based clustering 없이도 빠른 sequential clustering이 가능해진다.

## 3. 상세 방법 설명

전체 파이프라인은 크게 세 부분으로 이해할 수 있다. 첫째, 네트워크는 각 픽셀에 대해 offset vector $o_i$를 예측한다. 픽셀 좌표 $x_i$에 이 offset을 더하면 spatial embedding $e_i = x_i + o_i$가 된다. 이 embedding은 해당 픽셀이 속한 객체의 중심 또는 중심 근처 영역을 향하도록 학습된다. 둘째, 각 픽셀은 sigma도 함께 예측한다. 이 sigma는 객체별 clustering margin의 크기를 조절하는 역할을 하며, 결국 같은 중심을 향한 픽셀들을 어느 범위까지 한 객체로 묶을지 결정한다. 셋째, 각 semantic class마다 seed map을 예측해, 어느 픽셀의 embedding이 객체 중심에 가장 가까운지 알려준다. 추론 시에는 이 seed map을 기반으로 객체 중심 후보를 고르고, sigma를 사용해 해당 객체의 픽셀들을 모은다.

기존의 단순 regression 접근은 각 픽셀이 자신의 instance centroid $C_k$를 직접 가리키도록 한다. 이때 인스턴스 $S_k$의 centroid는 다음과 같이 정의된다.

$$
C_k = \frac{1}{|S_k|}\sum_{x \in S_k} x
$$

그리고 픽셀 $x_i$에 대한 정답 offset은 $\hat{o}_i = C_k - x_i$이며, 보통 regression loss는 다음과 같이 쓸 수 있다.

$$
\mathcal{L}_{reg} = \sum_i |o_i - \hat{o}_i|
$$

하지만 이 방식은 학습 후에 다시 centroid를 찾고, 각 embedding을 어느 centroid에 배정할지 별도의 clustering 규칙이 필요하다. 즉, 훈련 목표와 실제 추론 절차가 완전히 일치하지 않는다. 논문은 바로 이 지점을 문제로 본다.

이를 개선하기 위해 저자들은 고정 margin을 갖는 hinge 형태의 생각에서 출발한다. 픽셀 embedding이 중심으로부터 margin $\delta$ 안에 들어오도록 강제하면, 추론 시에도 같은 규칙으로 픽셀을 할당할 수 있다. 그러나 고정 margin은 작은 객체와 큰 객체가 섞여 있는 데이터셋에서 치명적인 제약이 된다. 작은 객체를 분리할 수 있을 정도로 margin을 작게 잡으면, 큰 객체의 가장자리 픽셀은 그 작은 영역 안으로 들어오기 어렵다. 반대로 margin을 크게 잡으면 서로 붙어 있는 작은 객체를 분리하기 힘들다. 이 논문은 그래서 고정 margin 대신 학습 가능한 instance-specific margin을 도입한다.

구체적으로, 인스턴스 $k$에 대해 Gaussian function $\phi_k$를 정의하고, embedding $e_i$가 그 인스턴스에 속할 확률을 다음처럼 계산한다.

$$
\phi_k(e_i) = \exp\left(-\frac{|e_i - C_k|^2}{2\sigma_k^2}\right)
$$

여기서 $\sigma_k$가 클수록 중심 주변의 허용 영역이 넓어진다. 논문은 $\phi_k(e_i) > 0.5$이면 픽셀 $x_i$를 인스턴스 $k$에 속한다고 본다. 따라서 실질적인 margin은

$$
margin = \sqrt{-2\sigma_k^2 \ln 0.5}
$$

에 비례한다고 이해할 수 있다. 원문 식 (6)은 임계값 0.5에 대응되는 거리 제곱 형태를 통해 margin과 sigma의 관계를 설명한다. 중요한 점은 sigma가 단순 보조 변수로 끝나는 것이 아니라, 실제 clustering 규칙을 직접 결정한다는 점이다. 객체별 sigma는 그 객체에 속한 픽셀들의 sigma 평균으로 계산된다.

$$
\sigma_k = \frac{1}{|S_k|}\sum_{\sigma_i \in S_k}\sigma_i
$$

이 설계의 의미는 분명하다. 큰 객체는 큰 sigma를 가져 넓은 영역으로 완화된 supervision을 받고, 작은 객체는 작은 sigma를 가져 더 정밀하게 분리된다.

손실 함수 측면에서 이 논문은 cross-entropy 대신 Lovasz-hinge loss를 사용한다. 각 인스턴스마다 Gaussian으로부터 foreground/background probability map을 만들고, 그 지도와 정답 binary mask 사이의 손실을 Lovasz-hinge로 계산한다. Lovasz-hinge는 Jaccard loss, 즉 IoU의 convex surrogate이므로, 이 선택은 결국 “embedding과 sigma를 통해 인스턴스 마스크의 IoU를 직접 높이겠다”는 설계와 일치한다. 또한 foreground와 background의 클래스 불균형을 따로 크게 걱정하지 않아도 되는 장점이 있다. 논문이 특히 강조하는 점은 sigma와 offset에 대한 직접적인 정답 supervision이 없다는 것이다. 이 둘은 오직 최종 instance mask IoU를 높이는 방향으로 jointly optimized된다.

논문은 두 가지 확장도 제안한다. 첫째는 scalar sigma 대신 2차원 sigma $(\sigma_x, \sigma_y)$를 사용하는 elliptical margin이다. 이 경우 Gaussian은

$$
\phi_k(e_i) = \exp\left(
-\frac{(e_i^x - C_k^x)^2}{2(\sigma_k^x)^2}
-\frac{(e_i^y - C_k^y)^2}{2(\sigma_k^y)^2}
\right)
$$

형태가 되어, 보행자처럼 세로로 긴 객체나 열차처럼 길쭉한 객체에 더 잘 적응할 수 있다. 둘째는 Center of Attraction을 centroid로 고정하지 않고, 해당 인스턴스의 embedding 평균으로 두는 learnable center이다.

$$
\phi_k(e_i) = \exp\left(
-\frac{
\left| e_i - \frac{1}{|S_k|}\sum_{e_j \in S_k} e_j \right|^2
}{
2\sigma_k^2
}
\right)
$$

이렇게 하면 네트워크가 임베딩 자체를 조절해서 더 유리한 중심 위치를 만들 수 있다. 즉, 중심도 고정된 기하학적 centroid가 아니라 학습 가능한 attractor가 된다.

seed map은 추론 속도와 안정성을 위해 매우 중요하다. 픽셀마다 seediness score를 예측하는데, 이 값은 해당 embedding이 인스턴스 중심에 얼마나 가까운지를 나타낸다. foreground 픽셀은 자신의 Gaussian 출력값에 가깝게, background는 0에 가깝게 회귀시키며, 손실은 다음과 같이 표현된다.

$$
\mathcal{L}_{seed} = \frac{1}{N}\sum_i \mathbf{1}*{{s_i \in S_k}} |s_i - \phi_k(e_i)|^2 + \mathbf{1}*{{s_i \in bg}} |s_i - 0|^2
$$

추론 시에는 각 클래스별 seed map에서 가장 높은 값을 가진 픽셀을 골라 그 위치의 embedding을 중심 $\hat{C}_k$, 그 위치의 sigma를 $\hat{\sigma}_k$로 사용한다. 그리고

$$
e_i \in S_k \iff
\exp\left(-\frac{|e_i - \hat{C}_k|^2}{2\hat{\sigma}_k^2}\right) > 0.5
$$

를 만족하는 픽셀을 현재 인스턴스로 묶는다. 그 뒤 이 픽셀들을 seed map에서 제거하고, 남은 seed 중 최고값을 다시 뽑는 순차적 clustering을 반복한다. 추가로 같은 인스턴스 내부의 sigma가 너무 들쑥날쑥하지 않도록 smoothness loss

$$
\mathcal{L}_{smooth} = \frac{1}{|S_k|}\sum_{\sigma_i \in S_k}|\sigma_i - \sigma_k|^2
$$

를 더한다. 이 항은 추론 시 특정 한 위치에서 읽어온 $\hat{\sigma}_k$가 전체 인스턴스 평균 sigma와 비슷하도록 도와준다.

구현 면에서는 ERFNet을 backbone으로 사용하고 encoder는 공유, decoder는 두 갈래로 분리한다. 한 갈래는 offset과 sigma를 예측하고, 다른 갈래는 semantic class별 seed map을 출력한다. offset은 tanh로 $[-1,1]$ 범위에 제한되고, sigma는 exponential activation으로 양수가 되게 만든다. Cityscapes의 2048×1024 해상도에서는 좌표 맵을 $x \in [0,2]$, $y \in [0,1]$ 범위로 정규화하여 인접 픽셀 간 좌표 차이가 가로와 세로 모두 $1/1024$가 되도록 설정한다. 학습은 먼저 객체 중심 500×500 crop으로 200 epoch pretraining을 하고, 이후 1024×1024 crop으로 50 epoch finetuning한다. optimizer는 Adam이고, polynomial decay learning rate를 사용한다. 이런 설계는 정확도뿐 아니라 real-time 성능까지 염두에 둔 매우 실용적인 구성이었다.

## 4. 실험 및 결과

실험은 Cityscapes dataset에서 수행되었다. 이 데이터셋은 2048×1024 해상도의 도시 장면 이미지 5,000장에 fine annotation, 20,000장에 coarse annotation을 제공한다. instance segmentation 평가는 8개 클래스(person, rider, car, truck, bus, train, motorcycle, bicycle)에 대해 region-level AP를 기준으로 이루어진다. 논문 본문에서는 주로 fine train set만 사용해 학습한 결과를 제시하며, 특정 클래스, 특히 truck, bus, train은 샘플 수가 매우 적어서 성능이 불리할 수 있다고 명시한다. 따라서 단순한 AP 숫자 비교뿐 아니라 어떤 추가 데이터(fine only, fine+coarse, fine+COCO)를 썼는지 함께 보는 것이 중요하다.

ablation 실험은 제안한 loss의 각 요소가 실제로 의미가 있는지를 꽤 설득력 있게 보여준다. 먼저 fixed sigma와 learnable sigma를 비교하면, validation set의 ground-truth sampling 기준에서 AP가 28.0에서 38.7로 크게 상승한다. 이는 객체별로 다른 margin을 학습하는 것이 사실상 핵심 성분임을 뜻한다. 또한 Center of Attraction을 단순 centroid 대신 learnable center로 바꾸면 scalar sigma와 2D sigma 모두에서 성능이 올라간다. 마지막으로 circular margin보다 elliptical margin이 더 좋은데, 이는 객체 형상이 원형이 아니라 길쭉한 경우가 많다는 직관과 잘 맞는다. 가장 좋은 조합은 $\sigma_{xy}$와 learnable center를 함께 사용하는 설정으로, 표 2에서 $AP^{[val]}_{gt}=40.5$를 기록한다. 이 결과는 논문의 설계 선택이 단순 아이디어 수준이 아니라 실제 성능으로 연결된다는 점을 보여준다.

또 하나 흥미로운 분석은 sigma와 객체 크기 사이의 상관관계다. 저자들은 figure 4에서 학습된 margin이 객체 크기와 양의 상관관계를 가진다고 보고한다. 이는 제안한 메커니즘이 기대한 방식대로 작동하고 있음을 보여주는 정성적 근거다. 즉, 네트워크는 사람이 직접 규칙을 넣지 않아도, 큰 객체에는 더 큰 clustering region이 유리하다는 사실을 데이터로부터 학습했다. 이는 loss가 단순히 heuristic처럼 보이지 않고, 실제로 의미 있는 구조를 학습한다는 점을 뒷받침한다.

Cityscapes test set의 본 결과에서 제안 방법은 fine-only 학습 조건에서 AP 27.6, $AP_{50}$ 50.9를 기록한다. 표 1 기준으로 이는 Mask R-CNN의 26.2 AP를 넘고, PANet 31.8 AP보다는 낮지만, PANet은 여전히 훨씬 느린 계열이다. 클래스별로 보면 person 34.5, rider 26.1, car 52.4로 Mask R-CNN(fine)의 30.5, 23.7, 46.9보다 높다. 반면 truck, bus, train 같은 희소 클래스에서는 coarse나 COCO를 추가로 사용한 방법들보다 불리하다. 저자들은 이를 데이터 불균형의 영향으로 해석한다. 따라서 이 논문의 결과는 “전체 AP 하나만 압도적으로 최고”라기보다는, real-time 조건과 fine-only 설정을 감안할 때 매우 강력한 accuracy-speed tradeoff를 제시했다는 쪽에 더 가깝다.

속도 측면에서는 이 논문의 장점이 특히 두드러진다. 표 3에서 제안 방법은 2048×1024 해상도 기준 약 11 FPS를 기록한다. 비교 대상으로 제시된 Mask R-CNN은 2.2 FPS, SGN은 0.6 FPS, PANet은 1 FPS 미만이다. Box2Pix가 10.9 FPS로 비슷한 속도를 보이지만, AP는 13.1로 크게 낮다. 논문은 자신들의 방법이 “높은 정확도와 real-time 성능을 동시에 달성한 최초의 proposal-free 방법”이라고 주장한다. 세부적으로는 2MP 입력에서 forward pass가 65ms, clustering이 26ms라고 보고한다. 즉, clustering을 포함한 전체 시스템 관점에서도 충분히 빠르다는 뜻이다. 이 점은 loss function 제안이 단지 성능 개선용이 아니라, 시스템 전체 효율과도 연결되어 있음을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 학습 목표와 추론 절차의 불일치를 줄였다는 점이다. 기존의 regression 기반 중심 예측 방식은 결국 후처리 clustering이 따로 필요했지만, 이 논문은 Gaussian probability와 Lovasz-hinge를 통해 clustering 조건 자체를 loss 안으로 끌어들였다. 그 결과, 픽셀 임베딩을 잘 만드는 것과 좋은 instance mask를 얻는 것이 더 직접적으로 연결된다. 또한 instance-specific margin을 학습하게 한 설계는 데이터셋의 다양한 객체 크기에 자연스럽게 대응하며, 실제 ablation에서 큰 성능 향상으로 이어졌다. 더불어 ERFNet 기반의 가벼운 네트워크와 간단한 sequential clustering을 결합해, 연구 아이디어가 실제 real-time 시스템으로 구현 가능함을 보여준 점도 강점이다.

또 다른 강점은 해석 가능성이다. offset은 픽셀이 어디를 향하는지 보여주고, sigma는 객체별 허용 반경을 의미하며, seed map은 중심 후보를 나타낸다. 즉, 모델 내부 표현이 비교적 직관적이다. figure 4처럼 객체 크기와 sigma의 상관관계를 분석할 수 있다는 점도 이런 구조적 해석 가능성을 뒷받침한다. 많은 딥러닝 방법이 “잘 되지만 왜 되는지”를 설명하기 어려운 반면, 이 방법은 각 구성요소의 역할이 상대적으로 명확하다.

한계도 분명하다. 첫째, 논문은 semantic class 예측과 instance grouping을 seed map 기반으로 결합하지만, 희소 클래스에 대한 성능은 여전히 훈련 데이터 불균형의 영향을 크게 받는다. 즉, proposal-free clustering loss만으로 모든 클래스에서 강건한 성능을 보장하지는 못한다. 둘째, sequential clustering은 간단하지만, 서로 매우 밀집되어 있거나 중심이 애매한 경우 seed 선택이 잘못되면 연쇄적으로 오류가 누적될 가능성이 있다. 논문은 seed map 설계를 통해 이를 완화하지만, 그러한 실패 사례를 체계적으로 분석하지는 않는다. 셋째, 객체 중심 또는 center of attraction을 기준으로 한 표현은 복잡하게 휘어진 비볼록(non-convex) 형상이나 매우 긴 구조에서 완벽히 적합하지 않을 수 있다. elliptical margin이 일부 보완하지만, 여전히 표현의 기본 가정은 “하나의 중심 주위로 모일 수 있다”는 쪽에 가깝다. 이 점은 논문 텍스트에 직접 광범위하게 논의되지는 않지만, 제안된 수식 구조로부터 자연스럽게 읽히는 제약이다.

비판적으로 보면, 이 논문은 매우 좋은 문제 설정과 elegant한 loss 설계를 제시했지만, benchmark 절대 성능 면에서는 PANet 같은 최고 정확도 모델을 완전히 넘지는 못했다. 따라서 “최고 accuracy”보다는 “실시간에 가까운 환경에서 매우 경쟁력 있는 정확도”가 더 정확한 평가다. 그럼에도 불구하고 proposal-free 계열이 실제 시스템에서 통할 수 있다는 점을 강하게 입증했다는 점에서 학술적, 실용적 가치가 크다. 특히 이후 tracking이나 video instance segmentation으로 확장되는 연구 흐름의 토대가 되었다고 볼 수 있다. 다만 이 마지막 확장 가능성 자체는 이 논문 본문에서 직접 실험한 내용은 아니므로, 여기서는 잠재적 의미 수준으로만 언급하는 것이 적절하다.

## 6. 결론

이 논문은 proposal-free instance segmentation에서 spatial embedding, learnable clustering bandwidth, seed map을 결합한 새로운 clustering loss framework를 제안했다. 핵심은 각 픽셀이 객체 중심 “한 점”에 정확히 맞추도록 강제하는 대신, 객체별로 최적인 중심 주변 영역으로 모이게 만들고, 그 영역의 크기까지 네트워크가 스스로 학습하게 한 점이다. 여기에 Lovasz-hinge를 사용해 instance mask의 IoU를 직접 최적화함으로써, 후처리와 학습의 간극을 줄였다. 실험적으로는 Cityscapes에서 fine-only 조건으로 27.6 AP와 11 FPS를 달성해, proposal-free 방법이 real-time 성능과 높은 정확도를 동시에 달성할 수 있음을 보여주었다.

실제 적용 관점에서 이 연구의 의미는 분명하다. 자율주행처럼 고해상도 입력과 낮은 지연이 동시에 필요한 환경에서는, 마스크 해상도와 처리 속도 모두가 중요하다. 이 논문은 그런 조건에서 쓸 수 있는 instance segmentation 설계를 제시했고, 이후 bottom-up instance grouping 계열 방법에 중요한 영향을 준 아이디어로 볼 수 있다. 향후 연구에서는 더 복잡한 형상 표현, 더 안정적인 중심 선택, class imbalance에 강한 학습, 그리고 video setting에서의 시간적 일관성까지 확장될 여지가 크다. 다만 이러한 향후 방향은 본문에서 직접 실험된 결과가 아니라, 본 논문의 구조로부터 자연스럽게 이어지는 가능성으로 이해하는 것이 적절하다.
