# Active Boundary Loss for Semantic Segmentation

- **저자**: Chi Wang, Yunke Zhang, Miaomiao Cui, Peiran Ren, Yin Yang, Xuansong Xie, Xian-Sheng Hua, Hujun Bao, Weiwei Xu
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2102.02696

## 1. 논문 개요

이 논문은 semantic segmentation에서 물체 경계가 흐려지는 문제를 해결하기 위해 `Active Boundary Loss (ABL)`를 제안한다. 기존의 cross-entropy loss는 각 픽셀의 클래스 분류 정확도는 잘 감독하지만, 예측된 경계(predicted boundaries, PDBs)가 정답 경계(ground-truth boundaries, GTBs)와 실제로 얼마나 잘 맞는지는 직접적으로 강제하지 않는다. 그 결과, 내부 영역은 대체로 맞더라도 경계가 번지거나 얇은 구조물이 무너지는 현상이 자주 발생한다.

저자들은 이 문제를 단순히 “경계 픽셀을 더 잘 맞추자” 수준이 아니라, 현재 네트워크가 만든 예측 경계를 기준으로 그 경계를 GTB 쪽으로 점진적으로 이동시키는 학습 문제로 재정의한다. 즉, 경계 자체를 정적인 supervision 대상으로 보지 않고, 현재 예측 상태에 따라 동적으로 움직여야 할 방향을 계산해 학습에 반영한다.

이 문제가 중요한 이유는 semantic segmentation의 실제 품질이 내부 영역 정확도만으로 결정되지 않기 때문이다. 자율주행, 의료영상, 영상 편집, video object segmentation 같은 응용에서는 경계 정밀도가 매우 중요하다. 특히 traffic light, pole, bicycle 같은 가늘고 작은 구조물은 내부 픽셀 수가 적어서 일반적인 region 중심 학습만으로는 복원이 어렵다.

## 2. 핵심 아이디어

이 논문의 핵심 직관은 간단하다. 현재 segmentation 결과에서 얻은 예측 경계 PDB를 출발점으로 삼고, 각 경계 픽셀이 GTB에 더 가까워지려면 어느 방향으로 이동해야 하는지를 계산한 뒤, 그 방향으로 경계가 “밀리도록” loss를 설계한다. 저자들은 이를 differentiable한 direction vector prediction 문제로 바꾸었다.

이 아이디어가 기존 방법과 다른 점은 두 가지다. 첫째, 후처리(post-processing) 방식이 아니라 end-to-end training loss라는 점이다. DenseCRF나 Segfix는 예측 결과를 사후 보정하지만, ABL은 네트워크가 애초에 더 좋은 경계를 만들도록 학습시킨다. 둘째, boundary-aware branch를 추가하는 multi-task 설계가 아니라, 기존 segmentation 네트워크 구조를 그대로 둔 채 loss만 추가할 수 있는 model-agnostic 방법이라는 점이다.

또 하나의 중요한 차별점은 ABL이 현재 예측 결과에 의존해 매 iteration마다 supervision target이 달라지는 동적 손실이라는 점이다. 저자들은 이를 classical active contour와 유사한 관점으로 해석한다. 즉, 현재 경계를 보고 다음 이동 방향을 정하고, 이를 반복해 GTB 쪽으로 수렴시키는 방식이다.

## 3. 상세 방법 설명

전체 구조는 두 단계로 이루어진다. 먼저 현재 네트워크 출력에서 PDB를 검출하고, 각 PDB 픽셀에 대해 GTB에 더 가까워지는 방향을 정한다. 그 다음, 그 방향이 선택되도록 확률 분포를 학습시키는 loss를 계산한다. 최종 학습 손실은 기존 pixel-wise loss와 region-level loss에 ABL을 더한 형태다.

네트워크 출력은 클래스 수가 $C$이고 해상도가 $H \times W$인 클래스 확률 맵 $P \in \mathbb{R}^{C \times H \times W}$이다. 먼저 각 픽셀 $i$가 경계인지 판단하기 위해, 2-neighborhood 내부 이웃 픽셀 $j$와의 KL divergence를 본다. 식은 다음과 같다.

$$
B_i =
\begin{cases}
1 & \text{if } \exists \ \mathrm{KL}(P_i, P_j) > \epsilon,\ j \in N_2(i) \\
0 & \text{otherwise}
\end{cases}
$$

여기서 $B_i=1$이면 그 픽셀은 predicted boundary로 간주된다. 중요한 점은 threshold $\epsilon$을 고정하지 않고 adaptive하게 정한다는 것이다. 전체 이미지 픽셀 중 boundary 픽셀이 1%를 넘지 않도록 threshold를 잡는다. 저자들은 이렇게 해야 초기 학습 단계에서 GTB와 멀리 떨어진 noisy boundary가 너무 많이 생기는 것을 막을 수 있다고 설명한다.

그 다음 GTB도 유사하게 얻는다. 다만 GT의 경우 확률 분포가 아니라 이웃 픽셀의 ground-truth class label이 같은지 다른지를 기준으로 boundary를 정의한다. GTB가 정해지면 distance transform을 적용하여 각 픽셀이 가장 가까운 GTB까지 얼마나 떨어져 있는지 나타내는 distance map $M$을 만든다.

이제 PDB 위의 픽셀 $i$에 대해, 8-neighborhood 안에서 distance value가 가장 작은 방향을 고른다. 즉, GTB에 가장 가까워지는 이웃 방향을 정답 방향으로 설정한다. 이를 one-hot 8차원 벡터로 나타낸 target direction map $D^g$로 표현한다.

$$
D_i^g = \Phi\left(\arg \min_j M_{i+\Delta_j}\right), \quad j \in \{0,1,\dots,7\}
$$

여기서 $\Delta_j$는 8방향 offset이고, $\Phi$는 해당 방향 인덱스를 one-hot 벡터로 바꾸는 함수다. 직관적으로는 “이 경계 픽셀은 다음 step에서 어느 방향으로 움직이면 GTB에 가까워지는가?”를 묻는 것이다. 구현에서는 PDB를 1픽셀 dilation해서 더 넓은 영역에 대해 이 연산을 수행해 경계 이동을 가속한다.

두 번째 단계에서는 실제 예측 방향 분포를 만든다. 픽셀 $i$와 그 8개 이웃 사이의 KL divergence를 logits처럼 사용하고, softmax를 취해 8차원 확률 분포 $D_i^p$를 만든다.

$$
D_i^p =
\left\{
\frac{e^{\mathrm{KL}(P_i, P_{i+\Delta_k})}}
{\sum_{m=0}^{7} e^{\mathrm{KL}(P_i, P_{i+\Delta_m})}}
,\ k \in \{0,1,\dots,7\}
\right\}
$$

이 분포는 “어느 방향의 이웃과 가장 분포 차이가 큰가”를 나타낸다. 경계 픽셀에서는 경계 바깥과 안쪽의 분포가 다르므로, 적절한 방향으로 KL divergence가 커져야 한다. 저자들의 의도는 GTB 방향에 있는 이웃과의 KL divergence를 키우고, 나머지 방향과의 관계는 상대적으로 줄이게 만드는 것이다. 이렇게 하면 현재 경계가 GTB 쪽으로 밀리는 효과가 난다.

최종 ABL은 방향 예측에 대한 weighted cross-entropy loss로 정의된다.

$$
ABL = \frac{1}{N_b}\sum_i^{N_b} \Lambda(M_i)\, CE(D_i^p, D_i^g)
$$

여기서 $N_b$는 PDB 픽셀 수이고, $\Lambda$는 거리 기반 가중치 함수다.

$$
\Lambda(x)=\frac{\min(x,\theta)}{\theta}
$$

논문에서는 $\theta=20$을 사용한다. GTB에서 멀수록 더 강하게 페널티를 주되, 일정 거리 이상은 clipping한다. 만약 $M_i=0$이면 이미 GTB 위에 있는 픽셀이므로 ABL 계산에서 제외한다.

이 손실의 핵심 의미는 단순히 “경계 픽셀을 맞춰라”가 아니라, “현재 경계 픽셀이 GTB에 더 가까워지는 방향으로 분포 구조를 바꿔라”이다. 그래서 경계 정렬(boundary alignment)을 직접적으로 유도한다.

논문은 여기서 중요한 문제를 하나 지적한다. 서로 인접한 두 PDB 픽셀이 각자 다른 방향으로 움직어야 하는 상황에서는 gradient conflict가 생길 수 있다. 예를 들어 어떤 픽셀 쌍 $(V_1, V_2)$에 대해, $V_1$ 입장에서는 $V_2$와의 KL divergence를 키워야 하지만, $V_2$ 입장에서는 오히려 $V_1$와의 KL divergence를 줄여야 하는 모순이 생길 수 있다. 이 경우 gradient가 서로 상쇄되거나 엇갈려 성능이 크게 떨어진다.

이를 해결하기 위해 저자들은 PyTorch의 detach를 사용한다. 즉, ABL gradient는 PDB 픽셀 자신에게만 흐르고, 이웃 픽셀로는 흐르지 않게 막는다. 이렇게 하면 한 픽셀의 경계 이동을 위한 신호가 다른 픽셀을 직접 끌어당기거나 밀어내며 충돌하는 상황을 줄일 수 있다. 저자들은 detach를 제거하면 mIoU가 약 3% 떨어졌다고 보고한다. 이 부분은 방법의 성패를 좌우하는 핵심 구현 요소다.

또한 one-hot direction target에 label smoothing을 적용한다. 가장 큰 확률을 0.8, 나머지 7개 방향은 각각 $0.2/7$로 둔다. 이유는 여러 방향의 거리가 같을 수 있기 때문이다. 즉, 정확히 하나의 방향만 절대적으로 정답이라고 강하게 밀면 over-confident update가 생길 수 있으므로, 약간의 완화를 주는 것이다.

최종 학습 손실은 다음과 같다.

$$
L_t = CE + IoU + w_a ABL
$$

여기서 $CE$는 일반적인 cross-entropy, $IoU$는 Lovasz-Softmax loss이며, $w_a$는 ABL 가중치다. 논문에서 $IoU$ 항을 함께 쓰는 이유는 두 가지라고 설명한다. 첫째, 작은 객체가 학습 중 무시되지 않게 하여 PDB 자체가 사라지는 것을 막는다. 둘째, 초기 학습 단계의 noisy PDB를 안정화한다. 즉, ABL 단독보다는 region-level regularization과 함께 써야 더 잘 작동한다는 것이다.

## 4. 실험 및 결과

실험은 semantic image segmentation과 video object segmentation 두 영역에서 수행되었다. 이미지 segmentation에서는 DeepLabV3, OCR, UperNet(Swin Transformer backbone)을 사용했고, VOS에서는 STM을 fine-tuning했다. 데이터셋은 주로 Cityscapes와 ADE20K를 사용했고, 추가로 WMH 데이터셋에서 Boundary Loss와 비교했다. VOS는 DAVIS-2016과 YouTube-VOS training data를 사용해 fine-tuning 후 DAVIS-2016 validation에서 평가했다.

평가 지표는 pixel accuracy, mIoU, boundary F-score다. boundary F-score는 GT boundary를 1, 3, 5 pixel dilation한 영역 안에서 계산했다. 이는 경계 정렬 품질을 보기 위한 지표다.

먼저 ablation 결과를 보면, ABL은 단독으로 늦은 시점에 조금 추가하는 정도로는 효과가 제한적이었다. Cityscapes에서 DeepLabV3 기준으로 `CE`는 mIoU 79.5, `CE + ABL(마지막 20%)`는 79.6으로 0.1% 향상에 그쳤다. 반면 `CE + IoU`는 80.2, `CE + IABL`은 80.5로 올라갔다. 즉, ABL은 IoU 계열 loss와 함께 쓸 때 의미 있게 작동했다. detach를 제거하면 `CE + IABL w/o detach`가 76.0으로 크게 떨어져, conflict suppression이 필수임을 보여준다.

ADE20K에서도 비슷한 경향이 나타난다. OCR 기준 single-scale mIoU는 `CE` 44.51, `CE + IoU` 44.73, `CE + IABL` 45.38이다. multi-scale에서는 45.66, 46.54, 46.88 순으로 증가한다. UperNet(Swin-T)에서도 `CE + IABL`이 `CE + IoU`와 `CE + IoU + BL`보다 약간 더 좋다. 즉, CNN과 Transformer 기반 segmentation 모두에서 성능 개선이 확인되었다.

대표적인 최종 성능은 다음과 같다. ADE20K validation에서 OCR은 multi-scale mIoU 45.66인데 OCR+IABL은 46.88로 1.22% 향상되었다. UperNet(Swin-B)은 51.66에서 52.40으로 0.74% 향상되었다. Cityscapes validation에서는 OCR이 82.2, OCR+IABL이 82.9로 0.7% 상승했다.

Segfix와 비교한 결과도 흥미롭다. Cityscapes validation에서 DeepLabV3 기준 Segfix와 IABL 모두 baseline 대비 mIoU를 1.0 올려 80.5를 기록했다. 그러나 OCR 기준으로는 Segfix가 81.7, IABL이 82.0으로 IABL이 0.3 높다. boundary F-score에서는 1-pixel처럼 매우 엄격한 조건에서는 Segfix가 약간 유리하지만, 3-pixel과 5-pixel에서는 IABL이 더 높다. 특히 thin object인 traffic light 클래스에서 IABL이 모든 dilation 설정에서 더 높은 boundary F-score를 보였다. 저자들은 이것이 interior pixel propagation 기반인 Segfix가 얇은 물체에서는 약할 수 있음을 보여준다고 해석한다.

Boundary Loss(BL)와의 비교도 수행되었다. WMH 데이터셋에서는 `GDL`만 썼을 때 DSC 0.727, `GDL+BL`은 0.748, `GDL+ABL`은 0.768이었다. Hausdorff distance도 1.045, 0.987, 0.980으로 개선되었다. 또한 Cityscapes와 ADE20K에서 multi-class 확장 BL과 비교했을 때도 `IoU+ABL`이 `IoU+BL`보다 높은 mIoU를 보였다. 논문은 BL이 region integral 관점이라 GTB에 아주 가까운 픽셀의 영향이 상대적으로 약해질 수 있고, ABL은 PDB 자체에 직접 집중하므로 정렬에 더 효과적이라고 주장한다.

FKL(full KL-divergence) loss와의 비교도 있다. 이 손실은 경계 픽셀만이 아니라 이미지 내 모든 인접 픽셀 쌍에 대해 KL divergence를 감독한다. 어느 정도 성능 향상은 있었지만, `CE+IFKL`은 `CE+IABL`보다 consistently 낮았다. 저자들은 그 이유를, FKL은 모든 픽셀을 동일하게 다루지만 ABL은 PDB에만 집중해 더 점진적이고 안정적으로 네트워크를 조정하기 때문이라고 설명한다.

threshold 1% 설정의 타당성도 실험했다. Cityscapes에서 FCN(HRNetV2-W18s) 기준 1%가 mIoU 75.59%, 2%는 75.46%, 0.5%는 75.41%였다. 큰 차이는 아니지만 1%가 가장 좋았다.

또한 IoU loss에 대한 ABL 의존성을 보기 위해 training 중 IoU weight를 1에서 0으로 줄이고 ABL weight를 0에서 1로 늘리는 실험을 했다. 결과는 mIoU 75.65%로, 고정된 `CE+IoU+ABL`의 75.59%와 비슷했다. 이는 ABL이 단지 IoU에 얹힌 보조 신호가 아니라, 어느 정도 독립적으로 boundary refinement에 기여함을 보여준다.

VOS에서도 효과가 확인되었다. STM을 1k iteration fine-tuning했을 때, baseline은 $J$-mean 88.67, $F$-mean 89.86이었다. `CE+IoU` fine-tuning은 89.08 / 90.66, `CE+IABL`은 89.29 / 90.82였다. 수치 차이는 크지 않지만 contour accuracy인 $F$-mean이 가장 높았고, qualitative 결과에서도 scooter나 motorcycle의 경계가 더 자연스럽게 정렬되었다.

정성 결과에서는 traffic light, pole, motorcycle tail 같은 가늘고 복잡한 구조에서 PDB가 GTB 쪽으로 점진적으로 이동하는 모습이 제시된다. 논문 Figure 4는 학습 iteration이 진행될수록 빨간색 PDB가 파란색 GTB 쪽으로 붙는 과정을 보여주며, ABL의 설계 의도와 잘 맞는다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 boundary alignment를 직접적으로 다루는 loss를 매우 단순한 형태로 설계했다는 점이다. 네트워크 구조를 바꾸지 않고 loss만 추가하면 되므로 적용성이 높다. 실제로 DeepLabV3, OCR, UperNet, STM 등 서로 다른 구조에서 모두 성능 향상을 보였다. 이는 ABL이 특정 아키텍처에 종속된 trick이 아니라는 점을 뒷받침한다.

두 번째 강점은 “현재 예측 경계를 기반으로 supervision target이 달라진다”는 동적 설계다. 많은 loss가 고정된 GT에 대해 픽셀별 오차만 보는데, ABL은 현재 예측 상태를 반영해 다음에 어디로 움직여야 하는지까지 정의한다. 이 점이 단순한 boundary regularization보다 한 단계 더 적극적인 제어로 보인다.

세 번째 강점은 얇은 물체나 복잡한 경계에서 실질적인 개선을 보였다는 점이다. 논문은 단순히 평균 mIoU만 올렸다고 주장하지 않고, boundary F-score, class-wise 성능, qualitative 예시를 통해 ABL이 실제로 경계 품질을 개선함을 보여준다. 특히 traffic light 같은 thin structure에 대한 분석은 설득력이 있다.

하지만 한계도 분명하다. 첫째, ABL은 PDB가 존재해야 작동한다. 저자들도 명시하듯이, 예측 경계가 GTB와 너무 멀리 떨어져 있거나 아예 작은 객체를 놓쳐 PDB가 잘 형성되지 않으면 ABL 신호는 유용하지 않을 수 있다. 그래서 Lovasz-Softmax 같은 IoU loss를 함께 써야 한다. 즉, ABL 단독의 자립성은 제한적이다.

둘째, 방법의 성공이 detach 같은 구현 세부사항에 크게 의존한다. conflict suppression이 없으면 성능이 크게 무너진다는 점은, 이 손실이 본질적으로 불안정한 gradient interaction을 가지고 있음을 뜻한다. 논문도 이를 완전히 해결한 것은 아니고, practical한 우회책으로 막았다고 보는 편이 정확하다.

셋째, 방향 선택이 local 8-neighborhood에 기반한다는 점도 제한이다. 이는 계산량 면에서는 효율적이지만, GTB가 더 멀리 있거나 복잡한 형태를 가질 때는 장거리 구조를 충분히 반영하지 못할 수 있다. 논문은 global search가 느리기 때문에 쓰지 않았다고 설명하지만, 그렇다면 현재 방법은 local heuristic에 가깝다.

넷째, 개선 폭이 항상 매우 크지는 않다. Cityscapes나 ADE20K에서 mIoU 향상은 대체로 0.3%에서 1.2% 정도다. segmentation 분야에서는 의미 있는 향상일 수 있지만, 계산 복잡도와 구현 복잡성을 고려할 때 모든 상황에서 압도적인 개선이라고 보기는 어렵다. 특히 경계 품질 향상이 주된 가치이고, region 성능 향상은 보조적이라고 해석하는 것이 더 적절하다.

또한 논문은 계산 비용 증가량, 학습 시간 오버헤드, 메모리 비용을 정량적으로 자세히 보고하지 않는다. boundary 검출, distance transform, local direction prediction이 추가되므로 분명 비용이 생기지만, 그 규모는 본문에서 충분히 분석되지 않았다. 이 부분은 명확히 제시되지 않은 정보다.

## 6. 결론

이 논문은 semantic segmentation의 경계 품질 문제를 해결하기 위해 Active Boundary Loss를 제안했다. ABL은 현재 예측 경계에서 출발해, 각 경계 픽셀이 GT boundary에 가까워지도록 이동 방향을 예측하는 형태로 loss를 설계한다. 이 과정에서 KL divergence 기반 boundary detection, distance transform 기반 direction target, weighted cross-entropy, detach를 통한 conflict suppression이 핵심 요소로 사용된다.

실험적으로 ABL은 Cityscapes, ADE20K, DAVIS-2016, WMH 등 여러 환경에서 mIoU, boundary F-score, contour accuracy를 개선했다. 특히 thin object와 복잡한 경계에서 효과가 두드러졌고, CNN 기반 모델뿐 아니라 Transformer 기반 모델과 VOS 모델에도 적용 가능함을 보였다.

종합하면, 이 연구의 주요 기여는 “경계를 잘 맞추는 loss”를 제안한 것이 아니라, 예측 경계를 능동적으로 GT 쪽으로 이동시키는 학습 관점을 제시했다는 데 있다. 실제 응용에서는 자율주행, 의료영상, 비디오 객체 분할처럼 경계 정밀도가 중요한 작업에 의미가 크다. 향후에는 논문 저자들이 언급하듯 conflict를 더 근본적으로 줄이는 방법, 그리고 depth prediction 같은 다른 dense prediction task로의 확장이 중요한 후속 연구 방향이 될 가능성이 높다.
