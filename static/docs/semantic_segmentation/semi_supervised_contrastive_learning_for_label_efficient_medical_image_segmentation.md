# Semi-supervised Contrastive Learning for Label-efficient Medical Image Segmentation

- **저자**: Xinrong Hu, Dewen Zeng, Xiaowei Xu, Yiyu Shi
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2109.07407

## 1. 논문 개요

이 논문은 의료 영상 분할에서 라벨이 매우 적을 때도 성능을 높일 수 있는 semi-supervised contrastive learning 방법을 제안한다. 의료 영상 분할은 각 픽셀 또는 voxel에 대해 정확한 class를 예측해야 하므로, 일반적인 image classification보다 훨씬 많은 정밀 annotation이 필요하다. 그러나 의료 영상 annotation은 전문가 지식이 필요하고 pixel-wise labeling 비용이 매우 크기 때문에, 충분한 labeled data를 확보하기 어렵다.

기존의 label-efficient segmentation 접근 중 하나는 contrastive learning을 이용해 unlabeled data에서 representation을 먼저 학습한 뒤, 소량의 labeled data로 fine-tuning하는 방식이다. 하지만 기존 방법들은 pre-training 단계에서 label 정보를 거의 쓰지 않거나 전혀 쓰지 않았다. 이 논문은 바로 이 지점을 문제로 본다. 저자들은 “이미 소량이라도 pixel-wise label이 있다면, 그것을 pre-training 단계의 local representation 학습에 직접 반영하는 것이 더 낫다”는 관점을 취한다.

따라서 논문의 핵심 문제는 다음과 같이 정리할 수 있다. unlabeled data를 활용하는 contrastive learning의 장점을 유지하면서도, 제한된 pixel-wise label을 pre-training에 효과적으로 포함시켜 segmentation에 더 적합한 feature를 학습할 수 있는가? 저자들은 이에 대해 supervised local contrastive loss를 제안하고, 계산량 문제를 줄이기 위한 practical strategy까지 함께 제시한다. 의료 영상처럼 annotation 비용이 높은 분야에서는 이런 접근이 실제로 매우 중요하다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 contrastive learning을 두 수준으로 나누는 것이다. 먼저 unlabeled data 전체를 사용하여 image-level representation을 학습하는 self-supervised global contrast를 수행하고, 그 다음 소량의 labeled data를 사용하여 pixel-level representation을 학습하는 supervised local contrast를 수행한다. 즉, 전역적 의미 정보는 unlabeled data에서, 세밀한 픽셀 수준 구분 정보는 limited label에서 끌어오겠다는 설계다.

기존 연구 [4]의 local contrast는 self-supervised 방식이었다. 이 경우 같은 원본 이미지에서 augmentation된 두 feature map의 “같은 위치”를 positive pair로 간주했다. 하지만 segmentation에서는 spatial transformation이 들어가면 같은 위치가 반드시 같은 anatomical structure를 의미하지 않으며, 또 label이 없으면 선택된 point들이 대부분 background일 수도 있다. 이런 점 때문에 local feature를 class-aware하게 학습하는 데 한계가 있다.

반면 이 논문은 supervised local contrast를 통해 같은 class label을 가진 pixel embedding끼리는 latent space에서 가깝게, 다른 class의 pixel embedding끼리는 멀어지게 만든다. 이것은 segmentation 문제의 본질과 직접 연결된다. 결국 이 방법은 단순히 “representation을 잘 배우자”가 아니라, “segmentation에 실제로 필요한 pixel-level discrimination을 label guidance로 미리 학습하자”는 접근이라고 볼 수 있다.

## 3. 상세 방법 설명

전체 framework는 세 단계로 이해할 수 있다. 첫째, encoder $E$만 사용해 self-supervised global contrastive learning을 수행한다. 둘째, decoder $D$를 붙인 뒤 supervised local contrastive learning으로 encoder와 decoder를 함께 학습한다. 셋째, 이렇게 얻은 model을 initialization으로 사용하여 labeled data에 대해 일반적인 segmentation fine-tuning을 수행한다.

global contrast 단계는 SimCLR과 유사하다. 입력 batch $B=\{x_1, x_2, ..., x_b\}$의 각 2D slice에 대해 augmentation을 두 번 적용해 positive pair를 만든다. 각 augmented image $a_i$는 encoder와 projection head $h_1(\cdot)$를 거쳐 normalized feature $z_i$로 변환된다. 논문은 global contrastive loss를 다음과 같이 정의한다.

$$
L_g=-\frac{1}{|A|}\sum_{i\in I}\log \frac{\exp(z_i \cdot z_{j(i)} / \tau)}{\sum_{k\in I-\{i\}} \exp(z_i \cdot z_k / \tau)}
$$

여기서 $z_{j(i)}$는 $a_i$와 같은 원본 이미지에서 나온 다른 augmentation의 feature이고, $\tau$는 temperature constant이다. 이 loss는 같은 이미지에서 나온 두 view의 전역 representation을 가깝게 만들고, 다른 이미지의 representation과는 멀어지게 한다. 이 단계는 image-level semantic feature를 학습하는 역할을 한다.

하지만 segmentation은 이미지 전체의 분류가 아니라 각 픽셀의 분류 문제이므로, global contrast만으로는 충분하지 않다. 그래서 저자들은 local contrastive learning을 추가한다. decoder의 상위 block $D_l$에서 얻은 feature map에 point-wise convolution head $h_2(\cdot)$를 적용해 local embedding map을 만든다. 논문 표기에서 이 feature는 $f^l(\tilde{x_i}) = h_2(D_l(E(a_i)))$로 나타난다. 각 위치 $(u,v)$의 feature vector $f^l_{u,v} \in \mathbb{R}^c$에 대해 local contrastive loss는 다음과 같이 정의된다.

$$
loss(a_i)=-\frac{1}{|\Omega|}\sum_{(u,v)\in \Omega}\frac{1}{|P(u,v)|}\log
\frac{\sum_{(u_p,v_p)\in P(u,v)} \exp(f^l_{u,v}\cdot f^l_{u_p,v_p}/\tau)}
{\sum_{(u',v')\in N(u,v)} \exp(f^l_{u,v}\cdot f^l_{u',v'}/\tau)}
$$

그리고 전체 local loss는

$$
L_l=\frac{1}{|A|}\sum_{a_i \in A} loss(a_i)
$$

이다.

여기서 중요한 것은 $\Omega$, $P(u,v)$, $N(u,v)$의 정의다. 기존 self-supervised local contrast에서는 positive set이 대체로 같은 spatial position의 대응점으로 구성된다. 하지만 이 논문에서는 supervised setting으로 바꾸어, 어떤 위치 $(u,v)$의 positive set $P(u,v)$를 “동일한 annotation label을 가진 모든 feature”로 정의한다. negative set $N(u,v)$는 다른 class label을 가진 모든 feature다. 즉, 같은 class끼리 모으고 다른 class끼리 분리하는 supervised contrastive learning 철학을 pixel level에 적용한 셈이다.

또 하나의 설계 포인트는 background 처리다. 의료 영상에서 background pixel은 매우 많기 때문에, background끼리의 positive pair가 지나치게 많아지면 실제 segmentation에 중요한 organ/structure 구분 학습이 약해질 수 있다. 그래서 저자들은 supervised local contrast에서 $\Omega$를 non-background annotation을 가진 위치들만 포함하도록 정의한다. 다만 background pixel은 negative set에는 여전히 포함된다. 이것은 실제로 foreground structure discrimination을 강화하려는 의도다.

문제는 계산량이다. 입력 해상도가 $h \times h$일 때, supervised local contrast는 positive set과 negative set이 매우 커질 수 있어 overall complexity가 $O(h^4)$까지 올라간다. 특히 MMWHS처럼 원래 해상도가 크고 resize 후에도 $160 \times 160$이면, loss 계산 비용이 매우 크다. 이를 해결하기 위해 저자들은 두 가지 전략을 제안한다.

첫 번째는 downsampling이다. feature map에서 일정 stride로 점을 건너뛰며 contrastive loss를 계산한다. 인접 픽셀들은 비슷한 정보를 가지므로 일부만 써도 된다는 직관이다. stride를 $s$로 두면 positive set 크기와 계산 복잡도는 각각 대략 $s^2$, $s^4$ 배 감소한다. 논문에서는 OOM을 피할 수 있는 최소 stride로 $s=4$를 사용했다.

두 번째는 block division이다. feature map을 여러 개의 작은 block으로 나누고, 각 block 내부에서만 local contrastive loss를 계산한 뒤 평균을 낸다. block 크기를 $h' \times h'$라고 하면, 한 위치의 positive/negative 비교 범위가 block 내부로 제한되므로 계산량이 크게 줄어든다. 전체 complexity는 $O(h^2 h'^2)$ 수준으로 감소한다. 논문에서는 $16 \times 16$ block을 사용했다. 모든 pixel이 background인 block은 loss 계산에서 제외한다.

정리하면, 이 방법의 구조적 핵심은 global contrast로 coarse image representation을 얻고, supervised local contrast로 class-aware pixel embedding을 학습한 뒤, 이를 segmentation fine-tuning의 좋은 초기값으로 활용하는 것이다.

## 4. 실험 및 결과

실험은 서로 다른 modality의 두 공개 의료 영상 데이터셋에서 수행되었다. 첫 번째는 MRI 기반 Hippocampus dataset이며, MICCAI 2018 Medical Segmentation Decathlon의 데이터다. 총 260개의 3D volume이 있고 anterior/posterior hippocampus body annotation이 포함되어 있다. 두 번째는 CT 기반 MMWHS 데이터셋으로, MICCAI 2017 challenge 데이터이며 총 20개의 3D cardiac CT volume과 7개 심장 구조의 annotation이 제공된다.

전처리로는 volume 단위 intensity normalization을 적용하고, 같은 데이터셋 내 모든 slice를 bilinear interpolation으로 resize했다. 최종 해상도는 Hippocampus가 $64 \times 64$, MMWHS가 $160 \times 160$이다. backbone은 PyTorch 기반 2D U-Net이며 encoder와 decoder에 각각 3개 block을 사용했다.

데이터는 training set $X_{tr}$, validation set $X_{vl}$, test set $X_{ts}$로 나누었다. Hippocampus는 3:1:1, MMWHS는 2:1:1 비율이다. 이후 training set 안에서 일부 volume만 labeled로 두고 나머지는 unlabeled로 취급했다. 데이터 분할은 네 번 독립적으로 수행하여 4-fold 평균 Dice score를 비교 지표로 사용했다.

최적화는 contrastive learning과 segmentation training 모두 Adam을 사용했다. learning rate는 contrastive learning에서 0.0001, segmentation fine-tuning에서 0.00001이다. contrastive pre-training은 70 epoch, segmentation training은 최대 120 epoch까지 수행했다. 실험 자원은 Hippocampus에 NVIDIA GTX 1080 두 장, MMWHS에 NVIDIA Tesla P100 두 장을 사용했다.

비교 방법은 다음과 같다. `random`은 scratch 학습, `global`은 SimCLR형 global contrast만 사용, `global+local(self)`는 기존 self-supervised global/local contrast 방식 [4], `Mixup`은 data augmentation baseline, `TCSM`은 semi-supervised segmentation baseline [14]다. 제안 방법은 `local(stride)`, `local(block)`, `global+local(stride)`, `global+local(block)`의 네 변형으로 평가했다.

결과를 보면, 먼저 모든 contrastive learning 기반 방법이 대체로 `random`보다 높은 Dice를 보였다. 이는 label이 부족한 상황에서 contrastive pre-training이 효과적인 initialization임을 보여준다. 둘째, supervised local contrast만 사용한 `local(stride)`와 `local(block)`도 기존 contrastive baseline들과 비슷하거나 일부 조건에서 더 나은 성능을 냈다. 논문 저자들은 이를 통해, 비록 labeled data만 사용하더라도 supervised local contrast가 pixel-wise class representation 학습에 상당히 효율적이라고 해석한다.

가장 중요한 결과는 global contrast와 supervised local contrast를 결합한 경우다. Hippocampus에서 `global+local(block)`은 labeled ratio 5%, 10%, 20%에서 각각 0.824, 0.849, 0.866 Dice를 기록했다. MMWHS에서는 labeled ratio 10%, 20%, 40%에서 각각 0.382, 0.553, 0.764를 기록했다. 특히 MMWHS 20%와 40% labeled setting에서 `global+local(block)`은 모든 비교 방법 중 최고 성능을 보였다. `global+local(stride)`도 대체로 강한 결과를 보였지만, block division variant가 조금 더 우수했다.

정성적 결과도 제시되었다. MMWHS 40% labeled setting의 segmentation visualization에서는 제안 방법들이 `random`과 기존 `global+local(self)`보다 ground truth에 더 가까운 구조 분할을 보였다. 또한 t-SNE visualization에서는 `global+local(block)`이 같은 class feature를 더 조밀하게 cluster하고, 다른 class feature를 더 잘 분리하는 모습을 보였다. 이는 supervised local contrast가 실제로 embedding space를 더 class-discriminative하게 만든다는 논문의 주장과 일치한다.

다만 실험에서 통계적 유의성 검정이나 class별 상세 score는 본문에 명시되지 않았다. 또한 supervised local contrast가 오직 labeled subset만 사용한다는 점에서, unlabeled data 활용 효율과 label 활용 효율 사이의 trade-off를 더 세밀하게 분석한 결과는 본문에 충분히 제시되지 않았다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 segmentation 문제의 구조에 맞게 contrastive learning을 재설계했다는 점이다. 단순히 image-level representation을 잘 학습하는 데 그치지 않고, 실제 segmentation 성능과 직접 연결되는 pixel-level embedding을 class-aware하게 학습하도록 supervised local contrast를 설계했다. 이는 문제 설정과 방법론이 잘 맞물린 사례다.

또 다른 강점은 제한된 label을 pre-training 단계에 적극적으로 활용했다는 점이다. 기존 방법은 pre-training에서는 label을 쓰지 않고 fine-tuning에서만 사용했는데, 이 논문은 그 label을 earlier stage에 투입하면 representation 자체를 더 좋은 방향으로 끌고 갈 수 있음을 실험적으로 보여준다. 특히 global contrast와 supervised local contrast가 상호보완적이라는 실험 결과는 설득력이 있다.

실용성 측면에서도 계산량 문제를 피하지 않고 downsampling과 block division이라는 구체적 전략을 제시한 점이 좋다. contrastive loss를 pixel level에서 계산하면 메모리와 연산량이 폭증하는데, 저자들은 단순하지만 효과적인 근사 전략을 통해 실제 학습 가능성을 확보했다.

한계도 분명하다. 첫째, 방법은 supervised local contrast를 위해 pixel-wise annotation이 있는 labeled subset을 필요로 한다. 완전 self-supervised는 아니며, labeled data가 극단적으로 적거나 class imbalance가 심한 경우 positive set 구성이 불안정할 수 있다. 둘째, 계산량 완화를 위해 downsampling이나 block restriction을 사용하면 원래 contrastive learning의 중요한 요소인 많은 negative pair를 일부 포기하게 된다. 논문도 이 trade-off를 인정하며, stride와 block size를 OOM 기준으로 정했다고 설명한다. 즉, 이 하이퍼파라미터는 이론적으로 최적화되었다기보다 하드웨어 제약에 맞춘 practical choice에 가깝다.

셋째, backbone은 2D U-Net이고 실험도 두 데이터셋에 제한되어 있다. 3D segmentation, 더 큰 해상도, 더 다양한 organ/task에 대해 얼마나 일반화되는지는 논문 본문만으로는 확인할 수 없다. 넷째, supervised local contrast가 어떤 decoder layer에서 가장 효과적인지, projection head 설계가 결과에 미치는 영향, class imbalance에 대한 민감도 같은 세부 분석은 충분히 제공되지 않는다.

비판적으로 보면, 이 논문은 명확한 개선을 보이지만 성능 향상의 원인을 완전히 분해해서 보여주지는 않는다. 예를 들어 global contrast의 기여, supervised local contrast의 기여, foreground-only anchor sampling의 기여가 어느 정도씩 분리되는지는 제한적으로만 드러난다. 그럼에도 제안의 방향성은 타당하고, 실험 결과도 비교적 일관적이다.

## 6. 결론

이 논문은 label-efficient medical image segmentation을 위해 supervised local contrastive loss를 제안하고, 이를 self-supervised global contrast와 결합하는 semi-supervised framework를 제시했다. 핵심 기여는 소량의 pixel-wise annotation을 pre-training 단계의 local representation 학습에 직접 활용하여, 같은 class의 pixel embedding은 가깝게, 다른 class의 embedding은 멀어지게 만드는 데 있다. 또한 downsampling과 block division을 통해 이 loss의 높은 계산 비용을 완화했다.

실험 결과, 제안 방법은 Hippocampus와 MMWHS 두 의료 영상 데이터셋에서 기존 contrastive 방법과 다른 semi-supervised baseline들을 전반적으로 능가했다. 특히 global contrast와 supervised local contrast를 함께 사용할 때 가장 좋은 성능을 보였으며, 이는 image-level representation과 pixel-level representation이 상호보완적이라는 점을 시사한다.

실제 적용 측면에서 이 연구는 annotation 비용이 큰 의료 영상 분야에서 매우 의미가 있다. 앞으로 더 다양한 backbone, 3D setting, class imbalance 대응, memory-efficient contrastive design과 결합된다면, label-efficient segmentation 연구의 중요한 출발점 또는 확장 기반으로 활용될 가능성이 크다.
