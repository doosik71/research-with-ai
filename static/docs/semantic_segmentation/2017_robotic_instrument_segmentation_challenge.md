# 2017 Robotic Instrument Segmentation Challenge

- **저자**: M. Allan, A. Shvets, T. Kurmann, Z. Zhang, R. Duggal, Y. H. Su, N. Rieke, I. Laina, N. Kalavakonda, S. Bodenstedt, L. C. Garcia-Peraza-Herrera, W. Li, V. Iglovikov, H. Luo, J. Yang, D. Stoyanov, L. Maier-Hein, S. Speidel, M. Azizian
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1902.06426

## 1. 논문 개요

이 논문은 새로운 segmentation model을 제안하는 논문이라기보다, robotic minimally invasive surgery 장면에서 **da Vinci 수술 도구를 segmentation하는 공통 benchmark를 설계하고, 2017 challenge의 결과를 체계적으로 비교 분석한 보고서**에 가깝다. 저자들은 컴퓨터 비전 분야에서 ImageNet, COCO, KITTI 같은 공개 데이터셋이 연구 발전을 가속한 것처럼, robotic surgery 분야에도 공통 데이터와 평가 기준이 필요하다고 본다.

연구 문제는 명확하다. 수술 장면에서 instrument를 정확히 pixel 단위로 분할하는 것은 instrument tracking, automation, guidance assistance, augmented reality overlay masking 같은 응용의 기반이다. 그러나 당시 이 분야는 공통 데이터셋과 일관된 validation 체계가 부족했고, 2015 challenge 데이터셋은 배경 다양성이 낮고, robot forward kinematics 기반 자동 annotation의 오차가 크며, 복잡한 운동을 충분히 반영하지 못하는 한계가 있었다.

이 논문의 중요성은 두 가지다. 첫째, 실제 수술 비디오에 가까운 장면에서 hand-created label을 제공함으로써 이후 연구의 비교 가능성을 높였다. 둘째, 단순한 binary segmentation만이 아니라 **binary, parts, type segmentation**이라는 점진적으로 어려운 세 가지 과제를 설정해, 방법들의 강점과 약점을 더 입체적으로 드러냈다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 “더 좋은 모델” 자체보다, **더 신뢰할 수 있는 benchmark와 challenge 설계**를 제공하는 데 있다. 이를 위해 저자들은 2015 challenge 대비 세 가지를 개선했다. 첫째, 자동 라벨 대신 **수작업 annotation**을 사용해 label quality를 높였다. 둘째, 10개의 서로 다른 porcine procedure를 사용해 배경과 상황의 다양성을 늘렸다. 셋째, instrument 전체뿐 아니라 **instrument part와 instrument type까지 세분화한 label**을 제공했다.

세부 과제는 다음과 같다. 첫 번째는 instrument 대 background를 나누는 binary segmentation이다. 두 번째는 instrument를 shaft, wrist, jaws 같은 part로 나누는 part segmentation이다. 세 번째는 Large Needle Driver, Prograsp Forceps, Monopolar Curved Scissors, Vessel Sealer 등 **도구 종류 자체를 구분하는 type segmentation**이다. 이 구성은 난이도를 단계적으로 높이며, 단순 foreground detection에서 fine-grained recognition으로 이어지는 실제 문제 구조를 반영한다.

기존 접근과의 차별점은 데이터와 평가 방식에 있다. 이 challenge는 수술 데이터에서 흔한 문제인 작은 데이터 규모, 높은 프레임 상관성, specular reflection, smoke, blur, 기구 종류 불균형 같은 현실적 어려움을 그대로 포함한다. 따라서 높은 점수를 얻는 방법이 단순히 “깨끗한 데이터셋에 과적합된 방법”이 아니라, 실제 surgical scene에서 견딜 수 있는지를 어느 정도 보여준다.

## 3. 상세 방법 설명

이 논문의 방법론적 중심은 하나의 통합 모델이 아니라, **challenge dataset 설계, annotation 방식, 평가 프로토콜, 그리고 참가팀들의 대표적 방법 비교**에 있다.

데이터는 da Vinci Xi 시스템으로 촬영한 **10개의 abdominal porcine procedure sequence**로 구성된다. 각 procedure에서 instrument motion과 visibility가 충분한 active sequence를 고른 뒤, 초당 1프레임 비율로 300프레임을 샘플링했다. motion이 거의 없는 프레임은 수동으로 제거하고, 총 300프레임이 되도록 sequence를 연장했다. stereo camera의 left/right image와 calibration 정보도 제공되었지만, annotation은 labelling 시간을 줄이기 위해 **left eye 영상만** 수행되었다.

학습/평가 분할은 조금 특이하다. 8개 sequence에서는 처음 225프레임을 training으로, 마지막 75프레임을 test로 두었고, 나머지 2개 sequence는 전체 300프레임을 test set으로 유지했다. 다만 동일 sequence에서 train/test가 함께 존재하므로, 특정 split test set을 평가할 때는 그와 대응하는 training sequence를 사용하지 못하게 규정했다. 그 결과 일부 팀은 공정한 평가를 위해 **최소 9개의 모델**을 따로 학습해야 했다. 추가 surgical data 사용은 금지되었고, 공개된 비수술 데이터로 pretraining한 CNN만 허용되었다.

라벨링은 Intuitive Surgical의 전담 segmentation team이 VIAME 도구를 사용해 polygon 기반으로 프레임별 수작업 annotation을 수행했다. 라벨은 instance level로 제공되었다. 데이터에는 7종의 robotic surgical instrument와 drop-in ultrasound probe가 등장하지만, challenge task 정의상 probe는 instrument class가 아닌 background로 처리되는 경우가 있다. 이 점은 binary task에서 혼동 요인이 된다.

참가 방법들은 대부분 encoder-decoder 계열 CNN을 사용했다. University of Bern은 cascaded FCN 구조를 사용했는데, 먼저 binary segmentation을 수행하고 그 결과를 part/type segmentation refinement에 활용했다. 기본 네트워크는 U-Net과 유사한 encoder-decoder 구조이며, 각 block은 convolution, batch normalization, activation, residual connection으로 이루어진다. MIT 팀은 U-Net 계열 구조에 **pretrained VGG11/VGG16 encoder**를 결합한 TernausNet을 사용했다. SIAT는 **SegNet** 기반 encoder-decoder를 사용했고, parts segmentation으로 학습한 가중치를 type segmentation에 전달해 fine-tuning했다. UCL은 ToolNet이라는 실시간 multiscale network를 사용했고, TUM은 ResNet-50 encoder 기반 FCRN 스타일 구조에 long-range skip connection을 붙였다. 반면 UW는 machine learning을 쓰지 않고, color filtering, GrabCut, blur-aware feature weighting, border prior를 결합한 전통적 computer vision 방법을 제안했다. 따라서 challenge는 deep learning 계열과 non-deep learning 계열을 함께 비교한 셈이다.

논문에 제시된 대표적인 수식은 세 가지다. 먼저 SIAT 팀은 class imbalance를 완화하기 위해 softmax에 class weight를 반영한 식을 사용했다고 설명한다.

$$
\sigma_i(z)=\frac{k_i \exp(z_i)}{\sum_{j=1}^{m}\exp(z_j)}, \quad i=1,\dots,m
$$

여기서 $k_i$는 class weight, $m$은 class 수, $z_i$는 클래스 $i$의 예측값이다. 논문 표현 그대로 따르면 softmax 출력에 class weight를 곱해 특정 희소 class의 영향력을 키우려는 의도다.

UCL ToolNet은 multi-scale prediction을 사용하며, scale별 예측을 합친 평균 prediction과 개별 scale prediction 모두에 대해 IoU loss를 적용한다. 논문은 이를 다음과 같이 적는다.

$$
\hat{y}^{(\bar{s})}(z,\theta)=\sum_{j=1}^{M} w_j \hat{y}^{(s_j)}(z,\theta)
$$

$$
L_{MSIoU}(y,z,\theta)=\bar{\lambda}L_{IoU}(\hat{y}^{(\bar{s})}(z,\theta),y)+\sum_{j=1}^{M}\lambda_j L_{IoU}(\hat{y}^{(s_j)}(z,\theta),y)
$$

여기서 $y$는 ground truth, $z$는 입력 영상, $\theta$는 네트워크 파라미터, $\hat{y}^{(s_j)}$는 scale $j$에서의 prediction, $\hat{y}^{(\bar{s})}$는 여러 scale prediction의 가중합이다. 핵심은 한 해상도에서만 supervision을 주는 것이 아니라, 여러 scale에서 동시에 supervision을 걸어 fine-grained segmentation과 global context를 함께 학습한다는 점이다.

Challenge 전체의 평가는 mean Intersection-over-Union으로 이루어진다. 한 클래스의 IoU는

$$
IoU=\frac{TP}{TP+FP+FN}
$$

로 정의된다. 여기서 $TP$는 true positive, $FP$는 false positive, $FN$은 false negative다. 프레임별로 존재하는 클래스들에 대해서만 arithmetic mean을 구하고, 이후 프레임 평균과 dataset 크기 가중 평균으로 overall score를 계산했다. 이는 segmentation benchmark에서 널리 쓰이는 standard metric이다.

논문은 또한 augmentation과 post-processing이 성능에 중요하다고 설명한다. 상위권 팀들은 horizontal/vertical flip, rotation, zoom, color augmentation 등을 활용했고, 특히 TUM은 instrument shaft를 따라 artificial specular reflection을 추가하는 **application-specific augmentation**을 도입했다. TUM과 UW는 “instrument는 화면 경계에서 들어온다”는 수술 장면의 prior를 post-processing에 반영해, border와 연결되지 않는 작은 island prediction을 제거했다.

## 4. 실험 및 결과

실험은 세 가지 task로 구성된다. binary segmentation은 instrument 전체 대 background를 구분하는 문제다. part segmentation은 shaft, wrist, clasper를 분리한다. type segmentation은 도구 종류를 구분한다. 데이터셋은 10개 sequence이며, dataset 9와 10은 다른 dataset보다 프레임 수 비중이 커 최종 평균에 더 큰 영향을 준다.

binary segmentation 결과를 보면, 전체 평균 mean IoU는 **MIT 0.888**, **UB 0.875**, **TUM 0.873**, **NCT 0.843**, **SIAT 0.803** 순으로 높다. dataset별 최고 성능은 MIT가 6개, UB가 3개, TUM이 1개 dataset에서 차지했다. 논문 본문에서는 binary task의 최고 평균 점수가 0.854라고 서술되어 있는데, 제공된 표(Table I)의 최종 평균 행에는 MIT가 0.888로 기록되어 있다. 보고서 관점에서는 **표에 제시된 수치와 본문 서술 사이에 불일치가 있다**고 보는 것이 안전하다. 이 task는 세 과제 중 가장 쉬웠지만, ultrasound probe나 needle이 instrument와 비슷한 외형과 색을 가져 혼동을 유발했다. dataset 1의 평균 IoU가 0.589로 가장 낮았고, dataset 7은 Vessel Sealer와 복잡한 조명 때문에 방법 간 차이가 크게 나타났다.

part segmentation에서는 전체 평균이 **TUM 0.751**, **MIT 0.737**, **UB 0.700**, **NCT 0.699** 순이다. 흥미로운 점은 MIT가 10개 중 7개 dataset에서 최고 점수를 기록했지만, overall average는 TUM이 더 높다는 점이다. 논문은 dataset 9와 10이 다른 dataset보다 4배 큰 비중을 가져 TUM이 최종 평균에서 유리했다고 설명한다. 즉, “몇 개 dataset를 이겼는가”보다 “가중 평균에서 큰 비중을 차지하는 dataset에서 얼마나 잘했는가”가 최종 순위에 영향을 준다.

part를 더 세부적으로 보면 shaft는 상대적으로 잘 분할되었다. shaft component mean IoU는 TUM 0.822, MIT 0.786, UB 0.751 수준이다. 반면 wrist는 가장 어렵다. wrist mean IoU는 TUM 0.634, MIT 0.625, NCT 0.578 수준으로 크게 낮다. clasper도 어렵고, TUM 0.556, MIT 0.542, UB 0.507 정도다. 이는 손목과 집게 끝 부분이 크기가 작고, articulation이 많으며, motion blur와 specular reflection 영향을 더 심하게 받기 때문으로 해석할 수 있다. 다만 이 마지막 해석은 실험 패턴에서 자연스럽게 읽히는 내용이며, 논문이 이를 명시적 인과로 완전히 증명한 것은 아니다.

type segmentation은 가장 어려운 task다. 논문 발췌본에는 서술적 분석이 일부 생략되어 있지만, Table VI를 보면 **MIT 0.542**가 최고 overall mean IoU이고, UB 0.453, NCT 0.409, SIAT 0.371, UA 0.346, UCL 0.337 순이다. dataset별로도 MIT가 7개를 이겼고, SIAT, NCT, UB가 각 1개씩 최고를 차지했다. Figure 12 설명에 따르면 mask 자체는 대체로 정확해도, **instrument identity를 맞히는 것은 훨씬 더 어렵고**, 부분적으로 혹은 전체적으로 오분류되는 경우가 많다. 즉, “어디에 도구가 있는가”보다 “그 도구가 정확히 무엇인가”가 더 어려운 recognition problem임을 보여준다.

정성적 결과도 흥미롭다. 논문은 smoke가 낀 장면에서 UB 방법이 거의 완전히 실패하는 사례를 보여주며, 같은 프레임에서 TUM은 비교적 견디는 모습을 보고한다. 이는 동일한 overall accuracy를 가진 두 방법이라도 특정 failure mode에 대한 robustness는 크게 다를 수 있음을 보여준다. 또한 Vessel Sealer, specular highlight, motion blur, smoke 같은 현상이 segmentation 성능을 크게 떨어뜨린다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 robotic surgery vision 분야에 실제로 유용한 benchmark를 제공했다는 점이다. hand-created label, 10개 procedure 기반 데이터 구성, binary/parts/type라는 다층적 task 정의는 당시 기준으로 매우 실용적이다. 또한 단일 방법의 성능 보고가 아니라, 서로 다른 architecture와 심지어 비딥러닝 접근까지 비교해 어떤 설계가 잘 작동하는지 폭넓게 보여준다. U-Net 계열과 pretrained encoder 기반 방법들이 강세를 보였고, augmentation과 task-specific prior가 실제로 중요하다는 것도 실험적으로 드러난다.

또 다른 강점은 challenge 설계의 문제점까지 숨기지 않고 상세히 논의했다는 점이다. 저자들은 데이터셋이 총 3000프레임, 그중 training이 1800프레임에 불과하다는 사실을 분명히 인정한다. 이는 COCO 같은 대규모 benchmark와 비교하면 매우 작다. 따라서 ranking이 generalization 능력보다는 dataset-specific overfitting 영향을 받을 가능성이 있다. 또한 동일 procedure에서 train/test가 갈리는 구조 때문에 참가팀이 여러 모델을 학습해야 했고, 공정성 유지가 규칙 준수에 부분적으로 의존했다.

annotation 품질 측면의 한계도 솔직하게 드러난다. blur가 심한 프레임에서는 경계 자체가 애매하고, smoke나 lighting이 강하면 사람 annotator에게도 어려운 사례가 있다. 더 나아가 사용한 annotation tool이 polygon hole을 지원하지 않아 grasping instrument 내부 hole이 일관되게 라벨링되지 못했다. 이는 세밀한 segmentation을 학습하려는 모델 입장에서 noise가 된다. 저자들은 당시 일정 제약 때문에 각 샘플을 단일 annotator가 라벨링하고 expert 1명이 review하는 수준에 머물렀다고 적고 있으며, majority voting 같은 더 강한 quality control은 적용하지 못했다.

비판적으로 보면, 논문은 다양한 참가 방법을 소개하지만 각 방법의 학습 세부 조건이 완전히 동일하지 않다. 어떤 팀은 ImageNet이나 PASCAL pretraining을 썼고, 어떤 팀은 쓰지 않았으며, augmentation도 다르고, post-processing도 다르다. 따라서 결과 비교는 “benchmark 상의 실제 성능 비교”로서는 유효하지만, “순수한 architecture 비교 실험”으로 보기는 어렵다. 또한 type segmentation에 대한 본문 설명은 비교적 짧고, 왜 특정 도구 쌍이 특히 헷갈리는지에 대한 error analysis는 제한적이다. 제공된 발췌본 기준으로는 type task의 세부 오차 양상을 충분히 해부했다고 보기는 어렵다.

## 6. 결론

이 논문은 robotic surgical instrument segmentation을 위한 2017 challenge를 소개하고, 데이터셋 설계와 참가 방법, 그리고 정량적 결과를 체계적으로 정리한 benchmark 논문이다. 핵심 기여는 수술 로봇 영상에서 신뢰할 수 있는 수작업 annotation 데이터셋과 공통 평가 기준을 제공했다는 점, 그리고 binary segmentation에서 part segmentation, type segmentation으로 이어지는 더 어려운 문제 설정을 통해 분야의 현재 수준을 드러냈다는 점이다.

실험 결과를 종합하면, 당시에는 pretrained encoder를 사용하는 U-Net/FCN 계열 deep network가 가장 강력했으며, part와 특히 type segmentation은 여전히 어려운 문제였다. 또한 smoke, specular reflection, motion blur, class imbalance, 데이터 부족 같은 surgical vision 특유의 난제가 모델 성능을 크게 제한했다. 따라서 이 연구는 단순히 leaderboard를 제시한 것에 그치지 않고, 이후 더 큰 데이터셋, 더 정교한 annotation, 더 강한 robustness, 그리고 anatomy까지 포함한 full scene understanding으로 나아가야 한다는 방향을 제시했다. 실제 적용 측면에서도 instrument masking, tracking, AR assistance 같은 시스템의 기반 기술로 중요한 의미를 가진다.
