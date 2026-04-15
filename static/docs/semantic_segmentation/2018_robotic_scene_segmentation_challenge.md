# 2018 Robotic Scene Segmentation Challenge

- **저자**: M. Allan, S. Kondo, S. Bodenstedt, S. Leger, R. Kadkhodamohammadi, I. Luengo, F. Fuentes, E. Flouty, A. Mohammed, M. Pedersen, A. Kori, V. Alex, G. Krishnamurthi, D. Rauber, R. Mendel, C. Palm, S. Bano, G. Saibro, C. S. Shih, H. A. Chiang, J. Zhuang, J. Yang, V. Iglovikov, A. Dobrenkii, X. Liu, C. Gao, M. Unberath, M. Reddiboina, A. Reddy, M. Kim, C. Kim, C. Kim, H. Kim, G. Lee, I. Ullah, M. Luna, S. H. Park, M. Azizian, D. Stoyanov, L. Maier-Hein, S. Speidel
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2001.11190

## 1. 논문 개요

이 논문은 MICCAI 2018 EndoVis의 robotic scene segmentation challenge를 정리한 보고서이다. 핵심 목표는 robot-assisted minimally invasive surgery 영상에서 수술 기구뿐 아니라 해부학 구조와 각종 의료 기기를 픽셀 단위로 분할하는 semantic segmentation 문제를 정의하고, 공개 데이터셋과 참가팀 결과를 통해 당시 방법들의 성능 수준과 어려운 지점을 평가하는 것이다.

연구 문제는 단순한 instrument segmentation을 넘어, 실제 수술 장면에서 등장하는 다양한 대상들을 동시에 인식하는 것이다. 논문은 da Vinci instrument의 shaft, wrist, jaws/clasper 같은 세부 파트뿐 아니라 ultrasound probe, suturing needle, thread, surgical clips, 그리고 kidney parenchyma, covered kidney, small intestine 같은 anatomical class까지 포함한다. 이는 단순히 “도구가 어디 있는가”를 넘어서 “현재 시야에 어떤 구조물이 있고, 수술자가 무엇을 보고 조작하는가”를 이해하기 위한 기반 문제로 제시된다.

이 문제가 중요한 이유는 수술 내시경 영상 위에 pre-operative CT 같은 다른 의료 영상을 지능적으로 겹쳐 보여주려면, 먼저 현재 영상에서 무엇이 어디에 있는지 알아야 하기 때문이다. 논문은 surgeon view를 clutter 없이 증강하려면 장면 이해가 필수이며, 그 출발점이 pixel-wise segmentation이라고 본다. 동시에 의료 영상 분야에서는 고품질 라벨 데이터 부족이 매우 큰 병목이기 때문에, challenge 형식으로 데이터를 공개하고 성능을 비교하는 것이 실제 연구 진전을 위해 중요하다고 주장한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 새로운 segmentation 모델 하나를 제안하는 것이 아니라, 보다 현실적인 robotic surgical scene segmentation benchmark를 구성하고 이를 통해 현재 접근법들의 강점과 약점을 드러내는 데 있다. 2017 challenge가 주로 instrument segmentation에 집중했다면, 2018 버전은 anatomical objects와 additional medical devices를 추가하여 문제 난도를 한 단계 올렸다.

차별점은 크게 세 가지다. 첫째, ex-vivo tissue와 단순한 배경이 아니라 porcine surgery에서 획득한 실제에 가까운 장면을 사용했다. 둘째, instrument만이 아니라 anatomy와 non-biological objects를 함께 다뤘다. 셋째, annotation 설계 자체에서 의료 영상 특유의 모호성을 정면으로 다뤘다. 대표적으로 kidney surface 위를 fascia와 fat가 덮고 있는 경우를 별도 class인 `covered kidney`로 정의했는데, 이는 anatomy segmentation을 실제 수술적 맥락에 맞게 만들기 위한 설계라고 볼 수 있다.

또 하나의 중요한 메시지는, segmentation 성능이 class마다 크게 다르며 “쉽게 드러난 organ surface”와 “fat/fascia에 가려진 구조” 사이에 난도 차이가 매우 크다는 점이다. 논문은 이를 통해 단순 평균 성능만 보는 것이 아니라 어떤 class가 구조적으로 어려운지 해석해야 한다는 점을 보여준다.

## 3. 상세 방법 설명

이 논문의 방법론적 본체는 challenge 설계와 평가 프로토콜이며, 동시에 참가팀들의 대표적 접근을 비교해 보여준다.

전체 데이터셋은 총 19개 sequence로 구성되며, 15개는 training, 4개는 test에 사용되었다. 각 sequence는 porcine training procedure에서 기록되었고, significant camera motion 또는 tissue interaction이 있는 구간을 뽑은 뒤 1 Hz로 subsample하였다. 비슷한 프레임은 수동으로 제거하여 sequence당 300프레임을 만들었다. 각 프레임은 stereo pair이며 해상도는 $1280 \times 1024$이고, 카메라 calibration 정보도 제공되었다. 다만 annotation time을 줄이기 위해 stereo pair 중 left eye만 라벨링했다.

라벨 체계는 medical device class와 anatomical class로 나뉜다. medical device에는 da Vinci instrument의 parts, drop-in ultrasound probe, suturing needle, suturing thread, suction-irrigation device, surgical clips가 포함된다. anatomical class에는 kidney parenchyma, covered kidney, small intestine가 포함되며, 그 외 anatomy는 background로 묶였다.

annotation 측면에서 논문은 anatomy labeling이 instrument labeling보다 훨씬 어렵다고 분명히 말한다. instrument는 경계와 class 정의가 비교적 명확하지만, anatomy는 지방 조직과 결합조직에 가려지거나 시점에 따라 식별이 애매할 수 있다. 특히 `covered kidney`는 “중요한 장기 위를 덮고 있는 connective tissue/fat를 별도 의미 있는 class로 볼 것인가”라는 실용적 문제를 반영한 class다. 하지만 fascia가 당겨져 일시적으로 장기 표면에서 떨어진 경우처럼, 일관된 protocol을 정의하기 어려운 예외 상황이 발생한다. 또한 annotator가 단일 프레임만 봐서는 구조를 식별하기 어렵고, 시간적으로 이어진 영상 문맥이 필요하다는 점도 지적한다.

참가 방법들은 대부분 CNN 기반 semantic segmentation 모델이며, 당시 주류 backbone과 decoder를 적극 활용했다. U-Net 계열, DeepLab V3+, PSPNet, GCN, Pix2Pix 기반 접근이 등장한다. ImageNet pretrained encoder를 사용하는 경우가 많고, augmentation으로 brightness, contrast, rotation, scale, flip, crop, color jitter 등을 폭넓게 사용한다.

예를 들어 Digital Surgery 팀은 DeepLab V3+ 기반 모델을 사용했고, 원래 10개 class scoremap뿐 아니라 semantically related classes를 병합한 5개 class scoremap도 함께 예측했다. 이 팀은 다음 손실을 사용했다고 적고 있다.

$$
L = \sum_{i=0}^{N} \big[y_i \log(\hat y_i) + (1-y_i)\log(1-\hat y_i)\big]
\times \left[\frac{d_{\max}-\min_{b_j \in B} d(i,b_j)}{2d_{\max}} + 0.5\right]
$$

여기서 $B$는 boundary pixel들의 집합, $N$은 이미지 내 pixel 수, $d(a,b)$는 Euclidean distance, $d_{\max}$는 batch 내 boundary까지의 최대 거리이다. 직관적으로 보면 기본적인 pixel-wise classification loss에 boundary 관련 가중치를 곱해, 경계 근처와 멀리 떨어진 영역을 다르게 다루도록 만든 형태로 해석할 수 있다. 논문 원문은 이 식의 목적을 길게 해설하지는 않지만, boundary-aware weighting을 통해 segmentation 경계 품질을 개선하려는 의도로 읽힌다.

여러 팀이 loss를 조합했다는 점도 중요하다. weighted cross entropy, Dice loss, Jaccard loss, focal loss, adversarial loss, L1 loss 등이 사용되었다. 이는 class imbalance와 boundary precision 문제가 모두 중요했음을 보여준다. 예를 들어 IIT Madras는 robotic tools와 organs를 분리해 각각 다른 network를 학습했고, weighted cross entropy와 Dice loss를 함께 사용했으며, 후처리로 CRF를 적용했다. Yale의 MEDYI 팀은 modified U-Net with ResNet-101 backbone에 focal loss와 weighted cross entropy를 결합했고, 4개 모델 ensemble에 majority voting을 사용했다. JHU는 Pix2Pix 기반으로 segmentation을 수행하면서 class-weighted cross entropy, $L1$ loss, adversarial loss를 함께 사용했다.

또 다른 흥미로운 방향은 stereo 정보를 쓰는 접근이다. Norwegian University of Science and Technology 팀의 StreoScenNet은 left/right frame을 각각 다른 encoder에 넣고, 최종적으로 left frame용 segmentation mask를 예측했다. 이는 stereo input이 surgical scene understanding에 도움이 될 수 있다는 가정을 반영한다. 반면 대부분의 다른 방법은 단일 이미지 기반이다.

다만 모든 참가팀의 세부 내용이 충분히 공개된 것은 아니다. NUS 제출은 추가 details가 없다고 명시되어 있고, Team Banana 역시 팀 정보와 방법 설명이 제공되지 않았다. 따라서 이 둘의 구체적 architecture나 학습 전략은 논문만으로는 알 수 없다.

## 4. 실험 및 결과

평가는 mean IoU로 수행되었다. 한 class에 대한 IoU는 다음과 같이 정의된다.

$$
IoU = \frac{TP}{TP + FP + FN}
$$

여기서 $TP$는 true positive, $FP$는 false positive, $FN$은 false negative이다. mean IoU는 한 프레임에 실제로 등장한 class들에 대해서만 IoU를 평균낸다. 만약 특정 class set이 프레임에 아예 없으면 해당 프레임은 계산에서 제외한다. 이후 frame-level score를 dataset 단위로 평균하고, 전체 점수는 dataset 크기에 따라 가중 평균한다. 즉, 이 평가는 단순한 global pixel accuracy가 아니라 class별 overlap 품질을 보는 보다 엄격한 segmentation 지표다.

실험은 4개의 test dataset에서 수행되었다. 각 dataset은 서로 다른 시야, organ exposure 정도, instrument 종류, ultrasound probe 사용 여부 등을 가진다.

Test dataset 1은 zoomed-out exploratory sequence로 시작해 kidney parenchyma에 도달하고, 이후 tissue dissection과 suturing이 포함된다. 이 데이터에서는 kidney class에서 3개 팀이 0.9 이상의 IoU를 기록했다. 평균 overall score는 0.500이었다. class별 평균을 보면 parenchyma는 0.674, instrument shaft는 0.731로 비교적 높지만, needle은 0.003으로 매우 낮다. overall 상위권은 OTH Regensburg 0.691, IRCAD 0.688, UNC 0.663, NCT 2 0.658, Digital Surgery 0.636 순이다. 논문은 이 sequence가 exposed parenchyma를 많이 포함해 비교적 쉬운 편이라고 해석한다.

Test dataset 2는 처음에 kidney가 fascia와 perirenal fat에 가려져 있고, 이를 제거하는 과정이 담겨 있다. 평균 overall score는 0.478이다. 이 데이터에서 parenchyma 평균 IoU는 0.449인데 비해 covered kidney는 0.216으로 절반 이하이다. 논문은 qualitative result를 근거로 covered kidney가 자주 background로 오분류된다고 설명한다. overall 상위는 UNC 0.578, OTH Regensburg 0.575, NCT 2 0.555, Digital Surgery 0.549 정도다. instrument shaft 평균은 0.865로 여전히 높았지만 anatomy의 가려진 부분은 매우 어려웠다.

Test dataset 3은 parenchyma close-up 장면이며 언뜻 쉬워 보이지만 heavily covered surface 때문에 점수가 생각보다 낮았다고 논문은 해석한다. 평균 overall score는 0.658로 네 개 중 가장 높다. parenchyma 평균 0.698, instrument shaft 평균 0.820, clasper 0.568, wrist 0.555, intestine 0.394이다. 최고 overall은 OTH Regensburg 0.829, UNC 0.814, Digital Surgery 0.806, ODS.ai 0.799, IRCAD 0.790이다. anatomy가 충분히 노출된 경우에는 좋은 성능이 가능하지만, 여전히 intestine 같은 class는 안정적이지 않음을 보여준다.

Test dataset 4는 kidney가 fascia와 fat에 많이 가려져 있고, intestine가 많이 보이며, ultrasound probe가 등장한다. 평균 overall score는 0.279로 가장 낮다. parenchyma 평균은 0.091, covered kidney 평균은 0.232에 불과하다. 반면 instrument shaft는 0.534, wrist는 0.385, intestine는 0.470이다. 이 결과는 “가려진 kidney anatomy”가 challenge 전체에서 가장 어려운 축임을 분명히 보여준다. 최고 overall은 OTH Regensburg 0.390, UNC 0.373, Satoshi Kondo 0.368, Fan Voyage 0.366, NCT 2 0.362였다.

전체 평균 결과는 Table V에 정리되어 있다. overall score 기준 상위 방법은 OTH Regensburg 0.621, UNC 0.607, NCT 2 0.585, Digital Surgery 0.579, IRCAD 0.573, Fan Voyage 0.570 순이다. class별 평균을 보면 instrument shaft가 0.738로 가장 높고, clips는 0.536, parenchyma는 0.479, instrument clasper는 0.460, wrist는 0.457 수준이다. 반면 covered kidney는 0.224, thread는 0.154, US probe는 0.128, needle은 0.003으로 매우 낮다. 즉, 큰 기구 파트는 비교적 잘 분할되지만, 작고 얇거나 시각적으로 애매한 구조물은 거의 해결되지 않았다고 봐야 한다.

이 결과는 당시 강한 backbone과 pretrained encoder를 사용한 DeepLab/U-Net 류 모델이 주류였고, 그중에서도 OTH Regensburg가 전체적으로 가장 안정적이었다는 점을 보여준다. 특히 이 팀은 overall winner일 뿐 아니라 10개 class 중 6개 class에서 최고 점수를 기록했다. 다만 anatomy와 small thin object에서 절대 성능 자체는 여전히 낮아서, challenge가 아직 해결되지 않았음을 시사한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 실제 수술 장면에 가까운 robotic scene segmentation benchmark를 체계적으로 제시했다는 점이다. 2017 instrument-only setting에서 더 나아가 anatomy와 다양한 devices를 추가함으로써 실제 scene understanding 문제에 가까워졌다. 또한 단순히 데이터만 배포한 것이 아니라 class definition, annotation ambiguity, evaluation protocol, 참가 방법, dataset별 결과 해석까지 함께 제시해 후속 연구가 참고할 수 있는 기준점을 만들었다.

또 다른 강점은 결과 해석이 구체적이라는 점이다. 논문은 평균 점수만 보고 끝내지 않고, 왜 어떤 dataset이 어려운지, 왜 exposed parenchyma는 쉽고 covered kidney는 어려운지, 어떤 class가 background와 혼동되는지 설명한다. 이는 benchmark 논문으로서 매우 중요한 부분이다. 또한 여러 참가팀의 접근을 비교함으로써 당시 surgical segmentation 커뮤니티에서 어떤 설계가 실무적으로 쓰였는지도 보여준다.

한계도 분명하다. 첫째, 데이터가 여전히 porcine surgery 기반이다. 논문 스스로도 human tissue에 비해 porcine tissue는 fatty tissue occlusion이 적어 훨씬 단순하다고 말한다. 따라서 실제 human surgery generalization은 이 결과만으로 판단할 수 없다. 둘째, anatomy annotation protocol이 본질적으로 어렵고 class 정의가 모호하다. `covered kidney` 같은 class는 실용적으로는 의미가 있지만, 시각적 경계가 애매한 경우가 많아 annotator consistency 문제가 생긴다. 셋째, stereo pair를 제공하면서도 left eye만 annotation했다는 점은 효율성 측면에서는 합리적이지만, 양안 정보를 fully supervised하게 활용하는 연구에는 제약이 있다.

정량 결과 측면의 한계도 크다. overall 최고 성능이 0.621에 그쳤고, 작은 object인 needle과 US probe, thread의 평균 성능은 매우 낮다. 이는 모델 구조의 문제일 수도 있지만 데이터 수, class imbalance, annotation difficulty가 복합적으로 작용한 결과로 보인다. 다만 논문은 각 방법의 training budget, hardware, inference setting을 완전히 통일해 비교한 것은 아니므로, leaderboard를 절대적인 architecture ranking으로 과도하게 해석하면 안 된다.

비판적으로 보면, 참가 방법 소개는 유용하지만 어떤 요소가 성능 차이를 만들었는지에 대한 systematic ablation은 없다. 그러나 이것은 benchmark/challenge report의 성격상 자연스러운 제한으로 볼 수 있다. 또한 test dataset별 상세 결과는 충분히 제공되지만, statistical significance 분석이나 error taxonomy가 더 있었다면 후속 연구 설계에 더 도움이 되었을 것이다.

## 6. 결론

이 논문은 robotic surgical scene segmentation을 위한 2018 EndoVis challenge를 정리하면서, 보다 현실적인 수술 장면 이해를 위한 공개 benchmark와 당시 state of the art 성능 수준을 제시했다. 핵심 기여는 anatomy와 device를 함께 포함하는 dataset 구축, annotation의 실제적 어려움에 대한 정리, mean IoU 기반 평가 프로토콜 제공, 그리고 다양한 참가 방법의 비교 분석에 있다.

결과적으로 이 연구는 “instrument segmentation은 어느 정도 가능하지만, anatomy understanding과 small object segmentation은 아직 어렵다”는 사실을 명확히 보여준다. 특히 covered kidney, thread, needle, US probe처럼 가려지거나 작은 대상은 성능이 매우 낮아 향후 연구 여지가 크다. 따라서 이 challenge는 수술 장면의 higher-level understanding, image-guided surgery, context-aware AR overlay 같은 실제 응용으로 가기 위한 중요한 중간 단계로 볼 수 있다. 다만 논문이 제공한 근거만 놓고 보면, human surgical scene에 대한 직접적 일반화는 아직 입증되지 않았으며, 더 다양한 데이터와 더 정교한 class 정의, temporal context 활용, 작은 구조물에 강한 segmentation 기법이 앞으로 필요하다.
