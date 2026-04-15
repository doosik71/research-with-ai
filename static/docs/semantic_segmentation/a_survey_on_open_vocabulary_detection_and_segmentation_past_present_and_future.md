# A Survey on Open-Vocabulary Detection and Segmentation: Past, Present, and Future

- **저자**: Chaoyang Zhu, Long Chen
- **발표연도**: 2024
- **arXiv**: https://arxiv.org/abs/2307.09220

## 1. 논문 개요

이 논문은 Open-Vocabulary Detection (OVD)와 Open-Vocabulary Segmentation (OVS) 연구 전반을 체계적으로 정리한 survey이다. 저자들은 기존의 object detection 및 segmentation 모델이 대부분 닫힌 집합(closed vocabulary) 범주에서만 동작한다는 점을 문제로 삼는다. 예를 들어 Pascal VOC는 20개, COCO는 80개, LVIS도 1,203개 정도의 범주만 주석이 달려 있는데, 실제 세계의 시각 개념은 훨씬 더 많고 계속 확장된다. 따라서 기존 fully-supervised detector/segmentor는 미리 정의되지 않은 새로운 범주를 인식하는 데 구조적 한계가 있다.

이 논문이 다루는 핵심 연구 문제는 다음과 같다. 어떻게 하면 bounding box나 mask 주석이 없는 novel category까지 탐지하거나 분할할 수 있는가, 그리고 이 문제를 기존 zero-shot 계열 연구와 최근의 vision-language model 기반 open-vocabulary 계열 연구를 포함해 하나의 공통된 틀로 설명할 수 있는가이다. 저자들은 단순히 개별 방법을 나열하지 않고, 약한 감독 신호(weak supervision signal)의 허용 여부와 활용 방식이 방법론을 가르는 가장 중요한 축이라고 본다.

이 문제는 실제 적용 측면에서 중요성이 크다. 자율주행, 로보틱스, 장면 이해 같은 응용에서는 학습 시 정의되지 않았던 객체도 다뤄야 한다. 따라서 “사전에 정해진 클래스만 맞히는 모델”에서 “자연어 이름이나 설명으로 열린 범주의 객체를 다룰 수 있는 모델”로의 전환은 장면 이해 연구의 실용성과 확장성을 크게 높인다. 이 논문은 바로 그 전환의 흐름을 detection, semantic segmentation, instance segmentation, panoptic segmentation, 3D scene understanding, video understanding까지 넓게 정리한다.

## 2. 핵심 아이디어

이 survey의 가장 중요한 기여는 taxonomy의 제안이다. 저자들은 기존 연구를 task별로만 나누지 않고, weak supervision signal을 사용할 수 있는지 여부와 어떤 종류의 신호를 쓰는지에 따라 큰 틀에서 정리한다. 이 관점은 zero-shot과 open-vocabulary를 분리하면서도, 동시에 detection, segmentation, 3D, video 전반에 공통으로 적용할 수 있다는 점에서 설득력이 있다.

논문이 제시한 큰 분류는 여섯 가지이다. zero-shot setting에서는 weak supervision 없이 동작하는 두 계열, 즉 Visual-Semantic Space Mapping과 Novel Visual Feature Synthesis가 중심이다. 반면 open-vocabulary setting에서는 weak supervision이나 pretrained Vision-Language Models(VLMs)를 활용하는 네 계열, 즉 Region-Aware Training, Pseudo-Labeling, Knowledge Distillation, Transfer Learning이 중심이다. 저자들은 이 여섯 범주가 객체 탐지와 여러 분할 문제를 통합적으로 설명할 수 있다고 주장한다.

또 하나의 핵심 아이디어는 zero-shot과 open-vocabulary를 완전히 분리된 세계로 보지 않는다는 점이다. 저자들은 두 설정 모두 closed-vocabulary 제약을 넘어서려는 시도라는 공통 목적을 가진다고 본다. 다만 zero-shot은 학습 시 unseen object를 전혀 보지 못하거나 weak supervision도 허용하지 않는 반면, open-vocabulary는 image-text pair나 CLIP 같은 대규모 VLM을 활용할 수 있다. 이 차이가 성능 격차를 결정하는 가장 큰 원인이라고 설명한다.

기존 접근과의 차별점도 분명하다. 초기 zero-shot 방법은 Word2Vec, GloVe, BERT 같은 semantic embedding을 classifier 대용으로 사용했지만, 이 임베딩은 텍스트 말뭉치만으로 학습되어 시각 정보와의 alignment가 약하다. 반면 open-vocabulary 방법은 CLIP 같은 VLM의 text encoder가 만든 text embedding을 frozen classifier로 사용하고, 이 임베딩은 image-text contrastive pretraining을 통해 시각 modality와 더 잘 정렬되어 있다. 저자들은 바로 이 점이 ZSD/ZSS에서 OVD/OVS로 넘어오며 큰 성능 향상이 나타난 이유라고 본다.

## 3. 상세 방법 설명

논문은 먼저 문제 정의를 분명히 한다. 전체 클래스 집합을 $C$라 할 때, 이를 base class 집합 $C_B$와 novel class 집합 $C_N$으로 나눈다. 두 집합은 서로 겹치지 않고, $C = C_B \cup C_N$이다. 학습에서는 주석이 있는 클래스가 $C_{train} = C_B$이고, 테스트에서는 $C_{test} = C_B \cup C_N$이다. 즉 모델은 학습 중 base class annotation만 보지만, 테스트에서는 base와 novel을 함께 구분해야 한다.

또한 open-vocabulary 방법들은 closed-set detector와 달리 class-agnostic localization branch를 자주 사용한다. 예를 들어 bounding box regression의 마지막 출력은 클래스별 박스가 아니라 단일 박스 좌표 $[x_1, y_1, x_2, y_2]$ 또는 $[x, y, w, h]$로 예측한다. 이는 novel class에 대해서도 localization이 되도록 만들기 위한 설계이다.

### Visual-Semantic Space Mapping

이 계열은 visual feature와 semantic embedding 사이의 대응을 학습하는 방식이다. 가장 기본적인 형태는 visual feature를 semantic space로 projection한 뒤, 고정된 semantic embedding과 비교해 분류하는 것이다. 반대로 semantic을 visual space로 보내는 방식이나, 둘의 joint space를 학습하는 방식도 소개된다.

핵심 가정은 semantic space가 클래스 간 관계를 어느 정도 반영하므로, visual feature를 여기에 잘 매핑하면 unseen class도 분류 가능하다는 것이다. 그러나 이 방식은 여러 어려움을 갖는다. 첫째, semantic embedding 자체가 시각 정보와 직접 정렬되어 있지 않아 noisy하다. 둘째, unseen annotation이 없으므로 prediction confidence가 seen class 쪽으로 치우친다. 셋째, background와 unseen object를 구분하기 어렵다.

논문은 joint mapping의 필요성도 설명한다. 어떤 클래스 쌍은 semantic space에서는 가깝지만 visual space에서는 멀리 떨어져 있을 수 있고, 반대 경우도 가능하다. 따라서 visual과 semantic 양쪽 구조를 함께 활용하는 접근이 등장했다. 또한 semantic-to-visual mapping은 visual-to-semantic mapping이 feature variance를 줄여 hubness problem을 일으킬 수 있다는 비판에 대한 대안으로 제시된다.

### Novel Visual Feature Synthesis

이 계열은 unseen visual feature를 직접 생성해 문제를 “가짜이지만 supervised에 가까운 문제”로 바꾸는 접근이다. 논문은 이를 4단계 파이프라인으로 설명한다.

첫째, base class annotation으로 기본 detector 또는 segmentor를 fully supervised하게 학습한다. 둘째, seen class의 semantic embedding $w \in W_s$와 실제 seen visual feature $f_s \in F_s$를 이용해 생성기 $G: W \times Z \mapsto \tilde{F}$를 학습한다. 셋째, unseen semantic embedding과 random noise $z \sim \mathcal{N}(0,1)$를 입력으로 넣어 fake unseen visual feature를 합성한다. 넷째, real seen feature와 fake unseen feature를 함께 사용해 classifier를 다시 학습한 뒤, 이 classifier를 원래 모델에 삽입한다.

이 방법의 장점은 unseen class를 학습 시점에 classifier가 직접 보게 된다는 점이다. 따라서 novel과 background의 혼동이나 seen bias를 줄일 수 있다. 논문은 DELO, GTNet, RRFS 같은 방법을 예로 들며, objectness consistency, category consistency, semantic consistency, IoU-aware synthesis, intra-class/inter-class regularization 같은 다양한 보강 기법을 설명한다. 다만 생성 품질과 실제 region context 간 격차가 중요한 문제로 남아 있다고 정리한다.

### Region-Aware Training

open-vocabulary setting의 첫 번째 핵심 방법론은 image-text pair를 사용해 region과 word를 약하게 정렬하는 방식이다. 이 방법은 region-word의 정확한 1:1 대응은 모르지만, 같은 이미지-문장 쌍 안의 proposal과 noun을 서로 가깝게 만드는 contrastive 또는 grounding loss를 쓴다.

논문은 대표적인 양방향 손실을 다음과 같이 제시한다.

$$
L_{T \to I} = - \log \frac{\exp(\mathrm{sim}(I,T))}{\sum_{I' \in B}\exp(\mathrm{sim}(I',T))}
$$

$$
L_{I \to T} = - \log \frac{\exp(\mathrm{sim}(I,T))}{\sum_{T' \in B}\exp(\mathrm{sim}(I,T'))}
$$

여기서 $B$는 batch이고, image-caption similarity는 다음처럼 정의된다.

$$
\mathrm{sim}(I,T) = \frac{1}{N_T}\sum_{i=1}^{N_T}\sum_{j=1}^{N_I}\alpha_{i,j}\langle e_i^T \cdot e_j^I \rangle
$$

가중치 $\alpha_{i,j}$는 proposal별 soft assignment로 계산된다.

$$
\alpha_{i,j} = \frac{\exp \langle e_i^T \cdot e_j^I \rangle}{\sum_{j'=1}^{N_I}\langle e_i^T \cdot e_{j'}^I \rangle}
$$

직관적으로 말하면, 캡션의 noun과 이미지의 여러 proposal을 모두 비교하되, 더 잘 맞는 proposal에 더 큰 가중치를 주는 방식이다. 이 방법은 exact correspondence 없이도 novel vocabulary가 포함된 caption을 이용해 base-to-novel generalization을 높인다. 다만 “bag-of-regions to bag-of-words” 수준의 느슨한 정렬에 그칠 수 있다는 한계가 있다.

### Pseudo-Labeling

이 계열은 region-aware training보다 더 강한 형태의 정렬을 한다. teacher model 또는 모델 자기 자신이 pseudo label을 만들어서 region-word 또는 region-caption의 정확한 대응을 부여한다. 논문은 이를 soft alignment와 대비되는 hard alignment로 설명한다.

이 방법의 장점은 대응 관계가 명시적이어서 학습 신호가 더 직접적이라는 점이다. 예를 들어 CLIP이나 Grounding 계열 teacher를 이용해 novel class의 pseudo bounding box를 만들고, 이를 base annotation과 합쳐 detector를 학습할 수 있다. 그러나 중요한 약점도 있다. 대개 학습 단계에서 novel class 이름을 미리 알아야 하므로, 엄밀한 의미의 완전한 open-vocabulary setting을 일부 훼손할 수 있다.

논문은 pseudo region-word pair, pseudo region-caption pair, pseudo caption으로 나누어 접근들을 정리한다. 예를 들어 Detic은 이미지 레벨 label을 가장 큰 proposal에 할당하는 단순한 multiple instance learning 스타일 접근을 취하고, GLIP과 Grounding DINO는 grounding 데이터를 활용해 더 강한 teacher를 만든다. PB-OVD는 GradCAM 기반으로 pseudo box를 만들고, ProxyDet은 base class의 convex hull 안에 novel class가 있다는 관찰을 이용해 proxy novel class를 합성한다.

### Knowledge Distillation

이 계열은 teacher VLM의 visual embedding을 student detector/segmentor가 모방하도록 학습시키는 방식이다. 주로 CLIP image encoder가 teacher가 되며, detector의 region embedding이 teacher의 region embedding에 가까워지도록 distillation loss를 둔다.

핵심 직관은 RoI가 novel object를 포함하고 있을 가능성이 있으므로, 해당 RoI의 teacher embedding을 모방하면 detector도 CLIP의 잘 정렬된 visual-semantic space에 가까워진다는 것이다. 논문은 single-region level, bag-of-regions level, image-region level 등 distillation granularity의 차이를 설명한다. 예를 들어 ViLD는 base와 novel proposal 모두에 대해 CLIP embedding을 모방하게 하고, BARON은 여러 region을 묶은 bag-of-regions를 다룬다. OADP는 whole-image와 patch까지 포함한 pyramid distillation을 사용한다.

distillation objective도 단순한 $L_1$ loss에만 머물지 않는다. 어떤 방법은 teacher와 student embedding 사이의 관계 구조를 맞추려 하고, 어떤 방법은 같은 객체는 가깝고 다른 객체는 멀어지게 하는 contrastive 형태를 쓴다. DetPro는 continuous prompt를 학습해 novel/background proposal이 base class로 오인되지 않도록 돕는다.

### Transfer Learning

이 계열은 detector backbone과 VLM image encoder를 따로 두지 않고, VLM image encoder 자체를 downstream detection/segmentation에 옮겨 쓰는 접근이다. 저자들은 이를 네 가지로 정리한다. frozen VLM image encoder를 feature extractor로 쓰는 경우, VLM image encoder 전체를 fine-tuning하는 경우, visual prompt를 학습하는 경우, adapter를 붙이는 경우이다.

예를 들어 OWL-ViT는 CLIP image encoder 위에 detection head를 붙여 end-to-end fine-tuning한다. F-VLM은 frozen CLIP image encoder를 그대로 쓰고 detection head만 학습한다. segmentation에서는 LSeg가 CLIP text embedding으로 classifier를 대체하는 단순한 출발점을 제시하고, OVSeg는 masked image crop으로 fine-tuning하여 natural image와 masked crop 사이의 domain gap을 줄인다. SAN은 frozen CLIP에 lightweight adapter를 붙여 효율적으로 적응시킨다.

이 계열의 장점은 large VLM의 사전학습 지식을 직접 활용할 수 있다는 점이다. 하지만 논문은 pretraining 단계와 downstream dense prediction 사이에 해상도, 문맥, crop 형태, task 통계 차이가 크다는 점을 반복해서 지적한다. 이 때문에 full fine-tuning은 catastrophic forgetting을 유발할 수 있고, prompt tuning이나 adapter가 중요한 역할을 한다고 설명한다.

## 4. 실험 및 결과

이 논문은 survey이므로 새로운 실험을 제안하기보다, 여러 task의 대표 benchmark와 성능 표를 체계적으로 모은다. Appendix의 Table 2에서 task별 데이터셋과 metric을 정리한다. ZSD는 Pascal VOC, COCO, ILSVRC-2017 Detection, Visual Genome을 사용하고, ZSS는 Pascal VOC와 Pascal Context를 사용한다. OVD는 COCO, LVIS, Objects365, OpenImages를 주로 다루며, OVSS는 Pascal VOC, COCO Stuff, ADE20K, Pascal Context 등을 포함한다. OVIS, OVPS, OV3D, OV3IS, OVVU에 대해서도 각각 적절한 benchmark를 제시한다.

평가 지표는 task별로 다르다. detection과 instance segmentation은 주로 $AP_{50}$, $AP$, 혹은 base/novel을 나눈 $AP_B$, $AP_N$을 사용한다. semantic segmentation은 $mIoU_B$, $mIoU_N$, 그리고 이 둘의 harmonic mean인 $hIoU$를 사용한다. 논문은 다음 식으로 $hIoU$를 제시한다.

$$
hIoU = \frac{2 \cdot mIoU_B \cdot mIoU_N}{mIoU_B + mIoU_N}
$$

이 지표는 novel 성능과 base 성능을 동시에 잘 유지하는지를 보는 데 적합하다. panoptic segmentation은 PQ, SQ, RQ를 사용하고, 3D/video 쪽은 이미지 도메인의 대응 metric을 확장해 사용한다.

정량 결과의 큰 흐름은 명확하다. zero-shot 계열보다 open-vocabulary 계열이 전반적으로 더 강하다. 이는 weak supervision과 VLM 덕분에 text와 image 사이 정렬 품질이 훨씬 높기 때문이다. 예를 들어 OVD on COCO에서 초기 OVR-CNN은 novel $AP_{50}$이 22.8인데, 이후 DITO는 38.6, BARON은 42.7, LP-OVOD는 40.5까지 도달한다. 같은 맥락에서 LVIS에서도 초기 방법보다 later method가 rare class 성능을 꾸준히 개선한다.

semantic segmentation에서도 transfer learning과 CLIP 적응 계열의 성능 향상이 눈에 띈다. 예를 들어 generalized setting에서 Pascal VOC 기준 ZegFormer는 $mIoU(N/B/HM)=63.6/86.4/73.3$이고, 이후 MAFT는 $81.8/91.4/86.3$, TagCLIP은 $85.2/93.5/89.2$를 기록한다. 이는 단순히 text embedding만 쓰는 수준을 넘어, mask-aware adaptation이나 prompt learning이 실제로 novel segmentation 성능을 끌어올렸음을 보여준다.

3D와 video는 상대적으로 덜 성숙한 분야로 제시된다. 예를 들어 open-vocabulary 3D detection에서는 OV-3DET, FM-OV3D, CoDA, L3Det 등이 소개되지만, 2D에 비해 데이터 부족과 modality gap 문제로 성능과 방법 다양성이 제한적이다. video에서도 OV2Seg, OpenVIS, BriVIS 등이 등장하지만 benchmark와 방법론 모두 아직 발전 중이라는 인상이 강하다.

중요한 점은 이 논문이 “어떤 방법이 무조건 최고”라고 단정하지 않는다는 것이다. 결과 표는 매우 많지만, backbone, detector architecture, pretraining data, ensemble 유무, validation split 차이 등이 섞여 있어서 완전히 공정한 직접 비교는 어렵다고 저자들이 직접 밝힌다. 따라서 이 survey의 실험 파트는 절대 순위표라기보다, 방법론의 발전 경향과 비교 기준을 보여주는 역할에 가깝다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 정리 방식의 명확성이다. 단순히 task별 문헌조사가 아니라, weak supervision의 사용 여부와 방식이라는 기준으로 zero-shot과 open-vocabulary를 하나의 큰 지형도로 묶는다. 이 분류는 detection, semantic/instance/panoptic segmentation, 3D, video까지 비교적 일관되게 적용된다. survey 논문으로서 구조적 기여가 크다.

또 다른 강점은 배경 설명의 충실함이다. canonical closed-set detector/segmentor, CLIP/ALIGN/DINO/MAE/diffusion model, PEFT까지 먼저 정리하고 들어가기 때문에, OVD/OVS 문헌이 왜 그런 방향으로 발전했는지 맥락을 이해하기 쉽다. 또한 단순한 장점 나열이 아니라 base bias, background confusion, region-word misalignment, VLM adaptation gap, inference speed, evaluation metric 문제처럼 실제 연구 난제를 분명히 드러낸다.

실험 정리도 유용하다. appendix에 task별 benchmark, dataset split, metric, 대표 성능표를 폭넓게 모아 두어 연구 입문자나 후속 연구자가 기준점을 잡기에 좋다. 특히 generalized evaluation과 cross-dataset transfer evaluation 같은 설정 차이를 명시한 점이 좋다.

한계도 있다. 우선 survey 특성상 개별 방법의 내부 수식이나 구현 세부를 모두 깊게 파고들지는 않는다. 예를 들어 어떤 손실이 실제로 어떻게 optimization stability에 영향을 주는지, 어떤 설계가 특정 benchmark에서 특히 잘 먹히는지 같은 미세한 분석은 제한적이다. 물론 survey의 범위를 고려하면 자연스러운 한계다.

또한 성능표는 방대하지만 backbone, pretraining corpus, ensemble 여부, prompt 종류 등이 통제되지 않은 비교가 많다. 저자들도 이 점을 인정한다. 따라서 표만 보고 방법론 자체의 우열을 단정하기는 어렵다. 이 논문은 이를 정직하게 밝히고 있으나, 독자는 결과를 해석할 때 주의가 필요하다.

비판적으로 보면, taxonomy가 매우 유용하지만 모든 방법이 완전히 배타적인 범주에 깔끔히 들어가는 것은 아니다. 예를 들어 어떤 방법은 transfer learning과 distillation 성격을 동시에 띠거나, pseudo-labeling과 region-aware training을 함께 섞기도 한다. 논문도 이를 어느 정도 암묵적으로 인정하지만, 큰 분류를 위해 경계가 다소 단순화된 면은 있다.

또 하나의 중요한 미해결 질문은 “진정한 open-vocabulary”의 정의 문제다. 논문은 pseudo-labeling 계열이 학습 중 novel class 이름을 미리 알아야 하는 경우가 많아 open-vocabulary setting을 일부 깨뜨린다고 명확히 지적한다. 이는 현재 문헌 전반의 구조적 한계이며, survey가 잘 짚은 부분이다.

## 6. 결론

이 논문은 OVD와 OVS를 중심으로 zero-shot에서 open-vocabulary로 이어지는 연구 흐름을 폭넓고 체계적으로 정리한 survey이다. 가장 중요한 기여는 weak supervision signal의 허용 여부와 활용 방식에 기반한 통합 taxonomy를 제시하고, 이를 detection, segmentation, 3D, video까지 확장해 설명한 점이다. 또한 large VLM, weak supervision, pseudo-labeling, distillation, transfer learning이 왜 핵심 도구가 되었는지를 분명한 문맥 속에서 설명한다.

실제 적용 측면에서도 이 연구 흐름은 중요하다. 앞으로 장면 이해 시스템은 미리 정의된 소수 범주만 다루는 모델에서 벗어나야 하며, 언어를 매개로 더 넓은 visual concept를 처리해야 한다. 이 논문은 그 전환의 현재 위치를 잘 보여준다. 동시에 real-time OVD/OVS, 2D-3D 통합, multimodal LLM 기반 perception, foundation model 결합, 평가 지표 재정의 같은 향후 연구 방향도 구체적으로 제시한다. 따라서 이 survey는 단순한 문헌 정리를 넘어, open-vocabulary scene understanding 연구의 로드맵 역할을 하는 논문으로 볼 수 있다.
