# MATIS: Masked-Attention Transformers for Surgical Instrument Segmentation

- **저자**: Nicol as Ayobi, Alejandra Perez-Rondon, Santiago Rodriguez, Pablo Arbelaez
- **발표연도**: 2024
- **arXiv**: https://arxiv.org/abs/2303.09514

## 1. 논문 개요

이 논문은 로봇 보조 수술 장면에서 수술 도구를 더 정확하고 일관되게 분할하기 위한 방법인 MATIS를 제안한다. 문제 설정은 단순한 semantic segmentation이 아니라, 서로 다른 instrument subtype을 구분하면서 각 개별 도구 인스턴스까지 구별해야 하는 surgical instrument segmentation이다. 저자들은 기존 방법들이 주로 픽셀 단위 분류에 의존해 왔기 때문에, 도구가 여러 개 등장하는 상황에서 공간적 일관성이 약하고 instance-level 구조를 충분히 반영하지 못한다고 본다.

논문이 다루는 핵심 연구 문제는 두 가지다. 첫째, 서로 시각적으로 매우 비슷한 수술 도구들을 정확히 구분하고 분할하는 것이다. 둘째, 비디오에서 시간 정보를 활용해 프레임 간 예측 일관성을 높이는 것이다. 수술 비디오는 조명 변화, 가림(occlusion), 도구 간 시각적 유사성, 데이터 불균형 문제를 동시에 가지므로 이 문제는 실제로 쉽지 않다.

이 문제의 중요성은 분명하다. 수술 도구 분할은 instrument tracking, pose estimation, surgical phase estimation 같은 상위 과제의 기반이 된다. 따라서 분할의 정확도와 시간적 안정성이 개선되면, 더 정교한 computer-assisted intervention 시스템으로 이어질 가능성이 크다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 수술 도구 분할을 “모든 픽셀을 직접 클래스 분류하는 문제”가 아니라, “고정 개수의 region proposal을 만들고 그것을 분류하는 set prediction 문제”로 다루는 것이다. 이를 위해 저자들은 Mask2Former 기반의 masked attention 구조를 사용해 각 프레임에서 도구 인스턴스에 해당하는 마스크 후보들을 생성하고, 이후 이 후보들을 instrument type으로 분류한다.

여기에 더해, 저자들은 비디오 수준의 장기 temporal information을 활용하는 temporal consistency module을 추가한다. 이 모듈은 단일 프레임만 보고 애매했던 분류를 앞뒤 프레임 문맥을 통해 보정하는 역할을 한다. 예를 들어 어떤 도구가 한 프레임에서는 Prograsp Forceps처럼 보이지만, 인접 프레임들의 움직임과 문맥을 함께 보면 Large Needle Driver일 가능성이 높다는 식의 보정이 가능해진다.

기존 접근과의 차별점은 크게 세 가지다. 첫째, CNN 기반 per-pixel 분류 대신 fully transformer-based 구조를 사용한다. 둘째, deformable attention과 masked attention을 결합한 최신 segmentation architecture를 surgical instrument segmentation에 본격적으로 적용한다. 셋째, optical flow 중심의 단기 temporal prior가 아니라, video transformer를 활용한 장기 비디오 문맥을 mask classification 단계에 통합한다.

## 3. 상세 방법 설명

MATIS는 2-stage 구조다. 첫 번째 단계는 각 프레임에서 instrument region proposal과 segment embedding을 생성하는 단계이고, 두 번째 단계는 비디오 문맥을 이용해 그 proposal의 클래스를 더 정확히 분류하는 단계다.

첫 번째 단계인 masked attention baseline은 Mask2Former를 기반으로 한다. Mask2Former는 DETR 계열의 set prediction formulation을 따르며, 고정된 개수 $N$개의 learnable query를 사용해 $N$개의 예측 결과를 만든다. 각 결과는 클래스 확률과 binary mask의 쌍이다. 이 논문에서는 $N = 100$을 사용한다. 즉, 한 프레임에서 최대 100개의 region candidate를 예측하고, 이들 각각에 대해 어떤 instrument class인지와 어느 픽셀 영역을 차지하는지 동시에 예측한다.

이 baseline의 중요한 특징은 multi-scale deformable attention pixel decoder와 masked attention mechanism을 사용한다는 점이다. 논문은 Mask2Former의 세부 내부 구조를 반복 설명하지는 않고 기존 논문을 참조하라고 하지만, 여기서 중요한 해석은 다음과 같다. deformable attention은 영상의 여러 해상도(feature scale)에서 필요한 위치만 선택적으로 참조할 수 있게 해 주므로, 복잡한 수술 장면에서 도구의 미세한 형태를 더 유연하게 파악할 수 있다. masked attention은 이미 형성된 마스크 가설을 바탕으로 관련 영역에 집중하면서 region-level 분할을 수행하게 해 준다. 저자들은 이 조합이 기존 self-attention 또는 단순 cross-attention보다 더 localized하고 flexible한 visual understanding을 제공한다고 해석한다.

논문은 수술 도구 분할에 맞게 Mask2Former의 segmentation module을 데이터셋 클래스 수에 맞춰 조정했다. 그리고 추론 시에는 일반적인 thresholding만 쓰지 않고 class-specific inference strategy를 설계한다. 구체적으로는 각 도구 클래스별로 한 프레임에 등장 가능한 인스턴스 수를 prior로 사용해 top-$k$ region을 고른다. 예를 들어 항상 하나만 등장하는 도구는 $k=1$, 여러 개 나올 수 있는 도구는 $k=2$를 사용한다. 또한 클래스별 confidence score 분포가 데이터 불균형 때문에 크게 다르므로, 모든 클래스에 공통 threshold를 쓰지 않고 class-specific score threshold를 둔다. 이 설계는 적은 학습 샘플을 가진 클래스가 낮은 confidence를 내는 문제를 보완하기 위한 것이다.

두 번째 단계인 temporal consistency module은 TAPIR와 유사한 video analysis 구조를 사용한다. backbone은 Multi-Scale Vision Transformer(MViT)이며, 중심 프레임(keyframe)을 기준으로 한 시간 창(time window)을 입력받아 spatio-temporal feature를 계산한다. 이 비디오 feature는 가운데 프레임의 복잡한 temporal context를 담는다. 여기서 핵심은 이 전역적 시간 문맥 정보와 1단계에서 나온 per-segment embedding을 결합한다는 점이다.

논문 설명에 따르면, 시간 특징은 먼저 시간 축으로 pooling된다. 그다음 이 pooled time feature에 MLP를 적용한다. 이를 논문에서는 단순한 feature 결합보다 더 잘 작동하는 수정으로 제시한다. 이후 이 결과를 segment embedding의 선형 변환 결과와 concatenate한 뒤, 최종 linear classifier로 각 region의 instrument class를 예측한다. 즉, 최종 분류는 “이 region 자체가 가진 국소 정보”와 “이 프레임이 포함된 비디오 문맥 정보”를 함께 본 결과다.

또한 저자들은 추가 supervision을 넣는다. pooled time feature에 또 다른 MLP를 적용해 middle frame에서 각 instrument가 존재하는지를 multi-label classification으로 예측한다. 이 보조 과제는 각 프레임에 어떤 도구 종류들이 등장하는지 맞히는 문제이며, binary cross-entropy loss로 학습된다. 논문은 이 보조 손실이 time feature가 더 풍부한 instrument-type 정보를 담도록 유도한다고 설명한다.

손실 함수의 전체 수식은 논문 본문에 명시적으로 자세히 써 있지 않다. 다만 서술에 따르면, temporal consistency module에서는 원래의 mask classification loss에 instrument presence supervision을 위한 binary cross-entropy loss를 추가한다. 따라서 개념적으로는 다음과 같이 이해할 수 있다.

$$
\mathcal{L}_{total} = \mathcal{L}_{mask\_classification} + \lambda \mathcal{L}_{presence}
$$

여기서 $\mathcal{L}_{presence}$는 middle frame에서 각 도구 클래스의 존재 여부를 예측하는 multi-label BCE loss이다. 하지만 논문 추출 텍스트에는 $\lambda$ 값이나 정확한 결합 방식은 명시되어 있지 않으므로, 그 부분은 추측할 수 없다.

학습 절차를 보면, 1단계 Mask2Former baseline은 MS-COCO instance segmentation으로 pretrained된 공식 구현을 사용한다. backbone은 Swin Small이며, 100 epoch 동안 4개의 NVIDIA Quadro RTX 8000 GPU에서 batch size 24로 ADAMW optimizer를 사용해 학습한다. 2단계 temporal consistency module은 Kinetics-400으로 pretrained된 TAPIR 공식 구현을 사용하며, window size 8, stride 1, batch size 12로 20 epoch 학습한다. 중요한 점은 temporal module 학습과 검증 모두에서 baseline의 inference strategy로 선택된 region만 사용한다는 것이다.

## 4. 실험 및 결과

실험은 Endovis 2017과 Endovis 2018이라는 두 개의 공개 benchmark에서 수행된다. Endovis 2017은 기존 연구와의 공정 비교를 위해 4-fold cross-validation을 따르고, Endovis 2018은 ISINet에서 제공한 추가 instance annotation과 predefined split을 사용한다. 평가지표는 mIoU, IoU, mcIoU 세 가지다. Endovis 2017에서는 fold 간 standard deviation도 함께 보고한다.

먼저 단일 프레임 기반 baseline만 평가한 결과가 MATIS Frame이다. 이 모델은 Endovis 2017과 Endovis 2018 모두에서 이전 state-of-the-art를 전반적으로 넘어선다. Endovis 2017에서는 MATIS Frame이 mIoU $68.79 \pm 2.98$, IoU $62.74$, mcIoU $37.30$을 기록했고, temporal module까지 포함한 MATIS Full은 mIoU $71.36 \pm 3.46$, IoU $66.28$, mcIoU $41.09$를 기록했다. Endovis 2018에서는 MATIS Frame이 mIoU $82.37$, IoU $77.01$, mcIoU $48.65$, MATIS Full이 mIoU $84.26$, IoU $79.12$, mcIoU $54.04$를 달성했다.

이 수치는 비교 대상들과의 차이를 보면 더 의미가 분명해진다. 예를 들어 Endovis 2018에서 TraSeTR는 mIoU 76.20 수준인데, MATIS Frame은 82.37, MATIS Full은 84.26까지 올라간다. 즉, transformer를 썼다는 사실만 중요한 것이 아니라, 어떤 transformer segmentation design을 택했는지가 성능 차이에 직접 연결된다고 볼 수 있다. 저자들은 이를 통해 multi-scale deformable attention과 masked attention이 instrument visual characteristic을 더 localized하고 flexible하게 파악한다고 해석한다.

클래스별 결과를 보면 흥미로운 패턴이 있다. 학습 샘플이 많은 클래스에서는 MATIS가 TraSeTR보다 매우 크게 앞서는 경우가 있고, 일부 클래스에서는 TraSeTR가 더 나은 경우도 있다. 논문은 그 이유를 TraSeTR의 implicit tracking 성질에서 찾는다. 즉, TraSeTR는 이전 프레임 embedding을 query로 활용하기 때문에 특정 희소 클래스에서는 시간적 연속성의 이득을 볼 수 있었던 것이다. 반면 MATIS baseline은 그 단계만 놓고 보면 장기 비디오 문맥을 아직 사용하지 않는다.

논문에서 특히 중요한 분석은 upper bound 실험이다. 저자들은 자신들의 region proposal 품질과 classification 오류를 분리해서 보기 위해 두 종류의 upper bound를 계산한다. Inferred Upper Bound는 실제 inference 단계에서 선택된 region들 중 각 GT instance와 IoU가 가장 잘 맞는 예측을 대응시킨 경우다. Total Upper Bound는 inference filtering 이전의 100개 전체 region 중에서 같은 작업을 수행한 경우다. Endovis 2017에서는 Inferred Upper Bound가 mIoU 83.44, mcIoU 81.86이고, Total Upper Bound는 mIoU 90.75, mcIoU 90.44다. Endovis 2018에서는 각각 mIoU 88.84, mcIoU 80.68과 mIoU 91.20, mcIoU 88.67이다.

이 결과는 매우 중요하다. 마스크 자체의 픽셀 정확도는 상당히 높으며, 주요 오류 원인이 “mask generation”보다 “region classification”에 있다는 뜻이기 때문이다. 즉, 모델은 어디가 도구인지는 꽤 잘 찾아내지만, 그 도구가 정확히 어떤 subtype인지는 헷갈리는 경우가 많다. 이 해석은 temporal consistency module이 왜 유효한지도 설명해 준다. 시간 문맥은 새로운 mask를 만드는 것이 아니라, 이미 잘 만들어진 mask의 class를 더 정확히 보정하는 데 특히 유용하다.

실제로 MATIS Full은 MATIS Frame 대비 모든 전체 지표를 향상시킨다. 논문은 특히 Endovis 2018에서 Large Needle Driver가 baseline에서는 Prograsp Forceps로 자주 잘못 분류되었는데, temporal consistency module이 이를 상당 부분 교정한다고 설명한다. 다만 temporal reasoning이 항상 좋은 것만은 아니다. 일부 클래스에서는 장기 문맥이 오히려 classification noise를 도입해 성능이 감소했다고 저자들은 인정한다. 또한 Endovis 2017에서는 fold별 성능 향상 폭이 달라 standard deviation이 커졌다.

Ablation 실험도 비교적 설득력 있게 구성되어 있다. 먼저 baseline inference 방법에 대한 실험에서, 모든 100개 mask를 그대로 쓰는 것보다 class-specific top-$k$와 per-class threshold를 조합하는 방식이 가장 좋았다. Endovis 2018에서 이 조합은 mIoU 82.37, IoU 77.01, mcIoU 48.65를 기록했고, 일반적인 0.5 threshold나 NMS보다 우수했다. 이는 수술 도구 클래스마다 confidence 분포와 등장 패턴이 다르기 때문에, 획일적인 후처리보다 class-aware inference가 더 적절하다는 점을 보여준다.

Temporal consistency module의 설계에 대한 ablation에서는 두 요소가 모두 성능 향상에 기여했다. pooled time feature에 대한 Time MLP가 없고 presence supervision도 없을 때는 mIoU 83.24, IoU 78.10, mcIoU 50.31이었다. Time MLP만 넣으면 83.50 / 78.31 / 51.21, presence supervision만 넣으면 83.63 / 78.23 / 50.96이 된다. 둘 다 넣으면 최종적으로 84.26 / 79.12 / 54.02에 도달한다. 즉, 시간 특징을 segment embedding과 더 잘 결합되도록 변환하는 과정과, global instrument presence를 보조적으로 맞히게 하는 supervision이 함께 가장 큰 효과를 냈다.

마지막으로 입력 시간 창(window)과 stride에 대한 검증에서는 작은 window는 장기 정보 부족 때문에 성능이 떨어졌고, stride를 크게 하면 데이터 샘플링이 성기게 되어 성능이 감소했다. 반대로 너무 큰 window도 먼 프레임의 잡음이 많아져 약간 성능이 나빠졌다. 이 결과는 temporal context가 중요하지만, 그 길이와 샘플링 간격에는 적절한 균형이 필요함을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제를 정확히 다시 정식화했다는 점이다. 수술 도구 분할은 본질적으로 multi-instance 문제인데, 기존 per-pixel classification 접근은 이 구조를 충분히 반영하지 못했다. MATIS는 이를 region set prediction으로 다루고, transformer 기반 mask classification으로 연결해 더 일관된 segmentation을 얻는다. 단순히 backbone만 transformer로 바꾼 것이 아니라, Mask2Former류의 현대적 segmentation 설계를 수술 도메인에 맞춰 성공적으로 이식했다는 점이 강하다.

또 다른 강점은 오류 원인 분석이 비교적 명확하다는 점이다. upper bound 실험을 통해 “mask는 잘 만들고 class가 틀린다”는 사실을 정량적으로 보여 주었고, 그 결과 temporal consistency module의 역할이 왜 필요한지 설득력 있게 설명했다. 논문의 서사가 방법 제안, 오류 분석, 모듈 추가의 논리로 잘 연결된다.

실용적 관점에서도 장점이 있다. Endovis 2017과 2018 두 공개 benchmark에서 일관되게 strong result를 냈고, baseline만으로도 이전 방법들을 넘어서며, temporal module까지 넣으면 추가 개선이 확인된다. 즉, 구조의 핵심 기여가 단일 데이터셋에만 국한된 결과로 보이지는 않는다.

한계도 분명하다. 첫째, 분류 오류가 여전히 주된 병목이다. upper bound 결과가 높다는 것은 반대로 말하면, 아직 최종 class assignment는 충분히 해결되지 않았다는 뜻이다. 둘째, 논문은 overlapped or discontinuous instances에서 여전히 어려움을 겪는다고 인정한다. 수술 장면에서는 도구가 서로 가리거나 한 도구가 시각적으로 끊겨 보이는 경우가 흔하므로, 이는 실제 적용에서 중요한 한계다.

셋째, temporal reasoning이 모든 클래스에 항상 이롭지는 않았다. Endovis 2018의 일부 클래스에서는 오히려 성능이 감소했으며, 저자들은 이를 long-term video reasoning이 introduced classification noise를 만들기 때문이라고 본다. 즉, 시간 문맥 활용이 항상 안정적이지는 않으며, 어떤 클래스나 상황에서는 오히려 bias를 줄 수도 있다.

넷째, 손실 함수의 정확한 구성, 각 loss weight, temporal module 내부 classifier의 상세 설계 같은 일부 구현 세부는 추출된 본문 기준으로 충분히 자세히 제시되어 있지 않다. 따라서 재현이나 더 깊은 이론적 해석을 위해서는 원문 전체의 방법 섹션 또는 공개 코드가 추가로 필요할 수 있다. 이 보고서는 제공된 추출 텍스트에 근거해 작성되었으므로, 텍스트에 명시되지 않은 세부는 단정할 수 없다.

비판적으로 보면, 이 논문은 segmentation mask 생성 자체보다 classification 개선이 더 핵심 병목임을 보여 주었는데, 그 해결이 아직 temporal module 중심에 머물러 있다. 앞으로는 mask embedding과 class semantics를 더 강하게 정렬하는 metric learning, class imbalance 보정, label hierarchy 활용 같은 방향도 가능해 보인다. 다만 이런 제안은 논문 바깥의 해석이며, 논문 본문이 직접 주장한 내용은 아니다.

## 6. 결론

이 논문은 surgical instrument segmentation을 위해 masked attention과 deformable attention을 활용한 fully transformer-based 2-stage 구조 MATIS를 제안했다. 첫 단계에서는 Mask2Former 기반으로 high-quality instrument mask proposal을 만들고, 두 번째 단계에서는 MViT 기반 temporal consistency module이 장기 비디오 문맥을 활용해 각 mask의 클래스를 더 안정적으로 분류한다.

핵심 기여는 세 가지로 요약할 수 있다. 첫째, 수술 도구 분할을 instance-aware mask classification 문제로 다뤄 기존 pixel-classification 방식의 한계를 극복했다. 둘째, 최신 transformer segmentation 메커니즘을 의료 수술 비디오 도메인에 성공적으로 적용했다. 셋째, temporal video reasoning을 결합해 classification consistency를 개선했고, 그 결과 Endovis 2017과 2018에서 새로운 state-of-the-art를 달성했다.

실제 적용 측면에서 이 연구는 수술 장면 이해의 기반 기술을 더 정교하게 만든다는 점에서 의미가 크다. instrument tracking, workflow analysis, pose estimation, robot-assisted surgery support system 같은 후속 시스템의 성능 향상에 직접 연결될 수 있다. 동시에 논문은 mask 품질은 이미 매우 높고 classification이 남은 핵심 병목이라는 사실도 보여 주므로, 이후 연구는 temporal reasoning과 class discrimination을 더 정교하게 결합하는 방향으로 발전할 가능성이 크다.