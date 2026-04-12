# Identifying Surgical Instruments in Laparoscopy Using Deep Learning Instance Segmentation

- **저자**: Sabrina Kletz, Klaus Schoeffmann, Jenny Benois-Pineau, Heinrich Husslein
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/2508.21399

## 1. 논문 개요

이 논문은 복강경 수술 영상에서 수술 기구를 자동으로 찾아내고, 기구의 픽셀 영역을 분할하며, 가능하면 그 기구의 종류까지 식별하는 문제를 다룬다. 저자들은 이 문제를 단순한 존재 여부 분류가 아니라, 각 기구 인스턴스를 개별적으로 구분해야 하는 **instance segmentation** 문제로 정의한다. 구체적으로는 두 가지 설정을 비교한다. 첫째는 기구인지 배경인지만 구분하는 **binary instrument segmentation**이고, 둘째는 기구의 종류까지 구분하는 **multi-class instrument recognition**이다.

연구 문제는 의료 영상 분석에서 매우 실용적이다. 복강경 수술 영상은 수술 후 품질 평가(Surgical Quality Assessment), 술기 분석, 수술 워크플로 분석, 의료 비디오 검색 시스템의 핵심 입력이 될 수 있다. 특히 수술 기구는 영상에서 가장 중요한 관심 객체이기 때문에, 기구를 정확히 찾고 분리하는 기술은 후속 분석의 기반이 된다. 그러나 실제 복강경 영상은 반사, 연기, 블러, 가려짐(occlusion), 복잡한 배경 등으로 인해 자동 분석이 매우 어렵다.

이 논문의 또 다른 중요한 문제의식은 데이터 부족이다. 저자들에 따르면, 전통적 복강경(traditional laparoscopy) 분야에는 로봇 수술 분야와 달리 적절한 공개 segmentation dataset이 거의 없었다. 그래서 이 논문은 gynecologic laparoscopy에 특화된 자체 데이터셋을 구축하고, 그 위에서 Mask R-CNN 기반 접근이 어느 정도까지 유효한지 실험적으로 검증한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 복강경 기구 인식을 단순 분류나 semantic segmentation으로 제한하지 않고, **“기구 하나하나를 따로 찾고, 각 기구의 마스크를 만들며, 가능하면 클래스까지 붙이는 문제”**로 정식화한 데 있다. 즉, 영상 속에서 여러 기구가 동시에 등장할 수 있고 서로 겹칠 수 있으므로, “이 픽셀은 기구다” 수준이 아니라 “이 픽셀들은 특정 기구 인스턴스 A에 속한다”는 수준까지 가야 한다는 것이다.

이를 위해 저자들은 **Mask R-CNN**을 선택한다. 이 모델은 먼저 후보 영역(region proposal)을 만들고, 각 영역의 클래스와 bounding box를 예측한 뒤, 각 객체 영역에 대한 segmentation mask를 생성한다. 따라서 한 장의 이미지 안에서 여러 기구를 각각 독립된 객체로 다룰 수 있다. 이 점이 semantic segmentation 중심의 기존 연구와 가장 큰 차별점이다. 논문에서도 기존 연구 다수가 robotic surgery에서 semantic segmentation 또는 tracking 중심이었고, gynecologic laparoscopy에서 deep learning 기반 multi-class instance segmentation을 다룬 연구는 없었다고 밝힌다.

또 하나의 핵심은 **작은 데이터셋에서도 학습이 가능하도록 데이터 증강을 체계적으로 설계한 점**이다. 저자들은 rotation, scaling, translation, mirroring, Gaussian blur를 조합해 오프라인/온라인 증강을 함께 사용한다. 특히 아무 변환이나 적용하는 것이 아니라, 변환 후에도 원래 segmentation label이 크게 훼손되지 않는 경우만 유지하는 기준을 둔다. 즉, 기구가 잘려 나가거나 mask가 크게 변형된 샘플은 버린다.

## 3. 상세 방법 설명

전체 시스템은 Mask R-CNN에 기반한다. 논문은 이를 region-based Fully Convolutional Network라고 설명하며, 크게 두 부분으로 본다. 하나는 **Region Proposal Network (RPN)** 로, 이미지에서 객체가 있을 만한 후보 영역을 제안한다. 다른 하나는 **Fully Convolutional Network (FCN)** 계열의 마스크 예측 모듈로, 이미 제안된 객체 영역 내부에서 픽셀 단위 foreground/background 분리를 수행한다. 클래스 판별은 Faster R-CNN 계열의 검출 단계에서 담당하고, 마스크 분할은 별도 분기에서 수행된다.

입력 이미지가 들어오면 모델은 다음과 같은 출력을 만든다.

1. 각 객체 후보의 bounding box 좌표
2. 각 후보가 어떤 클래스일 확률인지
3. 해당 후보 영역에 대한 segmentation mask

이 구조 때문에 모델은 “어디에 객체가 있는가”, “무슨 기구인가”, “정확히 어떤 픽셀들인가”를 한 번에 처리할 수 있다.

백본 네트워크로는 **ResNet-101**을 사용한다. 저자들은 robotic surgery semantic segmentation에서 ResNet이 성공적으로 사용된 점을 근거로 든다. 이 선택은 깊은 네트워크의 표현력을 활용해 복강경 영상의 복잡한 시각 패턴을 다루려는 의도로 볼 수 있다.

학습은 end-to-end로 수행되며, 총 손실은 세 가지 손실의 합으로 구성된다고 설명한다.

- 분류 손실(classification loss)
- 위치 추정 손실(localization loss)
- 평균 binary cross-entropy loss

마지막 손실은 이미 분류된 객체 영역에 대해 segmentation mask를 생성하는 데 사용된다. 논문은 손실의 정확한 수식을 풀어서 제시하지는 않았지만, 설명상 전체 목적 함수는 다음과 같은 형태로 이해할 수 있다.

$$
L = L_{\text{cls}} + L_{\text{loc}} + L_{\text{mask}}
$$

여기서 $L_{\text{cls}}$는 객체 클래스 예측 오차, $L_{\text{loc}}$는 bounding box 회귀 오차, $L_{\text{mask}}$는 mask 예측을 위한 binary cross-entropy 성격의 손실이다. 논문은 $L_{\text{mask}}$의 세부 수식은 적지 않았으므로, 그 이상의 구체적 형태를 단정할 수는 없다.

데이터셋은 저자들이 직접 구축했다. 총 333장의 프레임을 여러 gynecologic laparoscopy 수술 영상에서 랜덤하게 추출했고, 각 프레임에 대해 보이는 기구 인스턴스를 수작업으로 분할하여 총 561개의 segmentation mask를 만들었다. 영상 해상도는 $540 \times 360$이다. 클래스는 총 12개로 볼 수 있는데, 실제 기구 종류 11개와 식별 불가한 경우를 담은 “Other” 클래스가 포함된다. 등장하는 클래스는 Bipolar Grasper, Hook, Sealer and Divider, Grasper, Irrigator, Knot-Pusher, Needle-Holder, Scissors, Morcellator, Needle, Trocar, Other이다.

데이터 증강은 이 논문의 중요한 실험 변수다. 저자들은 다음 변환들을 사용한다.

- 회전: $\alpha \in \{0, 45, 90, 135, 180, 225, 270, 315\}$
- 스케일: $c \in \{1.25, 1.50, 1.75\}$
- 평행이동: $(u, v)$, 여기서 $u, v \in \{-0.1, 0.1\}$로 이미지 폭과 높이의 10% 이동
- x축, y축 미러링
- Gaussian blur: $\sigma \in [0, 3.0]$

새로 생기는 빈 픽셀은 해당 이미지의 평균 RGB 값으로 채운다. 또한 오프라인 증강에서는 변환 후 mask 면적이 원본과 너무 달라져 label 손실이 커지는 경우를 제외한다. 온라인 증강에서는 blur와 flipping을 각 입력에 50% 확률로 적용한다.

평가는 COCO 방식의 평균 정밀도(AP)와 평균 재현율(AR)을 쓴다. 핵심 유사도 척도는 **Intersection over Union (IoU)** 이며, 논문은 이를 다음과 같이 정의한다.

$$
IoU = \frac{|T \cap D|}{|T \cup D|}
$$

여기서 $T$는 ground-truth 영역, $D$는 검출 또는 예측된 영역이다. AP50은 IoU threshold 0.5에서의 평균 정밀도이고, AP50:95는 0.50부터 0.95까지 0.05 간격 threshold에서 평균낸 값이다. AR은 최대 검출 개수 조건하의 평균 재현율이다.

실험에서는 데이터를 60% train, 20% validation, 20% test로 분할했다. 각 split에는 클래스별 인스턴스 수가 균등하게 포함되도록 구성했고, validation/test에는 각 기구 클래스당 7개의 예시가 들어가도록 맞췄다고 설명한다. 학습은 공개된 Matterport Mask R-CNN 구현을 사용했고, COCO pretrained weight를 이용한 transfer learning과 scratch 학습을 비교했다. 그 결과, COCO pretrained initialization 후 전체 레이어를 추가 학습하는 방식이 유리했다고 보고한다. optimizer는 SGD, momentum은 $\mu = 0.9$이며, learning rate는 $\eta \in \{0.01, 0.001, 0.0001\}$를 비교해 최종적으로 $\eta = 0.001$이 가장 적절했다고 한다.

## 4. 실험 및 결과

실험은 두 가지 큰 과제를 나눠 평가한다. 첫째는 **binary instrument segmentation**, 둘째는 **multi-class instrument segmentation and recognition**이다. 전자는 기구를 배경과 분리하는 데 집중하고, 후자는 어떤 종류의 기구인지까지 맞혀야 하므로 훨씬 어렵다.

### 데이터셋과 평가 설정

실험 데이터는 총 333장의 원본 이미지, 561개의 인스턴스 mask로 구성된다. 오프라인 증강 후에는 총 3,274장 이미지, 4,340개의 인스턴스로 늘어난다. 클래스 분포는 균등하지 않다. 예를 들어 “Other”는 90개로 많고, Bipolar는 38개, Hook은 36개, Needle은 55개 등 클래스별 차이가 있다. Trocar는 증강 후 738개로 가장 많다.

평가는 segmentation mask와 bounding box를 각각 따로 측정한다. 따라서 표에는 mask 기준 AP와 bounding box 기준 APbb가 동시에 제시된다.

### Binary segmentation 결과

binary segmentation에서는 상당히 좋은 성능이 나온다. 증강을 하지 않은 baseline에서 segmentation mask 성능은 30 epoch 기준으로 AP50이 0.820, AP50:95가 0.522다. 증강을 적용한 경우 30 epoch에서 AP50은 0.814, AP50:95는 0.543이다. 즉, IoU 0.5 기준의 쉬운 판정에서는 증강이 거의 이득을 주지 않았고, 더 엄격한 threshold를 평균낸 AP50:95에서는 약간 좋아졌다.

bounding box 기준으로도 비슷한 경향이 나온다. 증강 없는 경우 30 epoch에서 APbb50은 0.833, APbb50:95는 0.606이고, 증강 있는 경우 30 epoch에서 APbb50은 0.831, APbb50:95는 0.645다. 저자들의 해석은 분명하다. **binary task에서는 데이터 증강이 정확도를 크게 끌어올리기보다 학습 시간을 많이 늘리는 효과가 더 크다.** 논문에서는 epoch당 학습 시간이 약 12배 증가했다고 설명한다.

즉, “기구냐 배경이냐” 수준의 문제는 적은 데이터로도 비교적 잘 풀린다. 이는 기구가 배경과 시각적으로 어느 정도 구분 가능하다는 뜻이다. 실제 결론 부분에서도 저자들은 50% overlap 기준으로 평균 정밀도 $ \ge 81\%$ 수준의 reliable segmentation이 가능하다고 강조한다.

### Multi-class segmentation and recognition 결과

반면 multi-class recognition은 훨씬 어렵다. 증강 없는 경우 50 epoch에서 segmentation mask 기준 AP50은 0.511, AP50:95는 0.331이다. bounding box 기준 APbb50은 0.532, APbb50:95는 0.339다. 즉, 위치는 어느 정도 맞혀도, 클래스까지 정확히 맞히는 것은 어려웠다.

증강을 적용하면 성능이 뚜렷하게 개선된다. 50 epoch에서 segmentation mask 기준 AP50은 0.613, AP50:95는 0.429이고, bounding box 기준 APbb50은 0.627, APbb50:95는 0.457이다. 이 결과는 binary task와 달리, **multi-class task에서는 data augmentation이 실질적으로 중요하다**는 점을 보여준다. 클래스 간 외형 차이가 미묘하고 샘플 수도 적기 때문에, 다양한 시각 변형을 학습시키는 것이 도움이 된 것이다.

논문에는 학습 곡선도 제시된다. training loss는 점차 감소하고 validation loss도 전반적으로 내려가지만 변동이 있으며, 50 epoch 이후에는 validation loss가 빠르게 증가하여 overfitting 경향을 보인다. 따라서 저자들은 50 epoch 부근의 모델을 최종 비교 대상으로 본다.

### 클래스별 결과 해석

클래스별 AP50과 AR1을 보면 어떤 기구는 잘 되고 어떤 기구는 매우 어렵다.

성능이 좋은 클래스에는 Sealer-Divider, Irrigator, Hook, Bipolar가 포함된다. 예를 들어 Sealer-Divider는 segmentation mask 기준 AP50이 1.000, bounding box 기준 APbb50도 1.000이다. Irrigator도 AP50이 0.869, Hook은 0.812, Bipolar는 0.851이다. 이는 외형적 특징이 비교적 뚜렷하거나 구분이 쉬운 경우일 가능성이 높다.

반면 Needle은 매우 어렵다. segmentation mask 기준 AP50이 0.218, AP50:95는 0.051이다. 저자들은 needle이 얇고 배경과 구분이 어려워 segmentation이 힘들다고 정성적으로도 설명한다. Needle-Holder, Grasper, Scissors, Knot-Pusher도 상호 유사성이 높아 클래스 구분이 어렵다. 특히 Needle-Holder는 AP50이 0.517이지만 AR1이 0.057로 매우 낮다. 이는 맞힐 때는 어느 정도 맞히더라도, 전반적으로 잘 검출되지 않는 경우가 많다는 뜻으로 읽을 수 있다.

또 흥미로운 점은 어떤 클래스는 recall은 높은데 precision은 낮다는 것이다. 예를 들어 Knot-Pusher는 segmentation AP50이 0.574인데 AR1은 0.871이다. 이는 꽤 자주 검출되지만, 그중 상당수는 부정확하거나 오검출일 수 있음을 시사한다. 반대로 Irrigator는 AP50은 0.869로 높은데 AR1은 0.343으로 낮다. 즉, 잡히는 경우는 정확하지만 놓치는 경우도 많은 모델일 수 있다.

### 정성적 결과

논문의 Figure 3 설명에 따르면, binary segmentation은 전반적으로 꽤 정확한 마스크를 예측한다. 하지만 복잡한 모양을 가진 Bipolar나 매우 얇은 Needle은 여전히 어렵다. multi-class 결과에서는 같은 위치에 기구가 있다는 사실은 어느 정도 맞혀도, 종류를 구분하는 데서 오류가 늘어난다. 증강을 추가한 multi-class 모델은 증강 없는 경우보다 시각적으로 더 나은 결과를 보인다.

이 정성 결과는 정량 결과와 일관된다. 즉, **기구 영역 자체를 찾는 일은 가능하지만, 종류까지 안정적으로 구분하는 일은 아직 어렵다**는 것이 논문의 핵심 실험 결론이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정의 적절성이다. 복강경 영상 분석에서 실제로 중요한 것은 기구 존재 여부만이 아니라, 기구의 위치와 윤곽, 그리고 가능하면 종류까지 아는 것이다. 이 논문은 이를 instance segmentation 문제로 정식화했고, Mask R-CNN이라는 강력한 프레임워크를 적용해 실질적인 기준 성능을 제시했다.

둘째 강점은 **custom dataset 구축**이다. 기존에 gynecologic laparoscopy용 segmentation 데이터가 부족한 상황에서, 저자들은 333개의 프레임과 561개의 수작업 마스크를 제공할 수 있는 기반을 마련했다. 의료 영상 분야에서 데이터셋 구축 자체가 중요한 기여인 경우가 많다는 점에서 의미가 있다.

셋째 강점은 **binary task와 multi-class task를 명확히 분리해 비교했다는 점**이다. 이 덕분에 “기구 분할은 가능하지만 fine-grained recognition은 어렵다”는 사실을 실험적으로 분명히 보여준다. 또한 data augmentation이 어떤 설정에서 유의미한지까지 드러낸다. binary task에서는 큰 이득이 없지만, multi-class recognition에서는 확실한 성능 향상이 있었다.

넷째 강점은 평가가 비교적 표준적이라는 점이다. COCO의 AP50, AP50:95, AR 지표를 사용해 segmentation과 localization을 분리해서 측정했기 때문에, 결과 해석이 명확하다.

한계도 분명하다. 가장 큰 한계는 데이터 규모다. 원본 프레임이 333장뿐이고, 클래스별 표본 수가 많지 않다. 특히 클래스 간 외형이 매우 비슷한 기구들이 많기 때문에, 소량 데이터 상황에서 multi-class recognition이 불안정해질 수밖에 없다. 저자들도 이 점을 인정하며, 일부 기구는 매우 유사해서 구분이 어렵다고 말한다.

또 다른 한계는 데이터셋의 범위다. 이 데이터는 gynecologic myomectomy와 hysterectomy에 특화되어 있다. 따라서 다른 복강경 수술 종류, 다른 장비, 다른 병원 환경, 다른 화질 조건에서도 같은 성능이 유지된다고 말할 수는 없다. 논문도 외부 일반화 성능에 대한 실험은 제시하지 않는다.

방법론적으로도 novelty는 제한적이다. 핵심 모델은 기존 Mask R-CNN이고, 주된 기여는 새 도메인에 대한 적용과 데이터셋 구축, 실험적 분석에 있다. 즉, 완전히 새로운 네트워크 구조나 학습 알고리즘을 제안하는 논문은 아니다. 이 점은 방법론 논문으로서의 새로움은 약할 수 있지만, 응용 연구로서는 충분히 의미가 있다.

또한 논문은 손실 함수의 구체적 수식, anchor 설정, batch size, epoch당 iteration, inference threshold 등 구현 세부를 매우 자세히 적지는 않는다. 따라서 재현성 측면에서 추가 정보가 필요할 수 있다. 공개 구현을 사용했다고는 하지만, 실험을 정확히 복제하려면 더 많은 설정 정보가 있었으면 좋았을 것이다.

비판적으로 보면, 결론 부분의 “multi-class classification approach에서도 50% overlap 기준 평균 정밀도 $\ge 81\%$”라는 문장은 본문 표와 완전히 자연스럽게 맞아떨어지지 않는다. 표를 보면 multi-class recognition의 최종 AP50은 약 61.3% 수준이고, 81% 이상은 binary segmentation 또는 특정 클래스 단위 성능에 더 가깝다. 따라서 이 부분은 문맥상 다소 과장되었거나 서술이 부정확해 보인다. 논문 본문 수치에 근거하면, **81% 이상이라는 평가는 전체 multi-class 평균 성능이 아니라 일부 설정이나 클래스에 국한된 해석**으로 보는 것이 안전하다.

## 6. 결론

이 논문은 gynecologic laparoscopy 영상에서 수술 기구를 자동으로 분할하고 식별하는 문제를 Mask R-CNN 기반 instance segmentation으로 다룬다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, 복강경 기구 인식을 binary segmentation과 multi-class recognition으로 분리해 체계적으로 평가했다. 둘째, gynecologic surgery용 custom instrument segmentation dataset을 구축했다. 셋째, 적은 데이터에서도 binary segmentation은 꽤 높은 정확도로 가능하지만, 기구 종류까지 구분하는 fine-grained recognition은 여전히 어렵고 data augmentation이 특히 중요하다는 점을 실험적으로 보였다.

실제 적용 측면에서 이 연구는 수술 영상 검색, 수술 품질 평가, 술기 분석, 후속 행동 인식 모델의 전처리 단계 등에 유용할 가능성이 크다. 특히 binary instrument segmentation은 이미 꽤 실용적인 수준에 근접해 보인다. 반면 multi-class recognition은 더 큰 데이터셋, 더 정교한 클래스 설계, temporal information 활용, 더 강한 도메인 특화 모델이 필요해 보인다.

따라서 이 논문은 “복강경 수술 기구를 deep learning으로 개별 인스턴스 단위 분할할 수 있는가?”라는 질문에는 긍정적인 답을 주지만, “기구 종류까지 안정적으로 세밀하게 구분할 수 있는가?”라는 질문에는 아직 제한적이라는 답을 준다. 바로 그 점이 이 연구의 현실적인 가치이자, 동시에 다음 연구가 풀어야 할 과제를 분명하게 보여주는 부분이다.
