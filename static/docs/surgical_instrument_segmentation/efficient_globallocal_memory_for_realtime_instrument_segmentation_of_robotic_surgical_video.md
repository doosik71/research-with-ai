# Efficient Global-Local Memory for Real-time Instrument Segmentation of Robotic Surgical Video

Jiacheng Wang, Yueming Jin, Liansheng Wang, Shuntian Cai, Pheng-Ann Heng, Jing Qin

## 🧩 Problem to Solve

로봇 보조 수술 비디오에서 실시간으로 정확한 수술 도구 분할(instrument segmentation)을 수행하는 것은 수술 성능과 환자 안전을 향상시키는 데 매우 중요합니다. 기존의 대부분 방법은 단일 프레임의 시각적 단서만을 사용하거나, 광학 흐름(optical flow)을 사용하여 제한된 두 프레임 간의 움직임만 모델링하며 높은 계산 비용을 초래합니다. 본 연구는 수술 도구 인식에 있어 **인접 프레임의 지역적 시간 의존성(local temporal dependency)**과 **장기적인 전역적 의미 상관관계(global semantic correlation)**라는 두 가지 중요한 시공간적 단서를 활용하면서, 정확도를 높이고 실시간 예측 능력을 유지하는 문제를 다룹니다.

## ✨ Key Contributions

* **DMNet (Dual-Memory Network) 제안**: 전역적 및 지역적 시공간 지식을 효율적으로 통합하여 현재 피처를 강화하고 수술 도구 분할 성능을 향상시키는 새로운 듀얼 메모리 네트워크인 DMNet을 제안했습니다.
* **효율적인 지역 시간 집계 (Efficient Local Aggregation, ELA) 모듈 개발**: ConvLSTM과 Non-local 메커니즘의 상호 보완적인 장점을 활용하여 지역 범위의 시간 의존성을 효율적으로 포착하는 모듈을 설계했습니다.
* **능동적 전역 시간 집계 (Active Global Aggregation, AGA) 모듈 개발**: 모델 불확실성(representativeness)과 프레임 유사성(similarity)을 기반으로 가장 유익한 프레임을 선별하여 장기적인 전역적 의미 상관관계를 현재 프레임에 통합하는 능동적 전역 메모리 모듈을 개발했습니다.
* **우수한 성능 입증**: 두 가지 공개 수술 비디오 벤치마크 데이터셋에서 최신 기술(SOTA) 대비 분할 정확도를 크게 능가하면서도 실시간 처리 속도를 유지함을 실험적으로 입증했습니다.

## 📎 Related Works

* **단일 프레임 기반 분할**: ToolNet, Shvets et al. (EndoVis Challenge 2017 우승), LWANet, PAANet 등은 주로 단일 이미지의 시각적 단서에만 의존하여 시퀀스 비디오의 시간적 정보를 무시했습니다.
* **광학 흐름 기반 분할**: Jin et al., Zhao et al.은 광학 흐름을 활용하여 시간적 정보를 통합했으나, 계산 비용이 높고 제한적인 두 프레임 사이의 움직임 정보만을 포착한다는 한계가 있었습니다.
* **장거리 시간 정보 활용**: TCN(workflow recognition), Memory Enhanced Global-Local Aggregation (video object detection), Space-Time Memory Networks (video object segmentation) 등 자연 영상 분야에서 장거리 시간 정보를 활용하는 연구들이 있었으나, 수술 도구 분할 작업에 직접 적용하기 어렵고 실시간 예측에 대한 고려가 부족했습니다.

## 🛠️ Methodology

본 연구에서 제안하는 DMNet은 두 가지 메모리(지역 메모리, 전역 메모리)와 해당 메모리를 활용하는 두 가지 집계 모듈(ELA, AGA)을 통해 시공간 지식을 효율적으로 통합합니다.

1. **DMNet 아키텍처**:
    * 현재 프레임 $I_t$에 대해 지역 메모리 $L_t$와 전역 메모리 $G_t$를 사용합니다.
    * $L_t$는 이전 $\tau$($=4$)개 프레임 $\{f_{t-\tau}, ..., f_{t-1}\}$에서 추출된 피처 맵으로 구성되며, ELA에 사용됩니다.
    * $G_t$는 모든 이전 프레임 중에서 능동적으로 선택된 가장 유익한 피처 맵들로 구성되며, AGA에 사용됩니다.
    * 인코더-디코더 구조를 기반으로 하며, MobileNetv2 인코더와 RefineNet 디코더를 사용합니다.

2. **효율적인 지역 시간 집계 (Efficient Local Aggregation, ELA)**:
    * **BottleneckLSTM 활용**: 지역 메모리 클립에서 시간 차원 정보를 집계하기 위해 BottleneckLSTM을 사용합니다. 이는 표준 ConvLSTM보다 효율적입니다. 이를 통해 시간적으로 풍부한 피처 $\tilde{f}_t$를 얻습니다.
    * **Non-local 메커니즘 통합**: BottleneckLSTM의 작은 수용 영역 한계를 보완하기 위해 Non-local 메커니즘을 적용합니다.
        * $\tilde{f}_t$로부터 키($\tilde{k}_t$)와 값($\tilde{v}_t$) 피처 맵을 생성합니다.
        * 유사도 함수 $F_{sim}(x,y) = \exp(x \circ y)$를 사용하여 $\tilde{k}_t$의 모든 공간 위치와 다른 위치들 간의 유사도를 계산합니다.
        * 이 유사도 가중치를 사용하여 $\tilde{v}_t$에서 값을 가중합하고, 이를 원본 $\tilde{v}_t$와 연결하여 지역적으로 집계된 피처 $f_{local_t}$를 생성합니다.
        $$ f^{i}_{local_t} = F_{NL}(\tilde{f}^i_t) = \left[ \tilde{v}^i_t, \frac{1}{Z} \sum_{\forall j} F_{sim}(\tilde{k}^i_t, \tilde{k}^j_t) \tilde{v}^j_t \right] $$

3. **능동적 전역 시간 집계 (Active Global Aggregation, AGA)**:
    * **능동적 메모리 업데이트 전략**: 전역 메모리 $G_t$를 구축하기 위해 "대표성(representativeness)"과 "유사성(similarity)"의 두 가지 기준을 사용하여 유익한 프레임만을 선택합니다.
        * **대표성**: 현재 프레임의 예측 마스크 $\hat{M}_t$에 대한 엔트로피 $r = \frac{1}{N}\sum_i \sum_c p^i_c \log p^i_c$를 계산하여 예측 불확실성을 측정합니다. $r > \alpha$($=-0.08$)인 경우에만 프레임을 $G_t$에 추가합니다.
        * **유사성**: 최신으로 $G_t$에 추가된 프레임 $f^{G_{latest}}_t$와의 음의 유클리드 거리 $s = -\sqrt{\sum(f_t - f^{G_{latest}}_t)^2}$를 계산하여 유사도를 측정합니다. $s < \beta$($=-4.65$)인 경우에만 프레임을 추가하여 중복을 피합니다.
    * **전역 집계**: $G_t$에서 $n$($=4$)개의 프레임 피처 $\{f_g\}^n_{g=1}$를 무작위로 선택하여 ELA로 강화된 현재 프레임 피처 $f_{local_t}$와 함께 Non-local 메커니즘을 통해 집계합니다. 이는 전역적으로 집계된 피처 $f_{global}$을 생성합니다. $f_{global} = F_{NL}(f_{local_t}, \{f_g\}^n_{g=1})$
    * **최종 출력**: $f_{global}$은 디코딩되어 수술 도구의 최종 마스크를 생성합니다. 모델 최적화를 위해 Dice loss를 사용합니다.

## 📊 Results

* **데이터셋**: EndoVis17(도구 유형 분할) 및 EndoVis18(도구 부분 분할) 두 가지 공개 벤치마크 데이터셋에서 성능을 검증했습니다.
* **정확도**:
  * **EndoVis17**: mDice 61.03%, mIoU 53.89%를 달성하여 SOTA 모델(예: TDNet의 mDice 54.64%, mIoU 49.24%) 대비 크게 향상된 성능을 보였습니다.
  * **EndoVis18**: mDice 77.53%, mIoU 67.50%를 달성하여 마찬가지로 경쟁 모델들을 능가했습니다.
* **실시간 성능**: 평균 26.37ms의 추론 시간(38 FPS)을 기록하여 실시간 예측 요구사항을 충족했습니다. (TDNet은 22.23ms로 DMNet과 유사한 속도를 가짐)
* **개별 모듈의 효과**:
  * ELA 단독 적용 시 mIoU가 5.74% 증가했습니다.
  * AGA 단독 적용 시 mIoU가 5.13% 증가했습니다.
  * ELA와 AGA를 모두 적용했을 때 mIoU가 7.38% 증가하여 두 모듈이 상호 보완적으로 작동함을 입증했습니다.
* **ELA 모듈 설계 분석**: 제안된 ELA 모듈은 Non-local 및 BottleneckLSTM보다 더 나은 mIoU 성능(52.25%)을 보이면서도, 표준 ConvLSTM(53.07% mIoU)과 비슷한 성능을 유지하며 FLOPS는 거의 절반 수준으로 효율성과 정확도 간의 좋은 균형을 보여주었습니다.

## 🧠 Insights & Discussion

* 본 연구는 수술 비디오에서 도구 분할을 위한 지역적 및 전역적 시공간 정보의 중요성을 명확히 보여줍니다. 단일 프레임 정보나 제한적인 광학 흐름에 의존하는 기존 방법의 한계를 극복했습니다.
* ELA 모듈은 ConvLSTM의 시간 모델링 능력과 Non-local 메커니즘의 넓은 공간적 수용 영역 능력을 결합하여, 작은 수용 영역으로 인한 정보 손실 없이 효율적으로 지역적 시간 정보를 통합합니다.
* AGA 모듈은 능동 학습 전략(대표성 및 유사성 기준)을 도입하여 전역 메모리의 정보 밀도를 높이고, 메모리 비용과 계산 복잡성을 줄이면서도 가장 유익한 장거리 시간 정보를 추출합니다. 이는 motion blur나 반복되는 유사한 장면과 같은 불필요한 정보를 걸러내는 데 효과적입니다.
* DMNet은 자세 변화, 심한 가림, 조명 변화 등 복잡한 수술 환경에서 발생하는 어려운 상황에서도 높은 정확도를 유지하며, 실시간 수술 도구 분할의 가능성을 크게 확장했습니다. 이는 수술 과정 중 의사에게 중요한 상황 인식을 제공하고 의사 결정을 지원하며 잠재적 편차에 대한 경고를 생성하는 데 기여할 수 있습니다.

## 📌 TL;DR

**문제**: 로봇 수술 비디오에서 실시간으로 정확한 수술 도구 분할을 달성하기 위해 지역적 및 전역적 시공간 정보를 효율적으로 통합해야 합니다.

**방법**: 본 연구는 DMNet(Dual-Memory Network)을 제안합니다. DMNet은 ConvLSTM과 Non-local 메커니즘을 결합한 ELA(Efficient Local Aggregation) 모듈을 통해 인접 프레임의 지역적 시간 종속성을 포착합니다. 또한, 모델 불확실성(representativeness)과 프레임 유사성(similarity)을 기반으로 가장 유익한 장거리 프레임을 능동적으로 선택하는 AGA(Active Global Aggregation) 모듈을 통해 전역적 의미 정보를 통합합니다.

**결과**: DMNet은 EndoVis17 및 EndoVis18 데이터셋에서 mDice 및 mIoU 측면에서 최신 기술 대비 뛰어난 분할 정확도를 달성했으며, 동시에 실시간 추론 속도(38 FPS)를 성공적으로 유지했습니다.
