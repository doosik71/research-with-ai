# Segment Any Anomaly without Training via Hybrid Prompt Regularization

- **저자**: Yunkang Cao, Xiaohao Xu, Chen Sun, Yuqi Cheng, Zongwei Du, Liang Gao, Weiming Shen
- **발표연도**: 2023
- **arXiv**: https://arxiv.org/abs/2305.10724

## 1. 논문 개요

이 논문은 추가 학습 없이 anomaly segmentation을 수행하는 문제를 다룬다. 저자들은 이를 zero-shot anomaly segmentation(ZSAS)으로 정의하며, 테스트 대상 카테고리에 대해 정상 이미지도, 비정상 이미지도 제공되지 않는 상태에서 입력 이미지의 각 픽셀에 anomaly degree를 할당하는 것을 목표로 한다. 즉, 학습 데이터셋이 비어 있는 상태 $\emptyset$ 에서 이미지 $I \in \mathbb{R}^{h \times w \times 3}$ 를 입력받아 anomaly map $A \in [0,1]^{h \times w \times 1}$ 를 생성하는 문제다.

이 문제는 산업 검사에서 특히 중요하다. 기존 anomaly segmentation 방법들은 대체로 정상 샘플을 모아 카테고리별로 모델을 학습해야 한다. 그러나 실제 산업 현장에는 제품 종류가 매우 많고, 각 제품마다 별도의 정상 데이터셋을 구축하는 것은 비용이 크며 초기 배치 단계에서는 사실상 불가능할 수 있다. 논문은 이러한 배경에서, 최근의 foundation model이 보여준 zero-shot 일반화 능력을 anomaly segmentation에도 활용할 수 있는지 탐구한다.

저자들의 핵심 출발점은 GroundingDINO 같은 text-guided object detection 모델과 SAM 같은 segmentation foundation model을 조합하면 “학습 없이도” anomaly 영역을 찾을 수 있을 것이라는 점이다. 하지만 단순히 “anomaly” 같은 거친 언어 프롬프트만 쓰면 false alarm이 심해진다. 논문은 이 문제를 prompt design의 문제로 보고, domain expert knowledge와 target image context를 함께 이용하는 hybrid prompt regularization을 제안한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 anomaly segmentation을 위한 새로운 학습 모델을 만드는 것이 아니라, 이미 학습된 foundation model들의 능력을 적절한 프롬프트 설계로 “정렬”하는 것이다. 기본 버전인 SAA는 anomaly region generator와 anomaly region refiner를 순차적으로 연결한 구조다. 먼저 언어 프롬프트로 의심 영역의 bounding box를 만들고, 그 다음 SAM으로 이를 픽셀 단위 마스크로 정제한다.

하지만 저자들은 단순 프롬프트가 심각한 language ambiguity를 낳는다고 지적한다. 예를 들어 “anomaly”라는 단어는 foundation model의 사전학습 데이터 분포에서 anomaly segmentation 문맥으로 충분히 학습되지 않았을 수 있고, 실제 어떤 것이 anomaly인지는 물체 맥락에 따라 달라진다. 촛불 이미지에서 모든 wick을 anomaly로 보는 것은 이런 모호성의 전형적 사례다.

이를 해결하기 위해 SAA+는 하이브리드 프롬프트를 도입한다. 첫째, domain expert knowledge를 이용해 더 구체적인 class-specific language prompt와 object property prompt를 준다. 둘째, target image context를 이용해 saliency와 confidence ranking 기반 프롬프트를 추가한다. 전자는 “무엇을 anomaly로 봐야 하는가”를 더 구체화하고, 후자는 “이 이미지 안에서 상대적으로 얼마나 튀는가”를 반영해 후보의 신뢰도를 다시 조정한다. 기존 접근과의 차별점은, CLIP 기반 텍스트-비주얼 유사도로 직접 anomaly를 점수화하는 대신, 먼저 region proposal을 만들고 그것을 다양한 prompt로 정규화해 최종 anomaly map을 얻는다는 점이다.

## 3. 상세 방법 설명

전체 파이프라인은 두 단계로 이해할 수 있다. 먼저 SAA가 foundation model assembly로 기본 후보 영역을 만들고, 그 위에 SAA+가 hybrid prompts로 후보를 걸러내고 재점수화한다.

### 3.1 SAA: 기본 후보 생성과 정제

첫 단계는 Anomaly Region Generator이다. 이 모듈은 GroundingDINO를 사용해 텍스트 프롬프트 $T$ 와 이미지 $I$ 를 입력받아 bounding-box 수준의 후보 영역 $R_B$ 와 각 후보의 confidence $S$ 를 만든다. 논문은 이를 다음과 같이 쓴다.

$$
R_B, S := Generator(I, T)
$$

의미는 간단하다. “anomaly”, “defect” 같은 텍스트를 질의로 넣으면 모델이 그 설명에 맞는 영역 후보를 박스로 반환한다.

두 번째는 Anomaly Region Refiner이다. 여기서는 SAM을 사용해 박스 후보 $R_B$ 를 픽셀 단위의 segmentation mask 집합 $R$ 로 바꾼다.

$$
R := Refiner(I, R_B)
$$

즉, GroundingDINO가 대략적인 위치를 찾고, SAM이 그 박스를 더 정밀한 경계로 바꾸는 역할을 맡는다. 이 두 모듈을 합친 기본 시스템이 SAA이며, naive class-agnostic prompt $T_n$ 을 이용해 다음처럼 표현된다.

$$
R, S := SAA(I, T_n)
$$

### 3.2 왜 기본 SAA만으로는 부족한가

저자들에 따르면 SAA의 가장 큰 문제는 language ambiguity다. “anomaly”라는 단어 자체가 너무 거칠고, 실제 이상 상태는 물체 종류에 따라 매우 다르다. 가죽에서는 scratch가 anomaly일 수 있지만, 헤이즐넛에서는 crack이 anomaly일 수 있다. 따라서 foundation model이 일반 의미의 단어를 보고 정확한 불량 영역만 찾는 것은 어렵다. 이 때문에 false positive가 많이 발생한다.

### 3.3 SAA+: 하이브리드 프롬프트 정규화

SAA+는 크게 두 부류의 프롬프트를 추가한다. 하나는 domain expert knowledge에서 오고, 다른 하나는 target image context에서 온다.

#### 3.3.1 Domain expert knowledge 기반 프롬프트

첫 번째는 anomaly language expression prompt다. 이는 두 종류로 나뉜다. class-agnostic prompt $T_a$ 는 “anomaly”, “defect”처럼 일반적인 표현이고, class-specific prompt $T_s$ 는 전문가가 비슷한 제품 경험을 바탕으로 제시하는 보다 구체적인 상태 표현이다. 예를 들어 “black hole”, “white bubble”처럼 사전학습 데이터에서 더 자연스럽게 등장했을 표현을 쓰는 방식이다. 저자들의 해석은, “anomaly를 찾아라”보다 “이런 상태의 물체를 찾아라”가 foundation model에게 더 쉬운 질의라는 것이다. 이 둘을 묶어 $P_L = \{T_a, T_s\}$ 로 둔다.

두 번째는 object property prompt다. 언어만으로는 count, size, location 같은 속성을 정확히 제어하기 어렵기 때문에, 논문은 이를 규칙 기반 필터로 넣는다. 사용한 속성은 위치와 면적이다.

위치 프롬프트는 anomaly가 보통 검사 대상 물체 내부에 있어야 한다는 가정에 기반한다. 먼저 foundation model로 검사 대상 object를 찾고, anomaly 후보와 object 사이의 IoU를 계산한 뒤, 전문가가 정한 임계값 $\theta_{\text{IoU}}$ 보다 작은 후보는 버린다.

면적 프롬프트는 anomaly가 대개 전체 object보다 훨씬 작다는 직관을 반영한다. 전문가가 준 $\theta_{\text{area}}$ 를 이용해, 후보 영역의 크기가 $\theta_{\text{area}} \cdot \text{ObjectArea}$ 와 맞지 않는 경우 제거한다.

이 두 속성 프롬프트를 묶어 $P_P = \{\theta_{\text{area}}, \theta_{\text{IoU}}\}$ 라고 하고, 후보 집합을 필터링한다.

$$
R_P, S_P := Filter(R, P_P)
$$

즉, $R_P$ 는 규칙을 통과한 더 신뢰할 만한 anomaly 후보들이다.

#### 3.3.2 Target image context 기반 프롬프트

세 번째는 anomaly saliency prompt다. 논문은 사람이 anomaly를 볼 때 주변과의 불일치를 본다는 점에 착안한다. 이를 위해 이미지의 각 픽셀 특징이 주변 이웃 픽셀들과 얼마나 다른지 측정해 saliency map $s$ 를 만든다. 수식은 다음과 같다.

$$
s_{ij} := \frac{1}{N} \sum_{f \in N_p(f_{ij})} \left(1 - \langle f_{ij}, f \rangle \right)
$$

여기서 $(i,j)$ 는 픽셀 위치, $f_{ij}$ 는 그 픽셀의 feature, $N_p(f_{ij})$ 는 그 픽셀 feature의 $N$ 개 최근접 이웃, $\langle \cdot,\cdot \rangle$ 는 cosine similarity다. 직관적으로는 “주변과 많이 다를수록 saliency가 높다”는 뜻이다. feature는 ImageNet으로 사전학습된 WideResNet50에서 얻는다.

그 다음 각 후보 마스크 내부의 평균 saliency를 구하고 지수 함수를 씌워 region-level saliency prompt $P_S$ 를 만든다.

$$
P_S := \left\{ \exp \left( \frac{\sum_{ij} r_{ij} s_{ij}}{\sum_{ij} r_{ij}} \right) \mid r \in R_P \right\}
$$

이 값은 해당 후보가 이미지 안에서 얼마나 튀는지를 나타낸다. 이후 기존 confidence $S_P$ 와 곱해 새 점수 $S_S$ 를 만든다.

$$
S_S := \{ p \cdot s \mid p \in P_S, s \in S_P \}
$$

즉, foundation model이 높게 본 후보라도 이미지 문맥상 평범하면 점수가 낮아지고, 반대로 주변과 확실히 다른 후보는 더 강조된다.

네 번째는 anomaly confidence prompt다. 논문은 일반적으로 한 이미지 안의 anomaly 영역 수가 많지 않다고 보고, confidence가 높은 상위 $K$ 개 후보만 사용한다.

$$
R_C, S_C := TopK(R_P, S_S)
$$

그리고 최종 anomaly map은 이 top-$K$ 후보들을 픽셀별로 가중 평균해 만든다.

$$
A_{ij} := \frac{\sum_{r_C \in R_C} r^{ij}_C \cdot s_C}{\sum_{r_C \in R_C} r^{ij}_C}
$$

이 수식의 의미는 어떤 픽셀이 여러 후보 마스크에 포함될 경우, 그 후보들의 신뢰도 $s_C$ 를 반영해 최종 anomaly score를 정한다는 것이다. 결국 SAA+는 언어, 속성, saliency, confidence의 네 종류 프롬프트 $P_L, P_P, P_S, P_C$ 로 기본 foundation model 출력을 정규화한 시스템이다.

## 4. 실험 및 결과

실험은 VisA, MVTec-AD, KSDD2, MTD 네 개의 pixel-level annotation 데이터셋에서 수행되었다. VisA와 MVTec-AD는 다양한 object subset을 포함하고, KSDD2와 MTD는 texture anomaly 비중이 큰 데이터셋이다. 저자들은 전체 카테고리를 texture와 object로도 나누어 성능을 비교했다.

평가 지표는 두 가지다. 첫 번째는 pixel-wise segmentation의 최적 임계값 F1인 max-F1-pixel ($F_p$) 이다. 두 번째는 논문이 새로 제안한 max-F1-region ($F_r$) 으로, 큰 결함에만 유리한 편향을 줄이기 위해 region 단위 F1을 측정한다. region overlap이 $0.6$ 을 넘으면 positive로 간주한다.

구현 측면에서 SAA는 official GroundingDINO와 SAM 구현을 사용해 구성되었다. saliency 계산에는 ImageNet pretrained WideResNet50을 사용했고, 최근접 이웃 수는 $N=400$, confidence top-$K$ 는 기본값 $K=5$ 로 설정했다. 입력 해상도는 $400 \times 400$ 이다.

정량 결과에서 SAA+는 모든 비교 방법보다 큰 폭으로 우수했다. 전체 평균 기준으로 $F_p$ 는 SAA+가 34.85를 기록했고, ClipSeg는 20.58, UTAD는 16.19, SAA는 18.22였다. $F_r$ 도 SAA+가 34.07로 가장 높았고, ClipSeg는 13.06, UTAD는 11.49, SAA는 19.74였다. 특히 texture category에서 강했는데, $F_p$ 기준 texture는 53.79, object는 28.82였고, $F_r$ 기준 texture는 60.40, object는 25.70이었다. 저자들은 이를 texture anomaly detection에서 annotation 없이도 상당한 능력을 보였다는 증거로 해석한다.

데이터셋별로 보면 SAA+의 $F_p$ 는 VisA 27.07, MVTec-AD 39.40, KSDD2 59.19, MTD 35.40이었고, $F_r$ 는 각각 14.46, 49.67, 39.34, 30.27이었다. 특히 KSDD2와 texture 계열 성능이 높다. 반면 기본 SAA는 MVTec-AD의 $F_r$ 가 32.49로 일부 가능성을 보였지만, 전반적으로 texture에서 매우 약했다. 이는 단순 foundation model 조합만으로는 false positive를 충분히 제어하지 못함을 보여준다.

정성 결과에서도 SAA+가 더 정확한 영역을 찾는 것으로 보고된다. 논문은 가죽의 작은 scratch, 헤이즐넛의 crack, candle의 overlong wick 같은 예시를 제시하며, saliency와 hybrid prompt가 실제로 이상 부위를 강조하는 데 도움이 된다고 설명한다.

Ablation study는 이 논문의 설계 논리를 강하게 뒷받침한다. language prompt를 제거하면 전체 $F_p$ 가 34.85에서 30.95로, $F_r$ 가 34.07에서 29.17로 감소한다. class-agnostic과 class-specific 설명은 각각도 유효하지만, 같이 쓸 때 가장 좋았다. property prompt를 빼면 texture의 $F_p$ 가 53.79에서 21.83으로 크게 떨어진다. 이는 위치와 면적 필터가 특히 texture 데이터에서 false detection을 크게 줄였음을 뜻한다. saliency prompt를 제거하면 전체 성능이 소폭 하락하며, 정성 예시에서는 cracked hazelnut이나 overlong wick 같은 미세한 이상을 더 잘 찾는 데 도움이 되었다고 설명한다. confidence prompt 역시 상위 $K$ 개만 사용하는 방식으로 false positive를 줄였고, $K \approx 5$ 부근에서 최적 성능이 관찰되었다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 “학습 없는 anomaly segmentation”이라는 매우 도전적인 설정에서 foundation model assembly만으로 실질적인 성능 향상을 보여줬다는 점이다. 단순히 기존 foundation model을 붙여 쓰는 데 그치지 않고, 왜 실패하는지를 language ambiguity라는 관점에서 분석하고, 이를 hybrid prompt regularization으로 해결한 점이 설계적으로 설득력 있다. 또한 property prompt, saliency prompt, confidence prompt처럼 서로 다른 정보원에서 온 규제를 체계적으로 결합했고, 각 구성 요소의 기여를 ablation으로 분리해 확인했다.

또 다른 강점은 region proposal 기반 접근의 실용성이다. CLIP 유사도 기반처럼 픽셀마다 텍스트 정렬을 직접 계산하는 방식보다, GroundingDINO와 SAM을 이용해 후보를 만들고 정제한 후 다시 점수화하는 구조는 anomaly segmentation 문제에 더 직접적이다. 특히 texture anomaly에서 큰 성능 이득을 얻은 점은 산업 검사 응용에서 의미가 있다.

한편 한계도 분명하다. 첫째, class-specific prompt와 property threshold는 domain expert knowledge에 의존한다. 즉, “학습은 없지만 완전한 무지식 설정도 아니다.” 실제 적용 시 어떤 이상 표현을 넣을지, $\theta_{\text{IoU}}$ 와 $\theta_{\text{area}}$ 를 어떻게 정할지는 전문가 개입이 필요하다. 논문은 supplementary material에 프롬프트 상세가 있다고만 말하고 본문에는 구체적 작성 절차를 충분히 제시하지 않는다. 따라서 재현성이나 현장 일반화 측면에서 이 부분은 부담이 될 수 있다.

둘째, confidence top-$K$ 가정은 anomaly 수가 적다는 전제에 기반한다. 많은 이상이 동시에 존재하는 경우에는 이 설계가 불리할 가능성이 있다. 논문은 $K$ 민감도 분석을 제시하지만, anomaly 개수가 실제로 많은 경우에 대한 별도 실험은 제공하지 않는다.

셋째, saliency 기반 재점수화는 “주변과 다른 것”을 이상으로 보는 강한 귀납 bias를 가진다. 이는 texture anomaly에는 유리할 수 있지만, object 구조가 복잡하거나 정상 내부 변동성이 큰 경우에는 항상 유효하다고 단정하기 어렵다. 실제로 object category 성능이 texture보다 낮다는 점도 이런 해석과 맞닿아 있다.

넷째, 논문 스스로도 computation restriction 때문에 더 큰 foundation model에서 scaling effect를 검증하지 못했다고 인정한다. 따라서 제안법이 더 강력한 최신 모델에서도 동일한 방식으로 통하는지는 본문만으로는 확인할 수 없다.

## 6. 결론

이 논문은 zero-shot anomaly segmentation을 위해 GroundingDINO와 SAM을 결합한 SAA를 출발점으로 삼고, domain expert knowledge와 target image context를 활용하는 hybrid prompt regularization을 추가한 SAA+를 제안했다. 핵심 기여는 학습 없이도 anomaly segmentation이 가능함을 보였다는 점, 그리고 그 성공의 열쇠가 단순 모델 교체가 아니라 prompt design에 있음을 체계적으로 보여준 점이다.

실험적으로 SAA+는 VisA, MVTec-AD, KSDD2, MTD에서 기존 동시대 방법들보다 우수한 성능을 보였고, 특히 texture anomaly에서 강한 결과를 냈다. 이 연구는 label-free model adaptation이라는 관점에서 의미가 있으며, 향후 더 큰 foundation model, 더 자동화된 prompt generation, 전문가 의존도를 줄이는 규칙 학습 방향으로 확장될 가능성이 크다. 다만 실제 적용에서는 전문가 프롬프트 설계와 threshold 설정이 필요하므로, 완전 자동화된 산업 배포를 위해서는 이 부분의 추가 연구가 중요해 보인다.
