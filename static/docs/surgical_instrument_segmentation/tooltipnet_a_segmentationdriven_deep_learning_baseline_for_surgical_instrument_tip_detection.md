# ToolTipNet: A Segmentation-Driven Deep Learning Baseline for Surgical Instrument Tip Detection

- **저자**: Zijian Wu, Shuojue Yang, Yueming Jin, Septimiu E. Salcudean
- **발표연도**: 2025
- **arXiv**: https://arxiv.org/abs/2504.09700

## 1. 논문 개요

이 논문은 복강경 수술 영상에서 **surgical instrument tip**, 특히 좌우 gripper tip의 2D 위치를 더 정확하게 찾기 위한 비전 기반 방법을 제안한다. 저자들이 다루는 핵심 문제는 기존의 da Vinci 로봇 API에서 읽은 kinematics 정보만으로는 실제 tip 위치를 정확히 얻기 어렵다는 점이다. 논문에 따르면 이는 cable의 flexibility와 backlash 때문에 발생하며, hand-eye calibration만으로 충분히 보정하기 어렵다.

이 문제는 단순한 detection 문제를 넘어, 실제 컴퓨터 보조 수술과 surgical AI 전반에 중요한 의미를 가진다. 예를 들어 RALP에서는 tool tip으로 조직 표면을 찌른 위치를 ultrasound와 laparoscope 사이의 대응점으로 활용할 수 있기 때문에, tip 위치를 정확히 추정하는 일은 TRUS-laparoscope registration의 정확도와 직접 연결된다. 또한 tip trajectory는 surgical skill assessment에서 유용하고, imitation learning이나 surgery automation에서도 keypoint 기반 action representation을 만드는 데 중요하다.

논문의 기본 관점은 다음과 같다. 최근 Segment Anything 계열의 segmentation foundation model 덕분에 수술 도구 segmentation은 이전보다 쉬워졌고, 그렇다면 RGB 영상 전체를 직접 해석하기보다 **도구의 part-level segmentation mask만으로 tool tip을 찾을 수 있는가**라는 질문이 자연스럽게 생긴다. 저자들은 이 질문에 답하기 위해, mask를 입력으로 받아 tip 위치 heatmap을 예측하는 **ToolTipNet**을 제안하고, hand-crafted image processing 방법과 비교해 시뮬레이션 및 실제 데이터에서 더 나은 성능을 보였다고 보고한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **tool tip은 instrument silhouette 상의 특별한 기하학적 위치**라는 점에 있다. 특히 gripper tip은 gripper region의 corner point에 해당하므로, RGB texture나 조명 정보보다도 mask가 주는 형상 정보만으로도 충분히 강한 단서를 줄 수 있다는 것이 저자들의 출발점이다.

기존의 deep learning 기반 surgical instrument pose/keypoint detection 연구들은 있었지만, 이 논문은 그중에서도 **좌우 gripper tip detection 자체를 segmentation-driven problem으로 단순화**했다는 점이 특징이다. 즉, “영상 전체에서 tip을 찾는다”가 아니라 “이미 확보된 part-level mask에서 tip을 찾는다”는 문제 설정을 취한다. 이 설정은 실제 시스템 전체로 보면 segmentation 모듈에 의존하지만, 그 대신 tip detection 문제를 더 직접적이고 안정적으로 풀 수 있게 해 준다.

또 하나의 핵심은 gripper part mask를 이용한 **mask-guided attention**이다. 단순히 전체 instrument mask만 backbone에 넣는 것이 아니라, tip이 위치할 가능성이 높은 gripper 영역을 attention 형태로 강조함으로써, 모델이 손잡이 외의 shaft나 배경과 무관한 영역보다 tip 근처 형상에 더 집중하도록 설계했다. 이 점은 단순한 silhouette 기반 회귀보다 더 구조적인 inductive bias를 제공한다.

## 3. 상세 방법 설명

전체 입력은 **instrument의 part-level segmentation mask**이고, 출력은 **두 개의 tool tip 위치**이다. 논문에 따르면 한 프레임에는 하나의 instrument가 있고, 그 instrument에는 두 개의 tip이 존재한다고 가정한다. 따라서 문제는 사실상 단일 도구의 좌우 gripper tip 두 점을 검출하는 keypoint detection 문제로 정식화된다.

모델 backbone으로는 **HRNet-w32**를 사용한다. 저자들이 HRNet을 선택한 이유는 high-resolution representation을 유지하는 구조가 keypoint detection처럼 위치 민감한 task에 적합하기 때문이다. 이는 합리적인 선택이다. tip은 매우 작고 정밀한 localization이 중요하므로, 저해상도 latent map만 사용하는 backbone보다 공간 해상도를 보존하는 backbone이 유리할 가능성이 크다.

모델의 흐름은 다음과 같이 정리할 수 있다.

첫째, part-level mask를 HRNet backbone에 입력한다.  
둘째, backbone에서 여러 해상도의 feature pyramid를 추출한다. 논문에 명시된 해상도는 $1/16$, $1/8$, $1/4$, $1/2$ 스케일이다.  
셋째, 이 multi-resolution feature들을 fuse하여 최종적으로 크기 $(H/4, W/4)$의 feature map을 만든다.  
넷째, gripper part mask로부터 **mask-guided attention map**을 생성하고, 이를 fused feature map에 곱한다.  
다섯째, attention이 적용된 feature를 prediction head에 넣어 tool tip heatmap을 출력한다.  
마지막으로 heatmap으로부터 tip 위치를 예측한다.

논문은 prediction head의 구체적인 내부 구조를 자세히 설명하지는 않는다. 따라서 head가 몇 개의 convolution layer로 구성되는지, 두 tip을 어떤 방식으로 분리해 출력하는지, heatmap decoding을 argmax로 하는지 등은 본문 발췌만으로는 알 수 없다. 이 부분은 논문에 명확히 제시되지 않은 정보로 보는 것이 맞다.

손실 함수는 다음과 같다.

$$
L = 0.1 \cdot L_1 + 100 \cdot L_{\mathrm{MSE}}
$$

여기서 $L_1$은 tool tip 좌표의 L1 loss이고, $L_{\mathrm{MSE}}$는 heatmap에 대한 mean squared error loss이다. 즉, 이 모델은 단순 heatmap supervision만 사용하는 것이 아니라, 최종 좌표 수준에서도 직접 오차를 줄이도록 학습된다. 직관적으로 보면 $L_{\mathrm{MSE}}$는 공간적으로 “어디가 tip인가”를 분포 형태로 학습하게 하고, $L_1$은 최종 좌표를 더 직접적으로 맞추게 만든다. 계수 크기를 보면 heatmap loss의 비중이 매우 크게 설정되어 있는데, 이는 localization prior를 heatmap 수준에서 강하게 주려는 설계로 해석할 수 있다.

입력 마스크는 모두 $512 \times 640$으로 resize된다. 데이터 증강은 random flip과 scale augmentation을 사용한다. 학습은 100 epochs 동안 수행되며 optimizer는 Adam, 초기 learning rate는 $1 \times 10^{-4}$, batch size는 12이다. learning rate schedule은 Cosine Annealing을 사용했다. 실험은 단일 NVIDIA RTX 4090 GPU에서 수행되었다고 기술되어 있다.

비교 대상으로는 image-processing 기반 방법을 사용한다. 이 방법은 EndoVis Challenge 2015 계열의 전통적인 접근을 따른다. 구체적으로 gripper mask에 대해 SVD를 이용해 principal axis를 구하고, wrist에서 가장 먼 축상의 픽셀을 instrument tip으로 간주한다. 이 방법은 딥러닝 없이 mask의 기하학적 구조만으로 tip을 찾는 hand-crafted baseline이다. 논문은 이 방법이 gripper 두 개가 가까워질 때 connected region 분리가 어려워지고, 그 결과 정확한 principal axis를 구하지 못해 오검출이 자주 발생한다고 설명한다.

## 4. 실험 및 결과

실험 데이터는 시뮬레이션 데이터와 실제 데이터로 구성된다. 시뮬레이션 데이터는 저자들의 이전 연구에서 개발한 simulator를 이용해 da Vinci large needle driver CAD model을 다양한 pose에 배치하고, 그에 대응하는 part-level semantic mask를 생성해서 만들었다. 총 9,915개의 simulated data를 생성했다고 한다.

실제 데이터는 SurgPose 데이터 수집 방식으로 일부 real data clip을 모았고, 여기서 **SAM 2**를 사용해 part-level semantic segmentation mask를 추출했다고 기술되어 있다. 즉, 실제 데이터에서는 사람이 직접 전부 mask를 주석한 것이 아니라, foundation model 기반 segmentation 결과를 사용한 것으로 읽힌다. 다만 실제 mask의 품질 관리 방식이나 후처리 여부는 발췌문에 명시되어 있지 않다.

학습에는 mixed dataset을 사용한다. 구성은 시뮬레이션 마스크 8,092장과 실제 마스크 266장이다. 테스트는 시뮬레이션 마스크 1,823장, 실제 마스크 96장에서 수행한다. 데이터 분할 기준이나 환자 단위 분리 여부, clip leakage 방지 여부 같은 실험 프로토콜의 세부 사항은 발췌문만으로는 확인되지 않는다.

평가 지표는 두 가지다.  
첫째는 예측한 tip 위치와 ground truth tip 위치 사이의 **RMSE**이다.  
둘째는 tip detection accuracy인데, 저자들은 **RMSE가 10 pixels 미만이면 accurate**하다고 정의한다.

정량 결과는 매우 명확하다. 비교 대상은 Yang et al. [7]로 표기된 hand-crafted 방법이고, 제안 방법은 ToolTipNet이다.

시뮬레이션 데이터에서 hand-crafted 방법의 RMSE는 28.14, accuracy는 0.287이다. 반면 ToolTipNet은 RMSE 3.73, accuracy 0.959를 기록했다. 이는 오차를 크게 줄였고, 10 pixel 기준 정확도도 압도적으로 향상되었음을 의미한다.

실제 데이터에서도 hand-crafted 방법의 RMSE는 27.64, accuracy는 0.583이고, ToolTipNet은 RMSE 9.04, accuracy 0.813이다. 실제 데이터에서는 시뮬레이션보다 성능이 다소 낮아졌지만, 여전히 baseline보다 크게 우수하다. 특히 RMSE가 10 pixel 이하로 내려왔다는 점은 이 논문이 정의한 accuracy 기준과도 잘 맞아떨어진다.

정성 결과 그림에서도 같은 경향을 보인다고 설명한다. 저자들의 경험적 관찰에 따르면 hand-crafted 방법은 두 gripper가 서로 가까운 경우 잘못된 위치를 자주 예측한다. 이 현상은 SVD 기반 principal axis 추정이 gripper를 둘로 분리하지 못할 때 특히 심해진다. 반면 ToolTipNet은 그런 경우에도 더 안정적으로 tip을 찾는다. 이는 딥러닝 모델이 단순 축 추정보다 더 복잡한 shape pattern을 학습했기 때문으로 이해할 수 있다.

실험 결과의 의미는 분명하다. 이 논문은 “정확한 segmentation mask가 주어졌을 때, tool tip detection은 별도의 복잡한 RGB 기반 모델 없이도 상당히 잘 풀릴 수 있다”는 점을 보인다. 다시 말해, 이 연구의 기여는 end-to-end surgical perception 전체를 푼 것이 아니라, 그 파이프라인 안의 tip localization 문제를 mask 기반으로 강하게 정리한 데 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정이 명확하고 실용적이라는 점이다. 수술 로봇의 kinematics 기반 tip localization이 부정확하다는 실제 문제에서 출발했고, segmentation foundation model의 발전이라는 최근 흐름을 이용해 보다 단순하고 강한 baseline을 제시했다. 특히 입력을 mask로 제한했음에도 실제 데이터에서 hand-crafted 방법을 크게 앞서는 결과를 보여, silhouette 정보만으로도 높은 localization 성능이 가능하다는 점을 실험적으로 설득력 있게 제시한다.

또 다른 강점은 모델 설계가 과도하게 복잡하지 않다는 점이다. HRNet backbone, feature fusion, gripper-guided attention, heatmap prediction이라는 구조는 이해하기 쉽고 재현 가능성도 비교적 높은 편이다. surgical AI 분야에서는 문제 자체가 복잡해서 모델도 복잡해지기 쉬운데, 이 논문은 오히려 segmentation이 확보되었을 때의 tip detection이라는 좁지만 중요한 문제를 깔끔하게 다뤘다.

하지만 한계도 분명하다. 가장 본질적인 한계는 **known segmentation mask에 대한 의존성**이다. 저자들 스스로도 결론에서 이 점을 인정한다. 실제 수술 장면에서는 segmentation이 항상 정확하게 주어지지 않으며, 특히 part-level mask가 안정적으로 나오는지 자체가 또 다른 어려운 문제다. 따라서 현재의 ToolTipNet은 완전한 end-to-end 현실 시스템이라기보다, “좋은 segmentation이 있을 때 사용할 수 있는 tip detector”에 가깝다.

또한 비교 실험이 hand-crafted method 중심이라는 점도 해석에 주의가 필요하다. 논문 서론에서는 여러 deep learning 기반 pose estimation 또는 keypoint detection 연구를 언급하지만, 정량 비교 표에서는 사실상 전통적 baseline 하나와만 직접 비교한다. 따라서 ToolTipNet이 기존의 RGB 기반 deep keypoint detector나 multi-frame model보다도 우수한지는 이 발췌문만으로는 알 수 없다. 즉, 이 논문의 실험은 “딥러닝 mask-based baseline이 hand-crafted mask-based 방법보다 낫다”는 점은 강하게 보여 주지만, broader SOTA 비교까지 충분히 수행했다고 보기는 어렵다.

데이터 규모와 일반화에 대해서도 질문이 남는다. 실제 학습용 real mask가 266장, 테스트용이 96장으로 비교적 작다. 따라서 다양한 수술 도구 종류, 다양한 환자 환경, 조명 변화, occlusion, motion blur 등에 대해 얼마나 일반화되는지는 아직 판단하기 어렵다. 게다가 이 논문은 “for each frame, there is only one instrument with two tips”라고 가정하므로, 다중 도구 상황이나 복잡한 interaction scene으로 바로 확장되는지는 확인되지 않았다.

마지막으로, tip prediction의 안정성이나 temporal consistency도 다뤄지지 않았다. 실제 수술 응용에서는 단일 프레임 정확도뿐 아니라 프레임 간 jitter가 적고 trajectory가 부드러운지도 중요할 수 있는데, 본 논문은 현재까지는 frame-wise detection baseline 성격이 강하다.

## 6. 결론

이 논문은 surgical instrument tip detection 문제를 **segmentation-driven keypoint detection**으로 재정의하고, 이를 위한 간단하지만 효과적인 딥러닝 baseline인 **ToolTipNet**을 제안했다. 핵심은 part-level segmentation mask를 입력으로 사용하고, HRNet 기반 multi-scale feature와 gripper mask 기반 attention을 결합해 두 개의 tool tip heatmap을 예측하는 것이다. 실험에서는 시뮬레이션과 실제 데이터 모두에서 hand-crafted SVD 기반 방법보다 훨씬 낮은 RMSE와 높은 accuracy를 달성했다.

이 연구의 실질적인 의미는, surgical perception pipeline에서 segmentation과 tip localization을 분리했을 때 tip localization만으로도 상당히 강한 성능을 얻을 수 있음을 보여 준 데 있다. 실제 적용 측면에서는 ultrasound-laparoscope registration, surgical skill assessment, imitation learning, robot autonomy 같은 downstream task에서 유용한 중간 모듈이 될 가능성이 있다. 향후 연구 방향으로 저자들이 제안한 것처럼 segmentation과 tip detection을 multi-task learning으로 통합하거나, foundation model backbone을 이용해 mask 의존성을 줄인다면 더 현실적인 end-to-end 수술 지능 시스템으로 발전할 가능성이 크다.
