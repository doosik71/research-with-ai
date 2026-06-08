# VitalDB, a high-fidelity multiparameter vital signs database in surgical patients

* **저자**: Hyung-Chul Lee, Yoonsang Park, Soo Bin Yoon, Seong Mi Yang, Dongnyeok Park, Chul-Woo Jung
* **발표연도**: 2022

## 1. 논문 개요

이 논문은 수술 및 마취 중 수집되는 고해상도 다중 생체신호 데이터를 공개 데이터셋으로 구축한 **VitalDB**를 소개하는 데이터 논문이다. VitalDB는 수술 환자의 실시간 생체신호를 기반으로 한 machine learning 연구를 촉진하기 위해 설계되었으며, 6,388건의 수술 사례에서 수집된 waveform 및 numeric data track을 포함한다. 데이터는 196개의 수술 중 모니터링 parameter, 73개의 perioperative clinical parameter, 34개의 time-series laboratory result parameter로 구성되어 있다.

이 논문의 핵심 연구 문제는 “수술 중 환자의 동적인 생리 상태를 machine learning으로 분석하기 위해 사용할 수 있는 대규모, 고해상도, 다중 parameter 생체신호 데이터셋을 어떻게 구축하고 공개할 것인가”이다. 기존의 EMR 또는 automated anesthesia record는 대부분 낮은 시간 해상도의 numeric data만 저장하며, electrocardiography, photoplethysmography, electroencephalography, airway pressure waveform과 같은 중요한 waveform data는 비용 및 기술적 한계로 충분히 저장하지 못한다. 또한 여러 마취 장비에서 나오는 신호를 시간 동기화하여 통합적으로 기록하는 기능도 제한적이다.

이 문제는 임상적으로도 중요하다. 수술 중 vital signs는 환자의 생리적 상태를 반영하는 핵심 정보이며, 저혈압, 저체온, 낮은 cardiac output, 마취 깊이 변화 등은 수술 후 합병증이나 사망률과 관련될 수 있다. 그러나 이러한 신호는 복잡한 time-series 형태로 나타나며, 여러 신호 사이의 상호작용까지 고려해야 하므로 숙련된 마취과 의사도 해석이 쉽지 않다. 따라서 고품질의 대규모 데이터셋은 biosignal analysis, clinical decision support, intraoperative event prediction, monitoring algorithm validation 등 다양한 연구의 기반이 된다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 수술실에서 실제로 사용되는 여러 마취 및 모니터링 장비의 신호를 **Vital Recorder**라는 소프트웨어를 통해 자동으로, 고해상도로, 시간 동기화하여 수집하고, 이를 환자의 perioperative clinical information 및 laboratory results와 결합하여 공개 데이터셋으로 제공하는 것이다.

기존 데이터셋과 비교했을 때 VitalDB의 차별점은 크게 세 가지로 정리할 수 있다. 첫째, 이 데이터셋은 ICU나 일반 병동 데이터가 아니라 **perioperative patient care**, 특히 수술 및 마취 중의 생체신호에 초점을 맞춘다. 둘째, 단순한 저해상도 numeric vital sign뿐 아니라 arterial pressure waveform, ECG, plethysmogram, EEG, airway pressure waveform 등 machine learning에 직접 활용할 수 있는 waveform data를 포함한다. 셋째, 여러 장비에서 얻은 신호가 하나의 case file 안에 time-synchronized data track으로 저장되어 있어, 서로 다른 생리 신호 간의 동적 관계를 분석할 수 있다.

논문에서 강조하는 또 하나의 중요한 관점은 데이터가 의도적으로 과도하게 전처리되지 않았다는 점이다. 실제 마취 환경의 생체신호에는 sensor detachment, blood sampling 중 arterial pressure 이상값, electrocautery나 electrophysiologic monitoring으로 인한 ECG 및 EEG noise 등이 포함된다. 저자들은 이러한 noise를 제거하지 않고 남겨 두었는데, 이는 실제 임상 환경에서 작동 가능한 robust monitoring algorithm을 개발하려면 현실적인 noise를 포함한 데이터가 필요하기 때문이다.

## 3. 상세 방법 설명

VitalDB의 전체 파이프라인은 크게 네 단계로 구성된다. 첫째, 수술 중 여러 마취 장비에서 고해상도 생체신호를 자동 수집한다. 둘째, 수집된 vital file을 EMR의 수술 스케줄과 대조하여 case matching을 수행한다. 셋째, 데이터 track의 무결성과 유효성을 검증하고 불필요하거나 무효한 track을 정리한다. 넷째, EMR에서 얻은 perioperative clinical information 및 laboratory results를 결합한 뒤 de-identification을 수행하여 공개 데이터셋으로 배포한다.

데이터 수집에는 저자들이 이전에 개발한 **Vital Recorder** 프로그램이 사용되었다. 이 프로그램은 patient monitor, anesthesia machine, brain monitor, cardiac monitor, target-controlled infusion pump, rapid infusion system 등 다양한 장비에서 나오는 데이터를 하나의 case file에 시간 동기화하여 저장한다. 논문에서 사용된 Vital Recorder 버전은 1.7.4이다. 프로그램이 실행되는 laptop computer는 serial cable을 통해 여러 모니터링 장비에 연결되었고, 31개 수술실 중 10개 수술실에 동일한 기록 시스템이 설치되어 약 1년간 24시간 운영되었다.

case recording은 자동으로 시작되고 종료되었다. heart rate와 percutaneous oxygen saturation 신호가 동시에 감지되면 환자 모니터링이 시작된 것으로 판단하여 case recording을 시작했다. 반대로 heart rate와 oxygen saturation signal 입력이 10분 이상 사라지면 환자 모니터링이 종료된 것으로 보고 recording을 자동 중지했다. 이 방식은 수술실의 일상 workflow에 맞게 데이터를 대규모로 수집할 수 있게 해 준다.

연구 대상은 2016년 8월부터 2017년 6월까지 서울대학교병원에서 non-cardiac surgery를 받은 환자이다. 대상 수술은 general, thoracic, urological, gynecological surgery를 포함한다. 전체 eligible case는 7,051건이었으며, local anesthesia 239건, incomplete recording 279건, essential data track loss 145건을 제외한 뒤 최종적으로 6,388건이 데이터셋에 포함되었다. 포함된 anesthesia type은 general anesthesia, spinal anesthesia, sedation/analgesia이다.

데이터 track processing 과정도 명시되어 있다. 모든 값이 0이거나 sample 수가 10개 미만인 track은 삭제되었다. 대응되는 numeric track이 없는 waveform track도 삭제되었다. 데이터 사용성을 높이기 위해 track name이 변경되었으며, 예를 들어 femoral arterial catheter가 anesthesia record에서 확인된 경우 ART 관련 waveform 및 numeric track이 FEM 관련 이름으로 바뀌었다. 또한 PUMP_RATE와 PUMP_VOL처럼 일반적인 pump 이름으로 되어 있던 track은 infusion pump data나 anesthesia record에서 얻은 약물명에 따라 EPI_RATE, PPF20_VOL과 같은 구체적 이름으로 바뀌었다.

de-identification은 환자 식별 정보를 제거하고 시간 정보를 상대 시간으로 변환하는 방식으로 이루어졌다. 실제 환자 번호 대신 caseid가 1부터 6,388까지 무작위로 부여되었고, 재수술 환자 식별을 위해 subjectid도 1부터 6,090까지 부여되었다. recording start time은 항상 0으로 설정되었으며, surgery start/end time과 anesthesia start/end time은 casestart를 기준으로 한 상대 시간으로 변환되었다. 따라서 각 time-series data는 실제 날짜와 시간이 아니라 case 시작점으로부터 몇 초가 지났는지를 나타낸다.

데이터는 세 가지 주요 구성으로 제공된다. 첫째, 6,388개의 vital file이 있으며, 이는 intraoperative vital signs data를 담고 있다. 둘째, clinical information.csv에는 caseid, subjectid, demographic data, surgery and anesthesia data, outcome data, preoperative laboratory data 등이 포함된다. 셋째, lab results.csv에는 수술 전후 90일 이내의 34개 blood test 결과가 time-series 형태로 저장되어 있다.

Vital signs data는 최대 12개의 waveform track과 184개의 numeric track으로 구성될 수 있으며, 전체 track 수는 486,451개이다. case마다 사용된 장비가 다르므로 각 case file에 포함된 parameter 수는 16개에서 129개까지 다르다. numeric data의 시간 해상도는 장비에 따라 1초에서 7초 간격이며, waveform data는 62.5Hz에서 500Hz의 sampling rate를 가진다. 평균적으로 각 case file은 약 280만 개의 data point를 포함한다.

주요 장비와 역할은 다음과 같이 해석할 수 있다. Solar 8000M patient monitor는 모든 환자에서 heart rate, blood pressure, oxygen saturation, temperature, gas concentration 등 주요 numeric vital sign을 기록했다. TramRac-4A와 저자들이 개발한 analog-to-digital converter는 ECG, capnography, plethysmogram, blood pressure waveform 등을 기록했다. Primus anesthesia machine은 gas concentration, volume, flow, airway pressure 등을 기록했다. BIS Vista는 EEG waveform과 bispectral index 관련 parameter를 수집했다. Orchestra target-controlled infusion pump는 propofol, remifentanil 등 정맥마취 약물의 target concentration, plasma concentration, effect-site concentration, infusion volume, infusion rate 등을 기록했다. Vigileo, EV1000, Vigilance II, CardioQ-ODM+ 등 cardiac output monitor는 stroke volume, cardiac output 및 관련 parameter를 제공했다.

이 논문은 특정 deep learning model을 제안하는 논문이 아니므로, 새로운 neural network architecture, loss function, optimization objective, training algorithm은 제시하지 않는다. 대신 데이터셋이 향후 deep learning 및 machine learning 연구에 어떻게 사용될 수 있는지를 설명한다. 예를 들어 arterial pressure waveform 기반 cardiac output algorithm, propofol 및 remifentanil의 pharmacokinetic-pharmacodynamic modeling, bispectral index prediction, intraoperative hypotension prediction, mortality prediction 등이 응용 사례로 언급된다.

수학적 방정식은 논문 본문에 핵심 방법론으로 제시되어 있지 않다. 다만 데이터의 시간 정렬 관점에서 모든 time-series data는 case recording 시작 시점인 $t = 0$을 기준으로 상대 시간 $t$로 표현된다고 이해할 수 있다. 예를 들어 어떤 생체신호 track은 시간과 값의 쌍 $(t_i, x_i)$로 나타낼 수 있으며, 여러 장비에서 수집된 track은 동일한 casestart 기준으로 정렬된다. waveform data의 경우 web-based API에서 start time, time interval, end time이 제공되며, 각 sample의 시간은 일정한 간격으로 증가하는 monotonic sequence로 복원된다. 이는 여러 생체신호를 같은 시간축 위에 올려 놓고 분석할 수 있게 하는 핵심 설계이다.

사용 방법도 두 가지로 제공된다. 첫째, OSF repository에서 전체 데이터를 다운로드하여 Python package인 vitaldb를 사용할 수 있다. `load_case` 함수는 특정 caseid와 track name 목록을 입력받아 원하는 시간 간격으로 track data를 불러온다. `VitalFile` class는 vital file을 읽고, 다시 vital format으로 저장하거나, numpy array 또는 pandas DataFrame으로 변환하는 기능을 제공한다. 둘째, web-based API를 통해 clinical information, track list, individual track data, laboratory results를 CSV 또는 compressed CSV 형태로 받을 수 있다. 연구자는 먼저 clinical information을 바탕으로 inclusion/exclusion criteria를 정하고, 필요한 track name을 확인한 뒤, track list에서 tid를 찾아 실제 time-value data를 다운로드할 수 있다.

## 4. 실험 및 결과

이 논문은 새로운 예측 모델의 성능을 평가하는 일반적인 machine learning 논문이라기보다, 데이터셋의 구축과 검증을 보고하는 **Data Descriptor** 논문이다. 따라서 실험 및 결과는 model accuracy, AUC, F1 score와 같은 성능 지표보다 데이터셋의 규모, 구성, 품질 검증, 활용 가능성을 중심으로 제시된다.

데이터셋 규모 측면에서 VitalDB는 총 6,388건의 수술 사례를 포함한다. 수술 분야별로는 general surgery가 4,930건으로 가장 많고, thoracic surgery가 1,111건, urology가 117건, gynecology가 230건이다. 전체 환자의 51%가 남성이며, 전체 연령의 median은 59세, interquartile range는 48세에서 68세이다. 전체 수술 접근 방식은 open surgery가 53%, videoscopic surgery가 43%, robotic surgery가 4%로 보고되었다. anesthesia type은 general anesthesia가 95%로 대부분을 차지하며, spinal anesthesia가 4%, sedation/analgesia가 1%이다.

마취 및 수술 시간도 데이터셋의 특성을 보여준다. 전체 anesthesia duration의 median은 150분, surgery duration의 median은 110분, data recording duration의 median은 165분이다. 이는 단순한 짧은 생체신호 segment가 아니라 실제 수술 전후의 비교적 긴 intraoperative trajectory를 포함한다는 점에서 의미가 있다.

장비 사용률을 보면 Solar 8000M은 모든 case에서 사용되었고, Primus anesthesia machine도 거의 모든 case에서 사용되었다. BIS Vista는 5,566건, Orchestra target-controlled infusion pump는 4,927건에서 사용되었다. Cardiac output monitor나 cerebral/somatic oximeter, rapid infusion system은 임상의 판단에 따라 일부 case에서만 사용되었다. 이로 인해 VitalDB는 모든 case가 동일한 feature set을 가지는 정형 데이터셋이 아니라, 실제 임상 환경처럼 case마다 존재하는 track이 다른 heterogeneous dataset이다.

데이터 품질 검증은 두 가지 관점에서 수행되었다. 첫째, case matching 검증이다. recording 중 장비 연결 상태는 real-time remote monitoring으로 확인되었고, 수술 후 자동 기록된 case file은 EMR에서 얻은 operation schedule과 주 단위로 대조되었다. 파일명에는 operating room name, recording date, recording time이 포함되므로 해당 환자 수술과 연결할 수 있었고, recording time과 실제 operation time을 비교하여 matching을 확인했다.

둘째, vital file integrity 검증이다. case-matched file은 Vital Recorder program으로 다시 불러와 네 명의 마취과 의사가 시각적으로 확인했다. invalid data track은 제거되었고, total intravenous anesthesia 중 inhalation anesthesia 관련 parameter처럼 의미 없는 track도 삭제되었다. 대응되는 numeric value가 없는 waveform track도 제거되었다. 그러나 실제 임상 환경에서 발생하는 noise는 의도적으로 보존되었다. 예를 들어 BIS, cerebral oximeter, ECG, plethysmography의 temporary sensor detachment로 인한 data loss, blood sampling 중 arterial pressure abnormal value, electrocautery 중 ECG 및 EEG noise 등이 그대로 포함된다.

논문은 VitalDB가 이미 여러 연구에 사용되었다고 보고한다. 예시로 arterial pressure waveform 기반 cardiac output algorithm, intravenous anesthetics의 pharmacokinetic-pharmacodynamic study, bispectral index algorithm 분석, intraoperative bispectral index와 postoperative mortality의 관계 분석, arterial waveform 기반 intraoperative hypotension prediction 등이 언급된다. 이는 데이터셋이 단순히 공개된 것에 그치지 않고 실제 biosignal machine learning 연구에 활용될 수 있음을 보여준다.

정량적 model 성능 비교는 이 논문의 주요 목적이 아니므로, 특정 baseline model 대비 성능 향상 수치나 ablation study는 제공되지 않는다. 따라서 이 논문을 algorithm paper로 읽기보다는, 수술 중 고해상도 multi-parameter biosignal research를 가능하게 하는 infrastructure paper 또는 benchmark resource paper로 이해하는 것이 적절하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 실제 수술 및 마취 환경에서 수집된 대규모 고해상도 생체신호 데이터를 공개했다는 점이다. 생체신호 machine learning 연구에서 가장 큰 병목 중 하나는 고품질 데이터 접근성인데, VitalDB는 waveform data와 numeric data를 함께 제공함으로써 이 문제를 크게 완화한다. 특히 arterial pressure waveform, ECG, EEG, plethysmogram, airway pressure waveform 같은 고주파 signal은 기존 EMR 기반 데이터셋에서 얻기 어려운 경우가 많다.

두 번째 강점은 여러 장비의 data track이 time-synchronized 형태로 저장되어 있다는 점이다. 수술 중 환자 상태는 단일 vital sign만으로 설명되기 어렵다. 예를 들어 혈압 변화는 마취 깊이, 약물 주입 속도, 수액 투여, 출혈량, 심박수, 산소포화도, ventilator parameter 등과 함께 해석되어야 한다. VitalDB는 이러한 multi-modal 또는 multi-parameter temporal relationship을 연구할 수 있는 기반을 제공한다.

세 번째 강점은 clinical information과 laboratory results가 함께 제공된다는 점이다. 단순한 signal processing 연구뿐 아니라 acute kidney injury, length of hospital stay, ICU stay, in-hospital mortality 같은 clinical outcome과 intraoperative biosignal 사이의 관계를 분석할 수 있다. 이는 algorithm development뿐 아니라 clinical decision support 연구에도 중요하다.

네 번째 강점은 데이터 사용성을 높이기 위한 API와 Python library를 제공한다는 점이다. vital file format만 공개하는 것이 아니라 `vitaldb` package, web-based API, sample codes를 함께 제공함으로써 연구자가 데이터를 쉽게 불러오고 분석할 수 있도록 했다. 이는 데이터셋의 실제 활용 가능성을 높이는 중요한 요소이다.

그러나 한계도 명확하다. 논문이 직접 언급한 가장 중요한 한계는 데이터가 단일 기관에서 수집되었고, 환자군이 단일 인종, 즉 Asian population에 한정되어 있다는 점이다. 이로 인해 데이터셋으로 학습한 algorithm이 다른 병원, 다른 장비 구성, 다른 인종 또는 다른 임상 workflow에서도 동일하게 일반화될지는 보장되지 않는다. 저자들은 이 문제가 overfitting으로 이어질 수 있다고 명시하며, multicenter biosignal research가 필요하다고 설명한다.

또 다른 한계는 데이터가 실제 임상 환경을 반영하기 때문에 missing data, noise, 장비별 sampling interval 차이, case별 track 구성 차이가 존재한다는 점이다. 이는 실용적 algorithm 개발에는 장점이지만, 연구자가 분석을 수행할 때는 preprocessing, synchronization, missing value handling, artifact detection을 신중하게 설계해야 한다. 특히 waveform track과 numeric track의 sampling rate가 서로 다르고, numeric data에서는 missing row가 제거되어 time interval이 일정하지 않을 수 있으므로 naive한 resampling은 잘못된 결론을 만들 수 있다.

또한 논문은 데이터셋 구축 논문이기 때문에, 특정 clinical prediction task에 대한 표준 train/validation/test split, benchmark baseline, 평가 프로토콜을 중심적으로 제시하지 않는다. 따라서 후속 연구자가 VitalDB를 이용해 model을 개발할 때는 cohort selection, label definition, leakage 방지, temporal validation strategy 등을 직접 엄밀하게 설계해야 한다. 예를 들어 postoperative outcome을 예측하는 연구에서는 수술 후 laboratory result나 discharge 정보가 feature에 섞이지 않도록 주의해야 한다.

비판적으로 보면, VitalDB는 high-fidelity dataset이라는 점에서 매우 가치가 크지만, 바로 “일반화 가능한 의료 AI 모델”을 보장하는 데이터셋은 아니다. 공개된 데이터셋의 크기와 해상도는 우수하지만, 단일 병원 기반이라는 구조적 한계와 실제 임상 noise를 포함한 복잡성 때문에, 모델 개발보다도 데이터 이해와 검증 설계가 연구 성패를 좌우할 가능성이 크다. 그럼에도 이 논문은 perioperative biosignal AI 연구에서 매우 중요한 공공 인프라를 제공했다는 점에서 기여가 분명하다.

## 6. 결론

이 논문은 수술 환자의 고해상도 multi-parameter vital signs data를 대규모로 수집하고 공개한 VitalDB 데이터셋을 소개한다. VitalDB는 6,388건의 수술 사례, 486,451개의 waveform 및 numeric data track, 196개의 intraoperative monitoring parameter, 73개의 perioperative clinical parameter, 34개의 time-series laboratory result parameter를 포함한다. 데이터는 Vital Recorder를 통해 여러 마취 장비에서 시간 동기화되어 수집되었고, EMR 기반 clinical information 및 laboratory results와 결합되었다.

이 연구의 주요 기여는 새로운 deep learning architecture를 제안한 것이 아니라, 수술 중 생체신호 machine learning 연구를 가능하게 하는 공개 데이터 인프라를 구축했다는 데 있다. VitalDB는 intraoperative hypotension prediction, arterial waveform 기반 cardiac output estimation, anesthesia depth modeling, drug effect estimation, postoperative outcome prediction 등 다양한 연구에 활용될 수 있다.

실제 적용 측면에서 VitalDB는 수술 중 환자 모니터링 algorithm, clinical decision support system, robust biosignal preprocessing method, multimodal temporal modeling 연구의 기반이 될 가능성이 높다. 다만 단일 기관 및 단일 인종 데이터라는 한계가 있으므로, VitalDB로 개발된 모델은 외부 기관 및 다양한 환자군에서 추가 검증되어야 한다. 향후 multicenter dataset과 결합된다면, VitalDB는 perioperative AI 연구에서 일반화 가능한 생체신호 알고리즘 개발의 중요한 출발점이 될 수 있다.
