---
marp: true
title: Adversarially Robust One-class Novelty Detection
paginate: true
---

## Adversarially Robust One-class Novelty Detection

- Shao-Yuan Lo, Poojan Oza, Vishal M. Patel
- IEEE TPAMI 2022 (arXiv:2108.11168)
- 발표자료

---

## 문제 배경

- 원클래스 novelty detection: 정상 클래스만 학습, 신규 클래스 탐지
- AE 기반 모델이 주류
- 딥러닝 모델은 적대적 공격에 취약
- 기존 방어는 분류 문제 중심 → 원클래스 환경에서 효과 제한

---

## 연구 질문

- 원클래스 novelty detector는 적대적 공격에 얼마나 취약한가?
- 기존 방어(AT, FD 등)는 충분한가?
- 원클래스 특성을 활용한 방어 설계가 가능한가?

---

## 핵심 아이디어: PrincipaLS

- 원클래스 문제는 정상 클래스 정보만 유지하면 됨
- 잠재 공간을 크게 조작해도 성능 저하가 적음
- 잠재 공간을 PCA로 정제하여 공격 노이즈 제거

---

## PrincipaLS 구조

- Vector-PCA: 채널(벡터) 차원 정제
- Spatial-PCA: 공간 차원 정제
- 순차 PCA 후 역변환 → principal latent space
- 디코더는 정제된 잠재공간으로 재구성

---

## 공격 설정

- AE 재구성 오류를 조작하는 공격 정의
- 정상 데이터: 재구성 오류 증가
- 이상 데이터: 재구성 오류 감소
- PGD, FGSM, MI-FGSM, MultAdv, AF 등 8개 공격

---

## 실험 구성

- 데이터셋: MNIST, Fashion-MNIST, CIFAR-10, MVTec-AD, UCSD Ped2
- 모델: AE, VAE, AAE, ALOCC, GPND, ARAE, OC-GAN 등 7종
- 지표: AUROC, FPR@95%TPR

---

## 주요 결과

- 기존 모델은 공격 시 AUROC 급락
- PrincipaLS는 다양한 공격/데이터셋에서 AUROC 유지
- PGD-AT, FD 등 기존 방어 대비 일관된 개선
- Clean 성능 저하 없이 robust 성능 향상

---

## 추가 분석

- VQ-VAE 기반 방어 대비 PrincipaLS 우수
- 공격 강도 증가 시에도 안정적
- 원클래스 특성에 맞는 방어 설계의 유효성 확인

---

## 한계

- 원클래스 novelty detection에 특화
- PCA 기반 정제의 표현력 한계 가능
- 다른 신뢰성 요소(공정성/프라이버시)와 통합 미검증

---

## 결론

- 원클래스 novelty detector의 공격 취약성 체계 분석
- PrincipaLS: 잠재공간 PCA 정제로 robust novelty detection 달성
- 광범위 벤치마크에서 일관된 성능 향상

---

## 참고

- arXiv: https://arxiv.org/abs/2108.11168
- 코드: https://github.com/shaoyuanlo/PrincipaLS
