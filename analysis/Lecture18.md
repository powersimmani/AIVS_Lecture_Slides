# Lecture 18 분석 보고서
## Advanced Unsupervised Learning (Self-Supervised & Contrastive)

**분석 일자**: 2026-02-15
**품질 등급**: A (우수)

---

## 1. 강의 구조 개요

| Part | 주제 | 슬라이드 | 평가 |
|------|------|----------|------|
| Part 1 | Self-Supervised Learning 개요 | 도입 | 우수 |
| Part 2 | Contrastive Learning | SimCLR, MoCo | 매우 우수 |
| Part 3 | 시계열 클러스터링 | DTW, K-Shape | 매우 우수 |
| Part 4 | Autoencoder 변형 | VAE | 우수 |
| Part 5 | Semi-Supervised Learning | Pseudo-labeling | 우수 |

---

## 2. 긍정적 평가

### 2.1 Self-Supervised Learning 개념
- "라벨 없이 라벨 만들기" 개념 명확
- Pretext task 예시:
  - 이미지: Rotation prediction, Jigsaw puzzle
  - 텍스트: MLM, NSP (BERT)
- Representation learning 목적 설명

### 2.2 Contrastive Learning 상세
- **SimCLR**:
  - Positive/Negative pair 개념
  - Data augmentation 중요성
  - Temperature parameter 역할
  - NT-Xent Loss
- **MoCo**:
  - Memory bank/queue 아이디어
  - Momentum encoder
  - SimCLR과 비교

### 2.3 시계열 클러스터링
- **DTW (Dynamic Time Warping)**:
  - 시간 축 정렬 문제 해결
  - 동적 프로그래밍 기반
  - O(n²) 복잡도
- **K-Shape**:
  - Shape-based distance
  - Cross-correlation 기반
  - DTW보다 빠름

### 2.4 VAE (Variational Autoencoder)
- Encoder-Decoder 구조
- Latent space regularization
- Reparameterization trick
- KL divergence + Reconstruction loss

### 2.5 Semi-Supervised Learning
- Pseudo-labeling 기법
- Consistency regularization
- MixMatch, FixMatch 언급

---

## 3. 개선 권장사항

### 3.1 [중요] BYOL / SimSiam 추가

**위치**: Part 2 Contrastive Learning

**현재 상태**: SimCLR, MoCo만 상세

**중요성**:
- Negative sample 없는 방법
- 더 간단한 구조

**추가 권장 내용**:
```markdown
## BYOL (Bootstrap Your Own Latent)

### 핵심 아이디어
- Negative sample 불필요
- 두 네트워크: online + target
- Target은 EMA로 업데이트

### 구조
Online: Encoder → Projector → Predictor
Target: Encoder → Projector (EMA copy)

### 왜 collapse 안 하는가?
- Predictor의 비대칭성
- EMA target의 천천히 변화

### SimSiam (더 단순화)
- EMA 없이 stop-gradient만 사용
- "Simplest Siamese Network"

### 비교
| 방법 | Negative | Memory | 성능 |
|------|----------|--------|------|
| SimCLR | 필요 | 큰 배치 | 높음 |
| MoCo | Queue | 작은 배치 OK | 높음 |
| BYOL | 불필요 | 중간 | 높음 |
| SimSiam | 불필요 | 작음 | 높음 |
```

---

### 3.2 [중요] DINO / DINOv2 추가

**위치**: Part 2 또는 새 섹션

**중요성**:
- Self-supervised Vision Transformer
- 현재 SOTA self-supervised
- Feature extraction에 널리 사용

**추가 권장**:
```markdown
## DINO / DINOv2

### DINO (2021)
- Self-Distillation with Vision Transformer
- Teacher-Student (EMA)
- Centering + Sharpening

### 발견
- Self-supervised ViT가 semantic segmentation 학습
- Attention map이 객체 분할 수행
- ImageNet 라벨 없이!

### DINOv2 (2023)
- 더 큰 데이터셋 (LVD-142M)
- ViT-g (1.1B params)
- 범용 visual feature 추출기

### 사용법
```python
import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
features = model(images)  # [batch, 768]
```

### 응용
- Zero-shot segmentation
- Feature extraction for downstream
- Image retrieval
```

---

### 3.3 [중요] MAE (Masked Autoencoder) 추가

**위치**: Part 4 Autoencoder 또는 Part 1

**중요성**:
- Vision에서의 BERT 스타일 학습
- 간단하면서 효과적
- 2022년 주요 발전

**추가 권장**:
```markdown
## MAE (Masked Autoencoder)

### 아이디어
- 이미지 패치 75% 마스킹
- 보이는 패치만으로 전체 재구성
- BERT의 MLM과 유사

### 왜 효과적인가?
- 이미지: 높은 redundancy
- 75% 마스킹해도 재구성 가능
- 의미 있는 representation 학습 강제

### 아키텍처
Encoder: 보이는 패치만 처리 (효율적)
Decoder: mask token 포함 전체 재구성

### 장점
- Pre-training 빠름 (75% 스킵)
- Contrastive 불필요 (단순)
- 대용량 데이터에 확장 용이

### 성능
ImageNet-1K fine-tune: 87.8% (ViT-H)
```

---

### 3.4 [권장] 시계열 SSL 확장

**위치**: Part 3

**추가 권장**:
```markdown
## 시계열 Self-Supervised Learning

### TS2Vec
- Temporal contrastive learning
- 다양한 granularity에서 학습
- Hierarchical contrastive loss

### TNC (Temporal Neighborhood Coding)
- 시간적 이웃 개념
- Positive: 가까운 시점
- Negative: 먼 시점

### CoST (Contrastive Seasonal)
- 계절성 패턴 학습
- Trend + Seasonal 분리

### 응용
- 시계열 분류
- 이상 탐지
- 예측 (forecasting)
```

---

### 3.5 [권장] VICReg 소개

**위치**: Part 2 끝부분

**추가 권장**:
```markdown
## VICReg (Variance-Invariance-Covariance)

### 문제
- Contrastive: 큰 배치 필요
- BYOL/SimSiam: collapse 이해 어려움

### VICReg 접근
명시적으로 세 가지 목표:
1. **Variance**: 각 차원 분산 유지 (collapse 방지)
2. **Invariance**: augmented view 유사하게
3. **Covariance**: 차원 간 상관 최소화

### 장점
- 수학적으로 명확
- Negative 불필요
- 작은 배치 가능

### Loss
L = λ·var_loss + μ·inv_loss + ν·cov_loss
```

---

### 3.6 [권장] Pseudo-labeling 상세화

**파일**: Part 5

**추가 권장**:
```markdown
## Pseudo-labeling 전략

### 기본 방법
1. Labeled data로 모델 학습
2. Unlabeled data에 예측
3. 높은 confidence 예측을 pseudo-label로
4. 확장된 데이터로 재학습

### FixMatch (2020)
- Weak augmentation: pseudo-label 생성
- Strong augmentation: 학습
- Threshold τ (0.95 일반적)

### 문제점
- Confirmation bias: 틀린 예측 강화
- Class imbalance 악화

### 해결책
- Curriculum: 점진적 threshold 완화
- Distribution alignment
- Meta-learning
```

---

## 4. 기술적 정확성 검증 결과

| 항목 | 슬라이드 | 검증 |
|------|----------|------|
| SimCLR NT-Xent loss | - | ✅ 정확 |
| MoCo momentum update | - | ✅ 정확 |
| DTW 동적 프로그래밍 | - | ✅ 정확 |
| VAE ELBO | - | ✅ 정확 |
| Reparameterization trick | - | ✅ 정확 |
| Pseudo-labeling 개념 | - | ✅ 정확 |

---

## 5. 우선순위별 작업 체크리스트

### 다음 업데이트 시 (중요)
- [ ] BYOL / SimSiam 추가
- [ ] DINO / DINOv2 추가
- [ ] MAE (Masked Autoencoder) 추가

### 시간 있을 때 (권장)
- [ ] 시계열 SSL 확장
- [ ] VICReg 소개
- [ ] Pseudo-labeling 상세화

### 선택적 개선
- [ ] SwAV (Clustering-based) 언급
- [ ] Audio SSL (wav2vec) 언급
- [ ] Video SSL 간략 소개

---

## 6. 다른 강의와의 연계성

| 연계 강의 | 관련 내용 | 상태 |
|-----------|-----------|------|
| Lecture 17 (클러스터링) | 비지도 기초 | ✅ 좋은 연결 |
| Lecture 14 (PLM) | BERT MLM | ✅ 연계됨 |
| Lecture 11-12 (시퀀스) | 시계열 기초 | ✅ 연계됨 |
| Lecture 15-16 (생성) | VAE, Autoencoder | ✅ 연계됨 |
| Lecture 10 (아키텍처) | ResNet backbone | ✅ 연계됨 |

---

## 7. 특별 참고사항

### Self-Supervised Learning 진화
```
2018: Pretext tasks (rotation, jigsaw)
2019: MoCo, CPC
2020: SimCLR, BYOL, SwAV
2021: DINO, MAE
2022-: DINOv2, I-JEPA
```

### Contrastive Learning 핵심 요소
```python
# SimCLR 스타일 구현 핵심
def contrastive_loss(z_i, z_j, temperature=0.5):
    # z_i, z_j: augmented view의 representation
    batch_size = z_i.shape[0]

    # 모든 쌍의 similarity 계산
    z = torch.cat([z_i, z_j], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

    # Positive pairs
    sim_ij = torch.diag(sim, batch_size)
    sim_ji = torch.diag(sim, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    # NT-Xent loss
    nominator = torch.exp(positives / temperature)
    denominator = torch.sum(torch.exp(sim / temperature), dim=1) - 1

    return -torch.mean(torch.log(nominator / denominator))
```

### Data Augmentation 중요성
SimCLR 연구에서 발견:
- Random crop + resize (필수)
- Color jitter (중요)
- Gaussian blur (도움)
- 조합이 핵심 (단일 aug 불충분)

---

## 8. 참고 자료

- [SimCLR Paper (2020)](https://arxiv.org/abs/2002.05709)
- [MoCo Paper (2019)](https://arxiv.org/abs/1911.05722)
- [BYOL Paper (2020)](https://arxiv.org/abs/2006.07733)
- [SimSiam Paper (2020)](https://arxiv.org/abs/2011.10566)
- [DINO Paper (2021)](https://arxiv.org/abs/2104.14294)
- [MAE Paper (2021)](https://arxiv.org/abs/2111.06377)
- [FixMatch Paper (2020)](https://arxiv.org/abs/2001.07685)
- [DTW Tutorial](https://rtavenar.github.io/blog/dtw.html)
- [Lightly Library (SSL)](https://github.com/lightly-ai/lightly)
