# Lecture 15 분석 보고서
## Generative Models - GAN

**분석 일자**: 2026-02-15
**품질 등급**: A+ (매우 우수)

---

## 1. 강의 구조 개요

| Part | 주제 | 슬라이드 | 평가 |
|------|------|----------|------|
| Part 1 | 도입 및 동기 | 03-06 | 매우 우수 |
| Part 2 | 수학적 기초 | 07-12 | 매우 우수 |
| Part 3 | 학습 알고리즘 | 13-18 | 매우 우수 |
| Part 4 | 핵심 과제 | 19-24 | 매우 우수 |
| Part 5 | 개선 기법 | 25-29 | 매우 우수 |
| Part 6 | 실습 및 응용 | 30-33 | 우수 |

---

## 2. 긍정적 평가

### 2.1 직관적 비유 (Part 1)
- 위조범(Generator) vs 경찰(Discriminator) 비유
- 생성 모델의 필요성 명확히 설명
- VAE, Flow, GAN 비교

### 2.2 수학적 기초 상세 (Part 2)
- Minimax 게임 공식화:
  - min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]
- 최적 Discriminator 유도:
  - D*(x) = p_data(x) / (p_data(x) + p_g(x))
- 전역 최적점에서 Jensen-Shannon Divergence
- 확률 분포 관점 설명

### 2.3 학습 알고리즘 상세 (Part 3)
- 교대 학습 과정 시각화
- Gradient Flow 분석
- Non-Saturating Loss 중요성:
  - 기존: min_G log(1-D(G(z))) → gradient 소실
  - 개선: max_G log(D(G(z))) → 안정적 학습
- 실용적 팁 제공

### 2.4 핵심 문제점 상세 (Part 4)
- **Mode Collapse**: 다양성 부족
- **학습 불안정성**: D와 G 균형
- **평가 어려움**: IS, FID 언급
- **Vanishing Gradient**: 상세 설명
- **실패 패턴**: 시각적 예시

### 2.5 개선 기법 (Part 5)
- **DCGAN (2015)**: Conv 아키텍처 가이드라인
- **cGAN**: 조건부 생성
- **WGAN**: Wasserstein 거리, 안정적 학습
- 기타 개선: Label Smoothing, Spectral Norm

---

## 3. 개선 권장사항

### 3.1 [중요] StyleGAN 상세 추가

**위치**: Part 5 개선 기법

**현재 상태**: 언급만 됨

**중요성**:
- 고해상도 이미지 생성의 SOTA
- 현재까지 널리 사용

**추가 권장 내용**:
```markdown
## StyleGAN (2018-2021)

### StyleGAN v1 핵심 아이디어
1. **Mapping Network**: z → w (intermediate latent)
2. **AdaIN (Adaptive Instance Norm)**: style 주입
3. **Progressive Growing**: 저해상도 → 고해상도

### StyleGAN v2 개선
- AdaIN → Weight Demodulation
- Perceptual Path Length regularization
- No progressive growing

### StyleGAN v3 (2021)
- Alias-free: translation equivariance
- 더 자연스러운 이미지 변환

### 핵심 수식
style injection: y = γ(w) * norm(x) + β(w)

### 응용
- 얼굴 생성 (thispersondoesnotexist.com)
- 도메인별 생성 (차량, 건물 등)
- Image editing (latent manipulation)

### Latent Space 특성
- W space: disentangled representation
- Style mixing: 여러 w 조합 가능
```

---

### 3.2 [중요] FID/IS 평가 지표 상세 추가

**위치**: Part 4 또는 Part 6

**현재 상태**: 평가 어려움만 언급

**추가 권장 내용**:
```markdown
## GAN 평가 지표

### Inception Score (IS)
IS = exp(E_x[KL(p(y|x) || p(y))])

- p(y|x): 생성 이미지의 class 분포 (sharp)
- p(y): 전체 class 분포 (uniform 선호)
- 높을수록 좋음 (품질 + 다양성)

### Fréchet Inception Distance (FID)
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_rΣ_g)^0.5)

- Inception v3의 특징 공간에서 비교
- 실제 vs 생성 분포의 거리
- 낮을수록 좋음

### 비교
| 지표 | 측정 | 장점 | 단점 |
|------|------|------|------|
| IS | 품질+다양성 | 빠름 | 분포 비교 안함 |
| FID | 분포 거리 | 이론적 | 계산 비용 |
| Precision | 품질 | 직관적 | 다양성 무시 |
| Recall | 다양성 | 직관적 | 품질 무시 |

### 권장
- FID를 주 지표로 사용
- IS는 보조 지표
```

---

### 3.3 [중요] GAN Inversion 추가

**위치**: Part 6 응용

**중요성**:
- 실제 이미지 편집에 필수
- 최신 GAN 응용 분야

**추가 권장**:
```markdown
## GAN Inversion

### 개념
- 실제 이미지 → latent code 역산
- 이미지 편집 가능하게 함

### 방법
1. **Optimization-based**
   - 직접 최적화: min ||G(w) - x||
   - 정확하지만 느림

2. **Encoder-based**
   - Encoder E: x → w
   - 빠르지만 부정확

3. **Hybrid**
   - Encoder 초기화 + Optimization
   - 균형 잡힌 접근

### 응용
- 얼굴 편집 (나이, 표정, 포즈)
- Style Transfer
- Image restoration
```

---

### 3.4 [권장] Spectral Normalization 상세

**파일**: 슬라이드 29 (Other Improvements)

**추가 권장**:
```markdown
## Spectral Normalization

### 아이디어
- Weight matrix의 spectral norm 제한
- Lipschitz constant = 1 유지

### 수식
W_SN = W / σ(W)

where σ(W) = max singular value

### Power Iteration
- 매 iteration σ(W) 근사
- 계산 효율적

### 효과
- Discriminator 안정화
- WGAN-GP의 gradient penalty 대안
- 구현 간단

### PyTorch
```python
from torch.nn.utils import spectral_norm
conv = spectral_norm(nn.Conv2d(3, 64, 3))
```
```

---

### 3.5 [권장] BigGAN 언급

**위치**: Part 5

**추가 권장**:
```markdown
## BigGAN (2018)

### 핵심
- 대규모 배치 (2048)
- Class-conditional generation
- Truncation trick

### Truncation Trick
z ~ N(0, I) → z ~ N(0, truncation)

| truncation | 다양성 | 품질 |
|------------|--------|------|
| 1.0 | 높음 | 보통 |
| 0.5 | 중간 | 높음 |
| 0.0 | 없음 | 최고 |

### Self-Attention
- Long-range dependency 포착
- ImageNet 품질 대폭 향상
```

---

## 4. 기술적 정확성 검증 결과

| 항목 | 슬라이드 | 검증 |
|------|----------|------|
| Minimax 목적 함수 | 09-10 | ✅ 정확 |
| 최적 D* 수식 | 11 | ✅ 정확 |
| JS Divergence 연결 | 12 | ✅ 정확 |
| Non-saturating loss | 17 | ✅ 정확 |
| DCGAN 가이드라인 | 26 | ✅ 정확 |
| WGAN Earth Mover's distance | 28 | ✅ 정확 |
| Mode Collapse 정의 | 20 | ✅ 정확 |

---

## 5. 우선순위별 작업 체크리스트

### 다음 업데이트 시 (중요)
- [ ] StyleGAN 시리즈 상세 추가
- [ ] FID/IS 평가 지표 상세 추가
- [ ] GAN Inversion 추가

### 시간 있을 때 (권장)
- [ ] Spectral Normalization 상세
- [ ] BigGAN 언급
- [ ] Progressive GAN 설명

### 선택적 개선
- [ ] Pix2Pix, CycleGAN 이미지 변환
- [ ] Neural Style Transfer와 연결
- [ ] GAN vs Diffusion 비교

---

## 6. 다른 강의와의 연계성

| 연계 강의 | 관련 내용 | 상태 |
|-----------|-----------|------|
| Lecture 16 (Diffusion) | 생성 모델 대안 | ✅ 좋은 연결 |
| Lecture 10 (아키텍처) | CNN 구조 | ✅ 기초 제공됨 |
| Lecture 09 (정규화) | Spectral Norm | ✅ 연계됨 |
| Lecture 08 (최적화) | Adam, Learning Rate | ✅ 연계됨 |
| Lecture 17 (비지도) | 생성 모델 연결 | ✅ 연계됨 |

---

## 7. 특별 참고사항

### GAN의 역사적 위치
- 2014년 Ian Goodfellow 제안
- "Adversarial" 패러다임의 시작
- 2014-2020 이미지 생성 주류
- 2020년 이후 Diffusion에 밀림

### GAN 학습 팁 요약
```python
# 권장 설정
optimizer_G = Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Label smoothing
real_labels = torch.full((batch_size,), 0.9)  # not 1.0
fake_labels = torch.full((batch_size,), 0.0)

# Two Timescale Update Rule (TTUR)
# D 학습률 > G 학습률
```

### GAN vs Diffusion 현재 상황
| 항목 | GAN | Diffusion |
|------|-----|-----------|
| 생성 속도 | 빠름 (1-shot) | 느림 (1000 steps) |
| 학습 안정성 | 불안정 | 안정 |
| Mode coverage | 약함 | 강함 |
| 이미지 품질 | 높음 | 매우 높음 |
| 현재 트렌드 | 감소 | 증가 |

---

## 8. 참고 자료

- [GAN 원논문 - Goodfellow (2014)](https://arxiv.org/abs/1406.2661)
- [DCGAN Paper (2015)](https://arxiv.org/abs/1511.06434)
- [WGAN Paper (2017)](https://arxiv.org/abs/1701.07875)
- [StyleGAN Paper (2018)](https://arxiv.org/abs/1812.04948)
- [StyleGAN2 Paper (2019)](https://arxiv.org/abs/1912.04958)
- [BigGAN Paper (2018)](https://arxiv.org/abs/1809.11096)
- [FID Paper (2017)](https://arxiv.org/abs/1706.08500)
- [GAN Training Tips - Salimans](https://arxiv.org/abs/1606.03498)
