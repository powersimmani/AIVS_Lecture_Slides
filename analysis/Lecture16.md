# Lecture 16 분석 보고서
## Generative Models - Diffusion

**분석 일자**: 2026-02-15
**품질 등급**: A+ (매우 우수)

---

## 1. 강의 구조 개요

| Part | 주제 | 슬라이드 | 평가 |
|------|------|----------|------|
| Part 1 | 도입 및 동기 | 03-06 | 우수 |
| Part 2 | Forward Process | 07-12 | 매우 우수 |
| Part 3 | Reverse Process | 13-18 | 매우 우수 |
| Part 4 | Sampling | 19-23 | 매우 우수 |
| Part 5 | Architecture | 24-27 | 매우 우수 |
| Part 6 | Advanced Techniques | 28-31 | 매우 우수 |
| Part 7 | Applications | 32-34 | 우수 |

---

## 2. 긍정적 평가

### 2.1 직관적 설명 (Part 1)
- "잉크 방울이 물에 퍼지는 과정" 비유
- GAN과 명확한 비교
- 생성 모델 진화 타임라인

### 2.2 Forward Process 상세 (Part 2)
- 점진적 노이즈 추가 수식:
  - q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
- Cumulative effect:
  - q(x_t | x_0) = N(x_t; √(ᾱ_t)x_0, (1-ᾱ_t)I)
- 수학적 속성 설명
- 시각화 포함

### 2.3 Reverse Process 상세 (Part 3)
- Score function 설명: ∇_x log p(x)
- 신경망 파라미터화
- Denoising 목적 함수:
  - L = E[||ε - ε_θ(x_t, t)||²]
- 학습 알고리즘 명확

### 2.4 Sampling 방법 (Part 4)
- **DDPM**: 전체 1000 step
- **DDIM**: deterministic, 빠른 샘플링
- **Conditional Generation**: class conditioning
- **Classifier-Free Guidance**: cfg scale 설명

### 2.5 U-Net 아키텍처 (Part 5)
- Encoder-Decoder 구조
- Skip connections 중요성
- Attention 메커니즘 통합
- Condition injection 방법:
  - Time embedding
  - Cross-attention for text

### 2.6 고급 기법 (Part 6)
- **Latent Diffusion (Stable Diffusion)**:
  - Pixel space → Latent space
  - 계산 효율성 대폭 향상
- Noise schedule 개선
- 기타 최적화 기법

---

## 3. 개선 권장사항

### 3.1 [중요] ELBO 유도 추가

**위치**: Part 3 Reverse Process

**현재 상태**: Loss 수식만 제시

**중요성**:
- Diffusion의 이론적 기반
- VAE와의 연결 이해

**추가 권장 내용**:
```markdown
## ELBO (Evidence Lower Bound) 유도

### 목표
- log p(x_0) 최대화

### 유도 과정
log p(x_0) ≥ E_q[log p(x_0:T)/q(x_1:T|x_0)]

### 분해
L = L_0 + L_1 + ... + L_{T-1} + L_T

where:
- L_T = D_KL(q(x_T|x_0) || p(x_T))  # 상수
- L_t = D_KL(q(x_t|x_{t+1},x_0) || p_θ(x_t|x_{t+1}))
- L_0 = -log p_θ(x_0|x_1)  # reconstruction

### 단순화
최종적으로:
L_simple = E_t,ε[||ε - ε_θ(x_t, t)||²]

### 의미
- 각 step에서 노이즈 예측
- 단순한 MSE loss로 귀결
```

---

### 3.2 [중요] Score Matching 이론 추가

**위치**: Part 3 또는 별도 슬라이드

**현재 상태**: Score function 간략 언급

**추가 권장**:
```markdown
## Score Matching

### Score Function
s(x) = ∇_x log p(x)

- 확률 밀도의 gradient
- 고밀도 방향을 가리킴

### Denoising Score Matching
- 직접 score 학습 어려움
- 대신 노이즈 예측으로 대체

### 관계
ε_θ(x_t, t) ∝ -s_θ(x_t, t)

노이즈 예측 = 음의 score 방향

### Song et al. (2020) 해석
- SDE (Stochastic Differential Equation) 관점
- Forward: dx = f(x,t)dt + g(t)dw
- Reverse: dx = [f - g²s(x)]dt + gdw̄

### 의미
- Diffusion = Score-based 모델
- 두 관점의 통합
```

---

### 3.3 [중요] ControlNet 추가

**위치**: Part 6 또는 Part 7

**중요성**:
- Stable Diffusion의 주요 확장
- 조건부 제어의 표준

**추가 권장**:
```markdown
## ControlNet (2023)

### 아이디어
- 추가 조건(pose, edge, depth)으로 생성 제어
- 원본 SD weights 동결, ControlNet만 학습

### 구조
- U-Net encoder 복사
- Zero Convolution으로 연결
- 원본 output + ControlNet output

### 지원 조건
| 조건 | 입력 | 용도 |
|------|------|------|
| Canny | Edge map | 윤곽선 유지 |
| Pose | Skeleton | 포즈 제어 |
| Depth | Depth map | 구조 유지 |
| Seg | Segmentation | 영역 제어 |
| Scribble | Sketch | 대략적 형태 |

### 특징
- 기존 SD 품질 유지
- 다양한 조건 결합 가능
- 빠른 학습 (원본 동결)
```

---

### 3.4 [권장] Consistency Models 소개

**위치**: Part 6

**중요성**:
- OpenAI 최신 연구
- 1-step 생성 가능

**추가 권장**:
```markdown
## Consistency Models (2023)

### 문제
- Diffusion: 수십~수천 step 필요
- 느린 추론 속도

### 아이디어
- 궤적의 어느 점에서든 동일 출력
- f(x_t, t) = f(x_{t'}, t') = x_0

### 장점
- 1-step generation 가능
- 품질은 다소 저하
- 학습 방법:
  1. Distillation (teacher 필요)
  2. Direct training (teacher 불필요)

### 비교
| 모델 | Steps | FID (CIFAR-10) |
|------|-------|----------------|
| DDPM | 1000 | 3.17 |
| DDIM | 50 | 4.67 |
| CM | 1 | 6.20 |
| CM | 2 | 3.55 |
```

---

### 3.5 [권장] Inpainting/Outpainting 응용

**위치**: Part 7 Applications

**추가 권장**:
```markdown
## Image Editing with Diffusion

### Inpainting
- 마스크 영역 재생성
- 주변 context 유지
- Repaint: 반복 denoising

### Outpainting
- 이미지 경계 확장
- Blending 기법 필요

### Image-to-Image
- 부분적 노이즈 추가 후 재생성
- strength 파라미터로 변화량 조절
- SDEdit 방식

### Prompt-based Editing
- Prompt-to-Prompt: attention 조작
- InstructPix2Pix: 텍스트 명령으로 편집
```

---

### 3.6 [권장] Noise Schedule 비교 상세화

**파일**: 슬라이드 30

**추가 권장**:
```markdown
## Noise Schedule 비교

### Linear Schedule
β_t = β_1 + (β_T - β_1)(t-1)/(T-1)
- 원조 DDPM
- 초반에 정보 손실 빠름

### Cosine Schedule
ᾱ_t = cos((t/T + s)/(1+s) × π/2)²
- 더 균일한 정보 감소
- 초반 정보 보존 우수

### 비교
| Schedule | 장점 | 단점 |
|----------|------|------|
| Linear | 단순 | 초반 손실 급격 |
| Cosine | 균일 | 계산 복잡 |
| Sigmoid | 안정 | 하이퍼파라미터 |

### 시각화
- Linear: β 선형 증가
- Cosine: ᾱ 부드러운 감소
```

---

## 4. 기술적 정확성 검증 결과

| 항목 | 슬라이드 | 검증 |
|------|----------|------|
| Forward process 수식 | 08-10 | ✅ 정확 |
| Reparameterization trick | 09 | ✅ 정확 |
| Denoising objective | 17 | ✅ 정확 |
| DDIM deterministic sampling | 21 | ✅ 정확 |
| Classifier-Free Guidance | 23 | ✅ 정확 |
| U-Net skip connection | 25 | ✅ 정확 |
| Latent Diffusion 개념 | 29 | ✅ 정확 |

---

## 5. 우선순위별 작업 체크리스트

### 다음 업데이트 시 (중요)
- [ ] ELBO 유도 추가
- [ ] Score Matching 이론 추가
- [ ] ControlNet 상세 추가

### 시간 있을 때 (권장)
- [ ] Consistency Models 소개
- [ ] Inpainting/Outpainting 응용
- [ ] Noise Schedule 비교 상세화

### 선택적 개선
- [ ] DiT (Diffusion Transformer) 언급
- [ ] SDXL 아키텍처 차이
- [ ] LoRA for Diffusion

---

## 6. 다른 강의와의 연계성

| 연계 강의 | 관련 내용 | 상태 |
|-----------|-----------|------|
| Lecture 15 (GAN) | 생성 모델 비교 | ✅ 좋은 연결 |
| Lecture 10 (아키텍처) | U-Net, Skip Connection | ✅ 기초 제공됨 |
| Lecture 13 (Transformer) | Cross-Attention | ✅ 연계됨 |
| Lecture 17 (비지도) | VAE와 연결 | ⚠️ VAE 상세 연결 권장 |
| Lecture 14 (PLM) | Text Encoder (CLIP) | ✅ 연계됨 |

---

## 7. 특별 참고사항

### Diffusion의 현재 위치
- 2020년 DDPM으로 주목
- 2022년 Stable Diffusion으로 폭발적 성장
- 현재 이미지/비디오/오디오 생성의 주류

### 핵심 하이퍼파라미터
```python
# DDPM 기본 설정
T = 1000  # total timesteps
beta_start = 1e-4
beta_end = 0.02
schedule = "linear"  # or "cosine"

# Classifier-Free Guidance
cfg_scale = 7.5  # 일반적인 값
# 높을수록 prompt 충실, 낮을수록 다양성
```

### Stable Diffusion 핵심 구성요소
| 구성요소 | 역할 | 모델 |
|----------|------|------|
| VAE Encoder | 이미지 → Latent | AutoencoderKL |
| U-Net | 노이즈 예측 | UNet2DCondition |
| Text Encoder | 텍스트 → Embedding | CLIP |
| VAE Decoder | Latent → 이미지 | AutoencoderKL |

### 추론 속도 최적화
```
1000 steps (DDPM) → 50 steps (DDIM) → 20 steps (DPM++) → 4 steps (LCM)
```

---

## 8. 참고 자료

- [DDPM Paper - Ho et al. (2020)](https://arxiv.org/abs/2006.11239)
- [DDIM Paper - Song et al. (2020)](https://arxiv.org/abs/2010.02502)
- [Score SDE - Song et al. (2020)](https://arxiv.org/abs/2011.13456)
- [Stable Diffusion - Rombach (2021)](https://arxiv.org/abs/2112.10752)
- [Classifier-Free Guidance (2022)](https://arxiv.org/abs/2207.12598)
- [ControlNet Paper (2023)](https://arxiv.org/abs/2302.05543)
- [Consistency Models (2023)](https://arxiv.org/abs/2303.01469)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
- [Diffusers Library](https://github.com/huggingface/diffusers)
