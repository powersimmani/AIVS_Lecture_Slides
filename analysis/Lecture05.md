# Lecture 05 분석 보고서
## From Logistic Regression to Multi-layer Perceptrons

**분석 일자**: 2026-02-15
**품질 등급**: A (우수)

---

## 1. 강의 구조 개요

| Part | 주제 | 슬라이드 | 평가 |
|------|------|----------|------|
| Part 1 | 신경망 동기 (XOR, UAT) | 03-11 | 우수 |
| Part 2 | MLP 구조와 Forward Propagation | 12-20 | 우수 |
| Part 3 | Backpropagation과 학습 | 21-30 | 매우 우수 |

---

## 2. 긍정적 평가

### 2.1 XOR 문제로 동기 부여 (Part 1)
- 선형 모델의 한계를 명확히 시각화
- Minsky & Papert (1969) 역사적 맥락 언급
- Feature space transformation 아이디어 소개

### 2.2 Universal Approximation Theorem
- Cybenko (1989), Hornik et al. (1989) 인용
- "Width vs Depth" 트레이드오프 설명
- Caveats 명시 (학습 가능성은 보장하지 않음)

### 2.3 활성화 함수 비교 (Part 2)
- **Sigmoid/Tanh**: Vanishing gradient 문제 설명
- **ReLU**: Dying ReLU 문제 언급
- **Leaky ReLU, ELU**: 변형들 비교
- 각 함수별 색상 코딩으로 구분 (슬라이드 18)

### 2.4 Backpropagation 유도 (Part 3)
- 단계별 수학적 유도 (슬라이드 24, 34KB 상세)
- Chain rule 설명 명확
- Computational graph 개념 소개
- Autograd 설명 및 PyTorch 예시

### 2.5 실용적인 구현 가이드
- Mini-batch vs Batch vs SGD 비교표
- 디버깅 팁 (gradient checking, overfit small batch)
- PyTorch/TensorFlow 코드 예시 포함

### 2.6 HTML lang 속성
- **이미 `lang="en"`으로 올바르게 설정됨**

---

## 3. 개선 권장사항

### 3.1 [중요] GELU (Gaussian Error Linear Unit) 추가

**파일**: `Lecture05/Lecture05_18_ReLU and Its Variants.html`

**중요성**:
- BERT, GPT, Vision Transformer 등 최신 모델에서 표준
- 현재 가장 많이 사용되는 활성화 함수 중 하나

**추가 권장 내용**:
```markdown
## GELU (Gaussian Error Linear Unit)

### 수식
GELU(x) = x · Φ(x)
        ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])

where Φ(x) = CDF of standard normal distribution

### 특징
- Smooth approximation to ReLU
- Non-monotonic (unlike ReLU)
- Stochastic regularization effect
- Default in Transformers (BERT, GPT)

### PyTorch
```python
nn.GELU()  # PyTorch 1.4+
```

### 비교
| 활성화 함수 | Transformer | CNN | 비고 |
|-------------|-------------|-----|------|
| ReLU | △ | ✓ | 기본 선택 |
| GELU | ✓ | △ | NLP/Transformer 표준 |
| Swish | ✓ | ✓ | EfficientNet |
```

---

### 3.2 [중요] Swish/SiLU 활성화 함수 추가

**파일**: 슬라이드 18 또는 새 슬라이드

**추가 권장 내용**:
```markdown
## Swish / SiLU

### 수식
Swish(x) = x · σ(βx)
SiLU(x) = x · σ(x)  # β = 1인 경우

where σ = sigmoid function

### 특징
- Google Brain에서 자동 탐색으로 발견 (2017)
- EfficientNet에서 사용
- ReLU보다 깊은 네트워크에서 성능 우수
- β는 학습 가능한 파라미터 (일반적으로 1 사용)

### PyTorch
```python
nn.SiLU()  # PyTorch 1.7+
```
```

---

### 3.3 [중요] Weight Initialization 수식 상세화

**파일**: `Lecture05/Lecture05_15_Weights and Biases.html`

**현재 상태**: "Xavier/Glorot: Good for tanh/sigmoid", "He: Good for ReLU" 언급만

**추가 권장 내용**:
```markdown
## Weight Initialization 상세

### Xavier/Glorot Initialization (2010)
목적: Variance를 레이어 간 일정하게 유지

**Uniform**:
W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))

**Normal**:
W ~ N(0, 2/(n_in + n_out))

적합: Sigmoid, Tanh (선형 영역 가정)

### He Initialization (2015)
목적: ReLU의 비대칭성 고려

**Normal**:
W ~ N(0, 2/n_in)

**Uniform**:
W ~ U(-√(6/n_in), √(6/n_in))

적합: ReLU, Leaky ReLU

### PyTorch
```python
nn.init.xavier_uniform_(layer.weight)
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

### 왜 중요한가?
- 잘못된 초기화 → Vanishing/Exploding gradients
- 학습 시작부터 활성화 분포가 너무 크거나 작으면 학습 불가
```

---

### 3.4 [권장] PReLU (Parametric ReLU) 추가

**파일**: 슬라이드 18 ReLU Variants

**현재 상태**: Leaky ReLU만 언급 (α ≈ 0.01 고정)

**추가 권장**:
```markdown
## PReLU (Parametric ReLU)

### 수식
PReLU(x) = max(αx, x)

where α is a **learnable** parameter

### vs Leaky ReLU
- Leaky ReLU: α = 0.01 (고정)
- PReLU: α를 학습 (네트워크가 최적값 찾음)

### 장점
- 각 채널별로 다른 α 가능
- Dying ReLU 방지 + 최적 기울기 학습

### 단점
- 추가 파라미터 (미미함)
- 과적합 위험 (데이터 적을 때)

### PyTorch
```python
nn.PReLU(num_parameters=1)  # 전체 공유
nn.PReLU(num_parameters=channels)  # 채널별
```
```

---

### 3.5 [권장] Batch Normalization 미리보기

**위치**: Part 2 끝부분 또는 Part 3

**중요성**:
- MLP에서도 학습 안정화에 중요
- Lecture 09 (정규화)에서 상세히 다룰 수 있지만 미리 소개 권장

**추가 권장 내용**:
```markdown
## Batch Normalization Preview

### 문제: Internal Covariate Shift
- 레이어 입력 분포가 학습 중 계속 변함
- 이전 레이어 가중치 변화 → 다음 레이어 입력 분포 변화

### 해결책: Normalize each layer's inputs
```
x̂ = (x - μ_batch) / √(σ²_batch + ε)
y = γx̂ + β  # learnable scale and shift
```

### 장점
- 높은 learning rate 사용 가능
- Initialization에 덜 민감
- 일종의 regularization 효과

### 위치
- Linear/Conv 후, Activation 전 (일반적)
- [Linear → BN → ReLU] 순서

### 상세 내용: Lecture 09 참조
```

---

### 3.6 [선택] Gradient Vanishing/Exploding 시각화 추가

**파일**: Part 3 Chain Rule 또는 Backpropagation 슬라이드

**추가 권장**:
```markdown
## Gradient Flow 문제 시각화

### Vanishing Gradients
Layer 10 → Layer 1으로 backprop 시:

σ'(z) ≤ 0.25 (sigmoid 최대 미분값)
0.25^10 ≈ 0.000001

→ 초기 레이어 거의 학습 안됨

### Exploding Gradients
|W| > 1인 경우:
1.5^10 ≈ 57

→ Gradient가 매우 커져서 발산

### 해결책 요약
| 문제 | 해결책 |
|------|--------|
| Vanishing | ReLU, ResNet skip connections, LSTM |
| Exploding | Gradient clipping, 적절한 초기화 |
```

---

## 4. 기술적 정확성 검증 결과

| 항목 | 슬라이드 | 검증 |
|------|----------|------|
| XOR truth table | 05 | ✅ 정확 |
| Sigmoid σ(z) = 1/(1+e⁻ᶻ) | 17 | ✅ 정확 |
| Tanh 수식 | 17 | ✅ 정확 |
| ReLU = max(0, z) | 18 | ✅ 정확 |
| Leaky ReLU (α=0.01) | 18 | ✅ 정확 |
| ELU 수식 | 18 | ✅ 정확 |
| Forward propagation 공식 | 16 | ✅ 정확 |
| Backprop δ⁽ˡ⁾ 공식 | 24 | ✅ 정확 |
| Gradient ∂L/∂W = δ(a)ᵀ | 24 | ✅ 정확 |
| Mini-batch 범위 32-1024 | 28 | ✅ 일반적 권장 범위 |

---

## 5. 우선순위별 작업 체크리스트

### 다음 업데이트 시 (중요)
- [ ] GELU 활성화 함수 추가 (Transformer 표준)
- [ ] Swish/SiLU 활성화 함수 추가
- [ ] Xavier/He 초기화 수식 상세화

### 시간 있을 때 (권장)
- [ ] PReLU (Parametric ReLU) 추가
- [ ] Batch Normalization 미리보기
- [ ] Gradient vanishing/exploding 시각화

### 선택적 개선
- [ ] Mish 활성화 함수 (YOLOv4에서 사용)
- [ ] Softplus: log(1 + eˣ) (ReLU의 smooth 버전)

---

## 6. 다른 강의와의 연계성

| 연계 강의 | 관련 내용 | 상태 |
|-----------|-----------|------|
| Lecture 04 (로지스틱) | 로지스틱 → MLP 확장 | ✅ 자연스러운 연결 |
| Lecture 07 (DNN) | 더 깊은 네트워크 | ✅ 좋은 기초 |
| Lecture 08 (최적화) | 옵티마이저 상세 | ⚠️ Adam 간략 언급, 상세는 Lec08 |
| Lecture 09 (정규화) | BN, Dropout 등 | ⚠️ 미리보기 추가 권장 |
| Lecture 11-12 (RNN) | 시퀀스 모델 | ✅ MLP 기초 필요 |

---

## 7. 특별 참고사항

### 생물학적 뉴런 비교 (슬라이드 08)
- 34KB의 상세한 슬라이드
- Dendrite, Soma, Axon, Synapse 비유 적절
- 다만 "Key Difference" 섹션에서 실제 뉴런과의 차이점 더 강조 가능
  - 실제 뉴런: Spike timing, temporal coding
  - 인공 뉴런: Rate coding (연속 값)

### PyTorch/TensorFlow 코드 품질 (슬라이드 30)
- 두 프레임워크 모두 예시 제공
- Training loop 구조 명확
- `optimizer.zero_grad()` 순서 정확

### 파일 중복 확인
- `Lecture 5_ From Logistic Regression to Multi-layer Perceptrons.html` (공백 포함)
- `Lecture05_01_Lecture 5 From Logistic Regression to Multi-layer Perceptrons.html`
- 두 파일이 비슷한 이름으로 존재 → 정리 권장

---

## 8. 참고 자료

- [Glorot & Bengio (2010) - Xavier Initialization](http://proceedings.mlr.press/v9/glorot10a.html)
- [He et al. (2015) - Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852)
- [Hendrycks & Gimpel (2016) - GELU](https://arxiv.org/abs/1606.08415)
- [Ramachandran et al. (2017) - Swish](https://arxiv.org/abs/1710.05941)
- [Universal Approximation Theorem - Wikipedia](https://en.wikipedia.org/wiki/Universal_approximation_theorem)
