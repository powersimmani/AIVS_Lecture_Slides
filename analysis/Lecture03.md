# Lecture 03 분석 보고서
## From Set Theory to Linear Regression

**분석 일자**: 2026-02-15
**품질 등급**: A (우수)

---

## 1. 강의 구조 개요

| Part | 주제 | 슬라이드 | 평가 |
|------|------|----------|------|
| Part 1 | 수학적 기초 (집합, 선형대수, 미적분) | 03-12 | 우수 |
| Part 2 | 확률과 통계 기초 | 13-21 | 매우 우수 |
| Part 3 | 선형 회귀 모델 | 22-30 | 우수 |

---

## 2. 긍정적 평가

### 2.1 체계적인 수학적 흐름
- 집합론 → 함수 → 벡터공간 → 내적 → 행렬 → 고유값 → 미분의 자연스러운 전개
- 각 개념의 ML 응용 예시 포함
- 추상적 개념에서 구체적 응용으로 연결

### 2.2 고유값/고유벡터 예제 (슬라이드 11)
- 단계별 계산 과정 명시적으로 보여줌
- 특성방정식 → 고유값 → 고유벡터 순서 명확
- 기하학적 해석 (방향 유지, 스케일링) 시각화
- Spectral Theorem 언급

### 2.3 정규방정식과 경사하강법 비교 (슬라이드 26)
- 두 가지 해법 모두 제시
- 계산 복잡도 비교: O(np² + p³) vs O(np) per iteration
- 대규모 데이터에서의 선택 기준 제시

### 2.4 확률/통계 섹션
- MLE vs MAP 비교 명확
- CLT (중심극한정리) 설명 충실
- 상관관계 vs 인과관계 별도 슬라이드로 강조
- Bayes' Theorem의 Prior/Likelihood/Posterior 구조

### 2.5 회귀 가정 명시
- LINE 약어로 기억하기 쉽게 정리
  - **L**inearity
  - **I**ndependence
  - **N**ormality
  - **E**qual variance (Homoscedasticity)

---

## 3. 개선 권장사항

### 3.1 [중요] SVD (특이값 분해) 상세 설명 추가

**파일**: `Lecture03/Lecture03_11_Eigenvalues and Eigenvectors.html`

**현재 상태**: "SVD: A = UΣVᵀ (for any matrix)" 한 줄 언급

**문제점**:
- SVD는 ML에서 매우 중요 (추천 시스템, 차원 축소, 행렬 근사)
- 고유값 분해와의 차이점 불명확
- 직사각 행렬에 적용 가능한 점 강조 필요

**추가 권장 내용**:
```markdown
## SVD (Singular Value Decomposition)

### 정의
A = UΣVᵀ where:
- A: m×n matrix (any matrix)
- U: m×m orthogonal (left singular vectors)
- Σ: m×n diagonal (singular values σ₁ ≥ σ₂ ≥ ... ≥ 0)
- V: n×n orthogonal (right singular vectors)

### 고유값 분해와의 비교
| 항목 | 고유값 분해 | SVD |
|------|-------------|-----|
| 적용 행렬 | 정방행렬만 | 모든 행렬 |
| 분해 형태 | A = QΛQ⁻¹ | A = UΣVᵀ |
| 값 | 고유값 (음수 가능) | 특이값 (항상 ≥ 0) |

### ML 응용
- Low-rank approximation (행렬 압축)
- Pseudo-inverse 계산: A⁺ = VΣ⁺Uᵀ
- 추천 시스템 (Matrix Factorization)
- LSA (Latent Semantic Analysis)
```

---

### 3.2 [중요] Regularization 소개 추가

**위치**: Part 3 - Normal Equation 이후 또는 새 슬라이드

**현재 상태**: 정규화(Regularization) 언급 없음

**중요성**:
- 과적합 방지의 핵심 기법
- MAP과 Regularization의 연결 (Part 2에서 MAP 소개)
- Lecture 09 (Regularization)로의 자연스러운 연결

**추가 권장 내용**:
```html
<div class="concept-card">
    <div class="card-title">Regularized Regression (Preview)</div>
    <div class="card-content">
        <strong>Ridge (L2):</strong> β = (XᵀX + λI)⁻¹Xᵀy
        <br>- 항상 역행렬 존재 (수치적 안정성)
        <br>- MAP with Gaussian prior
        <br><br>
        <strong>Lasso (L1):</strong> argmin ||y - Xβ||² + λ||β||₁
        <br>- 희소 해 (Feature selection)
        <br>- MAP with Laplace prior
    </div>
</div>
```

**연결 포인트**:
- Part 2의 MAP → L2 regularization = Gaussian prior
- Normal Equation의 역행렬 문제 → Ridge가 해결

---

### 3.3 [중요] 다중공선성 진단 도구 추가

**파일**: `Lecture03/Lecture03_28_Multiple Linear Regression Extension.html` 또는 슬라이드 29

**현재 상태**: "Multicollinearity: High correlation between predictors causes unstable estimates" 언급만

**추가 권장 내용**:
```markdown
## 다중공선성 진단

### VIF (Variance Inflation Factor)
VIF_j = 1 / (1 - R²_j)

where R²_j = X_j를 다른 변수로 회귀했을 때의 R²

### 해석
| VIF | 해석 |
|-----|------|
| 1 | 공선성 없음 |
| 1-5 | 중간 정도 |
| > 5 | 높음, 주의 필요 |
| > 10 | 심각, 조치 필요 |

### 대응 방법
1. 변수 제거
2. PCA로 차원 축소
3. Ridge regression (L2 regularization)
```

---

### 3.4 [권장] 조건수 (Condition Number) 설명 추가

**파일**: 슬라이드 26 (Normal Equation Solution)

**현재 상태**: "Numerical stability issues with ill-conditioned matrices" 언급만

**추가 권장 내용**:
```markdown
## 수치적 안정성

### 조건수 (Condition Number)
κ(A) = ||A|| · ||A⁻¹|| = σ_max / σ_min

### 해석
- κ ≈ 1: Well-conditioned (안정적)
- κ >> 1: Ill-conditioned (불안정)
- κ = ∞: Singular matrix

### 실용적 기준
- κ > 10^k: 약 k 자릿수 정밀도 손실
- NumPy: np.linalg.cond(X.T @ X)

### 해결책
1. Feature scaling (표준화)
2. Ridge regression (λI 추가)
3. SVD 기반 pseudo-inverse 사용
```

---

### 3.5 [권장] Adjusted R² 수식 추가

**파일**: 슬라이드 29 (Model Assumptions and Diagnostics)

**현재 상태**: "Adjusted R²: Accounts for number of predictors" 언급만

**추가 권장**:
```markdown
## R² vs Adjusted R²

### R² (Coefficient of Determination)
R² = 1 - SS_res / SS_tot = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²

문제: 변수 추가 시 항상 증가 또는 유지

### Adjusted R²
R²_adj = 1 - (1 - R²)(n - 1) / (n - p - 1)

where n = 샘플 수, p = 변수 수

장점: 불필요한 변수 추가 시 감소 가능
```

---

### 3.6 [사소] HTML lang 속성 수정

**현재 상태**: 모든 HTML 파일이 `lang="ko"`

**수정 방법**:
```bash
cd Lecture03
sed -i 's/lang="ko"/lang="en"/g' *.html
```

---

### 3.7 [선택] 행렬 미분 공식 정리표 추가

**위치**: Part 1 미분 섹션 또는 Part 3 정규방정식 유도

**추가 권장**:
```markdown
## 유용한 행렬 미분 공식

| 함수 f(β) | ∇_β f |
|-----------|-------|
| aᵀβ | a |
| βᵀAβ | (A + Aᵀ)β |
| ||Xβ - y||² | 2Xᵀ(Xβ - y) |
| βᵀβ | 2β |
```

---

## 4. 기술적 정확성 검증 결과

| 항목 | 슬라이드 | 검증 |
|------|----------|------|
| 고유값 정의 Av = λv | 11 | ✅ 정확 |
| 특성방정식 det(A - λI) = 0 | 11 | ✅ 정확 |
| 예제 계산 (λ=5, λ=2) | 11 | ✅ 정확 |
| 정규방정식 β = (XᵀX)⁻¹Xᵀy | 26 | ✅ 정확 |
| 경사하강법 업데이트 규칙 | 26 | ✅ 정확 |
| MSE 편미분 | 26 | ✅ 정확 |
| Bayes' theorem | 17 | ✅ 정확 |
| CLT 공식 | 18 | ✅ 정확 |
| MLE 정의 | 19 | ✅ 정확 |
| LINE 가정 | 29 | ✅ 정확 |

---

## 5. 우선순위별 작업 체크리스트

### 다음 업데이트 시 (중요)
- [ ] SVD 상세 설명 및 예제 추가
- [ ] Ridge/Lasso regularization 미리보기 추가
- [ ] VIF (다중공선성 진단) 추가

### 시간 있을 때 (권장)
- [ ] 조건수 (Condition Number) 설명 추가
- [ ] Adjusted R² 수식 명시
- [ ] 행렬 미분 공식 정리표 추가

### 선택적 개선
- [ ] HTML lang="en" 으로 변경
- [ ] Python 코드 예제에 statsmodels VIF 계산 추가

---

## 6. 다른 강의와의 연계성

| 연계 강의 | 관련 내용 | 상태 |
|-----------|-----------|------|
| Lecture 02 (시각화) | Residual plots, Q-Q plots | ✅ 연계 좋음 |
| Lecture 04 (로지스틱) | 선형 회귀 → 분류로 확장 | ✅ 자연스러운 흐름 |
| Lecture 09 (정규화) | Regularization 상세 | ⚠️ Part 3에 미리보기 추가 권장 |
| Lecture 17 (비지도) | PCA, 고유값 분해 | ✅ 기초 잘 다룸 |

---

## 7. 특별 참고: 슬라이드 26의 두 페이지 구조

슬라이드 26은 HTML body에 두 개의 container를 포함:
1. Normal Equation Solution (정규방정식)
2. Gradient Descent Process (경사하강법)

**장점**: 두 방법을 한 파일에서 비교 가능
**주의**: slideshow에서 올바르게 표시되는지 확인 필요

---

## 8. 참고 자료

- [Gilbert Strang - Linear Algebra and Learning from Data](https://math.mit.edu/~gs/learningfromdata/)
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) - 행렬 미분 참고
- [scikit-learn VIF Tutorial](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html)
- [Condition Number - Wikipedia](https://en.wikipedia.org/wiki/Condition_number)
