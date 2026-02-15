# Lecture 06 분석 보고서
## Supervised Learning Evaluation

**분석 일자**: 2026-02-15
**품질 등급**: A+ (매우 우수)

---

## 1. 강의 구조 개요

| Part | 주제 | 슬라이드 | 평가 |
|------|------|----------|------|
| Part 1 | 평가의 중요성과 기초 | 03-11 | 매우 우수 |
| Part 2 | 회귀 평가 지표 | 12-16 | 우수 |
| Part 3 | 분류 평가 지표 | 17-24 | 매우 우수 |
| Part 4 | 모델 검증 기법 | 25-31 | 매우 우수 |

**참고**: 4개 파트로 구성 (다른 강의보다 1개 더 많음) - 평가의 중요성 반영

---

## 2. 긍정적 평가

### 2.1 종합적인 데이터 분할 전략 (Part 1)
- Train/Validation/Test 역할 명확히 구분
- Stratified Sampling 설명
- **Time Series Split** - 시계열 데이터 특수 처리 포함
- **Data Leakage** 방지 섹션 - 실무에서 매우 중요

### 2.2 Bias-Variance Tradeoff
- 수식: Total Error = Bias² + Variance + Irreducible Error
- 시각적 설명과 함께 제공
- Overfitting/Underfitting 연결

### 2.3 분류 지표 상세 (Part 3)
- **ROC Curve** (슬라이드 22, 37KB): 매우 상세
  - 색상 그라데이션으로 AUC 스케일 시각화
  - 다양한 threshold에 따른 변화 설명
- **PR Curve** (슬라이드 23, 38KB): 매우 상세
  - ROC vs PR 사용 시점 명확
  - 불균형 데이터에서 PR 선호 이유
- **Multi-class metrics**: Macro/Micro/Weighted 평균 구분

### 2.4 고급 검증 기법 (Part 4)
- K-Fold, Stratified K-Fold, LOOCV 비교
- **Bootstrapping**: 0.632 estimator 포함
- **Nested Cross-Validation**: 편향 방지 설명
- **Bayesian Optimization**: Optuna, hyperopt, scikit-optimize 언급

### 2.5 실용적인 코드 예시
```python
# 모든 주요 기법에 scikit-learn 코드 제공
cross_val_score(model, X, y, cv=5)
GridSearchCV(SVC(), param_grid, cv=5)
classification_report(y_true, y_pred)
```

### 2.6 HTML lang 속성
- **이미 `lang="en"`으로 올바르게 설정됨**

---

## 3. 개선 권장사항

### 3.1 [중요] Matthews Correlation Coefficient (MCC) 추가

**파일**: Part 3 분류 지표 섹션 (슬라이드 21 또는 새 슬라이드)

**중요성**:
- 불균형 데이터에서 가장 균형 잡힌 단일 지표
- TP, TN, FP, FN 모두 고려
- Google, Facebook 등에서 불균형 분류에 권장

**추가 권장 내용**:
```markdown
## Matthews Correlation Coefficient (MCC)

### 수식
MCC = (TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]

### 특징
- Range: [-1, +1]
  - +1: Perfect prediction
  - 0: Random prediction
  - -1: Total disagreement
- 모든 confusion matrix 요소 사용
- 불균형 데이터에서 Accuracy, F1보다 신뢰성 높음

### vs F1 Score
| 상황 | F1 | MCC |
|------|-----|-----|
| 클래스 균형 | 적합 | 적합 |
| 클래스 불균형 | 편향 가능 | 더 신뢰적 |
| TN 중요 | 무시 | 반영 |

### scikit-learn
```python
from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_true, y_pred)
```
```

---

### 3.2 [중요] Cohen's Kappa 추가

**파일**: Part 3 분류 지표 섹션

**중요성**:
- 우연에 의한 일치를 보정
- 불균형 데이터에서 Accuracy 대안
- 의료, 심리학 등에서 표준 지표

**추가 권장 내용**:
```markdown
## Cohen's Kappa (κ)

### 수식
κ = (p_o - p_e) / (1 - p_e)

where:
- p_o = observed agreement (accuracy)
- p_e = expected agreement by chance

### 해석
| κ | 해석 |
|---|------|
| < 0 | 우연보다 나쁨 |
| 0.0 - 0.20 | Slight |
| 0.21 - 0.40 | Fair |
| 0.41 - 0.60 | Moderate |
| 0.61 - 0.80 | Substantial |
| 0.81 - 1.00 | Almost perfect |

### vs Accuracy
- Accuracy: 우연 일치 포함
- Kappa: 우연 일치 제외
- 불균형 데이터에서 Kappa가 더 보수적

### scikit-learn
```python
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(y_true, y_pred)
```
```

---

### 3.3 [중요] Log Loss (Cross-Entropy Loss) 추가

**파일**: Part 3 분류 지표 섹션

**현재 상태**: 분류 지표에서 확률 예측 평가 지표 없음

**중요성**:
- 확률 예측의 품질 평가
- Kaggle 등 경진대회 표준 지표
- Calibration과 연결

**추가 권장 내용**:
```markdown
## Log Loss (Cross-Entropy Loss)

### 수식 (Binary)
LogLoss = -(1/n) Σ[yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]

### 특징
- Range: [0, ∞)
- 낮을수록 좋음
- **확신 있는 오답**에 큰 페널티
  - y=1, p=0.01 → -log(0.01) ≈ 4.6
  - y=1, p=0.5 → -log(0.5) ≈ 0.7

### vs Accuracy
| 예측 | y=1 | Accuracy | Log Loss |
|------|-----|----------|----------|
| p=0.51 | ✓ | 100% | ~0.67 |
| p=0.99 | ✓ | 100% | ~0.01 |

→ Log Loss는 **확신도** 반영

### scikit-learn
```python
from sklearn.metrics import log_loss
ll = log_loss(y_true, y_prob)
```

### 주의
- 예측 확률이 0 또는 1이면 undefined
- 일반적으로 clip: np.clip(y_prob, 1e-15, 1-1e-15)
```

---

### 3.4 [권장] Brier Score 추가

**위치**: Part 3 또는 Part 4

**중요성**:
- 확률 예측의 calibration 평가
- Lecture 04에서 언급한 probability calibration과 연결

**추가 권장 내용**:
```markdown
## Brier Score

### 수식
BS = (1/n) Σ(pᵢ - yᵢ)²

### 특징
- Range: [0, 1]
- 낮을수록 좋음 (MSE for probabilities)
- Calibration + Discrimination 모두 반영

### 분해 (Murphy decomposition)
BS = Reliability - Resolution + Uncertainty

- Reliability: Calibration 오차
- Resolution: 분별력
- Uncertainty: 데이터 자체 불확실성

### scikit-learn
```python
from sklearn.metrics import brier_score_loss
bs = brier_score_loss(y_true, y_prob)
```
```

---

### 3.5 [권장] Optuna 코드 예시 추가

**파일**: `Lecture06/Lecture06_30_Hyperparameter Tuning.html`

**현재 상태**: "Libraries: Optuna, hyperopt, scikit-optimize" 언급만

**추가 권장 내용**:
```python
## Optuna Example

import optuna

def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)

    # Train model
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )

    # Cross-validation
    score = cross_val_score(model, X, y, cv=5, scoring='f1').mean()
    return score

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Best parameters
print(study.best_params)
print(f"Best F1: {study.best_value:.4f}")
```

### Optuna 장점
- Pruning (조기 종료)
- Dashboard 시각화
- Parallel execution
- Tree-structured Parzen Estimator (TPE) 기본
```

---

### 3.6 [선택] Specificity와 NPV 추가

**파일**: Part 3 Precision/Recall 슬라이드

**현재 상태**: TPR(Recall), FPR만 강조

**추가 권장**:
```markdown
## 완전한 Confusion Matrix 지표

| 지표 | 수식 | 설명 |
|------|------|------|
| Sensitivity (Recall) | TP/(TP+FN) | 양성 중 탐지율 |
| Specificity | TN/(TN+FP) | 음성 중 탐지율 |
| PPV (Precision) | TP/(TP+FP) | 예측 양성의 정확도 |
| NPV | TN/(TN+FN) | 예측 음성의 정확도 |

### 의료 예시
- Sensitivity: 환자 중 양성 판정 비율 (놓치면 안됨)
- Specificity: 건강인 중 음성 판정 비율 (오진 방지)
- PPV: 양성 판정 받은 사람 중 실제 환자
- NPV: 음성 판정 받은 사람 중 실제 건강인
```

---

## 4. 기술적 정확성 검증 결과

| 항목 | 슬라이드 | 검증 |
|------|----------|------|
| MSE 수식 | 13 | ✅ 정확 |
| RMSE = √MSE | 13 | ✅ 정확 |
| R² = 1 - SS_res/SS_tot | 14 | ✅ 정확 |
| Adjusted R² 수식 | 14 | ✅ 정확 |
| MAPE 수식 | 15 | ✅ 정확 |
| Confusion Matrix 구조 | 18 | ✅ 정확 |
| Precision = TP/(TP+FP) | 20 | ✅ 정확 |
| Recall = TP/(TP+FN) | 20 | ✅ 정확 |
| F1 harmonic mean | 21 | ✅ 정확 |
| Fβ 수식 | 21 | ✅ 정확 |
| ROC: TPR vs FPR | 22 | ✅ 정확 |
| Bootstrap 63.2% unique | 29 | ✅ 정확 (1-1/e ≈ 0.632) |

---

## 5. 우선순위별 작업 체크리스트

### 다음 업데이트 시 (중요)
- [ ] Matthews Correlation Coefficient (MCC) 추가
- [ ] Cohen's Kappa 추가
- [ ] Log Loss (Cross-Entropy) 추가

### 시간 있을 때 (권장)
- [ ] Brier Score 추가 (calibration 평가)
- [ ] Optuna 코드 예시 추가
- [ ] Specificity, NPV 설명 추가

### 선택적 개선
- [ ] Balanced Accuracy 추가
- [ ] G-mean (Geometric Mean) 추가
- [ ] Cost-sensitive evaluation 언급

---

## 6. 다른 강의와의 연계성

| 연계 강의 | 관련 내용 | 상태 |
|-----------|-----------|------|
| Lecture 02 (시각화) | ROC/PR Curve 시각화 | ✅ 잘 연계 |
| Lecture 03 (회귀) | R², Residual 분석 | ✅ 잘 연계 |
| Lecture 04 (로지스틱) | Cross-Entropy, Threshold | ✅ 잘 연계 |
| Lecture 04 (Calibration) | 확률 보정 | ⚠️ Brier Score 추가 권장 |
| Lecture 09 (정규화) | Early Stopping | ✅ 언급됨 |

---

## 7. 특별 참고사항

### 슬라이드 크기 분석
| 슬라이드 | 크기 | 상세도 |
|----------|------|--------|
| ROC Curve (22) | 37KB | 매우 상세 |
| PR Curve (23) | 38KB | 매우 상세 |
| Multi-class (24) | 25KB | 상세 |
| K-fold CV (26) | 23KB | 상세 |
| Model Selection (31) | 19KB | 상세 |

→ 분류 평가 섹션이 특히 풍부

### 강의 범위
이 강의는 평가에 대한 **가장 종합적인** 강의로, 다른 강의들이 참조하는 핵심 강의입니다. 4개 파트로 구성된 것이 이를 반영합니다.

### Kaggle/실무 관련성
- Grid/Random Search: 기본
- Bayesian Optimization: 고급 (언급됨)
- Cross-Validation: 필수
- Stratified Split: 불균형 필수

---

## 8. 참고 자료

- [scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Matthews Correlation Coefficient](https://en.wikipedia.org/wiki/Phi_coefficient)
- [Cohen's Kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [ROC vs PR Curves (Davis & Goadrich, 2006)](https://www.biostat.wisc.edu/~page/rocpr.pdf)
