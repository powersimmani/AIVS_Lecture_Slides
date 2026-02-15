# Lecture 11 분석 보고서
## Sequence Models (Part 1 - Statistical Foundations)

**분석 일자**: 2026-02-15
**품질 등급**: A (우수)

---

## 1. 강의 구조 개요

| Part | 주제 | 슬라이드 | 평가 |
|------|------|----------|------|
| Part 1 | 시퀀스 모델링 기초 | 03-10 | 우수 |
| Part 2 | 시퀀스 데이터 유형 | 11-19 | 우수 |
| Part 3 | 통계적 접근법 | 20-30 | 우수 |

**참고**: Lecture 11은 시퀀스 모델의 기초/통계적 방법 중심, Lecture 12는 RNN/LSTM 중심으로 나뉨

---

## 2. 긍정적 평가

### 2.1 시퀀스 데이터 다양성 커버리지 (Part 1-2)
- Time Series, Text, Audio, Video 모두 다룸
- 각 데이터 유형의 특성과 전처리 설명
- "Dog bites man" vs "Man bites dog" 예시로 순서 중요성 설명

### 2.2 통계적 시계열 분석 기초 (Part 3)
- MA, AR, ARMA, ARIMA 순차적 설명
- ACF/PACF 플롯 활용
- SARIMA (계절성) 포함
- 전통 방법의 한계 명시

### 2.3 Feature Engineering 실용 가이드
- Sliding Window, Lag Features
- Rolling Statistics (Moving Average, Std)
- Temporal Features (시간, 요일, 계절)

---

## 3. 개선 권장사항

### 3.1 [중요] Prophet 상세 설명 추가

**위치**: Part 3 통계적 접근법

**현재 상태**: "Prophet: Facebook's decomposition model" 한 줄 언급

**중요성**:
- 실무에서 가장 많이 사용되는 시계열 예측 도구
- AutoML 성격으로 비전문가도 사용 가능

**추가 권장 내용**:
```markdown
## Prophet (Facebook/Meta)

### 모델 구조
y(t) = g(t) + s(t) + h(t) + ε

- g(t): Growth (linear or logistic)
- s(t): Seasonality (Fourier series)
- h(t): Holiday effects
- ε: Error term

### 주요 특징
- 결측값 자동 처리
- 이상치에 강건
- 해석 가능한 분해
- 불규칙 간격 데이터 지원

### Python 사용
```python
from prophet import Prophet
model = Prophet()
model.fit(df)  # df must have 'ds' and 'y' columns
forecast = model.predict(future_df)
```

### 장점
- 비전문가도 사용 가능
- 빠른 프로토타이핑
- 자동 계절성 탐지

### 한계
- 복잡한 비선형 패턴 제한
- 외생 변수 처리 제한적
- 딥러닝 대비 성능 한계
```

---

### 3.2 [중요] Stationarity 테스트 추가

**위치**: Part 3 ARIMA 섹션

**현재 상태**: "Stationarity requirement" 언급만

**추가 권장 내용**:
```markdown
## Stationarity (정상성)

### 정의
- 시계열의 통계적 특성이 시간에 따라 변하지 않음
- Mean, Variance가 일정
- Autocovariance가 시차에만 의존

### 테스트 방법
1. **ADF (Augmented Dickey-Fuller)**
   - H₀: 비정상 (단위근 존재)
   - p < 0.05: 정상성 가정

2. **KPSS Test**
   - H₀: 정상
   - ADF와 함께 사용 권장

### Python
```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(series)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
```

### 비정상 → 정상 변환
1. 차분 (Differencing): y'_t = y_t - y_{t-1}
2. 로그 변환: log(y_t)
3. 계절 차분: y_t - y_{t-s}
```

---

### 3.3 [중요] Temporal Convolutional Networks (TCN) 미리보기

**위치**: Part 3 끝부분

**중요성**:
- RNN 대안으로 부상
- 병렬 처리 가능
- Lecture 12와 연결

**추가 권장 내용**:
```markdown
## Temporal Convolutional Networks (Preview)

### TCN vs RNN
| 항목 | RNN/LSTM | TCN |
|------|----------|-----|
| 처리 방식 | 순차적 | 병렬 |
| 학습 속도 | 느림 | 빠름 |
| 장기 의존성 | 어려움 | Dilated Conv |
| Gradient | Vanishing | 안정 |

### 핵심 아이디어
- 1D Convolution + Dilated Convolution
- 과거 정보만 사용 (Causal)
- 깊이에 따라 수용 영역 확장

### 상세 내용: Lecture 12 참조
```

---

### 3.4 [권장] Time Series Cross-Validation 추가

**위치**: Part 1 또는 별도 슬라이드

**중요성**:
- 시계열에서 일반적인 k-fold CV 사용 불가
- 시간 순서 보존 필수

**추가 권장 내용**:
```markdown
## Time Series Cross-Validation

### 왜 일반 CV가 안 되는가?
- 미래 데이터로 과거 예측 = Data Leakage
- 시간 순서 위반

### 올바른 방법
1. **Walk-Forward Validation**
   - Train: [1:t], Test: [t+1]
   - 점진적으로 train 확장

2. **Sliding Window**
   - 고정 크기 train window 이동

3. **Expanding Window**
   - Train 시작은 고정, 끝 확장

### sklearn
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # Train always before Test
```
```

---

### 3.5 [권장] 다변량 시계열 강화

**위치**: Part 2 또는 Part 3

**현재 상태**: VAR만 간략 언급

**추가 권장**:
```markdown
## 다변량 시계열 (Multivariate Time Series)

### VAR (Vector Autoregression)
- 모든 변수가 서로 영향
- 경제학에서 많이 사용

### Granger Causality
- 한 시계열이 다른 시계열 예측에 도움이 되는지
- 인과관계는 아님 (예측 관계)

### Multivariate LSTM (Preview)
- 여러 feature를 동시 입력
- Lecture 12에서 상세

### 실무 예시
- 날씨 예보: 온도 + 습도 + 기압 + 풍속
- 주식: 가격 + 거래량 + 관련 종목
```

---

## 4. 기술적 정확성 검증 결과

| 항목 | 슬라이드 | 검증 |
|------|----------|------|
| MA(q) 수식 | Part 3 | ✅ 정확 |
| AR(p) 수식 | Part 3 | ✅ 정확 |
| ARIMA(p,d,q) 의미 | Part 3 | ✅ 정확 |
| 차분 공식 y'=y_t-y_{t-1} | Part 3 | ✅ 정확 |
| ACF/PACF 용도 | Part 3 | ✅ 정확 |

---

## 5. 우선순위별 작업 체크리스트

### 다음 업데이트 시 (중요)
- [ ] Prophet 상세 설명 추가
- [ ] Stationarity 테스트 (ADF, KPSS) 추가
- [ ] TCN 미리보기 추가

### 시간 있을 때 (권장)
- [ ] Time Series Cross-Validation 추가
- [ ] 다변량 시계열 강화 (VAR, Granger)
- [ ] Exponential Smoothing 언급

### 선택적 개선
- [ ] Facebook NeuralProphet 언급
- [ ] 이상치 탐지 (Anomaly Detection) 간략 소개

---

## 6. 다른 강의와의 연계성

| 연계 강의 | 관련 내용 | 상태 |
|-----------|-----------|------|
| Lecture 07 (특징 추출) | 시계열 통계 특징 | ✅ 연계됨 |
| Lecture 12 (RNN/LSTM) | 딥러닝 시퀀스 모델 | ✅ 자연스러운 연결 |
| Lecture 13-14 (Transformer) | Attention 메커니즘 | ⚠️ 시계열 Transformer 미리보기 권장 |
| Lecture 17 (비지도) | 시계열 클러스터링 | △ 간략 연계 가능 |

---

## 7. 특별 참고사항

### 강의 분할 구조
- Lecture 11: 시퀀스 기초 + 통계적 방법
- Lecture 12: RNN, LSTM, GRU, Seq2Seq

이 분할이 적절함 - 전통적 방법 → 딥러닝 전환

### ARIMA vs Deep Learning 비교표 추가 권장
```markdown
| 항목 | ARIMA | RNN/LSTM |
|------|-------|----------|
| 선형성 | 선형 | 비선형 |
| 데이터 양 | 적어도 가능 | 많이 필요 |
| 해석성 | 높음 | 낮음 |
| 다변량 | VAR 필요 | 자연스러움 |
| 장기 예측 | 약함 | 강함 |
```

---

## 8. 참고 자료

- [ARIMA - Statsmodels](https://www.statsmodels.org/stable/tsa.html)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Time Series Forecasting Best Practices](https://arxiv.org/abs/2004.10240)
- [ADF Test - Wikipedia](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test)
- [Temporal Convolutional Networks](https://arxiv.org/abs/1803.01271)
