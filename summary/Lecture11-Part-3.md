# Lecture 11 - Part 3: Statistical Approaches

## Overview
This part covers traditional statistical methods for time series analysis, including moving averages, autoregressive models, and their combinations.

## Key Topics

### 1. Moving Average (MA)
- **Concept**: Prediction based on weighted past errors
- **MA(q) Formula**:
  ```
  yₜ = μ + εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θqεₜ₋q
  ```
  - μ: Mean
  - ε: Error terms (white noise)
  - θ: MA coefficients
  - q: Order (number of past errors)
- **Use Case**: Captures short-term shocks
- **Limitation**: Limited memory

### 2. Autoregressive (AR)
- **Concept**: Prediction based on past values
- **AR(p) Formula**:
  ```
  yₜ = c + φ₁yₜ₋₁ + φ₂yₜ₋₂ + ... + φpyₜ₋p + εₜ
  ```
  - c: Constant
  - φ: AR coefficients
  - p: Order (number of past values)
- **Use Case**: Data with persistence/momentum
- **Key Property**: Stationarity requirement

### 3. ARMA & ARIMA
- **ARMA(p,q)**: Combines AR and MA
  ```
  yₜ = c + Σφᵢyₜ₋ᵢ + εₜ + Σθⱼεₜ₋ⱼ
  ```
- **ARIMA(p,d,q)**: ARMA with differencing
  - p: AR order
  - d: Differencing order (for non-stationarity)
  - q: MA order
- **Differencing**: y'ₜ = yₜ - yₜ₋₁
- **Selection**: Use ACF/PACF plots, AIC/BIC
- **Seasonal ARIMA (SARIMA)**: Handles seasonality

### 4. Regression-Based Extensions
- **ARIMAX**: ARIMA with exogenous variables
- **VAR (Vector Autoregression)**: Multivariate time series
- **GARCH**: Models changing variance (volatility)
- **State Space Models**: Kalman filter approach
- **Prophet**: Facebook's decomposition model

### 5. Limitations of Traditional Methods
- **Linearity Assumption**: Cannot capture non-linear patterns
- **Stationarity Requirement**: Real data often non-stationary
- **Univariate Focus**: Multivariate extensions complex
- **Manual Feature Engineering**: Requires domain expertise
- **Long-range Dependencies**: Difficult to capture
- **High-dimensional Data**: Not designed for images, text, etc.

## Important Takeaways
1. MA models capture short-term error effects
2. AR models capture persistence based on past values
3. ARIMA combines both and handles non-stationarity
4. Model selection requires understanding data properties
5. Traditional methods assume linearity and stationarity
6. Deep learning methods address many limitations

