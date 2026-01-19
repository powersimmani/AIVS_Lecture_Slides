# Lecture 03 - Part 2: Probability and Statistics Fundamentals

## Overview
This part covers probability theory and statistical concepts essential for machine learning, including distributions, estimation methods, and the connection between probability and inference.

## Key Topics

### 1. Probability Spaces and Random Variables
- **Probability Space** (Ω, F, P):
  - Ω: Sample space (all possible outcomes)
  - F: Event space (subsets of Ω)
  - P: Probability function (0 ≤ P(A) ≤ 1)
- **Random Variable**: Function mapping outcomes to real numbers
  - Discrete: Countable values (X ∈ {0, 1, 2, ...})
  - Continuous: Any real value (X ∈ ℝ)
- **Probability Rules**:
  - P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
  - P(Aᶜ) = 1 - P(A)

### 2. Probability Distributions - Discrete and Continuous
- **Discrete Distributions**:
  - **Bernoulli**: Single binary trial (P(X=1) = p)
  - **Binomial**: n independent Bernoulli trials
  - **Poisson**: Count of events in fixed interval (λ = rate)
- **Continuous Distributions**:
  - **Uniform**: Equal probability over interval [a, b]
  - **Normal/Gaussian**: N(μ, σ²), bell-shaped curve
  - **Exponential**: Time between Poisson events
- **PDF vs PMF**: Continuous uses density (integral = probability), discrete uses mass function
- **CDF**: F(x) = P(X ≤ x), cumulative distribution function

### 3. Expectation, Variance, and Covariance
- **Expectation (Mean)**:
  - E[X] = Σ xᵢP(xᵢ) (discrete)
  - E[X] = ∫ x f(x) dx (continuous)
  - Properties: E[aX + b] = aE[X] + b
- **Variance**: Var(X) = E[(X - μ)²] = E[X²] - (E[X])²
  - Standard deviation: σ = √Var(X)
- **Covariance**: Cov(X,Y) = E[(X - μₓ)(Y - μᵧ)]
  - Measures linear relationship between variables
- **Correlation**: ρ = Cov(X,Y) / (σₓσᵧ), normalized to [-1, 1]

### 4. Conditional Probability and Bayes' Theorem
- **Conditional Probability**: P(A|B) = P(A ∩ B) / P(B)
- **Independence**: P(A|B) = P(A) or P(A ∩ B) = P(A)P(B)
- **Bayes' Theorem**:
  ```
  P(A|B) = P(B|A) × P(A) / P(B)
  ```
  - P(A): Prior probability
  - P(B|A): Likelihood
  - P(A|B): Posterior probability
- **ML Application**: Naive Bayes, Bayesian inference, probabilistic models

### 5. Central Limit Theorem and Law of Large Numbers
- **Law of Large Numbers**:
  - Sample mean converges to population mean as n → ∞
  - X̄ₙ → μ (in probability)
- **Central Limit Theorem (CLT)**:
  - Sum of i.i.d. random variables approaches normal distribution
  - √n(X̄ₙ - μ)/σ → N(0,1) as n → ∞
  - Works regardless of original distribution shape
- **Implications**:
  - Justifies normal approximations
  - Foundation for confidence intervals
  - Explains why normal distribution is so common

### 6. Parameter Estimation - MLE and MAP
- **Maximum Likelihood Estimation (MLE)**:
  - Find θ that maximizes P(data|θ)
  - θ_MLE = argmax L(θ) = argmax Π P(xᵢ|θ)
  - Often maximize log-likelihood instead
- **Maximum A Posteriori (MAP)**:
  - Include prior: θ_MAP = argmax P(θ|data) = argmax P(data|θ)P(θ)
  - MLE with regularization interpretation
- **Comparison**: MLE assumes no prior knowledge; MAP incorporates prior beliefs
- **ML Application**: Training neural networks, fitting distributions

### 7. Hypothesis Testing and Confidence Intervals
- **Hypothesis Testing**:
  - H₀: Null hypothesis (default assumption)
  - H₁: Alternative hypothesis
  - p-value: Probability of observing data given H₀ is true
  - Significance level α (typically 0.05)
- **Types of Errors**:
  - Type I: Reject H₀ when true (false positive)
  - Type II: Accept H₀ when false (false negative)
- **Confidence Intervals**:
  - Range of plausible values for parameter
  - 95% CI: [X̄ - 1.96σ/√n, X̄ + 1.96σ/√n]
- **ML Application**: Model comparison, A/B testing, significance of results

### 8. Correlation vs Causation
- **Correlation**: Statistical association between variables
  - Does NOT imply causation
  - Can be spurious (third variable)
- **Causation**: One variable directly affects another
- **Confounding Variables**: Hidden factors affecting both variables
- **Establishing Causation**:
  - Randomized controlled trials
  - Causal inference methods (do-calculus)
  - Careful experimental design
- **ML Consideration**: Models learn correlations; interpreting as causal requires caution

## Important Takeaways
1. Probability provides the foundation for reasoning under uncertainty
2. MLE and MAP are fundamental parameter estimation methods
3. The CLT explains why normal distributions are ubiquitous
4. Bayes' theorem enables updating beliefs with new evidence
5. Correlation does not imply causation - be careful with interpretations
6. Variance and covariance quantify uncertainty and relationships
7. Statistical testing helps validate ML model improvements
