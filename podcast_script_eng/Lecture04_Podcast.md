# Lecture 04 Podcast: Regularization and Logistic Regression - From Regression to Classification

## Episode Information
- **Topic**: Polynomial regression, regularization (Ridge/Lasso), logistic regression
- **Estimated Time**: 15 minutes
- **Target Audience**: ML beginners, those learning classification algorithms

---

## Script

**[Intro - 0:00]**

**Host A**: Hello! This is the AI Vision Systems Podcast. Last time we learned about linear regression, and today we're going to take it one step further.

**Host B**: Yes! Today we'll cover three core topics. First, polynomial regression to capture nonlinear patterns. Second, regularization to prevent overfitting. Third, logistic regression for classification problems!

**Host A**: If linear regression can only draw straight lines, how do we draw curves?

**Host B**: Great question! Let's move right into polynomial regression.

---

**[Section 1: Polynomial Regression and Basis Expansion - 1:30]**

**Host B**: The core idea of polynomial regression is expanding x to x, x², x³, etc. y = β₀ + β₁x + β₂x² + β₃x³ + ...

**Host A**: So it's still linear with respect to the parameters?

**Host B**: Exactly! It's nonlinear with respect to input x, but linear with respect to the β's, so we can use the same optimization methods. This is called Basis Expansion.

**Host A**: How do you do it in sklearn?

**Host B**: Transform features with PolynomialFeatures and apply regular LinearRegression. Simple! But if you make the degree too high, problems occur.

**Host A**: Overfitting?

**Host B**: Correct! It fits the training data perfectly, but performs terribly on new data. The curve becomes too wiggly.

---

**[Section 2: Ridge Regression (L2 Regularization) - 3:30]**

**Host A**: How do you prevent overfitting?

**Host B**: Regularization! Ridge regression adds an L2 penalty to the loss function. L = Σ(y - ŷ)² + λΣβ²

**Host A**: What is λ?

**Host B**: It's the regularization strength. As λ increases, it strongly constrains coefficients to be small. When coefficients are large, the model's output changes greatly even with small changes in input.

**Host A**: How do you solve it mathematically?

**Host B**: The normal equation changes slightly. β = (XᵀX + λI)⁻¹Xᵀy. Adding λI ensures the inverse always exists, solving multicollinearity problems too!

**Host A**: Do all coefficients become small?

**Host B**: Yes, they approach 0 but don't become exactly 0. You keep all features while just reducing their magnitudes.

---

**[Section 3: Lasso Regression (L1 Regularization) - 5:30]**

**Host A**: How is Lasso different?

**Host B**: Lasso uses an L1 penalty. L = Σ(y - ŷ)² + λΣ|β|. Absolute values!

**Host A**: What's the difference between squaring and absolute value?

**Host B**: That's the key difference! L1 can make coefficients exactly 0. In other words, it automatically performs Feature Selection!

**Host A**: Why does that happen?

**Host B**: Geometrically, the L1 constraint region is diamond-shaped, so the optimal solution is more likely to occur at vertices. Vertices are points where some coefficients are 0.

**Host A**: So when do you use Ridge, when Lasso?

**Host B**: If all features seem important, use Ridge. If only some seem important, use Lasso. If interpretability is important, Lasso is good.

---

**[Section 4: Elastic Net and Feature Selection - 7:30]**

**Host A**: Can we combine the two?

**Host B**: Elastic Net does exactly that! L = Σ(y - ŷ)² + λ₁Σ|β| + λ₂Σβ². It uses both L1 and L2.

**Host A**: When is it useful?

**Host B**: When there are many correlated features. Lasso tends to select only one of correlated features, but Elastic Net selects them as a group. Thanks to L2's stability.

**Host A**: How can you know feature importance?

**Host B**: After standardization, look at coefficient magnitude |β|. In Lasso, non-zero coefficients are important features. There are also methods like RFE (Recursive Feature Elimination).

---

**[Section 5: From Regression to Classification - 9:00]**

**Host A**: Now shall we move to classification? Why can't we do classification with linear regression?

**Host B**: Because there's no limit on the output! Linear regression output can range from -∞ to +∞, but in classification we need probabilities between 0 and 1.

**Host A**: Can't we just cut at 0.5?

**Host B**: There are problems. If there are outliers, the decision boundary gets pushed significantly. And you can't interpret results as probabilities.

**Host A**: What about the Perceptron?

**Host B**: It's a historically important algorithm! It classifies with sign(wᵀx + b) and updates weights for each misclassified sample. But convergence is only guaranteed for linearly separable data.

---

**[Section 6: Sigmoid and Logistic Regression - 10:30]**

**Host A**: So that's where the sigmoid function comes in!

**Host B**: Right! σ(z) = 1 / (1 + e⁻ᶻ). It maps any real number to between 0 and 1. When z is large, it's close to 1, when z is small, close to 0.

**Host A**: What's the logistic regression model?

**Host B**: P(Y=1|X) = σ(wᵀx + b). You pass the linear combination through sigmoid to get a probability. The decision boundary is wᵀx + b = 0, where the probability is exactly 0.5.

**Host A**: Why "logistic"?

**Host B**: It comes from the concept of Odds. p/(1-p) is the odds, and log(p/(1-p)) is called the logit. Logistic regression assumes the logit is linear!

---

**[Section 7: Loss Function and Optimization - 12:00]**

**Host A**: How do you train logistic regression?

**Host B**: We use Maximum Likelihood Estimation (MLE). Find parameters that maximize the probability of observed data. Flip this and it becomes Binary Cross-Entropy loss.

**Host A**: What's the formula?

**Host B**: L = -1/n Σ[y log(p) + (1-y) log(1-p)]. When y=1, loss is small if p is large, when y=0, loss is small if p is small.

**Host A**: Can you solve it all at once like the normal equation?

**Host B**: No, there's no closed-form solution so we must use Gradient Descent. Fortunately it's a convex function so we can find the global minimum.

---

**[Section 8: Multi-class Classification and Regularization - 13:30]**

**Host A**: What if there are 3 or more classes?

**Host B**: There are two methods. One-vs-Rest creates K binary classifiers, and Softmax regression models all classes simultaneously.

**Host A**: What's Softmax?

**Host B**: P(Y=k|X) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx). It's a function that normalizes so all class probabilities sum to 1. It's widely used in neural network output layers.

**Host A**: Do you apply regularization to logistic regression too?

**Host B**: Of course! L2 regularization is standard. sklearn's LogisticRegression has L2 enabled by default. The C parameter is 1/λ, so smaller C means stronger regularization.

**Host A**: What are practical application tips?

**Host B**: Feature scaling is essential! If there's class imbalance, use the class_weight='balanced' option. And the threshold doesn't always have to be 0.5 - adjust it to fit the situation.

---

**[Outro - 14:30]**

**Host A**: Shall we summarize today?

**Host B**: First, polynomial regression can capture nonlinear patterns, but watch out for overfitting!

**Host A**: Second, Ridge makes coefficients small, Lasso makes them 0. Elastic Net is a combination of both!

**Host B**: Third, in classification, we use sigmoid to model probabilities. That's logistic regression!

**Host A**: Fourth, we train with Cross-Entropy loss and Gradient Descent, and can do multi-class classification with Softmax!

**Host B**: Logistic regression is also a foundation for deep learning. We'll dive deeper next time!

**Host A**: Thank you!

---

## Key Keywords
- Polynomial Regression, Basis Expansion
- Ridge (L2), Lasso (L1), Elastic Net
- Feature Selection, Regularization Strength
- Perceptron, Linear Separability
- Sigmoid Function, Logit, Odds
- Logistic Regression, Binary Cross-Entropy
- Maximum Likelihood Estimation, Gradient Descent
- Softmax, Categorical Cross-Entropy
