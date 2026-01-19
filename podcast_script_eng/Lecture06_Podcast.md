# Lecture 06 Podcast: Everything About Model Evaluation - Metrics and Validation Strategies

## Episode Information
- **Topic**: Data Splitting, Regression/Classification Metrics, Cross-Validation
- **Duration**: 15 minutes
- **Audience**: Everyone who wants to properly evaluate ML models

---

## Script

**[Intro - 0:00]**

**Host A**: Hello! Welcome to the AI Vision System podcast. Today we'll be covering model evaluation.

**Host B**: Model evaluation is just as important as training! Wrong evaluation leads to wrong decisions.

**Host A**: When I see someone bragging "My model has 99% accuracy!", I immediately become suspicious.

**Host B**: Haha, exactly! In most cases, there's either Data Leakage or they're just looking at accuracy on imbalanced data. Today we'll learn how to avoid these pitfalls.

---

**[Section 1: Basics of Data Splitting - 1:30]**

**Host A**: Let's start with Train/Validation/Test splits, shall we?

**Host B**: The fundamentals! Training Set is 60-80% for model training, Validation Set is 10-20% for hyperparameter tuning, and Test Set is 10-20% for final evaluation only.

**Host A**: The Test Set should really only be used once at the very end, right?

**Host B**: Yes! If you keep checking with the Test Set during development, that affects your training too. The Test Set should be sealed and only opened at the end.

**Host A**: Can we just split randomly?

**Host B**: For classification problems, you should use Stratified Sampling. It splits while maintaining class ratios. Just one stratify=y option is all you need.

---

**[Section 2: Overfitting and Underfitting - 3:30]**

**Host A**: What's the difference between Overfitting and Underfitting?

**Host B**: Underfitting is when the model is too simple to even fit the training data. The training loss doesn't decrease. You need to increase model capacity.

**Host A**: And Overfitting is the opposite?

**Host B**: It perfectly fits the training data but fails on the test set. The model has memorized noise rather than patterns. Solve it with regularization, more data, or simpler models.

**Host A**: The Bias-Variance Tradeoff is related, right?

**Host B**: Total Error = Bias² + Variance + Irreducible Error. High Bias means underfitting, high Variance means overfitting. You need to find the sweet spot to reduce both.

---

**[Section 3: Preventing Data Leakage - 5:00]**

**Host A**: What is Data Leakage?

**Host B**: It's when test information leaks into training. The most common mistake is normalizing the entire dataset before splitting.

**Host A**: Why is that a problem?

**Host B**: The mean and standard deviation of the test data gets reflected in training! The principle is: split first, preprocess later. sklearn's Pipeline automatically enforces this.

**Host A**: You need to be extra careful with time series data, right?

**Host B**: Never use random splitting! You'd be predicting the past with the future. Always train on the past and test on the future. Use Walk-forward or Rolling Window approaches.

---

**[Section 4: Regression Evaluation Metrics - 7:00]**

**Host A**: What metrics do you use to evaluate regression models?

**Host B**: MSE, RMSE, and MAE are fundamental. MSE is Σ(y-ŷ)²/n, giving bigger penalties to larger errors. RMSE is √MSE with the same units as the original target, making it easier to interpret.

**Host A**: What about MAE?

**Host B**: Σ|y-ŷ|/n, it's less sensitive to outliers. If you have many outliers, MAE is more robust.

**Host A**: What does R² mean?

**Host B**: It's the proportion of variance explained. R² = 1 - (residual sum of squares/total sum of squares). 1 is perfect, 0 means you're just predicting the mean, and negative means you're worse than the mean!

**Host A**: So higher R² means a better model?

**Host B**: Be careful! Don't just look at R², also check residual plots. If you see patterns, the model is missing something.

---

**[Section 5: Classification Evaluation Metrics - 9:00]**

**Host A**: For classification, we start with the Confusion Matrix, right?

**Host B**: Yes! TP (correctly predicted positive), TN (correctly predicted negative), FP (false positive, Type I error), FN (missed positive, Type II error). All classification metrics come from here.

**Host A**: Accuracy = (TP+TN)/(total), right?

**Host B**: Correct. But it's meaningless with class imbalance! In data that's 99% negative, predicting everything as negative gives you 99% accuracy.

**Host A**: What about Precision and Recall?

**Host B**: Precision = TP/(TP+FP), "the proportion of true positives among predicted positives". Recall = TP/(TP+FN), "the proportion of actual positives that were found".

**Host A**: Which should you prioritize?

**Host B**: It depends! Spam filters prioritize Precision because FP (marking normal mail as spam) is bad. Cancer diagnosis prioritizes Recall because FN (missing cancer) is critical. F1 is the harmonic mean of both.

---

**[Section 6: ROC and PR Curves - 11:00]**

**Host A**: What is an ROC curve?

**Host B**: X-axis is FPR (False Positive Rate), Y-axis is TPR (Recall). You draw the curve by varying the threshold. The diagonal is a random classifier, the closer to the upper left, the better.

**Host A**: What about AUC?

**Host B**: Area Under Curve, the area under the ROC. 0.5 is random, 1 is perfect. It can be interpreted as "the probability that a random positive scores higher than a random negative".

**Host A**: When do you use PR curves?

**Host B**: With imbalanced data! ROC can look good when there are many TNs because FPR becomes low. PR curves focus on the positive class for a more realistic evaluation.

---

**[Section 7: Cross-Validation - 12:30]**

**Host A**: Why do we need Cross Validation?

**Host B**: A single split can be influenced by luck. K-fold CV splits the data into K parts, uses each fold as validation while training on the rest. Averaging K results gives more stability.

**Host A**: What's a typical K?

**Host B**: 5 or 10 are common. For classification, use Stratified K-fold to maintain class ratios in each fold.

**Host A**: What about LOOCV?

**Host B**: Leave-One-Out, where K=n. Leave one sample out and train on the rest. Good when data is very small (under 100), but computationally expensive.

**Host A**: What about hyperparameter tuning?

**Host B**: GridSearchCV tries all combinations, or RandomizedSearchCV samples randomly. Recently, Bayesian optimization like Optuna is popular too. The key is to use it with CV!

---

**[Section 8: Complete Evaluation Pipeline - 14:00]**

**Host A**: Can you outline the process from start to finish?

**Host B**: First, Train/Test split (Stratified). Second, CV on Train for hyperparameter tuning. Third, retrain on entire Train with optimal parameters. Fourth, final evaluation on Test (only once!).

**Host A**: What about Nested CV?

**Host B**: The outer loop estimates generalization performance, the inner loop selects hyperparameters. It prevents optimistic bias that occurs when using the same data for selection and evaluation.

**Host A**: How do you report results?

**Host B**: Report mean ± standard deviation! "85.2% ± 2.3% accuracy (5-fold CV)" provides much more information than just "85% accuracy".

---

**[Outro - 14:30]**

**Host A**: Let's summarize today's key points?

**Host B**: First, seal the Test Set and only open it at the end!

**Host A**: Second, don't just look at Accuracy on imbalanced data. Look at Precision, Recall, F1, and AUC together.

**Host B**: Third, watch out for data leakage. Split first, preprocess later!

**Host A**: Fourth, use K-fold CV for more stable performance estimates, and report both mean and standard deviation!

**Host B**: Next time we'll learn how to handle images with CNNs!

**Host A**: Thank you!

---

## Key Keywords
- Train/Validation/Test Split, Stratified Sampling
- Overfitting, Underfitting, Bias-Variance Tradeoff
- Data Leakage, Time Series Split
- MSE, RMSE, MAE, R², MAPE
- Confusion Matrix, Precision, Recall, F1 Score
- ROC Curve, AUC, PR Curve, AP
- K-fold CV, Stratified K-fold, LOOCV
- GridSearchCV, RandomizedSearchCV, Nested CV
