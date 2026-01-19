# Lecture 06 Podcast: 모델 평가의 모든 것 - 메트릭과 검증 전략

## 에피소드 정보
- **주제**: 데이터 분할, 회귀/분류 평가 지표, 교차 검증
- **예상 시간**: 15분
- **대상**: ML 모델을 제대로 평가하고 싶은 모든 분들

---

## 스크립트

**[인트로 - 0:00]**

**Host A**: 안녕하세요! AI 비전 시스템 팟캐스트입니다. 오늘은 모델 평가에 대해 다뤄볼 거예요.

**Host B**: 모델 학습만큼이나 평가가 중요해요! 잘못된 평가는 잘못된 결정으로 이어지니까요.

**Host A**: "내 모델 정확도 99%야!"라고 자랑하는 사람 보면 의심부터 하게 되더라고요.

**Host B**: 하하, 맞아요! 그런 경우 대부분 데이터 누수(Data Leakage)가 있거나, 불균형 데이터에서 accuracy만 봤을 가능성이 높죠. 오늘 그런 함정들을 피하는 방법을 배울 거예요.

---

**[섹션 1: 데이터 분할의 기본 - 1:30]**

**Host A**: 먼저 Train/Validation/Test 분할부터 얘기해볼까요?

**Host B**: 기본 중의 기본이에요! Training Set은 60-80%로 모델 학습에, Validation Set은 10-20%로 하이퍼파라미터 튜닝에, Test Set은 10-20%로 최종 평가에만 써요.

**Host A**: Test Set은 정말 마지막에 한 번만 써야 하죠?

**Host B**: 네! 개발 중에 Test Set으로 계속 확인하면, 그것도 학습에 영향을 주는 셈이에요. Test Set은 봉인해두고 마지막에만 열어야 해요.

**Host A**: 무작위로 나누면 되나요?

**Host B**: 분류 문제라면 Stratified Sampling을 써야 해요. 클래스 비율을 유지하면서 나누는 거죠. stratify=y 옵션 하나면 돼요.

---

**[섹션 2: 과적합과 과소적합 - 3:30]**

**Host A**: Overfitting이랑 Underfitting은 뭐가 달라요?

**Host B**: Underfitting은 모델이 너무 단순해서 훈련 데이터도 못 맞추는 거예요. 훈련 손실이 안 줄어들어요. 모델 용량을 늘려야 해요.

**Host A**: Overfitting은 반대로요?

**Host B**: 훈련 데이터는 완벽히 맞추는데 테스트에서 망하는 거예요. 모델이 패턴이 아니라 노이즈까지 암기한 거죠. 정규화, 더 많은 데이터, 더 단순한 모델로 해결해요.

**Host A**: Bias-Variance Tradeoff도 관련 있죠?

**Host B**: Total Error = Bias² + Variance + Irreducible Error예요. 높은 Bias는 과소적합, 높은 Variance는 과적합. 둘 다 줄이려면 균형점을 찾아야 해요.

---

**[섹션 3: 데이터 누수 방지 - 5:00]**

**Host A**: Data Leakage가 뭐예요?

**Host B**: 테스트 정보가 훈련에 섞여 들어가는 거예요. 가장 흔한 실수가 전체 데이터로 정규화하고 나서 분할하는 거예요.

**Host A**: 왜 문제예요?

**Host B**: 테스트 데이터의 평균과 표준편차가 훈련에 반영되잖아요! 분할 먼저, 전처리 나중에가 원칙이에요. sklearn의 Pipeline을 쓰면 자동으로 지켜져요.

**Host A**: 시계열 데이터는 더 조심해야 하죠?

**Host B**: 무작위 분할하면 안 돼요! 미래 데이터로 과거를 예측하게 되니까요. 항상 과거로 훈련하고 미래로 테스트해야 해요. Walk-forward나 Rolling Window 방식을 써요.

---

**[섹션 4: 회귀 평가 지표 - 7:00]**

**Host A**: 회귀 모델은 어떤 지표로 평가해요?

**Host B**: MSE, RMSE, MAE가 기본이에요. MSE는 Σ(y-ŷ)²/n, 큰 오차에 더 큰 페널티를 줘요. RMSE는 √MSE로 단위가 원래 타겟과 같아서 해석하기 좋아요.

**Host A**: MAE는요?

**Host B**: Σ|y-ŷ|/n으로 이상치에 덜 민감해요. 이상치가 많으면 MAE가 더 강건해요.

**Host A**: R²는 어떤 의미예요?

**Host B**: 설명된 분산의 비율이에요. R² = 1 - (잔차제곱합/총제곱합). 1이면 완벽, 0이면 평균만 예측하는 수준, 음수면 평균보다 못한 거예요!

**Host A**: R² 높으면 좋은 모델인 거죠?

**Host B**: 조심해야 해요! R²만 보면 안 되고 잔차 플롯도 봐야 해요. 패턴이 보이면 모델이 뭔가 놓치고 있는 거예요.

---

**[섹션 5: 분류 평가 지표 - 9:00]**

**Host A**: 분류 지표는 Confusion Matrix부터 시작하죠?

**Host B**: 네! TP(맞게 예측한 양성), TN(맞게 예측한 음성), FP(잘못된 양성, Type I 에러), FN(놓친 양성, Type II 에러). 모든 분류 지표가 여기서 나와요.

**Host A**: Accuracy = (TP+TN)/(전체)죠?

**Host B**: 맞아요. 근데 클래스 불균형이 있으면 의미 없어요! 99%가 음성인 데이터에서 모두 음성으로 예측해도 99% 정확도가 나오니까요.

**Host A**: Precision이랑 Recall은요?

**Host B**: Precision = TP/(TP+FP), "양성으로 예측한 것 중 실제 양성 비율"이에요. Recall = TP/(TP+FN), "실제 양성 중 찾아낸 비율"이고요.

**Host A**: 어떤 걸 더 중시해야 해요?

**Host B**: 상황에 따라요! 스팸 필터는 FP(정상 메일을 스팸으로)가 나쁘니까 Precision 중시. 암 진단은 FN(암을 놓침)이 치명적이니까 Recall 중시. F1은 둘의 조화평균이에요.

---

**[섹션 6: ROC와 PR 커브 - 11:00]**

**Host A**: ROC 커브는 뭐예요?

**Host B**: X축이 FPR(False Positive Rate), Y축이 TPR(Recall)이에요. 임계값을 바꿔가며 그리는 커브죠. 대각선은 랜덤 분류기, 왼쪽 위로 갈수록 좋아요.

**Host A**: AUC는요?

**Host B**: Area Under Curve, ROC 아래 면적이에요. 0.5면 랜덤, 1이면 완벽. "랜덤 양성이 랜덤 음성보다 높은 점수를 받을 확률"로 해석할 수 있어요.

**Host A**: PR 커브는 언제 써요?

**Host B**: 불균형 데이터에서! ROC는 TN이 많으면 FPR이 낮게 나와서 좋아 보일 수 있어요. PR 커브는 양성 클래스에 집중해서 더 현실적인 평가를 해줘요.

---

**[섹션 7: 교차 검증 - 12:30]**

**Host A**: Cross Validation은 왜 필요해요?

**Host B**: 단일 분할은 운에 좌우될 수 있어요. K-fold CV는 데이터를 K개로 나눠서, 각 fold를 돌아가며 검증에 쓰고 나머지로 훈련해요. K번의 결과를 평균내면 더 안정적이에요.

**Host A**: K는 보통 얼마예요?

**Host B**: 5나 10이 흔해요. 분류면 Stratified K-fold를 써서 각 fold에도 클래스 비율을 유지하세요.

**Host A**: LOOCV는요?

**Host B**: Leave-One-Out, K=n인 거예요. 샘플 하나씩 빼고 나머지로 훈련. 데이터가 아주 적을 때(100개 이하) 좋지만, 계산이 많이 필요해요.

**Host A**: 하이퍼파라미터 튜닝은요?

**Host B**: GridSearchCV로 모든 조합을 시도하거나, RandomizedSearchCV로 랜덤 샘플링해요. 최근에는 Optuna 같은 베이지안 최적화도 많이 써요. 핵심은 CV와 함께 써야 한다는 거예요!

---

**[섹션 8: 완전한 평가 파이프라인 - 14:00]**

**Host A**: 처음부터 끝까지 순서를 정리해주세요!

**Host B**: 첫째, Train/Test 분할(Stratified). 둘째, Train에서 CV로 하이퍼파라미터 튜닝. 셋째, 최적 파라미터로 전체 Train에서 재훈련. 넷째, Test로 최종 평가(한 번만!).

**Host A**: Nested CV는요?

**Host B**: 외부 루프는 일반화 성능 추정, 내부 루프는 하이퍼파라미터 선택. 같은 데이터를 선택과 평가에 쓰면 낙관적 편향이 생기는데, 이걸 방지해요.

**Host A**: 결과 보고는 어떻게 해요?

**Host B**: 평균 ± 표준편차로 보고하세요! "정확도 85%"보다 "정확도 85.2% ± 2.3% (5-fold CV)"가 훨씬 정보가 많아요.

---

**[아웃트로 - 14:30]**

**Host A**: 오늘 핵심을 정리해볼까요?

**Host B**: 첫째, Test Set은 봉인하고 마지막에만 열어요!

**Host A**: 둘째, 불균형 데이터에서 Accuracy만 보면 안 돼요. Precision, Recall, F1, AUC를 함께 보세요.

**Host B**: 셋째, 데이터 누수를 조심하세요. 분할 먼저, 전처리 나중에!

**Host A**: 넷째, K-fold CV로 더 안정적인 성능 추정을 하고, 평균과 표준편차를 함께 보고하세요!

**Host B**: 다음 시간에는 CNN으로 이미지를 다루는 방법을 배울 거예요!

**Host A**: 감사합니다!

---

## 핵심 키워드
- Train/Validation/Test Split, Stratified Sampling
- Overfitting, Underfitting, Bias-Variance Tradeoff
- Data Leakage, Time Series Split
- MSE, RMSE, MAE, R², MAPE
- Confusion Matrix, Precision, Recall, F1 Score
- ROC Curve, AUC, PR Curve, AP
- K-fold CV, Stratified K-fold, LOOCV
- GridSearchCV, RandomizedSearchCV, Nested CV
