# Lecture 19 Podcast: 설명가능한 AI (XAI) 입문 - 모델 해석과 Feature Importance

## 에피소드 정보
- **주제**: XAI 개념, Feature Importance, PDP, ICE, LIME, Anchor
- **예상 시간**: 15분
- **대상**: ML 모델을 해석하고 설명해야 하는 모든 분들

---

## 스크립트

**[인트로 - 0:00]**

**Host A**: 안녕하세요! AI 비전 시스템 팟캐스트입니다. 오늘은 설명가능한 AI, XAI에 대해 다뤄볼 거예요.

**Host B**: 모델 성능만큼이나 "왜 이런 예측을 했는지"가 중요해졌어요! Black box 모델을 열어보는 시간이 될 겁니다.

**Host A**: 맞아요. "내 모델이 정확도 95%야!"라고 해도, 왜 그런 결정을 내렸는지 설명 못 하면 실무에서 쓰기 어려운 경우가 많죠.

**Host B**: 특히 의료, 금융, 법률 분야에서는 설명 없이는 AI를 도입할 수 없어요. 오늘 그 해결책들을 배워볼 거예요!

---

**[섹션 1: XAI가 필요한 이유 - 1:30]**

**Host A**: 먼저 XAI가 왜 필요한지부터 얘기해볼까요?

**Host B**: 여러 이유가 있어요. 첫째, 신뢰예요. 이해관계자들이 AI 결정을 신뢰하려면 이유를 알아야 해요. 의사가 "AI가 암이래요"라고만 하면 환자가 받아들이기 어렵잖아요.

**Host A**: 디버깅도 중요하죠?

**Host B**: 매우 중요해요! 모델이 이상한 예측을 할 때 왜 그런지 알아야 고칠 수 있어요. 그리고 규제 준수도 있어요. EU GDPR에는 "설명을 받을 권리"가 명시되어 있거든요.

**Host A**: 모델 개선에도 도움이 되나요?

**Host B**: 물론이죠! 어떤 특징이 중요한지 알면 더 나은 feature engineering이 가능하고, 모델 안전성 검증에도 필수예요.

---

**[섹션 2: XAI 분류 체계 - 3:00]**

**Host A**: XAI 방법들은 어떻게 분류하나요?

**Host B**: 세 가지 축으로 분류해요. 첫째, Scope로 Global과 Local이 있어요. Global은 전체 모델 행동을 설명하고, Local은 개별 예측을 설명해요.

**Host A**: 둘째는요?

**Host B**: Stage예요. Ante-hoc은 본질적으로 해석 가능한 모델, 즉 선형 회귀나 결정 트리 같은 거예요. Post-hoc은 학습 후에 설명을 추출하는 방식이죠.

**Host A**: 셋째는요?

**Host B**: Dependency예요. Model-specific은 특정 모델 유형에만 적용되고, Model-agnostic은 어떤 모델이든 적용 가능해요. 오늘은 주로 model-agnostic 방법들을 다룰 거예요.

---

**[섹션 3: Interpretability vs Accuracy Trade-off - 4:30]**

**Host A**: 해석 가능성과 정확도 사이에 trade-off가 있다고 하던데요?

**Host B**: 맞아요. 일반적으로 더 강력한 모델일수록 해석이 어려워요. Linear Regression과 Decision Tree는 해석이 쉽지만, Random Forest와 Neural Network는 어렵죠.

**Host A**: 그럼 항상 복잡한 모델을 쓰고 post-hoc 설명을 해야 하나요?

**Host B**: 상황에 따라 달라요. 고위험 결정이면서 데이터가 단순하면 해석 가능한 모델을 쓰세요. 복잡한 태스크에서 정확도가 필수면 post-hoc 설명을 쓰고요. 규제 준수가 필요하면 global과 local 설명 모두 준비하세요.

---

**[섹션 4: Permutation Importance - 6:00]**

**Host A**: 이제 구체적인 방법들로 들어가볼까요? Permutation Importance부터요.

**Host B**: 아이디어가 정말 직관적이에요. 특정 feature의 값을 무작위로 섞었을 때 성능이 얼마나 떨어지는지 보는 거예요. 많이 떨어지면 중요한 feature인 거죠.

**Host A**: 과정을 설명해주세요.

**Host B**: 먼저 모델을 학습하고 기준 성능을 측정해요. 그다음 하나의 feature를 shuffle하고, 성능 하락을 측정해요. 이 하락 폭이 중요도가 되죠. 모든 feature에 대해 반복하면 돼요.

**Host A**: sklearn에서 어떻게 쓰나요?

**Host B**: from sklearn.inspection import permutation_importance 하고, result = permutation_importance(model, X_test, y_test, n_repeats=10)으로 쉽게 계산할 수 있어요. model-agnostic이라 어떤 모델이든 적용 가능해요!

---

**[섹션 5: PDP와 ICE - 7:30]**

**Host A**: PDP는 뭔가요?

**Host B**: Partial Dependence Plot이에요. 특정 feature가 변할 때 예측이 어떻게 변하는지 보여줘요. 다른 feature들의 값은 평균 내서 marginal effect를 보는 거죠.

**Host A**: 수식으로는요?

**Host B**: PD(x_s) = (1/n) * 시그마 f(x_s, x_c^(i))예요. x_s는 관심 feature, x_c는 나머지 feature들이에요. 모든 샘플에 대해 평균을 내는 거죠.

**Host A**: ICE는 뭐가 다른가요?

**Host B**: Individual Conditional Expectation이에요. PDP가 평균이라면, ICE는 각 인스턴스별로 선을 그려요. 이렇게 하면 heterogeneity, 즉 개인차를 볼 수 있어요. 어떤 사람에게는 feature 효과가 크고 어떤 사람에게는 작을 수 있거든요.

---

**[섹션 6: ALE와 Feature Interaction - 9:00]**

**Host A**: PDP의 한계는 뭔가요?

**Host B**: Feature 독립성을 가정해요. 하지만 실제로 feature들은 상관관계가 있잖아요. 예를 들어 키와 몸무게는 상관있는데, PDP는 이걸 무시하고 불가능한 조합도 평가해요.

**Host A**: 그래서 ALE가 나온 거군요?

**Host B**: 맞아요! Accumulated Local Effects예요. Conditional distribution을 사용해서 상관된 feature를 더 잘 처리해요. Local effect를 계산하고 누적하는 방식이라 PDP보다 정확하고 빠르기도 해요.

**Host A**: Feature interaction 분석은요?

**Host B**: H-statistic으로 interaction 강도를 측정할 수 있어요. 두 feature의 joint PDP에서 individual PDP를 뺀 게 interaction이에요. SHAP interaction values도 많이 쓰이는데, 이건 다음 시간에 자세히 다룰게요.

---

**[섹션 7: LIME의 원리 - 10:30]**

**Host A**: LIME은 정말 많이 들어봤는데, 정확히 어떤 원리인가요?

**Host B**: Local Interpretable Model-agnostic Explanations의 약자예요. 핵심 아이디어는 복잡한 모델을 국소적으로 단순한 모델로 근사하는 거예요.

**Host A**: 과정을 설명해주세요.

**Host B**: 다섯 단계예요. 첫째, 설명하려는 인스턴스 주변에서 perturbed sample을 생성해요. 둘째, 이 샘플들에 대해 black box 예측을 얻어요. 셋째, 원래 인스턴스와의 거리로 가중치를 부여해요. 넷째, 가중치를 적용해서 선형 모델을 학습해요. 다섯째, 선형 계수가 local explanation이 돼요!

**Host A**: 데이터 타입별로 다르게 적용하나요?

**Host B**: 네! Tabular data는 feature 값을 perturb하고, 텍스트는 단어를 제거하고, 이미지는 superpixel을 on/off 해요. 각 도메인에 맞게 perturbation 전략이 달라요.

---

**[섹션 8: LIME 실습과 Anchor - 12:00]**

**Host A**: LIME을 코드로 어떻게 쓰나요?

**Host B**: lime 라이브러리를 설치하고, LimeTabularExplainer로 explainer를 만들어요. 그다음 explain_instance 메서드로 특정 인스턴스를 설명하면 돼요. show_in_notebook()으로 시각화까지 가능하죠.

**Host A**: Anchor Explanation은 뭔가요?

**Host B**: LIME의 확장인데, 규칙 형태로 설명을 줘요. "IF 조건들 THEN 예측 WITH 높은 정밀도" 형식이에요. 예를 들어 "IF age > 30 AND income > 50k THEN approved (95% precision)" 이런 식이죠.

**Host A**: LIME보다 나은 점이 뭔가요?

**Host B**: Coverage와 Precision을 명시적으로 제공해요. Coverage는 얼마나 많은 인스턴스에 적용되는지, Precision은 그 범위 내에서 얼마나 정확한지예요. 규칙 기반이라 비전문가도 이해하기 쉽고요.

---

**[섹션 9: 실무 가이드라인 - 13:30]**

**Host A**: 어떤 상황에서 어떤 방법을 쓰면 좋을까요?

**Host B**: 정리해드릴게요. 개별 예측 설명이 필요하면 LIME이나 SHAP을 쓰세요. Feature 효과를 보려면 PDP나 ALE, 전체 feature 중요도는 Permutation Importance, 규칙 기반 설명은 Anchor가 좋아요.

**Host A**: 검증은 어떻게 해요?

**Host B**: 세 가지를 확인하세요. 첫째, 설명이 말이 되는지 domain expert에게 검토받으세요. 둘째, 여러 방법의 결과를 비교하세요. 셋째, sanity check로 명백한 케이스에서 예상대로 작동하는지 확인하세요.

---

**[아웃트로 - 14:30]**

**Host A**: 오늘 핵심을 정리해볼까요?

**Host B**: 첫째, XAI는 신뢰, 디버깅, 규제 준수, 모델 개선을 위해 필수예요!

**Host A**: 둘째, Permutation Importance는 feature를 섞어서 중요도를 측정하고, PDP와 ICE는 feature 효과를 시각화해요.

**Host B**: 셋째, LIME은 복잡한 모델을 국소적으로 선형 모델로 근사해서 설명해요.

**Host A**: 넷째, 상황에 맞는 방법을 선택하고 항상 검증하는 게 중요해요!

**Host B**: 다음 시간에는 SHAP에 대해 깊이 다룰 거예요. 게임 이론 기반의 가장 강력한 설명 방법이죠!

**Host A**: 감사합니다!

---

## 핵심 키워드
- Explainable AI (XAI), Interpretability, Explainability
- Global vs Local Explanation, Model-Agnostic
- Permutation Importance, Drop-Column Importance
- Partial Dependence Plot (PDP), Individual Conditional Expectation (ICE)
- Accumulated Local Effects (ALE), Feature Interaction
- LIME (Local Interpretable Model-agnostic Explanations)
- Surrogate Model, Anchor Explanation
- Fidelity, Comprehensibility, Stability
