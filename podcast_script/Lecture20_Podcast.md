# Lecture 20 Podcast: SHAP과 딥러닝 XAI - 게임 이론부터 Grad-CAM까지

## 에피소드 정보
- **주제**: SHAP 이론, Shapley Values, TreeSHAP, DeepSHAP, Gradient 기반 방법, Grad-CAM
- **예상 시간**: 15분
- **대상**: ML/DL 모델 해석에 깊이 들어가고 싶은 분들

---

## 스크립트

**[인트로 - 0:00]**

**Host A**: 안녕하세요! AI 비전 시스템 팟캐스트입니다. 오늘은 XAI의 핵심인 SHAP과 딥러닝 설명 방법을 다뤄볼 거예요.

**Host B**: 지난 시간에 LIME을 배웠는데, 오늘 배울 SHAP은 이론적으로 더 탄탄해요! 게임 이론에서 출발하거든요.

**Host A**: 게임 이론이라니, 흥미롭네요. 그리고 CNN이 어디를 보는지 알려주는 Grad-CAM도 다루죠?

**Host B**: 네! 이미지 분류 모델이 고양이를 보고 왜 고양이라고 했는지, 어디를 봤는지 시각화하는 방법이에요. 시작해볼까요?

---

**[섹션 1: 게임 이론과 Shapley Values - 1:30]**

**Host A**: SHAP의 기반이 되는 Shapley Values부터 설명해주세요.

**Host B**: 협조 게임 이론에서 온 개념이에요. 상상해보세요. 여러 사람이 팀으로 일해서 수익을 냈어요. 이 수익을 공정하게 분배하려면 어떻게 해야 할까요?

**Host A**: 각자 기여도에 따라요?

**Host B**: 맞아요! 근데 기여도를 어떻게 계산할까요? Shapley Value는 각 플레이어의 평균 한계 기여도예요. 모든 가능한 순서에서 그 플레이어가 합류했을 때 가치 증가분의 평균이죠.

**Host A**: ML에서는 어떻게 적용해요?

**Host B**: Feature가 플레이어가 되고, 예측값이 payoff가 돼요. 각 feature가 예측에 얼마나 기여했는지 공정하게 분배하는 거예요.

---

**[섹션 2: SHAP의 수학적 기초 - 3:30]**

**Host A**: Shapley Value 수식을 설명해주세요.

**Host B**: phi_i = 시그마 [|S|!(|N|-|S|-1)!/|N|!] * [v(S union {i}) - v(S)]예요. S는 feature i를 제외한 부분집합, v(S)는 그 feature들만으로 예측한 값이에요.

**Host A**: 직관적으로 설명하면요?

**Host B**: Feature i가 없는 모든 조합 S에 대해, i가 추가됐을 때 예측값 변화를 계산하고, 적절한 가중치로 평균 내는 거예요. 가중치는 조합의 크기에 따라 결정되죠.

**Host A**: SHAP의 핵심 성질은 뭔가요?

**Host B**: 네 가지예요. Efficiency는 SHAP 값의 합이 예측값과 base value의 차이와 같다는 거예요. Symmetry는 동일한 기여면 동일한 값. Dummy는 기여 없으면 0. Additivity는 모델 간 일관성이에요. 이 성질들이 SHAP을 이론적으로 견고하게 만들어요.

---

**[섹션 3: SHAP 해석하기 - 5:30]**

**Host A**: SHAP 값은 어떻게 해석하나요?

**Host B**: 정말 직관적이에요. 양수 SHAP은 예측을 높이는 방향, 음수는 낮추는 방향이에요. 절대값이 크면 그만큼 영향력이 큰 거고요.

**Host A**: 예시를 들어주세요.

**Host B**: 집값 예측이라고 해볼게요. Base value가 평균 집값 2억이에요. 방 4개: +3천만원, 위치 도심: +5천만원, 건축연도 50년: -2천만원. 다 더하면 예측값 2억 6천만원이 돼요.

**Host A**: LIME과 비교하면요?

**Host B**: LIME은 local linear approximation이라 일관성이 보장 안 돼요. 같은 feature가 비슷한 상황에서 다른 중요도를 가질 수 있어요. SHAP은 이론적 성질 덕분에 일관성이 보장되고, global view로 자연스럽게 확장돼요.

---

**[섹션 4: SHAP 구현 방법들 - 7:30]**

**Host A**: 정확한 Shapley Value 계산은 어렵다던데요?

**Host B**: 네, O(2^n)이에요. Feature가 20개만 돼도 백만 개 이상의 조합을 봐야 해요. 그래서 효율적인 근사 방법들이 개발됐어요.

**Host A**: KernelSHAP은요?

**Host B**: LIME과 비슷한 아이디어인데, Shapley kernel로 가중치를 주는 weighted linear regression이에요. Sampling 기반이라 모든 모델에 적용 가능하지만 근사치예요.

**Host A**: TreeSHAP은요?

**Host B**: 트리 모델 전용인데, 정확한 SHAP을 다항 시간 O(TLD^2)에 계산해요! T는 트리 수, L은 리프 수, D는 깊이예요. XGBoost, LightGBM, Random Forest에 쓸 수 있고, 정확하면서도 빨라요.

**Host A**: DeepSHAP은요?

**Host B**: DeepLIFT와 Shapley를 결합한 거예요. 신경망에서 backpropagation을 활용해 attribution을 계산해요. 딥러닝 모델 설명에 효과적이죠.

---

**[섹션 5: SHAP 시각화 - 9:00]**

**Host A**: SHAP의 시각화 종류를 알려주세요.

**Host B**: 여러 가지가 있어요. 먼저 Waterfall Plot은 단일 예측을 분해해서 보여줘요. Base value에서 시작해서 각 feature가 더하거나 빼서 최종 예측에 도달하는 과정을 보여주죠.

**Host A**: Summary Plot은요?

**Host B**: Global feature importance와 분포를 동시에 보여줘요. Beeswarm 형태인데, 각 점이 하나의 인스턴스예요. X축이 SHAP 값, 색상이 feature 값의 높낮이를 나타내요. 예를 들어 어떤 feature가 높으면 항상 예측을 높이는지 낮추는지 한눈에 보여요.

**Host A**: Dependence Plot은요?

**Host B**: Feature 효과와 interaction을 보여줘요. X축이 feature 값, Y축이 SHAP 값, 색상이 자동으로 감지된 interaction feature예요. 비선형 효과와 interaction을 시각적으로 파악할 수 있어요.

---

**[섹션 6: 딥러닝 XAI - Gradient 기반 방법 - 10:30]**

**Host A**: 이제 딥러닝 특화 XAI로 넘어가볼까요?

**Host B**: 네! Gradient 기반 방법부터 볼게요. 가장 기본은 Saliency Map이에요. 출력의 입력에 대한 gradient 절대값, |df/dx|를 시각화해요. 어떤 픽셀이 예측에 민감한지 보여주죠.

**Host A**: 문제점은 없나요?

**Host B**: 노이즈가 많고 해석이 어려울 수 있어요. 그래서 SmoothGrad가 나왔는데, 노이즈를 추가한 여러 샘플의 gradient를 평균 내요. 더 깔끔한 결과가 나오죠.

**Host A**: Integrated Gradients는요?

**Host B**: 가장 이론적으로 탄탄해요. Baseline, 보통 검은 이미지에서 실제 이미지까지 경로를 따라 gradient를 적분해요. Completeness axiom을 만족해서 attribution의 합이 예측 차이와 같아요.

---

**[섹션 7: Grad-CAM - 12:00]**

**Host A**: Grad-CAM은 정말 많이 쓰이죠?

**Host B**: CNN 설명의 표준이 됐어요! Class Activation Mapping을 일반화한 거예요. 마지막 convolutional layer의 feature map을 class에 대한 gradient로 가중치를 매겨 시각화해요.

**Host A**: 수식은요?

**Host B**: L_c = ReLU(시그마_k alpha_k^c * A^k)예요. alpha_k^c는 gradient의 global average pooling이고, A^k는 k번째 feature map이에요. ReLU로 양수만 취해서 positive contribution만 보여줘요.

**Host A**: 코드로는요?

**Host B**: pytorch_grad_cam 라이브러리가 있어요. GradCAM(model, target_layers)로 객체 만들고, cam(input_tensor)로 쉽게 계산할 수 있어요. Grad-CAM++, Score-CAM 같은 변형도 있어요.

---

**[섹션 8: Concept-based Explanations - 13:30]**

**Host A**: 픽셀 단위 설명의 한계는 뭔가요?

**Host B**: 사람이 픽셀로 생각하지 않는다는 거예요. "이 픽셀들이 중요해요"보다 "줄무늬 패턴 때문에 호랑이로 분류했어요"가 더 이해하기 쉽잖아요.

**Host A**: TCAV가 그런 거죠?

**Host B**: 맞아요! Testing with Concept Activation Vectors예요. 사람이 정의한 concept, 예를 들어 줄무늬, 바퀴, 털 같은 것의 방향을 activation space에서 찾아요. 그리고 예측이 그 concept에 얼마나 민감한지 측정해요.

**Host A**: 미래 방향은요?

**Host B**: 몇 가지 트렌드가 있어요. Built-in interpretability, 즉 설계부터 해석 가능하게 만드는 것, multi-modal explanations로 텍스트와 이미지를 함께 설명하는 것, 그리고 causality integration으로 상관관계를 넘어 인과관계를 파악하는 것이 연구되고 있어요.

---

**[아웃트로 - 14:30]**

**Host A**: 오늘 핵심을 정리해볼까요?

**Host B**: 첫째, SHAP은 게임 이론의 Shapley Value에 기반해서 이론적으로 견고하고, 값의 합이 예측과 base의 차이와 같아요.

**Host A**: 둘째, TreeSHAP은 트리 모델에서 정확하고 빠르며, DeepSHAP은 신경망에 적용 가능해요.

**Host B**: 셋째, Gradient 기반 방법과 Grad-CAM으로 CNN이 어디를 보는지 시각화할 수 있어요.

**Host A**: 마지막으로, TCAV 같은 concept-based 방법으로 더 인간 친화적인 설명이 가능해요!

**Host B**: 이것으로 XAI 시리즈를 마무리합니다. 모델을 이해하고 설명하는 능력이 점점 더 중요해지고 있어요. 꼭 실습해보세요!

**Host A**: 감사합니다!

---

## 핵심 키워드
- SHAP (SHapley Additive exPlanations), Shapley Values
- Cooperative Game Theory, Feature Attribution
- Efficiency, Symmetry, Dummy, Additivity Properties
- KernelSHAP, TreeSHAP, DeepSHAP
- Waterfall Plot, Summary Plot, Dependence Plot
- Saliency Map, SmoothGrad, Integrated Gradients
- CAM, Grad-CAM, Grad-CAM++, Score-CAM
- TCAV (Testing with Concept Activation Vectors)
- Attention Mechanism, Faithfulness
