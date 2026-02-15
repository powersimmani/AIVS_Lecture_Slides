# Lecture 20 분석 보고서
## Model Explainability - SHAP and Deep Learning XAI

**분석 일자**: 2026-02-15
**품질 등급**: A+ (매우 우수)

---

## 1. 강의 구조 개요

| Part | 주제 | 슬라이드 | 평가 |
|------|------|----------|------|
| Part 1 | SHAP 이론 및 기초 | 03-11 | 매우 우수 |
| Part 2 | SHAP 구현 방법 | 12-19 | 매우 우수 |
| Part 3 | SHAP 시각화 및 분석 | 20-25 | 매우 우수 |
| Part 4 | 딥러닝 XAI 고급 기법 | 26-31 | 매우 우수 |

---

## 2. 긍정적 평가

### 2.1 Shapley Values 이론 (Part 1)
- **게임 이론 기초**: 협력 게임 이론 설명
- **Shapley Values 수학적 정의**:
  - φᵢ = Σ [|S|!(n-|S|-1)!/n!] × [v(S∪{i}) - v(S)]
  - 각 구성요소 상세 설명
  - 실제 계산 예시 (주택 가격 예측)
- **4가지 공정성 공리**:
  - Efficiency (효율성)
  - Symmetry (대칭성)
  - Dummy (더미)
  - Additivity (가법성)
- **SHAP vs LIME 비교**
- **SHAP 값 해석 방법**
- **복잡도 경고**: O(2ⁿ)

### 2.2 SHAP 구현 방법 (Part 2)
- **KernelSHAP**:
  - Model-agnostic
  - LIME 스타일 근사
- **TreeSHAP**:
  - 트리 모델 전용
  - O(TLD²) 빠른 계산
- **DeepSHAP (DeepLIFT + SHAP)**:
  - 딥러닝 모델용
  - 기울기 기반 근사
- **GradientSHAP**:
  - Integrated Gradients + SHAP
  - 노이즈 추가 평균
- **SHAP Approximation Techniques**: 근사 기법
- **SHAP Interaction Values**: 상호작용 값
- **Explainer 비교 실습**

### 2.3 SHAP 시각화 (Part 3)
- **Waterfall Plot**: 개별 예측 분해
- **Force Plot**: 힘 다이어그램
- **Decision Plot**: 누적 기여도
- **Summary Plot**: 전역 특징 중요도
- **Dependence Plot**: 특징-SHAP 관계
- **시계열 SHAP**
- **텍스트/이미지 SHAP**

### 2.4 딥러닝 XAI (Part 4)
- **Attention Mechanisms as Explanations**:
  - Transformer 어텐션 시각화
  - 한계점 언급
- **Gradient-based Methods**:
  - Vanilla Gradients
  - Integrated Gradients
  - SmoothGrad
  - Grad-CAM
  - PyTorch 구현 코드
- **CAM-family Methods**:
  - CAM → Grad-CAM → Grad-CAM++ → Score-CAM → Layer-CAM
  - 진화 타임라인
  - 의료 이미지 응용 예시
  - 수식 포함
- **Concept-based Explanations**: TCAV 등
- **XAI 미래 및 과제**

---

## 3. 개선 권장사항

### 3.1 [중요] Integrated Gradients 상세화

**위치**: Part 4 Gradient-based Methods

**현재 상태**: 수식만 제시

**추가 권장 내용**:
```markdown
## Integrated Gradients 상세

### 핵심 아이디어
- Baseline (x̄)에서 입력 (x)까지의 경로 적분
- Attribution = ∫ gradients along path

### 수식
IG_i(x) = (x_i - x̄_i) × ∫₀¹ ∂F(x̄ + α(x-x̄))/∂x_i dα

### Baseline 선택
| 데이터 유형 | 권장 Baseline |
|-------------|---------------|
| 이미지 | 검은 이미지 (zeros) |
| 텍스트 | [PAD] 토큰 |
| 표형 | 평균값 또는 zeros |

### 핵심 속성
1. **Sensitivity**: 차이 있으면 attribution ≠ 0
2. **Implementation Invariance**: 같은 함수 → 같은 attribution
3. **Completeness**: Σ IG_i = F(x) - F(x̄)

### 구현
```python
def integrated_gradients(model, input, baseline, steps=50):
    # 경로 생성
    alphas = torch.linspace(0, 1, steps)
    scaled_inputs = [baseline + alpha * (input - baseline)
                     for alpha in alphas]

    # 기울기 계산
    gradients = []
    for scaled_input in scaled_inputs:
        scaled_input.requires_grad_(True)
        output = model(scaled_input)
        output.backward()
        gradients.append(scaled_input.grad)

    # 적분 근사 (Riemann sum)
    avg_gradients = torch.mean(torch.stack(gradients), dim=0)
    integrated_gradients = (input - baseline) * avg_gradients

    return integrated_gradients
```

### vs Vanilla Gradients
| 항목 | Vanilla | Integrated |
|------|---------|------------|
| 노이즈 | 많음 | 적음 |
| Saturation | 영향받음 | 해결 |
| 이론 기반 | 약함 | 강함 |
```

---

### 3.2 [중요] Attention은 설명인가? 논쟁

**위치**: Part 4 Attention 섹션

**중요성**:
- 흔한 오해 해소
- 연구 동향 반영

**추가 권장**:
```markdown
## Attention as Explanation: 논쟁

### 긍정적 관점
- 직관적: "모델이 어디를 보는가"
- 시각화 용이
- 많은 연구에서 사용

### 부정적 관점 (Jain & Wallace, 2019)

#### 실험 결과
1. Attention ≠ Gradient-based importance
2. 다른 attention → 같은 예측 가능
3. Adversarial attention 쉽게 생성

#### 의미
- Attention은 faithful하지 않을 수 있음
- 설명 목적으로 무비판적 사용 위험

### 반박 (Wiegreffe & Pinter, 2019)
- "설명"의 정의에 따라 다름
- Attention은 유용한 정보 제공
- 완벽한 faithfulness 불필요할 수 있음

### 권장 접근
```
Attention 사용 시:
├── 단독 사용 ✗
├── 다른 방법과 교차 검증 ✓
└── "참고용" 시각화로 활용 ✓
```

### 대안
- Gradient × Attention
- Attribution Patching
- Probing classifiers
```

---

### 3.3 [중요] SHAP TreeExplainer 최적화

**위치**: Part 2 TreeSHAP

**추가 권장**:
```markdown
## TreeSHAP 상세 및 최적화

### 알고리즘 핵심
- 트리 구조를 활용한 효율적 계산
- 모든 가능한 경로 추적 (동적 프로그래밍)

### 복잡도
| Explainer | 복잡도 | 1000샘플 예상 시간 |
|-----------|--------|-------------------|
| KernelSHAP | O(2ᵐ) | 수 분~수 시간 |
| TreeSHAP | O(TLD²) | 수 초 |

where T=트리 수, L=최대 잎, D=깊이

### 사용법
```python
import shap

# XGBoost, LightGBM, CatBoost, sklearn RandomForest 지원
explainer = shap.TreeExplainer(model)

# feature_perturbation 옵션
# - 'interventional': 인과적 해석 (권장)
# - 'tree_path_dependent': 빠르지만 상관관계 포함
explainer = shap.TreeExplainer(
    model,
    feature_perturbation='interventional',
    data=X_background  # 배경 데이터 필요
)

shap_values = explainer.shap_values(X_test)
```

### 주의사항
- **상관된 특징**: tree_path_dependent는 왜곡 가능
- **배경 데이터**: interventional은 배경 필요
- **메모리**: 큰 데이터셋은 샘플링 권장
```

---

### 3.4 [권장] GradCAM++ 수식 추가

**위치**: Part 4 CAM-family

**추가 권장**:
```markdown
## Grad-CAM++ 상세

### Grad-CAM 한계
- 다중 객체 탐지 어려움
- 부분 객체 하이라이트

### Grad-CAM++ 개선
가중치를 고차 도함수로 계산:

### 수식
α_k^c = Σᵢⱼ w^{abc}_k × ReLU(∂y^c/∂A^k_ij)

where:
w^{abc}_k = (∂²y^c/∂A^k_ij)² / [2(∂²y^c/∂A^k_ij)² + Σₘₙ A^k_mn(∂³y^c/∂A^k_ij)³]

### 직관
- 2차, 3차 도함수 사용
- 양수 기울기에 더 가중
- 더 정확한 localization

### 구현
```python
from pytorch_grad_cam import GradCAMPlusPlus

gradcam_pp = GradCAMPlusPlus(model, target_layer=model.layer4[-1])
cam = gradcam_pp(input_tensor=image, target_category=class_idx)
```

### Grad-CAM vs Grad-CAM++
| 항목 | Grad-CAM | Grad-CAM++ |
|------|----------|------------|
| 다중 객체 | 약함 | 강함 |
| 계산 비용 | 낮음 | 중간 |
| 복잡도 | 단순 | 복잡 |
```

---

### 3.5 [권장] TCAV (Testing with CAVs) 추가

**위치**: Part 4 Concept-based

**추가 권장**:
```markdown
## TCAV (Testing with Concept Activation Vectors)

### 아이디어
- 사람이 이해하는 "개념"으로 설명
- "줄무늬가 얼마나 중요한가?" (얼룩말 분류)

### Concept Activation Vector (CAV)
1. 개념 예시 수집 (줄무늬 이미지 vs 비줄무늬)
2. 활성화 공간에서 선형 분류기 학습
3. 분류 경계 법선 벡터 = CAV

### TCAV Score
TCAV^c_k = |{x: S_{C,k,l}(x) > 0}| / |X_k|

where S = 개념 방향 민감도

### 해석
- TCAV = 1.0: 해당 개념이 항상 양의 영향
- TCAV = 0.5: 영향 없음 (random)
- TCAV = 0.0: 항상 음의 영향

### 예시
```python
# 얼룩말 분류에서 "줄무늬" 개념 테스트
# TCAV("얼룩말", "줄무늬") = 0.95 → 줄무늬가 중요
# TCAV("얼룩말", "점박이") = 0.52 → 무관
```

### 장점
- 인간 친화적 설명
- 편향 탐지 가능 (성별, 인종 개념)
- 다양한 개념 테스트 가능
```

---

### 3.6 [권장] XAI 도구 비교표

**위치**: Part 4 끝부분 또는 요약

**추가 권장**:
```markdown
## XAI 도구 선택 가이드

### 모델 유형별 권장
| 모델 유형 | 1순위 | 2순위 | 3순위 |
|----------|-------|-------|-------|
| Tree Ensemble | TreeSHAP | Permutation | LIME |
| Deep Neural Net | DeepSHAP | IG | GradCAM |
| CNN (이미지) | GradCAM | SHAP Image | LIME Image |
| Transformer | Attention | IG | SHAP |
| Linear/Logistic | 계수 직접 | SHAP | Permutation |

### 상황별 선택
```
빠른 전역 중요도 필요?
├── 트리 모델 → TreeSHAP (Summary)
├── 신경망 → Permutation Importance
└── 기타 → KernelSHAP

개별 예측 설명 필요?
├── 표형 데이터 → SHAP Force/Waterfall
├── 이미지 → GradCAM + SHAP Image
└── 텍스트 → LIME Text + Attention

디버깅/편향 탐지?
├── 개념 기반 → TCAV
├── 반사실 → Counterfactuals
└── 상호작용 → SHAP Interaction
```

### 라이브러리 비교
| 라이브러리 | 강점 | 모델 지원 |
|------------|------|-----------|
| shap | 포괄적, 시각화 우수 | 범용 |
| captum | PyTorch 통합 | DNN |
| lime | 간단, 범용 | 모델 불문 |
| pytorch-grad-cam | CAM 계열 특화 | CNN |
| alibi | 반사실, Anchors | 범용 |
| interpret | MS 지원, 통합 | 범용 |
```

---

## 4. 기술적 정확성 검증 결과

| 항목 | 슬라이드 | 검증 |
|------|----------|------|
| Shapley Value 수식 | 06 | ✅ 정확 |
| 4가지 공정성 공리 | 06, 10 | ✅ 정확 |
| TreeSHAP 복잡도 | 14 | ✅ 정확 |
| KernelSHAP 개념 | 13 | ✅ 정확 |
| GradCAM 수식 | 29 | ✅ 정확 |
| Integrated Gradients 수식 | 28 | ✅ 정확 |
| CAM 진화 타임라인 | 29 | ✅ 정확 |
| Shapley 계산 예시 | 06 | ✅ 정확 (효율성 검증) |

---

## 5. 우선순위별 작업 체크리스트

### 다음 업데이트 시 (중요)
- [ ] Integrated Gradients 상세화
- [ ] Attention 논쟁 추가
- [ ] TreeSHAP 최적화 가이드

### 시간 있을 때 (권장)
- [ ] GradCAM++ 수식 추가
- [ ] TCAV 상세 추가
- [ ] XAI 도구 비교표

### 선택적 개선
- [ ] Score-CAM 상세
- [ ] LRP (Layer-wise Relevance Propagation)
- [ ] Occlusion Sensitivity

---

## 6. 다른 강의와의 연계성

| 연계 강의 | 관련 내용 | 상태 |
|-----------|-----------|------|
| Lecture 19 (XAI 기초) | LIME, Feature Importance | ✅ 완벽한 연결 |
| Lecture 10 (CNN) | GradCAM 적용 대상 | ✅ 연계됨 |
| Lecture 13 (Transformer) | Attention 시각화 | ✅ 연계됨 |
| Lecture 06 (의사결정 트리) | TreeSHAP | ✅ 연계됨 |
| Lecture 14 (PLM) | LLM 해석 | ⚠️ LLM Probing 연결 권장 |

---

## 7. 특별 참고사항

### SHAP 값 해석 요약
```
SHAP 값 > 0: 해당 특징이 예측을 높이는 방향으로 기여
SHAP 값 < 0: 해당 특징이 예측을 낮추는 방향으로 기여
SHAP 값 = 0: 기여 없음

Σ SHAP_i = f(x) - E[f(X)]  (Efficiency)
```

### SHAP 시각화 선택
| 목적 | 시각화 | 범위 |
|------|--------|------|
| 개별 예측 분해 | Waterfall | Local |
| 다중 예측 비교 | Force (stacked) | Local |
| 전역 중요도 | Summary (bar) | Global |
| 특징 분포 + 효과 | Summary (dot) | Global |
| 특징 상호작용 | Dependence | Global |
| 누적 기여 경로 | Decision | Local |

### Grad-CAM 실전 코드
```python
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 모델과 타겟 레이어 설정
model = torchvision.models.resnet50(pretrained=True)
target_layers = [model.layer4[-1]]

# GradCAM 객체 생성
cam = GradCAM(model=model, target_layers=target_layers)

# 특정 클래스에 대한 CAM 생성
targets = [ClassifierOutputTarget(281)]  # 고양이 클래스
grayscale_cam = cam(input_tensor=image, targets=targets)

# 시각화
from pytorch_grad_cam.utils.image import show_cam_on_image
visualization = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
```

### SHAP Explainer 선택 가이드
```python
import shap

# 1. 트리 모델 (XGBoost, LightGBM, RandomForest)
explainer = shap.TreeExplainer(model)  # 가장 빠름

# 2. 딥러닝 (PyTorch, TensorFlow)
explainer = shap.DeepExplainer(model, background_data)  # 중간 속도

# 3. 범용 (어떤 모델이든)
explainer = shap.KernelExplainer(model.predict, background_data)  # 느림

# 4. 선형 모델
explainer = shap.LinearExplainer(model, background_data)  # 정확

# 5. Permutation 기반 (비교적 빠르고 범용)
explainer = shap.Explainer(model, background_data)  # 자동 선택
```

---

## 8. 참고 자료

- [SHAP Paper - Lundberg & Lee (2017)](https://arxiv.org/abs/1705.07874)
- [SHAP GitHub](https://github.com/slundberg/shap)
- [GradCAM Paper (2017)](https://arxiv.org/abs/1610.02391)
- [GradCAM++ Paper (2018)](https://arxiv.org/abs/1710.11063)
- [Integrated Gradients Paper (2017)](https://arxiv.org/abs/1703.01365)
- [Attention is not Explanation (2019)](https://arxiv.org/abs/1902.10186)
- [Attention is not not Explanation (2019)](https://arxiv.org/abs/1908.04626)
- [TCAV Paper (2018)](https://arxiv.org/abs/1711.11279)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- [Captum (PyTorch)](https://captum.ai/)
- [Alibi Explain](https://github.com/SeldonIO/alibi)
- [The Building Blocks of Interpretability (Distill)](https://distill.pub/2018/building-blocks/)
