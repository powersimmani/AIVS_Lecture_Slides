# Lecture 02 분석 보고서
## Data Visualization

**분석 일자**: 2026-02-15
**품질 등급**: A (우수)

---

## 1. 강의 구조 개요

| Part | 주제 | 슬라이드 | 평가 |
|------|------|----------|------|
| Part 1 | 시각화 기초 원리 | 03-11 | 우수 |
| Part 2 | 기본 차트 유형 마스터 | 12-20 | 우수 |
| Part 3 | ML을 위한 고급 시각화 | 21-30 | 매우 우수 |

---

## 2. 긍정적 평가

### 2.1 시각적 품질
- Visual Encoding Hierarchy 슬라이드 매우 직관적
- 색상 팔레트 예시 (Sequential, Diverging, Categorical) 실용적
- PCA/t-SNE/UMAP 비교 SVG 시각화 탁월
- Scree Plot 예시 포함으로 PCA 이해도 향상

### 2.2 접근성 강조
- 색맹 고려 가이드라인 포함 (~8% 남성 통계)
- Red-Green 조합 회피 명시
- ColorBrewer, Viridis 팔레트 추천
- WCAG 대비 기준 언급

### 2.3 ML 워크플로우 연계
- EDA → Feature Engineering → Model Evaluation 흐름
- SHAP, LIME, Attention 시각화 포함
- Learning Curves로 Underfitting/Overfitting 진단
- ROC vs PR Curve 사용 시점 명확

### 2.4 실용적 도구 소개
- Matplotlib, Seaborn, Plotly 언급
- pandas-profiling, sweetviz 자동 EDA 도구
- Streamlit, Dash, Tableau 대시보드 도구

---

## 3. 개선 권장사항

### 3.1 [중요] Visual Encoding 수치 출처 명시

**파일**: `Lecture02/Lecture02_05_Visual Encoding Principles.html`

**현재 상태**: 수치만 제시 (100%, 85%, 70%, 55%, 40%, 28%, 18%)

**문제**: 이 수치의 학술적 출처가 명시되어 있지 않음

**권장 수정**:
슬라이드 하단 또는 subtitle에 출처 추가:

```html
<div class="subtitle">
    Effectiveness Ranking from Most to Least Accurate
    <br><small style="color: #999;">Based on Cleveland & McGill (1984)</small>
</div>
```

**참고 문헌**:
- Cleveland, W. S., & McGill, R. (1984). "Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods." Journal of the American Statistical Association.

---

### 3.2 [중요] Anscombe's Quartet 시각화 추가

**위치**: Part 1 (시각화 기초) - 새 슬라이드 또는 기존 슬라이드 보강

**현재 상태**: Summary에서 언급되지만 실제 시각화 없음

**추가 권장**:
```html
<!-- Anscombe's Quartet 시각화 -->
<div class="quartet-grid">
    <!-- 4개의 산점도: 같은 통계량, 다른 패턴 -->
    <!-- Dataset I: 선형 관계 -->
    <!-- Dataset II: 비선형 관계 -->
    <!-- Dataset III: 이상치 영향 -->
    <!-- Dataset IV: 고레버리지 포인트 -->
</div>
<div class="stats-same">
    Mean X = 9, Mean Y = 7.5,
    Variance X = 11, Variance Y = 4.1,
    Correlation = 0.816, Regression: Y = 3 + 0.5X
</div>
```

**중요성**: "왜 시각화가 필요한가"의 가장 강력한 예시

---

### 3.3 [중요] UMAP 파라미터 설명 추가

**파일**: `Lecture02/Lecture02_25_Dimensionality Reduction Visualization (PCA, t-SNE, UMAP).html`

**현재 상태**:
- t-SNE: "Perplexity: 5-50" 명시
- UMAP: 파라미터 설명 없음

**수정 권장**:
```html
<div class="comparison-row">
    <div class="comp-header">
        <span>🟢</span>
        <span>UMAP</span>
    </div>
    <div class="comp-item">
        <span class="comp-icon">•</span>
        <span>Faster than t-SNE</span>
    </div>
    <div class="comp-item">
        <span class="comp-icon">•</span>
        <span>Local + global</span>
    </div>
    <!-- 추가 -->
    <div class="comp-item">
        <span class="comp-icon">•</span>
        <span>n_neighbors: 5-50</span>
    </div>
    <div class="comp-item">
        <span class="comp-icon">•</span>
        <span>min_dist: 0.0-0.99</span>
    </div>
</div>
```

**UMAP 주요 파라미터**:
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| n_neighbors | 15 | 로컬 구조 보존 정도 (작을수록 로컬) |
| min_dist | 0.1 | 포인트 간 최소 거리 (작을수록 클러스터 밀집) |
| metric | 'euclidean' | 거리 측정 방법 |

---

### 3.4 [권장] 최신 시각화 라이브러리 추가

**파일**: Part 1 또는 Part 3 관련 슬라이드

**현재 언급된 도구**: Matplotlib, Seaborn, Plotly, Tableau, Power BI

**추가 권장**:
```markdown
## 추가 권장 도구

### Altair (선언적 시각화)
- Vega-Lite 기반, 문법 간결
- Jupyter와 통합 우수
- 예: `alt.Chart(df).mark_point().encode(x='x', y='y')`

### Bokeh (인터랙티브)
- 대용량 데이터 처리 우수
- 웹 앱 내장 용이

### HoloViews (고수준 API)
- 탐색적 분석 최적화
- Panel과 결합 시 대시보드 구축

### PyViz 생태계
- HoloViews + Panel + hvPlot + Datashader
- 대용량 데이터 시각화
```

---

### 3.5 [사소] HTML lang 속성 수정

**현재 상태**: 모든 HTML 파일이 `lang="ko"`

**수정 방법**:
```bash
cd Lecture02
sed -i 's/lang="ko"/lang="en"/g' *.html
```

---

### 3.6 [선택] Datasaurus Dozen 추가 고려

**배경**: Anscombe's Quartet의 현대판 (2017)

**설명**:
- 13개의 다른 시각적 패턴 (공룡, 별, 원 등)
- 모두 동일한 요약 통계량
- 시각화의 중요성을 더 극적으로 보여줌

**참고**: https://www.autodesk.com/research/publications/same-stats-different-graphs

---

## 4. 기술적 정확성 검증 결과

| 항목 | 슬라이드 | 검증 |
|------|----------|------|
| Visual Encoding 순서 | 05 | ✅ Cleveland & McGill과 일치 |
| Gestalt 원리 5가지 | 06 | ✅ 정확 (Proximity, Similarity, Continuity, Closure, Figure-Ground) |
| 색맹 남성 비율 ~8% | 07 | ✅ 정확 (실제 7-8%) |
| Data-Ink Ratio 개념 | 09 | ✅ Tufte 원칙 정확 반영 |
| t-SNE Perplexity 범위 5-50 | 25 | ✅ 표준 권장 범위 |
| ROC: TPR vs FPR | 27 | ✅ 정확 |
| PR Curve: Recall vs Precision | 27 | ✅ 정확 |
| Q-Q Plot 해석 | 28 | ✅ 정규성 검정 설명 정확 |

---

## 5. 우선순위별 작업 체크리스트

### 다음 업데이트 시 (중요)
- [ ] Visual Encoding 수치 출처 추가 (Cleveland & McGill 1984)
- [ ] Anscombe's Quartet 실제 시각화 추가
- [ ] UMAP n_neighbors, min_dist 파라미터 설명 추가

### 시간 있을 때 (권장)
- [ ] Altair, Bokeh 등 최신 도구 언급
- [ ] Datasaurus Dozen 예시 추가 고려
- [ ] HTML lang="en" 으로 변경

### 선택적 개선
- [ ] 각 차트 유형에 Python 코드 스니펫 추가
- [ ] Color palette 선택 도구 링크 추가 (coolors.co, paletton.com)

---

## 6. 다른 강의와의 연계성

| 연계 강의 | 관련 내용 | 상태 |
|-----------|-----------|------|
| Lecture 06 (평가) | Confusion Matrix, ROC, PR Curve | ✅ 일관성 유지 |
| Lecture 17 (비지도) | PCA, t-SNE, Clustering 시각화 | ✅ 내용 겹침 적절 |
| Lecture 19-20 (XAI) | SHAP, LIME, Attention 시각화 | ✅ 미리 소개 적절 |

---

## 7. 참고 자료

- [Cleveland & McGill (1984) - Graphical Perception](https://www.jstor.org/stable/2288400)
- [Tufte - The Visual Display of Quantitative Information](https://www.edwardtufte.com/tufte/books_vdqi)
- [ColorBrewer 2.0](https://colorbrewer2.org/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [Anscombe's Quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet)
- [Datasaurus Dozen](https://www.autodesk.com/research/publications/same-stats-different-graphs)
