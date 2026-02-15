# Lecture 17 분석 보고서
## Clustering and Unsupervised Learning Fundamentals

**분석 일자**: 2026-02-15
**품질 등급**: A (우수)

---

## 1. 강의 구조 개요

| Part | 주제 | 슬라이드 | 평가 |
|------|------|----------|------|
| Part 1 | 비지도 학습 개요 | 도입 | 우수 |
| Part 2 | K-Means 클러스터링 | 상세 | 매우 우수 |
| Part 3 | DBSCAN | 상세 | 매우 우수 |
| Part 4 | 계층적 클러스터링 | 상세 | 우수 |
| Part 5 | GMM | 상세 | 매우 우수 |
| Part 6 | 평가 지표 | 상세 | 우수 |
| Part 7 | 차원 축소 | PCA, t-SNE | 우수 |

---

## 2. 긍정적 평가

### 2.1 학습 패러다임 비교
- Supervised vs Unsupervised vs Semi-supervised 명확한 구분
- 비지도 학습의 응용 분야 설명

### 2.2 K-Means 상세 분석
- 알고리즘 단계별 설명
- Elbow Method 시각화
- K-Means++ 초기화 중요성
- 시간 복잡도: O(nkT)
- 한계점 명시 (구형 클러스터 가정)

### 2.3 DBSCAN 상세 분석
- 밀도 기반 클러스터링 원리
- eps, min_samples 파라미터 설명
- Core point, Border point, Noise point 구분
- 임의 형태 클러스터 처리 가능

### 2.4 GMM (Gaussian Mixture Model)
- 확률적 클러스터링 개념
- EM 알고리즘 (E-step, M-step)
- Soft assignment vs Hard assignment
- Covariance type (full, diag, spherical)

### 2.5 평가 지표 종합
- **Silhouette Score**: 내부 응집도 + 외부 분리도
- **Davies-Bouldin Index**: 클러스터 간 유사도
- **Calinski-Harabasz Index**: 분산 비율
- **외부 지표**: ARI, NMI (레이블 있을 때)

### 2.6 차원 축소 기법
- **PCA**: 선형 차원 축소, 분산 최대화
- **t-SNE**: 비선형, 시각화용
- 용도별 구분 명확

---

## 3. 개선 권장사항

### 3.1 [중요] UMAP 추가

**위치**: Part 7 차원 축소

**현재 상태**: t-SNE만 상세히 다룸

**중요성**:
- t-SNE의 현대적 대안
- 속도 및 전역 구조 보존 우수
- 실무에서 널리 사용

**추가 권장 내용**:
```markdown
## UMAP (Uniform Manifold Approximation and Projection)

### t-SNE vs UMAP
| 항목 | t-SNE | UMAP |
|------|-------|------|
| 속도 | 느림 | 빠름 |
| 전역 구조 | 약함 | 강함 |
| 대용량 데이터 | 어려움 | 가능 |
| 하이퍼파라미터 | perplexity | n_neighbors, min_dist |

### 핵심 파라미터
- **n_neighbors**: 지역 구조 (15-50)
- **min_dist**: 점 간 최소 거리 (0.0-0.99)
- **metric**: 거리 측정 방법

### Python
```python
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
embedding = reducer.fit_transform(X)
```

### 언제 사용?
- 대용량 데이터: UMAP
- 논문 시각화: t-SNE (더 익숙)
- 클러스터링 전처리: UMAP (전역 구조)
```

---

### 3.2 [중요] HDBSCAN 추가

**위치**: Part 3 DBSCAN 이후

**중요성**:
- DBSCAN의 주요 개선
- eps 자동 결정
- 다양한 밀도 클러스터 처리

**추가 권장**:
```markdown
## HDBSCAN (Hierarchical DBSCAN)

### DBSCAN 한계
- eps 선택 어려움
- 다양한 밀도 클러스터 처리 실패

### HDBSCAN 개선
- 다양한 eps에서 DBSCAN 실행 (개념적)
- 계층적 구조로 안정적 클러스터 추출
- 자동 outlier 탐지

### 파라미터
- **min_cluster_size**: 최소 클러스터 크기
- **min_samples**: core point 기준 (선택)

### vs DBSCAN
| 항목 | DBSCAN | HDBSCAN |
|------|--------|---------|
| eps 필요 | O | X |
| 다양한 밀도 | X | O |
| 계층 구조 | X | O |
| 속도 | 빠름 | 중간 |

### Python
```python
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
labels = clusterer.fit_predict(X)
```
```

---

### 3.3 [중요] 클러스터 수 결정 방법 종합

**위치**: Part 6 또는 별도 슬라이드

**추가 권장**:
```markdown
## 최적 클러스터 수 결정

### 1. Elbow Method
- 관성(inertia) vs k 그래프
- "팔꿈치" 지점 선택
- 주관적 해석 필요

### 2. Silhouette Analysis
- 각 k에서 silhouette score 계산
- 최대값 선택
- 클러스터별 분포 시각화

### 3. Gap Statistic
- 실제 데이터 vs 랜덤 데이터 비교
- 통계적으로 더 robust
- 계산 비용 높음

### 4. BIC/AIC (GMM)
- 모델 복잡도 페널티
- 낮은 값 선택
- GMM에 적합

### 권장 프로세스
1. Elbow로 대략적 범위 파악
2. Silhouette으로 확인
3. 도메인 지식 반영
```

---

### 3.4 [권장] Spectral Clustering 추가

**위치**: Part 4 이후

**추가 권장**:
```markdown
## Spectral Clustering

### 아이디어
- 그래프 라플라시안 고유벡터 사용
- 저차원 공간에서 K-Means
- 비볼록 클러스터 처리 가능

### 알고리즘
1. Similarity matrix A 구성
2. Laplacian L = D - A
3. 작은 고유값의 고유벡터 계산
4. 고유벡터 공간에서 K-Means

### 장점
- 복잡한 형태 클러스터
- 이론적 기반 강함

### 단점
- 대용량 데이터에 비효율
- k 사전 지정 필요

### sklearn
```python
from sklearn.cluster import SpectralClustering
sc = SpectralClustering(n_clusters=3, affinity='rbf')
labels = sc.fit_predict(X)
```
```

---

### 3.5 [권장] 시계열 클러스터링 언급

**위치**: 별도 슬라이드

**추가 권장**:
```markdown
## 시계열 클러스터링 (Preview)

### 문제
- 일반 거리 측정 부적합
- 시간 축 정렬 필요

### DTW (Dynamic Time Warping)
- 시간 축 신축 허용
- 최적 정렬 탐색
- O(n²) 복잡도

### K-Shape
- DTW보다 빠름
- Shape-based distance
- Cross-correlation 기반

### 참고: Lecture 18에서 상세
```

---

## 4. 기술적 정확성 검증 결과

| 항목 | 슬라이드 | 검증 |
|------|----------|------|
| K-Means 목적함수 | - | ✅ 정확 |
| DBSCAN core point 정의 | - | ✅ 정확 |
| Silhouette 수식 | - | ✅ 정확 |
| PCA 분산 최대화 | - | ✅ 정확 |
| GMM EM 알고리즘 | - | ✅ 정확 |
| t-SNE perplexity | - | ✅ 정확 |

---

## 5. 우선순위별 작업 체크리스트

### 다음 업데이트 시 (중요)
- [ ] UMAP 추가
- [ ] HDBSCAN 추가
- [ ] 클러스터 수 결정 방법 종합

### 시간 있을 때 (권장)
- [ ] Spectral Clustering 추가
- [ ] 시계열 클러스터링 미리보기
- [ ] 클러스터링 파이프라인 예시

### 선택적 개선
- [ ] Mini-batch K-Means 언급
- [ ] Affinity Propagation 언급
- [ ] 클러스터링 시각화 팁

---

## 6. 다른 강의와의 연계성

| 연계 강의 | 관련 내용 | 상태 |
|-----------|-----------|------|
| Lecture 18 (SSL) | 자기지도 학습 연결 | ✅ 좋은 연결 |
| Lecture 07 (특징 추출) | Feature Engineering | ✅ 연계됨 |
| Lecture 11 (시계열) | DTW, 시계열 분석 | ⚠️ DTW 연결 권장 |
| Lecture 15-16 (생성) | VAE latent | ⚠️ latent 클러스터링 연결 |

---

## 7. 특별 참고사항

### 클러스터링 알고리즘 선택 가이드
| 상황 | 권장 알고리즘 |
|------|--------------|
| 구형 클러스터, k 알려짐 | K-Means |
| 임의 형태, 노이즈 존재 | DBSCAN/HDBSCAN |
| 확률적 할당 필요 | GMM |
| 계층 구조 필요 | Agglomerative |
| 비볼록, 중간 규모 | Spectral |

### sklearn 클러스터링 비교
```python
# 알고리즘별 특성
algorithms = {
    'KMeans': {'scalability': 'large', 'geometry': 'flat'},
    'DBSCAN': {'scalability': 'medium', 'geometry': 'arbitrary'},
    'GMM': {'scalability': 'medium', 'geometry': 'flat'},
    'Spectral': {'scalability': 'small', 'geometry': 'arbitrary'}
}
```

### 차원 축소 목적별 선택
```
목적: 노이즈 제거/특징 추출 → PCA
목적: 시각화 (지역 구조) → t-SNE
목적: 시각화 (전역 구조) → UMAP
목적: 클러스터링 전처리 → UMAP 또는 PCA
```

---

## 8. 참고 자료

- [K-Means++ Paper](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)
- [DBSCAN Paper - Ester et al. (1996)](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)
- [HDBSCAN Paper (2017)](https://arxiv.org/abs/1705.07321)
- [UMAP Paper (2018)](https://arxiv.org/abs/1802.03426)
- [t-SNE Paper - van der Maaten (2008)](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
- [sklearn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html)
- [Silhouette Analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
