# Lecture 18 Podcast: 자기지도학습과 고급 클러스터링 기법

## 에피소드 정보
- **주제**: Self-Supervised Learning, Contrastive Learning, 시계열/그래프 클러스터링, Deep Clustering
- **예상 시간**: 15분
- **대상**: ML/DL을 공부하는 학생 및 실무자

---

## 스크립트

**[인트로 - 0:00]**

**Host A**: 안녕하세요! AI 비전 시스템 팟캐스트에 오신 것을 환영합니다. 저는 호스트 A이고요.

**Host B**: 안녕하세요, 호스트 B입니다! 오늘은 정말 흥미로운 주제를 다뤄볼 거예요. 바로 자기지도학습과 고급 클러스터링 기법인데요, 라벨 없이도 강력한 표현을 학습하는 방법에 대해 얘기해볼 겁니다.

**Host A**: 맞아요. 요즘 대용량 데이터는 넘쳐나는데 라벨링은 비용이 너무 많이 들잖아요. 자기지도학습이 그 해결책으로 떠오르고 있죠.

**Host B**: 그렇죠! 그리고 시계열 데이터나 그래프 데이터처럼 특수한 형태의 데이터를 클러스터링하는 방법도 함께 다룰 거예요.

---

**[섹션 1: 자기지도학습이란? - 1:30]**

**Host A**: 자, 그럼 Self-Supervised Learning부터 시작해볼까요? 이게 정확히 뭔가요?

**Host B**: 좋은 질문이에요! 자기지도학습은 라벨 없는 데이터에서 스스로 감독 신호를 만들어 학습하는 방법이에요. Pretext Task라고 부르는 인위적인 과제를 통해서요.

**Host A**: Pretext Task가 뭐예요?

**Host B**: 예를 들어볼게요. 이미지 회전 예측, 직소 퍼즐 맞추기, 흑백 이미지 컬러화 같은 거예요. 이런 과제를 풀면서 모델이 자연스럽게 유용한 특징을 배우게 되죠.

**Host A**: 아, 그래서 나중에 다운스트림 태스크에 전이학습으로 활용하는 거군요!

**Host B**: 정확해요! 그리고 최근에는 Contrastive Learning이 가장 성공적인 자기지도학습 방법으로 자리잡았어요.

---

**[섹션 2: Contrastive Learning의 핵심 - 3:30]**

**Host A**: Contrastive Learning은 어떤 원리인가요?

**Host B**: 핵심 아이디어는 간단해요. "비슷한 건 가깝게, 다른 건 멀게"예요. Positive pair는 임베딩 공간에서 가깝게, Negative pair는 멀게 밀어내는 거죠.

**Host A**: Positive pair는 어떻게 만들어요?

**Host B**: 보통 같은 이미지에 서로 다른 augmentation을 적용해요. Random crop, color jitter, Gaussian blur 같은 거요. 두 개의 augmented view가 positive pair가 되는 거죠.

**Host A**: Loss 함수는 뭘 쓰나요?

**Host B**: InfoNCE Loss를 많이 써요. 수식으로 보면 L = -log(exp(sim(z_i, z_j)/temperature) / 전체 합)인데, 본질적으로 positive pair의 유사도를 최대화하고 negative와의 유사도를 최소화해요.

**Host A**: temperature 파라미터는 뭔가요?

**Host B**: 분포의 sharpness를 조절해요. 작으면 더 sharp하게, 크면 더 smooth하게 되죠. 보통 0.07에서 0.5 사이 값을 써요.

---

**[섹션 3: SimCLR와 MoCo - 5:30]**

**Host A**: SimCLR에 대해 좀 더 자세히 설명해주세요.

**Host B**: SimCLR은 Google에서 발표한 Simple Framework for Contrastive Learning이에요. 과정을 설명하면, 먼저 이미지 x를 두 번 augment해서 x_i, x_j를 만들고, encoder로 h_i, h_j를 얻고, projection head로 z_i, z_j를 만든 뒤 contrastive loss를 계산해요.

**Host A**: 핵심 발견이 뭐였나요?

**Host B**: 세 가지예요. 첫째, augmentation의 조합이 정말 중요해요. 특히 color distortion이 핵심이죠. 둘째, 배치 사이즈가 클수록 성능이 좋아요. 4096까지도 써요! 셋째, projection head가 representation 품질을 크게 높여요.

**Host A**: MoCo는 어떻게 다른가요?

**Host B**: MoCo, Momentum Contrast는 메모리 효율성 문제를 해결했어요. 큰 배치 대신 negative sample queue를 유지하고, momentum encoder로 consistency를 보장해요. theta_k = m * theta_k + (1-m) * theta_q 이런 식으로 업데이트하죠.

---

**[섹션 4: BYOL과 최신 방법들 - 7:30]**

**Host A**: BYOL은 negative sample이 필요없다던데요?

**Host B**: 맞아요! Bootstrap Your Own Latent의 약자인데, 놀랍게도 negative sample 없이 작동해요. Online network와 target network 두 개를 사용하고, target은 online의 momentum 버전이에요.

**Host A**: 그런데 negative 없이 어떻게 collapse를 피해요? 모든 게 같은 점으로 수렴하면 안 되잖아요.

**Host B**: 비대칭 구조가 핵심이에요. Online network에만 predictor가 있어서 collapse를 방지해요. 정확한 이유는 아직 활발히 연구 중이지만, 매우 효과적이에요.

**Host A**: 다른 최신 방법들도 있나요?

**Host B**: DINO는 self-distillation 방식이고, MAE는 Masked Autoencoder로 이미지 패치를 가리고 복원하는 방식이에요. SwAV는 clustering과 contrastive를 결합했고, Barlow Twins는 redundancy reduction을 활용해요. 추세는 explicit negative에서 벗어나는 방향이에요.

---

**[섹션 5: 시계열 데이터 클러스터링 - 9:30]**

**Host A**: 이제 특수한 데이터 유형의 클러스터링으로 넘어가볼까요? 시계열부터요.

**Host B**: 시계열 데이터는 특별한 도전이 있어요. 시간 순서가 중요하고, 길이가 다를 수 있고, 비슷한 패턴이 다른 속도로 나타날 수 있어요. 유클리드 거리로는 부족하죠.

**Host A**: DTW가 뭔가요?

**Host B**: Dynamic Time Warping이에요. 두 시계열 사이의 최적 정렬을 찾아요. 하나가 다른 것보다 빠르거나 느려도 유사성을 잡아낼 수 있죠. 동적 프로그래밍으로 O(n * m)에 계산해요.

**Host A**: 클러스터링에 어떻게 쓰나요?

**Host B**: tslearn 라이브러리의 TimeSeriesKMeans를 쓰면 DTW를 거리 메트릭으로 사용할 수 있어요. 단점은 계산 비용이 높다는 거예요. 대안으로 K-Shape 알고리즘이 있는데, cross-correlation 기반이라 더 빨라요.

---

**[섹션 6: 그래프 데이터 클러스터링 - 11:00]**

**Host A**: 그래프 데이터는 어떻게 클러스터링하나요?

**Host B**: Spectral Clustering이 기본이에요. Graph Laplacian의 고유벡터를 사용해요. 과정은 이래요. Affinity matrix A 구성, Laplacian L = D - A 계산, 작은 k개 고유벡터 찾기, 그 공간에서 K-means 적용.

**Host A**: 소셜 네트워크 같은 데서 커뮤니티 찾는 건요?

**Host B**: Louvain 알고리즘이 유명해요! Modularity를 최대화하는 방식인데, 클러스터 수를 미리 정할 필요가 없고 매우 빨라요. 대규모 네트워크에도 적용 가능하죠.

**Host A**: GNN 기반 방법도 있죠?

**Host B**: 네! Graph Neural Network로 node embedding을 학습하고, 그 embedding을 클러스터링해요. Deep Graph Infomax 같은 self-supervised 방법으로 GNN을 학습시킨 뒤 K-means를 적용하는 식이죠.

---

**[섹션 7: Deep Clustering - 12:30]**

**Host A**: Deep Clustering은 뭐예요?

**Host B**: 표현 학습과 클러스터링을 동시에 하는 거예요! DeepCluster가 대표적인데, CNN으로 특징 추출, K-means로 클러스터링, cluster assignment를 pseudo-label로 사용, CNN을 그 label로 학습, 이걸 반복해요.

**Host A**: SwAV도 Deep Clustering의 일종이죠?

**Host B**: 맞아요! Online clustering을 학습 중에 해요. 두 augmented view의 특징을 계산하고, prototype, 즉 cluster center에 할당한 뒤, 한 view의 assignment를 다른 view에서 예측해요. Explicit negative 없이 clustering으로 contrast 효과를 내는 거죠.

---

**[섹션 8: 실무 적용과 고려사항 - 13:30]**

**Host A**: 실제 산업에서는 어떻게 쓰이나요?

**Host B**: 금융에서는 고객 세그먼테이션, 사기 탐지용 거래 클러스터링, 시장 국면 탐지에 써요. 헬스케어에서는 환자 층화, 질병 하위유형 분류, 의료 영상 그룹핑에 활용되고요. 리테일은 상품 분류, 쇼핑 행동 패턴 분석에 쓰죠.

**Host A**: 주의할 점은요?

**Host B**: 몇 가지 흔한 실수가 있어요. K를 임의로 정하지 말고 elbow나 silhouette 방법을 쓰세요. 스케일링을 항상 하세요. 구형 클러스터를 가정하지 말고 DBSCAN이나 GMM도 고려하세요. 그리고 항상 여러 메트릭으로 검증하세요!

---

**[아웃트로 - 14:30]**

**Host A**: 오늘 정말 많은 내용을 다뤘네요! 정리하자면요?

**Host B**: 첫째, 자기지도학습은 라벨 없이 pretext task로 유용한 표현을 학습해요. Contrastive learning이 가장 성공적이죠.

**Host A**: 둘째, SimCLR, MoCo, BYOL 같은 방법들이 supervised learning과의 격차를 좁히고 있어요.

**Host B**: 셋째, 시계열은 DTW, 그래프는 Spectral Clustering이나 GNN 기반 방법이 효과적이에요.

**Host A**: 마지막으로, Deep Clustering은 표현 학습과 클러스터링을 동시에 최적화해서 더 좋은 결과를 얻을 수 있어요!

**Host B**: 다음 에피소드에서는 설명가능한 AI, XAI에 대해 다룰 예정이에요. 구독과 좋아요 부탁드려요!

**Host A**: 감사합니다! 다음 시간에 만나요!

---

## 핵심 키워드
- Self-Supervised Learning, Pretext Task
- Contrastive Learning, InfoNCE Loss
- SimCLR, MoCo, BYOL, DINO, MAE
- DTW (Dynamic Time Warping), K-Shape
- Spectral Clustering, Louvain Algorithm
- GNN (Graph Neural Network), Message Passing
- DeepCluster, SwAV, Deep Clustering
- Mini-batch K-Means, FAISS
