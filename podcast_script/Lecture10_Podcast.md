# Lecture 10 Podcast: 왜 깊은 네트워크가 필요한가 - 현대 아키텍처의 비밀

## 에피소드 정보
- **주제**: 딥 네트워크의 필요성, 활성화 함수, 스킵 연결과 현대 아키텍처
- **예상 시간**: 15분
- **대상**: 딥러닝 아키텍처 설계 원리를 이해하고 싶은 분들

---

## 스크립트

**[인트로 - 0:00]**

**Host A**: 안녕하세요! AI 비전 시스템 팟캐스트입니다. 오늘은 딥러닝에서 "딥"이 왜 중요한지 알아볼 거예요.

**Host B**: 네! 왜 얕은 네트워크보다 깊은 네트워크가 더 좋은지, ResNet이 왜 혁명적이었는지 다룰 거예요.

**Host A**: 1000층짜리 네트워크를 어떻게 학습시키는지도 궁금하지 않으세요?

**Host B**: 정말 궁금하죠! 그 비밀을 오늘 파헤쳐 봅시다.

---

**[섹션 1: 얕은 네트워크의 한계 - 1:30]**

**Host A**: 먼저 왜 깊이가 필요한지 설명해주세요.

**Host B**: Universal Approximation Theorem이라는 게 있어요. 이론적으로 단일 은닉층으로 어떤 함수든 근사할 수 있대요.

**Host A**: 그럼 한 층이면 되는 거 아니에요?

**Host B**: 문제는 "얼마나 많은 뉴런이 필요한가"예요. 복잡한 함수를 표현하려면 기하급수적으로 많은 뉴런이 필요해요. 비효율적이죠.

**Host A**: 예를 들면요?

**Host B**: 100개의 특징 조합을 표현하려고 해봐요. 얕은 네트워크는 각 조합마다 뉴런이 필요할 수 있어요. 하지만 깊은 네트워크는 계층적으로 조합해서 훨씬 적은 파라미터로 표현할 수 있어요.

---

**[섹션 2: 깊이의 힘 - 계층적 표현 - 3:00]**

**Host A**: 계층적 표현이 뭐예요?

**Host B**: 복잡한 것을 단순한 것들의 조합으로 분해하는 거예요. 시각 시스템을 생각해보세요. 첫 번째 층은 엣지를 감지하고, 두 번째는 텍스처, 세 번째는 부분, 네 번째는 객체 전체를 인식해요.

**Host A**: 사람의 시각 피질이랑 비슷하네요?

**Host B**: 바로 그거예요! V1은 엣지, V2는 텍스처, V4는 복잡한 패턴, IT는 객체 인식을 담당해요. 딥 네트워크가 이런 생물학적 구조를 모방한 거죠.

**Host A**: 그래서 딥러닝이 비전에서 특히 성공한 거군요.

**Host B**: 맞아요! 이미지는 본질적으로 계층적이거든요. 픽셀에서 엣지, 텍스처, 부분, 객체로 자연스럽게 조합돼요.

---

**[섹션 3: 파라미터 효율성 - 4:30]**

**Host A**: 깊은 네트워크가 파라미터도 더 효율적이라고요?

**Host B**: 네! 같은 표현력을 위해 얕은 네트워크는 훨씬 더 많은 파라미터가 필요해요. 반면 깊은 네트워크는 특징을 재사용하거든요.

**Host A**: 재사용이요?

**Host B**: 예를 들어, "수평선 검출기"가 초기 레이어에 한 번 있으면, 상위 레이어의 여러 패턴이 이걸 공유해서 쓸 수 있어요. 100개의 엣지 검출기가 10,000개의 텍스처 조합을 만들 수 있는 거죠.

**Host A**: 실제 비교 데이터가 있어요?

**Host B**: ResNet-18은 약 1,100만 개의 파라미터로 ImageNet에서 비슷한 성능을 내려면 얕은 네트워크는 몇 배 더 필요해요. 더 적은 파라미터로 더 좋은 일반화를 얻을 수 있죠.

---

**[섹션 4: 기울기 소실 문제 - 6:00]**

**Host A**: 그런데 깊은 네트워크 학습이 어려웠잖아요?

**Host B**: 네, Vanishing Gradient 문제 때문이에요. 역전파에서 그래디언트가 레이어를 거칠 때마다 곱해지는데, 각 값이 1보다 작으면 기하급수적으로 사라져요.

**Host A**: Sigmoid가 문제였죠?

**Host B**: Sigmoid의 최대 그래디언트가 0.25예요. 10개 레이어만 지나도 그래디언트가 거의 0이 돼요. 초기 레이어는 거의 학습이 안 되죠.

**Host A**: 그래서 ReLU가 나온 거예요?

**Host B**: 바로 그거예요! ReLU는 양수 영역에서 그래디언트가 항상 1이에요. 사라지지 않죠. 2012년 AlexNet이 ReLU로 딥러닝 혁명을 일으킨 거예요.

---

**[섹션 5: ReLU와 변형들 - 7:30]**

**Host A**: ReLU도 문제가 있다던데요?

**Host B**: Dead ReLU 문제가 있어요. 한 번 음수 영역에 빠지면 그래디언트가 0이라서 영원히 안 깨어날 수 있어요. 네트워크의 40%가 죽은 뉴런이 되기도 해요!

**Host A**: 해결책은요?

**Host B**: Leaky ReLU가 있어요. 음수에서도 작은 기울기, 보통 0.01을 줘요. PReLU는 이 기울기를 학습 가능하게 했고요.

**Host A**: 요즘 Transformer에서 많이 쓰는 GELU는요?

**Host B**: GELU는 Gaussian Error Linear Unit이에요. 부드러운 곡선이고, 단조롭지 않아요. BERT, GPT, Vision Transformer 모두 GELU를 써요. 현재 Transformer의 표준이에요!

**Host A**: 언제 뭘 써야 해요?

**Host B**: CNN에서는 ReLU나 Leaky ReLU가 기본이에요. Transformer에서는 GELU가 표준이고요. 출력층은 태스크에 맞게, 분류면 Softmax, 회귀면 Linear을 쓰세요.

---

**[섹션 6: Skip Connection의 혁명 - 9:00]**

**Host A**: ResNet의 Skip Connection이 왜 혁명적이었어요?

**Host B**: 2015년 ResNet이 나오기 전에는 20층도 학습하기 어려웠어요. Skip Connection으로 152층, 나중에는 1000층 이상도 학습 가능해졌어요!

**Host A**: 원리가 뭐예요?

**Host B**: 출력이 F(x) + x예요. 입력 x를 그대로 더해주는 거죠. 그래디언트 관점에서 보면, 항상 1이라는 성분이 추가돼요. 그래디언트가 사라지지 않고 직접 흐를 수 있는 "고속도로"가 생기는 거예요.

**Host A**: 학습 관점에서는요?

**Host B**: 네트워크가 H(x)를 직접 학습하는 대신, 잔차 F(x) = H(x) - x를 학습해요. 항등 함수에서 얼마나 벗어나야 하는지만 배우면 되니까 훨씬 쉬워요.

---

**[섹션 7: DenseNet과 Bottleneck - 10:30]**

**Host A**: DenseNet도 비슷한 아이디어죠?

**Host B**: DenseNet은 더 극단적이에요! 각 레이어가 이전의 모든 레이어 출력을 입력으로 받아요. 더하기가 아니라 연결, concatenation이에요.

**Host A**: 장점이 뭐예요?

**Host B**: 특징 재사용이 극대화돼요. 그래디언트도 훨씬 잘 흘러요. 같은 정확도를 위해 ResNet보다 더 적은 파라미터가 필요해요!

**Host A**: Bottleneck 구조는요?

**Host B**: 1x1 Conv로 채널을 줄이고, 3x3 Conv로 처리하고, 다시 1x1 Conv로 늘리는 구조예요. 계산량을 크게 줄이면서 표현력은 유지해요. ResNet-50 이상에서 필수죠.

---

**[섹션 8: 1x1 Convolution과 Inception - 12:00]**

**Host A**: 1x1 Convolution이 왜 중요해요?

**Host B**: 공간은 건드리지 않고 채널만 조작해요. 채널 수를 줄이면 계산량이 크게 감소하고, 늘리면 표현력이 증가해요. 채널 간 정보도 섞어주고요.

**Host A**: Inception 모듈은요?

**Host B**: "여러 스케일을 동시에 보자"는 아이디어예요. 1x1, 3x3, 5x5 컨볼루션과 풀링을 병렬로 수행하고 결과를 합쳐요. 1x1으로 먼저 채널을 줄여서 효율적이에요.

**Host A**: GoogLeNet이 이걸 썼죠?

**Host B**: 네! 22층인데 파라미터는 500만 개밖에 안 돼요. AlexNet은 8층에 6,000만 개인데 말이에요. 효율성의 극치죠!

---

**[섹션 9: 효율적 아키텍처와 NAS - 13:30]**

**Host A**: 모바일용 효율적인 아키텍처도 있죠?

**Host B**: Depthwise Separable Convolution이 핵심이에요. 일반 컨볼루션을 채널별 처리와 포인트별 처리로 분리해요. 8-9배 계산량 절감이 가능해요!

**Host A**: MobileNet이 이걸 쓰죠?

**Host B**: 네! EfficientNet도 이걸 기반으로 해요. EfficientNet은 Neural Architecture Search, NAS로 찾은 구조인데, 정확도 대비 효율성이 최고예요.

**Host A**: NAS가 뭐예요?

**Host B**: 신경망 구조를 자동으로 찾는 거예요! 강화학습이나 진화 알고리즘으로 최적의 레이어 조합을 탐색해요. 사람이 설계한 것보다 더 좋은 구조를 찾기도 해요.

---

**[섹션 10: 모델 압축 - 14:00]**

**Host A**: 큰 모델을 작게 만드는 방법도 있죠?

**Host B**: 여러 가지가 있어요! Pruning은 중요하지 않은 가중치를 제거해요. 10배 압축도 가능해요. Quantization은 FP32를 INT8로 바꿔서 4배 작고 2-4배 빠르게 해요.

**Host A**: Knowledge Distillation도 있죠?

**Host B**: 큰 모델(Teacher)의 지식을 작은 모델(Student)에 전달해요. 정답만 배우는 게 아니라 틀린 답의 확률 분포, "dark knowledge"도 배워서 더 잘 일반화해요.

---

**[아웃트로 - 14:30]**

**Host A**: 오늘 내용을 정리해볼까요?

**Host B**: 첫째, 깊은 네트워크는 계층적 표현 학습으로 얕은 네트워크보다 훨씬 효율적이에요.

**Host A**: 둘째, ReLU가 Sigmoid의 기울기 소실 문제를 해결했고, GELU가 Transformer의 표준이 됐어요.

**Host B**: 셋째, Skip Connection이 초깊은 네트워크 학습을 가능하게 했어요. ResNet, DenseNet 모두 이 원리를 사용해요.

**Host A**: 마지막으로, 1x1 Conv, Depthwise Separable Conv, NAS로 효율적인 아키텍처를 만들 수 있어요!

**Host B**: 다음 시간에는 CNN 아키텍처의 역사와 발전에 대해 더 자세히 다룰 거예요!

**Host A**: 감사합니다!

---

## 핵심 키워드
- Universal Approximation Theorem
- Hierarchical Representation, Feature Reuse
- Parameter Efficiency, Depth vs Width
- Vanishing/Exploding Gradients
- ReLU, Leaky ReLU, PReLU, ELU, SELU
- GELU, Swish, Dead ReLU Problem
- Skip Connection, Residual Learning
- ResNet, DenseNet, Dense Connection
- Bottleneck Architecture, 1x1 Convolution
- Inception Module, Multi-scale Processing
- Depthwise Separable Convolution
- MobileNet, EfficientNet
- Neural Architecture Search (NAS)
- Pruning, Quantization, Knowledge Distillation
