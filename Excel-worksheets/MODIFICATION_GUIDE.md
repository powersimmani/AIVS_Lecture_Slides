# Excel Worksheets 수정 가이드

이 문서는 AIVS 강의 Excel 워크시트들의 피드백을 바탕으로 한 수정 지침입니다.

## 전반적인 수정 사항

### 1. Backpropagation 개선 (모든 파일)
- **문제**: 모든 backpropagation 부분에서 gradient 구하는 방법이 충분히 세세하지 않음
- **수정 방향**:
  - 각 gradient 계산 단계를 더 상세하게 표시
  - 수식과 중간 계산 과정을 명시적으로 추가
  - Chain rule 적용 과정을 단계별로 보여주기

### 2. 수식의 '=' 기호 문제 (Lecture 11, 12, 13)
- **문제**: 식 안에 '=' 기호가 있어서 Excel이 equation으로 인식하여 #ERROR 발생
- **수정 방향**:
  - 수식 표시 셀에서 '=' 기호를 제거하거나
  - 텍스트 형식으로 변경 (셀 앞에 작은따옴표 ' 추가)
  - 수식을 이미지나 텍스트 박스로 표시하는 것 고려

---

## 파일별 상세 수정 사항

### Lecture03_Linear_Algebra.xlsx

#### Vector Operations
- **문제**: 답은 맞는데 검증 로직이 틀렸다고 표시
- **수정**:
  - 조건부 서식 또는 검증 수식 확인
  - 부동소수점 비교 시 ROUND 함수 사용 고려
  - 예: `=IF(ABS(학생답-정답)<0.001, "✓", "✗")`

#### Matrix Operations
- **문제**: 2번 문제만 정상 작동, 나머지는 답이 맞아도 틀렸다고 표시
- **수정**:
  - 2번 문제의 검증 로직을 참고하여 다른 문제들도 동일하게 수정
  - 셀 참조와 비교 수식 재검토

#### Eigenvalue & Eigenvector
- **문제**:
  - 답이 맞아도 틀렸다고 표시됨
  - Step 4에서 error 표시가 2개 존재
- **수정**:
  - 검증 수식 전체 재검토
  - Step 4의 중복 error 제거
  - 셀 범위와 조건부 서식 확인

### Lecture03_Linear_Regression.xlsx

#### Forward Propagation
- **문제**:
  - 엑셀 셀 사이즈가 부적절
  - 값이 밀려있음 (0.9가 두 번째 칸부터 시작해야 함)
  - 노란색 입력 칸들의 검증이 작동하지 않음
- **수정**:
  1. 셀 너비/높이 조정으로 가독성 개선
  2. 데이터 위치 조정:
     - 0.9 값을 올바른 셀 위치로 이동
     - 관련된 모든 수식 참조 업데이트
  3. 노란색 입력 칸 검증 로직 수정:
     - 각 셀의 조건부 서식 확인
     - 정답 비교 수식 재작성
     - 허용 오차 범위 설정

### Lecture04_Logistic_Regression.xlsx

- **문제**: 계산이 계속 틀리게 나옴
- **수정**:
  1. BCE (Binary Cross Entropy) 계산 수식 재확인
  2. Student 열과 Verification 열의 계산 로직 검토
  3. 각 샘플(s1-s10)의 예측값과 손실 계산 수식 확인
  4. 부동소수점 정밀도 문제 확인
  5. 정답 시트와 비교하여 올바른 계산식 적용

### Lecture05_MLP.xlsx

#### Hidden State 누락
- **문제**: Hidden state 1개가 없어져있음
- **수정**:
  - Hidden layer 구조 확인하여 누락된 hidden state 추가
  - 관련 weight 및 bias 파라미터 추가
  - Forward/backward propagation 수식 업데이트

#### 수식 Error
- **문제**: 식 표시 전체를 에러로 인식 (#ERROR!)
- **수정**:
  - Hidden Layer Gradient 섹션의 dL/dh[0], dL/dh[1], dL/dh[2], dL/dh[3] 수식 수정
  - '=' 기호를 텍스트로 표시하도록 변경
  - 예: `'= W2^T × dL/dz2:` 형태로 작성

### Lecture07_CNN.xlsx

#### Standardization 설명 추가
- **수정**:
  - Standardization이 필요한 이유를 설명하는 셀/텍스트박스 추가
  - 내용 예시:
    - "학습 안정성 향상"
    - "Gradient vanishing/exploding 방지"
    - "수렴 속도 개선"

#### Convolution 계산 오류
- **문제**: 답이 조금 다름
- **수정**:
  - Convolution 연산 수식 재계산
  - Kernel/filter 적용 위치 확인
  - Stride, padding 설정 확인
  - 정답 값 업데이트

### Lecture11_LSTM_Gates.xlsx (Lecture 11-A)

#### Cell Size 조정
- **문제**: Cell size가 너무 작아서 글자가 안보임
- **수정**:
  - 행 높이와 열 너비를 적절히 증가
  - 특히 수식이 들어간 셀들의 크기 확대
  - 가독성 확보

#### 계산식 vs 계산값 명확화
- **문제**: 전체적인 계산식을 넣을지 계산 값을 넣을지 불명확
- **수정**:
  - 명확한 지침 제공:
    - 노란색 셀: 학생이 계산값 입력
    - 회색/파란색 셀: 수식 표시 (텍스트)
  - 혼란스러운 부분에 레이블 추가

#### BPTT Learning Rate
- **문제**: Learning rate이 생략됨
- **수정**:
  - Learning rate 값 명시 (0.1인지 확인 후)
  - Parameter 섹션에 η(eta) 또는 lr 추가
  - 모든 gradient update 수식에서 learning rate 포함

### Lecture11_RNN_TeacherForcing.xlsx (Lecture 11-B)

#### Teacher Forcing 설명 추가
- **수정**:
  - Teacher forcing 개념 설명 추가:
    - "학습 시 이전 time step의 예측값 대신 실제 정답값을 입력으로 사용"
    - "수렴 속도 향상 및 학습 안정화"
  - 수식과 함께 시각적 다이어그램 추가 고려

#### Answer Key 추가
- **문제**: Answer 키 없음
- **수정**:
  - 별도 시트에 정답 추가 또는
  - 숨겨진 열/시트에 정답 배치
  - 검증 수식이 참조할 수 있도록 구성

### Lecture12_Seq2Seq_Translation.xlsx (Lecture 12-A)

#### Decoding Parameter 추가
- **수정**:
  - Encoding parameter처럼 decoding parameter도 같은 페이지에 표시
  - Decoder weights, biases 명시적으로 배치
  - 가독성 향상

#### Softmax 과정 상세화
- **문제**: Decoder teaching force에서 softmax로 넘어가는 과정이 불명확
- **수정**:
  - Logit 계산 단계 추가
  - Softmax 수식 명시: `exp(z_i) / Σexp(z_j)`
  - 각 단계별 중간값 표시

#### Backpropagation 수식 추가
- **수정**:
  - Decoder backpropagation 수식 상세히 작성
  - dL/dW, dL/db 계산 과정 단계별 표시
  - Chain rule 적용 명시

### Lecture12_Seq2Seq_Attention.xlsx (Lecture 12-B)

#### Encoder 계산식 추가
- **수정**:
  - Encoder forward pass 계산식을 한 번 더 표시
  - 참고하기 쉽도록 같은 시트 내에 배치

#### Backpropagation Cell Size 및 Error 표시
- **수정**:
  - Cell size 조절로 내용 가독성 확보
  - Error 부분 (#ERROR!)을 수식(텍스트)으로 변경
  - '=' 제거하고 텍스트 형식으로 수식 표시

### Lecture13_Transformer_Attention.xlsx (Lecture 13-A)

#### Input Embedding Error 수정
- **문제**: '= Input:' 부분에서 error 발생
- **수정**:
  - '=' 기호 제거: `Input:` 또는 `'= Input:`으로 변경
  - 셀 형식을 텍스트로 설정
  - 조건부 서식에서 error 표시 제거

#### Self Attention Error 수정
- **문제**: Self attention에서 error 발생
- **수정**:
  - 수식 표시 셀에서 '=' 제거
  - 모든 수식을 텍스트 형식으로 변경
  - 예: `'= Q × K^T / sqrt(d_k)` 형태로 작성

### Lecture13_Transformer_Full.xlsx (Lecture 13-B)

#### B열 확장
- **문제**: 전체적으로 모든 파트에서 B열이 좁음
- **수정**:
  - B열 너비를 최소 15-20 이상으로 확대
  - 내용이 잘리지 않도록 조정

#### Cross Attention 상세화
- **수정**:
  - Cross attention weight 계산 과정을 단계별로 추가:
    1. Q(from decoder) × K^T(from encoder)
    2. Scaling: / sqrt(d_k)
    3. Softmax 적용
    4. Weighted sum with V
  - 각 단계의 수식과 차원(dimension) 명시

#### Backward Propagation 상세화
- **수정**:
  - Softmax 미분: `∂softmax/∂z = softmax(z) × (I - softmax(z)^T)`
  - Transpose 곱 연산 명시
  - Matrix chain rule 적용 과정 표시
  - 각 layer별 gradient 계산 단계 추가

### Lecture17_PCA.xlsx

#### Eigencomposition 설명 추가
- **수정**:
  - Eigencomposition 개념 설명 추가:
    - "공분산 행렬을 eigenvalue와 eigenvector로 분해"
    - "주성분(PC)은 eigenvalue가 큰 순서의 eigenvector"
    - "설명되는 분산 비율 = eigenvalue / Σeigenvalues"
  - 수식과 함께 간단한 예시 추가

---

## 우선순위

### 긴급 (Critical)
1. **Lecture 03-A, 03-B, 04**: 검증 로직 수정 (답이 맞는데 틀렸다고 표시)
2. **Lecture 05**: #ERROR 수정 및 누락된 hidden state 추가
3. **Lecture 11, 12, 13**: '=' 기호로 인한 #ERROR 전체 수정

### 높음 (High)
4. **Lecture 03-B**: 값 정렬 문제 해결 (0.9 위치)
5. **Lecture 11-A**: Cell size 조정
6. **Lecture 13-B**: B열 너비 확장
7. **Lecture 11-B**: Answer key 추가

### 중간 (Medium)
8. **모든 파일**: Backpropagation 설명 상세화
9. **Lecture 07**: Convolution 계산 검증
10. **Lecture 12-A, 12-B**: 계산 과정 상세화
11. **Lecture 13-A, 13-B**: Attention 메커니즘 설명 보강

### 낮음 (Low)
12. **Lecture 07**: Standardization 이유 설명 추가
13. **Lecture 17**: Eigencomposition 설명 추가
14. **Lecture 11-A**: BPTT learning rate 명시
15. **Lecture 11-B**: Teacher forcing 설명 추가

---

## 검증 체크리스트

수정 완료 후 각 파일에 대해 다음을 확인:

- [ ] 모든 #ERROR 표시 제거됨
- [ ] 입력 칸의 검증 로직이 정상 작동
- [ ] 셀 크기가 적절하여 모든 내용이 보임
- [ ] 수식 표시가 명확하고 '=' 기호 문제 없음
- [ ] Backpropagation 설명이 충분히 상세함
- [ ] 누락된 내용(hidden state, answer key 등)이 추가됨
- [ ] 색상 표시와 체크마크가 정확히 작동함

---

## 기술적 팁

### 검증 수식 개선
```excel
// 부동소수점 비교
=IF(ABS(학생입력셀-정답셀)<0.001, "✓", "✗")

// 조건부 서식에서
=$B$2=C2  // 대신
=ABS($B$2-C2)<0.001  // 사용
```

### 수식을 텍스트로 표시
```excel
// 셀 시작 부분에 작은따옴표 추가
'= y = wx + b

// 또는 CONCATENATE 사용
="= y = " & A1 & "x + " & B1
```

### Cell 보호하면서 입력 허용
1. 모든 셀 잠금 해제
2. 정답/수식 셀만 다시 잠금
3. 시트 보호 활성화 (학생 입력 셀은 편집 가능)

---

## 참고 사항

- 모든 수정 전에 원본 파일 백업 필수
- 수정 후 학생 관점에서 전체 워크시트 테스트
- 각 강의의 학습 목표에 부합하는지 확인
- 너무 복잡하게 만들지 말고 교육적 가치에 집중
