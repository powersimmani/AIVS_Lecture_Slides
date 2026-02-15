# Lecture 14 분석 보고서
## Pre-trained Language Models & LLM Era

**분석 일자**: 2026-02-15
**품질 등급**: A+ (매우 우수)

---

## 1. 강의 구조 개요

| Part | 주제 | 슬라이드 | 평가 |
|------|------|----------|------|
| Part 1 | 도입 및 패러다임 변화 | 03-05 | 우수 |
| Part 2 | Pre-training 개념 | 06-09 | 매우 우수 |
| Part 3 | BERT (Encoder 모델) | 10-14 | 매우 우수 |
| Part 4 | GPT (Decoder 모델) | 15-19 | 매우 우수 |
| Part 5 | Encoder-Decoder 모델 | 20-22 | 우수 |
| Part 6 | Fine-tuning 전략 | 23-26 | 매우 우수 |
| Part 7 | Prompting & ICL | 27-30 | 매우 우수 |
| Part 8 | 현재 트렌드 | 31-33 | 매우 우수 |
| Part 9 | 윤리 및 실습 | 34-35 | 우수 |

---

## 2. 긍정적 평가

### 2.1 Pre-training 패러다임 설명 (Part 1-2)
- "Train once, use everywhere" 개념 명확
- Scale의 중요성 강조
- Language Modeling Objective:
  - MLM (Masked Language Modeling)
  - CLM (Causal Language Modeling)

### 2.2 BERT 상세 분석 (Part 3)
- Bidirectional 특성 설명
- Pre-training tasks:
  - MLM (15% masking)
  - NSP (Next Sentence Prediction)
- Fine-tuning 방법:
  - Classification: [CLS] token
  - Token-level: 각 token output
  - QA: Start/End prediction
- BERT Family: RoBERTa, ALBERT, DistilBERT

### 2.3 GPT 시리즈 상세 (Part 4)
- Autoregressive 특성
- GPT → GPT-2 → GPT-3 진화
- Few-shot Learning 설명
- Scaling Laws 언급

### 2.4 Parameter-Efficient Fine-tuning (Part 6)
- **LoRA** 상세 설명:
  - Low-rank decomposition: W' = W + AB
  - 시각화 및 계산 예시
  - 99% 파라미터 감소 가능
- Adapter Layers
- Prefix/Prompt Tuning
- BitFit (bias만 학습)

### 2.5 Prompt Engineering (Part 7)
- Zero-shot, One-shot, Few-shot
- Chain-of-Thought (CoT)
- In-Context Learning
- Fine-tuning vs Prompting 비교

### 2.6 RLHF 및 현재 트렌드 (Part 8)
- Reinforcement Learning from Human Feedback
- Instruction Tuning
- GPT-4, Claude 등 현대 LLM 언급

---

## 3. 개선 권장사항

### 3.1 [중요] QLoRA 추가

**위치**: Part 6 PEFT 섹션

**현재 상태**: LoRA만 상세히 다룸

**중요성**:
- 4-bit quantization + LoRA
- 단일 GPU로 65B 모델 Fine-tuning 가능
- 실무에서 매우 중요

**추가 권장 내용**:
```markdown
## QLoRA (Quantized LoRA)

### 핵심 아이디어
- Base model: 4-bit 양자화
- LoRA adapters: 16-bit
- 메모리 대폭 절감

### 구성 요소
1. **4-bit NormalFloat (NF4)**: 새로운 양자화 데이터 타입
2. **Double Quantization**: 양자화 상수도 양자화
3. **Paged Optimizers**: GPU 메모리 관리

### 메모리 비교 (LLaMA 65B)
| 방법 | GPU 메모리 | GPU 수 |
|------|------------|--------|
| Full Fine-tune | ~780GB | ~10 A100 |
| LoRA | ~156GB | ~2 A100 |
| QLoRA | ~48GB | 1 A100 |

### 코드
```python
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

# 4-bit 양자화
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config
)

# LoRA 적용
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
```
```

---

### 3.2 [중요] RLHF 상세 설명 추가

**위치**: Part 8 (현재 간략)

**추가 권장 내용**:
```markdown
## RLHF 상세 과정

### 3단계 과정
1. **Supervised Fine-tuning (SFT)**
   - 고품질 대화 데이터로 Fine-tune
   - 기본적인 instruction following 학습

2. **Reward Model Training**
   - 인간 선호도 데이터 수집
   - 응답 A vs B 중 선호 선택
   - Reward 점수 예측 모델 학습

3. **PPO Training**
   - Policy model이 응답 생성
   - Reward model이 점수 부여
   - PPO로 policy 업데이트
   - KL divergence로 과적합 방지

### 수식
Objective = E[R(x,y)] - β × KL(π || π_ref)

### 대안적 방법
- **DPO (Direct Preference Optimization)**
  - Reward model 없이 직접 최적화
  - 더 간단하고 안정적
  - LLaMA 2, Zephyr 등에서 사용

### InstructGPT 논문 (2022)
- GPT-3 → ChatGPT 변환
- RLHF의 성공적 적용 사례
```

---

### 3.3 [중요] Tokenization 상세 추가

**위치**: Part 2 또는 Part 3

**현재 상태**: 언급 부족

**추가 권장**:
```markdown
## Tokenization in PLMs

### BPE (Byte Pair Encoding)
- GPT 시리즈 사용
- 빈도 기반 병합

### WordPiece
- BERT 사용
- "##" prefix로 subword 표시
- "playing" → "play" + "##ing"

### SentencePiece
- T5, LLaMA 사용
- 언어 독립적
- Unigram 또는 BPE 지원

### 토크나이저 비교
| 모델 | Tokenizer | 어휘 크기 |
|------|-----------|-----------|
| BERT | WordPiece | 30,522 |
| GPT-2 | BPE | 50,257 |
| GPT-4 | tiktoken | ~100,000 |
| LLaMA | SentencePiece | 32,000 |
| T5 | SentencePiece | 32,000 |

### 효율성 고려
- 더 큰 어휘 = 더 짧은 시퀀스
- 언어별 토큰 효율성 차이 (한국어 ~3x)
```

---

### 3.4 [권장] Mixture of Experts (MoE) 추가

**위치**: Part 8 Current Trends

**중요성**:
- GPT-4, Mixtral 등 최신 모델 기반
- 효율적 스케일링

**추가 권장**:
```markdown
## Mixture of Experts (MoE)

### 아이디어
- 모든 파라미터를 항상 사용하지 않음
- Router가 입력에 따라 expert 선택
- 파라미터 수 ↑, 계산량 일정

### 구조
Input → Router → Top-k Experts → Combine output

### 예시: Mixtral 8x7B
- 8개 expert, 각 7B 파라미터
- 총 파라미터: ~47B
- 활성 파라미터: ~13B (top-2)
- 성능: 70B dense 모델 수준

### 장점
- 스케일링 효율성
- 추론 비용 절감
- 전문화된 expert

### 한계
- 학습 불안정 (load balancing)
- 메모리 사용량 여전히 높음
```

---

### 3.5 [권장] Multimodal LLM 소개

**위치**: Part 8 또는 새 슬라이드

**추가 권장**:
```markdown
## Multimodal LLMs

### 개념
- Text + Image + Audio 통합 이해
- Vision-Language Models

### 주요 모델
| 모델 | 회사 | 모달리티 |
|------|------|----------|
| GPT-4V | OpenAI | Text + Image |
| Claude 3 | Anthropic | Text + Image |
| Gemini | Google | Text + Image + Audio |
| LLaVA | UW | Open-source VLM |

### 아키텍처 패턴
1. **Vision Encoder + LLM**
   - CLIP/ViT로 이미지 인코딩
   - LLM에 visual tokens로 입력

2. **Unified Architecture**
   - 모든 모달리티 동일 토큰으로 처리
   - Gemini 스타일

### 응용
- Image Captioning
- Visual QA
- Document Understanding
- Code from Screenshots
```

---

### 3.6 [권장] Scaling Laws 상세화

**파일**: 슬라이드 09 (Scale의 중요성)

**추가 권장**:
```markdown
## Scaling Laws

### Kaplan et al. (2020) 발견
Loss ∝ 1/(Parameters^0.076)
Loss ∝ 1/(Data^0.095)
Loss ∝ 1/(Compute^0.050)

### Chinchilla Scaling (2022)
- 이전: 큰 모델 + 적은 데이터
- Chinchilla: 모델과 데이터 균형
- 동일 compute → 더 작은 모델, 더 많은 데이터

### 실무적 함의
| Compute | 권장 모델 크기 | 권장 토큰 수 |
|---------|---------------|-------------|
| 10^18 | ~125M | ~2.5B |
| 10^20 | ~1.3B | ~26B |
| 10^22 | ~13B | ~260B |
| 10^24 | ~130B | ~2.6T |
```

---

## 4. 기술적 정확성 검증 결과

| 항목 | 슬라이드 | 검증 |
|------|----------|------|
| BERT MLM 15% masking | 12 | ✅ 정확 |
| BERT [CLS] for classification | 13 | ✅ 정확 |
| GPT Autoregressive | 17 | ✅ 정확 |
| T5 Text-to-Text | 21 | ✅ 정확 |
| LoRA W' = W + AB | 25 | ✅ 정확 |
| LoRA rank r 개념 | 25 | ✅ 정확 |
| Few-shot ICL 개념 | 18, 28 | ✅ 정확 |
| CoT prompting | 29 | ✅ 정확 |

---

## 5. 우선순위별 작업 체크리스트

### 다음 업데이트 시 (중요)
- [ ] QLoRA 추가
- [ ] RLHF 상세 (3단계) 추가
- [ ] Tokenization 상세 추가

### 시간 있을 때 (권장)
- [ ] MoE (Mixture of Experts) 추가
- [ ] Multimodal LLM 소개
- [ ] Scaling Laws 상세화
- [ ] DPO (Direct Preference Optimization) 언급

### 선택적 개선
- [ ] Constitutional AI 언급
- [ ] LLM 평가 방법 (MMLU, HumanEval 등)
- [ ] Open-source LLM 비교 (LLaMA, Mistral 등)

---

## 6. 다른 강의와의 연계성

| 연계 강의 | 관련 내용 | 상태 |
|-----------|-----------|------|
| Lecture 13 (Transformer) | 아키텍처 기초 | ✅ 완벽한 연결 |
| Lecture 07 (특징 추출) | Word Embedding 기초 | ✅ 확장됨 |
| Lecture 08 (최적화) | AdamW, Warm-up | ✅ 연계됨 |
| Lecture 09 (정규화) | LayerNorm, Dropout | ✅ 연계됨 |
| Lecture 19-20 (XAI) | LLM 해석 | ⚠️ LLM Probing 연계 권장 |

---

## 7. 특별 참고사항

### 강의의 핵심 위치
이 강의는 **현대 NLP의 핵심**을 다룸:
1. Pre-training → Fine-tuning 패러다임
2. BERT vs GPT (이해 vs 생성)
3. Efficient Fine-tuning (LoRA, PEFT)
4. LLM 시대의 Prompt Engineering

### BERT vs GPT 핵심 비교
| 항목 | BERT | GPT |
|------|------|-----|
| 방향 | Bidirectional | Unidirectional |
| Pre-training | MLM + NSP | CLM |
| 강점 | 이해, 분류 | 생성 |
| Fine-tuning | 필수 | 선택적 (Prompting) |
| 대표 용도 | NER, QA, 분류 | 대화, 요약, 번역 |

### Fine-tuning 선택 가이드
```
데이터 양이 많고 compute 충분?
    → Full Fine-tuning

데이터 양이 적거나 compute 제한?
    → LoRA / QLoRA

아예 학습 불가?
    → Prompting / ICL
```

### LoRA 하이퍼파라미터 가이드
```python
# 권장 설정
lora_config = LoraConfig(
    r=8,                    # rank (8-64 일반적)
    lora_alpha=16,          # scaling factor
    target_modules=[        # 적용 대상
        "q_proj", "v_proj", # 필수
        "k_proj", "o_proj", # 선택
        "gate_proj", "up_proj", "down_proj"  # FFN
    ],
    lora_dropout=0.05,
    bias="none"
)
```

---

## 8. 참고 자료

- [BERT Paper (2018)](https://arxiv.org/abs/1810.04805)
- [GPT-2 Paper (2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-3 Paper (2020)](https://arxiv.org/abs/2005.14165)
- [LoRA Paper (2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper (2023)](https://arxiv.org/abs/2305.14314)
- [InstructGPT/RLHF (2022)](https://arxiv.org/abs/2203.02155)
- [DPO Paper (2023)](https://arxiv.org/abs/2305.18290)
- [Scaling Laws (2020)](https://arxiv.org/abs/2001.08361)
- [Chinchilla (2022)](https://arxiv.org/abs/2203.15556)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
