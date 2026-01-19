# Lecture 14 Podcast: Pre-trained Language Models - BERT, GPT, and Beyond

## Episode Information
- **Topic**: Pre-training Paradigm, BERT, GPT, Fine-tuning, Prompting, RLHF
- **Estimated Time**: 15 minutes
- **Target Audience**: Students and practitioners studying ML/DL

---

## Script

**[Intro - 0:00]**

**Host A**: Hello! Welcome to the AI Vision Systems Podcast. I'm Host A.

**Host B**: Hello, I'm Host B! Today we're covering pre-trained language models, the core of modern AI. ChatGPT, Claude, Gemini... these are the concepts underlying all of them!

**Host A**: Right. Since BERT and GPT came out in 2018, the AI field has completely changed. Let's learn how the "pre-train then fine-tune" paradigm revolutionized everything!

---

**[Section 1: Paradigm Shift - 1:00]**

**Host B**: How did we do things before?

**Host A**: Before 2018, we trained models from scratch for each task. Sentiment analysis model, named entity recognition model, translation model... each made separately with different task-specific architectures.

**Host B**: What were the problems?

**Host A**: First, you need lots of labeled data. Second, training from scratch for each task is inefficient. Third, knowledge doesn't transfer between tasks.

**Host B**: What's the new paradigm?

**Host A**: "Pre-train then fine-tune." First, learn general language understanding with massive text data, then adapt to specific tasks with small labeled data.

**Host B**: An analogy?

**Host A**: It's like humans first learning to read, then learning specific jobs. With basic reading ability, learning new tasks becomes much easier!

---

**[Section 2: Core of Pre-training - 2:30]**

**Host B**: What exactly is pre-training?

**Host A**: Learning general representations from large-scale text corpora. The key is self-supervised learning. Labels come from the data itself, so you can use billions of words.

**Host B**: What objective functions are used?

**Host A**: Two main types. First, autoregressive language modeling. GPT style. Given previous tokens, predict the next token. Maximize P(x_t | x_1, ..., x_{t-1}).

**Host B**: Second?

**Host A**: Masked language modeling, BERT style. Mask some tokens and predict them from context. You can use bidirectional context, making it strong for understanding tasks.

**Host B**: Where does the data come from?

**Host A**: Books, web text, Wikipedia, code, papers, etc. GPT-3 trained on 300 billion tokens, LLaMA on 1.4 trillion tokens. The scale is massive!

---

**[Section 3: Deep Dive into BERT - 4:30]**

**Host B**: Let's look at BERT in detail.

**Host A**: BERT stands for Bidirectional Encoder Representations from Transformers. Google released it in 2018, achieving state-of-the-art on 11 NLP benchmarks.

**Host B**: What was BERT's key innovation?

**Host A**: Bidirectional context. GPT only looks left-to-right, but BERT looks both ways. When predicting [MASK] in "The [MASK] sat on the mat," it can also reference "sat on the mat."

**Host B**: How does masked language modeling work?

**Host A**: Randomly select 15% of input tokens. Of those, 80% are replaced with [MASK], 10% with random tokens, 10% unchanged. This mixture makes training more robust.

**Host B**: There was also next sentence prediction, right?

**Host A**: NSP, yes. Predict whether two sentences are consecutive. But later RoBERTa research showed it doesn't help much. Modern models don't use it.

---

**[Section 4: BERT Fine-tuning - 6:00]**

**Host B**: How do you use BERT for specific tasks?

**Host A**: Fine-tuning! Add task-specific layers on top of pre-trained BERT and train the whole thing with small labeled data.

**Host B**: Examples?

**Host A**: For classification, attach a linear layer to [CLS] token embedding. For named entity recognition, attach linear layers to each token embedding to predict labels. For question answering, predict answer start/end positions.

**Host B**: There are BERT variants too?

**Host A**: Yes! RoBERTa removes NSP and trains longer with more data for better performance. ALBERT reduces size with parameter sharing, and DistilBERT is 40% smaller while maintaining 97% performance through knowledge distillation.

---

**[Section 5: GPT Series - 7:30]**

**Host B**: Now let's talk about GPT.

**Host A**: GPT is Generative Pre-trained Transformer. OpenAI released it in 2018, an autoregressive model using only the decoder.

**Host B**: How is it different from BERT?

**Host A**: BERT is an encoder, good for bidirectional understanding, GPT is a decoder, good for text generation. BERT for classification, GPT for generation.

**Host B**: What makes GPT-3 special?

**Host A**: Scale! 175 billion parameters, trained on 300 billion tokens. And few-shot learning emerged. Without fine-tuning, give a few examples in the prompt and it performs the task.

**Host B**: Example?

**Host A**: "Translate English to French: sea otter → loutre de mer, cheese →" and the model generates "fromage." No gradient updates, just prompting!

---

**[Section 6: Encoder-Decoder Models - 9:00]**

**Host B**: There are encoder-decoder models like T5 too?

**Host A**: Yes! T5 is Text-to-Text Transfer Transformer. It unified all tasks into text-to-text format.

**Host B**: How?

**Host A**: Classification: "classify: I love this!" → "positive", translation: "translate English to French: Hello" → "Bonjour", summarization: "summarize: [long text]" → "[summary]".

**Host B**: What's the advantage?

**Host A**: Same architecture, same loss function for all tasks. Easy to add new tasks. BART is similar, using denoising pre-training, especially strong for summarization.

---

**[Section 7: Parameter-Efficient Fine-tuning - 10:30]**

**Host B**: Full fine-tuning a 175B parameter model for each task would be expensive?

**Host A**: Right! That's why PEFT methods emerged. Parameter-Efficient Fine-Tuning. LoRA is representative.

**Host B**: What's LoRA?

**Host A**: Low-Rank Adaptation. Instead of updating all weights, decompose weight changes into low-rank matrices. In W' = W + BA, only train B and A. With r=8, you only train 0.1% of total parameters!

**Host B**: Other methods?

**Host A**: Adapters insert small modules between layers. Prompt Tuning attaches learnable soft prompts before input. Prefix Tuning adds learnable vectors to attention layers.

---

**[Section 8: Prompting and In-Context Learning - 12:00]**

**Host B**: Let's look at prompt engineering.

**Host A**: Prompting gives text conditions to the model. You can perform tasks without parameter updates. There's zero-shot, one-shot, and few-shot.

**Host B**: Advanced prompting techniques?

**Host A**: Chain-of-Thought is representative! Adding "Let's think step by step" greatly improves reasoning performance. Especially effective for math or logic problems.

**Host B**: Example?

**Host A**: "What's 80% of 15?" Asked directly it might err, but "Let's think step by step, 15% is 0.15, 0.15 times 80 is 12" guides to accuracy.

**Host B**: Fine-tuning or prompting?

**Host A**: For quick prototyping or no labeled data, use prompting. For maximum performance with data, use fine-tuning. Nowadays people combine both.

---

**[Section 9: RLHF and Latest Trends - 13:30]**

**Host B**: How was ChatGPT made?

**Host A**: RLHF, Reinforcement Learning from Human Feedback! Three stages. First, supervised fine-tuning with demonstration data. Second, train reward model with human preferences. Third, use PPO to train policy to maximize reward.

**Host B**: Why is this needed?

**Host A**: Pre-trained models are only good at next token prediction, not following instructions or giving helpful answers. RLHF aligns the model with human preferences.

**Host B**: What's Constitutional AI?

**Host A**: Principle-based learning. Define principles like "be helpful, harmless, and honest," and have the model critique and improve its own responses. Claude uses this approach.

---

**[Outro - 14:30]**

**Host A**: We covered massive content today! Let's summarize.

**Host B**: First, the pre-train then fine-tune paradigm revolutionized AI. Learn general knowledge from large-scale unlabeled data and transfer it.

**Host A**: Second, BERT is strong for bidirectional understanding, GPT for generation. Encoder-decoder like T5 does both well.

**Host B**: Third, PEFT methods like LoRA enable efficient adaptation, and prompting can perform many tasks without fine-tuning.

**Host A**: Fourth, RLHF aligns models with human preferences, creating useful assistants like ChatGPT!

**Host B**: AI is advancing really fast. Don't forget the importance of ethical considerations, bias issues, and responsible use!

**Host A**: Thank you! See you next time!

---

## Key Keywords
- Pre-training, Fine-tuning, Transfer Learning
- Self-supervised Learning, Language Modeling
- BERT (Bidirectional Encoder Representations from Transformers)
- Masked Language Modeling (MLM), Next Sentence Prediction (NSP)
- GPT (Generative Pre-trained Transformer), Autoregressive
- Zero-shot, One-shot, Few-shot, In-context Learning
- T5 (Text-to-Text Transfer Transformer), BART
- PEFT (Parameter-Efficient Fine-Tuning), LoRA, Adapter, Prompt Tuning
- Chain-of-Thought (CoT), Prompt Engineering
- RLHF (Reinforcement Learning from Human Feedback), Constitutional AI
- Instruction Tuning, Alignment, Safety
