# Lecture 07 Podcast: Everything About Data Modalities and Feature Extraction

## Episode Information
- **Topic**: Various Data Types and Traditional/Deep Learning-based Feature Extraction
- **Estimated Time**: 15 minutes
- **Target Audience**: ML practitioners and researchers working with various data types

---

## Script

**[Intro - 0:00]**

**Host A**: Hello! Welcome to the AI Vision Systems Podcast. Today we're going to cover a really important topic.

**Host B**: Yes! Today we'll talk about data modalities. Text, images, audio, video... We'll explore what characteristics each data type has and how they should be processed.

**Host A**: Multimodal AI is hot these days. GPT-4 understands images and processes voice too. These are the foundational concepts for that technology.

**Host B**: Exactly! Understanding the characteristics of each data type is the first step in multimodal learning.

---

**[Section 1: What is Data Modality? - 1:30]**

**Host A**: First, can you explain what data modality is?

**Host B**: Modality simply means the type of data. There's text, images, audio, video, time series, graphs, tabular data, etc. Each has completely different structures.

**Host A**: What's the difference between structured and unstructured data?

**Host B**: Structured data is neatly organized in table format. Like CSV files or databases. On the other hand, unstructured data like text, images, and audio has no fixed format.

**Host A**: I heard 80% of enterprise data is unstructured?

**Host B**: Yes, that's why unstructured data processing technology is important. Deep learning revolutionized this area.

---

**[Section 2: Text and Image Data - 3:30]**

**Host A**: Shall we look more closely at the characteristics of each modality? Starting with text data.

**Host B**: Text is sequential and variable-length. It consists of discrete tokens like words or characters. The biggest challenge is ambiguity. The same word can have different meanings depending on context.

**Host A**: For example?

**Host B**: "Bank" could mean a financial institution or a riverbank - you need context to know. And each language requires different processing methods. Korean needs morphological analysis.

**Host A**: What about images?

**Host B**: Images are grids of pixels. Height times width times channels. If it's RGB, that's 3 channels. A 1000 by 1000 by 3 image has 3 million values!

**Host A**: That's extremely high-dimensional.

**Host B**: That's why we need CNNs. They efficiently process by utilizing spatial structure. Images also have challenges like viewpoint changes, occlusion, and lighting.

---

**[Section 3: Audio and Video Data - 5:30]**

**Host A**: What characteristics does audio data have?

**Host B**: Audio is a one-dimensional time signal. The sample rate is important - CD quality is 44.1kHz, meaning 44,100 samples per second! For speech recognition, we usually use 16kHz.

**Host A**: How do you represent audio?

**Host B**: There are several methods. You can use the raw waveform as is, or convert it to a spectrogram to treat it like a 2D image. Spectrograms are time-frequency representations.

**Host A**: What about video?

**Host B**: Video is a sequence of images with an added time dimension. Time times height times width times channels. The dimensionality is really high!

**Host A**: That's why video processing is difficult.

**Host B**: Yes, the computational cost is enormous. We use frame sampling, 3D CNNs, or optical flow.

---

**[Section 4: Graph and Multimodal Data - 7:30]**

**Host A**: Graph data is special, right?

**Host B**: Yes! Graphs consist of nodes and edges. They represent relational data like social networks, molecular structures, and knowledge graphs. They're not Euclidean structures, so they need special processing.

**Host A**: That's why GNNs, Graph Neural Networks, were created?

**Host B**: Exactly! They learn both node features and connection structure together. Since they're variable-sized with permutation invariance, they need different approaches than CNNs or RNNs.

**Host A**: What about multimodal data?

**Host B**: It handles multiple modalities together. Movies have video and audio together, SNS posts have images and text. VQA, Visual Question Answering, is a representative task.

**Host A**: What are the challenges of multimodal learning?

**Host B**: There's alignment between modalities, integrating different representation methods, handling missing modalities. Fusion strategies are also important.

---

**[Section 5: Traditional Feature Extraction - Text - 9:00]**

**Host A**: Now let's move on to feature extraction? How was it done before deep learning?

**Host B**: Traditional methods are still useful! For text, Bag of Words, BoW, is basic. It represents documents as word frequency vectors.

**Host A**: What are the disadvantages?

**Host B**: It ignores word order. And treats all words equally. TF-IDF improves this. It multiplies Term Frequency by Inverse Document Frequency to reduce the weight of common words.

**Host A**: What about N-grams?

**Host B**: It treats N consecutive words as one. Bigram captures word pairs like "machine learning". It can reflect local context to some extent.

---

**[Section 6: Traditional Feature Extraction - Image/Audio - 10:30]**

**Host A**: What about image feature extraction?

**Host B**: There were methods like SIFT, SURF, and HOG. SIFT finds keypoints that are invariant to scale and rotation. It creates 128-dimensional descriptors.

**Host A**: What's HOG?

**Host B**: Histogram of Oriented Gradients. It divides images into cells and creates gradient direction histograms for each cell. It was widely used for pedestrian detection.

**Host A**: What about audio?

**Host B**: We use FFT to convert from time domain to frequency domain and create spectrograms. MFCC is the most standard - it uses the Mel scale that reflects human auditory characteristics. It typically represents speech with 13-20 coefficients.

**Host A**: Are these still used in the deep learning era?

**Host B**: Yes, MFCC is still widely used. And traditional methods help understand how deep learning works.

---

**[Section 7: Deep Learning-based Representation Learning - 12:00]**

**Host A**: What's different about deep learning's representation learning?

**Host B**: The key is automatically learning features. It finds optimal representations from data without manual design.

**Host A**: Can you give an example?

**Host B**: Word embeddings like Word2Vec. They represent words as 50-300 dimensional dense vectors. The amazing thing is vector arithmetic like "king - man + woman = queen" works!

**Host A**: What about CNN's feature learning?

**Host B**: CNNs learn hierarchically. Early layers capture edges or textures, middle layers parts, and upper layers recognize whole objects. Features learned on ImageNet transfer well to other tasks.

**Host A**: There are autoencoders too, right?

**Host B**: Yes! They learn to compress input and then reconstruct it. The latent space becomes a useful representation. VAE adds probabilistic elements so it can also be used as a generative model.

---

**[Section 8: Transfer Learning and Fine-tuning - 13:30]**

**Host A**: Why is transfer learning important?

**Host B**: It's especially important when you have little data! CNN features learned on ImageNet work well on other vision tasks too. Same with language representations like BERT and GPT.

**Host A**: What's the difference between feature extraction and fine-tuning?

**Host B**: Feature extraction freezes the pretrained model and only trains a new classifier. It's fast and works with little data. Fine-tuning also updates the pretrained weights. Performance is better but needs more data.

**Host A**: When should you use which?

**Host B**: Use feature extraction when you have little data and similar domains, fine-tuning when you have lots of data or different domains. There's also gradual unfreezing of layers.

---

**[Outro - 14:30]**

**Host A**: We covered a lot today! Shall we summarize?

**Host B**: First, each data modality has unique characteristics requiring appropriate processing methods.

**Host A**: Second, traditional feature extraction methods like TF-IDF, SIFT, and MFCC are still useful and helpful to understand.

**Host B**: Third, deep learning automatically learns hierarchical features, replacing manual design.

**Host A**: Finally, transfer learning and fine-tuning are key strategies for achieving good performance with limited data!

**Host B**: Next time we'll cover loss functions and optimization. Please stay tuned!

**Host A**: Thank you! See you next time!

---

## Key Keywords
- Data Modality, Structured/Unstructured Data
- Text: Bag of Words, TF-IDF, N-gram, Tokenization
- Image: SIFT, SURF, HOG, Edge Detection
- Audio: FFT, Spectrogram, MFCC, Mel Scale
- Video: 3D CNN, Optical Flow, Frame Sampling
- Graph: GNN, Node, Edge, Non-Euclidean
- Representation Learning, Word2Vec, GloVe
- Transfer Learning, Fine-tuning, Feature Extraction
- Autoencoder, VAE, Multimodal Fusion
