# Lecture 05 Podcast: The Birth of Neural Networks - From Perceptron to MLP

## Episode Information
- **Topic**: Neural network motivation, MLP structure, forward propagation, backpropagation
- **Estimated Time**: 15 minutes
- **Target Audience**: Deep learning beginners, those wanting to understand neural network principles

---

## Script

**[Intro - 0:00]**

**Host A**: Hello! This is the AI Vision Systems Podcast. It's finally time to talk about the core of deep learning - neural networks!

**Host B**: Yes! Today we'll start with why neural networks are needed, then cover how they're structured and how they learn.

**Host A**: Logistic regression is good, so why do we need neural networks?

**Host B**: That question is the starting point! Let's look at the famous XOR problem first.

---

**[Section 1: XOR Problem and Linear Inseparability - 1:30]**

**Host B**: Looking at the XOR truth table, (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0. If you plot points on a graph, diagonal pairs belong to the same class.

**Host A**: You can never separate them with a single line!

**Host B**: Exactly! In 1969, Minsky and Papert proved that perceptrons cannot solve XOR. This was one of the causes of the AI winter.

**Host A**: So how do you solve it?

**Host B**: The key is feature space transformation! Even if it's linearly inseparable in the original space, it becomes separable with the right transformation. Neural networks automatically learn this transformation.

**Host A**: Not manual feature engineering?

**Host B**: Right! That's the core of Representation Learning. The network finds good representations by itself.

---

**[Section 2: Multi-Layer Perceptron Structure - 3:30]**

**Host A**: So that's where Multi-Layer Perceptron, MLP, comes in!

**Host B**: Correct. The structure is input layer → hidden layer(s) → output layer. Each layer goes through linear transformation and nonlinear activation.

**Host A**: What's the formula?

**Host B**: h⁽ˡ⁾ = σ(W⁽ˡ⁾h⁽ˡ⁻¹⁾ + b⁽ˡ⁾). Multiply the previous layer's output by a weight matrix, add bias, and apply activation function.

**Host A**: For example, MNIST classification?

**Host B**: You could have 784 inputs (28x28 image), hidden layers of 256→128→64, and 10 outputs (digits 0-9). The number of parameters is the sum of all weights and biases in each layer connection.

**Host A**: That's quite a lot of parameters.

**Host B**: Yes! 784×256 + 256 + 256×128 + 128 + ... and so on. That's why initialization is important. Use Xavier initialization or He initialization.

---

**[Section 3: Importance of Activation Functions - 5:30]**

**Host A**: Why do we need activation functions?

**Host B**: That's a key question! If you stack multiple layers without activation functions, it ultimately becomes one linear transformation. W₂(W₁x + b₁) + b₂ = W₂W₁x + constant, which is just a linear model!

**Host A**: You need nonlinearity to represent complex functions.

**Host B**: Right! Sigmoid outputs to the (0,1) range so it can be interpreted like a probability. tanh outputs to (-1,1) and is zero-centered, so it trains better than sigmoid.

**Host A**: But these days ReLU is used a lot, right?

**Host B**: Yes! ReLU(z) = max(0, z). It's simple, and when z>0 the gradient is 1, so there's no vanishing gradient problem. But when z<0 the gradient is 0, causing the "dying ReLU" problem.

**Host A**: So that's why Leaky ReLU and ELU came out.

**Host B**: Exactly! Leaky ReLU gives a small slope (about 0.01) when z<0. Basically use ReLU, and if problems arise, try variations.

---

**[Section 4: Output Layer and Loss Functions - 7:30]**

**Host A**: The output layer varies by problem, right?

**Host B**: Yes! For regression, no activation (Linear) with MSE loss. For binary classification, sigmoid with Binary Cross-Entropy. For multi-class, Softmax with Categorical Cross-Entropy.

**Host A**: Do you have to match them?

**Host B**: They're mathematically connected. The Softmax + Cross-Entropy combination produces clean gradients. Wrong combinations can prevent learning or make it unstable.

**Host A**: I've heard of the Universal Approximation Theorem.

**Host B**: It's a theorem stating that with just one hidden layer, if you have enough neurons, you can approximate any continuous function! But "enough" can be exponentially many, so deep networks are more efficient.

---

**[Section 5: Forward Propagation Algorithm - 9:00]**

**Host A**: How does Forward Propagation work?

**Host B**: I'll explain step by step. First, input a⁽⁰⁾ = x. At each layer l, compute linear combination z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾, then activate a⁽ˡ⁾ = σ(z⁽ˡ⁾). Finally ŷ = a⁽ᴸ⁾ is the prediction.

**Host A**: Can you give a simple numerical example?

**Host B**: If input is [1.0, 0.5] and after the first layer z⁽¹⁾ = [0.8, 1.2], applying sigmoid gives a⁽¹⁾ = [0.69, 0.77]. If at the next layer z⁽²⁾ = 1.1, the final output is ŷ = σ(1.1) = 0.75.

**Host A**: Loss is calculated with this prediction and the actual label, right?

**Host B**: Exactly! Learning is updating weights so the loss becomes smaller. That's Backpropagation.

---

**[Section 6: Backpropagation Algorithm - 10:30]**

**Host A**: Backpropagation is the core of deep learning!

**Host B**: Yes! The core idea is the Chain Rule. The derivative of a composite function is the product of derivatives at each step. d/dx[f(g(x))] = f'(g(x)) · g'(x).

**Host A**: How is it applied to neural networks?

**Host B**: Calculate error at the output layer δ⁽ᴸ⁾ = ∂L/∂z⁽ᴸ⁾, and propagate it backward. δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾ ⊙ σ'(z⁽ˡ⁾). The gradient for each layer is ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ.

**Host A**: It would be complex to calculate manually.

**Host B**: That's why we have PyTorch and TensorFlow! Autograd creates a computational graph and automatically calculates gradients. One line of loss.backward() and it's done!

---

**[Section 7: Mini-batch Gradient Descent - 12:00]**

**Host A**: How do you actually train?

**Host B**: There are three methods. Batch GD uses all data but is slow. SGD uses one sample at a time, fast but unstable. Mini-batch is a compromise!

**Host A**: How do you determine mini-batch size?

**Host B**: Usually use powers of 2: 32, 64, 128, 256... Good to make it as large as possible while fitting in GPU memory. Too small means high noise, too large means slow convergence.

**Host A**: Learning Rate is also important, right?

**Host B**: It's the most important hyperparameter! Too large and it diverges, too small and it doesn't converge. Usually start with about 0.001 and adjust. Adaptive optimizers like Adam automatically adjust the learning rate.

---

**[Section 8: Practical Tips and Debugging - 13:30]**

**Host A**: Please give us some practical training tips!

**Host B**: First, input normalization! Adjusting to mean 0 and standard deviation 1 makes training much more stable. Second, try overfitting on a small batch! Check if the model can learn first.

**Host A**: What if problems occur?

**Host B**: If loss doesn't decrease, check the learning rate. If you get NaN, the learning rate is too high or there's a data problem. You can also use Gradient Checking to verify backpropagation implementation.

**Host A**: What's the PyTorch code structure?

**Host B**: Define network with model = nn.Sequential(...), set loss function with criterion, optimizer with optimizer. In the loop: output = model(x), loss = criterion(output, y), optimizer.zero_grad(), loss.backward(), optimizer.step() in that order!

---

**[Outro - 14:30]**

**Host A**: Shall we summarize the key points today?

**Host B**: First, as the XOR problem shows, to overcome the limitations of linear models, we need multi-layer structure!

**Host A**: Second, activation functions provide nonlinearity. ReLU is the default choice.

**Host B**: Third, predict with forward propagation, calculate gradients with backpropagation, and update weights.

**Host A**: Fourth, balance efficiency and stability with mini-batch, and modern frameworks automatically calculate gradients for us!

**Host B**: Next time we'll learn how to handle images with CNN. Look forward to it!

**Host A**: Thank you!

---

## Key Keywords
- XOR Problem, Linear Separability
- Representation Learning, Feature Transformation
- MLP (Multi-Layer Perceptron)
- Activation Functions: Sigmoid, Tanh, ReLU, Leaky ReLU
- Universal Approximation Theorem
- Forward Propagation, Backward Propagation
- Chain Rule, Computational Graph, Autograd
- Mini-batch Gradient Descent, Learning Rate
