# Lecture 20 Podcast: SHAP and Deep Learning XAI - From Game Theory to Grad-CAM

## Episode Information
- **Topic**: SHAP Theory, Shapley Values, TreeSHAP, DeepSHAP, Gradient-based Methods, Grad-CAM
- **Estimated Time**: 15 minutes
- **Target Audience**: Those wanting to dive deep into ML/DL model interpretation

---

## Script

**[Intro - 0:00]**

**Host A**: Hello! AI Vision Systems Podcast. Today we're covering SHAP, the core of XAI, and deep learning explanation methods.

**Host B**: We learned LIME last time, but SHAP we'll learn today is theoretically more solid! It starts from game theory.

**Host A**: Game theory, that's interesting. And we also cover Grad-CAM which shows where CNNs are looking?

**Host B**: Yes! A method to visualize where the image classification model looked and why it said cat. Let's begin!

---

**[Section 1: Game Theory and Shapley Values - 1:30]**

**Host A**: Start by explaining Shapley Values, the foundation of SHAP.

**Host B**: It's a concept from cooperative game theory. Imagine several people work as a team and earn profit. How should we fairly distribute this profit?

**Host A**: According to each person's contribution?

**Host B**: Right! But how do you calculate contribution? Shapley Value is each player's average marginal contribution. The average value increase when that player joins, across all possible orderings.

**Host A**: How is it applied in ML?

**Host B**: Features become players, prediction value becomes payoff. Fairly distributing how much each feature contributed to the prediction.

---

**[Section 2: Mathematical Foundation of SHAP - 3:30]**

**Host A**: Explain the Shapley Value formula.

**Host B**: phi_i = sum [|S|!(|N|-|S|-1)!/|N|!] * [v(S union {i}) - v(S)]. S is a subset excluding feature i, v(S) is the prediction with only those features.

**Host A**: Explain intuitively?

**Host B**: For all combinations S without feature i, calculate prediction value change when i is added, and average with appropriate weights. Weights are determined by combination size.

**Host A**: What are SHAP's key properties?

**Host B**: Four. Efficiency means SHAP values sum equals difference between prediction and base value. Symmetry means equal contribution gives equal value. Dummy means no contribution gives 0. Additivity means consistency across models. These properties make SHAP theoretically solid.

---

**[Section 3: Interpreting SHAP - 5:30]**

**Host A**: How do you interpret SHAP values?

**Host B**: Really intuitive. Positive SHAP increases prediction, negative decreases it. Larger absolute value means greater influence.

**Host A**: Give an example.

**Host B**: For house price prediction. Base value is average house price 200 million. 4 bedrooms: +30 million, downtown location: +50 million, 50-year-old building: -20 million. Sum them for 260 million prediction.

**Host A**: Compared to LIME?

**Host B**: LIME is local linear approximation, so consistency isn't guaranteed. Same feature can have different importance in similar situations. SHAP guarantees consistency thanks to theoretical properties and naturally extends to global view.

---

**[Section 4: SHAP Implementation Methods - 7:30]**

**Host A**: I heard computing exact Shapley Value is difficult?

**Host B**: Yes, it's O(2^n). Even with 20 features, you need to look at over a million combinations. So efficient approximation methods were developed.

**Host A**: What about KernelSHAP?

**Host B**: Similar idea to LIME but weighted linear regression using Shapley kernel. Sampling-based, applicable to all models but approximate.

**Host A**: TreeSHAP?

**Host B**: Tree model specific, computes exact SHAP in polynomial time O(TLD^2)! T is trees, L is leaves, D is depth. Usable for XGBoost, LightGBM, Random Forest, exact yet fast.

**Host A**: DeepSHAP?

**Host B**: Combines DeepLIFT and Shapley. Uses backpropagation in neural networks to compute attribution. Effective for deep learning model explanation.

---

**[Section 5: SHAP Visualization - 9:00]**

**Host A**: Tell me about SHAP visualization types.

**Host B**: Several types. First Waterfall Plot decomposes a single prediction. Shows the process starting from base value, each feature adding or subtracting to reach final prediction.

**Host A**: What about Summary Plot?

**Host B**: Shows global feature importance and distribution simultaneously. Beeswarm form where each point is one instance. X-axis is SHAP value, color shows feature value high/low. For example, shows at a glance if high feature always increases or decreases prediction.

**Host A**: Dependence Plot?

**Host B**: Shows feature effects and interactions. X-axis is feature value, Y-axis is SHAP value, color is automatically detected interaction feature. Can visually identify nonlinear effects and interactions.

---

**[Section 6: Deep Learning XAI - Gradient-based Methods - 10:30]**

**Host A**: Now let's move to deep learning specific XAI.

**Host B**: Yes! Starting with Gradient-based methods. Most basic is Saliency Map. Visualize output's gradient magnitude with respect to input, |df/dx|. Shows which pixels are sensitive to prediction.

**Host A**: Any problems?

**Host B**: Can be noisy and hard to interpret. So SmoothGrad emerged, averaging gradients over multiple samples with added noise. Gives cleaner results.

**Host A**: What about Integrated Gradients?

**Host B**: Most theoretically solid. Integrates gradients along the path from baseline (usually black image) to actual image. Satisfies completeness axiom, so attribution sum equals prediction difference.

---

**[Section 7: Grad-CAM - 12:00]**

**Host A**: Grad-CAM is really widely used?

**Host B**: It's become the standard for CNN explanation! Generalization of Class Activation Mapping. Visualizes by weighting last convolutional layer's feature maps with gradients for the class.

**Host A**: Formula?

**Host B**: L_c = ReLU(sum_k alpha_k^c * A^k). alpha_k^c is global average pooling of gradients, A^k is kth feature map. ReLU takes only positive to show positive contribution.

**Host A**: In code?

**Host B**: There's pytorch_grad_cam library. Create object with GradCAM(model, target_layers), easily compute with cam(input_tensor). There are variants like Grad-CAM++, Score-CAM.

---

**[Section 8: Concept-based Explanations - 13:30]**

**Host A**: What are pixel-level explanation limitations?

**Host B**: People don't think in pixels. "These pixels are important" is harder to understand than "Classified as tiger because of stripe pattern."

**Host A**: That's what TCAV is about?

**Host B**: Right! Testing with Concept Activation Vectors. Finds directions of human-defined concepts in activation space, like stripes, wheels, fur. Then measures how sensitive prediction is to those concepts.

**Host A**: Future directions?

**Host B**: Several trends. Built-in interpretability - designing for interpretability from the start, multi-modal explanations combining text and images, and causality integration to understand causation beyond correlation.

---

**[Outro - 14:30]**

**Host A**: Let's summarize today's key points.

**Host B**: First, SHAP is theoretically solid based on game theory's Shapley Values, and value sum equals prediction minus base difference.

**Host A**: Second, TreeSHAP is exact and fast for tree models, DeepSHAP is applicable to neural networks.

**Host B**: Third, Gradient-based methods and Grad-CAM visualize where CNNs are looking.

**Host A**: Finally, concept-based methods like TCAV enable more human-friendly explanations!

**Host B**: This concludes our XAI series. The ability to understand and explain models is becoming increasingly important. Please practice!

**Host A**: Thank you!

---

## Key Keywords
- SHAP (SHapley Additive exPlanations), Shapley Values
- Cooperative Game Theory, Feature Attribution
- Efficiency, Symmetry, Dummy, Additivity Properties
- KernelSHAP, TreeSHAP, DeepSHAP
- Waterfall Plot, Summary Plot, Dependence Plot
- Saliency Map, SmoothGrad, Integrated Gradients
- CAM, Grad-CAM, Grad-CAM++, Score-CAM
- TCAV (Testing with Concept Activation Vectors)
- Attention Mechanism, Faithfulness
