# Lecture 9: Initialization and Normalization

## 📋 Overview

**Instructor:** Ho-min Park  
**Email:** homin.park@ghent.ac.kr | powersimmani@gmail.com

This lecture implements stable and efficient training through weight initialization and normalization techniques.

---

## 🎯 Learning Objectives

1. Understand and apply initialization strategies
2. Compare and select normalization techniques
3. Understand symmetry problem
4. Understand relationship between batch size and normalization
5. Select optimal techniques for different architectures

---

## 📚 Key Topics

**Initialization**: Zero (fails), Random, Xavier/Glorot, He, LSUV
**Batch Normalization**: Reduces internal covariate shift, accelerates training
**Layer Normalization**: Suitable for Transformers, batch size independent
**Group Normalization**: Suitable for small batches
**Instance Normalization**: Used in style transfer

---

## 💡 Key Concepts

- Zero initialization fails due to symmetry problem
- Xavier for Tanh/Sigmoid, He for ReLU
- Batch Norm accelerates and stabilizes training
- Layer Norm suitable for sequence models
- Normalization enables higher learning rates

---

## 🛠️ Prerequisites

- Basic Python programming
- Understanding of previous lecture content
- Basic machine learning concepts

---

## 📖 Additional Resources

For detailed code examples, practice materials, and slides, please refer to the original lecture files.
Lecture materials: HTML-based interactive slides provided
