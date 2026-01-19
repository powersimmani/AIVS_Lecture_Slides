# Lecture 03 - Part 1: Mathematical Foundations

## Overview
This part covers the essential mathematical foundations for machine learning, including set theory, linear algebra, and calculus concepts that underpin ML algorithms.

## Key Topics

### 1. Set Theory Basics and Notation
- **Set Definition**: Collection of distinct objects
- **Key Notation**:
  - ∈ (element of), ∉ (not element of)
  - ⊂ (subset), ⊆ (subset or equal)
  - ∪ (union), ∩ (intersection)
  - ∅ (empty set)
- **Set Operations**: Union, intersection, complement, difference
- **ML Application**: Feature spaces, sample spaces, hypothesis classes

### 2. Functions and Mapping Concepts
- **Function Definition**: f: A → B (maps elements from domain A to codomain B)
- **Types**:
  - **Injective** (one-to-one): Different inputs map to different outputs
  - **Surjective** (onto): Every element in codomain has a preimage
  - **Bijective**: Both injective and surjective
- **Composition**: (f ∘ g)(x) = f(g(x))
- **ML Application**: Neural network layers as function compositions

### 3. Vector Spaces and Basis
- **Vector Space**: Set with vector addition and scalar multiplication
- **Key Properties**: Closure, associativity, commutativity, identity, inverse
- **Basis**: Minimal set of linearly independent vectors that span the space
- **Dimension**: Number of vectors in a basis
- **Subspaces**: Subsets that are also vector spaces
- **ML Application**: Feature vectors, weight spaces, latent spaces

### 4. Inner Product and Orthogonality
- **Inner Product (Dot Product)**: ⟨u, v⟩ = Σuᵢvᵢ
- **Properties**:
  - ||v|| = √⟨v, v⟩ (norm/length)
  - cos(θ) = ⟨u, v⟩ / (||u|| ||v||) (angle between vectors)
- **Orthogonality**: ⟨u, v⟩ = 0 means vectors are perpendicular
- **Orthonormal Basis**: Orthogonal vectors with unit length
- **ML Application**: Cosine similarity, PCA, orthogonal regularization

### 5. Matrix Operations and Properties
- **Basic Operations**: Addition, scalar multiplication, matrix multiplication
- **Matrix Multiplication**: (AB)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ
- **Special Matrices**:
  - Identity matrix I
  - Diagonal matrix
  - Symmetric matrix (A = Aᵀ)
  - Orthogonal matrix (AᵀA = I)
- **Transpose**: (AB)ᵀ = BᵀAᵀ
- **ML Application**: Linear transformations, weight matrices, covariance matrices

### 6. Inverse Matrices and Determinants
- **Matrix Inverse**: A⁻¹ such that AA⁻¹ = I
- **Conditions for Invertibility**: Square matrix with non-zero determinant
- **Determinant**: Scalar value indicating matrix properties
  - det(A) = 0 → matrix is singular (not invertible)
  - det(AB) = det(A)det(B)
- **Computing Inverse**: Gaussian elimination, adjugate method
- **Pseudo-inverse**: (AᵀA)⁻¹Aᵀ for non-square matrices
- **ML Application**: Normal equation solution, matrix decompositions

### 7. Eigenvalues and Eigenvectors
- **Definition**: Av = λv where λ is eigenvalue, v is eigenvector
- **Characteristic Equation**: det(A - λI) = 0
- **Properties**:
  - Sum of eigenvalues = trace(A)
  - Product of eigenvalues = det(A)
- **Eigen Decomposition**: A = QΛQ⁻¹ (for diagonalizable matrices)
- **SVD (Singular Value Decomposition)**: A = UΣVᵀ (for any matrix)
- **ML Application**: PCA, spectral clustering, understanding transformations

### 8. Differentiation and Partial Derivatives
- **Derivative**: Rate of change, df/dx = lim(Δx→0) [f(x+Δx) - f(x)]/Δx
- **Partial Derivative**: Derivative with respect to one variable, holding others constant
- **Gradient**: ∇f = (∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ)
- **Chain Rule**: d/dx[f(g(x))] = f'(g(x)) · g'(x)
- **Jacobian**: Matrix of partial derivatives for vector-valued functions
- **Hessian**: Matrix of second-order partial derivatives
- **ML Application**: Gradient descent, backpropagation, optimization

## Important Takeaways
1. Linear algebra provides the language for representing data and transformations in ML
2. Eigenvector decomposition is fundamental to dimensionality reduction
3. Matrix operations must be efficient for large-scale ML (vectorization)
4. The gradient points in the direction of steepest ascent
5. Understanding matrix properties helps diagnose ML problems (e.g., invertibility)
6. Vector spaces formalize the notion of features and representations
