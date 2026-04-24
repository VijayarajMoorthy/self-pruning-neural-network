Self-Pruning Neural Network

A PyTorch implementation of a self-pruning neural network that learns to remove its own unnecessary weights during training using learnable gating mechanisms and sparsity regularization.

---

Overview

Traditional pruning is applied after training. This project implements a dynamic pruning approach, where the network learns which connections to keep or remove during training itself.

Each weight is paired with a learnable gate that controls its importance. Over time, unimportant connections are driven toward zero, resulting in a sparser and more efficient model.

---
Idea

- Every weight has an associated gate parameter
- Gates are passed through a sigmoid function → values between 0 and 1
- Effective weight = "weight × gate"
- Add L1 regularization on gates to encourage sparsity

---

Architecture

- Custom "PrunableLinear" layer

- Feedforward neural network built using prunable layers

- Training on CIFAR-10 dataset

- Loss function:
  
  Total Loss = Classification Loss + λ × Sparsity Loss

---

Implementation Details

1. Prunable Linear Layer

- Learnable weights and biases
- Additional "gate_scores" parameter
- Sigmoid activation to generate gates
- Element-wise gating applied to weights

2. Sparsity Regularization

- L1 penalty on gate values
- Encourages many gates → 0
- Controlled using hyperparameter λ (lambda)

3. Training

- Optimizer: Adam
- Loss = CrossEntropy + λ × L1(gates)
- Multiple λ values tested to study trade-off


Evaluation Metrics
- Test Accuracy
- Sparsity Level (%)
  - % of weights with gate value < threshold (e.g., 1e-2)

---


Visualization

- Distribution plot of gate values
- Expected outcome:
  - Large spike near 0 (pruned weights)
  - Cluster away from 0 (important weights)

---

 How to Run

git clone <your-repo-link>
cd self-pruning-neural-network

pip install -r requirements.txt

python train.py

---

Project Structure

├── model.py          # PrunableLinear + network architecture
├── train.py          # Training + evaluation loop
├── utils.py          # Helper functions (sparsity calc, etc.)
├── results/          # Plots and outputs
├── README.md

---

Key Learnings

- Dynamic model compression during training
- L1 regularization for inducing sparsity
- Custom layer design in PyTorch
- Trade-off between efficiency and accuracy

---

Future Improvements

- Structured pruning (neurons instead of weights)
- Hard threshold pruning during training
- Integration with quantization
- Extension to CNN architectures

---

Conclusion

This project demonstrates how neural networks can adapt their own structure during training, leading to more efficient models without requiring a separate pruning phase.

---

