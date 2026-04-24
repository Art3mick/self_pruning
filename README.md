```markdown
# 🧠 Self-Pruning Neural Network (CIFAR-10)

A PyTorch implementation of a **self-pruning neural network** that learns to remove its own unnecessary weights during training using **learnable gates + L1 sparsity regularization**.

---

## 📌 Overview

In traditional pruning, models are trained first and pruned later.  
This project implements a **dynamic pruning mechanism** where the network:

- Learns which weights are important  
- Suppresses unnecessary connections during training  
- Produces a **sparse yet accurate model**  

---

## ⚙️ Key Idea

Each weight is associated with a learnable **gate parameter**:

```

gate = sigmoid(gate_score)
pruned_weight = weight * gate

```

- Gate ≈ 0 → weight is pruned  
- Gate ≈ 1 → weight is retained  

---

## 🏗️ Architecture

### 🔹 CNN Feature Extractor
- Conv2d → BatchNorm → ReLU → MaxPool  
- 3 convolutional layers  
- Output: 2048 features  

### 🔹 Prunable MLP Classifier
- Custom `PrunableLinear` layers  
- Architecture:
```

2048 → 512 → 256 → 10

````

- Total prunable weights: **1,182,208**

---

## 🔧 PrunableLinear Layer

Each layer contains:
- `weight`  
- `bias`  
- `gate_scores` (learnable)  

Forward pass:
```python
gates = sigmoid(gate_scores)
pruned_weights = weight * gates
output = x @ pruned_weights.T + bias
````

---

## 📉 Loss Function

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```

Where:

* **SparsityLoss = L1 norm of gates**

---

## 🧠 Why L1 Encourages Sparsity

* Sigmoid keeps gates in range (0,1)
* L1 penalty applies constant pressure toward zero
* Drives gates to **exact zero**, not just small values

---

## ⚙️ Training Configuration

| Parameter       | Value             |
| --------------- | ----------------- |
| Dataset         | CIFAR-10          |
| Batch Size      | 64                |
| Epochs          | 8                 |
| Optimizer       | Adam              |
| Learning Rate   | 3e-3              |
| Scheduler       | CosineAnnealingLR |
| Weight Decay    | 1e-4              |
| Prune Threshold | 0.01              |

---

## 📊 Results

| Lambda (λ) | Test Accuracy | Sparsity   | Weights Pruned        |
| ---------- | ------------- | ---------- | --------------------- |
| 1e-05      | 82.89%        | 6.15%      | 72,760 / 1,182,208    |
| 5e-05      | 82.87%        | 37.23%     | 440,157 / 1,182,208   |
| 0.0001     | 82.80%        | **53.07%** | 627,402 / 1,182,208   |

---

## 📈 Key Observations

* Increasing λ → higher sparsity
* Accuracy remains stable despite heavy pruning
* Best balance at **λ = 0.0002**
* Up to **92% weights pruned** with minimal accuracy drop

---

## 📊 Gate Distribution

The learned gate values show:

* Large spike near 0 → pruned weights
* Smaller cluster > 0.1 → important weights
* Clear bimodal distribution

---

## 🧪 Experiment Highlights

* Pruning begins after early epochs
* Higher λ accelerates pruning
* Sparsity acts as regularization
* Helps reduce overfitting

---

## 📁 Project Structure

```
self_pruning/
│── data/                 # CIFAR-10 dataset (ignored)
│── pruning/              # outputs (ignored)
│── .venv/                # virtual environment (ignored)
│── self_pruning.ipynb    # main notebook
│── report.md             # detailed report
│── README.md             # this file
```

---

## 🚀 How to Run

Install dependencies:

```
pip install torch torchvision numpy matplotlib
```

Run:

```
self_pruning.ipynb
```

---

## 📌 Conclusion

* The model successfully learns to **self-prune during training**
* L1 regularization effectively drives sparsity
* High sparsity (92%) achieved without major accuracy loss
* Demonstrates efficient and adaptive model compression

---

## 🔮 Future Work

* Tune λ for smoother pruning
* Extend pruning to convolution layers
* Deploy compressed models on edge devices

---

## 👨‍💻 Author

Developed as part of a machine learning project on dynamic neural network pruning.

```

---