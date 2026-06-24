# AutoDiff Engine + Neural Network From Scratch

> *Built backpropagation from scratch — the same algorithm that trains every neural network in the world.*

A lightweight deep learning engine built using only NumPy. No PyTorch. No TensorFlow. No autograd libraries. Every gradient computed manually through a custom computational graph.

Implements both **Forward Mode** and **Reverse Mode** Automatic Differentiation, then uses the reverse mode engine to train a 2-layer neural network for classification.

---

## Why This Project

Most ML engineers use `loss.backward()` without understanding what happens inside. This project builds that mechanism from the ground up — dual numbers for forward mode, dynamic computation graphs for reverse mode, broadcasting-aware gradient propagation, and a working neural network on top.

---

## What Is Built

### Forward Mode — Dual Numbers

Each variable carries two values simultaneously:

```
x = (value, derivative)
```

Operations propagate both automatically. Supports: `add`, `sub`, `mul`, `div`, `pow`, `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `relu`.

### Reverse Mode — Computation Graph

Each node in the graph stores:

```
value      →  result of forward computation
gradient   →  accumulated during backward pass
parents    →  nodes this node depends on
backward   →  local gradient function
```

Backward pass traverses the graph in reverse topological order, applying chain rule at each node.

Supports:
- Tensor and matrix gradients
- Matrix multiplication gradients
- Broadcasting-aware gradient correction
- Scalar loss backpropagation

---

## Broadcasting Handling

During forward pass — tensors broadcast automatically.

During backward pass — gradients must return to original tensor shapes. Custom logic reduces broadcasted gradients by summing over expanded dimensions before continuing reverse propagation.

---

## Loss Functions

**Mean Squared Error** — for regression:

$$L = \frac{1}{n} \sum (\hat{y} - y)^2 \qquad \frac{\partial L}{\partial \hat{y}} = \frac{2(\hat{y} - y)}{n}$$

**Cross Entropy Loss** — for classification:

$$\frac{\partial L}{\partial z} = \frac{p - y}{m}$$

where $p$ = predicted probabilities, $y$ = one-hot labels, $m$ = batch size.

---

## Neural Network

2-layer network built directly on the reverse mode engine:

$$z_1 = W_1 X + b_1 \qquad a_1 = \text{ReLU}(z_1)$$

$$z_2 = W_2 a_1 + b_2 \qquad \hat{y} = \text{Softmax}(z_2)$$

Gradient descent update:

$$\theta_{\text{new}} = \theta - \eta \frac{\partial L}{\partial \theta}$$

---

## Training Pipeline

```
Forward Pass
      ↓
Scalar Loss
      ↓
Backpropagation (graph traversal)
      ↓
Gradient Descent
      ↓
Parameter Update
```

---

## Example Output

```python
Initial loss  —  2.34
Final loss    —  0.07
```

Predicted probabilities:
```python
[[0.99  0.60  0.04  0.00]
 [0.01  0.39  0.95  1.00]]
```

Predicted classes:
```python
[0  0  1  1]
```

---

## Project Structure

```
autodiff/
├── core/
│   ├── forward_mode.py     # Dual class + forward mode ops
│   └── matrix_form.py      # Tensor + backward engine
├── nn/
│   └── neural_network.py   # 2-layer network
└── main.py                 # Demo — forward mode + reverse mode + MLP
```

---

## Run

```bash
python main.py
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python |
| Numerics | NumPy only |
| Frameworks | None |

---

*No autograd libraries used. Every derivative is computed through the custom engine built in this project.*

---

## Author

**Dola Sreecharan** — Self-taught Machine Learning Engineer

Built this project to understand backpropagation from first principles, not just use it.

- 🐙 GitHub: [sreecharan-dola](https://github.com/sreecharan-dola)
- 💼 LinkedIn: [Dola Sreecharan](https://linkedin.com/in/sreecharan-dola)
- 📧 sreecharan.dola@gmail.com

---

*If this helped you understand AutoDiff — give it a ⭐*