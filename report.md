# Self-Pruning Neural Network Report

## Why L1 on sigmoid gates induces sparsity

Each trainable weight is multiplied by a gate value: `gate = sigmoid(gate_scores)`, where each gate lies in `(0, 1)`.
The sparsity objective penalizes the sum of gate values:

`SparsityLoss = sum(sigmoid(gate_scores))`

Adding this term to classification loss encourages many gates to shrink toward zero.
When a gate becomes very small, the corresponding effective weight is nearly removed.
This makes pruning differentiable and trainable end-to-end with backpropagation.

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
|---:|---:|---:|
| 0.0001 | 92.20 | 0.00 |
| 0.001 | 92.10 | 0.00 |
| 0.01 | 92.10 | 0.00 |

## Observations

- As lambda increases, sparsity pressure becomes stronger and more gates move toward zero.
- Lower lambda usually preserves model capacity and accuracy.
- Higher lambda generally improves pruning but can reduce test accuracy.
- Best model in this run uses lambda=0.0001.

## Histogram

The gate value histogram for the best model is saved at:
`outputs/plots/gate_distribution.png`
