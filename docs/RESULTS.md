# Test Results

All results are from experiments conducted during development.
Code was tested on CPU (no GPU required).

## Closed-set accuracy (5 seeds)

| Dataset | MLP | CF-Voronoi (best config) | Config | Params ratio |
|---|---|---|---|---|
| digits | 0.9572 ± 0.0072 | **0.9711 ± 0.0028** | D=32, P=3 | 0.88× |
| wine | 0.9833 ± 0.0136 | **0.9889 ± 0.0136** | D=16, P=5 | 2.2× |
| breast_cancer | 0.9684 ± 0.0119 | 0.9737 ± 0.0055 | D=16, P=5 | 1.9× |

**Observation**: CF-Voronoi is more stable (lower std) on digits.

## Open-set AUROC (digits: known=[0-5], unknown=[6-9])

| Method | AUROC | FPR@95 | Verdict |
|---|---|---|---|
| MLP + softmax | 0.944 | 0.294 | baseline |
| **MLP + Energy** | **0.977** | **0.105** | 🏆 best |
| MLP + Mahalanobis | 0.911 | 0.483 | |
| ProtoNet | 0.969 | 0.119 | |
| CF-Voronoi | 0.955 | 0.266 | good, not best |

## Open-set AUROC (Fashion-MNIST: known=[0-4], unknown=[5-9])

| Method | AUROC | Verdict |
|---|---|---|
| MLP + softmax | 0.652 | |
| MLP + Energy | 0.508 | broken on high-dim |
| **MLP + Mahalanobis** | **0.866** | 🏆 best |
| ProtoNet | 0.771 | |
| CF-Voronoi | 0.721 | worse than Mahalanobis |

**Observation**: Euclidean distance to prototypes fails on complex 
data where class distributions overlap. Mahalanobis (class covariance) 
is needed. **This is a real limitation, not a bug.**

## Few-shot class extension (digits, 4 new classes × 5 examples)

| Method | Data used | Accuracy |
|---|---|---|
| MLP full retrain | 1437 samples | 0.950 |
| **CF-Voronoi +5-shot** | **20 samples** | **0.806** |

**This is the strongest practical argument**: 85% of MLP accuracy with 1.4% of the data, zero gradient steps.

## Stability

CF-Voronoi `D=32, P=3` on digits:
- mean accuracy: 0.9711
- std over 5 seeds: 0.0028
- **2.6× more stable** than MLP (std 0.0072)

## What was NOT tested

- CIFAR-10, CIFAR-100, ImageNet
- Real-world production data
- int8 quantization / actual MCU deployment
- Latency benchmarks on ARM Cortex-M
- Comparison with modern OOD methods (ODIN, ensembles)
- Hyperparameter sensitivity (how much does D matter exactly?)

If you test any of these, please open a PR with results.

## Known bugs

- None currently reported. If you find one, open an issue.
