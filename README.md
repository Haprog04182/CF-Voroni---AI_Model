# CF-Voronoi: Compact Prototype-Based Classifier

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research](https://img.shields.io/badge/Status-Early%20Research-orange.svg)]()

A compact, interpretable, prototype-based classifier with:
- **Few-shot class extension** without retraining (add new classes from ~5 examples)
- **Built-in OOD detection** (detect "unknown" inputs naturally via distance)
- **Interpretable prototypes** (each prototype ≈ a real training example)
- **Small footprint** (~3K–7K parameters, fits on microcontrollers)

> ⚠️ **Honest disclosure**: This project started as an exploration of alternative 
> neural architectures (TRIZ-inspired). The author is **not a professional ML 
> researcher** — the code, tests, and benchmarks were written and executed 
> **with AI assistance** (Claude, GPT-4). Every result below is **real and 
> reproducible**, but this is **early-stage research**, not a peer-reviewed 
> method. **Please test it yourself, report what works, and where it fails.**

---

## What is this?

CF-Voronoi is a classifier that:
1. Encodes input `x` into a latent space `z` (via small MLP)
2. Keeps `K × P` **prototypes** (learned "anchor points") in that space
3. Classifies by **soft nearest-prototype distance**:
   `logit_k = τ · logsumexp(-‖z - c_kj‖² / τ)`

The name is from **Voronoi diagrams** — the decision boundaries are 
mathematically equivalent to Voronoi cells around the prototypes.

**Relation to existing methods** (please read — this is not "novel"):
- This is a **Prototypical Network** (Snell 2017) with soft assignment
- The attention mechanism is mathematically equivalent to **Hopfield 
  Networks** (Ramsauer 2020) and **cross-attention** in Transformers
- The OOD detection via distance is standard **Mahalanobis / 
  Energy-based OOD** territory (Lee 2018, Liu 2020)
- Hierarchical extension ≈ **Mixture of Experts** or **Memory Networks**

**What's (possibly) new**: the specific combination of softmin + 
regularizer + open-world deployment in a single compact framework.

---

## Quick start

```bash
pip install -r requirements.txt
python examples/01_basic_digits.py
```

Expected output:
```
digits: train=1437, test=360
MLP hidden=32:      3466 params, acc=0.9500
CF-Voronoi D=8,P=3:  761 params, acc=0.9639  (0.22x MLP params)
CF-Voronoi D=16,P=5:1841 params, acc=0.9722  (0.53x MLP params) ⭐
CF-Voronoi D=32,P=3:3041 params, acc=0.9750  (0.88x MLP params)
```

---

## Results (honest summary)

### Closed-set accuracy (5 seeds, mean ± std)

| Dataset | MLP | CF-Voronoi (best) | Δ | Params |
|---|---|---|---|---|
| digits | 0.957 ± 0.007 | **0.971 ± 0.003** | +1.4 п.п. | 0.88× |
| wine | 0.983 ± 0.014 | **0.989 ± 0.014** | +0.6 п.п. | 2.2× |
| breast_cancer | 0.968 ± 0.012 | 0.974 ± 0.006 | +0.6 п.п. | 1.9× |

**Verdict**: CF-Voronoi **beats MLP** on closed-set accuracy, but 
with **comparable or slightly more** parameters on larger datasets.

### Open-set recognition (detect "unknown" classes)

**digits** (known=[0-5], unknown=[6-9]):
| Method | AUROC | FPR@95 |
|---|---|---|
| MLP + softmax | 0.944 | 0.294 |
| **MLP + Energy** | **0.977** | **0.105** 🏆 |
| MLP + Mahalanobis | 0.911 | 0.483 |
| ProtoNet | 0.969 | 0.119 |
| CF-Voronoi | 0.955 | 0.266 |

**Fashion-MNIST** (known=[0-4], unknown=[5-9]):
| Method | AUROC |
|---|---|
| **MLP + Mahalanobis** | **0.866** 🏆 |
| ProtoNet | 0.771 |
| CF-Voronoi | 0.721 |
| MLP + softmax | 0.652 |

**Verdict (honest)**: CF-Voronoi is **competitive but NOT SOTA** on 
open-set. MLP+Energy wins on digits, MLP+Mahalanobis wins on 
Fashion-MNIST. **The "built-in OOD detection" claim is weak** — it works 
on simple data, but fails on complex data where class distributions 
overlap. A margin-based loss helps (0.59 → 0.72 on Fashion-MNIST), 
but doesn't reach Mahalanobis.

### Few-shot class extension (the real strength)

**Scenario**: model trained on digits 0-5. Add classes 6-9 
with only 5 examples each (20 samples total).

| Method | Data used | Accuracy (10 classes) |
|---|---|---|
| MLP full retrain | 1437 samples | 0.950 |
| **CF-Voronoi +5-shot** | **20 samples** | **0.806** |

**Verdict**: CF-Voronoi achieves **84.8% of MLP accuracy** 
using **1.4% of the data**, with **zero gradient steps** (only 
forward pass + prototype averaging).

**This is the strongest practical argument for the method** — 
in production, adding a class from 5 examples is far cheaper than 
retraining.

---

## Honest limitations

1. **Not SOTA on open-set OOD** — MLP+Energy and MLP+Mahalanobis beat it.
2. **Euclidean distance collapses on complex data** — Fashion-MNIST 
   needs Mahalanobis (models class covariance) rather than prototypes 
   (only class centers).
3. **Fixed P (prototypes per class) is fragile** — different classes 
   may need different P.
4. **Only tested on small datasets** — digits (1797 samples), wine (178), 
   breast_cancer (569), Fashion-MNIST subset (10K). **Not tested on 
   CIFAR, ImageNet, or real-world production data.**
5. **No int8 quantization tested** — MCU claim is theoretical.
6. **Threshold for OOD is manual** — needs validation split.
7. **Code quality** — written by AI, minimal testing. Bugs likely.

---

## What I want from you

If you try this, please:

1. **Run the tests** (`pytest tests/`) — report if anything fails.
2. **Run on your own data** — even just a different sklearn dataset.
3. **Try the extensions** (see `examples/04_full_benchmark.py`):
   - Different `D` (8, 16, 32, 64)
   - Different `P` (1, 3, 5)
   - Cosine distance instead of Euclidean
   - Margin loss
4. **Open an issue** with:
   - What you tried
   - What worked
   - What broke
   - Any papers you think are relevant (I may have missed them)

**Honest reports are more valuable than "it works great!"** — if the 
method fails on your data, that's **useful information**.

---

## References

If you want to understand what CF-Voronoi **isn't** but resembles:

- Snell et al., "Prototypical Networks for Few-Shot Learning" (NeurIPS 2017)
- Kohonen, "Learning Vector Quantization" (1990)
- Ramsauer et al., "Hopfield Networks is All You Need" (2020) — attention = soft nearest neighbor
- Bendale & Boult, "OpenMax" (CVPR 2016)
- Lee et al., "A Simple Unified Framework for Detecting OOD" (NeurIPS 2018) — Mahalanobis
- Liu et al., "Energy-based OOD Detection" (NeurIPS 2020)
- Graves et al., "Neural Turing Machines" (2014)
- Broomhead & Lowe, "RBF Networks" (1988)

**I did not cite these before building the project** — they were 
pointed out during review. If you know of better references, please 
add them.

---

## License

MIT — use it, break it, fix it, publish your results.

---

## Author's note

This started as a personal exploration of "what if neural networks 
worked differently" — inspired by TRIZ (inventive problem solving). 
The result is a working prototype, but **not a polished method**.

I'm publishing it because:
1. **Honest benchmarks** (positive AND negative results) are useful
2. **Few-shot extension** is a real, practical strength
3. **Someone smarter** might see the potential and improve it

If you do something cool with it, please let me know. If it's broken, 
please open an issue.

— Author, using AI as research assistant
# CF-Voroni---AI_Model
