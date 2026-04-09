# DEDUCE Reproduction (ICLR 2026)

DEDUCE is a continual learning method that combines Dark Experience Replay++ (DER++) with Learned Unlearning Mechanism (LUM) and Gradient Update Mechanism (GUM) to mitigate catastrophic forgetting. This repository reproduces the results from the official DEDUCE codebase on CIFAR-100.

## Setup

```bash
pip install -r requirements.txt
```

CIFAR-100 is automatically downloaded on first run.

## Usage

```bash
# DER++ baseline
./run_baseline.sh

# DEDUCE (full reproduction)
./run_deduce.sh
```

## Analysis

Setting: CIFAR-100, 10 tasks, Offline CL, OUR(G), DER++ based, buffer 500, seed 0, RTX 3090

| Model | CIL (%) | TIL (%) | BWT |
|-------|--------:|--------:|----:|
| Baseline (DER++) | 38.28 | 77.09 | -53.5 |
| **DEDUCE** | **44.39** | **79.79** | **-28.2** |
| DEDUCE (LUM + Projection) | 43.9 | 80.1 | -28.2 |
| DEDUCE (Fisher Multi-task) | 36.96 | 76.36 | -53.5 |

## License

The official DEDUCE code follows the original authors' license. See `LICENSE` for details.
