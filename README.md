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

## Reproduction Results

Setting: CIFAR-100, 10 tasks, Offline CL, OUR(G), DER++ based, buffer 500, seed 0, RTX 3090

| Model | CIL (%) | TIL (%) | BWT |
|-------|--------:|--------:|----:|
| Baseline (DER++) | 38.28 | 77.09 | -53.51 |
| **DEDUCE** | **44.39** | **79.79** | **-25.62** |

## License

The official DEDUCE code follows the original authors' license. See `LICENSE` for details.
