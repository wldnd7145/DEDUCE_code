# DEDUCE Reproduction & Ablation

DEDUCE (ICLR 2026) 논문 재현 환경. 공식 코드(`DEDUCE_code/`) 위에 추가 ablation 및
변형 실험을 통합했다. 모든 실험은 `seq-cifar100` (10 tasks, 100 classes), buffer 500,
seed 0 기준이다.

## 환경 설치

Python 3.10, CUDA 11.8 호환 PyTorch 사용. CUDA 드라이버가 11.4여도 PyTorch가 cu118 런타임을
번들로 포함하므로 문제 없이 동작한다.

```bash
conda create -n deduce python=3.10 -y
conda activate deduce
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 디렉토리 구조

```
DEDUCE_code/
├── DEDUCE_code/                # 공식 코드 + custom 모델
│   ├── models/
│   │   ├── derpp.py            # 공식 DEDUCE (DER++ + LUM + GUM + EWC)
│   │   ├── derpp_baseline.py   # 순수 DER++ baseline (DEDUCE 제거)
│   │   ├── exp_a_lum_proj.py   # LUM + 학습 시 gradient projection
│   │   ├── exp_fisher_multitask.py        # LUM 대체: Fisher 가중 multi-task
│   │   ├── exp_fisher_multitask_strong.py # 강한 selectivity 버전
│   │   └── exp_*.py            # 기타 변형 실험들
│   ├── utils/
│   │   ├── main.py             # 진입점 (--stop_after 지원 추가)
│   │   ├── training.py
│   │   └── ...
│   └── ...
├── logs/                       # 실험 로그
├── run_baseline.sh             # DER++ baseline
├── run_deduce.sh               # 공식 DEDUCE 재현
├── run_exp.sh                  # custom 실험 실행
└── requirements.txt
```

## 실행 방법

### 1. DER++ Baseline

```bash
./run_baseline.sh
# 또는 GPU 지정:
CUDA_VISIBLE_DEVICES=1 ./run_baseline.sh
```

기대 결과 (10 tasks): **CIL 38.3, TIL 77.1, BWT -53.5**

### 2. DEDUCE 재현 (공식 코드)

```bash
./run_deduce.sh
```

기대 결과 (10 tasks): **CIL 44.4, TIL 79.8, BWT -28.2**
(논문 Table 1: CIL ~39.8, TIL ~78.4)

### 3. Custom 실험

`run_exp.sh <model_name> [stop_after]` 형식으로 호출.

```bash
# Exp A: LUM + gradient projection (DEDUCE에 가장 근접)
./run_exp.sh exp_a_lum_proj
# → CIL 43.9, TIL 80.1, BWT -28.2

# Fisher Multi-task (LUM 대체, weak selectivity)
CUDA_VISIBLE_DEVICES=1 ./run_exp.sh exp_fisher_multitask
# → CIL ~39.6, TIL ~80.2, BWT ~-45.3

# Fisher Multi-task (strong selectivity)
./run_exp.sh exp_fisher_multitask_strong
# → CIL ~37.3, TIL ~77.4, BWT ~-54.4

# 4 태스크까지만 빠르게 (디버그 / ablation):
./run_exp.sh exp_a_lum_proj 4
```

## 실험 결과 요약 (Task 10, seq-cifar100, buffer 500, seed 0)

| 실험 | CIL | TIL | BWT | 비고 |
|------|----:|----:|----:|------|
| Baseline (DER++) | 38.3 | 77.1 | -53.5 | LUM/GUM/EWC 전부 제거 |
| **DEDUCE full** | **44.4** | **79.8** | **-28.2** | 공식 코드 |
| Exp A (LUM+Proj) | 43.9 | 80.1 | -28.2 | LUM + 학습 단계 gradient projection |
| Exp A+Conflict Mask | 43.1 | 79.1 | -31.2 | LUM의 Fisher를 conflict mask로 |
| Exp Cumul Conflict | 39.6 | 80.2 | -45.3 | 누적 conflict score 가중 |
| Fisher Multi (weak) | 39.6* | 80.2* | -45.3* | LUM 대체, c=0.001 |
| Fisher Multi (strong) | 37.3* | 77.4* | -54.4* | LUM 대체, c=0.1 |
| Exp Layerwise Conflict | 37.3 | 77.4 | -54.4 | 레이어별 conflict intensity |

\* 진행 중일 수 있음. 최신 값은 `logs/` 참조.

## 핵심 발견

1. **DEDUCE의 LUM(gradient ascent + Fisher 스케일링)이 핵심 기여**
   - LUM을 다른 메커니즘으로 교체하면 baseline 수준으로 회귀
   - gradient projection (PCGrad), contrastive loss, soft conflict weight, g_old direction
     unlearning 등 모두 실패
2. **EWC penalty는 거의 기여 없음** — Exp A+No EWC가 Exp A와 동일 성능
3. **GUM은 대체 가능** — 뉴런 재초기화 대신 stochastic dropout도 비슷한 효과
4. **Fisher 가중치만으로는 불충분** — gradient ascent의 명시적 "잊기" 동작이 필요

## 모델 파일 추가/수정 사항

- `models/derpp_baseline.py`: 순수 DER++ baseline (신규)
- `models/exp_*.py`: 11개 ablation 모델 (신규)
- `models/__init__.py`: 누락된 의존성 import 실패 시 skip하도록 수정
- `datasets/__init__.py`: 동일
- `utils/conf.py`: `base_path()`를 로컬 경로로, `base_path_img()` 추가
- `utils/main.py`: DataParallel을 단일 GPU에서도 동작하도록 수정
- `utils/training.py`: `gradients`/`unlearn_flag` 변수 초기화 추가, `--stop_after` 지원
- `utils/args.py`: `--stop_after` 인자 추가
- `utils/status.py`: `progress_bar`의 `\r`을 `\n`으로 변경 (로그 파일에 진행률 보이도록)

## 라이선스

공식 DEDUCE 코드는 원 저자의 라이선스를 따른다. 본 저장소의 ablation 코드는 연구/교육 용도.
