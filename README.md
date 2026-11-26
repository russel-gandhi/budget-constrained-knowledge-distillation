# Budget-Constrained Knowledge Distillation: Compressing Neural Networks via Sparse Optimisation

**Team:** SirCook

**Members:**
- Russel Gandhi (BA2025041)
- Pratyush Prasad (BA2025034)
- Naman Sharma (BA2025025)

## Short description

A large "teacher" CNN (1.87M parameters) is compressed into a small "student" MLP
(784-300-100-10, 266k parameters) by **knowledge distillation**: the student is
trained to match the teacher's softened output probabilities. On top of that we
impose a **size budget** on the student and solve the resulting problem

    min_w  F_KD(w) + h(w)

with **proximal gradient methods**, where h(w) is one of

| method | h(w) | proximal operator |
|---|---|---|
| KD + l1 | `lam * ||w||_1` (convex relaxation of the budget) | soft-thresholding |
| KD + IHT | indicator of `{ ||w||_0 <= k }` (exact budget) | keep top-k magnitudes (hard-thresholding) |
| KD + group lasso | `lam * sum_g sqrt(|g|) ||w_g||_2`, g = one neuron | block soft-thresholding (deletes neurons) |

and compare them with label-only training, a dense distilled student and
one-shot magnitude pruning, analysing the **accuracy-versus-size trade-off**.
A convex special case (linear student) is used to verify the solvers against
cvxpy and against the O(1/k) / O(1/k^2) convergence rates of ISTA / FISTA.

Dataset: Fashion-MNIST (downloaded automatically by torchvision).

## Key results (Fashion-MNIST, test accuracy)

- Teacher CNN: 93.04%. Dense student: 89.7%.
- **KD + IHT keeps 1% of the student's weights (2,662 non-zeros) at 87.7%**: 100x fewer non-zeros for a 2-point drop. At 5% the drop is 0.5 points.
- Optimising under the budget beats prune-after-training, and the gap grows as the budget shrinks: +4.6 points at 1%.
- Counting real (dense) network size instead, l1 and group lasso win: a 6 + 9 neuron network (4,873 params, 55x smaller) reaches 82.9%.
- The convex check: FISTA matches cvxpy to a 1.7e-6 objective gap and a 99.4% support overlap, and ISTA / FISTA follow their O(1/k) / O(1/k^2) bounds.
- Distillation itself gives only +0.1 to +0.3 points at moderate budgets (about 1 std over 3 seeds) and none at 1 to 2%.

See `report/report.pdf` for the full report, and `results/summary.md` and `results/figures/` for all numbers and plots.

## Libraries used

- **PyTorch / torchvision**: networks, automatic differentiation, dataset download.
  The optimisation loop, extrapolation (momentum), step-size schedule, backtracking
  line search and all proximal operators are implemented by hand in `src/`
  (no `torch.optim` for the students).
- **NumPy**: convex check (hand-derived gradients, ISTA, FISTA, backtracking).
- **cvxpy + Clarabel**: reference solution of the convex problem.
- **Matplotlib**: figures.

## Setup

Python 3.10+ is required. A CUDA GPU is optional; the code runs on the CPU too (slower).

```bash
pip install -r requirements.txt
```

For a GPU, install the CUDA build of PyTorch from https://pytorch.org first.

## How to run (in this order)

```bash
python train_teacher.py                # 1. train the teacher CNN           (~3 min on GPU)
python convex_check.py                 # 2. ISTA / FISTA vs cvxpy (convex)  (~2 min, CPU)
python run_experiments.py --suite all  # 3. all methods, ablations, seeds   (~1-1.5 h on GPU)
python make_plots.py                   # 4. figures + summary table
python -m tests.test_prox              # optional: unit tests of the proximal operators
python demo.py                         # optional: one live KD + IHT run at a 2% budget (~30 s on GPU, needs step 1)
```

`run_experiments.py` skips runs whose results already exist, so it can be
interrupted and resumed. Use `--overwrite` to redo them. Run
`python run_experiments.py -h` to list all options (epochs, step size, T, alpha, budgets, lambdas).

## Project layout

```
src/prox.py        proximal operators (soft / hard / block soft thresholding)
src/losses.py      distillation objective F(w)
src/solvers.py     stochastic accelerated proximal gradient + the h(w) terms
src/sparsity.py    size measurement, removal of dead neurons
src/models.py      teacher CNN and student MLP
src/data.py        dataset loading, mini-batching, augmentation
train_teacher.py   step 1
convex_check.py    step 2
run_experiments.py step 3
make_plots.py      step 4
tests/test_prox.py checks that every prox really is the argmin it should be
results/           outputs: runs/*.json, figures/*.png, summary.md, convex/
```
