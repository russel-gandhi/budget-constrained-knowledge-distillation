"""Step 4: turn results/runs/*.json into figures and a summary table.

    python make_plots.py
Writes results/figures/*.png and results/summary.md
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results"
FIG = os.path.join(OUT, "figures")
STYLE = {
    "iht_kd":   dict(color="#1f77b4", marker="o", label="KD + IHT (hard budget)"),
    "l1_kd":    dict(color="#d62728", marker="s", label=r"KD + $\ell_1$ (soft-threshold)"),
    "group_kd": dict(color="#2ca02c", marker="^", label="KD + group lasso (neurons)"),
    "prune_kd": dict(color="#9467bd", marker="v", label="KD, then one-shot magnitude pruning"),
    "iht_ce":   dict(color="#7f7f7f", marker="o", ls="--", label="labels only + IHT (no KD)"),
}


def load():
    runs = {}
    for p in glob.glob(os.path.join(OUT, "runs", "*.json")):
        with open(p) as f:
            r = json.load(f)
        if r.get("diverged"):
            print(f"note: {r['name']} diverged and is left out of the plots")
            continue
        runs[r["name"]] = r
    return runs


def by_method(runs):
    """Seed-0 runs grouped by method (extra seeds are only used in kd_gain)."""
    g = defaultdict(list)
    for r in runs.values():
        if r.get("seed", 0) == 0:
            g[r["method"]].append(r)
    for m in g:
        g[m].sort(key=lambda r: r["size"]["nnz_weights"])
    return g


def baselines(ax, runs):
    if "dense_kd" in runs:
        ax.axhline(100 * runs["dense_kd"]["test_acc"], color="k", lw=1, ls="-", label="dense student, KD")
    if "dense_ce" in runs:
        ax.axhline(100 * runs["dense_ce"]["test_acc"], color="k", lw=1, ls=":", label="dense student, labels only")
    t = next(iter(runs.values()))["teacher_test_acc"]
    ax.axhline(100 * t, color="#ff7f0e", lw=1, ls="-.", label="teacher (1.9M params)")


def pareto(runs, g, xkey, fname, xlabel, title):
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for m, st in STYLE.items():
        if m in g:
            xs = [xkey(r) for r in g[m]]
            ys = [100 * r["test_acc"] for r in g[m]]
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            ax.plot([xs[i] for i in order], [ys[i] for i in order], ms=5, **st)
    baselines(ax, runs)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    lo = min(100 * r["test_acc"] for m in STYLE if m in g for r in g[m])
    ax.set_ylim(max(lo - 1, 50), None)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=160)
    plt.close(fig)


def kd_gain(runs):
    """KD minus label-only test accuracy at each budget, mean +- std over seeds
    (the two runs of a pair share the seed, i.e. the same initial weights)."""
    import numpy as np
    acc = defaultdict(dict)                      # (budget) -> {(method, seed): acc}
    for r in runs.values():
        if r["method"] in ("iht_kd", "iht_ce"):
            acc[r["budget"]][(r["method"], r.get("seed", 0))] = r["test_acc"]
        elif r["method"] in ("dense_kd", "dense_ce"):
            acc[1.0][(r["method"].replace("dense", "iht"), r.get("seed", 0))] = r["test_acc"]
    bs, mean, std, ns, rows = [], [], [], [], []
    for b in sorted(acc, reverse=True):
        seeds = sorted({s for (_, s) in acc[b]})
        d = [100 * (acc[b][("iht_kd", s)] - acc[b][("iht_ce", s)]) for s in seeds
             if ("iht_kd", s) in acc[b] and ("iht_ce", s) in acc[b]]
        if d:
            bs.append(b); mean.append(np.mean(d)); std.append(np.std(d)); ns.append(len(d))
            kd = [100 * acc[b][("iht_kd", s)] for s in seeds if ("iht_kd", s) in acc[b]]
            ce = [100 * acc[b][("iht_ce", s)] for s in seeds if ("iht_ce", s) in acc[b]]
            rows.append((b, np.mean(kd), np.std(kd), np.mean(ce), np.std(ce), np.mean(d), np.std(d), len(d)))
    if not bs:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels = [("dense" if b == 1.0 else f"{100 * b:g}%") + f"\n(n={n})" for b, n in zip(bs, ns)]
    ax.bar(labels, mean, yerr=[s if n > 1 else 0 for s, n in zip(std, ns)], capsize=4,
           color=["#1f77b4" if n > 1 else "#9ecae1" for n in ns])
    for i, (v, s) in enumerate(zip(mean, std)):
        ax.text(i, v + (s if v >= 0 else -s), f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("budget (fraction of student weights kept)")
    ax.set_ylabel("accuracy gain from KD (pp)")
    ax.set_title("KD vs. label-only, same size and solver (mean ± std over n seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "kd_gain.png"), dpi=160)
    plt.close(fig)
    return rows


def finetune_effect(g):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in ["iht_kd", "l1_kd", "group_kd", "prune_kd"]:
        if m not in g:
            continue
        st = STYLE[m]
        xs = [r["size"]["nnz_weights"] for r in g[m]]
        ax.plot(xs, [100 * r["pre_ft_test_acc"] for r in g[m]], ls=":", marker=st["marker"],
                color=st["color"], ms=4, alpha=0.7)
        ax.plot(xs, [100 * r["test_acc"] for r in g[m]], ls="-", marker=st["marker"],
                color=st["color"], ms=4, label=st["label"])
    ax.set_xscale("log")
    ax.set_xlabel("non-zero weights")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Before (dotted) vs. after (solid) masked fine-tuning")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5)
    ax.set_ylim(15, None)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "finetune_effect.png"), dpi=160)
    plt.close(fig)


def ablations(runs):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, prefix, key, xl in [(axes[0], "abl_T", "T", "temperature T  (alpha = 0.9)"),
                                (axes[1], "abl_alpha", "alpha", "alpha  (T = 4)")]:
        rs = sorted([r for r in runs.values() if r["method"] == prefix], key=lambda r: r[key])
        if not rs:
            continue
        xs = [r[key] for r in rs]
        ax.plot(xs, [100 * r["val_acc"] for r in rs], "o-", label="validation")
        ax.plot(xs, [100 * r["test_acc"] for r in rs], "s--", label="test")
        if prefix == "abl_alpha":
            ce = [r for r in runs.values() if r["method"] == "iht_ce" and r.get("budget") == rs[0].get("budget")]
            if ce:
                ax.plot([0.0], [100 * ce[0]["test_acc"]], "kx", ms=8, label="alpha = 0 (labels only), test")
        if prefix == "abl_T":
            ax.set_xscale("log", base=2)
        ax.set_xlabel(xl)
        ax.set_ylabel("accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    b = next((r.get("budget") for r in runs.values() if r["method"] == "abl_T"), None)
    fig.suptitle(f"Distillation hyper-parameters at a {100 * b:g}% budget (KD + IHT)" if b else "")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ablation_T_alpha.png"), dpi=160)
    plt.close(fig)


def curves(runs):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axes[0]
    for name, lab in [("abl_l1_beta0", r"$\ell_1$ 2e-3, $\beta$=0 (prox-SGD)"),
                      ("abl_l1_beta0.9", r"$\ell_1$ 2e-3, $\beta$=0.9 (accelerated)"),
                      ("l1_kd_0.004", r"$\ell_1$ 4e-3, with line search"),
                      ("abl_l1_nobacktrack", r"$\ell_1$ 4e-3, no line search"),
                      ("abl_group_nobacktrack", "group 2e-3, no line search")]:
        if name in runs:
            h = [e for e in runs[name]["history"] if "objective" in e]
            ax.plot([e["epoch"] for e in h], [e["objective"] for e in h], "o-", ms=3, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("F(w) + h(w)  (train)")
    ax.set_title("Objective: momentum and line search")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, lab in [("abl_l1_beta0", r"$\ell_1$, $\beta$=0"), ("abl_l1_beta0.9", r"$\ell_1$, $\beta$=0.9"),
                      ("iht_kd_0.02", "IHT 2% (gradual budget)"), ("abl_iht_noramp", "IHT 2% (fixed from step 0)")]:
        if name in runs:
            h = runs[name]["history"]
            ax.plot([e["epoch"] for e in h], [100 * e["density"] for e in h], "o-", ms=3, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("non-zero weights (%)")
    ax.set_title("Sparsity during training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for name, lab in [("abl_l1_beta0", r"$\ell_1$, $\beta$=0"), ("abl_l1_beta0.9", r"$\ell_1$, $\beta$=0.9"),
                      ("iht_kd_0.02", "IHT 2% (gradual)"), ("abl_iht_noramp", "IHT 2% (fixed)"),
                      ("abl_iht_beta0", r"IHT 2%, $\beta$=0")]:
        if name in runs:
            h = runs[name]["history"] + runs[name].get("ft_history", [])
            ax.plot(range(1, len(h) + 1), [100 * e["val_acc"] for e in h], "-", label=lab)
    ax.set_xlabel("epoch (incl. fine-tuning)")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("Validation accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "training_curves.png"), dpi=160)
    plt.close(fig)


def layers(g):
    """Where does each method put its budget? Density per layer at ~2-5%."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    picks = []
    for m in ["iht_kd", "prune_kd", "l1_kd", "group_kd"]:
        if m in g:
            r = min(g[m], key=lambda r: abs(r["size"]["density"] - 0.03))
            picks.append((m, r))
    w = 0.8 / max(1, len(picks))
    names = ["fc1 (784x300)", "fc2 (300x100)", "fc3 (100x10)"]
    for i, (m, r) in enumerate(picks):
        dens = [100 * n / s for n, s in zip(r["size"]["layer_nnz"], r["size"]["layer_size"])]
        ax.bar([j + i * w for j in range(3)], dens, w, color=STYLE[m]["color"],
               label=f"{STYLE[m]['label']} ({100 * r['size']['density']:.1f}% overall)")
    ax.set_xticks([j + w * (len(picks) - 1) / 2 for j in range(3)])
    ax.set_xticklabels(names)
    ax.set_yscale("log")
    ax.set_ylabel("non-zero weights in layer (%)")
    ax.set_title("How each method distributes its budget across layers")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "layer_density.png"), dpi=160)
    plt.close(fig)


def neurons(g):
    if "group_kd" not in g:
        return
    rs = sorted(g["group_kd"], key=lambda r: r["lam"])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    lams = [r["lam"] for r in rs]
    ax.plot(lams, [r["structured"]["neurons_alive"][0] for r in rs], "o-", label="hidden layer 1 (of 300)")
    ax.plot(lams, [r["structured"]["neurons_alive"][1] for r in rs], "s-", label="hidden layer 2 (of 100)")
    ax.plot(lams, [r["structured"]["inputs_alive"] for r in rs], "^--", label="input pixels used (of 784)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"group-lasso $\lambda$")
    ax.set_ylabel("count")
    ax.set_title("Group lasso deletes whole neurons")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "group_neurons.png"), dpi=160)
    plt.close(fig)


def summary(runs, gain_rows):
    order = ["dense_ce", "dense_kd", "iht_kd", "iht_ce", "prune_kd", "l1_kd", "group_kd",
             "abl_T", "abl_alpha", "abl_beta", "abl_ramp"]
    rows = sorted(runs.values(), key=lambda r: (order.index(r["method"]) if r["method"] in order else 99,
                                                -r["size"]["nnz_weights"]))
    t = next(iter(runs.values()))["teacher_test_acc"]
    lines = [f"Teacher test accuracy: {100 * t:.2f}%\n",
             "| run | test acc % | pre-FT % | val % | non-zero weights | density % | sparse KB | "
             "neurons (h1, h2) | inputs used | compact dense params |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s, st = r["size"], r["structured"]
        lines.append(
            f"| {r['name']} | {100 * r['test_acc']:.2f} | {100 * r['pre_ft_test_acc']:.2f} | "
            f"{100 * r['val_acc']:.2f} | {s['nnz_weights']:,} | {100 * s['density']:.2f} | "
            f"{s['sparse_kb']:.1f} | {st['neurons_alive'][0]}, {st['neurons_alive'][1]} | "
            f"{st['inputs_alive']} | {st['compact_params']:,} |")
    if gain_rows:
        lines += ["", "KD vs. labels only (IHT), mean ± std over seeds:", "",
                  "| budget | KD test % | labels-only test % | gain (pp) | seeds |", "|---|---|---|---|---|"]
        for b, km, ks, cm, cs, dm, ds, n in gain_rows:
            lines.append(f"| {'dense' if b == 1.0 else f'{100 * b:g}%'} | {km:.2f} ± {ks:.2f} | "
                         f"{cm:.2f} ± {cs:.2f} | {dm:+.2f} ± {ds:.2f} | {n} |")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(FIG, exist_ok=True)
    runs = load()
    g = by_method(runs)
    pareto(runs, g, lambda r: r["size"]["nnz_weights"], "pareto_nnz.png",
           "non-zero weights (log scale)  —  dense student = 266,200",
           "Accuracy vs. size (unstructured sparsity)")
    pareto(runs, g, lambda r: r["structured"]["compact_params"], "pareto_structured.png",
           "parameters of the equivalent compact dense network (log scale)",
           "Accuracy vs. size after physically removing dead neurons/inputs")
    gain_rows = kd_gain(runs)
    finetune_effect(g)
    ablations(runs)
    curves(runs)
    layers(g)
    neurons(g)
    summary(runs, gain_rows)
    print(f"{len(runs)} runs -> {FIG}/ and {OUT}/summary.md")


if __name__ == "__main__":
    main()
