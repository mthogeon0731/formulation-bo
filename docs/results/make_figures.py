"""docs/results/figures/*.png 생성. analyze.py를 먼저 실행해 data/*.csv·json을 만든다.

    python docs/results/analyze.py
    python docs/results/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from formulation_bo import gem_tc, kd_viscosity  # noqa: E402

DATA, FIG = HERE / "data", HERE / "figures"
FIG.mkdir(exist_ok=True)

# 실험군 빨강 · 대조군 회색 · 초기점 옅은 회색 · 측정 후 예측 검정
EXP, CTL, INIT, INK, MUTED, GRID = "#D42A1E", "#6B6B6B", "#B5B5B5", "#161616", "#5A5A5A", "#E6E6E6"

for name in ("Segoe UI", "Helvetica Neue", "Arial"):
    if any(name == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = [name, "DejaVu Sans"]
        break
plt.rcParams.update({
    "axes.unicode_minus": False, "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.labelcolor": INK, "axes.edgecolor": "#9A9A9A", "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 200,
})
K_LABEL = r"$k_\mathrm{app}$ (W·m$^{-1}$·K$^{-1}$)"
ETA_LABEL = r"Viscosity $\eta$ (Pa·s)"
DCV = r"$D_\mathrm{CV}$"


def load():
    df = pd.read_csv(DATA / "tim_vlab_results.csv")
    df["passed"] = df["passed"].astype(str).str.lower().eq("true")
    m = json.loads((DATA / "summary_metrics.json").read_text())
    return df, m


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("saved", name)


def front(points: pd.DataFrame) -> pd.DataFrame:
    x, y = points.eta.to_numpy(), points.tc.to_numpy()
    keep = [not any(j != i and x[j] <= x[i] and y[j] >= y[i] and (x[j] < x[i] or y[j] > y[i])
                    for j in range(len(x))) for i in range(len(x))]
    return points[keep].sort_values("eta")


# ------------------------------------------------------------------ fig1 Pareto
def fig_pareto(df, m):
    p = df[df.passed]
    init, exp, ctl = p[p.arm == "initial"], p[p.arm == "experimental"], p[p.arm == "control"]
    hv = m["hypervolume"]
    fig, ax = plt.subplots(figsize=(8, 5))
    dominated = init[~init.index.isin(set(front(pd.concat([init, exp])).index) | set(front(pd.concat([init, ctl])).index))]
    ax.scatter(dominated.eta, dominated.tc, s=36, facecolors="none", edgecolors=INIT, lw=1.2, zorder=2)
    ax.scatter(init[~init.index.isin(dominated.index)].eta, init[~init.index.isin(dominated.index)].tc,
               s=36, facecolors="white", edgecolors=MUTED, lw=1.2, zorder=3, label="Initial 15 (shared)")
    for pts, color, ls, label in ((ctl, CTL, (0, (4, 3)), f"Control front · HV {hv['hv_control']:.3f}"),
                                  (exp, EXP, "-", f"Experimental front · HV {hv['hv_experimental']:.3f}")):
        fr = front(pd.concat([init, pts]))
        ax.step(fr.eta, fr.tc, where="post", color=color, lw=2, ls=ls, zorder=4, label=label)
    ax.scatter(ctl.eta, ctl.tc, marker="s", s=46, color=CTL, edgecolors="white", lw=1, zorder=5,
               label=f"Control (φ only, {len(ctl)} passed)")
    ax.scatter(exp.eta, exp.tc, marker="^", s=62, color=EXP, edgecolors="white", lw=1, zorder=6,
               label=f"Experimental (uses {DCV}, {len(exp)} passed)")

    e8 = exp[exp.iteration == 8].iloc[0]
    i15 = init[init.iteration == 15].iloc[0]
    e10 = exp[exp.iteration == 10].iloc[0]
    ax.scatter([e8.eta], [e8.tc], s=260, facecolors="none", edgecolors=EXP, lw=1.6, zorder=7)
    ax.scatter([i15.eta], [i15.tc], s=260, facecolors="none", edgecolors=INK, lw=1.2, ls=(0, (2, 2)), zorder=7)
    ax.annotate(f"Experimental round 8 (φ {e8.phi:.3f})\n{e8.tc:.3f} @ {e8.eta:.1f} Pa·s", (e8.eta, e8.tc),
                xytext=(-190, 22), textcoords="offset points", color=EXP, fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=EXP, lw=0.8))
    ax.annotate(f"Best initial run (φ {i15.phi:.3f})\n{i15.tc:.3f} @ {i15.eta:.1f} Pa·s", (i15.eta, i15.tc),
                xytext=(14, 26), textcoords="offset points", color=INK, fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.annotate(f"round 10\n{e10.tc:.3f} @ {e10.eta:.0f} Pa·s", (e10.eta, e10.tc), xytext=(6, -10),
                textcoords="offset points", color=MUTED, fontsize=9, ha="right", va="top")
    ax.annotate("", xy=(e8.eta * 1.08, 1.27), xytext=(i15.eta * 0.93, 1.27),
                arrowprops=dict(arrowstyle="->", color=EXP, lw=1.4))
    ax.text(np.sqrt(e8.eta * i15.eta), 1.285, "same $k_\\mathrm{app}$, η −48%", ha="center", color=EXP, fontsize=9.5)

    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 200])
    ax.set_xticklabels(["10", "20", "50", "100", "200"])
    ax.set_ylim(0.38, 1.37)
    ax.set_xlabel(ETA_LABEL + "  · log scale, ← better")
    ax.set_ylabel(K_LABEL + "  · ↑ better")
    ax.set_title("Pareto front: 15 shared initial runs + 10 BO rounds per arm (single campaign)", loc="left")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "fig1_pareto_front.png")


# ------------------------------------------------------------------ fig2 누적 HV
def fig_hv(df, m):
    c = pd.read_csv(DATA / "hypervolume_by_iteration.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.step(c.iteration, c.hv_control, where="post", color=CTL, lw=2, ls=(0, (4, 3)), label="Control (φ only)")
    ax.step(c.iteration, c.hv_experimental, where="post", color=EXP, lw=2.2, label=f"Experimental (uses {DCV})")
    ax.step(c.iteration, c.hv_experimental_excl_iter8, where="post", color=EXP, lw=1.2, ls=":",
            alpha=0.9, label="Experimental, round 8 removed")
    ax.scatter(c.iteration, c.hv_control, marker="s", s=22, color=CTL, zorder=3)
    ax.scatter(c.iteration, c.hv_experimental, marker="^", s=30, color=EXP, zorder=4)
    last = c.iloc[-1]
    for v, col, txt in ((last.hv_experimental, EXP, f"{last.hv_experimental:.3f}"),
                        (last.hv_control, CTL, f"{last.hv_control:.3f}"),
                        (last.hv_experimental_excl_iter8, EXP, f"{last.hv_experimental_excl_iter8:.3f} (round 8 removed)")):
        ax.text(10.25, v, txt, color=col, va="center", fontsize=9.5)
    ax.annotate("round 8 alone: +0.045", xy=(8, 0.7859), xytext=(5.0, 0.778), color=EXP, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=EXP, lw=0.9))
    ax.text(0.0, 0.7262, f"round 0 = shared initial 15 ({c.hv_experimental[0]:.3f})", color=MUTED, fontsize=9, va="bottom")
    ax.set_xlim(-0.3, 13.2)
    ax.set_xticks(range(0, 11))
    ax.set_ylim(0.724, 0.795)
    ax.set_xlabel("BO round")
    ax.set_ylabel("Cumulative HV (initial + arm)")
    ax.set_title("Cumulative Pareto hypervolume by round", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "fig2_hypervolume_by_iteration.png")


# ------------------------------------------------------------------ fig3 parity
def fig_parity(df, m):
    o = pd.read_csv(DATA / "cv_out_of_fold.csv")
    cv = m["cross_validation"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), sharex=True, sharey=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("dcv", ["#F6C9C4", "#7A120B"])
    norm = matplotlib.colors.Normalize(o.d_cv.min(), o.d_cv.max())
    panels = (("pred_phi_only", f"φ only (no {DCV})", cv["k_r2_without_split_mean"], cv["k_rmse_without"]),
              ("pred_phi_dcv", f"φ + measured {DCV}", cv["k_r2_with_split_mean"], cv["k_rmse_with"]))
    for ax, (col, title, r2, rmse) in zip(axes, panels):
        ax.plot([0.35, 1.6], [0.35, 1.6], color="#9A9A9A", lw=1, ls=(0, (3, 3)), zorder=1)
        for arm, mk, s in (("initial", "o", 44), ("experimental", "^", 60)):
            q = o[o.arm == arm]
            sc = ax.scatter(q.tc, q[col], c=q.d_cv, cmap=cmap, norm=norm, marker=mk, s=s,
                            edgecolors=INK, linewidths=0.5, zorder=3)
        ax.set_title(title, loc="left", fontsize=11)
        ax.text(0.04, 0.95, f"R² {r2:.2f}\nRMSE {rmse:.3f}", transform=ax.transAxes, va="top",
                fontsize=10.5, color=INK, fontweight="bold")
        ax.set_xlim(0.35, 1.6)
        ax.set_ylim(0.35, 1.6)
        ax.set_aspect("equal")
        ax.set_xlabel("Measured " + K_LABEL)
    axes[0].set_ylabel("Cross-validated " + K_LABEL)
    hi = o.sort_values("d_cv").tail(2)
    for (_, r), dxy in zip(hi.iterrows(), ((10, -4), (14, -28))):
        axes[0].annotate(f"$D_\\mathrm{{CV}}$ {r.d_cv:.2f}", (r.tc, r.pred_phi_only), xytext=dxy, textcoords="offset points",
                         arrowprops=dict(arrowstyle="-", color="#7A120B", lw=0.7, shrinkB=4),
                         fontsize=8.5, color="#7A120B")
    cb = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label(f"Measured {DCV} (↑ more agglomerated)")
    cb.outline.set_visible(False)
    axes[0].scatter([], [], marker="o", color="#CCCCCC", edgecolors=INK, lw=0.5, label="Initial 15")
    axes[0].scatter([], [], marker="^", color="#CCCCCC", edgecolors=INK, lw=0.5, label="Experimental 10")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.canvas.draw()
    top = axes[0].get_position().y1
    fig.suptitle(f"Cross-validation after measurement: same {cv['n']} runs, 5-fold × 20 split seeds (mean prediction shown)",
                 x=axes[0].get_position().x0 - 0.05, ha="left", fontsize=11.5, fontweight="bold", y=top + 0.13)
    save(fig, "fig3_cv_parity.png")


# ------------------------------------------------------------------ fig4 φ만 모델 잔차 vs D_CV
def fig_residual(df, m):
    o = pd.read_csv(DATA / "cv_out_of_fold.csv")
    cv = m["cross_validation"]
    res = o.tc - o.pred_phi_only
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhline(0, color="#9A9A9A", lw=1, ls=(0, (3, 3)))
    b = np.polyfit(o.d_cv, res, 1)
    xs = np.linspace(o.d_cv.min(), o.d_cv.max(), 50)
    ax.plot(xs, np.polyval(b, xs), color=INK, lw=1.3, alpha=0.7)
    for arm, mk, s, fc in (("initial", "o", 42, "white"), ("experimental", "^", 58, EXP)):
        q = o[o.arm == arm]
        ax.scatter(q.d_cv, (q.tc - q.pred_phi_only), marker=mk, s=s, facecolors=fc,
                   edgecolors=MUTED if arm == "initial" else "white", lw=1.1, zorder=3,
                   label="Initial 15" if arm == "initial" else "Experimental 10")
    ax.text(0.97, 0.95, f"r = {cv['r_phi_only_residual_vs_dcv']:.2f} (n = {cv['n']})".replace("-", "−"),
            transform=ax.transAxes, ha="right", va="top", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel(f"Measured {DCV} (↑ more agglomerated)")
    ax.set_ylabel("Measured − φ-only prediction (W·m$^{-1}$·K$^{-1}$)")
    ax.set_title(f"The φ-only model's error tracks {DCV}", loc="left", fontsize=11.5)
    ax.text(0.02, 0.03, "negative = over-predicted · predictions averaged over 20 CV splits", transform=ax.transAxes,
            fontsize=8.5, color=MUTED)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.08), fontsize=9)
    save(fig, "fig4_residual_vs_dcv.png")


# ------------------------------------------------------------------ fig5 추천 시점 사전 예측
def fig_prospective(df, m):
    p = pd.read_csv(DATA / "prospective_predictions.csv")
    s = m["prospective_seed0"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    off = 0.2
    for i, r in p.iterrows():
        ax.plot([r.iteration - 0.38, r.iteration + 0.38], [r.tc, r.tc], color=INK, lw=2, zorder=2,
                label="Measured" if i == 0 else None)
    specs = (("pred_phi_only", "sd_phi_only", -off, "s", CTL, f"Before mixing · φ only (±2σ) — MAE {s['mae_phi_only']:.3f}"),
             ("pred_phi_dhat", "sd_phi_dhat", 0.0, "o", EXP,
              f"Before mixing · φ + predicted $\\hat{{D}}_\\mathrm{{CV}}$ (±2σ) — MAE {s['mae_phi_dhat']:.3f}"),
             ("pred_phi_dmeas", None, off, "D", INK, f"After measuring · φ + measured {DCV} — MAE {s['mae_phi_dmeas']:.3f}"))
    for col, sd, dx, mk, color, label in specs:
        x = p.iteration + dx
        if sd:
            ax.errorbar(x, p[col], yerr=2 * p[sd], fmt=mk, ms=6.5, color=color, ecolor=color, elinewidth=1.2,
                        capsize=2.5, mfc=color if mk != "o" else "white", mew=1.4, zorder=4, label=label)
            miss = (p.tc - p[col]).abs() > 2 * p[sd]
            ax.scatter(x[miss], p[col][miss], s=190, facecolors="none", edgecolors=color, lw=1,
                       ls=(0, (2, 2)), zorder=3)
        else:
            ax.scatter(x, p[col], marker=mk, s=34, color=color, zorder=5, label=label)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Experimental-arm round (each predicted from the data available before it)")
    ax.set_ylabel(K_LABEL)
    ax.set_title(f"Prediction before mixing, same 10 rounds (inside ±2σ: {s['cover2s_phi_only']}/10 → {s['cover2s_phi_dhat']}/10, "
                 f"Wilcoxon p = {s['wilcoxon_p']:.2f})", loc="left", fontsize=11)
    ax.text(0.99, 0.02, "dotted circle = measurement outside ±2σ", transform=ax.transAxes, ha="right", fontsize=8.5, color=MUTED)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0.45, 1.9)
    save(fig, "fig5_prospective_prediction.png")


# ------------------------------------------------------------------ fig6 물리 prior vs 실측
def fig_prior(df, m):
    p = df[df.passed]
    phi = np.linspace(0.30, 0.56, 200)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    marks = (("initial", "o", "white", MUTED, "Initial"), ("experimental", "^", EXP, "white", "Experimental"),
             ("control", "s", CTL, "white", "Control"))
    ax = axes[0]
    ax.plot(phi, gem_tc(phi), color=INK, lw=1.6, label="GEM prior (McLachlan)")
    for arm, mk, fc, ec, lab in marks:
        q = p[p.arm == arm]
        ax.scatter(q.phi, q.tc, marker=mk, s=40, facecolors=fc, edgecolors=ec, lw=1.1, zorder=3, label=lab)
    i15 = p[(p.arm == "initial") & (p.iteration == 15)].iloc[0]
    g = float(gem_tc(i15.phi)[0])
    ax.plot([i15.phi, i15.phi], [i15.tc, g], color=EXP, lw=1.6)
    ax.annotate(f"δ = {i15.tc:.3f} − {g:.2f}", (i15.phi, (i15.tc + g) / 2), xytext=(-92, 10),
                textcoords="offset points", color=EXP, fontsize=9)
    ax.set_xlabel("Filler volume fraction φ")
    ax.set_ylabel(K_LABEL)
    ax.set_title("Thermal conductivity: GEM prior + GP residual", loc="left", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    ax = axes[1]
    ax.plot(phi, kd_viscosity(phi), color=INK, lw=1.6, label="KD prior (Krieger–Dougherty)")
    for arm, mk, fc, ec, lab in marks:
        q = p[p.arm == arm]
        ax.scatter(q.phi, q.eta, marker=mk, s=40, facecolors=fc, edgecolors=ec, lw=1.1, zorder=3, label=lab)
    ax.set_yscale("log")
    ax.set_yticks([10, 20, 50, 100, 200])
    ax.set_yticklabels(["10", "20", "50", "100", "200"])
    ax.set_xlabel("Filler volume fraction φ")
    ax.set_ylabel(ETA_LABEL + " · log scale")
    ax.set_title("Viscosity: KD prior + GP log-residual", loc="left", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    for a in axes:
        a.axvspan(0.55, 0.56, color="#EFEFEF", zorder=0)
    fig.text(0.01, -0.03, f"Prior parameters = library defaults (k_m 0.20, k_f 30, φ_c 0.64, t 1.5 · η_m 5 Pa·s, [η] 2.5, φ_m 0.64). "
             f"{len(p)} passed runs. Grey band = beyond the 0.55 search limit.", fontsize=8.5, color=MUTED)
    save(fig, "fig6_physics_prior.png")


# ------------------------------------------------------------------ fig7 추천 φ 궤적
def fig_trajectory(df, m):
    fig, ax = plt.subplots(figsize=(8, 4))
    for arm, color, mk, ls, lab in (("control", CTL, "s", (0, (4, 3)), "Control"), ("experimental", EXP, "^", "-", "Experimental")):
        q = df[df.arm == arm].sort_values("iteration")
        ax.plot(q.iteration, q.phi, color=color, lw=1.8, ls=ls, zorder=2)
        qp = q[q.passed]
        ax.scatter(qp.iteration, qp.phi, marker=mk, s=50, color=color, edgecolors="white", lw=1, zorder=3, label=lab)
        qf = q[~q.passed]
        if len(qf):
            ax.scatter(qf.iteration, qf.phi, marker="X", s=90, color=color, zorder=4, label="Control fail (would not spread)")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("BO round")
    ax.set_ylabel("Recommended φ")
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(range(1, 11))
    sec.set_xticklabels([f"{0.05 + 0.9 * (t - 1) / 9:.2f}" for t in range(1, 11)], fontsize=8.5)
    sec.set_xlabel("ParEGO weight $\\lambda_1$ on conductivity (viscosity first → conductivity first)", fontsize=9.5, color=MUTED)
    sec.tick_params(colors=MUTED)
    ax.set_title("Recommended φ by round", loc="left", fontsize=11.5, pad=34)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "fig7_phi_trajectory.png")


def main():
    df, m = load()
    fig_pareto(df, m)
    fig_hv(df, m)
    fig_parity(df, m)
    fig_residual(df, m)
    fig_prospective(df, m)
    fig_prior(df, m)
    fig_trajectory(df, m)


if __name__ == "__main__":
    main()
