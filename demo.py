"""실험 장비 없이 돌려보는 데모.

McLachlan GEM(열전도도)과 Krieger-Dougherty(점도) 물리모델로 "진짜" 배합-물성
관계를 가상으로 만들고, FormulationOptimizer가 ask()/tell() 루프만으로 그
가상의 정답에 수렴해 가는 과정을 그래프로 보여준다.

실행: python demo.py  ->  demo_convergence.png 생성
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from formulation_bo import FormulationOptimizer, Objective, gem_tc, kd_viscosity

SEED = 2026
X_BOUNDS = (0.30, 0.55)
X_MAX = 0.64  # McLachlan GEM/KD가 발산하는 임계 충전율 -> 물리적 상한
N_INIT = 10
N_ITERS = 15


def true_mediator(x: float, rng: np.random.Generator) -> float:
    """가상의 '진짜' 매개변수(예: 필러 분산도). x가 클수록 분산이 나빠진다."""
    return 0.1 + 0.5 * x + rng.normal(0, 0.015)


def true_system(x: float, rng: np.random.Generator):
    """가상의 '진짜' 물리계. GEM/KD 위에 노이즈와 매개변수 효과를 얹는다.

    반환: (mediator, tc, eta) 또는 None(과충전 제작 실패, x>0.52에서 확률적으로 발생).
    """
    if x > 0.52 and rng.uniform() < 0.6:
        return None
    mediator = true_mediator(x, rng)
    tc = float(gem_tc(x)[0]) * (1.15 - 0.5 * mediator) * (1 + rng.normal(0, 0.03))
    eta = float(kd_viscosity(x)[0]) * np.exp(rng.normal(0, 0.08))
    return mediator, tc, eta


def main() -> None:
    rng = np.random.default_rng(SEED)
    opt = FormulationOptimizer(
        bounds=X_BOUNDS,
        objectives={
            "tc": Objective(direction="max", prior=gem_tc, fail_value=0.0),
            "eta": Objective(direction="min", prior=kd_viscosity, transform="log"),
        },
        x_max=X_MAX,
        use_mediator=True,
        seed=SEED,
    )

    # 초기 관측 (그리드 스캔 흉내)
    for x in np.linspace(*X_BOUNDS, N_INIT):
        outcome = true_system(x, rng)
        if outcome is None:
            opt.tell(x, passed=False)
        else:
            mediator, tc, eta = outcome
            opt.tell(x, {"tc": tc, "eta": eta}, mediator=mediator)

    # BO 루프: ask() -> 가상 실험 실행 -> tell()
    history = []
    for i in range(N_ITERS):
        rec = opt.ask(seed=100 + i)
        x_next = rec["recommended_x"]
        outcome = true_system(x_next, rng)
        if outcome is None:
            opt.tell(x_next, passed=False)
            history.append((i + 1, x_next, None, False))
        else:
            mediator, tc, eta = outcome
            opt.tell(x_next, {"tc": tc, "eta": eta}, mediator=mediator)
            history.append((i + 1, x_next, tc, True))

    passed_tc = [tc for _, _, tc, ok in history if ok]
    running_best = np.maximum.accumulate(passed_tc) if passed_tc else np.array([])

    print(f"{N_INIT} initial observations + {N_ITERS} BO-recommended trials done")
    print(f"{'iter':>4} | {'x':>7} {'TC':>7} {'P/F':>3}")
    for i, x, tc, ok in history:
        tc_str = f"{tc:7.3f}" if tc is not None else f"{'--':>7}"
        print(f"{i:>4} | {x:7.4f} {tc_str} {'P' if ok else 'F':>3}")
    if len(running_best):
        print(f"\nBest TC found by BO: {running_best[-1]:.3f}")

    # Left: best-TC-so-far convergence. Right: last ask()'s GP prediction vs the true GEM curve.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(range(1, len(running_best) + 1), running_best, "o-", color="#2563eb")
    ax1.set_xlabel("BO iteration (successful trials only)")
    ax1.set_ylabel("Best TC found so far")
    ax1.set_title("Optimization convergence")
    ax1.grid(alpha=0.3)

    x_grid = np.array(rec["curves"]["x"])
    tc_mean = np.array(rec["curves"]["tc_mean"])
    tc_std = np.array(rec["curves"]["tc_std"])
    ax2.plot(x_grid, gem_tc(x_grid), "--", color="gray", label="True GEM prior (mediator=0 baseline)")
    ax2.plot(x_grid, tc_mean, color="#dc2626", label="GP predicted mean (TC)")
    ax2.fill_between(x_grid, tc_mean - tc_std, tc_mean + tc_std, color="#dc2626", alpha=0.15)
    ax2.axvline(rec["recommended_x"], color="#16a34a", linestyle=":", label="Latest recommended x")
    ax2.set_xlabel("x (formulation variable, e.g. filler volume fraction)")
    ax2.set_ylabel("TC")
    ax2.set_title("Last ask() prediction curve")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path = "demo_convergence.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    main()
