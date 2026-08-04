"""FormulationOptimizer 자기 점검. pytest 없이 `python tests/test_optimizer.py`로 실행."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from formulation_bo import FormulationOptimizer, Objective, PhysicalLimitError, gem_tc, kd_viscosity

results = []


def check(name, fn):
    try:
        fn()
        results.append((name, "PASS", ""))
    except AssertionError as e:
        results.append((name, "FAIL", str(e)))
    except Exception as e:
        results.append((name, "ERROR", f"{type(e).__name__}: {e}"))


def make_direct_opt(seed=0):
    opt = FormulationOptimizer(
        bounds=(0.30, 0.55),
        objectives={
            "tc": Objective(direction="max", prior=gem_tc, fail_value=0.0),
            "eta": Objective(direction="min", prior=kd_viscosity, transform="log"),
        },
        x_max=0.64,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    for x in np.linspace(0.30, 0.55, 8):
        opt.tell(
            x,
            {"tc": float(gem_tc(x)[0]) * (1 + rng.normal(0, 0.03)), "eta": float(kd_viscosity(x)[0])},
        )
    return opt


def t_failfast_on_physical_limit():
    opt = make_direct_opt()
    try:
        opt.tell(0.64, {"tc": 1.0, "eta": 1.0})
        raise AssertionError("x_max 초과가 조용히 통과됨 (클리핑 회귀)")
    except PhysicalLimitError:
        pass
check("1. x_max 초과 시 PhysicalLimitError (클리핑 아님)", t_failfast_on_physical_limit)


def t_ask_returns_within_bounds():
    opt = make_direct_opt()
    r = opt.ask()
    assert 0.30 <= r["recommended_x"] <= 0.55, r["recommended_x"]
    assert set(r["predictions"]) == {"tc", "eta"}
check("2. ask() 결과가 bounds 안에 있고 목적함수별 예측 포함", t_ask_returns_within_bounds)


def t_min_pass_enforced():
    opt = FormulationOptimizer(
        bounds=(0.1, 0.9), objectives={"y": Objective(direction="max")}, seed=0
    )
    opt.tell(0.2, {"y": 1.0})
    try:
        opt.ask()
        raise AssertionError("관측 부족인데 ask()가 성공함")
    except RuntimeError:
        pass
check("3. 최소 관측 수 미달 시 ask() 거부", t_min_pass_enforced)


def t_mediator_hierarchy():
    opt = FormulationOptimizer(
        bounds=(0.1, 0.9),
        objectives={"y": Objective(direction="max", fail_value=0.0)},
        use_mediator=True,
        seed=0,
    )
    rng = np.random.default_rng(0)
    for x in np.linspace(0.1, 0.9, 6):
        med = 0.5 * x + rng.normal(0, 0.01)
        opt.tell(x, {"y": float(x + med)}, mediator=med)
    opt.tell(0.85, passed=False)  # 실패 행: mediator 없이도 기록 가능해야 함
    r = opt.ask()
    assert r["n_fail"] == 1
    assert 0.1 <= r["recommended_x"] <= 0.9
check("4. mediator 계층 구조 + 실패 행(mediator 없음) 처리", t_mediator_hierarchy)


def t_reproducibility():
    opt = make_direct_opt()
    r1 = opt.ask(seed=42)
    r2 = opt.ask(seed=42)
    assert r1["recommended_x"] == r2["recommended_x"]
    assert r1["EI"] == r2["EI"]
check("5. 동일 시드 재현성", t_reproducibility)


print(f"\n{'=' * 62}")
for name, status, msg in results:
    mark = "PASS" if status == "PASS" else "FAIL"
    print(f"[{mark}] {name}")
    if msg:
        print(f"    -> {msg}")
n_pass = sum(1 for _, s, _ in results if s == "PASS")
print(f"{'=' * 62}\n{n_pass}/{len(results)} 통과")
sys.exit(0 if n_pass == len(results) else 1)
