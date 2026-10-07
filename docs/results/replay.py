"""실측 캠페인(35회)을 FormulationOptimizer로 다시 돌려 보는 검증 스크립트.

BO 회차마다 그 시점에 있던 자료(공통 초기 15점 + 해당 군의 이전 회차)만 넣고,
ask()가 실험실에서 실제로 배합한 φ를 그대로 내놓는지 확인한다.

저장소 루트에서 실행:  python docs/results/replay.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from formulation_bo import FormulationOptimizer, Objective, gem_tc, kd_viscosity  # noqa: E402

df = pd.read_csv(HERE / "data" / "tim_vlab_results.csv")
initial = df[df.arm == "initial"]


def recommend(arm: str, t: int) -> float:
    """arm의 t회차에서 최적화기가 추천하는 φ (그 이전 측정값만 사용)."""
    use_dcv = arm == "experimental"
    opt = FormulationOptimizer(
        bounds=(0.30, 0.55),
        objectives={
            "tc": Objective(direction="max", prior=gem_tc, fail_value=0.0),
            "eta": Objective(direction="min", prior=kd_viscosity, transform="log"),
        },
        x_max=0.64,
        use_mediator=use_dcv,
        seed=t - 1,  # 캠페인 시드 규칙: 해당 군에 이미 기록된 결과 수
    )
    seen = pd.concat([initial, df[(df.arm == arm) & (df.iteration < t)]])
    for r in seen.itertuples():
        if r.passed:
            opt.tell(r.phi, {"tc": r.tc, "eta": r.eta}, mediator=r.d_cv if use_dcv else None)
        else:
            opt.tell(r.phi, passed=False)
    lam1 = 0.05 + 0.9 * (t - 1) / 9  # 캠페인 가중치 스케줄: 점도 중시 -> 열전도도 중시
    return opt.ask(lam=(lam1, 1 - lam1))["recommended_x"]


if __name__ == "__main__":
    worst = 0.0
    print(f"{'arm':<13}{'round':>5}{'lab':>9}{'replay':>9}")
    for arm in ("experimental", "control"):
        for t in range(1, 11):
            lab = float(df[(df.arm == arm) & (df.iteration == t)].phi.iloc[0])
            got = recommend(arm, t)
            worst = max(worst, abs(got - lab))
            print(f"{arm:<13}{t:>5}{lab:>9.4f}{got:>9.4f}")
    # 실험 기록의 φ는 소수 넷째 자리까지 저장돼 있다
    assert worst < 1e-4, f"replay diverged from the lab log by {worst:.4f}"
    print("\nAll 20 recommendations match the lab log.")
