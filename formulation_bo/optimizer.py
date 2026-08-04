"""범용 배합 최적화기: ask() / tell() 인터페이스.

어떤 배합 최적화 문제든(필러 충전율, 첨가제 농도, 도즈량 등 연속 변수 x
하나로 표현되는 문제) 아래 3가지를 주면 그대로 쓸 수 있다.

  - bounds       : 탐색할 x 범위 (lo, hi)
  - objectives   : {이름: Objective(direction="max"/"min", ...)} 1개 이상
  - x_max        : (선택) x가 넘으면 안 되는 물리적 한계. 넘으면 항상
                    PhysicalLimitError를 던진다 — 절대 조용히 클리핑하지 않는다.

objective별로 물리 prior(예: GEM 열전도 모델)를 선택적으로 꽂을 수 있고,
mediator(예: 분산도)처럼 x와 objective 사이를 매개하는 보조 관측치를
계층적으로 쓸지(use_mediator=True/False) 선택할 수 있다.

사용법:
    opt = FormulationOptimizer(bounds=(0.30, 0.55), objectives={...})
    x = opt.ask()["recommended_x"]
    y = run_real_experiment(x)
    opt.tell(x, y)

범위를 벗어난 확장(다성분 심플렉스 배합 설계 등)은 다루지 않는다 —
연속 스칼라 x 하나짜리 배합 문제가 이 구현의 대상이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd

from .acquisition import draw_lambda, empirical_ei, parego_scalarize
from .models import MediatorModel, ResidualObjectiveModel
from .physics import PhysicalLimitError, check_domain


@dataclass
class Objective:
    """목적함수 하나의 정의."""

    direction: Literal["max", "min"]
    prior: Callable[[np.ndarray], np.ndarray] | None = None
    transform: Literal["linear", "log"] = "linear"
    fail_value: float | None = None
    """실패(제작 불가 등)한 관측에 대입할 값. None이면 실패 행은 이 목적함수
    학습에서 제외한다. 0.0을 주면(예: TC) 실패를 페널티로 학습에 포함시킨다."""


class FormulationOptimizer:
    def __init__(
        self,
        bounds: tuple[float, float],
        objectives: dict[str, Objective],
        x_max: float | None = None,
        use_mediator: bool = False,
        n_candidates: int = 400,
        M: int = 200,
        xi: float = 0.01,
        seed: int = 0,
        min_pass: int = 3,
    ):
        lo, hi = bounds
        if not (0 < lo < hi):
            raise ValueError(f"bounds가 올바르지 않습니다: {bounds}")
        if x_max is not None and hi >= x_max:
            raise PhysicalLimitError(
                f"bounds 상한({hi})이 물리적 한계 x_max({x_max}) 이상입니다."
            )
        if not objectives:
            raise ValueError("objectives는 1개 이상이어야 합니다.")

        self.bounds = (float(lo), float(hi))
        self.x_max = x_max
        self.objectives = objectives
        self.use_mediator = use_mediator
        self.n_candidates = n_candidates
        self.M = M
        self.xi = xi
        self.seed = seed
        self.min_pass = min_pass
        self._rows: list[dict] = []
        self.last_result_: dict | None = None

    def tell(
        self,
        x: float,
        y: dict[str, float] | None = None,
        mediator: float | None = None,
        passed: bool = True,
    ) -> "FormulationOptimizer":
        """관측 1건 기록. 실패(passed=False)면 y는 몰라도 된다."""
        x = float(x)
        check_domain(np.array([x]), self.x_max, name="x")
        if x <= 0:
            raise ValueError("x는 양수여야 합니다.")
        if self.use_mediator and passed and mediator is None:
            raise ValueError("use_mediator=True에서 성공 관측은 mediator 값이 필요합니다.")

        y = y or {}
        row: dict = {
            "x": x,
            "mediator": float(mediator) if mediator is not None else np.nan,
            "passed": bool(passed),
        }
        for name, obj in self.objectives.items():
            if passed:
                if name not in y:
                    raise ValueError(f"성공 관측에는 목적함수 '{name}' 값이 필요합니다.")
                row[name] = float(y[name])
            else:
                row[name] = obj.fail_value if obj.fail_value is not None else np.nan
        self._rows.append(row)
        self.last_result_ = None
        return self

    def ask(
        self,
        seed: int | None = None,
        lam: tuple[float, ...] | np.ndarray | None = None,
    ) -> dict:
        """다음 시도할 x와 예측 곡선을 담은 결과 dict를 반환."""
        if not self._rows:
            raise RuntimeError("관측이 없습니다. tell()을 먼저 호출하세요.")
        df = pd.DataFrame(self._rows)
        passed_mask = df["passed"].to_numpy()
        passed_df = df[passed_mask]
        if len(passed_df) < self.min_pass:
            raise RuntimeError(
                f"최소 {self.min_pass}개의 성공 관측이 필요합니다 (현재 {len(passed_df)}개)."
            )

        rng = np.random.default_rng(seed if seed is not None else self.seed)
        lo, hi = self.bounds
        x_grid = np.linspace(lo, hi, self.n_candidates)
        names = list(self.objectives.keys())
        lam_arr = np.asarray(lam, dtype=float) if lam is not None else draw_lambda(rng, len(names))

        prior_grid = {
            name: (obj.prior(x_grid) if obj.prior is not None else np.ones_like(x_grid))
            for name, obj in self.objectives.items()
        }

        x_all = df["x"].to_numpy(dtype=float)
        mediator_all = df["mediator"].to_numpy(dtype=float, copy=True)

        mediator_model = None
        if self.use_mediator:
            mediator_model = MediatorModel(self.seed).fit(
                passed_df["x"], passed_df["mediator"]
            )
            nan_mask = np.isnan(mediator_all)
            if nan_mask.any():
                mu_fill, _ = mediator_model.predict(x_all[nan_mask])
                mediator_all[nan_mask] = mu_fill

        models: dict[str, ResidualObjectiveModel] = {}
        ranges: dict[str, tuple[float, float]] = {}
        directions: dict[str, str] = {}
        for name, obj in self.objectives.items():
            directions[name] = obj.direction
            if obj.fail_value is not None:
                x_fit, y_fit = x_all, df[name].to_numpy(dtype=float)
                med_fit = mediator_all if self.use_mediator else None
                ranges[name] = (float(df[name].min()), float(df[name].max()))
            else:
                x_fit = passed_df["x"].to_numpy(dtype=float)
                y_fit = passed_df[name].to_numpy(dtype=float)
                med_fit = mediator_all[passed_mask] if self.use_mediator else None
                ranges[name] = (float(y_fit.min()), float(y_fit.max()))
            models[name] = ResidualObjectiveModel(
                self.use_mediator, obj.prior, obj.transform, self.seed
            ).fit(x_fit, y_fit, med_fit)

        samples = {name: np.empty((self.M, self.n_candidates)) for name in names}
        if self.use_mediator:
            mu1, s1 = mediator_model.predict(x_grid)
            for m in range(self.M):
                med_samp = rng.normal(mu1, np.maximum(s1, 1e-9))
                for name, obj in self.objectives.items():
                    mu, s = models[name].predict(x_grid, mediator=med_samp, prior_cache=prior_grid[name])
                    draw = rng.normal(mu, np.maximum(s, 1e-9))
                    samples[name][m] = np.exp(draw) if obj.transform == "log" else draw
        else:
            for name, obj in self.objectives.items():
                mu, s = models[name].predict(x_grid, prior_cache=prior_grid[name])
                draw = rng.normal(mu, np.maximum(s, 1e-9), size=(self.M, self.n_candidates))
                samples[name] = np.exp(draw) if obj.transform == "log" else draw

        g_samples = parego_scalarize(samples, directions, ranges, lam_arr, names)
        obs_values = {name: passed_df[name].to_numpy(dtype=float) for name in names}
        g_obs = parego_scalarize(obs_values, directions, ranges, lam_arr, names)
        g_best = float(np.min(g_obs))
        incumbent_idx = int(np.argmin(g_obs))

        ei = empirical_ei(g_samples, g_best, self.xi)
        best = int(np.argmax(ei))

        result: dict = {
            "recommended_x": float(x_grid[best]),
            "EI": float(ei[best]),
            "parego_lambda": lam_arr.tolist(),
            "g_best": g_best,
            "incumbent_x": float(passed_df["x"].iloc[incumbent_idx]),
            "n_observations": int(len(df)),
            "n_pass": int(len(passed_df)),
            "n_fail": int(len(df) - len(passed_df)),
            "seed": seed if seed is not None else self.seed,
            "predictions": {
                name: {
                    "mean": float(samples[name][:, best].mean()),
                    "std": float(samples[name][:, best].std()),
                }
                for name in names
            },
            "curves": {"x": x_grid.tolist(), "ei": ei.tolist()},
        }
        for name in names:
            result["curves"][f"{name}_mean"] = samples[name].mean(axis=0).tolist()
            result["curves"][f"{name}_std"] = samples[name].std(axis=0).tolist()

        self.last_result_ = result
        return result
