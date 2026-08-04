"""GP 대리모델.

목적함수 하나당 ResidualObjectiveModel 하나:
  use_mediator=False:  x -> y - prior(x)              (직접 잔차 학습)
  use_mediator=True :  (x, mediator) -> y - prior(x)   (매개변수 계층 학습)
transform="log"이면 y 대신 log(y)에 대해 같은 잔차 학습을 수행한다
(양수/배수 스케일 목적함수, 예: 점도).

매개변수(mediator) 자체도 MediatorModel(x -> mediator)로 별도 학습해서
관측되지 않은 지점(예: 실패한 배합)의 mediator를 사후평균으로 채울 수 있다.

설계 원칙 (원 코어 유지):
1) 재현성: 모든 GPR에 random_state 주입.
2) 탐색 신호 순도: predict()가 반환하는 std는 epistemic 성분만 포함.
   WhiteKernel(관측 노이즈)은 학습 안정화에만 쓰고 예측 분산에서 제거한다.
"""
from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


def _make_gp(n_dims: int, seed: int) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=[1.0] * n_dims, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=seed,
    )


def _extract_noise_level(gp: GaussianProcessRegressor) -> float:
    k = gp.kernel_
    if hasattr(k, "k2") and isinstance(k.k2, WhiteKernel):
        return float(k.k2.noise_level)
    return 0.0


class _BaseGP:
    """스케일러 + GP + epistemic 분리를 묶은 공통 베이스."""

    def __init__(self, n_dims: int, seed: int):
        self.scaler = StandardScaler()
        self.gp = _make_gp(n_dims, seed)
        self._noise_level = 0.0
        self._y_std = 1.0

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(y) < 3:
            raise ValueError(f"GP 학습에 최소 3개 관측이 필요합니다 (현재 {len(y)}개).")
        self.scaler.fit(X)
        self.scaler.scale_[self.scaler.scale_ == 0] = 1.0
        Xs = self.scaler.transform(X)
        self.gp.fit(Xs, y)
        self._noise_level = _extract_noise_level(self.gp)
        self._y_std = float(np.std(y)) if np.std(y) > 0 else 1.0

    def _predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xs = self.scaler.transform(X)
        mu, std_total = self.gp.predict(Xs, return_std=True)
        noise_var_y = self._noise_level * (self._y_std ** 2)
        var_epi = np.maximum(std_total ** 2 - noise_var_y, 0.0)
        return mu, np.sqrt(var_epi)


class MediatorModel(_BaseGP):
    """x -> mediator (관측되지 않은 지점의 mediator를 사후평균으로 채우는 데 사용)."""

    def __init__(self, seed: int = 0):
        super().__init__(n_dims=1, seed=seed)

    def fit(self, x, mediator) -> "MediatorModel":
        X = np.asarray(x, dtype=float).reshape(-1, 1)
        self._fit(X, np.asarray(mediator, dtype=float))
        return self

    def predict(self, x):
        X = np.asarray(x, dtype=float).reshape(-1, 1)
        return self._predict(X)


class ResidualObjectiveModel(_BaseGP):
    """물리 prior 잔차를 학습하는 목적함수 GP. use_mediator로 계층 구조 선택."""

    def __init__(
        self,
        use_mediator: bool,
        prior_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        transform: Literal["linear", "log"] = "linear",
        seed: int = 0,
    ):
        super().__init__(n_dims=2 if use_mediator else 1, seed=seed)
        self.use_mediator = use_mediator
        self.prior_fn = prior_fn or (lambda x: np.ones_like(np.asarray(x, dtype=float)))
        self.transform = transform

    def _features(self, x, mediator=None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.use_mediator:
            if mediator is None:
                raise ValueError("use_mediator=True인 모델은 mediator 값이 필요합니다.")
            return np.column_stack([x, np.asarray(mediator, dtype=float)])
        return x.reshape(-1, 1)

    def fit(self, x, y, mediator=None) -> "ResidualObjectiveModel":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        prior = self.prior_fn(x)
        if self.transform == "log":
            if np.any(y <= 0) or np.any(prior <= 0):
                raise ValueError("transform='log' 목적함수는 y와 prior가 모두 양수여야 합니다.")
            resid = np.log(y) - np.log(prior)
        else:
            resid = y - prior
        self._fit(self._features(x, mediator), resid)
        return self

    def predict(self, x, mediator=None, prior_cache: np.ndarray | None = None):
        """(mu, std) 반환. transform='log'이면 log 공간의 (mu, std)."""
        x = np.asarray(x, dtype=float)
        mu_r, s = self._predict(self._features(x, mediator))
        prior = prior_cache if prior_cache is not None else self.prior_fn(x)
        base = np.log(prior) if self.transform == "log" else prior
        return mu_r + base, s
