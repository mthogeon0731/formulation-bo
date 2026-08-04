"""물리 한계 검증 유틸 + 예시 물리 prior 모델.

핵심 계약: 입력값이 정의역을 벗어나면 조용히 클리핑하지 않고 항상
PhysicalLimitError를 던진다 (Fail-Fast). 물리적으로 불가능한 배합비를
그럴듯한 숫자로 바꿔치기하는 것이 이 종류 시스템의 최악 실패 양식이다.

gem_tc / kd_viscosity는 demo.py가 가상 정답 함수를 만드는 데 쓰는
예시 prior일 뿐, FormulationOptimizer 코어는 어떤 prior 함수든
(x: ndarray) -> ndarray 시그니처만 맞으면 그대로 받는다.
"""
from __future__ import annotations

import numpy as np


class PhysicalLimitError(ValueError):
    """입력이 물리적으로 불가능한 정의역을 벗어났을 때 발생."""


def check_domain(x: np.ndarray, x_max: float | None, name: str = "x") -> None:
    """x가 [0, x_max) 안에 있는지 확인. 벗어나면 클리핑 대신 예외를 던진다."""
    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)):
        raise ValueError(f"{name}에 NaN/inf가 포함되어 있습니다.")
    if np.any(x < 0):
        raise ValueError(f"{name}는 0 이상이어야 합니다.")
    if x_max is not None and np.any(x >= x_max):
        worst = float(np.max(x))
        raise PhysicalLimitError(
            f"입력된 {name}(최대 {worst:.4g})가 물리적 한계({name}_max={x_max})를 "
            f"초과했습니다. bounds를 다시 설정하세요."
        )


def gem_tc(
    phi,
    kappa_m: float = 0.20,
    kappa_f: float = 30.0,
    phi_c: float = 0.64,
    t: float = 1.5,
) -> np.ndarray:
    """McLachlan GEM 방정식으로 유효 열전도도 계산 (벡터화 이분법 60회).

    demo.py 전용 예시 prior. phi_c 근처에서 발산하므로 phi >= phi_c
    입력은 PhysicalLimitError로 거부한다.
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    check_domain(phi, phi_c, name="phi")

    A = (1.0 - phi_c) / phi_c
    km_t = kappa_m ** (1.0 / t)
    kf_t = kappa_f ** (1.0 / t)

    def f(ke: np.ndarray) -> np.ndarray:
        ke_t = ke ** (1.0 / t)
        return (
            (1.0 - phi) * (km_t - ke_t) / (km_t + A * ke_t)
            + phi * (kf_t - ke_t) / (kf_t + A * ke_t)
        )

    lo = np.full_like(phi, 1e-4)
    hi = np.full_like(phi, kappa_f)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pos = f(mid) > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    return 0.5 * (lo + hi)


def kd_viscosity(
    phi,
    eta_m: float = 5.0,
    phi_m: float = 0.64,
    intrinsic_eta: float = 2.5,
) -> np.ndarray:
    """Krieger-Dougherty 상대점도 x 매트릭스 점도. demo.py 전용 예시 prior."""
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    check_domain(phi, phi_m, name="phi")
    return eta_m * (1.0 - phi / phi_m) ** (-intrinsic_eta * phi_m)
