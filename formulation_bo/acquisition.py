"""획득함수: MC 불확실성 전파 + ParEGO 스칼라화(N개 목적함수) + 경험적 EI.

핵심 설계 (원 코어 유지):
- 스칼라화는 GP 학습 전이 아니라 MC 샘플 단계에서 수행한다 (물리 prior 구조 보존).
- MC 전파 후 예측분포는 비정규이므로 폐형 EI(norm.cdf) 대신
  경험적 EI = mean(max(g_best - g_sample - xi, 0)) 를 사용한다.
- lambda(가중치)는 호출당 1회, 목적함수 개수만큼 디리클레 분포에서 추출한다
  (N=2일 때 Dirichlet(1,1) = 원 코어의 uniform(0,1) 2-way split과 동일).
"""
from __future__ import annotations

import numpy as np

PAREGO_RHO = 0.05


def draw_lambda(rng: np.random.Generator, n_objectives: int) -> np.ndarray:
    """단위 심플렉스에서 가중치 1회 추출 (목적함수 개수만큼)."""
    return rng.dirichlet(np.ones(n_objectives))


def parego_scalarize(
    values: dict[str, np.ndarray],
    directions: dict[str, str],
    ranges: dict[str, tuple[float, float]],
    lam: np.ndarray,
    names: list[str],
    rho: float = PAREGO_RHO,
) -> np.ndarray:
    """증강 체비셰프 스칼라화. 모든 목적을 [0,1] '최소화' 방향으로 정규화 후 결합.

    direction="max"인 목적은 1 - norm(y)로 뒤집어서 최소화 방향으로 맞춘다.
    g = max_i(lam_i * f_i) + rho * sum_i(lam_i * f_i)
    """
    weighted = []
    for i, name in enumerate(names):
        lo, hi = ranges[name]
        span = max(hi - lo, 1e-9)
        f = np.clip((values[name] - lo) / span, -0.5, 1.5)
        if directions[name] == "max":
            f = 1.0 - f
        weighted.append(lam[i] * f)
    stacked = np.stack(weighted, axis=0)
    return np.max(stacked, axis=0) + rho * np.sum(stacked, axis=0)


def empirical_ei(
    g_samples: np.ndarray,  # (M, n_candidates)
    g_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """비정규 예측분포용 경험적 EI (최소화 방향)."""
    improvement = np.maximum(g_best - g_samples - xi, 0.0)
    return improvement.mean(axis=0)
