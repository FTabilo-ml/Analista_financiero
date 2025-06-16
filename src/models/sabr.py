"""
Implementa la volatilidad implícita SABR (Hagan et al., 2002).

σ_BS(F, K) ≈ α ϕ(z) / χ(z) · [1 + ( (2γ₂ - γ₁²) / 24 · (Tσ²) + … ) ]

Donde:
    F : forward
    K : strike
    α : sigma_0 (vol nivel)
    β : elasticidad (0→1)
    ρ : correlación
    ν : vol-of-vol
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

__all__ = ["SABRParams", "implied_vol", "calibrate_beta_fixed"]


class SABRParams(NamedTuple):
    alpha: float
    beta: float
    rho: float
    nu: float


def _z(F: float, K: float, params: SABRParams) -> float:
    alpha, beta, rho, nu = params
    if F == K:
        return 0.0
    ln_fk = math.log(F / K)
    return (nu / alpha) * (F * K) ** ((1 - beta) / 2) * ln_fk


def _chi(z: float, rho: float) -> float:
    a = math.sqrt(1 - 2 * rho * z + z**2)
    return math.log((a + z - rho) / (1 - rho))


def implied_vol(F: float, K: float, T: float, params: SABRParams) -> float:
    """Fórmula de Hagan σ_impl BS."""
    alpha, beta, rho, nu = params

    if F == K:
        # ATM simplificado
        one = (
            (1 - beta) ** 2 / 24 * (alpha**2) / (F ** (2 - 2 * beta))
            + (rho * beta * nu * alpha) / (4 * F ** (1 - beta))
            + (2 - 3 * rho**2) * nu**2 / 24
        )
        return alpha / F ** (1 - beta) * (1 + one * T)

    z = _z(F, K, params)
    chi = _chi(z, rho)
    # prefactor
    A = alpha / ((F * K) ** ((1 - beta) / 2) * (1 + ((1 - beta) ** 2 / 24) * (math.log(F / K)) ** 2))
    B = z / chi
    # Correcciones tiempo T
    C = (
        (1 - beta) ** 2 / 24 * (alpha**2) / ((F * K) ** (1 - beta))
        + (rho * beta * nu * alpha) / (4 * (F * K) ** ((1 - beta) / 2))
        + (2 - 3 * rho**2) * nu**2 / 24
    )
    return A * B * (1 + C * T)


# ---------- Calibración básica (β fijo) -------------------------------------


def calibrate_beta_fixed(
    strikes: np.ndarray,
    vols: np.ndarray,
    F: float,
    T: float,
    beta: float = 0.5,
) -> SABRParams:
    """
    Ajusta α, ρ, ν vía mínimos cuadrados, manteniendo β fijo.

    Retorna SABRParams(α, β, ρ, ν)
    """
    from scipy.optimize import minimize

    def loss(x):
        a, r, n = x
        params = SABRParams(a, beta, r, n)
        model = np.array([implied_vol(F, k, T, params) for k in strikes])
        return np.mean((model - vols) ** 2)

    # Boundaries: α>0, |ρ|<=0.999, ν>0
    bounds = [(1e-4, None), (-0.999, 0.999), (1e-4, None)]
    res = minimize(loss, x0=[0.2, 0.0, 0.5], bounds=bounds, method="L-BFGS-B")
    return SABRParams(res.x[0], beta, res.x[1], res.x[2])
