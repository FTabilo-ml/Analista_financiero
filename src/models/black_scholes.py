"""
Black‑Scholes‑Merton pricer y griegas (dividendo continuo)
(c) Nodum AI – 2025
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from scipy.stats import norm as _norm

__all__ = [
    "OptionType",
    "price",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
]

# ────────────────────────────────────────────────────────────────────────────
# Utilidades internas
# ────────────────────────────────────────────────────────────────────────────


def _N(x: float) -> float:  # CDF
    return float(_norm.cdf(x))


def _phi(x: float) -> float:  # pdf
    return float(_norm.pdf(x))


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


def _d1(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    return (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def _d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * math.sqrt(T)


# ────────────────────────────────────────────────────────────────────────────
# Precio
# ────────────────────────────────────────────────────────────────────────────


def price(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    kind: Literal["call", "put"] | OptionType = OptionType.CALL,
) -> float:
    d1 = _d1(S, K, r, q, sigma, T)
    d2 = _d2(d1, sigma, T)
    if kind in (OptionType.CALL, "call"):
        return S * math.exp(-q * T) * _N(d1) - K * math.exp(-r * T) * _N(d2)
    else:
        return K * math.exp(-r * T) * _N(-d2) - S * math.exp(-q * T) * _N(-d1)


# ────────────────────────────────────────────────────────────────────────────
# Griegas
# ────────────────────────────────────────────────────────────────────────────


def delta(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    kind: Literal["call", "put"] | OptionType = OptionType.CALL,
) -> float:
    d1 = _d1(S, K, r, q, sigma, T)
    if kind in (OptionType.CALL, "call"):
        return math.exp(-q * T) * _N(d1)
    else:
        return math.exp(-q * T) * (_N(d1) - 1)


def gamma(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    d1 = _d1(S, K, r, q, sigma, T)
    return math.exp(-q * T) * _phi(d1) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    d1 = _d1(S, K, r, q, sigma, T)
    return S * math.exp(-q * T) * _phi(d1) * math.sqrt(T) / 100  # convención /100


def theta(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    kind: Literal["call", "put"] | OptionType = OptionType.CALL,
) -> float:
    d1 = _d1(S, K, r, q, sigma, T)
    d2 = _d2(d1, sigma, T)
    first = -S * math.exp(-q * T) * _phi(d1) * sigma / (2 * math.sqrt(T))
    if kind in (OptionType.CALL, "call"):
        second = q * S * math.exp(-q * T) * _N(d1)
        third = -r * K * math.exp(-r * T) * _N(d2)
    else:
        second = -q * S * math.exp(-q * T) * _N(-d1)
        third = r * K * math.exp(-r * T) * _N(-d2)
    return (first + second + third) / 365  # por día


def rho(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    kind: Literal["call", "put"] | OptionType = OptionType.CALL,
) -> float:
    d1 = _d1(S, K, r, q, sigma, T)
    d2 = _d2(d1, sigma, T)
    if kind in (OptionType.CALL, "call"):
        return K * T * math.exp(-r * T) * _N(d2) / 100
    else:
        return -K * T * math.exp(-r * T) * _N(-d2) / 100
