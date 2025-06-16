import numpy as np
from src.models.sabr import SABRParams, implied_vol

def test_sabr_atm_limit():
    F, T = 100, 0.25
    params = SABRParams(alpha=0.2, beta=0.5, rho=-0.3, nu=0.6)
    atm = implied_vol(F, F, T, params)
    # Debe ser positiva y < 150 %
    assert 0.005 < atm < 1.5
