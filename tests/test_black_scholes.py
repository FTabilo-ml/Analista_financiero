import math

from src.models.black_scholes import price, delta

def test_put_call_parity():
    S, K, r, q, sigma, T = 100, 100, 0.02, 0.01, 0.25, 0.5
    call = price(S, K, r, q, sigma, T, "call")
    put  = price(S, K, r, q, sigma, T, "put")
    parity = S*math.exp(-q*T) - K*math.exp(-r*T)
    assert abs(call - put - parity) < 1e-8

def test_delta_sign():
    """Delta call debería ser positiva y put negativa."""
    S, K, r, q, sigma, T = 100, 100, 0.03, 0.0, 0.2, 1
    assert delta(S, K, r, q, sigma, T, "call") > 0
    assert delta(S, K, r, q, sigma, T, "put") < 0
