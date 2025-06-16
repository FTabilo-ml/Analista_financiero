"""
SABR + SAC Strategy
-------------------

* Calcula σ_impl teórica con SABR.
* Identifica desvíos vs. mercado.
* Agente SAC decide abrir/cerrar posición long/short vol o market‑making.

Este módulo contiene solo la interface mínima; el entrenamiento se hará
en otro script / notebook para mantener el código desacoplado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from stable_baselines3 import SAC

from src.models.sabr import SABRParams, implied_vol
from src.models.black_scholes import price as bs_price


@dataclass
class Observation:
    spot: float
    strike: float
    maturity: float  # años
    market_iv: float
    features: Dict[str, float]  # VIX, tasas, spreads...


class SABRSACStrategy:
    """
    Inferfaz de alto nivel:

        strategy = SABRSACStrategy(agent_path="models/sac.zip", sabr_params=params)
        action = strategy.act(obs)  # {'side': 'buy', 'qty': 10}

    """

    def __init__(self, agent_path: str, sabr_params: SABRParams):
        self.agent = SAC.load(agent_path)  # type: ignore
        self.sabr_params = sabr_params

    # --------------------------------------------------------------------- #
    # Feature engineering
    # --------------------------------------------------------------------- #

    def _build_state(self, obs: Observation) -> np.ndarray:
        theo_iv = implied_vol(
            obs.spot * np.exp(-obs.features.get("div_yield", 0) * obs.maturity),
            obs.strike,
            obs.maturity,
            self.sabr_params,
        )
        mispricing = (obs.market_iv - theo_iv) / theo_iv
        extra_feats = np.array(list(obs.features.values()), dtype=np.float32)
        return np.concatenate(([mispricing], extra_feats))

    # --------------------------------------------------------------------- #
    # API pública
    # --------------------------------------------------------------------- #

    def act(self, obs: Observation) -> Dict[str, float | str]:
        state = self._build_state(obs)
        action = self.agent.predict(state, deterministic=True)[0]
        # Aquí mapearías la acción SAC → ejecución real.
        # Ejemplo sencillo:
        side = "buy" if action[0] > 0 else "sell"
        qty = abs(action[0]) * 100  # escala arbitraria
        return {"side": side, "qty": qty}
