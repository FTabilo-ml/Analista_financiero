## 📈 Nodum Trading Bot

**Nodum Trading Bot** es un _toolkit_ de I+D para generar y explotar
superficies de volatilidad implícita libres de arbitraje en mercados de
derivados (opciones sobre IPSA, SPY, futuros de cobre, etc.).  
Integra tres pilares:

| Pilar | Descripción | Módulos |
|-------|-------------|---------|
| **Explainable Surfaces** | Modelo generativo (Diffusion/Transformer) que crea σ‑surfaces arbitrage‑free condicionadas a _macro features_ (VIX, tasas, microestructura). | `notebooks/02_surface_gen.ipynb` |
| **SABR Pricer** | Ajuste semianalítico (Hagan 2002) para estimar precios teóricos y griegas. | `src/models/sabr.py` |
| **RL Trader (SAC)** | Agente **Soft Actor‑Critic** que detecta desvíos de volatilidad y optimiza _timing_ / horizonte (market‑making o toma de posición). | `src/strategy.py` |

### Arquitectura


data/ # 🔹 datasets (no versionados)
notebooks/ # 🔹 experimentos Jupyter
src/
├── models/ # • black_scholes.py (closed‑form + griegas)
│ └── sabr.py # • SABR impl. vol & calibración
├── strategy.py # • wrapper SAC + señales mispricing
└── init.py # • rutas DATA_DIR, NOTEBOOKS_DIR
tests/ # 🔹 PyTest (precio y SABR)


### Instalación rápida

```bash
conda env create -f environment.yml   # requerimientos científicos + RL
conda activate nodum-trading
pip install -e .                      # modo editable
pytest -q                             # ✔ 3 tests verdes
```

Próximos hitos
 Calibrar SABR con cadenas de opciones IPSA y SPY → data/vol_surface_*.parquet

 Entrenar Diffusion para generar superficies condicionales

 Back‑test RL (VectorBT) con fricción real y walk‑forward

 API REST (FastAPI) para publicar precios, griegas y señales
