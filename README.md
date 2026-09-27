# AQOL — Adaptive Quantum-classical Optimization Loop

**Fewer measurements. Less energy. Better results.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.0-orange)](https://github.com/saimon-prog/aqol)
[![Status](https://img.shields.io/badge/status-beta-yellow)](https://github.com/saimon-prog/aqol)
[![Donate](https://img.shields.io/badge/❤_Support-Solana-purple?logo=solana&logoColor=white)](#-support--fund-the-project)

AQOL is an adaptive optimization loop designed for problems where **every measurement is expensive** — in time, money, electricity, or physical resources.

Quantum circuits · Hardware calibration · Clinical trials · Lab synthesis · High-performance simulation

Instead of exploring at random, AQOL learns from past measurements to intelligently choose the next point to evaluate.  
**Result: fewer trials to reach a good optimum — and a lighter energy footprint.**

---

## Why AQOL exists

In many domains, a single measurement can:
- consume **kWh** (sometimes tens of them) of electricity,
- occupy rare equipment for hours,
- cost hundreds or thousands of euros/dollars,
- generate waste or non-negligible environmental impact.

Random search (or a naive grid) wastes most of these measurements in regions already known to be mediocre.  
AQOL focuses effort where information is most useful.

### Concrete impact

| Aspect                    | Random search            | AQOL                          |
|---------------------------|--------------------------|-------------------------------|
| Number of measurements    | High                     | Reduced (often 2–5× fewer)    |
| Electricity consumed      | Proportional             | Strongly reduced              |
| Machine / lab time        | Long                     | Shorter                       |
| Experimental cost         | High                     | Lower                         |
| Carbon footprint          | Larger                   | Lighter                       |

**Data & energy economy**: every avoided measurement means electricity not consumed, machine time freed, and a concrete contribution to more frugal science.

---

## Installation

```bash
pip install -e .
```

Or from the repository:

```bash
git clone https://github.com/saimon-prog/aqol.git
cd aqol
pip install -e .
```

**Dependencies**: `numpy` and `scipy` only.

---

## Quick start

### Single parameter

```python
from aqol import AQOL

opt = AQOL(bounds=(-3, 3))

for _ in range(30):                 # your measurement budget
    x = opt.ask()                   # next point to measure
    y = my_expensive_measurement(x) # your real measurement
    opt.tell(x, y)

x_best, estimation = opt.best()
print(f"Best point: {x_best:.4f}  (estimated value ~ {estimation:.4f})")
```

### Multiple parameters

```python
from aqol import AQOL

opt = AQOL(bounds=[(-3, 3), (0, 10), (-1, 1)])

for _ in range(60):
    x = opt.ask()                   # np.ndarray of shape (3,)
    y = my_expensive_measurement(*x)
    opt.tell(x, y)

x_best, estimation = opt.best()
```

See `examples/optimize_demo.py` for a full demonstration compared to random search.

---

## Advanced options

```python
opt = AQOL(
    bounds=(-5, 5),
    n_init=5,                 # initial exploration points
    n_candidates=2000,        # candidates to maximize EI
    length_scale=0.8,         # RBF kernel length scale
    signal_var=1.0,
    noise_var=0.05,           # measurement noise variance
    xi=0.01,                  # exploration parameter (Expected Improvement)
    adapt_hyperparams=True,   # automatic hyperparameter adaptation
    random_state=42,
)
```

- `best()` returns the point with the lowest **model mean** (not the noisy observed minimum) → avoids the optimistic bias of noise.
- `history()` returns the full history of points and values.
- `n_observations` indicates how many measurements have been made so far.

---

## How it works (briefly)

1. **Exploration phase**: a few initial points (Sobol) to build a first model.
2. **Surrogate model**: Gaussian Process (GP) that estimates the function and its uncertainty.
3. **Acquisition**: Expected Improvement (EI) — we pick the point with the highest expected gain.
4. **Adaptation**: GP hyperparameters are lightly adjusted as measurements arrive.
5. **Bias correction**: the “best” point is evaluated via the GP mean, not the raw noisy value.

AQOL does not reinvent Bayesian optimization. It assembles it under a simple, robust API focused on expensive measurements.

---

## Typical use cases

- Quantum circuit / gate parameter optimization
- Physical hardware calibration (lasers, cryostats, sensors…)
- Chemical synthesis / formulation (expensive trials)
- Clinical or biological trials with limited budget
- Hyperparameter tuning of heavy models (when each run is costly)
- Any problem where evaluation cost dominates compute cost

---

## Known limitations

- Best suited to **low / medium dimensions** (typically ≤ 8–10 parameters). Beyond that, gradient-based approaches (e.g. BoTorch) become preferable.
- The GP remains an approximation: reliable in explored regions, more uncertain elsewhere.
- AQOL does not reduce the intrinsic noise of an individual measurement (e.g. quantum probabilistic nature) — it reduces the **number** of measurements needed.

---

## Roadmap

- [x] Simple 1D / nD API
- [x] Numerical stability (Cholesky)
- [x] Light hyperparameter adaptation
- [x] Sobol sampling
- [ ] Natural language interface (problem description → configuration)
- [ ] LLM-assisted hybrid acquisition
- [ ] Multi-fidelity (cheap + expensive measurements)
- [ ] Memory / transfer across similar problems
- [ ] Constraint support

---

## 💚 Support & fund the project

AQOL is **100% free and open-source (MIT)**.  
The code will always remain free. To fund development (and future systems), several options exist:

### 1. Crypto donation (fast & direct)

**Solana (SOL · USDC · USDT)**

```
GVEk2qeagteALBLcsVgr52b2pj3hSjvdsh2dMQ8SFVrN
```

<p align="center">
  <img src="donate-qr.png"
      alt="Solana donation QR code (USDC / USDT / SOL)" width="220">
</p>

<p align="center"><em>Scan → send what you want → it helps grow AQOL 🚀</em></p>

### 2. GitHub Sponsors 



### 3. Services & consulting  

If you use AQOL in production or in the lab and need:
- adaptation to a specific problem (quantum, calibration, multi-objective…)
- integration into an existing pipeline
- training / consulting

→ contact me via GitHub Discussions or open a “Consulting” issue.

### 4. Future options

- Open Collective / Liberapay (transparency)
- Bounties on priority features
- Lab / quantum startup partnerships

Any support (even symbolic) funds:
- development time
- tests on real hardware
- evolution toward **AQOL-AI**

Thank you so much 💚

---

## Contributing

**Pull requests**, **issues**, and **real-world feedback** (especially on expensive use cases) are very welcome.

You can also help without coding:
- ⭐ Star the repository
- Share AQOL in your community (quantum, lab, optimization)
- Report a bug or suggest a feature
- Write an example notebook

---

## License

MIT — see [LICENSE](LICENSE).

You are free to use, modify, and redistribute AQOL, including in commercial projects.

---

*Every avoided measurement is a bit of electricity saved and a bit more frugal science.*

---

### Recommended GitHub Topics (Settings → Topics)
`bayesian-optimization` · `gaussian-process` · `quantum-computing` · `energy-efficiency` · `green-computing` · `sample-efficiency` · `expensive-function` · `noisy-optimization` · `scientific-computing` · `optimization` · `python`
