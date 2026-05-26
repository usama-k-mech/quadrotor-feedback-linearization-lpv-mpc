# quadrotor-feedback-linearization-lpv-mpc
6-DOF quadrotor simulation with a two-level cascade controller: feedback linearisation outer loop for position tracking and qLPV-MPC inner loop for attitude control. Built on full Newton-Euler dynamics, RK4 integration, aerodynamic drag, and gyroscopic rotor coupling. Validated on figure-8, yaw-sweep, and noise robustness scenarios.

---

## What This Demonstrates
- **Feedback Linearisation (Outer Loop)**: exact input-output linearisation, pole-placement gains, no small-angle assumption, runs at 5 Hz
- **qLPV-MPC (Inner Loop)**: scheduling on Euler rates, ZOH discretisation, incremental (Δu) formulation with integral action, DARE terminal weight, runs at 20 Hz
- **Full Newton-Euler Plant**: 12-state rigid-body model, RK4 integration, gyroscopic rotor coupling (J_tp · Ω_net), quadratic aerodynamic drag
- **Constrained QP**: absolute torque limits + increment rate limits, solved via `quadprog` (active-set) with `scipy` L-BFGS-B fallback
- **Trajectory Generator**: 7th-order minimum-jerk polynomial ramp, Lissajous figure-8, yaw-sweep variant
- **Noise Robustness**: stress-tested at noise_std = 5e-2 (uniform per-state), zero rotor saturations maintained

---

## Results

### Nominal Figure-8 (drag on, no noise, 40 s)

| Metric | Value |
|--------|-------|
| Position RMSE | 0.0075 m |
| Position max error | 0.0307 m |
| Attitude RMSE | 0.080 deg |
| Attitude max error | 0.651 deg |
| Control effort (mean torque norm) | 0.00055 N·m |
| Rotor saturations | 0 / 801 steps (0.0%) |

### Noise Robustness (noise_std = 5e-2, figure-8, 40 s)

| Metric | Nominal | Stress (noise_std=5e-2) |
|--------|---------|------------------------|
| Position RMSE | 0.0075 m | 0.0097 m |
| Position max | 0.0307 m | 0.0323 m |
| Rotor saturations | 0 (0.0%) | 0 (0.0%) |

---

## Visualizations

![Figure-8 Nominal](figures/quad_results_figure8.png)
![Yaw Sweep](figures/quad_results_yaw.png)
![Noise Robustness](figures/quad_results_noise.png)

---

## Controller Architecture

```
trajectory     ┌─────────────────────────────────────────┐
reference  ──► │  OUTER LOOP  (5 Hz)                     │
               │  PositionController                      │
               │  Feedback linearisation + pole placement │
               │  Output: φ_ref, θ_ref, U1               │
               └──────────────────┬──────────────────────┘
                                  │
               ┌──────────────────▼──────────────────────┐
               │  INNER LOOP  (20 Hz)                    │
               │  LPVMPCController                        │
               │  qLPV model · ZOH · Δu form · DARE      │
               │  Output: U2, U3, U4                      │
               └──────────────────┬──────────────────────┘
                                  │
               ┌──────────────────▼──────────────────────┐
               │  MIXER  (constant, precomputed)          │
               │  [ω1,ω2,ω3,ω4] = M⁻¹ · U               │
               └─────────────────────────────────────────┘
```

---

## Files

- `utils.py` — rotation matrices, propulsion mixer, aerodynamic drag, trajectory generators
- `dynamics.py` — 6-DOF Newton-Euler plant, RK4 integrator, `QuadParams` dataclass
- `controllers.py` — `PositionController` (feedback linearisation) + `LPVMPCController` (qLPV-MPC)
- `simulate.py` — end-to-end simulation runner, plotting, CLI entry point

---

## Usage

```bash
python simulate.py                  # figure-8, drag on, no noise
python simulate.py --hover          # hover at (0, 0, -1.5) m
python simulate.py --yaw            # figure-8 with yaw sweep
python simulate.py --no-drag        # disable aerodynamic drag
python simulate.py --noise 1e-3     # add process noise σ=1e-3
python simulate.py --duration 60    # run for 60 seconds
python simulate.py --no-plot        # skip matplotlib output
```

---

## Dependencies

```bash
pip install numpy scipy matplotlib quadprog
```

`quadprog` is optional but recommended — without it the solver falls back to `scipy` L-BFGS-B, which is slower and encodes absolute input constraints as a conservative box approximation rather than exact linear constraints.

---

## References

1. Beard & McLain, *Small Unmanned Aircraft*, Princeton UP, 2012
2. Mahony, Müller, Corke, *Multirotor Aerial Vehicles*, IEEE RA-M, 2012
3. Camacho & Bordons, *Model Predictive Control*, Springer, 2004
4. Rugh & Shamma, *Research on gain scheduling*, Automatica, 2000
5. Rawlings & Mayne, *Model Predictive Control: Theory and Design*, 2009
