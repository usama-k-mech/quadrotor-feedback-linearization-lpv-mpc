"""
simulate.py  ―  Quadcopter Cascade Control Simulation
======================================================

End-to-end simulation demonstrating trajectory tracking with the
two-level cascade controller:

  Outer loop : PositionController  (feedback linearisation)
  Inner loop : LPVMPCController    (LPV-MPC)
  Plant      : QuadDynamics        (6-DOF Newton-Euler, RK4)

Timing
------
  dt_mpc         = 0.05 s   inner-loop (MPC) sample period
  outer_ratio    = 4        outer step runs every 4 inner steps
  dt_outer       = 0.20 s   outer-loop sample period

Usage
-----
  python simulate.py                  # figure-8, drag on, no noise
  python simulate.py --hover          # hover at (0,0,-1.5)
  python simulate.py --no-drag        # disable aerodynamic drag
  python simulate.py --noise 1e-3     # add process noise σ=1e-3
  python simulate.py --duration 60    # run for 60 seconds
  python simulate.py --no-plot        # skip matplotlib output

Output
------
  Console  : step-by-step telemetry + final performance metrics
  File     : quad_results.png  (position, attitude, controls, 3-D path)
"""

from __future__ import annotations

import argparse
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from dynamics    import QuadDynamics, QuadParams
from controllers import PositionController, PosCtrlParams, LPVMPCController, MPCParams
from utils       import traj_hover, traj_figure8, traj_figure8_yaw, Waypoint


# ════════════════════════════════
#  Simulation runner
# ════════════════════════════════

def run(
    scenario:    str   = 'figure8',
    enable_drag: bool  = True,
    noise_std:   float = 0.0,
    duration:    float = 40.0,
    verbose:     bool  = True,
) -> dict:
    """
    Run the full cascade-control simulation.

    Parameters
    ----------
    scenario    : 'figure8', 'figure8_yaw', or 'hover'
    enable_drag : Toggle aerodynamic drag in the plant
    noise_std   : Per-state process-noise magnitude (0 = none).
                  Applied uniformly — not sensor-realistic.
                  Use 1e-3 for mild robustness testing, 5e-2 for stress testing.
    duration    : Simulation end time [s]
    verbose     : Print telemetry to console

    Returns
    -------
    dict with keys: t, states, refs, U, omegas, sat
    """

    # ── Timing ────────────────────────────────────────────────────────────────
    dt_mpc      = 0.05          # inner-loop (MPC) step [s]
    outer_ratio = 4             # outer runs every this many inner steps
    log_every   = 40            # console print interval (inner steps)
    T_lead_in   = 4.0           # seconds of hover before dynamic trajectory
    n_steps     = int(duration / dt_mpc) + 1

    # ── Physics parameters ────────────────────────────────────────────────────
    p = QuadParams(drag_on=enable_drag)

    # Trim condition
    omega_h = p.hover_omega()
    U1_h    = p.hover_thrust()

    if verbose:
        print(f"  Hover rotor speed : {omega_h:.2f} rad/s  "
              f"({omega_h*60/(2*np.pi):.1f} RPM)")
        print(f"  Hover thrust U1*  : {U1_h:.4f} N  (= {p.mass:.3f} × {p.g:.5f})")

    # ── Initial state  (at rest, z = −1.5 m in NED = 1.5 m above ground) ─────
    x = np.zeros(12)
    x[8] = -1.5          # NED z-position  [m]

    # ── Controllers ───────────────────────────────────────────────────────────
    # Outer-loop pole selection
    # ─────────────────────────────────────────────────────────────────────────
    # Poles at −1.5 ± 0.5j give ω_n ≈ 1.58 rad/s, ζ ≈ 0.95 (near-critical).
    # More aggressive poles (e.g. −3) produce larger initial angle demands that
    # can saturate the inner loop and cause pitch divergence during the
    # velocity-step transient at trajectory start.  The cascade timescale
    # rule-of-thumb is:  |Re(outer poles)| ≤ |Re(inner poles)| / 5.
    pos_ctrl = PositionController(
        p,
        PosCtrlParams(
            poles_x = (-1.5 + 0.5j, -1.5 - 0.5j),
            poles_y = (-1.5 + 0.5j, -1.5 - 0.5j),
            poles_z = (-2.0 + 0.3j, -2.0 - 0.3j),
        ),
    )

    att_ctrl = LPVMPCController(
        p,
        MPCParams(
            horizon  = 8,
            Q_diag   = np.array([20.0, 20.0, 10.0]),
            S_diag   = np.array([40.0, 40.0, 20.0]),
            R_diag   = np.array([ 5.0,  5.0, 12.0]),
            U2_max   = 0.45, U3_max = 0.45, U4_max = 0.12,
            dU2_max  = 0.18, dU3_max= 0.18, dU4_max= 0.05,
        ),
        dt = dt_mpc,
    )

    # ── Plant ─────────────────────────────────────────────────────────────────
    plant = QuadDynamics(p)

    # ── Process noise (per-state σ vector, scaled by noise_std) ──────────────
    # noise_std is applied uniformly across all noisy states with equal weight.
    # This is NOT a sensor-realistic model — it is a robustness stress test.
    # At noise_std=5e-2 each noisy state receives σ=0.05 per sqrt(s), which
    # is intentionally large (≈3° on Euler angles) to verify closed-loop
    # stability under severe disturbance.  For sensor-realistic noise, replace
    # the 1.0 weights with physically motivated per-state scaling, e.g.:
    #   body velocities    : 1e-2   (≈ accelerometer integration noise)
    #   body angular rates : 5e-4   (≈ MEMS gyro noise density)
    #   Euler angles       : 5e-4   (≈ attitude estimator output noise)
    if noise_std > 0.0:
        proc_noise = noise_std * np.array([
            1.0, 1.0, 1.0,    # body velocities        [m/s]
            1.0, 1.0, 1.0,    # body angular rates     [rad/s]
            0.0, 0.0, 0.0,     # positions           (no position noise)
            1.0, 1.0, 1.0,    # Euler angles           [rad]
        ])
    else:
        proc_noise = None

    # ── Logging ───────────────────────────────────────────────────────────────
    t_log     = np.zeros(n_steps)
    state_log = np.zeros((n_steps, 12))
    ref_log   = np.zeros((n_steps, 6))   # [x,y,z, φ,θ,ψ]_ref
    U_log     = np.zeros((n_steps, 4))
    omega_log = np.zeros((n_steps, 4))
    sat_log   = np.zeros(n_steps, dtype=bool)

    # ── Main loop ─────────────────────────────────────────────────────────────
    U       = np.array([U1_h, 0.0, 0.0, 0.0])
    omega   = np.full(4, omega_h)
    omega_n = 0.0                            # Ω_net = 0 at symmetric hover
    phi_r   = theta_r = psi_r = 0.0

    t_wall0 = time.perf_counter()

    for k in range(n_steps):
        t = k * dt_mpc

        # ── Reference trajectory ──────────────────────────────────────────
        # A hover lead-in phase lets the inner MPC establish attitude trim
        # before the dynamic trajectory begins, preventing the large velocity
        # step at t=0 from driving the outer loop to demand extreme angles.
        if scenario in ('figure8', 'figure8_yaw'):
            if t < T_lead_in:
                wp: Waypoint = traj_hover(np.array([0.0, 0.0, -1.5]), t=t)
            else:
                # ── 7th-order minimum-jerk polynomial ramp ────────────────────
                # Smoothly scales trajectory amplitude from 0 → 1 over T_ramp s.
                #
                # The 7th-order polynomial  s(τ) = 35τ⁴ − 84τ⁵ + 70τ⁶ − 20τ⁷
                # satisfies:  s(0)=0, s(1)=1
                #             s'(0)=s'(1)=0   (velocity continuous)
                #             s''(0)=s''(1)=0  (acceleration continuous)
                #             s'''(0)=s'''(1)=0 (jerk continuous)
                # → zero velocity, acceleration, AND jerk at both endpoints.
                #
                # A 5th-order polynomial (s = 10τ³−15τ⁴+6τ⁵) satisfies the
                # vel/acc boundary conditions but has s'''(0) = 60 ≠ 0, leaving
                # a jerk step at τ=0 that excites attitude transients.  The
                # 7th-order form eliminates this by adding the jerk constraint,
                # at the cost of two extra polynomial terms.
                #
                # T_ramp=4.0s: long enough that the peak velocity demand stays
                # well within the outer-loop bandwidth, avoiding large angle
                # commands from the feedback-linearisation inversion.
                T_ramp = 4.0
                tau = min((t - T_lead_in) / T_ramp, 1.0)
                # 7th-order minimum-jerk polynomial (zero vel/acc/jerk at endpoints)
                # s(τ)   = 35τ⁴ − 84τ⁵ + 70τ⁶ − 20τ⁷
                # s'(0)=s'(1)=0, s''(0)=s''(1)=0, s'''(0)=s'''(1)=0  ✓
                # The 5th-order cosine ramp had s'''(0)=60 ≠ 0 (jerk discontinuous).
                s   = 35*tau**4 - 84*tau**5 + 70*tau**6 - 20*tau**7
                sd  = (140*tau**3 - 420*tau**4 + 420*tau**5 - 140*tau**6) / T_ramp
                sdd = (420*tau**2 - 1680*tau**3 + 2100*tau**4 - 840*tau**5) / T_ramp**2
                if scenario == 'figure8_yaw':
                    wp_full = traj_figure8_yaw(t - T_lead_in, A=2.0, B=1.0,
                                               z0=-1.5, omega=0.25, yaw_rate=0.05)
                else:
                    wp_full = traj_figure8(t - T_lead_in, A=2.0, B=1.0,
                                           z0=-1.5, omega=0.25)
                hover_pos = np.array([0.0, 0.0, -1.5])
                # Apply chain rule: d/dt[s·f] = s'·f + s·f'
                # Yaw is also ramped smoothly from 0 to wp_full.psi
                wp = Waypoint(
                    t       = wp_full.t,
                    pos     = s   * wp_full.pos  + (1-s) * hover_pos,
                    vel     = sd  * (wp_full.pos - hover_pos) + s * wp_full.vel,
                    acc     = sdd * (wp_full.pos - hover_pos)
                              + 2*sd * wp_full.vel
                              + s    * wp_full.acc,
                    psi     = s  * wp_full.psi,          # ramp yaw from 0 → wp_full.psi
                    psi_dot = sd * wp_full.psi + s * wp_full.psi_dot,  # chain rule
                )
        else:
            wp = traj_hover(np.array([0.0, 0.0, -1.5]), t=t)

        # ── Outer loop (position controller) — updated every outer_ratio steps ─
        if k % outer_ratio == 0:
            phi_r, theta_r, U1 = pos_ctrl.compute(
                wp.pos, wp.vel, wp.acc, wp.psi, x
            )
            U[0] = U1

        # ── Inner loop (LPV-MPC) ─────────────────────────────────────────
        psi_r  = wp.psi   # track waypoint yaw (0 for figure8, sweeping for figure8_yaw)
        U_att  = att_ctrl.compute(x, omega_n, phi_r, theta_r, psi_r)
        U[1:4] = U_att

        # ── Mixer ─────────────────────────────────────────────────────────
        omega, saturated = plant.controls_to_rotors(U)
        omega_n = omega[0] - omega[1] + omega[2] - omega[3]

        # ── Log ───────────────────────────────────────────────────────────
        t_log[k]      = t
        state_log[k]  = x.copy()
        ref_log[k]    = [wp.pos[0], wp.pos[1], wp.pos[2], phi_r, theta_r, psi_r]
        U_log[k]      = U.copy()
        omega_log[k]  = omega.copy()
        sat_log[k]    = saturated

        # ── Integrate plant ───────────────────────────────────────────────
        try:
            x = plant.step_rk4(x, U, omega_n, dt_mpc, noise_std=proc_noise)
        except ValueError as e:
            # Gimbal-lock detected (|θ| >= 85°).
            # Recovery strategy (W8 fix):
            #   1. Zero all attitude-control torques.
            #   2. Keep thrust at m*g to prevent free-fall.
            #   3. Allow one free step — gravity pulls nose back to level.
            # If theta is still >= 85° after one recovery step, the
            # simulation is physically lost; terminate to avoid bad data.
            print("  [t={:.2f}s] WARNING: Gimbal-lock — attempting recovery.".format(t))
            U_rec = np.array([plant.p.hover_thrust(), 0.0, 0.0, 0.0])
            try:
                x_r = plant.step_rk4(x, U_rec, 0.0, dt_mpc)
                if abs(x_r[10]) < np.radians(85.0):
                    x = x_r
                    print("  [t={:.2f}s] Recovery OK — |theta|={:.1f} deg.".format(
                        t, np.degrees(abs(x_r[10]))))
                else:
                    print("  [t={:.2f}s] Recovery FAILED. Terminating.".format(t))
                    n_steps = k + 1
                    break
            except Exception:
                print("  [t={:.2f}s] Recovery step failed. Terminating.".format(t))
                n_steps = k + 1
                break

        # ── Console telemetry ─────────────────────────────────────────────
        if verbose and k % log_every == 0:
            pos = state_log[k, 6:9]
            eul = np.degrees(state_log[k, 9:12])
            err = np.linalg.norm(wp.pos - pos)
            sat_str = 'SAT' if saturated else '   '
            print(
                "t={:6.2f}s | pos=[{:+5.2f},{:+5.2f},{:+5.2f}] | "
                "Euler=[{:+5.1f},{:+5.1f},{:+5.1f}]deg | "
                "|e|={:.3f}m | U1={:.2f}N | {}".format(
                    t, pos[0], pos[1], pos[2],
                    eul[0], eul[1], eul[2],
                    err, U[0], sat_str
                )
            )

    t_wall = time.perf_counter() - t_wall0
    if verbose:
        print(f"\n  Completed {n_steps} steps in {t_wall:.2f} s "
              f"({n_steps/t_wall:.0f} steps/s)")

    return dict(
        t      = t_log[:n_steps],
        states = state_log[:n_steps],
        refs   = ref_log[:n_steps],
        U      = U_log[:n_steps],
        omegas = omega_log[:n_steps],
        sat    = sat_log[:n_steps],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(data: dict, title: str = "", out_file: str = "quad_results.png") -> None:
    """Six-panel figure: position, attitude, controls, rotor speeds, 3-D path."""
    t, st, rf, U, om = (
        data['t'], data['states'], data['refs'], data['U'], data['omegas']
    )
    p_params = QuadParams()

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(title or "Quadcopter LPV-MPC Cascade Controller",
                 fontsize=13, fontweight='bold')
    gs = GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)

    # ── Position (row 0) ───────────────────────────────────────────────────
    for i, lbl in enumerate(['x North [m]', 'y East [m]', 'z Down [m]']):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(t, rf[:, i],     '--r', lw=1.5, label='ref')
        ax.plot(t, st[:, 6 + i], '-b',  lw=1.2, label='actual')
        ax.set_xlabel('t [s]', fontsize=8); ax.set_ylabel(lbl, fontsize=8)
        ax.set_title(f'Position: {lbl}', fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Attitude (row 1) ───────────────────────────────────────────────────
    for i, lbl in enumerate(['φ roll [°]', 'θ pitch [°]', 'ψ yaw [°]']):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(t, np.degrees(rf[:, 3 + i]),  '--r', lw=1.5, label='ref')
        ax.plot(t, np.degrees(st[:, 9 + i]),  '-g',  lw=1.2, label='actual')
        ax.set_xlabel('t [s]', fontsize=8); ax.set_ylabel(lbl, fontsize=8)
        ax.set_title(f'Attitude: {lbl}', fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Controls U1–U4 (rows 2–3) ─────────────────────────────────────────
    ctrl_specs = [
        (gs[2, 0], 0, 'U1 thrust [N]',        'tab:blue'),
        (gs[2, 1], 1, 'U2 roll torque [N·m]', 'tab:orange'),
        (gs[2, 2], 2, 'U3 pitch torque [N·m]','tab:green'),
        (gs[3, 0], 3, 'U4 yaw torque [N·m]',  'tab:red'),
    ]
    for (spec, idx, lbl, col) in ctrl_specs:
        ax = fig.add_subplot(spec)
        ax.plot(t, U[:, idx], color=col, lw=1.2)
        ax.set_xlabel('t [s]', fontsize=8); ax.set_ylabel(lbl, fontsize=8)
        ax.set_title(lbl, fontsize=9); ax.grid(alpha=0.3)

    # ── Rotor speeds (row 3, col 1) ────────────────────────────────────────
    ax_om = fig.add_subplot(gs[3, 1])
    omega_h = p_params.hover_omega()
    d_om = om - omega_h                         # delta from hover [rad/s]
    for i, lb in enumerate(['ω1(CCW)', 'ω2(CW)', 'ω3(CCW)', 'ω4(CW)']):
        ax_om.plot(t, d_om[:, i], lw=1.0, label=lb)
    ax_om.axhline(0.0, color='k', ls='--', lw=0.8, label='hover')
    ax_om.set_xlabel('t [s]', fontsize=8)
    ax_om.set_ylabel('Δω [rad/s]', fontsize=8)
    ax_om.set_title(f'Rotor speed deviation from hover ({omega_h:.1f} rad/s)', fontsize=9)
    ax_om.legend(fontsize=6, ncol=2); ax_om.grid(alpha=0.3)
    

    # ── 3-D trajectory (row 3, col 2) ─────────────────────────────────────
    ax3d = fig.add_subplot(gs[3, 2], projection='3d')
    # Trim lead-in (t<4s hover phase) so 3D plot shows only the figure-8 loop
    trim = t > 4.0
    ax3d.plot(st[trim, 6], st[trim, 7], -st[trim, 8], 'b-',  lw=1.2, label='actual')
    ax3d.plot(rf[trim, 0], rf[trim, 1], -rf[trim, 2], 'r--', lw=0.9, label='ref', alpha=0.7)
    ax3d.set_xlabel('x [m]', fontsize=7); ax3d.set_ylabel('y [m]', fontsize=7)
    ax3d.set_zlabel('altitude [m]', fontsize=7)
    ax3d.set_title('3-D trajectory', fontsize=9); ax3d.legend(fontsize=7)

    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"  Plot saved → {out_file}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _parse():
    ap = argparse.ArgumentParser(description='Quadcopter LPV-MPC Simulation')
    ap.add_argument('--hover',    action='store_true', help='Hover scenario')
    ap.add_argument('--yaw',      action='store_true', help='Figure-8 with yaw sweep')
    ap.add_argument('--no-drag',  action='store_true', help='Disable drag')
    ap.add_argument('--noise',    type=float, default=0.0, metavar='σ')
    ap.add_argument('--duration', type=float, default=40.0, metavar='T')
    ap.add_argument('--no-plot',  action='store_true')
    args, _unknown = ap.parse_known_args()
    return args



# ═══════════════════════════════════════════════════════════════════════════════
#  JUPYTER / IPYTHON USAGE
# ═══════════════════════════════════════════════════════════════════════════════
#
#  This file works both as a script (python simulate.py) and inside Jupyter.
#  In Jupyter, ignore the __main__ block below and call run() directly:
#
#  ┌─ Jupyter cell ──────────────────────────────────────────────────────────┐
#  │  from simulate import run, plot_results                                 │
#  │                                                                         │
#  │  # Figure-8 trajectory (default, 40 s, drag on)                        │
#  │  data = run()                                                           │
#  │  plot_results(data)                                                     │
#  │                                                                         │
#  │  # Hover scenario, no drag, 20 s                                        │
#  │  data = run(scenario='hover', enable_drag=False, duration=20.0)         │
#  │                                                                         │
#  │  # With process noise                                                   │
#  │  data = run(noise_std=1e-3, duration=30.0)                              │
#  │                                                                         │
#  │  # Access results directly                                              │
#  │  import numpy as np                                                     │
#  │  t, states, refs = data['t'], data['states'], data['refs']              │
#  │  pos_error = np.linalg.norm(refs[:,:3] - states[:,6:9], axis=1)        │
#  └─────────────────────────────────────────────────────────────────────────┘

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    args = _parse()

    scenario = 'hover' if args.hover else ('figure8_yaw' if args.yaw else 'figure8')
    print(f"\n{'='*62}")
    print(f"  Quadcopter LPV-MPC Cascade Control Simulation")
    print(f"  Scenario : {scenario}")
    print(f"  Drag     : {'OFF' if args.no_drag else 'ON'}")
    print(f"  Noise σ  : {args.noise}")
    print(f"  Duration : {args.duration} s")
    print(f"{'='*62}\n")

    data = run(
        scenario    = scenario,
        enable_drag = not args.no_drag,
        noise_std   = args.noise,
        duration    = args.duration,
        verbose     = True,
    )

    # ── Performance metrics ────────────────────────────────────────────────
    t, st, rf = data['t'], data['states'], data['refs']
    e_pos = np.linalg.norm(rf[:, :3] - st[:, 6:9], axis=1)
    e_att = np.degrees(np.linalg.norm(rf[:, 3:] - st[:, 9:12], axis=1))

    # Settled window: after lead-in + ramp (t > 8s)
    settled = t > 8.0
    e_pos_s = e_pos[settled]
    e_att_s = e_att[settled]
    U_s     = data['U'][settled]

    rmse_pos = float(np.sqrt(np.mean(e_pos_s**2)))
    rmse_att = float(np.sqrt(np.mean(e_att_s**2)))
    effort   = float(np.mean(np.linalg.norm(U_s[:, 1:], axis=1)))  # torque norm

    print(f"\n── Performance Metrics (settled window t>8s) ────────────")
    print(f"  Position RMSE   : {rmse_pos:.4f} m")
    print(f"  Position max    : {e_pos_s.max():.4f} m")
    print(f"  Attitude RMSE   : {rmse_att:.3f} deg")
    print(f"  Attitude max    : {e_att_s.max():.3f} deg")
    print(f"  Control effort  : {effort:.5f} N·m  (mean torque norm)")
    print(f"  Rotor saturations : {data['sat'].sum()} steps "
          f"({100*data['sat'].mean():.1f}%)")

    if not args.no_plot:
        plot_results(
            data,
            title=f"[{scenario}] Quadcopter LPV-MPC  "
                  f"(drag={'on' if not args.no_drag else 'off'}, "
                  f"noise={args.noise})",
            out_file=f"quad_results_{scenario}.png"
        )

    # ── Additional validation runs ─────────────────────────────────────────
    if not args.no_plot and scenario == 'figure8':
        
        # Yaw sweep 
        print(f"\n{'='*62}")
        print(f"  Running yaw-sweep validation (figure8_yaw, 40s)")
        print(f"{'='*62}\n")
        data_yaw = run(scenario='figure8_yaw', enable_drag=not args.no_drag,
                       noise_std=args.noise, duration=args.duration, verbose=False)
        plot_results(data_yaw,
                     title="[figure8_yaw] Quadcopter LPV-MPC (yaw channel test)",
                     out_file="quad_results_yaw.png")

        # Noise robustness
        print(f"\n{'='*62}")
        print(f"  Running noise robustness test (noise_std=5e-2, 40s)")
        print(f"{'='*62}\n")
        data_noise = run(scenario='figure8', enable_drag=not args.no_drag,
                         noise_std=5e-2, duration=args.duration, verbose=False)

        t_n, st_n, rf_n = data_noise['t'], data_noise['states'], data_noise['refs']
        settled_n = t_n > 8.0
        e_n = np.linalg.norm(rf_n[settled_n, :3] - st_n[settled_n, 6:9], axis=1)
        print(f"  Noise run — Position RMSE: {np.sqrt(np.mean(e_n**2)):.4f} m  "
              f"max: {e_n.max():.4f} m  "
              f"saturations: {data_noise['sat'].sum()}")
        plot_results(data_noise,
                     title="[figure8] Quadcopter LPV-MPC (noise_std=5e-2)",
                     out_file="quad_results_noise.png")