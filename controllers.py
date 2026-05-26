"""
controllers.py  ―  Feedback Linearisation + LPV-MPC Attitude Controller
========================================================================

Implements a two-level cascade controller for quadrotor trajectory tracking.

Architecture
────────────
                 ┌───────────────────────────────────────────────┐
  trajectory     │  OUTER LOOP  (runs every  T_outer = N_ratio·dt)│
  reference  ──► │  PositionController  (feedback linearisation)  │
                 │  Input : pos/vel/acc ref + full state           │
                 │  Output: φ_ref, θ_ref, U1                      │
                 └──────────────────┬────────────────────────────┘
                                    │ φ_ref, θ_ref, ψ_ref, U1
                 ┌──────────────────▼────────────────────────────┐
                 │  INNER LOOP  (runs every  dt_mpc)             │
                 │  Quasi LPVMPCController  (qLPV-MPC)                  │
                 │  Input : angle refs + full state               │
                 │  Output: U2, U3, U4                            │
                 └──────────────────┬────────────────────────────┘
                                    │ U = [U1,U2,U3,U4]
                 ┌──────────────────▼────────────────────────────┐
                 │  MIXER  (constant, precomputed)                │
                 │  [ω1,ω2,ω3,ω4] = M⁻¹ · U, clamped to limits  │
                 └───────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
  MODULE 1 — PositionController  (feedback linearisation)
═══════════════════════════════════════════════════════════════════

Theory
------
The NED translational dynamics after expressing body-frame thrust in
world coordinates are:

    ẍ = (U1/m)(cψsθcφ + sψsφ)
    ÿ = (U1/m)(sψsθcφ − cψsφ)
    z̈ = −g + (U1/m)cθcφ           [NED: z positive downward → −g + thrust/m]

These are coupled, nonlinear in the angles.  Exact input-output feedback
linearisation introduces virtual inputs v = [vx, vy, vz]:

    vi = ẍ_ref,i  +  k1_i·(pos_ref,i − pos_i)  +  k2_i·(vel_ref,i − vel_i)

which reduces each axis to  ẍ_i = v_i  (decoupled double integrators).

Gain design via pole placement: closed-loop characteristic polynomial
    s² + k2·s + k1 = (s − p1)(s − p2)
gives
    k1 = Re(p1·p2)     k2 = −Re(p1+p2)
with p1,p2 chosen in the open left-half plane.

Angle inversion (exact, no small-angle assumption)
---------------------------------------------------
The desired thrust direction in world frame is:
    T̂_W = [−vx, −vy, −(g−vz)] / ‖…‖        [NED sign: g−vz ≥ 0 for hover]
              ↑ in x        ↑ in y    ↑ upward

After rotating by the desired yaw ψ_ref, the desired pitch and roll are:
    sin(θ_ref) = −(cos(ψ)·T̂_Wx + sin(ψ)·T̂_Wy)
    sin(φ_ref) = (sin(ψ)·T̂_Wx − cos(ψ)·T̂_Wy) / cos(θ_ref)

These formulas are valid for any ψ and for |θ| < 90°.
No singularity with respect to ψ; the earlier code had a switching
formula to avoid cos(ψ)≈0 or sin(ψ)≈0 which is not needed here.

═══════════════════════════════════════════════════════════════════
  MODULE 2 — LPVMPCController  (LPV model + MPC optimisation)
═══════════════════════════════════════════════════════════════════

LPV model
---------
State  x_att = [φ, φ̇, θ, θ̇, ψ, ψ̇]^T ∈ ℝ⁶
Input  u_att = [U2, U3, U4]^T ∈ ℝ³
Output y     = [φ, θ, ψ]^T  ∈ ℝ³

Continuous-time model   ẋ_att = A(σ)·x_att + B·u_att,   y = C·x_att

Scheduling variables σ = (φ̇, θ̇, ψ̇, Ω_net) are the Euler rates,
consistent with the LPV state vector x_att which contains Euler rates,
not body rates.  At small angles (±5°) the difference is negligible;
at ±20°+ body-rate scheduling introduces O(sin20°) ≈ 34% cross-coupling
error in the A(σ) entries.  Scheduling on Euler rates reduces this to
O(angle²) across the operating envelope.

Non-zero entries of A(σ):
  A[0,1] = 1   A[2,3] = 1   A[4,5] = 1           (kinematics)

  φ̈ row — full eq: ṗ = θ̇·ψ̇·(Iy−Iz)/Ix − (J_tp/Ix)·θ̇·Ω
  Bilinear θ̇·ψ̇ factored as  theta_d·ψ̇ + psi_d·θ̇  (both halves):
    A[1,3] = theta_d·(Iy−Iz)/Ix − J_tp·Ω_net/Ix    (φ̈←θ̇: Euler + gyro)
    A[1,5] = psi_d·(Iy−Iz)/Ix                       (φ̈←ψ̇: Euler)

  θ̈ row — full eq: q̇ = φ̇·ψ̇·(Iz−Ix)/Iy + (J_tp/Iy)·φ̇·Ω
  Bilinear φ̇·ψ̇ factored as  phi_d·ψ̇ + psi_d·φ̇  (both halves):
    A[3,1] = phi_d·(Iz−Ix)/Iy + J_tp·Ω_net/Iy      (θ̈←φ̇: Euler + gyro)
    A[3,5] = psi_d·(Iz−Ix)/Iy                       (θ̈←ψ̇: Euler)

  ψ̈ row — full eq: ṙ = φ̇·θ̇·(Ix−Iy)/Iz
  Bilinear φ̇·θ̇ factored as  theta_d·φ̇ + phi_d·θ̇  (both halves):
    A[5,1] = theta_d·(Ix−Iy)/Iz                     (ψ̈←φ̇: sched on θ̇)
    A[5,3] = phi_d·(Ix−Iy)/Iz                       (ψ̈←θ̇: sched on φ̇)

B[1,0]=1/Ix  B[3,1]=1/Iy  B[5,2]=1/Iz
C = diag-identity selecting rows 0,2,4  → [φ,θ,ψ]

Discretisation: Zero-Order Hold (exact for piecewise-constant inputs).
Forward-Euler is NOT used (it can make the discrete model unstable).

Incremental (Δu) form
---------------------
Augmented state  x̃ = [x_att; u_prev] ∈ ℝ⁹

    x̃(k+1) = Ã·x̃(k) + B̃·Δu(k)
    y(k)   = C̃·x̃(k)

    Ã = |Ad  Bd|   B̃ = |Bd|   C̃ = [Cd  0]
        | 0   I|       | I|

This introduces integral action, removing steady-state tracking offset.

Condensed (batch) QP
--------------------
Over horizon N, the prediction is:
    Y = Ψ·x̃(k) + Θ·ΔŪ

Cost function (N steps, output weighting Q_bar, terminal weight S, input-rate R):
    J = (Y − R_ref)^T Q̄ (Y − R_ref) + ΔŪ^T R̄ ΔŪ
      = ΔŪ^T H ΔŪ + f^T ΔŪ + const

    H = Θ^T Q̄ Θ + R̄          (always PD if R̄ > 0)
    f = 2·Θ^T Q̄·(Ψ·x̃ − R_ref)

Constraints
-----------
Box constraints on input increments:    ΔU_lb ≤ Δu(k+i) ≤ ΔU_ub
Box constraints on absolute inputs:     U_lb  ≤ u_prev + Σ Δu ≤ U_ub
  (the second set is enforced via the cumulative-sum matrix L)

The resulting bound-constrained QP is solved with quadprog (active-set,
exact Hessian) when available, falling back to scipy L-BFGS-B otherwise.
quadprog solves:  min  0.5·x'Gx − a'x   s.t.  C'x ≥ b
which maps to:    G=H, a=−f, with box constraints as C'=[I;−I], b=[lb;−ub].
When the absolute-input bounds and increment bounds conflict (can occur
when u_prev is near a limit), the infeasibility is resolved by widening
the increment upper bound to match the lower bound (zero-increment fallback
for that step), which is safe because the Δu = 0 solution keeps u = u_prev,
a feasible absolute value by construction.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
try:
    import quadprog as _quadprog
    _QUADPROG_AVAILABLE = True
except ImportError:
    _QUADPROG_AVAILABLE = False

from dynamics import QuadParams
from utils import R_body_to_world, euler_rate_matrix


# ═════════════════════════════════════════════════════════════
#  MODULE 1 — Position Feedback Linearisation (Outer Loop)
# ═════════════════════════════════════════════════════════════

@dataclass
class PosCtrlParams:
    """
    Closed-loop pole locations for each translational axis.

    Choose poles with negative real parts.  The damped natural frequency
    determines how quickly errors are corrected; the imaginary part sets the
    damping ratio.  A reasonable starting point for a 0.7 kg platform:
        ω_n ≈ 2–4 rad/s,   ζ ≈ 0.7–1.0  →  poles at −ω_n·ζ ± j·ω_n·√(1−ζ²)

    Cascade timescale rule:
        |Re(outer poles)| ≤ |Re(inner poles)| / 5
    With inner-loop MPC bandwidth ~8 rad/s, outer poles should stay ≤ 1.6
    on x/y.  The z-axis can be slightly more aggressive because altitude
    is decoupled from roll/pitch and does not risk saturating the inner loop.

    Default values are validated against the figure-8 trajectory in
    simulate.py (drag on, no noise, 40 s).  More aggressive poles are
    possible but require re-validating inner-loop saturation margins.

    Outer-loop sample period should be 3–5× slower than the inner-loop MPC
    to respect the cascade timescale assumption.
    """
    poles_x: Tuple = (-1.5 + 0.5j, -1.5 - 0.5j)   # ωn≈1.58, ζ≈0.95
    poles_y: Tuple = (-1.5 + 0.5j, -1.5 - 0.5j)   # ωn≈1.58, ζ≈0.95
    poles_z: Tuple = (-2.0 + 0.3j, -2.0 - 0.3j)   # ωn≈2.02, ζ≈0.99 

    # Safety clamps on reference angles [rad]
    phi_max:   float = np.radians(30.0)
    theta_max: float = np.radians(30.0)


class PositionController:
    """
    Outer-loop position controller via exact feedback linearisation.

    Converts world-frame position / velocity / acceleration references into:
        φ_ref   [rad]   desired roll angle
        θ_ref   [rad]   desired pitch angle
        U1      [N]     required total thrust
    """

    def __init__(
        self,
        params:      QuadParams,
        ctrl_params: Optional[PosCtrlParams] = None,
    ):
        self.p  = params
        self.cp = ctrl_params or PosCtrlParams()
        self.k1x, self.k2x = self._pole_gains(self.cp.poles_x)
        self.k1y, self.k2y = self._pole_gains(self.cp.poles_y)
        self.k1z, self.k2z = self._pole_gains(self.cp.poles_z)

    # ── static helper ───────────────────────────────────────

    @staticmethod
    def _pole_gains(poles: Tuple) -> Tuple[float, float]:
        """
        Pole-placement gains for a double integrator.

        Characteristic polynomial: (s−p1)(s−p2) = s² − (p1+p2)s + p1·p2
        Control law: v = ref_acc + k1·e + k2·ė
        Error dynamics: ë + k2·ė + k1·e = 0  →  k1 = p1·p2,  k2 = −(p1+p2)

        Stability requires k1 > 0 and k2 > 0.
        """
        p1, p2 = complex(poles[0]), complex(poles[1])
        k1 = float(np.real(p1 * p2))
        k2 = float(-np.real(p1 + p2))
        if k1 <= 0 or k2 <= 0:
            raise ValueError(
                f"Unstable gains k1={k1:.3f}, k2={k2:.3f} — "
                "ensure both poles have strictly negative real parts."
            )
        return k1, k2

    # ── main compute ─────────────────────────────────────────

    def compute(
        self,
        ref_pos: np.ndarray,   # [x_r, y_r, z_r]    m
        ref_vel: np.ndarray,   # [ẋ_r, ẏ_r, ż_r]    m/s
        ref_acc: np.ndarray,   # [ẍ_r, ÿ_r, z̈_r]    m/s²
        psi_ref: float,        # ψ_r                rad
        state:   np.ndarray,   # full state [12]
    ) -> Tuple[float, float, float]:
        """
        Compute (φ_ref, θ_ref, U1).

        Steps
        -----
        1.  Compute world-frame position and velocity from current state.
        2.  PD virtual-control law: vi = ref_acci + k1i·ei + k2i·ėi
        3.  Thrust direction from virtual accelerations: exact inversion.
        4.  Recover φ_ref, θ_ref via arc-sin (no switching on ψ needed).
        5.  Safety-clamp reference angles.
        """
        m, g = self.p.mass, self.p.g

        # ── Step 1: current position and world-frame velocity ──────────────
        phi, theta, psi = state[9], state[10], state[11]
        pos_W = state[6:9].copy()
        vel_W = R_body_to_world(phi, theta, psi) @ state[0:3]

        # ── Step 2: virtual control (desired world-frame acceleration) ─────
        e_pos = ref_pos - pos_W          # position error  [m]
        e_vel = ref_vel - vel_W          # velocity error  [m/s]

        vx = ref_acc[0] + self.k1x * e_pos[0] + self.k2x * e_vel[0]
        vy = ref_acc[1] + self.k1y * e_pos[1] + self.k2y * e_vel[1]
        vz = ref_acc[2] + self.k1z * e_pos[2] + self.k2z * e_vel[2]

        # ── Step 3: required thrust direction ─────────────────────────────
        # Force balance in NED:  [ẍ,ÿ,z̈] = [0,0,g] − (U1/m)·R_WB·ê_z_body
        # where ê_z_body = [0,0,1] (body +z = NED +z = downward).
        # Upward thrust in world: F_W = −(U1/m)·R_WB·[0,0,1]_body = [ẍ,ÿ,z̈]−[0,0,g]
        # So:   F_W = [vx, vy, vz − g]   (note: vz−g is negative for upward thrust)
        #
        # NED: z positive downward.  g acts in +z.  Thrust acts in −z (upward).
        # For hover: vz = 0  →  F_W,z = 0−g = −g  →  upward force = m·g  ✓

        Fw = np.array([vx, vy, vz - g])        # desired specific force [m/s²]
        Fw_norm = np.linalg.norm(Fw)

        # Degenerate case: near-zero thrust request (free-fall region)
        if Fw_norm < 1e-3:
            return 0.0, 0.0, float(m * 0.1)

        U1 = float(m * Fw_norm)
        T_hat = Fw / Fw_norm                # unit thrust direction in world  

        # ── Step 4: recover θ_ref and φ_ref ───────────────────────────────
        # After applying desired yaw rotation, the thrust direction components are:
        #   Tx' = cos(ψ)·Tx + sin(ψ)·Ty   (rotated x-component)
        #   Ty' = −sin(ψ)·Tx + cos(ψ)·Ty  (rotated y-component)
        #   Tz' = Tz                         (z-component unchanged by yaw)
        #
        # Desired thrust direction in body frame (yaw-aligned, before roll/pitch):
        #   T̂_yaw = [Tx', Ty', Tz']
        # From rotation geometry:
        #   T̂_yaw,x = sin(θ_ref)                 → θ_ref = arcsin(T̂_yaw,x)
        #   T̂_yaw,y = −sin(φ_ref)·cos(θ_ref)     → φ_ref = arcsin(−T̂_yaw,y/cos(θ))
        #
        # Note the sign:  T̂ points in the direction of thrust (upward = −NED-z).
        # But our F_W = [vx, vy, vz−g] has Fz < 0 for normal flight (thrust up),
        # so T̂_W,z < 0.  After yaw rotation T̂_yaw,z = T̂_W,z still.
        # sin(θ_ref) captures the forward tilt.

        cp, sp = np.cos(psi_ref), np.sin(psi_ref)

        # Thrust direction in yaw-rotated frame
        Tx_psi =  cp * T_hat[0] + sp * T_hat[1]
        Ty_psi = -sp * T_hat[0] + cp * T_hat[1]

        sin_theta =  -Tx_psi
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        theta_ref = float(np.arcsin(sin_theta))

        cos_theta = np.cos(theta_ref)
        if abs(cos_theta) < 1e-6:
            phi_ref = 0.0       # gimbal lock of reference — set zero roll
        else:
            sin_phi = Ty_psi / cos_theta
            sin_phi = np.clip(sin_phi, -1.0, 1.0)
            phi_ref = float(np.arcsin(sin_phi))

        # ── Step 5: safety clamp ───────────────────────────────────────────
        phi_ref   = float(np.clip(phi_ref,   -self.cp.phi_max,   self.cp.phi_max))
        theta_ref = float(np.clip(theta_ref, -self.cp.theta_max, self.cp.theta_max))

        return phi_ref, theta_ref, U1


# ═════════════════════════════════════════════════════════════
#  MODULE 2 — LPV-MPC Attitude Controller  (Inner Loop)
# ═════════════════════════════════════════════════════════════

@dataclass
class MPCParams:
    """
    Tuning parameters for the LPV-MPC inner-loop attitude controller.

    Cost function
    -------------
    J = Σ_{i=1}^{N-1} (y(k+i)−r)^T Q (y(k+i)−r)
      + (y(k+N)−r)^T S (y(k+N)−r)       ← terminal weight
      + Σ_{i=0}^{N-1} Δu(k+i)^T R Δu(k+i)

    Choose S ≥ Q for terminal stability enhancement.
    Larger Q/R ratio: faster tracking, more aggressive inputs.
    Larger R/Q ratio: smoother inputs, slower tracking.

    Constraints (per-step, in absolute torques and increments)
    -----------------------------------------------------------
    −U_max ≤ [U2,U3,U4] ≤ U_max       (physical torque limits)
    −dU_max ≤ [ΔU2,ΔU3,ΔU4] ≤ dU_max  (actuator rate limits)
    """
    horizon:  int   = 10

    # Tracking weights on [φ, θ, ψ]
    Q_diag:  np.ndarray = field(default_factory=lambda: np.array([20.0, 20.0, 10.0]))
    # Terminal cost weights.
    # When use_dare_terminal=True in LPVMPCController, S_diag is IGNORED and
    # the terminal weight is computed from the Discrete Algebraic Riccati Equation
    # (DARE), giving the infinite-horizon LQR cost as the terminal weight.
    # This provides a formal stability certificate for the MPC.
    # Set use_dare_terminal=False to use S_diag as a manual heuristic instead.
    S_diag:  np.ndarray = field(default_factory=lambda: np.array([40.0, 40.0, 20.0]))
    # Input-rate cost weights on [ΔU2, ΔU3, ΔU4]
    R_diag:  np.ndarray = field(default_factory=lambda: np.array([5.0,  5.0,  12.0]))

    # Absolute torque limits [N·m]  (set from rotor-speed analysis)
    U2_max: float = 0.45
    U3_max: float = 0.45
    U4_max: float = 0.12

    # Per-step increment limits [N·m]
    dU2_max: float = 0.18
    dU3_max: float = 0.18
    dU4_max: float = 0.05


class LPVMPCController:
    """
    Inner-loop attitude controller using Linear Parameter-Varying MPC.

    At each control step:
      1. Build continuous-time LPV matrices A(σ), B, C using current body rates.
      2. Discretise with ZOH to obtain Ad, Bd, Cd.
      3. Form augmented (Δu) state-space Ã, B̃, C̃.
      4. Build condensed prediction matrices Ψ, Θ (computed analytically).
      5. Solve bound-constrained QP for optimal ΔŪ*.
      6. Apply first increment; update u_prev.
    """

    def __init__(
        self,
        params:            QuadParams,
        mpc_params:        Optional[MPCParams] = None,
        dt:                float = 0.05,
        use_dare_terminal: bool  = True,
    ):
        self.p  = params
        self.mp = mpc_params or MPCParams()
        self.dt = dt
        self.use_dare_terminal = use_dare_terminal
        self.u_prev = np.zeros(3)           # [U2, U3, U4] from last step

        # DARE cache — recompute only when scheduling point shifts
        # significantly (> _dare_threshold in L2 norm), reducing DARE calls
        # from 20 Hz to ~2-5 Hz during typical flight.
        self._dare_cache: dict = {
            'Ad': None,   # last scheduling tuple as np.ndarray
            'Bd': None,
            'P':  None,   # cached 6x6 DARE solution
        }
        self._dare_threshold = 1e-3   # rad/s change to trigger recompute

        # Pre-build constant constraint arrays
        self._N_cache: int = -1
        self._rebuild_constraints()

    # ── LPV continuous matrices ────────────────────────────────────────

    def _lpv_continuous(
        self,
        phi_d: float,    # φ̇  Euler rate  [rad/s] 
        theta_d: float,  # θ̇  Euler rate  [rad/s] 
        psi_d: float,    # ψ̇  Euler rate  [rad/s] 
        omega_net: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build A_c(σ), B_c, C for the continuous-time LPV attitude model.

        State ordering: x_att = [φ, φ̇, θ, θ̇, ψ, ψ̇]^T

        Scheduling on Euler rates (phi_d, theta_d, psi_d) to
        be consistent with the LPV state vector which uses Euler rates
        """
        Ix, Iy, Iz = self.p.Ix, self.p.Iy, self.p.Iz
        Jtp        = self.p.J_tp

        # ── A_c  (6×6) ──────────────────────────────────────────────
        A_c = np.zeros((6, 6))

        # Kinematic coupling rows: angle rate = derivative of angle
        A_c[0, 1] = 1.0   # φ̇  = φ̇
        A_c[2, 3] = 1.0   # θ̇  = θ̇
        A_c[4, 5] = 1.0   # ψ̇  = ψ̇

        # ── φ̈ row (idx 1) ───────────────────────────────────────────
        # Full Euler eq:  ṗ = q·r·(Iy−Iz)/Ix − (J_tp/Ix)·q·Ω + U2/Ix
        #
        # Bilinear LPV factoring of  θ̇·ψ̇  (Euler-rate form):
        #   θ̇·ψ̇ ≈ theta_d·ψ̇ + psi_d·θ̇ − theta_d·psi_d  (Taylor; const dropped)
        #   → A[1,3] · θ̇  +  A[1,5] · ψ̇
        #
        # Gyroscopic term  −(J_tp/Ix)·q·Ω ≈ −(J_tp/Ix)·theta_d·Ω (Euler-rate approx).
        #   Combined into A[1,3]:
        A_c[1, 3] = theta_d * (Iy - Iz) / Ix  -  (Jtp / Ix) * omega_net   # φ̈←θ̇
        A_c[1, 5] = psi_d   * (Iy - Iz) / Ix                               # φ̈←ψ̇

        # ── θ̈ row (idx 3) ───────────────────────────────────────────────────
        # Full Euler eq:  q̇ = p·r·(Iz−Ix)/Iy + (J_tp/Iy)·p·Ω + U3/Iy
        #
        # Bilinear LPV factoring of  φ̇·ψ̇  (Euler-rate form):
        #   φ̇·ψ̇ ≈ phi_d·ψ̇ + psi_d·φ̇ − phi_d·psi_d
        #   → A[3,1] · φ̇  +  A[3,5] · ψ̇
        #
        # Gyroscopic term  +(J_tp/Iy)·p·Ω ≈ +(J_tp/Iy)·phi_d·Ω (Euler-rate approx).
        #   Combined into A[3,1]:
        A_c[3, 1] = phi_d * (Iz - Ix) / Iy  +  (Jtp / Iy) * omega_net   # θ̈←φ̇
        A_c[3, 5] = psi_d * (Iz - Ix) / Iy                               # θ̈←ψ̇

        # ── ψ̈ row (idx 5) ───────────────────────────────────────────────────
        # Full Euler eq:  ṙ = p·q·(Ix−Iy)/Iz + U4/Iz
        #
        # Bilinear LPV factoring of  φ̇·θ̇  (Euler-rate form):
        #   φ̇·θ̇ ≈ theta_d·φ̇ + phi_d·θ̇  →  A[5,1]·φ̇ + A[5,3]·θ̇
        A_c[5, 1] = theta_d * (Ix - Iy) / Iz   # ψ̈←φ̇  (theta_d scheduled)
        A_c[5, 3] = phi_d   * (Ix - Iy) / Iz   # ψ̈←θ̇  (phi_d scheduled)

        # ── B_c  (6×3) ───────────────────────────────────────────────────
        # Torques enter the angular-acceleration (odd-indexed) rows
        B_c = np.zeros((6, 3))
        B_c[1, 0] = 1.0 / Ix    # U2 → φ̈
        B_c[3, 1] = 1.0 / Iy    # U3 → θ̈
        B_c[5, 2] = 1.0 / Iz    # U4 → ψ̈

        # ── C  (3×6) — output: angles only ───────────────────────────────
        C = np.zeros((3, 6))
        C[0, 0] = 1.0   # φ
        C[1, 2] = 1.0   # θ
        C[2, 4] = 1.0   # ψ

        return A_c, B_c, C

    # ── ZOH discretisation ────────────────────────────────────────────────

    @staticmethod
    def _zoh(
        A_c: np.ndarray, B_c: np.ndarray, C: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Zero-Order Hold discretisation.

        Computes  Ad = expm(A_c·dt)  and
                  Bd = (∫₀ᵈᵗ expm(A_c·τ)dτ) · B_c

        using scipy.signal.cont2discrete, which implements the exact ZOH
        formula via the matrix exponential.  This is more accurate than
        Forward Euler (O(dt²)) and inherits the stability of the continuous
        system (eigenvalues mapped from s-plane via z = e^(s·dt)).
        """
        n, m, p = A_c.shape[0], B_c.shape[1], C.shape[0]
        D = np.zeros((p, m))
        Ad, Bd, Cd, _, _ = cont2discrete((A_c, B_c, C, D), dt, method='zoh')
        return Ad, Bd, Cd

    # ── DARE terminal weight ──────────────────────────────────────────────

    def _dare_terminal(
        self,
        Ad: np.ndarray,
        Bd: np.ndarray,
        Q:  np.ndarray,   # 6x6 state cost (angle+rate, extended from output Q)
        R:  np.ndarray,   # 3x3 input cost
    ) -> np.ndarray:
        """
        Compute the infinite-horizon LQR terminal weight via the Discrete
        Algebraic Riccati Equation (DARE).

            P = Q + Ad^T * P * Ad
              - Ad^T * P * Bd * (R + Bd^T * P * Bd)^{-1} * Bd^T * P * Ad

        The solution P is the unique positive semi-definite matrix satisfying
        the DARE and gives the terminal cost that makes the MPC equivalent to
        an infinite-horizon LQR at the terminal step.  This provides a formal
        stability certificate for the closed-loop system
        
        Returns the full 6x6 DARE solution P_dare 

        Parameters
        ----------
        Ad : 6x6 discrete-time state matrix
        Bd : 6x3 discrete-time input matrix
        Q  : 6x6 state weighting matrix 
        R  : 3x3 input weighting matrix 

        Returns
        -------
        P_full : 6x6 full-state DARE solution.
        """
        try:
            P_dare = solve_discrete_are(Ad, Bd, Q, R)
            return P_dare          # 6x6, caller projects to output space
        except Exception:
            # DARE may fail if (Ad, Bd) is not stabilisable at the current
            # scheduling point (e.g., near-zero body rates).  
            #Fall back to a heuristic 6x6 diagonal weight gracefully.
          
            return 2.0 * np.diag([self.mp.Q_diag[0], self.mp.Q_diag[0] * 0.1,
                                   self.mp.Q_diag[1], self.mp.Q_diag[1] * 0.1,
                                   self.mp.Q_diag[2], self.mp.Q_diag[2] * 0.1])

    # ── Augmented system ─────────────────────────────────────────────────────

    @staticmethod
    def _augment(
        Ad: np.ndarray, Bd: np.ndarray, Cd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Augment state with previous input to enable incremental (Δu) MPC.

            Ã = |Ad  Bd|   B̃ = |Bd|   C̃ = [Cd  0]
                | 0   I|       | I|
        """
        n, m, p = Ad.shape[0], Bd.shape[1], Cd.shape[0]
        A_a = np.block([[Ad,              Bd         ],
                        [np.zeros((m,n)), np.eye(m)  ]])
        B_a = np.block([[Bd        ],
                        [np.eye(m) ]])
        C_a = np.block([Cd, np.zeros((p, m))])
        return A_a, B_a, C_a

    # ── Prediction matrices ──────────────────────────────────────────────────

    @staticmethod
    def _prediction_matrices(
        A_a: np.ndarray, B_a: np.ndarray, C_a: np.ndarray, N: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build condensed prediction matrices Ψ (shape p·N × ñ) and
        Θ (shape p·N × m·N) such that   Y = Ψ·x̃ + Θ·ΔŪ.

        Ψ[i] = C̃·Ã^(i+1)                             row block i ∈ [0,N)
        Θ[i,j] = C̃·Ã^(i−j)·B̃   for j ≤ i,  else 0   (lower-triangular Toeplitz)

        The double loop is O(N²·n³) in cost — acceptable for small N (≤ 20).
        For large horizons, use a recursive Toeplitz build instead.
        """
        n_a = A_a.shape[0]
        m   = B_a.shape[1]
        p   = C_a.shape[0]

        Psi   = np.zeros((p * N, n_a))
        Theta = np.zeros((p * N, m * N))

        # Precompute Ã^1 … Ã^N
        A_pows = [np.eye(n_a)]
        for _ in range(N):
            A_pows.append(A_pows[-1] @ A_a)

        for i in range(N):
            rs, re = p * i, p * (i + 1)
            Psi[rs:re, :] = C_a @ A_pows[i + 1]
            for j in range(i + 1):
                cs, ce = m * j, m * (j + 1)
                Theta[rs:re, cs:ce] = C_a @ A_pows[i - j] @ B_a

        return Psi, Theta

    # ── QP matrices ──────────────────────────────────────────

    @staticmethod
    def _qp_matrices(
        Psi:   np.ndarray,
        Theta: np.ndarray,
        Q_bar: np.ndarray,
        R_bar: np.ndarray,
        x_aug: np.ndarray,
        r_vec: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute H and f for the standard QP:
            min_{ΔŪ}  ½·ΔŪ^T H ΔŪ + f^T ΔŪ

        Derivation:
            J = (Θ·ΔŪ + Ψ·x̃ − r)^T Q̄ (…) + ΔŪ^T R̄ ΔŪ
            ∂J/∂ΔŪ = 2·Θ^T Q̄ Θ·ΔŪ + 2·Θ^T Q̄ (Ψ·x̃−r) + 2·R̄·ΔŪ = 0
            H = Θ^T Q̄ Θ + R̄    (PD since R̄ > 0)
            f = Θ^T Q̄ (Ψ·x̃ − r)
        """
        H = Theta.T @ Q_bar @ Theta + R_bar
        H = 0.5 * (H + H.T)            # symmetrise (suppress float rounding)
        f = Theta.T @ Q_bar @ (Psi @ x_aug - r_vec)
        return H, f

    # ── Constraint setup ───────────────────────────────────────────────────

    def _rebuild_constraints(self) -> None:
        """
        Pre-build arrays that define the box constraints.

        Called once, or when the horizon changes.

        Two constraint families (per step j ∈ [0,N)):
          (A) Increment bounds:    −dU_max ≤ Δu_j ≤ dU_max
          (B) Absolute bounds:     U_min  ≤ u_{j+1} = u_prev + L_j @ ΔŪ ≤ U_max
              where L_j is the j-th block-row of the cumulative-sum matrix L.
        """
        N, m = self.mp.horizon, 3

        # ── Increment bound arrays ────────────────────────────────────────
        dU_max = np.array([self.mp.dU2_max, self.mp.dU3_max, self.mp.dU4_max])
        self._dU_lb = np.tile(-dU_max, N)
        self._dU_ub = np.tile( dU_max, N)

        # ── Absolute bound arrays ─────────────────────────────────────────
        self._U_min = np.array([-self.mp.U2_max, -self.mp.U3_max, -self.mp.U4_max])
        self._U_max = np.array([ self.mp.U2_max,  self.mp.U3_max,  self.mp.U4_max])

        # Cumulative-sum matrix L ∈ ℝ^{mN × mN} (block lower-triangular identity)
        # L @ ΔŪ gives [u_0−u_{-1}, u_1−u_{-1}, …, u_{N-1}−u_{-1}]
        #            = [Δu_0, Δu_0+Δu_1, …, Σ Δu_j]
        L = np.zeros((m * N, m * N))
        for i in range(N):
            for j in range(i + 1):
                L[m*i:m*(i+1), m*j:m*(j+1)] = np.eye(m)
        self._L = L
        self._N_cache = N

    # ── Main compute ─────────────────────────────────────────────────────────

    def compute(
        self,
        state:     np.ndarray,   # full 12-state vector
        omega_net: float,        # Ω_net [rad/s]
        ref_phi:   float,        # φ_ref [rad]
        ref_theta: float,        # θ_ref [rad]
        ref_psi:   float,        # ψ_ref [rad]
    ) -> np.ndarray:
        """
        Compute optimal attitude torques [U2, U3, U4].

        Returns
        -------
        U_att : np.ndarray [3]   updated [U2, U3, U4] in N·m
        """
        N, m, p, n_att = self.mp.horizon, 3, 3, 6

        if self._N_cache != N:
            self._rebuild_constraints()

        # ── Extract current attitude states ───────────────────────────────
        phi, theta, psi = state[9], state[10], state[11]
        p_b, q_b, r_b   = state[3], state[4],  state[5]   # body rates [rad/s]

        # Euler angle rates (for LPV state vector) via T-matrix
        try:
            T_mat = euler_rate_matrix(phi, theta)
            phi_d, theta_d, psi_d = T_mat @ np.array([p_b, q_b, r_b])
        except ValueError:
            # Near gimbal lock — hold previous rates as safe fallback
            phi_d = theta_d = psi_d = 0.0

        # ── Step 1: build LPV continuous model ───────────────────────────
        # schedule on Euler rates (phi_d, theta_d, psi_d), consistent
        # with the LPV state x_att = [phi, phi_d, theta, theta_d, psi, psi_d].
        A_c, B_c, C_c = self._lpv_continuous(phi_d, theta_d, psi_d, omega_net)

        # ── Step 2: ZOH discretisation ────────────────────────────────────
        Ad, Bd, Cd = self._zoh(A_c, B_c, C_c, self.dt)

        # ── Step 3: augmented incremental system ──────────────────────────
        A_a, B_a, C_a = self._augment(Ad, Bd, Cd)

        # ── Step 4: prediction matrices ───────────────────────────────────
        Psi, Theta = self._prediction_matrices(A_a, B_a, C_a, N)

        # ── Step 5: weight matrices ───────────────────────────────────────
        Q = np.diag(self.mp.Q_diag)
        S = np.diag(self.mp.S_diag)
        R = np.diag(self.mp.R_diag)

        # ── Terminal weight───────────────────
        # DARE gives the infinite-horizon LQR cost at the terminal step,
        # which is the standard condition for MPC stability (Rawlings & Mayne).
        # Build a full 6x6 state-space Q for DARE from the 3x3 output Q:
        # angles are weighted by Q_diag, rates by a fraction of Q_diag.
        #
        # _dare_terminal now returns the full 6x6 P_dare (not projected).
        # We project here via C_out @ P_full @ C_out.T so the 3x3 result is
        # derived from the true LQR solution, giving a valid stability weight.
        #
        # Cache the DARE solution and recompute only when the discrete
        # matrices (Ad, Bd) have changed by more than _dare_threshold (max
        # element-wise difference).  Keying on (Ad, Bd) directly is more
        # accurate than keying on the raw scheduling vector sigma, since two
        # different sigma values that produce the same discrete matrices do not
        # need a DARE recompute.  This reduces DARE calls from 20 Hz to
        # ~2-5 Hz during typical flight, saving 10-40 ms/s of compute.
        if self.use_dare_terminal:
            Q6 = np.zeros((6, 6))
            Q6[0, 0] = self.mp.Q_diag[0]
            Q6[1, 1] = self.mp.Q_diag[0] * 0.1
            Q6[2, 2] = self.mp.Q_diag[1]
            Q6[3, 3] = self.mp.Q_diag[1] * 0.1
            Q6[4, 4] = self.mp.Q_diag[2]
            Q6[5, 5] = self.mp.Q_diag[2] * 0.1

            # Recompute DARE only when Ad or Bd have changed by more than _dare_threshold.
            # This reduces expensive DARE solves from 20 Hz to ~2-5 Hz during typical flight.
            cache = self._dare_cache
            recompute = (
                cache['Ad'] is None
                or np.max(np.abs(Ad - cache['Ad'])) > self._dare_threshold
                or np.max(np.abs(Bd - cache['Bd'])) > self._dare_threshold
            )
            if recompute:
                P_full = self._dare_terminal(Ad, Bd, Q6, R)
                cache['Ad'] = Ad.copy()
                cache['Bd'] = Bd.copy()
                cache['P']  = P_full
            else:
                P_full = cache['P']

            # project full 6x6 DARE solution to 3x3 output space
            C_out = np.zeros((3, 6))
            C_out[0, 0] = 1.0   # phi
            C_out[1, 2] = 1.0   # theta
            C_out[2, 4] = 1.0   # psi
            S_out = C_out @ P_full @ C_out.T
        else:
            S_out = S   # use heuristic S_diag from MPCParams

        # Block-diagonal Q̄ with terminal S_out
        Q_bar = np.zeros((p * N, p * N))
        for i in range(N):
            wi = p * i
            Q_bar[wi:wi+p, wi:wi+p] = S_out if i == N - 1 else Q

        R_bar = np.kron(np.eye(N), R)

        # ── Step 6: current augmented state ──────────────────────────────
        x_att = np.array([phi, phi_d, theta, theta_d, psi, psi_d])
        x_aug = np.concatenate([x_att, self.u_prev])      # shape (9,)

        # ── Step 7: reference vector ──────────────────────────────────────
        r_vec = np.tile([ref_phi, ref_theta, ref_psi], N)  # shape (3N,)

        # ── Step 8: QP cost matrices ──────────────────────────────────────
        H, f = self._qp_matrices(Psi, Theta, Q_bar, R_bar, x_aug, r_vec)

        # ── Step 9: constraint bounds ─────────────────────────────────────
        # Increment bounds (box, correct as-is):
        lb = self._dU_lb    # -dU_max per step
        ub = self._dU_ub    # +dU_max per step

        # Absolute-input constraints  U_min <= u_prev + L@DU <= U_max
        # Rearranged for the decision variable DU:
        #     U_min - u_prev <= L@DU <= U_max - u_prev
        # This is a general linear constraint on L@DU, NOT a box on DU.
        # The previous code computed  L @ tile(u_prev, N) = [u, 2u, ..., Nu],
        # creating a per-step offset of (j+1)*u_prev instead of u_prev, which
        # falsely tightened/inverted bounds at later horizon steps.
        #
        # Correct: subtract a flat u_prev offset (not L @ u_tile):
        u_prev_tiled = np.tile(self.u_prev, N)   # shape (mN,) — flat, not L-scaled
        abs_rhs_lb   = np.tile(self._U_min, N) - u_prev_tiled
        abs_rhs_ub   = np.tile(self._U_max, N) - u_prev_tiled

        n_qp = m * N

        # ── Step 10: solve mixed-constraint QP ────────────────────────────
        #
        # Problem:  min  0.5*DU'*H*DU + f'*DU
        #   s.t.  lb  <=  DU  <= ub          (increment box)
        #         abs_rhs_lb  <=  L@DU  <= abs_rhs_ub  (absolute via L)
        #
        # Solver priority:
        #   1. quadprog (active-set, exact Hessian)
        #      API: solve_qp(G, a, C, b)  =>  min 0.5*x'Gx - a'x  s.t. C'x >= b
        #      Combined constraint matrix:
        #        increment box :  [+I; -I] @ DU  >= [lb; -ub]
        #        absolute via L:  [+L; -L] @ DU  >= [abs_rhs_lb; -abs_rhs_ub]
        #   2. scipy L-BFGS-B (fallback) with explicit bounds + linear constraints.

        if _QUADPROG_AVAILABLE:
            try:
                # Build combined constraint matrix C_all' (rows = constraints)
                # quadprog expects C such that C'*x >= b  =>  C shape (n, n_con)
                C_inc = np.vstack([ np.eye(n_qp), -np.eye(n_qp)])           # (2*n_qp, n_qp)
                C_abs = np.vstack([ self._L,       -self._L      ])          # (2*mN,  n_qp)
                C_all = np.vstack([C_inc, C_abs]).T                          # (n_qp, 4*n_qp)

                b_inc = np.concatenate([lb, -ub])
                b_abs = np.concatenate([abs_rhs_lb, -abs_rhs_ub])
                b_all = np.concatenate([b_inc, b_abs])

                # quadprog requires G strictly PD; regularise to guard near-singular H
                G_qp = H + 1e-10 * np.eye(n_qp)
                sol  = _quadprog.solve_qp(G_qp, -f, C_all, b_all)
                dU_opt = sol[0]
            except Exception:
                # quadprog failed (infeasible or numerical issue) — use fallback
                # For L-BFGS-B: encode absolute constraint as tightened box on DU
                # (conservative but always feasible).
                abs_box_lb = np.maximum(lb, abs_rhs_lb)
                abs_box_ub = np.minimum(ub, abs_rhs_ub)
                abs_box_ub = np.maximum(abs_box_ub, abs_box_lb)   # guarantee feasibility
                res = minimize(
                    fun = lambda dU: 0.5 * float(dU @ H @ dU) + float(f @ dU),
                    jac = lambda dU: H @ dU + f,
                    x0  = np.zeros(n_qp),
                    method  = 'L-BFGS-B',
                    bounds  = list(zip(abs_box_lb, abs_box_ub)),
                    options = dict(maxiter=200, ftol=1e-12, gtol=1e-9),
                )
                dU_opt = res.x
        else:
            # quadprog not installed — scipy fallback (conservative box encoding)
            abs_box_lb = np.maximum(lb, abs_rhs_lb)
            abs_box_ub = np.minimum(ub, abs_rhs_ub)
            abs_box_ub = np.maximum(abs_box_ub, abs_box_lb)   # guarantee feasibility
            res = minimize(
                fun = lambda dU: 0.5 * float(dU @ H @ dU) + float(f @ dU),
                jac = lambda dU: H @ dU + f,
                x0  = np.zeros(n_qp),
                method  = 'L-BFGS-B',
                bounds  = list(zip(abs_box_lb, abs_box_ub)),
                options = dict(maxiter=200, ftol=1e-12, gtol=1e-9),
            )
            dU_opt = res.x
        delta_U = dU_opt[:m]                      # [ΔU2, ΔU3, ΔU4]
        U_new   = self.u_prev + delta_U

        # Defensive absolute clamp (redundant with QP constraints, but safe)
        U_new[0] = np.clip(U_new[0], -self.mp.U2_max, self.mp.U2_max)
        U_new[1] = np.clip(U_new[1], -self.mp.U3_max, self.mp.U3_max)
        U_new[2] = np.clip(U_new[2], -self.mp.U4_max, self.mp.U4_max)

        self.u_prev = U_new.copy()
        return U_new
