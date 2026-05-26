"""
dynamics.py  ―  Quadcopter 6-DOF Nonlinear Dynamics
=====================================================

Implements the complete Newton-Euler rigid-body model for a quadrotor:

  • Translational dynamics  (Newton, body frame)
  • Rotational dynamics     (Euler moment equations, body frame)
  • Translational kinematics (body → world via R_WB)
  • Euler-angle kinematics   (body rates → Euler rates via T-matrix)
  • Optional quadratic aerodynamic drag
  • RK4 numerical integration with optional process noise

═══════════════════════════════════════════════════════
  STATE VECTOR   x ∈ ℝ¹²   (SI units)
═══════════════════════════════════════════════════════
  idx  symbol   description                frame   unit
   0    u       forward body-frame vel.    Body    m/s
   1    v       lateral body-frame vel.    Body    m/s
   2    w       vertical body-frame vel.   Body    m/s  (+ve downward, NED)
   3    p       roll  angular rate         Body    rad/s
   4    q       pitch angular rate         Body    rad/s
   5    r       yaw   angular rate         Body    rad/s
   6    x       North position             World   m
   7    y       East  position             World   m
   8    z       Down  position             World   m    (+ve downward, NED)
   9    φ       roll  angle (ZYX)          Euler   rad
  10    θ       pitch angle (ZYX)          Euler   rad
  11    ψ       yaw   angle (ZYX)          Euler   rad

═══════════════════════════════════════════════════════
  CONTROL INPUTS   U ∈ ℝ⁴   (SI units)
═══════════════════════════════════════════════════════
  U1  total thrust force (along body −z, upward)   N
  U2  roll  torque (about body x)                  N·m
  U3  pitch torque (about body y)                  N·m
  U4  yaw   torque (about body z)                  N·m

═══════════════════════════════════════════════════════
  EQUATIONS OF MOTION   (body frame, NED convention)
═══════════════════════════════════════════════════════
  Translational (Newton's 2nd law in rotating body frame):
    u̇ = (vr − wq) + g·sin(θ)             + F_drag,x/m
    v̇ = (wp − ur) − g·cos(θ)·sin(φ)     + F_drag,y/m
    ẇ = (uq − vp) + g·cos(θ)·cos(φ)     + F_drag,z/m  − U1/m

  Gravity body-frame components come from R_WB^T · [0, 0, g]:
    g_Bx =  g·sin(θ)          g_By = −g·cos(θ)·sin(φ)    g_Bz = +g·cos(θ)·cos(φ)
  Note: g_Bz is positive (+) because in NED gravity pulls in +z direction,
  and at hover U1/m = g cancels it → ẇ = 0  ✓

  Rotational (Euler's rigid-body equations):
    ṗ = q·r·(Iy−Iz)/Ix  −  (J_tp/Ix)·q·Ω_net  +  U2/Ix
    q̇ = p·r·(Iz−Ix)/Iy  +  (J_tp/Iy)·p·Ω_net  +  U3/Iy
    ṙ = p·q·(Ix−Iy)/Iz                          +  U4/Iz

  Gyroscopic propeller term:
    Ω_net = ω1 − ω2 + ω3 − ω4   (net rotor angular momentum)
    Rotors 1,3 CCW (+); rotors 2,4 CW (−).

  Kinematics:
    [ẋ, ẏ, ż]^T    = R_WB(φ,θ,ψ) · [u, v, w]^T
    [φ̇, θ̇, ψ̇]^T = T(φ,θ)       · [p, q, r]^T

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from utils import (
    R_body_to_world,
    euler_rate_matrix,
    aero_drag_body,
    build_mixer,
    rotor_speeds_from_controls,
    wrap_to_pi,
)


# ═════════════════════════════════════════════════
#  Physical parameters  (AscTec Hummingbird)
# ═════════════════════════════════════════════════

@dataclass
class QuadParams:
    """
    Physical constants for the quadrotor.

    Default values match the AscTec Hummingbird research platform.
    All quantities in SI units.

    Thrust / torque coefficients
    ────────────────────────────
    The AscTec datasheet gives k_F = 7.6184×10⁻⁸ N/RPM² and
    k_M = 2.6839×10⁻⁹ N·m/RPM².  Convert to (rad/s)²:

        c_T = k_F · (60 / 2π)²  = 6.9920×10⁻⁴  N·s²/rad²
        c_Q = k_M · (60 / 2π)²  = 2.4639×10⁻⁵  N·m·s²/rad²
    """
    # ── Inertial properties ────────────────────────────────────────────
    mass:      float = 0.698        # total mass m            [kg]
    g:         float = 9.80665      # gravitational accel.    [m/s²]
    Ix:        float = 3.40e-3      # roll  inertia           [kg·m²]
    Iy:        float = 3.40e-3      # pitch inertia           [kg·m²]
    Iz:        float = 6.00e-3      # yaw   inertia           [kg·m²]

    # ── Propulsion ──────────────────────────────────────────────────────
    c_T:       float = 6.9920e-4    # thrust coefficient      [N·s²/rad²]
    c_Q:       float = 2.4639e-5    # torque coefficient      [N·m·s²/rad²]
    arm:       float = 0.171        # arm length (rotor→CoM)  [m]
    J_tp:      float = 1.302e-6     # rotor+prop axial inertia [kg·m²]
    omega_max: float = 900.0        # max rotor speed         [rad/s]
    omega_min: float = 10.0         # idle rotor speed        [rad/s]

    # ── Aerodynamic drag ─────────────────────────────────────────────────
    drag_on:   bool  = True
    rho:       float = 1.225        # air density             [kg/m³]
    C_D: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.25, 0.25]))
    A_ref: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.02, 0.02]))

    # ── Derived (computed on demand) ─────────────────────────────────────
    def hover_omega(self) -> float:
        """Rotor speed required for hover (all four equal)."""
        return float(np.sqrt(self.mass * self.g / (4.0 * self.c_T)))

    def hover_thrust(self) -> float:
        """Total thrust at hover: U1* = m·g."""
        return self.mass * self.g


# ═══════════════════════════
#  Dynamics
# ═══════════════════════════

class QuadDynamics:
    """
    Full nonlinear 6-DOF quadrotor dynamics.

    Responsibilities
    ────────────────
    1. Compute state derivatives  ẋ = f(x, U, Ω_net)
    2. Integrate one step forward with RK4
    3. Map virtual controls [U1…U4] ↔ rotor speeds [ω1…ω4]
    """

    def __init__(self, p: Optional[QuadParams] = None):
        self.p = p or QuadParams()
        # Build mixer M once; its inverse is also constant.
        self._M     = build_mixer(self.p.c_T, self.p.c_Q, self.p.arm)
        self._M_inv = np.linalg.inv(self._M)

    # ── Public properties ───────────────────────────────────────────

    @property
    def mixer(self) -> np.ndarray:
        """4×4 constant mixer matrix  M  (U = M·[ω²])."""
        return self._M

    @property
    def mixer_inv(self) -> np.ndarray:
        """Precomputed  M⁻¹  ([ω²] = M⁻¹·U)."""
        return self._M_inv

    # ── Mixer wrapper ────────────────────────────────────────────────

    def controls_to_rotors(
        self, U: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Map U = [U1, U2, U3, U4] → rotor speeds [ω1…ω4], with saturation.

        Returns (omega [rad/s], saturated flag).
        """
        return rotor_speeds_from_controls(
            U, self._M_inv, self.p.omega_max, self.p.omega_min
        )

    # ── Core derivative ───────────────────────────────────────────────

    def derivatives(
        self,
        x:         np.ndarray,        # state  [12]
        U:         np.ndarray,        # controls [U1,U2,U3,U4]
        omega_net: float,             # Ω_net = ω1−ω2+ω3−ω4  [rad/s]
    ) -> np.ndarray:
        """
        Compute  ẋ = f(x, U, Ω_net).

        All intermediate quantities are annotated with units.
        The function is kept deliberately readable — no compressed
        one-liners — so that every term can be traced to its physical origin.

        Parameters
        ----------
        x         : state vector [12]
        U         : [U1(N), U2(N·m), U3(N·m), U4(N·m)]
        omega_net : net rotor angular momentum [rad/s]

        Returns
        -------
        dx : state derivative [12]
        """
        # ── Unpack state ───────────────────────────────────────────────────
        u_b, v_b, w_b = x[0], x[1], x[2]     # body-frame translational vel  [m/s]
        p,   q,   r   = x[3], x[4], x[5]     # body-frame angular rates      [rad/s]
        #  x[6:9]  = inertial position  (used only in kinematic rows)
        phi, theta, psi = x[9], x[10], x[11]  # Euler angles                 [rad]

        # ── Unpack controls ────────────────────────────────────────────────
        U1, U2, U3, U4 = float(U[0]), float(U[1]), float(U[2]), float(U[3])

        # ── Shorthand constants ────────────────────────────────────────────
        m, g         = self.p.mass, self.p.g
        Ix, Iy, Iz   = self.p.Ix,  self.p.Iy,  self.p.Iz
        J_tp         = self.p.J_tp

        sp, cp = np.sin(phi),   np.cos(phi)
        st, ct = np.sin(theta), np.cos(theta)

        # ── Aerodynamic drag ──────────────────
        if self.p.drag_on:
            F_d = aero_drag_body(
                np.array([u_b, v_b, w_b]),
                self.p.C_D, self.p.rho, self.p.A_ref
            )
        else:
            F_d = np.zeros(3)

        # ══ TRANSLATIONAL DYNAMICS (body frame) ═══════════════════════════
        #
        # Newton in rotating frame:  m·(v̇_b + ω×v_b) = F_gravity_b + F_thrust + F_drag
        #
        # Gravity in body frame  g_b = R_WB^T · [0,0,g]_NED :
        #   g_b,x = +g·sin(θ)
        #   g_b,y = −g·cos(θ)·sin(φ)
        #   g_b,z = +g·cos(θ)·cos(φ)    ← positive (gravity pulls +z in NED)
        #
        # Coriolis acceleration from rotating frame: ω×v = [vr−wq, wp−ur, uq−vp]
        #
        # Thrust U1 acts along body −z  → subtracts from ẇ
        #
        # Hover check: ẇ = 0 + g·1·1 + 0 − mg/m = g − g = 0 

        u_dot = (v_b*r  - w_b*q)  +  g*st         +  F_d[0]/m
        v_dot = (w_b*p  - u_b*r)  -  g*ct*sp       +  F_d[1]/m
        w_dot = (u_b*q  - v_b*p)  +  g*ct*cp       +  F_d[2]/m  -  U1/m

        # ══ ROTATIONAL DYNAMICS (body frame) ══════════════════════════════
        #
        # Euler's moment equations for a rigid body (principal-axis frame):
        #   Ix·ṗ = (Iy−Iz)·q·r  −  J_tp·q·Ω_net  +  U2
        #   Iy·q̇ = (Iz−Ix)·p·r  +  J_tp·p·Ω_net  +  U3
        #   Iz·ṙ = (Ix−Iy)·p·q                    +  U4
        #
        # Gyroscopic term J_tp·Ω_net: reaction torque from net angular
        # momentum of spinning rotors, coupling roll ↔ pitch.

        p_dot = q*r*(Iy-Iz)/Ix  -  (J_tp/Ix)*q*omega_net  +  U2/Ix
        q_dot = p*r*(Iz-Ix)/Iy  +  (J_tp/Iy)*p*omega_net  +  U3/Iy
        r_dot = p*q*(Ix-Iy)/Iz                             +  U4/Iz

        # ══ KINEMATIC ROWS ════════════════════════════════════════════════

        # World-frame velocity:  ṙ_W = R_WB · v_b
        R_wb  = R_body_to_world(phi, theta, psi)
        pos_dot = R_wb @ np.array([u_b, v_b, w_b])     # [ẋ, ẏ, ż]  [m/s]

        # Euler-angle rates:  [φ̇,θ̇,ψ̇] = T(φ,θ)·[p,q,r]
        # T-matrix singularity at |θ|≥85° raises ValueError.
        T_mat    = euler_rate_matrix(phi, theta)
        euler_dot = T_mat @ np.array([p, q, r])         # [φ̇,θ̇,ψ̇] [rad/s]

        # ── Assemble derivative vector ─────────────────────────────────────
        return np.array([
            u_dot,     v_dot,     w_dot,
            p_dot,     q_dot,     r_dot,
            pos_dot[0], pos_dot[1], pos_dot[2],
            euler_dot[0], euler_dot[1], euler_dot[2],
        ])

    # ── RK4 integrator ───────────────────────────────────────────────────────

    def step_rk4(
        self,
        x:          np.ndarray,
        U:          np.ndarray,
        omega_net:  float,
        dt:         float,
        noise_std:  Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Integrate one step using classical 4th-order Runge-Kutta.

            k1 = f(x)
            k2 = f(x + dt/2·k1)
            k3 = f(x + dt/2·k2)
            k4 = f(x + dt·k3)
            x_new = x + (dt/6)·(k1 + 2k2 + 2k3 + k4)

        Local truncation error  O(dt⁵);  global error  O(dt⁴).
        U and omega_net are held constant (zero-order hold) — consistent
        with discrete digital controllers.

        Parameters
        ----------
        x         : current state [12]
        U         : control inputs [4], constant over [t, t+dt]
        omega_net : Ω_net [rad/s], constant over the step
        dt        : step size [s]
        noise_std : optional per-state process noise σ [12];
                    additive Gaussian noise applied to x_new only
                    (not to intermediate RK slopes, preserving RK4 accuracy)

        Returns
        -------
        x_new : state at t + dt [12]
        """
        k1 = self.derivatives(x,              U, omega_net)
        k2 = self.derivatives(x + 0.5*dt*k1,  U, omega_net)
        k3 = self.derivatives(x + 0.5*dt*k2,  U, omega_net)
        k4 = self.derivatives(x +     dt*k3,  U, omega_net)

        x_new = x + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

        # Additive process noise on the integrated state
        if noise_std is not None:
            x_new += np.random.randn(12) * noise_std * np.sqrt(dt)

        # Wrap yaw to (−π, π]  to prevent unbounded accumulation
        x_new[11] = wrap_to_pi(x_new[11])

        return x_new
