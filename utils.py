"""
utils.py  ―  Quadcopter Simulation Utilities
=============================================

Rotation matrices, kinematic helpers, propulsion mixer, and trajectory
generators used across dynamics.py, controllers.py, and simulate.py.

Conventions (held throughout the entire codebase)
──────────────────────────────────────────────────
Frame   Name        x         y        z
──────  ──────────  ────────  ───────  ─────────────
W       World/NED   North     East     Down (↓ +ve)
B       Body        Forward   Right    Down

Euler-angle sequence: ZYX  (ψ yaw first, then θ pitch, then φ roll).
  R_WB = R_z(ψ) · R_y(θ) · R_x(φ)  maps body → world.

Gravity in world frame: g_W = [0, 0, g]  (positive downward, NED).
Gravity in body frame : g_B = R_WB^T · g_W
  g_B,x =  g · sin(θ)
  g_B,y = −g · cos(θ)·sin(φ)
  g_B,z =  g · cos(θ)·cos(φ)   ← positive (downward) at hover

Thrust U1 acts along body −z (upward in NED), so  ẇ − U1/m  at hover = 0.

SI units throughout (m, s, kg, rad, N, N·m).

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


# ══════════════════════════════════════
#  1.  Rotation matrices
# ══════════════════════════════════════

def Rx(phi: float) -> np.ndarray:
    """Elementary rotation about x-axis (roll φ).  R_x(φ) ∈ SO(3)."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=float)


def Ry(theta: float) -> np.ndarray:
    """Elementary rotation about y-axis (pitch θ).  R_y(θ) ∈ SO(3)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)


def Rz(psi: float) -> np.ndarray:
    """Elementary rotation about z-axis (yaw ψ).  R_z(ψ) ∈ SO(3)."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)


def R_body_to_world(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    ZYX body-to-world rotation matrix.

        R_WB = R_z(ψ) · R_y(θ) · R_x(φ)

    Usage
    -----
    v_world = R_WB @ v_body
    v_body  = R_WB.T @ v_world

    The expanded matrix (kept symbolic for clarity, computed numerically):

        R_WB = | cψcθ   cψsθsφ−sψcφ   cψsθcφ+sψsφ |
               | sψcθ   sψsθsφ+cψcφ   sψsθcφ−cψsφ |
               | −sθ       cθsφ            cθcφ    |
    """
    return Rz(psi) @ Ry(theta) @ Rx(phi)


def euler_rate_matrix(phi: float, theta: float) -> np.ndarray:
    """
    T-matrix: maps body angular rates [p, q, r]^T → Euler rates [φ̇, θ̇, ψ̇]^T.

        [φ̇]   [1  sin(φ)tan(θ)   cos(φ)tan(θ) ]   [p]
        [θ̇] = [0    cos(φ)         −sin(φ)     ] · [q]
        [ψ̇]   [0  sin(φ)/cos(θ)  cos(φ)/cos(θ)]   [r]

    Singularity
    -----------
    The matrix is singular at θ = ±π/2 (gimbal lock).
    A ValueError is raised when |θ| > 85° so callers can detect it early.
    For manoeuvres beyond ±85° use quaternion kinematics instead.
    """
    if abs(theta) >= np.radians(85.0):
        raise ValueError(
            f"Gimbal-lock singularity: |θ| = {np.degrees(abs(theta)):.1f}° ≥ 85°. "
            "Switch to quaternion representation for large pitch manoeuvres."
        )
    sp, cp = np.sin(phi), np.cos(phi)
    tt = np.tan(theta)
    ct = np.cos(theta)
    return np.array([
        [1.0,  sp * tt,  cp * tt],
        [0.0,       cp,      -sp],
        [0.0,  sp / ct,  cp / ct],
    ], dtype=float)


def wrap_to_pi(angle: float) -> float:
    """Wrap a scalar angle to (−π, π]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


# ════════════════════════════════════════════════════════════════════
#  2.  Propulsion mixer  (X-configuration, CCW = 1,3;  CW = 2,4)
# ════════════════════════════════════════════════════════════════════
#
#       motor layout (top view)
#         1(CCW) ── front ── 3(CCW)
#                \         /
#                  centre
#                /         \
#         2(CW)  ── rear ──  4(CW)
#
#   U1 =  c_T · (ω1² + ω2² + ω3² + ω4²)           total thrust     [N]
#   U2 =  c_T · L · (ω2² − ω4²)                   roll torque      [N·m]
#   U3 =  c_T · L · (ω3² − ω1²)                   pitch torque     [N·m]
#   U4 =  c_Q · (−ω1² + ω2² − ω3² + ω4²)          yaw torque       [N·m]
#
#   L = arm length (rotor centre to CoM)

def build_mixer(c_T: float, c_Q: float, L: float) -> np.ndarray:
    """
    4×4 mixer matrix M such that
        [U1, U2, U3, U4]^T = M · [ω1², ω2², ω3², ω4²]^T.

    Parameters
    ----------
    c_T : thrust coefficient   [N·s²/rad²]
    c_Q : torque coefficient   [N·m·s²/rad²]
    L   : arm length           [m]
    """
    return np.array([
        [ c_T,    c_T,    c_T,    c_T  ],
        [ 0.0,   c_T*L,   0.0,  -c_T*L],
        [-c_T*L,  0.0,   c_T*L,   0.0  ],
        [-c_Q,    c_Q,   -c_Q,    c_Q  ],
    ], dtype=float)


def rotor_speeds_from_controls(
    U:         np.ndarray,   # [U1, U2, U3, U4]
    M_inv:     np.ndarray,   # precomputed inv(mixer)
    omega_max: float,
    omega_min: float = 10.0, # non-zero minimum to model idle speed
) -> Tuple[np.ndarray, bool]:
    """
    Convert virtual controls → rotor speeds, with physical saturation.

    Returns
    -------
    omega     : [ω1, ω2, ω3, ω4]  [rad/s]
    saturated : True if any rotor was clamped
    """
    omega_sq = M_inv @ U
    omega_sq_clamped = np.clip(omega_sq, omega_min**2, omega_max**2)
    saturated = not np.allclose(omega_sq, omega_sq_clamped, atol=1e-12)
    return np.sqrt(np.maximum(omega_sq_clamped, 0.0)), saturated


# ════════════════════════════════
#  3.  Aerodynamic drag
# ════════════════════════════════

def aero_drag_body(
    v_body:  np.ndarray,   # [u, v, w]  body-frame velocity  [m/s]
    C_D:     np.ndarray,   # drag coefficients per axis       [-]
    rho:     float,        # air density                      [kg/m³]
    A_ref:   np.ndarray,   # reference areas per axis         [m²]
) -> np.ndarray:
    """
    Quadratic aerodynamic drag in the body frame.

        F_drag,i = −½ · ρ · C_D,i · A_ref,i · v_i · |v_i|     i ∈ {x,y,z}

    The signed product  v·|v|  ensures drag always opposes velocity,
    including for negative velocities (unlike v² which has no sign).

    Returns
    -------
    F_drag : body-frame drag force [N], same shape as v_body
    """
    return -0.5 * rho * C_D * A_ref * v_body * np.abs(v_body)


# ═════════════════════════════════════
#  4.  Trajectory generators
# ═════════════════════════════════════

@dataclass
class Waypoint:
    """
    A single trajectory waypoint with up to second-order derivatives.

    All quantities in world / NED frame unless noted.
    """
    t:       float               # time                     [s]
    pos:     np.ndarray          # [x, y, z]                [m]
    vel:     np.ndarray          # [ẋ, ẏ, ż]               [m/s]
    acc:     np.ndarray          # [ẍ, ÿ, z̈]              [m/s²]
    psi:     float = 0.0         # desired yaw ψ             [rad]
    psi_dot: float = 0.0         # desired yaw rate ψ̇        [rad/s]

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float)
        self.vel = np.asarray(self.vel, dtype=float)
        self.acc = np.asarray(self.acc, dtype=float)


def traj_hover(pos: np.ndarray, t: float = 0.0) -> Waypoint:
    """Static hover reference at a given NED position."""
    return Waypoint(t=t, pos=np.asarray(pos, float),
                    vel=np.zeros(3), acc=np.zeros(3))


def traj_figure8(
    t:     float,
    A:     float = 2.0,    # x-axis amplitude   [m]
    B:     float = 1.0,    # y-axis amplitude   [m]
    z0:    float = -1.5,   # constant altitude (NED, negative = above ground) [m]
    omega: float = 0.25,   # angular frequency  [rad/s]
) -> Waypoint:
    """
    Smooth Lissajous figure-8 in the horizontal plane.

        x(t) = A · sin(ω t)
        y(t) = B · sin(2ω t)
        z(t) = z0   (constant)

    All derivatives are analytic.
    """
    w = omega
    return Waypoint(
        t   = t,
        pos = np.array([ A*np.sin(w*t),       B*np.sin(2*w*t),       z0  ]),
        vel = np.array([ A*w*np.cos(w*t),   2*B*w*np.cos(2*w*t),    0.0  ]),
        acc = np.array([-A*w**2*np.sin(w*t),-4*B*w**2*np.sin(2*w*t), 0.0 ]),
    )


def traj_figure8_yaw(
    t:        float,
    A:        float = 2.0,    # x-axis amplitude          [m]
    B:        float = 1.0,    # y-axis amplitude          [m]
    z0:       float = -1.5,   # constant altitude (NED)   [m]
    omega:    float = 0.25,   # trajectory frequency      [rad/s]
    yaw_rate: float = 0.15,   # constant yaw sweep rate   [rad/s]
) -> Waypoint:
    """
    Figure-8 trajectory with a constant yaw sweep.

    Tests the yaw channel of the LPV-MPC, which is never exercised when
    psi_ref = 0 throughout (W7 fix).

    Yaw is ramped at a constant rate:  ψ(t) = yaw_rate · t (wrapped to ±π).
    All position/velocity/acceleration derivatives remain analytic.
    """
    w = omega
    psi     = float((yaw_rate * t + np.pi) % (2 * np.pi) - np.pi)
    psi_dot = yaw_rate

    return Waypoint(
        t       = t,
        pos     = np.array([ A*np.sin(w*t),       B*np.sin(2*w*t),       z0  ]),
        vel     = np.array([ A*w*np.cos(w*t),   2*B*w*np.cos(2*w*t),    0.0  ]),
        acc     = np.array([-A*w**2*np.sin(w*t),-4*B*w**2*np.sin(2*w*t), 0.0 ]),
        psi     = psi,
        psi_dot = psi_dot,
    )
