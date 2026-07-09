import numpy as np

from .force_model import kang_force_numpy


def compute_joint_forces_and_accel(
    x_mm,
    dx_mm_s,
    Pf_abs_kPa,
    Pe_abs_kPa,
    Fext_N,
    coeffs,
    m_eff=304.65,
    b_eff=6500.0,
    L0_mm=200.0,
):
    """Compute antagonistic PAM forces and joint acceleration."""
    P_atm = 101.325
    kPa_per_psi = 6.89476

    dx_m_s = dx_mm_s / 1000.0

    Pf_gauge_psi = np.maximum(Pf_abs_kPa - P_atm, 0.0) / kPa_per_psi
    Pe_gauge_psi = np.maximum(Pe_abs_kPa - P_atm, 0.0) / kPa_per_psi

    eps1 = x_mm / L0_mm
    eps2 = -x_mm / L0_mm

    epsdot1 = dx_mm_s / L0_mm
    epsdot2 = -dx_mm_s / L0_mm

    F1 = kang_force_numpy(Pf_gauge_psi, eps1, epsdot1, coeffs)
    F2 = kang_force_numpy(Pe_gauge_psi, eps2, epsdot2, coeffs)

    ddx_m_s2 = (Fext_N - b_eff * dx_m_s - F2 + F1) / m_eff

    return ddx_m_s2, F1, F2


def simulate_joint(
    ts,
    x0_mm,
    dx0_mm_s,
    Pf_abs_kPa,
    Pe_abs_kPa,
    Fext_N,
    coeffs,
    m_eff=304.65,
    b_eff=6500.0,
):
    """Forward simulate displacement and velocity using measured pressures and force input."""
    T = len(ts)

    x_sim_mm = np.zeros(T)
    dx_sim_mm_s = np.zeros(T)
    ddx_sim_m_s2 = np.zeros(T)
    F1_sim = np.zeros(T)
    F2_sim = np.zeros(T)

    x_sim_mm[0] = x0_mm
    dx_sim_mm_s[0] = dx0_mm_s

    for k in range(T - 1):
        dt = ts[k + 1] - ts[k]

        ddx, F1, F2 = compute_joint_forces_and_accel(
            x_sim_mm[k],
            dx_sim_mm_s[k],
            Pf_abs_kPa[k],
            Pe_abs_kPa[k],
            Fext_N[k],
            coeffs,
            m_eff=m_eff,
            b_eff=b_eff,
        )

        ddx_sim_m_s2[k] = ddx
        F1_sim[k] = F1
        F2_sim[k] = F2

        dx_sim_m_s = dx_sim_mm_s[k] / 1000.0 + dt * ddx
        x_sim_m = x_sim_mm[k] / 1000.0 + dt * (dx_sim_mm_s[k] / 1000.0)

        dx_sim_mm_s[k + 1] = dx_sim_m_s * 1000.0
        x_sim_mm[k + 1] = x_sim_m * 1000.0

    ddx_sim_m_s2[-1], F1_sim[-1], F2_sim[-1] = compute_joint_forces_and_accel(
        x_sim_mm[-1],
        dx_sim_mm_s[-1],
        Pf_abs_kPa[-1],
        Pe_abs_kPa[-1],
        Fext_N[-1],
        coeffs,
        m_eff=m_eff,
        b_eff=b_eff,
    )

    return x_sim_mm, dx_sim_mm_s, ddx_sim_m_s2, F1_sim, F2_sim
