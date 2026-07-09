import numpy as np
import torch


def softplus_inv(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    return torch.log(torch.expm1(x))


def unpack_params(raw):
    """
    Constrained Kang coefficients.

    raw[0] -> cq1 > 0
    raw[1] -> cq2 < 0
    raw[2] -> cv  > 0
    raw[3] -> cc  > 0
    """
    cq1 = torch.nn.functional.softplus(raw[0])
    cq2 = -torch.nn.functional.softplus(raw[1])
    cv = torch.nn.functional.softplus(raw[2])
    cc = torch.nn.functional.softplus(raw[3])
    return cq1, cq2, cv, cc


def predict_force_torch(raw, eps, pg_psi, epsdot, sign_epsdot,
                        D0_mm, alpha0, psi_to_N_per_mm2):
    cq1, cq2, cv, cc = unpack_params(raw)

    q = 1.0 + cq1 * torch.exp(cq2 * pg_psi)

    A0 = np.pi * D0_mm ** 2 / 4.0
    pg_N_per_mm2 = pg_psi * psi_to_N_per_mm2

    geom = 3.0 * (1.0 - q * eps) ** 2 / np.tan(alpha0) ** 2 \
           - 1.0 / np.sin(alpha0) ** 2

    F_geo = A0 * pg_N_per_mm2 * geom
    F_friction = cv * epsdot + cc * sign_epsdot

    return F_geo - F_friction


def kang_force_numpy(pg_psi, eps, epsdot, coeffs,
                     D0_mm=10.0, alpha0_deg=27.5):
    """Kang-type single-PAM force model in NumPy."""
    cq1, cq2, cv, cc = coeffs

    psi_to_N_per_mm2 = 0.00689476
    alpha0 = np.deg2rad(alpha0_deg)

    pg_psi = np.asarray(pg_psi, dtype=float)
    eps = np.asarray(eps, dtype=float)
    epsdot = np.asarray(epsdot, dtype=float)

    q = 1.0 + cq1 * np.exp(cq2 * pg_psi)

    A0 = np.pi * D0_mm ** 2 / 4.0
    pg_N_per_mm2 = pg_psi * psi_to_N_per_mm2

    geom = 3.0 * (1.0 - q * eps) ** 2 / np.tan(alpha0) ** 2 \
           - 1.0 / np.sin(alpha0) ** 2

    F_geo = A0 * pg_N_per_mm2 * geom

    sign_epsdot = np.sign(epsdot)
    F_friction = cv * epsdot + cc * sign_epsdot

    return F_geo - F_friction
