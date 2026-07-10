import os
import numpy as np
import torch
import scipy.signal as signal

from .force_model import softplus_inv, unpack_params, predict_force_torch


def embedded_process_data_py(file_path):
    """Load the raw single-PAM data format used by the Kang fitting script."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    line_cnt = 0
    params = {}
    fs = None

    with open(file_path, "r", errors="ignore") as f:
        for _ in range(8):
            f.readline()
            line_cnt += 1

        while True:
            line = f.readline()
            line_cnt += 1

            if line.strip() == "#":
                break

            if "#" in line and "=" in line:
                txt = line.split("#", 1)[1].strip()
                name, val = txt.split("=", 1)
                name = name.strip()
                val = val.strip().split()[0]

                try:
                    params[name] = float(val)
                except ValueError:
                    pass

                if name == "Fs":
                    fs = params[name]

        header = f.readline()
        line_cnt += 1

    labels = header.split()
    if len(labels) > 0 and labels[0] == "#":
        labels = labels[1:]

    D = np.genfromtxt(file_path, skip_header=line_cnt)

    if D.ndim == 1:
        D = D.reshape(1, -1)

    if D.shape[1] != len(labels):
        raise ValueError(f"Labels/column mismatch: labels={len(labels)}, data columns={D.shape[1]}")

    order = np.argsort(D[:, 0])
    D = D[order, :]

    _, unique_idx = np.unique(D[:, 0], return_index=True)
    D = D[np.sort(unique_idx), :]

    if fs is None:
        fs = 1000.0

    dt_stamp = np.diff(D[:, 0])
    gap_idx = np.where(dt_stamp != 1)[0]

    if len(gap_idx) > 0:
        rows = []
        start = 0

        for gi in gap_idx:
            rows.append(D[start:gi + 1, :])

            frame1 = int(D[gi, 0] + 1)
            frame2 = int(D[gi + 1, 0] - 1)

            if frame2 >= frame1:
                Dnew = np.full((frame2 - frame1 + 1, D.shape[1]), np.nan)
                Dnew[:, 0] = np.arange(frame1, frame2 + 1)
                rows.append(Dnew)

            start = gi + 1

        rows.append(D[start:, :])
        D = np.vstack(rows)

    data = {"time": (D[:, 0] - D[0, 0]) * (1.0 / fs)}

    for j, lab in enumerate(labels):
        data[lab] = D[:, j]

    return {"data": data, "params": params}


def branch_filter(y, b, a):
    y = np.asarray(y).reshape(-1)
    if len(y) > 3 * max(len(a), len(b)):
        return signal.filtfilt(b, a, y)
    return y


def filtered_velocity_from_indices(x_all, idx, dt, b, a):
    idx = np.asarray(idx).astype(int).reshape(-1)
    v_selected = np.zeros(len(idx))

    if len(idx) == 0:
        return v_selected

    breaks = np.r_[0, np.where(np.diff(idx) > 1)[0] + 1, len(idx)]

    for s in range(len(breaks) - 1):
        local = np.arange(breaks[s], breaks[s + 1])
        idx_seg = idx[local]

        if len(idx_seg) < 2:
            v_selected[local] = 0.0
            continue

        x_seg = x_all[idx_seg]
        x_seg_filt = branch_filter(x_seg, b, a)

        if np.any(~np.isfinite(x_seg_filt)):
            good = np.isfinite(x_seg_filt)
            if np.sum(good) < 2:
                v_selected[local] = 0.0
                continue

            x_seg_filt = np.interp(
                np.arange(len(x_seg_filt)),
                np.where(good)[0],
                x_seg_filt[good],
            )

        v_seg = np.gradient(x_seg_filt, dt)
        v_seg = branch_filter(v_seg, b, a)

        v_selected[local] = v_seg

    return v_selected


def fit_kang_single_pam(
    single_pam_dir,
    ids=None,
    c_force=4.4482,
    fs=1000,
    D0_mm=10.0,
    L0_mm=200.0,
    alpha0_deg=27.5,
    loadingspeed=0.03,
    disp_min_mm=-25.0,
    disp_max_mm=5.0,
    adam_steps=3000,
    lbfgs_steps=1000,
    device=None,
):
    """
    Fit Kang coefficients from single-PAM data.

    Returns:
        coeffs: np.ndarray [cq1, cq2, cv, cc]
        fit_info: dict with RMSE, MAE, R2, and settings
    """
    if ids is None:
        ids = list(range(0, 71, 10))

    ids = [int(i) for i in ids]

    psi_to_N_per_mm2 = 0.00689476
    dt = 1.0 / fs
    alpha0 = np.deg2rad(alpha0_deg)

    noloadperiod = np.arange(4999, 9000)
    standbyperiod = np.arange(7999, 9000)

    startpoint_matlab = 9000
    start_idx = startpoint_matlab - 1

    bl, al = signal.butter(1, 2 * 1 / 1000, btype="low")

    fc_vel = 2
    bv, av = signal.butter(2, fc_vel / (fs / 2), btype="low")

    S = {}
    Pd = {}

    for i in ids:
        file_path = os.path.join(single_pam_dir, f"single{i:02d}")
        raw = embedded_process_data_py(file_path)

        force = np.asarray(raw["data"]["force"]).reshape(-1)
        position = np.asarray(raw["data"]["position"]).reshape(-1)
        pm = np.asarray(raw["data"]["pm"]).reshape(-1)

        force_N = (force - np.nanmean(force[noloadperiod])) * c_force

        m = np.where(position != 0)[0]
        if len(m) == 0:
            raise ValueError(f"No nonzero position found in single{i:02d}.")

        position_mm = position - position[m[0] + 1]
        pm_psi = pm.copy()

        S[i] = {
            "force_N": force_N,
            "position_mm": position_mm,
            "pm_psi": pm_psi,
        }

        Pd[i] = np.nanmean(pm_psi[standbyperiod])

    Data_up = {}
    Data_dn = {}

    for i in ids:
        F = S[i]["force_N"]
        x = S[i]["position_mm"]
        p = S[i]["pm_psi"]

        gradData = np.gradient(F[start_idx:])
        gradFilt = signal.lfilter(bl, al, gradData)

        idx_up = np.where(gradFilt > loadingspeed)[0] + startpoint_matlab
        idx_dn = np.where(gradFilt < -loadingspeed)[0] + startpoint_matlab

        idx_up = idx_up[idx_up < len(F)]
        idx_dn = idx_dn[idx_dn < len(F)]

        xdot_up = filtered_velocity_from_indices(x, idx_up, dt, bv, av)
        xdot_dn = filtered_velocity_from_indices(x, idx_dn, dt, bv, av)

        epsdot_up = -xdot_up / L0_mm
        epsdot_dn = -xdot_dn / L0_mm

        Pd_vec = np.full(len(F), Pd[i])

        Data_up[i] = np.column_stack([
            Pd_vec[idx_up], x[idx_up], F[idx_up], p[idx_up], epsdot_up
        ])

        Data_dn[i] = np.column_stack([
            Pd_vec[idx_dn], x[idx_dn], F[idx_dn], p[idx_dn], epsdot_dn
        ])

    eps_all = []
    F_all = []
    pg_all = []
    epsdot_all = []
    branch_all = []

    for i in ids:
        up = Data_up[i]
        dn = Data_dn[i]

        up = up[(up[:, 1] >= disp_min_mm) & (up[:, 1] <= disp_max_mm)]
        dn = dn[(dn[:, 1] >= disp_min_mm) & (dn[:, 1] <= disp_max_mm)]

        eps_up = -up[:, 1] / L0_mm
        eps_dn = -dn[:, 1] / L0_mm

        eps_all.append(eps_up)
        eps_all.append(eps_dn)

        F_all.append(up[:, 2])
        F_all.append(dn[:, 2])

        pg_all.append(up[:, 3])
        pg_all.append(dn[:, 3])

        epsdot_all.append(up[:, 4])
        epsdot_all.append(dn[:, 4])

        branch_all.append(-np.ones(len(up)))
        branch_all.append(np.ones(len(dn)))

    eps_all = np.concatenate(eps_all)
    F_all = np.concatenate(F_all)
    pg_all = np.concatenate(pg_all)
    epsdot_all = np.concatenate(epsdot_all)
    branch_all = np.concatenate(branch_all)

    valid = np.isfinite(eps_all) & np.isfinite(F_all) & np.isfinite(pg_all) & \
            np.isfinite(epsdot_all) & (pg_all >= 0)

    eps_all = eps_all[valid]
    F_all = F_all[valid]
    pg_all = pg_all[valid]
    epsdot_all = epsdot_all[valid]
    branch_all = branch_all[valid]

    if len(F_all) == 0:
        raise ValueError("No valid fitting data after displacement range selection.")

    sign_epsdot = np.sign(epsdot_all)
    zero_idx = np.abs(sign_epsdot) < 1e-9
    sign_epsdot[zero_idx] = branch_all[zero_idx]

    print("\n=========== SINGLE-PAM FIT DATA CHECK ===========", flush=True)
    print(f"N samples      = {len(F_all)}", flush=True)
    print(f"epsilon range  = [{eps_all.min():.6f}, {eps_all.max():.6f}]", flush=True)
    print(f"pg range       = [{pg_all.min():.4f}, {pg_all.max():.4f}] psi gauge", flush=True)
    print(f"force range    = [{F_all.min():.4f}, {F_all.max():.4f}] N", flush=True)
    print(f"epsdot range   = [{epsdot_all.min():.6f}, {epsdot_all.max():.6f}] 1/s", flush=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    eps_t = torch.tensor(eps_all, dtype=torch.float64, device=device)
    F_t = torch.tensor(F_all, dtype=torch.float64, device=device)
    pg_t = torch.tensor(pg_all, dtype=torch.float64, device=device)
    epsdot_t = torch.tensor(epsdot_all, dtype=torch.float64, device=device)
    sign_t = torch.tensor(sign_epsdot, dtype=torch.float64, device=device)

    p0 = np.array([2.25, -0.0262, 200.0, 40.0], dtype=float)

    raw0 = torch.stack([
        softplus_inv(p0[0]),
        softplus_inv(-p0[1]),
        softplus_inv(p0[2]),
        softplus_inv(p0[3]),
    ]).to(device)

    raw = torch.nn.Parameter(raw0.clone())

    opt = torch.optim.Adam([raw], lr=1e-2)

    for _ in range(int(adam_steps)):
        opt.zero_grad()
        F_pred = predict_force_torch(
            raw, eps_t, pg_t, epsdot_t, sign_t,
            D0_mm, alpha0, psi_to_N_per_mm2,
        )
        loss = torch.mean((F_pred - F_t) ** 2)
        loss.backward()
        opt.step()

    opt_lbfgs = torch.optim.LBFGS(
        [raw],
        lr=1.0,
        max_iter=int(lbfgs_steps),
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt_lbfgs.zero_grad()
        F_pred = predict_force_torch(
            raw, eps_t, pg_t, epsdot_t, sign_t,
            D0_mm, alpha0, psi_to_N_per_mm2,
        )
        loss = torch.mean((F_pred - F_t) ** 2)
        loss.backward()
        return loss

    opt_lbfgs.step(closure)

    with torch.no_grad():
        F_pred_t = predict_force_torch(
            raw, eps_t, pg_t, epsdot_t, sign_t,
            D0_mm, alpha0, psi_to_N_per_mm2,
        )
        cq1, cq2, cv, cc = unpack_params(raw)

        F_pred = F_pred_t.cpu().numpy()
        coeffs = np.array([cq1.item(), cq2.item(), cv.item(), cc.item()])

    rmse = np.sqrt(np.mean((F_pred - F_all) ** 2))
    mae = np.mean(np.abs(F_pred - F_all))
    r2 = 1.0 - np.sum((F_all - F_pred) ** 2) / np.sum((F_all - np.mean(F_all)) ** 2)
    resnorm = np.sum((F_pred - F_all) ** 2)

    fit_info = {
        "cq1": float(coeffs[0]),
        "cq2": float(coeffs[1]),
        "cv": float(coeffs[2]),
        "cc": float(coeffs[3]),
        "alpha0_deg": float(alpha0_deg),
        "D0_mm": float(D0_mm),
        "L0_mm": float(L0_mm),
        "rmse_N": float(rmse),
        "mae_N": float(mae),
        "r2": float(r2),
        "resnorm": float(resnorm),
        "n_samples": int(len(F_all)),
        "single_pam_ids": ids,
        "single_pam_dir": str(single_pam_dir),
        "adam_steps": int(adam_steps),
        "lbfgs_steps": int(lbfgs_steps),
    }

    print("\n=========== FITTED KANG COEFFICIENTS ===========", flush=True)
    print(f"cq1        = {coeffs[0]:.10g}", flush=True)
    print(f"cq2        = {coeffs[1]:.10g}  [1/psi]", flush=True)
    print(f"cv         = {coeffs[2]:.10g}  [N*s]", flush=True)
    print(f"cc         = {coeffs[3]:.10g}  [N]", flush=True)
    print(f"alpha0     = {alpha0_deg:.10g}  [deg, fixed]", flush=True)
    print(f"RMSE       = {rmse:.6f} N", flush=True)
    print(f"MAE        = {mae:.6f} N", flush=True)
    print(f"R2         = {r2:.6f}", flush=True)
    print(f"resnorm    = {resnorm:.10g}", flush=True)

    return coeffs, fit_info
