import json
import os
import numpy as np

from .dynamics import compute_joint_forces_and_accel, simulate_joint
from .utils import compute_r2, compute_rmse


def save_fit_summary(fit_info, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "Kang_fitted_coefficients.json")
    txt_path = os.path.join(out_dir, "Kang_fitted_coefficients.txt")

    with open(json_path, "w") as f:
        json.dump(fit_info, f, indent=2)

    with open(txt_path, "w") as f:
        f.write("Kang analytical fitted coefficients\n")
        f.write("====================================\n")
        for key, value in fit_info.items():
            f.write(f"{key}: {value}\n")

    print(f"Saved Kang coefficient summary to: {json_path}", flush=True)
    print(f"Saved Kang coefficient summary to: {txt_path}", flush=True)


def run_joint_dynamics_one_dataset(
    coeffs,
    dataset_tuple,
    code,
    out_dir,
    save_response_data=False,
    m_eff=304.65,
    b_eff=6500.0,
):
    x0, ts, y_true, u_fn, mf, me, xeq, Force_array = dataset_tuple

    ts = np.asarray(ts, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    Force_array = np.asarray(Force_array, dtype=float)

    x_mm = y_true[:, 0]
    dx_mm_s = y_true[:, 1]
    Pf_abs_kPa = y_true[:, 2]
    Pe_abs_kPa = y_true[:, 3]

    Fext_N = Force_array / 1000.0
    ddx_meas_m_s2 = np.gradient(dx_mm_s / 1000.0, ts)

    ddx_model_m_s2, F1, F2 = compute_joint_forces_and_accel(
        x_mm,
        dx_mm_s,
        Pf_abs_kPa,
        Pe_abs_kPa,
        Fext_N,
        coeffs,
        m_eff=m_eff,
        b_eff=b_eff,
    )

    x_sim_mm, dx_sim_mm_s, ddx_sim_m_s2, F1_sim, F2_sim = simulate_joint(
        ts,
        x_mm[0],
        dx_mm_s[0],
        Pf_abs_kPa,
        Pe_abs_kPa,
        Fext_N,
        coeffs,
        m_eff=m_eff,
        b_eff=b_eff,
    )

    R2_x = compute_r2(x_mm, x_sim_mm)
    R2_dx = compute_r2(dx_mm_s, dx_sim_mm_s)

    RMSE_x = compute_rmse(x_mm, x_sim_mm)
    RMSE_dx = compute_rmse(dx_mm_s, dx_sim_mm_s)

    if save_response_data:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"Kang_response_data_nn_{code}.npz")

        np.savez(
            out_path,
            code=code,
            ts=ts,
            x_gt_mm=x_mm,
            x_model_mm=x_sim_mm,
            dx_gt_mm_s=dx_mm_s,
            dx_model_mm_s=dx_sim_mm_s,
            ddx_gt_m_s2=ddx_meas_m_s2,
            ddx_model_m_s2=ddx_model_m_s2,
            ddx_sim_m_s2=ddx_sim_m_s2,
            Fext_N=Fext_N,
            F1_N=F1,
            F2_N=F2,
            F1_sim_N=F1_sim,
            F2_sim_N=F2_sim,
            Pf_abs_kPa=Pf_abs_kPa,
            Pe_abs_kPa=Pe_abs_kPa,
            coeffs=coeffs,
            m_eff=m_eff,
            b_eff=b_eff,
        )

        print(f"Saved response data to: {out_path}", flush=True)

    print(
        f"nn_{code}: "
        f"mf={float(mf):.6f}, me={float(me):.6f}, "
        f"R2_x={R2_x:.6f}, R2_dx={R2_dx:.6f}, "
        f"RMSE_x={RMSE_x:.6f} mm, RMSE_dx={RMSE_dx:.6f} mm/s",
        flush=True,
    )

    return [
        code,
        float(mf),
        float(me),
        R2_x,
        R2_dx,
        RMSE_x,
        RMSE_dx,
    ]


def evaluate_kang(
    coeffs,
    data_items,
    out_dir="checkpoint_Kang",
    save_response_data=False,
    m_eff=304.65,
    b_eff=6500.0,
):
    """
    Evaluate the Kang analytical model on joint datasets.

    data_items should be a list of:
        (code, dataset_tuple)
    """
    os.makedirs(out_dir, exist_ok=True)

    results = []

    for code, dataset_tuple in data_items:
        row = run_joint_dynamics_one_dataset(
            coeffs=coeffs,
            dataset_tuple=dataset_tuple,
            code=code,
            out_dir=out_dir,
            save_response_data=save_response_data,
            m_eff=m_eff,
            b_eff=b_eff,
        )
        results.append(row)

    csv_path = os.path.join(out_dir, "Kang_R2_results.csv")
    header = "dataset,mf,me,R2_x,R2_dx,RMSE_x,RMSE_dx"

    with open(csv_path, "w") as f:
        f.write(header + "\n")

        for row in results:
            f.write(
                f"{row[0]},"
                f"{row[1]:.8f},"
                f"{row[2]:.8f},"
                f"{row[3]:.8f},"
                f"{row[4]:.8f},"
                f"{row[5]:.8f},"
                f"{row[6]:.8f}\n"
            )

    print(f"\nSaved Kang R2/RMSE results to: {csv_path}", flush=True)

    return results
