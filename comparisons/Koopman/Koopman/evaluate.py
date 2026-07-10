import os
import numpy as np

from .model import rollout_koopman
from .utils import compute_r2, compute_rmse


def evaluate_koopman(
    model_dict,
    data_list,
    save_path,
    warmup_steps=1,
    clip_x_norm=8.0,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    results = []

    for idx, data in enumerate(data_list):
        ts, y_true, y_pred, force = rollout_koopman(
            model_dict,
            data,
            warmup_steps=warmup_steps,
            clip_x_norm=clip_x_norm,
        )

        mf = data[4]
        me = data[5]

        R2_x = compute_r2(y_true[:, 0], y_pred[:, 0])
        R2_dx = compute_r2(y_true[:, 1], y_pred[:, 1])
        R2_Pf = compute_r2(y_true[:, 2], y_pred[:, 2])
        R2_Pe = compute_r2(y_true[:, 3], y_pred[:, 3])

        RMSE_x = compute_rmse(y_true[:, 0], y_pred[:, 0])
        RMSE_dx = compute_rmse(y_true[:, 1], y_pred[:, 1])
        RMSE_Pf = compute_rmse(y_true[:, 2], y_pred[:, 2])
        RMSE_Pe = compute_rmse(y_true[:, 3], y_pred[:, 3])

        results.append([
            idx + 1,
            float(mf),
            float(me),
            R2_x,
            R2_dx,
            R2_Pf,
            R2_Pe,
            RMSE_x,
            RMSE_dx,
            RMSE_Pf,
            RMSE_Pe,
        ])

        print(
            f"Dataset {idx + 1}: "
            f"mf={float(mf):.4f}, me={float(me):.4f}, "
            f"R2_x={R2_x:.4f}, R2_dx={R2_dx:.4f}, "
            f"R2_Pf={R2_Pf:.4f}, R2_Pe={R2_Pe:.4f}, "
            f"RMSE_x={RMSE_x:.4f}, RMSE_dx={RMSE_dx:.4f}, "
            f"RMSE_Pf={RMSE_Pf:.4f}, RMSE_Pe={RMSE_Pe:.4f}"
        )

    header = (
        "dataset_id,mf,me,"
        "R2_x,R2_dx,R2_Pf,R2_Pe,"
        "RMSE_x,RMSE_dx,RMSE_Pf,RMSE_Pe"
    )

    results = np.asarray(results, dtype=np.float64)
    np.savetxt(save_path, results, delimiter=",", header=header, comments="")

    print(f"\nSaved Koopman R2/RMSE results to: {save_path}")

    return results


def evaluate_node_style_loss_koopman(
    model_dict,
    data_list,
    warmup_steps=1,
    clip_x_norm=8.0,
):
    losses = []

    for idx, data in enumerate(data_list):
        ts, y_true, y_pred, force = rollout_koopman(
            model_dict,
            data,
            warmup_steps=warmup_steps,
            clip_x_norm=clip_x_norm,
        )

        mse_pos = np.mean((y_pred[:, 0:2] - y_true[:, 0:2]) ** 2)
        mse_pressure = np.mean((y_pred[:, 2:4] - y_true[:, 2:4]) ** 2)

        loss = 100.0 * mse_pos + mse_pressure
        losses.append(loss)

        print(
            f"Dataset {idx + 1}: "
            f"NODE-style Koopman loss={loss:.6f}, "
            f"mse_pos={mse_pos:.6e}, "
            f"mse_pressure={mse_pressure:.6e}"
        )

    mean_loss = float(np.mean(losses))
    print(f"\nMean NODE-style Koopman loss over {len(data_list)} datasets: {mean_loss:.6f}")

    return mean_loss, np.asarray(losses)
