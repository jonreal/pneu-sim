import os
import numpy as np
import torch

from .dataset import build_input_array
from .model import get_device
from .utils import compute_r2, compute_rmse


@torch.no_grad()
def rollout_gru(
    model,
    dataset_tuple,
    input_scaler,
    output_scaler,
    seq_len,
    use_mass=False,
    device="auto",
):
    device = get_device(device)

    x0, ts, y_true, u_fn, mf, me, xeq, Force_array = dataset_tuple

    y_true = np.asarray(y_true, dtype=np.float32)
    force = np.asarray(Force_array, dtype=np.float32)
    ts = np.asarray(ts)

    T = len(y_true)
    y_pred = np.zeros_like(y_true, dtype=np.float32)

    warmup_len = min(int(seq_len), T)
    y_pred[:warmup_len, :] = y_true[:warmup_len, :]

    if T <= seq_len:
        return ts, y_true, y_pred, force

    model.eval()

    inp_hist = build_input_array(
        y_true[:seq_len, :],
        force[:seq_len],
        mf=mf,
        me=me,
        use_mass=use_mass,
    )

    inp_hist_n = input_scaler.transform(inp_hist).astype(np.float32)
    x_seq = torch.from_numpy(inp_hist_n[None, :, :]).to(device)

    y_next_n, h_state = model.warmup(x_seq)

    for k in range(seq_len, T):
        y_k = output_scaler.inverse_transform(y_next_n.cpu().numpy()[0])
        y_pred[k, :] = y_k.astype(np.float32)

        if k < T - 1:
            input_k = build_input_array(
                y_pred[k:k + 1, :],
                force[k:k + 1],
                mf=mf,
                me=me,
                use_mass=use_mass,
            )

            input_k_n = input_scaler.transform(input_k).astype(np.float32)
            x_step_k = torch.from_numpy(input_k_n[None, :, :]).to(device)

            y_next_n, h_state = model.step(x_step_k, h_state)

    return ts, y_true, y_pred, force


def evaluate_gru(
    model,
    data_list,
    input_scaler,
    output_scaler,
    seq_len,
    use_mass=False,
    save_path="checkpoint_RNN/GRU_R2_results.csv",
    device="auto",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    results = []

    for idx, data in enumerate(data_list):
        ts, y_true, y_pred, force = rollout_gru(
            model,
            data,
            input_scaler,
            output_scaler,
            seq_len=seq_len,
            use_mass=use_mass,
            device=device,
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

    print(f"Saved GRU R2/RMSE results to: {save_path}")

    return results


def evaluate_node_style_loss_gru(
    model,
    data_list,
    input_scaler,
    output_scaler,
    seq_len,
    use_mass=False,
    device="auto",
):
    losses = []

    for idx, data in enumerate(data_list):
        ts, y_true, y_pred, force = rollout_gru(
            model,
            data,
            input_scaler,
            output_scaler,
            seq_len=seq_len,
            use_mass=use_mass,
            device=device,
        )

        mse_pos = np.mean((y_pred[:, 0:2] - y_true[:, 0:2]) ** 2)
        mse_pressure = np.mean((y_pred[:, 2:4] - y_true[:, 2:4]) ** 2)

        loss = 100.0 * mse_pos + mse_pressure
        losses.append(loss)

        print(
            f"Dataset {idx + 1}: "
            f"NODE-style GRU loss={loss:.6f}, "
            f"mse_pos={mse_pos:.6e}, "
            f"mse_pressure={mse_pressure:.6e}"
        )

    mean_loss = float(np.mean(losses))
    print(f"Mean NODE-style GRU loss over {len(data_list)} datasets: {mean_loss:.6f}")

    return mean_loss, np.asarray(losses)
