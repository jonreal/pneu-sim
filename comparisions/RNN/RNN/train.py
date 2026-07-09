import os
import copy
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import build_input_array, GRUDataset
from .model import GRUForwardModel, get_device, torch_load_checkpoint
from .utils import MinMaxScaler


def weighted_mse(pred, target, w_pos=100.0, w_pressure=1.0):
    weights = torch.tensor(
        [w_pos, w_pos, w_pressure, w_pressure],
        device=pred.device,
        dtype=pred.dtype,
    )
    return torch.mean(weights * (pred - target) ** 2)


def multistep_rollout_loss(
    model,
    x_seq,
    y_future,
    force_future,
    mass_pair,
    input_scaler,
    output_scaler,
    use_mass=False,
    w_pos=100.0,
    w_pressure=1.0,
):
    device = x_seq.device
    dtype = x_seq.dtype
    H = y_future.shape[1]

    in_min = torch.tensor(input_scaler.data_min, device=device, dtype=dtype)
    in_max = torch.tensor(input_scaler.data_max, device=device, dtype=dtype)
    out_min = torch.tensor(output_scaler.data_min, device=device, dtype=dtype)
    out_max = torch.tensor(output_scaler.data_max, device=device, dtype=dtype)

    weights = torch.tensor(
        [w_pos, w_pos, w_pressure, w_pressure],
        device=device,
        dtype=dtype,
    )

    preds = []

    y_next_n, h_state = model.warmup(x_seq)

    for step_idx in range(H):
        preds.append(y_next_n)

        y01 = (y_next_n + 1.0) / 2.0
        y_next_raw = y01 * (out_max - out_min) + out_min

        f_raw = force_future[:, step_idx:step_idx + 1].to(device=device, dtype=dtype)

        if use_mass:
            mf_me = mass_pair.to(device=device, dtype=dtype)
            next_input_raw = torch.cat([y_next_raw, f_raw, mf_me], dim=1)
        else:
            next_input_raw = torch.cat([y_next_raw, f_raw], dim=1)

        x01 = (next_input_raw - in_min) / (in_max - in_min)
        next_input_n = 2.0 * x01 - 1.0
        next_input_n = next_input_n.unsqueeze(1)

        if step_idx < H - 1:
            y_next_n, h_state = model.step(next_input_n, h_state)

    pred_seq = torch.stack(preds, dim=1)

    return torch.mean(weights.view(1, 1, 4) * (pred_seq - y_future) ** 2)


def train_gru(
    data_list,
    seq_len=100,
    pred_horizon=20,
    hidden_dim=96,
    num_layers=2,
    dropout=0.0,
    batch_size=256,
    lr=1e-2,
    epochs=1000,
    patience=20,
    use_mass=False,
    w_pos=100.0,
    w_pressure=1.0,
    load_checkpoint_dir="checkpoint_RNN",
    load_model_name="model_1.pt",
    save_checkpoint_dir="checkpoint_RNN",
    save_model_name="model_1.pt",
    resume=False,
    device="auto",
):
    device = get_device(device)
    os.makedirs(save_checkpoint_dir, exist_ok=True)

    all_inputs = []
    all_outputs = []

    for x0, ts, y_true, u_fn, mf, me, xeq, Force_array in data_list:
        y = np.asarray(y_true, dtype=np.float32)
        force = np.asarray(Force_array, dtype=np.float32)
        inp = build_input_array(y, force, mf, me, use_mass=use_mass)
        all_inputs.append(inp)
        all_outputs.append(y)

    all_inputs = np.concatenate(all_inputs, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    input_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    output_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))

    input_dim = all_inputs.shape[1]

    load_base_name = load_model_name.replace(".pt", "")
    save_base_name = save_model_name.replace(".pt", "")

    load_ckpt_path = os.path.join(load_checkpoint_dir, f"{load_base_name}_checkpoint.pt")
    save_ckpt_path = os.path.join(save_checkpoint_dir, f"{save_base_name}_checkpoint.pt")

    ckpt = None
    if resume:
        if not os.path.exists(load_ckpt_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {load_ckpt_path}")

        ckpt = torch_load_checkpoint(load_ckpt_path, device=device)

        if int(ckpt["input_dim"]) != int(input_dim):
            raise ValueError(f"input_dim mismatch: checkpoint={ckpt['input_dim']}, current={input_dim}")

        input_scaler.data_min = ckpt["input_scaler_min"]
        input_scaler.data_max = ckpt["input_scaler_max"]
        output_scaler.data_min = ckpt["output_scaler_min"]
        output_scaler.data_max = ckpt["output_scaler_max"]
    else:
        input_scaler.fit(all_inputs)
        output_scaler.fit(all_outputs)

    dataset = GRUDataset(
        data_list=data_list,
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        use_mass=use_mass,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    model = GRUForwardModel(
        input_dim=input_dim,
        output_dim=4,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    start_epoch = 0

    if resume:
        if int(ckpt["hidden_dim"]) != int(hidden_dim):
            raise ValueError(f"hidden_dim mismatch: checkpoint={ckpt['hidden_dim']}, current={hidden_dim}")
        if int(ckpt["num_layers"]) != int(num_layers):
            raise ValueError(f"num_layers mismatch: checkpoint={ckpt['num_layers']}, current={num_layers}")
        if int(ckpt["seq_len"]) != int(seq_len):
            raise ValueError(f"seq_len mismatch: checkpoint={ckpt['seq_len']}, current={seq_len}")
        if bool(ckpt["use_mass"]) != bool(use_mass):
            raise ValueError(f"use_mass mismatch: checkpoint={ckpt['use_mass']}, current={use_mass}")
        if int(ckpt["pred_horizon"]) != int(pred_horizon):
            raise ValueError(f"pred_horizon mismatch: checkpoint={ckpt['pred_horizon']}, current={pred_horizon}")

        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

        best_state = copy.deepcopy(model.state_dict())
        print(f"Resumed GRU from: {load_ckpt_path}")

    print("Starting GRU training...")
    print(f"Device: {device}")
    print(f"Input dim: {input_dim}")
    print(f"Use mass: {use_mass}")
    print(f"Seq len: {seq_len}")
    print(f"Pred horizon: {pred_horizon}")
    print(f"Load checkpoint: {load_ckpt_path}")
    print(f"Save checkpoint: {save_ckpt_path}")
    print("")
    print("    Epoch    EpochTime(s)    TotalTime(s)    TrainingLoss    BestLoss    BestEpoch")
    print("    _____    ____________    ____________    ____________    ________    _________")

    train_start_time = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        total_count = 0

        for x_seq, y_future, force_future, mass_pair in loader:
            x_seq = x_seq.to(device)
            y_future = y_future.to(device)
            force_future = force_future.to(device)
            mass_pair = mass_pair.to(device)

            loss = multistep_rollout_loss(
                model=model,
                x_seq=x_seq,
                y_future=y_future,
                force_future=force_future,
                mass_pair=mass_pair,
                input_scaler=input_scaler,
                output_scaler=output_scaler,
                use_mass=use_mass,
                w_pos=w_pos,
                w_pressure=w_pressure,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item()) * x_seq.shape[0]
            total_count += x_seq.shape[0]

        train_loss = total_loss / max(total_count, 1)

        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0

            ckpt = {
                "model_state": best_state,
                "optimizer_state": optimizer.state_dict(),
                "input_scaler_min": input_scaler.data_min,
                "input_scaler_max": input_scaler.data_max,
                "output_scaler_min": output_scaler.data_min,
                "output_scaler_max": output_scaler.data_max,
                "seq_len": seq_len,
                "pred_horizon": pred_horizon,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "use_mass": use_mass,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
            }

            torch.save(ckpt, save_ckpt_path)
            print(f"Saved new best GRU at epoch {epoch}, loss={best_loss:.6e}")
            print(f"   checkpoint: {save_ckpt_path}")
        else:
            no_improve += 1

        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - train_start_time

        print(
            f"{epoch:9d}"
            f"{epoch_time:16.2f}"
            f"{total_time:16.2f}"
            f"{train_loss:16.6f}"
            f"{best_loss:12.6f}"
            f"{best_epoch:13d}"
        )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}: no improvement for {patience} epochs.")
            break

    if best_state is None:
        raise RuntimeError("Training finished without saving a best model. Check loss values.")

    model.load_state_dict(best_state)
    model.eval()

    return model, input_scaler, output_scaler, {
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "seq_len": seq_len,
        "pred_horizon": pred_horizon,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "use_mass": use_mass,
        "checkpoint_path": save_ckpt_path,
    }
