import numpy as np
import torch
from torch.utils.data import Dataset


def build_input_array(y, force, mf=None, me=None, use_mass=False):
    """
    Build GRU input features.

    Without mass:
        [x, dx, Pf, Pe, Force]

    With mass:
        [x, dx, Pf, Pe, Force, mf, me]
    """
    y = np.asarray(y, dtype=np.float32)
    force = np.asarray(force, dtype=np.float32).reshape(-1, 1)

    if use_mass:
        mf_col = np.ones((len(y), 1), dtype=np.float32) * float(mf)
        me_col = np.ones((len(y), 1), dtype=np.float32) * float(me)
        return np.concatenate([y, force, mf_col, me_col], axis=1)

    return np.concatenate([y, force], axis=1)


class GRUDataset(Dataset):
    """Sequence dataset for recursive multi-step GRU training."""

    def __init__(
        self,
        data_list,
        seq_len,
        pred_horizon,
        input_scaler,
        output_scaler,
        use_mass=False,
        stride=10,
    ):
        self.data_list = data_list
        self.seq_len = int(seq_len)
        self.pred_horizon = int(pred_horizon)
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.use_mass = bool(use_mass)
        self.stride = int(stride)

        self.inputs = []
        self.outputs = []
        self.forces = []
        self.masses = []
        self.lengths = []

        for x0, ts, y_true, u_fn, mf, me, xeq, Force_array in data_list:
            y = np.asarray(y_true, dtype=np.float32)
            force = np.asarray(Force_array, dtype=np.float32)

            inp = build_input_array(y, force, mf, me, use_mass=self.use_mass)

            inp_n = input_scaler.transform(inp).astype(np.float32)
            y_n = output_scaler.transform(y).astype(np.float32)

            self.inputs.append(inp_n)
            self.outputs.append(y_n)
            self.forces.append(force.astype(np.float32))
            self.masses.append((float(mf), float(me)))

            n_valid = len(y) - self.seq_len - self.pred_horizon + 1
            self.lengths.append(max(0, (n_valid + self.stride - 1) // self.stride))

        total = int(np.sum(self.lengths))
        if total <= 0:
            raise RuntimeError(
                "No valid GRU training sequences were built. "
                "Reduce --seq-len or --pred-horizon, or check dataset length."
            )

        self.cum_lengths = np.cumsum(self.lengths)
        self.total_len = int(self.cum_lengths[-1])

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        ds_idx = int(np.searchsorted(self.cum_lengths, idx, side="right"))
        prev = 0 if ds_idx == 0 else self.cum_lengths[ds_idx - 1]
        local_idx = int(idx - prev)

        k = local_idx * self.stride + self.seq_len

        x_seq = self.inputs[ds_idx][k - self.seq_len:k, :]
        y_future = self.outputs[ds_idx][k:k + self.pred_horizon, :]

        force_future = self.forces[ds_idx][k:k + self.pred_horizon]
        mf, me = self.masses[ds_idx]

        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(y_future),
            torch.from_numpy(force_future),
            torch.tensor([mf, me], dtype=torch.float32),
        )
