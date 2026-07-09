import torch
import torch.nn as nn

from .utils import MinMaxScaler


def get_device(device="auto"):
    if device is None or str(device).lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def torch_load_checkpoint(path, device="auto"):
    device = get_device(device)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class GRUForwardModel(nn.Module):
    def __init__(self, input_dim, output_dim=4, hidden_dim=96, num_layers=2, dropout=0.0):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_seq):
        out, h = self.gru(x_seq)
        y_next = self.head(out[:, -1, :])
        return y_next

    def warmup(self, x_seq):
        out, h = self.gru(x_seq)
        y_next = self.head(out[:, -1, :])
        return y_next, h

    def step(self, x_step, h):
        out, h = self.gru(x_step, h)
        y_next = self.head(out[:, -1, :])
        return y_next, h


def load_gru_checkpoint(checkpoint_path, device="auto"):
    """Load a saved GRU checkpoint for evaluation or rollout."""
    device = get_device(device)
    ckpt = torch_load_checkpoint(checkpoint_path, device=device)

    model = GRUForwardModel(
        input_dim=int(ckpt["input_dim"]),
        output_dim=4,
        hidden_dim=int(ckpt["hidden_dim"]),
        num_layers=int(ckpt["num_layers"]),
        dropout=float(ckpt.get("dropout", 0.0)),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    input_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    output_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))

    input_scaler.data_min = ckpt["input_scaler_min"]
    input_scaler.data_max = ckpt["input_scaler_max"]
    output_scaler.data_min = ckpt["output_scaler_min"]
    output_scaler.data_max = ckpt["output_scaler_max"]

    info = {
        "seq_len": int(ckpt["seq_len"]),
        "pred_horizon": int(ckpt["pred_horizon"]),
        "input_dim": int(ckpt["input_dim"]),
        "hidden_dim": int(ckpt["hidden_dim"]),
        "num_layers": int(ckpt["num_layers"]),
        "dropout": float(ckpt.get("dropout", 0.0)),
        "use_mass": bool(ckpt["use_mass"]),
        "best_loss": float(ckpt.get("best_loss", float("nan"))),
        "best_epoch": int(ckpt.get("best_epoch", -1)),
    }

    print(f"Loaded GRU checkpoint from: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Model info: {info}")

    return model, input_scaler, output_scaler, info
