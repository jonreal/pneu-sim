import numpy as np


def parse_ranges(ranges_args):
    """Parse CLI ranges like ['10-20', '30-40'] into [(10, 20), (30, 40)]."""
    out = []
    for token in ranges_args:
        try:
            a, b = token.split("-")
            out.append((int(a), int(b)))
        except Exception:
            raise SystemExit(f"Bad range format: '{token}'. Use like 10-20 30-40")
    return out


def compute_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(mask) < 2:
        return np.nan

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def compute_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(mask) < 1:
        return np.nan

    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


class MinMaxScaler:
    """Simple NumPy min-max scaler using the range [-1, 1] by default."""

    def __init__(self, feature_range=(-1.0, 1.0)):
        self.data_min = None
        self.data_max = None
        self.scale_min = float(feature_range[0])
        self.scale_max = float(feature_range[1])

    def fit(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        self.data_min = np.min(arr, axis=0)
        self.data_max = np.max(arr, axis=0)

        same = np.abs(self.data_max - self.data_min) < 1e-8
        self.data_max[same] = self.data_min[same] + 1.0

    def transform(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        arr01 = (arr - self.data_min) / (self.data_max - self.data_min)
        return arr01 * (self.scale_max - self.scale_min) + self.scale_min

    def inverse_transform(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        arr01 = (arr - self.scale_min) / (self.scale_max - self.scale_min)
        return arr01 * (self.data_max - self.data_min) + self.data_min
