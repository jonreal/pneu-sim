import numpy as np


def parse_ranges(ranges_args):
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


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, arr):
        arr = np.asarray(arr, dtype=np.float64)
        self.mean = np.mean(arr, axis=0)
        self.std = np.std(arr, axis=0)

        small = self.std < 1e-10
        self.std[small] = 1.0

    def transform(self, arr):
        arr = np.asarray(arr, dtype=np.float64)
        return (arr - self.mean) / self.std

    def inverse_transform(self, arr):
        arr = np.asarray(arr, dtype=np.float64)
        return arr * self.std + self.mean
