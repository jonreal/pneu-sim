import os
import re
import numpy as np


def parse_ranges(ranges_args):
    """Parse CLI ranges like ['10-20', '30-40'] -> [(10, 20), (30, 40)]."""
    out = []
    for token in ranges_args:
        try:
            a, b = token.split("-")
            out.append((int(a), int(b)))
        except Exception:
            raise SystemExit(f"Bad range format: '{token}'. Use like 10-20 30-40")
    return out


def compute_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(mask) < 2:
        return np.nan

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def compute_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(mask) < 1:
        return np.nan

    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def code_from_file_path(file_path, prefix="nn_"):
    """Extract code from filenames like nn_1010 or nn_1010.txt."""
    name = os.path.basename(str(file_path))
    if name.startswith(prefix):
        name = name[len(prefix):]
    name = os.path.splitext(name)[0]

    match = re.search(r"\d+", name)
    return match.group(0) if match else name
