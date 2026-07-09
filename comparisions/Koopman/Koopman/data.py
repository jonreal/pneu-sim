import numpy as np

from .lifting import lift_state
from .utils import StandardScaler


def dataset_to_arrays(dataset_tuple, force_scale=1.0):
    """
    Loader return format:
        x0, ts, y_true, u_fn, mf, me, xeq, Force_array

    y_true:
        [x, dx, Pf, Pe]

    Force_array:
        used as Koopman input.
        force_scale=1 keeps loader units.
        force_scale=1000 converts mN to N if needed.
    """
    x0, ts, y_true, u_fn, mf, me, xeq, Force_array = dataset_tuple

    ts = np.asarray(ts, dtype=np.float64)
    X = np.asarray(y_true, dtype=np.float64)
    U = np.asarray(Force_array, dtype=np.float64).reshape(-1, 1) / float(force_scale)

    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"Expected y_true shape [T, 4], got {X.shape}")

    if len(ts) != len(X) or len(U) != len(X):
        raise ValueError(
            f"Length mismatch: len(ts)={len(ts)}, len(X)={len(X)}, len(U)={len(U)}"
        )

    if not np.all(np.isfinite(X)):
        raise ValueError("Non-finite values found in y_true.")

    if not np.all(np.isfinite(U)):
        raise ValueError("Non-finite values found in Force_array.")

    return ts, X, U


def fit_scalers(data_list, force_scale=1.0):
    X_all = []
    U_all = []

    for data in data_list:
        _, X, U = dataset_to_arrays(data, force_scale=force_scale)
        X_all.append(X)
        U_all.append(U)

    X_all = np.vstack(X_all)
    U_all = np.vstack(U_all)

    x_scaler = StandardScaler()
    u_scaler = StandardScaler()

    x_scaler.fit(X_all)
    u_scaler.fit(U_all)

    return x_scaler, u_scaler


def build_snapshot_matrices(
    data_list,
    x_scaler,
    u_scaler,
    lift_type="poly2",
    force_scale=1.0,
):
    """
    Build:
        G = [psi(x_k); u_k]
        Y = x_{k+1}

    Snapshot pairs are formed inside each dataset only.
    The last sample of dataset i is not connected to the first sample
    of dataset i+1.
    """
    G_list = []
    Y_list = []
    Xk_list = []

    for data_idx, data in enumerate(data_list):
        _, X_raw, U_raw = dataset_to_arrays(data, force_scale=force_scale)

        if len(X_raw) < 2:
            print(f"Skipping dataset {data_idx}: too short.")
            continue

        Xn = x_scaler.transform(X_raw)
        Un = u_scaler.transform(U_raw)

        Xk = Xn[:-1, :]
        Yk = Xn[1:, :]
        Uk = Un[:-1, :]

        Zk = lift_state(Xk, lift_type=lift_type)
        Gk = np.hstack([Zk, Uk])

        G_list.append(Gk)
        Y_list.append(Yk)
        Xk_list.append(Xk)

    if not G_list:
        raise RuntimeError("No snapshot pairs were built.")

    G = np.vstack(G_list)
    Y = np.vstack(Y_list)
    Xk_all = np.vstack(Xk_list)

    return G, Y, Xk_all
