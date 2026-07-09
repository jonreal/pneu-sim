import numpy as np


def lift_state(X, lift_type="poly2"):
    """
    X: [N, 4]
       columns = [x, dx, Pf, Pe]

    The input X should already be normalized.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.shape[1] != 4:
        raise ValueError(f"Expected X shape [N, 4], got {X.shape}")

    x = X[:, 0:1]
    dx = X[:, 1:2]
    pf = X[:, 2:3]
    pe = X[:, 3:4]

    if lift_type == "linear":
        Z = np.hstack([
            np.ones_like(x),
            x,
            dx,
            pf,
            pe,
        ])

    elif lift_type == "poly2":
        Z = np.hstack([
            np.ones_like(x),

            x,
            dx,
            pf,
            pe,

            x ** 2,
            dx ** 2,
            pf ** 2,
            pe ** 2,

            x * dx,
            x * pf,
            x * pe,
            dx * pf,
            dx * pe,
            pf * pe,

            pf - pe,
            pf + pe,
        ])

    elif lift_type == "poly3":
        Z = np.hstack([
            np.ones_like(x),

            x,
            dx,
            pf,
            pe,

            x ** 2,
            dx ** 2,
            pf ** 2,
            pe ** 2,

            x * dx,
            x * pf,
            x * pe,
            dx * pf,
            dx * pe,
            pf * pe,

            pf - pe,
            pf + pe,

            x ** 3,
            dx ** 3,
            pf ** 3,
            pe ** 3,

            x * (pf - pe),
            dx * (pf - pe),
            x ** 2 * pf,
            x ** 2 * pe,
            dx ** 2 * pf,
            dx ** 2 * pe,
        ])

    else:
        raise ValueError(
            f"Unknown lift_type='{lift_type}'. "
            "Use 'linear', 'poly2', or 'poly3'."
        )

    if not np.all(np.isfinite(Z)):
        raise ValueError("Non-finite values found in lifted state.")

    return Z
