import os
import numpy as np
import matplotlib.pyplot as plt

from .model import rollout_koopman


def save_koopman_response_data(ts, y_true, y_pred, force, out_dir, dataset_idx):
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"Koopman_response_data_dataset_{dataset_idx + 1}.npz")

    np.savez(
        out_path,
        ts=np.asarray(ts),
        y_true=np.asarray(y_true),
        y_model=np.asarray(y_pred),
        y_pred=np.asarray(y_pred),
        Force_array=np.asarray(force),

        x_true_mm=np.asarray(y_true)[:, 0],
        x_model_mm=np.asarray(y_pred)[:, 0],

        dx_true_mm_s=np.asarray(y_true)[:, 1],
        dx_model_mm_s=np.asarray(y_pred)[:, 1],

        Pf_true_kPa=np.asarray(y_true)[:, 2],
        Pf_model_kPa=np.asarray(y_pred)[:, 2],

        Pe_true_kPa=np.asarray(y_true)[:, 3],
        Pe_model_kPa=np.asarray(y_pred)[:, 3],
    )

    print(f"Saved Koopman response data to: {out_path}")


def plot_koopman_datasets(
    model_dict,
    data_list,
    which="all",
    save=False,
    out_dir="checkpoint_Koopman",
    warmup_steps=1,
    clip_x_norm=8.0,
):
    os.makedirs(out_dir, exist_ok=True)

    if which == "all":
        indices = list(range(len(data_list)))
    else:
        indices = list(which)

    labels = [
        r"$x$ (mm)",
        r"$\dot{x}$ (mm/s)",
        r"$P_f$ (kPa)",
        r"$P_e$ (kPa)",
    ]

    for idx in indices:
        data = data_list[idx]

        ts, y_true, y_pred, force = rollout_koopman(
            model_dict,
            data,
            warmup_steps=warmup_steps,
            clip_x_norm=clip_x_norm,
        )

        save_koopman_response_data(
            ts=ts,
            y_true=y_true,
            y_pred=y_pred,
            force=force,
            out_dir=out_dir,
            dataset_idx=idx,
        )

        fig, axs = plt.subplots(5, 1, figsize=(12, 8), sharex=True)

        for i in range(4):
            axs[i].plot(ts, y_true[:, i], "k--", linewidth=1.0, label="Truth")
            axs[i].plot(ts, y_pred[:, i], linewidth=1.2, label="Koopman")
            axs[i].set_ylabel(labels[i])
            axs[i].grid(True, linestyle=":", linewidth=0.5)

            if i == 0:
                axs[i].legend(frameon=False, loc="best")

        axs[4].plot(ts, force / 1000.0, "k", linewidth=1.0)
        axs[4].set_ylabel("Force (N)")
        axs[4].set_xlabel("Time (s)")
        axs[4].grid(True, linestyle=":", linewidth=0.5)

        fig.suptitle(f"Koopman Response Prediction - Dataset {idx + 1}")
        fig.tight_layout()

        if save:
            svg_path = os.path.join(out_dir, f"Koopman_Response_Dataset_{idx + 1}.svg")
            png_path = os.path.join(out_dir, f"Koopman_Response_Dataset_{idx + 1}.png")

            fig.savefig(
                svg_path,
                dpi=600,
                bbox_inches="tight",
                transparent=True,
            )

            fig.savefig(
                png_path,
                dpi=300,
                bbox_inches="tight",
            )

            print(f"Saved plot: {svg_path}")
            print(f"Saved plot: {png_path}")

    plt.show()
