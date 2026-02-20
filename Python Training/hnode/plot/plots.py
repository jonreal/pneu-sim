# hnode/plot/plots.py
import os
import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from scipy.signal import correlate, butter, freqz

def compute_lag(x, pe):
    def _lag(a, b):
        a0 = a - np.mean(a)
        b0 = b - np.mean(b)

        corr = correlate(b0, a0, mode='full')
        lags = np.arange(-len(a0)+1, len(a0))

        return lags[np.argmax(corr)]

    lag_pe = _lag(x, pe)

    lag = int(np.round(lag_pe))

    return lag
# ------------------------ helpers ------------------------

def _simulate_dataset(model, x0, ts, u_fn, mf, me, dt0=1e-3, max_steps=100000):
    """Run the same diffrax solve your raw code uses for one dataset."""
    wrapped = lambda t, y, args: model(t, y, u_fn, mf, me)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(wrapped),
        diffrax.Tsit5(),
        ts[0], ts[-1],
        dt0=dt0,
        y0=x0,
        saveat=diffrax.SaveAt(ts=ts),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=max_steps,
    )
    return sol.ys  # shape (T, 4): [x, dx, Pf, Pe]

def mask_surface(Z, zmin=0, zmax=800000):
    """Same masking logic as your raw code."""
    Z = jnp.asarray(Z)
    return jnp.where((Z >= zmin) & (Z <= zmax), Z, jnp.nan)

def build_frequency_profile(ts):
    f = np.zeros_like(ts)

    t0 = 0
    t1 = 40     # 0.5 Hz
    t2 = 60     # 1 Hz
    t3 = 70     # 2 Hz

    f[(ts>=t0) & (ts<t1)] = 0.5
    f[(ts>=t1) & (ts<t2)] = 1.0
    f[(ts>=t2) & (ts<=t3)] = 2.0

    return f

# ------------------------ dataset plots (timeseries + hysteresis) ------------------------
def plot_datasets(model, data_list, which="all", *, save=True, out_dir="checkpoint",
                  dt0=1e-3, max_steps=100000):

    rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.8
    })

    if which == "all":
        indices = list(range(len(data_list)))
    else:
        indices = list(which)

    os.makedirs(out_dir, exist_ok=True)
    figs = []

    freqs = np.array([0.5, 1.0, 2.0])

    colors = ["#2166AC", "#4393C3", "#B2182B", "#D6604D"]
    labels = [r"$x$ (mm)", r"$\dot{x}$ (mm/s)", r"$P_f$ (kPa)", r"$P_e$ (kPa)"]

    for idx in indices:
        x0, ts, y_true, u_fn, mf, me, xeq, Force_array = data_list[idx]

        lag = compute_lag(
            y_true[:,0],   # x
            y_true[:,3]    # Pe
        )
        print(lag)

        lag1 = compute_lag(y_true[:39999, 0],       y_true[:39999, 3])
        lag2 = compute_lag(y_true[40000:59999, 0],  y_true[40000:59999, 3])
        lag3 = compute_lag(y_true[60000:, 0],       y_true[60000:, 3])

        fs = 1000.0
        lags = np.array([lag1, lag2, lag3])

        tau = lags / fs
        phase_meas = -2 * np.pi * freqs * tau
        phase_meas_deg = phase_meas * 180 / np.pi

        print(lag1, lag2, lag3)

        ys = _simulate_dataset(model, x0, ts, u_fn, mf, me, dt0=dt0, max_steps=max_steps)

        fig = plt.figure(figsize=(15,6))

        # main grid
        gs = GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.15)

        # left: 4x1
        gs_left = gs[0,0].subgridspec(5,1, hspace=0.35)
        axs_l = [fig.add_subplot(gs_left[i,0]) for i in range(5)]

        # right: 3x3 (equal height)
        gs_right = gs[0,1].subgridspec(3,3, hspace=0.65, wspace=0.35)

        # column 1
        ax_xF   = fig.add_subplot(gs_right[0,0])
        ax_pfF  = fig.add_subplot(gs_right[1,0])
        ax_peF  = fig.add_subplot(gs_right[2,0])

        # column 2
        ax_xF2  = fig.add_subplot(gs_right[0,1])
        ax_pfF2 = fig.add_subplot(gs_right[1,1])
        ax_peF2 = fig.add_subplot(gs_right[2,1])

        # column 3
        ax_xF3  = fig.add_subplot(gs_right[0,2])
        ax_pfF3 = fig.add_subplot(gs_right[1,2])
        ax_peF3 = fig.add_subplot(gs_right[2,2])

        for i in range(4):
            axs_l[i].plot(ts, y_true[:, i], "k--", linewidth=1.0, label="Truth")
            axs_l[i].plot(ts, ys[:, i], color=colors[i], linewidth=1.2, label="Model")

            axs_l[i].set_ylabel(labels[i], fontname="Times New Roman")
            axs_l[i].yaxis.set_label_coords(-0.05, 0.5)
            axs_l[i].grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
            axs_l[i].spines["top"].set_visible(False)
            axs_l[i].spines["right"].set_visible(False)
            axs_l[i].set_xlim([0, 70])

            if i == 0:
                axs_l[i].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.1), ncol=2)

        freq = build_frequency_profile(ts)

        axs_l[4].plot(ts, freq, "k--", lw=1.5)
        axs_l[4].set_ylabel("Freq (Hz)", fontname="Times New Roman")
        axs_l[4].yaxis.set_label_coords(-0.05, 0.5)
        axs_l[4].set_ylim([0, 2.2])
        axs_l[4].set_xlim([0, 70])
        axs_l[4].grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        axs_l[4].spines["top"].set_visible(False)
        axs_l[4].spines["right"].set_visible(False)

        axs_l[4].set_xlabel("Time (s)")

        # ---------- hysteresis ----------
        F = Force_array

        # x - F
        ax_xF.plot(y_true[:39999,0], F[:39999]/1000, "k--", linewidth=1.0)
        ax_xF.plot(ys[:39999,0], F[:39999]/1000, lw=1.0, color=colors[0])
        ax_xF.set_ylabel("Force (N)")
        ax_xF.set_ylim([-210, 210])

        # Pf - F
        ax_pfF.plot(y_true[lag1:lag1+39999, 2], F[:39999]/1000, "k--", linewidth=1.0)
        ax_pfF.plot(ys[:39999,2], F[:39999]/1000, lw=1.0, color=colors[2])
        ax_pfF.set_ylabel("Force (N)")
        ax_pfF.set_ylim([-210, 210])

        # Pe - F
        ax_peF.plot(y_true[lag1:lag1+39999, 3], F[:39999]/1000, "k--", linewidth=1.0)
        ax_peF.plot(ys[:39999,3], F[:39999]/1000, lw=1.0, color=colors[3])
        ax_peF.set_ylabel("Force (N)")
        ax_peF.set_ylim([-210, 210])

        # -------------------------------------------
        # x - F
        ax_xF2.plot(y_true[40000:59999,0], F[40000:59999]/1000, "k--", linewidth=1.0)
        ax_xF2.plot(ys[40000:59999,0], F[40000:59999]/1000, lw=1.0, color=colors[0])
        ax_xF2.set_xlabel("x (mm)")
        ax_xF2.set_ylim([-210, 210])

        # Pf - F
        ax_pfF2.plot(y_true[lag2+40000:lag2+59999, 2], F[40000:59999]/1000, "k--", linewidth=1.0)
        ax_pfF2.plot(ys[40000:59999,2], F[40000:59999]/1000, lw=1.0, color=colors[2])
        ax_pfF2.set_xlabel("Pf (kPa)")
        ax_pfF2.set_ylim([-210, 210])

        # Pe - F
        ax_peF2.plot(y_true[lag2+40000:lag2+59999, 3], F[40000:59999]/1000, "k--", linewidth=1.0)
        ax_peF2.plot(ys[40000:59999,3], F[40000:59999]/1000, lw=1.0, color=colors[3])
        ax_peF2.set_xlabel("Pe (kPa)")
        ax_peF2.set_ylim([-210, 210])

        # -------------------------------------------
        # x - F
        ax_xF3.plot(y_true[60000:,0], F[60000:]/1000, "k--", linewidth=1.0)
        ax_xF3.plot(ys[-10000:,0], F[-10000:]/1000, lw=1.0, color=colors[0])
        ax_xF3.set_ylim([-210, 210])

        # Pf - F
        ax_pfF3.plot(y_true[lag3+60000:lag3+69000, 2], F[60000:69000]/1000, "k--", linewidth=1.0)
        ax_pfF3.plot(ys[-10000:,2], F[-10000:]/1000, lw=1.0, color=colors[2])
        ax_pfF3.set_ylim([-210, 210])

        # Pe - F
        ax_peF3.plot(y_true[lag3+60000:lag3+69000, 3], F[60000:69000]/1000, "k--", linewidth=1.0)
        ax_peF3.plot(ys[-10000:,3], F[-10000:]/1000, lw=1.0, color=colors[3])
        ax_peF3.set_ylim([-210, 210])

        for ax in [ax_xF, ax_pfF, ax_peF,
                ax_xF2, ax_pfF2, ax_peF2,
                ax_xF3, ax_pfF3, ax_peF3]:

            for line in ax.lines:
                line.set_alpha(0.8)

        for ax in [ax_xF, ax_pfF, ax_peF, ax_xF2, ax_pfF2, ax_peF2, ax_xF3, ax_pfF3, ax_peF3]:
            ax.grid(True, ls=":", lw=0.5)
        
        top_axes = [ax_xF, ax_xF2, ax_xF3]
        txts   = ["0.5 Hz", "1 Hz", "2 Hz"]

        for ax, txt in zip(top_axes, txts):
            bbox = ax.get_position()
            xmid = (bbox.x0 + bbox.x1) / 2
            ytop = bbox.y1

            fig.text(xmid - 0.022, ytop - 0.035, txt,
                    ha="center", va="bottom", fontsize=12)

        if save:
            filename = os.path.join(out_dir, f"Responses Comparison.svg")
            fig.savefig(filename, dpi=600, bbox_inches="tight", transparent=True)
        figs.append(fig)
    plt.show()

    return figs
