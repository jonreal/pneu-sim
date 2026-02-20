# main.py
import os
import argparse
import jax.random as jr

# ---- Your package pieces (as we defined) ----
from hnode.data.loaders import generate_file_paths, load_training_data_from_file
from hnode.train.loop import TrainCfg, train_model, evaluate_r2
from hnode.plot.plots import plot_datasets


def parse_args():
    p = argparse.ArgumentParser(description="Train hnode model and make plots (faithful to raw code).")

    # --- Data selection ---
    p.add_argument("--data-dir", type=str, default="TRAIN DATA",
                   help="Folder containing nn_xxyy.txt files (e.g., TRAIN DATA)")
    p.add_argument("--prefix", type=str, default="nn_",
                   help="Filename prefix (default nn_)")
    p.add_argument("--suffix", type=str, default="",
                   help="Filename suffix (default "")")
    p.add_argument("--codes", nargs="*", default=None,
                   help="Specific code tags (e.g., 1010 5050 8080). If set, 'ranges' is ignored.")
    p.add_argument("--ranges", nargs="*", default=None,
                   help="Pairs like 10-10 50-50 80-80 10-80 80-10. Ignored if 'codes' provided.")

    # --- NN architecture ---
    p.add_argument("--width", type=int, default=96, help="NN hidden width")
    p.add_argument("--depth", type=int, default=2, help="NN hidden depth")

    # --- Training hyperparams (faithful) ---
    p.add_argument("--lr", type=float, default=1e-2, help="Learning rate (Adam)")
    p.add_argument("--epochs", type=int, default=10000, help="Max epochs")
    p.add_argument("--patience", type=int, default=500, help="Patience before LR reduce")
    p.add_argument("--max-patience", type=int, default=1000, help="Hard stop patience")
    p.add_argument("--reduce-factor", type=float, default=0.95, help="LR reduce factor")
    p.add_argument("--threshold-per-dataset", type=float, default=3.0,
                   help="Early-stop threshold per dataset (raw: 3.0)")

    # Resume / checkpoint flags (replace your previous ones)
    p.add_argument("--resume", action="store_true", help="Resume training from load-* files")

    # LOAD side
    p.add_argument("--load-checkpoint-dir", type=str, default="checkpoint")
    p.add_argument("--load-model-name", type=str, default="hybrid_model_puresine_16.eqx")
    p.add_argument("--load-opt-name", type=str, default="hybrid_opt_state_puresine_16.eqx")
    p.add_argument("--load-info-name", type=str, default="best_info_puresine_16.txt")

    # SAVE side
    p.add_argument("--save-checkpoint-dir", type=str, default="checkpoint")
    p.add_argument("--save-model-name", type=str, default="hybrid_model_puresine_new.eqx")
    p.add_argument("--save-opt-name", type=str, default="hybrid_opt_state_puresine_new.eqx")
    p.add_argument("--save-info-name", type=str, default="best_info_puresine_new.txt")

    # --- Datasets Plotting ---
    p.add_argument("--plot-datasets", nargs="*", default=["all"],
                   help='Indices to plot (0-based). Use "all" to plot all. Example: 0 2 4 or all')
    p.add_argument("--save-plots", action="store_true", help="Save dataset plots to checkpoint dir")

    # --- Evaluation ---
    p.add_argument("--eval-r2", action="store_true", help="Compute R^2 after training")

    return p.parse_args()


def _parse_ranges(ranges_args):
    """Parse CLI ranges like ['10-20','30-40'] -> [(10,20),(30,40)]."""
    out = []
    for token in ranges_args:
        try:
            a, b = token.split("-")
            out.append((int(a), int(b)))
        except Exception:
            raise SystemExit(f"Bad range format: '{token}'. Use like 10-20 30-40")
    return out


def main():
    args = parse_args()

    # --- Build file list (codes take precedence over ranges) ---
    if args.codes:
        file_paths = generate_file_paths(
            base_dir=args.data_dir,
            prefix=args.prefix,
            suffix=args.suffix,
            codes=args.codes,
            ranges=None,
        )
    elif args.ranges:
        ranges = _parse_ranges(args.ranges)
        file_paths = generate_file_paths(
            base_dir=args.data_dir,
            prefix=args.prefix,
            suffix=args.suffix,
            codes=None,
            ranges=ranges,
        )
    else:
        file_paths = generate_file_paths(
            base_dir=args.data_dir,
            prefix=args.prefix,
            suffix=args.suffix,
            codes=None,
            ranges=None,
        )

    if not file_paths:
        raise SystemExit(f"No dataset files found in '{args.data_dir}'. "
                         f"Check --data-dir / --codes / --ranges / --prefix / --suffix.")

    print("Datasets:")
    for i, p in enumerate(file_paths):
        print(f"  [{i}] {p}")

    # --- Load datasets (raw loader; poly44 is loaded inside the module as in your code) ---
    data_list = []
    for path in file_paths:
        x0, ts, y_true, u_fn, mf, me, xeq, Force_array = load_training_data_from_file(path)
        # NOTE: train loop expects tuples shaped like your raw loss function
        data_list.append((x0, ts, y_true, u_fn, mf, me, xeq, Force_array))

    # --- Training config faithful to your script ---
    cfg = TrainCfg(
        width=args.width,
        depth=args.depth,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        max_patience=args.max_patience,
        reduce_factor=args.reduce_factor,
        threshold_per_dataset=args.threshold_per_dataset,
        resume_training=args.resume,

        # separate LOAD
        load_checkpoint_dir=args.load_checkpoint_dir,
        load_model_name=args.load_model_name,
        load_opt_state_name=args.load_opt_name,
        load_info_name=args.load_info_name,

        # separate SAVE
        save_checkpoint_dir=args.save_checkpoint_dir,
        save_model_name=args.save_model_name,
        save_opt_state_name=args.save_opt_name,
        save_info_name=args.save_info_name,
    )

    # --- Train (resume or new, exactly as raw code logic) ---
    best_model, info = train_model(data_list, cfg=cfg, seed=42)
    print("Best:", info)

    # --- Save R-square
    if args.eval_r2:
        evaluate_r2(best_model, data_list)

    # --- Plot selected datasets (time series + hysteresis) ---
    which = "all"
    if args.plot_datasets is not None:
        if len(args.plot_datasets) == 1 and args.plot_datasets[0].lower() == "all":
            which = "all"
        else:
            which = [int(x) for x in args.plot_datasets]

    plot_datasets(
        best_model,
        data_list,
        which=which,
        save=bool(args.save_plots),
        out_dir=args.save_checkpoint_dir,
        dt0=1e-3,
        max_steps=100000,
    )

    print("Done.")

if __name__ == "__main__":
    main()
