# main.py
import os
import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hnode.data.loaders import generate_file_paths, load_training_data_from_file

from Koopman.model import (
    train_koopman,
    load_koopman_model,
    save_koopman_model,
)
from Koopman.evaluate import (
    evaluate_koopman,
    evaluate_node_style_loss_koopman,
)
from Koopman.plot import plot_koopman_datasets


def parse_args():
    p = argparse.ArgumentParser(description="Train Koopman/EDMDc model and make plots.")

    # --- Data selection ---
    p.add_argument("--data-dir", type=str, default=str(ROOT_DIR / "TRAIN DATA"),
                   help="Folder containing nn_xxyy files.")
    p.add_argument("--prefix", type=str, default="nn_",
                   help="Filename prefix (default nn_).")
    p.add_argument("--suffix", type=str, default="",
                   help="Filename suffix (default empty).")
    p.add_argument("--codes", "--train-codes", nargs="*", dest="codes", default=None,
                   help="Training code tags, e.g. 1010 4545 8080. If set, ranges is ignored.")
    p.add_argument("--ranges", "--train-ranges", nargs="*", dest="ranges", default=None,
                   help="Training ranges, e.g. 10-80. Ignored if codes are provided.")
    p.add_argument("--eval-codes", nargs="*", default=None,
                   help="Evaluation-only code tags, e.g. 3030 5050.")
    p.add_argument("--eval-ranges", nargs="*", default=None,
                   help="Evaluation-only ranges, e.g. 10-80.")

    # --- Koopman model settings ---
    p.add_argument("--lift-type", type=str, default="poly2", choices=["linear", "poly2", "poly3"],
                   help="Lifting dictionary type.")
    p.add_argument("--ridge", type=float, default=1e-3,
                   help="Ridge regularization. Recommended: 1e-3 or 1e-2.")
    p.add_argument("--force-scale", type=float, default=1.0,
                   help="Divisor for Force_array before normalization. Use 1000.0 to convert mN to N.")
    p.add_argument("--warmup-steps", type=int, default=1,
                   help="Number of measured initial samples used before recursive rollout.")
    p.add_argument("--clip-x-norm", type=float, default=8.0,
                   help="Clip normalized predicted state during rollout. Use 0 or negative to disable.")

    # --- Save / load ---
    p.add_argument("--save-checkpoint-dir", type=str, default=str(SCRIPT_DIR / "checkpoint_Koopman"),
                   help="Directory to save model, plots, response data, and CSV.")
    p.add_argument("--save-model-name", type=str, default="koopman_model.npz",
                   help="Saved model filename.")
    p.add_argument("--load-model-path", type=str,
                   default=str(SCRIPT_DIR / "checkpoint_Koopman" / "koopman_model.npz"),
                   help="Load existing Koopman model and skip training. Default points to the saved model in this folder.")
    p.add_argument("--train-new", action="store_true",
                   help="Ignore the default saved model and train a new Koopman model.")

    # --- Evaluation / plotting ---
    p.add_argument("--eval-r2", action="store_true",
                   help="Compute and save R2/RMSE CSV.")
    p.add_argument("--plot-datasets", nargs="*", default=["all"],
                   help='Dataset indices to plot, or "all". Example: 0 1 2 or all.')
    p.add_argument("--save-plots", action="store_true",
                   help="Save plots as SVG and PNG.")
    p.add_argument("--no-plot", action="store_true",
                   help="Disable plotting.")

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


def _build_file_paths(data_dir, prefix, suffix, codes=None, ranges_args=None, default_all=False):
    """Build file paths using code tags first, then pressure ranges."""
    if codes:
        return generate_file_paths(
            base_dir=data_dir,
            prefix=prefix,
            suffix=suffix,
            codes=codes,
            ranges=None,
        )

    if ranges_args:
        ranges = _parse_ranges(ranges_args)
        return generate_file_paths(
            base_dir=data_dir,
            prefix=prefix,
            suffix=suffix,
            codes=None,
            ranges=ranges,
        )

    if default_all:
        return generate_file_paths(
            base_dir=data_dir,
            prefix=prefix,
            suffix=suffix,
            codes=None,
            ranges=None,
        )

    return []


def main():
    args = parse_args()

    os.makedirs(args.save_checkpoint_dir, exist_ok=True)

    clip_x_norm = args.clip_x_norm
    if clip_x_norm <= 0:
        clip_x_norm = None

    # --- Build training file list: codes take precedence over ranges ---
    train_file_paths = _build_file_paths(
        data_dir=args.data_dir,
        prefix=args.prefix,
        suffix=args.suffix,
        codes=args.codes,
        ranges_args=args.ranges,
        default_all=True,
    )

    if not train_file_paths:
        raise SystemExit(
            f"No training dataset files found in '{args.data_dir}'. "
            f"Check --data-dir / --codes / --ranges / --prefix / --suffix."
        )

    print("\n=========== TRAINING DATASETS ===========")
    for i, path in enumerate(train_file_paths):
        print(f"  [{i}] {path}")

    # --- Load training datasets ---
    train_data_list = []
    for path in train_file_paths:
        x0, ts, y_true, u_fn, mf, me, xeq, Force_array = load_training_data_from_file(path)
        print(
            f"Loaded {path}: "
            f"T={len(ts)}, "
            f"mf={float(mf):.8f}, "
            f"me={float(me):.8f}"
        )
        train_data_list.append((x0, ts, y_true, u_fn, mf, me, xeq, Force_array))

    # --- Build optional evaluation file list ---
    eval_file_paths = _build_file_paths(
        data_dir=args.data_dir,
        prefix=args.prefix,
        suffix=args.suffix,
        codes=args.eval_codes,
        ranges_args=args.eval_ranges,
        default_all=False,
    )

    if eval_file_paths:
        print("\n=========== EVALUATION DATASETS ===========")
        for i, path in enumerate(eval_file_paths):
            print(f"  [{i}] {path}")

        eval_data_list = []
        for path in eval_file_paths:
            x0, ts, y_true, u_fn, mf, me, xeq, Force_array = load_training_data_from_file(path)
            print(
                f"Loaded {path}: "
                f"T={len(ts)}, "
                f"mf={float(mf):.8f}, "
                f"me={float(me):.8f}"
            )
            eval_data_list.append((x0, ts, y_true, u_fn, mf, me, xeq, Force_array))
    else:
        print("\nNo separate evaluation datasets provided. Evaluating on training datasets.")
        eval_data_list = train_data_list

    print("\n=========== DATA SUMMARY ===========")
    print(f"Training datasets   = {len(train_data_list)}")
    print(f"Evaluation datasets = {len(eval_data_list)}")

    # --- Train or load Koopman model ---
    load_model_path = None if args.train_new else args.load_model_path

    if load_model_path and os.path.exists(load_model_path):
        model_dict = load_koopman_model(load_model_path)
    else:
        if load_model_path and not os.path.exists(load_model_path):
            print(f"Saved Koopman model was not found at {load_model_path}. Training a new model instead.")

        model_dict = train_koopman(
            data_list=train_data_list,
            lift_type=args.lift_type,
            force_scale=args.force_scale,
            ridge=args.ridge,
        )

        save_model_path = os.path.join(args.save_checkpoint_dir, args.save_model_name)
        save_koopman_model(model_dict, save_model_path)

    # --- Rollout evaluation ---
    print("\n=========== ROLLOUT EVALUATION ===========")
    evaluate_node_style_loss_koopman(
        model_dict,
        eval_data_list,
        warmup_steps=args.warmup_steps,
        clip_x_norm=clip_x_norm,
    )

    # --- Save R-square / RMSE ---
    if args.eval_r2:
        evaluate_koopman(
            model_dict,
            eval_data_list,
            save_path=os.path.join(args.save_checkpoint_dir, "Koopman_R2_results_eval.csv"),
            warmup_steps=args.warmup_steps,
            clip_x_norm=clip_x_norm,
        )

    # --- Plot selected datasets ---
    if not args.no_plot:
        which = "all"
        if args.plot_datasets is not None:
            if len(args.plot_datasets) == 1 and args.plot_datasets[0].lower() == "all":
                which = "all"
            else:
                which = [int(x) for x in args.plot_datasets]

        plot_koopman_datasets(
            model_dict,
            eval_data_list,
            which=which,
            save=bool(args.save_plots),
            out_dir=args.save_checkpoint_dir,
            warmup_steps=args.warmup_steps,
            clip_x_norm=clip_x_norm,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
