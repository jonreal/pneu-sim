import os
import sys
import argparse
from pathlib import Path

# main_RNN.py is inside:
# pneu-sim/comparisions/RNN/main_RNN.py
# parents[2] points back to pneu-sim/, so Python can import the top-level hnode package.
ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hnode.data.loaders import generate_file_paths, load_training_data_from_file

from RNN.utils import parse_ranges
from RNN.train import train_gru
from RNN.model import load_gru_checkpoint
from RNN.evaluate import evaluate_gru, evaluate_node_style_loss_gru
from RNN.plot import plot_gru_datasets


def parse_args():
    p = argparse.ArgumentParser(description="Train GRU/RNN baseline for PAM joint response prediction.")

    # --- Data selection ---
    p.add_argument(
        "--data-dir",
        type=str,
        default=str(ROOT_DIR / "TRAIN DATA"),
        help="Folder containing nn_xxyy files.",
    )
    p.add_argument("--prefix", type=str, default="nn_", help="Filename prefix.")
    p.add_argument("--suffix", type=str, default="", help="Filename suffix.")
    p.add_argument(
        "--codes",
        nargs="*",
        default=None,
        help="Specific code tags, e.g. 1010 5050 8080. If set, ranges are ignored.",
    )
    p.add_argument(
        "--ranges",
        nargs="*",
        default=None,
        help="Pairs like 10-10 50-50 80-80 10-80 80-10. Ignored if codes are provided.",
    )

    # --- GRU architecture ---
    p.add_argument("--seq-len", type=int, default=100, help="Input history length.")
    p.add_argument("--pred-horizon", type=int, default=100, help="Recursive multi-step training horizon.")
    p.add_argument("--hidden-dim", type=int, default=96, help="GRU hidden dimension.")
    p.add_argument("--num-layers", type=int, default=2, help="Number of GRU layers.")
    p.add_argument("--dropout", type=float, default=0.0, help="Dropout between GRU layers.")

    # --- Training ---
    p.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    p.add_argument("--epochs", type=int, default=1000, help="Maximum training epochs.")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    p.add_argument("--w-pos", type=float, default=100.0, help="Loss weight for x and dx.")
    p.add_argument("--w-pressure", type=float, default=1.0, help="Loss weight for Pf and Pe.")
    p.add_argument("--use-mass", action="store_true", help="Include mf and me as constant inputs.")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Use "auto", "cpu", or "cuda". Default: auto.',
    )

    # --- Resume / load ---
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from load-* checkpoint.",
    )
    p.add_argument(
        "--load-checkpoint-dir",
        type=str,
        default=str(SCRIPT_DIR / "checkpoint_RNN"),
        help="Directory containing checkpoint used for resume.",
    )
    p.add_argument(
        "--load-model-name",
        type=str,
        default="model_1.pt",
        help="Base model name for resume. Actual checkpoint is *_checkpoint.pt.",
    )
    p.add_argument(
        "--load-model-path",
        type=str,
        default=str(SCRIPT_DIR / "checkpoint_RNN" / "model_1_checkpoint.pt"),
        help="Load an existing GRU checkpoint and skip training. Default points to the saved checkpoint in this folder.",
    )
    p.add_argument(
        "--train-new",
        action="store_true",
        help="Ignore the default saved checkpoint and train a new GRU model.",
    )

    # --- Save ---
    p.add_argument(
        "--save-checkpoint-dir",
        type=str,
        default=str(SCRIPT_DIR / "checkpoint_RNN"),
        help="Directory to save model, plots, response data, and CSV.",
    )
    p.add_argument(
        "--save-model-name",
        type=str,
        default="model_1.pt",
        help="Base saved model name. Actual checkpoint is *_checkpoint.pt.",
    )

    # --- Evaluation / plotting ---
    p.add_argument("--eval-r2", action="store_true", help="Save R2/RMSE CSV.")
    p.add_argument(
        "--plot-datasets",
        nargs="*",
        default=["all"],
        help='Dataset indices to plot, or "all". Example: --plot-datasets 0 1 2',
    )
    p.add_argument("--save-plots", action="store_true", help="Save plots as SVG and PNG.")
    p.add_argument("--no-plot", action="store_true", help="Disable plotting.")

    return p.parse_args()


def build_file_paths_from_args(args):
    # codes take precedence over ranges, matching the HNODE main.py style
    if args.codes:
        file_paths = generate_file_paths(
            base_dir=args.data_dir,
            prefix=args.prefix,
            suffix=args.suffix,
            codes=args.codes,
            ranges=None,
        )
    elif args.ranges:
        ranges = parse_ranges(args.ranges)
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

    return file_paths


def load_data_list(file_paths):
    data_list = []

    for path in file_paths:
        x0, ts, y_true, u_fn, mf, me, xeq, Force_array = load_training_data_from_file(path)

        print(
            f"Loaded {path}: "
            f"T={len(ts)}, "
            f"mf={float(mf):.8f}, "
            f"me={float(me):.8f}"
        )

        data_list.append((x0, ts, y_true, u_fn, mf, me, xeq, Force_array))

    return data_list


def main():
    args = parse_args()

    os.makedirs(args.save_checkpoint_dir, exist_ok=True)

    # --- Build file list ---
    file_paths = build_file_paths_from_args(args)

    if not file_paths:
        raise SystemExit(
            f"No dataset files found in '{args.data_dir}'. "
            f"Check --data-dir / --codes / --ranges / --prefix / --suffix."
        )

    print("Datasets:")
    for i, path in enumerate(file_paths):
        print(f"  [{i}] {path}")

    # --- Load datasets ---
    data_list = load_data_list(file_paths)

    # --- Train, resume, or load existing checkpoint ---
    load_model_path = None if (args.train_new or args.resume) else args.load_model_path

    if load_model_path and os.path.exists(load_model_path):
        model, input_scaler, output_scaler, info = load_gru_checkpoint(
            load_model_path,
            device=args.device,
        )
        model_use_mass = bool(info["use_mass"])
        model_seq_len = int(info["seq_len"])

    else:
        if load_model_path and not os.path.exists(load_model_path):
            print(f"Saved GRU checkpoint was not found at {load_model_path}. Training a new model instead.")

        model, input_scaler, output_scaler, info = train_gru(
            data_list=data_list,
            seq_len=args.seq_len,
            pred_horizon=args.pred_horizon,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            use_mass=args.use_mass,
            w_pos=args.w_pos,
            w_pressure=args.w_pressure,
            load_checkpoint_dir=args.load_checkpoint_dir,
            load_model_name=args.load_model_name,
            save_checkpoint_dir=args.save_checkpoint_dir,
            save_model_name=args.save_model_name,
            resume=args.resume,
            device=args.device,
        )
        model_use_mass = bool(args.use_mass)
        model_seq_len = int(args.seq_len)

    print("Best GRU:", info)

    # --- Rollout loss evaluation ---
    evaluate_node_style_loss_gru(
        model,
        data_list,
        input_scaler,
        output_scaler,
        seq_len=model_seq_len,
        use_mass=model_use_mass,
        device=args.device,
    )

    # --- R2 / RMSE evaluation ---
    if args.eval_r2:
        evaluate_gru(
            model,
            data_list,
            input_scaler,
            output_scaler,
            seq_len=model_seq_len,
            use_mass=model_use_mass,
            save_path=os.path.join(args.save_checkpoint_dir, "GRU_R2_results.csv"),
            device=args.device,
        )

    # --- Plot selected datasets ---
    if not args.no_plot:
        if len(args.plot_datasets) == 1 and args.plot_datasets[0].lower() == "all":
            which = "all"
        else:
            which = [int(x) for x in args.plot_datasets]

        plot_gru_datasets(
            model,
            data_list,
            input_scaler,
            output_scaler,
            seq_len=model_seq_len,
            which=which,
            use_mass=model_use_mass,
            save=bool(args.save_plots),
            out_dir=args.save_checkpoint_dir,
            device=args.device,
        )

    print("Done.")


if __name__ == "__main__":
    main()
