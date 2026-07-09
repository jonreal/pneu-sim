# main_Kang.py
import os
import sys
import argparse
from pathlib import Path

# main_Kang.py is located at:
# pneu-sim/comparisions/Kang (analytical)/main_Kang.py
# parents[2] points back to pneu-sim/
ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hnode.data.loaders import generate_file_paths, load_training_data_from_file

from Kang.single_pam import fit_kang_single_pam
from Kang.evaluate import evaluate_kang, save_fit_summary
from Kang.utils import parse_ranges, code_from_file_path


def parse_args():
    p = argparse.ArgumentParser(description="Run Kang analytical baseline for PAM joint response prediction.")

    # --- Data selection ---
    p.add_argument("--data-dir", type=str, default="TRAIN DATA",
                   help="Folder containing nn_xxyy files for joint datasets")
    p.add_argument("--single-pam-dir", type=str, default="Single PAM data",
                   help="Folder containing single-PAM files like single00, single10, ...")
    p.add_argument("--prefix", type=str, default="nn_",
                   help="Joint dataset filename prefix")
    p.add_argument("--suffix", type=str, default="",
                   help="Joint dataset filename suffix")
    p.add_argument("--codes", nargs="*", default=None,
                   help="Specific joint dataset code tags, e.g. 1010 5050 8080")
    p.add_argument("--ranges", nargs="*", default=None,
                   help="Joint dataset ranges, e.g. 10-80")
    p.add_argument("--single-pam-ids", nargs="*", type=int, default=None,
                   help="Single-PAM file IDs, e.g. 0 10 20 30 40 50 60 70")

    # --- Kang fitting settings ---
    p.add_argument("--adam-steps", type=int, default=3000,
                   help="Adam iterations for single-PAM coefficient fitting")
    p.add_argument("--lbfgs-steps", type=int, default=1000,
                   help="LBFGS iterations for single-PAM coefficient fitting")
    p.add_argument("--D0-mm", type=float, default=10.0,
                   help="Initial PAM diameter used in Kang force model")
    p.add_argument("--L0-mm", type=float, default=200.0,
                   help="Initial PAM length used in Kang force model")
    p.add_argument("--alpha0-deg", type=float, default=27.5,
                   help="Initial braid angle used in Kang force model")

    # --- Joint dynamics settings ---
    p.add_argument("--m-eff", type=float, default=304.65,
                   help="Effective joint mass used by analytical simulation")
    p.add_argument("--b-eff", type=float, default=6500.0,
                   help="Effective damping used by analytical simulation")

    # --- Output ---
    p.add_argument("--output-dir", type=str, default="checkpoint_Kang",
                   help="Output folder for coefficients, CSV, and optional response data")
    p.add_argument("--save-response-data", action="store_true",
                   help="Save one NPZ response-data file per joint dataset")

    return p.parse_args()


def _build_file_paths(args):
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


def _resolve_output_dir(output_dir):
    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = SCRIPT_DIR / out_path
    return str(out_path)


def _load_joint_data_items(file_paths, prefix="nn_"):
    data_items = []

    for path in file_paths:
        code = code_from_file_path(path, prefix=prefix)
        x0, ts, y_true, u_fn, mf, me, xeq, Force_array = load_training_data_from_file(path)

        print(
            f"Loaded {path}: "
            f"code={code}, T={len(ts)}, "
            f"mf={float(mf):.8f}, me={float(me):.8f}",
            flush=True,
        )

        data_items.append((code, (x0, ts, y_true, u_fn, mf, me, xeq, Force_array)))

    return data_items


def main():
    args = parse_args()

    output_dir = _resolve_output_dir(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- Build joint dataset list ---
    file_paths = _build_file_paths(args)

    if not file_paths:
        raise SystemExit(
            f"No joint dataset files found in '{args.data_dir}'. "
            f"Check --data-dir / --codes / --ranges / --prefix / --suffix."
        )

    print("Joint datasets:")
    for i, path in enumerate(file_paths):
        print(f"  [{i}] {path}")

    data_items = _load_joint_data_items(file_paths, prefix=args.prefix)

    # --- Fit Kang coefficients from single-PAM data every run ---
    coeffs, fit_info = fit_kang_single_pam(
        single_pam_dir=args.single_pam_dir,
        ids=args.single_pam_ids,
        D0_mm=args.D0_mm,
        L0_mm=args.L0_mm,
        alpha0_deg=args.alpha0_deg,
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
    )

    save_fit_summary(fit_info, output_dir)

    # --- Run analytical joint simulation and evaluation ---
    evaluate_kang(
        coeffs=coeffs,
        data_items=data_items,
        out_dir=output_dir,
        save_response_data=bool(args.save_response_data),
        m_eff=args.m_eff,
        b_eff=args.b_eff,
    )

    print("Done.")


if __name__ == "__main__":
    main()
