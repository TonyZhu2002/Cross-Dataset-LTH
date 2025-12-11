"""
main.py

High-level experiment runner for CrossDS_LTH.

Pipeline (for LeNet5):
1) Prepare single-dataset tickets on MNIST and FashionMNIST (scripts/singlepruning.py)
2) Prepare cross-dataset ticket (scripts/doublepruning.py)
3) Train & evaluate universal tickets on CIFAR-10 (scripts/universal.py)
4) Generate all plots (scripts/result_visulization.py)
"""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd):
    """Run a subprocess command with logging and error checking."""
    print("\n[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


# ---------- Stage 1: single-dataset pruning (MNIST + FMNIST) ----------

def stage_single_tickets(args):
    """
    Run singlepruning.py on MNIST and FashionMNIST with LeNet5
    to produce A (MNIST ticket) and B (FMNIST ticket).

    Outputs:
        saves/lenet5/mnist/*.pth.tar
        saves/lenet5/fashionmnist/*.pth.tar
        dumps/lt/lenet5/mnist/*.dat, *_mask_*.pkl
        dumps/lt/lenet5/fashionmnist/*.dat, *_mask_*.pkl
    """
    for dataset in ["mnist", "fashionmnist"]:
        cmd = [
            sys.executable, "-m", "scripts.singlepruning",
            "--dataset", dataset,
            "--arch_type", "lenet5",
            "--prune_type", "lt",
            "--gpu", args.gpu,
        ]

        # Optional overrides: only add if provided on main.py CLI
        if args.single_lr is not None:
            cmd += ["--lr", str(args.single_lr)]
        if args.single_batch_size is not None:
            cmd += ["--batch_size", str(args.single_batch_size)]
        if args.single_start_iter is not None:
            cmd += ["--start_iter", str(args.single_start_iter)]
        if args.single_end_iter is not None:
            cmd += ["--end_iter", str(args.single_end_iter)]
        if args.single_prune_percent is not None:
            cmd += ["--prune_percent", str(args.single_prune_percent)]
        if args.single_prune_iterations is not None:
            cmd += ["--prune_iterations", str(args.single_prune_iterations)]
        if args.single_target_remaining is not None:
            cmd += ["--target_remaining", str(args.single_target_remaining)]

        run_cmd(cmd)


# ---------- Stage 2: cross-dataset pruning ----------

def stage_crossds(args):
    """
    Run doublepruning.py to obtain the cross-dataset ticket C.

    By default we start from FashionMNIST with LeNet5.
    All hyperparameters can be overridden via main.py flags.
    """
    cmd = [
        sys.executable, "-m", "scripts.doublepruning",
        "--dataset", "fashionmnist",
        "--arch_type", "lenet5",
        "--prune_type", "lt",
        "--gpu", args.gpu,
    ]

    # Optional overrides
    if args.cross_lr is not None:
        cmd += ["--lr", str(args.cross_lr)]
    if args.cross_batch_size is not None:
        cmd += ["--batch_size", str(args.cross_batch_size)]
    if args.cross_start_iter is not None:
        cmd += ["--start_iter", str(args.cross_start_iter)]
    if args.cross_end_iter is not None:
        cmd += ["--end_iter", str(args.cross_end_iter)]
    if args.cross_prune_percent is not None:
        cmd += ["--prune_percent", str(args.cross_prune_percent)]
    if args.cross_prune_iterations is not None:
        cmd += ["--prune_iterations", str(args.cross_prune_iterations)]
    if args.cross_target_remaining is not None:
        cmd += ["--target_remaining", str(args.cross_target_remaining)]
    if args.cross_pretrained_path is not None:
        cmd += ["--pretrained_path", args.cross_pretrained_path]

    run_cmd(cmd)


# ---------- Stage 3: universal evaluation on CIFAR-10 ----------

def stage_universal(args):
    """
    Run universal.py on CIFAR-10 for 5 ticket types:
        MNIST_ticket, FMNIST_ticket, crossds_ticket, dense_ticket, random_ticket

    universal.py will:
        - load / generate the corresponding mask
        - train for 10 epochs on grayscale CIFAR-10
        - save models to outputs_uni/models
        - save loss/acc curves to outputs_uni/plots
    """
    tickets = [
        "MNIST_ticket",
        "FMNIST_ticket",
        "crossds_ticket",
        "dense_ticket",
        "random_ticket",
    ]
    for name in tickets:
        cmd = [
            sys.executable, "-m", "scripts.universal",
            "--name", name,
            "--gpu", args.gpu,
        ]
        run_cmd(cmd)


# ---------- Stage 4: result visualization ----------

def stage_visualization(args):
    """
    Call result_visulization.py to generate all figures:

    - LeNet5 ticket pruning curves (MNIST / FashionMNIST / crossds)
    - CIFAR-10 epoch curves for 5 tickets
    - CIFAR-10 confusion matrices & per-class accuracy
    - LT vs reinit pruning curves for all arch/dataset combinations
    """
    cmd = [
        sys.executable, "-m", "scripts.result_visulization",
        "--task", "all",
    ]
    run_cmd(cmd)


# ---------- Main CLI ----------

def main():
    parser = argparse.ArgumentParser(description="CrossDS_LTH experiment runner")

    # Which stage to run
    parser.add_argument(
        "--stage",
        default="all",
        choices=["tickets", "crossds", "universal", "viz", "all"],
        help="Which part of the pipeline to run",
    )

    # Shared
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU id to use (passed to child scripts as --gpu)",
    )

    # Stage 1 (singlepruning) hyperparameters
    parser.add_argument("--single_lr", type=float, default=None,
                        help="Override learning rate for singlepruning")
    parser.add_argument("--single_batch_size", type=int, default=None,
                        help="Override batch size for singlepruning")
    parser.add_argument("--single_start_iter", type=int, default=None,
                        help="Override start_iter for singlepruning")
    parser.add_argument("--single_end_iter", type=int, default=None,
                        help="Override end_iter for singlepruning")
    parser.add_argument("--single_prune_percent", type=int, default=None,
                        help="Override prune_percent for singlepruning")
    parser.add_argument("--single_prune_iterations", type=int, default=None,
                        help="Override prune_iterations for singlepruning")
    parser.add_argument("--single_target_remaining", type=float, default=None,
                        help="Override target_remaining for singlepruning")

    # Stage 2 (doublepruning) hyperparameters
    parser.add_argument("--cross_lr", type=float, default=None,
                        help="Override learning rate for doublepruning")
    parser.add_argument("--cross_batch_size", type=int, default=None,
                        help="Override batch size for doublepruning")
    parser.add_argument("--cross_start_iter", type=int, default=None,
                        help="Override start_iter for doublepruning")
    parser.add_argument("--cross_end_iter", type=int, default=None,
                        help="Override end_iter for doublepruning")
    parser.add_argument("--cross_prune_percent", type=int, default=None,
                        help="Override prune_percent for doublepruning")
    parser.add_argument("--cross_prune_iterations", type=int, default=None,
                        help="Override prune_iterations for doublepruning")
    parser.add_argument("--cross_target_remaining", type=float, default=None,
                        help="Override target_remaining for doublepruning")
    parser.add_argument("--cross_pretrained_path", type=str, default=None,
                        help="Checkpoint path to start cross-dataset pruning from")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.stage in ["tickets", "all"]:
        print("\n===== Stage 1: single-dataset pruning (MNIST + FMNIST) =====")
        stage_single_tickets(args)

    if args.stage in ["crossds", "all"]:
        print("\n===== Stage 2: cross-dataset pruning =====")
        stage_crossds(args)

    if args.stage in ["universal", "all"]:
        print("\n===== Stage 3: universal evaluation on CIFAR-10 =====")
        stage_universal(args)

    if args.stage in ["viz", "all"]:
        print("\n===== Stage 4: result visualization =====")
        stage_visualization(args)


if __name__ == "__main__":
    main()
