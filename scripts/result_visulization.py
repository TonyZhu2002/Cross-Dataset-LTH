import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix


# =========================
# 全局配置
# =========================

ROOT_PLOTS = "plots_universal"
os.makedirs(ROOT_PLOTS, exist_ok=True)

# CIFAR-10 类别名（confusion matrix 用）
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 通用工具函数
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================
# 1) lenet5 Ticket Pruning 曲线（来自 plottt.py）
# ============================================================

def load_pruning_dat(base_dir, prune_type="lt"):
    """
    从 dumps 目录里读出压缩率(comp) 和 best accuracy 数据。
    对应 main.py 里：
        comp.dump(.../{prune_type}_compression.dat)
        bestacc.dump(.../{prune_type}_bestaccuracy.dat)

    同时：
      - 过滤掉 comp=0 或 acc=0 的点
      - 去掉“最后一个”数据点（避免极端点影响）
    """
    comp_path = os.path.join(base_dir, f"{prune_type}_compression.dat")
    acc_path = os.path.join(base_dir, f"{prune_type}_bestaccuracy.dat")

    if not (os.path.exists(comp_path) and os.path.exists(acc_path)):
        raise FileNotFoundError(
            f"Cannot find {comp_path} or {acc_path}. Check paths and prune_type."
        )

    comp = np.load(comp_path, allow_pickle=True)
    bestacc = np.load(acc_path, allow_pickle=True)

    valid_mask = (comp > 0) & (bestacc > 0)
    comp = comp[valid_mask]
    bestacc = bestacc[valid_mask]

    if len(comp) > 0:
        comp = comp[:-1]
        bestacc = bestacc[:-1]

    return comp, bestacc


def plot_single_pruning_curve(comp, bestacc, title, save_path, color="tab:blue"):
    """
    画一张 Test Accuracy vs Unpruned Weights Percentage 曲线
    - 只画线，不画点
    - y 轴固定 0~100
    - x 轴从 100% → 稀疏（反转）
    """
    plt.figure(figsize=(8, 5))

    plt.plot(comp, bestacc, linewidth=2, color=color)

    plt.xlabel("Unpruned Weights Percentage", fontsize=11)
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.title(title, fontsize=13)

    plt.gca().invert_xaxis()
    plt.ylim(0, 100)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.grid(True, axis="both", linestyle=":", linewidth=0.7, alpha=0.8)
    plt.xticks(comp, [f"{c:.1f}" for c in comp], rotation=60)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved pruning curve to {save_path}")


def plot_overall_pruning(comp_a, acc_a, comp_b, acc_b, comp_c, acc_c, save_path):
    """
    画三条 lenet5 ticket 的 pruning 曲线在一张图里
    """
    plt.figure(figsize=(9, 5))

    plt.plot(comp_a, acc_a, linewidth=2, color="tab:blue",   label="MNIST Ticket")
    plt.plot(comp_b, acc_b, linewidth=2, color="tab:orange", label="FashionMNIST Ticket")
    plt.plot(comp_c, acc_c, linewidth=2, color="tab:green",  label="Cross-DS Ticket")

    plt.xlabel("Unpruned Weights Percentage", fontsize=11)
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.title("Pruning Curves: Test Accuracy vs Unpruned Weights Percentage (LeNet5)", fontsize=13)

    plt.gca().invert_xaxis()
    plt.ylim(0, 100)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, axis="both", linestyle=":", linewidth=0.7, alpha=0.8)

    all_comp = np.unique(np.concatenate([comp_a, comp_b, comp_c]))
    plt.xticks(all_comp, [f"{c:.1f}" for c in all_comp], rotation=60)

    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved overall pruning comparison to {save_path}")


def task_lenet5_pruning():
    """
    生成 lenet5 下三种 ticket 的 pruning 曲线 + overall
    """
    out_dir = os.path.join(ROOT_PLOTS, "pruning_lenet5_tickets")
    ensure_dir(out_dir)

    base_mnist = os.path.join("dumps", "lt", "lenet5", "mnist")
    base_fmnist = os.path.join("dumps", "lt", "lenet5", "fashionmnist")
    base_cross = os.path.join("dumps_dp", "lt", "lenet5", "fashionmnist")

    comp_a, acc_a = load_pruning_dat(base_mnist, prune_type="lt")
    plot_single_pruning_curve(
        comp_a,
        acc_a,
        title="MNIST Ticket: Test Accuracy vs Unpruned Weights Percentage (LeNet5)",
        save_path=os.path.join(out_dir, "pruning_curve_mnist_ticket.png"),
        color="tab:blue",
    )

    comp_b, acc_b = load_pruning_dat(base_fmnist, prune_type="lt")
    plot_single_pruning_curve(
        comp_b,
        acc_b,
        title="FashionMNIST Ticket: Test Accuracy vs Unpruned Weights Percentage (LeNet5)",
        save_path=os.path.join(out_dir, "pruning_curve_fmnist_ticket.png"),
        color="tab:orange",
    )

    comp_c, acc_c = load_pruning_dat(base_cross, prune_type="lt")
    plot_single_pruning_curve(
        comp_c,
        acc_c,
        title="Cross-DS Ticket: Test Accuracy vs Unpruned Weights Percentage (LeNet5)",
        save_path=os.path.join(out_dir, "pruning_curve_crossds_ticket.png"),
        color="tab:green",
    )

    plot_overall_pruning(
        comp_a,
        acc_a,
        comp_b,
        acc_b,
        comp_c,
        acc_c,
        save_path=os.path.join(out_dir, "pruning_curve_overall_comparison.png"),
    )


# ============================================================
# 2) CIFAR-10 5 个 ticket 的 Epoch 曲线（来自 bigplot.py）
# ============================================================

LOGS = {
    "MNIST_ticket": {
        "acc": [26.51, 34.65, 37.94, 41.14, 42.82, 45.30, 47.87, 48.56, 49.41, 50.35],
        "loss": [2.1915, 1.9329, 1.7992, 1.7036, 1.6028, 1.5237, 1.4628, 1.4136, 1.3715, 1.3336],
    },
    "FMNIST_ticket": {
        "acc": [26.36, 36.40, 38.60, 40.40, 42.50, 45.23, 45.56, 47.57, 48.23, 49.93],
        "loss": [2.1815, 1.9331, 1.7952, 1.7112, 1.6345, 1.5627, 1.4940, 1.4361, 1.3875, 1.3424],
    },
    "crossds_ticket": {
        "acc": [45.25, 51.91, 55.41, 57.97, 59.02, 60.50, 61.42, 61.20, 61.27, 60.46],
        "loss": [1.7793, 1.4223, 1.2579, 1.1411, 1.0484, 0.9747, 0.9115, 0.8472, 0.7899, 0.7393],
    },
    "dense_ticket": {
        "acc": [56.60, 61.24, 62.06, 62.39, 62.66, 62.09, 60.89, 60.99, 60.82, 60.76],
        "loss": [1.5971, 1.1078, 0.8836, 0.7002, 0.5392, 0.3973, 0.3012, 0.2375, 0.1897, 0.1692],
    },
    "random_ticket": {
        "acc": [29.74, 36.83, 40.86, 43.44, 46.53, 47.58, 48.51, 50.63, 50.30, 51.42],
        "loss": [2.1725, 1.8626, 1.7260, 1.5973, 1.4976, 1.4320, 1.3822, 1.3427, 1.3002, 1.2663],
    },
}

LOG_COLORS = {
    "MNIST_ticket": "tab:blue",
    "FMNIST_ticket": "tab:orange",
    "crossds_ticket": "tab:green",
    "dense_ticket": "tab:red",
    "random_ticket": "tab:purple",
}


def plot_all_tickets_epoch(out_dir):
    """
    多 ticket 大图：5 个 ticket 的 acc/loss 全部画在一张图里
    """
    ensure_dir(out_dir)
    save_path = os.path.join(out_dir, "comparison_all_tickets_full10.png")

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

    for name, data in LOGS.items():
        epochs = np.arange(1, len(data["acc"]) + 1)
        c = LOG_COLORS[name]
        ax1.plot(epochs, data["acc"], color=c, linewidth=2)
        ax2.plot(epochs, data["loss"], color=c, linestyle="--", linewidth=1.8)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax2.set_ylabel("Train Loss", fontsize=12)
    ax1.set_ylim(20, 75)

    plt.title("CIFAR-10: Accuracy & Loss vs Epoch (Different Tickets)", fontsize=16)

    color_handles = [Line2D([0], [0], color=LOG_COLORS[n], lw=3) for n in LOGS.keys()]
    ax1.legend(
        handles=color_handles,
        labels=list(LOGS.keys()),
        title="Tickets (colors)",
        fontsize=10,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
    )

    style_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="Accuracy"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="Loss"),
    ]
    ax2.legend(
        handles=style_handles,
        title="Line Style",
        fontsize=10,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved {save_path}")


def plot_vertical_subplots_epoch(out_dir):
    """
    5 行竖排 subplot：每个 ticket 一行，acc/loss 双轴
    """
    ensure_dir(out_dir)
    save_path = os.path.join(out_dir, "comparison_vertical_subplots_full10.png")

    fig, axes = plt.subplots(5, 1, figsize=(10, 18), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    for idx, (name, data) in enumerate(LOGS.items()):
        ax1 = axes[idx]
        ax2 = ax1.twinx()
        epochs = np.arange(1, len(data["acc"]) + 1)
        c = LOG_COLORS[name]

        ax1.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

        ax1.plot(epochs, data["acc"], color=c, linewidth=2)
        ax2.plot(epochs, data["loss"], color=c, linestyle="--", linewidth=1.8)

        ax1.set_ylabel("Acc (%)")
        ax2.set_ylabel("Loss")
        ax1.set_ylim(20, 75)
        ax1.set_title(name, fontsize=12, loc="left")

    axes[-1].set_xlabel("Epoch")

    style_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="Accuracy"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="Loss"),
    ]
    fig.legend(
        handles=style_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.99),
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved {save_path}")


def task_epoch_curves():
    out_dir = os.path.join(ROOT_PLOTS, "epoch_curves")
    plot_all_tickets_epoch(out_dir)
    plot_vertical_subplots_epoch(out_dir)


# ============================================================
# 3) Confusion Matrix & Per-class Accuracy（来自 analyze_confusion.py）
# ============================================================

TICKETS = [
    "MNIST_ticket",
    "FMNIST_ticket",
    "crossds_ticket",
    "dense_ticket",
    "random_ticket",
]


def get_cifar10_test_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.CIFAR10("../data", train=False,
                                    download=True, transform=transform)
    loader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2)
    return loader


def eval_confusion(model_path, ticket_name, loader):
    print(f"[INFO] Evaluating {ticket_name} from {model_path}")
    model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.to(DEVICE)
    model.eval()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(10))
    acc = np.diag(cm) / cm.sum(axis=1).clip(min=1)

    overall_acc = (y_true == y_pred).mean() * 100.0
    print(f"[{ticket_name}] Overall test acc: {overall_acc:.2f}%")

    return cm, acc


def plot_confusion(cm, ticket_name, out_dir):
    ensure_dir(out_dir)
    plt.figure(figsize=(6, 5))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix ({ticket_name})")
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    plt.yticks(tick_marks, CLASS_NAMES, fontsize=8)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"cm_{ticket_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrix to {out_path}")


def plot_per_class_accuracy(acc_dict, out_path):
    ensure_dir(os.path.dirname(out_path) or ".")
    tickets = list(acc_dict.keys())
    n_class = len(CLASS_NAMES)

    x = np.arange(n_class)
    width = 0.15

    plt.figure(figsize=(10, 4))
    for i, t in enumerate(tickets):
        plt.bar(x + (i - len(tickets) / 2) * width + width / 2,
                acc_dict[t] * 100,
                width=width,
                label=t)

    plt.xticks(x, CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Per-class Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Per-class Accuracy Comparison")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved per-class accuracy plot to {out_path}")


def plot_confusion_diff(cm_ref, cm_other, name_ref, name_other, out_dir):
    ensure_dir(out_dir)
    cm_ref_norm = cm_ref / cm_ref.sum(axis=1, keepdims=True).clip(min=1)
    cm_other_norm = cm_other / cm_other.sum(axis=1, keepdims=True).clip(min=1)
    diff = cm_ref_norm - cm_other_norm

    plt.figure(figsize=(6, 5))
    vmax = np.max(np.abs(diff))
    plt.imshow(diff, interpolation="nearest", cmap="bwr",
               vmin=-vmax, vmax=vmax)
    plt.title(f"{name_ref} - {name_other} (row-normalized)")
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    plt.yticks(tick_marks, CLASS_NAMES, fontsize=8)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path = os.path.join(out_dir,
                            f"cm_diff_{name_ref}_minus_{name_other}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved diff matrix to {out_path}")


def task_confusion():
    """
    生成：
      - 5 张 confusion matrix
      - 一张 per-class accuracy 柱状图
      - crossds_ticket 与其他 ticket 的差分 heatmap
    """
    loader = get_cifar10_test_loader()
    cm_dict = {}
    acc_dict = {}

    base_dir = os.path.join(ROOT_PLOTS, "confusion")
    matrices_dir = os.path.join(base_dir, "matrices")
    diff_dir = os.path.join(base_dir, "diff")
    per_class_path = os.path.join(base_dir, "per_class_accuracy.png")

    for ticket in TICKETS:
        model_path = os.path.join("outputs_uni", "models", f"{ticket}.pth.tar")
        cm, acc = eval_confusion(model_path, ticket, loader)
        cm_dict[ticket] = cm
        acc_dict[ticket] = acc
        plot_confusion(cm, ticket, matrices_dir)

    plot_per_class_accuracy(acc_dict, out_path=per_class_path)

    ref = "crossds_ticket"
    for other in TICKETS:
        if other == ref:
            continue
        plot_confusion_diff(cm_dict[ref], cm_dict[other], ref, other, diff_dir)


# ============================================================
# 4) 所有架构 & 数据集的 LT vs REINIT（来自 combine_plots.py）
# ============================================================

def task_lt_combined():
    """
    对 fc1/lenet5/resnet18 × mnist/fashionmnist/cifar10/cifar100
    画 Winning tickets vs Random reinit 的 Pruning 曲线。
    """
    DPI = 1200
    prune_iterations = 35
    arch_types = ["fc1", "lenet5", "resnet18"]
    datasets = ["mnist", "fashionmnist", "cifar10", "cifar100"]

    out_dir = os.path.join(ROOT_PLOTS, "pruning_all_arch")
    ensure_dir(out_dir)

    for arch_type in arch_types:
        for dataset in datasets:
            base = os.path.join("dumps", "lt", arch_type, dataset)
            comp_path = os.path.join(base, "lt_compression.dat")
            lt_acc_path = os.path.join(base, "lt_bestaccuracy.dat")
            reinit_acc_path = os.path.join(base, "reinit_bestaccuracy.dat")

            if not (os.path.exists(comp_path) and
                    os.path.exists(lt_acc_path) and
                    os.path.exists(reinit_acc_path)):
                print(f"[WARN] Missing data for {arch_type} | {dataset}, skip.")
                continue

            d = np.load(comp_path, allow_pickle=True)
            b = np.load(lt_acc_path, allow_pickle=True)
            c = np.load(reinit_acc_path, allow_pickle=True)

            a = np.arange(prune_iterations)
            plt.figure(figsize=(8, 5))
            plt.plot(a, b, c="blue", label="Winning tickets")
            plt.plot(a, c, c="red", label="Random reinit")

            plt.title(f"Test Accuracy vs Weights % ({arch_type} | {dataset})")
            plt.xlabel("Weights %")
            plt.ylabel("Test accuracy")
            plt.xticks(a, d, rotation="vertical")
            plt.ylim(0, 100)
            plt.legend()
            plt.grid(color="gray")

            out_path = os.path.join(out_dir, f"combined_{arch_type}_{dataset}.png")
            plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
            plt.close()
            print(f"[INFO] Saved {out_path}")


# =========================
# main：任务路由
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["lenet5_pruning", "epoch_curves", "confusion", "lt_combined", "all"],
        help="Which plotting task to run",
    )
    args = parser.parse_args()

    if args.task in ["lenet5_pruning", "all"]:
        print("[TASK] lenet5_pruning")
        task_lenet5_pruning()

    if args.task in ["epoch_curves", "all"]:
        print("[TASK] epoch_curves")
        task_epoch_curves()

    if args.task in ["confusion", "all"]:
        print("[TASK] confusion")
        task_confusion()

    if args.task in ["lt_combined", "all"]:
        print("[TASK] lt_combined")
        task_lt_combined()


if __name__ == "__main__":
    main()
