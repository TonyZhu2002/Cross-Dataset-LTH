import argparse
import os
import time
import numpy as np
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 使用 MNIST 的 LeNet5（1通道、28×28）
from archs.mnist.LeNet5 import LeNet5


# =========================
# 基础工具函数
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


# =========================
# Mask 相关：两种加载途径
# =========================

def load_mask_from_pkl(mask_path: str, model: nn.Module, device: torch.device):
    print(f"[INFO] Loading mask from pkl: {mask_path}")

    # 读取 main.py 里 dump 出来的 mask（list[np.ndarray]）
    with open(mask_path, "rb") as f:
        mask_list = pickle.load(f)

    if not isinstance(mask_list, (list, tuple)):
        raise TypeError(f"Expected mask_list to be list/tuple, got {type(mask_list)}")

    masks = {}
    idx = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            if idx >= len(mask_list):
                raise ValueError(
                    f"Not enough masks in {mask_path}: "
                    f"needed more for parameter {name}"
                )
            m_np = mask_list[idx]
            idx += 1

            # 转成 tensor 并检查形状
            m = torch.from_numpy(m_np).to(device=device, dtype=torch.float32)
            if m.shape != param.shape:
                raise ValueError(
                    f"Mask shape {m.shape} for parameter {name} does not match "
                    f"param shape {param.shape}"
                )
            masks[name] = m

    if idx != len(mask_list):
        print(f"[WARN] Unused masks in {mask_path}: used {idx}, total {len(mask_list)}")

    return masks


def load_mask_from_ticket(ticket_path: str, model: nn.Module, device: torch.device):

    print(f"[INFO] Loading ticket from checkpoint: {ticket_path}")

    obj = torch.load(ticket_path, map_location="cpu", weights_only=False)

    if isinstance(obj, nn.Module):
        ticket_state = obj.state_dict()
    elif isinstance(obj, dict):
        ticket_state = obj["state_dict"] if "state_dict" in obj else obj
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    masks = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            if name in ticket_state and ticket_state[name].shape == param.shape:
                m = (ticket_state[name] != 0).float()
            else:
                # 形状不匹配，比如最后一层 → dense
                m = torch.ones_like(param)
            masks[name] = m.to(device)

    return masks


def create_random_mask(model: nn.Module, device: torch.device, keep_ratio=0.1):

    masks = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            rand = torch.rand_like(param, device=device)
            numel = param.numel()
            k = int(numel * keep_ratio)
            threshold = torch.topk(rand.view(-1), numel - k, largest=False).values.max()
            masks[name] = (rand >= threshold).float()
    return masks


def create_dense_mask(model: nn.Module, device: torch.device):

    masks = {
        name: torch.ones_like(param, device=device).float()
        for name, param in model.named_parameters()
        if "weight" in name
    }
    return masks


def apply_mask_to_weights(model: nn.Module, masks: dict):

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                param.mul_(masks[name])


def apply_mask_to_grads(model: nn.Module, masks: dict):

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and param.grad is not None:
                param.grad.mul_(masks[name])


# =========================
# Train + Test
# =========================

def train_one_epoch(model, loader, criterion, optimizer, device, masks=None):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()

        if masks is not None:
            apply_mask_to_grads(model, masks)

        optimizer.step()

        batch = imgs.size(0)
        total_loss += loss.item() * batch
        total_samples += batch

    return total_loss / total_samples


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            batch = imgs.size(0)
            total_loss += loss.item() * batch
            total_samples += batch

            pred = outputs.argmax(1)
            correct += (pred == labels).sum().item()

    return total_loss / total_samples, 100.0 * correct / total_samples


# =========================
# plots
# =========================

def plot_curves(train_losses, test_accuracies, name, out_dir="./outputs_uni/plots"):
    """
    画双 y 轴曲线：
      - 左 y 轴：Test Accuracy (%)
      - 右 y 轴：Train Loss（原始值）
      - x 轴：Epoch
    """
    ensure_dir(out_dir)
    epochs = np.arange(1, len(train_losses) + 1)

    losses = np.array(train_losses)
    acc = np.array(test_accuracies)

    fig, ax1 = plt.subplots()

    # 左 y 轴：Accuracy
    color_acc = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy (%)", color=color_acc)
    ax1.plot(
        epochs,
        acc,
        marker="o",
        linestyle="-",
        label="Test Acc (%)",
        color=color_acc,
    )
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(0, 100)

    # 右 y 轴：Loss（原始值）
    ax2 = ax1.twinx()
    color_loss = "tab:blue"
    ax2.set_ylabel("Train Loss", color=color_loss)
    ax2.plot(
        epochs,
        losses,
        marker="s",
        linestyle="--",
        label="Train Loss",
        color=color_loss,
    )
    ax2.tick_params(axis="y", labelcolor=color_loss)

    fig.suptitle(f"Loss & Accuracy vs Epoch ({name})")
    fig.tight_layout()

    # 合并两边的 legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    out_path = os.path.join(out_dir, f"plot_{name}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved plot to {out_path}")


# =========================
# main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        required=True,
        choices=[
            "MNIST_ticket",
            "FMNIST_ticket",
            "crossds_ticket",
            "dense_ticket",
            "random_ticket",
        ],
    )
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")

    # 固定超参
    lr = 1.2e-3
    batch_size = 60
    num_epochs = 10
    num_classes = 10  # CIFAR10 → 10 类

# load data
    transform = transforms.Compose(
        [
            transforms.Grayscale(),          # 3通道 → 1通道
            transforms.Resize((28, 28)),     # CIFAR 32 → 28
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        "../data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.CIFAR10(
        "../data", train=False, transform=transform, download=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Init model
    model = LeNet5(num_classes=num_classes).to(device)
    model.apply(weight_init)

    ticket = args.name

    # load mask
    if ticket in ["dense_ticket", "crossds_ticket"]:
        if ticket == "crossds_ticket":
            mask_path = "dumps_dp/lt/lenet5/fashionmnist/lt_mask_11.1.pkl"
            masks = load_mask_from_pkl(mask_path, model, device)
        else:  # dense_ticket
            masks = create_dense_mask(model, device)

    else:
        # MNIST_ticket / FMNIST_ticket / random_ticket
        if ticket == "MNIST_ticket":
            ckpt_path = "saves/lenet5/mnist/21_model_lt.pth.tar"
            masks = load_mask_from_ticket(ckpt_path, model, device)
        elif ticket == "FMNIST_ticket":
            ckpt_path = "saves/lenet5/fashionmnist/22_model_lt.pth.tar"
            masks = load_mask_from_ticket(ckpt_path, model, device)
        elif ticket == "random_ticket":
            masks = create_random_mask(model, device, keep_ratio=0.1)
        else:
            raise ValueError(f"Unsupported ticket name: {ticket}")

    apply_mask_to_weights(model, masks)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses = []
    test_accuracies = []

    print("\n[INFO] Training started...\n")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, masks
        )
        _, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_accuracies.append(test_acc)

        print(
            f"[{ticket}] Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

    end_time = time.time()
    print(f"\n[INFO] Training finished in {end_time - start_time:.2f} seconds.\n")

    # save models
    ensure_dir("./outputs_uni/models")
    torch.save(model, f"./outputs_uni/models/{ticket}.pth.tar")
    print(f"[INFO] Saved model: ./outputs_uni/models/{ticket}.pth.tar")

    # save plots
    plot_curves(train_losses, test_accuracies, ticket)


if __name__ == "__main__":
    main()
