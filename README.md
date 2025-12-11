
---

# **Cross Dataset LTH — A More Universal Ticket**

*A Lightweight Framework for Exploring Cross-Dataset Lottery Tickets*

---

## **Overview**

This repository contains an implementation of our study on cross-dataset pruning under the Lottery Ticket Hypothesis (LTH).
Unlike traditional LTH pipelines that prune a model on a **single dataset**, our work explores whether **sequential pruning across multiple datasets** can produce a *more universal sparse initialization*—one that transfers better to unseen downstream tasks.

We evaluate “universal tickets” obtained from MNIST, FashionMNIST, and a Cross-Dataset pruning process, and compare them against standard single-dataset tickets, dense baselines, and random sparse networks. Downstream performance is measured on grayscale-processed CIFAR-10.

This repository includes the complete experimental pipeline, pruning scripts, universal evaluation, and visualization utilities required to fully reproduce our results.

<p align="center">
  <img src="display.gif" alt="Pruning Visualization" width="650">
</p>

*A conceptual illustration of pruning progress and the emergence of sparse subnetworks.*
---

## **Environment Setup**

We recommend **Python 3.13** and a CUDA-enabled environment (CUDA 13.0).

### **1. Install PyTorch with CUDA 13.0**

```bash
pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 torchaudio==2.9.1+cu130 \
    --index-url https://download.pytorch.org/whl/cu130
```

### **2. Install remaining dependencies**

```bash
pip install -r requirements.txt
```

---

## **Repository Structure**

```
CrossDS_LTH/
│
├── archs/                     # Model architectures (MNIST, CIFAR10/100)
│   ├── mnist/
│   │   ├── LeNet5.py
│   │   ├── resnet.py
│   │   └── vgg.py
│   ├── cifar10/
│   └── cifar100/
│
├── scripts/
│   ├── utils.py               # Utility functions (paths, logging, pruning helpers)
│   ├── singlepruning.py       # Stage 1: Single-dataset iterative magnitude pruning
│   ├── doublepruning.py       # Stage 2: Cross-dataset iterative pruning
│   ├── universal.py           # Stage 3: Universal ticket evaluation on CIFAR-10
│   └── result_visulization.py # Stage 4: Full plotting suite (pruning curves, CM, etc.)
│
├── main.py                    # High-level pipeline runner
├── requirements.txt
└── README.md
```

---

## **Experimental Workflow**

The end-to-end experiment consists of **four sequential stages**:

1. **Single-Dataset Pruning**
   Generate MNIST (A) and FashionMNIST (B) winning tickets via IMP.

2. **Cross-Dataset Pruning**
   Sequentially prune a pre-trained ticket across datasets to obtain
   a *Cross-Dataset Ticket* (C).

3. **Universal Evaluation on CIFAR-10**
   Evaluate the transferability of:

   * MNIST ticket
   * FashionMNIST ticket
   * Cross-Dataset ticket
   * Dense model
   * Random sparse model

   All models are trained from scratch on grayscale CIFAR-10.

4. **Visualization & Analysis**
   Automatically produces:

   * Pruning curves
   * Epoch-wise accuracy/loss curves
   * Confusion matrices
   * Per-class accuracy
   * LT vs. Reinitialization comparisons

---

## **Reproducing the Experiments**

We provide **two recommended reproduction modes**.

---

### **A) One-Command Full Pipeline (Not Recommended for New Users)**

Runs all stages with default hyperparameters:

```bash
python main.py --stage all --gpu 0
```

**Warning:**
Since this executes all scripts sequentially, mismatched paths or accidental partial runs may cause errors.
For research use, we strongly recommend **Mode B** below.

---

### **B) Recommended Mode — Run Stage by Stage**

This mode gives you full control over hyperparameters and avoids path conflicts.

---

### **Stage 1 — Single-Dataset Pruning**

Generates MNIST and FashionMNIST tickets:

```bash
python main.py --stage tickets --gpu 0
```

Override hyperparameters (optional):

```bash
python main.py --stage tickets --gpu 0 \
    --single_lr 0.001 \
    --single_batch_size 128 \
    --single_end_iter 120 \
    --single_prune_percent 10 \
    --single_prune_iterations 35
```

---

### **Stage 2 — Cross-Dataset Pruning**

Explicitly specify checkpoint to begin from:

```bash
python main.py --stage crossds --gpu 0 \
    --cross_pretrained_path saves/lenet5/mnist/21_model_lt.pth.tar
```

Optional hyperparameters:

```bash
--cross_lr, --cross_batch_size, --cross_end_iter,
--cross_prune_percent, --cross_prune_iterations, etc.
```

---

### **Stage 3 — Universal Evaluation on CIFAR-10**

Runs evaluation for 5 ticket types:

```bash
python main.py --stage universal --gpu 0
```

Produces:

```
outputs_uni/models/*.pth.tar
outputs_uni/plots/*.png
```
> [!IMPORTANT]
> If this stage throws an error, it is very likely due to outdated or incorrect mask paths caused by moving directories or renaming files. Please go to:
> `scripts/universal.py` → `main()` function and manually update the following mask paths to match your environment:
> 
> - **MNIST ticket mask**
> - **FashionMNIST ticket mask**
> - **Cross-dataset ticket mask**

---

### **Stage 4 — Visualization Suite**

Generates all analysis figures:

```bash
python main.py --stage viz
```

Outputs saved to:

```
plots_universal/
```

---

## **Credits**

This implementation is developed and maintained by:

* **Mingyu (Tony) Zhu**
* **Danqi He**
* **Yuxuan Nan**
* **Guanhua Chen**

UCLA MEng in Data Science & Artificial Intelligence
Fall 2025 – CS260D Final Project