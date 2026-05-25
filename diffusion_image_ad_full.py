# -*- coding: utf-8 -*-


import os
import math
import json
import time
import random
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms, utils


# =====================================================
# 1. 配置
# =====================================================

@dataclass
class Config:
    # 基础
    seed: int = 42
    out_dir: str = "outputs_single"
    data_root: str = "./data"

    # 数据
    image_size: int = 32
    normal_class: int = 0
    train_limit: int = 8000
    test_normal_limit: int = 500
    test_anomaly_limit: int = 500

    # 扩散过程
    T: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    repair_t: int = 120
    stochastic_reverse: bool = False

    # 训练
    batch_size: int = 128
    epochs: int = 8
    lr: float = 2e-4
    num_workers: int = 2

    # 异常生成
    anomaly_type: str = "mixed"   # rect / scratch / noise / mixed
    anomaly_prob_train: float = 0.8
    min_rect: int = 4
    max_rect: int = 12
    scratch_prob: float = 0.5

    # mask 条件
    eval_mask_mode: str = "all"   # all / gt / none
    mask_loss_weight: float = 4.0

    # baseline
    use_blur_baseline: bool = True
    blur_kernel_size: int = 5

    # 可视化
    save_n: int = 16

    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


FASHION_CLASSES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


# =====================================================
# 2. 工具函数
# =====================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_to_model(x):
    """
    [0,1] -> [-1,1]
    """
    return x * 2.0 - 1.0


def denormalize_to_image(x):
    """
    [-1,1] -> [0,1]
    """
    return (x.clamp(-1, 1) + 1.0) / 2.0


def save_image_grid(tensor, path, nrow=4):
    ensure_dir(os.path.dirname(path))
    tensor = denormalize_to_image(tensor.detach().cpu())
    utils.save_image(tensor, path, nrow=nrow)


def save_mask_grid(mask_tensor, path, nrow=4):
    """
    mask: 0/1 -> image 0/1
    """
    ensure_dir(os.path.dirname(path))
    utils.save_image(mask_tensor.detach().cpu().clamp(0, 1), path, nrow=nrow)


def save_heatmap_grid(amap_tensor, path, nrow=4):
    """
    保存灰度异常热力图。
    amap_tensor: [B,1,H,W], non-negative
    """
    ensure_dir(os.path.dirname(path))
    amap = amap_tensor.detach().cpu()
    amap = amap / (amap.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
    utils.save_image(amap, path, nrow=nrow)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =====================================================
# 3. 数据与异常生成
# =====================================================

def make_random_anomaly(img, cfg: Config):
    """
    生成不同类型的人工异常。

    img:
        [1,H,W], value in [-1,1]

    return:
        corrupted: [1,H,W]
        mask:      [1,H,W], 1 表示异常区域
    """
    _, H, W = img.shape
    corrupted = img.clone()
    mask = torch.zeros_like(img)

    anomaly_type = cfg.anomaly_type
    if anomaly_type == "mixed":
        anomaly_type = random.choice(["rect", "scratch", "noise"])

    if anomaly_type == "rect":
        rect_h = random.randint(cfg.min_rect, cfg.max_rect)
        rect_w = random.randint(cfg.min_rect, cfg.max_rect)
        y0 = random.randint(0, H - rect_h)
        x0 = random.randint(0, W - rect_w)

        patch_value = random.choice([-1.0, 1.0]) * random.uniform(0.4, 1.0)
        corrupted[:, y0:y0 + rect_h, x0:x0 + rect_w] = patch_value
        mask[:, y0:y0 + rect_h, x0:x0 + rect_w] = 1.0

    elif anomaly_type == "scratch":
        y = random.randint(0, H - 1)
        thickness = random.randint(1, 3)
        x_start = random.randint(0, W // 3)
        x_end = random.randint(W // 2, W - 1)

        value = random.choice([-1.0, 1.0]) * random.uniform(0.5, 1.0)
        y1 = max(0, y - thickness)
        y2 = min(H, y + thickness + 1)

        corrupted[:, y1:y2, x_start:x_end] = value
        mask[:, y1:y2, x_start:x_end] = 1.0

    elif anomaly_type == "noise":
        rect_h = random.randint(cfg.min_rect, cfg.max_rect)
        rect_w = random.randint(cfg.min_rect, cfg.max_rect)
        y0 = random.randint(0, H - rect_h)
        x0 = random.randint(0, W - rect_w)

        noise_patch = torch.rand((1, rect_h, rect_w), dtype=img.dtype) * 2.0 - 1.0
        corrupted[:, y0:y0 + rect_h, x0:x0 + rect_w] = noise_patch
        mask[:, y0:y0 + rect_h, x0:x0 + rect_w] = 1.0

    else:
        raise ValueError(f"Unknown anomaly_type: {cfg.anomaly_type}")

    return corrupted, mask


class NormalTrainDataset(Dataset):
    """
    只使用正常类别训练。
    训练时动态制造伪异常，作为条件图像 cond。
    clean 是原始正常图像。
    """

    def __init__(self, base_dataset, cfg: Config):
        self.base = base_dataset
        self.cfg = cfg

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = normalize_to_model(img)

        if random.random() < self.cfg.anomaly_prob_train:
            cond, mask = make_random_anomaly(img, self.cfg)
        else:
            cond = img.clone()
            mask = torch.zeros_like(img)

        return {
            "clean": img,
            "cond": cond,
            "mask": mask,
            "label": torch.tensor(0, dtype=torch.long),
        }


class AnomalyTestDataset(Dataset):
    """
    测试集：
    - 前 n_normal 张：正常样本；
    - 后 n_anomaly 张：正常图像加人工异常，构造异常样本。
    """

    def __init__(self, normal_imgs, anomaly_imgs, cfg: Config):
        self.normal_imgs = normal_imgs
        self.anomaly_imgs = anomaly_imgs
        self.cfg = cfg
        self.n_normal = len(normal_imgs)
        self.n_anom = len(anomaly_imgs)

    def __len__(self):
        return self.n_normal + self.n_anom

    def __getitem__(self, idx):
        if idx < self.n_normal:
            img, _ = self.normal_imgs[idx]
            img = normalize_to_model(img)
            cond = img.clone()
            mask = torch.zeros_like(img)
            y = 0
        else:
            img, _ = self.anomaly_imgs[idx - self.n_normal]
            img = normalize_to_model(img)
            cond, mask = make_random_anomaly(img, self.cfg)
            y = 1

        return {
            "clean": img,
            "cond": cond,
            "mask": mask,
            "label": torch.tensor(y, dtype=torch.long),
        }


def build_dataloaders(cfg: Config):
    tfm = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])

    train_all = datasets.FashionMNIST(
        root=cfg.data_root,
        train=True,
        download=True,
        transform=tfm,
    )

    test_all = datasets.FashionMNIST(
        root=cfg.data_root,
        train=False,
        download=True,
        transform=tfm,
    )

    train_indices = [i for i, (_, y) in enumerate(train_all) if y == cfg.normal_class]
    test_indices = [i for i, (_, y) in enumerate(test_all) if y == cfg.normal_class]

    if cfg.train_limit > 0:
        train_indices = train_indices[:cfg.train_limit]

    available = len(test_indices)

    n_normal = min(cfg.test_normal_limit, available // 2)
    n_anomaly = min(cfg.test_anomaly_limit, available - n_normal)

    normal_test_indices = test_indices[:n_normal]
    anomaly_test_indices = test_indices[n_normal:n_normal + n_anomaly]

    print(f"[Data] normal class: {cfg.normal_class} ({FASHION_CLASSES.get(cfg.normal_class, 'unknown')})")
    print(f"[Data] train normal samples: {len(train_indices)}")
    print(f"[Data] available test samples: {available}")
    print(f"[Data] normal test samples: {len(normal_test_indices)}")
    print(f"[Data] anomaly test samples: {len(anomaly_test_indices)}")

    if len(normal_test_indices) == 0 or len(anomaly_test_indices) == 0:
        raise ValueError(
            "测试集中正常样本或异常样本为空。请减小 test_normal_limit / test_anomaly_limit。"
        )

    train_subset = Subset(train_all, train_indices)
    normal_test_subset = Subset(test_all, normal_test_indices)
    anomaly_test_subset = Subset(test_all, anomaly_test_indices)

    train_ds = NormalTrainDataset(train_subset, cfg)
    test_ds = AnomalyTestDataset(normal_test_subset, anomaly_test_subset, cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    return train_loader, test_loader


# =====================================================
# 4. 扩散过程
# =====================================================

class DiffusionSchedule:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.betas = torch.linspace(
            cfg.beta_start,
            cfg.beta_end,
            cfg.T,
            device=cfg.device,
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        """
        正向扩散闭式采样：
        x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) epsilon
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1, 1)
        sqrt_omab = torch.sqrt(1.0 - self.alpha_bars[t]).view(-1, 1, 1, 1)

        xt = sqrt_ab * x0 + sqrt_omab * noise
        return xt, noise


# =====================================================
# 5. 模型：轻量 UNet
# =====================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        device = t.device

        emb = torch.exp(
            torch.arange(half, device=device) * -(math.log(10000) / (half - 1))
        )
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        time_bias = self.time_proj(t_emb).view(t_emb.shape[0], -1, 1, 1)
        h = h + time_bias

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.skip(x)


class SmallUNet(nn.Module):
    """
    输入通道：
        xt:   1
        cond: 1
        mask: 1
        total = 3

    输出：
        predicted noise: 1
    """

    def __init__(self, in_ch=3, base_ch=64, time_dim=128):
        super().__init__()

        self.time_embed = TimeEmbedding(time_dim)

        self.enc1 = ResBlock(in_ch, base_ch, time_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)

        self.enc2 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)

        self.mid = ResBlock(base_ch * 2, base_ch * 4, time_dim)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec2 = ResBlock(base_ch * 4, base_ch * 2, time_dim)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec1 = ResBlock(base_ch * 2, base_ch, time_dim)

        self.out = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, xt, cond, mask, t):
        t_emb = self.time_embed(t)

        x = torch.cat([xt, cond, mask], dim=1)

        e1 = self.enc1(x, t_emb)
        d1 = self.down1(e1)

        e2 = self.enc2(d1, t_emb)
        d2 = self.down2(e2)

        m = self.mid(d2, t_emb)

        u2 = self.up2(m)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2(u2, t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1(u1, t_emb)

        return self.out(u1)


# =====================================================
# 6. 训练
# =====================================================

def train_model(cfg: Config, model, schedule, train_loader):
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=1e-4,
    )

    losses = []

    for epoch in range(1, cfg.epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")

        for batch in pbar:
            clean = batch["clean"].to(cfg.device)
            cond = batch["cond"].to(cfg.device)
            mask = batch["mask"].to(cfg.device)

            B = clean.shape[0]

            t = torch.randint(0, cfg.T, (B,), device=cfg.device)
            noise = torch.randn_like(clean)

            xt, true_noise = schedule.q_sample(clean, t, noise)
            pred_noise = model(xt, cond, mask, t)

            weight = 1.0 + cfg.mask_loss_weight * mask
            loss = ((pred_noise - true_noise) ** 2 * weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return losses


def plot_loss(losses, cfg: Config):
    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Diffusion Repair Training Loss")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "training_loss.png"), dpi=200)
    plt.close()


# =====================================================
# 7. 反向修正
# =====================================================

@torch.no_grad()
def p_sample(model, schedule, xt, cond, mask, t_scalar, cfg: Config):
    B = xt.shape[0]
    t = torch.full((B,), t_scalar, device=cfg.device, dtype=torch.long)

    beta_t = schedule.betas[t_scalar]
    alpha_t = schedule.alphas[t_scalar]
    alpha_bar_t = schedule.alpha_bars[t_scalar]

    eps_pred = model(xt, cond, mask, t)

    mean = (1.0 / torch.sqrt(alpha_t)) * (
        xt - beta_t / torch.sqrt(1.0 - alpha_bar_t) * eps_pred
    )

    if t_scalar > 0 and cfg.stochastic_reverse:
        z = torch.randn_like(xt)
        sigma_t = torch.sqrt(beta_t)
        return mean + sigma_t * z

    return mean


@torch.no_grad()
def repair_images(model, schedule, cond, mask, cfg: Config):
    """
    异常修正：
        cond -> q_sample 到 repair_t -> reverse denoise -> repaired
    """
    model.eval()

    B = cond.shape[0]
    repair_t = min(cfg.repair_t, cfg.T - 1)

    t = torch.full((B,), repair_t, device=cfg.device, dtype=torch.long)
    xt, _ = schedule.q_sample(cond, t)

    cur = xt
    for step in reversed(range(repair_t + 1)):
        cur = p_sample(model, schedule, cur, cond, mask, step, cfg)

    return cur.clamp(-1, 1)


# =====================================================
# 8. Baseline
# =====================================================

def gaussian_blur_simple(x, kernel_size=5):
    """
    简单平均模糊 baseline。
    """
    pad = kernel_size // 2
    weight = torch.ones(1, 1, kernel_size, kernel_size, device=x.device)
    weight = weight / weight.sum()
    return F.conv2d(x, weight, padding=pad)


# =====================================================
# 9. 评估与可视化
# =====================================================

def get_eval_mask(mask, cfg: Config):
    if cfg.eval_mask_mode == "all":
        return torch.ones_like(mask)
    elif cfg.eval_mask_mode == "gt":
        return mask
    elif cfg.eval_mask_mode == "none":
        return torch.zeros_like(mask)
    else:
        raise ValueError(f"Unknown eval_mask_mode: {cfg.eval_mask_mode}")


def save_visual_examples(cfg: Config, cond, repaired, amap, mask, n=None):
    """
    只保存包含异常区域的样本，避免 ground_truth_mask 全黑。
    """
    if n is None:
        n = cfg.save_n

    ensure_dir(cfg.out_dir)

    mask_sum = mask.view(mask.shape[0], -1).sum(dim=1)
    anom_idx = torch.where(mask_sum > 0)[0]

    if len(anom_idx) == 0:
        print("[Warning] No anomaly samples in this batch, skip visualization.")
        return False

    anom_idx = anom_idx[:n]

    cond_v = cond[anom_idx].detach().cpu()
    repaired_v = repaired[anom_idx].detach().cpu()
    amap_v = amap[anom_idx].detach().cpu()
    mask_v = mask[anom_idx].detach().cpu()

    save_image_grid(cond_v, os.path.join(cfg.out_dir, "01_input_corrupted.png"), nrow=4)
    save_image_grid(repaired_v, os.path.join(cfg.out_dir, "02_repaired.png"), nrow=4)
    save_heatmap_grid(amap_v, os.path.join(cfg.out_dir, "03_anomaly_map.png"), nrow=4)
    save_mask_grid(mask_v, os.path.join(cfg.out_dir, "04_ground_truth_mask.png"), nrow=4)

    return True


@torch.no_grad()
def evaluate(cfg: Config, model, schedule, test_loader):
    model.eval()

    image_scores = []
    image_labels = []

    pixel_scores = []
    pixel_labels = []

    blur_image_scores = []
    blur_pixel_scores = []

    saved = False

    for batch in tqdm(test_loader, desc="Evaluating"):
        cond = batch["cond"].to(cfg.device)
        mask = batch["mask"].to(cfg.device)
        label = batch["label"].cpu().numpy()

        use_mask = get_eval_mask(mask, cfg)

        repaired = repair_images(model, schedule, cond, use_mask, cfg)
        amap = torch.abs(cond - repaired)

        # 图像级异常分数：top 5% 像素均值
        flat = amap.view(amap.shape[0], -1)
        k = max(1, int(flat.shape[1] * 0.05))
        topk_score = torch.topk(flat, k=k, dim=1).values.mean(dim=1)

        image_scores.extend(topk_score.detach().cpu().numpy().tolist())
        image_labels.extend(label.tolist())

        pixel_scores.extend(amap.detach().cpu().numpy().reshape(-1).tolist())
        pixel_labels.extend(mask.detach().cpu().numpy().reshape(-1).tolist())

        # blur baseline
        if cfg.use_blur_baseline:
            blurred = gaussian_blur_simple(cond, kernel_size=cfg.blur_kernel_size)
            blur_amap = torch.abs(cond - blurred)

            blur_flat = blur_amap.view(blur_amap.shape[0], -1)
            blur_k = max(1, int(blur_flat.shape[1] * 0.05))
            blur_score = torch.topk(blur_flat, k=blur_k, dim=1).values.mean(dim=1)

            blur_image_scores.extend(blur_score.detach().cpu().numpy().tolist())
            blur_pixel_scores.extend(blur_amap.detach().cpu().numpy().reshape(-1).tolist())

        if (not saved) and (mask.sum().item() > 0):
            saved = save_visual_examples(cfg, cond, repaired, amap, mask)

    image_labels_arr = np.array(image_labels)
    pixel_labels_arr = np.array(pixel_labels)

    print("[Debug] Unique image labels:", np.unique(image_labels_arr, return_counts=True))
    print("[Debug] Pixel positive count:", pixel_labels_arr.sum())

    if len(np.unique(image_labels_arr)) < 2:
        img_auc = float("nan")
        print("[Warning] ROC-AUC cannot be computed: only one image class exists.")
    else:
        img_auc = roc_auc_score(image_labels, image_scores)

    if pixel_labels_arr.sum() == 0:
        pix_auprc = 0.0
        print("[Warning] Pixel-level AUPRC cannot be computed: no positive anomaly pixels.")
    else:
        pix_auprc = average_precision_score(pixel_labels, pixel_scores)

    results = {
        "ROC_AUC": float(img_auc),
        "Pixel_AUPRC": float(pix_auprc),
    }

    if cfg.use_blur_baseline:
        if len(np.unique(image_labels_arr)) < 2:
            blur_auc = float("nan")
        else:
            blur_auc = roc_auc_score(image_labels, blur_image_scores)

        if pixel_labels_arr.sum() == 0:
            blur_auprc = 0.0
        else:
            blur_auprc = average_precision_score(pixel_labels, blur_pixel_scores)

        results["Blur_ROC_AUC"] = float(blur_auc)
        results["Blur_Pixel_AUPRC"] = float(blur_auprc)

    return results


# =====================================================
# 10. 单次实验
# =====================================================

def run_single_experiment(cfg: Config):
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    print("\n========== Experiment Config ==========")
    print(cfg)
    print("=======================================\n")

    save_json(asdict(cfg), os.path.join(cfg.out_dir, "config.json"))

    train_loader, test_loader = build_dataloaders(cfg)

    schedule = DiffusionSchedule(cfg)
    model = SmallUNet(in_ch=3, base_ch=64, time_dim=128).to(cfg.device)

    start = time.time()

    losses = train_model(cfg, model, schedule, train_loader)
    plot_loss(losses, cfg)

    results = evaluate(cfg, model, schedule, test_loader)

    elapsed = time.time() - start
    results["Elapsed_Seconds"] = elapsed

    save_json(results, os.path.join(cfg.out_dir, "results.json"))

    with open(os.path.join(cfg.out_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write("========== Results ==========\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
        f.write("\n========== Config ==========\n")
        f.write(str(cfg))

    print("\n========== Results ==========")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print(f"Outputs saved to: {cfg.out_dir}")
    print("=============================\n")

    return results


# =====================================================
# 11. 批量实验
# =====================================================

def clone_cfg(cfg: Config, **kwargs):
    d = asdict(cfg)
    d.update(kwargs)
    return Config(**d)


def run_experiment_group(base_cfg: Config, group_name: str, cfg_list):
    group_dir = os.path.join(base_cfg.out_dir, group_name)
    ensure_dir(group_dir)

    rows = []

    for name, cfg in cfg_list:
        cfg.out_dir = os.path.join(group_dir, name)

        print(f"\n\n############ Running {group_name}/{name} ############")
        res = run_single_experiment(cfg)

        row = {
            "group": group_name,
            "name": name,
            "normal_class": cfg.normal_class,
            "anomaly_type": cfg.anomaly_type,
            "repair_t": cfg.repair_t,
            "T": cfg.T,
            "eval_mask_mode": cfg.eval_mask_mode,
            "stochastic_reverse": cfg.stochastic_reverse,
        }
        row.update(res)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(group_dir, f"{group_name}_summary.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n[Summary saved] {csv_path}")
    print(df)

    return df


def make_repair_sweep(base_cfg: Config):
    vals = [40, 80, 120, 160]
    cfgs = []
    for v in vals:
        cfg = clone_cfg(base_cfg, repair_t=v)
        cfgs.append((f"repair_t_{v}", cfg))
    return cfgs


def make_anomaly_sweep(base_cfg: Config):
    vals = ["rect", "scratch", "noise", "mixed"]
    cfgs = []
    for v in vals:
        cfg = clone_cfg(base_cfg, anomaly_type=v)
        cfgs.append((f"anomaly_{v}", cfg))
    return cfgs


def make_mask_sweep(base_cfg: Config):
    vals = ["all", "gt", "none"]
    cfgs = []
    for v in vals:
        cfg = clone_cfg(base_cfg, eval_mask_mode=v)
        cfgs.append((f"mask_{v}", cfg))
    return cfgs


def make_class_sweep(base_cfg: Config):
    vals = [0, 2, 7, 8]
    cfgs = []
    for v in vals:
        cfg = clone_cfg(base_cfg, normal_class=v)
        cfgs.append((f"class_{v}_{FASHION_CLASSES.get(v, 'unknown').replace('/', '_')}", cfg))
    return cfgs


def make_T_sweep(base_cfg: Config):
    pairs = [
        (100, 60),
        (200, 120),
        (400, 240),
    ]
    cfgs = []
    for T, repair_t in pairs:
        cfg = clone_cfg(base_cfg, T=T, repair_t=repair_t)
        cfgs.append((f"T_{T}_repair_{repair_t}", cfg))
    return cfgs


def make_reverse_sweep(base_cfg: Config):
    cfgs = [
        ("reverse_deterministic", clone_cfg(base_cfg, stochastic_reverse=False)),
        ("reverse_stochastic", clone_cfg(base_cfg, stochastic_reverse=True)),
    ]
    return cfgs


def run_all(base_cfg: Config):
    all_dfs = []

    all_dfs.append(run_experiment_group(base_cfg, "repair_sweep", make_repair_sweep(base_cfg)))
    all_dfs.append(run_experiment_group(base_cfg, "anomaly_sweep", make_anomaly_sweep(base_cfg)))
    all_dfs.append(run_experiment_group(base_cfg, "mask_sweep", make_mask_sweep(base_cfg)))
    all_dfs.append(run_experiment_group(base_cfg, "class_sweep", make_class_sweep(base_cfg)))
    all_dfs.append(run_experiment_group(base_cfg, "T_sweep", make_T_sweep(base_cfg)))
    all_dfs.append(run_experiment_group(base_cfg, "reverse_sweep", make_reverse_sweep(base_cfg)))

    df_all = pd.concat(all_dfs, axis=0, ignore_index=True)
    path = os.path.join(base_cfg.out_dir, "all_experiments_summary.csv")
    df_all.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"\n[All summary saved] {path}")
    print(df_all)

    return df_all


# =====================================================
# 12. 命令行入口
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=[
            "single",
            "repair_sweep",
            "anomaly_sweep",
            "mask_sweep",
            "class_sweep",
            "T_sweep",
            "reverse_sweep",
            "all",
        ],
        help="实验模式",
    )

    parser.add_argument("--out_dir", type=str, default="outputs_single")
    parser.add_argument("--data_root", type=str, default="./data")

    parser.add_argument("--normal_class", type=int, default=0)
    parser.add_argument("--train_limit", type=int, default=8000)
    parser.add_argument("--test_normal_limit", type=int, default=500)
    parser.add_argument("--test_anomaly_limit", type=int, default=500)

    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--repair_t", type=int, default=120)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument(
        "--anomaly_type",
        type=str,
        default="mixed",
        choices=["rect", "scratch", "noise", "mixed"],
    )

    parser.add_argument(
        "--eval_mask_mode",
        type=str,
        default="all",
        choices=["all", "gt", "none"],
    )

    parser.add_argument("--stochastic_reverse", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config(
        seed=args.seed,
        out_dir=args.out_dir,
        data_root=args.data_root,
        image_size=args.image_size,
        normal_class=args.normal_class,
        train_limit=args.train_limit,
        test_normal_limit=args.test_normal_limit,
        test_anomaly_limit=args.test_anomaly_limit,
        T=args.T,
        repair_t=args.repair_t,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        anomaly_type=args.anomaly_type,
        eval_mask_mode=args.eval_mask_mode,
        stochastic_reverse=args.stochastic_reverse,
        num_workers=args.num_workers,
    )

    if cfg.repair_t >= cfg.T:
        raise ValueError(f"repair_t must be smaller than T. Got repair_t={cfg.repair_t}, T={cfg.T}")

    if args.mode == "single":
        run_single_experiment(cfg)

    elif args.mode == "repair_sweep":
        cfg.out_dir = args.out_dir
        run_experiment_group(cfg, "repair_sweep", make_repair_sweep(cfg))

    elif args.mode == "anomaly_sweep":
        cfg.out_dir = args.out_dir
        run_experiment_group(cfg, "anomaly_sweep", make_anomaly_sweep(cfg))

    elif args.mode == "mask_sweep":
        cfg.out_dir = args.out_dir
        run_experiment_group(cfg, "mask_sweep", make_mask_sweep(cfg))

    elif args.mode == "class_sweep":
        cfg.out_dir = args.out_dir
        run_experiment_group(cfg, "class_sweep", make_class_sweep(cfg))

    elif args.mode == "T_sweep":
        cfg.out_dir = args.out_dir
        run_experiment_group(cfg, "T_sweep", make_T_sweep(cfg))

    elif args.mode == "reverse_sweep":
        cfg.out_dir = args.out_dir
        run_experiment_group(cfg, "reverse_sweep", make_reverse_sweep(cfg))

    elif args.mode == "all":
        cfg.out_dir = args.out_dir
        run_all(cfg)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()