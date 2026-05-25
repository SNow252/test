# ddpm_mnist_demo.py
# -*- coding: utf-8 -*-

"""
最基础 DDPM 扩散模型 Demo：MNIST 图像生成

功能：
1. 使用 MNIST 训练一个小型 U-Net 噪声预测网络；
2. 实现正向扩散 q(x_t | x_0)；
3. 实现反向采样 p_theta(x_{t-1} | x_t)；
4. 从纯高斯噪声生成手写数字图像。

运行：
python ddpm_mnist_demo.py
"""

import os
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils
from tqdm import tqdm


# =========================
# 1. 配置
# =========================

@dataclass
class Config:
    image_size: int = 28
    channels: int = 1

    # 扩散步数。越大越接近标准 DDPM，但训练和采样更慢
    T: int = 200

    beta_start: float = 1e-4
    beta_end: float = 0.02

    batch_size: int = 128
    epochs: int = 20
    lr: float = 2e-4

    sample_every_epoch: bool = True
    num_samples: int = 64

    data_dir: str = "./data"
    out_dir: str = "./ddpm_outputs"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
os.makedirs(cfg.out_dir, exist_ok=True)


# =========================
# 2. 时间步嵌入
# =========================

class SinusoidalTimeEmbedding(nn.Module):
    """
    将时间步 t 编码成向量。
    扩散模型需要知道当前处于第几个噪声等级。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        """
        t: [B]
        return: [B, dim]
        """
        half_dim = self.dim // 2
        device = t.device

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        return emb


# =========================
# 3. 小型 U-Net
# =========================

class ResBlock(nn.Module):
    """
    简化残差块。
    输入图像特征 + 时间嵌入。
    """

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_ch)

        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # 将时间嵌入加到特征图上
        time_bias = self.time_mlp(t_emb)
        time_bias = time_bias[:, :, None, None]
        h = h + time_bias

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.skip(x)


class SimpleUNet(nn.Module):
    """
    一个非常小的 U-Net。
    输入 x_t 和 t，输出预测噪声 epsilon_theta(x_t, t)。
    """

    def __init__(self, img_ch=1, base_ch=64, time_dim=128):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Encoder
        self.enc1 = ResBlock(img_ch, base_ch, time_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)

        self.enc2 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)

        # Middle
        self.mid = ResBlock(base_ch * 2, base_ch * 4, time_dim)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock(base_ch * 4, base_ch * 2, time_dim)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock(base_ch * 2, base_ch, time_dim)

        self.out = nn.Conv2d(base_ch, img_ch, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

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


# =========================
# 4. DDPM 扩散过程
# =========================

class DDPM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device

        # beta_t: 每一步加入的噪声强度
        self.betas = torch.linspace(
            cfg.beta_start,
            cfg.beta_end,
            cfg.T,
            device=self.device
        )

        self.alphas = 1.0 - self.betas

        # alpha_bar_t = alpha_1 * ... * alpha_t
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def q_sample(self, x0, t, noise=None):
        """
        正向扩散闭式公式：

        x_t = sqrt(alpha_bar_t) * x_0
              + sqrt(1 - alpha_bar_t) * epsilon

        x0: [B, C, H, W]
        t: [B]
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]

        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

        return xt, noise

    @torch.no_grad()
    def p_sample(self, model, xt, t_index):
        """
        单步反向采样：
        从 x_t 得到 x_{t-1}
        """
        B = xt.shape[0]
        t = torch.full((B,), t_index, device=self.device, dtype=torch.long)

        beta_t = self.betas[t_index]
        alpha_t = self.alphas[t_index]
        alpha_bar_t = self.alpha_bars[t_index]

        # 模型预测噪声
        eps_theta = model(xt, t)

        # DDPM 反向均值公式
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            xt - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
        )

        if t_index == 0:
            return mean

        noise = torch.randn_like(xt)
        sigma_t = torch.sqrt(beta_t)

        return mean + sigma_t * noise

    @torch.no_grad()
    def sample(self, model, n):
        """
        从纯高斯噪声开始，逐步去噪生成图像。
        """
        model.eval()

        x = torch.randn(
            n,
            self.cfg.channels,
            self.cfg.image_size,
            self.cfg.image_size,
            device=self.device
        )

        for t_index in tqdm(reversed(range(self.cfg.T)), desc="Sampling"):
            x = self.p_sample(model, x, t_index)

        return x


# =========================
# 5. 数据加载
# =========================

def get_dataloader(cfg: Config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # [0,1] -> [-1,1]
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    return loader


# =========================
# 6. 保存图像
# =========================

def save_samples(samples, path):
    """
    samples: [-1,1]
    保存为图片。
    """
    samples = (samples.clamp(-1, 1) + 1.0) / 2.0
    utils.save_image(samples, path, nrow=8)


# =========================
# 7. 训练
# =========================

def train():
    print("Using device:", cfg.device)

    loader = get_dataloader(cfg)

    model = SimpleUNet(
        img_ch=cfg.channels,
        base_ch=64,
        time_dim=128
    ).to(cfg.device)

    ddpm = DDPM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}")

        total_loss = 0.0

        for x0, _ in pbar:
            x0 = x0.to(cfg.device)

            B = x0.shape[0]

            # 随机采样时间步 t
            t = torch.randint(
                low=0,
                high=cfg.T,
                size=(B,),
                device=cfg.device
            )

            # 真实噪声 epsilon
            noise = torch.randn_like(x0)

            # 正向扩散得到 x_t
            xt, true_noise = ddpm.q_sample(x0, t, noise)

            # 模型预测噪声
            pred_noise = model(xt, t)

            # DDPM 最基本训练目标：预测噪声
            loss = F.mse_loss(pred_noise, true_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: avg loss = {avg_loss:.6f}")

        # 每轮保存生成图
        if cfg.sample_every_epoch:
            samples = ddpm.sample(model, cfg.num_samples)
            save_path = os.path.join(cfg.out_dir, f"samples_epoch_{epoch}.png")
            save_samples(samples, save_path)
            print("Saved:", save_path)

        # 保存模型
        torch.save(model.state_dict(), os.path.join(cfg.out_dir, "ddpm_mnist.pth"))

    print("Training finished.")


if __name__ == "__main__":
    train()