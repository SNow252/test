# Diffusion Anomaly Detection Simulation

This project implements a simple diffusion-model-based image anomaly detection simulation on Fashion-MNIST.

The model learns the normal image distribution through a forward diffusion and reverse denoising process. During testing, abnormal images are repaired by the diffusion model, and anomaly maps are obtained from the difference between the input image and the repaired image.

## Features

- Diffusion-based anomaly detection
- Fashion-MNIST simulation
- Artificial anomalies: rectangle, scratch, noise, and mixed
- Evaluation with ROC-AUC and Pixel-level AUPRC
- Ablation experiments for repair steps, anomaly types, mask conditions, and reverse sampling

## Installation and quick start

```bash
pip install -r requirements.txt
python diffusion_image_ad_full.py --mode single --epochs 8 --repair_t 40
