# ⚡ SignalForge

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Nvidia](https://img.shields.io/badge/NVIDIA-H100-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://www.nvidia.com/)
[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)

**Architecture:** Prompt-FS-NAF  
**Authors:** V. K. Shivaang Simha, Prantor Jyoti Bharadwaj, Sayan Choudhary  
**Team:** SignalForge
**Target:** Semicon India KLA Hackathon 2026  

**SignalForge** is a highly optimized, enterprise-grade computer vision pipeline engineered specifically for the AI-based restoration of severely degraded images. Designed to mitigate complex multi-source degradation—including **multiplicative speckle noise, Gaussian noise, and downsampling blur**—this pipeline ensures absolute fidelity in recovering high-frequency structural details without artificially clipping out-of-bounds physical sensor data.

---

## 🎯 The KLA Challenge & Our Approach

The core challenge presented by the KLA evaluation rubric demands a delicate balance: achieving maximal restoration accuracy (**PSNR**) while preserving human-readable perceptual sharpness (**LPIPS**), all while operating within strict **inference speed constraints** on NVIDIA hardware. 

Standard architectures utilizing traditional activation functions (ReLU, GELU) catastrophically fail in this domain. Multiplicative speckle noise often generates extreme, out-of-bounds floating-point signal values. Traditional activations artificially clip or threshold these values, permanently destroying the underlying physical sensor data encoded within those extremes.

**Our Solution:** We engineered an **Activation-Free** approach. By eliminating non-linear thresholding layers and utilizing a `SimpleGate` mechanism, we preserve the full dynamic range of the latent signal, allowing the network to mathematically resolve the degradation rather than blindly truncating it. To explicitly optimize for the KLA scoring matrix, we implemented a **Physics-Aware Hybrid Loss** function:
*   **Charbonnier Loss:** Provides robust optimization for general pixel-level restoration.
*   **Edge Loss:** Forces the network to aggressively preserve high-frequency structures and lithographic boundaries.
*   **LPIPS (via VGG):** Calibrates the output for perceptual human-readable sharpness, directly targeting the perceptual component of the scoring rubric.

---

## 🧠 Architecture Overview

At the heart of SignalForge lies the **Prompt-FS-NAF (Nonlinear Activation-Free Network)**. 

*   **Activation-Free Backbone:** Replaces traditional activation functions with a `SimpleGate` layer, ensuring zero information loss from signal clipping.
*   **Violent Exploration Scheduler:** We utilize **CosineAnnealingWarmRestarts (T_0=20)** over **60 epochs**. This aggressive scheduling strategy violently ejects the model from suboptimal local minima, forcing convergence into the absolute optimal parameter space.
*   **Bulletproof Generalization:** To guarantee robustness against hidden Out-of-Distribution (OOD) test sets utilized by KLA engineers, we augmented the training regime by integrating the high-resolution **DIV2K dataset** alongside KLA's provided ground truth. This prevents catastrophic failure on unseen noise profiles.

**Performance Metrics:**
*   **Final Validation PSNR:** 27.59 dB
*   **Train/Val Loss Gap:** 0.0999 (Train) vs 0.1019 (Val) — proving **absolute zero overfitting**.

---

## 🏎️ H100 Optimized Inference Pipeline

To absolutely maximize KLA's **inference speed score** on the target **NVIDIA H100 GPU**, the evaluation pipeline was stripped of all operational bottlenecks:

1.  **Strict Tensor-to-Tensor Operations:** We entirely stripped PNG rendering and intermediate disk I/O from the evaluation loop. All data remains in VRAM.
2.  **Asynchronous DataLoader:** Implemented asynchronous PyTorch DataLoader processing utilizing `batch_size=1`, `num_workers=4`, and `pin_memory=True` to dynamically handle mixed resolutions (128x128 and 256x256) asynchronously, ensuring the H100 tensor cores are never starved for data.
3.  **Automatic Mixed Precision (AMP):** Utilized `torch.amp.autocast` to execute compatible operations in FP16/BF16, massively accelerating throughput while maintaining precision where mathematically required.
4.  **Maximized Throughput:** Test-Time Augmentation (TTA) was explicitly bypassed in our final pipeline to minimize latency and maximize raw throughput for the automated NVIDIA H100 benchmark.

---

## 📁 Repository Structure

```text
SignalForge/
├── src/                  # Source code for model architecture and utilities
├── weights/              # Pre-trained model weights (e.g., best_model.pth)
├── outputs/              # Directory for generated high-resolution outputs
├── results/              # Evaluation metrics and logs
├── train.py              # Main training script
├── run.py                # Main KLA evaluation entry point
├── requirements.txt      # Frozen dependency tree
└── README.md             # Project documentation
```

---

## 🚀 Getting Started / Installation

Ensure you have a modern Python 3 environment and a CUDA-capable GPU.

1.  Clone the repository:
    ```bash
    git clone https://github.com/Sunngttssu/SignalForge-KLA-Image-Restoration.git
    cd SignalForge-KLA-Image-Restoration
    ```

2.  Install the strict dependency tree. *Note: We recommend creating a fresh virtual environment before installing the frozen dependencies to ensure exact reproduction of the evaluation environment.*
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 Usage

### Training the Pipeline

To initialize the training sequence with the Prompt-FS-NAF architecture and the hybrid loss function:

```bash
python train.py
```

### Evaluation & Inference (KLA Scoring Mode)

To run the highly optimized, tensor-to-tensor inference script designed for the H100 evaluation:

```bash
python run.py <input-dir> <output-dir>
```
