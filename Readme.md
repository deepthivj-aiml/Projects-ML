# Automatic Lens Correction — A100-Optimized CNN

A high-performance deep learning pipeline for automatic lens distortion correction using EfficientNetB0 and differentiable Brown-Conrady undistortion, optimized for NVIDIA A100 GPUs.

## 🎯 Overview

This project trains a CNN to predict lens distortion coefficients (k₁, k₂, p₁, p₂) from distorted images and applies differentiable undistortion to recover ground truth geometry. The implementation is specifically optimized for Colab Pro A100 instances with a **10-minute end-to-end pipeline**.

### Key Features

- **bfloat16 Mixed Precision**: 3× throughput vs float32 on A100, no loss scaling needed
- **XLA JIT Compilation**: 20–40% additional speedup via kernel fusion
- **EfficientNetB0 Backbone**: Pretrained ImageNet weights, efficient feature extraction
- **Differentiable Geometry**: Brown-Conrady distortion model with backprop-safe sampling
- **Two-Phase Training**: Frozen backbone → progressive fine-tuning
- **Memory-Safe Pipeline**: ~1.3 GB peak RAM via tf.data + local SSD streaming
- **Parallel I/O**: gsutil bulk download + native TF decode (15–20× faster than GCS Python clients)

## 📊 Architecture
Distorted Image (384×384) ↓ 
[CNN Encoder] • EfficientNetB0 (pretrained) • Global Average Pooling → (1280,) ↓ 
[Regression Head] • Dense(512) + Dropout(0.3) • Dense(128) + Dropout(0.2) • Dense(4, tanh) + Scale ↓ 
[k₁, k₂, p₁, p₂] coefficients ↓ 
[Differentiable Brown-Conrady Undistortion] ↓
Undistorted Image (384×384) ↓
Loss = 0.8 × (1 - SSIM) + 0.2 × L1


## ⚙️ Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Input Size** | 224×224 | CNN input (EfficientNetB0 standard) |
| **Undistort Size** | 384×384 | Higher res = better geometric detail |
| **Batch Size** | 64 | Fills A100 40GB VRAM |
| **Epochs** | 15 | ~15s/epoch on A100 |
| **Learning Rate** | 3e-4 | Linear scaling for batch size 64 |
| **Phase 2 Start** | Epoch 6 | Unfreeze top-60 backbone layers |
| **Loss Alpha** | 0.8 | 80% SSIM + 20% L1 |
| **Early Stop Patience** | 3 | Stop if val SSIM doesn't improve |

## 📈 Expected Performance (A100 Colab)

| Stage | Time |
|-------|------|
| GCS download | 2–3 min |
| Training (15 epochs) | 3–5 min |
| Evaluation + ZIP | 1–2 min |
| **Total** | **6–10 min** ✓ |

### Per-Epoch Breakdown
- **Epoch 1**: ~25s (XLA JIT compilation)
- **Epochs 2–5**: ~8s each (Phase 1: head only)
- **Epochs 6–15**: ~15–20s each (Phase 2: fine-tune)

## 🚀 Quick Start

### Prerequisites
- Google Colab Pro with A100 GPU access
- GCP project with GCS bucket containing lens distortion datasets
- `tensorflow >= 2.14`, `opencv-python`, `scikit-image`

### Installation

```python
# In Colab cell 1:
from google.colab import auth
auth.authenticate_user()

# Run the full notebook (it handles all setup)
Usage
Update Config (cell with CONFIG section):

Python
GCP_PROJECT_ID      = "your-project-id"
GCS_BUCKET_NAME     = "your-bucket"
GCS_TRAIN_FULL_PATH = "path/to/training/images/"
GCS_TEST_FULL_PATH  = "path/to/test/images/"
Run the notebook top to bottom:

Step 1: Downloads training/test images via gsutil -m cp
Step 2: Loads image helpers
Step 3: Builds differentiable undistortion layer
Step 4: Constructs CNN model
Step 5: Defines loss functions
Step 6: Creates tf.data pipeline
Step 7: Two-phase training loop
Step 8: Evaluation + visualization
Outputs saved to ./output/:

lens_cnn_model_a100.keras — trained model
lens_correction_cnn_a100.zip — predictions + side-by-side comparisons
training_curves_a100.png — loss/SSIM plots
🔧 Advanced Tuning
For T4 GPUs (Colab Free)
Python
# Reduce resolution (less memory needed)
UNDISTORT_SIZE = 256
BATCH_SIZE = 32

# Use float16 with loss scaling
tf.keras.mixed_precision.set_global_policy('mixed_float16')
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE,
                                      loss_scale='dynamic')
For RTX 3090 / 4090 (Local)
Python
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
UNDISTORT_SIZE = 512
📚 Key Implementation Details
Why gsutil -m cp?
Direct I/O optimization: Google's own CLI tool uses C extensions + XML multipart API
Automatic parallelization: Splits work across multiple OS threads
No Python bottleneck: GIL-free, region-aware routing
Result: ~23,000 image pairs downloaded in 60–90 seconds
Why bfloat16?
Wide exponent range: Same as float32, no underflow → no LossScaleOptimizer needed
Native A100 support: 3× faster than float32 on tensor cores
Stable training: No loss scaling artifacts with plain Adam
Why XLA?
Kernel fusion: Combines individual GPU ops into larger, optimized kernels
20–40% speedup: On top of bfloat16 benefits
First-call overhead: ~10s compilation, cached thereafter
Why two-phase training?
Phase 1 (head only): Quick convergence on new task, ~8s/epoch
Phase 2 (fine-tune): Adapt ImageNet features to lens distortion, ~15s/epoch
BatchNorm frozen: Preserves ImageNet statistics in backbone
📊 Data Format
Training Data
Code
gs://bucket/path/
├── image_001_original.jpg    (distorted)
├── image_001_generated.jpg   (ground truth, undistorted)
├── image_002_original.jpg
├── image_002_generated.jpg
└── ...
Test Data
Code
gs://bucket/path/
├── test_001.jpg
├── test_002.jpg
└── ...
🎓 Loss Function
Code
Loss = α × (1 - SSIM) + (1 - α) × L1

where:
  SSIM  : Structural Similarity Index (rewards geometric accuracy)
  L1    : Mean Absolute Pixel Error (prevents zero-coefficient degenerate solutions)
  α     : 0.8 (80% SSIM, 20% L1)
📝 Output Coefficients
The model predicts Brown-Conrady distortion parameters scaled to physical ranges:

Coefficient	Range	Meaning
k₁	[-1.0, 1.0]	Primary radial distortion
k₂	[-0.5, 0.5]	Secondary radial distortion
p₁	[-0.1, 0.1]	Tangential distortion (x-axis)
p₂	[-0.1, 0.1]	Tangential distortion (y-axis)
🔍 Evaluation Metrics
SSIM (Structural Similarity): Primary metric — higher is better [0, 1]
L1 Loss: Mean absolute pixel error [0, 255]
Validation SSIM: Reported per epoch; model saved when improved
📦 Dependencies
Code
tensorflow >= 2.14
numpy
pandas
opencv-python
scikit-image
matplotlib
google-cloud-storage
google-auth
Pillow
tqdm
psutil
🐛 Troubleshooting
"❌ No GPU detected!"
→ Runtime → Change runtime type → Hardware accelerator → A100 GPU → Save

gsutil auth errors
→ Re-run auth.authenticate_user() at the top, then cell with gsutil commands

OOM (Out of Memory)
Python
# Reduce batch size or resolution
BATCH_SIZE = 32
UNDISTORT_SIZE = 256
Slow epoch times
→ Ensure A100 selected (not T4); check GPU with nvidia-smi

📄 References
Brown-Conrady Distortion Model
EfficientNet
bfloat16 on A100
XLA: Optimizing Compiler for TensorFlow
📄 License
[Add your license here]

👤 Author
Deepthi V J
Joshua Jose
