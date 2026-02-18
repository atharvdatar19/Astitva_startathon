Stratathon Hackathon Submission

🌵 Problem Statement:

Off‑road and desert environments contain visually similar terrain categories such as sand, dry grass, bushes, rocks, and sky. Accurate pixel‑level classification is critical for:

Autonomous navigation

Terrain understanding

Robotics perception

Off-road vehicle safety systems

Traditional computer vision approaches struggle in unstructured environments.
This project implements a deep learning–based multi-class semantic segmentation system optimized for Intersection over Union (IoU).

💡 Proposed Solution
We develop a high-performance segmentation pipeline that:

Classifies each pixel into one of 10 terrain categories

Uses transfer learning from a pretrained Vision Transformer

Optimizes specifically for IoU performance

Runs efficiently on NVIDIA RTX GPUs

🧠 Model Architecture
🔹 Backbone
We use DINOv2 ViT-S/14 (Vision Transformer) as a pretrained feature extractor.

Loaded via torch.hub

Frozen during training for stability

Provides strong semantic embeddings

Embedding dimension: 384

🔹 Segmentation Head
A lightweight ConvNeXt-style segmentation head is trained on top of DINOv2 features:

Depthwise separable convolutions

GELU activations

1×1 classification layer

Patch reshaping compatible with ViT token output

This hybrid approach combines:

Transformer-level global understanding

Convolutional spatial refinement

📂 Dataset Structure
Dataset is split into:

train/
    Color_Images/
    Segmentation/

val/
    Color_Images/
    Segmentation/

test/
    Color_Images/
Each segmentation mask contains pixel values mapped to 10 semantic classes:

Background

Trees

Lush Bushes

Dry Grass

Dry Bushes

Ground Clutter

Logs

Rocks

Landscape

Sky

⚠️ Dataset not included due to size constraints.

⚙️ Training Strategy
🔥 Loss Function (IoU Optimized)
We combine:

CrossEntropy Loss

Dice Loss

This improves:

Class balance handling

Boundary precision

Overall IoU score

Final Loss:

Loss = CrossEntropy + DiceLoss
⚡ Performance Optimizations
Frozen DINOv2 backbone

Mixed Precision Training (AMP)

AdamW optimizer

Strong augmentations:

Horizontal Flip

Color Jitter

Patch-safe resizing (multiple of 14)

🖥️ Hardware & Environment
Hardware Used

NVIDIA RTX 4050 (Laptop GPU)

Intel i5 (HP Victus)

Software

Windows 11

Python 3.10+

PyTorch

torchvision

NumPy

OpenCV

tqdm

Matplotlib

GPU usage is automatically detected:

CUDA Available: True
Using Device: cuda
📊 Final Validation Results
Metric	Value
Best Validation IoU	0.3682
Validation Dice Score	0.5223
Validation Pixel Accuracy	0.7299
Final Training IoU	0.4272
The model demonstrates:

Strong generalization

Stable convergence

Balanced multi-class performance

▶️ How To Run
1️⃣ Activate Environment
conda activate hackenv
or activate your virtual environment.

2️⃣ Train Model
python train_segmentation.py
Outputs:

Best model saved as BEST_segmentation_head.pth

Training logs printed per epoch

Validation IoU tracked

3️⃣ Run Testing / Inference
python test_segmentation.py
Outputs:

Raw predicted masks

Colored segmentation masks

IoU evaluation metrics

Comparison visualizations

Saved inside:

/predictions/
📈 Interpretation of Results
Vegetation and sky classes achieve high IoU.

Minor boundary noise appears in dry grass vs dry bush regions.

Overall segmentation is stable and visually consistent.

Higher IoU indicates better region overlap between predicted and ground truth masks.

♻️ Reproducibility
To reproduce results:

Maintain the dataset folder structure.

Use default hyperparameters.

Train → Validate → Test sequentially.

Ensure CUDA-enabled GPU is available for optimal performance.

🏆 Key Strengths:

Uses modern pretrained Vision Transformer (DINOv2)

IoU-optimized hybrid loss function

Efficient GPU-accelerated training

Strong augmentation strategy

Clean modular training & evaluation pipeline

Hackathon-ready reproducibility

👥 Team:

Atharv Datar

Raunak Sharma

Raghav Mishra

Aditi Sharma

🎯 Final Statement:

This project demonstrates a practical and scalable deep learning solution for off-road semantic segmentation using transformer-based feature extraction and IoU-focused optimization. The model achieves strong validation performance while maintaining computational efficiency suitable for real-world deployment.

