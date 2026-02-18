 Domain‑Robust Semantic Segmentation for Synthetic Desert Environments
TEAM ASTITVA
Startathon Hackathon 2026 Submission

Atharv Datar
Raunak Sharma
Raghav Mishra
Aditi Sharma

🚀 Project Overview
Autonomous navigation in desert and off‑road environments requires precise pixel‑level scene understanding. Terrain types such as sand, dry grass, shrubs, rocks, logs, and sky often appear visually similar, making traditional computer vision approaches unreliable.

This project implements a deep learning‑based multi‑class semantic segmentation system trained on synthetic desert twin data generated via FalconEditor. The system performs pixel‑wise classification and is optimized for Intersection over Union (IoU) performance under hackathon constraints.

🎯 Objective
Perform multi‑class semantic segmentation on desert terrain images

Maximize Mean IoU score

Ensure generalization to unseen validation/test environments

Provide an interactive GUI for real‑time inference

🧠 Methodology
1️⃣ Backbone Network
DINOv2 Vision Transformer (ViT‑S/14)

Pretrained foundation model

Frozen backbone during training

Extracts high‑quality patch embeddings

2️⃣ Custom Segmentation Head
We implemented an enhanced segmentation head featuring:

Convolutional feature extraction

Depthwise separable convolutions

Multi‑scale ASPP blocks

Feature fusion layers

Dropout regularization

This improves contextual understanding across large desert scenes.

3️⃣ Loss Functions
To improve segmentation quality across imbalanced classes:

Cross Entropy Loss

Dice Loss (IoU‑oriented optimization)

Focal Loss (hard class emphasis)

Final loss combination:

0.4 CE + 0.4 Dice + 0.2 Focal
4️⃣ Optimization Strategy
AdamW optimizer

Cosine Learning Rate Scheduler

Mixed Precision (AMP)

Early Stopping

Best Model Checkpointing

📂 Dataset Structure
Offroad_Segmentation_Training_Dataset/
│
├── train/
│   ├── Color_Images/
│   ├── Segmentation/
│
├── val/
│   ├── Color_Images/
│   ├── Segmentation/
Classes (10 Total)
ID	Class Name
0	Background
100	Trees
200	Lush Bushes
300	Dry Grass
500	Dry Bushes
550	Ground Clutter
700	Logs
800	Rocks
7100	Landscape
10000	Sky
📊 Performance Results
Best Validation Mean IoU Achieved:
0.408
Observations:
Sky and Landscape achieved high IoU

Fine‑grained vegetation classes remain challenging

Model generalizes well to unseen terrain variations

🖥️ GUI Demonstration
An interactive Streamlit GUI was developed for real‑time inference.

Features:
Upload custom desert image

Run segmentation prediction

Visualize predicted mask

Save prediction output

Run GUI:
streamlit run app.py
▶️ How to Reproduce Results
1️⃣ Create Environment
conda env create -f environment.yml
conda activate startathon-segmentation
2️⃣ Train Model
python train_segmentation.py
Best model automatically saved as:

BEST_hackathon_model.pth
3️⃣ Run Evaluation
python test_segmentation.py
Outputs:

Mean IoU

Per‑class IoU

Saved predicted masks

4️⃣ Run GUI
streamlit run app.py
⚙️ Tech Stack
Python 3.10

PyTorch

Torchvision

DINOv2

OpenCV

Streamlit

NumPy

Matplotlib

🧩 Challenges Faced
Severe class imbalance across vegetation types

High visual similarity between dry grass and dry bushes

Domain shift between synthetic and validation environments

Memory constraints on RTX 4050

🔮 Future Improvements
Unfreezing last transformer blocks for fine‑tuning

Class‑weighted Dice/Focal loss

Data augmentation enhancement

Multi‑scale training

Test‑time augmentation

Lightweight deployment model for edge inference

🏁 Conclusion
This project demonstrates a robust approach to synthetic desert semantic segmentation using foundation models and multi‑scale convolutional heads. Despite class imbalance challenges, the system achieves competitive IoU performance and includes a complete training‑evaluation‑deployment pipeline with GUI visualization.

The combination of strong technical implementation and clear documentation makes this solution ready for real‑world off‑road perception systems.

👥 Team ASTITVA
Atharv Datar
Raunak Sharma
Raghav Mishra
Aditi Sharma

Startathon Hackathon 2026

