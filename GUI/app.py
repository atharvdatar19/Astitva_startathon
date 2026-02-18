import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2

# ================================
# CONFIG
# ================================

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3,
    500: 4, 550: 5, 700: 6,
    800: 7, 7100: 8, 10000: 9
}
n_classes = len(value_map)

h, w = 532, 952   # must match training resize

# ================================
# Enhanced Head (MATCH TRAINING)
# ================================

class EnhancedSegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.conv1 = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.depthwise = nn.Conv2d(512, 512, 7, padding=3, groups=512)
        self.pointwise = nn.Conv2d(512, 256, 1)
        self.bn2 = nn.BatchNorm2d(256)

        self.aspp1 = nn.Conv2d(256, 128, 1)
        self.aspp2 = nn.Conv2d(256, 128, 3, padding=2, dilation=2)
        self.aspp3 = nn.Conv2d(256, 128, 3, padding=4, dilation=4)

        self.fusion = nn.Conv2d(384, 256, 1)
        self.dropout = nn.Dropout2d(0.3)
        self.classifier = nn.Conv2d(256, out_channels, 1)

        self.act = nn.GELU()

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)

        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.depthwise(x))
        x = self.act(self.bn2(self.pointwise(x)))

        a1 = self.act(self.aspp1(x))
        a2 = self.act(self.aspp2(x))
        a3 = self.act(self.aspp3(x))

        x = torch.cat([a1, a2, a3], dim=1)
        x = self.act(self.fusion(x))
        x = self.dropout(x)

        return self.classifier(x)

# ================================
# LOAD MODEL
# ================================

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    st.write("Using Device:", device)

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.to(device)
    backbone.eval()

    # Get embedding dim
    dummy = torch.randn(1, 3, h, w).to(device)
    with torch.no_grad():
        tokens = backbone.forward_features(dummy)["x_norm_patchtokens"]

    embed_dim = tokens.shape[2]

    head = EnhancedSegmentationHead(
        embed_dim, n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    ).to(device)

    checkpoint = torch.load("BEST_hackathon_model.pth", map_location=device)

    # Correct key loading
    head.load_state_dict(checkpoint["model_state_dict"])

    head.eval()

    return backbone, head, device

# ================================
# STREAMLIT UI
# ================================

st.title("🚀 Desert Segmentation Hackathon GUI")

backbone, head, device = load_model()

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", use_column_width=True)

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        tokens = backbone.forward_features(x)["x_norm_patchtokens"]
        logits = head(tokens)
        outputs = F.interpolate(logits, size=(h, w), mode="bilinear")

    pred = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()

    # Colorize prediction
    palette = np.random.randint(0, 255, (n_classes, 3), dtype=np.uint8)
    color_mask = palette[pred]

    st.image(color_mask, caption="Predicted Mask", use_column_width=True)

    cv2.imwrite("prediction.png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    st.success("Saved prediction.png successfully!")
