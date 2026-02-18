import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
from tqdm import tqdm

# ================================
# MASK MAP
# ================================

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3,
    500: 4, 550: 5, 700: 6, 800: 7,
    7100: 8, 10000: 9
}

n_classes = len(value_map)

# Color palette
color_palette = np.array([
    [0, 0, 0],
    [34, 139, 34],
    [0, 255, 0],
    [210, 180, 140],
    [139, 90, 43],
    [128, 128, 0],
    [139, 69, 19],
    [128, 128, 128],
    [160, 82, 45],
    [135, 206, 235],
], dtype=np.uint8)


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in value_map.items():
        new_arr[arr == raw] = new
    return new_arr


def mask_to_color(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in range(n_classes):
        out[mask == cid] = color_palette[cid]
    return out


# ================================
# DATASET
# ================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_dir = os.path.join(data_dir, "Color_Images")
        self.ids = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, name


# ================================
# ENHANCED HEAD (MATCH TRAINING)
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
# MAIN TEST
# ================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device:", device)

    h, w = 518, 938  # MUST MATCH TRAINING

    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(script_dir, "BEST_hackathon_model.pth")

    test_dir = os.path.join(
    script_dir,
    "Offroad_Segmentation_Training_Dataset",
    "train")


    output_dir = os.path.join(script_dir, "TEST_PREDICTIONS")
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = MaskDataset(test_dir, transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    print("Loaded test images:", len(dataset))

    # Load backbone
    print("Loading DINOv2 backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.to(device)
    backbone.eval()

    # Get embedding dim
    sample, _ = dataset[0]
    sample = sample.unsqueeze(0).to(device)

    with torch.no_grad():
        tokens = backbone.forward_features(sample)["x_norm_patchtokens"]

    embed_dim = tokens.shape[2]

    # Load head
    head = EnhancedSegmentationHead(
        embed_dim, n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    head.load_state_dict(checkpoint["model_state_dict"])
    head.eval()

    print(" Model Loaded Successfully!")

    # Inference
    with torch.no_grad():
        for imgs, names in tqdm(loader):
            imgs = imgs.to(device)

            tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = head(tokens)

            outputs = F.interpolate(logits, size=(h, w), mode="bilinear")
            preds = torch.argmax(outputs, dim=1)

            for i in range(preds.shape[0]):
                mask = preds[i].cpu().numpy()
                color_mask = mask_to_color(mask)

                save_path = os.path.join(output_dir, names[i].replace(".png", "_pred.png"))
                cv2.imwrite(save_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

    print("\n==============================")
    print("DONE! Predictions saved in:")
    print(output_dir)
    print("==============================")


if __name__ == "__main__":
    main()
