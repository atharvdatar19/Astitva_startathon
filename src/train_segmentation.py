"""
OPTIMIZED Hackathon Segmentation Training Script for RTX 4050
------------------------------------------------------------
✅ DINOv2 Backbone Frozen
✅ CrossEntropy + Dice Loss + Focal Loss
✅ Mixed Precision AMP
✅ Optimized Augmentations for Laptop
✅ Learning Rate Scheduling
✅ Early Stopping & Model Checkpointing
✅ Memory-Efficient Batch Processing
✅ Enhanced Loss Functions
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# CONFIGURATION FOR RTX 4050 OPTIMIZATION
# =====================================================

# Memory and performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# =====================================================
# MASK VALUE MAP
# =====================================================

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

n_classes = len(value_map)

# =====================================================
# ENHANCED MASK CONVERSION
# =====================================================

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    
    for raw, new in value_map.items():
        new_arr[arr == raw] = new
    
    return Image.fromarray(new_arr)

# =====================================================
# ENHANCED DATASET WITH MEMORY OPTIMIZATION
# =====================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None, is_training=True):
        self.image_dir = os.path.join(data_dir, "Color_Images")
        self.mask_dir = os.path.join(data_dir, "Segmentation")
        self.ids = sorted(os.listdir(self.image_dir))
        self.transform = transform
        self.mask_transform = mask_transform
        self.is_training = is_training

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        
        # Efficient image loading
        img = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name))
        mask = convert_mask(mask)

        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = torch.squeeze(mask).long()
        return img, mask

# =====================================================
# IMPROVED SEGMENTATION HEAD
# =====================================================

class EnhancedSegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        
        # Multi-scale feature processing
        self.conv1 = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        
        # Depthwise separable convolutions for efficiency
        self.depthwise = nn.Conv2d(512, 512, 7, padding=3, groups=512)
        self.pointwise = nn.Conv2d(512, 256, 1)
        self.bn2 = nn.BatchNorm2d(256)
        
        # Multi-scale context
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
        
        # Feature extraction
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.depthwise(x))
        x = self.act(self.bn2(self.pointwise(x)))
        
        # Multi-scale processing
        aspp1 = self.act(self.aspp1(x))
        aspp2 = self.act(self.aspp2(x))
        aspp3 = self.act(self.aspp3(x))
        
        # Feature fusion
        x = torch.cat([aspp1, aspp2, aspp3], dim=1)
        x = self.act(self.fusion(x))
        x = self.dropout(x)
        
        return self.classifier(x)

# =====================================================
# ENHANCED LOSS FUNCTIONS
# =====================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def enhanced_dice_loss(pred, target, smooth=1e-6):
    pred = torch.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=n_classes)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()
    
    # Class-wise dice calculation
    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    
    # Weight by class frequency (optional)
    return 1 - dice.mean()

# =====================================================
# ENHANCED IOU METRIC
# =====================================================

def compute_enhanced_iou(pred, target, ignore_background=False):
    pred = torch.argmax(pred, dim=1)
    ious = []
    
    start_cls = 1 if ignore_background else 0
    
    for cls in range(start_cls, n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            continue
            
        ious.append((intersection / union).item())
    
    return np.mean(ious) if ious else 0.0

# =====================================================
# LEARNING RATE SCHEDULER
# =====================================================

class CosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + np.cos(np.pi * self.T_cur / self.T_0)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.T_cur = epoch % self.T_0
        super().step(epoch)

# =====================================================
# MAIN TRAINING FUNCTION
# =====================================================

def main():
    print("\n" + "="*50)
    print("OPTIMIZED TRAINING FOR RTX 4050")
    print("="*50)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    # Optimized parameters for RTX 4050
    h, w = 518, 938  # Slightly adjusted for better memory usage
    batch_size = 6   # Optimal for 6GB VRAM
    lr = 2e-4        # Higher learning rate for faster convergence
    epochs = 25      # More epochs for better performance
    
    print(f"\nTraining Configuration:")
    print(f"Image Size: {h}x{w}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {lr}")
    print(f"Epochs: {epochs}")
    
    # Enhanced transforms
    train_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=Image.NEAREST),
        transforms.PILToTensor()
    ])

    # Dataset paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, "Offroad_Segmentation_Training_Dataset", "train")
    val_dir = os.path.join(script_dir, "Offroad_Segmentation_Training_Dataset", "val")

    # Datasets
    trainset = MaskDataset(train_dir, train_transform, mask_transform, True)
    valset = MaskDataset(val_dir, val_transform, mask_transform, False)

    # Optimized data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"Train samples: {len(trainset)}")
    print(f"Val samples: {len(valset)}")

    # Load backbone
    print("\nLoading DINOv2 backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.to(device)
    backbone.eval()
    
    for p in backbone.parameters():
        p.requires_grad = False

    # Get embedding dimension
    sample_imgs, _ = next(iter(train_loader))
    sample_imgs = sample_imgs.to(device)
    
    with torch.no_grad():
        tokens = backbone.forward_features(sample_imgs)["x_norm_patchtokens"]
    
    embed_dim = tokens.shape[2]
    print(f"Embedding dimension: {embed_dim}")

    # Enhanced segmentation head
    head = EnhancedSegmentationHead(
        embed_dim, n_classes,
        tokenW=w // 14, tokenH=h // 14
    ).to(device)

    # Optimized optimizer and scheduler
    optimizer = optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Enhanced loss functions
    ce_loss = nn.CrossEntropyLoss()
    focal_loss = FocalLoss(alpha=1, gamma=2)
    scaler = torch.cuda.amp.GradScaler()

    # Training tracking
    best_iou = 0
    patience = 7
    patience_counter = 0

    print(f"\nStarting training...")
    print("="*50)

    # Training loop
    for epoch in range(epochs):
        # Training phase
        head.train()
        train_loss = 0
        train_ious = []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for imgs, masks in train_pbar:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            with torch.no_grad():
                tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits = head(tokens)
                outputs = F.interpolate(logits, size=masks.shape[1:], 
                                      mode="bilinear", align_corners=False)

                # Combined loss
                loss = (0.4 * ce_loss(outputs, masks) + 
                       0.4 * enhanced_dice_loss(outputs, masks) +
                       0.2 * focal_loss(outputs, masks))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            
            # Calculate training IoU
            with torch.no_grad():
                train_iou = compute_enhanced_iou(outputs, masks)
                train_ious.append(train_iou)
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{train_iou:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        # Validation phase
        head.eval()
        val_ious = []
        val_loss = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for imgs, masks in val_pbar:
                imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

                tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = head(tokens)
                outputs = F.interpolate(logits, size=masks.shape[1:], 
                                      mode="bilinear", align_corners=False)

                # Validation loss
                loss = (0.4 * ce_loss(outputs, masks) + 
                       0.4 * enhanced_dice_loss(outputs, masks) +
                       0.2 * focal_loss(outputs, masks))
                val_loss += loss.item()

                val_iou = compute_enhanced_iou(outputs, masks)
                val_ious.append(val_iou)
                
                val_pbar.set_postfix({'IoU': f'{val_iou:.4f}'})

        # Epoch results
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_iou = np.mean(train_ious)
        avg_val_iou = np.mean(val_ious)

        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, "BEST_hackathon_model.pth")
            print(f" New best model saved! IoU: {best_iou:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            break

        scheduler.step()
        
        # Memory cleanup
        torch.cuda.empty_cache()
        
        print("-" * 50)

    print(f"\n{'='*50}")
    print(" TRAINING COMPLETED!")
    print(f" Best Validation IoU: {best_iou:.4f}")
    print(f" Model saved as: BEST_hackathon_model.pth")
    print(f"{'='*50}")

    # Load best model for final evaluation
    checkpoint = torch.load("BEST_hackathon_model.pth")
    head.load_state_dict(checkpoint['model_state_dict'])
    print(f" Best model loaded from epoch {checkpoint['epoch']+1}")

    return head, best_iou

if __name__ == "__main__":
    model, final_iou = main()
