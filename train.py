import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.prompt_fs_naf import PromptFSNAF
from utils.dataset import SignalForgeDataset
from utils.losses_and_sim import CharbonnierLoss, EdgeLoss
import lpips

def calculate_psnr(pred, gt, eps=1e-8):
    mse = torch.mean((pred - gt) ** 2, dim=[1, 2, 3])
    return torch.mean(10.0 * torch.log10(1.0 / (mse + eps))).item()

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    KLA_DIR = "/kaggle/input/datasets/shivaang78987/kla-hackathon-2026-data/train/train" 
    EXT_DIR = "/kaggle/input/datasets/soumikrakshit/div2k-high-resolution-images/DIV2K_train_HR/DIV2K_train_HR" 
    
    # Train dataset patched to 128x128. Validation processes full images.
    train_dataset = SignalForgeDataset(kla_dir=KLA_DIR, external_dir=EXT_DIR, is_train=True, patch_size=128)
    val_dataset = SignalForgeDataset(kla_dir=KLA_DIR, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    # Validation MUST be batch_size=1 to handle mixed 256x256 and 128x128 input images safely
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    model = PromptFSNAF(width=32, blocks=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # CosineAnnealingWarmRestarts to escape local minima, resetting every 20 epochs
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-6)
    
    # Updated GradScaler syntax
    scaler = torch.amp.GradScaler('cuda')
    
    criterion_char = CharbonnierLoss().to(device)
    criterion_edge = EdgeLoss().to(device)
    criterion_lpips = lpips.LPIPS(net='vgg').to(device)
    
    history_file = "outputs/train_history.csv"
    with open(history_file, mode='w', newline='') as f:
        csv.writer(f).writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_PSNR"])
        
    best_psnr = -float('inf')
    epochs, warmup = 60, 3

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        if epoch <= warmup:
            for pg in optimizer.param_groups: 
                pg['lr'] = 3e-4 * (epoch / warmup)
        
        for lr_img, gt_img in train_loader:
            lr_img, gt_img = lr_img.to(device, non_blocking=True), gt_img.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                pred = model(lr_img)
                
                # Expand 1-channel grayscale to 3-channel RGB for VGG/LPIPS compatibility
                pred_rgb = pred.repeat(1, 3, 1, 1)
                gt_rgb = gt_img.repeat(1, 3, 1, 1)
                
                loss_char = criterion_char(pred, gt_img)
                loss_edge = 0.15 * criterion_edge(pred, gt_img)
                loss_perceptual = 0.1 * criterion_lpips(pred_rgb, gt_rgb).mean()
                
                loss = loss_char + loss_edge + loss_perceptual
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping to prevent exploding gradients from out-of-bounds speckle
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            
        if epoch > warmup: 
            scheduler.step()
            
        train_loss = running_loss / len(train_loader)
        
        model.eval()
        v_loss_sum, v_psnr_sum = 0.0, 0.0
        with torch.no_grad():
            for lr_img, gt_img in val_loader:
                lr_img, gt_img = lr_img.to(device, non_blocking=True), gt_img.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    pred = model(lr_img)
                    
                    pred_rgb = pred.repeat(1, 3, 1, 1)
                    gt_rgb = gt_img.repeat(1, 3, 1, 1)
                    
                    v_loss_char = criterion_char(pred, gt_img)
                    v_loss_edge = 0.15 * criterion_edge(pred, gt_img)
                    v_loss_perceptual = 0.1 * criterion_lpips(pred_rgb, gt_rgb).mean()
                    
                    v_loss_sum += (v_loss_char + v_loss_edge + v_loss_perceptual).item()
                v_psnr_sum += calculate_psnr(pred, gt_img)
                
        val_loss, val_psnr = v_loss_sum / len(val_loader), v_psnr_sum / len(val_loader)
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), "weights/prompt_fs_naf.pt")
            
        print(f"Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | PSNR: {val_psnr:.2f}dB")
        with open(history_file, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_psnr])

if __name__ == "__main__":
    train()
