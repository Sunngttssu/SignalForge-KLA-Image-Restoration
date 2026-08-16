import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('weight_x', k)
        self.register_buffer('weight_y', k.transpose(2, 3))

    def forward(self, pred, target):
        pred_x = F.conv2d(pred, self.weight_x, padding=1)
        pred_y = F.conv2d(pred, self.weight_y, padding=1)
        tgt_x = F.conv2d(target, self.weight_x, padding=1)
        tgt_y = F.conv2d(target, self.weight_y, padding=1)
        return torch.mean(torch.abs(pred_x - tgt_x) + torch.abs(pred_y - tgt_y))

def apply_random_degradation(image_tensor):
    degradations = ['speckle', 'gaussian', 'downsample']
    random.shuffle(degradations) 
    
    out = image_tensor.clone()
    for deg in degradations:
        if deg == 'speckle':
            speckle = torch.randn_like(out) * random.uniform(0.05, 0.2)
            out = out * (1 + speckle)
        elif deg == 'gaussian':
            gaussian = torch.randn_like(out) * random.uniform(0.01, 0.1)
            out = out + gaussian
        elif deg == 'downsample':
            out = F.interpolate(out.unsqueeze(0), scale_factor=0.5, mode='bicubic', align_corners=False).squeeze(0)
            
    return out
