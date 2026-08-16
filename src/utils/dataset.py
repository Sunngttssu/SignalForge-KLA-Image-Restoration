import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from utils.losses_and_sim import apply_random_degradation

class SignalForgeDataset(Dataset):
    def __init__(self, kla_dir=None, external_dir=None, is_train=True, val_split=0.1, patch_size=128):
        self.samples = []
        self.is_train = is_train
        self.patch_size = patch_size # Strictly set to 128 to match smallest KLA LR images
        
        if kla_dir and os.path.exists(os.path.join(kla_dir, "GT")):
            gt_dir = os.path.join(kla_dir, "GT")
            lr_dir = os.path.join(kla_dir, "NoisyLR")
            all_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])
            
            random.seed(42)
            shuffled_files = all_files.copy()
            random.shuffle(shuffled_files)
            
            split_idx = int(len(shuffled_files) * (1.0 - val_split))
            selected_files = shuffled_files[:split_idx] if is_train else shuffled_files[split_idx:]
            
            for f in selected_files:
                self.samples.append({'type': 'kla', 'gt': os.path.join(gt_dir, f), 'lr': os.path.join(lr_dir, f)})
        
        if is_train and external_dir and os.path.exists(external_dir):
            for f in os.listdir(external_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append({'type': 'external', 'gt': os.path.join(external_dir, f)})

    def __len__(self):
        return len(self.samples)

    def _augment(self, lr, gt):
        if random.random() > 0.5:
            lr, gt = torch.flip(lr, dims=[2]), torch.flip(gt, dims=[2])
        if random.random() > 0.5:
            lr, gt = torch.flip(lr, dims=[1]), torch.flip(gt, dims=[1])
        k = random.randint(0, 3)
        if k > 0:
            lr, gt = torch.rot90(lr, k, dims=[1, 2]), torch.rot90(gt, k, dims=[1, 2])
        return lr, gt

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if sample['type'] == 'kla':
            gt = torch.from_numpy(np.load(sample['gt'])).unsqueeze(0).float()
            lr = torch.from_numpy(np.load(sample['lr'])).unsqueeze(0).float()
            
            if self.is_train:
                # Guarantee identical 128x128 patch sizes for batching
                lr_h, lr_w = lr.shape[1], lr.shape[2]
                ps = self.patch_size
                top = random.randint(0, max(0, lr_h - ps))
                left = random.randint(0, max(0, lr_w - ps))
                
                lr = lr[:, top:top + ps, left:left + ps]
                gt = gt[:, top * 2:(top + ps) * 2, left * 2:(left + ps) * 2]
                lr, gt = self._augment(lr, gt)
                
        else: 
            img = Image.open(sample['gt']).convert('L')
            gt_full = transforms.ToTensor()(img)
            
            gt_size = self.patch_size * 2 # 256x256 GT patch
            if gt_full.shape[1] >= gt_size and gt_full.shape[2] >= gt_size:
                i, j, h, w = transforms.RandomCrop.get_params(gt_full, output_size=(gt_size, gt_size))
                gt = TF.crop(gt_full, i, j, h, w)
            else:
                gt = TF.resize(gt_full, [gt_size, gt_size], interpolation=transforms.InterpolationMode.BICUBIC)
            
            # Physics simulator scales down 2x, resulting in exactly a 128x128 LR patch
            lr = apply_random_degradation(gt) 
            lr, gt = self._augment(lr, gt)
            
        return lr, gt
