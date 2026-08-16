import torch
import torch.nn as nn

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = nn.LayerNorm(c)
        self.conv1 = nn.Conv2d(c, c * 2, 1)
        self.conv2 = nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(c, c, 1)
        
        self.norm2 = nn.LayerNorm(c)
        self.ffn1 = nn.Conv2d(c, c * 2, 1)
        self.ffn2 = nn.Conv2d(c, c, 1)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        out = self.norm1(x.view(b, c, -1).transpose(1, 2)).transpose(1, 2).view(b, c, h, w)
        out = self.conv3(self.sg(self.conv2(self.conv1(out))))
        x = x + out
        
        out = self.norm2(x.view(b, c, -1).transpose(1, 2)).transpose(1, 2).view(b, c, h, w)
        out = self.ffn2(self.sg(self.ffn1(out)))
        return identity + out

class PromptFSNAF(nn.Module):
    def __init__(self, in_c=1, out_c=1, width=32, blocks=8):
        super().__init__()
        self.intro = nn.Conv2d(in_c, width, 3, padding=1)
        
        self.pgm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width, width, 1),
            nn.Sigmoid()
        )
        
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(blocks)])
        
        self.upconv = nn.Conv2d(width, width * 4, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.outro = nn.Conv2d(width, out_c, 3, padding=1)

    def forward(self, x):
        feat = self.intro(x)
        prompt = self.pgm(feat)
        feat = feat * prompt 
        
        feat = self.body(feat)
        
        feat = self.pixel_shuffle(self.upconv(feat))
        out = self.outro(feat)
        
        base = torch.nn.functional.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        return out + base
