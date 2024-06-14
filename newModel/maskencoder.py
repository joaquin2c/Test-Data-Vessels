import torch
import torch.nn as nn
from segment_anything.modeling.common import LayerNorm2d
class mask_downscaling(nn.Module):
    def __init__(
        self,
        mask_in_chans,
        embed_dim
    ):
        super().__init__()
        self.mask_encoder = nn.Sequential(
                nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
                LayerNorm2d(mask_in_chans // 4),
                nn.GELU(),
                nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
                LayerNorm2d(mask_in_chans),
                nn.GELU(),
                nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
            
        )


    def forward(self, mask):
        mask_embedding= self.mask_encoder(mask)
        return mask_embedding