import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class newModel(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_encoder,
        vit,
        decoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_encoder = mask_encoder
        self.vit = vit
        self.decoder = decoder
        # freeze prompt encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, mask=None):
        # do not compute gradients for image encoder
        with torch.no_grad():       
            image_embedding = self.image_encoder(image)       
            
        mask_embedding= self.mask_encoder(mask)
        
        src=image_embedding+mask_embedding
        
        embeddings=self.vit(src)
        
        low_res_masks, _=self.decoder(embeddings,multimask_output=False)
        
        """
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        """
        return low_res_masks