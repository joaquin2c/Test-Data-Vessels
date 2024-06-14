from functools import partial
from pathlib import Path
import urllib.request
import torch

from maskencoder import mask_downscaling
from VitMod import ViTEncoder
from model import newModel
from segment_anything.modeling import (
    ImageEncoderViT,
    MaskDecoder,
    TwoWayTransformer,
)


def build_model_vit_b(checkpoint=None,img_size=512,in_chans=3):
    return _build_model(
        img_size=img_size,
        in_chans=in_chans,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


model_registry = {
    "default": build_model_vit_b,
    "vit_b": build_model_vit_b,
}


def _build_model(
    img_size,
    in_chans,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    vit_patch_size = 16
    mask_in_chans=16
    image_embedding_size = img_size // vit_patch_size
    model = newModel(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=img_size,
            in_chans=in_chans,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        mask_encoder=mask_downscaling(
            mask_in_chans=mask_in_chans,
            embed_dim=prompt_embed_dim,
        ),
        vit=ViTEncoder(
            depth=3,
            embed_dim=prompt_embed_dim,
            img_size=img_size,
            patch_size=vit_patch_size,
            num_heads=2,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            use_rel_pos=True,
            global_attn_indexes=[2],
            rel_pos_zero_init=True,
            window_size=14,
            out_chans=prompt_embed_dim
        ),
        decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
    )
    
    model.eval()
    checkpoint = Path(checkpoint)
    if checkpoint.name == "sam_vit_b_01ec64.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_b_01ec64.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == "y":
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-B checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict,strict=False)
        print("Image encoder loaded")
    return model
