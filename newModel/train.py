import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import albumentations as A
from albumentations.core.composition import Compose
from collections import OrderedDict
from eval_step import eval_model
from Dataset import DatasetColorectal, DatasetColorectal2D
from build_model import model_registry

# set seeds
torch.manual_seed(2024)
torch.cuda.empty_cache()


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--csv_path",
    type=str,
    default="../../Data/colorectal2d/resize/kfolds",
    help="path to training image files; two subfolders: mask/0 and imgs",
)
parser.add_argument(
    "-checkpoint", type=str, default="SAM/sam_vit_b_01ec64.pth"
)
#

parser.add_argument('-img_path', type=str, default="../../Data/colorectal2d/resize/images")
parser.add_argument('-mask_path', type=str, default="../../Data/colorectal2d/resize/masks")
parser.add_argument('-kfold', type=int, default=1)
parser.add_argument('-classes', type=int, default=5)
parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument("-work_dir", type=str, default="./work_dir")
parser.add_argument("-name", type=str, default="NewModel")

# train
parser.add_argument("-num_epochs", type=int, default=1000)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=4)
parser.add_argument("-img_size", type=int, default=1024)
parser.add_argument("-in_channels", type=int, default=3)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
args = parser.parse_args()

# %% set up model for training
# device = args.device
model_save_path = os.path.join(args.work_dir, args.name)
device = torch.device(args.device)
# %% set up model



def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, os.path.join(model_save_path, os.path.basename(__file__))
    )

    model = model_registry["default"](checkpoint="SAM/sam_vit_b_01ec64.pth",img_size=1024,in_chans=args.in_channels)
    model = model.to(device)
    model.train()
    print(
        "Number of total parameters: ",
        sum(p.numel() for p in model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )  # 93729252

    img_mask_encdec_params = list(model.mask_encoder.parameters()) + list(model.vit.parameters()) + list(model.decoder.parameters())
    
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    best_loss = 1e10
    best_iou=0
    best_dice=0

    """
    transform_train_lits = Compose([
	    A.Resize(args.img_size, args.img_size),
	    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
	    A.HorizontalFlip(p=0.5),
	    A.Rotate (limit=10, interpolation=1, border_mode=1, value=None, crop_border=False, always_apply=False, p=0.5),
	    #trns.Normalize(),
	])

    transform_val_lits = Compose([
	    A.Resize(args.img_size, args.img_size),
	    #trns.Normalize(),
	])
    """
    
    colorectal_train_dataset=DatasetColorectal2D(args.csv_path,args.img_path,args.mask_path, args.kfold, "Train")
    colorectal_val_dataset=DatasetColorectal2D(args.csv_path,args.img_path,args.mask_path, args.kfold, "Val")

    nice_train_loader = DataLoader(colorectal_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    nice_val_loader = DataLoader(colorectal_val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("Number of training samples: ", len(colorectal_train_dataset))
    print("Number of validation samples: ", len(colorectal_val_dataset))

    log = OrderedDict([
        ('epoch', []),
        ('loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])


    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

    for epoch in range(start_epoch, num_epochs):
    	model.train()
    	epoch_loss = 0
    	print(f"Epoch: {epoch}")
    	for step, (image, mask,mask_prev,_,_) in enumerate(tqdm(nice_train_loader)):
            optimizer.zero_grad()
            image, mask, mask_prev = image.to(device), mask.to(device), mask_prev.to(device)
            model_pred = model(image, mask_prev)
            loss = seg_loss(model_pred, mask) + ce_loss(model_pred, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
    	iou,dice=eval_model(model,device,nice_val_loader,len(colorectal_val_dataset))

    	epoch_loss /= step
    	log["epoch"].append(epoch)
    	log["loss"].append(epoch_loss)
    	log["val_iou"].append(iou)
    	log["val_dice"].append(dice)
    	pd.DataFrame(log).to_csv(os.path.join(model_save_path, 'log.csv'), index=False)
    	"""
    	print(
    	    f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}, IoU: {epoch_iou}, DICE: {epoch_dice}'
    	)
    	"""
        ## save the latest model
    	checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "iou": iou,
            "dice": dice,
            "loss": epoch_loss,
        }
    	torch.save(checkpoint, os.path.join(model_save_path, "medsam_model_latest.pth"))
        
        ## save the best models
    	if best_iou < iou:
            best_iou = iou
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "iou": iou,
                "dice": dice,
                "loss": epoch_loss,
            }
            torch.save(checkpoint, os.path.join(model_save_path, "medsam_model_best_iou.pth"))

    	if best_dice < dice:
            best_dice = dice
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "iou": iou,
                "dice": dice,
                "loss": epoch_loss,
            }
            torch.save(checkpoint, os.path.join(model_save_path, "medsam_model_best_dice.pth"))

if __name__ == "__main__":
    main()
