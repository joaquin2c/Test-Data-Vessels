import os

import numpy as np
"""
import pydicom
import pydicom_seg
"""
import pandas as pd
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
import torchvision.transforms as T

class DatasetColorectal(torch.utils.data.Dataset):
    def __init__(self, img_dir,mode, preproces=None, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        """
        list_pct=list(sorted(os.listdir(img_dir)))
        train_img_ids, val_img_ids = train_test_split(list_pct, test_size=0.2, random_state=41)
        if mode=='Train':
            self.img_ids = train_img_ids
        else :
            self.img_ids = val_img_ids if mode=='Val' else list_pct
        self.img_dir = img_dir
        self.transform = transform
        self.preproces = preproces

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        name_pct = self.img_ids[idx]
        
        temp0 = list(os.listdir(self.img_dir + name_pct))
        temp1 = list(sorted(os.listdir(self.img_dir + name_pct + '/' + temp0[0])))
        if temp1[0].find("Segmentation") != -1:
            path_seg = self.img_dir + name_pct + '/' + temp0[0] + '/' + temp1[0]
            path_img = self.img_dir + name_pct + '/' + temp0[0] + '/' + temp1[1]
        else: 
            path_seg = self.img_dir + name_pct + '/' + temp0[0] + '/' + temp1[1]
            path_img = self.img_dir + name_pct + '/' + temp0[0] + '/' + temp1[0]

        msk_dcm = pydicom.read_file(path_seg + '/' + '1-1.dcm')
        reader = pydicom_seg.SegmentReader()
        result = reader.read(msk_dcm)
        M = len(msk_dcm[0x00081115][0][0x0008114a].value)
        N = len(result.available_segments)
        strings = []
        for i in range(0,M):
            strings.append(msk_dcm[0x00081115][0][0x0008114a][i][0x00081155].value)
        
        list_imgs  = list(sorted(os.listdir(path_img)))
        
        image3D=[]
        mask3D=[]
        dim=0
        for name in list_imgs:
            image = pydicom.read_file(path_img + '/' + name)
            imgss = image.pixel_array #(512, 512) [-1024,3071] dtype('int16')
            img_min, img_max = -100, 400 #  (-17,201) # Estadistica de los datos (0.5%, 99.5%) pixeles que pertecen a las clases
            imgss = np.clip(imgss, img_min, img_max)
            image3D.append(imgss)
            # Generar mask
            mskss = np.zeros((512,512),dtype=np.uint8)
            if image[0x00080018].value in strings:
                indice = strings.index(image[0x00080018].value)
                #Temp0 = mskss
                #for i in range(5,N+1):
                #    Temp0 = Temp0 + 4*result.segment_data(i)[indice,:,:] # Tumor
                #mskss = mskss + (4*(Temp0>=4)).astype(np.uint8)  # (0,4)   
                Temp1 = mskss + 2*result.segment_data(3)[indice,:,:]  # Hepatic --0.2.4.6
                Temp2 = (2*(Temp1==2)).astype(np.uint8)
                mskss = mskss + Temp2 # (0,2)
                Temp1 = mskss + 3*result.segment_data(4)[indice,:,:]  # Portal --0.2.3.5.7 
                Temp2 = (3*(Temp1==3)).astype(np.uint8)
                mskss = mskss + Temp2 # (0,2,3)
                Temp1 = mskss + result.segment_data(1)[indice,:,:]   # Lives
                Temp2 = (1*(Temp1==1)).astype(np.uint8)
                mskss = mskss + Temp2 # (0,1,2,3)
            
            dim+=1
            mask3D.append(mskss)

        if self.preproces is not None:
            preproces = self.preproces(image=image3D, mask=mask3D)
            image3D = preproces['image']
            mask3D = preproces['mask']
        
        if self.transform is not None:
            augmented = self.transform(image=image3D, mask=mask3D)
            image3D = augmented['image']
            mask3D = augmented['mask']
        """
        img = image3D.astype('float32') / 255
        img = image3D.transpose(2, 0, 1)
        mask = mask3D.astype('float32') / 255
        mask = mask3D.transpose(2, 0, 1)
        """
        return torch.Tensor(image3D), torch.Tensor(mask3D), dim, {'img_id': name_pct}


class DatasetColorectal2D(torch.utils.data.Dataset):
    def __init__(self, csv_dir,img_path,mask_path, kfold, mode, crop=False, preprocessing=None, transform=None):
        """
        Args:
            img_dir: Image file directory.
            kfold: kfold to use in validation.
            mode: 'Train', 'Val' or 'Test'.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
            preprocess (Compose, optional): Compose preprocess. Defaults to None.
        """
        if mode=='Train':
            self.img_ids = self.get_train_folds(csv_dir,kfold)
        else:
            self.img_ids = self.get_val_fold(csv_dir,kfold,mode)
        self.mode = mode
        self.crop = crop
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.preprocessing = preprocessing
        self.transform_mask= T.Resize((256,256))
        self.transform_img = T.Resize((1024,1024))

        self.MEAN = [0.485, 0.456, 0.406] 
        self.STD  = [0.229, 0.224, 0.225]

    def __getitem__(self, idx):
        # load images ad masks
        img_data = self.img_ids[idx]
        img, mask, mask_prev = self.read_file(img_data) # (h,w,3) RGB - float32
        #img = (np.load(os.path.join(self.img_path, img_id[]))).astype(np.float32) # (h,w) RGB - float32
        img = img.astype(np.float32) - 1024.0
        img_min, img_max = -360, 440 #  (-17,201) # Estadistica de los datos (0.5%, 99.5%) pixeles que pertecen a las clases
        img = np.clip(img, img_min, img_max)
        # img_mean, img_std = 108.507, 64.996 # Estadistica de los datos -- Entrenas en modelo sin pesos pre-entrenados
        # img = (img - img_mean) / img_std 
        img = (img - img_min) / (img_max - img_min) # [0,1] -> Entre 0 y 1 para aumentación de datos Y/O normalizar respecto a pesos de Imagenet
        
        mask = np.expand_dims(mask.astype(np.float32), axis = -1) # (h,w,1)
        mask_prev = np.expand_dims(mask_prev.astype(np.float32), axis = -1) # (h,w,1)
        # apply augmentations
        """
        if self.transform is not None:
            augmented= self.transform(image=img, mask=mask_prev)
            img, mask_prev = augmented['image'], augmented['mask'] # (h,w,c), (h,w,c)
        
        # apply preprocessing
        """
        if self.preprocessing is not None:
            sample = self.preprocessing(image=img, mask=mask_prev) # [0,1]
            img, mask_prev = sample['image'], sample['mask']    # (h,w,c), (h,w,c)
        else:
            img = ((img) - self.MEAN ) / self.STD
        
        # (h,w,c) -> (c,h,w)  
        img  = np.transpose(img,  (2,0,1)).astype(np.float32) #(c,h,w)
        mask = np.transpose(mask, (2,0,1)).astype(np.float32) #(c,h,w)
        mask_prev = np.transpose(mask_prev, (2,0,1)).astype(np.float32) #(c,h,w)
        
        img= self.transform_img(torch.Tensor(img))
        target = self.transform_mask(torch.Tensor(mask)) # [0,1,2,3,4] (1-liver,2-hepatic,3-portal,4-tumor)
        target_prev = self.transform_mask(torch.Tensor(mask_prev)) # [0,1,2,3,4] (1-liver,2-hepatic,3-portal,4-tumor) t_slice
        return img, target, target_prev, img_data["patient"], img_data["t_slice"]

    def read_file(self,file):
        x_min=None
        x_max=None
        y_min=None
        y_max=None
        
        if self.crop==True:
            x_min=file["xmin"]
            x_max=file["xmax"]
            y_min=file["ymin"]
            y_max=file["ymax"]
            
        prev_slide=np.load(os.path.join(self.img_path, file["t-1"]))
        slide=np.load(os.path.join(self.img_path, file["t"]))
        next_slide=np.load(os.path.join(self.img_path, file["t+1"]))
        mask = np.load(os.path.join(self.mask_path, file["t"])) # (h,w) - uint8
        mask_prev = np.load(os.path.join(self.mask_path, file["t-1"])) # (h,w) - uint8
        return np.stack([prev_slide,slide,next_slide], axis=2)[y_min:y_max,x_min:x_max,:], mask[y_min:y_max,x_min:x_max], mask_prev[y_min:y_max,x_min:x_max]
    
    def __len__(self):
        return len(self.img_ids)

    def get_val_fold(self,folder,kfold,mode):
        assert mode=="Val" or mode=="Test", "'Mode' must be 'Train', 'Val' or 'Test'"
        if mode=="Val":
            val_fold=pd.read_csv(os.path.join(folder,f'fold_{kfold}.csv'))
        else:
            val_fold=pd.read_csv(os.path.join(folder,f'Dataset.csv'))
        #return val_fold, val_uniques
        return val_fold.to_dict('records')
        
    def get_train_folds(self,folder,kfold,numfolds=5):
        train_folds=np.delete(np.arange(1,numfolds+1), kfold-1)
        train_folds=[pd.read_csv(os.path.join(folder,f'fold_{i}.csv')) for i in train_folds]           
        train_folds = pd.concat(train_folds)
        return train_folds.to_dict('records')