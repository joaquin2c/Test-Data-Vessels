# Data Vessels

## Dataset
1. Ir a la pagina de [Colorectal](https://www.cancerimagingarchive.net/collection/colorectal-liver-metastases/)
2. Descargar `NBIA Data Retriever`
3. Descargar `Images, Segmentations (10.91GB)` de la pagina, debería abrirse el programa antes instalado.

## Procesamiento Dataset
1. Una vez descargado los archivo tipo DICOM, ejecute el notebook `NII_LColorectal_Dataset` y complete todas sus casillas.
   Con esto se obtendra en archivos npy las slides de las ct en tres formatos: con las 5 clases, con solo higado y solo los data vessels
2. Con el notebook `GenerationDatasets` se generalizar los nombres de los archivos en la misma carpeta o en una nueva, y se crean los
   troch Datasets para los diferentes modelos.

## Modelos


##Maskformer
1. Seguir instrucciones de instalación del [Readme de Maskformer](MaskFormer/README.md).
2. Entrenar con el comando:
```
python train_net.py --config-file configs/ade20k-150/swin/maskformer_swin_small_bs16_160k.yaml  OUTPUT_DIR /OUTPUT_DIR   MODEL.DEVICE "cuda:0"
```
