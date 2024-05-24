from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import random
import os
import cv2
def get_lits_dicts(img_dir):
    files = os.listdir(os.path.join(img_dir,"images"))
    #print(len(files),files[0])
    dataset_dicts = []
    for idx, v in enumerate(files):
        record = {}
        filenameImg = os.path.join(img_dir,"images", v)
        filenameMsk = os.path.join(img_dir,"masks","0", v)
        height, width = cv2.imread(filenameImg).shape[:2]

        record["file_name"] = filenameImg
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        record["sem_seg_file_name "] = filenameMsk

        dataset_dicts.append(record)
    #print(dataset_dicts)
    return dataset_dicts

for d in ["", "val"]:
    DatasetCatalog.register("lits_" + d, lambda d=d: get_balloon_dicts("../../../Data/liver_only/test_model/full_data/" + d))
    MetadataCatalog.get("lits_" + d).set(thing_classes=["lits"])
"""
metadata = MetadataCatalog.get("lits_")

dataset_dicts = get_lits_dicts("../../../Data/liver_only/test_model")
for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    print("GOOD")
    out = visualizer.draw_dataset_dict(d)
    cv2.imwrite("xd.png",out.get_image()[:, :, ::-1])
"""
#get_lits_dicts("../../../Data/liver_only/test_model")