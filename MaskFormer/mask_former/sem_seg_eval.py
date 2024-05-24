# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
import csv
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
import torch.nn.functional as F
import pandas as pd
from detectron2.evaluation.evaluator import DatasetEvaluator


class SemSegiou(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        csv_data,
        csv_epoch
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        print("TEST")
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self.csv_data=csv_data
        self.csv_epoch=csv_epoch

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.thing_classes
        self._num_classes = len(meta.thing_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        num_test=len(self.input_file_to_gt_file)
        self.iou  = np.zeros((self._num_classes,num_test))
        self.dice = np.zeros((self._num_classes,num_test))
        self.metric_history={ "miou": [], "mdice": []}
        self.cci = np.zeros((1,self._num_classes))
        self.m=0

        self.name=os.path.join(self._output_dir, "log.csv")
        self.name_CSV=os.path.join(self._output_dir, "log_CSV.csv")


    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        


        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt = torch.Tensor(np.load(f)).type(torch.long)

            gt = F.one_hot(gt.type(torch.long),num_classes=self._num_classes).permute(2,0,1).contiguous()
            output=F.one_hot(output,num_classes=self._num_classes).permute(2,0,1).contiguous()
            
            cc=0
            for j in range(self._num_classes):
                self.iou[j,self.m], self.dice[j,self.m], ci = self.calculate_metric_percase(np.array(output[j]), np.array(gt[j]))
                cc += ci 
                self.cci[0,j] += ci

                    
            self.metric_history["miou"].append(np.sum(self.iou[:,self.m])/cc)
            self.metric_history["mdice"].append(np.sum(self.dice[:,self.m])/cc)
            self.m=self.m+1

        

    def writeCSV(self,data):
        with open(self.name, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            #writer.writeheader()
            writer.writerow(data)

    def binary_dc(self,result, reference):
        result = np.atleast_1d(result.astype(bool))
        reference = np.atleast_1d(reference.astype(bool))
        intersection = np.count_nonzero(result & reference)
        size_i1 = np.count_nonzero(result)
        size_i2 = np.count_nonzero(reference)
        try:
            dc = 2. * intersection / float(size_i1 + size_i2)
        except ZeroDivisionError:
            dc = 0.0
        return dc

    def binary_jc(self,result, reference):
        result = np.atleast_1d(result.astype(bool))
        reference = np.atleast_1d(reference.astype(bool))
        intersection = np.count_nonzero(result & reference)
        union = np.count_nonzero(result | reference) 
        jc = float(intersection) / float(union)
        return jc

    def calculate_metric_percase(self,pred, gt):
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        if (gt.sum()>0) :
            if (pred.sum()>0) : # Algun pixel verdadero y algun pixel predicho
                iou  = self.binary_jc(pred, gt)   #metric.binary.jc(pred, gt)
                dice = self.binary_dc(pred, gt)   #metric.binary.dc(pred, gt)
                #hd   = bmetric.binary_hd(pred, gt)   #metric.binary.hd(pred, gt)
                #hd95 = bmetric.binary_hd95(pred, gt) #metric.binary.hd95(pred, gt)
                return iou, dice, 1
            else:
                return 0, 0, 1 # Algun pixel verdadero y ningun pixel predicho --> Como incorporar hd y hd95 en este caso
        else:
            if (pred.sum()>0) :
                return 0, 0, 0 # Ningun pixel verdadero y algun pixel predicho 
            else:
                return 0, 0, 0 # Ningun pixel verdadero y ningun pixel predicho 


    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        res = {}
        res["iou_BG"]=np.sum(self.iou[0,:])/self.cci[0,0]
        res["iou_LV"]=np.sum(self.iou[1,:])/self.cci[0,1]
        res["iou_HV"]=np.sum(self.iou[2,:])/self.cci[0,2]
        res["iou_PV"]=np.sum(self.iou[3,:])/self.cci[0,3]
        res["iou_TM"]=np.sum(self.iou[4,:])/self.cci[0,4]
        res["dice_BG"]=np.sum(self.dice[0,:])/self.cci[0,0]
        res["dice_LV"]=np.sum(self.dice[1,:])/self.cci[0,1]
        res["dice_HV"]=np.sum(self.dice[2,:])/self.cci[0,2]
        res["dice_PV"]=np.sum(self.dice[3,:])/self.cci[0,3]
        res["dice_TM"]=np.sum(self.dice[4,:])/self.cci[0,4]
        res["iou_5c"]=sum(self.metric_history["miou"])/self.m
        res["dice_5c"]=sum(self.metric_history["mdice"])/self.m
        res["iou_4c"]=0.25*(np.sum(self.iou[1,:])/self.cci[0,1] + np.sum(self.iou[2,:])/self.cci[0,2]+np.sum(self.iou[3,:])/self.cci[0,3] + np.sum(self.iou[4,:])/self.cci[0,4])
        res["dice_4c"]=0.25*(np.sum(self.dice[1,:])/self.cci[0,1] + np.sum(self.dice[2,:])/self.cci[0,2]+np.sum(self.dice[3,:])/self.cci[0,3] + np.sum(self.dice[4,:])/self.cci[0,4])
        res["wLV"]=self.m - self.cci[0,1]
        res["wHV"]=self.m - self.cci[0,2]
        res["wPV"]=self.m - self.cci[0,3]
        res["wTM"]=self.m - self.cci[0,4]
        self.csv_data["epoch"].append(self.csv_epoch)
        self.csv_epoch=self.csv_epoch+1
        self.csv_data["iou_BG"].append(res["iou_BG"])
        self.csv_data["iou_LV"].append(res["iou_LV"])
        self.csv_data["iou_HV"].append(res["iou_HV"])
        self.csv_data["iou_PV"].append(res["iou_PV"])
        self.csv_data["iou_TM"].append(res["iou_TM"])
        self.csv_data["dice_BG"].append(res["dice_BG"])
        self.csv_data["dice_LV"].append(res["dice_LV"])
        self.csv_data["dice_HV"].append(res["dice_HV"])
        self.csv_data["dice_PV"].append(res["dice_PV"])
        self.csv_data["dice_TM"].append(res["dice_TM"])
        self.csv_data["iou_5c"].append(res["iou_5c"])
        self.csv_data["dice_5c"].append(res["dice_5c"])
        self.csv_data["iou_4c"].append(res["iou_4c"])
        self.csv_data["dice_4c"].append(res["dice_4c"])
        self.csv_data["wLV"].append(res["wLV"])
        self.csv_data["wHV"].append(res["wHV"])
        self.csv_data["wPV"].append(res["wPV"])
        self.csv_data["wTM"].append(res["wTM"])
        pd.DataFrame(self.csv_data).to_csv(self.name_CSV,index=True)
        print("---Metrics (Image)---")
        print("     (  BG,     LV,     HV,     PV,     TM)")
        print("IoU  : (%.4f, %.4f, %.4f, %.4f, %.4f)" % (res["iou_BG"], res["iou_LV"] , res["iou_HV"] , res["iou_PV"] , res["iou_TM"] ))
        print("Dice : (%.4f, %.4f, %.4f, %.4f, %.4f)" % (res["dice_BG"], res["dice_LV"], res["dice_HV"], res["dice_PV"], res["dice_TM"]))
        print("mIoU (5c)  : %.4f" % (res["iou_5c"]))
        print("mDice (5c) : %.4f" % (res["dice_5c"]))
        print("mIoU (4c)  : %.4f" % (res["iou_4c"]))
        print("mDice (4c) : %.4f" % (res["dice_4c"]))
        print("Sin LV - HV - PV - TM: (%.0f, %.0f, %.0f, %.0f)" %  (res["wLV"], res["wHV"], res["wPV"], res["wTM"]) )
        self.iou=[]
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            self.writeCSV(res)
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list

