from utils.metrics import calculate_metric_percase
import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np
def eval_model(model,device,test_dl,num_test):
    # Load the model
    #model = load_model(filepath,device)
    model.eval()
        
    iou  = np.zeros(num_test)
    dice = np.zeros(num_test)
    metric_history={ "miou": [], "mdice": []}
    m = 0
    cci=0
    patient_prev=0
    target_prev=torch.Tensor(np.zeros((1,1,256,256)))
    with torch.inference_mode():
        for img, target, _, patient, _ in tqdm(test_dl):
            target=target.cpu()
            if patient[0].numpy()!=patient_prev:
                target_prev=torch.Tensor(np.zeros((1,1,256,256)))
            output = model(img.to(device),target_prev.to(device))
            output = output.cpu() # (N,C,H,W)
            target=target[:,0,:,:].type(torch.long) # (N,1,H,W)
            pred = F.softmax(output,dim=1) # (N,C,H,W)
            pred[pred>0]=1
            target_prev=pred
            patient_prev=patient[0].numpy()
            
            iou[m], dice[m], ci= calculate_metric_percase(np.array(output), np.array(target))
            m = m + 1;
            cci += ci 
    print(cci)
    iou5c=sum(iou)/cci
    dice5c=sum(dice)/cci
    print("---Metrics (Image)---")
    print("mIoU  : %.4f" % iou5c)
    print("mDice : %.4f" % dice5c)
    """
    iou4c=0.25*(np.sum(iou[1,:])/cci[0,1] + np.sum(iou[2,:])/cci[0,2]+np.sum(iou[3,:])/cci[0,3] + np.sum(iou[4,:])/cci[0,4])
    dice4c=0.25*(np.sum(dice[1,:])/cci[0,1] + np.sum(dice[2,:])/cci[0,2]+np.sum(dice[3,:])/cci[0,3] + np.sum(dice[4,:])/cci[0,4])
    print("---Metrics (Image)---")
    print("     (  BG,     LV,     HV,     PV,     TM)")
    print("IoU  : (%.4f, %.4f, %.4f, %.4f, %.4f)" % (np.sum(iou[0,:])/cci[0,0], np.sum(iou[1,:])/cci[0,1], np.sum(iou[2,:])/cci[0,2], np.sum(iou[3,:])/cci[0,3], np.sum(iou[4,:])/cci[0,4]))
    print("Dice : (%.4f, %.4f, %.4f, %.4f, %.4f)" % (np.sum(dice[0,:])/cci[0,0], np.sum(dice[1,:])/cci[0,1], np.sum(dice[2,:])/cci[0,2], np.sum(dice[3,:])/cci[0,3], np.sum(dice[4,:])/cci[0,4]))
    print("mIoU (5c)  : %.4f" % iou5c)
    print("mDice (5c) : %.4f" % dice5c)
    print("mIoU (4c)  : %.4f" % iou4c)
    print("mDice (4c) : %.4f" % dice4c)
    print("Sin LV - HV - PV - TM: (%.0f, %.0f, %.0f, %.0f)" %  (m - cci[0,1], m - cci[0,2], m - cci[0,3], m - cci[0,4] ) )  
    """
    return iou5c,dice5c