import numpy as np
import torch


def one_masking_statistic(prediction_map):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    # first-round masking
    total=36
    pred_one_mask = np.diag(prediction_map)
    pred,cnt = np.unique(pred_one_mask,return_counts=True)
    if len(pred) == 1: # unanimous agreement in the first-round masking
        return 0,pred # Case I: agreed prediction
    else:
        sorted_cnt = np.sort(cnt)
        sorted_idx = np.argsort(cnt)
        majority_pred = pred[sorted_idx][-1]
        return total-sorted_cnt[-1],majority_pred

def double_masking_detection_for_one_mask_agree(pred_one_mask,prediction_map,bear):
    for i in range(len(pred_one_mask)):
        first_label = pred_one_mask[i]
        # check all two-mask predictions
        tmp = prediction_map[i] != first_label
        disagreer_second_count = np.sum(tmp == 1)
        if disagreer_second_count > bear:
            return -4  #-4 for cert and warn
        else:
            continue
    return -3  # -3 for cert and not warn





def double_masking_detection(prediction_map,bear):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    # first-round masking
    pred_one_mask = np.diag(prediction_map)
    pred,cnt = np.unique(pred_one_mask,return_counts=True)

    if len(pred) == 1: # unanimous agreement in the first-round masking
        return pred[0],double_masking_detection_for_one_mask_agree(pred_one_mask,prediction_map,bear)

    # get majority prediction and disagreer prediction
    sorted_idx = np.argsort(cnt)
    majority_pred = pred[sorted_idx][-1]
    disagreer_pred = pred[sorted_idx][:-1]

    # second-round masking
    # get index list of the disagreer mask
    tmp = np.zeros_like(pred_one_mask,dtype=bool)
    for dis in disagreer_pred:
        tmp = np.logical_or(tmp,pred_one_mask==dis)
    disagreer_pred_mask_idx = np.where(tmp)[0]

    for i in disagreer_pred_mask_idx:
        dis = pred_one_mask[i]
        # check all two-mask predictions
        tmp = prediction_map[i]!=dis
        disagreer_second_count=np.sum(tmp==1)
        if disagreer_second_count>bear:
            continue
        else:
            return majority_pred, -2 #-2 for not cert and warn

    return majority_pred,-1 #-1 for cert and warn


def double_masking_detection_nolemma1(prediction_map,bear):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    # first-round masking
    pred_one_mask = np.diag(prediction_map)
    pred,cnt = np.unique(pred_one_mask,return_counts=True)

    if len(pred) == 1: # unanimous agreement in the first-round masking
        return pred[0],double_masking_detection_for_one_mask_agree(pred_one_mask,prediction_map,bear) # Case I: agreed prediction

    # get majority prediction and disagreer prediction
    sorted_idx = np.argsort(cnt)
    majority_pred = pred[sorted_idx][-1]
    disagreer_pred = pred[sorted_idx][:-1]

    # second-round masking
    # get index list of the disagreer mask
    tmp = np.zeros_like(pred_one_mask,dtype=bool)
    for dis in disagreer_pred:
        tmp = np.logical_or(tmp,pred_one_mask==dis)
    disagreer_pred_mask_idx = np.where(tmp)[0]

    for i in range(len(tmp)):
        dis = pred_one_mask[i]
        # check all two-mask predictions
        tmp = prediction_map[i]!=dis
        disagreer_second_count=np.sum(tmp==1)
        if disagreer_second_count>bear:
            continue
        else:
            return majority_pred, -2

    return majority_pred,-1 # Case III: majority prediction