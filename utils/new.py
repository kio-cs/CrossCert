import numpy as np
import torch
from collections import defaultdict
torch.set_printoptions(profile='full')


def majority_of_mask_single(prediction_map):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    # first-round masking
    pred_one_mask = np.diag(prediction_map)
    pred, cnt = np.unique(pred_one_mask, return_counts=True)
    sorted_idx = np.argsort(cnt)
    majority_pred = pred[sorted_idx][-1]

    return majority_pred


def majority_of_drs_single(prediction_map):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    # first-round masking
    pred, cnt = np.unique(prediction_map, return_counts=True)
    sorted_idx = np.argsort(cnt)
    majority_pred = pred[sorted_idx][-1]

    return majority_pred


def disagree_second_mask(prediction_map_old):
    # second-round masking
    pred_mutant_list, cnt_mutant = np.unique(prediction_map_old, return_counts=True)
    # # get majority prediction and disagreer prediction
    # sorted_idx = np.argsort(cnt_mutant)
    # majority_pred = pred_mutant_list[sorted_idx][-1]

    return pred_mutant_list, cnt_mutant

def pc_malicious_label(prediction_map_old, prediction_label):
    malicious_label_list = []
    pred_one_mask_mutant = np.diag(prediction_map_old)
    pred_one_mask_mutant_list, cnt_pred_one_mask_mutant = np.unique(pred_one_mask_mutant, return_counts=True)
    pred_two_mask_mutant_list, cnt_pred_two_mask_mutant = np.unique(prediction_map_old, return_counts=True)

    if len(pred_one_mask_mutant_list) == 1:  # agreement in the first-round masking
        for pred_two_mask_mutant in pred_two_mask_mutant_list:
            if not pred_two_mask_mutant == prediction_label:
                malicious_label_list.append(pred_two_mask_mutant)
        return malicious_label_list  # success
    else:
        for pred_one_mask_mutant in pred_one_mask_mutant_list:
            if not pred_one_mask_mutant == prediction_label:
                malicious_label_list.append(pred_one_mask_mutant)
        for pred_two_mask_mutant in pred_two_mask_mutant_list:
            if not pred_two_mask_mutant == prediction_label:
                malicious_label_list.append(pred_two_mask_mutant)
        return malicious_label_list  # success


def pc_malicious_label_with_location(prediction_map_old, prediction_label, num_mask=6):
    malicious_label_dict = {}
    for idx_one in range(len(prediction_map_old)):
        for idx_two in range(len(prediction_map_old)):
            label_for_mutant = prediction_map_old[idx_one][idx_two]
            if not label_for_mutant == prediction_label:
                key_name = idx_one * num_mask * num_mask + idx_two
                malicious_label_dict[key_name] = label_for_mutant
    return malicious_label_dict


def certified_drs(prediction_map_drs, ablation_size, patch_size):
    delta = ablation_size + patch_size - 1
    pred_list, cnt = np.unique(prediction_map_drs, return_counts=True)
    sorted_idx = np.argsort(cnt)
    majority_pred = pred_list[sorted_idx][-1]
    sorted_value = np.sort(cnt)
    # get majority prediction and disagreer prediction
    if len(sorted_value) > 1:
        gap = sorted_value[-1] - sorted_value[-2]
    else:
        gap = sorted_value[-1]
    if gap > 2 * delta:
        return majority_pred, True
    else:
        return majority_pred, False


def drs_malicious_label_with_location_fast_jiaozhun(idx_one, prediction_map_drs, ablation_size, patch_size, label):
    malicious_label_dict = []
    # pred_list, cnt = np.unique(prediction_map_drs, return_counts=True)
    # sorted_idx = np.argsort(cnt)
    # majority_pred_now = pred_list[sorted_idx][-1]
    delta = ablation_size + patch_size - 1
    # jiaozhun
    idx_one=idx_one-ablation_size
    prediction_map_drs_copy = prediction_map_drs.copy()
    if idx_one < 0:
        prediction_map_drs_copy[len(prediction_map_drs_copy) + idx_one:len(prediction_map_drs_copy)] = label
        prediction_map_drs_copy[0:idx_one + delta] = label
        if idx_one + delta < 0:
            print("wrong")
    elif idx_one + delta > len(prediction_map_drs_copy):
        # prediction_map_drs_copy[idx_one:idx_one + delta] = label #problem
        prediction_map_drs_copy[idx_one:len(prediction_map_drs_copy)] = label
        prediction_map_drs_copy[0:idx_one + delta - len(prediction_map_drs_copy)] = label
    else:
        prediction_map_drs_copy[idx_one:idx_one + delta] = label
    pred_list, cnt = np.unique(prediction_map_drs_copy, return_counts=True)
    sorted_idx = np.argsort(cnt)
    majority_pred = pred_list[sorted_idx][-1]
    return majority_pred

#column in place 3
def suspect_column_list_cal_fix(mask_list):
    maskfree_list_malicious_column_list = []
    for mask_1 in mask_list:
        for mask_2 in mask_list:
            mask_malicious_column_list = []
            for idx in range(mask_1.shape[3]):
                column_1 = ~mask_1[:, :, :,idx]
                column_2 = ~mask_2[:, :, :, idx]
                if column_1.any() or column_2.any():
                    mask_malicious_column_list.append(idx)
            maskfree_list_malicious_column_list.append(mask_malicious_column_list)
    return maskfree_list_malicious_column_list

def certified_with_location(malicious_label_dict_pc_with_location, suspect_column_list_pc, patch_size,
                            prediction_map_drs, ablation_size):
    pass_list = defaultdict(set)
    for idx in malicious_label_dict_pc_with_location:
        suspect_column_l_pc = suspect_column_list_pc[idx]
        malicious_label = malicious_label_dict_pc_with_location.get(idx)
        for suspect_column_pc in suspect_column_l_pc:
            # if suspect_column_pc + patch_size - 1 in suspect_column_l_pc:
            list_of_attack_position_drs=[]
            first_idx=suspect_column_pc-patch_size
            for i in range(patch_size):
                new_idx=first_idx+i+1
                if new_idx < 0:
                    new_idx = patch_size + new_idx
                list_of_attack_position_drs.append(new_idx)
            for idx in list_of_attack_position_drs:
                if idx not in pass_list[malicious_label]:
                    output_label = drs_malicious_label_with_location_fast_jiaozhun(idx, prediction_map_drs,
                                                                                   ablation_size, patch_size,
                                                                                   malicious_label)
                    if output_label == malicious_label:
                        return False
                    else:
                        try:
                            pass_list[malicious_label].add(idx)
                        except KeyError:
                            pass_list[malicious_label] = {idx}
            # else:
            #     continue

    return True

def double_masking_precomputed_with_case_num(prediction_map):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    # first-round masking
    pred_one_mask = np.diag(prediction_map)
    pred, cnt = np.unique(pred_one_mask, return_counts=True)

    if len(pred) == 1:  # unanimous agreement in the first-round masking
        return pred[0], 1  # Case I: agreed prediction

    # get majority prediction and disagreer prediction
    sorted_idx = np.argsort(cnt)
    majority_pred = pred[sorted_idx][-1]
    disagreer_pred = pred[sorted_idx][:-1]

    # second-round masking
    # get index list of the disagreer mask
    tmp = np.zeros_like(pred_one_mask, dtype=bool)
    for dis in disagreer_pred:
        tmp = np.logical_or(tmp, pred_one_mask == dis)
    disagreer_pred_mask_idx = np.where(tmp)[0]

    for i in disagreer_pred_mask_idx:
        dis = pred_one_mask[i]
        # check all two-mask predictions
        tmp = prediction_map[i] == dis
        if np.all(tmp):
            return dis, 2  # Case II: disagreer prediction

    return majority_pred, 3  # Case III: majority prediction

def double_masking_precomputed_with_case_num_modify(prediction_map):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    # first-round masking
    pred_one_mask = np.diag(prediction_map)
    pred, cnt = np.unique(pred_one_mask, return_counts=True)

    if len(pred) == 1:  # unanimous agreement in the first-round masking
        return pred[0], 1  # Case I: agreed prediction

    # get majority prediction and disagreer prediction
    sorted_idx = np.argsort(cnt)
    majority_pred = pred[sorted_idx][-1]
    disagreer_pred = pred[sorted_idx][:-1]

    # second-round masking
    # get index list of the disagreer mask
    tmp = np.zeros_like(pred_one_mask, dtype=bool)
    for dis in disagreer_pred:
        tmp = np.logical_or(tmp, pred_one_mask == dis)
    disagreer_pred_mask_idx = np.where(tmp)[0]

    for i in range(len(pred_one_mask)):
        dis = pred_one_mask[i]
        # check all two-mask predictions
        tmp = prediction_map[i] == dis
        if np.all(tmp):
            return dis, 2  # Case II: disagreer prediction

    return majority_pred, 3  # Case III: majority prediction
