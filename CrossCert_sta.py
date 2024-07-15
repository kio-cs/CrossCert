
import torch.backends.cudnn as cudnn

import numpy as np
import os
import argparse
import joblib
from time import time

from utils.new import  certified_drs, double_masking_precomputed_with_case_num, pc_malicious_label_with_location, certified_with_location, suspect_column_list_cal_fix, double_masking_precomputed_with_case_num_modify
from utils.setup import get_model, get_data_loader
from utils.defense import gen_mask_set, certify_precomputed

#
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default='', type=str, help="directory of checkpoints")
parser.add_argument('--data_dir', default='', type=str, help="directory of data")
parser.add_argument('--dataset', default='cifar', type=str,
                    choices=('imagenette', 'imagenet', 'cifar', 'cifar100', 'svhn', 'flower102'), help="dataset")
parser.add_argument("--pc_model", default='vit_base_patch16_224_cutout2_128', type=str, help="model name")
parser.add_argument("--drs_model", default='vit_base_patch16_224', type=str, help="model name")
# parser.add_argument("--pc_model", default='resnetv2_50x1_bit_distilled_cutout2_128', type=str, help="model name")
# parser.add_argument("--drs_model", default='resnetv2_50x1_bit_distilled', type=str, help="model name")

parser.add_argument("--num_img", default=-1, type=int,
                    help="number of randomly selected images for this experiment (-1: using the all images)")
parser.add_argument("--mask_stride", default=-1, type=int, help="mask stride s (square patch; conflict with num_mask)")
parser.add_argument("--num_mask", default=6, type=int,
                    help="number of mask in one dimension (square patch; conflict with mask_stride)")
parser.add_argument("--patch_size", default=35, type=int, help="size of the adversarial patch (square patch)")
parser.add_argument("--pa", default=-1, type=int,
                    help="size of the adversarial patch (first axis; for rectangle patch)")
parser.add_argument("--pb", default=-1, type=int,
                    help="size of the adversarial patch (second axis; for rectangle patch)")
parser.add_argument("--dump_dir", default='dump', type=str, help='directory to dump two-mask predictions')
parser.add_argument("--override", action='store_true', help='override dumped file')
parser.add_argument("--ablation_size", type=int, default=19, help='override dumped file')
parser.add_argument("--ablation_type", type=str, default='column', help='override dumped file')
parser.add_argument("--revise", type=bool, default=True, help='override dumped file')

parser.add_argument("--modify", type=bool, default=True, help='override dumped file')

args = parser.parse_args()
print(args)
print(args.patch_size)
DATASET = args.dataset
MODEL_DIR = os.path.join('.', args.model_dir)
DATA_DIR = os.path.join(args.data_dir, DATASET)
DUMP_DIR = os.path.join('.', args.dump_dir)
if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)
NUM_IMG = args.num_img
PC_MODEL_NAME = args.pc_model
MODEL_NAME = PC_MODEL_NAME
DRS_MODEL_NAME = args.drs_model
ablation_size = args.ablation_size
patch_size = args.patch_size
modify = args.modify
ablation_type=args.ablation_type
revise=args.revise
model = get_model(MODEL_NAME, DATASET, MODEL_DIR)
# print(model.named_parameters())
val_loader, NUM_IMG, ds_config = get_data_loader(DATASET, DATA_DIR, model, batch_size=16, num_img=NUM_IMG, train=False)

device = 'cuda'
# model = model.to(device)
model.eval()
cudnn.benchmark = True

# generate the mask set
mask_list, MASK_SIZE, MASK_STRIDE = gen_mask_set(args, ds_config)
if ablation_type=='column':
    if not args.override and os.path.exists(
            os.path.join(DUMP_DIR, 'suspect_column_list' + str(args.num_mask) + '_' + str(patch_size) + '.z')):
        print('loading column')
        suspect_column_list = joblib.load(
            os.path.join(DUMP_DIR, 'suspect_column_list' + str(args.num_mask) + '_' + str(patch_size) + '.z'))
    else:
        suspect_column_list = suspect_column_list_cal_fix(mask_list)
        joblib.dump(suspect_column_list,
                    os.path.join(DUMP_DIR, 'suspect_column_list' + str(args.num_mask) + '_' + str(patch_size) + '.z'))
        print('saving column')


prediction_map_list_pc = joblib.load(os.path.join(DUMP_DIR,
                                                  "prediction_map_list_two_mask_{}_{}_m{}_s{}_{}.z".format(DATASET,
                                                                                                           PC_MODEL_NAME,
                                                                                                           str(MASK_SIZE),
                                                                                                           str(MASK_STRIDE),
                                                                                                           NUM_IMG)))
label_list = joblib.load(
    os.path.join(DUMP_DIR, "label_list_{}_{}_{}.z".format(DATASET, PC_MODEL_NAME, NUM_IMG)))

prediction_map_list_drs = joblib.load(os.path.join(DUMP_DIR,
                                                   "prediction_map_list_drs_two_mask_{}_{}_{}_drs_{}_{}_{}_m{}_s1_{}.z".format(
                                                       DATASET, DRS_MODEL_NAME, DATASET,ablation_size,DATASET,ablation_type,ablation_size,
                                                       NUM_IMG)))

def static_cert_analysis(output_label_pc, output_label_drs, robust_pc, robust_drs):
    if robust_pc or (output_label_pc == output_label_drs and robust_drs):
        return True
    return False


def static_cert_very_stable_analysis(output_label_pc, output_label_drs, robust_pc, robust_drs):
    if output_label_pc == output_label_drs and (robust_pc and robust_drs):
        return True
    return False

correct_sample=0
location_certified_detected_sample=0
stable_sample =0
static_certified_detected_sample = 0

# if modify==True:
print(time())
start = time()
for i, (prediction_map_pc, label, prediction_map_drs) in enumerate(
        zip(prediction_map_list_pc, label_list, prediction_map_list_drs)):
    # print("label"+str(label))
    # init
    # generate a symmetric matrix from a triangle matrix
    prediction_map_pc = prediction_map_pc + prediction_map_pc.T - np.diag(np.diag(prediction_map_pc))
    if revise==False:
        # output label of PC
        # print("revise==False")
        output_label_pc, case_num = double_masking_precomputed_with_case_num(prediction_map_pc)
    else:
        # output label of PC with modify
        # print("revise==True")
        output_label_pc, case_num = double_masking_precomputed_with_case_num_modify(prediction_map_pc)
    # certified from pc
    robust_pc = certify_precomputed(prediction_map_pc, output_label_pc)
    # certified from drs
    output_label_drs, robust_drs = certified_drs(prediction_map_drs, ablation_size, patch_size)
    malicious_label_dict_pc_with_location = pc_malicious_label_with_location(prediction_map_pc, output_label_pc,
                                                                             num_mask=args.num_mask)
    stable_cert = static_cert_very_stable_analysis(output_label_pc, output_label_drs, robust_pc, robust_drs)
    if not stable_cert:
        location_cert = certified_with_location(malicious_label_dict_pc_with_location, suspect_column_list, patch_size,
                                                prediction_map_drs, ablation_size)
        static_cert_result = static_cert_analysis(output_label_pc, output_label_drs, robust_pc, robust_drs)
    else:
        location_cert = True
        static_cert_result = True

    if output_label_pc==label:
        correct_sample+=1
        if location_cert==True:
            location_certified_detected_sample +=1
        if stable_cert==True:
            stable_sample +=1
        if static_cert_result==True:
            static_certified_detected_sample +=1
    print("correct_sample " + str(correct_sample) + ' ' + str(correct_sample*100 / NUM_IMG))
    print("location_certified_detected_sample " + str(location_certified_detected_sample) + ' ' + str(location_certified_detected_sample*100 / NUM_IMG))
    print("stable_sample " + str(stable_sample) + ' ' + str(stable_sample*100 / NUM_IMG))
    print("static_certified_detected_sample " + str(static_certified_detected_sample) + ' ' + str(static_certified_detected_sample*100 / NUM_IMG))

    print("\n")
print(args)
