import torch
import torch.backends.cudnn as cudnn

import numpy as np
import os
import argparse
import time

from matplotlib import pyplot as plt
from tqdm import tqdm
import joblib

from utils.drs import  gen_ablation_set_column_fix, gen_ablation_set_row_fix
from utils.setup import get_model, get_data_loader
from utils.defense import gen_mask_set, double_masking_precomputed, certify_precomputed
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default='', type=str, help="directory of checkpoints")
parser.add_argument('--data_dir', default='', type=str, help="directory of data")
parser.add_argument('--dataset', default='imagenette', type=str,
                    choices=('imagenette', 'imagenet', 'cifar', 'cifar100', 'svhn', 'flowers102', 'gtsrb'),
                    help="dataset")
parser.add_argument("--model", default='vit_base_patch16_224', type=str, help="model name")
parser.add_argument("--num_img", default=-1, type=int,
                    help="number of randomly selected images for this experiment (-1: using the all images)")
parser.add_argument("--mask_stride", default=-1, type=int, help="mask stride s (square patch; conflict with num_mask)")
parser.add_argument("--num_mask", default=6, type=int,
                    help="number of mask in one dimension (square patch; conflict with mask_stride)")
parser.add_argument("--patch_size", default=-1, type=int, help="size of the adversarial patch (square patch)")
parser.add_argument("--pa", default=-1, type=int,
                    help="size of the adversarial patch (first axis; for rectangle patch)")
parser.add_argument("--pb", default=-1, type=int,
                    help="size of the adversarial patch (second axis; for rectangle patch)")
parser.add_argument("--dump_dir", default='dump', type=str, help='directory to dump two-mask predictions')
parser.add_argument("--override", action='store_true', help='override dumped file')
parser.add_argument("--ablation_size",type=int,default=19, help='override dumped file')
parser.add_argument("--ablation_type",type=str,default='column', help='override dumped file')

print("mask predict")
args = parser.parse_args()
print(str(args.num_img)+" num_img")
print(args)
DATASET = args.dataset
print(args.model)
args.model=args.model+'_'+args.dataset+'_drs_'+str(args.ablation_size)+'_'+args.dataset+'_'+args.ablation_type
MODEL_DIR = os.path.join('.', args.model_dir)
DATA_DIR = os.path.join(args.data_dir, DATASET)
DUMP_DIR = os.path.join('.', args.dump_dir)
if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)
# if DATASET == 'imagenette':
#     DATA_DIR = os.path.join(DATA_DIR, DATASET)
MODEL_NAME = args.model
NUM_IMG = args.num_img

# get model and data loader
model = get_model(MODEL_NAME, DATASET, MODEL_DIR)
val_loader, NUM_IMG, ds_config = get_data_loader(DATASET, DATA_DIR, model, batch_size=32, num_img=NUM_IMG, train=False)

device = 'cuda'
# model = torch.nn.DataParallel(model)
model = model.to(device)
model.eval()
cudnn.benchmark = True
ablation_size=args.ablation_size
# generate the mask set
# ablation_list, MASK_SIZE, MASK_STRIDE = gen_ablation_set(ablation_size)
ablation_type=args.ablation_type

if ablation_type=="column":
    ablation_list, MASK_SIZE, MASK_STRIDE = gen_ablation_set_column_fix(ablation_size)
elif ablation_type=="row":
    ablation_list, MASK_SIZE, MASK_STRIDE = gen_ablation_set_row_fix(ablation_size)
else:
    print("WRONG!")
# elif ablation_type=="block":
#     ablation_list, MASK_SIZE, MASK_STRIDE = gen_ablation_set_block(ablation_size)

# the computation of two-mask predictions is expensive; will dump (or resue the dumped) two-mask predictions.
SUFFIX = '_two_mask_{}_{}_m{}_s{}_{}.z'.format(DATASET, MODEL_NAME, MASK_SIZE, MASK_STRIDE, NUM_IMG)
print("MASK_SIZE" + str(MASK_SIZE) + "MASK_STRIDE" + str(MASK_STRIDE))
if False:
    print(" ")
    # if not args.override and os.path.exists(os.path.join(DUMP_DIR, 'prediction_map_list' + SUFFIX)):
    # print('loading two-mask predictions')
    # prediction_map_list = joblib.load(os.path.join(DUMP_DIR, 'prediction_map_list' + SUFFIX))
    # orig_prediction_list = joblib.load(
    #     os.path.join(DUMP_DIR, 'orig_prediction_list_{}_{}_{}.z'.format(DATASET, MODEL_NAME, NUM_IMG)))
    # label_list = joblib.load(os.path.join(DUMP_DIR, 'label_list_{}_{}_{}.z'.format(DATASET, MODEL_NAME, NUM_IMG)))
else:
    print('computing two-mask predictions')
    prediction_map_list = []
    confidence_map_list = []
    label_list = []
    orig_prediction_list = []
    for data, labels in tqdm(val_loader):
        data = data.to(device)
        labels = labels.numpy()
        num_img = data.shape[0]
        num_mask = len(ablation_list)
        # two-mask predictions
        prediction_map = np.zeros([num_img, num_mask], dtype=int)
        confidence_map = np.zeros([num_img, num_mask])
        for i, mask in enumerate(ablation_list):
            masked_output = model(torch.where(mask, data, torch.tensor(0.).cuda()))
            # plt.imshow(
            #     (torch.where(mask, data, torch.tensor(0.).cuda()).cpu()[0]).permute(1, 2, 0))
            # plt.show()
            masked_output = torch.nn.functional.softmax(masked_output, dim=1)
            masked_conf, masked_pred = masked_output.max(1)
            masked_conf = masked_conf.detach().cpu().numpy()
            confidence_map[:, i] = masked_conf
            masked_pred = masked_pred.detach().cpu().numpy()
            prediction_map[:, i] = masked_pred

        # vanilla predictions
        clean_output = model(data)
        clean_conf, clean_pred = clean_output.max(1)
        clean_pred = clean_pred.detach().cpu().numpy()
        orig_prediction_list.append(clean_pred)
        prediction_map_list.append(prediction_map)
        confidence_map_list.append(confidence_map)
        label_list.append(labels)

    prediction_map_list = np.concatenate(prediction_map_list)
    confidence_map_list = np.concatenate(confidence_map_list)
    orig_prediction_list = np.concatenate(orig_prediction_list)
    label_list = np.concatenate(label_list)

    joblib.dump(confidence_map_list, os.path.join(DUMP_DIR, 'confidence_map_list_drs' + SUFFIX))
    joblib.dump(prediction_map_list, os.path.join(DUMP_DIR, 'prediction_map_list_drs' + SUFFIX))
    joblib.dump(orig_prediction_list,
                os.path.join(DUMP_DIR, 'orig_prediction_list_{}_{}_{}_drs.z'.format(DATASET, MODEL_NAME, NUM_IMG)))
    joblib.dump(label_list, os.path.join(DUMP_DIR, 'label_list_{}_{}_{}_drs.z'.format(DATASET, MODEL_NAME, NUM_IMG)))

# clean_corr = 0
# robust = 0
# orig_corr = 0
# for i, (prediction_map, label, orig_pred) in enumerate(zip(prediction_map_list, label_list, orig_prediction_list)):
#     prediction_map = prediction_map + prediction_map.T - np.diag(
#         np.diag(prediction_map))  # generate a symmetric matrix from a triangle matrix
#     robust += certify_precomputed(prediction_map, label)
#     clean_corr += double_masking_precomputed(prediction_map) == label
#     orig_corr += orig_pred == label
#
# print("------------------------------")
# print("Certified robust accuracy:", robust / NUM_IMG)
# print("Clean accuracy with defense:", clean_corr / NUM_IMG)
# print("Clean accuracy without defense:", orig_corr / NUM_IMG)

