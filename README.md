# CrossCert

This is the code for CrossCert: A Cross-Checking Detection Approach to Patch Robustness Certification for Deep Learning Models in FSE 2024.



## Acknowledgment

This packgae is partly built upon [PatchCleanser](https://github.com/inspire-group/PatchCleanser), which is a quire tidy and clear project. For more details about masking-based recovery, please refer to their [repository](https://github.com/inspire-group/PatchCleanser).  As for the implementation of voting-based recovery, we refer in part to [DRS](https://github.com/alevine0/patchSmoothing). We sincerely thank their contribution to this research topic.



## Environment

The code is implemented in Python==3.8, timm==0.9.10, torch==2.0.1.



## Files

├── train_model.py              #Training for masking-based recovery base model 

├── train_drs.py              #Training for voting-based recovery base model 

├── pc_certification.py                  #Evaluate masking-based recovery defender

├── certification_drs.py                  #Evaluate voting-based recovery defender

├── CrossCert_sta.py                  #Evaluate CrossCert, which is based on the results of the above two defenders  



## Datasets

- [ImageNet](https://image-net.org/download.php) (ILSVRC2012)

- [CIFAR-10/CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

## Demo

0. You may first need to configure your local location of the directory.

   

1. First, train base DL models of the two recovery defenders. 

  ```python
python train_drs.py --dataset imagenet --ablation_type column --model vit_base_patch16_224 --ablation_size 19
python train_model.py --model vit_base_patch16_224 --dataset imagenet
  ```

  

2. Then, get the inference results of samples in the dataset from the two recovery defenders.

  ```python
python certification_drs.py --dataset imagenet --ablation_type column --model vit_base_patch16_224 --ablation_size 19
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_mask 6 --patch_size 32
  ```



3. Finally, set the directory, dataset name, and model name in CrossCert_sta.py, and then run it to see the results of CrossCert.
