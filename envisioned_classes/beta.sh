#!/bin/bash
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
random_seed=${2:-2021}
gpu=6
# =========================Office-31=========================
i=1
beta=1

#echo "OPDA Adaptation ON Office"
#python train_target_beta.py --dataset Office --s_idx 0 --t_idx 1 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta
#python train_target_beta.py --dataset Office --s_idx 0 --t_idx 2 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta
#python train_target_beta.py --dataset Office --s_idx 1 --t_idx 0 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta
#python train_target_beta.py --dataset Office --s_idx 1 --t_idx 2 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta
#python train_target_beta.py --dataset Office --s_idx 2 --t_idx 0 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta
#python train_target_beta.py --dataset Office --s_idx 2 --t_idx 1 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta

# =========================Office-Home=========================
echo "OPDA Adaptation ON Office-Home"
python train_target_beta.py --dataset OfficeHome --s_idx 0 --t_idx 1 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta
python train_target_beta.py --dataset OfficeHome --s_idx 0 --t_idx 2 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta
python train_target_beta.py --dataset OfficeHome --s_idx 0 --t_idx 3 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta
python train_target_beta.py --dataset OfficeHome --s_idx 1 --t_idx 0 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta
python train_target_beta.py --dataset OfficeHome --s_idx 1 --t_idx 2 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta
python train_target_beta.py --dataset OfficeHome --s_idx 1 --t_idx 3 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta
python train_target_beta.py --dataset OfficeHome --s_idx 2 --t_idx 0 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta
python train_target_beta.py --dataset OfficeHome --s_idx 2 --t_idx 1 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta
python train_target_beta.py --dataset OfficeHome --s_idx 2 --t_idx 3 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta
python train_target_beta.py --dataset OfficeHome --s_idx 3 --t_idx 0 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta
python train_target_beta.py --dataset OfficeHome --s_idx 3 --t_idx 1 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta
python train_target_beta.py --dataset OfficeHome --s_idx 3 --t_idx 2 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15 --beta $beta


## =========================VisDA=========================
#echo "OPDA Adaptation ON VisDA"
#python train_target_beta.py --lr 0.01 --dataset VisDA --s_idx 0 --t_idx 1 --target_label_type OPDA --index $i --gpu $gpu --epochs 10 --beta $beta
#
## =========================DomainNet=========================
#echo "OPDA Adaptation ON DomainNet"
#python train_target_beta.py --dataset DomainNet --s_idx 0 --t_idx 1 --lr 0.01 --target_label_type OPDA --index $i --gpu $gpu --epochs 10 --beta $beta
#python train_target_beta.py --dataset DomainNet --s_idx 0 --t_idx 2 --lr 0.01 --target_label_type OPDA --index $i --gpu $gpu --epochs 10 --beta $beta
#python train_target_beta.py --dataset DomainNet --s_idx 1 --t_idx 0 --lr 0.01 --target_label_type OPDA --index $i --gpu $gpu --epochs 10 --beta $beta
#python train_target_beta.py --dataset DomainNet --s_idx 1 --t_idx 2 --lr 0.01 --target_label_type OPDA --index $i --gpu $gpu --epochs 10 --beta $beta
#python train_target_beta.py --dataset DomainNet --s_idx 2 --t_idx 0 --lr 0.01 --target_label_type OPDA --index $i --gpu $gpu --epochs 10 --beta $beta
#python train_target_beta.py --dataset DomainNet --s_idx 2 --t_idx 1 --lr 0.01 --target_label_type OPDA --index $i --gpu $gpu --epochs 10 --beta $beta
