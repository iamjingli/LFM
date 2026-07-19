#!/bin/bash
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
random_seed=${2:-2021}
gpu=2
# =========================Office-31=========================
#echo "OPDA SOURCE TRAIN ON Office"
#python train_source.py  --dataset Office --s_idx 0  --target_label_type OPDA --epochs 50 --lr 0.01 --gpu $gpu
#python train_source.py  --dataset Office --s_idx 1  --target_label_type OPDA --epochs 50 --lr 0.01 --gpu $gpu
#python train_source.py  --dataset Office --s_idx 2  --target_label_type OPDA --epochs 50 --lr 0.01 --gpu $gpu
i=1

#echo "OPDA Adaptation ON Office"
python train_target_r1.py --dataset Office --s_idx 0 --t_idx 1 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_r1.py --dataset Office --s_idx 0 --t_idx 2 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_r1.py --dataset Office --s_idx 1 --t_idx 0 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15
#python train_target.py --dataset Office --s_idx 1 --t_idx 2 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15
#python train_target.py --dataset Office --s_idx 2 --t_idx 0 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15
#python train_target.py --dataset Office --s_idx 2 --t_idx 1 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15


# =========================Office-Home=========================
#echo "OPDA SOURCE TRAIN ON Office-Home" --gpu $gpu
#python train_source.py  --dataset OfficeHome --s_idx 0  --target_label_type OPDA --epochs 50 --lr 0.01 --gpu $gpu
#python train_source.py  --dataset OfficeHome --s_idx 1  --target_label_type OPDA --epochs 50 --lr 0.01 --gpu $gpu
#python train_source.py  --dataset OfficeHome --s_idx 2  --target_label_type OPDA --epochs 50 --lr 0.01 --gpu $gpu
#python train_source.py  --dataset OfficeHome --s_idx 3  --target_label_type OPDA --epochs 50 --lr 0.01 --gpu $gpu

#echo "OPDA Adaptation ON Office-Home"
#python train_target_r1.py --dataset OfficeHome --s_idx 0 --t_idx 1 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15
#python train_target_r1.py --dataset OfficeHome --s_idx 0 --t_idx 2 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15
#python train_target_r1.py --dataset OfficeHome --s_idx 0 --t_idx 3 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15
#python train_target_r1.py --dataset OfficeHome --s_idx 1 --t_idx 0 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15
#python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 2 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15
#python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 3 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15
#python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 0 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15
#python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 1 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15
#python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 3 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15
#python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 0 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15
#python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 1 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15
#python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 2 --target_label_type OPDA --lr 0.01 --index $i --gpu $gpu --epochs 15


# =========================VisDA=========================
#echo "OPDA SOURCE TRAIN ON VisDA"
#python train_source.py --backbone_arch resnet50 --dataset VisDA --s_idx 0  --target_label_type OPDA --epochs 10 --lr 0.001 --gpu $gpu


#echo "OPDA Adaptation ON VisDA"
#python train_target_r1.py --lr 0.01 --dataset VisDA --s_idx 0 --t_idx 1 --target_label_type OPDA --index $i --gpu $gpu --epochs 10


# =========================DomainNet=========================
#echo "OPDA SOURCE TRAIN ON DomainNet"
#python train_source.py  --dataset DomainNet --s_idx 0  --target_label_type OPDA --epochs 50 --lr 0.01 --gpu $gpu
#python train_source.py  --dataset DomainNet --s_idx 1  --target_label_type OPDA --epochs 50 --lr 0.01 --gpu $gpu
#python train_source.py  --dataset DomainNet --s_idx 2  --target_label_type OPDA --epochs 50 --lr 0.01 --gpu $gpu


#echo "OPDA Adaptation ON DomainNet"
#python train_target_r1.py --dataset DomainNet --s_idx 0 --t_idx 1 --lr 0.01 --target_label_type OPDA --index $i --gpu $gpu --epochs 10
#python train_target_r1.py --dataset DomainNet --s_idx 0 --t_idx 2 --lr 0.01 --target_label_type OPDA --index $i --gpu $gpu --epochs 10
#python train_target_r1.py --dataset DomainNet --s_idx 1 --t_idx 0 --lr 0.01 --target_label_type OPDA --index $i --gpu $gpu --epochs 10
#python train_target.py --dataset DomainNet --s_idx 1 --t_idx 2 --lr 0.01 --target_label_type OPDA --index $i --gpu $gpu --epochs 10
#python train_target.py --dataset DomainNet --s_idx 2 --t_idx 0 --lr 0.01 --target_label_type OPDA --index $i --gpu $gpu --epochs 10
#python train_target.py --dataset DomainNet --s_idx 2 --t_idx 1 --lr 0.01 --target_label_type OPDA --index $i --gpu $gpu --epochs 10
