#!/bin/bash
random_seed=${2:-2021}
gpu=0
i=1
# =========================Office-31=========================
echo "PDA SOURCE TRAIN ON Office"
python train_source.py  --dataset Office --s_idx 0  --target_label_type PDA --epochs 50 --lr 0.01 --gpu $gpu
python train_source.py  --dataset Office --s_idx 1  --target_label_type PDA --epochs 50 --lr 0.01 --gpu $gpu
python train_source.py  --dataset Office --s_idx 2  --target_label_type PDA --epochs 50 --lr 0.01 --gpu $gpu

echo "PDA Adaptation ON Office"
python train_target.py --dataset Office --s_idx 0 --t_idx 1 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target.py --dataset Office --s_idx 0 --t_idx 2 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target.py --dataset Office --s_idx 1 --t_idx 0 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target.py --dataset Office --s_idx 1 --t_idx 2 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target.py --dataset Office --s_idx 2 --t_idx 0 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target.py --dataset Office --s_idx 2 --t_idx 1 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15

# =========================Office-Home=========================
echo "PDA SOURCE TRAIN ON Office-Home"
python train_source.py  --dataset OfficeHome --s_idx 0  --target_label_type PDA --epochs 50 --lr 0.01 --gpu $gpu
python train_source.py  --dataset OfficeHome --s_idx 1  --target_label_type PDA --epochs 50 --lr 0.01 --gpu $gpu
python train_source.py  --dataset OfficeHome --s_idx 2  --target_label_type PDA --epochs 50 --lr 0.01 --gpu $gpu
python train_source.py  --dataset OfficeHome --s_idx 3  --target_label_type PDA --epochs 50 --lr 0.01 --gpu $gpu

echo "PDA Adaptation ON Office-Home"
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 1 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 2 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 3 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 0 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 2 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 3 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 0 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 1 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 3 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 0 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 1 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 2 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15

# =========================VisDA=========================
echo "PDA SOURCE TRAIN ON VisDA"
python train_source.py --backbone_arch resnet50 --dataset VisDA --s_idx 0  --target_label_type PDA --epochs 10 --lr 0.001 --gpu $gpu

echo -e "\nPDA Adaptation ON VisDA" >> result_tp.txt
python train_target.py --lr 0.01 --dataset VisDA --s_idx 0 --t_idx 1 --target_label_type PDA --index $i --gpu $gpu --epochs 10

