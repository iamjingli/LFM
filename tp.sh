#!/bin/bash
random_seed=${2:-2021}
gpu=3
# =========================Source=========================
#echo "PDA SOURCE TRAIN ON Office"
#python train_source.py  --dataset Office --s_idx 0  --target_label_type PDA --epochs 50 --lr 0.01
#python train_source.py  --dataset Office --s_idx 1  --target_label_type PDA --epochs 50 --lr 0.01
#python train_source.py  --dataset Office --s_idx 2  --target_label_type PDA --epochs 50 --lr 0.01

#echo "PDA SOURCE TRAIN ON Office-Home"
#python train_source.py  --dataset OfficeHome --s_idx 0  --target_label_type PDA --epochs 50 --lr 0.01
#python train_source.py  --dataset OfficeHome --s_idx 1  --target_label_type PDA --epochs 50 --lr 0.01
#python train_source.py  --dataset OfficeHome --s_idx 2  --target_label_type PDA --epochs 50 --lr 0.01
#python train_source.py  --dataset OfficeHome --s_idx 3  --target_label_type PDA --epochs 50 --lr 0.01

#echo "PDA SOURCE TRAIN ON VisDA"
#python train_source.py --backbone_arch resnet50 --dataset VisDA --s_idx 0  --target_label_type PDA --epochs 10 --lr 0.001

# =========================Target=========================
i=1
echo -e "\nPDA Adaptation ON Office" >> result_tp.txt
python train_target.py --dataset Office --s_idx 0 --t_idx 1 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu
python train_target.py --dataset Office --s_idx 0 --t_idx 2 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu
python train_target.py --dataset Office --s_idx 1 --t_idx 0 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu
python train_target.py --dataset Office --s_idx 1 --t_idx 2 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu
python train_target.py --dataset Office --s_idx 2 --t_idx 0 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu
python train_target.py --dataset Office --s_idx 2 --t_idx 1 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu

echo -e "\nPDA Adaptation ON Office-Home" >> result_tp.txt
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 1 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 2 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 3 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 0 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 2 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 3 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 0 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 1 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 3 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 0 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 1 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 2 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu

echo -e "\nPDA Adaptation ON VisDA" >> result_tp.txt
python train_target.py --lr 0.01 --dataset VisDA --s_idx 0 --t_idx 1 --target_label_type PDA  --epochs 10 --index $i --gpu $gpu