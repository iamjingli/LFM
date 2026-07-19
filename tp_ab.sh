#!/bin/bash
random_seed=${2:-2021}
gpu=3
# =========================Target=========================
i=1
echo -e "\nPDA Adaptation ON Office" >> result_tp_ab.txt
python train_target_ab.py --dataset Office --s_idx 0 --t_idx 1 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_ab.py --dataset Office --s_idx 0 --t_idx 2 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_ab.py --dataset Office --s_idx 1 --t_idx 0 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_ab.py --dataset Office --s_idx 1 --t_idx 2 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_ab.py --dataset Office --s_idx 2 --t_idx 0 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_ab.py --dataset Office --s_idx 2 --t_idx 1 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15

echo -e "\nPDA Adaptation ON Office-Home" >> result_tp_ab.txt
python train_target_ab.py --dataset OfficeHome --s_idx 0 --t_idx 1 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_ab.py --dataset OfficeHome --s_idx 0 --t_idx 2 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_ab.py --dataset OfficeHome --s_idx 0 --t_idx 3 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_ab.py --dataset OfficeHome --s_idx 1 --t_idx 0 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_ab.py --dataset OfficeHome --s_idx 1 --t_idx 2 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_ab.py --dataset OfficeHome --s_idx 1 --t_idx 3 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_ab.py --dataset OfficeHome --s_idx 2 --t_idx 0 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_ab.py --dataset OfficeHome --s_idx 2 --t_idx 1 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_ab.py --dataset OfficeHome --s_idx 2 --t_idx 3 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_ab.py --dataset OfficeHome --s_idx 3 --t_idx 0 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_ab.py --dataset OfficeHome --s_idx 3 --t_idx 1 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15
python train_target_ab.py --dataset OfficeHome --s_idx 3 --t_idx 2 --target_label_type PDA --lr 0.01 --index $i --gpu $gpu --epochs 15

echo -e "\nPDA Adaptation ON VisDA" >> result_tp_ab.txt
python train_target_ab.py --lr 0.01 --dataset VisDA --s_idx 0 --t_idx 1 --target_label_type PDA  --epochs 10 --index $i --gpu $gpu