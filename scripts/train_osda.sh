#!/bin/bash
random_seed=${2:-2021}
gpu=0
# =========================Office-31=========================
echo "OSDA SOURCE TRAIN ON Office"
python train_source.py  --dataset Office --s_idx 0  --target_label_type OSDA --epochs 50 --lr 0.01 --gpu $gpu
python train_source.py  --dataset Office --s_idx 1  --target_label_type OSDA --epochs 50 --lr 0.01 --gpu $gpu
python train_source.py  --dataset Office --s_idx 2  --target_label_type OSDA --epochs 50 --lr 0.01 --gpu $gpu

for i in {1..3}
do
  echo "OSDA Adaptation ON Office"
  python train_target.py --dataset Office --s_idx 0 --t_idx 1 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
  python train_target.py --dataset Office --s_idx 0 --t_idx 2 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
  python train_target.py --dataset Office --s_idx 1 --t_idx 0 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
  python train_target.py --dataset Office --s_idx 1 --t_idx 2 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
  python train_target.py --dataset Office --s_idx 2 --t_idx 0 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
  python train_target.py --dataset Office --s_idx 2 --t_idx 1 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
done

# =========================Office-Home=========================
echo "OSDA SOURCE TRAIN ON Office-Home"
python train_source.py  --dataset OfficeHome --s_idx 0  --target_label_type OSDA --epochs 50 --lr 0.01 --gpu $gpu
python train_source.py  --dataset OfficeHome --s_idx 1  --target_label_type OSDA --epochs 50 --lr 0.01 --gpu $gpu
python train_source.py  --dataset OfficeHome --s_idx 2  --target_label_type OSDA --epochs 50 --lr 0.01 --gpu $gpu
python train_source.py  --dataset OfficeHome --s_idx 3  --target_label_type OSDA --epochs 50 --lr 0.01 --gpu $gpu

for i in {1..3}
do
  echo "OSDA Adaptation ON Office-Home"
  python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 1 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
  python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 2 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
  python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 3 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
  python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 0 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
  python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 2 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
  python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 3 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
  python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 0 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
  python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 1 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
  python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 3 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
  python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 0 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
  python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 1 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
  python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 2 --target_label_type OSDA --lr 0.01 --index $i --gpu $gpu --epochs 15
done

# =========================VisDA=========================
echo "OSDA SOURCE TRAIN ON VisDA"
python train_source.py --backbone_arch resnet50 --dataset VisDA --s_idx 0 --target_label_type OSDA --epochs 10 --lr 0.001 --gpu $gpu

for i in {1..3}
do
  echo "OSDA Adaptation ON VisDA"
  python train_target.py --lr 0.01 --dataset VisDA --s_idx 0 --t_idx 1 --target_label_type OSDA --index $i --gpu $gpu --epochs 10
done


