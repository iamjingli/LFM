import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import torch
import tqdm
from torch.utils.data.dataloader import DataLoader

from config.model_config import build_args
from dataset.dataset import SFUniDADataset
from model.SFUniDA import SFUniDA

args = build_args()
torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)


def plot_confusion_matrix_seaborn(cm, t):
    figure = plt.figure(dpi=300)
    ax = sns.heatmap(cm, annot=False, cmap='Blues', cbar=False)
    # ax.set_title("Confusion Matrix")
    # ax.set_xlabel("Predicted label")
    # ax.set_ylabel("True label")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'cm/{t}.jpg', format='jpg')


def confusion_matrix(y_true, y_pred, num_classes):
    cm = torch.zeros(num_classes, num_classes + 1, dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


# ============== Source model ==============
sou_model = SFUniDA(args)
checkpoint = os.path.join("checkpoints", args.dataset, "source_{}".format(args.s_idx),
                          "source_{}_{}".format(args.source_train_type, args.target_label_type),
                          "latest_source_checkpoint.pth")
if checkpoint is not None and os.path.isfile(checkpoint):
    c = torch.load(checkpoint, map_location=torch.device("cpu"))
    sou_model.load_state_dict(c["model_state_dict"])
else:
    print(args.checkpoint)
    raise ValueError("YOU MUST SET THE APPROPORATE SOURCE CHECKPOINT FOR TARGET MODEL ADPTATION!!!")
sou_model.cuda()
sou_model.eval()

# ============== Ours ==============
ours_model = SFUniDA(args)
checkpoint = os.path.join("checkpoints", "model", args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                          args.target_label_type, "ours", "{}_checkpoint.pth".format(args.dataset))
if checkpoint is not None and os.path.isfile(checkpoint):
    c = torch.load(checkpoint, map_location=torch.device("cpu"))
    ours_model.load_state_dict(c["model_state_dict"])
else:
    print(args.checkpoint)
    raise ValueError("YOU MUST SET THE APPROPORATE SOURCE CHECKPOINT FOR TARGET MODEL ADPTATION!!!")
ours_model.cuda()
ours_model.eval()

# ============== GLC ==============
glc_model = SFUniDA(args)
checkpoint = os.path.join("checkpoints", "model", args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                          args.target_label_type, "glc", "{}_checkpoint.pth".format(args.dataset))
if checkpoint is not None and os.path.isfile(checkpoint):
    c = torch.load(checkpoint, map_location=torch.device("cpu"))
    glc_model.load_state_dict(c["model_state_dict"])
else:
    print(args.checkpoint)
    raise ValueError("YOU MUST SET THE APPROPORATE SOURCE CHECKPOINT FOR TARGET MODEL ADPTATION!!!")
glc_model.cuda()
glc_model.eval()

# ============== LEAD ==============
lead_model = SFUniDA(args)
checkpoint = os.path.join("checkpoints", "model", args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                          args.target_label_type, "lead", "{}_checkpoint.pth".format(args.dataset))
if checkpoint is not None and os.path.isfile(checkpoint):
    c = torch.load(checkpoint, map_location=torch.device("cpu"))
    lead_model.load_state_dict(c["model_state_dict"])
else:
    print(args.checkpoint)
    raise ValueError("YOU MUST SET THE APPROPORATE SOURCE CHECKPOINT FOR TARGET MODEL ADPTATION!!!")
lead_model.cuda()
lead_model.eval()

target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
print(len(target_dataset))

target_test_dataloader = DataLoader(target_dataset, batch_size=args.batch_size * 2, shuffle=False,
                                    num_workers=args.num_workers, drop_last=False)

task = ["source", "ours", "glc", "lead"]
model = [sou_model, ours_model, glc_model, lead_model]
for i in range(4):
    loop = tqdm.tqdm(target_test_dataloader, ncols=80)
    loop.set_description_str(f"TASK-{task[i]}")
    with torch.no_grad():
        gt_label = []
        pred_label = []
        for _, img, label, _ in loop:
            _, softmax_out = model[i](img.cuda())
            p_label = softmax_out.argmax(1).cpu()

            gt_label.append(label)
            pred_label.append(p_label)
        gt = torch.cat(gt_label)
        pred = torch.cat(pred_label)
        idx = torch.where(pred > 24)[0]
        pred[idx] = 25
        class_num = 25

        c = confusion_matrix(gt, pred, class_num)
        plot_confusion_matrix_seaborn(c, task[i])
