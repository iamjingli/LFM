import os
import os.path as osp
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torch.utils.data.dataloader import DataLoader
from config.model_config import build_args
from dataset.dataset import SFUniDADataset
from model.SFUniDA import SFUniDA

sns.set()


@torch.no_grad()
def get_data(args):
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = SFUniDA(args)

    if args.m == "Ours":
        # ours
        args.checkpoint = os.path.join("checkpoints", "model", args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                                       args.target_label_type, "ours", "{}_checkpoint.pth".format(args.dataset))
    elif args.m == "GLC":
        # GLC
        args.checkpoint = os.path.join("checkpoints", "model", args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                                       args.target_label_type, "glc", "{}_checkpoint.pth".format(args.dataset))
    elif args.m == "LEAD":
        # LEAD
        args.checkpoint = os.path.join("checkpoints", "model", args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                                       args.target_label_type, "lead", "{}_checkpoint.pth".format(args.dataset))
    elif args.m == "Source":
        args.checkpoint = os.path.join("checkpoints", args.dataset, "source_{}".format(args.s_idx),
                                       "source_{}_{}".format(args.source_train_type, args.target_label_type),
                                       "latest_source_checkpoint.pth")

    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(args.checkpoint)
        raise ValueError("YOU MUST SET THE APPROPORATE SOURCE CHECKPOINT FOR TARGET MODEL ADPTATION!!!")

    model = model.cuda()
    model.eval()

    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    print(len(target_dataset))

    target_test_dataloader = DataLoader(target_dataset, batch_size=args.batch_size * 2, shuffle=False,
                                        num_workers=args.num_workers, drop_last=False)

    gt_label_bank = []
    feats_bank = []
    for _, img, label, _ in target_test_dataloader:
        img = img.cuda()
        feats, _ = model(img, apply_softmax=True)
        gt_label_bank.append(label)
        feats_bank.append(feats)
    gt_label = torch.cat(gt_label_bank, dim=0)
    feats = torch.cat(feats_bank, dim=0)
    feats = feats / torch.norm(feats, p=2, dim=1, keepdim=True)

    data = {"feats": feats.cpu().numpy(), "labels": gt_label.cpu().numpy()}
    return data


class FeatureVisualize(object):
    """
    Visualize features by T-SNE
    """

    def __init__(self, feature, label, hue=None):
        """
        features: (m,n)
        labels: (m,)
        """
        self.features = feature
        self.labels = label
        self.hue = hue

    def plot_t_sne(self, perplexity=None):
        """
        Plot T-SNE figure. Set save_eps=True if you want to save a .eps file.
        """
        if perplexity is None:
            t_sne = TSNE(n_components=2, init='pca', random_state=2021)
        else:
            t_sne = TSNE(n_components=2, init='pca', random_state=2021, perplexity=perplexity)

        feats = t_sne.fit_transform(self.features)
        x_min, x_max = np.min(feats, 0), np.max(feats, 0)
        data = (feats - x_min) / (x_max - x_min)

        del feats

        df = pd.DataFrame(data, columns=['x', 'y'])
        df['label'] = self.labels
        df['hue'] = self.hue
        return df


def draw(data, path):
    figure = plt.figure(dpi=300)
    sns.set_style("white")
    # 获取当前坐标轴对象
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # colors = ["#FF5733", "#33FF57", "#5733FF", "#FF33E0", "#33E0FF", "#E0FF33", "#990000", "#009900", "#000099",
    #           "#999900", "#808080"]
    colors = ["#FF5733", "#33FF57", "#5733FF", "#FF33E0", "#33E0FF", "#E0FF33", "#808080"]
    sns.scatterplot(x="x", y="y", data=data, hue="label", palette=sns.color_palette(colors, 7), s=5, legend=False)

    plt.savefig(f'{path}/t.jpg', format='jpg')


if __name__ == '__main__':
    start = time.time()
    a = build_args()

    c = ["Ours", "GLC", "LEAD", "Source"]
    a.m = c[0]

    data_ = get_data(a)

    log_dir = f"tsne/{a.m}/"
    if not osp.exists(log_dir):
        os.system('mkdir -p ' + log_dir)
    if not osp.exists(log_dir):
        os.mkdir(log_dir)

    print(f"====================={a.m}=====================")
    da_ = FeatureVisualize(data_['feats'], data_['labels']).plot_t_sne(50)
    draw(da_, log_dir)

    print("T-SNE 画图完成")

    end = time.time()

    m = int((end - start) // 60)
    s = (end - start) % 60
    print(f"用时: {m} 分 {s:.1f} 秒")
