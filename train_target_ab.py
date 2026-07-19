import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # for SSLError: HTTPSConnectionPool
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from config.model_config import build_args
from dataset.dataset import SFUniDADataset
from model.SFUniDA import SFUniDA, set_model_clip
from utils.net_utils import compute_h_score, load_llm_classes, get_ood_scores_clip
from utils.net_utils import set_logger, set_random_seed
from sklearn.mixture import GaussianMixture


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def get_distances(x, y, dist_type="cosine"):
    if dist_type == "euclidean":
        distances = torch.cdist(x, y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(x, y.T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")
    return distances


@torch.no_grad()
def get_pseudo_labels(model, test_dataloader, k_idx, unk_idx, clip_label, args, clip_max_s):
    model.eval()
    pred_cls_bank = []
    embed_feat_bank = []
    gt_label_bank = []
    for _, img_test, img_label, _ in tqdm(test_dataloader, ncols=60):
        img_test = img_test.cuda()
        embed_feat, pred_cls = model(img_test, apply_softmax=True)
        pred_cls_bank.append(pred_cls)
        embed_feat_bank.append(embed_feat)
        gt_label_bank.append(img_label.cuda())
    pred_cls_bank = torch.cat(pred_cls_bank, 0)
    embed_feat_bank = torch.cat(embed_feat_bank, 0)
    embed_feat_bank = embed_feat_bank / torch.norm(embed_feat_bank, p=2, dim=1, keepdim=True)
    gt_label_bank = torch.cat(gt_label_bank, 0)

    # ===================Pseudo-label refinement===================
    pseudo_labels = clip_label[0].clone()
    # data_len = embed_feat_bank[k_idx].size(0)
    # batch_size = args.batch_size
    # num_batches = data_len // batch_size
    # soft_label = []
    # max_score = []
    # for i in range(num_batches):
    #     start_idx = i * batch_size
    #     end_idx = (i + 1) * batch_size
    #     x_batch = embed_feat_bank[k_idx[start_idx:end_idx]]
    #     cosine_distance_batch = 1 - torch.matmul(x_batch, embed_feat_bank[k_idx].T)
    #     k_nbr = torch.sort(cosine_distance_batch)[1][:, 1: args.nbr + 1]
    #     k_so_pred = pred_cls_bank[k_idx[k_nbr]]
    #     m_k_so_pred = torch.mean(k_so_pred, dim=1)
    #     max_score.append(m_k_so_pred.max(1)[0])
    #     k_soft_label = m_k_so_pred.argmax(1)
    #     soft_label.append(k_soft_label)
    # last_idx = num_batches * batch_size
    # last_x = embed_feat_bank[k_idx[last_idx:]]
    # cosine_distance_batch = 1 - torch.matmul(last_x, embed_feat_bank[k_idx].T)
    # k_nbr = torch.sort(cosine_distance_batch)[1][:, 1: args.nbr + 1]
    # k_so_pred = pred_cls_bank[k_idx[k_nbr]]
    # m_k_so_pred = torch.mean(k_so_pred, dim=1)
    # max_score.append(m_k_so_pred.max(1)[0])
    # k_soft_label = m_k_so_pred.argmax(1)
    # soft_label.append(k_soft_label)
    # soft_label = torch.cat(soft_label)
    # max_score = torch.cat(max_score)

    # tw_idx = soft_label == clip_label[1][k_idx]
    # ex_s_idx = max_score[tw_idx] > clip_max_s[k_idx[tw_idx]]
    # pseudo_labels[k_idx[tw_idx][ex_s_idx]] = soft_label[tw_idx][ex_s_idx]
    #
    # tr_label = soft_label == clip_label[2][k_idx]
    # ex_s_idx = max_score[tr_label] > clip_max_s[k_idx[tr_label]]
    # pseudo_labels[k_idx[tr_label][ex_s_idx]] = soft_label[tr_label][ex_s_idx]
    # ===================Pseudo-label refinement===================

    acc = (pseudo_labels[k_idx] == gt_label_bank[k_idx]).sum() / len(k_idx)
    unk_acc = (gt_label_bank[unk_idx] == pred_cls_bank.size(1)).sum() / (len(unk_idx) + 1e-5)
    clip_acc = (clip_label[0][k_idx] == gt_label_bank[k_idx]).sum() / len(k_idx)

    if args.target_label_type == "PDA":
        h_score = clip_acc
    else:
        h_score = 2 * clip_acc * unk_acc / (clip_acc + unk_acc)

    args.CLIP_h_score = h_score

    args.logger.info(f"pseudo label: {acc:.1%} unk acc: {unk_acc:.1%}\n")
    args.logger.info(f"clip label acc: {clip_acc:.1%} clip unk num: {len(unk_idx)}\n")
    args.logger.info(f"clip h-score: {h_score:.1%}\n")
    return pseudo_labels


def get_idx(args, dataloader):
    net, preprocess = set_model_clip(args)
    net.eval()

    llm_labels = load_llm_classes(args)
    score, clip_label, clip_max_score, j_score = get_ood_scores_clip(args, net, dataloader, llm_labels)
    sort_score = torch.sort(j_score)[0]

    high_score = sort_score[-int(len(score) * 0.3):]
    low_score = sort_score[:int(len(score) * 0.3)]

    c_score = torch.cat([high_score, low_score])

    mean = torch.mean(c_score)
    std = torch.std(c_score)
    cv_value = std / mean

    all_idx = torch.arange(0, len(score))
    if cv_value > args.delta:
        data = score.numpy().reshape(-1, 1)
        gmm = GaussianMixture(n_components=2)
        gmm.fit(data)
        means = gmm.means_.flatten()
        if means[0] < means[1]:
            lower_mean_label = 0
        else:
            lower_mean_label = 1
        labels = gmm.predict(data)
        new_labels = torch.tensor([0 if label == lower_mean_label else 1 for label in labels])
        k_idx = torch.where(new_labels == 1)[0]
    else:
        k_idx = all_idx

    unk_idx = all_idx[~torch.isin(all_idx, k_idx)]
    return k_idx.cuda(), unk_idx.cuda(), clip_label, clip_max_score


def tr(args, model, train_dataloader, test_dataloader, optimizer, epoch_idx, k_idx, unk_idx, clip_label, clip_max_s):
    args.logger.info("\n")
    args.logger.info(f"{args.target_label_type} {args.dataset} {args.s_idx}->{args.t_idx} "
                     f"Epoch: {epoch_idx + 1} / {args.epochs}")
    pseudo_label = get_pseudo_labels(model, test_dataloader, k_idx, unk_idx,
                                     clip_label, args, clip_max_s)
    model.train()
    iter_idx = epoch_idx * len(train_dataloader)
    iter_max = args.epochs * len(train_dataloader)
    for img_train, _, _, img_idx in tqdm(train_dataloader, ncols=60):
        iter_idx += 1
        img_idx = img_idx.cuda()
        img_train = img_train.cuda()
        _, pred_cls = model(img_train, apply_softmax=True)
        b_k_idx = torch.where(torch.isin(img_idx, k_idx))[0].cuda()
        b_unk_idx = torch.where(torch.isin(img_idx, unk_idx))[0]

        loss_total = torch.tensor(0.).cuda()

        b_pseudo_label = torch.eye(pred_cls.size(1))[pseudo_label[img_idx]].cuda()
        loss_cls = torch.sum(- b_pseudo_label[b_k_idx] * torch.log(pred_cls[b_k_idx]), 1).mean()
        loss_total += loss_cls

        loss_unk = torch.sum(pred_cls[b_unk_idx] * torch.log(pred_cls[b_unk_idx]), 1).mean()
        loss_total += loss_unk

        lr_scheduler(optimizer, iter_idx, iter_max)
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()


@torch.no_grad()
def test(args, model, dataloader, src_flg=False):
    model.eval()
    gt_label_stack = []
    pred_cls_stack = []

    if src_flg:
        class_list = args.source_class_list
        open_flg = False
    else:
        class_list = args.target_class_list
        open_flg = args.target_private_class_num > 0

    for _, imgs_test, imgs_label, _ in dataloader:
        imgs_test = imgs_test.cuda()
        _, pred_cls = model(imgs_test, apply_softmax=True)
        gt_label_stack.append(imgs_label)
        pred_cls_stack.append(pred_cls.cpu())

    gt_label_all = torch.cat(gt_label_stack, dim=0)  # [N]
    pred_cls_all = torch.cat(pred_cls_stack, dim=0)  # [N, C]

    h_score, known_acc, unknown_acc, _ = compute_h_score(args, class_list, gt_label_all, pred_cls_all, open_flg,
                                                         open_thresh=args.w_0)
    return h_score, known_acc, unknown_acc


def main(args):
    torch.cuda.set_device(args.gpu)
    this_dir = os.path.join(os.path.dirname(__file__), ".")
    torch.multiprocessing.set_sharing_strategy('file_system')

    model = SFUniDA(args)

    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(args.checkpoint)
        raise ValueError("YOU MUST SET THE APPROPORATE SOURCE CHECKPOINT FOR TARGET MODEL ADPTATION!!!")

    model = model.cuda()
    save_dir = os.path.join(this_dir, "checkpoints_ab", args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                            args.target_label_type, args.note)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    args.logger = set_logger(args, log_name=f"log_target_training_{args.index}.txt")

    param_group = []
    for k, v in model.backbone_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 0.1}]

    for k, v in model.feat_embed_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    for k, v in model.class_layer.named_parameters():
        v.requires_grad = False

    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)

    target_train_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, drop_last=True)
    target_test_dataloader = DataLoader(target_dataset, batch_size=args.batch_size * 2, shuffle=False,
                                        num_workers=args.num_workers, drop_last=False)

    notation_str = "\n=======================================================\n"
    notation_str += "   START TRAINING ON THE TARGET:{} BASED ON SOURCE:{}  \n".format(args.t_idx, args.s_idx)
    notation_str += "======================================================="

    args.logger.info(notation_str)
    best_h_score = 0.0
    best_known_acc = 0.0
    best_unknown_acc = 0.0
    k_idx, unk_idx, clip_label, clip_max_score = get_idx(args, target_test_dataloader)
    for epoch_idx in range(args.epochs):
        # Train on target
        tr(args, model, target_train_dataloader, target_test_dataloader, optimizer, epoch_idx, k_idx, unk_idx,
           clip_label, clip_max_score)
        # loss_dict = train(args, model, target_train_dataloader, target_test_dataloader, optimizer, epoch_idx)
        # args.logger.info("Epoch: {}/{},          train_all_loss:{:.3f},\n\
        #                   train_psd_loss:{:.3f}, train_knn_loss:{:.3f},".format(epoch_idx + 1, args.epochs,
        #                                                                         loss_dict["all_pred_loss"],
        #                                                                         loss_dict["psd_pred_loss"],
        #                                                                         loss_dict["knn_pred_loss"]))

        # Evaluate on target
        hscore, knownacc, unknownacc = test(args, model, target_test_dataloader, src_flg=False)
        args.logger.info(
            "Current: H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(hscore, knownacc, unknownacc))

        if args.target_label_type == 'PDA' or args.target_label_type == 'CLDA':
            if knownacc >= best_known_acc:
                best_h_score = hscore
                best_known_acc = knownacc
                best_unknown_acc = unknownacc

                # checkpoint_file = "{}_SFDA_best_target_checkpoint.pth".format(args.dataset)
                # torch.save({
                #     "epoch": epoch_idx,
                #     "model_state_dict": model.state_dict()}, os.path.join(save_dir, checkpoint_file))
        else:
            if hscore >= best_h_score:
                best_h_score = hscore
                best_known_acc = knownacc
                best_unknown_acc = unknownacc

                # checkpoint_file = "{}_SFDA_best_target_checkpoint.pth".format(args.dataset)
                # torch.save({
                #     "epoch": epoch_idx,
                #     "model_state_dict": model.state_dict()}, os.path.join(save_dir, checkpoint_file))

        args.logger.info(
            "Best   : H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(best_h_score, best_known_acc,
                                                                                 best_unknown_acc))

    log = f"{args.target_label_type} {args.dataset} {args.s_idx}->{args.t_idx} -{args.index}" \
          f" Best: H-Score: {best_h_score:.3f}, KnownAcc:{best_known_acc:.3f}, UnknownAcc:{best_unknown_acc:.3f}\t"
    log += f"CLIP H-Score:{args.CLIP_h_score:.3f}\n"
    if args.dataset == "Office" and args.s_idx == 2 and args.t_idx == 1:
        log += "\n"
    if args.dataset == "OfficeHome" and args.s_idx == 3 and args.t_idx == 2:
        log += "\n"
    if args.dataset == "VisDA":
        log += "\n"
    if args.target_label_type == "OSDA":
        path = "result_tos_ab.txt"
    elif args.target_label_type == "OPDA":
        path = "result_top_ab.txt"
    else:
        path = "result_tp_ab.txt"
    with open(path, "a") as f:
        f.write(log)


if __name__ == "__main__":
    a = build_args()
    set_random_seed(a.seed)

    # SET THE CHECKPOINT
    a.checkpoint = os.path.join("checkpoints", a.dataset, "source_{}".format(a.s_idx),
                                "source_{}_{}".format(a.source_train_type, a.target_label_type),
                                "latest_source_checkpoint.pth")
    main(a)
