import asyncio
import http.client
import json
import logging
import os
import random
import sys
from openai import OpenAI
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPTokenizer

from .prompt_pool import PromptGenerator


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def log_args(args):
    s = "\n==========================================\n"

    s += ("python" + " ".join(sys.argv) + "\n")

    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)

    s += "==========================================\n"

    return s


def set_logger(args, log_name="train_log.txt"):
    # creating logger.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # file logger handler
    if args.test:
        # Append the test results on existing logging file.
        file_handler = logging.FileHandler(os.path.join(args.save_dir, log_name), mode="a")
        file_format = logging.Formatter("%(message)s")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_format)
    else:
        # Init the logging file.
        file_handler = logging.FileHandler(os.path.join(args.save_dir, log_name), mode="w")

        file_format = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_format)

    # terminal logger handler
    terminal_handler = logging.StreamHandler()
    terminal_format = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    terminal_handler.setLevel(logging.INFO)
    terminal_handler.setFormatter(terminal_format)

    logger.addHandler(file_handler)
    logger.addHandler(terminal_handler)
    if not args.test:
        logger.debug(log_args(args))

    return logger


def compute_h_score(args, class_list, gt_label_all, pred_cls_all, open_flag=True, open_thresh=0.5, pred_unc_all=None):
    # class_list:
    #   :source [0, 1, ..., N_share - 1, ...,           N_share + N_src_private - 1]
    #   :target [0, 1, ..., N_share - 1, N_share + N_src_private + N_tar_private -1]
    # gt_label_all [N]
    # pred_cls_all [N, C]
    # open_flag    True/False
    # pred_unc_all [N], if exists. [0~1.0]

    per_class_num = np.zeros((len(class_list)))
    per_class_correct = np.zeros_like(per_class_num)
    pred_label_all = torch.max(pred_cls_all, dim=1)[1]  # [N]

    if open_flag:
        cls_num = pred_cls_all.shape[1]

        if pred_unc_all is None:
            # If there is not pred_unc_all tensor,
            # We normalize the Shannon entropy to [0, 1] to denote the uncertainty.
            pred_unc_all = Entropy(pred_cls_all) / np.log(cls_num)  # [N]

        unc_idx = torch.where(pred_unc_all > open_thresh)[0]
        pred_label_all[unc_idx] = cls_num  # set these pred results to unknown

    for i, label in enumerate(class_list):
        label_idx = torch.where(gt_label_all == label)[0]
        correct_idx = torch.where(pred_label_all[label_idx] == label)[0]
        per_class_num[i] = float(len(label_idx))
        per_class_correct[i] = float(len(correct_idx))

    per_class_acc = per_class_correct / (per_class_num + 1e-5)

    if open_flag:
        known_acc = per_class_acc[:-1].mean()
        unknown_acc = per_class_acc[-1]
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
    else:
        known_acc = per_class_correct.sum() / (per_class_num.sum() + 1e-5)
        unknown_acc = 0.0
        h_score = 0.0

    return h_score, known_acc, unknown_acc, per_class_acc


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, targets, applied_softmax=True):
        """
        Args:
            inputs: prediction matrix (after softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size, num_classes).
        """
        if applied_softmax:
            log_probs = torch.log(inputs)
        else:
            log_probs = self.logsoftmax(inputs)

        if inputs.shape != targets.shape:
            # this means that the target data shape is (B,)
            targets = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)

        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss


def load_llm_classes(args):
    folder_path = os.path.join("envisioned_classes", f"{args.dataset}")
    test_labels = args.test_labels
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print('Envisioning Outlier Exposure...')
    file_path = os.path.join(folder_path, f"{args.L}_{args.llm_model}_{args.index}.json")

    if not os.path.exists(file_path):
        asyncio.run(obtain_gpt_class_and_save(args, file_path, test_labels))

    print('=== load json: ', file_path)
    gpt_class_dict = load_json(file_path)
    print('Get Envisioned Candidate Class Names.')

    gpt_class = []
    for key, value in gpt_class_dict.items():
        gpt_class.extend(value)

    # remove repeat
    gpt_class = [item.lower() for item in gpt_class]
    gpt_class = list(set(gpt_class))
    print('After set: ', len(gpt_class))

    test_labels = [item.lower() for item in test_labels]
    gpt_class = [item for item in gpt_class if item.lower() not in test_labels]
    print('After test set: ', len(gpt_class))

    return gpt_class


async def obtain_gpt_class_and_save(args, file_path, class_list):
    prompt_gen = PromptGenerator()
    if args.L < 50:
        envision_nums = args.L
        envision_times = 0
    else:
        envision_nums = 50
        envision_times = max(int(args.L / envision_nums) - 1, 0)
    print(f"class num: {args.class_num}, class list: {class_list}")
    print(envision_nums)

    prompts = prompt_gen.get_prompt(args.class_num, class_info=class_list, envision_nums=envision_nums)
    prefix = "Before my question, I will give you an example; please strictly follow my answer format(just use '-' before every answer) and just give me the answer (the answer starts with -), no other words!!!  Do not include my questions and examples in your answer.\n"
    prompts = prefix + prompts
    response_texts = get_completion(args, prompts)
    # print(response_texts)
    context = []
    context.append({"role": "user", "content": prompts})
    context.append({"role": "assistant", "content": response_texts})

    for i in range(envision_times):
        new_prompt = prompt_gen.get_prompt_again(envision_nums=envision_nums)
        generated_text, context = get_completion_from_messages(args, context, new_prompt)
        response_texts = response_texts + generated_text

    # print(response_texts)
    descriptors_list = stringtolist(response_texts)
    descriptors = {"unknown": descriptors_list}

    # print(response_texts)
    with open(file_path, 'w') as fp:
        json.dump(descriptors, fp)


def load_json(filename):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)


# def get_completion(args, prompts):
#     conn = http.client.HTTPSConnection(args.url)
#     payload = json.dumps({
#         "model": args.llm_model,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompts
#             }
#         ],
#         "stream": False
#     })
#     headers = {
#         'Authorization': args.api_key,
#         'Content-Type': 'application/json'
#     }
#     conn.request("POST", "/v1/chat/completions", payload, headers)
#     res = conn.getresponse()
#     data = res.read()
#     print(json.loads(data.decode("utf-8"))['model'])
#     return json.loads(data.decode("utf-8"))['choices'][0]['message']['content']


def get_completion(args, prompts):
    client = OpenAI(
        base_url=args.url,
        api_key=args.api_key
    )
    completion = client.chat.completions.create(
        model=args.llm_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompts}
        ]
    )
    return completion.choices[0].message.content


def get_completion_from_messages(args, context, new_prompt):
    conn = http.client.HTTPSConnection(args.url)
    context.append({"role": "user", "content": new_prompt})
    payload = json.dumps({
        "model": args.llm_model,
        "messages": context,
        "stream": False
    })
    headers = {
        'Authorization': args.api_key,
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/chat/completions", payload, headers)
    data = conn.getresponse().read()
    print(data.decode("utf-8"))
    generated_text = json.loads(data.decode("utf-8"))['choices'][0]['message']['content']
    context.append({"role": "assistant", "content": generated_text})
    return generated_text, context


def stringtolist(description):
    return [descriptor[2:] for descriptor in description.split('\n') if
            (descriptor != '') and (descriptor.startswith('- '))]


def test_scores_clip(args, net, loader, gpt_labels, softmax=False):
    test_labels = args.test_labels
    id_class_nums = len(test_labels)
    tokenizer = CLIPTokenizer.from_pretrained(args.ckpt)

    gpt_labels = remove_overlap_class(test_labels, gpt_labels)
    total_features, known_features = pre_filter(net, tokenizer, test_labels, gpt_labels)
    total_features /= total_features.norm(dim=-1, keepdim=True)

    total_correct = 0
    total_unk = 0
    total_samples = 0

    tqdm_object = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for batch_idx, (_, images, target, _) in enumerate(tqdm_object):
            images = images.cuda()
            target = target.cuda()
            
            image_features = net.get_image_features(pixel_values=images).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            output = image_features @ total_features.T
            output = F.softmax(output / args.t, dim=1)
            
            # 获取硬预测
            _, pred = torch.max(output, dim=1)
            
            pred = torch.where(pred >= id_class_nums, 
                            torch.tensor(-1).cuda(),  # 使用-1表示预测为未知
                            pred)
            
            # 统计所有样本
            batch_size = target.size(0)
            total_samples += batch_size
            total_unk += (pred == -1).sum().item()

            correct_mask = pred == target
            total_correct += correct_mask.sum().item()

    # 计算总体准确率
    accuracy = total_correct / (total_samples + 1e-5)
    
    # 输出结果
    s = ""
    if args.s_idx == 0 and args.t_idx == 1:
        s += f"数据集：{args.dataset} 任务类型：{args.target_label_type} 提示类别索引:{args.index}\n"
    
    with open("TMM_MR/clip_llm.txt", "a") as f:
        f.write(s)
        f.write(f"任务：{args.s_idx}->{args.t_idx} Accuracy: {accuracy:.3f} ({total_correct}/{total_samples}, 误分成未知:{total_unk})\n")
    
    return accuracy


def get_ood_scores_clip(args, net, loader, gpt_labels, softmax=False):
    test_labels = args.test_labels
    to_np = lambda x: x.data.cpu().numpy()
    _score = []
    k_score = []
    j_score = []
    max_score = []
    id_class_nums = len(test_labels)
    tokenizer = CLIPTokenizer.from_pretrained(args.ckpt)

    gpt_labels = remove_overlap_class(test_labels, gpt_labels)
    total_features, known_features = pre_filter(net, tokenizer, test_labels, gpt_labels)
    total_features /= total_features.norm(dim=-1, keepdim=True)
    known_features /= known_features.norm(dim=-1, keepdim=True)

    tqdm_object = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for batch_idx, (_, images, _, _) in enumerate(tqdm_object):
            images = images.cuda()

            image_features = net.get_image_features(pixel_values=images).float()  # 500x512
            image_features /= image_features.norm(dim=-1, keepdim=True)

            output = image_features @ total_features.T

            k_output = image_features @ known_features.T
            k_output = F.softmax(k_output / args.t, dim=1)

            j_smax = to_np(F.softmax(output / args.t, dim=1))

            if softmax:
                smax = to_np(F.softmax(output, dim=1))
            else:
                smax = to_np(output)

            smax = np.max(smax[:, :id_class_nums], axis=1) - args.beta * np.max(smax[:, id_class_nums:], axis=1)
            j_smax = np.max(j_smax[:, :id_class_nums], axis=1) - args.beta * np.max(j_smax[:, id_class_nums:], axis=1)

            _score.append(torch.tensor(smax))
            k_score.append(k_output)
            j_score.append(torch.tensor(j_smax))
            max_score.append(k_output.max(1)[0])
        clip_pred = torch.sort(torch.cat(k_score, dim=0), descending=True)[1].T
        max_score = torch.cat(max_score, dim=0)
        j_score = torch.cat(j_score, dim=0)
    return torch.cat(_score, dim=0), clip_pred, max_score, j_score
    # clip zero shot test retyrn
    return torch.cat(_score, dim=0), torch.cat(k_score, dim=0), max_score, j_score


def remove_overlap_class(test_labels, gpt_labels):
    words_set = {word for phrase in test_labels for word in phrase.split()}
    gpt_labels = [phrase for phrase in gpt_labels if not any(word in words_set for word in phrase.split())]
    return gpt_labels


def pre_filter(net, tokenizer, test_labels, gpt_labels):
    test_labels_inputs = tokenizer([f"a photo of a {c}" for c in test_labels], padding=True, return_tensors="pt")
    test_labels_features = net.get_text_features(input_ids=test_labels_inputs['input_ids'].cuda(),
                                                 attention_mask=test_labels_inputs['attention_mask'].cuda()).float()

    gpt_labels_inputs = tokenizer([f"a photo of a {c}" for c in gpt_labels], padding=True, return_tensors="pt")
    gpt_labels_features = net.get_text_features(input_ids=gpt_labels_inputs['input_ids'].cuda(),
                                                attention_mask=gpt_labels_inputs['attention_mask'].cuda()).float()

    cosine_sim = torch.empty((gpt_labels_features.shape[0], test_labels_features.shape[0]))
    for i in range(gpt_labels_features.shape[0]):
        cosine_sim[i] = F.cosine_similarity(gpt_labels_features[i].unsqueeze(0), test_labels_features, dim=1)

    threshold = 1.0
    mask = (cosine_sim > threshold).any(dim=1)

    if threshold >= 1:
        # print(cosine_sim[cosine_sim > 1])
        assert torch.all(~mask), "pre_filter: error in mask"

    gpt_labels_features_filtered = gpt_labels_features[~mask]
    return torch.cat((test_labels_features, gpt_labels_features_filtered), dim=0), test_labels_features
