import cv2
import torch
import numpy as np
import tqdm
import os
import os.path as osp
from model.SFUniDA import SFUniDA
from dataset.dataset import SFUniDADataset_cam
from torch.utils.data.dataloader import DataLoader
from config.model_config import build_args

# ================= 参数 =================
args = build_args()
torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
# =======================================

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
sou_CAM_RESULT_PATH = r'cam/source/'
if not osp.exists(sou_CAM_RESULT_PATH):
    os.system('mkdir -p ' + sou_CAM_RESULT_PATH)
if not osp.exists(sou_CAM_RESULT_PATH):
    os.mkdir(sou_CAM_RESULT_PATH)

RAW_RESULT_PATH = r'cam/raw/'
if not osp.exists(RAW_RESULT_PATH):
    os.system('mkdir -p ' + RAW_RESULT_PATH)
if not osp.exists(RAW_RESULT_PATH):
    os.mkdir(RAW_RESULT_PATH)

sou_model.cuda()
sou_model.eval()
sou_model_features = torch.nn.Sequential(*list(sou_model.backbone_layer.children())[:-1])
sou_bottleneck_weights = sou_model.feat_embed_layer.state_dict()["bottleneck.weight"].cpu().numpy()
sou_fc_weights = sou_model.class_layer.state_dict()['fc.weight_v'].cpu().numpy()
print(sou_model.feat_embed_layer.state_dict())
# ==========================================


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
ours_CAM_RESULT_PATH = r'cam/ours/'
if not osp.exists(ours_CAM_RESULT_PATH):
    os.system('mkdir -p ' + ours_CAM_RESULT_PATH)
if not osp.exists(ours_CAM_RESULT_PATH):
    os.mkdir(ours_CAM_RESULT_PATH)

ours_model.cuda()
ours_model.eval()
ours_model_features = torch.nn.Sequential(*list(ours_model.backbone_layer.children())[:-1])
ours_bottleneck_weights = ours_model.feat_embed_layer.state_dict()["bottleneck.weight"].cpu().numpy()
ours_fc_weights = ours_model.class_layer.state_dict()['fc.weight_v'].cpu().numpy()
# ==========================================

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
glc_CAM_RESULT_PATH = r'cam/glc/'
if not osp.exists(glc_CAM_RESULT_PATH):
    os.system('mkdir -p ' + glc_CAM_RESULT_PATH)
if not osp.exists(glc_CAM_RESULT_PATH):
    os.mkdir(glc_CAM_RESULT_PATH)

glc_model.cuda()
glc_model.eval()
glc_model_features = torch.nn.Sequential(*list(glc_model.backbone_layer.children())[:-1])
glc_bottleneck_weights = glc_model.feat_embed_layer.state_dict()["bottleneck.weight"].cpu().numpy()
glc_fc_weights = glc_model.class_layer.state_dict()['fc.weight_v'].cpu().numpy()
# ==========================================

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
lead_CAM_RESULT_PATH = r'cam/lead/'
if not osp.exists(lead_CAM_RESULT_PATH):
    os.system('mkdir -p ' + lead_CAM_RESULT_PATH)
if not osp.exists(lead_CAM_RESULT_PATH):
    os.mkdir(lead_CAM_RESULT_PATH)

lead_model.cuda()
lead_model.eval()
lead_model_features = torch.nn.Sequential(*list(lead_model.backbone_layer.children())[:-1])
lead_bottleneck_weights = lead_model.feat_embed_layer.state_dict()["bottleneck.weight"].cpu().numpy()
lead_fc_weights = lead_model.class_layer.state_dict()['fc.weight_v'].cpu().numpy()


# ==========================================


target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
target_dataset = SFUniDADataset_cam(args, args.target_data_dir, target_data_list, d_type="test", preload_flg=True)
print(len(target_dataset))

target_test_dataloader = DataLoader(target_dataset, batch_size=args.batch_size * 2, shuffle=False,
                                    num_workers=args.num_workers, drop_last=False)


def returnCAM(feature_conv, weight_softmax, class_idx, bottleneck_weights):
    c, h, w = feature_conv.shape
    output_cam = []
    for i in class_idx:
        temp = bottleneck_weights.dot(feature_conv.reshape((c, h * w)))
        cam = weight_softmax[i].dot(temp)
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cam_img)
    return output_cam


task = ["source", "ours", "glc", "lead"]
model_features = [sou_model_features, ours_model_features, glc_model_features, lead_model_features]
model = [sou_model, ours_model, glc_model, lead_model]
fc_weights = [sou_fc_weights, ours_fc_weights, glc_fc_weights, lead_fc_weights]
bottleneck_weights = [sou_bottleneck_weights, ours_bottleneck_weights, glc_bottleneck_weights, lead_bottleneck_weights]
CAM_RESULT_PATH = [sou_CAM_RESULT_PATH, ours_CAM_RESULT_PATH, glc_CAM_RESULT_PATH, lead_CAM_RESULT_PATH]

for i in range(4):
    loop = tqdm.tqdm(target_test_dataloader, ncols=80)
    loop.set_description_str(f"CAM-{task[i]}")
    iid = 0
    with torch.no_grad():
        for img, path in loop:
            features = model_features[i](img.cuda()).detach().cpu().numpy()
            _, softmax_out = model[i](img.cuda())
            idx = softmax_out.argmax(1).cpu().numpy()

            for k in range(img.size(0)):
                CAMs = returnCAM(features[k], fc_weights[i], [idx[k]], bottleneck_weights[i])
                img = cv2.imread(path[k])
                height, width, _ = img.shape  # 读取输入图片的尺寸
                heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
                result = heatmap * 0.4 + img * 0.6

                image_name_ = "cam_img" + str(iid)
                cv2.imwrite(CAM_RESULT_PATH[i] + image_name_ + '.jpg', result)

                if i == 0:
                    image_name_ = "raw_img" + str(iid)
                    cv2.imwrite(RAW_RESULT_PATH + image_name_ + '.jpg', img)
                iid += 1
