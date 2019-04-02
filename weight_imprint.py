import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from config import configurations
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from util.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy

from collections import OrderedDict
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os


if __name__ == '__main__':

    #======= hyperparameters & data loaders =======#
    cfg = configurations[2]

    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    MODEL_ROOT = cfg['MODEL_ROOT'] # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT'] # the root to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME'] # support: ['Focal', 'Softmax']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST'] # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR'] # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES'] # epoch stages to decay learning rate

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)

    writer = SummaryWriter(LOG_ROOT) # writer for buffering intermedium results

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])
    NUM_CLASS = 1000
    #======= model & loss & optimizer =======#
    BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE), 
                     'ResNet_101': ResNet_101(INPUT_SIZE), 
                     'ResNet_152': ResNet_152(INPUT_SIZE),
                     'IR_50': IR_50(INPUT_SIZE), 
                     'IR_101': IR_101(INPUT_SIZE), 
                     'IR_152': IR_152(INPUT_SIZE),
                     'IR_SE_50': IR_SE_50(INPUT_SIZE), 
                     'IR_SE_101': IR_SE_101(INPUT_SIZE), 
                     'IR_SE_152': IR_SE_152(INPUT_SIZE)}
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    # load pretrained weights
    # PRETRAINED_BACKBONE = "backbone_ir50_ms1m_epoch120.pth" 
    # PRETRAINED_BACKBONE = "backbone_ir50_asia.pth"
    PRETRAINED_BACKBONE = "backbone_ir50_ms1m_epoch63.pth"
    BACKBONE.load_state_dict(torch.load(os.path.join(MODEL_ROOT, PRETRAINED_BACKBONE)))
    # weight imprinting
    train_df = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
    pl_test_df = pd.read_csv(os.path.join(DATA_ROOT, "pseudo_label_test.csv"))
    train_img = "train/" + train_df['image']
    pl_test_img = "test/" + pl_test_df['image']
    img_files =  np.append(train_img.values, pl_test_img.values)
    labels = np.append(train_df['label'].values, pl_test_df['label'].values)
    weight = []

    BACKBONE.to(DEVICE)
    BACKBONE.eval()
    with torch.no_grad():
        for i in tqdm(range(NUM_CLASS)):
            class_img_files = img_files[labels == i]
            # print("\nClass {} with samples {}".format(i, class_imgs))
            class_weight_vector = torch.zeros(1, EMBEDDING_SIZE).to(DEVICE)
            if len(class_img_files) == 1:
                img_path = os.path.join(DATA_ROOT, "face_align", class_img_files[0])
                img = Image.open(img_path).convert("RGB")
                img = val_transform(img)
                img = img.unsqueeze_(0)
                img = img.to(DEVICE)
                class_weight_vector += BACKBONE(img)
            else:
                for img_file in class_img_files:
                    img_path = os.path.join(DATA_ROOT, "face_align", img_file)
                    img = Image.open(img_path).convert("RGB")
                    img = val_transform(img)
                    img = img.unsqueeze_(0)
                    img = img.to(DEVICE)
                    class_weight_vector += BACKBONE(img) / len(class_img_files)
            weight.append(class_weight_vector)
    weight = torch.cat(weight, 0)
    new_head_state_dict = OrderedDict()
    new_head_state_dict["weight"] = weight.cpu()
    torch.save(new_head_state_dict, os.path.join(MODEL_ROOT, PRETRAINED_BACKBONE.replace("backbone", "head")))