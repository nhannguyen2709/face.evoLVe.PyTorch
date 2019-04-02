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


class CustomDataset(torch.utils.data.Dataset):
    """
    Custom dataset.
    """
    def __init__(self, data_root, fold, mode, transform):
        self.data_root = data_root
        self.mode = mode
        self.transform = transform

        # self.train_df = self._duplicate_low_shot_classes()
        # self.all_folds = self._kfold_split()
        # if self.mode == 'train':
        #     self.imgs, self.labels, _, _ = self.all_folds[fold]
        # elif self.mode == 'val':
        #     _, _, self.imgs, self.labels = self.all_folds[fold]
        self.train_df = pd.read_csv(os.path.join(self.data_root, "train.csv"))
        self.imgs = self.train_df["image"].values
        self.labels = self.train_df["label"].values
        if self.mode == 'test':
            self.test_df = pd.read_csv(os.path.join(self.data_root, "sample_submission.csv"))
            self.imgs = self.test_df["image"].values

    # def _duplicate_low_shot_classes(self):
    #     train_df = pd.read_csv(os.path.join(self.data_root, "train.csv"))
    #     label = train_df['label'].values
    #     classes, class_counts = np.unique(label, return_counts=True)
    #     low_shot_classes = classes[class_counts == 1]
    #     low_shot_classes = np.append(low_shot_classes, classes[class_counts == 2])

    #     for cl in low_shot_classes:
    #         single_class_df = train_df[train_df['label']==cl]
    #         single_class_df = single_class_df.sample(n=1, random_state=8)
    #         train_df = train_df.append([single_class_df], ignore_index=True)
    #     return train_df
    
    # def _kfold_split(self):
    #     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
    #     x, y = self.train_df['image'].values, self.train_df['label'].values
    #     all_folds = {}
    #     for i, (train_idx, val_idx) in enumerate(skf.split(x, y)):
    #         x_train, y_train = x[train_idx], y[train_idx]
    #         x_val, y_val = x[val_idx], y[val_idx]
    #         all_folds[i] = (x_train, y_train, x_val, y_val)
    #     return all_folds
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_file = self.imgs[index]
        img_path = os.path.join(self.data_root, "face_align/train", img_file)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.mode == 'test':
            return img, img_file
        else:
            label = self.labels[index]
            return img, label


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

    train_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
        transforms.TenCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = RGB_MEAN,
                                                     std = RGB_STD)])(crop) 
            for crop in crops]))
    ])

    dataset_train = CustomDataset(DATA_ROOT, 0, "train", train_transform)
    dataset_val = CustomDataset(DATA_ROOT, 0, "val", val_transform)
    dataset_test = CustomDataset(DATA_ROOT, 0, "test", test_transform)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = BATCH_SIZE, shuffle=True,
        pin_memory = PIN_MEMORY, num_workers = NUM_WORKERS, drop_last = DROP_LAST
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size = BATCH_SIZE, pin_memory = PIN_MEMORY,
        num_workers = NUM_WORKERS
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size = BATCH_SIZE, pin_memory = PIN_MEMORY,
        num_workers = NUM_WORKERS
    )

    NUM_CLASS = 1000
    print("Number of Training Classes: {}".format(NUM_CLASS))

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

    HEAD_DICT = {'ArcFace': ArcFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'CosFace': CosFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'SphereFace': SphereFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'Am_softmax': Am_softmax(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID)}
    HEAD = HEAD_DICT[HEAD_NAME]
    print("=" * 60)
    print(HEAD)
    print("{} Head Generated".format(HEAD_NAME))
    print("=" * 60)

    LOSS_DICT = {'Focal': FocalLoss(), 
                 'Softmax': nn.CrossEntropyLoss()}
    LOSS = LOSS_DICT[LOSS_NAME]
    print("=" * 60)
    print(LOSS)
    print("{} Loss Generated".format(LOSS_NAME))
    print("=" * 60)

    if BACKBONE_NAME.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)
    OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}, {'params': backbone_paras_only_bn}], lr = LR, momentum = MOMENTUM)
    
    output_layer_paras_only_bn, output_layer_paras_wo_bn = separate_irse_bn_paras(BACKBONE.output_layer)
    OPTIMIZER = optim.SGD(
        [{'params': output_layer_paras_wo_bn, 'weight_decay': WEIGHT_DECAY, 'lr': LR * 0.05},
         {'params': output_layer_paras_only_bn, 'lr': LR * 0.05},
         {'params': head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY, 'lr': LR}],
        lr = LR, momentum = MOMENTUM)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)

    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT and HEAD_RESUME_ROOT:
        print("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT) and os.path.isfile(HEAD_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
            print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
            HEAD.load_state_dict(torch.load(HEAD_RESUME_ROOT))
        else:
            print("No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT))
        print("=" * 60)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)


    #======= train & validation & save checkpoint =======#
    DISP_FREQ = len(train_loader) // 20 # frequency to display training loss & acc

    NUM_EPOCH_WARM_UP = NUM_EPOCH // 25  # use the first 1/25 epochs to warm up
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up
    batch = 0  # batch index

    for epoch in range(NUM_EPOCH): # start training process
        
        if epoch == STAGES[0]: # adjust LR for each training stage after warm up, you can also choose to adjust LR manually (with slight modification) once plaueau observed
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[1]:
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[2]:
            schedule_lr(OPTIMIZER)

        # for p in BACKBONE.parameters():
        #     p.requires_grad = False
        # BACKBONE.eval()

        for p in BACKBONE.module.input_layer.parameters():
            p.requires_grad = False
        for p in BACKBONE.module.body.parameters():
            p.requires_grad = False
        BACKBONE.module.input_layer.eval()
        BACKBONE.module.body.eval()
        BACKBONE.module.output_layer.train()
        HEAD.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for inputs, labels in tqdm(train_loader):

            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (batch + 1 <= NUM_BATCH_WARM_UP): # adjust LR for each training batch during warm up
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER)

            # compute output
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            features = BACKBONE(inputs)
            outputs = HEAD(features, labels)
            loss = LOSS(outputs, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk = (1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))
            top5.update(prec5.data.item(), inputs.size(0))

            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()
            
            # dispaly training loss & acc every DISP_FREQ
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                print("=" * 60)
                print('Epoch {}/{} Batch {}/{}\t'
                      'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch + 1, NUM_EPOCH, batch + 1, len(train_loader) * NUM_EPOCH, loss = losses, top1 = top1, top5 = top5))
                print("=" * 60)

            batch += 1 # batch index

        # training statistics per epoch (buffer for visualization)
        epoch_loss = losses.avg
        epoch_acc = top1.avg
        writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
        writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
        print("=" * 60)
        print('Epoch: {}/{}\t'
              'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            epoch + 1, NUM_EPOCH, loss = losses, top1 = top1, top5 = top5))
        print("=" * 60)

        # perform validation & save checkpoints per epoch
        # validation statistics per epoch (buffer for visualization)
        print("=" * 60)
        BACKBONE.eval()
        HEAD.eval()

        actual = []
        predicted = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).long()
                features = BACKBONE(inputs)
                outputs = HEAD(features, None)
                predicted.append(outputs)
                actual.append(labels)
        actual = torch.cat(actual, 0)
        predicted = torch.cat(predicted, 0)
        val_acc = accuracy_score(actual.cpu().numpy(),
            np.argmax(predicted.cpu().numpy(), 1))
        print("Epoch {}/{}, Evaluation metric: {}".format(epoch + 1, NUM_EPOCH, val_acc))
        print("=" * 60)

        # save checkpoints per epoch
        if MULTI_GPU:
            torch.save(BACKBONE.module.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch, get_time())))
            torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch, get_time())))
        else:
            torch.save(BACKBONE.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch, get_time())))
            torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch, get_time())))
    
    # compute predictions on test set
    print("=" * 60)
    print("Compute predictions on test set")
    BACKBONE.eval()
    HEAD.eval()

    img_files = []
    preds = []
    with torch.no_grad():
        for inputs, index in tqdm(test_loader):    
            bs, ncrops, c, h, w = inputs.size()
            results = BACKBONE(inputs.view(-1, c, h, w)) # fuse batch size and ncrops
            results = results.view(bs, ncrops, -1).mean(1) # avg over crops
            img_files.append(index)
            preds.append(results)
    # Extract top 5 predictions
    preds = torch.cat(preds, 0)
    top_preds, _ = torch.topk(preds, k=5, dim=1).cpu().numpy()
    submit_df = pd.concat([pd.Series(img_files), pd.Series(top_preds)], axis=1)
    submit_df.columns = ["image", "label"]
    submit_df.to_csv("retrain_backbone_output_layer.csv", index=False)