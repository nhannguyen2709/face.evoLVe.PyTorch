import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from data import make_data_loader

from config import configurations
from model_imagenet import ResNet
from loss.focal import FocalLoss
from util.utils import accuracy
from utils.comm import get_rank, get_world_size, synchronize
from utils.logger import setup_logger
from utils.metric_logger import MetricLogger
from lr_scheduler import WarmupMultiStepLR

import argparse
import datetime
import logging
import os   
from tqdm import tqdm
import time


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()


def reduce_loss(loss):
    world_size = get_world_size()
    with torch.no_grad():
        dist.reduce(loss, dst=0)
        if dist.get_rank() == 0:
            loss /= world_size
    return loss


if __name__ == '__main__':

    #======= hyperparameters & data loaders =======#
    cfg = configurations[2]

    torch.manual_seed(cfg['SEED'])
    
    if cfg['DISTRIBUTED']:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    logger = setup_logger("landmark", cfg['LOG_ROOT'], get_rank())
    logger.info("Overall configurations \n{}".format(cfg))

    train_loader = make_data_loader(is_distributed=cfg['DISTRIBUTED'])
    num_class = train_loader.dataset.num_class
    logger.info("Number of training classes: {}".format(num_class))

    #======= model & loss & optimizer =======#
    model = ResNet(cfg, num_class)
    logger.info("{} backbone generated".format(cfg['BACKBONE_NAME']))    
    logger.info("{} head generated".format(cfg['HEAD_NAME'])) 
    loss_dict = {'Focal': FocalLoss(), 
                 'Softmax': nn.CrossEntropyLoss()}
    loss_function = loss_dict[cfg['LOSS_NAME']]
    logger.info("{} loss generated".format(cfg['LOSS_NAME']))
    
    # create optimizer
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg['LR']
        if "bias" in key and "bn" not in key:
            lr = cfg['LR'] * 2
        params += [{"params": [value], "lr": lr}]
    optimizer = torch.optim.Adam(params, cfg['LR'])
    scheduler = WarmupMultiStepLR(optimizer, 
        milestones=cfg['MILESTONES'],
        warmup_iters=cfg['WARMUP_ITERS'])
    
    # optionally resume from a checkpoint
    if cfg['RESUME']:
        checkpoint = torch.load(cfg['RESUME'], map_location="cpu")
        logger.info("Loading model's state dict from checkpoint {}".format(
            cfg['RESUME']))
        model.load_state_dict(checkpoint.pop("state_dict"))
        logger.info("Loading optimizer from checkpoint {}".format(
            cfg['RESUME']))
        optimizer.load_state_dict(checkpoint.pop("optimizer"))
        logger.info("Loading scheduler from checkpoint {}".format(
            cfg['RESUME']))
        scheduler.load_state_dict(checkpoint.pop("optimizer"))
    else:
        logger.info("No checkpoint found")

    if cfg['MULTI_GPU']:
        # multi-GPU setting
        if cfg['DISTRIBUTED']:
            model = model.to(cfg['DEVICE'])
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank,
                # this should be removed if we update BatchNorm stats
                # broadcast_buffers=False,
            )
        else:
            model.to(cfg['DEVICE'])
            model = torch.nn.parallel.DataParallel(
                backbone, device_ids=cfg['GPU_ID'])
    else:
        # single-GPU setting
        model = model.to(cfg['DEVICE'])
    
    #======= train & validation & save checkpoint =======#
    logger.info("Start training")
    for epoch in range(cfg['NUM_EPOCH']): # start training process        
        model.train()  # set to training mode
        meters = MetricLogger(delimiter=" ") 
        max_iter = len(train_loader)
        end = time.time()

        for iteration, (inputs, labels) in enumerate(train_loader):
            data_time = time.time() - end
            iteration = iteration + 1
            scheduler.step()

            # compute output
            inputs = inputs.to(cfg['DEVICE'])
            labels = labels.long().to(cfg['DEVICE'])
            outputs = model(inputs, labels)
            loss = loss_function(outputs, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk = (1, 5))
            loss_reduced = reduce_loss(loss)
            prec1_reduced = reduce_loss(prec1)
            meters.update(loss=loss_reduced, top1=prec1_reduced)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            # display training loss & acc every DISP_FREQ
            if iteration % cfg['DISP_FREQ'] == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
            )
            # save checkpoints
            if iteration % cfg['CHECKPOINT_PERIOD'] == 0 and get_rank() == 0:
                checkpoint = {
                    'state_dict': model.module.state_dict() if cfg['MULTI_GPU'] else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                checkpoint_file = "{}_{}_epoch_{}_iter_{}.pth".format(
                    cfg['BACKBONE_NAME'], cfg['HEAD_NAME'], epoch + 1, iteration)
                checkpoint_path = os.path.join(cfg['MODEL_ROOT'], checkpoint_file)
                logger.info("Saving checkpoint to {}".format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)            