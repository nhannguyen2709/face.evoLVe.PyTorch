import torch


configurations = {
    1: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/media/pc/6T/jasonjzhao/data/faces_emore', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/media/pc/6T/jasonjzhao/buffer/model', # the root to buffer your checkpoints
        LOG_ROOT = '/media/pc/6T/jasonjzhao/buffer/log', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = './', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = './', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 125, # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [35, 65, 95], # epoch stages to decay learning rate

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
),
    2: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = './', # the parent root where your train/val/test data are stored
        MODEL_ROOT = 'src/checkpoint/', # the root to buffer your checkpoints
        LOG_ROOT = 'src/log/', # the root to log your train/val status
        # RESUME = 'src/checkpoint/se_resnext50_32x4d_ArcFace_epoch_1_iter_50.pth', # the path to resume training from a saved checkpoint
        RESUME = None,

        BACKBONE_NAME = 'se_resnext50_32x4d', # support: [...]
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [256, 648], # support: [...]
        RGB_MEAN = [0.485, 0.456, 0.406], # for normalize inputs to [-1, 1]
        RGB_STD = [0.229, 0.224, 0.225],
        EMBEDDING_SIZE = 2048, # feature dimension
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        
        # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        DEVICE = "cuda",
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        DISTRIBUTED = True,
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids
        PIN_MEMORY = True,
        NUM_WORKERS = 4,
        
        DISP_FREQ = 20,
        CHECKPOINT_PERIOD = 5000,

        # LR = 5e-5, # initial LR
        # BATCH_SIZE = 16,
        # NUM_EPOCH = 12, # total epoch number
        # STAGES = [8, 10], # epoch stages to decay learning rate
        # WARMUP_ITERS = 500,
        # MILESTONES = [480000, 600000]
        
        LR = 1e-4, # initial LR
        BATCH_SIZE = 32,
        NUM_EPOCH = 6, # total epoch number
        STAGES = [4, 5], # epoch stages to decay learning rate
        WARMUP_ITERS = 500,
        MILESTONES = [240000, 300000]
),
}
