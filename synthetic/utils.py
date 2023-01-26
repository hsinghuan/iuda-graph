import torch
import numpy as np
import random
import os

def set_model_seed(model_seed):
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed(model_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(model_seed)
    random.seed(model_seed)
    os.environ['PYTHONHASHSEED'] = str(model_seed)

def get_device(gpuID:int):
    if torch.cuda.is_available():
        device = "cuda:" + str(gpuID)
    else:
        device = "cpu"
    return device