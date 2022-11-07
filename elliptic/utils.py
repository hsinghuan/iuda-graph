import torch
import random
import numpy as np

def set_model_seed(random_seed:int):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_device(gpuID:int):
    if torch.cuda.is_available():
        device = "cuda:" + str(gpuID)
    else:
        device = "cpu"
    return device