import torch

BATCH_SIZE = 32
EPOCH = 30
LR = 0.001
DOWNLOAD_TRAIN_DATA = True
DOWNLOAD_TEST_DATA = True

if torch.cuda.is_available():
    gpu = '0'
else:
    gpu = ''
if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")
