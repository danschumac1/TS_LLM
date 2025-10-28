import sys
sys.path.append("./src/")
sys.path.append("./src/method/classification/InstructTime/TStokenizer")
import warnings
warnings.filterwarnings('ignore')

import os
import torch
import torch.utils.data as Data

import random
import numpy as np

from dataset import Dataset
from args import args
from process import Trainer
from model import VQVAE

from utils.logging_utils import MasterLogger

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def main():
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(args.data_path)
    logger = MasterLogger(log_path='./logs/InstructTime/TStokenizer.log')
    seed_everything(seed=2023)

    train_dataset = Dataset(device=args.device, mode='train', args=args)
    print(f"train_dataset.ecgs_images.shape: {train_dataset.ecgs_images.shape}")
    
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    args.data_shape = train_dataset.shape()
    print(f"data shape: {args.data_shape}")
    test_dataset = Dataset(device=args.device, mode='test', args=args)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
    logger.info(f"train len: {len(train_dataset)}, test len: {len(test_dataset)}")

    VQVAE_config = (
        args.data_shape,   # (seq_len, feat_dim) e.g., (561, 1)
        args.d_model,
        args.n_embed,
        args.block_num,
        args.wave_length,
    )

    model = VQVAE(*VQVAE_config).to(args.device)



    logger.info(f"VQVAE_config:\n{VQVAE_config}")

    model = VQVAE(
        *VQVAE_config
    ).to(args.device)
    
    print('model initial ends')

    trainer = Trainer(args, model, train_loader, test_loader, verbose=True)
    print('trainer initial ends')

    trainer.train()


if __name__ == '__main__':
    main()

