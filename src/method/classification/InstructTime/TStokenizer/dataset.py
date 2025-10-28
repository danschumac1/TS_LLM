import torch
import torch.utils.data as Data
from args import Train_data, Test_data

class Dataset(Data.Dataset):
    def __init__(self, device, mode, args):
        self.args = args
        if mode == 'train':
            self.ecgs_images = Train_data
        else:
            self.ecgs_images = Test_data
        self.device = device
        self.mode = mode

    def __len__(self):
        return len(self.ecgs_images)

    def __getitem__(self, item):
        ecg = torch.tensor(self.ecgs_images[item], dtype=torch.float32)
        ecg = ecg.unsqueeze(-1)  # (561,) -> (561, 1)
        return ecg * 2.5
    # def shape(self):
    #     print(f"self.ecgs_images.shape: {self.ecgs_images.shape}")
    #     print("Taking 0th dimension as dataset size")
    #     return self.ecgs_images.shape[0]
    def shape(self):
        shape = self.ecgs_images.shape
        if len(shape) == 2:
            # (N, L) → univariate
            seq_len = shape[1]
            feat_dim = 1
        elif len(shape) == 3:
            # (N, L, D) → multivariate
            seq_len = shape[1]
            feat_dim = shape[2]
        else:
            raise ValueError(f"Unexpected data shape {shape}")
        return (seq_len, feat_dim)
