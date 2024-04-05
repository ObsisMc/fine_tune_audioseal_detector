import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
import numpy as np
import random
import soundfile as sf
import os


class MyDataset(Dataset):
    def __init__(self, root, seed=42):
        super().__init__()
        
        self.root = root
        self.pos_dir = os.path.join(self.root, "pos") # with watermark dir
        self.neg_dir = os.path.join(self.root, "neg")  # without watermark dir
        
        self.pos_samples = [(os.path.join(self.pos_dir, f), 1) for f in os.listdir(self.pos_dir)]
        self.neg_samples = [(os.path.join(self.neg_dir, f), 0) for f in os.listdir(self.neg_dir)]
        
        self.samples = self.pos_samples + self.neg_samples
        
        self.seed = seed
        if seed is not None:
            torch.manual_seed(self.seed)  # TODO: check seed
        self.permute_id = torch.randperm(len(self.samples))
        
    
    def __getitem__(self, index):
        idx = self.permute_id[index]
        
        audio_path, label = self.samples[idx.item()]
        audio, sr = sf.read(audio_path)
        
        assert len(audio.shape) == 1
        audio = audio[None, :]
        
        return (audio, sr), label
    
    def __len__(self):
        return len(self.samples)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic=True

def test_dataset():
    dataset = MyDataset("./data/train_valid", 42)
    print(dataset.permute_id)
    
    i = 0
    (audio, sr), label = dataset[i]
    print(dataset.samples[dataset.permute_id[i]], audio.shape, sr, label)

def prepare_dataset():
    dataset  = MyDataset("./data/train_valid", 42)
    
    
    indices = torch.randperm(len(dataset)).tolist()

    # Split into training and validation indices
    train_indices = indices[:int(len(indices) * 0.8)]
    val_indices = indices[int(len(indices) * 0.8):]

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, val_indices)

    # Then, create DataLoaders for both sets
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Shuffle for training
    validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False)  # No need to shuffle validation set

def train():
    pass

    criterion = nn.BCE()
    optimizer = optim.Adam()
    




def main():
    pass


if __name__ == "__main__":
    test_dataset()