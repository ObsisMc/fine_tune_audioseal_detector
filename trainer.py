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

def prepare_dataset(val_ratio=0.2, batch_size=64):
    dataset  = MyDataset("./data/train_valid", 42)
    
    indices = torch.randperm(len(dataset)).tolist()

    # Split into training and validation indices
    train_indices = indices[:int(len(indices) * (1 - val_ratio))]
    val_indices = indices[int(len(indices) * val_ratio):]

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, val_indices)

    # Then, create DataLoaders for both sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # Shuffle for training
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle validation set

    return train_loader, validation_loader

def train(model, train_loader, val_loader):
    criterion = nn.BCE()
    optimizer = optim.Adam()
    
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            # Clear the gradients
            optimizer.zero_grad()
            
            audios, srs = inputs
            
            # Forward pass
            outputs = model(audios)  # Ensure inputs are float
            loss = criterion(outputs.squeeze(), labels.float())  # Squeeze and float labels
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * audios.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Validation loop (optional, but recommended)
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
            validation_loss = 0.0
            for inputs, labels in val_loader:
                audios, srs = inputs
                
                outputs = model(audios)
                loss = criterion(outputs.squeeze(), labels.float())
                validation_loss += loss.item() * audios.size(0)
            
            validation_loss = validation_loss / len(val_loader.dataset)
            print(f'Validation Loss: {validation_loss:.4f}')

def init_model():
    pass

def main():
    pass


if __name__ == "__main__":
    test_dataset()