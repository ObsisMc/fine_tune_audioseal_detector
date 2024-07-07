import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
import numpy as np
import random
import soundfile as sf
import os
import time
import argparse

from audioseal import AudioSeal


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
    print(len(dataset))
    print(dataset.permute_id[:10])
    
    for i in range(10):
        (audio, sr), label = dataset[i]
        print(dataset.samples[dataset.permute_id[i]], audio.shape, sr, label)


def prepare_dataset(dataset_path, val_ratio=0.2, batch_size=64):
    dataset  = MyDataset(dataset_path, 42)
    
    indices = torch.randperm(len(dataset)).tolist()

    # Split into training and validation indices
    train_num = int(len(indices) * (1 - val_ratio))
    train_indices = indices[:train_num]
    val_indices = indices[train_num:]

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, val_indices)

    # Then, create DataLoaders for both sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # Shuffle for training
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle validation set

    return train_loader, validation_loader


def train(model, train_loader, val_loader, num_epochs, device):
    start_time = time.strftime("%Y%m%d%H%M%S",time.localtime(time.time()))
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    
    
    best_valid_loss = torch.inf
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            # Clear the gradients
            optimizer.zero_grad()
            
            audios, srs = inputs
            audios = audios.float().to(device)
            labels = labels.long().to(device)
            
            # Forward pass
            outputs, _ = model(audios, srs[0].item())  # (B,C,T)
            labels = labels[:, None].repeat(1, outputs.shape[-1])  # (B,T)
            loss = criterion(outputs.log(), labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * audios.size(0)
            
            if i != 0 and i % 100 == 0:
                print(f"Train iter {i} in Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Validation loop (optional, but recommended)
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
            validation_loss = 0.0
            for i, (inputs, labels) in enumerate(val_loader):
                audios, srs = inputs
                audios = audios.float().to(device)
                labels = labels.long().to(device)
                
                outputs, _ = model(audios, srs[0].item())
                labels = labels[:, None].repeat(1, outputs.shape[-1])  # (B,T)
                loss = criterion(outputs.log(), labels)
                validation_loss += loss.item() * audios.size(0)
                
                if i != 0 and i % 100 == 0:
                    print(f"Valid iter {i}, Loss: {loss.item():.4f}")
            
            validation_loss = validation_loss / len(val_loader.dataset)
            print(f'Validation Loss: {validation_loss:.4f}')
            
            # scheduler.step(validation_loss)
            
            if validation_loss < best_valid_loss:
                best_valid_loss = validation_loss
                best_epoch = epoch + 1
                
                if not os.path.exists("./checkpoints"):
                    os.makedirs("./checkpoints")
                
                # torch.save({
                #     'epoch': epoch + 1,
                #     'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'loss': validation_loss,
                # }, f'./checkpoints/model_best_val_loss_epoch{best_epoch}_{time.strftime("%Y%m%d%H%M%S",time.localtime(time.time()))}.pth')
                torch.save(model.state_dict(), f'./checkpoints/model_best_val_loss_{start_time}.pth')
                print("Saved checkpoint for epoch {} with validation loss: {:.4f}".format(epoch+1, validation_loss))

            
            
            
def init_model(exp: bool=False):
    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    
    if exp:
        # change last layers
        detector.detector[1] = torch.nn.Sequential(
                detector.detector[1],
                torch.nn.ReLU(),
                torch.nn.Conv1d(detector.detector[1].out_channels, detector.detector[1].out_channels, 1)
            ) 
    
    # froze params
    for name, param in detector.named_parameters():
        if "detector.1" not in name:
            param.requires_grad = False
    
    return detector

def main(dataset_path, num_epochs=2000, exp: bool=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    detector = init_model(exp)
    detector = detector.to(device)
    
    train_loader, valid_loader = prepare_dataset(dataset_path, val_ratio=0.2, batch_size=128)
    train(detector, train_loader, valid_loader, num_epochs=num_epochs, device=device)


if __name__ == "__main__":
    
    # test_dataset()
    
    # train_loader, valid_loader = prepare_dataset()
    # print(len(train_loader.dataset), len(valid_loader.dataset))
    
    # detector = init_model()
    
    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', required=True)
        parser.add_argument("--epochs", default=1500)
        parser.add_argument("--exp", action="store_true", required=False, default=False)
        
        return parser
    
    parser = get_parser()
    args = parser.parse_args()
    
    # dataset = "data/generated_audios_noneft_audiosealft/train_valid"
    # num_epochs = 1500
    # exp = False
    main(dataset_path=args.dataset, num_epochs=args.epochs, exp=args.exp)
    