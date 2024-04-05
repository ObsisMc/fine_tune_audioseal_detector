import torch
import os
import soundfile as sf
import tqdm
from audioseal import AudioSeal


def test_model(model_path, audio_paths, labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    detector = AudioSeal.load_detector(model_path).to(device)
    
    acc = []
    probs = []
    
    bar = tqdm.tqdm(total=len(audio_paths))
    for p, lb in zip(audio_paths, labels):
        audio, sr = sf.read(p)
        audio = torch.tensor(audio)[None, None, :].float().to(device)
    
        result, message = detector.detect_watermark(audio, sr)
        probs.append(result)
        acc.append(result < 0.5 if lb == 0 else result > 0.5)
        
        bar.update()
        
    print(f"Avg prob: {sum(probs) / len(probs)}")
    print(f"Avg acc: {sum(acc) / len(acc)}")
    
    
def test_ckpt(ckpt_path):
    model = torch.load(ckpt_path)
    print(model)
    # print(model["model_state_dict"])
    # torch.save(model["model_state_dict"], "checkpoints/model_best_val_loss_epoch89_20240405021648.pth")
    
    
if __name__ == "__main__":
    
    audio_paths = ["data/test/pos/audio_90.wav"]
    audio_root = "data/test/pos/"
    audio_paths = [os.path.join(audio_root, f) for f in os.listdir(audio_root)]
    labels = [1] * len(audio_paths)
    test_model("model_best_val_loss_epoch76_20240405024411", 
               audio_paths,
               labels)
    
    # test_ckpt("checkpoints/model_best_val_loss_epoch89_20240405021648.pth")