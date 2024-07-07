import torch
import os
import soundfile as sf
import tqdm
import argparse

from audioseal import AudioSeal


def test_model(model_path, audio_paths, labels, exp=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if exp:
        detector = AudioSeal.load_detector_exp(model_path).to(device)
    else:
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
    
    avg_prob = sum(probs) / len(probs)
    avg_acc = sum(acc) / len(acc)
    print(f"Avg prob: {avg_prob}")
    print(f"Avg acc: {avg_acc}")
    return avg_prob, avg_acc
    
    
def test_ckpt(ckpt_path):
    model = torch.load(ckpt_path)
    print(model)
    # print(model["model_state_dict"])
    # torch.save(model["model_state_dict"], "checkpoints/model_best_val_loss_epoch89_20240405021648.pth")
    
    
if __name__ == "__main__":
    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', required=True)
        parser.add_argument("--ckpt_name", required=True, help="ckpt_name is in src/audioseal/cards, should be without .yaml")
        parser.add_argument("--exp", action="store_true", required=False, default=False)
        
        return parser
    
    parser = get_parser()
    args = parser.parse_args()
    
    # roots = ["data/generated_audios_noneft_audiosealft/test_ood_prompts/neg/", "data/generated_audios_noneft_audiosealft/test_ood_prompts/pos/"]
    roots = [f"{args.dataset}/neg/", f"{args.dataset}/pos/"]
   
    model_name = "model_best_val_loss_20240411022941"
    model_name = args.ckpt_name
    
    accus = [] 
    for i in range(len(roots)):
        audio_root = roots[i]
        audio_paths = [os.path.join(audio_root, f) for f in os.listdir(audio_root)]
        
        labels = [i] * len(audio_paths)
        prob, acc = test_model(model_name, 
                                audio_paths,
                                labels,
                                exp=args.exp)
        accus.append(acc)
    print(accus)
    print(f"Acg acc: {sum(accus) / len(accus)}")
    
    # test_ckpt("checkpoints/model_best_val_loss_epoch89_20240405021648.pth")