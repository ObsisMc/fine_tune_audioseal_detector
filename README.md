# Fine Tune Detector of Audioseal

Use your own data to fine tune the detector [audioseal](https://github.com/facebookresearch/audioseal) by modifying the last output layer of the detector.

The task is binary classification. The training data is all `.wav` and it has binary label, with/without watermark.

## Install
follow the instruction of [audioseal](https://github.com/facebookresearch/audioseal)


## Dataset

Your data should be in `./data` and follow the following structure

```yaml
- data/
  - your_dataset_path/
    - neg/
      - xxx.wav
      - ...
    - pos/
      - yyy.wav
      - ...
```
- `your_dataset_path`: your dataset directory, and it has two directory `neg` and `pos`
- `neg`: contains audio without watermarks
- `pos`: contains audio with watermarks

## Fine Tuning
```shell
python trainer.py --dataset data/your_dataset_name --epochs 1500
```
Then, you get a checkpoint in `checkpoints/`. Create a `.yaml` in `src/audioseal/cards/` and the content is the same as `audioseal_detector_16bits.yaml` except for the checkpoint field which should be your checkpoint path.

## Test
```shell
python tester.py --dataset data/your_dataset_name --ckpt_name your_ckpt_name
```

Then, you will get average accuarcy of the detection. 