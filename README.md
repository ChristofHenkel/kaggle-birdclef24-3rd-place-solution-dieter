## Introduction

This repository represents Dieters part of teams NVBird solution to the BirdCLEF24 competition on kaggle.

## Installation

I used the `nvcr.io/nvidia/pytorch:24.03-py3` container from the [ngc catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) to have a consistent environment between team members. You can run it via

`docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:24.03-py3`

Within the container clone this repository and install necessary packages with 
```
git clone https://github.com/ChristofHenkel/kaggle-birdclef24-3rd-place-solution-dieter
cd kaggle-birdclef24-3rd-place-solution-dieter
pip install -r requirements.txt
```

By default training is logged via neptune.ai. You need to use your own neptune project which is set via `cfg.neptune_project` in `configs/cfg_XXX.py`


## Preparations

### npy files 

We have preprocessed the dataset into numpy format files for fast loading. 

After downlaoding the competition data to `/mount/birdclef24/data/` run `scripts/convert_np_first_last_10_v2.py` to create npy files containing wav signal of first and last 10sec of competition data.

### folds and test fiels

Run `scripts/create_train_folded_v3c.py` to process the original comeptition meta data and handle some duplicates and create folds.
Run `scripts/create_test_fake_gt.py` to enable prediction on unlabled soundscapes. 

### AVES weights
Download model weights from [AVES](https://github.com/earthspecies/aves/) and put torchaudio config and weights of `birdaves-biox-base` into `/mount/birdclef24/data/`

## background noise

Download background noise from https://www.kaggle.com/datasets/honglihang/background-noise and put into `/mount/birdclef24/data/background/`

## Training

### First round 

run the following commans to train 5 seeds each for 3 different model configs

```
5x python train.py -C cfg_1 --fold -1
5x python train.py -C cfg_2 --fold -1
5x python train.py -C cfg_3 --fold -1
```

This results in 15 csv files pl_df_SEED in respective model folders. Simply average predictions from 3 models each to create 5x different 

```
pl_blended_0.csv 
pl_blended_1.csv 
pl_blended_2.csv 
pl_blended_3.csv 
pl_blended_4.csv 
```

needed for round 2

### Second round


run 

```
python train.py -C cfg_pl_1 --fold -1 --pl_df pl_blended_0.csv 
python train.py -C cfg_pl_1 --fold -1 --pl_df pl_blended_1.csv 
python train.py -C cfg_pl_1 --fold -1 --pl_df pl_blended_2.csv 
python train.py -C cfg_pl_1 --fold -1 --pl_df pl_blended_3.csv 
python train.py -C cfg_pl_1 --fold -1 --pl_df pl_blended_4.csv 
```

to get final model weights

## Inference

see public notebook