import os
import sys
from importlib import import_module
import platform
import json
import numpy as np
import pandas as pd

from default_config import basic_cfg as cfg
import glob

# paths

cfg.name = os.path.basename(__file__).split(".")[0]
cfg.output_dir = f"/mount/birdclef24/models/{os.path.basename(__file__).split('.')[0]}"

cfg.data_folder = f"/mount/birdclef24/data/birdclef-2024/train_audio_npy_10_v2/"
cfg.data_folder2 = f"/mount/birdclef24/data/birdclef-2024/train_audio_npy_last10_v2/"
cfg.train_df = '/mount/birdclef24/data/train_folded_v3c.csv'
cfg.test_df = '/mount/birdclef24/data/birdclef-2024/test.csv'
cfg.test_gt = '/mount/birdclef24/data/birdclef-2024/test_fake_gt.csv'
cfg.test_data_folder = '/mount/birdclef24/data/birdclef-2024/unlabeled_soundscapes/'
cfg.test_suffix = '.ogg'
cfg.test_duration = 240
cfg.birds = ['asbfly', 'ashdro1', 'ashpri1', 'ashwoo2', 'asikoe2', 'asiope1',
   'aspfly1', 'aspswi1', 'barfly1', 'barswa', 'bcnher', 'bkcbul1',
   'bkrfla1', 'bkskit1', 'bkwsti', 'bladro1', 'blaeag1', 'blakit1',
   'blhori1', 'blnmon1', 'blrwar1', 'bncwoo3', 'brakit1', 'brasta1',
   'brcful1', 'brfowl1', 'brnhao1', 'brnshr', 'brodro1', 'brwjac1',
   'brwowl1', 'btbeat1', 'bwfshr1', 'categr', 'chbeat1', 'cohcuc1',
   'comfla1', 'comgre', 'comior1', 'comkin1', 'commoo3', 'commyn',
   'compea', 'comros', 'comsan', 'comtai1', 'copbar1', 'crbsun2',
   'cregos1', 'crfbar1', 'crseag1', 'dafbab1', 'darter2', 'eaywag1',
   'emedov2', 'eucdov', 'eurbla2', 'eurcoo', 'forwag1', 'gargan',
   'gloibi', 'goflea1', 'graher1', 'grbeat1', 'grecou1', 'greegr',
   'grefla1', 'grehor1', 'grejun2', 'grenig1', 'grewar3', 'grnsan',
   'grnwar1', 'grtdro1', 'gryfra', 'grynig2', 'grywag', 'gybpri1',
   'gyhcaf1', 'heswoo1', 'hoopoe', 'houcro1', 'houspa', 'inbrob1',
   'indpit1', 'indrob1', 'indrol2', 'indtit1', 'ingori1', 'inpher1',
   'insbab1', 'insowl1', 'integr', 'isbduc1', 'jerbus2', 'junbab2',
   'junmyn1', 'junowl1', 'kenplo1', 'kerlau2', 'labcro1', 'laudov1',
   'lblwar1', 'lesyel1', 'lewduc1', 'lirplo', 'litegr', 'litgre1',
   'litspi1', 'litswi1', 'lobsun2', 'maghor2', 'malpar1', 'maltro1',
   'malwoo1', 'marsan', 'mawthr1', 'moipig1', 'nilfly2', 'niwpig1',
   'nutman', 'orihob2', 'oripip1', 'pabflo1', 'paisto1', 'piebus1',
   'piekin1', 'placuc3', 'plaflo1', 'plapri1', 'plhpar1', 'pomgrp2',
   'purher1', 'pursun3', 'pursun4', 'purswa3', 'putbab1', 'redspu1',
   'rerswa1', 'revbul', 'rewbul', 'rewlap1', 'rocpig', 'rorpar',
   'rossta2', 'rufbab3', 'ruftre2', 'rufwoo2', 'rutfly6', 'sbeowl1',
   'scamin3', 'shikra1', 'smamin1', 'sohmyn1', 'spepic1', 'spodov',
   'spoowl1', 'sqtbul1', 'stbkin1', 'sttwoo1', 'thbwar1', 'tibfly3',
   'tilwar1', 'vefnut1', 'vehpar1', 'wbbfly1', 'wemhar1', 'whbbul2',
   'whbsho3', 'whbtre1', 'whbwag1', 'whbwat1', 'whbwoo2', 'whcbar1',
   'whiter2', 'whrmun', 'whtkin2', 'woosan', 'wynlau1', 'yebbab1',
   'yebbul3', 'zitcis1']
cfg.test_epochs = 0

# stages
cfg.test = True
cfg.train = True
cfg.train_val =  False
cfg.eval_epochs = 1

#logging
cfg.neptune_project = 'XXX'
cfg.neptune_connection_mode = "async"
cfg.tags = "base"

#model
cfg.model = "mdl_3"

cfg.backbone_config_path = '/mount/birdclef24/data/aves/birdaves-biox-base.torchaudio.model_config.json'
cfg.backbone_model_path = '/mount/birdclef24/data/aves/birdaves-biox-base.torchaudio.pt'
cfg.pretrained = True
cfg.in_chans = 1
cfg.resample_train = 10

cfg.rare_birds = []

cfg.labels = np.array(cfg.birds)
cfg.targets = {v : i for i,v in enumerate(cfg.labels)}
cfg.rare_ids = np.array([cfg.targets[b] for b in cfg.rare_birds])
cfg.num_labels = len(cfg.labels)
# augmentations
cfg.resample_train = 10
cfg.other_samples = 1
cfg.max_shift = 1

cfg.n_classes = len(cfg.birds)
cfg.sample_rate = 32000
cfg.sr = cfg.sample_rate
cfg.duration = 5


# OPTIMIZATION & SCHEDULE
cfg.fold = -1
cfg.epochs = 100
cfg.lr = 5e-5
cfg.optimizer = "AdamW"
cfg.weight_decay = 0.001
cfg.clip_grad = 10.
cfg.warmup = 2
cfg.batch_size = 48
cfg.batch_size_test = 1
cfg.mixed_precision = True # True
cfg.pin_memory = False
cfg.grad_accumulation = 1.
cfg.num_workers = 8


# DATASET
cfg.dataset = "ds_3"
cfg.label_secondary = 0.5
cfg.suffix = '.npy'
cfg.eval_epochs = 20

cfg.sample_weights = False
cfg.normalization = 'channel'

#EVAL
cfg.calc_metric = False
cfg.simple_eval = False

# augs & tta

# Postprocess
cfg.post_process_pipeline =  "pp_dummy"
cfg.metric = "metric_1"
# augs & tta

#Saving
cfg.save_weights_only = True
cfg.save_only_last_ckpt = True


from augmentations import (
    CustomCompose,
    CustomOneOf,
    NoiseInjection,
    GaussianNoise,
    PinkNoise,
    AddGaussianNoise,
    AddGaussianSNR,
)

cfg.np_audio_transforms = CustomCompose(
    [
        CustomOneOf(
            [
                NoiseInjection(p=1, max_noise_level=0.04),
                GaussianNoise(p=1, min_snr=5, max_snr=20),
                PinkNoise(p=1, min_snr=5, max_snr=20),
                AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.03, p=0.5),
                AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=15, p=0.5),
            ],
            p=0.3,
        ),
    ]
)

from audiomentations import Compose as amCompose
from audiomentations import OneOf as amOneOf
from audiomentations import AddBackgroundNoise, Gain, GainTransition

bg_folder = '/mount/birdclef24/data/background/honglihang_background_noise_sec_wav/'
cfg.freefield = glob.glob(f"{bg_folder}freefield/*")
cfg.warblrb = glob.glob(f"{bg_folder}warblrb/*")
cfg.birdvox = glob.glob(f"{bg_folder}birdvox/*")
cfg.rainforest = glob.glob(f"{bg_folder}rainforest/*")

cfg.am_audio_transforms = amCompose(
    [

        amOneOf([AddBackgroundNoise(
                cfg.rainforest,
                min_snr_in_db=3.,
                max_snr_in_db=30.,
                p=0.7,),
                AddBackgroundNoise(
            cfg.freefield + cfg.warblrb + cfg.birdvox,
            min_snr_in_db=3.,
            max_snr_in_db=30.,
            p=0.35,
        )
                ],p=0.25),

        amOneOf(
            [
                Gain(min_gain_in_db=-15, max_gain_in_db=15, p=0.8),
                GainTransition(min_gain_in_db=-15, max_gain_in_db=15, p=0.8),
            ],
        ),
    ]
)


