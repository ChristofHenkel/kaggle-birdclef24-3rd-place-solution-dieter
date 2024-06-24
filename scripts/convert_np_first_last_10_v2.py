#!/usr/bin/env python
# coding: utf-8


import torchaudio
import numpy as np
import pandas as pd
import os
import glob
import librosa
import multiprocessing as mp
from tqdm import tqdm

fns = glob.glob('/mount/birdclef24/data/birdclef-2024/train_audio/*/*.ogg')

TARGET_FOLDER = '/mount/birdclef24/data/birdclef-2024/train_audio_npy_10_v2/'


sub_folders = [item.split('/')[-1] for item in glob.glob('/mount/birdclef24/data/birdclef-2024/train_audio/*')]
for s in sub_folders:
    os.makedirs(TARGET_FOLDER + s,exist_ok=True)

SR = 32000


def do_one(fn):
    fn2 = '/'.join(fn.split('/')[-2:]).replace('.ogg','.npy')
    data = librosa.load(fn, sr=SR)[0].astype(np.float32)
    np.save(TARGET_FOLDER + fn2, data[:10*SR])



with mp.Pool(32) as p:
    res = list(tqdm(p.imap(do_one,fns)))
    
    
    
## last 10 sec

TARGET_FOLDER = '/mount/birdclef24/data/birdclef-2024/train_audio_npy_last10_v2/'

for s in sub_folders:
    os.makedirs(TARGET_FOLDER + s,exist_ok=True)
    
def do_one(fn):
    fn2 = '/'.join(fn.split('/')[-2:]).replace('.ogg','.npy')
    data = librosa.load(fn, sr=SR)[0].astype(np.float32)
    np.save(TARGET_FOLDER + fn2, data[-10*SR:])

with mp.Pool(32) as p:
    res = list(tqdm(p.imap(do_one,fns)))