from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch 
import numpy as np
import pandas as pd
from tqdm import tqdm

import torchaudio
import ast

from torch import nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

def batch_to_device(batch,device):
    batch_dict = {key:batch[key].to(device) for key in batch}
    return batch_dict


tr_collate_fn = None
val_collate_fn = None


class CustomDataset(Dataset):

    def __init__(self, df, cfg, aug, mode='train'):

        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()
        
        self.bird2id = {bird:idx for idx,bird in enumerate(cfg.birds)}
        
        if self.mode == 'test':
            self.data_folder = cfg.test_data_folder
            self.data_folder2 = cfg.test_data_folder
            self.duration = cfg.test_duration
            self.suffix = cfg.test_suffix
        else:
            self.data_folder = cfg.data_folder
            self.data_folder2 = cfg.data_folder2
            self.duration = cfg.duration
            self.suffix = cfg.suffix
        
        if self.mode == 'train':
            self.df = self.df[self.df['rating'] >= self.cfg.min_rating]  
            self.weights = np.clip(df["rating"].values / df["rating"].max(), 0.1, 1.0)
        else:
            self.weights = np.ones(len(self.df))
        if 'primary_label' in self.df.columns:
            labels = self.df['primary_label'].map(self.bird2id).astype(int).values
            self.labels2 = self.df['secondary_labels'].apply(lambda x: self.secondary2target(x))
            self.targets = np.eye(self.cfg.n_classes)[labels]
            for i, t in enumerate(self.labels2):
                try:
                    if len(t) > 0:
                        self.targets[i,t] = 1
                except:
                    print(i)
        else:
            self.targets = np.zeros((self.df.shape[0],self.cfg.n_classes))

        #set augs
        self.aug_am = aug[0]
            
        if self.mode == 'test':
#             self.preprocessing = torch.nn.Sequential(MelSpectrogram(**cfg.mel_spec_args),AmplitudeToDB(**cfg.db_args))
#             self.norm_by = cfg.norm_by
            self.test_parts = self.cfg.wav_crop_len // self.cfg.infer_duration

            

    def crop_or_pad(self,wav):
        
        expected_length = (self.cfg.wav_crop_len * self.cfg.sample_rate)
        if wav.shape[0] < expected_length:
            pad = self.cfg.wav_crop_len * self.cfg.sample_rate - wav.shape[0]

            wav_orig = wav.clone()

            l = wav.shape[0]

            if pad >= l:
                while wav.shape[0] <= expected_length:
                    wav = torch.cat([wav, wav_orig], dim=0)
            else:
                max_offset = l - pad
                offset = np.random.randint(max_offset)
                wav = torch.cat([wav, wav_orig[offset:offset+pad]], dim=0)
        elif wav.shape[0] > expected_length:
            start = np.random.randint(0, wav.shape[0] - expected_length)
            wav = wav[start : start + expected_length]

        wav = wav[:expected_length]  
        return wav
    

    def __getitem__(self, idx):
        
        
     
        row = self.df.iloc[idx]
        fn = row['filename']
        wav_tensor = self.load_one(fn)
        if self.mode != 'test':
            wav_tensor = wav_tensor[:self.cfg.sample_rate*self.cfg.wav_max_len]
        target = self.targets[idx]
        weight = self.weights[idx]
        wav_tensor = self.crop_or_pad(wav_tensor)

        feature_dict = {'target':torch.tensor(target.astype(np.float32)),'weight':torch.tensor(weight.astype(np.float32))}
        if self.mode == 'test':
            #cut
            wav_tensor = wav_tensor.reshape(self.test_parts,wav_tensor.shape[0]//self.test_parts)
            feature_dict.update({'input':wav_tensor}) # seq_len
#             spec = self.preprocessing(wav_tensor)
#             #convert to melspec and norm
#             spec = (spec + self.norm_by) / self.norm_by
#             feature_dict.update({'input':spec}) # parts, mels, freqs
        else:
            if self.aug_am is not None:
                wav_tensor = torch.from_numpy(self.aug_am(wav_tensor.numpy(),sample_rate=self.cfg.sample_rate))
            feature_dict.update({'input':wav_tensor}) # seq_len
        

        return feature_dict
    
    def __len__(self):
        return len(self.df)


    def load_one(self, id_):
        fp = self.data_folder + id_
        f_id = fp.split('.')[0]
        try:
            if self.suffix == '.npy':
                data = torch.from_numpy(np.load(f_id + self.suffix))
            else:
                data, rate = torchaudio.load(f_id + self.suffix)
                data = data[0]
        except:
            print("FAIL READING rec", fp)
           
        return data
                             
    def secondary2target(self,secondary_label):
        birds = ast.literal_eval(secondary_label)
        target = [self.bird2id.get(item) for item in birds if not item == 'nocall']
        target = [t for t in target if not t is None]
        return target
