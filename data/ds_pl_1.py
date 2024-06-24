from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import torchaudio
import torch
import librosa
import pandas as pd
import ast


def collate_fn(batch):
    
    new_d = {}
    k = batch[0].keys()
    for k in batch[0].keys():
        new_d[k] = torch.cat([b[k] for b in batch])
        
    return new_d

tr_collate_fn = collate_fn
val_collate_fn = None



def upsample(df, cfg):
    new_train = []
    for species, sub_df in df.groupby('primary_label'):
        if len(df) < cfg.resample_train:
            n = np.ceil(cfg.resample_train/len(sub_df)).astype(int)
            sub_df = pd.concat([sub_df] * n)
        new_train += [sub_df]
    new_train = pd.concat(new_train).reset_index(drop=True)  
    return new_train  

class CustomDataset(Dataset):
    def __init__(self, train, cfg, aug, mode='train'):
        self.cfg = cfg
        self.mode = mode
        if self.mode == 'train':
            self.df = upsample(train.copy(), cfg)
            
            pl_df = pd.read_csv(cfg.pl_df)
            pl_df['file_id'] = pl_df['row_id'].apply(lambda x: '_'.join(x.split('_')[:-1]))
            pl_df['sec'] = pl_df['row_id'].apply(lambda x: int(x.split('_')[-1]))
            vcs = pl_df['file_id'].value_counts()
            file_id60 = vcs[vcs == cfg.parts].index.values
            pl_df = pl_df[pl_df['file_id'].isin(file_id60)]
            self.pl_files = pl_df['file_id'].unique()
            self.pl_df = pl_df.groupby('file_id')
            self.w = torch.ones(len(self.df))/len(self.df)
            
        else:
            self.df = train.copy()
            self.pl_df = None
            
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
        
        self.istrain = mode=='train'
        
        self.filename = train.filename.values

        
        #self.primary_label = train.primary_label.values
        if 'primary_label' in train.columns:
            if 'first_species' not in train.columns:
                train['first_species'] = train['primary_label'].values
            if 'last_species' not in train.columns:
                train['last_species'] = train['primary_label'].values
                
                
            
            self.secondary_labels = [ast.literal_eval(item) for item in train.secondary_labels.values]
            self.first_species = train.first_species.values
            self.last_species = train.last_species.values
        else:
            self.secondary_labels = np.array([cfg.birds[0]] * train.shape[0])
            self.first_species = np.array([cfg.birds[0]] * train.shape[0])
            self.last_species = np.array([cfg.birds[0]] * train.shape[0])
            
        if self.mode == 'test':
            self.test_parts = self.duration // 5
            
        
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.filename) // 64
        else:
            return len(self.filename)
    
    def get_audio(self, idx):
        filename = self.filename[idx]
        duration = self.cfg.sr * self.duration
        if self.istrain:
            first = np.random.rand() < 0.5
            audio = self.load_audio(filename, first, True, self.cfg)
            if len(audio) < duration:
                pad_length = np.random.randint(0, duration - len(audio) + 1) 
                audio = np.pad(audio, ((pad_length, duration - len(audio) - pad_length),), mode='constant')
            else:
                start = np.random.randint(0, len(audio) - duration + 1)
                audio = audio[start : start + duration]
        else:
            audio = self.load_audio(filename, True, False, self.cfg)
            audio = audio[:duration]
            if len(audio) < duration:
                pad_length = (duration - len(audio)) // 2
                audio = np.pad(audio, ((pad_length, duration - len(audio) - pad_length),), mode='constant')
        return audio
      
    def __getitem__(self, idx):
        
        
     

        if self.istrain:
            
            inner_bs = 64
            idxs = torch.multinomial(self.w,num_samples=inner_bs)
            audio = np.stack([self.get_audio(idx2) for idx2 in idxs])

            targets = np.zeros((inner_bs,len(self.cfg.labels)), dtype=np.float32) 
            first_species = self.first_species[idxs]
            mask = np.where(self.cfg.labels[None,:] == first_species[:,None])
            targets[mask] = 1

            last_species = self.last_species[idxs]
            mask = np.where(self.cfg.labels[None,:] == last_species[:,None])
            targets[mask] = 1

            secondary_mask = np.ones((inner_bs,len(self.cfg.labels)), dtype=np.float32)
            secondary_labelss = [self.secondary_labels[j] for j in idxs]
            for i, secondary_labels in enumerate(secondary_labelss):
                if len(secondary_labels) > 0:
                    for label in secondary_labels:
                        if label in self.cfg.targets:
                            secondary_mask[i,self.cfg.targets[label]] = 0   
            
            pl_idx = torch.randint(low=0,high=len(self.pl_files),size=(1,))
            pl_file_id = self.pl_files[pl_idx]
            pl_audio = self.load_pl_audio(pl_file_id,self.cfg)
            pl_target = self.pl_df.get_group(pl_file_id).sort_values('sec')[self.cfg.birds].values
            pl_audio = pl_audio[:5*pl_target.shape[0]*self.cfg.sr]
            if pl_audio.shape[0] < 5*pl_target.shape[0]*self.cfg.sr:
                pl_audio_padded = np.zeros(5*pl_target.shape[0]*self.cfg.sr)
                pl_audio_padded[:pl_audio.shape[0]] = pl_audio.shape[0]
                pl_audio = pl_audio_padded
            pl_audio = pl_audio.reshape(pl_target.shape[0], -1)
            pl_secondary = np.ones_like(pl_target)
            
            audio = np.concatenate([audio,pl_audio])
            targets = np.concatenate([targets,pl_target])
            secondary_mask = np.concatenate([secondary_mask,pl_secondary])
            
            mixup = (np.random.rand() < self.cfg.mixup_p)
            if mixup:
                bs = targets.shape[0]
                perm = torch.randperm(bs)
                weight = 0.1 ** (self.cfg.db_range *  np.random.rand() / 10)
                audio = audio + weight * audio[perm]
                secondary_mask = np.minimum(secondary_mask, secondary_mask[perm])
                targets = np.maximum(targets, targets[perm])
                secondary_mask = np.maximum(secondary_mask, targets)               
            
            
#             if (torch.rand(1) < cfg.mixup_p):
#                 pl_idx = torch.randint(len(self.pl_df))
                
#                 pl_audio = self.load_pl_audio(self.pl_df['row_id'].values[pl_idx], self.cfg)
#                 pl_target = self.pl_df[self.cfg.birds].values[pl_idx]
#                 weight = 0.1 ** (self.cfg.db_range *  np.random.rand() / 10)
                
#                 audio += weight * pl_audio
#                 targets = np.maximum(targets, pl_target)
#                 secondary_mask = torch.maximum(secondary_mask, targets)   
                
#             num_samples = np.random.randint(0, self.cfg.other_samples + 1)
#             for _ in range(num_samples):
#                 other_idx = np.random.randint(len(self.filename))
#                 other_audio = self.get_audio(other_idx)
#                 weight = 0.2 + 0.8 *  np.random.rand()
#                 audio += weight * other_audio
#                 targets[self.cfg.targets[self.first_species[other_idx]]] = 1.0
#                 targets[self.cfg.targets[self.last_species[other_idx]]] = 1.0
#                 secondary_labels = self.secondary_labels[other_idx]
#                 if len(secondary_labels) > 0:
#                     for label in secondary_labels:
#                         if label in self.cfg.targets:
#                             secondary_mask[self.cfg.targets[label]] = 0

        else:
            audio = self.get_audio(idx)
            targets = np.zeros(len(self.cfg.labels), dtype=np.float32)        
            targets[self.cfg.targets[self.first_species[idx]]] = 1.0
            targets[self.cfg.targets[self.last_species[idx]]] = 1.0
            secondary_mask = np.ones(len(self.cfg.labels), dtype=np.float32)
            secondary_labels = self.secondary_labels[idx]
            if len(secondary_labels) > 0:
                for label in secondary_labels:
                    if label in self.cfg.targets:
                        secondary_mask[self.cfg.targets[label]] = 0        
            
        secondary_mask = np.maximum(secondary_mask, targets) 
        
        wav_tensor = torch.from_numpy(audio)
        if self.mode == 'test':
            #cut
            wav_tensor = wav_tensor.reshape(self.test_parts,wav_tensor.shape[0]//self.test_parts)
        
        out = {
            'input' : wav_tensor,
            'targets' : torch.from_numpy(targets),
            'secondary_mask' : torch.from_numpy(secondary_mask),
        }
        return out
    
    def load_audio(self, filename, first, istrain, cfg):
        f_id = filename.split('.')[0]
        if istrain:
            max_duration =  int((self.duration + cfg.max_shift) * cfg.sr)
        else:
            max_duration = self.duration * cfg.sr
        if first:
            fp = f'{self.data_folder}/{f_id}{self.suffix}'
            if self.suffix == '.wav':
                audio, rate = torchaudio.load(fp)
                audio = audio[0].numpy()
            elif self.suffix in ['.ogg','.flac']:
                audio = librosa.load(fp, sr=cfg.sr)[0].astype(np.float32)
            else:
                audio = np.load(fp)
            audio = audio[:max_duration]
        else:
            fp = f'{self.data_folder2}/{f_id}{self.suffix}'
            if self.suffix in ['.wav','.ogg']:
                audio, rate = torchaudio.load(fp)
                audio = audio[0].numpy()
            else:

    #         filepath = filepath / f"first10_{fname}.npy"
                audio = np.load(fp)        
            audio = audio[-max_duration:]
        return audio
    
    def load_pl_audio(self, filename, cfg):
        fp = f'{cfg.pl_data_folder}/{filename}{cfg.test_suffix}'
        if cfg.test_suffix in ['.ogg','.flac']:
            audio = librosa.load(fp, sr=cfg.sr)[0].astype(np.float32)
        else:
            audio = np.load(fp)
        return audio

def batch_to_device(batch, device):
    return {k:batch[k].to(device, non_blocking=True) for k in batch.keys() if k not in []}
