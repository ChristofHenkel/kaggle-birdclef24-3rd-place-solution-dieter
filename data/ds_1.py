from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import torchaudio
import torch
import librosa

tr_collate_fn = None
val_collate_fn = None






#     def load_one(self, id_):
#         fp = self.cfg.data_folder + id_
#         try:
#             data, rate = torchaudio.load(fp)
#         except:
#             print("FAIL READING rec", fp)
           
#         return data[0]

class CustomDataset(Dataset):
    def __init__(self, train, cfg, aug, mode='train'):
        self.cfg = cfg
        self.df = train.copy()
        print(self.df.head())
        self.mode = mode
        self.istrain = mode=='train'
        
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
        
        self.filename = train.filename.values
        #self.primary_label = train.primary_label.values
        if not 'primary_label' in train.columns:
            train['primary_label'] = np.array(['asbfly'] * train.shape[0])
        if 'first_species' not in train.columns:
            train['first_species'] = train['primary_label'].values
            train['last_species'] = train['primary_label'].values
        if 'secondary_labels' in train.columns:
            self.secondary_labels = train.secondary_labels.values
            self.first_species = train.first_species.values
            self.last_species = train.last_species.values
        else:
            self.secondary_labels = np.array(['asbfly'] * train.shape[0])
            self.first_species = np.array(['asbfly'] * train.shape[0])
            self.last_species = np.array(['asbfly'] * train.shape[0])
            
        if self.mode == 'test':
            self.test_parts = self.duration // 5
            
        
        
    def __len__(self):
        return len(self.filename)
    
    def get_audio(self, idx):
        filename = self.filename[idx]
        duration = self.cfg.sr * self.duration
        if self.istrain:
            first = np.random.rand() < 0.5
            audio = self.load_audio(filename, first, True, self.cfg)
            if len(audio) < duration:
                pad_length = np.random.randint(0, duration - len(audio) + 1) 
                audio = np.pad(audio, 
                               ((pad_length, duration - len(audio) - pad_length),), 
                               mode='constant')
            else:
                start = np.random.randint(0, len(audio) - duration + 1)
                audio = audio[start : start + duration]
        else:
            audio = self.load_audio(filename, True, False, self.cfg)
            audio = audio[:duration]
            if len(audio) < duration:
                pad_length = (duration - len(audio)) // 2
                audio = np.pad(audio, 
                               ((pad_length, duration - len(audio) - pad_length),), 
                               mode='constant')
        return audio
      
    def __getitem__(self, idx):
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
        if self.istrain and self.cfg.other_samples:
            num_samples = np.random.randint(0, self.cfg.other_samples + 1)
            for _ in range(num_samples):
                other_idx = np.random.randint(len(self.filename))
                other_audio = self.get_audio(other_idx)
                weight = 0.2 + 0.8 *  np.random.rand()
                audio += weight * other_audio
                targets[self.cfg.targets[self.first_species[other_idx]]] = 1.0
                targets[self.cfg.targets[self.last_species[other_idx]]] = 1.0
                secondary_labels = self.secondary_labels[other_idx]
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
            'secondary_mask' : secondary_mask,
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
            elif self.suffix == '.ogg':
                audio = librosa.load(fp, sr=cfg.sr)[0].astype(np.float32)
            else:
                audio = np.load(fp)
            audio = audio[:max_duration]
        else:
            fp = f'{self.data_folder2}/{f_id}{self.suffix}'
            if cfg.suffix in ['.wav','.ogg']:
                audio, rate = torchaudio.load(fp)
                audio = audio[0].numpy()
            else:

    #         filepath = filepath / f"first10_{fname}.npy"
                audio = np.load(fp)        
            audio = audio[-max_duration:]
        return audio

def batch_to_device(batch, device):
    return {k:batch[k].to(device, non_blocking=True) for k in batch.keys() if k not in []}