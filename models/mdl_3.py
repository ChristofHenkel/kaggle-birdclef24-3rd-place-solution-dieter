from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torch 
from torch import nn
import timm
from torch.cuda.amp import autocast
import numpy as np

from torchaudio.models import wav2vec2_model
from torch import nn
import json
import torch

class AvesTorchaudioWrapper(nn.Module):

    def __init__(self, config_path, model_path):

        super().__init__()

        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html

        self.config = self.load_config(config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load(model_path))
        self.model.feature_extractor.requires_grad_(False)

    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)

        return obj

    def forward(self, sig):
        # extract_feature in the sorchaudio version will output all 12 layers' output, -1 to select the final one
        out = self.model.extract_features(sig)[0][-1]
        out = out.mean(dim=1)  
        return out

def bce_with_mask(preds, targets):
    loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
#     loss = loss * mask
    loss = loss.mean()
    return loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.backbone = AvesTorchaudioWrapper(cfg.backbone_config_path, cfg.backbone_model_path)
        self.head = nn.Linear(self.backbone.config['encoder_embed_dim'],cfg.n_classes)
        
        self.loss_fn = bce_with_mask
        print('Net params: ',count_parameters(self))
         
    def forward(self, input_dict):
        x = input_dict['input']
        targets = input_dict['targets']
#         secondary_mask = input_dict['secondary_mask']
        
        if self.training:
            bs = x.shape[0]
            perm = torch.randperm(bs)
            weights = 0.2 + 0.8 * torch.rand(bs,device=x.device)
            x =  x + weights[:,None] * x[perm]
            targets = torch.max(targets,targets[perm])
#             secondary_mask = torch.min(secondary_mask,secondary_mask[perm])
        
        #if test then flatten bs and parts
        if len(x.shape) == 3:
            bs, parts, seq_len = x.shape
            targets = torch.repeat_interleave(targets[:,None],parts,dim=1)
#             secondary_mask = torch.repeat_interleave(secondary_mask[:,None],parts,dim=1)
            x = x.reshape(bs*parts,seq_len)#.unsqueeze(1)
            n_classes = targets.shape[-1]
            targets = targets.reshape(bs*parts,n_classes)
#             secondary_mask = secondary_mask.reshape(bs*parts,n_classes)

#         return x, targets, secondary_mask

        x = self.backbone(x)
        logits = self.head(x)
   
        result = {'logits': logits}
        loss = self.loss_fn(logits, targets)
        result.update({'loss': loss, 'target': targets})

        return result


