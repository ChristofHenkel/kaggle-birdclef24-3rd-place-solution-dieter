from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torch 
from torch import nn
import timm
from torch.cuda.amp import autocast



class Preprocessor(nn.Module):
    def __init__(self, cfg):
        super(Preprocessor, self).__init__()
        self.mel_ampdb = torch.nn.Sequential(MelSpectrogram(**cfg.mel_spec_args),AmplitudeToDB(top_db=cfg.top_db))
        self.m = cfg.norm_ms[0]
        self.s = cfg.norm_ms[1]
        
    def forward(self, x):
        with autocast(enabled=False), torch.no_grad():
            x = x / torch.std(x, 1, keepdim=True)
            x = x.float()
            x = self.mel_ampdb(x)
            x = (x - self.m) / self.s     
        return x
        
def bce_with_mask(preds, targets, mask):
    loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
    loss = loss * mask
    loss = loss.mean()
    return loss
        
class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.preprocessing = Preprocessor(cfg)
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            drop_rate = 0.1,
            #drop_path_rate = 0.2,
            num_classes=cfg.num_labels, 
            
            #global_pool=''
        )
        self.loss_fn = bce_with_mask

         
    def forward(self, input_dict):
        x = input_dict['input']
        targets = input_dict['targets']
        
        if len(x.shape) == 3:
            bs, parts, seq_len = x.shape
            targets = torch.repeat_interleave(targets[:,None],parts,dim=1)
#             secondary_mask = torch.repeat_interleave(secondary_mask[:,None],parts,dim=1)
            x = x.reshape(bs*parts,seq_len)#.unsqueeze(1)
            n_classes = targets.shape[-1]
            targets = targets.reshape(bs*parts,n_classes)
        
        x = self.preprocessing(x)
        with torch.no_grad():
            x = x.unsqueeze(1)
            pos = torch.linspace(0., 1., x.size(2)).to(x.device)
            pos = pos.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            pos = pos.expand(x.size(0), 1, x.size(2), x.size(3))
            x = x.expand(-1, 2, -1, -1)
            x = torch.cat([x, pos], 1)

        logits = self.backbone(x)
        
   
        result = {'logits': logits}
        
        secondary_mask = input_dict['secondary_mask']
        loss = self.loss_fn(logits, targets, secondary_mask)
        result.update({'loss': loss, 'target': targets})

        return result

class TestNet(Net):

    def forward(self,x):
        bs, parts, seq_len = x.shape
        x = x.reshape(bs*parts,seq_len)#.unsqueeze(1)
        x = self.preprocessing(x)
        with torch.no_grad():
            x = x.unsqueeze(1)
            pos = torch.linspace(0., 1., x.size(2)).to(x.device)
            pos = pos.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            pos = pos.expand(x.size(0), 1, x.size(2), x.size(3))
            x = x.expand(-1, 2, -1, -1)
            x = torch.cat([x, pos], 1)
        logits = self.backbone(x)
        
        return logits