import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import scipy as sp
import pandas as pd



from torcheval.metrics.functional import binary_auroc

import time

    

def calc_metric(cfg, pp_out, val_df, pre="val"):
    
    preds = pp_out['logits'].cpu().numpy()
    if pre == 'test':
        sub_df = val_df.copy()
        parts = cfg.test_duration // 5
        n = sub_df.shape[0]
        sub_df = sub_df.loc[sub_df.index.repeat(parts)].copy().reset_index(drop=True)
        sub_df['sec'] = np.arange(5,(parts+1)*5,5)[None,:].repeat(n,axis=0).reshape(-1)
        sub_df['row_id'] = sub_df['filename'].apply(lambda x: x.split('.')[0]).astype(str) + '_' + sub_df['sec'].astype(str)
        preds_df = pd.DataFrame(preds,columns=cfg.birds)
        sub_df = pd.concat([sub_df[['row_id']],preds_df], axis=1)
        
        test_gt = pd.read_csv(cfg.test_gt)
        
        row_ids = np.intersect1d(sub_df['row_id'],test_gt['row_id'])
        sub_df2 = sub_df.set_index('row_id')
        test_gt2 = test_gt.set_index('row_id')
        
        sub_df2 = sub_df2.loc[row_ids]
        
        sub_df2.to_csv(f'{cfg.output_dir}/fold{cfg.fold}/pl_df_{cfg.seed}.csv')
        test_gt2 = test_gt2.loc[row_ids]
        
        preds = sub_df2[test_gt2.columns].values
        if np.isnan(preds).sum() > 0:
            print(f'replaceing {np.isnan(preds).sum()} with 0')
            preds = np.nan_to_num(preds)
        target = test_gt2.values        
    
    else:
        target = (pp_out['target'] > 0.5).float().cpu().numpy()
        
    good_idx = target.sum(0) > 0

#     s = time.time()
    num_tasks = good_idx.sum()
    if num_tasks > 1:
        score = binary_auroc(torch.from_numpy(preds[:,good_idx].T).cuda().float(),torch.from_numpy(target[:,good_idx].T).cuda(),num_tasks=num_tasks).mean()
    else:
        score = binary_auroc(torch.from_numpy(preds[:,good_idx].T[0]).cuda().float(),torch.from_numpy(target[:,good_idx].T[0]).cuda(),num_tasks=num_tasks).mean()

    return score

