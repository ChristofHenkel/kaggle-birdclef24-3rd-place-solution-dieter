#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchaudio
import torch
import pandas as pd
import numpy as np


# In[2]:


import glob
import multiprocessing as mp
from tqdm import tqdm


# In[3]:


df = pd.read_csv('/mount/birdclef24/data/birdclef-2024/test.csv')
df


# In[5]:


def do_one(fn):
    data, sr = torchaudio.load('/mount/birdclef24/data/birdclef-2024/unlabeled_soundscapes/' + fn)
    return data.shape[1] / sr


# In[7]:


with mp.Pool(32) as p:
    res = list(tqdm(p.imap(do_one,df['filename'].values)))


# In[8]:


df['len'] = res
df['len'] = df['len'].clip(0,240)


fns=[]
secs = []
for i in range(df.shape[0]):
    fn, l = df.iloc[i]
    sec = list(range(5,int(l + 5),5))
    secs += sec
    fns += [fn] * len(sec)


df2 = pd.DataFrame({'row_id':[str(fn.replace('.ogg','')) + '_' + str(s) for fn, s in zip(fns,secs)]})


df2['asbfly'] = 0
df2.loc[:df2.shape[0]//2,'asbfly'] = 1


df2.to_csv('/mount/birdclef24/data/birdclef-2024/test_fake_gt.csv',index=False)


# In[5]:


fns = sorted(glob.glob('/mount/birdclef24/data/birdclef-2024/unlabeled_soundscapes/*'))
fns[:5]


# In[10]:


test = pd.DataFrame({'filename':[fn.split('/')[-1] for fn in fns]})


# In[11]:


test


# In[12]:


test.to_csv('/mount/birdclef24/data/birdclef-2024/test.csv',index=False)




