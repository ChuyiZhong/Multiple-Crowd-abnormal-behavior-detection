#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
from dataset import MLCDataset


# In[2]:


data_root = "/home/zcy/crowd_protest/train/"


# In[3]:


cls_dict = {"crowd" : 0, "destroy": 1, "smoke": 2, 
                "fire": 3,  "building": 4, "fight": 5, "rob": 6, "religious": 7, "speech": 8}
enc_dict = {"crowd" : [1, 0, 0, 0, 0, 0, 0, 0, 0], "destroy" : [ 0, 1, 0, 0, 0, 0, 0, 0, 0], 
                "smoke": [0, 0, 1, 0, 0, 0, 0, 0, 0], "fire": [0, 0, 0, 1, 0, 0, 0, 0, 0],
                "building" : [0, 0, 0, 0, 1, 0, 0, 0, 0], "fight": [0, 0, 0, 0, 0, 1, 0, 0, 0], "rob": [0, 0, 0, 0, 0, 0, 1, 0, 0],
                "religious" : [0, 0, 0, 0, 0, 0, 0, 1, 0], "speech": [0, 0, 0, 0, 0, 0, 0, 0, 1,]}


# In[4]:


train_dataset = MLCDataset(data_root, cls_dict, enc_dict)


# In[5]:


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)


# In[6]:


x, y = next(iter(train_dataloader))


# In[7]:


train_dataset.ann_pd.image_id.tolist()


# In[8]:


torchvision.transforms.ToPILImage()(x[0])


# In[9]:


y


# In[ ]:




