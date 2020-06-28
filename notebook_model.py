#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision
from dataset import MLCDataset
from model import MCLNet
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import run
import os


# configs

# In[2]:


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

train_root = "/home/zcy/crowd_protest/train/"
val_root = "/home/zcy/crowd_protest/validation/"
outmodel_root = "./outmodel/"
outmodel_root2 = "/home/zcy/crowd_protest/outmodel/"
resume_name = "best4.pth"

#resume = True
resume = False

num_cls = 9
num_epochs = 20
#cls_dict = {"circle" : 0, "square": 1, "triangle": 2}
#icls_dict = {0: "circle", 1: "square", 2: "triangle"}
#enc_dict = {"circle" : [1, 0, 0], "square": [0, 1, 0], "triangle": [0, 0, 1]}
icls_dict = {0: "crowd", 1: "destroy", 2: "smoke", 
                 3: "fire", 4: "building", 5: "fight", 6: "rob", 7: "religious", 8: "speech"}


cls_dict = {"crowd" : 0, "destroy": 1, "smoke": 2, 
                "fire": 3,  "building": 4, "fight": 5, "rob": 6, "religious": 7, "speech": 8}
enc_dict = {"crowd" : [1, 0, 0, 0, 0, 0, 0, 0, 0], "destroy" : [ 0, 1, 0, 0, 0, 0, 0, 0, 0], 
                "smoke": [0, 0, 1, 0, 0, 0, 0, 0, 0], "fire": [0, 0, 0, 1, 0, 0, 0, 0, 0],
                "building" : [0, 0, 0, 0, 1, 0, 0, 0, 0], "fight": [0, 0, 0, 0, 0, 1, 0, 0, 0], "rob": [0, 0, 0, 0, 0, 0, 1, 0, 0],
                "religious" : [0, 0, 0, 0, 0, 0, 0, 1, 0], "speech": [0, 0, 0, 0, 0, 0, 0, 0, 1,]}


# In[3]:


if resume:
    checkpoint = torch.load(os.path.join(outmodel_root, resume_name))
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['valloss']
else:
    start_epoch = 0
    best_loss = 80


# load dataset

# In[4]:


train_dataset = MLCDataset(train_root, cls_dict, enc_dict)
val_dataset = MLCDataset(val_root, cls_dict, enc_dict, train=False)


# In[5]:


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)


# define model

# In[6]:


model_ft = MCLNet(num_cls)
model_ft = model_ft.to("cuda")

if resume:
    model_ft.load_state_dict(checkpoint['model_state_dict'])
    #model.load_state_dict(torch.load(f'./outmodel/best.pth'))


# optimizer

# In[7]:


optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-6)
# if resume:
#     optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])


# criterion

# In[8]:


criterion = nn.BCEWithLogitsLoss()


# train loop

# In[ ]:


best_loss = 80
train_loss_lst = []
val_loss_lst = []
for epoch in range(start_epoch, num_epochs):
    train_loss = run.run_epoch(model_ft, criterion, train_dataloader, epoch, optimizer_ft)
    val_loss = run.run_epoch(model_ft, criterion, val_dataloader, epoch)
    train_loss_lst.append(train_loss)
    val_loss_lst.append(val_loss)
    
    if val_loss < best_loss:
        print("save model at epoch : {:}".format(epoch))
        best_loss = val_loss
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : model_ft.state_dict(),
            'optimizer_state_dict' : optimizer_ft.state_dict(),
            'valloss' : val_loss,
        }, os.path.join(outmodel_root2, "best4.pth"))
    
    print("epoch {:}: train - {:}".format(epoch, train_loss))
    print("epoch {:}: val - {:}".format(epoch, val_loss))


# In[ ]:





# In[ ]:


#sample_x, sample_y = next(iter(val_dataloader))


# In[ ]:


#output = run.inference(model_ft, sample_x, icls_dict, 0.9)


# In[ ]:


#output


# In[ ]:


#torchvision.transforms.ToPILImage()(sample_x[-2])


# In[ ]:


#val_loss = run.run_epoch(model_ft, criterion, val_dataloader, epoch)


# In[ ]:





# In[ ]:


#val_loss


# In[ ]:





# In[ ]:




