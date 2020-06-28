#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import cv2
import numpy
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torchvision
from dataset import MLCDataset
from dataset import TestDataset
from model import MCLNet
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import run
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook
from PIL import Image, ImageDraw, ImageFont


# In[2]:


chn_label_dict = {
         "crowd": '人群聚集', 
        "destroy":  '破坏' ,
        "smoke": '烟',
        "fire": '火',
        "building": '政府楼',
       "fight":  '互殴',
       "rob":  '抢', 
        "religious":  '宗教',
         "speech": '演讲'}


# In[ ]:





# In[3]:


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

root_path = "/raid/ly/Dataset/crowd_protest/"
video_path = os.path.join(root_path, "video")
train_root = "/raid/zcy/crowd_protest/train/"
val_root = "/raid/zcy/crowd_protest/validation/"
#outmodel_root = "./outmodel/"
#resume_name = "best.pth"
outmodel_root = "/raid/zcy/crowd_protest/outmodel/"
resume_name = "best4.pth"
out_path_test = "/raid/zcy/crowd_protest/test_img/"
out_path_test_result_img = "/raid/zcy/crowd_protest/test_result_img/"
out_path_test_result_video = "/raid/zcy/crowd_protest/test_result_video/"


# In[4]:


# vid_names = [ "cjss1","cjss2","cjss3","cjss4","cjss5","cjss6","cjss7","cjss8","cjss9","cjss10","cjss11",
#                     "dzq1","dzq2","dzq3","dzq4","dzq5","dzq6",
#                     "ct1","ct2","ct4","ct5",
#                     "zj1","zj2","zj3","zj4","zj5","zj6","zj7","zj8"] 


# In[5]:


vid_names = [ "zj8"]


# In[6]:



test_root = "/home/zcy/crowd_protest/test_img/"


# In[7]:


num_cls = 9
num_epochs = 20
icls_dict = {0: "crowd", 1: "destroy", 2: "smoke", 
                 3: "fire", 4: "building", 5: "fight", 6: "rob", 7: "religious", 8: "speech"}
cls_dict = {"crowd" : 0, "destroy": 1, "smoke": 2, 
                "fire": 3,  "building": 4, "fight": 5, "rob": 6, "religious": 7, "speech": 8}
enc_dict = {"crowd" : [1, 0, 0, 0, 0, 0, 0, 0, 0], "destroy" : [ 0, 1, 0, 0, 0, 0, 0, 0, 0], 
                "smoke": [0, 0, 1, 0, 0, 0, 0, 0, 0], "fire": [0, 0, 0, 1, 0, 0, 0, 0, 0],
                "building" : [0, 0, 0, 0, 1, 0, 0, 0, 0], "fight": [0, 0, 0, 0, 0, 1, 0, 0, 0], "rob": [0, 0, 0, 0, 0, 0, 1, 0, 0],
                "religious" : [0, 0, 0, 0, 0, 0, 0, 1, 0], "speech": [0, 0, 0, 0, 0, 0, 0, 0, 1,]}


# In[8]:


#if resume:
checkpoint = torch.load(os.path.join(outmodel_root, resume_name))
    #start_epoch = checkpoint['epoch']
    #best_loss = checkpoint['valloss']
#else:
    #start_epoch = 0
    #best_loss = 20


# In[9]:


model_ft = MCLNet(num_cls)
#model_ft = torch.load('./outmodel/best.pth')
model_ft = model_ft.to("cuda")
#if resume:
model_ft.load_state_dict(checkpoint['model_state_dict'])
    #model.load_state_dict(torch.load(f'./outmodel/best.pth'))


# In[10]:


optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-6)
# if resume:
#     optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])


# In[11]:


criterion = nn.BCEWithLogitsLoss()


# In[12]:


#train_dataset = MLCDataset(train_root, cls_dict, enc_dict)
#val_dataset = MLCDataset(val_root, cls_dict, enc_dict, train=False)
test_dataset = TestDataset(test_root)

#train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
#val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle = False)


# In[13]:


for vid_name in os.listdir(video_path):
     if vid_name[:-4] in vid_names:
        videoCapture = cv2.VideoCapture(os.path.join(video_path, vid_name))
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
       #print("fps=",fps,"frames=",frames)

        for i in range(int(frames)):
            ret,frame_or = videoCapture.read()
            [height,width,pixels] = frame_or.shape
            frame = cv2.resize(frame_or,(256,256),interpolation=cv2.INTER_AREA)
            #cv2.imwrite("/home/zcy/crowd_protest/test_img/cjss4/cjss4_(%d).jpg"%i,frame)
            #cv2.imwrite(os.path.join(out_path_test, vid_name[:-4],"{:}_{:}.jpg".format(vid_name[:-4], i)), frame)
    
            input_img = transforms.ToTensor()(frame)    
            input_img2 = torch.unsqueeze(input_img, 0)
            #input_img2 = next(iter(test_dataloader))
            output = run.inference(model_ft, input_img2, icls_dict, 0.5)
            #print(output)
            for m, txts in enumerate(output):
                cv2img = cv2.cvtColor(frame_or, cv2.COLOR_BGR2RGB)
                pilimg = Image.fromarray(cv2img)
                draw = ImageDraw.Draw(pilimg)
                chnfont = "/raid/zcy/crowd_protest/simsun.ttc"
                font = ImageFont.truetype(chnfont, 50, encoding="utf-8")
                for j, txt in enumerate(txts):
                    w = width
                    h = height
                    draw.text((int(w/10), int(h/10*(j+1))), chn_label_dict[txt] , (255, 0, 0), font=font) 
                frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)       
            #cv2.imwrite("/home/zcy/crowd_protest/test_result_img/cjss4/cjss4_(%d).jpg"%i,frame)
                #cv2.imwrite(os.path.join(out_path_test_result_img, vid_name[:-4],"{:}_{:}.jpg".format(vid_name[:-4], i)), frame)
                
#         img2video_root = os.path.join(out_path_test_result_img, vid_name[:-4])
#         Fps = fps    
#         size=(width,height)
#         fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
#         #videoWriter = cv2.VideoWriter("/home/zcy/crowd_protest/test_result_video/cjss4_result.avi",fourcc,Fps,size)#最后一个是保存图片的尺寸
#         videoWriter = cv2.VideoWriter(os.path.join(out_path_test_result_video, "{:}_result.avi".format(vid_name[:-4])),fourcc,Fps,size)
#         for n in range(int(frames)):
#             v_frame = cv2.imread(img2video_root+vid_name[:-4]+'_'+str(n)+'.jpg')
#             videoWriter.write(v_frame)
#         videoWriter.release()


            


# In[ ]:


img2video_root = "/raid/zcy/crowd_protest/test_result_img/zj8/"
Fps = fps    
size=(width,height)
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
videoWriter = cv2.VideoWriter("/raid/zcy/crowd_protest/test_result_video/zj8_result.avi",fourcc,Fps,size)#最后一个是保存图片的尺寸
for n in range(int(frames)):
    v_frame = cv2.imread(img2video_root+'zj8_'+str(n)+'.jpg')
    videoWriter.write(v_frame)
videoWriter.release()


# In[ ]:


#type(vid_name[:-4])


# In[ ]:


height


# In[ ]:


width


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




       


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# type(output)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




