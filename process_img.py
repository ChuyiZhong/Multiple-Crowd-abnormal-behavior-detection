#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook


# In[2]:


root_path = "/raid/ly/Dataset/crowd_protest/"
video_path = os.path.join(root_path, "video")
label_path = os.path.join(root_path, "labels/raw_label.xlsx")
out_path = os.path.join(root_path, "labeledImgs")
out_label_path = os.path.join(root_path, "video_label")


# In[3]:


vid_names = ["cjss1"]


# In[4]:


def process_df(df, fr):
    label_dict = {
        '人群近' : 0,
        '人群中' : 1,
        '人群远':2, 
        '破坏' : 3,
        '烟':4,
        '火':5,
        '政府楼':6,
        '互殴':7,
        '抢':8,
        '宗教':9,
        '演讲':10}
    
    label_dict = {
        '人群近' : "crowd_n",
        '人群中' : "crowd_i",
        '人群远': "crowd_f", 
        '破坏' : "destroy",
        '烟': "smoke",
        '火': "fire",
        '政府楼': "building",
        '互殴': "fight",
        '抢': "rob",
        '宗教': "religious",
        '演讲': "speech"}
    
    tag_dict = {}
    tag_enc_dict = {}
    time_dict = {}
    for row in df.iterrows():
        time_step = row[1][0]
        tags = row[1][1:].tolist()
        tag_dict[time_step] = tags
        tag_code = []
        for tag in tags:
            if tag in label_dict.keys():
                tag_code.append(label_dict[tag])
        tag_enc_dict[time_step] = tag_code
        time_dict[time_step] = [cvt_time(t, fr) for t in time_step.split("-")]
        
    return tag_dict, tag_enc_dict, time_dict


# In[5]:


def cvt_time(t, fr):
    m, s = t.split(":")
    t_s = (int(m) * 60 + int(s)) * fr
    return t_s


# In[6]:


vid_label_dict = {}
for vid_name in os.listdir(video_path):
    vid = cv2.VideoCapture(os.path.join(video_path, vid_name))
    fr = vid.get(cv2.CAP_PROP_FPS)
    num_fr = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    if vid_name[:-4] in vid_names:
        # load label
        df = pd.read_excel(open(label_path, 'rb'), sheet_name=vid_name[:-4], header=None)
        # preprocess label
        tag_dict, tag_enc_dict, time_dict = process_df(df, fr)
        # generate label
        vid_label = {}
        for fid in range(int(num_fr)):
            vid_label[fid] = []
            for tstep, trng in time_dict.items():
                if (fid >= trng[0]) & (fid < trng[1]):
                    vid_label[fid] = [x for x in tag_enc_dict[tstep] if isinstance(x, str)]                
       # np.save(os.path.join(out_label_path, '{:}.npy'.format(vid_name[:-4])), vid_label)

        
        
        # save image & label
        vid = cv2.VideoCapture(os.path.join(video_path, vid_name))
        for fid in tqdm_notebook(range(int(num_fr))):
            ret, frame = vid.read()
            h,w, _ = frame.shape
            # cv2.imwrite(os.path.join(out_path, "{:}-{:}.jpg".format(vid_name[:-4], str(fid).zfill(5))), frame)
            if len(vid_label[fid]) > 0:
                for i, txt in enumerate(vid_label[fid]):
                    cv2.putText(frame, txt,
                                (int(w/10), int(h/10*(i+1))), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                int(h / 180), (0, 0 ,255), 4)
            #cv2.imwrite(os.path.join(out_path, "{:}-{:}-labeled.jpg".format(vid_name[:-4], str(fid).zfill(5))), frame)
            
        


# In[10]:


vid_label[fid]


# In[7]:


vid_labe


# In[ ]:




