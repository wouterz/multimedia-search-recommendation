#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[10]:


import pandas as pd
import numpy as np

from src.VideoReader import VideoReader
from src.Video import Video
from src.histograms import compute_histograms
from src import preprocess as prep
import cv2
import os


# In[151]:


def find(target, videos):
    video_path = os.path.join(target + ".mp4")
    target_video = VideoReader()
    target_video.open(video_path)
    frames = target_video.get_frames()
    
#     todo: average per novelty section in video?
    first_frame = list(frames)[videos[0].segments[2].frame_start]
    
    target_histograms = compute_histograms(first_frame)
    
    distances = []
    for video in videos:
        segment_dist = []       
        for segment in video.segments:
            dist = 0
            for h in range(len(segment.histograms[0])):
#                   Method kiezen, voor nu intersection
                dist += cv2.compareHist(target_histograms[0][h], segment.histograms[0][h], cv2.HISTCMP_INTERSECT)
            segment_dist.append(dist)
        
        distances.append(segment_dist)
    

    best_dist_indices = []
    for d in distances:
        best_dist_indices.append(np.argpartition(d, -5)[-5:])
    
    sub_distances = []
    i = 0
    for video in videos:
        segment_dist = []       
        for segment_index in best_dist_indices[i]:
            segment = video.segments[segment_index]
            
            dist = 0
            for i_sub_hist in range(1,5):
                nr_hists = len(target_histograms[0])
                for h in range(nr_hists):
                    dist += cv2.compareHist(target_histograms[i_sub_hist][h], segment.histograms[i_sub_hist][h], cv2.HISTCMP_INTERSECT)
                
            segment_dist.append(dist)    
        sub_distances.append(segment_dist)
        i = i + 1

    # Find index of maximum value in matrix
    result = np.where(sub_distances == np.amax(sub_distances))
    print(result)
    match_vid = result[0][0]
    match_seg = best_dist_indices[result[0][0]][result[1][0]]
    print('video {} - segment {}'.format(match_vid, match_seg))


    


# In[152]:


# vid1 = prep.process_video("00001")
# vid2 = prep.process_video("00002")


# In[153]:


find(r'data/movies/00001', [vid1, vid2])


# In[ ]:




