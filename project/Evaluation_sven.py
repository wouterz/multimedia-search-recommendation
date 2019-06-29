#!/usr/bin/env python
# coding: utf-8

# # Evaluation

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import pandas as pd
import numpy as np
from src import preprocess as prep
from src.evaluation import pick_test_segments, generate_test_segments, evaluate_segments
from src.Video import Video
from src.Segment import Segment
from src import search
import random
import cv2


# ## Parameters

# In[3]:


NUM_VIDEOS = 20
GRID_SIZE = 2
BINS = [180, 180]
HIST_FRAME_SKIP = 20
REFRESH = False

# vergeet gebruikte params soms dus print ze maar afentoe
def printParams():
    print('Num. Vid {} - Grid {} - Bins {} - Skip {}'.format(NUM_VIDEOS, GRID_SIZE, BINS, HIST_FRAME_SKIP))


# ## Load training set / generate test set

# In[4]:


printParams()
training_set = prep.load_training_set(range(1, NUM_VIDEOS+1), GRID_SIZE, BINS, HIST_FRAME_SKIP, force_refresh=REFRESH)


# In[5]:


# Set of 100 custom fragments with duration 20sec
test_set, labels = generate_test_segments(training_set, n=100, duration=20)


# In[6]:


# Print statistics
print("TRAINING SET:")
print("Num. videos:   {:d}".format( len(training_set)) )
print("Num. segments: {:d}".format( np.sum([len(video.segments) for video in training_set])) )
print("Duration:      {:,.1f} s".format( np.sum([np.sum([segment.duration() for segment in video.segments]) for video in training_set])) )
print("Num frames:      {:d}".format( np.sum([np.sum([segment.num_frames() for segment in video.segments]) for video in training_set])) )
print("Num histograms:      {:d}".format( np.sum([np.sum([len(segment.histograms) for segment in video.segments]) for video in training_set])) )


# In[7]:


# Print statistics
print("TEST SET:")
print("Size: {:d}".format( len(test_set) ))


# <br>

# ## Run model on test set

# In[42]:


for method in [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.cv2.HISTCMP_INTERSECT,
               cv2.HISTCMP_BHATTACHARYYA, cv2.HISTCMP_CHISQR_ALT, cv2.HISTCMP_KL_DIV]:
    print('{}'.format(method))
    get_ipython().run_line_magic('timeit', '-n 10 search.findFrame(test_set[0][0], training_set, method)')

for ch in [[0], [1], [0, 1]]:
    print('{}'.format(ch))
    get_ipython().run_line_magic('timeit', '-n 10 search.findFrame(test_set[0][0], training_set, cv2.HISTCMP_CORREL, channels=ch)')


# In[1]:


results = []

for i, histogram in enumerate(test_set):
    print('\rSearching segment {}/{}'.format(i+1, len(test_set)), end='', flush=True)
    
#     results.append(search.find(histogram, training_set, cv2.HISTCMP_INTERSECT))
    results.append(search.findFrame(segment_histograms[0], training_set, 1, cv2.HISTCMP_CHISQR_ALT))


# ## Evaluate performance

# In[ ]:


evaluate_segments(results, labels)

