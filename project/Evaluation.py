#!/usr/bin/env python
# coding: utf-8

# # Evaluation

# In[189]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[190]:


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

# In[252]:


NUM_VIDEOS = 100
GRID_SIZE = 3
BINS = [180, 256]
HIST_FRAME_SKIP = 20
REFRESH = False

# vergeet gebruikte params soms dus print ze maar afentoe
def printParams():
    print('Num. Vid {} - Grid {} - Bins {} - Skip {}'.format(NUM_VIDEOS, GRID_SIZE, BINS, HIST_FRAME_SKIP))


# ## Load training set / generate test set

# In[253]:


printParams()
training_set = prep.load_training_set(range(1, NUM_VIDEOS+1), GRID_SIZE, BINS, HIST_FRAME_SKIP, force_refresh=REFRESH)


# In[254]:


# Set of 100 custom fragments with duration 20sec
test_set, labels = generate_test_segments(training_set, n=100, duration=20)


# In[255]:


# Print statistics
print("TRAINING SET:")
print("Num. videos:   {:d}".format( len(training_set)) )
print("Num. segments: {:d}".format( np.sum([len(video.segments) for video in training_set])) )
print("Duration:      {:,.1f} s".format( np.sum([np.sum([segment.duration() for segment in video.segments]) for video in training_set])) )
print("Num frames:      {:d}".format( np.sum([np.sum([segment.num_frames() for segment in video.segments]) for video in training_set])) )
print("Num histograms:      {:d}".format( np.sum([np.sum([len(segment.histograms) for segment in video.segments]) for video in training_set])) )

print("TEST SET:")
print("Size: {:d}".format( len(test_set) ))


# # Small manual test

# In[264]:


pr = False
for i in range(10):
#     x = random.choice(range(len(test_set[i])))
    found = search.findFrame(test_set[i][0], training_set, cv2.HISTCMP_CHISQR, 2, prints= pr, warnings=pr)
    print('Found {} - Expected {}'.format(found, labels[i]))


# ## Run model on test set

# In[261]:


for method in [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.cv2.HISTCMP_INTERSECT,
               cv2.HISTCMP_BHATTACHARYYA, cv2.HISTCMP_CHISQR_ALT, cv2.HISTCMP_KL_DIV]:
    get_ipython().run_line_magic('timeit', '-n 10 search.findFrame(test_set[0][0], training_set, method, warnings = False)')

# for ch in [[0], [1], [0, 1]]:
#     print('{}'.format(ch))
#     %timeit -n 10 search.findFrame(test_set[0], training_set, cv2.HISTCMP_CORREL, channels=ch)


# In[262]:


results = []

for i, histogram in enumerate(test_set):
    print('\rSearching segment {}/{} - Histograms {}'.format(i+1, len(test_set), len(histogram), end='', flush=True))
    
    
    
    results.append(search.findFrame(histogram[0], training_set, cv2.HISTCMP_CHISQR_ALT, 2, warnings = False))


# ## Evaluate performance

# In[267]:


evaluate_segments(results, labels)


# In[ ]:




