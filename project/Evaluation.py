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
from src.evaluation import evaluate_segments
import random
from src.Video import Video
from src.Segment import Segment
from src import search
import cv2


# ## Parameters

# In[55]:


NUM_VIDEOS = 20
GRID_SIZE = 2
BINS = [180, 180]
HIST_FRAME_SKIP = 20
REFRESH = True


# ## Load training set

# In[416]:


training_set = prep.load_training_set(range(1, NUM_VIDEOS+1), GRID_SIZE, BINS, HIST_FRAME_SKIP, force_refresh=REFRESH)


# In[417]:


# Print statistics
print("TRAINING SET:")
print("Num. videos:   {:d}".format( len(training_set)) )
print("Num. segments: {:d}".format( np.sum([len(video.segments) for video in training_set])) )
print("Duration:      {:,.1f} s".format( np.sum([np.sum([segment.duration() for segment in video.segments]) for video in training_set])) )
print("Num frames:      {:d}".format( np.sum([np.sum([segment.num_frames() for segment in video.segments]) for video in training_set])) )
print("Num histograms:      {:d}".format( np.sum([np.sum([len(segment.histograms) for segment in video.segments]) for video in training_set])) )


# ## Select random test set

# In[57]:


test_n_segments = 100
test_set = []
labels = []

for i in range(test_n_segments):
    
    # Find random video
    video = random.choice(training_set)
    
    # Select random segment and add histogram to test set
    segment = random.choice(video.segments)
    test_set.append(segment.histograms)
    labels.append(segment)


# In[63]:


# Print statistics
print("TEST SET:")
print("Num. histograms: {:d}".format( np.sum([len(histogram) for histogram in test_set]) ))


# <br>

# ## Run model on test set

# In[78]:


get_ipython().run_line_magic('timeit', 'search.findFrame(test_set[0], 0, training_set, cv2.HISTCMP_CORREL)')
get_ipython().run_line_magic('timeit', 'search.findFrame(test_set[0], 0, training_set, cv2.HISTCMP_CHISQR)')
get_ipython().run_line_magic('timeit', 'search.findFrame(test_set[0], 0, training_set, cv2.HISTCMP_INTERSECT)')
get_ipython().run_line_magic('timeit', 'search.findFrame(test_set[0], 0, training_set, cv2.HISTCMP_BHATTACHARYYA)')
get_ipython().run_line_magic('timeit', 'search.findFrame(test_set[0], 0, training_set, cv2.HISTCMP_CHISQR_ALT)')
get_ipython().run_line_magic('timeit', 'search.findFrame(test_set[0], 0, training_set, cv2.HISTCMP_KL_DIV)')

get_ipython().run_line_magic('timeit', 'search.findFrame(test_set[0], 0, training_set, cv2.HISTCMP_KL_DIV, 5)')

# %timeit search.find(test_set[0], training_set, cv2.HISTCMP_CORREL)
# %timeit search.find(test_set[0], training_set, cv2.HISTCMP_CHISQR)
# %timeit search.find(test_set[0], training_set, cv2.HISTCMP_INTERSECT)
# %timeit search.find(test_set[0], training_set, cv2.HISTCMP_BHATTACHARYYA)
# %timeit search.find(test_set[0], training_set, cv2.HISTCMP_CHISQR_ALT)
# %timeit search.find(test_set[0], training_set, cv2.HISTCMP_KL_DIV)


# In[67]:


results = []

for i, histograms in enumerate(test_set):
    print('\rSearching segment {}/{} - Frames in segment: {}'.format(i+1, len(test_set), len(histograms)), end='', flush=True)

#     results.append(search.find(histogram, training_set, cv2.HISTCMP_INTERSECT))
    results.append(search.findFrame(histograms,  0, training_set, cv2.HISTCMP_INTERSECT))


# ## Evaluate performance

# In[79]:


evaluate_segments(results, labels)


# In[37]:


hists = training_set[0].segments[5].histograms
display(len(hists))
# 0 = altijd perfect match
tf = 1
search.findFrame(hists, tf, training_set, cv2.HISTCMP_INTERSECT, prints=True)


# In[ ]:




