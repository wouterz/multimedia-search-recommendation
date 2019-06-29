#!/usr/bin/env python
# coding: utf-8

# # Evaluation

# In[74]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[75]:


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

# In[76]:


NUM_VIDEOS = 2
GRID_SIZE = 2
BINS = [180, 180]
HIST_FRAME_SKIP = 20
REFRESH = False

# vergeet gebruikte params soms dus print ze maar afentoe
def printParams():
    print('Num. Vid {} - Grid {} - Bins {} - Skip {}'.format(NUM_VIDEOS, GRID_SIZE, BINS, HIST_FRAME_SKIP))


# ## Load training set

# In[77]:


printParams()
training_set = prep.load_training_set(range(1, NUM_VIDEOS+1), GRID_SIZE, BINS, HIST_FRAME_SKIP, force_refresh=REFRESH)


# In[78]:


# Print statistics
print("TRAINING SET:")
print("Num. videos:   {:d}".format( len(training_set)) )
print("Num. segments: {:d}".format( np.sum([len(video.segments) for video in training_set])) )
print("Duration:      {:,.1f} s".format( np.sum([np.sum([segment.duration() for segment in video.segments]) for video in training_set])) )
print("Num frames:      {:d}".format( np.sum([np.sum([segment.num_frames() for segment in video.segments]) for video in training_set])) )
print("Num histograms:      {:d}".format( np.sum([np.sum([len(segment.histograms) for segment in video.segments]) for video in training_set])) )


# ## Select random test set

# In[79]:


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


# In[80]:


# Print statistics
print("TEST SET:")
print("Num. histograms: {:d}".format( np.sum([len(histogram) for histogram in test_set]) ))


# <br>

# ## Run model on test set

# In[81]:


printParams()

for method in [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.cv2.HISTCMP_INTERSECT,
               cv2.HISTCMP_BHATTACHARYYA, cv2.HISTCMP_CHISQR_ALT, cv2.HISTCMP_KL_DIV]:
    print('{}'.format(method))
    get_ipython().run_line_magic('timeit', '-n 10 search.findFrame(test_set[0][0], training_set, method)')

for ch in [[0], [1], [0, 1]]:
    print('{}'.format(ch))
    get_ipython().run_line_magic('timeit', '-n 10 search.findFrame(test_set[0][0], training_set, cv2.HISTCMP_CORREL, channels=ch)')


# %timeit search.find(test_set[0], training_set, cv2.HISTCMP_CORREL)
# %timeit search.find(test_set[0], training_set, cv2.HISTCMP_CHISQR)
# %timeit search.find(test_set[0], training_set, cv2.HISTCMP_INTERSECT)
# %timeit search.find(test_set[0], training_set, cv2.HISTCMP_BHATTACHARYYA)
# %timeit search.find(test_set[0], training_set, cv2.HISTCMP_CHISQR_ALT)
# %timeit search.find(test_set[0], training_set, cv2.HISTCMP_KL_DIV)


# In[82]:


printParams()

results = []

for i, segment_histograms in enumerate(test_set):
    print('\rSearching segment {}/{}'.format(i+1, len(test_set)), end='', flush=True)
    
    x = random.choice(range(len(segment_histograms)))
    results.append(search.findFrame(segment_histograms[0], training_set, 5, cv2.HISTCMP_CHISQR_ALT))


# ## Evaluate performance

# In[83]:


evaluate_segments(results, labels)


# # Manual Evaluation

# In[73]:


#Manually check what happens
test_vid = 1
for i in range(len(training_set[test_vid].segments)):
    
    hists = training_set[test_vid].segments[i].histograms

    #Possibly shrink set for readability
    # train = training_set.copy()
    # for t in train:
    #     t.segments = t.segments[:20]

    # 0 = first histogram of segment, so perfect match
    tf = 0
    search.findFrame(hists[tf], train, cv2.HISTCMP_CHISQR_ALT, 5, prints=False, printRes = True)


# In[ ]:





# In[ ]:




