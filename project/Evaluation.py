#!/usr/bin/env python
# coding: utf-8

# # Evaluation

# In[120]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[121]:


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

# In[122]:


NUM_VIDEOS = 20
GRID_SIZE = 2
BINS = [180, 180]
HIST_FRAME_SKIP = 20
REFRESH = False

# vergeet gebruikte params soms dus print ze maar afentoe
def printParams():
    print('Num. Vid {} - Grid {} - Bins {} - Skip {}'.format(NUM_VIDEOS, GRID_SIZE, BINS, HIST_FRAME_SKIP))


# ## Load training set

# In[123]:


printParams()
training_set = prep.load_training_set(range(1, NUM_VIDEOS+1), GRID_SIZE, BINS, HIST_FRAME_SKIP, force_refresh=REFRESH)


# In[124]:


# Print statistics
print("TRAINING SET:")
print("Num. videos:   {:d}".format( len(training_set)) )
print("Num. segments: {:d}".format( np.sum([len(video.segments) for video in training_set])) )
print("Duration:      {:,.1f} s".format( np.sum([np.sum([segment.duration() for segment in video.segments]) for video in training_set])) )
print("Num frames:      {:d}".format( np.sum([np.sum([segment.num_frames() for segment in video.segments]) for video in training_set])) )
print("Num histograms:      {:d}".format( np.sum([np.sum([len(segment.histograms) for segment in video.segments]) for video in training_set])) )


# ## Select random test set

# In[125]:


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


# In[126]:


# Print statistics
print("TEST SET:")
print("Num. histograms: {:d}".format( np.sum([len(histogram) for histogram in test_set]) ))


# <br>

# ## Run model on test set

# In[89]:


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


# In[133]:


printParams()

results = []

for i, segment_histograms in enumerate(test_set):
    print('\rSearching segment {}/{}'.format(i+1, len(test_set)), end='', flush=True)
    
    x = random.choice(range(len(segment_histograms)))
#     results.append(search.find(segment_histograms, training_set, 5, cv2.HISTCMP_CHISQR_ALT))
    results.append(search.findFrame(segment_histograms[0], training_set, 1, cv2.HISTCMP_CHISQR_ALT))


# ## Evaluate performance

# In[134]:


evaluate_segments(results, labels)


# # Manual Evaluation

# In[99]:


#Manually check what happens
test_vid = 6
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




