#!/usr/bin/env python
# coding: utf-8

# # Evaluation

# In[83]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[84]:


import pandas as pd
import numpy as np
from src import preprocess as prep
import random
from src.Video import Video
from src.Segment import Segment
from src import search
import cv2


# ## Parameters

# In[95]:


NUM_VIDEOS = 200
GRID_SIZE = 2
BINS = [180, 180]


# ## Load training set

# In[96]:


training_set_generator = prep.load_training_set(range(1, NUM_VIDEOS+1), GRID_SIZE, BINS)


# In[97]:


training_set = list(training_set_generator)


# In[98]:


# Print statistics
print("TRAINING SET:")
print("Num. videos:   {:d}".format( len(training_set)) )
print("Num. segments: {:d}".format( np.sum([len(video.segments) for video in training_set])) )
print("Duration:      {:,.1f} s".format( np.sum([np.sum([segment.duration() for segment in video.segments]) for video in training_set])) )


# ## Select random test set

# In[99]:


test_n_segments = 1000
test_set = []
labels = []

for i in range(test_n_segments):
    
    # Find random video
    video = random.choice(training_set)
    
    # Select random segment and add histogram to test set
    segment = random.choice(video.segments)
    test_set.append(segment.histograms)
    labels.append(segment)


# In[100]:


# Print statistics
print("TEST SET:")
print("Num. histograms: {:d}".format( len(test_set) ))


# <br><br>

# ## Run model on test set

# In[101]:


get_ipython().run_line_magic('timeit', 'search.find(test_set[0], training_set, cv2.HISTCMP_CORREL)')
get_ipython().run_line_magic('timeit', 'search.find(test_set[0], training_set, cv2.HISTCMP_CHISQR_ALT)')
get_ipython().run_line_magic('timeit', 'search.find(test_set[0], training_set, cv2.HISTCMP_INTERSECT)')
get_ipython().run_line_magic('timeit', 'search.find(test_set[0], training_set, cv2.HISTCMP_BHATTACHARYYA)')
get_ipython().run_line_magic('timeit', 'search.find(test_set[0], training_set, cv2.HISTCMP_CHISQR_ALT)')
get_ipython().run_line_magic('timeit', 'search.find(test_set[0], training_set, cv2.HISTCMP_KL_DIV)')


# In[42]:


results = []

for histogram in test_set:
    results.append(search.find(histogram, training_set, cv2.HISTCMP_INTERSECT))


# ## Evaluate performance

# In[46]:


movie_correct = 0
movie_wrong = 0

for segment, label in zip(results, labels):
    
    # Check if movie is correct
    if segment == label:
        movie_correct += 1
    else:
        movie_wrong += 1

total = movie_correct + movie_wrong
fraction = movie_correct / total if total > 0 else 0

print("Correct: {:d}".format(movie_correct))
print("Wrong:   {:d}".format(movie_wrong))
print("Total:   {:d}".format(total))
print("TPR:     {:.1f}%".format(movie_correct / total * 100))


# In[ ]:




