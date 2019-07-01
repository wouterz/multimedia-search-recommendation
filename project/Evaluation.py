#!/usr/bin/env python
# coding: utf-8

# # Evaluation

# In[8]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[9]:


import pandas as pd
import numpy as np
from src import preprocess as prep
from src.evaluation import pick_test_segments, generate_test_segments, evaluate_segments, evaluate
from src.Video import Video
from src.Segment import Segment
from src import search
import random
import cv2


# ## Parameters

# In[356]:


NUM_VIDEOS = 2
GRID_SIZE = 2
BINS = [int(180/4), int(256/4)]
# negative value is average
HIST_FRAME_SKIP = 1
REFRESH = True

# vergeet gebruikte params soms dus print ze maar afentoe
def printParams():
    print('Num. Vid {} - Grid {} - Bins {} - Skip {}'.format(NUM_VIDEOS, GRID_SIZE, BINS, HIST_FRAME_SKIP))


# ## Load training set / generate test set

# In[357]:


printParams()
training_set = prep.load_training_set(range(1, NUM_VIDEOS+1), GRID_SIZE,
                                      BINS, HIST_FRAME_SKIP, force_refresh=REFRESH)


# In[354]:


# Set of 100 custom fragments with duration 20sec
test_set, labels = prep.get_test_video_set(NUM_VIDEOS, GRID_SIZE, BINS, n=10)


# In[358]:


# Print statistics
print("TRAINING SET:")
print("Num. videos:    {:d}".format( len(training_set)) )
print("Num. segments:  {:d}".format( np.sum([len(video.segments) for video in training_set])) )
print("Duration:       {:,.1f} s".format( np.sum([np.sum([segment.duration() for segment in video.segments]) for video in training_set])) )
print("Num frames:     {:d}".format( np.sum([np.sum([segment.num_frames() for segment in video.segments]) for video in training_set])) )
print("Num histograms: {:d}".format( np.sum([np.sum([len(segment.histograms) for segment in video.segments]) for video in training_set])) )

print("\nTEST SET:")
print("Size: {:d}".format( len(test_set) ))


# # Small manual test

# In[345]:


for i, test_segment in enumerate(test_set):
    found = search.knownImageSearch(test_segment, training_set, cv2.HISTCMP_CHISQR_ALT, 5, HIST_FRAME_SKIP)
    
    print('Found {} - Expected {}'.format(found, labels[i]))


# ## Run model on test set

# In[172]:


for method in [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT,
               cv2.HISTCMP_BHATTACHARYYA, cv2.HISTCMP_CHISQR_ALT, cv2.HISTCMP_KL_DIV]:
    get_ipython().run_line_magic('timeit', '-n 1 search.knownImageSearch(test_set[0], training_set, cv2.HISTCMP_CHISQR_ALT, 5, HIST_FRAME_SKIP)')


# In[174]:


results = []

for i, test_segment in enumerate(test_set):
    print("\rSearching segment {}/{}".format(i+1, len(test_set), len(test_segment)), end='', flush=True)
    
    results.append(search.knownImageSearch(test_segment, training_set, cv2.HISTCMP_CHISQR_ALT, 5, 
                                    HIST_FRAME_SKIP))


# ## Evaluate performance

# In[185]:


movie_results, start_frame_dist = evaluate(results, labels)

fractions = (movie_results[0] / movie_results[2]*100 if movie_results[2] > 0 else 0,              movie_results[1] / movie_results[0]*100 if movie_results[0] > 0 else 0)

print("TEST RESULTS\n")

printParams()
print("\nCorrect video: {:d} / {:d} ({:.1f}%)".format(movie_results[0], movie_results[2], fractions[0]))
print("Inside fragment: {:d} / {:d} ({:.1f}%)".format(movie_results[1], movie_results[0], fractions[1]))
print("Average distance to center of segment: {:.0f} +/- {:.0f} frames (approx. {:.1f} sec)".format(
    start_frame_dist[0], start_frame_dist[1], start_frame_dist[0]/30))


# In[352]:


def gen():
    for i in range(1,20):
        yield(i)
        
g = gen()

for i in g:
    print(i)
    for _ in range(2):
        try:
            print('inner', next(g))
        except StopIteration:
            pass
    


# In[ ]:




