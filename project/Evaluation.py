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
from src.evaluation import pick_test_segments, generate_test_segments, evaluate_segments, evaluate
from src.Video import Video
from src.Segment import Segment
from src import search
import random
import cv2


# ## Parameters

# In[73]:


NUM_VIDEOS = 10
GRID_SIZE = 2
BINS = [180, 256]
HIST_FRAME_SKIP = 3
REFRESH = False

# vergeet gebruikte params soms dus print ze maar afentoe
def printParams():
    print('Num. Vid {} - Grid {} - Bins {} - Skip {}'.format(NUM_VIDEOS, GRID_SIZE, BINS, HIST_FRAME_SKIP))


# ## Load training set / generate test set

# In[74]:


printParams()
training_set = prep.load_training_set(range(1, NUM_VIDEOS+1), GRID_SIZE, BINS, HIST_FRAME_SKIP, force_refresh=REFRESH)


# In[75]:


# Set of 100 custom fragments with duration 20sec
test_set, labels = generate_test_segments(training_set, n=100, duration=20)


# In[76]:


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

# In[87]:


pr = False
for i in range(0,1):
    for j in range(5):
        x = random.choice(range(len(test_set[i])))
        found = search.findFrame(test_set[i][x], training_set, cv2.HISTCMP_CHISQR, 10, prints= pr, warnings=pr)
        print('Found {} - Expected {}'.format(found, labels[i]))

        
test_histograms = prep.get_test_video("{:05d}".format(9), GRID_SIZE, BINS)
for i in range(10):
    found = search.findFrame(test_histograms[i], training_set, cv2.HISTCMP_CHISQR, 10, prints= pr, warnings=pr)
    print(found)


# ## Run model on test set

# In[9]:


for method in [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.cv2.HISTCMP_INTERSECT,
               cv2.HISTCMP_BHATTACHARYYA, cv2.HISTCMP_CHISQR_ALT, cv2.HISTCMP_KL_DIV]:
# for method in [cv2.HISTCMP_BHATTACHARYYA]:
    get_ipython().run_line_magic('timeit', '-n 1 search.findFrame_old(test_set[0][0], training_set, method, warnings = False)')
    get_ipython().run_line_magic('timeit', '-n 1 search.findFrame(test_set[0][0], training_set, method, warnings = False)')
    print()

# for ch in [[0], [1], [0, 1]]:
#     print('{}'.format(ch))
#     %timeit -n 10 search.findFrame(test_set[0], training_set, cv2.HISTCMP_CORREL, channels=ch)


# In[127]:


results = []

for i, histogram in enumerate(test_set):
    print("\rSearching segment {}/{}".format(i+1, len(test_set), len(histogram)), end='', flush=True)
    
    results.append(search.findFrame(histogram[0], training_set, cv2.HISTCMP_CHISQR_ALT, 2, warnings = False))


# ## Evaluate performance

# In[129]:


movie_results, start_frame_dist = evaluate(results, labels)

fractions = (movie_results[0] / movie_results[2]*100 if movie_results[2] > 0 else 0, movie_results[1] / movie_results[0]*100 if movie_results[0] > 0 else 0)

print("TEST RESULTS\n")

printParams()
print("\nCorrect video:                   {:d} / {:d} ({:.1f}%)".format(movie_results[0], movie_results[2], fractions[0]))
print("Inside fragment:                 {:d} / {:d} ({:.1f}%)".format(movie_results[1], movie_results[0], fractions[1]))
print("Average distance to start frame: {:.0f} +/- {:.0f} frames (approx. {:.1f} sec)".format(start_frame_dist[0], start_frame_dist[1], start_frame_dist[0]/25))

