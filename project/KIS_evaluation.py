#!/usr/bin/env python
# coding: utf-8

# # Evaluation

# In[522]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[550]:


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

# In[603]:


NUM_VIDEOS = 200
GRID_SIZE = 2
BINS = [int(180/10), int(256/10)]
# negative value is average; -2 averages two frames, takes every 2nd frame (only skips one) (if frame_id % 2 == 0).
HIST_FRAME_SKIP = 10
REFRESH = True

# vergeet gebruikte params soms dus print ze maar afentoe
def printParams():
    print('Num. Vid {} - Grid {} - Bins {} - Skip {}'.format(NUM_VIDEOS, GRID_SIZE, BINS, HIST_FRAME_SKIP))


# ## Load training set / generate test set

# In[ ]:


printParams()
training_set = prep.load_training_set(range(1, NUM_VIDEOS+1), GRID_SIZE,
                                      BINS, HIST_FRAME_SKIP, force_refresh=REFRESH)


# In[592]:


# Set of 100 custom fragments with duration 20sec
test_set, labels = prep.get_test_video_set(NUM_VIDEOS, GRID_SIZE, BINS, n=20)


# In[593]:


# Print statistics
print("TRAINING SET:")
print("Num. videos:    {:d}".format( len(training_set)) )
print("Num. segments:  {:d}".format( np.sum([len(video.segments) for video in training_set])) )
print("Duration:       {:,.1f} s".format( np.sum([np.sum([segment.duration() for segment in video.segments]) for video in training_set])) )
print("Num frames:     {:d}".format( np.sum([np.sum([segment.num_frames() for segment in video.segments]) for video in training_set])) )
print("Num histograms: {:d}".format( np.sum([np.sum([len(segment.histograms) for segment in video.segments]) for video in training_set])) )

print("\nTEST SET:")
print("Size: {:d}".format( len(test_set) ))


# ## Run model on test set

# In[601]:


get_ipython().run_cell_magic('time', '', 'results = []\n\nfor i, test_segment in enumerate(test_set):\n    print("\\rSearching segment {}/{}".format(i+1, len(test_set), len(test_segment)), end=\'\', flush=True)\n    \n    results.append(search.knownImageSearch(test_segment, training_set, cv2.HISTCMP_INTERSECT, 5, \n                                    HIST_FRAME_SKIP, [0]))')


# ## Evaluate performance

# In[602]:


movie_results, start_frame_dist = evaluate(results, labels)

fractions = (movie_results[0] / movie_results[2]*100 if movie_results[2] > 0 else 0,              movie_results[1] / movie_results[0]*100 if movie_results[0] > 0 else 0)

print("TEST RESULTS\n")

printParams()
print("\nCorrect video: {:d} / {:d} ({:.1f}%)".format(movie_results[0], movie_results[2], fractions[0]))
print("Inside fragment: {:d} / {:d} ({:.1f}%)".format(movie_results[1], movie_results[0], fractions[1]))
print("Average distance to center of segment: {:.0f} +/- {:.0f} frames (approx. {:.1f} sec)".format(
    start_frame_dist[0], start_frame_dist[1], start_frame_dist[0]/30))


# # Benchmark

# In[596]:


#all metrics
for method in [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT,
               cv2.HISTCMP_BHATTACHARYYA, cv2.HISTCMP_CHISQR_ALT, cv2.HISTCMP_KL_DIV]:
    get_ipython().run_line_magic('timeit', '-n 1 search.knownImageSearch(test_set[0], training_set, method, 5, HIST_FRAME_SKIP)')
    
print()
# channels
for channel in [[0], [1], [0,1]]:
    get_ipython().run_line_magic('timeit', '-n 1 search.knownImageSearch(test_set[0], training_set, cv2.HISTCMP_CHISQR_ALT, 5, HIST_FRAME_SKIP, channel)')

print()

# best n matches per vid
for matches in [2, 5, 10]:
    get_ipython().run_line_magic('timeit', '-n 1 search.knownImageSearch(test_set[0], training_set, cv2.HISTCMP_CHISQR_ALT, matches, HIST_FRAME_SKIP, [0])')


# In[ ]:




