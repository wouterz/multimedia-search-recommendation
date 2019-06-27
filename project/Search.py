#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[12]:


import cv2
import src.preprocess as prep
from src import search


# In[6]:


vids = prep.load_training_set(range(200))
vids_list = list(vids)


# In[20]:



for i in range(1,20):
    search.find(vids_list[4].segments[i].histograms, vids_list, cv2.HISTCMP_CORREL, True)


# In[ ]:




