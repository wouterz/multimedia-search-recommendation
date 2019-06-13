#!/usr/bin/env python
# coding: utf-8

# # Movie preprocessing

# In[3]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[4]:


import pandas as pd
import numpy as np
from src import preprocess as prep


# In[5]:


vid = prep.process_video("00001")

