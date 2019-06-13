import pandas as pd
import numpy as np

import cv2

def splitFrame(array, gridSize):
    r, h, _ = array.shape
    
    new_r = int(r/gridSize)
    new_h = int(h/gridSize)

    frames = []
    for i in range(gridSize):
        for j in range(gridSize):

            subframe = array[i*new_r:(i+1)*new_r,j*new_h:(j+1)*new_h,:]
            frames.append(subframe)
    
    return frames




def extract_frame_hs_histogram(frame):
  histogram = []
  hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
  range = [[0, 180], [0, 256]]
  channels = [0,1]
  for i in channels:
    channel = frame[:,:,i]
    channel_range = range[i]


    channel_histogram = cv2.calcHist(
        [hsv_image],
        [i],
        None, # one could specify an optional mask here (we don't use this here),
        [8],
        channel_range
        )
        
    histogram.append(channel_histogram)

    return histogram
#   return (histogram / np.sum(histogram)).reshape((2, 8))  # Return a normalized histogram.



def compute_histograms(frame, grid_size = 2, hist_func = extract_frame_hs_histogram):
    histograms = [extract_frame_hs_histogram(frame)]
        
    subframes = splitFrame(frame, grid_size)
    
    for i in range(grid_size**2):
        sub_hist = subframes[i]
        histograms.append(extract_frame_hs_histogram(subframes[i]))

    return histograms