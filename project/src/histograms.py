import pandas as pd
import numpy as np

import cv2


def frame_to_hs_hist(hsv_frame, bins):
    histograms = []
    
    channels = [0,1] # H / S
    ranges = [[0, 180], [0, 256]]
    
    for i in channels:
        histograms.append(cv2.calcHist(
            [hsv_frame], # Source image
            [i],         # Channel index
            None,        # Optional mask
            [bins[i]],   # Histogram size
            ranges[i]    # Range
        ))

    return histograms
#   return (histogram / np.sum(histogram)).reshape((2, 8))  # Return a normalized histogram.


def compute_histograms(frame, hist_func=frame_to_hs_hist, grid_size=2, bins=[180, 180]):
    """
    compute_histograms computes the full histogram and grid_size**2 subhistograms from top left horizontally to bottom right.
    """
    
    # Convert image to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Initialize array with main histogram
    histograms = frame_to_hs_hist(hsv_frame, bins)
    
    # Split frame into grids and calculate histograms
    # TODO: why save these at all and not just 'generate' and check them only for the best matches?
    if grid_size and grid_size > 0:
        for sub_frame in split_frame(hsv_frame, grid_size):
            histograms += hist_func(sub_frame, bins)

    return histograms


def split_frame(array, grid_size):
#     r, h, _ = array.shape
    
#     new_r = int(r/gridSize)
#     new_h = int(h/gridSize)

#     frames = []
#     for i in range(gridSize):
#         for j in range(gridSize):
#             subframe = array[i*new_r:(i+1)*new_r,j*new_h:(j+1)*new_h,:]
#             frames.append(subframe)
    
    blocks = list(blockgen(array, [grid_size, grid_size, 1]))
    
    # Debug:
#     display(len(blocks))
#     display(array.shape)
#     display(blocks[0].shape)
#     display(blocks[1].shape)
#     display(blocks[2].shape)
#     display(blocks[3].shape)
    
    return blocks


def blockgen(array, bpa):
    """Creates a generator that yields multidimensional blocks from the given
array(_like); bpa is an array_like consisting of the number of blocks per axis
(minimum of 1, must be a divisor of the corresponding axis size of array). As
the blocks are selected using normal numpy slicing, they will be views rather
than copies; this is good for very large multidimensional arrays that are being
blocked, and for very large blocks, but it also means that the result must be
copied if it is to be modified (unless modifying the original data as well is
intended)."""
    bpa = np.asarray(bpa) # in case bpa wasn't already an ndarray

    # parameter checking
    if array.ndim != bpa.size:         # bpa doesn't match array dimensionality
        raise ValueError("Size of bpa must be equal to the array dimensionality.")
    if (bpa.dtype != np.int            # bpa must be all integers
        or (bpa < 1).any()             # all values in bpa must be >= 1
        or (array.shape % bpa).any()): # % != 0 means not evenly divisible
        raise ValueError("bpa ({0}) must consist of nonzero positive integers "
                         "that evenly divide the corresponding array axis "
                         "size".format(bpa))


    # generate block edge indices
    rgen = (np.r_[:array.shape[i]+1:array.shape[i]//blk_n]
            for i, blk_n in enumerate(bpa))

    # build slice sequences for each axis (unfortunately broadcasting
    # can't be used to make the items easy to operate over
    c = [[np.s_[i:j] for i, j in zip(r[:-1], r[1:])] for r in rgen]

    # Now to get the blocks; this is slightly less efficient than it could be
    # because numpy doesn't like jagged arrays and I didn't feel like writing
    # a ufunc for it.
    for idxs in np.ndindex(*bpa):
        blockbounds = tuple(c[j][idxs[j]] for j in range(bpa.size))

        yield array[blockbounds]