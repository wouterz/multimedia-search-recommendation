import pandas as pd
import numpy as np
import cv2

def isSimilarityMetric(metric):
    if metric == cv2.HISTCMP_CORREL or metric == cv2.HISTCMP_INTERSECT:
        return True
    elif metric == cv2.HISTCMP_CHISQR or metric == cv2.HISTCMP_BHATTACHARYYA or metric == cv2.HISTCMP_CHISQR_ALT or metric == cv2.HISTCMP_KL_DIV:
        return False
    
    raise 'Unknown hist metric'
        

def find(target_histograms : np.ndarray, videos, histMetric, prints = False):
    isSimilarity = isSimilarityMetric(histMetric)

    # 2D array, an array for each video containing the distance per segement
    distances = []
    for video in videos:
        # Array to store the distance per segment
        segment_dist = []       
        for segment in video.segments:
            dist = 0
            # per channel
            for i, h in enumerate(segment.histograms[0]):
                # Method kiezen, voor nu intersection
                # Compare the full image histogram
                dist += cv2.compareHist(target_histograms[0][i], h, histMetric)
            segment_dist.append(dist)
        distances.append(segment_dist)
    
    # Compute top 5 segments -with the lowest distance- for each video
    best_dist_indices = []
    if isSimilarity:
        for d in distances:
            best_dist_indices.append(np.argpartition(d, -5)[-5:])
    else:
        for d in distances:
            best_dist_indices.append(np.argpartition(d, 5)[:5])
    
    sub_distances = []
    i = 0
    for video in videos:
        segment_dist = []       
        for segment_index in best_dist_indices[i]:
            segment = video.segments[segment_index]
            dist = 0
            for j, grid_hists in enumerate(segment.histograms): 
                for k, channel_hist in enumerate(grid_hists):
                    dist += cv2.compareHist(target_histograms[j][k], channel_hist, histMetric)
            segment_dist.append(dist)    
            
        sub_distances.append(segment_dist)
        i = i + 1

    # Find index of maximum value in matrix
    result = []
    if isSimilarity:
        result = np.where(sub_distances == np.amax(sub_distances))
    else:
        result = np.where(sub_distances == np.amin(sub_distances))
        
    match_vid = result[0][0]
    match_seg = best_dist_indices[result[0][0]][result[1][0]]
    
    if prints:
        print('video {:05d} - segment {}'.format(match_vid+1, match_seg))

    return videos[match_vid].segments[match_seg]