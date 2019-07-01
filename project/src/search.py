import pandas as pd
import numpy as np
import cv2

def isSimilarityMetric(metric):
    if metric == cv2.HISTCMP_CORREL or metric == cv2.HISTCMP_INTERSECT:
        return True
    elif metric == cv2.HISTCMP_CHISQR or metric == cv2.HISTCMP_BHATTACHARYYA or metric == cv2.HISTCMP_CHISQR_ALT or metric == cv2.HISTCMP_KL_DIV:
        return False
    
    raise 'Unknown hist metric'
        
def knownImageSearch(segment, training_set, method, best_n, frame_skip):
    """
    Perform a known image search, looking for segment in the training set. 
    This function recursively looks for a start and end frame that belong to the
    same video and within 800 frames of each other.
    """
    maxFrame = len(segment)-1
    
    #avg hotfix
    frame_skip = abs(frame_skip)
    
    found_start = findFrame(segment[0], training_set, method, best_n,
                             hist_frame_skip=frame_skip, warnings = False)
    found_end = findFrame(segment[maxFrame], training_set, method, best_n, 
                             hist_frame_skip=frame_skip, warnings = False)
    
    if found_start[0] == found_end[0] and abs(found_start[1] - found_end[1]) < 800:
        return (found_start[0], (found_start[1]+found_end[1])/2)
    else:
        return knownImageSearch(segment[1:maxFrame], training_set, method, best_n, frame_skip)


def findFrame(target_histograms, videos, histMetric, best_n_full_hist = 10, channels = [0, 1], hist_frame_skip = 20, prints = False, warnings = True):
    """
    Try to find the target_histograms (Array containing histograms per channel(H/s)), in the videos.   
    """

    # Check if chosen metric is similarity or dissimilarity
    isSimilarity = isSimilarityMetric(histMetric)
    
    assert best_n_full_hist >= 2, 'If smaller than 2 never know if top list is exhaustive'
    
    # Array containing the distances per video for each segment
    distances = []
    
    for video in videos:                                         # Loop over all videos
        segment_dist = np.zeros(len(video.segments))             # Array to store the distance per segment

        i = 0
        for segment in video.segments:                               # Video has many segments
            frame_dist = np.zeros(len(segment.histograms))           # Array to store distances by per frame
            
            j = 0
            for frame_hists in segment.histograms:          # Segment has many frames (list of histograms per frame)
                dist = 0                                    # Set distance to zero
                
                for channel in channels:                    # Sum distance per channel (index '0' is full histogram)
                    dist += cv2.compareHist(target_histograms[0][channel], frame_hists[0][channel], histMetric)

                frame_dist[j] = dist
                j+=1
            
            if prints:
                print('frame_dists', frame_dist)
            
            # Currenly only interested in the best score per segment, to find the matching segment
            segment_dist[i] = frame_dist.max() if isSimilarity else frame_dist.min()
            i+=1

        distances.append(segment_dist)
    
    if prints:
        print('distances', distances)
    
    # Compute top n segments for each video
    best_segment_dist_indices = []
    for d in distances:
        
        n_best = min(best_n_full_hist, len(d))
        
        if isSimilarity:
            best_n_idx = np.argpartition(d, -n_best)[-n_best:]
            values = [d[idx] for idx in best_n_idx]
            if warnings and values.count(values[0]) == n_best:
                print('WARNING: n_best too small might miss best value', values)
            best_segment_dist_indices.append(best_n_idx)
        else:
            best_n_idx = np.argpartition(d, n_best)[:n_best]
            values = [d[idx] for idx in best_n_idx]
            if warnings and values.count(values[0]) == n_best:
                print('WARNING: n_best too small might miss best value', values)
            best_segment_dist_indices.append(best_n_idx)

        
    if prints:
        print('best_segment_dist_indices', best_segment_dist_indices)
        values = []
        for i, tmp in enumerate(best_segment_dist_indices):
            for ind in tmp:
                values.append(distances[i][ind])
        print('values', values)

        
        
    # Now using the most likely segments, take closer look using sub_grids    
    
    # Array containing the distances per video for each segment
    sub_distances = []
    for i, video in enumerate(videos):
        
        segment_dist = np.zeros(len(best_segment_dist_indices[i]))          # Array to store the best distance per segment
        z = 0
        
        for segment_index in best_segment_dist_indices[i]:                  # Check the segments that matched in the previous
            segment = video.segments[segment_index]
            
            frame_dist = np.zeros(len(segment.histograms))
            j = 0
            
            for frame_hists in segment.histograms:  # Segment has many frames (list of histograms per frame)
                dist = 0

                for hist_index, hists in enumerate(frame_hists):            # Frame has many (sub-)histograms
                    
                    # Skip the first one as these are the full histogram
                    if hist_index < 1: continue
                        
                    for channel in channels:                                # Sum distance per channel
                        dist += cv2.compareHist(target_histograms[hist_index][channel], hists[channel], histMetric)

                frame_dist[j] = dist
                j += 1
                    
            # Currenly only interested in the best score per segment, to find the matching segment
            segment_dist[z] = frame_dist.max() if isSimilarity else frame_dist.min()
            z += 1
            
        sub_distances.append(segment_dist)
    
    # Find index of maximum value in matrix
    # TODO Check and handle if there are multiple candidates....
    if prints:
        print('sub_distances', sub_distances)

    result = np.where(sub_distances == (np.amax(sub_distances) if isSimilarity else np.amin(sub_distances)))
    result_idx = np.argwhere(sub_distances == (np.amax(sub_distances) if isSimilarity else np.amin(sub_distances)))
    
    # Check if there are still multiple candidates, then TODO
    if warnings and len(result[0]) > 1:
        print('WARNING: multiple final matches found, returing one from candidates:', list(zip(result[0], result[1])))
    
    match_vid = result[0][0]
    match_seg = best_segment_dist_indices[result[0][0]][result[1][0]]
    matched_frame_idx = best_segment_dist_indices[result_idx[0][0]][result_idx[0][0]]
        
    if prints:
        print('video {:05d} - segment {}'.format(match_vid+1, match_seg))

    seg = videos[match_vid].segments[match_seg]
    
    matched_frame = seg.frame_start + int(matched_frame_idx * hist_frame_skip)
    return ('{:05d}.mp4'.format(match_vid+1), matched_frame)
