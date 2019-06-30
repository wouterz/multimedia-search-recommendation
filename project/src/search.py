import pandas as pd
import numpy as np
import cv2

def isSimilarityMetric(metric):
    if metric == cv2.HISTCMP_CORREL or metric == cv2.HISTCMP_INTERSECT:
        return True
    elif metric == cv2.HISTCMP_CHISQR or metric == cv2.HISTCMP_BHATTACHARYYA or metric == cv2.HISTCMP_CHISQR_ALT or metric == cv2.HISTCMP_KL_DIV:
        return False
    
    raise 'Unknown hist metric'
        


def findFrame(target_histograms, videos, histMetric, best_n_full_hist = 10, channels = [0, 1], prints = False, warnings = True):
    """
    Try to find the target_histograms (Array containing histograms per channel(H/s)), in the videos.   
    """

    # Check if chosen metric is similarity or dissimilarity
    isSimilarity = isSimilarityMetric(histMetric)
    
    assert best_n_full_hist >= 2, 'If smaller than 2 never know if top list is exhaustive'
    
    # Array containing the distances per video for each segment
    distances = []
    
    for video in videos:                                             # Loop over all videos
        segment_dist = []                                            # Array to store the distance per segment

        for segment in video.segments:                               # Video has many segments
            frame_dist = []                                          # Array to store distances by per frame
            
            for frame_hists in segment.histograms:          # Segment has many frames (list of histograms per frame)
                dist = 0                                    # Set distance to zero

                for channel in channels:                    # Sum distance per channel (index '0' is full histogram)
                    dist += cv2.compareHist(target_histograms[0][channel], frame_hists[0][channel], histMetric)

                frame_dist.append(dist)
            
            if prints:
                print('frame_dists', frame_dist)
                
            # Currenly only interested in the best score per segment, to find the matching segment
            if isSimilarity:
                segment_dist.append(max(frame_dist))
            else:
                segment_dist.append(min(frame_dist))

        distances.append(segment_dist)
    
    # TODO: distances and segment_dist are lists and not arrays
    distances = np.array(distances)
    
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
        segment_dist = []                                                   # Array to store the best distance per segment
        
        for segment_index in best_segment_dist_indices[i]:                  # Check the segments that matched in the previous
            segment = video.segments[segment_index]
            
            for frame_index, frame_hists in enumerate(segment.histograms):  # Segment has many frames (list of histograms per frame)
                dist = 0

                for hist_index, hists in enumerate(frame_hists):            # Frame has many (sub-)histograms
                    # Skip the first one as this is the full histogram
                    if hist_index == 0:
                        continue
                        
                    for channel in channels:                                # Sum distance per channel
                        dist += cv2.compareHist(target_histograms[hist_index][channel], hists[channel], histMetric)

                frame_dist.append(dist)
                    
            # Currenly only interested in the best score per segment, to find the matching segment
            if isSimilarity:
                segment_dist.append(max(frame_dist))
            else:
                segment_dist.append(min(frame_dist))
            
        sub_distances.append(segment_dist)
    
    # Find index of maximum value in matrix
    # TODO Check and handle if there are multiple candidates....
    if prints:
        print('sub_distances', sub_distances)
        
    result = []
    if isSimilarity:
        result = np.where(sub_distances == np.amax(sub_distances))
    else:
        result = np.where(sub_distances == np.amin(sub_distances))
    
    # Check if there are still multiple candidates, then TODO
    if warnings and len(result[0]) > 1:
        print('WARNING: multiple final matches found, returing one from candidates:', list(zip(result[0], result[1])))
    
    match_vid = result[0][0]
    match_seg = best_segment_dist_indices[result[0][0]][result[1][0]]
    
    if prints:
        print('video {:05d} - segment {}'.format(match_vid+1, match_seg))

    seg = videos[match_vid].segments[match_seg]
    return ('{:05d}.mp4'.format(match_vid+1), seg.frame_start + 0, seg.frame_end + 20 * 27)
 


def find(target_histograms, videos, histMetric, prints = False, printRes = False):
    isSimilarity = isSimilarityMetric(histMetric)
    
    # 2D array, an array for each video containing the distance per segment
    distances = []
    
    for video in videos:
        
        # Array to store the distance per segment
        segment_dist = []
        
        for segment in video.segments:  # Video has many segments
            dist = 0
            
            for frame_index, frame_hists in enumerate(segment.histograms):  # Segment has many frames (list of histograms per frame)
                # If we use the following line, processing time is very low (~25ms instead of ~200ms) but then we only consider one channel
#                 dist += cv2.compareHist([frame_index][0][0], frame_hists[0][0], histMetric)
                
                # Sum distance per channel (index '0' is full histogram)
                for channel, h in enumerate(frame_hists[0]): # TODO: maybe merge channels and check simultaneously for speed
                    dist += cv2.compareHist(target_histograms[0][0][channel], h, histMetric)
                
            segment_dist.append(dist)
        distances.append(segment_dist)
    
    # TODO: distances and segment_dist are lists and not arrays
    distances = np.asarray(distances)
        
    # Compute top 5 segments -with the lowest distance- for each video
    best_dist_indices = []
    if isSimilarity:
        for d in distances: best_dist_indices.append(np.argpartition(d, -5)[-5:])
    else:
        for d in distances: best_dist_indices.append(np.argpartition(d, 5)[:5])
    
    sub_distances = []
    
    # TODO: maybe invert this loop? First iterate best_dist_indices and then loop videos?
    for i, video in enumerate(videos):
        segment_dist = []
        
        for segment_index in best_dist_indices[i]:
            segment = video.segments[segment_index]
            dist = 0
            
            for frame_index, frame_hists in enumerate(segment.histograms):  # Segment has many frames (list of histograms per frame)
                for hist_index, hists in enumerate(frame_hists):            # Frame has many (sub-)histograms
                    
                    # Sum distance per channel
                    for channel, h in enumerate(hists):
                        dist += cv2.compareHist(target_histograms[0][hist_index][channel], h, histMetric)
                    
            segment_dist.append(dist)    
            
        sub_distances.append(segment_dist)

    
    # Find index of maximum value in matrix
    result = []
    if isSimilarity:
        result = np.where(sub_distances == np.amax(sub_distances))
    else:
        result = np.where(sub_distances == np.amin(sub_distances))
        
    match_vid = result[0][0]
    match_seg = best_dist_indices[result[0][0]][result[1][0]]
    
    if printRes:
        print('video {:05d} - segment {}'.format(match_vid+1, match_seg))

    return videos[match_vid].segments[match_seg]