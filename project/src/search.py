import pandas as pd
import numpy as np
import cv2

def isSimilarityMetric(metric):
    if metric == cv2.HISTCMP_CORREL or metric == cv2.HISTCMP_INTERSECT:
        return True
    elif metric == cv2.HISTCMP_CHISQR or metric == cv2.HISTCMP_BHATTACHARYYA or metric == cv2.HISTCMP_CHISQR_ALT or metric == cv2.HISTCMP_KL_DIV:
        return False
    
    raise 'Unknown hist metric'
        


def findFrame(target_histograms, frame_id, videos, histMetric, best_n_full_hist = 10, prints = False):
    assert len(target_histograms) >= frame_id
    
    isSimilarity = isSimilarityMetric(histMetric)
    
    # 2D array, an array for each video containing the distance per segment
    distances = []
    
    for video in videos:
        
        # Array to store the distance per segment
        segment_dist = []
        
        for segment in video.segments:  # Video has many segments
            frame_dist = []
            
            for frame_hists in segment.histograms:  # Segment has many frames (list of histograms per frame)
                dist = 0

                # Sum distance per channel (index '0' is full histogram)
#                 TODO Enumerate seems almost double runtime 35 to 65
                for channel, h in enumerate(frame_hists[0]): # TODO: maybe merge channels and check simultaneously for speed
                    dist += cv2.compareHist(target_histograms[frame_id][0][channel], frame_hists[0][channel], histMetric)
#                 dist += cv2.compareHist(target_histograms[frame_id][0][1], frame_hists[0][1], histMetric)
    
                frame_dist.append(dist)

            segment_dist.append(max(frame_dist))
        distances.append(segment_dist)
    
    # TODO: distances and segment_dist are lists and not arrays
    distances = np.array(distances)
    
    if prints:
        print('distances', distances)
    
    # Compute top n segments for each video
    best_dist_indices = []
    if isSimilarity:
        for d in distances: best_dist_indices.append(np.argpartition(d, -best_n_full_hist)[-best_n_full_hist:])
#         best_dist_indices = np.argpartition(distances, -best_n_dist)[-best_n_dist:]
    else:
        for d in distances: best_dist_indices.append(np.argpartition(d, best_n_full_hist)[:best_n_full_hist])

    if prints:
        print('best_dist_indices', best_dist_indices)

    sub_distances = []
    
    # TODO: maybe invert this loop? First iterate best_dist_indices and then loop videos?
    for i, video in enumerate(videos):
        segment_dist = []
        
        for segment_index in best_dist_indices[i]:
            segment = video.segments[segment_index]
            
            for frame_index, frame_hists in enumerate(segment.histograms):  # Segment has many frames (list of histograms per frame)
                dist = 0

                for hist_index, hists in enumerate(frame_hists):            # Frame has many (sub-)histograms
                    # Sum distance per channel
                    for channel, h in enumerate(hists):
                        dist += cv2.compareHist(target_histograms[frame_id][hist_index][channel], h, histMetric)
                
                frame_dist.append(dist)
                    
            segment_dist.append(max(frame_dist))    
            
        sub_distances.append(segment_dist)
    
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








def find(target_histograms, videos, histMetric, prints = False):
    """
    def find finds the sementsdjfoshdf
    def find finds the sementsdjfoshdf
    def find finds the sementsdjfoshdf
    def find finds the sementsdjfoshdf
    def find finds the sementsdjfoshdf
    def find finds the sementsdjfoshdf  
    """
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
                    dist += cv2.compareHist(target_histograms[frame_index][0][channel], h, histMetric)
                
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
                        dist += cv2.compareHist(target_histograms[frame_index][hist_index][channel], h, histMetric)
                    
            segment_dist.append(dist)    
            
        sub_distances.append(segment_dist)

    print('subdistances len', len(sub_distances))
    
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