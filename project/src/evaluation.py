from .Video import Video
from .Segment import Segment
import random
import numpy as np

def pick_test_segments(training_set: [Video], n_segments=100):
    """
    Pick n random segments from the training set.
    Returned labels are the actual segments.
    """
    
    test_set = []
    labels = []
    
    for i in range(n_segments):
    
        # Find random video
        video = random.choice(training_set)

        # Select random segment and add histogram to test set
        segment = random.choice(video.segments)
        test_set.append(segment.histograms)
        labels.append(segment)
        
    return test_set, labels


def generate_test_segments(training_set: [Video], n=100, duration=20):
    """
    Generate n custom segments of duration d from the training set.
    Returned labels are 3-tuples of video ID, start frame and end frame.
    """
    
    test_set = []
    labels = []
    
    i = 0
    while i < n:
    
        # Find random video
        video = random.choice(training_set)
        
        # Skip videos that are not long enough
        if video.duration() < duration:
            continue
        
        # Calculate required number of frames in test segment
        frames_in_selection = int(duration * video.frame_rate)
        
        # Calculate start / end frames of segment
        max_start_frame = video.num_frames - frames_in_selection - 1
        selection_start_frame = int(max_start_frame * random.random())
        selection_end_frame = selection_start_frame + frames_in_selection
        
        # Get histograms
        histograms = []
        
        for segment in video.segments:
            
            if segment.frame_end < selection_start_frame or segment.frame_start > selection_end_frame:
                # Pass if fragment outside bounds
                pass
            
            elif segment.frame_start >= selection_start_frame and segment.frame_end <= selection_end_frame:
                
                # Add the entire segment to the selection
                histograms += segment.histograms
                
            else:
                
                # Add only part of this segment to the selection
                if segment.frame_start >= selection_start_frame:
                    slice_start = None
                    slice_end = selection_end_frame - segment.frame_start  # The last part of the selection
                    
                elif segment.frame_end <= selection_end_frame:
                    slice_start = selection_start_frame - segment.frame_start # The first part of the selection
                    slice_end = None
                
                else:
                    slice_start = selection_start_frame - segment.frame_start # Entirely inside this segment
                    slice_end = selection_end_frame - segment.frame_start
                
                histograms += segment.histograms[slice_start:slice_end]
        
        # TODO Sometimes selected range seems to have no histogram? 
        if len(histograms) == 0:
            continue
        
        # Add histogram list to test set
        test_set.append(histograms)
        
        # Add labels
        labels.append( (video.name, selection_start_frame, selection_end_frame) )
        
        i += 1
    return test_set, labels


def evaluate_segments(predicted: [Segment], labels: [Segment]):
    
    movie_correct = 0
    movie_wrong = 0
    start_frame_dist = 0

    for pred, label in zip(predicted, labels):

        # Check if movie is correct
        if pred[0] == label[0]:
            movie_correct += 1
            start_frame_dist += abs(pred[1]-label[1])
        else: movie_wrong += 1
            
        

    total = movie_correct + movie_wrong
    fraction = movie_correct / total if total > 0 else 0
    

    print("Segment evaluation:")
    print("Correct movies: {:d}".format(movie_correct))
    print("Wrong movies:   {:d}".format(movie_wrong))
    print("Total:          {:d}".format(total))
    print("TPR:     {:.1f}%".format(fraction * 100))
    print("\nStart frame distance (correct movies only):   {:d}".format(start_frame_dist))
    print("Avg Start frame distance (correct movies only): {:.2f}".format(start_frame_dist/total), flush=True)


def evaluate(predicted, labels):
    """
    Calculate error metrics of predicted labels.
    Input: 3-tuple (String, int, int) with (video_id, start_frame, end_frame)
    """
    
    assert len(predicted) == len(labels), "Different number of predictions and labels."
    
    total = len(predicted)
    movie_correct = 0
    location_correct = 0
    
    center_frame_dist = []    
    overlaps = []
    
    for pred, label in zip(predicted, labels):
        
        dist = 0
        
        if pred[0] == label[0]: # Check if movie is correct
            movie_correct += 1
            
            dist = abs(pred[1] - ((label[1]+label[2])/2))    
            center_frame_dist.append(dist)
            
            correct = False
            if label[1] <= pred[1] <= label[2]:
                correct = True
                location_correct += 1

            
#         print("Label: ({:s}, {:d}, {:d}), predicted: ({:s}, {:d}), location correct: {!s:}, start_frame_dist: {:d}, overlap: {:d}".format(
#             *label,
#             *pred,
#             correct,
#             dist
#         ))
        
    # Return (# movies correct, # correct location, # total movies) and (avg start frame distance, std)
    return (movie_correct, location_correct, total), (np.mean(center_frame_dist), np.std(center_frame_dist))