from .Video import Video
from .Segment import Segment
import random

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
    
    for i in range(n):
    
        # Find random video
        video = random.choice(training_set)
        
        # Skip videos that are not long enough
        if video.duration() < duration:
            i -= 1
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
        
        # Add histogram list to test set
        test_set.append(histograms)
        
        # Add labels
        labels.append( (video.name, selection_start_frame, selection_end_frame) )
        
    return test_set, labels


def evaluate_segments(predicted: [Segment], labels: [Segment]):
    
    movie_correct = 0
    movie_wrong = 0

    for segment, label in zip(predicted, labels):

        # Check if movie is correct
        if segment == label: movie_correct += 1
        else: movie_wrong += 1

    total = movie_correct + movie_wrong
    fraction = movie_correct / total if total > 0 else 0

    print("Segment evaluation:")
    print("Correct: {:d}".format(movie_correct))
    print("Wrong:   {:d}".format(movie_wrong))
    print("Total:   {:d}".format(total))
    print("TPR:     {:.1f}%".format(fraction * 100), flush=True)