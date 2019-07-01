import numpy as np

from src.Segment import Segment
from src.Video import Video
from .VideoReader import VideoReader
from .histograms import compute_histograms
import os
import pickle
import itertools
import random
from joblib import Parallel, delayed

DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
SEGMENTS_PATH = os.path.join(DATA_PATH, "segments")
MOVIE_PATH = os.path.join(DATA_PATH, "movies")
PICKLE_PATH = os.path.join(DATA_PATH, "pickle")


def load_training_set(video_set, grid_size, bins, skip_val, force_refresh=False):
    """
    Load and process all videos in provided training set.

    video_set: List of integers corresponding to video files (5 char left zero padded).
    """

    if force_refresh:
    
        # Process in parallel if we need to refresh (much faster)
        videos = Parallel(n_jobs=-1, prefer="threads")(
            delayed(process_video)(i, grid_size, bins, skip_val, force_refresh) for i in video_set)
        
        # Remove None values
        list(filter(None.__ne__, videos))
        
    else:
        
        videos = []
        for i in video_set:
            video = process_video(i, grid_size, bins, skip_val, force_refresh)
            if video is not None: videos.append(video)

    return videos


def process_video(i: int, grid_size : int, bins: [], skip_val, force_refresh=False) -> Video:
    
    name = "{:05d}".format(i)
    
    pickle_dir = os.path.join(PICKLE_PATH, str(int(grid_size)), '_'.join(str(int(b)) for b in bins), str(skip_val))
    
    # Create folder if it doenst exist
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
        force_refresh= True
        
    pickle_path = os.path.join(pickle_dir, name + ".pickle")

    # If processed pickle exists, load that
    if not force_refresh and os.path.isfile(pickle_path):

        # Load pickle
        with open(pickle_path, 'rb') as f:
            video = pickle.load(f)

    # Else process video again and store pickled
    else:
        video_path = os.path.join(MOVIE_PATH, name + ".mp4")
        segments_path = os.path.join(SEGMENTS_PATH, name + ".tsv")

        # Check if file exists
        if not os.path.isfile(video_path):
            print("Cannot open video %s.mp4 -- File does not exist." % name)
            return

        # Load movie in memory
        source_video = VideoReader()
        source_video.open(video_path)
        
        # Load segments file
        segment_data = np.genfromtxt(segments_path, delimiter="\t", skip_header=1, filling_values=1)

        # Create video object and convert segments
        video = Video(name + ".mp4", source_video.get_number_of_frames(), source_video.get_frame_rate())
        frame_iter = source_video.get_frames()
        video.segments = np.apply_along_axis(lambda row: create_segment(
                                             name + ".mp4", frame_iter, row, grid_size, bins, skip_val),
                                             arr=segment_data, axis=1)

        # Dump to pickle
        with open(pickle_path, 'wb+') as f:
            pickle.dump(video, f)

    return video


def create_segment(movie_id: str, video_frames, row: np.ndarray, grid_size : int, bins : [], skip_val: int) -> Segment:
    """"
    Row layout: [startframe, starttime, endframe, endtime]
    """
    

    # Create new segment
    s = Segment(movie_id, row[1], row[3], row[0], row[2])
    
    s.histograms = []
        
    # Generate histograms for frames in segment
    
    
    # Generate histograms for frames in segment
    for i in range(0, s.num_frames()):
        if skip_val > 0 and i % skip_val == 0:
            frame_histograms = compute_histograms(next(video_frames), grid_size=grid_size, bins=bins)
            s.histograms.append(frame_histograms)
        elif skip_val > 0:
            next(video_frames)
        elif skip_val < 0 and i % abs(skip_val) == 0:
            # make pos again
            avg_val = abs(skip_val)
        
            # Sometimes error here as it seems next(video_frames) is already end of list, why....
            avg_hist = compute_histograms(next(video_frames), grid_size=grid_size, bins=bins)
            hist_count = 1

            # Try to sum the next hists, however might run into end of segment
            try:
                for _ in range(1, avg_val-1):
                    tmp_hist = compute_histograms(next(video_frames), grid_size=grid_size, bins=bins)
                    np.add(avg_hist, tmp_hist)
                    hist_count += 1
            except StopIteration:
                pass

            # Average hists
            avg_hist = np.divide(avg_hist, hist_count)
            s.histograms.append(avg_hist)
        
    return s



def get_test_video(name: str, grid_size : int, bins: []):
    """
    Get a random 20 second sample from video: name. 
    Compute the histograms for this sample with grid_size and bins.
    """
    video_path = os.path.join(MOVIE_PATH, name + ".mp4")
    segments_path = os.path.join(SEGMENTS_PATH, name + ".tsv")

    # Check if file exists
    if not os.path.isfile(video_path):
        print("Cannot open video %s.mp4 -- File does not exist." % name)
        return

    # Load movie in memory
    source_video = VideoReader()
    source_video.open(video_path)

    nr_frames_to_get = int(source_video.get_frame_rate() * 20)
    start_frame = random.choice(range(0, (source_video.get_number_of_frames() - nr_frames_to_get)))
    end_frame = start_frame+nr_frames_to_get

    i = 0
    frames = source_video.get_frames()
    histograms = []
    for f in frames:
        if i < start_frame:
            i += 1
            continue
        
        if i == end_frame:
            break
        
        histograms.append(compute_histograms(f, grid_size=grid_size, bins=bins))
        i += 1

    return histograms, int(start_frame), int(end_frame)

def get_test_video_set(max_vid, grid_size, bins, n=100, duration=20):
    """
    Get or create test n videos. Try to load from pickle to save time, or compute new ones.
    """
    test_set = []
    labels = []
    
    pickle_test_dir = os.path.join(PICKLE_PATH, 'test', str(int(grid_size)), '_'.join(str(int(b)) for b in bins))
    # Create folder if it doenst exist
    if not os.path.exists(pickle_test_dir):
        os.makedirs(pickle_test_dir)
    
    #Try to load existing test pickles
    filenames = os.listdir(pickle_test_dir)
    while len(test_set) < n and len(test_set) < len(filenames):
        # pick random file
        filename = random.choice(filenames)
        file_split = filename.split('.')[0].split('_')
        
        if int(file_split[0]) > max_vid:
            continue

        pickle_file = os.path.join(pickle_test_dir, filename)
        if os.path.isfile(pickle_file): 
            # Load pickle

            with open(pickle_file, 'rb') as f:
                test_set.append(pickle.load(f))
                  
            labels.append(('{}.mp4'.format(file_split[0]), int(file_split[1]), int(file_split[2])))
     
    
    # create remaining ones      
    for i in range(n-len(test_set)):
        print('\rprocessing {}/{}'.format(i+1, n), end='', flush=True)
       
        
        vid = random.choice(range(1, max_vid+1))
        vid_name = "{:05d}".format(vid)
        hists, start_frame, end_frame = get_test_video(vid_name, grid_size, bins)
        
        test_set.append(hists)
        labels.append(('{}.mp4'.format(vid_name), start_frame, end_frame))
    
        # Save for future usage
        pickle_path = os.path.join(pickle_test_dir, '_'.join([vid_name, str(start_frame), str(end_frame)]) + ".pickle")
        with open(pickle_path, 'wb+') as f:
            pickle.dump(hists, f)


    return test_set, labels
    
    