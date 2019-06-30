import numpy as np

from src.Segment import Segment
from src.Video import Video
from .VideoReader import VideoReader
from .histograms import compute_histograms
import os
import pickle
import itertools
import random

DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
SEGMENTS_PATH = os.path.join(DATA_PATH, "segments")
MOVIE_PATH = os.path.join(DATA_PATH, "movies")
PICKLE_PATH = os.path.join(DATA_PATH, "pickle")


def load_training_set(video_set, grid_size, bins, skip_val, force_refresh=False):
    """
    Load and process all videos in provided training set.

    video_set: List of integers corresponding to video files (5 char left zero padded).
    """

    print('Loading / processing dataset...', flush=True)

    videos = []

    for i in video_set:
        # Int to movie name
        name = "{:05d}".format(i)
        print('\rprocessing {}'.format(name), end='', flush=True)

        # Process
        video = process_video(name, grid_size, bins, skip_val, force_refresh)
        if video is not None: videos.append(video)

    print('\rDone processing!', end='', flush=True)

    return videos


def process_video(name: str, grid_size : int, bins: [], skip_val, force_refresh=False) -> Video:
    
    pickle_dir = os.path.join(PICKLE_PATH, str(grid_size), '_'.join(str(b) for b in bins), str(skip_val))
    
#     Create folder if it doenst exist
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
        
    pickle_path = os.path.join(pickle_dir, name + ".pickle")

#     # If processed pickle exists, load that
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
    for i in range(0, s.num_frames(), skip_val):
        if i % skip_val == 0:
            frame_histograms = compute_histograms(next(video_frames), grid_size=grid_size, bins=bins)
            s.histograms.append(frame_histograms)
        next(video_frames)
        
        # Only convert the first frame for now
#         break;
    
    return s

def get_test_video(name: str, grid_size : int, bins: []):
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
    print('start_frame', start_frame, end_frame, nr_frames_to_get)

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

    return histograms