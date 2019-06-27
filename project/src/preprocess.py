import numpy as np

from src.Segment import Segment
from src.Video import Video
from .VideoReader import VideoReader
from .histograms import compute_histograms
import os
import pickle

DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
SEGMENTS_PATH = os.path.join(DATA_PATH, "segments")
MOVIE_PATH = os.path.join(DATA_PATH, "movies")
PICKLE_PATH = os.path.join(DATA_PATH, "pickle")


def load_training_set(video_set, force_renew=False):
    """
    Load and process all videos in provided training set.
    
    video_set: List of integers corresponding to video files (5 char left zero padded).
    """

    for i in video_set:
        
        # Int to movie name
        name = "{:05d}".format(i)
        print('processing {}'.format(name), end='\r', flush=True)
        
        # Process
        yield from process_video(name, force_renew)


def process_video(name: str, force_renew=False) -> Video:
    pickle_path = os.path.join(PICKLE_PATH, name + ".pickle")
    
    # If processed pickle exists, load that
    if not force_renew and os.path.isfile(pickle_path):
    
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
        video = Video(name + ".mp4")
        frame_iter = source_video.get_frames()
        video.segments = np.apply_along_axis(lambda row: create_segment(name + ".mp4", frame_iter, row), arr=segment_data, axis=1)
        
        # Dump to pickle
        with open(pickle_path, 'wb+') as f:
            pickle.dump(video, f)
        
    yield video


def create_segment(movie_id: str, video_frames, row: np.ndarray) -> Segment:
    """"
    Row layout: [startframe, starttime, endframe, endtime]
    """

    # Create new segment
    s = Segment(movie_id, row[1], row[3], row[0], row[2])
    
    # Accumulate frames in segment
    framebuffer = [next(video_frames) for _ in range(s.num_frames()+1)]

        
    # Generate histograms
    s.histograms = generate_histograms(np.asarray(framebuffer))

    return s


def generate_histograms(framebuffer: np.ndarray) -> np.ndarray:
#     print(framebuffer.shape)
#     print(framebuffer)
    
    histograms = []
    
    for frame in framebuffer:
        
        if len(histograms) == 0:
            histograms.append(compute_histograms(frame))
        else:
            pass
            # Change detection
    
    return np.asarray(histograms)